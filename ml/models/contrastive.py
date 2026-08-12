"""
Stage 1 of the Layer 3 retrieval redesign — an **auxiliary InfoNCE contrastive loss** on
the Supplier embeddings that Transformer 2's retrieval actually compares.

**Why this exists.** `reports/layer3_testing.md` §2 found Transformer 2's retrieval at
chance on every group type, at both scales, in all four fusion arms, and diagnosed the
cause in §7: the embeddings being compared (`z_impact`, Variant A's own bounded-residual
blend, which `ml/models/rgcn_attn_variant_a_transformer2.py` feeds to
`Transformer2GlobalAttention`) are shaped entirely by the delay/shortage/impact focal
losses. Nothing in that objective rewards preserving co-membership, so there is no reason
the cosine geometry should encode it. This module points a loss directly at that geometry
and asks whether the representation *can* encode the signal when explicitly taught to.

**DISCLOSED DEVIATION — privileged ground truth shapes gradients here.** Every prior
privileged read in this project (Mechanism E's `resilience` for the Layer 2 gate
diagnostic, `HP_GROUPS` for `reports/layer3_testing.md` §2's discovery check) was
**eval-only**: read to score a model, never to fit one. This is the first component in
either project where the generator's internal `HP_GROUPS` is used as a **training
target**. It is a legitimate experiment — it establishes a ceiling — but it is a
different and strictly easier claim than emergent discovery, and every number it
produces must be reported as such in the same breath. It is still never a model
*input*: the group labels enter only through this loss term, and the forward pass is
byte-for-byte the plain Transformer 2 arm's.

**The objective.** Standard InfoNCE with one positive per term and shared negatives:

    L = mean over (anchor i, positive j) of  -log( exp(s_ij/T) /
                                                   (exp(s_ij/T) + sum_{k not in group(i)} exp(s_ik/T)) )

where `s` is cosine similarity between L2-normalised Supplier embeddings — the *same*
quantity `Transformer2GlobalAttention` ranks its top-k candidate pool by, so the loss
acts on the retrieval geometry itself rather than on some parallel projection head that
retrieval would never consult.

* **Anchors and positives: Type A and Type B members only.** Both are real,
  structurally-generated groups (Mechanism B draws all three types by one identical
  process; only the downstream treatment differs — see `db/generate_dataset.py` Phase 3).
* **Negatives: everything outside the anchor's own group**, which is exactly "Type C
  plus random non-member pairs" — Type C members, unrelated A/B groups' members, and
  the ~80% of suppliers in no group at all.

**Zero contribution at construction**, per this project's standing convention that every
new component starts identical to the baseline it extends (`t2_scale` init 0, Variant A's
zero-initialised residual MLP, the Markov prior's fixed bias). A loss term has no
parameter to zero, so the equivalent is a **warmup ramp**: the weight is exactly 0.0 on
epoch 1 and rises linearly to its target over `warmup_epochs`. Epoch 1 is therefore
numerically identical to the plain T2 arm's epoch 1, and the objective only moves away
from that validated start deliberately. Same idiom as `train.py`'s existing
`anneal_lambda` schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

DEFAULT_TEMPERATURE = 0.1
DEFAULT_WARMUP_EPOCHS = 3


@dataclass
class GroupSupervision:
    """Privileged co-membership targets, resolved onto Supplier node indices.

    `group_id[i]` is the hidden-parent group index of Supplier row `i`, or -1 for the
    suppliers in no group. `anchor_mask[i]` is True only for members of **supervised**
    Type A/B groups — Type C never anchors (it is the decoy the retriever must keep
    ignoring) and held-out groups never anchor (see `held_out_mask`).

    `held_out_mask[i]` flags members of A/B groups deliberately **excluded from the
    loss**. Scoring retrieval separately on supervised and held-out groups is what
    separates "the representation memorised 90 specific groups" from "the representation
    learned a notion of co-membership that transfers" — two very different findings that
    a single pooled number would collapse into one.
    """

    group_id: torch.Tensor       # [N] long
    anchor_mask: torch.Tensor    # [N] bool
    held_out_mask: torch.Tensor  # [N] bool
    n_supervised_groups: int
    n_held_out_groups: int

    def to(self, device) -> "GroupSupervision":
        return GroupSupervision(
            self.group_id.to(device), self.anchor_mask.to(device),
            self.held_out_mask.to(device), self.n_supervised_groups, self.n_held_out_groups)


def build_supervision(groups: list[dict], sup_ids: list[str], held_out_frac: float = 0.3,
                      seed: int = 0) -> GroupSupervision:
    """Resolve `HP_GROUPS` onto this snapshot's Supplier row order.

    Supplier node order is invariant across snapshots (`ml/data/loader.py` builds
    `_id_maps_static["Supplier"]` once from `suppliers.csv`), so one supervision object
    serves every bundle of a variant-seed; the caller asserts that invariance rather
    than trusting it.
    """
    idx = {s: i for i, s in enumerate(sup_ids)}
    n = len(sup_ids)
    group_id = torch.full((n,), -1, dtype=torch.long)
    anchor = torch.zeros(n, dtype=torch.bool)
    held = torch.zeros(n, dtype=torch.bool)

    ab = [g for g in groups if g["type"] in ("A", "B")]
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ab), generator=rng).tolist()
    n_held = int(round(len(ab) * held_out_frac))
    held_set = {order[i] for i in range(n_held)}

    for gi, g in enumerate(groups):
        rows = [idx[m] for m in g["members"] if m in idx]
        for r in rows:
            group_id[r] = gi
    for j, g in enumerate(ab):
        rows = [idx[m] for m in g["members"] if m in idx]
        for r in rows:
            if j in held_set:
                held[r] = True
            else:
                anchor[r] = True
    return GroupSupervision(group_id, anchor, held, len(ab) - n_held, n_held)


def infonce_loss(z: torch.Tensor, sup: GroupSupervision,
                 temperature: float = DEFAULT_TEMPERATURE) -> torch.Tensor:
    """InfoNCE over co-membership, computed on the L2-normalised embedding geometry.

    `z`: `[N, hidden]`, the live (non-detached) Supplier embedding the impact gate
    produced this forward pass. Returns a scalar; zero when no anchor has a co-member,
    which keeps the term well-defined on a variant with Mechanism B off.

    Held-out groups are excluded from the numerator (they never anchor and are never
    positives) but are **not** excluded from the denominator — masking them out of the
    negatives would leak which suppliers are held-out group members into the objective,
    which is the same class of mistake as feeding the labels in directly.
    """
    if not bool(sup.anchor_mask.any()):
        return z.sum() * 0.0

    zn = F.normalize(z, dim=-1)
    sim = (zn @ zn.T) / temperature                              # [N, N]
    n = sim.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)

    gid = sup.group_id
    same = (gid.unsqueeze(0) == gid.unsqueeze(1)) & (gid.unsqueeze(1) >= 0)
    # Positives: supervised A/B anchors paired with their own co-members, excluding any
    # co-member that landed in the held-out split (a group is held out whole, so this is
    # a no-op in practice; kept explicit so the invariant is enforced, not assumed).
    pos = same & ~eye & sup.anchor_mask.unsqueeze(1) & sup.anchor_mask.unsqueeze(0)
    if not bool(pos.any()):
        return z.sum() * 0.0

    # Negatives: every supplier outside the anchor's own group. Type C members, other
    # groups' members and the unaffiliated majority all land here, which is exactly the
    # negative set the brief specifies.
    neg_scores = sim.masked_fill(same | eye, float("-inf"))
    neg_lse = torch.logsumexp(neg_scores, dim=-1)                # [N]

    rows, cols = pos.nonzero(as_tuple=True)
    s_pos = sim[rows, cols]
    denom = torch.logaddexp(s_pos, neg_lse[rows])
    return (denom - s_pos).mean()


def warmup_weight(target: float, epoch: int, warmup_epochs: int = DEFAULT_WARMUP_EPOCHS) -> float:
    """Exactly 0.0 on epoch 1, linear to `target` by epoch `warmup_epochs + 1`.

    `epoch` is 1-based, matching `ml/train.py`'s loop. The epoch-1 zero is the point:
    it makes the augmented objective start identical to the baseline objective, so any
    later divergence is a deliberate move rather than a different starting problem.
    """
    if warmup_epochs <= 0:
        return target
    return target * min(1.0, max(0.0, (epoch - 1) / warmup_epochs))
