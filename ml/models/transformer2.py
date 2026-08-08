"""
Transformer 2 — global same-type attention over Supplier embeddings (Claim 3,
`docs/10_AI_ML_Documentation.md` §8.3; `05_Database_Design.md` §6.25). Built on top of
the finalized Layer 1 + Layer 2 stack: SHARE (RGCN+Attention, `rgcn_attn`) as the
encoder, Variant A (bounded residual on the Markov floor, `reports/rung5_types.md`
Type A) as the depth-selection mechanism. A new, purely ADDITIVE component — this
file changes nothing about SHARE's encoder, SHARP, SHARK, or Variant A's own gate
(`ml/models/rgcn_attn_rung5_variant_a.py`, imported and reused unmodified).

**The problem this exists for.** Two suppliers can fail together through a shared,
unrecorded upstream source. No edge chain connects them, so message passing at ANY
depth is structurally blind to it — the co-parent path (`Supplier -> Component ->
rev_Component -> Supplier`) requires an actual shared-component-supplier row to exist
(`component_suppliers`); a hidden dependency, by construction, has none. The synthetic
dataset plants exactly this scenario (`db/README.md`, "the hidden-dependency
scenario"; ground truth reconstructed in `ml/graph/hidden_dependency_ground_truth.py`)
specifically so this claim has something real to be tested against.

**Mechanism — dense self-attention over Supplier embeddings, no adjacency mask.**
Operates on SHARE's own Supplier-type node embeddings, POST-encoder, at whatever
depth the impact head currently reads under Variant A (`z_impact_i`, Variant A's own
per-node bounded-residual blend — not a raw fixed `h^4` read, since Variant A's blend
is itself the thing "the impact head currently reads" under the finalized stack). The
same-type constraint (Suppliers attending only to other Suppliers) is mandatory, not a
tuning choice — attending across node types would partially undo the type separation
the structural encoder spends real parameters building. Deliberately NO adjacency
mask: this is the entire point — Transformer 2 exists specifically to connect pairs
with NO graph path between them (the "moralization" case,
"Markov Scoping and Transformer 1.md" §1.3/§2.4), so restricting the candidate pool to
graph-adjacent pairs would defeat the mechanism outright.

**Bounded candidate pool.** Top-`k=64` by embedding cosine similarity per Supplier
(self excluded), matching `docs/10_AI_ML_Documentation.md` §8.3's own bounding choice
— NOT bounded by shared `component_type`, which that section explicitly flags as
wrong (same-`component_type` suppliers are already 2-hop reachable via the co-parent
path, so that bounding would exclude exactly the cross-type correlation the
hidden-dependency scenario and Transformer 2 both exist to find; H_POLYMER's own 4
members are deliberately NOT polymer-exclusive for this reason).

**`is_frontier` override — implemented, but INERT this round, disclosed explicitly.**
The design calls for always including frontier-flagged suppliers (no recorded
upstream) in the candidate pool regardless of cosine rank. `docs/05_Database_Design.md`
§6.23 documents that `suppliers.is_frontier` is "data-gated" and does NOT exist in the
live schema (`SUB_SUPPLIES`/`supplier_relationships` remain unbuilt) — confirmed live
before writing this file (`\\d suppliers` has no `is_frontier` column). The override
path below (`is_frontier` parameter to `forward()`) is implemented and will function
correctly once that data exists — frontier candidates displace the LOWEST-scoring
cosine candidates in each row, keeping a fixed pool width `k` rather than growing it —
but is never exercised in this round's actual training (`is_frontier=None` throughout,
`ml/run_transformer2_pilot.py`), since there is nothing to flag. This is a disclosed
limitation carried over from the schema's own documented gap, not a silently-dropped
requirement.

**Fusion into the impact head.** `z_impact_fused = z_impact + t2_scale * t2_output`,
where `t2_scale` is a single learned scalar initialized to exactly 0 — matching this
project's own established convention throughout Step 6 (Markov/Variant A/Variant C:
every new component starts from a state IDENTICAL to the validated baseline it
extends, so training can only move away from that safe start deliberately). At
`t2_scale=0`, this component's forward pass is, to numerical precision, identical to
plain Variant A's impact readout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Transformer2GlobalAttention(nn.Module):
    """Same-type-only global attention over a set of node embeddings, bounded to a
    top-`k` cosine-similarity candidate pool per node (self excluded), with an
    optional frontier-node override. No adjacency mask — the candidate pool is drawn
    from cosine similarity over the FULL node population, not graph neighbors."""

    def __init__(self, hidden: int, top_k: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.top_k = top_k
        self.hidden = hidden

    def forward(self, z: torch.Tensor, is_frontier: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`z`: `[N, hidden]`, one embedding per Supplier. `is_frontier`: optional
        `[N]` bool tensor — if given, frontier-flagged suppliers displace each row's
        lowest-cosine candidates so they're always in the pool (see module docstring
        for why this is inert, not unimplemented, this round). Returns
        `(t2_output [N, hidden], alpha [N, k], candidate_idx [N, k])` — `alpha` are
        the raw per-pair attention weights persisted to `hidden_dependency_links`."""
        n = z.size(0)
        k = min(self.top_k, n - 1)

        normed = F.normalize(z, dim=-1)
        cos_sim = (normed @ normed.T).clone()
        cos_sim.fill_diagonal_(float("-inf"))  # exclude self from the candidate pool
        topk_val, topk_idx = cos_sim.topk(k, dim=-1)  # [N, k]

        if is_frontier is not None and is_frontier.any():
            topk_idx = _splice_in_frontier(topk_idx, topk_val, is_frontier)

        q = self.q_proj(z)
        key = self.k_proj(z)
        value = self.v_proj(z)
        key_pool = key[topk_idx]      # [N, k, hidden]
        value_pool = value[topk_idx]  # [N, k, hidden]

        scores = (q.unsqueeze(1) * key_pool).sum(-1) / (self.hidden ** 0.5)  # [N, k]
        alpha = torch.softmax(scores, dim=-1)
        t2_output = (alpha.unsqueeze(-1) * value_pool).sum(1)  # [N, hidden]
        return t2_output, alpha, topk_idx


def _splice_in_frontier(topk_idx: torch.Tensor, topk_val: torch.Tensor,
                         is_frontier: torch.Tensor) -> torch.Tensor:
    """For each row, replace the lowest-scoring candidates not already present with
    any frontier-flagged indices missing from that row's pool, keeping pool width
    fixed. Row-wise Python loop — never exercised in this round (see module
    docstring), so simplicity is prioritized over vectorization here."""
    n, k = topk_idx.shape
    frontier_ids = is_frontier.nonzero(as_tuple=True)[0]
    out = topk_idx.clone()
    for i in range(n):
        row = out[i]
        present = set(row.tolist())
        missing = [f.item() for f in frontier_ids if f.item() != i and f.item() not in present]
        if not missing:
            continue
        # ascending order of that row's own topk_val -> lowest-scoring slots first
        order = torch.argsort(topk_val[i])
        for slot, new_id in zip(order.tolist(), missing):
            row[slot] = new_id
    return out
