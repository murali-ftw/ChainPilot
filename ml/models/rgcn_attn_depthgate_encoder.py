"""
Step 6 — learned, per-task depth gate on top of SHARE (`ml/models/rgcn_attn_encoder.py`,
RGCN+Attention, `rgcn_attn` in code). Motivated by `reports/step6_preflight.md`'s 5-seed-
confirmed finding that delay wants a shallow (~L1) read while impact wants a deep (~L4)
one, from the *same* shared encoder trunk -- i.e. no single fixed depth serves every task,
which is exactly the condition a per-task learned gate is for.

**What this is NOT.** This is not a new encoder architecture in the sense `rgcn_relemb`/
`rgcn_battn` are -- the message-passing math (`RGCNAttnEncoder`'s basis-decomposed
transform + shared attention scorer) is completely unchanged. The only new thing is *which
layer's output each task head reads*, and that choice is now soft and learned instead of a
hard-coded index (`ml/models/depth.py::readout_layer_for_task`).

**Mechanism.** `RGCNAttnEncoder.forward()` returns `h^1..h^L` (`L` layers); it computes `h^0`
(the raw per-node-type `lin_in` projection, before any message passing) internally but never
returns it. `RGCNAttnDepthGateEncoder` below prepends `h^0`, so callers get `L+1` candidate
depths. Each task then reads a WEIGHTED SUM across all `L+1` depths, not one hard index:

    h_task = sum_{l=0}^{L} softmax(w_task)_l * h_l

`w_task` is a small learned vector, length `L+1`, one per task (delay/shortage/impact don't
share one) -- `3 * (L+1) = 15` new parameters at `L=4`, negligible next to the ~750K-param
encoder. This is a genuinely simple, "does ANY task-specific depth read help at all" probe,
not a per-node gate (`project_HADES.md` §4.4's eventual per-node MLP gate is a separate,
larger design point this experiment deliberately doesn't build yet).

**Anchoring.** An unconstrained softmax(w_task) could drift to depths no measurement ever
supported. Each task's learned distribution is KL-penalized against a FIXED (non-learned)
prior over the same `L+1` depths, reusing `project_HADES.md` §4.3's own prior-shape formula
(`p[k] = exp(-|k-peak|/tau) / sum`) rather than inventing new prior math -- same direction,
`KL(learned || prior)`, as §4.4's own (not-yet-built) per-node gate design already specifies.
Peaks and widths are read directly off measured results, not chosen freely:

    delay:    peak=1,            tau=0.8  (tight)  -- step6_preflight.md: L1 beats baseline,
                                                        CONSISTENT across all 5 seeds; L2 shows
                                                        nothing.
    impact:   peak=num_layers,   tau=0.8  (tight)  -- rgcn_types.md's AUC-vs-depth sweep for
                                                        SHARE peaks at L4.
    shortage: peak=3,            tau=3.0  (wide)   -- every sweep to date found shortage
                                                        depth-indifferent; peak kept at its
                                                        historical structural-prior depth (h^3)
                                                        but with a much wider tau, i.e. close to
                                                        uniform rather than genuinely flat, per
                                                        this experiment's own framing
                                                        ("closer to uniform/matched-baseline").

Total loss (built in `ml/train.py::_epoch_forward`, not here):
`sum_task FocalLoss_task + lambda * sum_task KL(softmax(w_task) || prior_task)`.

**Why a self-contained model class, not a `HADESModel` edit.** `HADESModel`/`depth.py`'s
`readout_layer_for_task` indexing is the contract every other architecture in this project
relies on; branching a per-node-type weighted-sum-with-KL mechanism into that shared path
would risk every other architecture's behavior for a Step-6-only experiment. `DepthGateHADESModel`
below is fully self-contained (encoder + 3 gates + 3 `PredictionHead`s) and returns the exact
same `(logits, layers)` 2-tuple `HADESModel.forward()` does, so `ml/evaluate.py` needs zero
changes -- the KL term and the learned gate weights (the actual point of this experiment) are
stashed as instance attributes (`_last_kl_total`, `_last_gate_weights`) read back by a small,
backward-compatible addition to `train.py::_epoch_forward` (a no-op for every architecture that
doesn't set those attributes).
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_encoder import RGCNAttnEncoder

_KL_EPS = 1e-8


class RGCNAttnDepthGateEncoder(RGCNAttnEncoder):
    """Identical to `RGCNAttnEncoder` except `forward()` prepends `h^0` (the
    raw `lin_in` projection, before any message passing) to the returned
    layer list -- `L+1` per-node-type dicts (`h^0..h^L`) instead of `L`
    (`h^1..h^L`). The depth gate needs the shallowest possible read (zero
    hops) as a real, addressable option, not just something starting at one
    hop, since delay's own confirmed preference (`reports/step6_preflight.md`)
    is for the shallowest depth tested."""

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h0 = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        rest = super().forward(x_dict, edge_index_dict)
        return [h0] + rest


def _peaked_prior(num_depths: int, peak: float, tau: float) -> torch.Tensor:
    """Same exp-decay-from-peak formula as `project_HADES.md` §4.3's Markov
    depth prior, `p[k] = exp(-|k-peak|/tau) / sum` -- reused verbatim here as
    the depth gate's anchor prior, not invented fresh. Depths are 0..num_depths-1
    (`num_depths = num_layers + 1`, since h^0 is now a real candidate)."""
    depths = torch.arange(num_depths, dtype=torch.float32)
    logits = -(depths - peak).abs() / tau
    return torch.softmax(logits, dim=0)


def depth_gate_priors(num_layers: int) -> dict[str, torch.Tensor]:
    """{task: prior distribution over depths 0..num_layers}, per-task peak/tau
    read directly off measured results -- see module docstring for the exact
    citations behind each choice."""
    n = num_layers + 1
    return {
        "delay": _peaked_prior(n, peak=1.0, tau=0.8),
        "impact": _peaked_prior(n, peak=float(num_layers), tau=0.8),
        "shortage": _peaked_prior(n, peak=3.0, tau=3.0),
    }


class DepthGateHead(nn.Module):
    """One task's learned depth gate: a small weight vector `w` (length
    `num_depths`), softmax'd into a distribution over which depth(s) to read,
    weighted-summed against that task's entity embeddings at every depth.
    Anchored by a KL penalty against a FIXED (registered buffer, not
    learned) prior -- see `depth_gate_priors()` above."""

    def __init__(self, num_depths: int, prior: torch.Tensor):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_depths))
        self.register_buffer("prior", prior)

    def forward(self, embeddings_by_depth: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """`embeddings_by_depth`: list of length `num_depths`, each `[N, hidden]`
        for ONE entity type, depth-ordered (index 0 = h^0). Returns
        `(h_task [N, hidden], kl scalar)`."""
        alpha = torch.softmax(self.w, dim=0)
        stacked = torch.stack(embeddings_by_depth, dim=0)  # [num_depths, N, hidden]
        h_task = torch.einsum("d,dnh->nh", alpha, stacked)
        kl = (alpha * (alpha.clamp_min(_KL_EPS).log() - self.prior.clamp_min(_KL_EPS).log())).sum()
        return h_task, kl

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.w, dim=0).detach()


class DepthGateHADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder`) + one
    `DepthGateHead` per task + one `PredictionHead` per task, composed.
    Returns the same `(logits, layers)` 2-tuple `HADESModel.forward()` does
    -- `ml/evaluate.py` needs zero changes -- with the KL term and learned
    gate weights stashed as instance attributes for the training loop /
    reporting code to read back (see module docstring)."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 lambda_kl: float = 0.1):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_depthgate", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        priors = depth_gate_priors(num_layers)
        self.gates = nn.ModuleDict({task: DepthGateHead(num_layers + 1, priors[task]) for task in TASKS})
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.lambda_kl = lambda_kl
        self.architecture = "rgcn_attn_depthgate"

        # Side-channel outputs of the most recent forward() call -- read by
        # ml/train.py::_epoch_forward (KL term) and by reporting code (gate
        # weights), never by ml/evaluate.py, which only ever unpacks the
        # (logits, layers) return value.
        self._last_kl_total: torch.Tensor | None = None
        self._last_gate_weights: dict[str, torch.Tensor] | None = None

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)
        logits = {}
        kl_total = None
        gate_weights = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            embeddings_by_depth = [layer[entity_type] for layer in layers]
            h_task, kl = self.gates[task](embeddings_by_depth)
            logits[task] = self.heads[task](h_task)
            kl_total = kl if kl_total is None else kl_total + kl
            gate_weights[task] = self.gates[task].weights()
        self._last_kl_total = kl_total
        self._last_gate_weights = gate_weights
        return logits, layers

    def gate_weight_summary(self) -> dict[str, list[float]]:
        """Softmax'd gate weights per task, depth-ordered (index 0 = h^0) --
        the actual point of this experiment, made directly inspectable."""
        with torch.no_grad():
            return {task: self.gates[task].weights().tolist() for task in TASKS}

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
