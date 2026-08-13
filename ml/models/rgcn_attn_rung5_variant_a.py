"""
Step 6 Phase 2, Rung 5 Variant A -- fixed Markov prior + bounded residual gate. ONE
isolated change on top of the ORIGINAL Rung 5 gate (`ml/models/rgcn_attn_rung5_encoder.py`,
`reports/step6_rung_pilot.md`), tested standalone against the same shared baselines as
Variants B/C/D -- never stacked with any of them.

**Motivation.** `reports/step6_rung_pilot.md`'s headline finding was that both per-node
gates showed wildly seed-dependent, bimodal stability: some seeds stayed pinned exactly at
the Markov prior, others let 75-98% of nodes drift to a different depth, with no AUC
difference distinguishing the two regimes -- an UNBOUNDED gate can move arbitrarily far
from a prior that was itself empirically validated (`reports/step6_markov_phase1.md`).
This variant tests the direct fix: make it structurally impossible to drift far, and see
if that alone fixes the instability.

**The one change (everything else identical to the original Rung 5 gate).**

1. `position_bias` becomes a FIXED buffer (`register_buffer`, not `nn.Parameter`) --
   never trained, permanently equal to
   `init_near_onehot_bias(5, MARKOV_READOUT_DEPTH[task])` (`ml/models/rung_gate_common.py`).
   The original's `position_bias` was a *learned* parameter that merely started near
   one-hot; here it is pinned there for the model's entire life.
2. The MLP's raw output is passed through `tanh` and scaled by a small scalar
   `lambda_bound` (default 0.3; 0.1 and 0.5 also run, see the pilot script) before being
   added to the fixed bias:

       logits_i = position_bias + lambda_bound * tanh(MLP(pooled_i))

   Since `tanh` is bounded in `[-1, 1]`, the residual can NEVER exceed
   `±lambda_bound` in any logit component, regardless of what training does to the MLP's
   weights -- a hard, structural ceiling on how far off-prior the gate can ever drift, not
   just a soft encouragement (contrast with `rgcn_attn_depthgate`'s KL penalty, which only
   discourages drift, or the original Rung 5, which had no ceiling at all beyond the
   zero-initialization that training could freely undo).

Everything else -- mean-pooling, the `hidden -> 32 -> 5` MLP shape, the
zero-initialized final MLP layer (so `logits = position_bias` exactly at construction,
identical to the original) -- is UNCHANGED from `ml/models/rgcn_attn_rung5_encoder.py`.

**What this tests in isolation.** Whether the instability `reports/step6_rung_pilot.md`
found was a consequence of the gate being *unbounded*, independent of every other design
choice (pooling, structural features, learning-rate schedule) -- those are Variants B/C/D's
jobs, each tested in its own separate file/model, never combined with this one.
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rung_gate_common import NUM_DEPTHS, init_near_onehot_bias

DEFAULT_LAMBDA_BOUND = 0.3


class Rung5VariantAGateHead(nn.Module):
    """One task's bounded-residual depth gate. See module docstring."""

    def __init__(self, hidden: int, target_depth: int, mlp_hidden: int = 32,
                 num_depths: int = NUM_DEPTHS, lambda_bound: float = DEFAULT_LAMBDA_BOUND):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_depths),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # FIXED buffer, not nn.Parameter -- never trained, the core of this variant.
        self.register_buffer("position_bias", init_near_onehot_bias(num_depths, target_depth))
        self.lambda_bound = lambda_bound
        self.num_depths = num_depths

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens`: `[N, num_depths, hidden]`, depth-ordered (index 0 = h^0). Returns
        `(z [N, hidden], alpha [N, num_depths])`."""
        pooled = tokens.mean(dim=1)
        residual = self.lambda_bound * torch.tanh(self.mlp(pooled))
        logits = self.position_bias + residual
        alpha = torch.softmax(logits, dim=-1)
        z = torch.einsum("nd,ndh->nh", alpha, tokens)
        return z, alpha


class Rung5VariantAHADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder("rgcn_attn_rung5_a", ...)`) +
    one `Rung5VariantAGateHead` + one `PredictionHead` per task. Returns the same
    `(logits, layers)` 2-tuple every other architecture returns. Per-node gate weights for
    the most recent forward pass are stashed in `_last_gate_weights` for reporting, exactly
    matching `reports/step6_rung_pilot.md`'s analysis contract."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 mlp_hidden: int = 32, lambda_bound: float = DEFAULT_LAMBDA_BOUND):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_rung5_a", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.gates = nn.ModuleDict({
            task: Rung5VariantAGateHead(hidden, MARKOV_READOUT_DEPTH[task], mlp_hidden=mlp_hidden,
                                         lambda_bound=lambda_bound)
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.lambda_bound = lambda_bound
        self.architecture = "rgcn_attn_rung5_a"
        self._last_gate_weights: dict[str, torch.Tensor] | None = None

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)
        logits = {}
        gate_weights = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            tokens = torch.stack([layer[entity_type] for layer in layers], dim=1)
            z, alpha = self.gates[task](tokens)
            logits[task] = self.heads[task](z)
            gate_weights[task] = alpha.detach()
        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
