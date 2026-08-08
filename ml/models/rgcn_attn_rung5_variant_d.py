"""
Step 6 Phase 2, Rung 5 Variant D -- depth-identity-preserving projection instead of
mean-pooling. ONE isolated change on top of the ORIGINAL Rung 5 gate
(`ml/models/rgcn_attn_rung5_encoder.py`, `reports/step6_rung_pilot.md`), tested standalone
against the same shared baselines as Variants A/B/C -- never stacked with any of them.

**Motivation.** The original Rung 5 gate's own docstring names this exact trade-off:
mean-pooling `[h^0_i .. h^4_i]` into one `hidden`-dim vector before the MLP ever sees it
"discards each depth's identity before the gate ever sees it (the gate infers 'which
depths are represented in this pooled summary', not 'what does h^1 specifically look like
versus h^3')". This variant removes exactly that limitation, at a small parameter cost,
without adding attention (Rung 4's job, already tested and found not to help,
`reports/step6_rung_pilot.md`).

**The one change (everything else identical to the original Rung 5 gate).** Replace
mean-pooling with a SHARED small linear projection (`hidden -> 8`) applied to each of
`h^0..h^4` INDIVIDUALLY (the same `nn.Linear` weights reused at every depth position, so
the parameter count stays small and every depth is projected through an identical
transform -- a fair comparison, not one depth getting privileged treatment), then
concatenated into one `5*8=40`-dim vector before the rest of the gate:

    proj_i = [ W(h^0_i), W(h^1_i), W(h^2_i), W(h^3_i), W(h^4_i) ]   # [40], W: hidden->8
    logits_i = MLP(proj_i) + position_bias                          # MLP: 40 -> 32 -> 5

Position bias stays a LEARNED `nn.Parameter` (as in the original) and the learning rate is
unchanged. The blend `z_i` still uses the ORIGINAL (unprojected) tokens, exactly as every
other rung in this line does -- the projection only feeds the gate's DECISION, never the
value being blended.

**What this tests in isolation.** Whether preserving each depth's identity through the
gate's input (rather than averaging it away) changes match-rate stability or the
degree-deviation correlation -- independent of bounding (Variant A), structural features
(Variant B), or learning-rate scheduling (Variant C).
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rung_gate_common import NUM_DEPTHS, init_near_onehot_bias

DEFAULT_PROJ_DIM = 8


class Rung5VariantDGateHead(nn.Module):
    """One task's depth-identity-preserving depth gate. See module docstring."""

    def __init__(self, hidden: int, target_depth: int, proj_dim: int = DEFAULT_PROJ_DIM,
                 mlp_hidden: int = 32, num_depths: int = NUM_DEPTHS):
        super().__init__()
        self.proj = nn.Linear(hidden, proj_dim)  # SHARED across all `num_depths` positions
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * num_depths, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_depths),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.position_bias = nn.Parameter(init_near_onehot_bias(num_depths, target_depth))
        self.num_depths = num_depths
        self.proj_dim = proj_dim

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens`: `[N, num_depths, hidden]`, depth-ordered (index 0 = h^0). Returns
        `(z [N, hidden], alpha [N, num_depths])`."""
        projected = self.proj(tokens)  # nn.Linear broadcasts over the depth dim -> [N, num_depths, proj_dim]
        flat = projected.reshape(projected.size(0), -1)  # [N, num_depths*proj_dim]
        logits = self.mlp(flat) + self.position_bias
        alpha = torch.softmax(logits, dim=-1)
        z = torch.einsum("nd,ndh->nh", alpha, tokens)  # blend uses the ORIGINAL tokens
        return z, alpha


class Rung5VariantDHADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder("rgcn_attn_rung5_d", ...)`) +
    one `Rung5VariantDGateHead` + one `PredictionHead` per task. Returns the same
    `(logits, layers)` 2-tuple every other architecture returns. Per-node gate weights
    stashed in `_last_gate_weights` for reporting."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 mlp_hidden: int = 32, proj_dim: int = DEFAULT_PROJ_DIM):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_rung5_d", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.gates = nn.ModuleDict({
            task: Rung5VariantDGateHead(hidden, MARKOV_READOUT_DEPTH[task], proj_dim=proj_dim,
                                         mlp_hidden=mlp_hidden)
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = "rgcn_attn_rung5_d"
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
