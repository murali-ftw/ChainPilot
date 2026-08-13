"""
Step 6 Phase 2, Rung 5 -- prior-initialized MLP PER-NODE depth gate on top of SHARE
(RGCN+Attention, `rgcn_attn` in code). "Markov Scoping and Transformer 1.md" Part 6's
ladder, rung 5 ("prior-initialized gate (d->32->4)", ~2.2K params at the doc's reference
d=64) -- the doc's OWN recommended rung (§6.4). Companion to Rung 4
(`ml/models/rgcn_attn_rung4_encoder.py`) and Markov Phase 1
(`ml/models/rgcn_attn_markov_encoder.py`, `reports/step6_markov_phase1.md`) -- all three
trained together in `ml/run_rung_pilot.py` for a direct three-way comparison.

**What's new here vs. the Step 6 depth-gate.** Same upgrade as Rung 4's: a genuine
PER-NODE distribution, not one shared vector per task. Unlike Rung 4, there is no
attention between depth tokens -- the gate reads a single POOLED representation of a
node's five depth tokens and emits weights directly, exactly the theory doc's own
framing: "A convex combination of h^1-h^4 with learned weights summing to 1 is a gate,
not a Transformer" (§6.1).

**Pooling choice: MEAN, not concat.** Concatenating `[h^0_i .. h^4_i]` into a
`5*hidden`-dim vector would make the first MLP layer's parameter count scale with
`5*hidden*32` instead of `hidden*32` -- a 5x blow-up that would push Rung 5 well past
Rung 4's cost and defeat the entire point of the rung (the doc's own ladder is ordered by
cost, and rung 5 is supposed to be the cheap one). Mean-pooling across the depth
dimension keeps the MLP's input width at `hidden`, matching the doc's own `d -> 32 -> 4`
sizing almost exactly (this file uses `d -> 32 -> 5`, the one extra column for `h^0`).
The trade-off, stated plainly: mean-pooling discards each depth's identity before the
gate ever sees it (the gate infers "which depths are represented in this pooled
summary", not "what does h^1 specifically look like versus h^3") -- a real limitation
relative to Rung 4's per-token attention, and part of why Rung 4 costs more.

**Mechanism, per task, per node:**

    tokens_i = [h^0_i, h^1_i, h^2_i, h^3_i, h^4_i]              # [5, hidden]
    pooled_i = mean(tokens_i, dim=0)                             # [hidden]
    logits_i = MLP(pooled_i) + position_bias                     # [5] -- MLP:
                                                                    # hidden->32->5,
                                                                    # ReLU between layers
    alpha_i  = softmax(logits_i)                                  # [5], per-node
    z_i      = sum_d alpha_i[d] * tokens_i[d]                     # [hidden]

`z_i` blends the ORIGINAL tokens, identical mechanism to Rung 4's, so the two rungs
differ ONLY in how `alpha` is computed.

**Prior-initialization.** The MLP's FINAL linear layer's weight AND bias are
zero-initialized, so at construction `MLP(pooled_i) = 0` for every node regardless of
what the first layer or the pooling computed -- `logits_i = position_bias`, identical to
Rung 4's init strategy and using the same shared helper
(`ml/models/rung_gate_common.py::init_near_onehot_bias`). Same consequence: epoch 1's
forward pass matches Markov's fixed-depth behaviour to numerical precision.

**Parameter count at SHARE's real hidden=128** (not the theory doc's reference d=64):
first layer `128*32+32=4,128`, second layer `32*5+5=165`, `+5` position bias
`=4,298` per task, `x3` tasks `=12,894` total -- see the pilot script's printed
`parameter_count()` for the exact live number. ~2x the theory doc's ~2.2K reference
(that was sized for a 4-token gate at d=64; this project's d=128 and the extra `h^0`
token both push it up, as the doc's own appendix anticipated) -- still an order of
magnitude cheaper than Rung 4's ~198K.
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rung_gate_common import NUM_DEPTHS, init_near_onehot_bias


class Rung5GateHead(nn.Module):
    """One task's per-node prior-initialized MLP depth gate. See module docstring."""

    def __init__(self, hidden: int, target_depth: int, mlp_hidden: int = 32,
                 num_depths: int = NUM_DEPTHS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_depths),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.position_bias = nn.Parameter(init_near_onehot_bias(num_depths, target_depth))
        self.num_depths = num_depths

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens`: `[N, num_depths, hidden]`, depth-ordered (index 0 = h^0). Returns
        `(z [N, hidden], alpha [N, num_depths])`."""
        pooled = tokens.mean(dim=1)
        logits = self.mlp(pooled) + self.position_bias
        alpha = torch.softmax(logits, dim=-1)
        z = torch.einsum("nd,ndh->nh", alpha, tokens)
        return z, alpha


class Rung5HADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder("rgcn_attn_rung5", ...)`) +
    one `Rung5GateHead` + one `PredictionHead` per task. Returns the same `(logits,
    layers)` 2-tuple every other architecture returns. Per-node gate weights for the most
    recent forward pass are stashed in `_last_gate_weights` for reporting."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 mlp_hidden: int = 32):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_rung5", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.gates = nn.ModuleDict({
            task: Rung5GateHead(hidden, MARKOV_READOUT_DEPTH[task], mlp_hidden=mlp_hidden)
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = "rgcn_attn_rung5"
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
