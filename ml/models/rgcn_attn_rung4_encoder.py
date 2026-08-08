"""
Step 6 Phase 2, Rung 4 -- attention-only PER-NODE depth gate on top of SHARE
(RGCN+Attention, `rgcn_attn` in code). "Markov Scoping and Transformer 1.md" Part 6's
ladder, rung 4 ("1-layer, attention only (no FFN)", ~17K params at the doc's reference
d=64). Companion to Rung 5 (`ml/models/rgcn_attn_rung5_encoder.py`) and Markov Phase 1
(`ml/models/rgcn_attn_markov_encoder.py`, `reports/step6_markov_phase1.md`) -- all three
trained together in `ml/run_rung_pilot.py` for a direct three-way comparison.

**What's new here vs. the Step 6 depth-gate (`rgcn_attn_depthgate`,
`reports/step6_depth_gate.md`).** That gate learned ONE shared weight vector per task --
every node of a given task read the exact same depth blend. This is a genuine PER-NODE
hybrid: each node gets its OWN depth-weight distribution, computed from its own five
depth-token representations `[h^0_i .. h^4_i]` via a single self-attention layer (so
`h^1_i` and `h^3_i` can attend to each other before the gate decides the blend for node
`i` specifically) -- the capability a per-task-global gate could never express (hub vs.
leaf suppliers, or any other node-level distinction, getting different depths).

**Mechanism, per task, per node:**

    tokens_i   = [h^0_i, h^1_i, h^2_i, h^3_i, h^4_i]            # [5, hidden]
    attn_out_i = SelfAttention(tokens_i)                         # [5, hidden], no FFN
    logits_i   = Linear(hidden -> 1)(attn_out_i).squeeze(-1)
                 + position_bias                                 # [5]
    alpha_i    = softmax(logits_i)                                # [5], per-node
    z_i        = sum_d alpha_i[d] * tokens_i[d]                   # [hidden]

`z_i` blends the ORIGINAL tokens, not the attention output -- attention only decides the
WEIGHTS, exactly matching Rung 5's framing too ("a gate, not a Transformer",
"Markov Scoping and Transformer 1.md" §6.1). Kept identical across both rungs on purpose,
so they differ ONLY in how `alpha` is computed, nothing else -- a clean ablation of
"does attention between depth tokens help, over a plain gate."

**Prior-initialization (the "shouldn't plausibly do worse than Markov" requirement).**
`readout`'s weight AND bias are zero-initialized, so at construction `logits_i =
position_bias` for EVERY node (identical, content-independent) --
`position_bias = init_near_onehot_bias(5, MARKOV_READOUT_DEPTH[task])`
(`ml/models/rung_gate_common.py`), i.e. ~99.9% of the softmax mass sits on the exact
depth Markov Phase 1 already validated for that task. Training can only move logits away
from that safe starting point through the (zero-initialized) `readout` weight and the
attention block's own randomly-initialized parameters -- multiplied by a zeroed readout
at the very start, so epoch 1's forward pass is, to numerical precision, identical to
Markov's fixed-depth behaviour; any subsequent per-node divergence is something training
actually learned, not an initialization artifact.

**Attention config.** `nn.MultiheadAttention(embed_dim=hidden, num_heads=4,
batch_first=True)`, self-attention (query=key=value=`tokens`), consistent with this
project's other 4-head attention modules (`ml/models/encoder.py::HGTEncoder`). Parameter
count at SHARE's real hidden=128 (NOT the theory doc's reference d=64):
`4*(128^2+128) = 66,048` (Q/K/V/O projections) `+ 129` (readout: 128 weights + 1 bias)
`+ 5` (position bias) `= 66,182` per task, `x3` tasks `= 198,546` total -- see the pilot
script's printed `parameter_count()` for the exact live number. The theory doc's own
~17K figure was derived at d=64 and scales roughly with `d^2`, which is why d=128 costs
~4x more -- exactly what the doc's own appendix note anticipated ("Recompute all of
these once real dimensions... exist").
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rung_gate_common import NUM_DEPTHS, init_near_onehot_bias


class Rung4GateHead(nn.Module):
    """One task's per-node attention-only depth gate. See module docstring."""

    def __init__(self, hidden: int, target_depth: int, num_heads: int = 4,
                 num_depths: int = NUM_DEPTHS):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)
        self.readout = nn.Linear(hidden, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.position_bias = nn.Parameter(init_near_onehot_bias(num_depths, target_depth))
        self.num_depths = num_depths

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens`: `[N, num_depths, hidden]`, depth-ordered (index 0 = h^0). Returns
        `(z [N, hidden], alpha [N, num_depths])`."""
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        logits = self.readout(attn_out).squeeze(-1) + self.position_bias
        alpha = torch.softmax(logits, dim=-1)
        z = torch.einsum("nd,ndh->nh", alpha, tokens)
        return z, alpha


class Rung4HADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder("rgcn_attn_rung4", ...)`) +
    one `Rung4GateHead` + one `PredictionHead` per task. Returns the same `(logits,
    layers)` 2-tuple every other architecture returns, so `ml/evaluate.py` needs zero
    changes. Per-node gate weights for the most recent forward pass are stashed in
    `_last_gate_weights` (dict[task] -> `[N_task, num_depths]` tensor) for reporting --
    never read by `ml/evaluate.py`."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 num_heads: int = 4):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_rung4", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.gates = nn.ModuleDict({
            task: Rung4GateHead(hidden, MARKOV_READOUT_DEPTH[task], num_heads=num_heads)
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = "rgcn_attn_rung4"
        self._last_gate_weights: dict[str, torch.Tensor] | None = None

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)
        logits = {}
        gate_weights = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            tokens = torch.stack([layer[entity_type] for layer in layers], dim=1)  # [N,5,hidden]
            z, alpha = self.gates[task](tokens)
            logits[task] = self.heads[task](z)
            gate_weights[task] = alpha.detach()
        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
