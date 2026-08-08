"""
Transformer 2, Variant 2 — Per-Node Trust Gate. ONE isolated change on top of the
ORIGINAL additive-fusion Transformer 2
(`ml/models/rgcn_attn_variant_a_transformer2.py`, `reports/step7_transformer2_pilot.md`),
tested standalone against the same shared baselines as Variants 1/3 — never stacked
with either.

**Motivation.** The original design's `t2_scale` is a single global scalar -- every
Supplier gets exactly the same fusion strength, learned once for the whole population.
This variant tests whether letting the model LEARN, per node, how much to trust
Transformer 2's output -- rather than deriving trust deterministically from retrieval
statistics alone (Variant 1's job) -- finds a useful pattern: are there specific kinds
of suppliers (by degree, by H_POLYMER membership, by anything else the encoder's own
embedding captures) the model learns to weight the discovered signal more heavily for?

**The one change.** Reuses the Variant A pattern EXACTLY (`ml/models/
rgcn_attn_rung5_variant_a.py`'s own `Rung5VariantAGateHead` bounded-residual design,
imported and reused unmodified for delay/shortage/impact's own depth gates) as the
template for a NEW, small per-node trust gate over Transformer 2's contribution
specifically:

    trust_i = tanh(MLP(z_impact_i))          MLP: hidden -> 32 -> 1, ReLU between
    z_impact_fused_i = z_impact_i + trust_i * t2_output_i

The MLP's FINAL layer is zero-initialized, so `trust_i = tanh(0) = 0` for EVERY node
at construction -- numerically identical to the original's `t2_scale = 0` safe start
(no worse than the current baseline for any node, regardless of that node's own
features). `tanh` bounds `trust_i` to `[-1, 1]`, the same bounding device Variant A's
own gate uses -- training can only move a node's trust away from zero deliberately,
and can never make it unboundedly large.

**Per-node stability analysis (the actual point of this experiment, per this round's
own instruction).** Unlike depth-selection (Step 6), there is no discrete prior
"depth" for trust to match or deviate from -- trust is a continuous scalar, not a
softmax over candidate positions. The adaptation used here, computed by the run
script, not this file: (1) the post-training DISTRIBUTION of `trust_i` across all test
Suppliers (mean/std, share pushed meaningfully positive vs. negative vs. still near
zero); (2) correlation between `|trust_i|` and Supplier degree (same style as Step 6's
depth-gate degree-correlation checks); (3) whether the 4 known H_POLYMER members
(`ml/graph/hidden_dependency_ground_truth.py`) receive systematically higher trust
than the Supplier population at large -- a direct, interpretable sanity check specific
to this dataset's own planted scenario.
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rgcn_attn_rung5_variant_a import DEFAULT_LAMBDA_BOUND, Rung5VariantAGateHead
from ml.models.transformer2 import Transformer2GlobalAttention

ARCHITECTURE_ID = "rgcn_attn_t2_trustgate"


class TrustGate(nn.Module):
    """Per-node learned trust in Transformer 2's contribution. Same
    zero-init-final-layer + bounded-nonlinearity convention as
    `ml/models/rgcn_attn_rung5_variant_a.py::Rung5VariantAGateHead`."""

    def __init__(self, hidden: int, mlp_hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, z_impact: torch.Tensor) -> torch.Tensor:
        """`z_impact`: `[N, hidden]`. Returns `trust [N, 1]`, in `[-1, 1]`, `0` at
        construction for every node."""
        return torch.tanh(self.mlp(z_impact))


class Transformer2TrustGateHADESModel(nn.Module):
    """Encoder + Variant A gates (reused unmodified) + `Transformer2GlobalAttention`
    (reused unmodified) + a per-node `TrustGate` replacing the global `t2_scale`,
    fused into ONLY the impact path. Returns the same `(logits, layers)` 2-tuple
    every other architecture returns. Per-node trust values for the most recent
    forward pass are stashed in `_last_trust` for the stability/degree analysis."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 mlp_hidden: int = 32, lambda_bound: float = DEFAULT_LAMBDA_BOUND,
                 t2_top_k: int = 64):
        super().__init__()
        self.encoder = build_encoder(ARCHITECTURE_ID, metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.gates = nn.ModuleDict({
            task: Rung5VariantAGateHead(hidden, MARKOV_READOUT_DEPTH[task], mlp_hidden=mlp_hidden,
                                         lambda_bound=lambda_bound)
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.transformer2 = Transformer2GlobalAttention(hidden, top_k=t2_top_k)
        self.trust_gate = TrustGate(hidden, mlp_hidden=mlp_hidden)
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = ARCHITECTURE_ID

        self._last_gate_weights: dict[str, torch.Tensor] | None = None
        self._last_t2: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_z_impact: torch.Tensor | None = None
        self._last_trust: torch.Tensor | None = None

    def forward(self, x_dict: dict, edge_index_dict: dict,
                is_frontier_supplier: torch.Tensor | None = None) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)

        z_by_task = {}
        gate_weights = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            tokens = torch.stack([layer[entity_type] for layer in layers], dim=1)
            z, alpha = self.gates[task](tokens)
            z_by_task[task] = z
            gate_weights[task] = alpha.detach()

        z_impact = z_by_task["impact"]
        t2_out, t2_alpha, t2_idx = self.transformer2(z_impact, is_frontier=is_frontier_supplier)
        trust = self.trust_gate(z_impact)  # [N, 1]
        z_impact_fused = z_impact + trust * t2_out
        self._last_t2 = (t2_idx.detach(), t2_alpha.detach())
        self._last_z_impact = z_impact.detach()
        self._last_trust = trust.detach().squeeze(-1)

        logits = {}
        for task in TASKS:
            z = z_impact_fused if task == "impact" else z_by_task[task]
            logits[task] = self.heads[task](z)

        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
