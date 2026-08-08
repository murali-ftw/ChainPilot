"""
SHARE + Variant A + Transformer 2 — the finalized Layer 1 + Layer 2 stack plus Claim 3
(Transformer 2, global same-type attention, `ml/models/transformer2.py`). A new,
purely additive composition: reuses `Rung5VariantAGateHead` and `PredictionHead`
UNMODIFIED (imported, not reimplemented) for every task's depth-selection and
readout, and inserts `Transformer2GlobalAttention` ONLY into the impact path.

**DEVIATION FROM `10_AI_ML_Documentation.md` §8.3, FLAGGED EXPLICITLY.** §8.3
specifies wiring Transformer 2's output to the delay head only ("Phase 1: wired to
the delay head only (supplier-level)"). This implementation wires it to the IMPACT
head instead, per this project's own Step 6 Round 1 reach analysis
(`reports/layer_2.md` Round 1 / `reports/step6_preflight.md`'s original content):
the co-parent/hidden-dependency path needs a hard minimum of 3 hops from delay's
actual prediction target (Shipment) — `Shipment -> Supplier (1) -> Component (2) ->
co-parent Supplier (3)`, confirmed 100% of supplier-sourced shipments, 0% at hop 1
or 2 — beyond what delay ever reads under Markov/Variant A (`h^1`, one hop). Delay's
own readout structurally cannot reach this signal at ANY depth tested in this
project's history, baseline included. The SAME analysis found impact's target
(Supplier, hop 0 by definition) DOES have real co-parent signal available from hop 2
onward, and impact reads `h^4` under Markov/Variant A — well within reach. Wiring
Transformer 2 to delay per the original doc's assumption would attach it to a head
that is structurally incapable of using it; wiring it to impact attaches it to the
one head whose own readout depth and own reach-analysis evidence both support it.

**Composition.**

    layers = encoder(x, edge_index)                         # SHARE, h^0..h^4, unchanged
    for task in {delay, shortage, impact}:
        z_task, alpha_task = VariantA_gate[task](tokens)     # unchanged, per-task
    t2_out, t2_alpha, t2_idx = Transformer2(z_impact)         # NEW, Supplier-only
    z_impact_fused = z_impact + t2_scale * t2_out             # t2_scale init=0 (safe start)
    logits[impact] = PredictionHead[impact](z_impact_fused)
    logits[delay]     = PredictionHead[delay](z_delay)         # unchanged
    logits[shortage]  = PredictionHead[shortage](z_shortage)   # unchanged

Returns the same `(logits, layers)` 2-tuple every other architecture returns, so
`ml/evaluate.py` needs zero changes. Transformer 2's attention weights/candidate
indices for the most recent forward pass are stashed in `_last_t2` (for
`hidden_dependency_links` logging and the hidden-dependency validation check) —
never read by `ml/evaluate.py`.
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

ARCHITECTURE_ID = "rgcn_attn_variant_a_transformer2"


class Transformer2HADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder(ARCHITECTURE_ID, ...)`)
    + one `Rung5VariantAGateHead` per task (imported unmodified from
    `ml/models/rgcn_attn_rung5_variant_a.py`) + one `Transformer2GlobalAttention`
    fused into ONLY the impact path + one `PredictionHead` per task."""

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
        self.t2_scale = nn.Parameter(torch.zeros(1))  # learned, init 0 -- safe start
        self.num_layers = num_layers
        self.hidden = hidden
        self.lambda_bound = lambda_bound
        self.t2_top_k = t2_top_k
        self.architecture = ARCHITECTURE_ID

        self._last_gate_weights: dict[str, torch.Tensor] | None = None
        self._last_t2: tuple[torch.Tensor, torch.Tensor] | None = None  # (candidate_idx, alpha)
        self._last_z_impact: torch.Tensor | None = None  # pre-fusion, for hidden-dependency analysis

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
        z_impact_fused = z_impact + self.t2_scale * t2_out
        self._last_t2 = (t2_idx.detach(), t2_alpha.detach())
        self._last_z_impact = z_impact.detach()

        logits = {}
        for task in TASKS:
            z = z_impact_fused if task == "impact" else z_by_task[task]
            logits[task] = self.heads[task](z)

        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
