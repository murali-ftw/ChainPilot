"""
Transformer 2, Variant 3 — Cross-Attention Fusion. ONE isolated change on top of the
ORIGINAL additive-fusion Transformer 2
(`ml/models/rgcn_attn_variant_a_transformer2.py`, `reports/step7_transformer2_pilot.md`),
tested standalone against the same shared baselines as Variants 1/2 — never stacked
with either.

**Motivation.** The original design collapses the whole top-64 retrieval pool into
ONE vector (`t2_output`, Transformer 2's own internal attention-weighted sum) before
the impact head ever sees it -- whatever Transformer 2's own attention decided is
important is final; the impact head has no further say. This variant tests whether
letting the impact embedding itself CROSS-ATTEND over the raw retrieved pool --
choosing its own weighting of the same 64 candidates, informed by what the impact
head specifically needs, rather than inheriting Transformer 2's own generic weighting
-- finds a more useful fusion.

**The one change.** `Transformer2GlobalAttention` (`ml/models/transformer2.py`) is
reused UNMODIFIED for the retrieval step (same top-64 cosine pool, same relation to
the original pilot) -- but this file does its OWN small, self-contained top-64
cosine-similarity retrieval locally (duplicating that ~10-line computation rather than
extending `Transformer2GlobalAttention`'s return signature) specifically so this
variant's pool-value access doesn't require changing a shared file every other variant
and the original pilot also import -- keeping all three variants strictly isolated
from one another, per this round's own "never stacked" requirement.

    pool_values_i = V(h^d_j) for j in top-64(i)          # [N, 64, hidden], SAME pool
                                                            # Transformer2GlobalAttention
                                                            # would retrieve
    query_i       = z_impact_i                            # impact's OWN embedding
    cross_out_i   = MultiheadAttention(query_i, pool_values_i, pool_values_i)
    z_impact_fused_i = z_impact_i + Linear_zero_init(cross_out_i)

`nn.MultiheadAttention` (4 heads, matching this project's other attention modules)
computes the cross-attention itself; a SEPARATE, zero-initialized `Linear` layer sits
after it, so the fused contribution is EXACTLY zero at construction regardless of what
the (non-zero-initialized) attention internals compute -- the same "wrap in a zeroed
final layer" convention used everywhere else in this project (Rung 5's gates, Variant
A, the original Transformer 2 pilot's own `t2_scale=0`). Only a single query token per
node (`z_impact_i` itself) attends over the 64-token pool -- not the reverse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rgcn_attn_rung5_variant_a import DEFAULT_LAMBDA_BOUND, Rung5VariantAGateHead

ARCHITECTURE_ID = "rgcn_attn_t2_crossattn"


class CrossAttentionFusion(nn.Module):
    """Self-contained top-64 cosine retrieval + cross-attention fusion. Does its own
    retrieval (not `Transformer2GlobalAttention`) so this variant never touches that
    shared file's return signature -- see module docstring."""

    def __init__(self, hidden: int, top_k: int = 64, num_heads: int = 4):
        super().__init__()
        self.v_proj = nn.Linear(hidden, hidden)
        self.mha = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.top_k = top_k
        self.hidden = hidden

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`z`: `[N, hidden]`, one embedding per Supplier (the impact readout).
        Returns `(fusion_delta [N, hidden], candidate_idx [N, k])` -- `fusion_delta`
        is exactly `0` at construction; `candidate_idx` is exposed for the
        hidden-dependency validation/logging, matching the original pilot's own
        contract."""
        n = z.size(0)
        k = min(self.top_k, n - 1)
        normed = F.normalize(z, dim=-1)
        cos_sim = (normed @ normed.T).clone()
        cos_sim.fill_diagonal_(float("-inf"))
        _, topk_idx = cos_sim.topk(k, dim=-1)  # [N, k]

        pool_values = self.v_proj(z)[topk_idx]  # [N, k, hidden]
        query = z.unsqueeze(1)  # [N, 1, hidden]
        cross_out, cross_alpha = self.mha(query, pool_values, pool_values, need_weights=True)
        cross_out = cross_out.squeeze(1)  # [N, hidden]
        fusion_delta = self.out_proj(cross_out)  # 0 at construction
        return fusion_delta, topk_idx, cross_alpha.squeeze(1)  # alpha: [N, k]


class Transformer2CrossAttnHADESModel(nn.Module):
    """Encoder + Variant A gates (reused unmodified) + `CrossAttentionFusion` fused
    into ONLY the impact path. Returns the same `(logits, layers)` 2-tuple every other
    architecture returns. `_last_t2` matches the original pilot's `(candidate_idx,
    alpha)` contract so the SAME hidden-dependency analysis code applies unchanged."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 mlp_hidden: int = 32, lambda_bound: float = DEFAULT_LAMBDA_BOUND,
                 t2_top_k: int = 64, num_heads: int = 4):
        super().__init__()
        self.encoder = build_encoder(ARCHITECTURE_ID, metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.gates = nn.ModuleDict({
            task: Rung5VariantAGateHead(hidden, MARKOV_READOUT_DEPTH[task], mlp_hidden=mlp_hidden,
                                         lambda_bound=lambda_bound)
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.cross_fusion = CrossAttentionFusion(hidden, top_k=t2_top_k, num_heads=num_heads)
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = ARCHITECTURE_ID

        self._last_gate_weights: dict[str, torch.Tensor] | None = None
        self._last_t2: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_z_impact: torch.Tensor | None = None

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
        fusion_delta, topk_idx, cross_alpha = self.cross_fusion(z_impact)
        z_impact_fused = z_impact + fusion_delta
        self._last_t2 = (topk_idx.detach(), cross_alpha.detach())
        self._last_z_impact = z_impact.detach()

        logits = {}
        for task in TASKS:
            z = z_impact_fused if task == "impact" else z_by_task[task]
            logits[task] = self.heads[task](z)

        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
