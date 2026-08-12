"""
Transformer 2, Variant 1 — Confidence-Aware Fusion. ONE isolated change on top of the
ORIGINAL additive-fusion Transformer 2
(`ml/models/rgcn_attn_variant_a_transformer2.py`, `reports/step7_transformer2_pilot.md`),
tested standalone against the same shared baselines as Variants 2/3 — never stacked
with either.

**Motivation.** The original design fuses Transformer 2's output with a single GLOBAL
learned scalar (`t2_scale`) applied identically to every Supplier, regardless of how
good that Supplier's own top-64 retrieval actually was. A Supplier whose top-64 pool
is all weak, near-random cosine matches gets the same fusion strength as one whose
pool contains a handful of very strong, confident matches. This variant tests whether
making fusion strength vary by RETRIEVAL CONFIDENCE — computed directly and
DETERMINISTICALLY from the existing retrieval mechanism, no new learned parameters for
the score itself, per this round's own instruction — changes anything.

**The one change.** `Transformer2GlobalAttention` (`ml/models/transformer2.py`) is
reused UNMODIFIED — same top-64 cosine-similarity pool, same attention mechanism, same
`(t2_output, alpha, topk_idx)` return signature. A per-node confidence score is
computed from `alpha` (the pool's own attention weights) with NO new learned
parameters:

    mean_topk_cos_i   = mean cosine similarity of node i's top-64 pool (higher = a
                         genuinely similar candidate pool exists at all)
    attn_entropy_i    = -sum(alpha_i * log(alpha_i)) over the pool (lower = the
                         attention concentrated on a few strong matches, rather than
                         spreading weakly across all 64 -- "confident" retrieval)
    confidence_i      = 0.5 * (minmax(mean_topk_cos)_i + minmax(1 - attn_entropy/log(k))_i)

both terms min-max normalized ACROSS THE BATCH (relative confidence within this
snapshot's Supplier population, not an absolute threshold), then averaged. This
confidence score is a plain tensor computation -- `torch.no_grad()`-safe, no
`nn.Parameter` anywhere in it.

**Fusion.** `z_impact_fused = z_impact + t2_scale * confidence_i * t2_output_i` --
`t2_scale` remains a single LEARNED global scalar (so training can still learn an
overall magnitude), now additionally modulated per-node by the non-learned confidence
score. At construction, `t2_scale = 0` (same safe-start convention as the original),
so `z_impact_fused = z_impact` regardless of any node's confidence value -- confidence
only matters once training has moved `t2_scale` away from zero.
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

ARCHITECTURE_ID = "rgcn_attn_t2_confidence"


def _minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def compute_confidence(alpha: torch.Tensor, topk_cos: torch.Tensor) -> torch.Tensor:
    """`alpha`: `[N, k]` attention weights over the retrieval pool. `topk_cos`: `[N, k]`
    the pool's own cosine similarities (pre-attention). Returns `[N]` confidence in
    `[0, 1]`, no learned parameters. Exposed as a module-level function (not just
    inlined) so the run script's Prerequisite 1 diagnostic can call it directly on an
    already-trained model without re-deriving the formula."""
    mean_topk_cos = topk_cos.mean(dim=-1)
    attn_entropy = -(alpha * alpha.clamp_min(1e-8).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(alpha.size(-1))))
    entropy_confidence = 1.0 - (attn_entropy / max_entropy)
    return 0.5 * (_minmax(mean_topk_cos) + _minmax(entropy_confidence))


class Transformer2ConfidenceHADESModel(nn.Module):
    """Encoder + Variant A gates (reused unmodified) + `Transformer2GlobalAttention`
    (reused unmodified) + a confidence-modulated fusion into ONLY the impact path.
    Returns the same `(logits, layers)` 2-tuple every other architecture returns.
    Confidence scores and T2 internals for the most recent forward pass are stashed
    for reporting/diagnostics -- never read by `ml/evaluate.py`."""

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
        self.architecture = ARCHITECTURE_ID

        self._last_gate_weights: dict[str, torch.Tensor] | None = None
        self._last_t2: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_z_impact: torch.Tensor | None = None
        self._last_confidence: torch.Tensor | None = None

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

        # Recompute the pool's own cosine similarities (pre-attention) for the
        # confidence score -- cheap, and keeps `Transformer2GlobalAttention` itself
        # unmodified rather than having it expose an extra return value.
        with torch.no_grad():
            normed = torch.nn.functional.normalize(z_impact, dim=-1)
            full_cos = normed @ normed.T
            topk_cos = full_cos.gather(1, t2_idx)
            confidence = compute_confidence(t2_alpha, topk_cos)

        z_impact_fused = z_impact + self.t2_scale * confidence.unsqueeze(-1) * t2_out
        self._last_t2 = (t2_idx.detach(), t2_alpha.detach())
        self._last_z_impact = z_impact.detach()
        self._last_confidence = confidence.detach()

        logits = {}
        for task in TASKS:
            z = z_impact_fused if task == "impact" else z_by_task[task]
            logits[task] = self.heads[task](z)

        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
