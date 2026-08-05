"""
Prediction heads + multi-task focal loss — `docs/10_AI_ML_Documentation.md`
§8.4, `project_HADES.md` §6.1-6.2.

    delay_v     = sigma( MLP(z_delay_v) )     Shipment
    shortage_v  = sigma( MLP(z_shortage_v) )  Product (see note below)
    impact_v    = sigma( MLP(z_impact_v) )    Supplier

Each head: 2-layer MLP, `Linear(d,d) -> ReLU -> Linear(d,1)`, matching
`project_HADES.md` §6.1's `(d^2+d) + (d+1)` parameter count. Heads return
raw logits; `FocalLoss` and evaluation both apply sigmoid themselves
(`BCEWithLogits`-style, for numerical stability).

Note on the shortage head's target: `project_HADES.md` describes shortage
as a Product x Warehouse-level prediction, and the data matches that
exactly -- `training_labels` carries one row per (product, warehouse)
stocking instance (194 rows for 80 products, confirmed by direct query:
every product has 1-3 rows, one per warehouse that stocks it), all tagged
`entity_type='product'` with the product's id repeated across its
warehouse instances. There is no separate Product-Warehouse *node* in this
graph (`STOCKED_AT` is an edge, not a node type), so the shortage head reads
the Product node's own embedding for every one of that product's labelled
instances -- multiple loss terms (and multiple held-out evaluation rows)
share the same embedding, trained toward whichever warehouse instance's
label they're paired with. This is a real, documented coarsening (the node
granularity is coarser than the label granularity), not a bug: an edge-level
head (fed by both Product and Warehouse embeddings) would resolve it, but
is out of scope for the Steps 3-5 baseline/ablation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PredictionHead(nn.Module):
    """2-layer MLP producing a single logit per node."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al. 2017), gamma=2, alpha from inverse class
    frequency (`project_HADES.md` §6.2) -- disruptions are a minority class
    at this dataset's label rates (~11%/~7%/~4%, `db/README.md`).

    `alpha` up-weights the positive class: alpha_t = alpha for y=1,
    (1-alpha) for y=0. Passing `alpha = 1 - positive_rate` (the standard
    inverse-frequency choice) means alpha > 0.5 whenever positives are the
    minority, which is exactly this dataset's case for all three tasks.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


def alpha_from_positive_rate(positive_rate: float) -> float:
    """`alpha = 1 - positive_rate`, the inverse-class-frequency choice
    (`project_HADES.md` §6.2), clamped away from 0/1 for numerical safety
    on tasks with very few positives (impact: 11/300 in the train window)."""
    return float(min(max(1.0 - positive_rate, 0.05), 0.95))
