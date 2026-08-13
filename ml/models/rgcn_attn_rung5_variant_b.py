"""
Step 6 Phase 2, Rung 5 Variant B -- structural features added to the gate input. ONE
isolated change on top of the ORIGINAL Rung 5 gate (`ml/models/rgcn_attn_rung5_encoder.py`,
`reports/step6_rung_pilot.md`), tested standalone against the same shared baselines as
Variants A/C/D -- never stacked with any of them.

**Motivation.** `reports/step6_rung_pilot.md`'s per-node analysis found a real, 5-seed
CONSISTENT correlation between deviation-from-prior and node degree, but only for the
original Rung 5's mean-pooled embedding input -- the gate had to *infer* structural
position (hub vs. leaf) indirectly, purely from what the encoder's message passing had
already folded into `h^0..h^4`. This variant gives the gate that structural information
directly, to test whether an explicit signal produces a cleaner, more consistent
depth-selection pattern than an implicit one has to reconstruct on its own.

**The one change (everything else identical to the original Rung 5 gate).** The gate's
MLP input widens from `hidden` (the mean-pooled depth-token embedding alone) to
`hidden + 1 + num_node_types + num_task_relations`, by concatenating, per node:

    1. normalized total degree (edges touching this node in this snapshot, across every
       relation, divided by that entity type's mean degree in the same snapshot -- so the
       feature is centered near 1.0 rather than depending on the graph's raw scale)
    2. one-hot node type (over ALL node types in the graph's metadata) -- NOTE: within any
       one task's gate this is a CONSTANT vector (every node the gate ever sees belongs to
       that task's single fixed entity type, `ml/models/depth.py::TASK_ENTITY_TYPE`), so it
       cannot discriminate between nodes -- implemented as specified regardless, and flagged
       here explicitly rather than silently omitted, since it's a legitimate (if degenerate
       for this design) structural feature.
    3. a per-relation edge-count histogram: one count per relation type where this node's
       entity type appears as the destination (`data.edge_index_dict`'s dst-side, matching
       `ml/run_rung_pilot.py::_entity_degree`'s convention) -- e.g. "how many `SUPPLIES`
       edges, how many `SHIPS_FROM` edges" rather than one pooled total.

Position bias stays a LEARNED `nn.Parameter` (unlike Variant A) and the learning rate is
unchanged -- identical to the original Rung 5 in every respect except the gate's input
width and content.

**What this tests in isolation.** Whether making structural position explicit, rather
than requiring the gate to infer it from pooled embeddings alone, changes match-rate
stability or the degree-deviation correlation -- independent of bounding (Variant A),
learning-rate scheduling (Variant C), or token pooling (Variant D).
"""

from __future__ import annotations

import torch
from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.models.rung_gate_common import NUM_DEPTHS, init_near_onehot_bias


class Rung5VariantBGateHead(nn.Module):
    """One task's structural-features-augmented depth gate. See module docstring."""

    def __init__(self, hidden: int, extra_dim: int, target_depth: int, mlp_hidden: int = 32,
                 num_depths: int = NUM_DEPTHS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden + extra_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_depths),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.position_bias = nn.Parameter(init_near_onehot_bias(num_depths, target_depth))
        self.num_depths = num_depths

    def forward(self, tokens: torch.Tensor, structural_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens`: `[N, num_depths, hidden]`. `structural_features`: `[N, extra_dim]`.
        Returns `(z [N, hidden], alpha [N, num_depths])`."""
        pooled = tokens.mean(dim=1)
        gate_input = torch.cat([pooled, structural_features], dim=-1)
        logits = self.mlp(gate_input) + self.position_bias
        alpha = torch.softmax(logits, dim=-1)
        z = torch.einsum("nd,ndh->nh", alpha, tokens)
        return z, alpha


class Rung5VariantBHADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder("rgcn_attn_rung5_b", ...)`) +
    one `Rung5VariantBGateHead` + one `PredictionHead` per task, with each task's gate fed
    a structural-features vector computed fresh from `edge_index_dict` on every forward
    pass. Returns the same `(logits, layers)` 2-tuple every other architecture returns.
    Per-node gate weights stashed in `_last_gate_weights` for reporting."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 mlp_hidden: int = 32):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_rung5_b", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)

        node_types, edge_types = metadata
        self._node_types = sorted(node_types)
        # Per task's entity type, the fixed list of (src, rel, dst) edge_types where it is
        # the DESTINATION -- computed once here, reused every forward pass, so the
        # per-relation histogram's column order/width is stable across calls.
        self._task_relations = {
            task: [et for et in edge_types if et[2] == entity_type]
            for task, entity_type in TASK_ENTITY_TYPE.items()
        }

        self.gates = nn.ModuleDict({
            task: Rung5VariantBGateHead(
                hidden,
                extra_dim=1 + len(self._node_types) + len(self._task_relations[task]),
                target_depth=MARKOV_READOUT_DEPTH[task],
                mlp_hidden=mlp_hidden,
            )
            for task in TASKS
        })
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = "rgcn_attn_rung5_b"
        self._last_gate_weights: dict[str, torch.Tensor] | None = None

    def _structural_features(self, x_dict: dict, edge_index_dict: dict, task: str) -> torch.Tensor:
        entity_type = TASK_ENTITY_TYPE[task]
        n = x_dict[entity_type].size(0)
        relations = self._task_relations[task]
        device = x_dict[entity_type].device

        rel_hist = torch.zeros(n, len(relations), device=device)
        for j, edge_type in enumerate(relations):
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() > 0:
                rel_hist[:, j] = torch.bincount(edge_index[1], minlength=n).float()
        total_degree = rel_hist.sum(dim=1)
        mean_degree = total_degree.mean().clamp_min(1e-6)
        norm_degree = (total_degree / mean_degree).unsqueeze(-1)

        type_idx = self._node_types.index(entity_type)
        one_hot = torch.zeros(n, len(self._node_types), device=device)
        one_hot[:, type_idx] = 1.0

        return torch.cat([norm_degree, one_hot, rel_hist], dim=-1)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)
        logits = {}
        gate_weights = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            tokens = torch.stack([layer[entity_type] for layer in layers], dim=1)
            structural = self._structural_features(x_dict, edge_index_dict, task)
            z, alpha = self.gates[task](tokens, structural)
            logits[task] = self.heads[task](z)
            gate_weights[task] = alpha.detach()
        self._last_gate_weights = gate_weights
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
