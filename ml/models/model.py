"""
`HADESModel` — encoder + structural depth-prior readout + prediction heads,
ported from `HADES_v1/ml/models/model.py`. The only thing that changes between
architectures is a constructor argument, never the training or evaluation code.
"""

from __future__ import annotations

from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS, readout_layer_for_task
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead


class HADESModel(nn.Module):
    def __init__(self, architecture: str, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, shared_depth: int | None = None,
                 depth_prior: dict[str, int] | None = None, dropout: float = 0.2,
                 num_bases: int = 8, relation_embed_dim: int = 16, num_bases_attn: int = 8):
        super().__init__()
        self.encoder = build_encoder(architecture, metadata, in_dims, hidden=hidden,
                                     num_layers=num_layers, dropout=dropout, num_bases=num_bases,
                                     relation_embed_dim=relation_embed_dim,
                                     num_bases_attn=num_bases_attn)
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.shared_depth = shared_depth
        self.depth_prior = depth_prior
        self.architecture = architecture
        self.hidden = hidden
        self.num_bases = num_bases

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)
        logits = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            layer_idx = readout_layer_for_task(task, len(layers), self.shared_depth,
                                               self.depth_prior) - 1
            logits[task] = self.heads[task](layers[layer_idx][entity_type])
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
