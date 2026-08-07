"""
`HADESModel` — encoder + depth-prior readout + prediction heads composed
into one module, so the *only* thing that changes between Step 3's baseline,
Step 4's L-sweep, and Step 5's architecture ablation is a constructor
argument (`architecture`, `num_layers`, `shared_depth`), never the training
or evaluation code around it.
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
                 num_bases: int = 8):
        super().__init__()
        self.encoder = build_encoder(architecture, metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, dropout=dropout, num_bases=num_bases)
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.shared_depth = shared_depth
        self.depth_prior = depth_prior  # None -> STRUCTURAL_DEPTH_PRIOR (as-documented)
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
