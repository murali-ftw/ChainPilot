"""
Heterogeneous GAT for the ChainPilot / HADES multi-task problem.

Each relation type gets its own GATConv (heads attend within that relation);
HeteroConv sums the per-relation messages into each node type per layer.
Three linear heads read out task logits on the node types that actually
carry labels: shipment (delay), inventory (shortage), supplier (impact).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv, Linear


class HeteroGAT(nn.Module):
    def __init__(self, metadata, in_channels: dict, hidden_channels=64,
                 num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        node_types, edge_types = metadata

        # per-node-type input projection to a common width (GATConv needs
        # matching dims across a relation's src/dst after the first layer)
        self.input_proj = nn.ModuleDict({
            nt: Linear(in_channels[nt], hidden_channels) for nt in node_types
        })

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            out_dim = hidden_channels
            conv = HeteroConv({
                et: GATConv((-1, -1), out_dim // heads, heads=heads,
                            dropout=dropout, add_self_loops=False)
                for et in edge_types
            }, aggr="sum")
            self.convs.append(conv)

        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.head_delay = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1))
        self.head_shortage = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1))
        self.head_impact = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, 1))

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.input_proj[nt](x).relu() for nt, x in x_dict.items()}
        for conv, norm in zip(self.convs, self.norms):
            new_x = conv(x_dict, edge_index_dict)
            # residual connection + norm + dropout per node type
            x_dict = {
                nt: self.dropout(norm[nt](new_x[nt] + x_dict[nt]).relu())
                for nt in new_x
            }
        logits = {
            "shipment": self.head_delay(x_dict["shipment"]).squeeze(-1),
            "inventory": self.head_shortage(x_dict["inventory"]).squeeze(-1),
            "supplier": self.head_impact(x_dict["supplier"]).squeeze(-1),
        }
        return logits, x_dict
