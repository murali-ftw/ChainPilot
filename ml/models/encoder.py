"""
HGT / GraphSAGE / GAT encoders — Steps 3 and 5
(`docs/14_Model_Development_Roadmap.md` §6, §8; `docs/10_AI_ML_Documentation.md`
§8.1; `project_HADES.md` Part 3).

All three architectures share the same input projection (`W_in` per node
type into a shared `d`-dim space, `project_HADES.md` §2.1/§7.1) and retain
every intermediate layer's output `[h^1 ... h^L]` rather than only the
last, so `ml/models/depth.py`'s structural depth-prior readout applies
identically regardless of which encoder is in use -- the encoder is the
only thing that differs between Step 5's three ablation arms.

HGT uses `torch_geometric.nn.HGTConv` directly (it natively parameterizes
per-node-type and per-relation-type projections/attention, matching
`project_HADES.md` §3.3's math). GraphSAGE/GAT are type-blind by
construction, so they're wired up via `HeteroConv` wrapping one
`SAGEConv`/`GATConv` per relation -- the standard PyG pattern for a
heterogeneous baseline that still respects relation boundaries for message
routing, without any type- or relation-specific weight matrices.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, HGTConv, Linear, SAGEConv

ARCHITECTURES = ("graphsage", "gat", "hgt")

# docs/05_Database_Design.md §6.21's example `architecture` column values use
# the full name; this module's factory keys stay short. Both are accepted.
_ARCHITECTURE_ALIASES = {"heterogeneous_graph_transformer": "hgt"}


def _normalize_architecture(architecture: str) -> str:
    return _ARCHITECTURE_ALIASES.get(architecture, architecture)


class HGTEncoder(nn.Module):
    """`L` stacked `HGTConv` layers over a shared `d`-dim space, heads=4
    (`project_HADES.md` §3.3-3.5). Returns every layer's per-node-type
    output, `h^1 ... h^L`."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 heads: int = 4, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.convs = nn.ModuleList(
            [HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        layers = []
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers


class HeteroGNNEncoder(nn.Module):
    """
    `L` stacked `HeteroConv` layers, each wrapping one type-blind
    `SAGEConv`/`GATConv` per relation. Node types with no incoming edges in
    a given layer keep their previous-layer value (`HeteroConv` only
    returns types it updated) so depth-prior readout never sees a stale
    all-zero vector for those types.
    """

    def __init__(self, metadata, in_dims: dict[str, int], conv_name: str, hidden: int = 64,
                 num_layers: int = 4, dropout: float = 0.2, heads: int = 1):
        super().__init__()
        _, edge_types = metadata
        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})

        def make_conv():
            if conv_name == "gat":
                # Single-head: project_HADES.md §8.3 derives the matched-parameter
                # arm assuming GAT has *fewer* per-relation parameters than
                # GraphSAGE at equal d (single attention vector vs SAGE's two
                # full d x d matrices) -- multi-head GAT would invert that
                # ordering and break the documented d=116 matched target.
                return GATConv(hidden, hidden, heads=heads, concat=False, add_self_loops=False)
            return SAGEConv(hidden, hidden)

        self.convs = nn.ModuleList(
            [HeteroConv({et: make_conv() for et in edge_types}, aggr="sum") for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        layers = []
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers


def build_encoder(architecture: str, metadata, in_dims: dict[str, int], hidden: int = 64,
                   num_layers: int = 4, dropout: float = 0.2) -> nn.Module:
    """Factory: `architecture` in {'hgt', 'graphsage', 'gat', 'hgt_sparse'}
    (or the doc's full name 'heterogeneous_graph_transformer' for 'hgt').
    'hgt_sparse' -- Task 2's sparse-relation-merged variant, see
    `ml/models/sparse_hgt_encoder.py`."""
    architecture = _normalize_architecture(architecture)
    if architecture == "hgt":
        return HGTEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if architecture == "hgt_sparse":
        from ml.models.sparse_hgt_encoder import HGTSparseMergedEncoder
        return HGTSparseMergedEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if architecture in ("graphsage", "gat"):
        conv_name = "gat" if architecture == "gat" else "sage"
        return HeteroGNNEncoder(metadata, in_dims, conv_name, hidden=hidden,
                                 num_layers=num_layers, dropout=dropout)
    raise ValueError(f"unknown architecture: {architecture!r}, expected one of {ARCHITECTURES}")
