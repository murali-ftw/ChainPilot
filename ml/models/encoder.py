"""
HGT / GraphSAGE / GAT / RGCN / RGCN+attn / RGCN+relemb / RGCN+battn encoders — Steps 3 and 5
(`docs/14_Model_Development_Roadmap.md` §6, §8; `docs/10_AI_ML_Documentation.md`
§8.1; `project_HADES.md` Part 3).

All architectures share the same input projection (`W_in` per node type into
a shared `d`-dim space, `project_HADES.md` §2.1/§7.1) and retain every
intermediate layer's output `[h^1 ... h^L]` rather than only the last, so
`ml/models/depth.py`'s structural depth-prior readout applies identically
regardless of which encoder is in use -- the encoder is the only thing that
differs between Step 5's ablation arms.

HGT uses `torch_geometric.nn.HGTConv` directly (it natively parameterizes
per-node-type and per-relation-type projections/attention, matching
`project_HADES.md` §3.3's math). GraphSAGE/GAT are type-blind by
construction, so they're wired up via `HeteroConv` wrapping one
`SAGEConv`/`GATConv` per relation -- the standard PyG pattern for a
heterogeneous baseline that still respects relation boundaries for message
routing, without any type- or relation-specific weight matrices. RGCN
(`ml/models/rgcn_encoder.py`) sits between those two extremes: every
relation gets its own parameters like HGT, but drawn from a small shared
basis pool (Schlichtkrull et al. 2018) rather than a fully dedicated
`hidden x hidden` matrix -- it needs a bespoke `nn.Module` over
`edge_index_dict` for the same reason `hgt_sparse` does (`RGCNConv`'s public
API expects a homogeneous single-feature-matrix graph, not this codebase's
per-relation `HeteroData` convention). RGCN+attn (`ml/models/
rgcn_attn_encoder.py`, "Option 1") is a fifth arm hybridizing RGCN's
transform with a shared (relation-agnostic) attention-weighted aggregation
in place of RGCN's per-relation-mean-then-sum scheme -- see that module's
docstring for the full design rationale. RGCN+relemb (`ml/models/
rgcn_relemb_encoder.py`, "Option 3") is a sixth arm, extending RGCN+attn's
joint-softmax attention with one extra additive term -- a small per-relation
embedding fed through one more shared scorer -- so the attention logit knows
which relation an edge arrived through, not just its message content and
destination; see that module's docstring for the full design rationale.
RGCN+battn (`ml/models/rgcn_battn_encoder.py`, "Option 2") is a seventh arm,
the most expensive of the three RGCN-attention hybrids: a SECOND
basis-decomposed pool dedicated to attention, giving every relation its own
per-relation attention MATRIX (bilinear scoring on the original embeddings)
rather than a shared vector scorer; see that module's docstring for the full
design rationale.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, HGTConv, Linear, SAGEConv

ARCHITECTURES = ("graphsage", "gat", "hgt", "rgcn", "rgcn_attn", "rgcn_relemb", "rgcn_battn",
                  "rgcn_attn_depthgate", "rgcn_attn_markov", "rgcn_attn_rung4", "rgcn_attn_rung5")

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
                   num_layers: int = 4, dropout: float = 0.2, num_bases: int = 8,
                   relation_embed_dim: int = 16, num_bases_attn: int = 8) -> nn.Module:
    """Factory: `architecture` in {'hgt', 'graphsage', 'gat', 'hgt_sparse',
    'rgcn', 'rgcn_attn', 'rgcn_relemb', 'rgcn_battn', 'rgcn_attn_depthgate'}
    (or the doc's full name 'heterogeneous_graph_transformer' for 'hgt').
    'hgt_sparse' -- Task 2's sparse-relation-merged variant, see
    `ml/models/sparse_hgt_encoder.py`. 'rgcn_attn' -- "Option 1" hybrid
    (SHARE), RGCN's basis-decomposition transform plus a shared
    attention-weighted aggregation, see `ml/models/rgcn_attn_encoder.py`.
    'rgcn_relemb' -- "Option 3" (SHARP), Option 1 plus a per-relation
    embedding additive term in the attention logit, see
    `ml/models/rgcn_relemb_encoder.py`. 'rgcn_battn' -- "Option 2" (SHARK),
    Option 1 with a SECOND basis-decomposed pool dedicated to attention
    (per-relation attention matrices, bilinear scoring) rather than a shared
    vector scorer, see `ml/models/rgcn_battn_encoder.py`.
    'rgcn_attn_depthgate' -- Step 6 pilot, SHARE's encoder with a learned,
    KL-anchored per-TASK depth gate (one shared distribution per task) in
    place of the fixed structural-prior readout; returns `num_layers + 1`
    layer-dicts (`h^0..h^L`), NOT `num_layers` like every other
    architecture here -- see `ml/models/rgcn_attn_depthgate_encoder.py`.
    'rgcn_attn_markov' -- Step 6 Phase 1, the SAME `h^0..h^L` encoder (this
    identifier maps to the identical `RGCNAttnDepthGateEncoder` class --
    it differs from 'rgcn_attn_depthgate' only in what's built on top: a
    fixed per-task index-select, zero new parameters), see
    `ml/models/rgcn_attn_markov_encoder.py`. 'rgcn_attn_rung4' /
    'rgcn_attn_rung5' -- Step 6 Phase 2, the two per-node depth-gate hybrids
    from "Markov Scoping and Transformer 1.md" Part 6's ladder (rung 4:
    attention-only; rung 5: prior-initialized MLP) -- SAME `h^0..h^L`
    encoder again; each node gets its OWN learned depth-weight distribution
    per task (the upgrade over 'rgcn_attn_depthgate''s one-per-task
    distribution), see `ml/models/rgcn_attn_rung4_encoder.py` /
    `ml/models/rgcn_attn_rung5_encoder.py`. `num_bases` applies to 'rgcn',
    'rgcn_attn', 'rgcn_relemb', 'rgcn_battn', 'rgcn_attn_depthgate',
    'rgcn_attn_markov', 'rgcn_attn_rung4', and 'rgcn_attn_rung5' (all built
    on the same basis-decomposition relation count for the message
    transform, `ml/models/rgcn_encoder.py`); `relation_embed_dim` applies
    only to 'rgcn_relemb'; `num_bases_attn` applies only to 'rgcn_battn'
    (its separate attention-side basis count); every other architecture
    ignores whichever of these doesn't apply to it."""
    architecture = _normalize_architecture(architecture)
    if architecture == "hgt":
        return HGTEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if architecture == "hgt_sparse":
        from ml.models.sparse_hgt_encoder import HGTSparseMergedEncoder
        return HGTSparseMergedEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if architecture == "rgcn":
        from ml.models.rgcn_encoder import RGCNEncoder
        return RGCNEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                            num_bases=num_bases, dropout=dropout)
    if architecture == "rgcn_attn":
        from ml.models.rgcn_attn_encoder import RGCNAttnEncoder
        return RGCNAttnEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                num_bases=num_bases, dropout=dropout)
    if architecture == "rgcn_relemb":
        from ml.models.rgcn_relemb_encoder import RGCNRelEmbAttnEncoder
        return RGCNRelEmbAttnEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                      num_bases=num_bases, relation_embed_dim=relation_embed_dim,
                                      dropout=dropout)
    if architecture == "rgcn_battn":
        from ml.models.rgcn_battn_encoder import RGCNBasisAttnEncoder
        return RGCNBasisAttnEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                     num_bases=num_bases, num_bases_attn=num_bases_attn,
                                     dropout=dropout)
    if architecture == "rgcn_attn_depthgate":
        # Step 6 pilot: identical to 'rgcn_attn' except forward() also
        # returns h^0 (the pre-message-passing input projection), giving
        # L+1 candidate depths instead of L -- see
        # `ml/models/rgcn_attn_depthgate_encoder.py`'s module docstring.
        # NOTE: unlike every other architecture here, this encoder's
        # forward() returns `num_layers + 1` layer-dicts, not `num_layers`
        # -- callers that assume `len(layers) == num_layers` (e.g. the
        # generic parametrized encoder test) do not apply to this one.
        from ml.models.rgcn_attn_depthgate_encoder import RGCNAttnDepthGateEncoder
        return RGCNAttnDepthGateEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                         num_bases=num_bases, dropout=dropout)
    if architecture == "rgcn_attn_markov":
        # Step 6 Phase 1 (Markov-blanket-derived fixed depth,
        # "Markov Scoping and Transformer 1.md" §7.1): same h^0..h^L
        # exposure as 'rgcn_attn_depthgate' -- deliberately the identical
        # encoder class, since the fixed-vs-learned readout choice lives in
        # the model built on top (`ml/models/rgcn_attn_markov_encoder.py`),
        # not in the encoder. Also returns `num_layers + 1` layer-dicts, not
        # `num_layers`, same exception as 'rgcn_attn_depthgate' above.
        from ml.models.rgcn_attn_depthgate_encoder import RGCNAttnDepthGateEncoder
        return RGCNAttnDepthGateEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                         num_bases=num_bases, dropout=dropout)
    if architecture in ("rgcn_attn_rung4", "rgcn_attn_rung5"):
        # Step 6 Phase 2 (per-node depth-gate hybrids, "Markov Scoping and
        # Transformer 1.md" Part 6's ladder, rungs 4 and 5): same h^0..h^L
        # exposure again -- the encoder is identical across
        # 'rgcn_attn_depthgate'/'rgcn_attn_markov'/'rgcn_attn_rung4'/
        # 'rgcn_attn_rung5'; all four differ only in what's built on top
        # (`ml/models/rgcn_attn_rung4_encoder.py`,
        # `ml/models/rgcn_attn_rung5_encoder.py`). Also returns
        # `num_layers + 1` layer-dicts, same exception as above.
        from ml.models.rgcn_attn_depthgate_encoder import RGCNAttnDepthGateEncoder
        return RGCNAttnDepthGateEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                         num_bases=num_bases, dropout=dropout)
    if architecture in ("graphsage", "gat"):
        conv_name = "gat" if architecture == "gat" else "sage"
        return HeteroGNNEncoder(metadata, in_dims, conv_name, hidden=hidden,
                                 num_layers=num_layers, dropout=dropout)
    raise ValueError(f"unknown architecture: {architecture!r}, expected one of {ARCHITECTURES}")
