"""
Encoder factory — the nine architectures this benchmark compares.

Ported from `HADES_v1/ml/models/encoder.py`. The V1 file also carried the Step-6
and Step-7 research arms (depth gates, rungs, Transformer-2 fusion variants);
those are out of scope here and are not carried over, so this factory exposes
exactly the baseline set plus SHARE/SHARP/SHARK:

    graphsage, gat, hgt, rgcn, gps          -- baselines
    rgcn_attn    (SHARE)                    -- V1's production architecture
    rgcn_relemb  (SHARP)
    rgcn_battn   (SHARK)

`rgcn`, `rgcn_attn`, `rgcn_relemb` and `rgcn_battn` are byte-identical copies of
V1's modules — the port is meant to validate, not to redesign. Every encoder
takes the same per-node-type input projection into a shared `d`-dim space and
returns EVERY layer's output `[h^1 ... h^L]`, so `ml/models/depth.py`'s
structural-prior readout applies unchanged.

**GraphGPS is new in V2 and is an adaptation, not a port.** V1 never ran it. PyG's
`GPSConv` pairs a local MPNN with *dense* global attention over all nodes, which
is O(n^2) in the node count; V2's spec-scale snapshots carry ~780k nodes, so dense
global attention is not merely slow but impossible to allocate. `GPSEncoder` below
keeps GraphGPS's actual structure — local message passing plus a global,
graph-wide attention term, summed, then a feed-forward block — with the global
term computed by **linear (kernelised) attention within each node type**, which is
O(n·d^2) and preserves the "every node can see every same-type node" property that
distinguishes GPS from a pure MPNN. This deviation is stated in
`reports/phase7_training_results.md`; a GraphGPS result here is a result for that
adaptation, not for the dense-attention original.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, HGTConv, Linear, SAGEConv

ARCHITECTURES = ("gcn", "graphsage", "gat", "hgt", "rgcn", "gps",
                 "rgcn_attn", "rgcn_relemb", "rgcn_battn",
                 # Phase 7c: the depth-selection family, ported from V1's Step 6 Rounds 3-5.
                 # All four share ONE encoder class -- `RGCNAttnDepthGateEncoder`, which is
                 # `RGCNAttnEncoder` with h^0 (the raw lin_in projection) prepended, so the
                 # readout has L+1 = 5 candidate depths instead of L. They differ only in what
                 # is built on top: a fixed index-select (markov), a per-node MLP gate (rung5),
                 # that gate with a tanh-bounded residual (rung5_a), or with structural features
                 # concatenated onto its input (rung5_b). `rung5_c` is not an architecture at
                 # all -- it is `rung5` plus a freeze schedule in ml/train.py.
                 "rgcn_attn_markov", "rgcn_attn_rung5", "rgcn_attn_rung5_a",
                 "rgcn_attn_rung5_b", "rgcn_attn_rung5_c", "rgcn_attn_rung5_ac",
                 # Layer 3: Transformer 2 and its three fusion designs. All four use the
                 # SAME h^0..h^L encoder as the Rung 5 family -- Transformer 2 is purely
                 # additive on top of Variant A's readout, changing nothing below it.
                 "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
                 "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn")

# The three named architectures, mapped to their code identifiers, so reports can
# print the name the project uses without the caller re-deriving it.
ARCHITECTURE_NAMES = {
    "rgcn_attn": "SHARE", "rgcn_relemb": "SHARP", "rgcn_battn": "SHARK",
    "gcn": "GCN", "graphsage": "GraphSAGE", "gat": "GAT", "hgt": "HGT", "rgcn": "RGCN",
    "gps": "GraphGPS",
    "rgcn_attn_markov": "Markov", "rgcn_attn_rung5": "Rung5",
    "rgcn_attn_rung5_a": "Rung5-A", "rgcn_attn_rung5_b": "Rung5-B",
    "rgcn_attn_rung5_c": "Rung5-C", "rgcn_attn_rung5_ac": "Rung5-A+C",
    "rgcn_attn_variant_a_transformer2": "T2-plain", "rgcn_attn_t2_confidence": "T2-conf",
    "rgcn_attn_t2_trustgate": "T2-trust", "rgcn_attn_t2_crossattn": "T2-cross",
}

_ARCHITECTURE_ALIASES = {"heterogeneous_graph_transformer": "hgt", "graphgps": "gps"}


def _normalize_architecture(architecture: str) -> str:
    return _ARCHITECTURE_ALIASES.get(architecture, architecture)


class HGTEncoder(nn.Module):
    """`L` stacked `HGTConv` layers over a shared `d`-dim space, heads=4."""

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

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HeteroGNNEncoder(nn.Module):
    """`L` stacked `HeteroConv` layers, each wrapping one type-blind
    `SAGEConv`/`GATConv` per relation. Node types with no incoming edges in a
    given layer keep their previous-layer value."""

    def __init__(self, metadata, in_dims: dict[str, int], conv_name: str, hidden: int = 64,
                 num_layers: int = 4, dropout: float = 0.2, heads: int = 1):
        super().__init__()
        _, edge_types = metadata
        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})

        def make_conv():
            if conv_name == "gat":
                # Single-head, matching V1: multi-head GAT would invert the
                # parameter ordering the matched-d arm is derived from.
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

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GCNRelConv(nn.Module):
    """One relation's GCN propagation, written out rather than wrapped around
    `torch_geometric.nn.GCNConv`.

    `GCNConv` assumes a single homogeneous adjacency: it refuses bipartite input
    (`x` as a `(src, dst)` tuple), and its symmetric normalisation and implicit
    self-loops both assume a square adjacency. Every relation in this graph is
    bipartite between two different node types, so the two defining properties of
    GCN — a single shared weight per relation and **symmetric degree
    normalisation** `d_dst^-1/2 * d_src^-1/2` — are applied directly here. Self
    loops are not added inside the relation (there is no "self" across two node
    types); the encoder adds a per-node-type self term once per layer, outside
    the relation loop, which is the bipartite analogue of GCN folding self-loops
    into the neighbourhood.

    This is what distinguishes the GCN arm from the GraphSAGE arm, which uses
    mean aggregation with a separate root weight and no degree normalisation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x_src: torch.Tensor, edge_index: torch.Tensor, num_dst: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = self.lin(x_src)
        deg_src = torch.zeros(x_src.size(0), device=x_src.device).index_add_(
            0, src, torch.ones_like(src, dtype=msg.dtype))
        deg_dst = torch.zeros(num_dst, device=x_src.device).index_add_(
            0, dst, torch.ones_like(dst, dtype=msg.dtype))
        norm = deg_src.clamp(min=1).pow(-0.5)[src] * deg_dst.clamp(min=1).pow(-0.5)[dst]
        out = torch.zeros(num_dst, msg.size(-1), device=msg.device, dtype=msg.dtype)
        return out.index_add_(0, dst, msg[src] * norm.unsqueeze(-1))


class GCNEncoder(nn.Module):
    """`L` layers of per-relation GCN propagation summed across relations, plus
    one per-node-type root transform per layer. Type-blind by construction: the
    relation gets its own weight only because src and dst dimensions must line
    up, exactly as the GraphSAGE/GAT baselines do through `HeteroConv`."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = list(edge_types)
        self.num_layers = num_layers
        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.convs = nn.ModuleList([
            nn.ModuleDict({"__".join(et): GCNRelConv(hidden, hidden) for et in edge_types})
            for _ in range(num_layers)])
        self.root = nn.ModuleList([
            nn.ModuleDict({nt: nn.Linear(hidden, hidden) for nt in node_types})
            for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        layers = []
        for i in range(self.num_layers):
            h_new = {nt: self.root[i][nt](h[nt]) for nt in h}
            for et, edge_index in edge_index_dict.items():
                if edge_index.numel() == 0:
                    continue
                src_type, _, dst_type = et
                h_new[dst_type] = h_new[dst_type] + self.convs[i]["__".join(et)](
                    h[src_type], edge_index, h[dst_type].size(0))
            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GPSEncoder(nn.Module):
    """GraphGPS (Rampasek et al. 2022), adapted to heterogeneous graphs at this
    scale — see the module docstring for why the global term is linear attention
    rather than dense softmax attention.

    Per layer, per node type:  `h <- FFN( local(h) + global(h) )`, where `local`
    is one `SAGEConv` per relation summed over relations (the same type-blind
    local MPNN the GraphSAGE baseline uses, so the GPS-vs-GraphSAGE contrast
    isolates the global term) and `global` is
    `phi(Q) @ (phi(K)^T V) / (phi(Q) @ sum phi(K))` with `phi = elu + 1`, the
    standard linear-attention kernel, computed over all nodes of that type."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, dropout: float = 0.2, heads: int = 4):
        super().__init__()
        node_types, edge_types = metadata
        if hidden % heads:
            raise ValueError(f"gps hidden={hidden} must be divisible by heads={heads}")
        self.hidden, self.heads, self.num_layers = hidden, heads, num_layers
        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.local = nn.ModuleList(
            [HeteroConv({et: SAGEConv(hidden, hidden) for et in edge_types}, aggr="sum")
             for _ in range(num_layers)])
        self.qkv = nn.ModuleList(
            [nn.ModuleDict({nt: nn.Linear(hidden, 3 * hidden) for nt in node_types})
             for _ in range(num_layers)])
        self.attn_out = nn.ModuleList(
            [nn.ModuleDict({nt: nn.Linear(hidden, hidden) for nt in node_types})
             for _ in range(num_layers)])
        self.ffn = nn.ModuleList(
            [nn.ModuleDict({nt: nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                                              nn.Linear(2 * hidden, hidden)) for nt in node_types})
             for _ in range(num_layers)])
        self.norm = nn.ModuleList(
            [nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types}) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def _global(self, h: torch.Tensor, qkv: nn.Linear, out: nn.Linear) -> torch.Tensor:
        n, d = h.shape
        heads = self.heads
        dh = d // heads
        q, k, v = qkv(h).view(n, 3, heads, dh).permute(1, 0, 2, 3)
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0
        kv = torch.einsum("nhd,nhe->hde", k, v)              # (heads, dh, dh)
        z = 1.0 / (torch.einsum("nhd,hd->nh", q, k.sum(0)) + 1e-6)
        ctx = torch.einsum("nhd,hde,nh->nhe", q, kv, z)
        return out(ctx.reshape(n, d))

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        layers = []
        for i in range(self.num_layers):
            local = self.local[i](h, edge_index_dict)
            h_new = {}
            for nt in h:
                mix = local.get(nt, torch.zeros_like(h[nt])) + \
                    self._global(h[nt], self.qkv[i][nt], self.attn_out[i][nt])
                z = self.norm[i][nt](h[nt] + self.dropout(mix))
                h_new[nt] = self.dropout(torch.relu(z + self.ffn[i][nt](z)))
            h = h_new
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_encoder(architecture: str, metadata, in_dims: dict[str, int], hidden: int = 64,
                  num_layers: int = 4, dropout: float = 0.2, num_bases: int = 8,
                  relation_embed_dim: int = 16, num_bases_attn: int = 8) -> nn.Module:
    """Factory over `ARCHITECTURES`. `num_bases` applies to the RGCN family;
    `relation_embed_dim` only to SHARP; `num_bases_attn` only to SHARK."""
    architecture = _normalize_architecture(architecture)
    if architecture == "hgt":
        return HGTEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if architecture == "gcn":
        return GCNEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
    if architecture == "gps":
        return GPSEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers, dropout=dropout)
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
    if architecture in ("rgcn_attn_markov", "rgcn_attn_rung5", "rgcn_attn_rung5_a",
                        "rgcn_attn_rung5_b", "rgcn_attn_rung5_c", "rgcn_attn_rung5_ac",
                 # Layer 3: Transformer 2 and its three fusion designs. All four use the
                 # SAME h^0..h^L encoder as the Rung 5 family -- Transformer 2 is purely
                 # additive on top of Variant A's readout, changing nothing below it.
                 "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
                 "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn"):
        # One encoder for all five: h^0..h^L exposure. NOTE this returns `num_layers + 1`
        # layer-dicts, not `num_layers` -- the depth-selection models index into that list,
        # every other architecture here does not.
        from ml.models.rgcn_attn_depthgate_encoder import RGCNAttnDepthGateEncoder
        return RGCNAttnDepthGateEncoder(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                        num_bases=num_bases, dropout=dropout)
    if architecture in ("graphsage", "gat"):
        conv_name = "gat" if architecture == "gat" else "sage"
        return HeteroGNNEncoder(metadata, in_dims, conv_name, hidden=hidden,
                                num_layers=num_layers, dropout=dropout)
    raise ValueError(f"unknown architecture: {architecture!r}, expected one of {ARCHITECTURES}")
