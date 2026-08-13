"""
RGCN encoder — Task 4 of the post-v3 follow-up, a fourth architecture arm
alongside HGT/GraphSAGE/GAT (`ml/models/encoder.py`).

Basis-decomposition R-GCN (Schlichtkrull et al. 2018), adapted from its usual
homogeneous-graph form (`torch_geometric.nn.RGCNConv`'s single feature matrix
+ integer edge-type tensor) to this codebase's per-relation
`edge_index_dict`/`HeteroData` convention -- `RGCNConv` itself is NOT used
here, for the same reason `ml/models/sparse_hgt_encoder.py` writes a bespoke
module rather than reshaping `HGTConv`'s API: the public conv's expected
input shape doesn't match what `ml/graph/builder.py` produces.

**The bet this architecture is testing.** HGT (`HGTConv`) gives every one of
the 20 meta-relations its own full `hidden x hidden` attention/message
parameter set. RGCN's basis decomposition instead expresses every relation's
transform as a linear combination of a small shared pool of `num_bases`
basis matrices (`W_r = sum_b a_r[b] * V_b`) -- relation-specific cost is just
a `num_bases`-length coefficient vector, not a full matrix. If HGT's
per-relation parameterization is overfitting on the thinnest relations
(the same hypothesis Task 2's sparse-merged-HGT encoder tested by literally
sharing one `SAGEConv` across the 4 thinnest relations), RGCN's much cheaper
but *structured* form of sharing -- every relation still gets its own
coefficients, just drawn from a shared basis -- is a second, more principled
way to probe the same question at a matched parameter budget.

**Design, mapped onto `edge_index_dict`:**
- `rel_basis`: one `nn.Parameter` of shape `[num_bases, hidden, hidden]` --
  the shared pool of basis matrices `V_b`. A SINGLE tensor for the whole
  encoder (not one per layer): every layer's relation transform draws from
  the same basis pool, so parameter cost for the relational part does not
  grow with `num_layers`.
- `rel_coeff`: one `nn.Parameter` of shape `[num_relations, num_bases]` --
  per-relation coefficients `a_r`, indexed by that relation's position in
  `metadata`'s `edge_types` list. Also a single tensor, shared across layers.
- Per layer, per relation `r`: `W_r = sum_b a_r[b] * V_b` (`[hidden,
  hidden]`), computed fresh each layer from the (layer-shared) basis +
  coefficients. Source-node embeddings for `r` are transformed by `W_r`,
  then mean-aggregated into destination nodes via
  `torch_geometric.utils.scatter`, then summed across every relation that
  shares the same destination node type -- the basis-decomposition analogue
  of `HeteroConv(aggr="sum")`'s per-relation-then-cross-relation reduction.
- `self_loops`: UNLIKE the relational part, one full `hidden x hidden`
  `Linear` per node type PER LAYER (`nn.ModuleList` of length `num_layers`,
  each a `nn.ModuleDict` keyed by node type) -- not basis-shared. This plays
  the identity-carrying role HGT's residual connection and GraphSAGE's
  self+neighbor combine play; it's applied to every node type every layer
  regardless of whether any relation lands on that type this layer, so
  (unlike `HeteroGNNEncoder`'s `HeteroConv`-wrapped convs, which only return
  types they actually touched) `h_new` always has a real, non-stale entry
  for every node type by construction. The `... if nt in h_new else h[nt]`
  fallback below is kept anyway, verbatim from `HeteroGNNEncoder`, as a
  defensive match to the rest of the encoder family rather than a path that
  should ever actually trigger here.
- `lin_in`: the same per-node-type input projection as every other encoder,
  once (not per layer).

Returns every layer's per-node-type dict (`h^1 ... h^L`), matching the
`HGTEncoder`/`HeteroGNNEncoder` contract exactly, so `ml/models/depth.py`'s
depth-prior readout and `ml/models/heads.py` need zero changes to accept
this as a fourth `architecture`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter


class RGCNEncoder(nn.Module):
    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        num_relations = len(edge_types)
        self.rel_to_idx = {et: i for i, et in enumerate(edge_types)}
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_bases = num_bases

        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})

        # Shared basis pool + per-relation coefficients -- ONE set for the
        # whole encoder, reused identically at every layer (see module
        # docstring: this is what keeps the relational parameter cost from
        # scaling with num_layers).
        self.rel_basis = nn.Parameter(torch.empty(num_bases, hidden, hidden))
        self.rel_coeff = nn.Parameter(torch.empty(num_relations, num_bases))
        nn.init.xavier_uniform_(self.rel_basis.view(num_bases, -1))
        nn.init.xavier_uniform_(self.rel_coeff)

        # Per-node-type self-loop, NOT basis-shared, one full matrix per
        # node type PER LAYER.
        self.self_loops = nn.ModuleList(
            [nn.ModuleDict({nt: Linear(hidden, hidden) for nt in node_types}) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        layers = []
        for layer_idx in range(self.num_layers):
            self_loop = self.self_loops[layer_idx]
            h_new = {nt: self_loop[nt](h[nt]) for nt in h}

            for et, edge_index in edge_index_dict.items():
                src_type, _rel, dst_type = et
                rel_idx = self.rel_to_idx[et]
                w_r = torch.einsum("b,bij->ij", self.rel_coeff[rel_idx], self.rel_basis)
                src_idx, dst_idx = edge_index[0], edge_index[1]
                msg = h[src_type][src_idx] @ w_r
                agg = scatter(msg, dst_idx, dim=0, dim_size=h[dst_type].size(0), reduce="mean")
                h_new[dst_type] = h_new[dst_type] + agg

            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
