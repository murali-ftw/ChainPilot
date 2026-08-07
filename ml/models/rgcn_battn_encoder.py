"""
RGCN + basis-decomposed attention ("Option 2") — a seventh architecture arm,
the third and most expensive of the RGCN-attention hybrids
(`ml/models/rgcn_attn_encoder.py`'s "Option 1", `ml/models/
rgcn_relemb_encoder.py`'s "Option 3" being the other two). Where Option 1
scores edges with a single shared vector-valued scorer and Option 3 adds a
small per-relation embedding on top of that same scorer, Option 2 gives
attention its OWN full per-relation MATRIX, basis-decomposed the same way
the message transform already is.

**What carries over from `RGCNAttnEncoder`, unchanged.** `lin_in`
(per-node-type input projection), `rel_basis`/`rel_coeff` (the
transformation-side shared basis pool and `W_r = sum_b a_r[b] * V_b`
computation, unchanged), `self_loops` (per-node-type, per-layer, not
attention-weighted), the `by_dst` relation grouping, and the same single
joint softmax per destination node, spanning every relation feeding that
node together (`torch_geometric.utils.softmax`). None of that changes here
-- Option 2 only replaces HOW the attention logit itself is computed.

**What's new: a second, separate basis pool dedicated to attention, scoring
bilinearly on the ORIGINAL embeddings.** Options 1 and 3 score edges with
an additive scorer over the already-transformed message (`att_msg(msg) +
att_dst(h[dst])`, optionally `+ rel_term`) -- a vector-valued attention
parameter. Option 2 instead gives every relation its own attention MATRIX
`A_r`, basis-decomposed exactly like the transform's own `W_r`:

    att_basis: [num_bases_attn, hidden, hidden]   -- shared pool, one set
                                                       for the whole encoder
    att_coeff: [num_relations, num_bases_attn]     -- per-relation coefficients
    A_r = sum_b att_coeff[r, b] * att_basis[b]      -- per relation, `[hidden, hidden]`

and scores each edge with a scaled bilinear form on the PRE-transform node
embeddings (not the message `m_ij = h_j @ W_r`, which still supplies the
value/content side of attention, computed from the transformation pool as
in `RGCNAttnEncoder`):

    e_ij = (h_i^T A_r h_j) / sqrt(hidden)

matching the scaled dot-product convention (`GATConv`'s own single-vector
scoring generalizes this; Transformer-style scaled dot-product attention is
the same idea with `A_r` fixed to a projection product rather than learned
directly). Every relation's `(msg, dst_idx, e)` triples are concatenated
into one pool per destination node type before the SAME joint softmax
Options 1/3 already use runs over them -- relation information enters
through which matrix `A_r` scored a given edge, not through a change to the
softmax's grouping or scope. Aggregation is unchanged:
`h_i' = self_loop(h_i) + sum_j alpha_ij * m_ij`.

**Why this is the most expensive of the three hybrids, and why bilinear
scoring is the natural fit for it.** A vector-valued scorer (Options 1/3)
can only ask "how salient is this message/destination pair," independent of
relation-specific structure in how source and destination should interact.
A full attention MATRIX per relation can express relation-specific
interaction patterns a vector scorer cannot (e.g. "for `SUPPLIES`, weight
component axes 3 and 7 of the destination against axis 12 of the source" --
not expressible as two independent linear scores summed together). That
expressiveness costs a second full `num_bases_attn x hidden x hidden` basis
pool on top of the transform's own -- roughly double the relational
parameter budget of plain `RGCNEncoder`, versus Option 1's two
`Linear(hidden, 1)` vectors per layer or Option 3's small per-relation
embedding table. Basis-decomposing the attention matrices, rather than
giving each relation a fully dedicated `A_r`, is what keeps this hybrid's
cost in the same family as the other RGCN variants at all -- a fully
dedicated per-relation attention matrix, un-decomposed, would cost as much
as HGT's own per-relation attention parameterization, defeating the entire
point of the RGCN line of experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter, softmax


class RGCNBasisAttnEncoder(nn.Module):
    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, num_bases_attn: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        num_relations = len(edge_types)
        self.rel_to_idx = {et: i for i, et in enumerate(edge_types)}
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.num_bases_attn = num_bases_attn

        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})

        # Transformation-side shared basis pool -- verbatim from RGCNEncoder
        # / RGCNAttnEncoder, see their module docstrings.
        self.rel_basis = nn.Parameter(torch.empty(num_bases, hidden, hidden))
        self.rel_coeff = nn.Parameter(torch.empty(num_relations, num_bases))
        nn.init.xavier_uniform_(self.rel_basis.view(num_bases, -1))
        nn.init.xavier_uniform_(self.rel_coeff)

        # NEW relative to RGCNAttnEncoder: a SECOND, separate basis pool
        # dedicated to attention -- per-relation attention MATRICES, not a
        # vector scorer.
        self.att_basis = nn.Parameter(torch.empty(num_bases_attn, hidden, hidden))
        self.att_coeff = nn.Parameter(torch.empty(num_relations, num_bases_attn))
        nn.init.xavier_uniform_(self.att_basis.view(num_bases_attn, -1))
        nn.init.xavier_uniform_(self.att_coeff)

        # Per-node-type self-loop, one full matrix per node type PER LAYER --
        # verbatim from RGCNEncoder/RGCNAttnEncoder, not attention-weighted.
        self.self_loops = nn.ModuleList(
            [nn.ModuleDict({nt: Linear(hidden, hidden) for nt in node_types}) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}

        by_dst: dict[str, list] = {}
        for et in edge_index_dict:
            by_dst.setdefault(et[2], []).append(et)

        layers = []
        for layer_idx in range(self.num_layers):
            self_loop = self.self_loops[layer_idx]
            h_new = {nt: self_loop[nt](h[nt]) for nt in h}

            for dst_type, rel_list in by_dst.items():
                msgs, dsts, logits = [], [], []
                for et in rel_list:
                    edge_index = edge_index_dict[et]
                    if edge_index.numel() == 0:
                        continue
                    src_type = et[0]
                    rel_idx = self.rel_to_idx[et]
                    w_r = torch.einsum("b,bij->ij", self.rel_coeff[rel_idx], self.rel_basis)
                    a_r = torch.einsum("b,bij->ij", self.att_coeff[rel_idx], self.att_basis)
                    src_idx, dst_idx = edge_index[0], edge_index[1]

                    # Value/content side of attention: the same transformed
                    # message RGCNEncoder/RGCNAttnEncoder use.
                    msg = h[src_type][src_idx] @ w_r

                    # Attention logit: scaled BILINEAR form on the ORIGINAL
                    # (pre-transform) embeddings, using the attention-side
                    # basis pool -- e_ij = (h_dst^T A_r h_src) / sqrt(hidden).
                    h_dst_edges = h[dst_type][dst_idx]
                    h_src_edges = h[src_type][src_idx]
                    logit = (h_dst_edges @ a_r * h_src_edges).sum(-1) / (self.hidden ** 0.5)

                    msgs.append(msg)
                    dsts.append(dst_idx)
                    logits.append(logit)

                if not msgs:
                    # Same fallback as RGCNEncoder/RGCNAttnEncoder: no relation
                    # fed dst_type this layer, h_new keeps the self-loop value.
                    continue

                msg_all = torch.cat(msgs, dim=0)
                dst_all = torch.cat(dsts, dim=0)
                logit_all = torch.cat(logits, dim=0)

                # Same single joint softmax as RGCNAttnEncoder/
                # RGCNRelEmbAttnEncoder -- relation identity enters through
                # which A_r scored a given edge, not through a change to
                # this normalization scope.
                alpha = softmax(logit_all, dst_all, num_nodes=h[dst_type].size(0))
                weighted = msg_all * alpha.unsqueeze(-1)
                agg = scatter(weighted, dst_all, dim=0, dim_size=h[dst_type].size(0), reduce="sum")
                h_new[dst_type] = h_new[dst_type] + agg

            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
