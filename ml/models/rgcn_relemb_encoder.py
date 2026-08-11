"""
RGCN + shared attention + relation embeddings ("Option 3") — a sixth
architecture arm, extending `ml/models/rgcn_attn_encoder.py`'s ("Option 1")
joint-softmax-across-relations attention with one extra additive term so the
attention scorer knows which relation a given edge came through, not just
what its message content and destination look like.

**What carries over from `RGCNAttnEncoder`, unchanged.** Everything: `lin_in`
(per-node-type input projection), the shared `rel_basis`/`rel_coeff` basis
pool and the `W_r = sum_b a_r[b] * V_b` message transform, `self_loops`
(per-node-type, per-layer, not attention-weighted), `att_msg`/`att_dst`
(shared attention scorers), the `by_dst` relation grouping, and -- most
importantly -- the SAME single joint softmax per destination node, spanning
every relation feeding that node together
(`torch_geometric.utils.softmax`). Read `rgcn_attn_encoder.py`'s own
docstring for the full rationale behind all of that; none of it changes
here. Option 3 only adds one more additive term to the attention logit
computed inside that same loop -- it does not introduce a second
normalization scope or otherwise touch how attention weights are computed
once the logits exist.

**What's new: a relation identity signal in the logit.** `RGCNAttnEncoder`'s
attention score depends only on the message content (`att_msg(msg)`) and
the destination node's own state (`att_dst(h[dst])`) -- nothing in that
score tells the model *which relation* an edge arrived through. Two edges
with coincidentally similar message vectors from different relations would
score identically. Option 3 adds a small per-relation embedding
(`rel_embed`, `[num_relations, relation_embed_dim]`) fed through one more
shared scorer (`att_rel`, `Linear(relation_embed_dim, 1)` per layer) to
produce a per-relation scalar term, added into every edge of that relation's
logit as a constant offset:

    rel_term = att_rel[layer](rel_embed[rel_idx])          # (1,) -- once per relation, not per edge
    logit = LeakyReLU(att_msg(msg) + att_dst(h[dst]) + rel_term)

`rel_term` is computed once per relation per layer (it depends on neither
`msg` nor `h[dst]`) and broadcast across that relation's edges before the
SAME joint softmax already used by Option 1 runs over the concatenated pool
-- relation information enters through this extra additive term, not
through any change to the softmax's grouping or scope. `att_msg`, `att_dst`,
and `att_rel` are all shared across every relation and node type (exactly
one instance of each per layer); only `rel_embed` itself varies by
relation, and it is a small vector (`relation_embed_dim`, default 16), not a
matrix.

**Why this is the cheapest of the two relation-aware attention hybrids
worth trying.** Option 2 (basis-decomposed attention -- not built in this
round) would give attention its own second basis-decomposed weight pool,
costing `O(num_bases_attn x hidden^2)` on top of Option 1 -- the same order
of magnitude as the message-transform basis pool itself. Option 3 instead
represents "which relation is this" as a lookup into a small embedding
table plus one more `Linear(relation_embed_dim, 1)` scorer --
`O(num_relations x relation_embed_dim)` for the embeddings themselves
(rows, not matrices) plus `O(relation_embed_dim)` per layer for `att_rel`'s
own weights. At this codebase's 20 meta-relations and `relation_embed_dim`
`= 16`, that's a few hundred parameters, not tens of thousands -- a
relation-identity signal at a small fraction of what a second basis pool
would cost.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter, softmax


class RGCNRelEmbAttnEncoder(nn.Module):
    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, relation_embed_dim: int = 16,
                 dropout: float = 0.2, negative_slope: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        num_relations = len(edge_types)
        self.rel_to_idx = {et: i for i, et in enumerate(edge_types)}
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.relation_embed_dim = relation_embed_dim

        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})

        # Shared basis pool + per-relation coefficients -- verbatim from
        # RGCNEncoder/RGCNAttnEncoder, see module docstring.
        self.rel_basis = nn.Parameter(torch.empty(num_bases, hidden, hidden))
        self.rel_coeff = nn.Parameter(torch.empty(num_relations, num_bases))
        nn.init.xavier_uniform_(self.rel_basis.view(num_bases, -1))
        nn.init.xavier_uniform_(self.rel_coeff)

        # Per-node-type self-loop, one full matrix per node type PER LAYER --
        # verbatim from RGCNEncoder/RGCNAttnEncoder, not attention-weighted.
        self.self_loops = nn.ModuleList(
            [nn.ModuleDict({nt: Linear(hidden, hidden) for nt in node_types}) for _ in range(num_layers)]
        )

        # Shared attention scorers, verbatim from RGCNAttnEncoder.
        self.att_msg = nn.ModuleList([Linear(hidden, 1) for _ in range(num_layers)])
        self.att_dst = nn.ModuleList([Linear(hidden, 1) for _ in range(num_layers)])
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # NEW relative to RGCNAttnEncoder: a small per-relation embedding
        # table (rows, not matrices) plus one more shared scorer per layer.
        self.rel_embed = nn.Parameter(torch.empty(num_relations, relation_embed_dim))
        nn.init.xavier_uniform_(self.rel_embed)
        self.att_rel = nn.ModuleList([Linear(relation_embed_dim, 1) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}

        by_dst: dict[str, list] = {}
        for et in edge_index_dict:
            by_dst.setdefault(et[2], []).append(et)

        layers = []
        for layer_idx in range(self.num_layers):
            self_loop = self.self_loops[layer_idx]
            att_msg, att_dst, att_rel = self.att_msg[layer_idx], self.att_dst[layer_idx], self.att_rel[layer_idx]
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
                    src_idx, dst_idx = edge_index[0], edge_index[1]
                    msg = h[src_type][src_idx] @ w_r

                    # Computed once per relation, not per edge -- depends on
                    # neither msg nor h[dst], only on this relation's identity.
                    rel_term = att_rel(self.rel_embed[rel_idx].unsqueeze(0)).squeeze(-1)  # (1,)
                    logit = self.leaky_relu(
                        att_msg(msg).squeeze(-1) + att_dst(h[dst_type][dst_idx]).squeeze(-1)
                        + rel_term.expand(msg.size(0))
                    )
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

                # Same single joint softmax as RGCNAttnEncoder -- relation
                # identity entered through the logit's extra additive term
                # above, not through a change to this normalization scope.
                alpha = softmax(logit_all, dst_all, num_nodes=h[dst_type].size(0))
                weighted = msg_all * alpha.unsqueeze(-1)
                agg = scatter(weighted, dst_all, dim=0, dim_size=h[dst_type].size(0), reduce="sum")
                h_new[dst_type] = h_new[dst_type] + agg

            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
