"""
RGCN + shared attention ("Option 1") — a fifth architecture arm, hybridizing
`ml/models/rgcn_encoder.py`'s basis-decomposition message transform with
HGT-style attention over the aggregation step.

**What carries over from `RGCNEncoder`, unchanged.** The relation-specific
message *transform* is identical: `lin_in` (per-node-type input
projection), `rel_basis`/`rel_coeff` (one shared basis pool + one
per-relation coefficient vector for the WHOLE encoder, reused identically
at every layer -- relational parameter cost still doesn't scale with
`num_layers`), the `W_r = sum_b a_r[b] * V_b` computation itself, and
`self_loops` (one full `hidden x hidden` `Linear` per node type PER LAYER,
NOT basis-shared -- still the thing that carries identity across layers,
untouched and not attention-weighted). Option 1 only changes what happens
*after* messages are transformed: how they get combined into each
destination node.

**What's different: joint softmax over ALL incoming edges, not
per-relation-mean-then-sum.** `RGCNEncoder` aggregates each relation
separately (`scatter(..., reduce="mean")`) and then sums the per-relation
results -- every relation gets an equal vote regardless of how informative
its edges are for a given node this layer. Option 1 instead scores every
incoming edge to a destination node -- across every relation feeding that
node, mixed together -- with one shared attention scorer, then takes a
SINGLE softmax over that whole pool per destination node
(`torch_geometric.utils.softmax`, grouped by destination index). This is
what lets the model single out the one or two most relevant incoming
neighbors regardless of which relation they arrived through, matching
`HGTConv`'s per-node softmax scope -- but at a fraction of HGT's parameter
cost, since the scorer itself is shared.

**Why "shared" (not per-relation, not per-node-type) attention.** This is
deliberately the cheapest of the three hybrid designs worth trying: the
attention scorer (`att_msg`, `att_dst` -- two `Linear(hidden, 1)` layers
PER LAYER, nothing else) is the ONLY new learnable parameter relative to
`RGCNEncoder`, shared across every relation type and every node type. A
per-relation or per-node-type scorer would reintroduce exactly the kind of
dedicated-per-relation parameterization the whole RGCN line of experiments
(`ml/models/rgcn_encoder.py`'s module docstring, and Task 2's
`ml/models/sparse_hgt_encoder.py` before it) is testing whether HGT
over-relies on for the four thinnest relations -- a shared scorer keeps
that overfitting risk at its lowest point while still letting the model
express "not every incoming edge deserves equal weight."

**Attention score, mechanically (`GATConv`-style additive scoring, scoped
per destination node across every relation together, not per relation):**
for edge `(src_type, rel, dst_type)`, message `msg = h[src_type][src_idx] @
W_r` (identical transform to `RGCNEncoder`), then
`logit = LeakyReLU(att_msg(msg) + att_dst(h[dst_type][dst_idx]))`, using the
SAME `att_msg`/`att_dst` for every relation contributing to that layer.
Every relation's `(msg, dst_idx, logit)` triples are concatenated into one
pool per destination node type before the softmax runs, so the resulting
per-edge weights sum to 1 across a node's ENTIRE incoming edge set --
across relations, not within each one separately.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter, softmax


class RGCNAttnEncoder(nn.Module):
    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2,
                 negative_slope: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        num_relations = len(edge_types)
        self.rel_to_idx = {et: i for i, et in enumerate(edge_types)}
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_bases = num_bases

        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})

        # Shared basis pool + per-relation coefficients -- verbatim from
        # RGCNEncoder, see module docstring.
        self.rel_basis = nn.Parameter(torch.empty(num_bases, hidden, hidden))
        self.rel_coeff = nn.Parameter(torch.empty(num_relations, num_bases))
        nn.init.xavier_uniform_(self.rel_basis.view(num_bases, -1))
        nn.init.xavier_uniform_(self.rel_coeff)

        # Per-node-type self-loop, one full matrix per node type PER LAYER --
        # verbatim from RGCNEncoder, not attention-weighted.
        self.self_loops = nn.ModuleList(
            [nn.ModuleDict({nt: Linear(hidden, hidden) for nt in node_types}) for _ in range(num_layers)]
        )

        # The ONLY new learnable parameters relative to RGCNEncoder: one
        # shared attention scorer per layer -- not per relation, not per
        # node type.
        self.att_msg = nn.ModuleList([Linear(hidden, 1) for _ in range(num_layers)])
        self.att_dst = nn.ModuleList([Linear(hidden, 1) for _ in range(num_layers)])
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}

        # Precomputed once, not per layer: every relation type grouped by its
        # destination node type (edge_index_dict's key set is fixed for the
        # whole forward call).
        by_dst: dict[str, list] = {}
        for et in edge_index_dict:
            by_dst.setdefault(et[2], []).append(et)

        layers = []
        for layer_idx in range(self.num_layers):
            self_loop = self.self_loops[layer_idx]
            att_msg, att_dst = self.att_msg[layer_idx], self.att_dst[layer_idx]
            h_new = {nt: self_loop[nt](h[nt]) for nt in h}

            for dst_type, rel_list in by_dst.items():
                msgs, dsts, logits = [], [], []
                for et in rel_list:
                    edge_index = edge_index_dict[et]
                    if edge_index.numel() == 0:
                        continue
                    src_type = et[0]
                    w_r = torch.einsum("b,bij->ij", self.rel_coeff[self.rel_to_idx[et]], self.rel_basis)
                    src_idx, dst_idx = edge_index[0], edge_index[1]
                    msg = h[src_type][src_idx] @ w_r
                    logit = self.leaky_relu(
                        att_msg(msg).squeeze(-1) + att_dst(h[dst_type][dst_idx]).squeeze(-1)
                    )
                    msgs.append(msg)
                    dsts.append(dst_idx)
                    logits.append(logit)

                if not msgs:
                    # No relation feeding dst_type contributed any edges this
                    # layer -- same fallback RGCNEncoder documents: h_new
                    # keeps the self-loop-only value, never a stale zero.
                    continue

                msg_all = torch.cat(msgs, dim=0)
                dst_all = torch.cat(dsts, dim=0)
                logit_all = torch.cat(logits, dim=0)

                # THE core mechanism change: one joint softmax over every
                # incoming edge to a destination node, spanning every
                # relation together -- not RGCNEncoder's per-relation-mean-
                # then-summed scheme. Matches HGTConv's per-node softmax
                # scope, at a fraction of its parameter cost.
                alpha = softmax(logit_all, dst_all, num_nodes=h[dst_type].size(0))
                weighted = msg_all * alpha.unsqueeze(-1)
                agg = scatter(weighted, dst_all, dim=0, dim_size=h[dst_type].size(0), reduce="sum")
                h_new[dst_type] = h_new[dst_type] + agg

            h = {nt: self.dropout(torch.relu(h_new[nt])) if nt in h_new else h[nt] for nt in h}
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
