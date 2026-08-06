"""
Sparse-relation-merged HGT encoder — Task 2 of the post-v3 follow-up
(`reports/step5_result_v3.md`'s Claim 1: HGT consistently loses to
GraphSAGE on shortage, at both fixed-d and matched-d, across every seed).

**Hypothesis under test:** HGT allocates one full relation-specific
parameter set (`W_ATT`, `W_MSG`, `mu` — `k_rel`/`v_rel`/`p_rel` in
`torch_geometric.nn.HGTConv`'s implementation) to *every* meta-relation,
including the thinnest ones. If those thin relations' dedicated parameters
overfit to noise rather than learning real structure, that could explain a
GraphSAGE (relation-agnostic) advantage on a task like shortage that leans
on exactly two of the thinnest relations (`STOCKED_AT`, `USED_IN`).

**Why "drop" isn't the primary test here.** `STOCKED_AT` (Product-Warehouse
stock/threshold context) and `USED_IN` (Component-Product BOM structure) are
causally central to shortage's own reasoning -- dropping them doesn't test
"thin-relation overfitting," it tests "does removing shortage's own signal
hurt," a different and less interesting question. Merging is the fairer
test: same information reaches the encoder, just through a shared rather
than a dedicated parameter set.

**Why "merge" required a custom encoder, not a `metadata` relabel.**
`HGTConv`'s public API parameterizes strictly per `(src_type, rel, dst_type)`
triple, and its per-relation weights (`k_rel`/`v_rel`) are gathered using the
*source type name* from that triple -- relations with different type-pairs
can't share one triple's parameters by renaming alone (the four thin
relations don't even share a common src or dst type: Supplier->Component,
Component->Product, Product->Warehouse, Product->Factory). The
literal-parameter-sharing trick used here: build one ordinary
`SAGEConv(hidden, hidden)` instance and pass the *same object* into
`HeteroConv` under all 8 thin-relation keys (4 forward + 4 reverse) --
`torch.nn.ModuleDict` registers it under multiple string keys, but since
it's one Python object, its `Parameter` tensors -- and every gradient
update -- are genuinely shared across all 8, exactly what "merge into one
generic relation" means in practice. The remaining 12 (dense) meta-relations
keep full per-relation HGT parameterization, built via a second `HGTConv`
whose `metadata` is restricted to just those 12.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, HGTConv, Linear, SAGEConv

# The 4 thinnest forward meta-relations in the v3 graph, by edge count at the
# largest snapshot (Product__MANUFACTURED_AT__Factory 1,933; Supplier__
# SUPPLIES__Component 2,400; Product__STOCKED_AT__Warehouse 3,194; Component__
# USED_IN__Product 7,092 -- next-thinnest, Shipment__SHIPS_FROM__Supplier, is
# already 15,608, more than double USED_IN's count: a real gap, not an
# arbitrary cutoff).
THIN_FORWARD = {
    ("Product", "MANUFACTURED_AT", "Factory"),
    ("Supplier", "SUPPLIES", "Component"),
    ("Product", "STOCKED_AT", "Warehouse"),
    ("Component", "USED_IN", "Product"),
}
THIN_REVERSE = {(dst, f"rev_{rel}", src) for (src, rel, dst) in THIN_FORWARD}
THIN_RELATIONS = THIN_FORWARD | THIN_REVERSE  # 8 of 20 meta-relations


class HGTSparseMergedEncoder(nn.Module):
    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 heads: int = 4, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        node_types, edge_types = metadata
        dense_edges = [et for et in edge_types if et not in THIN_RELATIONS]
        thin_edges = [et for et in edge_types if et in THIN_RELATIONS]
        assert len(thin_edges) == 8, f"expected 8 thin meta-relations, found {len(thin_edges)}"
        assert len(dense_edges) == 12, f"expected 12 dense meta-relations, found {len(dense_edges)}"
        dense_metadata = (node_types, dense_edges)

        self.lin_in = nn.ModuleDict({nt: Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.dense_convs = nn.ModuleList(
            [HGTConv(hidden, hidden, dense_metadata, heads=heads) for _ in range(num_layers)]
        )
        # One SAGEConv instance per layer, reused (same object) across all 8
        # thin relation keys -- the actual parameter merge.
        self.thin_convs = nn.ModuleList()
        for _ in range(num_layers):
            shared = SAGEConv(hidden, hidden)
            self.thin_convs.append(HeteroConv({et: shared for et in thin_edges}, aggr="sum"))
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict: dict, edge_index_dict: dict) -> list[dict]:
        h = {nt: self.lin_in[nt](x) for nt, x in x_dict.items()}
        dense_edge_index = {et: e for et, e in edge_index_dict.items() if et not in THIN_RELATIONS}
        thin_edge_index = {et: e for et, e in edge_index_dict.items() if et in THIN_RELATIONS}

        layers = []
        for dense_conv, thin_conv in zip(self.dense_convs, self.thin_convs):
            h_dense = dense_conv(h, dense_edge_index)
            h_thin = thin_conv(h, thin_edge_index)
            h_new = {}
            for nt, prev in h.items():
                parts = [v[nt] for v in (h_dense, h_thin) if nt in v]
                h_new[nt] = sum(parts) if parts else prev
            h = {nt: self.dropout(torch.relu(h_new[nt])) for nt in h}
            layers.append(h)
        return layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
