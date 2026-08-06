"""
Graph-correctness tests — Step 2 (`docs/13_Testing_Documentation.md` §6-7,
`docs/14_Model_Development_Roadmap.md` §5).

Run: pytest ml/tests/test_graph.py -v -s
"""

from __future__ import annotations

import datetime as dt

import pytest
import torch

from ml.data.db import get_connection
from ml.data.snapshots import get_snapshot_schedule
from ml.graph.builder import NODE_FEATURE_BUILDERS, build_snapshot

UTC = dt.timezone.utc


@pytest.fixture(scope="module")
def conn():
    connection = get_connection()
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def schedule(conn):
    return get_snapshot_schedule(conn)


@pytest.fixture(scope="module")
def last_snapshot(conn, schedule):
    """The most fully-populated snapshot (Dec 1) -- used by tests that just
    need *a* representative graph, not every t0."""
    row = schedule.iloc[-1]
    data, id_maps = build_snapshot(conn, row.t0)
    return data, id_maps


# ---------------------------------------------------------------------------
# Snapshot assembly -- every t0 in the schedule builds cleanly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t0_index", range(6))
def test_snapshot_builds_without_nan_or_shape_mismatch(conn, schedule, t0_index):
    row = schedule.iloc[t0_index]
    data, id_maps = build_snapshot(conn, row.t0)

    assert set(data.node_types) == set(NODE_FEATURE_BUILDERS.keys())

    for node_type in data.node_types:
        x = data[node_type].x
        assert x.dim() == 2, f"{node_type} feature tensor is not 2D"
        assert x.size(0) == len(id_maps[node_type]), f"{node_type} row count != id map size"
        assert not torch.isnan(x).any(), f"{node_type} feature tensor contains NaN at t0={row.t0}"
        assert not torch.isinf(x).any(), f"{node_type} feature tensor contains Inf at t0={row.t0}"

    for (src, rel, dst) in data.edge_types:
        edge_index = data[src, rel, dst].edge_index
        assert edge_index.dim() == 2 and edge_index.size(0) == 2
        if edge_index.numel() > 0:
            assert edge_index[0].max().item() < data[src].num_nodes, (
                f"{(src, rel, dst)} source index out of range"
            )
            assert edge_index[1].max().item() < data[dst].num_nodes, (
                f"{(src, rel, dst)} target index out of range"
            )
        edge_attr = data[src, rel, dst].get("edge_attr")
        if edge_attr is not None:
            assert edge_attr.size(0) == edge_index.size(1), f"{(src, rel, dst)} edge_attr/edge_index length mismatch"
            assert not torch.isnan(edge_attr).any(), f"{(src, rel, dst)} edge_attr contains NaN"


# ---------------------------------------------------------------------------
# Reverse-relation generation -- exactly 20 meta-relations, SHIPS_FROM twice
# ---------------------------------------------------------------------------

def test_exactly_20_meta_relations(last_snapshot):
    data, _ = last_snapshot
    assert len(data.edge_types) == 20, f"expected 20 meta-relations, got {len(data.edge_types)}: {data.edge_types}"

    ships_from_forward = [et for et in data.edge_types if et[1] == "SHIPS_FROM"]
    assert len(ships_from_forward) == 2, "SHIPS_FROM should be 2 distinct meta-relations (-> Supplier, -> Factory)"
    assert {et[2] for et in ships_from_forward} == {"Supplier", "Factory"}

    rev_ships_from = [et for et in data.edge_types if et[1] == "rev_SHIPS_FROM"]
    assert len(rev_ships_from) == 2, "rev_SHIPS_FROM should also be 2 distinct meta-relations"


def test_all_ten_reverse_relations_present(last_snapshot):
    data, _ = last_snapshot
    forward_rels = {rel for (_, rel, _) in data.edge_types if not rel.startswith("rev_")}
    reverse_rels = {rel for (_, rel, _) in data.edge_types if rel.startswith("rev_")}
    assert len(forward_rels) == 9, f"expected 9 distinct forward relation names (SHIPS_FROM shared), got {forward_rels}"
    assert reverse_rels == {f"rev_{r}" for r in forward_rels}


def test_supplier_has_nonzero_indegree_from_rev_supplies(last_snapshot):
    """Spot-check from docs/14_Model_Development_Roadmap.md §5's 'done when':
    a Supplier node has nonzero in-degree from rev_SUPPLIES."""
    data, _ = last_snapshot
    rev_supplies = data["Component", "rev_SUPPLIES", "Supplier"].edge_index
    in_degree = torch.zeros(data["Supplier"].num_nodes, dtype=torch.long)
    in_degree.scatter_add_(0, rev_supplies[1], torch.ones(rev_supplies.size(1), dtype=torch.long))
    assert (in_degree > 0).any(), "no Supplier node has nonzero in-degree from rev_SUPPLIES"
    # Suppliers follow a power-law component-count distribution (db/README.md)
    # -- not every supplier has a component, so most (not all) is the honest bar.
    pct_with_indegree = (in_degree > 0).float().mean().item()
    assert pct_with_indegree > 0.5, (
        f"expected most suppliers to have >=1 component (rev_SUPPLIES in-edge), got {pct_with_indegree:.0%}"
    )


# ---------------------------------------------------------------------------
# Dimension verification -- node feature width matches real one-hot cardinality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node_type", list(NODE_FEATURE_BUILDERS.keys()))
def test_feature_dimension_matches_builder_output(conn, schedule, node_type):
    """
    The tensor's column count must equal exactly what its own feature
    builder produced -- catching any silent column drop/duplication between
    `ml/data/features.py` and `ml/graph/builder.py`'s tensor conversion.
    """
    t0 = schedule.iloc[-1].t0
    builder = NODE_FEATURE_BUILDERS[node_type]
    df = builder(conn, t0)
    expected_dim = len([c for c in df.columns if c != "entity_id"])

    data, _ = build_snapshot(conn, t0)
    assert data[node_type].x.size(1) == expected_dim


def test_one_hot_cardinality_matches_distinct_db_values(conn, last_snapshot):
    """
    Country one-hot width (Supplier) must equal the actual distinct country
    count in `suppliers` -- guards against a hardcoded category list drifting
    from the real data (`docs/06_Graph_Database_Design.md` §7's "recompute
    against the real one-hot widths" instruction).
    """
    data, _ = last_snapshot
    distinct_countries = pytest_read_distinct(conn, "SELECT DISTINCT country FROM suppliers")
    # Supplier columns: on_time_rate_30/90/180d, trend_slope, lateness_variance,
    # days_since_last_late (6) + lead_time_days_z + capacity_score_z (2) + country one-hot
    non_onehot_cols = 6 + 2
    assert data["Supplier"].x.size(1) - non_onehot_cols == len(distinct_countries)


def pytest_read_distinct(conn, sql: str) -> list:
    import pandas as pd
    return sorted(v for (v,) in pd.read_sql(sql, conn).itertuples(index=False, name=None) if v is not None)


# ---------------------------------------------------------------------------
# As-of edge filtering -- BOM (product_components) validity windows
# ---------------------------------------------------------------------------

# A known substitution from the synthetic dataset (db/README.md: "4
# substitutions Feb-May, count-neutral swaps"): product c0f44500... dropped
# one component on 2024-02-12 and picked up another the same day. Component
# UUIDs shift whenever the dataset is regenerated with a different world
# size (db/generate_dataset.py's SUP_N/SCALE) -- the RNG draw order changes
# which specific component lands in this BOM slot, even though the
# substitution DATE and PRODUCT (index 0, so its uid() hash is stable) don't.
# Re-derive via: SELECT product_id, component_id, deactivated_at FROM
# product_components WHERE deactivated_at IS NOT NULL ORDER BY deactivated_at LIMIT 1;
_SUBSTITUTED_PRODUCT = "c0f44500-e1d1-59e0-b53b-e0e98b24a7cf"
_OLD_COMPONENT = "0810a1fe-71ab-5234-ad85-27da3b9b87ef"
_NEW_COMPONENT = "5a9ef7e6-98e2-53bd-b040-83ca307b68cb"
_SUBSTITUTION_DATE = dt.datetime(2024, 2, 12, 13, 30, tzinfo=UTC)


def _has_used_in_edge(data, id_maps, component_id: str, product_id: str) -> bool:
    if component_id not in id_maps["Component"] or product_id not in id_maps["Product"]:
        return False
    c_idx, p_idx = id_maps["Component"][component_id], id_maps["Product"][product_id]
    edge_index = data["Component", "USED_IN", "Product"].edge_index
    return bool(((edge_index[0] == c_idx) & (edge_index[1] == p_idx)).any())


def test_bom_edge_excluded_after_deactivation(conn):
    """Given a BOM entry with deactivated_at before t0, the edge is excluded."""
    t0_after = _SUBSTITUTION_DATE + dt.timedelta(days=7)
    data, id_maps = build_snapshot(conn, t0_after)
    assert not _has_used_in_edge(data, id_maps, _OLD_COMPONENT, _SUBSTITUTED_PRODUCT), (
        "deactivated BOM edge should be excluded once t0 is past deactivated_at"
    )
    assert _has_used_in_edge(data, id_maps, _NEW_COMPONENT, _SUBSTITUTED_PRODUCT), (
        "the replacement BOM edge should be included once t0 is past its created_at"
    )


def test_bom_edge_included_before_deactivation(conn):
    """Given a BOM entry with created_at before t0 < deactivated_at, the edge is included."""
    t0_before = _SUBSTITUTION_DATE - dt.timedelta(days=7)
    data, id_maps = build_snapshot(conn, t0_before)
    assert _has_used_in_edge(data, id_maps, _OLD_COMPONENT, _SUBSTITUTED_PRODUCT), (
        "BOM edge active as of t0 (before its deactivated_at) should be included"
    )
    assert not _has_used_in_edge(data, id_maps, _NEW_COMPONENT, _SUBSTITUTED_PRODUCT), (
        "the replacement BOM edge should not exist yet before its created_at"
    )


def test_product_factory_edge_excluded_before_created_at(conn):
    """
    All `product_factories` rows in this dataset share created_at =
    2024-01-01; a t0 strictly before that must exclude every MANUFACTURED_AT
    edge (as-of edge filter, `product_factories` half).
    """
    t0_before = dt.datetime(2023, 6, 1, tzinfo=UTC)
    data, _ = build_snapshot(conn, t0_before)
    assert data["Product", "MANUFACTURED_AT", "Factory"].edge_index.numel() == 0


# ---------------------------------------------------------------------------
# Sparse-relation-merged HGT encoder (post-v3 follow-up, Task 2:
# reports/step5_result_v3.md's shortage HGT-vs-GraphSAGE gap)
# ---------------------------------------------------------------------------

def test_sparse_hgt_encoder_forward_pass_and_param_reduction(last_snapshot):
    """`hgt_sparse` must (a) run cleanly on the real 20-meta-relation graph
    (its 8-thin/12-dense split assertions are schema-specific, unlike the
    small fixture `test_model.py` uses for the other architectures), and
    (b) actually have fewer parameters than plain `hgt` -- confirming the
    8 thin relations really share one SAGEConv instance rather than each
    keeping its own HGTConv relation-parameter set."""
    from ml.models.encoder import build_encoder

    data, _ = last_snapshot
    metadata = data.metadata()
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types}

    hgt = build_encoder("hgt", metadata, in_dims, hidden=64, num_layers=4)
    sparse = build_encoder("hgt_sparse", metadata, in_dims, hidden=64, num_layers=4)

    layers = sparse(data.x_dict, data.edge_index_dict)
    assert len(layers) == 4
    for node_type in data.node_types:
        h = layers[-1][node_type]
        assert h.shape == (data[node_type].num_nodes, 64)
        assert not torch.isnan(h).any()

    hgt_params = sum(p.numel() for p in hgt.parameters())
    sparse_params = sum(p.numel() for p in sparse.parameters())
    assert sparse_params < hgt_params, (
        f"hgt_sparse ({sparse_params:,}) should have fewer params than hgt ({hgt_params:,}) "
        "-- the 8 thin relations should be sharing one conv instance, not each keeping its own"
    )
