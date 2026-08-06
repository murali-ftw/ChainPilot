"""
`HeteroData` graph-snapshot assembly — Step 2 of
`docs/14_Model_Development_Roadmap.md` §5 (`docs/06_Graph_Database_Design.md`
§5-11, `docs/10_AI_ML_Documentation.md` §7).

One `HeteroData` object per `t0`: 8 node types (`ml/data/features.py`'s
`NODE_FEATURE_BUILDERS`), 10 forward relations built here, then
`ToUndirected()` to reach 20 meta-relations. There is no incremental-update
path (`docs/06_Graph_Database_Design.md` §11) -- every snapshot is built
fresh from the as-of state of the source tables.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from ml.data.features import NODE_FEATURE_BUILDERS

FEATURE_SPEC_VERSION = "v1"

# (src_type, relation, dst_type) -> everything needed to pull that edge's
# endpoint id columns (+ optional edge_attr columns) as of t0.
# `sql` must select exactly `src_id`, `dst_id`, and any attr_cols, filtered
# to rows/validity-windows active at t0 (docs/10_AI_ML_Documentation.md §6.3).
FORWARD_RELATIONS = [
    {
        "src": "Supplier", "rel": "SUPPLIES", "dst": "Component",
        # Task 3 follow-up experiment (`reports/step5_result_v3.md` addendum):
        # union the mandatory primary supplier (components.supplier_id, one
        # per component, unconditional) with any SECONDARY suppliers from the
        # `component_suppliers` junction table valid as of t0. On the base v3
        # dataset (component_suppliers empty) this UNION ALL is a no-op --
        # identical edge set to before. Only on the Task 3 dataset (~15-20%
        # of components dual-sourced) does the second branch add rows, which
        # is what finally makes docs/06_Graph_Database_Design.md §6.1's
        # Supplier->SUPPLIES->Component->rev_SUPPLIES->Supplier co-parent
        # path reach a real second supplier.
        "sql": """
            SELECT supplier_id AS src_id, id AS dst_id FROM components
            UNION ALL
            SELECT supplier_id AS src_id, component_id AS dst_id
            FROM component_suppliers
            WHERE created_at <= %(t0)s
              AND (deactivated_at IS NULL OR deactivated_at > %(t0)s)
        """,
        "attrs": [],
    },
    {
        "src": "Component", "rel": "USED_IN", "dst": "Product",
        "sql": """
            SELECT component_id AS src_id, product_id AS dst_id, quantity_required
            FROM product_components
            WHERE created_at <= %(t0)s
              AND (deactivated_at IS NULL OR deactivated_at > %(t0)s)
        """,
        "attrs": ["quantity_required"],
    },
    {
        "src": "Product", "rel": "STOCKED_AT", "dst": "Warehouse",
        "sql": """
            WITH latest_inv AS (
                SELECT DISTINCT ON (inventory_id)
                    product_id, warehouse_id, stock_level, reorder_threshold
                FROM inventory_history
                WHERE observed_at <= %(t0)s AND recorded_at <= %(t0)s
                ORDER BY inventory_id, observed_at DESC
            )
            SELECT product_id AS src_id, warehouse_id AS dst_id,
                   stock_level, reorder_threshold
            FROM latest_inv
        """,
        "attrs": ["stock_level", "reorder_threshold"],
    },
    {
        "src": "Product", "rel": "MANUFACTURED_AT", "dst": "Factory",
        "sql": """
            SELECT product_id AS src_id, factory_id AS dst_id, capacity_units_per_day
            FROM product_factories
            WHERE created_at <= %(t0)s
              AND (deactivated_at IS NULL OR deactivated_at > %(t0)s)
        """,
        "attrs": ["capacity_units_per_day"],
    },
    {
        "src": "Shipment", "rel": "SHIPS_FROM", "dst": "Supplier",
        "sql": """
            SELECT id AS src_id, supplier_id AS dst_id FROM shipments
            WHERE created_at <= %(t0)s AND supplier_id IS NOT NULL
        """,
        "attrs": [],
    },
    {
        "src": "Shipment", "rel": "SHIPS_FROM", "dst": "Factory",
        "sql": """
            SELECT id AS src_id, factory_id AS dst_id FROM shipments
            WHERE created_at <= %(t0)s AND factory_id IS NOT NULL
        """,
        "attrs": [],
    },
    {
        "src": "Shipment", "rel": "SHIPS_TO", "dst": "Warehouse",
        "sql": """
            SELECT id AS src_id, warehouse_id AS dst_id,
                   EXTRACT(EPOCH FROM (eta - %(t0)s)) / 86400.0 AS days_to_eta
            FROM shipments
            WHERE created_at <= %(t0)s AND warehouse_id IS NOT NULL
        """,
        "attrs": ["days_to_eta"],
    },
    {
        "src": "Shipment", "rel": "FULFILLS", "dst": "Order",
        "sql": """
            SELECT id AS src_id, order_id AS dst_id FROM shipments
            WHERE created_at <= %(t0)s AND order_id IS NOT NULL
        """,
        "attrs": [],
    },
    {
        "src": "Order", "rel": "ORDERED", "dst": "Product",
        "sql": """
            SELECT order_id AS src_id, product_id AS dst_id, quantity
            FROM order_items
            WHERE created_at <= %(t0)s
        """,
        "attrs": ["quantity"],
    },
    {
        "src": "Order", "rel": "PLACED_BY", "dst": "Customer",
        "sql": """
            SELECT id AS src_id, customer_id AS dst_id FROM orders
            WHERE placed_at <= %(t0)s
        """,
        "attrs": [],
    },
]


def _current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_node_tables(conn, t0: dt.datetime) -> dict[str, pd.DataFrame]:
    """One as-of feature DataFrame per node type, keyed by node type name."""
    return {node_type: builder(conn, t0) for node_type, builder in NODE_FEATURE_BUILDERS.items()}


def _to_feature_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Feature columns (everything but `entity_id`) as a float32 tensor.
    NaNs (e.g. a product with no inventory_history row yet as of an early t0)
    are filled with 0.0 -- the neutral value for both z-scored numeric
    columns and one-hot indicators -- so the assembled graph never carries a
    NaN into message passing (`docs/13_Testing_Documentation.md` §7).
    """
    cols = [c for c in df.columns if c != "entity_id"]
    values = df[cols].astype(float).fillna(0.0).to_numpy()
    return torch.tensor(values, dtype=torch.float)


def _build_edges(conn, t0: dt.datetime, id_maps: dict[str, dict[str, int]]) -> dict:
    """
    For every FORWARD_RELATIONS spec, pull (src_id, dst_id, *attrs) as of t0
    and translate UUIDs to tensor-row indices via `id_maps`. Rows whose
    endpoint isn't in that node type's current id map (shouldn't happen for
    this dataset's fully-populated master tables, but would for a partial
    load) are dropped rather than crashing the whole snapshot.
    """
    edges = {}
    for spec in FORWARD_RELATIONS:
        df = pd.read_sql(spec["sql"], conn, params={"t0": t0})
        src_map, dst_map = id_maps[spec["src"]], id_maps[spec["dst"]]
        src_idx = df["src_id"].map(src_map)
        dst_idx = df["dst_id"].map(dst_map)
        valid = src_idx.notna() & dst_idx.notna()
        src_idx, dst_idx = src_idx[valid].astype(int), dst_idx[valid].astype(int)

        edge_index = torch.tensor(np.vstack([src_idx.to_numpy(), dst_idx.to_numpy()]), dtype=torch.long)
        edge_attr = None
        if spec["attrs"]:
            attr_values = df.loc[valid, spec["attrs"]].astype(float).fillna(0.0).to_numpy()
            edge_attr = torch.tensor(attr_values, dtype=torch.float)

        key = (spec["src"], spec["rel"], spec["dst"])
        edges[key] = (edge_index, edge_attr)
    return edges


def build_snapshot(conn, t0: dt.datetime) -> tuple[HeteroData, dict[str, dict[str, int]]]:
    """
    Assemble one leakage-clean `HeteroData` object for `t0`: 8 node types,
    10 forward relations, then `ToUndirected()` for the 10 reverse relations
    (`docs/06_Graph_Database_Design.md` §6). Returns the graph plus each
    node type's UUID -> tensor-row-index map (needed by evaluation/
    explanation to translate predictions back to PostgreSQL entities).
    """
    node_tables = _build_node_tables(conn, t0)
    id_maps = {
        node_type: {entity_id: idx for idx, entity_id in enumerate(df["entity_id"])}
        for node_type, df in node_tables.items()
    }

    data = HeteroData()
    for node_type, df in node_tables.items():
        data[node_type].x = _to_feature_tensor(df)
        data[node_type].node_id = df["entity_id"].tolist()

    for (src, rel, dst), (edge_index, edge_attr) in _build_edges(conn, t0, id_maps).items():
        data[src, rel, dst].edge_index = edge_index
        if edge_attr is not None:
            data[src, rel, dst].edge_attr = edge_attr

    data = ToUndirected()(data)
    return data, id_maps


def snapshot_counts(data: HeteroData) -> tuple[dict, dict]:
    """node_counts / edge_counts in the shape `graph_snapshots` expects."""
    node_counts = {node_type: data[node_type].num_nodes for node_type in data.node_types}
    edge_counts = {
        f"{src}__{rel}__{dst}": data[src, rel, dst].edge_index.size(1)
        for (src, rel, dst) in data.edge_types
    }
    return node_counts, edge_counts


def _label_counts(conn, snapshot_id) -> dict:
    df = pd.read_sql(
        """
        SELECT task, count(*) FILTER (WHERE label) AS positives
        FROM training_labels
        WHERE snapshot_id = %(sid)s
        GROUP BY task
        """,
        conn,
        params={"sid": str(snapshot_id)},
    )
    return {f"{row.task}_pos": int(row.positives) for row in df.itertuples()}


def persist_snapshot_metadata(conn, snapshot_id, node_counts: dict, edge_counts: dict,
                                label_counts: dict, construction_seconds: float) -> None:
    """
    Refresh the existing `graph_snapshots` row for this `t0` with the real
    counts from this build (`docs/14_Model_Development_Roadmap.md` §5:
    "Write one graph_snapshots row per t0 with real node/edge/label
    counts"). Updates in place rather than inserting a new
    (t0, feature_spec_version) row -- one row per t0 stays the invariant;
    the seed rows written at dataset-generation time predate any actual
    graph-assembly code and used different (informal) key names.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE graph_snapshots
            SET node_counts = %(node_counts)s,
                edge_counts = %(edge_counts)s,
                label_counts = %(label_counts)s,
                feature_spec_version = %(fsv)s,
                git_commit = %(git_commit)s,
                construction_seconds = %(seconds)s
            WHERE id = %(sid)s
            """,
            {
                "node_counts": json.dumps(node_counts),
                "edge_counts": json.dumps(edge_counts),
                "label_counts": json.dumps(label_counts),
                "fsv": FEATURE_SPEC_VERSION,
                "git_commit": _current_git_commit(),
                "seconds": construction_seconds,
                "sid": str(snapshot_id),
            },
        )
    conn.commit()


def build_and_persist_snapshot(conn, snapshot_id, t0: dt.datetime) -> tuple[HeteroData, dict]:
    """Build one snapshot, write its metadata, return (data, id_maps)."""
    start = time.monotonic()
    data, id_maps = build_snapshot(conn, t0)
    elapsed = time.monotonic() - start

    node_counts, edge_counts = snapshot_counts(data)
    label_counts = _label_counts(conn, snapshot_id)
    persist_snapshot_metadata(conn, snapshot_id, node_counts, edge_counts, label_counts, elapsed)
    return data, id_maps
