"""
As-of feature computation and the leakage exclusion list.

Step 1 of `docs/14_Model_Development_Roadmap.md` §4 — the leakage contract,
restated precisely in `docs/10_AI_ML_Documentation.md` §6.1:

    FEATURES  <- computed ONLY from rows with timestamp <= t0
    LABEL     <- did the event occur within (t0, t0 + H] ?

Every function below takes a `t0` cutoff and reads only rows that existed
at or before it. The exclusion list (`docs/10_AI_ML_Documentation.md` §6.2)
is enforced structurally, not just by convention: the three excluded
columns are never named in any SELECT in this file —

    shipments.status, shipments.delivered_at   -> reconstruct from
                                                   shipment_status_history
    inventory.stock_level                      -> reconstruct from
                                                   inventory_history
    suppliers.reliability_history               -> reconstruct from
                                                   supplier_temporal_features

Each `*_features_asof` function returns a DataFrame with one row per entity
and a uniform `entity_id` column (so `ml/data/snapshots.py` can join labels
without per-entity-type key juggling), plus feature columns only — no
timestamps, no excluded columns, nothing that isn't a feature.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Static reference categories (fixed one-hot column sets, so the feature
# frame has the same columns at every t0 even if a category happens to have
# zero rows in a given snapshot). Suppliers/components/products/customers are
# reference data with no valid-time dimension in this schema, so reading
# their category-defining columns directly (country, component_type,
# category, priority_tier) is not a leakage risk -- only their *state*
# columns are (see module docstring).
# ---------------------------------------------------------------------------

SHIPMENT_STATUSES = ["scheduled", "in_transit", "delivered", "delayed"]
CUSTOMER_TIERS = ["strategic", "standard", "low"]
ORDER_STATUSES = ["open", "fulfilled", "cancelled", "at_risk"]


def _one_hot(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(series, categories=categories)
    return pd.get_dummies(cat, prefix=prefix, dtype=float)


def _zscore(series: pd.Series) -> pd.Series:
    mean, std = series.mean(), series.std(ddof=0)
    if not std or np.isnan(std):
        return series * 0.0
    return (series - mean) / std


def _fetch_distinct(conn, sql: str) -> list[str]:
    return sorted(v for (v,) in pd.read_sql(sql, conn).itertuples(index=False, name=None) if v is not None)


# ---------------------------------------------------------------------------
# Supplier features
# ---------------------------------------------------------------------------

def supplier_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """
    Supplier node features as of t0.

    lead_time_days, capacity_score: static reference columns on `suppliers`,
        scaled (z-score) over all suppliers -- a fixed transform, not
        label-derived, so it carries no leakage risk regardless of t0.
    country one-hot: static reference (`suppliers.country`).
    on_time_rate_30d/90d/180d, trend_slope, lateness_variance,
        days_since_last_late: the most recent `supplier_temporal_features`
        row with `as_of_date <= t0` -- NEVER `suppliers.reliability_history`.
    """
    countries = _fetch_distinct(conn, "SELECT DISTINCT country FROM suppliers")

    # suppliers.created_at is dataset-generation metadata (when this row was
    # written), not a business "supplier onboarded" date -- every supplier is
    # active across the whole 2024 snapshot schedule, so it is not filtered
    # by t0 (unlike orders.placed_at, which is a genuine business timestamp).
    base = pd.read_sql(
        "SELECT id AS entity_id, lead_time_days, capacity_score, country FROM suppliers",
        conn,
    )

    temporal = pd.read_sql(
        """
        SELECT DISTINCT ON (supplier_id)
            supplier_id AS entity_id,
            on_time_rate_30d, on_time_rate_90d, on_time_rate_180d,
            trend_slope, lateness_variance, days_since_last_late
        FROM supplier_temporal_features
        WHERE as_of_date <= %(t0_date)s
        ORDER BY supplier_id, as_of_date DESC
        """,
        conn,
        params={"t0_date": t0.date()},
    )

    df = base.merge(temporal, on="entity_id", how="left")
    df["lead_time_days_z"] = _zscore(df["lead_time_days"].astype(float))
    df["capacity_score_z"] = _zscore(df["capacity_score"].astype(float))
    df = pd.concat([df, _one_hot(df["country"], countries, "country")], axis=1)
    df = df.drop(columns=["lead_time_days", "capacity_score", "country"])
    return df


# ---------------------------------------------------------------------------
# Shipment features
# ---------------------------------------------------------------------------

def shipment_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """
    Shipment node features as of t0.

    days_to_eta: `eta - t0`, recomputed per snapshot (`eta` is a plan/schedule
        field, not a terminal-state column -- it is not on the exclusion
        list).
    as-of status one-hot: reconstructed from `shipment_status_history` at
        `changed_at <= t0` -- NEVER `shipments.status` or `.delivered_at`.
    days_since_dispatch: `t0 - dispatched_at`, only if dispatch had already
        happened by t0.
    carrier on-time rate: nearest `carrier_performance_snapshots` row with
        `as_of_date <= t0`, averaged across routes for that carrier.
    """
    base = pd.read_sql(
        """
        SELECT id AS entity_id, supplier_id, eta, dispatched_at, carrier,
               origin_location, created_at
        FROM shipments
        WHERE created_at <= %(t0)s
        """,
        conn,
        params={"t0": t0},
    )

    as_of_status = pd.read_sql(
        """
        SELECT DISTINCT ON (shipment_id)
            shipment_id AS entity_id, status
        FROM shipment_status_history
        WHERE changed_at <= %(t0)s
        ORDER BY shipment_id, changed_at DESC
        """,
        conn,
        params={"t0": t0},
    )

    carrier_perf = pd.read_sql(
        """
        WITH by_date AS (
            SELECT carrier, as_of_date, avg(on_time_rate_90d) AS on_time_rate_90d
            FROM carrier_performance_snapshots
            WHERE as_of_date <= %(t0_date)s
            GROUP BY carrier, as_of_date
        )
        SELECT DISTINCT ON (carrier)
            carrier, on_time_rate_90d AS carrier_on_time_rate_90d
        FROM by_date
        ORDER BY carrier, as_of_date DESC
        """,
        conn,
        params={"t0_date": t0.date()},
    )

    df = base.merge(as_of_status, on="entity_id", how="left")
    df = df.merge(carrier_perf[["carrier", "carrier_on_time_rate_90d"]], on="carrier", how="left")

    t0_ts = pd.Timestamp(t0)
    df["days_to_eta"] = (pd.to_datetime(df["eta"], utc=True) - t0_ts).dt.total_seconds() / 86400
    dispatched = pd.to_datetime(df["dispatched_at"], utc=True)
    days_since_dispatch = (t0_ts - dispatched).dt.total_seconds() / 86400
    df["days_since_dispatch"] = days_since_dispatch.where(dispatched <= t0_ts)

    df = pd.concat([df, _one_hot(df["status"], SHIPMENT_STATUSES, "status")], axis=1)

    df = df.drop(columns=["supplier_id", "eta", "dispatched_at", "carrier",
                           "origin_location", "created_at", "status"])
    return df


# ---------------------------------------------------------------------------
# Product features (drives the shortage task, which is product-level)
# ---------------------------------------------------------------------------

def product_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """
    Product node features as of t0.

    stock/reorder-threshold ratios: aggregated (min/mean across warehouses)
        over the latest `inventory_history` row per (product, warehouse)
        with `observed_at <= t0 AND recorded_at <= t0` -- NEVER
        `inventory.stock_level`.
    category one-hot: static reference (`products.category`).
    bom_component_count / bom_mean_quantity_required: active
        `product_components` edges as of t0 (`created_at <= t0` and
        `deactivated_at IS NULL OR deactivated_at > t0`).
    """
    categories = _fetch_distinct(conn, "SELECT DISTINCT category FROM products")

    # products.created_at is dataset-generation metadata, same caveat as
    # suppliers above -- not filtered by t0.
    base = pd.read_sql(
        "SELECT id AS entity_id, category FROM products",
        conn,
    )

    inv = pd.read_sql(
        """
        WITH latest_inv AS (
            SELECT DISTINCT ON (inventory_id)
                product_id, warehouse_id, stock_level, reorder_threshold
            FROM inventory_history
            WHERE observed_at <= %(t0)s AND recorded_at <= %(t0)s
            ORDER BY inventory_id, observed_at DESC
        )
        SELECT
            product_id AS entity_id,
            min(stock_level::float / NULLIF(reorder_threshold, 0)) AS min_stock_ratio,
            avg(stock_level::float / NULLIF(reorder_threshold, 0)) AS avg_stock_ratio,
            sum(stock_level)::float AS total_stock,
            sum(reorder_threshold)::float AS total_reorder_threshold,
            count(*)::float AS warehouse_count
        FROM latest_inv
        GROUP BY product_id
        """,
        conn,
        params={"t0": t0},
    )

    bom = pd.read_sql(
        """
        SELECT
            product_id AS entity_id,
            count(*)::float AS bom_component_count,
            avg(quantity_required)::float AS bom_mean_quantity_required
        FROM product_components
        WHERE created_at <= %(t0)s
          AND (deactivated_at IS NULL OR deactivated_at > %(t0)s)
        GROUP BY product_id
        """,
        conn,
        params={"t0": t0},
    )

    df = base.merge(inv, on="entity_id", how="left").merge(bom, on="entity_id", how="left")
    df = pd.concat([df, _one_hot(df["category"], categories, "category")], axis=1)
    df = df.drop(columns=["category"])
    return df


# ---------------------------------------------------------------------------
# Order / customer features (no labelled task reads these yet -- entity_type
# 'order' has no rows in training_labels at this dataset version -- but the
# roadmap's Step 1.2 scope calls for them, and Step 2's graph assembly will
# need them as Order/Customer node features).
# ---------------------------------------------------------------------------

def order_customer_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """
    Order node features as of t0, with the placing customer's tier attached.

    days_to_due: `due_at - t0`, recomputed per snapshot.
    status one-hot: NOTE -- unlike shipments, this schema has no
        `order_status_history` table, so there is no as-of reconstruction
        path for order status; `orders.status` is read directly. This is
        not on the documented exclusion list (`docs/10_AI_ML_Documentation.md`
        §6.2), but it is a current-state column, so it is the single-feature
        leakage test's job to catch it if it turns out to leak in practice.
    customer priority_tier one-hot: static reference (`customers.priority_tier`).
    """
    base = pd.read_sql(
        """
        SELECT o.id AS entity_id, o.status, o.due_at, o.order_value,
               c.priority_tier
        FROM orders o
        JOIN customers c ON c.id = o.customer_id
        WHERE o.placed_at <= %(t0)s
        """,
        conn,
        params={"t0": t0},
    )

    t0_ts = pd.Timestamp(t0)
    df = base.copy()
    df["days_to_due"] = (pd.to_datetime(df["due_at"], utc=True) - t0_ts).dt.total_seconds() / 86400
    df = pd.concat([
        df,
        _one_hot(df["status"], ORDER_STATUSES, "status"),
        _one_hot(df["priority_tier"], CUSTOMER_TIERS, "priority_tier"),
    ], axis=1)
    df = df.drop(columns=["status", "due_at", "priority_tier"])
    return df


# ---------------------------------------------------------------------------
# Component / Factory / Warehouse / Customer / Order node features — Step 2
# graph assembly (`ml/graph/builder.py`). Distinct from `order_customer_
# features_asof` above: here Order and Customer are separate graph node
# types (`docs/06_Graph_Database_Design.md` §5), so customer priority_tier
# lives on the Customer node only, not duplicated onto Order's own vector --
# it reaches Order nodes through the `PLACED_BY`/`rev_PLACED_BY` edge instead.
# ---------------------------------------------------------------------------

def component_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """
    Component node features. Static reference data (`components.created_at`
    is dataset-generation metadata, same caveat as suppliers/products) --
    not filtered by t0.
    """
    component_types = _fetch_distinct(conn, "SELECT DISTINCT component_type FROM components")
    df = pd.read_sql(
        "SELECT id AS entity_id, unit_cost, component_type FROM components",
        conn,
    )
    df["unit_cost_z"] = _zscore(df["unit_cost"].astype(float))
    df = pd.concat([df, _one_hot(df["component_type"], component_types, "component_type")], axis=1)
    return df.drop(columns=["unit_cost", "component_type"])


def _location_bucket(location: pd.Series) -> pd.Series:
    """
    Coarsen a free-text 'City, Country' location string to its trailing
    token (country, or the whole string when there's no comma -- e.g.
    'Singapore'). A raw location one-hot would be one column per site,
    collinear with node identity for this dataset's small factory/warehouse
    counts; bucketing to country groups sites the way the graph's other
    country one-hot (Supplier) already does.
    """
    return location.fillna("unknown").str.split(",").str[-1].str.strip()


def factory_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """Factory node features. Static reference data -- not filtered by t0."""
    df = pd.read_sql(
        "SELECT id AS entity_id, capacity_units_per_day, location FROM factories",
        conn,
    )
    df["location_bucket"] = _location_bucket(df["location"])
    buckets = sorted(df["location_bucket"].unique())
    df["capacity_units_per_day_z"] = _zscore(df["capacity_units_per_day"].astype(float))
    df = pd.concat([df, _one_hot(df["location_bucket"], buckets, "location")], axis=1)
    return df.drop(columns=["capacity_units_per_day", "location", "location_bucket"])


def warehouse_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """Warehouse node features. Static reference data -- not filtered by t0."""
    df = pd.read_sql(
        "SELECT id AS entity_id, capacity_units, location FROM warehouses",
        conn,
    )
    df["location_bucket"] = _location_bucket(df["location"])
    buckets = sorted(df["location_bucket"].unique())
    df["capacity_units_z"] = _zscore(df["capacity_units"].astype(float))
    df = pd.concat([df, _one_hot(df["location_bucket"], buckets, "location")], axis=1)
    return df.drop(columns=["capacity_units", "location", "location_bucket"])


def customer_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """Customer node features. Static reference data -- not filtered by t0."""
    df = pd.read_sql("SELECT id AS entity_id, priority_tier FROM customers", conn)
    df = pd.concat([df, _one_hot(df["priority_tier"], CUSTOMER_TIERS, "priority_tier")], axis=1)
    return df.drop(columns=["priority_tier"])


def order_features_asof(conn, t0: dt.datetime) -> pd.DataFrame:
    """
    Order node features (Customer's priority_tier is not duplicated here --
    see module note above). `orders.placed_at` is genuine business time
    (unlike the reference tables above), so it is filtered by t0.
    """
    df = pd.read_sql(
        """
        SELECT id AS entity_id, status, due_at
        FROM orders
        WHERE placed_at <= %(t0)s
        """,
        conn,
        params={"t0": t0},
    )
    t0_ts = pd.Timestamp(t0)
    df["days_to_due"] = (pd.to_datetime(df["due_at"], utc=True) - t0_ts).dt.total_seconds() / 86400
    df = pd.concat([df, _one_hot(df["status"], ORDER_STATUSES, "status")], axis=1)
    return df.drop(columns=["status", "due_at"])


NODE_FEATURE_BUILDERS = {
    "Supplier": supplier_features_asof,
    "Component": component_features_asof,
    "Product": product_features_asof,
    "Factory": factory_features_asof,
    "Warehouse": warehouse_features_asof,
    "Shipment": shipment_features_asof,
    "Order": order_features_asof,
    "Customer": customer_features_asof,
}


ENTITY_FEATURE_BUILDERS = {
    "supplier": supplier_features_asof,
    "shipment": shipment_features_asof,
    "product": product_features_asof,
    "order": order_customer_features_asof,
}
