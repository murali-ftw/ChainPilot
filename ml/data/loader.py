"""
V2 data-loading layer: `db/csv/v<variant>_seed<n>/*.csv.gz` -> the `HeteroData`
bundles SHARE trains on.

V1 assembled snapshots by querying PostgreSQL (`HADES_v1/ml/graph/builder.py`,
`HADES_v1/ml/data/features.py`). V2's benchmark artifact is the gzipped CSV
directory, not a database, so this module reproduces those SQL semantics
directly over pandas. The feature definitions are ported one-for-one -- same
columns, same z-scores, same one-hot category sets, same as-of rules -- because
step 2 of this build compares the ported model's numbers against V1's recorded
ones, and a feature that quietly changed would make that comparison meaningless.

Three things genuinely differ from V1, all forced by V2 and all deliberate:

1. **The as-of clock is `recorded_at`, not `changed_at`.** V1's shipment-status
   SQL read `WHERE changed_at <= t0` because V1 had no reporting delay. Under
   Mechanism G the two clocks diverge on purpose: `changed_at` is true event
   time (which drives labels) and `recorded_at` is the earliest moment a model
   may know about the transition. Reading `changed_at` here would hand the model
   exactly the information G exists to withhold, so this loader reads
   `recorded_at`. On a variant without G the two differ only by the generator's
   5-45 minute write jitter, which can only matter for a transition recorded
   inside that window of a t0 boundary -- `--report-clock-delta` counts them.
2. **`supplier_upstream` becomes an eleventh forward relation**,
   `(Supplier, UPSTREAM_OF, Supplier)`. Mechanism J emits it; Mechanism A
   truncates it at the visibility frontier (the generator only writes an edge
   when both endpoints survive). Variants without J emit no such file and simply
   have no such relation -- the loader does not fabricate an empty one, since an
   empty relation still costs the RGCN family a coefficient row.
3. **Mechanism A shrinks the node set, not just the features.** Hidden suppliers
   are absent from `suppliers.csv` entirely, so any edge pointing at one is
   dropped by the same id-map filter V1 used for partial loads. That is the
   observable-graph truncation, working as specified.

**Hidden state is not read because it is not there.** Mechanism E's resilience,
Mechanism F's attenuation, Mechanism B's hidden-parent identity and Mechanism D's
coupling term are never emitted to CSV by design. This loader names its columns
explicitly (never `SELECT *`), so a column that does not exist cannot be picked
up by accident, and `verify_no_hidden_state()` asserts the emitted files carry
none of the hidden-state column names.
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

FEATURE_SPEC_VERSION = "v2"

SHIPMENT_STATUSES = ["scheduled", "in_transit", "delivered", "delayed"]
CUSTOMER_TIERS = ["strategic", "standard", "low"]
ORDER_STATUSES = ["open", "fulfilled", "cancelled", "at_risk"]

NODE_TYPES = ["Supplier", "Component", "Product", "Factory", "Warehouse",
              "Shipment", "Order", "Customer"]

TASK_ENTITY_TYPE = {"delay": "Shipment", "shortage": "Product", "impact": "Supplier"}
TASK_CSV_ENTITY = {"delay": "shipment", "shortage": "product", "impact": "supplier"}

# Column names that would indicate a mechanism's hidden state leaked into an
# emitted table. None of these exist in any V2 CSV; the check is here so that a
# future generator change that starts emitting one is caught by the loader
# rather than silently improving every model's score.
HIDDEN_STATE_COLUMNS = {
    "resilience", "resilience_value", "absorption", "attenuation", "recv_atten",
    "hidden_parent", "hidden_parent_id", "hidden_parent_type", "group_type",
    "alpha", "coupling", "upstream_stress", "own_stress", "stress", "mitigation",
}


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _read(path_dir: str, name: str, usecols=None, parse_dates=(), dtype=None) -> pd.DataFrame:
    """Read `<name>.csv.gz`, falling back to a plain `.csv`, like load_data.py."""
    gz, plain = os.path.join(path_dir, name + ".csv.gz"), os.path.join(path_dir, name + ".csv")
    path = gz if os.path.exists(gz) else plain
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    return df


def verify_no_hidden_state(path_dir: str) -> list[str]:
    """Header-only scan of every emitted table for a hidden-state column name.
    Returns the offending `table.column` strings; empty is the expected result."""
    bad = []
    for fname in sorted(os.listdir(path_dir)):
        if not fname.endswith(".csv.gz"):
            continue
        with gzip.open(os.path.join(path_dir, fname), "rt") as fh:
            header = fh.readline().strip().split(",")
        for col in header:
            if col in HIDDEN_STATE_COLUMNS:
                bad.append(f"{fname[:-7]}.{col}")
    return bad


# ---------------------------------------------------------------------------
# Feature helpers -- ported from HADES_v1/ml/data/features.py
# ---------------------------------------------------------------------------

def _one_hot(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(series, categories=categories)
    return pd.get_dummies(cat, prefix=prefix, dtype=float)


def _zscore(series: pd.Series) -> pd.Series:
    mean, std = series.mean(), series.std(ddof=0)
    if not std or np.isnan(std):
        return series * 0.0
    return (series - mean) / std


def _naive(series: pd.Series) -> np.ndarray:
    """tz-aware column -> naive `datetime64[ns]` array. Every timestamp in the
    corpus is UTC, so dropping the zone is lossless and lets the as-of masks be
    plain numpy comparisons instead of per-element Timestamp comparisons."""
    return series.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")


def _location_bucket(location: pd.Series) -> pd.Series:
    return location.fillna("unknown").str.split(",").str[-1].str.strip()


def _last_per_group(group_codes: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Row positions of the LAST kept row per group, for an array already sorted
    by (group, time). Vectorised stand-in for `DISTINCT ON (g) ... ORDER BY g,
    t DESC`: among the kept rows, a row is the group's last one exactly when the
    next kept row belongs to a different group."""
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return idx
    g = group_codes[idx]
    is_last = np.empty(g.shape, dtype=bool)
    is_last[-1] = True
    is_last[:-1] = g[1:] != g[:-1]
    return idx[is_last]


# ---------------------------------------------------------------------------
# The variant dataset
# ---------------------------------------------------------------------------

class VariantDataset:
    """Every table of one variant-seed, read once, plus the derived as-of
    machinery each snapshot needs. Build snapshots with `bundles()`."""

    def __init__(self, path_dir: str, asof_clock: str = "recorded"):
        if asof_clock not in ("recorded", "changed"):
            raise ValueError("asof_clock must be 'recorded' or 'changed'")
        self.dir = path_dir
        self.asof_clock = asof_clock
        with open(os.path.join(path_dir, "resolved_config.json")) as fh:
            self.manifest = json.load(fh)

        self.suppliers = _read(path_dir, "suppliers",
                               usecols=["id", "country", "capacity_score", "lead_time_days"])
        self.components = _read(path_dir, "components",
                                usecols=["id", "supplier_id", "component_type", "unit_cost"])
        self.component_suppliers = _read(path_dir, "component_suppliers",
                                         usecols=["id", "component_id", "supplier_id",
                                                  "created_at", "deactivated_at"],
                                         parse_dates=("created_at", "deactivated_at"))
        self.supplier_upstream = _read(path_dir, "supplier_upstream",
                                       usecols=["supplier_id", "upstream_supplier_id",
                                                "created_at", "deactivated_at"],
                                       parse_dates=("created_at", "deactivated_at"))
        self.products = _read(path_dir, "products", usecols=["id", "category"])
        self.product_components = _read(path_dir, "product_components",
                                        usecols=["product_id", "component_id",
                                                 "quantity_required", "created_at", "deactivated_at"],
                                        parse_dates=("created_at", "deactivated_at"))
        self.factories = _read(path_dir, "factories",
                               usecols=["id", "capacity_units_per_day", "location"])
        self.product_factories = _read(path_dir, "product_factories",
                                       usecols=["product_id", "factory_id",
                                                "capacity_units_per_day", "created_at", "deactivated_at"],
                                       parse_dates=("created_at", "deactivated_at"))
        self.warehouses = _read(path_dir, "warehouses", usecols=["id", "capacity_units", "location"])
        self.customers = _read(path_dir, "customers", usecols=["id", "priority_tier"])
        self.orders = _read(path_dir, "orders",
                            usecols=["id", "customer_id", "status", "placed_at", "due_at"],
                            parse_dates=("placed_at", "due_at"))
        self.order_items = _read(path_dir, "order_items",
                                 usecols=["order_id", "product_id", "quantity", "created_at"],
                                 parse_dates=("created_at",))
        self.shipments = _read(path_dir, "shipments",
                               usecols=["id", "supplier_id", "factory_id", "warehouse_id",
                                        "order_id", "carrier", "eta", "dispatched_at", "created_at"],
                               parse_dates=("eta", "dispatched_at", "created_at"))
        self.carrier_perf = _read(path_dir, "carrier_performance_snapshots",
                                  usecols=["carrier", "as_of_date", "on_time_rate_90d"],
                                  parse_dates=("as_of_date",))
        self.stf = _read(path_dir, "supplier_temporal_features",
                         usecols=["supplier_id", "as_of_date", "on_time_rate_30d",
                                  "on_time_rate_90d", "on_time_rate_180d", "trend_slope",
                                  "lateness_variance", "days_since_last_late"],
                         parse_dates=("as_of_date",))
        self.snapshots = _read(path_dir, "graph_snapshots",
                               usecols=["id", "t0", "horizon_days"], parse_dates=("t0",))
        self.snapshots = self.snapshots.sort_values("t0").reset_index(drop=True)
        self.labels = _read(path_dir, "training_labels",
                            usecols=["snapshot_id", "entity_type", "entity_id", "task",
                                     "label", "event_at", "warehouse_id"],
                            parse_dates=("event_at",))
        self._prepare_inventory_history()
        self._prepare_status_history()
        self._prepare_static_features()
        self._id_maps_static = {
            "Supplier": {v: i for i, v in enumerate(self.suppliers["id"])},
            "Component": {v: i for i, v in enumerate(self.components["id"])},
            "Product": {v: i for i, v in enumerate(self.products["id"])},
            "Factory": {v: i for i, v in enumerate(self.factories["id"])},
            "Warehouse": {v: i for i, v in enumerate(self.warehouses["id"])},
            "Customer": {v: i for i, v in enumerate(self.customers["id"])},
        }

    # -- as-of preparation ------------------------------------------------

    def _prepare_inventory_history(self) -> None:
        """Sorted by (inventory pair, observed_at) once, so each t0's
        `DISTINCT ON (inventory_id) ... ORDER BY observed_at DESC` is a
        boolean mask plus a last-per-group scan instead of a fresh groupby."""
        ih = _read(self.dir, "inventory_history",
                   usecols=["product_id", "warehouse_id", "stock_level",
                            "reorder_threshold", "observed_at", "recorded_at"],
                   parse_dates=("observed_at", "recorded_at"))
        ih["pair"] = ih["product_id"] + "|" + ih["warehouse_id"]
        ih = ih.sort_values(["pair", "observed_at"], kind="stable").reset_index(drop=True)
        self.ih_pair_codes = pd.Categorical(ih["pair"]).codes
        self.ih_obs = _naive(ih["observed_at"])
        self.ih_rec = _naive(ih["recorded_at"])
        self.ih_product = ih["product_id"].to_numpy()
        self.ih_warehouse = ih["warehouse_id"].to_numpy()
        self.ih_stock = ih["stock_level"].to_numpy(dtype=float)
        self.ih_thr = ih["reorder_threshold"].to_numpy(dtype=float)

    def _prepare_status_history(self) -> None:
        ssh = _read(self.dir, "shipment_status_history",
                    usecols=["shipment_id", "status", "changed_at", "recorded_at"],
                    parse_dates=("changed_at", "recorded_at"))
        clock = "recorded_at" if self.asof_clock == "recorded" else "changed_at"
        ssh = ssh.sort_values(["shipment_id", clock], kind="stable").reset_index(drop=True)
        self.ssh_ship_codes = pd.Categorical(ssh["shipment_id"]).codes
        self.ssh_ship = ssh["shipment_id"].to_numpy()
        self.ssh_status = ssh["status"].to_numpy()
        self.ssh_clock = _naive(ssh[clock])
        self.ssh_changed = _naive(ssh["changed_at"])
        self.ssh_recorded = _naive(ssh["recorded_at"])

    def clock_delta_rows(self, t0s) -> int:
        """How many transitions are visible under `changed_at <= t0` but not
        under `recorded_at <= t0`, summed over the snapshot schedule -- the exact
        size of deviation 1 in this module's docstring."""
        total = 0
        for t0 in t0s:
            t = np.datetime64(pd.Timestamp(t0).tz_convert(None))
            total += int(((self.ssh_changed <= t) & (self.ssh_recorded > t)).sum())
        return total

    # -- static (t0-invariant) node features -------------------------------

    def _prepare_static_features(self) -> None:
        countries = sorted(self.suppliers["country"].dropna().unique())
        sup = self.suppliers[["id"]].rename(columns={"id": "entity_id"}).copy()
        sup["lead_time_days_z"] = _zscore(self.suppliers["lead_time_days"].astype(float))
        sup["capacity_score_z"] = _zscore(self.suppliers["capacity_score"].astype(float))
        self.supplier_static = pd.concat(
            [sup, _one_hot(self.suppliers["country"], countries, "country")], axis=1)

        ctypes = sorted(self.components["component_type"].dropna().unique())
        comp = self.components[["id"]].rename(columns={"id": "entity_id"}).copy()
        comp["unit_cost_z"] = _zscore(self.components["unit_cost"].astype(float))
        self.component_features = pd.concat(
            [comp, _one_hot(self.components["component_type"], ctypes, "component_type")], axis=1)

        fac = self.factories[["id"]].rename(columns={"id": "entity_id"}).copy()
        bucket = _location_bucket(self.factories["location"])
        fac["capacity_units_per_day_z"] = _zscore(self.factories["capacity_units_per_day"].astype(float))
        self.factory_features = pd.concat(
            [fac, _one_hot(bucket, sorted(bucket.unique()), "location")], axis=1)

        wh = self.warehouses[["id"]].rename(columns={"id": "entity_id"}).copy()
        bucket = _location_bucket(self.warehouses["location"])
        wh["capacity_units_z"] = _zscore(self.warehouses["capacity_units"].astype(float))
        self.warehouse_features = pd.concat(
            [wh, _one_hot(bucket, sorted(bucket.unique()), "location")], axis=1)

        cust = self.customers[["id"]].rename(columns={"id": "entity_id"}).copy()
        self.customer_features = pd.concat(
            [cust, _one_hot(self.customers["priority_tier"], CUSTOMER_TIERS, "priority_tier")], axis=1)

        self.product_categories = sorted(self.products["category"].dropna().unique())

    # -- per-t0 node features ---------------------------------------------

    def _supplier_features(self, t0: pd.Timestamp) -> pd.DataFrame:
        temporal = self.stf[self.stf["as_of_date"] <= t0]
        temporal = (temporal.sort_values(["supplier_id", "as_of_date"], kind="stable")
                    .groupby("supplier_id", as_index=False).tail(1)
                    .rename(columns={"supplier_id": "entity_id"})
                    .drop(columns=["as_of_date"]))
        return self.supplier_static.merge(temporal, on="entity_id", how="left")

    def _latest_inventory(self, t0: pd.Timestamp) -> pd.DataFrame:
        t = np.datetime64(t0.tz_convert(None))
        keep = (self.ih_obs <= t) & (self.ih_rec <= t)
        rows = _last_per_group(self.ih_pair_codes, keep)
        return pd.DataFrame({
            "product_id": self.ih_product[rows],
            "warehouse_id": self.ih_warehouse[rows],
            "stock_level": self.ih_stock[rows],
            "reorder_threshold": self.ih_thr[rows],
        })

    def _product_features(self, t0: pd.Timestamp, latest_inv: pd.DataFrame) -> pd.DataFrame:
        base = self.products[["id"]].rename(columns={"id": "entity_id"}).copy()
        inv = latest_inv.copy()
        ratio = inv["stock_level"] / inv["reorder_threshold"].replace(0, np.nan)
        inv["ratio"] = ratio
        agg = inv.groupby("product_id").agg(
            min_stock_ratio=("ratio", "min"), avg_stock_ratio=("ratio", "mean"),
            total_stock=("stock_level", "sum"),
            total_reorder_threshold=("reorder_threshold", "sum"),
            warehouse_count=("ratio", "size")).reset_index().rename(columns={"product_id": "entity_id"})
        pc = self.product_components
        active = pc[(pc["created_at"] <= t0) & (pc["deactivated_at"].isna() | (pc["deactivated_at"] > t0))]
        bom = active.groupby("product_id").agg(
            bom_component_count=("component_id", "size"),
            bom_mean_quantity_required=("quantity_required", "mean")).reset_index().rename(
            columns={"product_id": "entity_id"})
        df = base.merge(agg, on="entity_id", how="left").merge(bom, on="entity_id", how="left")
        return pd.concat([df, _one_hot(self.products["category"], self.product_categories, "category")],
                         axis=1)

    def _shipment_features(self, t0: pd.Timestamp) -> pd.DataFrame:
        ships = self.shipments[self.shipments["created_at"] <= t0]
        t = np.datetime64(t0.tz_convert(None))
        keep = self.ssh_clock <= t
        rows = _last_per_group(self.ssh_ship_codes, keep)
        status = pd.DataFrame({"entity_id": self.ssh_ship[rows], "status": self.ssh_status[rows]})

        perf = self.carrier_perf[self.carrier_perf["as_of_date"] <= t0]
        perf = (perf.groupby(["carrier", "as_of_date"], as_index=False)["on_time_rate_90d"].mean()
                .sort_values(["carrier", "as_of_date"], kind="stable")
                .groupby("carrier", as_index=False).tail(1)
                .rename(columns={"on_time_rate_90d": "carrier_on_time_rate_90d"})
                .drop(columns=["as_of_date"]))

        df = ships[["id", "eta", "dispatched_at", "carrier"]].rename(columns={"id": "entity_id"}).copy()
        df = df.merge(status, on="entity_id", how="left").merge(perf, on="carrier", how="left")
        df["days_to_eta"] = (df["eta"] - t0).dt.total_seconds() / 86400
        since = (t0 - df["dispatched_at"]).dt.total_seconds() / 86400
        df["days_since_dispatch"] = since.where(df["dispatched_at"] <= t0)
        df = pd.concat([df, _one_hot(df["status"], SHIPMENT_STATUSES, "status")], axis=1)
        return df.drop(columns=["eta", "dispatched_at", "carrier", "status"])

    def _order_features(self, t0: pd.Timestamp) -> pd.DataFrame:
        orders = self.orders[self.orders["placed_at"] <= t0]
        df = orders[["id", "status", "due_at"]].rename(columns={"id": "entity_id"}).copy()
        df["days_to_due"] = (df["due_at"] - t0).dt.total_seconds() / 86400
        df = pd.concat([df, _one_hot(df["status"], ORDER_STATUSES, "status")], axis=1)
        return df.drop(columns=["status", "due_at"])

    # -- edges -------------------------------------------------------------

    def _edges(self, t0: pd.Timestamp, id_maps: dict, latest_inv: pd.DataFrame) -> dict:
        def pack(src_ids, dst_ids, src_type, dst_type, attrs=None):
            s = pd.Series(src_ids).map(id_maps[src_type])
            d = pd.Series(dst_ids).map(id_maps[dst_type])
            ok = s.notna() & d.notna()
            ei = torch.tensor(np.vstack([s[ok].to_numpy(dtype=np.int64),
                                         d[ok].to_numpy(dtype=np.int64)]), dtype=torch.long)
            ea = None
            if attrs is not None:
                ea = torch.tensor(np.asarray(attrs)[ok.to_numpy()], dtype=torch.float)
            return ei, ea

        e = {}
        # SUPPLIES: mandatory primary supplier UNION ALL active secondary
        # sources from component_suppliers, exactly as V1's SQL does.
        cs = self.component_suppliers
        cs_active = cs[(cs["created_at"] <= t0) & (cs["deactivated_at"].isna() | (cs["deactivated_at"] > t0))]
        src = np.concatenate([self.components["supplier_id"].to_numpy(), cs_active["supplier_id"].to_numpy()])
        dst = np.concatenate([self.components["id"].to_numpy(), cs_active["component_id"].to_numpy()])
        e[("Supplier", "SUPPLIES", "Component")] = pack(src, dst, "Supplier", "Component")

        pc = self.product_components
        act = pc[(pc["created_at"] <= t0) & (pc["deactivated_at"].isna() | (pc["deactivated_at"] > t0))]
        e[("Component", "USED_IN", "Product")] = pack(
            act["component_id"].to_numpy(), act["product_id"].to_numpy(), "Component", "Product",
            act[["quantity_required"]].astype(float).to_numpy())

        e[("Product", "STOCKED_AT", "Warehouse")] = pack(
            latest_inv["product_id"].to_numpy(), latest_inv["warehouse_id"].to_numpy(),
            "Product", "Warehouse",
            latest_inv[["stock_level", "reorder_threshold"]].astype(float).to_numpy())

        pf = self.product_factories
        act = pf[(pf["created_at"] <= t0) & (pf["deactivated_at"].isna() | (pf["deactivated_at"] > t0))]
        e[("Product", "MANUFACTURED_AT", "Factory")] = pack(
            act["product_id"].to_numpy(), act["factory_id"].to_numpy(), "Product", "Factory",
            act[["capacity_units_per_day"]].astype(float).to_numpy())

        ships = self.shipments[self.shipments["created_at"] <= t0]
        has_sup = ships[ships["supplier_id"].notna()]
        e[("Shipment", "SHIPS_FROM", "Supplier")] = pack(
            has_sup["id"].to_numpy(), has_sup["supplier_id"].to_numpy(), "Shipment", "Supplier")
        has_fac = ships[ships["factory_id"].notna()]
        e[("Shipment", "SHIPS_FROM", "Factory")] = pack(
            has_fac["id"].to_numpy(), has_fac["factory_id"].to_numpy(), "Shipment", "Factory")
        has_wh = ships[ships["warehouse_id"].notna()]
        e[("Shipment", "SHIPS_TO", "Warehouse")] = pack(
            has_wh["id"].to_numpy(), has_wh["warehouse_id"].to_numpy(), "Shipment", "Warehouse",
            ((has_wh["eta"] - t0).dt.total_seconds() / 86400).to_numpy().reshape(-1, 1))
        has_ord = ships[ships["order_id"].notna()]
        e[("Shipment", "FULFILLS", "Order")] = pack(
            has_ord["id"].to_numpy(), has_ord["order_id"].to_numpy(), "Shipment", "Order")

        oi = self.order_items[self.order_items["created_at"] <= t0]
        e[("Order", "ORDERED", "Product")] = pack(
            oi["order_id"].to_numpy(), oi["product_id"].to_numpy(), "Order", "Product",
            oi[["quantity"]].astype(float).to_numpy())

        orders = self.orders[self.orders["placed_at"] <= t0]
        e[("Order", "PLACED_BY", "Customer")] = pack(
            orders["id"].to_numpy(), orders["customer_id"].to_numpy(), "Order", "Customer")

        # V2: Mechanism J's upstream tiers, already truncated by Mechanism A in
        # the generator. Absent for variants without J -- no empty relation.
        su = self.supplier_upstream
        if not su.empty:
            act = su[(su["created_at"] <= t0) & (su["deactivated_at"].isna() | (su["deactivated_at"] > t0))]
            if not act.empty:
                e[("Supplier", "UPSTREAM_OF", "Supplier")] = pack(
                    act["upstream_supplier_id"].to_numpy(), act["supplier_id"].to_numpy(),
                    "Supplier", "Supplier")
        return e

    # -- assembly ----------------------------------------------------------

    def _tensor(self, df: pd.DataFrame) -> torch.Tensor:
        cols = [c for c in df.columns if c != "entity_id"]
        return torch.tensor(df[cols].astype(float).fillna(0.0).to_numpy(), dtype=torch.float)

    def build_snapshot(self, t0: pd.Timestamp) -> tuple[HeteroData, dict]:
        latest_inv = self._latest_inventory(t0)
        frames = {
            "Supplier": self._supplier_features(t0),
            "Component": self.component_features,
            "Product": self._product_features(t0, latest_inv),
            "Factory": self.factory_features,
            "Warehouse": self.warehouse_features,
            "Shipment": self._shipment_features(t0),
            "Order": self._order_features(t0),
            "Customer": self.customer_features,
        }
        id_maps = dict(self._id_maps_static)
        id_maps["Shipment"] = {v: i for i, v in enumerate(frames["Shipment"]["entity_id"])}
        id_maps["Order"] = {v: i for i, v in enumerate(frames["Order"]["entity_id"])}

        data = HeteroData()
        for nt in NODE_TYPES:
            data[nt].x = self._tensor(frames[nt])
            # Row index -> entity id, so a per-node diagnostic can join model output back
            # to the world that produced it (gate-disagreement analysis needs this; V1's
            # PostgreSQL builder carried the same `node_id` list for the same reason).
            data[nt].node_id = frames[nt]["entity_id"].tolist()
        for key, (ei, ea) in self._edges(t0, id_maps, latest_inv).items():
            data[key].edge_index = ei
            if ea is not None:
                data[key].edge_attr = ea
        return ToUndirected()(data), id_maps

    def snapshot_labels(self, snapshot_id: str, id_maps: dict) -> dict:
        rows = self.labels[self.labels["snapshot_id"] == snapshot_id]
        out = {}
        for task, entity in TASK_CSV_ENTITY.items():
            sub = rows[(rows["task"] == task) & (rows["entity_type"] == entity)]
            if sub.empty:
                out[task] = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float))
                continue
            idx = sub["entity_id"].map(id_maps[TASK_ENTITY_TYPE[task]])
            ok = idx.notna()
            out[task] = (
                torch.tensor(idx[ok].to_numpy(dtype=np.int64), dtype=torch.long),
                torch.tensor((sub.loc[ok.to_numpy(), "label"].astype(str) == "True").to_numpy()
                             .astype(float), dtype=torch.float),
            )
        return out


class SnapshotBundle:
    """One t0: its graph, its per-task (index, label) tensors, its identity."""

    __slots__ = ("t0", "snapshot_id", "data", "labels")

    def __init__(self, t0, snapshot_id, data, labels):
        self.t0, self.snapshot_id, self.data, self.labels = t0, snapshot_id, data, labels

    def to(self, device):
        self.data = self.data.to(device)
        self.labels = {t: (i.to(device), y.to(device)) for t, (i, y) in self.labels.items()}
        return self


def load_bundles(path_dir: str, cache_dir: str | None = None, asof_clock: str = "recorded",
                 verbose: bool = False) -> tuple[list[SnapshotBundle], dict]:
    """Every snapshot of one variant-seed, as bundles, plus its resolved config.

    Assembly is the expensive part (minutes at spec scale) and is identical for
    every architecture and every training seed, so the result is cached to
    `cache_dir` keyed by the variant directory, the feature-spec version and the
    as-of clock. Nine architectures x five model seeds means the cache is read
    45 times for each time it is written."""
    # The key must identify the BUILD, not just its directory name: two different
    # configurations both produce a `vB_seed42` directory, and keying on the basename
    # alone silently serves one configuration's bundles for the other's request. The
    # parent directory disambiguates them (db/csv_v1scale vs db/csv_mid vs db/csv).
    abs_dir = os.path.abspath(path_dir.rstrip("/"))
    key = (f"{os.path.basename(os.path.dirname(abs_dir))}__{os.path.basename(abs_dir)}_"
           f"{FEATURE_SPEC_VERSION}_{asof_clock}_ids.pt")
    cache_path = os.path.join(cache_dir, key) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        return blob["bundles"], blob["manifest"]

    ds = VariantDataset(path_dir, asof_clock=asof_clock)
    bundles = []
    for row in ds.snapshots.itertuples():
        data, id_maps = ds.build_snapshot(row.t0)
        bundles.append(SnapshotBundle(row.t0, row.id, data, ds.snapshot_labels(row.id, id_maps)))
        if verbose:
            print(f"  {row.t0.date()}  nodes={sum(data[nt].num_nodes for nt in data.node_types):>9,}  "
                  f"edges={sum(data[et].edge_index.size(1) for et in data.edge_types):>10,}", flush=True)
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"bundles": bundles, "manifest": ds.manifest}, cache_path)
    return bundles, ds.manifest


def split_bundles(bundles: list[SnapshotBundle], fractions=(0.4, 0.2, 0.4)) -> tuple[list, list, list]:
    """Temporal split by snapshot COUNT, never randomly.

    V1's v3 schedule used 6 train / 3 validation / 6 test out of 15 snapshots --
    40/20/40, which is the proportion carried forward here and applied to
    whatever `SNAPSHOTS` the data on disk actually has (40 -> 16/8/16). The step
    brief quotes "60/20/40", which sums to 120% and cannot be a proportion; V1's
    actual 40/20/40 is used instead, and the discrepancy is recorded in
    `reports/phase7_training_results.md`."""
    n = len(bundles)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return bundles[:n_train], bundles[n_train:n_train + n_val], bundles[n_train + n_val:]
