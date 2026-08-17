"""
Build one heterogeneous PyG graph per t0 snapshot, honoring the dataset's
temporal-integrity contract (see db/README.md "Temporal integrity"):

  - every feature is computed strictly from data at or before t0
  - `suppliers.reliability_history` is NEVER read (display-only, leaky per
    schema.sql comment -- use supplier_temporal_features instead)
  - `inventory.stock_level`/`reorder_threshold` are CURRENT state, not as-of
    t0 -- shortage-relevant stock features come from inventory_history via
    an as-of join instead
  - `shipments.status` and `shipments.delivered_at` are NEVER read as
    features (schema.sql: "status ... display only ... reading it as a
    feature is leakage", "delivered_at ... LABEL COMPONENT -- never a model
    feature"). Shipment state is reconstructed from
    shipment_status_history, filtered to changed_at <= t0.

Node types : supplier, component, product, factory, warehouse, customer,
             inventory (= one product@warehouse slot), order, shipment
Edge types : SUPPLIES        supplier  -> component
             QUALIFIED       supplier  -> component   (component_suppliers, secondary)
             USED_IN         component -> product      (BOM, as-of validity window)
             MANUFACTURED_AT product   -> factory       (as-of validity window)
             HAS_STOCK       product   -> inventory
             STORED_AT       inventory -> warehouse
             PLACED          customer  -> order         (as-of: placed_at <= t0)
             CONTAINS        order     -> product       (order_items, as-of)
             DISPATCHED_FROM shipment  -> supplier / factory
             DELIVERS_TO     shipment  -> warehouse
             FOR_ORDER       shipment  -> order

All edges also get a reverse relation added via T.ToUndirected() at the end
so attention can flow both ways.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


def _asof_join(entities: pd.DataFrame, id_col: str, history: pd.DataFrame,
                time_col: str, t0: pd.Timestamp, value_cols: list[str]) -> pd.DataFrame:
    """For each id in `entities`, take the latest `history` row with
    time_col <= t0. Returns one row per id (NaN value_cols if no history yet)."""
    h = history[history[time_col] <= t0].sort_values([id_col, time_col])
    latest = h.groupby(id_col, as_index=False).last()[[id_col] + value_cols]
    out = entities[[id_col]].merge(latest, on=id_col, how="left")
    return out


def _one_hot(series: pd.Series, categories: list[str]) -> np.ndarray:
    codes = pd.Categorical(series.fillna("__missing__"), categories=categories + ["__missing__"])
    return np.eye(len(categories) + 1, dtype=np.float32)[codes.codes]


class IDMaps:
    """Stable UUID -> row-index maps, built once over the full (static) tables
    so indices are consistent across every snapshot."""
    def __init__(self, dfs):
        self.supplier = {u: i for i, u in enumerate(dfs["suppliers"]["id"])}
        self.component = {u: i for i, u in enumerate(dfs["components"]["id"])}
        self.product = {u: i for i, u in enumerate(dfs["products"]["id"])}
        self.factory = {u: i for i, u in enumerate(dfs["factories"]["id"])}
        self.warehouse = {u: i for i, u in enumerate(dfs["warehouses"]["id"])}
        self.customer = {u: i for i, u in enumerate(dfs["customers"]["id"])}
        self.inventory = {u: i for i, u in enumerate(dfs["inventory"]["id"])}
        # order / shipment sets grow over time -> maps are built per snapshot instead


def build_snapshot(dfs: dict, ids: IDMaps, t0: pd.Timestamp, horizon_days: int = 14) -> HeteroData:
    data = HeteroData()
    t0 = pd.Timestamp(t0)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")

    suppliers, components, products = dfs["suppliers"], dfs["components"], dfs["products"]
    factories, warehouses, customers = dfs["factories"], dfs["warehouses"], dfs["customers"]
    inventory = dfs["inventory"]

    # ---------------- static node features ----------------
    # supplier: non-leaky static attrs + as-of temporal features
    stf_src = dfs["supplier_temporal_features"].drop(columns=["id"]).rename(columns={"supplier_id": "id"})
    stf = _asof_join(suppliers, "id", stf_src,
                      "as_of_date", t0,
                      ["on_time_rate_30d", "on_time_rate_90d", "on_time_rate_180d",
                       "trend_slope", "lateness_variance", "days_since_last_late", "shipment_count_180d"])
    sup_feats = np.concatenate([
        suppliers[["capacity_score", "lead_time_days"]].to_numpy(dtype=np.float32),
        stf.drop(columns="id").fillna(0.0).to_numpy(dtype=np.float32),
        stf.drop(columns="id").isna().to_numpy(dtype=np.float32),  # missingness flags
    ], axis=1)
    data["supplier"].x = torch.tensor(sup_feats, dtype=torch.float)
    data["supplier"].node_uuid = suppliers["id"].tolist()

    comp_types = sorted(components["component_type"].dropna().unique().tolist())
    comp_feats = np.concatenate([
        _one_hot(components["component_type"], comp_types),
        components[["unit_cost"]].fillna(0.0).to_numpy(dtype=np.float32),
    ], axis=1)
    data["component"].x = torch.tensor(comp_feats, dtype=torch.float)
    data["component"].node_uuid = components["id"].tolist()

    prod_cats = sorted(products["category"].dropna().unique().tolist())
    prod_feats = _one_hot(products["category"], prod_cats)
    data["product"].x = torch.tensor(prod_feats, dtype=torch.float)
    data["product"].node_uuid = products["id"].tolist()

    data["factory"].x = torch.tensor(
        factories[["capacity_units_per_day"]].fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float)
    data["factory"].node_uuid = factories["id"].tolist()

    data["warehouse"].x = torch.tensor(
        warehouses[["capacity_units"]].fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float)
    data["warehouse"].node_uuid = warehouses["id"].tolist()

    tiers = ["strategic", "standard", "low"]
    data["customer"].x = torch.tensor(_one_hot(customers["priority_tier"], tiers), dtype=torch.float)
    data["customer"].node_uuid = customers["id"].tolist()

    # inventory node features: AS-OF stock/threshold from inventory_history, NEVER
    # from the `inventory` table's current columns (leaky -- "current state only")
    invh_src = dfs["inventory_history"].drop(columns=["id"]).rename(columns={"inventory_id": "id"})
    inv_hist_feats = _asof_join(inventory, "id", invh_src,
                                 "observed_at", t0, ["stock_level", "reorder_threshold"])
    has_history = inv_hist_feats["stock_level"].notna().to_numpy(dtype=np.float32)
    inv_feats = np.concatenate([
        inv_hist_feats[["stock_level", "reorder_threshold"]].fillna(0.0).to_numpy(dtype=np.float32),
        has_history.reshape(-1, 1),
    ], axis=1)
    data["inventory"].x = torch.tensor(inv_feats, dtype=torch.float)
    data["inventory"].node_uuid = inventory["id"].tolist()
    data["inventory"].product_id = inventory["product_id"].tolist()
    data["inventory"].warehouse_id = inventory["warehouse_id"].tolist()

    # ---------------- orders (existence + features as-of t0) ----------------
    ord_mask = dfs["orders"]["placed_at"] <= t0
    orders = dfs["orders"][ord_mask].reset_index(drop=True)
    order_id_map = {u: i for i, u in enumerate(orders["id"])}
    days_to_due = (orders["due_at"] - t0).dt.total_seconds() / 86400.0
    order_feats = np.concatenate([
        orders[["order_value"]].fillna(0.0).to_numpy(dtype=np.float32),
        days_to_due.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1),
    ], axis=1)
    data["order"].x = torch.tensor(order_feats, dtype=torch.float)
    data["order"].node_uuid = orders["id"].tolist()

    # ---------------- shipments (existence + as-of-reconstructed status) ----------------
    ship_mask = dfs["shipments"]["created_at"] <= t0
    shipments = dfs["shipments"][ship_mask].reset_index(drop=True)
    ship_id_map = {u: i for i, u in enumerate(shipments["id"])}

    ssh = dfs["shipment_status_history"].drop(columns=["id"]).rename(columns={"shipment_id": "id"})
    asof_status = _asof_join(shipments, "id", ssh, "changed_at", t0, ["status"])
    status_cats = ["scheduled", "in_transit", "delivered", "delayed"]
    status_oh = _one_hot(asof_status["status"], status_cats)
    days_since_dispatch = (t0 - shipments["dispatched_at"]).dt.total_seconds() / 86400.0
    days_since_dispatch = days_since_dispatch.clip(lower=0).fillna(-1.0)
    days_to_eta = (shipments["eta"] - t0).dt.total_seconds() / 86400.0
    ship_feats = np.concatenate([
        status_oh,
        days_since_dispatch.to_numpy(dtype=np.float32).reshape(-1, 1),
        days_to_eta.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1),
    ], axis=1)
    data["shipment"].x = torch.tensor(ship_feats, dtype=torch.float)
    data["shipment"].node_uuid = shipments["id"].tolist()

    # ---------------- edges ----------------
    def edge_index(src_ids, src_map, dst_ids, dst_map):
        pairs = [(src_map[s], dst_map[d]) for s, d in zip(src_ids, dst_ids)
                 if s in src_map and d in dst_map and pd.notna(s) and pd.notna(d)]
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        arr = np.array(pairs, dtype=np.int64).T
        return torch.tensor(arr, dtype=torch.long)

    data["supplier", "supplies", "component"].edge_index = edge_index(
        components["supplier_id"], ids.supplier, components["id"], ids.component)

    cs = dfs["component_suppliers"]
    cs_live = cs[(cs["created_at"] <= t0) & (cs["deactivated_at"].isna() | (cs["deactivated_at"] > t0))]
    data["supplier", "qualified_for", "component"].edge_index = edge_index(
        cs_live["supplier_id"], ids.supplier, cs_live["component_id"], ids.component)

    pc = dfs["product_components"]
    pc_live = pc[(pc["created_at"] <= t0) & (pc["deactivated_at"].isna() | (pc["deactivated_at"] > t0))]
    data["component", "used_in", "product"].edge_index = edge_index(
        pc_live["component_id"], ids.component, pc_live["product_id"], ids.product)

    pf = dfs["product_factories"]
    pf_live = pf[(pf["created_at"] <= t0) & (pf["deactivated_at"].isna() | (pf["deactivated_at"] > t0))]
    data["product", "manufactured_at", "factory"].edge_index = edge_index(
        pf_live["product_id"], ids.product, pf_live["factory_id"], ids.factory)

    data["product", "has_stock", "inventory"].edge_index = edge_index(
        inventory["product_id"], ids.product, inventory["id"], ids.inventory)
    data["inventory", "stored_at", "warehouse"].edge_index = edge_index(
        inventory["id"], ids.inventory, inventory["warehouse_id"], ids.warehouse)

    data["customer", "placed", "order"].edge_index = edge_index(
        orders["customer_id"], ids.customer, orders["id"], order_id_map)

    oi = dfs["order_items"]
    oi_live = oi[(oi["created_at"] <= t0) & (oi["order_id"].isin(order_id_map))]
    data["order", "contains", "product"].edge_index = edge_index(
        oi_live["order_id"], order_id_map, oi_live["product_id"], ids.product)

    data["shipment", "dispatched_from_supplier", "supplier"].edge_index = edge_index(
        shipments["id"], ship_id_map, shipments["supplier_id"], ids.supplier)
    data["shipment", "dispatched_from_factory", "factory"].edge_index = edge_index(
        shipments["id"], ship_id_map, shipments["factory_id"], ids.factory)
    data["shipment", "delivers_to", "warehouse"].edge_index = edge_index(
        shipments["id"], ship_id_map, shipments["warehouse_id"], ids.warehouse)
    data["shipment", "for_order", "order"].edge_index = edge_index(
        shipments["id"], ship_id_map, shipments["order_id"], order_id_map)

    # ---------------- labels ----------------
    for nt in ["shipment", "inventory", "supplier"]:
        n = data[nt].x.shape[0]
        data[nt].y_delay = torch.full((n,), float("nan")) if nt == "shipment" else None
    data["shipment"].y = torch.full((len(shipments),), float("nan"))
    data["inventory"].y = torch.full((len(inventory),), float("nan"))
    data["supplier"].y = torch.full((len(suppliers),), float("nan"))

    return data, order_id_map, ship_id_map


def attach_labels(data: HeteroData, ids: IDMaps, order_id_map, ship_id_map,
                   labels: pd.DataFrame, inventory_df: pd.DataFrame):
    inv_key_to_idx = {(pid, wid): i for i, (pid, wid) in
                       enumerate(zip(inventory_df["product_id"], inventory_df["warehouse_id"]))}

    delay = labels[labels["task"] == "delay"]
    y = data["shipment"].y.clone()
    for eid, lab in zip(delay["entity_id"], delay["label"]):
        if eid in ship_id_map:
            y[ship_id_map[eid]] = float(lab)
    data["shipment"].y = y

    shortage = labels[labels["task"] == "shortage"]
    y = data["inventory"].y.clone()
    for pid, wid, lab in zip(shortage["entity_id"], shortage["warehouse_id"], shortage["label"]):
        key = (pid, wid)
        if key in inv_key_to_idx:
            y[inv_key_to_idx[key]] = float(lab)
    data["inventory"].y = y

    impact = labels[labels["task"] == "impact"]
    y = data["supplier"].y.clone()
    for eid, lab in zip(impact["entity_id"], impact["label"]):
        if eid in ids.supplier:
            y[ids.supplier[eid]] = float(lab)
    data["supplier"].y = y

    return data


def finalize(data: HeteroData) -> HeteroData:
    data = T.ToUndirected(merge=False)(data)
    return data
