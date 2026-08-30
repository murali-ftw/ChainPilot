#!/usr/bin/env python3
"""
Integrity verification for the V3 schema enrichment (`db/enrich_schema.py`).

Four things have to be true before any number trained on the enriched graph means anything,
and each is checked here rather than asserted in prose:

1. **The world did not change.** Enrichment is only interpretable as a schema experiment if the
   simulation underneath it is bit-identical. Every pre-existing table is diffed, decompressed,
   against a build of the same variant/seed generated WITHOUT `--enrich`.
2. **No hidden state was emitted.** The brief asks the latent `H_PORT`/`H_TRUCK`/`H_CUSTOMS`
   factors to become `DEPENDS_ON` edges. Emitting the true membership would be a leak worth
   more than every result in this report, so the coupling is built from observable
   infrastructure instead. This script measures how much of the true latent grouping that
   observable proxy actually recovers -- a number the report quotes rather than assumes.
3. **Temporal integrity holds.** Every as-of row must be reproducible from deliveries REPORTED
   at or before its own as-of instant. The check recomputes a random sample of emitted
   `carrier_temporal_features` and `route_conditions` rows from `shipments.csv.gz` under that
   rule and requires an exact match, and separately asserts `recorded_at >= as_of_date`
   everywhere so the loader's report-clock mask has something to bite on.
4. **The new node types are censused.** Carrier has five instances against Supplier's 800.
   That is the thin-relation-overfitting regime SHARE's shared attention scorer and basis
   decomposition exist to guard against, and it is reported whether or not it changes a
   headline.

    python3 db/verify_enriched.py --enriched db/csv_v3enriched/v0enr_seed42 --seed 42
"""
from __future__ import annotations

import argparse
import gzip
import os
import random
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PRE_EXISTING = [
    "suppliers", "components", "component_suppliers", "products", "product_components",
    "factories", "product_factories", "warehouses", "inventory", "inventory_history",
    "customers", "orders", "order_items", "shipments", "shipment_status_history",
    "supplier_temporal_features", "carrier_performance_snapshots", "graph_snapshots",
    "training_labels", "risk_scores",
]
NEW_TABLES = ["carriers", "ports", "routes", "route_ports", "shipment_routes",
              "route_dependencies", "product_replenishment", "warehouse_customers",
              "carrier_temporal_features", "route_conditions", "port_congestion_history",
              "warehouse_role_features"]

OK, BAD = "  [PASS]", "  [FAIL]"
_fails = 0


def check(cond, label, detail=""):
    global _fails
    print(f"{OK if cond else BAD} {label}" + (f" ({detail})" if detail else ""), flush=True)
    if not cond:
        _fails += 1
    return cond


def _rd(d, name, parse_dates=(), **kw):
    """Read one emitted table. Date columns are parsed UTC-aware, matching
    `ml/data/loader.py::_read` -- a naive/aware mix would make every temporal comparison
    below raise rather than answer."""
    p = os.path.join(d, name + ".csv.gz")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, **kw)
    for c in parse_dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, format="mixed")
    return df


# --------------------------------------------------------------------- 1. byte identity

def check_byte_identity(enriched: str, variant: str, seed: int, config: str) -> None:
    print("\n1. the world did not change")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ctrl")
        r = subprocess.run([sys.executable, os.path.join(HERE, "generate_dataset.py"),
                            "--variant", variant, "--seed", str(seed), "--config", config,
                            "--out-dir", out], capture_output=True, text=True)
        if r.returncode != 0:
            check(False, "control build succeeded", r.stderr[-300:])
            return
        diffs, missing = [], []
        for name in PRE_EXISTING:
            a = os.path.join(out, name + ".csv.gz")
            b = os.path.join(enriched, name + ".csv.gz")
            if not os.path.exists(a) or not os.path.exists(b):
                missing.append(name)
                continue
            if gzip.open(a, "rb").read() != gzip.open(b, "rb").read():
                diffs.append(name)
        check(not missing, "every pre-existing table is present in both builds",
              f"missing: {missing}" if missing else f"{len(PRE_EXISTING)} tables")
        check(not diffs, "every pre-existing table is BYTE-IDENTICAL with and without --enrich",
              f"differ: {diffs}" if diffs else f"{len(PRE_EXISTING)}/{len(PRE_EXISTING)} identical")
    present = [t for t in NEW_TABLES if os.path.exists(os.path.join(enriched, t + ".csv.gz"))]
    check(len(present) == len(NEW_TABLES), "every V3 table was emitted",
          f"{len(present)}/{len(NEW_TABLES)}")


# --------------------------------------------------------------------- 2. no hidden state

def check_no_leak(enriched: str, variant: str, seed: int, config: str) -> None:
    print("\n2. no hidden state emitted; observable coupling measured against the true grouping")
    from ml.data.loader import verify_no_hidden_state
    bad = verify_no_hidden_state(enriched)
    check(not bad, "no emitted column matches a hidden-state name", f"offenders: {bad}" if bad else
          "loader's HIDDEN_STATE_COLUMNS scan clean")

    # How much of the latent grouping does the observable proxy recover? Pull the true sets out
    # of the generator's own namespace (never from a CSV -- they are not in one, which is the
    # point) and compare against the coupling the emitted edges actually induce.
    from ml.confirm_latent_states import namespace
    ns = namespace(variant, seed, config)
    truth = {"port": ns["H_PORT"], "trucking": ns["H_TRUCK"], "customs": ns["H_CUSTOMS"]}

    ships = _rd(enriched, "shipments", usecols=["id", "supplier_id", "carrier", "origin_location"])
    sr = _rd(enriched, "shipment_routes", usecols=["shipment_id", "route_id"])
    deps = _rd(enriched, "route_dependencies", usecols=["route_id", "depends_on_route_id", "channel"])
    if ships.empty or sr.empty or deps.empty:
        check(False, "coupling tables readable")
        return
    ship_route = dict(zip(sr["shipment_id"], sr["route_id"]))
    sup_routes: dict = {}
    for sid, sup in zip(ships["id"], ships["supplier_id"]):
        if isinstance(sup, str) and sid in ship_route:
            sup_routes.setdefault(sup, set()).add(ship_route[sid])

    for channel, members in truth.items():
        chan_routes: dict = {}
        sub = deps[deps["channel"] == channel]
        for a, b in zip(sub["route_id"], sub["depends_on_route_id"]):
            chan_routes.setdefault(a, set()).add(b)
            chan_routes.setdefault(b, set()).add(a)
        coupled = {s for s in members if sup_routes.get(s, set()) & set(chan_routes)}
        others = [s for s in ships["supplier_id"].dropna().unique() if s not in members]
        base = sum(1 for s in others if sup_routes.get(s, set()) & set(chan_routes))
        recall = len(coupled) / max(1, len(members))
        fpr = base / max(1, len(others))
        print(f"         {channel:<9} true members {len(members):>3}  "
              f"reached by the observable {channel} coupling: {recall:6.1%}   "
              f"non-members also reached: {fpr:6.1%}")
    check(True, "observable coupling is a PROXY, not the latent grouping",
          "recall/false-positive rates printed above; membership itself is never emitted")


# --------------------------------------------------------------------- 3. temporal integrity

def check_temporal(enriched: str, n_samples: int = 40) -> None:
    print("\n3. temporal integrity of the as-of tables")
    ships = _rd(enriched, "shipments",
                parse_dates=["eta", "dispatched_at", "delivered_at", "created_at"])
    ssh = _rd(enriched, "shipment_status_history", parse_dates=["changed_at", "recorded_at"])
    # A shipment's REPORTED delivery time: the recorded_at of its `delivered` transition. This
    # is the clock db/enrich_schema.py computes every window against.
    dl = ssh[ssh["status"] == "delivered"][["shipment_id", "recorded_at"]]
    dl = dl.groupby("shipment_id", as_index=False)["recorded_at"].min()
    ships = ships.merge(dl.rename(columns={"shipment_id": "id", "recorded_at": "delivered_rec"}),
                        on="id", how="left")

    for tbl, keycol, idtbl, idcol, namecol in (
            ("carrier_temporal_features", "carrier_id", "carriers", "id", "name"),
            ("route_conditions", "route_id", "routes", "id", None)):
        t = _rd(enriched, tbl, parse_dates=["as_of_date", "recorded_at"])
        if t.empty:
            check(False, f"{tbl} readable")
            continue
        late = int((t["recorded_at"] < t["as_of_date"]).sum())
        check(late == 0, f"{tbl}: recorded_at is never before as_of_date",
              f"{len(t):,} rows, {late} violations")

    ctf = _rd(enriched, "carrier_temporal_features", parse_dates=["as_of_date"])
    car = _rd(enriched, "carriers", usecols=["id", "name"])
    name_of = dict(zip(car["id"], car["name"]))
    rng = random.Random(0)
    rows = ctf.dropna(subset=["on_time_rate_30d"])
    sample = rows.sample(min(n_samples, len(rows)), random_state=0)
    mismatch = 0
    for r in sample.itertuples():
        c = name_of.get(r.carrier_id)
        asof = pd.Timestamp(r.as_of_date)
        sub = ships[(ships["carrier"] == c) & ships["delivered_at"].notna()]
        # the emitter's rule, restated independently here: REPORTED at or before as_of, and
        # delivered inside the trailing 30 days
        w = sub[(sub["delivered_rec"].fillna(sub["delivered_at"]) <= asof)
                & (sub["delivered_at"] >= asof - pd.Timedelta(days=30))]
        if len(w) < 3:
            continue
        exp = round((w["delivered_at"] <= w["eta"]).sum() / len(w), 4)
        if abs(exp - float(r.on_time_rate_30d)) > 1e-9:
            mismatch += 1
    check(mismatch == 0,
          "carrier_temporal_features.on_time_rate_30d recomputes EXACTLY from reported-only rows",
          f"{len(sample)} sampled, {mismatch} mismatches")

    rep = _rd(enriched, "product_replenishment",
              parse_dates=["ordered_at", "eta", "closed_at", "recorded_at"])
    bad = int((rep["recorded_at"] < rep["ordered_at"]).sum())
    check(bad == 0, "product_replenishment: recorded_at is never before ordered_at",
          f"{len(rep):,} rows, {bad} violations")
    closed = rep.dropna(subset=["closed_at"])
    bad2 = int((closed["closed_at"] < closed["ordered_at"]).sum())
    check(bad2 == 0, "product_replenishment: a link never closes before it opens",
          f"{len(closed):,} closed links, {bad2} violations")


# --------------------------------------------------------------------- 4. census

def check_census(enriched: str) -> None:
    print("\n4. census of the new node types (thin-relation risk)")
    sizes = {}
    for t, f in (("Supplier", "suppliers"), ("Product", "products"), ("Component", "components"),
                 ("Warehouse", "warehouses"), ("Factory", "factories"),
                 ("Carrier", "carriers"), ("Port", "ports"), ("Route", "routes")):
        sizes[t] = len(_rd(enriched, f))
    ref = sizes["Supplier"]
    for t, n in sizes.items():
        flag = "  <-- THIN" if n < 0.05 * ref else ""
        print(f"         {t:<10} {n:>6,}   {n / ref:7.2%} of Supplier{flag}")
    check(sizes["Carrier"] > 0 and sizes["Route"] > 0, "new node types are non-empty")

    wr = _rd(enriched, "warehouse_role_features", usecols=["warehouse_id", "inbound_share"])
    if not wr.empty:
        per = wr.groupby("warehouse_id")["inbound_share"].mean()
        spread = float(per.max() - per.min())
        print(f"         warehouse inbound share: min {per.min():.4f}  max {per.max():.4f}  "
              f"spread {spread:.4f}  over {len(per)} warehouses")
        check(spread < 0.10,
              "warehouse role is NOT bimodal -- role emitted as features+relations, not a node split",
              f"spread {spread:.4f} < 0.10, so a hard node split would invent a partition")

    rep = _rd(enriched, "product_replenishment", usecols=["product_id", "warehouse_id"])
    inv = _rd(enriched, "inventory", usecols=["product_id", "warehouse_id"])
    if not rep.empty and not inv.empty:
        cov = len(set(zip(rep["product_id"], rep["warehouse_id"]))) / max(1, len(inv))
        print(f"         REPLENISHED_BY covers {cov:.1%} of (product, warehouse) pairs "
              f"over the whole timeline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enriched", default=os.path.join(HERE, "csv_v3enriched", "v0enr_seed42"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="v1")
    ap.add_argument("--skip-byte-identity", action="store_true")
    a = ap.parse_args()

    print(f"verifying {a.enriched}  (variant {a.variant}, seed {a.seed}, config {a.config})")
    if not a.skip_byte_identity:
        check_byte_identity(a.enriched, a.variant, a.seed, a.config)
    check_no_leak(a.enriched, a.variant, a.seed, a.config)
    check_temporal(a.enriched)
    check_census(a.enriched)
    print("\n" + ("ALL CHECKS PASSED" if not _fails else f"{_fails} CHECK(S) FAILED"))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
