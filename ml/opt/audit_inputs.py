"""Phase 10 Stage 0 — input feasibility audit for guide 10.1 and 10.2.

This project has found 29 all-zero numeric columns and one entirely empty table. An optimiser written on top of a
column that is absent, empty or constant is a solver with nothing to solve, so every input is audited BEFORE any
model is built: does the column exist, how much of it is null, how much is zero, is it constant, what is its range.

`inventory_position_weekly` is never read. The guard below makes that a runtime error rather than a promise.

  python ml/opt/audit_inputs.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from config import WORLDS, ARTIFACTS

FORBIDDEN = "inventory_position_weekly"          # Phase 9 blocker; never an input to anything in this project

# (table, column, which step needs it, what it is for)
NEEDED = [
    # ---- 10.1 delivery schedule
    ("part_demand_weekly", "gross_requirement_p50", "10.1", "monthly/weekly requirement"),
    ("part_demand_weekly", "gross_requirement_p90", "10.1", "requirement, upper quantile"),
    ("part_plant", "safety_stock_qty", "10.1", "safety-stock floor"),
    ("part_plant", "reorder_point_qty", "10.1", "reorder point (balance proxy candidate)"),
    ("part_plant", "min_order_qty", "10.1", "MOQ"),
    ("part_plant", "lot_size", "10.1", "lot-size multiple"),
    ("part_plant", "planning_lead_time_days", "10.1", "offset from order to receipt"),
    ("supplier_contracts", "moq", "10.1", "MOQ, contract side"),
    ("supplier_contracts", "lot_size", "10.1", "lot size, contract side"),
    ("supplier_contracts", "max_volume_cap", "10.1", "capacity limit per period"),
    ("supplier_contracts", "min_volume_commitment", "10.1/10.2", "minimum-volume commitment"),
    ("supplier_contracts", "penalty_clause_inr", "10.2", "penalty for breaching the commitment"),
    ("part_costs", "unit_cost_inr", "10.1/10.2", "purchase cost"),
    ("part_costs", "freight_cost_inr", "10.1", "freight cost"),
    ("logistics_lanes", "standard_transit_days", "10.1", "transit time"),
    ("logistics_lanes", "distance_km", "10.1", "freight scaling"),
    ("calendar", "is_working_day", "10.1", "feasible receipt weeks"),
    ("calendar", "is_shutdown", "10.1", "infeasible weeks"),
    # ---- 10.2 allocation
    ("supplier_allocation", "allocation_pct", "10.2", "incumbent split"),
    ("alternate_sources", "qualification_status", "10.2", "is the alternate qualified"),
    ("alternate_sources", "qualification_lead_days", "10.2", "PPAP lead time"),
    ("alternate_sources", "ramp_rate_pct_per_month", "10.2", "ramp-rate limit"),
    ("alternate_sources", "cost_delta_pct", "10.2", "price delta of the alternate"),
    ("sourcing_channels", "is_approved", "10.2", "qualified supplier set"),
    ("sourcing_channels", "approval_status", "10.2", "qualified supplier set"),
    ("tooling", "is_transferable", "10.2", "tooling constraint"),
    ("tooling", "duplicate_exists", "10.2", "tooling constraint"),
    ("tooling", "transfer_lead_days", "10.2", "tooling transfer time"),
    ("revealed_capacity_monthly", "revealed_capacity_est", "10.2", "supplier capacity ceiling"),
    ("revealed_capacity_monthly", "evidence_strength", "10.2", "confidence in that ceiling"),
    # ---- the opening stock balance: every candidate, none of them inventory_position_weekly
    ("inventory_snapshots", "qty_on_hand", "10.1", "OPENING BALANCE candidate"),
    ("inventory_transactions", "qty", "10.1", "OPENING BALANCE candidate (cumulative)"),
]


def audit_column(csv_dir, table, col):
    path = os.path.join(csv_dir, table + ".csv")
    assert FORBIDDEN not in path, f"refusing to read {FORBIDDEN}"
    if not os.path.exists(path):
        return dict(exists=False)
    head = pd.read_csv(path, nrows=0).columns
    if col not in head:
        return dict(exists=False, table_exists=True)
    s = pd.read_csv(path, usecols=[col])[col]
    n = len(s)
    out = dict(exists=True, rows=int(n), null_frac=float(s.isna().mean()), n_distinct=int(s.nunique(dropna=True)))
    v = pd.to_numeric(s, errors="coerce")
    numeric = v.notna().sum() > 0 and not pd.api.types.is_object_dtype(s.dropna().infer_objects())
    if v.notna().any():
        vv = v.dropna()
        out.update(numeric=True, zero_frac=float((vv == 0).mean()), min=float(vv.min()), max=float(vv.max()),
                   mean=float(vv.mean()))
    else:
        out.update(numeric=False, values=sorted(map(str, s.dropna().unique()))[:6])
    out["constant"] = out["n_distinct"] <= 1
    return out


def main(out_path):
    R = {}
    for w, csv_dir in WORLDS.items():
        for table, col, step, purpose in NEEDED:
            R[f"{w}|{table}.{col}"] = dict(world=w, table=table, column=col, step=step, purpose=purpose,
                                           **audit_column(csv_dir, table, col))
    json.dump(R, open(out_path, "w"), indent=1, default=float)

    print(f"{'world':5s} {'step':8s} {'table.column':52s} {'rows':>9s} {'null%':>7s} {'zero%':>7s} {'distinct':>8s} {'range':>24s}")
    for k, r in R.items():
        if not r.get("exists"):
            print(f"{r['world']:5s} {r['step']:8s} {r['table'] + '.' + r['column']:52s} {'ABSENT':>9s}")
            continue
        rng = (f"[{r['min']:.4g}, {r['max']:.4g}]" if r.get("numeric") else ",".join(r.get("values", []))[:24])
        print(f"{r['world']:5s} {r['step']:8s} {r['table'] + '.' + r['column']:52s} {r['rows']:9,d} "
              f"{100 * r['null_frac']:7.2f} {100 * r.get('zero_frac', float('nan')):7.2f} {r['n_distinct']:8,d} {rng:>24s}"
              + ("  CONSTANT" if r["constant"] else ""))

    # the pre-registered prediction, checked in code rather than by eye
    cost_cols = [c for c in ("holding_cost", "carrying_cost", "holding_cost_inr", "order_cost", "ordering_cost")
                 if any(c in pd.read_csv(os.path.join(WORLDS["v6"], t + ".csv"), nrows=0).columns
                        for t in ("part_costs", "part_plant", "parts", "product_economics"))]
    print("\n0.3 PRE-REGISTERED PREDICTION")
    print(f"  holding-cost columns found anywhere in part_costs / part_plant / parts / product_economics: {cost_cols or 'NONE'}")
    print("  -> the guide's 10.1 gate ('no capacity constraint and ZERO holding cost -> order in the last feasible "
          "week') has a premise\n     EVERY part satisfies, and the quantity it was written to catch does not exist. "
          "The gate cannot fail: see the report.")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "phase10_input_audit.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    main(a.out)
