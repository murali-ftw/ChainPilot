"""Guide 10.1 — delivery-schedule MILP.

WHAT THIS SOLVES, AND WHAT IT DOES NOT.

The guide's formulation is

    min  sum_w ( c_hold * I_w + c_order * y_w + c_freight * ceil(x_w / truck_cap) )
    s.t. stock balance, I_w >= safety_stock, x_w >= MOQ * y_w, x_w = 0 (mod lot_size),
         sum_w x_w = requirement, y_w in {0,1}

Stage 0's audit (ml/opt/audit_inputs.py) found that three of those terms have no data behind them:

  * c_hold      -- NO holding-cost column exists anywhere in the 49-table schema.
  * c_order     -- NO ordering/setup-cost column exists either.
  * truck_cap   -- no truck capacity, so the freight term cannot take the guide's ceil() form.

and that the stock balance has no opening level I_0 in any usable form:

  * inventory_position_weekly -- every numeric column zero, and never read by this project;
  * inventory_snapshots.qty_on_hand -- 97.5% negative in v6, 54.9% in v7;
  * inventory_transactions cumsum -- agrees with the stated balance on 0.33% of rows.

So the DELIVERABLE (`solve`, mode="coverage") drops the stock balance and the safety-stock floor and records them
as dropped. It schedules receipts to cover the horizon's requirement under the constraints that DO have data --
lot-size multiples, MOQ, weekly capacity and feasible weeks -- and its objective carries only the freight term the
data supports, priced per unit from part_costs.freight_cost_inr.

`mode="balance"` adds the stock balance, the safety-stock floor and a holding term, every one of them driven by
INJECTED parameters (i0, c_hold, c_order). It is a labelled SENSITIVITY, never the deliverable: with a fabricated
opening balance its schedule is a statement about the assumption, not about the data.

CONSEQUENCE, stated once here and again in the report: with c_hold = 0 -- which is what the data gives -- the
timing of a coverage schedule is DEGENERATE. Every schedule with the same number of order weeks costs the same, so
which week the solver picks is tie-breaking, not economics. That is why the guide's own verify gate cannot fail
(see `gates.py`), and why nothing in this module should be read as a claim about WHEN to order.

As-of discipline: every table is filtered on its recorded/as-of column (`as_of_date`, `effective_from`,
`valid_from`) at t0, never on an event date. `inventory_position_weekly` is never read.

  python ml/opt/schedule_lp.py --world v6 --parts 3
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from config import WORLDS, ARTIFACTS

FORBIDDEN = "inventory_position_weekly"

# Injected assumptions: no column in the dataset carries any of these. Every number here is a STATED ASSUMPTION,
# not a measurement, and the report names it as such.
ASSUMPTIONS = dict(
    c_hold_per_unit_week=0.0,      # Stage 0.3: absent from the schema. 0.0 is the data-faithful value and makes
                                   # timing degenerate; the sensitivity sweeps it.
    c_order_per_week=0.0,          # absent from the schema
    truck_cap=None,                # absent: the guide's ceil(x/truck_cap) freight term cannot be formed
    i0=0.0,                        # BALANCE MODE ONLY: no opening balance exists. Fabricated; never a deliverable.
)


def _asof(df, col, t0):
    """Rows visible at t0 by their recorded/effective date. Never an event date (standing rule)."""
    if col not in df.columns:
        return df
    d = pd.to_datetime(df[col], errors="coerce")
    return df[d.isna() | (d <= pd.Timestamp(t0))]


def load_inputs(world, t0, horizon_weeks=13, n_parts=None, seed=0, series="forward"):
    """series="forward"  the TRUE planning horizon: rows visible at t0 (as_of <= t0) whose week lies ahead of t0.
                         Stage 0 found this is ONE week per part-plant: part_demand_weekly carries 12 as-of dates
                         per part-plant, each with a single week 30 days out, and production_plan is entirely
                         retrospective (target_period is 1-8 days BEFORE recorded_ts in both worlds). There is no
                         multi-week forward requirement anywhere in this dataset.
       series="observed" the 12 requirement observations for that part-plant that are already in the past at t0.
                         A MECHANISM DEMONSTRATION for the multi-week solver -- it is NOT a plan and no schedule
                         built on it may be presented as one.
    """
    csv = WORLDS[world]
    assert FORBIDDEN not in csv, "guard"
    t0 = pd.Timestamp(t0)
    dem = pd.read_csv(os.path.join(csv, "part_demand_weekly.csv"),
                      usecols=["part_id", "plant_id", "week_start", "as_of_date", "gross_requirement_p50"])
    dem = _asof(dem, "as_of_date", t0)                                   # as-of: the planner's visible forecast
    dem["week_start"] = pd.to_datetime(dem.week_start)
    if series == "forward":
        dem = dem[(dem.week_start > t0) & (dem.week_start <= t0 + pd.Timedelta(weeks=int(horizon_weeks)))]
    else:
        dem = dem[dem.week_start <= t0].groupby(["part_id", "plant_id"]).tail(int(horizon_weeks))
    pp = _asof(pd.read_csv(os.path.join(csv, "part_plant.csv")), "effective_from", t0)
    pp = pp.set_index(["part_id", "plant_id"])
    costs = _asof(pd.read_csv(os.path.join(csv, "part_costs.csv")), "effective_from", t0)
    costs = costs.groupby("part_id")[["unit_cost_inr", "freight_cost_inr"]].mean()
    con = _asof(pd.read_csv(os.path.join(csv, "supplier_contracts.csv")), "valid_from", t0)
    cap = float(con.max_volume_cap.median()) if len(con) else np.inf
    cal = pd.read_csv(os.path.join(csv, "calendar.csv"), usecols=["date", "plant_id", "is_working_day", "is_shutdown"])
    cal["date"] = pd.to_datetime(cal.date)
    cal["week"] = cal.date.dt.to_period("W").dt.start_time
    wk = cal.groupby(["plant_id", "week"]).agg(working=("is_working_day", "max"), shut=("is_shutdown", "max"))

    keys = list(dem.groupby(["part_id", "plant_id"]).groups)
    if n_parts:
        rng = np.random.default_rng(seed)
        tot = dem.groupby(["part_id", "plant_id"]).gross_requirement_p50.sum()
        keys = [k for k in keys if tot.get(k, 0) > 0]
        keys = [keys[i] for i in rng.choice(len(keys), size=min(n_parts, len(keys)), replace=False)]
    out = []
    for part, plant in keys:
        g = dem[(dem.part_id == part) & (dem.plant_id == plant)].sort_values("week_start")
        if g.empty:
            continue
        weeks = g.week_start.tolist()
        feasible = [bool(wk.loc[(plant, w), "working"]) and not bool(wk.loc[(plant, w), "shut"])
                    if (plant, w) in wk.index else True for w in weeks]
        row = pp.loc[(part, plant)] if (part, plant) in pp.index else None
        c = costs.loc[part] if part in costs.index else None
        out.append(dict(part_id=part, plant_id=plant, weeks=[str(w.date()) for w in weeks],
                        demand=g.gross_requirement_p50.to_numpy(float), feasible=feasible,
                        safety_stock=float(row.safety_stock_qty) if row is not None else 0.0,
                        moq=float(row.min_order_qty) if row is not None else 1.0,
                        lot_size=float(row.lot_size) if row is not None else 1.0,
                        weekly_cap=cap,
                        freight_per_unit=float(c.freight_cost_inr) if c is not None else 0.0,
                        unit_cost=float(c.unit_cost_inr) if c is not None else 0.0))
    return out


def solve(part, mode="coverage", c_hold=None, c_order=None, i0=None, time_limit=10.0, disable=()):
    """Receipt quantities per week.

    mode="coverage" (DELIVERABLE): cover the horizon requirement; no stock balance, no safety-stock floor.
    mode="balance"  (SENSITIVITY): adds the balance and the floor on an INJECTED opening level i0.

    Variables per week w: q_w integer (in lots, x_w = lot_size * q_w) and y_w binary (an order is placed).

    `disable` is for gates.py ONLY: it removes a constraint or cost term so a verify gate can be shown FAILING on a
    deliberately broken solver. Standing rule 1 -- a gate nobody has seen fail is not a gate. Never set it in
    production use.
    """
    disable = set(disable)
    c_hold = ASSUMPTIONS["c_hold_per_unit_week"] if c_hold is None else c_hold
    c_order = ASSUMPTIONS["c_order_per_week"] if c_order is None else c_order
    i0 = ASSUMPTIONS["i0"] if i0 is None else i0
    d = np.asarray(part["demand"], float)
    T = len(d)
    lot, moq, cap = part["lot_size"], part["moq"], part["weekly_cap"]
    if "lot" in disable:
        lot = 1.0                                     # broken: lot-size multiples not enforced
    if "moq" in disable:
        moq = 0.0                                     # broken: MOQ floor not enforced
    R = float(d.sum())
    qmax = int(np.ceil(cap / lot)) if np.isfinite(cap) else int(np.ceil(R / lot)) + 1
    nq, ny = T, T
    n = nq + ny

    # objective: freight is per unit (the guide's ceil(x/truck_cap) needs a truck capacity the data does not have);
    # holding is charged on the end-of-week position, which only exists in balance mode.
    c = np.zeros(n)
    c[:nq] = part["freight_per_unit"] * lot
    c[nq:] = c_order
    if mode == "balance" and c_hold and "holding" not in disable:
        # I_w = i0 + sum_{u<=w} (lot*q_u - d_u): holding on q_u is c_hold * lot * (T - u) summed over the horizon
        for u in range(T):
            c[u] += c_hold * lot * (T - u)

    A, lo, hi = [], [], []
    # coverage: coverage of the horizon requirement, to within one lot (exact equality is infeasible when R is not
    # a lot multiple -- the guide's "sum x_w = requirement" assumes it is)
    r = np.zeros(n); r[:nq] = lot
    if "coverage" in disable:
        A.append(r); lo.append(0.0); hi.append(np.inf)               # broken: nothing forces the requirement to be met
    else:
        A.append(r); lo.append(R); hi.append(R + lot - 1e-9)
    # MOQ linking, both directions: an order week carries at least the MOQ, a non-order week carries nothing
    for w in range(T):
        r = np.zeros(n); r[w] = lot; r[nq + w] = -moq
        A.append(r); lo.append(0.0); hi.append(np.inf)                    # lot*q_w >= moq*y_w
        r = np.zeros(n); r[w] = lot; r[nq + w] = -(cap if np.isfinite(cap) else R + lot)
        A.append(r); lo.append(-np.inf); hi.append(0.0)                   # lot*q_w <= cap*y_w
    if mode == "balance":
        for w in range(T):                                                # I_w >= safety_stock
            r = np.zeros(n); r[:w + 1] = lot
            A.append(r); lo.append(part["safety_stock"] + d[:w + 1].sum() - i0); hi.append(np.inf)

    ub_q = np.full(nq, float(qmax)); ub_y = np.ones(ny)
    if "feasible" not in disable:
        for w, ok in enumerate(part["feasible"]):
            if not ok:                                                    # non-working or shutdown week: no receipt
                ub_q[w] = 0.0; ub_y[w] = 0.0
    res = milp(c=c, constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.concatenate([ub_q, ub_y])),
               options=dict(time_limit=time_limit))
    out = dict(part_id=part["part_id"], plant_id=part["plant_id"], mode=mode, status=res.status,
               message=res.message, success=bool(res.success), requirement=R, lot_size=lot, moq=moq,
               weeks=part["weeks"], demand=d.tolist())
    if res.success:
        q = np.round(res.x[:nq]).astype(int); y = np.round(res.x[nq:]).astype(int)
        x = q * lot
        out.update(receipts=x.tolist(), order_weeks=[part["weeks"][w] for w in range(T) if y[w]],
                   n_order_weeks=int(y.sum()), total_received=float(x.sum()), cost=float(res.fun),
                   binding=binding_constraints(part, x, y, R, mode, i0))
    return out


def binding_constraints(part, x, y, R, mode, i0):
    """Which constraints actually bind for this part -- named, not inferred from the objective value."""
    d = np.asarray(part["demand"], float)
    lot, moq, cap = part["lot_size"], part["moq"], part["weekly_cap"]
    b = []
    if abs(x.sum() - R) <= lot:
        b.append(f"coverage (sum x = {x.sum():.0f} vs requirement {R:.0f}, within one lot of {lot:.0f})")
    if any(v > 0 and abs(v % lot) < 1e-6 for v in x):
        b.append(f"lot-size multiple ({lot:.0f})")
    if any(0 < v <= moq + 1e-9 for v in x):
        b.append(f"MOQ ({moq:.0f}) -- an order week sits at the floor")
    if any(v >= cap - 1e-6 for v in x) and np.isfinite(cap):
        b.append(f"weekly capacity ({cap:.0f})")
    if not all(part["feasible"]):
        b.append(f"infeasible weeks ({sum(not f for f in part['feasible'])} of {len(x)})")
    if mode == "balance":
        I = i0 + np.cumsum(x - d)
        if np.any(np.abs(I - part["safety_stock"]) < 1e-6):
            b.append(f"safety-stock floor ({part['safety_stock']:.0f}) [SENSITIVITY: injected i0={i0:.0f}]")
    return b or ["none binding: the solution is interior"]


def main(a):
    parts = load_inputs(a.world, a.t0, a.horizon, a.parts, series=a.series)
    rows = [solve(p, mode=a.mode) for p in parts]
    ok = [r for r in rows if r["success"]]
    status = {}
    for r in rows:
        status[r["message"][:40]] = status.get(r["message"][:40], 0) + 1
    print(f"world {a.world}  t0 {a.t0}  horizon {a.horizon}w  mode {a.mode}  series {a.series}  parts {len(rows)}")
    if a.series == "observed":
        print("  NOTE: series=observed uses PAST requirement weeks. It demonstrates the solver; it is not a plan.")
    else:
        w = sorted({len(r["weeks"]) for r in rows})
        print(f"  forward horizon actually available: {w} week(s) per part-plant (Stage 0: the data carries one)")
    print(f"solver status: {json.dumps(status)}")
    for r in ok[:a.show]:
        print(f"\n  {r['part_id']} @ {r['plant_id']}: requirement {r['requirement']:.0f}, lot {r['lot_size']:.0f}, "
              f"MOQ {r['moq']:.0f} -> received {r['total_received']:.0f} in {r['n_order_weeks']} order week(s) "
              f"{r['order_weeks']}, cost {r['cost']:.1f}")
        print(f"    binding: {'; '.join(r['binding'])}")
    out = os.path.join(ARTIFACTS, f"phase10_schedule_{a.world}_{a.mode}.json")
    json.dump(dict(world=a.world, t0=a.t0, horizon_weeks=a.horizon, mode=a.mode, series=a.series, assumptions=ASSUMPTIONS,
                   dropped_constraints=(["stock balance (no opening level I0 exists)",
                                         "safety-stock floor (needs the balance)"] if a.mode == "coverage" else []),
                   rows=rows), open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v6")
    ap.add_argument("--t0", default="2025-01-06")
    ap.add_argument("--horizon", type=int, default=13)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--mode", default="coverage", choices=["coverage", "balance"])
    ap.add_argument("--series", default="forward", choices=["forward", "observed"],
                    help="forward = the true (one-week) planning horizon; observed = past weeks, a mechanism demo only")
    main(ap.parse_args())
