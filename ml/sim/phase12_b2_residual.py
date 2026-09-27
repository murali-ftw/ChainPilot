"""Phase 12 B2.7 -- what the residual gap implicates, measured in the held-out 2025 store. No simulation.

After the order policy the simulation still over-projects part-plant-weeks below safety stock. Every flow the
simulation models or omits is measured over each 2025 test snapshot's 13-week horizon, per part-plant, and set beside
the simulation's own flows (ml/artifacts/phase12_b2_validate.json, arm policy_rop):

  modelled   consumption (the plan x N(1, 0.1265)), supplier receipts (policy + pipeline)
  OMITTED    inter-plant transfers in/out (generator section 4: surplus plants cover short plants for the same
             part -- a replenishment path that targets exactly the part-plants that are short), adjustments,
             scrap, expedites (generator 1a: a physical acceleration of an open line)

Also the conditional question that matters most: of the store's part-plant-weeks that are NOT below safety stock,
how many were kept there by a transfer in that week? Transfers are read from inventory_transactions (event_ts in the
horizon) -- the realised ledger, used here as the evaluation reference, never as a simulation input.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import phase12_common as C
from config import WORLDS
import montecarlo as MC
from phase12_b2_validate import test_snapshots

W = MC.HORIZON_WEEKS


def main():
    st = C.require_clean()
    D = WORLDS["v8"]
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts"])
    tx["event_ts"] = pd.to_datetime(tx.event_ts)
    ex = pd.read_csv(f"{D}/expedite_events.csv", usecols=["part_id", "plant_id", "qty_expedited", "event_ts"])
    ex["event_ts"] = pd.to_datetime(ex.event_ts)
    pw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "safety_stock_qty"])
    pw["week_start"] = pd.to_datetime(pw.week_start)
    pp, _, _ = MC.part_plant_universe("v8")
    sub = pp.reset_index(drop=True)
    npp = len(sub)
    rows = []
    for s in test_snapshots("v8"):
        hi = s + pd.Timedelta(weeks=W)
        t = tx[(tx.event_ts > s) & (tx.event_ts <= hi)]
        f = t.groupby("txn_type").qty.sum() / npp
        plan = MC.forward_requirement("v8", s, sub, W).sum(1)
        e = ex[(ex.event_ts > s) & (ex.event_ts <= hi)]
        # which below-SS-threatened weeks were rescued by a transfer: weekly net transfer-in vs the store's position
        tw = t[t.txn_type == "transfer_in"].assign(week=lambda d: d.event_ts.dt.to_period("W").dt.start_time)
        tin = tw.groupby(["part_id", "plant_id", "week"]).qty.sum()
        h = pw[(pw.week_start > s) & (pw.week_start <= hi)].copy()
        h["tin"] = [tin.get((a, b, w), 0.0) for a, b, w in zip(h.part_id, h.plant_id, h.week_start)]
        below = h.qty_on_hand < h.safety_stock_qty
        rescued = (~below) & (h.qty_on_hand - h.tin < h.safety_stock_qty)
        rows.append(dict(snapshot=str(s.date()),
                         plan_per_pp=float(plan.mean()),
                         issue_per_pp=float(-f.get("issue_to_production", 0.0)),
                         receipt_per_pp=float(f.get("receipt", 0.0)),
                         transfer_in_per_pp=float(f.get("transfer_in", 0.0)),
                         transfer_out_per_pp=float(-f.get("transfer_out", 0.0)),
                         adjustment_per_pp=float(f.get("adjustment", 0.0)),
                         scrap_per_pp=float(-f.get("scrap", 0.0)),
                         expedites=int(len(e)), expedited_qty_per_pp=float(e.qty_expedited.sum() / npp),
                         observed_below_ss=float(below.mean()),
                         weeks_above_ss_only_because_of_transfer_in=float(rescued.mean()),
                         observed_below_ss_without_that_weeks_transfer_in=float((below | rescued).mean())))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    summ = {c: float(df[c].mean()) for c in df.columns if c != "snapshot"}
    summ["actual_issue_over_plan"] = summ["issue_per_pp"] / summ["plan_per_pp"]
    summ["transfer_in_over_issue"] = summ["transfer_in_per_pp"] / summ["issue_per_pp"]
    sim = json.load(open(os.path.join(C.ART, "phase12_b2_validate.json")))["policy_rop"]
    summ["sim_policy_rop"] = dict(below_ss=sim["below_ss"], arrivals_per_pp=sim["arrivals_per_pp"],
                                  consumption_per_pp=sim["consumption_per_pp"])
    summ["sim_arrivals_over_actual_receipts"] = sim["arrivals_per_pp"] / summ["receipt_per_pp"]
    summ["sim_consumption_over_actual_issue"] = sim["consumption_per_pp"] / summ["issue_per_pp"]
    print(json.dumps(summ, indent=1))
    print(C.dump(dict(stamp=st, per_snapshot=rows, summary=summ), "phase12_b2_residual.json"))


if __name__ == "__main__":
    main()
