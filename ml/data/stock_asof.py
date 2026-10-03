"""Phase 20 Stage 4c -- `stock_asof`: a LEGITIMATE as-of position-vs-reorder signal from OBSERVED tables only.

inventory_position_weekly is never read. Sources, every row asserted recorded_ts <= t0 (`fwd_load.assert_asof`):
  inventory_transactions  the part x plant ledger (opening `adjustment` posted 2015-12-28, then receipts, issues, scrap,
                          transfers). On hand at t0 = the sum of every ledger row RECORDED by t0 -- a book balance that
                          misses movements not yet keyed in, as a real ERP's would.
  po_lines / grn_lines    open PO quantity at t0 (Phase 18's open rule: raised in the 26 weeks to t0, no final receipt recorded)
  part_plant (master)     safety_stock_qty and planning_lead_time_days, both STATIC. reorder_point_qty is NOT used: the
                          generator writes it from the reorder point AFTER the simulation's last week (generator_v8.py
                          l. 1675), i.e. the 2026 value -- future information at any t0 (deviation 179).
Reorder point, estimated as of t0 = trailing-13-week ledger issue rate (units / week) x planning lead (weeks) + safety stock.

Columns (per part-plant, joined to each channel row through its part x plant):
  stk_onhand, stk_open_po, stk_issue_rate_13w, stk_rop_est,
  stk_ip_minus_rop_wk     (on hand + open PO - estimated reorder point) / issue rate   -- weeks above the trigger
  stk_onhand_minus_ss_wk  (on hand - safety stock) / issue rate

  python ml/data/stock_asof.py build [--world v8]     # -> ml/artifacts/phase20/stock_asof_{world}.npz
  python ml/data/stock_asof.py falsify
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
import fwd_load as FL
from config import WORLDS, ARTIFACTS

OUT_DIR = os.path.join(ARTIFACTS, "phase20")
COLS = ["stk_onhand", "stk_open_po", "stk_issue_rate_13w", "stk_rop_est", "stk_ip_minus_rop_wk", "stk_onhand_minus_ss_wk"]


def load_sources(world):
    S = FL.load_sources(world)
    D = WORLDS[world]
    ch = S["ch"]
    ppk = (ch.part_id + "|" + ch.plant_id).to_numpy()
    pp_u = sorted(set(ppk)); pidx = {k: i for i, k in enumerate(pp_u)}
    inv = pd.read_csv(D + "/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts", "recorded_ts"])
    inv["pp"] = (inv.part_id + "|" + inv.plant_id).map(pidx)
    inv = inv[inv.pp.notna()].copy(); inv["pp"] = inv.pp.astype(np.int64)
    inv["event_ts"] = pd.to_datetime(inv.event_ts); inv["recorded_ts"] = pd.to_datetime(inv.recorded_ts)
    pm = pd.read_csv(D + "/part_plant.csv", usecols=["part_id", "plant_id", "safety_stock_qty", "planning_lead_time_days"])
    pm["pp"] = (pm.part_id + "|" + pm.plant_id).map(pidx)
    pm = pm[pm.pp.notna()].set_index("pp").sort_index()
    ss = np.zeros(len(pp_u)); lead = np.full(len(pp_u), np.nan)
    ss[pm.index.astype(int)] = pm.safety_stock_qty.values; lead[pm.index.astype(int)] = pm.planning_lead_time_days.values / 7.0
    S.update(inv=inv, pp_of_ch=np.array([pidx[k] for k in ppk]), n_pp=len(pp_u), ss=ss, lead_wk=lead)
    return S


def snapshot_stock(S, t0):
    t0 = pd.Timestamp(t0); n = 0; NPP = S["n_pp"]; inv = S["inv"]
    rec = inv[inv.recorded_ts.values <= t0.to_datetime64()]
    n += FL.assert_asof(rec.recorded_ts, t0, "inventory_transactions (ledger)")
    onhand = np.bincount(rec.pp.values, weights=rec.qty.values.astype(float), minlength=NPP)
    iss = rec[(rec.txn_type == "issue_to_production").values & (rec.event_ts.values > (t0 - pd.Timedelta(weeks=13)).to_datetime64())]
    rate = -np.bincount(iss.pp.values, weights=iss.qty.values.astype(float), minlength=NPP) / 13.0
    # open PO per part-plant (Phase 18 open rule), from rows asserted recorded by t0
    pol, grn = S["pol"], S["grn"]
    po = pol[(pol.recorded_ts.values <= t0.to_datetime64()) & (pol.created_ts.values > (t0 - FL.OPEN_LOOKBACK).to_datetime64())]
    n += FL.assert_asof(po.recorded_ts, t0, "po_lines (open)")
    g = grn[(grn.recorded_ts.values <= t0.to_datetime64()) & grn.po_line_id.isin(po.po_line_id).values]
    n += FL.assert_asof(g.recorded_ts, t0, "grn_lines (open)")
    rq = g.groupby("po_line_id").agg(q=("qty_received", "sum"), fin=("is_final_receipt", "max"))
    r_q = po.po_line_id.map(rq.q).fillna(0).to_numpy(float); r_f = po.po_line_id.map(rq.fin).fillna(False).to_numpy(bool)
    open_q = np.where(r_f, 0.0, np.maximum(po.qty_ordered.to_numpy(float) - r_q, 0.0))
    open_pp = np.bincount(S["pp_of_ch"][po.ci.values], weights=open_q, minlength=NPP)
    rop = rate * S["lead_wk"] + S["ss"]
    den = np.maximum(rate, 1.0)
    F = np.column_stack([onhand, open_pp, rate, rop, (onhand + open_pp - rop) / den, (onhand - S["ss"]) / den])
    return F[S["pp_of_ch"]].astype(np.float32), n


def build(world="v8"):
    t = time.time()
    S = load_sources(world); snaps = FL.snapshot_dates(world)
    X = np.zeros((len(snaps), len(S["ch"]), len(COLS)), np.float32); asserted = 0
    for i, t0 in enumerate(snaps):
        X[i], k = snapshot_stock(S, t0); asserted += k
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"stock_asof_{world}.npz")
    np.savez_compressed(path, X=X, snapshots=np.array([str(pd.Timestamp(s).date()) for s in snaps]),
                        channels=S["ch"].channel_id.to_numpy().astype(str), cols=np.array(COLS))
    meta = dict(world=world, n_snapshots=len(snaps), cols=COLS, source_rows_asserted_asof=int(asserted), seconds=round(time.time() - t, 1),
                median={c: float(np.nanmedian(X[..., j])) for j, c in enumerate(COLS)}, negative_onhand_share=float((X[..., 0] < 0).mean()))
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def load(world="v8"):
    z = np.load(os.path.join(OUT_DIR, f"stock_asof_{world}.npz"), allow_pickle=False)
    return z["X"], list(z["snapshots"]), list(z["channels"]), list(z["cols"])


def falsify(world="v8"):
    S = load_sources(world); t0 = pd.Timestamp(FL.snapshot_dates(world)[40])
    _, n = snapshot_stock(S, t0)
    leaky = S["inv"][S["inv"].event_ts <= t0]              # bucketing on event_ts, not recorded_ts -- the classic mistake
    try:
        FL.assert_asof(leaky.recorded_ts, t0, "inventory_transactions bucketed on event_ts")
        res = "DID NOT FIRE"
    except FL.AsOfViolation as e:
        res = f"fired: {e}"
    print(json.dumps(dict(t0=str(t0.date()), clean_rows_asserted=n, leaky=res), indent=1))
    assert res.startswith("fired")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "falsify"])
    ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    {"build": build, "falsify": falsify}[a.mode](a.world)
