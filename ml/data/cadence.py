"""Phase 19 -- `cadence`: per-channel order cadence as of t0 (the legitimate stand-in for "when will the next line be raised").

Phase 18's oracle put the whole arrival ceiling on the creation week of the line. A model cannot know that week, but it
can see the channel's rhythm. Per channel c at snapshot t0, from rows asserted recorded_ts <= t0 (`fwd_load.assert_asof`;
a failing assertion is a STOP):

  cad_days_since_last   t0 - created_ts of the most recent line of c recorded by t0 (missing if none)
  cad_gap_median_d      median gap, in days, between consecutive created_ts of c's lines raised in the 52 weeks to t0
  cad_gap_iqr_d         IQR of those gaps (both missing with fewer than 2 gaps)
  cad_open_count        lines raised in the 26 weeks to t0 with no FINAL receipt recorded by t0 (Phase 18's open rule)
  cad_oldest_open_d     age in days of the oldest of those (missing if none)

inventory_position_weekly is never read.

  python ml/data/cadence.py build [--world v8]      # -> ml/artifacts/phase19/cadence_{world}.npz
  python ml/data/cadence.py falsify
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
from config import WORLDS
import fwd_load as FL

OUT_DIR = os.path.join(FL.ARTIFACTS, "phase19")
COLS = ["cad_days_since_last", "cad_gap_median_d", "cad_gap_iqr_d", "cad_open_count", "cad_oldest_open_d"]
GAP_WIN = pd.Timedelta(weeks=52)


def load_sources(world):
    S = FL.load_sources(world)            # po_lines (ci, created_ts, recorded_ts), grn_lines (is_final_receipt, recorded_ts)
    S["pol"] = S["pol"].sort_values(["ci", "created_ts"], kind="stable").reset_index(drop=True)
    return S


def snapshot_cadence(S, t0):
    t0 = pd.Timestamp(t0); NCH = len(S["ch"]); n = 0
    pol, grn = S["pol"], S["grn"]
    rec = pol[pol.recorded_ts.values <= t0.to_datetime64()]
    n += FL.assert_asof(rec.recorded_ts, t0, "po_lines (cadence)")
    F = np.full((NCH, len(COLS)), np.nan)
    last = rec.groupby("ci").created_ts.max()
    F[last.index.values, 0] = (t0 - last).dt.total_seconds().values / 86400.0
    win = rec[rec.created_ts.values > (t0 - GAP_WIN).to_datetime64()]
    ci = win.ci.values; ct = win.created_ts.values.astype("datetime64[s]").astype(np.int64) / 86400.0
    same = np.r_[False, ci[1:] == ci[:-1]]                                   # rows sorted by (ci, created_ts)
    gaps = pd.DataFrame({"ci": ci[same], "g": (ct[1:] - ct[:-1])[same[1:]]})
    if len(gaps):
        q = gaps.groupby("ci").g.quantile([0.25, 0.5, 0.75]).unstack()
        cnt = gaps.groupby("ci").size()
        ok = cnt.index[cnt >= 2]
        F[ok.values, 1] = q.loc[ok, 0.5].values
        F[ok.values, 2] = (q.loc[ok, 0.75] - q.loc[ok, 0.25]).values
    # open lines: raised in the 26 weeks to t0, recorded by t0, no final receipt recorded by t0
    po = rec[rec.created_ts.values > (t0 - FL.OPEN_LOOKBACK).to_datetime64()]
    g = grn[(grn.recorded_ts.values <= t0.to_datetime64()) & grn.is_final_receipt.values & grn.po_line_id.isin(po.po_line_id).values]
    n += FL.assert_asof(g.recorded_ts, t0, "grn_lines (cadence, final receipts)")
    op = po[~po.po_line_id.isin(g.po_line_id)]
    oc = np.bincount(op.ci.values, minlength=NCH).astype(float)
    F[:, 3] = oc
    old = op.groupby("ci").created_ts.min()
    F[old.index.values, 4] = (t0 - old).dt.total_seconds().values / 86400.0
    return F.astype(np.float32), n


def build(world="v8"):
    t = time.time()
    S = load_sources(world)
    snaps = FL.snapshot_dates(world)
    X = np.zeros((len(snaps), len(S["ch"]), len(COLS)), np.float32); asserted = 0
    for i, t0 in enumerate(snaps):
        X[i], k = snapshot_cadence(S, t0); asserted += k
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"cadence_{world}.npz")
    np.savez_compressed(path, X=X, snapshots=np.array([str(pd.Timestamp(s).date()) for s in snaps]),
                        channels=S["ch"].channel_id.to_numpy().astype(str), cols=np.array(COLS))
    meta = dict(world=world, n_snapshots=len(snaps), cols=COLS, source_rows_asserted_asof=int(asserted),
                seconds=round(time.time() - t, 1),
                missing_share={c: float(np.isnan(X[..., j]).mean()) for j, c in enumerate(COLS)},
                median={c: float(np.nanmedian(X[..., j])) for j, c in enumerate(COLS)})
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return path


def load(world="v8"):
    z = np.load(os.path.join(OUT_DIR, f"cadence_{world}.npz"), allow_pickle=False)
    return z["X"], list(z["snapshots"]), list(z["channels"]), list(z["cols"])


def falsify(world="v8"):
    """A caller that drops the recorded_ts bound must be stopped: hand the assertion the lines raised before t0
    regardless of when they were recorded."""
    S = load_sources(world)
    t0 = pd.Timestamp(FL.snapshot_dates(world)[40])
    _, n = snapshot_cadence(S, t0)
    p = S["pol"]
    leaky = p[p.created_ts <= t0 + pd.Timedelta(days=14)]
    try:
        FL.assert_asof(leaky.recorded_ts, t0, "po_lines (cadence, no recorded_ts bound)")
        res = "DID NOT FIRE"
    except FL.AsOfViolation as e:
        res = f"fired: {e}"
    out = dict(t0=str(t0.date()), clean_rows_asserted=n, leaky=res)
    print(json.dumps(out, indent=1))
    assert res.startswith("fired")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "falsify"])
    ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    {"build": build, "falsify": falsify}[a.mode](a.world)
