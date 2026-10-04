"""Phase 18 Stage 5 -- global pulse: network-wide as-of statistics, the same vector for every channel at a snapshot t0.

Why: `regime` enters every channel's lead time and every supplier's capacity, and the graph has no global node
(reports/phase18/stage1_generator.md (c)). A network-wide trailing reading is the cheapest way to show it to a model.

Window: the 4 weeks ending at t0, by RECORDED time -- (t0 - 28 d, t0]. Every source row is asserted recorded_ts <= t0
(`fwd_load.assert_asof`; a failing assertion is a STOP).
  pulse_lateness_days  mean (first receipt event - current promise date) over first receipts recorded in the window
  pulse_late_rate      share of those receipts later than the current promise date
  pulse_fill           mean received / ordered over lines whose FINAL receipt is recorded in the window
                       (receipts summed over everything recorded by t0)
  pulse_strain         sum over suppliers of quantity ordered on lines recorded in the window, divided by the sum of
                       their declared monthly capacity in force at t0 (supplier_capacity recorded by t0) x 28/30.44
inventory_position_weekly is never read.

  python ml/data/pulse.py build [--world v8]      # -> ml/artifacts/phase18/pulse_{world}.npz
  python ml/data/pulse.py falsify
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
from config import WORLDS, ARTIFACTS
from fwd_load import assert_asof, AsOfViolation, snapshot_dates, OUT_DIR

COLS = ["pulse_lateness_days", "pulse_late_rate", "pulse_fill", "pulse_strain"]
WIN = pd.Timedelta(days=28)


def load_sources(world):
    D = WORLDS[world]
    pol = pd.read_csv(D + "/po_lines.csv", usecols=["po_line_id", "channel_id", "qty_ordered", "current_promise_date",
                                                     "recorded_ts"])
    grn = pd.read_csv(D + "/grn_lines.csv", usecols=["po_line_id", "qty_received", "receipt_sequence", "is_final_receipt",
                                                      "event_ts", "recorded_ts"])
    cap = pd.read_csv(D + "/supplier_capacity.csv", usecols=["supplier_id", "part_id", "capacity_qty_per_month",
                                                             "effective_from", "effective_to", "recorded_ts"])
    ch = pd.read_csv(D + "/sourcing_channels.csv", usecols=["channel_id", "supplier_id"])
    for df, cols in ((pol, ("current_promise_date", "recorded_ts")), (grn, ("event_ts", "recorded_ts")),
                     (cap, ("effective_from", "effective_to", "recorded_ts"))):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    grn["is_final_receipt"] = grn.is_final_receipt.astype(str).str.lower().isin(["true", "1"])
    pol["supplier_id"] = pol.channel_id.map(ch.set_index("channel_id").supplier_id)
    grn = grn.merge(pol[["po_line_id", "qty_ordered", "current_promise_date"]], on="po_line_id", how="left")
    cap = cap[cap.part_id.isna()]                                  # supplier-level declarations only
    return dict(pol=pol, grn=grn, cap=cap)


def snapshot_pulse(S, t0):
    t0 = pd.Timestamp(t0); lo = t0 - WIN
    g, pol, cap = S["grn"], S["pol"], S["cap"]
    n = 0
    first = g[(g.receipt_sequence == 1) & (g.recorded_ts > lo) & (g.recorded_ts <= t0)]
    n += assert_asof(first.recorded_ts, t0, "grn_lines (first receipts)")
    late_d = (first.event_ts - first.current_promise_date).dt.days.to_numpy(float)
    fin = g[g.is_final_receipt & (g.recorded_ts > lo) & (g.recorded_ts <= t0)]
    n += assert_asof(fin.recorded_ts, t0, "grn_lines (final receipts)")
    upto = g[(g.recorded_ts <= t0) & g.po_line_id.isin(fin.po_line_id)]
    n += assert_asof(upto.recorded_ts, t0, "grn_lines (received to date)")
    rq = upto.groupby("po_line_id").qty_received.sum()
    fl = (fin.po_line_id.map(rq) / fin.qty_ordered.clip(lower=1)).clip(upper=1).to_numpy(float)
    po = pol[(pol.recorded_ts > lo) & (pol.recorded_ts <= t0)]
    n += assert_asof(po.recorded_ts, t0, "po_lines (window)")
    cu = cap[(cap.recorded_ts <= t0) & (cap.effective_from <= t0) & (cap.effective_to >= t0)]
    n += assert_asof(cu.recorded_ts, t0, "supplier_capacity")
    cu = cu.sort_values("recorded_ts").groupby("supplier_id").capacity_qty_per_month.last()
    ordq = po.groupby("supplier_id").qty_ordered.sum()
    sup = cu.index.intersection(ordq.index.union(cu.index))
    strain = float(ordq.reindex(sup).fillna(0).sum() / (cu.reindex(sup).sum() * 28 / 30.44)) if len(sup) else np.nan
    v = [float(np.mean(late_d)) if len(late_d) else np.nan, float(np.mean(late_d > 0)) if len(late_d) else np.nan,
         float(np.mean(fl)) if len(fl) else np.nan, strain]
    return np.array(v, np.float32), n, dict(first_receipts=len(first), final_receipts=len(fin), po_lines=len(po),
                                             suppliers_with_capacity=int(len(cu)))


def build(world="v8"):
    t = time.time()
    S = load_sources(world)
    snaps = snapshot_dates(world)
    X = np.zeros((len(snaps), len(COLS)), np.float32); asserted = 0; counts = []
    for i, t0 in enumerate(snaps):
        X[i], n, c = snapshot_pulse(S, t0); asserted += n; counts.append(c)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"pulse_{world}.npz")
    np.savez_compressed(path, X=X, snapshots=np.array([str(pd.Timestamp(s).date()) for s in snaps]), cols=np.array(COLS))
    meta = dict(world=world, n_snapshots=len(snaps), cols=COLS, source_rows_asserted_asof=int(asserted),
                min_counts={k: int(min(c[k] for c in counts)) for k in counts[0]}, seconds=round(time.time() - t, 1),
                per_snapshot={str(pd.Timestamp(s).date()): [round(float(x), 4) for x in X[i]] for i, s in enumerate(snaps)})
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_snapshot"}, indent=1))
    return path


def load(world="v8"):
    z = np.load(os.path.join(OUT_DIR, f"pulse_{world}.npz"), allow_pickle=False)
    return z["X"], list(z["snapshots"]), list(z["cols"])


def falsify(world="v8"):
    """A caller that drops the recorded_ts upper bound must be stopped: pass the window rows recorded up to t0 + 7 d."""
    S = load_sources(world)
    t0 = pd.Timestamp(snapshot_dates(world)[40])
    v, n, _ = snapshot_pulse(S, t0)
    g = S["grn"]
    leaky = g[(g.receipt_sequence == 1) & (g.recorded_ts > t0 - WIN) & (g.recorded_ts <= t0 + pd.Timedelta(days=7))]
    try:
        assert_asof(leaky.recorded_ts, t0, "grn_lines (first receipts, leaky window)")
        res = "DID NOT FIRE"
    except AsOfViolation as e:
        res = f"fired: {e}"
    out = dict(t0=str(t0.date()), clean_rows_asserted=n, clean_vector=[float(x) for x in v], leaky_window=res)
    print(json.dumps(out, indent=1))
    assert res.startswith("fired")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "falsify"])
    ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    {"build": build, "falsify": falsify}[a.mode](a.world)
