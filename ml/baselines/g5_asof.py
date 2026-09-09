"""G5 -- is the as-of gate binding on what the model actually reads?

Shuffle `recorded_ts` WITHIN each source table, rebuild the visible-week bucketing from the
shuffled column, re-derive the channel weekly fill features, and retrain the fill head.

If performance does not drop, the features do not depend on when a fact became knowable, and
every downstream number is suspect.

Only the FEATURES are rebuilt. Labels are forward outcomes and are untouched, so the only
thing that changes between the two arms is when information becomes visible.
"""
from __future__ import annotations
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from loader import read_df                                     # noqa: E402


def week_idx(x, w0):
    d = pd.to_datetime(pd.Series(np.asarray(x)).values).normalize()
    ws = d - pd.to_timedelta(d.weekday, unit="D")
    return ((ws - w0).days // 7).to_numpy()


def build_fill_panel(csv_dir, shuffle_recorded=False, seed=7):
    """-> fill_last13 [NCH, T], plus the ordered/received panels it came from."""
    rng = np.random.default_rng(seed)
    ch = read_df(csv_dir, "sourcing_channels", usecols=["channel_id"])
    cidx = {v: i for i, v in enumerate(ch.channel_id)}
    NCH = len(cidx)
    cw = read_df(csv_dir, "channel_performance_weekly", usecols=["week_start"])
    w0 = pd.to_datetime(cw.week_start).min()
    T = int((pd.to_datetime(cw.week_start).max() - w0).days // 7) + 1

    pol = read_df(csv_dir, "po_lines",
                  usecols=["po_line_id", "channel_id", "qty_ordered", "created_ts", "recorded_ts"])
    grn = read_df(csv_dir, "grn_lines", usecols=["po_line_id", "qty_received", "event_ts", "recorded_ts"])

    rec_o = pol.recorded_ts.fillna(pol.created_ts)
    rec_r = grn.recorded_ts.fillna(grn.event_ts)
    if shuffle_recorded:                      # the treatment: destroy WHEN a fact became knowable
        rec_o = pd.Series(rng.permutation(rec_o.to_numpy()), index=rec_o.index)
        rec_r = pd.Series(rng.permutation(rec_r.to_numpy()), index=rec_r.index)

    vo = np.maximum(week_idx(pol.created_ts, w0), week_idx(rec_o, w0))
    ci = pol.channel_id.map(cidx).to_numpy()
    ok = (vo >= 0) & (vo < T) & ~pd.isna(ci)
    O = np.zeros((NCH, T), np.float64)
    np.add.at(O, (ci[ok].astype(int), vo[ok]), pol.qty_ordered.to_numpy(float)[ok])

    g = grn.merge(pol[["po_line_id", "channel_id"]], on="po_line_id", how="left")
    vr = np.maximum(week_idx(g.event_ts, w0), week_idx(rec_r, w0))
    ci2 = g.channel_id.map(cidx).to_numpy()
    ok2 = (vr >= 0) & (vr < T) & ~pd.isna(ci2)
    R = np.zeros((NCH, T), np.float64)
    np.add.at(R, (ci2[ok2].astype(int), vr[ok2]), g.qty_received.to_numpy(float)[ok2])

    # per-week fill where the channel ordered, forward-filled, then a 13-week rolling mean
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(O > 0, np.minimum(R / np.maximum(O, 1e-9), 1.0), np.nan)
    fdf = pd.DataFrame(f).ffill(axis=1)
    f13 = fdf.T.rolling(13, min_periods=1).mean().T.to_numpy(np.float32)
    return dict(O=O.astype(np.float32), R=R.astype(np.float32),
                f13=np.nan_to_num(f13), w0=w0, T=T, cidx=cidx)
