"""Phase 11B Stage C — a lateness metric whose reference is available at t0.

THE DEFECT. The shipped lateness metric is `arrival_week > promise_week`, scored on
`prediction - promise_week`. `promise_week` is derived from a po_line raised ~7 weeks AFTER the
forecast instant (B9 = 0.00% on every v8 dataset seed), so every arm -- the head included -- is
ranked against information no planner holds at t0. It cannot carry a deployment claim.

A SECOND DEFECT, found while building this and recorded because it changes an explanation.
`ml/eval/phase7_score.py`'s `arrival_promise` branch does not score the promise arm's own
prediction for lateness; it SUBSTITUTES a constant:

    late = M.arrival_scores(np.full(len(Y), float(spec["const"])), Y, EV, AUX)

That is why `promise_only` and `naive_global_median` both report 0.75181 -- it is the same
computation, not two arms coinciding. Scored honestly, `promise_only`'s lateness score is
identically zero (one distinct value) and its ROC-AUC is 0.5. reports/part2/phase-11a.md S3.1
attributed the identity to the metric's algebra; the cause is this substitution.
That report is protected, so the correction is recorded here (deviation 72).

THE REFERENCES BUILT HERE. Both are computable from data recorded on or before t0.

  (a) AS-OF CHANNEL LEAD -- the channel's median observed lead time (receipt - order, in weeks)
      over GRN lines RECORDED on or before t0, plus a global offset fitted on the TRAINING fold
      only. "Will this channel run slower than it has been running."

      The offset is needed because the label `arrival_week` is measured from t0 and therefore
      contains two terms: the wait until the order is raised, and the lead time once it is. Only
      the second varies by channel; the first is a property of the label's sampling window and is
      the same for every channel. So the channel-varying part of the reference is the historical
      lead, and the constant part is fitted once, on training rows, and never refitted.

  (b) CONTRACTED LEAD -- `sourcing_channels.contracted_lead_time_days` effective at t0, same
      construction. Static per channel, so a weaker test by design; carried as the conservative
      reference.

AS-OF IS ASSERTED, NOT ASSUMED: `assert_asof_receipts` fails if any receipt used for a snapshot
carries `recorded_ts > t0`, and the assertion is demonstrated firing.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "train")]
import numpy as np, pandas as pd
from config import WORLDS, FIT_WINDOW
from metrics import roc_auc


def assert_asof_receipts(used: pd.DataFrame, t0: pd.Timestamp, min_rows: int = 1):
    """Every receipt feeding a snapshot's reference must be RECORDED on or before t0.

    The emptiness guard is not decoration. Without it this check passes vacuously on an EMPTY
    set -- which is exactly what a naive falsification (shift every recorded_ts into the future)
    produces, and the reference would then silently collapse to the cold-start fallback for every
    channel. An assertion that passes hardest when it has nothing to look at is the defect this
    project has caught repeatedly.
    """
    assert len(used) >= min_rows, (
        f"as-of history for snapshot {t0.date()} is EMPTY ({len(used)} rows). The reference "
        f"would silently fall back to the global constant for every channel.")
    bad = int((pd.to_datetime(used["recorded_ts"]) > t0).sum())
    assert bad == 0, (f"as-of violation: {bad} receipts used for snapshot {t0.date()} carry "
                      f"recorded_ts > t0. The reference would read the future.")
    return True


def channel_lead_history(world: str):
    """One frame: per GRN line, its channel, its lead in weeks, and its RECORDED timestamp."""
    D = WORLDS[world]
    gl = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "event_ts", "recorded_ts"])
    pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "channel_id", "created_ts"])
    g = gl.merge(pl, on="po_line_id", how="inner")
    g["lead_weeks"] = ((pd.to_datetime(g.event_ts) - pd.to_datetime(g.created_ts)).dt.days / 7.0)
    g = g[np.isfinite(g.lead_weeks) & (g.lead_weeks >= 0)]
    g["recorded_ts"] = pd.to_datetime(g.recorded_ts)
    return g[["channel_id", "lead_weeks", "recorded_ts"]].sort_values("recorded_ts")


def asof_channel_median_lead(hist: pd.DataFrame, snapshots, check_asof=True):
    """-> {snapshot -> Series(channel_id -> median lead weeks)} using only receipts recorded <= t0."""
    out, rec = {}, hist["recorded_ts"].to_numpy()
    for s in snapshots:
        t0 = pd.Timestamp(s)
        used = hist.iloc[:int(np.searchsorted(rec, np.datetime64(t0), "right"))]
        if check_asof:
            assert_asof_receipts(used, t0)
        out[s] = used.groupby("channel_id")["lead_weeks"].median() if len(used) else pd.Series(dtype=float)
    return out


def contracted_lead_weeks(world: str):
    D = WORLDS[world]
    ch = pd.read_csv(f"{D}/sourcing_channels.csv",
                     usecols=["channel_id", "contracted_lead_time_days"])
    return ch.set_index("channel_id")["contracted_lead_time_days"].astype(float) / 7.0


def build_reference(kind, world, lb, tr_mask, check_asof=True):
    """-> reference R in WEEKS FROM t0, aligned to lb's rows. Offset fitted on TRAINING rows only."""
    snaps = sorted(pd.unique(lb.snapshot_date))
    if kind == "asof_channel_lead":
        hist = channel_lead_history(world)
        per = asof_channel_median_lead(hist, snaps, check_asof=check_asof)
        base = np.full(len(lb), np.nan)
        chan = lb.key.to_numpy()
        sd = lb.snapshot_date.to_numpy()
        for s in snaps:
            m = sd == s
            tbl = per[s]
            base[m] = pd.Series(chan[m]).map(tbl).to_numpy(float)
    elif kind == "contracted_lead":
        tbl = contracted_lead_weeks(world)
        base = lb.key.map(tbl).to_numpy(float)
    else:
        raise ValueError(kind)
    # global fallback for channels with no history yet (cold start), from TRAINING rows only
    gl_base = np.nanmedian(base[tr_mask]) if np.isfinite(base[tr_mask]).any() else 0.0
    base = np.where(np.isfinite(base), base, gl_base)
    # the constant term: fitted on TRAINING rows only, never refitted
    y = lb.label_value.to_numpy(float)
    ev = ~lb.label_censored.to_numpy(bool)
    fitm = tr_mask & ev
    c = float(np.median(y[fitm] - base[fitm])) if fitm.any() else 0.0
    return base + c, dict(kind=kind, offset_c=c, cold_start_fallback=float(gl_base),
                          n_cold_start=int((~np.isfinite(base)).sum()))


def lateness(ET, Y, EV, R):
    """yl = Y > R ; pl = ET - R. Both use only R, so a t0-available R makes the whole metric so."""
    m = EV & np.isfinite(R)
    yl = (Y[m] > R[m]).astype(int)
    pl = np.asarray(ET, float)[m] - R[m]
    if yl.min() == yl.max():
        return float("nan"), float(yl.mean()), int(m.sum())
    return float(roc_auc(yl, pl)), float(yl.mean()), int(m.sum())
