"""Phase 22 Stage 1a-b -- the systematic leak scan of every panel column the stored models read.

AUDIT INSTRUMENT, never a model input. To reproduce the generator's weekly store EXACTLY, the rebuild reads the
generator's own per-line arrays from the world's `_sim.npz` (pt, pch, pq, pd, pl, pa, contracted, DECL, MONTHKEY; asserted
aligned with po_lines / grn_lines row for row) plus the emitted timestamps (po_lines created / recorded, grn_lines event /
recorded, po_line_revisions event / recorded, supplier_capacity recorded). It follows generator_v8.py's "weekly stores"
block line for line (visible-week bucketing, forward fills, rolls).

Scan (pre-registration D1). For a sample of panel weeks t and two horizons:
  H_week  keep source rows VISIBLE in a week <= t (the panel row's own week); poison everything visible later
  H_date  keep rows visible in a week <  t (strict recorded_ts <= the snapshot date W[t]); poison the rest
"Poison" = replace the value-bearing fields of every later-visible row with noise:
  * a line whose ORDER is visible later: qty, reporting lag
  * a line whose OUTCOME (first receipt; for an unshipped line, never) is visible later: delivered, lead
    (a receipt already visible keeps its quantity even if its order was recorded late)
  * a revision visible later: dropped;  a declared capacity recorded later: noise
and the column's value at row t is recomputed. A column CHANGES if any channel's row-t value moves (beyond 1e-9).
Constructed failing case: lead_time_actual_days, lead_time_ratio, otd_rate_last13 MUST change under H_week, or the scan
is invalid (STOP). Constructed passing case: qty_ordered must NOT change under H_week.

  python ml/eval/phase22_leakscan.py [--world v8]     -> ml/artifacts/phase22/leakscan_{world}.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP

COLS = ["qty_ordered", "qty_received", "is_active_week", "fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52",
        "lead_time_actual_days", "lead_time_ratio", "otd_rate_last13", "ack_gap_ratio", "load_ratio", "reporting_lag_days",
        "active_weeks_in_52", "revision_count"]
KNOWN = ["lead_time_actual_days", "lead_time_ratio", "otd_rate_last13"]
DAY = np.timedelta64(1, "D")


def week_index(ts, W0):
    d = pd.to_datetime(pd.Series(np.asarray(ts))).dt.normalize()
    ws = d - pd.to_timedelta(d.dt.weekday, unit="D")
    return ((ws - W0).dt.days // 7).to_numpy()


class Sources:
    def __init__(self, world):
        D = config.WORLDS[world]
        z = np.load(os.path.join(D, "_sim.npz"))
        self.pch, self.pq, self.pd, self.pl, self.pa, self.pt = (z[k] for k in ("pch", "pq", "pd", "pl", "pa", "pt"))
        self.contracted, self.DECL, self.MONTHKEY = z["contracted"], z["DECL"], z["MONTHKEY"]
        self.util_obs = z["util_obs"]
        self.CS = z["CS"]
        pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "qty_ordered", "created_ts", "recorded_ts"])
        ids = pl.po_line_id.str[3:].astype(int).to_numpy()
        assert (ids == np.arange(len(ids))).all() and (self.pq == pl.qty_ordered.to_numpy()).all(), "_sim.npz not aligned with po_lines"
        self.W0 = pd.Timestamp("2016-01-04"); self.T = len(self.MONTHKEY); self.NCH = len(self.contracted)
        self.ev_po, self.rec_po = pd.to_datetime(pl.created_ts).to_numpy(), pd.to_datetime(pl.recorded_ts).to_numpy()
        self.vw_ord = np.maximum(week_index(self.ev_po, self.W0), week_index(self.rec_po, self.W0))
        self.lag = (self.rec_po - self.ev_po) / DAY
        g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "qty_received", "event_ts", "recorded_ts"])
        gi = g.po_line_id.str[3:].astype(int).to_numpy()
        assert (self.pd[gi] == g.qty_received.to_numpy()).all(), "_sim.npz delivered != grn_lines"
        self.gi = gi
        self.vw_rec = np.full(len(ids), 10 ** 6)
        self.vw_rec[gi] = np.maximum(week_index(g.event_ts, self.W0), week_index(g.recorded_ts, self.W0))
        rv = pd.read_csv(f"{D}/po_line_revisions.csv", usecols=["po_line_id", "event_ts", "recorded_ts"])
        self.rev_line = rv.po_line_id.str[3:].astype(int).to_numpy()
        self.rev_vw = np.maximum(week_index(rv.event_ts, self.W0), week_index(rv.recorded_ts, self.W0))
        sc = pd.read_csv(f"{D}/supplier_capacity.csv", usecols=["supplier_id", "effective_from", "recorded_ts"])
        sc["m"] = (pd.to_datetime(sc.effective_from).dt.year - 2016) * 12 + pd.to_datetime(sc.effective_from).dt.month - 1
        sups = sorted(sc.supplier_id.unique()); si = {s: i for i, s in enumerate(sups)}
        self.decl_rec = np.full(self.DECL.shape, np.datetime64("2100-01-01", "ns"))
        ok = (sc.m >= 0) & (sc.m < self.DECL.shape[0])
        self.decl_rec[sc.m[ok].to_numpy(), sc.supplier_id[ok].map(si).to_numpy()] = pd.to_datetime(sc.recorded_ts[ok]).to_numpy()


def ffill_rows(x):
    return pd.DataFrame(x).ffill(axis=1).to_numpy()


def roll(x, k):
    return pd.DataFrame(x).T.rolling(k, min_periods=1).mean().T.to_numpy()


def rebuild(S, pq=None, pdv=None, pl=None, lag=None, rev_keep=None, DECL=None):
    """generator_v8.py weekly-store derivation, from (possibly poisoned) per-line arrays. -> {col: [NCH, T]}"""
    pq = S.pq if pq is None else pq; pdv = S.pd if pdv is None else pdv; pl = S.pl if pl is None else pl
    lag = S.lag if lag is None else lag; DECL = S.DECL if DECL is None else DECL
    NCH, T = S.NCH, S.T
    IN = (S.vw_ord >= 0) & (S.vw_ord < T)
    cw = S.pch[IN] * T + S.vw_ord[IN]
    O = np.bincount(cw, weights=pq[IN], minlength=NCH * T).reshape(NCH, T)
    shipped = (pdv > 0) & (S.pa < T)
    INR = shipped & (S.vw_rec >= 0) & (S.vw_rec < T)
    Rv = np.bincount(S.pch[INR] * T + S.vw_rec[INR], weights=pdv[INR], minlength=NCH * T).reshape(NCH, T)
    line_fill = np.where(pq > 0, np.minimum(pdv / np.maximum(pq, 1), 1.0), 1.0)
    num = np.bincount(cw, weights=line_fill[IN] * pq[IN], minlength=NCH * T)
    den = np.bincount(cw, weights=pq[IN].astype(float), minlength=NCH * T)
    fillw = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).reshape(NCH, T)
    f_ff = ffill_rows(fillw)
    leadw = np.full((NCH, T), np.nan)
    leadw[S.pch[IN], S.vw_ord[IN]] = pl[IN]
    lead_ff = ffill_rows(leadw)
    otd = np.where(np.isnan(leadw), np.nan, (leadw <= S.contracted[:, None]).astype(float))
    lnum = np.bincount(cw, weights=lag[IN], minlength=NCH * T); lcnt = np.bincount(cw, minlength=NCH * T)
    lagw = np.where(lcnt > 0, lnum / np.maximum(lcnt, 1), np.nan).reshape(NCH, T)
    keep = np.ones(len(S.rev_line), bool) if rev_keep is None else rev_keep
    rin = keep & (S.rev_vw >= 0) & (S.rev_vw < T)
    REVW = np.bincount(S.pch[S.rev_line[rin]] * T + S.rev_vw[rin], minlength=NCH * T).reshape(NCH, T)
    month_ordered = np.zeros(DECL.shape)
    np.add.at(month_ordered, (S.MONTHKEY[S.pt], S.CS[S.pch]), pq)
    util_obs = month_ordered / DECL
    load = util_obs[S.MONTHKEY][:, S.CS].T
    return {"qty_ordered": O, "qty_received": Rv, "is_active_week": (O > 0).astype(float), "fill_rate": f_ff,
            "fill_rate_last4": roll(f_ff, 4), "fill_rate_last13": roll(f_ff, 13), "fill_rate_last52": roll(f_ff, 52),
            "lead_time_actual_days": lead_ff, "lead_time_ratio": lead_ff / S.contracted[:, None],
            "otd_rate_last13": roll(ffill_rows(otd), 13), "ack_gap_ratio": 1 - f_ff, "load_ratio": load,
            "reporting_lag_days": ffill_rows(lagw), "active_weeks_in_52": np.repeat(np.minimum(np.arange(T) + 1, 52)[None, :], NCH, 0).astype(float),
            "revision_count": REVW.astype(float)}


TOL = {"lead_time_actual_days": 0.006, "lead_time_ratio": 6e-5, "load_ratio": 6e-5, "reporting_lag_days": 0.006}


def reproduce(S, R, world):
    from cache import load_panel
    panel, miss, active, meta = load_panel(os.path.join(config.CACHE, world))
    out = {}
    for c in COLS:
        j = meta["cols"].index(c)
        P = np.asarray(panel[:, :, j], float)
        if c in meta["nullable"]:
            m = np.asarray(miss[:, :, meta["nullable"].index(c)]) > 0
        else:
            m = np.ones(P.shape, bool)
        r = R[c]
        ok_null = np.isfinite(r) == m
        tol = TOL.get(c, 6e-6)
        both = m & np.isfinite(r)
        close = np.abs(P[both] - r[both]) <= tol + 1e-9
        out[c] = dict(null_pattern_agrees=float(ok_null.mean()), value_agrees=float(close.mean()), rows=int(both.sum()))
    return out


def poison(S, t, strict, rng):
    """Arrays with every source row visible after the horizon replaced by noise. strict=False: H_week (keep vw <= t);
    strict=True: H_date (keep vw < t)."""
    later = (lambda vw: vw >= t) if strict else (lambda vw: vw > t)
    pq, pdv, pl, lag = S.pq.astype(float).copy(), S.pd.astype(float).copy(), S.pl.copy(), S.lag.copy()
    o = later(S.vw_ord)
    pq[o] = rng.integers(1, 5000, o.sum()); lag[o] = rng.uniform(0, 30, o.sum())
    out = later(S.vw_rec)                           # the outcome (receipt) is not visible by the horizon (never, if unshipped)
    pdv[out] = np.floor(rng.uniform(0, 1, out.sum()) * pq[out]); pl[out] = rng.uniform(3, 200, out.sum())
    rev_keep = ~later(S.rev_vw)
    hz = S.W0 + pd.Timedelta(days=7 * (int(t) if strict else int(t) + 1))
    DECL = S.DECL.copy()
    d = S.decl_rec > np.datetime64(hz)
    DECL[d] = rng.uniform(60, 5e4, d.sum())
    return dict(pq=pq, pdv=pdv, pl=pl, lag=lag, rev_keep=rev_keep, DECL=DECL)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="v8"); a = ap.parse_args()
    PP.register()
    st = C.require_clean()
    t0 = time.time()
    S = Sources(a.world)
    R = rebuild(S)
    rep = reproduce(S, R, a.world)
    print("reproduction:", json.dumps(rep), flush=True)
    snaps = pd.to_datetime(pd.read_csv(os.path.join(config.WORLDS[a.world], "snapshots.csv")).as_of_ts)
    snaps = snaps[(snaps >= "2019-01-01") & (snaps <= "2025-12-31")]
    weeks = ((snaps - S.W0).dt.days // 7).to_numpy()
    weeks = weeks[np.linspace(0, len(weeks) - 1, 12).astype(int)]
    rng = np.random.default_rng(22)
    res = {c: {"H_week": [], "H_date": []} for c in COLS}
    for t in weeks:
        for hz, strict in (("H_week", False), ("H_date", True)):
            Rp = rebuild(S, **poison(S, int(t), strict, rng))
            for c in COLS:
                a_, b_ = np.nan_to_num(R[c][:, t], nan=-999.0), np.nan_to_num(Rp[c][:, t], nan=-999.0)
                res[c][hz].append(float((np.abs(a_ - b_) > 1e-9).mean()))
        print(f"  week {t} ({(S.W0 + pd.Timedelta(days=7 * int(t))).date()}) done", flush=True)
    table = {c: dict(reproduced=rep[c], changes_H_week=bool(max(res[c]["H_week"]) > 0), changes_H_date=bool(max(res[c]["H_date"]) > 0),
                     share_channels_changed_H_week_mean=float(np.mean(res[c]["H_week"])),
                     share_channels_changed_H_date_mean=float(np.mean(res[c]["H_date"]))) for c in COLS}
    valid = all(table[c]["changes_H_week"] for c in KNOWN) and not table["qty_ordered"]["changes_H_week"]
    out = dict(stamp=st, world=a.world, weeks=[str((S.W0 + pd.Timedelta(days=7 * int(t))).date()) for t in weeks],
               table=table, scan_valid=valid, leaking_H_week=[c for c in COLS if table[c]["changes_H_week"]],
               seconds=time.time() - t0,
               note="AUDIT ONLY: reads the world's _sim.npz to reproduce the generator's weekly store exactly; never a model input")
    C.dump(out, f"phase22/leakscan_{a.world}.json")
    for c in COLS:
        print(f"  {c:24s} repro {rep[c]['value_agrees']:.4f}/{rep[c]['null_pattern_agrees']:.4f}  H_week {table[c]['changes_H_week']} "
              f"({table[c]['share_channels_changed_H_week_mean']:.4f})  H_date {table[c]['changes_H_date']} ({table[c]['share_channels_changed_H_date_mean']:.4f})")
    print("SCAN VALID" if valid else "SCAN INVALID -- STOP", flush=True)
    assert valid, "the constructed failing case did not fire (or qty_ordered changed): the scan is invalid"


if __name__ == "__main__":
    main()
