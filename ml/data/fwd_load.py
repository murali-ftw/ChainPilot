"""Phase 18 Stage 4 -- forward committed load (`fwd_load`), per channel and per supplier, as of each snapshot t0.

Why: the arrival / fill / capacity labels are about PO lines and months that START after t0 (reports/phase18/
stage1_generator.md (a)). The model sees supplier load as of t0; the outcome depends on load 1-12 weeks later. The plan
of record at t0 says how much requirement is coming. This module reads it, and nothing that is not recorded by t0.

Sources, every row filtered on recorded_ts <= t0 and then ASSERTED (`assert_asof`; a failing assertion is a STOP):
  part_demand_weekly  part x plant x week gross requirement P50, a version every 8 weeks (as_of_date).
                      Per (part, plant, target week) the latest version recorded by t0 is taken. The published table
                      carries NO recorded_ts (the schema drops the generator's column). The generator draws it as
                      as_of_date + 0..2 days (generator_v8.py l. 1418), so the recorded time used and asserted here is
                      the conservative upper bound as_of_date + 2 days (deviation 162).
  po_lines            the channel's share of its part-plant's ordered quantity over the trailing 52 weeks; open quantity
  grn_lines           trailing-52-week delivered throughput; receipts against open lines

Features (`COLS`), for channel c of supplier s at t0 -- weeks are t0 + 1 .. t0 + 13:
  fwd_ch_req_w{1_4,5_8,9_13}     channel requirement = part-plant P50 summed over the window x the channel's share
  fwd_ch_ratio_w*                the same / (channel trailing-52-week delivered per week x weeks in the window)
  fwd_ch_open_ratio              open PO quantity / channel trailing delivered per week (weeks of throughput)
  fwd_sup_req_w*, fwd_sup_ratio_w*, fwd_sup_open_ratio    the same summed over the supplier's channels
  fwd_pp_cov_w9_13               share of the far window's part-plant-weeks a version recorded by t0 covers
A window with no covered week is missing (NaN), not zero. A zero throughput gives a missing ratio.

Never used: realised future ordered volume. That is the PRIVILEGED hindsight control (reports/phase18/oracle/), not a
feature. inventory_position_weekly is never read.

  python ml/data/fwd_load.py build [--world v8]      # -> ml/artifacts/phase18/fwd_load_{world}.npz
  python ml/data/fwd_load.py falsify                 # the as-of assertion fires on a future-recorded row
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..")]
import numpy as np, pandas as pd
from config import WORLDS, ARTIFACTS, FIT_WINDOW

OUT_DIR = os.path.join(ARTIFACTS, "phase18")
WINDOWS = ((1, 4), (5, 8), (9, 13))
WTAG = ["w1_4", "w5_8", "w9_13"]
COLS = ([f"fwd_ch_req_{w}" for w in WTAG] + [f"fwd_ch_ratio_{w}" for w in WTAG] + ["fwd_ch_open_ratio"]
        + [f"fwd_sup_req_{w}" for w in WTAG] + [f"fwd_sup_ratio_{w}" for w in WTAG] + ["fwd_sup_open_ratio"]
        + ["fwd_pp_cov_w9_13"])
TRAIL = pd.Timedelta(weeks=52)
OPEN_LOOKBACK = pd.Timedelta(weeks=26)
PDW_MAX_LAG = pd.Timedelta(days=2)          # generator_v8.py l. 1418: recorded_ts = as_of_date + integers(0, 3) days


class AsOfViolation(AssertionError):
    """A source row recorded after t0 reached a feature. This is a STOP, not a warning."""


def assert_asof(recorded_ts, t0, what):
    rec = pd.to_datetime(pd.Series(recorded_ts))
    if rec.isna().any():
        raise AsOfViolation(f"{what}: {int(rec.isna().sum())} rows with no recorded_ts reached a feature at {t0.date()}")
    bad = rec > t0
    if bad.any():
        raise AsOfViolation(f"{what}: {int(bad.sum())} rows recorded after t0 = {t0.date()} "
                            f"(latest {rec.max()}) reached a feature")
    return int(len(rec))


def load_sources(world):
    D = WORLDS[world]
    ch = pd.read_csv(D + "/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "part_id", "plant_id"])
    pdw = pd.read_csv(D + "/part_demand_weekly.csv",
                      usecols=["part_id", "plant_id", "week_start", "as_of_date", "gross_requirement_p50"])
    pdw["recorded_ts"] = pd.to_datetime(pdw.as_of_date) + PDW_MAX_LAG     # derived upper bound, see the docstring
    pol = pd.read_csv(D + "/po_lines.csv", usecols=["po_line_id", "channel_id", "qty_ordered", "created_ts", "recorded_ts"])
    grn = pd.read_csv(D + "/grn_lines.csv", usecols=["po_line_id", "qty_received", "is_final_receipt", "event_ts", "recorded_ts"])
    for df, cols in ((pdw, ("week_start", "as_of_date", "recorded_ts")), (pol, ("created_ts", "recorded_ts")),
                     (grn, ("event_ts", "recorded_ts"))):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    cidx = {c: i for i, c in enumerate(ch.channel_id)}
    pp_key = (ch.part_id + "|" + ch.plant_id).to_numpy()
    pp_u = {k: i for i, k in enumerate(sorted(set(pp_key)))}
    ch_pp = np.array([pp_u[k] for k in pp_key])
    sup_u = {s: i for i, s in enumerate(sorted(ch.supplier_id.unique()))}
    ch_sup = ch.supplier_id.map(sup_u).to_numpy()
    pdw["pp"] = (pdw.part_id + "|" + pdw.plant_id).map(pp_u)
    pdw = pdw[pdw.pp.notna()].copy(); pdw["pp"] = pdw.pp.astype(np.int64)
    pol["ci"] = pol.channel_id.map(cidx).astype(np.int64)
    grn["ci"] = grn.po_line_id.map(pol.set_index("po_line_id").ci).astype(np.int64)
    grn["is_final_receipt"] = grn.is_final_receipt.astype(str).str.lower().isin(["true", "1"])
    return dict(ch=ch, pdw=pdw, pol=pol, grn=grn, ch_pp=ch_pp, ch_sup=ch_sup, n_pp=len(pp_u), n_sup=len(sup_u),
                cidx=cidx)


def snapshot_features(S, t0):
    """-> float32 [n_channels, len(COLS)] for one snapshot t0, and the number of source rows asserted."""
    t0 = pd.Timestamp(t0)
    NCH, n_pp, n_sup = len(S["ch"]), S["n_pp"], S["n_sup"]
    ch_pp, ch_sup = S["ch_pp"], S["ch_sup"]
    n_rows = 0
    # ---- plan of record at t0: latest version recorded by t0, per (part-plant, target week)
    pdw = S["pdw"]
    m = (pdw.recorded_ts.values <= t0.to_datetime64()) & (pdw.week_start.values > t0.to_datetime64()) \
        & (pdw.week_start.values <= (t0 + pd.Timedelta(weeks=13)).to_datetime64())
    used = pdw.loc[m, ["pp", "week_start", "as_of_date", "gross_requirement_p50", "recorded_ts"]]
    n_rows += assert_asof(used.recorded_ts, t0, "part_demand_weekly")
    used = used.sort_values("as_of_date", kind="stable").groupby(["pp", "week_start"], sort=False).tail(1)
    k = ((used.week_start - t0).dt.days // 7).to_numpy()
    req = np.full((n_pp, 3), np.nan); cov = np.zeros((n_pp, 3))
    for j, (a, b) in enumerate(WINDOWS):
        sel = (k >= a) & (k <= b)
        s = np.bincount(used.pp.values[sel], weights=used.gross_requirement_p50.values[sel].astype(float), minlength=n_pp)
        c = np.bincount(used.pp.values[sel], minlength=n_pp).astype(float)
        L = b - a + 1
        req[:, j] = np.where(c > 0, s / np.maximum(c, 1) * L, np.nan)        # weekly mean over covered weeks x window
        cov[:, j] = c / L
    # ---- the channel's share of its part-plant's ordered quantity, trailing 52 weeks, recorded by t0
    pol = S["pol"]
    m = (pol.recorded_ts.values <= t0.to_datetime64()) & (pol.created_ts.values > (t0 - TRAIL).to_datetime64())
    pu = pol.loc[m, ["ci", "qty_ordered", "recorded_ts"]]
    n_rows += assert_asof(pu.recorded_ts, t0, "po_lines (share)")
    q_ch = np.bincount(pu.ci.values, weights=pu.qty_ordered.values.astype(float), minlength=NCH)
    q_pp = np.bincount(ch_pp, weights=q_ch, minlength=n_pp)
    n_in_pp = np.bincount(ch_pp, minlength=n_pp).astype(float)
    share = np.where(q_pp[ch_pp] > 0, q_ch / np.maximum(q_pp[ch_pp], 1e-12), 1.0 / n_in_pp[ch_pp])
    ch_req = req[ch_pp] * share[:, None]
    sup_req = np.full((n_sup, 3), np.nan)
    for j in range(3):
        ok = np.isfinite(ch_req[:, j])
        s = np.bincount(ch_sup[ok], weights=ch_req[ok, j], minlength=n_sup)
        c = np.bincount(ch_sup[ok], minlength=n_sup)
        sup_req[:, j] = np.where(c > 0, s, np.nan)
    # ---- trailing-52-week delivered throughput, receipts recorded by t0
    grn = S["grn"]
    m = (grn.recorded_ts.values <= t0.to_datetime64()) & (grn.event_ts.values > (t0 - TRAIL).to_datetime64())
    gu = grn.loc[m, ["ci", "qty_received", "recorded_ts"]]
    n_rows += assert_asof(gu.recorded_ts, t0, "grn_lines (throughput)")
    thr_ch = np.bincount(gu.ci.values, weights=gu.qty_received.values.astype(float), minlength=NCH) / 52.0
    thr_sup = np.bincount(ch_sup, weights=thr_ch, minlength=n_sup)
    # ---- open PO quantity at t0: lines raised in the last 26 weeks, recorded by t0, net of receipts recorded by t0;
    #      a line whose final receipt is recorded by t0 is closed (a short delivery does not stay open forever)
    m = (pol.recorded_ts.values <= t0.to_datetime64()) & (pol.created_ts.values > (t0 - OPEN_LOOKBACK).to_datetime64())
    po = pol.loc[m, ["po_line_id", "ci", "qty_ordered", "recorded_ts"]]
    n_rows += assert_asof(po.recorded_ts, t0, "po_lines (open)")
    gm = (grn.recorded_ts.values <= t0.to_datetime64()) & grn.po_line_id.isin(po.po_line_id).values
    g2 = grn.loc[gm, ["po_line_id", "qty_received", "is_final_receipt", "recorded_ts"]]
    n_rows += assert_asof(g2.recorded_ts, t0, "grn_lines (open)")
    rec = g2.groupby("po_line_id").agg(q=("qty_received", "sum"), fin=("is_final_receipt", "max"))
    r_q = po.po_line_id.map(rec.q).fillna(0).to_numpy(float)
    r_f = po.po_line_id.map(rec.fin).fillna(False).to_numpy(bool)
    open_q = np.where(r_f, 0.0, np.maximum(po.qty_ordered.to_numpy(float) - r_q, 0.0))
    open_ch = np.bincount(po.ci.values, weights=open_q, minlength=NCH)
    open_sup = np.bincount(ch_sup, weights=open_ch, minlength=n_sup)

    def ratio(a, b):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)
    Lw = np.array([b - a + 1 for a, b in WINDOWS], float)
    F = np.column_stack([ch_req, ratio(ch_req, thr_ch[:, None] * Lw), ratio(open_ch, thr_ch),
                         sup_req[ch_sup], ratio(sup_req[ch_sup], thr_sup[ch_sup][:, None] * Lw),
                         ratio(open_sup, thr_sup)[ch_sup], cov[ch_pp, 2]])
    assert F.shape[1] == len(COLS)
    return F.astype(np.float32), n_rows


def snapshot_dates(world):
    lb = pd.read_csv(WORLDS[world] + "/training_labels.csv", usecols=["snapshot_date", "task"])
    lb = lb[lb.task.isin(["arrival_week", "fill_rate", "capacity_strain"])]
    d = pd.to_datetime(lb.snapshot_date)
    d = d[(d >= FIT_WINDOW[0]) & (d <= FIT_WINDOW[1])]
    return sorted(pd.unique(d))


def build(world="v8"):
    t = time.time()
    S = load_sources(world)
    snaps = snapshot_dates(world)
    X = np.zeros((len(snaps), len(S["ch"]), len(COLS)), np.float32)
    asserted = 0
    for i, t0 in enumerate(snaps):
        X[i], n = snapshot_features(S, t0)
        asserted += n
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"fwd_load_{world}.npz")
    np.savez_compressed(path, X=X, snapshots=np.array([str(pd.Timestamp(s).date()) for s in snaps]),
                        channels=S["ch"].channel_id.to_numpy().astype(str),
                        suppliers=S["ch"].supplier_id.to_numpy().astype(str), cols=np.array(COLS))
    meta = dict(world=world, n_snapshots=len(snaps), n_channels=len(S["ch"]), cols=COLS,
                source_rows_asserted_asof=int(asserted), seconds=round(time.time() - t, 1),
                missing_share={c: float(np.isnan(X[..., j]).mean()) for j, c in enumerate(COLS)})
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return path


def load(world="v8"):
    z = np.load(os.path.join(OUT_DIR, f"fwd_load_{world}.npz"), allow_pickle=False)
    return z["X"], list(z["snapshots"]), list(z["channels"]), list(z["cols"]), z["suppliers"]


def falsify(world="v8"):
    """The gate must be able to fire. Append ONE future-recorded row to each source in turn and require the STOP."""
    S = load_sources(world)
    t0 = pd.Timestamp(snapshot_dates(world)[40])
    F, n = snapshot_features(S, t0)
    out = dict(t0=str(t0.date()), clean_rows_asserted=n)
    for name, df, row in (
            ("part_demand_weekly", "pdw", dict(week_start=t0 + pd.Timedelta(weeks=2), as_of_date=t0 + pd.Timedelta(days=9),
                                               recorded_ts=t0 + pd.Timedelta(days=9))),
            ("po_lines", "pol", dict(created_ts=t0 - pd.Timedelta(days=3), recorded_ts=t0 + pd.Timedelta(days=1))),
            ("grn_lines", "grn", dict(event_ts=t0 - pd.Timedelta(days=3), recorded_ts=t0 + pd.Timedelta(hours=6)))):
        S2 = dict(S); base = S[df]
        bad = base.iloc[[0]].copy()
        for c, v in row.items():
            bad[c] = v
        S2[df] = pd.concat([base, bad], ignore_index=True)
        # bypass the row FILTER, as a defective caller would: the assertion must still fire on the rows used
        try:
            _unfiltered_check(S2, df, t0)
            out[name] = "DID NOT FIRE"
        except AsOfViolation as e:
            out[name] = f"fired: {e}"
    print(json.dumps(out, indent=1))
    assert all(v.startswith("fired") for k, v in out.items() if k in ("part_demand_weekly", "po_lines", "grn_lines"))
    return out


def _unfiltered_check(S, df, t0):
    """What a caller that forgot the recorded_ts filter would pass to the assertion: every row inside the time window."""
    d = S[df]
    if df == "pdw":
        used = d[(d.week_start > t0) & (d.week_start <= t0 + pd.Timedelta(weeks=13))]
    elif df == "pol":
        used = d[d.created_ts > t0 - TRAIL]
    else:
        used = d[d.event_ts > t0 - TRAIL]
    assert_asof(used.recorded_ts, t0, df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "falsify"])
    ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    {"build": build, "falsify": falsify}[a.mode](a.world)
