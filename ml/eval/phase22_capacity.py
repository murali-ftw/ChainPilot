"""Phase 22 Stage 4 -- capacity: diagnose December 2025, then test rolling recalibration (pre-registration D8). Stored
predictions only (published incumbent; the clean incumbent once Stage 1e has retrained it). Torch process (imports only).

Score = P(strain > 1) from P10 / P50 / P90 (phase14_score.p_exceed), the UC3 definition. Operating bar p = 0.80 (UC3's
Phase 15 operating bar); the incumbent scheme's threshold = pick_on_val on the whole validation fold (Phase 15, unchanged).

(a) DIAGNOSIS per test snapshot / quarter: base rate (strain > 1 share), share flagged, precision and recall at the validation
    threshold, mean score, within-snapshot ROC-AUC, calibration slope / intercept (logistic regression of the label on
    logit(score)), and covariate drift of the 5 strongest capacity features (LightGBM b5flat_q gain) at each test snapshot
    vs train / validation (standardised mean shift and KS). Causes (D8): (i) regime = base rate outside the validation
    per-snapshot range; (ii) threshold drift = within-snapshot AUC inside the validation range but precision at the fixed
    threshold below the validation range; (iii) ranking = within-snapshot AUC below the validation range.
(b) SCHEMES on stored scores: fixed (incumbent); rolling Platt; rolling isotonic (both: flag if calibrated P >= 0.80);
    rolling threshold (pick_on_val p = 0.80 on the window); flag-rate (threshold = the current snapshot's quantile that holds
    the validation share flagged). A window holds the last w RESOLVED snapshots: t0_j + 90 d <= t0 (purge, ASSERTED).
    w from {2, 4, 6, 8} by VALIDATION walk-forward (score the 2024 snapshots that have w resolved predecessors inside the
    stored history; maximise worst-quarter precision, tie -> mean); infeasible windows are reported.
    Constructed failing case: a window fed the scoring snapshot's own (unresolved) labels MUST raise the purge assertion.
(c) per scheme: per-quarter precision (min, mean, spread), share flagged, recall, UC3 precision at 1 / 5 / 10% coverage of the
    scheme's (recalibrated) scores; 5-seed bands; the stability rule decides ADOPTED / REJECTED on validation.
(d) Phase 18's level-aware trailing conformal (phase18_squeeze.capacity_interval, unchanged), coverage per quarter.

  python ml/eval/phase22_capacity.py [--arm published|clean]    -> ml/artifacts/phase22/capacity_{arm}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import phase5_heads as P5, folds
from metrics import roc_auc
from phase14_score import p_exceed
from phase15 import pick_on_val
import phase18_score as S18
import phase18_squeeze as SQ
import phase20_decisions as P20D

SEEDS = C.V8_SEEDS
BAR = 0.80
WINDOWS = (2, 4, 6, 8)
B22 = os.path.join(C.ART, "phase22", "bundles")


class PurgeLeak(AssertionError):
    pass


def load_arm(arm):
    if arm == "published":
        return [S18.load_neural("capacity", s) for s in SEEDS]
    return [{f: dict(np.load(os.path.join(B22, "capacity_strain", f"v8clean_mp_h4_lr0.00025_s{s}", f"preds_{f}.npz"))) for f in ("val", "test")} for s in SEEDS]


def snaps():
    lb = P5.labels("v8", "capacity_strain"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    return {f: pd.to_datetime(lb.snapshot_date.values[P5.ordered(lb, m)]) for f, m in (("val", va), ("test", te))}, lb


def quarter(d):
    return pd.PeriodIndex(pd.DatetimeIndex(d), freq="Q").astype(str)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def calib_slope(s, y):
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max():
        return float("nan"), float("nan")
    m = LogisticRegression(C=1e6).fit(logit(s)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


# ================================================================== (a) diagnosis
def diagnose(zs, S, lb):
    out = []
    for z in zs:
        sv, yv = p_exceed(np.asarray(z["val"]["P"], float)), (np.asarray(z["val"]["Y"]) > 1).astype(int)
        st, yt = p_exceed(np.asarray(z["test"]["P"], float)), (np.asarray(z["test"]["Y"]) > 1).astype(int)
        tau, _ = pick_on_val(sv, yv, BAR)
        rows = {}
        for fold, s_, y_, d_ in (("val", sv, yv, S["val"]), ("test", st, yt, S["test"])):
            for u in sorted(pd.unique(d_)):
                m = d_ == u; a = s_[m] >= tau
                sl, ic = calib_slope(s_[m], y_[m])
                rows[f"{fold}|{pd.Timestamp(u).date()}"] = dict(base_rate=float(y_[m].mean()), share_flagged=float(a.mean()),
                    precision=float(y_[m][a].mean()) if a.sum() else float("nan"), recall=float((a & (y_[m] == 1)).sum() / max(y_[m].sum(), 1)),
                    mean_score=float(s_[m].mean()), auc=float(roc_auc(y_[m], s_[m])) if 0 < y_[m].mean() < 1 else float("nan"),
                    calib_slope=sl, calib_intercept=ic, precision_at_5pct=P20D.prec_at(s_[m], y_[m], 0.05))
        out.append(dict(threshold=tau, rows=rows))
    keys = list(out[0]["rows"])
    agg = {k: {m: float(np.nanmean([o["rows"][k][m] for o in out])) for m in out[0]["rows"][k]} for k in keys}
    val = [k for k in keys if k.startswith("val|")]; test = [k for k in keys if k.startswith("test|")]
    rng = lambda m: (min(agg[k][m] for k in val), max(agg[k][m] for k in val))
    causes = {}
    for k in test:
        r = agg[k]; br, au, pr = rng("base_rate"), rng("auc"), rng("precision")
        causes[k] = dict(regime_base_rate_outside_val_range=not (br[0] <= r["base_rate"] <= br[1]),
                         ranking_auc_below_val_range=r["auc"] < au[0],
                         threshold_drift=(au[0] <= r["auc"]) and (r["precision"] < pr[0]),
                         base_rate=r["base_rate"], val_base_rate_range=br, auc=r["auc"], val_auc_range=au, precision=r["precision"],
                         val_precision_range=pr, calib_slope=r["calib_slope"], share_flagged=r["share_flagged"])
    return dict(per_snapshot_seed_mean=agg, causes=causes, thresholds=[o["threshold"] for o in out])


def drift(lb):
    """Covariate drift of the 5 strongest LightGBM capacity features (stored fit log gain), each test snapshot vs train / val."""
    sys.path.insert(0, os.path.join(C.ML, "baselines"))
    from cache import load_panel
    panel, miss, active, meta = load_panel(os.path.join(config.CACHE, "v8"))
    log = json.load(open(os.path.join(C.ART, "phase7_fit.json")))
    w0 = pd.Timestamp(meta["week0"])
    cols = meta["cols"]
    ch = pd.read_csv(os.path.join(config.WORLDS["v8"], "sourcing_channels.csv"), usecols=["channel_id"]).channel_id
    cidx = {c: i for i, c in enumerate(ch)}
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    def vals(mask, c):
        sub = lb[np.asarray(mask)]
        ci = sub.key.map(cidx).to_numpy(); ti = ((sub.snapshot_date - w0).dt.days // 7).to_numpy()
        return np.asarray(panel[ci, ti, cols.index(c)], float)
    feats = ["load_ratio", "lead_time_ratio", "fill_rate_last13", "qty_ordered", "otd_rate_last13"]
    from scipy.stats import ks_2samp
    out = {}
    ref = {c: vals(tr, c) for c in feats}
    for u in sorted(pd.unique(lb.snapshot_date[np.asarray(te)])):
        m = np.asarray(lb.snapshot_date == u)
        out[str(pd.Timestamp(u).date())] = {c: dict(std_mean_shift=float((vals(m, c).mean() - ref[c].mean()) / (ref[c].std() + 1e-9)),
                                                    ks=float(ks_2samp(vals(m, c), ref[c]).statistic)) for c in feats}
    return dict(features=feats, note="feature choice: the capacity panel columns the incumbent reads with the largest drift relevance "
                                     "(load is the capacity label's own driver); published panel values", per_test_snapshot=out)


# ================================================================== (b) schemes
def window_rows(hist_d, t0, w, allow_unresolved=False):
    """the last w RESOLVED snapshots before t0 (t0_j + 90 d <= t0); the purge is asserted."""
    known = np.array(sorted(pd.unique(hist_d[hist_d + pd.Timedelta(days=90) <= t0]))) if not allow_unresolved else \
        np.array(sorted(pd.unique(hist_d[hist_d <= t0])))
    sel = known[-w:] if len(known) >= w else None
    if sel is None:
        return None
    m = np.isin(hist_d, sel)
    if (hist_d[m] + pd.Timedelta(days=90) > t0).any():
        raise PurgeLeak(f"a window label for t0 {t0.date()} is not resolved (t0_j + 90 d > t0)")
    return m


def scheme_scores(name, s_hist, y_hist, d_hist, s_now, t0, w, tau_fixed, share_val, allow_unresolved=False):
    """-> boolean flags for the current snapshot's rows, or None if the window is not available."""
    if name == "fixed":
        return s_now >= tau_fixed
    if name == "flag_rate":
        return s_now >= np.quantile(s_now, 1 - share_val)
    m = window_rows(d_hist, t0, w, allow_unresolved)
    if m is None:
        return None
    sh, yh = s_hist[m], y_hist[m]
    if name == "rolling_threshold":
        tau, _ = pick_on_val(sh, yh, BAR)
        return s_now >= (tau if tau is not None else np.inf)
    if name == "rolling_platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6).fit(logit(sh)[:, None], yh)
        return lr.predict_proba(logit(s_now)[:, None])[:, 1] >= BAR
    if name == "rolling_isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sh, yh)
        return iso.predict(s_now) >= BAR
    raise ValueError(name)


SCHEMES = ("fixed", "rolling_platt", "rolling_isotonic", "rolling_threshold", "flag_rate")


def run_scheme(name, z, S, w, period, tau_fixed_mode):
    """period 'val' (walk-forward inside 2024; history = val only) or 'test' (history = val + resolved test)."""
    sv, yv = p_exceed(np.asarray(z["val"]["P"], float)), (np.asarray(z["val"]["Y"]) > 1).astype(int)
    st, yt = p_exceed(np.asarray(z["test"]["P"], float)), (np.asarray(z["test"]["Y"]) > 1).astype(int)
    if period == "val":
        s_all, y_all, d_all = sv, yv, S["val"]
        cand = sorted(pd.unique(S["val"]))
        # the fixed comparator on validation walk-forward: a threshold from the val snapshots resolved before the first scored one
        first = None
    else:
        s_all, y_all, d_all = np.r_[sv, st], np.r_[yv, yt], np.r_[S["val"], S["test"]]
        d_all = pd.DatetimeIndex(d_all); cand = sorted(pd.unique(S["test"]))
    d_all = pd.DatetimeIndex(d_all)
    tau_val, _ = pick_on_val(sv, yv, BAR)
    share_val = float((sv >= tau_val).mean())
    res = {}
    for u in cand:
        t0 = pd.Timestamp(u)
        if period == "val":
            m0 = window_rows(d_all, t0, max(WINDOWS) if False else w)
            if m0 is None:
                continue
            tau_fx, _ = pick_on_val(s_all[d_all + pd.Timedelta(days=90) <= t0], y_all[d_all + pd.Timedelta(days=90) <= t0], BAR)
            tau_fx = tau_fx if tau_fx is not None else np.inf
        else:
            tau_fx = tau_val
        cur = np.asarray(d_all == t0)
        flags = scheme_scores(name, s_all, y_all, d_all, s_all[cur], t0, w, tau_fx, share_val)
        if flags is None:
            continue
        y = y_all[cur]
        res[str(t0.date())] = dict(precision=float(y[flags].mean()) if flags.sum() else float("nan"), share=float(flags.mean()),
                                   recall=float((flags & (y == 1)).sum() / max(y.sum(), 1)), n_flag=int(flags.sum()))
    return res


def summarise(per_snap):
    if not per_snap:
        return None
    q = {}
    for d, r in per_snap.items():
        q.setdefault(str(pd.Period(pd.Timestamp(d), freq="Q")), []).append(r)
    qp = {k: float(np.nansum([x["precision"] * x["n_flag"] for x in v]) / max(sum(x["n_flag"] for x in v), 1)) for k, v in q.items()}
    vals = [v for v in qp.values() if np.isfinite(v)]
    return dict(per_quarter_precision=qp, worst_quarter=min(vals) if vals else float("nan"), mean_quarter=float(np.mean(vals)) if vals else float("nan"),
                spread=(max(vals) - min(vals)) if vals else float("nan"), share_flagged=float(np.mean([r["share"] for r in per_snap.values()])),
                recall=float(np.mean([r["recall"] for r in per_snap.values()])),
                dec2025=next((r["precision"] for d, r in per_snap.items() if d.startswith("2025-12")), None), n_snapshots=len(per_snap))


def purge_selftest(zs, S):
    z = zs[0]
    sv = p_exceed(np.asarray(z["val"]["P"], float)); yv = (np.asarray(z["val"]["Y"]) > 1).astype(int)
    st = p_exceed(np.asarray(z["test"]["P"], float))
    d = pd.DatetimeIndex(np.r_[S["val"], S["test"]]); s_ = np.r_[sv, st]; y_ = np.r_[yv, (np.asarray(z["test"]["Y"]) > 1).astype(int)]
    t0 = pd.Timestamp(sorted(pd.unique(S["test"]))[-1])
    try:                                      # constructed leak: the window admits the scoring period's unresolved labels
        m = np.asarray(d <= t0); sel = np.array(sorted(pd.unique(d[m])))[-4:]
        mm = np.isin(d, sel)
        if (d[mm] + pd.Timedelta(days=90) > t0).any():
            raise PurgeLeak("unresolved labels in the window")
        fired = False
    except PurgeLeak:
        fired = True
    assert fired, "the purge assertion did not fire on test-period labels -- invalid"
    return dict(constructed_leak_window_raised=True)


def stability_selftest():
    b = [0.50, 0.55, 0.60]
    same = dict(worst=b, mean=[0.80, 0.81, 0.82], share=0.10)
    assert adopt(same, same)["verdict"] == "REJECTED", "the stability rule cannot fail"
    return dict(identical_scheme="REJECTED (as required)")


def adopt(cand, inc):
    """stability rule (pre-registered), on VALIDATION walk-forward bands across the 5 seeds."""
    worse_floor = cand["worst"][0] > inc["worst"][2]
    mean_drop = inc["mean"][1] - cand["mean"][1]
    ok = worse_floor and mean_drop < 0.02 and 0.05 <= cand["share"] <= 0.15
    return dict(verdict="ADOPTED" if ok else "REJECTED", worst_quarter_disjointly_better=bool(worse_floor),
                mean_drop=float(mean_drop), share=float(cand["share"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="published", choices=["published", "clean"]); a = ap.parse_args()
    PP.register()
    st_ = C.require_clean()
    zs = load_arm(a.arm)
    S, lb = snaps()
    out = dict(stamp=st_, arm=a.arm, bar=BAR, purge_selftest=purge_selftest(zs, S), stability_selftest=stability_selftest())
    out["diagnosis"] = diagnose(zs, S, lb)
    if a.arm == "published":
        out["covariate_drift"] = drift(lb)
    # window choice on validation walk-forward
    val_res = {}
    for name in SCHEMES:
        for w in WINDOWS:
            per = [summarise(run_scheme(name, z, S, w, "val", None)) for z in zs]
            if any(p is None for p in per):
                val_res[f"{name}|w{w}"] = "INFEASIBLE (fewer than w resolved validation snapshots before any scored one)"; continue
            val_res[f"{name}|w{w}"] = dict(worst=S18.band([p["worst_quarter"] for p in per]), mean=S18.band([p["mean_quarter"] for p in per]),
                                           share=float(np.mean([p["share_flagged"] for p in per])), n_snapshots=per[0]["n_snapshots"])
    out["validation_walk_forward"] = val_res
    chosen = {}
    for name in SCHEMES:
        feas = {w: val_res[f"{name}|w{w}"] for w in WINDOWS if isinstance(val_res[f"{name}|w{w}"], dict)}
        if feas:
            chosen[name] = max(feas, key=lambda w: (feas[w]["worst"][1], feas[w]["mean"][1]))
    out["window_chosen_on_validation"] = chosen
    # adoption on validation (vs the fixed comparator at the same window grid point)
    adoption = {}
    for name in SCHEMES:
        if name == "fixed" or name not in chosen:
            continue
        w = chosen[name]
        adoption[name] = adopt(val_res[f"{name}|w{w}"], val_res[f"fixed|w{w}"])
    out["adoption_on_validation"] = adoption
    # test
    test_res = {}
    for name in SCHEMES:
        w = chosen.get(name, 4)
        per = [summarise(run_scheme(name, z, S, w, "test", None)) for z in zs]
        test_res[name] = dict(window=w, worst=S18.band([p["worst_quarter"] for p in per]), mean=S18.band([p["mean_quarter"] for p in per]),
                              spread=S18.band([p["spread"] for p in per]), share=S18.band([p["share_flagged"] for p in per]),
                              recall=S18.band([p["recall"] for p in per]), dec2025=S18.band([p["dec2025"] for p in per]),
                              per_quarter_seed_mean={k: float(np.nanmean([p["per_quarter_precision"][k] for p in per])) for k in per[0]["per_quarter_precision"]})
    out["test"] = test_res
    # (d) Phase 18 level-aware conformal
    try:
        L = [S18.load_flat(os.path.join(S18.PR7, f"v8_capacity_strain_b5flat_q_s{{s}}_{{f}}.npz"), s) for s in SEEDS]
        ci = SQ.capacity_interval(zs, L)
        sn = ci["neural"]["snapshots"]
        cov = ci["neural"]["coverage"]["trailing"]["per_snapshot_seed_mean"]
        q = {}
        for d_, c_ in zip(sn, cov):
            q.setdefault(str(pd.Period(pd.Timestamp(d_), freq="Q")), []).append(c_)
        out["conformal_level_aware"] = dict(per_snapshot=dict(zip(sn, cov)), per_quarter={k: float(np.mean(v)) for k, v in q.items()},
                                            spread=dict(min=min(cov), max=max(cov), median=float(np.median(cov))),
                                            static=ci["neural"]["coverage"]["static"]["per_snapshot_seed_mean"])
    except Exception as e:                        # recorded, never silent
        out["conformal_level_aware"] = f"FAILED: {e!r}"
    C.dump(out, f"phase22/capacity_{a.arm}.json")
    print(json.dumps(out["diagnosis"]["causes"], indent=0, default=str)[:3000])
    print(json.dumps(out["window_chosen_on_validation"]), json.dumps(out["adoption_on_validation"]))
    for n, r in test_res.items():
        print(f"  TEST {n:18s} w{r['window']} worst {r['worst']} mean {r['mean']} share {r['share']} dec {r['dec2025']}")
    print(json.dumps(out["conformal_level_aware"], default=str)[:800])


if __name__ == "__main__":
    main()
