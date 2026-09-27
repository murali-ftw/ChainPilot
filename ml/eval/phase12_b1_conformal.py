"""Phase 12 Wave B1 -- Test 3.1: conformalised quantile regression on capacity intervals, v6/v7 backtest.

WHY v6/v7. The coverage failure (80% coverage 0.72-0.81, below nominal in 12 of 16 windows, every model
class) was measured on v6/v7's 16-window rolling-origin backtest. v8 has no rolling origins. Worlds are not
mixed: every number here is v6 or v7, per world, per origin.

THE METHOD (Romano, Patterson & Candes 2019), interval-only:
    E_i  = max( q_lo(x_i) - y_i , y_i - q_hi(x_i) )           on the calibration window
    Qhat = the ceil((n+1)(1-alpha))-th smallest E_i
    [q_lo - Qhat , q_hi + Qhat]                                widened (or narrowed, if Qhat < 0)
alpha = 0.20 (the P10-P90 interval, nominal 80%). CALIBRATION WINDOW: the origin's own 12-month validation
slice, which ends the day before the evaluation window starts -- the trailing window, fixed BEFORE any
result. A trailing 6-month sub-window is run as a SENSITIVITY only; nothing is selected on evaluation.

THE GUARANTEE AND ITS PREMISE. Coverage >= 1 - alpha holds in finite samples IF calibration and evaluation
rows are exchangeable. Phase 8 section 4 8D.4 showed they are not whenever the evaluation window's
utilisation level sits above the calibration window's (P90 exceedance vs that gap: Spearman ~0.8, every
class). So the guarantee holds on the OWN window almost by construction (B1.5), and the NEXT window is the
test. Two caveats on the own window: rows within a window are correlated (supplier-months recur across
snapshots), and the validation slice also chose the early-stopping epoch.

B1.2: P50 is never touched. Asserted byte-identical before and after, and the assertion is shown firing.
"""
from __future__ import annotations
import glob, os
import phase12_common as C
import numpy as np, pandas as pd
from scipy import stats
import phase5_heads as P5, folds as FO

ALPHA = 0.20
BAND = (0.78, 0.82)
PASS_WINDOWS = 14
PREDS = os.path.join(C.ART, "backtest", "preds")
BTB = os.path.join(C.ART, "backtest", "bundles", "capacity_strain")


def cqr(Pcal, ycal, alpha=ALPHA):
    E = np.maximum(Pcal[:, 0] - ycal, ycal - Pcal[:, 2])
    n = len(E)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    assert 1 <= k <= n, f"calibration window too small for alpha={alpha}: n={n}"
    return float(np.sort(E)[k - 1])


def cqr_normalised(Pcal, ycal, alpha=ALPHA):
    """Pre-declared SECONDARY: nonconformity scaled by the predicted interval width, so the widening
    grows wherever the model's own interval grows (Romano et al. sec. 6 / Sesia & Candes)."""
    w = np.maximum(Pcal[:, 2] - Pcal[:, 0], 1e-6)
    E = np.maximum(Pcal[:, 0] - ycal, ycal - Pcal[:, 2]) / w
    n = len(E); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(E)[k - 1])


def widen_norm(P, q):
    W = P.astype(np.float64).copy()
    w = np.maximum(W[:, 2] - W[:, 0], 1e-6)
    W[:, 0] -= q * w; W[:, 2] += q * w
    return W


def widen(P, q):
    W = P.astype(np.float64).copy()
    W[:, 0] -= q
    W[:, 2] += q
    return W


def assert_p50_identical(before, after):
    assert before[:, 1].tobytes() == after[:, 1].tobytes(), "P50 changed -- B1 is interval-only"
    return True


def coverage(P, y):
    return float(((y >= P[:, 0]) & (y <= P[:, 2])).mean())


def exceed(P, y):
    return float((y > P[:, 2]).mean())


def arm_files(world, o):
    out = {}
    h4 = sorted(glob.glob(f"{BTB}/o{o}/{world}_mp_h4_lr0.00025_s*"))
    h0 = sorted(glob.glob(f"{BTB}/o{o}/{world}_none_h0_lr0.00025_s*"))
    out["h4"] = [(f"{d}/preds_val.npz", f"{d}/preds_test.npz") for d in h4]
    out["h0"] = [(f"{d}/preds_val.npz", f"{d}/preds_test.npz") for d in h0]
    b5 = sorted(glob.glob(f"{PREDS}/{world}_capacity_strain_o{o}_b5flat_q_s*_test.npz"))
    out["B5"] = [(f.replace("_test.npz", "_val.npz"), f) for f in b5]
    return out


def cell(world, o, lb):
    tr, va, te = FO.rolling_split(lb.snapshot_date, o)
    ov, ot = P5.ordered(lb, va), P5.ordered(lb, te)
    yv_lb, yt_lb = lb.label_value.to_numpy(float)[ov], lb.label_value.to_numpy(float)[ot]
    dv = pd.to_datetime(lb.snapshot_date.values[ov])
    trail6 = dv > (dv.max() - pd.DateOffset(months=6))
    res = {}
    for arm, files in arm_files(world, o).items():
        per = []
        for fv, ft in files:
            zv, zt = np.load(fv), np.load(ft)
            Pv, Pt, yv, yt = zv["P"].astype(np.float64), zt["P"].astype(np.float64), zv["Y"], zt["Y"]
            aligned = len(yv) == len(yv_lb) and np.allclose(yv, yv_lb)
            assert len(yt) == len(yt_lb) and np.allclose(yt, yt_lb), f"{ft}: test rows do not align"
            q = cqr(Pv, yv)
            Wv, Wt = widen(Pv, q), widen(Pt, q)
            assert_p50_identical(Pt, Wt); assert_p50_identical(Pv, Wv)
            r = dict(file=os.path.basename(os.path.dirname(ft)) if arm != "B5" else os.path.basename(ft),
                     qhat=q, raw_cov_eval=coverage(Pt, yt), raw_exc_eval=exceed(Pt, yt),
                     raw_cov_calib=coverage(Pv, yv),
                     cqr_cov_calib=coverage(Wv, yv), cqr_cov_eval=coverage(Wt, yt), cqr_exc_eval=exceed(Wt, yt),
                     width_raw=float((Pt[:, 2] - Pt[:, 0]).mean()), width_cqr=float((Wt[:, 2] - Wt[:, 0]).mean()))
            qn = cqr_normalised(Pv, yv); Wn = widen_norm(Pt, qn); assert_p50_identical(Pt, Wn)
            r.update(qhat_norm=qn, cqrn_cov_eval=coverage(Wn, yt), cqrn_exc_eval=exceed(Wn, yt),
                     # PREMISE CHECK (B1.6): does the model's own median follow the level? Label-free on the
                     # prediction side: the P50 shift between windows vs the realised label shift.
                     p50_shift=float(Pt[:, 1].mean() - Pv[:, 1].mean()),
                     p50_bias_eval=float(Pt[:, 1].mean() - yt.mean()),
                     p50_bias_calib=float(Pv[:, 1].mean() - yv.mean()))
            if aligned:
                q6 = cqr(Pv[trail6], yv[trail6])
                r.update(qhat_trail6=q6, cqr6_cov_eval=coverage(widen(Pt, q6), yt),
                         cqr6_exc_eval=exceed(widen(Pt, q6), yt), cqr6_cov_calib=coverage(widen(Pv[trail6], q6), yv[trail6]))
            # exchangeable sanity: split the calibration window at random, calibrate on half, cover the other half
            rng = np.random.default_rng(o * 100 + len(per))
            m = rng.random(len(yv)) < 0.5
            qh = cqr(Pv[m], yv[m])
            r["exch_half_cov"] = coverage(widen(Pv[~m], qh), yv[~m])
            r["exch_half_cov_alpha50"] = coverage(widen(Pv[~m], cqr(Pv[m], yv[m], 0.5)), yv[~m])
            per.append(r)
        df = pd.DataFrame(per)
        agg = {c: [float(df[c].mean()), float(df[c].min()), float(df[c].max())]
               for c in df.columns if c != "file"}
        res[arm] = dict(n_seeds=len(per), seeds=list(df.file), stats=agg,
                        level_gap=float(yt.mean() - yv.mean()),
                        level_gap_trail6=float(yt.mean() - yv[trail6].mean()) if aligned else None)
    return res


def falsify_p50():
    P = np.array([[0.5, 0.8, 1.1], [0.4, 0.9, 1.3]])
    Q = P.copy(); Q[0, 1] += 1e-12
    try:
        assert_p50_identical(P, Q)
        return "DID NOT FIRE"
    except AssertionError:
        return "FIRES on a 1e-12 change to one P50"


def main():
    st = C.require_clean()
    out = dict(stamp=st, alpha=ALPHA, band=BAND, calibration_window="origin's 12-month validation slice (trailing)",
               sensitivity_window="last 6 months of that slice", p50_assert_falsification=falsify_p50(), cells=[])
    for world in ("v6", "v7"):
        lb = P5.labels(world, "capacity_strain")
        for o in range(1, 9):
            c = cell(world, o, lb)
            out["cells"].append(dict(world=world, origin=o, **c))
            h = c["h4"]["stats"]
            print(world, o, f"h4 n={c['h4']['n_seeds']} raw {h['raw_cov_eval'][0]:.4f} cqr-calib {h['cqr_cov_calib'][0]:.4f} "
                  f"cqr-next {h['cqr_cov_eval'][0]:.4f} qhat {h['qhat'][0]:+.4f} gap {c['h4']['level_gap']:+.4f}", flush=True)

    summ = {}
    for arm in ("h4", "h0", "B5"):
        rows = [dict(world=c["world"], origin=c["origin"], gap=c[arm]["level_gap"],
                     **{k: v[0] for k, v in c[arm]["stats"].items()}, n=c[arm]["n_seeds"]) for c in out["cells"]]
        df = pd.DataFrame(rows)
        inband = lambda s: int(((s >= BAND[0]) & (s <= BAND[1])).sum())
        s = dict(n_windows=len(df), seeds_min=int(df.n.min()), seeds_max=int(df.n.max()),
                 in_band_raw=inband(df.raw_cov_eval), in_band_cqr_calib=inband(df.cqr_cov_calib),
                 in_band_cqr_next=inband(df.cqr_cov_eval),
                 below_band_cqr_next=int((df.cqr_cov_eval < BAND[0]).sum()),
                 above_band_cqr_next=int((df.cqr_cov_eval > BAND[1]).sum()),
                 cov_range_raw=[float(df.raw_cov_eval.min()), float(df.raw_cov_eval.max())],
                 cov_range_cqr_next=[float(df.cqr_cov_eval.min()), float(df.cqr_cov_eval.max())],
                 exch_half_cov_range=[float(df.exch_half_cov.min()), float(df.exch_half_cov.max())],
                 exch_alpha50_range=[float(df.exch_half_cov_alpha50.min()), float(df.exch_half_cov_alpha50.max())])
        for lab, col in (("raw", "raw_exc_eval"), ("cqr", "cqr_exc_eval")):
            r, p = stats.spearmanr(df[col], df.gap)
            s[f"rho_exceed_vs_gap_{lab}"] = [float(r), float(p)]
        r, p = stats.spearmanr(df.cqr_cov_eval, df.gap); s["rho_cov_vs_gap_cqr"] = [float(r), float(p)]
        if "cqr6_cov_eval" in df and df.cqr6_cov_eval.notna().all():
            s["in_band_cqr6_next"] = inband(df.cqr6_cov_eval)
            r, p = stats.spearmanr(df.cqr6_exc_eval, df.gap); s["rho_exceed_vs_gap_cqr6"] = [float(r), float(p)]
        s["in_band_cqrn_next"] = inband(df.cqrn_cov_eval)
        r, p = stats.spearmanr(df.cqrn_exc_eval, df.gap); s["rho_exceed_vs_gap_cqrn"] = [float(r), float(p)]
        s["premise"] = dict(
            corr_p50_shift_vs_label_gap=float(np.corrcoef(df.p50_shift, df.gap)[0, 1]),
            slope_p50_shift_on_gap=float(np.polyfit(df.gap, df.p50_shift, 1)[0]),
            corr_eval_bias_vs_gap=float(np.corrcoef(df.p50_bias_eval, df.gap)[0, 1]),
            slope_eval_bias_on_gap=float(np.polyfit(df.gap, df.p50_bias_eval, 1)[0]),
            rho_exceed_cqr_vs_eval_bias=[float(x) for x in stats.spearmanr(df.cqr_exc_eval, -df.p50_bias_eval)])
        s["coverage_pass"] = s["in_band_cqr_next"] >= PASS_WINDOWS
        s["rho_pass"] = bool(abs(s["rho_exceed_vs_gap_cqr"][0]) <= 0.4 and s["rho_exceed_vs_gap_cqr"][1] >= 0.05)
        summ[arm] = s
    out["summary"] = summ
    for a, s in summ.items():
        print(a, s)
    print(C.dump(out, "phase12_b1.json"))


if __name__ == "__main__":
    main()
