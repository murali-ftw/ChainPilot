"""Phase 12 Wave A1 -- does arrival carry a SEASONAL residual the shipped h4 head leaves behind?

Gate for Test C1 (trend/seasonal TCN). No training. Reads the five shipped v8 h4 arrival bundles'
stored predictions (test 2025 and validation 2024) and the adopted t0 lateness reference.

THE DESIGN PROBLEM, stated before any result. v8's label snapshots are MONTHLY and SPARSE: the 2025
test fold is nine snapshots of 4,000 rows (Jan Feb Mar May Jun Aug Sep Oct Dec -- April, July and
November are absent), validation 2024 is eight. So on one fold "calendar month" and "snapshot" are
the SAME variable. A difference between months on the test fold cannot be told apart from a
one-off shock to that snapshot. The only thing that separates seasonality from a snapshot shock is
REPETITION ACROSS YEARS: a seasonal residual must recur in the same calendar month of 2024 and 2025.
Five months are shared (Feb Mar Jun Sep Oct). That is the replication test, and its power is
reported before its result.

Per row (uncensored rows only -- the lateness metric's population):
  resid   = Y - P                                  arrival-week residual, weeks
  p_late  = P(R < T <= H | T <= H) from the survival curve, H = 12
  e_late  = 1{Y > R} - p_late                      lateness calibration error
Per month: mean resid, mean e_late, lateness ROC-AUC; seed-mean prediction; row bootstrap CI.
"""
from __future__ import annotations
import itertools
import phase12_common as C
import numpy as np, pandas as pd
from scipy import stats
import phase5_heads as P5, folds
from lateness_metric import adopted_reference
from phase11b_lateness import channel_lead_history
from metrics import roc_auc

H = 12
Z80, Z05, Z05_BONF9 = 0.8416, 1.95996, 2.7729   # power 0.80; alpha 0.05 two-sided; Bonferroni over 9 months
HEAD_H4, HEAD_B5 = 0.70905, 0.70535


def season_shape(m):
    """generator_v8.season(): the demand-seasonality shape with amplitude A.seas factored out."""
    s = 0.0
    s += 1.0 if m in (8, 9, 10, 11) else 0.0
    s += 0.72 if m == 3 else 0.0
    s += 0.41 if m in (6, 7, 8) else 0.0
    s += -0.36 if m in (12, 1) else 0.0
    return s


def lag_season(m):
    """lead's congestion term reads LAST month's utilisation: sup_load_prev = util_hist[m-1]."""
    return season_shape(12 if m == 1 else m - 1)


def p_late_cond(S, R):
    """P(R < T <= H | T <= H). S[:, k] = P(T > k+1). Late means Y > R with Y integer weeks."""
    S = np.asarray(S, float)
    n = len(S)
    Sfull = np.concatenate([np.ones((n, 1)), S], 1)          # Sfull[:, k] = P(T > k), k = 0..12
    k = np.clip(np.floor(R).astype(int), 0, H)
    s_r = Sfull[np.arange(n), k]
    s_h = Sfull[:, H]
    return np.clip((s_r - s_h) / np.maximum(1 - s_h, 1e-9), 0, 1)


def fold_rows(lb, mask, R, seeds, which):
    o = P5.ordered(lb, mask)
    Ps, Ss, per_seed = [], [], {}
    for s in seeds:
        z = np.load(f"{C.BUND}/arrival_week/v8_lite_h4_lr0.00025_s{s}/preds_{which}.npz")
        assert np.allclose(z["Y"], lb.label_value.values[o]), "prediction rows do not align to labels"
        Ps.append(z["P"].astype(float)); Ss.append(z["S"].astype(float)); per_seed[s] = z
    Y, EV = per_seed[seeds[0]]["Y"], per_seed[seeds[0]]["EV"].astype(bool)
    Ro = R[o]
    snap = pd.to_datetime(lb.snapshot_date.values[o])
    d = pd.DataFrame(dict(month=snap.month, snap=snap, Y=Y, EV=EV, R=Ro))
    for s, P, S in zip(seeds, Ps, Ss):
        d[f"P_{s}"] = P
        d[f"pl_{s}"] = p_late_cond(S, Ro)
    d["P"] = np.mean(Ps, 0)
    d["pl"] = p_late_cond(np.mean(Ss, 0), Ro)
    d = d[d.EV & np.isfinite(d.R)].copy()
    d["yl"] = (d.Y > d.R).astype(int)
    d["resid"] = d.Y - d.P
    d["e_late"] = d.yl - d.pl
    for s in seeds:
        d[f"resid_{s}"] = d.Y - d[f"P_{s}"]
        d[f"e_late_{s}"] = d.yl - d[f"pl_{s}"]
    return d


def per_month(d, seeds, tag):
    rows = []
    for m, g in d.groupby("month"):
        rest = d[d.month != m]
        r_mu, r_lo, r_hi, r_se = C.boot_mean_ci(g.resid.values - d.resid.mean(), seed=m)
        e_mu, e_lo, e_hi, e_se = C.boot_mean_ci(g.e_late.values - d.e_late.mean(), seed=100 + m)
        _, _, _, r_se_rest = C.boot_mean_ci(rest.resid.values, B=500, seed=200 + m)
        _, _, _, e_se_rest = C.boot_mean_ci(rest.e_late.values, B=500, seed=300 + m)
        aucs = [roc_auc(g.yl.values, g[f"P_{s}"].values - g.R.values) for s in seeds]
        rng = np.random.default_rng(400 + m)
        pl = g.P.values - g.R.values
        bauc = []
        for _ in range(300):
            i = rng.integers(0, len(g), len(g))
            if g.yl.values[i].min() != g.yl.values[i].max():
                bauc.append(roc_auc(g.yl.values[i], pl[i]))
        auc_se = float(np.std(bauc))
        rows.append(dict(
            fold=tag, month=int(m), snapshot=str(g.snap.iloc[0].date()), n_uncensored=int(len(g)),
            late_rate=float(g.yl.mean()), mean_Y=float(g.Y.mean()), mean_P=float(g.P.mean()),
            resid_centered=r_mu, resid_ci=[r_lo, r_hi], resid_se=r_se,
            resid_seed_range=[float(min((g[f"resid_{s}"].mean() - d[f"resid_{s}"].mean()) for s in seeds)),
                              float(max((g[f"resid_{s}"].mean() - d[f"resid_{s}"].mean()) for s in seeds))],
            e_late_centered=e_mu, e_late_ci=[e_lo, e_hi], e_late_se=e_se,
            auc_seed_mean=float(np.mean(aucs)), auc_seed_range=[float(min(aucs)), float(max(aucs))],
            auc_boot_se=auc_se,
            mde_resid_vs_rest=float((Z05 + Z80) * np.hypot(r_se, r_se_rest)),
            mde_resid_vs_rest_bonf9=float((Z05_BONF9 + Z80) * np.hypot(r_se, r_se_rest)),
            mde_e_late_vs_rest=float((Z05 + Z80) * np.hypot(e_se, e_se_rest)),
            mde_auc=float((Z05 + Z80) * auc_se * np.sqrt(2)),
            lag_season=lag_season(int(m)), season=season_shape(int(m))))
    return pd.DataFrame(rows)


def anova(d, col):
    groups = [g[col].values for _, g in d.groupby("month")]
    F, p = stats.f_oneway(*groups)
    return dict(F=float(F), p=float(p), k=len(groups), n=int(len(d)))


def exact_perm_corr(x, y):
    """Exact permutation p-value (one-sided, r >= observed) over all orderings of y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r0 = float(np.corrcoef(x, y)[0, 1])
    rs = np.array([np.corrcoef(x, np.array(p))[0, 1] for p in itertools.permutations(y)])
    return r0, float((rs >= r0 - 1e-12).mean()), rs


def replication_power(n_perm_rs, rho_true, sd_noise_ratio, n=5, sims=4000, seed=0):
    """Power of the exact permutation test at alpha 0.05 for a seasonal signal with true
    between-year correlation rho_true (simulated bivariate normal, n shared months)."""
    rng = np.random.default_rng(seed)
    crit = np.quantile(n_perm_rs, 0.95)
    cov = [[1, rho_true], [rho_true, 1]]
    hits = 0
    for _ in range(sims):
        z = rng.multivariate_normal([0, 0], cov, n)
        hits += np.corrcoef(z[:, 0], z[:, 1])[0, 1] > crit
    return float(hits / sims), float(crit)


def raw_lead_seasonality():
    """Is the generator's path VISIBLE in raw data? Mean observed lead of GRN lines by the calendar
    month the PO was raised, training years only (<= 2023), against the lagged seasonal shape."""
    D = C.REPO + "/db/gen_v8/seed_1001"
    gl = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "event_ts"])
    pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "created_ts"])
    g = gl.merge(pl, on="po_line_id")
    ct = pd.to_datetime(g.created_ts)
    g["lead_w"] = (pd.to_datetime(g.event_ts) - ct).dt.days / 7.0
    g = g[(ct.dt.year >= 2019) & (ct.dt.year <= 2023) & np.isfinite(g.lead_w) & (g.lead_w >= 0)]
    g["month"] = ct[g.index].dt.month
    by = g.groupby("month").lead_w.agg(["mean", "count"])
    by["lag_season"] = [lag_season(m) for m in by.index]
    by["season"] = [season_shape(m) for m in by.index]
    r_lag = float(np.corrcoef(by["mean"], by.lag_season)[0, 1])
    r_now = float(np.corrcoef(by["mean"], by.season)[0, 1])
    # year-by-year replication of the raw monthly pattern (is it seasonal or noise?)
    g["year"] = ct[g.index].dt.year
    piv = g.groupby(["year", "month"]).lead_w.mean().unstack()
    yr_corr = piv.T.corr().values[np.triu_indices(len(piv), 1)]
    return dict(by_month={int(m): dict(mean_lead_w=float(r["mean"]), n=int(r["count"]),
                                        lag_season=float(r.lag_season)) for m, r in by.iterrows()},
                amplitude_weeks=float(by["mean"].max() - by["mean"].min()),
                corr_with_lag_season=r_lag, corr_with_same_month_season=r_now,
                between_year_corr_of_monthly_pattern=dict(mean=float(yr_corr.mean()),
                                                          min=float(yr_corr.min()), max=float(yr_corr.max())))


def pooled_lag_test(mt, mv, n_perm=20000, seed=0):
    """Both folds' snapshots pooled (17), residual against the generator's LAGGED seasonal shape.
    The shape is fixed a priori from generator_v8.season(), so this is one pre-specified contrast,
    far more powerful than the 5-month replication. Permutation p over snapshot labels."""
    m = pd.concat([mt, mv])
    out = {}
    rng = np.random.default_rng(seed)
    for col in ("resid_centered", "e_late_centered"):
        x, y = m.lag_season.values, m[col].values
        r0 = float(np.corrcoef(x, y)[0, 1])
        rs = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n_perm)])
        out[col] = dict(r=r0, p_perm_one_sided=float((rs >= r0).mean()), n_snapshots=int(len(m)),
                        slope_per_unit_lag_season=float(np.polyfit(x, y, 1)[0]))
    return out


def val_fitted_seasonal_shift(dva, dte, seeds):
    """The CEILING of a month-level seasonal correction, measured without training.
    Fit resid ~ b * lag_season(month) on VALIDATION ONLY (selection on validation), shift every test
    prediction by b * lag_season(month), rescore lateness ROC-AUC per seed. A seasonal branch can do
    more than a month-level shift (per-supplier timing), so this bounds the month-level part only."""
    res = {}
    for s in seeds:
        xv = np.array([lag_season(int(m)) for m in dva.month])
        b = float(np.polyfit(xv, dva[f"resid_{s}"].values, 1)[0])
        xt = np.array([lag_season(int(m)) for m in dte.month])
        base = roc_auc(dte.yl.values, dte[f"P_{s}"].values - dte.R.values)
        adj = roc_auc(dte.yl.values, dte[f"P_{s}"].values + b * xt - dte.R.values)
        res[str(s)] = dict(slope_val=b, auc_base=float(base), auc_shifted=float(adj), delta=float(adj - base))
    d = [v["delta"] for v in res.values()]
    return dict(per_seed=res, delta_mean=float(np.mean(d)), delta_range=[float(min(d)), float(max(d))],
                base_band=[float(min(v["auc_base"] for v in res.values())), float(max(v["auc_base"] for v in res.values()))],
                shifted_band=[float(min(v["auc_shifted"] for v in res.values())), float(max(v["auc_shifted"] for v in res.values()))])


def main():
    st = C.require_clean()
    lb = P5.labels("v8", "arrival_week")
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    R, meta = adopted_reference("v8", lb, tr)
    seeds = list(C.V8_SEEDS)
    dte = fold_rows(lb, te, R, seeds, "test")
    dva = fold_rows(lb, va, R, seeds, "val")

    # reproduction check: the fold-level headline must come back, or nothing below is trusted
    fold_auc = [roc_auc(dte.yl.values, dte[f"P_{s}"].values - dte.R.values) for s in seeds]
    assert abs(np.mean(fold_auc) - HEAD_H4) < 5e-5, f"did not reproduce 0.70905: {np.mean(fold_auc)}"

    mt, mv = per_month(dte, seeds, "test_2025"), per_month(dva, seeds, "val_2024")

    # the null of no monthly structure, per fold (== no SNAPSHOT structure on one fold)
    tests = {f: dict(resid=anova(d, "resid"), e_late=anova(d, "e_late"))
             for f, d in (("test_2025", dte), ("val_2024", dva))}
    # seed consistency of the month profile on test
    prof = np.array([[dte[dte.month == m][f"resid_{s}"].mean() - dte[f"resid_{s}"].mean()
                      for m in sorted(dte.month.unique())] for s in seeds])
    seed_prof_corr = np.corrcoef(prof)[np.triu_indices(len(seeds), 1)]

    # THE seasonal test: does the month effect recur in the same calendar month across years?
    shared = sorted(set(mt.month) & set(mv.month))
    xt = mt.set_index("month").loc[shared]
    xv = mv.set_index("month").loc[shared]
    rep = {}
    for col in ("resid_centered", "e_late_centered", "auc_seed_mean"):
        r0, p, rs = exact_perm_corr(xv[col].values, xt[col].values)
        pw = {str(rho): replication_power(rs, rho, 1.0)[0] for rho in (0.5, 0.7, 0.9)}
        rep[col] = dict(r=r0, p_exact_one_sided=p, n_months=len(shared), critical_r_05=float(np.quantile(rs, 0.95)),
                        power_at_true_rho=pw)

    # consistency with the generator's path (A1.4)
    path = {}
    for f, m in (("test_2025", mt), ("val_2024", mv)):
        path[f] = dict(
            resid_vs_lag_season=float(np.corrcoef(m.resid_centered, m.lag_season)[0, 1]),
            e_late_vs_lag_season=float(np.corrcoef(m.e_late_centered, m.lag_season)[0, 1]),
            meanY_vs_lag_season=float(np.corrcoef(m.mean_Y, m.lag_season)[0, 1]),
            meanP_tracks_meanY=float(np.corrcoef(m.mean_P, m.mean_Y)[0, 1]),
            meanY_range_weeks=float(m.mean_Y.max() - m.mean_Y.min()),
            resid_range_weeks=float(m.resid_centered.max() - m.resid_centered.min()))
    raw = raw_lead_seasonality()
    pooled = pooled_lag_test(mt, mv)
    ceiling = val_fitted_seasonal_shift(dva, dte, seeds)

    # power, stated before the result (A1.3)
    dec = mt[mt.month == 12].iloc[0]
    power = dict(
        test_fold_rows=int(te.sum()), test_fold_uncensored=int(len(dte)),
        december_rows=4000, december_uncensored=int(dec.n_uncensored),
        full_fold_seed_spread_auc=float(max(fold_auc) - min(fold_auc)),
        per_month_auc_boot_se=dict(min=float(mt.auc_boot_se.min()), max=float(mt.auc_boot_se.max())),
        per_month_auc_seed_spread=dict(min=float(min(r[1] - r[0] for r in mt.auc_seed_range)),
                                       max=float(max(r[1] - r[0] for r in mt.auc_seed_range))),
        mde_auc_one_month=dict(min=float(mt.mde_auc.min()), max=float(mt.mde_auc.max()),
                               december=float(dec.mde_auc)),
        mde_resid_weeks_one_month_vs_rest=dict(min=float(mt.mde_resid_vs_rest.min()),
                                               max=float(mt.mde_resid_vs_rest.max()),
                                               bonferroni9_max=float(mt.mde_resid_vs_rest_bonf9.max())),
        mde_e_late_one_month_vs_rest=dict(min=float(mt.mde_e_late_vs_rest.min()),
                                          max=float(mt.mde_e_late_vs_rest.max())),
        headroom_auc_vs_b5flat=round(HEAD_H4 - HEAD_B5, 5),
        headroom_if_concentrated_in_k_months={k: round((HEAD_H4 - HEAD_B5) * 9 / k, 4) for k in (1, 2, 3, 9)},
        replication=dict(shared_months=shared, n=len(shared),
                         critical_r=rep["resid_centered"]["critical_r_05"],
                         power_at_true_rho=rep["resid_centered"]["power_at_true_rho"]),
        note=("row bootstrap treats rows as independent; rows in one snapshot share its shocks, so "
              "every per-month SE here is a LOWER bound and every MDE an optimistic one"))

    out = dict(stamp=st, reference=meta, fold_auc_per_seed=dict(zip(map(str, seeds), fold_auc)),
               fold_auc_mean=float(np.mean(fold_auc)),
               power=power, per_month=pd.concat([mt, mv]).to_dict("records"),
               null_tests=tests, seed_profile_corr=dict(mean=float(seed_prof_corr.mean()),
                                                        min=float(seed_prof_corr.min())),
               replication=rep, pooled_lag_test=pooled, month_shift_ceiling=ceiling, generator_path=path, raw_lead_seasonality=raw)
    print(C.dump(out, "phase12_a1.json"))


if __name__ == "__main__":
    main()
