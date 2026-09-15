"""Phase 5 — metrics for the new heads, beside the Phase 2-4 ones in metrics.py.

fill      CRPS two ways (legacy step formula, for continuity; exact integral, which can see a point
          mass) and ECE by bin reliability on two partitions (legacy 20 bins; 22 cells with the
          endpoints separate). Quantile coverage is deliberately absent -- degenerate on this label.
arrival   C-index, ROC-AUC on lateness, calibration of the predicted arrival-week distribution
capacity  pinball per quantile, 80% interval coverage, crossing rate, rank correlation of P50
shortage  PR-AUC, ROC-AUC (metrics.py)
"""
from __future__ import annotations
import numpy as np
from metrics import cindex, roc_auc, pr_auc, bootstrap_ci, crps as legacy_crps_mean

LEGACY_EDGES = np.linspace(0, 1, 21)


# ================================================================== fill
def legacy_bin(y):
    return np.clip(np.digitize(np.asarray(y, float), LEGACY_EDGES[1:-1]), 0, 19)


def legacy_to_cells(P20):
    """A 20-bin forecaster has no point masses: its bin 0 is spread over [0, .05), bin 19 over [.95, 1]."""
    P20 = np.asarray(P20, float)
    z = np.zeros((len(P20), 1))
    return np.concatenate([z, P20, z], 1)


def crps_legacy_rows(P20, y):
    """Per-row version of metrics.crps -- the formula every earlier report quotes."""
    cdf = np.cumsum(np.asarray(P20, float), 1)
    ind = (np.asarray(y, float)[:, None] <= LEGACY_EDGES[None, 1:]).astype(float)
    return (((cdf - ind) ** 2) * 0.05).sum(1)


def crps_exact_rows(P22, y):
    """Exact CRPS = integral_0^1 (F(x) - 1[x >= y])^2 dx of the 22-cell distribution, per row.

    Cell 0 is a point mass at 0, cell 21 a point mass at 1, cells 1..20 uniform over their bin.
    F is piecewise linear, so each piece integrates in closed form: over a stretch of length l on
    which F - I runs linearly from a to b, the integral is l (a^2 + ab + b^2) / 3. Each bin is split
    at y when y falls inside it.
    """
    P22 = np.asarray(P22, float); y = np.asarray(y, float)
    N = len(y)
    lo = LEGACY_EDGES[:-1][None, :]; hi = LEGACY_EDGES[1:][None, :]       # [1, 20]
    m = P22[:, 1:21]
    F_lo = P22[:, :1] + np.concatenate([np.zeros((N, 1)), np.cumsum(m, 1)[:, :-1]], 1)
    F_hi = F_lo + m
    yy = y[:, None]
    cut = np.clip(yy, lo, hi)                                              # split point inside the bin
    F_cut = F_lo + m * (cut - lo) / (hi - lo)

    def seg(a, b, l):
        return l * (a * a + a * b + b * b) / 3.0
    # [lo, cut): x < y, indicator 0.   [cut, hi): x >= y, indicator 1.
    total = seg(F_lo, F_cut, cut - lo) + seg(F_cut - 1.0, F_hi - 1.0, hi - cut)
    return total.sum(1)


def ece_marginal(P, cells):
    """ECE = sum_b | mean_i P_i(b) - observed frequency(b) |, the phase1_2 §7 definition."""
    P = np.asarray(P, float); K = P.shape[1]
    obs = np.bincount(cells, minlength=K) / max(1, len(cells))
    pred = P.mean(0)
    return float(np.abs(pred - obs).sum()), pred, obs


def ece_reliability(p, o, n_bins=10):
    """Conditional calibration of one probability: equal-frequency bins of p, weighted |mean p - rate|.

    The marginal ECE above is calibration-in-the-large per bin -- a forecaster that emits the
    training marginal on every row scores near zero on it while ranking nothing. This one asks
    whether rows given 0.97 fill completely 97% of the time. Reported beside it, not instead.
    """
    p = np.asarray(p, float); o = np.asarray(o, float)
    order = np.argsort(p, kind="stable")
    tot = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            tot += len(chunk) * abs(p[chunk].mean() - o[chunk].mean())
    return float(tot / max(1, len(p)))


def fill_scores(P, y, kind, n_boot=1000, seed=11):
    """kind 'cells22' (new head, LightGBM-22) or 'legacy20' (old head, LightGBM-20, naive)."""
    from heads import fill_cell, fill_to_legacy
    y = np.asarray(y, float)
    if kind == "cells22":
        P22 = np.asarray(P, float); P20 = fill_to_legacy(P22)
    else:
        P20 = np.asarray(P, float); P22 = legacy_to_cells(P20)
    n = len(y)
    lb = legacy_bin(y); cell = fill_cell(y)
    out = {}
    r_leg = crps_legacy_rows(P20, y); r_ex = crps_exact_rows(P22, y)
    out["crps_legacy"] = (float(r_leg.mean()),) + bootstrap_ci(lambda i: r_leg[i].mean(), n, n_boot, seed)
    out["crps_exact"] = (float(r_ex.mean()),) + bootstrap_ci(lambda i: r_ex[i].mean(), n, n_boot, seed)
    e20, pred20, obs20 = ece_marginal(P20, lb)
    out["ece20"] = (e20,) + bootstrap_ci(lambda i: ece_marginal(P20[i], lb[i])[0], n, n_boot, seed)
    out["_table20"] = [(b, float(pred20[b]), float(obs20[b])) for b in range(20)]
    # P(fill in [.95, 1]) reliability exists for every forecaster
    p_top = P20[:, 19]; o_top = (lb == 19).astype(float)
    out["rel_top20"] = (ece_reliability(p_top, o_top),) + bootstrap_ci(
        lambda i: ece_reliability(p_top[i], o_top[i]), n, n_boot, seed)
    if kind == "cells22":
        e22, pred22, obs22 = ece_marginal(P22, cell)
        out["ece22"] = (e22,) + bootstrap_ci(lambda i: ece_marginal(P22[i], cell[i])[0], n, n_boot, seed)
        out["_table22"] = [(b, float(pred22[b]), float(obs22[b])) for b in range(22)]
        p1 = P22[:, 21]; o1 = (y >= 1).astype(float)
        out["rel_one"] = (ece_reliability(p1, o1),) + bootstrap_ci(
            lambda i: ece_reliability(p1[i], o1[i]), n, n_boot, seed)
    # the check guide 5.2 asks for: predicted P(complete) against the observed complete rate
    out["p_complete_pred"] = (float(P22[:, 21].mean()) if kind == "cells22" else float("nan"),)
    out["p_complete_obs"] = (float((y >= 1).mean()),)
    out["n_test"] = (n,)
    return out


# ================================================================== arrival
def arrival_scores(ET, Y, EV, AUX, S=None, pT=None, n_boot=1000, seed=11, W=12):
    """ET: the ranking score (expected arrival week, in WEEKS -- same units as AUX)."""
    ET = np.asarray(ET, float); Y = np.asarray(Y, float); EV = np.asarray(EV, bool)
    n = len(Y)
    out = {"cindex": (cindex(ET, Y, EV),) + bootstrap_ci(
        lambda i: cindex(ET[i], Y[i], EV[i], n=20_000, seed=13), n, n_boot, seed)}
    m = EV & np.isfinite(AUX)
    yl = (Y[m] > AUX[m]).astype(int)
    pl = ET[m] - AUX[m]                   # rank LATENESS: prediction - promise, never the raw prediction
    out["roc_auc_late"] = (roc_auc(yl, pl),) + bootstrap_ci(lambda i: roc_auc(yl[i], pl[i]), len(yl), n_boot, seed)
    out["late_rate"] = (float(yl.mean()),)
    if S is not None:
        # distributional alternative: P(T > promise) read off the survival curve
        k = np.clip(np.floor(AUX[m]).astype(int), 0, W)
        Sfull = np.concatenate([np.ones((len(S), 1)), S], 1)[m]
        p_late = Sfull[np.arange(len(k)), k]
        out["roc_auc_late_ptail"] = (roc_auc(yl, p_late),) + bootstrap_ci(
            lambda i: roc_auc(yl[i], p_late[i]), len(yl), n_boot, seed)
        # calibration of the arrival-week distribution: 12 weeks + "not by week 12"
        cell = np.where(EV, np.clip(Y, 1, W).astype(int) - 1, W)
        P13 = np.concatenate([pT, S[:, -1:]], 1)
        e, pred, obs = ece_marginal(P13, cell)
        out["ece_week"] = (e,) + bootstrap_ci(lambda i: ece_marginal(P13[i], cell[i])[0], n, n_boot, seed)
        out["max_abs_err_w1_6"] = (float(np.abs(pred[:6] - obs[:6]).max()),)
        out["_table_week"] = [(w + 1, float(pred[w]), float(obs[w])) for w in range(W)] + \
                             [(">12", float(pred[W]), float(obs[W]))]
        pa = 1 - S[:, -1]; oa = EV.astype(float)
        out["rel_arrive_by_12"] = (ece_reliability(pa, oa),) + bootstrap_ci(
            lambda i: ece_reliability(pa[i], oa[i]), n, n_boot, seed)
        out["sum_check_max_dev"] = (float(np.abs(pT.sum(1) + S[:, -1] - 1).max()),)
        out["monotone_violations"] = (int((np.diff(S, axis=1) > 0).sum()),)
    out["n_test"] = (n,)
    return out


# ================================================================== capacity
QS = (0.1, 0.5, 0.9)


def pinball_rows(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def capacity_scores(Q, y, n_boot=1000, seed=11):
    """Q [N, 3] = P10, P50, P90 on the label's own scale."""
    from scipy.stats import spearmanr
    Q = np.asarray(Q, float); y = np.asarray(y, float); n = len(y)
    out = {}
    rows = {t: pinball_rows(y, Q[:, j], t) for j, t in enumerate(QS)}
    for t in QS:
        r = rows[t]
        out[f"pinball_{int(t*100)}"] = (float(r.mean()),) + bootstrap_ci(lambda i: r[i].mean(), n, n_boot, seed)
    mean_rows = (rows[0.1] + rows[0.5] + rows[0.9]) / 3
    out["pinball_mean"] = (float(mean_rows.mean()),) + bootstrap_ci(lambda i: mean_rows[i].mean(), n, n_boot, seed)
    cov = ((y >= Q[:, 0]) & (y <= Q[:, 2])).astype(float)
    out["coverage80"] = (float(cov.mean()),) + bootstrap_ci(lambda i: cov[i].mean(), n, n_boot, seed)
    # empirical exceedance (Phase 8): nominal 10% above P90 and 10% below P10; additive keys, no existing number changes
    above = (y > Q[:, 2]).astype(float); below = (y < Q[:, 0]).astype(float)
    out["exceed_p90"] = (float(above.mean()),) + bootstrap_ci(lambda i: above[i].mean(), n, n_boot, seed)
    out["below_p10"] = (float(below.mean()),) + bootstrap_ci(lambda i: below[i].mean(), n, n_boot, seed)
    cross = ((Q[:, 0] > Q[:, 1]) | (Q[:, 1] > Q[:, 2])).astype(float)
    out["crossing_rate"] = (float(cross.mean()),) + bootstrap_ci(lambda i: cross[i].mean(), n, n_boot, seed)
    out["mae_p50"] = (float(np.abs(y - Q[:, 1]).mean()),)
    out["spearman_p50"] = (float(spearmanr(Q[:, 1], y)[0]),) + bootstrap_ci(
        lambda i: spearmanr(Q[i, 1], y[i])[0], n, 200, seed)
    out["width80_mean"] = (float((Q[:, 2] - Q[:, 0]).mean()),)
    out["n_test"] = (n,)
    return out


def shortage_scores(P, Y, n_boot=1000, seed=11):
    n = len(Y)
    return {"pr_auc": (pr_auc(Y, P),) + bootstrap_ci(lambda i: pr_auc(Y[i], P[i]), n, n_boot, seed),
            "roc_auc": (roc_auc(Y, P),) + bootstrap_ci(lambda i: roc_auc(Y[i], P[i]), n, n_boot, seed),
            "base_rate": (float(np.mean(Y)),), "n_test": (n,)}
