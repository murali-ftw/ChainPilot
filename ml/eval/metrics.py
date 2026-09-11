"""Run 9 — scoring with uncertainty.

A bare point estimate is not quotable, so every metric here comes with a 1,000-resample
bootstrap 95% interval, and each task carries a second metric alongside its headline one:

  arrival   C-index (headline)  +  ROC-AUC on the binarised late/on-time outcome
  shortage  PR-AUC  (headline)  +  ROC-AUC, because ROC-AUC is what gets quoted
  fill      CRPS    (headline)  +  calibration of the binned CDF

The bootstrap resamples ROWS of the test fold, which is the right unit: it answers "how much
would this score move on a different draw of test POs from the same world", not "how much would
it move on a different world". The second question is not answerable from within-world data.
"""
from __future__ import annotations
import numpy as np

NBIN = 20
EDGES = np.linspace(0, 1, NBIN + 1)


# ---------------------------------------------------------------- point metrics
def crps(cdf, y):
    step = float(np.diff(EDGES)[0])
    ind = (y[:, None] <= EDGES[None, 1:]).astype(float)
    return float((((cdf - ind) ** 2) * step).sum(1).mean())


def cindex(pred, t, e, n=1_000_000, seed=7):
    """Concordance over randomly sampled comparable pairs. Right-censoring aware."""
    rng = np.random.default_rng(seed)
    N = len(t)
    if N < 2:
        return float("nan")
    i = rng.integers(0, N, n); j = rng.integers(0, N, n)
    ok = ((t[i] < t[j]) & e[i]) | ((t[j] < t[i]) & e[j])
    if ok.sum() == 0:
        return float("nan")
    i, j = i[ok], j[ok]
    ear = t[i] < t[j]
    conc = np.where(ear, pred[i] < pred[j], pred[j] < pred[i])
    ties = pred[i] == pred[j]
    return float((conc.sum() + 0.5 * ties.sum()) / len(i))


def roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def cdf_calibration(P, y):
    """Bin reliability of the binned distributional forecast -- expected calibration error.

    Quantile-COVERAGE calibration is degenerate on this data: 91.4% of fill labels are
    exactly 1.0, so every nominal level from 0.1 to 0.9 maps to the same predicted
    threshold and the coverage error pins at 0.5 for any forecaster. Bin reliability does
    not have that failure mode. For each of the 20 bins, compare the MEAN PREDICTED
    probability of that bin against the OBSERVED frequency of outcomes in it:

        ECE = sum_b | mean_i P_i(b) - freq(b) |

    0 is perfect. A forecaster that puts all its mass on the modal bin scores roughly
    2 x (1 - modal share). Returns (ECE, per-bin table).
    """
    yb = np.clip(np.digitize(np.asarray(y, float), EDGES[1:-1]), 0, NBIN - 1)
    obs = np.bincount(yb, minlength=NBIN).astype(float) / max(1, len(yb))
    pred = np.asarray(P, float).mean(0)
    rows = [(int(b), float(pred[b]), float(obs[b]), float(pred[b] - obs[b]))
            for b in range(NBIN)]
    return float(np.abs(pred - obs).sum()), rows


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(fn, n_rows, n_boot=1000, seed=11, alpha=0.05):
    """fn(idx) -> scalar. Resamples row indices with replacement. Percentile interval."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, n_rows)
        vals[b] = fn(idx)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def score_with_ci(task, P, Y, EV, aux=None, n_boot=1000, seed=11):
    """-> {metric: (point, lo, hi)} for the task's headline and secondary metrics."""
    out = {}
    n = len(Y)
    if task == "arrival_week":
        out["cindex"] = (cindex(P, Y, EV),) + bootstrap_ci(
            # fewer pairs per resample: 1000 x 1e6 pairs would take hours and the pair
            # sampler's own noise is far below the row-resampling noise we are measuring
            lambda idx: cindex(P[idx], Y[idx], EV[idx], n=20_000, seed=13), n, n_boot, seed)
        if aux is not None:
            m = EV & np.isfinite(aux)                 # observed arrivals only
            yl = (Y[m] > aux[m]).astype(int)
            # Score LATENESS, not arrival week. A channel with a long lead time has both a
            # late predicted arrival and a late promise; only the DIFFERENCE carries the
            # late/on-time signal. Ranking on the raw prediction scores below 0.5.
            pl = P[m] - aux[m]
            if len(np.unique(yl)) == 2:
                out["roc_auc_late"] = (roc_auc(yl, pl),) + bootstrap_ci(
                    lambda idx: roc_auc(yl[idx], pl[idx]), len(yl), n_boot, seed)
                out["late_rate"] = (float(yl.mean()), float("nan"), float("nan"))
                out["n_binarised"] = (int(len(yl)), float("nan"), float("nan"))
    elif task == "shortage_qty":
        out["pr_auc"] = (pr_auc(Y, P),) + bootstrap_ci(
            lambda idx: pr_auc(Y[idx], P[idx]), n, n_boot, seed)
        out["roc_auc"] = (roc_auc(Y, P),) + bootstrap_ci(
            lambda idx: roc_auc(Y[idx], P[idx]), n, n_boot, seed)
        out["base_rate"] = (float(np.mean(Y)), float("nan"), float("nan"))
    else:                                              # fill_rate
        cdf = np.cumsum(P, 1)
        out["crps"] = (crps(cdf, Y),) + bootstrap_ci(
            lambda idx: crps(cdf[idx], Y[idx]), n, n_boot, seed)
        ece, rows = cdf_calibration(P, Y)
        out["calib_ece"] = (ece,) + bootstrap_ci(
            lambda idx: cdf_calibration(P[idx], Y[idx])[0], n, n_boot, seed)
        out["_calib_table"] = rows
    out["n_test"] = (int(n), float("nan"), float("nan"))
    return out
