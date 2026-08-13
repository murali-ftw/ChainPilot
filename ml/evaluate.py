"""
Metrics, confidence intervals and the paired comparison test.

Ported from `HADES_v1/ml/evaluate.py`, with one method upgrade that V2's scale
makes possible. V1 explained that `docs/05_Database_Design.md` §6.21 asks for a
paired, time-blocked bootstrap over whole snapshots -- repeated entities across
snapshots are autocorrelated, so an i.i.d. row bootstrap understates the interval
-- but V1's test split was only 2 snapshots, and block-resampling 2 blocks is
degenerate, so it fell back to row-level bootstrap and labelled it honestly.

**V2's test split is 16 snapshots**, so the block bootstrap is no longer
degenerate and is the default here (`ci_method='paired_block_bootstrap'`).
`row_bootstrap` remains available and is selected automatically when a split has
fewer than 4 blocks, still labelled for what it is.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ml.models.depth import TASKS

N_BOOTSTRAP = 2000
RNG_SEED = 0
MIN_BLOCKS = 4


@torch.no_grad()
def collect_predictions(model, bundles) -> dict[str, dict]:
    """task -> {y, p, block}: labels, predicted probabilities, and the snapshot
    index each row came from (the resampling block)."""
    model.eval()
    out = {task: {"y": [], "p": [], "block": []} for task in TASKS}
    for b_i, bundle in enumerate(bundles):
        logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
        for task in TASKS:
            idx, y = bundle.labels[task]
            if idx.numel() == 0:
                continue
            probs = torch.sigmoid(logits[task][idx]).float().cpu().numpy()
            out[task]["y"].append(y.cpu().numpy())
            out[task]["p"].append(probs)
            out[task]["block"].append(np.full(len(probs), b_i, dtype=np.int64))
    return {
        task: {k: (np.concatenate(v) if v else np.array([])) for k, v in d.items()}
        for task, d in out.items()
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """The alert operating threshold: the cutoff maximising F1 on VALIDATION
    predictions, then frozen for test -- never tuned against test itself.
    Evaluated on a percentile grid rather than every distinct probability, which
    at V2's label volume (up to ~500k rows) is the difference between a second
    and an hour, and cannot move the chosen threshold materially."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 512)))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(bins):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def _metric_value(metric_name: str, y_true, y_prob, threshold: float) -> float:
    if metric_name == "roc_auc":
        return roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    if metric_name == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric_name == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if metric_name == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric_name == "calibration_error":
        return calibration_error(y_true, y_prob)
    raise ValueError(metric_name)


METRIC_NAMES = ("roc_auc", "precision", "recall", "f1", "calibration_error")


def _block_index(block: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(block == b) for b in np.unique(block)]


def _resample(rng, blocks: list[np.ndarray] | None, n: int) -> np.ndarray:
    if blocks is None:
        return rng.integers(0, n, size=n)
    pick = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in pick])


def bootstrap_ci(y_true, y_prob, metric_name: str, threshold: float, block=None,
                 n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED):
    """95% percentile CI. Block-resamples whole snapshots when the split has at
    least `MIN_BLOCKS` of them, otherwise falls back to row-level i.i.d.
    Returns `(lo, hi, method)` or None when the metric is undefined."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or y_true.sum() == 0 or y_true.sum() == n:
        return None
    blocks = None
    method = "row_bootstrap"
    if block is not None and len(np.unique(block)) >= MIN_BLOCKS:
        blocks, method = _block_index(block), "paired_block_bootstrap"
    values = []
    for _ in range(n_boot):
        sample = _resample(rng, blocks, n)
        yt, yp = y_true[sample], y_prob[sample]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            values.append(_metric_value(metric_name, yt, yp, threshold))
        except ValueError:
            continue
    if len(values) < n_boot // 2:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), method


def evaluate_split(model, bundles, thresholds: dict[str, float],
                   metrics=METRIC_NAMES) -> dict:
    """task -> metric -> {value, ci_lower, ci_upper, ci_method, n, positives}"""
    preds = collect_predictions(model, bundles)
    results = {}
    for task in TASKS:
        y, p, blk = preds[task]["y"], preds[task]["p"], preds[task]["block"]
        results[task] = {}
        if len(y) == 0:
            continue
        threshold = thresholds.get(task, 0.5)
        for metric_name in metrics:
            try:
                value = _metric_value(metric_name, y, p, threshold)
            except ValueError:
                continue
            ci = bootstrap_ci(y, p, metric_name, threshold, block=blk)
            results[task][metric_name] = {
                "value": value,
                "ci_lower": ci[0] if ci else None,
                "ci_upper": ci[1] if ci else None,
                "ci_method": ci[2] if ci else None,
                "n": int(len(y)), "positives": int(y.sum()),
            }
    return results


def paired_delta_auc_ci(y_true, prob_a, prob_b, block=None, n_boot: int = N_BOOTSTRAP,
                        seed: int = RNG_SEED) -> dict:
    """95% CI on AUC(a) - AUC(b), resampling the SAME indices for both models on
    every draw so shared noise cancels. A CI excluding 0 means the gap clears the
    noise floor at this label volume."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or y_true.sum() in (0, n):
        return {"delta": None, "ci_lower": None, "ci_upper": None,
                "significant": None, "ci_method": None}
    blocks, method = None, "row_bootstrap"
    if block is not None and len(np.unique(block)) >= MIN_BLOCKS:
        blocks, method = _block_index(block), "paired_block_bootstrap"
    deltas = []
    for _ in range(n_boot):
        sample = _resample(rng, blocks, n)
        yt = y_true[sample]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        deltas.append(roc_auc_score(yt, prob_a[sample]) - roc_auc_score(yt, prob_b[sample]))
    if len(deltas) < n_boot // 2:
        return {"delta": None, "ci_lower": None, "ci_upper": None,
                "significant": None, "ci_method": method}
    point = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {"delta": point, "ci_lower": lo, "ci_upper": hi,
            "significant": bool(lo > 0 or hi < 0), "ci_method": method}


def sign_consistency(deltas: list[float]) -> dict:
    """Across-seed agreement on the DIRECTION of an effect. Every statistical
    check in this project reports this rather than a mean alone, because a mean
    that averages +0.02 and -0.02 is not evidence of anything."""
    vals = [d for d in deltas if d is not None]
    if not vals:
        return {"n": 0, "positive": 0, "negative": 0, "agreement": None, "consistent": None}
    pos = sum(1 for d in vals if d > 0)
    neg = sum(1 for d in vals if d < 0)
    agreement = max(pos, neg) / len(vals)
    return {"n": len(vals), "positive": pos, "negative": neg,
            "agreement": agreement, "consistent": bool(agreement == 1.0)}
