"""
Metrics, bootstrap confidence intervals, per-claim validation routes —
Steps 3-8 (`docs/10_AI_ML_Documentation.md` §9.3-9.4, `project_HADES.md`
§9.1-9.4).

**CI method note.** `docs/05_Database_Design.md` §6.21 specifies a paired,
time-blocked bootstrap over whole snapshots as the correct default, because
repeated entities across snapshots are autocorrelated and i.i.d. resampling
understates the interval. This dataset's test split is exactly 2 snapshots
(Nov, Dec) -- block-resampling 2 blocks is degenerate (only 3 distinct
resamples exist: {Nov,Nov}, {Nov,Dec}, {Dec,Dec}), so this module uses
row-level i.i.d. bootstrap instead and records `ci_method='row_bootstrap'`
rather than silently mislabelling it `paired_block_bootstrap`. Per §9.3's
own caveat, this likely *understates* true uncertainty -- flagged here
rather than hidden, exactly the failure mode the schema note warns about.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from ml.models.depth import TASKS

N_BOOTSTRAP = 2000
RNG_SEED = 0


@torch.no_grad()
def collect_predictions(model, bundles) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """task -> (y_true, y_prob) concatenated across all bundles in this split."""
    model.eval()
    out = {task: ([], []) for task in TASKS}
    for bundle in bundles:
        logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
        for task in TASKS:
            idx, y = bundle.labels[task]
            if idx.numel() == 0:
                continue
            probs = torch.sigmoid(logits[task][idx]).numpy()
            out[task][0].append(y.numpy())
            out[task][1].append(probs)
    return {
        task: (np.concatenate(ys) if ys else np.array([]), np.concatenate(ps) if ps else np.array([]))
        for task, (ys, ps) in out.items()
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """The 'alert operating threshold' (`project_HADES.md` §9.1): the
    probability cutoff that maximizes F1 on this split. Called on
    VALIDATION predictions only and then frozen for test evaluation --
    never tuned against the test set itself."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    candidates = np.unique(y_prob)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: mean |predicted - observed| across
    equal-width probability bins, weighted by bin occupancy."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(bins):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def _metric_value(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
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


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_name: str, threshold: float,
                  n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float] | None:
    """Row-level i.i.d. percentile bootstrap, 95% CI. Returns None if the
    metric can't be computed at all (e.g. only one class present)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or y_true.sum() == 0 or y_true.sum() == n:
        return None
    values = []
    for _ in range(n_boot):
        sample = rng.integers(0, n, size=n)
        yt, yp = y_true[sample], y_prob[sample]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            values.append(_metric_value(metric_name, yt, yp, threshold))
        except ValueError:
            continue
    if len(values) < n_boot // 2:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate_split(model, bundles, thresholds: dict[str, float]) -> dict[str, dict[str, dict]]:
    """task -> metric_name -> {value, ci_lower, ci_upper, ci_method, n, positives}"""
    preds = collect_predictions(model, bundles)
    results = {}
    for task in TASKS:
        y_true, y_prob = preds[task]
        results[task] = {}
        if len(y_true) == 0:
            continue
        threshold = thresholds.get(task, 0.5)
        for metric_name in METRIC_NAMES:
            try:
                value = _metric_value(metric_name, y_true, y_prob, threshold)
            except ValueError:
                continue
            ci = bootstrap_ci(y_true, y_prob, metric_name, threshold)
            results[task][metric_name] = {
                "value": value,
                "ci_lower": ci[0] if ci else None,
                "ci_upper": ci[1] if ci else None,
                "ci_method": "row_bootstrap" if ci else None,
                "n": len(y_true),
                "positives": int(y_true.sum()),
            }
    return results


def log_evaluation_runs(conn, model_version: str, architecture: str, results: dict,
                         dataset_split: str, depth_l: int | None = None,
                         test_start_t0=None, test_end_t0=None, train_end_t0=None) -> None:
    rows = []
    for task, metrics in results.items():
        for metric_name, m in metrics.items():
            rows.append((
                model_version, architecture, metric_name, m["value"], dataset_split, task,
                depth_l, m["ci_lower"], m["ci_upper"], m["ci_method"],
                train_end_t0, test_start_t0, test_end_t0,
            ))
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO model_evaluation_runs
                (model_version, architecture, metric_name, metric_value, dataset_split, task,
                 depth_l, ci_lower, ci_upper, ci_method, train_end_t0, test_start_t0, test_end_t0)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Over-smoothing measurement (Step 4) -- replaces the theoretical reach table
# with an empirical depth ceiling (`project_HADES.md` §9.4).
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_pairwise_cosine_similarity(embeddings: torch.Tensor, max_nodes: int = 300,
                                     seed: int = 0) -> float:
    """Mean pairwise cosine similarity among node embeddings of one type at
    one layer. Subsamples to `max_nodes` for the O(n^2) pairwise computation
    when a node type is large (e.g. 1000+ Shipments)."""
    n = embeddings.size(0)
    if n < 2:
        return float("nan")
    if n > max_nodes:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_nodes]
        embeddings = embeddings[idx]
        n = max_nodes
    normed = torch.nn.functional.normalize(embeddings, dim=-1)
    sim = normed @ normed.T
    off_diag = sim - torch.eye(n) * sim.diag()
    return float(off_diag.sum() / (n * (n - 1)))


@torch.no_grad()
def over_smoothing_profile(model, bundle) -> dict[int, dict[str, float]]:
    """{layer_index (1-based): {node_type: mean pairwise cosine similarity}}"""
    model.eval()
    layers = model.encoder(bundle.data.x_dict, bundle.data.edge_index_dict)
    return {
        i + 1: {nt: mean_pairwise_cosine_similarity(h) for nt, h in layer.items()}
        for i, layer in enumerate(layers)
    }


# ---------------------------------------------------------------------------
# Nested significance test -- paired bootstrap on Delta-AUC between two
# models evaluated on the IDENTICAL held-out split (Step 4/5).
# ---------------------------------------------------------------------------

def paired_delta_auc_ci(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                         n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> dict:
    """95% CI on AUC(a) - AUC(b), resampling the SAME indices for both
    models each draw (paired) so shared noise cancels. CI excluding 0 means
    the gap clears the noise floor at this label volume."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        sample = rng.integers(0, n, size=n)
        yt = y_true[sample]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        auc_a = roc_auc_score(yt, prob_a[sample])
        auc_b = roc_auc_score(yt, prob_b[sample])
        deltas.append(auc_a - auc_b)
    if len(deltas) < n_boot // 2:
        return {"delta": None, "ci_lower": None, "ci_upper": None, "significant": None}
    point = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {"delta": point, "ci_lower": lo, "ci_upper": hi, "significant": bool(lo > 0 or hi < 0)}
