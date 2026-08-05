"""
The single-feature leakage test (`docs/10_AI_ML_Documentation.md` §9.5,
`docs/13_Testing_Documentation.md` §6-7).

Train on one feature at a time, held out via cross-validation. Any solo
feature scoring AUC > 0.9 against its task's label is a leak. This runs
against the *assembled* feature tables (`ml/data/snapshots.py`), not the raw
CSVs -- `db/README.md`'s realism audit already covers the raw CSVs (max
solo-feature AUC ~= 0.59-0.63); this test's job is to catch a leak
introduced during feature *assembly* (e.g. an accidental join to a
post-t0 row), a different failure surface.

Run: pytest ml/tests/test_leakage.py -v -s
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ml.data.db import get_connection
from ml.data.snapshots import build_multi_snapshot_dataset

LEAKAGE_AUC_THRESHOLD = 0.90
# db/README.md's realism audit found max solo-feature AUC ~= 0.59-0.63 on the
# raw CSVs; this is a generous sanity band around that for the assembled
# tables, not a hard pass/fail gate -- only LEAKAGE_AUC_THRESHOLD is a gate.
EXPECTED_AUC_CEILING = 0.75

NON_FEATURE_COLUMNS = {"entity_id", "label", "snapshot_t0"}

# (entity_type, task) pairs actually present in training_labels at this
# dataset version (order/customer labels don't exist yet -- see
# ml/data/features.py's order_customer_features_asof docstring).
TASKS = [
    ("shipment", "delay"),
    ("product", "shortage"),
    ("supplier", "impact"),
]


def _held_out_auc(x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float | None:
    """
    Cross-validated AUC of a single feature column via a 1-feature logistic
    regression, held out per fold. Returns None when the column/label pair
    can't support a meaningful split (degenerate variance, too few positives).
    """
    mask = ~np.isnan(x)
    x, y = x[mask], y[mask]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return None

    min_class_count = np.bincount(y.astype(int)).min()
    n_splits = min(n_splits, min_class_count)
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros_like(x, dtype=float)
    for train_idx, test_idx in skf.split(x.reshape(-1, 1), y):
        clf = LogisticRegression()
        clf.fit(x[train_idx].reshape(-1, 1), y[train_idx])
        oof[test_idx] = clf.predict_proba(x[test_idx].reshape(-1, 1))[:, 1]
    return roc_auc_score(y, oof)


@pytest.fixture(scope="module")
def conn():
    connection = get_connection()
    yield connection
    connection.close()


@pytest.mark.parametrize("entity_type,task", TASKS)
def test_no_single_feature_leaks(conn, entity_type, task):
    df = build_multi_snapshot_dataset(conn, entity_type, task)
    y = df["label"].astype(int).to_numpy()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    assert len(df) > 0, f"no rows assembled for {entity_type}/{task}"

    results = {}
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        auc = _held_out_auc(x, y)
        if auc is None:
            continue
        results[col] = max(auc, 1 - auc)  # direction-agnostic separability

    leaks = {col: auc for col, auc in results.items() if auc > LEAKAGE_AUC_THRESHOLD}
    max_col = max(results, key=results.get) if results else None
    max_auc = results.get(max_col, float("nan"))

    print(
        f"\n[{entity_type}/{task}] n={len(df)} positives={y.sum()} "
        f"({100 * y.mean():.2f}%) strongest solo feature: "
        f"{max_col} AUC={max_auc:.4f}"
    )
    if max_auc > EXPECTED_AUC_CEILING:
        print(
            f"  NOTE: {max_col} AUC={max_auc:.4f} exceeds the "
            f"{EXPECTED_AUC_CEILING} sanity band (informational only -- "
            f"{LEAKAGE_AUC_THRESHOLD} is the actual leakage gate)"
        )

    assert not leaks, (
        f"leakage detected for {entity_type}/{task}: "
        f"{ {col: round(a, 4) for col, a in leaks.items()} }"
    )
