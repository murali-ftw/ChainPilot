"""
Multi-snapshot dataset builder — one feature/label matrix per t0, concatenated
across the full snapshot schedule (`docs/14_Model_Development_Roadmap.md` §4).

This is the assembly layer the single-feature leakage test
(`ml/tests/test_leakage.py`) runs against -- not the raw CSVs, which
`db/README.md`'s audit already covers. The failure mode this step guards
against is different: an accidental join to a post-t0 row introduced during
*assembly*, even when every source table is leakage-clean on its own.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ml.data.features import ENTITY_FEATURE_BUILDERS


def get_snapshot_schedule(conn) -> pd.DataFrame:
    """All t0's in `graph_snapshots`, ordered chronologically."""
    return pd.read_sql(
        "SELECT id AS snapshot_id, t0, horizon_days FROM graph_snapshots ORDER BY t0",
        conn,
    )


def get_labelled_tasks(conn) -> pd.DataFrame:
    """Every (entity_type, task) combination actually present in training_labels."""
    return pd.read_sql(
        "SELECT DISTINCT entity_type, task FROM training_labels ORDER BY 1, 2",
        conn,
    )


def _labels_for_snapshot(conn, snapshot_id, entity_type: str, task: str) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT entity_id, label, event_at
        FROM training_labels
        WHERE snapshot_id = %(sid)s AND entity_type = %(etype)s AND task = %(task)s
        """,
        conn,
        params={"sid": str(snapshot_id), "etype": entity_type, "task": task},
    )


def _assert_label_window(labels: pd.DataFrame, t0: dt.datetime, horizon_days: int) -> None:
    """
    Re-verify the leakage contract's label side at assembly time
    (`docs/10_AI_ML_Documentation.md` §6.1): every positive's `event_at`
    must lie strictly inside `(t0, t0 + horizon_days]`. `db/load_data.py`
    already checked this against the loaded tables; re-checking here catches
    a bug introduced in the assembly join itself, which is Step 1's actual
    job (`docs/14_Model_Development_Roadmap.md` §4).
    """
    positives = labels[labels["label"]]
    if positives.empty:
        return
    t0_ts = pd.Timestamp(t0)
    horizon_end = t0_ts + pd.Timedelta(days=horizon_days)
    event_at = pd.to_datetime(positives["event_at"], utc=True)
    violations = positives[(event_at <= t0_ts) | (event_at > horizon_end)]
    if not violations.empty:
        raise AssertionError(
            f"label window violation at t0={t0}: {len(violations)} positive label(s) "
            f"with event_at outside (t0, t0+{horizon_days}d]"
        )


def build_labelled_snapshot(conn, snapshot_id, t0: dt.datetime, horizon_days: int,
                             entity_type: str, task: str) -> pd.DataFrame:
    """One snapshot's feature+label matrix for a given (entity_type, task)."""
    builder = ENTITY_FEATURE_BUILDERS[entity_type]
    features = builder(conn, t0)
    labels = _labels_for_snapshot(conn, snapshot_id, entity_type, task)
    _assert_label_window(labels, t0, horizon_days)

    merged = features.merge(labels[["entity_id", "label"]], on="entity_id", how="inner")
    merged["snapshot_t0"] = t0
    return merged


def build_multi_snapshot_dataset(conn, entity_type: str, task: str) -> pd.DataFrame:
    """
    The full leakage-contract-checked feature/label matrix for one
    (entity_type, task) pair, across every t0 in the snapshot schedule.
    """
    schedule = get_snapshot_schedule(conn)
    frames = [
        build_labelled_snapshot(conn, row.snapshot_id, row.t0, row.horizon_days, entity_type, task)
        for row in schedule.itertuples()
    ]
    return pd.concat(frames, ignore_index=True)
