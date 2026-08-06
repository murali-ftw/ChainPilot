"""
Task 2 (post-v3 follow-up, `reports/step5_result_v3.md`) — sparse-relation
ablation for the shortage HGT-vs-GraphSAGE gap.

**Direct test of the thin-relation-overfitting hypothesis**: does merging
the 4 thinnest meta-relations (`ml/models/sparse_hgt_encoder.py`:
MANUFACTURED_AT, SUPPLIES, STOCKED_AT, USED_IN + their 4 reverses -- 8 of
20 -- onto one shared SAGEConv instance, instead of each keeping its own
HGTConv relation-parameter set) close, narrow, or leave unchanged HGT's
seed-consistent loss to GraphSAGE on shortage (`reports/step5_result_v3.md`
Part I: -0.012 fixed-d, -0.014 matched-d, both consistent across 5 seeds).

Only 5 runs, not 10: `hgt`'s own fixed-d and matched-d entries were
IDENTICAL in the original ablation (d=64 for both -- HGT is the anchor
architecture, its own d never changes between arms; only GraphSAGE/GAT's d
moves to match HGT's parameter count). `hgt_sparse` inherits that same
property, so one 5-seed run at d=64 serves as the comparison point against
BOTH `graphsage-fixed-d` (d=64) and `graphsage-matched-d` (d=66) --
training it twice would be redundant, not more rigorous.

Governance tables backed up first (SQL dump + in-DB copy tables) --
`reports/step5_result_v3.md`'s own operational-mistake lesson, not repeated
a third time. This script only INSERTs new `-sparserelation-v3-seed{n}`
rows; it never runs `--drop` or touches any existing row.

Run: python -u -m ml.run_task2_sparse_relation
"""

from __future__ import annotations

import statistics
import time

import psycopg2.extras

from ml.data.db import get_connection
from ml.evaluate import (
    best_f1_threshold,
    collect_predictions,
    evaluate_split,
    log_evaluation_runs,
)
from ml.models.depth import TASKS
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    load_all_snapshot_bundles,
    run_training_job,
    split_bundles,
)

SEEDS = [0, 1, 2, 3, 4]
MODEL_VERSION_TEMPLATE = "hgt-sparserelation-v3-seed{seed}"


def main() -> None:
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    sparse_auc = {}    # seed -> {task: auc}
    sparse_preds = {}  # seed -> {task: (y_true, y_prob)}

    overall_start = time.monotonic()
    for seed in SEEDS:
        model_version = MODEL_VERSION_TEMPLATE.format(seed=seed)
        t0 = time.monotonic()
        result = run_training_job(
            conn, model_version, "hgt_sparse", train_b, val_b,
            num_layers=4, shared_depth=None, hidden=64, epochs=100, seed=seed,
            purpose="Task 2 sparse-relation ablation -- 4 thinnest meta-relations merged "
                    "onto one shared SAGEConv, direct test of thin-relation overfitting "
                    "as an explanation for HGT losing to GraphSAGE on shortage",
        )
        model = result["model"]
        thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
        results = evaluate_split(model, test_b, thresholds)
        log_evaluation_runs(conn, model_version, "hgt_sparse", results, "test",
                             train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
        elapsed = time.monotonic() - t0
        aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
        sparse_auc[seed] = aucs
        sparse_preds[seed] = {task: collect_predictions(model, test_b)[task] for task in TASKS}
        print(f"  [{elapsed:6.1f}s] {model_version:28s} best_epoch={result['best_epoch']:3d}  "
              f"params={model.parameter_count():,}  " +
              "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== 5 runs complete in {total_elapsed/60:.1f} min ===\n", flush=True)

    print("=" * 78)
    print("AUC spread across 5 seeds, hgt_sparse")
    print("=" * 78)
    for task in TASKS:
        values = [sparse_auc[s][task] for s in SEEDS]
        print(f"  {task:10s} mean={statistics.fmean(values):.4f}  std={statistics.pstdev(values):.4f}  "
              f"min={min(values):.4f}  max={max(values):.4f}  values={[round(v,4) for v in values]}")

    # Pull the already-logged graphsage AUC point values back out of
    # model_evaluation_runs for the comparison below. Only the AUC point
    # value is persisted there, not the raw (y_true, y_prob) arrays, so this
    # is a same-seed-index AUC delta (sparse's seed s vs graphsage's seed s),
    # not a same-process paired bootstrap like Step 4/5 used -- a weaker but
    # still valid application of the same "sign-consistency across seeds"
    # standard; retraining graphsage in-process to get paired predictions
    # would be redundant given its results are already governance-logged.
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT model_version, task, metric_value
            FROM model_evaluation_runs
            WHERE model_version LIKE %s AND metric_name = 'roc_auc' AND dataset_split = 'test'
            """,
            ("%-v3-seed%",),
        )
        rows = cur.fetchall()

    def _auc_by_seed(prefix: str, task: str) -> dict[int, float]:
        out = {}
        for r in rows:
            if r["model_version"].startswith(prefix) and r["task"] == task:
                seed = int(r["model_version"].rsplit("seed", 1)[1])
                out[seed] = float(r["metric_value"])
        return out

    print("\n" + "=" * 78)
    print("Shortage: hgt_sparse vs graphsage, vs the original hgt gap -- per-seed AUC deltas")
    print("=" * 78)
    for axis_name, graphsage_prefix, original_hgt_gap in [
        ("fixed-d", "graphsage-fixed-d-v3-seed", -0.0123),
        ("matched-d", "graphsage-matched-d-v3-seed", -0.0137),
    ]:
        sage_auc = _auc_by_seed(graphsage_prefix, "shortage")
        deltas = [sparse_auc[s]["shortage"] - sage_auc[s] for s in SEEDS if s in sage_auc]
        signs = {d > 0 for d in deltas}
        consistency = "CONSISTENT" if len(signs) == 1 else "FLIPS (noise)"
        print(f"\n  -- {axis_name} axis (original hgt-vs-graphsage gap: {original_hgt_gap:+.4f}, consistent loss) --")
        print(f"    hgt_sparse - graphsage per-seed dAUC: {[round(d,4) for d in deltas]}")
        print(f"    mean={statistics.fmean(deltas):+.4f}  -> {consistency}")
        if consistency == "CONSISTENT" and statistics.fmean(deltas) > 0:
            verdict = "gap CLOSES (hgt_sparse now beats graphsage)"
        elif consistency == "CONSISTENT" and abs(statistics.fmean(deltas)) < abs(original_hgt_gap):
            verdict = "gap NARROWS but persists"
        elif consistency == "CONSISTENT":
            verdict = "gap PERSISTS (or widens) -- not a relation-sparsity artifact"
        else:
            verdict = "inconsistent across seeds -- no clear verdict either way"
        print(f"    verdict: {verdict}")

    conn.close()


if __name__ == "__main__":
    main()
