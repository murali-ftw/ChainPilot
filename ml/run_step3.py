"""
Step 3 — HGT Baseline (`docs/14_Model_Development_Roadmap.md` §6).

Trains the L=4, fixed-structural-depth-prior HGT baseline, writes one
`model_registry` row, and evaluates it on train/validation/test with
bootstrap CIs written to `model_evaluation_runs`.

Run: python -m ml.run_step3
"""

from __future__ import annotations

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

MODEL_VERSION = "hgt-baseline-v1"


def main() -> None:
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[b.t0.date() for b in train_b]}")
    print(f"val   t0s: {[b.t0.date() for b in val_b]}")
    print(f"test  t0s: {[b.t0.date() for b in test_b]}")

    result = run_training_job(
        conn, MODEL_VERSION, "heterogeneous_graph_transformer", train_b, val_b,
        num_layers=4, shared_depth=None, hidden=64, epochs=100,
        purpose="Step 3 baseline -- L=4, fixed structural depth prior",
    )
    model = result["model"]
    print(f"parameter_count={model.parameter_count():,}  best_epoch={result['best_epoch']}  "
          f"best_val_auc={result['best_val_auc']}")

    thresholds = {
        task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS
    }
    print("thresholds (from validation):", thresholds)

    for split_name, split_bundles_ in [("train", train_b), ("validation", val_b), ("test", test_b)]:
        results = evaluate_split(model, split_bundles_, thresholds)
        log_evaluation_runs(
            conn, MODEL_VERSION, "heterogeneous_graph_transformer", results, split_name,
            train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START if split_name == "test" else None,
        )
        print(f"\n=== {split_name} ===")
        for task, metrics in results.items():
            for metric_name, m in metrics.items():
                ci = f"[{m['ci_lower']:.3f}, {m['ci_upper']:.3f}]" if m["ci_lower"] is not None else "n/a"
                print(f"  {task:10s} {metric_name:18s} {m['value']:.4f}  CI95={ci}  n={m['n']} pos={m['positives']}")

    conn.close()


if __name__ == "__main__":
    main()
