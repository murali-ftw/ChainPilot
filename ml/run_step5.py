"""
Step 5 — Architecture Ablation (Claim 1)
(`docs/14_Model_Development_Roadmap.md` §8).

Six runs: {GraphSAGE, GAT, HGT} x {fixed-d=64, matched-parameter}. Matched-d
values (GraphSAGE=66, GAT=92) were found by a small search against this
dataset's REAL recomputed feature dimensions (Step 2), landing near HGT's
701K-parameter encoder -- not `project_HADES.md` §8.3's placeholder values
(86/116), which were derived against the doc's placeholder node-feature
dims, not the real ones (`ml/graph/builder.py` smoke test: Supplier=14 not
21, Component=6 not 10, etc.). HGT's own d=64 is identical between the
fixed and matched arms by construction (it's the anchor).

All six share the identical graph, time split, loss, and regularization as
Step 3's baseline; only the encoder differs.

Run: python -m ml.run_step5
"""

from __future__ import annotations

from ml.data.db import get_connection
from ml.evaluate import (
    best_f1_threshold,
    collect_predictions,
    evaluate_split,
    log_evaluation_runs,
    paired_delta_auc_ci,
)
from ml.models.depth import TASKS
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    load_all_snapshot_bundles,
    run_training_job,
    split_bundles,
)

# (model_version, architecture db-label, internal key, hidden dim, arm)
# model_version names match Step 5.3's exact spec (no d-suffix); the actual
# per-architecture d is recorded in model_registry.hyperparameters instead.
RUNS = [
    ("hgt-fixed-d", "heterogeneous_graph_transformer", "hgt", 64, "fixed"),
    ("graphsage-fixed-d", "graphsage", "graphsage", 64, "fixed"),
    ("gat-fixed-d", "gat", "gat", 64, "fixed"),
    ("hgt-matched-d", "heterogeneous_graph_transformer", "hgt", 64, "matched"),
    ("graphsage-matched-d", "graphsage", "graphsage", 66, "matched"),
    ("gat-matched-d", "gat", "gat", 92, "matched"),
]


def main() -> None:
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)

    models = {}
    for model_version, arch_label, arch_key, hidden, arm in RUNS:
        result = run_training_job(
            conn, model_version, arch_label, train_b, val_b,
            num_layers=4, shared_depth=None, hidden=hidden, epochs=100,
            purpose=f"Step 5 architecture ablation -- {arm}-parameter arm, d={hidden}",
        )
        model = result["model"]
        models[model_version] = model
        thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
        results = evaluate_split(model, test_b, thresholds)
        log_evaluation_runs(conn, model_version, arch_label, results, "test",
                             train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
        print(f"\n=== {model_version} ({arch_key}, d={hidden}, {arm} arm, "
              f"{model.parameter_count():,} params, best_epoch={result['best_epoch']}) ===")
        for task, metrics in results.items():
            auc = metrics.get("roc_auc")
            if auc:
                print(f"  {task:10s} roc_auc={auc['value']:.4f}  CI95=[{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}]  "
                      f"n={auc['n']} pos={auc['positives']}")

    def report_axis(axis_name: str, versions: list[str]) -> None:
        print(f"\n=== Claim 1 -- {axis_name} axis ===")
        for task in TASKS:
            y_true, _ = collect_predictions(models[versions[0]], test_b)[task]
            if len(y_true) == 0:
                continue
            probs = {v: collect_predictions(models[v], test_b)[task][1] for v in versions}
            hgt_v = versions[0]
            for other in versions[1:]:
                cmp = paired_delta_auc_ci(y_true, probs[hgt_v], probs[other])
                if cmp["delta"] is None:
                    print(f"  {task:10s} {hgt_v} vs {other}: insufficient data for CI")
                    continue
                sig = "SIGNIFICANT" if cmp["significant"] else "not significant (CIs overlap)"
                print(f"  {task:10s} {hgt_v} vs {other}: dAUC={cmp['delta']:+.4f}  "
                      f"CI95=[{cmp['ci_lower']:+.4f}, {cmp['ci_upper']:+.4f}]  ({sig})")

    report_axis("fixed-d=64", ["hgt-fixed-d", "graphsage-fixed-d", "gat-fixed-d"])
    report_axis("matched-parameter", ["hgt-matched-d", "graphsage-matched-d", "gat-matched-d"])

    conn.close()


if __name__ == "__main__":
    main()
