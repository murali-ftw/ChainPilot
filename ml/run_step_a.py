"""
Step A (post-Steps-0-5 follow-up) — test the corrected impact depth prior.

`reports/steps_0-5_findings.md`'s Step 4 result: the L=1..4 sweep found
impact's AUC peaks at L2 (+0.292 vs L1, significant) then drops at L3
(-0.292 vs L2, significant) -- a structural signal at L2, not the documented
h^3. impact's label is supplier-level (`db/generate_dataset.py`), a
shallower structural distance than the Order/Customer-level path
`project_HADES.md` §4.2 used to derive h^3 for it.

This trains a second baseline (`hgt-baseline-v2-impact-h2`) identical to
`hgt-baseline-v1` in every respect except `STRUCTURAL_DEPTH_PRIOR_V2`
(impact reads h^2 instead of h^3), and compares impact AUC against the
original v1 result via the same paired bootstrap Delta-AUC test Step 4 used.

The original prior (`STRUCTURAL_DEPTH_PRIOR`) is untouched -- this is a
controlled A/B, not a replacement.

Run: python -m ml.run_step_a
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
from ml.models.depth import STRUCTURAL_DEPTH_PRIOR_V2, TASKS
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    load_all_snapshot_bundles,
    run_training_job,
    split_bundles,
)

ARCHITECTURE = "heterogeneous_graph_transformer"
V1_MODEL_VERSION = "hgt-baseline-v1"
V2_MODEL_VERSION = "hgt-baseline-v2-impact-h2"


def main() -> None:
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)

    # Retrain v1 in this process too, so the paired comparison below runs
    # against a model built in the same run (same discipline as run_step4.py).
    v1_result = run_training_job(
        conn, V1_MODEL_VERSION, ARCHITECTURE, train_b, val_b,
        num_layers=4, shared_depth=None, depth_prior=None, hidden=64, epochs=100,
        purpose="Step 3 baseline -- L=4, fixed structural depth prior (retrained for Step A comparison)",
    )
    v1_model = v1_result["model"]

    v2_result = run_training_job(
        conn, V2_MODEL_VERSION, ARCHITECTURE, train_b, val_b,
        num_layers=4, shared_depth=None, depth_prior=STRUCTURAL_DEPTH_PRIOR_V2, hidden=64, epochs=100,
        purpose="Step A -- corrected impact depth prior (h^2 instead of h^3), controlled A/B vs hgt-baseline-v1",
    )
    v2_model = v2_result["model"]

    for label, model, result in [("v1 (impact=h3, as-documented)", v1_model, v1_result),
                                   ("v2 (impact=h2, corrected)", v2_model, v2_result)]:
        thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
        results = evaluate_split(model, test_b, thresholds)
        model_version = V1_MODEL_VERSION if "v1" in label else V2_MODEL_VERSION
        log_evaluation_runs(conn, model_version, ARCHITECTURE, results, "test",
                             train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
        print(f"\n=== {label}  (best_epoch={result['best_epoch']}) ===")
        for task, metrics in results.items():
            auc = metrics.get("roc_auc")
            if auc:
                print(f"  {task:10s} roc_auc={auc['value']:.4f}  CI95=[{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}]  "
                      f"n={auc['n']} pos={auc['positives']}")

    print("\n=== Step A verdict: does the corrected impact prior (h2) beat the original (h3)? ===")
    y_true, prob_v1 = collect_predictions(v1_model, test_b)["impact"]
    _, prob_v2 = collect_predictions(v2_model, test_b)["impact"]
    cmp = paired_delta_auc_ci(y_true, prob_v2, prob_v1)
    if cmp["delta"] is None:
        print("  insufficient data for a paired CI")
    else:
        sig = "SIGNIFICANT" if cmp["significant"] else "not significant (CIs overlap)"
        print(f"  impact: v2 - v1 dAUC={cmp['delta']:+.4f}  CI95=[{cmp['ci_lower']:+.4f}, {cmp['ci_upper']:+.4f}]  ({sig})")
        if cmp["significant"] and cmp["delta"] > 0:
            print("  -> the corrected prior beats the original by more than the CI.")
        elif cmp["significant"] and cmp["delta"] < 0:
            print("  -> the corrected prior is significantly WORSE than the original.")
        else:
            print("  -> not distinguishable from the original at this label volume.")

    conn.close()


if __name__ == "__main__":
    main()
