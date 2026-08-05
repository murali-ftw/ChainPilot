"""
Step 4 — Validate the Depth Prior (Claim 2, Part 1)
(`docs/14_Model_Development_Roadmap.md` §7).

1. Layer-depth sweep: train/evaluate HGT at L=1,2,3,4 with a *single shared*
   readout depth (every task reads h^L) -- the thing the structural prior
   is being compared against.
2. Over-smoothing measurement: mean pairwise embedding cosine similarity
   per layer, on the L=4 sweep model.
3. Nested significance test: paired bootstrap Delta-AUC between adjacent L
   values, AND between the best shared-depth L and Step 3's fixed
   structural-prior baseline (retrained here under the same model_version
   so the comparison runs against a model built in this same process).

Run: python -m ml.run_step4
"""

from __future__ import annotations

import math

from ml.data.db import get_connection
from ml.evaluate import (
    best_f1_threshold,
    collect_predictions,
    evaluate_split,
    log_evaluation_runs,
    over_smoothing_profile,
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

ARCHITECTURE = "heterogeneous_graph_transformer"


def main() -> None:
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)

    models = {}
    for depth_l in [1, 2, 3, 4]:
        model_version = f"hgt-lsweep-L{depth_l}"
        result = run_training_job(
            conn, model_version, ARCHITECTURE, train_b, val_b,
            num_layers=depth_l, shared_depth=depth_l, hidden=64, epochs=100,
            purpose=f"Step 4 layer-depth sweep L={depth_l} (single shared readout)",
        )
        model = result["model"]
        models[depth_l] = model
        thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
        results = evaluate_split(model, test_b, thresholds)
        log_evaluation_runs(conn, model_version, ARCHITECTURE, results, "test", depth_l=depth_l,
                             train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
        print(f"\n=== L={depth_l} (shared depth, {model.parameter_count():,} params, "
              f"best_epoch={result['best_epoch']}) ===")
        for task, metrics in results.items():
            auc = metrics.get("roc_auc")
            if auc:
                print(f"  {task:10s} roc_auc={auc['value']:.4f}  CI95=[{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}]  "
                      f"n={auc['n']} pos={auc['positives']}")

    # Also retrain Step 3's fixed-structural-prior baseline in THIS process,
    # so the significance test below compares models built in the same run.
    baseline_result = run_training_job(
        conn, "hgt-baseline-v1", ARCHITECTURE, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=64, epochs=100,
        purpose="Step 3 baseline -- L=4, fixed structural depth prior (retrained for Step 4 comparison)",
    )
    baseline_model = baseline_result["model"]

    # --- Nested significance test: adjacent L values, shared-depth sweep ---
    print("\n=== Nested significance test: adjacent shared-depth L values ===")
    for task in TASKS:
        y_true, _ = collect_predictions(models[1], test_b)[task]
        if len(y_true) == 0:
            continue
        probs = {L: collect_predictions(models[L], test_b)[task][1] for L in [1, 2, 3, 4]}
        for lo, hi in [(1, 2), (2, 3), (3, 4)]:
            cmp = paired_delta_auc_ci(y_true, probs[hi], probs[lo])
            if cmp["delta"] is None:
                print(f"  {task:10s} L={hi} vs L={lo}: insufficient data for CI")
                continue
            sig = "SIGNIFICANT" if cmp["significant"] else "not significant"
            print(f"  {task:10s} L={hi} vs L={lo}: dAUC={cmp['delta']:+.4f}  "
                  f"CI95=[{cmp['ci_lower']:+.4f}, {cmp['ci_upper']:+.4f}]  ({sig})")

    # --- Structural prior (Step 3 baseline) vs each shared-depth L ---
    print("\n=== Structural prior (fixed h2/h3 split) vs single shared depth ===")
    for task in TASKS:
        y_true, prob_prior = collect_predictions(baseline_model, test_b)[task]
        if len(y_true) == 0:
            continue
        for L in [1, 2, 3, 4]:
            prob_shared = collect_predictions(models[L], test_b)[task][1]
            cmp = paired_delta_auc_ci(y_true, prob_prior, prob_shared)
            if cmp["delta"] is None:
                print(f"  {task:10s} prior vs L={L}: insufficient data for CI")
                continue
            sig = "SIGNIFICANT" if cmp["significant"] else "not significant"
            print(f"  {task:10s} prior vs L={L}: dAUC={cmp['delta']:+.4f}  "
                  f"CI95=[{cmp['ci_lower']:+.4f}, {cmp['ci_upper']:+.4f}]  ({sig})")

    # --- Over-smoothing profile on the L=4 sweep model ---
    print("\n=== Over-smoothing: mean pairwise cosine similarity per layer (L=4 model, first test snapshot) ===")
    profile = over_smoothing_profile(models[4], test_b[0])
    for layer_idx, by_type in profile.items():
        row = "  ".join(f"{nt}={sim:.3f}" for nt, sim in by_type.items() if not math.isnan(sim))
        print(f"  layer {layer_idx}: {row}")

    conn.close()


if __name__ == "__main__":
    main()
