"""
Step B (post-Steps-0-5 follow-up) — multi-seed variance check.

Step 4's L=1..4 sweep was run on a single training seed. This retrains the
5 configurations that sweep covers (baseline L=4 fixed-prior + shared-depth
L=1,2,3,4) across 5 seeds (model init/dropout only -- dataset, split, and
feature assembly are already fixed/deterministic upstream, so this isolates
model-fit variance cleanly). Each of the 25 runs gets its own
`model_registry` row and `model_evaluation_runs` rows with CIs, exactly like
every other run in this pipeline.

Then: for the three findings reports/steps_0-5_findings.md flagged as
possibly single-seed noise --

    impact  L2 vs L1  (original: +0.292, significant)
    impact  L3 vs L2  (original: -0.292, significant)
    shortage L4 vs L3  (original: +0.039, significant)

-- this reports the per-seed AUC spread (min/max/std) per (task, config)
cell, and the per-seed paired difference for each of the three findings
above, so "does the sign hold across all 5 seeds" can be answered directly
rather than trusting one bootstrap CI computed from one trained model.

Run: python -m ml.run_step_b
"""

from __future__ import annotations

import statistics

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

ARCHITECTURE = "heterogeneous_graph_transformer"
SEEDS = [0, 1, 2, 3, 4]

# (config_name, num_layers, shared_depth)
CONFIGS = [
    ("baseline", 4, None),  # fixed structural depth prior, as Step 3
    ("L1", 1, 1),
    ("L2", 2, 2),
    ("L3", 3, 3),
    ("L4", 4, 4),
]


def main() -> None:
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)

    # auc_by[(config_name, task)] = [auc_seed0, auc_seed1, ...]
    auc_by: dict[tuple[str, str], list[float]] = {(c, t): [] for c, _, _ in CONFIGS for t in TASKS}
    # per_seed_auc[config_name][seed][task] = auc, for the paired per-seed diffs below
    per_seed_auc: dict[str, dict[int, dict[str, float]]] = {c: {} for c, _, _ in CONFIGS}

    for seed in SEEDS:
        for config_name, num_layers, shared_depth in CONFIGS:
            model_version = f"hgt-seedvar-{config_name}-s{seed}"
            purpose = (
                f"Step B multi-seed variance check -- {config_name} "
                f"({'fixed structural prior' if shared_depth is None else f'shared depth L={shared_depth}'}), seed={seed}"
            )
            result = run_training_job(
                conn, model_version, ARCHITECTURE, train_b, val_b,
                num_layers=num_layers, shared_depth=shared_depth, hidden=64, epochs=100,
                seed=seed, purpose=purpose,
            )
            model = result["model"]
            thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
            results = evaluate_split(model, test_b, thresholds)
            depth_l = shared_depth if shared_depth is not None else 4
            log_evaluation_runs(conn, model_version, ARCHITECTURE, results, "test", depth_l=depth_l,
                                 train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)

            seed_aucs = {}
            for task in TASKS:
                auc = results.get(task, {}).get("roc_auc")
                value = auc["value"] if auc else float("nan")
                auc_by[(config_name, task)].append(value)
                seed_aucs[task] = value
            per_seed_auc[config_name][seed] = seed_aucs
            print(f"seed={seed} {config_name:10s} " +
                  "  ".join(f"{t}={seed_aucs[t]:.4f}" for t in TASKS) +
                  f"  (best_epoch={result['best_epoch']})")

    print(f"\n{'=' * 78}\nAUC spread across {len(SEEDS)} seeds, per (task, config)\n{'=' * 78}")
    for config_name, _, _ in CONFIGS:
        for task in TASKS:
            values = auc_by[(config_name, task)]
            print(f"  {config_name:10s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v, 4) for v in values]}")

    print(f"\n{'=' * 78}\nDo the flagged findings survive across seeds?\n{'=' * 78}")

    def per_seed_diffs(config_a: str, config_b: str, task: str) -> list[float]:
        return [per_seed_auc[config_a][s][task] - per_seed_auc[config_b][s][task] for s in SEEDS]

    findings = [
        ("impact", "L2", "L1", "+0.292 (significant, single seed)"),
        ("impact", "L3", "L2", "-0.292 (significant, single seed)"),
        ("shortage", "L4", "L3", "+0.039 (significant, single seed)"),
    ]
    for task, hi, lo, original in findings:
        diffs = per_seed_diffs(hi, lo, task)
        signs = {d > 0 for d in diffs}
        agrees = len(signs) == 1
        print(f"\n  {task} {hi} vs {lo}  (original single-seed finding: {original})")
        print(f"    per-seed dAUC: {[round(d, 4) for d in diffs]}")
        print(f"    mean={statistics.fmean(diffs):+.4f}  std={statistics.pstdev(diffs):.4f}  "
              f"min={min(diffs):+.4f}  max={max(diffs):+.4f}")
        if agrees:
            print(f"    -> sign is CONSISTENT across all {len(SEEDS)} seeds -- more likely a real effect.")
        else:
            print("    -> sign FLIPS across seeds -- the original single-seed 'significant' result "
                  "is within model-fit noise, not a robust finding.")

    conn.close()


if __name__ == "__main__":
    main()
