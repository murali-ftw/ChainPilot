"""
Steps 3-5 architecture ablation, 4 arms — HGT/GraphSAGE/GAT + RGCN
(`ml/models/rgcn_encoder.py`), against the v3 dataset (800 suppliers, 15
monthly snapshots Jul 2024 - Sep 2025 -- confirmed live against the database
at the time this script was written; the `component_suppliers` junction
table from Task 3's follow-up is present, but this run does NOT re-enable
that path -- it's an architecture-only ablation, same graph-construction
code as `run_step_v3.py`'s Step 5 arm).

40 total training runs: 5 seeds x 4 architectures x 2 param-arms
(fixed-d=64, matched-d). Multi-seed treatment is built in from the start for
every arm -- this run doesn't repeat the v1/v2 single-seed architecture
ablation mistake `run_step_v3.py`'s docstring already flagged.

Does NOT include the depth-config sweep (Step 6 is on hold pending the
reach/over-smoothing issues found in the Task 1/2 follow-up
(`reports/step5_result_v3_followup.md`) -- this script is Step 3-5's
architecture matrix only, re-run with a fourth arm added).

Matched-d values: GraphSAGE=66, GAT=92 (unchanged from `run_step5.py`/
`run_step_v3.py` -- re-verified against this run's real recomputed feature
dims before use, see module-level ARCH_ARMS comment). RGCN's matched arm
(hidden=138, num_bases=4) was found by grid-searching num_bases in
{4, 8, 12, 16} at hidden=64 first (none landed within an order of magnitude
of HGT's anchor -- self-loop + lin_in dominate RGCN's parameter count at
d=64, unlike GraphSAGE/GAT's HeteroConv-per-relation design), then widening
hidden alongside num_bases: hidden=138, num_bases=4 lands at 699,602 params,
0.21% off HGT's real 701,088-param anchor (recomputed fresh against this
run's live feature dims, not reused from v3's own docstring number).

Every run gets its own model_registry row, suffixed `-4arch-seed{n}` --
never upserted over any prior v1/v2/v3/StepA/StepB/Task2/Task3 row.

Run: python -u -m ml.run_step_v4_4arch   (unbuffered -- this runs for hours)
"""

from __future__ import annotations

import itertools
import statistics
import time

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

SEEDS = [0, 1, 2, 3, 4]
HGT_LABEL = "heterogeneous_graph_transformer"

# (arm_name, internal_key, db_label, hidden, arm, num_bases)
# GraphSAGE=66/GAT=92 matched-d values carried over from run_step5.py/
# run_step_v3.py (re-verified against this run's real recomputed feature
# dims in Phase 4's pre-flight search -- unchanged, dims haven't shifted
# since v3). RGCN's fixed arm uses build_encoder's num_bases default (8);
# its matched arm (hidden=138, num_bases=4) is this round's new search
# result -- see module docstring.
ARCH_ARMS = [
    ("hgt-fixed-d", "hgt", HGT_LABEL, 64, "fixed", 8),
    ("graphsage-fixed-d", "graphsage", "graphsage", 64, "fixed", 8),
    ("gat-fixed-d", "gat", "gat", 64, "fixed", 8),
    ("rgcn-fixed-d", "rgcn", "rgcn", 64, "fixed", 8),
    ("hgt-matched-d", "hgt", HGT_LABEL, 64, "matched", 8),
    ("graphsage-matched-d", "graphsage", "graphsage", 66, "matched", 8),
    ("gat-matched-d", "gat", "gat", 92, "matched", 8),
    ("rgcn-matched-d", "rgcn", "rgcn", 138, "matched", 4),
]


def _train_eval_log(conn, model_version, arch_label, arch_key, train_b, val_b, test_b,
                     hidden, seed, purpose, num_bases):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, arch_label, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=hidden, epochs=100,
        seed=seed, purpose=purpose, num_bases=num_bases,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, arch_label, results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    preds = {task: collect_predictions(model, test_b)[task] for task in TASKS}
    n_params = model.parameter_count()
    print(f"  [{elapsed:6.1f}s] {model_version:24s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, preds, elapsed, n_params


def _sign_consistency(deltas: list[float]) -> str:
    signs = {d > 0 for d in deltas}
    return "CONSISTENT" if len(signs) == 1 else "FLIPS (noise)"


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    # arch_auc[arm][seed][task] = auc ; arch_preds[arm][seed][task] = (y_true, y_prob)
    arch_auc = {a[0]: {} for a in ARCH_ARMS}
    arch_preds = {a[0]: {} for a in ARCH_ARMS}
    arch_params = {}  # arm_name -> parameter count (identical across seeds; init only affects weights)
    run_elapsed = []

    print("\n=== Steps 3-5, 4-architecture matrix: 5 seeds x 8 arms = 40 runs ===", flush=True)
    for seed in SEEDS:
        for arm_name, arch_key, arch_label, hidden, arm, num_bases in ARCH_ARMS:
            model_version = f"{arm_name}-4arch-seed{seed}"
            purpose = (f"4-arch (HGT/GraphSAGE/GAT/RGCN) architecture ablation -- "
                       f"{arm} arm, d={hidden}" +
                       (f", num_bases={num_bases}" if arch_key == "rgcn" else "") +
                       f", seed={seed}")
            aucs, preds, elapsed, n_params = _train_eval_log(conn, model_version, arch_label, arch_key,
                                                                train_b, val_b, test_b, hidden, seed,
                                                                purpose, num_bases)
            arch_auc[arm_name][seed] = aucs
            arch_preds[arm_name][seed] = preds
            arch_params[arm_name] = n_params
            run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 40 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- summary stats
    print("=" * 90)
    print("AUC spread across 5 seeds, per (task, architecture arm)")
    print("=" * 90)
    for arm_name, *_ in ARCH_ARMS:
        for task in TASKS:
            values = [arch_auc[arm_name][s][task] for s in SEEDS]
            print(f"  {arm_name:22s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    # --------------------------------------------------------------- architecture significance
    print("\n" + "=" * 90)
    print("Architecture ablation -- per seed then sign-consistency across seeds")
    print("Comparisons: every arch vs HGT, plus GAT/RGCN vs GraphSAGE (per axis)")
    print("=" * 90)
    for axis_name, arm_suffix in [("fixed-d", "fixed-d"), ("matched-d", "matched-d")]:
        hgt_arm = f"hgt-{arm_suffix}"
        graphsage_arm = f"graphsage-{arm_suffix}"
        gat_arm = f"gat-{arm_suffix}"
        rgcn_arm = f"rgcn-{arm_suffix}"
        print(f"\n  -- {axis_name} axis --")
        for anchor, others in [(hgt_arm, [graphsage_arm, gat_arm, rgcn_arm]),
                                (graphsage_arm, [gat_arm, rgcn_arm])]:
            for task in TASKS:
                for other in others:
                    deltas = []
                    for seed in SEEDS:
                        yt, pa = arch_preds[anchor][seed][task]
                        _, po = arch_preds[other][seed][task]
                        cmp = paired_delta_auc_ci(yt, pa, po)
                        deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
                    print(f"    {task:10s} {anchor} vs {other}: per-seed dAUC={[round(d,4) for d in deltas]}  "
                          f"mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}")

    # --------------------------------------------------------------- parameter counts
    print("\n" + "=" * 90)
    print("Parameter counts (identical across seeds -- seed only affects init, not architecture size)")
    print("=" * 90)
    for arm_name, *_ in ARCH_ARMS:
        print(f"  {arm_name:22s} {arch_params[arm_name]:,}")

    print("\n" + "=" * 90)
    print("Per-run elapsed time")
    print("=" * 90)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:28s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
