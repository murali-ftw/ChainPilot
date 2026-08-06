"""
Steps 3-5 multi-seed run matrix against the v3 dataset (800 suppliers, 15
monthly snapshots Jul 2024 - Sep 2025).

55 total training runs:
  - 5 seeds x 5 depth-configs (fixed structural-prior baseline + shared-depth
    L=1..4)                                                          = 25
  - 5 seeds x 6 architecture arms (GraphSAGE/GAT/HGT x fixed-d/matched-d) = 30

Multi-seed treatment is built in for BOTH the depth sweep and the
architecture ablation from the start. The prior round (v1/v2) only did this
for the depth sweep (`ml/run_step_b.py`) -- and that's exactly what caught
three single-seed "significant" findings that were actually noise. Repeating
that gap for the architecture ablation was avoided by construction here.

Every one of the 55 runs gets its own `model_registry` row, suffixed
`-v3-seed{n}` -- never upserted over v1/v2's existing rows (the v1->v2
transition did that and destroyed v1's rows; v2's rows must survive this
pass).

Run: python -u -m ml.run_step_v3   (unbuffered -- this runs for hours)
"""

from __future__ import annotations

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

# (config_name, num_layers, shared_depth) -- architecture is always hgt, d=64
DEPTH_CONFIGS = [
    ("baseline", 4, None),
    ("L1", 1, 1),
    ("L2", 2, 2),
    ("L3", 3, 3),
    ("L4", 4, 4),
]

# (arm_name, internal_key, db_label, hidden, arm)
ARCH_ARMS = [
    ("hgt-fixed-d", "hgt", HGT_LABEL, 64, "fixed"),
    ("graphsage-fixed-d", "graphsage", "graphsage", 64, "fixed"),
    ("gat-fixed-d", "gat", "gat", 64, "fixed"),
    ("hgt-matched-d", "hgt", HGT_LABEL, 64, "matched"),
    ("graphsage-matched-d", "graphsage", "graphsage", 66, "matched"),
    ("gat-matched-d", "gat", "gat", 92, "matched"),
]


def _train_eval_log(conn, model_version, arch_label, arch_key, train_b, val_b, test_b,
                     num_layers, shared_depth, hidden, seed, purpose, depth_l=None):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, arch_label, train_b, val_b,
        num_layers=num_layers, shared_depth=shared_depth, hidden=hidden, epochs=100,
        seed=seed, purpose=purpose,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, arch_label, results, "test", depth_l=depth_l,
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    preds = {task: collect_predictions(model, test_b)[task] for task in TASKS}
    print(f"  [{elapsed:6.1f}s] {model_version:28s} best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, preds


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

    # depth_auc[config][seed][task] = auc ; depth_preds[config][seed][task] = (y_true, y_prob)
    depth_auc = {c[0]: {} for c in DEPTH_CONFIGS}
    depth_preds = {c[0]: {} for c in DEPTH_CONFIGS}

    print("\n=== Steps 3-4: depth-config sweep, 5 seeds x 5 configs = 25 runs ===", flush=True)
    for seed in SEEDS:
        for config_name, num_layers, shared_depth in DEPTH_CONFIGS:
            model_version = f"hgt-{config_name}-v3-seed{seed}" if config_name != "baseline" \
                else f"hgt-baseline-v3-seed{seed}"
            depth_l = shared_depth if shared_depth is not None else 4
            purpose = (f"v3 multi-seed depth sweep -- {config_name} "
                       f"({'fixed structural prior' if shared_depth is None else f'shared depth L={shared_depth}'}), seed={seed}")
            aucs, preds = _train_eval_log(conn, model_version, HGT_LABEL, "hgt", train_b, val_b, test_b,
                                           num_layers, shared_depth, 64, seed, purpose, depth_l=depth_l)
            depth_auc[config_name][seed] = aucs
            depth_preds[config_name][seed] = preds

    # arch_auc[arm][seed][task] = auc ; arch_preds[arm][seed][task] = (y_true, y_prob)
    arch_auc = {a[0]: {} for a in ARCH_ARMS}
    arch_preds = {a[0]: {} for a in ARCH_ARMS}

    print("\n=== Step 5: architecture ablation, 5 seeds x 6 arms = 30 runs ===", flush=True)
    for seed in SEEDS:
        for arm_name, arch_key, arch_label, hidden, arm in ARCH_ARMS:
            model_version = f"{arm_name}-v3-seed{seed}"
            purpose = f"v3 multi-seed architecture ablation -- {arm} arm, d={hidden}, seed={seed}"
            aucs, preds = _train_eval_log(conn, model_version, arch_label, arch_key, train_b, val_b, test_b,
                                           4, None, hidden, seed, purpose)
            arch_auc[arm_name][seed] = aucs
            arch_preds[arm_name][seed] = preds

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 55 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- summary stats
    print("=" * 78)
    print("AUC spread across 5 seeds, per (task, depth-config)")
    print("=" * 78)
    for config_name, _, _ in DEPTH_CONFIGS:
        for task in TASKS:
            values = [depth_auc[config_name][s][task] for s in SEEDS]
            print(f"  {config_name:10s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    print("\n" + "=" * 78)
    print("AUC spread across 5 seeds, per (task, architecture arm)")
    print("=" * 78)
    for arm_name, _, _, _, _ in ARCH_ARMS:
        for task in TASKS:
            values = [arch_auc[arm_name][s][task] for s in SEEDS]
            print(f"  {arm_name:22s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    # --------------------------------------------------------------- Claim 2: depth significance, per-seed
    print("\n" + "=" * 78)
    print("Claim 2 -- nested significance, per seed then sign-consistency across seeds")
    print("=" * 78)
    for task in TASKS:
        print(f"\n  -- {task} --")
        for hi, lo in [("L2", "L1"), ("L3", "L2"), ("L4", "L3")]:
            deltas = []
            for seed in SEEDS:
                yt, ph = depth_preds[hi][seed][task]
                _, pl = depth_preds[lo][seed][task]
                cmp = paired_delta_auc_ci(yt, ph, pl)
                deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
            print(f"    {hi} vs {lo}: per-seed dAUC={[round(d,4) for d in deltas]}  "
                  f"mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}")
        for cfg in ["L1", "L2", "L3", "L4"]:
            deltas = []
            for seed in SEEDS:
                yt, pp = depth_preds["baseline"][seed][task]
                _, ps = depth_preds[cfg][seed][task]
                cmp = paired_delta_auc_ci(yt, pp, ps)
                deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
            print(f"    prior vs {cfg}: per-seed dAUC={[round(d,4) for d in deltas]}  "
                  f"mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}")

    # --------------------------------------------------------------- Claim 1: architecture significance, per-seed
    print("\n" + "=" * 78)
    print("Claim 1 -- architecture ablation, per seed then sign-consistency across seeds")
    print("=" * 78)
    for axis_name, hgt_arm, others in [
        ("fixed-d", "hgt-fixed-d", ["graphsage-fixed-d", "gat-fixed-d"]),
        ("matched-d", "hgt-matched-d", ["graphsage-matched-d", "gat-matched-d"]),
    ]:
        print(f"\n  -- {axis_name} axis --")
        for task in TASKS:
            for other in others:
                deltas = []
                for seed in SEEDS:
                    yt, ph = arch_preds[hgt_arm][seed][task]
                    _, po = arch_preds[other][seed][task]
                    cmp = paired_delta_auc_ci(yt, ph, po)
                    deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
                print(f"    {task:10s} {hgt_arm} vs {other}: per-seed dAUC={[round(d,4) for d in deltas]}  "
                      f"mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
