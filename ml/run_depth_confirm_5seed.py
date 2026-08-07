"""
5-seed confirmation: does RGCN+Attention (SHARE)'s delay-peaks-at-L1 pattern
from `reports/rgcn_types.md`'s over-smoothing sweep survive proper multi-seed
testing, or was it 3-seed noise? Against the v3 dataset (800 suppliers, 15
monthly snapshots -- confirmed live, dual-sourcing/co-parent mechanism
confirmed active per `reports/entropy_test.md`, not assumed).

`reports/rgcn_types.md` found, at only 3 seeds (explicitly flagged there as
diagnostic, not a formal claim): delay AUC at L1 (shared_depth=1, 0.8154)
noticeably above baseline (structural-prior readout, delay reads layer 2 per
`STRUCTURAL_DEPTH_PRIOR`, 0.8105). This script retrains baseline, L1, and L2
fresh, 5 seeds each, so a genuine PAIRED bootstrap comparison is possible --
no model checkpoint is ever persisted in this codebase (only final AUC gets
logged to `model_registry`), so re-using the old logged baseline number
would not give paired predictions on the identical test set.

15 runs: 3 configs x 5 seeds, architecture fixed at rgcn_attn's established
matched-d config (hidden=128, num_bases=10). Every run gets its own
`model_registry` row, suffixed `-depthconfirm-{config}-seed{n}` -- never
touches any existing row, including `rgcn_types.md`'s own
`-oversmooth-rgcn_attn-*` rows.

Run: python -u -m ml.run_depth_confirm_5seed
"""

from __future__ import annotations

import statistics
import time

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs, paired_delta_auc_ci
from ml.models.depth import TASKS
from ml.train import TEST_START, TRAIN_CUTOFF, load_all_snapshot_bundles, run_training_job, split_bundles

SEEDS = [0, 1, 2, 3, 4]
HIDDEN, NUM_BASES = 128, 10  # rgcn_attn's established matched-d config

# (config_name, num_layers, shared_depth)
DEPTH_CONFIGS = [
    ("baseline", 4, None),   # structural-prior readout -- delay reads layer 2
    ("L1", 1, 1),            # every task forced to read layer 1
    ("L2", 2, 2),            # every task forced to read layer 2
]


def _train_eval_log(conn, model_version, train_b, val_b, test_b, num_layers, shared_depth, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, "rgcn_attn", train_b, val_b,
        num_layers=num_layers, shared_depth=shared_depth, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, "rgcn_attn", results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    preds = {task: collect_predictions(model, test_b)[task] for task in TASKS}
    n_params = model.parameter_count()
    print(f"  [{elapsed:6.1f}s] {model_version:32s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, preds, elapsed


def _sign_consistency(deltas: list[float]) -> str:
    signs = {d > 0 for d in deltas}
    return "CONSISTENT" if len(signs) == 1 else "FLIPS (noise)"


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM suppliers")
        n_suppliers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM graph_snapshots")
        n_snapshots = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM component_suppliers")
        n_coparent = cur.fetchone()[0]
    print(f"=== Live dataset check: {n_suppliers} suppliers, {n_snapshots} graph_snapshots, "
          f"{n_coparent} component_suppliers rows (dual-sourcing) ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    # auc[config][seed][task] ; preds[config][seed][task] = (y_true, y_prob)
    auc = {c[0]: {} for c in DEPTH_CONFIGS}
    preds = {c[0]: {} for c in DEPTH_CONFIGS}
    run_elapsed = []

    print("\n=== 5-seed depth confirmation: rgcn_attn (hidden=128, num_bases=10), "
          "3 configs x 5 seeds = 15 runs ===", flush=True)
    for seed in SEEDS:
        for config_name, num_layers, shared_depth in DEPTH_CONFIGS:
            model_version = f"rgcn_attn-depthconfirm-{config_name}-seed{seed}"
            purpose = (f"5-seed depth confirmation -- {config_name} "
                       f"(num_layers={num_layers}, shared_depth={shared_depth}), seed={seed}, "
                       f"testing whether the L1-delay-peak from rgcn_types.md's 3-seed sweep survives")
            aucs, run_preds, elapsed = _train_eval_log(conn, model_version, train_b, val_b, test_b,
                                                          num_layers, shared_depth, seed, purpose)
            auc[config_name][seed] = aucs
            preds[config_name][seed] = run_preds
            run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 15 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- AUC table, all 3 tasks
    print("=" * 90)
    print("AUC spread across 5 seeds, per (config, task)")
    print("=" * 90)
    for config_name, _, _ in DEPTH_CONFIGS:
        for task in TASKS:
            values = [auc[config_name][s][task] for s in SEEDS]
            print(f"  {config_name:10s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    # --------------------------------------------------------------- paired comparison, delay
    print("\n" + "=" * 90)
    print("Paired bootstrap + sign-consistency -- baseline vs L1, baseline vs L2, ALL TASKS "
          "(delay is the question this script answers; shortage/impact reported since shared_depth "
          "changes what every head reads from the shared encoder trunk)")
    print("=" * 90)
    verdicts = {}
    for hi_config in ["L1", "L2"]:
        for task in TASKS:
            deltas = []
            for seed in SEEDS:
                yt, p_base = preds["baseline"][seed][task]
                _, p_hi = preds[hi_config][seed][task]
                cmp = paired_delta_auc_ci(yt, p_hi, p_base)  # delta = hi_config - baseline
                deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
            verdict = _sign_consistency(deltas)
            if task == "delay":
                verdicts[hi_config] = (deltas, verdict)
            print(f"    {task:10s} {hi_config} vs baseline: per-seed dAUC={[round(d,4) for d in deltas]}  "
                  f"mean={statistics.fmean(deltas):+.4f}  -> {verdict}", flush=True)

    # --------------------------------------------------------------- explicit verdict
    print("\n" + "=" * 90)
    print("VERDICT -- does L1's apparent delay edge over baseline hold at 5 seeds?")
    print("=" * 90)
    for hi_config in ["L1", "L2"]:
        deltas, verdict = verdicts[hi_config]
        mean_d = statistics.fmean(deltas)
        real = verdict == "CONSISTENT" and mean_d > 0
        print(f"  {hi_config} vs baseline, delay: mean dAUC={mean_d:+.4f}, sign-consistency={verdict}")
        print(f"    -> {'REAL: ' + hi_config + ' beats baseline on delay, consistent across all 5 seeds.' if real else 'NOT CONFIRMED at 5 seeds -- ' + ('sign flips across seeds (noise), matching the original 3-seed diagnostic caveat.' if verdict != 'CONSISTENT' else hi_config + ' does not beat baseline on delay on average.')}")

    print("\n" + "=" * 90)
    print("Per-run elapsed time")
    print("=" * 90)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:36s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
