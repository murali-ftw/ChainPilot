"""
Step 6 Phase 1 pilot — Markov-blanket-derived FIXED depth readout on top of SHARE
(RGCN+Attention, `rgcn_attn` in code), against the v3 dataset (800 suppliers, 15 monthly
snapshots -- confirmed live before this script was written, not assumed). Full design
rationale: `ml/models/rgcn_attn_markov_encoder.py`'s module docstring, and
"Markov Scoping and Transformer 1.md" §7.1 ("Phase 1 -- Markov only").

Per-task fixed readout depth (no learned weights, zero new parameters vs. plain `rgcn_attn`):
delay -> h^1 (empirically confirmed, `reports/step6_preflight.md`, NOT the theory doc's
untested h^2 floor), shortage -> h^3, impact -> h^4 (both matching the theory doc's blanket
estimate AND `reports/step6_depth_gate.md`'s learned-gate convergence point).

10 runs total: 5 `architecture="rgcn_attn_markov"` (fixed depth) + 5
`architecture="rgcn_attn"` (`shared_depth=None`, plain baseline) retrained fresh in this SAME
process for a genuine PAIRED comparison -- no model checkpoint is ever persisted in this
codebase, so pairing against any previously-logged baseline number isn't possible without
retraining, same rule every prior round in this line has followed. `reports/step6_depth_gate.md`'s
learned-gate numbers are cited below as UNPAIRED reference context only, not retrained (that
round's own job, not this script's).

Every run gets its own `model_registry` row, suffixed `-markovpilot-seed{n}` (fixed depth) or
`-markovpilot-baselinecmp-seed{n}` (baseline retrain) -- never touches any existing row,
including the depth-gate pilot's own `-depthgate-*` rows.

Run: python -u -m ml.run_markov_pilot
"""

from __future__ import annotations

import statistics
import time

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs, paired_delta_auc_ci
from ml.models.depth import TASKS
from ml.train import TEST_START, TRAIN_CUTOFF, load_all_snapshot_bundles, run_training_job, split_bundles

SEEDS = [0, 1, 2, 3, 4]
HIDDEN, NUM_BASES = 128, 10  # SHARE's established matched-d config, reports/step6_preflight.md

# Unpaired reference context from reports/step6_depth_gate.md -- NOT retrained here, cited
# as-is for comparison only. (arm, task) -> mean AUC, 5 seeds.
DEPTH_GATE_REFERENCE = {
    ("baseline_retrain", "delay"): 0.8083, ("baseline_retrain", "shortage"): 0.7977, ("baseline_retrain", "impact"): 0.9363,
    ("gate_lambda0.01", "delay"): 0.8080, ("gate_lambda0.01", "shortage"): 0.8041, ("gate_lambda0.01", "impact"): 0.9373,
    ("gate_lambda0.1", "delay"): 0.8113, ("gate_lambda0.1", "shortage"): 0.8021, ("gate_lambda0.1", "impact"): 0.9370,
    ("gate_lambda1.0", "delay"): 0.8099, ("gate_lambda1.0", "shortage"): 0.8018, ("gate_lambda1.0", "impact"): 0.9383,
}


def _train_eval_log(conn, model_version, architecture, train_b, val_b, test_b, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, architecture, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, architecture, results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    preds = {task: collect_predictions(model, test_b)[task] for task in TASKS}
    n_params = model.parameter_count()
    print(f"  [{elapsed:6.1f}s] {model_version:36s} params={n_params:,} "
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
    print(f"=== Live dataset check: {n_suppliers} suppliers, {n_snapshots} graph_snapshots ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    # auc[arm][seed][task] ; preds[arm][seed][task]
    auc = {"markov": {}, "baseline_retrain": {}}
    preds = {"markov": {}, "baseline_retrain": {}}
    run_elapsed = []

    print("\n=== Markov fixed-depth (architecture=rgcn_attn_markov): 5 seeds ===", flush=True)
    for seed in SEEDS:
        model_version = f"rgcn_attn_markov-markovpilot-seed{seed}"
        purpose = (f"Step 6 Phase 1 -- Markov-blanket-derived FIXED per-task depth "
                   f"(delay=h1, shortage=h3, impact=h4), zero new parameters, seed={seed}")
        aucs, run_preds, elapsed = _train_eval_log(conn, model_version, "rgcn_attn_markov",
                                                     train_b, val_b, test_b, seed, purpose)
        auc["markov"][seed] = aucs
        preds["markov"][seed] = run_preds
        run_elapsed.append((model_version, elapsed))

    print("\n=== Baseline retrain (fixed structural-prior, architecture=rgcn_attn): 5 seeds ===", flush=True)
    for seed in SEEDS:
        model_version = f"rgcn_attn-markovpilot-baselinecmp-seed{seed}"
        purpose = (f"Step 6 Phase 1 pilot -- plain rgcn_attn baseline, retrained fresh "
                   f"for paired comparison against the Markov fixed-depth design, seed={seed}")
        aucs, run_preds, elapsed = _train_eval_log(conn, model_version, "rgcn_attn",
                                                     train_b, val_b, test_b, seed, purpose)
        auc["baseline_retrain"][seed] = aucs
        preds["baseline_retrain"][seed] = run_preds
        run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 10 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- AUC table
    print("=" * 95)
    print("AUC spread across 5 seeds, per (arm, task)")
    print("=" * 95)
    for arm in ["markov", "baseline_retrain"]:
        for task in TASKS:
            values = [auc[arm][s][task] for s in SEEDS]
            print(f"  {arm:18s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    print("\n  -- reports/step6_depth_gate.md reference (unpaired, NOT retrained here) --")
    for arm in ["baseline_retrain", "gate_lambda0.01", "gate_lambda0.1", "gate_lambda1.0"]:
        for task in TASKS:
            print(f"  {arm:18s} {task:10s} mean={DEPTH_GATE_REFERENCE[(arm, task)]:.4f}")

    # --------------------------------------------------------------- paired bootstrap vs baseline_retrain
    print("\n" + "=" * 95)
    print("Paired bootstrap + sign-consistency -- markov vs baseline_retrain, PAIRED (both trained "
          "in-process this round)")
    print("=" * 95)
    for task in TASKS:
        deltas = []
        for seed in SEEDS:
            yt, p_markov = preds["markov"][seed][task]
            _, p_base = preds["baseline_retrain"][seed][task]
            cmp = paired_delta_auc_ci(yt, p_markov, p_base)
            deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
        verdict = _sign_consistency(deltas)
        print(f"    {task:10s} per-seed dAUC={[round(d,4) for d in deltas]}  "
              f"mean={statistics.fmean(deltas):+.4f}  -> {verdict}", flush=True)

    # --------------------------------------------------------------- plain verdict
    print("\n" + "=" * 95)
    print("Verdict -- does the free, theory-derived fixed depth design help, hurt, or make no")
    print("detectable difference relative to baseline, per task?")
    print("=" * 95)
    for task in TASKS:
        deltas = []
        for seed in SEEDS:
            yt, p_markov = preds["markov"][seed][task]
            _, p_base = preds["baseline_retrain"][seed][task]
            cmp = paired_delta_auc_ci(yt, p_markov, p_base)
            deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
        mean_delta = statistics.fmean(deltas)
        verdict = _sign_consistency(deltas)
        if verdict == "CONSISTENT" and mean_delta > 0:
            label = "HELPS (5-seed consistent gain)"
        elif verdict == "CONSISTENT" and mean_delta < 0:
            label = "HURTS (5-seed consistent loss)"
        else:
            label = "NO DETECTABLE DIFFERENCE (sign flips across seeds)"
        print(f"    {task:10s} mean dAUC={mean_delta:+.4f}  -> {label}")

    print("\n" + "=" * 95)
    print("Per-run elapsed time")
    print("=" * 95)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:40s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
