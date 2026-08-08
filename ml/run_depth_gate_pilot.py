"""
Step 6 pilot — learned, per-task depth gate on top of SHARE (RGCN+Attention,
`rgcn_attn` in code), against the v3 dataset (800 suppliers, 15 monthly
snapshots -- confirmed live before this script was written, not assumed).
Full design rationale: `ml/models/rgcn_attn_depthgate_encoder.py`'s module
docstring. Motivated by `reports/step6_preflight.md`'s 5-seed-confirmed
finding that delay wants a shallow (~L1) read while impact wants a deep
(~L4) one, from the SAME shared encoder trunk -- no single fixed depth
serves every task, which is exactly the condition a per-task learned gate
is for.

15 gated runs (3 lambda values x 5 seeds, `architecture="rgcn_attn_depthgate"`)
+ 5 fixed structural-prior baseline runs (`architecture="rgcn_attn"`,
`shared_depth=None`), retrained fresh in this same process for a genuine
PAIRED comparison -- `reports/step6_preflight.md`'s own baseline/L1/L2
predictions aren't recoverable (no model checkpoint is ever persisted in
this codebase), so pairing against them isn't possible without retraining,
same rule every prior round in this line has followed. L1/L2's own already-
logged numbers are cited below as UNPAIRED reference context only, not
retrained (that's `reports/step6_preflight.md`'s own job, not this script's).

20 runs total. Every run gets its own `model_registry` row, suffixed
`-depthgate-lambda{lam}-seed{n}` (gated) or `-depthgate-baselinecmp-seed{n}`
(baseline retrain) -- never touches any existing row, including
`reports/step6_preflight.md`'s own `-depthconfirm-*` rows.

Run: python -u -m ml.run_depth_gate_pilot
"""

from __future__ import annotations

import statistics
import time

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs, paired_delta_auc_ci
from ml.models.depth import TASKS
from ml.train import TEST_START, TRAIN_CUTOFF, load_all_snapshot_bundles, run_training_job, split_bundles

SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.01, 0.1, 1.0]
HIDDEN, NUM_BASES = 128, 10  # SHARE's established matched-d config, reports/step6_preflight.md

# Unpaired reference context from reports/step6_preflight.md -- NOT retrained
# here, cited as-is for comparison only. (config, task) -> mean AUC, 5 seeds.
STEP6_PREFLIGHT_REFERENCE = {
    ("baseline", "delay"): 0.8116, ("baseline", "shortage"): 0.7991, ("baseline", "impact"): 0.9375,
    ("L1", "delay"): 0.8185, ("L1", "shortage"): 0.7941, ("L1", "impact"): 0.9215,
    ("L2", "delay"): 0.8106, ("L2", "shortage"): 0.7962, ("L2", "impact"): 0.9352,
}


def _train_eval_log(conn, model_version, architecture, train_b, val_b, test_b,
                     seed, purpose, lambda_kl=0.1):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, architecture, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES, lambda_kl=lambda_kl,
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
    gate_weights = model.gate_weight_summary() if hasattr(model, "gate_weight_summary") else None
    print(f"  [{elapsed:6.1f}s] {model_version:36s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    if gate_weights is not None:
        for task in TASKS:
            print(f"      {task:10s} gate weights (h0..h4) = "
                  f"{[round(w,4) for w in gate_weights[task]]}", flush=True)
    return aucs, preds, elapsed, gate_weights


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

    # auc[arm][seed][task] ; preds[arm][seed][task] ; gate_weights[lam][seed][task]
    auc = {"baseline_retrain": {}}
    preds = {"baseline_retrain": {}}
    gate_weights_by_lambda: dict[float, dict[int, dict[str, list[float]]]] = {}
    run_elapsed = []

    print("\n=== Baseline retrain (fixed structural-prior, architecture=rgcn_attn): 5 seeds ===", flush=True)
    for seed in SEEDS:
        model_version = f"rgcn_attn-depthgate-baselinecmp-seed{seed}"
        purpose = (f"Step 6 depth-gate pilot -- fixed structural-prior baseline, retrained fresh "
                   f"for paired comparison against the learned gate, seed={seed}")
        aucs, run_preds, elapsed, _gw = _train_eval_log(conn, model_version, "rgcn_attn",
                                                           train_b, val_b, test_b, seed, purpose)
        auc["baseline_retrain"][seed] = aucs
        preds["baseline_retrain"][seed] = run_preds
        run_elapsed.append((model_version, elapsed))

    print("\n=== Learned depth gate: 3 lambda values x 5 seeds = 15 runs ===", flush=True)
    for lam in LAMBDAS:
        arm = f"gate_lambda{lam}"
        auc[arm] = {}
        preds[arm] = {}
        gate_weights_by_lambda[lam] = {}
        for seed in SEEDS:
            model_version = f"rgcn_attn_depthgate-lambda{lam}-seed{seed}"
            purpose = f"Step 6 depth-gate pilot -- learned per-task gate, lambda_kl={lam}, seed={seed}"
            aucs, run_preds, elapsed, gw = _train_eval_log(conn, model_version, "rgcn_attn_depthgate",
                                                              train_b, val_b, test_b, seed, purpose, lambda_kl=lam)
            auc[arm][seed] = aucs
            preds[arm][seed] = run_preds
            gate_weights_by_lambda[lam][seed] = gw
            run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 20 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- AUC table
    print("=" * 95)
    print("AUC spread across 5 seeds, per (arm, task)")
    print("=" * 95)
    arms = ["baseline_retrain"] + [f"gate_lambda{lam}" for lam in LAMBDAS]
    for arm in arms:
        for task in TASKS:
            values = [auc[arm][s][task] for s in SEEDS]
            print(f"  {arm:22s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    print("\n  -- reports/step6_preflight.md reference (unpaired, NOT retrained here) --")
    for config in ["baseline", "L1", "L2"]:
        for task in TASKS:
            print(f"  {config:22s} {task:10s} mean={STEP6_PREFLIGHT_REFERENCE[(config, task)]:.4f}")

    # --------------------------------------------------------------- paired bootstrap vs baseline_retrain
    print("\n" + "=" * 95)
    print("Paired bootstrap + sign-consistency -- gate(lambda) vs baseline_retrain, PAIRED (both trained "
          "in-process this round)")
    print("=" * 95)
    for lam in LAMBDAS:
        arm = f"gate_lambda{lam}"
        for task in TASKS:
            deltas = []
            for seed in SEEDS:
                yt, p_gate = preds[arm][seed][task]
                _, p_base = preds["baseline_retrain"][seed][task]
                cmp = paired_delta_auc_ci(yt, p_gate, p_base)
                deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
            verdict = _sign_consistency(deltas)
            print(f"    lambda={lam:<5} {task:10s} per-seed dAUC={[round(d,4) for d in deltas]}  "
                  f"mean={statistics.fmean(deltas):+.4f}  -> {verdict}", flush=True)

    # --------------------------------------------------------------- gate weights, the actual point
    print("\n" + "=" * 95)
    print("Learned gate weights (softmax(w_task), depth-ordered h^0..h^4), mean +/- std across 5 seeds")
    print("=" * 95)
    for lam in LAMBDAS:
        print(f"\n  -- lambda={lam} --")
        for task in TASKS:
            per_seed = [gate_weights_by_lambda[lam][s][task] for s in SEEDS]
            n_depths = len(per_seed[0])
            means = [statistics.fmean(per_seed[s][d] for s in range(len(SEEDS))) for d in range(n_depths)]
            stds = [statistics.pstdev(per_seed[s][d] for s in range(len(SEEDS))) for d in range(n_depths)]
            print(f"    {task:10s} mean=[{', '.join(f'{m:.3f}' for m in means)}]  "
                  f"std=[{', '.join(f'{s:.3f}' for s in stds)}]  argmax_depth={means.index(max(means))}")

    # --------------------------------------------------------------- lambda sensitivity
    print("\n" + "=" * 95)
    print("Lambda sensitivity -- how much does the learned gate shift across lambda values?")
    print("=" * 95)
    for task in TASKS:
        print(f"\n  -- {task} --")
        for lam in LAMBDAS:
            per_seed = [gate_weights_by_lambda[lam][s][task] for s in SEEDS]
            n_depths = len(per_seed[0])
            means = [statistics.fmean(per_seed[s][d] for s in range(len(SEEDS))) for d in range(n_depths)]
            print(f"    lambda={lam:<5} mean gate=[{', '.join(f'{m:.3f}' for m in means)}]")

    print("\n" + "=" * 95)
    print("Per-run elapsed time")
    print("=" * 95)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:40s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
