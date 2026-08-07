"""
RGCN+attn ("Option 1") pilot — HGT vs RGCN vs RGCN+attn, fixed-d=64 only,
5 seeds each (15 runs total), against the v3 dataset (800 suppliers, 15
monthly snapshots Jul 2024 - Sep 2025 -- confirmed live against the
database before this script was written, not assumed).

Scope, deliberately narrow: this is a PILOT, not a full ablation arm. Only
the fixed-d=64 axis is run -- no matched-parameter search for RGCN+attn
(`ml/models/rgcn_attn_encoder.py`) is attempted here; that's future work,
worth doing only if this pilot shows something promising. Does not touch
`run_step5.py`, `run_step_v3.py`, or `run_step_v4_4arch.py`, and does not
re-run graphsage/gat -- this pilot is scoped to the three architectures
actually in question (does RGCN's basis-shared transform benefit from a
joint cross-relation attention step over its own per-relation-mean-then-sum
aggregation, and how does the hybrid compare to HGT).

Every run gets its own `model_registry` row, suffixed `-rgcnattn-pilot-
seed{n}` -- never upserted over any `-4arch-`, `-v3-`, `-sparserelation-`,
or `-coparent-` row from any prior round.

Run: python -u -m ml.run_rgcn_attn_pilot
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
from ml.models.encoder import build_encoder
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    load_all_snapshot_bundles,
    run_training_job,
    split_bundles,
)

SEEDS = [0, 1, 2, 3, 4]
HGT_LABEL = "heterogeneous_graph_transformer"

# (arm_name, internal_key, db_label) -- fixed-d=64 for all three, num_bases=8
# for both rgcn and rgcn_attn (their shared default; no search run this pass).
ARMS = [
    ("hgt", "hgt", HGT_LABEL),
    ("rgcn", "rgcn", "rgcn"),
    ("rgcn_attn", "rgcn_attn", "rgcn_attn"),
]
HIDDEN = 64
NUM_BASES = 8


def _train_eval_log(conn, model_version, arch_label, arch_key, train_b, val_b, test_b, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, arch_label, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, arch_label, results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    preds = {task: collect_predictions(model, test_b)[task] for task in TASKS}
    print(f"  [{elapsed:6.1f}s] {model_version:26s} params={model.parameter_count():,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, preds, elapsed


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

    # Real encoder-only parameter counts, recomputed live against this run's
    # real dims -- not assumed or reused from any prior round's numbers.
    metadata = train_b[0].data.metadata()
    in_dims = {nt: train_b[0].data[nt].x.size(-1) for nt in train_b[0].data.node_types}
    print("\n=== Real encoder-only parameter counts, hidden=64 (recomputed live) ===", flush=True)
    encoder_params = {}
    for arm_name, arch_key, _arch_label in ARMS:
        enc = build_encoder(arch_key, metadata, in_dims, hidden=HIDDEN, num_layers=4, num_bases=NUM_BASES)
        n = sum(p.numel() for p in enc.parameters())
        encoder_params[arm_name] = n
        print(f"  {arm_name:10s} {n:,}", flush=True)

    # auc[arm][seed][task] = auc ; preds[arm][seed][task] = (y_true, y_prob)
    auc = {a[0]: {} for a in ARMS}
    preds = {a[0]: {} for a in ARMS}
    run_elapsed = []

    print("\n=== RGCN+attn pilot: 5 seeds x 3 arms = 15 runs, fixed-d=64 only ===", flush=True)
    for seed in SEEDS:
        for arm_name, arch_key, arch_label in ARMS:
            model_version = f"{arm_name}-rgcnattn-pilot-seed{seed}"
            purpose = f"RGCN+attn (Option 1) pilot -- {arm_name} arm, fixed-d=64, num_bases={NUM_BASES}, seed={seed}"
            aucs, run_preds, elapsed = _train_eval_log(conn, model_version, arch_label, arch_key,
                                                          train_b, val_b, test_b, seed, purpose)
            auc[arm_name][seed] = aucs
            preds[arm_name][seed] = run_preds
            run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 15 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- summary stats
    print("=" * 90)
    print("AUC spread across 5 seeds, per (task, arm)")
    print("=" * 90)
    for arm_name, *_ in ARMS:
        for task in TASKS:
            values = [auc[arm_name][s][task] for s in SEEDS]
            print(f"  {arm_name:10s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    # --------------------------------------------------------------- sign-consistency
    print("\n" + "=" * 90)
    print("Sign-consistency across seeds -- rgcn_attn vs hgt, rgcn_attn vs rgcn, rgcn vs hgt")
    print("(re-measured here for internal consistency with this pilot, not assumed from the 4-arch round)")
    print("=" * 90)
    for anchor, other in [("rgcn_attn", "hgt"), ("rgcn_attn", "rgcn"), ("rgcn", "hgt")]:
        for task in TASKS:
            deltas, cis = [], []
            for seed in SEEDS:
                yt, pa = preds[anchor][seed][task]
                _, po = preds[other][seed][task]
                cmp = paired_delta_auc_ci(yt, pa, po)
                deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
                cis.append((cmp["ci_lower"], cmp["ci_upper"]))
            print(f"    {task:10s} {anchor} vs {other}:")
            for seed, d, (lo, hi) in zip(SEEDS, deltas, cis):
                sig = "" if lo is None else ("  (significant)" if (lo > 0 or hi < 0) else "  (CI overlaps 0)")
                lo_s = f"{lo:+.4f}" if lo is not None else "n/a"
                hi_s = f"{hi:+.4f}" if hi is not None else "n/a"
                print(f"      seed{seed}: dAUC={d:+.4f}  CI95=[{lo_s}, {hi_s}]{sig}")
            print(f"      -> mean={statistics.fmean(deltas):+.4f}  {_sign_consistency(deltas)}")

    # --------------------------------------------------------------- parameter counts
    print("\n" + "=" * 90)
    print("Real encoder-only parameter counts (hidden=64, num_bases=8 where applicable)")
    print("=" * 90)
    for arm_name, *_ in ARMS:
        print(f"  {arm_name:10s} {encoder_params[arm_name]:,}")

    print("\n" + "=" * 90)
    print("Per-run elapsed time")
    print("=" * 90)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:30s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
