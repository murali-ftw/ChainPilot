"""
Task 3 (post-v3 follow-up, `reports/step5_result_v3.md` addendum) --
**a genuinely separate experiment**, not a fix to the v3 result: does HGT's
relation-type-aware message passing win over GraphSAGE on shortage/delay once
a REAL causal co-parent signal actually exists in the graph -- something the
base v1/v2/v3 datasets never had (`components.supplier_id` is a single
not-null FK, so the Supplier->SUPPLIES->Component->rev_SUPPLIES->Supplier
path in `docs/06_Graph_Database_Design.md` §6.1 provably could not reach a
second supplier; Task 1 re-confirmed 0/800 on the base v3 graph).

Dataset for this run: `db/generate_dataset.py` + `db/schema.sql`, both now
carrying the Task 3 addendum -- a `component_suppliers` junction table
(~15-20% of components get a second qualified supplier) with a real causal
coupling (`COPARENT_COUPLING = 0.35`): a co-parent partner's own stress
measurably bleeds into its partner's stress, discoverable only by actually
traversing the co-parent edge (not derivable from either supplier's own
history alone). NOTE: because `stress()` feeds directly into shipment-delay
branching (`random.random() < p_delay`), which consumes a variable number of
downstream RNG draws, this coupling change desyncs the shared `random`
stream from that point on -- so this is a genuinely different generated
world throughout, not "v3 plus one extra table with everything else held
fixed." Still fully deterministic (byte-identical across reruns of this
script) and passes every existing validation check.

Re-verified first (`ml/graph/reach.py`): co-parent reach on this graph is
449/800 (56%), vs 0/800 on the base v3 graph -- the structural precondition
Claim 2's h^2 "co-parent" story always assumed is now actually true.

5 seeds x 2 architectures (hgt, graphsage) at matched hidden=64 (fixed-d
only -- this is the restricted comparison the task asked for, not a repeat
of the full 6-arm ablation matrix) = 10 runs. Both trained in-process here,
so shortage/delay AUC deltas use the SAME paired-bootstrap CI method
(`ml/evaluate.py::paired_delta_auc_ci`) Steps 4/5 used throughout, not the
weaker same-seed-index delta Task 2 had to fall back to.

Governance: backed up model_registry/model_evaluation_runs (pg_dump +
in-DB copy tables) before the `--drop` reload this experiment required,
restored immediately after, per the v3 --drop lesson -- not repeated a
third time. All rows from this script use `-coparent-v4-seed{n}` (a new
"v4" tag, since this is a distinct generated world, not v3).

Run: python -u -m ml.run_task3_coparent
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
ARMS = [
    ("hgt-coparent", "hgt", HGT_LABEL, 64),
    ("graphsage-coparent", "graphsage", "graphsage", 64),
]


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

    arch_auc = {a[0]: {} for a in ARMS}     # [arm][seed][task] = auc
    arch_preds = {a[0]: {} for a in ARMS}   # [arm][seed][task] = (y_true, y_prob)

    print("\n=== Task 3: restricted architecture comparison, 5 seeds x 2 arms = 10 runs ===", flush=True)
    for seed in SEEDS:
        for arm_name, arch_key, arch_label, hidden in ARMS:
            model_version = f"{arm_name}-v4-seed{seed}"
            t0 = time.monotonic()
            result = run_training_job(
                conn, model_version, arch_label, train_b, val_b,
                num_layers=4, shared_depth=None, hidden=hidden, epochs=100, seed=seed,
                purpose=f"Task 3 co-parent-signal experiment (separate from v1/v2/v3) -- "
                        f"{arch_key} arm, d={hidden}, seed={seed}, on the component_suppliers-"
                        f"augmented graph (real causal co-parent coupling, reach 449/800)",
            )
            model = result["model"]
            thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
            results = evaluate_split(model, test_b, thresholds)
            log_evaluation_runs(conn, model_version, arch_label, results, "test",
                                 train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
            elapsed = time.monotonic() - t0
            aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
            arch_auc[arm_name][seed] = aucs
            arch_preds[arm_name][seed] = {task: collect_predictions(model, test_b)[task] for task in TASKS}
            print(f"  [{elapsed:6.1f}s] {model_version:28s} best_epoch={result['best_epoch']:3d}  "
                  f"params={model.parameter_count():,}  " +
                  "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== 10 runs complete in {total_elapsed/60:.1f} min ===\n", flush=True)

    print("=" * 78)
    print("AUC spread across 5 seeds, per (task, arm)")
    print("=" * 78)
    for arm_name, _, _, _ in ARMS:
        for task in TASKS:
            values = [arch_auc[arm_name][s][task] for s in SEEDS]
            print(f"  {arm_name:20s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")

    print("\n" + "=" * 78)
    print("hgt-coparent vs graphsage-coparent -- paired per-seed AUC deltas (shortage, delay)")
    print("(same paired-bootstrap method as Steps 4/5 and the main v3 round; "
          "original v3 fixed-d gap for reference: shortage -0.0123, consistent loss)")
    print("=" * 78)
    for task in ("shortage", "delay"):
        deltas = []
        for seed in SEEDS:
            yt, ph = arch_preds["hgt-coparent"][seed][task]
            _, po = arch_preds["graphsage-coparent"][seed][task]
            cmp = paired_delta_auc_ci(yt, ph, po)
            deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
        consistency = _sign_consistency(deltas)
        print(f"\n  -- {task} --")
        print(f"    hgt_coparent - graphsage_coparent per-seed dAUC: {[round(d,4) for d in deltas]}")
        print(f"    mean={statistics.fmean(deltas):+.4f}  -> {consistency}")
        if consistency == "CONSISTENT" and statistics.fmean(deltas) > 0:
            verdict = "HGT wins under the same 5-seed sign-consistency standard used throughout"
        elif consistency == "CONSISTENT":
            verdict = "HGT still loses (or ties), even with a real co-parent signal now present"
        else:
            verdict = "inconsistent across seeds -- no clear verdict either way"
        print(f"    verdict: {verdict}")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
