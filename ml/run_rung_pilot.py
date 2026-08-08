"""
Step 6 Phase 2 pilot — Rung 4 (attention-only) and Rung 5 (prior-initialized MLP)
PER-NODE depth gates on top of SHARE (RGCN+Attention, `rgcn_attn` in code), trained
against a freshly-retrained Markov Phase 1 baseline (`ml/models/rgcn_attn_markov_encoder.py`,
`reports/step6_markov_phase1.md`) for a direct three-way comparison. Full design
rationale: `ml/models/rung_gate_common.py`, `ml/models/rgcn_attn_rung4_encoder.py`,
`ml/models/rgcn_attn_rung5_encoder.py`'s module docstrings, and
"Markov Scoping and Transformer 1.md" Part 6 (the rung ladder) / §7.2 ("Phase 2 -- add
the gate").

15 runs total: 5 seeds x {rgcn_attn_markov, rgcn_attn_rung4, rgcn_attn_rung5}, ALL
retrained fresh in this SAME process -- no model checkpoint is ever persisted in this
codebase, so pairing against `reports/step6_markov_phase1.md`'s own already-logged
numbers isn't valid; retrain fresh, same rule every prior round has followed.

Beyond AUC, this script's main job is the PER-NODE analysis rung4/rung5 exist to enable
(the earlier task-level gate, `reports/step6_depth_gate.md`, could never produce this,
since it only ever had one distribution per task): for every node of a task's entity
type in the test split, does its post-training argmax depth match the Markov prior depth
that gate was initialized at? For nodes that deviate, does the deviation correlate with
node degree (hub vs. leaf, in the theory doc's own language, §5.1)? And did the gate's
weights move meaningfully from their near-one-hot initialization at all, or stay
pinned -- the "training stability" question.

Run: python -u -m ml.run_rung_pilot
"""

from __future__ import annotations

import math
import statistics
import time

import torch

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs, paired_delta_auc_ci
from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.train import TEST_START, TRAIN_CUTOFF, load_all_snapshot_bundles, run_training_job, split_bundles

SEEDS = [0, 1, 2, 3, 4]
HIDDEN, NUM_BASES = 128, 10  # SHARE's established matched-d config
ARMS = ["rgcn_attn_markov", "rgcn_attn_rung4", "rgcn_attn_rung5"]
ARM_LABEL = {"rgcn_attn_markov": "markov", "rgcn_attn_rung4": "rung4", "rgcn_attn_rung5": "rung5"}


def _entity_degree(data, entity_type: str) -> torch.Tensor:
    """Total degree touching each node of `entity_type` in this snapshot, summed across
    every relation where it appears as the destination side. `ml/graph/builder.py`
    applies `ToUndirected()`, so every original edge has a reverse counterpart -- summing
    dst-side occurrences across ALL edge_types already counts each undirected edge
    exactly once per endpoint, giving a consistent per-node degree without double
    counting src- and dst-side of the same underlying edge."""
    n = data[entity_type].x.size(0)
    degree = torch.zeros(n, dtype=torch.long)
    for (_src, _rel, dst), edge_index in data.edge_index_dict.items():
        if dst == entity_type and edge_index.numel() > 0:
            degree += torch.bincount(edge_index[1], minlength=n)
    return degree


@torch.no_grad()
def _per_node_gate_stats(model, test_bundles, task: str) -> dict:
    """Pools every node of `task`'s entity type across all test snapshots (not just
    labelled ones -- the gate runs on every node regardless of label) and reports:
    match_rate (share whose argmax depth equals the Markov prior depth), the mean node
    degree of matching vs. deviating nodes, and the post-training mean/std gate weight
    per depth (std captures per-node spread -- the thing a task-level gate could never
    have)."""
    entity_type = TASK_ENTITY_TYPE[task]
    target_depth = MARKOV_READOUT_DEPTH[task]
    model.eval()
    alphas, degrees = [], []
    for bundle in test_bundles:
        model(bundle.data.x_dict, bundle.data.edge_index_dict)
        alphas.append(model._last_gate_weights[task])
        degrees.append(_entity_degree(bundle.data, entity_type))
    alpha = torch.cat(alphas, dim=0)
    degree = torch.cat(degrees, dim=0).float()
    argmax = alpha.argmax(dim=1)
    match = argmax == target_depth
    n_deviate = int((~match).sum().item())
    return {
        "n_nodes": alpha.size(0),
        "match_rate": match.float().mean().item(),
        "n_deviate": n_deviate,
        "mean_degree_match": degree[match].mean().item() if match.any() else float("nan"),
        "mean_degree_deviate": degree[~match].mean().item() if n_deviate > 0 else float("nan"),
        "mean_alpha": alpha.mean(dim=0).tolist(),
        "std_alpha": alpha.std(dim=0).tolist(),
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
    print(f"  [{elapsed:6.1f}s] {model_version:32s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)

    gate_stats = None
    if hasattr(model, "_last_gate_weights"):
        gate_stats = {task: _per_node_gate_stats(model, test_b, task) for task in TASKS}
        for task in TASKS:
            gs = gate_stats[task]
            print(f"      {task:10s} match_rate={gs['match_rate']:.4f} (n={gs['n_nodes']}, "
                  f"deviate={gs['n_deviate']})  mean_alpha={[round(a, 4) for a in gs['mean_alpha']]}  "
                  f"std_alpha={[round(a, 4) for a in gs['std_alpha']]}  "
                  f"deg_match={gs['mean_degree_match']:.2f} deg_deviate={gs['mean_degree_deviate']:.2f}",
                  flush=True)
    return aucs, preds, elapsed, n_params, gate_stats


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

    auc = {a: {} for a in ARMS}
    preds = {a: {} for a in ARMS}
    params = {a: {} for a in ARMS}
    gate_stats = {a: {} for a in ARMS}  # only populated for rung4/rung5
    run_elapsed = []

    for arch in ARMS:
        print(f"\n=== {arch}: 5 seeds ===", flush=True)
        for seed in SEEDS:
            model_version = f"{arch}-rungpilot-seed{seed}"
            purpose = (f"Step 6 Phase 2 pilot -- three-way markov/rung4/rung5 comparison, "
                       f"architecture={arch}, seed={seed}")
            aucs, run_preds, elapsed, n_params, gs = _train_eval_log(
                conn, model_version, arch, train_b, val_b, test_b, seed, purpose)
            auc[arch][seed] = aucs
            preds[arch][seed] = run_preds
            params[arch][seed] = n_params
            if gs is not None:
                gate_stats[arch][seed] = gs
            run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 15 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- AUC table
    print("=" * 95)
    print("AUC spread across 5 seeds, per (arm, task)")
    print("=" * 95)
    for arch in ARMS:
        for task in TASKS:
            values = [auc[arch][s][task] for s in SEEDS]
            print(f"  {ARM_LABEL[arch]:8s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v, 4) for v in values]}")

    print("\n" + "=" * 95)
    print("Parameter counts, per (arm, seed) -- should be constant within an arm")
    print("=" * 95)
    for arch in ARMS:
        print(f"  {ARM_LABEL[arch]:8s} {[params[arch][s] for s in SEEDS]}")

    # --------------------------------------------------------------- pairwise paired bootstrap
    print("\n" + "=" * 95)
    print("Paired bootstrap + sign-consistency, pairwise (all trained in-process this round)")
    print("=" * 95)
    pairs = [("rgcn_attn_rung4", "rgcn_attn_markov"), ("rgcn_attn_rung5", "rgcn_attn_markov"),
             ("rgcn_attn_rung4", "rgcn_attn_rung5")]
    for arm_a, arm_b in pairs:
        print(f"\n  -- {ARM_LABEL[arm_a]} vs {ARM_LABEL[arm_b]} --")
        for task in TASKS:
            deltas = []
            for seed in SEEDS:
                yt, p_a = preds[arm_a][seed][task]
                _, p_b = preds[arm_b][seed][task]
                cmp = paired_delta_auc_ci(yt, p_a, p_b)
                deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
            verdict = _sign_consistency(deltas)
            print(f"    {task:10s} per-seed dAUC={[round(d, 4) for d in deltas]}  "
                  f"mean={statistics.fmean(deltas):+.4f}  -> {verdict}", flush=True)

    # --------------------------------------------------------------- per-node gate analysis
    print("\n" + "=" * 95)
    print("Per-node gate analysis (Rung 4 / Rung 5) -- the actual point of building a per-node hybrid")
    print("=" * 95)
    for arch in ["rgcn_attn_rung4", "rgcn_attn_rung5"]:
        print(f"\n  -- {ARM_LABEL[arch]} --")
        for task in TASKS:
            match_rates = [gate_stats[arch][s][task]["match_rate"] for s in SEEDS]
            n_deviates = [gate_stats[arch][s][task]["n_deviate"] for s in SEEDS]
            n_nodes = [gate_stats[arch][s][task]["n_nodes"] for s in SEEDS]
            mean_alphas = [gate_stats[arch][s][task]["mean_alpha"] for s in SEEDS]
            std_alphas = [gate_stats[arch][s][task]["std_alpha"] for s in SEEDS]
            n_depths = len(mean_alphas[0])
            avg_mean_alpha = [statistics.fmean(m[d] for m in mean_alphas) for d in range(n_depths)]
            avg_std_alpha = [statistics.fmean(s[d] for s in std_alphas) for d in range(n_depths)]

            print(f"    {task:10s} n_nodes/seed={n_nodes}")
            print(f"               match_rate mean={statistics.fmean(match_rates):.4f}  "
                  f"per-seed={[round(m, 4) for m in match_rates]}")
            print(f"               n_deviate per-seed={n_deviates}")
            print(f"               mean_alpha (avg across seeds, h^0..h^4)={[round(a, 4) for a in avg_mean_alpha]}")
            print(f"               std_alpha  (avg across seeds, per-node spread)={[round(a, 4) for a in avg_std_alpha]}")

            deg_pairs = []
            for s in SEEDS:
                gs = gate_stats[arch][s][task]
                dm, dd = gs["mean_degree_match"], gs["mean_degree_deviate"]
                if gs["n_deviate"] > 0 and not math.isnan(dd):
                    deg_pairs.append((s, dm, dd, dd - dm))
            if deg_pairs:
                diffs = [p[3] for p in deg_pairs]
                verdict = _sign_consistency(diffs) if len(diffs) > 1 else "n/a (only 1 seed w/ deviates)"
                print(f"               degree(deviate)-degree(match), seeds with deviates="
                      f"{[(p[0], round(p[3], 2)) for p in deg_pairs]}  mean={statistics.fmean(diffs):+.2f}  -> {verdict}")
            else:
                print("               no deviating nodes in ANY seed -- gate stayed pinned at Markov's depth for every node")

    # --------------------------------------------------------------- verdict
    print("\n" + "=" * 95)
    print("Verdict -- best arm per task, and whether the per-node hybrids found real deviation")
    print("=" * 95)
    for task in TASKS:
        means = {arch: statistics.fmean(auc[arch][s][task] for s in SEEDS) for arch in ARMS}
        best_arch = max(means, key=means.get)
        print(f"  {task:10s} means: " + "  ".join(f"{ARM_LABEL[a]}={means[a]:.4f}" for a in ARMS) +
              f"  -> best (point estimate): {ARM_LABEL[best_arch]}")

    print("\n" + "=" * 95)
    print("Per-run elapsed time")
    print("=" * 95)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:36s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
