"""
Step 6 Phase 2 -- five INDEPENDENT isolated-variant follow-ups to the original Rung 5 gate
(`ml/models/rgcn_attn_rung5_encoder.py`, `reports/step6_rung_pilot.md`), each testing
exactly ONE proposed fix in isolation against the SAME shared baselines -- never stacked
on top of each other. Motivated by that report's headline finding: both per-node gates
(Rung 4, Rung 5) showed wildly seed-dependent, bimodal stability (some seeds stayed pinned
exactly at the Markov prior, others let 75-98% of nodes drift to a different depth), with
no AUC difference distinguishing the two regimes.

    Variant A (`ml/models/rgcn_attn_rung5_variant_a.py`) -- fixed (non-learned) position
        bias + tanh-bounded residual, scaled by lambda in {0.1, 0.3, 0.5}. Tests: does
        making drift structurally impossible beyond a hard ceiling fix the instability?
    Variant B (`ml/models/rgcn_attn_rung5_variant_b.py`) -- structural features (degree,
        node-type one-hot, per-relation edge-count histogram) concatenated onto the
        pooled embedding. Tests: does giving the gate explicit structural signal, instead
        of making it infer position from embeddings alone, change its behaviour?
    Variant C (training-loop-only change in `ml/train.py::train_model`, architecture id
        `rgcn_attn_rung5_c`, same model class as the original) -- gate frozen for the
        first 15 epochs, then unfrozen at 0.1x the main LR. Tests: does letting the
        encoder settle first, before the gate can move the readout, help?
    Variant D (`ml/models/rgcn_attn_rung5_variant_d.py`) -- shared per-depth linear
        projection (128->8) applied to each of h^0..h^4 individually and concatenated,
        replacing mean-pooling. Tests: does preserving depth identity (lost by averaging)
        change the gate's decisions?
    Variant E (post-hoc, no training here) -- weight-average the 5 already-trained
        original-Rung-5 seeds' full parameters into one model, evaluate directly. Tests:
        does averaging away seed-to-seed variance produce a more stable, better gate than
        any individual seed?

===============================================================================================
DEVICE CHECK (verified before writing this script, not assumed)
===============================================================================================
1. Correctness: `torch_geometric.utils.scatter`/`softmax` (the exact ops
   `RGCNAttnEncoder`'s joint-softmax attention and the Rung 5 gate's einsum-blend rely on),
   run forward+backward on a synthetic batch on CPU vs MPS: outputs and gradients matched
   to `atol=1e-4` (max observed diff ~2.4e-7). `torch.bincount` (Variant B's structural
   features, and the per-node degree analysis) also verified working correctly on MPS.
2. Speed: 5 real training iterations (forward+backward+optimizer.step, real dataset, real
   Rung5HADESModel) timed 0.316s/iter on CPU vs 0.124s/iter on MPS -- a genuine ~2.5x
   speedup, not measurement noise.
3. End-to-end: a full 5-epoch `run_training_job(..., device="mps")` call, including the
   `.cpu().numpy()` fix required in `ml/evaluate.py::collect_predictions` and
   `ml/train.py::_mean_val_auc` (both previously assumed CPU-resident tensors; the fix is
   a no-op on CPU, so every other architecture's behaviour is unaffected), completed with
   a real DB write in 9.9s (~2s/epoch, vs CPU's observed ~3.5-4.5s/epoch for this model
   class) -- confirmed against a throwaway model_version, deleted immediately after.

Both criteria (correctness AND speedup) were met, so **this script trains on MPS**
(`DEVICE = "mps"` below). `ml/train.py::move_bundles_to_device` moves every bundle once,
up front, so the transfer cost is paid exactly once for all 40+ runs, not once per run.

Run: python -u -m ml.run_rung5_variant_pilot
"""

from __future__ import annotations

import copy
import math
import statistics
import time

import torch

from ml.data.db import get_connection
from ml.evaluate import (
    best_f1_threshold,
    collect_predictions,
    evaluate_split,
    log_evaluation_runs,
    paired_delta_auc_ci,
)
from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    activate_model,
    load_all_snapshot_bundles,
    move_bundles_to_device,
    register_model,
    run_training_job,
    split_bundles,
)

SEEDS = [0, 1, 2, 3, 4]
HIDDEN, NUM_BASES = 128, 10
DEVICE = "mps"  # see module docstring for the verification that justified this
RUNG5A_LAMBDAS = [0.1, 0.3, 0.5]

# (label, architecture, extra train_model kwargs). "rung5" and "markov" are the shared
# baselines every variant is compared against -- both retrained fresh in this same
# process, never pulled from `reports/step6_rung_pilot.md`'s already-logged numbers.
PHASES: list[tuple[str, str, dict]] = (
    [("markov", "rgcn_attn_markov", {}), ("rung5", "rgcn_attn_rung5", {})]
    + [(f"rung5_a_lam{lam}", "rgcn_attn_rung5_a", {"rung5a_lambda_bound": lam}) for lam in RUNG5A_LAMBDAS]
    + [("rung5_b", "rgcn_attn_rung5_b", {}), ("rung5_c", "rgcn_attn_rung5_c", {}), ("rung5_d", "rgcn_attn_rung5_d", {})]
)

GATED_ARCHS = {"rgcn_attn_rung5", "rgcn_attn_rung5_a", "rgcn_attn_rung5_b", "rgcn_attn_rung5_c", "rgcn_attn_rung5_d"}


def _model_version(label: str, arch: str, seed: int) -> str:
    if label.startswith("rung5_a_lam"):
        lam = label.replace("rung5_a_lam", "")
        return f"rgcn_attn_rung5_a-lam{lam}-seed{seed}"
    return f"{arch}-r5v-seed{seed}"


def _entity_degree(data, entity_type: str) -> torch.Tensor:
    """Same convention as `ml/run_rung_pilot.py::_entity_degree` -- total degree touching
    each node of `entity_type`, summed dst-side across every relation (every edge has a
    reverse counterpart under `ToUndirected()`, so this double-counts nothing)."""
    n = data[entity_type].x.size(0)
    degree = torch.zeros(n, dtype=torch.long, device=data[entity_type].x.device)
    for (_src, _rel, dst), edge_index in data.edge_index_dict.items():
        if dst == entity_type and edge_index.numel() > 0:
            degree += torch.bincount(edge_index[1], minlength=n)
    return degree


@torch.no_grad()
def _per_node_gate_stats(model, test_bundles, task: str) -> dict:
    """Identical contract to `ml/run_rung_pilot.py::_per_node_gate_stats`, so stability is
    measured the SAME way across every variant here and is directly comparable to
    `reports/step6_rung_pilot.md`'s original Rung 4/Rung 5 numbers."""
    entity_type = TASK_ENTITY_TYPE[task]
    target_depth = MARKOV_READOUT_DEPTH[task]
    model.eval()
    alphas, degrees = [], []
    for bundle in test_bundles:
        model(bundle.data.x_dict, bundle.data.edge_index_dict)
        alphas.append(model._last_gate_weights[task].cpu())
        degrees.append(_entity_degree(bundle.data, entity_type).cpu())
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


def _train_eval_log(conn, model_version, architecture, train_b, val_b, test_b, seed, purpose, extra):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, architecture, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES, device=DEVICE, **extra,
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
    print(f"  [{elapsed:6.1f}s] {model_version:34s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)

    gate_stats = None
    if architecture in GATED_ARCHS:
        gate_stats = {task: _per_node_gate_stats(model, test_b, task) for task in TASKS}
        for task in TASKS:
            gs = gate_stats[task]
            print(f"      {task:10s} match_rate={gs['match_rate']:.4f} (n={gs['n_nodes']}, "
                  f"deviate={gs['n_deviate']})  mean_alpha={[round(a, 4) for a in gs['mean_alpha']]}",
                  flush=True)
    return aucs, preds, elapsed, n_params, gate_stats, model


def _sign_consistency(deltas: list[float]) -> str:
    signs = {d > 0 for d in deltas}
    return "CONSISTENT" if len(signs) == 1 else "FLIPS (noise)"


def _paired_vs(preds_a, preds_b, seeds_a, seeds_b, task):
    """Paired bootstrap dAUC(a - b) for every (seed_a, seed_b) pairing when a and b have
    matched seed lists (the normal 5-vs-5 case); when `seeds_a`/`seeds_b` differ in length
    (Variant E's single averaged model vs 5 baseline seeds), pairs index i of the SHORTER
    list 1:1 against the corresponding index of the longer one is wrong -- instead every
    entry of the shorter list is compared against every entry of the longer one."""
    deltas = []
    for sa in seeds_a:
        for sb in seeds_b:
            yt, pa = preds_a[sa][task]
            _, pb = preds_b[sb][task]
            cmp = paired_delta_auc_ci(yt, pa, pb)
            deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
    return deltas


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM suppliers")
        n_suppliers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM graph_snapshots")
        n_snapshots = cur.fetchone()[0]
    print(f"=== Live dataset check: {n_suppliers} suppliers, {n_snapshots} graph_snapshots ===", flush=True)
    print(f"=== Device: {DEVICE} (correctness + ~2.5x speedup verified, see module docstring) ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    move_bundles_to_device(train_b, DEVICE)
    move_bundles_to_device(val_b, DEVICE)
    move_bundles_to_device(test_b, DEVICE)
    print(f"Bundles moved to {DEVICE}.", flush=True)

    auc: dict[str, dict[int, dict]] = {}
    preds: dict[str, dict[int, dict]] = {}
    params: dict[str, dict[int, int]] = {}
    gate_stats: dict[str, dict[int, dict]] = {}
    run_elapsed = []
    rung5_state_dicts = []  # for Variant E, captured on CPU

    for label, arch, extra in PHASES:
        print(f"\n=== {label} ({arch}{', ' + str(extra) if extra else ''}): 5 seeds ===", flush=True)
        auc[label], preds[label], params[label], gate_stats[label] = {}, {}, {}, {}
        for seed in SEEDS:
            model_version = _model_version(label, arch, seed)
            purpose = f"Step 6 Rung 5 variant pilot -- {label}, seed={seed}, device={DEVICE}"
            aucs, run_preds, elapsed, n_params, gs, model = _train_eval_log(
                conn, model_version, arch, train_b, val_b, test_b, seed, purpose, extra)
            auc[label][seed] = aucs
            preds[label][seed] = run_preds
            params[label][seed] = n_params
            if gs is not None:
                gate_stats[label][seed] = gs
            run_elapsed.append((model_version, elapsed))
            if label == "rung5":
                rung5_state_dicts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # --------------------------------------------------------------- Variant E: post-hoc weight averaging
    print("\n=== Variant E: post-hoc weight-averaging of the 5 rung5 seeds (no training) ===", flush=True)
    from ml.models.rgcn_attn_rung5_encoder import Rung5HADESModel

    avg_state = {
        k: torch.stack([sd[k].float() for sd in rung5_state_dicts], dim=0).mean(dim=0).to(rung5_state_dicts[0][k].dtype)
        for k in rung5_state_dicts[0]
    }
    metadata = train_b[0].data.metadata()
    in_dims = {nt: train_b[0].data[nt].x.size(-1) for nt in train_b[0].data.node_types}
    avg_model = Rung5HADESModel(metadata, in_dims, hidden=HIDDEN, num_layers=4, num_bases=NUM_BASES).to(DEVICE)
    avg_model.load_state_dict(avg_state)

    thresholds_e = {task: best_f1_threshold(*collect_predictions(avg_model, val_b)[task]) for task in TASKS}
    results_e = evaluate_split(avg_model, test_b, thresholds_e)
    model_version_e = "rgcn_attn_rung5-weightavg5seeds"
    hyperparams_e = {
        "architecture": "rgcn_attn_rung5", "hidden": HIDDEN, "num_bases": NUM_BASES, "num_layers": 4,
        "device": DEVICE,
        "source": ("post-hoc weight average of 5 independently trained rgcn_attn_rung5 seeds "
                   "(seeds 0-4, this same pass) -- NOT independently trained; averaging is over "
                   "the FULL model (encoder+gates+heads), not gate parameters alone, since "
                   "evaluating a gate in isolation requires an encoder to sit under it and "
                   "per-seed encoders differ (see module docstring / report for the full "
                   "interpretation note)"),
    }
    register_model(conn, model_version_e, "rgcn_attn_rung5", hyperparams_e,
                   avg_model.parameter_count(), "Step 6 Rung 5 variant pilot -- Variant E, weight-averaging validation")
    log_evaluation_runs(conn, model_version_e, "rgcn_attn_rung5", results_e, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    activate_model(conn, model_version_e, "active")

    auc_e = {task: results_e.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    preds_e = {task: collect_predictions(avg_model, test_b)[task] for task in TASKS}
    gate_stats_e = {task: _per_node_gate_stats(avg_model, test_b, task) for task in TASKS}
    print(f"  params={avg_model.parameter_count():,}  " + "  ".join(f"{t}={auc_e[t]:.4f}" for t in TASKS), flush=True)
    for task in TASKS:
        gs = gate_stats_e[task]
        print(f"      {task:10s} match_rate={gs['match_rate']:.4f} (n={gs['n_nodes']}, deviate={gs['n_deviate']})  "
              f"mean_alpha={[round(a, 4) for a in gs['mean_alpha']]}", flush=True)

    total_elapsed = time.monotonic() - overall_start
    n_runs = sum(len(SEEDS) for _ in PHASES)
    print(f"\n=== ALL {n_runs} TRAINING RUNS + VARIANT E COMPLETE in {total_elapsed/3600:.2f}h "
          f"({total_elapsed:.0f}s) ===\n", flush=True)

    # =============================================================== REPORTING
    print("=" * 100)
    print("AUC spread across 5 seeds, per (phase, task)")
    print("=" * 100)
    for label, _arch, _extra in PHASES:
        for task in TASKS:
            values = [auc[label][s][task] for s in SEEDS]
            print(f"  {label:16s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v, 4) for v in values]}")
    print(f"\n  {'variant_e':16s} (single averaged model, no seed spread)")
    for task in TASKS:
        print(f"    {task:10s} auc={auc_e[task]:.4f}")

    # ---------------------------------------------------------------- pairwise paired bootstrap
    print("\n" + "=" * 100)
    print("Paired bootstrap + sign-consistency -- EACH variant vs the SAME shared markov/rung5 baselines")
    print("(never variant vs variant)")
    print("=" * 100)
    variant_labels = [lbl for lbl, _a, _e in PHASES if lbl not in ("markov", "rung5")]
    for label in variant_labels:
        print(f"\n  -- {label} vs markov --")
        for task in TASKS:
            deltas = _paired_vs(preds[label], preds["markov"], SEEDS, SEEDS, task)
            print(f"    {task:10s} mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}  "
                  f"(n={len(deltas)} seed-pairs)")
        print(f"  -- {label} vs rung5 (original) --")
        for task in TASKS:
            deltas = _paired_vs(preds[label], preds["rung5"], SEEDS, SEEDS, task)
            print(f"    {task:10s} mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}  "
                  f"(n={len(deltas)} seed-pairs)")

    print("\n  -- variant_e (weight-avg) vs markov --")
    for task in TASKS:
        deltas = _paired_vs({0: preds_e}, preds["markov"], [0], SEEDS, task)
        print(f"    {task:10s} mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}  "
              f"(n={len(deltas)}, vs each of the 5 markov seeds)")
    print("  -- variant_e (weight-avg) vs rung5 (original, all 5 individual seeds) --")
    for task in TASKS:
        deltas = _paired_vs({0: preds_e}, preds["rung5"], [0], SEEDS, task)
        print(f"    {task:10s} mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}  "
              f"(n={len(deltas)}, vs each of the 5 rung5 seeds)")
    print("  -- variant_e (weight-avg) vs BEST individual rung5 seed, per task --")
    for task in TASKS:
        best_seed = max(SEEDS, key=lambda s: auc["rung5"][s][task])
        yt, p_e = preds_e[task]
        _, p_best = preds["rung5"][best_seed][task]
        cmp = paired_delta_auc_ci(yt, p_e, p_best)
        print(f"    {task:10s} best_seed={best_seed} (auc={auc['rung5'][best_seed][task]:.4f})  "
              f"variant_e_auc={auc_e[task]:.4f}  delta={cmp['delta']:+.4f}  "
              f"ci=[{cmp['ci_lower']:+.4f}, {cmp['ci_upper']:+.4f}]  significant={cmp['significant']}")

    # ---------------------------------------------------------------- per-node gate analysis
    print("\n" + "=" * 100)
    print("Per-node gate analysis, identical methodology to reports/step6_rung_pilot.md")
    print("=" * 100)
    for label in [lbl for lbl, arch, _e in PHASES if arch in GATED_ARCHS]:
        print(f"\n  -- {label} --")
        for task in TASKS:
            match_rates = [gate_stats[label][s][task]["match_rate"] for s in SEEDS]
            n_deviates = [gate_stats[label][s][task]["n_deviate"] for s in SEEDS]
            print(f"    {task:10s} match_rate mean={statistics.fmean(match_rates):.4f}  "
                  f"per-seed={[round(m, 4) for m in match_rates]}  n_deviate per-seed={n_deviates}")
            deg_pairs = []
            for s in SEEDS:
                gs = gate_stats[label][s][task]
                dm, dd = gs["mean_degree_match"], gs["mean_degree_deviate"]
                if gs["n_deviate"] > 0 and not math.isnan(dd):
                    deg_pairs.append((s, dd - dm))
            if deg_pairs:
                diffs = [p[1] for p in deg_pairs]
                verdict = _sign_consistency(diffs) if len(diffs) > 1 else "n/a (1 seed w/ deviates)"
                print(f"               degree(deviate)-degree(match): {[(p[0], round(p[1], 2)) for p in deg_pairs]}  "
                      f"mean={statistics.fmean(diffs):+.2f}  -> {verdict}")
            else:
                print("               no deviating nodes in ANY seed -- gate stayed pinned at Markov's depth")

    print("\n  -- variant_e (weight-avg) --")
    for task in TASKS:
        gs = gate_stats_e[task]
        print(f"    {task:10s} match_rate={gs['match_rate']:.4f}  n_deviate={gs['n_deviate']}  "
              f"deg_match={gs['mean_degree_match']:.2f}  deg_deviate={gs['mean_degree_deviate']:.2f}")

    # ---------------------------------------------------------------- lambda sensitivity (variant A)
    print("\n" + "=" * 100)
    print("Variant A lambda sensitivity (0.1 / 0.3 / 0.5)")
    print("=" * 100)
    for task in TASKS:
        print(f"\n  -- {task} --")
        for lam in RUNG5A_LAMBDAS:
            label = f"rung5_a_lam{lam}"
            auc_mean = statistics.fmean(auc[label][s][task] for s in SEEDS)
            match_mean = statistics.fmean(gate_stats[label][s][task]["match_rate"] for s in SEEDS)
            print(f"    lambda={lam:<4} auc_mean={auc_mean:.4f}  match_rate_mean={match_mean:.4f}")

    # ---------------------------------------------------------------- summary table + verdict
    print("\n" + "=" * 100)
    print("SUMMARY TABLE -- markov, rung5 (original), variants A(lam0.3)/B/C/D/E")
    print("=" * 100)
    summary_rows = [("markov", "markov"), ("rung5", "rung5 (original)"), ("rung5_a_lam0.3", "variant A (lam=0.3)"),
                     ("rung5_b", "variant B"), ("rung5_c", "variant C"), ("rung5_d", "variant D")]
    for label, disp in summary_rows:
        aucs_str = "  ".join(f"{t}={statistics.fmean(auc[label][s][t] for s in SEEDS):.4f}" for t in TASKS)
        if label in gate_stats and gate_stats[label]:
            match_str = "  ".join(
                f"{t}_match={statistics.fmean(gate_stats[label][s][t]['match_rate'] for s in SEEDS):.3f}"
                for t in TASKS)
        else:
            match_str = "(fixed depth, no gate)"
        print(f"  {disp:22s} {aucs_str}   {match_str}")
    aucs_str_e = "  ".join(f"{t}={auc_e[t]:.4f}" for t in TASKS)
    match_str_e = "  ".join(f"{t}_match={gate_stats_e[t]['match_rate']:.3f}" for t in TASKS)
    print(f"  {'variant E (weight-avg)':22s} {aucs_str_e}   {match_str_e}")

    print("\n" + "=" * 100)
    print("Per-run elapsed time")
    print("=" * 100)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:38s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
