"""
Step 7 — Transformer 2 (Claim 3, global same-type attention) on top of the finalized
Layer 1 + Layer 2 stack: SHARE (RGCN+Attention, `rgcn_attn`) as the encoder, Variant A
(bounded residual on the Markov floor, `reports/rung5_types.md` Type A) as the
depth-selection mechanism. A new, purely additive component
(`ml/models/transformer2.py`, `ml/models/rgcn_attn_variant_a_transformer2.py`) -- no
changes to SHARE's encoder, SHARP, SHARK, or Variant A's own gate.

**DEVIATION FROM `10_AI_ML_Documentation.md` §8.3, flagged here and in the model's own
docstring.** §8.3 wires Transformer 2 to the delay head only. This pilot wires it to
IMPACT instead, per Step 6 Round 1's reach analysis (`reports/layer_2.md`): the
co-parent/hidden-dependency path needs >=3 hops from delay's target (Shipment),
beyond what delay ever reads (`h^1` under Variant A); impact's target (Supplier, hop
0) has real co-parent signal from hop 2 onward, and impact reads `h^4`.

**Device.** MPS -- correctness verified (topk/gather/attention forward+backward on a
synthetic batch, CPU vs MPS, exact match) and a real (if modest, ~1.3x for this
component alone; the encoder's own ~2.5x dominates total wall-clock) speedup measured
before choosing it, same protocol as every recent round.

**Validation -- the actual pass/fail gate.** `ml/graph/hidden_dependency_ground_truth.py`
holds the 4 planted H_POLYMER supplier IDs (reconstructed read-only from
`db/csv/components.csv`'s generation-order rows, cross-checked against
`db/README.md`'s own published table -- exact match). This script checks, per trained
seed, whether Transformer 2's discovered candidate pool contains the 6 H_POLYMER
pairs at a rate better than a concrete chance baseline: (1) the GLOBAL discovery rate
-- what fraction of ALL C(800,2)=319,600 possible supplier pairs land in at least one
endpoint's top-64 pool, purely from the pool-size mechanics, vs. H_POLYMER's own
discovery rate; (2) each pair's percentile rank among ALL pairs' cosine similarity.
Transformer 2 is adopted only if this clears the bar; otherwise this is reported as a
plain negative result, per this project's standing practice.

**`hidden_dependency_links` logging -- bounded, not literal "every pair."** Logging
every discovered pair across all 5 seeds x 6 test snapshots would be
5 x 6 x up to ~25,600 unique pairs/snapshot -- not a reviewable "investigation leads"
list per `05_Database_Design.md` §6.25's own framing ("a row is an investigation
lead, never a fact"). This script logs the top-500 pairs by symmetric_score PLUS the
6 H_POLYMER pairs unconditionally (even if they don't clear that cut), for the most
recent test snapshot only, across all 5 seeds -- a disclosed bounding decision, not a
silent one.

Run: python -u -m ml.run_transformer2_pilot
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs, paired_delta_auc_ci
from ml.graph.hidden_dependency_ground_truth import h_polymer_pairs
from ml.models.depth import TASKS
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    load_all_snapshot_bundles,
    move_bundles_to_device,
    run_training_job,
    split_bundles,
)

SEEDS = [0, 1, 2, 3, 4]
HIDDEN, NUM_BASES = 128, 10
DEVICE = "mps"
LAMBDA_BOUND = 0.3
T2_TOP_K = 64
LOG_TOP_PAIRS = 500
T2_ARCH = "rgcn_attn_variant_a_transformer2"
BASELINE_ARCH = "rgcn_attn_rung5_a"


@torch.no_grad()
def _hidden_dependency_analysis(model, bundle) -> dict:
    """Percentile rank + pool-discovery status for each H_POLYMER pair, plus the
    global chance-baseline discovery rate, using the model's own most recent forward
    pass (`model._last_z_impact`, `model._last_t2`)."""
    node_ids = bundle.data["Supplier"].node_id
    uuid_to_idx = {u: i for i, u in enumerate(node_ids)}
    n = len(node_ids)

    z = model._last_z_impact
    normed = F.normalize(z, dim=-1)
    cos_sim = (normed @ normed.T).clone()
    cos_sim.fill_diagonal_(float("-inf"))

    idx, alpha = model._last_t2  # [N, k] each
    idx_cpu, alpha_cpu = idx.cpu(), alpha.cpu()

    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    discovered = torch.zeros(n, n, dtype=torch.bool)
    rows = torch.arange(n).unsqueeze(1).expand(-1, idx_cpu.size(1))
    discovered[rows.reshape(-1), idx_cpu.reshape(-1)] = True
    discovered = discovered | discovered.T
    total_pairs = int(upper.sum().item())
    global_discovery_rate = float((discovered & upper).sum().item()) / total_pairs

    cos_cpu = cos_sim.cpu()
    upper_vals, _ = torch.sort(cos_cpu[upper])

    pairs = []
    for a_id, b_id in h_polymer_pairs():
        if a_id not in uuid_to_idx or b_id not in uuid_to_idx:
            continue
        i, j = uuid_to_idx[a_id], uuid_to_idx[b_id]
        score = max(cos_cpu[i, j].item(), cos_cpu[j, i].item())
        pct_rank = float(torch.searchsorted(upper_vals, torch.tensor(score)).item()) / total_pairs

        pool_i, pool_j = idx_cpu[i].tolist(), idx_cpu[j].tolist()
        att_i_to_j = alpha_cpu[i, pool_i.index(j)].item() if j in pool_i else 0.0
        att_j_to_i = alpha_cpu[j, pool_j.index(i)].item() if i in pool_j else 0.0
        pairs.append({
            "supplier_a": a_id, "supplier_b": b_id, "cos_sim": score,
            "percentile_rank": pct_rank, "attention_a_to_b": att_i_to_j,
            "attention_b_to_a": att_j_to_i, "symmetric_score": (att_i_to_j + att_j_to_i) / 2,
            "discovered": (j in pool_i) or (i in pool_j),
        })
    return {"pairs": pairs, "global_discovery_rate": global_discovery_rate,
            "n_suppliers": n, "total_pairs": total_pairs}


def _build_pair_records(node_ids: list[str], idx: torch.Tensor, alpha: torch.Tensor) -> dict[tuple[int, int], list[float]]:
    """Directed (src, dst)->weight entries from `idx`/`alpha` collapsed into canonical
    (a<b) unordered pairs with both directional weights (0.0 for a missing direction).
    """
    n, k = idx.shape
    src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1).tolist()
    dst = idx.reshape(-1).tolist()
    w = alpha.reshape(-1).tolist()
    pairs: dict[tuple[int, int], list[float]] = {}
    for s, d, wt in zip(src, dst, w):
        a, b = (s, d) if s < d else (d, s)
        rec = pairs.setdefault((a, b), [0.0, 0.0])
        if s == a:
            rec[0] = wt  # a -> b
        else:
            rec[1] = wt  # b -> a
    return pairs


def _log_hidden_dependency_links(conn, model, bundle, model_version: str, h_polymer_set: set[tuple[str, str]]) -> int:
    node_ids = bundle.data["Supplier"].node_id
    idx, alpha = model._last_t2
    idx_cpu, alpha_cpu = idx.cpu(), alpha.cpu()
    pairs = _build_pair_records(node_ids, idx_cpu, alpha_cpu)

    scored = []
    for (a, b), (a_to_b, b_to_a) in pairs.items():
        sym = (a_to_b + b_to_a) / 2
        a_id, b_id = node_ids[a], node_ids[b]
        canon_a, canon_b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        if canon_a != a_id:
            a_to_b, b_to_a = b_to_a, a_to_b
        scored.append((sym, canon_a, canon_b, a_to_b, b_to_a))
    scored.sort(key=lambda r: r[0], reverse=True)

    top = scored[:LOG_TOP_PAIRS]
    top_pairs = {(r[1], r[2]) for r in top}
    for sym, a_id, b_id, a_to_b, b_to_a in scored:
        if (a_id, b_id) in h_polymer_set and (a_id, b_id) not in top_pairs:
            top.append((sym, a_id, b_id, a_to_b, b_to_a))

    rows = []
    for sym, a_id, b_id, a_to_b, b_to_a in top:
        status = "confirmed" if (a_id, b_id) in h_polymer_set else "unvalidated"
        rows.append((a_id, b_id, round(a_to_b, 5), round(b_to_a, 5), round(sym, 5),
                     bundle.t0, f"topk{T2_TOP_K}_cosine", model_version, status))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO hidden_dependency_links
                (supplier_a_id, supplier_b_id, attention_a_to_b, attention_b_to_a,
                 symmetric_score, snapshot_t0, candidate_pool_version, model_version,
                 validation_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (supplier_a_id, supplier_b_id, snapshot_t0, model_version) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def _train_eval_log(conn, model_version, architecture, train_b, val_b, test_b, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, architecture, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=HIDDEN, epochs=100,
        seed=seed, purpose=purpose, num_bases=NUM_BASES, device=DEVICE,
        rung5a_lambda_bound=LAMBDA_BOUND, t2_top_k=T2_TOP_K,
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
    print(f"  [{elapsed:6.1f}s] {model_version:38s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, preds, elapsed, model


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
    print(f"=== Device: {DEVICE} (correctness verified, see module docstring) ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    move_bundles_to_device(train_b, DEVICE)
    move_bundles_to_device(val_b, DEVICE)
    move_bundles_to_device(test_b, DEVICE)

    h_polymer_set = set(h_polymer_pairs())
    print(f"H_POLYMER ground-truth pairs (planted hidden-dependency scenario): {sorted(h_polymer_set)}", flush=True)

    auc = {"t2": {}, "baseline": {}}
    preds = {"t2": {}, "baseline": {}}
    hd_analysis = {}  # seed -> list of per-snapshot analysis dicts
    run_elapsed = []
    t2_models = {}

    print("\n=== Baseline (Variant A alone, NO Transformer 2): 5 seeds ===", flush=True)
    for seed in SEEDS:
        model_version = f"{BASELINE_ARCH}-t2pilot-baselinecmp-seed{seed}"
        purpose = f"Step 7 Transformer 2 pilot -- Variant A baseline, retrained fresh, seed={seed}"
        aucs, run_preds, elapsed, _model = _train_eval_log(
            conn, model_version, BASELINE_ARCH, train_b, val_b, test_b, seed, purpose)
        auc["baseline"][seed] = aucs
        preds["baseline"][seed] = run_preds
        run_elapsed.append((model_version, elapsed))

    print("\n=== Variant A + Transformer 2 (fused into impact): 5 seeds ===", flush=True)
    for seed in SEEDS:
        model_version = f"{T2_ARCH}-t2pilot-seed{seed}"
        purpose = f"Step 7 Transformer 2 pilot -- SHARE+VariantA+Transformer2 (impact-wired), seed={seed}"
        aucs, run_preds, elapsed, model = _train_eval_log(
            conn, model_version, T2_ARCH, train_b, val_b, test_b, seed, purpose)
        auc["t2"][seed] = aucs
        preds["t2"][seed] = run_preds
        run_elapsed.append((model_version, elapsed))
        t2_models[seed] = model

        # hidden-dependency validation across all test snapshots
        per_snapshot = []
        for bundle in test_b:
            model(bundle.data.x_dict, bundle.data.edge_index_dict)
            per_snapshot.append(_hidden_dependency_analysis(model, bundle))
        hd_analysis[seed] = per_snapshot

        # log leads to hidden_dependency_links -- most recent test snapshot only (see module docstring)
        last_bundle = test_b[-1]
        model(last_bundle.data.x_dict, last_bundle.data.edge_index_dict)
        n_logged = _log_hidden_dependency_links(conn, model, last_bundle, model_version, h_polymer_set)
        print(f"      logged {n_logged} hidden_dependency_links rows for snapshot {last_bundle.t0.date()}", flush=True)

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 10 TRAINING RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- AUC table
    print("=" * 100)
    print("AUC spread across 5 seeds, per (arm, task)")
    print("=" * 100)
    for arm in ["baseline", "t2"]:
        for task in TASKS:
            values = [auc[arm][s][task] for s in SEEDS]
            print(f"  {arm:10s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v, 4) for v in values]}")

    # --------------------------------------------------------------- paired bootstrap on impact
    print("\n" + "=" * 100)
    print("Paired bootstrap + sign-consistency -- Transformer2 vs baseline (Variant A alone), impact task")
    print("=" * 100)
    for task in TASKS:
        deltas = []
        for seed in SEEDS:
            yt, p_t2 = preds["t2"][seed][task]
            _, p_base = preds["baseline"][seed][task]
            cmp = paired_delta_auc_ci(yt, p_t2, p_base)
            deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
        verdict = _sign_consistency(deltas)
        print(f"    {task:10s} per-seed dAUC={[round(d, 4) for d in deltas]}  "
              f"mean={statistics.fmean(deltas):+.4f}  -> {verdict}", flush=True)

    # --------------------------------------------------------------- hidden-dependency validation
    print("\n" + "=" * 100)
    print("Hidden-dependency validation -- H_POLYMER recovery vs. chance, per seed")
    print("=" * 100)
    all_pct_ranks = []
    all_discovery_flags = []
    all_global_rates = []
    for seed in SEEDS:
        snapshots = hd_analysis[seed]
        seed_pct = [p["percentile_rank"] for snap in snapshots for p in snap["pairs"]]
        seed_disc = [p["discovered"] for snap in snapshots for p in snap["pairs"]]
        seed_global = [snap["global_discovery_rate"] for snap in snapshots]
        all_pct_ranks.extend(seed_pct)
        all_discovery_flags.extend(seed_disc)
        all_global_rates.extend(seed_global)
        print(f"  seed {seed}: mean H_POLYMER percentile_rank={statistics.fmean(seed_pct):.4f}  "
              f"discovery_rate={statistics.fmean(seed_disc):.4f} ({sum(seed_disc)}/{len(seed_disc)} pair-snapshots)  "
              f"global_chance_discovery_rate={statistics.fmean(seed_global):.4f}", flush=True)

    mean_pct_rank = statistics.fmean(all_pct_ranks)
    mean_discovery = statistics.fmean(all_discovery_flags)
    mean_global_rate = statistics.fmean(all_global_rates)
    chance_all6_prob = mean_global_rate ** 6

    print(f"\n  OVERALL (5 seeds x 6 test snapshots x 6 pairs = {len(all_pct_ranks)} pair-observations):")
    print(f"    mean H_POLYMER percentile_rank = {mean_pct_rank:.4f}  (chance expectation: 0.5000)")
    print(f"    mean H_POLYMER discovery_rate  = {mean_discovery:.4f}  "
          f"(global chance discovery rate    = {mean_global_rate:.4f})")
    print(f"    P(all 6 H_POLYMER pairs discovered by chance alone) ~= "
          f"global_rate^6 = {chance_all6_prob:.6f}  (crude independence approximation, reported as a "
          f"supporting stat, not the primary metric)", flush=True)

    pass_bar = mean_discovery > mean_global_rate and mean_pct_rank > 0.5
    print(f"\n  VALIDATION VERDICT: {'PASS -- recovered above chance' if pass_bar else 'FAIL -- not distinguishable from chance'}", flush=True)

    print("\n" + "=" * 100)
    print("Per-run elapsed time")
    print("=" * 100)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:46s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
