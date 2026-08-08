"""
Step 7b — three INDEPENDENT fusion-improvement variants for Transformer 2, each
testing exactly ONE isolated change against the SAME shared baselines — never
stacked. Build-and-evaluate only, scoped to SHARE + Variant A + Transformer 2
(`rgcn_attn_variant_a_transformer2`, `reports/step7_transformer2_pilot.md`) — no
changes to SHARP, SHARK, or the existing Markov/Variant A depth-selection logic.

===============================================================================================
PREREQUISITE 0 — Causal coupling verdict (code-reading finding, stated here; the
run itself does not "compute" this, it's a fact about `db/generate_dataset.py`)
===============================================================================================
**NO cross-supplier causal coupling exists for H_POLYMER, unlike Task 3's
`COPARENT_COUPLING`.** Read directly from `db/generate_dataset.py`:

  - `own_stress(sup_id, ...)` (line ~225): a supplier's latent stress from base
    reliability + shared-hidden-factor EVENT-POOL membership (H_PORT, H_TRUCK,
    H_CUSTOMS, **and H_POLYMER** -- all four pools are handled identically here) +
    its own idiosyncratic outage. Fully self-contained: an H_POLYMER member's
    `own_stress` already reflects its own H_POLYMER event exposure completely.
  - `stress(sup_id, ...)` (line ~246): `own_stress(sup_id) + COPARENT_COUPLING *
    own_stress(partner) for partner in coparents.get(sup_id, ())` -- the ONLY
    cross-supplier bleed-through term in the whole generator, and `coparents` is
    populated EXCLUSIVELY from `component_suppliers` (Task 3's dual-sourcing
    mechanism, a separate random assignment, `_coparent_rng.sample`), never from
    H_POLYMER membership.
  - Every label (delay, shortage, and -- confirmed by reading the label-generation
    loop directly, line ~568 -- **impact**, `lab = s["id"] in sup_hit`, i.e. "did
    this supplier's OWN shipments get delayed") derives from `stress()`, which for
    an H_POLYMER member (absent an unrelated, coincidental `component_suppliers`
    partner) reduces to exactly `own_stress()` -- fully derivable from that
    supplier's own observable history alone.

The H_POLYMER scenario IS a real, causally-grounded correlation (all 4 members are
genuinely, simultaneously exposed to the same disruption event windows, magnitude
0.95/0.85 -- not fabricated), but it is **structurally REDUNDANT for prediction
purposes**: each member's own temporal features already fully capture everything
the co-parent correlation could add, because nothing about a PARTNER's current
stress leaks into a member's OWN stress computation the way `COPARENT_COUPLING`
explicitly does for Task 3's dual-sourced suppliers. **This is the blocking finding
this round's own instructions anticipated: no fusion mechanism, however well-built,
can produce a genuine AUC improvement from discovering H_POLYMER specifically,
because there is nothing INCREMENTALLY predictive in that discovery** -- flagged in
every AUC section below, not just here.

===============================================================================================
DEVICE CHECK (done before any of the 3 variants)
===============================================================================================
Entropy computation, the per-node gate MLP pattern, and `nn.MultiheadAttention`
cross-attention all verified forward+backward, CPU vs MPS, on tensors/weights
CLONED identically across devices (not independently drawn -- an earlier,
incorrectly-constructed check using independent per-device random draws produced
spurious large diffs purely from CPU/MPS having different RNG streams, caught and
corrected before trusting any result): all four outputs (entropy, gate, attn_out,
input gradient) matched to `atol=1e-4` (max diff ~1.2e-6). Cross-attention alone
timed 1.85x faster on MPS (1.20ms/iter vs 2.22ms/iter). Both criteria met -> MPS
used for all training below.

Run: python -u -m ml.run_transformer2_fusion_pilot
"""

from __future__ import annotations

import math
import statistics
import time

import torch
import torch.nn.functional as F

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs, paired_delta_auc_ci
from ml.graph.hidden_dependency_ground_truth import H_POLYMER_SUPPLIER_IDS, h_polymer_pairs
from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.transformer2_confidence import compute_confidence
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

BASE_A_ARCH = "rgcn_attn_rung5_a"
BASE_T2_ARCH = "rgcn_attn_variant_a_transformer2"
VARIANT_ARCHS = {
    "confidence": "rgcn_attn_t2_confidence",
    "trustgate": "rgcn_attn_t2_trustgate",
    "crossattn": "rgcn_attn_t2_crossattn",
}


def _entity_degree(data, entity_type: str) -> torch.Tensor:
    n = data[entity_type].x.size(0)
    degree = torch.zeros(n, dtype=torch.long, device=data[entity_type].x.device)
    for (_src, _rel, dst), edge_index in data.edge_index_dict.items():
        if dst == entity_type and edge_index.numel() > 0:
            degree += torch.bincount(edge_index[1], minlength=n)
    return degree


@torch.no_grad()
def _hidden_dependency_analysis(model, bundle) -> dict:
    """Same contract/logic as `ml/run_transformer2_pilot.py`'s own version --
    duplicated here (not imported) since run scripts in this project are standalone
    entry points, not shared libraries (matches `_entity_degree`'s own precedent
    across `ml/run_rung_pilot.py` / `ml/run_rung5_variant_pilot.py`)."""
    node_ids = bundle.data["Supplier"].node_id
    uuid_to_idx = {u: i for i, u in enumerate(node_ids)}
    n = len(node_ids)

    z = model._last_z_impact
    normed = F.normalize(z, dim=-1)
    cos_sim = (normed @ normed.T).clone()
    cos_sim.fill_diagonal_(float("-inf"))

    idx, alpha = model._last_t2
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
            "supplier_a": a_id, "supplier_b": b_id, "percentile_rank": pct_rank,
            "symmetric_score": (att_i_to_j + att_j_to_i) / 2,
            "discovered": (j in pool_i) or (i in pool_j),
        })
    return {"pairs": pairs, "global_discovery_rate": global_discovery_rate,
            "n_suppliers": n, "total_pairs": total_pairs}


def _build_pair_records(node_ids, idx, alpha):
    n, k = idx.shape
    src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1).tolist()
    dst = idx.reshape(-1).tolist()
    w = alpha.reshape(-1).tolist()
    pairs = {}
    for s, d, wt in zip(src, dst, w):
        a, b = (s, d) if s < d else (d, s)
        rec = pairs.setdefault((a, b), [0.0, 0.0])
        if s == a:
            rec[0] = wt
        else:
            rec[1] = wt
    return pairs


def _log_hidden_dependency_links(conn, model, bundle, model_version, h_polymer_set):
    node_ids = bundle.data["Supplier"].node_id
    idx, alpha = model._last_t2
    pairs = _build_pair_records(node_ids, idx.cpu(), alpha.cpu())

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
    print(f"  [{elapsed:6.1f}s] {model_version:36s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, preds, elapsed, model


def _sign_consistency(deltas: list[float]) -> str:
    signs = {d > 0 for d in deltas}
    return "CONSISTENT" if len(signs) == 1 else "FLIPS (noise)"


@torch.no_grad()
def _prerequisite1_diagnostic(models: dict[int, "nn.Module"], test_bundles) -> None:
    """Correlation between retrieval confidence (Variant 1's own formula, applied
    diagnostically to the ORIGINAL additive-fusion model -- no new architecture
    needed for this check) and how much Transformer 2 actually shifted each
    supplier's impact prediction. Uses the freshly-retrained baseline T2 models
    (this same process), one seed at a time, pooled across all test snapshots."""
    print("\n" + "=" * 100)
    print("PREREQUISITE 1 -- does retrieval confidence correlate with downstream usefulness?")
    print("=" * 100)
    all_conf, all_shift = [], []
    per_seed_corr = []
    for seed, model in models.items():
        model.eval()
        seed_conf, seed_shift = [], []
        for bundle in test_bundles:
            model(bundle.data.x_dict, bundle.data.edge_index_dict)
            z_impact = model._last_z_impact
            idx, alpha = model._last_t2
            normed = F.normalize(z_impact, dim=-1)
            topk_cos = (normed @ normed.T).gather(1, idx)
            confidence = compute_confidence(alpha, topk_cos)

            v_proj = model.transformer2.v_proj
            value_pool = v_proj(z_impact)[idx]
            t2_out = (alpha.unsqueeze(-1) * value_pool).sum(1)
            with_t2 = torch.sigmoid(model.heads["impact"](z_impact + model.t2_scale * t2_out))
            without_t2 = torch.sigmoid(model.heads["impact"](z_impact))
            shift = (with_t2 - without_t2).abs()

            seed_conf.extend(confidence.cpu().tolist())
            seed_shift.extend(shift.cpu().tolist())
        all_conf.extend(seed_conf)
        all_shift.extend(seed_shift)

        n = len(seed_conf)
        mean_c, mean_s = statistics.fmean(seed_conf), statistics.fmean(seed_shift)
        cov = sum((c - mean_c) * (s - mean_s) for c, s in zip(seed_conf, seed_shift)) / n
        std_c = statistics.pstdev(seed_conf)
        std_s = statistics.pstdev(seed_shift)
        pearson_r = cov / (std_c * std_s) if std_c > 0 and std_s > 0 else float("nan")
        per_seed_corr.append(pearson_r)
        print(f"  seed {seed}: pearson_r(confidence, |prediction_shift|) = {pearson_r:+.4f}  "
              f"(n={n}, mean_confidence={mean_c:.4f}, mean_shift={mean_s:.6f})", flush=True)

    n = len(all_conf)
    mean_c, mean_s = statistics.fmean(all_conf), statistics.fmean(all_shift)
    cov = sum((c - mean_c) * (s - mean_s) for c, s in zip(all_conf, all_shift)) / n
    std_c, std_s = statistics.pstdev(all_conf), statistics.pstdev(all_shift)
    overall_r = cov / (std_c * std_s) if std_c > 0 and std_s > 0 else float("nan")
    print(f"\n  OVERALL (5 seeds pooled, n={n}): pearson_r = {overall_r:+.4f}", flush=True)
    verdict = ("weak/no correlation -- Confidence-Aware Fusion (Variant 1) is UNLIKELY to help, "
               "since confidence as computed doesn't track how much T2 actually moves predictions"
               if abs(overall_r) < 0.2 else
               "meaningful correlation -- Confidence-Aware Fusion has a plausible mechanism to exploit")
    print(f"  Diagnostic verdict: {verdict}", flush=True)


def _print_prerequisite0() -> None:
    print("=" * 100)
    print("PREREQUISITE 0 -- H_POLYMER causal coupling verdict (see module docstring for the full")
    print("code-reading evidence from db/generate_dataset.py)")
    print("=" * 100)
    print("  VERDICT: NO cross-supplier causal coupling exists for H_POLYMER (unlike Task 3's")
    print("  COPARENT_COUPLING=0.35). own_stress() is fully self-contained per supplier; the ONLY")
    print("  cross-supplier bleed-through term (stress()'s COPARENT_COUPLING loop) reads from")
    print("  `coparents`, which is populated exclusively by component_suppliers (Task 3's dual-")
    print("  sourcing), never by H_POLYMER membership. H_POLYMER IS a real, causally-grounded")
    print("  correlation (shared simultaneous disruption-event exposure, magnitude 0.95/0.85) but")
    print("  is REDUNDANT for prediction: each member's own temporal features already fully")
    print("  capture what the co-parent correlation could add. BLOCKING FINDING: no fusion")
    print("  mechanism tested below can be expected to produce a genuine AUC improvement FROM")
    print("  DISCOVERING H_POLYMER SPECIFICALLY -- flagged in every AUC section.\n", flush=True)


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()

    _print_prerequisite0()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM suppliers")
        n_suppliers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM graph_snapshots")
        n_snapshots = cur.fetchone()[0]
    print(f"=== Live dataset check: {n_suppliers} suppliers, {n_snapshots} graph_snapshots ===", flush=True)
    print(f"=== Device: {DEVICE} (correctness + speedup verified, see module docstring) ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    move_bundles_to_device(train_b, DEVICE)
    move_bundles_to_device(val_b, DEVICE)
    move_bundles_to_device(test_b, DEVICE)

    h_polymer_set = set(h_polymer_pairs())
    print(f"H_POLYMER ground-truth pairs: {sorted(h_polymer_set)}", flush=True)

    auc: dict[str, dict[int, dict]] = {}
    preds: dict[str, dict[int, dict]] = {}
    hd_analysis: dict[str, dict[int, list]] = {}
    run_elapsed = []
    trust_by_seed = {}  # for Variant 2's per-node analysis

    # --------------------------------------------------------------- shared baselines
    print("\n=== Shared baseline (a): Variant A alone, NO Transformer 2 -- 5 seeds ===", flush=True)
    auc["base_a"], preds["base_a"] = {}, {}
    for seed in SEEDS:
        mv = f"{BASE_A_ARCH}-t2f-basecmp-seed{seed}"
        purpose = f"Step 7b fusion pilot -- shared baseline (Variant A alone), seed={seed}"
        aucs, run_preds, elapsed, _m = _train_eval_log(conn, mv, BASE_A_ARCH, train_b, val_b, test_b, seed, purpose)
        auc["base_a"][seed], preds["base_a"][seed] = aucs, run_preds
        run_elapsed.append((mv, elapsed))

    print("\n=== Shared baseline (b): original additive-fusion Transformer 2 -- 5 seeds ===", flush=True)
    auc["base_t2"], preds["base_t2"] = {}, {}
    hd_analysis["base_t2"] = {}
    base_t2_models = {}
    for seed in SEEDS:
        mv = f"{BASE_T2_ARCH}-t2f-seed{seed}"
        purpose = f"Step 7b fusion pilot -- shared baseline (original T2 fusion), seed={seed}"
        aucs, run_preds, elapsed, model = _train_eval_log(conn, mv, BASE_T2_ARCH, train_b, val_b, test_b, seed, purpose)
        auc["base_t2"][seed], preds["base_t2"][seed] = aucs, run_preds
        base_t2_models[seed] = model
        run_elapsed.append((mv, elapsed))
        per_snapshot = []
        for b in test_b:
            model(b.data.x_dict, b.data.edge_index_dict)
            per_snapshot.append(_hidden_dependency_analysis(model, b))
        hd_analysis["base_t2"][seed] = per_snapshot

    # --------------------------------------------------------------- Prerequisite 1
    _prerequisite1_diagnostic(base_t2_models, test_b)

    # --------------------------------------------------------------- 3 variants
    for label, arch in VARIANT_ARCHS.items():
        print(f"\n=== Variant ({label}): {arch} -- 5 seeds ===", flush=True)
        auc[label], preds[label], hd_analysis[label] = {}, {}, {}
        for seed in SEEDS:
            mv = f"{arch}-t2f-seed{seed}"
            purpose = f"Step 7b fusion pilot -- {label}, seed={seed}"
            aucs, run_preds, elapsed, model = _train_eval_log(conn, mv, arch, train_b, val_b, test_b, seed, purpose)
            auc[label][seed], preds[label][seed] = aucs, run_preds
            run_elapsed.append((mv, elapsed))

            per_snapshot = [_hidden_dependency_analysis(model, b) for b in test_b]
            hd_analysis[label][seed] = per_snapshot

            last_bundle = test_b[-1]
            model(last_bundle.data.x_dict, last_bundle.data.edge_index_dict)
            n_logged = _log_hidden_dependency_links(conn, model, last_bundle, mv, h_polymer_set)
            print(f"      logged {n_logged} hidden_dependency_links rows for snapshot {last_bundle.t0.date()}", flush=True)

            if label == "trustgate":
                trust_by_seed[seed] = {}
                for bundle in test_b:
                    model(bundle.data.x_dict, bundle.data.edge_index_dict)
                    trust_by_seed[seed][bundle.t0] = (
                        model._last_trust.cpu(),
                        _entity_degree(bundle.data, "Supplier").cpu(),
                        bundle.data["Supplier"].node_id,
                    )

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 25 TRAINING RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # =============================================================== REPORTING
    print("=" * 100)
    print("AUC spread across 5 seeds, per (arm, task)")
    print("=" * 100)
    arms = ["base_a", "base_t2"] + list(VARIANT_ARCHS.keys())
    for arm in arms:
        for task in TASKS:
            values = [auc[arm][s][task] for s in SEEDS]
            print(f"  {arm:12s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  values={[round(v, 4) for v in values]}")

    print("\n" + "=" * 100)
    print("Paired bootstrap + sign-consistency -- EACH variant vs BOTH shared baselines")
    print("(BLOCKING FINDING reminder: Prerequisite 0 found no incremental causal signal in")
    print(" H_POLYMER discovery specifically -- an AUC win here would be a general architectural")
    print(" benefit, not evidence the hidden-dependency mechanism itself pays off on this dataset)")
    print("=" * 100)
    for label in VARIANT_ARCHS:
        for base_label in ["base_a", "base_t2"]:
            print(f"\n  -- {label} vs {base_label} --")
            for task in TASKS:
                deltas = []
                for seed in SEEDS:
                    yt, p_v = preds[label][seed][task]
                    _, p_b = preds[base_label][seed][task]
                    cmp = paired_delta_auc_ci(yt, p_v, p_b)
                    deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
                print(f"    {task:10s} per-seed dAUC={[round(d, 4) for d in deltas]}  "
                      f"mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}", flush=True)

    print("\n" + "=" * 100)
    print("Hidden-dependency validation -- H_POLYMER recovery vs. chance, per arm")
    print("=" * 100)
    for arm in ["base_t2"] + list(VARIANT_ARCHS.keys()):
        pct_ranks, disc_flags, global_rates = [], [], []
        for seed in SEEDS:
            for snap in hd_analysis[arm][seed]:
                for p in snap["pairs"]:
                    pct_ranks.append(p["percentile_rank"])
                    disc_flags.append(p["discovered"])
                global_rates.append(snap["global_discovery_rate"])
        mean_pct = statistics.fmean(pct_ranks)
        mean_disc = statistics.fmean(disc_flags)
        mean_global = statistics.fmean(global_rates)
        passed = mean_disc > mean_global and mean_pct > 0.5
        print(f"  {arm:12s} mean_percentile_rank={mean_pct:.4f}  mean_discovery_rate={mean_disc:.4f}  "
              f"global_chance_rate={mean_global:.4f}  -> {'PASS' if passed else 'FAIL'}", flush=True)

    # --------------------------------------------------------------- Variant 2 per-node analysis
    print("\n" + "=" * 100)
    print("Variant 2 (Per-Node Trust Gate) -- per-node stability analysis")
    print("=" * 100)
    all_trust, all_absdegree_corr = [], []
    h_polymer_trust, other_trust = [], []
    for seed in SEEDS:
        seed_trust_vals = []
        seed_pairs = []  # (|trust|, degree)
        for t0, (trust, degree, node_ids) in trust_by_seed[seed].items():
            seed_trust_vals.extend(trust.tolist())
            seed_pairs.extend(zip(trust.abs().tolist(), degree.float().tolist()))
            for nid, tv in zip(node_ids, trust.tolist()):
                if nid in H_POLYMER_SUPPLIER_IDS:
                    h_polymer_trust.append(tv)
                else:
                    other_trust.append(tv)
        mean_t = statistics.fmean(seed_trust_vals)
        std_t = statistics.pstdev(seed_trust_vals)
        share_pos = sum(1 for t in seed_trust_vals if t > 0.01) / len(seed_trust_vals)
        share_neg = sum(1 for t in seed_trust_vals if t < -0.01) / len(seed_trust_vals)
        share_zero = 1 - share_pos - share_neg
        abs_t = [p[0] for p in seed_pairs]
        deg = [p[1] for p in seed_pairs]
        n = len(abs_t)
        mean_a, mean_d = statistics.fmean(abs_t), statistics.fmean(deg)
        cov = sum((a - mean_a) * (d - mean_d) for a, d in zip(abs_t, deg)) / n
        std_a, std_d = statistics.pstdev(abs_t), statistics.pstdev(deg)
        corr = cov / (std_a * std_d) if std_a > 0 and std_d > 0 else float("nan")
        all_absdegree_corr.append(corr)
        all_trust.extend(seed_trust_vals)
        print(f"  seed {seed}: mean_trust={mean_t:+.4f}  std_trust={std_t:.4f}  "
              f"share(|trust|>0.01)_pos/neg/~0={share_pos:.3f}/{share_neg:.3f}/{share_zero:.3f}  "
              f"corr(|trust|, degree)={corr:+.4f}", flush=True)

    print(f"\n  OVERALL mean_trust={statistics.fmean(all_trust):+.4f}  "
          f"std_trust={statistics.pstdev(all_trust):.4f}")
    print(f"  corr(|trust|, degree) sign-consistency across 5 seeds: {_sign_consistency(all_absdegree_corr)}  "
          f"(values={[round(c, 4) for c in all_absdegree_corr]})")
    print(f"  H_POLYMER members' mean trust: {statistics.fmean(h_polymer_trust):+.4f}  "
          f"(n={len(h_polymer_trust)})  vs. all other suppliers' mean trust: "
          f"{statistics.fmean(other_trust):+.4f}  (n={len(other_trust)})", flush=True)

    print("\n" + "=" * 100)
    print("Per-run elapsed time")
    print("=" * 100)
    for mv, elapsed in run_elapsed:
        print(f"  {mv:44s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
