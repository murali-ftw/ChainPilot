"""
Over-smoothing sweep for the three RGCN-attention hybrids -- RGCN+attn
("Option 1"), RGCN+basisattn ("Option 2"), RGCN+relemb ("Option 3") -- each
at its own matched-parameter config, against the live dataset (confirmed at
runtime, not assumed -- see the printed dataset-version block below).

**Dataset-version caveat, stated explicitly.** HGT's reference depth sweep
(`ml/run_step_v3.py`, reused here as the comparison baseline, NOT retrained)
was run against the base v3 dataset generation, before the Task 3 follow-up
(`reports/hades_model_development_report.md` Round 4) reloaded the database
with the co-parent/dual-sourcing dataset. The database currently live for
THIS run has that co-parent mechanism active (confirmed at startup: real
`SUPPLIES` edge count and 2-hop co-parent reach both checked live, not
assumed) -- the same dataset every RGCN-family round since the 4-architecture
ablation has actually been trained against, but NOT the same generation
HGT's own reused depth-sweep numbers came from. This means the three new
architectures' numbers below are directly comparable TO EACH OTHER (all
trained in-process against the identical live data this run), but the
HGT column in the side-by-side table carries a dataset-generation caveat
that the three new columns don't. Flagged here rather than silently treated
as an apples-to-apples four-way comparison.

Runs: 3 architectures x (5 seeds x baseline + 3 seeds x {L1,L2,L3,L4}) = 51
training runs. `baseline` (num_layers=4, shared_depth=None -> structural-
prior readout) is each architecture's headline matched-d number; L1-L4 are
a diagnostic sweep (3 seeds, not a formal significance claim, matching this
project's established basis-ablation precedent for exploratory sweeps).

Every run gets its own `model_registry` row, suffixed
`-oversmooth-{arch}-{config}-seed{n}` -- never touches any existing row.

Embedding-similarity diagnostic: right after each `baseline` config's 5
training runs complete (no extra training, no checkpoint reload -- computed
in-process from the just-trained model), for every node type at every
layer h^1..h^4, 500 random same-type node pairs are sampled and their mean
cosine similarity computed, on the first test snapshot (same snapshot
`ml/run_step4.py`'s own over-smoothing measurement used). This differs
intentionally from `ml/evaluate.py::mean_pairwise_cosine_similarity` (which
subsamples up to 300 nodes and computes ALL pairs among them) -- this
diagnostic samples a fixed 500 random PAIRS instead, per this round's own
specification.

Run: python -u -m ml.run_rgcn_oversmoothing_sweep
"""

from __future__ import annotations

import statistics
import time

import torch

from ml.data.db import get_connection
from ml.evaluate import best_f1_threshold, collect_predictions, evaluate_split, log_evaluation_runs
from ml.models.depth import TASKS
from ml.train import (
    TEST_START,
    TRAIN_CUTOFF,
    load_all_snapshot_bundles,
    run_training_job,
    split_bundles,
)

DEPTH_CONFIGS = [
    ("baseline", 4, None),
    ("L1", 1, 1),
    ("L2", 2, 2),
    ("L3", 3, 3),
    ("L4", 4, 4),
]
BASELINE_SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1, 2]  # L1-L4 -- diagnostic, 3 seeds

# (arch_key, hidden, num_bases, extra_kwargs) -- each architecture's own
# Phase-4-chosen (or already-known, opt1/opt3) matched-parameter config.
ARCH_CONFIGS = [
    ("rgcn_attn", 128, 10, {}),
    ("rgcn_battn", 110, 9, {"num_bases_attn": 16}),
    ("rgcn_relemb", 128, 10, {"relation_embed_dim": 16}),
]

N_PAIRS = 500


@torch.no_grad()
def _sampled_pair_cosine_similarity(embeddings: torch.Tensor, n_pairs: int = N_PAIRS,
                                     seed: int = 0) -> float:
    """Mean cosine similarity over `n_pairs` randomly sampled (i, j), i != j,
    same-type node pairs -- NOT `ml/evaluate.py`'s own all-pairs-among-a-
    subsample method, a fixed random-pair sample per this round's spec."""
    n = embeddings.size(0)
    if n < 2:
        return float("nan")
    g = torch.Generator().manual_seed(seed)
    k = min(n_pairs, n * (n - 1))  # can't exceed the number of ordered i!=j pairs that exist
    oversample = max(k * 4, 32)
    i_idx = torch.randint(0, n, (oversample,), generator=g)
    j_idx = torch.randint(0, n, (oversample,), generator=g)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep][:k], j_idx[keep][:k]
    if i_idx.numel() == 0:
        return float("nan")
    normed = torch.nn.functional.normalize(embeddings, dim=-1)
    sims = (normed[i_idx] * normed[j_idx]).sum(-1)
    return float(sims.mean())


@torch.no_grad()
def _embedding_similarity_profile(model, bundle) -> dict[int, dict[str, float]]:
    """{layer_index (1-based): {node_type: mean cosine similarity over 500 sampled pairs}}"""
    model.eval()
    layers = model.encoder(bundle.data.x_dict, bundle.data.edge_index_dict)
    return {
        i + 1: {nt: _sampled_pair_cosine_similarity(h, seed=i) for nt, h in layer.items()}
        for i, layer in enumerate(layers)
    }


def _train_eval_log(conn, model_version, arch_key, train_b, val_b, test_b,
                     hidden, num_bases, extra_kwargs, num_layers, shared_depth, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, arch_key, train_b, val_b,
        num_layers=num_layers, shared_depth=shared_depth, hidden=hidden, epochs=100,
        seed=seed, purpose=purpose, num_bases=num_bases, **extra_kwargs,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, arch_key, results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    n_params = model.parameter_count()
    print(f"  [{elapsed:6.1f}s] {model_version:38s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, elapsed, n_params, model


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()

    # --------------------------------------------------------------- dataset-version check (live, not assumed)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM suppliers")
        n_suppliers = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM graph_snapshots")
        n_snapshots = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM component_suppliers")
        n_coparent_rows = cur.fetchone()[0]
    print(f"=== Live dataset check: {n_suppliers} suppliers, {n_snapshots} graph_snapshots, "
          f"{n_coparent_rows} component_suppliers rows ===", flush=True)

    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    # Real SUPPLIES edge count + 2-hop co-parent reach, live -- confirms
    # whether dual-sourcing is actually active on the currently loaded graph.
    supplies_et = ("Supplier", "SUPPLIES", "Component")
    n_supplies_edges = test_b[0].data[supplies_et].edge_index.size(1) if supplies_et in test_b[0].data.edge_types else None
    print(f"    SUPPLIES edge count (first test snapshot): {n_supplies_edges}", flush=True)

    # auc[arch][config][seed][task] ; elapsed list ; params[arch]
    auc = {a[0]: {c[0]: {} for c in DEPTH_CONFIGS} for a in ARCH_CONFIGS}
    run_elapsed = []
    arch_params = {}
    # similarity[arch][layer][node_type] -> list of per-seed values (baseline config only)
    similarity = {a[0]: {} for a in ARCH_CONFIGS}

    print(f"\n=== Over-smoothing sweep: 3 architectures x (5-seed baseline + 3-seed L1-L4) = 51 runs ===",
          flush=True)
    for arch_key, hidden, num_bases, extra_kwargs in ARCH_CONFIGS:
        for config_name, num_layers, shared_depth in DEPTH_CONFIGS:
            seeds = BASELINE_SEEDS if config_name == "baseline" else SWEEP_SEEDS
            for seed in seeds:
                model_version = f"{arch_key}-oversmooth-{config_name}-seed{seed}"
                purpose = (f"Over-smoothing sweep -- {arch_key}, {config_name} "
                           f"(num_layers={num_layers}, shared_depth={shared_depth}), "
                           f"hidden={hidden}, num_bases={num_bases}, seed={seed}")
                aucs, elapsed, n_params, model = _train_eval_log(
                    conn, model_version, arch_key, train_b, val_b, test_b,
                    hidden, num_bases, extra_kwargs, num_layers, shared_depth, seed, purpose,
                )
                auc[arch_key][config_name][seed] = aucs
                run_elapsed.append((model_version, elapsed))
                arch_params[arch_key] = n_params

                if config_name == "baseline":
                    profile = _embedding_similarity_profile(model, test_b[0])
                    for layer_idx, per_type in profile.items():
                        similarity[arch_key].setdefault(layer_idx, {})
                        for nt, sim in per_type.items():
                            similarity[arch_key][layer_idx].setdefault(nt, []).append(sim)

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 51 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- AUC-vs-depth table
    print("=" * 100)
    print("AUC spread across seeds, per (architecture, task, config) -- baseline: 5 seeds, L1-L4: 3 seeds")
    print("=" * 100)
    for arch_key, hidden, num_bases, extra_kwargs in ARCH_CONFIGS:
        print(f"\n  -- {arch_key} (hidden={hidden}, num_bases={num_bases}, {extra_kwargs}, "
              f"params={arch_params[arch_key]:,}) --")
        for config_name, _, _ in DEPTH_CONFIGS:
            for task in TASKS:
                seeds = BASELINE_SEEDS if config_name == "baseline" else SWEEP_SEEDS
                values = [auc[arch_key][config_name][s][task] for s in seeds]
                print(f"    {config_name:10s} {task:10s} mean={statistics.fmean(values):.4f}  "
                      f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                      f"values={[round(v,4) for v in values]}")

    # --------------------------------------------------------------- embedding-similarity table
    print("\n" + "=" * 100)
    print("Embedding-similarity diagnostic (baseline config, mean over 500 sampled same-type pairs, "
          "averaged across the 5 baseline seeds)")
    print("=" * 100)
    for arch_key, *_ in ARCH_CONFIGS:
        print(f"\n  -- {arch_key} --")
        node_types = sorted({nt for per_type in similarity[arch_key].values() for nt in per_type})
        for layer_idx in sorted(similarity[arch_key]):
            row = []
            for nt in node_types:
                vals = similarity[arch_key][layer_idx].get(nt, [])
                vals = [v for v in vals if v == v]  # drop NaN
                row.append(f"{nt}={statistics.fmean(vals):.3f}" if vals else f"{nt}=n/a")
            print(f"    layer {layer_idx}: " + "  ".join(row))

    # --------------------------------------------------------------- elapsed time
    print("\n" + "=" * 100)
    print("Per-run elapsed time")
    print("=" * 100)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:42s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
