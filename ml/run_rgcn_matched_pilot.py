"""
Matched-parameter RGCN+attn ("Option 1") vs RGCN+relemb ("Option 3") vs HGT
-- against the v3 dataset (800 suppliers, 15 monthly snapshots Jul 2024 -
Sep 2025 -- confirmed live before this script was written, not assumed).

Trains ONLY the two RGCN-family hybrids, matched-d, 5 seeds each (10 runs
total). **HGT is NOT retrained** -- its fixed-d=64 numbers from the
rgcn_attn pilot (`-rgcnattn-pilot-seed{n}`, architecture=
heterogeneous_graph_transformer, `reports/rgcn_attn_pilot.md`) are pulled
directly from `model_evaluation_runs` and reused as-is. This is not a
shortcut of convenience: HGT's fixed-d and matched-d arms are identical by
construction (hidden=64 is the anchor for both, established since
`run_step5.py`), so a "matched-d HGT" run would train the exact same model
that's already logged five times over -- spending real compute to
reproduce a result already on record. Does not touch `run_step5.py`,
`run_step_v3.py`, `run_step_v4_4arch.py`, or `run_rgcn_attn_pilot.py`.

**Matched-parameter search (Phase E), reasoning for the (hidden, num_bases)
choice actually trained here:** `reports/step5_result_v4_4arch.md` found
that hitting HGT's ~701K-param anchor for plain RGCN forces `hidden` up and
`num_bases` down (self-loop/input-projection costs dominate at small
widths) -- and that `num_bases` shrinking specifically hurt plain RGCN's
impact score (RGCN's matched arm, num_bases=4, scored worse on impact than
RGCN's own fixed arm, num_bases=8). To avoid blindly repeating that
outcome, the search below reports THREE candidates per architecture: (a)
the tightest raw-parameter fit (unconstrained `num_bases`, which lands on a
very small `num_bases` just like plain RGCN's prior search), (b) the best
fit found while constraining `num_bases >= 8`, and (c) an intermediate
point. **Candidate (b) is what's actually trained below** -- deliberately
preserving `num_bases >= 8` even though the raw parameter match is looser
as a result.

**CI methodology note, stated explicitly (same spirit as
`ml/evaluate.py`'s own row_bootstrap-vs-time-blocked caveat):** rgcn_attn-
matched vs rgcn_relemb-matched uses a genuine PAIRED bootstrap
(`ml/evaluate.py::paired_delta_auc_ci`) -- both are trained in-process this
round, so their predictions on the identical test set are directly
available together. rgcn_attn-matched vs hgt and rgcn_relemb-matched vs hgt
CANNOT use that same paired method, because HGT's raw per-sample test-set
predictions from the original pilot run were never persisted (no model
checkpoint is saved anywhere in this codebase, only registry rows + summary
metrics) and this script deliberately does not retrain HGT to regenerate
them. Those two comparisons instead use each side's own already-computed
UNPAIRED single-model CI (new arch's freshly-computed CI from this run;
HGT's stored CI from `model_evaluation_runs`, both `ci_method=
'row_bootstrap'`) and report per-seed point-delta sign-consistency plus
whether the two independent CIs overlap -- a real but strictly weaker
significance test than the paired method used everywhere else in this
project, flagged as such rather than silently presented as equivalent.

Every new run gets its own `model_registry` row suffixed
`-rgcnmatched-pilot-seed{n}` -- never touches any existing row.

Run: python -u -m ml.run_rgcn_matched_pilot
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
RELATION_EMBED_DIM = 16  # default, unchanged from Phase A -- not part of Phase E's search

# rgcn_attn (fixed-d=64) pilot reference numbers, from reports/rgcn_attn_pilot.md
# -- for the before/after table only, not recomputed here (that pilot's own
# runs are already logged under `-rgcnattn-pilot-seed{n}`).
RGCN_ATTN_FIXED_D_PILOT_AUC = {
    "delay": {"mean": 0.8127, "std": 0.0035, "min": 0.8084, "max": 0.8161},
    "shortage": {"mean": 0.7985, "std": 0.0023, "min": 0.7953, "max": 0.8017},
    "impact": {"mean": 0.9393, "std": 0.0098, "min": 0.9261, "max": 0.9496},
}
RGCN_ATTN_FIXED_D_PARAMS = 170_984  # encoder-only, hidden=64, num_bases=8


def _matched_param_search(metadata, in_dims, anchor: int):
    """Phase E: for rgcn_attn and rgcn_relemb, grid-search (hidden,
    num_bases) at relation_embed_dim=16 (rgcn_relemb only), report 3
    candidates each. Returns {arch: {"a":..., "b":..., "c":...}} where each
    value is (hidden, num_bases, params)."""
    out = {}
    for arch in ("rgcn_attn", "rgcn_relemb"):
        results = []
        for hidden in range(64, 260, 2):
            for nb in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24]:
                enc = build_encoder(arch, metadata, in_dims, hidden=hidden, num_layers=4,
                                     num_bases=nb, relation_embed_dim=RELATION_EMBED_DIM)
                n = sum(p.numel() for p in enc.parameters())
                results.append((abs(n - anchor), hidden, nb, n))
        results.sort()
        a = results[0]
        b = min((r for r in results if r[2] >= 8), key=lambda r: r[0])
        lo, hi = sorted((a[2], b[2]))
        mid = [r for r in results if lo < r[2] < hi]
        c = min(mid, key=lambda r: r[0]) if mid else b
        out[arch] = {"a": a, "b": b, "c": c}
    return out


def _train_eval_log(conn, model_version, arch_label, arch_key, train_b, val_b, test_b,
                     hidden, num_bases, seed, purpose):
    t0 = time.monotonic()
    result = run_training_job(
        conn, model_version, arch_label, train_b, val_b,
        num_layers=4, shared_depth=None, hidden=hidden, epochs=100,
        seed=seed, purpose=purpose, num_bases=num_bases, relation_embed_dim=RELATION_EMBED_DIM,
    )
    model = result["model"]
    thresholds = {task: best_f1_threshold(*collect_predictions(model, val_b)[task]) for task in TASKS}
    results = evaluate_split(model, test_b, thresholds)
    log_evaluation_runs(conn, model_version, arch_label, results, "test",
                         train_end_t0=TRAIN_CUTOFF, test_start_t0=TEST_START)
    elapsed = time.monotonic() - t0
    aucs = {task: results.get(task, {}).get("roc_auc", {}).get("value", float("nan")) for task in TASKS}
    cis = {task: (results.get(task, {}).get("roc_auc", {}).get("ci_lower"),
                  results.get(task, {}).get("roc_auc", {}).get("ci_upper")) for task in TASKS}
    preds = {task: collect_predictions(model, test_b)[task] for task in TASKS}
    n_params = model.parameter_count()
    print(f"  [{elapsed:6.1f}s] {model_version:28s} params={n_params:,} "
          f"best_epoch={result['best_epoch']:3d}  " +
          "  ".join(f"{t}={aucs[t]:.4f}" for t in TASKS), flush=True)
    return aucs, cis, preds, elapsed, n_params


def _sign_consistency(deltas: list[float]) -> str:
    signs = {d > 0 for d in deltas}
    return "CONSISTENT" if len(signs) == 1 else "FLIPS (noise)"


def _query_hgt_pilot_rows(conn) -> tuple[dict, dict, int]:
    """Pull HGT's existing fixed-d=64 AUC + CI from the rgcn_attn pilot's
    5 rows -- never retrained here. Also pulls its logged parameter_count
    from model_registry, rather than hardcoding the number from a prior
    report."""
    hgt_auc, hgt_ci = {}, {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT model_version, task, metric_value, ci_lower, ci_upper
            FROM model_evaluation_runs
            WHERE model_version LIKE 'hgt-rgcnattn-pilot-seed%%'
              AND metric_name = 'roc_auc' AND dataset_split = 'test'
            """
        )
        rows = cur.fetchall()
        cur.execute(
            """
            SELECT parameter_count FROM model_registry
            WHERE model_version LIKE 'hgt-rgcnattn-pilot-seed%%'
            """
        )
        param_counts = {r[0] for r in cur.fetchall()}
    for model_version, task, metric_value, ci_lower, ci_upper in rows:
        seed = int(model_version.rsplit("seed", 1)[1])
        hgt_auc.setdefault(seed, {})[task] = float(metric_value)
        hgt_ci.setdefault(seed, {})[task] = (float(ci_lower) if ci_lower is not None else None,
                                              float(ci_upper) if ci_upper is not None else None)
    assert set(hgt_auc.keys()) == set(SEEDS), f"expected HGT rows for seeds {SEEDS}, found {sorted(hgt_auc)}"
    for seed in SEEDS:
        assert set(hgt_auc[seed]) == set(TASKS), f"seed {seed} missing tasks: {set(TASKS) - set(hgt_auc[seed])}"
    assert len(param_counts) == 1, f"expected identical parameter_count across HGT's 5 rows, found {param_counts}"
    return hgt_auc, hgt_ci, param_counts.pop()


def main() -> None:
    overall_start = time.monotonic()
    conn = get_connection()
    bundles = load_all_snapshot_bundles(conn)
    train_b, val_b, test_b = split_bundles(bundles)
    print(f"train t0s: {[str(b.t0.date()) for b in train_b]}", flush=True)
    print(f"val   t0s: {[str(b.t0.date()) for b in val_b]}", flush=True)
    print(f"test  t0s: {[str(b.t0.date()) for b in test_b]}", flush=True)

    metadata = train_b[0].data.metadata()
    in_dims = {nt: train_b[0].data[nt].x.size(-1) for nt in train_b[0].data.node_types}

    hgt_enc = build_encoder("hgt", metadata, in_dims, hidden=64, num_layers=4)
    anchor = sum(p.numel() for p in hgt_enc.parameters())
    print(f"\n=== HGT real encoder-only anchor (recomputed live): {anchor:,} ===", flush=True)

    print("\n=== Phase E: matched-parameter search, 3 candidates each ===", flush=True)
    search = _matched_param_search(metadata, in_dims, anchor)
    for arch, cands in search.items():
        print(f"  {arch}:", flush=True)
        for label, (diff, hidden, nb, n) in cands.items():
            print(f"    ({label}) hidden={hidden:4d} num_bases={nb:3d} params={n:,} "
                  f"diff={diff:,} ({100*diff/anchor:.2f}%)", flush=True)
    print("\n  Chosen for training (candidate b, num_bases>=8 preserved -- avoiding the "
          "num_bases-shrinkage-hurts-impact outcome the 4-arch round found for plain RGCN):",
          flush=True)
    chosen = {}
    for arch, cands in search.items():
        _diff, hidden, nb, n = cands["b"]
        chosen[arch] = (hidden, nb, n)
        print(f"    {arch}: hidden={hidden}, num_bases={nb}, params={n:,}", flush=True)

    print("\n=== Querying HGT's existing fixed-d=64 rows (NOT retrained) ===", flush=True)
    hgt_auc_by_seed, hgt_ci_by_seed, hgt_param_count = _query_hgt_pilot_rows(conn)
    print(f"  Found HGT rows for seeds {sorted(hgt_auc_by_seed)}, all 3 tasks each, "
          f"parameter_count={hgt_param_count:,}.", flush=True)

    # auc[arm][seed][task] ; cis[arm][seed][task] = (lo, hi) ; preds[arm][seed][task] = (y_true, y_prob)
    auc = {"rgcn_attn": {}, "rgcn_relemb": {}}
    cis = {"rgcn_attn": {}, "rgcn_relemb": {}}
    preds = {"rgcn_attn": {}, "rgcn_relemb": {}}
    run_elapsed = []
    encoder_params = {}

    print("\n=== Training: 5 seeds x 2 architectures = 10 runs, matched-d only ===", flush=True)
    for seed in SEEDS:
        for arch_key in ("rgcn_attn", "rgcn_relemb"):
            hidden, num_bases, _n = chosen[arch_key]
            model_version = f"{arch_key}-rgcnmatched-pilot-seed{seed}"
            purpose = (f"RGCN matched-parameter pilot -- {arch_key} matched-d, hidden={hidden}, "
                       f"num_bases={num_bases}, seed={seed}")
            aucs, run_cis, run_preds, elapsed, n_params = _train_eval_log(
                conn, model_version, arch_key, arch_key, train_b, val_b, test_b,
                hidden, num_bases, seed, purpose,
            )
            auc[arch_key][seed] = aucs
            cis[arch_key][seed] = run_cis
            preds[arch_key][seed] = run_preds
            encoder_params[arch_key] = n_params
            run_elapsed.append((model_version, elapsed))

    total_elapsed = time.monotonic() - overall_start
    print(f"\n=== ALL 10 RUNS COMPLETE in {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s) ===\n", flush=True)

    # --------------------------------------------------------------- summary stats
    print("=" * 90)
    print("AUC spread across 5 seeds, per (task, arm) -- rgcn_attn/rgcn_relemb matched-d, hgt fixed-d=64")
    print("=" * 90)
    for arch_key in ("rgcn_attn", "rgcn_relemb"):
        for task in TASKS:
            values = [auc[arch_key][s][task] for s in SEEDS]
            print(f"  {arch_key + '-matched':22s} {task:10s} mean={statistics.fmean(values):.4f}  "
                  f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
                  f"values={[round(v,4) for v in values]}")
    for task in TASKS:
        values = [hgt_auc_by_seed[s][task] for s in SEEDS]
        print(f"  {'hgt (fixed-d, reused)':22s} {task:10s} mean={statistics.fmean(values):.4f}  "
              f"std={statistics.pstdev(values):.4f}  min={min(values):.4f}  max={max(values):.4f}  "
              f"values={[round(v,4) for v in values]}")

    # --------------------------------------------------------------- before/after (rgcn_attn only)
    print("\n" + "=" * 90)
    print("Before/after: rgcn_attn fixed-d=64 pilot vs this round's matched-d")
    print("=" * 90)
    for task in TASKS:
        before = RGCN_ATTN_FIXED_D_PILOT_AUC[task]
        after_values = [auc["rgcn_attn"][s][task] for s in SEEDS]
        after_mean = statistics.fmean(after_values)
        print(f"  {task:10s} fixed-d=64  mean={before['mean']:.4f} std={before['std']:.4f} "
              f"[{before['min']:.4f},{before['max']:.4f}]  (params={RGCN_ATTN_FIXED_D_PARAMS:,})")
        print(f"  {'':10s} matched-d   mean={after_mean:.4f} std={statistics.pstdev(after_values):.4f} "
              f"[{min(after_values):.4f},{max(after_values):.4f}]  (params={encoder_params['rgcn_attn']:,})"
              f"  delta={after_mean - before['mean']:+.4f}")

    # --------------------------------------------------------------- rgcn_attn vs rgcn_relemb (paired)
    print("\n" + "=" * 90)
    print("Sign-consistency, PAIRED bootstrap (both trained in-process this round) -- "
          "rgcn_attn-matched vs rgcn_relemb-matched")
    print("=" * 90)
    for task in TASKS:
        deltas = []
        for seed in SEEDS:
            yt, pa = preds["rgcn_attn"][seed][task]
            _, pr = preds["rgcn_relemb"][seed][task]
            cmp = paired_delta_auc_ci(yt, pa, pr)
            deltas.append(cmp["delta"] if cmp["delta"] is not None else float("nan"))
        print(f"    {task:10s} per-seed dAUC={[round(d,4) for d in deltas]}  "
              f"mean={statistics.fmean(deltas):+.4f}  -> {_sign_consistency(deltas)}")

    # --------------------------------------------------------------- vs HGT (unpaired -- see docstring)
    print("\n" + "=" * 90)
    print("Sign-consistency vs HGT, UNPAIRED (HGT not retrained -- per-seed point-delta + "
          "independent-CI-overlap, NOT the paired method used above; see module docstring)")
    print("=" * 90)
    for arch_key in ("rgcn_attn", "rgcn_relemb"):
        for task in TASKS:
            deltas = []
            for seed in SEEDS:
                new_auc = auc[arch_key][seed][task]
                hgt_auc_val = hgt_auc_by_seed[seed][task]
                delta = new_auc - hgt_auc_val
                deltas.append(delta)
                new_lo, new_hi = cis[arch_key][seed][task]
                hgt_lo, hgt_hi = hgt_ci_by_seed[seed][task]
                overlap = "CI overlap" if (new_lo is None or hgt_hi is None or
                                            not (new_lo > hgt_hi or new_hi < hgt_lo)) else "CIs disjoint"
                print(f"      seed{seed} {task:10s} {arch_key}-matched={new_auc:.4f} "
                      f"[{new_lo:.4f},{new_hi:.4f}]  vs  hgt={hgt_auc_val:.4f} "
                      f"[{hgt_lo:.4f},{hgt_hi:.4f}]  dAUC={delta:+.4f}  ({overlap})")
            print(f"    -> {arch_key}-matched vs hgt, {task:10s} mean dAUC={statistics.fmean(deltas):+.4f}  "
                  f"{_sign_consistency(deltas)}", flush=True)

    # --------------------------------------------------------------- parameter counts
    print("\n" + "=" * 90)
    print("Real parameter counts (full model, incl. 3 task heads)")
    print("=" * 90)
    print(f"  hgt (fixed-d=64, reused)        {hgt_param_count:,}")
    for arch_key in ("rgcn_attn", "rgcn_relemb"):
        print(f"  {arch_key}-matched{'':14s} {encoder_params[arch_key]:,}")

    print("\n" + "=" * 90)
    print("Per-run elapsed time")
    print("=" * 90)
    for model_version, elapsed in run_elapsed:
        print(f"  {model_version:32s} {elapsed:7.1f}s")

    conn.close()
    print(f"\nTotal wall-clock: {total_elapsed/3600:.2f}h ({total_elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
