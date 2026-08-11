#!/usr/bin/env python3
"""
The V2 evaluation protocol: train, evaluate, compare, report.

Implements `docs/00_Benchmark_Specification.md`'s Evaluation Protocol against the
spec-scale CSV corpus in `db/csv/`:

* **Temporal split by snapshot count**, never randomly, in V1's established
  40/20/40 proportions applied to whatever `SNAPSHOTS` the data on disk has --
  read from `resolved_config.json`, not hardcoded from a date table.
* **Five seeds with sign consistency**, not an aggregate mean alone. Seeds 42/43
  are retained on disk; 44/45/46 are regenerated on demand via
  `db/regenerate_seed.py`, and this script will invoke it (`--auto-regenerate`)
  rather than silently skipping a seed.
* **Paired bootstrap confidence intervals** for every architecture comparison,
  block-resampling whole snapshots (`ml/evaluate.py`).
* **Variant interpretation rules enforced in the reporting code**: `REPORT_AGAINST`
  below is consulted by `comparisons()`, so a variant with a mandated reference
  cannot be reported as a standalone effect. Attempting it raises.
* **Positive counts printed beside every AUC**, so an underpowered cell (Variant K
  impact, 1,896 mean positives against the spec's 2,000 floor) is visible in the
  table rather than in a footnote.

Modes:

    sanity   Reproduce V1's recorded SHARE numbers on the byte-identical anchor
             before anything else is trusted (step 2 of the build).
    sweep    Architectures x variants x seeds at spec scale.
    report   Re-derive the tables from an existing results file, no training.

Usage:
    python3 ml/run_benchmark_eval.py sanity --out out/sanity.json
    python3 ml/run_benchmark_eval.py sweep --variants 0,K --archs rgcn_attn,hgt \\
        --seeds 42,43 --out out/sweep.json
    python3 ml/run_benchmark_eval.py report --results out/sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.loader import load_bundles, split_bundles  # noqa: E402
from ml.evaluate import (  # noqa: E402
    best_f1_threshold, bootstrap_ci, collect_predictions, paired_delta_auc_ci, sign_consistency,
)
from ml.models.depth import TASKS  # noqa: E402
from ml.models.encoder import ARCHITECTURE_NAMES  # noqa: E402
from ml.train import train_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, "db", "csv")   # overridable with --csv-dir
CACHE_DIR = os.path.join(REPO, "ml", ".cache")
ALL_VARIANTS = ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
RETAINED_SEEDS = [42, 43]

# Matched-parameter arms, carried over from HADES_v1/reports/info.md §4 so the
# V2 numbers sit on the same budget V1's conclusions were drawn at. Every count
# below was re-derived from this port and equals V1's recorded value exactly
# (except gps, which V1 never ran -- see ml/models/encoder.py).
ARCH_CONFIG = {
    "gcn":         dict(hidden=78),
    "graphsage":   dict(hidden=66),
    "gat":         dict(hidden=92),
    "hgt":         dict(hidden=64),
    "rgcn":        dict(hidden=138, num_bases=4),
    "gps":         dict(hidden=40),
    "rgcn_attn":   dict(hidden=128, num_bases=10),
    "rgcn_relemb": dict(hidden=128, num_bases=10, relation_embed_dim=16),
    "rgcn_battn":  dict(hidden=110, num_bases=9, num_bases_attn=16),
}

# V1's recorded matched-parameter results (HADES_v1/reports/info.md §4), the
# target the sanity check must land near.
V1_REFERENCE = {
    "rgcn_attn": {"delay": (0.8105, 0.8118), "shortage": (0.7971, 0.7982),
                  "impact": (0.9365, 0.9393), "params": 752211},
}

# The spec's variant dependency table, made executable. A variant listed here
# has NO standalone interpretation: its effect must be read as a delta against
# its reference variant, because the reference is what isolates the mechanism.
REPORT_AGAINST = {"A": "J", "D": "B", "F": "E", "K": "D"}

# Every other variant is `Base + one mechanism` and is read against Variant 0.
# `reference_for()` is the single place that decides, so no caller can invent one.
DEFAULT_REFERENCE = "0"


def reference_for(variant: str) -> str | None:
    if variant == DEFAULT_REFERENCE:
        return None
    return REPORT_AGAINST.get(variant, DEFAULT_REFERENCE)

# Caveats the reporting code must attach, not leave to the reader.
VARIANT_CAVEATS = {
    "K": ("impact is UNDER the spec's 2,000-positive floor on this configuration "
          "(1,896 mean over 5 seeds, in band on 3/5) — an impact result here is "
          "underpowered by construction, see docs/phase6_spec_scale_report.md §5"),
    "F": ("resilience recoverability INVERTS in sign between Variant E (direct, "
          "AUC 0.561) and Variant F (inverted, 0.396) — a reader comparing E and F "
          "head-to-head without this will read F as noise "
          "(docs/PHASE2_PHASE6_IMPLEMENTATION.md §8.2)"),
    "A": ("Mechanism A removes zero-shipper nodes, which makes Variant F, not "
          "Variant A, the harder case for resilience recovery "
          "(docs/PHASE2_PHASE6_IMPLEMENTATION.md §8.4)"),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def variant_dir(variant: str, seed: int, csv_dir: str | None = None) -> str:
    return os.path.join(csv_dir or CSV_DIR, f"v{variant}_seed{seed}")


def ensure_variant(variant: str, seed: int, auto_regenerate: bool, csv_dir: str | None = None) -> str:
    d = variant_dir(variant, seed, csv_dir)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "resolved_config.json")):
        return d
    if not auto_regenerate:
        raise FileNotFoundError(
            f"{d} is not on disk (seeds {RETAINED_SEEDS} are retained; 44/45/46 are "
            f"regenerated). Re-run with --auto-regenerate, or generate it with "
            f"`python3 db/regenerate_seed.py --variant {variant} --seed {seed}`.")
    print(f"  regenerating variant {variant} seed {seed} (not retained on disk) ...", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, "db", "regenerate_seed.py"),
                    "--variant", variant, "--seed", str(seed)], check=True, cwd=REPO)
    return d


def run_one(dataset_dir: str, architecture: str, model_seed: int, epochs: int,
            device: str, num_layers: int, patience: int | None,
            bundles=None, manifest=None, progress: bool = False) -> dict:
    """Train one (dataset, architecture, model seed) and evaluate on the test split.

    Returns metrics plus the raw test predictions, because every architecture
    comparison downstream is PAIRED on the identical held-out rows and cannot be
    reconstructed from summary statistics alone."""
    if bundles is None:
        bundles, manifest = load_bundles(dataset_dir, cache_dir=CACHE_DIR)
    train_b, val_b, test_b = split_bundles(bundles)

    cfg = dict(ARCH_CONFIG[architecture])
    result = train_model(architecture, train_b, val_b, num_layers=num_layers,
                         epochs=epochs, seed=model_seed, device=device,
                         patience=patience, progress=progress, **cfg)
    model = result["model"]

    # Threshold is chosen on VALIDATION and frozen for test.
    val_preds = collect_predictions(model, val_b)
    thresholds = {t: best_f1_threshold(val_preds[t]["y"], val_preds[t]["p"]) for t in TASKS}
    test_preds = collect_predictions(model, test_b)

    metrics = {}
    for task in TASKS:
        y, p, blk = test_preds[task]["y"], test_preds[task]["p"], test_preds[task]["block"]
        if len(y) == 0 or y.sum() in (0, len(y)):
            metrics[task] = None
            continue
        from sklearn.metrics import roc_auc_score
        ci = bootstrap_ci(y, p, "roc_auc", thresholds[task], block=blk)
        metrics[task] = {
            "roc_auc": float(roc_auc_score(y, p)),
            "ci_lower": ci[0] if ci else None, "ci_upper": ci[1] if ci else None,
            "ci_method": ci[2] if ci else None,
            "n": int(len(y)), "positives": int(y.sum()),
            "threshold": thresholds[task],
        }
    return {
        "architecture": architecture, "architecture_name": ARCHITECTURE_NAMES[architecture],
        "model_seed": model_seed, "dataset": os.path.basename(dataset_dir),
        "hyperparameters": result["hyperparameters"],
        "parameter_count": model.parameter_count(),
        "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
        "best_val_auc": result["best_val_auc"], "train_seconds": result["seconds"],
        "split": {"train": len(train_b), "val": len(val_b), "test": len(test_b),
                  "train_last_t0": str(train_b[-1].t0), "test_first_t0": str(test_b[0].t0)},
        "resolved_config": manifest["config"],
        "dataset_label_counts": manifest["label_counts"],
        "metrics": metrics,
        "_preds": {t: {k: v.tolist() for k, v in test_preds[t].items()} for t in TASKS},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _by(runs, **filters):
    return [r for r in runs if all(r[k] == v for k, v in filters.items())]


def aggregate(runs: list[dict]) -> list[dict]:
    """(dataset variant, architecture, task) -> mean/std AUC over model seeds,
    with the positive count carried alongside."""
    out = []
    keys = sorted({(r["variant"], r["architecture"]) for r in runs})
    for variant, arch in keys:
        rs = _by(runs, variant=variant, architecture=arch)
        for task in TASKS:
            vals = [r["metrics"][task]["roc_auc"] for r in rs if r["metrics"].get(task)]
            pos = [r["metrics"][task]["positives"] for r in rs if r["metrics"].get(task)]
            if not vals:
                continue
            out.append({
                "variant": variant, "architecture": arch,
                "architecture_name": ARCHITECTURE_NAMES[arch], "task": task,
                "n_runs": len(vals), "mean_auc": float(np.mean(vals)),
                "std_auc": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min_auc": float(np.min(vals)), "max_auc": float(np.max(vals)),
                "mean_positives": float(np.mean(pos)), "min_positives": int(np.min(pos)),
                "params": rs[0]["parameter_count"],
            })
    return out


def paired_comparison(runs_a: list[dict], runs_b: list[dict], task: str) -> dict:
    """Paired delta-AUC between two arms evaluated on the identical held-out
    rows, per seed, plus across-seed sign consistency."""
    per_seed = []
    for ra in runs_a:
        rb = next((r for r in runs_b if r["model_seed"] == ra["model_seed"]
                   and r["dataset"] == ra["dataset"]), None)
        if rb is None or not ra["metrics"].get(task) or not rb["metrics"].get(task):
            continue
        y = np.asarray(ra["_preds"][task]["y"])
        yb = np.asarray(rb["_preds"][task]["y"])
        if len(y) != len(yb) or not np.array_equal(y, yb):
            # Different held-out rows: not pairable. Happens only if the two arms
            # were run on different datasets, which the callers below never do.
            continue
        d = paired_delta_auc_ci(y, np.asarray(ra["_preds"][task]["p"]),
                                np.asarray(rb["_preds"][task]["p"]),
                                block=np.asarray(ra["_preds"][task]["block"]))
        d["model_seed"] = ra["model_seed"]
        per_seed.append(d)
    deltas = [d["delta"] for d in per_seed]
    return {"task": task, "per_seed": per_seed,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "sign_consistency": sign_consistency(deltas),
            "n_significant": sum(1 for d in per_seed if d["significant"])}


def variant_comparison(runs: list[dict], variant: str, reference: str, architecture: str) -> dict:
    """A variant's effect, read as a delta against its mandated reference variant.

    Enforces `REPORT_AGAINST`: a variant that has a reference CANNOT be reported
    standalone. The two arms are different DATASETS, so the held-out rows are not
    the same entities and a paired row-level bootstrap does not apply -- the
    comparison is between the two AUC distributions across seeds, reported with
    sign consistency, which is what the spec asks for."""
    if reference_for(variant) != reference:
        raise ValueError(f"variant {variant} must be reported against "
                         f"{reference_for(variant)}, not {reference}")
    out = {"variant": variant, "reference": reference, "architecture": architecture,
           "caveats": [VARIANT_CAVEATS[v] for v in (variant, reference) if v in VARIANT_CAVEATS],
           "tasks": {}}
    for task in TASKS:
        a = {r["model_seed"]: r for r in _by(runs, variant=variant, architecture=architecture)
             if r["metrics"].get(task)}
        b = {r["model_seed"]: r for r in _by(runs, variant=reference, architecture=architecture)
             if r["metrics"].get(task)}
        seeds = sorted(set(a) & set(b))
        if not seeds:
            continue
        deltas = [a[s]["metrics"][task]["roc_auc"] - b[s]["metrics"][task]["roc_auc"] for s in seeds]
        out["tasks"][task] = {
            "seeds": seeds, "deltas": deltas,
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "sign_consistency": sign_consistency(deltas),
            "variant_mean_auc": float(np.mean([a[s]["metrics"][task]["roc_auc"] for s in seeds])),
            "reference_mean_auc": float(np.mean([b[s]["metrics"][task]["roc_auc"] for s in seeds])),
            "variant_mean_positives": float(np.mean([a[s]["metrics"][task]["positives"] for s in seeds])),
        }
    return out


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'var':>3} {'architecture':<12} {'params':>9} " +
           " ".join(f"{t + ' AUC':>16} {'pos':>7}" for t in TASKS))
    print(hdr)
    print("-" * len(hdr))
    for variant in sorted({r["variant"] for r in rows}):
        for arch in ARCH_CONFIG:
            cells = []
            got = False
            for task in TASKS:
                r = next((x for x in rows if x["variant"] == variant
                          and x["architecture"] == arch and x["task"] == task), None)
                if r is None:
                    cells.append(f"{'—':>16} {'—':>7}")
                    continue
                got = True
                cells.append(f"{r['mean_auc']:>8.4f}±{r['std_auc']:<7.4f} "
                             f"{r['mean_positives']:>7,.0f}")
            if got:
                p = next(x["params"] for x in rows if x["variant"] == variant
                         and x["architecture"] == arch)
                print(f"{variant:>3} {ARCHITECTURE_NAMES[arch]:<12} {p:>9,} " + " ".join(cells))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_sanity(args) -> int:
    """Reproduce V1's recorded SHARE numbers on the byte-identical anchor.

    The anchor is Variant 0 at the **v1 preset** (800 suppliers, 15 snapshots,
    both sample rates 1.0) -- NOT `db/csv/v0_seed42`, which since Phase 6 holds
    Variant 0 at SPEC scale (4,000 suppliers, 40 snapshots) and is a different
    world. Only the v1-preset build is byte-identical to what V1 trained on, so
    only it can test the port against V1's numbers."""
    d = args.dataset
    print(f"sanity check: SHARE on {d}")
    bundles, manifest = load_bundles(d, cache_dir=CACHE_DIR)
    cfg = manifest["config"]
    print(f"  dataset: sup_n={cfg['sup_n']} snapshots={cfg['snapshots']} "
          f"delay_rate={cfg['delay_sample_rate']} shortage_rate={cfg['shortage_sample_rate']}")
    tr, va, te = split_bundles(bundles)
    print(f"  split: {len(tr)} train / {len(va)} val / {len(te)} test snapshots "
          f"(train ends {tr[-1].t0.date()}, test starts {te[0].t0.date()})")

    runs = []
    for seed in args.seeds:
        t = time.time()
        r = run_one(d, "rgcn_attn", seed, args.epochs, args.device, args.layers,
                    args.patience, bundles=bundles, manifest=manifest, progress=args.progress)
        r["variant"] = "0"
        runs.append(r)
        line = "  ".join(f"{t_}={r['metrics'][t_]['roc_auc']:.4f}" if r["metrics"].get(t_) else f"{t_}=n/a"
                         for t_ in TASKS)
        print(f"  seed {seed}: {line}  params={r['parameter_count']:,}  "
              f"({time.time() - t:.0f}s, best epoch {r['best_epoch']})", flush=True)

    print("\n  V1 recorded (HADES_v1/reports/info.md §4, matched-d=128, num_bases=10):")
    ref = V1_REFERENCE["rgcn_attn"]
    verdict_ok = runs[0]["parameter_count"] == ref["params"]
    print(f"    parameter count: ported {runs[0]['parameter_count']:,} vs V1 {ref['params']:,} "
          f"{'MATCH' if verdict_ok else 'MISMATCH'}")
    agg = aggregate(runs)
    for task in TASKS:
        row = next((a for a in agg if a["task"] == task), None)
        lo, hi = ref[task]
        if row is None:
            print(f"    {task:<9} ported n/a")
            continue
        inside = lo - 0.02 <= row["mean_auc"] <= hi + 0.02
        print(f"    {task:<9} ported {row['mean_auc']:.4f} ± {row['std_auc']:.4f}  "
              f"vs V1 {lo:.4f}–{hi:.4f}  "
              f"delta {row['mean_auc'] - (lo + hi) / 2:+.4f}  "
              f"{'within ±0.02' if inside else 'OUTSIDE ±0.02'}")
    save(args.out, runs, [], args)
    return 0


def mode_sweep(args) -> int:
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    runs = []
    for variant in variants:
        for data_seed in args.seeds:
            d = ensure_variant(variant, data_seed, args.auto_regenerate, args.csv_dir)
            t0 = time.time()
            bundles, manifest = load_bundles(d, cache_dir=CACHE_DIR)
            print(f"variant {variant} seed {data_seed}: {len(bundles)} snapshots "
                  f"loaded in {time.time() - t0:.0f}s", flush=True)
            for arch in archs:
                t = time.time()
                try:
                    r = run_one(d, arch, args.model_seed, args.epochs, args.device,
                                args.layers, args.patience, bundles=bundles,
                                manifest=manifest, progress=args.progress)
                except (RuntimeError, torch.OutOfMemoryError) as exc:
                    print(f"  {arch:<12} FAILED: {type(exc).__name__}: {exc}", flush=True)
                    continue
                r["variant"] = variant
                r["data_seed"] = data_seed
                # The data seed is the experimental seed here: each variant-seed is
                # a different generated world. The model seed is held fixed so the
                # variance measured across runs is the benchmark's, not the fit's.
                r["model_seed"] = data_seed
                runs.append(r)
                line = "  ".join(
                    f"{t_}={r['metrics'][t_]['roc_auc']:.4f}" if r["metrics"].get(t_) else f"{t_}=n/a"
                    for t_ in TASKS)
                print(f"  {ARCHITECTURE_NAMES[arch]:<12} {line}  "
                      f"({time.time() - t:.0f}s, {r['epochs_run']} epochs)", flush=True)
                save(args.out, runs, [], args)
            del bundles
    print()
    print_table(aggregate(runs))
    save(args.out, runs, [], args)
    return 0


def load_results(pattern: str) -> list[dict]:
    """Merge every results file matching `pattern` (the sweep writes one per
    variant so twelve variants can run as twelve processes). Runs are keyed by
    (variant, architecture, seed), so re-reading a partially-written file
    alongside a complete one cannot double-count."""
    import glob
    runs, seen = [], set()
    for path in sorted(glob.glob(pattern)):
        for r in json.load(open(path))["runs"]:
            key = (r.get("variant"), r["architecture"], r["model_seed"])
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def mode_report(args) -> int:
    runs = load_results(args.results)
    print(f"{len(runs)} runs from {args.results}\n")
    rows = aggregate(runs)
    print_table(rows)
    print("\nvariant effects, each against its reference variant "
          "(the spec's dependency table; Base+one-mechanism variants against Variant 0):")
    for variant in ALL_VARIANTS:
        reference = reference_for(variant)
        if reference is None:
            continue
        for arch in sorted({r["architecture"] for r in runs}):
            if not _by(runs, variant=variant, architecture=arch) or \
               not _by(runs, variant=reference, architecture=arch):
                continue
            c = variant_comparison(runs, variant, reference, arch)
            for task, t in c["tasks"].items():
                sc = t["sign_consistency"]
                print(f"  {variant} vs {reference} [{ARCHITECTURE_NAMES[arch]}] {task:<9} "
                      f"delta {t['mean_delta']:+.4f} ± {t['std_delta']:.4f}  "
                      f"sign {sc['positive']}+/{sc['negative']}- "
                      f"({'consistent' if sc['consistent'] else 'MIXED'})  "
                      f"pos {t['variant_mean_positives']:,.0f}")
            for cav in c["caveats"]:
                print(f"      caveat: {cav}")

    # Architecture comparisons: every arm against SHARE, PAIRED on the identical
    # held-out rows (same variant, same dataset seed), per seed, with the paired
    # block bootstrap and across-seed sign consistency.
    baseline = "rgcn_attn"
    variants = sorted({r["variant"] for r in runs
                       if _by(runs, variant=r["variant"], architecture=baseline)})
    for variant in variants:
        share = _by(runs, variant=variant, architecture=baseline)
        others = sorted({r["architecture"] for r in _by(runs, variant=variant)} - {baseline})
        if not others:
            continue
        print(f"\narchitecture comparisons on variant {variant} "
              f"(paired delta-AUC vs SHARE, + = SHARE better):")
        for arch in others:
            for task in TASKS:
                c = paired_comparison(share, _by(runs, variant=variant, architecture=arch), task)
                if c["mean_delta"] is None:
                    continue
                sc = c["sign_consistency"]
                print(f"  SHARE - {ARCHITECTURE_NAMES[arch]:<10} {task:<9} "
                      f"delta {c['mean_delta']:+.4f}  "
                      f"sign {sc['positive']}+/{sc['negative']}- "
                      f"({'consistent' if sc['consistent'] else 'MIXED'})  "
                      f"{c['n_significant']}/{sc['n']} seeds significant")
        if variant in VARIANT_CAVEATS:
            print(f"      caveat: {VARIANT_CAVEATS[variant]}")
    return 0


def save(path: str, runs: list[dict], comparisons: list[dict], args) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {
        "manifest": {
            "git_commit": git_commit(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "device": args.device,
            "argv": sys.argv, "arch_config": ARCH_CONFIG,
            "split_fractions": [0.4, 0.2, 0.4],
            "report_against": REPORT_AGAINST,
        },
        "runs": runs, "aggregate": aggregate(runs), "comparisons": comparisons,
    }
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sanity", "sweep", "report"])
    ap.add_argument("--dataset", default=os.path.join(REPO, "db", "csv_v1preset_superseded", "v0_seed42"),
                    help="sanity mode: the v1-preset Variant 0 build (the V1 anchor)")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--archs", default=",".join(ARCH_CONFIG))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="sweep mode: model-init seed, held fixed so across-run variance "
                         "is the benchmark's (different generated worlds), not the fit's")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience; omit to run all epochs as V1 did")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv-dir", default="",
                    help="directory of v<variant>_seed<n> builds (default db/csv, the spec-scale "
                         "corpus). Point at another configuration's builds to sweep at that scale.")
    ap.add_argument("--auto-regenerate", action="store_true",
                    help="generate non-retained seeds via db/regenerate_seed.py instead of failing")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--results", default="",
                    help="report mode: a results JSON, or a glob over several "
                         "(e.g. 'out/sweep_v1scale_*.json')")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.mode == "sanity":
        return mode_sanity(args)
    if args.mode == "sweep":
        return mode_sweep(args)
    return mode_report(args)


if __name__ == "__main__":
    sys.exit(main())
