#!/usr/bin/env python3
"""Phase 6 — benchmark evaluation protocol.

Implements the statistical machinery
`docs/00_Benchmark_Specification.md` §"Benchmark Evaluation Protocol" mandates,
as reusable functions plus a CLI that reports split composition for a generated
dataset:

  * temporal train / validation / test splits (identical across all variants)
  * paired bootstrap confidence intervals
  * sign consistency across seeds
  * mean and standard deviation

This module deliberately does NOT train models. It supplies the protocol a
submission must evaluate under, and verifies that a generated variant actually
carries usable labels in every split -- a variant whose test split has no
positives cannot support any comparison, however good the model is.

Usage:
    python3 db/benchmark_eval.py --dataset db/csv/vK_seed42
    python3 db/benchmark_eval.py --dataset db/csv/v0_seed42 --check-splits
"""
import argparse, csv, gzip, json, math, os, random, statistics, sys

# Split proportions. The spec fixes 60/20/20 by snapshot index and requires the
# SAME cutoffs across every variant, so a variant may change what happens inside
# a snapshot window but never which snapshots fall in which split.
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20


# ---------------------------------------------------------------- splits
def temporal_splits(t0s, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Chronological split of snapshot t0s into (train, validation, test).

    Purely positional, so it is identical across variants by construction -- the
    property that makes cross-variant comparison valid."""
    t0s = sorted(t0s)
    n = len(t0s)
    n_tr = int(round(n * train_frac))
    n_va = int(round(n * val_frac))
    return t0s[:n_tr], t0s[n_tr:n_tr + n_va], t0s[n_tr + n_va:]


# ---------------------------------------------------------------- statistics
def paired_bootstrap(delta_per_unit, reps=2000, alpha=0.05, seed=12345):
    """Percentile CI for a PAIRED difference (model A minus model B on the same
    evaluation units). Pairing is what makes this more powerful than comparing two
    independent CIs -- it removes the unit-level variance both models share.

    `delta_per_unit` is one delta per evaluation unit (e.g. per test snapshot, or
    per bootstrap-resampled AUC difference)."""
    if len(delta_per_unit) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(delta_per_unit)
    means = []
    for _ in range(reps):
        means.append(sum(delta_per_unit[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * reps)]
    hi = means[int((1 - alpha / 2) * reps) - 1]
    return statistics.fmean(delta_per_unit), lo, hi


def sign_consistency(deltas):
    """Fraction of seeds whose delta shares the majority sign, plus that sign.

    The spec requires an improvement to 'maintain directional consistency across
    seeds'. A delta whose CI excludes zero but whose sign flips across seeds is
    not an improvement -- this is the check that catches it."""
    if not deltas:
        return 0.0, 0
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    if pos == neg:
        return 0.5, 0
    sign = 1 if pos > neg else -1
    return max(pos, neg) / len(deltas), sign


def summarize(values):
    """Mean and standard deviation, as the protocol requires alongside every CI."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def significant(deltas, reps=2000, alpha=0.05, min_consistency=0.8):
    """The spec's full criterion in one call: an architectural improvement counts
    only when it is BOTH statistically distinguishable from its confidence interval
    AND directionally consistent across seeds."""
    mean, lo, hi = paired_bootstrap(deltas, reps=reps, alpha=alpha)
    cons, sign = sign_consistency(deltas)
    ci_excludes_zero = (lo > 0) or (hi < 0)
    return {
        "mean": mean, "ci_low": lo, "ci_high": hi,
        "ci_excludes_zero": ci_excludes_zero,
        "sign_consistency": cons, "sign": sign,
        "significant": bool(ci_excludes_zero and cons >= min_consistency),
    }


# ---------------------------------------------------------------- dataset inspection
def _open(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def _resolve(d, name):
    for cand in (os.path.join(d, name + ".gz"), os.path.join(d, name)):
        if os.path.exists(cand):
            return cand
    return None


def split_report(dataset_dir):
    """Label counts per task per split for a generated variant."""
    man_path = os.path.join(dataset_dir, "resolved_config.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    snap_path = _resolve(dataset_dir, "graph_snapshots.csv")
    lbl_path = _resolve(dataset_dir, "training_labels.csv")
    if not snap_path or not lbl_path:
        sys.exit(f"{dataset_dir}: missing graph_snapshots/training_labels")

    snap_t0, t0s = {}, []
    with _open(snap_path) as f:
        for r in csv.DictReader(f):
            snap_t0[r["id"]] = r["t0"]
            t0s.append(r["t0"])
    tr, va, te = temporal_splits(t0s)
    where = {t: "train" for t in tr}
    where.update({t: "val" for t in va})
    where.update({t: "test" for t in te})

    counts = {}
    with _open(lbl_path) as f:
        for r in csv.DictReader(f):
            sp = where.get(snap_t0.get(r["snapshot_id"], ""), "?")
            k = (r["task"], sp)
            c = counts.setdefault(k, [0, 0])
            c[0] += 1
            c[1] += int(r["label"] == "true")
    return man, (tr, va, te), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a generated variant directory")
    ap.add_argument("--check-splits", action="store_true",
                    help="exit non-zero if any split has zero positives for a task")
    args = ap.parse_args()

    man, (tr, va, te), counts = split_report(args.dataset)
    if man:
        print(f"{man.get('benchmark_version','?')}  variant {man.get('variant','?')}  "
              f"seed {man.get('generation_seed','?')}  "
              f"mechanisms {man.get('mechanisms_enabled') or '(base)'}")
    print(f"\ntemporal splits ({len(tr)}/{len(va)}/{len(te)} snapshots, "
          f"{TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{1-TRAIN_FRAC-VAL_FRAC:.0%})")
    print(f"  train      {tr[0][:10]} .. {tr[-1][:10]}")
    print(f"  validation {va[0][:10]} .. {va[-1][:10]}")
    print(f"  test       {te[0][:10]} .. {te[-1][:10]}")
    print("\n  these cutoffs are positional and therefore identical across all variants,")
    print("  which is what makes cross-variant comparison valid\n")

    print(f"{'task':<10}{'split':<8}{'n':>10}{'positives':>11}{'rate':>9}")
    empty = []
    for task in ("delay", "shortage", "impact"):
        for sp in ("train", "val", "test"):
            n, p = counts.get((task, sp), [0, 0])
            rate = f"{p/n*100:.2f}%" if n else "-"
            print(f"{task:<10}{sp:<8}{n:>10,}{p:>11,}{rate:>9}")
            if sp == "test" and p == 0:
                empty.append(task)
    if empty:
        print(f"\nWARNING: no test-split positives for {', '.join(empty)} — "
              f"this variant cannot support a comparison on those tasks.")
    if args.check_splits and empty:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
