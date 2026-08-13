#!/usr/bin/env python3
"""Measure the run-to-run reproduction floor: how far apart two trainings of the
**identical** configuration land.

Seeds 42/43 of the Layer 3 mid-scale grid were trained twice -- same architecture,
same dataset, same model seed, same 40-epoch budget -- once before `--save-preds`
existed and once after. Nothing differs but the process environment (thread count,
and therefore CPU float reduction order; scatter/index_add in message passing is not
order-deterministic either).

That makes the pairs a direct measurement of the noise floor, which is the number that
decides whether any delta in this session's tables is interpretable. An arm-vs-baseline
delta smaller than the floor is not evidence of anything, no matter how sign-consistent
it looks across seeds -- because the seeds share the floor.

**Extended for the retrieval redesign (report section 9).** The original version measured
a floor for downstream AUC only, which left the gap that session's own recommendation
called out: **retrieval quality had no floor of its own**, so there was no standard
against which a percentile rank of 0.51 or a Recall@K of 0.28 could be judged. With
`--tag-pairs` this script reads two identically-configured replicates of the same run
out of one directory (`ml/run_retrieval_redesign.py --tag r1` / `--tag r2`) and reports
how far percentile rank, discovery rate, Recall@K, Precision@K and MRR move with nothing
changed at all. A Stage 1 or Stage 2 result that does not clear its own floor by a
healthy margin is not a finding -- the same standard section 4 applied to every AUC delta.

    python3 ml/reproduction_floor.py --before out/t2mid_prerun --after out/t2mid
    python3 ml/reproduction_floor.py --results out/t2redesign --tag-pairs r1,r2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.depth import TASKS  # noqa: E402
from ml.retrieval_metrics import METRICS  # noqa: E402


def auc_floor_from_tags(results_dir: str, tags: tuple[str, str]) -> None:
    """The section 4 AUC floor, re-measured from replicate pairs in one directory.

    Section 4's own pairs came from two runs in different process environments on CPU.
    This session runs serially on MPS, one environment throughout, so its floor is a
    different number measured on different hardware and the two must not be quoted
    interchangeably -- which is the whole reason it is re-measured here rather than
    carried over.
    """
    by_key: dict[tuple, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        blob = json.load(open(path))
        if "arch" not in blob:
            continue
        by_key.setdefault((blob["arch"], blob["variant"], blob["seed"]), {})[
            blob.get("tag", "")] = blob
    pairs = [(k, v[tags[0]], v[tags[1]]) for k, v in sorted(by_key.items())
             if tags[0] in v and tags[1] in v]
    if not pairs:
        return
    print(f"\n## AUC reproduction floor -- {len(pairs)} identically-configured pairs "
          f"(this session's device)\n")
    print("| arch | task | mean signed | mean abs | max abs | std | pairs |")
    print("|---|---|---|---|---|---|---|")
    for arch in sorted({k[0] for k, _, _ in pairs}):
        for task in TASKS:
            d = [b["auc"][task] - a["auc"][task] for k, a, b in pairs
                 if k[0] == arch and a["auc"].get(task) is not None
                 and b["auc"].get(task) is not None]
            if not d:
                continue
            v = np.array(d)
            print(f"| {arch} | {task} | {v.mean():+.4f} | {np.abs(v).mean():.4f} | "
                  f"{np.abs(v).max():.4f} | {v.std():.4f} | {len(v)} |")


def retrieval_floor(results_dir: str, tags: tuple[str, str], checkpoint: str) -> int:
    """Floor for the retrieval metrics, from replicate pairs in one directory.

    A "pair" is two trainings of the identical architecture, dataset, model seed and
    epoch budget, differing in nothing a configuration could name. Reported per group
    type and per supervised/held-out split, because those are the units the result tables
    report and a floor pooled across them would be the wrong yardstick for any of them.
    """
    field = "retrieval" if checkpoint == "best" else "retrieval_final_epoch"
    by_key: dict[tuple, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        blob = json.load(open(path))
        if "arch" not in blob or not blob.get(field):
            continue
        by_key.setdefault((blob["arch"], blob["variant"], blob["seed"]), {})[
            blob.get("tag", "")] = blob

    pairs = [(k, v[tags[0]], v[tags[1]]) for k, v in sorted(by_key.items())
             if tags[0] in v and tags[1] in v]
    if not pairs:
        print(f"\n(no {tags[0]}/{tags[1]} replicate pairs with a `{field}` blob "
              f"in {results_dir})")
        return 0

    group_sets: list[str] = []
    for _, a, _b in pairs:
        for key in a[field]:
            if key not in ("n_nodes", "top_k", "chance_discovery_rate") and key not in group_sets:
                group_sets.append(key)

    print(f"\n## Retrieval reproduction floor -- {len(pairs)} identically-configured pairs "
          f"({checkpoint}-checkpoint weights)\n")
    print("| arch | group set | " + " | ".join(
        f"{m} mean abs | max abs" for m in METRICS) + " | pairs |")
    print("|---|---|" + "---|---|" * len(METRICS) + "---|")
    out: dict = {}
    for arch in sorted({k[0] for k, _, _ in pairs}):
        for key in group_sets:
            deltas = {m: [] for m in METRICS}
            n = 0
            for k, a, b in pairs:
                if k[0] != arch or key not in a[field] or key not in b[field]:
                    continue
                n += 1
                for m in METRICS:
                    deltas[m].append(b[field][key][m] - a[field][key][m])
            if not n:
                continue
            cells = []
            for m in METRICS:
                v = np.abs(np.array(deltas[m]))
                cells.append(f" {v.mean():.4f} | {v.max():.4f} |")
                out.setdefault(arch, {}).setdefault(key, {})[m] = {
                    "mean_abs": float(v.mean()), "max_abs": float(v.max()),
                    "signed": deltas[m]}
            print(f"| {arch} | {key} |" + "".join(cells) + f" {n} |")
    return len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default=os.path.join("out", "t2mid_prerun"))
    ap.add_argument("--after", default=os.path.join("out", "t2mid"))
    ap.add_argument("--results", default=None,
                    help="directory of replicate runs for the RETRIEVAL floor")
    ap.add_argument("--tag-pairs", default="r1,r2",
                    help="the two --tag values identifying the replicates")
    args = ap.parse_args()

    if args.results:
        tags = tuple(t.strip() for t in args.tag_pairs.split(","))
        auc_floor_from_tags(args.results, tags)
        for ck in ("best", "final"):
            retrieval_floor(args.results, tags, ck)
        return 0

    rows, per_task = [], {t: [] for t in TASKS}
    for path in sorted(glob.glob(os.path.join(args.before, "*.json"))):
        name = os.path.basename(path)
        after_path = os.path.join(args.after, name)
        if not os.path.exists(after_path):
            continue
        a, b = json.load(open(path)), json.load(open(after_path))
        row = {"run": name[:-5], "variant": a["variant"], "seed": a["seed"], "arch": a["arch"]}
        for task in TASKS:
            x, y = a["auc"].get(task), b["auc"].get(task)
            if x is None or y is None:
                continue
            row[task] = y - x
            per_task[task].append(y - x)
        rows.append(row)

    print(f"\n## Reproduction floor -- {len(rows)} identically-configured pairs\n")
    print("| run | " + " | ".join(TASKS) + " |")
    print("|---|" + "---|" * len(TASKS))
    for r in rows:
        print(f"| {r['run']} | " + " | ".join(
            f"{r[t]:+.4f}" if t in r else "--" for t in TASKS) + " |")

    print("\n| task | mean signed | mean abs | max abs | std |")
    print("|---|---|---|---|---|")
    for task in TASKS:
        v = np.array(per_task[task])
        if not len(v):
            continue
        print(f"| {task} | {v.mean():+.4f} | {np.abs(v).mean():.4f} | "
              f"{np.abs(v).max():.4f} | {v.std():.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
