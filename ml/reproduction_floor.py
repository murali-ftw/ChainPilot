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

    python3 ml/reproduction_floor.py --before out/t2mid_prerun --after out/t2mid
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default=os.path.join("out", "t2mid_prerun"))
    ap.add_argument("--after", default=os.path.join("out", "t2mid"))
    args = ap.parse_args()

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
