#!/usr/bin/env python3
"""
The retrieval redesign's stop conditions, evaluated in code rather than by eye.

`STEP4B` structures the session as a decision tree with hard stops: Stage 1's ceiling must
be measured **against its own reproduction floor** before Stage 2 may be built, and Stage 2
must be compared against both chance and Stage 1's ceiling before Stage 3 or 4 may be. This
script applies that arithmetic to the run directory and prints the verdict, so the go/no-go
is a computed result with its inputs on screen rather than a judgement call made after
looking at a table.

**The bar, stated once.** A metric clears when *all three* hold:

1. it beats chance in the right direction by more than the **worst** single reproduction
   gap observed for that same metric and group set (max abs, not mean abs -- the same
   conservative choice `reports/layer3_testing.md` §4 made when it judged deltas against
   0.0082 rather than 0.0026);
2. the per-seed values agree in sign; and
3. it does so on at least 4 of 5 dataset seeds individually.

Sign consistency alone is explicitly NOT sufficient, per §4: the floor is a property of
the training process, so all seeds share it and five same-signed sub-floor results are
five draws from one biased process, not five confirmations.

    python3 ml/stage_gate.py --results out/t2redesign --tag-pairs r1,r2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.retrieval_metrics import METRICS  # noqa: E402

CONTRASTIVE = "rgcn_attn_t2_contrastive"
CONTROL = "rgcn_attn_variant_a_transformer2"


def load(results_dir: str, field: str) -> dict:
    runs: dict = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        blob = json.load(open(path))
        if "arch" not in blob or not blob.get(field):
            continue
        runs.setdefault((blob["arch"], blob["variant"], blob.get("tag", "")), {})[
            blob["seed"]] = blob[field]
    return runs


def floors(runs: dict, arch: str, variant: str, tags: tuple[str, str]) -> dict:
    """Worst and mean absolute movement per (group set, metric) between two
    identically-configured replicates."""
    a, b = runs.get((arch, variant, tags[0]), {}), runs.get((arch, variant, tags[1]), {})
    out: dict = {}
    for seed in sorted(set(a) & set(b)):
        for key in a[seed]:
            if key in ("n_nodes", "top_k", "chance_discovery_rate"):
                continue
            if key not in b[seed]:
                continue
            for m in METRICS:
                out.setdefault((key, m), []).append(b[seed][key][m] - a[seed][key][m])
    return {k: {"mean_abs": float(np.abs(v).mean()), "max_abs": float(np.abs(v).max()),
                "n": len(v)} for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join("out", "t2redesign"))
    ap.add_argument("--tag-pairs", default="r1,r2")
    ap.add_argument("--checkpoint", default="final", choices=("best", "final"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tags = tuple(t.strip() for t in args.tag_pairs.split(","))
    field = "retrieval" if args.checkpoint == "best" else "retrieval_final_epoch"
    runs = load(args.results, field)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    dump: list = []

    for variant in variants:
        fl = floors(runs, CONTRASTIVE, variant, tags)
        for arch, label in ((CONTRASTIVE, "contrastive (privileged)"), (CONTROL, "control")):
            per_seed = runs.get((arch, variant, tags[0]), {})
            if not per_seed:
                continue
            print(f"\n### Variant {variant} -- {label} "
                  f"({args.checkpoint}-checkpoint, {len(per_seed)} seeds, "
                  f"floor from {tags[0]}/{tags[1]})\n")
            print("| group set | metric | observed | chance | ratio | obs-chance | "
                  "floor (max abs) | seeds beating chance | verdict |")
            print("|---|---|---|---|---|---|---|---|---|")
            keys = [k for k in next(iter(per_seed.values()))
                    if k not in ("n_nodes", "top_k", "chance_discovery_rate")]
            for key in keys:
                for m in METRICS:
                    vals = [per_seed[s][key][m] for s in sorted(per_seed) if key in per_seed[s]]
                    chs = [per_seed[s][key]["chance"][m] for s in sorted(per_seed)
                           if key in per_seed[s]]
                    if not vals:
                        continue
                    obs, ch = float(np.mean(vals)), float(np.mean(chs))
                    gap = obs - ch
                    f = fl.get((key, m), {}).get("max_abs")
                    n_beat = sum(1 for v, c in zip(vals, chs) if v > c)
                    sign_ok = all(v > c for v, c in zip(vals, chs)) or \
                              all(v < c for v, c in zip(vals, chs))
                    clears = (f is not None and gap > f and sign_ok and n_beat >= 4)
                    verdict = "CLEARS" if clears else (
                        "at chance" if abs(gap) <= (f or 0) else "below chance"
                        if gap < 0 else "sub-floor")
                    print(f"| {key} | {m} | {obs:.4f} | {ch:.4f} | "
                          f"{obs/ch:.2f}x | {gap:+.4f} | "
                          f"{'--' if f is None else f'{f:.4f}'} | {n_beat}/{len(vals)} | "
                          f"**{verdict}** |")
                    dump.append({"variant": variant, "arch": arch, "group_set": key,
                                 "metric": m, "observed": obs, "chance": ch,
                                 "gap": gap, "floor_max_abs": f, "n_seeds": len(vals),
                                 "n_beating_chance": n_beat, "sign_consistent": sign_ok,
                                 "clears": clears, "per_seed": vals})

    # ---- the stop condition itself ------------------------------------------
    print("\n\n## Stage 1 stop condition\n")
    print("The brief: *if contrastive retrieval -- with privileged supervision, the "
          "easiest possible\nversion of this problem -- cannot clear chance by more than "
          "its own reproduction floor,\nstop the entire line of investigation.*\n")
    sup = [d for d in dump if d["arch"] == CONTRASTIVE and d["group_set"].endswith("supervised")]
    held = [d for d in dump if d["arch"] == CONTRASTIVE and d["group_set"].endswith("held_out")]
    ctl = [d for d in dump if d["arch"] == CONTROL and d["group_set"] in ("A", "B")]
    for name, rows in (("supervised A/B groups (the ceiling the brief defines)", sup),
                       ("held-out A/B groups (generalisation)", held),
                       ("control arm, A/B (the section 2 replication)", ctl)):
        n = sum(1 for d in rows if d["clears"])
        print(f"* {name}: **{n} of {len(rows)}** metric/variant cells clear chance by "
              f"more than the floor.")
    opened = sum(1 for d in sup if d["clears"]) > 0
    print(f"\n**Gate: {'OPEN -- Stage 2 may be built' if opened else 'CLOSED -- stop'}.**")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dump, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
