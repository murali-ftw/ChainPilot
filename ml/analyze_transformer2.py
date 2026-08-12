#!/usr/bin/env python3
"""Aggregate the Layer 3 (Transformer 2) sweep into the two halves V1's
`reports/layer3.md` always reported together and never let collapse into one number:

1. **Discovery quality** -- percentile rank and pool-membership discovery rate against
   chance, broken out per V2 group type (A = causally coupled via Mechanism D,
   B = correlated but alpha=0, the V1 `H_POLYMER` replica, C = decoy).
2. **Downstream AUC against the no-T2 Variant A baseline**, per task, with across-seed
   sign consistency and a paired block bootstrap on the delta.

The bootstrap needs per-row test predictions, not AUC scalars, so it only covers runs
launched with `--save-preds`; the sign-consistency column uses every seed present. Both
coverages are printed, because a delta reported without saying how many seeds it rests on
is the failure mode this project's whole statistics convention exists to prevent.

    python3 ml/analyze_transformer2.py --results out/t2mid
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.evaluate import paired_delta_auc_ci, sign_consistency  # noqa: E402
from ml.models.depth import TASKS  # noqa: E402

BASELINE = "rgcn_attn_rung5_a"
ARM_ORDER = [BASELINE, "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
             "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn"]
ARM_LABEL = {BASELINE: "Variant A (no T2)", "rgcn_attn_variant_a_transformer2": "T2 plain",
             "rgcn_attn_t2_confidence": "T2 confidence", "rgcn_attn_t2_trustgate": "T2 trustgate",
             "rgcn_attn_t2_crossattn": "T2 crossattn"}


def load(results_dir: str) -> dict:
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        blob = json.load(open(path))
        if "arch" not in blob:
            continue
        runs[(blob["variant"], blob["arch"], blob["seed"])] = blob
    return runs


def preds_for(results_dir: str, variant: str, arch: str, seed: int, task: str):
    path = os.path.join(results_dir, f"{arch}_v{variant}_seed{seed}_preds.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if f"{task}_y" not in z:
            return None
        return z[f"{task}_y"], z[f"{task}_p"], z[f"{task}_block"]


def fmt(x, nd=4):
    return "--" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join("out", "t2mid"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--out", default=None, help="optional JSON dump of everything printed")
    args = ap.parse_args()

    runs = load(args.results)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    arms = [a for a in ARM_ORDER if any(k[1] == a for k in runs)]
    dump: dict = {"auc": [], "delta": [], "discovery": [], "trust": []}

    # ---- AUC table -------------------------------------------------------------
    print("\n## AUC (mean +/- std over dataset seeds)\n")
    print("| var | arm | params | seeds |" + "".join(f" {t} | pos |" for t in TASKS))
    print("|---|---|---|---|" + "---|---|" * len(TASKS))
    for variant in variants:
        for arm in arms:
            rs = [runs[k] for k in sorted(runs) if k[0] == variant and k[1] == arm]
            if not rs:
                continue
            row = [f"| {variant} | {ARM_LABEL.get(arm, arm)} | {rs[0]['params']:,} | {len(rs)} |"]
            rec = {"variant": variant, "arm": arm, "n_seeds": len(rs),
                   "seeds": [r["seed"] for r in rs], "params": rs[0]["params"]}
            for task in TASKS:
                v = [r["auc"][task] for r in rs if r["auc"].get(task) is not None]
                pos = int(np.mean([r["auc"][f"{task}_positives"] for r in rs]))
                row.append(f" {np.mean(v):.4f} +/- {np.std(v):.4f} | {pos:,} |" if v else " -- | -- |")
                rec[task] = {"mean": float(np.mean(v)) if v else None,
                             "std": float(np.std(v)) if v else None, "positives": pos}
            print("".join(row))
            dump["auc"].append(rec)

    # ---- Delta vs the no-T2 baseline, paired by dataset seed -------------------
    print("\n## Delta vs Variant A (no T2), paired by dataset seed\n")
    print("| var | arm | task | mean delta | seeds | sign | consistent | "
          "bootstrap seeds w/ CI excluding 0 |")
    print("|---|---|---|---|---|---|---|---|")
    for variant in variants:
        for arm in arms:
            if arm == BASELINE:
                continue
            for task in TASKS:
                deltas, boots = [], []
                for seed in sorted({k[2] for k in runs if k[0] == variant}):
                    b = runs.get((variant, BASELINE, seed))
                    a = runs.get((variant, arm, seed))
                    if not b or not a:
                        continue
                    if a["auc"].get(task) is None or b["auc"].get(task) is None:
                        continue
                    deltas.append(a["auc"][task] - b["auc"][task])
                    pa = preds_for(args.results, variant, arm, seed, task)
                    pb = preds_for(args.results, variant, BASELINE, seed, task)
                    if pa is not None and pb is not None and np.array_equal(pa[0], pb[0]):
                        boots.append(paired_delta_auc_ci(pa[0], pa[1], pb[1], block=pa[2]))
                if not deltas:
                    continue
                sc = sign_consistency(deltas)
                sig = sum(1 for c in boots if c.get("significant"))
                print(f"| {variant} | {ARM_LABEL.get(arm, arm)} | {task} | "
                      f"{np.mean(deltas):+.4f} | {len(deltas)} | {sc['positive']}+/{sc['negative']}- | "
                      f"{'YES' if sc['consistent'] else 'no'} | {sig}/{len(boots)} |")
                dump["delta"].append({
                    "variant": variant, "arm": arm, "task": task,
                    "mean_delta": float(np.mean(deltas)), "deltas": deltas,
                    "sign_consistency": sc, "n_bootstrap": len(boots),
                    "n_significant": sig, "bootstrap": boots})

    # ---- Discovery quality per group type --------------------------------------
    print("\n## Discovery quality (pooled over seeds; chance rank = 0.500)\n")
    print("| var | arm | seeds | type | percentile rank | discovery rate | chance | ratio | pairs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for variant in variants:
        for arm in arms:
            rs = [runs[k] for k in sorted(runs) if k[0] == variant and k[1] == arm
                  and runs[k].get("discovery")]
            if not rs:
                continue
            chance = float(np.mean([r["discovery"]["chance_discovery_rate"] for r in rs]))
            for t in "ABC":
                pct = [r["discovery"][f"pct_{t}"] for r in rs if f"pct_{t}" in r["discovery"]]
                rate = [r["discovery"][f"rate_{t}"] for r in rs if f"rate_{t}" in r["discovery"]]
                npairs = [r["discovery"][f"n_{t}"] for r in rs if f"n_{t}" in r["discovery"]]
                if not pct:
                    continue
                ratio = float(np.mean(rate)) / chance if chance else float("nan")
                print(f"| {variant} | {ARM_LABEL.get(arm, arm)} | {len(pct)} | {t} | "
                      f"{np.mean(pct):.4f} | {np.mean(rate):.4f} | {chance:.4f} | "
                      f"{ratio:.2f}x | {int(np.mean(npairs)):,} |")
                dump["discovery"].append({
                    "variant": variant, "arm": arm, "type": t, "n_seeds": len(pct),
                    "pct_rank": float(np.mean(pct)), "rate": float(np.mean(rate)),
                    "chance": chance, "ratio": ratio, "pairs": int(np.mean(npairs))})

    # ---- Trust-gate stability (V1 layer3.md section 2.7) -----------------------
    print("\n## Trust-gate stability, per run\n")
    print("| var | seed | mean trust | std | pos/neg/~0 | corr(|trust|,deg) | "
          "type A | type B | type C | non-member |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for variant in variants:
        for seed in sorted({k[2] for k in runs if k[0] == variant}):
            r = runs.get((variant, "rgcn_attn_t2_trustgate", seed))
            if not r or not r.get("trust"):
                continue
            t = r["trust"]
            print(f"| {variant} | {seed} | {t['mean_trust']:+.4f} | {t['std_trust']:.4f} | "
                  f"{t['share_pos']*100:.1f}% / {t['share_neg']*100:.1f}% / "
                  f"{t['share_zero']*100:.1f}% | {t['corr_abs_trust_degree']:+.4f} | "
                  + " | ".join(fmt(t.get(f"mean_trust_type_{x}")) for x in "ABC")
                  + f" | {fmt(t.get('mean_trust_non_member'))} |")
            dump["trust"].append(dict(variant=variant, seed=seed, **t))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(dump, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
