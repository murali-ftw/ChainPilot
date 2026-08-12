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
from ml.retrieval_metrics import METRICS  # noqa: E402

BASELINE = "rgcn_attn_rung5_a"
ARM_ORDER = [BASELINE, "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
             "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn", "rgcn_attn_t2_contrastive"]
ARM_LABEL = {BASELINE: "Variant A (no T2)", "rgcn_attn_variant_a_transformer2": "T2 plain",
             "rgcn_attn_t2_confidence": "T2 confidence", "rgcn_attn_t2_trustgate": "T2 trustgate",
             "rgcn_attn_t2_crossattn": "T2 crossattn",
             "rgcn_attn_t2_contrastive": "T2 contrastive (privileged)"}
METRIC_LABEL = {"pct_rank": "percentile rank", "discovery_rate": "discovery rate",
                "recall_at_k": "Recall@K", "precision_at_k": "Precision@K", "mrr": "MRR"}


def load(results_dir: str, tag: str | None = None) -> dict:
    """Runs keyed by (variant, arch, seed).

    `tag` selects one replicate when a directory holds several identically-configured
    ones (`ml/run_retrieval_redesign.py --tag`). Without it, a directory of replicates
    would silently collapse to whichever file sorted last, which is exactly the kind of
    quiet collision the `ml/data/loader.py` cache-key bug already cost this project once.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        blob = json.load(open(path))
        if "arch" not in blob:
            continue
        if tag is not None and blob.get("tag", "") != tag:
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


def retrieval_tables(runs: dict, variants: list[str], checkpoint: str, dump: dict) -> None:
    """The full IR metric set for runs carrying a `retrieval` blob
    (`ml/run_retrieval_redesign.py`): percentile rank and pool discovery rate -- the two
    `reports/layer3_testing.md` §2 already reported, on the same definitions -- plus
    Recall@K, Precision@K and MRR.

    Every metric is printed **against its own chance baseline** and as a ratio to it,
    because the five have wildly different natural scales: a percentile rank of 0.52 and
    a Recall@64 of 0.28 sound similar and are not remotely comparable claims. Group types
    A/B/C are reported separately, and where a run carries a supervised/held-out split
    that is reported separately too -- pooling those would hide the single most important
    contrast this session produces.

    `checkpoint` selects `retrieval` (best-validation-AUC weights, what the AUC table
    rests on) or `retrieval_final_epoch` (last-epoch weights).
    """
    field = "retrieval" if checkpoint == "best" else "retrieval_final_epoch"
    keys_seen: list[str] = []
    for k in runs:
        for key in runs[k].get(field, {}) or {}:
            if key not in ("n_nodes", "top_k", "chance_discovery_rate") and key not in keys_seen:
                keys_seen.append(key)
    if not keys_seen:
        return
    order = [k for k in ("A", "B", "C") if k in keys_seen] + \
            sorted(k for k in keys_seen if "|" in k)

    print(f"\n## Retrieval quality -- full IR metric set ({checkpoint}-checkpoint weights)\n")
    print("| var | arm | seeds | group set | " + " | ".join(
        f"{METRIC_LABEL[m]} | chance | ratio" for m in METRICS) + " | pairs |")
    print("|---|---|---|---|" + "---|---|---|" * len(METRICS) + "---|")
    for variant in variants:
        for arm in [a for a in ARM_ORDER if any(k[1] == a for k in runs)]:
            rs = [runs[k] for k in sorted(runs)
                  if k[0] == variant and k[1] == arm and runs[k].get(field)]
            if not rs:
                continue
            for key in order:
                cells, rec = [], {"variant": variant, "arm": arm, "group_set": key,
                                  "n_seeds": 0, "checkpoint": checkpoint}
                present = [r[field][key] for r in rs if key in r[field]]
                if not present:
                    continue
                rec["n_seeds"] = len(present)
                for m in METRICS:
                    obs = float(np.mean([p[m] for p in present]))
                    ch = float(np.mean([p["chance"][m] for p in present]))
                    cells.append(f" {obs:.4f} | {ch:.4f} | {obs/ch:.2f}x |" if ch
                                 else f" {obs:.4f} | -- | -- |")
                    rec[m] = {"observed": obs, "chance": ch,
                              "ratio": (obs / ch) if ch else None,
                              "per_seed": [p[m] for p in present]}
                pairs = int(np.mean([p["n_pairs"] for p in present]))
                print(f"| {variant} | {ARM_LABEL.get(arm, arm)} | {len(present)} | {key} |"
                      + "".join(cells) + f" {pairs:,} |")
                rec["pairs"] = pairs
                dump.setdefault("retrieval", []).append(rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join("out", "t2mid"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--out", default=None, help="optional JSON dump of everything printed")
    ap.add_argument("--tag", default=None,
                    help="select one replicate when the results directory holds several "
                         "identically-configured ones (e.g. r1, r2)")
    args = ap.parse_args()

    runs = load(args.results, tag=args.tag)
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

    # ---- Retrieval quality, full IR metric set (session 2 / report section 9) --
    retrieval_tables(runs, variants, "best", dump)
    retrieval_tables(runs, variants, "final", dump)

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
