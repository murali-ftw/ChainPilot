#!/usr/bin/env python3
"""
Format §10's results as the markdown tables that go into `reports/layer3_testing.md`.

Kept separate from `ml/run_hypothesis_module.py` for the reason every analysis script in this
project is: the run is expensive and the table layout is not, so the tables can be rebuilt
from the stored JSON without re-running anything, and the numbers in the report are provably
the numbers the run produced rather than transcribed by hand.

    python3 ml/analyze_hypothesis.py --results out/hypothesis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.hypothesis_labels import CLASSES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _f(x, nd=4, dash="n/a"):
    return dash if x is None else f"{x:.{nd}f}"


def composition_table(blob: dict) -> None:
    print("\n### candidate set composition\n")
    print("| variant | seed | instances | usable sup | shared_upstream | regional | "
          "shared_sourcing | unknown | A/B pairs evaluable |")
    print("|---|---|---|---|---|---|---|---|---|")
    for variant, v in blob["variants"].items():
        for s in v["per_seed"]:
            c, ce = s["composition"], s["ceiling"]
            p = c["positives"]
            print(f"| {variant} | {s['seed']} | {c['n_pairs']:,} | {s['n_usable']} | "
                  f"{p['shared_upstream']:,} | {p['regional_logistics']:,} | "
                  f"{p['shared_sourcing']:,} | {p['unknown']:,} | "
                  f"{ce['shared_upstream_pairs_both_usable']}/"
                  f"{ce['shared_upstream_pairs_total']} "
                  f"({100 * ce['shared_upstream_evaluable_frac']:.0f}%) |")
    print("\n**multi-label overlap** (pairs carrying more than one mechanism):\n")
    for variant, v in blob["variants"].items():
        for s in v["per_seed"]:
            ov = s["composition"]["overlap"]
            n_lab = s["composition"]["n_labels_per_pair"]
            print(f"- v{variant} s{s['seed']}: " +
                  ", ".join(f"{k}={val}" for k, val in ov.items() if val) +
                  f"  (labels/pair: {n_lab})")


def quality_table(blob: dict) -> None:
    print("\n### per-class ranking and calibration (pooled over folds)\n")
    print("| variant | class | base rate | positives | AUC | AP | ECE raw | ECE calibrated | "
          "Brier cal |")
    print("|---|---|---|---|---|---|---|---|---|")
    for variant, v in blob["variants"].items():
        for cls in CLASSES:
            r = v["cv"]["pooled"][cls]
            print(f"| {variant} | {cls} | {r['base_rate']:.5f} | {r['positives']:,} | "
                  f"**{_f(r['auc'])}** | {_f(r['ap'])} | {_f(r['ece_raw'])} | "
                  f"**{_f(r['ece_calibrated'])}** | {_f(r['brier_calibrated'], 6)} |")

    print("\n### shared_upstream, split by group type (the decoy is the control)\n")
    print("| variant | subtype | positives | AUC vs negatives ± binomial SE | "
          "model-seed floor (max abs) | clears floor? | folds > chance |")
    print("|---|---|---|---|---|---|---|")
    for variant, v in blob["variants"].items():
        fl = v.get("floor_subtype_model_seed", {})
        for name in ("type_a_pair", "type_b_pair", "type_c_pair"):
            r = v["subtypes"][name]
            auc = r["auc_vs_negatives"]
            f = fl.get(name, {}).get("max_abs_dev")
            clears = ("-" if auc is None or f is None
                      else ("yes" if abs(auc - 0.5) > f else "**no**"))
            # Binomial standard error of an AUC estimate, which at these positive counts is
            # a far bigger number than the reproduction floor and is the real limit on what
            # can be claimed. Hanley-McNeil's conservative form: 0.5 / sqrt(n_positive).
            se = 0.5 / np.sqrt(max(r["n_positive"], 1))
            print(f"| {variant} | {name} | {r['n_positive']} | {_f(auc)} ± {se:.3f} | "
                  f"{_f(f) if f is not None else 'n/a'} | {clears} | "
                  f"{r.get('folds_above_chance', '-')} |")

    print("\n### decoy-controlled contrast (real type − Type C), pooled\n")
    print("| variant | contrast | value | model-seed floor (max abs) |")
    print("|---|---|---|---|")
    for variant, v in blob["variants"].items():
        fl = v.get("floor_subtype_model_seed", {})
        for name in ("type_a_pair_minus_decoy", "type_b_pair_minus_decoy"):
            val = v["subtypes"].get(name)
            f = fl.get(name, {}).get("max_abs_dev")
            print(f"| {variant} | {name.replace('_minus_decoy', '')} − decoy | "
                  f"{_f(val) if val is not None else 'n/a'} | "
                  f"{_f(f) if f is not None else 'n/a'} |")


def floor_table(blob: dict) -> None:
    for key, label in (("floor_identical", "identical config (same model seed)"),
                       ("floor_model_seed", "model-init seed varied")):
        print(f"\n### reproduction floor — {label}\n")
        print("| variant | class | metric | n | mean | mean abs dev | max abs dev |")
        print("|---|---|---|---|---|---|---|")
        for variant, v in blob["variants"].items():
            for cls, metrics in v.get(key, {}).items():
                for metric, d in metrics.items():
                    if metric == "ece_raw":
                        continue
                    print(f"| {variant} | {cls} | {metric} | {d['n_replicates']} | "
                          f"{d['mean']:.4f} | {d['mean_abs_dev']:.4f} | "
                          f"**{d['max_abs_dev']:.4f}** |")


def reliability_table(blob: dict, cls: str) -> None:
    print(f"\n### reliability curve — {cls} (calibrated, equal-count bins)\n")
    print("| variant | bin | n | predicted | empirical | gap |")
    print("|---|---|---|---|---|---|")
    for variant, v in blob["variants"].items():
        for i, b in enumerate(v["cv"]["pooled"][cls]["reliability"], 1):
            print(f"| {variant} | {i} | {b['n']:,} | {b['confidence']:.4f} | "
                  f"{b['empirical']:.4f} | {b['empirical'] - b['confidence']:+.4f} |")


def reviewer_table(blob: dict, variant: str, n: int) -> None:
    v = blob["variants"][variant]
    rows = v["reviewer_sample"]
    seen, out = set(), []
    for r in rows:
        key = (r["supplier_a"], r["supplier_b"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    print(f"\n### reviewer output — variant {variant}, {len(out)} distinct instances\n")
    print("| supplier A | supplier B | corr | ranked hypotheses (calibrated) | truth |")
    print("|---|---|---|---|---|")
    for r in out[:n]:
        ranked = " · ".join(f"{h['hypothesis']} {100 * h['confidence']:.1f}%"
                            for h in r["ranked_hypotheses"][:3])
        print(f"| `{r['supplier_a'][:8]}` | `{r['supplier_b'][:8]}` | "
              f"{r['pattern']['corr_on_time_90d']:+.3f} | {ranked} | "
              f"{', '.join(r['truth'])} |")


def fusion_table(results_dir: str) -> None:
    """§10.5's AUC delta, paired by dataset seed, against the floor from replicate tags."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        b = json.load(open(path))
        if "arm" not in b:
            continue
        rows.setdefault((b["variant"], b["seed"], b.get("tag", "")), {})[b["arm"]] = b
    if not rows:
        print("\n(no fusion results)")
        return
    tasks = ("delay", "shortage", "impact")

    print("\n### fusion — paired delta (hypconf − baseline), per dataset seed\n")
    print("| variant | seed | tag | " + " | ".join(tasks) + " |")
    print("|---|---|---|" + "---|" * len(tasks))
    deltas: dict[tuple, list] = {}
    for (variant, seed, tag), arms in sorted(rows.items()):
        if "baseline" not in arms or "hypconf" not in arms:
            continue
        cells = []
        for t in tasks:
            a, c = arms["baseline"]["auc"].get(t), arms["hypconf"]["auc"].get(t)
            if a is None or c is None:
                cells.append("n/a")
                continue
            d = c - a
            cells.append(f"{d:+.4f}")
            deltas.setdefault((variant, t), []).append(d)
        print(f"| {variant} | {seed} | {tag or '-'} | " + " | ".join(cells) + " |")

    print("\n| variant | task | mean delta | sign consistency |")
    print("|---|---|---|---|")
    for (variant, t), ds in sorted(deltas.items()):
        pos = sum(1 for d in ds if d > 0)
        n = len(ds)
        agree = max(pos, n - pos)
        print(f"| {variant} | {t} | {np.mean(ds):+.4f} | {agree}/{n} "
              f"({'positive' if pos >= n - pos else 'negative'}) |")

    # Floor: two identically-configured replicates of the SAME arm, differing only in tag.
    by_arm: dict[tuple, dict] = {}
    for (variant, seed, tag), arms in rows.items():
        for arm, b in arms.items():
            by_arm.setdefault((arm, variant, seed), {})[tag] = b
    print("\n### fusion — reproduction floor from replicate pairs, this session's device\n")
    print("| arm | task | n pairs | mean abs | max abs |")
    print("|---|---|---|---|---|")
    for arm in ("baseline", "hypconf"):
        for t in tasks:
            gaps = []
            for (a, _v, _s), tags in by_arm.items():
                if a != arm or len(tags) < 2:
                    continue
                vals = [b["auc"].get(t) for b in tags.values() if b["auc"].get(t) is not None]
                if len(vals) >= 2:
                    gaps.append(abs(vals[0] - vals[1]))
            if gaps:
                print(f"| {arm} | {t} | {len(gaps)} | {np.mean(gaps):.4f} | "
                      f"**{np.max(gaps):.4f}** |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(REPO, "out", "hypothesis"))
    ap.add_argument("--file", default="detected.json")
    ap.add_argument("--fusion", default=os.path.join(REPO, "out", "hypfusion"))
    ap.add_argument("--reviewer-variant", default="D")
    ap.add_argument("--reviewer-rows", type=int, default=24)
    ap.add_argument("--sections", default="composition,quality,floor,reliability,reviewer,fusion")
    args = ap.parse_args()

    path = os.path.join(args.results, args.file)
    blob = json.load(open(path)) if os.path.exists(path) else None
    want = {s.strip() for s in args.sections.split(",")}
    if blob:
        print(f"# {args.file} — candidate set: "
              f"{blob['config'].get('candidate_set', 'detected')}, "
              f"top_k={blob['config']['top_k']}, "
              f"respect_t0={blob['config'].get('respect_t0')}")
        if "composition" in want:
            composition_table(blob)
        if "quality" in want:
            quality_table(blob)
        if "floor" in want:
            floor_table(blob)
        if "reliability" in want:
            reliability_table(blob, "regional_logistics")
            reliability_table(blob, "shared_upstream")
        if "reviewer" in want and args.reviewer_variant in blob["variants"]:
            reviewer_table(blob, args.reviewer_variant, args.reviewer_rows)
    if "fusion" in want and os.path.isdir(args.fusion):
        fusion_table(args.fusion)
    return 0


if __name__ == "__main__":
    sys.exit(main())
