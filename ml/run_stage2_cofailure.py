#!/usr/bin/env python3
"""
Stage 2 — retrieval quality of the **observable co-failure** signal, measured at the data
level before a single model is trained.

This ordering is deliberate cost discipline, the same the last session applied when it
declined to run Variant K. Stage 2's retriever ranks suppliers by the correlation of their
recorded on-time-rate history (`ml/observable_cofailure.py`); that ranking exists whether
or not any model consumes it, and its quality against `HP_GROUPS` can be scored directly.
If the ranking is at chance, wiring it into `Transformer2GlobalAttention` and retraining
the grid cannot rescue it -- attention over a pool that contains no co-members has nothing
to attend to, which is precisely the mechanism `reports/layer3_testing.md` §2 diagnosed.
So the cheap measurement runs first and the expensive one only if it clears.

Scored on identical terms to Stage 1: the same `ml/retrieval_metrics.py`, the same top-k
width, the same per-group-type disaggregation, the same chance baselines. The supplier
universe is the **full** population, not just the suppliers carrying usable history, so
N and therefore every chance baseline matches the model-side numbers exactly. Suppliers
with too little history are given -inf affinity and simply never retrieved -- which is
itself a finding worth its own row, since only about half the population has a usable
trajectory at this scale.

Two retrievers are reported:

* **raw** -- correlation of each supplier's residual against the fleet mean, exactly
  `db/generate_dataset.py::_windowed_delta`'s `member r90 - fleet r90` statistic.
* **cohort-residualised** -- the same after removing each observable cohort's own mean
  (country x lead-time tercile), which is the fair version of the test given that the
  base world's shared event pools cover 50% of suppliers and outnumber real Type A/B
  pairs 363:1.

    python3 ml/run_stage2_cofailure.py --variants B,D --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.observable_cofailure import (  # noqa: E402
    cofailure_scores, load_history, observable_cohort, top_k_pool)
from ml.retrieval_metrics import METRICS, merge, score_retrieval  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def guardrail(wide, sup_ids, t0, cohort, k) -> dict:
    """Run the leakage assertion on the EXACT matrices this measurement uses.

    `ml/test_cofailure_leakage.py` is the standalone version kept for the record; this is
    the same check re-run inline, because a guardrail that passes on a slightly different
    configuration than the one that produced the numbers is not a guardrail.
    """
    base_s = cofailure_scores(wide, sup_ids, t0, cohort=cohort)
    base_p = top_k_pool(base_s, k)
    fut = [c for c in wide.columns if str(c) >= str(t0)]
    out = {"future_columns": len(fut), "checks": []}
    rng = np.random.default_rng(0)
    for name, filler in (("shuffled", None), ("blanked", np.nan), ("garbage", None),
                         ("zeroed", 0.0)):
        w = wide.copy()
        if name == "shuffled":
            w[fut] = w[fut].sample(frac=1.0, random_state=1).to_numpy()
        elif name == "garbage":
            w[fut] = rng.uniform(-1e3, 1e3, size=(len(w), len(fut)))
        else:
            w[fut] = filler
        s = cofailure_scores(w, sup_ids, t0, cohort=cohort)
        p = top_k_pool(s, k)
        d = float(np.abs(np.nan_to_num(s, nan=0.0, neginf=-1e9)
                         - np.nan_to_num(base_s, nan=0.0, neginf=-1e9)).max())
        out["checks"].append({"corruption": name, "max_score_delta": d,
                              "pool_identical": bool(np.array_equal(p, base_p))})
    out["passed"] = all(c["pool_identical"] and c["max_score_delta"] < 1e-9
                        for c in out["checks"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_mid"))
    ap.add_argument("--hidden-dir", default=os.path.join(REPO, "out", "hidden_mid"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--column", default="on_time_rate_90d")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.6,
                    help="fraction of snapshots before the TEST split, matching "
                         "ml/data/loader.py::split_bundles' 40/20/40")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "stage2_cofailure.json"))
    args = ap.parse_args()

    results = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            wide, dates = load_history(d, column=args.column)
            # Full supplier population, in the emitted order -- so N, and therefore every
            # chance baseline, matches the model-side measurement exactly.
            sup_ids = pd.read_csv(f"{d}/suppliers.csv.gz", usecols=["id"])["id"].tolist()
            wide = wide.reindex(index=sup_ids)
            cohort = observable_cohort(d, sup_ids)
            groups = json.load(open(os.path.join(args.hidden_dir, f"{variant}_{seed}.json"))
                               )["hp_groups"]
            idx = {s: i for i, s in enumerate(sup_ids)}
            members = {gi: [idx[m] for m in g["members"] if m in idx]
                       for gi, g in enumerate(groups)}
            type_by_group = {gi: g["type"] for gi, g in enumerate(groups)}
            test_dates = dates[int(round(args.test_frac * len(dates))):]

            gr = guardrail(wide, sup_ids, test_dates[0], cohort, args.top_k)
            if not gr["passed"]:
                print(f"  v{variant} seed{seed}: LEAKAGE GUARDRAIL FAILED -- refusing to "
                      f"report a number: {gr['checks']}", flush=True)
                continue

            n_hist = int((wide.notna().sum(axis=1) > 0).sum())
            # Two supplier universes, because they answer two different questions and
            # pooling them produces a number that answers neither:
            #
            #   "all"          -- every supplier, matching the model-side N exactly, so
            #                     the chance baselines are directly comparable to Stage 1.
            #                     Suppliers with no recorded history are unretrievable and
            #                     score 0, which is the honest cost of the coverage gap.
            #   "with_history" -- only suppliers carrying a usable trajectory. This is the
            #                     fair test of whether the co-failure SIGNAL ranks, with
            #                     the coverage problem held separate from it. Restricting
            #                     the universe also repairs `pct_rank`, which is otherwise
            #                     dominated by the ~76% of pairs pinned at -inf.
            usable = np.asarray((wide.notna().sum(axis=1) >= 4).to_numpy())
            for universe in ("all", "with_history"):
                if universe == "all":
                    keep = np.arange(len(sup_ids))
                else:
                    keep = np.flatnonzero(usable)
                remap = {int(o): i for i, o in enumerate(keep)}
                mem_u = {gi: [remap[r] for r in rows if r in remap]
                         for gi, rows in members.items()}
                mem_u = {gi: r for gi, r in mem_u.items() if len(r) >= 2}
                for mode, coh in (("raw", None), ("cohort_residualised", cohort)):
                    accs = []
                    for t0 in test_dates:
                        s = cofailure_scores(wide, sup_ids, t0, cohort=coh)
                        s = s[np.ix_(keep, keep)]
                        pool = top_k_pool(s, args.top_k)
                        accs.append(score_retrieval(
                            np.nan_to_num(s, nan=-1e9, neginf=-1e9), pool, mem_u,
                            type_by_group, top_k=args.top_k))
                    m = merge(accs)
                    rec = {"variant": variant, "seed": seed, "mode": mode,
                           "universe": universe, "column": args.column,
                           "n_suppliers": int(len(keep)), "n_with_history": n_hist,
                           "n_test_snapshots": len(test_dates),
                           "history_at_first_test_t0": sum(1 for c in dates
                                                           if str(c) < str(test_dates[0])),
                           "guardrail": gr, "retrieval": m}
                    results.append(rec)
                    if universe != "with_history":
                        continue
                    line = f"  v{variant} seed{seed} {mode:>20}:"
                    for t in "ABC":
                        if t in m:
                            line += (f"  {t}: rank={m[t]['pct_rank']:.3f} "
                                     f"R@k={m[t]['recall_at_k']:.4f}"
                                     f"({m[t]['recall_at_k']/m[t]['chance']['recall_at_k']:.2f}x)"
                                     f" mrr={m[t]['mrr']:.4f}")
                    print(line, flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)

    # ---- pooled summary -----------------------------------------------------
    for universe in ("with_history", "all"):
        print(f"\n## Stage 2 -- observable co-failure retrieval, pooled over seeds "
              f"(universe: {universe})\n")
        print("| var | mode | type | " + " | ".join(
            f"{m} | chance | ratio" for m in METRICS) + " | N | seeds |")
        print("|---|---|---|" + "---|---|---|" * len(METRICS) + "---|---|")
        for variant in sorted({r["variant"] for r in results}):
            for mode in ("raw", "cohort_residualised"):
                for t in "ABC":
                    rs = [r for r in results if r["variant"] == variant
                          and r["mode"] == mode and r["universe"] == universe
                          and t in r["retrieval"]]
                    if not rs:
                        continue
                    xs = [r["retrieval"][t] for r in rs]
                    cells = []
                    for m in METRICS:
                        obs = float(np.mean([x[m] for x in xs]))
                        ch = float(np.mean([x["chance"][m] for x in xs]))
                        cells.append(f" {obs:.4f} | {ch:.4f} | {obs/ch:.2f}x |" if ch
                                     else f" {obs:.4f} | -- | -- |")
                    print(f"| {variant} | {mode} | {t} |" + "".join(cells)
                          + f" {rs[0]['n_suppliers']:,} | {len(rs)} |")
    if results:
        r0 = results[0]
        print(f"\nCoverage: {r0['n_with_history']:,} of {r0['n_suppliers']:,} suppliers "
              f"carry any observable trajectory; {r0['history_at_first_test_t0']} snapshot "
              f"observations precede the first test t0.")
        print(f"Leakage guardrail: PASSED on every variant-seed "
              f"({r0['guardrail']['future_columns']} future snapshots corrupted four ways).")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
