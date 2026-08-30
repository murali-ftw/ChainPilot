#!/usr/bin/env python3
"""
Calibration diagnostic for `delay`, `shortage` and `impact` — measurement only, no correction.

Every result this project has recorded for the three benchmark tasks is an AUC, and AUC is a
*ranking* statistic: it is invariant to any monotone transform of the scores, so a model that
says 0.999 when it means 0.60 scores exactly as well as one that says 0.60. That blind spot is
already known to matter here -- `reports/decision_support_build.md` §3.2 found **96.5% of
`delay`'s predictions at p >= 0.999** -- but it has never been checked for `shortage` or
`impact`, whose failure modes to date (schema gaps, relation redundancy) have been unrelated to
feature saturation. There is no basis for assuming their calibration behaves like `delay`'s in
either direction, so it is measured per task.

**Nothing is trained and nothing is corrected here.** Checkpoints are loaded from
`out/ds_ckpt/`, `freeze()` asserts every parameter is frozen, and the forward pass is
`ml/evaluate.py::collect_predictions` called exactly as every other evaluation calls it. No
temperature, Platt or isotonic map is fitted -- deciding *whether* a correction is warranted, and
for which task, is the entire output of this module.

**Which checkpoints.** Per the brief: `shortage` and `impact` are read from the bucketed-Carrier
("fixed") graph, the arm on which they currently pass (`reports/new_nodes_fix1.md`). `delay` is
read from the **old V2 graph**, the arm its reference number (+0.0357 against floor 0.0874) comes
from and the arm the 96.5% saturation finding was made on. Because both sweeps produce all three
tasks from one forward pass, the off-arm numbers are reported too, as a free cross-check rather
than as the headline.

**Two ECE definitions, both reported.** The reliability diagram uses **equal-width** deciles
(0-10%, ..., 90-100%), which is what the saturation question needs -- "what fraction of
predictions sit in the top bin" is meaningless under equal-count binning, since equal-count
binning forces every bin to hold 10%. `ml/hypothesis_ranker.py::ece` is the project's existing,
test-validated ECE and uses **equal-count** bins for a documented reason (these predictions pile
up near zero, so equal-width bins can be too sparse to estimate a rate in). Both are computed:
the equal-width one because it answers this question, the equal-count one so the number stays
comparable to `reports/layer3_uncertainty_aware.md` and the world-conditioned calibration work.

**Per-cell, then aggregated -- never a single pooled model.** Each of the 25 cells is a different
trained model, so ECE/Brier/AUC are computed *within* a cell and then averaged, which measures
each model's own honesty. A pooled reliability curve over all 25 cells is reported alongside for
the diagram, and labelled as the mixture it is.

    python3 ml/calibration_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone                  # noqa: E402
from ml.evaluate import calibration_error, collect_predictions  # noqa: E402
from ml.hypothesis_ranker import ece as ece_equalcount   # noqa: E402
from ml.hypothesis_ranker import roc_auc                 # noqa: E402
from ml.models.depth import TASKS                        # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")
N_BINS = 10

# arm -> (csv root, directory prefix, v3_schema). The primary arm per task is set below.
ARMS = {
    "old":          (os.path.join(REPO, "db", "csv_v1scale"), "v0_seed", "full"),
    "carrier_lane": (os.path.join(REPO, "db", "csv_v3enriched"), "v0enr_seed", "carrier_lane"),
}
PRIMARY_ARM = {"delay": "old", "shortage": "carrier_lane", "impact": "carrier_lane"}


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Equal-width reliability curve: one row per decile of PREDICTED probability.

    `mean_pred` is what the model claimed in that bin; `obs_rate` is what actually happened.
    A bin where `mean_pred` far exceeds `obs_rate` is overconfident there, and the `frac`
    column says how much of the prediction mass that bin holds -- which is the whole question
    for a saturated task, where one bin can hold nearly everything.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        out.append({"bin": f"{lo:.0%}-{hi:.0%}", "lo": float(lo), "hi": float(hi), "n": n,
                    "frac": float(n / len(p)) if len(p) else 0.0,
                    "mean_pred": float(p[m].mean()) if n else None,
                    "obs_rate": float(y[m].mean()) if n else None,
                    "gap": float(p[m].mean() - y[m].mean()) if n else None})
    return out


def cell_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Every per-cell number, computed on that one model's own predictions."""
    if not len(p):
        return {}
    auc = roc_auc(p, y)
    ece_ec, _curve = ece_equalcount(p, y)
    rel = reliability(p, y)
    return {
        "n": int(len(p)),
        "base_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "auc": auc,
        "ece_equalwidth": float(calibration_error(y, p, N_BINS)),
        "ece_equalcount": float(ece_ec),
        "brier": float(np.mean((p - y) ** 2)),
        # signed mean gap: >0 means the model claims more than happens = OVERCONFIDENT overall
        "mean_gap": float(p.mean() - y.mean()),
        "frac_bin_top": rel[-1]["frac"],
        "frac_bin_bottom": rel[0]["frac"],
        "frac_ge_999": float((p >= 0.999).mean()),
        "frac_le_001": float((p <= 1e-3).mean()),
        "n_unique": int(len(np.unique(np.round(p, 9)))),
        "reliability": rel,
    }


def run_arm(arm: str, dseeds, mseeds, device="cpu") -> dict:
    root, prefix, schema = ARMS[arm]
    cells: dict = {}
    raw: dict = {t: {"p": [], "y": [], "world": []} for t in TASKS}
    for d in dseeds:
        csv_dir = os.path.join(root, f"{prefix}{d}")
        for m in mseeds:
            model, meta = get_backbone(csv_dir, "0", d, m, device=device, verbose=False,
                                       v3_schema=schema)
            preds = collect_predictions(model, meta["test_bundles"])
            for t in TASKS:
                y, p = preds[t]["y"].astype(float), preds[t]["p"].astype(float)
                cells[(d, m, t)] = cell_metrics(p, y)
                raw[t]["p"].append(p)
                raw[t]["y"].append(y)
                raw[t]["world"].append(np.full(len(p), d))
            print(f"  [{arm}] d{d} m{m}  " + "  ".join(
                f"{t}: ECE={cells[(d,m,t)]['ece_equalwidth']:.4f} "
                f"top-bin={cells[(d,m,t)]['frac_bin_top']:.1%}" for t in TASKS), flush=True)
    pooled = {t: {k: np.concatenate(v) for k, v in raw[t].items()} for t in TASKS}
    return {"cells": cells, "pooled": pooled}


def agg(cells: dict, task: str, dseeds, mseeds, keys) -> dict:
    """Mean over cells, plus the per-world means and the init-seed spread that every other
    result in this project reports alongside a mean."""
    out = {}
    for k in keys:
        M = np.array([[cells[(d, m, task)].get(k) if cells[(d, m, task)].get(k) is not None
                       else np.nan for m in mseeds] for d in dseeds], dtype=float)
        out[k] = {
            "mean": float(np.nanmean(M)),
            "per_world": {str(d): float(np.nanmean(M[i])) for i, d in enumerate(dseeds)},
            "init_seed_spread_max": float(np.nanmax([np.nanmax(r) - np.nanmin(r) for r in M])),
            "world_spread": float(np.nanmax([np.nanmean(r) for r in M])
                                  - np.nanmin([np.nanmean(r) for r in M])),
        }
    return out


METRIC_KEYS = ("auc", "ece_equalwidth", "ece_equalcount", "brier", "mean_gap",
               "frac_bin_top", "frac_bin_bottom", "frac_ge_999", "base_rate", "mean_pred")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "calibration_diagnostic.json"))
    a = ap.parse_args()
    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    arms = {}
    for arm in ARMS:
        print(f"\n=== scoring arm '{arm}' ===")
        arms[arm] = run_arm(arm, dseeds, mseeds, a.device)

    blob = {"config": vars(a), "primary_arm": PRIMARY_ARM, "arms": {}}
    for arm, got in arms.items():
        blob["arms"][arm] = {
            "summary": {t: agg(got["cells"], t, dseeds, mseeds, METRIC_KEYS) for t in TASKS},
            "pooled_reliability": {t: reliability(got["pooled"][t]["p"], got["pooled"][t]["y"])
                                   for t in TASKS},
            "pooled_reliability_by_world": {
                t: {str(d): reliability(got["pooled"][t]["p"][got["pooled"][t]["world"] == d],
                                        got["pooled"][t]["y"][got["pooled"][t]["world"] == d])
                    for d in dseeds} for t in TASKS},
            "cells": {f"{d}|{m}|{t}": {k: v for k, v in got["cells"][(d, m, t)].items()
                                       if k != "reliability"}
                      for d in dseeds for m in mseeds for t in TASKS},
        }

    # ------------------------------------------------------------------ headline
    hdr = (f"{'task':<10}{'arm':<14}{'AUC':>8}{'ECE-ew':>9}{'ECE-ec':>9}{'Brier':>9}"
           f"{'base':>8}{'meanP':>8}{'gap':>9}{'top bin':>9}{'bot bin':>9}")
    print("\n" + "=" * len(hdr))
    print(f"CALIBRATION DIAGNOSTIC — {len(dseeds)} worlds x {len(mseeds)} init seeds, "
          f"per-cell metrics averaged")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for t in TASKS:
        for arm in ARMS:
            s = blob["arms"][arm]["summary"][t]
            star = " *" if PRIMARY_ARM[t] == arm else ""
            print(f"{t:<10}{arm + star:<14}{s['auc']['mean']:>8.4f}"
                  f"{s['ece_equalwidth']['mean']:>9.4f}{s['ece_equalcount']['mean']:>9.4f}"
                  f"{s['brier']['mean']:>9.4f}{s['base_rate']['mean']:>8.4f}"
                  f"{s['mean_pred']['mean']:>8.4f}{s['mean_gap']['mean']:>+9.4f}"
                  f"{s['frac_bin_top']['mean']:>9.1%}{s['frac_bin_bottom']['mean']:>9.1%}")
        print("-" * len(hdr))
    print("  * = primary arm for that task;  ECE-ew = equal-width deciles, "
          "ECE-ec = equal-count (project's existing definition)")
    print("  gap = mean(predicted) - mean(observed); >0 overconfident, <0 underconfident")

    # ------------------------------------------------------------------ reliability
    for t in TASKS:
        arm = PRIMARY_ARM[t]
        print(f"\nRELIABILITY — {t} (arm '{arm}', predictions pooled over all "
              f"{len(dseeds) * len(mseeds)} cells)")
        h2 = f"{'bin':>12}{'n':>10}{'frac':>9}{'mean pred':>11}{'obs rate':>10}{'gap':>9}"
        print(h2); print("-" * len(h2))
        for r in blob["arms"][arm]["pooled_reliability"][t]:
            if not r["n"]:
                print(f"{r['bin']:>12}{0:>10}{0.0:>9.1%}{'—':>11}{'—':>10}{'—':>9}")
                continue
            print(f"{r['bin']:>12}{r['n']:>10,}{r['frac']:>9.1%}{r['mean_pred']:>11.4f}"
                  f"{r['obs_rate']:>10.4f}{r['gap']:>+9.4f}")
        print("-" * len(h2))

    # ------------------------------------------------------------------ per world
    print("\nPER-DATASET-SEED WORLD (primary arm; ECE equal-width, and extreme-bin mass)")
    h3 = (f"{'task':<10}{'metric':<16}" + "".join(f"{'d'+str(d):>10}" for d in dseeds)
          + f"{'spread':>10}")
    print(h3); print("-" * len(h3))
    for t in TASKS:
        s = blob["arms"][PRIMARY_ARM[t]]["summary"][t]
        for key, label in (("ece_equalwidth", "ECE"), ("frac_bin_top", "top-bin frac"),
                           ("frac_bin_bottom", "bottom-bin frac"), ("auc", "AUC")):
            fmt = "{:>10.1%}" if key.startswith("frac") else "{:>10.4f}"
            print(f"{t:<10}{label:<16}"
                  + "".join(fmt.format(s[key]["per_world"][str(d)]) for d in dseeds)
                  + fmt.format(s[key]["world_spread"]))
        print("-" * len(h3))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
