#!/usr/bin/env python3
"""
Two-axis predictive ensemble: model-init seed and dataset seed, decomposed.

`reports/phase7_training_results.md` §5.6 measured **dataset-seed std at up to 10.3x
model-seed std** on this benchmark (delay: 0.0376 vs 0.0037). It measured them separately,
one axis at a time, on one architecture. An uncertainty estimate built from model-init
ensembling alone -- which is what Deep Ensembles, MC Dropout and single-dataset evidential
heads all give -- would therefore be estimating the *smaller* of the two sources and calling
it confidence.

This module builds the **5 x 5 grid** (5 dataset seeds x 5 model-init seeds) that lets the
decomposition be done properly rather than inferred from two separate one-way measurements,
and reports the fraction of total variance attributable to each axis.

---

**Two decompositions, because they answer different questions and only one is identifiable
per entity.**

*(a) Metric-level (the headline).* For a scalar metric `M` (test AUC per task), the 5 x 5 grid
gives a genuine two-way decomposition by the law of total variance:

    mu_d      = mean_m M[d,m]                     per-world mean over model seeds
    var_model = mean_d Var_m( M[d,m] )            E_world[ Var_model ]   -- within-world
    var_data  = Var_d( mu_d )                     Var_world[ E_model ]   -- across-world
    var_total = var_model + var_data

This is directly comparable to §5.6 and either confirms or revises it.

*(b) Prediction-level (per entity).* Within one world, an entity's prediction varies across
model seeds and `var_model(v)` is well defined. **Across worlds it is not**: dataset seed 42
and seed 43 generate *different suppliers*, so there is no entity to pair. Per-entity
dataset-seed variance is **not identifiable on this benchmark**, and this module does not
fabricate it -- it reports per-entity `var_model` and a population-level `var_data`, labelled
as the different objects they are. Claiming otherwise would be inventing identifiability that
the data does not support.

    python3 ml/uncertainty_ensemble.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone  # noqa: E402
from ml.evaluate import collect_predictions  # noqa: E402
from ml.models.depth import TASKS  # noqa: E402


def collect_grid(csv_root: str, variant: str, dseeds: list[int], mseeds: list[int],
                 device: str = "cpu") -> dict:
    """`{(dseed, mseed): {task: {y, p, block}}}` on each world's own test split.

    Predictions are collected with the existing `ml/evaluate.py::collect_predictions`, the
    same function every AUC in this project came from, so these numbers sit on the same axis
    as the rest of the record.
    """
    grid, aucs = {}, {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        for m in mseeds:
            model, meta = get_backbone(csv_dir, variant, d, m, device=device, verbose=False)
            preds = collect_predictions(model, meta["test_bundles"])
            grid[(d, m)] = {t: {"y": preds[t]["y"], "p": preds[t]["p"]} for t in TASKS}
            aucs[(d, m)] = meta["auc"]
    return {"grid": grid, "auc": aucs}


def metric_decomposition(aucs: dict, dseeds: list[int], mseeds: list[int]) -> dict:
    """Two-way variance decomposition of test AUC -- the §5.6 comparison, done on a full grid.

    `ddof=1` throughout: these are samples of seeds, not populations.
    """
    out = {}
    for task in TASKS:
        M = np.array([[aucs[(d, m)][task] for m in mseeds] for d in dseeds], dtype=float)
        if not np.isfinite(M).all():
            continue
        mu_d = M.mean(axis=1)                        # per-world mean over model seeds
        var_model = float(np.mean(M.var(axis=1, ddof=1)))     # E_world[ Var_model ]
        var_data = float(mu_d.var(ddof=1))                    # Var_world[ E_model ]
        total = var_model + var_data
        # Per-axis one-way stds, for direct comparison with §5.6's table, which reported
        # exactly these two numbers from two separate experiments.
        out[task] = {
            "n_worlds": len(dseeds), "n_model_seeds": len(mseeds),
            "grand_mean": float(M.mean()),
            "var_model": var_model, "var_dataset": var_data, "var_total": total,
            "std_model": float(np.sqrt(var_model)), "std_dataset": float(np.sqrt(var_data)),
            "dataset_share": float(var_data / total) if total > 0 else None,
            "std_ratio_dataset_over_model": (float(np.sqrt(var_data / var_model))
                                             if var_model > 0 else None),
            "per_world_mean": {str(d): float(v) for d, v in zip(dseeds, mu_d)},
            "per_world_std_over_models": {str(d): float(M[i].std(ddof=1))
                                          for i, d in enumerate(dseeds)},
        }
    return out


def prediction_stats(grid: dict, dseeds: list[int], mseeds: list[int]) -> dict:
    """Per-entity ensemble mean and within-world variance, plus a population-level
    across-world term.

    The across-world term is deliberately computed on the *distribution* of predictions
    (mean and spread of predicted probability over the scored population), not per entity,
    because entities do not persist across dataset seeds.
    """
    out = {}
    for task in TASKS:
        per_world = {}
        for d in dseeds:
            P = np.stack([grid[(d, m)][task]["p"] for m in mseeds])     # [M, N]
            y = grid[(d, mseeds[0])][task]["y"]
            for m in mseeds[1:]:
                assert np.array_equal(grid[(d, m)][task]["y"], y), \
                    "label vectors differ across model seeds within a world"
            per_world[d] = {
                "mu": P.mean(axis=0), "var_model": P.var(axis=0, ddof=1), "y": y,
                "n": int(len(y)), "positives": int(y.sum()),
            }
        pop_means = np.array([per_world[d]["mu"].mean() for d in dseeds])
        out[task] = {
            "per_world": per_world,
            "mean_var_model": float(np.mean([per_world[d]["var_model"].mean()
                                             for d in dseeds])),
            "population_mean_prediction_per_world": {str(d): float(per_world[d]["mu"].mean())
                                                     for d in dseeds},
            "var_across_world_population_mean": float(pop_means.var(ddof=1)),
            "identifiability_note": (
                "var_model is per-entity; the across-world term is population-level only. "
                "Entities do not persist across dataset seeds, so per-entity dataset-seed "
                "variance is not identifiable on this benchmark."),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "uncertainty_grid.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    got = collect_grid(a.csv_root, a.variant, dseeds, mseeds, a.device)
    dec = metric_decomposition(got["auc"], dseeds, mseeds)

    print(f"\n5x5 AUC variance decomposition ({len(dseeds)} worlds x {len(mseeds)} model seeds)\n")
    print(f"{'task':<10}{'mean':>9}{'std_model':>11}{'std_data':>10}{'ratio':>8}{'data_share':>12}")
    for t in TASKS:
        if t not in dec:
            continue
        d = dec[t]
        r = d["std_ratio_dataset_over_model"]
        print(f"{t:<10}{d['grand_mean']:>9.4f}{d['std_model']:>11.4f}{d['std_dataset']:>10.4f}"
              f"{(f'{r:.1f}x' if r else 'n/a'):>8}{d['dataset_share']:>11.1%}")

    blob = {"config": vars(a), "metric_decomposition": dec,
            "auc_grid": {f"{d}|{m}": got["auc"][(d, m)] for d in dseeds for m in mseeds}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    np.savez_compressed(
        a.out.replace(".json", "_preds.npz"),
        **{f"{t}_{d}_{m}_{k}": got["grid"][(d, m)][t][k]
           for t in TASKS for d in dseeds for m in mseeds for k in ("y", "p")})
    print(f"wrote {a.out.replace('.json', '_preds.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
