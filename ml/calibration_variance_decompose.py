#!/usr/bin/env python3
"""
STEP 5 of the world-conditioned calibration follow-up — does world identity explain the
calibration-curve differences, or does model-init noise?

**This is a hypothesis test with a named precedent, not a fresh guess.** The prior HADES_v2
session recorded (`reports/decision_support_build.md` §2.2, surviving in this repository as
`docs/14_Project_Roadmap.md` §"Confidence / uncertainty") that dataset-seed identity explains
**98.7% / 67.9% / 70.8%** of total AUC variance on delay / shortage / impact, via the two-way
law-of-total-variance split

    var_model   = mean_d Var_m(metric)      # within world, across model seeds
    var_dataset = Var_d(mean_m metric)      # between worlds, on the world means
    pct_world   = var_dataset / (var_dataset + var_model)

Those are a different task set from Supply Stress, but the same benchmark family and the same
generator, so the same decomposition is the right first hypothesis to test here. This module
applies it -- same estimator, so the numbers are comparable -- to *calibration-curve position*
rather than to AUC.

**Two caveats that change how the output must be read, both structural rather than incidental.**

1. **The second axis is the head init seed, not the backbone seed** -- carried verbatim from
   `reports/layer3_uncertainty_aware.md` §3, and forced by the same constraint: only `m0`
   backbone checkpoints exist in `out/ds_ckpt/`, and building `m1..m4` would mean training
   SHARE. §2.2's `var_model` is backbone-seed variance, which is the LARGER of the two model-side
   sources. Substituting the smaller one inflates `pct_world`, so every percentage below is an
   **upper bound** on the world share as §2.2 would have measured it, and is not directly
   substitutable into the 98.7 / 67.9 / 70.8 series.

2. **`Var_d(mean_m .)` is a biased estimator of the between-world component.** The mean over M
   init seeds still carries `var_model / M` of within-world noise, so `var_dataset` as defined
   above overstates the true between-world variance by that amount. §2.2 used this estimator, so
   it is reported first for comparability -- and the bias-corrected value
   `max(0, var_dataset - var_model / M)` is reported directly beside it, because a percentage
   that survives the correction and one that does not are different findings.

**And the percentage gets a null**, on the standing rule that no metric is believed until it is
known how far it moves with nothing changed. The null is the same one STEP 4 gates on: fit one
pooled isotonic curve across all five worlds, treat it as a common truth, resample each world's
labels from it once per world (shared across that world's five members, exactly as the real
labels are), and re-run the whole decomposition. That gives the `pct_world` distribution when
world identity explains *nothing* -- which is not 0%, because five worlds with finite samples
produce apparent between-world spread on their own.

    python3 ml/calibration_variance_decompose.py --variants A,E --out out/wcc/step5_variance.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.calibration_curve_compare import decompose, shared_grid          # noqa: E402
from ml.calibration_grid_cache import DSEEDS, INIT_SEEDS, load_grid      # noqa: E402
from ml.hypothesis_ranker import ece, isotonic_apply, isotonic_fit       # noqa: E402

N_BINS = 10
NULL_REPEATS = 200

# The v2 precedent this decomposition is being compared against, kept next to the code that
# cites it so the comparison cannot drift. Source: docs/14_Project_Roadmap.md, quoting
# reports/decision_support_build.md §2.2 (that report is not present in this repository).
V2_PCT_WORLD = {"delay": 98.7, "shortage": 67.9, "impact": 70.8}


# ---------------------------------------------------------------------------
# per-cell metrics
# ---------------------------------------------------------------------------

def cell_metrics(p: np.ndarray, y: np.ndarray, x: np.ndarray,
                 pooled_curve) -> dict:
    """Curve-position metrics for one (world, init seed) cell.

    `ece_own_map` is STEP 5's headline quantity in the prompt's words -- *the ECE achieved by
    each world's own map*. It is a within-world fit and therefore optimistically biased in
    absolute terms; that does not matter here, because the question is how its variance splits
    between the two axes, not how low it is.

    `shift` / `slope` / `rms_residual` are STEP 4's three components, measured against the
    **pooled** curve rather than pairwise, so every cell has a value on a common reference and
    the two-way decomposition is well defined.
    """
    curve = isotonic_fit(p, y.astype(float))
    g = isotonic_apply(curve, x)
    comp = decompose(g, isotonic_apply(pooled_curve, x), x)
    return {
        "ece_raw": ece(p, y, N_BINS)[0],
        "ece_own_map": ece(isotonic_apply(curve, p), y, N_BINS)[0],
        "shift": comp["shift"],
        "slope": comp["slope"],
        "rms_residual": comp["rms_residual"],
    }


METRICS = ("ece_raw", "ece_own_map", "shift", "slope", "rms_residual")


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------

def two_way(cells: dict, dseeds: list[int], members: list[int], metric: str) -> dict:
    """`var_model = mean_d Var_m(.)`, `var_dataset = Var_d(mean_m .)` -- §2.2's estimator.

    Population variance (`ddof=0`) is used on both axes so the two terms are on the same
    footing and `pct_world` is a genuine share of a single total.
    """
    M = np.array([[cells[(d, m)][metric] for m in members] for d in dseeds], dtype=float)
    var_model = float(np.mean(np.var(M, axis=1)))
    world_means = M.mean(axis=1)
    var_dataset = float(np.var(world_means))
    total = var_model + var_dataset
    corrected = max(0.0, var_dataset - var_model / len(members))
    total_c = var_model + corrected
    return {
        "metric": metric,
        "world_means": world_means.tolist(),
        "var_model_within_world": var_model,
        "var_dataset_between_world": var_dataset,
        "total": total,
        "pct_world": 100.0 * var_dataset / total if total > 0 else float("nan"),
        "var_dataset_bias_corrected": corrected,
        "pct_world_bias_corrected": 100.0 * corrected / total_c if total_c > 0 else float("nan"),
    }


def build_cells(grid: dict, dseeds: list[int], members: list[int],
                labels: dict | None = None) -> tuple[dict, np.ndarray, tuple]:
    """One metric row per (world, init seed), against a pooled-across-worlds reference curve.

    `labels` overrides the real label vectors -- that is the hook the null uses, and the only
    thing it changes.
    """
    P = {(d, m): grid[d]["p_te"][m] for d in dseeds for m in members}
    Y = {d: (labels[d] if labels is not None else grid[d]["y_te"]) for d in dseeds}
    x, _ = shared_grid([grid[d]["p_te"][members].mean(axis=0) for d in dseeds])
    pooled_curve = isotonic_fit(
        np.concatenate([grid[d]["p_te"][members].mean(axis=0) for d in dseeds]),
        np.concatenate([Y[d] for d in dseeds]).astype(float))
    cells = {(d, m): cell_metrics(P[(d, m)], Y[d], x, pooled_curve)
             for d in dseeds for m in members}
    return cells, x, pooled_curve


def null_distribution(grid: dict, dseeds: list[int], members: list[int],
                      repeats: int = NULL_REPEATS, seed: int = 20260815) -> dict:
    """`pct_world` when world identity explains nothing.

    One pooled curve is the common truth. Each world's labels are resampled from it **once per
    world**, shared across that world's five members -- which is how the real labels behave, and
    is what makes the null's within-world axis comparable to the observed one. Anything else
    would hand the null a within-world variance the real data does not have.
    """
    rng = np.random.default_rng(seed)
    mus = {d: grid[d]["p_te"][members].mean(axis=0) for d in dseeds}
    pooled_curve = isotonic_fit(np.concatenate([mus[d] for d in dseeds]),
                                np.concatenate([grid[d]["y_te"] for d in dseeds]).astype(float))
    truth = {d: isotonic_apply(pooled_curve, mus[d]) for d in dseeds}

    draws = {m: [] for m in METRICS}
    for _ in range(repeats):
        lab = {d: (rng.random(len(mus[d])) < truth[d]).astype(float) for d in dseeds}
        cells, _, _ = build_cells(grid, dseeds, members, labels=lab)
        for m in METRICS:
            draws[m].append(two_way(cells, dseeds, members, m)["pct_world"])
    return {"n_repeats": repeats,
            **{m: {"mean": statistics.fmean(draws[m]),
                   "p05": float(np.quantile(draws[m], 0.05)),
                   "p95": float(np.quantile(draws[m], 0.95))} for m in METRICS}}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(variant: str, state: str, dseeds: list[int], init_seeds: list[int],
        null_repeats: int = NULL_REPEATS) -> dict:
    grid, depth = load_grid(variant, state, dseeds, init_seeds)
    members = list(range(len(init_seeds)))
    cells, _, _ = build_cells(grid, dseeds, members)
    obs = {m: two_way(cells, dseeds, members, m) for m in METRICS}
    null = null_distribution(grid, dseeds, members, null_repeats)
    return {
        "variant": variant, "state": state, "depth": depth,
        "dataset_seeds": dseeds, "init_seeds": init_seeds,
        "cells": {f"{d}_{m}": cells[(d, m)] for d in dseeds for m in members},
        "decomposition": obs,
        "null": null,
        "clears_null": {m: bool(obs[m]["pct_world"] > null[m]["p95"]) for m in METRICS},
        "v2_reference_pct_world": V2_PCT_WORLD,
        "second_axis": "head init seed (NOT backbone seed) — see module docstring caveat 1",
    }


def report(r: dict) -> None:
    print("\n" + "=" * 100)
    print(f"STEP 5 — VARIANCE DECOMPOSITION   variant {r['variant']} / {r['state']} @ h^{r['depth']}"
          f"   {len(r['dataset_seeds'])} worlds x {len(r['init_seeds'])} head init seeds")
    print("=" * 100)
    print(f"{'metric':<16}{'var_model':>12}{'var_world':>12}{'% world':>10}"
          f"{'% corrected':>13}{'null mean':>11}{'null p95':>10}  clears?")
    for m in METRICS:
        o, n = r["decomposition"][m], r["null"][m]
        print(f"{m:<16}{o['var_model_within_world']:>12.3e}"
              f"{o['var_dataset_between_world']:>12.3e}{o['pct_world']:>9.1f}%"
              f"{o['pct_world_bias_corrected']:>12.1f}%{n['mean']:>10.1f}%{n['p95']:>9.1f}%"
              f"  {'YES' if r['clears_null'][m] else 'no'}")
    v2 = r["v2_reference_pct_world"]
    print(f"\nv2 §2.2 precedent (AUC, backbone-seed axis): "
          + "  ".join(f"{k} {v}%" for k, v in v2.items()))
    print(f"second axis here: {r['second_axis']}")
    print(f"null: {r['null']['n_repeats']} label resamples from one pooled curve "
          f"(world identity explains nothing)")
    print("\nper-world means:")
    for m in ("ece_own_map", "shift", "rms_residual"):
        vals = r["decomposition"][m]["world_means"]
        print(f"  {m:<16}" + "  ".join(f"{d}:{v:+.4f}" for d, v in
                                       zip(r["dataset_seeds"], vals)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,E")
    ap.add_argument("--state", default="supply_stress")
    ap.add_argument("--seeds", default=",".join(str(d) for d in DSEEDS))
    ap.add_argument("--init-seeds", default=",".join(str(s) for s in INIT_SEEDS))
    ap.add_argument("--null-repeats", type=int, default=NULL_REPEATS)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    blob = {}
    for v in [s.strip() for s in a.variants.split(",") if s.strip()]:
        r = run(v, a.state, dseeds, iseeds, a.null_repeats)
        report(r)
        blob[v] = r
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(blob, f, indent=1, default=float)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
