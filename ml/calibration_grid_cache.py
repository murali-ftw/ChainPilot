#!/usr/bin/env python3
"""
Shared prediction cache for the world-conditioned calibration follow-up.

Every module in this follow-up (`calibration_curve_compare.py`,
`calibration_variance_decompose.py`, `world_conditioned_calibration.py`) needs the *same*
object: Model A's per-entity probabilities on a 5 worlds x 5 head-init-seeds grid, exactly as
`ml/layer3_uncertainty.py::build_grid` produces them. Building it takes ~11 s per world and is
bit-deterministic (frozen backbone loaded from cache, head init seeded), so it is built once and
cached to `.npz` rather than rebuilt per module. Nothing here re-implements the grid -- it calls
`build_grid` and serialises what comes back.

**This module trains nothing.** `build_grid` loads the `m0` backbone checkpoint from
`out/ds_ckpt/` and calls `assert_backbone_frozen` after head construction and after every head
fit, same as STEP 3 did. SHARE and the Layer-2 Markov readout are untouched.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_CSV_ROOT = os.path.join(REPO, "db", "csv_v1scale")
CACHE_DIR = os.path.join(REPO, "out", "wcc")
DSEEDS = [42, 43, 44, 45, 46]
INIT_SEEDS = [0, 1, 2, 3, 4]


def best_depth(variant: str, state: str) -> int:
    """Phase 1's recorded best depth for this (variant, state) -- read, never re-selected."""
    p1 = os.path.join(REPO, "out", "phase1", f"heads_{variant}.json")
    return int(json.load(open(p1))["summary"][state]["best_depth"])


def cache_path(variant: str, state: str) -> str:
    return os.path.join(CACHE_DIR, f"grid_{variant}_{state}.npz")


def load_grid(variant: str = "A", state: str = "supply_stress",
              dseeds: list[int] | None = None, init_seeds: list[int] | None = None,
              csv_root: str = DEFAULT_CSV_ROOT, config: str = "v1",
              rebuild: bool = False, verbose: bool = True) -> tuple[dict, int]:
    """`({dseed: {p_va, y_va, p_te, y_te, median_threshold}}, depth)`.

    Cache hit is exact, not merely equivalent: the backbone checkpoint, the temporal split, the
    train-median binarisation and the head init seeds are all fixed, so re-running `build_grid`
    reproduces the same arrays. `--rebuild` forces the fit anyway, which is what the Steps 1-3
    reproduction check uses.
    """
    dseeds = dseeds or DSEEDS
    init_seeds = init_seeds or INIT_SEEDS
    depth = best_depth(variant, state)
    path = cache_path(variant, state)

    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=False)
        grid = {int(d): {"p_va": z[f"{d}_p_va"], "y_va": z[f"{d}_y_va"],
                         "p_te": z[f"{d}_p_te"], "y_te": z[f"{d}_y_te"],
                         "median_threshold": float(z[f"{d}_thr"])}
                for d in dseeds}
        if verbose:
            print(f"  [cached] {os.path.basename(path)}  "
                  f"{len(dseeds)} worlds x {len(init_seeds)} init seeds @ h^{depth}", flush=True)
        return grid, depth

    from ml.layer3_uncertainty import build_grid  # noqa: PLC0415 -- heavy import, on demand
    if verbose:
        print(f"\nbuilding grid: variant {variant}, {state} @ h^{depth}, "
              f"{len(dseeds)} worlds x {len(init_seeds)} head init seeds", flush=True)
    grid = build_grid(variant, state, depth, dseeds, config, init_seeds, csv_root)

    os.makedirs(CACHE_DIR, exist_ok=True)
    blob = {}
    for d, g in grid.items():
        # Dtypes are preserved EXACTLY as `build_grid` produced them. This is not fussiness:
        # the head emits float32, so `ens`'s mean over members is a float32 reduction. Widening
        # the cache to float64 changes that mean in the ~1e-9 place, which is enough to flip a
        # tie in PAVA's stable sort or in `ece`'s equal-count bin edges -- and on one world that
        # moved ECE by 8e-4. Small against the member floor, but it would mean the cached grid
        # was not the object STEP 3 measured, which is the one thing this cache must guarantee.
        for k in ("p_va", "y_va", "p_te", "y_te"):
            blob[f"{d}_{k}"] = np.asarray(g[k])
        blob[f"{d}_thr"] = np.float64(g["median_threshold"])
    np.savez_compressed(path, **blob)
    if verbose:
        print(f"  cached -> {path}", flush=True)
    return grid, depth


def ens(grid: dict, d: int, split: str = "te", members=(0, 1, 2, 3, 4)):
    """Ensemble mean, per-entity across-member variance, labels -- within one world.

    Identical to `ml/layer3_uncertainty.py::ens`; duplicated here only so the cache module has
    no import-time dependency on the heavy Layer-3 module.
    """
    P = grid[d][f"p_{split}"][list(members)]
    return P.mean(axis=0), P.var(axis=0, ddof=1), grid[d][f"y_{split}"]
