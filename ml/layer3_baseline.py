#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 0 — freeze the baseline the four gates are measured against.

`layer3_neurosymbolic_build_prompt.md` §2 asks for two quantities, retained side by side:

    P(Y | X)        the entity's own observable features, no graph
    P(Y | X, H)     the trained SHARE encoder + Markov-blanket depth readout

for `delay`, `shortage` and `impact` at their **retained** depths -- `delay -> h^1`,
`shortage -> h^3`, `impact -> h^4` (`ml/models/rgcn_attn_markov_encoder.py`'s
`MARKOV_READOUT_DEPTH`, zero new parameters, unchanged here) -- across the full
**5 dataset seeds x 5 model-init seeds** grid, together with **both** reproduction floors.

**Why both floors, and why the max of them.** `reports/decision_support_build.md` §1.3.1 is the
reason this module exists in the form it does: a counterfactual sign agreement of 0.524 looked
like a 2.4-point edge over chance until the *model-init* floor was measured at 0.324, and the
same quantity averaged over three init seeds came out at 0.478 -- below chance. The mirror-image
trap is `reports/phase7_training_results.md` §5.6's: on `delay`, dataset-seed variance is ~99% of
total AUC variance, so a result stable across model seeds within one world says almost nothing.
A Layer 3 component is therefore admitted only if its delta clears

    floor_task = max( init-seed floor , dataset-seed floor )

**How each floor is defined here**, matching `ml/identifiability_check.py::summarise` so the two
modules' floors are the same object:

    init-seed floor    = max over worlds d of ( max_m AUC[d,m] - min_m AUC[d,m] )
    dataset-seed floor = max_d mean_m AUC[d,m] - min_d mean_m AUC[d,m]

The init floor is a max rather than a mean deliberately: it is the *worst* reproduction gap the
same configuration produces from nothing but a different seed, which is the number a candidate
improvement has to beat to be distinguishable from noise anywhere in the grid.

**`P(Y|X)`'s head is `ml/latent_state_head.py`'s, unmodified** -- the same `in -> 32 -> 1` MLP,
the same positive-weighted BCE, the same seeding -- so the two arms differ in the *representation*
they are handed and in nothing else, and the model-init seed means the same thing on both.

**Nothing here trains, modifies or touches SHARE or the Markov readout.** Backbones come from
`ml/ds_backbone.py`'s cache and `freeze()` asserts every parameter is frozen before use.

    python3 ml/layer3_baseline.py --variant 0 --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone  # noqa: E402
from ml.evaluate import collect_predictions  # noqa: E402
from ml.latent_state_head import train_head  # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE, TASKS  # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH  # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")


# --------------------------------------------------------------------------- P(Y|X)

def feature_table(bundles, task: str) -> tuple[np.ndarray, np.ndarray]:
    """`(X, y)` over the labelled rows of `bundles` for `task`, from the entity's OWN
    feature vector only.

    This is the whole of the `P(Y|X)` arm's input: `bundle.data[entity_type].x[idx]`, the same
    tensor SHARE's `lin_in` consumes at layer 0, with no message passing applied to it. There is
    no neighbourhood term, no aggregation, and no depth -- which is exactly what makes the gap
    against `P(Y|X,H)` attributable to the graph.
    """
    et = TASK_ENTITY_TYPE[task]
    Xs, ys = [], []
    for b in bundles:
        idx, y = b.labels[task]
        if idx.numel() == 0:
            continue
        Xs.append(b.data[et].x[idx].detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    if not Xs:
        return np.zeros((0, 0)), np.zeros((0,))
    return np.concatenate(Xs).astype(float), np.concatenate(ys).astype(float)


def px_auc(train_bundles, test_bundles, task: str, init_seed: int) -> float | None:
    Xtr, ytr = feature_table(train_bundles, task)
    Xte, yte = feature_table(test_bundles, task)
    if not len(Xtr) or not len(Xte):
        return None
    return train_head(Xtr, ytr, Xte, yte, init_seed)


# --------------------------------------------------------------------------- floors

def floors(auc_by_cell: dict, dseeds: list[int], mseeds: list[int]) -> dict:
    """Both reproduction floors plus the grid summary, per task.

    Cells that are `None` (a degenerate split for that task in that world) are dropped rather
    than imputed; the count that survived is reported so a thin cell cannot hide.
    """
    out = {}
    for task in TASKS:
        M = np.array([[auc_by_cell[(d, m)].get(task) if auc_by_cell[(d, m)].get(task)
                       is not None else np.nan for m in mseeds] for d in dseeds], dtype=float)
        if not np.isfinite(M).any():
            out[task] = {"available": False}
            continue
        per_world_spread = [float(np.nanmax(r) - np.nanmin(r)) for r in M
                            if np.isfinite(r).sum() > 1]
        world_means = np.array([np.nanmean(r) for r in M], dtype=float)
        init_floor = max(per_world_spread) if per_world_spread else float("nan")
        dset_floor = (float(np.nanmax(world_means) - np.nanmin(world_means))
                      if np.isfinite(world_means).sum() > 1 else float("nan"))
        out[task] = {
            "available": True,
            "n_cells": int(np.isfinite(M).sum()),
            "grand_mean": float(np.nanmean(M)),
            "per_world_mean": {str(d): (float(v) if np.isfinite(v) else None)
                               for d, v in zip(dseeds, world_means)},
            "per_world_init_spread": {str(d): (float(np.nanmax(M[i]) - np.nanmin(M[i]))
                                               if np.isfinite(M[i]).sum() > 1 else None)
                                      for i, d in enumerate(dseeds)},
            "init_seed_floor": init_floor,
            "dataset_seed_floor": dset_floor,
            "floor": max([f for f in (init_floor, dset_floor) if f == f] or [float("nan")]),
            "std_model": float(np.nanmean([np.nanstd(r, ddof=1) for r in M
                                           if np.isfinite(r).sum() > 1])),
            "std_dataset": float(np.nanstd(world_means, ddof=1)),
        }
    return out


# --------------------------------------------------------------------------- driver

def run(csv_root: str, variant: str, dseeds: list[int], mseeds: list[int],
        device: str = "cpu") -> dict:
    pxh: dict = {}
    px: dict = {}
    sat: dict = {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        for m in mseeds:
            model, meta = get_backbone(csv_dir, variant, d, m, device=device, verbose=False)
            pxh[(d, m)] = {t: meta["auc"].get(t) for t in TASKS}
            # The P(Y|X) arm needs the train split, which `get_backbone` does not return.
            from ml.ds_backbone import load_world
            tr, _va, te, _ids = load_world(csv_dir, device)
            px[(d, m)] = {t: px_auc(tr, te, t, m) for t in TASKS}
            if m == mseeds[0]:
                preds = collect_predictions(model, meta["test_bundles"])
                sat[d] = {t: _saturation(preds[t]["p"]) for t in TASKS}
            print(f"  d{d} m{m}  " + "  ".join(
                f"{t}: P(Y|X,H)={_f(pxh[(d,m)][t])} P(Y|X)={_f(px[(d,m)][t])}"
                for t in TASKS), flush=True)
    return {"pxh": pxh, "px": px, "saturation": sat}


def _saturation(p: np.ndarray) -> dict:
    """Where the prediction population sits. Carried into Phase 0 rather than left to Gate 1
    because `reports/decision_support_build.md` §3.2 found 96.5% of `delay` predictions at
    p >= 0.999, and that fact bounds what any relevance method can be asked to do."""
    if not len(p):
        return {"n": 0}
    return {"n": int(len(p)),
            "sat_high_frac": float((p >= 0.999).mean()),
            "sat_low_frac": float((p <= 1e-4).mean()),
            "explainable_frac": float(((p < 0.999) & (p > 1e-4)).mean()),
            "median_p": float(np.median(p))}


def _f(v) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "phase0_baseline.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    got = run(a.csv_root, a.variant, dseeds, mseeds, a.device)
    f_pxh = floors(got["pxh"], dseeds, mseeds)
    f_px = floors(got["px"], dseeds, mseeds)

    print("\n" + "=" * 104)
    print(f"PHASE 0 — baseline, variant {a.variant}, {len(dseeds)} worlds x {len(mseeds)} "
          f"init seeds  ({len(dseeds)*len(mseeds)} cells)")
    print("=" * 104)
    hdr = (f"{'task':<10}{'depth':>7}{'P(Y|X)':>10}{'P(Y|X,H)':>11}{'graph gain':>12}"
           f"{'init floor':>12}{'dset floor':>12}{'FLOOR':>10}{'gain>floor':>12}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for t in TASKS:
        if not f_pxh[t].get("available"):
            continue
        gain = f_pxh[t]["grand_mean"] - f_px[t]["grand_mean"]
        rows[t] = {"markov_depth": MARKOV_READOUT_DEPTH[t],
                   "p_y_given_x": f_px[t], "p_y_given_xh": f_pxh[t],
                   "graph_gain": gain,
                   "graph_gain_clears_floor": bool(gain > f_pxh[t]["floor"])}
        print(f"{t:<10}{'h^'+str(MARKOV_READOUT_DEPTH[t]):>7}{f_px[t]['grand_mean']:>10.4f}"
              f"{f_pxh[t]['grand_mean']:>11.4f}{gain:>+12.4f}"
              f"{f_pxh[t]['init_seed_floor']:>12.4f}{f_pxh[t]['dataset_seed_floor']:>12.4f}"
              f"{f_pxh[t]['floor']:>10.4f}"
              f"{('YES' if gain > f_pxh[t]['floor'] else 'no'):>12}")
    print("-" * len(hdr))
    print("\nsaturation of P(Y|X,H) on the test split (model seed "
          f"{mseeds[0]}), mean over worlds:")
    for t in TASKS:
        vals = [got["saturation"][d][t] for d in dseeds if got["saturation"][d][t].get("n")]
        if not vals:
            continue
        print(f"  {t:<10} p>=0.999: {statistics.fmean(v['sat_high_frac'] for v in vals):>7.1%}"
              f"   explainable band: "
              f"{statistics.fmean(v['explainable_frac'] for v in vals):>7.1%}"
              f"   median p: {statistics.fmean(v['median_p'] for v in vals):.4f}")

    blob = {"config": vars(a), "markov_depths": MARKOV_READOUT_DEPTH, "tasks": rows,
            "auc_grid_pxh": {f"{d}|{m}": got["pxh"][(d, m)] for d in dseeds for m in mseeds},
            "auc_grid_px": {f"{d}|{m}": got["px"][(d, m)] for d in dseeds for m in mseeds},
            "saturation": {str(d): got["saturation"][d] for d in got["saturation"]}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
