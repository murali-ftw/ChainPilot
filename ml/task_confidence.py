#!/usr/bin/env python3
"""
Layer 3 v2, PHASE 4 -- confidence, kept separate from probability, built cheaply.

**Probability and confidence are two different concepts and are not collapsed into one number.**
A calibrated probability of 0.85 says the estimated event probability is 85%; it says nothing
about whether *this particular estimate* is trustworthy. Every task therefore carries three
separate fields -- Prediction (Phase 3), Evidence status (Phase 5), Confidence (here).

**No new learned confidence head, no separate uncertainty network, no additional multi-objective
loss.** The confidence candidate is prediction disagreement across the existing ensemble, which
is free: the same 5 x 5 grid Phase 2 already builds for its floor computation. (A learned
confidence head was built and measured under the v1 redesign -- `reports/layer3_redesign_build.md`
Phase 3 -- and it did not beat the free `|p - 0.5|` margin on any arm, which is the empirical
reason this phase is specified the cheap way and the reason that route is not reopened here.)

---

**Validation, and why the dataset-seed axis is the binding part.**

A confidence signal is reported only if higher confidence actually corresponds to lower empirical
error. Init-seed disagreement alone can look perfectly stable while the model remains highly
seed-sensitive at the *dataset* level -- the axis already shown to carry 68-99% of total
variance -- so a check that varies only init seed can pass while hiding exactly the false
confidence that failed the old Step 4. Per-entity dataset-seed variance is **not identifiable**
on this benchmark (dataset seeds 42 and 43 generate different suppliers, so there is no entity to
pair across worlds -- `ml/uncertainty_ensemble.py`'s own stated limitation), and this module does
not fabricate it. What it does instead is require the relationship to survive the dataset-seed
axis in the two ways that ARE identifiable:

    per-world monotonicity   the confidence -> error relation must hold in EVERY world
                             independently, not only pooled. Pooling across worlds can
                             manufacture a monotone curve out of a between-world mean shift.
    cross-world transfer     the confidence BAND EDGES fitted on one world must still order
                             error on a different world, rotated over all ordered pairs. A
                             signal whose thresholds only work in the world that produced them
                             is not a deployable confidence signal.

**If higher confidence does not correspond to greater reliability, the signal is rejected** and
the Confidence field is reported as absent. An absent confidence field is a legitimate result and
is not backfilled with a number that failed its own validation.

A validated signal is discretised into High / Moderate / Low bands; a raw continuous score is
never reported as if it were confidence.

---

This module also owns the **shared prediction grid**, because Phase 3 needs the identical object:
`emit_pred_grid()` writes an `.npz` in exactly the key layout
`ml/uncertainty_calibrate.py::load_grid` expects (`{task}_{dseed}_{mseed}_{y|p}`), so Phase 3 runs
that module **unmodified, through its own CLI**, rather than reimplementing PAVA or ECE here.

    python3 ml/task_confidence.py --variant A --emit-preds out/layer3_v2/preds_A.npz \
        --out out/layer3_v2/phase4_confidence_A.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world                  # noqa: E402
from ml.hypothesis_ranker import roc_auc                             # noqa: E402
from ml.latent_state_head import assert_backbone_frozen              # noqa: E402
from ml.layer3_sufficiency import fit_probs                          # noqa: E402
from ml.task_incremental_value import task_rows                      # noqa: E402
from ml.task_latent_encoder import TASK_DEPTH, train_encoder         # noqa: E402

N_BANDS = 3
BAND_NAMES = ("Low", "Moderate", "High")


def brier(p, y):
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# the shared prediction grid (also Phase 3's input)
# ---------------------------------------------------------------------------

def build_grid(variant: str, tasks: list[str], dseeds: list[int], init_seeds: list[int],
               csv_root: str) -> dict:
    """`{(task, dseed, mseed): {"y":…, "p":…}}` for the Phase-2 `layer3` arm.

    Exactly the arm Phase 2 gated on: encoder fitted on `tr`, downstream head on `va` over
    `[X ; H ; Z]`, scored on `te`. `assert_backbone_frozen` runs inside every fit.
    """
    grid = {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        model, _ = get_backbone(csv_dir, variant, d, mseed=0, device="cpu")
        assert_backbone_frozen(model)
        tr, va, te, _ = load_world(csv_dir, "cpu")
        for task in tasks:
            R = {k: task_rows(model, b, task) for k, b in (("tr", tr), ("va", va), ("te", te))}
            if not all(R.values()):
                continue
            Ftr = np.concatenate([R["tr"]["X"], R["tr"]["H"]], axis=1)
            Fva = np.concatenate([R["va"]["X"], R["va"]["H"]], axis=1)
            Fte = np.concatenate([R["te"]["X"], R["te"]["H"]], axis=1)
            for m, s in enumerate(init_seeds):
                Zs = train_encoder(Ftr, R["tr"]["Y"], [Fva, Fte], s, model=model)
                if Zs is None:
                    continue
                A = np.concatenate([R["va"]["X"], R["va"]["H"], Zs[0]], axis=1)
                B = np.concatenate([R["te"]["X"], R["te"]["H"], Zs[1]], axis=1)
                pr = fit_probs(A, R["va"]["Y"], [B], s, model=model)
                if pr is not None:
                    grid[(task, d, m)] = {"y": R["te"]["Y"].astype(float), "p": pr[0]}
            print(f"  seed {d} {task:<9} h^{TASK_DEPTH[task]}  te {len(R['te']['Y']):,} rows "
                  f"({int(R['te']['Y'].sum()):,} pos)  mean AUC "
                  f"{statistics.fmean([roc_auc(grid[(task, d, m)]['p'], R['te']['Y'].astype(bool)) for m in range(len(init_seeds)) if (task, d, m) in grid]):.4f}",
                  flush=True)
    return grid


def emit_pred_grid(grid: dict, path: str) -> str:
    """Write the grid in `ml/uncertainty_calibrate.py::load_grid`'s own key layout, so Phase 3
    runs that module unmodified through its CLI."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {}
    for (task, d, m), v in grid.items():
        blob[f"{task}_{d}_{m}_y"] = v["y"]
        blob[f"{task}_{d}_{m}_p"] = v["p"]
    np.savez_compressed(path, **blob)
    return path


# ---------------------------------------------------------------------------
# the confidence candidate and its validation
# ---------------------------------------------------------------------------

def world_arrays(grid, task, d, members):
    P = np.stack([grid[(task, d, m)]["p"] for m in members])
    return P.mean(axis=0), P.var(axis=0, ddof=1), grid[(task, d, members[0])]["y"]


def band_edges(conf: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """Equal-count band edges on the confidence score. Returned so they can be FITTED on one
    world and APPLIED to another -- which is what the cross-world transfer check needs."""
    qs = np.linspace(0, 100, n_bands + 1)[1:-1]
    return np.percentile(conf, qs)


def bands_from_edges(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(conf, edges)


def band_stats(p, y, b, n_bands: int = N_BANDS) -> list:
    out = []
    for k in range(n_bands):
        m = b == k
        if m.sum() < 10:
            out.append(None)
            continue
        out.append({"band": BAND_NAMES[k], "n": int(m.sum()),
                    "brier": brier(p[m], y[m]),
                    "abs_error": float(np.abs(p[m] - y[m]).mean()),
                    "base_rate": float(y[m].mean())})
    return out


def monotone_decreasing_error(stats: list) -> bool:
    """Low -> Moderate -> High confidence must show non-increasing error."""
    v = [s["brier"] for s in stats if s]
    return len(v) == len([s for s in stats if s]) and all(
        v[i] >= v[i + 1] for i in range(len(v) - 1))


def validate(grid: dict, task: str, dseeds: list[int], members: list[int]) -> dict:
    """Per-world monotonicity + cross-world band-edge transfer + a member floor.

    Confidence is `-var` across the ensemble members (higher = more confident), which is the
    free signal the existing machinery already produces.
    """
    per_world, edges_of = {}, {}
    for d in dseeds:
        mu, var, y = world_arrays(grid, task, d, members)
        conf = -var
        e = band_edges(conf)
        edges_of[d] = e
        st = band_stats(mu, y, bands_from_edges(conf, e))
        per_world[d] = {"bands": st, "monotone": monotone_decreasing_error(st),
                        "mean_var": float(var.mean()),
                        "brier_low_minus_high": ((st[0]["brier"] - st[-1]["brier"])
                                                 if st[0] and st[-1] else None)}

    # CROSS-WORLD TRANSFER: edges fitted on one world, applied to a DIFFERENT world.
    folds = []
    for src, dst in itertools.permutations(dseeds, 2):
        mu, var, y = world_arrays(grid, task, dst, members)
        st = band_stats(mu, y, bands_from_edges(-var, edges_of[src]))
        folds.append({"edge_world": src, "test_world": dst,
                      "monotone": monotone_decreasing_error(st),
                      "brier_low_minus_high": ((st[0]["brier"] - st[-1]["brier"])
                                               if st[0] and st[-1] else None)})

    # Member floor on the band-Brier spread: how far the spread moves when the ensemble is
    # composed of a different subset of members. A "monotone" curve inside this is not evidence.
    floor_vals = []
    for combo in itertools.combinations(members, 3):
        sp = []
        for d in dseeds:
            mu, var, y = world_arrays(grid, task, d, list(combo))
            st = band_stats(mu, y, bands_from_edges(-var, band_edges(-var)))
            if st[0] and st[-1]:
                sp.append(st[0]["brier"] - st[-1]["brier"])
        if sp:
            floor_vals.append(statistics.fmean(sp))
    member_floor = (max(floor_vals) - min(floor_vals)) if len(floor_vals) > 1 else float("nan")

    spreads = [v["brier_low_minus_high"] for v in per_world.values()
               if v["brier_low_minus_high"] is not None]
    mean_spread = statistics.fmean(spreads) if spreads else float("nan")
    worlds_mono = sum(1 for v in per_world.values() if v["monotone"])
    folds_mono = sum(1 for f in folds if f["monotone"])

    accepted = bool(worlds_mono == len(dseeds)
                    and folds_mono >= (len(folds) + 1) // 2
                    and mean_spread > member_floor)
    reasons = []
    if worlds_mono != len(dseeds):
        reasons.append(f"not monotone in every world ({worlds_mono}/{len(dseeds)}) — pooling "
                       f"would have hidden this")
    if folds_mono < (len(folds) + 1) // 2:
        reasons.append(f"band edges do not transfer cross-world ({folds_mono}/{len(folds)} "
                       f"ordered pairs monotone)")
    if not (mean_spread > member_floor):
        reasons.append(f"Low-minus-High Brier spread {mean_spread:+.4f} does not clear the "
                       f"member floor {member_floor:.4f}")
    return {"per_world": per_world, "cross_world_folds": folds,
            "worlds_monotone": worlds_mono, "n_worlds": len(dseeds),
            "cross_world_folds_monotone": folds_mono, "n_folds": len(folds),
            "mean_brier_low_minus_high": mean_spread,
            "member_floor_on_spread": member_floor,
            "accepted": accepted, "reasons": reasons,
            "bands": list(BAND_NAMES) if accepted else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--tasks", default="delay,shortage,impact")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--emit-preds", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    print(f"building the shared 5x5 prediction grid (variant {a.variant})", flush=True)
    t = time.time()
    grid = build_grid(a.variant, tasks, dseeds, iseeds, a.csv_root)
    print(f"  ({time.time() - t:.0f}s)", flush=True)
    if a.emit_preds:
        print(f"  preds -> {emit_pred_grid(grid, a.emit_preds)}", flush=True)

    members = list(range(len(iseeds)))
    res = {"variant": a.variant, "dataset_seeds": dseeds, "init_seeds": iseeds,
           "tasks": tasks, "signal": "-var across ensemble members (no new head)",
           "per_task": {}}
    for task in tasks:
        if not any((task, d, 0) in grid for d in dseeds):
            continue
        res["per_task"][task] = validate(grid, task, dseeds, members)

    print("\n" + "=" * 118)
    print(f"PHASE 4 — confidence validation, variant {a.variant}")
    print("=" * 118)
    hdr = (f"{'task':<10}{'worlds mono':>13}{'x-world mono':>14}{'Low-High Brier':>16}"
           f"{'member floor':>14}{'CONFIDENCE':>13}")
    print(hdr); print("-" * len(hdr))
    for task, v in res["per_task"].items():
        print(f"{task:<10}{f'{v['worlds_monotone']}/{v['n_worlds']}':>13}"
              f"{f'{v['cross_world_folds_monotone']}/{v['n_folds']}':>14}"
              f"{v['mean_brier_low_minus_high']:>+16.4f}{v['member_floor_on_spread']:>14.4f}"
              f"{('ACCEPTED' if v['accepted'] else 'REJECTED'):>13}")
    print("-" * len(hdr))
    for task, v in res["per_task"].items():
        if not v["accepted"]:
            print(f"\n{task} — confidence REJECTED, field reported as absent:")
            for r in v["reasons"]:
                print(f"  - {r}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
