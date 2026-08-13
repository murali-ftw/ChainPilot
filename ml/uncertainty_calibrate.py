#!/usr/bin/env python3
"""
Phase 2 calibration — are the ensemble's probabilities honest, and is the claim above noise?

Reuses, rather than reimplements, the isotonic-regression and ECE machinery already written
and validated in `ml/hypothesis_ranker.py`. That code is checked by
`ml/test_hypothesis_calibration.py` (13 checks: ROC-AUC and average precision against sklearn
to 1e-9, ECE returning ~0.0016 on a predictor calibrated by construction at 200,000 samples and
**exactly 0.60** on a 0.9-always predictor of a 30% class, isotonic monotone, repairing a
squashed predictor, leaving a calibrated one alone, and never improving ranking out of sample).
Rewriting any of that here would discard a validated implementation for an unvalidated one.

**Protocol, matching `reports/layer3_testing.md` §10.3.** The calibration map is fitted on a
**held-out world**, never on the world it is evaluated on. §5.6's finding -- dataset-seed
variance up to 10.3x model-seed variance -- makes within-world calibration fitting optimistic
in exactly the place the uncertainty lives, so the split is by world and the rotation covers
every choice of held-out pair.

**And the calibration claim gets its own floor.** Per this project's standing discipline
(`reports/layer3_testing.md` §4, §9.2, §10.3.1), a metric is not believed until it is known how
far it moves with nothing changed. Two floors are relevant and they are different objects:

* the **fit floor** -- refit the isotonic map on fixed member predictions. Deterministic given
  fixed inputs, so this is expected to be exactly 0.0000, exactly as §10.3.1 found for the
  comparable head. Reported as a fact, not celebrated as a result.
* the **member floor** -- the operative one: vary which model seeds compose the ensemble and
  re-measure. That is what an ECE difference has to clear.

    python3 ml/uncertainty_calibrate.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml.hypothesis_ranker import ece, isotonic_apply, isotonic_fit, roc_auc  # noqa: E402
from ml.models.depth import TASKS  # noqa: E402

N_BINS = 10


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def load_grid(npz_path: str, dseeds: list[int], mseeds: list[int]) -> dict:
    z = np.load(npz_path)
    out = {}
    for t in TASKS:
        for d in dseeds:
            for m in mseeds:
                ky, kp = f"{t}_{d}_{m}_y", f"{t}_{d}_{m}_p"
                if ky in z:
                    out[(t, d, m)] = {"y": z[ky], "p": z[kp]}
    return out


def ensemble_world(grid: dict, task: str, d: int, mseeds: list[int]) -> tuple:
    """Ensemble mean over model seeds within one world, with its label vector."""
    P = np.stack([grid[(task, d, m)]["p"] for m in mseeds])
    y = grid[(task, d, mseeds[0])]["y"]
    for m in mseeds[1:]:
        assert np.array_equal(grid[(task, d, m)]["y"], y), "labels differ across model seeds"
    return P.mean(axis=0), P.var(axis=0, ddof=1), y


def evaluate_fold(grid: dict, task: str, member_ds: list[int], calib_d: int, test_d: int,
                  mseeds: list[int]) -> dict:
    """One rotation: fit isotonic on `calib_d`, report on `test_d`.

    Both the single-model and the ensemble arms are reported, because the interesting question
    is not only "is it calibrated" but "what did ensembling buy over one model".
    """
    mu_c, _, y_c = ensemble_world(grid, task, calib_d, mseeds)
    mu_t, var_t, y_t = ensemble_world(grid, task, test_d, mseeds)
    single_t = grid[(task, test_d, mseeds[0])]["p"]
    single_c = grid[(task, calib_d, mseeds[0])]["p"]

    curve_ens = isotonic_fit(mu_c, y_c.astype(float))
    curve_one = isotonic_fit(single_c, y_c.astype(float))
    cal_ens = isotonic_apply(curve_ens, mu_t)
    cal_one = isotonic_apply(curve_one, single_t)

    def pack(p, tag):
        e, curve = ece(p, y_t, N_BINS)
        return {f"ece_{tag}": e, f"brier_{tag}": brier(p, y_t),
                f"nll_{tag}": nll(p, y_t), f"auc_{tag}": roc_auc(p, y_t),
                f"reliability_{tag}": curve}

    out = {"task": task, "calib_world": calib_d, "test_world": test_d,
           "n": int(len(y_t)), "positives": int(y_t.sum()),
           "mean_var_model": float(var_t.mean())}
    out.update(pack(single_t, "single_raw"))
    out.update(pack(cal_one, "single_cal"))
    out.update(pack(mu_t, "ens_raw"))
    out.update(pack(cal_ens, "ens_cal"))
    return out


def member_floor(grid: dict, task: str, calib_d: int, test_d: int, mseeds: list[int],
                 k: int = 3) -> dict:
    """How far ECE moves when the ensemble is composed of a different subset of model seeds.

    This is the operative floor: the fit itself is deterministic, so the uncertainty that
    matters is which members happened to be trained.
    """
    vals = []
    for combo in itertools.combinations(mseeds, k):
        mu_c, _, y_c = ensemble_world(grid, task, calib_d, list(combo))
        mu_t, _, y_t = ensemble_world(grid, task, test_d, list(combo))
        cal = isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t)
        e, _ = ece(cal, y_t, N_BINS)
        vals.append(e)
    if len(vals) < 2:
        return {}
    d = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
    return {"n_subsets": len(vals), "subset_size": k, "mean": float(np.mean(vals)),
            "mean_abs_dev": float(np.mean(d)), "max_abs_dev": float(np.max(d))}


def fit_floor(grid: dict, task: str, calib_d: int, test_d: int, mseeds: list[int],
              n: int = 8) -> dict:
    """Refit the calibration map `n` times on identical inputs.

    Expected to be exactly 0.0000 -- PAVA is deterministic. Reported so the distinction
    between a deterministic fit and a genuinely stable result is on the record, the same
    distinction §10.3.1 drew.
    """
    vals = []
    for _ in range(n):
        mu_c, _, y_c = ensemble_world(grid, task, calib_d, mseeds)
        mu_t, _, y_t = ensemble_world(grid, task, test_d, mseeds)
        cal = isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t)
        e, _ = ece(cal, y_t, N_BINS)
        vals.append(e)
    d = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
    return {"n_repeats": n, "mean_abs_dev": float(np.mean(d)) if d else 0.0,
            "max_abs_dev": float(np.max(d)) if d else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default=os.path.join(REPO, "out", "uncertainty_grid_preds.npz"))
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "uncertainty_calibration.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]
    grid = load_grid(a.preds, dseeds, mseeds)

    blob = {"config": vars(a), "folds": [], "floors": {}}
    for task in TASKS:
        for calib_d, test_d in itertools.permutations(dseeds, 2):
            member_ds = [d for d in dseeds if d not in (calib_d, test_d)]
            try:
                blob["folds"].append(
                    evaluate_fold(grid, task, member_ds, calib_d, test_d, mseeds))
            except (KeyError, AssertionError) as exc:
                blob.setdefault("skipped", []).append(
                    {"task": task, "calib": calib_d, "test": test_d, "reason": str(exc)})
        blob["floors"][task] = {
            "fit_floor": fit_floor(grid, task, dseeds[0], dseeds[1], mseeds),
            "member_floor": member_floor(grid, task, dseeds[0], dseeds[1], mseeds),
        }

    print(f"\n{'task':<10}{'arm':<12}{'ECE':>9}{'Brier':>9}{'NLL':>9}{'AUC':>9}")
    for task in TASKS:
        f = [x for x in blob["folds"] if x["task"] == task]
        if not f:
            continue
        for tag in ("single_raw", "single_cal", "ens_raw", "ens_cal"):
            print(f"{task:<10}{tag:<12}"
                  f"{np.mean([x[f'ece_{tag}'] for x in f]):>9.4f}"
                  f"{np.mean([x[f'brier_{tag}'] for x in f]):>9.4f}"
                  f"{np.mean([x[f'nll_{tag}'] for x in f]):>9.4f}"
                  f"{np.mean([x[f'auc_{tag}'] for x in f if x[f'auc_{tag}'] is not None]):>9.4f}")
        fl = blob["floors"][task]
        print(f"{'':<10}{'floors':<12} fit max_abs={fl['fit_floor']['max_abs_dev']:.6f}   "
              f"member max_abs={fl['member_floor'].get('max_abs_dev', float('nan')):.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
