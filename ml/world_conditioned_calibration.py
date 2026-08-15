#!/usr/bin/env python3
"""
STEPS 1-3 (reproduction) and STEP 7 (world-conditioned calibration) of the follow-up to
`reports/layer3_uncertainty_aware.md` STEP 3's **STOP — G**.

STEP 3 established, on Supply Stress / Variants A and E: the miscalibration is real (raw ECE at
2.22x / 2.18x the binomial floor), a within-world isotonic fit removes essentially all of it
(+0.0202 / +0.0224), and a map fitted on one world **harms** a different one (-0.0054 / -0.0029
over 20 leave-one-world-out folds; 7/20 and 8/20 folds improved, against member floors of 0.0132
/ 0.0117). Fitting within-world is leakage, so it classified STOP — G.

This module asks whether *conditioning* the map on the world rescues it, and it builds both
readings of "world-conditioned" because they have completely different status:

**7a — conditioned on dataset-seed identity. Diagnostic only, never deployable.** A per-world
isotonic map, and a single pooled head taking a one-hot world ID. Both can only be evaluated
in-sample over the five known worlds: a sixth world has no seed ID, so 7a has no defined
behaviour on it at all. If 7a recovers the within-world ceiling, that CONFIRMS the STEP 5
world-variance hypothesis. It does not produce a shippable calibrator, and nothing downstream
may treat it as one.

**7b — conditioned on observable graph-computed summary statistics. Deployable, and tested as
such.** The conditioning vector comes from `ml/world_summary_features.py`: five statistics read
off the graph, no seed ID and no generator config. It is evaluated under the **same
leave-one-world-out protocol STEP 3 used** -- fit on four worlds, test on the fifth, rotate --
so it is a fair test of generalisation to a world the map has never seen.

**Two controls the comparison cannot do without.** 7b differs from STEP 3's baseline in *two*
ways at once: it conditions on features, and it pools four worlds instead of fitting on one. So
`pool4_isotonic` and `pool4_platt` are run alongside -- the same pooled four worlds, no
conditioning. Without them an improvement over STEP 3 could be entirely the pooling, and the
conditioning could be worth nothing or less than nothing. STEP 3's own baseline is reproduced
here as `step3_pairwise` (the 4 single-world maps applied to the held-out world, averaged), which
aggregates to exactly the 20-fold number STEP 3 reports.

**Gating.** Per the adopt-only-if-gated rule, an improvement by 7b over STEP 3's baseline is not
adoption. 7b's metric gets its **own freshly measured reproduction floor** -- STEP 3's member
floor (0.0132 / 0.0117) and the within-world ceiling answer different questions and are not
inherited. Two floors are measured, matching `ml/uncertainty_calibrate.py`'s pair:

* the **refit floor** -- 10 identically configured refits, nothing changed. The solver is IRLS
  with a fixed ridge path and no random initialisation, so this is expected to be exactly 0 and
  is reported as a fact rather than as a result -- the same thing STEP 3's fit floor did.
* the **member floor** -- re-run the entire 5-fold 7b pipeline once per 3-of-5 head-init-seed
  subset (10 refits) and take the maximum pairwise deviation of the resulting improvement. This
  is the operative floor, and it is measured on 7b's own metric in 7b's own setup.

    python3 ml/world_conditioned_calibration.py --variants A,E --out out/wcc/step7.json
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

from ml.calibration_grid_cache import DSEEDS, INIT_SEEDS, ens, load_grid  # noqa: E402
from ml.hypothesis_ranker import ece, isotonic_apply, isotonic_fit, roc_auc  # noqa: E402
from ml.world_summary_features import FEATURES, load_all                 # noqa: E402

N_BINS = 10
EPS = 1e-6
RIDGE_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
REFIT_FLOOR_REPEATS = 10


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logit(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ---------------------------------------------------------------------------
# a deterministic ridge logistic solver
# ---------------------------------------------------------------------------

def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 60,
                 tol: float = 1e-9) -> np.ndarray:
    """IRLS / Newton with an L2 penalty on every coefficient except the intercept (column 0).

    Deterministic given `(X, y, lam)` -- no random initialisation, no minibatching, no early
    stopping on a shuffled stream. That is a deliberate choice: it makes the "refit with nothing
    changed" floor exactly zero and therefore *uninformative*, which is the honest state of
    affairs rather than a floor manufactured out of optimiser noise. The operative floor is
    measured on the ensemble-composition axis instead.

    The penalty is not optional: STEP 7b's conditioning block is built from at most four distinct
    world-level feature values, so the design matrix is close to rank-deficient by construction
    and an unpenalised fit is not identified.
    """
    n, k = X.shape
    w = np.zeros(k)
    pen = np.full(k, lam)
    pen[0] = 0.0
    P = np.diag(pen)
    for _ in range(iters):
        p = sigmoid(X @ w)
        s = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (p - y) + pen * w
        H = (X * s[:, None]).T @ X + P
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(k), g)
        except np.linalg.LinAlgError:                    # pragma: no cover
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


# ---------------------------------------------------------------------------
# design matrices
# ---------------------------------------------------------------------------

def design(p: np.ndarray, phi: np.ndarray | None) -> np.ndarray:
    """`[1, z, phi, z*phi]` with `z = logit(p)`.

    The interaction block is what makes this a *conditional* calibrator rather than a global one
    with extra offsets: without `z*phi` the world features could only shift the curve, and STEP 4
    reports the shift term is not what dominates the between-world difference.
    """
    z = logit(p)[:, None]
    cols = [np.ones_like(z), z]
    if phi is not None and phi.size:
        F = np.tile(phi, (len(p), 1)) if phi.ndim == 1 else phi
        cols += [F, z * F]
    return np.hstack(cols)


def usable_features(feats: dict, dseeds: list[int], names=FEATURES) -> list[str]:
    """Drop features with no between-world variation *in this variant*.

    On Variant E this removes `upstream_edge_density`: those worlds emit no
    `(Supplier, UPSTREAM_OF, Supplier)` relation at all, so the column is identically zero. A
    constant column cannot be standardised and cannot condition anything; carrying it would only
    add an unidentified coefficient.
    """
    keep = []
    for k in names:
        col = np.array([feats[d][k] for d in dseeds], dtype=float)
        if col.std() > 1e-12:
            keep.append(k)
    return keep


def standardiser(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/scale from the TRAINING worlds only. Using all five would leak the held-out world."""
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def score(p, y) -> dict:
    e, _ = ece(p, y, N_BINS)
    return {"ece": e, "brier": brier(p, y), "auc": roc_auc(p, y.astype(bool)),
            "n": int(len(y)), "positives": int(y.sum())}


def _pooled(grid, worlds, members):
    P = np.concatenate([ens(grid, d, "te", members)[0] for d in worlds])
    Y = np.concatenate([ens(grid, d, "te", members)[2] for d in worlds])
    return P, Y


def select_ridge(grid, train_worlds, members, feats, names, mu, sd) -> tuple[float, list]:
    """Inner leave-one-world-out over the FOUR training worlds only.

    The held-out world's test partition is not read here in any form -- not its labels, not its
    predictions, not its summary features. Selecting the penalty on the outer test world would
    be exactly the leakage STEP 3 refused, dressed as a hyperparameter.
    """
    trace = []
    for lam in RIDGE_GRID:
        vals = []
        for held in train_worlds:
            inner = [d for d in train_worlds if d != held]
            Xs, Ys = [], []
            for d in inner:
                p, _, y = ens(grid, d, "te", members)
                Xs.append(design(p, (np.array([feats[d][k] for k in names]) - mu) / sd))
                Ys.append(y)
            w = fit_logistic(np.vstack(Xs), np.concatenate(Ys), lam)
            p, _, y = ens(grid, held, "te", members)
            X = design(p, (np.array([feats[held][k] for k in names]) - mu) / sd)
            vals.append(ece(sigmoid(X @ w), y, N_BINS)[0])
        trace.append({"lam": lam, "inner_mean_ece": statistics.fmean(vals)})
    best = min(trace, key=lambda t: t["inner_mean_ece"])
    return best["lam"], trace


def fold(grid, dseeds, held, members, feats, names) -> dict:
    """One leave-one-world-out fold: fit on the other four worlds, evaluate on `held`."""
    train = [d for d in dseeds if d != held]
    p_t, _, y_t = ens(grid, held, "te", members)
    out = {"held_out_world": held, "train_worlds": train, "raw": score(p_t, y_t)}

    # -- STEP 3's own baseline, regrouped: the 4 single-world maps, averaged --------------
    pair = []
    for c in train:
        p_c, _, y_c = ens(grid, c, "te", members)
        pair.append(score(isotonic_apply(isotonic_fit(p_c, y_c.astype(float)), p_t), y_t))
    out["step3_pairwise"] = {k: statistics.fmean([r[k] for r in pair])
                             for k in ("ece", "brier", "auc")}
    out["step3_pairwise_per_calib_world"] = dict(zip(train, [r["ece"] for r in pair]))

    # -- controls: same four worlds pooled, NO conditioning --------------------------------
    P4, Y4 = _pooled(grid, train, members)
    out["pool4_isotonic"] = score(
        isotonic_apply(isotonic_fit(P4, Y4.astype(float)), p_t), y_t)
    w0 = fit_logistic(design(P4, None), Y4, lam=0.0)
    out["pool4_platt"] = score(sigmoid(design(p_t, None) @ w0), y_t)

    # -- 7b: conditioned on observable summary features -------------------------------------
    F_train = np.array([[feats[d][k] for k in names] for d in train], dtype=float)
    mu, sd = standardiser(F_train)
    for tag, nm in (("cond_full", names), ("cond_single", names[:1])):
        m, s = standardiser(np.array([[feats[d][k] for k in nm] for d in train], dtype=float))
        lam, trace = select_ridge(grid, train, members, feats, nm, m, s)
        Xs, Ys = [], []
        for d in train:
            p, _, y = ens(grid, d, "te", members)
            Xs.append(design(p, (np.array([feats[d][k] for k in nm]) - m) / s))
            Ys.append(y)
        w = fit_logistic(np.vstack(Xs), np.concatenate(Ys), lam)
        X = design(p_t, (np.array([feats[held][k] for k in nm]) - m) / s)
        out[tag] = score(sigmoid(X @ w), y_t)
        out[tag].update({"lambda": lam, "features": list(nm), "ridge_trace": trace,
                         "coef": w.tolist()})
    out["standardiser"] = {"mean": mu.tolist(), "sd": sd.tolist(), "features": list(names)}
    return out


ARMS = ("raw", "step3_pairwise", "pool4_isotonic", "pool4_platt", "cond_full", "cond_single")


def step7b(grid, dseeds, members, feats, names) -> dict:
    folds = [fold(grid, dseeds, d, members, feats, names) for d in dseeds]
    mean = {a: {k: statistics.fmean([f[a][k] for f in folds])
                for k in ("ece", "brier", "auc")} for a in ARMS}
    base = mean["raw"]["ece"]
    return {
        "protocol": "leave-one-world-out, fit on 4 worlds, test on the held-out 5th, 5 folds",
        "folds": folds, "mean": mean,
        "ece_improvement_vs_raw": {a: base - mean[a]["ece"] for a in ARMS},
        "folds_improved_vs_raw": {
            a: int(sum(1 for f in folds if f["raw"]["ece"] - f[a]["ece"] > 0)) for a in ARMS},
        "improvement_per_fold": {
            a: [f["raw"]["ece"] - f[a]["ece"] for f in folds] for a in ARMS},
    }


def step7a(grid, dseeds, members) -> dict:
    """Conditioned on dataset-seed identity. IN-SAMPLE ONLY — see module docstring.

    Two forms, because the prompt admits either and they are not the same object: a per-world
    isotonic map (which is exactly STEP 3's within-world leaky arm, reproduced here so 7a and 7b
    sit on one table), and a single pooled head taking a one-hot world ID -- the direct
    architectural analogue of 7b's head, which is what makes the 7a/7b comparison apples to
    apples rather than isotonic-versus-logistic.
    """
    per_iso, per_head = [], []
    Xs, Ys = [], []
    idx = {d: i for i, d in enumerate(dseeds)}
    for d in dseeds:
        p, _, y = ens(grid, d, "te", members)
        oh = np.zeros(len(dseeds))
        oh[idx[d]] = 1.0
        Xs.append(design(p, oh))
        Ys.append(y)
        cal = isotonic_apply(isotonic_fit(p, y.astype(float)), p)
        per_iso.append({"world": d, "raw": score(p, y), "cal": score(cal, y)})
    w = fit_logistic(np.vstack(Xs), np.concatenate(Ys), lam=0.0)
    for d in dseeds:
        p, _, y = ens(grid, d, "te", members)
        oh = np.zeros(len(dseeds))
        oh[idx[d]] = 1.0
        per_head.append({"world": d, "cal": score(sigmoid(design(p, oh) @ w), y)})

    raw = statistics.fmean([r["raw"]["ece"] for r in per_iso])
    iso = statistics.fmean([r["cal"]["ece"] for r in per_iso])
    head = statistics.fmean([r["cal"]["ece"] for r in per_head])
    return {
        "status": "DIAGNOSTIC ONLY — in-sample over the 5 known worlds; undefined on a 6th world",
        "per_world_isotonic": per_iso, "per_world_onehot_head": per_head,
        "mean_ece_raw": raw, "mean_ece_isotonic": iso, "mean_ece_onehot_head": head,
        "improvement_isotonic": raw - iso, "improvement_onehot_head": raw - head,
        "coef_onehot_head": w.tolist(),
    }


# ---------------------------------------------------------------------------
# why 7b behaves the way it does
# ---------------------------------------------------------------------------

def feature_curve_association(grid, dseeds, members, feats, names) -> dict:
    """Does any summary feature track a world's calibration curve at all?

    If 7b fails, there are two different reasons it could fail and they call for different
    follow-ups: *the features carry no information about calibration*, or *four training worlds
    cannot identify a dependence on them even if it exists*. This separates them by asking the
    question directly -- correlate each feature, across the five worlds, with three descriptions
    of that world's calibration: its raw ECE, the shift of its own isotonic curve away from the
    pooled curve, and how much its own map improves its ECE.

    **With five worlds this is a 5-point correlation, which is very weak evidence, so the
    p-value is exact rather than asymptotic**: all 120 permutations of five labels are
    enumerated. |r| must reach ~0.878 to clear p < 0.05 two-tailed at n = 5. A null result here
    is a non-detection at this scale, not a demonstration that the features are uninformative --
    the same framing every other result in this project carries.
    """
    from ml.calibration_curve_compare import decompose, shared_grid  # noqa: PLC0415

    mus = {d: ens(grid, d, "te", members)[0] for d in dseeds}
    ys = {d: ens(grid, d, "te", members)[2] for d in dseeds}
    x, _ = shared_grid([mus[d] for d in dseeds])
    pooled = isotonic_apply(
        isotonic_fit(np.concatenate([mus[d] for d in dseeds]),
                     np.concatenate([ys[d] for d in dseeds]).astype(float)), x)
    targets = {"ece_raw": [], "curve_shift_vs_pooled": [], "own_map_ece_gain": []}
    for d in dseeds:
        curve = isotonic_fit(mus[d], ys[d].astype(float))
        raw = ece(mus[d], ys[d], N_BINS)[0]
        targets["ece_raw"].append(raw)
        targets["curve_shift_vs_pooled"].append(
            decompose(isotonic_apply(curve, x), pooled, x)["shift"])
        targets["own_map_ece_gain"].append(
            raw - ece(isotonic_apply(curve, mus[d]), ys[d], N_BINS)[0])

    def exact_p(a: np.ndarray, b: np.ndarray, r_obs: float) -> float:
        perms = [np.corrcoef(a, b[list(q)])[0, 1] for q in itertools.permutations(range(len(b)))]
        return float(np.mean([abs(v) >= abs(r_obs) - 1e-12 for v in perms]))

    out = {"n_worlds": len(dseeds), "targets": targets, "association": {}}
    for k in names:
        f = np.array([feats[d][k] for d in dseeds], dtype=float)
        out["association"][k] = {}
        for t, vals in targets.items():
            v = np.array(vals, dtype=float)
            r = float(np.corrcoef(f, v)[0, 1]) if f.std() > 0 and v.std() > 0 else float("nan")
            out["association"][k][t] = {"pearson_r": r, "exact_p": exact_p(f, v, r)}
    sig = [(k, t) for k, d_ in out["association"].items()
           for t, s in d_.items() if s["exact_p"] < 0.05]
    out["significant_at_p05"] = [{"feature": k, "target": t} for k, t in sig]
    out["n_tests"] = len(names) * len(targets)
    return out


# ---------------------------------------------------------------------------
# 7b's own reproduction floor
# ---------------------------------------------------------------------------

def refit_floor(grid, dseeds, members, feats, names,
                repeats: int = REFIT_FLOOR_REPEATS) -> dict:
    """`repeats` identically configured refits, nothing changed."""
    vals = [step7b(grid, dseeds, members, feats, names)["ece_improvement_vs_raw"]["cond_full"]
            for _ in range(repeats)]
    dev = [abs(a - b) for a, b in itertools.combinations(vals, 2)]
    return {"n_repeats": repeats, "values": vals,
            "max_abs_dev": float(max(dev)) if dev else 0.0,
            "note": "IRLS is deterministic given (X, y, lam); an exactly-zero floor here is a "
                    "property of the solver, not evidence that the metric is stable"}


def member_floor(grid, dseeds, all_members, feats, names) -> dict:
    """Re-run the whole 5-fold 7b pipeline once per 3-of-5 init-seed subset. The operative floor.

    Freshly measured on 7b's own metric in 7b's own setup -- STEP 3's member floor (0.0132 /
    0.0117) was measured on a different arm answering a different question and is not inherited.
    """
    out = {}
    for combo in itertools.combinations(all_members, 3):
        r = step7b(grid, dseeds, list(combo), feats, names)
        for a in ("cond_full", "cond_single", "pool4_isotonic", "pool4_platt", "step3_pairwise"):
            out.setdefault(a, []).append(r["ece_improvement_vs_raw"][a])
    return {"n_subsets": len(next(iter(out.values()))),
            "per_arm": {a: {"values": v,
                            "max_abs_dev": float(max(abs(x - y) for x, y in
                                                     itertools.combinations(v, 2)))}
                        for a, v in out.items()}}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(variant: str, state: str, dseeds: list[int], init_seeds: list[int],
        skip_floor: bool = False) -> dict:
    grid, depth = load_grid(variant, state, dseeds, init_seeds)
    members = list(range(len(init_seeds)))
    feats = load_all((variant,), tuple(dseeds))[variant]
    names = usable_features(feats, dseeds)

    a7 = step7a(grid, dseeds, members)
    b7 = step7b(grid, dseeds, members, feats, names)
    out = {"variant": variant, "state": state, "depth": depth,
           "dataset_seeds": dseeds, "init_seeds": init_seeds,
           "features_available": list(FEATURES), "features_used": names,
           "features_dropped_no_between_world_variation":
               [k for k in FEATURES if k not in names],
           "step7a": a7, "step7b": b7,
           "feature_curve_association":
               feature_curve_association(grid, dseeds, members, feats, names)}
    if not skip_floor:
        out["floors_7b"] = {
            "refit": refit_floor(grid, dseeds, members, feats, names),
            "member": member_floor(grid, dseeds, members, feats, names),
        }
        mf = out["floors_7b"]["member"]["per_arm"]
        imp = b7["ece_improvement_vs_raw"]
        # The question the headline table cannot answer on its own. 7b changes TWO things at
        # once relative to STEP 3's baseline -- it conditions on features, and it pools four
        # worlds instead of fitting on one. Beating STEP 3 therefore does not mean conditioning
        # worked; the pooling controls may be carrying all of it. This contrasts each
        # conditioned arm against each unconditioned control on the SAME folds, and floors the
        # contrast on the same 3-of-5 init-seed subsets, so the difference gets its own floor
        # rather than borrowing either arm's.
        out["conditioning_contrast"] = {
            f"{a}_minus_{c}": {
                "delta_improvement": imp[a] - imp[c],
                "floor_max_abs_dev": float(max(
                    abs(x - y) for x, y in itertools.combinations(
                        [p - q for p, q in zip(mf[a]["values"], mf[c]["values"])], 2))),
                "conditioning_adds_anything": bool(
                    (imp[a] - imp[c]) > max(
                        abs(x - y) for x, y in itertools.combinations(
                            [p - q for p, q in zip(mf[a]["values"], mf[c]["values"])], 2))),
            }
            for a in ("cond_full", "cond_single")
            for c in ("pool4_platt", "pool4_isotonic")}
        out["gate_7b"] = {
            arm: {"improvement_vs_raw": imp[arm],
                  "improvement_vs_step3_baseline": imp[arm] - imp["step3_pairwise"],
                  "own_member_floor": mf[arm]["max_abs_dev"],
                  "clears_own_floor": bool(imp[arm] > mf[arm]["max_abs_dev"]),
                  "beats_step3_baseline": bool(imp[arm] > imp["step3_pairwise"])}
            for arm in ("cond_full", "cond_single", "pool4_isotonic", "pool4_platt")}
    return out


def report(r: dict) -> None:
    v = r["variant"]
    a7, b7 = r["step7a"], r["step7b"]
    print("\n" + "=" * 100)
    print(f"STEP 7 — WORLD-CONDITIONED CALIBRATION   variant {v} / {r['state']} @ h^{r['depth']}")
    print("=" * 100)
    print(f"features used ({len(r['features_used'])}): {', '.join(r['features_used'])}")
    if r["features_dropped_no_between_world_variation"]:
        print(f"dropped (no between-world variation on this variant): "
              f"{', '.join(r['features_dropped_no_between_world_variation'])}")

    print(f"\n-- 7a  {a7['status']}")
    print(f"{'arm':<28}{'mean ECE':>11}{'improvement':>14}")
    print(f"{'raw (uncalibrated)':<28}{a7['mean_ece_raw']:>11.4f}{'—':>14}")
    print(f"{'per-world isotonic':<28}{a7['mean_ece_isotonic']:>11.4f}"
          f"{a7['improvement_isotonic']:>+14.4f}")
    print(f"{'pooled head, one-hot world':<28}{a7['mean_ece_onehot_head']:>11.4f}"
          f"{a7['improvement_onehot_head']:>+14.4f}")

    print(f"\n-- 7b  {b7['protocol']}")
    print(f"{'arm':<20}{'mean ECE':>11}{'Brier':>10}{'AUC':>9}{'impr vs raw':>14}"
          f"{'folds impr':>12}")
    for a in ARMS:
        m = b7["mean"][a]
        imp = b7["ece_improvement_vs_raw"][a]
        fi = b7["folds_improved_vs_raw"][a]
        print(f"{a:<20}{m['ece']:>11.4f}{m['brier']:>10.4f}{m['auc']:>9.4f}"
              f"{(f'{imp:+.4f}' if a != 'raw' else '—'):>14}"
              f"{(f'{fi}/5' if a != 'raw' else '—'):>12}")

    fa = r["feature_curve_association"]
    print(f"\n-- does any feature track a world's calibration curve?  "
          f"(n={fa['n_worlds']} worlds, exact permutation p over 120 orderings)")
    print(f"{'feature':<26}" + "".join(f"{t[:20]:>26}" for t in fa["targets"]))
    for k, d_ in fa["association"].items():
        print(f"{k:<26}" + "".join(f"{s['pearson_r']:>+18.3f} p={s['exact_p']:<5.3f}"
                                   for s in d_.values()))
    print(f"significant at p<0.05: {len(fa['significant_at_p05'])}/{fa['n_tests']} tests"
          + (f"  {fa['significant_at_p05']}" if fa["significant_at_p05"] else ""))

    if "gate_7b" not in r:
        return
    fl = r["floors_7b"]
    print(f"\nrefit floor ({fl['refit']['n_repeats']} identical refits, nothing changed): "
          f"max|dev| {fl['refit']['max_abs_dev']:.8f}")
    print(f"  {fl['refit']['note']}")
    print(f"\n{'arm':<20}{'impr vs raw':>13}{'vs step3':>11}{'own floor':>12}"
          f"{'clears own':>12}{'beats step3':>13}")
    for a, g in r["gate_7b"].items():
        print(f"{a:<20}{g['improvement_vs_raw']:>+13.4f}"
              f"{g['improvement_vs_step3_baseline']:>+11.4f}{g['own_member_floor']:>12.4f}"
              f"{('YES' if g['clears_own_floor'] else 'no'):>12}"
              f"{('YES' if g['beats_step3_baseline'] else 'no'):>13}")
    print(f"  (member floor over {fl['member']['n_subsets']} 3-of-5 init-seed subsets, "
          f"freshly measured on 7b's own metric)")
    print(f"\ndoes the CONDITIONING add anything over the same four worlds pooled without it?")
    print(f"{'contrast':<38}{'delta':>10}{'own floor':>12}{'adds?':>8}")
    for k, c in r["conditioning_contrast"].items():
        print(f"{k:<38}{c['delta_improvement']:>+10.4f}{c['floor_max_abs_dev']:>12.4f}"
              f"{('YES' if c['conditioning_adds_anything'] else 'no'):>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,E")
    ap.add_argument("--state", default="supply_stress")
    ap.add_argument("--seeds", default=",".join(str(d) for d in DSEEDS))
    ap.add_argument("--init-seeds", default=",".join(str(s) for s in INIT_SEEDS))
    ap.add_argument("--skip-floor", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    blob = {}
    for v in [s.strip() for s in a.variants.split(",") if s.strip()]:
        r = run(v, a.state, dseeds, iseeds, a.skip_floor)
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
