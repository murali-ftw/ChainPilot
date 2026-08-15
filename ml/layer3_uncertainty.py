#!/usr/bin/env python3
"""
Layer 3, STEPS 3-5 — calibration, per-entity confidence, and an insufficient-evidence threshold.

Runs only for states that reached here: Case 1 in STEP 2 (identifiable AND recoverable) *and*
PASS in STEP 1 (downstream-sufficient). Everything below operates on Model A's head, unmodified,
attached to the frozen backbone.

---

**The grid.** `ml/uncertainty_ensemble.py` builds a 5 x 5 grid over (dataset seed x model-init
seed) for the *backbone's* task heads. The Layer-3 analogue keeps the same two axes and the
same 5 x 5 shape, with one substitution that is forced rather than chosen: the second axis is
the **head init seed**, not the backbone seed. Only `m0` backbone checkpoints exist in
`out/ds_ckpt/`, and building `m1..m4` would mean *training SHARE*, which this work is not
permitted to do. The head init seed is also the axis Phase 1 already measures its reproduction
floor along, so the grid sits on the same axis as every prior number.

`ml/uncertainty_ensemble.py`'s identifiability caveat carries over verbatim and is not
softened: **per-entity variance is identifiable only WITHIN a world.** Dataset seeds 42 and 43
generate different suppliers, so there is no entity to pair across worlds. Per-entity spread
below is therefore always across the 5 head init seeds inside one world, and the dataset-seed
axis appears only as a population-level term and as the leave-one-world-out rotation.

**STEP 3 — calibration.** Isotonic PAVA and equal-count ECE are imported from
`ml/hypothesis_ranker.py`; nothing is reimplemented. The map is fitted on one world and
evaluated on a **different** world, rotating over all ordered seed pairs, because
dataset-seed variance dominates on this benchmark and a within-world fit would be optimistic
in exactly the place the uncertainty lives. Gate: mean ECE improvement must exceed the
**member-composition floor** -- how far ECE moves when the ensemble is composed of a different
subset of head init seeds, which is the operative floor because the PAVA fit itself is
deterministic.

**STEP 4 — per-entity confidence.** Confidence is `-var_over_init_seeds(p)`. The gate is
NOT that the variance is small; it is that the variance *orders* error correctly. Predictions
are binned by confidence and the gate asks for error to fall monotonically across bins. That is
a different claim from Step 3's -- a model can be perfectly calibrated in aggregate and still
have a confidence signal that means nothing per entity -- so it is measured separately.

**STEP 5 — insufficient-evidence threshold.** Selection uses the **validation** partition of
the four selection worlds only; the held-out world's test partition is untouched until the
threshold is frozen. The criterion is fixed before any evaluation and stated in
`select_threshold`'s docstring: *the smallest rejection rate whose accepted-set Brier
improvement clears the member floor* -- i.e. reject as little as possible while still buying a
real reliability gain, and report "no candidate qualifies" rather than reaching for the
best-looking one. A **matched-coverage random-rejection control** is reported alongside every
result: dropping any 20% of predictions changes Brier somewhat, and a rejection rule is only
worth anything to the extent it beats dropping the same number at random.

    python3 ml/layer3_uncertainty.py --variant A --state supply_stress \
        --out out/layer3/step345_A.json
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

from ml.ds_backbone import get_backbone, load_world                       # noqa: E402
from ml.hypothesis_ranker import ece, isotonic_apply, isotonic_fit, roc_auc  # noqa: E402
from ml.latent_state_head import (                                        # noqa: E402
    align, assert_backbone_frozen, binarise, depth_embeddings, latent_targets,
)
from ml.layer3_sufficiency import fit_probs                               # noqa: E402

N_BINS = 10
CONF_BINS = 5
# Rejection rates considered, lowest first. Ordering matters: `select_threshold` takes the
# FIRST qualifying candidate, so the criterion prefers coverage by construction.
REJECT_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y):
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def build_grid(variant: str, state: str, depth: int, dseeds: list[int], config: str,
               init_seeds: list[int], csv_root: str) -> dict:
    """`{dseed: {"p_va": [M,N], "y_va": [N], "p_te": [M,N], "y_te": [N]}}`.

    One frozen backbone per world (`mseed=0`, loaded from cache), one Model A head per init
    seed. `assert_backbone_frozen` runs inside `fit_probs` after construction and after
    training on every fit.
    """
    grid = {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        model, _ = get_backbone(csv_dir, variant, d, mseed=0, device="cpu")
        assert_backbone_frozen(model)
        tr, va, te, _ = load_world(csv_dir, "cpu")
        tgt = latent_targets(variant, d, config,
                             [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])[state]
        Xtr, ytr = align(*depth_embeddings(model, tr)[depth], tgt)
        Xva, yva = align(*depth_embeddings(model, va)[depth], tgt)
        Xte, yte = align(*depth_embeddings(model, te)[depth], tgt)
        # Median threshold from TRAIN only, applied unchanged to va and te -- Phase 1's rule.
        ytr_b, yte_b, thr = binarise(ytr, yte, tgt["kind"])
        yva_b = yva if tgt["kind"].startswith("binary") else (yva > thr).astype(float)

        P_va, P_te = [], []
        for s in init_seeds:
            got = fit_probs(Xtr, ytr_b, [Xva, Xte], s, model=model)
            P_va.append(got[0])
            P_te.append(got[1])
        grid[d] = {"p_va": np.stack(P_va), "y_va": yva_b,
                   "p_te": np.stack(P_te), "y_te": yte_b, "median_threshold": thr}
        print(f"  seed {d}: va {len(yva_b):,} rows ({int(yva_b.sum()):,} pos)  "
              f"te {len(yte_b):,} rows ({int(yte_b.sum()):,} pos)  "
              f"mean te AUC {statistics.fmean([roc_auc(p, yte_b.astype(bool)) for p in P_te]):.4f}",
              flush=True)
    return grid


def ens(grid, d, split, members):
    """Ensemble mean and per-entity across-member variance, within one world."""
    P = grid[d][f"p_{split}"][list(members)]
    return P.mean(axis=0), P.var(axis=0, ddof=1), grid[d][f"y_{split}"]


# ---------------------------------------------------------------------------
# STEP 3
# ---------------------------------------------------------------------------

def step3(grid, dseeds, init_seeds) -> dict:
    members = list(range(len(init_seeds)))
    folds = []
    for calib_d, test_d in itertools.permutations(dseeds, 2):
        mu_c, _, y_c = ens(grid, calib_d, "te", members)
        mu_t, var_t, y_t = ens(grid, test_d, "te", members)
        single_c = grid[calib_d]["p_te"][0]
        single_t = grid[test_d]["p_te"][0]
        cal_ens = isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t)
        cal_one = isotonic_apply(isotonic_fit(single_c, y_c.astype(float)), single_t)

        def pack(p, tag):
            e, curve = ece(p, y_t, N_BINS)
            return {f"ece_{tag}": e, f"brier_{tag}": brier(p, y_t), f"nll_{tag}": nll(p, y_t),
                    f"auc_{tag}": roc_auc(p, y_t.astype(bool)), f"reliability_{tag}": curve}

        f = {"calib_world": calib_d, "test_world": test_d, "n": int(len(y_t)),
             "positives": int(y_t.sum()), "mean_var_model": float(var_t.mean())}
        for p, tag in ((single_t, "single_raw"), (cal_one, "single_cal"),
                       (mu_t, "ens_raw"), (cal_ens, "ens_cal")):
            f.update(pack(p, tag))
        folds.append(f)

    # -- floors -----------------------------------------------------------
    # fit floor: refit PAVA on identical inputs. Deterministic, so this is expected to be
    # exactly 0 and is reported as a fact, not as a result.
    c0, t0 = dseeds[0], dseeds[1]
    fit_vals = []
    for _ in range(8):
        mu_c, _, y_c = ens(grid, c0, "te", members)
        mu_t, _, y_t = ens(grid, t0, "te", members)
        fit_vals.append(ece(isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t),
                            y_t, N_BINS)[0])
    fit_dev = [abs(a - b) for a, b in itertools.combinations(fit_vals, 2)]

    # member floor: the operative one -- vary WHICH init seeds compose the ensemble.
    mem_vals = []
    for combo in itertools.combinations(members, 3):
        mu_c, _, y_c = ens(grid, c0, "te", combo)
        mu_t, _, y_t = ens(grid, t0, "te", combo)
        mem_vals.append(ece(isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t),
                            y_t, N_BINS)[0])
    mem_dev = [abs(a - b) for a, b in itertools.combinations(mem_vals, 2)]

    member_floor = float(np.max(mem_dev)) if mem_dev else float("nan")
    raw = statistics.fmean([f["ece_ens_raw"] for f in folds])
    cal = statistics.fmean([f["ece_ens_cal"] for f in folds])
    improvement = raw - cal
    per_fold_improved = [f["ece_ens_raw"] - f["ece_ens_cal"] for f in folds]
    return {
        "n_folds": len(folds), "folds": folds,
        "mean": {tag: statistics.fmean([f[f"ece_{tag}"] for f in folds])
                 for tag in ("single_raw", "single_cal", "ens_raw", "ens_cal")},
        "mean_brier": {tag: statistics.fmean([f[f"brier_{tag}"] for f in folds])
                       for tag in ("single_raw", "single_cal", "ens_raw", "ens_cal")},
        "mean_auc": {tag: statistics.fmean([f[f"auc_{tag}"] for f in folds])
                     for tag in ("single_raw", "single_cal", "ens_raw", "ens_cal")},
        "fit_floor_max_abs_dev": float(np.max(fit_dev)) if fit_dev else 0.0,
        "member_floor_max_abs_dev": member_floor,
        "member_floor_n_subsets": len(mem_vals),
        "ece_improvement": improvement,
        "improvement_per_fold": per_fold_improved,
        "folds_improved": int(sum(1 for v in per_fold_improved if v > 0)),
        "clears_member_floor": bool(improvement > member_floor),
        "gate_pass": bool(improvement > member_floor
                          and all(v > 0 for v in per_fold_improved)),
    }


def step3_diagnostics(grid, dseeds, init_seeds, s3: dict, n_sim: int = 200) -> dict:
    """Why did the calibration gate fail? Two measurements, so the classification is derived
    rather than asserted.

    *Binomial ECE floor.* Equal-count ECE is a mean of |empirical - confidence| over bins, and
    each bin's empirical rate is a binomial estimate from n/10 samples. So even a PERFECTLY
    calibrated predictor posts a non-zero ECE at finite n. This resamples labels from the
    model's own predicted probabilities -- making the predictor calibrated by construction --
    and measures the ECE that results. If the observed raw ECE sits at that floor, there was
    nothing to calibrate and the gate was unreachable; if it sits well above it, the
    miscalibration is real and isotonic genuinely failed to remove it.

    *Within-world (leaky) calibration.* Fitting the isotonic map on the SAME world it is
    evaluated on is optimistically biased and is never gated on -- it is reported purely as an
    upper bound. It separates "isotonic cannot fix this score" from "isotonic can fix it, but
    the map does not transfer between worlds", which are different diagnoses.
    """
    members = list(range(len(init_seeds)))
    rng = np.random.default_rng(20260814)
    floors = []
    for d in dseeds:
        mu, _, y = ens(grid, d, "te", members)
        for _ in range(n_sim // len(dseeds)):
            y_sim = (rng.random(len(mu)) < mu).astype(float)
            floors.append(ece(mu, y_sim, N_BINS)[0])

    leaky = []
    for d in dseeds:
        mu, _, y = ens(grid, d, "te", members)
        cal = isotonic_apply(isotonic_fit(mu, y.astype(float)), mu)
        leaky.append({"world": d, "ece_raw": ece(mu, y, N_BINS)[0],
                      "ece_within_world_cal": ece(cal, y, N_BINS)[0]})
    raw = s3["mean"]["ens_raw"]
    floor = float(np.mean(floors))
    return {
        "binomial_ece_floor": {
            "n_simulations": len(floors), "mean": floor,
            "p05": float(np.quantile(floors, 0.05)),
            "p95": float(np.quantile(floors, 0.95)),
            "observed_raw_ece": raw,
            "observed_over_floor": raw / floor if floor else None,
            "raw_ece_is_at_the_floor": bool(raw <= float(np.quantile(floors, 0.95))),
        },
        "within_world_leaky_calibration": {
            "per_world": leaky,
            "mean_ece_raw": statistics.fmean([x["ece_raw"] for x in leaky]),
            "mean_ece_cal": statistics.fmean([x["ece_within_world_cal"] for x in leaky]),
            "mean_improvement": statistics.fmean(
                [x["ece_raw"] - x["ece_within_world_cal"] for x in leaky]),
            "note": "optimistically biased on purpose; an upper bound, never a gate",
        },
    }


# ---------------------------------------------------------------------------
# STEP 4
# ---------------------------------------------------------------------------

def confidence_bins(p, var, y, n_bins=CONF_BINS) -> list:
    """Equal-count bins ordered from LEAST to MOST confident (highest to lowest variance)."""
    order = np.argsort(-var, kind="stable")
    out = []
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        out.append({"n": int(len(b)), "mean_var": float(var[b].mean()),
                    "brier": brier(p[b], y[b]),
                    "abs_error": float(np.abs(p[b] - y[b]).mean()),
                    "base_rate": float(y[b].mean()),
                    "auc": roc_auc(p[b], y[b].astype(bool))})
    return out


def step4(grid, dseeds, init_seeds) -> dict:
    """Per-entity confidence, evaluated on each world's own test split.

    Calibration is applied first, using a DIFFERENT world's map (the world before it in the
    rotation), so the errors being binned are the errors of the Step-3 pipeline rather than of
    a raw uncalibrated score.
    """
    members = list(range(len(init_seeds)))
    per_world, pooled = {}, []
    for i, d in enumerate(dseeds):
        calib_d = dseeds[(i + 1) % len(dseeds)]
        mu_c, _, y_c = ens(grid, calib_d, "te", members)
        mu_t, var_t, y_t = ens(grid, d, "te", members)
        cal = isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t)
        bins = confidence_bins(cal, var_t, y_t)
        briers = [b["brier"] for b in bins]
        per_world[d] = {
            "calib_world": calib_d, "n": int(len(y_t)),
            "mean_var": float(var_t.mean()), "bins": bins,
            "monotone_brier": all(briers[j] >= briers[j + 1] for j in range(len(briers) - 1)),
            "brier_lowconf_minus_highconf": briers[0] - briers[-1],
        }
        pooled.append((cal, var_t, y_t))

    P = np.concatenate([x[0] for x in pooled])
    V = np.concatenate([x[1] for x in pooled])
    Y = np.concatenate([x[2] for x in pooled])
    pbins = confidence_bins(P, V, Y)
    pb = [b["brier"] for b in pbins]
    # Floor for the monotonicity claim: how far the binned Brier curve moves when the ensemble
    # is composed of a different subset of init seeds. A "monotone" curve whose steps are
    # smaller than this is not evidence of anything.
    step_floor = []
    for combo in itertools.combinations(members, 3):
        sub = []
        for i, d in enumerate(dseeds):
            calib_d = dseeds[(i + 1) % len(dseeds)]
            mu_c, _, y_c = ens(grid, calib_d, "te", combo)
            mu_t, var_t, y_t = ens(grid, d, "te", combo)
            sub.append((isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t),
                        var_t, y_t))
        sbins = confidence_bins(np.concatenate([x[0] for x in sub]),
                                np.concatenate([x[1] for x in sub]),
                                np.concatenate([x[2] for x in sub]))
        step_floor.append([b["brier"] for b in sbins])
    arr = np.array(step_floor)
    floor = float(np.max(arr.max(axis=0) - arr.min(axis=0)))
    spread = pb[0] - pb[-1]
    return {
        "per_world": per_world, "pooled_bins": pbins,
        "pooled_monotone_brier": all(pb[j] >= pb[j + 1] for j in range(len(pb) - 1)),
        "pooled_brier_lowconf_minus_highconf": spread,
        "worlds_monotone": int(sum(1 for v in per_world.values() if v["monotone_brier"])),
        "n_worlds": len(dseeds),
        "member_floor_on_bin_brier": floor,
        "spread_exceeds_floor": bool(spread > floor),
        "gate_pass": bool(all(pb[j] >= pb[j + 1] for j in range(len(pb) - 1))
                          and spread > floor),
    }


# ---------------------------------------------------------------------------
# STEP 5
# ---------------------------------------------------------------------------

def select_threshold(grid, sel_dseeds, members, member_floor: float) -> dict:
    """Choose the variance threshold on the SELECTION worlds' VALIDATION partitions only.

    Criterion, fixed before any evaluation and not revised afterwards:

        take the SMALLEST rejection rate in `REJECT_GRID` whose accepted-set Brier improves
        on the all-predictions Brier by more than the Step-3 member-composition floor.

    Two properties of that rule are deliberate. It prefers coverage -- the grid is walked from
    5% upward and the first qualifying candidate wins, so the rule never rejects more than it
    has to. And it can return NOTHING: if no candidate clears the floor, `chosen` is None and
    that is reported as the answer rather than quietly falling back to whichever cut happened
    to score best.
    """
    ps, vs, ys = [], [], []
    for d in sel_dseeds:
        mu, var, y = ens(grid, d, "va", members)
        ps.append(mu)
        vs.append(var)
        ys.append(y)
    P, V, Y = np.concatenate(ps), np.concatenate(vs), np.concatenate(ys)
    base = brier(P, Y)
    cands = []
    for r in REJECT_GRID:
        tau = float(np.quantile(V, 1.0 - r))
        keep = V <= tau
        if keep.sum() < 50 or len(np.unique(Y[keep])) < 2:
            continue
        b = brier(P[keep], Y[keep])
        cands.append({"reject_rate": r, "tau": tau, "brier_accepted": b,
                      "brier_all": base, "improvement": base - b,
                      "qualifies": bool((base - b) > member_floor),
                      "n_accepted": int(keep.sum()), "n_total": int(len(Y))})
    chosen = next((c for c in cands if c["qualifies"]), None)
    return {"selection_worlds": sel_dseeds, "split": "validation",
            "brier_all_selection": base, "member_floor": member_floor,
            "candidates": cands, "chosen": chosen}


def apply_threshold(grid, test_d, calib_d, members, tau: float, rng_seed: int,
                    n_random: int = 200) -> dict:
    """Freeze `tau`, apply it once to the held-out world's TEST partition, report everything.

    The random-rejection control is not optional garnish. Dropping any fixed fraction of
    predictions moves Brier on its own; a rejection rule is worth something only to the extent
    it beats dropping the same number at random, so the control is run at MATCHED coverage and
    reported next to every accepted-set number.
    """
    mu_c, _, y_c = ens(grid, calib_d, "te", members)
    mu_t, var_t, y_t = ens(grid, test_d, "te", members)
    cal = isotonic_apply(isotonic_fit(mu_c, y_c.astype(float)), mu_t)
    keep = var_t <= tau
    n, k = int(len(y_t)), int(keep.sum())

    def score(p, y):
        return {"n": int(len(y)), "positives": int(y.sum()),
                "auc": roc_auc(p, y.astype(bool)), "brier": brier(p, y),
                "ece": ece(p, y, N_BINS)[0], "nll": nll(p, y)}

    before = score(cal, y_t)
    after = (score(cal[keep], y_t[keep])
             if k >= 20 and len(np.unique(y_t[keep])) > 1 else None)

    rng = np.random.default_rng(rng_seed)
    rb, re_, ra = [], [], []
    for _ in range(n_random):
        idx = rng.choice(n, size=k, replace=False)
        if len(np.unique(y_t[idx])) < 2:
            continue
        rb.append(brier(cal[idx], y_t[idx]))
        re_.append(ece(cal[idx], y_t[idx], N_BINS)[0])
        a = roc_auc(cal[idx], y_t[idx].astype(bool))
        if a is not None:
            ra.append(a)
    control = {"n_repeats": len(rb), "brier": float(np.mean(rb)),
               "brier_p95": float(np.quantile(rb, 0.95)),
               "ece": float(np.mean(re_)), "auc": float(np.mean(ra)) if ra else None}
    return {
        "test_world": test_d, "calib_world": calib_d, "tau": tau,
        "n_total": n, "n_accepted": k, "n_rejected": n - k,
        "reject_rate": (n - k) / n if n else float("nan"),
        "before_rejection": before, "after_rejection": after,
        "random_rejection_control": control,
        "beats_random_on_brier": (
            None if after is None else bool(after["brier"] < control["brier"])),
        "beats_random_on_ece": (
            None if after is None else bool(after["ece"] < control["ece"])),
    }


def step5(grid, dseeds, init_seeds, member_floor: float) -> dict:
    members = list(range(len(init_seeds)))
    rotations = []
    for i, held in enumerate(dseeds):
        sel = [d for d in dseeds if d != held]
        pick = select_threshold(grid, sel, members, member_floor)
        if pick["chosen"] is None:
            rotations.append({"held_out_world": held, "selection": pick, "applied": None})
            continue
        applied = apply_threshold(grid, held, dseeds[(i + 1) % len(dseeds)], members,
                                  pick["chosen"]["tau"], rng_seed=held)
        rotations.append({"held_out_world": held, "selection": pick, "applied": applied})

    ok = [r for r in rotations if r["applied"]]
    rates = [r["selection"]["chosen"]["reject_rate"] for r in ok]
    taus = [r["selection"]["chosen"]["tau"] for r in ok]
    out = {"n_rotations": len(rotations), "n_with_a_threshold": len(ok),
           "rotations": rotations,
           "chosen_reject_rates": rates, "chosen_taus": taus,
           "reject_rate_stability": {
               "distinct_values": sorted(set(rates)),
               "same_in_every_rotation": len(set(rates)) == 1} if rates else None,
           "tau_stability": {"min": min(taus), "max": max(taus),
                             "spread_over_mean": (max(taus) - min(taus)) / statistics.fmean(taus)
                             } if taus else None}
    if ok:
        def agg(path):
            vals = []
            for r in ok:
                node = r["applied"]
                for k in path:
                    node = node[k] if node else None
                if node is not None:
                    vals.append(node)
            return statistics.fmean(vals) if vals else None
        out["pooled"] = {
            "mean_reject_rate": agg(["reject_rate"]),
            "n_total": sum(r["applied"]["n_total"] for r in ok),
            "n_accepted": sum(r["applied"]["n_accepted"] for r in ok),
            "n_rejected": sum(r["applied"]["n_rejected"] for r in ok),
            "before": {k: agg(["before_rejection", k]) for k in ("auc", "brier", "ece", "nll")},
            "after": {k: agg(["after_rejection", k]) for k in ("auc", "brier", "ece", "nll")},
            "random_control": {k: agg(["random_rejection_control", k])
                               for k in ("auc", "brier", "ece")},
            "rotations_beating_random_on_brier": sum(
                1 for r in ok if r["applied"]["beats_random_on_brier"]),
            "rotations_beating_random_on_ece": sum(
                1 for r in ok if r["applied"]["beats_random_on_ece"]),
        }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--state", default="supply_stress")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--phase1", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    p1 = a.phase1 or os.path.join(REPO, "out", "phase1", f"heads_{a.variant}.json")
    depth = int(json.load(open(p1))["summary"][a.state]["best_depth"])

    print(f"\nbuilding grid: variant {a.variant}, {a.state} @ h^{depth}, "
          f"{len(dseeds)} worlds x {len(iseeds)} head init seeds", flush=True)
    grid = build_grid(a.variant, a.state, depth, dseeds, a.config, iseeds, a.csv_root)

    s3 = step3(grid, dseeds, iseeds)
    s3["diagnostics"] = step3_diagnostics(grid, dseeds, iseeds, s3)
    s4 = step4(grid, dseeds, iseeds)
    s5 = step5(grid, dseeds, iseeds, s3["member_floor_max_abs_dev"])

    print("\n" + "=" * 96)
    print(f"STEP 3 — CALIBRATION   variant {a.variant} / {a.state} @ h^{depth}   "
          f"{s3['n_folds']} leave-one-world-out folds")
    print("=" * 96)
    print(f"{'arm':<14}{'ECE':>10}{'Brier':>10}{'AUC':>10}")
    for tag in ("single_raw", "single_cal", "ens_raw", "ens_cal"):
        print(f"{tag:<14}{s3['mean'][tag]:>10.4f}{s3['mean_brier'][tag]:>10.4f}"
              f"{s3['mean_auc'][tag]:>10.4f}")
    print(f"\nECE improvement (ens_raw -> ens_cal): {s3['ece_improvement']:+.4f}"
          f"   folds improved {s3['folds_improved']}/{s3['n_folds']}")
    print(f"fit floor (deterministic)   max|dev| {s3['fit_floor_max_abs_dev']:.6f}")
    print(f"member floor ({s3['member_floor_n_subsets']} subsets) max|dev| "
          f"{s3['member_floor_max_abs_dev']:.4f}   <- the gate")
    print(f"GATE: {'PASS' if s3['gate_pass'] else 'STOP (classify G)'}")
    dg = s3["diagnostics"]
    bf, wl = dg["binomial_ece_floor"], dg["within_world_leaky_calibration"]
    print(f"\n  diagnosis — binomial ECE floor at this n/bin count: {bf['mean']:.4f} "
          f"[p05 {bf['p05']:.4f}, p95 {bf['p95']:.4f}];  observed raw ECE {bf['observed_raw_ece']:.4f} "
          f"= {bf['observed_over_floor']:.2f}x floor;  raw ECE is AT the floor: "
          f"{bf['raw_ece_is_at_the_floor']}")
    print(f"  diagnosis — within-world (leaky, upper bound) ECE {wl['mean_ece_raw']:.4f} -> "
          f"{wl['mean_ece_cal']:.4f}, improvement {wl['mean_improvement']:+.4f}")

    print("\n" + "=" * 96)
    print(f"STEP 4 — PER-ENTITY CONFIDENCE   {CONF_BINS} equal-count bins, "
          f"least confident first")
    print("=" * 96)
    print(f"{'bin':<6}{'n':>9}{'mean var':>12}{'Brier':>10}{'|err|':>10}{'AUC':>9}"
          f"{'base rate':>11}")
    for i, b in enumerate(s4["pooled_bins"]):
        print(f"{i + 1:<6}{b['n']:>9,}{b['mean_var']:>12.6f}{b['brier']:>10.4f}"
              f"{b['abs_error']:>10.4f}"
              f"{(b['auc'] if b['auc'] is not None else float('nan')):>9.4f}"
              f"{b['base_rate']:>11.4f}")
    print(f"\nmonotone (Brier falls as confidence rises): {s4['pooled_monotone_brier']}   "
          f"per-world {s4['worlds_monotone']}/{s4['n_worlds']}")
    print(f"least- minus most-confident Brier: {s4['pooled_brier_lowconf_minus_highconf']:+.4f}"
          f"   member floor on bin Brier {s4['member_floor_on_bin_brier']:.4f}")
    print(f"GATE: {'PASS' if s4['gate_pass'] else 'STOP (classify G)'}")

    print("\n" + "=" * 96)
    print("STEP 5 — INSUFFICIENT-EVIDENCE THRESHOLD   (selected on validation partitions of "
          "4 worlds, frozen, applied to the 5th's test partition)")
    print("=" * 96)
    if s5["n_with_a_threshold"] == 0:
        print("no candidate rejection rate cleared the member floor in ANY rotation — "
              "no threshold is frozen and none is applied.")
    else:
        print(f"{'held out':>9}{'chosen r':>10}{'tau':>12}{'reject':>9}{'AUC bef':>9}"
              f"{'AUC aft':>9}{'Brier bef':>11}{'Brier aft':>11}{'Brier rnd':>11}"
              f"{'ECE bef':>9}{'ECE aft':>9}{'ECE rnd':>9}")
        for r in s5["rotations"]:
            ap_ = r["applied"]
            if not ap_:
                print(f"{r['held_out_world']:>9}{'— no candidate qualified —':>50}")
                continue
            b, af, c = ap_["before_rejection"], ap_["after_rejection"], \
                ap_["random_rejection_control"]
            print(f"{r['held_out_world']:>9}{r['selection']['chosen']['reject_rate']:>10.2f}"
                  f"{ap_['tau']:>12.6f}{ap_['reject_rate']:>9.3f}"
                  f"{b['auc']:>9.4f}{af['auc']:>9.4f}"
                  f"{b['brier']:>11.4f}{af['brier']:>11.4f}{c['brier']:>11.4f}"
                  f"{b['ece']:>9.4f}{af['ece']:>9.4f}{c['ece']:>9.4f}")
        p = s5["pooled"]
        print(f"\npooled: {p['n_total']:,} test predictions, {p['n_accepted']:,} usable, "
              f"{p['n_rejected']:,} insufficient-evidence "
              f"({p['n_rejected'] / p['n_total']:.1%} rejected)")
        print(f"  before rejection  AUC {p['before']['auc']:.4f}  "
              f"Brier {p['before']['brier']:.4f}  ECE {p['before']['ece']:.4f}")
        print(f"  on accepted       AUC {p['after']['auc']:.4f}  "
              f"Brier {p['after']['brier']:.4f}  ECE {p['after']['ece']:.4f}")
        print(f"  random control    AUC {p['random_control']['auc']:.4f}  "
              f"Brier {p['random_control']['brier']:.4f}  "
              f"ECE {p['random_control']['ece']:.4f}   (matched coverage)")
        print(f"  rotations beating the random control: Brier "
              f"{p['rotations_beating_random_on_brier']}/{s5['n_with_a_threshold']}, "
              f"ECE {p['rotations_beating_random_on_ece']}/{s5['n_with_a_threshold']}")
        print(f"  chosen rejection rate across rotations: {s5['chosen_reject_rates']} "
              f"(identical in every rotation: "
              f"{s5['reject_rate_stability']['same_in_every_rotation']})")
        print(f"  tau spread / mean: {s5['tau_stability']['spread_over_mean']:.3f}")

    blob = {"variant": a.variant, "state": a.state, "depth": depth, "config": a.config,
            "dataset_seeds": dseeds, "init_seeds": iseeds,
            "step3": s3, "step4": s4, "step5": s5}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(blob, f, indent=1, default=float)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
