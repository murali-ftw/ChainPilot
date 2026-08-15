#!/usr/bin/env python3
"""
STEP 4 of the world-conditioned calibration follow-up — *how* do the per-world calibration
curves differ, not just *that* transfer fails.

`reports/layer3_uncertainty_aware.md` STEP 3 established the aggregate fact: a map fitted on one
world makes ECE **worse** on a different one (-0.0054 / -0.0029 mean over 20 leave-one-world-out
folds; 7/20 and 8/20 folds improved) while a within-world fit removes essentially all of the
miscalibration (+0.0202 / +0.0224). ECE is a scalar, so it says transfer fails and nothing about
the shape of the failure. This module measures the shape, because the shape is what decides
whether the fix in STEP 7 is even the right kind of object.

**The decomposition.** Fit a per-world isotonic curve `g_d` for each of the 5 worlds. On a shared
grid of predicted probabilities, the difference between two worlds' curves is split into three
orthogonal-by-construction pieces:

    g_i(x) - g_j(x)  =  shift  +  slope * (x - xbar)  +  residual(x)

* **shift** -- the mean vertical offset. Two curves differing only here are parallel: a single
  global additive offset would reconcile them.
* **slope (scale)** -- the least-squares slope of the difference in `x`. Two curves differing
  only here differ in how steeply they map confidence to frequency: a single global temperature
  or Platt scaling would reconcile them.
* **residual** -- everything left over. This is the part that *no* affine recalibration can
  remove. Large residual means the curves are not parallel and not stretched versions of each
  other; they bend differently and can cross, so only a world-specific map can reconcile them.

Component sizes are reported as RMS over the grid, which puts all three on the same units
(probability) and makes "which dominates" a comparison rather than a judgement call.

**The grid is the *overlap* of the five worlds' predicted-probability ranges**, not [0, 1].
`isotonic_apply` clamps outside its fitted support (`left=ys[0], right=ys[-1]`), so evaluating
outside the overlap would compare a real curve against a flat clamp and manufacture a residual
that is an artefact of extrapolation rather than a difference in calibration.

**Three floors, because a difference is not a finding until it is known how far the quantity
moves with nothing changed.** They answer different questions and only one of them is the gate:

* **same-curve null (the operative floor).** The null hypothesis this step must reject is *"all
  five worlds share one calibration curve, and the observed curve differences are label noise at
  these sample sizes."* So it is simulated directly: fit one pooled isotonic curve on all five
  worlds, treat it as the common truth, resample each world's labels from it at that world's own
  `n` and its own predicted-probability distribution, refit a curve per world, and decompose the
  pairs. This is the same construction STEP 3's binomial ECE floor used
  (`layer3_uncertainty.py::step3_diagnostics` resamples labels from the model's own
  probabilities), applied to curve shape instead of to ECE. It is matched on n, on the
  prediction distribution and on the number of pairs, which the other two floors are not.
* **member floor** -- refit both curves on the same world from different 3-member subsets of the
  5 head init seeds. The floor construction STEP 3 gated on (`ml/uncertainty_calibrate.py`'s
  member floor), applied to the curve components. Reported for continuity; it holds the sample
  fixed and so cannot answer the sampling question above.
* **split-half floor** -- refit both curves on disjoint random halves of one world's test rows.
  Measured at n/2 while the cross-world pairs are measured at n, and isotonic curve noise grows
  as the sample shrinks, so this **overstates** the sampling contribution. Reported as context,
  explicitly *not* used as the gate -- inheriting an n/2 floor for an n-sized comparison would
  be the mirror image of the leakage STEP 3 refused.

    python3 ml/calibration_curve_compare.py --variants A,E --out out/wcc/step4_curves.json
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
from ml.hypothesis_ranker import isotonic_apply, isotonic_fit           # noqa: E402

N_GRID = 101
SPLIT_HALF_REPEATS = 10
NULL_REPEATS = 200


# ---------------------------------------------------------------------------
# the shared evaluation grid
# ---------------------------------------------------------------------------

def shared_grid(preds: list[np.ndarray], n: int = N_GRID) -> tuple[np.ndarray, dict]:
    """Equally spaced points across the OVERLAP of every predictor's support.

    Returns the grid and a coverage record, so the report can state what fraction of real
    predictions the comparison actually speaks for rather than implying it covers all of them.
    """
    lo = max(float(p.min()) for p in preds)
    hi = min(float(p.max()) for p in preds)
    if not hi > lo:                                     # pragma: no cover - degenerate worlds
        raise ValueError(f"empty overlap: lo={lo} hi={hi}")
    x = np.linspace(lo, hi, n)
    inside = [float(np.mean((p >= lo) & (p <= hi))) for p in preds]
    return x, {"lo": lo, "hi": hi, "n_points": n,
               "fraction_of_predictions_inside_per_world": inside,
               "fraction_of_predictions_inside_mean": statistics.fmean(inside)}


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------

def decompose(gi: np.ndarray, gj: np.ndarray, x: np.ndarray) -> dict:
    """Split `gi - gj` on grid `x` into shift + slope*(x - xbar) + residual.

    The three pieces are orthogonal by construction: `shift` is the mean of the difference and
    `slope` is the OLS coefficient on the centred grid, so the residual has zero mean and zero
    correlation with `x`. That is what lets the three RMS values be compared directly -- they
    partition the total, `rms_total**2 == shift**2 + rms_slope**2 + rms_residual**2`.
    """
    d = gi - gj
    xc = x - x.mean()
    shift = float(d.mean())
    denom = float((xc ** 2).sum())
    slope = float((d * xc).sum() / denom) if denom > 0 else 0.0
    fitted = shift + slope * xc
    resid = d - fitted
    sgn = np.sign(d)
    nz = sgn[sgn != 0]
    crossings = int((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0
    return {
        "shift": shift,
        "slope": slope,
        "rms_shift": abs(shift),
        "rms_slope": float(np.sqrt(np.mean((slope * xc) ** 2))),
        "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
        "rms_total": float(np.sqrt(np.mean(d ** 2))),
        "max_abs_diff": float(np.abs(d).max()),
        "sign_crossings": crossings,
    }


def _fractions(rec: dict) -> dict:
    """Each component's share of the total difference *energy* (squares sum to the total)."""
    tot = rec["rms_total"] ** 2
    if tot <= 0:
        return {"frac_shift": float("nan"), "frac_slope": float("nan"),
                "frac_residual": float("nan")}
    return {"frac_shift": rec["rms_shift"] ** 2 / tot,
            "frac_slope": rec["rms_slope"] ** 2 / tot,
            "frac_residual": rec["rms_residual"] ** 2 / tot}


def dominant(rec: dict) -> str:
    return max(("shift", "slope", "residual"),
               key=lambda k: rec[f"rms_{k}"])


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def per_world_curves(grid: dict, dseeds: list[int], members) -> dict:
    """One isotonic curve per world, fitted on that world's own test split."""
    out = {}
    for d in dseeds:
        mu, _, y = ens(grid, d, "te", members)
        out[d] = {"curve": isotonic_fit(mu, y.astype(float)), "p": mu, "y": y}
    return out


def cross_world_pairs(grid: dict, dseeds: list[int], members) -> dict:
    cur = per_world_curves(grid, dseeds, members)
    x, cov = shared_grid([cur[d]["p"] for d in dseeds])
    g = {d: isotonic_apply(cur[d]["curve"], x) for d in dseeds}
    pairs = []
    for i, j in itertools.combinations(dseeds, 2):
        rec = decompose(g[i], g[j], x)
        rec.update(_fractions(rec))
        rec.update({"world_i": i, "world_j": j, "dominant": dominant(rec)})
        pairs.append(rec)
    return {"grid": cov, "pairs": pairs, "n_pairs": len(pairs),
            "curves_on_grid": {int(d): g[d].tolist() for d in dseeds},
            "grid_x": x.tolist()}


def member_floor(grid: dict, dseeds: list[int], members) -> dict:
    """Same world, two different 3-member ensembles -- the STEP 3 member-floor construction."""
    combos = list(itertools.combinations(members, 3))
    recs = []
    for d in dseeds:
        sub = {}
        for c in combos:
            mu, _, y = ens(grid, d, "te", c)
            sub[c] = (isotonic_fit(mu, y.astype(float)), mu)
        x, _ = shared_grid([v[1] for v in sub.values()])
        gg = {c: isotonic_apply(sub[c][0], x) for c in combos}
        for a, b in itertools.combinations(combos, 2):
            rec = decompose(gg[a], gg[b], x)
            rec["world"] = d
            recs.append(rec)
    return _summarise_floor(recs, "member (3-of-5 init-seed subsets, same world)")


def split_half_floor(grid: dict, dseeds: list[int], members, repeats: int = SPLIT_HALF_REPEATS,
                     seed: int = 20260815) -> dict:
    """Same world, disjoint random halves of its test rows.

    Measured at n/2 per curve while the cross-world pairs are measured at n, so this OVERSTATES
    the sampling contribution. Reported as such.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for d in dseeds:
        mu, _, y = ens(grid, d, "te", members)
        n = len(mu)
        for _ in range(repeats):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            ca = isotonic_fit(mu[a], y[a].astype(float))
            cb = isotonic_fit(mu[b], y[b].astype(float))
            x, _ = shared_grid([mu[a], mu[b]])
            rec = decompose(isotonic_apply(ca, x), isotonic_apply(cb, x), x)
            rec["world"] = d
            recs.append(rec)
    return _summarise_floor(recs, "split-half (disjoint halves of one world's test rows, n/2)")


def same_curve_null(grid: dict, dseeds: list[int], members, repeats: int = NULL_REPEATS,
                    seed: int = 20260815) -> dict:
    """The gate's null: *all five worlds share one calibration curve.*

    One isotonic curve is fitted on the pooled predictions of all five worlds and taken as the
    common truth. Each world's labels are then resampled from that common curve, evaluated at
    that world's own predictions and its own row count, and a fresh per-world curve is fitted to
    the resampled labels. Decomposing those pairs gives the distribution of shift / slope /
    residual that arises when the worlds genuinely do *not* differ -- matched on n, on the
    predicted-probability distribution, and on the 10-pair aggregation.

    Anything the observed cross-world pairs show above this is a real difference in curve shape.
    """
    rng = np.random.default_rng(seed)
    mus, ys = {}, {}
    for d in dseeds:
        mu, _, y = ens(grid, d, "te", members)
        mus[d], ys[d] = mu, y
    pooled_curve = isotonic_fit(np.concatenate([mus[d] for d in dseeds]),
                                np.concatenate([ys[d] for d in dseeds]).astype(float))
    truth = {d: isotonic_apply(pooled_curve, mus[d]) for d in dseeds}
    x, _ = shared_grid([mus[d] for d in dseeds])

    recs, per_rep_mean = [], []
    for _ in range(repeats):
        g = {}
        for d in dseeds:
            y_sim = (rng.random(len(mus[d])) < truth[d]).astype(float)
            g[d] = isotonic_apply(isotonic_fit(mus[d], y_sim), x)
        this = []
        for i, j in itertools.combinations(dseeds, 2):
            rec = decompose(g[i], g[j], x)
            recs.append(rec)
            this.append(rec)
        per_rep_mean.append({k: statistics.fmean([r[k] for r in this])
                             for k in ("rms_shift", "rms_slope", "rms_residual", "rms_total")})
    out = _summarise_floor(recs, f"same-curve null ({repeats} label resamples from one pooled "
                                 f"isotonic curve, matched n)")
    # The observed statistic is a MEAN over 10 pairs, so the null it is compared against must
    # also be the distribution of that mean -- not of a single pair.
    out["mean_over_pairs_null"] = {
        k: {"mean": statistics.fmean([r[k] for r in per_rep_mean]),
            "p95": float(np.quantile([r[k] for r in per_rep_mean], 0.95)),
            "max": max(r[k] for r in per_rep_mean)}
        for k in ("rms_shift", "rms_slope", "rms_residual", "rms_total")}
    out["n_repeats"] = repeats
    return out


def _summarise_floor(recs: list[dict], label: str) -> dict:
    keys = ("rms_shift", "rms_slope", "rms_residual", "rms_total", "max_abs_diff")
    return {
        "label": label, "n_comparisons": len(recs),
        "mean": {k: statistics.fmean([r[k] for r in recs]) for k in keys},
        "p95": {k: float(np.quantile([r[k] for r in recs], 0.95)) for k in keys},
        "max": {k: max(r[k] for r in recs) for k in keys},
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(variant: str, state: str, dseeds: list[int], init_seeds: list[int]) -> dict:
    grid, depth = load_grid(variant, state, dseeds, init_seeds)
    members = list(range(len(init_seeds)))
    cross = cross_world_pairs(grid, dseeds, members)
    null = same_curve_null(grid, dseeds, members)
    mf = member_floor(grid, dseeds, members)
    sf = split_half_floor(grid, dseeds, members)

    keys = ("rms_shift", "rms_slope", "rms_residual", "rms_total")
    mean_cross = {k: statistics.fmean([p[k] for p in cross["pairs"]]) for k in keys}
    mean_frac = {k: statistics.fmean([p[k] for p in cross["pairs"]])
                 for k in ("frac_shift", "frac_slope", "frac_residual")}
    # The gate is the same-curve null on the SAME statistic that is observed: the mean over the
    # 10 pairs. The member and split-half floors are reported but not gated on -- see module doc.
    op_floor = {k: null["mean_over_pairs_null"][k]["p95"] for k in keys}
    counts = {c: sum(1 for p in cross["pairs"] if p["dominant"] == c)
              for c in ("shift", "slope", "residual")}
    worst = max(cross["pairs"], key=lambda p: p["rms_residual"])
    return {
        "variant": variant, "state": state, "depth": depth,
        "dataset_seeds": dseeds, "init_seeds": init_seeds,
        "cross_world": cross,
        "mean_components": mean_cross,
        "mean_energy_fractions": mean_frac,
        "dominant_counts": counts,
        "dominant_on_average": max(("shift", "slope", "residual"),
                                   key=lambda k: mean_cross[f"rms_{k}"]),
        "floors": {"same_curve_null": null, "member": mf, "split_half": sf,
                   "operative_p95": op_floor,
                   "operative_source": "same_curve_null.mean_over_pairs_null.p95"},
        "components_clearing_floor": {
            k: bool(mean_cross[k] > op_floor[k]) for k in keys},
        "ratio_over_null": {k: mean_cross[k] / null["mean_over_pairs_null"][k]["mean"]
                            for k in keys},
        "pairs_with_residual_over_floor": sum(
            1 for p in cross["pairs"] if p["rms_residual"] > null["p95"]["rms_residual"]),
        "largest_residual_pair": {"world_i": worst["world_i"], "world_j": worst["world_j"],
                                  "rms_residual": worst["rms_residual"],
                                  "frac_residual": worst["frac_residual"],
                                  "sign_crossings": worst["sign_crossings"]},
        "pairs_that_cross": sum(1 for p in cross["pairs"] if p["sign_crossings"] > 0),
    }


def report(r: dict) -> None:
    print("\n" + "=" * 96)
    print(f"STEP 4 — CALIBRATION-CURVE DIFFERENCES   variant {r['variant']} / {r['state']} "
          f"@ h^{r['depth']}   {r['cross_world']['n_pairs']} world pairs")
    print("=" * 96)
    g = r["cross_world"]["grid"]
    print(f"shared grid: [{g['lo']:.4f}, {g['hi']:.4f}] over {g['n_points']} points; "
          f"covers {g['fraction_of_predictions_inside_mean']:.1%} of predictions on average")
    print(f"\n{'pair':<12}{'shift':>10}{'slope*x':>10}{'residual':>11}{'total':>10}"
          f"{'%resid':>9}{'cross':>7}  dominant")
    for p in r["cross_world"]["pairs"]:
        print(f"{p['world_i']}-{p['world_j']:<9}{p['rms_shift']:>10.4f}{p['rms_slope']:>10.4f}"
              f"{p['rms_residual']:>11.4f}{p['rms_total']:>10.4f}"
              f"{p['frac_residual']:>8.1%}{p['sign_crossings']:>7}  {p['dominant']}")
    m = r["mean_components"]
    print(f"\n{'MEAN':<12}{m['rms_shift']:>10.4f}{m['rms_slope']:>10.4f}"
          f"{m['rms_residual']:>11.4f}{m['rms_total']:>10.4f}"
          f"{r['mean_energy_fractions']['frac_residual']:>8.1%}")
    f = r["floors"]
    nm = f["same_curve_null"]["mean_over_pairs_null"]
    print(f"{'NULL mean':<12}{nm['rms_shift']['mean']:>10.4f}{nm['rms_slope']['mean']:>10.4f}"
          f"{nm['rms_residual']['mean']:>11.4f}{nm['rms_total']['mean']:>10.4f}"
          f"   same-curve null, {f['same_curve_null']['n_repeats']} resamples")
    print(f"{'NULL p95':<12}{nm['rms_shift']['p95']:>10.4f}{nm['rms_slope']['p95']:>10.4f}"
          f"{nm['rms_residual']['p95']:>11.4f}{nm['rms_total']['p95']:>10.4f}"
          f"   <- THE GATE")
    rr = r["ratio_over_null"]
    print(f"{'obs / null':<12}{rr['rms_shift']:>9.2f}x{rr['rms_slope']:>9.2f}x"
          f"{rr['rms_residual']:>10.2f}x{rr['rms_total']:>9.2f}x")
    print(f"{'floor member':<12}{f['member']['p95']['rms_shift']:>10.4f}"
          f"{f['member']['p95']['rms_slope']:>10.4f}"
          f"{f['member']['p95']['rms_residual']:>11.4f}"
          f"{f['member']['p95']['rms_total']:>10.4f}   (p95, {f['member']['n_comparisons']} comps)")
    print(f"{'floor split':<12}{f['split_half']['p95']['rms_shift']:>10.4f}"
          f"{f['split_half']['p95']['rms_slope']:>10.4f}"
          f"{f['split_half']['p95']['rms_residual']:>11.4f}"
          f"{f['split_half']['p95']['rms_total']:>10.4f}   (p95, "
          f"{f['split_half']['n_comparisons']} comps, n/2)")
    print(f"\ndominant component per pair: {r['dominant_counts']}   "
          f"on average: {r['dominant_on_average'].upper()}")
    print(f"clears operative floor: {r['components_clearing_floor']}")
    print(f"pairs whose residual exceeds the floor: "
          f"{r['pairs_with_residual_over_floor']}/{r['cross_world']['n_pairs']}   "
          f"pairs whose curves cross: {r['pairs_that_cross']}/{r['cross_world']['n_pairs']}")
    lr = r["largest_residual_pair"]
    print(f"largest residual: worlds {lr['world_i']}-{lr['world_j']}  "
          f"rms {lr['rms_residual']:.4f}  ({lr['frac_residual']:.1%} of the difference, "
          f"{lr['sign_crossings']} crossings)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,E")
    ap.add_argument("--state", default="supply_stress")
    ap.add_argument("--seeds", default=",".join(str(d) for d in DSEEDS))
    ap.add_argument("--init-seeds", default=",".join(str(s) for s in INIT_SEEDS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    blob = {}
    for v in [s.strip() for s in a.variants.split(",") if s.strip()]:
        r = run(v, a.state, dseeds, iseeds)
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
