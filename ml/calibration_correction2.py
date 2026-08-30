#!/usr/bin/env python3
"""
Calibration correction 2 — is the sparse high-probability tail residual real, and if so, does a
2-parameter beta top-up survive leave-one-world-out?

`reports/calibration_correction_1.md` shipped: the analytic prior shift for all three tasks
(zero fitting, 86-89% of ECE removed), plus a LOWO-validated scalar temperature for `impact`
alone (T ~ 1.314; rejected for `shortage` on a failed Brier check, passed-but-inert for `delay`).
What survives on all three is concentrated in sparse high-probability bins where the corrected
model is now UNDER-confident -- the opposite of the original failure, and thin enough (well under
1% of prediction mass) that a point gap estimate may be describing sampling noise.

**This module never removes or downgrades anything already shipped.** It only asks whether an
ADDITIONAL top-up is justified, and it answers "no" by leaving the task exactly as it was.

STEP 1 -- is the residual real? Wilson score intervals on the observed rate in each sparse bin.
A bin whose mean predicted value falls INSIDE the 95% CI of its own observed rate is not evidence
of bias at this sample size, and nothing is fitted to it. Wilson rather than the normal
approximation because these bins hold 1-500 points at rates near 0 or 1, exactly where the normal
interval misbehaves.

STEP 2 -- only where step 1 finds a CI-excluding residual: `p' = sigmoid(a*logit(p) + b)`, fitted
by NLL. One parameter more than the temperature, deliberately: after the analytic step the
residual is no longer single-signed (bulk and tail pull opposite ways), and a single scalar cannot
represent that -- which is precisely why `shortage`'s temperature failed. Still not isotonic:
that flexibility is what failed to transfer (**STOP - G**).

STEP 3 -- LOWO, identical protocol to the temperature fit, with the same acceptance rule already
established: **a fit that improves ECE while degrading Brier on held-out folds is REJECTED**,
because that pattern is bin-boundary gaming rather than calibration gain.

**Out-of-fold shipped probabilities.** `impact`'s shipped correction contains a fitted T, so
evaluating its residual on a T fitted to the same data would flatter it. Every `impact` number
here uses the T fitted on the OTHER four worlds. `delay` and `shortage` ship analytic-only, which
is unfitted, so no such care is needed for them.

    python3 ml/calibration_correction2.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.calibration_correction import (EPS, _logit, build_cache,  # noqa: E402
                                       fit_temperature, prior_shift, temperature)
from ml.evaluate import calibration_error                          # noqa: E402
from ml.models.depth import TASKS                                  # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")

# What each task currently ships, per reports/calibration_correction_1.md section 6.
SHIPPED = {"delay": "analytic", "shortage": "analytic", "impact": "analytic+temperature"}
# The sparse tail each task is tested over, per the brief and section 2's decile table.
TAIL_FLOOR = {"delay": 0.2, "shortage": 0.2, "impact": 0.1}
DECILES = np.linspace(0.0, 1.0, 11)


# --------------------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate.

    Chosen over the normal approximation because these bins hold as few as 1 point at rates near
    0 or 1, where the normal interval can extend outside [0,1] and badly under-cover. Wilson is
    the standard fix and is well behaved at n=1.
    """
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def beta_calibrate(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """`sigmoid(a*logit(p) + b)`. Strictly increasing in p for a>0, so ranking is preserved."""
    return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))


def fit_beta(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit `(a, b)` by NLL -- a proper scoring rule, the same discipline the temperature fit used.

    `a` is constrained positive (parameterised as exp) so the map cannot invert the ranking; that
    would change AUC, which this whole correction chain is required not to do.
    """
    z = _logit(p)

    def nll(theta):
        a, b = np.exp(theta[0]), theta[1]
        q = np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), EPS, 1.0 - EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    best = min((minimize(nll, x0, method="Nelder-Mead",
                         options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
                for x0 in ([0.0, 0.0], [np.log(1.3), 0.0], [np.log(0.8), 0.5])),
               key=lambda r: r.fun)
    return float(np.exp(best.x[0])), float(best.x[1])


def brier(p, y):
    return float(np.mean((p - y) ** 2))


# --------------------------------------------------------------------------- shipped chain

def shipped_oof(store, task, dseeds, mseeds):
    """The probabilities this task currently ships, computed OUT OF FOLD where anything was
    fitted. Returns `(p_shipped, y, world, mseed, per_fold_T)`."""
    p, y = store[f"{task}_p"], store[f"{task}_y"]
    w, m = store[f"{task}_w"], store[f"{task}_m"]
    trainpi = {d: float(store[f"{task}_trainpi"][i]) for i, d in enumerate(dseeds)}
    pa = np.concatenate([prior_shift(p[w == d], trainpi[d]) for d in dseeds])
    ya = np.concatenate([y[w == d] for d in dseeds])
    wa = np.concatenate([w[w == d] for d in dseeds])
    ma = np.concatenate([m[w == d] for d in dseeds])
    Ts = {}
    if SHIPPED[task] == "analytic+temperature":
        ps = np.empty_like(pa)
        for held in dseeds:                    # each world corrected by the OTHER four's T
            T = fit_temperature(pa[wa != held], ya[wa != held])
            Ts[held] = T
            ps[wa == held] = temperature(pa[wa == held], T)
    else:
        ps = pa
    return ps, ya, wa, ma, Ts


def tail_test(p, y, floor) -> list[dict]:
    """Per-decile Wilson test over the sparse tail, plus one pooled row for the whole tail."""
    rows = []
    for i in range(10):
        lo, hi = DECILES[i], DECILES[i + 1]
        if hi <= floor:
            continue
        m = (p >= lo) & (p < hi) if i < 9 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        clo, chi = wilson(k, n)
        mp = float(p[m].mean())
        rows.append({"bin": f"{lo:.0%}-{hi:.0%}", "n": n, "pos": k, "mean_pred": mp,
                     "obs_rate": k / n, "ci_lo": clo, "ci_hi": chi,
                     "outside_ci": bool(mp < clo or mp > chi),
                     "direction": "under-confident" if mp < clo else
                                  ("over-confident" if mp > chi else "indistinguishable")})
    m = p >= floor
    n, k = int(m.sum()), int(y[m].sum())
    if n:
        clo, chi = wilson(k, n)
        mp = float(p[m].mean())
        rows.append({"bin": f">={floor:.0%} (pooled)", "n": n, "pos": k, "mean_pred": mp,
                     "obs_rate": k / n, "ci_lo": clo, "ci_hi": chi,
                     "outside_ci": bool(mp < clo or mp > chi),
                     "direction": "under-confident" if mp < clo else
                                  ("over-confident" if mp > chi else "indistinguishable"),
                     "pooled": True})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "calibration_correction2.json"))
    a = ap.parse_args()
    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    store = build_cache(dseeds, mseeds)
    blob = {"config": vars(a), "shipped": SHIPPED, "tail_floor": TAIL_FLOOR,
            "step1": {}, "step2": {}, "usage": {}}

    ship = {}
    for t in TASKS:
        ship[t] = shipped_oof(store, t, dseeds, mseeds)

    # ------------------------------------------------------------------ STEP 1
    print("=" * 112)
    print("STEP 1 — is the sparse tail residual distinguishable from noise? (95% Wilson CI on "
          "the observed rate)")
    print("=" * 112)
    h = (f"{'task':<10}{'bin':>16}{'n':>9}{'pos':>7}{'mean pred':>11}{'obs rate':>10}"
         f"{'95% CI on observed':>24}{'verdict':>22}")
    print(h); print("-" * len(h))
    for t in TASKS:
        ps, ya, wa, ma, Ts = ship[t]
        rows = tail_test(ps, ya, TAIL_FLOOR[t])
        blob["step1"][t] = rows
        for r in rows:
            ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            mark = "REAL bias" if r["outside_ci"] else "noise"
            print(f"{t if r is rows[0] else '':<10}{r['bin']:>16}{r['n']:>9,}{r['pos']:>7,}"
                  f"{r['mean_pred']:>11.4f}{r['obs_rate']:>10.4f}{ci:>24}"
                  f"{mark + ' / ' + r['direction']:>22}")
        print("-" * len(h))
    real = {t: any(r["outside_ci"] for r in blob["step1"][t]) for t in TASKS}
    pooled_real = {t: [r for r in blob["step1"][t] if r.get("pooled")][0]["outside_ci"]
                   for t in TASKS}
    print("\n  tail residual real (any bin CI-excluding): "
          + "  ".join(f"{t}={'YES' if real[t] else 'no'}" for t in TASKS))
    print("  tail residual real (pooled tail):           "
          + "  ".join(f"{t}={'YES' if pooled_real[t] else 'no'}" for t in TASKS))

    # ------------------------------------------------------------------ STEP 2/3
    print("\n" + "=" * 112)
    print("STEP 2/3 — beta top-up p'=sigmoid(a*logit(p)+b), fitted on 4 worlds, scored on the "
          "held-out 5th")
    print("=" * 112)
    h2 = (f"{'task':<10}{'held-out':>9}{'a':>8}{'b':>9}{'ECE shipped':>13}{'ECE +beta':>11}"
          f"{'dECE':>9}{'Brier ship':>12}{'Brier +beta':>13}{'dBrier':>10}")
    for t in TASKS:
        if not real[t]:
            print(f"\n{t}: NO FIT ATTEMPTED — tail residual is statistically indistinguishable "
                  f"from noise; left exactly as reports/calibration_correction_1.md shipped it.")
            blob["step2"][t] = {"attempted": False,
                                "reason": "tail residual within 95% CI at current sample size"}
            continue
        print(f"\n{t} (on top of: {SHIPPED[t]})")
        print(h2); print("-" * len(h2))
        p, y = store[f"{t}_p"], store[f"{t}_y"]
        w = store[f"{t}_w"]
        trainpi = {d: float(store[f"{t}_trainpi"][i]) for i, d in enumerate(dseeds)}
        pa = np.concatenate([prior_shift(p[w == d], trainpi[d]) for d in dseeds])
        ya = np.concatenate([y[w == d] for d in dseeds])
        wa = np.concatenate([w[w == d] for d in dseeds])
        ma = np.concatenate([store[f"{t}_m"][w == d] for d in dseeds])
        folds = []
        for held in dseeds:
            fit = wa != held
            # rebuild the ENTIRE shipped chain inside the fold, so the baseline and the top-up
            # see exactly the same information
            if SHIPPED[t] == "analytic+temperature":
                T = fit_temperature(pa[fit], ya[fit])
                base_fit, base_held = temperature(pa[fit], T), temperature(pa[~fit], T)
            else:
                T = None
                base_fit, base_held = pa[fit], pa[~fit]
            aa, bb = fit_beta(base_fit, ya[fit])
            q_held = beta_calibrate(base_held, aa, bb)
            yh, mh = ya[~fit], ma[~fit]
            e0, e1, b0, b1 = [], [], [], []
            for s in mseeds:
                k = mh == s
                e0.append(calibration_error(yh[k], base_held[k], 10))
                e1.append(calibration_error(yh[k], q_held[k], 10))
                b0.append(brier(base_held[k], yh[k]))
                b1.append(brier(q_held[k], yh[k]))
            row = {"held_out": held, "a": aa, "b": bb, "T": T,
                   "ece_shipped": float(np.mean(e0)), "ece_beta": float(np.mean(e1)),
                   "brier_shipped": float(np.mean(b0)), "brier_beta": float(np.mean(b1))}
            row["ece_delta"] = row["ece_shipped"] - row["ece_beta"]
            row["brier_delta"] = row["brier_shipped"] - row["brier_beta"]
            folds.append(row)
            print(f"{'':<10}{held:>9}{aa:>8.4f}{bb:>9.4f}{row['ece_shipped']:>13.4f}"
                  f"{row['ece_beta']:>11.4f}{row['ece_delta']:>+9.4f}"
                  f"{row['brier_shipped']:>12.4f}{row['brier_beta']:>13.4f}"
                  f"{row['brier_delta']:>+10.4f}")
        ne = sum(1 for f in folds if f["ece_delta"] > 0)
        nb = sum(1 for f in folds if f["brier_delta"] > 0)
        rec = {"attempted": True, "folds": folds,
               "a_mean": float(np.mean([f["a"] for f in folds])),
               "b_mean": float(np.mean([f["b"] for f in folds])),
               "a_spread": float(max(f["a"] for f in folds) - min(f["a"] for f in folds)),
               "b_spread": float(max(f["b"] for f in folds) - min(f["b"] for f in folds)),
               "ece_shipped_mean": float(np.mean([f["ece_shipped"] for f in folds])),
               "ece_beta_mean": float(np.mean([f["ece_beta"] for f in folds])),
               "brier_shipped_mean": float(np.mean([f["brier_shipped"] for f in folds])),
               "brier_beta_mean": float(np.mean([f["brier_beta"] for f in folds])),
               "folds_improved_ece": ne, "folds_improved_brier": nb, "n_folds": len(folds)}
        rec["ece_delta_mean"] = rec["ece_shipped_mean"] - rec["ece_beta_mean"]
        rec["brier_delta_mean"] = rec["brier_shipped_mean"] - rec["brier_beta_mean"]
        # The acceptance rule, matching the precedent set in reports/calibration_correction_1.md:
        # BRIER ARBITRATES. There, `impact`'s temperature improved Brier on 5/5 folds and was
        # accepted, while `shortage`'s improved ECE on 4/5 but degraded Brier on 5/5 and was
        # rejected as bin-boundary gaming. Generalised here to the bar this project's STOP - G
        # history justifies: a FITTED correction ships only if it improves the proper scoring
        # rule on EVERY held-out world. One world where it hurts is precisely the transfer
        # failure that has already been recorded twice. ECE is reported but does not arbitrate:
        # it is a binned statistic and can move for reasons Brier cannot.
        rec["accepted"] = bool(nb == len(folds))
        rec["rejected_for_brier"] = bool(nb < len(folds))
        rec["gaming_signature"] = bool(ne > nb and nb < len(folds))
        blob["step2"][t] = rec
        print("-" * len(h2))
        print(f"{'':<10}{'MEAN':>9}{rec['a_mean']:>8.4f}{rec['b_mean']:>9.4f}"
              f"{rec['ece_shipped_mean']:>13.4f}{rec['ece_beta_mean']:>11.4f}"
              f"{rec['ece_delta_mean']:>+9.4f}{rec['brier_shipped_mean']:>12.4f}"
              f"{rec['brier_beta_mean']:>13.4f}{rec['brier_delta_mean']:>+10.4f}")
        print(f"{'':<10}ECE improved {ne}/5 folds, Brier improved {nb}/5 folds  ->  "
              f"{'ACCEPT' if rec['accepted'] else 'REJECT'}"
              + ("  (Brier degrades on a held-out world = does not transfer)"
                 if rec["rejected_for_brier"] else "")
              + ("  [ECE-up/Brier-down gaming signature]" if rec["gaming_signature"] else ""))

    # ------------------------------------------------------------------ STEP 5 usage bands
    # Built on the FINAL shipped chain, i.e. including any beta accepted above, and always
    # out-of-fold for anything fitted.
    final = {}
    for t in TASKS:
        ps, ya, wa, ma, _T = ship[t]
        rec = blob["step2"].get(t, {})
        if rec.get("accepted"):
            pf = np.empty_like(ps)
            for f in rec["folds"]:                 # each world scored by the OTHER four's (a,b)
                k = wa == f["held_out"]
                pf[k] = beta_calibrate(ps[k], f["a"], f["b"])
            final[t] = (pf, ya, SHIPPED[t] + "+beta")
        else:
            final[t] = (ps, ya, SHIPPED[t])
    blob["final_chain"] = {t: final[t][2] for t in TASKS}

    print("\n" + "=" * 108)
    print("USAGE GUIDANCE — where the FINAL shipped probability is trustworthy as an "
          "ABSOLUTE number")
    print("=" * 108)
    print("  absolute      : |mean predicted - observed| < 0.05 (accurate enough to read as a probability)")
    print("  rank only     : gap >= 0.05 AND predicted outside the 95% CI -- demonstrated bias")
    print("  insufficient  : gap >= 0.05 but predicted inside the CI -- too few points to tell either way")
    h3 = (f"{'task':<10}{'band':>14}{'mass':>9}{'n':>9}{'mean pred':>11}{'obs rate':>10}"
          f"{'|gap|':>9}{'in CI?':>9}{'use':>15}")
    print(h3); print("-" * len(h3))
    for t in TASKS:
        ps, ya, _lbl = final[t]
        bands = []
        for i in range(10):
            lo, hi = DECILES[i], DECILES[i + 1]
            m = (ps >= lo) & (ps < hi) if i < 9 else (ps >= lo) & (ps <= hi)
            n = int(m.sum())
            if not n:
                continue
            k = int(ya[m].sum())
            clo, chi = wilson(k, n)
            mp = float(ps[m].mean())
            inci = bool(clo <= mp <= chi)
            gap = abs(mp - k / n)
            trust = ("absolute" if gap < 0.05
                     else ("insufficient" if inci else "rank only"))
            bands.append({"band": f"{lo:.0%}-{hi:.0%}", "n": n, "frac": n / len(ps),
                          "mean_pred": mp, "obs_rate": k / n, "gap": gap,
                          "in_ci": inci, "trust": trust})
            print(f"{t if len(bands) == 1 else '':<10}{bands[-1]['band']:>14}"
                  f"{bands[-1]['frac']:>9.1%}{n:>9,}{mp:>11.4f}{k / n:>10.4f}{gap:>9.4f}"
                  f"{('yes' if inci else 'NO'):>9}{trust:>15}")
        blob["usage"][t] = bands
        ok = sum(b["frac"] for b in bands if b["trust"] == "absolute")
        rk = sum(b["frac"] for b in bands if b["trust"] == "rank only")
        un = sum(b["frac"] for b in bands if b["trust"] == "insufficient")
        blob["usage_summary"] = blob.get("usage_summary", {})
        blob["usage_summary"][t] = {"absolute_mass": ok, "rank_only_mass": rk,
                                    "insufficient_mass": un, "chain": final[t][2]}
        print(f"{'':<10}-> chain '{final[t][2]}':  absolute {ok:.2%}   "
              f"rank-only {rk:.2%}   insufficient {un:.2%}")
        print("-" * len(h3))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
