#!/usr/bin/env python3
"""
Operating-point analysis for `delay`, `shortage`, `impact` — precision and recall at real
decision thresholds, from predictions already computed.

**Read-only.** No retraining, no new forward passes, no change to any shipped calibration
correction. Everything comes from the prediction cache built by `ml/calibration_correction.py`.

**Why not accuracy.** At base rates of 12.4 % / 5.4 % / 3.25 %, a model that answers "no" to
everything scores 87.6 % / 94.6 % / 96.75 % "accuracy" while catching zero real cases. This module
therefore does not compute a combined accuracy figure for any task, deliberately and by
instruction -- precision and recall are always reported together and never collapsed.

**Which probability is thresholded.** Each task's FINAL shipped chain from
`reports/calibration_correction_2.md`:

    delay     analytic prior shift only
    shortage  analytic prior shift only
    impact    analytic + temperature + beta

Anything fitted is applied OUT OF FOLD -- `impact`'s T and (a, b) for a given world are the ones
fitted on the other four -- so no operating point is read off a correction tuned on the data it is
scored against.

**Per-cell, then averaged.** Precision and recall are computed inside each of the 25 cells on a
shared threshold grid and then averaged across cells, matching the convention every other metric
in this project uses. A single pooled mixture of 25 models would hide exactly the world-to-world
spread item 3 exists to expose.

**Threshold selection is a fit, so it is also checked leave-one-world-out.** A threshold is one
parameter chosen from data; `reports/world_conditioned_calibration.md`'s STOP - G history applies
to it as much as to a calibration map. The headline thresholds are chosen on the 25-cell averaged
curve, and a LOWO transfer check (choose on 4 worlds, score on the 5th) quantifies how much of the
reported performance is optimism.

    python3 ml/operating_points.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import warnings

import numpy as np

# Thresholds above every prediction in a cell flag nothing, so precision there is undefined and
# recorded as NaN; averaging a column that is NaN in every cell is expected, not an error.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.calibration_correction import (build_cache, fit_temperature,  # noqa: E402
                                       prior_shift, temperature)
from ml.calibration_correction2 import SHIPPED, beta_calibrate, fit_beta  # noqa: E402
from ml.models.depth import TASKS                                     # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")
# The FINAL chain each task ships after reports/calibration_correction_2.md. `SHIPPED` (imported)
# names the chain the beta step was fitted ON TOP OF, which for `impact` is one stage short.
FINAL_CHAIN = {"delay": "analytic", "shortage": "analytic",
               "impact": "analytic + temperature + beta"}
# reports/calibration_correction_2.md section 4: above this the ABSOLUTE probability is not
# trustworthy (the ranking still is).
RANK_ONLY_ABOVE = {"delay": 0.20, "shortage": 0.20, "impact": 0.40}


# --------------------------------------------------------------------------- shipped chain

def shipped(store, task, dseeds):
    """Final shipped probabilities, out-of-fold wherever anything was fitted."""
    p, y, w, m = (store[f"{task}_p"], store[f"{task}_y"],
                  store[f"{task}_w"], store[f"{task}_m"])
    tp = {d: float(store[f"{task}_trainpi"][i]) for i, d in enumerate(dseeds)}
    pa = np.concatenate([prior_shift(p[w == d], tp[d]) for d in dseeds])
    ya = np.concatenate([y[w == d] for d in dseeds])
    wa = np.concatenate([w[w == d] for d in dseeds])
    ma = np.concatenate([m[w == d] for d in dseeds])
    raw = np.concatenate([p[w == d] for d in dseeds])
    if SHIPPED[task] == "analytic":
        return raw, pa, ya, wa, ma
    ps = np.empty_like(pa)
    for held in dseeds:                       # T then beta, both fitted on the OTHER four worlds
        fit = wa != held
        T = fit_temperature(pa[fit], ya[fit])
        aa, bb = fit_beta(temperature(pa[fit], T), ya[fit])
        ps[wa == held] = beta_calibrate(temperature(pa[wa == held], T), aa, bb)
    return raw, ps, ya, wa, ma


# --------------------------------------------------------------------------- PR machinery

def pr_at(p, y, grid):
    """Precision and recall at every threshold in `grid`, for one cell.

    One descending sort plus a cumulative sum, then `searchsorted` per threshold -- O(n log n)
    once rather than O(n) per threshold, which matters at ~19k rows x 25 cells x 4k thresholds.
    A threshold that flags nothing has undefined precision; it is recorded as NaN and excluded
    from the average rather than imputed as 0 or 1, either of which would bias the curve.
    """
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    cum = np.concatenate([[0.0], np.cumsum(ys)])
    npos = float(ys.sum())
    k = np.searchsorted(-ps, -grid, side="right")        # how many have p >= t
    tp = cum[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(k > 0, tp / np.maximum(k, 1), np.nan)
        rec = tp / npos if npos > 0 else np.full_like(tp, np.nan)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return prec, rec, f1, k


def curve(p, y, w, m, dseeds, mseeds, grid):
    """Mean precision / recall / F1 over the 25 cells, plus the per-world means."""
    P, R, F, FLAG = [], [], [], []
    per_world = {}
    for d in dseeds:
        wp, wr, wf = [], [], []
        for s in mseeds:
            k = (w == d) & (m == s)
            pr, rc, f1, nf = pr_at(p[k], y[k], grid)
            P.append(pr); R.append(rc); F.append(f1); FLAG.append(nf / max(1, k.sum()))
            wp.append(pr); wr.append(rc); wf.append(f1)
        per_world[d] = (np.nanmean(wp, axis=0), np.nanmean(wr, axis=0), np.nanmean(wf, axis=0))
    return (np.nanmean(P, axis=0), np.nanmean(R, axis=0), np.nanmean(F, axis=0),
            np.nanmean(FLAG, axis=0), per_world)


MIN_RECALL = 0.05        # a "high-precision" point that catches almost nothing is not an
                         # operating point; see `pick`.


def pick(grid, mp, mr, mf, mode, target=0.80):
    """Index of the chosen operating point. Returns `(idx, attainable)`.

    When 80 % precision is unattainable, the fallback is the most precise threshold that still
    catches at least `MIN_RECALL` of real cases -- NOT the global precision maximum. Unconstrained,
    that maximum sits where only a handful of items are flagged per cell (precision estimated on
    1-2 predictions, recall ~0), which is a statistical artifact rather than something anyone
    could operate at.
    """
    if mode == "f1":
        return int(np.nanargmax(mf)), True
    if mode == "recall":
        ok = np.where(mr >= target)[0]                    # highest threshold still catching >=80%
        return (int(ok[-1]), True) if len(ok) else (int(np.nanargmax(mr)), False)
    ok = np.where(mp >= target)[0]                        # lowest threshold still >=80% precise
    if len(ok):
        return int(ok[0]), True
    usable = np.where(mr >= MIN_RECALL)[0]
    if not len(usable):
        return int(np.nanargmax(mp)), False
    return int(usable[np.nanargmax(mp[usable])]), False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "operating_points.json"))
    a = ap.parse_args()
    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    store = build_cache(dseeds, mseeds)
    blob = {"config": vars(a), "shipped": FINAL_CHAIN, "rank_only_above": RANK_ONLY_ABOVE,
            "tasks": {}, "percentile_check": {}, "lowo_threshold": {}}

    MODES = [("f1", "balanced (max F1)"), ("recall", "high-recall (~80% caught)"),
             ("precision", "high-precision (~80% of flags real)")]

    for t in TASKS:
        raw, ps, y, w, m = shipped(store, t, dseeds)
        # Grid over the actual distribution of the shipped score, so thresholds land where the
        # data is rather than uniformly across an empty [0,1].
        grid = np.unique(np.concatenate([
            np.quantile(ps, np.linspace(0, 1, 4001)), np.linspace(0, 1, 201)]))
        mp, mr, mf, mflag, per_world = curve(ps, y, w, m, dseeds, mseeds, grid)
        base = float(y.mean())
        rows = []
        for mode, label in MODES:
            i, attainable = pick(grid, mp, mr, mf, mode)
            pw_p = {str(d): float(per_world[d][0][i]) for d in dseeds}
            pw_r = {str(d): float(per_world[d][1][i]) for d in dseeds}
            rows.append({
                "mode": mode, "label": label, "threshold": float(grid[i]),
                "precision": float(mp[i]), "recall": float(mr[i]), "f1": float(mf[i]),
                "flagged_frac": float(mflag[i]), "attainable": attainable,
                "rank_only": bool(grid[i] > RANK_ONLY_ABOVE[t]),
                "per_world_precision": pw_p, "per_world_recall": pw_r,
                "precision_world_spread": float(max(pw_p.values()) - min(pw_p.values())),
                "recall_world_spread": float(max(pw_r.values()) - min(pw_r.values())),
            })
        blob["tasks"][t] = {"base_rate": base, "chain": FINAL_CHAIN[t], "operating_points": rows}

        # ---- item 4a: matched-PERCENTILE cutoffs must be identical raw vs corrected --------
        worst_p = worst_r = 0.0
        fracs = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
        for d in dseeds:
            for s in mseeds:
                k = (w == d) & (m == s)
                pr_, yr_ = raw[k], y[k]
                pc_ = ps[k]
                n, npos = len(yr_), float(yr_.sum())
                for f in fracs:
                    K = max(1, int(round(f * n)))
                    ir = np.argsort(-pr_, kind="stable")[:K]
                    ic = np.argsort(-pc_, kind="stable")[:K]
                    worst_p = max(worst_p, abs(yr_[ir].sum() / K - yr_[ic].sum() / K))
                    if npos:
                        worst_r = max(worst_r, abs(yr_[ir].sum() / npos - yr_[ic].sum() / npos))
        blob["percentile_check"][t] = {"max_abs_precision_diff": worst_p,
                                       "max_abs_recall_diff": worst_r,
                                       "fractions": fracs.tolist()}

        # ---- item 4b: the SAME three operating points on the RAW probability ---------------
        graw = np.unique(np.concatenate([
            np.quantile(raw, np.linspace(0, 1, 4001)), np.linspace(0, 1, 201)]))
        rp, rr, rf, rflag, _pw = curve(raw, y, w, m, dseeds, mseeds, graw)
        rrows = []
        for mode, label in MODES:
            i, attainable = pick(graw, rp, rr, rf, mode)
            rrows.append({"mode": mode, "label": label, "threshold": float(graw[i]),
                          "precision": float(rp[i]), "recall": float(rr[i]),
                          "f1": float(rf[i]), "flagged_frac": float(rflag[i]),
                          "attainable": attainable})
        blob["tasks"][t]["operating_points_raw"] = rrows

        # ---- item 3 extension: does the balanced threshold TRANSFER between worlds? --------
        folds = []
        for held in dseeds:
            keep = [d for d in dseeds if d != held]
            _p, _r, _f, _fl, _pw2 = curve(ps, y, w, m, keep, mseeds, grid)
            j = int(np.nanargmax(_f))
            hp, hr, hf, _hfl, _ = curve(ps, y, w, m, [held], mseeds, grid)
            folds.append({"held_out": held, "threshold_from_4": float(grid[j]),
                          "precision_held": float(hp[j]), "recall_held": float(hr[j]),
                          "f1_held": float(hf[j])})
        insample = [r for r in rows if r["mode"] == "f1"][0]
        blob["lowo_threshold"][t] = {
            "folds": folds,
            "f1_lowo_mean": float(np.mean([f["f1_held"] for f in folds])),
            "f1_insample": insample["f1"],
            "threshold_spread": float(max(f["threshold_from_4"] for f in folds)
                                      - min(f["threshold_from_4"] for f in folds))}

    # ------------------------------------------------------------------ printing
    print("=" * 118)
    print("OPERATING POINTS — precision and recall on the FINAL shipped probability "
          "(25 cells, per-cell then averaged)")
    print("=" * 118)
    h = (f"{'task':<10}{'operating point':<34}{'thresh':>8}{'catches N/100':>15}"
         f"{'M/100 flags real':>18}{'F1':>7}{'% flagged':>11}{'band':>11}")
    print(h); print("-" * len(h))
    for t in TASKS:
        d = blob["tasks"][t]
        for r in d["operating_points"]:
            note = "RANK-ONLY" if r["rank_only"] else "absolute"
            star = "" if r["attainable"] else "  (target not attainable; best available)"
            print(f"{t if r is d['operating_points'][0] else '':<10}{r['label']:<34}"
                  f"{r['threshold']:>8.4f}{r['recall'] * 100:>15.1f}{r['precision'] * 100:>18.1f}"
                  f"{r['f1']:>7.3f}{r['flagged_frac'] * 100:>10.2f}%{note:>11}{star}")
        print(f"{'':<10}base rate {d['base_rate']:.2%}  chain '{d['chain']}'")
        print("-" * len(h))

    print("\nWORLD-TO-WORLD SPREAD AT THE BALANCED POINT")
    h2 = (f"{'task':<10}{'metric':<12}" + "".join(f"{'d' + str(x):>9}" for x in dseeds)
          + f"{'mean':>9}{'spread':>9}")
    print(h2); print("-" * len(h2))
    for t in TASKS:
        r = [x for x in blob["tasks"][t]["operating_points"] if x["mode"] == "f1"][0]
        for key, lbl in (("per_world_recall", "recall"), ("per_world_precision", "precision")):
            v = r[key]
            print(f"{t if lbl == 'recall' else '':<10}{lbl:<12}"
                  + "".join(f"{v[str(x)]:>9.3f}" for x in dseeds)
                  + f"{np.mean(list(v.values())):>9.3f}"
                  + f"{max(v.values()) - min(v.values()):>9.3f}")
        print("-" * len(h2))

    print("\nTHRESHOLD TRANSFER (balanced point chosen on 4 worlds, scored on the held-out 5th)")
    h3 = f"{'task':<10}{'F1 in-sample':>14}{'F1 held-out':>13}{'drop':>9}{'thresh spread':>15}"
    print(h3); print("-" * len(h3))
    for t in TASKS:
        L = blob["lowo_threshold"][t]
        print(f"{t:<10}{L['f1_insample']:>14.3f}{L['f1_lowo_mean']:>13.3f}"
              f"{L['f1_insample'] - L['f1_lowo_mean']:>+9.3f}{L['threshold_spread']:>15.4f}")

    print("\nRAW (pre-correction) vs CORRECTED at MATCHED PERCENTILE cutoffs — must be identical")
    h4 = f"{'task':<10}{'max |d precision|':>20}{'max |d recall|':>17}"
    print(h4); print("-" * len(h4))
    for t in TASKS:
        c = blob["percentile_check"][t]
        print(f"{t:<10}{c['max_abs_precision_diff']:>20.2e}{c['max_abs_recall_diff']:>17.2e}")

    print("\nSAME OPERATING POINTS ON THE RAW PROBABILITY (thresholds differ; P/R need not match)")
    h5 = (f"{'task':<10}{'operating point':<34}{'raw thresh':>12}{'catches N/100':>15}"
          f"{'M/100 real':>13}{'F1':>7}")
    print(h5); print("-" * len(h5))
    for t in TASKS:
        for r in blob["tasks"][t]["operating_points_raw"]:
            print(f"{t if r is blob['tasks'][t]['operating_points_raw'][0] else '':<10}"
                  f"{r['label']:<34}{r['threshold']:>12.4f}{r['recall'] * 100:>15.1f}"
                  f"{r['precision'] * 100:>13.1f}{r['f1']:>7.3f}")
        print("-" * len(h5))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
