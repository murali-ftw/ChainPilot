#!/usr/bin/env python3
"""
Post-hoc calibration CORRECTION for `delay`, `shortage`, `impact` — analytic first, then a
leave-one-world-out-validated residual only where the analytic step does not close the gap.

`reports/calibration_diagnostic.md` established that all three tasks are overconfident
(3.0x / 6.7x / 5.0x overstatement), that the miscalibration is **uniformly single-signed**
(ECE collapses to |mean(p) - mean(y)| because every populated bin errs in the same direction),
and that the cause is known: `ml/train.py` fits `FocalLoss(gamma=2, alpha=1-pi)`, and
`alpha = 1-pi` rebalances the effective training prior to **exactly 0.5** for every task.

**Nothing is retrained.** SHARE and the Markov readout are untouched; this operates on
already-computed probabilities. Both corrections applied here are **strictly monotone in p**, so
every ranking statistic -- AUC included -- is mathematically invariant. `--assert-auc-invariant`
checks that numerically rather than asking the reader to take it on trust, because the one thing
a calibration layer must not do is quietly move a pass/fail bar defined on AUC.

STEP 1 -- analytic prior shift, ZERO data fitting. A model trained under effective prior `pi'`
and deployed under true prior `pi` is corrected by the standard prior-shift identity, applied to
the odds:

    odds_corrected = odds_raw * (pi / (1 - pi)) * ((1 - pi') / pi')

With `pi' = 0.5` the second factor is 1, giving the brief's form
`p_corr = p*pi / (p*pi + (1-p)*(1-pi))`. Nothing is tuned to any sample.

**Which `pi`.** The TRAIN split's positive rate, per world -- literally the quantity
`ml/train.py::compute_task_alphas` computed `alpha` from ("alpha is a training hyperparameter,
never tuned against held-out data"). Using the *test* base rate would be a leak, small but real,
and would make the "zero data fitting" claim false. The pooled-test-rate variant the brief quotes
is reported as a sensitivity check, not used.

STEP 2 -- residual scalar temperature, ONLY where a gap survives step 1, and only ever scored
leave-one-world-out. `reports/world_conditioned_calibration.md` and
`reports/layer3_uncertainty_aware.md` STEP 3 found a map fitted in one world *harms* another
(classified **STOP - G**), so a residual fit is admissible here only under LOWO and only as the
least flexible instrument that can work: one scalar, `p' = sigmoid(logit(p_corr) / T)`, fitted by
NLL on 4 worlds and scored on the 5th. Isotonic is deliberately NOT used -- its flexibility is
what failed to transfer before.

    python3 ml/calibration_correction.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.calibration_diagnostic import (ARMS, METRIC_KEYS, PRIMARY_ARM,  # noqa: E402
                                       agg, cell_metrics, reliability)
from ml.ds_backbone import get_backbone, load_world       # noqa: E402
from ml.evaluate import calibration_error, collect_predictions  # noqa: E402
from ml.models.depth import TASKS                         # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")
CACHE = os.path.join(REPO, "ml", ".cache", "calibration_predictions.npz")
EPS = 1e-7


# --------------------------------------------------------------------------- prediction cache

def build_cache(dseeds, mseeds, device="cpu") -> dict:
    """Raw test probabilities + labels for every cell of the PRIMARY arm of each task, plus each
    world's TRAIN-split positive rate.

    Cached to `.npz` for the same reason `ml/calibration_grid_cache.py` caches its grid: the
    forward passes are deterministic on frozen checkpoints, so rebuilding them per experiment is
    pure waste. Nothing here trains.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    store: dict = {}
    for task in TASKS:
        arm = PRIMARY_ARM[task]
        root, prefix, schema = ARMS[arm]
        P, Y, W, M, TR = [], [], [], [], {}
        for d in dseeds:
            csv_dir = os.path.join(root, f"{prefix}{d}")
            tr, _va, _te, _ids = load_world(csv_dir, device, schema)
            ys = [b.labels[task][1] for b in tr if b.labels[task][1].numel() > 0]
            TR[d] = float(np.concatenate([y.cpu().numpy() for y in ys]).mean())
            for m in mseeds:
                model, meta = get_backbone(csv_dir, "0", d, m, device=device, verbose=False,
                                           v3_schema=schema)
                pr = collect_predictions(model, meta["test_bundles"])[task]
                P.append(pr["p"].astype(np.float64))
                Y.append(pr["y"].astype(np.float64))
                W.append(np.full(len(pr["p"]), d))
                M.append(np.full(len(pr["p"]), m))
            print(f"  cached {task} d{d} (train pi={TR[d]:.4f})", flush=True)
        store[f"{task}_p"] = np.concatenate(P)
        store[f"{task}_y"] = np.concatenate(Y)
        store[f"{task}_w"] = np.concatenate(W)
        store[f"{task}_m"] = np.concatenate(M)
        store[f"{task}_trainpi"] = np.array([TR[d] for d in dseeds], dtype=float)
    store["dseeds"] = np.array(dseeds)
    store["mseeds"] = np.array(mseeds)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **store)
    return store


# --------------------------------------------------------------------------- the corrections

def prior_shift(p: np.ndarray, pi: float) -> np.ndarray:
    """`p*pi / (p*pi + (1-p)*(1-pi))` -- the analytic step. Strictly increasing in `p` for any
    `pi` in (0,1), so it cannot change any ranking."""
    num = p * pi
    return num / (num + (1.0 - p) * (1.0 - pi))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def temperature(p: np.ndarray, T: float) -> np.ndarray:
    """`sigmoid(logit(p)/T)`. Strictly increasing in `p` for any `T>0`; ranking-invariant."""
    return 1.0 / (1.0 + np.exp(-_logit(p) / T))


def fit_temperature(p: np.ndarray, y: np.ndarray, lo=0.05, hi=20.0, iters=200) -> float:
    """One scalar `T` minimising held-in NLL, by golden-section search on log T.

    NLL rather than ECE: NLL is a proper scoring rule and is smooth in T, whereas ECE is a
    step function of the binning and can be driven to a spurious optimum. Golden section rather
    than a gradient step because the search is one-dimensional and this is unimodal in log T --
    no optimiser state, no learning rate, nothing to tune.
    """
    def nll(T):
        q = np.clip(temperature(p, T), EPS, 1.0 - EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    a, b = np.log(lo), np.log(hi)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(np.exp(c)), nll(np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(np.exp(d))
        if b - a < 1e-6:
            break
    return float(np.exp((a + b) / 2.0))


# --------------------------------------------------------------------------- evaluation

def per_cell(store, task, transform, dseeds, mseeds) -> dict:
    """`cell_metrics` on every (world, init-seed) cell, exactly as the diagnostic computed them,
    with `transform(p, world)` applied first."""
    p, y = store[f"{task}_p"], store[f"{task}_y"]
    w, m = store[f"{task}_w"], store[f"{task}_m"]
    cells = {}
    for d in dseeds:
        for s in mseeds:
            k = (w == d) & (m == s)
            cells[(d, s, task)] = cell_metrics(transform(p[k], d), y[k])
    return cells


def pooled_reliability(store, task, transform, dseeds) -> list[dict]:
    p, y, w = store[f"{task}_p"], store[f"{task}_y"], store[f"{task}_w"]
    q = np.concatenate([transform(p[w == d], d) for d in dseeds])
    yy = np.concatenate([y[w == d] for d in dseeds])
    return reliability(q, yy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "calibration_correction.json"))
    a = ap.parse_args()
    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    store = build_cache(dseeds, mseeds, a.device)
    trainpi = {t: {d: float(store[f"{t}_trainpi"][i]) for i, d in enumerate(dseeds)}
               for t in TASKS}

    blob = {"config": vars(a), "primary_arm": PRIMARY_ARM, "train_pi": trainpi, "stages": {},
            "reliability": {}, "temperature": {}}

    ident = lambda p, d: p                                          # noqa: E731
    analytic = {t: (lambda p, d, _t=t: prior_shift(p, trainpi[_t][d])) for t in TASKS}

    # ---------------------------------------------------------------- stages 0 and 1
    for label, tf in (("raw", lambda t: ident), ("analytic", lambda t: analytic[t])):
        blob["stages"][label] = {}
        blob["reliability"][label] = {}
        for t in TASKS:
            cells = per_cell(store, t, tf(t), dseeds, mseeds)
            blob["stages"][label][t] = agg(cells, t, dseeds, mseeds, METRIC_KEYS)
            blob["reliability"][label][t] = pooled_reliability(store, t, tf(t), dseeds)

    print("\n=== train-split pi per world (the quantity focal alpha was derived from) ===")
    for t in TASKS:
        te = blob["stages"]["raw"][t]["base_rate"]["mean"]
        print(f"  {t:<9} train " + " ".join(f"d{d}:{trainpi[t][d]:.4f}" for d in dseeds)
              + f"   | test base rate {te:.4f}")

    hdr = (f"{'task':<10}{'stage':<12}{'AUC':>8}{'ECE':>9}{'Brier':>9}{'meanP':>9}"
           f"{'base':>8}{'gap':>10}{'|gap|/base':>12}")
    print("\n" + "=" * len(hdr))
    print("STEP 1 — analytic prior shift (zero data fitting)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for t in TASKS:
        for label in ("raw", "analytic"):
            s = blob["stages"][label][t]
            g = s["mean_gap"]["mean"]
            print(f"{t if label == 'raw' else '':<10}{label:<12}{s['auc']['mean']:>8.4f}"
                  f"{s['ece_equalwidth']['mean']:>9.4f}{s['brier']['mean']:>9.4f}"
                  f"{s['mean_pred']['mean']:>9.4f}{s['base_rate']['mean']:>8.4f}"
                  f"{g:>+10.4f}{abs(g) / s['base_rate']['mean']:>12.2f}")
        print("-" * len(hdr))

    # ---------------------------------------------------------------- AUC invariance
    worst = 0.0
    for t in TASKS:
        c_raw = per_cell(store, t, ident, dseeds, mseeds)
        c_an = per_cell(store, t, analytic[t], dseeds, mseeds)
        for k in c_raw:
            if c_raw[k].get("auc") is not None and c_an[k].get("auc") is not None:
                worst = max(worst, abs(c_raw[k]["auc"] - c_an[k]["auc"]))
    blob["auc_invariance_max_abs_diff"] = worst
    print(f"\nAUC invariance (monotone transform): max |AUC_raw - AUC_analytic| over "
          f"{len(TASKS) * len(dseeds) * len(mseeds)} cells = {worst:.2e}")

    # ---------------------------------------------------------------- step 2: LOWO temperature
    # Sequenced by the diagnostic's own transfer-risk ranking: impact (world spread 12% of its
    # gap), then shortage (18%), then delay (34% -- the regime where the earlier fit failed).
    order = ["impact", "shortage", "delay"]
    print("\n" + "=" * 100)
    print("STEP 2 — residual scalar temperature, fitted on 4 worlds, scored on the held-out 5th")
    print("=" * 100)
    h2 = (f"{'task':<10}{'held-out':>9}{'T (fit on 4)':>14}{'ECE analytic':>14}"
          f"{'ECE +temp':>11}{'delta':>10}{'Brier an.':>11}{'Brier +T':>10}{'delta':>10}")
    print(h2); print("-" * len(h2))
    for t in order:
        p, y = store[f"{t}_p"], store[f"{t}_y"]
        w, m = store[f"{t}_w"], store[f"{t}_m"]
        pc = np.concatenate([prior_shift(p[w == d], trainpi[t][d]) for d in dseeds])
        wc = np.concatenate([w[w == d] for d in dseeds])
        yc = np.concatenate([y[w == d] for d in dseeds])
        mc = np.concatenate([m[w == d] for d in dseeds])
        folds = []
        for held in dseeds:
            fit = wc != held
            T = fit_temperature(pc[fit], yc[fit])
            # score per CELL on the held-out world, then average -- the diagnostic's convention
            e_an, e_tp, b_an, b_tp = [], [], [], []
            for s in mseeds:
                k = (wc == held) & (mc == s)
                e_an.append(calibration_error(yc[k], pc[k], 10))
                b_an.append(float(np.mean((pc[k] - yc[k]) ** 2)))
                q = temperature(pc[k], T)
                e_tp.append(calibration_error(yc[k], q, 10))
                b_tp.append(float(np.mean((q - yc[k]) ** 2)))
            row = {"held_out": held, "T": T,
                   "ece_analytic": float(np.mean(e_an)), "ece_temp": float(np.mean(e_tp)),
                   "brier_analytic": float(np.mean(b_an)), "brier_temp": float(np.mean(b_tp))}
            row["ece_delta"] = row["ece_analytic"] - row["ece_temp"]
            row["brier_delta"] = row["brier_analytic"] - row["brier_temp"]
            folds.append(row)
            print(f"{t if held == dseeds[0] else '':<10}{held:>9}{T:>14.4f}"
                  f"{row['ece_analytic']:>14.4f}{row['ece_temp']:>11.4f}"
                  f"{row['ece_delta']:>+10.4f}{row['brier_analytic']:>11.4f}"
                  f"{row['brier_temp']:>10.4f}{row['brier_delta']:>+10.4f}")
        nimp = sum(1 for f in folds if f["ece_delta"] > 0)
        blob["temperature"][t] = {
            "folds": folds,
            "T_mean": float(np.mean([f["T"] for f in folds])),
            "T_spread": float(max(f["T"] for f in folds) - min(f["T"] for f in folds)),
            "ece_analytic_mean": float(np.mean([f["ece_analytic"] for f in folds])),
            "ece_temp_mean": float(np.mean([f["ece_temp"] for f in folds])),
            "ece_delta_mean": float(np.mean([f["ece_delta"] for f in folds])),
            "brier_analytic_mean": float(np.mean([f["brier_analytic"] for f in folds])),
            "brier_temp_mean": float(np.mean([f["brier_temp"] for f in folds])),
            "brier_delta_mean": float(np.mean([f["brier_delta"] for f in folds])),
            "folds_improved_brier": sum(1 for f in folds if f["brier_delta"] > 0),
            "folds_improved": nimp, "n_folds": len(folds),
            "lowo_passes": bool(nimp == len(folds)),
        }
        r = blob["temperature"][t]
        print(f"{'':<10}{'MEAN':>9}{r['T_mean']:>14.4f}{r['ece_analytic_mean']:>14.4f}"
              f"{r['ece_temp_mean']:>11.4f}{r['ece_delta_mean']:>+10.4f}"
              f"{r['brier_analytic_mean']:>11.4f}{r['brier_temp_mean']:>10.4f}"
              f"{r['brier_delta_mean']:>+10.4f}   "
              f"{'LOWO PASS' if r['lowo_passes'] else 'LOWO FAIL'} ({nimp}/{len(folds)} folds)")
        print("-" * len(h2))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
