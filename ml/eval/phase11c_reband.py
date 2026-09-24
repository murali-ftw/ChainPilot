"""Phase 11C Stage B — re-band the at-risk cells at five model seeds and restate the verdicts.

Scores every seed's stored predictions directly rather than reconstructing per-seed values from
Phase 8's recorded mean/min/max. The original Phase 8 bundles are present under
`ml/artifacts/backtest/bundles/`, and the seeds added in this phase land beside them.

THE ASYMMETRY, STATED HERE AND CARRIED INTO EVERY ROW. Only the h4 / head arm was widened to five
seeds. h0 stays at three; the LightGBM baselines keep their five deterministic fits. A capacity
h4-vs-h0 cell needs both arms at five (4 runs) and the ten tightest cells would have cost ~26
runs against a ~3.5 h budget. So where a comparison is decided against a 3-seed h0 band, the row
says so and the verdict is labelled accordingly.

A verdict is DISJOINT only when the two [min, max] bands do not overlap. Anything else is
UNDETERMINED -- not "tied". A tie asserts equality; undetermined asserts nothing, which is what
an overlap actually licenses.
"""
from __future__ import annotations
import os, sys, glob, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "models"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np
import phase5_metrics as M

BUND = "ml/artifacts/backtest/bundles"
PREDS = "ml/artifacts/backtest/preds"


def band(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)),
                spread=float(np.max(v) - np.min(v)), n=len(v), seeds=sorted(v))


def disjoint(a, b):
    return bool(a and b and (a["max"] < b["min"] or b["max"] < a["min"]))


def verdict(a, b, lower_is_better, name_a, name_b):
    if not a or not b:
        return "insufficient"
    if not disjoint(a, b):
        return "UNDETERMINED"
    return name_a if ((a["mean"] < b["mean"]) == lower_is_better) else name_b


def capacity_pinball(path):
    z = np.load(path)
    return M.capacity_scores(z["P"], z["Y"], n_boot=2)["pinball_mean"][0]


def fill_metric(path, which, recal_dir=None):
    """Fill's head must be scored RECALIBRATED, because that is what Phase 9A recorded.

    Phase 9A's head figure for v6 o8 ECE-22 is 0.02404 -- `ece22_recal`. Scoring the bundle's raw
    predictions instead gives 0.09166 and manufactures a 0.067 margin against baselines that ARE
    recalibrated. The first version of this function did exactly that; the numbers disagreed with
    the recorded at-risk margin by three orders of magnitude, which is what exposed it.
    """
    z = np.load(path)
    P = z["P"]
    if recal_dir is not None:
        import loop as LP
        rec = json.load(open(os.path.join(recal_dir, "recalibration.json")))
        P = LP.apply_recalibration(rec, "fill_rate", {"P": P.astype(float)})["P22"]
    s = M.fill_scores(P, z["Y"], "cells22", n_boot=2)
    return s[which][0]


def cap_cell(world, origin):
    """h4 (5 seeds here), h0 (3), B5 (5 deterministic fits) -- evaluation-fold pinball."""
    out = {}
    for arm, pat in (("h4", f"{BUND}/capacity_strain/o{origin}/{world}_mp_h4_*_s*/preds_test.npz"),
                     ("h0", f"{BUND}/capacity_strain/o{origin}/{world}_none_h0_*_s*/preds_test.npz")):
        out[arm] = band([capacity_pinball(p) for p in sorted(glob.glob(pat))])
    out["B5"] = band([capacity_pinball(p) for p in
                      sorted(glob.glob(f"{PREDS}/{world}_capacity_strain_o{origin}_b5flat_q_s*_test.npz"))])
    return out


def fill_cell(world, origin, which):
    """head (5 seeds here) against its recalibrated LightGBM and B2 arms."""
    out = {}
    out["head"] = band([fill_metric(p, which, recal_dir=os.path.dirname(p)) for p in
                        sorted(glob.glob(f"{BUND}/fill_rate/o{origin}/{world}_none_h0_*_s*/preds_test.npz"))])
    for arm, pat in (("lgbm22_id", f"{PREDS}/RECAL_{world}_fill_rate_o{origin}_lgbm22_id_s*_test.npz"),
                     ("b5flat22", f"{PREDS}/RECAL_{world}_fill_rate_o{origin}_b5flat22_s*_test.npz"),
                     ("b2", f"{PREDS}/RECAL_{world}_fill_rate_o{origin}_b2_rolling52_cdf_test.npz")):
        f = sorted(glob.glob(pat))
        if f:
            out[arm] = band([fill_metric(p, which) for p in f])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    R = {"capacity": [], "fill": []}

    CAP = [("v6", 7), ("v7", 3), ("v6", 3), ("v6", 4)]
    FILL = [("v6", 8), ("v6", 7)]

    print("=== capacity: evaluation-fold pinball (lower better) ===")
    for w, o in CAP:
        c = cap_cell(w, o)
        h4, h0, b5 = c["h4"], c["h0"], c["B5"]
        row = dict(world=w, origin=o, h4=h4, h0=h0, B5=b5,
                   v_h4_h0=verdict(h4, h0, True, "h4", "h0"),
                   v_h4_b5=verdict(h4, b5, True, "h4", "B5"),
                   h0_seeds=h0["n"] if h0 else 0)
        R["capacity"].append(row)
        print(f"  {w} o{o}: h4 n={h4['n']} {h4['mean']:.5f} [{h4['min']:.5f},{h4['max']:.5f}] "
              f"| h0 n={h0['n']} {h0['mean']:.5f} [{h0['min']:.5f},{h0['max']:.5f}] "
              f"| B5 n={b5['n']} {b5['mean']:.5f}")
        print(f"        h4 vs h0: {row['v_h4_h0']:13s}  h4 vs B5: {row['v_h4_b5']}")

    print("\n=== fill: evaluation-fold ECE-22 and exact CRPS (lower better) ===")
    for w, o in FILL:
        for which in ("ece22", "crps_exact"):
            c = fill_cell(w, o, which)
            head = c["head"]
            row = dict(world=w, origin=o, metric=which, head=head,
                       arms={k: v for k, v in c.items() if k != "head"},
                       verdicts={k: verdict(head, v, True, "head", k)
                                 for k, v in c.items() if k != "head"})
            R["fill"].append(row)
            print(f"  {w} o{o} {which}: head n={head['n']} {head['mean']:.5f} "
                  f"[{head['min']:.5f},{head['max']:.5f}]")
            for k, v in row["arms"].items():
                print(f"        vs {k:12s} n={v['n']} {v['mean']:.5f} "
                      f"[{v['min']:.5f},{v['max']:.5f}]  -> {row['verdicts'][k]}")
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1, default=str)
        print(f"\njson -> {a.json}")


if __name__ == "__main__":
    main()
