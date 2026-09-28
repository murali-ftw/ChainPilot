"""Phase 13 Stage 1A / 1B / 2A scorer. v8, fixed split.

TABLE A = every arm scored RAW (pre-recalibration). IT DECIDES.
TABLE B = every arm scored after ITS OWN recalibrator (fitted on validation by loop.finish_bundle; b5flat22's stored
          RECAL predictions). Reported for completeness only. Recalibrated VALIDATION interior error / ECE are ~0 by
          construction and are never used for anything. The regression control is N/A in Table B.
Bands: 5 model seeds for trained arms (min-max). Deterministic arms (rolling-52 histogram, F2 arm (a)) have no seeds:
they get a ROW-BOOTSTRAP 95% interval, labelled "bootstrap", never presented as a seed band.
A difference exists only when intervals are DISJOINT; otherwise UNDETERMINED.
"""
from __future__ import annotations
import os, sys, json, glob
import phase12_common as C
import numpy as np, pandas as pd
import phase5_heads as P5, folds, loop as L
import phase5_metrics as M
from heads import fill_cell
from metrics import roc_auc

INTERIOR = np.arange(1, 21)
PR = os.path.join(C.ART, "phase7_preds")
BF = os.path.join(C.BUND, "fill_rate")


def metrics(P, y, point=None, rows=None):
    P = np.asarray(P, float); y = np.asarray(y, float)
    if rows is not None:
        P, y = P[rows], y[rows]; point = point[rows] if point is not None else None
    cells = fill_cell(y)
    pred, obs = P.mean(0), np.bincount(cells, minlength=22) / len(y)
    err = pred - obs
    crps = np.abs(point - y) if point is not None else M.crps_exact_rows(P, y)   # a step CDF's exact CRPS is |y_hat - y|
    s = point if point is not None else P[:, 21]
    o1 = (y >= 1).astype(int)
    top = np.argsort(-np.abs(err[INTERIOR]))[:4] + 1
    return dict(n=int(len(y)), crps_exact=float(crps.mean()), interior_abs_err=float(np.abs(err[INTERIOR]).sum()),
                ece22=float(np.abs(err).sum()), roc_auc_p_complete=float(roc_auc(o1, s)) if 0 < o1.mean() < 1 else float("nan"),
                top4_cells=[int(c) for c in top], top4_share=float(np.abs(err[top]).sum() / max(np.abs(err[INTERIOR]).sum(), 1e-12)),
                cell20_pred=float(pred[20]), cell20_obs=float(obs[20]), cell20_ratio=float(pred[20] / max(obs[20], 1e-12)),
                atom1_pred=float(pred[21]), atom1_obs=float(obs[21]), _crps_rows=crps)


def seed_band(per, keys=("crps_exact", "interior_abs_err", "ece22", "roc_auc_p_complete")):
    out = dict(n_seeds=len(per), kind="5-seed [min, max]" if len(per) >= 5 else f"{len(per)}-seed")
    for k in keys:
        v = [p[k] for p in per]
        out[k] = [float(min(v)), float(np.mean(v)), float(max(v))]
    for k in ("top4_cells", "top4_share", "cell20_ratio", "cell20_pred", "cell20_obs", "atom1_pred", "atom1_obs"):
        out[k] = [p[k] for p in per]
    return out


def boot_band(m, P, y, point=None, B=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y); cr = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        cr.append(metrics(P, y, point, rows=i))
    out = dict(n_seeds=0, kind="row-bootstrap 95% (deterministic arm, no seeds)")
    for k in ("crps_exact", "interior_abs_err", "ece22", "roc_auc_p_complete"):
        v = np.array([c[k] for c in cr])
        out[k] = [float(np.percentile(v, 2.5)), float(m[k]), float(np.percentile(v, 97.5))]
    for k in ("top4_cells", "top4_share", "cell20_ratio", "cell20_pred", "cell20_obs", "atom1_pred", "atom1_obs"):
        out[k] = [m[k]]
    return out


def bundle_arm(pattern, split, recal):
    per = []
    for d in sorted(glob.glob(os.path.join(BF, pattern))):
        if not os.path.exists(os.path.join(d, f"preds_{split}.npz")):
            continue
        z = dict(np.load(os.path.join(d, f"preds_{split}.npz")))
        cfg = json.load(open(os.path.join(d, "config.json")))
        assert "+dirty" not in cfg["stamps"]["model_version"], f"{d} is stamped +dirty"
        if "Ppt" in z:
            if recal:
                return None                                      # N/A: a point forecast is not recalibrated
            per.append(metrics(z["P"], z["Y"], point=z["Ppt"]))
            continue
        P = z["P"]
        if recal:
            rec = json.load(open(os.path.join(d, "recalibration.json")))
            P = L.apply_recalibration(rec, "fill_rate", z)["P22"]
        per.append(metrics(P, z["Y"]))
    return seed_band(per) if per else None


def stored_arm(stem, split, recal):
    per = []
    for s in C.V8_SEEDS:
        f = os.path.join(PR, f"{'RECAL_' if recal else ''}v8_fill_rate_{stem}_s{s}_{split}.npz")
        z = np.load(f); per.append(metrics(z["P"], z["Y"]))
    return seed_band(per)


def hist_arm(split):
    z = np.load(os.path.join(PR, f"v8_fill_rate_b2_rolling52_cdf_{split}.npz"))
    return boot_band(metrics(z["P"], z["Y"]), z["P"], z["Y"])


def disjoint_below(a, b, k):
    return bool(a and b and a[k][2] < b[k][0])


def f1_tables(bw):
    arms = {"1_cells22_head": "v8_none_h0_lr0.000125_s[0-9]*", "3_regression_control": "v8_none_h0_lr0.000125_s*_headreg",
            "4_beta3": "v8_none_h0_lr0.000125_s*_headbeta3", f"5_boundary_{bw}": f"v8_none_h0_lr0.000125_s*_loss{bw}"}
    T = {"A_raw": {}, "B_recal": {}}
    for tab, recal in (("A_raw", False), ("B_recal", True)):
        for split in ("val", "test"):
            t = {}
            for name, pat in arms.items():
                if name == "1_cells22_head":
                    per = []
                    for s in C.V8_SEEDS:
                        d = os.path.join(BF, f"v8_none_h0_lr0.000125_s{s}")
                        z = dict(np.load(os.path.join(d, f"preds_{split}.npz"))); P = z["P"]
                        if recal:
                            P = L.apply_recalibration(json.load(open(os.path.join(d, "recalibration.json"))), "fill_rate", z)["P22"]
                        per.append(metrics(P, z["Y"]))
                    t[name] = seed_band(per)
                else:
                    t[name] = bundle_arm(pat, split, recal) if pat else None
                    if t[name] is None and name.startswith("3_") and recal:
                        t[name] = "N/A: a point forecast cannot be recalibrated"
            t["2_b5flat22"] = stored_arm("b5flat22", split, recal)
            t["6_rolling52_histogram"] = hist_arm(split) if not recal else "N/A: stored raw only (deterministic, not recalibrated)"
            T[tab][split] = t
    # verdicts, TABLE A, TEST, against arm 1's same-split band
    A = T["A_raw"]["test"]; head = A["1_cells22_head"]
    v = {}
    for name, a in A.items():
        if not isinstance(a, dict) or name == "1_cells22_head":
            continue
        interior_win = disjoint_below(a, head, "interior_abs_err")
        crps_not_worse = not disjoint_below(head, a, "crps_exact")
        roc_not_degraded = not (a["roc_auc_p_complete"][2] < head["roc_auc_p_complete"][0])
        still_same4 = sorted(set(a["top4_cells"][0])) == sorted(set(head["top4_cells"][0]))
        v[name] = dict(interior_disjointly_below_head=interior_win, crps_not_worse=crps_not_worse,
                       roc_not_degraded=roc_not_degraded, same_four_cells_dominate=still_same4,
                       verdict=("PASS" if interior_win and crps_not_worse and roc_not_degraded else
                                "FAIL" if disjoint_below(head, a, "interior_abs_err") or not crps_not_worse else "UNDETERMINED"))
    T["verdicts_test_tableA"] = v
    return T


def f2(lb_full):
    """Arms (a) ratio alone (deterministic point forecast) for every key x estimator, and (c) ratio as head input."""
    import fill_history as FHm
    tr, va, te = folds.fixed_split(lb_full.snapshot_date)
    out = {"a": {}, "c_A_raw": {}, "c_B_recal": {}, "per_tier": {}, "cold_start": {}}
    for split, m in (("val", va), ("test", te)):
        o = P5.ordered(lb_full, m)
        y = lb_full.label_value.to_numpy(float)[o]
        for key, est in (("ps", "ros"), ("ps", "mor"), ("sp", "ros"), ("sp", "mor"), ("psp", "ros"), ("psp", "mor"),
                         ("hier", "ros")):
            f = FHm.attach(lb_full[["snapshot_date", "key", "label_value"]], key, est)
            r = np.clip(f.ratio_x0.to_numpy(float)[o], 0, 1)
            P = np.eye(22)[fill_cell(r)]
            out["a"].setdefault(split, {})[f"{key}_{est}"] = boot_band(metrics(P, y, point=r), P, y, point=r)
        out["a"][split]["rolling52_histogram"] = hist_arm(split)
        for key in ("hier", "ps", "sp", "psp"):
            tag = f"{key}_ros_{'shrink' if key == 'hier' else 'raw'}_useinput"
            pat = f"v8_none_h0_lr0.000125_s*_rk{tag}"
            out["c_A_raw"].setdefault(split, {})[key] = bundle_arm(pat, split, False)
            out["c_B_recal"].setdefault(split, {})[key] = bundle_arm(pat, split, True)
            f = FHm.attach(lb_full[["snapshot_date", "key", "label_value"]], key, "ros")
            tier = f.ratio_tier.to_numpy()[o]; own = f.ratio_x1.to_numpy()[o] > 0
            ds = sorted(glob.glob(os.path.join(BF, pat)))
            if not ds:
                continue
            for group, mask in [(f"tier={t}", tier == t) for t in np.unique(tier)] + [("cold_start(no own history)", ~own)]:
                per = []
                for d in ds:
                    z = dict(np.load(os.path.join(d, f"preds_{split}.npz")))
                    if mask.sum() >= 1:
                        per.append(metrics(z["P"], z["Y"], rows=np.flatnonzero(mask)))
                if per:
                    b = seed_band(per)
                    b["row_share"] = float(mask.mean()); b["rows"] = int(mask.sum())
                    tgt = out["cold_start"] if group.startswith("cold") else out["per_tier"]
                    tgt.setdefault(split, {}).setdefault(key, {})[group] = b
    # the head (no ratio) on the same cold-start / tier rows, for comparison
    return out


def main():
    st = C.require_clean()
    R = dict(stamp=st)
    sel = os.path.join(C.ART, "phase13_f1_bw_selection.json")
    bw = json.load(open(sel))["selected_fill_loss"] if os.path.exists(sel) else "rps_bw10"
    R["F1"] = f1_tables(bw); R["F1"]["boundary_arm"] = bw
    lb = P5.labels("v8", "fill_rate")
    R["F2"] = f2(lb)
    strip = lambda o: {k: strip(v) for k, v in o.items() if not k.startswith("_")} if isinstance(o, dict) else o
    print(json.dumps(strip(R["F1"]["verdicts_test_tableA"]), indent=1))
    print(C.dump(strip(R), "phase13_scores.json"))


if __name__ == "__main__":
    main()
