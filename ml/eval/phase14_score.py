"""Phase 14 -- decision-level scoring from STORED predictions: UC1, UC1b, UC2, UC2b, UC3, UC4.

Decisions, positive classes, scores, censoring treatments and threshold objectives are those pre-registered in
reports/part2/phase-14-decisions.md (commit 9283aac). Nothing is trained. Files are read by EXACT per-seed names.

Threshold: fitted on VALIDATION, maximising F1 (primary) or MCC (pre-registered secondary); predicted positive iff
score >= tau; among tied objective values the LARGEST tau is taken. Applied unchanged to TEST.
RAW and RECALIBRATED are separate tables, each with its own validation-fitted tau.
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import phase5_heads as P5, folds, loop as L
from phase11b_lateness import build_reference

BA, BF, BC, BS = (os.path.join(C.BUND, t) for t in ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty"))
PR = os.path.join(C.ART, "phase7_preds")
SEEDS = C.V8_SEEDS


# ------------------------------------------------------------------ decision machinery
def fit_tau(s, y, objective="f1"):
    s = np.asarray(s, float); y = np.asarray(y, int)
    order = np.argsort(-s, kind="stable"); ss, yy = s[order], y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); P = yy.sum(); N = len(yy) - P
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]          # cut after each distinct score
    tp, fp, tau = tp[last], fp[last], ss[last]
    fn, tn = P - tp, N - fp
    if objective == "f1":
        obj = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    else:
        den = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
        obj = (tp * tn - fp * fn) / den
    # "predict nothing positive" is also a candidate (obj 0 for F1 and MCC)
    best = np.flatnonzero(obj == obj.max())
    i = best[np.argmax(tau[best])]
    return (float(tau[i]) if obj.max() > 0 else float("inf")), float(obj[i])


def decide(s, y, tau):
    s = np.asarray(s, float); y = np.asarray(y, int)
    p = s >= tau
    TP = int((p & (y == 1)).sum()); FP = int((p & (y == 0)).sum()); TN = int((~p & (y == 0)).sum()); FN = int((~p & (y == 1)).sum())
    n = len(y); pos = TP + FN
    acc = (TP + TN) / n; prec = TP / (TP + FP) if TP + FP else float("nan"); rec = TP / pos if pos else float("nan")
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    spec = TN / (TN + FP) if TN + FP else float("nan")
    den = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / den if den else 0.0
    base = pos / n
    maj_acc = max(base, 1 - base)
    maj_f1 = 2 * base / (1 + base) if base >= 0.5 else 0.0
    u = len(np.unique(s)) > 1 and 0 < pos < n
    return dict(n=n, positives=pos, base_rate=base, TP=TP, FP=FP, TN=TN, FN=FN, accuracy=acc, majority_accuracy=maj_acc,
                accuracy_gain_over_majority=acc - maj_acc, precision=prec, recall=rec, f1=f1, majority_f1=maj_f1,
                specificity=spec, mcc=mcc, balanced_accuracy=(rec + spec) / 2 if pos and n - pos else float("nan"),
                roc_auc=float(roc_auc_score(y, s)) if u else float("nan"),
                pr_auc=float(average_precision_score(y, s)) if u else float(base))


KEYS = ("accuracy", "majority_accuracy", "accuracy_gain_over_majority", "precision", "recall", "f1", "majority_f1",
        "specificity", "mcc", "balanced_accuracy", "roc_auc", "pr_auc", "TP", "FP", "TN", "FN", "n", "positives", "base_rate", "tau")


def run_arm(pairs, deterministic=False, B=500, seed=0, fixed_tau=None):
    """pairs: list over seeds of (s_val, y_val, s_test, y_test). -> {objective: banded test metrics + val tau}."""
    out = {}
    for obj in ("f1", "mcc"):
        per = []
        for sv, yv, st, yt in pairs:
            # a CONSTANT baseline takes a FIXED tau: fitting on a constant score always selects "all positive",
            # which would silently turn "always negative" into "always positive"
            tau, vobj = (fixed_tau, float("nan")) if fixed_tau is not None else fit_tau(sv, yv, obj)
            m = decide(st, yt, tau); m["tau"] = tau; m["val_objective"] = vobj
            per.append(m)
        if deterministic:
            sv, yv, st, yt = pairs[0]
            tau = per[0]["tau"]; rng = np.random.default_rng(seed); bs = []
            for _ in range(B):
                i = rng.integers(0, len(yt), len(yt)); bs.append(decide(st[i], yt[i], tau))
            band = {k: [float(np.nanpercentile([b[k] for b in bs], 2.5)), float(per[0][k]), float(np.nanpercentile([b[k] for b in bs], 97.5))]
                    for k in KEYS if k != "tau"}
            band["tau"] = [tau, tau, tau]; band["kind"] = "row bootstrap 95% (deterministic arm)"; band["n_seeds"] = 0
        else:
            band = {k: [float(np.nanmin([p[k] for p in per])), float(np.nanmean([p[k] for p in per])),
                        float(np.nanmax([p[k] for p in per]))] for k in KEYS}
            band["kind"] = f"{len(per)}-seed [min, mean, max]"; band["n_seeds"] = len(per)
        band["confusion_seed0"] = {k: per[0][k] for k in ("TP", "FP", "TN", "FN")}
        out[obj] = band
    return out


def const_pairs(yv, yt, value):
    return [(np.zeros(len(yv)), yv, np.zeros(len(yt)), yt)]


def const_arm(yv, yt, positive):
    """always-positive (positive=True) or always-negative, with a FIXED threshold."""
    return run_arm(const_pairs(yv, yt, 0), deterministic=True, fixed_tau=(-np.inf if positive else np.inf))


def majority_arm(yv, yt):
    return const_arm(yv, yt, positive=bool(yv.mean() >= 0.5))


# ------------------------------------------------------------------ UC1 / UC1b arrival
def surv_full(S):
    return np.concatenate([np.ones((len(S), 1)), np.asarray(S, float)], 1)      # [:, k] = P(T > k), k = 0..12


def p_late(S, R):
    Sf = surv_full(S); k = np.clip(np.floor(R).astype(int), 0, 12)
    return Sf[np.arange(len(R)), k]


def uc1():
    lb = P5.labels("v8", "arrival_week")
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    refs = {"contracted_(b)_primary": build_reference("contracted_lead", "v8", lb, tr)[0],
            "adopted_asof_(a)_secondary": build_reference("asof_channel_lead", "v8", lb, tr)[0]}
    ov, ot = P5.ordered(lb, va), P5.ordered(lb, te)
    arms = {}
    for s in SEEDS:
        for name, d in (("h4", f"{BA}/v8_lite_h4_lr0.00025_s{s}"), ("h0", f"{BA}/v8_none_h0_lr0.00025_s{s}")):
            rec = json.load(open(f"{d}/recalibration.json"))
            for split in ("val", "test"):
                z = dict(np.load(f"{d}/preds_{split}.npz"))
                arms.setdefault((name, "raw"), {}).setdefault(s, {})[split] = dict(kind="dist", S=z["S"], Y=z["Y"], EV=z["EV"])
                Q = L.apply_recalibration(rec, "arrival_week", z)
                arms.setdefault((name, "recal"), {}).setdefault(s, {})[split] = dict(kind="dist", S=Q["S"], Y=z["Y"], EV=z["EV"])
        for name, stem in (("b5flat_reg", "b5flat_reg"), ("lgbm_id_reg", "lgbm_id_reg")):
            for split in ("val", "test"):
                z = dict(np.load(f"{PR}/v8_arrival_week_{stem}_s{s}_{split}.npz"))
                arms.setdefault((name, "raw"), {}).setdefault(s, {})[split] = dict(kind="point", P=z["P"], Y=z["Y"], EV=z["EV"])
    res = {"UC1": {}, "UC1b": {}}
    for rname, R in refs.items():
        Rv, Rt = R[ov], R[ot]
        for treat in ("A_excluded", "B_censored_as_not_late", "C_resolved"):
            T = res["UC1"].setdefault(rname, {}).setdefault(treat, {})

            def lab(d, Rx):
                ev = np.asarray(d["EV"], bool); y = np.asarray(d["Y"], float)
                late = ev & (y > Rx)
                if treat == "A_excluded":
                    keep = ev
                elif treat == "B_censored_as_not_late":
                    keep = np.ones(len(y), bool)
                else:
                    late = late | (~ev & (Rx < 13)); keep = ev | (Rx < 13)
                return late.astype(int), keep
            for (name, cal), per in arms.items():
                pairs = []
                for s, dd in per.items():
                    yv, kv = lab(dd["val"], Rv); yt, kt = lab(dd["test"], Rt)
                    sc = (lambda d, Rx: p_late(d["S"], Rx)) if dd["val"]["kind"] == "dist" else (lambda d, Rx: np.asarray(d["P"], float) - Rx)
                    pairs.append((sc(dd["val"], Rv)[kv], yv[kv], sc(dd["test"], Rt)[kt], yt[kt]))
                T[f"{name}|{cal}"] = run_arm(pairs)
            d0 = arms[("h4", "raw")][7]
            yv, kv = lab(d0["val"], Rv); yt, kt = lab(d0["test"], Rt)
            T["always_late|-"] = const_arm(yv[kv], yt[kt], True)
            T["always_not_late|-"] = const_arm(yv[kv], yt[kt], False)
            T["channel_constant_ranker|-"] = run_arm([(-Rv[kv], yv[kv], -Rt[kt], yt[kt])], deterministic=True)
            T["_censoring"] = dict(test_rows=int(len(Rt)), test_censored=int((~d0["test"]["EV"].astype(bool)).sum()),
                                   test_rows_scored=int(kt.sum()), test_censored_resolved_late=int(((~d0["test"]["EV"].astype(bool)) & (Rt < 13)).sum()))
            print("UC1", rname, treat, {k: round(v["f1"]["f1"][1], 4) for k, v in T.items() if not k.startswith("_")}, flush=True)
    for H in (1, 2, 4):
        T = res["UC1b"].setdefault(f"H={H}", {})
        for (name, cal), per in arms.items():
            pairs = []
            for s, dd in per.items():
                lab = lambda d: (np.asarray(d["EV"], bool) & (np.asarray(d["Y"], float) <= H)).astype(int)
                sc = (lambda d: 1 - surv_full(d["S"])[:, H]) if dd["val"]["kind"] == "dist" else (lambda d: -np.asarray(d["P"], float))
                pairs.append((sc(dd["val"]), lab(dd["val"]), sc(dd["test"]), lab(dd["test"])))
            T[f"{name}|{cal}"] = run_arm(pairs)
        d0 = arms[("h4", "raw")][7]
        yv = (d0["val"]["EV"].astype(bool) & (d0["val"]["Y"] <= H)).astype(int); yt = (d0["test"]["EV"].astype(bool) & (d0["test"]["Y"] <= H)).astype(int)
        T["always_majority|-"] = majority_arm(yv, yt)
        print("UC1b", H, {k: round(v["f1"]["f1"][1], 4) for k, v in T.items()}, flush=True)
    return res


# ------------------------------------------------------------------ UC2 / UC2b fill
def uc2():
    res = {"UC2": {}, "UC2b_lt0.95": {}, "UC2b_lt0.75": {}}
    dec = {"UC2": (lambda P: P[:, 21], lambda y: (y >= 1)),
           "UC2b_lt0.95": (lambda P: P[:, :20].sum(1), lambda y: (y < 0.95)),
           "UC2b_lt0.75": (lambda P: P[:, :16].sum(1), lambda y: (y < 0.75))}
    bundle_arms = {"head_cells22": "", "boundary_bw3": "_lossrps_bw3", "beta3": "_headbeta3"}
    for uc, (sf, lf) in dec.items():
        T = res[uc]
        for arm, suf in bundle_arms.items():
            for cal in ("raw", "recal"):
                pairs = []
                for s in SEEDS:
                    d = f"{BF}/v8_none_h0_lr0.000125_s{s}{suf}"
                    rec = json.load(open(f"{d}/recalibration.json"))
                    zz = {}
                    for split in ("val", "test"):
                        z = dict(np.load(f"{d}/preds_{split}.npz"))
                        P = z["P"] if cal == "raw" else L.apply_recalibration(rec, "fill_rate", z)["P22"]
                        zz[split] = (sf(np.asarray(P, float)), lf(z["Y"]).astype(int))
                    pairs.append((*zz["val"], *zz["test"]))
                T[f"{arm}|{cal}"] = run_arm(pairs)
        for arm in ("b5flat22", "lgbm22_id"):
            for cal, pre in (("raw", ""), ("recal", "RECAL_")):
                pairs = []
                for s in SEEDS:
                    zz = {}
                    for split in ("val", "test"):
                        z = np.load(f"{PR}/{pre}v8_fill_rate_{arm}_s{s}_{split}.npz")
                        zz[split] = (sf(z["P"]), lf(z["Y"]).astype(int))
                    pairs.append((*zz["val"], *zz["test"]))
                T[f"{arm}|{cal}"] = run_arm(pairs)
        zz = {}
        for split in ("val", "test"):
            z = np.load(f"{PR}/v8_fill_rate_b2_rolling52_cdf_{split}.npz"); zz[split] = (sf(z["P"]), lf(z["Y"]).astype(int))
        T["rolling52_histogram|raw"] = run_arm([(*zz["val"], *zz["test"])], deterministic=True)
        yv, yt = zz["val"][1], zz["test"][1]
        T["always_majority|-"] = majority_arm(yv, yt)
        print(uc, {k: round(v["f1"]["f1"][1], 4) for k, v in T.items()}, flush=True)
    return res


# ------------------------------------------------------------------ UC3 capacity
def p_exceed(Q, x=1.0):
    """P(strain > x) from a piecewise-linear CDF through (P10,.1),(P50,.5),(P90,.9); tails extrapolated, clipped."""
    q = np.asarray(Q, float); q10, q50, q90 = q[:, 0], q[:, 1], np.maximum(q[:, 2], q[:, 1] + 1e-9)
    q10 = np.minimum(q10, q50 - 1e-9)
    F = np.where(x <= q50, 0.1 + 0.4 * (x - q10) / (q50 - q10), 0.5 + 0.4 * (x - q50) / (q90 - q50))
    return 1 - np.clip(F, 0, 1)


def uc3():
    T = {}
    for arm, pat in (("mp_h4", "v8_mp_h4_lr0.00025_s{s}"), ("h0", "v8_none_h0_lr0.00025_s{s}")):
        pairs = []
        for s in SEEDS:
            zz = {sp: np.load(f"{BC}/{pat.format(s=s)}/preds_{sp}.npz") for sp in ("val", "test")}
            pairs.append(tuple(x for sp in ("val", "test") for x in (p_exceed(zz[sp]["P"]), (zz[sp]["Y"] > 1).astype(int))))
        T[f"{arm}|raw"] = run_arm(pairs)
    pairs = []
    for s in SEEDS:
        zz = {sp: np.load(f"{PR}/v8_capacity_strain_b5flat_q_s{s}_{sp}.npz") for sp in ("val", "test")}
        pairs.append(tuple(x for sp in ("val", "test") for x in (p_exceed(zz[sp]["P"]), (zz[sp]["Y"] > 1).astype(int))))
    T["b5flat_q|raw"] = run_arm(pairs)
    for nm in ("naive_channel", "naive_supplier"):
        zz = {sp: np.load(f"{PR}/v8_capacity_strain_{nm}_{sp}.npz") for sp in ("val", "test")}
        T[f"{nm}|raw"] = run_arm([tuple(x for sp in ("val", "test") for x in (p_exceed(zz[sp]["P"]), (zz[sp]["Y"] > 1).astype(int)))], deterministic=True)
    yv = (np.load(f"{BC}/v8_mp_h4_lr0.00025_s7/preds_val.npz")["Y"] > 1).astype(int)
    yt = (np.load(f"{BC}/v8_mp_h4_lr0.00025_s7/preds_test.npz")["Y"] > 1).astype(int)
    T["always_majority|-"] = majority_arm(yv, yt)
    # WHY it scores as it does: the median's level lag (Phase 12 B1), measured here on the decision's own terms
    z = np.load(f"{BC}/v8_mp_h4_lr0.00025_s7/preds_test.npz"); zv = np.load(f"{BC}/v8_mp_h4_lr0.00025_s7/preds_val.npz")
    T["_level"] = dict(test_mean_label=float(z["Y"].mean()), test_mean_p50=float(z["P"][:, 1].mean()),
                       val_mean_label=float(zv["Y"].mean()), val_mean_p50=float(zv["P"][:, 1].mean()),
                       test_share_label_gt1=float((z["Y"] > 1).mean()), test_share_p50_gt1=float((z["P"][:, 1] > 1).mean()),
                       val_share_label_gt1=float((zv["Y"] > 1).mean()), val_share_p50_gt1=float((zv["P"][:, 1] > 1).mean()))
    print("UC3", {k: round(v["f1"]["f1"][1], 4) for k, v in T.items() if not k.startswith("_")}, T["_level"], flush=True)
    return {"UC3": T}


# ------------------------------------------------------------------ UC4 shortage head
def uc4():
    T = {}
    for arm, pat, seeds in (("mp_h1_shipped", "v8_mp_h1_lr0.000125_s{s}", SEEDS), ("h0_3seeds_NOT_QUOTABLE", "v8_none_h0_lr0.000125_s{s}", (7, 17, 27))):
        pairs = []
        for s in seeds:
            zz = {sp: np.load(f"{BS}/{pat.format(s=s)}/preds_{sp}.npz") for sp in ("val", "test")}
            pairs.append(tuple(x for sp in ("val", "test") for x in (zz[sp]["P"], zz[sp]["Y"].astype(int))))
        T[f"{arm}|raw"] = run_arm(pairs)
    pairs = []
    for s in SEEDS:
        zz = {sp: np.load(f"{PR}/v8_shortage_qty_b5flat_bin_s{s}_{sp}.npz") for sp in ("val", "test")}
        pairs.append(tuple(x for sp in ("val", "test") for x in (zz[sp]["P"], zz[sp]["Y"].astype(int))))
    T["b5flat_bin|raw"] = run_arm(pairs)
    zz = {sp: np.load(f"{PR}/v8_shortage_qty_naive_part_plant_rate_{sp}.npz") for sp in ("val", "test")}
    T["naive_part_plant_rate|raw"] = run_arm([tuple(x for sp in ("val", "test") for x in (zz[sp]["P"], zz[sp]["Y"].astype(int)))], deterministic=True)
    yv, yt = zz["val"]["Y"].astype(int), zz["test"]["Y"].astype(int)
    T["always_majority|-"] = majority_arm(yv, yt)
    print("UC4", {k: round(v["f1"]["f1"][1], 4) for k, v in T.items()}, flush=True)
    return {"UC4": T}


def main():
    st = C.require_clean()
    R = dict(stamp=st)
    R.update(uc1()); R.update(uc2()); R.update(uc3()); R.update(uc4())
    print(C.dump(R, "phase14_scores.json"))


if __name__ == "__main__":
    main()
