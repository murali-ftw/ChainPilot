"""Phase 22 Stage 2 -- the order-time arrival product (pre-registration D6). Torch process (imports only), no LightGBM.

Unit: a PO line scored on the day it is raised (Phase 21 at-placement rows; tau = the Monday on or before creation).
Every feature is leak-free (world v8clean). Absolute numbers are not compared with the snapshot arms.

Date estimators (expected lead in weeks): BASE_clean, + L4, latedays (LightGBM on lateness-in-days, back-transformed),
the standalone shrunk-KM median (+ the validation offset a), and the blend w * KM + (1 - w) * latedays (w on a 0.05 grid by
VALIDATION A3). The product's date = the estimator with the lowest VALIDATION A3.
Late flag: the LightGBM binary flag's P(late) (RAW) and isotonic-recalibrated on validation (RECALIBRATED), reported apart;
also every date estimator's (pred - contract) as a ranking score.
Interval: split-conformal on VALIDATION signed-day residuals (7 * (Y - pred)) of the product's date, central 80%
(q10, q90). Month-specific quantiles only if the validation per-creation-month coverage of the pooled interval spreads by
> 5 points (D6). Coverage and width overall and PER CREATION MONTH on test.
Gate (D5): arm X PASSES vs BASE_clean if disjoint on A3 or lateness AUC (5-seed bands), none disjointly worse, and the
cross-channel shuffled L4 control is not disjointly better. Self-test of the gate on constructed cases (equal bands -> FAIL;
control also better -> FAIL). Leak gate: every arm's feature list through phase22_proxy.leak_check (recorded in the fit
logs); the `leaked` arm MUST be flagged, no clean arm may be.

  python ml/eval/phase22_order_time.py [--world v8clean]    -> ml/artifacts/phase22/order_time_{world}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config, folds
import phase21_paths as PP
import grpstats as GS
import phase5_metrics as M
from metrics import roc_auc
from phase11b_lateness import lateness
import phase18_score as S18
import phase20_decisions as P20D
import phase21_score as S21

SEEDS = C.V8_SEEDS
B = 1000
PR22 = os.path.join(C.ART, "phase22", "preds")
LOGS = os.path.join(C.ART, "phase22", "logs")
K = 10
DIR = {"lateness_auc": True, "a3_median_abs_err_days": False, "week_hit_rate": True}


def fmt(world, arm, s, f):
    k = f"_k{K}" if arm in ("L4", "L4_xsh", "latedays", "flag", "leaked", "flag_lag1", "L4_lag1") else ""
    return os.path.join(PR22, f"{world}_arrival_place_p22_{arm}{k}_s{s}_{f}.npz")


def load(world, arm, s):
    return {f: dict(np.load(fmt(world, arm, s, f))) for f in ("val", "test")}


def have(world, arm):
    return all(os.path.exists(fmt(world, arm, s, f)) for s in SEEDS for f in ("val", "test"))


def date_metrics(P, Y, EV, R):
    return dict(lateness_auc=lateness(P, Y, EV, R)[0], a3_median_abs_err_days=float(np.median(np.abs(7 * P[EV] - 7 * Y[EV]))),
                week_hit_rate=float((np.abs(P[EV] - Y[EV]) <= 0.5).mean()))


def band_of(world, arm, fold="test"):
    per = []
    for s in SEEDS:
        z = load(world, arm, s)[fold]
        P, Y, EV, R = (np.asarray(z[k], float) for k in ("P", "Y", "EV", "AUX")); EV = EV.astype(bool)
        if arm in ("flag", "flag_lag1"):
            yl, keep = S21.uc1p_label(Y, EV, R)
            per.append(dict(lateness_auc=float(roc_auc((Y[EV] > R[EV]).astype(int), P[EV]))))
        else:
            per.append(date_metrics(P, Y, EV, R))
    return {k: S18.band([p[k] for p in per]) for k in per[0]}, per


def gate(fam, base, ctrl):
    c = {k: S18.compare(fam.get(k), base.get(k), h) for k, h in DIR.items() if k in fam and k in base}
    better = [k for k, v in c.items() if v == "better" and k != "week_hit_rate"]
    worse = [k for k, v in c.items() if v == "worse" and k != "week_hit_rate"]
    cb = [k for k, v in ({k: S18.compare(ctrl.get(k), base.get(k), h) for k, h in DIR.items() if k in ctrl and k in base} if ctrl else {}).items()
          if v == "better" and k != "week_hit_rate"]
    ok = bool(better) and not worse and not cb
    return dict(verdict="PASS" if ok else "FAIL", better=better, worse=worse, control_better=cb, vs_base=c)


def gate_selftest():
    b = {"lateness_auc": [0.69, 0.695, 0.70], "a3_median_abs_err_days": [12.6, 12.65, 12.7]}
    same = gate(dict(b), b, None)["verdict"]
    better = {"lateness_auc": [0.71, 0.715, 0.72], "a3_median_abs_err_days": [11.0, 11.1, 11.2]}
    ctrl_also = gate(better, b, better)["verdict"]
    assert same == "FAIL" and ctrl_also == "FAIL", "the gate cannot fail -- invalid"
    return dict(equal_bands="FAIL (as required)", control_also_better="FAIL (as required)")


def ens(world, arm, f):
    zs = [load(world, arm, s)[f] for s in SEEDS]
    return np.mean([np.asarray(z["P"], float) for z in zs], 0), zs[0]


def block_det(preds, Y, EV, R, blocks, seed, pairs):
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(B)]
    full = np.arange(len(Y))
    fns = {"lateness_auc": lambda p, i: roc_auc((Y[i][EV[i]] > R[i][EV[i]]).astype(int), p[i][EV[i]] - R[i][EV[i]]),
           "a3_median_abs_err_days": lambda p, i: float(np.median(np.abs(7 * p[i][EV[i]] - 7 * Y[i][EV[i]])))}
    out = dict(point={}, ci={}, diff={})
    for m, fn in fns.items():
        bs = {a: np.array([fn(p, i) for i in draws]) for a, p in preds.items()}
        out["point"][m] = {a: float(fn(p, full)) for a, p in preds.items()}
        out["ci"][m] = {a: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for a, v in bs.items()}
        for a, b in pairs:
            d = (bs[a] - bs[b]) if DIR[m] else (bs[b] - bs[a])
            lo, up = np.percentile(d, [2.5, 97.5])
            out["diff"].setdefault(m, {})[f"{a} vs {b}"] = dict(better_positive=[float(lo), float(d.mean()), float(up)],
                                                                verdict="better" if lo > 0 else "worse" if up < 0 else "undetermined")
    return out


def isotonic(pv, yv):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yv)


def calib(p, y, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    ece = sum(abs(p[q == b].mean() - y[q == b].mean()) * (q == b).mean() for b in range(bins) if (q == b).any())
    return dict(ece10=float(ece), brier=float(np.mean((p - y) ** 2)), mean_p=float(p.mean()), rate=float(y.mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="v8clean"); a = ap.parse_args()
    PP.register()
    for p in ("v8", "v8w1002"):
        config.WORLDS[p + "clean"] = config.WORLDS[p]
    st = C.require_clean()
    world = a.world; pw = world.replace("clean", "")
    out = dict(stamp=st, world=world, gate_selftest=gate_selftest())
    # ---- rows and the standalone KM date (store and k: Phase 21 on v8; this phase's on world 2)
    store_dir = os.path.join(C.ART, "phase21") if pw == "v8" else os.path.join(C.ART, "phase22")
    GS.OUT = store_dir
    Z, _ = GS.load(pw, "place")
    src = GS.Source(pw)
    li = Z["line"].astype(np.int64)
    cr = pd.DatetimeIndex(src.created[li]); tau = pd.to_datetime(Z["tau"]).values
    tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(cr)]
    idx = {f: np.flatnonzero(m)[np.argsort(tau[np.flatnonzero(m)], kind="stable")] for f, m in (("val", va), ("test", te))}
    z0 = load(world, "base", 7)
    for f in ("val", "test"):
        assert (np.asarray(z0[f]["entity"]).astype(str) == Z["entity"][idx[f]]).all(), f"store rows != fitted rows ({f})"
    Yf = {f: np.asarray(z0[f]["Y"], float) for f in ("val", "test")}; EVf = {f: np.asarray(z0[f]["EV"], bool) for f in ("val", "test")}
    Rf = {f: np.asarray(z0[f]["AUX"], float) for f in ("val", "test")}
    sub = lambda f: {k: v[idx[f]] for k, v in Z.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(li)}
    km_raw = {f: GS.standalone_arrival_weeks(sub(f), K) for f in ("val", "test")}
    k_note = "k = 10 (Phase 21, validation)"
    if pw != "v8":                                         # world 2: k re-chosen on ITS validation by the Phase 21 rule
        best = max(GS.K_GRID, key=lambda k: lateness(GS.standalone_arrival_weeks(sub("val"), k), Yf["val"], EVf["val"], Rf["val"])[0])
        km_raw = {f: GS.standalone_arrival_weeks(sub(f), best) for f in ("val", "test")}; k_note = f"k = {best} (world 2 validation)"
    a_off = float(np.median(Yf["val"][EVf["val"]] - km_raw["val"][EVf["val"]]))
    km = {f: km_raw[f] + a_off for f in ("val", "test")}
    # ---- 5-seed arms
    arms = [x for x in ("base", "base_lag1", "L4", "L4_xsh", "latedays", "flag", "leaked", "flag_lag1", "L4_lag1") if have(world, x)]
    bands = {x: band_of(world, x) for x in arms}
    out["bands"] = {x: b for x, (b, _) in bands.items()}
    out["gates"] = {}
    for fam in ("L4", "latedays", "flag"):
        if fam in bands and "base" in bands:
            out["gates"][fam] = gate(bands[fam][0], bands["base"][0], bands["L4_xsh"][0] if "L4_xsh" in bands else None)
    # ---- leak gate from the fit logs
    lk = {}
    for x in arms:
        lp = os.path.join(LOGS, f"{world if x != 'leaked' else world}_place_{x}.json")
        lg = json.load(open(lp))
        lk[x] = sorted({c for k_, v in lg.items() if isinstance(v, dict) for c in v.get("leak_flagged", [])})
    out["leak_gate"] = dict(flagged_by_arm=lk, leaked_arm_flagged=bool(lk.get("leaked")), clean_arms_unflagged=all(not v for x, v in lk.items() if x != "leaked"))
    if "leaked" in lk:
        assert lk["leaked"] and all(not v for x, v in lk.items() if x != "leaked"), "leak gate failed: the offender was not flagged or a clean arm was"
    # ---- date estimators (ensembles + rules), chosen on VALIDATION A3
    E = {x: {f: ens(world, x, f)[0] for f in ("val", "test")} for x in ("base", "L4", "latedays") if x in arms}
    E["km"] = km
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    a3v = lambda p: float(np.median(np.abs(7 * p[EVf["val"]] - 7 * Yf["val"][EVf["val"]])))
    if "latedays" in E:
        wcrit = [a3v(w * km["val"] + (1 - w) * E["latedays"]["val"]) for w in grid]
        wb = float(grid[int(np.argmin(wcrit))])
        E["blend_km_latedays"] = {f: wb * km[f] + (1 - wb) * E["latedays"][f] for f in ("val", "test")}
        out["blend_weight_km"] = wb
    val_a3 = {x: a3v(E[x]["val"]) for x in E}
    product = min(val_a3, key=val_a3.get)
    out["date_estimators"] = dict(validation_a3=val_a3, chosen_on_validation=product, km_note=k_note, km_offset_a_weeks=a_off,
                                  test={x: date_metrics(E[x]["test"], Yf["test"], EVf["test"], Rf["test"]) for x in E})
    tau_t = tau[idx["test"]]
    wk_blocks = [np.flatnonzero(tau_t == u) for u in np.unique(tau_t)]
    out["date_block"] = block_det({x: E[x]["test"] for x in E}, Yf["test"], EVf["test"], Rf["test"], wk_blocks, 41,
                                  [(x, "base") for x in E if x != "base"] + [(product, "km")] if product != "km" else [(x, "base") for x in E if x != "base"])
    # ---- interval: split-conformal on the product's validation residuals
    rv = 7 * (Yf["val"][EVf["val"]] - E[product]["val"][EVf["val"]])
    q10, q90 = float(np.quantile(rv, 0.10, method="lower")), float(np.quantile(rv, 0.90, method="higher"))
    mv = pd.DatetimeIndex(cr[idx["val"]]).month.to_numpy()[EVf["val"]]
    cov_v = {int(m): float(((rv[mv == m] >= q10) & (rv[mv == m] <= q90)).mean()) for m in np.unique(mv)}
    month_specific = (max(cov_v.values()) - min(cov_v.values())) > 0.05
    qm = {}
    if month_specific:
        for m in np.unique(mv):
            r = rv[mv == m]; qm[int(m)] = (float(np.quantile(r, 0.10, method="lower")), float(np.quantile(r, 0.90, method="higher")))
    rt = 7 * (Yf["test"] - E[product]["test"]); mt = pd.DatetimeIndex(cr[idx["test"]]).month.to_numpy()
    lo = np.array([qm.get(int(m), (q10, q90))[0] for m in mt]); hi = np.array([qm.get(int(m), (q10, q90))[1] for m in mt])
    inside = (rt >= lo) & (rt <= hi); ev = EVf["test"]
    out["interval"] = dict(estimator=product, q10_days=q10, q90_days=q90, width_days_pooled=q90 - q10,
                           validation_month_coverage_of_pooled=cov_v, month_specific_used=bool(month_specific), month_quantiles=qm,
                           test_coverage=float(inside[ev].mean()), test_width_days_mean=float((hi - lo)[ev].mean()),
                           test_coverage_by_month={int(m): float(inside[ev & (mt == m)].mean()) for m in np.unique(mt[ev])},
                           test_width_by_month={int(m): float((hi - lo)[ev & (mt == m)].mean()) for m in np.unique(mt[ev])},
                           coverage_block_ci=[float(x) for x in np.percentile([inside[np.concatenate([wk_blocks[j] for j in rr])][ev[np.concatenate([wk_blocks[j] for j in rr])]].mean()
                                                                               for rr in np.random.default_rng(5).integers(0, len(wk_blocks), (B, len(wk_blocks)))], [2.5, 97.5])])
    # ---- the flag: RAW and RECALIBRATED, apart
    if "flag" in arms:
        fv, _ = ens(world, "flag", "val"); ft, _ = ens(world, "flag", "test")
        ylv, kv = S21.uc1p_label(Yf["val"], EVf["val"], Rf["val"]); ylt, kt = S21.uc1p_label(Yf["test"], EVf["test"], Rf["test"])
        iso = isotonic(fv[kv], ylv[kv])
        out["flag"] = dict(raw=calib(ft[kt], ylt[kt]), recalibrated=calib(iso.predict(ft[kt]), ylt[kt]),
                           lateness_auc_raw=float(roc_auc((Yf["test"][EVf["test"]] > Rf["test"][EVf["test"]]).astype(int), ft[EVf["test"]])))
    # ---- decision level UC1-P (Phase 15 machinery), per creation month spread
    month_start = pd.DatetimeIndex(tau_t).to_period("M").to_timestamp().values
    per_seed = {f"{x}": [load(world, x, s) for s in SEEDS] for x in ("base", "L4", "latedays") if x in arms}
    det = {f"product date ({product})": dict(val=E[product]["val"], test=E[product]["test"], zv=z0["val"], zt=z0["test"]),
           "standalone KM": dict(val=km["val"], test=km["test"], zv=z0["val"], zt=z0["test"])}
    summ, _ = S21.decision_arrival(pw, per_seed, det, month_start, place=True)
    if "flag" in arms:
        flag_arms = {"flag P(late), per seed": []}
        for s in SEEDS:
            z = load(world, "flag", s)
            yv_, kv_ = S21.uc1p_label(np.asarray(z["val"]["Y"], float), np.asarray(z["val"]["EV"], bool), np.asarray(z["val"]["AUX"], float))
            yt_, kt_ = S21.uc1p_label(np.asarray(z["test"]["Y"], float), np.asarray(z["test"]["EV"], bool), np.asarray(z["test"]["AUX"], float))
            flag_arms["flag P(late), per seed"].append((np.asarray(z["val"]["P"], float)[kv_], yv_[kv_], np.asarray(z["test"]["P"], float)[kt_], yt_[kt_], month_start[kt_]))
        summ.update(P20D.summarise(flag_arms))
    if "flag_lag1" in arms:                    # STRICT (row tau - 1): what a planner could have on the day the line is raised
        fa = {"flag_lag1 P(late) STRICT, per seed": []}
        for s in SEEDS:
            z = load(world, "flag_lag1", s)
            yv_, kv_ = S21.uc1p_label(np.asarray(z["val"]["Y"], float), np.asarray(z["val"]["EV"], bool), np.asarray(z["val"]["AUX"], float))
            yt_, kt_ = S21.uc1p_label(np.asarray(z["test"]["Y"], float), np.asarray(z["test"]["EV"], bool), np.asarray(z["test"]["AUX"], float))
            fa["flag_lag1 P(late) STRICT, per seed"].append((np.asarray(z["val"]["P"], float)[kv_], yv_[kv_], np.asarray(z["test"]["P"], float)[kt_], yt_[kt_], month_start[kt_]))
        summ.update(P20D.summarise(fa))
    out["decisions_UC1P"] = S21.compact(summ)
    out["phase21_class_reference"] = "UC1-P (Phase 21, no-leak BASE_nl): ALERT, PARTIAL at 0.80, lift 2.22"
    name = f"phase22/order_time_{world}.json"
    C.dump(out, name)
    print(json.dumps({k: out[k] for k in ("gate_selftest", "gates", "leak_gate", "date_estimators", "interval")}, indent=1, default=str)[:7000])
    for n, r in out["decisions_UC1P"].items():
        print(f"  UC1-P {n:40s} {r['cls']} {r['reachable']} bar {r['operating_bar']} lift {r['lift_at_bar']} P@cov {[ (k, round(v['precision'][1],3)) for k,v in r['coverage'].items() if v.get('precision')]}")


if __name__ == "__main__":
    main()
