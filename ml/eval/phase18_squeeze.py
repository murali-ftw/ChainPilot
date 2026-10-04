"""Phase 18 Stage 3 -- no-training squeezes. STORED predictions only; nothing is fitted except one blend weight and the
conformal offsets, all on VALIDATION (or, for 3d, on labels known by t0). Torch process, no LightGBM.

  a  seed-ensemble: the mean of the 5 seeds' stored predictions -- arrival hazard distribution (S, pT) / LightGBM point;
     fill 22-cell distribution; capacity quantiles and P(strain > 1). Neural ensemble vs LightGBM ensemble.
  b  blend: w * neural-ensemble + (1 - w) * LightGBM-ensemble, w on a 101-point grid chosen on VALIDATION by the use
     case's proper score (arrival lateness ROC-AUC on the expected-week scale; fill exact CRPS; capacity mean pinball),
     frozen, applied to test.
  c  arrival split-conformal: signed-day residuals 7 (Y - P50) on validation uncensored rows; the interval is
     [P50 + q10, P50 + q90]; coverage and width on test uncensored rows (Phase 17 A3's rows: raw P10-P90 covers 93%).
  d  capacity interval, level-aware and keyed to recent drift (guide 11.1): for each test snapshot t0 the offset is
     calibrated on the most recent labelled rows whose 90-day window ENDED by t0 (snapshot <= t0 - 13 weeks), the last
     4 such snapshots. Score = max(q10 - y, y - q90) / q50 (level-aware); interval [q10 - c q50, q90 + c q50],
     c = the 80% conformal quantile. Coverage reported PER SNAPSHOT with the spread.
Ensembles, blends and conformal intervals are single deterministic predictors: they are compared by a PAIRED ROW
BOOTSTRAP (1,000 test-row resamples, 95% interval of the difference). "Improves" = the interval excludes 0 favourably;
ensemble vs single seeds is read against the MEAN of the five single-seed metrics on the same resample.
RAW and RECALIBRATED are reported in separate tables (3c / 3d are the recalibrated ones).

  python ml/eval/phase18_squeeze.py          # -> ml/artifacts/phase18/stage3_squeeze.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import phase5_heads as P5, folds
import phase5_metrics as M
from metrics import cindex, roc_auc
from heads import fill_cell
from phase14_score import p_exceed
from phase15 import coverage_curve, pick_on_val, apply
import phase18_score as S18

SEEDS = C.V8_SEEDS
B = 1000
GRID = np.linspace(0, 1, 101)


# ================================================================== loading
def neural(task):
    return [S18.load_neural(task, s) for s in SEEDS]


def lgbm(task):
    t = S18.TASK[task]
    return [S18.load_flat(os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz"), s) for s in SEEDS]


def ens(zs, f, keys):
    out = {k: np.mean([np.asarray(z[f][k], float) for z in zs], 0) for k in keys}
    out.update(Y=zs[0][f]["Y"], EV=zs[0][f]["EV"])
    return out


# ================================================================== paired bootstrap
def boot(metric_fns, n, seed=0, stratify=None):
    """metric_fns: {name: fn(idx) -> float}. -> {name: array[B]} on the SAME resamples."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B) for k in metric_fns}
    for b in range(B):
        i = rng.integers(0, n, n)
        for k, fn in metric_fns.items():
            out[k][b] = fn(i)
    return out


def diff_ci(a, b, higher):
    d = (a - b) if higher else (b - a)          # positive = a is better
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff_better_positive=[float(lo), float(np.mean(d)), float(hi)],
                verdict="better" if lo > 0 else "worse" if hi < 0 else "undetermined")


# ================================================================== a + b: arrival
def arrival(out):
    N, L = neural("arrival"), lgbm("arrival")
    R = S18.arrival_reference()
    pred = {}
    for f in ("val", "test"):
        en = ens(N, f, ("S", "pT")); el = ens(L, f, ("P",))
        pred[f] = dict(neural_ens=S18.expected_week(en), lgbm_ens=el["P"],
                       **{f"neural_s{s}": S18.expected_week(N[i][f]) for i, s in enumerate(SEEDS)},
                       **{f"lgbm_s{s}": np.asarray(L[i][f]["P"], float) for i, s in enumerate(SEEDS)},
                       Y=np.asarray(en["Y"], float), EV=np.asarray(en["EV"], bool), R=R[f])
    # blend weight on VALIDATION: lateness AUC of w*neural + (1-w)*lgbm
    v = pred["val"]; m = v["EV"]
    yl = (v["Y"][m] > v["R"][m]).astype(int)
    aucs = [roc_auc(yl, (w * v["neural_ens"] + (1 - w) * v["lgbm_ens"])[m] - v["R"][m]) for w in GRID]
    w = float(GRID[int(np.argmax(aucs))])
    for f in ("val", "test"):
        pred[f]["blend"] = w * pred[f]["neural_ens"] + (1 - w) * pred[f]["lgbm_ens"]
    t = pred["test"]; m = t["EV"]; Y, R = t["Y"], t["R"]
    yl = (Y[m] > R[m]).astype(int); idx_ev = np.flatnonzero(m)

    def late(p):
        return lambda i: roc_auc(yl[i], (p[m] - R[m])[i])

    def a3(p):
        e = np.abs(7 * p[m] - 7 * Y[m]); return lambda i: float(np.median(e[i]))

    def ci(p):
        return lambda i: float(cindex(p[i], Y[i], m[i], n=20_000, seed=13))
    arms = ["neural_ens", "lgbm_ens", "blend"] + [f"neural_s{s}" for s in SEEDS] + [f"lgbm_s{s}" for s in SEEDS]
    point = {a: dict(lateness_auc=late(t[a])(np.arange(m.sum())), a3_median_abs_err_days=a3(t[a])(np.arange(m.sum())),
                     cindex=float(cindex(t[a], Y, m))) for a in arms}
    bl = boot({a: late(t[a]) for a in arms}, int(m.sum()), 1)
    ba = boot({a: a3(t[a]) for a in arms}, int(m.sum()), 2)
    bc = boot({a: ci(t[a]) for a in arms}, len(Y), 3)
    res = dict(blend_weight_neural=w, val_lateness_auc_at_w=float(max(aucs)), point=point, compare={})
    for name, bs, higher in (("lateness_auc", bl, True), ("a3_median_abs_err_days", ba, False), ("cindex", bc, True)):
        mean_n = np.mean([bs[f"neural_s{s}"] for s in SEEDS], 0); mean_l = np.mean([bs[f"lgbm_s{s}"] for s in SEEDS], 0)
        res["compare"][name] = {"neural_ens vs mean single neural": diff_ci(bs["neural_ens"], mean_n, higher),
                                "lgbm_ens vs mean single lgbm": diff_ci(bs["lgbm_ens"], mean_l, higher),
                                "neural_ens vs lgbm_ens": diff_ci(bs["neural_ens"], bs["lgbm_ens"], higher),
                                "blend vs neural_ens": diff_ci(bs["blend"], bs["neural_ens"], higher),
                                "blend vs lgbm_ens": diff_ci(bs["blend"], bs["lgbm_ens"], higher)}
    out["arrival"] = res
    # ---- c: split-conformal on the P50 (and, as a diagnostic, on the expected week), per seed and for the ensemble
    out["arrival_conformal"] = conformal_arrival(N)


def dist_q(S, pT, p):
    F = 1 - np.asarray(S, float)
    return np.where((F >= p).any(1), 1 + np.argmax(F >= p, 1), 13).astype(float)


def conformal_arrival(N):
    rows = {}
    for name, zz in [(f"neural_s{s}", N[i]) for i, s in enumerate(SEEDS)] + [("neural_ens", None)]:
        if zz is None:
            zz = {f: ens(N, f, ("S", "pT")) for f in ("val", "test")}
        r = {}
        for center in ("p50", "expected_week"):
            c = {f: (dist_q(zz[f]["S"], zz[f]["pT"], 0.5) if center == "p50" else S18.expected_week(zz[f])) for f in ("val", "test")}
            ev_v = np.asarray(zz["val"]["EV"], bool); ev_t = np.asarray(zz["test"]["EV"], bool)
            res_v = 7 * (np.asarray(zz["val"]["Y"], float)[ev_v] - c["val"][ev_v])
            q10 = float(np.quantile(res_v, 0.10, method="lower"))           # conservative split-conformal ends
            q90 = float(np.quantile(res_v, 0.90, method="higher"))
            yt = 7 * np.asarray(zz["test"]["Y"], float)[ev_t]; ct = 7 * c["test"][ev_t]
            r[center] = dict(q10_days=q10, q90_days=float(q90), width_days=float(q90 - q10),
                             test_coverage=float(np.mean((yt >= ct + q10) & (yt <= ct + q90))),
                             val_coverage=float(np.mean((res_v >= q10) & (res_v <= q90))))
        p10 = {f: dist_q(zz[f]["S"], zz[f]["pT"], 0.1) for f in ("test",)}; p90 = {f: dist_q(zz[f]["S"], zz[f]["pT"], 0.9) for f in ("test",)}
        ev_t = np.asarray(zz["test"]["EV"], bool); Yt = np.asarray(zz["test"]["Y"], float)
        r["raw_p10_p90"] = dict(test_coverage=float(np.mean((Yt[ev_t] >= p10["test"][ev_t]) & (Yt[ev_t] <= p90["test"][ev_t]))),
                                mean_width_days=float(np.mean(7 * (p90["test"][ev_t] - p10["test"][ev_t]))))
        rows[name] = r
    seeds = [rows[f"neural_s{s}"] for s in SEEDS]
    summ = {k: dict(raw=S18.band([x["raw_p10_p90"][k] for x in seeds]) if k in seeds[0]["raw_p10_p90"] else None)
            for k in ("test_coverage", "mean_width_days")}
    return dict(per_arm=rows, raw_band=summ,
                conformal_p50_band=dict(test_coverage=S18.band([x["p50"]["test_coverage"] for x in seeds]),
                                        width_days=S18.band([x["p50"]["width_days"] for x in seeds])),
                conformal_expected_band=dict(test_coverage=S18.band([x["expected_week"]["test_coverage"] for x in seeds]),
                                             width_days=S18.band([x["expected_week"]["width_days"] for x in seeds])))


# ================================================================== a + b: fill
def fill(out):
    N, L = neural("fill"), lgbm("fill")
    P = {}
    for f in ("val", "test"):
        P[f] = dict(neural_ens=ens(N, f, ("P",))["P"], lgbm_ens=ens(L, f, ("P",))["P"],
                    **{f"neural_s{s}": np.asarray(N[i][f]["P"], float) for i, s in enumerate(SEEDS)},
                    **{f"lgbm_s{s}": np.asarray(L[i][f]["P"], float) for i, s in enumerate(SEEDS)}, Y=np.asarray(N[0][f]["Y"], float))
    yv = P["val"]["Y"]
    crps_w = [M.crps_exact_rows(w * P["val"]["neural_ens"] + (1 - w) * P["val"]["lgbm_ens"], yv).mean() for w in GRID]
    w = float(GRID[int(np.argmin(crps_w))])
    for f in ("val", "test"):
        P[f]["blend"] = w * P[f]["neural_ens"] + (1 - w) * P[f]["lgbm_ens"]
    t = P["test"]; y = t["Y"]; full = (y >= 1).astype(int); cells = fill_cell(y)
    arms = ["neural_ens", "lgbm_ens", "blend"] + [f"neural_s{s}" for s in SEEDS] + [f"lgbm_s{s}" for s in SEEDS]
    rows_crps = {a: M.crps_exact_rows(t[a], y) for a in arms}
    point = {a: dict(crps_exact=float(rows_crps[a].mean()), p_full_auc=float(roc_auc_score(full, t[a][:, 21])),
                     ece22=float(M.ece_marginal(t[a], cells)[0])) for a in arms}
    bc = boot({a: (lambda i, a=a: float(rows_crps[a][i].mean())) for a in arms}, len(y), 4)
    bu = boot({a: (lambda i, a=a: roc_auc(full[i], t[a][i, 21])) for a in arms}, len(y), 5)
    be = boot({a: (lambda i, a=a: float(M.ece_marginal(t[a][i], cells[i])[0])) for a in arms}, len(y), 6)
    res = dict(blend_weight_neural=w, val_crps_at_w=float(min(crps_w)), point=point, compare={})
    for name, bs, higher in (("crps_exact", bc, False), ("p_full_auc", bu, True), ("ece22", be, False)):
        mean_n = np.mean([bs[f"neural_s{s}"] for s in SEEDS], 0); mean_l = np.mean([bs[f"lgbm_s{s}"] for s in SEEDS], 0)
        res["compare"][name] = {"neural_ens vs mean single neural": diff_ci(bs["neural_ens"], mean_n, higher),
                                "lgbm_ens vs mean single lgbm": diff_ci(bs["lgbm_ens"], mean_l, higher),
                                "neural_ens vs lgbm_ens": diff_ci(bs["neural_ens"], bs["lgbm_ens"], higher),
                                "blend vs neural_ens": diff_ci(bs["blend"], bs["neural_ens"], higher),
                                "blend vs lgbm_ens": diff_ci(bs["blend"], bs["lgbm_ens"], higher)}
    out["fill"] = res


# ================================================================== a + b + d: capacity
def capacity(out):
    N, L = neural("capacity"), lgbm("capacity")
    Q, Sx = {}, {}
    for f in ("val", "test"):
        Q[f] = dict(neural_ens=ens(N, f, ("P",))["P"], lgbm_ens=ens(L, f, ("P",))["P"], Y=np.asarray(N[0][f]["Y"], float))
        Sx[f] = dict(neural_ens=np.mean([p_exceed(z[f]["P"]) for z in N], 0), lgbm_ens=np.mean([p_exceed(z[f]["P"]) for z in L], 0),
                     **{f"neural_s{s}": p_exceed(N[i][f]["P"]) for i, s in enumerate(SEEDS)},
                     **{f"lgbm_s{s}": p_exceed(L[i][f]["P"]) for i, s in enumerate(SEEDS)})
    yv = Q["val"]["Y"]
    pin = lambda Qx, y: float(np.mean([M.pinball_rows(y, Qx[:, i], q).mean() for i, q in enumerate(M.QS)]))
    pw = [pin(w * Q["val"]["neural_ens"] + (1 - w) * Q["val"]["lgbm_ens"], yv) for w in GRID]
    w = float(GRID[int(np.argmin(pw))])
    for f in ("val", "test"):
        Q[f]["blend"] = w * Q[f]["neural_ens"] + (1 - w) * Q[f]["lgbm_ens"]; Sx[f]["blend"] = p_exceed(Q[f]["blend"])
    lv, lt = (Q["val"]["Y"] > 1).astype(int), (Q["test"]["Y"] > 1).astype(int)
    arms = ["neural_ens", "lgbm_ens", "blend"] + [f"neural_s{s}" for s in SEEDS] + [f"lgbm_s{s}" for s in SEEDS]
    taus = {a: {p: pick_on_val(Sx["val"][a], lv, p)[0] for p in S18.PBAR} for a in arms}
    n = len(lt)

    def prec_at(a, c):
        def fn(i):
            s, y = Sx["test"][a][i], lt[i]; mm = max(1, int(np.ceil(c * len(y))))
            o = np.argsort(-s, kind="stable")[:mm]; return float(y[o].mean())
        return fn

    def rec_at(a, p):
        def fn(i):
            tau = taus[a][p]
            if tau is None:
                return np.nan
            s, y = Sx["test"][a][i], lt[i]; return float(((s >= tau) & (y == 1)).sum() / max(y.sum(), 1))
        return fn
    point, res = {}, dict(blend_weight_neural=w, val_pinball_at_w=float(min(pw)), compare={})
    for a in arms:
        point[a] = {f"precision_at_{int(c * 100)}pct": prec_at(a, c)(np.arange(n)) for c in S18.COV}
        point[a].update({f"recall_at_p{p:.2f}": rec_at(a, p)(np.arange(n)) for p in S18.PBAR})
    res["point"] = point
    for c in S18.COV:
        bs = boot({a: prec_at(a, c) for a in arms}, n, 7)
        res["compare"][f"precision_at_{int(c * 100)}pct"] = _cmp5(bs, True)
    for p in S18.PBAR:
        bs = boot({a: rec_at(a, p) for a in arms}, n, 8)
        res["compare"][f"recall_at_p{p:.2f}"] = _cmp5(bs, True)
    out["capacity"] = res
    out["capacity_interval"] = capacity_interval(N, L)


def _cmp5(bs, higher):
    mean_n = np.mean([bs[f"neural_s{s}"] for s in SEEDS], 0); mean_l = np.mean([bs[f"lgbm_s{s}"] for s in SEEDS], 0)
    return {"neural_ens vs mean single neural": diff_ci(bs["neural_ens"], mean_n, higher),
            "lgbm_ens vs mean single lgbm": diff_ci(bs["lgbm_ens"], mean_l, higher),
            "neural_ens vs lgbm_ens": diff_ci(bs["neural_ens"], bs["lgbm_ens"], higher),
            "blend vs neural_ens": diff_ci(bs["blend"], bs["neural_ens"], higher),
            "blend vs lgbm_ens": diff_ci(bs["blend"], bs["lgbm_ens"], higher)}


def capacity_interval(N, L, k_snap=4, alpha=0.20):
    """3d. Calibrate on labels KNOWN by t0 (window ended: snapshot <= t0 - 13 weeks), last k_snap such snapshots."""
    lb = P5.labels("v8", "capacity_strain"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    snap = {f: pd.to_datetime(lb.snapshot_date.values[P5.ordered(lb, m)]) for f, m in (("val", va), ("test", te))}
    out = {}
    for fam, zs in (("neural", N), ("lgbm", L)):
        per_seed = []
        for z in zs:
            Qa = np.vstack([z["val"]["P"], z["test"]["P"]]).astype(float); ya = np.r_[z["val"]["Y"], z["test"]["Y"]]
            sa = np.r_[snap["val"], snap["test"]]; is_test = np.r_[np.zeros(len(z["val"]["Y"]), bool), np.ones(len(z["test"]["Y"]), bool)]
            q10, q50, q90 = Qa[:, 0], np.maximum(Qa[:, 1], 1e-6), Qa[:, 2]
            score = np.maximum(q10 - ya, ya - q90) / q50
            # static: one offset from the whole validation fold (the fixed-factor reference the brief contrasts)
            sv = score[~is_test]; cs = float(np.quantile(sv, np.ceil((len(sv) + 1) * (1 - alpha)) / len(sv)))
            rows = []
            for t0 in sorted(pd.unique(sa[is_test])):
                t0 = pd.Timestamp(t0)
                known = np.unique(sa[sa <= t0 - pd.Timedelta(weeks=13)])
                cal_snaps = known[-k_snap:]
                cal = np.isin(sa, cal_snaps)
                assert (sa[cal] + pd.Timedelta(days=90) <= t0).all(), "a calibration label is not known by t0"
                sc = score[cal]; c = float(np.quantile(sc, min(1.0, np.ceil((len(sc) + 1) * (1 - alpha)) / len(sc))))
                tm = is_test & (sa == t0); y = ya[tm]
                rows.append(dict(snapshot=str(t0.date()), n=int(tm.sum()), calib_snapshots=[str(pd.Timestamp(x).date()) for x in cal_snaps],
                                 raw=float(np.mean((y >= q10[tm]) & (y <= q90[tm]))),
                                 static=float(np.mean((y >= q10[tm] - cs * q50[tm]) & (y <= q90[tm] + cs * q50[tm]))),
                                 trailing=float(np.mean((y >= q10[tm] - c * q50[tm]) & (y <= q90[tm] + c * q50[tm]))),
                                 width_raw=float(np.mean(q90[tm] - q10[tm])),
                                 width_static=float(np.mean(q90[tm] - q10[tm] + 2 * cs * q50[tm])),
                                 width_trailing=float(np.mean(q90[tm] - q10[tm] + 2 * c * q50[tm])), c=c))
            per_seed.append(dict(static_c=cs, per_snapshot=rows))
        def spread(key):
            v = np.array([[r[key] for r in ps["per_snapshot"]] for ps in per_seed])      # seeds x snapshots
            m = v.mean(0)                                                               # per snapshot, seed-mean
            return dict(per_snapshot_seed_mean=[round(float(x), 4) for x in m], min=float(m.min()),
                        q25=float(np.quantile(m, .25)), median=float(np.median(m)), q75=float(np.quantile(m, .75)),
                        max=float(m.max()), pooled_mean=float(m.mean()),
                        seed_band_of_pooled=S18.band(list(v.mean(1))))
        out[fam] = dict(snapshots=[r["snapshot"] for r in per_seed[0]["per_snapshot"]],
                        coverage=dict(raw=spread("raw"), static=spread("static"), trailing=spread("trailing")),
                        width=dict(raw=spread("width_raw"), static=spread("width_static"), trailing=spread("width_trailing")),
                        per_seed=per_seed)
    return out


def main():
    st = C.require_clean()
    out = dict(stamp=st, B=B)
    arrival(out); fill(out); capacity(out)
    C.dump(out, "phase18/stage3_squeeze.json")
    for k in ("arrival", "fill", "capacity"):
        print(f"== {k} (w_neural = {out[k]['blend_weight_neural']})")
        for a in ("neural_ens", "lgbm_ens", "blend"):
            print(f"  {a:12s}", {m: round(v, 4) for m, v in out[k]["point"][a].items()})
        for m, d in out[k]["compare"].items():
            print(f"  {m}: " + "; ".join(f"{c}: {v['verdict']}" for c, v in d.items()))
    print(json.dumps(dict(conf=out["arrival_conformal"]["conformal_p50_band"], conf_ew=out["arrival_conformal"]["conformal_expected_band"],
                          raw=out["arrival_conformal"]["raw_band"]), indent=1))
    for fam in ("neural", "lgbm"):
        print(fam, {k: (round(v["min"], 3), round(v["median"], 3), round(v["max"], 3)) for k, v in out["capacity_interval"][fam]["coverage"].items()})


if __name__ == "__main__":
    main()
