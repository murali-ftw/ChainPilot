"""Phase 22 Stage 1d-f -- the honest restatement: published (leaky) vs clean, never mixed in one column. Torch process (imports
only), no LightGBM. Exact per-seed filenames, no globs. Metric functions are the earlier phases' unchanged (phase18_score,
phase11b_lateness, phase15 / phase20_decisions, phase14_score).

Arms (pre-registration D3):
  LightGBM  published = phase7_preds v8 b5flat_*; nl = phase22 v8 p22_nl; clean = phase22 v8clean p22_base; lag1 (diagnostic)
  neural    published = bundles/{task}/v8_{incumbent}_s{s}; clean = phase22/bundles/{task}/v8clean_{incumbent}_s{s}
Per use case: 5-seed bands (all five seeds required), leak delta = clean - published (band of per-seed differences is NOT
used: bands are compared as bands) and the snapshot-block interval of the 5-seed ENSEMBLE difference (1,000 resamples).
Phase 15 classes (phase20_decisions.summarise) for published and clean, per seed and ensemble. Capacity also per calendar
quarter: precision of the top 5% of each quarter's rows and at the validation-chosen threshold for p = 0.80, share flagged.
Blends re-fitted on validation with clean arms (D3): arrival Phase 19 recipe (neural ensemble + LightGBM + fwd_load) and the
Phase 21 hybrid (incumbent + LightGBM season + cadence + L5), published weights beside them.

  python ml/eval/phase22_restate.py [--tasks arrival,fill,capacity]   -> ml/artifacts/phase22/restate.json
"""
from __future__ import annotations
import os, sys, json, argparse, itertools
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import phase5_heads as P5, folds
import phase5_metrics as M
from metrics import cindex, roc_auc
from phase11b_lateness import lateness, build_reference
from phase14_score import p_late, p_exceed
from phase15 import pick_on_val, apply
import phase18_score as S18
import phase20_score as P20S
import phase20_decisions as P20D
import phase21_score as S21

SEEDS = C.V8_SEEDS
B = 1000
PR22 = os.path.join(C.ART, "phase22", "preds")
B22 = os.path.join(C.ART, "phase22", "bundles")
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
INC = {"arrival": "lite_h4_lr0.00025", "fill": "none_h0_lr0.000125", "capacity": "mp_h4_lr0.00025"}
LG = {"arrival": "b5flat_reg", "fill": "b5flat22", "capacity": "b5flat_q"}


def register():
    PP.register()
    for p in ("v8", "v8w1002"):
        config.WORLDS[p + "clean"] = config.WORLDS[p]; config.EXPECTED_PANEL_D[p + "clean"] = 15


def path(kind, task, s, f):
    t = TASK[task]
    return {"lgbm_published": os.path.join(S18.PR7, f"v8_{t}_{LG[task]}_s{s}_{f}.npz"),
            "lgbm_nl": os.path.join(PR22, f"v8_{t}_p22_nl_s{s}_{f}.npz"),
            "lgbm_clean": os.path.join(PR22, f"v8clean_{t}_p22_base_s{s}_{f}.npz"),
            "lgbm_lag1": os.path.join(PR22, f"v8clean_{t}_p22_lag1_s{s}_{f}.npz"),
            "lgbm_fwdload_clean": os.path.join(PR22, f"v8clean_{t}_p22_fwdload_s{s}_{f}.npz"),
            "lgbm_scL5_clean": os.path.join(PR22, f"v8clean_{t}_p22_sc_L5_s{s}_{f}.npz"),
            "lgbm_fwdload_published": os.path.join(S18.PR18, f"v8_{t}_p18_fwd_load_s{s}_{f}.npz"),
            "neural_published": os.path.join(C.BUND, t, f"v8_{INC[task]}_s{s}", f"preds_{f}.npz"),
            "neural_clean": os.path.join(B22, t, f"v8clean_{INC[task]}_s{s}", f"preds_{f}.npz")}[kind]


def have(kind, task):
    return all(os.path.exists(path(kind, task, s, f)) for s in SEEDS for f in ("val", "test"))


def load(kind, task, s):
    return {f: dict(np.load(path(kind, task, s, f))) for f in ("val", "test")}


_RC = {}


def rc(fold):
    if not _RC:
        lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
        R = build_reference("contracted_lead", "v8", lb, tr)[0]
        _RC.update(val=R[P5.ordered(lb, va)], test=R[P5.ordered(lb, te)])
    return _RC[fold]


def test_snaps(task):
    lb = P5.labels("v8", TASK[task]); tr, va, te = folds.fixed_split(lb.snapshot_date)
    return lb.snapshot_date.values[P5.ordered(lb, te)]


# ================================================================== per-seed metrics
def arrival_metrics(zz):
    z = zz["test"]; Y, EV = np.asarray(z["Y"], float), np.asarray(z["EV"], bool)
    ET = S18.point("arrival", z); R = P20S.ref("v8", "test")
    out = dict(lateness_auc=lateness(ET, Y, EV, R)[0], a3_median_abs_err_days=float(np.median(np.abs(7 * ET[EV] - 7 * Y[EV]))),
               cindex=float(cindex(ET, Y, EV)))
    Rc = rc("test"); yl, keep = S21.uc1_label(Y, EV, Rc)
    sc = p_late(z["S"], Rc) if "S" in z else ET - Rc
    for c in (0.01, 0.05, 0.10, 0.20):
        out[f"uc1_precision_at_{int(c * 100)}pct"] = P20D.prec_at(sc[keep], yl[keep], c)
    return out


def fill_metrics(zz):
    z = zz["test"]; P, Y = np.asarray(z["P"], float), np.asarray(z["Y"], float)
    out = S18.fill_metrics(P, Y)
    out["uc2b_precision_at_5pct"] = P20D.prec_at(P[:, :20].sum(1), (Y < 0.95).astype(int), 0.05)
    out["uc2_precision_at_5pct"] = P20D.prec_at(P[:, 21], (Y >= 1).astype(int), 0.05)
    return out


def cap_metrics(zz, snaps):
    out = S18.capacity_metrics(zz["val"]["P"], zz["val"]["Y"], zz["test"]["P"], zz["test"]["Y"])
    st, lt = p_exceed(np.asarray(zz["test"]["P"], float)), (np.asarray(zz["test"]["Y"]) > 1).astype(int)
    sv, lv = p_exceed(np.asarray(zz["val"]["P"], float)), (np.asarray(zz["val"]["Y"]) > 1).astype(int)
    tau, _ = pick_on_val(sv, lv, 0.80)
    q = pd.PeriodIndex(pd.to_datetime(snaps), freq="Q").astype(str)
    pq, pt, sf = {}, {}, {}
    for u in sorted(set(q)):
        m = q == u
        pq[u] = P20D.prec_at(st[m], lt[m], 0.05)
        if tau is not None:
            a = st[m] >= tau
            pt[u] = float(lt[m][a].mean()) if a.sum() else float("nan"); sf[u] = float(a.mean())
    out.update(quarter_p5_min=min(pq.values()), quarter_p5_mean=float(np.mean(list(pq.values()))),
               quarter_thr80_min=(min(v for v in pt.values() if np.isfinite(v)) if pt else float("nan")),
               quarter_thr80_mean=(float(np.nanmean(list(pt.values()))) if pt else float("nan")),
               share_flagged_thr80=(float(np.mean(st >= tau)) if tau is not None else float("nan")))
    out["_per_quarter"] = dict(p5=pq, thr80_precision=pt, thr80_share=sf)
    return out


def metrics(task, zz, snaps=None):
    return arrival_metrics(zz) if task == "arrival" else fill_metrics(zz) if task == "fill" else cap_metrics(zz, snaps)


DIR = {"arrival": {"lateness_auc": True, "a3_median_abs_err_days": False, "cindex": True, **{f"uc1_precision_at_{c}pct": True for c in (1, 5, 10, 20)}},
       "fill": {"crps_exact": False, "p_full_auc": True, "uc2b_precision_at_5pct": True, "uc2_precision_at_5pct": True},
       "capacity": {"precision_at_1pct": True, "precision_at_5pct": True, "precision_at_10pct": True, "recall_at_p0.70": True,
                    "recall_at_p0.80": True, "recall_at_p0.85": True, "quarter_p5_min": True, "quarter_p5_mean": True}}


def bands(task, kind, snaps):
    per = [metrics(task, load(kind, task, s), snaps) for s in SEEDS]
    out = {}
    for k in DIR[task]:
        v = [p[k] for p in per]
        out[k] = S18.band(v) if all(x is not None and np.isfinite(x) for x in v) else None
    return out, per


# ================================================================== ensembles + block bootstrap of the leak delta
def ensemble(task, kind, f):
    zs = [load(kind, task, s)[f] for s in SEEDS]
    if "S" in zs[0]:
        return {"S": np.mean([z["S"] for z in zs], 0), "pT": np.mean([z["pT"] for z in zs], 0), "Y": zs[0]["Y"], "EV": zs[0]["EV"]}
    return {"P": np.mean([np.asarray(z["P"], float) for z in zs], 0), "Y": zs[0]["Y"], "EV": zs[0]["EV"]}


def block_delta(task, a, b, snaps, seed):
    """clean ensemble (a) minus published ensemble (b); better = positive."""
    blocks = [np.flatnonzero(snaps == u) for u in np.unique(snaps)]
    Y, EV = np.asarray(a["Y"], float), np.asarray(a["EV"], bool)
    if task == "arrival":
        R, Rc = P20S.ref("v8", "test"), rc("test")
        pa, pb = S18.point("arrival", a), S18.point("arrival", b)
        yl, keep = S21.uc1_label(Y, EV, Rc)
        sa = p_late(a["S"], Rc) if "S" in a else pa - Rc; sb = p_late(b["S"], Rc) if "S" in b else pb - Rc
        fns = {"lateness_auc": lambda p, i: roc_auc((Y[i][EV[i]] > R[i][EV[i]]).astype(int), p[i][EV[i]] - R[i][EV[i]]),
               "a3_median_abs_err_days": lambda p, i: float(np.median(np.abs(7 * p[i][EV[i]] - 7 * Y[i][EV[i]])))}
        args = {"lateness_auc": (pa, pb), "a3_median_abs_err_days": (pa, pb)}
        k5 = lambda s_, i: P20D.prec_at(s_[i][keep[i]], yl[i][keep[i]], 0.05)
        fns["uc1_precision_at_5pct"] = k5; args["uc1_precision_at_5pct"] = (sa, sb)
    elif task == "fill":
        ra, rb = M.crps_exact_rows(a["P"], Y), M.crps_exact_rows(b["P"], Y)
        full = (Y >= 1).astype(int); short = (Y < 0.95).astype(int)
        fns = {"crps_exact": lambda r, i: float(r[i].mean()), "p_full_auc": lambda P, i: roc_auc(full[i], P[i][:, 21]),
               "uc2b_precision_at_5pct": lambda P, i: P20D.prec_at(P[i][:, :20].sum(1), short[i], 0.05)}
        args = {"crps_exact": (ra, rb), "p_full_auc": (a["P"], b["P"]), "uc2b_precision_at_5pct": (a["P"], b["P"])}
    else:
        lt = (Y > 1).astype(int); sa, sb = p_exceed(a["P"]), p_exceed(b["P"])
        fns = {"precision_at_5pct": lambda s_, i: P20D.prec_at(s_[i], lt[i], 0.05)}
        args = {"precision_at_5pct": (sa, sb)}
    rng = np.random.default_rng(seed); out = {}
    draws = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(B)]
    full_i = np.arange(len(Y))
    for m, fn in fns.items():
        hi = DIR[task].get(m, True)
        d = np.array([fn(args[m][0], i) - fn(args[m][1], i) for i in draws])
        d = d if hi else -d
        lo, mu, up = np.percentile(d, 2.5), d.mean(), np.percentile(d, 97.5)
        out[m] = dict(clean=float(fn(args[m][0], full_i)), published=float(fn(args[m][1], full_i)),
                      delta_clean_minus_published=float(fn(args[m][0], full_i) - fn(args[m][1], full_i)),
                      block_ci_better_positive=[float(lo), float(mu), float(up)],
                      verdict="clean better" if lo > 0 else "clean worse" if up < 0 else "undetermined")
    return out


# ================================================================== Phase 15 classes
def classes(task, kind):
    zs = [load(kind, task, s) for s in SEEDS]
    snap = test_snaps(task)
    if task == "arrival":
        Rv, Rt = rc("val"), rc("test")
        def pack(z):
            out = []
            for f, R in (("val", Rv), ("test", Rt)):
                Y, EV = np.asarray(z[f]["Y"], float), np.asarray(z[f]["EV"], bool)
                yl, keep = S21.uc1_label(Y, EV, R)
                s_ = p_late(z[f]["S"], R) if "S" in z[f] else S18.point("arrival", z[f]) - R
                out += [s_[keep], yl[keep]]
                if f == "test":
                    out.append(snap[keep])
            return tuple(out)
        arms = {f"{kind}, per seed": [pack(z) for z in zs]}
        return {"UC1": S21.compact(P20D.summarise(arms))}
    if task == "fill":
        out = {}
        for uc, sf, lf in (("UC2", lambda P: P[:, 21], lambda y: (y >= 1).astype(int)),
                           ("UC2b_lt0.95", lambda P: P[:, :20].sum(1), lambda y: (y < 0.95).astype(int))):
            arms = {f"{kind}, per seed": [(sf(np.asarray(z["val"]["P"], float)), lf(np.asarray(z["val"]["Y"], float)),
                                           sf(np.asarray(z["test"]["P"], float)), lf(np.asarray(z["test"]["Y"], float)), snap) for z in zs]}
            out[uc] = S21.compact(P20D.summarise(arms))
        return out
    arms = {f"{kind}, per seed": [(p_exceed(np.asarray(z["val"]["P"], float)), (np.asarray(z["val"]["Y"]) > 1).astype(int),
                                   p_exceed(np.asarray(z["test"]["P"], float)), (np.asarray(z["test"]["Y"]) > 1).astype(int), snap) for z in zs]}
    return {"UC3": S21.compact(P20D.summarise(arms))}


# ================================================================== blends re-fitted on validation (D3)
def refit_arrival_blends():
    Rv = P20S.ref("v8", "val")
    def vauc(p, z):
        Y, EV = np.asarray(z["Y"], float), np.asarray(z["EV"], bool); m = EV
        return roc_auc((Y[m] > Rv[m]).astype(int), p[m] - Rv[m])
    out = {}
    for label, nk, lk in (("published", "neural_published", "lgbm_fwdload_published"), ("clean", "neural_clean", "lgbm_fwdload_clean")):
        if not (have(nk, "arrival") and have(lk, "arrival")):
            out[label] = "NOT AVAILABLE"; continue
        ne = {f: S18.expected_week(ensemble("arrival", nk, f)) for f in ("val", "test")}
        le = {f: ensemble("arrival", lk, f)["P"] for f in ("val", "test")}
        zv = ensemble("arrival", nk, "val")
        grid = np.linspace(0, 1, 101)
        crit = [vauc(w * ne["val"] + (1 - w) * le["val"], zv) for w in grid]
        w = float(grid[int(np.argmax(crit))])
        zt = ensemble("arrival", nk, "test")
        out[label] = dict(recipe="Phase 19: w * neural incumbent ensemble + (1 - w) * LightGBM + fwd_load ensemble", w_neural=w,
                          test=arrival_metrics({"test": {"P": w * ne["test"] + (1 - w) * le["test"], "Y": zt["Y"], "EV": zt["EV"]}}))
    # Phase 21 hybrid: incumbent + LightGBM season + cadence + L5 (the Phase 19 neural rf arm is not retrained clean)
    for label, nk, lk in (("hybrid_clean", "neural_clean", "lgbm_scL5_clean"),):
        if not (have(nk, "arrival") and have(lk, "arrival")):
            out[label] = "NOT AVAILABLE"; continue
        ne = {f: S18.expected_week(ensemble("arrival", nk, f)) for f in ("val", "test")}
        le = {f: ensemble("arrival", lk, f)["P"] for f in ("val", "test")}
        zv = ensemble("arrival", nk, "val"); zt = ensemble("arrival", nk, "test")
        grid = np.linspace(0, 1, 21)
        crit = [vauc(w * ne["val"] + (1 - w) * le["val"], zv) for w in grid]
        w = float(grid[int(np.argmax(crit))])
        out[label] = dict(recipe="Phase 21 hybrid, clean: w * incumbent ensemble + (1 - w) * LightGBM + season + cadence + L5", w_neural=w,
                          published_weights="incumbent 0.00, Phase 19 neural 0.55, LightGBM + season + cadence + L5 0.45 (Phase 21 stage 6a)",
                          test=arrival_metrics({"test": {"P": w * ne["test"] + (1 - w) * le["test"], "Y": zt["Y"], "EV": zt["EV"]}}))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default="arrival,fill,capacity"); a = ap.parse_args()
    register()
    st = C.require_clean()
    out = dict(stamp=st, tasks={})
    for task in a.tasks.split(","):
        snaps = test_snaps(task)
        res = dict(bands={}, per_seed={}, leak_delta={}, classes={})
        kinds = [k for k in ("lgbm_published", "lgbm_nl", "lgbm_clean", "lgbm_lag1", "neural_published", "neural_clean") if have(k, task)]
        z0 = load("lgbm_published", task, 7)
        for k in kinds:
            z = load(k, task, 7)
            for f in ("val", "test"):
                assert np.array_equal(np.asarray(z[f]["Y"], float), np.asarray(z0[f]["Y"], float)), f"{task} {k} {f}: rows differ"
            b, p = bands(task, k, snaps); res["bands"][k] = b; res["per_seed"][k] = p
        for fam, (pub, cl) in {"lgbm": ("lgbm_published", "lgbm_clean"), "lgbm_nl": ("lgbm_published", "lgbm_nl"),
                               "neural": ("neural_published", "neural_clean")}.items():
            if pub in kinds and cl in kinds:
                res["leak_delta"][fam] = dict(bands_compared={m: S18.compare(res["bands"][cl][m], res["bands"][pub][m], h)
                                                              for m, h in DIR[task].items()},
                                              block=block_delta(task, ensemble(task, cl, "test"), ensemble(task, pub, "test"), snaps, 31))
        for k in ("neural_published", "neural_clean", "lgbm_published", "lgbm_clean"):
            if k in kinds:
                res["classes"][k] = classes(task, k)
        out["tasks"][task] = res
        print(f"== {task}: arms {kinds}", flush=True)
        for k in kinds:
            print(f"  {k:18s}", {m: [round(x, 4) for x in v] if v else None for m, v in res["bands"][k].items()}, flush=True)
        for fam, d in res["leak_delta"].items():
            print(f"  DELTA {fam}:", {m: (round(x["delta_clean_minus_published"], 4), x["verdict"]) for m, x in d["block"].items()}, flush=True)
        for k, c in res["classes"].items():
            print(f"  CLASS {k}:", {uc: {n: (r['cls'], r['reachable']) for n, r in v.items()} for uc, v in c.items()}, flush=True)
    if "arrival" in a.tasks:
        out["arrival_blends"] = refit_arrival_blends()
        print("BLENDS", json.dumps({k: (v if isinstance(v, str) else dict(w=v["w_neural"], late=round(v["test"]["lateness_auc"], 4),
                                                                         a3=round(v["test"]["a3_median_abs_err_days"], 2))) for k, v in out["arrival_blends"].items()}))
    C.dump(out, f"phase22/restate_{'_'.join(a.tasks.split(','))}.json")


if __name__ == "__main__":
    main()
