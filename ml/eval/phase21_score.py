"""Phase 21 scorer -- Stages 3 (arrival snapshots), 4 (at placement), 5 (fill), 6d (slices), 7 (replication). Torch process,
no LightGBM. Exact per-seed filenames, no globs. Metric functions are the earlier phases' unchanged:
  arrival   lateness ROC-AUC against the world's as-of channel reference (phase11b_lateness.lateness, via phase20_score.ref),
            A3 median abs error in days on uncensored rows (Phase 18), C-index (context)
  fill      exact CRPS and P(fill = 1) AUC (phase18_score.fill_metrics); UC2b precision at 5% coverage = Phase 15's top-5%
            precision of the mass below 0.95 on the label fill < 0.95 (phase20_decisions.prec_at)
  place     (pre-registration D10) lateness AUC with the contracted lead as the reference (expected minus promise), A3
Bands [min, mean, max] need ALL FIVE seeds (Phase 19 deviation 171). better / worse only when DISJOINT.
Decision level: phase20_decisions.summarise (Phase 15's analyse + the Phase 20 class mapping), unchanged.
Deterministic predictors (standalone rules, seed ensembles): snapshot-block bootstrap (1,000 resamples of whole test
snapshots; at placement whole creation weeks), paired across arms.

  python ml/eval/phase21_score.py snap  [--world v8]     -> ml/artifacts/phase21/score_snap_{world}.json   (Stages 3, 5, 6d)
  python ml/eval/phase21_score.py place [--world v8]     -> ml/artifacts/phase21/score_place_{world}.json  (Stage 4)
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import phase21_paths as PP
import phase5_heads as P5, folds
from phase11b_lateness import build_reference, lateness
from metrics import cindex, roc_auc
import phase5_metrics as M
import phase18_score as S18
import phase20_score as P20S
import phase20_decisions as P20D
from phase14_score import p_late
import grpstats as GS

SEEDS = C.V8_SEEDS
B = 1000
PR21 = os.path.join(C.ART, "phase21", "preds")
PR20 = os.path.join(C.ART, "phase20", "preds")
TASKS = {"arrival": "arrival_week", "fill": "fill_rate", "place": "arrival_place"}
DIR = {"arrival": {"lateness_auc": True, "a3_median_abs_err_days": False},
       "fill": {"crps_exact": False, "p_full_auc": True, "uc2b_precision_at_5pct": True},
       "place": {"lateness_auc": True, "a3_median_abs_err_days": False}}
# family -> its controls (pre-registration "Which control gates which family")
GATES = {"arrival": {"L4": ["L4_xsh"], "L5": ["L5_perm", "L5_xsh"]},
         "fill": {"L4": ["L4_xsh"], "L5": ["L5_perm", "L5_xsh"], "ack": ["ack_xsh"], "L5_ack": ["L5_perm", "L5_xsh", "ack_xsh"]},
         "place": {"L4": [], "L5": ["L5_perm"]}}


def ksel(world):
    return json.load(open(os.path.join(C.ART, "phase21", f"k_select_{world}.json")))


def fmt(world, task, arm, k):
    t = TASKS[task]
    if arm == "base_nl":
        return os.path.join(PR21, f"{world}_{t}_p21_base_nl_s{{s}}_{{f}}.npz")
    if arm == "base":
        if task == "place":
            return os.path.join(PR21, f"{world}_{t}_p21_base_s{{s}}_{{f}}.npz")
        if world == "v8":
            return os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz")
        return os.path.join(PR20, f"{world}_{t}_p20_base_s{{s}}_{{f}}.npz")
    if arm == "sc" and task == "fill":
        return os.path.join(PR20, f"{world}_{t}_p20_lgbm_rf_s{{s}}_{{f}}.npz")      # BASE + season + cadence (Phase 20)
    return os.path.join(PR21, f"{world}_{t}_p21_{arm}_k{k}_s{{s}}_{{f}}.npz")


def load(world, task, arm, k, s):
    return {f: dict(np.load(fmt(world, task, arm, k).format(s=s, f=f))) for f in ("val", "test")}


def have(world, task, arm, k):
    return all(os.path.exists(fmt(world, task, arm, k).format(s=s, f=f)) for s in SEEDS for f in ("val", "test"))


# ================================================================== metrics
def uc2b_p5(P, y):
    return P20D.prec_at(np.asarray(P, float)[:, :20].sum(1), (np.asarray(y, float) < 0.95).astype(int), 0.05)


def metrics(world, task, zz):
    z = zz["test"]; Y, EV = np.asarray(z["Y"], float), np.asarray(z["EV"], bool)
    if task == "fill":
        out = S18.fill_metrics(z["P"], Y); out["uc2b_precision_at_5pct"] = uc2b_p5(z["P"], Y); return out
    ET = np.asarray(z["P"], float)
    R = P20S.ref(world, "test") if task == "arrival" else np.asarray(z["AUX"], float)
    return dict(lateness_auc=lateness(ET, Y, EV, R)[0], a3_median_abs_err_days=float(np.median(np.abs(7 * ET[EV] - 7 * Y[EV]))),
                cindex=float(cindex(ET, Y, EV)))


def bands(world, task, arm, k):
    per = [metrics(world, task, load(world, task, arm, k, s)) for s in SEEDS]
    out, unreach = {}, {}
    for key in per[0]:
        v = [p[key] for p in per]; ok = sum(x is not None and np.isfinite(x) for x in v)
        out[key] = S18.band(v) if ok == len(SEEDS) else None
        if ok < len(SEEDS):
            unreach[key] = f"UNREACHABLE for {len(SEEDS) - ok} of 5 seeds"
    return out, per, unreach


def cmp(task, a, b):
    return {key: S18.compare(a[key], b[key], h) for key, h in DIR[task].items()}


def gate(task, fam, base, ctrls):
    c = cmp(task, fam, base)
    better = [x for x, v in c.items() if v == "better"]; worse = [x for x, v in c.items() if v == "worse"]
    cb = {n: [x for x, v in cmp(task, b, base).items() if v == "better"] for n, b in ctrls.items()}
    bad = {n: v for n, v in cb.items() if v}
    ok = bool(better) and not worse and not bad
    why = ("better on " + ", ".join(better) + ("; no control better than BASE" if ctrls else "; (no control pre-registered)")) if ok else \
          ("worse on " + ", ".join(worse)) if worse else \
          (f"gain on {better} but control(s) also better than BASE: {bad}") if better and bad else \
          "no primary metric disjointly better (bands overlap)"
    return dict(verdict="PASS" if ok else "FAIL", why=why, vs_base=c, controls_better_than_base=cb)


def gain(a, b, key, higher):
    return (a[key][1] - b[key][1]) if higher else (b[key][1] - a[key][1])


# ================================================================== deterministic predictors and the block bootstrap
def fold_rows(world, task):
    """Ordered val / test row indices into the group store, asserted against the stored BASE files."""
    if task == "place":
        Z, _ = GS.load(world, "place")
        src = GS.Source(world)
        li = Z["line"].astype(np.int64)
        cr = pd.DatetimeIndex(src.created[li]); tau = pd.to_datetime(Z["tau"]).values
        tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(cr)]
        idx = {f: np.flatnonzero(m)[np.argsort(tau[np.flatnonzero(m)], kind="stable")] for f, m in (("val", va), ("test", te))}
        blocks_key = pd.to_datetime(Z["tau"]).values
        return Z, idx, blocks_key
    Z, _ = GS.load(world, "snap")
    lb = P5.labels(world, TASKS[task]); tr, va, te = folds.fixed_split(lb.snapshot_date)
    assert (lb.entity_id.to_numpy().astype(str) == Z["entity"]).all()
    idx = {"val": P5.ordered(lb, va), "test": P5.ordered(lb, te)}
    return Z, idx, lb.snapshot_date.values


def sub(Z, rows):
    return {k: v[rows] for k, v in Z.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(Z["month"])}


def block_boot(fns, blocks, seed):
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B) for k in fns}
    for b in range(B):
        i = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        for k, fn in fns.items():
            out[k][b] = fn(i)
    return out


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.mean(x)), float(np.percentile(x, 97.5))]


def det_compare(task, preds, Y, EV, R, blocks, seed, pairs):
    """preds: {name: test prediction} (arrival / place: weeks; fill: P22). -> point, block CI, paired diffs."""
    if task == "fill":
        rows = {a: M.crps_exact_rows(p, Y) for a, p in preds.items()}
        full = (Y >= 1).astype(int)
        mk = {"crps_exact": lambda a: (lambda i: float(rows[a][i].mean())),
              "p_full_auc": lambda a: (lambda i: roc_auc(full[i], preds[a][i, 21])),
              "uc2b_precision_at_5pct": lambda a: (lambda i: uc2b_p5(preds[a][i], Y[i]))}
    else:
        mk = {"lateness_auc": lambda a: (lambda i: roc_auc((Y[i][EV[i]] > R[i][EV[i]]).astype(int), preds[a][i][EV[i]] - R[i][EV[i]])),
              "a3_median_abs_err_days": lambda a: (lambda i: float(np.median(np.abs(7 * preds[a][i][EV[i]] - 7 * Y[i][EV[i]]))))}
    full_i = np.arange(len(Y))
    out = dict(point={}, ci={}, diff={})
    for mname, f in mk.items():
        bs = block_boot({a: f(a) for a in preds}, blocks, seed)
        out["point"][mname] = {a: float(f(a)(full_i)) for a in preds}
        out["ci"][mname] = {a: ci(bs[a]) for a in preds}
        hi = DIR[task][mname]
        for a, b in pairs:
            d = (bs[a] - bs[b]) if hi else (bs[b] - bs[a])
            c = ci(d)
            out["diff"].setdefault(mname, {})[f"{a} vs {b}"] = dict(better_positive=c, verdict="better" if c[0] > 0 else "worse" if c[2] < 0 else "undetermined")
    return out


# ================================================================== decision level (Phase 15 machinery, unchanged)
def uc1_label(Y, EV, R):
    late = (EV & (Y > R)) | (~EV & (R < 13)); keep = EV | (R < 13)
    return late.astype(int), keep


def uc1p_label(Y, EV, R):
    """At placement: censored lines are late if the contract is already shorter than their observed time."""
    late = (Y > R); keep = EV | (Y > R)
    return late.astype(int), keep


def contracted_ref(world):
    lb = P5.labels(world, "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    R = build_reference("contracted_lead", world, lb, tr)[0]
    return {"val": R[P5.ordered(lb, va)], "test": R[P5.ordered(lb, te)]}


def decision_arrival(world, per_seed, det, snap_test, place=False):
    """per_seed: {name: [zz per seed]} (point arms); det: {name: {val: pred, test: pred, Y/EV per fold}}."""
    arms = {}
    if place:
        lab = uc1p_label
        R = None
    else:
        lab = uc1_label
        R = contracted_ref(world)

    def pack(sv, st, zv, zt):
        Rv = np.asarray(zv["AUX"], float) if place else R["val"]; Rt = np.asarray(zt["AUX"], float) if place else R["test"]
        yv, kv = lab(np.asarray(zv["Y"], float), np.asarray(zv["EV"], bool), Rv)
        yt, kt = lab(np.asarray(zt["Y"], float), np.asarray(zt["EV"], bool), Rt)
        return ((sv - Rv)[kv], yv[kv], (st - Rt)[kt], yt[kt], snap_test[kt])
    for name, zs in per_seed.items():
        arms[name + ", per seed"] = [pack(np.asarray(z["val"]["P"], float), np.asarray(z["test"]["P"], float), z["val"], z["test"]) for z in zs]
        arms[name + ", 5-seed ensemble"] = [pack(np.mean([np.asarray(z["val"]["P"], float) for z in zs], 0),
                                                np.mean([np.asarray(z["test"]["P"], float) for z in zs], 0), zs[0]["val"], zs[0]["test"])]
    for name, d in det.items():
        arms[name] = [pack(d["val"], d["test"], d["zv"], d["zt"])]
    if world == "v8" and not place:            # the incumbent neural ensemble's P(late), Phase 20's reference arm
        inc = {f: [S18.load_neural("arrival", s)[f] for s in SEEDS] for f in ("val", "test")}
        Sm = {f: np.mean([np.asarray(z["S"], float) for z in inc[f]], 0) for f in ("val", "test")}
        z0 = inc["val"][0], inc["test"][0]
        yv, kv = lab(np.asarray(z0[0]["Y"], float), np.asarray(z0[0]["EV"], bool), R["val"])
        yt, kt = lab(np.asarray(z0[1]["Y"], float), np.asarray(z0[1]["EV"], bool), R["test"])
        arms["neural incumbent ensemble P(late) (Phase 20 reference)"] = [(p_late(Sm["val"], R["val"])[kv], yv[kv],
                                                                           p_late(Sm["test"], R["test"])[kt], yt[kt], snap_test[kt])]
    return P20D.summarise(arms), arms


def decision_fill(per_seed, det, snap_test):
    out = {}
    for uc, sf, lf in (("UC2", lambda P: P[:, 21], lambda y: (y >= 1).astype(int)),
                       ("UC2b_lt0.95", lambda P: P[:, :20].sum(1), lambda y: (y < 0.95).astype(int))):
        arms = {}
        for name, zs in per_seed.items():
            yv, yt = lf(np.asarray(zs[0]["val"]["Y"], float)), lf(np.asarray(zs[0]["test"]["Y"], float))
            arms[name + ", per seed"] = [(sf(np.asarray(z["val"]["P"], float)), yv, sf(np.asarray(z["test"]["P"], float)), yt, snap_test) for z in zs]
            arms[name + ", 5-seed ensemble"] = [(sf(np.mean([np.asarray(z["val"]["P"], float) for z in zs], 0)), yv,
                                                 sf(np.mean([np.asarray(z["test"]["P"], float) for z in zs], 0)), yt, snap_test)]
        for name, d in det.items():
            arms[name] = [(sf(d["val"]), lf(d["yv"]), sf(d["test"]), lf(d["yt"]), snap_test)]
        out[uc] = P20D.summarise(arms)
    return out


def compact(summ):
    keep = ("cls", "reachable", "operating_bar", "lift_at_bar", "stage_b", "base_rate", "coverage", "accuracy", "per_snapshot_precision_at_5pct")
    return {n: {k: r[k] for k in keep} for n, r in summ.items()}


# ================================================================== importances / resolved level
def importances(world, task, arm, k):
    log = json.load(open(os.path.join(C.ART, "phase21", f"proxy_fit_{world}_{task}.json")))
    t = TASKS[task]; agg = {}
    for s in SEEDS:
        imp = log[f"{world}|{t}|p21_{arm}_k{k}_s{s}"]["gain_importance"]
        tot = sum(imp.values())
        for c, v in imp.items():
            lvl = c.split("_")[0] if c.startswith("g") or c.startswith("ack") else "base"
            agg.setdefault(lvl, []).append(v / tot)
    return {lvl: float(np.sum(v) / len(SEEDS)) for lvl, v in agg.items()}


# ================================================================== stages
def stage_snap(world, out):
    ks = ksel(world); kA, kF = ks["k"]["arrival"], ks["k"]["fill"]
    Z, idx, snaps = fold_rows(world, "arrival")
    Zv, Zt = sub(Z, idx["val"]), sub(Z, idx["test"])
    snap_t = snaps[idx["test"]]
    blocks = [np.flatnonzero(snap_t == u) for u in np.unique(snap_t)]
    res = {}
    # ---------------------------------------------------------------- arrival (Stage 3)
    A = {}; UN = {}
    for arm in ("base", "L4", "L4_xsh", "L5", "L5_perm", "L5_xsh", "grp_only", "sc", "sc_L5"):
        if arm == "base" or have(world, "arrival", arm, kA):
            b, p, u = bands(world, "arrival", arm, kA); A[arm] = (b, p); UN[arm] = u
    base_b = A["base"][0]
    z0 = load(world, "arrival", "base", kA, 7)
    for f, Zf in (("val", Zv), ("test", Zt)):
        assert (np.asarray(z0[f]["entity"]).astype(str) == Zf["entity"]).all(), f"store rows != stored BASE rows ({f})"
    Bv = lambda arm: {a: x for a, x in A[arm][0].items()}
    g = {}
    for fam, ctrls in GATES["arrival"].items():
        if fam in A and all(c in A for c in ctrls):
            g[fam] = gate("arrival", A[fam][0], base_b, {c: A[c][0] for c in ctrls})
    if "sc" in A and "sc_L5" in A:
        g["L5 over season + cadence"] = dict(vs_sc=cmp("arrival", A["sc_L5"][0], A["sc"][0]), sc_vs_base=cmp("arrival", A["sc"][0], base_b))
    if "L5" in A and "L4" in A:
        g["control (c): L5 vs L4"] = cmp("arrival", A["L5"][0], A["L4"][0])
    if "grp_only" in A:
        g["grp_only vs BASE"] = cmp("arrival", A["grp_only"][0], base_b)
    # DIAGNOSTIC (deviation 199): the same comparison with the leaking panel columns removed from BASE
    NL = {}
    for arm in ("base_nl", "L4_nl", "L5_nl"):
        if have(world, "arrival", arm, kA):
            NL[arm] = bands(world, "arrival", arm, kA)[0]
    diag_nl = {}
    if len(NL) == 3:
        diag_nl = dict(bands=NL, L4_nl_vs_base_nl=cmp("arrival", NL["L4_nl"], NL["base_nl"]),
                       L5_nl_vs_base_nl=cmp("arrival", NL["L5_nl"], NL["base_nl"]), L5_nl_vs_L4_nl=cmp("arrival", NL["L5_nl"], NL["L4_nl"]),
                       base_nl_vs_base=cmp("arrival", NL["base_nl"], base_b),
                       gain_L5_nl=gain(NL["L5_nl"], NL["base_nl"], "lateness_auc", True), gain_L4_nl=gain(NL["L4_nl"], NL["base_nl"], "lateness_auc", True))
    # the standalone rule (D9) and seed ensembles, block bootstrap
    a_off = ks["snap"]["arrival"][str(kA)]["offset_a_weeks"] if str(kA) in ks["snap"]["arrival"] else ks["snap"]["arrival"][kA]["offset_a_weeks"]
    rule = {f: GS.standalone_arrival_weeks(Zf, kA) + a_off for f, Zf in (("val", Zv), ("test", Zt))}
    Yt, EVt = np.asarray(z0["test"]["Y"], float), np.asarray(z0["test"]["EV"], bool)
    Rt = P20S.ref(world, "test")
    ens = {arm: np.mean([np.asarray(load(world, "arrival", arm, kA, s)["test"]["P"], float) for s in SEEDS], 0) for arm in A}
    preds = {"rule": rule["test"], **{f"{a}_ens": v for a, v in ens.items()}}
    pairs = [("rule", "base_ens")] + [(f"{a}_ens", "base_ens") for a in A if a != "base"]
    det = det_compare("arrival", preds, Yt, EVt, Rt, blocks, 21, pairs)
    res["arrival"] = dict(k=kA, offset_a_weeks=a_off, bands={a: b for a, (b, _) in A.items()}, per_seed={a: p for a, (_, p) in A.items()},
                          unreachable={a: u for a, u in UN.items() if u}, gates=g, deterministic=det,
                          diagnostic_no_leak_columns=diag_nl)
    if "L5" in A:
        res["arrival"]["importance_L5"] = importances(world, "arrival", "L5", kA)
        # which resolved level do the rows that gain sit at? (ensembles, test, observed rows)
        lvl = GS.arrival_levels(Zt, kA, Zt["month"].astype(np.int64))[2]
        e_b = np.abs(ens["base"] - Yt); e_a = np.abs(ens["L5"] - Yt)
        rr = {}
        for L in (5, 4, 2, 1):
            m = (lvl == L) & EVt
            if m.sum() >= 50:
                rr[f"L{L}"] = dict(rows=int(m.sum()), share_of_rows_improved=float((e_a[m] < e_b[m]).mean()),
                                   lateness_auc_base_ens=lateness(ens["base"][m], Yt[m], EVt[m], Rt[m])[0] if False else
                                   float(roc_auc((Yt[m] > Rt[m]).astype(int), ens["base"][m] - Rt[m])),
                                   lateness_auc_L5_ens=float(roc_auc((Yt[m] > Rt[m]).astype(int), ens["L5"][m] - Rt[m])),
                                   mean_abs_err_days_base=float(7 * e_b[m].mean()), mean_abs_err_days_L5=float(7 * e_a[m].mean()))
            else:
                rr[f"L{L}"] = dict(rows=int(m.sum()))
        improved = EVt & (e_a < e_b - 1e-9)
        res["arrival"]["resolved_level"] = dict(by_level=rr, level_share_all_observed={f"L{L}": float((lvl[EVt] == L).mean()) for L in (5, 4, 2, 1)},
                                                level_share_improved_rows={f"L{L}": float((lvl[improved] == L).mean()) for L in (5, 4, 2, 1)})
    # decision level UC1
    per_seed = {"BASE (LightGBM flat)": [load(world, "arrival", "base", kA, s) for s in SEEDS]}
    for arm in ("L4", "L5", "sc_L5"):
        if arm in A:
            per_seed[f"BASE + {arm}"] = [load(world, "arrival", arm, kA, s) for s in SEEDS]
    detd = {f"standalone group rule (k={kA})": dict(val=rule["val"], test=rule["test"], zv=z0["val"], zt=z0["test"])}
    summ, _ = decision_arrival(world, per_seed, detd, snap_t)
    res["arrival"]["decisions_UC1"] = compact(summ)
    # ---------------------------------------------------------------- slices (Stage 6d)
    T = ks["snap"]["T_snap"]
    n4 = np.nan_to_num(Zt["A_L4"][:, GS.ASTATS.index("n")])
    sl = {"cold": n4 == 0, "thin": (n4 > 0) & (n4 < T), "normal": n4 >= T}
    best = max([a for a in ("L4", "L5", "sc_L5") if a in A], key=lambda a: A[a][0]["lateness_auc"][1]) if any(a in A for a in ("L4", "L5")) else None
    sres = {}
    if best:
        Rc = contracted_ref(world)["test"]
        for nm, m in sl.items():
            row = dict(rows=int(m.sum()), observed=int((m & EVt).sum()))
            if (m & EVt).sum() >= 50:
                for arm in ("base", best):
                    vals = []
                    for s in SEEDS:
                        z = load(world, "arrival", arm, kA, s)["test"]; P = np.asarray(z["P"], float)
                        yl, keep = uc1_label(Yt, EVt, Rc)
                        k2 = keep & m
                        vals.append(dict(lateness_auc=lateness(P[m], Yt[m], EVt[m], Rt[m])[0],
                                         a3=float(np.median(np.abs(7 * P[m & EVt] - 7 * Yt[m & EVt]))),
                                         uc1_p5=P20D.prec_at((P - Rc)[k2], yl[k2], 0.05) if k2.sum() >= 20 else float("nan")))
                    row[arm] = {key: S18.band([v[key] for v in vals]) for key in vals[0]}
                row["gain_vs_base"] = {key: cmp_slice(row[best][key], row["base"][key], key != "a3") for key in row["base"]}
            sres[nm] = row
        months = pd.DatetimeIndex(snap_t).month
        mres = {}
        for mo in np.unique(months):
            m = (months == mo)
            vals = {arm: [lateness(np.asarray(load(world, "arrival", arm, kA, s)["test"]["P"], float)[m], Yt[m], EVt[m], Rt[m])[0]
                          for s in SEEDS] for arm in ("base", best)}
            mres[int(mo)] = {arm: S18.band(v) for arm, v in vals.items()} | dict(rows=int(m.sum()))
        res["slices_arrival"] = dict(T=T, best_arm=best, by_n4=sres, by_month_of_t0=mres)
    # ---------------------------------------------------------------- fill (Stage 5)
    F = {}; UNF = {}
    for arm in ("base", "L4", "L4_xsh", "L5", "L5_perm", "L5_xsh", "ack", "ack_xsh", "L5_ack", "sc", "sc_L5"):
        if arm in ("base", "sc") or have(world, "fill", arm, kF):
            if arm == "sc" and not have(world, "fill", "sc", kF):
                continue
            b, p, u = bands(world, "fill", arm, kF); F[arm] = (b, p); UNF[arm] = u
    fb = F["base"][0]
    gf = {}
    for fam, ctrls in GATES["fill"].items():
        if fam in F and all(c in F for c in ctrls):
            gf[fam] = gate("fill", F[fam][0], fb, {c: F[c][0] for c in ctrls})
    if "sc" in F and "sc_L5" in F:
        gf["L5 over season + cadence"] = dict(vs_sc=cmp("fill", F["sc_L5"][0], F["sc"][0]))
    if "L5" in F and "L4" in F:
        gf["control (c): L5 vs L4"] = cmp("fill", F["L5"][0], F["L4"][0])
    zf = load(world, "fill", "base", kF, 7)
    Yf = {f: np.asarray(zf[f]["Y"], float) for f in ("val", "test")}
    sa = {f: GS.standalone_fill(Zf, kF) for f, Zf in (("val", Zv), ("test", Zt))}
    ensf = {arm: np.mean([np.asarray(load(world, "fill", arm, kF, s)["test"]["P"], float) for s in SEEDS], 0) for arm in F}
    predsf = {"standalone_dist": sa["test"], **{f"{a}_ens": v for a, v in ensf.items()}}
    detf = det_compare("fill", predsf, Yf["test"], None, None, blocks, 22,
                       [("standalone_dist", "base_ens")] + [(f"{a}_ens", "base_ens") for a in F if a != "base"])
    res["fill"] = dict(k=kF, bands={a: b for a, (b, _) in F.items()}, per_seed={a: p for a, (_, p) in F.items()},
                       unreachable={a: u for a, u in UNF.items() if u}, gates=gf, deterministic=detf)
    if "L5" in F:
        res["fill"]["importance_L5"] = importances(world, "fill", "L5", kF)
    if "L5_ack" in F:
        res["fill"]["importance_L5_ack"] = importances(world, "fill", "L5_ack", kF)
    psf = {"BASE (shipped b5flat22)": [load(world, "fill", "base", kF, s) for s in SEEDS]}
    for arm in ("L5", "ack", "L5_ack", "sc_L5"):
        if arm in F:
            psf[f"BASE + {arm}"] = [load(world, "fill", arm, kF, s) for s in SEEDS]
    res["fill"]["decisions"] = {uc: compact(v) for uc, v in decision_fill(psf, {f"standalone group distribution (k={kF})":
                                                                             dict(val=sa["val"], test=sa["test"], yv=Yf["val"], yt=Yf["test"])},
                                                                         snap_t).items()}
    out.update(res)


def cmp_slice(a, b, higher):
    return S18.compare(a, b, higher)


def stage_place(world, out, sfx=""):
    ks = ksel(world); kP = ks["k"]["place"]
    Z, idx, tau = fold_rows(world, "place")
    Zv, Zt = sub(Z, idx["val"]), sub(Z, idx["test"])
    P = {}; UN = {}
    for arm0 in ("base", "L4", "L5", "L5_perm"):
        arm = arm0 + sfx
        if have(world, "place", arm, kP):
            b, p, u = bands(world, "place", arm, kP); P[arm0] = (b, p); UN[arm0] = u
    z0 = load(world, "place", "base" + sfx, kP, 7)
    for f, Zf in (("val", Zv), ("test", Zt)):
        assert (np.asarray(z0[f]["entity"]).astype(str) == Zf["entity"]).all(), f"placement store rows != BASE rows ({f})"
    base_b = P["base"][0]
    g = {}
    for fam, ctrls in GATES["place"].items():
        if fam in P and all(c in P for c in ctrls):
            g[fam] = gate("place", P[fam][0], base_b, {c: P[c][0] for c in ctrls})
    if "L5" in P and "L4" in P:
        g["control (c) / P4: L5 (true month) vs L4"] = cmp("place", P["L5"][0], P["L4"][0])
    a_off = ks["placement"]["place"][str(kP)]["offset_a_weeks"]
    rule = {f: GS.standalone_arrival_weeks(Zf, kP) + a_off for f, Zf in (("val", Zv), ("test", Zt))}
    Yt, EVt, Rt = (np.asarray(z0["test"][x], float) for x in ("Y", "EV", "AUX"))
    EVt = EVt.astype(bool)
    tau_t = tau[idx["test"]]
    blocks = [np.flatnonzero(tau_t == u) for u in np.unique(tau_t)]
    ens = {arm: np.mean([np.asarray(load(world, "place", arm + sfx, kP, s)["test"]["P"], float) for s in SEEDS], 0) for arm in P}
    preds = {"rule": rule["test"], **{f"{a}_ens": v for a, v in ens.items()}}
    det = det_compare("place", preds, Yt, EVt, Rt, blocks, 23, [("rule", "base_ens")] + [(f"{a}_ens", "base_ens") for a in P if a != "base"]
                      + ([("L5_ens", "L4_ens")] if "L5" in P and "L4" in P else []))
    # base rates
    yl, keep = uc1p_label(Yt, EVt, Rt)
    br = dict(test_rows=int(len(Yt)), observed_share=float(EVt.mean()), late_vs_contract_observed=float((Yt[EVt] > Rt[EVt]).mean()),
              uc1p_base_rate=float(yl[keep].mean()), blocks_creation_weeks=len(blocks))
    month_t = pd.DatetimeIndex(tau_t).month.to_numpy()      # per-"snapshot" spread read per creation MONTH (12 blocks)
    per_seed = {"BASE (flat at creation)" + sfx: [load(world, "place", "base" + sfx, kP, s) for s in SEEDS]}
    for arm in ("L4", "L5"):
        if arm in P:
            per_seed[f"BASE + {arm}{sfx}"] = [load(world, "place", arm + sfx, kP, s) for s in SEEDS]
    detd = {f"standalone group rule (k={kP})": dict(val=rule["val"], test=rule["test"], zv=z0["val"], zt=z0["test"])}
    summ, _ = decision_arrival(world, per_seed, detd, month_t, place=True)
    imp = importances(world, "place", "L5" + sfx, kP) if "L5" in P else None
    # month slices at placement (true month)
    mres = {}
    for mo in range(1, 13):
        m = month_t == mo
        if m.sum():
            mres[mo] = {arm: S18.band([lateness(np.asarray(load(world, "place", arm + sfx, kP, s)["test"]["P"], float)[m], Yt[m], EVt[m], Rt[m])[0]
                                       for s in SEEDS]) for arm in P if arm in ("base", "L4", "L5")} | dict(rows=int(m.sum()))
    out["place" + sfx] = dict(k=kP, suffix=sfx, offset_a_weeks=a_off, base_rates=br, bands={a: b for a, (b, _) in P.items()},
                        per_seed={a: p for a, (_, p) in P.items()}, unreachable={a: u for a, u in UN.items() if u}, gates=g,
                        deterministic=det, decisions_UC1P=compact(summ), importance_L5=imp, by_creation_month=mres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["snap", "place"])
    ap.add_argument("--world", default="v8")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    PP.register()
    st = C.require_clean()
    out = dict(stamp=st, world=a.world)
    if a.stage == "snap":
        stage_snap(a.world, out)
    else:
        stage_place(a.world, out, a.suffix)
    C.dump(out, f"phase21/score_{a.stage}{a.suffix}_{a.world}.json")
    for task in ("arrival", "fill", "place", "place_nl"):
        if task in out:
            print(f"== {task}")
            for arm, b in out[task]["bands"].items():
                print(f"  {arm:10s}", {k: [round(x, 4) for x in v] if v else None for k, v in b.items() if k in DIR[task[:5] if task.startswith("place") else task]})
            for fam, gg in out[task]["gates"].items():
                print(f"  GATE {fam}: {gg.get('verdict', '')} {gg.get('why', gg)}")
            print("  DET", json.dumps(out[task]["deterministic"]["point"]))


if __name__ == "__main__":
    main()
