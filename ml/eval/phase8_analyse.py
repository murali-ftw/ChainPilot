"""Phase 8 — per-task tables and the four questions, read from the single scorer's output.

Inputs (all produced upstream, nothing re-scored here):
  ml/artifacts/backtest/phase8_scores.json    ml/eval/phase7_score.py --backtest   (test AND validation folds)
  ml/artifacts/backtest/bundle_index.json     ml/eval/backtest.py export            (drift, recalibration, epochs)
  ml/artifacts/backtest/phase8_origins.json   ml/eval/backtest.py plan              (surviving origins)
  ml/artifacts/backtest/preds/*.npz           observed label shift, validation -> evaluation window

Rules applied in code:
  * a difference exists only when the two configurations' seed ranges are DISJOINT (outside both measured bands);
    otherwise "not distinguishable"
  * bands are per metric, per configuration, per world, per origin -- never pooled or borrowed
  * selection evidence uses origins whose evaluation windows end before 2025 (1-6); origins 7-8 overlap the fixed
    split's 2025 test year and are reported as confirmation only

  python ml/eval/phase8_analyse.py [--preds DIR] [--scores FILE] [--index FILE]
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from scipy.stats import spearmanr
from config import ARTIFACTS, REPO

BT = os.path.join(ARTIFACTS, "backtest")
SEEDS = (7, 17, 27)
B5_SEEDS = (7, 17, 27, 37, 47)
ARR_G, ARR_0 = "lite_h4_lr0.00025", "none_h0_lr0.00025"
CAP_G, CAP_0 = "mp_h4_lr0.00025", "none_h0_lr0.00025"
FILL = "none_h0_lr0.000125"
B5 = "b5flat_q"
WORLDS = ("v6", "v7")


class Data:
    def __init__(self, scores, index, plan, preds):
        self.S = json.load(open(scores))["scores"]
        self.IX = json.load(open(index))
        self.plan = json.load(open(plan))
        self.preds = preds
        ship = json.load(open(os.path.join(REPO, "ml", "configs", "shipped.json")))
        self.bands_pp = ship["drift_bands_pp"]
        self.threshold = ship["tasks"]["arrival_week"]["fallback"]["threshold_pp"]
        surv = {}
        for r in self.plan["origins"]:
            surv[r["origin"]] = surv.get(r["origin"], True) and r["survives"]
        self.origins = [k for k in sorted(surv) if surv[k]]
        self.pre2025 = [k for k in self.origins if k <= 6]

    # ---------------------------------------------------------------- access
    def get(self, w, task, label, metric, fold="test"):
        v = self.S.get(f"{w}|{task}|{'VAL_' if fold == 'val' else ''}{label}", {}).get(metric)
        return None if v is None else float(v[0])

    def seeds(self, w, task, pre, k, cfg, metric, ss=SEEDS, fold="test"):
        return [x for x in (self.get(w, task, f"{pre}o{k}_{cfg}_s{s}", metric, fold) for s in ss) if x is not None]

    def drift(self, w, task, k, cfg, s, stat):
        e = self.IX.get(f"{w}|{task}|o{k}_{cfg}_s{s}")
        return None if e is None else 100.0 * (e["drift_test"][stat] - e["drift_val"][stat])

    def observed_shift(self, w, task, k, cfg):
        f = lambda fold: os.path.join(self.preds, f"{w}_{task}_o{k}_{cfg}_s7_{fold}.npz")
        if not (os.path.exists(f("val")) and os.path.exists(f("test"))):
            return None
        zv, zt = np.load(f("val")), np.load(f("test"))
        if task == "arrival_week":
            s = lambda z: float((~z["EV"].astype(bool)).mean())
        elif task == "fill_rate":
            s = lambda z: float((z["Y"] >= 1).mean())
        else:
            s = lambda z: float(z["Y"].mean())
        return 100.0 * (s(zt) - s(zv))


def band(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), mean=float(np.mean(xs)), spread=float(np.ptp(xs)), min=float(min(xs)), max=float(max(xs)))


def compare(a, b, lower_better):
    """Configuration a against b from their seed values. A difference only when the ranges are disjoint."""
    A, B = band(a), band(b)
    if not A or not B:
        return dict(verdict="missing", a=A, b=B)
    if A["n"] < 2 or B["n"] < 2:
        # one value has no spread: "disjoint" would be trivially true. No band, no comparison (brief rule).
        return dict(verdict="band not measured", a=A, b=B, margin=A["mean"] - B["mean"])
    margin = A["mean"] - B["mean"]
    if A["max"] < B["min"]:
        better = "a" if lower_better else "b"
    elif B["max"] < A["min"]:
        better = "b" if lower_better else "a"
    else:
        better = None
    ratio = lambda sp: (abs(margin) / sp) if sp > 0 else None
    return dict(a=A, b=B, margin=margin, margin_over_a_spread=ratio(A["spread"]), margin_over_b_spread=ratio(B["spread"]),
                verdict="not distinguishable" if better is None else f"{better} better")


def dist(xs):
    xs = [x for x in xs if x is not None]
    return dict(n=len(xs), min=float(min(xs)), median=float(np.median(xs)), max=float(max(xs))) if xs else None


def tally(verdicts):
    return {v: sum(1 for x in verdicts if x == v)
            for v in ("a better", "not distinguishable", "b better", "band not measured", "missing")}


def status(excess, B):
    if excess <= B["usable"]:
        return "usable (<= %.1f)" % B["usable"]
    if excess <= B["watch"]:
        return "watch (%.1f-%.1f, interpolated in Phase 6)" % (B["usable"], B["watch"])
    if excess <= B["degraded"]:
        return "degraded (%.1f-%.1f)" % (B["watch"], B["degraded"])
    return "unusable (> %.1f)" % B["degraded"]


# ================================================================== §3 tables
def tables(D):
    T = {"arrival": [], "fill": [], "capacity": [], "recalibration": []}
    for w in WORLDS:
        for k in D.origins:
            for cfg, lab in ((ARR_G, "shipped SHARE-lite h4"), (ARR_0, "h0")):
                T["arrival"].append(dict(world=w, origin=k, config=lab,
                                         cindex=band(D.seeds(w, "arrival_week", "", k, cfg, "cindex")),
                                         lateness_roc=band(D.seeds(w, "arrival_week", "", k, cfg, "roc_auc_late")),
                                         week_ece_recal=band(D.seeds(w, "arrival_week", "RECAL_", k, cfg, "ece_week"))))
            T["arrival"].append(dict(world=w, origin=k, config="promise date only",
                                     cindex=band([D.get(w, "arrival_week", f"PROMISE_o{k}_promise_only", "cindex")]),
                                     lateness_roc=band([D.get(w, "arrival_week", f"PROMISE_o{k}_promise_only", "roc_auc_late")])))
            T["fill"].append(dict(world=w, origin=k, config="shipped h0 (recalibrated)",
                                  crps_exact=band(D.seeds(w, "fill_rate", "RECAL_", k, FILL, "crps_exact")),
                                  ece20=band(D.seeds(w, "fill_rate", "RECAL_", k, FILL, "ece20")),
                                  rel_one=band(D.seeds(w, "fill_rate", "RECAL_", k, FILL, "rel_one")),
                                  crps_exact_raw=band(D.seeds(w, "fill_rate", "", k, FILL, "crps_exact")),
                                  ece20_raw=band(D.seeds(w, "fill_rate", "", k, FILL, "ece20"))))
            for cfg, lab, ss in ((CAP_G, "shipped HeteroMP h4", SEEDS), (CAP_0, "h0", SEEDS), (B5, "B5 LightGBM", B5_SEEDS)):
                row = dict(world=w, origin=k, config=lab)
                for m in ("pinball_10", "pinball_50", "pinball_90", "pinball_mean", "exceed_p90", "below_p10", "coverage80"):
                    row[m] = band(D.seeds(w, "capacity_strain", "", k, cfg, m, ss=ss))
                T["capacity"].append(row)
            for task, cfg in (("fill_rate", FILL), ("arrival_week", ARR_G), ("arrival_week", ARR_0)):
                for s in SEEDS:
                    e = D.IX.get(f"{w}|{task}|o{k}_{cfg}_s{s}")
                    if e:
                        r = e["recalibration"]
                        T["recalibration"].append(dict(world=w, origin=k, task=task, config=cfg, seed=s, method=r.get("method"),
                                                       vs_T=r.get("vs_T"), mm_ratio_range=(None if not r.get("mm_ratio") else
                                                       [float(min(r["mm_ratio"])), float(max(r["mm_ratio"]))]),
                                                       epochs=e["epochs"], stop=e["stop"], stamp=e["stamp"]))
    return T


def across_origins(D):
    """min / median / max of the 3-seed MEAN across origins, and how many origins each configuration wins."""
    out = {}
    specs = [("arrival_week", "", ARR_G, ARR_0, "cindex", False), ("arrival_week", "RECAL_", ARR_G, ARR_0, "ece_week", True),
             ("capacity_strain", "", CAP_G, CAP_0, "pinball_mean", True), ("capacity_strain", "", CAP_G, CAP_0, "exceed_p90", None)]
    for w in WORLDS:
        for task, pre, g, z, m, lower in specs:
            mg = [band(D.seeds(w, task, pre, k, g, m)) for k in D.origins]
            m0 = [band(D.seeds(w, task, pre, k, z, m)) for k in D.origins]
            row = dict(graph=dist([b["mean"] for b in mg if b]), h0=dist([b["mean"] for b in m0 if b]))
            if lower is not None:
                v = [compare(D.seeds(w, task, pre, k, g, m), D.seeds(w, task, pre, k, z, m), lower)["verdict"] for k in D.origins]
                row["per_origin"] = dict(zip(D.origins, v)); row["tally_graph_vs_h0"] = tally(v)
            out[f"{w}|{task}|{pre}{m}"] = row
        for m in ("crps_exact", "ece20", "rel_one"):
            out[f"{w}|fill_rate|RECAL_{m}"] = dict(h0=dist([(band(D.seeds(w, "fill_rate", "RECAL_", k, FILL, m)) or {}).get("mean") for k in D.origins]))
    return out


# ================================================================== 3a capacity depth
def q3a(D):
    out = {}
    for w in WORLDS:
        rows = []
        for k in D.origins:
            g = D.seeds(w, "capacity_strain", "", k, CAP_G, "pinball_mean")
            z = D.seeds(w, "capacity_strain", "", k, CAP_0, "pinball_mean")
            b5 = D.seeds(w, "capacity_strain", "", k, B5, "pinball_mean", ss=B5_SEEDS)
            gv = D.seeds(w, "capacity_strain", "", k, CAP_G, "pinball_mean", fold="val")
            zv = D.seeds(w, "capacity_strain", "", k, CAP_0, "pinball_mean", fold="val")
            obs = D.observed_shift(w, "capacity_strain", k, CAP_G)
            dg = [D.drift(w, "capacity_strain", k, CAP_G, s, "q50_mean") for s in SEEDS]
            dz = [D.drift(w, "capacity_strain", k, CAP_0, s, "q50_mean") for s in SEEDS]
            sign = lambda d: None if (d is None or obs is None) else ("with" if np.sign(d) == np.sign(obs) else "against")
            rows.append(dict(origin=k, pre_2025=k in D.pre2025,
                             h4_vs_h0_test=compare(g, z, True), h4_vs_h0_validation=compare(gv, zv, True),
                             h4_vs_b5_test=compare(g, b5, True), h0_vs_b5_test=compare(z, b5, True),
                             observed_label_shift_hundredths=obs, h4_q50_drift=dg, h0_q50_drift=dz,
                             h4_drift_sign=[sign(d) for d in dg], h0_drift_sign=[sign(d) for d in dz]))
        summ = lambda key, sel: tally([r[key]["verdict"] for r in rows if sel(r)])
        out[w] = dict(rows=rows,
                      h4_vs_h0_all=summ("h4_vs_h0_test", lambda r: True), h4_vs_h0_pre2025=summ("h4_vs_h0_test", lambda r: r["pre_2025"]),
                      h4_vs_h0_validation_all=summ("h4_vs_h0_validation", lambda r: True),
                      h4_vs_b5_all=summ("h4_vs_b5_test", lambda r: True), h0_vs_b5_all=summ("h0_vs_b5_test", lambda r: True),
                      h4_drift_against=sum(x == "against" for r in rows for x in r["h4_drift_sign"]),
                      h4_drift_with=sum(x == "with" for r in rows for x in r["h4_drift_sign"]),
                      h0_drift_against=sum(x == "against" for r in rows for x in r["h0_drift_sign"]),
                      h0_drift_with=sum(x == "with" for r in rows for x in r["h0_drift_sign"]),
                      origins_where_every_h4_seed_drifts_against=[r["origin"] for r in rows if r["h4_drift_sign"] and all(x == "against" for x in r["h4_drift_sign"])])
    return out


# ================================================================== 3b drift thresholds per task
def q3b(D):
    B = D.bands_pp
    obs = {"arrival_week": [], "capacity_strain": [], "fill_rate": []}
    for w in WORLDS:
        for k in D.origins:
            for s in SEEDS:
                dg, dz = D.drift(w, "arrival_week", k, ARR_G, s, "p_late_raw"), D.drift(w, "arrival_week", k, ARR_0, s, "p_late_raw")
                eg = D.get(w, "arrival_week", f"RECAL_o{k}_{ARR_G}_s{s}", "ece_week")
                ez = D.get(w, "arrival_week", f"RECAL_o{k}_{ARR_0}_s{s}", "ece_week")
                if None not in (dg, dz, eg, ez):
                    obs["arrival_week"].append(dict(world=w, origin=k, seed=s, excess=abs(dg - dz), damage=eg - ez, damage_abs=eg))
                dg, dz = D.drift(w, "capacity_strain", k, CAP_G, s, "q50_mean"), D.drift(w, "capacity_strain", k, CAP_0, s, "q50_mean")
                pg = D.get(w, "capacity_strain", f"o{k}_{CAP_G}_s{s}", "pinball_mean")
                pz = D.get(w, "capacity_strain", f"o{k}_{CAP_0}_s{s}", "pinball_mean")
                if None not in (dg, dz, pg, pz):
                    obs["capacity_strain"].append(dict(world=w, origin=k, seed=s, excess=abs(dg - dz), damage=pg - pz, damage_abs=pg))
                df = D.drift(w, "fill_rate", k, FILL, s, "p_complete_raw")
                ef = D.get(w, "fill_rate", f"RECAL_o{k}_{FILL}_s{s}", "ece20")
                if None not in (df, ef):
                    obs["fill_rate"].append(dict(world=w, origin=k, seed=s, excess=abs(df), damage=ef, damage_abs=ef,
                                                 note="no h0 reference: raw |drift|, damage = recalibrated ECE 20"))
    out = {}
    for task, L in obs.items():
        if not L:
            out[task] = dict(n=0); continue
        x = np.array([o["excess"] for o in L]); y = np.array([o["damage"] for o in L])
        per = {}
        for o in L:
            per.setdefault(status(o["excess"], B), []).append(o["damage"])
        rho = spearmanr(x, y)
        slope = float(np.polyfit(x, y, 1)[0]) if len(set(x)) > 1 else None
        # the refit: the smallest observed excess above which damage is positive on every observation
        xs = np.sort(np.unique(x))
        refit = next((float(t) for t in xs if (y[x >= t] > 0).all()), None)
        out[task] = dict(n=len(L), observations=L,
                         per_band={b: dict(n=len(v), median_damage=float(np.median(v)), min=float(min(v)), max=float(max(v)))
                                   for b, v in sorted(per.items())},
                         bands_with_no_observation=[b for b in (status(0.0, B), status((B["usable"] + B["watch"]) / 2, B),
                                                                status((B["watch"] + B["degraded"]) / 2, B), status(B["degraded"] + 1, B))
                                                    if b not in per],
                         spearman=dict(rho=float(rho[0]), p=float(rho[1])), slope_damage_per_pp=slope,
                         refit_threshold_all_damaged_above=refit, excess_range=[float(x.min()), float(x.max())])
    return out


# ================================================================== 3c drift gate vs always-h0
def q3c(D):
    out = {}
    for w in WORLDS:
        rows = []
        for k in D.origins:
            gated, always, engaged, excess = [], [], [], []
            for s in SEEDS:
                dg, dz = D.drift(w, "arrival_week", k, ARR_G, s, "p_late_raw"), D.drift(w, "arrival_week", k, ARR_0, s, "p_late_raw")
                eg = D.get(w, "arrival_week", f"RECAL_o{k}_{ARR_G}_s{s}", "ece_week")
                ez = D.get(w, "arrival_week", f"RECAL_o{k}_{ARR_0}_s{s}", "ece_week")
                if None in (dg, dz, eg, ez):
                    continue
                ex = abs(dg - dz); on = ex > D.threshold
                excess.append(ex); engaged.append(on)
                gated.append(ez if on else eg); always.append(ez)
            rows.append(dict(origin=k, pre_2025=k in D.pre2025, excess_pp=excess, gate_engaged=engaged,
                             gated_week_ece=band(gated), always_h0_week_ece=band(always),
                             always_h0_vs_gated=compare(always, gated, True),
                             ranking_cindex_shipped=band(D.seeds(w, "arrival_week", "", k, ARR_G, "cindex"))))
        out[w] = dict(rows=rows, tally_always_h0_vs_gated=tally([r["always_h0_vs_gated"]["verdict"] for r in rows]),
                      gate_engagements=sum(sum(r["gate_engaged"]) for r in rows),
                      gate_decisions=sum(len(r["gate_engaged"]) for r in rows),
                      origins_gate_split_across_seeds=[r["origin"] for r in rows if 0 < sum(r["gate_engaged"]) < len(r["gate_engaged"])])
    return out


# ================================================================== 3d(ii) lateness beyond the promise
def q3d(D):
    out = {}
    for w in WORLDS:
        rows = []
        for k in D.origins:
            p_roc = D.get(w, "arrival_week", f"PROMISE_o{k}_promise_only", "roc_auc_late")
            p_c = D.get(w, "arrival_week", f"PROMISE_o{k}_promise_only", "cindex")
            h = D.seeds(w, "arrival_week", "", k, ARR_G, "roc_auc_late")
            hc = D.seeds(w, "arrival_week", "", k, ARR_G, "cindex")
            z = D.seeds(w, "arrival_week", "", k, ARR_0, "roc_auc_late")
            hb = band(h)
            rows.append(dict(origin=k, promise_roc=p_roc, head_roc=hb, h0_roc=band(z),
                             head_minus_promise=(hb["mean"] - p_roc) if hb and p_roc is not None else None,
                             every_head_seed_above_promise=bool(h) and p_roc is not None and min(h) > p_roc,
                             margin_over_head_spread=((hb["mean"] - p_roc) / hb["spread"]) if hb and hb["spread"] > 0 and p_roc is not None else None,
                             promise_cindex=p_c, head_cindex=band(hc),
                             head_outranks_promise_cindex=bool(hc) and p_c is not None and min(hc) > p_c,
                             head_vs_h0_lateness=compare(h, z, False)))
        out[w] = dict(rows=rows, origins_head_beats_promise_lateness=sum(r["every_head_seed_above_promise"] for r in rows),
                      origins_head_outranks_promise_cindex=sum(r["head_outranks_promise_cindex"] for r in rows),
                      head_minus_promise=dist([r["head_minus_promise"] for r in rows]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(BT, "phase8_scores.json"))
    ap.add_argument("--index", default=os.path.join(BT, "bundle_index.json"))
    ap.add_argument("--plan", default=os.path.join(BT, "phase8_origins.json"))
    ap.add_argument("--preds", default=os.path.join(BT, "preds"))
    ap.add_argument("--out", default=os.path.join(BT, "phase8_analysis.json"))
    a = ap.parse_args()
    D = Data(a.scores, a.index, a.plan, a.preds)
    R = dict(origins=D.origins, pre_2025_origins=D.pre2025, tables=tables(D), across_origins=across_origins(D),
             q3a_capacity_depth=q3a(D), q3b_drift_thresholds=q3b(D), q3c_gate_vs_always_h0=q3c(D), q3d_lateness_beyond_promise=q3d(D))
    json.dump(R, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(dict(origins=D.origins,
                          q3a={w: {k: v for k, v in R["q3a_capacity_depth"][w].items() if k != "rows"} for w in WORLDS},
                          q3b={t: {k: v for k, v in R["q3b_drift_thresholds"][t].items() if k != "observations"} for t in R["q3b_drift_thresholds"]},
                          q3c={w: {k: v for k, v in R["q3c_gate_vs_always_h0"][w].items() if k != "rows"} for w in WORLDS},
                          q3d={w: {k: v for k, v in R["q3d_lateness_beyond_promise"][w].items() if k != "rows"} for w in WORLDS}),
                     indent=1, default=str))
    print(f"-> {a.out}")
