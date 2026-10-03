"""Phase 20 scorer for Stages 2-4 (non-privileged arms). Torch process, no LightGBM. Exact per-seed filenames, no globs.

World-aware: every world is scored against ITS OWN lateness reference (docs/specs/lateness_metric.md, built from that world's
receipts, offset c fitted on that world's training rows) and ITS OWN BASE. Bands need all five seeds (Phase 19 deviation 171).
Metric functions are Phase 18's (fill_metrics, capacity_metrics) or Phase 18's arrival definitions with the world's reference.

  Stage 2  Gate v2 on the second world (fwd_season / fwd_load supplier-specific / cadence with Phase 19's controls); the
           replication verdict per v8-1001 passing cell and metric (P3: same sign + disjoint vs BASE; P4: gain ratio in
           [0.5, 2]); the LightGBM-only blend recipe (lgbm_rf with LightGBM + fwd_load, weight on validation), snapshot-block
           bootstrap against each parent -- on both worlds.
  Stage 3  forecast-feature gate (BASE+fwd_load+fwd_pred vs BASE+fwd_load; permuted fwd_pred control), both worlds.
  Stage 4c legitimate stock arm gate (BASE+stock vs BASE; snapshot-permuted control), v8.

  python ml/eval/phase20_score.py      # -> ml/artifacts/phase20/scores.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np
import phase20_world as PW
import phase5_heads as P5, folds
from phase11b_lateness import build_reference, lateness
from metrics import cindex
import phase18_score as S18

SEEDS = C.V8_SEEDS
B = 1000
GRID = np.linspace(0, 1, 101)
PR18, PR19, PR20 = (os.path.join(C.ART, p, "preds") for p in ("phase18", "phase19", "phase20"))
TASKS = S18.TASK
D = S18.DIRECTION
CELLS = [("arrival", "fwd_season"), ("arrival", "fwd_load_supplier_specific"), ("arrival", "cadence"),
         ("fill", "fwd_season"), ("fill", "cadence"), ("capacity", "fwd_load_supplier_specific")]
_REF = {}


def ref(world, fold):
    if world not in _REF:
        lb = P5.labels(world, "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
        R, meta = build_reference("asof_channel_lead", world, lb, tr)
        _REF[world] = dict(val=R[P5.ordered(lb, va)], test=R[P5.ordered(lb, te)], meta={k: v for k, v in meta.items() if not isinstance(v, (list, dict))},
                           snap_test=lb.snapshot_date.values[P5.ordered(lb, te)])
    return _REF[world][fold]


def snaps_test(world, task):
    lb = P5.labels(world, TASKS[task]); tr, va, te = folds.fixed_split(lb.snapshot_date)
    return lb.snapshot_date.values[P5.ordered(lb, te)]


def metrics(world, task, zz):
    if task == "arrival":
        z = zz["test"]; R = ref(world, "test"); Y, EV = np.asarray(z["Y"], float), np.asarray(z["EV"], bool)
        ET = S18.point("arrival", z)
        return dict(lateness_auc=lateness(ET, Y, EV, R)[0], a3_median_abs_err_days=float(np.median(np.abs(7 * ET[EV] - 7 * Y[EV]))),
                    cindex=float(cindex(ET, Y, EV)))
    if task == "fill":
        return S18.fill_metrics(zz["test"]["P"], zz["test"]["Y"])
    return S18.capacity_metrics(zz["val"]["P"], zz["val"]["Y"], zz["test"]["P"], zz["test"]["Y"])


def fmt(world, task, arm):
    """Path template of an arm's per-seed predictions. v8's Phase 18-19 arms are reused from their stored files."""
    t = TASKS[task]
    if world == "v8":
        stored = {"base": os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz"),
                  "fwd_load": os.path.join(PR18, f"v8_{t}_p18_fwd_load_s{{s}}_{{f}}.npz"),
                  **{a: os.path.join(PR19, f"v8_{t}_p19_{a}_s{{s}}_{{f}}.npz")
                     for a in ("fwd_season", "fwd_season_sperm", "fwd_load_sperm", "cadence", "cadence_shuf")}}
        if arm in stored:
            return stored[arm]
    return os.path.join(PR20, f"{world}_{t}_p20_{arm}_s{{s}}_{{f}}.npz")


def load(world, task, arm, s):
    return {f: dict(np.load(fmt(world, task, arm).format(s=s, f=f))) for f in ("val", "test")}


def bands(world, task, arm):
    per = [metrics(world, task, load(world, task, arm, s)) for s in SEEDS]
    out, unreach = {}, {}
    for k in per[0]:
        v = [p[k] for p in per]; ok = sum(x is not None and np.isfinite(x) for x in v)
        out[k] = S18.band(v) if ok == len(SEEDS) else None
        if ok < len(SEEDS):
            unreach[k] = f"UNREACHABLE for {len(SEEDS) - ok} of 5 seeds"
    return out, per, unreach


def cmp(task, a, b):
    return {k: S18.compare(a[k], b[k], h) for k, h in D[task].items()}


def gate(task, fam_vs_ref, ctrl_vs_base):
    better = [k for k, v in fam_vs_ref.items() if v == "better"]; worse = [k for k, v in fam_vs_ref.items() if v == "worse"]
    ctrl = [k for k, v in ctrl_vs_base.items() if v == "better"]
    need = 3 if task == "capacity" else 1
    return dict(verdict="PASS" if (len(better) >= need and not worse and not ctrl) else "FAIL", better=better, worse=worse,
                control_better_than_base=ctrl, family_vs_reference=fam_vs_ref, control_vs_base=ctrl_vs_base)


# ================================================================== the LightGBM-only blend recipe, snapshot-block bootstrap
def blend_recipe(world, task):
    A = [load(world, task, "lgbm_rf", s) for s in SEEDS]; L = [load(world, task, "fwd_load", s) for s in SEEDS]
    P = {}
    for f in ("val", "test"):
        a = np.mean([S18.point(task, z[f]) if task == "arrival" else np.asarray(z[f]["P"], float) for z in A], 0)
        l = np.mean([S18.point(task, z[f]) if task == "arrival" else np.asarray(z[f]["P"], float) for z in L], 0)
        P[f] = dict(a=a, l=l, Y=np.asarray(A[0][f]["Y"], float), EV=np.asarray(A[0][f]["EV"], bool))
    import phase5_metrics as M
    from metrics import roc_auc
    v = P["val"]
    if task == "arrival":
        R = ref(world, "val"); m = v["EV"]; yl = (v["Y"][m] > R[m]).astype(int)
        crit = [-roc_auc(yl, (w * v["a"] + (1 - w) * v["l"])[m] - R[m]) for w in GRID]
    elif task == "fill":
        crit = [M.crps_exact_rows(w * v["a"] + (1 - w) * v["l"], v["Y"]).mean() for w in GRID]
    else:
        crit = [np.mean([M.pinball_rows(v["Y"], (w * v["a"] + (1 - w) * v["l"])[:, i], q).mean() for i, q in enumerate(M.QS)]) for w in GRID]
    w = float(GRID[int(np.argmin(crit))])
    t = P["test"]; t["blend"] = w * t["a"] + (1 - w) * t["l"]
    snap = snaps_test(world, task); blocks = [np.flatnonzero(snap == u) for u in np.unique(snap)]
    Y, EV = t["Y"], t["EV"]
    if task == "arrival":
        R = ref(world, "test")
        fn = lambda x, i: roc_auc((Y[i][EV[i]] > R[i][EV[i]]).astype(int), x[i][EV[i]] - R[i][EV[i]]); name, hi = "lateness_auc", True
    elif task == "fill":
        rows = {k: M.crps_exact_rows(t[k], Y) for k in ("a", "l", "blend")}
        fn = lambda x, i: None; name, hi = "crps_exact", False
    else:
        from phase14_score import p_exceed
        lt = (Y > 1).astype(int)
        def fn(x, i):
            s = p_exceed(x[i]); y = lt[i]; mm = max(1, int(np.ceil(0.05 * len(y)))); return float(y[np.argsort(-s, kind="stable")[:mm]].mean())
        name, hi = "precision_at_5pct", True
    rng = np.random.default_rng(77); d = {"blend vs lgbm_rf": [], "blend vs lgbm_fwd_load": []}
    for _ in range(B):
        i = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        if task == "fill":
            v_ = {k: float(rows[k][i].mean()) for k in rows}
        else:
            v_ = {k: fn(t[k], i) for k in ("a", "l", "blend")}
        sgn = 1 if hi else -1
        d["blend vs lgbm_rf"].append(sgn * (v_["blend"] - v_["a"])); d["blend vs lgbm_fwd_load"].append(sgn * (v_["blend"] - v_["l"]))
    full = np.arange(len(Y))
    point = ({k: float(rows[k].mean()) for k in rows} if task == "fill" else {k: fn(t[k], full) for k in ("a", "l", "blend")})
    res = dict(metric=name, weight_on_lgbm_rf=w, point={"lgbm_rf_ens": point["a"], "lgbm_fwd_load_ens": point["l"], "blend": point["blend"]})
    for k, x in d.items():
        lo, hi_ = np.percentile(x, [2.5, 97.5]); res[k] = dict(ci_better_positive=[float(lo), float(np.mean(x)), float(hi_)],
                                                              verdict="better" if lo > 0 else "worse" if hi_ < 0 else "undetermined")
    return res


def main():
    st = C.require_clean()
    PW.register()
    out = dict(stamp=st, references={}, worlds={}, replication={}, blend_recipe={}, stage3={}, stage4c={})
    v8gate = json.load(open(os.path.join(C.ART, "phase19", "gate_v2.json")))
    for world in ("v8", PW.WORLD):
        W = {}
        for task in TASKS:
            arms = ["base", "fwd_season", "fwd_load", "cadence", "fwd_season_sperm", "fwd_load_sperm", "cadence_shuf", "lgbm_rf",
                    "fwd_load_pred", "fwd_load_pred_sperm"] + (["stock", "stock_sperm"] if world == "v8" else [])
            A = {}
            for a in arms:
                if all(os.path.exists(fmt(world, task, a).format(s=s, f="test")) for s in SEEDS):
                    b, p, u = bands(world, task, a); A[a] = dict(bands=b, per_seed=p, unreachable=u)
            G = {}
            B_ = {a: v["bands"] for a, v in A.items()}
            if {"fwd_season", "fwd_season_sperm"} <= set(B_):
                G["fwd_season"] = gate(task, cmp(task, B_["fwd_season"], B_["base"]), cmp(task, B_["fwd_season_sperm"], B_["base"]))
            if {"fwd_load", "fwd_season", "fwd_load_sperm"} <= set(B_):
                G["fwd_load_supplier_specific"] = gate(task, cmp(task, B_["fwd_load"], B_["fwd_season"]), cmp(task, B_["fwd_load_sperm"], B_["base"]))
            if {"cadence", "cadence_shuf"} <= set(B_):
                G["cadence"] = gate(task, cmp(task, B_["cadence"], B_["base"]), cmp(task, B_["cadence_shuf"], B_["base"]))
            if {"fwd_load_pred", "fwd_load_pred_sperm"} <= set(B_):
                out["stage3"][f"{world}|{task}"] = gate(task, cmp(task, B_["fwd_load_pred"], B_["fwd_load"]), cmp(task, B_["fwd_load_pred_sperm"], B_["fwd_load"]))
            if {"stock", "stock_sperm"} <= set(B_):
                out["stage4c"][f"{world}|{task}"] = gate(task, cmp(task, B_["stock"], B_["base"]), cmp(task, B_["stock_sperm"], B_["base"]))
            W[task] = dict(arms=A, gate_v2=G)
            if "lgbm_rf" in B_ and "fwd_load" in B_:
                out["blend_recipe"][f"{world}|{task}"] = blend_recipe(world, task)
        out["worlds"][world] = W
        out["references"][world] = ref(world, "meta") if False else _REF.get(world, {}).get("meta")
    # ---- replication: every v8-1001 Gate v2 PASS cell, on each metric that was disjointly better there
    w2 = out["worlds"][PW.WORLD]
    for task, fam in CELLS:
        g = v8gate["gate_v2"][f"{task}|{fam}"]
        assert g["verdict"] == "PASS", f"{task}|{fam} did not pass in v8"
        mets = [k for k, v in g["family_vs_reference"].items() if v == "better"]
        arm = "fwd_load" if fam == "fwd_load_supplier_specific" else fam
        v8b = {a: v8gate["arms"][task][a]["bands"] for a in ("base", arm, "fwd_season")}
        nb = {a: w2[task]["arms"][a]["bands"] for a in ("base", arm, "fwd_season")}
        rows = {}
        for k in mets:
            hi = D[task][k]; sgn = 1 if hi else -1
            g8 = sgn * (v8b[arm][k][1] - v8b["base"][k][1]); g2 = sgn * (nb[arm][k][1] - nb["base"][k][1]) if nb[arm][k] and nb["base"][k] else None
            c2 = S18.compare(nb[arm][k], nb["base"][k], hi) if nb[arm][k] and nb["base"][k] else "n/a"
            ref2 = S18.compare(nb["fwd_load"][k], nb["fwd_season"][k], hi) if fam == "fwd_load_supplier_specific" else None
            rows[k] = dict(v8_gain_vs_base=g8, w2_gain_vs_base=g2, w2_vs_base=c2,
                           replicates="REPLICATES" if (c2 == "better") else ("DOES NOT" if c2 == "worse" or (g2 is not None and g2 <= 0) else "UNDETERMINED"),
                           ratio_w2_over_v8=(g2 / g8 if (g2 is not None and g8) else None),
                           within_factor_2=bool(g2 is not None and g8 and 0.5 <= g2 / g8 <= 2.0),
                           w2_gate_reference_fwd_load_vs_fwd_season=ref2)
        out["replication"][f"{task}|{fam}"] = dict(metrics=rows, w2_gate_v2=w2[task]["gate_v2"].get(fam, {}).get("verdict"))
    out["references"] = {w: v.get("meta") for w, v in _REF.items()}
    C.dump(out, "phase20/scores.json")
    for k, v in out["replication"].items():
        print(k, "w2 Gate v2:", v["w2_gate_v2"], {m: (r["replicates"], round(r["v8_gain_vs_base"], 4), r["w2_gain_vs_base"] and round(r["w2_gain_vs_base"], 4),
                                                       r["ratio_w2_over_v8"] and round(r["ratio_w2_over_v8"], 2)) for m, r in v["metrics"].items()})
    for k, v in out["blend_recipe"].items():
        print("blend", k, v["metric"], {a: round(x, 4) for a, x in v["point"].items()}, "w_rf", v["weight_on_lgbm_rf"],
              v["blend vs lgbm_rf"]["verdict"], v["blend vs lgbm_fwd_load"]["verdict"])
    for k, v in out["stage3"].items():
        print("stage3", k, v["verdict"], v["better"], v["worse"], "ctrl", v["control_better_than_base"])
    for k, v in out["stage4c"].items():
        print("stage4c", k, v["verdict"], v["better"], v["worse"], "ctrl", v["control_better_than_base"])


if __name__ == "__main__":
    main()
