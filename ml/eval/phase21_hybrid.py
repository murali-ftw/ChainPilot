"""Phase 21 Stage 6a-c -- hybrids of the stored neural ensembles with the best LightGBM + group arm, and the Phase 15
classification. Torch process (imports only), no LightGBM, no training. Weights on VALIDATION only, frozen for test.

  arrival  blend of three expected-week predictors, weights on the simplex (step 0.05), max validation lateness AUC:
           incumbent neural ensemble (h4), Phase 19 neural ensemble (rffwdload+season+cadence), best LightGBM + group arm's
           5-seed mean (best by VALIDATION lateness AUC among L4, L5, sc_L5)
  fill     w * Phase 19 neural season + cadence ensemble + (1 - w) * best LightGBM + group fill arm's 5-seed mean (P22),
           w on a 0.01 grid by validation exact CRPS (best arm by validation CRPS among L4, L5, ack, L5_ack, sc_L5)
Compared with both parents, the incumbent ensemble and the Phase 19 blend (Phase 19's validation weights, Phase 20 rule 4)
on the snapshot-block bootstrap (1,000 resamples of whole test snapshots, paired), then through Phase 15 (summarise),
with precision in the top 5% reported apart from the AUC.

  python ml/eval/phase21_hybrid.py      -> ml/artifacts/phase21/stage6_hybrid.json
"""
from __future__ import annotations
import os, json, itertools
import phase12_common as C
import numpy as np, pandas as pd
import phase21_paths as PP
import phase5_heads as P5, folds
import phase5_metrics as M
from metrics import roc_auc
from phase14_score import p_late
import phase18_score as S18
import phase20_decisions as P20D
import phase20_score as P20S
import phase21_score as S21

SEEDS = C.V8_SEEDS
W19 = json.load(open(os.path.join(C.ART, "phase19", "stage4b_block.json")))["tasks"]
BUND19 = os.path.join(C.ART, "phase19", "bundles")


def b19(path, f):
    return dict(np.load(os.path.join(BUND19, path, f"preds_{f}.npz")))


def mean(zs, k):
    return np.mean([np.asarray(z[k], float) for z in zs], 0)


def ew(zs):
    return S18.expected_week({"S": mean(zs, "S"), "pT": mean(zs, "pT")})


def arrival(ks):
    kA = ks["k"]["arrival"]
    inc = {f: [S18.load_neural("arrival", s)[f] for s in SEEDS] for f in ("val", "test")}
    rf = {f: [b19(f"arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence", f) for s in SEEDS] for f in ("val", "test")}
    lg = {f: [dict(np.load(os.path.join(S18.PR18, f"v8_arrival_week_p18_fwd_load_s{s}_{f}.npz"))) for s in SEEDS] for f in ("val", "test")}
    Rv, Rt = P20S.ref("v8", "val"), P20S.ref("v8", "test")
    Y = {f: np.asarray(inc[f][0]["Y"], float) for f in ("val", "test")}; EV = {f: np.asarray(inc[f][0]["EV"], bool) for f in ("val", "test")}
    def vauc(p):
        m = EV["val"]; return roc_auc((Y["val"][m] > Rv[m]).astype(int), p[m] - Rv[m])
    cand = {}
    for arm in ("L4", "L5", "sc_L5"):
        zs = [S21.load("v8", "arrival", arm, kA, s) for s in SEEDS]
        for f in ("val", "test"):
            for z in zs:
                assert np.array_equal(np.asarray(z[f]["Y"], float), Y[f]), "rows differ"
        cand[arm] = {f: mean([z[f] for z in zs], "P") for f in ("val", "test")}
    best = max(cand, key=lambda a: vauc(cand[a]["val"]))
    comp = {f: np.stack([ew(inc[f]), ew(rf[f]), cand[best][f]], 1) for f in ("val", "test")}
    grid = [w for w in itertools.product(np.round(np.arange(0, 1.0001, 0.05), 2), repeat=2) if w[0] + w[1] <= 1.0 + 1e-9]
    scores = [(vauc(comp["val"] @ np.array([a, b, 1 - a - b])), (a, b)) for a, b in grid]
    _, (a, b) = max(scores, key=lambda x: x[0])
    w = np.array([a, b, 1 - a - b])
    w19 = W19["arrival"]["blend_weight_neural"]
    P = {f: {"hybrid": comp[f] @ w, "incumbent_ens": comp[f][:, 0], "p19_neural_ens": comp[f][:, 1], f"lgbm_{best}_ens": comp[f][:, 2],
             "p19_blend": w19 * comp[f][:, 1] + (1 - w19) * mean(lg[f], "P")} for f in ("val", "test")}
    lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    snap = lb.snapshot_date.values[P5.ordered(lb, te)]
    blocks = [np.flatnonzero(snap == u) for u in np.unique(snap)]
    names = list(P["test"])
    det = S21.det_compare("arrival", P["test"], Y["test"], EV["test"], Rt, blocks, 61,
                          [("hybrid", n) for n in names if n != "hybrid"])
    # decision level UC1 (Phase 15): point arms pred - R_contract; the incumbent's distribution P(late) as Phase 20's reference
    Rc = S21.contracted_ref("v8")
    arms = {}
    for n in names:
        yv, kv = S21.uc1_label(Y["val"], EV["val"], Rc["val"]); yt, kt = S21.uc1_label(Y["test"], EV["test"], Rc["test"])
        arms[n + " (pred - R)"] = [((P["val"][n] - Rc["val"])[kv], yv[kv], (P["test"][n] - Rc["test"])[kt], yt[kt], snap[kt])]
    Sm = {f: mean(inc[f], "S") for f in ("val", "test")}
    arms["incumbent ensemble P(late)"] = [(p_late(Sm["val"], Rc["val"])[kv], yv[kv], p_late(Sm["test"], Rc["test"])[kt], yt[kt], snap[kt])]
    summ = P20D.summarise(arms)
    block = {}
    ref_name = "incumbent ensemble P(late)"
    for n in arms:
        if n != ref_name:
            for c in (0.01, 0.05, 0.10):
                block[f"{n} vs {ref_name} @ {int(c * 100)}%"] = P20D.block_compare((arms[n][0][2], arms[n][0][3], arms[n][0][4]),
                                                                                  (arms[ref_name][0][2], arms[ref_name][0][3], arms[ref_name][0][4]),
                                                                                  cov=c, seed=int(c * 1000))
    return dict(best_lgbm_group_arm=best, val_auc_of_candidates={a: vauc(cand[a]["val"]) for a in cand},
                weights=dict(incumbent=float(w[0]), p19_neural=float(w[1]), lgbm_group=float(w[2])), val_lateness_auc=float(max(scores)[0]),
                p19_blend_weight_neural=w19, deterministic=det, decisions_UC1=S21.compact(summ), block_at_coverage=block)


def fill(ks):
    kF = ks["k"]["fill"]
    inc = {f: [S18.load_neural("fill", s)[f] for s in SEEDS] for f in ("val", "test")}
    rf = {f: [b19(f"fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence", f) for s in SEEDS] for f in ("val", "test")}
    lg = {f: [dict(np.load(os.path.join(S18.PR18, f"v8_fill_rate_p18_fwd_load_s{s}_{f}.npz"))) for s in SEEDS] for f in ("val", "test")}
    Y = {f: np.asarray(inc[f][0]["Y"], float) for f in ("val", "test")}
    cand = {}
    for arm in ("L4", "L5", "ack", "L5_ack", "sc_L5"):
        if S21.have("v8", "fill", arm, kF):
            zs = [S21.load("v8", "fill", arm, kF, s) for s in SEEDS]
            for f in ("val", "test"):
                assert np.array_equal(np.asarray(zs[0][f]["Y"], float), Y[f])
            cand[arm] = {f: mean([z[f] for z in zs], "P") for f in ("val", "test")}
    vcrps = {a: float(M.crps_exact_rows(cand[a]["val"], Y["val"]).mean()) for a in cand}
    best = min(vcrps, key=vcrps.get)
    nr = {f: mean(rf[f], "P") for f in ("val", "test")}
    grid = np.linspace(0, 1, 101)
    crit = [M.crps_exact_rows(w * nr["val"] + (1 - w) * cand[best]["val"], Y["val"]).mean() for w in grid]
    w = float(grid[int(np.argmin(crit))])
    w19 = W19["fill"]["blend_weight_neural"]
    P = {f: {"hybrid": w * nr[f] + (1 - w) * cand[best][f], "incumbent_ens": mean(inc[f], "P"), "p19_neural_ens": nr[f],
             f"lgbm_{best}_ens": cand[best][f], "p19_blend": w19 * nr[f] + (1 - w19) * mean(lg[f], "P")} for f in ("val", "test")}
    lb = P5.labels("v8", "fill_rate"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    snap = lb.snapshot_date.values[P5.ordered(lb, te)]
    blocks = [np.flatnonzero(snap == u) for u in np.unique(snap)]
    names = list(P["test"])
    det = S21.det_compare("fill", P["test"], Y["test"], None, None, blocks, 62, [("hybrid", n) for n in names if n != "hybrid"])
    out = {}
    blocks_cmp = {}
    for uc, sf, lf in (("UC2", lambda P: P[:, 21], lambda y: (y >= 1).astype(int)),
                       ("UC2b_lt0.95", lambda P: P[:, :20].sum(1), lambda y: (y < 0.95).astype(int))):
        arms = {n: [(sf(P["val"][n]), lf(Y["val"]), sf(P["test"][n]), lf(Y["test"]), snap)] for n in names}
        out[uc] = S21.compact(P20D.summarise(arms))
        for n in names:
            if n != "incumbent_ens":
                blocks_cmp[f"{uc}: {n} vs incumbent_ens @ 5%"] = P20D.block_compare((arms[n][0][2], arms[n][0][3], snap),
                                                                                   (arms["incumbent_ens"][0][2], arms["incumbent_ens"][0][3], snap), cov=0.05, seed=50)
                if n != "p19_blend":
                    blocks_cmp[f"{uc}: {n} vs p19_blend @ 5%"] = P20D.block_compare((arms[n][0][2], arms[n][0][3], snap),
                                                                                    (arms["p19_blend"][0][2], arms["p19_blend"][0][3], snap), cov=0.05, seed=51)
    return dict(best_lgbm_group_arm=best, val_crps_of_candidates=vcrps, weight_p19_neural=w, p19_blend_weight_neural=w19,
                deterministic=det, decisions=out, block_at_5pct=blocks_cmp)


def main():
    PP.register()
    st = C.require_clean()
    ks = S21.ksel("v8")
    out = dict(stamp=st, arrival=arrival(ks), fill=fill(ks))
    C.dump(out, "phase21/stage6_hybrid.json")
    for t in ("arrival", "fill"):
        print("==", t, {k: v for k, v in out[t].items() if k in ("best_lgbm_group_arm", "weights", "weight_p19_neural")})
        print(json.dumps(out[t]["deterministic"]["point"], indent=0))
        for k, v in out[t]["deterministic"]["diff"].items():
            for n, d in v.items():
                print("  ", k, n, d["verdict"], [round(x, 4) for x in d["better_positive"]])
    for n, r in out["arrival"]["decisions_UC1"].items():
        print(f"  UC1 {n:40s} {r['cls']} {r['reachable']} P@5% {r['coverage']['5%']['precision']}")
    for k, v in out["arrival"]["block_at_coverage"].items():
        print("  ", k, v["verdict"], [round(x, 4) for x in v["diff_ci"]])


if __name__ == "__main__":
    main()
