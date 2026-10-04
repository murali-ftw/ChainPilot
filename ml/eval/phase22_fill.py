"""Phase 22 Stage 3 -- fill consolidation (pre-registration D7). Torch process (imports only), no LightGBM.

  gates   (3a) LightGBM on v8clean, 5 seeds, RAW: each family vs BASE_clean with its controls
            season + cadence  control sc_ctrl (season snapshot-permuted within split, cadence shuffled across channels;
                              preserves: BASE, the channel's own snapshot)
            ack-gap           controls ack_sperm (same channel, donor snapshot; preserves the channel's typical gap) and
                              ack_xsh (another channel, same snapshot; preserves the network-wide level at t0)
            L4                control L4_xsh (another channel, same snapshot; preserves the L2-supplier? no: the whole block
                              is another channel's, so it preserves only the snapshot's global (L1) columns)
          family PASSES: disjointly better than BASE_clean on a primary metric, none disjointly worse, no control disjointly
          better than BASE_clean. Gate self-test on constructed cases first.
  final   (3b-d) neural consolidated (bound with the PASSING families) GAIN / TIE / WORSE vs the clean incumbent band (self-test:
          incumbent vs itself -> TIE); blend neural-consolidated + LightGBM-consolidated (w by validation CRPS); the clean
          Phase 19 fill blend (clean neural season + cadence + clean LightGBM + fwd_load, w by validation CRPS); the PUBLISHED
          Phase 19 blend in its own column, never compared with clean arms. Snapshot-block bootstrap; Phase 15 classes;
          per-snapshot spread. Replication: world 2 LightGBM ack-gap and consolidated vs world 2's own BASE_clean.

  python ml/eval/phase22_fill.py gates
  python ml/eval/phase22_fill.py final --families season,cadence,ack,L4
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import phase5_heads as P5, folds
import phase5_metrics as M
from metrics import roc_auc
import phase18_score as S18
import phase20_decisions as P20D
import phase21_score as S21

SEEDS = C.V8_SEEDS
B = 1000
PR22 = os.path.join(C.ART, "phase22", "preds")
B22 = os.path.join(C.ART, "phase22", "bundles")
B19 = os.path.join(C.ART, "phase19", "bundles")
DIR = {"crps_exact": False, "p_full_auc": True, "uc2b_precision_at_5pct": True}
CTRL = {"sc": ["sc_ctrl"], "ack": ["ack_sperm", "ack_xsh"], "L4": ["L4_xsh"]}
SHORT = {"season": "season", "cadence": "cadence", "ack": "ack", "L4": "L4"}


def fmt(world, arm, s, f):
    return os.path.join(PR22, f"{world}_fill_rate_p22_{arm}_s{s}_{f}.npz")


def lgb(world, arm, s):
    return {f: dict(np.load(fmt(world, arm, s, f))) for f in ("val", "test")}


def have(world, arm):
    return all(os.path.exists(fmt(world, arm, s, f)) for s in SEEDS for f in ("val", "test"))


def metrics(P, Y):
    P = np.asarray(P, float); Y = np.asarray(Y, float)
    out = S18.fill_metrics(P, Y)
    out["uc2b_precision_at_5pct"] = P20D.prec_at(P[:, :20].sum(1), (Y < 0.95).astype(int), 0.05)
    out["uc2_precision_at_5pct"] = P20D.prec_at(P[:, 21], (Y >= 1).astype(int), 0.05)
    return out


def bands(loader):
    per = [metrics(z["test"]["P"], z["test"]["Y"]) for z in (loader(s) for s in SEEDS)]
    return {k: S18.band([p[k] for p in per]) for k in per[0]}, per


def gate(fam, base, ctrls):
    c = {k: S18.compare(fam[k], base[k], h) for k, h in DIR.items()}
    better = [k for k, v in c.items() if v == "better"]; worse = [k for k, v in c.items() if v == "worse"]
    cb = {n: [k for k, h in DIR.items() if S18.compare(b[k], base[k], h) == "better"] for n, b in ctrls.items()}
    bad = {n: v for n, v in cb.items() if v}
    return dict(verdict="PASS" if (better and not worse and not bad) else "FAIL", better=better, worse=worse,
                controls_better_than_base=cb, vs_base=c)


def gate_selftest():
    b = {"crps_exact": [0.139, 0.1395, 0.14], "p_full_auc": [0.60, 0.605, 0.61], "uc2b_precision_at_5pct": [0.37, 0.38, 0.39]}
    g = {"crps_exact": [0.130, 0.131, 0.132], "p_full_auc": [0.66, 0.665, 0.67], "uc2b_precision_at_5pct": [0.45, 0.46, 0.47]}
    r1 = gate(dict(b), b, {})["verdict"]; r2 = gate(g, b, {"ctrl": g})["verdict"]
    assert r1 == "FAIL" and r2 == "FAIL", "the gate cannot fail -- invalid"
    return dict(equal_bands=r1, control_also_better=r2)


def gates(world="v8clean"):
    out = dict(selftest=gate_selftest(), bands={}, gates={})
    arms = [a for a in ("base", "sc", "ack", "L4", "sc_ack", "sc_ack_L4", "sc_ctrl", "ack_sperm", "ack_xsh", "L4_xsh", "fwdload") if have(world, a)]
    z0 = lgb(world, "base", 7)
    for a in arms:
        z = lgb(world, a, 7)
        assert np.array_equal(np.asarray(z["test"]["Y"], float), np.asarray(z0["test"]["Y"], float)), f"rows differ: {a}"
        out["bands"][a] = bands(lambda s, a=a: lgb(world, a, s))[0]
    for fam, ctrls in CTRL.items():
        if fam in out["bands"] and all(c in out["bands"] for c in ctrls):
            out["gates"][fam] = gate(out["bands"][fam], out["bands"]["base"], {c: out["bands"][c] for c in ctrls})
    if "sc_ack_L4" in out["bands"]:
        out["consolidated_vs_base"] = {k: S18.compare(out["bands"]["sc_ack_L4"][k], out["bands"]["base"][k], h) for k, h in DIR.items()}
        if "sc" in out["bands"]:
            out["consolidated_vs_sc"] = {k: S18.compare(out["bands"]["sc_ack_L4"][k], out["bands"]["sc"][k], h) for k, h in DIR.items()}
    fam_map = {"sc": ["season", "cadence"], "ack": ["ack"], "L4": ["L4"]}
    out["passing_families"] = [f for fam, g in out["gates"].items() if g["verdict"] == "PASS" for f in fam_map[fam]]
    return out


def neural(task_dir, name_fmt, root=B22):
    return [{f: dict(np.load(os.path.join(root, task_dir, name_fmt.format(s=s), f"preds_{f}.npz"))) for f in ("val", "test")} for s in SEEDS]


def mean_p(zs, f):
    return np.mean([np.asarray(z[f]["P"], float) for z in zs], 0)


def fit_w(a, b, y):
    grid = np.linspace(0, 1, 101)
    crit = [M.crps_exact_rows(w * a + (1 - w) * b, y).mean() for w in grid]
    return float(grid[int(np.argmin(crit))])


def final(fams, world="v8clean"):
    out = dict(families=fams)
    rf = "+".join(fams)
    inc = neural("fill_rate", "v8clean_none_h0_lr0.000125_s{s}")
    cons = neural("fill_rate", "v8clean_none_h0_lr0.000125_s{s}_rf" + rf)
    p19c = neural("fill_rate", "v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence")
    Y = {f: np.asarray(inc[0][f]["Y"], float) for f in ("val", "test")}
    for zs in (cons, p19c):
        for f in ("val", "test"):
            assert np.array_equal(np.asarray(zs[0][f]["Y"], float), Y[f])
    bi = bands(lambda s: inc[SEEDS.index(s)])[0]; bc = bands(lambda s: cons[SEEDS.index(s)])[0]
    out["neural"] = dict(incumbent_clean=bi, consolidated_clean=bc,
                         verdict_vs_incumbent={k: S18.compare(bc[k], bi[k], h) for k, h in DIR.items()},
                         selftest_incumbent_vs_itself={k: S18.compare(bi[k], bi[k], h) for k, h in DIR.items()})
    better = [k for k, v in out["neural"]["verdict_vs_incumbent"].items() if v == "better"]
    worse = [k for k, v in out["neural"]["verdict_vs_incumbent"].items() if v == "worse"]
    out["neural"]["GAIN_TIE_WORSE"] = "GAIN" if better and not worse else "WORSE" if worse and not better else "TIE" if not better and not worse else "MIXED"
    assert all(v == "undetermined" for v in out["neural"]["selftest_incumbent_vs_itself"].values()), "neural verdict cannot tie"
    lgc = [lgb(world, "sc_ack_L4", s) for s in SEEDS]; lgf = [lgb(world, "fwdload", s) for s in SEEDS]
    P = {}
    for f in ("val", "test"):
        P[f] = {"incumbent_ens": mean_p(inc, f), "neural_consolidated_ens": mean_p(cons, f), "lgbm_consolidated_ens": mean_p(lgc, f),
                "p19_neural_clean_ens": mean_p(p19c, f), "lgbm_fwdload_clean_ens": mean_p(lgf, f)}
    w = fit_w(P["val"]["neural_consolidated_ens"], P["val"]["lgbm_consolidated_ens"], Y["val"])
    w19 = fit_w(P["val"]["p19_neural_clean_ens"], P["val"]["lgbm_fwdload_clean_ens"], Y["val"])
    for f in ("val", "test"):
        P[f]["consolidated_blend"] = w * P[f]["neural_consolidated_ens"] + (1 - w) * P[f]["lgbm_consolidated_ens"]
        P[f]["p19_blend_clean"] = w19 * P[f]["p19_neural_clean_ens"] + (1 - w19) * P[f]["lgbm_fwdload_clean_ens"]
    out["weights"] = dict(consolidated_w_neural=w, p19_clean_w_neural=w19, p19_published_w_neural=0.50)
    lb = P5.labels("v8", "fill_rate"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    snap = lb.snapshot_date.values[P5.ordered(lb, te)]
    out["clean"] = S21.det_compare("fill", P["test"], Y["test"], None, None, [np.flatnonzero(snap == u) for u in np.unique(snap)], 71,
                                   [("consolidated_blend", n) for n in P["test"] if n != "consolidated_blend"])
    out["clean_uc2_p5"] = {n: P20D.prec_at(p[:, 21], (Y["test"] >= 1).astype(int), 0.05) for n, p in P["test"].items()}
    # published Phase 19 blend, its own column (published = leaky inputs)
    pub_n = neural("fill_rate", "v8_none_h0_lr0.000125_s{s}_rfseason+cadence", root=B19)
    pub_l = [{f: dict(np.load(os.path.join(S18.PR18, f"v8_fill_rate_p18_fwd_load_s{s}_{f}.npz"))) for f in ("val", "test")} for s in SEEDS]
    pb = 0.5 * mean_p(pub_n, "test") + 0.5 * mean_p(pub_l, "test")
    out["published_p19_blend"] = dict(metrics(pb, Y["test"]), note="PUBLISHED (leaky inputs): reported beside, never compared with a clean arm")
    out["decisions"] = {}
    for uc, sf, lf in (("UC2", lambda P_: P_[:, 21], lambda y: (y >= 1).astype(int)),
                       ("UC2b_lt0.95", lambda P_: P_[:, :20].sum(1), lambda y: (y < 0.95).astype(int))):
        arms = {n: [(sf(P["val"][n]), lf(Y["val"]), sf(P["test"][n]), lf(Y["test"]), snap)] for n in P["test"]}
        out["decisions"][uc] = S21.compact(P20D.summarise(arms))
        out.setdefault("block_5pct", {})[uc] = {n: P20D.block_compare((arms["consolidated_blend"][0][2], arms["consolidated_blend"][0][3], snap),
                                                                     (arms[n][0][2], arms[n][0][3], snap), cov=0.05, seed=52)
                                                for n in arms if n != "consolidated_blend"}
    # replication on world 2 (LightGBM only, own BASE_clean)
    w2 = "v8w1002clean"
    if all(have(w2, a) for a in ("base", "ack", "sc_ack_L4", "ack_sperm", "ack_xsh")):
        b2 = {a: bands(lambda s, a=a: lgb(w2, a, s))[0] for a in ("base", "ack", "sc_ack_L4", "ack_sperm", "ack_xsh")}
        rep = {}
        for a in ("ack", "sc_ack_L4"):
            rep[a] = {}
            for k, h in DIR.items():
                g1 = (lambda bb, base: (bb[k][1] - base[k][1]) if h else (base[k][1] - bb[k][1]))
                v8b = gates(world)["bands"]
                gv8, gw2 = g1(v8b[a], v8b["base"]), g1(b2[a], b2["base"])
                c2 = S18.compare(b2[a][k], b2["base"][k], h)
                rep[a][k] = dict(gain_v8clean=gv8, gain_w2clean=gw2, w2_vs_base=c2, ratio=gw2 / gv8 if gv8 else None,
                                 verdict="REPLICATES" if c2 != "undetermined" and np.sign(gv8) == np.sign(gw2) else "DOES NOT" if c2 != "undetermined" else "UNDETERMINED")
        rep["ack_gate_w2"] = gate(b2["ack"], b2["base"], {"ack_sperm": b2["ack_sperm"], "ack_xsh": b2["ack_xsh"]})
        out["replication_w2"] = dict(bands=b2, verdicts=rep, note="LightGBM only; the neural half is not replicated; one generator")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["gates", "final"]); ap.add_argument("--families", default="")
    a = ap.parse_args()
    PP.register()
    for p in ("v8", "v8w1002"):
        config.WORLDS[p + "clean"] = config.WORLDS[p]
    st = C.require_clean()
    if a.stage == "gates":
        out = dict(stamp=st, **gates())
        C.dump(out, "phase22/fill_gates.json")
        for k, v in out["bands"].items():
            print(f"  {k:10s}", {m: [round(x, 4) for x in b] for m, b in v.items() if m in DIR or m == "uc2_precision_at_5pct"})
        print(json.dumps({k: (v["verdict"], v["controls_better_than_base"], v["vs_base"]) for k, v in out["gates"].items()}, indent=0))
        print("PASSING:", out["passing_families"], "| consolidated vs base:", out.get("consolidated_vs_base"), "| vs sc:", out.get("consolidated_vs_sc"))
    else:
        out = dict(stamp=st, **final(a.families.split(",")))
        C.dump(out, "phase22/fill_final.json")
        print(json.dumps({k: out[k] for k in ("families", "weights")}, indent=0))
        print(json.dumps(out["neural"]["verdict_vs_incumbent"]), out["neural"]["GAIN_TIE_WORSE"])
        print(json.dumps(out["clean"]["point"], indent=0)); print(json.dumps(out["clean_uc2_p5"]))
        print(json.dumps(out["published_p19_blend"]))


if __name__ == "__main__":
    main()
