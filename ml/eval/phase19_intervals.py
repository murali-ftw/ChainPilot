"""Phase 19 Stage 4 -- honest intervals: Phase 18's seed-ensemble / blend / conformal comparisons redone with a
SNAPSHOT-BLOCK bootstrap (1,000 resamples of whole test snapshots, with replacement; 9 test snapshots), and (4b) the same
comparisons for the Phase 19 neural arms. Torch process (imports only), no LightGBM. Blend weights are refitted on
VALIDATION by Phase 18's criteria (they are deterministic, so they equal Phase 18's).

Every comparison is paired: all arms are scored on the same resampled snapshots. The 95% interval of the difference
(better = positive) is reported beside Phase 18's ROW-bootstrap interval; a verdict SURVIVES when the block interval
still excludes 0 in the same direction.

  python ml/eval/phase19_intervals.py a      # stored predictions only   -> ml/artifacts/phase19/stage4a_block.json
  python ml/eval/phase19_intervals.py b      # + Phase 19 neural bundles -> ml/artifacts/phase19/stage4b_block.json
"""
from __future__ import annotations
import os, sys, json
import phase12_common as C
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import phase5_heads as P5, folds
import phase5_metrics as M
from metrics import cindex, roc_auc
from heads import fill_cell
from phase14_score import p_exceed
from phase15 import pick_on_val
import phase18_score as S18

SEEDS = C.V8_SEEDS
B = 1000
GRID = np.linspace(0, 1, 101)
BUND19 = os.path.join(C.ART, "phase19", "bundles")
RF = {"arrival": "arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence",
      "fill": "fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence",
      "capacity": "capacity_strain/v8_mp_h4_lr0.00025_s{s}_rffwdload"}


# ================================================================== snapshot blocks of the test fold
def test_blocks(task):
    lb = P5.labels("v8", S18.TASK[task]); tr, va, te = folds.fixed_split(lb.snapshot_date)
    snap = lb.snapshot_date.values[P5.ordered(lb, te)]
    u = np.unique(snap)
    return [np.flatnonzero(snap == s) for s in u], [str(pd.Timestamp(s).date()) for s in u]


def block_boot(fns, blocks, seed):
    rng = np.random.default_rng(seed)
    out = {k: np.empty(B) for k in fns}
    for b in range(B):
        i = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        for k, fn in fns.items():
            out[k][b] = fn(i)
    return out


def diff(a, b, higher):
    d = (a - b) if higher else (b - a)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff_better_positive=[float(lo), float(d.mean()), float(hi)], width=float(hi - lo),
                verdict="better" if lo > 0 else "worse" if hi < 0 else "undetermined")


# ================================================================== predictors
def load_arm(task, kind):
    t = S18.TASK[task]
    if kind == "neural":
        return [S18.load_neural(task, s) for s in SEEDS]
    if kind == "lgbm":
        return [S18.load_flat(os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz"), s) for s in SEEDS]
    if kind == "lgbm_fwd":
        return [S18.load_flat(os.path.join(S18.PR18, f"v8_{t}_p18_fwd_load_s{{s}}_{{f}}.npz"), s) for s in SEEDS]
    if kind == "neural_rf":
        return [{f: dict(np.load(os.path.join(BUND19, RF[task].format(s=s), f"preds_{f}.npz"))) for f in ("val", "test")} for s in SEEDS]
    raise ValueError(kind)


def scores(task, zs, f):
    """Per seed: the arm's own prediction in the form the metric reads."""
    if task == "arrival":
        return [S18.point("arrival", z[f]) for z in zs]
    return [np.asarray(z[f]["P"], float) for z in zs]


def build(task, pair):
    """pair = (neural kind, lgbm kind) -> dict of predictors on val and test incl. ensembles and the val-fitted blend."""
    nk, lk = pair
    N, L = load_arm(task, nk), load_arm(task, lk)
    P = {}
    for f in ("val", "test"):
        if task == "arrival":
            S = np.mean([np.asarray(z[f]["S"], float) for z in N], 0); pT = np.mean([np.asarray(z[f]["pT"], float) for z in N], 0)
            ne = S18.expected_week({"S": S, "pT": pT})
        else:
            ne = np.mean(scores(task, N, f), 0)
        le = np.mean(scores(task, L, f), 0)
        P[f] = dict(neural_ens=ne, lgbm_ens=le, **{f"neural_s{s}": x for s, x in zip(SEEDS, scores(task, N, f))},
                    **{f"lgbm_s{s}": x for s, x in zip(SEEDS, scores(task, L, f))},
                    Y=np.asarray(N[0][f]["Y"], float), EV=np.asarray(N[0][f]["EV"], bool))
    v = P["val"]
    if task == "arrival":
        R = S18.arrival_reference()["val"]; m = v["EV"]; yl = (v["Y"][m] > R[m]).astype(int)
        crit = [-roc_auc(yl, (w * v["neural_ens"] + (1 - w) * v["lgbm_ens"])[m] - R[m]) for w in GRID]
    elif task == "fill":
        crit = [M.crps_exact_rows(w * v["neural_ens"] + (1 - w) * v["lgbm_ens"], v["Y"]).mean() for w in GRID]
    else:
        crit = [np.mean([M.pinball_rows(v["Y"], (w * v["neural_ens"] + (1 - w) * v["lgbm_ens"])[:, i], q).mean()
                         for i, q in enumerate(M.QS)]) for w in GRID]
    w = float(GRID[int(np.argmin(crit))])
    for f in ("val", "test"):
        P[f]["blend"] = w * P[f]["neural_ens"] + (1 - w) * P[f]["lgbm_ens"]
    return P, w


# ================================================================== metric functions on a row index
def metric_fns(task, P):
    t, v = P["test"], P["val"]
    arms = [k for k in t if k not in ("Y", "EV")]
    Y, m = t["Y"], t["EV"]
    if task == "arrival":
        R = S18.arrival_reference()["test"]
        def late(a): return lambda i: roc_auc((Y[i][m[i]] > R[i][m[i]]).astype(int), t[a][i][m[i]] - R[i][m[i]])
        def a3(a): return lambda i: float(np.median(np.abs(7 * t[a][i][m[i]] - 7 * Y[i][m[i]])))
        def ci(a): return lambda i: float(cindex(t[a][i], Y[i], m[i], n=20_000, seed=13))
        return arms, {"lateness_auc": (late, True), "a3_median_abs_err_days": (a3, False), "cindex": (ci, True)}
    if task == "fill":
        full, cells = (Y >= 1).astype(int), fill_cell(Y)
        rows = {a: M.crps_exact_rows(t[a], Y) for a in arms}
        def crps(a): return lambda i: float(rows[a][i].mean())
        def auc(a): return lambda i: roc_auc(full[i], t[a][i, 21])
        def ece(a): return lambda i: float(M.ece_marginal(t[a][i], cells[i])[0])
        return arms, {"crps_exact": (crps, False), "p_full_auc": (auc, True), "ece22": (ece, False)}
    lt, lv = (Y > 1).astype(int), (v["Y"] > 1).astype(int)
    sc = {a: p_exceed(t[a]) for a in arms}
    tau = {a: {p: pick_on_val(p_exceed(v[a]), lv, p)[0] for p in S18.PBAR} for a in arms}
    def prec(c):
        def mk(a):
            def fn(i):
                s, y = sc[a][i], lt[i]; mm = max(1, int(np.ceil(c * len(y)))); return float(y[np.argsort(-s, kind="stable")[:mm]].mean())
            return fn
        return mk
    def rec(p):
        def mk(a):
            def fn(i):
                if tau[a][p] is None:
                    return np.nan
                s, y = sc[a][i], lt[i]; return float(((s >= tau[a][p]) & (y == 1)).sum() / max(y.sum(), 1))
            return fn
        return mk
    fns = {f"precision_at_{int(c * 100)}pct": (prec(c), True) for c in S18.COV}
    fns.update({f"recall_at_p{p:.2f}": (rec(p), True) for p in S18.PBAR})
    return arms, fns


COMPARE = [("neural_ens", "MEAN_neural"), ("lgbm_ens", "MEAN_lgbm"), ("neural_ens", "lgbm_ens"),
           ("blend", "neural_ens"), ("blend", "lgbm_ens")]


def run_task(task, pair, seed0):
    P, w = build(task, pair)
    blocks, snaps = test_blocks(task)
    arms, fns = metric_fns(task, P)
    res = dict(blend_weight_neural=w, n_test_snapshots=len(blocks), point={}, compare={})
    for mname, (mk, higher) in fns.items():
        bs = block_boot({a: mk(a) for a in arms}, blocks, seed0)
        bs["MEAN_neural"] = np.nanmean([bs[f"neural_s{s}"] for s in SEEDS], 0)
        bs["MEAN_lgbm"] = np.nanmean([bs[f"lgbm_s{s}"] for s in SEEDS], 0)
        full = np.arange(len(P["test"]["Y"]))
        res["point"][mname] = {a: float(mk(a)(full)) for a in ("neural_ens", "lgbm_ens", "blend")}
        res["compare"][mname] = {f"{a} vs {b}": diff(bs[a], bs[b], higher) for a, b in COMPARE}
    return res


def conformal_block():
    """Arrival intervals (Phase 18 3c) with block-bootstrap coverage intervals, per seed then banded."""
    import phase18_squeeze as SQ
    blocks, _ = test_blocks("arrival")
    out = {}
    for s in SEEDS:
        z = S18.load_neural("arrival", s)
        ev_v, ev_t = np.asarray(z["val"]["EV"], bool), np.asarray(z["test"]["EV"], bool)
        Yt = np.asarray(z["test"]["Y"], float)
        cent = {"p50": {f: SQ.dist_q(z[f]["S"], z[f]["pT"], 0.5) for f in ("val", "test")},
                "expected_week": {f: S18.expected_week(z[f]) for f in ("val", "test")}}
        cover = {}
        for c, cc in cent.items():
            r = 7 * (np.asarray(z["val"]["Y"], float)[ev_v] - cc["val"][ev_v])
            q10, q90 = float(np.quantile(r, 0.10, method="lower")), float(np.quantile(r, 0.90, method="higher"))
            cover[c] = ev_t & (7 * Yt >= 7 * cc["test"] + q10) & (7 * Yt <= 7 * cc["test"] + q90)
        p10, p90 = SQ.dist_q(z["test"]["S"], z["test"]["pT"], 0.1), SQ.dist_q(z["test"]["S"], z["test"]["pT"], 0.9)
        cover["raw_p10_p90"] = ev_t & (Yt >= p10) & (Yt <= p90)
        fn = {k: (lambda i, v=v: float(v[i].sum() / max(ev_t[i].sum(), 1))) for k, v in cover.items()}
        bs = block_boot(fn, blocks, 100 + s)
        out[f"s{s}"] = {k: dict(coverage=float(v.sum() / ev_t.sum()), block_ci=[float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))])
                        for k, v in cover.items()}
    return out


def main(mode):
    st = C.require_clean()
    if mode == "a":
        out = dict(stamp=st, B=B, kind="snapshot-block bootstrap of Phase 18 Stage 3 (stored predictions)", tasks={})
        old = json.load(open(os.path.join(C.ART, "phase18", "stage3_squeeze.json")))
        for k, task in enumerate(("arrival", "fill", "capacity")):
            r = run_task(task, ("neural", "lgbm"), 7 + k)
            for mname, comps in r["compare"].items():
                for cname, d in comps.items():
                    oc = cname.replace("MEAN_neural", "mean single neural").replace("MEAN_lgbm", "mean single lgbm")
                    o = old[task]["compare"].get(mname, {}).get(oc)
                    if o:
                        lo, mu, hi = o["diff_better_positive"]
                        d["row_bootstrap_phase18"] = dict(ci=[lo, hi], width=hi - lo, verdict=o["verdict"])
                        d["width_ratio_block_over_row"] = d["width"] / (hi - lo) if hi > lo else float("inf")
                        d["survives"] = (o["verdict"] == d["verdict"]) if o["verdict"] != "undetermined" else None
            out["tasks"][task] = r
        out["arrival_conformal"] = conformal_block()
        C.dump(out, "phase19/stage4a_block.json")
    else:
        out = dict(stamp=st, B=B, kind="Phase 19 neural arms: ensemble and val-fitted blend with LightGBM + fwd_load", tasks={})
        for k, task in enumerate(("arrival", "fill", "capacity")):
            if not all(os.path.exists(os.path.join(BUND19, RF[task].format(s=s), "preds_test.npz")) for s in SEEDS):
                out["tasks"][task] = "NOT RUN (fewer than 5 seeds of the Phase 19 neural arm)"; continue
            r = run_task(task, ("neural_rf", "lgbm_fwd"), 20 + k)
            # the new neural ensemble against the INCUMBENT neural ensemble, same resamples
            Pi, _ = build(task, ("neural", "lgbm")); Pn, _ = build(task, ("neural_rf", "lgbm_fwd"))
            blocks, _ = test_blocks(task)
            both = {"test": {"rf_ens": Pn["test"]["neural_ens"], "inc_ens": Pi["test"]["neural_ens"],
                             "rf_blend": Pn["test"]["blend"], "Y": Pn["test"]["Y"], "EV": Pn["test"]["EV"]},
                    "val": {"rf_ens": Pn["val"]["neural_ens"], "inc_ens": Pi["val"]["neural_ens"],
                            "rf_blend": Pn["val"]["blend"], "Y": Pn["val"]["Y"], "EV": Pn["val"]["EV"]}}
            arms, fns = metric_fns(task, both)
            r["vs_incumbent_ensemble"] = {}
            for mname, (mk, higher) in fns.items():
                bs = block_boot({a: mk(a) for a in arms}, blocks, 50 + k)
                r["vs_incumbent_ensemble"][mname] = {"rf_ens vs inc_ens": diff(bs["rf_ens"], bs["inc_ens"], higher),
                                                     "rf_blend vs inc_ens": diff(bs["rf_blend"], bs["inc_ens"], higher)}
            out["tasks"][task] = r
        C.dump(out, "phase19/stage4b_block.json")
    print(json.dumps({t: (v if isinstance(v, str) else {m: {c: (d["verdict"], [round(x, 4) for x in d["diff_better_positive"]],
                                                             d.get("survives")) for c, d in cc.items()} for m, cc in v["compare"].items()})
                      for t, v in out["tasks"].items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a")
