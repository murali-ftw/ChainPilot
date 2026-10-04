"""Phase 20 Stage 1 -- decision-level re-test of the Phase 18-19 gains, through Phase 15's machinery unchanged.

Phase 15's functions are imported, not copied: coverage_curve (precision / recall / lift at coverage, cells < 50 alerts
suppressed), pick_on_val (VALIDATION ONLY threshold for precision p), apply, analyse (Stage A curve, recall at p,
Stage B verdict = validation majority). Use cases and labels are Phase 15's:
  UC1  arrival late vs the CONTRACTED lead reference, censoring resolved (phase15.arrival_arrays lab_c)
  UC2  fill = 1 (score P(fill = 1));   UC2b  fill < 0.95 (score = mass below 0.95)
  UC3  strain > 1 (score P(strain > 1) from P10 / P50 / P90, phase14_score.p_exceed)
Distribution arms score P(late) off the survival curve (phase14_score.p_late); point arms score prediction - reference,
exactly as Phase 15 scored LightGBM.

REACHABLE (Phase 15 rule, verbatim in the pre-registration) and the class mapping fixed there:
  ALERT      REACHABLE YES / PARTIAL and lift >= 1.5 at the highest bar whose validation-chosen point meets the reachable
             condition (test precision >= 0.85 for YES, >= 0.70 for PARTIAL; recall >= 0.10; >= 50 alerts; every seed)
  WATCHLIST  not ALERT and Stage B TUNABLE / WEAKLY TUNABLE;   RETIRED  Stage B NOT TUNABLE
Deterministic arms (ensembles, blends) are one predictor; their intervals come from a snapshot-block bootstrap (1,000
resamples of whole test snapshots, paired across arms).

  python ml/eval/phase20_decisions.py      # -> ml/artifacts/phase20/stage1_decisions.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd
import phase5_heads as P5, folds
from phase11b_lateness import build_reference
from phase14_score import p_late, p_exceed
from phase15 import analyse, pick_on_val, coverage_curve, KS, PS, MIN_ALERTS
import phase18_score as S18

SEEDS = C.V8_SEEDS
B = 1000
BUND19 = os.path.join(C.ART, "phase19", "bundles")
PR18 = os.path.join(C.ART, "phase18", "preds")
PR7 = os.path.join(C.ART, "phase7_preds")
COV = (0.01, 0.05, 0.10, 0.20)
W19 = json.load(open(os.path.join(C.ART, "phase19", "stage4b_block.json")))["tasks"]


def bund(path, f):
    return dict(np.load(os.path.join(C.BUND, path, f"preds_{f}.npz")))


def b19(path, f):
    return dict(np.load(os.path.join(BUND19, path, f"preds_{f}.npz")))


def flat(path_fmt, s, f):
    return dict(np.load(path_fmt.format(s=s, f=f)))


def snapshots(task, fold):
    lb = P5.labels("v8", task); tr, va, te = folds.fixed_split(lb.snapshot_date)
    m = {"val": va, "test": te}[fold]
    return lb.snapshot_date.values[P5.ordered(lb, m)]


# ================================================================== arm builders -> {name: [ (sv, yv, st, yt, snap_t), ...per seed ]}
def arrival_arms():
    lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    R = build_reference("contracted_lead", "v8", lb, tr)[0]
    Rx = {"val": R[P5.ordered(lb, va)], "test": R[P5.ordered(lb, te)]}
    snap = {"val": snapshots("arrival_week", "val"), "test": snapshots("arrival_week", "test")}

    def lab(z, f):
        ev = np.asarray(z["EV"], bool); r = Rx[f]
        return ((ev & (z["Y"] > r)) | (~ev & (r < 13))).astype(int), ev | (r < 13)

    def pack(score_of):          # score_of(f) -> (score, z) for one predictor
        out = []
        for f in ("val", "test"):
            s, z = score_of(f); y, k = lab(z, f)
            out += [s[k], y[k]]
            if f == "test":
                out.append(snap["test"][k])
        return tuple(out)
    inc = {f: [bund(f"arrival_week/v8_lite_h4_lr0.00025_s{s}", f) for s in SEEDS] for f in ("val", "test")}
    rf = {f: [b19(f"arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence", f) for s in SEEDS] for f in ("val", "test")}
    lg = {f: [flat(os.path.join(PR18, "v8_arrival_week_p18_fwd_load_s{s}_{f}.npz"), s, f) for s in SEEDS] for f in ("val", "test")}
    mean = lambda zs, k: np.mean([np.asarray(z[k], float) for z in zs], 0)
    w = W19["arrival"]["blend_weight_neural"]

    def blend_et(f):
        ne = S18.expected_week({"S": mean(rf[f], "S"), "pT": mean(rf[f], "pT")})
        return w * ne + (1 - w) * mean(lg[f], "P")
    arms = {
        "neural incumbent h4, per seed (Phase 15's arm)": [pack(lambda f, i=i: (p_late(inc[f][i]["S"], Rx[f]), inc[f][i])) for i in range(5)],
        "neural incumbent ensemble": [pack(lambda f: (p_late(mean(inc[f], "S"), Rx[f]), inc[f][0]))],
        f"Phase 19 blend (w_neural {w})": [pack(lambda f: (blend_et(f) - Rx[f], inc[f][0]))],
        "LightGBM + fwd_load, per seed": [pack(lambda f, i=i: (np.asarray(lg[f][i]["P"], float) - Rx[f], lg[f][i])) for i in range(5)],
        "LightGBM + fwd_load ensemble": [pack(lambda f: (mean(lg[f], "P") - Rx[f], lg[f][0]))],
    }
    for f in ("val", "test"):
        for z in lg[f] + rf[f]:
            assert np.array_equal(z["Y"], inc[f][0]["Y"]), "arrival rows differ between arms"
    return {"UC1": arms}


def fill_arms():
    snap = {f: snapshots("fill_rate", f) for f in ("val", "test")}
    inc = {f: [bund(f"fill_rate/v8_none_h0_lr0.000125_s{s}", f) for s in SEEDS] for f in ("val", "test")}
    bw3 = {f: [bund(f"fill_rate/v8_none_h0_lr0.000125_s{s}_lossrps_bw3", f) for s in SEEDS] for f in ("val", "test")}
    rf = {f: [b19(f"fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence", f) for s in SEEDS] for f in ("val", "test")}
    lg = {f: [flat(os.path.join(PR18, "v8_fill_rate_p18_fwd_load_s{s}_{f}.npz"), s, f) for s in SEEDS] for f in ("val", "test")}
    sh = {f: [flat(os.path.join(PR7, "v8_fill_rate_b5flat22_s{s}_{f}.npz"), s, f) for s in SEEDS] for f in ("val", "test")}
    lid = {f: [flat(os.path.join(PR7, "v8_fill_rate_lgbm22_id_s{s}_{f}.npz"), s, f) for s in SEEDS] for f in ("val", "test")}
    mean = lambda zs: np.mean([np.asarray(z["P"], float) for z in zs], 0)
    w = W19["fill"]["blend_weight_neural"]
    Y = {f: np.asarray(inc[f][0]["Y"], float) for f in ("val", "test")}
    for f in ("val", "test"):
        for z in rf[f] + lg[f] + sh[f] + lid[f] + bw3[f]:
            assert np.array_equal(z["Y"], inc[f][0]["Y"]), "fill rows differ between arms"
    P = {"neural incumbent ensemble": lambda f: mean(inc[f]), "neural + season + cadence ensemble (Phase 19)": lambda f: mean(rf[f]),
         f"Phase 19 blend (w_neural {w})": lambda f: w * mean(rf[f]) + (1 - w) * mean(lg[f]),
         "LightGBM + fwd_load ensemble": lambda f: mean(lg[f])}
    per_seed = {"LightGBM + fwd_load, per seed": lg, "shipped b5flat22, per seed": sh,
                "Phase 15 UC2 arm: boundary bw3, per seed": bw3, "Phase 15 UC2b arm: lgbm22_id, per seed": lid}
    out = {}
    for uc, sf, lf in (("UC2", lambda P: P[:, 21], lambda y: (y >= 1).astype(int)),
                       ("UC2b_lt0.95", lambda P: P[:, :20].sum(1), lambda y: (y < 0.95).astype(int))):
        arms = {n: [(sf(fn("val")), lf(Y["val"]), sf(fn("test")), lf(Y["test"]), snap["test"])] for n, fn in P.items()}
        for n, zz in per_seed.items():
            arms[n] = [(sf(np.asarray(zz["val"][i]["P"], float)), lf(Y["val"]), sf(np.asarray(zz["test"][i]["P"], float)), lf(Y["test"]),
                        snap["test"]) for i in range(5)]
        out[uc] = arms
    return out


def capacity_arms():
    snap = snapshots("capacity_strain", "test")
    inc = {f: [bund(f"capacity_strain/v8_mp_h4_lr0.00025_s{s}", f) for s in SEEDS] for f in ("val", "test")}
    Y = {f: np.asarray(inc[f][0]["Y"], float) for f in ("val", "test")}
    q = lambda f: np.mean([np.asarray(z["P"], float) for z in inc[f]], 0)
    arms = {"neural incumbent mp h4, per seed (Phase 15's arm)": [(p_exceed(inc["val"][i]["P"]), (Y["val"] > 1).astype(int),
                                                                    p_exceed(inc["test"][i]["P"]), (Y["test"] > 1).astype(int), snap) for i in range(5)],
            "neural incumbent ensemble (mean quantiles)": [(p_exceed(q("val")), (Y["val"] > 1).astype(int), p_exceed(q("test")),
                                                            (Y["test"] > 1).astype(int), snap)]}
    return {"UC3": arms}


# ================================================================== the decision summary of one arm
def reachable_and_class(an):
    rp = an["recall_at_precision"]
    def meets(p, floor):
        x = rp[str(p)]
        if x.get("status") != "reached on validation":
            return False
        return x["test_precision"][0] >= floor and x["test_recall"][0] >= 0.10 and x["test_alerts"][0] >= MIN_ALERTS
    stage_b = an["stage_b"]["verdict"]
    yes = [p for p in PS if p >= 0.85 and meets(p, 0.85)]
    part = [p for p in PS if meets(p, 0.70)]
    if yes:
        reach, bar = "YES", max(yes)
    elif part:
        reach, bar = "PARTIAL", max(part)
    else:
        reach, bar = ("NO, TUNING" if stage_b == "NOT TUNABLE" else "NO, CEILING"), None
    base = an["base_rate_test"][1]
    lift = rp[str(bar)]["test_precision"][1] / base if bar is not None else None
    if reach in ("YES", "PARTIAL") and lift >= 1.5:
        cls = "ALERT"
    elif stage_b in ("TUNABLE", "WEAKLY TUNABLE"):
        cls = "WATCHLIST"
    else:
        cls = "RETIRED"
    return dict(reachable=reach, operating_bar=bar, lift_at_bar=lift, stage_b=stage_b, cls=cls)


def accuracy_at_val_f1(sv, yv, st, yt):
    o = np.argsort(-sv, kind="stable"); ys = yv[o]; tp = np.cumsum(ys); n = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (n + ys.sum()); i = int(np.argmax(f1)); tau = sv[o][i]
    pred = st >= tau
    return dict(threshold_val_max_f1=float(tau), test_accuracy=float((pred == (yt == 1)).mean()),
                majority_accuracy=float(max(yt.mean(), 1 - yt.mean())))


def prec_at(s, y, c):
    m = max(1, int(np.ceil(c * len(y)))); return float(y[np.argsort(-s, kind="stable")[:m]].mean())


def per_snapshot_p5(st, yt, snap):
    out = {}
    for u in np.unique(snap):
        k = snap == u
        out[str(pd.Timestamp(u).date())] = prec_at(st[k], yt[k], 0.05)
    return out


def summarise(arms):
    res = {}
    for name, pairs in arms.items():
        an = analyse([p[:4] for p in pairs])
        cls = reachable_and_class(an)
        acc = [accuracy_at_val_f1(*p[:4]) for p in pairs]
        snap_p5 = [per_snapshot_p5(p[2], p[3], p[4]) for p in pairs]
        keys = list(snap_p5[0])
        ps = {k: float(np.mean([d[k] for d in snap_p5])) for k in keys}
        cov = {}
        for c in COV:
            cell = next(x for x in an["curve"] if abs(x["k"] - c) < 1e-12)
            cov[f"{int(c * 100)}%"] = {k: cell.get(k) for k in ("precision", "recall", "lift", "alerts", "suppressed")}
        res[name] = dict(n_seeds=len(pairs), base_rate=an["base_rate_test"], coverage=cov,
                         recall_at_precision=an["recall_at_precision"], **cls,
                         accuracy=dict(test=S18.band([a["test_accuracy"] for a in acc]), majority=acc[0]["majority_accuracy"]),
                         per_snapshot_precision_at_5pct=dict(values=ps, min=min(ps.values()), median=float(np.median(list(ps.values()))),
                                                             max=max(ps.values())))
    return res


def block_compare(a, b, higher=True, cov=0.05, seed=0):
    """Paired snapshot-block bootstrap of precision at coverage: a minus b (both single predictors on the same rows)."""
    (sa, ya, snap), (sb, yb, _) = a, b
    assert np.array_equal(ya, yb)
    blocks = [np.flatnonzero(snap == u) for u in np.unique(snap)]
    rng = np.random.default_rng(seed); d = np.empty(B); pa = np.empty(B); pb = np.empty(B)
    for i in range(B):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        pa[i], pb[i] = prec_at(sa[idx], ya[idx], cov), prec_at(sb[idx], yb[idx], cov)
        d[i] = pa[i] - pb[i]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(point=[prec_at(sa, ya, cov), prec_at(sb, yb, cov)], diff_ci=[float(lo), float(np.mean(d)), float(hi)],
                a_ci=[float(np.percentile(pa, 2.5)), float(np.percentile(pa, 97.5))],
                b_ci=[float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))],
                verdict="better" if lo > 0 else "worse" if hi < 0 else "undetermined")


def main():
    st = C.require_clean()
    out = dict(stamp=st, use_cases={}, block={})
    A = {**arrival_arms(), **fill_arms(), **capacity_arms()}
    for uc, arms in A.items():
        out["use_cases"][uc] = summarise(arms)
        det = {n: p[0] for n, p in arms.items() if len(p) == 1}
        names = list(det)
        for c in COV:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    out["block"].setdefault(uc, {})[f"{a} vs {b} @ {int(c * 100)}%"] = block_compare(
                        (det[a][2], det[a][3], det[a][4]), (det[b][2], det[b][3], det[b][4]), cov=c, seed=int(c * 1000))
    C.dump(out, "phase20/stage1_decisions.json")
    for uc, arms in out["use_cases"].items():
        print(f"== {uc}")
        for n, r in arms.items():
            cv = {k: (round(v["precision"][1], 3) if v.get("precision") else None) for k, v in r["coverage"].items()}
            print(f"  {n:52s} {r['cls']:9s} {r['reachable']:11s} bar {r['operating_bar']} lift {r['lift_at_bar'] and round(r['lift_at_bar'], 2)} "
                  f"stageB {r['stage_b']} P@cov {cv} snapP5 [{r['per_snapshot_precision_at_5pct']['min']:.3f}, {r['per_snapshot_precision_at_5pct']['max']:.3f}]")
    for uc, d in out["block"].items():
        for k, v in d.items():
            if "@ 5%" in k:
                print(f"  BLOCK {uc} {k}: {v['verdict']} diff {[round(x, 4) for x in v['diff_ci']]}")


if __name__ == "__main__":
    main()
