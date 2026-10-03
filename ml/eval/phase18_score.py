"""Phase 18 scorer -- the primary metrics per use case, the proxy gate (Stages 4-5), and the shared scoring functions the
Stage 3 squeeze and the PRIVILEGED oracle scorer reuse. Torch process, no LightGBM. Exact per-seed filenames, no globs.

Primary metrics (reports/phase18-preregistration.md), all on TEST, operating points / thresholds on VALIDATION:
  arrival   lateness ROC-AUC against the as-of channel reference (docs/specs/lateness_metric.md); A3 median abs error in
            days on uncensored rows (Phase 17 A3: the neural arm's expected week from its distribution, LightGBM's point);
            C-index (for P6 only)
  fill      exact CRPS (22-cell); P(fill = 1) ROC-AUC; ECE-22 (for P6 only)
  capacity  Phase 15 UC3: score = P(strain > 1) from the P10/P50/P90; precision at 1 / 5 / 10% coverage; recall at
            p = 0.70 / 0.80 / 0.85 at the validation-chosen threshold per seed (phase15.pick_on_val)
Bands are [min, mean, max] over seeds 7/17/27/37/47. A difference counts only when the bands are DISJOINT.

The gate (pre-registered): a family PASSES a use case if at least one primary metric is disjointly better than BASE, none
is disjointly worse, and the shuffled arm is not disjointly better than BASE on the metrics that passed. Capacity needs
3 of its 6 points disjointly better.

  python ml/eval/phase18_score.py            # -> ml/artifacts/phase18/scores.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import phase5_heads as P5, folds
import phase5_metrics as M
from metrics import cindex
from heads import fill_cell
from phase11b_lateness import build_reference, lateness
from phase14_score import p_exceed
from phase15 import coverage_curve, pick_on_val, apply

SEEDS = C.V8_SEEDS
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
BUNDLE = {"arrival": "arrival_week/v8_lite_h4_lr0.00025_s{s}", "fill": "fill_rate/v8_none_h0_lr0.000125_s{s}",
          "capacity": "capacity_strain/v8_mp_h4_lr0.00025_s{s}"}
STORED_LGBM = {"arrival": "b5flat_reg", "fill": "b5flat22", "capacity": "b5flat_q"}
PR7 = os.path.join(C.ART, "phase7_preds")
PR18 = os.path.join(C.ART, "phase18", "preds")
COV = (0.01, 0.05, 0.10)
PBAR = (0.70, 0.80, 0.85)
# metric -> True if higher is better
DIRECTION = {"arrival": {"lateness_auc": True, "a3_median_abs_err_days": False},
             "fill": {"crps_exact": False, "p_full_auc": True},
             "capacity": {**{f"precision_at_{int(c * 100)}pct": True for c in COV},
                          **{f"recall_at_p{p:.2f}": True for p in PBAR}}}
EXTRA = {"arrival": {"cindex": True}, "fill": {"ece22": False}, "capacity": {}}


# ================================================================== references (as-of, built once)
_REF = {}


def arrival_reference():
    if not _REF:
        lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
        R, meta = build_reference("asof_channel_lead", "v8", lb, tr)
        _REF.update(val=R[P5.ordered(lb, va)], test=R[P5.ordered(lb, te)], meta=meta)
    return _REF


# ================================================================== per-arm loading
def load_neural(task, s):
    d = os.path.join(C.BUND, BUNDLE[task].format(s=s))
    return {f: dict(np.load(os.path.join(d, f"preds_{f}.npz"))) for f in ("val", "test")}


def load_flat(path_fmt, s):
    return {f: dict(np.load(path_fmt.format(s=s, f=f))) for f in ("val", "test")}


def expected_week(z):
    """Phase 17 A3: E[T] from the hazard distribution, censored mass at 13."""
    S = np.asarray(z["S"], float); pT = np.asarray(z["pT"], float)
    return (pT * np.arange(1, 13)).sum(1) + 13 * S[:, -1]


def point(task, z):
    """The arm's ranking / point prediction in arrival weeks (neural: expected week; LightGBM: the regression)."""
    return expected_week(z) if "S" in z else np.asarray(z["P"], float)


# ================================================================== metrics
def arrival_metrics(z, fold="test", with_cindex=False):
    R = arrival_reference()[fold]
    Y, EV = np.asarray(z["Y"], float), np.asarray(z["EV"], bool)
    ET = point("arrival", z)
    out = dict(lateness_auc=lateness(ET, Y, EV, R)[0],
               a3_median_abs_err_days=float(np.median(np.abs(7 * ET[EV] - 7 * Y[EV]))))
    if with_cindex:
        out["cindex"] = float(cindex(ET, Y, EV))
    return out


def fill_metrics(P, y):
    P = np.asarray(P, float); y = np.asarray(y, float)
    return dict(crps_exact=float(M.crps_exact_rows(P, y).mean()),
                p_full_auc=float(roc_auc_score((y >= 1).astype(int), P[:, 21])),
                ece22=float(M.ece_marginal(P, fill_cell(y))[0]))


def capacity_metrics(Qv, yv, Qt, yt):
    sv, st = p_exceed(np.asarray(Qv, float)), p_exceed(np.asarray(Qt, float))
    lv, lt = (np.asarray(yv) > 1).astype(int), (np.asarray(yt) > 1).astype(int)
    cur = {p["k"]: p for p in coverage_curve(st, lt)["points"]}
    out = {f"precision_at_{int(c * 100)}pct": float(cur[c]["precision"]) for c in COV}
    for p in PBAR:
        tau, maxp = pick_on_val(sv, lv, p)
        out[f"recall_at_p{p:.2f}"] = float(apply(st, lt, tau)["recall"]) if tau is not None else float("nan")
        out[f"test_precision_at_p{p:.2f}"] = float(apply(st, lt, tau)["precision"]) if tau is not None else float("nan")
    out["pinball_mean_test"] = float(np.mean([M.pinball_rows(np.asarray(yt, float), np.asarray(Qt, float)[:, i], q).mean()
                                              for i, q in enumerate(M.QS)]))
    return out


def metrics_for(task, zz, with_extra=False):
    if task == "arrival":
        return arrival_metrics(zz["test"], with_cindex=with_extra)
    if task == "fill":
        return fill_metrics(zz["test"]["P"], zz["test"]["Y"])
    return capacity_metrics(zz["val"]["P"], zz["val"]["Y"], zz["test"]["P"], zz["test"]["Y"])


def band(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    return [float(v.min()), float(v.mean()), float(v.max())] if len(v) else None


def arm_bands(task, loader, seeds=SEEDS, with_extra=False):
    per = [metrics_for(task, loader(s), with_extra) for s in seeds]
    return {k: band([p[k] for p in per]) for k in per[0]}, per


def compare(a, b, higher):
    """-> 'better' | 'worse' | 'undetermined' for arm a vs arm b (bands [min, mean, max])."""
    if a is None or b is None:
        return "n/a"
    if higher:
        return "better" if a[0] > b[2] else "worse" if a[2] < b[0] else "undetermined"
    return "better" if a[2] < b[0] else "worse" if a[0] > b[2] else "undetermined"


def gate(task, base, fam, shuf):
    D = DIRECTION[task]
    cmp = {k: compare(fam[k], base[k], h) for k, h in D.items()}
    better = [k for k, v in cmp.items() if v == "better"]; worse = [k for k, v in cmp.items() if v == "worse"]
    shuf_cmp = {k: compare(shuf[k], base[k], h) for k, h in D.items()}
    need = 3 if task == "capacity" else 1
    erased = all(shuf_cmp[k] != "better" for k in better)
    verdict = "PASS" if (len(better) >= need and not worse and erased) else "FAIL"
    why = ("disjoint gain on " + ", ".join(better) + "; shuffle erases it") if verdict == "PASS" else \
          ("disjointly worse on " + ", ".join(worse)) if worse else \
          (f"gain on {better} but the SHUFFLED arm also gains -- not the information") if better and not erased else \
          (f"only {len(better)} of 6 capacity points disjointly better (3 needed)") if task == "capacity" and better else \
          "no primary metric disjointly better (bands overlap)"
    return dict(verdict=verdict, why=why, vs_base=cmp, shuffled_vs_base=shuf_cmp)


def main():
    st = C.require_clean()
    out = dict(stamp=st, reference=None, arms={}, gate={})
    for task in TASK:
        t = TASK[task]
        A = {}
        A["neural_incumbent"] = arm_bands(task, lambda s: load_neural(task, s), with_extra=True)
        A["lgbm_flat_stored"] = arm_bands(task, lambda s: load_flat(os.path.join(PR7, f"v8_{t}_{STORED_LGBM[task]}_s{{s}}_{{f}}.npz"), s),
                                          with_extra=True)
        for arm in ("fwd_load", "fwd_load_shuf", "pulse", "pulse_shuf"):
            fmt = os.path.join(PR18, f"v8_{t}_p18_{arm}_s{{s}}_{{f}}.npz")
            if all(os.path.exists(fmt.format(s=s, f="test")) for s in SEEDS):
                A[arm] = arm_bands(task, lambda s, fmt=fmt: load_flat(fmt, s), with_extra=True)
        # row identity: every arm's labels equal the incumbent's, element by element
        ref = load_neural(task, 7)
        for arm, fmt in (("lgbm_flat_stored", os.path.join(PR7, f"v8_{t}_{STORED_LGBM[task]}_s7_{{f}}.npz")),
                         ("fwd_load", os.path.join(PR18, f"v8_{t}_p18_fwd_load_s7_{{f}}.npz"))):
            for f in ("val", "test"):
                if os.path.exists(fmt.format(f=f)):
                    z = np.load(fmt.format(f=f))
                    assert np.array_equal(z["Y"], ref[f]["Y"]) and np.array_equal(z["EV"], ref[f]["EV"]), f"{task} {arm} {f}: rows differ"
        out["arms"][task] = {a: dict(bands=b, per_seed=p) for a, (b, p) in A.items()}
        base = A["lgbm_flat_stored"][0]
        for fam in ("fwd_load", "pulse"):
            if fam in A and f"{fam}_shuf" in A:
                out["gate"][f"{task}|{fam}"] = gate(task, base, A[fam][0], A[f"{fam}_shuf"][0])
    out["reference"] = {k: v for k, v in arrival_reference()["meta"].items() if not isinstance(v, (list, dict))}
    C.dump(out, "phase18/scores.json")
    for task in TASK:
        print(f"== {task}")
        for a, d in out["arms"][task].items():
            print(f"  {a:18s}", {k: [round(x, 4) for x in v] for k, v in d["bands"].items() if v})
    print(json.dumps(out["gate"], indent=1))


if __name__ == "__main__":
    main()
