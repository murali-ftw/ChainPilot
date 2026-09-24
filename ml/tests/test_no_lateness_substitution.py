"""Every arm's lateness score must be computed from THAT ARM'S OWN prediction.

Deviation 72 / Phase 11C Stage A. `phase7_score.py`'s `arrival_promise` branch replaced the
promise arm's stored prediction with a constant before computing lateness:

    late = M.arrival_scores(np.full(len(Y), float(spec["const"])), Y, EV, AUX)
    out["roc_auc_late"] = late["roc_auc_late"]

The consequence was not local. Because `pl = ET - AUX`, ANY constant ET ranks by -AUX, so the
"promise date" arm and a naive constant arm are the same computation and report the same number.
Phase 8 §5's "the head adds lateness information beyond the promise date -- 8 of 8 windows,
margin +0.0014 to +0.0306" was measured against that constant, on all eight cells, and the promise
arm's own prediction scores 0.5000 (one distinct lateness value).

This test fails if any arm's `roc_auc_late` differs from the value its own stored prediction
produces. Run it against the pre-fix scorer and it FAILS on all eight Phase 8 promise cells; that
demonstration is in reports/part2/phase-11c.md §2.

    python ml/tests/test_no_lateness_substitution.py [--preds DIR]
"""
from __future__ import annotations
import os, sys, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval"),
                os.path.join(HERE, "..", "models")]
import numpy as np
from metrics import roc_auc


def honest_lateness(P, Y, EV, AUX):
    """The only definition: this arm's own prediction, minus the reference."""
    m = np.asarray(EV, bool) & np.isfinite(AUX)
    yl = (Y[m] > AUX[m]).astype(int)
    pl = np.asarray(P, float)[m] - AUX[m]
    if yl.min() == yl.max():
        return float("nan")
    if len(np.unique(np.round(pl, 9))) <= 1:
        return 0.5                      # a constant score cannot discriminate; say so
    return float(roc_auc(yl, pl))


def check_file(path, scorer):
    z = np.load(path)
    if "AUX" not in z or not np.isfinite(z["AUX"]).any():
        return None
    P, Y, EV, AUX = z["P"], z["Y"], z["EV"].astype(bool), z["AUX"]
    if P.ndim > 1:
        return None                     # distributional arms are scored elsewhere
    want = honest_lateness(P, Y, EV, AUX)
    got = scorer(path)
    if got is None or not np.isfinite(want):
        return None
    return dict(file=os.path.basename(path), honest=want, reported=got,
                delta=abs(got - want), SUBSTITUTED=bool(abs(got - want) > 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=os.path.join(HERE, "..", "artifacts", "backtest", "preds"))
    ap.add_argument("--pattern", default="*arrival_week*_test.npz")
    a = ap.parse_args()
    import phase7_score as PS
    import phase5_metrics as M
    # The bootstrap CI costs ~1000 roc_auc evaluations per file and contributes nothing to this
    # test, which compares POINT estimates. Neutralise the CI only -- the metric itself still
    # runs through the real scorer, which is the whole point of routing through score_job.
    M.bootstrap_ci = lambda *a, **k: (float("nan"), float("nan"))

    def scorer(path):
        """Route the file through the real scorer exactly as the pipeline does."""
        base = os.path.basename(path)
        m = PS.BT_NAME.match(base)
        if not m:
            return None
        pre, w, task, o, name, fold = m.groups()
        pre = pre or ""
        if task != "arrival_week":
            return None
        kind = ("arrival_promise" if pre == "PROMISE_"
                else "arrival_point" if "b5flat_reg" in name else "arrival_dist")
        spec = dict(kind=kind, path=path, label=base)
        if kind == "arrival_promise":
            spec["const"] = 1.0
        try:
            _, out = PS.score_job(spec)
        except Exception:
            return None
        v = out.get("roc_auc_late")
        return float(v[0]) if isinstance(v, (list, tuple)) else (float(v) if v is not None else None)

    rows = []
    for f in sorted(glob.glob(os.path.join(a.preds, a.pattern))):
        r = check_file(f, scorer)
        if r:
            rows.append(r)
    bad = [r for r in rows if r["SUBSTITUTED"]]
    print(f"arrival prediction files checked: {len(rows)}")
    print(f"arms whose reported lateness != their own prediction's: {len(bad)}\n")
    for r in bad:
        print(f"  SUBSTITUTED  {r['file']:52s} own {r['honest']:.4f}  reported {r['reported']:.4f}"
              f"  delta {r['delta']:+.4f}")
    assert not bad, (
        f"{len(bad)} arm(s) report a lateness score not computed from their own prediction. "
        f"A substituted value makes two different baselines indistinguishable and mislabels "
        f"whatever they are compared against (deviation 72).")
    print("PASS: every arm's lateness is computed from its own prediction.")


if __name__ == "__main__":
    main()
