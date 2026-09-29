"""Phase 15 supplement.

(1) BEST-ARM SELECTION ON VALIDATION. phase15.json stores TEST recall at each validation-chosen bar; choosing Stage G's
    best arm by it would be choosing on test. This computes each arm's VALIDATION recall at p = 0.85 (and its validation
    max precision) so Stage G can select on validation only.
(2) STAGE F, CORRECTED UPPER BOUND. phase15.py's PRIVILEGED oracle scaled validation AND test by the TEST ratio -- one
    scalar for both periods, which a re-fitted threshold absorbs, so it cannot show what level tracking buys. A perfect
    level tracker corrects EACH PERIOD by its own ratio: validation by val mean(y)/mean(P50), test by test mean(y)/mean(P50).
    That test ratio is future information: PRIVILEGED, an upper bound, not a result.
"""
from __future__ import annotations
import json
import phase12_common as C
import numpy as np
import phase15 as P
from phase14_score import p_exceed


def val_recall(pairs, p):
    out = []
    for sv, yv, st, yt in pairs:
        tau, maxp = P.pick_on_val(sv, yv, p)
        rec = float(((np.asarray(sv) >= tau) & (np.asarray(yv) == 1)).sum() / max(np.asarray(yv).sum(), 1)) if tau is not None else 0.0
        out.append(dict(val_recall=rec, val_max_precision=maxp))
    return dict(val_recall=P.band([x["val_recall"] for x in out]), val_max_precision=P.band([x["val_max_precision"] for x in out]))


def tracking_oracle():
    import phase5_heads as P5, folds
    lst = []
    for s in P.SEEDS:
        z = {f: np.load(f"{P.BC}/v8_mp_h4_lr0.00025_s{s}/preds_{f}.npz") for f in ("val", "test")}
        rv = float(z["val"]["Y"].mean() / z["val"]["P"][:, 1].mean()); rt = float(z["test"]["Y"].mean() / z["test"]["P"][:, 1].mean())
        lst.append((p_exceed(np.asarray(z["val"]["P"], float) * rv), (z["val"]["Y"] > 1).astype(int),
                    p_exceed(np.asarray(z["test"]["P"], float) * rt), (z["test"]["Y"] > 1).astype(int), rv, rt))
    return [x[:4] for x in lst], [dict(r_val=x[4], r_test_PRIVILEGED=x[5]) for x in lst]


def main():
    st = C.require_clean()
    fill, _ = P.fill_arrays(); arr = P.arrival_arrays(); cap, _, _ = P.capacity_arrays(); sho = P.shortage_arrays(); sim, _ = P.sim_arrays()
    ucs = {**arr, **fill, "UC3": cap, "UC4": sho, **sim}
    R = dict(stamp=st, selection={})
    for uc, arms in ucs.items():
        for arm, pairs in arms.items():
            R["selection"].setdefault(uc, {})[arm] = val_recall(pairs, 0.85)
    pairs, ratios = tracking_oracle()
    R["stage_f_tracking_oracle_PRIVILEGED"] = dict(ratios=ratios, analysis=P.analyse(pairs))
    a = R["stage_f_tracking_oracle_PRIVILEGED"]["analysis"]
    print({p: {k: v for k, v in a["recall_at_precision"][p].items()} for p in ("0.7", "0.8", "0.85", "0.9")})
    print(C.dump(R, "phase15_supp.json"))


if __name__ == "__main__":
    main()
