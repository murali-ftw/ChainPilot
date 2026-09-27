"""Phase 12 C3 -- score the part-relation ablation against the full h4 arm, h0 and the shuffled control.

Arms (v8, five model seeds each unless stated):
  full     v8_{lite,mp}_h4_lr0.00025_s{7..47}             (existing)
  droppart v8_{lite,mp}_h4_lr0.00025_s{7..47}_droppart    (trained in C3)
  h0       v8_none_h0_lr0.00025_s{7..47}                   (s7/17/27 existing; s37/47 trained in C3)
  shuffled v8_{lite,mp}_h4_lr0.00025_s{s}_shuf{s}          (Phase 11A/B, context only)

Metrics: validation surface (arrival C-index, capacity mean pinball -- the selection metrics) and test (the same, plus
arrival lateness ROC-AUC under the ADOPTED t0 metric). A difference exists only when five-seed bands are DISJOINT.

C3.3: the part path's share of the h4-over-h0 gap = (full - droppart) / (full - h0), sign-oriented so that a
positive share means removing the part relation costs performance.
"""
from __future__ import annotations
import glob, os
import phase12_common as C
import numpy as np, pandas as pd
import phase5_heads as P5, folds
from lateness_metric import adopted_reference, lateness_scores

ARMS = {"arrival_week": dict(full="v8_lite_h4_lr0.00025_s{s}", droppart="v8_lite_h4_lr0.00025_s{s}_droppart",
                             h0="v8_none_h0_lr0.00025_s{s}", shuffled="v8_lite_h4_lr0.00025_s{s}_shuf{s}"),
        "capacity_strain": dict(full="v8_mp_h4_lr0.00025_s{s}", droppart="v8_mp_h4_lr0.00025_s{s}_droppart",
                                h0="v8_none_h0_lr0.00025_s{s}", shuffled="v8_mp_h4_lr0.00025_s{s}_shuf{s}")}
HIGHER = {"arrival_week": True, "capacity_strain": False}


def band(v):
    v = [x for x in v if x is not None]
    return dict(n=len(v), mean=float(np.mean(v)), lo=float(min(v)), hi=float(max(v)), values=[float(x) for x in v]) if v else None


def disjoint(a, b):
    return bool(a and b and (a["lo"] > b["hi"] or b["lo"] > a["hi"]))


def main():
    st = C.require_clean()
    out = dict(stamp=st, tasks={})
    lba = P5.labels("v8", "arrival_week")
    tr, va, te = folds.fixed_split(lba.snapshot_date)
    R, _ = adopted_reference("v8", lba, tr)
    Rte = R[P5.ordered(lba, te)]
    for task, arms in ARMS.items():
        res = {}
        for arm, pat in arms.items():
            vals = dict(val=[], test=[], late=[], bundles=[], stamps=[])
            for s in C.V8_SEEDS:
                d = f"{C.BUND}/{task}/{pat.format(s=s)}"
                if not os.path.exists(f"{d}/preds_test.npz"):
                    continue
                zv, zt = dict(np.load(f"{d}/preds_val.npz")), dict(np.load(f"{d}/preds_test.npz"))
                vals["val"].append(P5.val_score(task, zv)); vals["test"].append(P5.val_score(task, zt))
                if task == "arrival_week":
                    vals["late"].append(lateness_scores({"a": zt["P"]}, zt["Y"], zt["EV"], Rte)["a"])
                vals["bundles"].append(os.path.basename(d))
                import json
                cfg = json.load(open(f"{d}/config.json"))
                vals["stamps"].append(cfg["stamps"]["model_version"])
            res[arm] = dict(val=band(vals["val"]), test=band(vals["test"]),
                            lateness_test=band(vals["late"]) if vals["late"] else None,
                            bundles=vals["bundles"], dirty=[x for x in vals["stamps"] if "+dirty" in x])
        sgn = 1.0 if HIGHER[task] else -1.0
        cmp = {}
        for surf in ("val", "test") + (("lateness_test",) if task == "arrival_week" else ()):
            f, dp, h0, sh = (res[a][surf] for a in ("full", "droppart", "h0", "shuffled"))
            gap = sgn * (f["mean"] - h0["mean"]) if f and h0 else None
            part = sgn * (f["mean"] - dp["mean"]) if f and dp else None
            edge = sgn * (f["mean"] - sh["mean"]) if f and sh else None
            cmp[surf] = dict(full_vs_droppart_disjoint=disjoint(f, dp), full_better_than_droppart=part is not None and part > 0,
                             droppart_vs_h0_disjoint=disjoint(dp, h0), droppart_vs_shuffled_disjoint=disjoint(dp, sh),
                             h4_over_h0_gap=gap, part_path_cost=part,
                             part_share_of_gap=(part / gap) if gap else None,
                             whole_edge_share_of_gap=(edge / gap) if gap and edge is not None else None)
        out["tasks"][task] = dict(arms=res, comparisons=cmp)
        print(task, {k: {a: (v["mean"], v["n"]) for a, v in ((a, res[a][k]) for a in res) if v} for k in ("val", "test")})
        print(task, cmp)
    print(C.dump(out, "phase12_c3_scores.json"))


if __name__ == "__main__":
    main()
