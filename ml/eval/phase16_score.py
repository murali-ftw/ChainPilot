"""Phase 16 -- score every arm against its own incumbent. Exact per-seed bundle names; NO globs (deviation 117).

Arrival (C-index on validation and test; lateness ROC-AUC under the ADOPTED t0 metric on test):
  A0/B0  v8_lite_h4_lr0.00025_s{s}                                   stored incumbent, not retrained
  A1     v8_lite_h4_lr0.00025_s{s}_encshare_traj
  A2     v8_lite_h4_lr0.00025_s{s}_encshare_traj_tdeltazeroed          the control: must NOT be disjoint from A0
  B1     v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnammms
  B2     v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnasum_ldk1
  B3     v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnammms_ldk{k}         k from phase16_k_selection.json
Capacity (mean pinball on validation and test; lower is better):
  B0     v8_mp_h4_lr0.00025_s{s}                                     stored incumbent
  B1     v8_mp_h4_lr0.00025_s{s}_encheteromp_pna_pnammms
  B2 = B0 and B3 = B1 by construction (HeteroMP has no softmax to bypass) -- not trained, not scored twice.

Verdicts (vs own incumbent, per surface): "gain" / "worse" only when the 5-seed [min, max] bands are DISJOINT;
otherwise "undetermined" (a tie). PASS = not disjointly worse on any surface. If A2 is disjoint from A0 on any
surface, A1 and A2 are both INVALID.
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np
import phase5_heads as P5, folds
from lateness_metric import adopted_reference, lateness_scores

SEEDS = (7, 17, 27, 37, 47)
HIGHER = {"arrival_week": True, "capacity_strain": False}


def arms():
    k = json.load(open(os.path.join(C.ART, "phase16_k_selection.json")))["selected_k"] \
        if os.path.exists(os.path.join(C.ART, "phase16_k_selection.json")) else None
    a = {"A0": "v8_lite_h4_lr0.00025_s{s}", "A1": "v8_lite_h4_lr0.00025_s{s}_encshare_traj",
         "A2": "v8_lite_h4_lr0.00025_s{s}_encshare_traj_tdeltazeroed",
         "B1": "v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnammms",
         "B2": "v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnasum_ldk1"}
    if k is not None:
        a["B3"] = "v8_lite_h4_lr0.00025_s{s}_encshare_pna_pnammms_ldk" + str(k)
    return {"arrival_week": a,
            "capacity_strain": {"B0": "v8_mp_h4_lr0.00025_s{s}", "B1": "v8_mp_h4_lr0.00025_s{s}_encheteromp_pna_pnammms"}}, k


def band(v):
    return dict(n=len(v), mean=float(np.mean(v)), lo=float(min(v)), hi=float(max(v)), values=[float(x) for x in v]) \
        if v else None


def verdict(arm, inc, higher):
    if not arm or not inc or arm["n"] < 5:
        return "incomplete"
    if arm["lo"] > inc["hi"]:
        return "gain" if higher else "worse"
    if arm["hi"] < inc["lo"]:
        return "worse" if higher else "gain"
    return "undetermined"


def main():
    st = C.require_clean()
    A, k = arms()
    lba = P5.labels("v8", "arrival_week")
    tr, va, te = folds.fixed_split(lba.snapshot_date)
    R, _ = adopted_reference("v8", lba, tr)
    Rte = R[P5.ordered(lba, te)]
    out = dict(stamp=st, selected_k=k, tasks={})
    for task, arm_pats in A.items():
        res = {}
        for arm, pat in arm_pats.items():
            v = dict(val=[], test=[], late=[], bundles=[], stamps=[], epochs=[])
            for s in SEEDS:
                d = os.path.join(C.BUND, task, pat.format(s=s))
                if not os.path.exists(os.path.join(d, "preds_test.npz")):
                    continue
                cfg = json.load(open(os.path.join(d, "config.json")))
                assert cfg["seed"] == s and cfg.get("complete"), d
                zv, zt = dict(np.load(os.path.join(d, "preds_val.npz"))), dict(np.load(os.path.join(d, "preds_test.npz")))
                v["val"].append(P5.val_score(task, zv)); v["test"].append(P5.val_score(task, zt))
                if task == "arrival_week":
                    v["late"].append(lateness_scores({"a": zt["P"]}, zt["Y"], zt["EV"], Rte)["a"])
                v["bundles"].append(os.path.basename(d)); v["stamps"].append(cfg["stamps"]["model_version"])
                v["epochs"].append(json.load(open(os.path.join(d, "train_log.json")))["epochs_run"])
            res[arm] = dict(val=band(v["val"]), test=band(v["test"]), lateness_test=band(v["late"]),
                            bundles=v["bundles"], stamps=v["stamps"], dirty=[x for x in v["stamps"] if "+dirty" in x],
                            epochs=v["epochs"])
        inc = "A0" if task == "arrival_week" else "B0"
        surfs = ("val", "test", "lateness_test") if task == "arrival_week" else ("val", "test")
        cmp = {}
        for arm in res:
            if arm == inc:
                continue
            ref = "A0"
            cmp[arm] = {sf: verdict(res[arm][sf], res[ref if task == "arrival_week" else inc][sf], HIGHER[task])
                        for sf in surfs}
            cmp[arm]["PASS_not_disjointly_worse"] = all(x != "worse" for x in cmp[arm].values() if isinstance(x, str)
                                                        and x != "incomplete") and "incomplete" not in cmp[arm].values()
        if task == "arrival_week" and "A2" in cmp:
            ctrl = [sf for sf in surfs if cmp["A2"][sf] in ("gain", "worse")]
            cmp["control_A2"] = dict(disjoint_from_A0_on=ctrl, A1_A2_valid=not ctrl and res["A2"]["val"] is not None
                                     and res["A2"]["val"]["n"] == 5)
        out["tasks"][task] = dict(arms=res, comparisons=cmp)
        for arm, r in res.items():
            print(task, arm, {sf: (round(r[sf]["mean"], 5), round(r[sf]["lo"], 5), round(r[sf]["hi"], 5), r[sf]["n"])
                              for sf in surfs if r[sf]}, "dirty" if r["dirty"] else "")
        print(task, json.dumps(cmp))
    print(C.dump(out, "phase16_scores.json"))


if __name__ == "__main__":
    main()
