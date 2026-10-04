"""Phase 19 Stage 3 scorer -- the bound neural arms against the STORED neural incumbent's 5-seed band. Torch process.

Metric functions: Phase 18's (ml/eval/phase18_score.py), unchanged; bands need all five seeds (deviation 171). RAW only
(the stored arrival and fill bundles carry recalibration; RAW and RECALIBRATED are never mixed, and every gate and
incumbent figure in Phases 17-19 is RAW).

Verdict per use case (pre-registered): GAIN = disjoint better on >= 1 primary metric and worse on none (capacity: >= 3 of
6 better); WORSE = disjoint worse on >= 1 and better on none; MIXED = better on one, worse on another; TIE otherwise.
A use case whose 5 seeds are not all complete is NOT SCORED (never reported as complete).

  python ml/eval/phase19_neural_score.py     # -> ml/artifacts/phase19/neural_score.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np
import phase18_score as S18
from phase19_score import bands5

SEEDS = C.V8_SEEDS
BUND19 = os.path.join(C.ART, "phase19", "bundles")
RF = {"arrival": "arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence",
      "fill": "fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence",
      "capacity": "capacity_strain/v8_mp_h4_lr0.00025_s{s}_rffwdload"}


def complete(task):
    return all(os.path.exists(os.path.join(BUND19, RF[task].format(s=s), "config.json")) and
               json.load(open(os.path.join(BUND19, RF[task].format(s=s), "config.json"))).get("complete") for s in SEEDS)


def verdict(task, cmp):
    better = [k for k, v in cmp.items() if v == "better"]; worse = [k for k, v in cmp.items() if v == "worse"]
    need = 3 if task == "capacity" else 1
    if better and worse:
        return "MIXED"
    if len(better) >= need:
        return "GAIN"
    if worse:
        return "WORSE"
    return "TIE"


def main():
    st = C.require_clean()
    out = dict(stamp=st, tasks={})
    for task in S18.TASK:
        if not complete(task):
            out["tasks"][task] = dict(status="NOT SCORED: fewer than 5 complete seeds",
                                      seeds_complete=[s for s in SEEDS if os.path.exists(os.path.join(BUND19, RF[task].format(s=s), "config.json"))])
            continue
        load = lambda s: {f: dict(np.load(os.path.join(BUND19, RF[task].format(s=s), f"preds_{f}.npz"))) for f in ("val", "test")}
        inc, _, inc_un = bands5(task, lambda s: S18.load_neural(task, s))
        rf, per, rf_un = bands5(task, load)
        D = S18.DIRECTION[task]
        cmp = {k: S18.compare(rf[k], inc[k], h) for k, h in D.items()}
        extra = {k: S18.compare(rf[k], inc[k], h) for k, h in S18.EXTRA[task].items()}
        logs = []
        for s in SEEDS:
            tl = json.load(open(os.path.join(BUND19, RF[task].format(s=s), "train_log.json")))
            logs.append(dict(seed=s, minutes=round(tl["wall_seconds_total"] / 60, 1), epochs=tl["epochs_run"], best_epoch=tl["best_epoch"],
                             sec_per_epoch=round(tl["sec_per_epoch"], 1), stop=tl["stop"], device=tl["device"],
                             concurrency_level=tl["concurrency_level"], commit=tl["code_commit"], best_val=tl["best_val"]))
        out["tasks"][task] = dict(row_family=RF[task].split("_rf")[1], incumbent=inc, phase19=rf, unreachable=dict(incumbent=inc_un, phase19=rf_un),
                                  vs_incumbent=cmp, extra_vs_incumbent=extra, verdict=verdict(task, cmp), per_seed=logs)
    C.dump(out, "phase19/neural_score.json")
    for task, r in out["tasks"].items():
        if "verdict" not in r:
            print(task, r); continue
        print(f"== {task} [{r['row_family']}]  VERDICT {r['verdict']}")
        for k in {**S18.DIRECTION[task], **S18.EXTRA[task]}:
            print(f"  {k:24s} incumbent {r['incumbent'][k] and [round(x, 4) for x in r['incumbent'][k]]}  phase19 {r['phase19'][k] and [round(x, 4) for x in r['phase19'][k]]}  "
                  f"{r['vs_incumbent'].get(k) or r['extra_vs_incumbent'].get(k)}")
        print("  minutes per seed:", [x["minutes"] for x in r["per_seed"]], "epochs:", [x["epochs"] for x in r["per_seed"]])


if __name__ == "__main__":
    main()
