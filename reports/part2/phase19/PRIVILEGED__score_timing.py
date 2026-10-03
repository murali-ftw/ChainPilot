"""Phase 19 Stage 2 -- PRIVILEGED timing test scorer. Reuses Phase 18's metric functions (reports -> ml, never back).

BASE vs BASE + TRUE creation week (privileged, no state) vs BASE + cadence (legitimate), arrival lateness ROC-AUC and A3;
fill shown for context. Rule (pre-registered): forecasting the order-raising time is worth building only if the privileged
arm adds >= 0.05 lateness AUC over BASE AND cadence recovers >= 30% of that addition (5-seed means).

  python reports/part2/phase19/PRIVILEGED__score_timing.py   # -> reports/part2/phase19/PRIVILEGED__timing_scores.json
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ml", "eval"))
import phase12_common as C
import numpy as np
import phase18_score as S18
from lateness_metric import assert_no_privileged_headline

PRED = os.path.join(HERE, "preds")
PR19 = os.path.join(C.ART, "phase19", "preds")
P18O = os.path.join(REPO, "reports", "phase18", "oracle", "preds")


def main():
    out = dict(stamp=C.stamp(), tasks={})
    for task in ("arrival", "fill"):
        t = S18.TASK[task]
        arms = {"BASE": os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz"),
                "PRIVILEGED__creation_week_only": os.path.join(PRED, f"v8_{t}_PRIVILEGED__timing_only_s{{s}}_{{f}}.npz"),
                "BASE+cadence": os.path.join(PR19, f"v8_{t}_p19_cadence_s{{s}}_{{f}}.npz"),
                "PRIVILEGED__oracle_state (Phase 18: timing + state)": os.path.join(P18O, f"v8_{t}_PRIVILEGED__oracle_state_s{{s}}_{{f}}.npz"),
                "PRIVILEGED__oracle_state_notiming (Phase 18)": os.path.join(P18O, f"v8_{t}_PRIVILEGED__oracle_state_notiming_s{{s}}_{{f}}.npz")}
        out["tasks"][task] = {a: S18.arm_bands(task, lambda s, fm=fm: S18.load_flat(fm, s), with_extra=True)[0] for a, fm in arms.items()}
    A = out["tasks"]["arrival"]
    b, tm, cd = A["BASE"]["lateness_auc"][1], A["PRIVILEGED__creation_week_only"]["lateness_auc"][1], A["BASE+cadence"]["lateness_auc"][1]
    add = tm - b; rec = (cd - b) / add if add > 0 else float("nan")
    worth = bool(add >= 0.05 and rec >= 0.30)
    out["PRIVILEGED__timing_rule"] = dict(base=b, creation_week_only=tm, cadence=cd, PRIVILEGED__timing_addition=add,
                                          cadence_share_recovered=rec, rule="addition >= 0.05 AND cadence recovers >= 30%",
                                          worth_building=worth,
                                          verdict=("forecasting the order-raising time IS worth building" if worth else
                                                   "arrival headroom is future supplier state, not reachable from as-of data"),
                                          vs_disjoint={"creation_week_only vs BASE": S18.compare(A["PRIVILEGED__creation_week_only"]["lateness_auc"], A["BASE"]["lateness_auc"], True),
                                                       "cadence vs BASE": S18.compare(A["BASE+cadence"]["lateness_auc"], A["BASE"]["lateness_auc"], True)})
    assert_no_privileged_headline({"PRIVILEGED__timing_addition": 0}, headline=["verdict"])
    json.dump(out, open(os.path.join(HERE, "PRIVILEGED__timing_scores.json"), "w"), indent=1, default=float)
    for task, arms in out["tasks"].items():
        print("==", task)
        for a, bd in arms.items():
            print(f"  {a:52s}", {k: [round(x, 4) for x in v] for k, v in bd.items() if v})
    print(json.dumps(out["PRIVILEGED__timing_rule"], indent=1))


if __name__ == "__main__":
    main()
