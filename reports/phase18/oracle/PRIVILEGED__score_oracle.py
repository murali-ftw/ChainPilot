"""Phase 18 Stage 2 scorer -- PRIVILEGED. Oracle ceiling per use case, and the Stage 4 hindsight-load upper bound.

Reuses ml/eval/phase18_score.py's metric functions (the dependency runs reports -> ml, never the other way). Every
quantity written here carries the PRIVILEGED__ prefix; lateness_metric.assert_no_privileged_headline is applied to the
headline set so an oracle number cannot be reported as a model claim.

Headroom (pre-registered) = ORACLE metric - MODEL metric, 5-seed means, on: arrival lateness ROC-AUC; fill P(fill = 1)
ROC-AUC; capacity precision at 5% coverage. < 0.02 AT CEILING; 0.02-0.05 SMALL; > 0.05 REAL. The brief's tier
(`oracle`) decides; `oracle_state` is reported beside it.

  python reports/phase18/oracle/PRIVILEGED__score_oracle.py      # -> reports/phase18/oracle/PRIVILEGED__oracle_scores.json
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
HEAD = {"arrival": "lateness_auc", "fill": "p_full_auc", "capacity": "precision_at_5pct"}


def verdict(h):
    return "AT CEILING" if h < 0.02 else "SMALL" if h <= 0.05 else "REAL"


def main():
    st = C.stamp()
    out = dict(stamp=st, headroom_metric=HEAD, use_cases={})
    for task, t in S18.TASK.items():
        A = {}
        A["model (neural incumbent)"] = S18.arm_bands(task, lambda s: S18.load_neural(task, s), with_extra=True)[0]
        A["lgbm_flat"] = S18.arm_bands(task, lambda s: S18.load_flat(os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz"), s),
                                       with_extra=True)[0]
        for arm in ("oracle", "oracle_state", "hindsight_load", "oracle_state_notiming"):
            fmt = os.path.join(PRED, f"v8_{t}_PRIVILEGED__{arm}_s{{s}}_{{f}}.npz")
            if not all(os.path.exists(fmt.format(s=s, f="test")) for s in S18.SEEDS):
                continue
            A[f"PRIVILEGED__{arm}"] = S18.arm_bands(task, lambda s, fmt=fmt: S18.load_flat(fmt, s), with_extra=True)[0]
        fmt = os.path.join(S18.PR18, f"v8_{t}_p18_fwd_load_s{{s}}_{{f}}.npz")
        if all(os.path.exists(fmt.format(s=s, f="test")) for s in S18.SEEDS):
            A["fwd_load (feature, as-of)"] = S18.arm_bands(task, lambda s: S18.load_flat(fmt, s), with_extra=True)[0]
        k = HEAD[task]
        model, orc, ors = A["model (neural incumbent)"][k], A["PRIVILEGED__oracle"][k], A["PRIVILEGED__oracle_state"][k]
        h, hs = orc[1] - model[1], ors[1] - model[1]
        out["use_cases"][task] = dict(
            arms=A, headroom_metric=k,
            PRIVILEGED__headroom=h, PRIVILEGED__headroom_state=hs, verdict=verdict(h), verdict_state_tier=verdict(hs),
            oracle_vs_model_disjoint=S18.compare(orc, model, True), oracle_vs_lgbm_flat=orc[1] - A["lgbm_flat"][k][1],
            hindsight_vs_lgbm_flat={m: S18.compare(A["PRIVILEGED__hindsight_load"][m], A["lgbm_flat"][m], hi)
                                    for m, hi in S18.DIRECTION[task].items()})
    assert_no_privileged_headline({k: 0 for u in out["use_cases"].values() for k in u if k.startswith("PRIVILEGED__")},
                                  headline=["verdict"])
    path = os.path.join(HERE, "PRIVILEGED__oracle_scores.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    for task, u in out["use_cases"].items():
        print(f"== {task}: headroom {u['PRIVILEGED__headroom']:+.4f} -> {u['verdict']} (state tier {u['PRIVILEGED__headroom_state']:+.4f} -> {u['verdict_state_tier']})")
        for a, b in u["arms"].items():
            print(f"  {a:28s}", {m: [round(x, 4) for x in v] for m, v in b.items() if v})


if __name__ == "__main__":
    main()
