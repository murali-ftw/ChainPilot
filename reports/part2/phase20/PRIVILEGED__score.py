"""Phase 20 -- PRIVILEGED scorer: Stage 3 recovery shares (against each world's own hindsight-load arm) and the Stage 4b
reorder-trigger probe. Reuses ml/eval/phase20_score.py's world-aware metrics (reports -> ml, never back).

  recovery share  = (BASE+fwd_load+fwd_pred - BASE+fwd_load) / (hindsight_load - BASE+fwd_load), 5-seed means, per metric
                    (for a lower-is-better metric both differences are taken as improvements, so the share keeps its sign)
  probe share     = (BASE+true position - BASE) / (BASE+true creation week - BASE) on lateness ROC-AUC (Phase 19's timing arm)

  python reports/part2/phase20/PRIVILEGED__score.py      # -> reports/part2/phase20/PRIVILEGED__scores.json
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ml", "eval"))
import phase12_common as C
import numpy as np
import phase20_score as P20
import phase18_score as S18
from lateness_metric import assert_no_privileged_headline

SEEDS = C.V8_SEEDS
HIND = {"v8": os.path.join(REPO, "reports", "phase18", "oracle", "preds", "v8_{t}_PRIVILEGED__hindsight_load_s{{s}}_{{f}}.npz"),
        "v8w1002": os.path.join(HERE, "preds", "v8w1002_{t}_PRIVILEGED__hindsight_load_s{{s}}_{{f}}.npz")}
TIMING = os.path.join(REPO, "reports", "part2", "phase19", "preds", "v8_arrival_week_PRIVILEGED__timing_only_s{s}_{f}.npz")
PROBE = os.path.join(HERE, "preds", "v8_arrival_week_PRIVILEGED__reorder_probe_s{s}_{f}.npz")


def bands_from(world, task, path_fmt):
    per = [P20.metrics(world, task, {f: dict(np.load(path_fmt.format(s=s, f=f))) for f in ("val", "test")}) for s in SEEDS]
    out = {}
    for k in per[0]:
        v = [p[k] for p in per]
        out[k] = S18.band(v) if all(x is not None and np.isfinite(x) for x in v) else None
    return out


def main():
    P20.PW.register()
    out = dict(stamp=C.stamp(), stage3={}, stage4b={})
    for world in ("v8", "v8w1002"):
        for task, t in S18.TASK.items():
            hp = HIND[world].format(t=t)
            if not all(os.path.exists(hp.format(s=s, f="test")) for s in SEEDS) or \
               not all(os.path.exists(P20.fmt(world, task, "fwd_load_pred").format(s=s, f="test")) for s in SEEDS):
                out["stage3"][f"{world}|{task}"] = "NOT RUN (arms missing)"; continue
            fl = P20.bands(world, task, "fwd_load")[0]; fp = P20.bands(world, task, "fwd_load_pred")[0]
            base = P20.bands(world, task, "base")[0]; hd = bands_from(world, task, hp)
            rows = {}
            for k, hi in S18.DIRECTION[task].items():
                if not (fl[k] and fp[k] and hd[k]):
                    rows[k] = "UNREACHABLE on some seed"; continue
                sg = 1 if hi else -1
                gap = sg * (hd[k][1] - fl[k][1]); got = sg * (fp[k][1] - fl[k][1])
                rows[k] = dict(base=base[k][1] if base[k] else None, fwd_load=fl[k][1], fwd_load_pred=fp[k][1], PRIVILEGED__hindsight=hd[k][1],
                               gap=gap, gain=got, recovery_share=(got / gap if gap > 0 else None))
            out["stage3"][f"{world}|{task}"] = rows
    base = P20.bands("v8", "arrival", "base")[0]; tm = bands_from("v8", "arrival", TIMING); pr = bands_from("v8", "arrival", PROBE)
    add_t = tm["lateness_auc"][1] - base["lateness_auc"][1]; add_p = pr["lateness_auc"][1] - base["lateness_auc"][1]
    out["stage4b"] = dict(base=base, PRIVILEGED__creation_week_only=tm, PRIVILEGED__reorder_probe=pr,
                          probe_recovery_share=add_p / add_t, probe_vs_base=S18.compare(pr["lateness_auc"], base["lateness_auc"], True),
                          a3_probe_vs_base=S18.compare(pr["a3_median_abs_err_days"], base["a3_median_abs_err_days"], False),
                          p7_threshold_lateness_auc=0.81, p7_met=bool(pr["lateness_auc"][1] >= 0.81 and add_p / add_t >= 0.5),
                          caveat="the simulator's position is exact and instantly current; a real ERP's book position lags and carries recording error, so this is an UPPER bound")
    assert_no_privileged_headline({"PRIVILEGED__x": 0}, headline=["probe_recovery_share"])
    json.dump(out, open(os.path.join(HERE, "PRIVILEGED__scores.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
