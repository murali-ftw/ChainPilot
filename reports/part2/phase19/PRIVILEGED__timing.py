"""Phase 19 Stage 2 -- PRIVILEGED timing-only arm. Torch-free (LightGBM process).

BASE (Phase 7's LightGBM-flat arm, unchanged) + ONE privileged column: the line's TRUE creation week minus t0
(`_sim.npz` `pt`, joined by PO-line index). NO state of any kind -- no supplier state, transit, regime, K or load.
Phase 18's oracle knew the creation week AND the state at that week; this arm isolates the timing.

Arrival and fill only: their rows are PO lines (capacity's rows are channels, which have no creation week).
Hidden generator state is never a feature: this file and its outputs live under reports/part2/phase19/PRIVILEGED__*,
and ml/tests/test_phase19_isolation.py asserts no ml/ module imports or reads them.

Before any fit the arrival label is re-derived from the simulation (arrival week - t0) and asserted equal on every
uncensored row, which proves the join.

  python reports/part2/phase19/PRIVILEGED__timing.py --task arrival
"""
from __future__ import annotations
import lightgbm as lgb
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ML = os.path.join(REPO, "ml")
sys.path[:0] = [os.path.join(ML, "baselines"), ML, os.path.join(ML, "train"), os.path.join(ML, "data"), os.path.join(ML, "eval")]
import numpy as np, pandas as pd
import phase7_fit as P7
import folds as FO
import phase12_common as C
assert "torch" not in sys.modules

SIM = os.path.join(REPO, "db", "gen_v8", "seed_1001", "_sim.npz")
OUT = os.path.join(HERE, "preds")
LOG = os.path.join(HERE, "PRIVILEGED__timing_fit.json")
TASK = {"arrival": "arrival_week", "fill": "fill_rate"}
W0 = pd.Timestamp("2016-01-04")             # generator_v8.py: W = date_range('2016-01-04', ..., freq='W-MON')


def creation_offset(lb):
    z = np.load(SIM)
    pt, pa, pq, pdl = z["pt"].astype(np.int64), z["pa"].astype(np.int64), z["pq"], z["pd"]
    k = lb.entity_id.str.slice(3).astype(np.int64).to_numpy()        # POL{index:09d}
    t0 = ((lb.snapshot_date - W0).dt.days // 7).to_numpy(np.int64)
    y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
    if lb.task.iloc[0] == "arrival_week":
        bad = ev & (np.abs(np.maximum(0, pa[k] - t0) - y) > 1e-9)
    else:
        bad = np.abs(np.round(np.where(pq[k] > 0, pdl[k] / np.maximum(pq[k], 1), 1.0), 4) - y) > 1.5e-4
    assert not bad.any(), f"{int(bad.sum())} labels do not match the simulation -- the join is wrong, BLOCKED"
    off = (pt[k] - t0).astype(np.float32)
    assert (off >= 1).all() and (off <= 13).all(), "creation week outside (t0, t0 + 13]"
    return pd.DataFrame({"PRIVILEGED__creation_offset_w": off})


def run(task, seeds, log):
    t_task = TASK[task]
    lb = P7.labels("v8", t_task); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Wd = P7.World("v8")
    X = pd.concat([Wd.channel_features(lb, with_ids=False, flat=True), creation_offset(lb)], axis=1).astype(np.float32)
    for s in seeds:
        name = f"PRIVILEGED__timing_only_s{s}"
        if all(os.path.exists(os.path.join(OUT, f"v8_{t_task}_{name}_{f}.npz")) for f in ("val", "test")):
            continue
        t = time.time()
        if task == "arrival":
            y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
            m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
            P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log,
                    dict(best_iter=int(m.best_iteration_ or 400), seed=s))
        else:
            m = P7.lgbm_fit("multi", X, P7.fill_cell(lb.label_value.to_numpy(float)), tr, va, s, K=22)
            P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log,
                    dict(best_iter=int(m.best_iteration_ or 400), seed=s))
        print(f"  {task} {name}: {time.time() - t:.0f}s", flush=True)
        json.dump(log, open(LOG, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P7.OUT = OUT
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(task=a.task, **C.stamp()))
    run(a.task, list(P7.SEEDS), log)
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
