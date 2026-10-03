"""Phase 20 Stage 3e -- PRIVILEGED hindsight-load arm on the SECOND world (v8w1002). Torch-free. Never a feature.

Phase 18's hindsight_load arm, rebuilt for world 2 so its recovery shares use that world's own upper bound (bands are never
borrowed across worlds): BASE (world 2's own LightGBM-flat fit) + the REALISED ordered quantity per supplier and per channel
in weeks 1-4 / 5-8 / 9-13 after t0, from world 2's `_sim.npz` PO-line registry (creation week, channel, quantity).
Same columns and construction as reports/phase18/oracle/PRIVILEGED__oracle.py::hindsight_features.

  python reports/part2/phase20/PRIVILEGED__hindsight_w2.py --task arrival
"""
from __future__ import annotations
import lightgbm as lgb
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ML = os.path.join(REPO, "ml")
sys.path[:0] = [os.path.join(ML, "baselines"), ML, os.path.join(ML, "train"), os.path.join(ML, "data"), os.path.join(ML, "eval")]
import numpy as np, pandas as pd
import phase20_world as PW
import phase7_fit as P7
import folds as FO
import phase12_common as C
assert "torch" not in sys.modules

OUT = os.path.join(HERE, "preds")
LOG = os.path.join(HERE, "PRIVILEGED__hindsight_w2_fit.json")
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
W0 = pd.Timestamp("2016-01-04")


def hindsight(lb, cidx):
    z = np.load(os.path.join(PW.PATH, "_sim.npz"))
    pt, pch, pq, CS = z["pt"].astype(np.int64), z["pch"].astype(np.int64), z["pq"].astype(float), z["CS"]
    T = len(z["regime"]) + 30
    sup_wk = np.zeros((T, CS.max() + 1)); np.add.at(sup_wk, (pt, CS[pch]), pq)
    ch_wk = np.bincount(pch * T + pt, weights=pq, minlength=len(CS) * T).reshape(len(CS), T)
    cw = np.vstack([np.zeros((1, sup_wk.shape[1])), np.cumsum(sup_wk, 0)]); chc = np.hstack([np.zeros((len(CS), 1)), np.cumsum(ch_wk, 1)])
    ci = lb.key.map(cidx).to_numpy(np.int64); sup = CS[ci]; t0 = ((lb.snapshot_date - W0).dt.days // 7).to_numpy(np.int64)
    F = {}
    for a, b, tag in ((1, 4, "w1_4"), (5, 8, "w5_8"), (9, 13, "w9_13")):
        F[f"PRIVILEGED__sup_ordered_{tag}"] = cw[t0 + b + 1, sup] - cw[t0 + a, sup]
        F[f"PRIVILEGED__ch_ordered_{tag}"] = chc[ci, t0 + b + 1] - chc[ci, t0 + a]
    return pd.DataFrame(F).astype(np.float32)


def main(task):
    PW.register()
    os.makedirs(OUT, exist_ok=True); P7.OUT = OUT
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(task=task, **C.stamp()))
    t = TASK[task]; w = PW.WORLD
    lb = P7.labels(w, t); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Wd = P7.World(w)
    X = pd.concat([Wd.channel_features(lb, with_ids=False, flat=True), hindsight(lb, Wd.cidx)], axis=1).astype(np.float32)
    for s in P7.SEEDS:
        name = f"PRIVILEGED__hindsight_load_s{s}"
        if all(os.path.exists(os.path.join(OUT, f"{w}_{t}_{name}_{f}.npz")) for f in ("val", "test")):
            continue
        if task == "arrival":
            y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
            m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
            P7.emit(w, t, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log, dict(seed=s))
        elif task == "fill":
            m = P7.lgbm_fit("multi", X, P7.fill_cell(lb.label_value.to_numpy(float)), tr, va, s, K=22)
            P7.emit(w, t, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log, dict(seed=s))
        else:
            y = lb.label_value.to_numpy(float)
            ms = [P7.lgbm_fit("quantile", X, y, tr, va, s, alpha=a) for a in P7.QS]
            P7.emit(w, t, name, lb, va, te, lambda o: (np.stack([mm.predict(X.iloc[o]) for mm in ms], 1), None), log, dict(seed=s))
        json.dump(log, open(LOG, "w"), indent=1)
        print(f"  {w} {task} {name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK))
    main(ap.parse_args().task)
    assert "torch" not in sys.modules
