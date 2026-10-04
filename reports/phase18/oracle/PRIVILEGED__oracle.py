"""Phase 18 Stage 2 (oracle ceiling) and the Stage 4 hindsight control -- PRIVILEGED. Torch-free (LightGBM process).

HIDDEN GENERATOR STATE IS NEVER A FEATURE. This file reads the generator's latents and writes predictions that exist only
to measure a ceiling. Everything it writes is under reports/phase18/oracle/ with the PRIVILEGED__ prefix, and
ml/tests/test_phase18_isolation.py asserts that no ml/ module imports it or reads its outputs.

The fit is Phase 7's guide-B5 LightGBM, unchanged (same labels, split, rows, flat as-of features, frozen GBM, objective,
early stopping), with privileged columns APPENDED:

  oracle_state   arrival / fill (rows are PO lines created after t0): the line's creation week offset, and at that
                 week the supplier state, destination-plant transit and demand state, regime, true K of that month.
                 capacity (rows are channel x snapshot): true K for each month of the label window, mean supplier
                 state and regime over the window.
  oracle         (the brief's tier; it decides the verdict) oracle_state PLUS the realised load: arrival / fill add
                 the supplier's load at creation (util of the previous month = ordered / true K), its effective capacity,
                 the supplier's ordered quantity in the creation week and month, and the line's own quantity;
                 capacity adds the supplier's realised ordered volume for each month of the window.
  oracle_state_notiming  DIAGNOSTIC (arrival / fill only): oracle_state WITHOUT the creation-week offset. arrival_week
                 is counted from t0, so it contains the wait until the line is raised; this tier shows how much of the
                 ceiling is that timing rather than state at creation.
  hindsight_load the Stage 4 upper-bound control: the realised ordered quantity per supplier and per channel over
                 weeks 1-4 / 5-8 / 9-13 after t0 -- what fwd_load tries to forecast, known exactly. Never a feature.
Never an input: the line's realised delivery, lead time, receipt or expedite -- those are the outcomes.

Before any fit, every label is recomputed from the simulation and asserted equal (arrival = arrival week - t0,
fill = delivered / ordered, capacity = mean monthly ordered / K over the window, capped at 3).

  python reports/phase18/oracle/PRIVILEGED__oracle.py --task arrival [--arms oracle,oracle_state,hindsight_load]
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
LAT = os.path.join(HERE, "PRIVILEGED__latents_v8s1001.npz")
OUT = os.path.join(HERE, "preds")
LOG = os.path.join(HERE, "PRIVILEGED__oracle_fit.json")
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
W0 = pd.Timestamp("2016-01-04")             # generator_v8.py: W = date_range('2016-01-04', ..., freq='W-MON')
ARMS = ("oracle", "oracle_state", "hindsight_load", "oracle_state_notiming")


class Sim:
    def __init__(self):
        z = np.load(SIM); L = np.load(LAT)
        self.pt, self.pch, self.pq, self.pd, self.pa = (z[k].astype(np.int64) for k in ("pt", "pch", "pq", "pd", "pa"))
        self.CS, self.CL, self.MK = z["CS"], z["CL"], z["MONTHKEY"].astype(np.int64)
        self.K, self.util, self.ordered = z["K"], z["util"], z["ordered"]
        self.sup_state, self.regime = z["sup_state"].astype(float), z["regime"]
        self.transit, self.dem = L["PRIVILEGED__transit_state"].astype(float), L["PRIVILEGED__dem_state"].astype(float)
        assert np.array_equal(L["CL"], self.CL) and np.array_equal(L["CS"], self.CS), "re-run latents misaligned"
        self.T = len(self.regime)
        NS = self.K.shape[1]
        # supplier ordered quantity per week, from the line registry
        self.sup_wk = np.zeros((self.T + 30, NS))
        np.add.at(self.sup_wk, (self.pt, self.CS[self.pch]), self.pq.astype(float))
        self.ch_wk = None

    def week(self, dates):
        return ((pd.to_datetime(pd.Series(dates)) - W0).dt.days // 7).to_numpy(np.int64)


def line_features(S, lb, arm):
    k = lb.entity_id.str.slice(3).astype(np.int64).to_numpy()         # POL{index:09d}
    t0 = S.week(lb.snapshot_date)
    pt = S.pt[k]; sup = S.CS[S.pch[k]]; pl = S.CL[S.pch[k]]; m = S.MK[np.minimum(pt, S.T - 1)]
    assert (S.pch[k] == lb.key.map(P7_cidx).to_numpy()).all(), "line -> channel mismatch"
    ptc = np.minimum(pt, S.T - 1)
    F = {"PRIVILEGED__creation_offset_w": (pt - t0).astype(float),
         "PRIVILEGED__sup_state_at_creation": S.sup_state[ptc, sup],
         "PRIVILEGED__transit_at_creation": S.transit[ptc, pl],
         "PRIVILEGED__dem_state_at_creation": S.dem[ptc, pl],
         "PRIVILEGED__regime_at_creation": S.regime[ptc],
         "PRIVILEGED__K_month_of_creation": S.K[m, sup]}
    if arm == "oracle_state_notiming":
        F.pop("PRIVILEGED__creation_offset_w")
    if arm == "oracle":
        lp = np.where(m > 0, S.util[np.maximum(m - 1, 0), sup], 0.0)
        F.update({"PRIVILEGED__load_prev_at_creation": lp,
                  "PRIVILEGED__effective_K": S.K[m, sup] * (1 - 0.18 * np.clip(lp - 1.0, 0, 1.5)),
                  "PRIVILEGED__sup_ordered_creation_week": S.sup_wk[pt, sup],
                  "PRIVILEGED__sup_ordered_creation_month": S.ordered[m, sup],
                  "PRIVILEGED__line_qty": S.pq[k].astype(float)})
    return pd.DataFrame(F).astype(np.float32), k, t0


def capacity_features(S, lb, arm):
    ci = lb.entity_id.map(P7_cidx).to_numpy(np.int64)
    sup = S.CS[ci]; t0 = S.week(lb.snapshot_date); t1 = np.minimum(S.T - 1, t0 + 90 // 7)
    m0, m1 = S.MK[t0], S.MK[t1]
    F = {}
    for j in range(4):
        mj = m0 + j; ok = mj <= m1
        F[f"PRIVILEGED__K_m{j}"] = np.where(ok, S.K[np.minimum(mj, len(S.K) - 1), sup], np.nan)
        if arm == "oracle":
            F[f"PRIVILEGED__ordered_m{j}"] = np.where(ok, S.ordered[np.minimum(mj, len(S.K) - 1), sup], np.nan)
    cs = np.cumsum(np.vstack([np.zeros((1, S.sup_state.shape[1])), S.sup_state]), 0)
    F["PRIVILEGED__sup_state_window_mean"] = (cs[t1 + 1, sup] - cs[t0, sup]) / (t1 - t0 + 1)
    cr = np.r_[0, np.cumsum(S.regime)]
    F["PRIVILEGED__regime_window_mean"] = (cr[t1 + 1] - cr[t0]) / (t1 - t0 + 1)
    # label recomputed from the simulation: mean over months m0..m1 of util, capped at 3
    lab = np.array([S.util[a:b + 1, s].mean() for a, b, s in zip(m0, m1, sup)])
    return pd.DataFrame(F).astype(np.float32), np.minimum(lab, 3.0)


def hindsight_features(S, lb):
    ci = lb.key.map(P7_cidx).to_numpy(np.int64); sup = S.CS[ci]; t0 = S.week(lb.snapshot_date)
    if S.ch_wk is None:
        S.ch_wk = {}
        key = S.pch * (S.T + 30) + S.pt
        S.ch_wk_flat = np.bincount(key, weights=S.pq.astype(float), minlength=len(S.CS) * (S.T + 30))
    cw = np.vstack([np.zeros((1, S.sup_wk.shape[1])), np.cumsum(S.sup_wk, 0)])
    ch = S.ch_wk_flat.reshape(len(S.CS), S.T + 30)
    chc = np.hstack([np.zeros((len(S.CS), 1)), np.cumsum(ch, 1)])
    F = {}
    for a, b, tag in ((1, 4, "w1_4"), (5, 8, "w5_8"), (9, 13, "w9_13")):
        F[f"PRIVILEGED__sup_ordered_{tag}"] = cw[t0 + b + 1, sup] - cw[t0 + a, sup]
        F[f"PRIVILEGED__ch_ordered_{tag}"] = chc[ci, t0 + b + 1] - chc[ci, t0 + a]
    return pd.DataFrame(F).astype(np.float32)


def check_labels(S, task, lb, k=None, caplab=None):
    y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
    t0 = S.week(lb.snapshot_date)
    if task == "arrival":
        exp = np.maximum(0, S.pa[k] - t0).astype(float)
        bad = ev & (np.abs(exp - y) > 1e-9)
    elif task == "fill":
        exp = np.where(S.pq[k] > 0, S.pd[k] / np.maximum(S.pq[k], 1), 1.0)
        bad = np.abs(np.round(exp, 4) - y) > 1.5e-4
    else:
        bad = np.abs(np.round(caplab, 4) - y) > 1.5e-4
    n_bad = int(bad.sum())
    assert n_bad == 0, f"{task}: {n_bad} labels do not equal their value recomputed from the simulation -- BLOCKED"
    return dict(rows=int(len(y)), recomputed_equal=True)


def run(task, arms, seeds, log):
    S = Sim()
    t_task = TASK[task]
    lb = P7.labels("v8", t_task); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Wd = P7.World("v8")
    global P7_cidx
    P7_cidx = pd.Series(Wd.cidx)
    Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
    if task == "capacity":
        _, lab = capacity_features(S, lb, "oracle_state")
        log[f"labels|{task}"] = check_labels(S, task, lb, caplab=lab)
    else:
        _, k, _ = line_features(S, lb, "oracle_state")
        log[f"labels|{task}"] = check_labels(S, task, lb, k=k)
    print(f"  {task}: labels recomputed from the simulation -- all equal", flush=True)
    for arm in arms:
        if arm == "hindsight_load":
            F = hindsight_features(S, lb)
        elif task == "capacity":
            F, _ = capacity_features(S, lb, arm)
        else:
            F, _, _ = line_features(S, lb, arm)
        X = pd.concat([Xflat, F], axis=1).astype(np.float32)
        for s in seeds:
            name = f"PRIVILEGED__{arm}_s{s}"
            if all(os.path.exists(os.path.join(OUT, f"v8_{t_task}_{name}_{f}.npz")) for f in ("val", "test")):
                continue
            t = time.time()
            if task == "arrival":
                y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
                m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), seed=s, arm=arm, cols=list(F.columns)))
            elif task == "fill":
                c22 = P7.fill_cell(lb.label_value.to_numpy(float))
                m = P7.lgbm_fit("multi", X, c22, tr, va, s, K=22)
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), seed=s, arm=arm, cols=list(F.columns)))
            else:
                y = lb.label_value.to_numpy(float)
                ms = [P7.lgbm_fit("quantile", X, y, tr, va, s, alpha=al) for al in P7.QS]
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (np.stack([mm.predict(X.iloc[o]) for mm in ms], 1), None),
                        log, dict(best_iters=[int(mm.best_iteration_ or 400) for mm in ms], seed=s, arm=arm, cols=list(F.columns)))
            print(f"  {task} {name}: {time.time() - t:.0f}s", flush=True)
            json.dump(log, open(LOG, "w"), indent=1)


P7_cidx = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(map(str, P7.SEEDS)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P7.OUT = OUT
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(task=a.task, arms=a.arms, **C.stamp()))
    run(a.task, a.arms.split(","), [int(x) for x in a.seeds.split(",")], log)
    json.dump(log, open(LOG, "w"), indent=1)
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
