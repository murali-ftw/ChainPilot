"""Phase 18 Stages 4-5 -- the LightGBM proxy gate. ONE torch-free process (LightGBM and torch cannot share a process).

Every arm is Phase 7's guide-B5 fit, unchanged: the same labels, split, row order, flat as-of features
(`World.channel_features(with_ids=False, flat=True)`), frozen GBM config, objective and early stopping
(`phase7_fit.lgbm_fit`). An arm only APPENDS columns:

  base           nothing appended. Run for seed 7 only, as a reproduction check: its predictions must equal the stored
                 `phase7_preds/v8_{task}_{b5flat arm}_s7_{fold}.npz` before any family arm is trusted. The 5-seed base
                 band is then the stored files themselves (the same computation).
  fwd_load       + ml/data/fwd_load.py COLS (Stage 4)
  fwd_load_shuf  + the same, shuffled ACROSS SUPPLIERS WITHIN EACH SNAPSHOT (supplier columns by a supplier
                 permutation, channel columns by a channel permutation), one permutation per snapshot per seed
  fwd_load_netmean  DIAGNOSTIC, no verdict (deviation 163): + the SNAPSHOT MEAN over channels of each fwd_load column,
                 the same vector for every row of a snapshot. It is exactly what the within-snapshot shuffle cannot
                 destroy, so it measures how much of fwd_load_shuf's gain is the network-wide forward requirement.
  pulse          + ml/data/pulse.py COLS (Stage 5), the same vector for every row of a snapshot
  pulse_shuf     + the same, shuffled ACROSS SNAPSHOTS (one permutation of the snapshot -> vector map per seed)

Tasks: arrival (L2 on observed rows, b5flat_reg), fill (22-class, b5flat22), capacity (quantile 0.1/0.5/0.9, b5flat_q).
Output: ml/artifacts/phase18/preds/v8_{task}_p18_{arm}_s{seed}_{fold}.npz, the phase7_preds format (P, Y, EV, AUX,
entity). A file that exists is never overwritten (resumable; identity asserted by name + the log entry).

  python ml/baselines/phase18_proxy.py --task fill --arms base            # reproduction check first
  python ml/baselines/phase18_proxy.py --task fill --arms fwd_load,fwd_load_shuf,pulse,pulse_shuf
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST -- before anything that could pull in torch
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import phase7_fit as P7
import folds as FO
import fwd_load as FL, pulse as PU
from config import ARTIFACTS
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
import phase12_common as C
assert "torch" not in sys.modules, "torch must not be imported in the LightGBM process"

OUT = os.path.join(ARTIFACTS, "phase18", "preds")
LOG = os.path.join(ARTIFACTS, "phase18", "proxy_fit.json")
STORED = {"arrival": "b5flat_reg", "fill": "b5flat22", "capacity": "b5flat_q"}
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
SEEDS = P7.SEEDS
ARMS = ("base", "fwd_load", "fwd_load_shuf", "pulse", "pulse_shuf", "fwd_load_netmean")


def family(arm, lb, seed):
    """-> DataFrame of appended columns aligned to lb's rows (or None for base)."""
    if arm == "base":
        return None
    snap = lb.snapshot_date.dt.strftime("%Y-%m-%d").to_numpy()
    if arm.startswith("fwd_load"):
        X, snaps, chans, cols, sups = FL.load()
        si = pd.Series(range(len(snaps)), index=snaps).reindex(snap).to_numpy()
        ci = pd.Series(range(len(chans)), index=chans).reindex(lb.key.to_numpy()).to_numpy()
        assert not np.isnan(si.astype(float)).any() and not np.isnan(ci.astype(float)).any(), "fwd_load rows missing"
        si, ci = si.astype(np.int64), ci.astype(np.int64)
        if arm == "fwd_load_shuf":
            rng = np.random.default_rng(10_000 + seed)
            sup_cols = [j for j, c in enumerate(cols) if c.startswith("fwd_sup_")]
            ch_cols = [j for j, c in enumerate(cols) if j not in sup_cols]
            usup, sup_of = np.unique(sups, return_inverse=True)
            first_ch = np.array([np.flatnonzero(sup_of == k)[0] for k in range(len(usup))])
            Xs = X.copy()
            for i in range(len(snaps)):
                ps = rng.permutation(len(usup)); pc = rng.permutation(len(chans))
                Xs[i][:, sup_cols] = X[i][first_ch[ps[sup_of]]][:, sup_cols]
                Xs[i][:, ch_cols] = X[i][pc][:, ch_cols]
            X = Xs
        if arm == "fwd_load_netmean":
            M = np.nanmean(X, axis=1)                                   # [snapshot, col]
            return pd.DataFrame(M[si], columns=[f"{c}_netmean" for c in cols])
        return pd.DataFrame(X[si, ci], columns=cols)
    X, snaps, cols = PU.load()
    si = pd.Series(range(len(snaps)), index=snaps).reindex(snap).to_numpy()
    assert not np.isnan(si.astype(float)).any(), "pulse rows missing"
    si = si.astype(np.int64)
    if arm == "pulse_shuf":
        perm = np.random.default_rng(20_000 + seed).permutation(len(snaps))
        X = X[perm]
    return pd.DataFrame(X[si], columns=cols)


def fit_task(Wd, task, arms, seeds, log):
    t_task = TASK[task]
    lb = P7.labels("v8", t_task); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
    for arm in arms:
        for s in (seeds if arm != "base" else [7]):
            name = f"p18_{arm}_s{s}"
            if all(os.path.exists(os.path.join(OUT, f"v8_{t_task}_{name}_{f}.npz")) for f in ("val", "test")):
                print(f"  {task} {name}: exists, skipped", flush=True); continue
            log_stamp = log["_stamps"][-1]["code_commit"]
            F = family(arm, lb, s)
            X = Xflat if F is None else pd.concat([Xflat, F], axis=1).astype(np.float32)
            t = time.time()
            if task == "arrival":
                y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
                m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), seed=s, arm=arm, n_features=X.shape[1], commit=log_stamp))
            elif task == "fill":
                c22 = P7.fill_cell(lb.label_value.to_numpy(float))
                m = P7.lgbm_fit("multi", X, c22, tr, va, s, K=22)
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), seed=s, arm=arm, n_features=X.shape[1], commit=log_stamp))
            else:
                y = lb.label_value.to_numpy(float)
                ms = [P7.lgbm_fit("quantile", X, y, tr, va, s, alpha=a) for a in P7.QS]
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (np.stack([mm.predict(X.iloc[o]) for mm in ms], 1), None),
                        log, dict(best_iters=[int(mm.best_iteration_ or 400) for mm in ms], seed=s, arm=arm,
                                  n_features=X.shape[1], commit=log_stamp))
            print(f"  {task} {name}: {time.time() - t:.0f}s", flush=True)
            json.dump(log, open(LOG, "w"), indent=1)
            if arm == "base":
                check_reproduction(t_task, task, log)


def check_reproduction(t_task, task, log):
    res = {}
    for f in ("val", "test"):
        a = np.load(os.path.join(OUT, f"v8_{t_task}_p18_base_s7_{f}.npz"))
        b = np.load(os.path.join(P7.ARTIFACTS, "phase7_preds", f"v8_{t_task}_{STORED[task]}_s7_{f}.npz"))
        assert (a["entity"] == b["entity"]).all() and np.array_equal(a["Y"], b["Y"]), f"{task} {f}: rows differ"
        res[f] = float(np.abs(np.asarray(a["P"], float) - np.asarray(b["P"], float)).max())
    log[f"v8|{t_task}|p18_base_s7|reproduction"] = dict(max_abs_diff=res, stored=f"{STORED[task]}_s7",
                                                        reproduced=bool(max(res.values()) <= 1e-9))
    json.dump(log, open(LOG, "w"), indent=1)
    print(f"  REPRODUCTION {task}: max |diff| vs stored {STORED[task]}_s7 = {res}", flush=True)
    assert max(res.values()) <= 1e-9, f"{task}: base does not reproduce the stored LightGBM-flat predictions -- STOP"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--arms", default=",".join(ARMS[1:5]))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P7.OUT = OUT                                   # phase7_fit.save writes to its module-level OUT
    st = C.require_clean()                         # nothing is produced from uncommitted ml/ code
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(task=a.task, arms=a.arms, seeds=a.seeds, **st))
    Wd = P7.World("v8")
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms
    fit_task(Wd, a.task, arms, [int(x) for x in a.seeds.split(",")], log)
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
