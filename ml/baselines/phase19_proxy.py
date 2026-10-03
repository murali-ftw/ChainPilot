"""Phase 19 Stage 1 -- the LightGBM proxy for Gate v2. ONE torch-free process.

Every arm is Phase 7's guide-B5 fit, unchanged (`phase7_fit.lgbm_fit` / `emit`, frozen GBM, the same labels, rows, flat
features, objective and early stopping), with columns APPENDED. Only the NEW arms are fitted here; BASE + fwd_load is
reused from Phase 18's stored predictions (ml/artifacts/phase18/preds/v8_{task}_p18_fwd_load_s{s}_{fold}.npz).

  base             seed 7 only: must reproduce the stored b5flat arm bit-exactly (else STOP)
  fwd_season       + ml/data/fwd_season.py (15 snapshot-level network means)
  fwd_season_sperm + the same, SNAPSHOT-PERMUTED: each snapshot's vector taken from a different snapshot of the same split
  fwd_load_sperm   + ml/data/fwd_load.py, SNAPSHOT-PERMUTED: each snapshot's whole [channel x 15] block taken from a
                     different snapshot of the same split; channel c receives channel c's row of the donor snapshot
  cadence          + ml/data/cadence.py (5 per-channel cadence columns)
  cadence_shuf     + the same, rows permuted across channels within each snapshot
The derangement (no snapshot keeps its own block) is drawn within train, within val and within test separately,
rng(30000 + seed); the cadence shuffle uses rng(40000 + seed).

Output: ml/artifacts/phase19/preds/v8_{task}_p19_{arm}_s{seed}_{fold}.npz (phase7_preds format). Never overwritten.

  python ml/baselines/phase19_proxy.py --task fill --arms base
  python ml/baselines/phase19_proxy.py --task fill
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST -- before anything that could pull in torch
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import phase7_fit as P7
import folds as FO
import fwd_load as FL, fwd_season as FS, cadence as CD
import phase12_common as C
from config import ARTIFACTS, SPLIT
assert "torch" not in sys.modules, "torch must not be imported in the LightGBM process"

OUT = os.path.join(ARTIFACTS, "phase19", "preds")
LOG = os.path.join(ARTIFACTS, "phase19", "proxy_fit.json")
STORED = {"arrival": "b5flat_reg", "fill": "b5flat22", "capacity": "b5flat_q"}
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
SEEDS = P7.SEEDS
ARMS = ("base", "fwd_season", "fwd_season_sperm", "fwd_load_sperm", "cadence", "cadence_shuf")


def split_of_snapshots(snaps):
    d = pd.to_datetime(pd.Series(snaps))
    te = d > pd.Timestamp(SPLIT["val_end"]); va = (d > pd.Timestamp(SPLIT["train_end"])) & ~te
    return np.where(te, 2, np.where(va, 1, 0))


def derangement_within_split(snaps, seed):
    """-> donor index per snapshot: a permutation with no fixed point inside each split."""
    rng = np.random.default_rng(30_000 + seed)
    grp = split_of_snapshots(snaps); donor = np.arange(len(snaps))
    for g in (0, 1, 2):
        idx = np.flatnonzero(grp == g)
        assert len(idx) >= 2, "a split with one snapshot cannot be deranged"
        while True:
            p = rng.permutation(len(idx))
            if not (p == np.arange(len(idx))).any():
                break
        donor[idx] = idx[p]
    assert (donor != np.arange(len(snaps))).all() and (grp[donor] == grp).all()
    return donor


def rows_index(lb, snaps, chans=None):
    snap = lb.snapshot_date.dt.strftime("%Y-%m-%d").to_numpy()
    si = pd.Series(range(len(snaps)), index=snaps).reindex(snap).to_numpy()
    assert not np.isnan(si.astype(float)).any(), "rows with no snapshot in the feature store"
    if chans is None:
        return si.astype(np.int64), None
    ci = pd.Series(range(len(chans)), index=chans).reindex(lb.key.to_numpy()).to_numpy()
    assert not np.isnan(ci.astype(float)).any(), "rows with no channel in the feature store"
    return si.astype(np.int64), ci.astype(np.int64)


def family(arm, lb, seed):
    if arm == "base":
        return None
    if arm.startswith("fwd_season"):
        X, snaps, cols = FS.load()
        if arm.endswith("_sperm"):
            X = X[derangement_within_split(snaps, seed)]
        si, _ = rows_index(lb, snaps)
        return pd.DataFrame(X[si], columns=cols)
    if arm == "fwd_load_sperm":
        X, snaps, chans, cols, _ = FL.load()
        X = X[derangement_within_split(snaps, seed)]
        si, ci = rows_index(lb, snaps, chans)
        return pd.DataFrame(X[si, ci], columns=cols)
    X, snaps, chans, cols = CD.load()
    if arm == "cadence_shuf":
        rng = np.random.default_rng(40_000 + seed)
        X = np.stack([X[i][rng.permutation(X.shape[1])] for i in range(len(snaps))])
    si, ci = rows_index(lb, snaps, chans)
    return pd.DataFrame(X[si, ci], columns=cols)


def fit_task(Wd, task, arms, seeds, log, stamp):
    t_task = TASK[task]
    lb = P7.labels("v8", t_task); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
    for arm in arms:
        for s in (seeds if arm != "base" else [7]):
            name = f"p19_{arm}_s{s}"
            if all(os.path.exists(os.path.join(OUT, f"v8_{t_task}_{name}_{f}.npz")) for f in ("val", "test")):
                print(f"  {task} {name}: exists, skipped", flush=True); continue
            F = family(arm, lb, s)
            X = Xflat if F is None else pd.concat([Xflat, F], axis=1).astype(np.float32)
            meta = dict(seed=s, arm=arm, n_features=X.shape[1], commit=stamp)
            t = time.time()
            if task == "arrival":
                y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
                m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), **meta))
            elif task == "fill":
                c22 = P7.fill_cell(lb.label_value.to_numpy(float))
                m = P7.lgbm_fit("multi", X, c22, tr, va, s, K=22)
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), **meta))
            else:
                y = lb.label_value.to_numpy(float)
                ms = [P7.lgbm_fit("quantile", X, y, tr, va, s, alpha=a) for a in P7.QS]
                P7.emit("v8", t_task, name, lb, va, te, lambda o: (np.stack([mm.predict(X.iloc[o]) for mm in ms], 1), None),
                        log, dict(best_iters=[int(mm.best_iteration_ or 400) for mm in ms], **meta))
            print(f"  {task} {name}: {time.time() - t:.0f}s", flush=True)
            json.dump(log, open(LOG, "w"), indent=1)
            if arm == "base":
                res = {}
                for f in ("val", "test"):
                    a = np.load(os.path.join(OUT, f"v8_{t_task}_p19_base_s7_{f}.npz"))
                    b = np.load(os.path.join(P7.ARTIFACTS, "phase7_preds", f"v8_{t_task}_{STORED[task]}_s7_{f}.npz"))
                    assert (a["entity"] == b["entity"]).all() and np.array_equal(a["Y"], b["Y"]), f"{task} {f}: rows differ"
                    res[f] = float(np.abs(np.asarray(a["P"], float) - np.asarray(b["P"], float)).max())
                log[f"v8|{t_task}|p19_base_s7|reproduction"] = dict(max_abs_diff=res, reproduced=max(res.values()) == 0.0)
                json.dump(log, open(LOG, "w"), indent=1)
                print(f"  REPRODUCTION {task}: max |diff| vs stored {STORED[task]}_s7 = {res}", flush=True)
                assert max(res.values()) == 0.0, f"{task}: BASE does not reproduce the stored LightGBM-flat arm bit-exactly -- STOP"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--arms", default=",".join(ARMS[1:]))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    P7.OUT = OUT
    st = C.require_clean()
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(task=a.task, arms=a.arms, seeds=a.seeds, **st))
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms
    fit_task(P7.World("v8"), a.task, arms, [int(x) for x in a.seeds.split(",")], log, st["code_commit"])
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
