"""Phase 20 Stages 2-3 -- the LightGBM proxy on ANY registered world (v8 = seed 1001, v8w1002 = the second world).
ONE torch-free process. Every arm is Phase 7's guide-B5 fit unchanged (frozen GBM, same objective, early stopping on the
world's own validation fold), with columns appended. A world's BASE is fitted on that world: nothing is borrowed.

  base            v8: seed 7 only (must reproduce the stored b5flat arm bit-exactly); v8w1002: all 5 seeds (its own BASE)
  fwd_season, fwd_load, cadence                     the Phase 18-19 families, built by the unchanged modules for that world
  fwd_season_sperm, fwd_load_sperm, cadence_shuf    Phase 19's controls (phase19_proxy.derangement_within_split, rng 30000+seed;
                                                    cadence rows permuted within snapshot, rng 40000+seed)
  lgbm_rf         the families the Phase 19 neural arm carried (arrival fwd_load+season+cadence; fill season+cadence;
                  capacity fwd_load): the LightGBM half of the Stage 2 blend recipe
  stock, stock_sperm   Stage 4c: BASE + ml/data/stock_asof.py (observed ledger position vs an as-of reorder estimate), and the
                  same SNAPSHOT-PERMUTED within split (rng 60000+seed)
  fwd_load_pred, fwd_load_pred_sperm   Stage 3: BASE + fwd_load + fwd_pred (the load forecaster's predictions, seed-matched),
                  and the same with fwd_pred SNAPSHOT-PERMUTED within split (rng 50000+seed)

Output: ml/artifacts/phase20/preds/{world}_{task}_p20_{arm}_s{seed}_{fold}.npz (phase7_preds format). Never overwritten.

  python ml/baselines/phase20_proxy.py --world v8w1002 --task fill --arms base,fwd_season,...
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST -- before anything that could pull in torch
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import phase20_world as PW
import phase7_fit as P7
import folds as FO
import fwd_load as FL, fwd_season as FS, cadence as CD, fwd_pred as FP, stock_asof as SK
import phase19_proxy as P19
import phase12_common as C
from config import ARTIFACTS
assert "torch" not in sys.modules, "torch must not be imported in the LightGBM process"

OUT = os.path.join(ARTIFACTS, "phase20", "preds")
LOG = os.path.join(ARTIFACTS, "phase20", "proxy_fit.json")
TASK = P19.TASK
ARMS = ("base", "fwd_season", "fwd_load", "cadence", "fwd_season_sperm", "fwd_load_sperm", "cadence_shuf", "lgbm_rf",
        "fwd_load_pred", "fwd_load_pred_sperm", "stock", "stock_sperm")
RF = {"arrival": ("fwd_load", "fwd_season", "cadence"), "fill": ("fwd_season", "cadence"), "capacity": ("fwd_load",)}


def block(fam, world, lb, seed, perm=None):
    """One family's columns for lb's rows. perm: None | 'sperm' (snapshot derangement within split) | 'shuf' (cadence)."""
    if fam == "fwd_season":
        X, snaps, cols = FS.load(world)
        if perm == "sperm":
            X = X[P19.derangement_within_split(snaps, seed)]
        si, _ = P19.rows_index(lb, snaps)
        return pd.DataFrame(X[si], columns=cols)
    if fam == "fwd_load":
        X, snaps, chans, cols, _ = FL.load(world)
        if perm == "sperm":
            X = X[P19.derangement_within_split(snaps, seed)]
    elif fam == "cadence":
        X, snaps, chans, cols = CD.load(world)
        if perm == "shuf":
            rng = np.random.default_rng(40_000 + seed)
            X = np.stack([X[i][rng.permutation(X.shape[1])] for i in range(len(snaps))])
    elif fam == "stock":
        X, snaps, chans, cols = SK.load(world)
        if perm == "sperm":
            X = X[P19.derangement_within_split(snaps, 30_000 + seed)]       # rng(60000 + seed)
    else:                                                     # fwd_pred, seed-matched forecaster
        X, snaps, chans, cols = FP.load(world, seed)
        if perm == "sperm":
            rng_seed = 50_000 + seed - 30_000                 # derangement_within_split adds 30000
            X = X[P19.derangement_within_split(snaps, rng_seed)]
    si, ci = P19.rows_index(lb, snaps, chans)
    return pd.DataFrame(X[si, ci], columns=cols)


def family(arm, task, world, lb, seed):
    if arm == "base":
        return None
    if arm in ("fwd_season", "fwd_load", "cadence"):
        return block(arm, world, lb, seed)
    if arm == "fwd_season_sperm":
        return block("fwd_season", world, lb, seed, "sperm")
    if arm == "fwd_load_sperm":
        return block("fwd_load", world, lb, seed, "sperm")
    if arm == "cadence_shuf":
        return block("cadence", world, lb, seed, "shuf")
    if arm == "lgbm_rf":
        return pd.concat([block(f, world, lb, seed) for f in RF[task]], axis=1)
    if arm == "stock":
        return block("stock", world, lb, seed)
    if arm == "stock_sperm":
        return block("stock", world, lb, seed, "sperm")
    if arm == "fwd_load_pred":
        return pd.concat([block("fwd_load", world, lb, seed), block("fwd_pred", world, lb, seed)], axis=1)
    if arm == "fwd_load_pred_sperm":
        return pd.concat([block("fwd_load", world, lb, seed), block("fwd_pred", world, lb, seed, "sperm")], axis=1)
    raise ValueError(arm)


def fit(world, task, arms, seeds, log, stamp):
    t_task = TASK[task]
    lb = P7.labels(world, t_task); tr, va, te = FO.fixed_split(lb.snapshot_date)
    Wd = P7.World(world)
    Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
    for arm in arms:
        for s in ([7] if (arm == "base" and world == "v8") else seeds):
            name = f"p20_{arm}_s{s}"
            if all(os.path.exists(os.path.join(OUT, f"{world}_{t_task}_{name}_{f}.npz")) for f in ("val", "test")):
                print(f"  {world} {task} {name}: exists, skipped", flush=True); continue
            F = family(arm, task, world, lb, s)
            X = Xflat if F is None else pd.concat([Xflat, F], axis=1).astype(np.float32)
            meta = dict(seed=s, arm=arm, world=world, n_features=X.shape[1], commit=stamp)
            t = time.time()
            if task == "arrival":
                y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
                m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
                P7.emit(world, t_task, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log, dict(best_iter=int(m.best_iteration_ or 400), **meta))
            elif task == "fill":
                m = P7.lgbm_fit("multi", X, P7.fill_cell(lb.label_value.to_numpy(float)), tr, va, s, K=22)
                P7.emit(world, t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log, dict(best_iter=int(m.best_iteration_ or 400), **meta))
            else:
                y = lb.label_value.to_numpy(float)
                ms = [P7.lgbm_fit("quantile", X, y, tr, va, s, alpha=a) for a in P7.QS]
                P7.emit(world, t_task, name, lb, va, te, lambda o: (np.stack([mm.predict(X.iloc[o]) for mm in ms], 1), None), log,
                        dict(best_iters=[int(mm.best_iteration_ or 400) for mm in ms], **meta))
            print(f"  {world} {task} {name}: {time.time() - t:.0f}s", flush=True)
            json.dump(log, open(LOG, "w"), indent=1)
            if arm == "base" and world == "v8":
                for f in ("val", "test"):
                    a = np.load(os.path.join(OUT, f"v8_{t_task}_{name}_{f}.npz"))
                    b = np.load(os.path.join(P7.ARTIFACTS, "phase7_preds", f"v8_{t_task}_{P19.STORED[task]}_s7_{f}.npz"))
                    d = float(np.abs(np.asarray(a["P"], float) - np.asarray(b["P"], float)).max())
                    assert d == 0.0 and (a["entity"] == b["entity"]).all(), f"{task} {f}: BASE does not reproduce bit-exactly ({d}) -- STOP"
                print(f"  REPRODUCTION v8 {task}: bit-exact", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=["v8", PW.WORLD])
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, P7.SEEDS)))
    a = ap.parse_args()
    if a.world == PW.WORLD:
        PW.register()
    os.makedirs(OUT, exist_ok=True)
    P7.OUT = OUT
    st = C.require_clean()
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(world=a.world, task=a.task, arms=a.arms, **st))
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms
    fit(a.world, a.task, arms, [int(x) for x in a.seeds.split(",")], log, st["code_commit"])
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
