"""Phase 20 Stage 3 -- the load forecaster. ONE torch-free process. Frozen GBM config (phase7_fit.GBM / lgbm_fit, L2).

Rows: every (snapshot, channel). Targets (ml/data/fwd_pred.targets): log1p of the channel's realised ordered quantity in
weeks 1-4 / 5-8 / 9-13 after t0 -- one regressor per window. Inputs, all as-of at t0: fwd_load (15), fwd_season (15),
cadence (5), ordering history (4), and the flat static channel features of Phase 7's B5 arm.

Leakage control (the brief, Stage 3b):
  * TRAINING TARGETS only from train snapshots whose 13-week window ends by the train end (t0 + 91 d <= 2023-12-31).
  * Out-of-fold predictions for EVERY train snapshot: K = 5 contiguous blocks of train snapshots; the model for block k is
    trained on eligible snapshots at least 13 weeks away from the block on both sides (purge), and predicts block k.
  * Validation and test predictions: one model on all eligible train rows.
  * Early stopping on validation snapshots whose window ends by the validation end (never on a held-out train block, never test).
Supplier predictions = the sum of the supplier's channel predictions (pre-registered def. 6). Forecaster seed = proxy seed.

Accuracy on TEST (before any downstream use): MAE on quantity and R^2 on quantity and on log1p, beside the naive forecast
= fwd_load's channel requirement for the same window.

  python ml/baselines/phase20_forecaster.py --world v8        # -> ml/artifacts/phase20/fwd_pred_{world}_s{seed}.npz + accuracy json
"""
from __future__ import annotations
import lightgbm as lgb
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import phase20_world as PW
import phase7_fit as P7
import fwd_load as FL, fwd_season as FS, cadence as CD, fwd_pred as FP
import phase12_common as C
from config import SPLIT
assert "torch" not in sys.modules

K = 5
PURGE = pd.Timedelta(weeks=13)
HOR = pd.Timedelta(days=91)


def design(world):
    Xl, snaps, chans, lcols, sups = FL.load(world)
    Xs, s2, scols = FS.load(world); Xc, s3, ch3, ccols = CD.load(world)
    H, s4, _ = FP.history(world); Y, s5 = FP.targets(world)
    assert [str(x) for x in snaps] == [str(x) for x in s2] == [str(x) for x in s3] == s4 == s5, "snapshot lists differ"
    assert list(chans) == list(ch3)
    Wd = P7.World(world)
    G = Wd.g.set_index("channel_id").loc[list(chans)][Wd.STATIC + Wd.FLAT].to_numpy(np.float32)
    NS, NCH = len(snaps), len(chans)
    F = np.concatenate([Xl, np.repeat(Xs[:, None, :], NCH, 1), Xc, H, np.repeat(G[None], NS, 0)], 2)
    cols = list(lcols) + list(scols) + list(ccols) + FP.HIST_COLS + Wd.STATIC + Wd.FLAT
    return F.astype(np.float32), Y, [pd.Timestamp(str(s)) for s in snaps], list(chans), np.asarray(sups), cols


def run(world, seeds):
    t_all = time.time()
    F, Y, snaps, chans, sups, cols = design(world)
    NS, NCH, NF = F.shape
    snaps = pd.DatetimeIndex(snaps)
    tr_s = snaps <= pd.Timestamp(SPLIT["train_end"]); va_s = (snaps > pd.Timestamp(SPLIT["train_end"])) & (snaps <= pd.Timestamp(SPLIT["val_end"]))
    elig = tr_s & (snaps + HOR <= pd.Timestamp(SPLIT["train_end"]))
    va_es = va_s & (snaps + HOR <= pd.Timestamp(SPLIT["val_end"]))
    tr_idx = np.flatnonzero(tr_s); blocks = np.array_split(tr_idx, K)
    Xall = pd.DataFrame(F.reshape(NS * NCH, NF), columns=cols)
    Yall = np.log1p(Y.reshape(NS * NCH, 3).astype(float))
    row_snap = np.repeat(np.arange(NS), NCH)
    rows_of = lambda mask_s: np.isin(row_snap, np.flatnonzero(mask_s))
    sup_u, sup_of = np.unique(sups, return_inverse=True)
    acc = dict(world=world, eligible_train_snapshots=int(elig.sum()), train_snapshots=int(tr_s.sum()),
               excluded_train_snapshots_window_past_train_end=int((tr_s & ~elig).sum()),
               early_stopping_val_snapshots=int(va_es.sum()), blocks=[[str(snaps[b[0]].date()), str(snaps[b[-1]].date())] for b in blocks],
               per_seed={})
    for s in seeds:
        out = FP.path(world, s)
        if os.path.exists(out):
            print(f"  {world} s{s}: exists, skipped", flush=True); continue
        t = time.time()
        P = np.full((NS * NCH, 3), np.nan)
        fold_log = []
        for j in range(3):
            y = Yall[:, j]
            for k, b in enumerate(blocks):
                lo, hi = snaps[b[0]] - PURGE, snaps[b[-1]] + PURGE
                train_s = elig & ~((snaps >= lo) & (snaps <= hi))
                assert not (train_s & np.isin(np.arange(NS), b)).any()
                m = P7.lgbm_fit("l2", Xall, y, None, None, s, rows_tr=rows_of(train_s), rows_va=rows_of(va_es))
                held = rows_of(np.isin(np.arange(NS), b))
                P[held, j] = m.predict(Xall[held])
                fold_log.append(dict(window=FP.WTAG[j], block=k, train_snapshots=int(train_s.sum()), best_iter=int(m.best_iteration_ or 400)))
            m = P7.lgbm_fit("l2", Xall, y, None, None, s, rows_tr=rows_of(elig), rows_va=rows_of(va_es))
            vt = rows_of(~tr_s)
            P[vt, j] = m.predict(Xall[vt])
            fold_log.append(dict(window=FP.WTAG[j], block="full", train_snapshots=int(elig.sum()), best_iter=int(m.best_iteration_ or 400)))
        assert np.isfinite(P).all(), "a row has no prediction"
        Q = np.clip(np.expm1(P), 0, None).reshape(NS, NCH, 3)
        Ssum = np.zeros((NS, len(sup_u), 3))
        for i in range(NS):
            for j in range(3):
                Ssum[i, :, j] = np.bincount(sup_of, weights=Q[i, :, j], minlength=len(sup_u))
        X6 = np.concatenate([Q, Ssum[:, sup_of, :]], 2).astype(np.float32)
        np.savez_compressed(out, X=X6, snapshots=np.array([str(x.date()) for x in snaps]), channels=np.array(chans), cols=np.array(FP.COLS))
        # ---- the forecaster's own accuracy on TEST, beside the naive plan requirement
        te_s = snaps > pd.Timestamp(SPLIT["val_end"]); te_rows = rows_of(te_s)
        Xl, _, _, lcols, _ = FL.load(world)
        naive = Xl.reshape(NS * NCH, -1)[:, [lcols.index(f"fwd_ch_req_{w}") for w in FL.WTAG]]
        r = {}
        for j, w in enumerate(FP.WTAG):
            yt = Y.reshape(NS * NCH, 3)[te_rows, j].astype(float); pt = Q.reshape(NS * NCH, 3)[te_rows, j]
            nv = naive[te_rows, j].astype(float); ok = np.isfinite(nv)
            r2 = lambda a, b: float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
            r[w] = dict(n=int(te_rows.sum()), mae_forecaster=float(np.mean(np.abs(pt - yt))), mae_naive_plan=float(np.mean(np.abs(nv[ok] - yt[ok]))),
                        r2_forecaster=r2(yt, pt), r2_naive_plan=r2(yt[ok], nv[ok]), r2_log_forecaster=r2(np.log1p(yt), np.log1p(pt)),
                        r2_log_naive_plan=r2(np.log1p(yt[ok]), np.log1p(np.clip(nv[ok], 0, None))), naive_rows_finite=int(ok.sum()),
                        # supplier level, test
                        )
        ys = np.zeros((NS, len(sup_u), 3)); [ys.__setitem__((i, slice(None), j), np.bincount(sup_of, weights=Y[i, :, j], minlength=len(sup_u))) for i in range(NS) for j in range(3)]
        for j, w in enumerate(FP.WTAG):
            a = ys[te_s][:, :, j].ravel(); b = Ssum[te_s][:, :, j].ravel()
            r[w]["supplier_mae"] = float(np.mean(np.abs(a - b))); r[w]["supplier_r2"] = float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))
        acc["per_seed"][f"s{s}"] = dict(test=r, folds=fold_log, seconds=round(time.time() - t, 1))
        print(f"  {world} s{s}: {time.time() - t:.0f}s  " + "  ".join(f"{w}: R2 {r[w]['r2_forecaster']:.3f} (plan {r[w]['r2_naive_plan']:.3f})" for w in FP.WTAG), flush=True)
        path = os.path.join(FP.OUT_DIR, f"forecaster_{world}.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update({k: v for k, v in acc.items() if k != "per_seed"}); old.setdefault("per_seed", {}).update(acc["per_seed"])
        old["stamp"] = C.stamp(); json.dump(old, open(path, "w"), indent=1)
    print(f"done in {time.time() - t_all:.0f}s; torch never imported", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=["v8", PW.WORLD])
    ap.add_argument("--seeds", default=",".join(map(str, P7.SEEDS)))
    a = ap.parse_args()
    if a.world == PW.WORLD:
        PW.register()
    C.require_clean()
    os.makedirs(FP.OUT_DIR, exist_ok=True)
    run(a.world, [int(x) for x in a.seeds.split(",")])
    assert "torch" not in sys.modules
