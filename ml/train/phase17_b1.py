"""Phase 17 B1 -- predict-the-rescue: P(a transfer-in is recorded at this part-plant in this week).

ROWS are Phase 14/15's UC5 universe: every part-plant of part_plant.csv x the 13 weeks after each snapshot (store
week_start = t0 + 7(w + 1), Monday-aligned), keeping only part-plant-weeks present in inventory_position_weekly -- the
`ok` filter phase15_sim.py applied. The val (2024) and test (2025) rows are REBUILT here and GATED against the stored
ml/artifacts/phase15_sim/{val,test}_s7.npz: pp, week and acted must agree element for element, or nothing is written.
Train rows are the same construction on the fixed split's training snapshots (fit window, <= 2023-12-31).

LABEL  y = a transfer_in recorded that week (inventory_transactions, event_ts week) = phase14_sim.store_labels' `acted`.
FEATURES -- none engineered:
  B1a  phase7_fit.World.part_plant_features (the b5flat_bin set) as of t0, plus the week offset w. w is the row key:
       without it the 13 rows of one part-plant-snapshot are indistinguishable.
  B1b  the shipped shortage head (mp h1, lr 1.25e-4, shipped.json) over the channel panel, part-plant mean-pooled exactly
       as phase5_heads.forward does, with the one-hot week offset through the existing per-row input (n_row_feats=13).
       Part-plants with no channel get a zero embedding: one extra pooled row with no members.
Batch convention unchanged: one optimiser step per snapshot over all of that snapshot's rows. LR unchanged.

inventory_position_weekly: read for the row filter (which part-plant-weeks exist) and nothing else. Never a feature.

  python ml/train/phase17_b1.py rows
  python ml/train/phase17_b1.py lgbm               (CPU, 5 seeds)
  python ml/train/phase17_b1.py neural --seed 7    (GPU, one cell)
  python ml/train/phase17_b1.py score
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval")]
import phase12_common as C          # puts ml/, train/, eval/, data/, models/, sim/, opt/ on sys.path; numpy only

D = os.path.join(C.REPO, "db", "gen_v8", "seed_1001")
OUT = os.path.join(C.ART, "phase17")
ROWS = os.path.join(OUT, "b1_rows.npz")
SIMD = os.path.join(C.ART, "phase15_sim")
W = 13
FOLDS = {"train": ("2019-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")}
FOLD_ID = {"train": 0, "val": 1, "test": 2}
SEEDS = C.V8_SEEDS
CONCURRENCY = 1                      # every Phase 17 cell runs alone on the GPU (4 GB card)


def snapshots(lo, hi):
    import pandas as pd
    s = pd.read_csv(f"{D}/snapshots.csv", usecols=["as_of_ts"]).as_of_ts
    return sorted(pd.Timestamp(x) for x in s if pd.Timestamp(lo) <= pd.Timestamp(x) <= pd.Timestamp(hi))


# ================================================================== rows (torch-free)
def build_rows():
    import numpy as np, pandas as pd
    import montecarlo as MC
    st = C.require_clean()
    pp, _, cov = MC.part_plant_universe("v8")
    sub = pp.reset_index(drop=True)                     # the order phase15_sim.py used for its pp index
    parts, plant = sub.part_id.to_numpy(), sub.plant_id.to_numpy()
    P = len(sub)
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts"])
    tx = tx[tx.txn_type.isin(["transfer_in", "transfer_out"])]
    tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
    g = tx.groupby(["part_id", "plant_id", "week", "txn_type"]).qty.sum().unstack(fill_value=0)
    acted = (g["transfer_in"] > 0)
    ipw = pd.read_csv(f"{D}/inventory_position_weekly.csv", usecols=["part_id", "plant_id", "week_start"])
    have = pd.MultiIndex.from_arrays([ipw.part_id, ipw.plant_id, pd.to_datetime(ipw.week_start)])
    A = {k: [] for k in ("snap", "pp", "w", "week", "y", "fold")}
    for fold, (lo, hi) in FOLDS.items():
        for t0 in snapshots(lo, hi):
            weeks = [t0 + pd.Timedelta(days=7 * (w + 1)) for w in range(W)]
            weeks = [x - pd.Timedelta(days=x.weekday()) for x in weeks]
            idx = pd.MultiIndex.from_arrays([np.repeat(parts, W), np.repeat(plant, W), np.tile(weeks, P)])
            ok = idx.isin(have)
            y = acted.reindex(idx).fillna(False).to_numpy(bool)
            A["snap"].append(np.full(int(ok.sum()), np.datetime64(t0.date()), "datetime64[D]"))
            A["pp"].append(np.repeat(np.arange(P), W)[ok].astype(np.int32))
            A["w"].append(np.tile(np.arange(W), P)[ok].astype(np.int8))
            A["week"].append(np.tile(np.array(weeks, "datetime64[D]"), P)[ok])
            A["y"].append(y[ok]); A["fold"].append(np.full(int(ok.sum()), FOLD_ID[fold], np.int8))
    R = {k: np.concatenate(v) for k, v in A.items()}
    # IDENTITY GATE against Phase 15's stored rows (row keys do not depend on the fill seed; seed 7 is the reference)
    gate = {}
    for fold in ("val", "test"):
        z = np.load(os.path.join(SIMD, f"{fold}_s7.npz"))
        m = R["fold"] == FOLD_ID[fold]
        gate[fold] = dict(n_rebuilt=int(m.sum()), n_stored=int(len(z["pp"])),
                          pp_equal=bool(np.array_equal(R["pp"][m], z["pp"])),
                          week_equal=bool(np.array_equal(R["week"][m], z["week"])),
                          acted_equal=bool(np.array_equal(R["y"][m], z["acted"])))
        assert all(gate[fold][k] for k in ("pp_equal", "week_equal", "acted_equal")), f"IDENTITY GATE failed: {gate}"
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(ROWS, **R, parts=parts.astype(str), plants=plant.astype(str),
                        n_channels=sub.n_channels.to_numpy(np.int32))
    info = dict(stamp=st, gate=gate, universe=cov,
                rows={f: int((R["fold"] == i).sum()) for f, i in FOLD_ID.items()},
                base_rate={f: float(R["y"][R["fold"] == i].mean()) for f, i in FOLD_ID.items()},
                snapshots={f: int(len(np.unique(R["snap"][R["fold"] == i]))) for f, i in FOLD_ID.items()},
                orphan_rows={f: int(((sub.n_channels.to_numpy()[R["pp"]] == 0) & (R["fold"] == i)).sum())
                             for f, i in FOLD_ID.items()})
    print(json.dumps({k: v for k, v in info.items() if k != "universe"}, indent=1))
    C.dump(info, "phase17/b1_rows.json")


def load_rows():
    import numpy as np
    z = np.load(ROWS, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ================================================================== B1a LightGBM (torch-free process)
def run_lgbm():
    import lightgbm as lgb                                # FIRST, as in phase7_fit
    import numpy as np, pandas as pd
    import phase7_fit as P7
    st = C.require_clean()
    R = load_rows()
    Wd = P7.World("v8")
    keys = np.char.add(np.char.add(R["parts"], "|"), R["plants"])
    pairs = pd.DataFrame({"snap": R["snap"], "pp": R["pp"]}).drop_duplicates().reset_index(drop=True)
    known = set(Wd.pp_keys)
    pk = keys[pairs.pp.to_numpy()]
    has = np.array([k in known for k in pk])
    lb = pd.DataFrame({"key": pk[has], "snapshot_date": pd.to_datetime(pairs.snap.to_numpy()[has])})
    Xk = Wd.part_plant_features(lb)
    cols = list(Xk.columns)
    F = np.full((len(pairs), len(cols)), np.nan, np.float32)
    F[has] = Xk.to_numpy(np.float32)                      # orphan part-plants: every feature NaN (LightGBM routes NaN)
    pos = pd.MultiIndex.from_frame(pairs).get_indexer(pd.MultiIndex.from_arrays([R["snap"], R["pp"]]))
    assert (pos >= 0).all()
    X = np.concatenate([F[pos], R["w"][:, None].astype(np.float32)], 1)
    y = R["y"].astype(int)
    tr, va, te = (R["fold"] == 0), (R["fold"] == 1), (R["fold"] == 2)
    log = dict(stamp=st, features=cols + ["week_offset"], n_features=X.shape[1], orphan_pairs=int((~has).sum()), seeds={})
    for s in SEEDS:
        t = time.time()
        m = lgb.LGBMClassifier(**{**P7.GBM, "random_state": s, "objective": "binary"})
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
        pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
        np.savez_compressed(os.path.join(OUT, f"b1a_lgbm_s{s}.npz"), Pv=pv.astype(np.float32), Pt=pt.astype(np.float32))
        log["seeds"][s] = dict(best_iter=int(m.best_iteration_ or 400), seconds=time.time() - t, concurrency_level=CONCURRENCY,
                               device="cpu")
        print(f"  B1a seed {s}: best_iter {log['seeds'][s]['best_iter']}  {time.time() - t:.0f}s", flush=True)
    C.dump(log, "phase17/b1a_lgbm.json")


# ================================================================== B1b neural (GPU)
def run_neural(seed, max_epochs=None, bundle_root=None):
    import copy, numpy as np, pandas as pd, torch
    import torch.nn.functional as F
    import loop as L, phase5_heads as P5, temporal_share as TS, artifact_identity as AI
    from sklearn.metrics import average_precision_score
    st = C.require_clean()
    shipped = json.load(open(os.path.join(C.ML, "configs", "shipped.json")))
    t_ship = shipped["tasks"]["shortage_qty"]
    cfg = dict(task="rescue_week", world="v8", seed=int(seed), arch=t_ship["arch"], depth=int(t_ship["depth"]),
               lr=float(t_ship["lr"]), max_epochs=int(max_epochs or shipped["common"]["max_epochs"]),
               patience=int(shipped["common"]["patience"]), gate=False, wsla=False, fill_loss="rps", origin=None)
    out = os.path.join(bundle_root or C.BUND, "rescue_week", AI.bundle_name(cfg))
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}"); return
    R = load_rows()
    keys = np.char.add(np.char.add(R["parts"], "|"), R["plants"])
    L.seed_all(seed)
    tr_snaps = np.unique(R["snap"][R["fold"] == 0])
    D = P5.device_inputs("v8", [pd.Timestamp(s) for s in tr_snaps], False)
    Wg = dict(D["W"]); n_pp = len(Wg["pp_uniq"])
    Wg["pp_uniq"] = {**Wg["pp_uniq"], "__orphan__": n_pp}     # one extra pooled row with no members -> zero embedding
    D2 = dict(D, W=Wg)
    pp_idx = np.array([Wg["pp_uniq"].get(k, n_pp) for k in keys], np.int64)
    eye = np.eye(W, dtype=np.float32)

    def batches(fold):
        out_b = []
        m = R["fold"] == FOLD_ID[fold]
        for s in np.unique(R["snap"][m]):
            ii = np.flatnonzero(m & (R["snap"] == s))
            out_b.append((P5.t0_of(D["W"], pd.Timestamp(s)), ii,
                          torch.from_numpy(pp_idx[R["pp"][ii]]).to(P5.DEV),
                          torch.from_numpy(eye[R["w"][ii]]).to(P5.DEV),
                          torch.from_numpy(R["y"][ii].astype(np.float32)).to(P5.DEV)))
        return out_b

    btr, bva, bte = batches("train"), batches("val"), batches("test")
    L.seed_all(seed)                                     # init independent of the data path, as in loop.train
    model = P5.HeadNet(D["X"].shape[2], "shortage_qty", cfg["arch"], cfg["depth"], n_row_feats=W).to(P5.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=TS.HP["wd"])

    @torch.no_grad()
    def predict(bb):
        model.eval(); P, I = [], []
        for t0, ii, idx, xr, y in bb:
            P.append(torch.sigmoid(P5.forward(model, D2, t0, idx, xrow=xr)).float().cpu().numpy()); I.append(ii)
        return np.concatenate(P), np.concatenate(I)

    best, best_ep, best_state, bad, losses, vals, ep_s = None, -1, None, 0, [], [], []
    t_start = time.time()
    for ep in range(cfg["max_epochs"]):
        e0 = time.time(); model.train(); ls = []
        for t0, ii, idx, xr, y in btr:
            z = P5.forward(model, D2, t0, idx, xrow=xr)
            loss = F.binary_cross_entropy_with_logits(z, y)
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss.detach()))
        pv, iv = predict(bva)
        v = float(average_precision_score(R["y"][iv], pv))          # = phase5_heads.val_score for a binary head (PR-AUC)
        losses.append(float(np.mean(ls))); vals.append(v); ep_s.append(time.time() - e0)
        if best is None or v > best:
            best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    pv, iv = predict(bva); pt, it = predict(bte)
    os.makedirs(out, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "checkpoint.pt"))
    for fold, P, I in (("val", pv, iv), ("test", pt, it)):
        np.savez_compressed(os.path.join(out, f"preds_{fold}.npz"), P=P.astype(np.float32), Y=R["y"][I], row=I.astype(np.int64))
    log = dict(epochs_run=len(losses), best_epoch=best_ep, best_val=best, stop="patience" if bad >= cfg["patience"] else "CAP (floor)",
               losses=losses, vals=vals, seconds=time.time() - t_start, sec_per_epoch=float(np.mean(ep_s)),
               params=sum(p.numel() for p in model.parameters()), device=str(P5.DEV), tf32=bool(torch.backends.cudnn.allow_tf32),
               panel_host=P5.PANEL_HOST, tcn_checkpoint_chunk=P5.TCN_CHUNK, concurrency_level=CONCURRENCY,
               orphan_rows_val=int((pp_idx[R["pp"][iv]] == n_pp).sum()),
               peak_cuda_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    json.dump({**cfg, "identity": AI.identity_of(cfg), "stamps": st, "HP": TS.HP, "n_row_feats": W,
               "row_input": "one-hot week offset w (0..12)", "trained_by": "ml/train/phase17_b1.py", "complete": True},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"  [{log['stop']}] best_ep {best_ep} of {len(losses)}  val PR-AUC {best:.5f}  {log['seconds']:.0f}s  "
          f"{log['sec_per_epoch']:.1f}s/ep  -> {out}", flush=True)


# ================================================================== score (B1a, B1b, Phase 14's crude proxy)
def score():
    import numpy as np
    import phase15 as P15
    from phase14_score import run_arm, majority_arm
    from sklearn.metrics import roc_auc_score, average_precision_score
    st = C.require_clean()
    R = load_rows()
    yv, yt = R["y"][R["fold"] == 1].astype(int), R["y"][R["fold"] == 2].astype(int)

    def arm_pairs(name):
        pairs, seeds = [], []
        for s in SEEDS:
            if name == "proxy":                            # Phase 14/15's simulation score, per fill seed
                zv, zt = (np.load(os.path.join(SIMD, f"{f}_s{s}.npz")) for f in ("val", "test"))
                pairs.append((zv["p"], zv["acted"].astype(int), zt["p"], zt["acted"].astype(int)))
            elif name == "b1a":
                p = os.path.join(OUT, f"b1a_lgbm_s{s}.npz")
                if not os.path.exists(p): continue
                z = np.load(p); pairs.append((z["Pv"], yv, z["Pt"], yt))
            else:
                b = os.path.join(C.BUND, "rescue_week", f"v8_mp_h1_lr0.000125_s{s}")
                if not (os.path.exists(os.path.join(b, "config.json")) and json.load(open(os.path.join(b, "config.json"))).get("complete")):
                    continue
                zv, zt = np.load(os.path.join(b, "preds_val.npz")), np.load(os.path.join(b, "preds_test.npz"))
                ov, ot = np.argsort(zv["row"]), np.argsort(zt["row"])
                pairs.append((zv["P"][ov], zv["Y"][ov].astype(int), zt["P"][ot], zt["Y"][ot].astype(int)))
            seeds.append(s)
        return pairs, seeds

    def prec_at_min_recall(pairs, r=0.20):
        """VALIDATION: the highest-precision threshold whose validation recall >= r; applied unchanged to TEST."""
        per = []
        for sv, yv_, st_, yt_ in pairs:
            ss, yy, tp, n = P15.sorted_cum(sv, yv_)
            last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
            rec, prec = tp[last] / yy.sum(), tp[last] / n[last]
            ok = rec >= r
            i = last[np.flatnonzero(ok)[np.argmax(prec[ok])]]
            a = P15.apply(st_, yt_, float(ss[i]))
            per.append(dict(tau=float(ss[i]), val_precision=float(tp[i] / n[i]), val_recall=float(tp[i] / yy.sum()),
                            test_precision=a["precision"], test_recall=a["recall"], test_coverage=a["coverage"]))
        return dict(per_seed=per, test_precision=P15.band([p["test_precision"] for p in per]),
                    test_recall=P15.band([p["test_recall"] for p in per]), val_precision=P15.band([p["val_precision"] for p in per]))

    def prec_at_recall(sv, yv_, r):
        ss, yy, tp, n = P15.sorted_cum(sv, yv_)
        i = int(np.searchsorted(tp / yy.sum(), r))
        return float(tp[min(i, len(tp) - 1)] / n[min(i, len(tp) - 1)])

    out = dict(stamp=st, concurrency_level=CONCURRENCY, base_rate=dict(val=float(yv.mean()), test=float(yt.mean())),
               majority=majority_arm(yv, yt)["f1"], arms={})
    proxy_pairs, _ = arm_pairs("proxy")
    # the proxy's VALIDATION recall at Phase 14's theta grid -> each arm's validation precision at that matched recall (P4)
    proxy_rec = {th: float(np.mean([((sv >= th) & (yv_ == 1)).sum() / yv_.sum() for sv, yv_, _, _ in proxy_pairs]))
                 for th in (0.1, 0.3, 0.5)}
    for name in ("proxy", "b1a", "b1b"):
        pairs, seeds = arm_pairs(name)
        if not pairs:
            out["arms"][name] = dict(status="NOT RUN"); continue
        A = P15.analyse(pairs)
        a = dict(seeds=seeds, n_seeds=len(seeds), coverage_curve=A["curve"], recall_at_precision=A["recall_at_precision"],
                 f1_threshold=run_arm(pairs)["f1"], precision_at_recall_ge_0_20=prec_at_min_recall(pairs),
                 roc_auc=dict(val=P15.band([roc_auc_score(p[1], p[0]) for p in pairs]), test=P15.band([roc_auc_score(p[3], p[2]) for p in pairs])),
                 pr_auc=dict(val=P15.band([average_precision_score(p[1], p[0]) for p in pairs]), test=P15.band([average_precision_score(p[3], p[2]) for p in pairs])),
                 val_precision_at_proxy_recall={str(th): dict(recall=r, precision=P15.band([prec_at_recall(p[0], p[1], r) for p in pairs]))
                                                for th, r in proxy_rec.items()})
        out["arms"][name] = a
        print(f"{name}: seeds {seeds}  test PR-AUC {a['pr_auc']['test']}  P@R>=0.20 test {a['precision_at_recall_ge_0_20']['test_precision']}")
    C.dump(out, "phase17/b1_score.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rows", "lgbm", "neural", "score"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only")
    ap.add_argument("--bundle-root", default=None, help="smoke runs only")
    a = ap.parse_args()
    {"rows": build_rows, "lgbm": run_lgbm, "score": score}.get(a.mode, lambda: run_neural(a.seed, a.max_epochs, a.bundle_root))()
