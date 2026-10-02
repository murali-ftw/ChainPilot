"""Phase 17 B2 -- the Delta-strain capacity head. The TARGET changes; the architecture, LR and batching do not.

  B2a  incumbent mp h4 (stored bundles v8_mp_h4_lr0.00025_s{7..47}, reused -- the Stage -1 handshake was inside band)
  B2b  the incumbent architecture (P5.HeadNet, mp, depth 4, lr 2.5e-4) trained on Delta = strain - m(channel, t0);
       its level forecast is m + the predicted Delta quantiles
  B2c  the same encoder with TWO quantile heads -- level and Delta -- trained on the sum of their pinball losses;
       alert on either crossing: score = max(P_level(strain > 1), P_delta(m + Delta > 1))

m(channel, t0) = the TRAILING-WINDOW MEDIAN: the median of that channel's capacity_strain labels whose 90-day label window
ENDED within the 52 weeks up to t0 (label_window_end in (t0 - 364 d, t0]). A label is only known once its window has ended,
so m is as-of by construction, and asserted. No label in the window: the channel's last ended label; none ever: the
training-fold median of m. Both fallbacks are counted.

The decision is Phase 14/15's UC3 ("demand exceeds capacity", strain > 1) scored by Phase 15's machinery: P(strain > 1)
through phase14_score.p_exceed, operating points chosen on VALIDATION per seed. PASS = a disjoint recall gain at fixed
precision over B2a.

  python ml/train/phase17_b2.py train --arm b2b --seed 7
  python ml/train/phase17_b2.py score
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval")]
import phase12_common as C

TASK = "capacity_strain"
SEEDS = C.V8_SEEDS
CONCURRENCY = 1
TARGET_OF = {"b2b": "delta", "b2c": "level_delta"}


def trailing_median(lb):
    """m aligned to lb's rows (lb = P5.labels(v8, capacity_strain)); as-of asserted."""
    import numpy as np, pandas as pd
    from config import WORLDS
    lab = pd.read_csv(os.path.join(WORLDS["v8"], "training_labels.csv"),
                      usecols=["entity_id", "task", "snapshot_date", "label_window_end", "label_value"])
    lab = lab[lab.task == TASK]
    lab["end"] = pd.to_datetime(lab.label_window_end)
    g = {c: (d.end.to_numpy(), d.label_value.to_numpy(float)) for c, d in lab.sort_values("end").groupby("entity_id")}
    m = np.full(len(lb), np.nan); how = np.zeros(len(lb), np.int8)     # 0 window, 1 last-ended, 2 global
    ents, snaps = lb.entity_id.to_numpy(), pd.to_datetime(lb.snapshot_date).to_numpy()
    year = np.timedelta64(364, "D")
    for i, (c, t0) in enumerate(zip(ents, snaps)):
        ends, vals = g[c]
        hi = int(np.searchsorted(ends, t0, "right")); lo = int(np.searchsorted(ends, t0 - year, "right"))
        if hi > lo:
            assert ends[hi - 1] <= t0, "as-of violation: a label window ending after t0 entered m"
            m[i] = np.median(vals[lo:hi])
        elif hi > 0:
            assert ends[hi - 1] <= t0
            m[i] = vals[hi - 1]; how[i] = 1
    return m, how


def train(arm, seed, max_epochs=None, bundle_root=None):
    import copy, numpy as np, pandas as pd, torch
    import loop as L, phase5_heads as P5, temporal_share as TS, artifact_identity as AI, phase5_metrics as M
    from heads import QuantileHead
    st = C.require_clean()
    shipped = json.load(open(os.path.join(C.ML, "configs", "shipped.json")))
    t_ship = shipped["tasks"][TASK]
    cfg = dict(task=TASK, world="v8", seed=int(seed), arch=t_ship["arch"], depth=int(t_ship["depth"]), lr=float(t_ship["lr"]),
               max_epochs=int(max_epochs or shipped["common"]["max_epochs"]), patience=int(shipped["common"]["patience"]),
               gate=False, wsla=False, fill_loss="rps", origin=None, cap_target=TARGET_OF[arm])
    out = os.path.join(bundle_root or C.BUND, TASK, AI.bundle_name(cfg))
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}"); return
    L.seed_all(seed)
    lb = P5.labels("v8", TASK)
    tr, va, te = L.split_of(cfg, lb.snapshot_date)
    m, how = trailing_median(lb)
    m_fill = float(np.nanmedian(m[tr])); n_global = int(np.isnan(m).sum()); m = np.where(np.isnan(m), m_fill, m)
    how = np.where(how == 0, how, how); y = lb.label_value.to_numpy(float); d = y - m
    ymu, ysd = float(y[tr].mean()), float(y[tr].std()); dmu, dsd = float(d[tr].mean()), float(d[tr].std())
    D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
    W = D["W"]
    lbd = lb.copy(); lbd["label_value"] = d
    def both(mask):
        bl, bd = P5.batches(TASK, lb, mask, W, ymu, ysd), P5.batches(TASK, lbd, mask, W, dmu, dsd)
        assert all(np.array_equal(a[2], b[2]) for a, b in zip(bl, bd)), "level/delta batches out of order"
        return [(t0, tg["idx"], tg["y"], tgd["y"], ii) for (t0, tg, ii), (_, tgd, _) in zip(bl, bd)]
    btr, bva, bte = both(tr), both(va), both(te)
    L.seed_all(seed)
    model = P5.HeadNet(D["X"].shape[2], TASK, cfg["arch"], cfg["depth"]).to(P5.DEV)
    two = arm == "b2c"
    if two:
        model.head_d = QuantileHead(model.head.net[0].in_features).to(P5.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=TS.HP["wd"])

    def enc(t0, idx):
        sl = slice(t0 - P5.WIN + 1, t0 + 1); X = D["X"][:, sl]
        if X.device.type != P5.DEV.type:
            X = X.to(P5.DEV, non_blocking=True)
        return model.encode(X, D["dt"][:, sl], D["obs"][:, sl], W)[idx]

    @torch.no_grad()
    def predict(bb):
        model.eval(); QL, QD, I = [], [], []
        for t0, idx, yl, yd, ii in bb:
            hi = enc(t0, idx)
            if two:
                QL.append((model.head(hi)[:, 0, :] * ysd + ymu).float().cpu().numpy())
                QD.append((model.head_d(hi)[:, 0, :] * dsd + dmu).float().cpu().numpy() + m[ii][:, None])
            else:
                QD.append((model.head(hi)[:, 0, :] * dsd + dmu).float().cpu().numpy() + m[ii][:, None])
            I.append(ii)
        I = np.concatenate(I)
        return (np.concatenate(QL) if two else None), np.concatenate(QD), I

    def vscore(QL, QD, I):
        pin = lambda Q: float(np.mean([M.pinball_rows(y[I], Q[:, j], q).mean() for j, q in enumerate(M.QS)]))
        return (pin(QL) + pin(QD)) / 2 if two else pin(QD)

    best, best_ep, best_state, bad, losses, vals, ep_s = None, -1, None, 0, [], [], []
    t_start = time.time()
    for ep in range(cfg["max_epochs"]):
        e0 = time.time(); model.train(); ls = []
        for t0, idx, yl, yd, ii in btr:
            hi = enc(t0, idx)
            if two:
                loss = model.head.loss(model.head(hi), yl) + model.head_d.loss(model.head_d(hi), yd)
            else:
                loss = model.head.loss(model.head(hi), yd)
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss.detach()))
        v = vscore(*predict(bva))
        losses.append(float(np.mean(ls))); vals.append(v); ep_s.append(time.time() - e0)
        if best is None or v < best:
            best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    os.makedirs(out, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "checkpoint.pt"))
    for fold, bb in (("val", bva), ("test", bte)):
        QL, QD, I = predict(bb)
        np.savez_compressed(os.path.join(out, f"preds_{fold}.npz"), P=(QL if two else QD).astype(np.float32),
                            PD=QD.astype(np.float32), M=m[I].astype(np.float32), Y=y[I], row=I.astype(np.int64))
    log = dict(epochs_run=len(losses), best_epoch=best_ep, best_val=best, stop="patience" if bad >= cfg["patience"] else "CAP (floor)",
               losses=losses, vals=vals, seconds=time.time() - t_start, sec_per_epoch=float(np.mean(ep_s)),
               params=sum(p.numel() for p in model.parameters()), device=str(P5.DEV), tf32=bool(torch.backends.cudnn.allow_tf32),
               panel_host=P5.PANEL_HOST, tcn_checkpoint_chunk=P5.TCN_CHUNK, concurrency_level=CONCURRENCY,
               trailing_median=dict(window_rows=int((how == 0).sum() - n_global), last_ended_fallback=int((how == 1).sum()),
                                    global_fallback=n_global, global_value=m_fill),
               target_moments=dict(level=[ymu, ysd], delta=[dmu, dsd]),
               peak_cuda_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    json.dump({**cfg, "identity": AI.identity_of(cfg), "stamps": st, "HP": TS.HP, "arm": arm,
               "trained_by": "ml/train/phase17_b2.py", "complete": True}, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"  [{log['stop']}] best_ep {best_ep} of {len(losses)}  val pinball {best:.5f}  {log['seconds']:.0f}s  "
          f"{log['sec_per_epoch']:.1f}s/ep  -> {out}", flush=True)


def score():
    import numpy as np
    import phase15 as P15, phase5_heads as P5, folds
    from phase14_score import p_exceed
    st = C.require_clean()
    inc, _, _ = P15.capacity_arrays()
    arms = {"B2a_incumbent_mp_h4": inc["mp_h4|raw"]}
    lb = P5.labels("v8", TASK); tr, va, te = folds.fixed_split(lb.snapshot_date)
    yv_ref, yt_ref = arms["B2a_incumbent_mp_h4"][0][1], arms["B2a_incumbent_mp_h4"][0][3]
    seeds_of = {"B2a_incumbent_mp_h4": list(SEEDS)}
    for arm, tgt in (("B2b_delta", "delta"), ("B2c_level_delta", "level_delta"),
                     ("DIAGNOSTIC_B2c_level_head_only", "level_head")):     # not an arm: no verdict is read off it
        suf = {"delta": "_tgtdelta", "level_delta": "_tgtlvldelta", "level_head": "_tgtlvldelta"}[tgt]
        pairs, seeds = [], []
        for s in SEEDS:
            b = os.path.join(C.BUND, TASK, f"v8_mp_h4_lr0.00025_s{s}{suf}")
            if not (os.path.exists(os.path.join(b, "config.json")) and json.load(open(os.path.join(b, "config.json"))).get("complete")):
                continue
            zv, zt = np.load(os.path.join(b, "preds_val.npz")), np.load(os.path.join(b, "preds_test.npz"))
            assert np.array_equal((zv["Y"] > 1).astype(int), yv_ref) and np.array_equal((zt["Y"] > 1).astype(int), yt_ref), \
                f"{arm} s{s}: rows misaligned with the stored incumbent"
            sc = ((lambda z: p_exceed(z["P"])) if tgt in ("delta", "level_head")
                  else (lambda z: np.maximum(p_exceed(z["P"]), p_exceed(z["PD"]))))
            pairs.append((sc(zv), yv_ref, sc(zt), yt_ref)); seeds.append(s)
        if pairs:
            arms[arm] = pairs; seeds_of[arm] = seeds
    out = dict(stamp=st, concurrency_level=CONCURRENCY, decision="UC3: strain > 1 (90-day), P(strain > 1) via p_exceed",
               base_rate=dict(val=float(yv_ref.mean()), test=float(yt_ref.mean())), arms={})
    for a, pairs in arms.items():
        A = P15.analyse(pairs)
        out["arms"][a] = dict(seeds=seeds_of[a], curve=A["curve"], recall_at_precision=A["recall_at_precision"], stage_b=A["stage_b"]["verdict"])
    inc_r = out["arms"]["B2a_incumbent_mp_h4"]["recall_at_precision"]
    for a in out["arms"]:
        if a == "B2a_incumbent_mp_h4":
            continue
        cmp = {}
        for p, r in out["arms"][a]["recall_at_precision"].items():
            ib = inc_r.get(p, {}).get("test_recall"); ab = r.get("test_recall")
            cmp[p] = ("UNREACHABLE" if ab is None or ib is None else
                      "GAIN (disjoint)" if ab[0] > ib[2] else "WORSE (disjoint)" if ab[2] < ib[0] else "UNDETERMINED (bands overlap)")
        out["arms"][a]["vs_B2a_recall_at_fixed_precision"] = cmp
    C.dump(out, "phase17/b2_score.json")
    for a, v in out["arms"].items():
        print(a, v["seeds"], {p: (r.get("test_recall"), r.get("test_precision")) for p, r in v["recall_at_precision"].items()},
              v.get("vs_B2a_recall_at_fixed_precision"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--arm", choices=["b2b", "b2c"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--bundle-root", default=None)
    a = ap.parse_args()
    train(a.arm, a.seed, a.max_epochs, a.bundle_root) if a.mode == "train" else score()
