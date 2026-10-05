"""Phase 24 Stage 6 -- the neural rescue head B1b on CLEAN inputs, all five seeds. New file.

B1b is ml/train/phase17_b1.py `run_neural`, reproduced here with exactly two changes (pre-registered,
reports/part2/phase-24-preregistration.md):
  1. the channel panel is Phase 22's CLEAN cache `v8clean` (the nine leaking columns replaced by as-of-safe ones) instead of `v8`;
  2. bundles are written under ml/artifacts/phase24/bundles/rescue_week/<artifact_identity.bundle_name(cfg)> (world = v8clean).
Everything else is Phase 17's: the shipped shortage head (mp h1, lr 1.25e-4, shipped.json), max_epochs 120, patience 8,
one optimiser step per snapshot, the one-hot week offset through the existing per-row input, the zero orphan row, AdamW
with temporal_share.HP["wd"], best-validation-PR-AUC state, concurrency 1.
Plus the pre-registered wall-clock cap (deviation 264): 40 minutes per seed; a seed that reaches it keeps its best state and
is recorded "CAP (floor)", exactly like the epoch cap.

ROWS: phase17_b1.build_rows' construction, written to ml/artifacts/phase24/b1_rows.npz (ml/artifacts/phase17 is absent on
this machine) and GATED element for element against the stored phase15_sim rows (G6; a flipped label must fire).
inventory_position_weekly: read for that row filter only (which part-plant-weeks exist), as in Phase 17. Never a feature.

  python ml/train/phase24_b1b_clean.py rows
  python ml/train/phase24_b1b_clean.py handshake          (Stage -1: stored arrival lite h4 s7 on v8, 3 epochs)
  python ml/train/phase24_b1b_clean.py neural --seed 7     (one cell; MPS; concurrency 1)
  python ml/train/phase24_b1b_clean.py score
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval"), HERE]
import phase12_common as C

ART24 = os.path.join(C.ART, "phase24")
ROWS = os.path.join(ART24, "b1_rows.npz")
BUND24 = os.path.join(ART24, "bundles")
SIMD = os.path.join(C.ART, "phase15_sim")
SIM24 = os.path.join(ART24, "sim")
D = os.path.join(C.REPO, "db", "gen_v8", "seed_1001")
W = 13
FOLDS = {"train": ("2019-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")}
FOLD_ID = {"train": 0, "val": 1, "test": 2}
SEEDS = (7, 17, 27, 37, 47)
WORLD = "v8clean"
CONCURRENCY = 1
WALL_CAP_MIN = 40.0
B1A_CLEAN_BAND = (0.7325, 0.7336)          # QUOTED, reports/part2/phase-23b.md §1 / phase23b/stage2_rescue_clean.md b
B1A_CLEAN_POINT = 0.733


class NotClean(Exception):
    pass


def require_clean_cache(world):
    """G7: Stage 6 reads only a clean cache."""
    from clean_panel import LEAKING
    meta = json.load(open(os.path.join(C.ART, "cache", world, "meta.json")))
    if not meta.get("clean_of") or list(meta.get("replaced_columns") or []) != list(LEAKING):
        raise NotClean(f"cache {world!r} is not a clean cache (clean_of={meta.get('clean_of')!r})")
    return dict(world=world, clean_of=meta["clean_of"], replaced_columns=meta["replaced_columns"])


def g7_failing_case():
    try:
        require_clean_cache("v8"); return "DID NOT RAISE"
    except NotClean as e:
        return f"RAISED: {e}"


# ================================================================== rows (phase17_b1.build_rows, output under phase24/)
def gate_rows(R, flip=False):
    import numpy as np
    gate = {}
    for fold in ("val", "test"):
        z = np.load(os.path.join(SIMD, f"{fold}_s7.npz"))
        m = R["fold"] == FOLD_ID[fold]
        y = R["y"][m].copy()
        if flip:
            y[0] = ~y[0]
        gate[fold] = dict(n_rebuilt=int(m.sum()), n_stored=int(len(z["pp"])), pp_equal=bool(np.array_equal(R["pp"][m], z["pp"])),
                          week_equal=bool(np.array_equal(R["week"][m], z["week"])), acted_equal=bool(np.array_equal(y, z["acted"])))
    return gate, all(g[k] for g in gate.values() for k in ("pp_equal", "week_equal", "acted_equal"))


def build_rows():
    import numpy as np, pandas as pd
    import montecarlo as MC
    from phase17_b1 import snapshots
    st = C.require_clean()
    pp, _, cov = MC.part_plant_universe("v8")
    sub = pp.reset_index(drop=True)
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
    gate, ok = gate_rows(R)
    _, flipped_ok = gate_rows(R, flip=True)
    assert ok, f"G6 IDENTITY GATE failed: {gate}"
    assert not flipped_ok, "G6 cannot fail: a flipped label still passes"
    os.makedirs(ART24, exist_ok=True)
    np.savez_compressed(ROWS, **R, parts=parts.astype(str), plants=plant.astype(str), n_channels=sub.n_channels.to_numpy(np.int32))
    info = dict(stamp=st, gate=gate, G6_failing_case_flipped_label_fires=not flipped_ok,
                rows={f: int((R["fold"] == i).sum()) for f, i in FOLD_ID.items()},
                base_rate={f: float(R["y"][R["fold"] == i].mean()) for f, i in FOLD_ID.items()},
                snapshots={f: int(len(np.unique(R["snap"][R["fold"] == i]))) for f, i in FOLD_ID.items()})
    print(json.dumps(info, indent=1, default=str))
    C.dump(info, "phase24/b1_rows.json")


def load_rows():
    import numpy as np
    z = np.load(ROWS, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ================================================================== Stage -1 handshake (phase22_train.handshake, output under phase24/)
def handshake():
    import loop as L
    import phase22_train as T22
    T22.register_clean()
    st = C.require_clean()
    cfg = T22._resolve("arrival", 7, "v8", max_epochs=3)
    t = time.time()
    _, _, _, _, log = L.train(cfg)
    stored = [json.load(open(os.path.join(C.BUND, T22.STORED["arrival"].format(s=s), "train_log.json")))["vals"][:3] for s in C.V8_SEEDS]

    def check(vals):
        rows = []
        for e in range(3):
            b = [v[e] for v in stored]
            rows.append(dict(epoch=e, here=vals[e], stored_s7=stored[0][e], abs_diff=abs(vals[e] - stored[0][e]),
                             band=[min(b), max(b)], inside=bool(min(b) <= vals[e] <= max(b))))
        return rows
    rows = check(log["vals"])
    falsify = check([v + 0.05 for v in log["vals"]])
    res = dict(stamp=st, device=str(L.DEV), seconds=time.time() - t, epochs=rows, inside_band=all(r["inside"] for r in rows),
               falsification_shifted_0p05=dict(rows=falsify, outside=not all(r["inside"] for r in falsify)),
               label="proceed" if all(r["inside"] for r in rows) else "within-machine comparison")
    os.makedirs(ART24, exist_ok=True)
    C.dump(res, "phase24/handshake.json")
    print(json.dumps(res, indent=1, default=str))
    assert res["falsification_shifted_0p05"]["outside"], "the handshake check cannot fail -- invalid"


# ================================================================== B1b neural, clean (phase17_b1.run_neural, two changes)
def run_neural(seed, max_epochs=None):
    import copy, numpy as np, pandas as pd, torch
    import torch.nn.functional as F
    import phase22_train as T22
    T22.register_clean()
    import loop as L, phase5_heads as P5, temporal_share as TS, artifact_identity as AI
    from sklearn.metrics import average_precision_score
    st = C.require_clean()
    cache = require_clean_cache(WORLD)
    shipped = json.load(open(os.path.join(C.ML, "configs", "shipped.json")))
    t_ship = shipped["tasks"]["shortage_qty"]
    cfg = dict(task="rescue_week", world=WORLD, seed=int(seed), arch=t_ship["arch"], depth=int(t_ship["depth"]),
               lr=float(t_ship["lr"]), max_epochs=int(max_epochs or shipped["common"]["max_epochs"]),
               patience=int(shipped["common"]["patience"]), gate=False, wsla=False, fill_loss="rps", origin=None)
    out = os.path.join(BUND24, "rescue_week", AI.bundle_name(cfg))
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}"); return
    R = load_rows()
    keys = np.char.add(np.char.add(R["parts"], "|"), R["plants"])
    L.seed_all(seed)
    tr_snaps = np.unique(R["snap"][R["fold"] == 0])
    Dd = P5.device_inputs(WORLD, [pd.Timestamp(s) for s in tr_snaps], False)
    Wg = dict(Dd["W"]); n_pp = len(Wg["pp_uniq"])
    Wg["pp_uniq"] = {**Wg["pp_uniq"], "__orphan__": n_pp}
    D2 = dict(Dd, W=Wg)
    pp_idx = np.array([Wg["pp_uniq"].get(k, n_pp) for k in keys], np.int64)
    eye = np.eye(W, dtype=np.float32)

    def batches(fold):
        out_b = []
        m = R["fold"] == FOLD_ID[fold]
        for s in np.unique(R["snap"][m]):
            ii = np.flatnonzero(m & (R["snap"] == s))
            out_b.append((P5.t0_of(Dd["W"], pd.Timestamp(s)), ii,
                          torch.from_numpy(pp_idx[R["pp"][ii]]).to(P5.DEV),
                          torch.from_numpy(eye[R["w"][ii]]).to(P5.DEV),
                          torch.from_numpy(R["y"][ii].astype(np.float32)).to(P5.DEV)))
        return out_b

    btr, bva, bte = batches("train"), batches("val"), batches("test")
    L.seed_all(seed)
    model = P5.HeadNet(Dd["X"].shape[2], "shortage_qty", cfg["arch"], cfg["depth"], n_row_feats=W).to(P5.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=TS.HP["wd"])

    @torch.no_grad()
    def predict(bb):
        model.eval(); P, I = [], []
        for t0, ii, idx, xr, y in bb:
            P.append(torch.sigmoid(P5.forward(model, D2, t0, idx, xrow=xr)).float().cpu().numpy()); I.append(ii)
        return np.concatenate(P), np.concatenate(I)

    best, best_ep, best_state, bad, losses, vals, ep_s = None, -1, None, 0, [], [], []
    t_start = time.time(); stop = None
    for ep in range(cfg["max_epochs"]):
        e0 = time.time(); model.train(); ls = []
        for t0, ii, idx, xr, y in btr:
            z = P5.forward(model, D2, t0, idx, xrow=xr)
            loss = F.binary_cross_entropy_with_logits(z, y)
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss.detach()))
        pv, iv = predict(bva)
        v = float(average_precision_score(R["y"][iv], pv))
        losses.append(float(np.mean(ls))); vals.append(v); ep_s.append(time.time() - e0)
        if best is None or v > best:
            best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"   s{seed} ep {ep} loss {losses[-1]:.5f} val PR-AUC {v:.5f} ({ep_s[-1]:.1f}s)", flush=True)
        if bad >= cfg["patience"]:
            stop = "patience"; break
        if (time.time() - t_start) / 60 >= WALL_CAP_MIN:
            stop = "CAP (floor): wall-clock 40 min"; break
    if stop is None:
        stop = "CAP (floor): 120 epochs"
    model.load_state_dict(best_state)
    pv, iv = predict(bva); pt, it = predict(bte)
    os.makedirs(out, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "checkpoint.pt"))
    for fold, P, I in (("val", pv, iv), ("test", pt, it)):
        np.savez_compressed(os.path.join(out, f"preds_{fold}.npz"), P=P.astype(np.float32), Y=R["y"][I], row=I.astype(np.int64))
    log = dict(epochs_run=len(losses), best_epoch=best_ep, best_val=best, stop=stop, losses=losses, vals=vals,
               seconds=time.time() - t_start, per_seed_minutes=(time.time() - t_start) / 60, sec_per_epoch=float(np.mean(ep_s)),
               params=sum(p.numel() for p in model.parameters()), device=str(P5.DEV), concurrency_level=CONCURRENCY,
               machine="Apple M4 Pro", wall_cap_min=WALL_CAP_MIN, clean_cache=cache,
               orphan_rows_val=int((pp_idx[R["pp"][iv]] == n_pp).sum()))
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    json.dump({**cfg, "identity": AI.identity_of(cfg), "stamps": st, "HP": TS.HP, "n_row_feats": W,
               "row_input": "one-hot week offset w (0..12)", "trained_by": "ml/train/phase24_b1b_clean.py", "complete": True},
              open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"  [{stop}] best_ep {best_ep} of {len(losses)}  val PR-AUC {best:.5f}  {log['per_seed_minutes']:.1f} min  "
          f"{log['sec_per_epoch']:.1f}s/ep  -> {out}", flush=True)


# ================================================================== score
def naive_rate(R):
    """phase17_b1.score.naive_rate, line for line (a deterministic reference row, not a model)."""
    import numpy as np, pandas as pd
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "event_ts", "recorded_ts"])
    tx = tx[tx.txn_type == "transfer_in"]
    tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
    tx["rec"] = pd.to_datetime(tx.recorded_ts)
    keymap = {(p, q): i for i, (p, q) in enumerate(zip(R["parts"], R["plants"]))}
    tx["pp"] = [keymap.get(k, -1) for k in zip(tx.part_id, tx.plant_id)]
    tx = tx[tx.pp >= 0]
    score = np.zeros(len(R["y"]))
    for s in np.unique(R["snap"][R["fold"] > 0]):
        t0 = pd.Timestamp(s)
        u = tx[(tx.rec <= t0) & (tx.week > t0 - pd.Timedelta(weeks=52))]
        assert (u.rec <= t0).all()
        rate = u.groupby("pp").week.nunique().reindex(range(len(R["parts"])), fill_value=0).to_numpy() / 52.0
        m = R["snap"] == s
        score[m] = rate[R["pp"][m]]
    return score


def b1b_pairs(seeds=SEEDS):
    import numpy as np
    import artifact_identity as AI
    if tuple(sorted(seeds)) != SEEDS:
        raise ValueError(f"G4: scorer requires all five seeds {SEEDS}; got {tuple(seeds)}")
    pairs, logs = [], {}
    for s in seeds:
        cfg = dict(task="rescue_week", world=WORLD, seed=s, arch="mp", depth=1, lr=0.000125, gate=False, wsla=False,
                   fill_loss="rps", origin=None, max_epochs=120, patience=8)
        b = os.path.join(BUND24, "rescue_week", AI.bundle_name(cfg))
        c = json.load(open(os.path.join(b, "config.json")))
        assert c.get("complete") and c["world"] == WORLD and c["seed"] == s, f"incomplete or wrong bundle {b}"
        zv, zt = np.load(os.path.join(b, "preds_val.npz")), np.load(os.path.join(b, "preds_test.npz"))
        ov, ot = np.argsort(zv["row"]), np.argsort(zt["row"])
        pairs.append((zv["P"][ov], zv["Y"][ov].astype(int), zt["P"][ot], zt["Y"][ot].astype(int)))
        logs[s] = json.load(open(os.path.join(b, "train_log.json")))
    return pairs, logs


def score():
    import numpy as np
    import phase15 as P15
    from phase20_decisions import reachable_and_class
    from phase24_sim_clean import prec_at_min_recall_tau
    from sklearn.metrics import roc_auc_score, average_precision_score
    st = C.require_clean()
    R = load_rows()
    yv, yt = R["y"][R["fold"] == 1].astype(int), R["y"][R["fold"] == 2].astype(int)
    snap_t = R["snap"][R["fold"] == 2]; su = np.unique(snap_t); snap_i = np.searchsorted(su, snap_t)
    out = dict(stamp=st, base_rate=dict(val=float(yv.mean()), test=float(yt.mean())), concurrency_level=CONCURRENCY)
    try:
        b1b_pairs(SEEDS[:4]); out["G4_failing_case"] = "DID NOT RAISE"
    except ValueError as e:
        out["G4_failing_case"] = f"RAISED: {e}"
    out["G7_failing_case"] = g7_failing_case()
    pairs, logs = b1b_pairs()
    for (sv, yv_, st_, yt_) in pairs:
        assert np.array_equal(yv_, yv) and np.array_equal(yt_, yt), "B1b rows differ from the B1 rows"
    sim = {a: [(np.load(os.path.join(SIM24, a, f"val_s{s}.npz"))["p"], np.load(os.path.join(SIM24, a, f"test_s{s}.npz"))["p"]) for s in SEEDS]
           for a in ("leaky", "clean")}
    nr = naive_rate(R); nv, nt = nr[R["fold"] == 1], nr[R["fold"] == 2]

    def arm_summary(ps):
        an = P15.analyse(ps)
        op = []
        for sv, yv_, st_, yt_ in ps:
            tau, vp, vr = prec_at_min_recall_tau(sv, yv_)
            a = P15.apply(st_, yt_, tau)
            op.append(dict(tau=tau, val_precision=vp, val_recall=vr, test_precision=a["precision"], test_recall=a["recall"],
                           test_coverage=a["coverage"]))
        cov = {f"{int(k * 100)}%": [x for x in an["curve"] if abs(x["k"] - k) < 1e-12][0].get("precision") for k in (0.01, 0.05, 0.10, 0.20)}
        return dict(n_seeds=len(ps), op_precision=P15.band([o["test_precision"] for o in op]), op_recall=P15.band([o["test_recall"] for o in op]),
                    op_per_seed=op, lift=P15.band([o["test_precision"] / yt.mean() for o in op]), precision_at_coverage=cov,
                    roc_auc=P15.band([roc_auc_score(p[3], p[2]) for p in ps]), pr_auc=P15.band([average_precision_score(p[3], p[2]) for p in ps]),
                    phase15_class=reachable_and_class(an))
    arms = {"b1b_clean": [(p[0], p[1], p[2], p[3]) for p in pairs],
            "sim_proxy_clean": [(v, yv, t, yt) for v, t in sim["clean"]],
            "sim_proxy_leaky": [(v, yv, t, yt) for v, t in sim["leaky"]]}
    out["arms"] = {k: arm_summary(v) for k, v in arms.items()}
    out["arms"]["own_history"] = dict(op=prec_at_min_recall_tau(nv, yv), test=P15.apply(nt, yt, prec_at_min_recall_tau(nv, yv)[0]))
    # ---- 5-seed ensembles at their own validation-chosen operating point; paired snapshot-block bootstrap
    ens = {"b1b_clean": (np.mean([p[0] for p in pairs], 0), np.mean([p[2] for p in pairs], 0)),
           "sim_proxy_clean": (np.mean([v for v, _ in sim["clean"]], 0), np.mean([t for _, t in sim["clean"]], 0)),
           "own_history": (nv, nt)}
    flags = {}
    for k, (v, t) in ens.items():
        tau = prec_at_min_recall_tau(v, yv)[0]; flags[k] = t >= tau
    K = len(su); tp = {k: np.bincount(snap_i, f & (yt == 1), K) for k, f in flags.items()}; al = {k: np.bincount(snap_i, f, K) for k, f in flags.items()}
    rng = np.random.default_rng(2024); bs = {k: [] for k in flags}; diffs = {"b1b - sim_proxy_clean": [], "b1b - own_history": []}
    point = {k: tp[k].sum() / al[k].sum() for k in flags}
    for _ in range(1000):
        i = rng.integers(0, K, K); pr = {k: tp[k][i].sum() / max(al[k][i].sum(), 1) for k in flags}
        for k in flags:
            bs[k].append(pr[k])
        diffs["b1b - sim_proxy_clean"].append(pr["b1b_clean"] - pr["sim_proxy_clean"]); diffs["b1b - own_history"].append(pr["b1b_clean"] - pr["own_history"])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out["ensemble_block"] = {k: dict(point=float(point[k]), ci=ci(bs[k])) for k in flags}
    out["ensemble_block_diffs"] = {k: dict(point=float(point[k.split(" - ")[0].replace("b1b", "b1b_clean")] - point[k.split(" - ")[1]]), ci=ci(v))
                                   for k, v in diffs.items()}
    out["per_snapshot_b1b_ensemble_precision"] = {str(su[k]): float(tp["b1b_clean"][k] / max(al["b1b_clean"][k], 1)) for k in range(K)}
    # ---- verdict vs B1a clean (QUOTED band; B1a predictions absent on this machine)
    b = out["arms"]["b1b_clean"]["op_precision"]
    verdict = "GAIN" if b[0] > B1A_CLEAN_BAND[1] else "WORSE" if b[2] < B1A_CLEAN_BAND[0] else "TIE"
    out["verdict_vs_b1a_clean"] = dict(b1b_band=b, b1a_clean_band_QUOTED=list(B1A_CLEAN_BAND), verdict=verdict,
                                       basis="5-seed bands (B1a clean predictions absent here: no paired block interval)")
    out["training"] = {str(s): {k: logs[s][k] for k in ("epochs_run", "best_epoch", "best_val", "stop", "per_seed_minutes", "sec_per_epoch", "device")}
                       for s in SEEDS}
    out["any_seed_capped"] = any(str(logs[s]["stop"]).startswith("CAP") for s in SEEDS)
    print(json.dumps({k: v for k, v in out.items() if k not in ("stamp",)}, indent=1, default=str)[:6000])
    C.dump(out, "phase24/b1b_score.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rows", "handshake", "neural", "score", "g7"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only")
    a = ap.parse_args()
    if a.mode == "rows":
        build_rows()
    elif a.mode == "handshake":
        handshake()
    elif a.mode == "neural":
        run_neural(a.seed, a.max_epochs)
    elif a.mode == "g7":
        print(g7_failing_case())
    else:
        score()
