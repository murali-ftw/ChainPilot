"""Phase 23B Stage 2 -- predict-the-rescue (Phase 17 B1a) re-scored on clean inputs; world-2 replication; controls.

ROWS. World 1 (v8): the stored Phase 17 rows (ml/artifacts/phase17/b1_rows.npz), re-gated against the stored phase15_sim rows
(G2: part-plant, week and label equal; constructed failing case: one label flipped must fire). World 2 (v8w1002): the same
construction as ml/train/phase17_b1.build_rows (part-plant universe x the 13 weeks after each snapshot, store week = t0 + 7(w + 1),
kept where the part-plant-week exists in inventory_position_weekly -- an EVALUATION row filter, as in Phase 17), its own CSVs.

ARMS (LightGBM, phase7_fit.GBM frozen, seeds 7 17 27 37 47, the Phase 17 feature builder phase7_fit.World.part_plant_features
plus the week offset, early stopping on validation exactly as phase17_b1.run_lgbm):
  published  World("v8")        -- the stored, leaky cache. G3: reproduces the stored b1a_lgbm_s{s}.npz (bit-exactly if possible)
  clean      World("v8clean")   -- the nine LEAKING columns replaced by Phase 22's as-of-safe columns (ml/data/clean_panel.py)
  nl         World("v8")        -- the 18 features derived from the nine LEAKING columns removed
  perm       clean features, TRAINING labels permuted uniformly across all training rows (so across snapshots); G4 control
  w2clean    World("v8w1002clean") on world 2's own rows
REFERENCE ROWS (not arms): the Phase 14/15 simulation proxy (world 1 only: phase15_sim p, per fill seed) and the as-of trailing
52-week transfer-in frequency (own history; transactions RECORDED <= t0, asserted), per world.

The clean LightGBM models (world 1) are saved as loadable bundles for ml/serve/rescue.py (Stage 4): booster text at its best
iteration + identity (GBM config hash, commit, feature list, cache meta).

  python ml/eval/phase23b_rescue.py fit          (torch-free process)
  python ml/eval/phase23b_rescue.py score        (Phase 15 / 20 machinery)
"""
from __future__ import annotations
import os, sys, json, time, hashlib, argparse
import phase12_common as C
import numpy as np, pandas as pd

OUT = os.path.join(C.ART, "phase23b")
PRED = os.path.join(OUT, "preds")
BUN = os.path.join(OUT, "bundles")
LOGS = os.path.join(OUT, "logs")
ROWS1 = os.path.join(C.ART, "phase17", "b1_rows.npz")
ROWS2 = os.path.join(OUT, "b1_rows_v8w1002.npz")
SIMD = os.path.join(C.ART, "phase15_sim")
SEEDS = (7, 17, 27, 37, 47)
W = 13
FOLDS = {"train": ("2019-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")}
FOLD_ID = {"train": 0, "val": 1, "test": 2}
LEAKING = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52", "lead_time_actual_days", "lead_time_ratio",
           "otd_rate_last13", "ack_gap_ratio", "load_ratio"]          # == ml/data/clean_panel.LEAKING (asserted in fit)
ARMS = {"published": ("v8", "v8"), "clean": ("v8clean", "v8"), "nl": ("v8", "v8"), "perm": ("v8clean", "v8"),
        "w2clean": ("v8w1002clean", "v8w1002")}                      # arm -> (feature cache world, row world)


def register():
    import config, phase21_paths as PP
    PP.register()
    for p in ("v8", "v8w1002"):
        config.WORLDS[p + "clean"] = config.WORLDS[p]; config.EXPECTED_PANEL_D[p + "clean"] = 15
    return config


def leaking_feature(f):
    return any(f in (c + "_mean", c + "_min") for c in LEAKING)


# ================================================================== rows
def gate_rows_world1():
    """G2: the stored Phase 17 rows still equal the stored phase15_sim rows; one flipped label must fire."""
    z = np.load(ROWS1)
    res = {}
    for fold in ("val", "test"):
        s = np.load(os.path.join(SIMD, f"{fold}_s7.npz")); m = z["fold"] == FOLD_ID[fold]
        res[fold] = bool(np.array_equal(z["pp"][m], s["pp"]) and np.array_equal(z["week"][m], s["week"]) and np.array_equal(z["y"][m], s["acted"]))
    y = z["y"][z["fold"] == 2].copy(); y[0] = ~y[0]
    s = np.load(os.path.join(SIMD, "test_s7.npz"))
    res["constructed_failing_case_fires"] = not np.array_equal(y, s["acted"])
    assert res["val"] and res["test"] and res["constructed_failing_case_fires"], f"G2 failed: {res}"
    return res


def build_rows(world):
    """phase17_b1.build_rows, parametrised by world (no stored proxy exists for world 2, so no proxy gate)."""
    import config, montecarlo as MC
    D = config.WORLDS[world]
    pp, _, cov = MC.part_plant_universe(world)
    sub = pp.reset_index(drop=True); parts, plant = sub.part_id.to_numpy(), sub.plant_id.to_numpy(); P = len(sub)
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts"])
    tx = tx[tx.txn_type.isin(["transfer_in", "transfer_out"])]
    tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
    g = tx.groupby(["part_id", "plant_id", "week", "txn_type"]).qty.sum().unstack(fill_value=0)
    acted = g["transfer_in"] > 0
    ipw = pd.read_csv(f"{D}/inventory_position_weekly.csv", usecols=["part_id", "plant_id", "week_start"])   # EVALUATION row filter
    have = pd.MultiIndex.from_arrays([ipw.part_id, ipw.plant_id, pd.to_datetime(ipw.week_start)])
    snaps = sorted(pd.Timestamp(x) for x in pd.read_csv(f"{D}/snapshots.csv", usecols=["as_of_ts"]).as_of_ts)
    A = {k: [] for k in ("snap", "pp", "w", "week", "y", "fold")}
    for fold, (lo, hi) in FOLDS.items():
        for t0 in [s for s in snaps if pd.Timestamp(lo) <= s <= pd.Timestamp(hi)]:
            weeks = [t0 + pd.Timedelta(days=7 * (w + 1)) for w in range(W)]
            weeks = [x - pd.Timedelta(days=x.weekday()) for x in weeks]
            idx = pd.MultiIndex.from_arrays([np.repeat(parts, W), np.repeat(plant, W), np.tile(weeks, P)])
            ok = idx.isin(have); y = acted.reindex(idx).fillna(False).to_numpy(bool)
            A["snap"].append(np.full(int(ok.sum()), np.datetime64(t0.date()), "datetime64[D]"))
            A["pp"].append(np.repeat(np.arange(P), W)[ok].astype(np.int32)); A["w"].append(np.tile(np.arange(W), P)[ok].astype(np.int8))
            A["week"].append(np.tile(np.array(weeks, "datetime64[D]"), P)[ok]); A["y"].append(y[ok])
            A["fold"].append(np.full(int(ok.sum()), FOLD_ID[fold], np.int8))
    R = {k: np.concatenate(v) for k, v in A.items()}
    np.savez_compressed(ROWS2, **R, parts=parts.astype(str), plants=plant.astype(str), n_channels=sub.n_channels.to_numpy(np.int32))
    return dict(rows={f: int((R["fold"] == i).sum()) for f, i in FOLD_ID.items()},
                base_rate={f: float(R["y"][R["fold"] == i].mean()) for f, i in FOLD_ID.items()},
                snapshots={f: int(len(np.unique(R["snap"][R["fold"] == i]))) for f, i in FOLD_ID.items()}, universe=cov["part_plants"])


def load_rows(world):
    z = np.load(ROWS1 if world == "v8" else ROWS2)
    return {k: z[k] for k in z.files}


# ================================================================== features (phase7_fit.World, unchanged)
def features(P7, cache_world, R):
    Wd = P7.World(cache_world)
    keys = np.char.add(np.char.add(R["parts"], "|"), R["plants"])
    pairs = pd.DataFrame({"snap": R["snap"], "pp": R["pp"]}).drop_duplicates().reset_index(drop=True)
    known = set(Wd.pp_keys); pk = keys[pairs.pp.to_numpy()]; has = np.array([k in known for k in pk])
    lb = pd.DataFrame({"key": pk[has], "snapshot_date": pd.to_datetime(pairs.snap.to_numpy()[has])})
    Xk = Wd.part_plant_features(lb); cols = list(Xk.columns)
    F = np.full((len(pairs), len(cols)), np.nan, np.float32); F[has] = Xk.to_numpy(np.float32)
    pos = pd.MultiIndex.from_frame(pairs).get_indexer(pd.MultiIndex.from_arrays([R["snap"], R["pp"]])); assert (pos >= 0).all()
    X = np.concatenate([F[pos], R["w"][:, None].astype(np.float32)], 1)
    return X, cols + ["week_offset"], dict(cache_world=cache_world, cache_meta_world=Wd.meta.get("world"), clean_of=Wd.meta.get("clean_of"),
                                         replaced_columns=Wd.meta.get("replaced_columns"), orphan_pairs=int((~has).sum()))


def gbm_hash(GBM):
    return hashlib.sha1(json.dumps(GBM, sort_keys=True).encode()).hexdigest()[:12]


def fit():
    import lightgbm as lgb                                                    # FIRST, as in phase7_fit
    sys.path.insert(0, os.path.join(C.ML, "baselines"))
    import phase7_fit as P7
    from clean_panel import LEAKING as LK
    assert list(LK) == LEAKING, "LEAKING differs from ml/data/clean_panel.LEAKING"
    register()
    st = C.require_clean()
    for d in (OUT, PRED, BUN, LOGS):
        os.makedirs(d, exist_ok=True)
    rec = dict(stamp=st, gate_rows_world1=gate_rows_world1(), lightgbm=lgb.__version__, gbm=P7.GBM, gbm_hash=gbm_hash(P7.GBM))
    t = time.time(); rec["rows_world2"] = build_rows("v8w1002"); rec["rows_world2"]["minutes"] = round((time.time() - t) / 60, 1)
    print("world-2 rows:", rec["rows_world2"], flush=True)
    cache = {}
    for arm, (fw, rw) in ARMS.items():
        t_arm = time.time()
        R = load_rows(rw); key = (fw, rw)
        if key not in cache:
            cache.clear(); cache[key] = features(P7, fw, R)
        X, names, fmeta = cache[key]
        cols = list(range(len(names)))
        if arm == "nl":
            cols = [i for i, n in enumerate(names) if not leaking_feature(n)]
        Xa, na = X[:, cols], [names[i] for i in cols]
        if arm in ("clean", "w2clean", "perm"):
            assert fmeta["clean_of"] and fmeta["replaced_columns"] == LEAKING, f"{arm}: not a clean cache: {fmeta}"
        y = R["y"].astype(int); tr, va, te = (R["fold"] == 0), (R["fold"] == 1), (R["fold"] == 2)
        log = dict(arm=arm, world=rw, features=na, n_features=len(na), feature_meta=fmeta, base_rate=dict(val=float(y[va].mean()), test=float(y[te].mean())),
                   concurrency="1 CPU job", seeds={})
        for s in SEEDS:
            ytr = y.copy()
            if arm == "perm":
                rng = np.random.default_rng(1000 + s); i = np.flatnonzero(tr); ytr[i] = y[i][rng.permutation(len(i))]
            ts = time.time()
            m = lgb.LGBMClassifier(**{**P7.GBM, "random_state": s, "objective": "binary"})
            m.fit(Xa[tr], ytr[tr], eval_set=[(Xa[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
            pv, pt = m.predict_proba(Xa[va])[:, 1], m.predict_proba(Xa[te])[:, 1]
            np.savez_compressed(os.path.join(PRED, f"{rw}_{arm}_s{s}.npz"), Pv=pv.astype(np.float32), Pt=pt.astype(np.float32))
            log["seeds"][s] = dict(best_iter=int(m.best_iteration_ or 400), seconds=round(time.time() - ts, 1))
            if arm == "clean":
                save_bundle(m, s, na, fmeta, st, P7.GBM, R, Xa, pt)
            print(f"  {arm} s{s}: best_iter {log['seeds'][s]['best_iter']} ({log['seeds'][s]['seconds']}s)", flush=True)
        log["minutes"] = round((time.time() - t_arm) / 60, 1)
        json.dump(log, open(os.path.join(LOGS, f"fit_{rw}_{arm}.json"), "w"), indent=1)          # per-(task, arm) log
    rec["arms"] = list(ARMS)
    C.dump(rec, "phase23b/stage2_fit.json")


def save_bundle(m, s, names, fmeta, st, GBM, R, Xa, pt):
    d = os.path.join(BUN, f"rescue_lgbm_v8clean_s{s}"); os.makedirs(d, exist_ok=True)
    assert not os.path.exists(os.path.join(d, "identity.json")), f"refusing to overwrite a stored bundle: {d}"
    m.booster_.save_model(os.path.join(d, "model.txt"), num_iteration=m.best_iteration_)
    te = np.flatnonzero(R["fold"] == 2)
    pick = te[np.linspace(0, len(te) - 1, 200).astype(int)]          # 200 test rows for the serving test
    ident = dict(task="rescue_week", model="lightgbm_b1a_clean", seed=s, world="v8clean", gbm_config=GBM, gbm_config_hash=gbm_hash(GBM),
                 commit=st["code_commit"], feature_list=names, n_features=len(names), best_iteration=int(m.best_iteration_ or 400),
                 cache_meta=fmeta, leaking_columns_excluded_in_generator_form=LEAKING, created_by="ml/eval/phase23b_rescue.py fit")
    ident["identity_hash"] = hashlib.sha1(json.dumps({k: ident[k] for k in ("feature_list", "gbm_config_hash", "seed", "world")},
                                                     sort_keys=True).encode()).hexdigest()[:16]
    json.dump(ident, open(os.path.join(d, "identity.json"), "w"), indent=1)
    rows = dict(row=pick.astype(np.int64), snap=R["snap"][pick], part=R["parts"][R["pp"][pick]], plant=R["plants"][R["pp"][pick]],
                w=R["w"][pick], X=Xa[pick], P_stored=pt[np.searchsorted(te, pick)].astype(np.float32))
    np.savez_compressed(os.path.join(d, "test_rows_200.npz"), **rows)


# ================================================================== scoring
def require_five(pairs, name):
    """G5 (Phase 19 deviation 171): no arm is scored without all five seeds."""
    if len(pairs) != len(SEEDS):
        raise AssertionError(f"{name}: {len(pairs)} of {len(SEEDS)} seeds -- refusing to score")
    return pairs


def naive_rate(world, R, yv, yt):
    import config
    D = config.WORLDS[world]
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "event_ts", "recorded_ts"])
    tx = tx[tx.txn_type == "transfer_in"]
    tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time; tx["rec"] = pd.to_datetime(tx.recorded_ts)
    keymap = {(p, q): i for i, (p, q) in enumerate(zip(R["parts"], R["plants"]))}
    tx["pp"] = [keymap.get(k, -1) for k in zip(tx.part_id, tx.plant_id)]; tx = tx[tx.pp >= 0]
    score = np.zeros(len(R["y"]))
    for s in np.unique(R["snap"][R["fold"] > 0]):
        t0 = pd.Timestamp(s); u = tx[(tx.rec <= t0) & (tx.week > t0 - pd.Timedelta(weeks=52))]
        assert (u.rec <= t0).all(), "as-of violation in the own-history baseline"
        rate = u.groupby("pp").week.nunique().reindex(range(len(R["parts"])), fill_value=0).to_numpy() / 52.0
        m = R["snap"] == s; score[m] = rate[R["pp"][m]]
    return score[R["fold"] == 1], score[R["fold"] == 2]


def op_point(sv, yv, st, yt, r=0.20):
    """Phase 17's operating point: VALIDATION, the highest-precision threshold with validation recall >= r; applied to TEST."""
    import phase15 as P15
    ss, yy, tp, n = P15.sorted_cum(sv, yv)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    rec, prec = tp[last] / yy.sum(), tp[last] / n[last]
    ok = rec >= r; i = last[np.flatnonzero(ok)[np.argmax(prec[ok])]]
    a = P15.apply(st, yt, float(ss[i]))
    return dict(tau=float(ss[i]), val_precision=float(tp[i] / n[i]), val_recall=float(tp[i] / yy.sum()),
                test_precision=a["precision"], test_recall=a["recall"], test_coverage=a["coverage"])


def prec_cov(s, y, c):
    m = max(1, int(np.ceil(c * len(y)))); return float(y[np.argsort(-s, kind="stable")[:m]].mean())


def block_ci(fn, snaps, B=1000, seed=0):
    blocks = [np.flatnonzero(snaps == u) for u in np.unique(snaps)]
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        vals.append(fn(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])))
    v = np.array(vals); return [float(np.percentile(v, 2.5)), float(np.mean(v)), float(np.percentile(v, 97.5))]


def band(v):
    v = [float(x) for x in v]; return [min(v), float(np.mean(v)), max(v)]


def reachable_and_class_unchanged():
    """phase20_decisions.reachable_and_class, its SOURCE TEXT taken from the committed file and executed unchanged. The module
    itself cannot be imported here: at import it loads ml/artifacts/phase19/stage4b_block.json, which exists only on the Mac."""
    import ast, phase15 as P15
    path = os.path.join(C.ML, "eval", "phase20_decisions.py"); src = open(path, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "reachable_and_class")
    ns = dict(PS=P15.PS, MIN_ALERTS=P15.MIN_ALERTS)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns["reachable_and_class"], hashlib.sha1(ast.get_source_segment(src, fn).encode()).hexdigest()[:12]


def score():
    import phase15 as P15
    reachable_and_class, rc_sha = reachable_and_class_unchanged()
    register()
    st = C.require_clean()
    out = dict(stamp=st, machine="Windows 11, CPU only", concurrency="1 CPU job", worlds={},
               class_rule="phase20_decisions.reachable_and_class, source executed unchanged (sha1 of source %s)" % rc_sha)
    # G5's constructed failing case
    try:
        require_five([None] * 4, "constructed: one seed dropped"); g5 = False
    except AssertionError:
        g5 = True
    out["G5_constructed_failing_case_fires"] = g5
    assert g5
    for world, arms in (("v8", ["published", "clean", "nl", "perm"]), ("v8w1002", ["w2clean"])):
        R = load_rows(world); yv, yt = R["y"][R["fold"] == 1].astype(int), R["y"][R["fold"] == 2].astype(int)
        snap_t = R["snap"][R["fold"] == 2]
        W_ = dict(base_rate=dict(val=float(yv.mean()), test=float(yt.mean())), rows=dict(val=int(len(yv)), test=int(len(yt))),
                  test_snapshots=int(len(np.unique(snap_t))), arms={})
        refs = {}
        nv, nt = naive_rate(world, R, yv, yt); refs["own_history"] = [(nv, yv, nt, yt)]
        if world == "v8":
            pr = []
            for s in SEEDS:
                zv, zt = (np.load(os.path.join(SIMD, f"{f}_s{s}.npz")) for f in ("val", "test"))
                assert np.array_equal(zt["acted"].astype(int), yt)
                pr.append((zv["p"], yv, zt["p"], yt))
            refs["sim_proxy"] = require_five(pr, "sim_proxy")
        stored = None
        if world == "v8":
            stored = []
            for s in SEEDS:
                z = np.load(os.path.join(C.ART, "phase17", f"b1a_lgbm_s{s}.npz")); stored.append((z["Pv"], yv, z["Pt"], yt))
        ens = {}
        for name, pairs in list(refs.items()) + [(a, None) for a in arms]:
            if pairs is None:
                pairs = require_five([(np.load(os.path.join(PRED, f"{world}_{name}_s{s}.npz"))["Pv"], yv,
                                       np.load(os.path.join(PRED, f"{world}_{name}_s{s}.npz"))["Pt"], yt) for s in SEEDS], name)
            ops = [op_point(*p) for p in pairs]
            an = P15.analyse([p[:4] for p in pairs], deterministic=len(pairs) == 1)
            cls = reachable_and_class(an)
            sv = np.mean([p[0] for p in pairs], 0); stt = np.mean([p[2] for p in pairs], 0); ens[name] = (sv, stt)
            o_e = op_point(sv, yv, stt, yt); tau = o_e["tau"]
            per_snap = {str(pd.Timestamp(u).date()): float(yt[(snap_t == u) & (stt >= tau)].mean()) if ((snap_t == u) & (stt >= tau)).any() else None
                        for u in np.unique(snap_t)}
            ps = [v for v in per_snap.values() if v is not None]
            W_["arms"][name] = dict(
                n_seeds=len(pairs), op_test_precision=band([o["test_precision"] for o in ops]), op_test_recall=band([o["test_recall"] for o in ops]),
                op_val_precision=band([o["val_precision"] for o in ops]),
                coverage={f"{int(c * 100)}%": band([prec_cov(p[2], yt, c) for p in pairs]) for c in (0.01, 0.05, 0.10, 0.20)},
                lift_at_op=band([o["test_precision"] / yt.mean() for o in ops]),
                ensemble=dict(op=o_e, op_precision_block_ci=block_ci(lambda i: float(yt[i][stt[i] >= tau].mean()) if (stt[i] >= tau).any() else np.nan, snap_t)),
                per_snapshot_precision_at_op=dict(values=per_snap, min=min(ps), median=float(np.median(ps)), max=max(ps)),
                phase15=cls, stage_b=an["stage_b"]["verdict"])
            print(f"{world} {name}: op precision {W_['arms'][name]['op_test_precision']} recall {W_['arms'][name]['op_test_recall']} class {cls['cls']}", flush=True)
        # paired block intervals of ensemble differences at each arm's own validation-chosen threshold
        def diff(a, b):
            (sva, sta), (svb, stb) = ens[a], ens[b]
            ta, tb = op_point(sva, yv, sta, yt)["tau"], op_point(svb, yv, stb, yt)["tau"]
            f = lambda i: (float(yt[i][sta[i] >= ta].mean()) - float(yt[i][stb[i] >= tb].mean()))
            ci = block_ci(f, snap_t); return dict(diff_block_ci=ci, verdict="better" if ci[0] > 0 else "worse" if ci[2] < 0 else "undetermined")
        comps = {}
        if world == "v8":
            for a, b in (("clean", "published"), ("nl", "published"), ("clean", "nl"), ("clean", "sim_proxy"), ("clean", "own_history"),
                         ("published", "perm"), ("clean", "perm")):
                comps[f"{a} - {b}"] = diff(a, b)
            pub_rep = [bool(np.array_equal(np.load(os.path.join(PRED, f"v8_published_s{s}.npz"))["Pt"], np.load(os.path.join(C.ART, "phase17", f"b1a_lgbm_s{s}.npz"))["Pt"]))
                       for s in SEEDS]
            pub_close = [float(np.abs(np.load(os.path.join(PRED, f"v8_published_s{s}.npz"))["Pt"].astype(float) -
                                      np.load(os.path.join(C.ART, "phase17", f"b1a_lgbm_s{s}.npz"))["Pt"].astype(float)).max()) for s in SEEDS]
            W_["G3_published_reproduction"] = dict(bit_exact_per_seed=pub_rep, max_abs_diff_per_seed=pub_close)
            perm_op = W_["arms"]["perm"]["op_test_precision"][1]
            W_["G4_permuted_control"] = dict(perm_op_precision=perm_op, base_rate=float(yt.mean()), collapses=bool(abs(perm_op - yt.mean()) <= 0.03),
                                             published_beats_by=W_["arms"]["published"]["op_test_precision"][1] - perm_op,
                                             constructed_failing_case_fires=bool(W_["arms"]["published"]["op_test_precision"][1] - perm_op > 0.10))
        else:
            comps["w2clean - own_history"] = diff("w2clean", "own_history")
        W_["comparisons"] = comps
        out["worlds"][world] = W_
    C.dump(out, "phase23b/stage2_score.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "score"]); a = ap.parse_args()
    fit() if a.mode == "fit" else score()
