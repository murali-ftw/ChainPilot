"""Phase 22 -- LightGBM arms: Stage 1d restatement, Stage 2 order-time arms, Stage 3a fill families. ONE torch-free process.

Every arm is Phase 7's guide-B5 fit, unchanged (frozen GBM config, same labels / rows / objective / early stopping on
validation), on a WORLD: v8 | v8clean | v8w1002 | v8w1002clean (clean = the leaking panel columns replaced,
ml/data/clean_panel.py). The world is in every file name, so a clean arm can never overwrite a stored one.

Snapshot tasks (arrival L2 on observed rows, fill 22-class, capacity quantile 0.1 / 0.5 / 0.9):
  base        the flat as-of features of the world (on v8clean this is BASE_clean)
  nl          (v8 / v8w1002 only) BASE with the nine LEAKING columns removed
  lag1        DIAGNOSTIC (D1 week-t0 convention): BASE_clean read at panel row t0 - 1 (strictly before the snapshot date)
  fwdload     + Phase 18 fwd_load (the LightGBM half of the clean Phase 19 blend recipe)
  fill families (D7): sc (season + cadence), ack, L4, sc_ack, sc_ack_L4 (the consolidated arm)
  controls: sc_ctrl (season snapshot-permuted within split AND cadence shuffled across channels within snapshot),
            ack_sperm (same channel, donor snapshot), ack_xsh (another channel, same snapshot), L4_xsh
Order-time task `place` (Phase 21 at-placement rows; tau = the Monday on or before creation):
  base, base_lag1 (row tau - 1), L4 (grpstats place store, k = 10), L4_xsh, latedays (regression of lead - contract in
  days on base + L4), flag (binary late vs contract, censoring resolved, on base + L4), leaked (base on the PUBLISHED v8 panel
  + L4: the constructed failing case the leak check must flag)
Every arm's feature list goes through `leak_check` (any LEAKING column on a non-clean world is flagged); the leaked arm
MUST be flagged and no clean arm may be.
--persist saves the fitted booster(s) (LightGBM text format) with an identity JSON under ml/artifacts/phase22/models/.

Output: ml/artifacts/phase22/preds/{world}_{task}_p22_{arm}[_k{k}]_s{seed}_{fold}.npz; log per (world, task, arm):
ml/artifacts/phase22/logs/{world}_{task}_{arm}.json (deviation 198).

  python ml/baselines/phase22_proxy.py --world v8clean --task fill --arms base,sc,ack,L4,sc_ack,sc_ack_L4
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST -- before anything that could pull in torch
import os, sys, json, time, argparse, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import phase7_fit as P7
import folds as FO
import grpstats as GS
import phase20_proxy as P20
import phase22_rows as R22
import clean_panel as CPN
import phase12_common as C
from config import ARTIFACTS
assert "torch" not in sys.modules, "torch must not be imported in the LightGBM process"

OUT = os.path.join(ARTIFACTS, "phase22", "preds")
LOGS = os.path.join(ARTIFACTS, "phase22", "logs")
MODELS = os.path.join(ARTIFACTS, "phase22", "models")
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain", "place": "arrival_place"}
SNAP_ARMS = ("base", "nl", "lag1", "fwdload", "sc_L5", "sc", "ack", "L4", "sc_ack", "sc_ack_L4", "sc_ctrl", "ack_sperm", "ack_xsh", "L4_xsh")
PLACE_ARMS = ("base", "base_lag1", "L4", "L4_xsh", "latedays", "flag", "leaked")
K_PLACE = 10


def register():
    PP.register()
    for parent in ("v8", "v8w1002"):
        config.WORLDS[parent + "clean"] = config.WORLDS[parent]; config.EXPECTED_PANEL_D[parent + "clean"] = 15


def parent(world):
    return world.replace("clean", "")


def leak_check(columns, world):
    """-> the LEAKING columns this feature list reads from a world whose panel still carries them."""
    if world.endswith("clean"):
        return []
    return [c for c in columns if c in CPN.LEAKING]


def lag_lb(lb, days):
    x = lb.copy(); x["snapshot_date"] = pd.to_datetime(x.snapshot_date) - pd.Timedelta(days=days)
    return x


def snap_family(arm, world, lb, seed):
    pw = parent(world)
    if arm == "fwdload":
        return P20.block("fwd_load", pw, lb, seed)
    if arm == "sc_L5":                      # Phase 21's hybrid LightGBM half: season + cadence + the L5 group block (k = 300)
        Z, _ = GS.load(pw, "snap")
        assert (lb.entity_id.to_numpy().astype(str) == Z["entity"]).all()
        return pd.concat([R22.block("season", pw, lb, seed), R22.block("cadence", pw, lb, seed), GS.assemble_arrival(Z, 300, "L5")], axis=1)
    parts = {"sc": ["season", "cadence"], "ack": ["ack"], "L4": ["L4"], "sc_ack": ["season", "cadence", "ack"],
             "sc_ack_L4": ["season", "cadence", "ack", "L4"]}
    if arm in parts:
        return pd.concat([R22.block(f, pw, lb, seed) for f in parts[arm]], axis=1)
    if arm == "sc_ctrl":
        cad = R22.block("cadence", pw, lb, seed)
        rng = np.random.default_rng(40_000 + seed)
        perm = np.arange(len(lb))
        for g in np.unique(lb.snapshot_date.to_numpy()):
            idx = np.flatnonzero(lb.snapshot_date.to_numpy() == g); perm[idx] = idx[rng.permutation(len(idx))]
        return pd.concat([R22.block("season", pw, lb, seed, "sperm"), cad.iloc[perm].reset_index(drop=True)], axis=1)
    if arm == "ack_sperm":
        return R22.block("ack", pw, lb, seed, "sperm")
    if arm == "ack_xsh":
        return R22.block("ack", pw, lb, seed, "xsh")
    if arm == "L4_xsh":
        return R22.block("L4", pw, lb, seed, "xsh")
    raise ValueError(arm)


def placement(world):
    pw = parent(world)
    Z, _ = GS.load(pw, "place")
    src = GS.Source(pw)
    li = Z["line"].astype(np.int64)
    assert (src.line_id[li].astype(str) == Z["entity"]).all()
    Y, EV, cw, qty = GS.placement_labels(src, li)
    lb = pd.DataFrame({"snapshot_date": pd.to_datetime(Z["tau"]), "entity_id": src.line_id[li],
                       "key": src.ch.channel_id.to_numpy()[src.chan[li]], "label_value": Y, "label_censored": ~EV,
                       "promise_week": cw, "created": pd.DatetimeIndex(src.created[li])})
    return lb, Z, np.log1p(qty)


def save_model(world, task, name, models, cols, meta):
    os.makedirs(MODELS, exist_ok=True)
    paths = []
    for i, m in enumerate(models):
        p = os.path.join(MODELS, f"{world}_{task}_{name}_m{i}.txt")
        assert not os.path.exists(p), f"refusing to overwrite a stored model: {p}"
        m.booster_.save_model(p); paths.append(p)
    ident = dict(world=world, task=task, name=name, features=list(cols), models=[os.path.basename(p) for p in paths],
                 sha1=[hashlib.sha1(open(p, "rb").read()).hexdigest() for p in paths], **meta)
    json.dump(ident, open(os.path.join(MODELS, f"{world}_{task}_{name}.identity.json"), "w"), indent=1)


def fit(world, task, arms, seeds, stamp, persist):
    t_task = TASK[task]
    Wd = P7.World(world)
    if task == "place":
        lb, Z, lq = placement(world)
        tr, va, te = [np.asarray(x, bool) for x in FO.fixed_split(lb.created)]
    else:
        lb = P7.labels(world, t_task)
        tr, va, te = [np.asarray(x, bool) for x in FO.fixed_split(lb.snapshot_date)]
    for arm in arms:
        log_path = os.path.join(LOGS, f"{world}_{task}_{arm}.json")
        log = json.load(open(log_path)) if os.path.exists(log_path) else {}
        log.setdefault("_stamps", []).append(dict(world=world, task=task, arm=arm, **stamp))
        for s in seeds:
            k = K_PLACE if task == "place" and arm in ("L4", "L4_xsh", "latedays", "flag", "leaked") else None
            name = f"p22_{arm}" + (f"_k{k}" if k else "") + f"_s{s}"
            if all(os.path.exists(os.path.join(OUT, f"{world}_{t_task}_{name}_{f}.npz")) for f in ("val", "test")) and not persist:
                print(f"  {world} {task} {name}: exists, skipped", flush=True); continue
            # ---- features
            if task == "place":
                fw = "v8" if arm == "leaked" else world
                Wf = Wd if fw == world else P7.World(fw)
                lbf = lag_lb(lb, 7) if arm == "base_lag1" else lb
                X = pd.concat([Wf.channel_features(lbf, with_ids=False, flat=True),
                               pd.DataFrame({"log1p_qty_ordered": lq.astype(np.float32)})], axis=1)
                if arm in ("L4", "latedays", "flag", "leaked"):
                    X = pd.concat([X, GS.assemble_arrival(Z, K_PLACE, "L4")], axis=1)
                elif arm == "L4_xsh":
                    F = GS.assemble_arrival(Z, K_PLACE, "L4")
                    from phase21_proxy import derange_within
                    X = pd.concat([X, F.iloc[derange_within(lb.snapshot_date.to_numpy(), 80_000 + s)].reset_index(drop=True)], axis=1)
                flagged = leak_check(X.columns, fw)
            else:
                lbf = lag_lb(lb, 7) if arm == "lag1" else lb
                X = Wd.channel_features(lbf, with_ids=False, flat=True)
                if arm == "nl":
                    assert not world.endswith("clean")
                    X = X.drop(columns=CPN.LEAKING)
                elif arm not in ("base", "lag1"):
                    X = pd.concat([X, snap_family(arm, world, lb, s)], axis=1)
                flagged = leak_check(X.columns, world) if arm != "nl" else []
            X = X.astype(np.float32)
            meta = dict(seed=s, arm=arm, world=world, n_features=X.shape[1], commit=stamp["code_commit"], leak_flagged=flagged)
            t = time.time()
            if task in ("arrival", "place") and arm != "flag":
                y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
                if arm == "latedays":
                    y = 7 * (y - lb.promise_week.to_numpy(float))          # lateness in days vs contract
                m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
                if arm == "latedays":
                    pw_ = lb.promise_week.to_numpy(float)
                    pred = lambda o: (pw_[o] + m.predict(X.iloc[o]) / 7.0, None)     # back to weeks of lead
                else:
                    pred = lambda o: (m.predict(X.iloc[o]), None)
                P7.emit(world, t_task, name, lb, va, te, pred, log, dict(best_iter=int(m.best_iteration_ or 400), **meta))
                models = [m]
            elif arm == "flag":
                Y = lb.label_value.to_numpy(float); EV = ~lb.label_censored.to_numpy(bool); R = lb.promise_week.to_numpy(float)
                late = (Y > R).astype(int); keep = EV | (Y > R)                    # UC1-P label, censoring resolved
                m = P7.lgbm_fit("binary", X, late, tr, va, s, rows_tr=tr & keep, rows_va=va & keep)
                P7.emit(world, t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o])[:, 1], None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), **meta))
                models = [m]
            elif task == "fill":
                m = P7.lgbm_fit("multi", X, P7.fill_cell(lb.label_value.to_numpy(float)), tr, va, s, K=22)
                P7.emit(world, t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), **meta))
                models = [m]
            else:
                y = lb.label_value.to_numpy(float)
                ms = [P7.lgbm_fit("quantile", X, y, tr, va, s, alpha=a_) for a_ in P7.QS]
                P7.emit(world, t_task, name, lb, va, te, lambda o: (np.stack([mm.predict(X.iloc[o]) for mm in ms], 1), None), log,
                        dict(best_iters=[int(mm.best_iteration_ or 400) for mm in ms], **meta))
                models = ms
            log[f"{world}|{t_task}|{name}"]["seconds_fit"] = time.time() - t
            log[f"{world}|{t_task}|{name}"]["gain_importance"] = {c: float(v) for c, v in zip(X.columns, models[0].booster_.feature_importance("gain"))}
            if persist:
                save_model(world, t_task, name, models, X.columns, dict(seed=s, arm=arm, commit=stamp["code_commit"]))
            print(f"  {world} {task} {name}: {time.time() - t:.0f}s ({X.shape[1]} features){' LEAK-FLAGGED ' + str(flagged) if flagged else ''}", flush=True)
            json.dump(log, open(log_path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=["v8", "v8clean", "v8w1002", "v8w1002clean"])
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, P7.SEEDS)))
    ap.add_argument("--persist", action="store_true")
    a = ap.parse_args()
    register()
    os.makedirs(OUT, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    P7.OUT = OUT
    st = C.require_clean()
    arms = a.arms.split(",")
    assert all(x in (PLACE_ARMS if a.task == "place" else SNAP_ARMS) for x in arms), arms
    fit(a.world, a.task, arms, [int(x) for x in a.seeds.split(",")], st, a.persist)
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
