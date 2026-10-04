"""Phase 21 Stages 3-5, 7 -- the LightGBM arms with group statistics appended. ONE torch-free process.

Every arm is Phase 7's guide-B5 fit, unchanged (`phase7_fit.lgbm_fit` / `emit`: frozen GBM config, same labels, rows,
flat as-of features, objective and early stopping on validation; arrival L2 on observed rows, fill 22-class). Columns are
APPENDED from ml/data/grpstats.py, shrunk with the k chosen on validation (ml/artifacts/phase21/k_select_{world}.json).

Snapshot arms (task arrival | fill; the Phase 19 / 20 rows):
  base        v8: seed 7 only, must reproduce the stored phase7_preds LightGBM-flat arm bit-exactly (else STOP);
              v8w1002: seed 7, must reproduce Phase 20's stored world-2 BASE bit-exactly
  L4 / L5     + the L4 (no month) / L5 (with the t0-month key) group block
  L4_xsh / L5_xsh   + the same, cross-channel shuffled: within each snapshot each row takes another row's block
              (a derangement, rng 80000 + seed)
  L5_perm     + the L5 block read at a DONOR month: month keys permuted across rows within each split (rng 70000 + seed)
  grp_only    the L5 block ALONE (no BASE columns): the pure form of the idea
  sc / sc_L5  + Phase 19 fwd_season + cadence (built by the unchanged modules), and the same + L5
  ack / ack_xsh / L5_ack   (fill) + the acknowledgement-gap block, its cross-channel shuffle, and L5 + ack
At-placement arms (task place; own rows, ml/data/grpstats.py placement_rows / placement_labels):
  base        flat as-of channel features at tau (the Monday on or before creation) + log1p(qty_ordered)
  L4 / L5 / L5_perm   as above, with the line's TRUE creation month as the key

Output: ml/artifacts/phase21/preds/{world}_{task}_p21_{arm}[_k{k}]_s{seed}_{fold}.npz (phase7_preds format; P, Y, EV, AUX,
entity). k is in the name, so two shrinkage settings can never collide. An existing file is never overwritten.

  python ml/baselines/phase21_proxy.py --world v8 --task arrival --arms base,L4,L5,...
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST -- before anything that could pull in torch
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import phase21_paths as PP
import phase7_fit as P7
import folds as FO
import grpstats as GS
import phase20_proxy as P20
import phase12_common as C
from config import ARTIFACTS
assert "torch" not in sys.modules, "torch must not be imported in the LightGBM process"

OUT = os.path.join(ARTIFACTS, "phase21", "preds")
LOG_FMT = os.path.join(ARTIFACTS, "phase21", "proxy_fit_{world}_{task}.json")   # one log per process (deviation 198)
TASK = {"arrival": "arrival_week", "fill": "fill_rate", "place": "arrival_place"}
# DIAGNOSTIC suffix "_nl" (deviation 199): the arm with the three channel_performance_weekly columns REMOVED from the
# BASE features that the generator builds from each line's EVENTUAL lead, bucketed on the line's ORDER week
# (generator_v8.py: leadw[pch, vw_ord] = pl_, forward-filled) -- future information at any t0 / tau in that week or later.
LEAK = ["lead_time_actual_days", "lead_time_ratio", "otd_rate_last13"]
ARMS = {"arrival": ("base", "L4", "L4_xsh", "L5", "L5_perm", "L5_xsh", "grp_only", "sc", "sc_L5", "base_nl", "L4_nl", "L5_nl"),
        "fill": ("base", "L4", "L4_xsh", "L5", "L5_perm", "L5_xsh", "ack", "ack_xsh", "L5_ack", "sc_L5"),
        "place": ("base", "L4", "L5", "L5_perm", "base_nl", "L4_nl", "L5_nl", "L5_perm_nl")}


def k_of(world, task):
    d = json.load(open(os.path.join(ARTIFACTS, "phase21", f"k_select_{world}.json")))
    return int(d["k"][{"arrival": "arrival", "fill": "fill", "place": "place"}[task]])


def split_masks(dates):
    tr, va, te = FO.fixed_split(dates)
    return [np.asarray(x, bool) for x in (tr, va, te)]


def derange_within(groups, seed):
    """donor[i] = another row of the same group (a cyclic shift of a random order: no fixed point)."""
    rng = np.random.default_rng(seed)
    donor = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < 2:
            continue
        p = idx[rng.permutation(len(idx))]
        donor[p] = np.roll(p, -1)
    return donor


def permuted_month(month, masks, seed):
    rng = np.random.default_rng(70_000 + seed)
    mk = month.astype(np.int64).copy()
    for m in masks:
        idx = np.flatnonzero(m)
        mk[idx] = mk[idx][rng.permutation(len(idx))]
    return mk


def group_block(task, Z, k, arm, month_key=None):
    if task == "fill":
        return GS.assemble_fill(Z, k, arm, month_key)
    return GS.assemble_arrival(Z, k, arm, month_key)


def family(task, arm, world, lb, Z, k, seed, groups, masks):
    """-> (DataFrame of appended columns, use_base: bool)."""
    if arm == "base":
        return None, True
    if arm in ("L4", "L5"):
        return group_block(task, Z, k, arm), True
    if arm in ("L4_xsh", "L5_xsh"):
        F = group_block(task, Z, k, arm[:2])
        return F.iloc[derange_within(groups, 80_000 + seed)].reset_index(drop=True), True
    if arm == "L5_perm":
        return group_block(task, Z, k, "L5", permuted_month(Z["month"], masks, seed)), True
    if arm == "grp_only":
        return group_block(task, Z, k, "L5"), False
    if arm == "sc":
        return pd.concat([P20.block("fwd_season", world, lb, seed), P20.block("cadence", world, lb, seed)], axis=1), True
    if arm == "sc_L5":
        return pd.concat([P20.block("fwd_season", world, lb, seed), P20.block("cadence", world, lb, seed),
                          group_block(task, Z, k, "L5")], axis=1), True
    if arm == "ack":
        return GS.assemble_ack(Z), True
    if arm == "ack_xsh":
        return GS.assemble_ack(Z).iloc[derange_within(groups, 80_000 + seed)].reset_index(drop=True), True
    if arm == "L5_ack":
        return pd.concat([group_block(task, Z, k, "L5"), GS.assemble_ack(Z)], axis=1), True
    raise ValueError(arm)


def placement_lb(world):
    Z, info = GS.load(world, "place")
    src = GS.Source(world)
    li = Z["line"].astype(np.int64)
    assert (src.line_id[li].astype(str) == Z["entity"]).all(), "placement store rows do not match the lines"
    Y, EV, cweeks, qty = GS.placement_labels(src, li)
    lb = pd.DataFrame({"snapshot_date": pd.to_datetime(Z["tau"]), "entity_id": src.line_id[li],
                       "key": src.ch.channel_id.to_numpy()[src.chan[li]], "label_value": Y, "label_censored": ~EV,
                       "promise_week": cweeks, "created": pd.DatetimeIndex(src.created[li])})
    return lb, Z, np.log1p(qty)


def fit(world, task, arms, seeds, log, stamp):
    t_task = TASK[task]
    Wd = P7.World(world)
    if task == "place":
        lb, Z, lq = placement_lb(world)
        masks = split_masks(lb.created)
        Xflat = pd.concat([Wd.channel_features(lb, with_ids=False, flat=True),
                           pd.DataFrame({"log1p_qty_ordered": lq.astype(np.float32)})], axis=1)
    else:
        lb = P7.labels(world, t_task)
        Z, info = GS.load(world, "snap")
        assert (lb.entity_id.to_numpy().astype(str) == Z["entity"]).all() and \
            (lb.snapshot_date.dt.strftime("%Y-%m-%d").to_numpy() == Z["snapshot"]).all(), "group store rows != label rows"
        masks = split_masks(lb.snapshot_date)
        Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
    tr, va, te = masks
    groups = lb.snapshot_date.to_numpy()
    k = k_of(world, task)
    for arm in arms:
        for s in ([7] if arm == "base" and task != "place" else seeds):
            name = f"p21_{arm}_s{s}" if arm in ("base", "base_nl") else f"p21_{arm}_k{k}_s{s}"
            exists = all(os.path.exists(os.path.join(OUT, f"{world}_{t_task}_{name}_{f}.npz")) for f in ("val", "test"))
            if exists and "gain_importance" in log.get(f"{world}|{t_task}|{name}", {}):
                print(f"  {world} {task} {name}: exists, skipped", flush=True); continue
            if exists:          # the record was lost (deviation 198): refit, assert bit-identical, record it
                stored = {f: np.load(os.path.join(OUT, f"{world}_{t_task}_{name}_{f}.npz")) for f in ("val", "test")}
                P7.OUT = os.path.join(OUT, "_refit"); os.makedirs(P7.OUT, exist_ok=True)
            nl = arm.endswith("_nl")
            F, use_base = family(task, arm[:-3] if nl else arm, world, lb, Z, k, s, groups, masks)
            Xb = Xflat.drop(columns=LEAK) if nl else Xflat
            assert not nl or not any(c in Xb.columns for c in LEAK)
            X = Xb if F is None else (pd.concat([Xb, F], axis=1) if use_base else F).astype(np.float32)
            meta = dict(seed=s, arm=arm, world=world, k=k, n_features=X.shape[1], commit=stamp)
            t = time.time()
            if task in ("arrival", "place"):
                y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
                m = P7.lgbm_fit("l2", X, y, tr, va, s, rows_tr=tr & ev, rows_va=va & ev)
                imp = dict(zip(X.columns, m.booster_.feature_importance("gain").tolist()))
                P7.emit(world, t_task, name, lb, va, te, lambda o: (m.predict(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), **meta))
            else:
                m = P7.lgbm_fit("multi", X, P7.fill_cell(lb.label_value.to_numpy(float)), tr, va, s, K=22)
                imp = dict(zip(X.columns, m.booster_.feature_importance("gain").tolist()))
                P7.emit(world, t_task, name, lb, va, te, lambda o: (m.predict_proba(X.iloc[o]), None), log,
                        dict(best_iter=int(m.best_iteration_ or 400), **meta))
            log[f"{world}|{t_task}|{name}"]["gain_importance"] = imp
            if exists:
                for f in ("val", "test"):
                    r = np.load(os.path.join(P7.OUT, f"{world}_{t_task}_{name}_{f}.npz"))
                    assert np.array_equal(r["P"], stored[f]["P"]) and (r["entity"] == stored[f]["entity"]).all(), \
                        f"{name} {f}: refit differs from the stored predictions -- STOP"
                    os.remove(os.path.join(P7.OUT, f"{world}_{t_task}_{name}_{f}.npz"))
                log[f"{world}|{t_task}|{name}"]["refit_bit_identical_to_stored"] = True
                P7.OUT = OUT
            print(f"  {world} {task} {name}: {time.time() - t:.0f}s ({X.shape[1]} features)", flush=True)
            json.dump(log, open(LOG_FMT.format(world=world, task=task), "w"), indent=1)
            if arm == "base" and task != "place":
                ref = (os.path.join(P7.ARTIFACTS, "phase7_preds", f"v8_{t_task}_{P20.P19.STORED[task]}_s7_{{f}}.npz") if world == "v8"
                       else os.path.join(ARTIFACTS, "phase20", "preds", f"{world}_{t_task}_p20_base_s7_{{f}}.npz"))
                for f in ("val", "test"):
                    a = np.load(os.path.join(OUT, f"{world}_{t_task}_{name}_{f}.npz")); b = np.load(ref.format(f=f))
                    d = float(np.abs(np.asarray(a["P"], float) - np.asarray(b["P"], float)).max())
                    assert d == 0.0 and (a["entity"] == b["entity"]).all() and np.array_equal(a["Y"], b["Y"]), \
                        f"{world} {task} {f}: BASE does not reproduce the stored predictions bit-exactly ({d}) -- STOP"
                log[f"{world}|{t_task}|{name}|reproduction"] = dict(reference=ref, bit_exact=True)
                json.dump(log, open(LOG_FMT.format(world=world, task=task), "w"), indent=1)
                print(f"  REPRODUCTION {world} {task}: bit-exact vs {ref}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=["v8", PP.WORLD2])
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, P7.SEEDS)))
    a = ap.parse_args()
    PP.register()
    os.makedirs(OUT, exist_ok=True)
    P7.OUT = OUT
    st = C.require_clean()
    LOG = LOG_FMT.format(world=a.world, task=a.task)
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log.setdefault("_stamps", []).append(dict(world=a.world, task=a.task, arms=a.arms, data=PP.register(), **st))
    arms = a.arms.split(",")
    assert all(x in ARMS[a.task] for x in arms), arms
    fit(a.world, a.task, arms, [int(x) for x in a.seeds.split(",")], log, st["code_commit"])
    assert "torch" not in sys.modules
    print("done; torch never imported", flush=True)
