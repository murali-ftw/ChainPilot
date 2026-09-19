"""Phase 6 — the training loop (guide 6.2–6.3) and the inference path.

A trained model is a BUNDLE, not a checkpoint. The eight steps, in order:

  1  load panel, build graph, slice windows            as-of asserted (folds.assert_no_leak)
  2  train with early stopping                         patience 8, cap 120, restore best
  3  SAVE checkpoint.pt                                 restored-best weights
  4  SAVE preds_val.npz, preds_test.npz, model_outputs.csv.gz
  5  fit recalibration on the VALIDATION fold           arrival: 13-cell moment matching;
                                                        fill: 22-cell MM or VS, selected on validation log score
  6  SAVE recalibration.json
  7  compute the drift baseline                         label-free statistic on validation inputs
  8  SAVE drift_baseline.json, normaliser.npz, metrics.json, train_log.json, config.json (seed, stamps)

A bundle without steps 5–8 is not shippable: the uncalibrated fill head is 2–5× worse on calibration.
Recalibration is refitted on EVERY training run — the fitted temperature differs across seeds.

The data path, model and losses are Phase 5's (`phase5_heads.py`), which produced every number this loop
must reproduce; this file owns the epoch loop, seeding, the pipeline and the bundle.

  python ml/train/loop.py train   --config ml/configs/shipped.json --task arrival_week --world v6 --seed 7
  python ml/train/loop.py train   --task fill_rate --world v6 --arch none --depth 0 --lr 1.25e-4 --seed 7
  python ml/train/loop.py convert --from-preds <phase-5 cell .npz> --task ... --world ... --arch ... --depth ... --lr ... --seed ...
  python ml/train/loop.py predict --bundle <dir> [--h0-bundle <dir>] [--fold test]
"""
from __future__ import annotations
import os, sys, json, time, copy, random, argparse, subprocess, warnings, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd, torch
import phase5_heads as P5
import temporal_share as TS
import folds as FO
import artifact_identity as AI
import phase5_metrics as M
from heads import HazardHead, fill_cell, fill_to_legacy
from phase5_recal import fit_mm, apply_mm, fit_vs, apply_vs, log_score
from metrics import cindex
from config import WORLDS, ARTIFACTS, REPO
from device import DTYPE, peak_rss_gb

DEV = P5.DEV
W12 = P5.HORIZON_WEEKS
BUNDLES = os.path.join(ARTIFACTS, "bundles")
BACKTEST_BUNDLES = os.path.join(ARTIFACTS, "backtest", "bundles")     # Phase 8.2 rolling-origin bundles
ENTITY = {"arrival_week": "po_line", "fill_rate": "po_line", "capacity_strain": "channel", "shortage_qty": "part_plant"}
MODEL_NAME = {t: f"hades-{t}" for t in ENTITY}          # model_outputs has no task column: model_name carries it
NONDET_OPS = set()


# ================================================================== seeding (guide 6.2)
def seed_all(seed):
    """torch, numpy and python seeded; deterministic algorithms requested WITH warn_only.

    Strict mode raises on this model: `index_put_with_accumulate_mps` has no deterministic MPS kernel.
    Two identical forward passes differ by ~3e-8, so reproduction is checked with allclose, never equal.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _capture_nondet(wlist):
    for w in wlist:
        msg = str(w.message)
        if "deterministic implementation" in msg:
            NONDET_OPS.add(msg.split(" does not have")[0])


# ================================================================== stamps (guide 6.3)
def stamps(world, cfg):
    snap = pd.read_csv(os.path.join(WORLDS[world], "snapshots.csv"))
    one = lambda c: sorted(snap[c].astype(str).unique())
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "ml"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    return dict(dataset_version=one("dataset_version"), feature_spec_version=one("feature_spec_version"),
                label_version=one("label_version"), generator_code_commit=one("code_commit"),
                code_commit=commit, code_dirty=dirty,
                model_version=f"{cfg['task']}-{cfg['arch']}-h{cfg['depth']}-lr{cfg['lr']:g}-s{cfg['seed']}-{commit}{'+dirty' if dirty else ''}")


def snapshot_ids(world):
    snap = pd.read_csv(os.path.join(WORLDS[world], "snapshots.csv"))
    return dict(zip(pd.to_datetime(snap.as_of_ts), snap.snapshot_id))


def bundle_dir(cfg):
    name = AI.bundle_name(cfg)                          # ml/artifact_identity.py -- the only place a name is built
    if cfg.get("origin"):
        return os.path.join(BACKTEST_BUNDLES, cfg["task"], f"o{cfg['origin']}", name)
    return os.path.join(BUNDLES, cfg["task"], name)


def split_of(cfg, dates):
    """The fixed split, or a Phase 8.2 rolling origin when cfg carries one. No-leak asserted on both."""
    if cfg.get("origin"):
        tr, va, te = FO.rolling_split(dates, cfg["origin"])
    else:
        tr, va, te = FO.fixed_split(dates)
    FO.assert_no_leak(dates, tr, va, te)
    return tr, va, te


def val_name(cfg):
    return f"validation slice of rolling origin {cfg['origin']}" if cfg.get("origin") else "validation (2024)"


# ================================================================== config
def resolve(args):
    base = json.load(open(args.config)) if args.config else {"common": {}, "tasks": {}}
    t = base["tasks"].get(args.task, {})
    c = base["common"]
    cfg = dict(task=args.task, world=args.world, seed=int(args.seed),
               arch=args.arch or t.get("arch", "none"), depth=int(args.depth if args.depth is not None else t.get("depth", 0)),
               lr=float(args.lr if args.lr is not None else t["lr"]),
               gate=bool(c.get("gate", False)), wsla=bool(c.get("wsla", False)), fill_loss=c.get("fill_loss", "rps"),
               max_epochs=int(args.max_epochs or c.get("max_epochs", 120)), patience=int(c.get("patience", 8)),
               drift_statistic=t.get("drift_statistic", DEFAULT_STAT[args.task]),
               config_file=args.config, config_version=base.get("version"), tag=args.tag or "")
    assert not cfg["gate"] and not cfg["wsla"], "shipped configuration: staleness gate and reconstructed feature are OFF"
    if getattr(args, "origin", None):
        cfg["origin"] = int(args.origin)                   # only present on backtest cells: fixed-split configs unchanged
    if getattr(args, "train_snapshots", None):
        cfg["train_snapshots"] = int(args.train_snapshots)  # Phase 9A Stage B: history held to n snapshots
    return cfg


DEFAULT_STAT = {"arrival_week": "p_late_raw", "fill_rate": "p_complete_raw", "capacity_strain": "q50_mean", "shortage_qty": "p_mean"}


# ================================================================== step 1-2: train
def train(cfg, verbose=False):
    task, w = cfg["task"], cfg["world"]
    seed_all(cfg["seed"])
    lb = P5.labels(w, task)
    tr, va, te = split_of(cfg, lb.snapshot_date)
    if cfg.get("train_snapshots"):
        tr = FO.truncate_train(lb.snapshot_date, tr, cfg["train_snapshots"])
    D = P5.device_inputs(w, np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"])
    ymu, ysd = 0.0, 1.0
    if task == "capacity_strain":
        ymu = float(lb.label_value[tr].mean()); ysd = float(lb.label_value[tr].std())
    seed_all(cfg["seed"])                                   # init independent of the data path, as in Phase 5
    model = P5.HeadNet(D["X"].shape[2], task, cfg["arch"], cfg["depth"],
                       gate_cols=D["gate_cols"] if cfg["gate"] else None, fill_loss=cfg["fill_loss"]).to(DEV).to(DTYPE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=TS.HP["wd"])
    train_b = P5.batches(task, lb, tr, D["W"], ymu, ysd)
    n_train = int(tr.sum())
    best, best_ep, best_state, bad, losses, vals, rows_seen, ep_s = None, -1, None, 0, [], [], [], []
    t_start = time.time()
    for ep in range(cfg["max_epochs"]):
        e0 = time.time(); model.train(); ep_l, n_rows = [], 0
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            for t0, tg, _ in train_b:
                z = P5.forward(model, D, t0, tg["idx"])
                L, n_c = P5.loss_of(model, z, tg)
                opt.zero_grad(); L.backward(); opt.step()
                ep_l.append(float(L.detach())); n_rows += n_c
        _capture_nondet(wl)
        # guide 5.3 / Phase 5: every training row enters the loss, every epoch -- asserted, not assumed
        assert n_rows == n_train, f"rows entering the loss {n_rows:,} != training population {n_train:,}"
        rows_seen.append(n_rows); losses.append(float(np.mean(ep_l)))
        v = P5.val_score(task, P5.predict(model, D, lb, va, ymu, ysd))
        vals.append(v)
        if best is None or (v > best if P5.HIGHER[task] else v < best):
            best, best_ep, bad = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        ep_s.append(time.time() - e0)
        if verbose:
            print(f"      ep {ep:>3}  loss {losses[-1]:.5f}  val {v:.5f}{'  *' if bad == 0 else ''}  {ep_s[-1]:.1f}s", flush=True)
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    log = dict(epochs_run=len(losses), best_epoch=best_ep, best_val=best,
               stop="patience" if bad >= cfg["patience"] else "CAP (floor)", losses=losses, vals=vals,
               rows_in_loss_each_epoch=sorted(set(rows_seen)), n_train_rows=n_train,
               seconds=time.time() - t_start, sec_per_epoch=float(np.mean(ep_s)),
               params=sum(p.numel() for p in model.parameters()), ymu=ymu, ysd=ysd)
    return model, D, lb, (tr, va, te), log


# ================================================================== step 5: recalibration
def p13(pr):
    return np.concatenate([pr["pT"].astype(float), pr["S"][:, -1:].astype(float)], 1)


def cells13(Y, EV):
    return np.where(np.asarray(EV, bool), np.clip(np.asarray(Y, float), 1, W12).astype(int) - 1, W12)


def fit_recalibration(task, pv):
    if task == "fill_rate":
        P, c = pv["P"].astype(float), fill_cell(pv["Y"])
        r = fit_mm(P, c); T, b = fit_vs(P, c)
        ls = {"none": log_score(P, c), "mm": log_score(apply_mm(P, r), c), "vs": log_score(apply_vs(P, T, b), c)}
        return dict(kind="fill22", method=min(ls, key=ls.get), mm_ratio=[float(x) for x in r], vs_T=float(T),
                    vs_bias=[float(x) for x in b], val_log_score=ls, fitted_on="validation fold (2024)")
    if task == "arrival_week":
        P, c = p13(pv), cells13(pv["Y"], pv["EV"])
        r = fit_mm(P, c)
        return dict(kind="arrival13", method="mm", mm_ratio=[float(x) for x in r],
                    val_log_score={"none": log_score(P, c), "mm": log_score(apply_mm(P, r), c)},
                    fitted_on="validation fold (2024)")
    return dict(kind=None, method="none")


def apply_recalibration(rec, task, pr):
    """-> dict with the recalibrated distribution (and its ranking score for arrival); raw copy if no recal."""
    if task == "fill_rate":
        P = pr["P"].astype(float)
        Q = P if rec["method"] == "none" else (apply_mm(P, np.array(rec["mm_ratio"])) if rec["method"] == "mm"
                                               else apply_vs(P, rec["vs_T"], np.array(rec["vs_bias"])))
        return {"P22": Q}
    if task == "arrival_week":
        P = p13(pr)
        Q = apply_mm(P, np.array(rec["mm_ratio"])) if rec["method"] == "mm" else P
        S = 1.0 - np.cumsum(Q[:, :W12], 1)
        return {"P13": Q, "S": S, "ET": 1.0 + S.sum(1)}
    return {}


# ================================================================== step 7: drift statistics (label-free)
def drift_stats(task, pr, rc):
    """Label-free: computed from predictions alone, so it can run on production inputs."""
    if task == "arrival_week":
        return dict(p_late_raw=float(pr["S"][:, -1].mean()), p_late_recal=float(rc["P13"][:, W12].mean()),
                    expected_week_raw=float(pr["P"].mean()))
    if task == "fill_rate":
        return dict(p_complete_raw=float(pr["P"][:, 21].mean()), p_complete_recal=float(rc["P22"][:, 21].mean()),
                    interior_raw=float(pr["P"][:, 1:21].sum(1).mean()))
    if task == "capacity_strain":
        return dict(q50_mean=float(pr["P"][:, 1].mean()), q90_mean=float(pr["P"][:, 2].mean()),
                    q10_mean=float(pr["P"][:, 0].mean()))
    return dict(p_mean=float(pr["P"].mean()))


# ================================================================== metrics on test
def test_metrics(task, pt, rc):
    out = {}
    if task == "arrival_week":
        s = M.arrival_scores(pt["P"], pt["Y"], pt["EV"], pt["AUX"], S=pt["S"], pT=pt["pT"])
        c = cells13(pt["Y"], pt["EV"])
        out = {"raw": s, "recal": {"ece_week": (M.ece_marginal(rc["P13"], c)[0],),
                                   "cindex_on_recal_ET": (cindex(rc["ET"], pt["Y"], pt["EV"]),)}}
    elif task == "fill_rate":
        out = {"raw": M.fill_scores(pt["P"], pt["Y"], "cells22"), "recal": M.fill_scores(rc["P22"], pt["Y"], "cells22")}
    elif task == "capacity_strain":
        out = {"raw": M.capacity_scores(pt["P"], pt["Y"])}
    else:
        out = {"raw": M.shortage_scores(pt["P"], pt["Y"])}
    return out


# ================================================================== model_outputs (guide 6.3)
MID22 = np.concatenate([[0.0], (np.arange(20) + 0.5) / 20, [1.0]])


def _quantile_from_cells(Pc, values, q):
    cdf = np.cumsum(Pc, 1)
    return values[np.clip((cdf < q).sum(1), 0, len(values) - 1)]


def model_outputs(cfg, lb, mask, pr, rc, stmp):
    order = P5.ordered(lb, mask)
    sid = snapshot_ids(cfg["world"])
    snap = pd.to_datetime(lb.snapshot_date.values[order])
    task = cfg["task"]
    n = len(order)
    dist = [""] * n
    if task == "arrival_week":
        weeks = np.arange(1, W12 + 2).astype(float)
        point, p10, p90 = rc["ET"], _quantile_from_cells(rc["P13"], weeks, 0.1), _quantile_from_cells(rc["P13"], weeks, 0.9)
        dist = [json.dumps([round(float(x), 5) for x in row]) for row in rc["P13"]]
        basis = "hazard head, 13-cell moment-matching recalibration fitted on the 2024 validation fold"
    elif task == "fill_rate":
        point = (rc["P22"] * MID22).sum(1)
        p10, p90 = _quantile_from_cells(rc["P22"], MID22, 0.1), _quantile_from_cells(rc["P22"], MID22, 0.9)
        dist = [json.dumps([round(float(x), 5) for x in row]) for row in rc["P22"]]
        basis = "point-mass CDF head, 22-cell recalibration fitted on the 2024 validation fold"
    elif task == "capacity_strain":
        point, p10, p90 = pr["P"][:, 1], pr["P"][:, 0], pr["P"][:, 2]
        basis = "quantile head, monotone by construction; no recalibration"
    else:
        point, p10, p90 = pr["P"], np.full(n, np.nan), np.full(n, np.nan)
        basis = "DIAGNOSTIC binary head -- not the shortage product path (Monte Carlo, Phase 9)"
    now = datetime.datetime.now().isoformat(timespec="seconds")
    return pd.DataFrame(dict(
        output_id=[f"{stmp['model_version']}-{i:06d}" for i in range(n)],
        snapshot_id=[sid[pd.Timestamp(d)] for d in snap], model_name=MODEL_NAME[task], model_version=stmp["model_version"],
        entity_type=ENTITY[task], entity_id=lb.entity_id.values[order], horizon_days=90,
        point_estimate=np.round(point, 6), p10=np.round(p10, 6), p90=np.round(p90, 6), distribution=dist,
        confidence_basis=basis, evidence_strength="recalibration and drift baseline fitted on the validation fold; see bundle",
        created_ts=now))


def verify_join(bundle):
    """guide 6.3 verify: model_outputs joined to training_labels returns one row per prediction, with its actual."""
    cfg = json.load(open(os.path.join(bundle, "config.json")))
    mo = pd.read_csv(os.path.join(bundle, "model_outputs.csv.gz"), usecols=["snapshot_id", "model_name", "entity_id"])
    mo["task"] = mo.model_name.map({v: k for k, v in MODEL_NAME.items()})
    lb = pd.read_csv(os.path.join(WORLDS[cfg["world"]], "training_labels.csv"),
                     usecols=["snapshot_id", "entity_id", "task", "label_value"])
    lb = lb[lb.task == cfg["task"]]
    j = mo.merge(lb, on=["snapshot_id", "entity_id", "task"], how="left", validate="one_to_one")
    return dict(predictions=len(mo), joined_with_actual=int(j.label_value.notna().sum()),
                one_row_per_prediction=bool(len(j) == len(mo)), all_have_actuals=bool(j.label_value.notna().all()))


# ================================================================== invariants re-checked on every bundle
def invariants(task, pr, lb):
    out = {}
    if task == "arrival_week":
        out["survival_monotone_violations"] = int((np.diff(pr["S"], axis=1) > 0).sum())
        out["sum_check_max_dev"] = float(np.abs(pr["pT"].sum(1) + pr["S"][:, -1] - 1).max())
        assert out["survival_monotone_violations"] == 0
    if task == "capacity_strain":
        Q = pr["P"]
        out["quantile_crossings"] = int(((Q[:, 0] > Q[:, 1]) | (Q[:, 1] > Q[:, 2])).sum())
        assert out["quantile_crossings"] == 0
    if task == "fill_rate":
        y = lb.label_value.to_numpy(float)
        ok = np.array_equal(fill_to_legacy(np.eye(22)[fill_cell(y)]).argmax(1), M.legacy_bin(y))
        out["partition_refines_legacy_bins"] = bool(ok); out["labels_checked"] = int(len(y))
        assert ok
    return out


# ================================================================== steps 3-8
def finish_bundle(cfg, model, D, lb, split, log, trained_by):
    task = cfg["task"]; tr, va, te = split
    out = bundle_dir(cfg); os.makedirs(out, exist_ok=True)
    stmp = stamps(cfg["world"], cfg)
    # 3
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(out, "checkpoint.pt"))
    np.savez(os.path.join(out, "normaliser.npz"), mu=D["norm_mu"], sd=D["norm_sd"], log1p_idx=np.array(D["norm_log1p_idx"]))
    # 4
    pv = P5.predict(model, D, lb, va, log["ymu"], log["ysd"])
    pt = P5.predict(model, D, lb, te, log["ymu"], log["ysd"])
    np.savez_compressed(os.path.join(out, "preds_val.npz"), **{k: v for k, v in pv.items() if not k.startswith("_")})
    np.savez_compressed(os.path.join(out, "preds_test.npz"), **{k: v for k, v in pt.items() if not k.startswith("_")})
    # 5-6
    rec = fit_recalibration(task, pv)
    if cfg.get("origin"):
        rec["fitted_on"] = val_name(cfg)                   # refitted on this origin's own validation slice, never carried
    json.dump(rec, open(os.path.join(out, "recalibration.json"), "w"), indent=1)
    rv, rt = apply_recalibration(rec, task, pv), apply_recalibration(rec, task, pt)
    # 7
    base = dict(statistic=cfg["drift_statistic"], fold=val_name(cfg), n_rows=int(len(pv["Y"])),
                snapshots=sorted(str(d.date()) for d in pd.to_datetime(lb.snapshot_date[va].unique())),
                values=drift_stats(task, pv, rv), label_free=True)
    json.dump(base, open(os.path.join(out, "drift_baseline.json"), "w"), indent=1)
    # 8
    mo = model_outputs(cfg, lb, te, pt, rt, stmp)
    if cfg.get("origin"):
        mo["confidence_basis"] = mo.confidence_basis.str.replace("the 2024 validation fold", val_name(cfg), regex=False)
    mo.to_csv(os.path.join(out, "model_outputs.csv.gz"), index=False)
    metrics = dict(test=test_metrics(task, pt, rt), invariants=invariants(task, pt, lb),
                   test_drift_values_label_free=drift_stats(task, pt, rt))
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    json.dump({**log, "trained_by": trained_by, "nondeterministic_ops_warned": sorted(NONDET_OPS),
               "peak_rss_gb": peak_rss_gb()}, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    json.dump({**cfg, "stamps": stmp, "split": FO.describe_origin(cfg["origin"]) if cfg.get("origin") else FO.describe_fixed(), "HP": TS.HP,
               "files": ["checkpoint.pt", "normaliser.npz", "preds_val.npz", "preds_test.npz", "recalibration.json",
                         "drift_baseline.json", "model_outputs.csv.gz", "metrics.json", "train_log.json"],
               "complete": True}, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump(verify_join(out), open(os.path.join(out, "join_check.json"), "w"), indent=1)
    return out


def run_train(cfg):
    out = bundle_dir(cfg)
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}", flush=True); return out
    print(f"\n=== train {cfg['task']} {cfg['world']} {cfg['arch']} h{cfg['depth']} lr {cfg['lr']:g} s{cfg['seed']} {cfg['tag']}", flush=True)
    t0 = time.time()
    model, D, lb, split, log = train(cfg)
    out = finish_bundle(cfg, model, D, lb, split, log, trained_by="ml/train/loop.py")
    print(f"  [{log['stop']}] best_ep {log['best_epoch']} of {log['epochs_run']}  val {log['best_val']:.5f}  "
          f"{time.time() - t0:.0f}s  {log['sec_per_epoch']:.1f}s/ep  RSS {peak_rss_gb():.2f} GB  -> {out}", flush=True)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def run_convert(cfg, preds_path):
    """Steps 3-8 for a Phase 5 cell that already has a restored-best checkpoint: no retraining.

    Reloads the checkpoint, recomputes validation predictions, and asserts they match the saved ones with allclose.
    """
    out = bundle_dir(cfg)
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}", flush=True); return out
    seed_all(cfg["seed"])
    lb = P5.labels(cfg["world"], cfg["task"])
    tr, va, te = FO.fixed_split(lb.snapshot_date); FO.assert_no_leak(lb.snapshot_date, tr, va, te)
    D = P5.device_inputs(cfg["world"], np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"])
    model = P5.HeadNet(D["X"].shape[2], cfg["task"], cfg["arch"], cfg["depth"], fill_loss=cfg["fill_loss"]).to(DEV)
    model.load_state_dict(torch.load(preds_path.replace(".npz", ".pt"), map_location="cpu"))
    ymu, ysd = 0.0, 1.0
    if cfg["task"] == "capacity_strain":
        ymu = float(lb.label_value[tr].mean()); ysd = float(lb.label_value[tr].std())
    pv_new = P5.predict(model, D, lb, va, ymu, ysd)
    pv_old = np.load(preds_path.replace(".npz", "_val.npz"))
    reload = float(np.abs(pv_new["P"] - pv_old["P"]).max())
    assert np.allclose(pv_new["P"], pv_old["P"], atol=1e-5), f"checkpoint does not reproduce its saved validation predictions (max diff {reload})"
    log = dict(epochs_run=None, best_epoch=None, best_val=None, stop="converted", ymu=ymu, ysd=ysd,
               reload_max_abs_diff_val=reload, source_predictions=preds_path)
    out = finish_bundle(cfg, model, D, lb, (tr, va, te), log, trained_by=f"converted from {os.path.basename(preds_path)}")
    print(f"  converted -> {out} (reload max diff {reload:.2e})", flush=True)
    return out


# ================================================================== inference
def load_bundle(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    assert cfg.get("complete"), f"{path} is not a complete bundle"
    return dict(path=path, cfg=cfg, rec=json.load(open(os.path.join(path, "recalibration.json"))),
                base=json.load(open(os.path.join(path, "drift_baseline.json"))),
                norm=np.load(os.path.join(path, "normaliser.npz")))


def _materialise(B, D):
    cfg = B["cfg"]
    m = P5.HeadNet(D["X"].shape[2], cfg["task"], cfg["arch"], cfg["depth"], fill_loss=cfg["fill_loss"]).to(DEV)
    m.load_state_dict(torch.load(os.path.join(B["path"], "checkpoint.pt"), map_location="cpu"))
    return m.eval()


def status_of(excess_pp, bands):
    if excess_pp is None: return "no h0 reference"
    if excess_pp <= bands["usable"]: return "usable"
    if excess_pp <= bands["watch"]: return "watch"
    if excess_pp <= bands["degraded"]: return "degraded"
    return "unusable"


def predict(bundle, fold="test", h0_bundle=None, shipped_config="ml/configs/shipped.json", threshold_override=None):
    """Load a bundle, apply its recalibration, and emit drift alongside every prediction batch.

    'Production inputs' here are a fold's windows; no label value is read on this path. Drift for a batch is the
    batch's statistic minus the validation baseline, in percentage points; with an h0 bundle, EXCESS drift is
    |model drift - h0 drift| on the same inputs, which is what Addendum B's thresholds are defined on.
    For arrival, excess above the configured threshold switches the Monte Carlo distribution to h0's.
    """
    shipped = json.load(open(os.path.join(REPO, shipped_config)))
    bands = shipped["drift_bands_pp"]
    B = load_bundle(bundle); cfg = B["cfg"]; task = cfg["task"]
    H = load_bundle(h0_bundle) if h0_bundle else None
    lb = P5.labels(cfg["world"], task)
    tr, va, te = split_of(cfg, lb.snapshot_date)
    mask = {"test": te, "val": va}[fold]
    D = P5.device_inputs(cfg["world"], np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"])
    assert np.allclose(D["norm_mu"], B["norm"]["mu"]) and np.allclose(D["norm_sd"], B["norm"]["sd"]), \
        "the rebuilt normaliser does not match the bundle's -- inputs changed since training"
    model = _materialise(B, D); h0 = _materialise(H, D) if H else None
    stat = cfg["drift_statistic"]
    ymu, ysd = 0.0, 1.0
    if task == "capacity_strain":
        tl = json.load(open(os.path.join(bundle, "train_log.json"))); ymu, ysd = tl["ymu"], tl["ysd"]
    order = P5.ordered(lb, mask); dates = lb.snapshot_date.values[order]
    keymap = D["W"]["pp_uniq"] if task == "shortage_qty" else D["W"]["cidx"]
    batches, parts, parts_h0 = [], [], []
    with torch.no_grad():
        for s in np.unique(dates):
            ii = order[dates == s]
            idx = torch.from_numpy(lb.key.iloc[ii].map(keymap).to_numpy(np.int64)).to(DEV)
            t0 = P5.t0_of(D["W"], s)
            pr = _head_outputs(task, P5.forward(model, D, t0, idx), ymu, ysd)
            rc = apply_recalibration(B["rec"], task, pr)
            d = drift_stats(task, pr, rc)[stat]
            rec = dict(snapshot=str(pd.Timestamp(s).date()), n=int(len(ii)), statistic=stat, value=d,
                       baseline=B["base"]["values"][stat], drift_pp=100 * (d - B["base"]["values"][stat]))
            if h0 is not None:
                ph = _head_outputs(task, P5.forward(h0, D, t0, idx), ymu, ysd)
                rh = apply_recalibration(H["rec"], task, ph)
                dh = drift_stats(task, ph, rh)[stat]
                rec.update(h0_drift_pp=100 * (dh - H["base"]["values"][stat]))
                rec["excess_pp"] = abs(rec["drift_pp"] - rec["h0_drift_pp"])
                parts_h0.append((ph, rh))
            batches.append(rec); parts.append((pr, rc))
    # overall, row-weighted -- Addendum B's thresholds were measured on whole-fold means, so decisions use this
    n = np.array([b["n"] for b in batches], float)
    overall = dict(statistic=stat, drift_pp=float((n * [b["drift_pp"] for b in batches]).sum() / n.sum()))
    if h0 is not None:
        overall["h0_drift_pp"] = float((n * [b["h0_drift_pp"] for b in batches]).sum() / n.sum())
        overall["excess_pp"] = abs(overall["drift_pp"] - overall["h0_drift_pp"])
    # bands were calibrated on arrival's P(T > 12); applying them to a utilisation level (capacity) is meaningless
    if shipped["tasks"][task].get("drift_bands_apply"):
        overall["status"] = status_of(overall.get("excess_pp"), bands)
    else:
        overall["status"] = "no calibrated threshold" if h0 is not None else "no h0 reference"
    fb = shipped["tasks"][task].get("fallback")
    threshold = fb["threshold_pp"] if fb else None
    if fb and threshold_override is not None:
        threshold = float(threshold_override)                      # testing only: exercise the engaged path
    overall["fallback_threshold_pp"] = threshold
    use_h0 = bool(fb and h0 is not None and overall.get("excess_pp", 0) > threshold)
    overall["distribution_source"] = "h0 (fallback engaged)" if use_h0 else "model"
    cat = lambda lst, k: np.concatenate([x[k] for x in lst])
    result = dict(batches=batches, overall=overall)
    if task == "arrival_week":
        result["ranking_score"] = cat([p for p, _ in parts], "P")                 # SHARE-lite ranks, always
        src = parts_h0 if use_h0 else parts
        result["distribution"] = cat([r for _, r in src], "P13")
    elif task == "fill_rate":
        result["distribution"] = cat([r for _, r in parts], "P22")
    elif task == "capacity_strain":
        result["quantiles"] = cat([p for p, _ in parts], "P")
    else:
        result["probability"] = cat([p for p, _ in parts], "P")
    return result


def _head_outputs(task, z, ymu, ysd):
    if task == "arrival_week":
        lam, S, pT = HazardHead.distribution(z)
        return {"P": HazardHead.expected_time(S).float().cpu().numpy(), "S": S.float().cpu().numpy(), "pT": pT.float().cpu().numpy()}
    if task == "fill_rate":
        return {"P": torch.softmax(z, -1).float().cpu().numpy()}
    if task == "capacity_strain":
        return {"P": (z[:, 0, :] * ysd + ymu).float().cpu().numpy()}
    return {"P": torch.sigmoid(z).float().cpu().numpy()}


# ================================================================== CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "convert", "predict"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--task"); ap.add_argument("--world"); ap.add_argument("--seed", default=7)
    ap.add_argument("--arch", default=None); ap.add_argument("--depth", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--origin", type=int, default=None, help="Phase 8.2 rolling origin 1-8; omit for the fixed split")
    ap.add_argument("--from-preds", default=None)
    ap.add_argument("--bundle", default=None); ap.add_argument("--h0-bundle", default=None); ap.add_argument("--fold", default="test")
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke runs only; shipped runs use the config's cap")
    ap.add_argument("--bundle-root", default=None, help="write bundles elsewhere (smoke tests must not occupy real bundle paths)")
    a = ap.parse_args()
    if a.bundle_root:
        BUNDLES = a.bundle_root
    if a.mode == "predict":
        r = predict(a.bundle, a.fold, a.h0_bundle)
        print(json.dumps(r["overall"], indent=1))
        for b in r["batches"]:
            print("  ", json.dumps(b))
        sys.exit(0)
    cfg = resolve(a)
    if a.mode == "train":
        run_train(cfg)
    else:
        run_convert(cfg, a.from_preds)
    print("DONE", flush=True)
