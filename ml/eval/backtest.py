"""Guide 8.2 — rolling-origin backtest (Phase 8).

  python ml/eval/backtest.py plan                      1a: windows, row counts, every leak assertion, per origin/world
  python ml/eval/backtest.py smoke                     2-epoch cells on origin 1 in a scratch root: pipeline + timing
  python ml/eval/backtest.py queue --queue 0 --queues 2   train the cells (resumable; ETA; 6 h budget stop)
  python ml/eval/backtest.py export                    bundle predictions -> Phase 7 file format for the single scorer

Cells per surviving origin, per world, seeds 7 / 17 / 27, all from ml/configs/shipped.json (no learning rate retuned):
  arrival   shipped SHARE-lite h4 @ 2.5e-4    and its h0 variant (encoder disabled, same head, same rate)
  capacity  shipped HeteroMP h4 @ 2.5e-4      and its h0 variant
  fill      shipped h0 @ 1.25e-4              (the shipped configuration IS the h0 variant: no second cell)
Shortage is diagnostic and is not queued. Capacity B5 LightGBM per origin is fitted torch-free by
`ml/baselines/phase7_fit.py --origins`. Scoring is `ml/eval/phase7_score.py --backtest` -- the one Phase 7 scorer.

Every cell is a complete Phase 6 bundle: early stopping, recalibration and the drift baseline are fitted on THAT
origin's validation slice, never carried across origins.
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from config import WORLDS, FIT_WINDOW, ARTIFACTS
import folds as FO

SHIPPED = "ml/configs/shipped.json"
BT = os.path.join(ARTIFACTS, "backtest")
PLAN = os.path.join(BT, "phase8_origins.json")
PREDS = os.path.join(BT, "preds")
INDEX = os.path.join(BT, "bundle_index.json")
STOP = os.path.join(BT, "STOP_over_budget.json")
TASKS = ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty")
SEEDS = (7, 17, 27)
BUDGET_H = 6.0


# ================================================================== 1a
def _labels_frame(w):
    lb = pd.read_csv(os.path.join(WORLDS[w], "training_labels.csv"), usecols=["snapshot_date", "task", "label_window_end"])
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    return lb[(lb.snapshot_date >= FIT_WINDOW[0]) & (lb.snapshot_date <= FIT_WINDOW[1])].reset_index(drop=True)


def _try(fn):
    try:
        fn(); return "PASS"
    except AssertionError as e:
        return f"FAIL: {e}"


def plan():
    os.makedirs(BT, exist_ok=True)
    out = dict(validation_months=FO.VAL_MONTHS, origins=[], fixed_split_label_overlap={})
    for w in ("v6", "v7"):
        lb = _labels_frame(w)
        cov = FO.covid_snapshots(w)
        # the spec's own assertion, over all eight origins at once, exactly as folds.py defines it
        spec_all = {t: _try(lambda t=t: FO.assert_rolling_origins(lb.snapshot_date[lb.task == t])) for t in TASKS}
        for t in TASKS:
            s = lb[lb.task == t]
            tr, va, te = FO.fixed_split(s.snapshot_date)
            out["fixed_split_label_overlap"][f"{w}|{t}"] = FO.label_window_overlap(s.snapshot_date, s.label_window_end, tr, va, te)
        for k, *_ in FO.ROLLING_ORIGINS:
            o = FO.origin_windows(k)
            row = dict(world=w, **FO.describe_origin(k), assertions={}, rows={}, snapshots={}, label_overlap={})
            for t in TASKS:
                s = lb[lb.task == t].reset_index(drop=True)
                d = s.snapshot_date
                tr, va, te = FO.rolling_split(d, k)
                tr_spec, ev_spec = FO.rolling_origin(d, k)
                A = {
                    "non_empty_folds": _try(lambda: (assert_(tr.any() and va.any() and te.any(), "an empty fold"))),
                    "no_leak (train < validation < test, fit window, disjoint)": _try(lambda: FO.assert_no_leak(d, tr, va, te)),
                    "spec §9.2 max(train incl. validation) < min(evaluate)": _try(
                        lambda: assert_(d[tr_spec].max() < d[ev_spec].min(), f"{d[tr_spec].max()} >= {d[ev_spec].min()}")),
                    "evaluation window equals the spec's": _try(lambda: assert_(np.array_equal(te, ev_spec), "test != spec window")),
                    "train + validation equals the spec's training cut": _try(
                        lambda: assert_(np.array_equal(tr | va, tr_spec), "train+validation != spec training rows")),
                    "folds.assert_rolling_origins (all 8, as defined)": spec_all[t],
                }
                row["assertions"][t] = A
                row["rows"][t] = dict(train=int(tr.sum()), validation=int(va.sum()), test=int(te.sum()))
                row["snapshots"][t] = dict(train=int(d[tr].nunique()), validation=int(d[va].nunique()), test=int(d[te].nunique()),
                                           covid_in_train=int(d[tr].isin(cov).groupby(d[tr]).first().sum()) if tr.any() else 0)
                if tr.any() and va.any() and te.any():
                    row["label_overlap"][t] = FO.label_window_overlap(d, s.label_window_end, tr, va, te)
            row["windows"] = {f: [str(a.date()), str(b.date())] for f, (a, b) in ((f, o[f]) for f in ("train", "validation", "test"))}
            row["survives"] = all(v == "PASS" for t in TASKS for v in row["assertions"][t].values())
            row["excluded_reason"] = None if row["survives"] else sorted(
                {f"{t}: {n} -> {v}" for t in TASKS for n, v in row["assertions"][t].items() if v != "PASS"})
            out["origins"].append(row)
    json.dump(out, open(PLAN, "w"), indent=1)
    for r in out["origins"]:
        print(f"\n{r['world']} origin {r['origin']}  train {r['windows']['train']}  validation {r['windows']['validation']}  "
              f"test {r['windows']['test']}  -> {'SURVIVES' if r['survives'] else 'EXCLUDED'}")
        for t in TASKS:
            print(f"   {t:16s} rows {r['rows'][t]}  snapshots {r['snapshots'][t]}")
            for n, v in r["assertions"][t].items():
                print(f"      [{v}] {n}")
            if t in r["label_overlap"]:
                print(f"      label-window diagnostic {r['label_overlap'][t]}")
    return out


def assert_(cond, msg):
    assert cond, msg


def surviving_origins():
    P = json.load(open(PLAN))
    ok = {}
    for r in P["origins"]:
        ok.setdefault(r["origin"], True)
        ok[r["origin"]] &= r["survives"]
    return [k for k in sorted(ok) if ok[k]]


# ================================================================== cells
def cells(origins, worlds=("v6", "v7"), tasks=None):
    """Origin order first, so the first origin completes before any other starts (the ETA is taken there)."""
    out = []
    for k in origins:
        for w in worlds:
            for s in SEEDS:
                out += [("arrival_week", w, s, None, None, None, k), ("arrival_week", w, s, "none", 0, 2.5e-4, k),
                        ("capacity_strain", w, s, None, None, None, k), ("capacity_strain", w, s, "none", 0, 2.5e-4, k),
                        ("fill_rate", w, s, None, None, None, k)]
    return [c for c in out if tasks is None or c[0] in tasks]


def cfg_of(c, tag="phase8", arrival_max_epochs=None, train_snapshots=None):
    """arrival_max_epochs: Stage 8E raises arrival's cap from the shipped 120 (deviation 22). Not a learning-rate change;
    it is written into each bundle's config, and bundle paths do not depend on it."""
    import loop as L
    task, w, s, arch, depth, lr, k = c
    ns = argparse.Namespace(config=SHIPPED, task=task, world=w, seed=s, arch=arch, depth=depth, lr=lr, tag=tag,
                            max_epochs=arrival_max_epochs if task == "arrival_week" else None, origin=k,
                            train_snapshots=train_snapshots)
    return L.resolve(ns)


def complete(cfg):
    import loop as L
    p = os.path.join(L.bundle_dir(cfg), "config.json")
    return os.path.exists(p) and json.load(open(p)).get("complete", False)


def train_snapshots(k):
    P = json.load(open(PLAN))
    return next(r["snapshots"]["arrival_week"]["train"] for r in P["origins"] if r["origin"] == k and r["world"] == "v6")


def projection(queues, tasks=None):
    """Mean measured seconds per (task, arch) cell on completed origins, scaled by training snapshots, / queues."""
    import loop as L
    ks = surviving_origins()
    done, todo = [], []
    for c in cells(ks, tasks=tasks):
        cfg = cfg_of(c)
        if complete(cfg):
            tl = json.load(open(os.path.join(L.bundle_dir(cfg), "train_log.json")))
            done.append((c, tl["seconds"] / train_snapshots(c[6])))
        else:
            todo.append(c)
    per = {}
    for c, sps in done:
        per.setdefault((c[0], c[3]), []).append(sps)
    rem = sum(np.mean(per.get((c[0], c[3]), [np.nan])) * train_snapshots(c[6]) for c in todo)
    return dict(cells_done=len(done), cells_todo=len(todo), remaining_gpu_hours=float(rem / 3600),
                remaining_wall_hours=float(rem / 3600 / max(1, queues)),
                total_wall_hours=float((rem + sum(s * train_snapshots(c[6]) for c, s in done)) / 3600 / max(1, queues)))


def queue(q, queues, allow_over_budget=False, only_origins=None, only_tasks=None, arrival_max_epochs=None,
          train_snapshots=None, shipped_only=False):
    import loop as L
    ks = surviving_origins()
    run_ks = [k for k in ks if only_origins is None or k in only_origins]  # the ETA decision is taken before origin 2 starts
    mine = [c for c in cells(run_ks, tasks=only_tasks) if not (shipped_only and c[3] is not None)]
    mine = [c for i, c in enumerate(mine) if i % queues == q]
    first = run_ks[0]
    for c in mine:
        if os.path.exists(STOP) and not allow_over_budget:
            print(f"[queue {q}] STOP file present ({STOP}); exiting before {c}", flush=True); return
        cfg = cfg_of(c, arrival_max_epochs=arrival_max_epochs, train_snapshots=train_snapshots)
        if complete(cfg):
            continue
        L.run_train(cfg)
        first_done = all(complete(cfg_of(x)) for x in cells([first], tasks=only_tasks))
        if first_done:
            p = projection(queues, only_tasks)
            print(f"[queue {q}] ETA after origin {first}: {json.dumps(p)}", flush=True)
            if p["total_wall_hours"] > BUDGET_H and not allow_over_budget:
                json.dump(dict(projection=p, budget_hours=BUDGET_H, written=time.ctime()), open(STOP, "w"), indent=1)
                print(f"[queue {q}] projected total {p['total_wall_hours']:.1f} h > {BUDGET_H} h: STOP", flush=True)
                return


# ================================================================== 8A: drift pre-pass, inference only
PREPASS = os.path.join(BT, "phase8a_drift_prepass.json")


def drift_prepass(source_origin=1, out=None):
    """Stage 8A. ONE origin's trained checkpoints, run over EVERY origin's evaluation-window inputs.

    No training, no refitting, no label value on the prediction path: each model's own label-free statistic against its
    own validation baseline, exactly as `loop.predict` computes it, and -- where an h0 twin of the same world and seed
    exists -- the excess the drift bands are defined on.

    This bounds the INPUT drift the eight windows contain. It is not Stage 3b: 3b needs each origin's own model, and a
    model run on a window far from its training period drifts for a reason 3b would not see.
    """
    import torch, loop as L, phase5_heads as P5
    ks = surviving_origins()
    rows = []
    for w in ("v6", "v7"):
        D, lbs = None, {}
        for cfgp in sorted(glob_bundles(L.BACKTEST_BUNDLES)):
            d = os.path.dirname(cfgp); cfg = json.load(open(cfgp))
            if cfg.get("origin") != source_origin or not cfg.get("complete") or cfg["world"] != w:
                continue
            task = cfg["task"]
            B = L.load_bundle(d)
            lb = lbs.setdefault(task, P5.labels(w, task))
            tr, _, _ = FO.rolling_split(lb.snapshot_date, source_origin)
            if D is None:
                D = P5.device_inputs(w, np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"])
            assert np.allclose(D["norm_mu"], B["norm"]["mu"]) and np.allclose(D["norm_sd"], B["norm"]["sd"]), \
                "rebuilt normaliser does not match the bundle's"
            model = L._materialise(B, D)
            ymu, ysd = 0.0, 1.0
            if task == "capacity_strain":
                tl = json.load(open(os.path.join(d, "train_log.json"))); ymu, ysd = tl["ymu"], tl["ysd"]
            stat = cfg["drift_statistic"]
            for k in ks:
                mask = FO.rolling_split(lb.snapshot_date, k)[2]
                order = P5.ordered(lb, mask); dates = lb.snapshot_date.values[order]
                keymap = D["W"]["pp_uniq"] if task == "shortage_qty" else D["W"]["cidx"]
                vals, ns = [], []
                with torch.no_grad():
                    for sdate in np.unique(dates):
                        ii = order[dates == sdate]
                        idx = torch.from_numpy(lb.key.iloc[ii].map(keymap).to_numpy(np.int64)).to(L.DEV)
                        pr = L._head_outputs(task, P5.forward(model, D, P5.t0_of(D["W"], sdate), idx), ymu, ysd)
                        vals.append(L.drift_stats(task, pr, L.apply_recalibration(B["rec"], task, pr))[stat]); ns.append(len(ii))
                n = np.array(ns, float); value = float((n * np.array(vals)).sum() / n.sum())
                rows.append(dict(world=w, task=task, arch=cfg["arch"], depth=cfg["depth"], seed=cfg["seed"], origin=k,
                                 statistic=stat, baseline=B["base"]["values"][stat], value=value,
                                 drift_pp=100 * (value - B["base"]["values"][stat]), rows=int(n.sum()),
                                 source_origin=source_origin, source_bundle=os.path.relpath(d, L.BACKTEST_BUNDLES)))
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        D = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    # excess against the h0 twin of the same world, task, seed and window -- what the bands are defined on
    by = {(r["world"], r["task"], r["seed"], r["origin"], r["depth"]): r for r in rows}
    for r in rows:
        h0 = by.get((r["world"], r["task"], r["seed"], r["origin"], 0))
        r["h0_drift_pp"] = None if h0 is None else h0["drift_pp"]
        r["excess_pp"] = None if (h0 is None or r["depth"] == 0) else abs(r["drift_pp"] - h0["drift_pp"])
    json.dump(dict(source_origin=source_origin, note=drift_prepass.__doc__, rows=rows), open(out or PREPASS, "w"), indent=1)
    print(f"pre-pass: {len(rows)} (bundle x origin) inferences -> {out or PREPASS}", flush=True)
    return rows


# ================================================================== export -> the single scorer
def export(bundle_root=None, preds=None, index_path=None):
    import loop as L
    global PREDS, INDEX
    PREDS = preds or PREDS; INDEX = index_path or INDEX
    root = bundle_root or L.BACKTEST_BUNDLES
    os.makedirs(PREDS, exist_ok=True)
    index = {}
    for cfgp in sorted(glob_bundles(root)):
        d = os.path.dirname(cfgp); cfg = json.load(open(cfgp))
        if not cfg.get("complete"):
            continue
        task, w, k = cfg["task"], cfg["world"], cfg["origin"]
        name = f"{cfg['arch']}_h{cfg['depth']}_lr{cfg['lr']:g}_s{cfg['seed']}"
        if cfg.get("train_snapshots"):
            name += f"_tr{cfg['train_snapshots']}"     # a truncated-history cell is its own configuration: never the
                                                       # same prediction file, index key or band as the full-history cell
        rec = json.load(open(os.path.join(d, "recalibration.json")))
        # fixed-width unicode, as Phase 7 saves it: an object array cannot be np.load-ed without pickle
        ent = pd.read_csv(os.path.join(d, "model_outputs.csv.gz"), usecols=["entity_id"]).entity_id.to_numpy().astype(str)
        for fold in ("val", "test"):
            z = dict(np.load(os.path.join(d, f"preds_{fold}.npz")))
            base = dict(Y=z["Y"], EV=z["EV"], AUX=z["AUX"])
            if fold == "test":
                assert len(ent) == len(z["Y"]), "model_outputs rows != test predictions"
                base["entity"] = ent
            stem = f"{w}_{task}_o{k}_{name}_{fold}.npz"
            if task == "arrival_week":
                np.savez_compressed(os.path.join(PREDS, stem), P=z["P"], S=z["S"], pT=z["pT"], **base)
                r = L.apply_recalibration(rec, task, z)
                np.savez_compressed(os.path.join(PREDS, "RECAL_" + stem), P=r["ET"], S=r["S"], pT=r["P13"][:, :L.W12], **base)
                if cfg["arch"] == "none" and cfg["seed"] == 7:          # promise-only, once per origin/world
                    np.savez_compressed(os.path.join(PREDS, f"PROMISE_{w}_{task}_o{k}_promise_only_{fold}.npz"), P=z["AUX"], **base)
            elif task == "fill_rate":
                np.savez_compressed(os.path.join(PREDS, stem), P=z["P"], **base)
                np.savez_compressed(os.path.join(PREDS, "RECAL_" + stem), P=L.apply_recalibration(rec, task, z)["P22"], **base)
            else:
                np.savez_compressed(os.path.join(PREDS, stem), P=z["P"], **base)
        tl = json.load(open(os.path.join(d, "train_log.json")))
        drift_b = json.load(open(os.path.join(d, "drift_baseline.json")))
        met = json.load(open(os.path.join(d, "metrics.json")))
        index[f"{w}|{task}|o{k}_{name}"] = dict(
            bundle=d, stamp=cfg["stamps"]["model_version"], split=cfg["split"], epochs=tl.get("epochs_run"),
            best_epoch=tl.get("best_epoch"), stop=tl.get("stop"), seconds=tl.get("seconds"),
            recalibration={k2: rec.get(k2) for k2 in ("method", "vs_T", "mm_ratio", "fitted_on")},
            drift_val=drift_b["values"], drift_test=met["test_drift_values_label_free"])
    json.dump(index, open(INDEX, "w"), indent=1)
    print(f"exported {len(index)} bundles -> {PREDS}", flush=True)
    return index


def glob_bundles(root):
    import glob
    return glob.glob(os.path.join(root, "*", "o*", "*", "config.json"))


# ================================================================== smoke: pipeline end to end + timing, scratch root
def smoke(root, max_epochs=2):
    import loop as L
    L.BACKTEST_BUNDLES = root
    rows = []
    for c in cells([1], worlds=("v6",)):
        if c[2] != 7:
            continue
        cfg = cfg_of(c, tag="phase8 smoke"); cfg["max_epochs"] = max_epochs
        t = time.time(); out = L.run_train(cfg)
        tl = json.load(open(os.path.join(out, "train_log.json")))
        rows.append(dict(cell=f"{cfg['task']} {cfg['arch']} h{cfg['depth']}", total_s=time.time() - t, train_s=tl["seconds"],
                         sec_per_epoch=tl["sec_per_epoch"], epochs=tl["epochs_run"], stamp=json.load(open(os.path.join(out, "config.json")))["stamps"]["model_version"]))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(BT, "smoke_timing.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "smoke", "queue", "export", "project", "prepass"])
    ap.add_argument("--queue", type=int, default=0); ap.add_argument("--queues", type=int, default=2)
    ap.add_argument("--allow-over-budget", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--origins", default=None, help="queue only these origins (comma list); projection still covers all")
    ap.add_argument("--preds", default=None); ap.add_argument("--index", default=None)
    ap.add_argument("--tasks", default=None, help="queue only these tasks (comma list), e.g. capacity_strain")
    ap.add_argument("--arrival-max-epochs", type=int, default=None, help="Stage 8E: arrival epoch cap (deviation 22)")
    ap.add_argument("--train-snapshots", type=int, default=None,
                    help="Phase 9A Stage B: truncate training to the most recent n snapshots (own bundle, own band)")
    ap.add_argument("--shipped-only", action="store_true", help="queue the shipped configuration only, not its h0 twin")
    a = ap.parse_args()
    if a.mode == "plan":
        plan()
    elif a.mode == "smoke":
        smoke(a.root or os.path.join(BT, "smoke_bundles"))
    elif a.mode == "queue":
        queue(a.queue, a.queues, a.allow_over_budget, [int(x) for x in a.origins.split(",")] if a.origins else None,
              a.tasks.split(",") if a.tasks else None, a.arrival_max_epochs, a.train_snapshots, a.shipped_only)
    elif a.mode == "project":
        print(json.dumps(projection(a.queues, a.tasks.split(",") if a.tasks else None), indent=1))
    elif a.mode == "prepass":
        drift_prepass()
    else:
        export(a.root, a.preds, a.index)
    print("DONE", flush=True)
