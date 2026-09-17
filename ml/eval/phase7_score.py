"""Phase 7 — score every baseline through the bundle scorer, on asserted-identical rows.

Runs in a torch process with NO LightGBM (the two cannot share a process here). Every metric comes from
`phase5_metrics` — the same functions `loop.test_metrics` used to score the Phase 6 bundles — with 1,000-resample
bootstrap intervals. Recalibration uses `loop.fit_recalibration`, the neural head's own protocol.

  1  row identity      every baseline file's labels (and entity order, where the bundle records it) must equal the
                       Phase 6 bundle's on the same fold; shortage is checked against Phase 5's predictions
  2  scoring           baselines (test fold), recalibration arms for the LightGBM-22 fits, model rows from bundles
  3  bands             mean / spread / sd per LightGBM arm per world
  4  drift             the label-free statistic on validation vs test inputs, per baseline and per bundle, and the
                       recalibration asymmetry: removable validation bias vs imported validation->test shift
  5  ablation          guide 7.3 ratio from existing h0 and full-model cells

  python ml/eval/phase7_score.py [--workers 6]
"""
from __future__ import annotations
import os, sys, json, glob, re, time, argparse
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "models")]
import numpy as np
from config import ARTIFACTS

PREDS = os.path.join(ARTIFACTS, "phase7_preds")
BUND = os.path.join(ARTIFACTS, "bundles")
OUT = os.path.join(ARTIFACTS, "phase7_scores.json")
W12 = 12
REF_BUNDLE = {"arrival_week": "{w}_lite_h4_lr0.00025_s7", "fill_rate": "{w}_none_h0_lr0.000125_s7",
              "capacity_strain": "{w}_mp_h4_lr0.00025_s7"}
SHIP = {"arrival_week": ("lite", 4, "0.00025"), "fill_rate": ("none", 0, "0.000125"), "capacity_strain": ("mp", 4, "0.00025")}


def cells13(Y, EV):
    return np.where(np.asarray(EV, bool), np.clip(np.asarray(Y, float), 1, W12).astype(int) - 1, W12)


# ================================================================== 1 row identity
def assert_rows():
    import pandas as pd
    checks = []
    for f in sorted(glob.glob(os.path.join(PREDS, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if name.startswith("RECAL_"):                 # skip BEFORE parsing: "RECAL_v6_..." is not a world prefix
            continue
        w, rest = name.split("_", 1)
        task = next(t for t in ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty") if rest.startswith(t))
        fold = name.rsplit("_", 1)[1]
        z = np.load(f)
        if task in REF_BUNDLE:
            b = os.path.join(BUND, task, REF_BUNDLE[task].format(w=w))
            ref = np.load(os.path.join(b, f"preds_{fold}.npz"))
            ok_y = np.array_equal(z["Y"], ref["Y"])
            ok_e = True
            if fold == "test":
                mo = pd.read_csv(os.path.join(b, "model_outputs.csv.gz"), usecols=["entity_id"]).entity_id.to_numpy().astype(str)
                ok_e = np.array_equal(z["entity"], mo)
            src = "bundle " + os.path.relpath(b, BUND)
        else:
            if fold == "test":
                ref = np.load(os.path.join(ARTIFACTS, "run10_preds", f"{w}_shortage_qty_none_h0_s7.npz"))
                ok_y, ok_e, src = np.array_equal(z["Y"], ref["Y"]), True, "Phase 5 shortage h0 predictions"
            else:
                ok_y, ok_e, src = True, True, "no reference (validation fold, shortage)"
        checks.append(dict(file=name, reference=src, labels_identical=bool(ok_y), entity_order_identical=bool(ok_e), rows=int(len(z["Y"]))))
        assert ok_y and ok_e, f"ROW IDENTITY FAILED for {name} against {src}"
    return checks


# ================================================================== 2 scoring (worker)
def score_job(spec):
    import phase5_metrics as M
    kind, path = spec["kind"], spec["path"]
    z = np.load(path)
    P, Y, EV, AUX = z["P"], z["Y"], z["EV"].astype(bool), z["AUX"]
    if kind == "arrival_dist":
        out = M.arrival_scores(P, Y, EV, AUX, S=z["S"], pT=z["pT"])
    elif kind == "arrival_point":
        out = M.arrival_scores(P, Y, EV, AUX)
    elif kind == "arrival_promise":
        # C-index ranks on the promise week itself; lateness ranks on (constant - promise), i.e. an earlier promise
        # is more likely to be missed -- both through the same scorer, as two calls
        out = M.arrival_scores(P, Y, EV, AUX)
        late = M.arrival_scores(np.full(len(Y), float(spec["const"])), Y, EV, AUX)
        out["roc_auc_late"] = late["roc_auc_late"]
    elif kind in ("cells22", "legacy20"):
        out = M.fill_scores(P, Y, kind)
    elif kind == "quantile":
        out = M.capacity_scores(P, Y)
    else:
        out = M.shortage_scores(P, Y)
    out["_file"] = os.path.basename(path)
    return spec["label"], out


def kind_of(task, name):
    if task == "arrival_week":
        return "arrival_dist" if "marginal13" in name else ("arrival_promise" if name == "promise_only" else "arrival_point")
    if task == "fill_rate":
        return "legacy20" if name.startswith("lgbm20") else "cells22"
    return "quantile" if task == "capacity_strain" else "binary"


# ================================================================== recalibration arms (main process: torch LBFGS)
def recal_arms():
    import loop as L
    out = {}
    for f in sorted(glob.glob(os.path.join(PREDS, "*_fill_rate_*_val.npz"))):
        name = os.path.basename(f)[:-8]
        if not re.search(r"(lgbm22_id_s\d+|b5flat22_s\d+)$", name):
            continue
        zv, zt = np.load(f), np.load(f.replace("_val.npz", "_test.npz"))
        rec = L.fit_recalibration("fill_rate", {"P": zv["P"], "Y": zv["Y"]})
        Q = L.apply_recalibration(rec, "fill_rate", {"P": zt["P"]})["P22"]
        Qv = L.apply_recalibration(rec, "fill_rate", {"P": zv["P"]})["P22"]
        np.savez_compressed(os.path.join(PREDS, f"RECAL_{name}_test.npz"), P=Q, Y=zt["Y"], EV=zt["EV"], AUX=zt["AUX"], entity=zt["entity"])
        np.savez_compressed(os.path.join(PREDS, f"RECAL_{name}_val.npz"), P=Qv, Y=zv["Y"], EV=zv["EV"], AUX=zv["AUX"], entity=zv["entity"])
        out[name] = dict(method=rec["method"], vs_T=rec["vs_T"], val_log_score=rec["val_log_score"])
    return out


# ================================================================== 4 drift + asymmetry
def drift_and_asymmetry():
    import phase5_metrics as M
    from heads import fill_cell
    rows = []
    stat = {"arrival_week": None, "fill_rate": "p_complete", "capacity_strain": "q50_mean", "shortage_qty": "p_mean"}

    def value(task, z, name):
        P = z["P"]
        if task == "fill_rate":
            P22 = P if P.shape[1] == 22 else None
            return None if P22 is None else float(P22[:, 21].mean())
        if task == "arrival_week":
            if "S" in z:
                return float(z["S"][:, -1].mean())                      # P(T > 12), the models' statistic
            return float(np.mean(P))                                    # point baselines: mean predicted week
        if task == "capacity_strain":
            return float(P[:, 1].mean())
        return float(np.mean(P))

    for fv in sorted(glob.glob(os.path.join(PREDS, "*_val.npz"))):
        name = os.path.basename(fv)[:-8]
        w, rest = name.replace("RECAL_", "").split("_", 1)
        task = next(t for t in stat if rest.startswith(t))
        bname = rest[len(task) + 1:]
        zv, zt = np.load(fv), np.load(fv.replace("_val.npz", "_test.npz"))
        v, t = value(task, zv, bname), value(task, zt, bname)
        if v is None:
            continue
        unit = "weeks" if (task == "arrival_week" and "S" not in zv) else ("P(T>12)" if task == "arrival_week" else
                                                                            {"fill_rate": "P(complete)", "capacity_strain": "mean P50",
                                                                             "shortage_qty": "mean P(short)"}[task])
        obs_v = obs_t = None
        if task == "fill_rate":
            obs_v, obs_t = float((zv["Y"] >= 1).mean()), float((zt["Y"] >= 1).mean())
        elif task == "arrival_week":
            obs_v, obs_t = float((~zv["EV"].astype(bool)).mean()), float((~zt["EV"].astype(bool)).mean())
        elif task == "capacity_strain":
            obs_v, obs_t = float(zv["Y"].mean()), float(zt["Y"].mean())
        else:
            obs_v, obs_t = float(zv["Y"].mean()), float(zt["Y"].mean())
        row = dict(world=w, task=task, baseline=("RECAL_" if name.startswith("RECAL_") else "") + bname, statistic=unit,
                   val=v, test=t, drift=(t - v) * (1 if unit == "weeks" else 100), observed_val=obs_v, observed_test=obs_t,
                   observed_shift=(obs_t - obs_v) * (1 if task == "capacity_strain" else 100))
        if task == "fill_rate" and zv["P"].shape[1] == 22:
            cv, ct = fill_cell(zv["Y"]), fill_cell(zt["Y"])
            row.update(val_ece22_bias=float(M.ece_marginal(zv["P"], cv)[0]), test_ece22=float(M.ece_marginal(zt["P"], ct)[0]),
                       val_test_marginal_distance=float(np.abs(np.bincount(cv, minlength=22) / len(cv) - np.bincount(ct, minlength=22) / len(ct)).sum()))
        rows.append(row)

    # bundles -- the same statistics, from their own stored baselines and test values
    for cfgp in sorted(glob.glob(os.path.join(BUND, "*", "*", "config.json"))):
        d = os.path.dirname(cfgp)
        cfg = json.load(open(cfgp)); base = json.load(open(os.path.join(d, "drift_baseline.json")))
        met = json.load(open(os.path.join(d, "metrics.json")))
        tv = met.get("test_drift_values_label_free", {}); bv = base["values"]
        for k in bv:
            if k in tv:
                rows.append(dict(world=cfg["world"], task=cfg["task"], baseline=f"MODEL {os.path.basename(d)}", statistic=k,
                                 val=bv[k], test=tv[k], drift=(tv[k] - bv[k]) * (1 if k == "expected_week_raw" else 100)))
        if cfg["task"] == "fill_rate":
            from heads import fill_cell as fc
            zv, zt = np.load(os.path.join(d, "preds_val.npz")), np.load(os.path.join(d, "preds_test.npz"))
            cv, ct = fc(zv["Y"]), fc(zt["Y"])
            rows.append(dict(world=cfg["world"], task="fill_rate", baseline=f"MODEL {os.path.basename(d)}", statistic="asymmetry",
                             val_ece22_bias=float(M.ece_marginal(zv["P"], cv)[0]), test_ece22=float(M.ece_marginal(zt["P"], ct)[0]),
                             val_test_marginal_distance=float(np.abs(np.bincount(cv, minlength=22) / len(cv) - np.bincount(ct, minlength=22) / len(ct)).sum())))
    return rows


# ================================================================== 3 model rows and bands
def model_rows():
    out = {}
    for task, (arch, depth, lr) in SHIP.items():
        for w in ("v6", "v7"):
            for s in (7, 17, 27):
                d = os.path.join(BUND, task, f"{w}_{arch}_h{depth}_lr{float(lr):g}_s{s}")
                if os.path.exists(os.path.join(d, "metrics.json")):
                    out[f"{w}|{task}|MODEL_s{s}"] = json.load(open(os.path.join(d, "metrics.json")))["test"]
    return out


def bands(S):
    import collections
    groups = collections.defaultdict(list)
    for k, v in S.items():
        w, task, name = k.split("|")
        m = re.match(r"(.*)_(s\d+|r\d+)$", name)
        if m:
            groups[(w, task, m.group(1))].append(v)
    out = {}
    for (w, task, arm), vs in groups.items():
        keys = [mk for mk in vs[0] if not mk.startswith("_") and isinstance(vs[0][mk], (list, tuple))   # scorer returns tuples
                and isinstance(vs[0][mk][0], float)]
        out[f"{w}|{task}|{arm}"] = {mk: dict(n=len(vs), mean=float(np.mean([x[mk][0] for x in vs])),
                                             spread=float(np.ptp([x[mk][0] for x in vs])),
                                             sd=float(np.std([x[mk][0] for x in vs], ddof=1)) if len(vs) > 1 else 0.0,
                                             values=[float(x[mk][0]) for x in vs]) for mk in keys}
    return out


# ================================================================== 5 ablation (guide 7.3)
def ablation(S, MOD):
    P5S = json.load(open(os.path.join(ARTIFACTS, "phase5_scores.json")))
    R10 = json.load(open(os.path.join(ARTIFACTS, "run10_scores.json")))
    rows = []
    for w in ("v6", "v7"):
        full = np.mean([MOD[f"{w}|arrival_week|MODEL_s{s}"]["raw"]["cindex"][0] for s in (7, 17, 27)])
        h0 = json.load(open(os.path.join(BUND, "arrival_week", f"{w}_none_h0_lr0.00025_s7", "metrics.json")))["test"]["raw"]["cindex"][0]
        rows.append(dict(task="arrival_week", world=w, metric="C-index", full=full, h0=h0, random=0.5, ratio=(full - h0) / (full - 0.5)))
        full = np.mean([MOD[f"{w}|capacity_strain|MODEL_s{s}"]["raw"]["pinball_mean"][0] for s in (7, 17, 27)])
        h0s = [P5S[f"{w}|capacity_strain|NEW_quantile_none_h0_s{s}_lr0.00025_g0_w0"]["pinball_mean"][0]
               for s in (7, 17, 27) if f"{w}|capacity_strain|NEW_quantile_none_h0_s{s}_lr0.00025_g0_w0" in P5S]
        h0 = float(np.mean(h0s)); rnd = S[f"{w}|capacity_strain|naive_global"]["pinball_mean"][0]
        rows.append(dict(task="capacity_strain", world=w, metric="mean pinball (lower better)", full=full, h0=h0, random=rnd,
                         ratio=(h0 - full) / (rnd - full), h0_seeds=len(h0s)))
        full = np.mean([MOD[f"{w}|fill_rate|MODEL_s{s}"]["raw"]["crps_exact"][0] for s in (7, 17, 27)])
        rnd = S[f"{w}|fill_rate|naive_global_cdf"]["crps_exact"][0]
        best_graph = min(P5S[k]["crps_exact"][0] for k in P5S if k.startswith(f"{w}|fill_rate|NEW_cdf22_") and re.search(r"_(lite|mp)_h[14]_s7_lr0\.000125_g0_w0$", k))
        rows.append(dict(task="fill_rate", world=w, metric="exact CRPS (lower better)", full=full, h0=full, random=rnd, ratio=0.0,
                         best_graph_variant=best_graph, graph_variant_ratio=(full - best_graph) / (rnd - full)))
        full = R10[f"{w}|shortage_qty|mp|h1|s7"]["pr_auc"][0]; h0 = R10[f"{w}|shortage_qty|none|h0|s7"]["pr_auc"][0]
        rnd = R10[f"{w}|shortage_qty|none|h0|s7"]["base_rate"][0]
        rows.append(dict(task="shortage_qty", world=w, metric="PR-AUC (DIAGNOSTIC)", full=full, h0=h0, random=rnd,
                         ratio=(full - h0) / (full - rnd), note="h0 is a floor (capped at 120 epochs, closeout)"))
    return rows


# ================================================================== Phase 8.2 backtest -- the same scorer, other files
BT_PREDS = os.path.join(ARTIFACTS, "backtest", "preds")
BT_OUT = os.path.join(ARTIFACTS, "backtest", "phase8_scores.json")
BT_NAME = re.compile(r"^(RECAL_|PROMISE_)?(v6|v7)_(arrival_week|fill_rate|capacity_strain|shortage_qty)_(o\d)_(.+)_(val|test)\.npz$")


def backtest_assert_rows():
    global BT_PREDS
    """Every prediction file for one (world, task, origin, fold) must carry identical labels, censoring, promise offset
    and -- where recorded -- entity order. Model bundles and torch-free LightGBM files are checked against each other."""
    groups = {}
    for f in sorted(glob.glob(os.path.join(BT_PREDS, "*.npz"))):
        m = BT_NAME.match(os.path.basename(f))
        assert m, f"unparseable backtest file {f}"
        groups.setdefault((m.group(2), m.group(3), m.group(4), m.group(6)), []).append(f)
    checks = []
    for key, files in sorted(groups.items()):
        ref = np.load(files[0])
        ent_ref = next((np.load(f)["entity"] for f in files if "entity" in np.load(f).files), None)
        for f in files:
            z = np.load(f)
            assert np.array_equal(z["Y"], ref["Y"]), f"labels differ: {f} vs {files[0]}"
            if key[1] == "arrival_week":
                assert np.array_equal(z["EV"].astype(bool), ref["EV"].astype(bool)), f"censoring differs: {f}"
                assert np.allclose(z["AUX"], ref["AUX"], equal_nan=True), f"promise offset differs: {f}"
            if "entity" in z.files and ent_ref is not None:
                assert np.array_equal(z["entity"].astype(str), ent_ref.astype(str)), f"entity order differs: {f}"
        checks.append(dict(group="|".join(key), files=len(files), rows=int(len(ref["Y"])), entity_checked=ent_ref is not None))
    return checks


def backtest_main(workers, preds=None, out=None):
    global BT_PREDS, BT_OUT
    BT_PREDS = preds or BT_PREDS; BT_OUT = out or BT_OUT
    t0 = time.time()
    R = {"row_identity": backtest_assert_rows()}
    print(f"backtest row identity asserted on {sum(c['files'] for c in R['row_identity'])} files in {len(R['row_identity'])} groups", flush=True)
    specs = []
    # test AND validation: 3a asks whether validation would have chosen the depth the evaluation window prefers
    for f in sorted(glob.glob(os.path.join(BT_PREDS, "*.npz"))):
        pre, w, task, o, name, fold = BT_NAME.match(os.path.basename(f)).groups()
        pre = pre or ""
        if fold == "val" and pre == "PROMISE_":
            continue
        kind = ("arrival_promise" if pre == "PROMISE_" else "arrival_point" if "b5flat_reg" in name else "arrival_dist") \
            if task == "arrival_week" else \
               {"fill_rate": "cells22", "capacity_strain": "quantile", "shortage_qty": "binary"}[task]
        spec = dict(kind=kind, path=f, label=f"{w}|{task}|{'VAL_' if fold == 'val' else ''}{pre}{o}_{name}")
        if kind == "arrival_promise":
            spec["const"] = 1.0                                           # a constant: lateness ranks on -promise alone
        specs.append(spec)
    S = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (lab, out) in enumerate(ex.map(score_job, specs), 1):
            S[lab] = out
            if i % 25 == 0:
                print(f"  scored {i}/{len(specs)} ({time.time() - t0:.0f}s)", flush=True)
    R["scores"] = S
    R["bands"] = bands(S)
    assert R["bands"], "bands() returned nothing -- see ml/tests/test_phase7_bands.py"
    json.dump(R, open(BT_OUT, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"DONE: {len(S)} backtest score sets, {len(R['bands'])} bands in {time.time() - t0:.0f}s -> {BT_OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--backtest", action="store_true", help="Phase 8.2: score ml/artifacts/backtest/preds instead")
    ap.add_argument("--backtest-preds", default=None); ap.add_argument("--backtest-out", default=None)
    a = ap.parse_args()
    if a.backtest:
        backtest_main(a.workers, a.backtest_preds, a.backtest_out); sys.exit(0)
    t0 = time.time()
    R = {"row_identity": assert_rows()}
    print(f"row identity asserted on {len(R['row_identity'])} files", flush=True)
    R["recalibration"] = recal_arms()
    print(f"recalibration fitted on {len(R['recalibration'])} fill fits", flush=True)
    specs = []
    for f in sorted(glob.glob(os.path.join(PREDS, "*_test.npz"))):
        name = os.path.basename(f)[:-9]
        recal = name.startswith("RECAL_")
        w, rest = name.replace("RECAL_", "").split("_", 1)
        task = next(t for t in ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty") if rest.startswith(t))
        bname = rest[len(task) + 1:]
        spec = dict(label=f"{w}|{task}|{'recal_' if recal else ''}{bname}", path=f, kind=kind_of(task, bname))
        if spec["kind"] == "arrival_promise":
            spec["const"] = float(np.median(np.load(f.replace("promise_only", "naive_global_median"))["P"]))
        specs.append(spec)
    S = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (label, out) in enumerate(ex.map(score_job, specs)):
            S[label] = out
            if (i + 1) % 10 == 0:
                print(f"  scored {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)", flush=True)
    R["scores"] = S
    MOD = model_rows(); R["models"] = MOD
    R["bands"] = bands(S)
    R["drift"] = drift_and_asymmetry()
    R["ablation"] = ablation(S, MOD)
    json.dump(R, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"DONE: {len(S)} baseline entries scored, {len(MOD)} model rows, {len(R['bands'])} bands, "
          f"{len(R['drift'])} drift rows in {time.time() - t0:.0f}s -> {OUT}", flush=True)
