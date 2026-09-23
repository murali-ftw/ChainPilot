"""Phase 1 on v8 — score the trained bundles on the fixed split's test fold, with 3-seed bands.

Reuses the existing machinery: loop.py's bundle loading / materialisation / recalibration and
phase5_metrics' scorers. Nothing is re-implemented, and no metric definition is changed --
fill's CRPS is the point-mass-aware `crps_exact_rows`, never the legacy formula.

ARRIVAL, per the corrected rule (reports/v8-clearance.md deviation 58): B9 did not clear, so the
promise-date C-index comparison is RETIRED. The head's C-index is reported as its own number.
The promise date's own C-index is computed but carried under an explicit `PRIVILEGED__` prefix,
because it reads a line raised ~7 weeks after the forecast instant and is not a fair comparison
in either direction. Lateness ROC-AUC is reported and its reference is likewise privileged.

    python ml/eval/phase1_v8_score.py --world v8 [--json out.json]
"""
from __future__ import annotations
import os, sys, json, glob, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "train")]
import numpy as np, pandas as pd, torch
import loop as LP
import phase5_heads as P5
import phase5_metrics as M
from metrics import cindex
from config import ARTIFACTS

DEV = P5.DEV


def raw_and_recal(bundle_path, fold="test"):
    """One forward pass over the fold, keeping BOTH the raw and the recalibrated distribution."""
    B = LP.load_bundle(bundle_path); cfg = B["cfg"]; task = cfg["task"]
    lb = P5.labels(cfg["world"], task, row_features=bool(cfg.get("row_features")))
    tr, va, te = LP.split_of(cfg, lb.snapshot_date)
    mask = {"test": te, "val": va}[fold]
    D = P5.device_inputs(cfg["world"], np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"])
    assert np.allclose(D["norm_mu"], B["norm"]["mu"]) and np.allclose(D["norm_sd"], B["norm"]["sd"]), \
        "the rebuilt normaliser does not match the bundle's -- inputs changed since training"
    model = LP._materialise(B, D)
    ymu, ysd = 0.0, 1.0
    if task == "capacity_strain":
        tl = json.load(open(os.path.join(bundle_path, "train_log.json")))
        ymu, ysd = tl["ymu"], tl["ysd"]
    order = P5.ordered(lb, mask); dates = lb.snapshot_date.values[order]
    keymap = D["W"]["pp_uniq"] if task == "shortage_qty" else D["W"]["cidx"]
    raw, rec = [], []
    with torch.no_grad():
        for s in np.unique(dates):
            ii = order[dates == s]
            idx = torch.from_numpy(lb.key.iloc[ii].map(keymap).to_numpy(np.int64)).to(DEV)
            pr = LP._head_outputs(task, P5.forward(model, D, LP.P5.t0_of(D["W"], s), idx), ymu, ysd)
            raw.append(pr); rec.append(LP.apply_recalibration(B["rec"], task, pr))
    cat = lambda lst, k: np.concatenate([x[k] for x in lst])
    rows = lb.iloc[order]
    return dict(cfg=cfg, rows=rows, raw=raw, rec=rec, cat=cat)


def score_bundle(bundle_path):
    o = raw_and_recal(bundle_path)
    cfg, rows, raw, rec, cat = o["cfg"], o["rows"], o["raw"], o["rec"], o["cat"]
    task = cfg["task"]
    y = rows.label_value.to_numpy(float)
    cen = rows.label_censored.to_numpy(bool)
    out = dict(task=task, world=cfg["world"], arch=cfg["arch"], depth=cfg["depth"],
               lr=cfg["lr"], seed=cfg["seed"], n=int(len(y)), bundle=bundle_path)

    if task == "arrival_week":
        ET = cat(raw, "P"); S = cat(raw, "S"); pT = cat(raw, "pT")
        aux = rows.promise_week.to_numpy(float)
        sc = M.arrival_scores(ET, y, ~cen, aux, S=S, pT=pT, n_boot=200)
        out["cindex"] = sc["cindex"][0]
        out["roc_auc_late"] = sc["roc_auc_late"][0]
        out["roc_auc_late_ptail"] = sc.get("roc_auc_late_ptail", (None,))[0]
        out["ece_week"] = sc.get("ece_week", (None,))[0]
        out["late_rate"] = sc["late_rate"][0]
        out["censored_pct"] = 100.0 * float(cen.mean())
        # PRIVILEGED -- not a fair comparison, never printed as a model result
        m = (~cen) & np.isfinite(aux)
        out["PRIVILEGED__promise_cindex"] = cindex(aux, y, ~cen)
        out["PRIVILEGED__note"] = ("the promise date belongs to a line raised ~7 weeks AFTER the "
                                   "forecast instant (B9 = 0%); retired as a comparison")
        # leak tripwire for S2.3: the ranking score must not be a function of the label
        out["diag_distinct_predictions"] = int(len(np.unique(np.round(ET, 6))))
        out["diag_corr_pred_label"] = float(np.corrcoef(ET, y)[0, 1])

    elif task == "fill_rate":
        P_raw = cat(raw, "P"); P_rec = cat(rec, "P22")
        s_raw = M.fill_scores(P_raw, y, "cells22", n_boot=200)
        s_rec = M.fill_scores(P_rec, y, "cells22", n_boot=200)
        out["crps_exact_raw"] = s_raw["crps_exact"][0]
        out["crps_exact_recal"] = s_rec["crps_exact"][0]
        out["ece22_raw"] = s_raw["ece22"][0]
        out["ece22_recal"] = s_rec["ece22"][0]
        out["p_complete_pred"] = s_rec["p_complete_pred"][0]
        out["p_complete_obs"] = s_rec["p_complete_obs"][0]

    elif task == "capacity_strain":
        Q = cat(raw, "P")
        sc = M.capacity_scores(Q, y, n_boot=200)
        for k in ("pinball_mean", "pinball_10", "pinball_50", "pinball_90",
                  "coverage80", "exceed_p90", "below_p10", "crossing_rate", "mae_p50"):
            out[k] = sc[k][0]

    else:  # shortage_qty
        P = cat(raw, "P").ravel()
        Y = (y > 0).astype(int)
        sc = M.shortage_scores(P, Y, n_boot=200)
        out["roc_auc"] = sc["roc_auc"][0]
        out["pr_auc"] = sc["pr_auc"][0]
        out["base_rate"] = sc["base_rate"][0]
        out["LEVEL_CAVEAT"] = ("v8's positive rate is ~23.6% against a real operating base rate "
                               "near 3%, and event rates run 4x-6x over Rane's stated bands "
                               "(deviation 56). ROC-AUC is a RANKING metric and is unaffected; "
                               "no shortage PROBABILITY from v8 is calibrated in level.")
    return out


def band(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)),
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n_seeds=len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v8")
    ap.add_argument("--json", default=None)
    ap.add_argument("--bundles", default=os.path.join(ARTIFACTS, "bundles"))
    a = ap.parse_args()
    # bundles/{task}/{world}_{arch}_h{depth}_lr{lr}_s{seed}/
    paths = sorted(os.path.dirname(c) for c in
                   glob.glob(os.path.join(a.bundles, "*", "*", "config.json")))
    cells, failed = [], []
    for p in paths:
        cfgp = os.path.join(p, "config.json")
        if not os.path.exists(cfgp) or not json.load(open(cfgp)).get("complete"):
            continue
        if json.load(open(cfgp)).get("world") != a.world:
            continue
        try:
            c = score_bundle(p)
            cells.append(c)
            print(f"  scored {c['task']:16s} h{c['depth']} s{c['seed']:<3}", flush=True)
        except Exception as e:                                      # noqa: BLE001
            failed.append(dict(bundle=p, error=f"{type(e).__name__}: {e}"))
            print(f"  FAILED {p}: {type(e).__name__}: {e}", flush=True)
    # 3-seed bands per (task, arch, depth)
    bands = {}
    for c in cells:
        k = f"{c['task']}|{c['arch']}|h{c['depth']}"
        bands.setdefault(k, {"cells": [], "seeds": []})
        bands[k]["cells"].append(c); bands[k]["seeds"].append(c["seed"])
    agg = {}
    for k, g in bands.items():
        metrics = [m for m in g["cells"][0] if isinstance(g["cells"][0][m], (int, float))
                   and m not in ("seed", "depth", "lr", "n")]
        agg[k] = {"n_seeds": len(g["cells"]), "seeds": sorted(g["seeds"]),
                  "n_test": g["cells"][0]["n"]}
        for m in metrics:
            b = band([c.get(m) for c in g["cells"]])
            if b:
                agg[k][m] = b
    R = dict(world=a.world, cells=cells, bands=agg, failed=failed)
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1, default=str)
        print(f"\n  json -> {a.json}")
    print(f"\n  {len(cells)} cells scored, {len(failed)} failed, {len(agg)} configurations")
    for k, v in sorted(agg.items()):
        print(f"    {k:34s} {v['n_seeds']} seeds  n_test {v['n_test']:,}")
    return R


if __name__ == "__main__":
    main()
