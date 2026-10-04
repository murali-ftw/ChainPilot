"""Phase 21 -- VALIDATION-ONLY choices (pre-registration D6, D9, D11). Torch-free, LightGBM-free. Reads no test row.

  k (shrinkage constant), one per task, from K_GRID = {1, 3, 10, 30, 100, 300}:
    arrival  max validation lateness AUC (as-of channel reference, docs/specs/lateness_metric.md) of the standalone rule
    place    max validation at-placement lateness AUC (score = shrunk KM median lead - contracted lead)
    fill     min validation exact CRPS of the standalone shrunk L5-chain histogram
  a (offset, weeks) of each standalone arrival rule = median(Y - rule) over validation observed rows, at the chosen k
  T (thin / normal threshold) = 20th percentile of n4 over validation rows with n4 > 0 (snapshot rows; placement rows)

  python ml/eval/phase21_select.py [--world v8]     -> ml/artifacts/phase21/k_select_{world}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config, folds
import phase21_paths as PP
import grpstats as GS
import phase5_metrics as M
from phase11b_lateness import build_reference
from metrics import roc_auc

assert "torch" not in sys.modules and "lightgbm" not in sys.modules


def labels(world, task):
    """phase7_fit.labels reproduced (that module imports lightgbm): same filter, merge, order and columns."""
    D = config.WORLDS[world]
    lb = pd.read_csv(D + "/training_labels.csv", usecols=["snapshot_date", "entity_id", "task", "label_value", "label_censored"])
    lb = lb[lb.task == task].copy()
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    lb = lb[(lb.snapshot_date >= config.FIT_WINDOW[0]) & (lb.snapshot_date <= config.FIT_WINDOW[1])]
    lb["label_censored"] = lb.label_censored.astype(str).str.lower().isin(["true", "1"])
    lb = lb.merge(pd.read_csv(D + "/po_lines.csv", usecols=["po_line_id", "channel_id"]), left_on="entity_id",
                  right_on="po_line_id", how="inner")
    lb["key"] = lb.channel_id
    return lb.reset_index(drop=True)


def ordered(dates, mask):
    idx = np.flatnonzero(mask)
    return idx[np.argsort(np.asarray(dates)[idx], kind="stable")]


def late_auc(score, Y, EV, R):
    m = EV & np.isfinite(R)
    return float(roc_auc((Y[m] > R[m]).astype(int), score[m] - R[m]))


def snapshot_choices(world, Z):
    lb = labels(world, "arrival_week")
    assert (lb.entity_id.to_numpy().astype(str) == Z["entity"]).all()
    tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(lb.snapshot_date)]
    R, rmeta = build_reference("asof_channel_lead", world, lb, tr)
    o = ordered(lb.snapshot_date.values, va)
    Y, EV, Rv = lb.label_value.to_numpy(float)[o], ~lb.label_censored.to_numpy(bool)[o], R[o]
    Zv = {k: v[o] for k, v in Z.items() if hasattr(v, "shape") and v.shape[:1] == (len(lb),)}
    per = {}
    for k in GS.K_GRID:
        w = GS.standalone_arrival_weeks(Zv, k)
        a = float(np.median(Y[EV] - w[EV]))
        per[k] = dict(val_lateness_auc=late_auc(w + a, Y, EV, Rv), offset_a_weeks=a,
                      val_a3_days=float(np.median(np.abs(7 * (w + a)[EV] - 7 * Y[EV]))))
    k_arr = max(per, key=lambda k: per[k]["val_lateness_auc"])
    # fill
    lf = labels(world, "fill_rate")
    assert (lf.entity_id.to_numpy() == lb.entity_id.to_numpy()).all()
    yf = lf.label_value.to_numpy(float)[o]
    perf = {}
    for k in GS.K_GRID:
        P = GS.standalone_fill(Zv, k)
        perf[k] = dict(val_crps=float(M.crps_exact_rows(P, yf).mean()))
    k_fill = min(perf, key=lambda k: perf[k]["val_crps"])
    n4 = np.nan_to_num(Zv["A_L4"][:, GS.ASTATS.index("n")])
    T = float(np.percentile(n4[n4 > 0], 20))
    return dict(arrival=per, fill=perf, k_arrival=k_arr, k_fill=k_fill, T_snap=T, reference_meta=rmeta)


def placement_choices(world):
    Z, _ = GS.load(world, "place")
    src = GS.Source(world)
    li = Z["line"].astype(np.int64)
    Y, EV, cw, _ = GS.placement_labels(src, li)
    created = pd.DatetimeIndex(src.created[li])
    tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(created)]
    o = ordered(pd.to_datetime(Z["tau"]).values, va)
    Zv = {k: v[o] for k, v in Z.items() if hasattr(v, "shape") and v.shape[:1] == (len(li),)}
    per = {}
    for k in GS.K_GRID:
        w = GS.standalone_arrival_weeks(Zv, k)
        a = float(np.median(Y[o][EV[o]] - w[EV[o]]))
        per[k] = dict(val_lateness_auc_at_placement=late_auc(w, Y[o], EV[o], cw[o]), offset_a_weeks=a,
                      val_a3_days=float(np.median(np.abs(7 * (w + a)[EV[o]] - 7 * Y[o][EV[o]]))))
    k_pl = max(per, key=lambda k: per[k]["val_lateness_auc_at_placement"])
    n4 = np.nan_to_num(Zv["A_L4"][:, GS.ASTATS.index("n")])
    return dict(place=per, k_place=k_pl, T_place=float(np.percentile(n4[n4 > 0], 20)),
                rows=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
                observed_share=dict(train=float(EV[tr].mean()), val=float(EV[va].mean()), test=float(EV[te].mean())))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="v8"); ap.add_argument("--no-place", action="store_true")
    a = ap.parse_args()
    PP.register()
    st = C.require_clean()
    Z, info = GS.load(a.world, "snap")
    out = dict(stamp=st, world=a.world, grid=list(GS.K_GRID), builder=dict(commit=info["stamp"]["code_commit"],
                                                                          config_hash=info["config_hash"]))
    out["snap"] = snapshot_choices(a.world, Z)
    out["k"] = dict(arrival=out["snap"]["k_arrival"], fill=out["snap"]["k_fill"])
    if not a.no_place:
        out["placement"] = placement_choices(a.world)
        out["k"]["place"] = out["placement"]["k_place"]
    C.dump(out, f"phase21/k_select_{a.world}.json")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
