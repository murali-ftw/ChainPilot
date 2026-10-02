"""Phase 17 B3 scorer -- 5-band ordinal fill head (B3b, v8_none_h0_lr0.000125_s{s}_headband5) against the stored
incumbent 22-cell head (B3a, v8_none_h0_lr0.000125_s{s}). RAW and RECALIBRATED are separate tables (deviation 80); each
arm's recalibration is the one loop.finish_bundle fitted on that bundle's own validation fold.

Metrics on TEST, 5-seed [min, mean, max]: exact CRPS, interior sum|err| (the 20 interior cells: |mean predicted mass -
observed frequency|, summed), ECE-22, P(fill = 1) ROC-AUC, and predicted vs actual frequency PER BAND
{0} | (0, .50) | [.50, .95) | [.95, 1) | {1} -- an aggregate that improves while the same bands dominate has fixed nothing.

  python ml/eval/phase17_b3.py
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np
import loop as L, phase5_metrics as M
from heads import fill_cell, FillBand5Head
from sklearn.metrics import roc_auc_score

BF = os.path.join(C.BUND, "fill_rate")
BAND_OF_CELL = np.array(FillBand5Head.BAND_OF_CELL)
BANDS = ("{0}", "(0,.50)", "[.50,.95)", "[.95,1)", "{1}")


def metrics(P, y):
    c = fill_cell(y)
    bands_pred = np.zeros((len(P), 5)); np.add.at(bands_pred.T, BAND_OF_CELL, P.T)
    bp, bo = bands_pred.mean(0), np.bincount(BAND_OF_CELL[c], minlength=5) / len(c)
    cp, co = P.mean(0), np.bincount(c, minlength=22) / len(c)
    return dict(crps_exact=float(M.crps_exact_rows(P, y).mean()), ece22=float(M.ece_marginal(P, c)[0]),
                interior_sum_abs_err=float(np.abs(cp[1:21] - co[1:21]).sum()),
                p_full_roc_auc=float(roc_auc_score((y >= 1).astype(int), P[:, 21])),
                band_pred=bp.tolist(), band_obs=bo.tolist(), band_ratio=(bp / np.maximum(bo, 1e-12)).tolist(),
                cell20_ratio=float(cp[20] / max(co[20], 1e-12)))


def main():
    st = C.require_clean()
    out = dict(stamp=st, bands=BANDS, arms={})
    for arm, suf in (("B3a_incumbent_cells22", ""), ("B3b_band5_ordinal", "_headband5")):
        per = {"raw": [], "recal": []}; seeds = []
        for s in C.V8_SEEDS:
            d = os.path.join(BF, f"v8_none_h0_lr0.000125_s{s}{suf}")
            if not (os.path.exists(os.path.join(d, "config.json")) and json.load(open(os.path.join(d, "config.json"))).get("complete")):
                continue
            z = dict(np.load(os.path.join(d, "preds_test.npz"))); rec = json.load(open(os.path.join(d, "recalibration.json")))
            per["raw"].append(metrics(z["P"].astype(float), z["Y"]))
            per["recal"].append(metrics(L.apply_recalibration(rec, "fill_rate", z)["P22"], z["Y"]))
            seeds.append(s)
        a = dict(seeds=seeds)
        for cal, rows in per.items():
            if not rows:
                continue
            band = lambda k: [float(min(r[k] for r in rows)), float(np.mean([r[k] for r in rows])), float(max(r[k] for r in rows))]
            a[cal] = {k: band(k) for k in ("crps_exact", "ece22", "interior_sum_abs_err", "p_full_roc_auc", "cell20_ratio")}
            a[cal]["band_pred_mean"] = np.mean([r["band_pred"] for r in rows], 0).tolist()
            a[cal]["band_obs"] = rows[0]["band_obs"]
            a[cal]["band_ratio_mean"] = np.mean([r["band_ratio"] for r in rows], 0).tolist()
        out["arms"][arm] = a
    A, B = out["arms"]["B3a_incumbent_cells22"], out["arms"]["B3b_band5_ordinal"]
    if B["seeds"]:
        v = {}
        for cal in ("raw", "recal"):
            for k, lower_better in (("crps_exact", True), ("ece22", True), ("interior_sum_abs_err", True), ("p_full_roc_auc", False)):
                a_, b_ = A[cal][k], B[cal][k]
                better = (b_[2] < a_[0]) if lower_better else (b_[0] > a_[2])
                worse = (b_[0] > a_[2]) if lower_better else (b_[2] < a_[0])
                v[f"{cal}|{k}"] = "GAIN (disjoint)" if better else "WORSE (disjoint)" if worse else "UNDETERMINED (bands overlap)"
        out["verdict_vs_B3a"] = v
    C.dump(out, "phase17/b3_score.json")
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
