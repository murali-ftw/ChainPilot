"""Phase 4 re-derivation — why fill's graph cells fail on 2025, and a label-free check that sees it coming.

Measured on the h1 cells: on the 2024 validation fold the graph cells are as well calibrated as h0 or better,
and validation prefers them; on 2025 test their mean predicted P(complete) jumps ~5pp while h0's barely moves,
and a validation-fitted recalibration cannot undo a shift that only exists in 2025.

Two kinds of column, kept apart on purpose:

  LABEL-FREE  mean prediction on validation inputs vs on test inputs. Needs no 2025 labels, so it is a
              legitimate deployment-time monitor: a model whose predictions move far more than h0's under the
              same input change is amplifying covariate shift.
  LABELLED    observed rates, ECE, CRPS on test -- the check the label-free signal is compared against.

Reads validation / test predictions for h0 (the Phase 5 re-train) and every fill depth cell; prints a table and
writes ml/artifacts/phase4r_fill_drift.json.
"""
from __future__ import annotations
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "models")]
import numpy as np
import phase5_metrics as M
from heads import fill_cell, fill_to_legacy
from config import ARTIFACTS

R = json.load(open(os.path.join(ARTIFACTS, "phase5_recal_scores.json")))
LR = "0.000125"


def cells(w):
    out = {"h0": os.path.join(ARTIFACTS, "phase5_preds_recal", f"{w}_fill_rate_cdf22_none_h0_s7_lr{LR}_g0_w0")}
    for arch in ("lite", "mp"):
        for d in (1, 4):
            out[f"{arch} h{d}"] = os.path.join(ARTIFACTS, "phase5_preds", f"{w}_fill_rate_cdf22_{arch}_h{d}_s7_lr{LR}_g0_w0")
    return {k: v for k, v in out.items() if os.path.exists(v + "_val.npz") and os.path.exists(v + ".npz")}


rows = []
print("| world | cell | pred P(complete) val → test (LABEL-FREE) | drift | pred interior val → test (LABEL-FREE) | "
      "obs P(complete) val → test | test ECE 20 raw | test ECE 20 recal | test CRPS exact |")
print("|---|---|---|---|---|---|---|---|---|")
for w in ("v6", "v7"):
    for name, base in cells(w).items():
        zv, zt = np.load(base + "_val.npz"), np.load(base + ".npz")
        Pv, Yv, Pt, Yt = zv["P"].astype(float), zv["Y"], zt["P"].astype(float), zt["Y"]
        e = R.get(os.path.basename(base), {})
        sel = e.get("selected_on_validation")
        rec = e[sel]["test"]["ece20"][0] if sel else float("nan")
        row = dict(world=w, cell=name,
                   pred_complete_val=float(Pv[:, 21].mean()), pred_complete_test=float(Pt[:, 21].mean()),
                   pred_interior_val=float(Pv[:, 1:21].sum(1).mean()), pred_interior_test=float(Pt[:, 1:21].sum(1).mean()),
                   obs_complete_val=float(np.mean(Yv >= 1)), obs_complete_test=float(np.mean(Yt >= 1)),
                   ece20_test_raw=float(M.ece_marginal(fill_to_legacy(Pt), M.legacy_bin(Yt))[0]),
                   ece20_test_recal=float(rec), recal_method=sel,
                   crps_exact_test=float(M.crps_exact_rows(Pt, Yt).mean()))
        row["drift_complete_pp"] = 100 * (row["pred_complete_test"] - row["pred_complete_val"])
        rows.append(row)
        print(f"| {w} | {name} | {row['pred_complete_val']:.4f} → {row['pred_complete_test']:.4f} | "
              f"**{row['drift_complete_pp']:+.1f} pp** | {row['pred_interior_val']:.4f} → {row['pred_interior_test']:.4f} | "
              f"{row['obs_complete_val']:.4f} → {row['obs_complete_test']:.4f} | {row['ece20_test_raw']:.4f} | "
              f"{row['ece20_test_recal']:.4f} ({sel}) | {row['crps_exact_test']:.4f} |")
json.dump(rows, open(os.path.join(ARTIFACTS, "phase4r_fill_drift.json"), "w"), indent=1)
