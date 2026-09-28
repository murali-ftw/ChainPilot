"""Phase 13 F1 arm 5: select the boundary weight ONCE, on RAW VALIDATION, then freeze it before any test number.

Grid w in {3, 10, 30} (w = 1 is arm 1 itself), seed 7. Criterion: raw validation interior sum|error| -- the failure
under test ([0.95, 1) over-predicted 2.9x). Exact CRPS is reported beside it; it is nearly blind to interior allocation
(Phase 5 section 6.4), so it cannot be the selector for a fix aimed at interior allocation.
"""
import os, json
import phase12_common as C
import numpy as np
from phase13_score import metrics

out = dict(stamp=C.require_clean(), criterion="raw VALIDATION interior sum|error|, seed 7", grid={})
for w in (3, 10, 30):
    d = os.path.join(C.BUND, "fill_rate", f"v8_none_h0_lr0.000125_s7_lossrps_bw{w}")
    z = np.load(os.path.join(d, "preds_val.npz"))
    m = metrics(z["P"], z["Y"])
    out["grid"][f"rps_bw{w}"] = {k: m[k] for k in ("interior_abs_err", "crps_exact", "cell20_ratio", "ece22")}
z = np.load(os.path.join(C.BUND, "fill_rate", "v8_none_h0_lr0.000125_s7", "preds_val.npz"))
m = metrics(z["P"], z["Y"]); out["grid"]["rps (w=1, arm 1, seed 7)"] = {k: m[k] for k in ("interior_abs_err", "crps_exact", "cell20_ratio", "ece22")}
cand = {k: v for k, v in out["grid"].items() if k.startswith("rps_bw")}
out["selected_fill_loss"] = min(cand, key=lambda k: cand[k]["interior_abs_err"])
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(C.ART, "phase13_f1_bw_selection.json"), "w"), indent=1, default=str)
