"""Training-fold predictions from a saved Phase 5 checkpoint — no re-training.

For each row of the given grid files, rebuilds the cell's model, loads the restored best weights the
harness saved beside its predictions (`<preds>.pt`), predicts the TRAINING fold, and writes
`<preds>_train.npz`. Used by phase5_recal.py's train + validation diagnostic fit. Test data is not read.
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import numpy as np, torch
import phase5_heads as P5
import temporal_share as TS

ap = argparse.ArgumentParser()
ap.add_argument("--grids", required=True, help="comma list of grid json files")
a = ap.parse_args()

for g in a.grids.split(","):
    for r in json.load(open(g)):
        f = r["preds"]; ck = f.replace(".npz", ".pt"); out = f.replace(".npz", "_train.npz")
        if os.path.exists(out):
            print(f"[skip] {os.path.basename(out)}", flush=True); continue
        assert os.path.exists(ck), f"no checkpoint beside {f}"
        lb = P5.labels(r["world"], r["task"])
        tr, va, te = TS.fold(lb.snapshot_date)
        D = P5.device_inputs(r["world"], np.sort(lb.snapshot_date[tr].unique()), bool(r["wsla"]))
        model = P5.HeadNet(D["X"].shape[2], r["task"], r["arch"], r["depth"],
                           gate_cols=D["gate_cols"] if r["gate"] else None,
                           fill_loss=r.get("fill_loss") or "rps").to(P5.DEV)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        pr = P5.predict(model, D, lb, tr, r["ymu"], r["ysd"])
        np.savez_compressed(out, **{k: v for k, v in pr.items() if not k.startswith("_")})
        print(f"  {os.path.basename(out)}  rows {len(pr['Y']):,}", flush=True)
print("DONE", flush=True)
