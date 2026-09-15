"""Phase 4, re-derived on the Phase 5 heads (reports/phase-5.md §10 and Gate).

Phase 4 derived readout depth on the OLD heads. The hazard head added more at h0 than Phase 4's whole
h0 -> h1 step did, and capacity has never been measured at any depth, so both are derived again here:

  capacity_strain first, then arrival_week
  encoders: HeteroMP and SHARE-lite (SHARE-lite matched full SHARE in all four Phase 4 arrival cells
            at 64% of the parameters, so full SHARE is not re-run)
  depths:   h1 and h4; h0 is the Phase 5 cell already on disk
  rate:     each task's Phase 5 validation-selected rate, frozen -- as Phase 4 froze per-task rates
  inputs:   base (no staleness gate, no reconstructed feature), per the Phase 5 ship decision

Fill runs LAST and separately (`--tasks fill_rate`): Phase 5 §6.6 required its calibration fix first,
and Addendum A has since run it. The harness now saves validation predictions for every cell, so each
fill depth cell is recalibrated by phase5_recal.py before depth is compared.

Waits until the four Phase 5 recalibration re-trains are on disk, so the queue runs after them.
Grids are named phase5_grid_p4_*.json, so phase5_score.py scores these cells automatically under
distinct labels (arch and depth are in the label).
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import phase5_heads as P5

CONFIGS = [("lite", 1), ("mp", 1), ("lite", 4), ("mp", 4)]
RECAL_GRIDS = {"recal_grid_A.json": 2, "recal_grid_B.json": 2}


def recal_done():
    for g, n in RECAL_GRIDS.items():
        p = os.path.join(P5.ARTIFACTS, g)
        if not os.path.exists(p) or len(json.load(open(p))) < n:
            return False
    return True


RATES = os.path.join(P5.ARTIFACTS, "phase4r_rates.json")


def main(tasks, worlds, poll=60, lr_override=None, reason=""):
    while not recal_done():
        time.sleep(poll)
    print("[p4] recalibration re-trains are on disk; starting", flush=True)
    for task in tasks:
        sel = json.load(open(os.path.join(P5.ARTIFACTS, f"phase5_selection_{task}.json")))[task]
        lr = sel["chosen"] if lr_override is None else lr_override
        grid = os.path.join(P5.ARTIFACTS, f"phase5_grid_p4_{task.split('_')[0]}.json")
        rates = json.load(open(RATES)) if os.path.exists(RATES) else {}
        rates[task] = dict(lr=lr, phase5_selected=sel["chosen"], override=lr_override is not None, reason=reason)
        json.dump(rates, open(RATES, "w"), indent=1)
        print(f"[p4] {task}: rate {lr:g} ({'override: ' + reason if lr_override is not None else 'frozen Phase 5 selection'})", flush=True)
        if lr_override is not None:
            # the h0 reference must sit at the SAME rate as h1 / h4; it goes into the Phase 5 grid, where an
            # existing sweep cell at this rate is skipped rather than re-trained
            p5grid = os.path.join(P5.ARTIFACTS, f"phase5_grid_{task.split('_')[0]}.json")
            for w in worlds:
                P5.run(w, task, lr, 7, "none", 0, False, False, "p4-h0-ref", p5grid, 120, 8)
        for arch, depth in CONFIGS:
            for w in worlds:
                P5.run(w, task, lr, 7, arch, depth, False, False, f"p4-{arch}-h{depth}", grid, 120, 8)
    print("[p4] QUEUE DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--worlds", default="v6,v7")
    ap.add_argument("--lr", type=float, default=None, help="override the frozen Phase 5 rate; writes phase4r_rates.json")
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    main(a.tasks.split(","), a.worlds.split(","), lr_override=a.lr, reason=a.reason)
