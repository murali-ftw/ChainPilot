"""Phase 20 Stage 2b -- base rates of the second world beside v8-1001's (labels only; each world's own as-of reference).

Per world and fold (train / val / test): arrival late share (Y > R on uncensored rows, R = that world's as-of channel
reference) and censoring share; P(fill = 1); capacity strain > 1 share. A difference of more than 5 points is FLAGGED: it
changes how absolute numbers (precision, lift) compare across the two worlds.

  python ml/eval/phase20_baserates.py     # -> ml/artifacts/phase20/baserates.json
"""
from __future__ import annotations
import json
import phase12_common as C
import numpy as np
import phase20_world as PW
import phase5_heads as P5, folds
from phase11b_lateness import build_reference


def rates(world):
    out = {}
    lb = P5.labels(world, "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    R, meta = build_reference("asof_channel_lead", world, lb, tr)
    y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
    for f, m in (("train", tr), ("val", va), ("test", te)):
        out.setdefault(f, {})["arrival_late_share"] = float((y[m & ev] > R[m & ev]).mean())
        out[f]["arrival_censored_share"] = float((~ev[m]).mean())
    out["reference_offset_c"] = meta["offset_c"]
    lb = P5.labels(world, "fill_rate"); tr, va, te = folds.fixed_split(lb.snapshot_date); y = lb.label_value.to_numpy(float)
    for f, m in (("train", tr), ("val", va), ("test", te)):
        out[f]["p_fill_eq_1"] = float((y[m] >= 1).mean()); out[f]["fill_lt_0.95"] = float((y[m] < 0.95).mean())
    lb = P5.labels(world, "capacity_strain"); tr, va, te = folds.fixed_split(lb.snapshot_date); y = lb.label_value.to_numpy(float)
    for f, m in (("train", tr), ("val", va), ("test", te)):
        out[f]["strain_gt_1"] = float((y[m] > 1).mean())
    return out


def main():
    st = C.require_clean(); PW.register()
    r = {"v8": rates("v8"), PW.WORLD: rates(PW.WORLD)}
    flags = {}
    for f in ("train", "val", "test"):
        for k, v in r["v8"][f].items():
            d = r[PW.WORLD][f][k] - v
            if abs(d) > 0.05:
                flags[f"{f}|{k}"] = round(d, 4)
    out = dict(stamp=st, rates=r, flagged_over_5_points=flags)
    C.dump(out, "phase20/baserates.json"); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
