#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 0 FOLLOW-UP — re-test the Markov readout DEPTH for `shortage` and `delay`,
without retraining SHARE and without modifying it.

`ml/layer3_baseline.py` froze the baseline at the retained depths (`delay -> h^1`,
`shortage -> h^3`, `impact -> h^4`) and found `delay` unproven (+0.0357 against a 0.0874 floor)
and `shortage` actively net-negative against features alone (-0.0856). The obvious next
question is whether the retained depth is the reason. This module answers it for three swaps:

    shortage @ h^2   (retained: h^3)
    delay    @ h^2   (retained: h^1)
    delay    @ h^3   (retained: h^1)

over the SAME 5 dataset seeds x 5 model-init seeds grid, from the SAME cached checkpoints
(`out/ds_ckpt/`), so every number is directly comparable to the frozen Phase 0 table.

**Nothing here trains, modifies or touches SHARE.** `get_backbone()` loads the cached
checkpoint and `freeze()`/`assert_backbone_frozen()` assert every encoder parameter is frozen.
SHARE emits the full `{h^0..h^4}` for every node in one forward pass; both arms below only
change *which index of that already-computed list* is read.

**Two readout arms, because "index-select" is only half the readout.**

  ARM A — `frozen_head`. The literal zero-parameter swap: `model.heads[task]` is applied to
  `layers[d][entity_type]` instead of `layers[retained][entity_type]`. Nothing is fitted at
  all. This is exactly what the brief describes, and it is reported first. Its caveat has to
  be stated rather than buried: that `PredictionHead` was trained *jointly with the encoder*
  against the retained depth's activations, so feeding it a different depth measures how well
  one depth's head transfers to another depth's geometry, which is not the same question as
  whether the other depth carries the signal. A collapse here is evidence about head transfer,
  not about depth.

  ARM B — `refit_head`. The depth question asked cleanly, still with zero SHARE retraining:
  the frozen `h^d` is handed to `ml/latent_state_head.py::train_head`, the SAME `in -> 32 -> 1`
  MLP, the same positive-weighted BCE, the same 120 epochs, the same seeding by model-init
  seed, that `ml/layer3_baseline.py::px_auc` already uses for the `P(Y|X)` arm. Because both
  arms of the gain then share one readout protocol, `mean AUC(h^d) - mean AUC(X)` isolates the
  representation and nothing else -- which Phase 0's own `P(Y|X,H)` (a jointly-trained
  `PredictionHead` under focal loss) did not do. Arm B is also computed at each task's
  RETAINED depth, so "does h^2 beat h^1" can be read within a single protocol.

**Floors are `ml/layer3_baseline.py::floors`, imported, not reimplemented** --
`max(init-seed floor, dataset-seed floor)`, cells that are `None` dropped rather than imputed.

`P(Y|X)` is not recomputed: it is read from `out/layer3_v3/phase0_baseline.json`'s per-cell
grid, because features-only does not depend on depth and reusing the frozen numbers keeps the
comparison exact rather than merely equivalent.

    python3 ml/layer3_depth_swap.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world          # noqa: E402
from ml.latent_state_head import (                           # noqa: E402
    assert_backbone_frozen, train_head)
from ml.models.depth import TASK_ENTITY_TYPE                 # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH   # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")

# The three swaps the brief asks for, plus each task's retained depth as the within-arm
# reference point. `impact` is untouched: Phase 0 already cleared its floor at h^4.
SWAPS = [("shortage", 2), ("delay", 2), ("delay", 3)]
RETAINED = {"shortage": MARKOV_READOUT_DEPTH["shortage"], "delay": MARKOV_READOUT_DEPTH["delay"]}
PROBE = sorted({(t, d) for t, d in SWAPS} | {(t, d) for t, d in RETAINED.items()})


@torch.no_grad()
def readout(model, bundles, probe) -> dict:
    """One forward pass per snapshot; returns, for every `(task, depth)` in `probe`,

        {"y": labels, "p_frozen": ARM A probabilities, "X": ARM B feature matrix}

    All three come off the SAME `layers` list SHARE already computed, indexed at the label
    rows `ml/evaluate.py::collect_predictions` and `ml/layer3_baseline.py::feature_table` use,
    so the rows line up with Phase 0's row-for-row.
    """
    model.eval()
    acc = {k: {"y": [], "p": [], "X": []} for k in probe}
    for b in bundles:
        layers = model.encoder(b.data.x_dict, b.data.edge_index_dict)
        for (task, depth) in probe:
            idx, y = b.labels[task]
            if idx.numel() == 0:
                continue
            h = layers[depth][TASK_ENTITY_TYPE[task]]
            acc[(task, depth)]["y"].append(y.cpu().numpy())
            acc[(task, depth)]["p"].append(
                torch.sigmoid(model.heads[task](h)[idx]).float().cpu().numpy())
            acc[(task, depth)]["X"].append(h[idx].detach().cpu().numpy())
    out = {}
    for k, d in acc.items():
        out[k] = {"y": np.concatenate(d["y"]).astype(float) if d["y"] else np.zeros(0),
                  "p_frozen": np.concatenate(d["p"]) if d["p"] else np.zeros(0),
                  "X": np.concatenate(d["X"]).astype(float) if d["X"] else np.zeros((0, 0))}
    return out


def _auc(y, p) -> float | None:
    return float(roc_auc_score(y, p)) if len(y) and 0 < y.sum() < len(y) else None


def run(csv_root: str, variant: str, dseeds, mseeds, device: str = "cpu") -> dict:
    arm_a: dict = {}
    arm_b: dict = {}
    check: list = []
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        for m in mseeds:
            model, meta = get_backbone(csv_dir, variant, d, m, device=device, verbose=False)
            assert_backbone_frozen(model)
            tr, _va, te, _ids = load_world(csv_dir, device)
            R_tr = readout(model, tr, PROBE)
            R_te = readout(model, te, PROBE)

            a_cell, b_cell = {}, {}
            for key in PROBE:
                task, depth = key
                a_cell[f"{task}@h{depth}"] = _auc(R_te[key]["y"], R_te[key]["p_frozen"])
                b_cell[f"{task}@h{depth}"] = train_head(
                    R_tr[key]["X"], R_tr[key]["y"], R_te[key]["X"], R_te[key]["y"], m,
                    model=model)
                # ARM A at the retained depth IS Phase 0's P(Y|X,H): assert it reproduces.
                if depth == RETAINED[task]:
                    ref = meta["auc"].get(task)
                    got = a_cell[f"{task}@h{depth}"]
                    if ref is not None and got is not None:
                        check.append((d, m, task, ref, got, abs(ref - got)))
            assert_backbone_frozen(model)
            arm_a[(d, m)], arm_b[(d, m)] = a_cell, b_cell
            print(f"  d{d} m{m}  " + "  ".join(
                f"{k}: A={_f(a_cell[k])} B={_f(b_cell[k])}" for k in sorted(a_cell)),
                flush=True)
    return {"arm_a": arm_a, "arm_b": arm_b, "reproduction_check": check}


def _f(v) -> str:
    return " n/a  " if v is None else f"{v:.4f}"


def grid_stats(cells: dict, key: str, dseeds, mseeds) -> dict:
    """`ml/layer3_baseline.py::floors`, specialised to one `(task, depth)` series.

    Same definitions, deliberately duplicated in arithmetic rather than in intent:
        init-seed floor    = max_d ( max_m AUC[d,m] - min_m AUC[d,m] )
        dataset-seed floor = max_d mean_m AUC[d,m] - min_d mean_m AUC[d,m]
    """
    M = np.array([[cells[(d, m)].get(key) if cells[(d, m)].get(key) is not None else np.nan
                   for m in mseeds] for d in dseeds], dtype=float)
    per_world_spread = [float(np.nanmax(r) - np.nanmin(r)) for r in M if np.isfinite(r).sum() > 1]
    world_means = np.array([np.nanmean(r) for r in M], dtype=float)
    init_floor = max(per_world_spread) if per_world_spread else float("nan")
    dset_floor = float(np.nanmax(world_means) - np.nanmin(world_means))
    return {"n_cells": int(np.isfinite(M).sum()),
            "grand_mean": float(np.nanmean(M)),
            "per_world_mean": {str(k): float(v) for k, v in zip(dseeds, world_means)},
            "init_seed_floor": init_floor,
            "dataset_seed_floor": dset_floor,
            "floor": max(init_floor, dset_floor)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", default=os.path.join(OUT_DIR, "phase0_baseline.json"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "phase0_depth_swap.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    base = json.load(open(a.baseline))
    px_cells = {(d, m): base["auc_grid_px"][f"{d}|{m}"] for d in dseeds for m in mseeds}

    got = run(a.csv_root, a.variant, dseeds, mseeds, a.device)

    worst = max((c[5] for c in got["reproduction_check"]), default=float("nan"))
    print(f"\nARM A vs cached Phase 0 P(Y|X,H) at the retained depth: "
          f"{len(got['reproduction_check'])} cells checked, max |diff| = {worst:.2e}")

    rows = {}
    for arm in ("arm_a", "arm_b"):
        for task, depth in PROBE:
            key = f"{task}@h{depth}"
            s = grid_stats(got[arm], key, dseeds, mseeds)
            px = grid_stats(px_cells, task, dseeds, mseeds)
            per_world_gain = {w: s["per_world_mean"][w] - px["per_world_mean"][w]
                              for w in s["per_world_mean"]}
            gain = s["grand_mean"] - px["grand_mean"]
            rows[f"{arm}|{key}"] = {
                "arm": arm, "task": task, "depth": depth,
                "retained_depth": RETAINED[task], "is_swap": (task, depth) in SWAPS,
                "p_y_given_x": px["grand_mean"], "p_y_given_xh": s["grand_mean"],
                "graph_gain": gain,
                "init_seed_floor": s["init_seed_floor"],
                "dataset_seed_floor": s["dataset_seed_floor"],
                "floor": s["floor"], "clears_floor": bool(gain > s["floor"]),
                "n_cells": s["n_cells"],
                "per_world_pxh": s["per_world_mean"],
                "per_world_px": px["per_world_mean"],
                "per_world_gain": per_world_gain,
                "worlds_positive": int(sum(v > 0 for v in per_world_gain.values())),
                "n_worlds": len(per_world_gain),
                "sign_consistent": bool(all(v > 0 for v in per_world_gain.values())
                                        or all(v < 0 for v in per_world_gain.values())),
            }

    hdr = (f"{'arm':<11}{'task':<10}{'depth':>7}{'P(Y|X)':>10}{'P(Y|X,H)':>11}"
           f"{'graph gain':>12}{'init floor':>12}{'dset floor':>12}{'FLOOR':>10}"
           f"{'gain>floor':>12}{'worlds +':>10}")
    for arm, label in (("arm_a", "ARM A — frozen head, pure index-select"),
                       ("arm_b", "ARM B — readout head refit on frozen h^d")):
        print("\n" + "=" * len(hdr))
        print(f"{label}   variant {a.variant}, {len(dseeds)} worlds x {len(mseeds)} init "
              f"seeds ({len(dseeds)*len(mseeds)} cells)")
        print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
        name = "ARM " + arm[-1].upper()
        for task, depth in PROBE:
            r = rows[f"{arm}|{task}@h{depth}"]
            tag = "" if r["is_swap"] else "  (retained)"
            print(f"{name:<11}{task:<10}{'h^' + str(depth):>7}"
                  f"{r['p_y_given_x']:>10.4f}{r['p_y_given_xh']:>11.4f}"
                  f"{r['graph_gain']:>+12.4f}{r['init_seed_floor']:>12.4f}"
                  f"{r['dataset_seed_floor']:>12.4f}{r['floor']:>10.4f}"
                  f"{('YES' if r['clears_floor'] else 'no'):>12}"
                  f"{str(r['worlds_positive'])+'/'+str(r['n_worlds']):>10}{tag}")
        print("-" * len(hdr))

    blob = {"config": vars(a), "swaps": SWAPS, "retained": RETAINED, "rows": rows,
            "reproduction_check": {"n": len(got["reproduction_check"]),
                                   "max_abs_diff": worst,
                                   "cells": got["reproduction_check"]},
            "grid_arm_a": {f"{d}|{m}": got["arm_a"][(d, m)] for d in dseeds for m in mseeds},
            "grid_arm_b": {f"{d}|{m}": got["arm_b"][(d, m)] for d in dseeds for m in mseeds}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
