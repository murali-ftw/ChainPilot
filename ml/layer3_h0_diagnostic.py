#!/usr/bin/env python3
"""
Layer 3 v3, DEPTH FOLLOW-UP #2 — is `delay`'s bottleneck Shipment's own thin raw feature
vector, or the encoder's neighbour-aggregation step?

`reports/new_task_depth.md` compared `delay`'s readout at h^1 (retained), h^2 and h^3 and found
no swap that beats h^1. What it could not answer is *why*, because every depth it tested has
already been through at least one round of message passing. The missing control is **h^0** --
`RGCNAttnDepthGateEncoder.forward()`'s `{nt: self.lin_in[nt](x)}`, Shipment's own raw feature
vector under a learned linear projection and NOTHING ELSE: no neighbour term, no aggregation,
no attention. Reading h^0 against h^1 isolates the FIRST round of neighbour-blending, which is
the one step the existing h^1/h^2/h^3 table holds fixed.

Three outcomes are distinguishable, and the report is required to say plainly which one the
data shows rather than to round an ambiguous result into a story:

    h^1 ~= h^0   the first blend changes nothing measurable -> the bottleneck is Shipment's own
                 features; neighbour information is not meaningfully reaching the readout.
    h^1 <  h^0   the blend is destroying signal the raw features already carried -> the
                 "summed into an undifferentiated hidden-slot card" hypothesis.
    h^1 >  h^0   the blend is net-helpful, and the deeper-hurts result is a separate effect.

**Nothing here trains, modifies or touches SHARE.** Same 25 frozen checkpoints in
`out/ds_ckpt/` (`v0_seed4{2..6}_v0_d4{2..6}_m{0..4}_e100.pt`), same 5 dataset seeds x 5
model-init seeds grid, variant 0, `db/csv_v1scale/`. SHARE emits the full `{h^0..h^4}` for
every node in one forward pass; the only thing that varies is which index of that
already-computed list the readout reads. `assert_backbone_frozen()` runs before *and* after
every cell, including the check that no backbone parameter reached an optimizer.

**Both arms, exactly as `ml/layer3_depth_swap.py` defines them.**

  ARM A — `frozen_head`: `model.heads["delay"]` applied to `layers[d]["Shipment"]`. Zero
  parameters fitted. Carries that file's caveat unchanged: the head was trained jointly with
  the encoder against h^1's geometry, so a collapse at another depth is evidence about head
  transfer, not about what that depth carries.

  ARM B — `refit_head`: the frozen `h^d` handed to `ml/latent_state_head.py::train_head` --
  the same `in -> 32 -> 1` MLP, the same positive-weighted BCE, the same 120 epochs, the same
  seeding by model-init seed that Phase 0's `P(Y|X)` arm uses. Still zero SHARE parameters
  trained: a ~4k-parameter probe on a frozen cached tensor.

**Saturation diagnostic.** `reports/new_nodes_fix1.md` diagnosed the `d45 m3` "dead head" by
counting unique predicted probabilities, their std, and the fraction at p >= 0.999. The same
three numbers are computed here on Arm B's own test predictions -- the very ones each reported
AUC was scored on -- and reported PER DATASET SEED, because that collapse was isolated to one
seed and a pooled mean would hide it.

**Reproduction checks, both of which must hold before any h^0 number is trusted.**
  1. ARM A at h^1 must reproduce Phase 0's cached `P(Y|X,H)` exactly (max |diff| = 0.00e+00).
  2. Every h^1/h^2/h^3 cell, both arms, must reproduce `out/layer3_v3/phase0_depth_swap.json`
     exactly -- i.e. this harness is the previous one, extended, not a re-implementation.

Floors and `grid_stats` are IMPORTED from `ml/layer3_depth_swap.py`, and `P(Y|X)` is read from
`out/layer3_v3/phase0_baseline.json`'s per-cell grid rather than recomputed, because
features-only does not depend on depth.

    python3 ml/layer3_h0_diagnostic.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
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
from ml.layer3_depth_swap import grid_stats                  # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE                 # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH   # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")

TASK = "delay"                       # scope: delay only. shortage/impact are settled elsewhere.
DEPTHS = [0, 1, 2, 3]                # h^0 is the new arm; h^1 retained; h^2/h^3 from the prior run
RETAINED = MARKOV_READOUT_DEPTH[TASK]


# --------------------------------------------------------------------------- saturation

def saturation(p: np.ndarray) -> dict:
    """`reports/new_nodes_fix1.md`'s dead-head diagnostic, on one cell's test predictions.

    Same three quantities that report used to show `d45 m3`'s delay head had collapsed to a
    constant (1 unique value, std 3.0e-08) while the same model's other heads were healthy,
    plus the p >= 0.999 fraction `reports/new_nodes_result.md` quotes for `delay`.
    """
    if p is None or not len(p):
        return {"n": 0}
    return {"n": int(len(p)),
            "n_unique": int(len(np.unique(np.round(p, 9)))),
            "std": float(np.std(p)),
            "frac_ge_999": float((p >= 0.999).mean()),
            "min_p": float(p.min()), "max_p": float(p.max()),
            "mean_p": float(p.mean())}


# --------------------------------------------------------------------------- one cell

@torch.no_grad()
def readout(model, bundles) -> dict:
    """`{depth: {"y", "p_frozen", "X"}}` for `delay`, off the ONE `layers` list SHARE already
    computed, indexed at the same label rows `ml/layer3_depth_swap.py::readout` uses so the
    rows line up with Phase 0's row-for-row."""
    model.eval()
    et = TASK_ENTITY_TYPE[TASK]
    acc = {d: {"y": [], "p": [], "X": []} for d in DEPTHS}
    for b in bundles:
        layers = model.encoder(b.data.x_dict, b.data.edge_index_dict)
        idx, y = b.labels[TASK]
        if idx.numel() == 0:
            continue
        for d in DEPTHS:
            h = layers[d][et]
            acc[d]["y"].append(y.cpu().numpy())
            acc[d]["p"].append(
                torch.sigmoid(model.heads[TASK](h)[idx]).float().cpu().numpy())
            acc[d]["X"].append(h[idx].detach().cpu().numpy())
    out = {}
    for d, a in acc.items():
        out[d] = {"y": np.concatenate(a["y"]).astype(float) if a["y"] else np.zeros(0),
                  "p_frozen": np.concatenate(a["p"]) if a["p"] else np.zeros(0),
                  "X": np.concatenate(a["X"]).astype(float) if a["X"] else np.zeros((0, 0))}
    return out


def _auc(y, p) -> float | None:
    return float(roc_auc_score(y, p)) if len(y) and 0 < y.sum() < len(y) else None


def run(csv_root: str, variant: str, dseeds, mseeds, device: str = "cpu") -> dict:
    arm_a: dict = {}
    arm_b: dict = {}
    sat: dict = {}
    check: list = []
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        for m in mseeds:
            model, meta = get_backbone(csv_dir, variant, d, m, device=device, verbose=False)
            assert_backbone_frozen(model)
            tr, _va, te, _ids = load_world(csv_dir, device)
            R_tr, R_te = readout(model, tr), readout(model, te)

            a_cell, b_cell, s_cell = {}, {}, {}
            for depth in DEPTHS:
                key = f"h{depth}"
                a_cell[key] = _auc(R_te[depth]["y"], R_te[depth]["p_frozen"])
                auc, p = train_head(R_tr[depth]["X"], R_tr[depth]["y"],
                                    R_te[depth]["X"], R_te[depth]["y"], m,
                                    model=model, return_preds=True)
                b_cell[key] = auc
                s_cell[key] = saturation(p)
                if depth == RETAINED:
                    ref, got = meta["auc"].get(TASK), a_cell[key]
                    if ref is not None and got is not None:
                        check.append((d, m, TASK, ref, got, abs(ref - got)))
            assert_backbone_frozen(model)
            arm_a[(d, m)], arm_b[(d, m)], sat[(d, m)] = a_cell, b_cell, s_cell
            print(f"  d{d} m{m}  " + "  ".join(
                f"{k}: A={_f(a_cell[k])} B={_f(b_cell[k])} u={s_cell[k]['n_unique']:>5}"
                for k in sorted(a_cell)), flush=True)
    return {"arm_a": arm_a, "arm_b": arm_b, "saturation": sat, "reproduction_check": check}


def _f(v) -> str:
    return " n/a  " if v is None else f"{v:.4f}"


def prior_run_check(got: dict, prior_path: str, dseeds, mseeds) -> dict:
    """Reproduction check #2: h^1/h^2/h^3 must match `phase0_depth_swap.json` cell for cell.

    Only depths the prior run actually computed are compared -- h^0 has no prior value by
    construction, which is the entire point of this module.
    """
    if not os.path.exists(prior_path):
        return {"available": False, "path": prior_path}
    prior = json.load(open(prior_path))
    diffs = []
    for arm in ("arm_a", "arm_b"):
        grid = prior.get(f"grid_{arm}", {})
        for d in dseeds:
            for m in mseeds:
                cell = grid.get(f"{d}|{m}", {})
                for depth in DEPTHS:
                    ref = cell.get(f"{TASK}@h{depth}")
                    new = got[arm][(d, m)].get(f"h{depth}")
                    if ref is None or new is None:
                        continue
                    diffs.append((arm, d, m, depth, ref, new, abs(ref - new)))
    return {"available": True, "path": prior_path, "n": len(diffs),
            "max_abs_diff": max((x[6] for x in diffs), default=float("nan")),
            "worst": max(diffs, key=lambda x: x[6]) if diffs else None}


def sat_by_world(sat: dict, key: str, dseeds, mseeds) -> dict:
    """Per-dataset-seed saturation: mean over that world's 5 init seeds, plus the WORST cell.

    The worst cell is carried because `reports/new_nodes_fix1.md`'s collapse was a single
    `(d45, m3)` cell whose signature (1 unique value) a 5-cell mean would dilute to ~20% of
    its size and hide.
    """
    out = {}
    for d in dseeds:
        pairs = [(m, sat[(d, m)][key]) for m in mseeds if sat[(d, m)][key].get("n")]
        if not pairs:
            continue
        cells = [c for _m, c in pairs]
        worst_m, worst = min(pairs, key=lambda p: p[1]["n_unique"])
        out[str(d)] = {
            "n_test": cells[0]["n"],
            "n_unique_mean": float(np.mean([c["n_unique"] for c in cells])),
            "n_unique_min": int(worst["n_unique"]),
            "n_unique_min_mseed": int(worst_m),
            "std_mean": float(np.mean([c["std"] for c in cells])),
            "std_min": float(min(c["std"] for c in cells)),
            "frac_ge_999_mean": float(np.mean([c["frac_ge_999"] for c in cells])),
            "frac_ge_999_max": float(max(c["frac_ge_999"] for c in cells)),
            "per_mseed_n_unique": [int(c["n_unique"]) for c in cells],
            "per_mseed_std": [float(c["std"]) for c in cells],
            "per_mseed_frac_ge_999": [float(c["frac_ge_999"]) for c in cells],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", default=os.path.join(OUT_DIR, "phase0_baseline.json"))
    ap.add_argument("--prior-swap", default=os.path.join(OUT_DIR, "phase0_depth_swap.json"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "delay_h0_diagnostic.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    base = json.load(open(a.baseline))
    px_cells = {(d, m): {TASK: base["auc_grid_px"][f"{d}|{m}"][TASK]}
                for d in dseeds for m in mseeds}

    got = run(a.csv_root, a.variant, dseeds, mseeds, a.device)

    worst = max((c[5] for c in got["reproduction_check"]), default=float("nan"))
    print(f"\nCHECK 1 — ARM A at h^{RETAINED} vs cached Phase 0 P(Y|X,H): "
          f"{len(got['reproduction_check'])} cells, max |diff| = {worst:.2e}")
    prior = prior_run_check(got, a.prior_swap, dseeds, mseeds)
    if prior["available"]:
        print(f"CHECK 2 — h^1/h^2/h^3 vs {os.path.basename(a.prior_swap)}: "
              f"{prior['n']} cells (both arms), max |diff| = {prior['max_abs_diff']:.2e}")
    else:
        print(f"CHECK 2 — skipped, {a.prior_swap} not present")

    px = grid_stats(px_cells, TASK, dseeds, mseeds)
    rows = {}
    for arm in ("arm_a", "arm_b"):
        for depth in DEPTHS:
            key = f"h{depth}"
            s = grid_stats(got[arm], key, dseeds, mseeds)
            per_world_gain = {w: s["per_world_mean"][w] - px["per_world_mean"][w]
                              for w in s["per_world_mean"]}
            gain = s["grand_mean"] - px["grand_mean"]
            rows[f"{arm}|{key}"] = {
                "arm": arm, "task": TASK, "depth": depth, "retained_depth": RETAINED,
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
                "passes": bool(gain > s["floor"]
                               and all(v > 0 for v in per_world_gain.values())),
            }

    hdr = (f"{'arm':<7}{'task':<8}{'depth':>7}{'P(Y|X)':>10}{'P(Y|X,H)':>11}"
           f"{'graph gain':>12}{'init floor':>12}{'dset floor':>12}{'FLOOR':>10}"
           f"{'gain>floor':>12}{'worlds +':>10}{'PASS':>7}")
    for arm, label in (("arm_a", "ARM A — frozen head, pure index-select"),
                       ("arm_b", "ARM B — readout head refit on frozen h^d")):
        print("\n" + "=" * len(hdr))
        print(f"{label}   variant {a.variant}, {len(dseeds)} worlds x {len(mseeds)} init "
              f"seeds ({len(dseeds)*len(mseeds)} cells)")
        print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
        name = "ARM " + arm[-1].upper()
        for depth in DEPTHS:
            r = rows[f"{arm}|h{depth}"]
            tag = "  (retained)" if depth == RETAINED else ("  (NEW)" if depth == 0 else "")
            print(f"{name:<7}{TASK:<8}{'h^' + str(depth):>7}"
                  f"{r['p_y_given_x']:>10.4f}{r['p_y_given_xh']:>11.4f}"
                  f"{r['graph_gain']:>+12.4f}{r['init_seed_floor']:>12.4f}"
                  f"{r['dataset_seed_floor']:>12.4f}{r['floor']:>10.4f}"
                  f"{('YES' if r['clears_floor'] else 'no'):>12}"
                  f"{str(r['worlds_positive'])+'/'+str(r['n_worlds']):>10}"
                  f"{('PASS' if r['passes'] else 'FAIL'):>7}{tag}")
        print("-" * len(hdr))

    sat_rows = {f"h{d}": sat_by_world(got["saturation"], f"h{d}", dseeds, mseeds)
                for d in DEPTHS}
    shdr = (f"{'depth':>6}{'world':>7}{'n test':>9}{'uniq mean':>11}{'uniq min':>10}"
            f"{'std mean':>11}{'std min':>11}{'p>=.999 mean':>14}{'p>=.999 max':>13}")
    print("\n" + "=" * len(shdr))
    print("ARM B saturation on the test set — per dataset seed (mean over 5 init seeds)")
    print("=" * len(shdr)); print(shdr); print("-" * len(shdr))
    for d in DEPTHS:
        for w, v in sat_rows[f"h{d}"].items():
            print(f"{'h^'+str(d):>6}{w:>7}{v['n_test']:>9,}{v['n_unique_mean']:>11.1f}"
                  f"{v['n_unique_min']:>10,}{v['std_mean']:>11.4f}{v['std_min']:>11.4f}"
                  f"{v['frac_ge_999_mean']:>13.2%}{v['frac_ge_999_max']:>13.2%}")
        print("-" * len(shdr))

    blob = {"config": vars(a), "task": TASK, "depths": DEPTHS, "retained_depth": RETAINED,
            "rows": rows, "saturation_by_world": sat_rows,
            "reproduction_check_phase0": {"n": len(got["reproduction_check"]),
                                          "max_abs_diff": worst,
                                          "cells": got["reproduction_check"]},
            "reproduction_check_prior_swap": prior,
            "grid_arm_a": {f"{d}|{m}": got["arm_a"][(d, m)] for d in dseeds for m in mseeds},
            "grid_arm_b": {f"{d}|{m}": got["arm_b"][(d, m)] for d in dseeds for m in mseeds},
            "grid_px": {f"{d}|{m}": px_cells[(d, m)][TASK] for d in dseeds for m in mseeds},
            "grid_saturation": {f"{d}|{m}": got["saturation"][(d, m)]
                                for d in dseeds for m in mseeds}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
