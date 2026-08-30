#!/usr/bin/env python3
"""
Layer 3 v3 — the Carrier fan-in fix, and the ablation that tests its premise.

`reports/new_nodes_result.md` found the enriched graph bought `shortage` a large gain
(−0.0856 → +0.0332, sign-flipped in 5/5 worlds) and paid for it in reproducibility: init-seed
floors rose 4x-16x on all three tasks at once, with seeds m0 and m2 bad on delay, shortage and
impact simultaneously -- a shared-trunk signature. The leading suspect was `Carrier`: 5 nodes
absorbing the entire shipment population, mean in-degree growing to ~7,100.

This module runs the two arms that settle it, over the SAME emitted CSVs:

    no_carrier    CONTROL, not a candidate. `HANDLED_BY` and the `Carrier` node type dropped.
                  If the floors fall back while `shortage`'s gain survives (its gain should come
                  from `REPLENISHED_BY`, not `HANDLED_BY`), Carrier's fan-in is confirmed as the
                  cause.
    carrier_lane  THE FIX. One node per observed `(carrier, lane)` pair instead of 5 carrier
                  nodes -- same fact, same features copied down unchanged, spread over 312 small
                  nodes. Measured before training: mean in-degree 7,136 -> 114 at t14, a 62x
                  reduction, with no new hub created (`ml/carrier_fan_in_census.py`).

**Everything else is held fixed.** Same 5x5 grid, same worlds, same protocol imported from
`ml/layer3_baseline.py`, and depths NOT re-tuned (`delay -> h¹`, `shortage -> h³`,
`impact -> h⁴`), so any movement is attributable to the Carrier schema and nothing else. All
three enriched arms read byte-identical CSVs -- the bucket is a regrouping of two already-emitted
columns -- so the comparison isolates graph SHAPE.

    python3 ml/layer3_carrier_fix.py --schemas no_carrier,carrier_lane
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world       # noqa: E402
from ml.layer3_baseline import floors, px_auc             # noqa: E402
from ml.models.depth import TASKS                         # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH  # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")
ARMS = ["old", "enriched", "no_carrier", "carrier_lane"]
ARM_LABEL = {"old": "old graph (V2)", "enriched": "enriched (Carrier hub)",
             "no_carrier": "ablation (no Carrier)", "carrier_lane": "fixed (bucketed Carrier)"}


def run(csv_root: str, prefix: str, variant: str, dseeds, mseeds, schema: str,
        device="cpu") -> dict:
    pxh, px, shapes = {}, {}, {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"{prefix}{d}")
        for m in mseeds:
            _model, meta = get_backbone(csv_dir, variant, d, m, device=device, verbose=False,
                                        v3_schema=schema)
            pxh[(d, m)] = {t: meta["auc"].get(t) for t in TASKS}
            tr, _va, te, _ids = load_world(csv_dir, device, schema)
            px[(d, m)] = {t: px_auc(tr, te, t, m) for t in TASKS}
            if not shapes:
                b = tr[0].data
                shapes = {"node_types": sorted(b.node_types), "n_node_types": len(b.node_types),
                          "n_relations": len(b.edge_types), "params": int(meta["params"])}
            print(f"  [{schema}] d{d} m{m}  " + "  ".join(
                f"{t}={_f(pxh[(d,m)][t])}" for t in TASKS), flush=True)
    return {"pxh": pxh, "px": px, "shapes": shapes}


def _f(v) -> str:
    return " n/a  " if v is None else f"{v:.4f}"


def summarise(pxh_cells, px_cells, dseeds, mseeds) -> dict:
    """Phase 0's own `floors()` on both arms, plus the per-world gain and sign count."""
    f_pxh, f_px = floors(pxh_cells, dseeds, mseeds), floors(px_cells, dseeds, mseeds)
    out = {}
    for t in TASKS:
        if not f_pxh[t].get("available"):
            continue
        gain = f_pxh[t]["grand_mean"] - f_px[t]["grand_mean"]
        pw = {str(d): f_pxh[t]["per_world_mean"][str(d)] - f_px[t]["per_world_mean"][str(d)]
              for d in dseeds}
        npos = sum(1 for v in pw.values() if v > 0)
        out[t] = {"depth": MARKOV_READOUT_DEPTH[t],
                  "p_y_given_x": f_px[t]["grand_mean"],
                  "p_y_given_xh": f_pxh[t]["grand_mean"],
                  "graph_gain": gain,
                  "init_seed_floor": f_pxh[t]["init_seed_floor"],
                  "dataset_seed_floor": f_pxh[t]["dataset_seed_floor"],
                  "floor": f_pxh[t]["floor"],
                  "clears_floor": bool(gain > f_pxh[t]["floor"]),
                  "per_world_gain": pw, "worlds_positive": npos, "n_worlds": len(pw),
                  "sign_consistent": bool(all(v > 0 for v in pw.values())
                                          or all(v < 0 for v in pw.values())),
                  "passes": bool(gain > f_pxh[t]["floor"] and npos == len(pw)),
                  "n_cells": f_pxh[t]["n_cells"],
                  "per_world_mean_pxh": f_pxh[t]["per_world_mean"]}
    return out


def by_init_seed(pxh_cells, dseeds, mseeds) -> dict:
    """Mean test AUC per model-init seed -- the view that exposed the m0/m2 correlation in
    `reports/new_nodes_result.md` §6. If the hub was the cause, that correlation breaks here."""
    out = {}
    for t in TASKS:
        M = np.array([[pxh_cells[(d, m)].get(t) if pxh_cells[(d, m)].get(t) is not None
                       else np.nan for m in mseeds] for d in dseeds], float)
        out[t] = {str(m): float(np.nanmean(M[:, j])) for j, m in enumerate(mseeds)}
        out[t]["_range"] = float(np.nanmax([np.nanmean(M[:, j]) for j in range(len(mseeds))])
                                 - np.nanmin([np.nanmean(M[:, j]) for j in range(len(mseeds))]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v3enriched"))
    ap.add_argument("--prefix", default="v0enr_seed")
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--schemas", default="no_carrier,carrier_lane")
    ap.add_argument("--baseline", default=os.path.join(OUT_DIR, "phase0_baseline.json"))
    ap.add_argument("--enriched", default=os.path.join(OUT_DIR, "phase0_enriched.json"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "carrier_fix.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]
    schemas = [x.strip() for x in a.schemas.split(",") if x.strip()]

    base = json.load(open(a.baseline))
    enr = json.load(open(a.enriched))

    arms: dict = {"old": {}, "enriched": {}}
    for t in TASKS:
        for src, key in ((base, "old"), (enr, "enriched")):
            r = src["tasks"][t]
            pw = r.get("per_world_gain")
            if pw is None:      # Phase 0 predates the per-world gain field; derive it
                pw = {w: r["p_y_given_xh"]["per_world_mean"][w] - r["p_y_given_x"]["per_world_mean"][w]
                      for w in r["p_y_given_xh"]["per_world_mean"]}
            arms[key][t] = {
                "depth": r["markov_depth"],
                "p_y_given_x": r["p_y_given_x"]["grand_mean"],
                "p_y_given_xh": r["p_y_given_xh"]["grand_mean"],
                "graph_gain": r["graph_gain"],
                "init_seed_floor": r["p_y_given_xh"]["init_seed_floor"],
                "dataset_seed_floor": r["p_y_given_xh"]["dataset_seed_floor"],
                "floor": r["p_y_given_xh"]["floor"],
                "clears_floor": bool(r["graph_gain"] > r["p_y_given_xh"]["floor"]),
                "per_world_gain": pw,
                "worlds_positive": sum(1 for v in pw.values() if v > 0),
                "n_worlds": len(pw),
                "passes": bool(r["graph_gain"] > r["p_y_given_xh"]["floor"]
                               and sum(1 for v in pw.values() if v > 0) == len(pw)),
                "per_world_mean_pxh": r["p_y_given_xh"]["per_world_mean"]}

    grids, seedview, shapes = {}, {}, {}
    for schema in schemas:
        got = run(a.csv_root, a.prefix, a.variant, dseeds, mseeds, schema, a.device)
        arms[schema] = summarise(got["pxh"], got["px"], dseeds, mseeds)
        seedview[schema] = by_init_seed(got["pxh"], dseeds, mseeds)
        shapes[schema] = got["shapes"]
        grids[schema] = {f"{d}|{m}": got["pxh"][(d, m)] for d in dseeds for m in mseeds}
        # P(Y|X) must still be untouched: no task entity's features change with the schema
        mx = max(abs(got["px"][(d, m)][t] - base["auc_grid_px"][f"{d}|{m}"][t])
                 for d in dseeds for m in mseeds for t in TASKS
                 if got["px"][(d, m)][t] is not None
                 and base["auc_grid_px"][f"{d}|{m}"][t] is not None)
        print(f"  [{schema}] P(Y|X) vs frozen Phase 0: max per-cell |diff| = {mx:.2e}")

    # ---------------------------------------------------------------- four-way table
    hdr = (f"{'task':<10}{'arm':<26}{'P(Y|X,H)':>10}{'gain':>10}{'init fl':>9}{'dset fl':>9}"
           f"{'FLOOR':>9}{'gain>FL':>9}{'worlds+':>9}{'PASS':>7}")
    print("\n" + "=" * len(hdr))
    print(f"FOUR-WAY COMPARISON — {len(dseeds)} worlds x {len(mseeds)} init seeds, depths fixed")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for t in TASKS:
        for arm in ARMS:
            r = arms.get(arm, {}).get(t)
            if not r:
                continue
            print(f"{t if arm == ARMS[0] else '':<10}{ARM_LABEL[arm]:<26}"
                  f"{r['p_y_given_xh']:>10.4f}{r['graph_gain']:>+10.4f}"
                  f"{r['init_seed_floor']:>9.4f}{r['dataset_seed_floor']:>9.4f}"
                  f"{r['floor']:>9.4f}{('YES' if r['clears_floor'] else 'no'):>9}"
                  f"{str(r['worlds_positive'])+'/'+str(r['n_worlds']):>9}"
                  f"{('PASS' if r['passes'] else '-'):>7}")
        print("-" * len(hdr))

    # ---------------------------------------------------------------- init-seed view
    print("\nMEAN TEST AUC BY MODEL-INIT SEED (did the m0/m2 correlation break?)")
    h2 = f"{'arm':<26}{'task':<10}" + "".join(f"{'m'+str(m):>9}" for m in mseeds) + f"{'range':>9}"
    print(h2); print("-" * len(h2))
    for arm in ARMS:
        for t in TASKS:
            if arm in seedview:
                v = seedview[arm][t]
            else:                                  # old / enriched: recover from stored grids
                src = base if arm == "old" else enr
                vals = [np.mean([src["auc_grid_pxh"][f"{d}|{m}"][t] for d in dseeds])
                        for m in mseeds]
                v = {str(m): vals[j] for j, m in enumerate(mseeds)}
                v["_range"] = float(max(vals) - min(vals))
            print(f"{ARM_LABEL[arm]:<26}{t:<10}"
                  + "".join(f"{v[str(m)]:>9.4f}" for m in mseeds) + f"{v['_range']:>9.4f}")
        print("-" * len(h2))

    blob = {"config": vars(a), "arms": arms, "by_init_seed": seedview, "shapes": shapes,
            "grids": grids, "markov_depths": MARKOV_READOUT_DEPTH}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
