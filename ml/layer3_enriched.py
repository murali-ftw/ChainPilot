#!/usr/bin/env python3
"""
Layer 3 v3 — SHARE + Markov retrained from scratch on the V3 ENRICHED graph.

`reports/new_task_depth.md` eliminated the two readout-side explanations for `delay` and
`shortage` failing Phase 0's bar: readout DEPTH (h¹/h²/h³/h⁴, none cleared its floor) and
readout WEIGHTING (a per-task probe refit on the same frozen representation, also failed). The
remaining candidate is that the emitted graph never carried the signal. `db/enrich_schema.py`
adds the node and edge types it was missing; this module measures what that bought.

**Protocol is `ml/layer3_baseline.py`'s, imported rather than restated.** `feature_table`,
`px_auc` and `floors` are the Phase 0 functions themselves, so `P(Y|X)`, the init-seed floor,
the dataset-seed floor and `FLOOR = max(...)` are the same objects Phase 0 reported, computed
by the same code. The grid is the same 5 dataset seeds x 5 model-init seeds.

**Depths are NOT re-tuned.** `MARKOV_READOUT_DEPTH` is untouched -- `delay -> h¹`,
`shortage -> h³`, `impact -> h⁴`. If the enriched graph moves the best depth, that is a
separate experiment; running it here would make it impossible to attribute a change to the
graph rather than to a new readout index.

**`impact` is included, and its bar is different.** The new relations change every existing
node's neighbourhood, not only the new nodes': `Shipment` gains `HANDLED_BY`/`MOVES_ON`, and
`Supplier` gains `ORIGINATES_FROM`, so `impact`'s own two-hop neighbourhood is different even
though nothing was added "for" it. It already cleared its floor on the old graph, so the
question asked of it is whether it REGRESSED, and the answer is reported with a direction
rather than only a pass/fail.

**`P(Y|X)` is recomputed, and asserted to be unchanged.** No task entity's feature vector was
touched by the enrichment -- every new column lives on `Carrier`/`Port`/`Route`/`Warehouse` --
so `P(Y|X)` must come out bit-identical to Phase 0's. It is recomputed from the enriched
bundles anyway and diffed against `out/layer3_v3/phase0_baseline.json`, because "should be
unchanged" is a claim the run can check rather than assert.

    python3 ml/layer3_enriched.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
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

# The frozen old-graph result every row here is compared against
# (`ml/layer3_baseline.py`, `out/layer3_v3/phase0_baseline.json`).
OLD = {"delay":    {"depth": 1, "px": 0.7416425873195044, "pxh": 0.7773715098213647,
                    "gain": 0.03572892250186033, "init": 0.013953772205674952,
                    "dset": 0.08736616080103254, "floor": 0.08736616080103254, "clears": False},
       "shortage": {"depth": 3, "px": 0.8726629729745242, "pxh": 0.787088781928408,
                    "gain": -0.08557419104611619, "init": 0.010115766262403514,
                    "dset": 0.023058482942992176, "floor": 0.023058482942992176, "clears": False},
       "impact":   {"depth": 4, "px": 0.8201940489875221, "pxh": 0.9338852415732251,
                    "gain": 0.11369119258570293, "init": 0.02614396333206015,
                    "dset": 0.028447763281878702, "floor": 0.028447763281878702, "clears": True}}


def run(csv_root: str, prefix: str, variant: str, dseeds, mseeds, device="cpu") -> dict:
    pxh, px, shapes = {}, {}, {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"{prefix}{d}")
        for m in mseeds:
            model, meta = get_backbone(csv_dir, variant, d, m, device=device, verbose=False)
            pxh[(d, m)] = {t: meta["auc"].get(t) for t in TASKS}
            tr, _va, te, _ids = load_world(csv_dir, device)
            px[(d, m)] = {t: px_auc(tr, te, t, m) for t in TASKS}
            if not shapes:
                b = tr[0].data
                shapes = {"node_types": {nt: [int(b[nt].num_nodes), int(b[nt].x.size(-1))]
                                         for nt in b.node_types},
                          "n_relations": len(b.edge_types),
                          "relations": [list(et) for et in b.edge_types],
                          "params": int(meta["params"])}
            print(f"  d{d} m{m}  " + "  ".join(
                f"{t}: P(Y|X,H)={_f(pxh[(d,m)][t])} P(Y|X)={_f(px[(d,m)][t])}" for t in TASKS),
                flush=True)
    return {"pxh": pxh, "px": px, "shapes": shapes}


def _f(v) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def per_world_gain(f_pxh: dict, f_px: dict, dseeds) -> dict:
    return {str(d): f_pxh["per_world_mean"][str(d)] - f_px["per_world_mean"][str(d)]
            for d in dseeds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v3enriched"))
    ap.add_argument("--prefix", default="v0enr_seed")
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", default=os.path.join(OUT_DIR, "phase0_baseline.json"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "phase0_enriched.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]

    got = run(a.csv_root, a.prefix, a.variant, dseeds, mseeds, a.device)
    f_pxh = floors(got["pxh"], dseeds, mseeds)
    f_px = floors(got["px"], dseeds, mseeds)

    # -- P(Y|X) must be unchanged: no task entity's feature vector was touched -----
    base = json.load(open(a.baseline))
    px_delta, px_cellmax = {}, 0.0
    for t in TASKS:
        px_delta[t] = f_px[t]["grand_mean"] - base["tasks"][t]["p_y_given_x"]["grand_mean"]
    for d in dseeds:
        for m in mseeds:
            for t in TASKS:
                o = base["auc_grid_px"][f"{d}|{m}"].get(t)
                n = got["px"][(d, m)].get(t)
                if o is not None and n is not None:
                    px_cellmax = max(px_cellmax, abs(o - n))
    print(f"\nP(Y|X) vs frozen Phase 0: max per-cell |diff| = {px_cellmax:.2e}   "
          + "  ".join(f"{t} {px_delta[t]:+.6f}" for t in TASKS))

    sh = got["shapes"]
    print(f"\nenriched graph: {len(sh['node_types'])} node types, {sh['n_relations']} relations "
          f"(incl. reverses), {sh['params']:,} parameters")

    rows = {}
    hdr = (f"{'task':<10}{'depth':>7}{'P(Y|X)':>10}{'P(Y|X,H)':>11}{'graph gain':>12}"
           f"{'init floor':>12}{'dset floor':>12}{'FLOOR':>10}{'gain>floor':>12}{'worlds +':>10}")
    print("\n" + "=" * len(hdr))
    print(f"ENRICHED GRAPH — variant {a.variant}, {len(dseeds)} worlds x {len(mseeds)} init "
          f"seeds ({len(dseeds)*len(mseeds)} cells)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for t in TASKS:
        if not f_pxh[t].get("available"):
            continue
        gain = f_pxh[t]["grand_mean"] - f_px[t]["grand_mean"]
        pw = per_world_gain(f_pxh[t], f_px[t], dseeds)
        npos = sum(1 for v in pw.values() if v > 0)
        rows[t] = {"markov_depth": MARKOV_READOUT_DEPTH[t],
                   "p_y_given_x": f_px[t], "p_y_given_xh": f_pxh[t], "graph_gain": gain,
                   "graph_gain_clears_floor": bool(gain > f_pxh[t]["floor"]),
                   "per_world_gain": pw, "worlds_positive": npos, "n_worlds": len(pw),
                   "sign_consistent": bool(all(v > 0 for v in pw.values())
                                           or all(v < 0 for v in pw.values())),
                   "old": OLD[t],
                   "delta_pxh_vs_old": f_pxh[t]["grand_mean"] - OLD[t]["pxh"],
                   "delta_gain_vs_old": gain - OLD[t]["gain"]}
        print(f"{t:<10}{'h^'+str(MARKOV_READOUT_DEPTH[t]):>7}{f_px[t]['grand_mean']:>10.4f}"
              f"{f_pxh[t]['grand_mean']:>11.4f}{gain:>+12.4f}"
              f"{f_pxh[t]['init_seed_floor']:>12.4f}{f_pxh[t]['dataset_seed_floor']:>12.4f}"
              f"{f_pxh[t]['floor']:>10.4f}"
              f"{('YES' if gain > f_pxh[t]['floor'] else 'no'):>12}"
              f"{str(npos)+'/'+str(len(pw)):>10}")
    print("-" * len(hdr))

    # -- row-for-row against the old graph ---------------------------------
    h2 = (f"{'task':<10}{'depth':>7}{'old P(Y|X,H)':>14}{'new P(Y|X,H)':>14}{'delta':>10}"
          f"{'old gain':>10}{'new gain':>10}{'old FLOOR':>11}{'new FLOOR':>11}"
          f"{'old':>6}{'new':>6}")
    print("\n" + "=" * len(h2))
    print("OLD (V2 graph, reports/new_task_depth.md) vs NEW (V3 enriched graph)")
    print("=" * len(h2)); print(h2); print("-" * len(h2))
    for t in TASKS:
        if t not in rows:
            continue
        r, o = rows[t], OLD[t]
        print(f"{t:<10}{'h^'+str(o['depth']):>7}{o['pxh']:>14.4f}"
              f"{r['p_y_given_xh']['grand_mean']:>14.4f}{r['delta_pxh_vs_old']:>+10.4f}"
              f"{o['gain']:>+10.4f}{r['graph_gain']:>+10.4f}"
              f"{o['floor']:>11.4f}{r['p_y_given_xh']['floor']:>11.4f}"
              f"{('YES' if o['clears'] else 'no'):>6}"
              f"{('YES' if r['graph_gain_clears_floor'] else 'no'):>6}")
    print("-" * len(h2))
    for t in TASKS:
        if t in rows:
            print(f"  {t:<9} per-world gain: " +
                  "  ".join(f"{w}:{v:+.4f}" for w, v in rows[t]["per_world_gain"].items()))

    blob = {"config": vars(a), "markov_depths": MARKOV_READOUT_DEPTH, "tasks": rows,
            "graph": sh,
            "px_check": {"max_abs_cell_diff": px_cellmax,
                         "grand_mean_delta": px_delta},
            "auc_grid_pxh": {f"{d}|{m}": got["pxh"][(d, m)] for d in dseeds for m in mseeds},
            "auc_grid_px": {f"{d}|{m}": got["px"][(d, m)] for d in dseeds for m in mseeds}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
