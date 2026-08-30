#!/usr/bin/env python3
"""
Fan-in census for the three Carrier graph schemas — run BEFORE any training on the fixed graph.

`reports/new_nodes_result.md` §6 attributed the cross-task init-seed instability to `Carrier`'s
fan-in: 5 nodes absorbing the whole shipment population, mean in-degree growing to ~7,100, each
one a near-global average under a single per-destination attention softmax. This module measures
whether the `(carrier, lane)` bucketing actually fixes that, and whether it accidentally
recreates a comparable hub somewhere else -- both of which have to be known before a grid is
worth running.

It also asserts the claim the whole fix rests on: **no carrier information is removed.** Every
shipment's true carrier must still be exactly recoverable from its bucket node, checked on every
shipment rather than sampled (`VariantDataset.assert_carrier_recoverable`).

    python3 ml/carrier_fan_in_census.py --csv-dir db/csv_v3enriched/v0enr_seed42
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

from ml.data.loader import V3_SCHEMAS, VariantDataset, load_bundles  # noqa: E402

CACHE = os.path.join(REPO, "ml", ".cache")


def census(bundles, snapshots=(0, -1)) -> dict:
    """Per-node-type in-degree at the chosen snapshots.

    In-degree is counted over the FORWARD relations only (`rev_*` are the `ToUndirected`
    mirrors), which is the same convention `reports/new_nodes_result.md`'s census used, so the
    numbers are directly comparable to the ones it published.
    """
    out = {}
    for si in snapshots:
        b = bundles[si].data
        tag = f"t{si if si >= 0 else len(bundles) + si}"
        rows = {}
        for nt in b.node_types:
            n = b[nt].num_nodes
            deg = np.zeros(n, dtype=np.int64)
            for et in b.edge_types:
                if et[1].startswith("rev_") or et[2] != nt:
                    continue
                idx = b[et].edge_index[1].cpu().numpy()
                if idx.size:
                    deg += np.bincount(idx, minlength=n)
            rows[nt] = {"n": int(n), "edges_in": int(deg.sum()),
                        "mean_in_degree": float(deg.mean()),
                        "max_in_degree": int(deg.max()) if n else 0,
                        "p95_in_degree": float(np.percentile(deg, 95)) if n else 0.0}
        out[tag] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_v3enriched", "v0enr_seed42"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "layer3_v3", "carrier_fan_in.json"))
    a = ap.parse_args()

    blob = {"csv_dir": a.csv_dir, "schemas": {}}

    print(f"carrier-information recoverability under the bucketed schema")
    ds = VariantDataset(a.csv_dir, v3_schema="carrier_lane")
    rec = ds.assert_carrier_recoverable()
    print(f"  [PASS] every shipment's carrier recoverable from its (carrier, lane) node "
          f"({rec['shipments_checked']:,} shipments, {rec['carrier_lane_nodes']} bucket nodes, "
          f"{rec['parent_carriers']} parent carriers)")
    blob["recoverability"] = rec

    for schema in V3_SCHEMAS:
        bundles, _ = load_bundles(a.csv_dir, cache_dir=CACHE, v3_schema=schema)
        c = census(bundles)
        blob["schemas"][schema] = c
        b0 = bundles[0].data
        print(f"\n=== schema '{schema}' — {len(b0.node_types)} node types, "
              f"{len(b0.edge_types)} relations (incl. reverses) ===")
        hdr = (f"{'node type':<14}{'n':>8}{'in-deg mean t0':>16}{'t14':>10}"
               f"{'max t0':>10}{'t14':>10}")
        print(hdr); print("-" * len(hdr))
        tags = list(c)
        for nt in sorted(c[tags[0]], key=lambda x: -c[tags[-1]][x]["mean_in_degree"]):
            r0, r1 = c[tags[0]][nt], c[tags[-1]][nt]
            flag = "   <-- HUB" if r1["mean_in_degree"] >= 1000 else ""
            print(f"{nt:<14}{r0['n']:>8,}{r0['mean_in_degree']:>16,.0f}"
                  f"{r1['mean_in_degree']:>10,.0f}{r0['max_in_degree']:>10,}"
                  f"{r1['max_in_degree']:>10,}{flag}")

    # the headline comparison the fix is judged on, before any training
    print("\n" + "=" * 78)
    print("FAN-IN OF THE CARRIER-BEARING NODE TYPE (the quantity the fix targets)")
    print("=" * 78)
    tags = list(blob["schemas"]["full"])
    for schema, nt in (("full", "Carrier"), ("carrier_lane", "CarrierLane")):
        r0 = blob["schemas"][schema][tags[0]].get(nt)
        r1 = blob["schemas"][schema][tags[-1]].get(nt)
        if r0:
            print(f"  {schema:<14} {nt:<12} n={r0['n']:<5} "
                  f"mean in-degree {r0['mean_in_degree']:>8,.0f} -> {r1['mean_in_degree']:>8,.0f}   "
                  f"max {r0['max_in_degree']:>7,} -> {r1['max_in_degree']:>7,}")
    f_hub = blob["schemas"]["full"][tags[-1]]["Carrier"]["mean_in_degree"]
    l_hub = blob["schemas"]["carrier_lane"][tags[-1]]["CarrierLane"]["mean_in_degree"]
    print(f"  reduction in mean in-degree at t14: {f_hub / max(1e-9, l_hub):.1f}x")
    worst = max((v["mean_in_degree"], k) for k, v in
                blob["schemas"]["carrier_lane"][tags[-1]].items())
    print(f"  largest mean in-degree anywhere in the fixed graph: "
          f"{worst[1]} at {worst[0]:,.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
