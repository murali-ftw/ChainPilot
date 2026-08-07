"""
Ad-hoc rerun of Task 1's k-hop reach analysis against the live,
confirmed-dual-sourcing-active dataset, sourcing from each task's actual
target entity (Shipment for delay, Supplier for impact) rather than the
generic "path from a Supplier" framing the original script used -- per this
round's explicit request. Reuses ml/graph/reach.py's own functions
unmodified; this is a standalone analysis script, not a permanent addition
to ml/graph/.
"""
from __future__ import annotations

import collections

from ml.data.db import get_connection
from ml.data.snapshots import get_snapshot_schedule
from ml.graph.builder import build_snapshot
from ml.graph.reach import (
    build_adjacency,
    bfs_reach,
    measure_reach_for_type,
    verify_co_parent_reachability,
    _print_type_summary,
    _print_co_parent,
)

conn = get_connection()
schedule = get_snapshot_schedule(conn)
last = schedule.iloc[-1]
data, _ = build_snapshot(conn, last.t0)
adjacency = build_adjacency(data)

print(f"Snapshot: t0={last.t0.date()}  " +
      "  ".join(f"{nt}={data[nt].num_nodes}" for nt in data.node_types))

# ------------------------------------------------------------------ Shipment (delay's actual target)
n_shipment = data["Shipment"].num_nodes
sample_size = min(500, n_shipment)
summary_shipment = measure_reach_for_type(data, adjacency, "Shipment", sample_size=sample_size, seed=0)
print(f"\n{'=' * 70}\nSource: Shipment (delay's actual target, sample {sample_size}/{n_shipment})\n{'=' * 70}")
_print_type_summary(summary_shipment)

# ------------------------------------------------------------------ Supplier (impact's actual target)
n_supplier = data["Supplier"].num_nodes
summary_supplier = measure_reach_for_type(data, adjacency, "Supplier")
print(f"\n{'=' * 70}\nSource: Supplier (impact's actual target, all {n_supplier})\n{'=' * 70}")
_print_type_summary(summary_supplier)

# ------------------------------------------------------------------ co-parent check (Supplier-sourced, unchanged method)
co_parent = verify_co_parent_reachability(data, adjacency, sample_size=800)
print(f"\n{'=' * 70}\nCo-parent reachability check (docs/06_Graph_Database_Design.md §6.1), "
      f"re-run against the live dual-sourcing-active dataset\n{'=' * 70}")
_print_co_parent(co_parent)

# ------------------------------------------------------------------ co-parent hop, specifically FROM Shipment
print(f"\n{'=' * 70}\nAt which hop (if any) does a co-parent Supplier appear, sourced from Shipment "
      f"specifically -- delay's own structural-prior depth is h=2\n{'=' * 70}")
sample_ships = range(min(300, n_shipment))
hop_of_second_supplier = collections.Counter()
checked = 0
for idx in sample_ships:
    visited = bfs_reach(adjacency, "Shipment", int(idx), max_hops=4)
    # the shipment's OWN (first-reached, i.e. hop-1) supplier(s)
    own_suppliers = {i for (t, i), hop in visited.items() if t == "Supplier" and hop == 1}
    if not own_suppliers:
        continue  # this shipment has no SHIPS_FROM->Supplier edge (e.g. factory-only shipment)
    checked += 1
    # first hop at which ANY supplier other than the shipment's own hop-1 supplier(s) appears
    other_supplier_hops = [hop for (t, i), hop in visited.items()
                            if t == "Supplier" and i not in own_suppliers]
    if other_supplier_hops:
        hop_of_second_supplier[min(other_supplier_hops)] += 1
    else:
        hop_of_second_supplier["never (within 4 hops)"] += 1

print(f"  Shipments checked (with a direct SHIPS_FROM->Supplier edge): {checked}")
for hop in [1, 2, 3, 4, "never (within 4 hops)"]:
    n = hop_of_second_supplier.get(hop, 0)
    print(f"    first hop a DIFFERENT supplier appears = {hop}: {n}/{checked} "
          f"({100*n/checked:.1f}%)" if checked else "  (no shipments to check)")

# ------------------------------------------------------------------ impact re-check
print(f"\n{'=' * 70}\nImpact's own characterization -- still hop-0 (direct target), "
      f"does dual-sourcing change what's reachable at h=3?\n{'=' * 70}")
print("  Impact's target IS the Supplier node itself (entity_type='supplier', hop 0 by definition).")
print("  What's reachable AT h<=3 FROM that same Supplier node (from the Source: Supplier table above,")
print("  hop<=3 row) is what impact's h=3 structural-prior readout would draw on if it looked outward")
print("  rather than reading its own hop-0 embedding -- printed above for reference.")

conn.close()
