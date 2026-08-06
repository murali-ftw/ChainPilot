"""
Empirical k-hop reach measurement per target node type — Step 2
(`docs/14_Model_Development_Roadmap.md` §5, `docs/06_Graph_Database_Design.md`
§11, §8's Cypher example, §6.1's co-parent claim).

"Do not assume — measure." (`project_HADES.md` §2.2, quoted again in
`docs/06_Graph_Database_Design.md` §11.) This module runs a real BFS over an
assembled `HeteroData` snapshot instead of reading off the theoretical
10-forward-relation table, because that table's hop count doesn't account
for which node types are actually *populated* along a given path for this
dataset. The output here is what Step 4's depth-prior sweep is justified
by -- not `project_HADES.md` §3.7's planning table.

Run: python -m ml.graph.reach
"""

from __future__ import annotations

import collections
import statistics

import numpy as np

from ml.data.db import get_connection
from ml.data.snapshots import get_snapshot_schedule
from ml.graph.builder import build_snapshot

MAX_HOPS = 4


def build_adjacency(data) -> dict[str, list[tuple]]:
    """
    `{src_type: [((src, rel, dst), {src_idx: np.array[dst_idx]}), ...]}` --
    one lookup table per meta-relation whose source is `src_type`, built
    once per snapshot so BFS doesn't rescan `edge_index` per node.
    """
    by_src_type = collections.defaultdict(list)
    for (src, rel, dst) in data.edge_types:
        edge_index = data[src, rel, dst].edge_index.numpy()
        lut = collections.defaultdict(list)
        for s, d in zip(edge_index[0], edge_index[1]):
            lut[int(s)].append(int(d))
        lut = {k: np.array(v, dtype=int) for k, v in lut.items()}
        by_src_type[src].append(((src, rel, dst), lut))
    return by_src_type


def bfs_reach(adjacency: dict, start_type: str, start_idx: int, max_hops: int = MAX_HOPS) -> dict:
    """
    Single-source BFS from one (start_type, start_idx) node. Returns
    `{(node_type, node_idx): first_hop_reached}` for every node reached
    within `max_hops`, `start` itself at hop 0.
    """
    visited = {(start_type, start_idx): 0}
    frontier = [(start_type, start_idx)]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        for (t, i) in frontier:
            for (src, rel, dst), lut in adjacency.get(t, []):
                for j in lut.get(i, ()):
                    key = (dst, int(j))
                    if key not in visited:
                        visited[key] = hop
                        next_frontier.append(key)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def measure_reach_for_type(data, adjacency: dict, target_type: str, max_hops: int = MAX_HOPS,
                            sample_size: int | None = None, seed: int = 0) -> dict:
    """
    Run BFS from every node of `target_type` (or a random sample of
    `sample_size` of them -- large node types like Shipment/Order at v3
    scale, tens of thousands of nodes, make a full sweep impractically slow;
    the v3 round killed exactly that run) and aggregate: for each hop
    1..max_hops, per reachable node type, the mean/median/min/max count of
    nodes reached *within* that many hops (cumulative, not "exactly at").
    """
    num_nodes = data[target_type].num_nodes
    if sample_size is not None and sample_size < num_nodes:
        rng = np.random.default_rng(seed)
        indices = rng.choice(num_nodes, size=sample_size, replace=False)
    else:
        indices = range(num_nodes)
    per_hop_per_type = {hop: collections.defaultdict(list) for hop in range(1, max_hops + 1)}

    for idx in indices:
        visited = bfs_reach(adjacency, target_type, int(idx), max_hops)
        for hop in range(1, max_hops + 1):
            counts = collections.Counter(
                node_type for (node_type, _), reached_at in visited.items()
                if 0 < reached_at <= hop
            )
            for node_type in data.node_types:
                per_hop_per_type[hop][node_type].append(counts.get(node_type, 0))

    summary = {}
    for hop, by_type in per_hop_per_type.items():
        summary[hop] = {
            node_type: {
                "mean": round(statistics.fmean(counts), 2),
                "median": statistics.median(counts),
                "max": max(counts),
                "pct_nonzero": round(100 * sum(c > 0 for c in counts) / len(counts), 1),
            }
            for node_type, counts in by_type.items()
        }
    return summary


def verify_co_parent_reachability(data, adjacency: dict, sample_size: int = 50) -> dict:
    """
    `docs/06_Graph_Database_Design.md` §6.1's specific claim: a Supplier's
    co-parents (other suppliers of the same component) are reachable in 2
    hops via `SUPPLIES` then `rev_SUPPLIES`. This is checked directly rather
    than assumed, because `components.supplier_id` is a single not-null FK
    (`db/schema.sql`) -- each Component row belongs to exactly one Supplier,
    so a `Supplier -> Component -> rev_SUPPLIES -> Supplier` round trip is
    structurally only capable of returning to the *same* supplier it
    started from. This function reports which is actually true.
    """
    num_suppliers = data["Supplier"].num_nodes
    sample = range(min(num_suppliers, sample_size))
    other_supplier_reached = 0
    for idx in sample:
        visited = bfs_reach(adjacency, "Supplier", idx, max_hops=2)
        other_suppliers = {i for (t, i) in visited if t == "Supplier" and i != idx}
        if other_suppliers:
            other_supplier_reached += 1
    return {
        "suppliers_checked": len(list(sample)),
        "suppliers_reaching_a_different_supplier_in_2_hops": other_supplier_reached,
        "claim_holds": other_supplier_reached > 0,
    }


def _print_type_summary(summary: dict) -> None:
    for hop, by_type in summary.items():
        print(f"\n  hop <= {hop}:")
        for node_type, stats in by_type.items():
            if stats["mean"] == 0:
                continue
            print(
                f"    {node_type:10s} mean={stats['mean']:>8.2f}  median={stats['median']:>6}  "
                f"max={stats['max']:>5}  reached-by={stats['pct_nonzero']:>5.1f}% of sources"
            )


def _print_co_parent(co_parent: dict) -> None:
    print(f"  suppliers checked: {co_parent['suppliers_checked']}")
    print(f"  reached a different supplier in 2 hops: {co_parent['suppliers_reaching_a_different_supplier_in_2_hops']}")
    print(f"  claim holds: {co_parent['claim_holds']}")
    if not co_parent["claim_holds"]:
        print(
            "  NOTE: components.supplier_id is a single not-null FK (one supplier per "
            "component) -- the SUPPLIES/rev_SUPPLIES 2-hop co-parent path cannot reach a "
            "different supplier under the current schema. This is a measured finding, not "
            "an assumption; Step 4's depth prior should be justified from this table, not "
            "from docs/06_Graph_Database_Design.md §6.1's claim as written."
        )


def main() -> None:
    """
    v3 follow-up (`reports/step5_result_v3.md` Task 1): the full all-node-
    type reach table died on Shipment's 35,687-node BFS last round. Restricted
    here to exactly the two sources the depth priors actually cite
    (`project_HADES.md` §4.2's "path from a Supplier" table):

      Supplier -> ...   (delay's h^2: children + co-parents)
      Order    -> ...   (shortage/impact's h^3: "...Product->rev_ORDERED->Order")

    Supplier is small enough (800) to run in full; Order is not (tens of
    thousands at later snapshots) so it's sampled.
    """
    conn = get_connection()
    schedule = get_snapshot_schedule(conn)
    # Report against the most fully-populated snapshot (the schedule's last
    # t0, after all growth has accumulated) as the representative case;
    # graph *topology* (which types are reachable at each hop) does not
    # change across t0, only the magnitudes do.
    last = schedule.iloc[-1]
    data, _ = build_snapshot(conn, last.t0)
    adjacency = build_adjacency(data)

    print(f"Snapshot: t0={last.t0.date()}  " +
          "  ".join(f"{nt}={data[nt].num_nodes}" for nt in data.node_types))

    summary = measure_reach_for_type(data, adjacency, "Supplier")
    print(f"\n{'=' * 70}\nSource: Supplier (all {data['Supplier'].num_nodes})\n{'=' * 70}")
    _print_type_summary(summary)

    summary = measure_reach_for_type(data, adjacency, "Order", sample_size=200)
    print(f"\n{'=' * 70}\nSource: Order (sample of 200 / {data['Order'].num_nodes})\n{'=' * 70}")
    _print_type_summary(summary)

    co_parent = verify_co_parent_reachability(data, adjacency, sample_size=800)
    print(f"\n{'=' * 70}\nCo-parent reachability check (docs/06_Graph_Database_Design.md §6.1)\n{'=' * 70}")
    _print_co_parent(co_parent)

    conn.close()


if __name__ == "__main__":
    main()
