"""Stage 1 — the shuffled-graph control.

v8's own G4 (docs/v8/v8_report.md S3.1) reports that the graph contributes nothing, established
with a shuffled neighbourhood on five dataset seeds. Phase 1 on v8 (reports/part2/phase-0-1-v8.md
S3.0) reports h4 > h0 with no band overlap. Both cannot be simply true, and "h4 > h0" on its own
cannot distinguish

    THE EDGES CARRY INFORMATION   from   THE EXTRA DEPTH AND PARAMETERS HELP REGARDLESS.

This module builds the arm that separates them: the SAME architecture, depth and parameter count,
reading a neighbourhood that has been randomly permuted within each relation type.

THE PERMUTATION. `W["rel"][j]` maps each channel to its entity under relation j (supplier, part,
plant). Applying a permutation pi to the CHANNEL axis --

    rel_shuffled[j][c] = rel[j][pi_j(c)]

-- leaves the multiset of entity assignments untouched, so EVERY entity keeps exactly the degree
it had, while the set of channels in its neighbourhood changes. Each relation gets its OWN
permutation: sharing one would move a channel's whole (supplier, part, plant) triple together and
preserve the co-occurrence structure the control is meant to destroy.

WHAT IS NOT SHUFFLED. `pp_of_chan` / `pp_uniq` map a channel to its part-plant for the shortage
readout. That is entity binding, not graph structure, and shuffling it would change which label a
row is scored against rather than which neighbours a node reads. It is left alone.
"""
from __future__ import annotations
import numpy as np
import torch

from share import build_graph


def permute_relations(rel_list, seed: int):
    """-> (shuffled rel arrays, per-relation stats). Degree-preserving by construction."""
    rng = np.random.default_rng(seed)
    out, stats = [], []
    for j, r in enumerate(rel_list):
        r = np.asarray(r)
        pi = rng.permutation(len(r))
        sh = r[pi]
        n_ent = int(max(r.max(), sh.max())) + 1
        deg_before = np.bincount(r, minlength=n_ent)
        deg_after = np.bincount(sh, minlength=n_ent)
        stats.append(dict(
            relation=j,
            n_edges=int(len(r)),
            n_entities=int((deg_before > 0).sum()),
            edges_endpoint_changed=int((sh != r).sum()),
            pct_endpoint_changed=round(100.0 * float((sh != r).mean()), 4),
            degree_distribution_identical=bool(np.array_equal(deg_before, deg_after)),
            degree_min=int(deg_before.min()), degree_max=int(deg_before.max()),
            degree_median=float(np.median(deg_before)),
            expected_pct_unchanged_if_random=round(
                100.0 * float((deg_before.astype(float) ** 2).sum() / len(r) ** 2), 4),
        ))
        out.append(sh)
    return out, stats


def assert_shuffle_is_real(stats, min_pct_changed=50.0):
    """STAGE 1.2 -- falsify the control before trusting it.

    A 'shuffle' that preserved most edges would make the whole stage vacuous, and a control that
    cannot fail is the defect this project has caught ten times. Two things must hold, and both
    can fail: every relation's degree distribution is EXACTLY preserved, and a large majority of
    edges actually moved.
    """
    for s in stats:
        assert s["degree_distribution_identical"], (
            f"relation {s['relation']}: the shuffle changed the degree distribution, so it is not "
            f"a degree-preserving control")
        assert s["pct_endpoint_changed"] >= min_pct_changed, (
            f"relation {s['relation']}: only {s['pct_endpoint_changed']}% of edges changed "
            f"endpoint (floor {min_pct_changed}%). A shuffle that preserves most edges makes the "
            f"control vacuous -- it would measure nothing.")
    return True


def shuffled_world(W: dict, seed: int, device):
    """A SHALLOW COPY of W whose graph is permuted. The cached real-graph world is not mutated.

    Mutating W in place would silently hand the real-graph arm a shuffled graph on the next call,
    which is exactly the kind of cross-contamination that makes a control worthless.
    """
    rel_sh, stats = permute_relations(W["rel"], seed)
    assert_shuffle_is_real(stats)
    V = dict(W)                                      # shallow copy: only the graph keys are replaced
    V["rel"] = rel_sh
    V["rel_t"] = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rel_sh]
    V["graph"] = build_graph(V["rel_t"], W["rel_sizes"], device)
    V["graph_shuffled"] = True
    V["graph_shuffle_seed"] = int(seed)
    V["graph_shuffle_stats"] = stats
    return V


def shuffle_summary(stats) -> dict:
    return dict(
        relations=len(stats),
        total_edges=sum(s["n_edges"] for s in stats),
        total_endpoint_changed=sum(s["edges_endpoint_changed"] for s in stats),
        pct_endpoint_changed=round(
            100.0 * sum(s["edges_endpoint_changed"] for s in stats)
            / max(1, sum(s["n_edges"] for s in stats)), 4),
        all_degree_distributions_identical=all(s["degree_distribution_identical"] for s in stats),
        per_relation=stats)
