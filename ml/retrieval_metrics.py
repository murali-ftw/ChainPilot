"""
Retrieval quality for Transformer 2, measured as an information-retrieval problem.

`reports/layer3_testing.md` §2 scored discovery with two metrics — percentile rank of a
ground-truth pair's similarity among all C(n,2) pairs, and pool-membership discovery rate
against a *measured* chance occupancy. Both are kept here unchanged, so §9's numbers sit
on the same axis as §2's. This module adds the three standard IR metrics the redesign
brief requires alongside them:

* **Recall@K** — of an anchor's co-members, what share land in its own top-K pool.
* **Precision@K** — of the K slots in an anchor's pool, what share are co-members.
* **MRR** — 1 / (rank of the anchor's highest-ranked co-member), over the full ranking.

Every metric is computed **per group type (A/B/C), per variant, per seed**, the same
disaggregation discipline every other metric in this project uses, and every one is
reported against a chance baseline rather than against 0.

**Chance is measured, not assumed.** Recall@K and Precision@K have exact analytic chance
values under a uniformly random ranking (`K/(N-1)` and `mean(c_i)/(N-1)` respectively),
but MRR's does not have a convenient closed form once anchors have different co-member
counts, so it is obtained by Monte Carlo over random rankings with a fixed seed. The
same permutation-null discipline `db/generate_dataset.py::_sep` applies to its own
separability checks, and for the same reason: a null that is assumed rather than measured
is where a small-sample bias hides.

The scoring function is deliberately agnostic about **where the ranking came from**. It
takes a score matrix and a pool, so it scores embedding-cosine retrieval (Stage 1) and
observable co-failure-correlation retrieval (Stage 2) on identical terms — which is the
only way the two stages' numbers can be compared to each other at all.
"""

from __future__ import annotations

import numpy as np

MC_REPS = 200
MC_SEED = 20260812


def _pair_percentile(score: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Percentile of each pair's score among all C(n,2) off-diagonal scores --
    `reports/layer3_testing.md` §2's own definition, reproduced exactly so the two
    reports' rank columns mean the same thing."""
    n = score.shape[0]
    iu = np.triu_indices(n, k=1)
    order = np.sort(score[iu])
    vals = np.array([score[a, b] for a, b in pairs], dtype=float)
    return np.searchsorted(order, vals) / max(1, len(order))


def score_retrieval(score: np.ndarray, pool: np.ndarray, members_by_group: dict[int, list[int]],
                    type_by_group: dict[int, str], top_k: int,
                    subset_by_group: dict[int, str] | None = None) -> dict:
    """One snapshot's retrieval quality.

    `score`: `[N, N]` retrieval affinity, higher = more retrievable. The diagonal is
    ignored. `pool`: `[N, k]` retrieved indices per anchor — passed in rather than
    re-derived, because the pool the *model actually attended over* is the thing whose
    quality matters, and a pool rebuilt from the score matrix could silently differ
    (frontier splicing, ties).

    `subset_by_group` optionally tags each group with a label (e.g. "supervised" /
    "held_out") so Stage 1 can report the two splits separately without a second pass.

    Returns per-`(type[, subset])` accumulators of raw per-anchor and per-pair values;
    the caller pools them across snapshots and seeds before taking any mean, so a
    snapshot with more measurable pairs is not silently up-weighted twice.
    """
    n = score.shape[0]
    s = score.copy()
    np.fill_diagonal(s, -np.inf)
    # Descending rank of every column within its row: rank 1 = this anchor's top pick.
    order = np.argsort(-s, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(n)[:, None]
    rank[rows, order] = np.arange(1, n + 1)[None, :]

    pool_sets = [set(r.tolist()) for r in pool]
    # Chance pool-membership rate, MEASURED from this run's own pool occupancy exactly as
    # `ml/run_transformer2.py::discovery` does -- retrieval is largely mutual, so the
    # analytic 2k/(n-1) overstates distinct-pair coverage.
    covered = set()
    for i, r in enumerate(pool_sets):
        for j in r:
            covered.add((i, j) if i < j else (j, i))
    chance_rate = len(covered) / max(1, n * (n - 1) / 2)

    acc: dict[str, dict[str, list]] = {}

    def bucket(key: str) -> dict:
        return acc.setdefault(key, {"pct": [], "found": [], "recall": [], "precision": [],
                                    "rr": [], "n_comembers": []})

    pairs_by_key: dict[str, list[tuple[int, int]]] = {}
    for gi, mem in members_by_group.items():
        if len(mem) < 2:
            continue
        t = type_by_group[gi]
        keys = [t]
        if subset_by_group is not None and gi in subset_by_group:
            keys.append(f"{t}|{subset_by_group[gi]}")
        for key in keys:
            b = bucket(key)
            pairs_by_key.setdefault(key, [])
            for ai, a in enumerate(mem):
                co = [m for m in mem if m != a]
                hit = sum(1 for m in co if m in pool_sets[a])
                b["recall"].append(hit / len(co))
                b["precision"].append(hit / max(1, pool.shape[1]))
                b["rr"].append(1.0 / min(rank[a, m] for m in co))
                b["n_comembers"].append(len(co))
                for bnode in mem[ai + 1:]:
                    pairs_by_key[key].append((a, bnode))
                    b["found"].append(float(bnode in pool_sets[a] or a in pool_sets[bnode]))

    for key, prs in pairs_by_key.items():
        if prs:
            acc[key]["pct"] = list(_pair_percentile(s, prs))

    return {"n_nodes": n, "top_k": int(pool.shape[1]), "chance_discovery_rate": chance_rate,
            "acc": acc}


def chance_baselines(n_nodes: int, top_k: int, n_comembers: list[int]) -> dict:
    """Analytic Recall@K / Precision@K chance plus a Monte-Carlo MRR chance for this
    exact distribution of co-member counts."""
    if not n_comembers:
        return {}
    c = np.asarray(n_comembers, dtype=float)
    rng = np.random.default_rng(MC_SEED)
    rr = []
    for _ in range(MC_REPS):
        # Rank of the best of c co-members under a uniformly random ranking of the other
        # N-1 suppliers: sample c distinct ranks, take the minimum.
        for ci in np.unique(c).astype(int):
            k = int((c == ci).sum())
            draws = rng.integers(1, n_nodes, size=(k, ci))
            rr.append(1.0 / draws.min(axis=1))
    rr = np.concatenate(rr) if rr else np.array([0.0])
    return {"recall_at_k": top_k / (n_nodes - 1),
            "precision_at_k": float(c.mean()) / (n_nodes - 1),
            "mrr": float(rr.mean()),
            "pct_rank": 0.5}


def merge(accs: list[dict]) -> dict:
    """Pool per-snapshot accumulators into one run's metrics, with chance attached.

    Means are taken over the pooled raw values, never over per-snapshot means -- the six
    test snapshots carry slightly different numbers of measurable anchors, and averaging
    averages would weight a thin snapshot equally with a full one.
    """
    if not accs:
        return {}
    keys = sorted({k for a in accs for k in a["acc"]})
    n_nodes = accs[0]["n_nodes"]
    top_k = accs[0]["top_k"]
    out = {"n_nodes": n_nodes, "top_k": top_k,
           "chance_discovery_rate": float(np.mean([a["chance_discovery_rate"] for a in accs]))}
    for key in keys:
        pooled = {f: [] for f in ("pct", "found", "recall", "precision", "rr", "n_comembers")}
        for a in accs:
            for f, v in a["acc"].get(key, {}).items():
                pooled[f].extend(v)
        if not pooled["pct"]:
            continue
        ch = chance_baselines(n_nodes, top_k, pooled["n_comembers"])
        out[key] = {
            "pct_rank": float(np.mean(pooled["pct"])),
            "discovery_rate": float(np.mean(pooled["found"])),
            "recall_at_k": float(np.mean(pooled["recall"])),
            "precision_at_k": float(np.mean(pooled["precision"])),
            "mrr": float(np.mean(pooled["rr"])),
            "n_pairs": len(pooled["pct"]),
            "n_anchors": len(pooled["recall"]),
            "chance": {**ch, "discovery_rate": out["chance_discovery_rate"]},
        }
    return out


# Metric names in the order every table in this session prints them, with the direction
# that counts as better. Kept here so the runner, the analyser and the floor script
# cannot drift apart on either.
METRICS = ("pct_rank", "discovery_rate", "recall_at_k", "precision_at_k", "mrr")
