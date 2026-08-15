#!/usr/bin/env python3
"""
STEP 7b's conditioning variable — observable, graph-computed summary statistics for a world.

STEP 7a conditions calibration on *which dataset seed a sample came from*. That is diagnostic
only: a sixth, unseen world has no seed ID, so 7a has no defined behaviour on it. For a
deployable conditional calibrator the conditioning variable has to be something a genuinely new
world would hand you. This module computes that: a small vector of summary statistics read
**off the graph itself**, at the point in the pipeline where a deployment would have it.

**Two hard constraints, both enforced rather than intended.**

1. *No generator config.* Nothing here opens `resolved_config.json`, reads a seed, or touches
   the `db/generate_dataset.py` namespace. Every feature is a reduction of the `HeteroData`
   snapshots `ml/data/loader.py` builds -- the same tensors SHARE consumes.
2. *No labels, and no test-partition leakage of the thing being calibrated.* Features are
   computed from node features and edge counts only. They are read from the **test** split's
   snapshots because that is the graph a deployment is predicting on; they never touch `y`.

**The five candidates, and why these five.**

| feature | reads | motivation |
|---|---|---|
| `mean_on_time_180d` | Supplier x, trailing 180-day on-time rate | aggregate stress level -- the direct observable shadow of the latent `stress` this calibration is about |
| `std_on_time_90d` | Supplier x, trailing 90-day on-time rate | dispersion of supplier reliability: two worlds can share a mean and differ entirely in spread |
| `mean_lateness_variance` | Supplier x, lateness variance | volatility rather than level; the generator's `p_delay` moves both, and not together |
| `upstream_edge_density` | `(Supplier, UPSTREAM_OF, Supplier)` edge count / n suppliers | disruption-propagation edge density -- Mechanism J's channel, the structure along which stress spreads |
| `supplier_shipment_degree` | `(Shipment, SHIPS_FROM, Supplier)` edge count / n suppliers | degree-distribution summary; also how much evidence per supplier the observable channel carries |

**One candidate is deliberately excluded, and the exclusion is the point.** `capacity_score` and
`lead_time_days` look like obvious world-summary features and are useless as such: the loader
z-scores them **within each world** (`ml/data/loader.py::_prepare_static_features`), so their
world-level mean is 0 by construction. `mean_capacity_score_z` is computed anyway and reported
as a **negative control** -- if it comes back at ~0 across all five worlds, the pipeline is
reading the columns it thinks it is reading.

**Column indexing.** The loader builds Supplier features as
`supplier_static.merge(temporal, on="entity_id")` and drops `entity_id` in `_tensor`, so the
layout is `[lead_time_days_z, capacity_score_z, country one-hots..., <6 temporal>]` with the six
temporal columns -- `on_time_rate_30d, on_time_rate_90d, on_time_rate_180d, trend_slope,
lateness_variance, days_since_last_late` -- always last and always in that order (fixed by the
`usecols` list at `ml/data/loader.py:206`). They are therefore addressed from the END of the
tensor, which is stable under a change in the number of countries; the country count is asserted
rather than assumed.

**A caveat that belongs on the number, not in a footnote.** The temporal merge is a left join, so
a supplier with no delivered shipment yet at `t0` carries NaN, which `_tensor` fills with 0.0.
`mean_on_time_180d` therefore mixes "on time 40% of the time" with "no history"; that is exactly
what SHARE sees, so it is the right feature for conditioning SHARE's calibration, but it is not
a clean on-time rate and is not reported as one.

    python3 ml/world_summary_features.py --variants A,E --out out/wcc/world_features.json
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

DEFAULT_CSV_ROOT = os.path.join(REPO, "db", "csv_v1scale")
CACHE = os.path.join(REPO, "out", "wcc", "world_features.json")

# Offsets from the END of the Supplier feature tensor. See "Column indexing" above.
N_TEMPORAL = 6
IDX = {"on_time_rate_30d": -6, "on_time_rate_90d": -5, "on_time_rate_180d": -4,
       "trend_slope": -3, "lateness_variance": -2, "days_since_last_late": -1}
IDX_CAPACITY_Z = 1                      # negative control; z-scored within world by the loader

# The features 7b actually conditions on, in a fixed order so coefficients stay interpretable.
FEATURES = ("mean_on_time_180d", "std_on_time_90d", "mean_lateness_variance",
            "upstream_edge_density", "supplier_shipment_degree")
NEGATIVE_CONTROL = "mean_capacity_score_z"


def _snapshot_features(data) -> dict:
    """Reduce one snapshot's graph to the summary statistics. Node features + edge counts only."""
    x = data["Supplier"].x.numpy()
    n_sup = x.shape[0]
    assert x.shape[1] >= N_TEMPORAL + 2, f"Supplier feature tensor too narrow: {x.shape}"

    def edges(key) -> int:
        return int(data[key].edge_index.shape[1]) if key in data.edge_types else 0

    return {
        "mean_on_time_180d": float(x[:, IDX["on_time_rate_180d"]].mean()),
        "std_on_time_90d": float(x[:, IDX["on_time_rate_90d"]].std()),
        "mean_lateness_variance": float(x[:, IDX["lateness_variance"]].mean()),
        "upstream_edge_density": edges(("Supplier", "UPSTREAM_OF", "Supplier")) / n_sup,
        "supplier_shipment_degree": edges(("Shipment", "SHIPS_FROM", "Supplier")) / n_sup,
        NEGATIVE_CONTROL: float(x[:, IDX_CAPACITY_Z].mean()),
        "n_suppliers": n_sup,
        "n_supplier_feature_cols": int(x.shape[1]),
    }


def world_features(variant: str, dseed: int, csv_root: str = DEFAULT_CSV_ROOT,
                   split: str = "te") -> dict:
    """Summary statistics for one world, averaged over that split's snapshots.

    The test split is the default because it is the graph a deployment would be predicting on --
    the same snapshots STEP 3's calibration is evaluated over. No label is read.
    """
    from ml.ds_backbone import load_world  # noqa: PLC0415 -- heavy import, on demand
    tr, va, te, _ = load_world(os.path.join(csv_root, f"v{variant}_seed{dseed}"), "cpu")
    bundles = {"tr": tr, "va": va, "te": te}[split]
    per = [_snapshot_features(b.data) for b in bundles]
    out = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    out.update({"variant": variant, "dataset_seed": dseed, "split": split,
                "n_snapshots": len(per),
                "per_snapshot_std": {k: float(np.std([p[k] for p in per]))
                                     for k in FEATURES}})
    return out


def load_all(variants=("A", "E"), dseeds=(42, 43, 44, 45, 46),
             csv_root: str = DEFAULT_CSV_ROOT, cache: str = CACHE,
             rebuild: bool = False, verbose: bool = True) -> dict:
    """`{variant: {dseed: features}}`, cached -- computing it means reloading 10 worlds."""
    if os.path.exists(cache) and not rebuild:
        blob = json.load(open(cache))
        if all(str(d) in blob.get(v, {}) for v in variants for d in dseeds):
            if verbose:
                print(f"  [cached] {os.path.basename(cache)}", flush=True)
            return {v: {int(d): blob[v][str(d)] for d in dseeds} for v in variants}
    out = {}
    for v in variants:
        out[v] = {}
        for d in dseeds:
            out[v][d] = world_features(v, d, csv_root)
            if verbose:
                f = out[v][d]
                print(f"  v{v} seed {d}: " + "  ".join(f"{k}={f[k]:.4f}" for k in FEATURES),
                      flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as fh:
        json.dump({v: {str(d): out[v][d] for d in out[v]} for v in out}, fh, indent=1)
    return out


def matrix(feats: dict, dseeds: list[int], names=FEATURES) -> np.ndarray:
    """`[n_worlds, n_features]` in `FEATURES` order."""
    return np.array([[feats[d][k] for k in names] for d in dseeds], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,E")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--csv-root", default=DEFAULT_CSV_ROOT)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    variants = [s.strip() for s in a.variants.split(",") if s.strip()]
    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    feats = load_all(variants, dseeds, a.csv_root, rebuild=a.rebuild)

    for v in variants:
        print("\n" + "=" * 96)
        print(f"WORLD SUMMARY FEATURES — variant {v}   (test-split snapshots, graph only)")
        print("=" * 96)
        print(f"{'world':<8}" + "".join(f"{k[:22]:>24}" for k in FEATURES))
        for d in dseeds:
            print(f"{d:<8}" + "".join(f"{feats[v][d][k]:>24.5f}" for k in FEATURES))
        M = matrix(feats[v], dseeds)
        rng_ = M.max(axis=0) - M.min(axis=0)
        sd = M.std(axis=0)
        print(f"{'range':<8}" + "".join(f"{r:>24.5f}" for r in rng_))
        print(f"{'sd':<8}" + "".join(f"{s:>24.5f}" for s in sd))
        print(f"{'sd/|mean|':<8}" + "".join(
            f"{(s / abs(m) if m else float('nan')):>24.5f}" for s, m in zip(sd, M.mean(axis=0))))
        nc = [feats[v][d][NEGATIVE_CONTROL] for d in dseeds]
        print(f"\nnegative control {NEGATIVE_CONTROL} (loader z-scores it WITHIN world, so this "
              f"must be ~0): max|.| = {max(abs(x) for x in nc):.2e}")
        print(f"rank of the centred {len(dseeds)}x{len(FEATURES)} world-feature matrix: "
              f"{np.linalg.matrix_rank(M - M.mean(axis=0)):d}  "
              f"(a fit on 4 training worlds can identify at most 3 centred directions)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({v: {str(d): feats[v][d] for d in dseeds} for v in variants}, f, indent=1)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
