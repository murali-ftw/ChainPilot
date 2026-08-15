#!/usr/bin/env python3
"""
Model B -- simple full-representation baseline for Layer 3 latent-state estimation.

The cheapest possible test of H1 ("information distributed across multiple SHARE depths
improves latent-state estimation") against H0 ("one task-specific depth plus a simple head --
Model A, `ml/latent_state_head.py` -- is sufficient").

    Model A:  h^d          -> MLP -> state      (d = the single best depth, per state)
    Model B:  concat(h^0..h^4) -> MLP -> state

**Literal concatenation only.** No per-level projection, no attention, no learned fusion. That
is the entire point of this arm: if concat-plus-MLP shows nothing, a more expressive combination
mechanism (Model C) is very unlikely to, on this project's own precedent that more expressiveness
stacked on a working representation has not bought anything here before.

**Methodology is Model A's, imported rather than reimplemented.** Target construction
(`latent_targets`), the median binarisation taken from TRAIN only (`binarise`), the 40/20/40
temporal split (`ds_backbone.load_world`), the head shape/optimiser/epochs (`train_head`,
`LatentStateHead`) and the AUC (`hypothesis_ranker.roc_auc`) are all the Phase 1 objects,
imported from `ml/latent_state_head.py`. **The only thing that changes is the width of the
feature vector fed to the head** (128 -> 640). Anything else changing would confound the
comparison this arm exists to make -- in particular this stays a CLASSIFICATION problem at the
same binarisation, deliberately not the regression framing the architecture proposal suggested
eventually, because changing the metric family and the architecture at once would make the
Model A vs Model B delta uninterpretable.

**The reproduction floor is measured fresh.** Model A's per-depth floors are floors for a
different quantity (a 128-dim head) and are NOT inherited. Both floors are re-measured here --
init-seed and dataset-seed -- and the gate uses the larger, exactly as Phase 1 did.

**Seed-to-seed std is a first-class output**, not an afterthought: this project's own SHARP/SHARK
precedent is that a more expressive combination mechanism can match on mean accuracy while
getting meaningfully less stable, and that is the specific failure mode this arm must be able to
catch. Mean AUC alone would not catch it.

**Backbone untouched.** `get_backbone()` freezes; `assert_backbone_frozen()` (Model A's, imported)
runs after head construction and after training on every fit.

    python3 ml/modelB_concat_head.py --variant A --seeds 42,43,44,45,46 --config v1 \
        --init-seeds 0,1,2,3,4 --out out/modelB/concat_A.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world           # noqa: E402
from ml.latent_state_head import (                            # noqa: E402
    DEPTHS,
    align,
    assert_backbone_frozen,
    binarise,
    depth_embeddings,
    latent_targets,
    train_head,
)

# The two states Phase 1 cleared. Supplier Reliability failed its Phase 1 gate independently on
# both variants (below chance, not sign-consistent, ~1.1% positive rate) and is excluded: Model B
# is a test of how to READ a state the representation is known to carry, not a rescue attempt for
# one it does not.
DEFAULT_STATES = ("supply_stress", "recovery_capability")


def concat_embeddings(emb: dict) -> tuple[np.ndarray, list, list]:
    """`{depth: (X, ids, times)}` -> `(concat(h^0..h^4), ids, times)`.

    Asserts the row alignment that makes column-wise concatenation meaningful: every depth must
    emit the same (supplier, t0) rows in the same order. They do -- `depth_embeddings` walks the
    same bundles in the same order for every depth -- but concatenating misaligned rows would
    silently produce a feature matrix that pairs one supplier's h^0 with another's h^3 and would
    still train and still report an AUC, so it is checked rather than assumed.
    """
    ref_ids, ref_times = emb[DEPTHS[0]][1], emb[DEPTHS[0]][2]
    mats = []
    for d in DEPTHS:
        X, ids, times = emb[d]
        if ids != ref_ids or times != ref_times:
            raise RuntimeError(f"row alignment differs at h^{d}; concatenation would be invalid")
        if X.shape[0] != len(ref_ids):
            raise RuntimeError(f"h^{d} row count {X.shape[0]} != {len(ref_ids)} ids")
        mats.append(X)
    widths = [m.shape[1] for m in mats]
    if len(set(widths)) != 1:
        # Step 1 confirmed all five are 128 on the actual frozen checkpoint. If this ever fires,
        # the fix is NOT to project silently -- projection is no longer literal concatenation and
        # would have to be reported as a deviation from Model B's definition.
        raise RuntimeError(f"depth widths differ {widths}; literal concatenation undefined "
                           f"(projecting would deviate from Model B's definition -- report it)")
    return np.concatenate(mats, axis=1), ref_ids, ref_times


def run(variant: str, dseeds: list[int], config: str, init_seeds: list[int],
        csv_root: str, states: tuple[str, ...], device: str = "cpu") -> dict:
    results: dict = {"model": "B_concat", "variant": variant, "config": config,
                     "dataset_seeds": dseeds, "init_seeds": init_seeds,
                     "per_seed": {}, "states_present": [], "depth_widths": None,
                     "concat_dim": None}

    for dseed in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
        print(f"\n=== variant {variant} seed {dseed} ({os.path.basename(csv_root)}) ===",
              flush=True)
        t_start = time.time()

        model, meta = get_backbone(csv_dir, variant, dseed, mseed=0, device=device)
        assert_backbone_frozen(model)
        tr, va, te, _sup_ids = load_world(csv_dir, device)

        targets = latent_targets(variant, dseed, config,
                                 [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])
        targets = {k: v for k, v in targets.items() if k in states}
        if not results["states_present"]:
            results["states_present"] = sorted(targets)
        print(f"  states in scope: {sorted(targets)}", flush=True)

        Xtr_all, ids_tr, times_tr = concat_embeddings(depth_embeddings(model, tr))
        Xte_all, ids_te, times_te = concat_embeddings(depth_embeddings(model, te))
        if results["concat_dim"] is None:
            results["depth_widths"] = [128] * len(DEPTHS)
            results["concat_dim"] = int(Xtr_all.shape[1])
        print(f"  concat feature dim: {Xtr_all.shape[1]} "
              f"({len(DEPTHS)} depths x {Xtr_all.shape[1] // len(DEPTHS)})", flush=True)

        seed_out: dict = {}
        for state, tgt in sorted(targets.items()):
            Xtr, ytr = align(Xtr_all, ids_tr, times_tr, tgt)
            Xte, yte = align(Xte_all, ids_te, times_te, tgt)
            if Xtr is None or Xte is None:
                continue
            ytr_b, yte_b, thr = binarise(ytr, yte, tgt["kind"])
            aucs = [train_head(Xtr, ytr_b, Xte, yte_b, s, model=model) for s in init_seeds]
            aucs = [a for a in aucs if a is not None]
            if not aucs:
                continue
            seed_out[state] = {
                "auc_mean": statistics.fmean(aucs),
                "auc_all": aucs,
                "init_seed_spread": max(aucs) - min(aucs),
                "init_seed_std": statistics.stdev(aucs) if len(aucs) > 1 else 0.0,
                "n_train": int(len(ytr_b)), "n_test": int(len(yte_b)),
                "pos_train": int(ytr_b.sum()), "pos_test": int(yte_b.sum()),
                "median_threshold": thr, "in_dim": int(Xtr.shape[1]),
            }
            c = seed_out[state]
            print(f"  {state:<22} concat  AUC {c['auc_mean']:.4f}  "
                  f"floor(init spread) {c['init_seed_spread']:.4f}  "
                  f"pos {c['pos_test']:,}/{c['n_test']:,}", flush=True)
        results["per_seed"][str(dseed)] = seed_out
        print(f"  ({time.time() - t_start:.0f}s)", flush=True)

    return results


def summarise(results: dict) -> dict:
    """Aggregate across dataset seeds. Same floor definition as Phase 1's `summarise`, so the
    two models' floors are the same kind of number: init floor = worst init-seed spread, dataset
    floor = max-min across dataset seeds, gate on the larger.

    `auc_std_across_dataset_seeds` is reported alongside and is NOT the gate -- it is the
    stability number the SHARP/SHARK precedent says must be compared between arms even when the
    means match.
    """
    summary: dict = {}
    for state in results["states_present"]:
        vals, spreads, pos, n = [], [], [], []
        for dseed in results["dataset_seeds"]:
            cell = results["per_seed"].get(str(dseed), {}).get(state)
            if cell:
                vals.append(cell["auc_mean"])
                spreads.append(cell["init_seed_spread"])
                pos.append(cell["pos_test"])
                n.append(cell["n_test"])
        if not vals:
            continue
        init_floor = max(spreads)
        dseed_floor = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
        floor = max([init_floor] + ([dseed_floor] if len(vals) > 1 else []))
        mean = statistics.fmean(vals)
        summary[state] = {
            "auc_mean": mean,
            "auc_per_dataset_seed": vals,
            "n_dataset_seeds": len(vals),
            "auc_std_across_dataset_seeds": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "init_seed_floor": init_floor,
            "dataset_seed_floor": dseed_floor,
            "reproduction_floor": floor,
            "above_chance_by": mean - 0.5,
            "clears_floor": (mean - 0.5) > floor,
            "sign_consistent": all(v > 0.5 for v in vals) or all(v < 0.5 for v in vals),
            "pos_test_total": sum(pos), "n_test_total": sum(n),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--states", default=",".join(DEFAULT_STATES))
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dseeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in args.init_seeds.split(",") if s.strip()]
    states = tuple(s.strip() for s in args.states.split(",") if s.strip())

    res = run(args.variant, dseeds, args.config, iseeds, args.csv_root, states)
    res["summary"] = summarise(res)

    print("\n" + "=" * 104)
    print(f"MODEL B (concat h^0..h^4) — variant {args.variant}, config {args.config}, "
          f"{len(dseeds)} dataset seeds x {len(iseeds)} init seeds")
    print("=" * 104)
    hdr = (f"{'state':<22} {'AUC':>8} {'std':>7} {'init fl':>8} {'dset fl':>8} "
           f"{'above .5':>9} {'clears':>7} {'sign':>5} {'pos/test':>17}")
    print(hdr); print("-" * len(hdr))
    for state, c in sorted(res["summary"].items()):
        print(f"{state:<22} {c['auc_mean']:>8.4f} {c['auc_std_across_dataset_seeds']:>7.4f} "
              f"{c['init_seed_floor']:>8.4f} {c['dataset_seed_floor']:>8.4f} "
              f"{c['above_chance_by']:>+9.4f} "
              f"{('YES' if c['clears_floor'] else 'no'):>7} "
              f"{('yes' if c['sign_consistent'] else 'NO'):>5} "
              f"{c['pos_test_total']:>7,}/{c['n_test_total']:<9,}")
        print(f"{'':22} per dataset seed: "
              f"[{', '.join(f'{v:.4f}' for v in c['auc_per_dataset_seed'])}]")
    print("-" * len(hdr))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True, default=str)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
