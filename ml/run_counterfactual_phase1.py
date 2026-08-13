#!/usr/bin/env python3
"""
Phase 1 runner — does the naive counterfactual track the generator's true causal effect?

Pairs, per `(supplier, snapshot)`:

* the **predicted** delta -- edit the graph, re-run the frozen SHARE + Markov forward pass,
  subtract (`ml/counterfactual_edit.py`);
* the **true** delta -- recompute the generator's own `stress -> p_delay -> p_impact` chain
  under the same structural change, holding the realised schedule fixed
  (`ml/counterfactual_ground_truth.py`).

**The partition that decides whether any of this means anything.** A structural change to the
co-parent graph moves the true risk of a handful of suppliers and leaves every other supplier
at *exactly* zero -- the generator's coupling term reaches only direct co-parents. Message
passing, by contrast, spreads the edit across the graph, so the model emits nonzero deltas for
many suppliers the generator says are untouched. Pooling the two populations would let a
metric be dominated by entities whose true answer is zero. Every number below is therefore
reported separately for:

* **affected** -- `|true delta| > eps`; the only place sign agreement is defined at all;
* **bystanders** -- `true delta == 0` exactly; scored by a *false-effect rate*, which is a
  different question and deserves a different number.

**Chance level is 0.5** for sign agreement on affected entities, and the comparison is against
that, with sign-consistency across the five dataset seeds reported the way every other result
in this project reports it.

    python3 ml/run_counterfactual_phase1.py --seeds 42,43,44,45,46 --interventions 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml.counterfactual_edit import naive_counterfactual  # noqa: E402
from ml.counterfactual_ground_truth import (  # noqa: E402
    CausalWorld, apply_intervention, generator_namespace, sample_interventions)
from ml.ds_backbone import get_backbone, predict  # noqa: E402

EPS = 1e-9
# A predicted delta below this is treated as "the model said nothing". Set at the scale of
# float noise in a sigmoid difference, not at a level chosen to flatter the false-effect rate.
PRED_EPS = 1e-4


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation, ties averaged. Written out rather than imported so the tie handling
    is inspectable -- these deltas contain many exact ties at zero."""
    if len(a) < 3:
        return None
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1)
        sx = x[order]
        i = 0
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def run_seed(dseed: int, variant: str, config: str, csv_root: str, n_per_kind: int,
             mseed: int, sample_seed: int, task: str) -> dict:
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
    model, meta = get_backbone(csv_dir, variant, dseed, mseed, device="cpu")
    test = meta["test_bundles"]
    sup_ids = meta["supplier_ids"]
    sup_index = {s: i for i, s in enumerate(sup_ids)}

    ns = generator_namespace(variant, dseed, config)
    world = CausalWorld(ns)
    n_all = len(world.T0S)
    # `split_bundles` is a positional 40/20/40 by snapshot COUNT; the test block is the tail.
    n_train = int(round(0.4 * n_all))
    n_val = int(round(0.2 * n_all))
    test_offset = n_train + n_val
    assert len(test) == n_all - test_offset, (
        f"test bundle count {len(test)} != expected {n_all - test_offset}")

    p0 = world.p_impact(world.base_coparents)
    specs = sample_interventions(world, n_per_kind, sample_seed)

    # The unedited prediction is identical for every intervention on a given snapshot, so it
    # is computed once per snapshot rather than once per (intervention, snapshot) -- half the
    # forward passes across the grid.
    baselines = [predict(model, b)[task] for b in test]

    rows = []
    for spec in specs:
        kind = spec["kind"]
        cop, owner, touched = apply_intervention(world, kind, spec)
        p1 = world.p_impact(cop, owner)

        for j, bundle in enumerate(test):
            ti = test_offset + j
            pred = naive_counterfactual(model, bundle, kind, spec, sup_index, task=task,
                                        baseline=baselines[j])
            if pred is None:
                continue
            true = np.zeros(len(sup_ids), dtype=float)
            defined = np.zeros(len(sup_ids), dtype=bool)
            for si, s in enumerate(sup_ids):
                k = (s, ti)
                if k in p0:
                    defined[si] = True
                    v1 = p1.get(k)
                    # A supplier whose shipments cease to exist has no defined post-value;
                    # excluded rather than imputed, since inventing one would be a choice
                    # the generator does not make.
                    true[si] = (v1 - p0[k]) if v1 is not None else np.nan
            ok = defined & np.isfinite(true)
            rows.append({"kind": kind, "t0_index": ti,
                         "true": true[ok], "pred": pred[ok],
                         "touched": np.array([sup_ids[i] in touched
                                              for i in np.where(ok)[0]])})
    return {"dseed": dseed, "mseed": mseed, "rows": rows,
            "auc": meta["auc"], "n_specs": len(specs)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    kinds = sorted({r["kind"] for r in rows})
    for kind in kinds + ["ALL"]:
        sel = rows if kind == "ALL" else [r for r in rows if r["kind"] == kind]
        true = np.concatenate([r["true"] for r in sel]) if sel else np.array([])
        pred = np.concatenate([r["pred"] for r in sel]) if sel else np.array([])
        if not len(true):
            continue
        aff = np.abs(true) > EPS
        by = ~aff
        d = {"n_pairs": int(len(true)),
             "n_affected": int(aff.sum()), "n_bystander": int(by.sum())}
        if aff.sum():
            ta, pa = true[aff], pred[aff]
            agree = np.sign(ta) == np.sign(pa)
            # Entities where the model said nothing at all cannot agree on sign; counted
            # separately so a "silent" model is not scored as if it disagreed.
            silent = np.abs(pa) < PRED_EPS
            d.update({
                "sign_agreement": float(agree.mean()),
                "sign_agreement_excl_silent": (float(agree[~silent].mean())
                                               if (~silent).sum() else None),
                "silent_fraction": float(silent.mean()),
                "spearman": spearman(ta, pa),
                "true_abs_mean": float(np.abs(ta).mean()),
                "pred_abs_mean": float(np.abs(pa).mean()),
                "magnitude_ratio": float(np.abs(pa).mean() / max(np.abs(ta).mean(), 1e-12)),
            })
        if by.sum():
            d["false_effect_rate"] = float((np.abs(pred[by]) > PRED_EPS).mean())
            d["bystander_pred_abs_mean"] = float(np.abs(pred[by]).mean())
        out[kind] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="0")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--model-seeds", default="0",
                    help="more than one measures the model-seed reproduction floor "
                         "for every metric below")
    ap.add_argument("--interventions", type=int, default=40, help="of EACH of the three kinds")
    ap.add_argument("--sample-seed", type=int, default=20260813)
    ap.add_argument("--task", default="impact")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "cf_phase1.json"))
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    mseeds = [int(s) for s in a.model_seeds.split(",") if s.strip()]
    blob = {"config": vars(a), "per_seed": []}

    for mseed in mseeds:
        for dseed in dseeds:
            r = run_seed(dseed, a.variant, a.config, a.csv_root, a.interventions,
                         mseed, a.sample_seed, a.task)
            s = summarise(r["rows"])
            blob["per_seed"].append({"dseed": dseed, "mseed": mseed,
                                     "auc": r["auc"], "n_specs": r["n_specs"],
                                     "summary": s})
            al = s.get("ALL", {})
            print(f"  d{dseed} m{mseed}: affected={al.get('n_affected',0):,} "
                  f"sign={al.get('sign_agreement')} "
                  f"rho={al.get('spearman')} "
                  f"false_effect={al.get('false_effect_rate')}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
