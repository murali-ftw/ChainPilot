#!/usr/bin/env python3
"""
Phase 3's mandatory gate — is the explanation faithful, or merely plausible?

This project's own history makes this test non-optional. `reports/layer3_testing.md` §9's
contrastive arm produced retrieval that looked confident and was shown to be memorisation, and
the same report flags repeatedly that attention weights and importance scores are not
explanations until the cited factor is removed and the prediction is confirmed to move as the
explanation implied. **An explanation that fails this test is not reported as a finding; it is
reported as a failed explanation.**

The test, for the top-attributed factor of each sampled prediction:

1. Ablate exactly that factor -- set that one feature of that one entity to the baseline,
   holding every other feature of every other node fixed -- and re-run the frozen forward pass.
2. **Direction:** the prediction must fall (the factor was cited as *raising* the risk).
3. **Magnitude:** the observed drop must be at least `rel_tol` of the drop the attribution
   claimed. Attribution and ablation are the same operation here, so a mismatch means the
   attribution was computed on a graph state that no longer holds, not that two different
   methods disagree.
4. **It must beat a null -- and the null has to be chosen carefully, because the obvious one
   is circular.** The cited factor is the *argmax* of occlusion attribution, and the
   faithfulness ablation is the *same operation*, so comparing it against other features of
   the **same** entity is won by construction and proves nothing. A pooled cross-entity
   threshold is not circular but is mis-scaled: one global percentile taken over all entities
   is set by the most ablation-sensitive ones, so an intrinsically insensitive prediction can
   never clear it however faithful its explanation is. Measured on seed 42, that pooled
   formulation passed 1 of 240 explanations while direction and magnitude both passed 100% --
   the threshold, not the explanation, was doing the work.

   The null used here is therefore a **specificity** test: ablate the *same cited feature* on
   other entities, and require the cited entity's response to exceed the 90th percentile of
   that distribution. This asks the question that actually matters -- "is this feature
   unusually important *for this entity*, or does occluding it move everyone equally" -- and
   it cannot be won by construction.

A factor passes only if all three hold. The headline number is the **pass rate**.

    python3 ml/test_explanation_faithfulness.py --seeds 42,43,44,45,46 --task delay
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, predict  # noqa: E402
from ml.explain_prediction import (  # noqa: E402
    _Shim, baseline_vector, occlusion_attribution, saturation_report, select_targets,
    top_factors)
from ml.models.depth import TASK_ENTITY_TYPE  # noqa: E402

REL_TOL = 0.5          # observed drop must be >= 50% of the claimed drop
NULL_PCTL = 90         # percentile of the cross-entity specificity null


@torch.no_grad()
def ablate_one(model, bundle, target: int, feat: int, task: str,
               bvec: torch.Tensor, base_p: np.ndarray | None = None) -> float:
    """Prediction change from ablating one feature of one entity: `p_before - p_after`.

    `base_p` is the unablated prediction vector for the snapshot. It is identical for every
    ablation on that snapshot, so passing it in removes one forward pass per call -- roughly
    half the cost of the whole faithfulness sweep, which is thousands of passes.
    """
    et = TASK_ENTITY_TYPE[task]
    x = bundle.data[et].x
    before = (predict(model, bundle)[task][target] if base_p is None else base_p[target])
    saved = x[target, feat].clone()
    x[target, feat] = bvec[feat]
    try:
        after = predict(model, _Shim(bundle.data))[task][target]
    finally:
        x[target, feat] = saved
    return float(before - after)


def specificity_null(model, bundle, cited_feat: int, task: str, bvec: torch.Tensor,
                     pool_idx: np.ndarray, rng: np.random.Generator, base_p: np.ndarray,
                     n_draws: int = 40) -> tuple[float, list[float]]:
    """Null for one cited feature: its ablation effect on **other comparable** entities.

    Non-circular, because it varies the entity rather than the feature. A factor passes only
    if occluding it hurts *this* entity more than it hurts a typical comparable one -- the
    difference between "this feature matters" and "this feature matters *here*".

    **The pool must be matched, and getting that wrong makes the test vacuous in either
    direction.** Drawing the pool uniformly at random returns mostly *saturated* entities
    (96% of delay predictions sit at p >= 0.999), where ablating anything moves nothing; the
    null then collapses to ~0 and every explanation passes trivially -- measured at pass rate
    1.000 on seed 42 before this was fixed. `pool_idx` is therefore the same non-saturated
    population the targets were drawn from, so the comparison is like for like.
    """
    pool = rng.choice(pool_idx, size=min(n_draws, len(pool_idx)), replace=False)
    vals = [abs(ablate_one(model, bundle, int(t), int(cited_feat), task, bvec, base_p))
            for t in pool]
    return (float(np.percentile(vals, NULL_PCTL)) if vals else 0.0), vals


def run_seed(csv_dir: str, variant: str, dseed: int, mseed: int, task: str,
             n_targets: int, baseline: str, rng_seed: int,
             target_mode: str = "stratified") -> dict:
    model, meta = get_backbone(csv_dir, variant, dseed, mseed, device="cpu", verbose=False)
    et = TASK_ENTITY_TYPE[task]
    rng = np.random.default_rng(rng_seed)
    per_snapshot = []

    for bi, b in enumerate(meta["test_bundles"]):
        p = predict(model, b)[task]
        if len(p) == 0:
            continue
        # Stratified across the risk range, EXCLUDING the saturated tail. Targeting the
        # highest-risk entities returns zero explanations: 96% of delay predictions sit at
        # p >= 0.999, where occluding one feature cannot move a saturated sigmoid and every
        # attribution is identically zero (ml/explain_prediction.py::select_targets).
        sat = saturation_report(p)
        # The population the null is drawn from: same non-saturated band as the targets.
        usable_idx = np.where((p < 0.999) & (p > 1e-4))[0]
        targets = select_targets(p, n_targets, mode=target_mode, rng_seed=rng_seed)
        if len(usable_idx) < 5:
            per_snapshot.append({"snapshot": bi, "n_targets": 0, "n_with_factor": 0,
                                 "saturation": sat})
            continue
        if len(targets) == 0:
            per_snapshot.append({"snapshot": bi, "n_targets": 0, "n_with_factor": 0,
                                 "saturation": sat})
            continue
        phi, names = occlusion_attribution(model, b, targets, task, baseline)
        tops = top_factors(phi, names, k=1)
        bvec = baseline_vector(b.data, et, baseline)

        cited, claimed = [], []
        for r in range(len(targets)):
            if tops[r]:
                cited.append(tops[r][0]["index"])
                claimed.append(tops[r][0]["attribution"])
            else:
                cited.append(-1)
                claimed.append(0.0)

        have = [r for r in range(len(targets)) if cited[r] >= 0]
        if not have:
            per_snapshot.append({"snapshot": bi, "n_targets": int(len(targets)),
                                 "n_with_factor": 0, "saturation": sat})
            continue

        # One null per distinct cited feature, reused across the targets citing it.
        null_by_feat, nulls = {}, []
        for f in sorted({cited[r] for r in have}):
            null_by_feat[f], v = specificity_null(model, b, f, task, bvec, usable_idx, rng, p)
            nulls.extend(v)
        tau_f = float(np.mean(list(null_by_feat.values()))) if null_by_feat else 0.0

        rows = []
        for r in have:
            obs = ablate_one(model, b, int(targets[r]), int(cited[r]), task, bvec, base_p=p)
            direction_ok = obs > 0
            magnitude_ok = obs >= REL_TOL * claimed[r]
            beats_null = abs(obs) >= null_by_feat[cited[r]]
            rows.append({
                "target": int(targets[r]), "p": float(p[targets[r]]),
                "feature": names[cited[r]], "claimed": float(claimed[r]),
                "observed": float(obs),
                "direction_ok": bool(direction_ok), "magnitude_ok": bool(magnitude_ok),
                "beats_null": bool(beats_null),
                "null_for_feature": float(null_by_feat[cited[r]]),
                "passed": bool(direction_ok and magnitude_ok and beats_null),
            })
        per_snapshot.append({
            "snapshot": bi, "n_targets": int(len(targets)), "n_with_factor": len(have),
            "tau_f": tau_f, "null_median": float(np.median(nulls)) if nulls else None,
            "saturation": sat, "rows": rows,
        })

    flat = [r for s in per_snapshot for r in s.get("rows", [])]
    sats = [s["saturation"] for s in per_snapshot if "saturation" in s]
    summary = {"n_explanations": len(flat),
               "mean_saturated_high_frac": (float(np.mean([x["saturated_high_frac"]
                                                           for x in sats])) if sats else None),
               "mean_explainable_frac": (float(np.mean([x["explainable_frac"]
                                                        for x in sats])) if sats else None)}
    if flat:
        summary.update({
            "pass_rate": float(np.mean([r["passed"] for r in flat])),
            "direction_rate": float(np.mean([r["direction_ok"] for r in flat])),
            "magnitude_rate": float(np.mean([r["magnitude_ok"] for r in flat])),
            "beats_null_rate": float(np.mean([r["beats_null"] for r in flat])),
            "mean_claimed": float(np.mean([r["claimed"] for r in flat])),
            "mean_observed": float(np.mean([r["observed"] for r in flat])),
            "top_features": _counts([r["feature"] for r in flat]),
            "top_features_passing": _counts([r["feature"] for r in flat if r["passed"]]),
        })
    return {"dseed": dseed, "mseed": mseed, "auc": meta["auc"],
            "per_snapshot": per_snapshot, "summary": summary}


def _counts(xs: list[str]) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--model-seeds", default="0")
    ap.add_argument("--task", default="delay")
    ap.add_argument("--n-targets", type=int, default=40)
    ap.add_argument("--baseline", default="median")
    ap.add_argument("--rng-seed", type=int, default=20260813)
    ap.add_argument("--target-mode", default="stratified", choices=("stratified", "top"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "explain_faithfulness.json"))
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    mseeds = [int(s) for s in a.model_seeds.split(",") if s.strip()]
    blob = {"config": vars(a), "runs": []}
    for m in mseeds:
        for d in dseeds:
            csv_dir = os.path.join(a.csv_root, f"v{a.variant}_seed{d}")
            r = run_seed(csv_dir, a.variant, d, m, a.task, a.n_targets, a.baseline,
                         a.rng_seed, a.target_mode)
            blob["runs"].append(r)
            s = r["summary"]
            print(f"  d{d} m{m}: n={s.get('n_explanations',0)} "
                  f"pass={s.get('pass_rate')} dir={s.get('direction_rate')} "
                  f"mag={s.get('magnitude_rate')} null={s.get('beats_null_rate')}",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
