"""Integration (2026-10-04): behaviour anchors, recomputed on the MERGED tree from stored bundles (read-only, no training).
Each must equal the published figure at the published precision. A deviation is a STOP.

  arrival   SHARE-lite h4, 5 seeds, test: C-index mean 0.6744, lateness ROC-AUC (as-of reference) mean 0.7090
            (phase-17.md §2.4: 0.67442 / 0.70905)
  fill      22-cell h0, 5 seeds, test: exact CRPS mean 0.13876 (phase-17.md §2.3)
  capacity  mp h4, 5 seeds, test, UC3: precision 0.904 / 0.851 / 0.810 at 1 / 5 / 10% coverage, recall 0.515 / 0.318 / 0.191
            at p = 0.70 / 0.80 / 0.85 (phase-15.md §1, phase-17.md §2.2)
  blend     Phase 18 Stage 3 arrival blend (validation-fit weight 0.59): lateness ROC-AUC 0.7247 (phase18/stage3_squeeze.md;
            phase-19.md quotes it as the Phase 18 blend)

    python ml/tests/test_integration_anchors.py
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval")]
import phase12_common as C
import numpy as np
import phase18_score as S18

ANCHORS = {("arrival", "cindex"): (0.67442, 5), ("arrival", "lateness_auc"): (0.70905, 5),
           ("fill", "crps_exact"): (0.13876, 5),
           ("capacity", "precision_at_1pct"): (0.904, 3), ("capacity", "precision_at_5pct"): (0.851, 3),
           ("capacity", "precision_at_10pct"): (0.810, 3), ("capacity", "recall_at_p0.70"): (0.515, 3),
           ("capacity", "recall_at_p0.80"): (0.318, 3), ("capacity", "recall_at_p0.85"): (0.191, 3)}


def test_anchors():
    got = {}
    for task in ("arrival", "fill", "capacity"):
        b, _ = S18.arm_bands(task, lambda s: S18.load_neural(task, s), with_extra=True)
        for (t, k), (pub, dp) in ANCHORS.items():
            if t == task:
                val = b[k][1]
                got[f"{t}|{k}"] = dict(recomputed=val, published=pub, decimals=dp, equal=round(val, dp) == round(pub, dp))
    import phase18_squeeze as SQ
    out = {}
    SQ.arrival(out)
    val = out["arrival"]["point"]["blend"]["lateness_auc"]
    got["arrival|phase18_blend_lateness_auc"] = dict(recomputed=val, published=0.7247, decimals=4, equal=round(val, 4) == 0.7247,
                                                     blend_weight=out["arrival"]["blend_weight_neural"])
    bad = {k: v for k, v in got.items() if not v["equal"]}
    assert not bad, f"STOP: anchors differ from the published figures: {bad}"
    return got


if __name__ == "__main__":
    r = test_anchors()
    for k, v in r.items():
        print(f"  PASS {k}: {v['recomputed']:.6f} == published {v['published']} ({v['decimals']} dp)")
    C.dump(dict(anchors=r, stamp=C.stamp()), "integration_anchors.json")
