#!/usr/bin/env python3
"""
Validation for §10's metric implementations, run before any §10 number is believed.

§9.2 validated `ml/retrieval_metrics.py` end to end against a perfect retriever and a random
one before quoting a single retrieval figure, on the grounds that a metric bug and a null
result look identical in a table. §10's headline quantity is **Expected Calibration Error**,
which is worse in that respect: a broken ECE reports a small number, and a small number is
exactly what a good result looks like. So the same discipline applies here, plus one check
that has no analogue in §9 -- a **deliberately miscalibrated** predictor, whose ECE must come
back *large* and at a value computable by hand.

    python3 ml/test_hypothesis_calibration.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.hypothesis_ranker import (  # noqa: E402
    average_precision, ece, isotonic_apply, isotonic_fit, roc_auc)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main() -> int:
    rng = np.random.default_rng(20260813)

    # -- ROC AUC against a known-good implementation, and at both degenerate extremes.
    p = rng.random(4000)
    y = (rng.random(4000) < 0.5 * p + 0.1).astype(int)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        check("roc_auc matches sklearn", abs(roc_auc(p, y) - roc_auc_score(y, p)) < 1e-9,
              f"({roc_auc(p, y):.6f} vs {roc_auc_score(y, p):.6f})")
        check("average_precision matches sklearn",
              abs(average_precision(p, y) - average_precision_score(y, p)) < 1e-9,
              f"({average_precision(p, y):.6f} vs {average_precision_score(y, p):.6f})")
    except ImportError:
        check("sklearn available for cross-check", False, "(skipped)")
    check("roc_auc of a perfect ranker is 1.0",
          roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0)
    check("roc_auc of a constant predictor is 0.5 (ties averaged)",
          roc_auc(np.full(100, 0.3), np.r_[np.zeros(50), np.ones(50)]) == 0.5,
          "-- the check that catches a tie-handling bug reading as signal")
    check("roc_auc is None on a degenerate class",
          roc_auc(p, np.zeros(len(p), dtype=int)) is None)

    # -- ECE on a predictor that is calibrated BY CONSTRUCTION: draw y ~ Bernoulli(p).
    # With 200k samples the estimate is dominated by binomial noise, which at 10 equal-count
    # bins of 20k is ~0.0035 -- so "small" here has a number attached to it rather than being
    # eyeballed.
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    e, curve = ece(p, y)
    check("ECE of a perfectly calibrated predictor is at binomial noise", e < 0.005,
          f"(ECE {e:.5f}, expected ~0.0035 from 10 bins of 20,000)")
    check("reliability curve tracks the diagonal",
          max(abs(b["confidence"] - b["empirical"]) for b in curve) < 0.01,
          f"(max |conf - emp| {max(abs(b['confidence'] - b['empirical']) for b in curve):.5f})")

    # -- A predictor that always says 0.9 on a class that is true 30% of the time. ECE must
    # be exactly 0.6. This is the check a broken ECE fails, and no small-number-looks-good
    # reading can rescue it.
    p_over = np.full(100_000, 0.9)
    y_over = (rng.random(100_000) < 0.3).astype(int)
    e_over, _ = ece(p_over, y_over)
    check("ECE of a 0.9-always predictor on a 30% class is 0.60",
          abs(e_over - 0.6) < 0.005, f"(ECE {e_over:.4f}, expected 0.600)")

    # -- Isotonic regression: must be monotone, must fix a squashed predictor, and must not
    # damage one that was already calibrated.
    q = rng.random(50_000)
    yq = (rng.random(50_000) < q).astype(int)
    squashed = q * 0.2 + 0.75          # monotone in q, badly miscalibrated
    curve_iso = isotonic_fit(squashed, yq)
    check("isotonic output is monotone non-decreasing",
          bool(np.all(np.diff(curve_iso[1]) >= -1e-9)))
    fixed = isotonic_apply(curve_iso, squashed)
    e_before, _ = ece(squashed, yq)
    e_after, _ = ece(fixed, yq)
    check("isotonic repairs a squashed predictor", e_after < 0.02 < e_before,
          f"(ECE {e_before:.4f} -> {e_after:.4f})")
    e_id_before, _ = ece(q, yq)
    e_id_after, _ = ece(isotonic_apply(isotonic_fit(q, yq), q), yq)
    check("isotonic does not damage an already-calibrated predictor",
          e_id_after <= e_id_before + 0.005,
          f"(ECE {e_id_before:.4f} -> {e_id_after:.4f})")
    # Isotonic is monotone NON-decreasing, so it merges scores into flat blocks (50,000
    # distinct values -> 89 here). Merging creates ties, and ties cost AUC, so a monotone map
    # cannot *improve* discrimination out of sample -- it can only lose a little to ties.
    # Checked the way the pipeline actually uses it, fit on one split and applied to another:
    # fitting and scoring on the SAME points does show a small AUC *rise* (+0.0005 measured),
    # which is the isotonic fit overfitting the ranking, and is precisely why
    # `run_hypothesis_module.py` fits the curve on a held-out calibration seed.
    half = len(q) // 2
    curve_half = isotonic_fit(squashed[:half], yq[:half])
    out_of_sample = isotonic_apply(curve_half, squashed[half:])
    a_before = roc_auc(squashed[half:], yq[half:])
    a_after = roc_auc(out_of_sample, yq[half:])
    check("out-of-sample isotonic does not improve ranking, and loses only ties",
          a_after <= a_before + 1e-12 and a_before - a_after < 0.01,
          f"(AUC {a_before:.6f} -> {a_after:.6f}, delta {a_after - a_before:+.6f})")

    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{'ALL CHECKS PASSED' if not failed else str(len(failed)) + ' CHECKS FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
