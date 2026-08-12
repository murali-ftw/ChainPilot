#!/usr/bin/env python3
"""
Leakage guardrail for Stage 2's observable co-failure retrieval.

The rule: for any prediction made as of `t0`, the co-failure correlation feeding that
prediction's retrieval must be computed **only from observable events timestamped before
`t0`**. This is the same discipline `ml/data/loader.py` already enforces for Mechanism G
(`recorded_at`, never `changed_at`) and `verify_no_hidden_state()` enforces for the
mechanisms' internal state.

The test is adversarial rather than declarative: it **corrupts the future** -- shuffling,
then blanking, then replacing with garbage every observation at or after `t0` -- and
asserts the retrieval output is bit-identical each time. A retriever that peeks cannot
survive any of the three. A test that merely inspected the code for a `<` would not catch
a peek that entered through a fleet mean, a cohort mean or a normalisation constant
computed over the full matrix, which is exactly where this class of bug actually lives.

It also runs the converse: corrupting the PAST must change the output. A test that only
checks invariance passes trivially against a retriever that ignores its input entirely,
and this project has already been burned once by a check that was signal-free and looked
healthy (`db/generate_dataset.py`'s `k % 3` type assignment, Phase 0).

    python3 ml/test_cofailure_leakage.py --csv-dir db/csv_mid/vB_seed42
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.observable_cofailure import (  # noqa: E402
    cofailure_scores, load_history, observable_cohort, top_k_pool)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_mid", "vB_seed42"))
    ap.add_argument("--top-k", type=int, default=64)
    args = ap.parse_args()

    wide, dates = load_history(args.csv_dir)
    sup_ids = list(wide.index)
    cohort = observable_cohort(args.csv_dir, sup_ids)
    # The first TEST snapshot under the 40/20/40 temporal split -- the earliest t0 at
    # which a Stage 2 number would actually be reported, and so the tightest case: it has
    # the least history behind it and the most future in front of it to leak from.
    t0 = dates[int(round(0.6 * len(dates)))]
    n_future = sum(1 for d in dates if str(d) >= str(t0))
    print(f"{len(sup_ids):,} suppliers, {len(dates)} snapshot dates; "
          f"as-of t0 = {t0} ({n_future} future dates to corrupt)\n")

    rng = np.random.default_rng(0)
    base_s = cofailure_scores(wide, sup_ids, t0, cohort=cohort)
    base_p = top_k_pool(base_s, args.top_k)
    fut = [c for c in wide.columns if str(c) >= str(t0)]
    past = [c for c in wide.columns if str(c) < str(t0)]
    assert fut, "no future columns -- the test would be vacuous"
    failures = []

    # Invariance is asserted on the RETRIEVAL OUTPUT exactly (the pool must be identical,
    # index for index) and on the score matrix to a tolerance. The tolerance is not a
    # loosening: corrupting future columns leaves the past sub-matrix bit-identical, but
    # writing to a DataFrame re-consolidates pandas' internal blocks, which changes the
    # memory layout handed to BLAS and therefore the reduction order inside `Z @ Z.T`.
    # The resulting disagreement is ~4e-15, i.e. float64 rounding. Any real peek at the
    # future would move a correlation by O(0.1), thirteen orders of magnitude above this,
    # so a 1e-9 bar cannot let leakage through -- and the observed maximum is printed on
    # every line rather than hidden behind the threshold.
    TOL = 1e-9

    def check(name, w, expect_same=True):
        s = cofailure_scores(w, sup_ids, t0, cohort=cohort)
        p = top_k_pool(s, args.top_k)
        a = np.nan_to_num(s, nan=0.0, neginf=-1e9)
        b = np.nan_to_num(base_s, nan=0.0, neginf=-1e9)
        dmax = float(np.abs(a - b).max())
        same_s, same_p = dmax <= TOL, np.array_equal(p, base_p)
        ok = (same_s and same_p) if expect_same else not (same_s and same_p)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max score delta {dmax:.2e}, "
              f"pool {'identical' if same_p else 'CHANGED'} "
              f"(expected {'unchanged' if expect_same else 'changed'})")
        if not ok:
            failures.append(name)

    print("corrupting the FUTURE -- retrieval must not move:")
    w = wide.copy()
    w[fut] = w[fut].sample(frac=1.0, random_state=1).to_numpy()
    check("shuffle every post-t0 observation across suppliers", w)

    w = wide.copy()
    w[fut] = np.nan
    check("blank every post-t0 observation", w)

    w = wide.copy()
    w[fut] = rng.uniform(-1e3, 1e3, size=(len(sup_ids), len(fut)))
    check("replace every post-t0 observation with garbage", w)

    w = wide.copy()
    w[fut] = 0.0
    check("zero every post-t0 observation", w)

    print("\ncorrupting the PAST -- retrieval MUST move (guards against a signal-free test):")
    w = wide.copy()
    w[past] = w[past].sample(frac=1.0, random_state=2).to_numpy()
    check("shuffle every pre-t0 observation across suppliers", w, expect_same=False)

    print()
    if failures:
        print(f"LEAKAGE GUARDRAIL FAILED on {len(failures)} check(s): {failures}")
        return 1
    print("LEAKAGE GUARDRAIL PASSED -- retrieval as of t0 depends on the past and only "
          "on the past.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
