"""
Stage 2 of the Layer 3 retrieval redesign — retrieval from **observable data only**.

Stage 1 asks whether the representation *can* encode co-membership when handed the answer.
Stage 2 asks the question a deployment would actually face: can two suppliers' shared
hidden parent be recovered from their **recorded event history**, with no privileged read
of any kind, at training time or inference time?

**Which observable, and why that one.** `reports/layer3_testing.md` §2.2's co-degradation
result -- the one that showed the signal is genuinely present in the data (mean delta
-0.0924 over 20 groups against a null of [-0.0393, +0.0429]) -- was produced by
`db/generate_dataset.py`'s own validation suite reading `stf_rows` column 4. That column
is `on_time_rate_90d` in `supplier_temporal_features.csv`, compared per supplier against
the **fleet mean at the same `as_of_date`**, restricted to snapshots inside the group's
own event windows (`_windowed_delta`). This module builds its retrieval feature from that
same column, detrended the same way, rather than from a different observable guessed at
independently -- so a Stage 2 null cannot be explained away as having pointed at the wrong
signal.

**The confound this has to survive, measured rather than assumed.** Variants B and D
enable Mechanism B (and D) only -- neither Mechanism H's shocks nor Mechanism C's rewiring
is in their generation path, so the audit's specific worry does not apply. But the base
world every variant inherits from V1 carries its own shared hidden-factor event pools
(H_PORT, H_TRUCK, H_CUSTOMS), and at `sup_n=2,000` those cover **50% of all suppliers**
and generate **185,906** co-degrading pairs against just **512** real Type A/B pairs -- a
363:1 ratio. A raw co-failure correlation therefore ranks base-pool companions far above
hidden-parent co-members by sheer weight of numbers. Two of the three pools are defined by
`country` (H_TRUCK = USA/Mexico, H_CUSTOMS = Germany) and the third by sea freight, all of
which are **observable**, so this module offers a cohort-residualised variant that removes
what the observable covariates can explain before correlating. That is still
observable-only; it is the fair version of the test rather than a handicapped one.

**Leakage guardrail (non-negotiable).** For a prediction made as of `t0`, the correlation
feeding retrieval is computed only from rows with `as_of_date < t0`. `ml/test_cofailure_leakage.py`
asserts it adversarially -- shuffling, blanking, zeroing and garbaging every post-`t0`
observation and checking the retrieved pool is identical index-for-index each time (the
score matrix agrees to 4e-15, which is float64 reduction-order noise from pandas
re-blocking, not a peek). This project has already paid once for this class of bug
(`ml/data/loader.py`'s snapshot-cache collision, `reports/layer3_testing.md` §6); the test
runs before any Stage 2 number is believed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# `supplier_temporal_features` columns that are pure recorded history. `on_time_rate_90d`
# is the one the generator's own co-degradation check reads; the 30d window is the same
# quantity at finer time resolution, which matters because a Type B flare runs ~110 days
# against a 21-month timeline and a 90-day window smears it.
OBSERVABLE_COLUMNS = ("on_time_rate_90d", "on_time_rate_30d")


def load_history(csv_dir: str, column: str = "on_time_rate_90d") -> tuple[pd.DataFrame, list]:
    """`(supplier x as_of_date)` matrix of one observable column, plus the date index.

    Read straight from the emitted CSV, never from the generator's namespace -- this is
    the half of the session that must be reproducible by anyone holding only the
    benchmark artifact.
    """
    path = f"{csv_dir}/supplier_temporal_features.csv.gz"
    df = pd.read_csv(path, usecols=["supplier_id", "as_of_date", column])
    wide = df.pivot_table(index="supplier_id", columns="as_of_date", values=column)
    return wide, list(wide.columns)


def cofailure_scores(wide: pd.DataFrame, sup_ids: list[str], as_of: str,
                     cohort: np.ndarray | None = None,
                     min_points: int = 4) -> np.ndarray:
    """`[N, N]` co-failure affinity between suppliers, as of `as_of`.

    **Only columns strictly before `as_of` are read.** That single line is the leakage
    guardrail; everything else here is bookkeeping.

    The series is the supplier's own residual against the fleet at each date -- exactly
    `_windowed_delta`'s `member r90 - fleet r90` -- and, when `cohort` is given, against
    that supplier's observable cohort mean instead, which removes the country- and
    freight-driven base event pools that would otherwise dominate the ranking.

    Affinity is the Pearson correlation of those residual series. Suppliers with fewer
    than `min_points` observations are given -inf affinity to everyone, so they are never
    retrieved rather than being retrieved on noise.
    """
    past = [c for c in wide.columns if str(c) < str(as_of)]
    X = wide.reindex(index=sup_ids)[past].to_numpy(dtype=float)   # [N, T]

    fleet = np.nanmean(X, axis=0, keepdims=True)
    resid = X - fleet
    if cohort is not None:
        # Subtract the mean residual of each supplier's own observable cohort, so a shared
        # country/freight shock no longer reads as evidence of a shared hidden parent.
        for c in np.unique(cohort):
            m = cohort == c
            if m.sum() >= 2 and np.isfinite(resid[m]).any():
                resid[m] -= np.nanmean(resid[m], axis=0, keepdims=True)

    ok = np.isfinite(resid)
    n_obs = ok.sum(axis=1)
    R = np.where(ok, resid, 0.0)
    mu = R.sum(axis=1, keepdims=True) / np.maximum(1, n_obs[:, None])
    Rc = np.where(ok, R - mu, 0.0)
    norm = np.sqrt((Rc ** 2).sum(axis=1, keepdims=True))
    norm[norm == 0] = np.inf
    Z = Rc / norm
    S = Z @ Z.T

    bad = n_obs < min_points
    S[bad, :] = -np.inf
    S[:, bad] = -np.inf
    np.fill_diagonal(S, -np.inf)
    return S


def top_k_pool(scores: np.ndarray, k: int) -> np.ndarray:
    """`[N, k]` retrieved indices, highest affinity first -- the same bounded-pool shape
    `Transformer2GlobalAttention` produces, so the two retrievers are scored identically."""
    n = scores.shape[0]
    k = min(k, n - 1)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(-scores[rows, idx], axis=1)
    return idx[rows, order]


def observable_cohort(csv_dir: str, sup_ids: list[str]) -> np.ndarray:
    """Cohort label per supplier from OBSERVABLE columns only: country crossed with a
    lead-time tercile. `country` is emitted directly and is what defines two of the three
    base event pools; lead time is the closest emitted proxy for the sea-freight flag that
    defines the third (`db/generate_dataset.py` picks H_PORT on `s["sea"]`, which is never
    written to CSV, but sea suppliers carry the long lead times).
    """
    sup = pd.read_csv(f"{csv_dir}/suppliers.csv.gz",
                      usecols=["id", "country", "lead_time_days"]).set_index("id")
    sup = sup.reindex(sup_ids)
    tercile = pd.qcut(sup["lead_time_days"].astype(float), 3, labels=False, duplicates="drop")
    key = sup["country"].astype(str) + "|" + tercile.astype(str)
    return pd.Categorical(key).codes
