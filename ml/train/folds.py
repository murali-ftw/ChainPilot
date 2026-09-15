"""Guide 6.1 — time-based folds. Never random.

What runs, and what is only defined:

  fixed_split       train <= 2023, validation 2024, test 2025, inside the 2019-2025 fit window. The split every
                    Phase 2-6 number was measured on, and the one the training loop uses.
  ROLLING_ORIGINS   specification §9.2's eight origins, implemented and leak-asserted, NOT run here: they are the
                    Phase 8.2 backtest (Phase 5 Addendum B open item 2).
  covid_snapshots   calendar.regime_flag = 'covid' covers 2020-03-01 -> 2020-09-30 at every plant, which is 5 of the
                    44 training snapshots (2020-03-09 ... 2020-08-24) in both worlds. Specification §9.4 says exclude
                    them; Phases 2-5 did not, and excluding them changes the population behind every number the
                    reproduction gate checks. Implemented, OFF by default, recorded as open.

`snapshots.csv` itself carries no regime flag; regime lives in `calendar`, per plant-day.
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
from config import SPLIT, FIT_WINDOW, WORLDS

TRAIN_END, VAL_END = pd.Timestamp(SPLIT["train_end"]), pd.Timestamp(SPLIT["val_end"])
FIT_LO, FIT_HI = pd.Timestamp(FIT_WINDOW[0]), pd.Timestamp(FIT_WINDOW[1])


def fixed_split(dates):
    d = pd.to_datetime(pd.Series(dates))
    tr = d <= TRAIN_END
    va = (d > TRAIN_END) & (d <= VAL_END)
    te = d > VAL_END
    return tr.to_numpy(), va.to_numpy(), te.to_numpy()


def assert_no_leak(dates, tr, va, te):
    """guide 6.1 verify, asserted: every training date precedes every validation date precedes every test date."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    assert tr.any() and va.any() and te.any(), "an empty fold"
    assert d[tr].max() < d[va].min(), f"train {d[tr].max()} overlaps validation {d[va].min()}"
    assert d[va].max() < d[te].min(), f"validation {d[va].max()} overlaps test {d[te].min()}"
    assert d.min() >= np.datetime64(FIT_LO) and d.max() <= np.datetime64(FIT_HI), "rows outside the 2019-2025 fit window"
    assert not (tr & va).any() and not (va & te).any() and not (tr & te).any(), "a row in two folds"


def describe_fixed():
    return dict(kind="fixed", train_through=str(TRAIN_END.date()), validation=f"{TRAIN_END.date()} < d <= {VAL_END.date()}",
                test=f"d > {VAL_END.date()}", fit_window=list(FIT_WINDOW), covid_excluded=False,
                note="rolling origins (spec §9.2) defined in folds.py, not run; Phase 8.2")


# specification §9.2 -- (fold, train through, evaluate from, evaluate through)
ROLLING_ORIGINS = [
    (1, "2021-12-31", "2022-01-01", "2022-06-30"),
    (2, "2022-06-30", "2022-07-01", "2022-12-31"),
    (3, "2022-12-31", "2023-01-01", "2023-06-30"),
    (4, "2023-06-30", "2023-07-01", "2023-12-31"),
    (5, "2023-12-31", "2024-01-01", "2024-06-30"),
    (6, "2024-06-30", "2024-07-01", "2024-12-31"),
    (7, "2024-12-31", "2025-01-01", "2025-06-30"),
    (8, "2025-06-30", "2025-07-01", "2025-09-30"),
]


def rolling_origin(dates, k):
    _, through, lo, hi = ROLLING_ORIGINS[k - 1]
    d = pd.to_datetime(pd.Series(dates))
    tr = ((d >= FIT_LO) & (d <= pd.Timestamp(through))).to_numpy()
    ev = ((d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))).to_numpy()
    return tr, ev


def assert_rolling_origins(dates):
    """For every origin with data on both sides: max(train date) < min(evaluation date). Returns per-fold counts."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    out = []
    for k, *_ in ROLLING_ORIGINS:
        tr, ev = rolling_origin(dates, k)
        if tr.any() and ev.any():
            assert d[tr].max() < d[ev].min(), f"rolling origin {k} leaks: {d[tr].max()} >= {d[ev].min()}"
        out.append(dict(fold=k, train_rows=int(tr.sum()), eval_rows=int(ev.sum()),
                        eval_snapshots=int(len(np.unique(d[ev])))))
    return out


# ================================================================== Phase 8.2 — the origins, run
# Specification §9.2 gives each origin a training cut and an evaluation window, but no validation slice. The loop needs
# one (early stopping, recalibration, drift baseline), so the 12 months before the cut are carved out of training as
# validation — the same length as the fixed split's 2024, so recalibration sees a full seasonal cycle. Evaluation is
# the specified window, untouched.
VAL_MONTHS = 12


def origin_windows(k):
    _, through, lo, hi = ROLLING_ORIGINS[k - 1]
    through = pd.Timestamp(through)
    val_from = through - pd.DateOffset(months=VAL_MONTHS)
    return dict(origin=k, train=(FIT_LO, val_from), validation=(val_from, through), test=(pd.Timestamp(lo), pd.Timestamp(hi)))


def rolling_split(dates, k):
    """-> tr, va, te masks. train: FIT_LO <= d <= val_from; validation: val_from < d <= cut; test: the origin's window."""
    o = origin_windows(k)
    d = pd.to_datetime(pd.Series(dates))
    tr = ((d >= o["train"][0]) & (d <= o["train"][1])).to_numpy()
    va = ((d > o["validation"][0]) & (d <= o["validation"][1])).to_numpy()
    te = ((d >= o["test"][0]) & (d <= o["test"][1])).to_numpy()
    return tr, va, te


def describe_origin(k):
    o = origin_windows(k)
    f = lambda a, b, lo_open=False: f"{a.date()} {'<' if lo_open else '<='} d <= {b.date()}"
    return dict(kind="rolling_origin", origin=k, train=f(*o["train"]), validation=f(*o["validation"], lo_open=True),
                test=f(*o["test"]), fit_window=list(FIT_WINDOW), covid_excluded=False, validation_months=VAL_MONTHS)


def label_window_overlap(snapshot_dates, window_end, tr, va, te):
    """Diagnostic, not an assertion: training / validation rows whose label window ends on or after the first evaluation
    date. The fixed split every prior phase used has the same property (a 2023-12 training row's outcome lands in 2024),
    so asserting it would exclude the shipped split as well; it is measured and reported instead."""
    end = pd.to_datetime(pd.Series(window_end)).to_numpy()
    d = pd.to_datetime(pd.Series(snapshot_dates)).to_numpy()
    ev0, va0 = d[te].min(), d[va].min()
    return dict(train_rows_label_end_in_or_after_validation=int((tr & (end >= va0)).sum()),
                train_rows_label_end_in_or_after_test=int((tr & (end >= ev0)).sum()),
                validation_rows_label_end_in_or_after_test=int((va & (end >= ev0)).sum()),
                max_label_window_days=int(((end - d).astype("timedelta64[D]").astype(int)).max()))


def covid_snapshots(world):
    """Snapshot dates on which any plant is flagged covid. Implemented for specification §9.4; not applied."""
    cal = pd.read_csv(os.path.join(WORLDS[world], "calendar.csv"), usecols=["date", "regime_flag"])
    cal["date"] = pd.to_datetime(cal.date)
    return set(cal.loc[cal.regime_flag == "covid", "date"])


def covid_mask(world, dates):
    cov = covid_snapshots(world)
    return pd.to_datetime(pd.Series(dates)).isin(cov).to_numpy()
