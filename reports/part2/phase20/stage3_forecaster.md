# Phase 20 Stage 3 — can a learned load forecaster recover more of the hindsight gap?

**Audience:** whoever decides whether to build a demand / load forecasting component in front of the risk models.
**Measured on:** v8 seed 1001 and the second world v8w1002 (Stage 2), fixed split, TEST, 5 seeds, LightGBM proxy (CPU).
**Code:** `ml/data/fwd_pred.py`, `ml/baselines/phase20_forecaster.py` (`d50e50f`), proxy arms `ml/baselines/phase20_proxy.py`,
gate `ml/eval/phase20_score.py` (`c303519`), recovery shares `reports/part2/phase20/PRIVILEGED__score.py` (`a98f2b9`).
**Status:** complete on both worlds. **The forecast feature PASSES on arrival and fill in both worlds and FAILS on
capacity in both** (P6 wrong). It recovers 3–4% of the hindsight gap on arrival and 14–37% on fill and capacity's top points.

## (a)–(b) The forecaster

| | |
|---|---|
| targets | log1p of the channel's realised ordered quantity in weeks 1–4 / 5–8 / 9–13 after t0 (PO lines created then): observed labels, never inputs |
| inputs (all as-of t0) | fwd_load (15), fwd_season (15), cadence (5), ordering history (quantity over 4 / 13 / 52 weeks, line count over 52; asserted `recorded_ts ≤ t0`), flat static channel features |
| model | one LightGBM L2 regressor per window, frozen GBM config, seeds 7 / 17 / 27 / 37 / 47 |
| purge | K = 5 contiguous blocks of the 44 train snapshots (2019-01 to 2019-12, 2020-01 to 2020-12, 2021-02 to 2022-01, 2022-02 to 2023-01, 2023-03 to 2023-12); each block's model trains on eligible snapshots ≥ 13 weeks from the block on both sides |
| eligible targets | **41** train snapshots; **3 excluded** (their 13-week window passes the train end) |
| val / test predictions | one model on all 41 eligible snapshots; early stopping on the 6 validation snapshots whose window ends by the validation end |
| supplier level | the sum of the supplier's channel predictions (pre-registered) |

## (c) The forecaster's own accuracy on TEST (seed 7; seeds agree to ±0.005)

| window | world | MAE forecaster | MAE naive plan | R² forecaster | R² naive plan | R² (log) forecaster | R² (log) naive plan | supplier R² |
|---|---|---|---|---|---|---|---|---|
| weeks 1–4 | v8 | **143.0** | 150.4 | 0.059 | **0.346** | **0.282** | −0.244 | −2.57 |
| weeks 5–8 | v8 | **148.8** | 150.2 | −0.020 | **0.346** | **0.231** | −0.263 | −3.10 |
| weeks 9–13 | v8 | 176.3 | **166.0** | 0.045 | **0.397** | **0.277** | −0.108 | −2.95 |
| weeks 1–4 | world 2 | **113.7** | 123.4 | 0.081 | **0.370** | **0.288** | −0.323 | −2.05 |

**Read plainly:** the forecaster is better on the log scale and on MAE in the near windows, and **much worse on raw-quantity
R²**. It is fitted in log1p, and back-transforming the mean log shrinks large orders toward zero. Summed to suppliers,
that shrinkage compounds (supplier R² −2 to −3). It is a good *ranking* of which channels will order, and a poor
*quantity* estimate. Nothing was retuned after seeing this; the pre-registered design stands.

## (d)–(e) The forecast feature downstream (Stage 0 rule)

Arms: BASE + fwd_load (the reference); BASE + fwd_load + fwd_pred; and the same with fwd_pred **snapshot-permuted
within split** (the control). Pass = disjointly better than BASE + fwd_load on ≥ 1 primary metric (capacity ≥ 3 of 6),
worse on none, control not better than BASE + fwd_load.

| world · use case | BASE + fwd_load | **+ fwd_pred** | + fwd_pred permuted | PRIVILEGED hindsight | **recovery share** | **verdict** |
|---|---|---|---|---|---|---|
| v8 · arrival lateness AUC | 0.7186 | **0.7206** [0.7199, 0.7212] | 0.7181 | 0.7792 | **3.2%** | **PASS** |
| v8 · arrival A3 (days) | 12.96 | 12.93 | 12.96 | 11.32 | 1.8% | |
| v8 · fill CRPS | 0.1355 | **0.1351** [0.1351, 0.1351] | 0.1354 | 0.1326 | **14.3%** | **PASS** |
| v8 · fill P(full) AUC | 0.6582 | **0.6623** [0.6621, 0.6625] | 0.6590 | 0.6885 | **13.7%** | |
| v8 · capacity P @ 1 / 5 / 10% | 0.877 / 0.813 / 0.796 | 0.900 / **0.842** / 0.806 | 0.862 / 0.808 / 0.796 | 0.967 / 0.891 / 0.832 | 26% / **37%** / 28% | **FAIL** (2 of 6 better) |
| v8 · capacity R @ 0.70 / 0.80 / 0.85 | 0.443 / 0.226 / 0.112 | **0.474** / 0.206 / 0.120 | 0.454 / 0.226 / 0.122 | 0.674 / 0.462 / 0.347 | 13% / −8% / 3% | |
| world 2 · arrival lateness AUC | 0.6966 | **0.7001** [0.6997, 0.7007] | 0.6960 | 0.7772 | **4.2%** | **PASS** |
| world 2 · fill CRPS / AUC | 0.0924 / 0.6605 | **0.0922 / 0.6648** | 0.0923 / 0.6622 | 0.0902 / 0.6995 | 9.6% / 11.0% | **PASS** |
| world 2 · capacity P @ 5% | 0.675 | 0.685 | 0.669 | 0.810 | 7.1% | **FAIL** (0 of 6 better) |

Recovery share = (fwd_pred − fwd_load) / (hindsight − fwd_load), 5-seed means. The world-2 hindsight arm is world 2's own
(`PRIVILEGED__hindsight_w2.py`); nothing is borrowed.

## Reading

- **The forecaster adds a small, real, replicated gain on arrival and fill.** It passes both gates in both worlds, and
  permuting it across snapshots erases it. But it recovers **3–4%** of arrival's hindsight gap (P5's "< 25%" holds easily)
  and **10–14%** of fill's.
- **Capacity: the largest share, but not a pass.** It recovers 37% of the gap at 5% coverage in v8 (P5's "≥ 15%" holds at
  the use case's headline point), yet only 2 of 6 points are disjoint, and on world 2 none is. Its precision gain at the
  top of the list does not survive the second world.
- **Why so little:** the gap is the realised order book, and most of it is noise a forecaster cannot see at t0. The
  forecaster beats the plan at ranking channels but not at quantity (raw R² 0.06 vs 0.35). The forward plan, which
  forecasts demand rather than orders, is already the better quantity signal.
- **Recommendation:** not worth a separate forecasting component for these use cases. If one is built anyway, fit it
  on the quantity scale (or correct the log back-transform), since the downstream models read quantities.
