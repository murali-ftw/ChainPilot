# Phase 23AC T1 — the order-time arrival product, demo-ready, against what Rane has today

**Audience:** whoever demos the order-time product and decides whether it replaces the promise date in planning.
**Measured on:**
- v8 seed 1001: 31,794 test lines raised in 2025, 30,818 receipted, 53 creation-week blocks.
- Replication: v8w1002.

All numbers are RECOMPUTED from stored artifacts. No model is fitted; the product is the Phase 22 bundle, unchanged.
**Instruments:** `ml/serve/order_time_card.py`, `ml/eval/phase23ac_t1_compare.py`, `ml/tests/test_phase23ac_t1_card.py`
(all at `d401cb3`) → `ml/artifacts/phase23ac/t1/compare_v8.json`, `compare_v8w1002.json`, `card_tests.json`;
`reports/part2/phase23ac/t1/compare_*.csv`, `sample_cards.json`. Intervals: creation-week-block bootstrap, 1,000 resamples.

## Reproduction first

| check | here | Phase 22 (`order_time_v8clean.json`) |
|---|---|---|
| product test A3 (days) | 10.0173 | 10.0173 (\|Δ\| 0.0) |
| interval test coverage | 0.80258 | 0.80258 |
| test rows, Y, EV vs stored Phase 22 BASE file | identical | — |

## a. The planner card

`OrderTimeCard` wraps the Phase 22 service. Every field comes from `svc.predict` or the service's own group statistics.

**Tests** (`card_tests.json`) — all PASS:
- **Bit-exact:** 200 test lines (5 creation weeks × 40), late-risk score and expected weeks with max |Δ| = **0.0**.
- **Guard:** a wrong name, a tampered model, a missing bundle and a foreign service each raise.
- **AST scan:** clean; 7 constructed offenders flagged.

The 200 cards resolved to the channel (L4) for 191 lines and to channel × month (L5) for 9. 7 carried the top-5% watch mark.

## b. Against what Rane has today (v8, receipted test lines, all as-of the day the line is raised)

| predictor | A3 (median \|error\|) | mean \|error\| | received within ±7 d | signed bias (pred − actual) |
|---|---|---|---|---|
| **the product** (shrunk Kaplan–Meier median + offset) | **10.0 d** | **17.3 d** | **37.8%** | **−8.4 d** |
| promise date (contracted lead) | 14.0 d | 18.4 d | 27.1% | +1.7 d |
| channel trailing mean lead (52 weeks, as-of) | 12.3 d | 19.5 d | 32.0% | −0.4 d |
| channel trailing median lead (52 weeks, as-of) | 11.0 d | 19.2 d | 35.4% | −3.8 d |

The channel averages fall back to the contracted lead for 1.1% of lines (no receipt in 52 weeks). Their as-of filter
fires when deliberately removed (10,108 future receipts caught at the last τ).

Baseline minus product (block 95% interval; positive = product better):

| baseline | A3 (days) | mean \|error\| (days) | within ±7 d (points) |
|---|---|---|---|
| promise date | **+3.75** [2.73, 4.31] | **+1.08** [0.63, 1.54] | **+10.7** [9.1, 12.3] |
| trailing mean | **+2.27** [2.02, 2.53] | **+2.16** [1.81, 2.50] | **+5.8** [4.9, 6.6] |
| trailing median | **+1.21** [0.85, 1.73] | **+1.88** [1.63, 2.14] | **+2.4** [1.7, 3.2] |

**The product is better than all three on every measure, block-disjoint.**

**One caution for the demo:** its mean signed error is **−8.4 days**, so on average it says *earlier* than the receipt.
It is a median of a right-skewed lead distribution, so it is accurate for the typical line but the late tail pulls the
mean. The interval, not the point, is what covers the tail.

**Interval** (month-specific split-conformal; coverage of receipted lines):

| creation month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coverage | **0.767** | 0.804 | 0.784 | 0.791 | 0.814 | 0.791 | 0.794 | 0.802 | 0.823 | 0.811 | 0.823 | 0.829 | **0.803** |
| width (days) | 39 | 37 | 44 | 50 | 45 | 47 | 51 | 64 | 70 | 70 | 65 | 59 | 54 |

**The leaked-arm gate fires.** The product's flag feature list read from the published panel is flagged by T3's
`check_feature_list` on all nine columns. The clean list is flagged on none.

## c. World 2 (own k = 3 chosen on world-2 validation, own offset, own conformal quantiles; one generator)

| predictor | A3 | mean \|error\| | within ±7 d | bias |
|---|---|---|---|---|
| **the product** | **8.8 d** | **15.0 d** | 41.5% | −3.1 d |
| promise date | 14.0 d | 17.7 d | 24.0% | +7.3 d |
| channel trailing mean | 9.9 d | 16.1 d | 38.6% | +0.5 d |
| channel trailing median | 9.0 d | 15.7 d | **42.2%** | −2.1 d |

| baseline | A3 vs product | mean \|error\| vs product | within ±7 d vs product | verdict |
|---|---|---|---|---|
| promise | better [4.95, 5.93] | better | better | replicates |
| trailing mean | better [0.88, 1.22] | better | better | replicates |
| **trailing median** | **undetermined** [−0.05, +0.39] | better [0.52, 1.03] | **worse** [−0.013, −0.001] | **does NOT replicate** |

- On world 2 the product still beats the promise date and the trailing mean clearly, and the trailing median on mean
  error.
- But it **ties** the trailing median on A3 and is **slightly worse** at hitting ±7 days.
- Interval coverage: 0.813 overall; months 0.788–0.835 (May 0.835, Nov 0.833 slightly over 0.83).

Phase 22 chose a KM + LightGBM blend for world 2. Per the pre-registration, T1 uses KM (k = 3) and recomputes its own
conformal quantiles for it (deviation 227).

## d. Five sample cards (seed 2310, receipted test lines), beside the outcome

| line | raised | card says | 80% range | late-risk | resolved to | actually received | miss |
|---|---|---|---|---|---|---|---|
| POL001031653 | 2025-11-21 | 2025-12-14 | 11-29 … 02-02 | 0.28 | channel, n 76 | 2025-12-21 | 7 d late, inside |
| POL001020846 | 2025-10-16 | 2025-11-25 | 11-12 … 01-22 | 0.24 | channel, n 53 | 2025-12-08 | 13 d late, inside |
| POL001022124 | 2025-10-22 | 2025-11-10 | 10-29 … 01-07 | 0.31 | channel, n 74 | 2025-11-15 | 5 d late, inside |
| POL000969260 | 2025-04-26 | 2025-06-10 | 05-24 … 07-13 | 0.32 | channel, n 57 | 2025-05-26 | **15 d early**, inside |
| POL000944011 | 2025-01-13 | 2025-03-03 | 02-13 … 03-25 | 0.39 | channel, n 37 | 2025-04-12 | **40 d late, OUTSIDE** |

Example card text: *"Expected around 2025-03-03 (week 10); 8 in 10 such lines arrive between 2025-02-13 and
2025-03-25. Late-risk 0.39: watchlist score, below the watch cut."* That line arrived on 12 April, a January-raised line,
the month the interval covers least. Two of five cards miss by more than 14 days, so no extra miss line was needed.

## Predictions

- **P1 RIGHT** (scored on v8 as pre-registered): block-disjointly better than all three baselines on A3, and a higher ±7-day
  share. *On world 2 the trailing-median comparison does not replicate.*
- **P2 WRONG:** overall 0.803 is in band, but January is 0.767 (< 0.77). On world 2 two months exceed 0.83.
