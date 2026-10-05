# Phase 22 Stage 2 — the order-time arrival product

**Audience:** whoever decides what the system tells a planner at the moment a PO line is raised.
**Measured on:** v8 seed 1001 (`v8clean`: every leaking panel column replaced, Stage 1c). Replication: v8w1002
(`v8w1002clean`). Phase 21's at-placement rows (218,881 lines; test = 31,794 lines created in 2025), lead labels, UC1-P
(late vs contract, censoring resolved; **test base rate 0.349**). TEST, RAW unless marked, 5 seeds.
**Instruments:**
- LightGBM arms: `ml/baselines/phase22_proxy.py --task place` (fits `625d0cc`; strict arms `2b602b5`).
- Scorer: `ml/eval/phase22_order_time.py` (`2b602b5` / `f2e1f85`) → `order_time_v8clean.json`, `order_time_v8w1002clean.json`.
- Service: `ml/serve/order_time.py`, tests `ml/tests/test_phase22_serve.py`.

**These numbers are not comparable with the snapshot arms.**

## The week-τ convention decides whether the flag is an alert

The panel row a placement-time feature reads (pre-registration D6) covers the week **starting** on τ, the Monday on or
before creation. That row holds events up to six days **after** the line was raised: other lines placed later that week,
and this line's own early revisions. D1 calls that horizon H_week. The **strict** arms read the row of the week before τ
(H_date), the only information a planner holds on the day.

| arm (BASE_clean + L4 unless stated) | lateness AUC (vs contract) | UC1-P precision @ 5% | Phase 15 class |
|---|---|---|---|
| late flag, τ week (pre-registered) | 0.6825 [0.6812, 0.6838] | 0.794 | **ALERT** (PARTIAL at 0.80, lift 2.25) |
| **late flag, STRICT (row before τ)** | **0.6228** [0.6202, 0.6246] | **0.615** | **WATCHLIST** (NO, CEILING) |
| BASE_clean, τ week | 0.6758 | 0.714 | WATCHLIST |
| BASE_clean, STRICT | 0.6145 | — | — |
| *leaked* (BASE on the published panel + L4): the constructed offender | 0.9865 | — | flagged by the leak gate |

**The order-time late flag is an ALERT only with information from the rest of the placement week.** Strictly as-of, it
is a watchlist (precision 0.62 at 5% on a 0.35 base, per-month 0.50–0.73).

The τ-week numbers are the pre-registered ones and are the ones P3 scores. The product (§4) uses the strict flag.

Phase 21's "honest" ALERT (precision 0.79, BASE_nl) still carried six leaking columns (deviation 210). On clean inputs the
date-ranking arms fall to WATCHLIST (0.71).

## Arms (v8clean, 5-seed bands; deterministic predictors with creation-week block intervals)

| date estimator | lateness AUC | A3 (days) | expected-week hit rate | validation A3 |
|---|---|---|---|---|
| BASE_clean (τ week) | 0.6758 [0.6751, 0.6767] | 12.63 [12.58, 12.68] | 0.134 | 12.74 |
| BASE_clean + L4 | 0.6720 [0.6699, 0.6751] | 12.61 [12.48, 12.74] | 0.131 | 12.85 |
| LightGBM lateness-in-days (BASE_clean + L4) | 0.6754 [0.6734, 0.6774] | 12.87 [12.71, 13.03] | 0.141 | 13.16 |
| **standalone shrunk-KM median** (k = 10, offset −0.32 wk) | 0.532 | **10.02** (block [9.67, 10.34]) | **0.199** | **9.95** |
| KM + LightGBM blend (w_KM on validation A3) | w_KM = **1.00**: identical to KM | 10.02 | 0.199 | 9.95 |

- The KM date beats BASE_clean on A3 by **+2.60 days [2.18, 3.04]** (block, better = positive).
- It is the worst ranker of lateness: −0.144 [−0.154, −0.135].
- **The date and the flag come from different predictors.**
- The blend degenerates to the KM median: validation gives the LightGBM date no weight.

**Gates (5-seed bands vs BASE_clean; control = L4 cross-channel shuffle).** The gate self-test fires on both
constructed cases: equal bands FAIL, and a control that also gains FAILs.

| family | verdict | why |
|---|---|---|
| L4 | FAIL | nothing disjoint; and the **control is better than BASE** on both metrics (12.07 d, 0.678) |
| latedays | FAIL | A3 disjointly worse |
| flag | FAIL | lateness better (0.6825), but the control also beats BASE |

**What the control preserves (constraint 6).**
- The shuffle moves only the L4 block between channels in the same creation week, and the block carries the
  week's **L1 (global) and L2 (supplier)** columns.
- The control therefore keeps the network's as-of lead trend for that week. That is what gains, not the channel's own
  history.
- The channel history adds nothing at order time.

**Leak gate.** Every arm's feature list goes through `leak_check`. The leaked arm is flagged with all nine columns; no
clean arm is flagged. **PASS.**

## Interval (split-conformal on validation, the product's date)

- Pooled 80%: [−15.7, +39.7] days around the KM date.
- Validation coverage of the pooled interval by creation month ran 0.75 (Sep) to 0.83 (Feb), a 7.5-point spread
  (> 5), so **month-specific quantiles** were used, as D6 pre-registered.

| | overall | by creation month (test) |
|---|---|---|
| coverage | **0.803** [0.797, 0.808] | 0.767 (Jan) … 0.829 (Dec); no month < 0.753 |
| width | 53.8 d | 37.4 d (Feb) … 70.3 d (Oct) |

## Flag calibration

| | ECE-10 | Brier | mean P vs rate |
|---|---|---|---|
| RAW | 0.016 | 0.198 | 0.364 vs 0.349 |
| RECALIBRATED (isotonic on validation) | 0.024 | 0.199 | 0.336 vs 0.349 |

Recalibration does not help: the raw flag is already calibrated.

## Decision level (UC1-P, Phase 15 machinery; per-creation-month spread of precision at 5%)

| arm | class | P @ 1% / 5% / 10% / 20% | per-month P @ 5% min / median / max |
|---|---|---|---|
| BASE_clean, per seed | WATCHLIST (NO, CEILING) | 0.862 / 0.714 / 0.670 / 0.624 | 0.52 / 0.71 / 0.82 |
| LightGBM lateness-days, ensemble | WATCHLIST | 0.836 / 0.723 / 0.665 / 0.619 | 0.52 / 0.73 / 0.83 |
| late flag, τ week, per seed | **ALERT** (PARTIAL, 0.80, lift 2.25) | 1.000 / 0.794 / 0.711 / 0.637 | 0.57 / 0.81 / 0.96 |
| **late flag, STRICT, per seed** | **WATCHLIST** | 0.713 / 0.615 / 0.562 / 0.506 | 0.50 / 0.61 / 0.73 |
| KM date (pred − contract) | WATCHLIST (WEAKLY TUNABLE) | 0.381 / 0.413 / 0.399 / 0.390 | 0.25 / 0.43 / 0.58 |

Phase 21's class for UC1-P: ALERT (no-leak BASE_nl, PARTIAL at 0.80, lift 2.22).
- The **τ-week flag keeps ALERT**.
- **Every strict or clean ranking arm is WATCHLIST.**

## Replication on the second world (LightGBM proxy only; own BASE, own bands, own conformal, k re-chosen = 3)

| result | v8clean | v8w1002clean | verdict |
|---|---|---|---|
| KM date vs BASE_clean, A3 (block) | +2.60 d [2.18, 3.04] | +2.81 d [2.50, 3.10] | **REPLICATES** (ratio 1.08) |
| interval coverage (month-specific) | 0.803 | 0.811 (months 0.78–0.83) | replicates |
| flag (τ week) gate | FAIL (control gains) | PASS (lateness 0.6859 vs 0.6718; control not better) | **DOES NOT** (gate) |
| flag (τ week) class | ALERT | **WATCHLIST** (0.712 at 5% on a 0.24 base) | **DOES NOT** |
| flag STRICT | WATCHLIST, 0.6228 | WATCHLIST, 0.6229 | replicates (both watchlist) |
| L4 | FAIL | FAIL (disjointly worse) | replicates (no gain) |

The neural half is not applicable at order time (no neural model is scored at placement). Both worlds share one
generator: within-generator replication only.

## Persisted and servable (Stage 2e)

`ml/artifacts/phase22/serve/order_time_v8clean/product.json`, named `order_time|world=v8clean|date=km_k10|flag=flag_lag1_k10|conformal=month`:
- the KM date configuration (k, offset);
- the month conformal quantiles;
- the five strict flag boosters, each with its SHA-1, saved from refits asserted bit-identical to the stored predictions;
- the watch threshold: the validation top 5%.

`ml/serve/order_time.py` (new file) is the service.
- It refuses a wrong name, a tampered model, a missing bundle, or an unclean panel cache.
- It reads the panel row before τ (asserted) and group statistics as-of τ (the builder's assertions).

`ml/tests/test_phase22_serve.py`: **all PASS.**
- 200 test lines (5 creation weeks) served, with flag and date **bit-exact** to the stored predictions (max |Δ| 0.0).
- All four mismatch cases raise; the correct bundle loads.
- AST scan: 0 forbidden names; 4 / 4 constructed offenders flagged; a docstring passes.

**What it says to a planner, for a line raised today:** "Expected receipt in N weeks (the channel's survival-corrected
median lead, shrunk toward its supplier), between −16 and +40 days of that in 8 cases out of 10 (wider for lines raised
in Aug–Nov). Late-risk score p: watch it if it is in the top 5%." The late-risk score is a watchlist, not an alert: at the
top 5% it is right about 6 times in 10, against a 3.5-in-10 base.
