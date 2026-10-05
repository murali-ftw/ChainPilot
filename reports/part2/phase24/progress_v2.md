# Phase 24: progress v2 (Phase 23AC progress.md with the two PENDING PHASE 23B rows filled)

Changed rows only: *Predict-the-rescue* and *Shortage simulation* (and the matching summary line). Everything else is `reports/part2/phase23ac/t4/progress.md` verbatim. Code `ml/eval/phase24_metrics_v2.py` at `cb6590e`.


**Audience:** whoever presents ChainPilot's progress to Aptimeta / Rane, and the modelling team.
**Measured on:** v8 seed 1001, TEST 2025, RAW, seeds 7 / 17 / 27 / 37 / 47. Full tables: `metrics_pack.md`.
**Code:** `ml/eval/phase23ac_t4_metrics.py` (`d401cb3`); charts: `ml/eval/phase23ac_t4_charts.py`
(a separate interpreter with matplotlib, deviation 224).
**Companions:** `definitions.md`, `metrics_pack.md`.
**Status:** complete.

## a. One summary table per use case

### UC1: arrival late vs contract (snapshot rows; base 0.73)

Headline: `cov.5%.precision` (higher is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | neural lite h4, per seed (Phase 15's arm), P(late) | incumbent neural 5-seed ensemble P(late) (Phase 20 kept it) | clean incumbent neural 5-seed ensemble P(late) |
| headline | 0.968 [0.963, 0.971] | 0.969 [0.955, 0.978] | 0.948 [0.930, 0.965] |
| base rate | 0.730 | 0.730 | 0.730 |
| Phase 15 class | WATCHLIST | WATCHLIST | WATCHLIST |

- **What the numbers mean:** of the 5% of cases the clean model ranks highest, 95% turn out positive, against 73% of all cases: right 9.5 times in 10 at the flagged cases against 7.3 in 10 by chance.
- **What we would need from Rane:** "for each supplier × part × plant (or the finest grain you hold), on-hand stock, on-order quantity, safety stock and reorder point **as they stood on each date**: snapshots or a ledger, with recorded timestamps." (Ask 2, `docs/client/data_request_v2.md`); "every PO line with its creation timestamp and recorded timestamp, partial receipts as **separate rows** (not overwritten), and the reorder rules or MRP parameters in force (review cycle, lot sizes, minimum order quantities, cover targets), with the dates they changed." (Ask 5, `docs/client/data_request_v2.md`)

### Arrival point estimate (snapshot rows)

Headline: `a3_days` (lower is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | neural lite h4, per seed: expected week | Phase 19 blend (w_neural 0.52) | clean Phase 19 recipe (w_neural 0.52) |
| headline | 13.06 [12.86, 13.30] | 12.34 [11.95, 12.84] | 12.57 [12.22, 13.05] |

- **What the numbers mean:** half of the clean predicted arrival dates are within 12.6 days of the actual arrival.
- **What we would need from Rane:** "for each supplier × part × plant (or the finest grain you hold), on-hand stock, on-order quantity, safety stock and reorder point **as they stood on each date**: snapshots or a ledger, with recorded timestamps." (Ask 2, `docs/client/data_request_v2.md`); "every PO line with its creation timestamp and recorded timestamp, partial receipts as **separate rows** (not overwritten), and the reorder rules or MRP parameters in force (review cycle, lot sizes, minimum order quantities, cover targets), with the dates they changed." (Ask 5, `docs/client/data_request_v2.md`)

### Order-time arrival: expected date and 80% interval (Phase 22 product)

Headline: `a3_days` (lower is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | — (none) | — (none) | product date: shrunk-KM median (k = 10) + offset; month-specific split-conformal 80% interval |
| headline | — | — | 10.02 [9.67, 10.34] (Q) |

- **What the numbers mean:** half of the clean predicted arrival dates are within 10.0 days of the actual arrival.
- **What we would need from Rane:** "every PO line with its creation timestamp and recorded timestamp, partial receipts as **separate rows** (not overwritten), and the reorder rules or MRP parameters in force (review cycle, lot sizes, minimum order quantities, cover targets), with the dates they changed." (Ask 5, `docs/client/data_request_v2.md`)

### Order-time arrival: strict as-of late flag (base 0.35)

Headline: `cov.5%.precision` (higher is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | — (none) | Phase 21 at-placement flag (BASE_nl; still carried six leaking columns, deviation 210) | strict as-of late flag (flag_lag1), per seed |
| headline | — | 0.790 (Q) | 0.615 [0.607, 0.619] |
| base rate | — | 0.350 | 0.349 |
| Phase 15 class | — | ALERT | WATCHLIST |

- **What the numbers mean:** of the 5% of cases the clean model ranks highest, 62% turn out positive, against 35% of all cases: right 6.2 times in 10 at the flagged cases against 3.5 in 10 by chance.
- **What we would need from Rane:** "every PO line with its creation timestamp and recorded timestamp, partial receipts as **separate rows** (not overwritten), and the reorder rules or MRP parameters in force (review cycle, lot sizes, minimum order quantities, cover targets), with the dates they changed." (Ask 5, `docs/client/data_request_v2.md`); "for each supplier × part × plant (or the finest grain you hold), on-hand stock, on-order quantity, safety stock and reorder point **as they stood on each date**: snapshots or a ledger, with recorded timestamps." (Ask 2, `docs/client/data_request_v2.md`)

### UC2: fill arrives in full (base 0.749)

Headline: `cov.5%.precision` (higher is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | boundary bw3, per seed (Phase 15's UC2 arm) | Phase 19 fill blend (w_neural 0.5) | consolidated blend (w_neural 0.02) |
| headline | 0.876 [0.866, 0.884] | 0.927 [0.859, 0.948] | 0.944 [0.928, 0.964] |
| base rate | 0.749 | 0.749 | 0.749 |
| Phase 15 class | WATCHLIST | WATCHLIST | WATCHLIST |

- **What the numbers mean:** of the 5% of cases the clean model ranks highest, 94% turn out positive, against 75% of all cases: right 9.4 times in 10 at the flagged cases against 7.5 in 10 by chance.
- **What we would need from Rane:** "every version of the forward requirement plan (or the MRP demand forecast) you have kept, at part × plant × week grain, each with the date that version was published or recorded." (Ask 1, `docs/client/data_request_v2.md`)

### UC2b: fill materially short, < 0.95 (base 0.245)

Headline: `cov.5%.precision` (higher is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | lgbm22_id, per seed (Phase 15's UC2b arm) | Phase 19 fill blend (w_neural 0.5) | clean Phase 19 fill blend (w_neural 0.49) |
| headline | 0.393 [0.389, 0.398] | 0.530 [0.444, 0.567] | 0.511 [0.446, 0.549] |
| base rate | 0.245 | 0.245 | 0.245 |
| Phase 15 class | WATCHLIST | WATCHLIST | WATCHLIST |

- **What the numbers mean:** of the 5% of cases the clean model ranks highest, 51% turn out positive, against 24% of all cases: right 5.1 times in 10 at the flagged cases against 2.4 in 10 by chance.
- **What we would need from Rane:** "every version of the forward requirement plan (or the MRP demand forecast) you have kept, at part × plant × week grain, each with the date that version was published or recorded." (Ask 1, `docs/client/data_request_v2.md`)

### Fill distribution

Headline: `crps_exact` (lower is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | neural none_h0, per seed (Phase 15 fill incumbent) | Phase 19 fill blend (w_neural 0.5) | consolidated blend (w_neural 0.02) |
| headline | 0.13876 [0.13866, 0.13882] | 0.13537 [0.11540, 0.15391] | 0.13444 [0.11526, 0.15244] |

- **What the numbers mean:** the clean fill distribution's average CRPS is 0.1344 on a 0-1 fill scale (lower is better); it ranks lines by P(full) only modestly.
- **What we would need from Rane:** "every version of the forward requirement plan (or the MRP demand forecast) you have kept, at part × plant × week grain, each with the date that version was published or recorded." (Ask 1, `docs/client/data_request_v2.md`)

### UC3: capacity strain > 1 in the next 90 days (base 0.405)

Headline: `cov.5%.precision` (higher is better).

| | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| arm | mp h4, per seed (Phase 15's arm) | incumbent 5-seed ensemble (mean quantiles) | clean mp h4, per seed |
| headline | 0.851 [0.832, 0.860] | 0.865 [0.762, 0.896] | 0.735 [0.703, 0.784] |
| base rate | 0.405 | 0.405 | 0.405 |
| Phase 15 class | ALERT | ALERT | WATCHLIST |

- **What the numbers mean:** of the 5% of cases the clean model ranks highest, 73% turn out positive, against 41% of all cases: right 7.3 times in 10 at the flagged cases against 4.1 in 10 by chance.
- **What we would need from Rane:** "every declared-capacity statement per supplier (with its date), and any evidence of actual capacity: audits, shift patterns, maximum demonstrated output, allocation constraints, notices of shutdowns or slowdowns." (Ask 4, `docs/client/data_request_v2.md`)

**Class changes (published → clean):** ORDER-TIME FLAG: ALERT → WATCHLIST; UC3: ALERT → WATCHLIST.

## b. Use cases without a clean model metric

- **Shortage head (retired):** RETIRED (not re-measured). 0.224 precision @ 5% coverage at a 3% base (re-weighted; 0.223 subsampled) (LEAKY-UNCONFIRMED; `reports/part2/phase-17.md` §1.1).
- **Predict-the-rescue:** RESTATED by Phase 23B. Clean B1a **0.733** [0.7325, 0.7336] precision at recall 0.254 against a 0.4535 base (lift 1.62), still GO, ALERT (PARTIAL @ 0.80) but marginal under block resampling ([0.697, 0.773]); published 0.823 was LEAKY (superseded, Phase 22) (QUOTED; `reports/part2/phase-23b.md` §1).
- **Shortage simulation:** RESTATED by Phase 24 on clean heads. Pre-rescue ratio **1.137x** (was 1.129x); rescue-detection precision **0.612** (was 0.619); rescued weeks missed **66.2%** (was 65.9%) (RECOMPUTED; `reports/part2/phase24/stage2_simulation.md`).
- **Delivery schedule (MILP):** blocked. no model metric: blocked. Needs holding, ordering, freight and shortage costs; 0 of 4 exist (n/a; `results/observation2.md` §3.6).
- **Supplier allocation:** blocked. no model metric: blocked. 57 of 120 part-plants have no allowed split; on the 63 feasible it ties the status quo (n/a; `results/observation2.md` §3.6).
- **Transfer recommendation:** blocked. no model metric: blocked. Its confidence is NOT TUNABLE (lift 0.03); "always yes" beats it on F1 (n/a; `results/observation2.md` §3.6).
- **What we would need from Rane (MILP, allocation, transfer):** "inventory carrying cost (rate or per-unit-week), ordering / setup cost per purchase order, freight structure (truck capacity and tariff), and the cost of a shortage (per unit, or per line stop)." (Ask 3, `docs/client/data_request_v2.md`).

## c. Charts (PNG in `charts/`, source CSV in `charts/data/`)

| chart | message | source CSV |
|---|---|---|
| `1_headline.png` | headline metric per use case, three columns | `reports/part2/phase23ac/t4/charts/data/chart1_headline.csv` |
| `2_precision_vs_base.png` | precision at the flagged cases against the base rate | `reports/part2/phase23ac/t4/charts/data/chart2_precision_vs_base.csv` |
| `3_capacity_snapshots.png` | capacity precision at 5% per test snapshot; worst quarter marked | `reports/part2/phase23ac/t4/charts/data/chart3_capacity_snapshots.csv` |
| `4_order_time.png` | order-time expected-date error vs promise date and channel averages | `reports/part2/phase23ac/t4/charts/data/chart4_order_time.csv` |
| `5_leak_story.png` | the nine leaking columns (share of channels changed) | `reports/part2/phase23ac/t4/charts/data/chart5_leak_columns.csv` |
| `5_leak_story.png` | the drop per use case when the leak is removed | `reports/part2/phase23ac/t4/charts/data/chart5_leak_drop.csv` |

## d. Honest summary (one page)

1. Phase 22 found nine leaking panel columns. Every INCUMBENT and BEST PUBLISHED figure is superseded; quote CLEAN only.
2. late list, precision @ 5%: 0.968 → 0.969 (leaky) → **0.948** clean.
3. capacity, precision @ 5%: 0.851 → 0.865 (leaky) → **0.735** clean.
4. materially-short list, precision @ 5%: 0.393 → 0.530 (leaky) → **0.511** clean.
5. arrival point A3 (days): 13.06 → 12.34 (leaky) → **12.57** clean.
6. fill CRPS: 0.13876 → 0.13537 (leaky) → **0.13444** clean.
7. New at order time: the expected date is within 10.02 days (median); the strict late flag is right 6.2 times in 10 in its top 5%.
8. Class changes (published → clean): ORDER-TIME FLAG: ALERT → WATCHLIST; UC3: ALERT → WATCHLIST. No use case gains a class on clean inputs.
9. Predict-the-rescue (clean 0.733, still GO, marginal) and the shortage simulation (restated on clean heads, Phase 24) are no longer pending. MILP, allocation and transfer are blocked on data.
10. 0 recomputed figure(s) did not reproduce a published value (FINDINGS in `metrics_pack.md`).
