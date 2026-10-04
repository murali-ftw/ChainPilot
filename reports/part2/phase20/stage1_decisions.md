# Phase 20 Stage 1 — do the Phase 18–19 gains change a decision?

**Audience:** whoever decides what a planner is alerted on for arrival, fill and capacity.
**Measured on:** v8 seed 1001, fixed split, TEST (9 snapshots), RAW, stored predictions only (nothing refitted).
**Code:** `ml/eval/phase20_decisions.py` (`49aaa32` → `ml/artifacts/phase20/stage1_decisions.json`), importing Phase 15's
`coverage_curve`, `pick_on_val`, `apply`, `analyse` unchanged.
**Status:** complete. **No use case changes class.** P1 is wrong: the arrival blend is worse than the incumbent at 5%
coverage on the decision. P2 is right: fill stays a WATCHLIST.

**Rules:** Phase 15's Stage B and REACHABLE rules (verbatim in the pre-registration) and the class mapping fixed there:
ALERT = REACHABLE YES / PARTIAL with lift ≥ 1.5 at the operating bar; WATCHLIST = otherwise TUNABLE / WEAKLY TUNABLE;
RETIRED = NOT TUNABLE. Operating points are chosen on validation per seed. Ensembles and blends are one predictor, with
snapshot-block bootstrap intervals.

**Anchors reproduced exactly (Phase 15 §1):** UC1 h⁴ recall @ p 0.85 = 0.396 (test precision 0.869); UC2 boundary arm
0.475 (0.821), PARTIAL, lift 1.10; UC2b `lgbm22_id` NO, CEILING; UC3 mp h⁴ precision 0.904 / 0.851 / 0.810 at 1 / 5 / 10%,
recall 0.191 (0.810), lift 2.0.

## The decision table (test; 5-seed means for per-seed arms)

| use case · arm | base | P @ 1% | P @ 5% | P @ 10% | P @ 20% | recall @ p 0.85 (test P) | REACHABLE (bar, lift) | Stage B | **class** | accuracy vs majority |
|---|---|---|---|---|---|---|---|---|---|---|
| **UC1 late vs contract** · neural h⁴ per seed (Phase 15's) | 0.730 | 0.986 | 0.968 | 0.947 | 0.908 | 0.396 (0.869) | YES (0.90, 1.25) | TUNABLE | **WATCHLIST** | 0.731 vs 0.730 |
| UC1 · neural incumbent ensemble | 0.730 | 0.991 | **0.969** | 0.952 | 0.912 | 0.409 (0.869) | YES (0.90, 1.25) | TUNABLE | **WATCHLIST** | 0.731 vs 0.730 |
| UC1 · **Phase 19 blend** (0.52 / 0.48) | 0.730 | 0.977 | **0.943** | 0.917 | 0.875 | 0.271 (0.865) | YES (0.90, 1.25) | TUNABLE | **WATCHLIST** | 0.737 vs 0.730 |
| UC1 · LightGBM + fwd_load ensemble | 0.730 | 0.969 | 0.910 | 0.874 | 0.839 | 0.198 (0.850) | YES (0.85, 1.16) | TUNABLE | **WATCHLIST** | 0.735 vs 0.730 |
| **UC2 fill = 1** · neural incumbent ensemble | 0.749 | 0.897 | 0.881 | 0.867 | 0.852 | 0.487 (0.820) | YES (0.90, 1.16) | WEAKLY | **WATCHLIST** | 0.749 vs 0.749 |
| UC2 · neural + season + cadence ensemble | 0.749 | 0.947 | 0.924 | 0.901 | 0.873 | 0.623 (0.815) | YES (0.90, 1.18) | TUNABLE | **WATCHLIST** | 0.750 vs 0.749 |
| UC2 · **Phase 19 blend** (0.50 / 0.50) | 0.749 | 0.961 | 0.927 | 0.912 | 0.882 | 0.627 (0.827) | YES (0.90, 1.17) | TUNABLE | **WATCHLIST** | 0.752 vs 0.749 |
| UC2 · LightGBM + fwd_load ensemble | 0.749 | 0.933 | 0.908 | 0.896 | 0.876 | 0.551 (0.835) | YES (0.90, 1.18) | TUNABLE | **WATCHLIST** | 0.753 vs 0.749 |
| UC2 · shipped `b5flat22` per seed | 0.749 | 0.867 | 0.864 | 0.862 | 0.849 | 0.189 (0.856) | YES (0.85, 1.14) | TUNABLE | **WATCHLIST** | 0.749 vs 0.749 |
| **UC2b fill < 0.95** · neural incumbent ensemble | 0.245 | 0.531 | 0.464 | 0.410 | 0.374 | UNREACHABLE | NO, CEILING | TUNABLE | **WATCHLIST** | 0.562 vs 0.755 |
| UC2b · neural + season + cadence ensemble | 0.245 | 0.569 | 0.478 | 0.430 | 0.396 | UNREACHABLE | NO, CEILING | TUNABLE | **WATCHLIST** | 0.624 vs 0.755 |
| UC2b · **Phase 19 blend** | 0.245 | **0.644** | **0.530** | 0.462 | 0.414 | UNREACHABLE (p 0.70: recall 0.007) | NO, CEILING | TUNABLE | **WATCHLIST** | 0.600 vs 0.755 |
| UC2b · LightGBM + fwd_load ensemble | 0.245 | 0.647 | 0.518 | 0.459 | 0.407 | UNREACHABLE | NO, CEILING | TUNABLE | **WATCHLIST** | 0.578 vs 0.755 |
| UC2b · shipped `b5flat22` per seed | 0.245 | 0.437 | 0.384 | 0.366 | 0.345 | UNREACHABLE | NO, CEILING | TUNABLE | **WATCHLIST** | 0.415 vs 0.755 |
| **UC3 strain > 1** · mp h⁴ per seed (Phase 15's) | 0.405 | 0.904 | 0.851 | 0.810 | 0.743 | 0.191 (0.810) | PARTIAL (0.85, 2.00) | TUNABLE | **ALERT** | 0.678 vs 0.595 |
| UC3 · neural incumbent ensemble | 0.405 | 0.907 | 0.865 | 0.815 | 0.749 | 0.231 (0.803) | PARTIAL (0.85, 1.98) | TUNABLE | **ALERT** | 0.690 vs 0.595 |

Lift at the top: UC1 ≤ 1.36× anywhere (base 0.73); UC2 ≤ 1.28×; **UC2b up to 2.6×** at 1% (blend 0.644 on a 0.245 base);
UC3 2.2×.

## Snapshot-block comparisons at 5% coverage (difference, 95% interval; + = first arm better)

| use case | comparison | P @ 5% | difference | verdict |
|---|---|---|---|---|
| **UC1** | neural incumbent ensemble vs **Phase 19 blend** | 0.969 vs 0.943 | **+0.027 [0.018, 0.036]** | **the blend is WORSE** |
| UC1 | Phase 19 blend vs LightGBM + fwd_load | 0.943 vs 0.910 | +0.033 [0.020, 0.044] | blend better |
| UC2 | Phase 19 blend vs neural incumbent ensemble | 0.927 vs 0.881 | +0.038 [0.004, 0.071] | blend better |
| UC2 | Phase 19 blend vs LightGBM + fwd_load | | +0.012 [−0.004, 0.028] | undetermined |
| **UC2b** | **Phase 19 blend vs neural incumbent ensemble** | **0.530 vs 0.464** | **+0.063 [0.043, 0.090]** | **blend better** |
| UC2b | Phase 19 blend vs neural + season + cadence ensemble | 0.530 vs 0.478 | +0.047 [0.028, 0.064] | blend better |
| UC2b | Phase 19 blend vs LightGBM + fwd_load | | +0.007 [−0.015, 0.026] | undetermined |

## Per-snapshot precision at 5% coverage (top 5% of each test snapshot), so one quarter cannot carry a verdict

| arm | min | median | max | per snapshot, Jan → Dec 2025 |
|---|---|---|---|---|
| UC1 neural incumbent ensemble | 0.939 | 0.969 | 0.985 | 0.97 0.95 0.98 0.94 0.98 0.95 0.97 0.97 0.98 |
| UC1 Phase 19 blend | 0.908 | 0.938 | 0.964 | 0.95 0.91 0.96 0.92 0.94 0.92 0.94 0.92 0.94 |
| UC2b Phase 19 blend | 0.300 | 0.455 | 0.640 | 0.46 0.46 0.46 0.47 0.59 0.64 0.54 0.41 0.30 |
| UC3 neural incumbent ensemble | **0.456** | 0.808 | 0.912 | 0.81 0.74 0.90 0.81 0.88 0.91 0.90 0.75 **0.46** |

The blend is below the incumbent on UC1 in **every** snapshot (9 of 9). UC3's alert is uneven: December 2025 (0.46) is
half the other quarters.

## What changes, and what does not

- **No use case changes class.** UC1 and UC2 stay WATCHLIST, UC2b stays WATCHLIST (NO, CEILING), UC3 stays ALERT.
- **Arrival (P1 wrong).** The Phase 18–19 lateness-AUC gain was measured against the as-of reference on uncensored rows.
  On the decision (late vs contract, censoring resolved, top of the ranked list) the blend's point score ranks the top
  5% **worse** than the incumbent's distributional P(late): 0.943 vs 0.969, in every snapshot. The gain lives in the
  middle of the ranking, not at its top. **For a ranked late-list, keep the incumbent ensemble's P(late).** The blend's
  value is the point estimate (A3 12.34 days), not the alert.
- **Fill.** The forward plan moves the **materially-short list** (UC2b) the most: precision in the top 5% rises from
  0.464 (incumbent) to 0.530 (blend), disjointly, and the top 1% reaches 0.644 at 2.6× the base rate. It still cannot hold
  p = 0.70 with 10% recall, so **fill remains a WATCHLIST (P2 right)**, a better one. The shipped `b5flat22` is the
  weakest fill arm on every decision metric here (UC2b P @ 5% 0.384).
- **Capacity.** Unchanged (no Phase 19 arm was asked for here). The incumbent ensemble keeps the ALERT, with the
  December 2025 snapshot at 0.46 as the caution.
- **Accuracy is no argument for any of these.** Arrival and UC2 sit at the majority baseline. UC2b's F1-optimal
  threshold is below the majority's accuracy, which is the expected trade for a minority-class list.
