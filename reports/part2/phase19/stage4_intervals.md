# Phase 19 Stage 4 — honest intervals, and what ships

**Audience:** whoever decides what ships for arrival, fill and capacity, and how much to trust each claimed gain.
**Measured on:** v8 seed 1001, fixed split, TEST (9 test snapshots), RAW, stored and Phase 19 predictions.
**Code:** `ml/eval/phase19_intervals.py`: 4a at `7c9ad7a` (→ `ml/artifacts/phase19/stage4a_block.json`), 4b at
`11f0c93` (→ `stage4b_block.json`). CPU.
**Status:** complete.

**Method:** 1,000 resamples of **whole test snapshots** with replacement, paired across arms; 95% interval of the
difference (positive = better). A verdict **survives** when the block interval still excludes 0 in the same direction
as Phase 18's row-bootstrap interval. Blend weights are refitted on validation (Phase 18's criteria).

## 4a — Phase 18's comparisons under the block bootstrap

**The block intervals are wider:** 48 of 53 comparisons widen, by a median of 1.9× (range 0.94×–16.6×). The widest
are capacity recall and fill CRPS between families, where whole snapshots move together.

| use case · comparison | metric | Phase 18 row CI | **block CI** | survives |
|---|---|---|---|---|
| arrival · **blend vs neural ensemble** | lateness AUC | +[0.0083, 0.0138] | **+[0.0084, 0.0136]** | **yes** |
| arrival · blend vs neural ensemble | A3 (days, + = fewer) | +[0.46, 0.72] | +[0.47, 0.75] | **yes** |
| arrival · blend vs LightGBM ensemble | lateness AUC / A3 / C-index | better / better / better | better / better / better | **yes** |
| arrival · neural ensemble vs single-seed mean | lateness AUC / C-index | better / better | better / better | **yes** |
| arrival · neural ensemble vs LightGBM ensemble | lateness AUC | +[0.0020, 0.0142] | [−0.0008, 0.0165] | **no** → undetermined |
| fill · neural ensemble vs single-seed mean | CRPS / AUC / ECE | better / better / better | better / better / better | **yes** |
| fill · neural ensemble vs LightGBM ensemble | CRPS | +[0.0007, 0.0015] | [−0.0007, 0.0030] | **no** |
| fill · neural ensemble vs LightGBM ensemble | P(full) AUC | +[0.014, 0.023] | +[0.0006, 0.038] | yes |
| fill · neural ensemble vs LightGBM ensemble | ECE | worse [−0.043, −0.032] | [−0.058, +0.002] | **no** |
| fill · blend vs LightGBM ensemble | CRPS / ECE | better / worse | undetermined / undetermined | **no / no** |
| capacity · neural ensemble vs LightGBM ensemble | all six points | better | better (P @ 1% +[0.018, 0.276]) | **yes** |
| capacity · neural ensemble vs single-seed mean | recall @ 0.70 / 0.80 / 0.85 | better | better | **yes** |
| capacity · neural ensemble vs single-seed mean | precision @ 5% | +[0.003, 0.024] | [−0.001, 0.026] | **no**\* |

\* 4a scores capacity's ensemble from the **mean of the quantiles**, as the blend is built. Phase 18's squeeze scored
it from the mean of the per-seed P(strain > 1), so the capacity ensemble rows compare slightly different predictors, and
Phase 18's capacity blend-vs-ensemble differences (±0.001) vanish by construction (w = 1.00) (deviation 174).

**Arrival interval coverage** (Phase 18 Stage 3c), per seed, with block 95% intervals (nominal 80%):

| interval | coverage (5 seeds) | block CI, widest seed |
|---|---|---|
| raw P10–P90 | 0.921–0.943 | [0.935, 0.951] |
| split-conformal on P50 (pre-registered in Phase 18) | 0.854–0.870 | [0.847, 0.869] |
| **split-conformal on the expected week** | **0.791–0.806** | [0.779, 0.803] |

The expected-week interval stays inside 0.78–0.82 on every seed's point estimate under the block bootstrap.

## 4b — the Phase 19 neural arms: ensemble and validation-fit blend

Neural side = the Phase 19 bound arms (Stage 3); LightGBM side = **LightGBM + fwd_load** (Phase 18's stored arm), as
P8 states. Validation-fitted blend weight on the neural side: arrival 0.52, fill 0.50, capacity 1.00.

| use case | metric | incumbent neural ensemble (4a) | **Phase 19 neural ensemble** | LightGBM + fwd_load ensemble | **Phase 19 blend** |
|---|---|---|---|---|---|
| arrival | lateness AUC | 0.7138 | 0.7170 | 0.7192 | **0.7313** |
| arrival | A3 (days) | 13.07 | 12.90 | 12.95 | **12.34** |
| fill | exact CRPS | 0.13853 | 0.1369 | 0.1354 | **0.1354** |
| fill | P(fill = 1) AUC | 0.6240 | 0.6447 | 0.6594 | **0.6639** |
| fill | ECE-22 | 0.091 | 0.092 | **0.033** | 0.060 |
| capacity | precision @ 1 / 5 / 10% | — | 0.951 / 0.908 / 0.867 | 0.853 / 0.812 / 0.797 | = neural (w 1.00) |
| capacity | recall @ 0.70 / 0.80 / 0.85 | — | 0.542 / 0.330 / 0.197 | 0.442 / 0.235 / 0.110 | = neural |

Block-bootstrap verdicts:

| comparison | arrival | fill | capacity |
|---|---|---|---|
| **Phase 19 blend vs INCUMBENT neural ensemble** | **better**: lateness +0.017 [0.015, 0.020]; A3 −0.74 d [0.54, 0.97]; C-index undet. | **better**: CRPS +0.0032 [0.0023, 0.0041]; AUC +0.039 [0.025, 0.058]; ECE +0.031 [0.005, 0.061] | = ensemble row |
| Phase 19 neural ensemble vs INCUMBENT neural ensemble | better: lateness +0.003 [0.001, 0.005]; A3 −0.18 d [0.01, 0.39] | better: CRPS +0.0016; AUC +0.020; ECE undet. | **better at 10%** (+0.052 [0.009, 0.098]); the other five undetermined |
| Phase 19 blend vs Phase 19 neural ensemble | **better** (lateness +0.014, A3 −0.56 d) | **better** (CRPS, AUC, ECE) | identical (w 1.00) |
| Phase 19 blend vs LightGBM + fwd_load ensemble | better (lateness +0.012, A3 −0.61 d, C-index) | AUC better; CRPS undet.; **ECE worse** (−0.026) | better on 5 of 6 (P @ 1% undet.) |

## 4c — shipping table (only verdicts that survived the block bootstrap)

| use case | **ships now** | blocked on data | closed |
|---|---|---|---|
| **arrival** | The **Phase 19 blend**: 0.52 × neural (fwd_load + season + cadence) + 0.48 × LightGBM + fwd_load. Lateness AUC **0.731** vs the incumbent ensemble's 0.714 (+0.017 [0.015, 0.020]); A3 **12.34** vs 13.07 days. Interval: **split-conformal on the expected week**, ~80% coverage at ~45 days (raw P10–P90: 93% at 52 days) | **Order timing.** Knowing when the line is raised is worth +0.205 lateness AUC (Stage 2), but no as-of input reaches it: cadence recovers 4%. It needs a signal of the planner's reorder trigger, which the generator keeps in `inventory_position_weekly`, a table never used as a feature | supplier / transit / regime state for arrival (Phase 18: adds 0.004); the global pulse (Phase 18); lane / checkpoint (Phase 18: no grouping in v8) |
| **fill** | The **Phase 19 blend**: 0.50 × neural (season + cadence) + 0.50 × LightGBM + fwd_load. Beats the incumbent neural ensemble on CRPS, AUC **and** ECE. Against LightGBM + fwd_load alone: better AUC, CRPS undetermined, worse ECE, so a planner who reads P(fill = 1) as a calibrated probability should keep LightGBM's calibration. **Recorded for a decision, not applied:** `ml/configs/shipped.json` ships LightGBM `b5flat22` and is protected in this phase | the supplier's weekly queue (Phase 18 oracle: +0.17 AUC from realised load at creation) is not observable at t0 | supplier-specific fwd_load for fill (Gate v2 FAIL: its gain survives a snapshot permutation, so it is a static descriptor, not a forecast) |
| **capacity** | The **Phase 19 neural ensemble** (mp h⁴ + fwd_load, 5 seeds): not worse than the incumbent ensemble on any point and **better at 10% coverage** (+0.052 [0.009, 0.098]). The single-arm verdict is a TIE, so this is a marginal improvement, not a step change | **the monthly capacity draw K** (Phase 18: ~60% of the oracle gap; declared capacity tracks it at 0.11) | fwd_season alone for capacity (worse at 1% coverage); cadence for capacity (worse at 10%); trailing-window conformal (Phase 18: no better than static) |
