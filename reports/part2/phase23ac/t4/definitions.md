# Phase 23AC T4.1: definitions, written before any metric is computed

**Audience:** anyone who reads `metrics_pack.md` or `progress.md`, or presents a figure from them.
**Measured on:** v8 seed 1001. Fixed split: train ≤ 2023, val 2024, test 2025. Seeds 7 / 17 / 27 / 37 / 47. Each figure
is RECOMPUTED from stored test predictions or QUOTED from a named report or artifact, and is marked as one or the other.
**Companions:** `reports/part2/phase-23ac-preregistration.md` (T4 and the gates), `reports/part2/phase-15.md` §1,
`reports/part2/phase20/stage1_decisions.md`, `reports/part2/phase-22.md` §1 and §6, `docs/client/data_request_v2.md`.
**Status:** definitions only. No number below was computed by this phase. The base rates are the published ones, and the
pack recomputes them.

## The three columns

| column | what it is | label |
|---|---|---|
| **INCUMBENT** | Phase 15's arm per use case (stored, published panel) | **LEAKY (superseded, Phase 22)** |
| **BEST PUBLISHED** | the best arm published in Phases 19–20 (stored, published panel) | **LEAKY (superseded, Phase 22)** |
| **CLEAN** | Phase 22's arm, on `v8clean`, where every one of the nine leaking panel columns is replaced | clean |

A leaky figure can never sit in the CLEAN column. The table builder raises if it is asked to put one there. Both the
arm's leaky flag and its source path are checked: a path under `bundles/<task>/v8_`, `phase7_preds/`, `phase18/preds/`,
`phase19/bundles/` or `capacity_published.json` is published, and so leaky.

## Common rules

- **Operating point (Phase 15 `pick_on_val`):** on VALIDATION only, per seed. It is the lowest threshold (the most
  recall) whose validation precision is ≥ p with ≥ 50 alerts, for p ∈ {0.70, 0.80, 0.85, 0.90}, applied unchanged to test.
  If no validation threshold reaches p, that bar is **UNREACHABLE on validation**.
- **Phase 15 class** (Phase 20 mapping, `phase20_decisions.reachable_and_class`):
  - **REACHABLE YES** if the validation-chosen point holds test precision ≥ 0.85, with recall ≥ 0.10 and ≥ 50 alerts, on
    every seed. **PARTIAL** if the same holds at ≥ 0.70.
  - **ALERT** = YES or PARTIAL, with lift ≥ 1.5 at the highest bar that holds.
  - **WATCHLIST** = not ALERT, and Stage B is TUNABLE or WEAKLY TUNABLE.
  - **RETIRED** = Stage B is NOT TUNABLE.
- **Accuracy** is read at the VALIDATION max-F1 threshold (Phase 20 rule 3) and is always printed beside the
  majority-class accuracy, max(base, 1 − base).
- **Precision at coverage c** is the precision of the top ⌈c·n⌉ test rows by score (stable sort), for c ∈ {1, 5, 10, 20}%.
  **Lift** is precision / base rate.
- **ROC-AUC and PR-AUC** are each computed twice:
  - by sklearn (`roc_auc_score`, `average_precision_score`);
  - by our own implementation: the Mann–Whitney statistic with mid-ranks for ROC-AUC, and the step-wise average
    precision over distinct score thresholds for PR-AUC.

  The two must agree to 1e-6. PR-AUC's floor is the base rate.
- **"Right N times in 10":** at the flagged cases (the top 5%), N = 10 × precision and M = 10 × base rate, each to one
  decimal: "right N times in 10 at the flagged cases against M in 10 by chance".
- **Intervals:**
  - Single-model arms report the 5-seed mean [min, max]. The scorers require all five seeds.
  - Ensembles and blends are one predictor. They report the point estimate and a snapshot-block bootstrap 95% interval:
    1,000 resamples of whole test snapshots, seed fixed, paired across arms on the same rows. The thresholds stay at
    their validation values inside each resample.
- **Spread:** every headline also gets its per-snapshot spread (min, median, max, with the worst snapshot named), or
  for the order-time arms its spread per creation month.

## Use cases

### UC1: arrival "late versus contract" (snapshot rows)

- **Binary question:** will this open line arrive later than its contracted lead?
- **Positive class:** late. Uncensored: Y > R. Censored: R < 13 (`phase21_score.uc1_label`). Here R is the contracted
  reference (`phase22_restate.rc`). Rows that are censored with R ≥ 13 are dropped.
- **Base rate:** **0.730** (Phase 15 §1).
- **Majority-class baseline:** "every line is late". Accuracy 0.730, precision 0.730, recall 1.0, so lift 1.0.
- **Score:** P(late) = S(R) off the hazard distribution (`phase14_score.p_late`) for distribution arms; for point arms,
  prediction − R.
- **Arms:**
  - INCUMBENT: neural lite h4, per seed.
  - BEST PUBLISHED: the incumbent 5-seed ensemble's P(late). Phase 20 kept it for the late list.
  - CLEAN: the clean incumbent 5-seed ensemble's P(late).

### Arrival point estimate (snapshot rows)

- **Questions answered:**
  - **A3:** the median |7·(predicted week − actual week)| in days, on uncensored rows. MAE is the mean of the same.
  - **Signed bias:** the mean of 7·(predicted − actual) in days. Positive means the prediction is later than the actual.
  - **C-index:** censoring-aware concordance (`metrics.cindex`, 1 M sampled pairs, seed 7).
  - **Lateness AUC:** the ROC-AUC of (prediction − R_asof) for (Y > R_asof), on uncensored rows, against the as-of
    channel reference (`docs/specs/lateness_metric.md`).
  - **Signed-days lateness** (Phase 17 A3) is the same signed error read against the contract. It is defined here and
    reported through the signed bias.
- **Interval** (neural arms only): the hazard distribution's central 80%, from the first week where the CDF reaches 0.1
  to the first week where it reaches 0.9. When the CDF never reaches 0.9 within 12 weeks, the upper end is open.
  - Coverage is measured on uncensored rows.
  - Width is in days, on rows with a closed upper end.
  - The share of rows with an open upper end is reported.
  - Blends have no interval (n/a).
- **Arms:**
  - INCUMBENT: neural h4, per seed (expected week, `phase18_score.expected_week`).
  - BEST PUBLISHED: the Phase 19 blend, 0.52 × the Phase 19 neural ensemble + 0.48 × the LightGBM fwd_load ensemble.
  - CLEAN: the clean Phase 19 recipe, w = 0.52 × the clean neural ensemble + 0.48 × the clean LightGBM fwd_load
    ensemble (w from `restate_arrival_fill_capacity.json`).

### Order-time arrival (Phase 22 product; at-placement lines, test = lines created in 2025)

- **Expected date:** A3 (days), the share within ±7 days, and the signed bias. QUOTED: `order_time_v8clean.json` gives
  A3; the share within ±7 days and the bias come from Track T1's `compare_v8.json` when present, and are otherwise
  marked PENDING T1.
- **80% interval** (month-specific split-conformal): coverage and width, overall and per creation month. QUOTED from
  `order_time_v8clean.json`.
- **Strict as-of late flag:**
  - Question: will the line arrive later than its contract, judged with only what was recorded before the week of
    placement?
  - Positive class: Y > contract (`phase21_score.uc1p_label`; a censored line is kept and counted late if its elapsed
    time already exceeds the contract).
  - **Base rate 0.35** (0.349, Phase 22 Stage 2).
  - Majority-class baseline: "no line is late". Accuracy 0.651, recall 0.
  - Arm: `flag_lag1`, per seed.
  - Spread: per month of τ, the Monday on or before creation.
- **INCUMBENT:** none. Phase 15 had no order-time product.
- **BEST PUBLISHED:** Phase 21's at-placement flag, ALERT at 0.79 (QUOTED, LEAKY: deviation 210).

### UC2: fill "arrives in full"

- **Question:** will this line be delivered in full (fill = 1)?
- **Positive class:** fill ≥ 1.
- **Score:** P(fill = 1), cell 21 of the 22-cell distribution.
- **Base rate:** **0.749**.
- **Majority-class baseline:** "every line arrives in full". Accuracy 0.749, precision 0.749, recall 1.0.
- **Arms:**
  - INCUMBENT: boundary bw3, per seed.
  - BEST PUBLISHED: the Phase 19 fill blend, 0.50 × neural season + cadence + 0.50 × LightGBM fwd_load.
  - CLEAN: the consolidated blend, w = 0.02 × the clean neural season + cadence ensemble + 0.98 × the LightGBM
    season + cadence + ack + L4 ensemble.

### UC2b: fill "materially short"

- **Question:** will this line be delivered materially short (fill < 0.95)?
- **Positive class:** fill < 0.95.
- **Score:** the probability mass below 0.95 (cells 0–19).
- **Base rate:** **0.245**.
- **Majority-class baseline:** "no line is short". Accuracy 0.755, recall 0, precision undefined.
- **Arms:**
  - INCUMBENT: `lgbm22_id`, per seed.
  - BEST PUBLISHED: the Phase 19 fill blend.
  - CLEAN: the clean Phase 19 fill blend, w = 0.49 × clean neural season + cadence + 0.51 × clean LightGBM fwd_load.

### Fill distribution

- **Exact CRPS:** of the 22-cell distribution (`phase5_metrics.crps_exact_rows`); lower is better.
- **P(fill = 1) AUC.**
- **ECE-22 RAW and RECALIBRATED, reported apart:** the marginal ECE, Phase 18's definition.
  - RECALIBRATED only where a stored recalibration exists: a bundle's `recalibration.json`, or the stored `RECAL_`
    predictions.
  - Elsewhere it is **n/a**. Nothing is fitted here.
- **Arms:**
  - INCUMBENT: the neural none_h0 bundle, per seed.
  - BEST PUBLISHED: the Phase 19 fill blend.
  - CLEAN: the consolidated blend.

### UC3: capacity "strain > 1 in the next 90 days"

- **Question:** will the supplier's ordered quantity exceed its capacity over the next 90 days?
- **Positive class:** strain > 1.
- **Score:** P(strain > 1), from the P10 / P50 / P90 (`phase14_score.p_exceed`).
- **Base rate:** **0.405**.
- **Majority-class baseline:** "no supplier is strained". Accuracy 0.595, recall 0.
- **Pinball loss:** at q = 0.1, 0.5 and 0.9, and their mean.
- **Interval coverage:**
  - The raw P10–P90 interval is recomputed, per quarter (min / mean / max).
  - The Phase 18 level-aware conformal interval is QUOTED per quarter from `capacity_clean.json` and
    `capacity_published.json` (`conformal_level_aware`).
- **Arms:**
  - INCUMBENT: mp h4, per seed.
  - BEST PUBLISHED: the incumbent 5-seed ensemble (mean quantiles).
  - CLEAN: the clean mp h4, per seed.

### Shortage head (retired)

- **Status:** QUOTED.
- **Figure:** re-weighted to a 3% base, precision in the top 5% is **0.224** (`reports/part2/phase-17.md` §1.1).
- **Positive class:** a part-plant short. At a 3% base, the majority baseline is "never short".
- **No clean re-measure.**

### Predict-the-rescue and the shortage simulation

- **Status:** **PENDING PHASE 23B.**
- **Rescue:** the Phase 17 figure is shown labelled **LEAKY-UNCONFIRMED**: precision 0.823 at recall 0.246, base
  0.4535 (`phase-17.md` §2.1).
- **Simulation:** 1.129× against the pre-rescue reference (`phase-13.md`); precision ceiling 0.28 (`phase-15.md`), also
  LEAKY-UNCONFIRMED.

### MILP delivery schedule, supplier allocation, transfer recommendation

- **Status:** **no model metric: blocked.** Each line quotes `results/observation2.md` §3.6:
  - **MILP:** needs holding, ordering, freight and shortage costs, and 0 of 4 exist.
  - **Allocation:** 57 of 120 part-plants have no allowed split; on the 63 feasible ones it ties the status quo.
  - **Transfer:** its confidence is NOT TUNABLE (lift 0.03), and "always yes" beats it on F1.

## Sanity checks the pack prints (each has a way to fail)

1. **Recomputed vs quoted:** every recomputed figure that has a published counterpart is compared at the published
   precision. A mismatch is listed as a FINDING and never smoothed.
2. **Accuracy and the majority baseline:** a line that prints accuracy without the majority baseline makes the writer
   raise. The pack scans its own markdown for this.
3. **Always-majority predictor:** a constructed always-majority predictor must show accuracy equal to the baseline. At
   a base above 0.5, its precision equals the base rate and its lift is 1. At a base below 0.5, its recall is 0.
4. **AUC by two methods:** ROC-AUC and PR-AUC agree between the two methods to 1e-6 on every computed AUC. A
   constructed tie-heavy case is checked against a hand value: y = [0, 0, 1, 1], s = [0.5, 0.5, 0.5, 0.9] gives
   ROC-AUC 0.75 and PR-AUC 0.75.
