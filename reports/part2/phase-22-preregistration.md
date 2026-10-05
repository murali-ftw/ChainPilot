# Phase 22 — pre-registration

**Committed before any Phase 22 measurement.** Nothing below is edited after a result is known. Each prediction is
marked right or wrong in `reports/part2/phase-22.md`; wrong ones are kept.

**Measured on:** v8 seed 1001 (primary); v8w1002 (`data_worlds/v8_seed1002/seed_1002`, Phase 20, not regenerated; its
recorded hashes re-verified) for replication. Fixed split: train ≤ 2023, val 2024, test 2025. Seeds 7 / 17 / 27 / 37 / 47.
**Branch / base:** `phase22`, cut from local `HADES-v4-ml-pipeline` at `cbc7088`, in a new worktree
(`../HADES_v4_phase22`). Not merged, not pushed. The user's working tree is not touched.
**Machine:** Apple M4 Pro (14 cores, 24 GB), macOS; torch 2.14.0 (MPS) for Stages 1e and 3b only, **concurrency 1**,
incumbent architectures and learning rates; LightGBM 4.7.0 on CPU.
**Wall-clock stop:** 14 h from the start at **00:53:40 IST 2026-10-05**, i.e. **14:53:40 IST 2026-10-05**. A stage not
finished by then is listed in `reports/part2/phase22/STOPPED.json` (completed, not_run, reason) and never reported as
complete.
**Deviations** are numbered from **209**.

## Predictions (from the brief, verbatim)

| # | Prediction |
|---|---|
| **P1** | Removing the leak lowers the stored neural arrival ensemble's lateness AUC by 0.004-0.015 (LightGBM lost 0.0066). A3 worsens by 0.1-0.5 d. |
| **P2** | Fill and capacity models are not materially affected (<0.003 AUC / <0.01 precision): the audit finds the three columns do not feed them, or feed them weakly. (Audit decides; if they do feed them, the retrain decides.) |
| **P3** | No Phase 15 class changes after the leak is removed for UC1, UC2, UC2b, UC3; the order-time arrival flag stays ALERT. |
| **P4** | A systematic leak scan finds at least one further column or table with an as-of violation beyond the three known. |
| **P5** | The order-time product's best date estimate is a shrunk-KM-median + LightGBM(L4) blend with A3 <= 10.0 d (block-disjoint better than LightGBM-flat 12.6 d) and its late flag has lateness AUC >= 0.69. |
| **P6** | The order-time 80% interval (split-conformal on validation, expected-week) covers 0.77-0.83 on test overall but undercovers by > 5 points in at least one creation month. |
| **P7** | Consolidated fill (neural + LightGBM, season + cadence + ack-gap + L4 channel statistics, leak-free) beats the Phase 19 blend on exact CRPS (block-disjoint) and the arrives-in-full list, and stays WATCHLIST. The materially-short (UC2b) list does NOT beat the Phase 19 blend's 0.530 precision at 5%. |
| **P8** | Rolling recalibration raises the worst-quarter capacity precision by > 0.05 but lowers the average, and does not rescue December 2025 if the diagnosis shows a base-rate / regime shift; if the diagnosis shows threshold drift only, it does. |
| **P9** | The capacity alert stays ALERT with a documented worst-quarter floor. |

How each is scored, fixed now (every comparison clean-vs-clean or published-vs-published, never mixed):
- **P1** right iff (clean neural arrival 5-seed ensemble − published ensemble) lateness AUC ∈ [−0.015, −0.004] **and**
  A3 worsens by 0.1–0.5 d (block point estimates; intervals reported).
- **P2** right iff, for fill (P(fill = 1) AUC) and capacity (UC3 precision at 5%), |clean − published| neural 5-seed
  ensemble change is < 0.003 AUC and < 0.01 precision respectively. If the audit finds fill / capacity read no
  leaking column, P2 is right by audit and the retrain is skipped.
- **P3** right iff the Phase 15 class (Phase 20 mapping) of UC1, UC2, UC2b, UC3 on the clean incumbent neural arm
  (per seed, as Phase 15 / 20 classify it) equals the published class, **and** Stage 2's clean UC1-P late flag is ALERT.
- **P4** right iff the Stage 1b scan flags at least one column or table beyond `lead_time_actual_days`,
  `lead_time_ratio`, `otd_rate_last13`.
- **P5** right iff the arm with the lowest **validation** A3 among the Stage 2 date estimators is the KM + LightGBM(L4)
  blend, its test A3 ≤ 10.0 d, its block interval vs LightGBM-flat (BASE_clean) excludes 0 in its favour, **and** the
  Stage 2 late flag (best by validation) has test lateness AUC ≥ 0.69.
- **P6** right iff overall test coverage ∈ [0.77, 0.83] **and** some creation month's coverage < (overall − 0.05).
- **P7** right iff the consolidated blend's CRPS block interval vs the **clean-restated** Phase 19 fill blend excludes 0
  in its favour, its UC2 precision at 5% is higher (point; block interval reported), its class is WATCHLIST, **and** its
  UC2b precision at 5% is < 0.530. The published Phase 19 blend is reported beside it, never compared with it.
- **P8** right iff the adopted-or-best recalibration scheme (by validation) raises test worst-quarter precision by
  > 0.05 and lowers test mean precision, **and** its December 2025 outcome matches the diagnosis as the prediction states
  (regime shift → not rescued, Dec 2025 precision stays < 0.70; threshold drift → rescued, ≥ 0.70).
- **P9** right iff the clean capacity incumbent is ALERT under the Phase 15 rule (Phase 20 mapping), with the
  worst-quarter precision at 5% reported as its floor.

## Rules (from the brief, verbatim)

- Leak delta = clean metric minus published metric, 5-seed band and block interval; REPORT as "published (leaky)" vs
  "clean" in separate columns, never mixed.
- Feature family PASSES a use case if BASE_clean+family beats BASE_clean with disjoint 5-seed bands on a primary metric,
  none disjointly worse, AND its control (below) is not disjointly better than BASE_clean.
- Primary metrics: arrival = lateness ROC-AUC, A3 median abs error, UC1 precision at 1/5/10/20% coverage; order-time =
  A3, expected-week error, UC1-P precision at 1/5/10/20%, interval coverage by creation month; fill = exact CRPS,
  P(fill=1) AUC, UC2b materially-short precision at 5%; capacity = Phase 15 UC3 precision at 1/5/10% coverage and recall
  at p = 0.70/0.80/0.85, plus per-quarter precision (min, mean) and share flagged.
- Capacity stability rule: a recalibration scheme is ADOPTED only if the worst-quarter precision on VALIDATION
  walk-forward improves with a disjoint band, mean precision drops by less than 0.02, and the share flagged stays within
  5-15%.

## Phase 15 classification rule

Copied verbatim from `reports/part2/phase-15.md` (its reading rules) and from `ml/eval/phase15.py`, where the rule itself is
written (Phase 20 deviation 178):

> - **Recall at p** is the TEST recall at the threshold chosen on validation to reach precision p. The TEST precision
>   actually achieved is in brackets. **✓** means the bar held on test for **every** seed with ≥ 50 alerts.
> - **Max precision** is the highest test precision at any coverage with ≥ 50 alerts. It is descriptive, not an
>   operating point.
> - **The base rate is on every row.** "REACHABLE" follows the brief's definition literally, and lift is added beside
>   it, because the definition ignores base rate (deviation 134).

```
STAGE B VERDICT RULES, fixed here before any curve was computed (committed with this file):
  precision is read on a 40-point geometric coverage grid 0.1%..100%, cells with < 50 alerts dropped.
  rho  = Spearman(coverage, precision)   (tunable scores have rho strongly NEGATIVE: tighter coverage, higher precision)
  lift = max precision - base rate;  drop = peak precision - precision at the tightest usable coverage
  TUNABLE        rho <= -0.7 AND drop <= 0.05 AND lift >= 0.10
  NOT TUNABLE    lift < 0.05 (flat)  OR  rho >= 0 (inverted)  OR  drop > 0.10 (peaks early, then falls)
  WEAKLY TUNABLE anything else
  Computed on VALIDATION per seed (the verdict is the majority over seeds) and on TEST (confirmation only).
REACHABLE (Stage G), as the brief defines it: YES if a validation-chosen point gives precision >= 0.85 with recall >= 0.10
and >= 50 alerts on TEST for EVERY seed; PARTIAL if 0.70-0.85; NO, TUNING if Stage B says NOT TUNABLE; NO, CEILING otherwise.
```

Class mapping (Phase 20 pre-registration rule 1, `phase20_decisions.reachable_and_class`, imported unchanged):
**ALERT** = REACHABLE YES / PARTIAL and test lift ≥ 1.5 at the highest bar whose validation-chosen point holds on test
with ≥ 50 alerts on every seed; **WATCHLIST** = not ALERT and Stage B TUNABLE / WEAKLY TUNABLE; **RETIRED** = NOT TUNABLE.
Published classes: UC1 WATCHLIST, UC2 WATCHLIST, UC2b WATCHLIST, UC3 ALERT; UC1-P (Phase 21, no-leak) ALERT.

## Definitions, fixed before any number is read

### D1. Leak horizons and the scan (Stage 1b)

A stored panel row t (`channel_performance_weekly`, week starting `W[t]`) is what the stored models read for a snapshot
whose date is `W[t]` (`phase7_fit.World.asof`, `temporal_share`). Two horizons:
- **H_week = W[t] + 7 days** (the end of the row's own week). The generator's labels start the week after t0 (`pt > t0`
  in weeks), so this is the information horizon the generator assumes. **A column is LEAKING if poisoning source rows
  visible after H_week changes its value at row t.** This is the scan's verdict.
- **H_date = W[t]** (the snapshot date itself, the strict `recorded_ts ≤ t0` reading). The share of rows that change
  under poisoning after H_date is reported per column as the **week-t0 convention** diagnostic, with one LightGBM
  diagnostic arm (BASE_clean read at row t − 1), no verdict.

**Method.** A new module rebuilds each panel column from the emitted source tables (`po_lines`, `grn_lines`,
`supplier_acknowledgements`, `po_line_revisions`, `supplier_capacity`), following the generator's own derivation
(`generator_v8.py`, weekly stores). A column is scanned by poisoning only if its rebuild reproduces the stored values
(≥ 99% of non-null rows within the stored rounding). A column that cannot be rebuilt is scanned by code audit, and that
is stated. Every derived feature store a stored model reads (fwd_load, fwd_season, cadence, pulse, fwd_pred, stock_asof,
grpstats, fill_history) is scanned by running its own as-of assertion / falsifier, or by poisoning where it recomputes
from source. Plan tables, ASN and schedules are listed with the models that read them.

**Constructed failing case (the scan must fire):** the three known columns. If any of them is not flagged, the scan is
invalid and Stage 1 stops. **Constructed passing case:** `qty_ordered` (bucketed on the order's visible week) must not
change under H_week poisoning.

### D2. Clean inputs (Stage 1c) and the clean world

- **Leaking columns** = those the 1b scan flags.
- **As-of-safe replacement**, keyed to when the outcome is **recorded**:
  - Lead, lead ratio, on-time rate: bucketed on the first receipt's visible week, forward-filled; `otd_rate_last13` is
    the 13-week rolling mean of the forward-filled on-time flag, as the generator computes it.
  - Fill and its rolls, ack-gap: line fill bucketed on the final receipt's visible week, or the zero signal's week for
    lines closed at zero (Phase 13 closed rule).
  - Load ratio: the supplier's ordered quantity **visible so far in the month** ÷ the declared capacity recorded ≤ the row's week.
  - Any other flagged column with no implementable replacement is **masked** (value 0, missing-indicator 0).
- **Clean world `v8clean`** (and `v8w1002clean`): the same CSVs; a new cache directory whose panel replaces the flagged
  columns. The world name enters every artifact identity, so every clean artifact has a new name and path. No stored
  file is overwritten.
- The replacement module carries its own future-poison and self-exclusion tests with constructed offenders (the stored
  leaky columns must be flagged).

### D3. Restatement arms (Stage 1d–f)

- **published** = the stored predictions.
- **nl** = LightGBM with the flagged columns removed.
- **clean** = LightGBM or neural on `v8clean`.
- LightGBM arms: Phase 7 B5 fit, frozen config; arrival / fill / capacity snapshot rows.
- Neural arms: arrival `lite h4 lr 0.00025`, fill `none h0 lr 0.000125`, capacity `mp h4 lr 0.00025`; `loop.train`
  unchanged; bundle root `ml/artifacts/phase22/bundles`; run only for a task whose model reads a flagged column (the
  panel feeds all three heads, so expected: all three).
- **Clean Phase 19 blends.**
  - Arrival: the Phase 19 recipe re-fitted on validation with clean arms. Clean neural incumbent ensemble (the Phase 19
    neural rf arm TIED the incumbent and is not retrained) + clean LightGBM + fwd_load ensemble.
  - Fill: the clean Phase 19 neural season + cadence arm (retrained, 5 seeds) + clean LightGBM + fwd_load, weight on
    validation CRPS.
  - The Phase 21 hybrid weights are re-fitted the same way.

### D4. Stage −1 handshake

Retrain stored `arrival_week v8_lite_h4_lr0.00025_s7` for 3 epochs; inside the stored 5-seed band at every epoch → proceed.
**Constructed failing case:** the same comparison against a stored value shifted by 0.05 must report OUTSIDE.

### D5. Gates and their constructed failing cases

| gate | rule | constructed failing case run by the code |
|---|---|---|
| family gate (Stages 2, 3) | the brief's PASS rule, disjoint 5-seed bands | (i) an arm whose bands equal BASE → FAIL; (ii) an arm better than BASE whose control is also better → FAIL |
| neural GAIN / TIE / WORSE | disjoint vs the clean incumbent band | the clean incumbent against itself → TIE |
| leak scan | D1 | the three known columns must be flagged |
| Stage 2 leaked arm | the leak scan's feature check over every Stage 2 arm's feature list | an arm re-adding the three columns must be flagged |
| serve guard | identity match | a mismatched bundle must raise |
| capacity purge | every snapshot in a recalibration window has t0 + 90 d ≤ the scoring t0 | a scheme fed the test period's own labels must raise |
| stability rule | the brief's rule | a scheme identical to the incumbent cannot be ADOPTED (no disjoint improvement) |

Every control is checked for what it preserves (printed beside it):
- A cross-channel shuffle keeps the snapshot-level season.
- A snapshot permutation keeps the channel identity.

### D6. Stage 2 (order time)

- **Rows and labels:** Phase 21's at-placement rows (218,881 lines; τ = Monday on or before creation), lead labels and
  UC1-P (late vs contract, censoring resolved).
- **BASE_clean:** flat as-of channel features from `v8clean` at τ + log1p(qty).
- **Arms:** BASE_clean; BASE_clean + L4 (grpstats, k = 10 as chosen in Phase 21 on validation); standalone shrunk-KM
  median; LightGBM regression of lateness-in-days (lead − contract) on BASE_clean + L4; the blend of KM date and
  LightGBM date (w on a 0.05 grid by validation A3); LightGBM late flag (binary, BASE_clean + L4); isotonic calibration
  of the flag fitted on validation.
- **Interval:** split-conformal on validation signed-day residuals of the chosen date estimator, central 80%.
  Month-specific quantiles are used only if validation per-month coverage of the pooled interval spreads by > 5 points.
- **Controls:** the cross-channel shuffle of the L4 block.
- **Persistence:** the chosen estimator, flag and calibration saved as loadable bundles with identity.

### D7. Stage 3 (fill)

- **Snapshot rows** as Phases 19–21; BASE_clean on `v8clean`.
- **Families and controls:**

  | family | control |
  |---|---|
  | season + cadence | season permuted across snapshots within split (Phase 19 derangement) |
  | ack-gap | permuted across snapshots within split **and** cross-channel shuffled |
  | L4 (k = 100, Phase 21) | cross-channel shuffle |

- **Arms:** consolidated = season + cadence + ack-gap + L4.
- **Neural:** fill h0 bound through `n_row_feats` (Phase 19 binding, new subclass) with the families that PASSED.
- **Blend:** consolidated neural + LightGBM, weight by validation CRPS.

### D8. Stage 4 (capacity)

- **Quarters:** calendar quarters of the snapshot date.
- **Causes, each with its evidence:**
  - (i) regime: the base rate shifts, beyond the validation range.
  - (ii) threshold drift: within-quarter AUC holds, but precision at the validation threshold falls.
  - (iii) ranking: within-quarter AUC falls.
- **Schemes:**
  - the fixed validation threshold;
  - rolling Platt;
  - rolling isotonic;
  - a rolling threshold holding the validation precision target;
  - a flag-rate-targeted threshold.
- **Windows:** in resolved snapshots, from {2, 4, 6, 8}, chosen by validation walk-forward over 2024 (maximise worst
  quarter, tie → mean). Labels resolve 90 days after t0; a snapshot enters a window only if its t0 + 90 d ≤ the scoring t0.
- **Bands:** across the five stored seeds.

### D9. Intervals

Seed ensembles, blends, rules and recalibrated schemes are single predictors. They use the snapshot-block bootstrap
(1,000 resamples; at placement whole creation weeks). No row bootstrap. RAW and RECALIBRATED are never in one comparison.
