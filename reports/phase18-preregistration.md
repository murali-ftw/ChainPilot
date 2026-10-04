# Phase 18 — pre-registration

**Committed before any Phase 18 measurement.** Nothing below is edited after a result is known. Each prediction is
marked right or wrong in `reports/phase18.md`; wrong ones are not dropped.

**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, val 2024, test 2025).
**Branch / base:** `phase18`, from `origin/phase17` at `a2301ef` (Phase 17 code + report; not yet merged to main).
**Machine:** Apple M4 Pro (14 cores, 24 GB), macOS, torch 2.14.0 (MPS), LightGBM 4.7.0. This is the machine that
trained every stored v8 baseline bundle.
**Wall-clock stop:** the brief leaves it blank (`<FILL IN>`). Its own example is used: **12 h from start, i.e.
06:39 IST 2026-10-04** (start 18:39 IST 2026-10-03). Deviations are numbered from 160.

## Predictions

| # | Prediction |
|---|---|
| **P1** | Arrival: the oracle beats the model on lateness ROC-AUC by ≥ 0.03 (headroom exists) |
| **P2** | Fill: the oracle is within 0.03 AUC of the model's P(fill = 1) AUC (fill is near its ceiling) |
| **P3** | Capacity: the oracle beats the model on precision at 5% coverage by ≥ 0.05 |
| **P4** | Forward load gives a disjoint LightGBM-proxy gain on capacity |
| **P5** | Forward load does **not** give a disjoint gain on fill |
| **P6** | Seed-ensembling improves exact CRPS / ECE (fill) and C-index (arrival), and the validation-fit neural + LightGBM blend ties or beats both parents |
| **P7** | The global pulse feature ties |
| **P8** | Split-conformal brings arrival P10–P90 test coverage into 0.78–0.82 (from 0.93) |

## Decision rules (from the brief, unchanged)

- **Oracle headroom** = oracle metric − model metric. < 0.02: **AT CEILING**, stop spending on that use case.
  0.02–0.05: **SMALL**. > 0.05: **REAL**.
- A feature family **PASSES** the proxy gate if the LightGBM proxy (5 seeds) beats the baseline LightGBM on the use
  case's primary metric with **disjoint 5-seed bands** AND the **shuffle control erases the gain**. Otherwise it
  **FAILS**, and no neural run is made for it.
- **Primary metrics:** arrival = lateness ROC-AUC (`docs/specs/lateness_metric.md`, as-of channel reference) and A3
  median absolute error (days, uncensored rows); fill = exact CRPS and P(fill = 1) ROC-AUC; capacity = Phase 15 UC3
  precision at 1 / 5 / 10% coverage and recall at p = 0.70 / 0.80 / 0.85 (operating points chosen on validation per seed).
- A margin inside the 5-seed band is **UNDETERMINED**, not a difference. Selection and thresholds on validation only.
  RAW and RECALIBRATED are never mixed in one comparison.

## Operational definitions fixed here (before any number)

These fill gaps the brief leaves open. Each is fixed now so it cannot be chosen after seeing a result.

1. **"The model"** per use case = the stored neural incumbent, 5 seeds (7/17/27/37/47), RAW:
   arrival `arrival_week/v8_lite_h4_lr0.00025_s{s}` (SHARE-lite h⁴); fill `fill_rate/v8_none_h0_lr0.000125_s{s}`
   (22-cell head, h⁰); capacity `capacity_strain/v8_mp_h4_lr0.00025_s{s}` (mp h⁴).
   **"LightGBM-flat"** = the stored guide-B5 arms `b5flat_reg_s{s}` / `b5flat22_s{s}` / `b5flat_q_s{s}` (`phase7_fit.py`, frozen GBM).
2. **Headroom metric** (one per use case, as P1–P3 name them): arrival lateness ROC-AUC; fill P(fill = 1) ROC-AUC;
   capacity precision at 5% coverage. The other primary metrics are reported alongside, without a verdict.
3. **Oracle** = the frozen GBM (`phase7_fit.GBM`) with the same objective as the use case's LightGBM-flat arm
   (arrival L2 on observed rows; fill 22-class; capacity quantile 0.1/0.5/0.9), on the flat as-of features **plus**
   privileged latents. Two tiers, both reported; **the brief's tier decides the verdict**:
   - **ORACLE** (brief): arrival and fill add the line's creation week, supplier state, plant transit state, regime and
     **the supplier load at order creation** (`util` of the month before creation, `K`, the supplier's ordered volume in
     the creation week and month); capacity adds **true K and the realised ordered volume** over the label window.
   - **ORACLE-STATE** (diagnostic): the same, minus every realised-ordering quantity (no load, no ordered volume). It
     separates "headroom from state" from "headroom from knowing the future order book".
   The line's own realised shortfall, lead time or receipt are **never** oracle inputs: they are the outcomes.
4. **Proxy gate arms** (5 seeds each, frozen GBM, same rows): BASE (must reproduce the stored LightGBM-flat
   predictions before it is used), BASE + family, BASE + shuffled family. "Erases the gain" = the shuffled arm is
   **not** disjointly better than BASE on the metric that passed.
   - A family passes on a use case if **any** of that use case's primary metrics passes and none is disjointly worse.
   - Capacity's primary metric list has six points; a pass needs ≥ 3 of the 6 disjointly better and none disjointly worse.
5. **Ensembles, blends and conformal intervals** are single deterministic predictors, not seed bands. They are compared
   by a **paired row bootstrap** (1,000 resamples of test rows; 95% interval of the difference). "Improves" means the
   interval excludes 0 in the favourable direction. Ensemble vs single seeds is read as ensemble minus the **mean of
   the five single-seed metrics**, paired on the same resample; never as ensemble vs the single-seed band.
6. **Blend** (Stage 3b): one weight w ∈ [0, 1] on a 101-point grid, chosen on validation by the use case's proper score
   (arrival: lateness ROC-AUC on the shared expected-arrival scale; fill: exact CRPS; capacity: mean pinball loss),
   frozen, applied to test. "Ties or beats both" = not worse than either parent by the bootstrap rule.
7. **Arrival split-conformal** (Stage 3c): residuals in signed days on validation, uncensored rows; the interval is
   the raw P50 widened or narrowed to the 10th/90th residual quantiles. Coverage is read on test uncensored rows,
   the same rows as Phase 17's 93%.
8. **Stage 6** runs only if Stage 1 (c) is non-empty **and** the lanes table carries a grouping with more than one
   channel per group. If every lane holds one channel, "other channels on the same lane" is empty by construction and
   the stage is BLOCKED.
