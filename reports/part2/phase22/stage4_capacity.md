# Phase 22 Stage 4 — capacity: December 2025 diagnosed, rolling recalibration tested

**Audience:** whoever owns the supplier-strain (capacity) product.
**Measured on:** v8 seed 1001. Test 2025 (9 snapshots, 4 quarters).
- **Clean:** the incumbent `mp h4 lr 0.00025` retrained on `v8clean` (Stage 1e) — the primary arm here.
- **Published:** the stored incumbent, reported apart.

Stored predictions only: nothing in this stage is retrained.
**Instrument:** `ml/eval/phase22_capacity.py` → `capacity_clean.json` (`5f9fc7d`), `capacity_published.json` (`f2e1f85`).
Score = P(strain > 1) from P10 / P50 / P90; operating bar p = 0.80; the fixed threshold is chosen on the whole validation fold.

## First: on clean inputs capacity is no longer an alert (Stage 1 restatement)

| | published (leaky) | clean |
|---|---|---|
| UC3 precision @ 1 / 5 / 10% (5-seed mean) | 0.904 / **0.851** / 0.810 | 0.828 / **0.735** / 0.679 |
| precision @ 5%, ensemble difference (block) | — | **−0.108**, clean worse |
| recall @ p 0.70 / 0.80 / 0.85 | 0.515 / 0.318 / 0.192 | 0.308 / 0.116 / UNREACHABLE (some seeds) |
| Phase 15 class | **ALERT** (PARTIAL at 0.85, lift 2.0) | **WATCHLIST** (NO, CEILING) |
| per-snapshot precision @ 5%, min / median / max | 0.42 / 0.78 / 0.91 | 0.45 / 0.64 / 0.89 |

The capacity label is the supplier's ordered quantity over true capacity across the next 90 days, starting with t0's own
month. `load_ratio` carried that month's eventual ordered quantity over declared capacity into every week of the month.
**Most of the alert was the label's own numerator, read early** (deviation 211).

## a. Diagnosis of December 2025

Per test snapshot (clean arm, 5-seed means; the fixed validation threshold):

| snapshot | base rate (strain > 1) | share flagged | precision | within-snapshot AUC | calibration slope | precision @ 5% |
|---|---|---|---|---|---|---|
| 2025-01-06 | 0.294 | 0.008 | 0.78 | 0.581 | 0.64 | 0.48 |
| 2025-03-31 | 0.394 | 0.123 | 0.70 | 0.714 | 1.24 | 0.77 |
| 2025-06-23 | 0.615 | 0.187 | 0.83 | 0.658 | 1.10 | 0.89 |
| 2025-08-04 | 0.686 | 0.041 | 0.84 | 0.585 | 0.78 | 0.75 |
| 2025-10-27 | 0.257 | 0.029 | 0.89 | 0.651 | 1.02 | 0.63 |
| **2025-12-08** | **0.199** | 0.004 | 0.63 | **0.601** | 0.62 | **0.45** |

Validation 2024 ranges: base rate 0.11–0.68; within-snapshot AUC 0.606–0.721 (clean; published 0.657–0.793).

| cause (pre-registered evidence) | December 2025, clean | published |
|---|---|---|
| (i) regime: base rate outside the validation range | **no** (0.199 inside 0.11–0.68) | no |
| (ii) threshold drift: AUC in range, precision at the fixed threshold below range | no | no |
| (iii) ranking degradation: within-snapshot AUC below the validation range | **yes** (0.601 < 0.606) | **yes** (0.637 < 0.657) |

Covariate drift at December 2025 vs train (published panel):
- standardised mean shifts: `load_ratio` −0.28, `fill_rate_last13` −0.29, `lead_time_ratio` +0.04, `qty_ordered` −0.05, `otd_rate_last13` −0.09;
- KS ≤ 0.14 on all five.

There is no gross drift. **December 2025 is a low-strain, seasonally quiet month in which the model ranks poorly.** It is
not a new regime and not a stale threshold. The same pattern appears in the low-base-rate months at the start of 2024 and
2025 (January 2025 AUC 0.581, clean).

## b–c. Rolling recalibration (stored scores; labels resolve 90 days after t0, purge asserted)

- **The purge assertion fires** on the constructed case: a window holding the scoring period's own unresolved labels.
- **The stability rule rejects** a scheme identical to the incumbent: it cannot be adopted.
- **Windows:** chosen by validation walk-forward over 2024. Stored predictions exist only for validation and test, so
  after the 90-day purge only 2–4 validation snapshots can be scored. Windows 6 and 8 are **infeasible**
  (deviation 213).

**Validation walk-forward (clean):** the fixed comparator and the rolling Platt / isotonic schemes **flag nothing** in
some seeds' scored quarters, so their precision is undefined. Only `rolling_threshold` (w2) and `flag_rate` (w2) are
defined. **Adoption: none.** Neither has a defined fixed comparator to beat, and the rule needs a disjoint improvement.
On the published arm, every scheme was REJECTED: no disjoint worst-quarter gain.

**Test (clean, 5-seed bands; window chosen on validation, or 4 where validation could not choose):**

| scheme | worst quarter | mean quarter | share flagged | December 2025 precision |
|---|---|---|---|---|
| fixed (incumbent) | 0.667 [0.556, 0.727] | 0.776 | 6.9% [1.0, 23.3] | 0.625 |
| rolling Platt (w4) | 0.718 | 0.843 | **0.7%** | 0.50 |
| rolling isotonic (w4) | 0.685 | 0.807 | 1.8% | no flag (some seeds) |
| rolling threshold (w2) | 0.619 | 0.767 | 0.8% | 0.84 |
| flag-rate target (w2) | 0.594 | 0.696 | 3.2% | 0.531 |

The schemes that raise precision do it by **flagging almost nothing** (0.7–1.8% of channel-horizons, below the 5–15%
band). None holds the flagged share inside 5–15% and improves the worst quarter. On the published arm the picture is the
same (rolling Platt: worst 0.886 at 1.1% flagged; fixed 0.642 at 17%). **No scheme is adopted.**

## d. Prediction intervals: Phase 18's level-aware trailing conformal (unchanged), coverage per quarter

| | 2025Q1 | Q2 | Q3 | Q4 | per-snapshot min / median / max |
|---|---|---|---|---|---|
| clean | 0.807 | 0.829 | 0.764 | 0.794 | 0.746 / 0.810 / 0.847 |
| published | 0.783 | 0.806 | 0.801 | 0.787 | 0.740 / 0.774 / 0.844 |

The intervals hold their nominal 80% in every quarter, within about ±5 points, on clean inputs too.

## e. What this means

- **The December 2025 drop is a ranking failure in quiet months, not a regime shift.** No recalibration fixes a model that
  ranks badly; it can only make the model say less.
- **The product answer is a monitor, not a fix.**
- Its **class is WATCHLIST on clean inputs**, so "the one alert in the project" is withdrawn pending a cleaner signal of
  capacity (the client data request's declared-capacity history).

**Proposed monitor (text only):**
- **What it tracks:** for each snapshot whose labels have resolved (t0 + 90 days), the precision of the flagged set and
  the base rate, on a trailing window of the last 4 resolved snapshots.
- **Trigger:** **raise a review** when the trailing precision falls below 0.60, or when the within-snapshot AUC of the
  latest resolved snapshot falls below the lowest validation value (0.61 clean).
- **Suppress, don't retune:** while the trigger is on, show the strain quantiles but not the alert flag.
- **On screen:** the gauge shows trailing precision, base rate and share flagged side by side, so a quiet month (low base
  rate) is not mistaken for a model failure, and vice versa.
