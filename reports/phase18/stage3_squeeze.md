# Phase 18 Stage 3 — no-training squeezes

**Audience:** whoever decides what can ship from the existing models without training anything.
**Measured on:** v8 seed 1001, fixed split, TEST; stored predictions only (neural incumbents `arrival lite h⁴`,
`fill h⁰ 22-cell`, `capacity mp h⁴`; LightGBM-flat `b5flat_reg / b5flat22 / b5flat_q`), seeds 7/17/27/37/47.
**Code:** `ml/eval/phase18_squeeze.py` (`61e92ef`) → `ml/artifacts/phase18/stage3_squeeze.json`.
**Status:** complete. **RAW** (3a, 3b) and **RECALIBRATED** (3c, 3d) are separate tables.

An ensemble, a blend or a conformal interval is one deterministic predictor, not a seed band. It is compared by a
**paired row bootstrap** (1,000 test-row resamples, 95% interval of the difference; pre-registered def. 5). Rows in one
snapshot share that snapshot's shocks, so these intervals are **optimistic** (deviation 167). Ensemble vs single seeds
is read against the mean of the five single-seed metrics on the same resample, never against the single-seed band.

## 3a / 3b — seed-ensemble and blend (RAW)

Blend weight on the neural ensemble, chosen on validation (pre-registered criterion): **arrival 0.59, fill 0.97,
capacity 1.00.**

| use case · metric | neural single (5-seed mean) | **neural ensemble** | LightGBM ensemble | **blend** |
|---|---|---|---|---|
| arrival · lateness ROC-AUC | 0.7090 | 0.7138 | 0.7056 | **0.7247** |
| arrival · A3 median abs err (days) | 13.06 | 13.07 | 13.23 | **12.50** |
| arrival · C-index | 0.6744 | **0.6773** | 0.6493 | 0.6764 |
| fill · exact CRPS | 0.13876 | **0.13853** | 0.13961 | **0.13852** |
| fill · P(fill = 1) ROC-AUC | 0.6204 | 0.6240 | 0.6058 | **0.6244** |
| fill · ECE-22 | 0.1235 | 0.0906 | **0.0518** | 0.0889 |
| capacity · precision @ 1 / 5 / 10% | 0.904 / 0.851 / 0.810 | 0.907 / 0.864 / 0.816 | 0.796 / 0.692 / 0.663 | 0.907 / 0.865 / 0.815 |
| capacity · recall @ p = 0.70 / 0.80 / 0.85 | 0.515 / 0.318 / 0.192 | 0.547 / 0.331 / 0.230 | 0.155 / 0.018 / 0.012 | 0.546 / 0.331 / 0.231 |

Paired-bootstrap verdicts (difference, better = positive, 95% interval):

| comparison | arrival | fill | capacity |
|---|---|---|---|
| **neural ensemble vs mean single neural** | lateness **better** (+0.0047 [0.0045, 0.0050]); C-index **better** (+0.0028 [0.0006, 0.0050]); A3 undetermined | CRPS **better**; AUC **better** (+0.0036); ECE **better** (−0.033) | precision @ 5% **better** (+0.014); recall @ 0.70 / 0.80 / 0.85 **better** (+0.032 / +0.013 / +0.039); precision @ 1%, 10% undetermined |
| LightGBM ensemble vs mean single LightGBM | lateness better (+0.0002); C-index **undetermined**; A3 undetermined | CRPS better; AUC better; ECE **undetermined** | recall @ 0.70 better; @ 0.80 **worse** (−0.0035); rest undetermined |
| neural ensemble vs LightGBM ensemble | better on lateness and C-index; A3 undetermined | better CRPS and AUC; **worse ECE** (−0.038) | better on all six |
| **blend vs neural ensemble** | lateness **better** (+0.011 [0.008, 0.014]); A3 **better** (−0.59 d [0.46, 0.72]); C-index undetermined | CRPS, AUC, ECE better (all < 0.002) | recall @ 0.70 **worse** (−0.001 [−0.002, −0.0001]); @ 0.85 better (+0.001); rest undetermined |
| **blend vs LightGBM ensemble** | better on all three | CRPS, AUC better; **ECE worse** (−0.036) | better on all six |

- **Arrival is the one use case where blending pays.** The validation-chosen 0.59 / 0.41 blend lifts lateness ROC-AUC
  from 0.709 (single neural) to **0.725**, and cuts A3 median error from 13.06 to **12.50 days**. Both are beyond the
  bootstrap interval against either parent. The two families rank lines differently enough to average.
- **Fill:** the neural seed-ensemble is better on all three metrics. The blend is 97% neural and adds nothing material.
  LightGBM keeps its calibration advantage (ECE 0.052 vs 0.089), the reason it was shipped (Phase 10).
- **Capacity:** validation puts the whole weight on the neural model (w = 1.00), so the blend is the neural quantile
  mean. It differs from the ensemble row only because the ensemble averages P(strain > 1) across seeds, while the blend
  reads it from averaged quantiles. The −0.001 recall difference at p = 0.70 is that construction, not a finding.

## 3c — arrival interval: split-conformal (RECALIBRATED)

Residuals 7·(Y − centre) in signed days on validation uncensored rows; interval [centre + q₁₀, centre + q₉₀]
(conservative ends: lower / higher order statistic). Coverage on **test uncensored rows**, the rows behind Phase 17's
93%. Nominal 80%.

| interval | test coverage (5 seeds) | width (days) |
|---|---|---|
| raw P10–P90 (reproduces Phase 17) | **0.930** [0.921, 0.943] | 51.8 [51.1, 53.0] mean |
| **split-conformal on the P50 (pre-registered)** | **0.862** [0.854, 0.870] | **49.0** (every seed: −35 / +14 d) |
| split-conformal on the expected week (diagnostic) | **0.801** [0.791, 0.806] | 45.1 [44.8, 45.4] |
| neural ensemble, P50 / expected week | 0.868 / 0.801 | 49.0 / 44.7 |

- **The pre-registered interval misses 0.78–0.82 (P8 wrong).** The P50 is a whole week and so is the label, so every
  residual is a multiple of 7 days. The 10% / 90% residual quantiles land on −35 / +14 days exactly, and the
  next-narrower choice drops below 80%. Coverage cannot be tuned finer than one week's step (deviation 166).
- **Centring on the continuous expected week fixes it:** 0.80 coverage at 45 days, 7 days narrower than the raw
  P10–P90's 52. This is a diagnostic added after the pre-registered row was read, so it is not scored against P8. It is
  the variant to adopt.

## 3d — capacity interval: level-aware conformal keyed to recent drift (RECALIBRATED)

Score = max(q₁₀ − y, y − q₉₀) / q₅₀ (level-aware). For each test snapshot t0 the offset c is the 80% conformal quantile
over the **last 4 snapshots whose 90-day labels had ended by t0** (asserted). Interval [q₁₀ − c·q₅₀, q₉₀ + c·q₅₀].
"Static" = one offset from the whole validation fold, the fixed-factor reference. Coverage per test snapshot (9),
seed-mean; nominal 80%.

| arm | interval | min | q25 | **median** | q75 | max | pooled (5-seed band) | median width |
|---|---|---|---|---|---|---|---|---|
| neural mp h⁴ | raw P10–P90 | 0.729 | 0.752 | 0.763 | 0.821 | 0.825 | 0.782 [0.776, 0.788] | 0.659 |
| | static conformal | 0.754 | 0.775 | 0.784 | 0.844 | 0.848 | 0.804 [0.791, 0.816] | 0.690 |
| | **trailing conformal** | 0.740 | 0.762 | 0.774 | 0.834 | 0.844 | 0.793 [0.787, 0.796] | 0.671 |
| LightGBM b5flat_q | raw | 0.724 | 0.767 | 0.818 | 0.832 | 0.853 | 0.803 [0.800, 0.805] | 0.811 |
| | static conformal | 0.728 | 0.773 | 0.823 | 0.837 | 0.859 | 0.809 [0.808, 0.810] | 0.822 |
| | **trailing conformal** | **0.696** | 0.771 | 0.790 | 0.844 | 0.864 | 0.795 [0.794, 0.796] | 0.771 |

Per snapshot, neural trailing: 0.763 / 0.771 / 0.815 / 0.774 / 0.838 / 0.757 / 0.844 / 0.834 / 0.740 (Jan → Dec 2025).

- **The raw neural interval is already close to nominal** (pooled 0.78), with a 0.73–0.83 spread across snapshots.
- **Trailing calibration does not narrow the spread.** Its per-snapshot range (0.74–0.84) is no tighter than the
  static offset's (0.75–0.85). For LightGBM it widens it (minimum 0.70). A trailing window of labels that ended 13+
  weeks earlier reacts to drift that has already passed: the label lag is as long as the shocks. **Keyed to recent
  drift, the conformal does not track drift on this data.** The static offset is as good and simpler.
