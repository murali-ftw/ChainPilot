# Phase 23AC T4.2: metric tables, INCUMBENT → BEST PUBLISHED → CLEAN

**Audience:** whoever presents ChainPilot's model figures. Read `definitions.md` first: it defines every question,
positive class, base rate and operating-point rule used here.
**Measured on:** v8 seed 1001. TEST 2025. RAW unless marked. Seeds 7 / 17 / 27 / 37 / 47.
**Code:** `ml/eval/phase23ac_t4_metrics.py` at `d401cb3`, run 2026-10-05T18:16:43. Long form:
`metrics_pack.csv`. Log: `ml/artifacts/phase23ac/t4/metrics.log`.
**Companions:** `definitions.md`, `progress.md`, `reports/part2/phase-23ac-preregistration.md`.
**Status:** complete. 0 recomputed figure(s) differ from the published one at its precision (FINDINGS below).

**How to read a cell.** Per-seed arms show the 5-seed mean [min, max]. Ensembles and blends show the point [snapshot-block 95%, 1,000 resamples].
(Q) marks a figure QUOTED from a stored artifact or a report; everything else is RECOMPUTED from stored test
predictions. Every number in the INCUMBENT and BEST PUBLISHED columns is **LEAKY (superseded, Phase 22)**: it was produced
from the published panel, which carries the nine leaking columns.

## Sanity checks (printed by the code; each can fail)

1. **Recomputed vs quoted:** 79 of 79 match at the published precision. 0 FINDING(s) are listed below and are not smoothed.
2. **Accuracy and the majority baseline:** the writer scans this file. Every line that mentions accuracy names the
   majority baseline. The constructed failing line is flagged: flagged ['test accuracy 0.731'].
3. **Always-majority predictor:** base 0.729: accuracy 0.729 = majority 0.729; precision 0.729 = base rate, lift 1.00 (exposed). base 0.239: accuracy 0.761 = majority 0.761; recall 0, F1 0, no flagged case (exposed).
4. **AUC by two methods:** 95 AUC pairs were computed in this run. The largest
   difference between methods is 2.22e-16; the tolerance is 1e-6. The
   hand tie case (y = [0, 0, 1, 1], s = [0.5, 0.5, 0.5, 0.9]) gives ROC-AUC 0.75 and PR-AUC 0.75 by both methods.
   Across 20 random tie-heavy cases, the largest difference is
   2.22e-16.

## FINDINGS: recomputed figures that do not reproduce the published value

None: every reference figure reproduces at its published precision.

<details><summary>All reference comparisons</summary>

| arm | metric | quoted | recomputed | verdict | source |
|---|---|---|---|---|---|
| uc1_inc | base_rate | 0.73 | 0.73024 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.85.precision | 0.869 | 0.86913 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.85.recall | 0.396 | 0.39564 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.70.precision | 0.731 | 0.73056 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.70.recall | 0.999 | 0.99932 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.80.precision | 0.825 | 0.82546 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.80.recall | 0.584 | 0.58436 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.90.precision | 0.91 | 0.90996 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | bar.0.90.recall | 0.244 | 0.24424 | MATCH | reports/part2/phase-15.md §1 |
| uc1_inc | cov.5%.precision | 0.968 | 0.96756 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_inc | phase15.cls | WATCHLIST | WATCHLIST | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_bp | cov.1%.precision | 0.991 | 0.99148 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_bp | cov.5%.precision | 0.969 | 0.96870 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_bp | cov.10%.precision | 0.952 | 0.95161 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_bp | cov.20%.precision | 0.912 | 0.91247 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_bp_blend | cov.5%.precision | 0.943 | 0.94308 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc1_cln | cov.5%.precision | 0.948 | 0.94764 | MATCH | reports/part2/phase-22.md §1 |
| uc1_cln | phase15.cls | WATCHLIST | WATCHLIST | MATCH | reports/part2/phase-22.md §1 |
| uc2_inc | base_rate | 0.749 | 0.74925 | MATCH | reports/part2/phase-15.md §1 |
| uc2_inc | bar.0.85.recall | 0.475 | 0.47520 | MATCH | reports/part2/phase-15.md §1 |
| uc2_inc | bar.0.85.precision | 0.821 | 0.82085 | MATCH | reports/part2/phase-15.md §1 |
| uc2_inc | bar.0.80.recall | 0.885 | 0.88502 | MATCH | reports/part2/phase-15.md §1 |
| uc2_inc | bar.0.80.precision | 0.774 | 0.77425 | MATCH | reports/part2/phase-15.md §1 |
| uc2_bp | cov.1%.precision | 0.961 | 0.96111 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc2_bp | cov.5%.precision | 0.927 | 0.92722 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc2_bp | bar.0.85.recall | 0.627 | 0.62674 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc2_bp | bar.0.85.precision | 0.827 | 0.82705 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc2_cln | cov.5%.precision | 0.944 | 0.94444 | MATCH | reports/part2/phase-22.md §1 |
| uc2_cln_p19 | cov.5%.precision | 0.927 | 0.92722 | MATCH | reports/part2/phase22/stage3_fill.md fill_final clean_uc2_p5 |
| uc2b_inc | base_rate | 0.245 | 0.24458 | MATCH | reports/part2/phase-15.md §1 |
| uc2b_inc | bar.0.70.status | UNREACHABLE on validation (0 of 5 seeds reach) | UNREACHABLE on validation (0 of 5 seeds reach) | MATCH | reports/part2/phase-15.md §1 |
| uc2b_bp | cov.1%.precision | 0.644 | 0.64444 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc2b_bp | cov.5%.precision | 0.53 | 0.53000 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc2b_cln | cov.5%.precision | 0.511 | 0.51056 | MATCH | reports/part2/phase-22.md §1 |
| uc2b_cln_cons | cov.5%.precision | 0.461 | 0.46111 | MATCH | reports/part2/phase-22.md §1 |
| uc3_inc | base_rate | 0.405 | 0.40524 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | cov.1%.precision | 0.904 | 0.90400 | MATCH | reports/part2/phase20/stage1_decisions.md anchors |
| uc3_inc | cov.5%.precision | 0.851 | 0.85084 | MATCH | reports/part2/phase20/stage1_decisions.md anchors |
| uc3_inc | cov.10%.precision | 0.81 | 0.81040 | MATCH | reports/part2/phase20/stage1_decisions.md anchors |
| uc3_inc | bar.0.85.recall | 0.191 | 0.19149 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | bar.0.85.precision | 0.81 | 0.81031 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | bar.0.70.recall | 0.515 | 0.51503 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | bar.0.70.precision | 0.681 | 0.68135 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | bar.0.80.recall | 0.318 | 0.31783 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | bar.0.80.precision | 0.762 | 0.76156 | MATCH | reports/part2/phase-15.md §1 |
| uc3_inc | phase15.cls | ALERT | ALERT | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc3_bp | cov.1%.precision | 0.907 | 0.90667 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc3_bp | cov.5%.precision | 0.865 | 0.86489 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc3_bp | cov.10%.precision | 0.815 | 0.81467 | MATCH | reports/part2/phase20/stage1_decisions.md decision table |
| uc3_bp | spread.p5.min | 0.456 | 0.45600 | MATCH | reports/part2/phase20/stage1_decisions.md per-snapshot table |
| uc3_cln | cov.1%.precision | 0.828 | 0.82756 | MATCH | reports/part2/phase22/stage4_capacity.md first table |
| uc3_cln | cov.5%.precision | 0.735 | 0.73476 | MATCH | reports/part2/phase-22.md §1 |
| uc3_cln | cov.10%.precision | 0.679 | 0.67938 | MATCH | reports/part2/phase22/stage4_capacity.md first table |
| uc3_cln | bar.0.70.recall | 0.308 | 0.30814 | MATCH | reports/part2/phase22/stage4_capacity.md first table |
| uc3_cln | bar.0.80.recall | 0.116 | 0.11617 | MATCH | reports/part2/phase22/stage4_capacity.md first table |
| uc3_cln | phase15.cls | WATCHLIST | WATCHLIST | MATCH | reports/part2/phase-22.md §1 |
| arr_inc | lateness_auc | 0.709 | 0.709048 | MATCH | reports/part2/phase-22.md §1 |
| arr_inc | a3_days | 13.06 | 13.0555 | MATCH | reports/part2/phase-22.md §1 |
| arr_cln_seed | lateness_auc | 0.6951 | 0.695104 | MATCH | reports/part2/phase-22.md §1 |
| arr_cln_seed | a3_days | 13.15 | 13.1519 | MATCH | reports/part2/phase-22.md §1 |
| arr_bp | a3_days | 12.34 | 12.3405 | MATCH | reports/part2/phase20/stage1_decisions.md 'What changes' |
| arr_cln | a3_days | 12.57 | 12.5705 | MATCH | reports/part2/phase-22.md §1 |
| arr_cln | lateness_auc | 0.7217 | 0.721691 | MATCH | reports/part2/phase-22.md §1 |
| fill_inc | crps_exact | 0.13876 | 0.1387619 | MATCH | reports/part2/phase-22.md §1 (published neural CRPS) |
| fill_inc | p_full_auc | 0.6204 | 0.620402 | MATCH | reports/part2/phase-22.md §1 |
| fill_bp | crps_exact | 0.1354 | 0.135370 | MATCH | reports/part2/phase-22.md §1 |
| fill_cln | crps_exact | 0.13444 | 0.1344415 | MATCH | reports/part2/phase-22.md §1 |
| fill_cln_nsc | crps_exact | 0.1375 | 0.137459 | MATCH | reports/part2/phase-22.md §1 |
| fill_cln_nsc | p_full_auc | 0.64 | 0.64018 | MATCH | reports/part2/phase-22.md §1 |
| fill_cln_p19 | crps_exact | 0.1358 | 0.135755 | MATCH | reports/part2/phase-22.md §6 errata |
| ot_flag_cln | cov.5%.precision | 0.615 | 0.61522 | MATCH | reports/part2/phase22/stage2_order_time.md first table |
| ot_flag_cln | base_rate | 0.349 | 0.34941 | MATCH | reports/part2/phase22/stage2_order_time.md header |
| ot_flag_cln | lateness_auc | 0.6228 | 0.622809 | MATCH | reports/part2/phase22/stage2_order_time.md first table |
| ot_flag_cln | phase15.cls | WATCHLIST | WATCHLIST | MATCH | reports/part2/phase22/stage2_order_time.md first table |
| weights | arrival_published_w_neural | 0.52 | 0.52 | MATCH | brief / stored JSON |
| weights | fill_published_w_neural | 0.5 | 0.5 | MATCH | brief / stored JSON |
| weights | arrival_clean_w_neural | 0.52 | 0.52 | MATCH | brief / stored JSON |
| weights | fill_consolidated_w_neural | 0.02 | 0.02 | MATCH | brief / stored JSON |
| weights | fill_p19_clean_w_neural | 0.49 | 0.49 | MATCH | brief / stored JSON |

</details>

## UC1: arrival late vs contract (snapshot rows; base 0.73)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| INCUMBENT | headline | neural lite h4, per seed (Phase 15's arm), P(late) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s{s}/preds_{f}.npz` (3f2b5c8, b1794c0, da18a91) |
| BEST PUBLISHED | headline | incumbent neural 5-seed ensemble P(late) (Phase 20 kept it) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s{s}/preds_{f}.npz` (3f2b5c8, b1794c0, da18a91) |
| BEST PUBLISHED | also | Phase 19 blend point score - contract (w_neural 0.52) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase19/bundles/arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence/preds_{f}.npz`; `ml/artifacts/phase18/preds/v8_arrival_week_p18_fwd_load_s{s}_{f}.npz` (11f0c93, untracked artifact (no stamp)) |
| CLEAN | headline | clean incumbent neural 5-seed ensemble P(late) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s{s}/preds_{f}.npz` (5f9fc7d, 75e77d7) |
| CLEAN | also | clean incumbent neural, per seed, P(late) | 5-seed mean [min, max] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s{s}/preds_{f}.npz` (5f9fc7d, 75e77d7) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| base rate (positive class) | 0.730 [0.730, 0.730] | 0.730 [0.703, 0.753] | 0.730 [0.703, 0.753] |
| majority-class predictor | positive (flag everything) | positive (flag everything) | positive (flag everything) |
| majority-class accuracy (the baseline) | 0.730 [0.730, 0.730] | 0.730 | 0.730 |
| ROC-AUC (sklearn) | 0.6753 [0.6725, 0.6789] | 0.6807 [0.6734, 0.6893] | 0.6596 [0.6459, 0.6746] |
| ROC-AUC (own rank implementation) | 0.6753 [0.6725, 0.6789] | 0.6807 | 0.6596 |
| PR-AUC (sklearn; floor = base rate) | 0.8509 [0.8493, 0.8530] | 0.8541 [0.8357, 0.8684] | 0.8394 [0.8191, 0.8564] |
| PR-AUC (own step implementation) | 0.8509 [0.8493, 0.8530] | 0.8541 | 0.8394 |
| precision @ 1% coverage | 0.986 [0.980, 0.991] | 0.991 [0.977, 1.000] | 0.966 [0.946, 0.983] |
| lift @ 1% | 1.35 [1.34, 1.36] | 1.36 | 1.32 |
| precision @ 5% coverage | 0.968 [0.963, 0.971] | 0.969 [0.955, 0.978] | 0.948 [0.930, 0.965] |
| lift @ 5% | 1.32 [1.32, 1.33] | 1.33 | 1.30 |
| precision @ 10% coverage | 0.947 [0.943, 0.950] | 0.952 [0.935, 0.964] | 0.924 [0.902, 0.943] |
| lift @ 10% | 1.30 [1.29, 1.30] | 1.30 | 1.26 |
| precision @ 20% coverage | 0.908 [0.905, 0.912] | 0.912 [0.894, 0.930] | 0.888 [0.869, 0.908] |
| lift @ 20% | 1.24 [1.24, 1.25] | 1.25 | 1.22 |
| recall @ 5% | 0.066 [0.066, 0.067] | 0.066 | 0.065 |
| TP @ 5% | 1700.0 [1692.0, 1706.0] | 1702 | 1665 |
| FP @ 5% | 57.0 [51.0, 65.0] | 55 | 92 |
| FN @ 5% | 23952.0 [23946.0, 23960.0] | 23950 | 23987 |
| TN @ 5% | 9419.0 [9411.0, 9425.0] | 9421 | 9384 |
| plain ratio @ 5% | right 9.7 times in 10 at the flagged cases against 7.3 in 10 by chance | right 9.7 times in 10 at the flagged cases against 7.3 in 10 by chance | right 9.5 times in 10 at the flagged cases against 7.3 in 10 by chance |
| validation max-F1 threshold | 0.305 [0.247, 0.374] | 0.315 | 0.416 |
| precision @ val max-F1 | 0.732 [0.731, 0.734] | 0.732 [0.705, 0.754] | 0.744 [0.717, 0.766] |
| recall @ val max-F1 | 0.996 [0.992, 0.999] | 0.997 [0.994, 0.998] | 0.954 [0.935, 0.970] |
| F1 @ val max-F1 | 0.844 [0.843, 0.844] | 0.844 [0.826, 0.858] | 0.836 [0.819, 0.849] |
| accuracy @ val max-F1 (vs majority) | 0.731 [0.730, 0.731] vs majority 0.730 | 0.731 [0.704, 0.752] vs majority 0.730 | 0.727 [0.704, 0.746] vs majority 0.730 |
| share flagged @ val max-F1 | 0.993 [0.987, 0.998] | 0.994 [0.990, 0.997] | 0.937 [0.912, 0.958] |
| lift @ val max-F1 | 1.00 [1.00, 1.00] | 1.00 | 1.02 |
| TP @ val max-F1 | 25538.2 [25438.0, 25616.0] | 25563 | 24476 |
| FP @ val max-F1 | 9344.4 [9234.0, 9433.0] | 9364 | 8427 |
| FN @ val max-F1 | 113.8 [36.0, 214.0] | 89 | 1176 |
| TN @ val max-F1 | 131.6 [43.0, 242.0] | 112 | 1049 |
| bar p = 0.70: status | reached on validation | reached on validation | reached on validation |
| bar 0.70: test precision | 0.731 [0.730, 0.731] | 0.730 [0.703, 0.753] | 0.730 [0.703, 0.753] |
| bar 0.70: test recall | 0.999 [0.998, 1.000] | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] |
| bar 0.70: share flagged | 0.999 [0.996, 1.000] | 0.999 | 0.999 |
| bar 0.70: lift | 1.00 [1.00, 1.00] | 1.00 | 1.00 |
| bar 0.70: accuracy (vs majority) | 0.730 [0.730, 0.731] vs majority 0.730 | 0.730 vs majority 0.730 | 0.730 vs majority 0.730 |
| bar 0.70: TP | 25634.6 [25595.0, 25649.0] | 25640 | 25639 |
| bar 0.70: FP | 9454.6 [9408.0, 9475.0] | 9460 | 9462 |
| bar 0.70: FN | 17.4 [3.0, 57.0] | 12 | 13 |
| bar 0.70: TN | 21.4 [1.0, 68.0] | 16 | 14 |
| bar p = 0.80: status | reached on validation | reached on validation | reached on validation |
| bar 0.80: test precision | 0.825 [0.815, 0.845] | 0.826 [0.812, 0.837] | 0.849 [0.826, 0.867] |
| bar 0.80: test recall | 0.584 [0.494, 0.639] | 0.597 [0.564, 0.625] | 0.426 [0.372, 0.473] |
| bar 0.80: share flagged | 0.518 [0.427, 0.572] | 0.527 | 0.366 |
| bar 0.80: lift | 1.13 [1.12, 1.16] | 1.13 | 1.16 |
| bar 0.80: accuracy (vs majority) | 0.606 [0.564, 0.630] vs majority 0.730 | 0.614 vs majority 0.730 | 0.525 vs majority 0.730 |
| bar 0.80: TP | 14990.0 [12670.0, 16388.0] | 15305 | 10920 |
| bar 0.80: FP | 3193.0 [2331.0, 3719.0] | 3225 | 1947 |
| bar 0.80: FN | 10662.0 [9264.0, 12982.0] | 10347 | 14732 |
| bar 0.80: TN | 6283.0 [5757.0, 7145.0] | 6251 | 7529 |
| bar p = 0.85: status | reached on validation | reached on validation | reached on validation |
| bar 0.85: test precision | 0.869 [0.857, 0.887] | 0.869 [0.858, 0.878] | 0.886 [0.867, 0.902] |
| bar 0.85: test recall | 0.396 [0.326, 0.451] | 0.409 [0.378, 0.438] | 0.258 [0.219, 0.292] |
| bar 0.85: share flagged | 0.333 [0.269, 0.384] | 0.343 | 0.212 |
| bar 0.85: lift | 1.19 [1.17, 1.21] | 1.19 | 1.21 |
| bar 0.85: accuracy (vs majority) | 0.515 [0.478, 0.544] vs majority 0.730 | 0.523 vs majority 0.730 | 0.434 vs majority 0.730 |
| bar 0.85: TP | 10149.0 [8374.0, 11575.0] | 10484 | 6607 |
| bar 0.85: FP | 1543.4 [1068.0, 1929.0] | 1577 | 854 |
| bar 0.85: FN | 15503.0 [14077.0, 17278.0] | 15168 | 19045 |
| bar 0.85: TN | 7932.6 [7547.0, 8408.0] | 7899 | 8622 |
| bar p = 0.90: status | reached on validation | reached on validation | reached on validation |
| bar 0.90: test precision | 0.910 [0.902, 0.923] | 0.913 [0.902, 0.921] | 0.926 [0.906, 0.941] |
| bar 0.90: test recall | 0.244 [0.201, 0.286] | 0.248 [0.223, 0.273] | 0.122 [0.099, 0.144] |
| bar 0.90: share flagged | 0.196 [0.159, 0.231] | 0.198 | 0.097 |
| bar 0.90: lift | 1.25 [1.24, 1.26] | 1.25 | 1.27 |
| bar 0.90: accuracy (vs majority) | 0.430 [0.404, 0.456] vs majority 0.730 | 0.434 vs majority 0.730 | 0.352 vs majority 0.730 |
| bar 0.90: TP | 6265.2 [5155.0, 7329.0] | 6362 | 3142 |
| bar 0.90: FP | 626.0 [429.0, 795.0] | 608 | 252 |
| bar 0.90: FN | 19386.8 [18323.0, 20497.0] | 19290 | 22510 |
| bar 0.90: TN | 8850.0 [8681.0, 9047.0] | 8868 | 9224 |
| **Phase 15 class** | WATCHLIST | WATCHLIST | WATCHLIST |
| REACHABLE | YES | YES | YES |
| operating bar | 0.9 | 0.9 | 0.9 |
| lift at the operating bar | 1.25 | 1.25 | 1.27 |
| Stage B | TUNABLE | TUNABLE | TUNABLE |
| per-snapshot precision @ 5%: min | 0.937 | 0.939 | 0.892 |
| per-snapshot precision @ 5%: median | 0.969 | 0.969 | 0.944 |
| per-snapshot precision @ 5%: max | 0.987 | 0.985 | 0.985 |
| worst snapshot | 2025-05-12 (0.937) | 2025-05-12 (0.939) | 2025-08-04 (0.892) |

Other arms in the same column (headline metric `cov.5%.precision`):

| column | arm | cov.5%.precision | class | label |
|---|---|---|---|---|
| BEST PUBLISHED | Phase 19 blend point score - contract (w_neural 0.52) | 0.943 [0.926, 0.954] | WATCHLIST | LEAKY (superseded, Phase 22) |
| CLEAN | clean incumbent neural, per seed, P(late) | 0.940 [0.929, 0.949] | WATCHLIST | clean |

- **INCUMBENT** (top 5%): right 9.7 times in 10 at the flagged cases against 7.3 in 10 by chance.
- **BEST PUBLISHED** (top 5%): right 9.7 times in 10 at the flagged cases against 7.3 in 10 by chance.
- **CLEAN** (top 5%): right 9.5 times in 10 at the flagged cases against 7.3 in 10 by chance.

## Arrival point estimate (snapshot rows)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| INCUMBENT | headline | neural lite h4, per seed: expected week | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s{s}/preds_{f}.npz` (3f2b5c8, b1794c0, da18a91) |
| BEST PUBLISHED | headline | Phase 19 blend (w_neural 0.52) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase19/bundles/arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence/preds_{f}.npz`; `ml/artifacts/phase18/preds/v8_arrival_week_p18_fwd_load_s{s}_{f}.npz` (11f0c93, untracked artifact (no stamp)) |
| CLEAN | headline | clean Phase 19 recipe (w_neural 0.52) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s{s}/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_arrival_week_p22_fwdload_s{s}_{f}.npz` (5f9fc7d, 75e77d7, untracked artifact (no stamp)) |
| CLEAN | also | clean incumbent neural, per seed: expected week | 5-seed mean [min, max] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s{s}/preds_{f}.npz` (5f9fc7d, 75e77d7) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| A3: median abs error (days) | 13.06 [12.86, 13.30] | 12.34 [11.95, 12.84] | 12.57 [12.22, 13.05] |
| MAE (days) | 14.68 [14.39, 15.05] | 13.77 [13.46, 14.16] | 13.93 [13.62, 14.29] |
| signed bias (days, + = late) | 5.96 [4.73, 7.55] | 3.12 [2.57, 3.62] | 3.11 [2.20, 3.99] |
| share within ±7 days | 0.285 [0.281, 0.289] | 0.286 [0.276, 0.295] | 0.281 [0.271, 0.291] |
| C-index | 0.6744 [0.6734, 0.6755] | 0.6804 | 0.6695 |
| lateness AUC (sklearn) | 0.7090 [0.7064, 0.7130] | 0.7313 [0.7247, 0.7374] | 0.7217 [0.7148, 0.7300] |
| lateness AUC (own rank implementation) | 0.7090 [0.7064, 0.7130] | 0.7313 | 0.7217 |
| 80% interval coverage | 0.930 [0.921, 0.943] | n/a (a blend has no interval) | n/a (a blend has no interval) |
| 80% interval width, median (days) | 63.00 [63.00, 63.00] | — | — |
| share with an open upper end (> 12 weeks) | 0.956 [0.912, 0.983] | — | — |
| per-snapshot A3: best | 12.25 | 11.30 | 11.74 |
| per-snapshot A3: median | 13.00 | 12.35 | 12.46 |
| per-snapshot A3: worst | 14.52 | 14.03 | 14.20 |
| worst snapshot | 2025-08-04 (14.52) | 2025-08-04 (14.03) | 2025-08-04 (14.2) |

Other arms in the same column (headline metric `a3_days`):

| column | arm | a3_days | class | label |
|---|---|---|---|---|
| CLEAN | clean incumbent neural, per seed: expected week | 13.15 [12.91, 13.40] | — | clean |

## Order-time arrival: expected date and 80% interval (Phase 22 product)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| CLEAN | headline | product date: shrunk-KM median (k = 10) + offset; month-specific split-conformal 80% interval | as quoted | clean | QUOTED | `ml/artifacts/phase22/order_time_v8clean.json`; `ml/artifacts/phase23ac/t1/compare_v8.json` (2b602b5, d401cb3) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| A3: median abs error (days) | — | — | 10.02 [9.67, 10.34] (Q) |
| share within ±7 days | — | — | 0.378 (Q) |
| signed bias (days) | — | — | -8.38 (Q) |
| expected-week hit rate | — | — | 0.199 (Q) |
| lateness AUC of the date | — | — | 0.5324 (Q) |
| 80% interval coverage | — | — | 0.803 [0.797, 0.808] (Q) |
| 80% interval width, mean (days) | — | — | 53.82 (Q) |
| coverage, worst creation month | — | — | 0.767 (Q) |
| coverage, best creation month | — | — | 0.829 (Q) |
| width, narrowest month (days) | — | — | 37.40 (Q) |
| width, widest month (days) | — | — | 70.31 (Q) |

## Order-time arrival: strict as-of late flag (base 0.35)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| BEST PUBLISHED | headline | Phase 21 at-placement flag (BASE_nl; still carried six leaking columns, deviation 210) | as quoted | LEAKY (superseded, Phase 22) | QUOTED | `reports/part2/phase-21.md` (eca525d) |
| CLEAN | headline | strict as-of late flag (flag_lag1), per seed | 5-seed mean [min, max] | clean | RECOMPUTED | `ml/artifacts/phase22/preds/v8clean_arrival_place_p22_flag_lag1_k10_s{s}_{f}.npz` (untracked artifact (no stamp)) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| base rate (positive class) | — | 0.350 (Q) | 0.349 [0.349, 0.349] |
| majority-class predictor | — | — | negative (flag nothing) |
| majority-class accuracy (the baseline) | — | — | 0.651 [0.651, 0.651] |
| ROC-AUC (sklearn) | — | — | 0.6245 [0.6221, 0.6262] |
| ROC-AUC (own rank implementation) | — | — | 0.6245 [0.6221, 0.6262] |
| PR-AUC (sklearn; floor = base rate) | — | — | 0.4729 [0.4711, 0.4751] |
| PR-AUC (own step implementation) | — | — | 0.4729 [0.4711, 0.4751] |
| lateness AUC (vs contract, uncensored) | — | — | 0.6228 [0.6202, 0.6246] |
| precision @ 1% coverage | — | — | 0.713 [0.692, 0.726] |
| lift @ 1% | — | — | 2.04 [1.98, 2.08] |
| precision @ 5% coverage | — | 0.790 (Q) | 0.615 [0.607, 0.619] |
| lift @ 5% | — | — | 1.76 [1.74, 1.77] |
| precision @ 10% coverage | — | — | 0.562 [0.559, 0.564] |
| lift @ 10% | — | — | 1.61 [1.60, 1.61] |
| precision @ 20% coverage | — | — | 0.506 [0.501, 0.513] |
| lift @ 20% | — | — | 1.45 [1.43, 1.47] |
| recall @ 5% | — | — | 0.088 [0.087, 0.089] |
| TP @ 5% | — | — | 978.2 [965.0, 985.0] |
| FP @ 5% | — | — | 611.8 [605.0, 625.0] |
| FN @ 5% | — | — | 10130.8 [10124.0, 10144.0] |
| TN @ 5% | — | — | 20073.2 [20060.0, 20080.0] |
| plain ratio @ 5% | — | right 7.9 times in 10 at the flagged cases against 3.5 in 10 by chance (Q) | right 6.2 times in 10 at the flagged cases against 3.5 in 10 by chance |
| validation max-F1 threshold | — | — | 0.276 [0.262, 0.287] |
| precision @ val max-F1 | — | — | 0.400 [0.398, 0.403] |
| recall @ val max-F1 | — | — | 0.772 [0.763, 0.776] |
| F1 @ val max-F1 | — | — | 0.527 [0.525, 0.528] |
| accuracy @ val max-F1 (vs majority) | — | — | 0.515 [0.511, 0.523] vs majority 0.651 |
| share flagged @ val max-F1 | — | — | 0.675 [0.661, 0.680] |
| lift @ val max-F1 | — | — | 1.14 [1.14, 1.15] |
| TP @ val max-F1 | — | — | 8575.0 [8475.0, 8623.0] |
| FP @ val max-F1 | — | — | 12877.8 [12543.0, 13028.0] |
| FN @ val max-F1 | — | — | 2534.0 [2486.0, 2634.0] |
| TN @ val max-F1 | — | — | 7807.2 [7657.0, 8142.0] |
| bar p = 0.70: status | — | — | reached on validation |
| bar 0.70: test precision | — | — | 0.713 [0.678, 0.759] |
| bar 0.70: test recall | — | — | 0.007 [0.002, 0.011] |
| bar 0.70: share flagged | — | — | 0.004 [0.001, 0.005] |
| bar 0.70: lift | — | — | 2.04 [1.94, 2.17] |
| bar 0.70: accuracy (vs majority) | — | — | 0.652 [0.651, 0.653] vs majority 0.651 |
| bar 0.70: TP | — | — | 82.8 [22.0, 117.0] |
| bar 0.70: FP | — | — | 34.2 [7.0, 48.0] |
| bar 0.70: FN | — | — | 11026.2 [10992.0, 11087.0] |
| bar 0.70: TN | — | — | 20650.8 [20637.0, 20678.0] |
| bar p = 0.80: status | — | — | UNREACHABLE on validation (0 of 5 seeds reach) |
| bar 0.80: max validation precision | — | — | 0.764 [0.706, 0.796] |
| bar p = 0.85: status | — | — | UNREACHABLE on validation (0 of 5 seeds reach) |
| bar 0.85: max validation precision | — | — | 0.764 [0.706, 0.796] |
| bar p = 0.90: status | — | — | UNREACHABLE on validation (0 of 5 seeds reach) |
| bar 0.90: max validation precision | — | — | 0.764 [0.706, 0.796] |
| **Phase 15 class** | — | ALERT (Q) | WATCHLIST |
| REACHABLE | — | — | NO, CEILING |
| operating bar | — | — | None |
| lift at the operating bar | — | 2.20 (Q) | n/a |
| Stage B | — | — | TUNABLE |
| per-snapshot precision @ 5%: min | — | — | 0.495 |
| per-snapshot precision @ 5%: median | — | — | 0.611 |
| per-snapshot precision @ 5%: max | — | — | 0.726 |
| worst snapshot | — | — | 2025-05-01 (0.495) |

- **BEST PUBLISHED** (top 5%): right 7.9 times in 10 at the flagged cases against 3.5 in 10 by chance.
- **CLEAN** (top 5%): right 6.2 times in 10 at the flagged cases against 3.5 in 10 by chance.

## UC2: fill arrives in full (base 0.749)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| INCUMBENT | headline | boundary bw3, per seed (Phase 15's UC2 arm) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s{s}_lossrps_bw3/preds_{f}.npz` (6fe2244) |
| BEST PUBLISHED | headline | Phase 19 fill blend (w_neural 0.5) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase19/bundles/fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase18/preds/v8_fill_rate_p18_fwd_load_s{s}_{f}.npz` (11f0c93, untracked artifact (no stamp)) |
| CLEAN | headline | consolidated blend (w_neural 0.02) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_fill_rate_p22_sc_ack_L4_s{s}_{f}.npz` (2cee433, untracked artifact (no stamp)) |
| CLEAN | also | clean Phase 19 fill blend (w_neural 0.49) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_fill_rate_p22_fwdload_s{s}_{f}.npz` (2cee433, untracked artifact (no stamp)) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| base rate (positive class) | 0.749 [0.749, 0.749] | 0.749 [0.706, 0.792] | 0.749 [0.706, 0.792] |
| majority-class predictor | positive (flag everything) | positive (flag everything) | positive (flag everything) |
| majority-class accuracy (the baseline) | 0.749 [0.749, 0.749] | 0.749 | 0.749 |
| ROC-AUC (sklearn) | 0.6222 [0.6216, 0.6226] | 0.6639 [0.6327, 0.6971] | 0.6688 [0.6429, 0.6951] |
| ROC-AUC (own rank implementation) | 0.6222 [0.6216, 0.6226] | 0.6639 | 0.6688 |
| PR-AUC (sklearn; floor = base rate) | 0.8194 [0.8180, 0.8209] | 0.8465 [0.8019, 0.8845] | 0.8546 [0.8186, 0.8853] |
| PR-AUC (own step implementation) | 0.8194 [0.8180, 0.8209] | 0.8465 | 0.8546 |
| precision @ 1% coverage | 0.892 [0.881, 0.925] | 0.961 [0.867, 0.983] | 0.969 [0.950, 0.989] |
| lift @ 1% | 1.19 [1.18, 1.23] | 1.28 | 1.29 |
| precision @ 5% coverage | 0.876 [0.866, 0.884] | 0.927 [0.859, 0.948] | 0.944 [0.928, 0.964] |
| lift @ 5% | 1.17 [1.16, 1.18] | 1.24 | 1.26 |
| precision @ 10% coverage | 0.866 [0.859, 0.871] | 0.912 [0.853, 0.940] | 0.927 [0.904, 0.945] |
| lift @ 10% | 1.16 [1.15, 1.16] | 1.22 | 1.24 |
| precision @ 20% coverage | 0.848 [0.847, 0.852] | 0.882 [0.839, 0.925] | 0.900 [0.873, 0.921] |
| lift @ 20% | 1.13 [1.13, 1.14] | 1.18 | 1.20 |
| recall @ 5% | 0.058 [0.058, 0.059] | 0.062 | 0.063 |
| TP @ 5% | 1576.2 [1559.0, 1591.0] | 1669 | 1700 |
| FP @ 5% | 223.8 [209.0, 241.0] | 131 | 100 |
| FN @ 5% | 25396.8 [25382.0, 25414.0] | 25304 | 25273 |
| TN @ 5% | 8803.2 [8786.0, 8818.0] | 8896 | 8927 |
| plain ratio @ 5% | right 8.8 times in 10 at the flagged cases against 7.5 in 10 by chance | right 9.3 times in 10 at the flagged cases against 7.5 in 10 by chance | right 9.4 times in 10 at the flagged cases against 7.5 in 10 by chance |
| validation max-F1 threshold | 0.328 [0.249, 0.405] | 0.498 | 0.457 |
| precision @ val max-F1 | 0.750 [0.749, 0.750] | 0.753 [0.712, 0.794] | 0.751 [0.710, 0.793] |
| recall @ val max-F1 | 1.000 [0.999, 1.000] | 0.996 [0.991, 0.999] | 0.996 [0.993, 0.999] |
| F1 @ val max-F1 | 0.857 [0.857, 0.857] | 0.857 [0.829, 0.884] | 0.857 [0.828, 0.884] |
| accuracy @ val max-F1 (vs majority) | 0.749 [0.749, 0.750] vs majority 0.749 | 0.752 [0.711, 0.794] vs majority 0.749 | 0.750 [0.708, 0.793] vs majority 0.749 |
| share flagged @ val max-F1 | 0.999 [0.998, 1.000] | 0.991 [0.984, 0.997] | 0.993 [0.989, 0.997] |
| lift @ val max-F1 | 1.00 [1.00, 1.00] | 1.00 | 1.00 |
| TP @ val max-F1 | 26959.6 [26940.0, 26969.0] | 26857 | 26873 |
| FP @ val max-F1 | 9010.2 [8981.0, 9026.0] | 8814 | 8887 |
| FN @ val max-F1 | 13.4 [4.0, 33.0] | 116 | 100 |
| TN @ val max-F1 | 16.8 [1.0, 46.0] | 213 | 140 |
| bar p = 0.70: status | reached on validation | reached on validation | reached on validation |
| bar 0.70: test precision | 0.749 [0.749, 0.749] | 0.749 [0.706, 0.792] | 0.749 [0.706, 0.792] |
| bar 0.70: test recall | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| bar 0.70: share flagged | 1.000 [1.000, 1.000] | 1.000 | 1.000 |
| bar 0.70: lift | 1.00 [1.00, 1.00] | 1.00 | 1.00 |
| bar 0.70: accuracy (vs majority) | 0.749 [0.749, 0.749] vs majority 0.749 | 0.749 vs majority 0.749 | 0.749 vs majority 0.749 |
| bar 0.70: TP | 26972.4 [26970.0, 26973.0] | 26973 | 26973 |
| bar 0.70: FP | 9026.8 [9026.0, 9027.0] | 9027 | 9027 |
| bar 0.70: FN | 0.6 [0.0, 3.0] | 0 | 0 |
| bar 0.70: TN | 0.2 [0.0, 1.0] | 0 | 0 |
| bar p = 0.80: status | reached on validation | reached on validation | reached on validation |
| bar 0.80: test precision | 0.774 [0.772, 0.776] | 0.772 [0.740, 0.806] | 0.774 [0.742, 0.807] |
| bar 0.80: test recall | 0.885 [0.876, 0.897] | 0.936 [0.884, 0.974] | 0.908 [0.841, 0.960] |
| bar 0.80: share flagged | 0.856 [0.845, 0.871] | 0.907 | 0.879 |
| bar 0.80: lift | 1.03 [1.03, 1.04] | 1.03 | 1.03 |
| bar 0.80: accuracy (vs majority) | 0.720 [0.718, 0.725] vs majority 0.749 | 0.745 vs majority 0.749 | 0.733 vs majority 0.749 |
| bar 0.80: TP | 23871.6 [23618.0, 24202.0] | 25235 | 24503 |
| bar 0.80: FP | 6961.0 [6801.0, 7143.0] | 7434 | 7156 |
| bar 0.80: FN | 3101.4 [2771.0, 3355.0] | 1738 | 2470 |
| bar 0.80: TN | 2066.0 [1884.0, 2226.0] | 1593 | 1871 |
| bar p = 0.85: status | reached on validation | reached on validation | reached on validation |
| bar 0.85: test precision | 0.821 [0.816, 0.825] | 0.827 [0.806, 0.852] | 0.833 [0.814, 0.856] |
| bar 0.85: test recall | 0.475 [0.433, 0.504] | 0.627 [0.473, 0.763] | 0.602 [0.450, 0.752] |
| bar 0.85: share flagged | 0.434 [0.394, 0.463] | 0.568 | 0.541 |
| bar 0.85: lift | 1.10 [1.09, 1.10] | 1.10 | 1.11 |
| bar 0.85: accuracy (vs majority) | 0.529 [0.507, 0.543] vs majority 0.749 | 0.622 vs majority 0.749 | 0.611 vs majority 0.749 |
| bar 0.85: TP | 12817.6 [11688.0, 13607.0] | 16905 | 16234 |
| bar 0.85: FP | 2800.4 [2479.0, 3072.0] | 3535 | 3250 |
| bar 0.85: FN | 14155.4 [13366.0, 15285.0] | 10068 | 10739 |
| bar 0.85: TN | 6226.6 [5955.0, 6548.0] | 5492 | 5777 |
| bar p = 0.90: status | reached on validation | reached on validation | reached on validation |
| bar 0.90: test precision | 0.871 [0.862, 0.885] | 0.880 [0.848, 0.902] | 0.901 [0.888, 0.910] |
| bar 0.90: test recall | 0.092 [0.046, 0.107] | 0.248 [0.101, 0.419] | 0.238 [0.149, 0.360] |
| bar 0.90: share flagged | 0.079 [0.039, 0.093] | 0.211 | 0.198 |
| bar 0.90: lift | 1.16 [1.15, 1.18] | 1.17 | 1.20 |
| bar 0.90: accuracy (vs majority) | 0.309 [0.281, 0.319] vs majority 0.749 | 0.411 vs majority 0.749 | 0.409 vs majority 0.749 |
| bar 0.90: TP | 2477.2 [1241.0, 2890.0] | 6697 | 6418 |
| bar 0.90: FP | 373.0 [161.0, 464.0] | 915 | 705 |
| bar 0.90: FN | 24495.8 [24083.0, 25732.0] | 20276 | 20555 |
| bar 0.90: TN | 8654.0 [8563.0, 8866.0] | 8112 | 8322 |
| **Phase 15 class** | WATCHLIST | WATCHLIST | WATCHLIST |
| REACHABLE | PARTIAL | YES | YES |
| operating bar | 0.85 | 0.9 | 0.9 |
| lift at the operating bar | 1.10 | 1.17 | 1.20 |
| Stage B | TUNABLE | TUNABLE | TUNABLE |
| per-snapshot precision @ 5%: min | 0.806 | 0.820 | 0.895 |
| per-snapshot precision @ 5%: median | 0.852 | 0.890 | 0.925 |
| per-snapshot precision @ 5%: max | 0.974 | 0.960 | 0.970 |
| worst snapshot | 2025-05-12 (0.806) | 2025-05-12 (0.820) | 2025-08-04 (0.895) |

Other arms in the same column (headline metric `cov.5%.precision`):

| column | arm | cov.5%.precision | class | label |
|---|---|---|---|---|
| CLEAN | clean Phase 19 fill blend (w_neural 0.49) | 0.927 [0.862, 0.943] | WATCHLIST | clean |

- **INCUMBENT** (top 5%): right 8.8 times in 10 at the flagged cases against 7.5 in 10 by chance.
- **BEST PUBLISHED** (top 5%): right 9.3 times in 10 at the flagged cases against 7.5 in 10 by chance.
- **CLEAN** (top 5%): right 9.4 times in 10 at the flagged cases against 7.5 in 10 by chance.

## UC2b: fill materially short, < 0.95 (base 0.245)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| INCUMBENT | headline | lgbm22_id, per seed (Phase 15's UC2b arm) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase7_preds/v8_fill_rate_lgbm22_id_s{s}_{f}.npz` (untracked artifact (no stamp)) |
| BEST PUBLISHED | headline | Phase 19 fill blend (w_neural 0.5) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase19/bundles/fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase18/preds/v8_fill_rate_p18_fwd_load_s{s}_{f}.npz` (11f0c93, untracked artifact (no stamp)) |
| CLEAN | headline | clean Phase 19 fill blend (w_neural 0.49) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_fill_rate_p22_fwdload_s{s}_{f}.npz` (2cee433, untracked artifact (no stamp)) |
| CLEAN | also | consolidated blend (w_neural 0.02) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_fill_rate_p22_sc_ack_L4_s{s}_{f}.npz` (2cee433, untracked artifact (no stamp)) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| base rate (positive class) | 0.245 [0.245, 0.245] | 0.245 [0.203, 0.287] | 0.245 [0.203, 0.287] |
| majority-class predictor | negative (flag nothing) | negative (flag nothing) | negative (flag nothing) |
| majority-class accuracy (the baseline) | 0.755 [0.755, 0.755] | 0.755 | 0.755 |
| ROC-AUC (sklearn) | 0.6046 [0.6036, 0.6057] | 0.6632 [0.6318, 0.6966] | 0.6624 [0.6330, 0.6938] |
| ROC-AUC (own rank implementation) | 0.6046 [0.6036, 0.6057] | 0.6632 | 0.6624 |
| PR-AUC (sklearn; floor = base rate) | 0.3205 [0.3188, 0.3217] | 0.3889 [0.3371, 0.4290] | 0.3856 [0.3357, 0.4238] |
| PR-AUC (own step implementation) | 0.3205 [0.3188, 0.3217] | 0.3889 | 0.3856 |
| precision @ 1% coverage | 0.455 [0.447, 0.464] | 0.644 [0.558, 0.728] | 0.656 [0.556, 0.694] |
| lift @ 1% | 1.86 [1.83, 1.90] | 2.63 | 2.68 |
| precision @ 5% coverage | 0.393 [0.389, 0.398] | 0.530 [0.444, 0.567] | 0.511 [0.446, 0.549] |
| lift @ 5% | 1.61 [1.59, 1.63] | 2.17 | 2.09 |
| precision @ 10% coverage | 0.367 [0.363, 0.373] | 0.462 [0.403, 0.507] | 0.461 [0.401, 0.497] |
| lift @ 10% | 1.50 [1.48, 1.53] | 1.89 | 1.89 |
| precision @ 20% coverage | 0.341 [0.337, 0.346] | 0.414 [0.355, 0.450] | 0.405 [0.355, 0.446] |
| lift @ 20% | 1.39 [1.38, 1.41] | 1.69 | 1.66 |
| recall @ 5% | 0.080 [0.080, 0.081] | 0.108 | 0.104 |
| TP @ 5% | 707.8 [701.0, 716.0] | 954 | 919 |
| FP @ 5% | 1092.2 [1084.0, 1099.0] | 846 | 881 |
| FN @ 5% | 8097.2 [8089.0, 8104.0] | 7851 | 7886 |
| TN @ 5% | 26102.8 [26096.0, 26111.0] | 26349 | 26314 |
| plain ratio @ 5% | right 3.9 times in 10 at the flagged cases against 2.4 in 10 by chance | right 5.3 times in 10 at the flagged cases against 2.4 in 10 by chance | right 5.1 times in 10 at the flagged cases against 2.4 in 10 by chance |
| validation max-F1 threshold | 0.182 [0.175, 0.190] | 0.214 | 0.219 |
| precision @ val max-F1 | 0.277 [0.275, 0.280] | 0.336 [0.310, 0.357] | 0.338 [0.313, 0.359] |
| recall @ val max-F1 | 0.822 [0.786, 0.842] | 0.649 [0.499, 0.771] | 0.649 [0.489, 0.780] |
| F1 @ val max-F1 | 0.414 [0.413, 0.415] | 0.443 [0.384, 0.485] | 0.444 [0.386, 0.486] |
| accuracy @ val max-F1 (vs majority) | 0.431 [0.417, 0.453] vs majority 0.755 | 0.600 [0.525, 0.687] vs majority 0.755 | 0.603 [0.522, 0.691] vs majority 0.755 |
| share flagged @ val max-F1 | 0.727 [0.687, 0.750] | 0.473 [0.323, 0.631] | 0.470 [0.309, 0.636] |
| lift @ val max-F1 | 1.13 [1.12, 1.14] | 1.37 | 1.38 |
| TP @ val max-F1 | 7234.8 [6921.0, 7417.0] | 5717 | 5715 |
| FP @ val max-F1 | 18922.6 [17800.0, 19584.0] | 11313 | 11202 |
| FN @ val max-F1 | 1570.2 [1388.0, 1884.0] | 3088 | 3090 |
| TN @ val max-F1 | 8272.4 [7611.0, 9395.0] | 15882 | 15993 |
| bar p = 0.70: status | UNREACHABLE on validation (0 of 5 seeds reach) | reached on validation | reached on validation |
| bar 0.70: test precision | — | 0.728 [0.595, 0.971] | 0.664 [0.571, 0.845] |
| bar 0.70: test recall | — | 0.007 [0.002, 0.011] | 0.010 [0.004, 0.016] |
| bar 0.70: share flagged | — | 0.002 | 0.004 |
| bar 0.70: lift | — | 2.98 | 2.72 |
| bar 0.70: accuracy (vs majority) | — | 0.756 vs majority 0.755 | 0.757 vs majority 0.755 |
| bar 0.70: TP | — | 59 | 91 |
| bar 0.70: FP | — | 22 | 46 |
| bar 0.70: FN | — | 8746 | 8714 |
| bar 0.70: TN | — | 27173 | 27149 |
| bar 0.70: max validation precision | 0.579 [0.547, 0.638] | — | — |
| bar p = 0.80: status | UNREACHABLE on validation (0 of 5 seeds reach) | UNREACHABLE on validation | UNREACHABLE on validation |
| bar 0.80: max validation precision | 0.579 [0.547, 0.638] | 0.750 | 0.733 |
| bar p = 0.85: status | UNREACHABLE on validation (0 of 5 seeds reach) | UNREACHABLE on validation | UNREACHABLE on validation |
| bar 0.85: max validation precision | 0.579 [0.547, 0.638] | 0.750 | 0.733 |
| bar p = 0.90: status | UNREACHABLE on validation (0 of 5 seeds reach) | UNREACHABLE on validation | UNREACHABLE on validation |
| bar 0.90: max validation precision | 0.579 [0.547, 0.638] | 0.750 | 0.733 |
| **Phase 15 class** | WATCHLIST | WATCHLIST | WATCHLIST |
| REACHABLE | NO, CEILING | NO, CEILING | NO, CEILING |
| operating bar | None | None | None |
| lift at the operating bar | n/a | n/a | n/a |
| Stage B | TUNABLE | TUNABLE | TUNABLE |
| per-snapshot precision @ 5%: min | 0.243 | 0.300 | 0.315 |
| per-snapshot precision @ 5%: median | 0.396 | 0.455 | 0.475 |
| per-snapshot precision @ 5%: max | 0.487 | 0.640 | 0.620 |
| worst snapshot | 2025-12-08 (0.243) | 2025-12-08 (0.300) | 2025-12-08 (0.315) |

Other arms in the same column (headline metric `cov.5%.precision`):

| column | arm | cov.5%.precision | class | label |
|---|---|---|---|---|
| CLEAN | consolidated blend (w_neural 0.02) | 0.461 [0.397, 0.510] | WATCHLIST | clean |

- **INCUMBENT** (top 5%): right 3.9 times in 10 at the flagged cases against 2.4 in 10 by chance.
- **BEST PUBLISHED** (top 5%): right 5.3 times in 10 at the flagged cases against 2.4 in 10 by chance.
- **CLEAN** (top 5%): right 5.1 times in 10 at the flagged cases against 2.4 in 10 by chance.

## Fill distribution

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| INCUMBENT | headline | neural none_h0, per seed (Phase 15 fill incumbent) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s{s}/preds_{f}.npz` (3f2b5c8, da18a91) |
| INCUMBENT | also | boundary bw3, per seed (UC2 arm) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s{s}_lossrps_bw3/preds_{f}.npz` (6fe2244) |
| INCUMBENT | also | lgbm22_id, per seed (UC2b arm) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase7_preds/v8_fill_rate_lgbm22_id_s{s}_{f}.npz`; `ml/artifacts/phase7_preds/RECAL_v8_fill_rate_lgbm22_id_s{s}_{f}.npz` (untracked artifact (no stamp)) |
| BEST PUBLISHED | headline | Phase 19 fill blend (w_neural 0.5) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/phase19/bundles/fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase18/preds/v8_fill_rate_p18_fwd_load_s{s}_{f}.npz` (11f0c93, untracked artifact (no stamp)) |
| CLEAN | headline | consolidated blend (w_neural 0.02) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_fill_rate_p22_sc_ack_L4_s{s}_{f}.npz` (2cee433, untracked artifact (no stamp)) |
| CLEAN | also | clean neural + season + cadence, per seed (ships as the distribution) | 5-seed mean [min, max] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz` (2cee433) |
| CLEAN | also | clean Phase 19 fill blend (w_neural 0.49) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence/preds_{f}.npz`; `ml/artifacts/phase22/preds/v8clean_fill_rate_p22_fwdload_s{s}_{f}.npz` (2cee433, untracked artifact (no stamp)) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| exact CRPS (lower = better) | 0.13876 [0.13866, 0.13882] | 0.13537 [0.11540, 0.15391] | 0.13444 [0.11526, 0.15244] |
| P(fill = 1) AUC (sklearn) | 0.6204 [0.6200, 0.6211] | 0.6639 [0.6327, 0.6971] | 0.6688 [0.6429, 0.6951] |
| P(fill = 1) AUC (own rank implementation) | 0.6204 [0.6200, 0.6211] | 0.6639 | 0.6688 |
| ECE-22 RAW | 0.12346 [0.11033, 0.14140] | 0.06040 | 0.03006 |
| ECE-22 RECALIBRATED (stored recalibration only) | 0.05494 [0.05182, 0.05759] | n/a (no stored recalibration) | n/a (no stored recalibration) |
| per-snapshot CRPS: best | 0.07473 | 0.07151 | 0.07269 |
| per-snapshot CRPS: median | 0.14256 | 0.13739 | 0.13690 |
| per-snapshot CRPS: worst | 0.18485 | 0.17872 | 0.17596 |
| worst snapshot | 2025-08-04 (0.1848) | 2025-08-04 (0.1787) | 2025-08-04 (0.176) |

Other arms in the same column (headline metric `crps_exact`):

| column | arm | crps_exact | class | label |
|---|---|---|---|---|
| INCUMBENT | boundary bw3, per seed (UC2 arm) | 0.13882 [0.13873, 0.13892] | — | LEAKY (superseded, Phase 22) |
| INCUMBENT | lgbm22_id, per seed (UC2b arm) | 0.13968 [0.13963, 0.13976] | — | LEAKY (superseded, Phase 22) |
| CLEAN | clean neural + season + cadence, per seed (ships as the distribution) | 0.13746 [0.13698, 0.13773] | — | clean |
| CLEAN | clean Phase 19 fill blend (w_neural 0.49) | 0.13576 [0.11586, 0.15484] | — | clean |

## UC3: capacity strain > 1 in the next 90 days (base 0.405)

| column | role | arm | interval kind | label | status | source (commit) |
|---|---|---|---|---|---|---|
| INCUMBENT | headline | mp h4, per seed (Phase 15's arm) | 5-seed mean [min, max] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/capacity_strain/v8_mp_h4_lr0.00025_s{s}/preds_{f}.npz` (3f2b5c8, da18a91) |
| INCUMBENT | headline | Phase 18 level-aware conformal 80% interval (Phase 22 Stage 4) | as quoted | LEAKY (superseded, Phase 22) | QUOTED | `ml/artifacts/phase22/capacity_published.json` (f2e1f85) |
| BEST PUBLISHED | headline | incumbent 5-seed ensemble (mean quantiles) | point [snapshot-block 95%, 1,000 resamples] | LEAKY (superseded, Phase 22) | RECOMPUTED | `ml/artifacts/bundles/capacity_strain/v8_mp_h4_lr0.00025_s{s}/preds_{f}.npz` (3f2b5c8, da18a91) |
| CLEAN | headline | clean mp h4, per seed | 5-seed mean [min, max] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/capacity_strain/v8clean_mp_h4_lr0.00025_s{s}/preds_{f}.npz` (5f9fc7d) |
| CLEAN | headline | Phase 18 level-aware conformal 80% interval (Phase 22 Stage 4) | as quoted | clean | QUOTED | `ml/artifacts/phase22/capacity_clean.json` (5f9fc7d) |
| CLEAN | also | clean 5-seed ensemble (mean quantiles) | point [snapshot-block 95%, 1,000 resamples] | clean | RECOMPUTED | `ml/artifacts/phase22/bundles/capacity_strain/v8clean_mp_h4_lr0.00025_s{s}/preds_{f}.npz` (5f9fc7d) |

| metric | INCUMBENT (LEAKY (superseded, Phase 22)) | BEST PUBLISHED (LEAKY (superseded, Phase 22)) | CLEAN |
|---|---|---|---|
| base rate (positive class) | 0.405 [0.405, 0.405] | 0.405 [0.305, 0.504] | 0.405 [0.405, 0.405] |
| majority-class predictor | negative (flag nothing) | negative (flag nothing) | negative (flag nothing) |
| majority-class accuracy (the baseline) | 0.595 [0.595, 0.595] | 0.595 | 0.595 [0.595, 0.595] |
| ROC-AUC (sklearn) | 0.7498 [0.7452, 0.7550] | 0.7545 [0.7025, 0.7877] | 0.6938 [0.6817, 0.7166] |
| ROC-AUC (own rank implementation) | 0.7498 [0.7452, 0.7550] | 0.7545 | 0.6938 [0.6817, 0.7166] |
| PR-AUC (sklearn; floor = base rate) | 0.6771 [0.6740, 0.6824] | 0.6840 [0.5504, 0.7627] | 0.5945 [0.5792, 0.6224] |
| PR-AUC (own step implementation) | 0.6771 [0.6740, 0.6824] | 0.6840 | 0.5945 [0.5792, 0.6224] |
| precision @ 1% coverage | 0.904 [0.893, 0.929] | 0.907 [0.880, 0.973] | 0.828 [0.791, 0.862] |
| lift @ 1% | 2.23 [2.20, 2.29] | 2.24 | 2.04 [1.95, 2.13] |
| precision @ 5% coverage | 0.851 [0.832, 0.860] | 0.865 [0.762, 0.896] | 0.735 [0.703, 0.784] |
| lift @ 5% | 2.10 [2.05, 2.12] | 2.13 | 1.81 [1.74, 1.93] |
| precision @ 10% coverage | 0.810 [0.803, 0.818] | 0.815 [0.675, 0.875] | 0.679 [0.652, 0.717] |
| lift @ 10% | 2.00 [1.98, 2.02] | 2.01 | 1.68 [1.61, 1.77] |
| precision @ 20% coverage | 0.743 [0.740, 0.748] | 0.749 [0.584, 0.824] | 0.635 [0.619, 0.664] |
| lift @ 20% | 1.83 [1.83, 1.85] | 1.85 | 1.57 [1.53, 1.64] |
| recall @ 5% | 0.105 [0.103, 0.106] | 0.107 | 0.091 [0.087, 0.097] |
| TP @ 5% | 957.2 [936.0, 968.0] | 973 | 826.6 [791.0, 882.0] |
| FP @ 5% | 167.8 [157.0, 189.0] | 152 | 298.4 [243.0, 334.0] |
| FN @ 5% | 8160.8 [8150.0, 8182.0] | 8145 | 8291.4 [8236.0, 8327.0] |
| TN @ 5% | 13214.2 [13193.0, 13225.0] | 13230 | 13083.6 [13048.0, 13139.0] |
| plain ratio @ 5% | right 8.5 times in 10 at the flagged cases against 4.1 in 10 by chance | right 8.6 times in 10 at the flagged cases against 4.1 in 10 by chance | right 7.3 times in 10 at the flagged cases against 4.1 in 10 by chance |
| validation max-F1 threshold | 0.381 [0.350, 0.403] | 0.390 | 0.389 [0.360, 0.401] |
| precision @ val max-F1 | 0.591 [0.549, 0.627] | 0.604 [0.529, 0.662] | 0.522 [0.474, 0.541] |
| recall @ val max-F1 | 0.695 [0.619, 0.772] | 0.686 [0.457, 0.855] | 0.735 [0.659, 0.893] |
| F1 @ val max-F1 | 0.635 [0.623, 0.648] | 0.642 [0.502, 0.727] | 0.606 [0.594, 0.623] |
| accuracy @ val max-F1 (vs majority) | 0.678 [0.651, 0.698] vs majority 0.595 | 0.690 [0.650, 0.731] vs majority 0.595 | 0.615 [0.554, 0.635] vs majority 0.595 |
| share flagged @ val max-F1 | 0.480 [0.400, 0.569] | 0.461 [0.255, 0.669] | 0.576 [0.494, 0.764] |
| lift @ val max-F1 | 1.46 [1.36, 1.55] | 1.49 | 1.29 [1.17, 1.34] |
| TP @ val max-F1 | 6333.0 [5645.0, 7036.0] | 6257 | 6702.4 [6010.0, 8142.0] |
| FP @ val max-F1 | 4464.2 [3362.0, 5777.0] | 4105 | 6253.2 [5098.0, 9050.0] |
| FN @ val max-F1 | 2785.0 [2082.0, 3473.0] | 2861 | 2415.6 [976.0, 3108.0] |
| TN @ val max-F1 | 8917.8 [7605.0, 10020.0] | 9277 | 7128.8 [4332.0, 8284.0] |
| bar p = 0.70: status | reached on validation | reached on validation | reached on validation |
| bar 0.70: test precision | 0.681 [0.645, 0.705] | 0.678 [0.600, 0.733] | 0.635 [0.562, 0.669] |
| bar 0.70: test recall | 0.515 [0.459, 0.584] | 0.546 [0.310, 0.715] | 0.308 [0.127, 0.699] |
| bar 0.70: share flagged | 0.308 [0.264, 0.367] | 0.326 | 0.207 [0.078, 0.503] |
| bar 0.70: lift | 1.68 [1.59, 1.74] | 1.67 | 1.57 [1.39, 1.65] |
| bar 0.70: accuracy (vs majority) | 0.704 [0.701, 0.708] vs majority 0.595 | 0.711 vs majority 0.595 | 0.638 [0.620, 0.658] vs majority 0.595 |
| bar 0.70: TP | 4696.0 [4184.0, 5325.0] | 4978 | 2809.6 [1162.0, 6371.0] |
| bar 0.70: FP | 2231.0 [1761.0, 2927.0] | 2363 | 1838.0 [595.0, 4956.0] |
| bar 0.70: FN | 4422.0 [3793.0, 4934.0] | 4140 | 6308.4 [2747.0, 7956.0] |
| bar 0.70: TN | 11151.0 [10455.0, 11621.0] | 11019 | 11544.0 [8426.0, 12787.0] |
| bar p = 0.80: status | reached on validation | reached on validation | reached on validation |
| bar 0.80: test precision | 0.762 [0.734, 0.786] | 0.764 [0.681, 0.820] | 0.757 [0.651, 0.809] |
| bar 0.80: test recall | 0.318 [0.246, 0.386] | 0.331 [0.165, 0.451] | 0.116 [0.020, 0.374] |
| bar 0.80: share flagged | 0.170 [0.127, 0.213] | 0.176 | 0.069 [0.010, 0.233] |
| bar 0.80: lift | 1.88 [1.81, 1.94] | 1.89 | 1.87 [1.61, 2.00] |
| bar 0.80: accuracy (vs majority) | 0.682 [0.667, 0.695] vs majority 0.595 | 0.688 vs majority 0.595 | 0.620 [0.601, 0.665] vs majority 0.595 |
| bar 0.80: TP | 2898.0 [2245.0, 3523.0] | 3020 | 1059.2 [181.0, 3414.0] |
| bar 0.80: FP | 931.2 [612.0, 1276.0] | 932 | 488.6 [47.0, 1832.0] |
| bar 0.80: FN | 6220.0 [5595.0, 6873.0] | 6098 | 8058.8 [5704.0, 8937.0] |
| bar 0.80: TN | 12450.8 [12106.0, 12770.0] | 12450 | 12893.4 [11550.0, 13335.0] |
| bar p = 0.85: status | reached on validation | reached on validation | UNREACHABLE on validation (4 of 5 seeds reach) |
| bar 0.85: test precision | 0.810 [0.777, 0.832] | 0.803 [0.738, 0.854] | 0.832 [0.697, 0.964] |
| bar 0.85: test recall | 0.191 [0.140, 0.259] | 0.231 [0.102, 0.324] | 0.071 [0.003, 0.216] |
| bar 0.85: share flagged | 0.096 [0.069, 0.135] | 0.117 | 0.040 [0.001, 0.125] |
| bar 0.85: lift | 2.00 [1.92, 2.05] | 1.98 | 2.05 [1.72, 2.38] |
| bar 0.85: accuracy (vs majority) | 0.654 [0.639, 0.670] vs majority 0.595 | 0.665 vs majority 0.595 | 0.612 [0.596, 0.644] vs majority 0.595 |
| bar 0.85: TP | 1746.0 [1281.0, 2360.0] | 2109 | 644.5 [27.0, 1966.0] |
| bar 0.85: FP | 422.8 [268.0, 676.0] | 518 | 246.2 [1.0, 855.0] |
| bar 0.85: FN | 7372.0 [6758.0, 7837.0] | 7009 | 8473.5 [7152.0, 9091.0] |
| bar 0.85: TN | 12959.2 [12706.0, 13114.0] | 12864 | 13135.8 [12527.0, 13381.0] |
| bar 0.85: max validation precision | — | — | 0.834 [0.834, 0.834] |
| bar p = 0.90: status | reached on validation | reached on validation | UNREACHABLE on validation (3 of 5 seeds reach) |
| bar 0.90: test precision | 0.898 [0.862, 0.958] | 0.912 [0.885, 0.984] | 0.864 [0.747, 1.000] |
| bar 0.90: test recall | 0.047 [0.003, 0.101] | 0.032 [0.011, 0.050] | 0.049 [0.001, 0.123] |
| bar 0.90: share flagged | 0.022 [0.001, 0.048] | 0.014 | 0.026 [0.000, 0.067] |
| bar 0.90: lift | 2.22 [2.13, 2.36] | 2.25 | 2.13 [1.84, 2.47] |
| bar 0.90: accuracy (vs majority) | 0.611 [0.596, 0.629] vs majority 0.595 | 0.606 vs majority 0.595 | 0.608 [0.595, 0.628] vs majority 0.595 |
| bar 0.90: TP | 432.2 [23.0, 925.0] | 292 | 442.7 [8.0, 1124.0] |
| bar 0.90: FP | 61.2 [1.0, 148.0] | 28 | 139.0 [0.0, 381.0] |
| bar 0.90: FN | 8685.8 [8193.0, 9095.0] | 8826 | 8675.3 [7994.0, 9110.0] |
| bar 0.90: TN | 13320.8 [13234.0, 13381.0] | 13354 | 13243.0 [13001.0, 13382.0] |
| bar 0.90: max validation precision | — | — | 0.847 [0.834, 0.860] |
| **Phase 15 class** | ALERT | ALERT | WATCHLIST |
| REACHABLE | PARTIAL | PARTIAL | NO, CEILING |
| operating bar | 0.85 | 0.85 | None |
| lift at the operating bar | 2.00 | 1.98 | n/a |
| Stage B | TUNABLE | TUNABLE | TUNABLE |
| per-snapshot precision @ 5%: min | 0.419 | 0.456 | 0.454 |
| per-snapshot precision @ 5%: median | 0.779 | 0.808 | 0.642 |
| per-snapshot precision @ 5%: max | 0.907 | 0.912 | 0.891 |
| worst snapshot | 2025-12-08 (0.419) | 2025-12-08 (0.456) | 2025-12-08 (0.454) |
| pinball loss q0.1 | 0.04156 [0.04109, 0.04208] | 0.04088 | 0.04541 [0.04475, 0.04633] |
| pinball loss q0.5 | 0.11446 [0.11349, 0.11615] | 0.11274 | 0.12282 [0.12115, 0.12369] |
| pinball loss q0.9 | 0.06230 [0.06135, 0.06403] | 0.06132 | 0.06530 [0.06398, 0.06611] |
| pinball loss, mean | 0.07277 [0.07213, 0.07408] | 0.07165 | 0.07785 [0.07663, 0.07835] |
| raw P10-P90 coverage (all test) | 0.782 [0.776, 0.788] | 0.790 | 0.782 [0.753, 0.798] |
| raw P10-P90 coverage, worst quarter | 0.767 [0.753, 0.782] | 0.785 | 0.756 [0.742, 0.766] |
| raw P10-P90 coverage, mean of quarters | 0.782 [0.777, 0.789] | 0.790 | 0.783 [0.754, 0.798] |
| raw P10-P90 coverage, best quarter | 0.796 [0.792, 0.798] | 0.801 | 0.810 [0.790, 0.816] |
| level-aware conformal 80%: worst quarter | 0.783 (Q) | — | 0.764 (Q) |
| level-aware conformal 80%: mean of quarters | 0.794 (Q) | — | 0.798 (Q) |
| level-aware conformal 80%: best quarter | 0.806 (Q) | — | 0.829 (Q) |
| level-aware conformal: worst quarter | 2025Q1 (Q) | — | 2025Q3 (Q) |

Other arms in the same column (headline metric `cov.5%.precision`):

| column | arm | cov.5%.precision | class | label |
|---|---|---|---|---|
| CLEAN | clean 5-seed ensemble (mean quantiles) | 0.757 [0.639, 0.849] | ALERT | clean |

- **INCUMBENT** (top 5%): right 8.5 times in 10 at the flagged cases against 4.1 in 10 by chance.
- **BEST PUBLISHED** (top 5%): right 8.6 times in 10 at the flagged cases against 4.1 in 10 by chance.
- **CLEAN** (top 5%): right 7.3 times in 10 at the flagged cases against 4.1 in 10 by chance.

## Use cases without a clean model metric

| use case | figure (QUOTED) | status | label | source |
|---|---|---|---|---|
| Shortage head (retired) | 0.224 precision @ 5% coverage at a 3% base (re-weighted; 0.223 subsampled) | **RETIRED (not re-measured)** | LEAKY-UNCONFIRMED | `reports/part2/phase-17.md` §1.1 (caf613c) |
| Predict-the-rescue | 0.823 [0.822, 0.824] precision at recall 0.246, base 0.4535 (LightGBM B1a) | **PENDING PHASE 23B** | LEAKY-UNCONFIRMED | `reports/part2/phase-17.md` §2.1 (caf613c) |
| Shortage simulation | 1.129x [1.121, 1.139] vs the pre-rescue reference; detector precision ceiling 0.28 | **PENDING PHASE 23B** | LEAKY-UNCONFIRMED | `reports/part2/phase-13.md` S1; phase-15.md §1 (UC5a 0.284) (ea1b8fe) |
| Delivery schedule (MILP) | no model metric: blocked. Needs holding, ordering, freight and shortage costs; 0 of 4 exist | **blocked** | n/a | `results/observation2.md` §3.6 (41a7f9a) |
| Supplier allocation | no model metric: blocked. 57 of 120 part-plants have no allowed split; on the 63 feasible it ties the status quo | **blocked** | n/a | `results/observation2.md` §3.6 (41a7f9a) |
| Transfer recommendation | no model metric: blocked. Its confidence is NOT TUNABLE (lift 0.03); "always yes" beats it on F1 | **blocked** | n/a | `results/observation2.md` §3.6 (41a7f9a) |
