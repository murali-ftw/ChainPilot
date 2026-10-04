# Phase 21 Stage 3 — arrival at snapshots: does the 5-D group block add anything?

**Audience:** whoever decides whether group statistics enter the arrival product.
**Measured on:** v8 seed 1001, TEST (2025, 9 snapshots, 36,000 rows, 18,534 uncensored), RAW. Seeds 7 / 17 / 27 / 37 / 47.
**Instruments:** `ml/baselines/phase21_proxy.py` (fits at `6248fae`, `cbb1167` for `_nl`); `ml/eval/phase21_score.py snap`
(`676ae0c`) → `ml/artifacts/phase21/score_snap_v8.json`. k = 300 (validation).
**Anchor:** BASE refitted for seed 7 reproduces the stored `phase7_preds/v8_arrival_week_b5flat_reg_s7` **bit-exactly**
(val and test); the BASE band 0.7053 [0.7048, 0.7057] equals the lateness spec's recorded b5flat band.

## Arms (5-seed bands [min, mean, max])

| arm | lateness AUC | A3 (days) | vs BASE |
|---|---|---|---|
| **BASE** (LightGBM flat) | **0.7053** [0.7048, 0.7057] | **13.25** [13.21, 13.27] | — |
| BASE + L4 (no month) | 0.7038 [0.7030, 0.7042] | 13.31 [13.26, 13.38] | **worse** (lateness); A3 undetermined |
| BASE + L4, cross-channel shuffle | 0.7036 [0.7031, 0.7044] | 13.33 | worse |
| **BASE + L5 (the user's 5-D idea)** | 0.7042 [0.7036, 0.7052] | 13.29 [13.21, 13.33] | undetermined (both) |
| BASE + L5, permuted month | 0.7039 [0.7034, 0.7043] | 13.28 | worse (lateness) |
| BASE + L5, cross-channel shuffle | 0.7031 [0.7025, 0.7034] | 13.35 | worse |
| group-only LightGBM (L5 block, no BASE columns) | 0.6873 [0.6868, 0.6877] | 13.64 | **worse** (both) |
| BASE + Phase 19 season + cadence | 0.7156 [0.7152, 0.7158] | 12.99 | better (both) |
| BASE + season + cadence + L5 | 0.7145 [0.7140, 0.7149] | 13.03 | vs season + cadence: **worse** (lateness); A3 undetermined |
| **standalone group rule** (no ML; 1 predictor) | **0.6800** | 13.54 | block interval vs the BASE ensemble: −0.026 [−0.030, −0.022] lateness, −0.31 [−0.47, −0.15] d A3: **worse** |

For reference: the incumbent neural h4 band is 0.7090 [0.7065, 0.7130] (Phase 18); the BASE 5-seed ensemble 0.7056.

## Gate verdicts

| family | verdict | why |
|---|---|---|
| **L4** | **FAIL** | disjointly **worse** than BASE on lateness AUC (−0.0015) |
| **L5** | **FAIL** | no metric disjointly better (−0.0011 lateness, −0.04 d A3; bands overlap) |
| control (c), L5 vs L4 (the month) | undetermined | +0.0004 lateness AUC |
| L5 on top of season + cadence | **worse** than season + cadence (−0.0011, disjoint) | |
| group-only | worse than BASE on both | the pure form of the idea |

**The group block adds nothing to arrival at snapshots, with or without the month.** Its gain importance in the BASE +
L5 fit is 25% of the total (L1 9%, L3 4%, L5 4%, L4 4%, L2 4%; BASE columns 75%), so LightGBM uses it: it trades off against the flat
channel features without improving the ranking.

**Which level resolves for the rows that gain:** with the validation-chosen k = 300, **every** test row resolves to L2
(supplier). The 48.1% of observed rows that L5 improves (absolute error vs the BASE ensemble) are therefore all L2
rows, and the rows it worsens are too. There is no thin-cell population for the finer levels to help.

## No-leak diagnostic (deviation 199, no verdict)

Three panel columns that every BASE carries (`lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13`) are built
by the generator from each line's **eventual** lead, written on the line's **order** week. At a snapshot they carry
future leads of lines ordered up to t0's week. With them removed:

| arm | lateness AUC | A3 |
|---|---|---|
| BASE_nl | 0.6987 [0.6983, 0.6990] (vs BASE −0.0066, disjoint) | 13.27 |
| BASE_nl + L4 | 0.6961 [0.6954, 0.6969]: **worse** than BASE_nl | 13.49: worse |
| BASE_nl + L5 | 0.6967 [0.6961, 0.6977]: **worse** than BASE_nl | 13.42: worse |

At snapshots the leak is worth 0.007 lateness AUC to the LightGBM BASE. It does **not** explain the group block's
failure: against an honest BASE the block is still disjointly worse.

## Decision level (Phase 15 UC1: late vs contracted lead, censoring resolved; base rate 0.730)

| arm | class | REACHABLE (bar, lift) | P @ 1% / 5% / 10% / 20% | per-snapshot P @ 5% min / median / max | accuracy (majority 0.730) |
|---|---|---|---|---|---|
| BASE, per seed | WATCHLIST | PARTIAL (0.85, 1.15) | 0.938 / 0.894 / 0.855 / 0.818 | 0.830 / 0.889 / 0.937 | 0.737 |
| BASE + L4, per seed | WATCHLIST | PARTIAL (0.80, 1.09) | 0.944 / 0.876 / 0.848 / 0.818 | 0.833 / 0.876 / 0.928 | 0.738 |
| BASE + L5, per seed | WATCHLIST | PARTIAL (0.85, 1.16) | 0.947 / 0.878 / 0.849 / 0.818 | 0.834 / 0.883 / 0.924 | 0.737 |
| BASE + L5, 5-seed ensemble | WATCHLIST | YES (0.85, 1.17) | 0.949 / 0.878 / 0.850 / 0.819 | 0.827 / 0.872 / 0.929 | 0.738 |
| BASE + season + cadence + L5, per seed | WATCHLIST | YES (0.85, 1.17) | 0.955 / 0.904 / 0.872 / 0.835 | 0.857 / 0.904 / 0.950 | 0.734 |
| standalone group rule | WATCHLIST | PARTIAL (0.80, 1.08); Stage B WEAKLY TUNABLE | 0.827 / 0.800 / 0.798 / 0.793 | 0.744 / 0.801 / 0.842 | 0.731 |
| **incumbent neural ensemble P(late)** (Phase 20 reference) | WATCHLIST | YES (0.90, 1.25) | **0.991 / 0.969 / 0.952 / 0.912** | 0.939 / 0.969 / 0.985 | 0.731 |

At 5% coverage the group block **lowers** the top of the list (0.894 → 0.878). No arrival arm here changes class, and
the incumbent's P(late) remains the best ranked list by a wide margin.
