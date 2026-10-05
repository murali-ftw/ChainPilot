# Phase 22 Stage 1 — the leak audit, clean inputs, and the honest restatement

**Audience:** whoever quotes any arrival, fill or capacity number from Phases 1–21, and whoever decides what ships.
**Measured on:** v8 seed 1001 (clean world `v8clean`); v8w1002 for the clean cache and replication inputs. TEST, RAW,
5 seeds.
**Instruments (commits):**
- `ml/eval/phase22_leakscan.py` (`c6d96d2`) → `leakscan_v8.json`
- `ml/eval/phase22_leak_rows.py` (`95a8dde`) → `leak_rows_v8.json`
- `ml/data/clean_panel.py` + `ml/tests/test_phase22_clean_panel.py` (`29f2054`) → `cache/v8clean/`, `clean_panel_tests_v8.json`
- `ml/eval/phase22_clean_check.py` (`e025965`) → `clean_check_v8.json`
- `ml/baselines/phase22_proxy.py` (`625d0cc`)
- `ml/train/phase22_train.py` (`10cb9f9`; handshake `980c10e`)
- `ml/eval/phase22_restate.py` → `restate_arrival_fill_capacity.json`

The two `leakscan` and `leak_rows` instruments read the generator's `_sim.npz` **for the audit only**; no model reads it.

## a. Deviation 199, verified independently

The generator's weekly-store block writes each line's **eventual** lead into its channel's row for the week the line was
**ordered**: `leadw[pch, vw_ord] = pl_`, forward-filled → `lead_time_actual_days`; `/ contracted` → `lead_time_ratio`;
`leadw <= contracted`, forward-filled, 13-week roll → `otd_rate_last13`.

On 8 randomly drawn lines (the last line ordered in its channel-week, receipt ≥ 4 weeks later), the stored value in the
order week equals that line's eventual lead **8 of 8 times**, written 6–20 weeks before the receipt existed:

| line | order week | stored `lead_time_actual_days` that week | the line's eventual lead | receipt visible |
|---|---|---|---|---|
| POL000940417 | 2025-01-06 | 137.87 | 137.87 | 2025-05-19 (19 weeks later) |
| POL000901510 | 2024-08-19 | 82.12 | 82.12 | 2024-11-11 (12 weeks) |
| POL001016829 | 2025-09-29 | 135.33 | 135.33 | 2026-02-16 (20 weeks) |
| POL000461009 | 2020-04-13 | 36.62 | 36.62 | 2020-05-25 (6 weeks) |

(Four more in `leak_rows_v8.json`.)

**Which snapshot rows are affected.** For every fit-window snapshot, the store was rebuilt with every source row visible
after t0's week poisoned. A row is affected if its channel's row-t0 value moves in any leaking column.

| label rows affected | train | val | test |
|---|---|---|---|
| arrival_week | 94.5% | 93.9% | 95.7% |
| fill_rate | 94.5% | 93.9% | 95.7% |
| capacity_strain | 94.1% | 93.0% | 95.7% |

Per snapshot, the share of channels touched has median 99.98% (minimum 54.7%).

## b. Systematic scan of every column a stored model reads

**Method (pre-registration D1).** The 15 panel columns the stored LightGBM (`phase7_fit.World`, row t0) and neural models
(`temporal_share`, a window ending at row t0) read were **rebuilt from the generator's per-line arrays and the emitted
timestamps, following the generator line for line**. The rebuild reproduces the stored panel on **100.000% of values and
100.000% of the null pattern for all 15 columns** (8.6 M channel-weeks). Every source row visible after the horizon was
then replaced with noise at 12 weeks spread over 2019–2025.

**Valid:** the constructed failing case (the three known columns) fires, and the constructed passing case
(`qty_ordered` under H_week) does not.

| column | table | changes under poison, H_week (row's own week) | share of channels changed | changes under H_date (strict snapshot date) | read by |
|---|---|---|---|---|---|
| qty_ordered | channel_performance_weekly | **N** | 0 | Y (0.13) | every LightGBM-flat and neural model |
| qty_received | " | **N** | 0 | Y (0.11) | " |
| is_active_week | " | **N** | 0 | N | " |
| **fill_rate** | " | **Y** | 0.537 | Y (0.60) | " |
| **fill_rate_last4** | " | **Y** | 0.538 | Y | " |
| **fill_rate_last13** | " | **Y** | 0.555 | Y | " |
| **fill_rate_last52** | " | **Y** | 0.596 | Y | " |
| **lead_time_actual_days** | " | **Y** | 0.540 | Y | " (known, deviation 199) |
| **lead_time_ratio** | " | **Y** | 0.540 | Y | " (known) |
| **otd_rate_last13** | " | **Y** | 0.299 | Y | " (known) |
| **ack_gap_ratio** | " | **Y** | 0.537 | Y | " |
| **load_ratio** | " | **Y** | **0.814** | Y (0.99) | " |
| reporting_lag_days | " | **N** | 0 | Y (0.13) | " |
| active_weeks_in_52 | " | **N** | 0 | N | " |
| revision_count | " | **N** | 0 | Y (0.04) | " |

**Mechanisms:**
- **fill family and `ack_gap_ratio`:** the generator's `fillw` is each ordered line's eventual delivered fraction,
  bucketed on its **order** week (`_num / _den` on `cw`), and `ack_gap_ratio = 1 − fill`.
- **`load_ratio`:** the supplier's **whole-month** ordered quantity over declared capacity (`util_obs_hist[MONTHKEY]`),
  copied to every week of the month. It contains orders not yet placed at t0, and the capacity label is built from the
  same monthly ordered totals over the forward months (deviation 211).

**Derived feature stores**, each with its own as-of assertion run against a constructed future row:

| store (reads) | falsifier | result |
|---|---|---|
| fwd_load (part_demand_weekly +2 d bound, po_lines, grn_lines) | `fwd_load.falsify` | fires on all three sources |
| fwd_season (network means of fwd_load) | inherits fwd_load's assertions | — |
| cadence (po_lines) | `cadence.falsify` | fires |
| pulse (grn_lines first receipts) | `pulse.falsify` | fires |
| stock_asof (inventory_transactions) | `stock_asof.falsify` | fires |
| grpstats (Phase 21: po_lines, grn_lines, acknowledgements, revisions) | Phase 21 poison / self-exclusion | PASS, offenders flagged |
| ASN, po_line_schedules, plan tables | read by **no** stored model (`loader.py` reads ASN / schedules only in its A1 / A2 validation checks; training imports only `read_df`) | — |

## c. As-of-safe replacements and the clean world

`ml/data/clean_panel.py` recomputes the nine columns from the CSVs only:
- outcomes enter in the week their receipt (or zero-close signal) **and** their order are both visible;
- load = the month's ordered quantity visible so far ÷ the declared capacity, which is recorded 5–44 days before each
  month.

Tests (`clean_panel_tests_v8.json`):
- **future poison:** 0 changes at 3 weeks; the order-keyed offender changes 1,502–8,877 channels;
- **self-exclusion:** 0 changes on 3 × 400 snapshot rows; dropping the order-visibility requirement admits 1,461–1,547.

The poison test caught one defect of mine before any model read the clean cache: a zero-close rule that consulted a
line's eventual received quantity (fixed, `29f2054`).

Equality with the stored columns on rows where no leak applies (40 sampled weeks, all channels):

| column | equal on no-leak rows | mean \|Δ\| on leak rows |
|---|---|---|
| fill_rate / ack_gap_ratio | 97.2% | 0.188 |
| lead_time_actual_days (tolerance 0.5 d: the CSV lead is whole days) | 95.1% | 25.0 days |
| lead_time_ratio | 95.0% | 0.56 |
| load_ratio (tolerance = stored rounding) | **100.0%** | 0.43 |
| fill_rate_last4 / 13 / 52, otd_rate_last13 (rolled) | 85% / 68% / 35% / 61% | 0.17 / 0.13 / 0.07 / 0.21 |

- **Point-in-time columns agree where the leak cannot reach.** The residual 3–5% are receipts arriving out of order:
  the stored value is the last-*ordered* line's lead, the clean one the last-*received* line's.
- **Rolled columns cannot agree exactly:** their 4–52-week windows include weeks where order-keying and receipt-keying
  place values differently.

`masked_view` (the feature-view switch) zeroes the nine columns and marks them missing for any stored-model read.

## d–f. Restatement

`ml/eval/phase22_restate.py` (`e79327f`) → `restate_arrival_fill_capacity.json`.
- **Neural clean:** the three incumbent configurations retrained on `v8clean`, five seeds each, MPS, concurrency 1,
  bundles under `ml/artifacts/phase22/bundles/{task}/v8clean_*`.
- **The audit (1b) says every stored model reads leaking columns:** the panel feeds all three heads, and fill reads the
  fill family while capacity reads `load_ratio`. So all three were retrained, and none was skipped.
- **Handshake (Stage −1):** INSIDE the stored band at all 3 epochs (|Δ| ≤ 2.0e-5), and its constructed failing case
  (+0.05) reports OUTSIDE.

Published (leaky) and clean are in **separate columns**. Δ = clean − published: 5-seed ensembles, snapshot-block 95%
interval, better = positive.

### Arrival (UC1 base 0.730)

| metric | published LightGBM | clean LightGBM | published neural | **clean neural** | neural Δ (block) |
|---|---|---|---|---|---|
| lateness AUC | 0.7053 [0.7048, 0.7057] | 0.6991 [0.6986, 0.6995] | 0.7090 [0.7064, 0.7130] | **0.6951** [0.6817, 0.7041] | **−0.0117** [−0.0169, −0.0057] worse |
| C-index | 0.6489 | 0.6421 | 0.6744 | 0.6621 | — |
| A3 (days) | 13.25 | 13.28 | 13.06 | 13.15 | +0.03 d [−0.16, +0.27]: undetermined |
| UC1 precision @ 1 / 5 / 10 / 20% | 0.938 / 0.894 / 0.855 / 0.818 | 0.931 / 0.868 / 0.840 / 0.810 | 0.986 / 0.968 / 0.947 / 0.908 | 0.969 / 0.940 / 0.916 / 0.886 | @ 5% (ensemble P(late)): **−0.021** [−0.034, −0.009] worse |
| Phase 15 class | WATCHLIST (PARTIAL) | WATCHLIST (PARTIAL) | WATCHLIST (YES, lift 1.25) | **WATCHLIST** (YES) | no change |

LightGBM Δ (block):
- lateness −0.0063, worse;
- UC1 @ 5% −0.026, worse;
- `nl` (columns dropped, no replacement) is worse still, at −0.0117.

**One clean neural seed is weak** (lateness 0.682), which widens the clean band.

### Fill (UC2b base 0.245)

| metric | published LightGBM | clean LightGBM | published neural | **clean neural** | neural Δ (block) |
|---|---|---|---|---|---|
| exact CRPS | 0.1397 | 0.1407 | 0.13876 | **0.1397** | +0.0011 [−0.0006, +0.0030]: undetermined |
| P(fill = 1) AUC | 0.6049 | 0.5879 | 0.6204 | **0.6007** | **−0.021** [−0.044, −0.0002] worse |
| UC2b precision @ 5% | 0.384 | 0.368 | 0.451 | **0.401** | **−0.058** [−0.078, −0.014] worse |
| UC2 precision @ 5% | 0.864 | 0.856 | 0.874 | 0.863 | — |
| Phase 15 class (UC2 / UC2b) | WATCHLIST / WATCHLIST | WATCHLIST / WATCHLIST | WATCHLIST / WATCHLIST | **WATCHLIST / WATCHLIST** | no change |

### Capacity (UC3 base 0.405)

| metric | published LightGBM | clean LightGBM | published neural | **clean neural** | neural Δ (block) |
|---|---|---|---|---|---|
| precision @ 1 / 5 / 10% | 0.780 / 0.692 / 0.662 | 0.617 / 0.585 / 0.562 | 0.904 / 0.851 / 0.810 | **0.828 / 0.735 / 0.679** | @ 5%: **−0.108** [−0.212, −0.023] worse |
| recall @ p 0.70 / 0.80 / 0.85 | 0.148 / 0.022 / UNREACHABLE | UNREACHABLE | 0.515 / 0.318 / 0.192 | 0.308 / 0.116 / **UNREACHABLE** | — |
| worst-quarter precision @ 5% (mean) | 0.497 | 0.472 | 0.608 | 0.553 | — |
| Phase 15 class | WATCHLIST | WATCHLIST | **ALERT** (PARTIAL at 0.85, lift 2.0) | **WATCHLIST** (NO, CEILING) | **ALERT → WATCHLIST** |

### Classes after the leak is removed

**UC1, UC2 and UC2b keep their class (WATCHLIST). UC3 capacity changes: ALERT → WATCHLIST.** The order-time flag (Stage 2) is
ALERT with the τ-week row and WATCHLIST strictly as-of.

### Week-t0 convention diagnostic (`lag1`, BASE_clean read at row t0 − 1; no verdict)

| | arrival lateness AUC | fill CRPS | capacity precision @ 5% |
|---|---|---|---|
| BASE_clean (row t0) | 0.6991 | 0.1407 | 0.585 |
| BASE_clean, row t0 − 1 | 0.6940 | 0.1406 | 0.569 |

At snapshots the row's own week is worth about 0.005 lateness AUC and 0.016 capacity precision to LightGBM. At order
time it is worth far more (Stage 2: 0.676 vs 0.615).

### Blends re-fitted on validation with clean arms

| recipe | published weights → test | clean weights → test |
|---|---|---|
| Phase 19 arrival recipe (incumbent neural ensemble + LightGBM + fwd_load) | w_neural 0.49 → lateness 0.7305, A3 12.41 d, UC1 @ 5% 0.942 | **w_neural 0.52** → lateness **0.7217**, A3 **12.57 d**, UC1 @ 5% 0.919 |
| Phase 21 hybrid | incumbent 0.00 / Phase 19 neural 0.55 / LightGBM + season + cadence + L5 0.45 → 0.729, 12.41 d | incumbent **0.60** / LightGBM + season + cadence + L5 0.40 → lateness **0.7164**, A3 **12.70 d** (the Phase 19 neural arm was not retrained clean, so it is absent) |

The published Phase 19 blend (Phase 19 neural rf arm + LightGBM fwd_load, 0.731, 12.34 d) cannot be restated exactly:
its neural half tied the incumbent and was not retrained (deviation 214). The recipe is restated with the incumbent in
its place, for both columns.

### Per-seed minutes (clean neural, MPS, concurrency 1; LightGBM CPU fits ran beside some cells: wall-clock only)

| use case | s7 | s17 | s27 | s37 | s47 | s/epoch |
|---|---|---|---|---|---|---|
| arrival lite h4 | 20.6 (59 ep, best 50) | 19.2 (56, 47) | 17.3 (51, 42) | 18.9 (56, 47) | 20.6 (61, 52) | 19.6–20.3 |
| capacity mp h4 | 7.6 (29, 20) | 7.3 (28, 19) | 13.4 (53, 44) | 7.6 (29, 20) | 10.5 (41, 32) | 14.7–14.8 |
| fill h0 | 15.2 (61, 52) | 14.5 (58, 49) | 15.9 (64, 55) | 13.8 (55, 46) | 13.6 (54, 45) | 14.3–14.4 |
