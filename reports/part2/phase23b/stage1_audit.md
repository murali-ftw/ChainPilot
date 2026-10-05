# Phase 23B Stage 1 — leak audit of predict-the-rescue and the shortage simulation

**Measured on:** v8 seed 1001 (`db/gen_v8/seed_1001`). **CPU only**, concurrency 1.
**Instrument:** `ml/eval/phase23b_audit.py` (`ee62f46`) → `ml/artifacts/phase23b/stage1_audit.json`, 33 s. It imports Phase 22's
scan code **unchanged** (`ml/eval/phase22_leakscan.py`: `Sources`, `rebuild`, `reproduce`, `poison`) and Phase 22's list
(`ml/data/clean_panel.LEAKING`). The scan reads the generator's `_sim.npz` for the audit only; no model reads it.

**The nine leaking columns** (taken from `reports/part2/phase22/stage1_leak.md` and `clean_panel.LEAKING`): `fill_rate`,
`fill_rate_last4`, `fill_rate_last13`, `fill_rate_last52`, `lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13`,
`ack_gap_ratio`, `load_ratio`.

## a. What each consumer reads (from code and stored configs)

| consumer | source of the list | inputs |
|---|---|---|
| **B1a** LightGBM (Phase 17) | `ml/artifacts/phase17/b1a_lgbm.json` (stored by `phase17_b1.run_lgbm`, stamp `0a19487`); built by `phase7_fit.World.part_plant_features` | **37 features:** per part-plant **mean and min of all 15 live panel columns** at row t0 (30), `channels_in_part_plant`, 5 graph counts (`suppliers_per_part_plant`, `sole_source_part_plant`, `alternate_sources_part_plant`, `channels_per_part`, `suppliers_per_part`), `week_offset` |
| **B1b** neural (Phase 17) | bundle configs `bundles/rescue_week/v8_mp_h1_lr0.000125_s{7,17}` + `phase5_heads.device_inputs` | the 15 panel columns + 10 missing indicators over a 52-week window ending at row t0, part-plant pooled; one-hot week offset |
| **Shortage simulation** (Phases 12 B2 / 13 / 14 / 15) | `ml/sim/montecarlo.read_heads`, called by `ml/opt/order_policy.py@9e2d59d make_draw` | two **neural heads**, recalibrated: arrival `bundles/arrival_week/v8_lite_h4_lr0.00025_s7` (13-cell) and fill `bundles/fill_rate/v8_none_h0_lr0.000125_s{fill_seed}` (22-cell, fill seeds 7–47). Each reads the 15 panel columns + 10 indicators. Also CSVs (lead pmf, policy parameters), `inventory_position_weekly` (opening position, as of t0: the sanctioned simulation path) and `part_demand_weekly` (the plan, as of t0). **No capacity head** |

## b. Intersection with the nine leaking columns, and the future-poison test

**By name:** B1a reads **all nine** (18 of its 37 features: the mean and min of each). B1b and both simulation heads read all
nine.

**By future poison, on B1a's own features.** The scan's rebuild reproduces the stored panel on **100.000%** of values for all
15 columns. At 6 snapshot weeks (2019-01-14 … 2025-12-08), every source row visible after the row's own week (H_week) was
poisoned and B1a's part-plant features at row t were recomputed:

| B1a feature (from a LEAKING column) | share of part-plants that move (max / mean over weeks) |
|---|---|
| `load_ratio_mean` / `_min` | **1.000 / 1.000** and 1.000 / 0.998 |
| `fill_rate_last52_mean`, `_last13_mean`, `lead_time_ratio_mean`, `lead_time_actual_days_mean`, `fill_rate_last4_mean`, `ack_gap_ratio_mean`, `fill_rate_mean` | 0.955–0.966 / 0.876–0.905 |
| the fill-family `_min` features | 0.844–0.902 / 0.727–0.825 |
| `otd_rate_last13_mean`, `lead_time_ratio_min`, `lead_time_actual_days_min` | 0.668–0.723 / 0.489–0.658 |
| `ack_gap_ratio_min`, `otd_rate_last13_min` | 0.321–0.487 / 0.254–0.273 |
| the 12 features from the other six panel columns | **0.000** (none move) |
| the 7 non-panel features | not panel-derived |

**Every one of the 18 features derived from a leaking column leaks under poison, and none of the other 19 does.** Name and
poison agree exactly.

**Gate G1, constructed failing case (pre-registered (a)):** the NL feature list (19 features) passes by name and by poison; the
NL list with `lead_time_actual_days_mean` re-added (a deliberately leaked arm) is **FLAGGED** by name and by poison. Valid.

## c. The simulation's inputs

| input | file(s) | read leaking columns? | Phase 22 restatement of that model | **status** |
|---|---|---|---|---|
| arrival distribution | `bundles/arrival_week/v8_lite_h4_lr0.00025_s7` | yes, all nine (panel) | lateness AUC −0.0117 [−0.0169, −0.0057] on clean inputs | **LEAKY** |
| fill distribution | `bundles/fill_rate/v8_none_h0_lr0.000125_s{7,17,27,37,47}` | yes, all nine (panel) | P(fill = 1) AUC −0.021, UC2b @ 5% −0.058 on clean inputs | **LEAKY** |
| capacity | none (the simulation reads no capacity head) | — | — | — |

## d. Plainly

- **B1a is affected, by all nine columns** (18 features). This is P1.
- **B1b is affected, by all nine.**
- **The simulation is affected through both of its model inputs** (arrival s7 and fill s7–s47), by all nine.
