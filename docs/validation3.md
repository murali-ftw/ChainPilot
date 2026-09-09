# External dataset validation — run 3 (three-way delta against runs 1 and 2)

**Verdict: not usable.**

The gates are nearly clear — 8 failures against run 2's 24 — but the joint-structure probes say
the marginals were fitted to the bands rather than produced by a process: late and short
arrivals are independent (φ = −0.003), 4 of 5 labels are uncorrelated with their own most
predictive feature (|r| ≤ 0.06), and 90.1% of shortage events sit on part-plants that have no
sourcing channel at all, so they cannot have an upstream cause.

| | run 3 | run 2 | run 1 |
|---|---|---|---|
| **passed / failed / skipped** | **167 / 8 / 5** | 152 / 24 / 13 | 88 / 74 / 20 |
| total checks | 194 | 202 | 195 |
| verdict | **not usable** | not usable | not usable |

`SKIP` is never a pass. 5 checks did not run; they are enumerated in §5.

## Instrument — unchanged

| | |
|---|---|
| validator SHA-256 recorded in `docs/validation2.md` | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| validator SHA-256 **before** run 3 | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| validator SHA-256 **after** run 3 | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |

**Identical to run 2's post-addition SHA. The validator was not edited: not a band, not a
tolerance, not a special case, not a skipped table.** Runs 2 and 3 used bit-identical
instruments, so the run-2 → run-3 delta is attributable entirely to the dataset.

The step-3 joint-structure probes were run as a **separate read-only script**
(`docs/validator_run3_joint_probes.py`), not as an edit to the validator. They add findings;
they cannot remove a failure.

### One incident to report — `docs/external_dataset_validation.md` was overwritten

`validator.py` writes `../docs/external_dataset_validation.md` **by default** on every run
(`REPORT_PATH`, line 94; suppressed only by `--no-report`). The control run was launched
without that flag and the file — run 1's report — was overwritten before the behaviour was
noticed. There is no git repository here, no Time Machine snapshot, and the Trash is empty,
so the original bytes are not recoverable.

**No measurement was lost.** Run 1's full stdout, `docs/validator_run_external.txt` (606
lines), is intact and is the source of every run-1 figure in the trend table below; run 2's
report also carries run-1 columns. Every subsequent run 3 command used `--no-report`.
`docs/validation2.md` is untouched. The clobbered file has been replaced with a notice
recording what happened rather than a lookalike reconstruction.

## Control — did not move

| run | instrument | result |
|---|---|---|
| run 1 | `196325…` frozen | 167 passed / 2 failed / 4 skipped |
| run 2 | `c49fc5…` augmented | 172 passed / 2 failed / 4 skipped |
| **run 3** | **`c49fc5…` augmented** | **172 passed / 2 failed / 4 skipped** |

Against `/Users/muralik/Documents/Programs/HADES/db/csv_full_seed1`. `diff` against run 2's
saved control transcript is clean over all 605 shared lines except two lines that are not
measurements: the spec path (the validator now sits in
`chainpilot_rane_15yr_synthetic_dataset_v3_FINAL/` rather than `db/`) and the trailing
"report written" line. **Every check line is byte-identical.** The two standing failures are
unchanged: `PO lines arriving late` 12.8% and its `[2019-2025]` twin 13.7%, against a 15–25%
band — the pre-existing `csv_full_seed1` finding from run 1.

The instrument did not move. The delta below is the dataset.

---

## 2. Trend table — every check that has failed in any run

83 checks have failed at least once across the three runs. Still-failing first, then the rest
alphabetically. Run-3 values in bold.

| check | run 1 | run 2 | run 3 | band | state r1>r2>r3 |
|---|---|---|---|---|---|
| `PO lines with fill < 1.0` | 72.7% | 20.0% | **15.9%** | 8%-15% | **FAIL** → **FAIL** → **FAIL** |
| `PO lines with fill < 1.0 [2019-2025]` | 72.2% | 20.1% | **16.6%** | 8%-15% | **FAIL** → **FAIL** → **FAIL** |
| `capacity unobservable` | 11.5% | 2.4% | **2.4%** | >= 70% | **FAIL** → **FAIL** → **FAIL** |
| `capacity unobservable [2019-2025]` | 11.5% | 2.2% | **2.2%** | >= 70% | **FAIL** → **FAIL** → **FAIL** |
| `structurally clean tables` | 37/49 | 40/49 | **48/49** | 49/49 | **FAIL** → **FAIL** → **FAIL** |
| `supplier-months constrained` | 88.5% | 97.6% | **97.6%** | 20%-30% | **FAIL** → **FAIL** → **FAIL** |
| `supplier-months constrained [2019-2025]` | 88.5% | 97.8% | **97.8%** | 20%-30% | **FAIL** → **FAIL** → **FAIL** |
| `training_labels` | 12,000 rows | 6,000 rows | **6,000 rows** | clean | ok → **FAIL** → **FAIL** |
| `PO lines arriving late` | 82.3% | 19.8% | **19.8%** | 15%-25% | **FAIL** → ok → ok |
| `PO lines arriving late [2019-2025]` | 82.7% | 20.0% | **20.0%** | 15%-25% | **FAIL** → ok → ok |
| `all spec tables present` | 47/49 | 49/49 | **49/49** | 49/49 | **FAIL** → ok → ok |
| `alternate_sources` | 144 rows | 120 rows | **120 rows** | clean | ok → **FAIL** → ok |
| `arrival_week: entity_type matches the spec` | 1,919/2,419 wrong | 0/1,200 wrong | **0/1,200 wrong** | 0 | **FAIL** → ok → ok |
| `arrival_week: something is censored` | 0.0% | 13.6% | **13.6%** | 1%-60% | **FAIL** → ok → ok |
| `asn: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.753 | **sd/\|mean\|=0.753** | >= 0.1 | **FAIL** → ok → ok |
| `calendar` | 37,548 rows | 37,555 rows | **37,555 rows** | clean | **FAIL** → ok → ok |
| `capacity_strain: entity_type matches the spec` | 1,916/2,431 wrong | 0/1,200 wrong | **0/1,200 wrong** | 0 | **FAIL** → ok → ok |
| `capacity_strain: something is censored` | 0.0% | 3.6% | **3.6%** | 1%-60% | **FAIL** → ok → ok |
| `capacity_strain: target is not a uniform random draw` | _absent_ | D=0.0160 p=0.926 | **D=0.3146 p=1.29e-100** | p < 0.01 | -- → **FAIL** → ok |
| `channel store conserves ordered units` | 138,615,579 vs 4,114,197 | 2,878,869 vs 1,944,910 | **1,944,910 vs 1,944,910** | exact | **FAIL** → **FAIL** → ok |
| `channel store conserves received units` | 110,778,603 vs 2,992,098 | 2,740,051 vs 1,870,498 | **1,870,498 vs 1,870,498** | exact | **FAIL** → **FAIL** → ok |
| `channel_performance_weekly` | 55,000 rows | 43,800 rows | **43,800 rows** | clean | **FAIL** → **FAIL** → ok |
| `demand_drift: entity_type matches the spec` | 1,902/2,403 wrong | 0/1,200 wrong | **0/1,200 wrong** | 0 | **FAIL** → ok → ok |
| `demand_drift: something is censored` | 0.0% | 4.6% | **4.6%** | 1%-60% | **FAIL** → ok → ok |
| `demand_drift: target is not a uniform random draw` | _absent_ | D=0.0277 p=0.339 | **D=0.4366 p=2.08e-191** | p < 0.01 | -- → **FAIL** → ok |
| `derived store buckets on visible week` | neither | neither | **max(event_week, recorded_week)** | max(event,recorded) | **FAIL** → **FAIL** → ok |
| `entity_id resolves for entity_type=channel` | 2,252/2,292 unresolved | 0/120 unresolved | **0/120 unresolved** | 0 | **FAIL** → ok → ok |
| `entity_id resolves for entity_type=po_line` | 87/2,261 unresolved | 0/2,097 unresolved | **0/2,097 unresolved** | 0 | **FAIL** → ok → ok |
| `entity_id resolves for entity_type=supplier` | 2,244/2,248 unresolved | _absent_ | **_absent_** | 0 | **FAIL** → -- → -- |
| `expedite_events: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.708 | **sd/\|mean\|=0.708** | >= 0.1 | **FAIL** → ok → ok |
| `expedite_events: two timestamps differ` | 0.0% | 83.4% | **83.4%** | >= 2% | **FAIL** → ok → ok |
| `expedites per year` | 93.7 | 131.6 | **131.6** | 100-150 | **FAIL** → ok → ok |
| `fill-rate mass at exactly 0` | 31.8% | 2.0% | **2.0%** | 1%-3% | **FAIL** → ok → ok |
| `fill-rate mass at exactly 0 [2019-2025]` | 32.1% | 1.9% | **1.9%** | 1%-3% | **FAIL** → ok → ok |
| `fill-rate mass at exactly 1.0` | 27.3% | 80.0% | **84.1%** | >= 60% | **FAIL** → ok → ok |
| `fill-rate mass at exactly 1.0 [2019-2025]` | 27.8% | 79.9% | **83.4%** | >= 60% | **FAIL** → ok → ok |
| `fill_rate: entity_type matches the spec` | 1,892/2,413 wrong | 0/1,200 wrong | **0/1,200 wrong** | 0 | **FAIL** → ok → ok |
| `fill_rate: something is censored` | 0.0% | 4.7% | **4.7%** | 1%-60% | **FAIL** → ok → ok |
| `goods_receipts: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.694 | **sd/\|mean\|=0.694** | >= 0.1 | **FAIL** → ok → ok |
| `grn_lines: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.684 | **sd/\|mean\|=0.684** | >= 0.1 | **FAIL** → ok → ok |
| `inventory_position_weekly` | 50,000 rows | 10,000 rows | **10,000 rows** | clean | **FAIL** → ok → ok |
| `inventory_snapshots` | 60,000 rows | 15,340 rows | **15,340 rows** | clean | **FAIL** → ok → ok |
| `inventory_snapshots: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.689 | **sd/\|mean\|=0.689** | >= 0.1 | **FAIL** → ok → ok |
| `inventory_snapshots: two timestamps differ` | 0.0% | 39.8% | **39.8%** | >= 2% | **FAIL** → ok → ok |
| `inventory_transactions: recorded_ts >= event` | 12,201 | 0 | **0** | 0 | **FAIL** → ok → ok |
| `lead-time distribution is right-skewed` | +0.05 | +0.88 | **+0.88** | >= 0.5 | **FAIL** → ok → ok |
| `lead-time distribution is right-skewed [2019-2025]` | +0.05 | +0.93 | **+0.93** | >= 0.5 | **FAIL** → ok → ok |
| `line stops per year` | 14.7 | 21.7 | **21.7** | 15-25 | **FAIL** → ok → ok |
| `line_stop_events: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.705 | **sd/\|mean\|=0.705** | >= 0.1 | **FAIL** → ok → ok |
| `line_stop_events: two timestamps differ` | 0.0% | 83.6% | **83.6%** | >= 2% | **FAIL** → ok → ok |
| `month-of-year seasonality in demand` | 1.03x peak/trough | 1.46x peak/trough | **1.46x peak/trough** | > 1.10 | **FAIL** → ok → ok |
| `month-of-year seasonality in demand [2019-2025]` | 1.03x peak/trough | 1.43x peak/trough | **1.43x peak/trough** | > 1.10 | **FAIL** → ok → ok |
| `part_demand_weekly` | 50,000 rows | 7,800 rows | **7,800 rows** | clean | **FAIL** → ok → ok |
| `parts` | 250 rows | 500 rows | **500 rows** | clean | **FAIL** → **FAIL** → ok |
| `po_line_revisions` | 6,000 rows | 5,121 rows | **5,121 rows** | clean | ok → **FAIL** → ok |
| `po_line_revisions: recorded_ts >= event` | 2,562 | 0 | **0** | 0 | **FAIL** → ok → ok |
| `po_lines: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.746 | **sd/\|mean\|=0.746** | >= 0.1 | **FAIL** → ok → ok |
| `po_lines: two timestamps differ` | 0.0% | 60.5% | **60.5%** | >= 2% | **FAIL** → ok → ok |
| `production_plan` | _skip_ | 64,837 rows | **64,837 rows** | clean | skip → **FAIL** → ok |
| `purchase_orders: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.754 | **sd/\|mean\|=0.754** | >= 0.1 | **FAIL** → ok → ok |
| `purchase_orders: two timestamps differ` | 0.0% | 60.2% | **60.2%** | >= 2% | **FAIL** → ok → ok |
| `quality_inspections: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.746 | **sd/\|mean\|=0.746** | >= 0.1 | **FAIL** → ok → ok |
| `revealed_capacity_monthly` | 4,000 rows | 7,560 rows | **10,651 rows** | clean | **FAIL** → ok → ok |
| `shortage events per year` | 120.5 | 235.6 | **235.6** | 180-250 | **FAIL** → ok → ok |
| `shortage_events: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.716 | **sd/\|mean\|=0.716** | >= 0.1 | **FAIL** → ok → ok |
| `shortage_events: two timestamps differ` | 0.0% | 82.6% | **82.6%** | >= 2% | **FAIL** → ok → ok |
| `shortage_qty: entity_type matches the spec` | 1,882/2,334 wrong | 0/1,200 wrong | **0/1,200 wrong** | 0 | **FAIL** → ok → ok |
| `shortage_qty: something is censored` | 0.0% | 3.8% | **3.8%** | 1%-60% | **FAIL** → ok → ok |
| `shortage_qty: target is not a uniform random draw` | _absent_ | D=0.0295 p=0.262 | **D=0.9905 p=0** | p < 0.01 | -- → **FAIL** → ok |
| `supplier store conserves ordered units` | 21,468,423 vs 4,114,197 | 2,878,869 vs 1,944,910 | **1,944,910 vs 1,944,910** | exact | **FAIL** → **FAIL** → ok |
| `supplier store conserves received units` | 17,303,481 vs 2,992,098 | 2,740,051 vs 1,870,498 | **1,870,498 vs 1,870,498** | exact | **FAIL** → **FAIL** → ok |
| `supplier_acknowledgements` | 15,600 rows | 9,214 rows | **9,214 rows** | clean | ok → **FAIL** → ok |
| `supplier_acknowledgements: recorded_ts >= event` | 4,227 | 0 | **0** | 0 | **FAIL** → ok → ok |
| `supplier_allocation: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.720 | **sd/\|mean\|=0.720** | >= 0.1 | **FAIL** → ok → ok |
| `supplier_capacity` | 15,930 rows | 10,639 rows | **10,639 rows** | clean | **FAIL** → ok → ok |
| `supplier_capacity: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.527 | **sd/\|mean\|=0.527** | >= 0.1 | **FAIL** → ok → ok |
| `supplier_financials` | 450 rows | 900 rows | **900 rows** | clean | **FAIL** → ok → ok |
| `supplier_performance_weekly` | 17,899 rows | 24,820 rows | **_absent_** | clean | **FAIL** → **FAIL** → -- |
| `supplier_upstream` | 180 rows | 300 rows | **300 rows** | clean | ok → **FAIL** → ok |
| `weekly store is a panel` | 0.0% | 100.0% | **100.0%** | >= 90% | **FAIL** → ok → ok |
| `weekly store is a panel [2019-2025]` | 0.0% | 100.0% | **100.0%** | >= 90% | **FAIL** → ok → ok |
| `zero-order channel-weeks` | 0.0% | 80.2% | **86.0%** | 60%-99% | **FAIL** → ok → ok |
| `zero-order channel-weeks [2019-2025]` | 0.0% | 80.2% | **86.0%** | 60%-99% | **FAIL** → ok → ok |

---

## 3. Newly fixed since run 2 — 15 checks moved FAIL → pass

No check moved SKIP → pass. None of these passes because a table became absent.

### 3a. Conservation — the run-2 headline defect, gone (5 checks)

| check | run 1 | run 2 | run 3 |
|---|---|---|---|
| `channel store conserves ordered units` | 138,615,579 vs 4,114,197 (+3269.2%) | 2,878,869 vs 1,944,910 (+48.0%) | **1,944,910 vs 1,944,910 (exact)** |
| `channel store conserves received units` | 110,778,603 vs 2,992,098 (+3602.4%) | 2,740,051 vs 1,870,498 (+46.5%) | **1,870,498 vs 1,870,498 (exact)** |
| `supplier store conserves ordered units` | 21,468,423 vs 4,114,197 (+421.8%) | 2,878,869 vs 1,944,910 (+48.0%) | **1,944,910 vs 1,944,910 (exact)** |
| `supplier store conserves received units` | 17,303,481 vs 2,992,098 (+478.3%) | 2,740,051 vs 1,870,498 (+46.5%) | **1,870,498 vs 1,870,498 (exact)** |
| `derived store buckets on visible week` | neither | neither | **max(event_week, recorded_week)** |

This is the real fix of the run. Both weekly stores now reproduce their source quantities to
the unit, and the bucketing probe resolves to the correct rule rather than "neither". Run 2's
reading — that the weekly store was synthesised on its own grid rather than aggregated from
the transactions — no longer holds. The derived layer is now an aggregation.

### 3b. Structural conformance — 8 of 9 Tier 1 failures cleared

| table | run 1 | run 2 | run 3 |
|---|---|---|---|
| `channel_performance_weekly` | 9 missing cols + 10,210 dup PKs | 9 missing cols | **clean** |
| `supplier_performance_weekly` | 10 missing cols + 4 type violations | 9 missing cols | **clean** |
| `parts` | `'180.0'` in INTEGER ×151 | ×236 | **clean** |
| `alternate_sources` | ok | `qualification_status` enum | **clean** |
| `po_line_revisions` | ok | `initiated_by` enum | **clean** |
| `production_plan` | _skip: no file_ | `plan_type` enum | **clean** |
| `supplier_acknowledgements` | ok | `ack_status` enum | **clean** |
| `supplier_upstream` | ok | `confidence` float-for-enum | **clean** |

`structurally clean tables` 40/49 → **48/49**. The retired-column defect is gone: both weekly
stores now carry `is_active_week`, `fill_rate_last4/13/52`, `otd_rate_last13`,
`active_weeks_in_52`, `reporting_lag_days`, `weeks_since_last_activity` and
`weeks_since_last_receipt`, and `staleness_days` is no longer present on them. All five enum
substitutions run 2 found are corrected to the spec's declared vocabularies.

### 3c. The KS gate — all five labels now pass (3 checks moved)

| task | run 2 | run 3 |
|---|---|---|
| `capacity_strain` | D=0.0160 **p=0.926** FAIL | **D=0.3146 p=1.29e-100** pass |
| `demand_drift` | D=0.0277 **p=0.339** FAIL | **D=0.4366 p=2.08e-191** pass |
| `shortage_qty` | D=0.0295 **p=0.262** FAIL | **D=0.9905 p=0** pass |

Run 2's three `Uniform(a,b)` labels are gone. §6 probe 6 reads these against what they mean.

---

## 4. Still failing after two attempts — 8 checks

A defect that survived two regenerations is structural. The column that matters most here is
**"moved r2 → r3"**: a value that has not budged means the fix did not touch the mechanism.

| check | run 2 | run 3 | band | moved r2 → r3? |
|---|---|---|---|---|
| `supplier-months constrained` | 97.6% | 97.6% | 20%–30% | **no — identical** |
| `supplier-months constrained [2019-2025]` | 97.8% | 97.8% | 20%–30% | **no — identical** |
| `capacity unobservable` | 2.4% | 2.4% | ≥ 70% | **no — identical** |
| `capacity unobservable [2019-2025]` | 2.2% | 2.2% | ≥ 70% | **no — identical** |
| `training_labels` (Tier 1) | 363/6,000 bad `censor_time` | 363/6,000 bad `censor_time` | clean | **no — same 363 rows** |
| `PO lines with fill < 1.0` | 20.0% | 15.9% | 8%–15% | yes, −4.1 pts (still 0.9 over) |
| `PO lines with fill < 1.0 [2019-2025]` | 20.1% | 16.6% | 8%–15% | yes, −3.5 pts (still 1.6 over) |
| `structurally clean tables` | 40/49 | 48/49 | 49/49 | yes, +8 tables |

**Five of the eight did not move at all.** Four of those five are one measurement stated twice
(`unobservable = 1 − constrained`) across two windows, and it is the same measurement that
moved *away* from band between runs 1 and 2 and has now been left exactly where it was. Under
`delivered = min(K, ordered)` an unconstrained month is the only kind that tells you nothing
about true capacity; a world where 97.6% of supplier-months look constrained has effectively
no unconstrained baseline, which is UC2's entire premise inverted. Two regenerations have not
touched it.

`training_labels` fails Tier 1 on exactly the same defect and exactly the same row count as
run 2: `censor_time` holds `'1.0'`, `'2.0'`, `'10.0'` … in a column declared INTEGER, on 363
of 6,000 rows (the 5,637 blank rows are fine). Identical count across two regenerations.

`PO lines with fill < 1.0` is the one standing failure that is genuinely converging: 72.7% →
20.0% → 15.9%, now 0.9 points above the band. It is also the one most plausibly a tuning
matter rather than a structural one.

---

## 5. Regressions, oscillations, and skips

### Regressions — none

No check that passed in run 2 fails in run 3. No check that passed in run 2 is skipped in run
3. Run 3's 8 failures are a strict subset of run 2's 24.

### Oscillations — 5 checks have flipped state more than once

All five currently read "pass". Flagged anyway: a value that goes pass → fail → pass across
three regenerations is being nudged, not derived.

| check | run 1 | run 2 | run 3 | what flipped |
|---|---|---|---|---|
| `alternate_sources` | pass | **FAIL** | pass | `qualification_status` gained `conditional`/`development`, then lost them |
| `po_line_revisions` | pass | **FAIL** | pass | `initiated_by` became `planner`/`supplier_portal`, then reverted |
| `supplier_acknowledgements` | pass | **FAIL** | pass | `ack_status` lost `rejected`/`date_change`, then regained them |
| `supplier_upstream` | pass | **FAIL** | pass | `confidence` became a float, then reverted to the 3-level enum |
| `production_plan` | _skip: no file_ | **FAIL** | pass | table appeared, `plan_type` wrong, then corrected |

Four of these five are the *same* class of defect — a categorical column that acquired a
vocabulary outside the spec in run 2 and lost it again in run 3. A column set that can lose
and regain its declared enum across two regenerations is being written from a value list that
is edited by hand, not from a mechanism that knows what the column means.

### Skipped in run 3 — 5 checks, none of them a pass

- 1 × `training_labels: cleared Tier 1` — the gating marker for the one table that failed
  Tier 1. Its Tier 2/3 results below are advisory, not evidence.
- 2 × `two timestamps differ` for `inventory_position_weekly` and `plan_drift_features` —
  derived stores with `recorded_ts` but no event timestamp to compare against.
- 2 × `entity_id resolves` for `part_plant` and `product_plant` — composite entities with no
  single master PK; resolution is not attempted rather than guessed.

All five were skipped in run 2 for the same reasons. Run 2's other 8 skips were the Tier 1
gating markers for the 8 tables that have since been fixed.

---

## 6. Joint structure

Six probes run as a separate read-only script (`docs/validator_run3_joint_probes.py`) against
the same PO-line population the gates use — `fill_population`'s rules mirrored exactly:
buyer-cancelled lines dropped, lines promised inside the last 90 days of the span dropped,
`fill = min(1, Σ qty_accepted / qty_ordered)`, short at `fill < 0.9999`, late at
`first receipt date > original_promise_date`. Every statistic is reported whether it passes or
fails.

### 6.1 Late and short co-occur — **no association**

On 6,605 normal-regime PO lines with both a first receipt and a promise date:

| | fill = 1.0 | fill < 1.0 | P(short) |
|---|---|---|---|
| **on time** | 4,548 | 751 | 14.17% |
| **late** | 1,124 | 182 | **13.94%** |

**φ = −0.0027. Lift = 0.98.** A late line is very slightly *less* likely to be short than an
on-time one. Robustness — every way of asking gives the same answer:

| formulation | statistic | n |
|---|---|---|
| line level, normal regime | φ = **−0.0027** | 6,605 |
| line level, all regimes | φ = **−0.0108** | 8,966 |
| continuous: days late vs shortfall magnitude | r = **−0.0128**, ρ = −0.0041 | 8,966 |
| channel level: late-rate vs short-rate | r = **−0.0047**, ρ = −0.0386 | 120 channels |
| supplier-month level: late-rate vs short-rate | r = **+0.0199**, ρ = +0.0302 | 916 supplier-months |

A supplier under strain misses on both dimensions at once; here the two dimensions are
statistically independent at every level of aggregation. **The two marginals were drawn
independently.** Both of them individually sit in or near their published bands — lateness
19.8% inside 15–25%, shortfall 15.9% just outside 8–15% — which is exactly the failure mode:
each distribution is right on its own, and nothing links them.

A corroborating detail from the same tables: **`current_promise_date` equals
`original_promise_date` on all 9,214 PO lines.** No promise date was ever revised — yet
`po_line_revisions` holds 5,121 revisions touching 3,422 distinct lines, 1,250 of them coded
`supplier_delay` and 1,259 `capacity`. The revision history records reschedules that never
reached the line they reschedule. (`reason_code` carries only `logistics` / `demand_change` /
`capacity` / `supplier_delay` — there is no `buyer_cancellation` value at all, so the gate's
cancellation filter removes nothing.)

### 6.2 Constraint predicts shortfall — **flag is decoration**

Joining `revealed_capacity_monthly.constrained_month_flag` to fill computed from
`po_lines`/`grn_lines` at (supplier, part, month):

| | supplier-part-months | mean fill |
|---|---|---|
| flagged **constrained** | 814 | **0.9503** |
| flagged unconstrained | 5,568 | **0.9706** |
| difference | | **2.03 points** |

**r(flag, fill) = −0.0584.** Constrained months are 2 points worse on fill. That is a
detectable difference in the right direction, but it is not what "constrained" means — a month
in which capacity actually binds should show materially depressed fill, not a 2-point nudge.

The stronger evidence is that the flag does not agree with the behaviour it names. The
validator computes constraint from the transactions themselves (received < ordered, or mean
lead-time ratio > 1.15). On the 6,426 supplier-part-months where both are defined:

| | behavioural: unconstrained | behavioural: constrained |
|---|---|---|
| **declared: unconstrained** | 177 | 5,435 |
| **declared: constrained** | 11 | 803 |

- declared constrained share: **12.67%**
- behavioural constrained share: **97.07%**
- agreement rate: **15.25%**
- **φ(declared, behavioural) = +0.0356** — essentially independent

Across the whole table the declared flag reads **2,663 / 10,651 = 25.00%**, which is dead
centre of the validator's published 20–30% band for this concept — while the behaviour the
band is about reads 97.6% and fails. The flag has been set to the number the band wants; the
world underneath it has not changed since run 2 (§4: 97.6% → 97.6%, unmoved).

`capacity_basis` is **the single constant string `estimated` on all 10,651 rows.** It carries
no information at all. `evidence_strength` takes exactly two values, `0.15` and `0.75`,
perfectly determined by `constrained_month_flag` — a deterministic recode of the flag, not
independent evidence. **`capacity_basis` is decoration**, as is `evidence_strength`.

### 6.3 Shortages follow their cause — **1.4% do**

For each of the 3,580 `shortage_events` rows, looking back 8 weeks over every sourcing channel
feeding that part-plant for a fill miss or a late arrival:

| | count | share |
|---|---|---|
| shortage events **on a part-plant with no sourcing channel at all** | **3,227** | **90.1%** |
| events with a channel, and an identifiable upstream cause | 50 | 1.4% of all events |
| events with a channel, no cause in the window | 303 | 8.5% of all events |
| **share of all shortage events with an identifiable upstream cause** | | **1.4%** |
| share *among the 353 events that have any channel* | | **14.2%** |
| **placebo: same computation with shortage dates shuffled** | | **15.3%** |

Two findings, and the first is the larger one.

**90.1% of shortage events are structurally incapable of having a cause.** `shortage_events`
covers 20 distinct part-plants; `sourcing_channels` covers 120; **the overlap is 2**. The
shortage table and the procurement table are describing almost disjoint sets of parts. A
shortage on a part nobody buys through any channel cannot be traced to a receipt, ever.

**Among the events that could have a cause, the antecedent rate is at chance.** 14.2% real
versus **15.3% under shuffled dates** — the placebo is *higher*. There is no temporal
relationship between upstream misses and downstream shortages. Shortages were inserted, not
caused.

### 6.4 Staleness and volatility co-move — **wrong sign**

Across the 120 channels, correlating a staleness measure against the variance of that
channel's realised lead time:

| staleness measure | vs lead-time variance | Pearson | Spearman |
|---|---|---|---|
| mean `weeks_since_last_activity` | variance | **−0.1519** | −0.1862 |
| mean `weeks_since_last_receipt` | variance | **−0.1338** | −0.1730 |
| max `weeks_since_last_activity` | variance | **−0.1402** | −0.1954 |
| zero-order week share | variance | **−0.0906** | −0.0685 |

Expected clearly positive — intermittent channels are less predictable. Measured **negative on
all four formulations**: in this dataset the more intermittent a channel is, the *more*
regular its lead time. That is the opposite of the mechanism, not merely its absence.

### 6.5 Labels track their own features — **4 of 5 detached**

For each task, the label against the single most obviously predictive feature in the
corresponding store, joined as-of the snapshot date (latest store row at or before
`snapshot_date`), on uncensored rows.

| task | feature | Pearson | Spearman | n | reading |
|---|---|---|---|---|---|
| `fill_rate` | `channel_performance_weekly.fill_rate_last13` | **+0.0230** | +0.0409 | 1,144 | **detached** |
| `arrival_week` | `channel_performance_weekly.otd_rate_last13` | **−0.0210** | −0.0230 | 1,037 | **detached** |
| `capacity_strain` | `channel_performance_weekly.load_ratio` | **+0.0557** | +0.0646 | 1,157 | **detached** |
| `shortage_qty` | `part_demand_weekly.gross_requirement_p50` | **−0.0545** | −0.0660 | 1,154 | **detached** |
| `demand_drift` | `plan_drift_features.drift_ratio` | **+0.2678** | **+0.4144** | 826 | real signal |

**This is the most important probe of the six, and 4 of the 5 labels fail it.** A fill-rate
label that correlates +0.023 with the channel's own trailing 13-week fill rate is not a
function of that channel's history. The features and the target were generated independently:
run 1's defect, wearing a much better distribution.

`demand_drift` is the exception and it is a genuine one — ρ = +0.41 against `drift_ratio` is
real, learnable signal. It is also the one label whose feature store (`plan_drift_features`)
carries the quantity the label is named after.

Two coverage facts found while joining, both structural:

- `shortage_qty` labels span **199 part-plants**; `inventory_position_weekly` and
  `shortage_events` each cover **20**, with **3** in common. The natural feature store for
  this task (`days_of_supply`) joins to 16 of 1,200 label rows. `part_demand_weekly` covers
  199/199, which is why it is used above.
- **1,143 of 1,154 uncensored `shortage_qty` labels are exactly 0.0**, and the column takes 13
  distinct values in total across 1,200 rows.

### 6.6 KS against uniform — all five pass, but read 6.5 first

Run 2's addition, re-run unchanged on all five targets:

| task | n uncensored | observed support | mean | sd | distinct | KS `D` | `p` | verdict |
|---|---|---|---|---|---|---|---|---|
| `fill_rate` | 1,144 | [0.0000, 1.0000] | 0.9614 | 0.1456 | 159 | 0.8281 | 0 | **pass** |
| `arrival_week` | 1,037 | [1.0000, 11.0000] | 5.9238 | 3.1782 | 11 | 0.0945 | 1.57e-08 | **pass** |
| `capacity_strain` | 1,157 | [0.0000, 1.5000] | 0.6274 | 0.5442 | 660 | 0.3146 | 1.29e-100 | **pass** |
| `demand_drift` | 1,145 | [0.6759, 1.7000] | 1.0337 | 0.2353 | 716 | 0.4366 | 2.08e-191 | **pass** |
| `shortage_qty` | 1,154 | [0.0000, 651.0000] | 2.3995 | 30.8896 | 12 | 0.9905 | 0 | **pass** |

Run 2's three `Uniform(a,b)` draws are genuinely gone — `capacity_strain` and `demand_drift`
now have real shape, and their moments no longer match midpoint and range/√12.

**But the KS test is a marginal test, and passing it is not evidence of signal.**
`shortage_qty` passes with D = 0.9905 because it is a point mass at zero: 1,143 of 1,154
uncensored values are 0.0, over 12 distinct values. A degenerate constant would pass this gate
too. That label satisfies every marginal gate in the validator — varies, censored,
entity_type, non-uniform — and still correlates −0.05 with the demand that is supposed to
drive it (§6.5). This is precisely the gap the joint probes exist to close.

### Verdict on generated vs fitted — **fitted to the bands**

The marginals are in far better shape than run 2 and the derived layer now genuinely
aggregates its sources (§3a — that part is real work, not tuning). But every probe that asks
whether two quantities move *together* comes back at or near zero, and two come back with the
wrong sign:

| probe | expected | measured | reading |
|---|---|---|---|
| late ↔ short | clearly positive | **φ = −0.003** | independent |
| constrained ↔ fill | materially lower | 2.0 pts, **φ(declared,behavioural) = +0.036** | flag is decoration |
| shortage ← upstream cause | most events | **1.4%**, placebo 15.3% > actual 14.2% | inserted, not caused |
| staleness ↔ lead-time variance | positive | **−0.15** | wrong sign |
| labels ↔ own features | clearly positive | **+0.02, −0.02, +0.06, −0.05**, +0.27 | 4 of 5 detached |

**The dataset looks fitted to the bands, not generated.** Each published band is satisfied by
adjusting the distribution it measures; the causal links between those distributions were
never built. A model trained on this learns nothing on four of the five tasks, because on
those four the features and the target are statistically independent. `demand_drift` is the
single exception and looks genuinely generated.

---

## 7. The three structural questions

### Q1. Are the labels real? — **No. Four of five are detached from their features.**

The KS gate now passes on all five (§6.6): no label is a uniform draw any more. That was run
2's finding and it has been fixed.

The replacement finding is worse in one respect, because it is invisible to every marginal
gate. Correlation of each label against its own most predictive feature (§6.5):

| task | r | verdict |
|---|---|---|
| `demand_drift` | **+0.268** (ρ +0.414) | real |
| `capacity_strain` | +0.056 | detached |
| `fill_rate` | +0.023 | detached |
| `arrival_week` | −0.021 | detached |
| `shortage_qty` | −0.055 | detached |

Entity binding remains fully correct: 0/1,200 wrong `entity_type` on all five tasks, 0/2,097
unresolved `po_line` ids, 0/120 unresolved `channel` ids. Censoring is present on all five
(3.6%–13.6%). The labels are correctly *shaped* and correctly *attached* — they are just not
functions of the features they sit beside.

### Q2. Do the derived stores conserve exactly? — **Yes. Exactly.**

| store | quantity | source | emitted | error | run 2 error | run 1 error |
|---|---|---|---|---|---|---|
| `channel_performance_weekly` | ordered | 1,944,910 | **1,944,910** | **0** | +48.02% | +3269.20% |
| `channel_performance_weekly` | received | 1,870,498 | **1,870,498** | **0** | +46.49% | +3602.37% |
| `supplier_performance_weekly` | ordered | 1,944,910 | **1,944,910** | **0** | +48.02% | +421.81% |
| `supplier_performance_weekly` | received | 1,870,498 | **1,870,498** | **0** | +46.49% | +478.31% |

Exact equality on all four, and `derived store buckets on visible week` now resolves to
`max(event_week, recorded_week)` — the correct rule — where both prior runs read "neither".
The weekly stores are aggregations of the transaction tables. **This is the one unambiguous
structural fix of run 3.**

### Q3. Is the weekly store a contiguous panel? — **Yes.**

| measure | run 1 | run 2 | run 3 | band |
|---|---|---|---|---|
| zero-order channel-week share | 0.0% | 80.2% | **86.0%** | 60–99% |
| contiguity (unbroken 7-day run) | 0.0% | 100.0% | **100.0%** | ≥ 90% |
| duplicate PKs | 10,210 / 55,000 | 0 / 43,800 | **0 / 43,800** | 0 |
| rows / channels | 55,000 / 320 = 172 | 43,800 / 120 = 365 | **43,800 / 120 = 365** | — |

120 channels × 365 weeks = 43,800 rows exactly. Every channel has a complete gap-free weekly
series, and the store now carries the full spec column set including `is_active_week`,
`active_weeks_in_52`, `reporting_lag_days`, `weeks_since_last_activity` and
`weeks_since_last_receipt`. Staleness reads P50 35d / P90 105d / max 406d.

The panel is real. What §6.4 adds is that the staleness columns on it do not co-move with
anything: intermittency and lead-time volatility correlate −0.15, the wrong way round.

---

## 8. Full tier tables

`run 2` and `run 1` columns show the same check's value in those runs. `_(new check)_` marks a
check with no counterpart in that run. A `SKIP` is shown as a skip and is never a pass.

### Tier 1


**TABLES**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| -- | extra tables (informational) | `0` | `0` | `0` | n/a |
| ok | all spec tables present | `49/49` | `49/49` | `47/49` | 49/49 |
| **FAIL** | structurally clean tables | `48/49` | `40/49` | `37/49` | 49/49 |

**PER-TABLE**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | alternate_sources | `120 rows` | `120 rows` | `144 rows` | clean |
| ok | asn | `17,218 rows` | `17,218 rows` | `7,000 rows` | clean |
| ok | bom | `1,319 rows` | `1,319 rows` | `700 rows` | clean |
| ok | business_units | `5 rows` | `5 rows` | `5 rows` | clean |
| ok | calendar | `37,555 rows` | `37,555 rows` | `37,548 rows` | clean |
| ok | channel_performance_weekly | `43,800 rows` | `43,800 rows` | `55,000 rows` | clean |
| ok | customers | `20 rows` | `20 rows` | `20 rows` | clean |
| ok | dataset_coverage | `48 rows` | `48 rows` | `46 rows` | clean |
| ok | expedite_events | `2,005 rows` | `2,005 rows` | `1,400 rows` | clean |
| ok | goods_receipts | `17,218 rows` | `17,218 rows` | `23,000 rows` | clean |
| ok | grn_lines | `17,218 rows` | `17,218 rows` | `23,000 rows` | clean |
| ok | inventory_position_weekly | `10,000 rows` | `10,000 rows` | `50,000 rows` | clean |
| ok | inventory_snapshots | `15,340 rows` | `15,340 rows` | `60,000 rows` | clean |
| ok | inventory_transactions | `1,534 rows` | `1,534 rows` | `45,000 rows` | clean |
| ok | line_stop_events | `330 rows` | `330 rows` | `220 rows` | clean |
| ok | logistics_lanes | `190 rows` | `190 rows` | `250 rows` | clean |
| ok | model_outputs | `624 rows` | `624 rows` | `900 rows` | clean |
| ok | part_costs | `120 rows` | `120 rows` | `700 rows` | clean |
| ok | part_demand_weekly | `7,800 rows` | `7,800 rows` | `50,000 rows` | clean |
| ok | part_plant | `1,680 rows` | `1,680 rows` | `580 rows` | clean |
| ok | parts | `500 rows` | `500 rows` | `250 rows` | clean |
| ok | plan_drift_features | `43,052 rows` | `43,052 rows` | _(new check)_ | clean |
| ok | plants | `7 rows` | `7 rows` | `7 rows` | clean |
| ok | po_line_revisions | `5,121 rows` | `5,121 rows` | `6,000 rows` | clean |
| ok | po_line_schedules | `14,755 rows` | `14,755 rows` | `40,000 rows` | clean |
| ok | po_lines | `9,214 rows` | `9,214 rows` | `20,000 rows` | clean |
| ok | product_economics | `120 rows` | `120 rows` | _(new check)_ | clean |
| ok | production_actual | `48,012 rows` | `48,012 rows` | _skip: no file_ | clean |
| ok | production_plan | `64,837 rows` | `64,837 rows` | _skip: no file_ | clean |
| ok | products | `120 rows` | `120 rows` | `120 rows` | clean |
| ok | purchase_orders | `9,214 rows` | `9,214 rows` | `6,667 rows` | clean |
| ok | quality_inspections | `17,218 rows` | `17,218 rows` | `23,000 rows` | clean |
| ok | revealed_capacity_monthly | `10,651 rows` | `7,560 rows` | `4,000 rows` | clean |
| ok | shortage_events | `3,580 rows` | `3,580 rows` | `1,800 rows` | clean |
| ok | snapshots | `100 rows` | `100 rows` | `401 rows` | clean |
| ok | sourcing_channels | `120 rows` | `120 rows` | `320 rows` | clean |
| ok | supplier_acknowledgements | `9,214 rows` | `9,214 rows` | `15,600 rows` | clean |
| ok | supplier_allocation | `120 rows` | `120 rows` | `320 rows` | clean |
| ok | supplier_audits | `500 rows` | `500 rows` | `500 rows` | clean |
| ok | supplier_capacity | `10,639 rows` | `10,639 rows` | `15,930 rows` | clean |
| ok | supplier_contracts | `120 rows` | `120 rows` | `500 rows` | clean |
| ok | supplier_financials | `900 rows` | `900 rows` | `450 rows` | clean |
| ok | supplier_quality_ppm | `12,480 rows` | `12,480 rows` | `900 rows` | clean |
| ok | supplier_sites | `190 rows` | `190 rows` | `75 rows` | clean |
| ok | supplier_upstream | `300 rows` | `300 rows` | `180 rows` | clean |
| ok | suppliers | `120 rows` | `120 rows` | `50 rows` | clean |
| ok | tooling | `120 rows` | `120 rows` | `150 rows` | clean |
| **FAIL** | training_labels | `6,000 rows` | `6,000 rows` | `12,000 rows` | clean |

### Tier 2


**GATING**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| `SKIP` | training_labels: cleared Tier 1 | _skip: no_ | _skip: no_ | _(new check)_ | n/a |

**AS-OF**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | asn: two timestamps differ | `59.9%` | `59.9%` | `14.5%` | >= 2% |
| ok | asn: lag is not constant | `sd/\|mean\|=0.753` | `sd/\|mean\|=0.753` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | asn: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | expedite_events: two timestamps differ | `83.4%` | `83.4%` | `0.0%` | >= 2% |
| ok | expedite_events: lag is not constant | `sd/\|mean\|=0.708` | `sd/\|mean\|=0.708` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | expedite_events: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | goods_receipts: two timestamps differ | `65.0%` | `65.0%` | `14.1%` | >= 2% |
| ok | goods_receipts: lag is not constant | `sd/\|mean\|=0.694` | `sd/\|mean\|=0.694` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | goods_receipts: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | grn_lines: two timestamps differ | `65.1%` | `65.1%` | `14.1%` | >= 2% |
| ok | grn_lines: lag is not constant | `sd/\|mean\|=0.684` | `sd/\|mean\|=0.684` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | grn_lines: recorded_ts >= event | `0` | `0` | `0` | 0 |
| `SKIP` | inventory_position_weekly: two timestamps differ | _skip: no event column_ | _skip: no event column_ | _skip: no event column_ | n/a |
| ok | inventory_snapshots: two timestamps differ | `39.8%` | `39.8%` | `0.0%` | >= 2% |
| ok | inventory_snapshots: lag is not constant | `sd/\|mean\|=0.689` | `sd/\|mean\|=0.689` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | inventory_snapshots: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | inventory_transactions: two timestamps differ | `40.2%` | `40.2%` | `24.8%` | >= 2% |
| ok | inventory_transactions: lag is not constant | `sd/\|mean\|=0.695` | `sd/\|mean\|=0.695` | `sd/\|mean\|=2.062` | >= 0.1 |
| ok | inventory_transactions: recorded_ts >= event | `0` | `0` | `12,201` | 0 |
| ok | line_stop_events: two timestamps differ | `83.6%` | `83.6%` | `0.0%` | >= 2% |
| ok | line_stop_events: lag is not constant | `sd/\|mean\|=0.705` | `sd/\|mean\|=0.705` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | line_stop_events: recorded_ts >= event | `0` | `0` | `0` | 0 |
| `SKIP` | plan_drift_features: two timestamps differ | _skip: no event column_ | _skip: no event column_ | _skip: no event column_ | n/a |
| ok | po_line_revisions: two timestamps differ | `60.1%` | `60.1%` | `44.0%` | >= 2% |
| ok | po_line_revisions: lag is not constant | `sd/\|mean\|=0.751` | `sd/\|mean\|=0.751` | `sd/\|mean\|=7.485` | >= 0.1 |
| ok | po_line_revisions: recorded_ts >= event | `0` | `0` | `2,562` | 0 |
| -- | po_line_schedules: two timestamps differ | `48.5%` | `48.5%` | `0.0%` | exempt |
| ok | po_line_schedules: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | po_lines: two timestamps differ | `60.5%` | `60.5%` | `0.0%` | >= 2% |
| ok | po_lines: lag is not constant | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | po_lines: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | production_actual: two timestamps differ | `60.1%` | `60.1%` | _skip: no file_ | >= 2% |
| ok | production_actual: lag is not constant | `sd/\|mean\|=0.753` | `sd/\|mean\|=0.753` | _(new check)_ | >= 0.1 |
| ok | production_actual: recorded_ts >= event | `0` | `0` | _(new check)_ | 0 |
| ok | production_plan: two timestamps differ | `60.3%` | `60.3%` | _skip: no file_ | >= 2% |
| ok | production_plan: lag is not constant | `sd/\|mean\|=0.750` | `sd/\|mean\|=0.750` | _(new check)_ | >= 0.1 |
| ok | production_plan: recorded_ts >= event | `0` | `0` | _(new check)_ | 0 |
| ok | purchase_orders: two timestamps differ | `60.2%` | `60.2%` | `0.0%` | >= 2% |
| ok | purchase_orders: lag is not constant | `sd/\|mean\|=0.754` | `sd/\|mean\|=0.754` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | purchase_orders: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | quality_inspections: two timestamps differ | `59.6%` | `59.6%` | `14.6%` | >= 2% |
| ok | quality_inspections: lag is not constant | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | quality_inspections: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | shortage_events: two timestamps differ | `82.6%` | `82.6%` | `0.0%` | >= 2% |
| ok | shortage_events: lag is not constant | `sd/\|mean\|=0.716` | `sd/\|mean\|=0.716` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | shortage_events: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | supplier_acknowledgements: two timestamps differ | `60.4%` | `60.4%` | `32.7%` | >= 2% |
| ok | supplier_acknowledgements: lag is not constant | `sd/\|mean\|=0.745` | `sd/\|mean\|=0.745` | `sd/\|mean\|=2.057` | >= 0.1 |
| ok | supplier_acknowledgements: recorded_ts >= event | `0` | `0` | `4,227` | 0 |
| ok | supplier_allocation: two timestamps differ | `60.8%` | `60.8%` | `100.0%` | >= 2% |
| ok | supplier_allocation: lag is not constant | `sd/\|mean\|=0.720` | `sd/\|mean\|=0.720` | `sd/\|mean\|=0.000` | >= 0.1 |
| -- | supplier_allocation: recorded_ts >= event | `0` | `0` | `0` | n/a |
| ok | supplier_audits: two timestamps differ | `78.8%` | `78.8%` | `100.0%` | >= 2% |
| ok | supplier_audits: lag is not constant | `sd/\|mean\|=0.727` | `sd/\|mean\|=0.727` | `sd/\|mean\|=0.572` | >= 0.1 |
| ok | supplier_audits: recorded_ts >= event | `0` | `0` | `0` | 0 |
| ok | supplier_capacity: two timestamps differ | `94.8%` | `94.8%` | `44.1%` | >= 2% |
| ok | supplier_capacity: lag is not constant | `sd/\|mean\|=0.527` | `sd/\|mean\|=0.527` | `sd/\|mean\|=0.000` | >= 0.1 |
| -- | supplier_capacity: recorded_ts >= event | `0` | `0` | `0` | n/a |
| ok | supplier_quality_ppm: two timestamps differ | `100.0%` | `100.0%` | `100.0%` | >= 2% |
| ok | supplier_quality_ppm: lag is not constant | `sd/\|mean\|=0.168` | `sd/\|mean\|=0.168` | `sd/\|mean\|=0.591` | >= 0.1 |
| ok | supplier_quality_ppm: recorded_ts >= event | `0` | `0` | `0` | 0 |

**CONSERVATION**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | channel store conserves ordered units | `1,944,910 vs 1,944,910` | `2,878,869 vs 1,944,910` | `138,615,579 vs 4,114,197` | exact |
| ok | channel store conserves received units | `1,870,498 vs 1,870,498` | `2,740,051 vs 1,870,498` | `110,778,603 vs 2,992,098` | exact |
| ok | derived store buckets on visible week | `max(event_week, recorded_week)` | `neither` | `neither` | max(event,recorded) |
| ok | supplier store conserves ordered units | `1,944,910 vs 1,944,910` | `2,878,869 vs 1,944,910` | `21,468,423 vs 4,114,197` | exact |
| ok | supplier store conserves received units | `1,870,498 vs 1,870,498` | `2,740,051 vs 1,870,498` | `17,303,481 vs 2,992,098` | exact |
| ok | part_demand_weekly horizon_days identity | `0/7,800` | `0/7,800` | `0/50,000` | 0 |
| ok | part_demand_weekly p90 >= p50 | `0/7,800` | `0/7,800` | `0/50,000` | 0 |
| ok | part_demand_weekly is strictly forward-looking | `0/7,800` | `0/7,800` | `0/50,000` | 0 |
| -- | part_demand_weekly unit conservation | `not asserted` | `not asserted` | `not asserted` | n/a |

**LABELS**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | label window opens after the snapshot | `0` | `0` | `0` | 0 |
| ok | arrival_week: something is censored | `13.6%` | `13.6%` | `0.0%` | 1%-60% |
| ok | arrival_week: uncensored target varies | `sd/\|mean\|=0.5365 over 11 values` | `sd/\|mean\|=0.5365 over 11 values` | `sd/\|mean\|=0.5831 over 2,164 values` | > 0.01 and >= 5 |
| ok | arrival_week: target is not a uniform random draw | `D=0.0945 p=1.57e-08` | `D=0.0945 p=1.57e-08` | _(new check)_ | p < 0.01 |
| ok | arrival_week: entity_type matches the spec | `0/1,200 wrong` | `0/1,200 wrong` | `1,919/2,419 wrong` | 0 |
| ok | capacity_strain: something is censored | `3.6%` | `3.6%` | `0.0%` | 1%-60% |
| ok | capacity_strain: uncensored target varies | `sd/\|mean\|=0.8674 over 660 values` | `sd/\|mean\|=0.5783 over 1,119 values` | `sd/\|mean\|=0.5878 over 2,135 values` | > 0.01 and >= 5 |
| ok | capacity_strain: target is not a uniform random draw | `D=0.3146 p=1.29e-100` | `D=0.0160 p=0.926` | _(new check)_ | p < 0.01 |
| ok | capacity_strain: entity_type matches the spec | `0/1,200 wrong` | `0/1,200 wrong` | `1,916/2,431 wrong` | 0 |
| ok | demand_drift: something is censored | `4.6%` | `4.6%` | `0.0%` | 1%-60% |
| ok | demand_drift: uncensored target varies | `sd/\|mean\|=0.2276 over 716 values` | `sd/\|mean\|=0.1954 over 1,060 values` | `sd/\|mean\|=0.5808 over 2,124 values` | > 0.01 and >= 5 |
| ok | demand_drift: target is not a uniform random draw | `D=0.4366 p=2.08e-191` | `D=0.0277 p=0.339` | _(new check)_ | p < 0.01 |
| ok | demand_drift: entity_type matches the spec | `0/1,200 wrong` | `0/1,200 wrong` | `1,902/2,403 wrong` | 0 |
| ok | fill_rate: something is censored | `4.7%` | `4.7%` | `0.0%` | 1%-60% |
| ok | fill_rate: uncensored target varies | `sd/\|mean\|=0.1515 over 159 values` | `sd/\|mean\|=0.0687 over 172 values` | `sd/\|mean\|=0.5797 over 2,138 values` | > 0.01 and >= 5 |
| ok | fill_rate: target is not a uniform random draw | `D=0.8281 p=0` | `D=0.8453 p=0` | _(new check)_ | p < 0.01 |
| ok | fill_rate: entity_type matches the spec | `0/1,200 wrong` | `0/1,200 wrong` | `1,892/2,413 wrong` | 0 |
| ok | shortage_qty: something is censored | `3.8%` | `3.8%` | `0.0%` | 1%-60% |
| ok | shortage_qty: uncensored target varies | `sd/\|mean\|=12.8735 over 12 values` | `sd/\|mean\|=0.5675 over 293 values` | `sd/\|mean\|=0.5706 over 2,076 values` | > 0.01 and >= 5 |
| ok | shortage_qty: target is not a uniform random draw | `D=0.9905 p=0` | `D=0.0295 p=0.262` | _(new check)_ | p < 0.01 |
| ok | shortage_qty: entity_type matches the spec | `0/1,200 wrong` | `0/1,200 wrong` | `1,882/2,334 wrong` | 0 |
| ok | entity_id resolves for entity_type=po_line | `0/2,097 unresolved` | `0/2,097 unresolved` | `87/2,261 unresolved` | 0 |
| ok | entity_id resolves for entity_type=channel | `0/120 unresolved` | `0/120 unresolved` | `2,252/2,292 unresolved` | 0 |
| `SKIP` | entity_id resolves for entity_type=part_plant | _skip: 199 ids_ | _skip: 199 ids_ | _skip: 2,231 ids_ | n/a |
| `SKIP` | entity_id resolves for entity_type=product_plant | _skip: 417 ids_ | _skip: 417 ids_ | _skip: 2,274 ids_ | n/a |

**SEPARATION**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | channel_performance_weekly: no forward-looking column | `none` | `none` | `none` | none |
| ok | inventory_position_weekly: no forward-looking column | `none` | `none` | `none` | none |
| ok | part_demand_weekly: no forward-looking column | `none` | `none` | `none` | none |
| ok | plan_drift_features: no forward-looking column | `none` | `none` | `none` | none |
| ok | revealed_capacity_monthly: no forward-looking column | `none` | `none` | `none` | none |
| ok | supplier_performance_weekly: no forward-looking column | `none` | `none` | `none` | none |
| ok | model_outputs disjoint from feature tables | `0 table(s) overlap` | `0 table(s) overlap` | `0 table(s) overlap` | 0 |

**SNAPSHOTS**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | snapshots: as_of_ts <= data_cutoff_ts | `0/100` | `0/100` | `0/401` | 0 |
| ok | snapshots: five version fields populated | `0 field(s) blank` | `0 field(s) blank` | `0 field(s) blank` | 0 |
| ok | snapshots: code_commit is consistent | `1 distinct` | `1 distinct` | `1 distinct` | 1 |
| ok | snapshots: horizon_days matches the labels | `labels reach +90d` | `labels reach +90d` | `labels reach +90d` | snapshot 90d |

### Tier 3


**SHAPE**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| **FAIL** | PO lines with fill < 1.0 | `15.9%` | `20.0%` | `72.7%` | 8%-15% |
| ok | PO lines arriving late | `19.8%` | `19.8%` | `82.3%` | 15%-25% |
| ok | fill-rate mass at exactly 1.0 | `84.1%` | `80.0%` | `27.3%` | >= 60% |
| ok | fill-rate mass at exactly 0 | `2.0%` | `2.0%` | `31.8%` | 1%-3% |
| ok | right-censored tail is the right size | `0.38x` | `0.38x` | `0.48x` | 0.3-1.2x |
| **FAIL** | supplier-months constrained | `97.6%` | `97.6%` | `88.5%` | 20%-30% |
| **FAIL** | capacity unobservable | `2.4%` | `2.4%` | `11.5%` | >= 70% |
| ok | lead-time distribution is right-skewed | `+0.88` | `+0.88` | `+0.05` | >= 0.5 |
| ok | shortage events per year | `235.6` | `235.6` | `120.5` | 180-250 |
| ok | line stops per year | `21.7` | `21.7` | `14.7` | 15-25 |
| ok | expedites per year | `131.6` | `131.6` | `93.7` | 100-150 |
| ok | month-of-year seasonality in demand | `1.46x peak/trough` | `1.46x peak/trough` | `1.03x peak/trough` | > 1.10 |
| **FAIL** | PO lines with fill < 1.0 [2019-2025] | `16.6%` | `20.1%` | `72.2%` | 8%-15% |
| ok | PO lines arriving late [2019-2025] | `20.0%` | `20.0%` | `82.7%` | 15%-25% |
| ok | fill-rate mass at exactly 1.0 [2019-2025] | `83.4%` | `79.9%` | `27.8%` | >= 60% |
| ok | fill-rate mass at exactly 0 [2019-2025] | `1.9%` | `1.9%` | `32.1%` | 1%-3% |
| ok | right-censored tail is the right size [2019-2025] | `1.13x` | `1.13x` | `0.91x` | 0.3-1.2x |
| **FAIL** | supplier-months constrained [2019-2025] | `97.8%` | `97.8%` | `88.5%` | 20%-30% |
| **FAIL** | capacity unobservable [2019-2025] | `2.2%` | `2.2%` | `11.5%` | >= 70% |
| ok | lead-time distribution is right-skewed [2019-2025] | `+0.93` | `+0.93` | `+0.05` | >= 0.5 |
| -- | shortage events per year [2019-2025] | `260.1` | `260.1` | `121.2` | 180-250 (not gated) |
| -- | line stops per year [2019-2025] | `24.3` | `24.3` | `16.6` | 15-25 (not gated) |
| -- | expedites per year [2019-2025] | `143.6` | `143.6` | `95.9` | 100-150 (not gated) |
| ok | month-of-year seasonality in demand [2019-2025] | `1.43x peak/trough` | `1.43x peak/trough` | `1.03x peak/trough` | > 1.10 |

**DENSITY**

| | check | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|
| ok | zero-order channel-weeks | `86.0%` | `80.2%` | `0.0%` | 60%-99% |
| ok | weekly store is a panel | `100.0%` | `100.0%` | `0.0%` | >= 90% |
| -- | staleness since last trade | `P50 35d / P90 105d / max 406d` | `P50 28d / P90 77d / max 406d` | _skip: no idle weeks_ | reported |
| ok | PO lines per channel per year | `5.42` | `5.42` | `4.15` | >= 3.0 |
| -- | training rows per snapshot | `60` | `60` | `30` | reported |
| ok | zero-order channel-weeks [2019-2025] | `86.0%` | `80.2%` | `0.0%` | 60%-99% |
| ok | weekly store is a panel [2019-2025] | `100.0%` | `100.0%` | `0.0%` | >= 90% |
| -- | staleness since last trade [2019-2025] | `P50 35d / P90 105d / max 406d` | `P50 28d / P90 77d / max 406d` | _skip: no idle weeks_ | reported |
| ok | PO lines per channel per year [2019-2025] | `7.63` | `7.63` | `4.15` | >= 3.0 |
| -- | training rows per snapshot [2019-2025] | `60` | `60` | `27` | reported |

---

## 9. Comparison against `csv_full_seed1`

Control figures are from run 3's own control run, which is byte-identical to run 2's on every
check line.

| Tier 3 figure | run 3 | run 2 | run 1 | `csv_full_seed1` | expected band | run 3 verdict |
|---|---|---|---|---|---|---|
| PO lines with fill < 1.0 | **15.9%** | 20.0% | 72.7% | 9.5% | 8–15% | **FAIL** |
| PO lines arriving late | 19.8% | 19.8% | 82.3% | 12.8% | 15–25% | ok |
| fill-rate mass at exactly 1.0 | 84.1% | 80.0% | 27.3% | 90.5% | ≥ 60% | ok |
| fill-rate mass at exactly 0 | 2.0% | 2.0% | 31.8% | 1.4% | 1–3% | ok |
| right-censored tail (× span-implied) | 0.38× | 0.38× | 0.48× | 0.56× | 0.30–1.20× | ok |
| supplier-months constrained | **97.6%** | 97.6% | 88.5% | 24.1% | 20–30% | **FAIL** |
| capacity unobservable | **2.4%** | 2.4% | 11.5% | 75.9% | ≥ 70% | **FAIL** |
| lead-time skewness | +0.88 | +0.88 | +0.05 | +2.36 | ≥ 0.50 | ok |
| lead time median / P90 / P99 (days) | 63 / 96 / 132 | 63 / 96 / 132 | 50 / 78 / 93 | 31 / 67 / 139 | — | — |
| shortage events / year | 235.6 | 235.6 | 120.5 | 214.7 | 180–250 | ok |
| line stops / year | 21.7 | 21.7 | 14.7 | 20.0 | 15–25 | ok |
| expedites / year | 131.6 | 131.6 | 93.7 | 124.8 | 100–150 | ok |
| demand peak/trough ratio | 1.46× | 1.46× | 1.03× | 1.30× | > 1.10 | ok |
| zero-order channel-weeks | 86.0% | 80.2% | 0.0% | 81.4% | 60–99% | ok |
| channels with a contiguous weekly panel | 100.0% | 100.0% | 0.0% | 100.0% | ≥ 90% | ok |
| channel-weeks per channel | 365 | 365 | 172 | 387 | — | — |
| PO lines per channel per year | 5.42 | 5.42 | 4.15 | 7.02 | ≥ 3.0 | ok |
| staleness P50 / P90 / max (days) | 35 / 105 / 406 | 28 / 77 / 406 | no idle weeks | 42 / 1085 / 3563 | — | — |
| training rows per snapshot | 60 | 60 | 30 | 44,007 | — | — |

**Eleven of these nineteen figures are byte-identical to run 2.** Lateness, the censored tail,
lead-time percentiles, all three rare-event rates, seasonality, panel contiguity, channel-weeks
per channel, PO lines per channel and training rows per snapshot did not move at all. Four
moved: fill-below-1 (−4.1 pts, toward band), fill-rate mass at 1.0 (+4.1), zero-order
channel-weeks (+5.8), staleness P50/P90. Four are the capacity pair, unmoved and failing.

Run 3 now sits close to `csv_full_seed1` on thirteen of the gated figures. The two that remain
far away are the standing failures: fill-below-1 overshoots (15.9% vs our 9.5%) and capacity
observability is inverted (2.4% unobservable vs our 75.9%).

Two ungated figures still worth flagging:

- **`training rows per snapshot` = 60**, against 44,007 in `csv_full_seed1` — 6,000 labels over
  100 snapshots, 1,200 per task. Not a gate, and the right count varies by orders of magnitude
  between presets, but 1,200 rows per task is a very small training set, and it is the
  population every correlation in §6.5 had to work with. Those correlations are nonetheless
  measured on n ≈ 1,000+, where |r| ≈ 0.02 is decisively distinguishable from a real effect.
- **Lead time median 63 days against the control's 31**, unchanged from run 2 — roughly double
  our world's central tendency, though now correctly right-skewed.

---

## 10. Gate readiness

### Delivery risk / arrival timing — **not trainable**

Blocking checks: none in Tier 1/2/3 that are specific to this head — but it is blocked on
signal, not conformance. `arrival_week` correlates **−0.021** with `otd_rate_last13`, the
channel's own on-time-delivery history (§6.5), and late arrivals are independent of short
arrivals (§6.1, φ = −0.003). The as-of contract is sound, the store conserves exactly, the
panel is contiguous, and every column the head needs is now present — and the target still
carries no relationship to the features.

This head would train and score at chance. It is closer to trainable than at any prior run in
every respect except the one that decides it.

### Fill rate — **not trainable**

Blocking checks: `PO lines with fill < 1.0` (15.9%, band 8–15%) — 0.9 points over, the closest
any standing failure has come. Plus, and decisively: `fill_rate` correlates **+0.023** with
`fill_rate_last13` (§6.5).

The label itself is well-formed — 4.7% censored, 84.1% mass at exactly 1.0, 2.0% at exactly 0,
KS D = 0.83 — and the feature store it reads is now exact and complete. The base rate needs a
0.9-point trim. The correlation with its own history needs a mechanism.

### Part shortage — **not trainable**

Blocking checks: `supplier-months constrained` (97.6%, band 20–30%), `capacity unobservable`
(2.4%, band ≥ 70%) — both unmoved across two regenerations — and `training_labels` failing
Tier 1.

Beyond the gates, this head is the worst-served of the three. 90.1% of shortage events sit on
part-plants with no sourcing channel (§6.3); the antecedent rate among the rest is at chance
(14.2% vs a 15.3% placebo); 1,143 of 1,154 uncensored `shortage_qty` labels are exactly zero
over 13 distinct values; and the label correlates **−0.055** with the demand meant to drive it.
The capacity envelope UC2 needs cannot be inferred when 97.6% of supplier-months look
constrained, and `capacity_basis` — the column that would say how capacity was established —
is the constant string `estimated` on all 10,651 rows.

### Summary

| task | run 1 | run 2 | run 3 | blocking |
|---|---|---|---|---|
| Delivery risk / arrival timing | not trainable | not trainable | **not trainable** | label–feature correlation −0.021 |
| Fill rate | not trainable | not trainable | **not trainable** | fill base rate 15.9%; label–feature correlation +0.023 |
| Part shortage | not trainable | not trainable | **not trainable** | capacity observability ×2 (unmoved); `training_labels` Tier 1; 90.1% of shortages have no channel |

No task moved to trainable. The blocking set has changed character: run 2's blockers were
conformance and conservation defects, and those are now fixed. What blocks all three heads in
run 3 is that the labels are not functions of the features.

---

## Appendix — files

| file | contents |
|---|---|
| `docs/external_dataset_validation.md` | **overwritten during run 3 — see the incident note in §1.** Now holds a notice, not run 1's report |
| `docs/validator_run_external.txt` | run 1 stdout, external (**intact** — source for all run-1 figures here) |
| `docs/validator_run_control_csv_full_seed1.txt` | run 1 stdout, control (intact) |
| `docs/validation2.md` | run 2 report (**not modified**) |
| `docs/validator_run2_external.txt` | run 2 stdout, external (not modified) |
| `docs/validator_run2_control_csv_full_seed1.txt` | run 2 control, augmented validator (not modified) |
| `docs/validator_run2_control_frozen_instrument.txt` | run 2 control, frozen validator (not modified) |
| `docs/validation3.md` | this report |
| `docs/validator_run3_external.txt` | run 3 stdout, external |
| `docs/validator_run3_control_csv_full_seed1.txt` | run 3 stdout, control |
| `docs/validator_run3_joint_probes.py` | the §6 probe script (read-only, separate from the validator) |

Validator SHA-256 at the end of run 3:
`c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` — unchanged from the SHA
recorded in `docs/validation2.md` and from the SHA printed before run 3 began. No edit was made
to `validator.py` at any point.

No `.csv`, `manifest.json`, `dataset_structure.md` or `schema.sql` was modified.
