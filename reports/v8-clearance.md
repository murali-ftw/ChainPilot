# v8 clearance audit — Stages 0–2

**Audience:** whoever decides whether v8 is trained on, and whoever owns the next generator round.
**Scope:** Stage 0 inventory and schema diff, Stage 1 the thirteen clearance blocks, Stage 2 the
frozen validator plus amendment 03 and the generator's promise/arrival construction.
**Measured on:** `db/gen_v8/seed_1001` for every detailed figure, with the three gate-blocking
blocks confirmed on all five seeds. Read-only throughout; no file under `db/gen_v8/**` was written.
**Budget:** 4 h. Stopped at the Stage 1 gate as instructed.

---

## 0. Verdict, up front

> **SUPERSEDED IN PART — read this first.** This report's gate rule made **B9 blocking**. That was
> an error in the brief it was written against, not in the measurement: B9 describes the **same**
> condition v6 and v7 carried, and Phases 1–11 trained on those worlds without incident. **B9 does
> not block Phase 0/1.** Every measurement below stands unchanged; only the gate rule is corrected.
> See **deviation 58** and `reports/phase-0-1-v8.md`, which carries the corrected rule and the
> Phase 0/1 results. **Deviation 55 is also withdrawn** — see §2 B12 and §4.

**Under the brief's rule as given, the Stage 1 gate FAILED and Phase 0 was not started.**

B1 and B12 clear. **B9 does not, and it is not close: 0.00% of arrival label rows have their
`po_line` recorded at or before their own snapshot, on every one of the five seeds.** The
distribution is the same shape Phase 11 measured on v6/v7 — the line is created 7 to 89 days
*after* the snapshot it is labelled at, median 47. The as-of leak that deviations 44, 45 and 48
describe is live and unchanged on v8.

Nine of the thirteen blocks did clear, several of them decisively, and v8 is a substantially
better world than v6 or v7 on almost every axis this audit measured. It does not clear the one
block that gates training.

| | block | verdict | the number that decided it | v6 / v7 for contrast |
|---|---|---|---|---|
| **gate** | **B1** opening stock balance | **CLEARED** | roll-forward reproduces the stated balance on **100.0000%** of 2,253,420 rows, 4,212 / 4,212 part-plants, max abs diff **0** | 0.33% |
| | B2 forward requirement plan | CLEARED | **12–26 periods per version** (median 19), **100.0%** forward, refreshed every **56 days** | 1 period, 0.00% forward, ~9.5 months |
| | B3 cost parameters | **NOT CLEARED** | **0 of 4** present; explicitly `null` in `parameters_v8.json`'s `EXCLUDED__by_design` | 0 of 4 |
| | B4 contract heterogeneity | CLEARED | all 8 columns vary; e.g. `moq` **547 distinct, [1, 1631]** | every one a single constant |
| | B5 min-volume period / penalty basis | **NOT CLEARED** | **neither column exists**; no period, no basis | absent |
| | B6 qualification status | **PARTIAL** | genuine mix (**83.3 / 11.7 / 5.1%**) but `is_approved` still **constant 1** on all 16,072 channels | constant "qualified", constant 1 |
| | B7 supplier capacity ceilings | CLEARED | `evidence_strength` **100% populated**; `revealed_capacity_est` populated on **100% of constrained months** | both 100% NULL |
| | B8 allocation coverage | CLEARED | **97.05%** of part-plants carry a recorded split | 19.5% / 25.7% |
| **gate** | **B9** arrival labels on open PO lines | **NOT CLEARED** | **0.0000%** positive, 332,000 / 332,000 rows, all five seeds | 0% |
| | B10 recorded vs event timestamps | **PARTIAL** | **0 negative lags** on 16 tables, `sd/\|mean\|` **0.23–2.61**; but 2 tables are not right-skewed | `sd/\|mean\|` 0.000, negative on 3 tables |
| | B11 all-zero / constant columns | **NOT CLEARED** | **16 all-zero, 15 all-null, 17 single-valued** numeric columns, 1 empty table | 29 all-zero, 1 empty table |
| **gate** | **B12** label class balance | **CLEARED** | shortage **both classes** (23.62% positive); fill KS vs Uniform(0,1) **p = 0.0**, mass at 1.0 **79.86%** | v6 13.49% positive post-repair; see §2 B12 |
| | B13 primary keys / derived stores | CLEARED | **0 duplicate PKs** across 30 tables; ordered and received units conserve at ratio **1.00000000** | v1: 10,210 dups, 33.7× |

**Gate rule as given:** B1, B9 and B12 must all be CLEARED. B9 is not. **Stop.**

**Gate rule as corrected (deviation 58, governs from here):** B1 and B12 CLEARED → Phase 0/1 may
proceed; they are. B9 NOT CLEARED → the as-of assertion stays armed and is never weakened, no
line-level feature is added to any head, and the promise-date comparison is **reported as retired,
not as a loss**. Phase 0/1 ran under that rule; results in `reports/phase-0-1-v8.md`.

---

## 1. Stage 0 — inventory and schema diff

### 1.1 What is there

| | |
|---|---|
| Layout | `db/gen_v8/<seed>/*.csv` — **the same layout as `gen_v6` / `gen_v7`**, not a new one |
| Seeds | **five**: 1001–1005 (v6 and v7 shipped two) |
| Tables | **49 CSVs per seed**, identical set to v6/v7 |
| Size | 3.7 GB per seed, **18 GB total** |
| World span | 2016-01-01 → 2026-03-31 (`calendar`); channel panel 2016-01-04 → 2026-03-30, 535 weeks |
| Snapshots | **83**, 2016-07-04 → 2025-12-08, horizon 90 d — same count and cadence as v6/v7 |
| Generator | `db/gen_v8/generator_v8.py`, `code_commit` **71de78afa645**, identical across all five manifests |
| Extra non-CSV files | `manifest.json`, `parameters_v8.json`, `measured_outcomes.json`, `mechanism_parameters.json`, `level4.json`, `allocation_disclaimer.txt` |

`§17` reproducibility holds: one code state, five seeds, one `code_commit`, parameter version
`params-v8.0` on every manifest.

### 1.2 Schema diff — there isn't one

This is the single most consequential structural fact in the audit, and it is easy to miss:

```
tables added / removed / renamed      0 / 0 / 0     (49 in all three worlds)
columns added / removed / retyped     0 / 0 / 0     (498 in all three worlds)
column ORDER                          identical, table for table
```

**v8 is byte-identical to v6 and v7 at the schema level.** Everything v8 changed, it changed in
*content*. Two consequences:

- **The existing loader works unchanged.** There is no schema-driven reason to touch
  `ml/data/loader.py`. The only change Phase 0 would have needed is a panel-width one (§1.4).
- **B3 was decided before a single row was read.** Holding cost, ordering cost, truck capacity and
  shortage cost cannot "now exist as columns" in a schema with no new columns. `parameters_v8.json`
  confirms it: all four are `null` under `EXCLUDED__by_design`, with the generator's reasoning that
  they are Rane business decisions, not mechanisms. That position is defensible and it is stated
  honestly — it is still NOT CLEARED against the criterion as written.

### 1.3 Where the content moved

| table | v6 | v8 | ×  | what it is |
|---|---|---|---|---|
| `supplier_allocation` | 2,300 | **628,694** | 273× | B8 — the incumbent split |
| `part_demand_weekly` | 52,080 | **3,667,774** | 70× | B2 — the forward plan |
| `inventory_position_weekly` | 52,080 | **2,253,420** | 43× | B1 — the opening balance store |
| `production_plan` | 15,120 | **312,695** | 21× | B2 — 7 plan versions |
| `po_line_revisions` | 34,441 | **296,346** | 8.6× | §10 feedback arcs |
| `line_stop_events` | 206 | **1,574** | 7.6× | §10 feedback arcs |
| `shortage_events` | 1,911 | **9,641** | 5.0× | §10 feedback arcs |
| `supplier_contracts` | 3,000 | **15,619** | 5.2× | B4 — per supplier-part |
| `training_labels` | 1,070,700 | 1,006,620 | 0.94× | 5 tasks, same 83 snapshots |

### 1.4 Replace or join?

Nothing in `db/gen_v8/` states an intention either way — no README, no note in `manifest.json`.
**I assumed JOIN, as instructed**, and treated v8 as a third world whose bands are its own. Nothing
measured on v6 or v7 is used as an expectation for v8 anywhere in this report; where v6/v7 numbers
appear they are labelled contrast, not baseline. The five-seed structure supports that reading —
v6 and v7 each shipped two seeds and were used as paired worlds, and a replacement would not need
five.

One structural consequence of the content changes that a joined world must carry:
**`revision_count` is now populated in v8** (0–5, 6 distinct, 3.3% non-zero) where deviation 4
recorded it constant zero in both v6 and v7. The other three columns deviation 4 names remain
all-zero. So the panel is **15 value + 10 indicator = 25 channels on v8**, against 14 + 10 = 24 on
v6/v7 — 32 with the calendar block rather than 31. That is a per-world panel width, and it is
exactly the sort of thing that must not be borrowed across worlds.

---

## 2. Stage 1 — the clearance audit

Every block below states its pass criterion, the measurement, and the number that decided it.
"The column exists and is non-null" was not accepted as clearance anywhere.

### B1 — opening stock balance · **CLEARED**

*Pass: a usable opening level per part-plant with a stated as-of timestamp; not all-zero;
negatives a small minority; rolls forward to reproduce the stated balance on a high fraction.*

| | v8 | v6 / v7 |
|---|---|---|
| **roll-forward reproduces the stated balance** | **100.0000%** of 2,253,420 rows | **0.33%** |
| part-plants exact on every week | **4,212 / 4,212** | — |
| max absolute difference | **0** | — |
| `qty_on_hand` negative | **0.0000%** | 97.5% (v6) / 54.9% (v7) |
| all-zero | **no** — mean 1,135.3, max 22,973 | v6 store entirely zero |
| stated as-of | **yes** — `as_of_date` on every row, plus `recorded_ts` and `staleness_days` (P50 2 d, P90 5 d) | — |

The opening posting is real and locatable: **14,826 `adjustment` rows at `event_ts`
2015-12-28**, covering 4,191 part-plants, every one positive, recorded 2015-12-28 → 2016-05-05.
Rolling the signed ledger forward from it reproduces `inventory_position_weekly.qty_on_hand`
exactly, every part-plant, every week.

I verified this independently rather than accepting the generator's claim, and I verified the
check can fail: deleting the opening posting drops the match to **0.5%**; netting 200 scrap
postings into their receipts drops it to **62.9%**. Across seeds: 100.0, 100.0, 100.0, 100.0,
**99.9944** (seed 1005, ~9 rows of ~155,700 on a 300-part-plant sample — noted, not material).

**This is the item `reports/phase-10.md` §6.1 called "the single highest-value item on this list —
the only one that blocks a whole phase." It is delivered.**

### B2 — forward requirement plan · **CLEARED**

*Pass: several future periods per plan version, target_period after the publication date.*

| | v8 | v6 / v7 |
|---|---|---|
| periods per version, `part_demand_weekly` | **12–26, median 19** | **1** |
| periods per version, `production_plan` | **12–26, median 19** | 1 |
| refresh interval | **56 days, exactly, 46 versions** | ~9.5 months, 12 versions |
| `week_start` after `as_of_date` | **100.0%** | — |
| `production_plan` forward (`target_period` > `recorded_ts`) | **100.0%** | **0.00%** |
| horizon | 7–182 days | 30 days only |
| `horizon_days = week_start − as_of_date` | **100.0%** exact | — |
| `p90 ≥ p50` | **100.0%** | — |

`production_plan` also carries 7 plan versions (V1→V7), a `rolling` / `firm` split (213,335 /
99,360) and weekly grain throughout. Deviation 38's "the deliverable schedule is single-period" no
longer describes v8. **`reports/phase-10.md` §6.1 item 2 is delivered.**

### B3 — cost parameters · **NOT CLEARED**

*Pass: holding/carrying, ordering/setup, truck capacity and shortage/stockout cost each exist with
non-trivial values.*

**0 of 4.** No new column exists (§1.2), and `parameters_v8.json` names all four explicitly:

```json
"EXCLUDED__by_design": {
  "carrying_cost_rate": null, "ordering_cost_per_po": null,
  "shortage_cost_per_unit": null, "freight_rate_structure": null,
  "truck_capacity": null, "holding_cost_per_unit_week": null,
  "part_costs.unit_cost_inr": "PLACEHOLDER - arbitrary draw ... Not a Rane price.",
  "part_costs.freight_cost_inr": "PLACEHOLDER - arbitrary draw. NOT a freight structure.",
  "expedite_events.premium_cost_inr": "PLACEHOLDER - ... not a costed Rane rate." }
```

The generator's argument — these are Rane business decisions, and any invented value only tests the
optimiser's sensitivity to the invention — is sound, and `allocation_disclaimer.txt` shipped beside
the data says so in the same terms. **It remains NOT CLEARED**, and it keeps deviations 36, 37 and
41 open exactly as they stand. This is a client ask, not a generator defect.

### B4 — contract heterogeneity · **CLEARED**

*Pass: real variation in each of seven columns. A constant column is NOT CLEARED however documented.*

| column | distinct | min | max | v6/v7 |
|---|---|---|---|---|
| `part_plant.min_order_qty` | **284** | 1 | 840 | constant 10 |
| `part_plant.lot_size` | **142** | 1 | 393 | constant 25 |
| `part_plant.planning_lead_time_days` | **56** | 7 | 69 | constant 30 |
| `supplier_contracts.moq` | **547** | 1 | 1,631 | constant 10 |
| `supplier_contracts.lot_size` | **280** | 1 | 808 | constant 10 |
| `supplier_contracts.max_volume_cap` | **1,423** | 93 | 3,225 | constant 5,000 |
| `supplier_contracts.min_volume_commitment` | **551** | 12 | 1,087 | constant 100 |
| `supplier_contracts.penalty_clause_inr` | **15,610** | 1,670 | 1,799,318 | constant 10,000 |

All eight vary; none is constant. Contracts went from 3,000 to 15,619, one per supplier-part.

**One caveat the generator itself raises and I am passing through rather than resolving.** Both
`generator_v8.py` (§2.6) and `allocation_disclaimer.txt` state these terms are **prediction-task
features only**, sampled from tier/type/class-conditioned distributions, and are **not valid inputs
to a quoted allocation or delivery-schedule recommendation**. B4 as written asks for variation, and
variation is what is there. It does not make deviation 37's capacity gate meaningful for 10.1 —
`max_volume_cap` now ranges [93, 3225] against requirements that reached 2,102 in v6, so the cap
*can* now bind, but binding against an invented spread is not the same as binding against a real
constraint.

### B5 — min-volume period and penalty basis · **NOT CLEARED**

*Pass: the commitment carries a period and the penalty carries a basis.*

`supplier_contracts` has 11 columns. **Neither a period column nor a basis column is among them.**
`min_volume_commitment` now varies (12–1,087) and `penalty_clause_inr` now varies (₹1,670–₹1.8 m),
but a quantity with no window still cannot be enforced, and a penalty with no basis still cannot be
priced.

`valid_from` / `valid_to` exist and span 730–2,599 days (median 1,662), one contract per
supplier-part. That is the **contract's** validity, not the commitment's measurement period: "139
units" against a 4.5-year contract is ambiguous between per-year, per-quarter and per-contract, and
the ambiguity is the whole of `reports/phase-10.md` §6.3 item 7. **v8 made the numbers vary without
making them interpretable.** Unchanged as a client ask.

### B6 — qualification status · **PARTIAL**

*Pass: a genuine mix of qualified / in-qualification / not-qualified, AND `is_approved` not constant 1.*

First half passes, second half fails.

| | v8 | v6/v7 |
|---|---|---|
| `alternate_sources.qualification_status` | **qualified 83.3%, in_progress 11.7%, potential 5.1%** | constant "qualified" |
| `sourcing_channels.is_approved` | **constant `true`, 16,072 / 16,072** | constant 1 |
| `sourcing_channels.approval_status` | **constant "approved"** | constant |
| `alternate_sources.qualification_lead_days` | 150 distinct, 30–179 | 30–179 |

A real 16.7% of alternate sources are now unqualified, so a PPAP lead time **can** bind on the
alternate-source path — which is the path 10.2 actually enumerates. But every sourcing channel is
still approved, so deviation 40's "INERT" verdict stands for any constraint keyed on
`sourcing_channels`. Half the mechanism arrived.

### B7 — supplier capacity ceilings · **CLEARED**

*Pass: `revealed_capacity_est` and `evidence_strength` populated.*

| | v8 | v6/v7 |
|---|---|---|
| `evidence_strength` non-null | **100.0%**, 13 distinct, 0.00–1.00, mean 0.291 | **100% NULL** |
| `revealed_capacity_est` non-null | 30.7% overall — **100% of constrained months** | **100% NULL** |
| `declared_capacity_qty` | 100% non-null, 3,918 distinct, 1,390–45,782 | — |

The 69.3% null is **not a gap, it is the mechanism.** `revealed_capacity_est` is non-null on exactly
the 15,853 rows where `capacity_basis = 'revealed'`, which are exactly the 15,853 rows where
`constrained_month_flag` is true — a perfect 2×2. That is `synthetic_rules.md` §6 working as
specified: where ordered ≪ K the month carries no information about K, and observability is an
outcome, never a setting. Declaring a number there would have been the defect.

I checked the crosstab rather than the null rate, because a null rate alone would have read as a
partial fix. **`reports/phase-10.md` §6.3 item 10 is delivered.**

### B8 — allocation coverage · **CLEARED**

*Pass: a recorded incumbent split for a high fraction of part-plants.*

| | v8 | v6 | v7 |
|---|---|---|---|
| **part-plant coverage** | **97.05%** (4,212 / 4,340) | 19.5% | 25.7% |
| allocation rows | 628,694 | 2,300 | 3,008 |
| versions with >1 supplier | **89.15%** | — | — |
| suppliers per version | 1–11, mean 3.64 | — | — |
| effective-from versions | 41, 2016-02 → 2026-02 | — | — |
| splits summing to 100 (±0.05) | **100.0%** | — | — |

Coverage holds at 97.05% as-of both 2022-01-01 and 2025-01-01 on `recorded_ts`, so the incumbent is
available at the snapshot, not only at end of world. `reports/phase-10.md` §6.3 item 11 is
delivered, and deviation 47's "there is nothing to recommend against" no longer applies.

*Falsification note:* my first mutation for this gate was inert and had to be rewritten — see
`docs/validation/validator_amendment_03.md`. The gate is sound; my first test of it was not.

### B9 — arrival labels on open PO lines · **NOT CLEARED** · **this is the gate**

*Pass: each arrival label row's `po_line` is created and recorded BEFORE its own snapshot, so line
age and `promise_week` are legitimately available at t0.*

| | v8 seed 1001 | v6 / v7 |
|---|---|---|
| arrival label rows | 332,000 | 244,000 |
| rows unresolved against `po_lines` | **0** | 0 |
| **rows whose line is CREATED before its snapshot** | **0 (0.0000%)** | 0 (0%) |
| **rows whose line is RECORDED at or before its snapshot** | **0 (0.0000%)** | 0 (0%) |
| rows whose line is recorded AFTER its snapshot | **332,000 (100.00%)** | 100.00% |

`snapshot_date − created_ts`, days — **every value negative**:

| min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| −89 | −86 | −70 | **−47** | −26 | −10 | **−7** |

Against `recorded_ts` the spread is −195.65 to −7.02, median −49.36. **The nearest row in the
entire dataset is still 7 days short.** There is no tail to keep and no subset to salvage — the
same finding Phase 11 §3.3 recorded, in the same words: there are no rows left, because it is 100%.

**All five seeds:**

| | 1001 | 1002 | 1003 | 1004 | 1005 |
|---|---|---|---|---|---|
| % created before snapshot | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| % recorded at/before snapshot | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |

`promise_date − snapshot` is +8 to +158 days, 100% in the future — the promise date is *about* the
future, which is not the same as being *known* at t0. It belongs to a line that will not be raised
for another seven weeks.

**Why, from the generator (§4 below): the label population is selected as `pt > t0` — PO lines
placed strictly after the snapshot. This is not a bug in the label writer. It is what the arrival
task is, and v8 did not change it.**

Consequences, stated plainly:
- **The as-of assertion in `ml/train/folds.py` must stay armed.** On v8 it would fire on the first
  training batch, exactly as it did in Phase 11.
- **Deviations 44, 45 and 48 carry over to v8 unchanged.** Line age and `promise_week` are still
  not available at prediction time; guide 11.2 stays withdrawn; the promise-date baseline's
  0.87–0.89 C-index is still privileged information and its lateness margin still measured against
  it.
- **Stage 4.3's pre-registered suspicion never got to apply.** It was written for the case where
  B9 cleared and arrival's C-index jumped. B9 did not clear, so there is no number to investigate,
  and any arrival C-index trained on v8 as it stands would be the leak, not the result.

### B10 — recorded vs event timestamps · **PARTIAL**

*Pass: a real distribution, not a constant offset, never negative.*

Sixteen tables carry a measurable lag. **Not one has a negative lag** (v1 had three, with 12,201
inventory transactions and 4,227 acknowledgements recorded before they happened). `sd/|mean|`
ranges **0.227 to 2.615**; v1's was 0.000. The G0 floor of 0.1 is cleared by every table.

Two tables miss `synthetic_rules.md` §3.2's `sd/|mean| ≥ 0.5` shape target:

| table | anchor | `sd/\|mean\|` | median | skew | note |
|---|---|---|---|---|---|
| `plan_drift_features` | `plan_date` | **0.227** | 131 d | **−0.69** | left-skewed, not the right-skewed shape §3.2 requires |
| `inventory_snapshots` | `snapshot_date` | **0.471** | 3 d | **−0.001** | symmetric, discrete 1–5 d — a uniform draw, not a log-normal |

The frozen validator additionally fails three tables on "two timestamps differ ≥ 2%", and its
headline verdict rests on them. **On inspection that verdict is only partly supported** — see §3.2.

### B11 — all-zero and constant columns · **NOT CLEARED**

*This check is cheap and it has caught something every single time it has been run. It did again.*

**16 all-zero numeric columns, 15 all-null, 17 single-valued, 1 empty table.** The frozen
validator's amendment-02 check independently counts 15 all-zero in 5 tables, corroborating.

**All-zero (16):**

| table | columns |
|---|---|
| **`supplier_performance_weekly`** | **`lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13`, `days_since_last_short`, `reporting_lag_days`, `weeks_since_last_activity`, `weeks_since_last_receipt`** — 7 of them |
| `channel_performance_weekly` | `days_since_last_short`, `weeks_since_last_activity`, `weeks_since_last_receipt` |
| `inventory_snapshots` | `qty_blocked`, `qty_in_transit`, `qty_reserved` |
| `inventory_position_weekly` | `qty_blocked` |
| `quality_inspections` | `qty_deviation_accepted` |
| `calendar` | `is_shutdown` |

**The `supplier_performance_weekly` block is the finding.** Seven of its numeric columns are zero
in all 224,700 rows, and one of them is **`otd_rate_last13`** — the column
`synthetic_rules.md` §16 names as the *decisive* Level-3 probe for arrival ("arrival against
`otd_rate_last13`"). The probe passes on v8 only because it is evaluated against the
**channel** store, where `otd_rate_last13` is populated (59 distinct, 21.2% zero). The supplier-level
store is empty of it. Anything that reads supplier-weekly performance reads zeros.

**All-null (15):** `alternate_sources.effective_to`, `bom.parent_part_id`, `bom.effective_to`,
`calendar.regime_note`, `dataset_coverage.known_gaps`, `logistics_lanes.via_checkpoint`,
`part_plant.effective_to`, `parts.criticality_reason`, `parts.standard_lead_time_days`,
`parts.introduced_date`, `parts.discontinued_date`, `products.introduced_date`,
`products.discontinued_date`, `sourcing_channels.effective_to`, `supplier_capacity.part_id`.

Several are benign (`effective_to` null means "current"). `bom.parent_part_id` and
`supplier_capacity.part_id` all-null are **not** benign — they are join keys.

**Single-valued (17):** `bom.bom_level`=1, `customers.is_active`=1, `dataset_coverage.available_years`=10.2,
`grn_lines.receipt_sequence`=1, `grn_lines.is_final_receipt`=1, `parts.is_active`=1, `plants.is_active`=1,
`po_line_schedules.schedule_version`=1, **`po_lines.line_number`=1**, `products.is_active`=1,
`snapshots.horizon_days`=90, **`sourcing_channels.is_approved`=1** (the B6 failure),
`supplier_capacity.shift_basis`=2, `supplier_quality_ppm.parts_supplied`=10000,
`supplier_sites.is_active`=1, `suppliers.is_active`=1, `training_labels.horizon_days`=90.

**Empty table (1):** `model_outputs`, 0 rows — as in v6 and v7.

The good news: `channel_performance_weekly` is down from four constant-zero columns to three,
because **`revision_count` is now populated**. That widens the panel (§1.4).

### B12 — label class balance and distribution · **CLEARED**

*Pass: shortage carries both classes; fill is not uniform-random; capacity has a plausible range;
arrival censoring is stated.*

| task | n | mean | sd | range | censored | verdict |
|---|---|---|---|---|---|---|
| `fill_rate` | 332,000 | 0.8580 | **0.3097** | [0, 1] | 48.61% | not uniform |
| `arrival_week` | 332,000 | 12.99 | 6.31 | [2, 174] | **48.61%** | stated |
| `capacity_strain` | 207,500 | 0.8733 | 0.3575 | **[0.118, 3.0]** | 3.97% | plausible |
| `shortage_qty` | 124,500 | 88.10 | 765.59 | [0, 67,109] | 3.09% | **both classes** |
| `demand_drift` | 10,620 | 0.9681 | 0.0625 | [0.769, 1.966] | 5.26% | — |

- **Fill is not Uniform(0,1).** KS against Uniform(0,1) rejects at **p = 0.0** (statistic 0.799).
  sd is 0.3097, not 1/√12 = 0.28868. Mass at exactly 1.0 is **79.86%** (§7 band ≥ 60%), at exactly
  0.0 is **2.05%** (§7 band 1–3%). Both point masses are present — the v1 tell is absent.
- **Shortage carries both classes:** 76.38% zero, **29,410 positives (23.62%)**.
- **Capacity strain** spans 0.118–3.00, clipped at 3.0 per deviation 9, mean 0.873.
- **Arrival censoring 48.61%**, inside §14's 1–60% requirement, non-zero and non-total.
- Entity binding is one-to-one on every task: arrival and fill → `po_line`, capacity → `channel`,
  shortage → `part_plant`, drift → `product_plant`, **0 unresolved**.

All five seeds clear: fill mass-at-1 79.9–86.4%, KS p = 0.0 throughout, shortage positives
17,285–29,410, arrival censoring 45.5–48.6%.

**What v8 should and should not be credited with here.** The zero-negative defect in v6 **was real
and has been repaired**; the repaired state is what `training_labels.csv` now holds. Both states are
preserved and both were measured:

| file | shortage rows | zeros | positives | positive rate |
|---|---|---|---|---|
| v6 `training_labels.pre_fix` (**pre-repair**) | 45,942 | **0** | 45,942 | **100.00%** — no negative class at all |
| v6 `training_labels.csv` (post-repair) | 124,500 | 107,705 | 16,795 | **13.49%** |
| v7 `training_labels.csv` | 124,500 | 108,545 | 15,955 | 12.82% |
| **v8** `training_labels.csv` | 124,500 | 95,090 | **29,410** | **23.62%** |

The pre-repair file is exactly the population `generator_v8.py`'s comment describes — 45,942 rows,
zero of them zero-valued, which is why PR-AUC was degenerate at 1.000. The Phase 1 label fix
sampled the part × plant universe instead of positives only, which is what produced the 124,500-row
post-repair file at 13.49%.

So: **the defect was real, it was repaired before v8, and v8 is not credited with fixing it.**
v8's genuine improvement is the positive rate, **13.49% → 23.62%**, which moves §9's ~3% operating
base rate further into reach by downsampling. B12 clears on v8 on its own terms.

*(An earlier draft of this report recorded the non-reproduction as deviation 55. That was wrong —
I had compared against the post-repair file only. **Deviation 55 is withdrawn**; see §4.)*

### B13 — primary keys and derived-store consistency · **CLEARED**

**0 duplicate primary keys** across 30 tables checked, including the weekly stores where v1 carried
10,210 on 55,000 rows. `supplier_capacity`'s key is `(supplier_id, part_id, effective_from)`, not
the `(supplier_id, month)` I first assumed; 0 duplicates under the correct key.

**Derived stores are not inflated.** Both §4 conservation identities are exact, bucketing on the
visible week `max(event_week, recorded_week)` and filtering both sides identically:

| quantity | source | store | ratio |
|---|---|---|---|
| ordered units | 304,313,857 | 304,313,857 | **1.00000000** |
| received units | 240,258,195 | 240,258,195 | **1.00000000** |

v1's were 33.7× and 37.0×. 0 GRN lines fail to resolve to a channel.

### Falsifiability — standing rule 1

Every gate above was tested for whether it *can* fail, by mutation. **23 gates, 23 demonstrated
firing.** Full table in `docs/validation/validator_amendment_03.md`; the mutations reproduce the
defects that are actually on this project's record (v6/v7 constants, v1's Uniform(0,1) labels,
v1's 33.7× inflation, v1's 10,210 duplicate PKs, v1's constant lag).

**One of my own gates was inert on first writing**, and it is recorded because that is the defect
this rule exists to catch: B8's coverage gate did not move when `supplier_allocation` was thinned
to a 20% sample of rows (97.05% → 97.05%), because each part-plant carries ~149 rows. Thinning the
**part-plant** population instead fires it correctly (19.5% → NOT CLEARED). The gate was fine; my
falsification of it was not.

The strongest demonstration is not synthetic: running the whole suite against `db/gen_v6/seed_1001`
flips **eight blocks** from CLEARED to NOT CLEARED, with v6's own numbers (B1 4.58%, B2 0.0%,
B8 18.99%). These checks discriminate between two real worlds.

---

## 3. Stage 2 — the frozen validator

### 3.1 `db/validator.py` unmodified against v8

```
157 passed, 22 FAILED, 5 skipped, 229 checks total
VERDICT: NOT USABLE
```

Two procedural notes first:

- **`db/validator.py` was touched in the v8 commit (86e86d1)** — `3 insertions, 3 deletions`.
  I checked the diff before treating this as a concern, and it is **not** one: all three lines are
  path strings following the `docs/` → `docs/validation/` reorganisation (`REPORT_PATH`, one
  comment reference, one `--help` string). **No check, band, threshold, gate condition or logic
  moved.** The instrument is materially still frozen, and its SHA differing from
  `validator_amendment_02.md`'s recorded `4dbb2298…` is fully explained by those three lines.
  Recorded as **deviation 50** only so the SHA mismatch does not read as an undocumented amendment
  to the next person who checks. v8's run is comparable to v6/v7's.
- **The validator overwrites `docs/validation/external_dataset_validation.md` on every run.**
  Running it against v8 silently replaced the existing v6/v7 report. I restored it from git; the
  working tree is clean. `--no-report` suppresses this and should be the default habit when
  validating a world you are not adopting.

The 22 failures, grouped:

| group | checks | measured | band |
|---|---|---|---|
| **all-zero columns** | 6 | 15 columns in 5 tables | 0 |
| **as-of "two timestamps differ"** | 3 | 0.0% on `production_plan`, `supplier_allocation`, `supplier_capacity` | ≥ 2% |
| **event rates** | 3 | shortage **940.8**/yr, line stops **153.6**/yr, expedites **640.4**/yr | 180–250, 15–25, 100–150 |
| **shape** | 8 | fill<1.0 **23.5%**, late **30.4%**, constrained **36.2%**, unobservable **63.8%** (×2 windows) | 8–15%, 15–25%, 20–30%, ≥70% |
| **structural** | 2 | `revealed_capacity_monthly` 1 type violation; 48/49 clean tables | 49/49 |

### 3.2 The "NOT USABLE" verdict does not survive inspection intact

The verdict text reads: *"there is no as-of structure: `recorded_ts` sits in the same week as the
event on ~every row of `production_plan`, `supplier_allocation`, `supplier_capacity`."* Measured
directly:

| table | `recorded_ts − anchor` | reading |
|---|---|---|
| `production_plan` | anchor `plan_date`; **0 to +3 days**, mean +1.50, sd 1.12, **0% negative** | a real dispersed lag, just short of a week boundary. The check counts *later-week* only, so a 0–3 d lag scores 0% by arithmetic. **A band miss, not an inversion.** |
| `supplier_allocation` | anchor `effective_from`; **−44 to −5 days**, mean −24.5 | `recorded_ts` is *before* `effective_from` — a figure agreed before it takes effect |
| `supplier_capacity` | anchor `effective_from`; **−44 to −5 days**, mean −24.6 | same |

For the latter two this is **exactly the exemption `synthetic_rules.md` §3.2 writes down**
("agreeing a figure before it takes effect is normal and must be exempted explicitly"), and the
frozen validator's *own* sibling check honours it — `supplier_allocation: recorded_ts >= event` is
reported `[----] not gated: the anchor is 'effective_from', a forward-dated validity-window start`.
The "two timestamps differ" check simply was not given the same exemption.

So: one genuine band miss on a short planning lag, and two artefacts of a frozen check's shape
meeting a forward-dated anchor. **The headline "no as-of structure" is not supported.** My B10
measurement — 16 tables, 0 negative lags, `sd/|mean|` 0.227–2.615 — is the more informative read,
and it is why B10 is PARTIAL rather than NOT CLEARED. **This is deviation 51.**

The **event-rate failures are real and large**, and they are not explained away by regime: 940.8
shortage events/yr against a 180–250 band, 153.6 line stops against 15–25, 640.4 expedites against
100–150 — **4× to 6× over**, on the normal-regime slice, against the rates Rane themselves stated.
`synthetic_rules.md` §0 calls those rates "the sanity anchor which the synthetic world must
reproduce". v8's §2.5 feedback-arc work (`po_line_revisions` 8.6×, `line_stop_events` 7.6×,
`shortage_events` 5.0×) appears to have over-fired. This does not block Phase 0/1, and it does
bear on any Phase 9/10 claim about how often anything happens.

The `revealed_capacity_monthly` type violation is benign and worth naming because of what caused
it: `revealed_capacity_est` is declared `INTEGER` and reads `float64` purely because 69.3% of its
values are now NULL (nullable-integer promotion); all 15,853 non-null values are whole numbers.
**The fix that cleared B7 is what trips this Tier-1 check.**

### 3.3 Amendment 03

`db/validator_amendment_03.py` — **462 insertions, one new file. `db/validator.py` is
byte-identical before and after**, SHA `188180bbb612…` unchanged, `0 files changed` against the
frozen instrument.

Amendments 01 and 02 edited `validator.py` in place. **I did not**, because the brief's standing
rules list that file as protected while Stage 2.2 asks for an amendment to it. A separate module
satisfies both readings: the frozen 229 checks return exactly what they returned, and the B1–B13
checks become repeatable. **This is deviation 49**; folding them inline is mechanical and changes
no verdict. Full rationale, usage and the falsification table are in
`docs/validation/validator_amendment_03.md`.

It reproduces all thirteen verdicts independently of the ad-hoc scripts used in §2, and exits
non-zero on the gate:

```
8 cleared, 1 partial, 3 not cleared, 1 errored → after fix: 9/1/3/0
BLOCKING (B1, B9, B12): {'B1': 'CLEARED', 'B9': 'NOT CLEARED', 'B12': 'CLEARED'}
STAGE 1 GATE: FAIL -- do not start Phase 0
```

### 3.4 The generator's promise and arrival construction

**Configuration and definitions only. No table under `db/gen_v8/**` was read for this step.**
From `db/gen_v8/generator_v8.py`:

```python
contracted   = r.integers(15, 70, NCH)                       # per channel, drawn once  (l.314)
promise      = ev_po + contracted[pch]                       #                          (l.868)
cur_promise  = promise - promise_off                         # buyer pull-in via revision (l.873)

base  = contracted[idx] * P['lead_base_frac']                # lead_base_frac = 0.51     (l.573)
lead  = exp(normal(log(max(base, 3)), P['lead_sigma']))      # lead_sigma    = 0.34      (l.574)
lead *= (1 + 0.55*max(sup_state,0) + 0.9*transit_state
           + 0.85*clip(sup_load_prev - 0.70, 0, 2)
           + (1.55 - 1)*regime)                              #                        (l.575-577)
lead  = lead * (1 + 0.85 * clip(shortfall, 0, 1))            # NEW in v8               (l.583)
arr_t = t + ceil(lead / 7)                                   #                          (l.586)
```

**Do promise and arrival still share `po_created` and `contracted[channel]`? Yes — identically to
v6/v7.** `promise` is `po_created + contracted[channel]`; `lead`'s median is
`0.51 × contracted[channel]` and its anchor is the same `po_created` week. Deviation 31's finding
and deviation 32's consequence carry over to v8 unchanged. `arrival_week` still never reads
`promise_date`.

**Does anything now make arrival ranking a fair comparison? No.** Two v8 changes touch this and
neither helps:

- **`lead_short_beta = 0.85`** couples lead time to the line's shortfall (§16 probe 3 — late and
  short as one event). This adds a line-level term to arrival that is *not* in promise, which
  weakens the shared structure slightly. It does not give the head anything, because the head
  still sees only the channel panel.
- **`promise_off`** — buyer pull-in via revision, with `po_line_revisions` now 8.6× larger — makes
  `current_promise_date` differ from `original_promise_date` per line. That is *more* line-level
  information in the promise, and **none of it is available at t0**, so it widens the privilege
  rather than closing it.

And the reason the question is moot, from the label loop itself (l.1469–1486):

```python
fut = (pt > t0) & (pt <= t1)      # pt = the PO line's PLACEMENT week
li  = np.where(fut)[0]            # the label population: lines placed AFTER the snapshot
aw  = pa[li] - t0                 # arrival week, measured from the snapshot
```

**The arrival label population is selected as lines placed strictly after `t0`, by construction.**
That is the mechanical cause of B9's 0.00%, it is identical to v6/v7, and it is not a defect in the
label writer — it is what this arrival task is: a channel-level forecast of orders not yet raised.
Deviation 45's precondition ("labels must attach to lines already raised at t0") is still unmet.

---

## 4. Deviations — continuing the guide's index

| # | Spec / brief says | Measured | Where |
|---|---|---|---|
| **49** | validator amendments are in-place edits to `db/validator.py` (as 01 and 02 were) | **amendment 03 is a separate module**, `db/validator_amendment_03.py`, 462 insertions, because the brief's standing rules mark `validator.py` protected while Stage 2.2 asks for an amendment to it. The frozen file is byte-identical; 0 checks changed | `docs/validation/validator_amendment_03.md` |
| **50** | `db/validator.py`'s SHA matches `validator_amendment_02.md`'s recorded `4dbb2298…` | **it does not** — it is `188180bbb612…`, because commit `86e86d1` changed **three path strings** (`docs/` → `docs/validation/`). Diff verified: no check, band, threshold or logic moved. The instrument is materially frozen and v8's run **is** comparable to v6/v7's. Recorded so the SHA mismatch is not mistaken for an undocumented amendment | §3.1 |
| **51** | the frozen validator's "NOT USABLE — no as-of structure" verdict on v8 is a finding | **partly unsupported.** 2 of the 3 driving checks fire on `effective_from`, a forward-dated anchor that `synthetic_rules.md` §3.2 exempts and that the validator's own sibling check *does* exempt; the third is a 0–3 d planning lag that never crosses a week boundary. 0 tables have a negative lag; `sd/\|mean\|` is 0.227–2.615 | §3.2 |
| **52** | v8 rectifies every block found in Phases 1–11 | **9 of 13 clear; B3, B5, B11 do not and B6, B10 are partial.** B9 — the gate — is 0.00% on all five seeds, unchanged from v6/v7 | §0 |
| **53** | `channel_performance_weekly` has four constant-zero columns (deviation 4) | **three on v8** — `revision_count` is now populated (6 distinct, 3.3% non-zero). The panel is **15 value + 10 indicator = 25** channels on v8, 32 with the calendar block, against 24 / 31 on v6/v7 | §1.4 |
| **54** | `otd_rate_last13` is the decisive Level-3 arrival probe | **all-zero in `supplier_performance_weekly`** (224,700 rows), along with 6 other columns there. The probe passes only against the **channel** store, where the column is populated | §2 B11 |
| **~~55~~** | ~~v6's zero-negative shortage premise does not reproduce~~ | **WITHDRAWN.** The premise is correct. I compared against the post-repair `training_labels.csv` and missed `training_labels.pre_fix`, which holds the pre-repair state: **45,942 rows, 0 zeros, 100% positive**. The defect was real, was repaired in the Phase 1 label fix, and **v8 is not credited with fixing it**. v8's genuine improvement is the positive rate **13.49% → 23.62%** | §2 B12 |
| **56** | v8's event rates reproduce Rane's stated operational figures | **4×–6× over band on the normal-regime slice**: shortage 940.8/yr (180–250), line stops 153.6/yr (15–25), expedites 640.4/yr (100–150). §2.5's feedback arcs over-fired | §3.1 |
| **58** | B9 blocks Phase 0/1 (this report's gate rule) | **it does not.** B9 describes the same condition v6/v7 carried, and Phases 1–11 trained on those worlds. B9 blocks only guide 11.2 (deviation 45), arrival ranking claims (32) and any deployable reading of the promise-date baseline (48) — all three already closed. Corrected rule: **B1 + B12 cleared → proceed**; B9 not cleared → assertion stays armed, no line-level feature in any head, promise-date comparison **retired, not reported as a loss** | §0, `reports/phase-0-1-v8.md` |
| **57** | populating `revealed_capacity_est` is a clean fix | it is, and it **trips a Tier-1 type check**: the column is declared `INTEGER` and reads `float64` because 69.3% are now NULL. All non-null values are whole numbers | §3.2 |

---

## 5. What each block unblocks, and what is still blocked

**Cleared, and what they unblock:**

| block | unblocks |
|---|---|
| **B1** | the Phase 9 stock roll-forward; 10.1's stock-balance form and safety-stock floor; 10.2's real objective. Deviation 39 is dischargeable on v8 |
| **B2** | any multi-week delivery schedule. Deviation 38 no longer describes v8 |
| **B4** | contract terms as **model features**. Explicitly *not* as quoted-optimiser inputs — the generator and `allocation_disclaimer.txt` both say so |
| **B7** | 10.2's capacity-ceiling constraint |
| **B8** | 10.2 has an incumbent to recommend against. Deviation 47's premise improves |
| **B12** | shortage has a negative class; fill and capacity heads have real label distributions |
| **B13** | the derived stores can be trusted against their sources |

**Not cleared, and what stays blocked:**

| block | keeps blocked |
|---|---|
| **B9** | ~~Phase 0 and Phase 1 on v8~~ — **corrected, deviation 58: it does not.** B9 blocks only guide 11.2 (deviation 45), any arrival ranking claim (32) and any deployable reading of the promise-date baseline (48) |
| **B3** | 10.1's objective (deviations 36, 37) and 10.2's shortage-cost objective (41). **Client ask, not a generator defect** |
| **B5** | the min-volume constraint and its penalty remain unenforceable |
| **B11** | anything reading `supplier_performance_weekly`; `bom.parent_part_id` and `supplier_capacity.part_id` as join keys |
| **B6** (partial) | constraints keyed on `sourcing_channels.is_approved` stay inert (deviation 40). The alternate-source path *can* now bind |
| **B10** (partial) | nothing hard; two tables' lag shapes are wrong rather than absent |

---

## 6. Open items

1. **B9 is the whole gate, and it is a generator-design question, not a bug fix.** The arrival
   label population is `pt > t0` by construction. Clearing it means either (a) labelling lines
   that are already open at `t0` — a different task, with different censoring — or (b) accepting
   that arrival stays a channel-level forecast of unraised orders and **retiring the comparison
   against the promise date entirely**, rather than reporting it as a loss. Option (b) costs
   nothing and is already what deviations 32 and 48 imply. **Resolved: option (b) was taken
   (deviation 58). Phase 0/1 proceeded; see `reports/phase-0-1-v8.md`.**
2. **Record the `db/validator.py` SHA change** (deviation 50) in `validator_amendment_02.md` or a
   short note, so the next person to hash the frozen instrument does not have to re-derive that
   the three changed lines are path strings. Low priority; nothing is blocked by it.
3. **The event rates (deviation 56)** need a mechanism fix or an explicit recorded miss, per
   `synthetic_rules.md` §8's own precedent. 4×–6× over the customer's stated rates is not a band
   quibble.
4. **`supplier_performance_weekly`'s seven all-zero columns** (deviation 54) — decide whether that
   store is live. If it is not, say so; if it is, populate it.
5. **B3 and B5 are client asks and have been for three worlds.** `reports/phase-10.md` §6 items
   3–7 are unchanged by v8 and no generator can close them.
6. **The panel width change (deviation 53)** must be applied per-world when Phase 0 does run —
   25 channels on v8, not 24. Do not borrow v6/v7's width.
7. **Seed 1005's B1 reconciliation** read 99.9944% on a 300-part-plant sample against 100.0000% on
   the other four. ~9 rows. Worth one look before B1 is quoted as exact across all seeds.

---

## 7. What was not done, and why

**`reports/phase-0-1-v8.md` was not written at the time this report was filed.** Stages 3 and 4 —
the loaders, the graph build, the leak assertions and the first training run — were not started,
because the Stage 1 gate rule as given was explicit: if B1, B9 or B12 is NOT CLEARED, stop and
report. B9 is NOT CLEARED on all five seeds.

**That rule was subsequently corrected (deviation 58): B9 does not block Phase 0/1.** Phase 0 and
Phase 1 then ran under the corrected rule, and `reports/phase-0-1-v8.md` now exists. Stopping was
the right call against the rule as written; the rule was wrong, the measurement was not.

Nothing was tuned, no configuration was changed, `ml/configs/shipped.json` is untouched, and no
assertion was disabled or weakened. `db/gen_v6/**`, `db/gen_v7/**`, `db/gen_v8/**`,
`db/validator.py`, `docs/specs/synthetic_rules.md`, `db/dataset_structure.md`,
`docs/specs/model_plan.md` and every existing report in `reports/` are unmodified.

Had the gate passed, §4.3's pre-registered suspicion would have governed the arrival result. It is
worth restating for whoever runs Phase 1 after B9 is resolved: **a large jump in arrival C-index is
both the expected correct result and exactly what leakage looks like**, and on v8 as it stands
today any such number would be the leak.
