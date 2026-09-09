# ChainPilot × Rane — Dataset Structure

**Target:** 7 years of history (84 months), covering all nine use cases in `requirements.md`.

This is the data request to put in front of Rane. Every table below is either something a
normal ERP already holds, or something derived from it by our own pipeline. Tables marked
**[DERIVED]** are ours to compute — do not ask Rane for them.

---

## 0. Two rules that govern the whole schema

**Rule 1 — every changing fact needs a history, not just a current value.**
Stock levels, PO status, supplier capacity and allocation must be logged as a time series of
past readings, never overwritten in place. A single "current stock" column makes it
impossible to reconstruct what was true on a past date, which makes honest backtesting
impossible.

**Rule 2 — every event row needs two timestamps.**

| Column | Meaning |
|---|---|
| `event_ts` | when the thing actually happened |
| `recorded_ts` | when it was entered into their system |

These differ constantly in a manually-updated ERP — batch syncs, data-entry lag, supplier
reporting delays. Training on `event_ts` alone means the model sees facts before anyone
knew them. **This is the single most important ask in the entire document**, and most ERP
exports fail it by default.

Every table below with an `event_ts` also carries `recorded_ts`. Where the two are always
identical (system-generated events), say so explicitly rather than omitting the column.

---

## 1. Scale — planning baseline

Numbers to confirm with Rane in the first data call. These drive storage, training time and
whether each model is feasible at all.

### Entities

| Entity | Active today | Distinct over 7 years | Notes |
|---|---|---|---|
| Business units | 5 | 5 | Brake lining, steering, engine valve, friction, aftermarket |
| Plants | 7 | 7 | Hosur, Chennai, Puducherry, Trichy, Bangalore, Pantnagar, Uttarakhand |
| Finished products (SKU) | 1,200 | 2,500 | Includes discontinued and superseded |
| Purchased parts | 4,500 | 7,000 | The prediction universe for shortage |
| — of which **critical** | 400 | 600 | Rane defines this; it sets the POC scoring population |
| Suppliers | 550 | 750 | Includes churn |
| Supplier sites | 700 | 950 | One supplier may ship from several plants |
| Tier-2 links known | 200 | 250 | Partial visibility is normal — do not expect completeness |
| Customers (OEMs) | 45 | 60 | |
| BOM lines (active) | 22,000 | 40,000 | With validity windows |
| **Sourcing channels** (supplier × part × plant) | 9,000 | 14,000 | **The core modelling entity** |

### Transaction volumes (7 years)

| Table | Approx rows | Notes |
|---|---|---|
| `production_plan` | 300,000 | product × plant × month × plan version |
| `production_actual` | 100,000 | product × plant × month |
| `purchase_orders` | 250,000 | ~3,000/month |
| `po_lines` | 900,000 | ~3.6 lines per PO |
| `po_line_schedules` | 1,800,000 | weekly delivery schedules against PO lines |
| `po_line_revisions` | 400,000 | date and quantity changes |
| `supplier_acknowledgements` | 700,000 | |
| `grn_lines` | 1,400,000 | **more than PO lines — partial receipts are separate rows** |
| `quality_inspections` | 900,000 | |
| `inventory_transactions` | 10,000,000+ | ⚠️ see note below |
| `inventory_snapshots` (weekly) | 2,200,000 | 6,000 part-plant combos × 365 weeks |
| `shortage_events` | 1,500 | |
| `line_stop_events` | 126 | 18/year × 7 |
| `expedite_events` | 840 | 120/year × 7 |

> ⚠️ **`inventory_transactions` is the volume risk.** Daily issue-to-production across 6,000
> part-plant combinations over 1,800 working days is ~10M rows. For the POC, request it
> **aggregated to weekly** unless daily is needed for a specific analysis. Ask for the raw
> table only if weekly proves insufficient.

### Time coverage

| | |
|---|---|
| **Request** | **10 years** (2016–2025) — matches Aptimeta's own deck |
| **Model on** | **7 years** (2019–2025) — see the regime note below |
| Reserve | 2016–2018 as a held-out sanity check, not training data |
| Snapshot grain for model training | **weekly** — gives ~365 timesteps per channel over 7 years |
| Reporting grain for planners | monthly buckets, weekly detail inside the 90-day horizon |
| Minimum acceptable | 4 years. Below that, seasonality is not estimable and channel trajectories are too short for the temporal encoder. |

> **Why request 10 and model on 7.** More years is not automatically better.
> **2016–2025 contains COVID**, and 2020–21 was not a bad year — it was a *different regime*.
> Supplier lead times, allocation behaviour and fill rates from that period do not describe
> how the network behaves now, and training on them as though they were normal actively
> hurts. There is also entity drift: over ten years Rane's supplier base, product mix and
> processes change enough that a 2016 channel and a 2025 channel sharing an ID may not be
> the same thing.
>
> So: **ask for everything, model on the representative window, and mark the breaks
> explicitly** via `calendar.regime_flag` rather than hoping the model works them out.

### Per-table coverage will differ — track it

Do not require every table to hold the full 10 years. Realistic expectation:

| Table | Likely coverage |
|---|---|
| `production_plan`, `production_actual` | 10 years |
| `purchase_orders`, `po_lines`, `grn_lines` | 10 years |
| `inventory_snapshots` | 8 years |
| `supplier_allocation` | 6 years |
| `supplier_capacity` | 5 years |
| `supplier_acknowledgements` | 4–6 years (often introduced with a later ERP upgrade) |
| `quality_inspections` | varies wildly by plant |

Record this explicitly so the pipeline can distinguish "no data" from "no history":

#### `dataset_coverage.csv`

| Column | Type | Description |
|---|---|---|
| `dataset_name` | VARCHAR(50) PK | Table name |
| `earliest_available_date` | DATE | First row present |
| `latest_available_date` | DATE | Last row present |
| `available_years` | DECIMAL(4,1) | Derived |
| `coverage_status` | VARCHAR(20) | `complete` / `partial` / `sparse` / `missing` |
| `known_gaps` | TEXT | JSON list of date ranges with no data |
| `grain` | VARCHAR(20) | `daily` / `weekly` / `monthly` / `event` |
| `row_count` | BIGINT | |
| `notes` | TEXT | e.g. "ack data starts at SAP migration, Apr-2020" |
| `assessed_ts` | TIMESTAMP | |

This matters operationally: a channel whose acknowledgement history starts in 2020 must have
its `ack_gap_ratio` feature masked before that date, not filled with zeros — a zero would
read as "supplier acknowledged nothing," which is the opposite of "we didn't record it."

---

## 2. File index

### Group A — Master data (slowly changing)
| File | One line |
|---|---|
| `business_units.csv` | Rane's operating divisions |
| `plants.csv` | Manufacturing sites |
| `suppliers.csv` | Supplier master |
| `supplier_sites.csv` | Physical dispatch locations per supplier |
| `parts.csv` | Purchased components |
| `products.csv` | Finished goods |
| `customers.csv` | OEM customers |
| `calendar.csv` | Working days, holidays, plant shutdowns |

### Group B — Structure and relationships
| File | One line |
|---|---|
| `bom.csv` | Which parts go into which products, with validity windows |
| `part_plant.csv` | Which plants consume which parts |
| `sourcing_channels.csv` | Approved supplier × part × plant combinations |
| `supplier_capacity.csv` | Stated capacity over time |
| `supplier_allocation.csv` | Planned volume split over time |
| `alternate_sources.csv` | Qualified alternates and qualification status |
| `supplier_upstream.csv` | Known tier-2 dependencies |
| `tooling.csv` | Where tooling physically sits |

### Group C — Planning
| File | One line |
|---|---|
| `production_plan.csv` | Versioned production plan |
| `production_actual.csv` | What was actually built |

### Group D — Procurement transactions
| File | One line |
|---|---|
| `purchase_orders.csv` | PO headers |
| `po_lines.csv` | PO lines — the prediction unit |
| `po_line_schedules.csv` | Weekly delivery schedules against a line |
| `po_line_revisions.csv` | Every date/quantity change with its timestamp |
| `supplier_acknowledgements.csv` | What the supplier committed to |
| `asn.csv` | Advance shipment notices |
| `goods_receipts.csv` | GRN headers |
| `grn_lines.csv` | Receipt lines — **partials as separate rows** |
| `quality_inspections.csv` | Accept/reject against receipts |

### Group E — Inventory
| File | One line |
|---|---|
| `inventory_snapshots.csv` | Periodic stock readings |
| `inventory_transactions.csv` | Every stock movement |

### Group F — Outcomes (the labels)
| File | One line |
|---|---|
| `shortage_events.csv` | Recorded material shortages |
| `line_stop_events.csv` | Production stoppages and their causes |
| `expedite_events.csv` | Emergency actions taken |

### Group G — Cost and commercial
| File | One line |
|---|---|
| `part_costs.csv` | Unit cost over time |
| `product_economics.csv` | Contribution margin per product |
| `supplier_contracts.csv` | Volume commitments, MOQ, lot size |

### Group H — Phase 2 / optional
| File | One line |
|---|---|
| `supplier_quality_ppm.csv` | Defect rates |
| `supplier_audits.csv` | Audit scores |
| `logistics_lanes.csv` | Transport routes |
| `supplier_financials.csv` | Financial health indicators |

### Group I — **[DERIVED]** feature stores (we build these)
| File | One line |
|---|---|
| `part_demand_weekly.csv` | Exploded gross requirements |
| `plan_drift_features.csv` | Plan-vs-actual history by horizon |
| `channel_performance_weekly.csv` | Rolling fill/lead-time stats per channel |
| `supplier_performance_weekly.csv` | Rolling stats per supplier |
| `revealed_capacity_monthly.csv` | Inferred capacity envelope |
| `inventory_position_weekly.csv` | Model-ready stock position |
| `training_labels.csv` | Task labels at each snapshot |
| `snapshots.csv` | Snapshot registry with version stamps |
| `model_outputs.csv` | **All predictions — never in a feature table** |

### Group J — Pipeline metadata
| File | One line |
|---|---|
| `dataset_coverage.csv` | What history actually exists, per table |

---

## 3. Group A — Master data

### `business_units.csv`
Rane's operating divisions. Needed for the corporate → BU → plant rollup.

| Column | Type | Description |
|---|---|---|
| `bu_id` | VARCHAR(20) PK | Business unit identifier |
| `bu_name` | VARCHAR(100) | e.g. "Rane Brake Lining" |
| `bu_code` | VARCHAR(10) | Short code used in reporting |

---

### `plants.csv`
Manufacturing sites. The plant is where shortage is predicted and where lines stop.

| Column | Type | Description |
|---|---|---|
| `plant_id` | VARCHAR(20) PK | Plant identifier |
| `plant_name` | VARCHAR(100) | e.g. "Hosur Plant 1" |
| `bu_id` | VARCHAR(20) FK | Owning business unit |
| `city` | VARCHAR(50) | |
| `state` | VARCHAR(50) | Regional exposure — floods, strikes, power |
| `country` | VARCHAR(50) | |
| `latitude` | DECIMAL(9,6) | For distance and shared-region features |
| `longitude` | DECIMAL(9,6) | |
| `capacity_units_per_day` | INTEGER | Nominal throughput ceiling |
| `commissioned_date` | DATE | Plants opened mid-history need handling |
| `is_active` | BOOLEAN | |

---

### `suppliers.csv`
Supplier master. **Static attributes only** — anything that changes over time goes in a
history table.

| Column | Type | Description |
|---|---|---|
| `supplier_id` | VARCHAR(20) PK | |
| `supplier_name` | VARCHAR(200) | |
| `supplier_group_id` | VARCHAR(20) | Parent company — two "different" suppliers under one group fail together |
| `country` | VARCHAR(50) | One-hot encoded; proxy for regional disruption exposure |
| `state` | VARCHAR(50) | |
| `city` | VARCHAR(50) | |
| `supplier_tier` | VARCHAR(10) | `tier1` / `tier2` |
| `supplier_type` | VARCHAR(30) | `manufacturer` / `trader` / `job_work` — traders behave very differently |
| `business_class` | VARCHAR(30) | `oem_approved` / `local` / `import` |
| `onboarded_date` | DATE | Age is a real reliability feature |
| `is_active` | BOOLEAN | |
| `msme_flag` | BOOLEAN | Small-supplier status; correlates with capacity fragility |
| `payment_terms_days` | INTEGER | Cash-flow stress is a genuine delivery driver |
| ~~`reliability_score`~~ | — | ❌ **Do not supply.** A mutable whole-history figure — using it leaks the outcome. We compute rolling equivalents ourselves. |

---

### `supplier_sites.csv`
A supplier may dispatch from several locations with different performance.

| Column | Type | Description |
|---|---|---|
| `site_id` | VARCHAR(20) PK | |
| `supplier_id` | VARCHAR(20) FK | |
| `site_name` | VARCHAR(200) | |
| `city` | VARCHAR(50) | |
| `state` | VARCHAR(50) | |
| `country` | VARCHAR(50) | |
| `latitude` | DECIMAL(9,6) | |
| `longitude` | DECIMAL(9,6) | |
| `is_active` | BOOLEAN | |

---

### `parts.csv`
Purchased components — the entity shortage is predicted on.

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) PK | |
| `part_number` | VARCHAR(50) | Rane's own numbering |
| `part_name` | VARCHAR(200) | |
| `part_category` | VARCHAR(50) | `casting` / `forging` / `rubber` / `fastener` / `electronic` / `friction_material` — different categories have different baseline risk |
| `material_type` | VARCHAR(50) | Steel, aluminium, polymer — commodity exposure |
| `uom` | VARCHAR(10) | Unit of measure — `EA`, `KG`, `MTR` |
| `is_critical` | BOOLEAN | **Rane's own definition.** Sets the POC scoring population — see §6 |
| `criticality_reason` | VARCHAR(50) | `single_source` / `long_lead` / `safety` / `high_value` |
| `standard_lead_time_days` | INTEGER | Planning lead time |
| `shelf_life_days` | INTEGER | NULL if none; affects how much buffer is possible |
| `is_active` | BOOLEAN | |
| `introduced_date` | DATE | |
| `discontinued_date` | DATE | NULL if active |

---

### `products.csv`
Finished goods sold to OEMs.

| Column | Type | Description |
|---|---|---|
| `product_id` | VARCHAR(30) PK | |
| `product_number` | VARCHAR(50) | |
| `product_name` | VARCHAR(200) | |
| `product_family` | VARCHAR(50) | Drift model buckets by this |
| `bu_id` | VARCHAR(20) FK | |
| `customer_id` | VARCHAR(20) FK | Primary OEM — NULL if multi-customer |
| `is_active` | BOOLEAN | |
| `introduced_date` | DATE | |
| `discontinued_date` | DATE | |

---

### `customers.csv`
OEM customers. Drives the priority weighting in exposure ranking.

| Column | Type | Description |
|---|---|---|
| `customer_id` | VARCHAR(20) PK | |
| `customer_name` | VARCHAR(200) | |
| `customer_tier` | VARCHAR(20) | `strategic` / `standard` / `low` |
| `penalty_per_unit_inr` | DECIMAL(12,2) | Contractual penalty for a missed unit — the exposure calculation needs this |
| `country` | VARCHAR(50) | |
| `is_active` | BOOLEAN | |

---

### `calendar.csv`
One row per date per plant. Underrated and cheap — Indian auto has very strong calendar
effects (Diwali shutdowns, monsoon, March fiscal-year push).

| Column | Type | Description |
|---|---|---|
| `date` | DATE PK | |
| `plant_id` | VARCHAR(20) PK | Composite key with date |
| `is_working_day` | BOOLEAN | |
| `shift_count` | SMALLINT | 1, 2 or 3 — capacity varies with this |
| `is_holiday` | BOOLEAN | |
| `holiday_name` | VARCHAR(100) | |
| `is_shutdown` | BOOLEAN | Planned maintenance shutdown |
| `fiscal_year` | VARCHAR(10) | Indian FY, e.g. `FY24-25` |
| `fiscal_quarter` | SMALLINT | 1–4 |
| `week_of_year` | SMALLINT | |
| `month_of_year` | SMALLINT | Seasonal feature |
| `regime_flag` | VARCHAR(30) | `normal` / `covid` / `demonetisation` / `chip_shortage` / `strike` / `other`. **Structural breaks must be labelled, not averaged through.** Lets the pipeline down-weight or exclude explicitly |
| `regime_note` | VARCHAR(200) | Free text describing the break |

---

## 4. Group B — Structure

### `bom.csv`
Which parts go into which products. **Validity windows are mandatory** — the model must see
the BOM as it stood at each past date, not as it stands today.

| Column | Type | Description |
|---|---|---|
| `bom_id` | VARCHAR(30) PK | |
| `product_id` | VARCHAR(30) FK | Parent |
| `part_id` | VARCHAR(30) FK | Child |
| `qty_per_unit` | DECIMAL(12,4) | How many of this part per one product |
| `scrap_factor` | DECIMAL(6,4) | Expected scrap, e.g. `0.02` for 2% |
| `bom_level` | SMALLINT | 1 = direct, 2+ = sub-assembly. Multi-level BOMs recurse |
| `parent_part_id` | VARCHAR(30) | For multi-level: the sub-assembly this sits under. NULL at level 1 |
| `effective_from` | DATE | **Mandatory** |
| `effective_to` | DATE | NULL = still current. **Mandatory** |
| `is_phantom` | BOOLEAN | Phantom assemblies pass through without stocking |

---

### `part_plant.csv`
Which plants consume which parts, and their local policy.

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) PK | |
| `plant_id` | VARCHAR(20) PK | Composite key |
| `safety_stock_qty` | INTEGER | Policy buffer |
| `reorder_point_qty` | INTEGER | |
| `min_order_qty` | INTEGER | |
| `lot_size` | INTEGER | Rounding multiple — makes allocation an integer program |
| `planning_lead_time_days` | INTEGER | |
| `abc_class` | CHAR(1) | A/B/C by value |
| `xyz_class` | CHAR(1) | X/Y/Z by demand variability |
| `effective_from` | DATE | Policies change; keep the history |
| `effective_to` | DATE | |

---

### `sourcing_channels.csv`
**The single most important structural table.** A channel is (supplier × part × plant) — the
entity that has a multi-year track record and that the temporal encoder runs over.

| Column | Type | Description |
|---|---|---|
| `channel_id` | VARCHAR(40) PK | Derived: `supplier_id`+`part_id`+`plant_id` |
| `supplier_id` | VARCHAR(20) FK | |
| `site_id` | VARCHAR(20) FK | Dispatching location |
| `part_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | Receiving plant |
| `is_approved` | BOOLEAN | PPAP complete — allocation cannot move volume to an unapproved channel |
| `ppap_date` | DATE | When qualification completed |
| `approval_status` | VARCHAR(20) | `approved` / `conditional` / `development` / `blocked` |
| `contracted_lead_time_days` | INTEGER | |
| `transport_mode` | VARCHAR(20) | `road` / `rail` / `sea` / `air` |
| `transport_distance_km` | INTEGER | |
| `effective_from` | DATE | |
| `effective_to` | DATE | |

---

### `supplier_capacity.csv`
Stated capacity over time. ⚠️ **The riskiest table in the request** — often a stale
contractual figure rather than a real number. Ask, but plan to derive capacity ourselves.

| Column | Type | Description |
|---|---|---|
| `supplier_id` | VARCHAR(20) FK | |
| `part_id` | VARCHAR(30) FK | NULL = supplier-level aggregate |
| `capacity_qty_per_month` | INTEGER | Stated monthly ceiling |
| `capacity_basis` | VARCHAR(30) | `contracted` / `declared` / `audited` / `estimated` — **critical**: tells us how much to trust it |
| `shift_basis` | SMALLINT | How many shifts the figure assumes |
| `effective_from` | DATE | |
| `effective_to` | DATE | |
| `recorded_ts` | TIMESTAMP | When this figure was entered |

---

### `supplier_allocation.csv`
The planned volume split across suppliers for a part. The decision variable in use case 8.

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `supplier_id` | VARCHAR(20) FK | |
| `allocation_pct` | DECIMAL(5,2) | Share of volume, 0–100 |
| `effective_from` | DATE | |
| `effective_to` | DATE | |
| `changed_by` | VARCHAR(50) | Who made the change |
| `change_reason` | VARCHAR(100) | Historical allocation changes are quasi-experiments — very valuable |
| `recorded_ts` | TIMESTAMP | |

---

### `alternate_sources.csv`
Which other suppliers *could* make this part, and how long it would take to switch.

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) FK | |
| `supplier_id` | VARCHAR(20) FK | Alternate supplier |
| `plant_id` | VARCHAR(20) FK | |
| `qualification_status` | VARCHAR(20) | `qualified` / `in_progress` / `potential` |
| `qualification_lead_days` | INTEGER | How long to make them usable — a hard constraint in allocation |
| `ramp_rate_pct_per_month` | DECIMAL(5,2) | Max share increase per month |
| `cost_delta_pct` | DECIMAL(6,2) | Price difference vs incumbent, ± |
| `effective_from` | DATE | |
| `effective_to` | DATE | |

---

### `supplier_upstream.csv`
Known tier-2 relationships. Expect this to be sparse — partial visibility is normal, and even
a partial map beats none, because it's what exposes correlated failure.

| Column | Type | Description |
|---|---|---|
| `supplier_id` | VARCHAR(20) FK | Tier-1 |
| `upstream_supplier_id` | VARCHAR(20) FK | Tier-2 |
| `part_id` | VARCHAR(30) FK | What flows on this link. NULL if general |
| `dependency_type` | VARCHAR(30) | `raw_material` / `sub_assembly` / `process` / `logistics` |
| `criticality` | VARCHAR(10) | `high` / `medium` / `low` |
| `is_sole_source` | BOOLEAN | Sole-sourced tier-2 is the classic hidden single point of failure |
| `confidence` | VARCHAR(10) | `confirmed` / `reported` / `inferred` — how we know this link exists |
| `known_since` | DATE | |

> **Tier-2 is core scope, not Phase 2** — it is what lets the simulation model *correlated
> failure*, the one thing a per-supplier risk table structurally cannot represent. But be
> explicit about the confidence difference in every output:
>
> | | |
> |---|---|
> | **Tier-1 risk** | High-confidence operational signal, backed by transaction history |
> | **Tier-2 risk** | Network propagation / early-warning signal, backed by partial visibility |
>
> Never present tier-2 exposure as though visibility were complete. Partial is genuinely
> useful; claiming completeness is how the number gets discredited the first time a
> disruption arrives through an unmapped link.

---

### `tooling.csv`
Where tooling physically sits. A hard constraint on allocation — you cannot move volume
away from a supplier whose plant holds the dies.

| Column | Type | Description |
|---|---|---|
| `tool_id` | VARCHAR(30) PK | |
| `part_id` | VARCHAR(30) FK | |
| `supplier_id` | VARCHAR(20) FK | Where it currently sits |
| `owned_by` | VARCHAR(20) | `rane` / `supplier` |
| `is_transferable` | BOOLEAN | |
| `transfer_lead_days` | INTEGER | |
| `duplicate_exists` | BOOLEAN | A second tool elsewhere removes the constraint |

---

## 5. Group C — Planning

### `production_plan.csv`
**Versioned.** Every re-plan creates new rows; old rows are never overwritten. This is what
makes the drift model possible.

| Column | Type | Description |
|---|---|---|
| `plan_id` | VARCHAR(30) PK | |
| `plan_version` | INTEGER | Increments on each re-plan |
| `plan_date` | DATE | **When this plan was made** — the horizon anchor |
| `product_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `target_period` | DATE | Month/week being planned for |
| `period_grain` | VARCHAR(10) | `month` / `week` |
| `planned_qty` | INTEGER | |
| `plan_type` | VARCHAR(20) | `annual` / `quarterly` / `rolling` / `firm` |
| `is_firm` | BOOLEAN | Firm periods behave very differently from tentative ones |
| `recorded_ts` | TIMESTAMP | |

> **Horizon is derived:** `horizon_days = target_period − plan_date`. The drift model
> conditions on this, and it's the whole reason their accuracy decays from month 1 to month 3.

---

### `production_actual.csv`
What was actually built.

| Column | Type | Description |
|---|---|---|
| `product_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `period` | DATE | |
| `period_grain` | VARCHAR(10) | `month` / `week` |
| `actual_qty` | INTEGER | Good units produced |
| `scrapped_qty` | INTEGER | |
| `rework_qty` | INTEGER | |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

## 6. Group D — Procurement transactions

### `purchase_orders.csv`

| Column | Type | Description |
|---|---|---|
| `po_id` | VARCHAR(30) PK | |
| `po_number` | VARCHAR(50) | |
| `supplier_id` | VARCHAR(20) FK | |
| `site_id` | VARCHAR(20) FK | |
| `plant_id` | VARCHAR(20) FK | Ship-to |
| `po_type` | VARCHAR(20) | `standard` / `schedule_agreement` / `blanket` / `emergency` |
| `currency` | CHAR(3) | |
| `incoterm` | VARCHAR(10) | |
| `created_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |
| `status` | VARCHAR(20) | Current status — **excluded from features**, reconstructed from revisions |

---

### `po_lines.csv`
**The prediction unit for fill rate and arrival timing.**

| Column | Type | Description |
|---|---|---|
| `po_line_id` | VARCHAR(30) PK | |
| `po_id` | VARCHAR(30) FK | |
| `line_number` | SMALLINT | |
| `part_id` | VARCHAR(30) FK | |
| `channel_id` | VARCHAR(40) FK | Derived join to `sourcing_channels` |
| `qty_ordered` | INTEGER | Denominator of fill rate |
| `unit_price` | DECIMAL(12,4) | |
| `original_promise_date` | DATE | Baseline for lateness |
| `current_promise_date` | DATE | After revisions — **as-of reconstruction required** |
| `requested_date` | DATE | What Rane asked for |
| `created_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |
| ~~`qty_received`~~ | — | ❌ **Do not supply as a column.** It is part of the label. Comes from `grn_lines` |
| ~~`status`~~ | — | ❌ Current-state field; reconstruct from `po_line_revisions` |

---

### `po_line_schedules.csv`
For schedule agreements — a monthly requirement broken into weekly drops. Standard in Indian
auto. Feeds use case 7.

| Column | Type | Description |
|---|---|---|
| `schedule_id` | VARCHAR(30) PK | |
| `po_line_id` | VARCHAR(30) FK | |
| `schedule_date` | DATE | Delivery date for this drop |
| `scheduled_qty` | INTEGER | |
| `schedule_version` | INTEGER | Schedules get revised constantly |
| `released_ts` | TIMESTAMP | When this version was sent to the supplier |
| `recorded_ts` | TIMESTAMP | |

---

### `po_line_revisions.csv`
Every change to a PO line, with when it happened and when it was recorded. This is how
"promise date as of a past date" gets reconstructed.

| Column | Type | Description |
|---|---|---|
| `revision_id` | VARCHAR(30) PK | |
| `po_line_id` | VARCHAR(30) FK | |
| `revision_number` | SMALLINT | |
| `field_changed` | VARCHAR(30) | `promise_date` / `qty` / `price` / `status` |
| `old_value` | VARCHAR(50) | |
| `new_value` | VARCHAR(50) | |
| `initiated_by` | VARCHAR(20) | `supplier` / `buyer` / `system` — **critical**: a buyer-initiated cut is not a supplier failure |
| `reason_code` | VARCHAR(50) | |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

### `supplier_acknowledgements.csv`
What the supplier committed to. **The strongest leading indicator in the entire dataset** —
a supplier confirming 800 against an order of 1,000 tells you weeks before goods move.

| Column | Type | Description |
|---|---|---|
| `ack_id` | VARCHAR(30) PK | |
| `po_line_id` | VARCHAR(30) FK | |
| `ack_qty` | INTEGER | Quantity confirmed |
| `ack_date` | DATE | Date confirmed for |
| `ack_status` | VARCHAR(20) | `full` / `partial` / `rejected` / `date_change` |
| `event_ts` | TIMESTAMP | When the supplier sent it |
| `recorded_ts` | TIMESTAMP | When Rane entered it |

---

### `asn.csv`
Advance shipment notices. Optional but valuable — narrows the arrival window sharply.

| Column | Type | Description |
|---|---|---|
| `asn_id` | VARCHAR(30) PK | |
| `po_line_id` | VARCHAR(30) FK | |
| `dispatched_qty` | INTEGER | |
| `dispatch_ts` | TIMESTAMP | |
| `expected_arrival_date` | DATE | |
| `transport_mode` | VARCHAR(20) | |
| `vehicle_id` | VARCHAR(30) | |
| `recorded_ts` | TIMESTAMP | |

---

### `goods_receipts.csv`

| Column | Type | Description |
|---|---|---|
| `grn_id` | VARCHAR(30) PK | |
| `grn_number` | VARCHAR(50) | |
| `supplier_id` | VARCHAR(20) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `receipt_ts` | TIMESTAMP | Physical arrival |
| `recorded_ts` | TIMESTAMP | GRN posted |

---

### `grn_lines.csv`
⚠️ **Partial receipts must be separate rows.** If the ERP stores one summed `qty_received`
against a PO line, arrival timing is unrecoverable and use case 4 collapses.

| Column | Type | Description |
|---|---|---|
| `grn_line_id` | VARCHAR(30) PK | |
| `grn_id` | VARCHAR(30) FK | |
| `po_line_id` | VARCHAR(30) FK | |
| `part_id` | VARCHAR(30) FK | |
| `qty_received` | INTEGER | **This receipt only, not cumulative** |
| `receipt_sequence` | SMALLINT | 1st, 2nd, 3rd partial against this line |
| `is_final_receipt` | BOOLEAN | Line closed after this |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

### `quality_inspections.csv`
A part received and then rejected is short just the same.

| Column | Type | Description |
|---|---|---|
| `inspection_id` | VARCHAR(30) PK | |
| `grn_line_id` | VARCHAR(30) FK | |
| `qty_inspected` | INTEGER | |
| `qty_accepted` | INTEGER | |
| `qty_rejected` | INTEGER | |
| `qty_deviation_accepted` | INTEGER | Accepted under concession |
| `rejection_reason` | VARCHAR(100) | |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

## 7. Group E — Inventory

### `inventory_snapshots.csv`
Periodic stock readings. **Weekly is the requested grain** for the POC.

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `snapshot_date` | DATE | |
| `qty_on_hand` | INTEGER | |
| `qty_blocked` | INTEGER | Quarantine / rejected, unusable |
| `qty_in_transit` | INTEGER | Dispatched, not received |
| `qty_reserved` | INTEGER | Committed to a production order |
| `qty_available` | INTEGER | Derived: on_hand − blocked − reserved |
| `event_ts` | TIMESTAMP | Physical count moment |
| `recorded_ts` | TIMESTAMP | |

---

### `inventory_transactions.csv`
Every stock movement. ⚠️ Highest-volume table — request weekly aggregation first.

| Column | Type | Description |
|---|---|---|
| `txn_id` | VARCHAR(30) PK | |
| `part_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `txn_type` | VARCHAR(30) | `receipt` / `issue_to_production` / `scrap` / `return` / `transfer_in` / `transfer_out` / `adjustment` |
| `qty` | INTEGER | Signed: positive in, negative out |
| `reference_id` | VARCHAR(30) | Links to GRN, production order, or transfer |
| `from_plant_id` | VARCHAR(20) | For transfers — **inter-plant pulls are a hidden consumption stream** |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

## 8. Group F — Outcomes (labels)

### `shortage_events.csv`
The label for use case 5.

| Column | Type | Description |
|---|---|---|
| `shortage_id` | VARCHAR(30) PK | |
| `part_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `shortage_qty` | INTEGER | Units short |
| `shortage_start_ts` | TIMESTAMP | |
| `shortage_end_ts` | TIMESTAMP | |
| `root_cause` | VARCHAR(50) | `supplier_delay` / `supplier_short` / `quality_reject` / `demand_spike` / `plan_change` / `logistics` |
| `responsible_supplier_id` | VARCHAR(20) | NULL if internal |
| `resolution_action` | VARCHAR(50) | `expedite` / `alternate_source` / `substitute` / `reschedule` |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

### `line_stop_events.csv`
The most expensive outcome and the rarest — ~18/year. Precious labels; treat them carefully.

| Column | Type | Description |
|---|---|---|
| `stop_id` | VARCHAR(30) PK | |
| `plant_id` | VARCHAR(20) FK | |
| `line_id` | VARCHAR(30) | |
| `product_id` | VARCHAR(30) FK | What could not be built |
| `part_id` | VARCHAR(30) FK | The missing part, if material-caused |
| `stop_start_ts` | TIMESTAMP | |
| `stop_end_ts` | TIMESTAMP | |
| `duration_minutes` | INTEGER | |
| `units_lost` | INTEGER | The headline business number |
| `cause_category` | VARCHAR(50) | `material` / `machine` / `manpower` / `quality` / `power` |
| `responsible_supplier_id` | VARCHAR(20) | |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

### `expedite_events.csv`
Emergency actions — 120/year today. Each one is a *near miss* that didn't become a line
stop, which makes them far more numerous and useful as labels than line stops.

| Column | Type | Description |
|---|---|---|
| `expedite_id` | VARCHAR(30) PK | |
| `part_id` | VARCHAR(30) FK | |
| `plant_id` | VARCHAR(20) FK | |
| `supplier_id` | VARCHAR(20) FK | |
| `expedite_type` | VARCHAR(30) | `air_freight` / `dedicated_truck` / `supplier_overtime` / `inter_plant_transfer` |
| `qty_expedited` | INTEGER | |
| `premium_cost_inr` | DECIMAL(12,2) | Cost avoided is a headline KPI |
| `triggered_by` | VARCHAR(50) | |
| `event_ts` | TIMESTAMP | |
| `recorded_ts` | TIMESTAMP | |

---

## 9. Group G — Cost and commercial

### `part_costs.csv`
| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) FK | |
| `supplier_id` | VARCHAR(20) FK | Prices differ by supplier — the allocation cost term |
| `unit_cost_inr` | DECIMAL(12,4) | |
| `freight_cost_inr` | DECIMAL(12,4) | |
| `effective_from` | DATE | |
| `effective_to` | DATE | |

### `product_economics.csv`
| Column | Type | Description |
|---|---|---|
| `product_id` | VARCHAR(30) FK | |
| `selling_price_inr` | DECIMAL(12,2) | |
| `contribution_margin_inr` | DECIMAL(12,2) | **Per-unit margin — this is what turns units at risk into ₹ at risk** |
| `effective_from` | DATE | |
| `effective_to` | DATE | |

### `supplier_contracts.csv`
Hard constraints for the allocation optimiser.

| Column | Type | Description |
|---|---|---|
| `contract_id` | VARCHAR(30) PK | |
| `supplier_id` | VARCHAR(20) FK | |
| `part_id` | VARCHAR(30) FK | |
| `min_volume_commitment` | INTEGER | Annual minimum; breaching it costs money |
| `max_volume_cap` | INTEGER | |
| `moq` | INTEGER | Minimum order quantity |
| `lot_size` | INTEGER | Rounding multiple |
| `price_break_qty` | INTEGER | |
| `penalty_clause_inr` | DECIMAL(12,2) | |
| `valid_from` | DATE | |
| `valid_to` | DATE | |

---

## 10. Group H — Phase 2 / optional

### `supplier_quality_ppm.csv`
| Column | Type | Description |
|---|---|---|
| `supplier_id` | VARCHAR(20) FK | |
| `part_id` | VARCHAR(30) FK | |
| `period` | DATE | Monthly |
| `parts_supplied` | INTEGER | |
| `parts_rejected` | INTEGER | |
| `ppm` | INTEGER | Defects per million |
| `recorded_ts` | TIMESTAMP | |

### `supplier_audits.csv`
| Column | Type | Description |
|---|---|---|
| `audit_id` | VARCHAR(30) PK | |
| `supplier_id` | VARCHAR(20) FK | |
| `audit_date` | DATE | |
| `audit_type` | VARCHAR(30) | `process` / `system` / `product` |
| `score` | DECIMAL(5,2) | |
| `major_ncs` | SMALLINT | Major non-conformances |
| `minor_ncs` | SMALLINT | |
| `recorded_ts` | TIMESTAMP | |

### `logistics_lanes.csv`
| Column | Type | Description |
|---|---|---|
| `lane_id` | VARCHAR(30) PK | |
| `origin_site_id` | VARCHAR(20) FK | |
| `dest_plant_id` | VARCHAR(20) FK | |
| `transport_mode` | VARCHAR(20) | |
| `distance_km` | INTEGER | |
| `standard_transit_days` | DECIMAL(5,2) | |
| `carrier_id` | VARCHAR(20) | |
| `via_checkpoint` | VARCHAR(50) | Port, border, toll corridor — **shared checkpoints create correlated failure** |

### `supplier_financials.csv`
| Column | Type | Description |
|---|---|---|
| `supplier_id` | VARCHAR(20) FK | |
| `period` | DATE | Annual or quarterly |
| `revenue_inr` | DECIMAL(15,2) | |
| `rane_share_of_revenue_pct` | DECIMAL(5,2) | **Dependency runs both ways** — a supplier for whom Rane is 60% of revenue behaves differently |
| `credit_rating` | VARCHAR(10) | |
| `days_payable_outstanding` | INTEGER | |

---

## 11. Group I — [DERIVED] feature stores

**We compute these. Do not request them from Rane.** Listed so the pipeline is specified.

> ### ⚠️ Feature tables never contain model outputs
>
> A predicted value — `arrival_probability`, `capacity_strain`, `network_risk_score`,
> `inventory_risk` — must **never** sit in a feature table. It belongs in
> `model_outputs.csv` (§11.9).
>
> This is not tidiness. It is how leakage returns after a careful pipeline is built:
> someone joins the feature store, the score is already sitting there, and the model trains
> on its own prediction. Every column in the tables below is either an observation or a
> deterministic function of observations — nothing here is predicted.

### `part_demand_weekly.csv`
Output of the BOM explosion (use case 1).

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) | |
| `plant_id` | VARCHAR(20) | |
| `week_start` | DATE | |
| `as_of_date` | DATE | Which snapshot this was computed at |
| `gross_requirement_p50` | INTEGER | Median demand |
| `gross_requirement_p90` | INTEGER | Conservative case for the simulation |
| `horizon_days` | INTEGER | Distance from `as_of_date` |
| `driving_products` | TEXT | JSON list of contributing products — needed for the exposure rollup |

### `channel_performance_weekly.csv`
Rolling statistics per sourcing channel — the temporal encoder's input sequence.
**20 columns.** About 82% of channel-weeks are empty, and three of the design decisions below
exist only because of that.

| Column | Type | Description |
|---|---|---|
| `channel_id` | VARCHAR(40) | |
| `week_start` | DATE | |
| `qty_ordered` | INTEGER | |
| `qty_received` | INTEGER | |
| `is_active_week` | BOOLEAN | `qty_ordered > 0`. The sequence mask, and a feature in its own right |
| `fill_rate` | DECIMAL(6,4) | accepted ÷ ordered. NULL on an empty week — **never 0** |
| `fill_rate_last4` | DECIMAL(6,4) | Mean fill over the last **4 active weeks** — not 4 calendar weeks. See the note below |
| `fill_rate_last13` | DECIMAL(6,4) | Last **13 active weeks** (median 85 calendar weeks) |
| `fill_rate_last52` | DECIMAL(6,4) | Last **52 active weeks**; for 59% of channels this is a lifetime mean |
| `lead_time_actual_days` | DECIMAL(6,2) | |
| `lead_time_ratio` | DECIMAL(6,4) | actual ÷ contracted |
| `otd_rate_last13` | DECIMAL(6,4) | On-time delivery over the last 13 **receipts** |
| `ack_gap_ratio` | DECIMAL(6,4) | acknowledged ÷ ordered. NULL on an empty week |
| `revision_count` | SMALLINT | Promise-date and quantity changes |
| `load_ratio` | DECIMAL(6,4) | **Key capacity feature:** ordered ÷ trailing-52-calendar-week max. NULL on an empty week |
| `active_weeks_in_52` | SMALLINT | Active weeks in the trailing 52 **calendar** weeks. This is what tells you the calendar span the `_lastN` columns actually cover |
| `days_since_last_short` | INTEGER | |
| `reporting_lag_days` | DECIMAL(6,4) | `recorded_ts − event_ts`, averaged over the week's events. NULL where nothing was posted (~61% of rows) |
| `weeks_since_last_activity` | INTEGER | Weeks since the channel last ordered |
| `weeks_since_last_receipt` | INTEGER | Weeks since the last GRN. **NULL means never**, not "long ago" |

> **`staleness_days` was removed, not renamed.** It meant reporting lag on active weeks and
> channel inactivity on empty ones — opposite meanings in one integer, which no learned gate
> can separate. Pooled it read P50 12 / P90 838 days, and that tail was inactivity.
> `reporting_lag_days` reads P50 1.3 / P90 5.7 / P99 14 / max 96. (An earlier revision quoted
> P90 2 / max 6 for it; that ceiling was a builder artefact — a row posted after its own week
> closed was discarded, and a 7-day week cannot hold a longer lag. Fixed.) `reporting_lag_days` and
> `weeks_since_last_activity` are the two halves. It was deleted rather than redefined so that
> analysis written against the old meaning fails loudly instead of silently changing subject.
>
> `inventory_position_weekly.staleness_days` is a **different** column and is unaffected: it
> is the age of the inventory reading feeding that row, is bounded (P50 14, P90 28, max 126
> days), and carries one meaning only.

> **The `_lastN` columns count active weeks, and the names say so.** They were `_4w`/`_13w`/
> `_52w`, which claimed a calendar window they never had: the builder appends only in weeks
> the channel ordered, so `fill_rate_last13` spans a **median of 85 calendar weeks**, not 13.
> The window still counts active weeks deliberately — a 13-calendar-week window at this
> cadence holds one or two observations and is mostly noise — but `active_weeks_in_52` is
> emitted alongside so the time scale is visible rather than assumed.

> **An empty week is not a zero.** A `fill_rate` of 0 is a total delivery failure; an empty
> week is no information. Every instantaneous column is NULL rather than 0 on an empty week,
> and the rolling ones carry the last known value forward with
> `weeks_since_last_activity` giving its age. `load_ratio` used to be emitted as 0 on empty
> weeks, which put 4.3M fabricated "zero load" rows — 73% of the table — into the store.

### `supplier_performance_weekly.csv`
Same 20 columns with `supplier_id` in place of `channel_id`, plus `active_channel_count`
(**21 columns**), aggregated to supplier level. Lets thin channels borrow from their parent.

`reporting_lag_days` is always NULL here: averaging posting lag across plants would hide the
one plant that posts late, which is the whole point of the `plant_data_discipline` mechanism.
Read it from the channel table.

### `plan_drift_features.csv`
Materialises the plan-vs-actual history the drift model trains on.

| Column | Type | Description |
|---|---|---|
| `product_id` | VARCHAR(30) | |
| `plant_id` | VARCHAR(20) | |
| `target_period` | DATE | |
| `plan_date` | DATE | When the plan version was made |
| `horizon_days` | INTEGER | `target_period − plan_date` — **the conditioning variable** |
| `plan_version` | INTEGER | |
| `original_plan_qty` | INTEGER | First version for this period |
| `latest_plan_qty` | INTEGER | Version current as of `plan_date` |
| `actual_qty` | INTEGER | What was built (NULL for future periods) |
| `drift_ratio` | DECIMAL(8,4) | `actual ÷ planned` — the target |
| `plan_revision_count` | SMALLINT | |
| `plan_volatility` | DECIMAL(8,4) | Std dev of plan quantity across versions |
| `is_firm` | BOOLEAN | Firm and tentative periods are separate populations |
| `regime_flag` | VARCHAR(30) | Joined from `calendar` |
| `recorded_ts` | TIMESTAMP | |

### `inventory_position_weekly.csv`
Model-ready stock position — the opening balance for the shortage simulation.

| Column | Type | Description |
|---|---|---|
| `part_id` | VARCHAR(30) | |
| `plant_id` | VARCHAR(20) | |
| `week_start` | DATE | |
| `as_of_date` | DATE | Snapshot this was computed at |
| `qty_on_hand` | INTEGER | |
| `qty_blocked` | INTEGER | Quarantine / rejected |
| `qty_reserved` | INTEGER | Committed to a production order |
| `qty_in_transit` | INTEGER | Dispatched, not received |
| `qty_available` | INTEGER | Derived: on_hand − blocked − reserved |
| `open_po_qty` | INTEGER | Ordered, not yet received |
| `safety_stock_qty` | INTEGER | From `part_plant` |
| `days_of_supply` | DECIMAL(6,2) | `qty_available ÷ mean weekly consumption` |
| `consumption_4w` / `_13w` | INTEGER | Trailing actual issues to production |
| `inter_plant_in_4w` | INTEGER | **Transfers in — a hidden consumption stream elsewhere** |
| `inter_plant_out_4w` | INTEGER | Transfers out |
| `staleness_days` | INTEGER | Age of the newest reading feeding this row |
| `recorded_ts` | TIMESTAMP | |

*No `inventory_risk` column.* That is a model output and lives in `model_outputs.csv`.

### `revealed_capacity_monthly.csv`
| Column | Type | Description |
|---|---|---|
| `supplier_id` | VARCHAR(20) | |
| `part_id` | VARCHAR(30) | |
| `month` | DATE | |
| `max_delivered_12m` | INTEGER | Hard lower bound on true capacity |
| `p95_delivered_12m` | INTEGER | |
| `constrained_month_flag` | BOOLEAN | Fill < 1 or lead time above baseline — **only these months carry capacity evidence** |
| `declared_capacity_qty` | INTEGER | The stated figure, carried alongside for comparison |
| `capacity_basis` | VARCHAR(30) | From `supplier_capacity` — how much to trust the declared number |
| `revealed_capacity_est` | INTEGER | Our inferred envelope (deterministic order statistic, **not** a prediction) |
| `capacity_utilisation_observed` | DECIMAL(6,4) | `delivered ÷ revealed_capacity_est` — historical, not forecast |
| `evidence_strength` | DECIMAL(4,2) | 0–1; how much constrained data backs the estimate |

*No `capacity_strain` column.* Strain at a future horizon is the UC2 model **target** and
output — it lives in `training_labels.csv` and `model_outputs.csv`, never here.

### `training_labels.csv`
The strict separation: **features are what was known at t₀; the label is what happened
after.** One row per (snapshot, entity, task).

| Column | Type | Description |
|---|---|---|
| `label_id` | VARCHAR(40) PK | |
| `snapshot_id` | VARCHAR(30) FK | |
| `snapshot_date` | DATE | The t₀ |
| `entity_type` | VARCHAR(20) | `po_line` / `channel` / `part_plant` / `supplier` / `product_plant` |
| `entity_id` | VARCHAR(40) | |
| `task` | VARCHAR(30) | See the task table below |
| `horizon_days` | INTEGER | |
| `label_window_start` | DATE | Start of the outcome window — **must be > snapshot_date** |
| `label_window_end` | DATE | |
| `label_value` | DECIMAL(12,4) | **Continuous.** Units or a ratio, never a flag |
| `label_censored` | BOOLEAN | True where the outcome is not yet observed — the hazard model consumes these rows rather than dropping them |
| `censor_time` | INTEGER | For censored rows: how many weeks observed so far |
| `label_definition` | VARCHAR(200) | Exact rule used, for audit |
| `label_version` | VARCHAR(10) | Label definitions change; keep them versioned |
| `created_ts` | TIMESTAMP | |

**Task definitions — all continuous, deliberately.**

| `task` | Entity | `label_value` | Notes |
|---|---|---|---|
| `fill_rate` | `po_line` | `qty_accepted ÷ qty_ordered` ∈ [0,1] | Per line, not aggregated to a forward window |
| `arrival_week` | `po_line` | Week index of first receipt | **Censored for open lines** — that is the point |
| `capacity_strain` | `channel` | Observed `fill_rate` and `lead_time_ratio` in the window | Strain is observable; capacity is not |
| `demand_drift` | `product_plant` | `actual ÷ planned` | Conditioned on `horizon_days` |
| `shortage_qty` | `part_plant` | Units short vs safety stock | ⚠️ **Cross-check only** — see below |

> ⚠️ **`shortage_qty` is not the primary shortage model.** Shortage is produced by the Monte
> Carlo simulation (UC5), which is auditable and composes into the what-if. The learned head
> exists as a *diagnostic*: where it disagrees sharply with the simulation, data is missing —
> an undocumented consumption stream, an inter-plant pull, an informal substitution. Do not
> let it become the primary output.
>
> **And no binary labels anywhere.** No `shortage_next_30d ∈ {0,1}`, no
> `arrival_within_7d ∈ {0,1}`. At these base rates a binary target produces a model with
> ~25% precision that nobody can act on — measured, not assumed, in HADES v3's
> `accuracy.md`. Every label is a quantity so that every output is a quantity, and quantities
> aggregate correctly across the four-level rollup where probabilities do not.

### `snapshots.csv`
| Column | Type | Description |
|---|---|---|
| `snapshot_id` | VARCHAR(30) PK | |
| `as_of_ts` | TIMESTAMP | The t₀ for this snapshot |
| `data_cutoff_ts` | TIMESTAMP | Hard bound: no row with `recorded_ts >` this may enter |
| `horizon_days` | INTEGER | |
| `node_counts` | TEXT | JSON |
| `edge_counts` | TEXT | JSON |
| `feature_spec_version` | VARCHAR(10) | Which feature definitions were used |
| `dataset_version` | VARCHAR(20) | Which extract of the source data |
| `label_version` | VARCHAR(10) | Which label definitions |
| `code_commit` | VARCHAR(40) | Git SHA of the pipeline |
| `created_ts` | TIMESTAMP | |

Together these five version fields make a training run **reproducible**: given a
`snapshot_id`, the exact feature set, label set, source extract and code can all be
recovered. Without them, "why did the numbers change?" is unanswerable six months in.

### `model_outputs.csv`
**Every prediction lands here — never in a feature table.**

| Column | Type | Description |
|---|---|---|
| `output_id` | VARCHAR(40) PK | |
| `snapshot_id` | VARCHAR(30) FK | |
| `model_name` | VARCHAR(50) | `capacity_strain` / `fill_rate` / `arrival_timing` / `shortage_sim` / `allocation_opt` |
| `model_version` | VARCHAR(20) | |
| `entity_type` | VARCHAR(20) | |
| `entity_id` | VARCHAR(40) | |
| `horizon_days` | INTEGER | |
| `point_estimate` | DECIMAL(14,4) | P50 |
| `p10` / `p90` | DECIMAL(14,4) | The band |
| `distribution` | TEXT | JSON — full CDF for binned heads, hazard vector for timing |
| `confidence_basis` | VARCHAR(30) | `observed` / `interpolated` / `extrapolated` — drives the UI shading |
| `evidence_strength` | DECIMAL(4,2) | 0–1 |
| `created_ts` | TIMESTAMP | |

Keeping outputs separate also gives you the backtest for free: join `model_outputs` to
`training_labels` on (`snapshot_id`, `entity_id`, `task`) and every historical prediction
sits beside what actually happened.

---

## 12. Model → data dependency matrix

The bridge between this document and `model_plan.md`. For a data engineer building the
pipeline, this is the page that matters.

| Model | Grain | Raw inputs | Derived inputs | Label | Horizon | Output |
|---|---|---|---|---|---|---|
| **BOM explosion**<br>`not ML` | part × plant × week | `production_plan`, `bom`, `products`, `part_plant`, `calendar` | — | — | 13 weeks | Gross requirements |
| **Plan drift**<br>`quantiles → LightGBM` | product family × plant × horizon | `production_plan`, `production_actual`, `calendar` | `plan_drift_features` | `demand_drift` | 30/60/90d | P50 / P90 demand |
| **Revealed capacity**<br>`not ML` | supplier × part × month | `grn_lines`, `po_lines`, `supplier_capacity` | `channel_performance_weekly` | — | historical | Capacity envelope + `evidence_strength` |
| **Capacity strain**<br>`temporal-SHARE → quantile` | channel (supplier × part × plant) | `po_lines`, `grn_lines`, `supplier_acknowledgements`, `sourcing_channels`, `supplier_allocation`, `supplier_upstream` | `channel_performance_weekly`, `supplier_performance_weekly`, `revealed_capacity_monthly`, `part_demand_weekly` | `capacity_strain` | 30/60/90d | Utilisation P10/P50/P90 + days-to-exceed |
| **Fill rate**<br>`temporal-SHARE → binned CDF` | PO line | `po_lines`, `grn_lines`, `supplier_acknowledgements`, `po_line_revisions`, `quality_inspections` | `channel_performance_weekly` | `fill_rate` | to promise date | Full distribution over [0,1] + expected shortfall in units |
| **Arrival timing**<br>`temporal-SHARE → hazard` | PO line | `po_lines`, `grn_lines`, `asn`, `po_line_schedules`, `po_line_revisions` | `channel_performance_weekly` | `arrival_week` **(censored)** | 12 weeks | `P(arrive in week w)`, w = 1…12 |
| **Shortage**<br>`Monte Carlo simulation` | part × plant × week | `inventory_snapshots`, `inventory_transactions`, `part_plant`, `supplier_upstream` | `inventory_position_weekly`, `part_demand_weekly` + **outputs of the three models above** | `shortage_qty` *(cross-check only)* | 13 weeks | P(short), expected units, earliest-risk window |
| **Production exposure**<br>`not ML` | product × plant × week → BU → corporate | `bom`, `production_plan`, `product_economics`, `customers` | shortage simulation **sample paths** | — | 13 weeks | Units at risk, ₹ at risk |
| **Delivery schedule**<br>`LP` | part × plant × week | `part_plant`, `supplier_contracts`, `logistics_lanes` | `part_demand_weekly` | — | 13 weeks | Weekly receipt quantities |
| **Allocation**<br>`enumeration → MILP` | supplier × part × month | `supplier_allocation`, `alternate_sources`, `sourcing_channels`, `supplier_contracts`, `tooling`, `part_costs` | capacity + fill-rate outputs, shortage simulation | — **no ML labels** | 90 days | Recommended split, expected shortage, cost delta |
| **Quality risk**<br>`LightGBM` · Phase 2 | supplier × part × month | `quality_inspections`, `supplier_quality_ppm`, `supplier_audits` | rolling PPM features | reject rate | 90 days | P(reject rate > threshold) |

**Three things this table encodes:**

**Only four rows are learned models** — plan drift, capacity strain, fill rate, arrival
timing — and three of those four share one encoder. Everything else is linear algebra, a
simulation, or a solver.

**Shortage and allocation deliberately have no primary ML label.** Shortage comes from
simulating the three predictions forward; allocation comes from optimising over them.
Creating artificial labels for either would replace an auditable, composable calculation
with a black box, for no accuracy gain.

**Every model consumes derived tables, never raw ones directly.** The feature stores are
where the as-of discipline is enforced once, rather than re-implemented in five places.

---

## 13. Data request checklist for Rane

**Essential — nothing can be built without these**

- [ ] `plants`, `suppliers`, `supplier_sites`, `parts`, `products`, `customers`
- [ ] `bom` **with `effective_from` / `effective_to` on every row**
- [ ] `part_plant` (safety stock, reorder point, MOQ, lot size)
- [ ] `sourcing_channels` (supplier × part × plant, with approval status)
- [ ] `production_plan` **versioned, with `plan_date` on every row**
- [ ] `production_actual`
- [ ] `purchase_orders`, `po_lines`
- [ ] `grn_lines` **with partial receipts as separate rows**
- [ ] `inventory_snapshots` (weekly minimum)
- [ ] `calendar`

**High value — materially improves accuracy**

- [ ] `supplier_acknowledgements` — strongest leading indicator available
- [ ] `po_line_revisions` with `initiated_by`
- [ ] `po_line_schedules`
- [ ] `quality_inspections`
- [ ] `shortage_events`, `line_stop_events`, `expedite_events`
- [ ] `supplier_allocation` with change history
- [ ] `alternate_sources` with qualification status
- [ ] `product_economics` (contribution margin)
- [ ] `supplier_contracts` (MOQ, lot size, commitments)

**Requested but expect it to be weak**

- [ ] `supplier_capacity` — ask for it *with* `capacity_basis` so we know how much to trust it
- [ ] `supplier_upstream` — partial is fine and still valuable

**Phase 2**

- [ ] `supplier_quality_ppm`, `supplier_audits`, `logistics_lanes`, `supplier_financials`

---

## 14. Four questions that decide feasibility

Ask these in the first technical call. Each one can invalidate a use case.

**1. Can you reconstruct state as of a past date?** Or are stock levels and PO statuses
overwritten current-value columns with no history? If it's the latter, honest backtesting is
impossible and this becomes a data-instrumentation engagement first. *This is the question
that kills projects.*

**2. Are partial receipts separate rows in the GRN table?** If the ERP stores one summed
quantity per PO line, arrival timing (use case 4) cannot be built, and shortage timing
degrades to monthly.

**3. Is the production plan versioned with the date each version was made?** Without
`plan_date`, horizon cannot be computed, the drift model cannot be conditioned on it, and
"accurate at 1 month, poor at 3" cannot be measured — let alone improved.

**4. What is `capacity_basis` for your capacity figures?** Contracted, declared, audited or
estimated. If everything is `contracted`, plan to derive capacity from delivery history and
treat the stated number as one weak feature.

---

## 15. Fields we explicitly do not want

| Field | Why not |
|---|---|
| Current `status` columns on POs/shipments | Terminal-state leakage. Reconstruct from revision history instead. |
| `qty_received` as a column on `po_lines` | It *is* the label for fill rate. |
| `delivered_at` on a line | Part of the arrival-timing label. |
| Any pre-computed supplier risk/reliability score | Mutable whole-history figure — leaks the outcome. The system exists to *infer* this. |
| Any internal score whose derivation Rane cannot explain | Unusable as a feature and dangerous as a label. |
| Any *calculated ML feature* | We build the derived layer (Group I). Asking Rane for it means inheriting their definition of "fill rate" rather than controlling it. |

---

## 16. Data lineage and versioning

Five identifiers make any historical result reproducible. Carry all of them on every
snapshot, and stamp the last three onto every row of `model_outputs.csv`.

| Identifier | Answers |
|---|---|
| `dataset_version` | Which extract of Rane's source data |
| `feature_spec_version` | Which feature definitions were in force |
| `label_version` | Which label definitions |
| `model_version` | Which trained weights |
| `code_commit` | Which pipeline code |

`dataset_coverage.csv` (§1) records what history existed at extract time, so a result
computed when acknowledgement data started in 2020 can be distinguished from the same result
recomputed after a backfill.

**The property this buys:** *"if this model had been deployed in January 2023, here is
exactly what it would have predicted"* — reproducible on demand, not asserted. That claim is
the backbone of the POC's credibility, and it is unsupportable without these fields.
