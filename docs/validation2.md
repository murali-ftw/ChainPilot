# External dataset validation — run 2 (delta against run 1)

**Verdict: not usable.**

Three of the five training labels are still uniform random draws — `capacity_strain`,
`demand_drift` and `shortage_qty` — and the derived weekly stores still do not conserve
their source quantities (+48.0% ordered, +46.5% received). Everything else about this
regeneration is a large, genuine improvement.

| | run 2 | run 1 |
|---|---|---|
| **passed / failed / skipped** | **152 / 24 / 13** | 88 / 74 / 20 |
| total checks | 202 | 195 |
| verdict | not usable | not usable |

`SKIP` is never a pass. 13 checks did not run; they are enumerated in §4.

## Instrument

| | |
|---|---|
| validator SHA-256 **before** run 2 (frozen, as used for run 1) | `196325843f741cef927b6ef41e96cd097602a948df34ecbd63ec1e400cac217f` |
| validator SHA-256 **after** the one permitted addition | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |

The SHA changed, deliberately and only once. **`db/validator.py` was not otherwise edited:
no band widened, no gate softened, no tolerance added, no table special-cased.** The single
change is the addition described below, and it makes the validator strictly stricter — it can
only add failures, never remove them.

### The one addition: `<task>: target is not a uniform random draw`

One check per task. A one-sample Kolmogorov–Smirnov test of the uncensored `label_value`
against Uniform(0,1), rescaled to the sample's own observed support. **Fails when p ≥ 0.01** —
that is, fails when uniformity *cannot* be rejected and the label is indistinguishable from
noise. Implemented in `ks_uniform()`; p uses the asymptotic Kolmogorov limiting distribution
with the Stephens small-sample correction. Not run below n = 20, where it is reported as
skipped rather than passed.

Rescaling to the observed support is the load-bearing detail: it means moving the range is
not a fix. A label drawn `Uniform(0, 299)` or `Uniform(0.7, 1.4)` is caught exactly as
readily as one on `[0,1]`. The rescaling pins the two endpoints at 0 and 1, which biases the
KS statistic slightly *downward* — the test is conservative, and at n in the low thousands
that bias is far below the decision threshold.

**Runs 1 and 2 did not use identical instruments.** Run 1 has 195 checks, run 2 has 202: five
are the new KS checks, and two more (`production_actual` / `production_plan` as-of pairs)
exist only because those tables are now present. Every other check is bit-identical code.

## Control — did not move

| run | validator | result |
|---|---|---|
| run 1 | `196325…` | 167 passed / 2 failed / 4 skipped |
| **run 2, frozen instrument** | `196325…` | **167 passed / 2 failed / 4 skipped** |
| run 2, augmented instrument | `c49fc5…` | 172 passed / 2 failed / 4 skipped |

The frozen-instrument control run against `/Users/muralik/Documents/Programs/HADES/db/csv_full_seed1`
is **byte-for-byte identical** to run 1's saved control transcript (`diff` clean over all 580
lines), and the SHA was re-printed at run time. The control did not move; the delta below is
attributable to the dataset alone.

The augmented run differs by exactly **+5 passed** — the five new KS checks, all of which
pass on our own world with D between 0.63 and 0.93 at p = 0. The KS gate does not misfire on
real labels. The two standing control failures are unchanged: `PO lines arriving late` at
12.83% against a 15–25% band, the pre-existing `csv_full_seed1` finding documented in run 1.

---

## 2. Fixed — 58 checks moved FAIL → pass, 3 moved SKIP → pass

Grouped by the class of defect. None of these passes because a table became absent; the
two tables that *appeared* are handled at the end of this section.

### 2a. The as-of contract — 21 checks (run 1's headline defect, gone)

Run 1: every lag in the dataset was a per-table constant (`sd/|mean| = 0.000`), six event
tables crossed a week boundary on 0.0% of rows, and three tables recorded rows *before* they
happened. All of it is fixed.

| check | run 1 | run 2 | band |
|---|---|---|---|
| `po_lines: two timestamps differ` | 0.0% | **60.5%** | ≥ 2% |
| `purchase_orders: two timestamps differ` | 0.0% | **60.2%** | ≥ 2% |
| `inventory_snapshots: two timestamps differ` | 0.0% | **39.8%** | ≥ 2% |
| `shortage_events: two timestamps differ` | 0.0% | **82.6%** | ≥ 2% |
| `line_stop_events: two timestamps differ` | 0.0% | **83.6%** | ≥ 2% |
| `expedite_events: two timestamps differ` | 0.0% | **83.4%** | ≥ 2% |
| `po_lines: lag is not constant` | sd/\|mean\| 0.000 | **0.746** | ≥ 0.1 |
| `purchase_orders: lag is not constant` | 0.000 | **0.754** | ≥ 0.1 |
| `grn_lines: lag is not constant` | 0.000 | **0.684** | ≥ 0.1 |
| `goods_receipts: lag is not constant` | 0.000 | **0.694** | ≥ 0.1 |
| `asn: lag is not constant` | 0.000 | **0.753** | ≥ 0.1 |
| `quality_inspections: lag is not constant` | 0.000 | **0.746** | ≥ 0.1 |
| `inventory_snapshots: lag is not constant` | 0.000 | **0.689** | ≥ 0.1 |
| `line_stop_events: lag is not constant` | 0.000 | **0.705** | ≥ 0.1 |
| `shortage_events: lag is not constant` | 0.000 | **0.716** | ≥ 0.1 |
| `expedite_events: lag is not constant` | 0.000 | **0.708** | ≥ 0.1 |
| `supplier_allocation: lag is not constant` | 0.000 | **0.720** | ≥ 0.1 |
| `supplier_capacity: lag is not constant` | 0.000 | **0.527** | ≥ 0.1 |
| `inventory_transactions: recorded_ts >= event` | 12,201 inversions | **0** | 0 |
| `po_line_revisions: recorded_ts >= event` | 2,562 | **0** | 0 |
| `supplier_acknowledgements: recorded_ts >= event` | 4,227 | **0** | 0 |

Reporting lag is now a genuine distribution — P50 ≈ 4.7–8.7 d, P90 ≈ 11.3–20.1 d, max 54 d —
rather than a fixed offset. Run 1's degenerate calendar is gone too: `shortage_events` and
`expedite_events` no longer place every event on a Monday.

### 2b. The label table's identity — 12 checks

| check | run 1 | run 2 |
|---|---|---|
| `arrival_week: entity_type matches the spec` | 1,919 / 2,419 wrong | **0 / 1,200** |
| `capacity_strain: entity_type matches the spec` | 1,916 / 2,431 wrong | **0 / 1,200** |
| `demand_drift: entity_type matches the spec` | 1,902 / 2,403 wrong | **0 / 1,200** |
| `fill_rate: entity_type matches the spec` | 1,892 / 2,413 wrong | **0 / 1,200** |
| `shortage_qty: entity_type matches the spec` | 1,882 / 2,334 wrong | **0 / 1,200** |
| `entity_id resolves for entity_type=po_line` | 87 / 2,261 unresolved | **0 / 2,097** |
| `entity_id resolves for entity_type=channel` | 2,252 / 2,292 unresolved | **0 / 120** |
| `arrival_week: something is censored` | 0.0% | **13.6%** |
| `capacity_strain: something is censored` | 0.0% | **3.6%** |
| `demand_drift: something is censored` | 0.0% | **4.6%** |
| `fill_rate: something is censored` | 0.0% | **4.7%** |
| `shortage_qty: something is censored` | 0.0% | **3.8%** |

Each task now binds one-to-one to its spec-declared entity, and every single-key `entity_id`
resolves into its master. Censoring exists on all five tasks.

### 2c. Structural conformance — 8 checks

`all spec tables present` 47/49 → **49/49**. Seven tables that failed Tier 1 in run 1 are now
clean, six of them because their duplicate primary keys are gone:

| table | run 1 | run 2 |
|---|---|---|
| `inventory_snapshots` | 1,236 duplicate PKs | **clean** |
| `inventory_position_weekly` | 867 duplicate PKs | **clean** |
| `part_demand_weekly` | 884 duplicate PKs | **clean** |
| `supplier_financials` | 177 duplicate PKs | **clean** |
| `supplier_capacity` | 53 duplicate PKs | **clean** |
| `revealed_capacity_monthly` | 3 duplicate PKs | **clean** |
| `calendar` | `regime_flag = 'capacity_constraint'` (outside the declared enum) | **clean** |

Not a single duplicate primary key remains anywhere in the dataset.

### 2d. Distributional shape — 17 checks

| check | run 1 | run 2 | band |
|---|---|---|---|
| `fill-rate mass at exactly 1.0` | 27.3% | **80.0%** | ≥ 60% |
| `fill-rate mass at exactly 0` | 31.8% | **2.0%** | 1–3% |
| `PO lines arriving late` | 82.3% | **19.8%** | 15–25% |
| `lead-time distribution is right-skewed` | +0.05 | **+0.88** | ≥ 0.50 |
| `shortage events per year` | 120.5 | **235.6** | 180–250 |
| `line stops per year` | 14.7 | **21.7** | 15–25 |
| `expedites per year` | 93.7 | **131.6** | 100–150 |
| `month-of-year seasonality in demand` | 1.03× | **1.46×** | > 1.10 |
| `zero-order channel-weeks` | 0.0% | **80.2%** | 60–99% |
| `weekly store is a panel` | 0.0% | **100.0%** | ≥ 90% |

(plus the `[2019-2025]` twins of `PO lines arriving late`, both fill-rate point masses,
lead-time skewness, seasonality, zero-order channel-weeks and panel contiguity — seven more.
The three rare-event rates are reported rather than gated in the windowed pass, so they have
no twin here.) The fill-rate distribution has
recovered its point-mass shape, lead time is right-skewed rather than Gaussian, and the
weekly store is a real panel instead of a scatter.

### 2e. Tables that appeared — 3 checks, SKIP → pass

`production_actual` was absent in run 1 and is now present and clean (48,012 rows), and both
it and `production_plan` now pass their as-of checks (60.1% / 60.3% later-week). These are
genuine additions, not reclassified skips. `production_plan` itself is present but **fails**
Tier 1 — see §4.

---

## 3. Still failing — 15 checks

### 3a. Conservation — 4 checks. Moved a great deal, still not exact.

| check | run 1 | run 2 |
|---|---|---|
| `channel store conserves ordered units` | 4,114,197 → 138,615,579 (**+3269.2%**) | 1,944,910 → 2,878,869 (**+48.0%**) |
| `channel store conserves received units` | 2,992,098 → 110,778,603 (**+3602.4%**) | 1,870,498 → 2,740,051 (**+46.5%**) |
| `supplier store conserves ordered units` | 4,114,197 → 21,468,423 (**+421.8%**) | 1,944,910 → 2,878,869 (**+48.0%**) |
| `supplier store conserves received units` | 2,992,098 → 17,303,481 (**+478.3%**) | 1,870,498 → 2,740,051 (**+46.5%**) |

The error fell by roughly 68× on the channel store. It is still a **surplus**, meaning the
store contains units the source does not: the identity requires *exact* equality because
these are integer quantities and an aggregation neither invents nor destroys them.

One structural fact worth reading off these numbers: the supplier store's emitted totals are
now *identical* to the channel store's (2,878,869 and 2,740,051 on both rows). The supplier
rollup is faithful. The defect is entirely upstream, between `po_lines`/`grn_lines` and
`channel_performance_weekly` — the weekly store is still being generated rather than
aggregated from the transactions.

### 3b. Bucketing rule — 1 check. Did not move.

| check | run 1 | run 2 |
|---|---|---|
| `derived store buckets on visible week` | neither (visible 23.6% / event 23.6% / neither 52.7%, n=3,181) | **neither** (visible 28.8% / event 33.4% / neither 37.8%, n=18,000) |

On the 18,000 channel-weeks where the two candidate rules give different answers, the store
matches `max(event_week, recorded_week)` 28.8% of the time and `event_week` 33.4%, with 37.8%
matching neither. It is not using the wrong rule; it is not a reconstruction of the source at
all. This is the same conclusion as run 1, on 5.7× more evidence. It is the direct corollary
of §3a.

### 3c. Capacity observability — 4 checks. Moved further out of band.

| check | run 1 | run 2 | band |
|---|---|---|---|
| `supplier-months constrained` | 88.5% | **97.6%** | 20–30% |
| `supplier-months constrained [2019-2025]` | 88.5% | **97.8%** | 20–30% |
| `capacity unobservable` | 11.5% | **2.4%** | ≥ 70% |
| `capacity unobservable [2019-2025]` | 11.5% | **2.2%** | ≥ 70% |

These two are one measurement stated twice (`unobservable = 1 − constrained`). It moved
**away** from band by 9 points, not toward it. 97.6% of supplier-months look capacity-
constrained, so only 2.4% carry no capacity evidence — against our own world's 75.9%. Under
`delivered = min(K, ordered)` an unconstrained month is one that tells you nothing about true
capacity, and a world where almost every month is informative is a world where capacity is
not actually a binding constraint. UC2's whole premise is that capacity is *mostly*
unobservable.

### 3d. Fill-rate base rate — 2 checks. Large move, overshoots the band.

| check | run 1 | run 2 | band |
|---|---|---|---|
| `PO lines with fill < 1.0` | 72.7% | **20.0%** | 8–15% |
| `PO lines with fill < 1.0 [2019-2025]` | 72.2% | **20.1%** | 8–15% |

Moved 52.7 points and landed 5.0 points above the top of the band. This is the closest of the
standing failures and the one most plausibly a tuning matter rather than a structural one —
but 20% is not 8–15%, and it is directly upstream of §3c: too many short lines is exactly
what makes 97.6% of supplier-months look constrained.

### 3e. Structural conformance — 4 checks. Partial.

| check | run 1 | run 2 |
|---|---|---|
| `structurally clean tables` | 37/49 | **40/49** |
| `channel_performance_weekly` | 9 missing cols + 10,210 dup PKs | **9 missing cols** (PKs fixed) |
| `supplier_performance_weekly` | 10 missing cols + 4 type violations | **9 missing cols** (type violations fixed) |
| `parts` | `shelf_life_days` = `'180.0'` in an INTEGER column, ×151 | **×236** |

Both weekly stores still carry the column names the spec explicitly **retired**. Present:
`fill_rate_4w`, `fill_rate_13w`, `fill_rate_52w`, `otd_rate_13w`, `staleness_days`. Required
and absent: `is_active_week`, `fill_rate_last4`, `fill_rate_last13`, `fill_rate_last52`,
`otd_rate_last13`, `active_weeks_in_52`, `reporting_lag_days`, `weeks_since_last_activity`,
`weeks_since_last_receipt`. §11 of the spec states `staleness_days` "was removed, not
renamed" precisely so that analysis written against the old meaning fails loudly. It is still
here, and `reporting_lag_days` — the column that carries how far the as-of displacement moved
a week — is still missing, even though the underlying lag structure now exists (§2a).

`parts.shelf_life_days` writes `'180.0'` where an INTEGER is declared; the count rose from
151 to 236 with the table's growth from 250 to 500 rows.

---

## 4. Newly failing, and newly skipped

**This section matters more than §2.** Six checks that passed or were skipped in run 1 now
fail. All six are the same class of defect: **a categorical column carrying a value outside
the enum the spec declares.** Five of the six are vocabulary substitutions that did not exist in run 1's data;
the sixth is a float written into an INTEGER column.

| table | column | values present (count) | rows violating | declared set |
|---|---|---|---|---|
| `production_plan` | `plan_type` | **`baseline`** 48,012; **`reforecast`** 16,825 | 64,837 / 64,837 | `annual` / `quarterly` / `rolling` / `firm` |
| `supplier_acknowledgements` | `ack_status` | `partial` 5,151; **`accepted`** 4,063 | 4,063 / 9,214 | `full` / `partial` / `rejected` / `date_change` |
| `po_line_revisions` | `initiated_by` | **`planner`** 1,724; `buyer` 1,712; **`supplier_portal`** 1,685 | 3,409 / 5,121 | `supplier` / `buyer` / `system` |
| `supplier_upstream` | `confidence` | **a continuous float per row** (`0.53`, `0.71`, `0.59`, …) | 300 / 300 | `confirmed` / `reported` / `inferred` |
| `alternate_sources` | `qualification_status` | `qualified` 68; **`conditional`** 26; **`development`** 26 | 52 / 120 | `qualified` / `in_progress` / `potential` |
| `training_labels` | `censor_time` | `'1.0'` and similar in an INTEGER column | 363 / 6,000 | INTEGER |

Bold marks values outside the declared set.

These are substitutions, not corruptions — each column holds a coherent vocabulary of its
own that simply is not the spec's. Three of them change what is representable:

- **`supplier_upstream.confidence` is a continuous float, not one of three levels.** The
  spec's `confirmed` / `reported` / `inferred` is the mechanism by which tier-2 exposure is
  presented as partial rather than complete — the provenance of the link, not a strength.
  A per-row score is a different quantity, and every one of the 300 rows uses it, so the
  provenance distinction is unrecoverable.
- **`po_line_revisions.initiated_by` splits `supplier`/`buyer`/`system` into
  `planner` / `buyer` / `supplier_portal`.** The spec calls this column **critical** because
  "a buyer-initiated cut is not a supplier failure." `buyer` survives (1,712 rows), but
  `planner` (1,724) and `supplier_portal` (1,685) are new: two thirds of the revision history
  uses labels whose mapping onto the buyer/supplier distinction is an assumption, not a
  reading. `supplier_portal` in particular names a *channel*, not an initiator.
- **`production_plan.plan_type` is `baseline` / `reforecast`.** The column does carry
  information — a 74/26 split across 64,837 rows — but it is the versioning distinction, not
  the spec's `annual` / `quarterly` / `rolling` / `firm` plan-horizon distinction, and
  `plan_version` already carries versioning. The horizon type the drift model conditions on
  is absent.

The other two are narrower. `supplier_acknowledgements.ack_status` reduces to two values,
`partial` (valid) and `accepted` (not declared, presumably meaning `full`) — `rejected` and
`date_change` are gone entirely, so a rejection is indistinguishable from a shortfall.
`alternate_sources.qualification_status` uses `conditional` and `development`, both of which
belong to `sourcing_channels.approval_status`'s enum rather than this one, and `in_progress`
/ `potential` are absent.

**Newly skipped: none.** No check that ran in run 1 is skipped in run 2.

**One check disappeared rather than failing:** `entity_id resolves for entity_type=supplier`
was a run-1 failure (2,244 / 2,248 unresolved). It does not appear in run 2 because no label
row carries `entity_type = supplier` any more — which is correct, since the spec's task table
declares no supplier-entity task. This is a genuine fix, not a hidden skip.

**Skipped in run 2 — 13 checks, none of them passes:**

- 9 × `<table>: cleared Tier 1` — the gating markers for the nine tables that failed Tier 1
  (`alternate_sources`, `channel_performance_weekly`, `parts`, `po_line_revisions`,
  `production_plan`, `supplier_acknowledgements`, `supplier_performance_weekly`,
  `supplier_upstream`, `training_labels`). These record that downstream results for those
  tables are advisory.
- 2 × `two timestamps differ` for `inventory_position_weekly` and `plan_drift_features` —
  derived stores with `recorded_ts` but no event timestamp to compare against.
- 2 × `entity_id resolves` for `part_plant` and `product_plant` — composite entities with no
  single master primary key; resolution is not attempted rather than guessed.

---

## 5. The three structural questions

### Q1. Are the labels real? — **No. Three of five are uniform noise.**

The new KS check, per task. `D` is the KS statistic against Uniform(0,1) after rescaling to
the observed support; `p ≥ 0.01` means uniformity cannot be rejected, and the check fails.

| task | n uncensored | observed support | mean | sd | KS `D` | `p` | verdict |
|---|---|---|---|---|---|---|---|
| `fill_rate` | 1,144 | [0.0000, 1.0000] | 0.9791 | 0.0672 | 0.8453 | 0 | **pass** |
| `arrival_week` | 1,037 | [1.0000, 11.0000] | 5.9238 | 3.1782 | 0.0945 | 1.57e-08 | **pass** |
| `capacity_strain` | 1,157 | [0.0001, 1.4995] | 0.7496 | 0.4335 | 0.0160 | **0.926** | **FAIL** |
| `demand_drift` | 1,145 | [0.7012, 1.3999] | 1.0458 | 0.2043 | 0.0277 | **0.339** | **FAIL** |
| `shortage_qty` | 1,154 | [0.0000, 299.0000] | 149.4593 | 84.8186 | 0.0295 | **0.262** | **FAIL** |

Read the three failures against the closed form for a uniform draw on the observed support —
mean at the midpoint, sd at range/√12:

| task | support | midpoint | observed mean | range/√12 | observed sd |
|---|---|---|---|---|---|
| `capacity_strain` | [0, 1.5) | 0.7500 | **0.7496** | 0.4330 | **0.4335** |
| `demand_drift` | [0.7, 1.4) | 1.0500 | **1.0458** | 0.2021 | **0.2043** |
| `shortage_qty` | [0, 299] | 149.50 | **149.4593** | 86.31 | **84.8186** |

Three decimal places of agreement on both moments. These are `Uniform(0, 1.5)`,
`Uniform(0.7, 1.4)` and `Uniform(0, 299)`. Run 1's five labels were all `Uniform(0,1)`; run 2
has given three of them plausible *ranges* — a strain ratio that reaches 1.5, a drift ratio
around 1.0, a part count in the hundreds — without giving any of them a distribution. That is
why the check rescales: moving the range is not a fix, and every one of the existing gates
(`varies`, `censored`, `entity_type`, `label window`) passes on all three.

The two that pass are visibly real. `fill_rate` is piled on 1.0 with a tail (D = 0.85);
`arrival_week` is a genuine integer week index over 1–11 (D = 0.094 — small, because a
discrete uniform-ish grid is not far from continuous uniform, but rejected decisively at
p = 1.6e-08 on n = 1,037).

**The run-1 tell is gone but the underlying fact is unchanged for the other two.** Run 1's
`arrival_week` and `shortage_qty` both sat inside [0,1] — a week index and a part count that
fit in the unit interval. In run 2 `arrival_week` is a real week index on [1, 11] and
`shortage_qty` is a real-looking part count on [0, 299]; but `shortage_qty` is still drawn
flat across that range, and a flat shortage distribution is not something a supply chain
produces.

**Entity binding: fully fixed.** Every task binds one-to-one to its spec-declared entity
(0 wrong out of 1,200 on all five, versus ~1,900 of ~2,400 wrong per task in run 1), and every
single-key `entity_id` resolves: 0 / 2,097 unresolved for `po_line`, 0 / 120 for `channel`.
`part_plant` and `product_plant` are composite and are reported as skipped, not passed.

### Q2. Do the derived stores conserve? — **No. +48.0% ordered, +46.5% received.**

| store | quantity | source | emitted | error | run 1 error |
|---|---|---|---|---|---|
| `channel_performance_weekly` | ordered | 1,944,910 | 2,878,869 | **+48.0207%** | +3269.2013% |
| `channel_performance_weekly` | received | 1,870,498 | 2,740,051 | **+46.4878%** | +3602.3721% |
| `supplier_performance_weekly` | ordered | 1,944,910 | 2,878,869 | **+48.0207%** | +421.8132% |
| `supplier_performance_weekly` | received | 1,870,498 | 2,740,051 | **+46.4878%** | +478.3060% |

Measured over the store's own span (2019-01-07 to 2025-12-29), restricted to channels in the
master, with both sides filtered identically. The identity is exact equality — integer
quantities, and an aggregation neither invents nor destroys them.

68× better than run 1 and still wrong by half. The store holds roughly 1.48 units for every
unit in `po_lines`/`grn_lines`. And the supplier store's emitted figures are now byte-identical
to the channel store's, so the rollup is correct — the surplus is created entirely at the
channel level. Combined with Q3's answer, the reading is that the weekly store is *synthesised
on its own weekly grid* rather than aggregated from the transaction tables. The bucketing
probe agrees: of 18,000 channel-weeks where the visible-week and event-week rules give
different answers, 37.8% match neither rule.

This is the check that distinguishes "sparse because the world is sparse" from "sparse because
the builder dropped rows." Here it says something else again: dense because the builder
invented rows.

### Q3. Is the weekly store a panel? — **Yes. Cleanly.**

| measure | run 1 | run 2 | band |
|---|---|---|---|
| zero-order channel-week share | 0.0% | **80.2%** | 60–99% |
| contiguity (channels with an unbroken 7-day run) | 0.0% | **100.0%** | ≥ 90% |
| duplicate PKs | 10,210 on 55,000 rows | **0** on 43,800 rows | 0 |
| rows / channels | 55,000 / 320 = 172 | 43,800 / 120 = **365** | — |

120 channels × 365 weeks = 43,800 rows exactly. Every channel has a complete, gap-free weekly
series across the store's span, 35,122 of the 43,800 cells are genuinely idle, and the
staleness distribution is plausible (P50 28 d, P90 77 d, max 406 d). Run 1's store was a
random scatter of channel-weeks over which no rolling window could be computed. This one is a
real panel.

### What the three answers say together

**The dataset was generated this time rather than sampled.** That is the finding, and it is a
large one. Run 1's artefacts — constant lags, Monday-only events, randomly-assigned entity
types, a sampled channel-week scatter, duplicate keys in six tables, a fill-rate distribution
with no point masses — are all gone, and the world now has a real reporting-lag structure, a
real procurement calendar, a real panel, and correctly-bound labels.

What it does not yet have is a **derived layer computed from its own transactions** (Q2, Q3's
corollary) and **labels drawn from the world rather than from a random number generator** on
three of five tasks (Q1). Those two are the same class of defect: the feature store and three
of the labels are parallel fabrications sitting beside a plausible transaction history, rather
than functions of it.

---

## 6. Full tier tables

`run 1` column shows the same check's run-1 value. `_(new check)_` marks a check that did not exist in run 1.

### Tier 1


**TABLES**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| -- | extra tables (informational) | `0` | `0` | n/a |
| ok | all spec tables present | `49/49` | `47/49` | 49/49 |
| **FAIL** | structurally clean tables | `40/49` | `37/49` | 49/49 |

**PER-TABLE**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| **FAIL** | alternate_sources | `120 rows` | `144 rows` | clean |
| ok | asn | `17,218 rows` | `7,000 rows` | clean |
| ok | bom | `1,319 rows` | `700 rows` | clean |
| ok | business_units | `5 rows` | `5 rows` | clean |
| ok | calendar | `37,555 rows` | `37,548 rows` | clean |
| **FAIL** | channel_performance_weekly | `43,800 rows` | `55,000 rows` | clean |
| ok | customers | `20 rows` | `20 rows` | clean |
| ok | dataset_coverage | `48 rows` | `46 rows` | clean |
| ok | expedite_events | `2,005 rows` | `1,400 rows` | clean |
| ok | goods_receipts | `17,218 rows` | `23,000 rows` | clean |
| ok | grn_lines | `17,218 rows` | `23,000 rows` | clean |
| ok | inventory_position_weekly | `10,000 rows` | `50,000 rows` | clean |
| ok | inventory_snapshots | `15,340 rows` | `60,000 rows` | clean |
| ok | inventory_transactions | `1,534 rows` | `45,000 rows` | clean |
| ok | line_stop_events | `330 rows` | `220 rows` | clean |
| ok | logistics_lanes | `190 rows` | `250 rows` | clean |
| ok | model_outputs | `624 rows` | `900 rows` | clean |
| ok | part_costs | `120 rows` | `700 rows` | clean |
| ok | part_demand_weekly | `7,800 rows` | `50,000 rows` | clean |
| ok | part_plant | `1,680 rows` | `580 rows` | clean |
| **FAIL** | parts | `500 rows` | `250 rows` | clean |
| ok | plan_drift_features | `43,052 rows` | `20,000 rows` | clean |
| ok | plants | `7 rows` | `7 rows` | clean |
| **FAIL** | po_line_revisions | `5,121 rows` | `6,000 rows` | clean |
| ok | po_line_schedules | `14,755 rows` | `40,000 rows` | clean |
| ok | po_lines | `9,214 rows` | `20,000 rows` | clean |
| ok | product_economics | `120 rows` | `1,800 rows` | clean |
| ok | production_actual | `48,012 rows` | _skip: no file_ | clean |
| **FAIL** | production_plan | `64,837 rows` | _skip: no file_ | clean |
| ok | products | `120 rows` | `120 rows` | clean |
| ok | purchase_orders | `9,214 rows` | `6,667 rows` | clean |
| ok | quality_inspections | `17,218 rows` | `23,000 rows` | clean |
| ok | revealed_capacity_monthly | `7,560 rows` | `4,000 rows` | clean |
| ok | shortage_events | `3,580 rows` | `1,800 rows` | clean |
| ok | snapshots | `100 rows` | `401 rows` | clean |
| ok | sourcing_channels | `120 rows` | `320 rows` | clean |
| **FAIL** | supplier_acknowledgements | `9,214 rows` | `15,600 rows` | clean |
| ok | supplier_allocation | `120 rows` | `320 rows` | clean |
| ok | supplier_audits | `500 rows` | `500 rows` | clean |
| ok | supplier_capacity | `10,639 rows` | `15,930 rows` | clean |
| ok | supplier_contracts | `120 rows` | `500 rows` | clean |
| ok | supplier_financials | `900 rows` | `450 rows` | clean |
| **FAIL** | supplier_performance_weekly | `24,820 rows` | `17,899 rows` | clean |
| ok | supplier_quality_ppm | `12,480 rows` | `900 rows` | clean |
| ok | supplier_sites | `190 rows` | `75 rows` | clean |
| **FAIL** | supplier_upstream | `300 rows` | `180 rows` | clean |
| ok | suppliers | `120 rows` | `50 rows` | clean |
| ok | tooling | `120 rows` | `150 rows` | clean |
| **FAIL** | training_labels | `6,000 rows` | `12,000 rows` | clean |

**TABLES**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| warn | declared-width breaches (warning) | `1 table(s)` | `3 table(s)` | 0 |

### Tier 2


**GATING**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| `SKIP` | alternate_sources: cleared Tier 1 | `no` | _(new check)_ | -- |
| `SKIP` | channel_performance_weekly: cleared Tier 1 | `no` | _skip: no_ | -- |
| `SKIP` | parts: cleared Tier 1 | `no` | _skip: no_ | -- |
| `SKIP` | po_line_revisions: cleared Tier 1 | `no` | _(new check)_ | -- |
| `SKIP` | production_plan: cleared Tier 1 | `no` | _(new check)_ | -- |
| `SKIP` | supplier_acknowledgements: cleared Tier 1 | `no` | _(new check)_ | -- |
| `SKIP` | supplier_performance_weekly: cleared Tier 1 | `no` | _skip: no_ | -- |
| `SKIP` | supplier_upstream: cleared Tier 1 | `no` | _(new check)_ | -- |
| `SKIP` | training_labels: cleared Tier 1 | `no` | _(new check)_ | -- |

**AS-OF**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| ok | asn: two timestamps differ | `59.9%` | `14.5%` | >= 2% |
| ok | asn: lag is not constant | `sd/|mean|=0.753` | `sd/|mean|=0.000` | >= 0.1 |
| ok | asn: recorded_ts >= event | `0` | `0` | 0 |
| ok | expedite_events: two timestamps differ | `83.4%` | `0.0%` | >= 2% |
| ok | expedite_events: lag is not constant | `sd/|mean|=0.708` | `sd/|mean|=0.000` | >= 0.1 |
| ok | expedite_events: recorded_ts >= event | `0` | `0` | 0 |
| ok | goods_receipts: two timestamps differ | `65.0%` | `14.1%` | >= 2% |
| ok | goods_receipts: lag is not constant | `sd/|mean|=0.694` | `sd/|mean|=0.000` | >= 0.1 |
| ok | goods_receipts: recorded_ts >= event | `0` | `0` | 0 |
| ok | grn_lines: two timestamps differ | `65.1%` | `14.1%` | >= 2% |
| ok | grn_lines: lag is not constant | `sd/|mean|=0.684` | `sd/|mean|=0.000` | >= 0.1 |
| ok | grn_lines: recorded_ts >= event | `0` | `0` | 0 |
| `SKIP` | inventory_position_weekly: two timestamps differ | `no event column` | _skip: no event column_ | -- |
| ok | inventory_snapshots: two timestamps differ | `39.8%` | `0.0%` | >= 2% |
| ok | inventory_snapshots: lag is not constant | `sd/|mean|=0.689` | `sd/|mean|=0.000` | >= 0.1 |
| ok | inventory_snapshots: recorded_ts >= event | `0` | `0` | 0 |
| ok | inventory_transactions: two timestamps differ | `40.2%` | `24.8%` | >= 2% |
| ok | inventory_transactions: lag is not constant | `sd/|mean|=0.695` | `sd/|mean|=2.062` | >= 0.1 |
| ok | inventory_transactions: recorded_ts >= event | `0` | `12,201` | 0 |
| ok | line_stop_events: two timestamps differ | `83.6%` | `0.0%` | >= 2% |
| ok | line_stop_events: lag is not constant | `sd/|mean|=0.705` | `sd/|mean|=0.000` | >= 0.1 |
| ok | line_stop_events: recorded_ts >= event | `0` | `0` | 0 |
| `SKIP` | plan_drift_features: two timestamps differ | `no event column` | _skip: no event column_ | -- |
| ok | po_line_revisions: two timestamps differ | `60.1%` | `44.0%` | >= 2% |
| ok | po_line_revisions: lag is not constant | `sd/|mean|=0.751` | `sd/|mean|=7.485` | >= 0.1 |
| ok | po_line_revisions: recorded_ts >= event | `0` | `2,562` | 0 |
| -- | po_line_schedules: two timestamps differ | `48.5%` | `0.0%` | exempt |
| ok | po_line_schedules: recorded_ts >= event | `0` | `0` | 0 |
| ok | po_lines: two timestamps differ | `60.5%` | `0.0%` | >= 2% |
| ok | po_lines: lag is not constant | `sd/|mean|=0.746` | `sd/|mean|=0.000` | >= 0.1 |
| ok | po_lines: recorded_ts >= event | `0` | `0` | 0 |
| ok | production_actual: two timestamps differ | `60.1%` | _skip: no file_ | >= 2% |
| ok | production_actual: lag is not constant | `sd/|mean|=0.753` | _(new check)_ | >= 0.1 |
| ok | production_actual: recorded_ts >= event | `0` | _(new check)_ | 0 |
| ok | production_plan: two timestamps differ | `60.3%` | _skip: no file_ | >= 2% |
| ok | production_plan: lag is not constant | `sd/|mean|=0.750` | _(new check)_ | >= 0.1 |
| ok | production_plan: recorded_ts >= event | `0` | _(new check)_ | 0 |
| ok | purchase_orders: two timestamps differ | `60.2%` | `0.0%` | >= 2% |
| ok | purchase_orders: lag is not constant | `sd/|mean|=0.754` | `sd/|mean|=0.000` | >= 0.1 |
| ok | purchase_orders: recorded_ts >= event | `0` | `0` | 0 |
| ok | quality_inspections: two timestamps differ | `59.6%` | `14.6%` | >= 2% |
| ok | quality_inspections: lag is not constant | `sd/|mean|=0.746` | `sd/|mean|=0.000` | >= 0.1 |
| ok | quality_inspections: recorded_ts >= event | `0` | `0` | 0 |
| ok | shortage_events: two timestamps differ | `82.6%` | `0.0%` | >= 2% |
| ok | shortage_events: lag is not constant | `sd/|mean|=0.716` | `sd/|mean|=0.000` | >= 0.1 |
| ok | shortage_events: recorded_ts >= event | `0` | `0` | 0 |
| ok | supplier_acknowledgements: two timestamps differ | `60.4%` | `32.7%` | >= 2% |
| ok | supplier_acknowledgements: lag is not constant | `sd/|mean|=0.745` | `sd/|mean|=2.057` | >= 0.1 |
| ok | supplier_acknowledgements: recorded_ts >= event | `0` | `4,227` | 0 |
| ok | supplier_allocation: two timestamps differ | `60.8%` | `100.0%` | >= 2% |
| ok | supplier_allocation: lag is not constant | `sd/|mean|=0.720` | `sd/|mean|=0.000` | >= 0.1 |
| -- | supplier_allocation: recorded_ts >= event | `0` | `0` | n/a |
| ok | supplier_audits: two timestamps differ | `78.8%` | `100.0%` | >= 2% |
| ok | supplier_audits: lag is not constant | `sd/|mean|=0.727` | `sd/|mean|=0.572` | >= 0.1 |
| ok | supplier_audits: recorded_ts >= event | `0` | `0` | 0 |
| ok | supplier_capacity: two timestamps differ | `94.8%` | `44.1%` | >= 2% |
| ok | supplier_capacity: lag is not constant | `sd/|mean|=0.527` | `sd/|mean|=0.000` | >= 0.1 |
| -- | supplier_capacity: recorded_ts >= event | `0` | `0` | n/a |
| ok | supplier_quality_ppm: two timestamps differ | `100.0%` | `100.0%` | >= 2% |
| ok | supplier_quality_ppm: lag is not constant | `sd/|mean|=0.168` | `sd/|mean|=0.591` | >= 0.1 |
| ok | supplier_quality_ppm: recorded_ts >= event | `0` | `0` | 0 |

**CONSERVATION**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| **FAIL** | channel store conserves ordered units | `2,878,869 vs 1,944,910` | `138,615,579 vs 4,114,197` | exact |
| **FAIL** | channel store conserves received units | `2,740,051 vs 1,870,498` | `110,778,603 vs 2,992,098` | exact |
| **FAIL** | derived store buckets on visible week | `neither` | `neither` | max(event,recorded) |
| **FAIL** | supplier store conserves ordered units | `2,878,869 vs 1,944,910` | `21,468,423 vs 4,114,197` | exact |
| **FAIL** | supplier store conserves received units | `2,740,051 vs 1,870,498` | `17,303,481 vs 2,992,098` | exact |
| ok | part_demand_weekly horizon_days identity | `0/7,800` | `0/50,000` | 0 |
| ok | part_demand_weekly p90 >= p50 | `0/7,800` | `0/50,000` | 0 |
| ok | part_demand_weekly is strictly forward-looking | `0/7,800` | `0/50,000` | 0 |
| -- | part_demand_weekly unit conservation | `not asserted` | `not asserted` | n/a |

**LABELS**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| ok | label window opens after the snapshot | `0` | `0` | 0 |
| ok | arrival_week: something is censored | `13.6%` | `0.0%` | 1%-60% |
| ok | arrival_week: uncensored target varies | `sd/|mean|=0.5365 over 11 values` | `sd/|mean|=0.5831 over 2,164 values` | > 0.01 and >= 5 |
| ok | arrival_week: target is not a uniform random draw | `D=0.0945 p=1.57e-08` | _(new check)_ | p < 0.01 |
| ok | arrival_week: entity_type matches the spec | `0/1,200 wrong` | `1,919/2,419 wrong` | 0 |
| ok | capacity_strain: something is censored | `3.6%` | `0.0%` | 1%-60% |
| ok | capacity_strain: uncensored target varies | `sd/|mean|=0.5783 over 1,119 values` | `sd/|mean|=0.5878 over 2,135 values` | > 0.01 and >= 5 |
| **FAIL** | capacity_strain: target is not a uniform random draw | `D=0.0160 p=0.926` | _(new check)_ | p < 0.01 |
| ok | capacity_strain: entity_type matches the spec | `0/1,200 wrong` | `1,916/2,431 wrong` | 0 |
| ok | demand_drift: something is censored | `4.6%` | `0.0%` | 1%-60% |
| ok | demand_drift: uncensored target varies | `sd/|mean|=0.1954 over 1,060 values` | `sd/|mean|=0.5808 over 2,124 values` | > 0.01 and >= 5 |
| **FAIL** | demand_drift: target is not a uniform random draw | `D=0.0277 p=0.339` | _(new check)_ | p < 0.01 |
| ok | demand_drift: entity_type matches the spec | `0/1,200 wrong` | `1,902/2,403 wrong` | 0 |
| ok | fill_rate: something is censored | `4.7%` | `0.0%` | 1%-60% |
| ok | fill_rate: uncensored target varies | `sd/|mean|=0.0687 over 172 values` | `sd/|mean|=0.5797 over 2,138 values` | > 0.01 and >= 5 |
| ok | fill_rate: target is not a uniform random draw | `D=0.8453 p=0` | _(new check)_ | p < 0.01 |
| ok | fill_rate: entity_type matches the spec | `0/1,200 wrong` | `1,892/2,413 wrong` | 0 |
| ok | shortage_qty: something is censored | `3.8%` | `0.0%` | 1%-60% |
| ok | shortage_qty: uncensored target varies | `sd/|mean|=0.5675 over 293 values` | `sd/|mean|=0.5706 over 2,076 values` | > 0.01 and >= 5 |
| **FAIL** | shortage_qty: target is not a uniform random draw | `D=0.0295 p=0.262` | _(new check)_ | p < 0.01 |
| ok | shortage_qty: entity_type matches the spec | `0/1,200 wrong` | `1,882/2,334 wrong` | 0 |
| ok | entity_id resolves for entity_type=po_line | `0/2,097 unresolved` | `87/2,261 unresolved` | 0 |
| ok | entity_id resolves for entity_type=channel | `0/120 unresolved` | `2,252/2,292 unresolved` | 0 |
| `SKIP` | entity_id resolves for entity_type=part_plant | `199 ids` | _skip: 2,231 ids_ | -- |
| `SKIP` | entity_id resolves for entity_type=product_plant | `417 ids` | _skip: 2,274 ids_ | -- |

**SEPARATION**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| ok | inventory_position_weekly: no forward-looking column | `none` | `none` | none |
| ok | part_demand_weekly: no forward-looking column | `none` | `none` | none |
| ok | plan_drift_features: no forward-looking column | `none` | `none` | none |
| ok | revealed_capacity_monthly: no forward-looking column | `none` | `none` | none |
| ok | model_outputs disjoint from feature tables | `0 table(s) overlap` | `0 table(s) overlap` | 0 |

**SNAPSHOTS**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| ok | snapshots: as_of_ts <= data_cutoff_ts | `0/100` | `0/401` | 0 |
| ok | snapshots: five version fields populated | `0 field(s) blank` | `0 field(s) blank` | 0 |
| ok | snapshots: code_commit is consistent | `1 distinct` | `1 distinct` | 1 |
| ok | snapshots: horizon_days matches the labels | `labels reach +90d` | `labels reach +90d` | snapshot 90d |

### Tier 3


**SHAPE**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| **FAIL** | PO lines with fill < 1.0 | `20.0%` | `72.7%` | 8%-15% |
| ok | PO lines arriving late | `19.8%` | `82.3%` | 15%-25% |
| ok | fill-rate mass at exactly 1.0 | `80.0%` | `27.3%` | >= 60% |
| ok | fill-rate mass at exactly 0 | `2.0%` | `31.8%` | 1%-3% |
| ok | right-censored tail is the right size | `0.38x` | `0.48x` | 0.3-1.2x |
| **FAIL** | supplier-months constrained | `97.6%` | `88.5%` | 20%-30% |
| **FAIL** | capacity unobservable | `2.4%` | `11.5%` | >= 70% |
| ok | lead-time distribution is right-skewed | `+0.88` | `+0.05` | >= 0.5 |
| ok | shortage events per year | `235.6` | `120.5` | 180-250 |
| ok | line stops per year | `21.7` | `14.7` | 15-25 |
| ok | expedites per year | `131.6` | `93.7` | 100-150 |
| ok | month-of-year seasonality in demand | `1.46x peak/trough` | `1.03x peak/trough` | > 1.10 |

**DENSITY**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| ok | zero-order channel-weeks | `80.2%` | `0.0%` | 60%-99% |
| ok | weekly store is a panel | `100.0%` | `0.0%` | >= 90% |
| -- | staleness since last trade | `P50 28d / P90 77d / max 406d` | _skip: no idle weeks_ | reported |
| ok | PO lines per channel per year | `5.42` | `4.15` | >= 3.0 |
| -- | training rows per snapshot | `60` | `30` | reported |

**SHAPE**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| **FAIL** | PO lines with fill < 1.0 [2019-2025] | `20.1%` | `72.2%` | 8%-15% |
| ok | PO lines arriving late [2019-2025] | `20.0%` | `82.7%` | 15%-25% |
| ok | fill-rate mass at exactly 1.0 [2019-2025] | `79.9%` | `27.8%` | >= 60% |
| ok | fill-rate mass at exactly 0 [2019-2025] | `1.9%` | `32.1%` | 1%-3% |
| ok | right-censored tail is the right size [2019-2025] | `1.13x` | `0.91x` | 0.3-1.2x |
| **FAIL** | supplier-months constrained [2019-2025] | `97.8%` | `88.5%` | 20%-30% |
| **FAIL** | capacity unobservable [2019-2025] | `2.2%` | `11.5%` | >= 70% |
| ok | lead-time distribution is right-skewed [2019-2025] | `+0.93` | `+0.05` | >= 0.5 |
| -- | shortage events per year [2019-2025] | `260.1` | `121.2` | 180-250 (not gated) |
| -- | line stops per year [2019-2025] | `24.3` | `16.6` | 15-25 (not gated) |
| -- | expedites per year [2019-2025] | `143.6` | `95.9` | 100-150 (not gated) |
| ok | month-of-year seasonality in demand [2019-2025] | `1.43x peak/trough` | `1.03x peak/trough` | > 1.10 |

**DENSITY**

| | check | run 2 | run 1 | expected |
|---|---|---|---|---|
| ok | zero-order channel-weeks [2019-2025] | `80.2%` | `0.0%` | 60%-99% |
| ok | weekly store is a panel [2019-2025] | `100.0%` | `0.0%` | >= 90% |
| -- | staleness since last trade [2019-2025] | `P50 28d / P90 77d / max 406d` | _skip: no idle weeks_ | reported |
| ok | PO lines per channel per year [2019-2025] | `7.63` | `4.15` | >= 3.0 |
| -- | training rows per snapshot [2019-2025] | `60` | `27` | reported |

---

## 7. Comparison against `csv_full_seed1`

Third column is run 1, so the movement is visible. Control figures are from the
frozen-instrument control run, which is byte-identical to run 1's.

| Tier 3 figure | run 2 | run 1 | `csv_full_seed1` | expected band | run 2 verdict |
|---|---|---|---|---|---|
| PO lines with fill < 1.0 | 20.0% | 72.7% | 9.5% | 8–15% | **FAIL** |
| PO lines arriving late | 19.8% | 82.3% | 12.8% | 15–25% | ok |
| fill-rate mass at exactly 1.0 | 80.0% | 27.3% | 90.5% | ≥ 60% | ok |
| fill-rate mass at exactly 0 | 2.0% | 31.8% | 1.4% | 1–3% | ok |
| right-censored tail (× span-implied) | 0.38× | 0.48× | 0.56× | 0.30–1.20× | ok |
| supplier-months constrained | 97.6% | 88.5% | 24.1% | 20–30% | **FAIL** |
| capacity unobservable | 2.4% | 11.5% | 75.9% | ≥ 70% | **FAIL** |
| lead-time skewness | +0.88 | +0.05 | +2.36 | ≥ 0.50 | ok |
| lead time median / P90 / P99 (days) | 63 / 96 / 132 | 50 / 78 / 93 | 31 / 67 / 139 | — | — |
| shortage events / year | 235.6 | 120.5 | 214.7 | 180–250 | ok |
| line stops / year | 21.7 | 14.7 | 20.0 | 15–25 | ok |
| expedites / year | 131.6 | 93.7 | 124.8 | 100–150 | ok |
| demand peak/trough ratio | 1.46× | 1.03× | 1.30× | > 1.10 | ok |
| zero-order channel-weeks | 80.2% | 0.0% | 81.4% | 60–99% | ok |
| channels with a contiguous weekly panel | 100.0% | 0.0% | 100.0% | ≥ 90% | ok |
| channel-weeks per channel | 365 | 172 | 387 | — | — |
| PO lines per channel per year | 5.42 | 4.15 | 7.02 | ≥ 3.0 | ok |
| staleness P50 / P90 / max (days) | 28 / 77 / 406 | no idle weeks | 42 / 1085 / 3563 | — | — |
| training rows per snapshot | 60 | 30 | 44,007 | — | — |

Run 2 now sits close to `csv_full_seed1` on nine of these where run 1 sat nowhere near any of
them. The three exceptions are the standing failures: fill-below-1 overshoots (20.0% vs our
9.5%), and capacity observability is inverted (2.4% unobservable vs our 75.9%).

Two figures outside the gates are worth flagging even though no band applies:

- **`training rows per snapshot` = 60**, against 44,007 in `csv_full_seed1`. 6,000 labels over
  100 snapshots, 1,200 per task across a 120-channel / 500-part world. This is not a gate —
  the right count varies by two orders of magnitude between presets — but 1,200 rows per task
  is a very small training set for a temporal GNN head, and it is the population the KS test
  in §5 had to work with.
- **Lead time median rose 50 → 63 days** while the control sits at 31. Longer and now
  correctly right-skewed, but roughly double our world's central tendency.

---

## 8. Gate readiness

### Delivery risk / arrival timing — **not trainable**

Blocked by: `channel store conserves received units` (+46.5%), `derived store buckets on
visible week` (neither rule), `channel_performance_weekly` (9 missing columns).

The as-of contract underneath this task is now sound — `po_lines` 60.5% later-week,
`grn_lines` 65.1%, zero inversions anywhere, lag P50 ≈ 4.7 d — and `arrival_week` is a real
censored week index (13.6% censored, KS p = 1.6e-08). That is a genuine reversal from run 1.
But the head reads `channel_performance_weekly`, and that store holds 1.46 units for every
unit in `grn_lines`, buckets on no identifiable rule, and is missing
`reporting_lag_days`, `weeks_since_last_receipt`, `active_weeks_in_52` and
`is_active_week` — the four columns that carry the as-of displacement into the features.
The lag structure exists in the raw tables and does not reach the model.

Also degraded: `po_line_revisions.initiated_by` now reads `planner` / `buyer` /
`supplier_portal`, so 3,409 of 5,121 rows carry a label outside the spec's
`supplier`/`buyer`/`system` set — the distinction the spec calls critical because a
buyer-initiated cut is not a supplier failure.

### Fill rate — **not trainable**, and closest to being so

Blocked by: `PO lines with fill < 1.0` (20.0%, band 8–15%), plus the same
`channel_performance_weekly` conservation and column defects.

The label itself is now good: `fill_rate` is the only task that passes every label gate
including KS (D = 0.85, p = 0), 4.7% censored, mass 80.0% at exactly 1.0 and 2.0% at exactly
0. The distribution has the point-mass-plus-tail shape the binned CDF head needs. The
blocking issues are the base rate overshooting its band by 5 points, and the feature store it
would train on being unconserved.

If the derived layer were rebuilt by aggregation from the transactions and the shortfall rate
tuned down to 8–15%, this head would clear.

### Part shortage — **not trainable**

Blocked by: `shortage_qty: target is not a uniform random draw` (p = 0.262), `capacity
unobservable` (2.4%, band ≥ 70%), `supplier-months constrained` (97.6%, band 20–30%).

`shortage_qty` is `Uniform(0, 299)`: mean 149.46 against a midpoint of 149.50, sd 84.82
against range/√12 of 86.31. The cross-check head has nothing to learn. The simulation path is
in better shape — `part_demand_weekly` passes all three of its identities, the inventory
tables are clean, `inventory_position_weekly` is clean, shortage/line-stop/expedite rates are
all in band, and seasonality is real at 1.46× — but the capacity envelope it needs from UC2 is
built on `revealed_capacity_monthly`, and with 97.6% of supplier-months flagged constrained
there is effectively no unconstrained baseline to infer an envelope against.

### Summary

| task | run 1 | run 2 | blocking checks |
|---|---|---|---|
| Delivery risk / arrival timing | not trainable | **not trainable** | 3 (conservation, bucketing, store columns) |
| Fill rate | not trainable | **not trainable** | 3 (fill base rate, conservation, store columns) |
| Part shortage | not trainable | **not trainable** | 3 (uniform label, capacity observability ×2) |

No task moved to trainable. All three are blocked by a smaller and more specific set of
defects than in run 1, and two of the three blockers are shared: rebuild the derived weekly
stores by aggregating the transaction tables, and the delivery-risk and fill-rate heads both
unblock substantially.

---

## Appendix — files

| file | contents |
|---|---|
| `docs/external_dataset_validation.md` | run 1 report (**not modified**) |
| `docs/validator_run_external.txt` | run 1 stdout, external |
| `docs/validator_run_control_csv_full_seed1.txt` | run 1 stdout, control |
| `docs/validation2.md` | this report |
| `docs/validator_run2_external.txt` | run 2 stdout, external (633 lines) |
| `docs/validator_run2_control_frozen_instrument.txt` | run 2 control, frozen validator `196325…` — byte-identical to run 1's |
| `docs/validator_run2_control_csv_full_seed1.txt` | run 2 control, augmented validator `c49fc5…` |

Validator SHA-256 at the end of run 2: `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` —
matches the post-addition SHA recorded in the header. No further edit was made during or after
the runs.

No `.csv`, `manifest.json`, `dataset_structure.md` or `schema.sql` was modified.
