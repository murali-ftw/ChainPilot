# External dataset validation — run 4 (four-way delta, restructured `db/`)

**Verdict: not usable.**

The rebuild is a genuine simulator in places — recording lag is now a real stochastic process,
shortages descend from acknowledgement shortfalls, and staleness↔volatility finally has the
right sign — but it fails harder than any previous run: **99 passed / 73 failed / 22 skipped**,
with 64 checks regressing from run 3 and none newly fixed. The decisive probe is unchanged:
every label still correlates ≈ 0 with its own most predictive feature, and the generator source
shows why — the labels are computed from `sin(i/17)` and `sin(i/31)` of a loop counter, not from
the world.

| | run 4 | run 3 | run 2 | run 1 |
|---|---|---|---|---|
| **passed / failed / skipped** | **99 / 73 / 22** | 167 / 8 / 5 | 152 / 24 / 13 | 88 / 74 / 20 |
| total checks | 210 | 194 | 202 | 195 |
| verdict | **not usable** | not usable | not usable | not usable |

`SKIP` is never a pass. 22 checks did not run; they are enumerated in §7.

## Instrument — not modified at all

| | |
|---|---|
| `shasum -a 256 db/validator.py` **before** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| `shasum -a 256 db/validator.py` **after** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |

**Diff: none. No change was made, because none was needed** — `discover()` at
[validator.py:582–608](../db/validator.py#L582) already walks the tree with `os.walk(root)`,
already accepts `.csv` and `.csv.gz`, and already detects and de-duplicates a table that appears
in two folders. The flat-directory assumption in the task brief does not hold: the validator was
written for "arbitrary directory layout (files are found recursively)" from the start
([validator.py:11–14](../db/validator.py#L11)).

The SHA is therefore also identical to run 3's, so runs 3 and 4 used a bit-identical instrument
and the entire delta is attributable to the dataset.

Discovery result against the new layout: **49 of 49 spec tables found, 0 missing, 0 extra CSVs.**
One table is physically duplicated — `inventory_transactions.csv` exists in both `inventory/`
and `outcomes/` and the two files are **byte-identical** (`aedbd757ba9c…`). The validator
reported the duplicate and kept `outcomes/`; because the files are identical the choice does not
affect any measurement.

## Control — did not move

| run | result |
|---|---|
| run 1 | 167 passed / 2 failed / 4 skipped (frozen instrument) |
| run 2 | 172 / 2 / 4 |
| run 3 | 172 / 2 / 4 |
| **run 4** | **172 / 2 / 4** |

`diff` of run 4's control transcript against run 3's is clean over all 605 shared lines except
two non-measurement lines: the spec path (`db/…` vs the v3 folder) and run 3's trailing
"report written" line. **Every check line is byte-identical.** The two standing failures are
unchanged: `PO lines arriving late` 12.8% and its `[2019-2025]` twin 13.7% against a 15–25%
band. The instrument did not move.

---

## 2. Folder map and schema reconciliation

### 2a. Inventory

50 CSV files, 49 distinct spec tables, 26,030,275 data rows — a ~52× scale-up on run 3.

| folder | tables | rows |
|---|---|---|
| `channel/` | 1 | 5,866,280 |
| `inventory/` | 4 (incl. the duplicate) | 3,250,332 |
| `logistics/` | 4 | 4,308,308 |
| `master/` | 23 | 1,536,070 |
| `model/` | 7 | 3,566,305 |
| `other_weekly/` | 2 | 2,463,750 |
| `outcomes/` | 4 (incl. the duplicate) | 1,026,930 |
| `po/` | 5 | 4,012,300 |

Per table, with column count and date span:

| folder | table | rows | cols | span |
|---|---|---|---|---|
| channel | channel_performance_weekly | 5,866,280 | 20 | 2019-01-07 → 2025-12-29 |
| inventory | inventory_snapshots | 2,190,332 | 10 | 2019-01-07 → 2025-12-29 |
| inventory | inventory_transactions | 1,000,000 | 9 | 2019-01-07 → 2026-01-05 |
| inventory | production_actual | 30,000 | 9 | 2016-01-29 → 2025-12-31 |
| inventory | production_plan | 30,000 | 11 | 2016-01-01 → 2025-12-02 |
| logistics | asn | 1,077,077 | 8 | 2019-01-08 → **2020-04-30** |
| logistics | goods_receipts | 1,077,077 | 6 | 2019-01-09 → **2020-05-16** |
| logistics | grn_lines | 1,077,077 | 9 | 2019-01-09 → **2020-05-16** |
| logistics | quality_inspections | 1,077,077 | 9 | 2019-01-10 → **2020-06-04** |
| master | alternate_sources | 6,000 | 9 | — |
| master | bom | 40,000 | 10 | — |
| master | business_units | 5 | 3 | — |
| master | calendar | 25,536 | 13 | 2016-01-04 → 2025-12-29 |
| master | customers | 60 | 6 | — |
| master | logistics_lanes | 2,000 | 8 | — |
| master | part_costs | 12,000 | 6 | — |
| master | part_plant | 6,000 | 11 | — |
| master | parts | 7,000 | 13 | 2014-01-01 → 2025-12-31 |
| master | plants | 7 | 11 | 2005-01-01 → 2011-01-01 |
| master | product_economics | 2,500 | 5 | — |
| master | products | 2,500 | 9 | 2014-01-01 → 2025-12-31 |
| master | sourcing_channels | 16,072 | 13 | 2016-01-01 → 2023-12-01 |
| master | supplier_allocation | 15,334 | 9 | **2019-01-05 only** |
| master | supplier_audits | 5,000 | 8 | 2019-01-01 → 2025-12-01 |
| master | supplier_capacity | 1,288,056 | 11 | 2019-01-01 → 2025-12-03 |
| master | supplier_contracts | 12,000 | 11 | — |
| master | supplier_financials | 5,250 | 6 | — |
| master | supplier_quality_ppm | 84,000 | 7 | 2019-12-05 → 2025-12-05 |
| master | supplier_sites | 750 | 9 | — |
| master | supplier_upstream | 250 | 8 | — |
| master | suppliers | 750 | 13 | 2010-01-01 → 2019-01-01 |
| master | tooling | 5,000 | 7 | — |
| model | dataset_coverage | 49 | 10 | 2016-01-04 → 2026-01-03 |
| model | inventory_position_weekly | 2,190,000 | 18 | 2019-01-07 → 2025-12-29 |
| model | model_outputs | 1,000 | 14 | 2026-01-03 |
| model | plan_drift_features | 80,000 | 15 | 2019-01-01 → 2025-12-03 |
| model | revealed_capacity_monthly | 1,288,056 | 11 | 2019-01-01 → 2025-12-01 |
| model | snapshots | 1,200 | 11 | 2019-05-27 → 2026-01-03 |
| model | training_labels | 6,000 | 15 | 2019-05-27 → 2026-01-04 |
| other_weekly | part_demand_weekly | 2,190,000 | 8 | 2019-01-07 → 2025-12-29 |
| other_weekly | supplier_performance_weekly | 273,750 | 21 | 2019-01-07 → 2025-12-29 |
| outcomes | expedite_events | 13,394 | 10 | 2019-01-22 → **2020-05-18** |
| outcomes | inventory_transactions | 1,000,000 | 9 | duplicate of `inventory/` |
| outcomes | line_stop_events | 142 | 13 | 2019-02-02 → **2020-03-01** |
| outcomes | shortage_events | 13,394 | 11 | 2019-01-21 → **2020-05-03** |
| po | po_line_revisions | 97,114 | 10 | 2019-01-13 → **2020-02-11** |
| po | po_line_schedules | 1,215,186 | 7 | 2019-01-07 → **2020-04-04** |
| po | po_lines | 900,000 | 12 | 2019-01-07 → **2020-03-31** |
| po | purchase_orders | 900,000 | 11 | 2019-01-07 → **2020-01-08** |
| po | supplier_acknowledgements | 900,000 | 7 | 2019-01-11 → **2020-01-31** |

**Missing spec tables: none. Non-spec CSVs: none.** Non-CSV files present in every folder
(`README_V4.md`, `local_validation.json`, `v4_parameters.json`, `validation_report.md`, plus
`master/requirements.md`) are ignored by discovery and cause no `extra tables` finding.

**The single most consequential fact in this table:** every transactional table stops between
**2020-01-08 and 2020-06-04**, while every derived weekly store runs to **2025-12-29**. The PO
and receipt history covers roughly 15 months; the feature stores cover 7 years. For ~82% of the
weekly panel's span there are no transactions underneath it at all. That alone makes the
conservation identity unsatisfiable, and §9's Q2 confirms it.

### 2b. Three-way schema reconciliation

**List 1 — in `dataset_structure.md`, absent from `schema.sql`: 1 (and it is not a staleness finding)**

| table | column |
|---|---|
| `supplier_performance_weekly` | *(whole table — defined by reference in the spec, "same 20 columns with `supplier_id` in place of `channel_id`, plus `active_channel_count`"; no column table exists to compare)* |

**List 2 — in `schema.sql`, absent from `dataset_structure.md`: 4**

| table | column |
|---|---|
| `inventory_position_weekly` | `consumption_4w` |
| `inventory_position_weekly` | `consumption_13w` |
| `model_outputs` | `p10` |
| `model_outputs` | `p90` |

All four are the **known markdown ambiguities the validator already documents**: the spec writes
"`consumption_4w` / `_13w`" and "`p10` / `p90`" as two column names inside a single table cell,
which the markdown parser cannot split, so it takes them from the DDL and reports the ambiguity.
They are artefacts of an ambiguous spec row, not of a stale DDL.

**List 3 — CSV columns matching neither source, or spec columns missing from the CSV: 4**

| table | column | finding |
|---|---|---|
| `goods_receipts` | `grn_id` | **in the spec, missing from the CSV.** The header reads `grn_number,grn_number,supplier_id,plant_id,receipt_ts,recorded_ts` — `grn_number` is written **twice** and the primary key `grn_id` is absent |
| `supplier_capacity` | `s` | present in the CSV, declared nowhere (values `1`) |
| `supplier_capacity` | `mo` | present in the CSV, declared nowhere (values `2019-01-01`) |
| `supplier_capacity` | `month` | present in the CSV, declared nowhere (values `2019-01-01`) |

The three `supplier_capacity` extras are scratch columns left in by
[`fix_capacity_only.py`](../db/generator/fix_capacity_only.py), which creates `s` and `mo` as
join keys and writes the frame back without dropping them.

**Type disagreements between the two sources on shared columns: 0.**

### Verdict: `schema.sql` is current, not stale

Every one of the 49 spec tables exists in `schema.sql`; no table appears in one source and not
the other; and there is **not a single base-type disagreement** between the markdown and the DDL
on any shared column. The only asymmetries are the four columns the spec itself renders
ambiguously, which the validator already resolves against `schema.sql` and reports as
ambiguities rather than silently patching.

**Consequence for the run-1→run-4 comparison: none.** Because `schema.sql` is current and
agrees with the spec, the resolution rule that runs 1–3 used — markdown decides the column set,
`schema.sql` decides types, nullability, enum sets, PK and FK
([`build_expected()`](../db/validator.py#L505), "Markdown decides the column SET; schema.sql
decides everything else") — was measuring the same thing then that it measures now. No check in
the trend table needs an asterisk on this account.

For the record, the type/enum, nullability and PK/FK checks that fail in run 4 were resolved as
follows: **enum sets** from `schema.sql`'s `CHECK (col IN (...))`; **nullability** from
`schema.sql`'s `NOT NULL` (the markdown has no such notion); **PK** from `schema.sql`'s
`PRIMARY KEY`, falling back to its `UNIQUE INDEX` where no PK is declared
(`inventory_position_weekly`); **column set** from `dataset_structure.md`.

---

## 3. Generator audit

`db/generator/` is **632 lines across 12 Python files**. They are sequential patch scripts —
`build_v4_fast.py` (265 lines) then `continue_v4.py`, `complete_v4.py`, `finish_remaining.py`,
`finish_weekly.py`, `finish_derived_fast.py`, `finalize_v4.py`, `finish_v4_metadata.py`,
`fast_transactions.py`, `finish_stale_temp.py`, `rewrite_stale.py`, `fix_capacity_only.py` —
several of which rewrite the same table more than once. There is no single simulator loop.

### 3.4 (lead) — Mechanism vs. outcome: **the dataset is fitted, not generated**

§4 of `V4_Corrections.md` states the rule: *"If a number appears in the right-hand column
[measured outcomes] and also appears anywhere in the generator as a constant, a clip, a post-hoc
adjustment, or an assertion, that is the defect this section exists to prevent."* Every hit
below is a measured outcome appearing as an input.

| # | file:line | code | outcome being set |
|---|---|---|---|
| 1 | `fix_capacity_only.py:3` | `ratio=m.load/m.cap.clip(lower=1); scale=max(.05,float(ratio.quantile(.75))); cap.capacity_qty_per_month=(cap.capacity_qty_per_month*scale)…` | **capacity observability.** Capacity is rescaled by the 75th percentile of load/capacity, which by construction makes ≈25% of supplier-months bind — the centre of the published 20–30% band |
| 2 | `build_v4_fast.py:81-82` | `# activity prob tuned to ~18.6% traded, 0 cold start`<br>`active=(rng2.random(N)<(0.19*traded[ci]))` | **zero-order channel-week share.** The comment says "tuned"; 0.19 yields the measured 82.1% against the 60–99% band and the reference world's 81.4% |
| 3 | `v4_parameters.json` | `"capacity_calibration_factor": 0.3` | **capacity observability**, declared as a mechanism parameter but named a calibration factor |
| 4 | `build_v4_fast.py:100` | `# Make history predictive and constrained months materially worse` | **joint probes 1 and 2**, named as targets in a source comment |
| 5 | `rewrite_stale.py:4`, `finish_stale_temp.py:6` | `ld=chunk.channel_id.map(mp)…; trait=np.clip((ld-30)/45,0,2); extra=rng.poisson(1+trait*3); …weeks_since_last_activity=np.where(inactive,np.maximum(1,base+extra),0)` | **joint probe 4.** Staleness is written as a function of the channel's *contracted lead time*, which manufactures the staleness↔lead-time-variance correlation the probe measures |
| 6 | `fix_capacity_only.py:3`, `complete_v4.py:19` | `rc['revealed_capacity_est']=(rc.declared_capacity_qty*.9).astype(int)`; `max_delivered_12m`/`p95_delivered_12m` = `declared_capacity_qty` | **revealed capacity.** "Revealed" quantities are a deterministic ×0.9 (and ×1.0) of the declared figure — nothing is revealed from behaviour |
| 7 | `build_v4_fast.py:89` | `season=1+.18*np.sin(2*np.pi*weekno/52)` | **demand peak/trough ratio** (measured 1.45×, band > 1.10) |
| 8 | data / `finish_v4_metadata.py:16` | `capacity_basis` values are `unobservable` / `inferred` / `reported` | **capacity observability written as a literal data value**, on 100% of 1,288,056 rows — and all three are outside the declared enum `{estimated, audited, declared, contracted}` |

Item 1 is the one §4 singles out by name. Its worked example is:

> ✗ Wrong: decide 25% of supplier-months are observable, mark them, generate accordingly.
> ✓ Right: simulate latent capacity K … apply `delivered = min(K, ordered)`. Observability falls
> out wherever `ordered ≥ K`.

`fix_capacity_only.py` — a file whose name is itself the finding — does the first. It reads the
realised load, computes the 75th percentile of the load/capacity ratio, and multiplies every
capacity value by it so that the top quartile of months bind. Observability is set, then
`constrained_month_flag` is written from it. **That alone means the dataset is fitted, however
well it validates** — and the effect is visible in the data: `capacity_utilisation_observed`
takes exactly **two distinct values across 1,288,056 rows, 0.0 and 2.0** (the two ends of
`.clip(0,2)`), and `constrained_month_flag` is perfectly collinear with which of the two a row
got. There is no graded utilisation anywhere in the table.

Item 5 deserves emphasis because it is new in kind: run 3's fitting targeted the *marginal*
bands. Run 4 also fits a **joint probe** — the staleness↔volatility correlation that run 3
failed with the wrong sign is now +0.46, and it is positive because staleness was written as a
function of lead time in a post-hoc rewrite pass, not because intermittent channels became less
predictable.

**Verdict: fitted.** Two of the six joint probes now read the way §9 asks, and in both cases the
generator contains the line that makes them read that way.

### 3.1 — Recording layer (§1): real process, wrong scale, and four tables violate it

The lag function is genuinely right-skewed and state-dependent:

```python
# build_v4_fast.py:16-19
def lag(table='x',stress=False):
    base={'po':1.2,'ack':2.0,'receipt':1.0,'revision':2.0,'inv':3.0,'quality':1.5,'short':2.5}.get(table,1.5)
    x=rng.lognormal(math.log(base),0.95)*(1+rng.uniform(.5,1.8) if stress else 1)
    return max(.05,float(x))
```

Log-normal, per-table base, a stress multiplier, and `max(.05, …)` so this function cannot
return a negative. Measured on the data, the shape is real:

| table | P50 | P90 | max | sd/\|mean\| | negative lags |
|---|---|---|---|---|---|
| `po_lines` | 0.04 d | 0.21 d | 4.0 d | 1.535 | 0 |
| `grn_lines` | 1.00 d | 3.38 d | 70.2 d | 1.193 | 0 |
| `po_line_revisions` | 0.04 d | 0.25 d | 3.6 d | 1.452 | 0 |
| `shortage_events` | 2.16 d | 8.02 d | 116.5 d | 1.319 | **180** |

Three defects against §1:

1. **The scale is far too small on the PO spine.** `po_lines` has a P50 lag of **one hour** and
   a P90 of five hours. The lag is a real distribution but it is so short that a fact is
   essentially never displaced across a reporting week — which is exactly what the as-of contract
   tests. `po_lines: two timestamps differ` reads **0.0%** against a ≥ 2% gate (run 3: 60.5%), as
   do `purchase_orders`, `asn`, `inventory_snapshots` and `supplier_allocation`.
2. **Seven tables have a literally constant lag** — `sd/|mean| = 0.000` on `inventory_snapshots`,
   `inventory_transactions`, `production_actual`, `production_plan`, `supplier_allocation`,
   `supplier_capacity` and `supplier_quality_ppm`. These tables are written by later patch
   scripts that do not call `lag()` at all and add a fixed offset instead. This is run 1's
   headline defect, returned.
3. **Lag is negative on four tables.** `supplier_audits` 4,584 rows, `expedite_events` 297,
   `shortage_events` 180, `line_stop_events` 2 — rows recorded before they happened. §1: *"Lag is
   never negative. A row recorded before its own event is a data-integrity failure."*

**Are some records never entered?** No. There is no drop or suppression step anywhere in the
generator, and the consequence is measurable: `label_censored` is `False` on **all 6,000** label
rows and `censor_time` is empty on all of them. All five `something is censored` checks fail at
**0.0%** against a 1–60% band. The mechanism §1 names as "what makes a fact unobservable rather
than merely late" was not built.

### 3.2 — Disruption entry points (§2): right node, only one class

Disruptions enter at supplier capacity and at transit, as required:

```python
# build_v4_fast.py:95-102
covid=((dates>=np.datetime64('2020-03-23'))&(dates<=np.datetime64('2021-09-30'))&(sup%6==0))
strike=((sup%53==0)&np.isin(month,[6,7]))
supply=np.clip(stress+.65*covid+.4*strike,0,1.6)
fill=np.clip(1-.34*supply+rng2.normal(0,.06,N),0,1)
lateprob=np.clip(.08+.30*supply,0,.75); late=(rng2.random(N)<lateprob)&active
lead_actual=np.where(active,lead*(1+.45*supply)+rng2.lognormal(np.log(1.5),.45,N),0)
```

`supply` feeds `fill` (capacity binding) and `lead_actual` (transit). It does **not** feed
material requirement. §2's correction is implemented.

**But there is no demand-side shock class.** Demand is a pure seasonal sine plus a linear trend
and lognormal noise:

```python
# build_v4_fast.py:218
gross=max(1,int(base*(1+.18*math.sin(2*math.pi*wd.isocalendar().week/52))*(1+.03*(wd.year-2019))*rng.lognormal(0,.12)))
```

No programme cancellation, no model-year pull-in, no demand event of any kind. §2 requires *both*
classes to exist and to be "distinguishable in their downstream signature", precisely so that
`demand_drift` and `capacity_strain` can be separated. One class is missing, so that separation
is not demonstrable — and in fact `demand_drift` is not generated from demand at all (§3.3).

Selection is also deterministic rather than stochastic: `sup%6==0` picks exactly one supplier in
six for COVID and `sup%53==0` one in 53 for the strike, for every run.

### 3.3 — Feedback loop (§3): two arcs of six, and the critical one is absent

| §3 required arc | present? | evidence |
|---|---|---|
| Shortage → expedite events | **yes** | `finish_remaining.py:13` builds one expedite per shortage row (`expedite_events` = `shortage_events` = 13,394 rows exactly) |
| Shortage → line stop | **yes** | `finish_remaining.py:12` `stop=sh[sh.shortage_qty>60].sample(frac=.3,random_state=22)` |
| Shortage → PO line revisions | **no** | revisions are generated from `late` only: `initiated_by=np.where(late,'supplier','buyer')`, `reason_code=np.where(late,'supplier_delay','buyer_cancellation')` (`fast_transactions.py:17`). No shortage term |
| Shortage → alternate sourcing / allocation | **no** | `alternate_sources` is static: `alt=1+((s+97)%750)`, `effective_from='2018-01-01'` (`build_v4_fast.py:65`). `supplier_allocation` is `allocation_pct=.5` for every row on a single date `2019-01-01`, `change_reason='initial'` (`finish_v4_metadata.py:18`) — the data confirms all 15,334 rows share one date |
| **Alternate sourcing → receiving supplier's utilisation** | **no** | nothing anywhere in the 632 lines raises a second supplier's load in response to re-sourcing |
| Line stop → production plan | **no** | `production_plan` is generated independently of `line_stop_events` |
| Sustained performance → allocation | **no** | allocation never changes |

§3 is explicit about which arc matters: *"shortage at part P forces re-sourcing to alternate
supplier S′, which raises S′'s utilisation, which degrades S′'s fill rate on its other channels.
That path — and only that path — is what a graph neural network reads."*

**That arc is absent.** No supplier's load is ever a function of another supplier's failure.
Failure does not propagate across the graph, so the heterogeneous-graph premise remains
unsupported on this data — the same conclusion §3 predicted for a chain that runs strictly
downward.

### 3.5 — Scale (§5): configured correctly, realised with ID collisions

| quantity | §5 reference | configured (`v4_parameters.json`) | **realised in the data** |
|---|---|---|---|
| Sourcing channels | 16,072 | 16,072 | 16,072 in `sourcing_channels`, but **15,621 distinct `channel_id` in the weekly panel** |
| Channels that ever traded | 15,334 | 15,334 | **15,069** |
| Cold-start channels | 738 (4.6%) | 738 | **552 (3.5%)** |
| Channel-weeks per channel | 386.8 | 365 | median **365**, min 151, **max 730** |
| PO lines per channel per year | 7.02 | — | **39.73** over the PO span; **7.45** over 2019–2025 |
| Zero-order channel-weeks | 81.4% | — | **82.1%** |
| Contiguous weekly panel | 100% | — | 100.0% (check passes) |

The channel population is the right size — a genuine and substantial fix on run 3's 320. But the
panel was written as 16,072 × 365 = 5,866,280 rows against only **15,621 distinct channel ids**,
so 451 ids carry two full 365-week series each. That produces **165,014 duplicate
`(channel_id, week_start)` primary keys** and is why `channel_performance_weekly` fails Tier 1.

`PO lines per channel per year` reads 39.73 on the full span because 900,000 lines are
compressed into the ~15 months the PO tables cover; the windowed figure of 7.45 sits near the
reference only because the window divides by seven years the lines do not span.

### 3.6 — Reproduction floor (§6): one seed, no variance band; hash consistent but partial

**Two seeds: no.** `seed: 1001` is the only seed in the generator or in any
`v4_parameters.json`. No second seed was generated and **no per-metric variance band is
published anywhere**. §6's requirement — *"Generate at minimum two seeds under identical code and
publish the per-metric variance band"* — is not met, which is the one failure that makes this
very trend table harder to read: no movement between runs 3 and 4 can be separated from seed
noise by evidence in the dataset.

**`code_commit` consistent: yes.** `snapshots.code_commit` holds a single distinct value
(`f8975f6872afbbbedd8ccd28de2565e04498a34d670d24127f9c4a27766bdc2c`) across all 1,200 rows, and
the validator's `code_commit is consistent` check passes.

**Does it match a hash of the current source? Partially — and the omission matters.** It matches
exactly the SHA-256 of six concatenated files:

```
build_v4_fast.py, continue_v4.py, fast_transactions.py,
finish_remaining.py, finalize_v4.py, finish_v4_metadata.py     -> f8975f68…  (verified: MATCH)
```

But `db/generator/` contains **twelve** `.py` files. The six omitted are `complete_v4.py`,
`finish_derived_fast.py`, `finish_weekly.py`, `finish_stale_temp.py`, `rewrite_stale.py` and
**`fix_capacity_only.py`** — which includes both post-hoc passes identified in §3.4 as fitting
the capacity band and the staleness probe. The recorded provenance hash covers the simulator and
excludes the calibration. The hash over all twelve files is
`2d2d5f32005f41700dd091f1d51f9935447e00c97abcf41d0e5309d25928726b`.

One point of credit: `local_validation.json` reports
`"supplier_month_constrained": "15.3% (local; below reference band and must be rechecked
externally)"` and `"external_validator": "not available in runtime"`. That is §7's honest-number
commitment being kept.

---

## 4. Four-run trend table

111 checks have failed in at least one run; 73 still fail in run 4.

| check | run 1 | run 2 | run 3 | run 4 | band | r1→r2→r3→r4 |
|---|---|---|---|---|---|---|
| `PO lines arriving late` | 82.3% | 19.8% | 19.8% | **61.9%** | 15%-25% | **FAIL** → ok → ok → **FAIL** |
| `PO lines arriving late [2019-2025]` | 82.7% | 20.0% | 20.0% | **61.9%** | 15%-25% | **FAIL** → ok → ok → **FAIL** |
| `PO lines with fill < 1.0` | 72.7% | 20.0% | 15.9% | **43.9%** | 8%-15% | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `PO lines with fill < 1.0 [2019-2025]` | 72.2% | 20.1% | 16.6% | **43.9%** | 8%-15% | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `arrival_timing: something is censored` | _absent_ | _absent_ | _absent_ | **0.0%** | 1%-60% | – → – → – → **FAIL** |
| `asn` | 7,000 rows | 17,218 rows | 17,218 rows | **1,077,077 rows** | clean | ok → ok → ok → **FAIL** |
| `asn: two timestamps differ` | 14.5% | 59.9% | 59.9% | **0.0%** | >= 2% | ok → ok → ok → **FAIL** |
| `calendar` | 37,548 rows | 37,555 rows | 37,555 rows | **25,536 rows** | clean | **FAIL** → ok → ok → **FAIL** |
| `capacity unobservable` | 11.5% | 2.4% | 2.4% | **51.0%** | >= 70% | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `capacity unobservable [2019-2025]` | 11.5% | 2.2% | 2.2% | **51.0%** | >= 70% | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `capacity_strain: something is censored` | 0.0% | 3.6% | 3.6% | **0.0%** | 1%-60% | **FAIL** → ok → ok → **FAIL** |
| `channel store conserves ordered units` | 138,615,579 vs 4,114,197 | 2,878,869 vs 1,944,910 | 1,944,910 vs 1,944,910 | **127,761,134 vs 113,893,014** | exact | **FAIL** → **FAIL** → ok → **FAIL** |
| `channel store conserves received units` | 110,778,603 vs 2,992,098 | 2,740,051 vs 1,870,498 | 1,870,498 vs 1,870,498 | **122,774,945 vs 118,923,111** | exact | **FAIL** → **FAIL** → ok → **FAIL** |
| `channel_performance_weekly` | 55,000 rows | 43,800 rows | 43,800 rows | **5,866,280 rows** | clean | **FAIL** → **FAIL** → ok → **FAIL** |
| `dataset_coverage` | 46 rows | 48 rows | 48 rows | **49 rows** | clean | ok → ok → ok → **FAIL** |
| `demand_drift: entity_type matches the spec` | 1,902/2,403 wrong | 0/1,200 wrong | 0/1,200 wrong | **1,200/1,200 wrong** | 0 | **FAIL** → ok → ok → **FAIL** |
| `demand_drift: something is censored` | 0.0% | 4.6% | 4.6% | **0.0%** | 1%-60% | **FAIL** → ok → ok → **FAIL** |
| `derived store buckets on visible week` | neither | neither | max(event_week, recorded_week) | **neither** | max(event,recorded) | **FAIL** → **FAIL** → ok → **FAIL** |
| `expedite_events` | 1,400 rows | 2,005 rows | 2,005 rows | **13,394 rows** | clean | ok → ok → ok → **FAIL** |
| `expedite_events: recorded_ts >= event` | 0 | 0 | 0 | **297** | 0 | ok → ok → ok → **FAIL** |
| `expedites per year` | 93.7 | 131.6 | 131.6 | **10213.3** | 100-150 | **FAIL** → ok → ok → **FAIL** |
| `fill-rate mass at exactly 0` | 31.8% | 2.0% | 2.0% | **0.0%** | 1%-3% | **FAIL** → ok → ok → **FAIL** |
| `fill-rate mass at exactly 0 [2019-2025]` | 32.1% | 1.9% | 1.9% | **0.0%** | 1%-3% | **FAIL** → ok → ok → **FAIL** |
| `fill-rate mass at exactly 1.0` | 27.3% | 80.0% | 84.1% | **56.1%** | >= 60% | **FAIL** → ok → ok → **FAIL** |
| `fill-rate mass at exactly 1.0 [2019-2025]` | 27.8% | 79.9% | 83.4% | **56.1%** | >= 60% | **FAIL** → ok → ok → **FAIL** |
| `fill_rate: entity_type matches the spec` | 1,892/2,413 wrong | 0/1,200 wrong | 0/1,200 wrong | **1,200/1,200 wrong** | 0 | **FAIL** → ok → ok → **FAIL** |
| `fill_rate: something is censored` | 0.0% | 4.7% | 4.7% | **0.0%** | 1%-60% | **FAIL** → ok → ok → **FAIL** |
| `goods_receipts` | 23,000 rows | 17,218 rows | 17,218 rows | **1,077,077 rows** | clean | ok → ok → ok → **FAIL** |
| `grn_lines` | 23,000 rows | 17,218 rows | 17,218 rows | **1,077,077 rows** | clean | ok → ok → ok → **FAIL** |
| `inventory_position_weekly` | 50,000 rows | 10,000 rows | 10,000 rows | **2,190,000 rows** | clean | **FAIL** → ok → ok → **FAIL** |
| `inventory_snapshots` | 60,000 rows | 15,340 rows | 15,340 rows | **2,190,332 rows** | clean | **FAIL** → ok → ok → **FAIL** |
| `inventory_snapshots: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.689 | sd/\|mean\|=0.689 | **sd/\|mean\|=0.000** | >= 0.1 | **FAIL** → ok → ok → **FAIL** |
| `inventory_snapshots: two timestamps differ` | 0.0% | 39.8% | 39.8% | **0.0%** | >= 2% | **FAIL** → ok → ok → **FAIL** |
| `inventory_transactions: lag is not constant` | sd/\|mean\|=2.062 | sd/\|mean\|=0.695 | sd/\|mean\|=0.695 | **sd/\|mean\|=0.000** | >= 0.1 | ok → ok → ok → **FAIL** |
| `lead-time distribution is right-skewed` | +0.05 | +0.88 | +0.88 | **+0.30** | >= 0.5 | **FAIL** → ok → ok → **FAIL** |
| `lead-time distribution is right-skewed [2019-2025]` | +0.05 | +0.93 | +0.93 | **+0.30** | >= 0.5 | **FAIL** → ok → ok → **FAIL** |
| `line stops per year` | 14.7 | 21.7 | 21.7 | **108.3** | 15-25 | **FAIL** → ok → ok → **FAIL** |
| `line_stop_events` | 220 rows | 330 rows | 330 rows | **142 rows** | clean | ok → ok → ok → **FAIL** |
| `line_stop_events: recorded_ts >= event` | 0 | 0 | 0 | **2** | 0 | ok → ok → ok → **FAIL** |
| `part_demand_weekly horizon_days identity` | 0/50,000 | 0/7,800 | 0/7,800 | **2,190,000/2,190,000** | 0 | ok → ok → ok → **FAIL** |
| `part_demand_weekly is strictly forward-looking` | 0/50,000 | 0/7,800 | 0/7,800 | **2,190,000/2,190,000** | 0 | ok → ok → ok → **FAIL** |
| `plan_drift_features` | 20,000 rows | 43,052 rows | 43,052 rows | **80,000 rows** | clean | warn → ok → ok → **FAIL** |
| `po_line_revisions: two timestamps differ` | 44.0% | 60.1% | 60.1% | **0.1%** | >= 2% | ok → ok → ok → **FAIL** |
| `po_lines: two timestamps differ` | 0.0% | 60.5% | 60.5% | **0.0%** | >= 2% | **FAIL** → ok → ok → **FAIL** |
| `production_actual` | _skip_ | 48,012 rows | 48,012 rows | **30,000 rows** | clean | skip → ok → ok → **FAIL** |
| `production_actual: lag is not constant` | _absent_ | sd/\|mean\|=0.753 | sd/\|mean\|=0.753 | **sd/\|mean\|=0.000** | >= 0.1 | – → ok → ok → **FAIL** |
| `production_plan` | _skip_ | 64,837 rows | 64,837 rows | **30,000 rows** | clean | skip → **FAIL** → ok → **FAIL** |
| `production_plan: lag is not constant` | _absent_ | sd/\|mean\|=0.750 | sd/\|mean\|=0.750 | **sd/\|mean\|=0.000** | >= 0.1 | – → ok → ok → **FAIL** |
| `purchase_orders: two timestamps differ` | 0.0% | 60.2% | 60.2% | **0.0%** | >= 2% | **FAIL** → ok → ok → **FAIL** |
| `right-censored tail is the right size` | 0.48x | 0.38x | 0.38x | **0.23x** | 0.3-1.2x | ok → ok → ok → **FAIL** |
| `right-censored tail is the right size [2019-2025]` | 0.91x | 1.13x | 1.13x | **0.23x** | 0.3-1.2x | ok → ok → ok → **FAIL** |
| `shortage events per year` | 120.5 | 235.6 | 235.6 | **10213.3** | 180-250 | **FAIL** → ok → ok → **FAIL** |
| `shortage_events` | 1,800 rows | 3,580 rows | 3,580 rows | **13,394 rows** | clean | ok → ok → ok → **FAIL** |
| `shortage_events: recorded_ts >= event` | 0 | 0 | 0 | **180** | 0 | ok → ok → ok → **FAIL** |
| `shortage_qty: entity_type matches the spec` | 1,882/2,334 wrong | 0/1,200 wrong | 0/1,200 wrong | **1,200/1,200 wrong** | 0 | **FAIL** → ok → ok → **FAIL** |
| `shortage_qty: something is censored` | 0.0% | 3.8% | 3.8% | **0.0%** | 1%-60% | **FAIL** → ok → ok → **FAIL** |
| `structurally clean tables` | 37/49 | 40/49 | 48/49 | **29/49** | 49/49 | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `supplier store conserves ordered units` | 21,468,423 vs 4,114,197 | 2,878,869 vs 1,944,910 | 1,944,910 vs 1,944,910 | **129,998,532 vs 113,893,014** | exact | **FAIL** → **FAIL** → ok → **FAIL** |
| `supplier store conserves received units` | 17,303,481 vs 2,992,098 | 2,740,051 vs 1,870,498 | 1,870,498 vs 1,870,498 | **124,926,841 vs 118,923,111** | exact | **FAIL** → **FAIL** → ok → **FAIL** |
| `supplier-months constrained` | 88.5% | 97.6% | 97.6% | **49.0%** | 20%-30% | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `supplier-months constrained [2019-2025]` | 88.5% | 97.8% | 97.8% | **49.0%** | 20%-30% | **FAIL** → **FAIL** → **FAIL** → **FAIL** |
| `supplier_acknowledgements` | 15,600 rows | 9,214 rows | 9,214 rows | **900,000 rows** | clean | ok → **FAIL** → ok → **FAIL** |
| `supplier_acknowledgements: two timestamps differ` | 32.7% | 60.4% | 60.4% | **0.1%** | >= 2% | ok → ok → ok → **FAIL** |
| `supplier_allocation: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.720 | sd/\|mean\|=0.720 | **sd/\|mean\|=0.000** | >= 0.1 | **FAIL** → ok → ok → **FAIL** |
| `supplier_allocation: two timestamps differ` | 100.0% | 60.8% | 60.8% | **0.0%** | >= 2% | ok → ok → ok → **FAIL** |
| `supplier_audits` | 500 rows | 500 rows | 500 rows | **5,000 rows** | clean | ok → ok → ok → **FAIL** |
| `supplier_audits: recorded_ts >= event` | 0 | 0 | 0 | **4,584** | 0 | ok → ok → ok → **FAIL** |
| `supplier_capacity` | 15,930 rows | 10,639 rows | 10,639 rows | **1,288,056 rows** | clean | **FAIL** → ok → ok → **FAIL** |
| `supplier_capacity: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.527 | sd/\|mean\|=0.527 | **sd/\|mean\|=0.000** | >= 0.1 | **FAIL** → ok → ok → **FAIL** |
| `supplier_quality_ppm: lag is not constant` | sd/\|mean\|=0.591 | sd/\|mean\|=0.168 | sd/\|mean\|=0.168 | **sd/\|mean\|=0.000** | >= 0.1 | ok → ok → ok → **FAIL** |
| `supplier_upstream` | 180 rows | 300 rows | 300 rows | **250 rows** | clean | ok → **FAIL** → ok → **FAIL** |
| `suppliers` | 50 rows | 120 rows | 120 rows | **750 rows** | clean | ok → ok → ok → **FAIL** |
| `training_labels` | 12,000 rows | 6,000 rows | 6,000 rows | **6,000 rows** | clean | ok → **FAIL** → **FAIL** → **FAIL** |
| `all spec tables present` | 47/49 | 49/49 | 49/49 | **49/49** | 49/49 | **FAIL** → ok → ok → ok |
| `alternate_sources` | 144 rows | 120 rows | 120 rows | **6,000 rows** | clean | ok → **FAIL** → ok → ok |
| `arrival_week: entity_type matches the spec` | 1,919/2,419 wrong | 0/1,200 wrong | 0/1,200 wrong | **_absent_** | 0 | **FAIL** → ok → ok → – |
| `arrival_week: something is censored` | 0.0% | 13.6% | 13.6% | **_absent_** | 1%-60% | **FAIL** → ok → ok → – |
| `asn: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.753 | sd/\|mean\|=0.753 | **sd/\|mean\|=1.745** | >= 0.1 | **FAIL** → ok → ok → ok |
| `capacity_strain: entity_type matches the spec` | 1,916/2,431 wrong | 0/1,200 wrong | 0/1,200 wrong | **0/1,200 wrong** | 0 | **FAIL** → ok → ok → ok |
| `capacity_strain: target is not a uniform random draw` | _absent_ | D=0.0160 p=0.926 | D=0.3146 p=1.29e-100 | **D=0.2727 p=1.74e-78** | p < 0.01 | – → **FAIL** → ok → ok |
| `demand_drift: target is not a uniform random draw` | _absent_ | D=0.0277 p=0.339 | D=0.4366 p=2.08e-191 | **D=0.1145 p=3.54e-14** | p < 0.01 | – → **FAIL** → ok → ok |
| `entity_id resolves for entity_type=channel` | 2,252/2,292 unresolved | 0/120 unresolved | 0/120 unresolved | **0/1,200 unresolved** | 0 | **FAIL** → ok → ok → ok |
| `entity_id resolves for entity_type=po_line` | 87/2,261 unresolved | 0/2,097 unresolved | 0/2,097 unresolved | **_absent_** | 0 | **FAIL** → ok → ok → – |
| `entity_id resolves for entity_type=supplier` | 2,244/2,248 unresolved | _absent_ | _absent_ | **_absent_** | 0 | **FAIL** → – → – → – |
| `expedite_events: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.708 | sd/\|mean\|=0.708 | **sd/\|mean\|=1.302** | >= 0.1 | **FAIL** → ok → ok → ok |
| `expedite_events: two timestamps differ` | 0.0% | 83.4% | 83.4% | **39.0%** | >= 2% | **FAIL** → ok → ok → ok |
| `goods_receipts: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.694 | sd/\|mean\|=0.694 | **sd/\|mean\|=1.208** | >= 0.1 | **FAIL** → ok → ok → ok |
| `grn_lines: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.684 | sd/\|mean\|=0.684 | **sd/\|mean\|=1.208** | >= 0.1 | **FAIL** → ok → ok → ok |
| `inventory_transactions: recorded_ts >= event` | 12,201 | 0 | 0 | **0** | 0 | **FAIL** → ok → ok → ok |
| `line_stop_events: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.705 | sd/\|mean\|=0.705 | **sd/\|mean\|=1.318** | >= 0.1 | **FAIL** → ok → ok → ok |
| `line_stop_events: two timestamps differ` | 0.0% | 83.6% | 83.6% | **39.4%** | >= 2% | **FAIL** → ok → ok → ok |
| `month-of-year seasonality in demand` | 1.03x peak/trough | 1.46x peak/trough | 1.46x peak/trough | **1.45x peak/trough** | > 1.10 | **FAIL** → ok → ok → ok |
| `month-of-year seasonality in demand [2019-2025]` | 1.03x peak/trough | 1.43x peak/trough | 1.43x peak/trough | **1.45x peak/trough** | > 1.10 | **FAIL** → ok → ok → ok |
| `part_demand_weekly` | 50,000 rows | 7,800 rows | 7,800 rows | **2,190,000 rows** | clean | **FAIL** → ok → ok → ok |
| `parts` | 250 rows | 500 rows | 500 rows | **7,000 rows** | clean | **FAIL** → **FAIL** → ok → ok |
| `po_line_revisions` | 6,000 rows | 5,121 rows | 5,121 rows | **97,114 rows** | clean | ok → **FAIL** → ok → ok |
| `po_line_revisions: recorded_ts >= event` | 2,562 | 0 | 0 | **0** | 0 | **FAIL** → ok → ok → ok |
| `po_lines: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.746 | sd/\|mean\|=0.746 | **sd/\|mean\|=1.540** | >= 0.1 | **FAIL** → ok → ok → ok |
| `purchase_orders: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.754 | sd/\|mean\|=0.754 | **sd/\|mean\|=1.540** | >= 0.1 | **FAIL** → ok → ok → ok |
| `quality_inspections: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.746 | sd/\|mean\|=0.746 | **sd/\|mean\|=1.205** | >= 0.1 | **FAIL** → ok → ok → ok |
| `revealed_capacity_monthly` | 4,000 rows | 7,560 rows | 10,651 rows | **1,288,056 rows** | clean | **FAIL** → ok → ok → ok |
| `shortage_events: lag is not constant` | sd/\|mean\|=0.000 | sd/\|mean\|=0.716 | sd/\|mean\|=0.716 | **sd/\|mean\|=1.319** | >= 0.1 | **FAIL** → ok → ok → ok |
| `shortage_events: two timestamps differ` | 0.0% | 82.6% | 82.6% | **38.7%** | >= 2% | **FAIL** → ok → ok → ok |
| `shortage_qty: target is not a uniform random draw` | _absent_ | D=0.0295 p=0.262 | D=0.9905 p=0 | **D=0.5558 p=0** | p < 0.01 | – → **FAIL** → ok → ok |
| `supplier_acknowledgements: recorded_ts >= event` | 4,227 | 0 | 0 | **0** | 0 | **FAIL** → ok → ok → ok |
| `supplier_financials` | 450 rows | 900 rows | 900 rows | **5,250 rows** | clean | **FAIL** → ok → ok → ok |
| `supplier_performance_weekly` | 17,899 rows | 24,820 rows | 24,820 rows | **273,750 rows** | clean | **FAIL** → **FAIL** → warn → warn |
| `weekly store is a panel` | 0.0% | 100.0% | 100.0% | **100.0%** | >= 90% | **FAIL** → ok → ok → ok |
| `weekly store is a panel [2019-2025]` | 0.0% | 100.0% | 100.0% | **100.0%** | >= 90% | **FAIL** → ok → ok → ok |
| `zero-order channel-weeks` | 0.0% | 80.2% | 86.0% | **82.1%** | 60%-99% | **FAIL** → ok → ok → ok |
| `zero-order channel-weeks [2019-2025]` | 0.0% | 80.2% | 86.0% | **82.1%** | 60%-99% | **FAIL** → ok → ok → ok |

---

## 5. Newly fixed since run 3 — **none**

**Not one check moved FAIL → pass between run 3 and run 4.** No check moved SKIP → pass either.
Every one of run 3's eight failures is still failing (§6), and 64 checks that passed in run 3
now fail (§7).

Three things did improve without a gate registering it, and they should be stated because they
are real:

- **Entity scale.** 16,072 channels against run 3's 120, with a 552-channel cold-start slice
  that run 3 did not have at all (§3.5).
- **Shortage provenance.** Shortages are now derived from acknowledgement shortfalls
  (`finish_remaining.py:10`: `mask = … & (po.ack_qty < po.qty_ordered)`), so 98.7% of them have
  an identifiable upstream antecedent against run 3's 1.4% (§8.3).
- **Staleness↔volatility sign.** +0.46 against run 3's −0.15 (§8.4) — though §3.4 item 5 shows
  the correlation was written in rather than emerging.

## 6. Still failing after three regenerations — 8 checks

These are run 3's eight failures, all still failing. "Moved since run 3" is the column that
matters: an unmoved value means the fix did not touch the mechanism.

| check | run 3 | run 4 | band | moved since run 3? |
|---|---|---|---|---|
| `training_labels` (Tier 1) | 6,000 rows, `censor_time` float-in-INTEGER | 6,000 rows, 1 type/enum violation | clean | **no — same table, same failure class** |
| `PO lines with fill < 1.0` | 15.9% | **43.9%** | 8–15% | yes — 28.0 pts **further out of band** |
| `PO lines with fill < 1.0 [2019-2025]` | 16.6% | **43.9%** | 8–15% | yes — 27.3 pts further out |
| `supplier-months constrained` | 97.6% | **49.0%** | 20–30% | yes — 48.6 pts toward band, still 19 pts over |
| `supplier-months constrained [2019-2025]` | 97.8% | **49.0%** | 20–30% | yes — 48.8 pts toward band |
| `capacity unobservable` | 2.4% | **51.0%** | ≥ 70% | yes — +48.6 pts toward band, still 19 pts short |
| `capacity unobservable [2019-2025]` | 2.2% | **51.0%** | ≥ 70% | yes — +48.8 pts toward band |
| `structurally clean tables` | 48/49 | **29/49** | 49/49 | yes — 19 tables **worse** |

The capacity pair is the one that moved most, and §3.4 explains how: it moved because
`fix_capacity_only.py` rescaled capacity to make a chosen quartile bind, not because an ordering
policy was adjusted and observability was allowed to fall out. It moved a long way and still
misses the band in both directions.

`training_labels` has now failed Tier 1 in three consecutive runs on the same column class. In
run 4 the failure is `censor_time` again — this time because the column is empty on all 6,000
rows while the label set declares no censoring at all.

## 7. Regressions and oscillations

### Regressions — 64 checks passed in run 3 and fail in run 4

This is the largest single-run regression in the series. Grouped by cause:

**The as-of contract collapsed again (14 checks).** `two timestamps differ` fell to 0.0% on
`po_lines` (was 60.5%), `purchase_orders` (60.2%), `asn` (59.9%), `inventory_snapshots` (39.8%)
and `supplier_allocation` (60.8%), and to 0.1% on `supplier_acknowledgements` and
`po_line_revisions`. `lag is not constant` returned to `sd/|mean| = 0.000` on
`inventory_snapshots`, `inventory_transactions`, `production_actual`, `production_plan`,
`supplier_allocation`, `supplier_capacity` and `supplier_quality_ppm`. See §3.1.

**`recorded_ts >= event` inversions returned (4 checks).** `supplier_audits` 4,584,
`expedite_events` 297, `shortage_events` 180, `line_stop_events` 2 — all were 0 in run 3.

**Conservation broke (5 checks).** Both stores, both quantities, plus the bucketing rule:
`max(event_week, recorded_week)` → `neither`. See §9 Q2.

**Label binding broke (7 checks).** `fill_rate`, `demand_drift` and `shortage_qty` all read
`1,200/1,200 wrong` on `entity_type` (run 3: `0/1,200`), and all five `something is censored`
checks fell from 3.6–13.6% to **0.0%**.

**Nineteen tables lost Tier 1** (`structurally clean tables` 48/49 → 29/49), including
`channel_performance_weekly` (165,014 duplicate PKs), `inventory_position_weekly` and
`inventory_snapshots` (332 each), `plan_drift_features` (27,500), `goods_receipts` (missing
`grn_id`), `grn_lines` (orphan FK), `asn` (NOT NULL nulls) and `supplier_capacity`
(`capacity_basis` wholly outside its declared enum).

**Rare-event rates exploded (3 checks).** `shortage events per year` 235.6 → **10,213.3**,
`expedites per year` 131.6 → **10,213.3**, `line stops per year` 21.7 → **108.3** — because
13,394 shortages and 13,394 expedites are compressed into the ~15 months the event tables span.

**Distributional shape degraded (7 checks).** `fill-rate mass at exactly 1.0` 84.1% → 56.1%,
`mass at exactly 0` 2.0% → 0.0%, `lead-time skewness` +0.88 → +0.30, `right-censored tail`
0.38× → 0.23×, `PO lines arriving late` 19.8% → 61.9%.

**Four checks disappeared rather than failing**, because the `arrival_week` task no longer
exists — the generator emits a task named **`arrival_timing`**
(`finish_v4_metadata.py:22`). `arrival_week: something is censored`, `: uncensored target
varies`, `: target is not a uniform random draw` and `: entity_type matches the spec` are all
absent from run 4 and are replaced by `arrival_timing` equivalents that carry no spec entity
binding. Three more disappeared because no label row is bound to `po_line`, `part_plant` or
`product_plant` any more: the corresponding `entity_id resolves` checks are gone.

### Oscillations — 41 checks have flipped state more than once

Four have flipped three times:

| check | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| `production_plan` | skip | **FAIL** | pass | **FAIL** |
| `supplier_acknowledgements` | pass | **FAIL** | pass | **FAIL** |
| `supplier_upstream` | pass | **FAIL** | pass | **FAIL** |

The dominant pattern across the other 38 is `FAIL → pass → pass → FAIL`: a defect fixed in run 2,
held through run 3, and reintroduced in run 4. It covers the whole as-of layer
(`po_lines`/`purchase_orders`/`inventory_snapshots`/`supplier_allocation` timestamps and lags),
the fill-rate point masses, lead-time skewness, all three rare-event rates, four Tier 1 tables
and four label checks.

Two checks currently **pass** but have oscillated and are worth flagging under the same rule:
`alternate_sources` and `po_line_revisions` both went pass → FAIL → pass → pass.

A defect class that can be fixed and reintroduced across three regenerations is not being fixed
at the mechanism; it is being rewritten by whichever patch script last touched the table. §3's
observation that twelve scripts overwrite each other's outputs is the direct explanation.

### Skipped in run 4 — 22 checks, none of them a pass

- **20 × `<table>: cleared Tier 1`** — the gating markers for the 20 tables that failed Tier 1.
  Their Tier 2/3 results are advisory, not evidence.
- **2 × `two timestamps differ`** for `inventory_position_weekly` and `plan_drift_features` —
  derived stores with `recorded_ts` but no event timestamp to compare against (skipped in every
  prior run for the same reason).

---

## 8. Joint structure

Run 3's six probes, re-run against the new layout by
`docs/validator_run4_joint_probes.py` (read-only, separate from the validator). The PO-line
population mirrors `fill_population`'s rules exactly: buyer-cancelled lines dropped (24,372),
lines promised inside the last 90 days of the span dropped, `fill = min(1, Σ qty_accepted /
qty_ordered)`, short at `fill < 0.9999`, late at `first receipt > original_promise_date`.
**837,380 PO lines** survive the filters, against run 3's 6,605 — every statistic below is on a
population two orders of magnitude larger.

### 8.1 Late and short co-occur — weakly positive (improved)

| | fill = 1.0 | fill < 1.0 | P(short) |
|---|---|---|---|
| **on time** | 186,939 | 131,887 | 41.37% |
| **late** | 282,808 | 235,746 | **45.46%** |

**φ = +0.0401** (run 3: −0.0027). Continuous form — days late against shortfall magnitude —
**r = +0.138, ρ = +0.089** on n = 837,380.

The sign is right and, unlike run 3, the association is real rather than noise. It is not
"clearly positive": a 4.1-point difference in shortfall probability between late and on-time
lines is a weak coupling for two failure modes that are supposed to share a cause. The
mechanism is visible in `build_v4_fast.py:99-101` — `fill` and `lateprob` are both monotone in
the same `supply` stress variable, so they share exactly one common driver and nothing else.

### 8.2 Constraint predicts shortfall — **undefined; the flag has no contrast**

The probe cannot be computed. Of 178,818 supplier-part-months that join to both a fill
measurement and `revealed_capacity_monthly`, **178,818 are flagged constrained and 0 are flagged
unconstrained.** There is no comparison group.

| | value |
|---|---|
| matched supplier-part-months | 178,818 |
| flagged constrained | **178,818 (100%)** |
| flagged unconstrained | **0** |
| mean fill, constrained | 0.9827 |
| mean fill, unconstrained | — |
| declared constrained share, whole table | 15.48% |

The reason is structural: `constrained_month_flag` is set from `util >= 1` where `util` is
clipped to `[0, 2]`, and the clip is saturating. Across all 1,288,056 rows
`capacity_utilisation_observed` takes **exactly two values**:

| | utilisation 0.0 | utilisation 2.0 |
|---|---|---|
| flag False | 1,088,714 | 0 |
| flag True | 0 | 199,342 |

The flag is perfectly collinear with "this supplier-month had any load at all". Every month with
PO activity is flagged constrained; every month without is not. So restricting to months where
fill is measurable selects the constrained set entirely, and the contrast the probe exists to
measure does not exist in the data.

`capacity_basis` is no longer the single constant it was in run 3 — it now takes `unobservable`
(1,080,968), `inferred` (108,017) and `reported` (99,071) — but all three values lie outside the
declared enum `{estimated, audited, declared, contracted}`, so the column fails Tier 1 on 100%
of rows, and `evidence_strength` remains a deterministic three-valued recode (0.05 / 0.25 / 0.80)
of the flag and the basis.

### 8.3 Shortages follow their cause — 98.7%, but the test barely discriminates

| | count | share |
|---|---|---|
| shortage events | 13,394 | |
| on a part-plant with **no** sourcing channel | **0** | 0.0% |
| with an identifiable upstream cause in the preceding 8 weeks | 13,224 | **98.73%** |
| **placebo — same computation, shortage dates shuffled** | | **96.89%** |

This is the largest improvement in the run and it is real in its mechanism: shortages are now
generated from acknowledgement shortfalls on actual PO lines
(`finish_remaining.py:10`), so every shortage has a parent transaction, and the run-3 finding
that 90.1% of shortages sat on part-plants with no channel at all is gone entirely.

**But the measured 98.7% overstates what has been achieved.** The placebo — the same lookback
against randomly reassigned shortage dates — returns 96.89%. The real-versus-chance gap is
**1.84 points**. Because 45% of all PO lines are short or late, almost any 8-week window on
almost any channel contains a miss, so the probe cannot distinguish "this shortage was caused by
that miss" from "misses are everywhere". §9's target of near-100% is met on the letter; the
attribution it was meant to demonstrate is not.

### 8.4 Staleness and volatility co-move — +0.46, correct sign, written in

Across 15,621 channels:

| | Pearson | Spearman |
|---|---|---|
| mean `weeks_since_last_activity` vs lead-time variance | **+0.461** | **+0.642** |

Run 3 measured −0.152. This is the correct sign and a strong association — and §3.4 item 5
identifies the code that produces it: `rewrite_stale.py:4` rewrites
`weeks_since_last_activity` as a Poisson draw whose rate is a function of the channel's
contracted lead time (`trait=np.clip((ld-30)/45,0,2); extra=rng.poisson(1+trait*3)`). The
correlation is an artefact of that rewrite, not of intermittent channels becoming less
predictable through any simulated process.

### 8.5 Labels track their own features — **all five near zero. Decisive failure.**

Every label in run 4 is bound to `entity_type = channel`, so all five are joined as-of the
snapshot date to the channel's own row in `channel_performance_weekly`.

| task | feature | Pearson | Spearman | n | reading |
|---|---|---|---|---|---|
| `fill_rate` | `fill_rate_last13` | **+0.014** | +0.036 | 1,180 | **detached** |
| `arrival_timing` | `otd_rate_last13` | **−0.013** | −0.013 | 1,007 | **detached** |
| `capacity_strain` | `load_ratio` | **+0.023** | +0.025 | 1,180 | **detached** |
| `demand_drift` | `qty_ordered` | **−0.043** | −0.064 | 1,180 | **detached** |
| `shortage_qty` | `days_since_last_short` | **undefined** | — | 1,180 | feature is **identically 0** on all 5,866,280 channel-weeks |

**Run 3's most important defect is unchanged, and this time it is provable from the source.**
The labels are not computed from the channel's history at all. They are computed from two scalars
that are sine functions of the *snapshot loop counter*:

```python
# finish_v4_metadata.py:22  (identical code at finalize_v4.py:49)
for i in range(1200):
 m=ch.iloc[(i*13)%15334]; wd=W[20+(i*7)%300]
 hist=np.clip(.75+.12*math.sin(i/17)+rng.normal(0,.05),.05,1)
 load=max(0,.35+.22*math.sin(i/31)+rng.normal(0,.1))
 otd=np.clip(hist-.12*load+rng.normal(0,.05),0,1)
 vals=[('fill_rate',np.clip(.35+.62*hist-.2*load+rng.normal(0,.07),.02,1)),
       ('arrival_timing',max(1,m.contracted_lead_time_days*(1+.5*(1-otd)+.35*load)+rng.normal(0,3.5))),
       ('capacity_strain',np.clip(.15+.85*load+rng.normal(0,.1),0,1.5)),
       ('demand_drift',np.clip(.1+.3*math.sin(2*math.pi*wd.isocalendar().week/52)+rng.normal(0,.08),-1,1)),
       ('shortage_qty',...)]
```

`hist` and `load` are `sin(i/17)` and `sin(i/31)` — functions of the loop index. The entity is
picked independently by a stride, `m = ch.iloc[(i*13) % 15334]`. Nothing reads
`channel_performance_weekly`. The labels are internally consistent with each other (they share
`hist` and `load`) and consistent with nothing in the feature store.

The `label_definition` column on all 6,000 rows reads
`'future outcome generated from causal state'`. That string is not accurate.

Three further defects in the same block:

- **All five tasks are bound to `channel`.** The spec binds `fill_rate` and `arrival_week` to
  `po_line`, `demand_drift` to `product_plant`, `shortage_qty` to `part_plant`. Three checks read
  `1,200/1,200 wrong`; `capacity_strain` passes only because `channel` happens to be its correct
  binding.
- **`arrival_week` does not exist.** The task is emitted as `arrival_timing`, so the spec's task
  is absent from the dataset and four of its checks vanish rather than fail.
- **Nothing is censored.** `cens = wd + 30d > 2025-12-29` is false for every snapshot, so
  `label_censored` is `False` on all 6,000 rows.

### 8.6 KS against Uniform(0,1) — all five pass

| task | n | support | mean | sd | distinct | `D` | `p` | verdict |
|---|---|---|---|---|---|---|---|---|
| `fill_rate` | 1,200 | [0.412, 1.000] | 0.7445 | 0.1011 | 1,191 | 0.2542 | 2.93e-68 | **pass** |
| `arrival_timing` | 1,200 | [8.37, 102.08] | 53.7211 | 21.1414 | 1,200 | 0.1160 | 1.47e-14 | **pass** |
| `capacity_strain` | 1,200 | [0.000, 1.078] | 0.4442 | 0.1880 | 1,196 | 0.2727 | 1.74e-78 | **pass** |
| `demand_drift` | 1,200 | [−0.415, 0.577] | 0.0885 | 0.2254 | 1,199 | 0.1145 | 3.54e-14 | **pass** |
| `shortage_qty` | 1,200 | [0, 200] | 34.1000 | 45.6557 | 151 | 0.5558 | 0 | **pass** |

No label is a uniform draw. As in run 3, this is a marginal test and passing it carries no
implication about §8.5: a label built from `sin(i/17)` is decisively non-uniform and decisively
uncorrelated with the world at the same time.

### Verdict on generated vs fitted — **fitted**

| probe | expected | run 3 | run 4 | reading |
|---|---|---|---|---|
| late ↔ short | clearly positive | −0.003 | **+0.040** (cont. +0.138) | weakly positive, one shared driver |
| constrained ↔ fill | materially lower | 2.0 pts | **undefined** — no unconstrained group | flag collinear with "had activity" |
| shortage ← cause | near 100% | 1.4% | **98.7%** (placebo 96.9%) | mechanism real, attribution not discriminating |
| staleness ↔ lead-time var | positive | −0.15 | **+0.46** | correct sign, written in by `rewrite_stale.py` |
| **labels ↔ own features** | clearly positive | +0.02 / −0.02 / +0.06 / −0.05 / +0.27 | **+0.014 / −0.013 / +0.023 / −0.043 / n.d.** | **all detached** |
| KS vs uniform | reject at p<0.01 | all pass | all pass | pass |

Two probes improved, one is uncomputable, one is manufactured, and the decisive one failed
completely — and for the first time we can name the lines that do it rather than infer it from
distributions. **The dataset is fitted, not generated.**

---

## 9. The three structural questions

### Q1. Are the labels real? — **No, and less real than run 3.**

Generated from `sin(i/17)` and `sin(i/31)` of a loop counter (§8.5). Correlation with the most
predictive feature in their own store: **+0.014, −0.013, +0.023, −0.043**, and undefined for
`shortage_qty` because `days_since_last_short` is identically zero. Additionally: 3 of 5 tasks
bound to the wrong entity type (1,200/1,200 wrong each), the `arrival_week` task absent
entirely, and 0.0% censoring on all five against a 1–60% band.

Run 3 had correctly-bound, correctly-censored labels that were detached from their features.
Run 4 has detached labels that are also mis-bound, mis-named and uncensored.

### Q2. Do the derived stores conserve exactly? — **No. +12.18% ordered, +3.24% received.**

| store | quantity | source | emitted | error | run 3 error |
|---|---|---|---|---|---|
| `channel_performance_weekly` | ordered | 113,893,014 | 127,761,134 | **+12.176%** | 0 (exact) |
| `channel_performance_weekly` | received | 118,923,111 | 122,774,945 | **+3.239%** | 0 (exact) |
| `supplier_performance_weekly` | ordered | 113,893,014 | 129,998,532 | **+14.141%** | 0 (exact) |
| `supplier_performance_weekly` | received | 118,923,111 | 124,926,841 | **+5.048%** | 0 (exact) |

`derived store buckets on visible week` reads **`neither`**, where run 3 read the correct
`max(event_week, recorded_week)`.

The cause is in §2a's table: the weekly stores span 2019-01-07 → 2025-12-29 while `po_lines`
runs only to 2020-03-31 and `grn_lines` to 2020-05-16. The store carries seven years of
channel-weeks generated by `build_v4_fast.py`'s own weekly loop, and the PO tables are a
separate 900,000-row sample drawn over roughly fifteen months. They are two parallel
fabrications again, exactly as in run 2 — the run-3 property that the store *was* an aggregation
has been lost.

Note also that the supplier store no longer matches the channel store (129,998,532 vs
127,761,134 ordered), so the supplier rollup is not even faithful to the channel store it is
supposedly aggregated from.

### Q3. Is the weekly store a contiguous panel? — **Contiguity yes, integrity no.**

| measure | run 3 | run 4 | band |
|---|---|---|---|
| zero-order channel-week share | 80.2% | **82.1%** | 60–99% |
| contiguity (unbroken 7-day run) | 100.0% | **100.0%** | ≥ 90% |
| **duplicate PKs** | **0** | **165,014** | 0 |
| distinct channel ids in the panel | 120 | **15,621** (vs 16,072 declared) | — |
| rows / declared channels | 43,800 / 120 = 365 | 5,866,280 / 16,072 = 365 | — |
| channel-weeks per channel | 365 | median 365, **max 730** | — |

The panel checks pass and the scale is right. But 451 channel ids carry two complete 365-week
series each, producing 165,014 duplicate `(channel_id, week_start)` keys and a Tier 1 failure on
`channel_performance_weekly`. `inventory_position_weekly` (332), `inventory_snapshots` (332) and
`plan_drift_features` (27,500) have the same defect.

A panel with duplicate keys is not a panel a rolling window can be computed over: the 451
affected channels would contribute each week twice, with different values.

---

## 10. Full tier tables

`run 3` / `run 2` / `run 1` columns show the same check's value in those runs. `_(new)_` marks a
check with no counterpart in that run. A `SKIP` is shown as a skip and is never a pass.

### Tier 1


**TABLES**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| -- | extra tables (informational) | `0` | `0` | `0` | `0` | n/a |
| warn | duplicate table files | `1` | _(new)_ | _(new)_ | _(new)_ | 0 |
| ok | all spec tables present | `49/49` | `49/49` | `49/49` | `47/49` | 49/49 |
| **FAIL** | structurally clean tables | `29/49` | `48/49` | `40/49` | `37/49` | 49/49 |
| warn | declared-width breaches (warning) | `5 table(s)` | `1 table(s)` | `1 table(s)` | `3 table(s)` | 0 |

**PER-TABLE**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| ok | alternate_sources | `6,000 rows` | `120 rows` | `120 rows` | `144 rows` | clean |
| **FAIL** | asn | `1,077,077 rows` | `17,218 rows` | `17,218 rows` | `7,000 rows` | clean |
| ok | bom | `40,000 rows` | `1,319 rows` | `1,319 rows` | `700 rows` | clean |
| ok | business_units | `5 rows` | `5 rows` | `5 rows` | `5 rows` | clean |
| **FAIL** | calendar | `25,536 rows` | `37,555 rows` | `37,555 rows` | `37,548 rows` | clean |
| **FAIL** | channel_performance_weekly | `5,866,280 rows` | `43,800 rows` | `43,800 rows` | `55,000 rows` | clean |
| ok | customers | `60 rows` | `20 rows` | `20 rows` | `20 rows` | clean |
| **FAIL** | dataset_coverage | `49 rows` | `48 rows` | `48 rows` | `46 rows` | clean |
| **FAIL** | expedite_events | `13,394 rows` | `2,005 rows` | `2,005 rows` | `1,400 rows` | clean |
| **FAIL** | goods_receipts | `1,077,077 rows` | `17,218 rows` | `17,218 rows` | `23,000 rows` | clean |
| **FAIL** | grn_lines | `1,077,077 rows` | `17,218 rows` | `17,218 rows` | `23,000 rows` | clean |
| **FAIL** | inventory_position_weekly | `2,190,000 rows` | `10,000 rows` | `10,000 rows` | `50,000 rows` | clean |
| **FAIL** | inventory_snapshots | `2,190,332 rows` | `15,340 rows` | `15,340 rows` | `60,000 rows` | clean |
| ok | inventory_transactions | `1,000,000 rows` | `1,534 rows` | `1,534 rows` | `45,000 rows` | clean |
| **FAIL** | line_stop_events | `142 rows` | `330 rows` | `330 rows` | `220 rows` | clean |
| ok | logistics_lanes | `2,000 rows` | `190 rows` | `190 rows` | `250 rows` | clean |
| ok | model_outputs | `1,000 rows` | `624 rows` | `624 rows` | `900 rows` | clean |
| ok | part_costs | `12,000 rows` | `120 rows` | `120 rows` | `700 rows` | clean |
| ok | part_demand_weekly | `2,190,000 rows` | `7,800 rows` | `7,800 rows` | `50,000 rows` | clean |
| ok | part_plant | `6,000 rows` | `1,680 rows` | `1,680 rows` | `580 rows` | clean |
| ok | parts | `7,000 rows` | `500 rows` | `500 rows` | `250 rows` | clean |
| **FAIL** | plan_drift_features | `80,000 rows` | `43,052 rows` | `43,052 rows` | `20,000 rows` | clean |
| ok | plants | `7 rows` | `7 rows` | `7 rows` | `7 rows` | clean |
| ok | po_line_revisions | `97,114 rows` | `5,121 rows` | `5,121 rows` | `6,000 rows` | clean |
| ok | po_line_schedules | `1,215,186 rows` | `14,755 rows` | `14,755 rows` | `40,000 rows` | clean |
| ok | po_lines | `900,000 rows` | `9,214 rows` | `9,214 rows` | `20,000 rows` | clean |
| ok | product_economics | `2,500 rows` | `120 rows` | `120 rows` | `1,800 rows` | clean |
| **FAIL** | production_actual | `30,000 rows` | `48,012 rows` | `48,012 rows` | _skip: no file_ | clean |
| **FAIL** | production_plan | `30,000 rows` | `64,837 rows` | `64,837 rows` | _skip: no file_ | clean |
| ok | products | `2,500 rows` | `120 rows` | `120 rows` | `120 rows` | clean |
| ok | purchase_orders | `900,000 rows` | `9,214 rows` | `9,214 rows` | `6,667 rows` | clean |
| ok | quality_inspections | `1,077,077 rows` | `17,218 rows` | `17,218 rows` | `23,000 rows` | clean |
| ok | revealed_capacity_monthly | `1,288,056 rows` | `10,651 rows` | `7,560 rows` | `4,000 rows` | clean |
| **FAIL** | shortage_events | `13,394 rows` | `3,580 rows` | `3,580 rows` | `1,800 rows` | clean |
| warn | snapshots | `1,200 rows` | `100 rows` | `100 rows` | `401 rows` | clean |
| ok | sourcing_channels | `16,072 rows` | `120 rows` | `120 rows` | `320 rows` | clean |
| **FAIL** | supplier_acknowledgements | `900,000 rows` | `9,214 rows` | `9,214 rows` | `15,600 rows` | clean |
| ok | supplier_allocation | `15,334 rows` | `120 rows` | `120 rows` | `320 rows` | clean |
| **FAIL** | supplier_audits | `5,000 rows` | `500 rows` | `500 rows` | `500 rows` | clean |
| **FAIL** | supplier_capacity | `1,288,056 rows` | `10,639 rows` | `10,639 rows` | `15,930 rows` | clean |
| ok | supplier_contracts | `12,000 rows` | `120 rows` | `120 rows` | `500 rows` | clean |
| ok | supplier_financials | `5,250 rows` | `900 rows` | `900 rows` | `450 rows` | clean |
| warn | supplier_performance_weekly | `273,750 rows` | `24,820 rows` | `24,820 rows` | `17,899 rows` | clean |
| ok | supplier_quality_ppm | `84,000 rows` | `12,480 rows` | `12,480 rows` | `900 rows` | clean |
| ok | supplier_sites | `750 rows` | `190 rows` | `190 rows` | `75 rows` | clean |
| **FAIL** | supplier_upstream | `250 rows` | `300 rows` | `300 rows` | `180 rows` | clean |
| **FAIL** | suppliers | `750 rows` | `120 rows` | `120 rows` | `50 rows` | clean |
| ok | tooling | `5,000 rows` | `120 rows` | `120 rows` | `150 rows` | clean |
| **FAIL** | training_labels | `6,000 rows` | `6,000 rows` | `6,000 rows` | `12,000 rows` | clean |

### Tier 2


**GATING**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| `SKIP` | asn: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | calendar: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _skip: no_ | n/a |
| `SKIP` | channel_performance_weekly: cleared Tier 1 | _skip: no_ | _(new)_ | _skip: no_ | _skip: no_ | n/a |
| `SKIP` | dataset_coverage: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | expedite_events: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | goods_receipts: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | grn_lines: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | inventory_position_weekly: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _skip: no_ | n/a |
| `SKIP` | inventory_snapshots: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _skip: no_ | n/a |
| `SKIP` | line_stop_events: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | plan_drift_features: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | production_actual: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | production_plan: cleared Tier 1 | _skip: no_ | _(new)_ | _skip: no_ | _(new)_ | n/a |
| `SKIP` | shortage_events: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | supplier_acknowledgements: cleared Tier 1 | _skip: no_ | _(new)_ | _skip: no_ | _(new)_ | n/a |
| `SKIP` | supplier_audits: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | supplier_capacity: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _skip: no_ | n/a |
| `SKIP` | supplier_upstream: cleared Tier 1 | _skip: no_ | _(new)_ | _skip: no_ | _(new)_ | n/a |
| `SKIP` | suppliers: cleared Tier 1 | _skip: no_ | _(new)_ | _(new)_ | _(new)_ | n/a |
| `SKIP` | training_labels: cleared Tier 1 | _skip: no_ | _skip: no_ | _skip: no_ | _(new)_ | n/a |

**AS-OF**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| **FAIL** | asn: two timestamps differ | `0.0%` | `59.9%` | `59.9%` | `14.5%` | >= 2% |
| ok | asn: lag is not constant | `sd/\|mean\|=1.745` | `sd/\|mean\|=0.753` | `sd/\|mean\|=0.753` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | asn: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| ok | expedite_events: two timestamps differ | `39.0%` | `83.4%` | `83.4%` | `0.0%` | >= 2% |
| ok | expedite_events: lag is not constant | `sd/\|mean\|=1.302` | `sd/\|mean\|=0.708` | `sd/\|mean\|=0.708` | `sd/\|mean\|=0.000` | >= 0.1 |
| **FAIL** | expedite_events: recorded_ts >= event | `297` | `0` | `0` | `0` | 0 |
| ok | goods_receipts: two timestamps differ | `14.6%` | `65.0%` | `65.0%` | `14.1%` | >= 2% |
| ok | goods_receipts: lag is not constant | `sd/\|mean\|=1.208` | `sd/\|mean\|=0.694` | `sd/\|mean\|=0.694` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | goods_receipts: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| ok | grn_lines: two timestamps differ | `14.6%` | `65.1%` | `65.1%` | `14.1%` | >= 2% |
| ok | grn_lines: lag is not constant | `sd/\|mean\|=1.208` | `sd/\|mean\|=0.684` | `sd/\|mean\|=0.684` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | grn_lines: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| `SKIP` | inventory_position_weekly: two timestamps differ | _skip: no event column_ | _skip: no event column_ | _skip: no event column_ | _skip: no event column_ | n/a |
| **FAIL** | inventory_snapshots: two timestamps differ | `0.0%` | `39.8%` | `39.8%` | `0.0%` | >= 2% |
| **FAIL** | inventory_snapshots: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.689` | `sd/\|mean\|=0.689` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | inventory_snapshots: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| ok | inventory_transactions: two timestamps differ | `20.0%` | `40.2%` | `40.2%` | `24.8%` | >= 2% |
| **FAIL** | inventory_transactions: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.695` | `sd/\|mean\|=0.695` | `sd/\|mean\|=2.062` | >= 0.1 |
| ok | inventory_transactions: recorded_ts >= event | `0` | `0` | `0` | `12,201` | 0 |
| ok | line_stop_events: two timestamps differ | `39.4%` | `83.6%` | `83.6%` | `0.0%` | >= 2% |
| ok | line_stop_events: lag is not constant | `sd/\|mean\|=1.318` | `sd/\|mean\|=0.705` | `sd/\|mean\|=0.705` | `sd/\|mean\|=0.000` | >= 0.1 |
| **FAIL** | line_stop_events: recorded_ts >= event | `2` | `0` | `0` | `0` | 0 |
| `SKIP` | plan_drift_features: two timestamps differ | _skip: no event column_ | _skip: no event column_ | _skip: no event column_ | _skip: no event column_ | n/a |
| **FAIL** | po_line_revisions: two timestamps differ | `0.1%` | `60.1%` | `60.1%` | `44.0%` | >= 2% |
| ok | po_line_revisions: lag is not constant | `sd/\|mean\|=1.452` | `sd/\|mean\|=0.751` | `sd/\|mean\|=0.751` | `sd/\|mean\|=7.485` | >= 0.1 |
| ok | po_line_revisions: recorded_ts >= event | `0` | `0` | `0` | `2,562` | 0 |
| -- | po_line_schedules: two timestamps differ | `14.3%` | `48.5%` | `48.5%` | `0.0%` | exempt |
| ok | po_line_schedules: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| **FAIL** | po_lines: two timestamps differ | `0.0%` | `60.5%` | `60.5%` | `0.0%` | >= 2% |
| ok | po_lines: lag is not constant | `sd/\|mean\|=1.540` | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | po_lines: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| ok | production_actual: two timestamps differ | `27.5%` | `60.1%` | `60.1%` | _skip: no file_ | >= 2% |
| **FAIL** | production_actual: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.753` | `sd/\|mean\|=0.753` | _(new)_ | >= 0.1 |
| ok | production_actual: recorded_ts >= event | `0` | `0` | `0` | _(new)_ | 0 |
| ok | production_plan: two timestamps differ | `13.3%` | `60.3%` | `60.3%` | _skip: no file_ | >= 2% |
| **FAIL** | production_plan: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.750` | `sd/\|mean\|=0.750` | _(new)_ | >= 0.1 |
| ok | production_plan: recorded_ts >= event | `0` | `0` | `0` | _(new)_ | 0 |
| **FAIL** | purchase_orders: two timestamps differ | `0.0%` | `60.2%` | `60.2%` | `0.0%` | >= 2% |
| ok | purchase_orders: lag is not constant | `sd/\|mean\|=1.540` | `sd/\|mean\|=0.754` | `sd/\|mean\|=0.754` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | purchase_orders: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| ok | quality_inspections: two timestamps differ | `23.6%` | `59.6%` | `59.6%` | `14.6%` | >= 2% |
| ok | quality_inspections: lag is not constant | `sd/\|mean\|=1.205` | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.746` | `sd/\|mean\|=0.000` | >= 0.1 |
| ok | quality_inspections: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |
| ok | shortage_events: two timestamps differ | `38.7%` | `82.6%` | `82.6%` | `0.0%` | >= 2% |
| ok | shortage_events: lag is not constant | `sd/\|mean\|=1.319` | `sd/\|mean\|=0.716` | `sd/\|mean\|=0.716` | `sd/\|mean\|=0.000` | >= 0.1 |
| **FAIL** | shortage_events: recorded_ts >= event | `180` | `0` | `0` | `0` | 0 |
| **FAIL** | supplier_acknowledgements: two timestamps differ | `0.1%` | `60.4%` | `60.4%` | `32.7%` | >= 2% |
| ok | supplier_acknowledgements: lag is not constant | `sd/\|mean\|=1.450` | `sd/\|mean\|=0.745` | `sd/\|mean\|=0.745` | `sd/\|mean\|=2.057` | >= 0.1 |
| ok | supplier_acknowledgements: recorded_ts >= event | `0` | `0` | `0` | `4,227` | 0 |
| **FAIL** | supplier_allocation: two timestamps differ | `0.0%` | `60.8%` | `60.8%` | `100.0%` | >= 2% |
| **FAIL** | supplier_allocation: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.720` | `sd/\|mean\|=0.720` | `sd/\|mean\|=0.000` | >= 0.1 |
| -- | supplier_allocation: recorded_ts >= event | `0` | `0` | `0` | `0` | n/a |
| ok | supplier_audits: two timestamps differ | `2.4%` | `78.8%` | `78.8%` | `100.0%` | >= 2% |
| ok | supplier_audits: lag is not constant | `sd/\|mean\|=0.638` | `sd/\|mean\|=0.727` | `sd/\|mean\|=0.727` | `sd/\|mean\|=0.572` | >= 0.1 |
| **FAIL** | supplier_audits: recorded_ts >= event | `4,584` | `0` | `0` | `0` | 0 |
| ok | supplier_capacity: two timestamps differ | `27.4%` | `94.8%` | `94.8%` | `44.1%` | >= 2% |
| **FAIL** | supplier_capacity: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.527` | `sd/\|mean\|=0.527` | `sd/\|mean\|=0.000` | >= 0.1 |
| -- | supplier_capacity: recorded_ts >= event | `0` | `0` | `0` | `0` | n/a |
| ok | supplier_quality_ppm: two timestamps differ | `57.1%` | `100.0%` | `100.0%` | `100.0%` | >= 2% |
| **FAIL** | supplier_quality_ppm: lag is not constant | `sd/\|mean\|=0.000` | `sd/\|mean\|=0.168` | `sd/\|mean\|=0.168` | `sd/\|mean\|=0.591` | >= 0.1 |
| ok | supplier_quality_ppm: recorded_ts >= event | `0` | `0` | `0` | `0` | 0 |

**CONSERVATION**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| **FAIL** | channel store conserves ordered units | `127,761,134 vs 113,893,014` | `1,944,910 vs 1,944,910` | `2,878,869 vs 1,944,910` | `138,615,579 vs 4,114,197` | exact |
| **FAIL** | channel store conserves received units | `122,774,945 vs 118,923,111` | `1,870,498 vs 1,870,498` | `2,740,051 vs 1,870,498` | `110,778,603 vs 2,992,098` | exact |
| **FAIL** | derived store buckets on visible week | `neither` | `max(event_week, recorded_week)` | `neither` | `neither` | max(event,recorded) |
| **FAIL** | supplier store conserves ordered units | `129,998,532 vs 113,893,014` | `1,944,910 vs 1,944,910` | `2,878,869 vs 1,944,910` | `21,468,423 vs 4,114,197` | exact |
| **FAIL** | supplier store conserves received units | `124,926,841 vs 118,923,111` | `1,870,498 vs 1,870,498` | `2,740,051 vs 1,870,498` | `17,303,481 vs 2,992,098` | exact |
| **FAIL** | part_demand_weekly horizon_days identity | `2,190,000/2,190,000` | `0/7,800` | `0/7,800` | `0/50,000` | 0 |
| ok | part_demand_weekly p90 >= p50 | `0/2,190,000` | `0/7,800` | `0/7,800` | `0/50,000` | 0 |
| **FAIL** | part_demand_weekly is strictly forward-looking | `2,190,000/2,190,000` | `0/7,800` | `0/7,800` | `0/50,000` | 0 |
| -- | part_demand_weekly unit conservation | `not asserted` | `not asserted` | `not asserted` | `not asserted` | n/a |

**LABELS**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| ok | label window opens after the snapshot | `0` | `0` | `0` | `0` | 0 |
| **FAIL** | arrival_timing: something is censored | `0.0%` | _(new)_ | _(new)_ | _(new)_ | 1%-60% |
| ok | arrival_timing: uncensored target varies | `sd/\|mean\|=0.3934 over 1,200 values` | _(new)_ | _(new)_ | _(new)_ | > 0.01 and >= 5 |
| ok | arrival_timing: target is not a uniform random draw | `D=0.1160 p=1.47e-14` | _(new)_ | _(new)_ | _(new)_ | p < 0.01 |
| **FAIL** | capacity_strain: something is censored | `0.0%` | `3.6%` | `3.6%` | `0.0%` | 1%-60% |
| ok | capacity_strain: uncensored target varies | `sd/\|mean\|=0.4231 over 1,196 values` | `sd/\|mean\|=0.8674 over 660 values` | `sd/\|mean\|=0.5783 over 1,119 values` | `sd/\|mean\|=0.5878 over 2,135 values` | > 0.01 and >= 5 |
| ok | capacity_strain: target is not a uniform random draw | `D=0.2727 p=1.74e-78` | `D=0.3146 p=1.29e-100` | `D=0.0160 p=0.926` | _(new)_ | p < 0.01 |
| ok | capacity_strain: entity_type matches the spec | `0/1,200 wrong` | `0/1,200 wrong` | `0/1,200 wrong` | `1,916/2,431 wrong` | 0 |
| **FAIL** | demand_drift: something is censored | `0.0%` | `4.6%` | `4.6%` | `0.0%` | 1%-60% |
| ok | demand_drift: uncensored target varies | `sd/\|mean\|=2.5456 over 1,199 values` | `sd/\|mean\|=0.2276 over 716 values` | `sd/\|mean\|=0.1954 over 1,060 values` | `sd/\|mean\|=0.5808 over 2,124 values` | > 0.01 and >= 5 |
| ok | demand_drift: target is not a uniform random draw | `D=0.1145 p=3.54e-14` | `D=0.4366 p=2.08e-191` | `D=0.0277 p=0.339` | _(new)_ | p < 0.01 |
| **FAIL** | demand_drift: entity_type matches the spec | `1,200/1,200 wrong` | `0/1,200 wrong` | `0/1,200 wrong` | `1,902/2,403 wrong` | 0 |
| **FAIL** | fill_rate: something is censored | `0.0%` | `4.7%` | `4.7%` | `0.0%` | 1%-60% |
| ok | fill_rate: uncensored target varies | `sd/\|mean\|=0.1358 over 1,191 values` | `sd/\|mean\|=0.1515 over 159 values` | `sd/\|mean\|=0.0687 over 172 values` | `sd/\|mean\|=0.5797 over 2,138 values` | > 0.01 and >= 5 |
| ok | fill_rate: target is not a uniform random draw | `D=0.2542 p=2.93e-68` | `D=0.8281 p=0` | `D=0.8453 p=0` | _(new)_ | p < 0.01 |
| **FAIL** | fill_rate: entity_type matches the spec | `1,200/1,200 wrong` | `0/1,200 wrong` | `0/1,200 wrong` | `1,892/2,413 wrong` | 0 |
| **FAIL** | shortage_qty: something is censored | `0.0%` | `3.8%` | `3.8%` | `0.0%` | 1%-60% |
| ok | shortage_qty: uncensored target varies | `sd/\|mean\|=1.3383 over 151 values` | `sd/\|mean\|=12.8735 over 12 values` | `sd/\|mean\|=0.5675 over 293 values` | `sd/\|mean\|=0.5706 over 2,076 values` | > 0.01 and >= 5 |
| ok | shortage_qty: target is not a uniform random draw | `D=0.5558 p=0` | `D=0.9905 p=0` | `D=0.0295 p=0.262` | _(new)_ | p < 0.01 |
| **FAIL** | shortage_qty: entity_type matches the spec | `1,200/1,200 wrong` | `0/1,200 wrong` | `0/1,200 wrong` | `1,882/2,334 wrong` | 0 |
| ok | entity_id resolves for entity_type=channel | `0/1,200 unresolved` | `0/120 unresolved` | `0/120 unresolved` | `2,252/2,292 unresolved` | 0 |

**SEPARATION**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| ok | channel_performance_weekly: no forward-looking column | `none` | `none` | `none` | `none` | none |
| ok | inventory_position_weekly: no forward-looking column | `none` | `none` | `none` | `none` | none |
| ok | part_demand_weekly: no forward-looking column | `none` | `none` | `none` | `none` | none |
| ok | plan_drift_features: no forward-looking column | `none` | `none` | `none` | `none` | none |
| ok | revealed_capacity_monthly: no forward-looking column | `none` | `none` | `none` | `none` | none |
| ok | supplier_performance_weekly: no forward-looking column | `none` | `none` | `none` | `none` | none |
| ok | model_outputs disjoint from feature tables | `0 table(s) overlap` | `0 table(s) overlap` | `0 table(s) overlap` | `0 table(s) overlap` | 0 |

**SNAPSHOTS**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| ok | snapshots: as_of_ts <= data_cutoff_ts | `0/1,200` | `0/100` | `0/100` | `0/401` | 0 |
| ok | snapshots: five version fields populated | `0 field(s) blank` | `0 field(s) blank` | `0 field(s) blank` | `0 field(s) blank` | 0 |
| ok | snapshots: code_commit is consistent | `1 distinct` | `1 distinct` | `1 distinct` | `1 distinct` | 1 |
| ok | snapshots: horizon_days matches the labels | `labels reach +30d` | `labels reach +90d` | `labels reach +90d` | `labels reach +90d` | snapshot 30d |

### Tier 3


**SHAPE**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| **FAIL** | PO lines with fill < 1.0 | `43.9%` | `15.9%` | `20.0%` | `72.7%` | 8%-15% |
| **FAIL** | PO lines arriving late | `61.9%` | `19.8%` | `19.8%` | `82.3%` | 15%-25% |
| **FAIL** | fill-rate mass at exactly 1.0 | `56.1%` | `84.1%` | `80.0%` | `27.3%` | >= 60% |
| **FAIL** | fill-rate mass at exactly 0 | `0.0%` | `2.0%` | `2.0%` | `31.8%` | 1%-3% |
| **FAIL** | right-censored tail is the right size | `0.23x` | `0.38x` | `0.38x` | `0.48x` | 0.3-1.2x |
| **FAIL** | supplier-months constrained | `49.0%` | `97.6%` | `97.6%` | `88.5%` | 20%-30% |
| **FAIL** | capacity unobservable | `51.0%` | `2.4%` | `2.4%` | `11.5%` | >= 70% |
| **FAIL** | lead-time distribution is right-skewed | `+0.30` | `+0.88` | `+0.88` | `+0.05` | >= 0.5 |
| **FAIL** | shortage events per year | `10213.3` | `235.6` | `235.6` | `120.5` | 180-250 |
| **FAIL** | line stops per year | `108.3` | `21.7` | `21.7` | `14.7` | 15-25 |
| **FAIL** | expedites per year | `10213.3` | `131.6` | `131.6` | `93.7` | 100-150 |
| ok | month-of-year seasonality in demand | `1.45x peak/trough` | `1.46x peak/trough` | `1.46x peak/trough` | `1.03x peak/trough` | > 1.10 |
| **FAIL** | PO lines with fill < 1.0 [2019-2025] | `43.9%` | `16.6%` | `20.1%` | `72.2%` | 8%-15% |
| **FAIL** | PO lines arriving late [2019-2025] | `61.9%` | `20.0%` | `20.0%` | `82.7%` | 15%-25% |
| **FAIL** | fill-rate mass at exactly 1.0 [2019-2025] | `56.1%` | `83.4%` | `79.9%` | `27.8%` | >= 60% |
| **FAIL** | fill-rate mass at exactly 0 [2019-2025] | `0.0%` | `1.9%` | `1.9%` | `32.1%` | 1%-3% |
| **FAIL** | right-censored tail is the right size [2019-2025] | `0.23x` | `1.13x` | `1.13x` | `0.91x` | 0.3-1.2x |
| **FAIL** | supplier-months constrained [2019-2025] | `49.0%` | `97.8%` | `97.8%` | `88.5%` | 20%-30% |
| **FAIL** | capacity unobservable [2019-2025] | `51.0%` | `2.2%` | `2.2%` | `11.5%` | >= 70% |
| **FAIL** | lead-time distribution is right-skewed [2019-2025] | `+0.30` | `+0.93` | `+0.93` | `+0.05` | >= 0.5 |
| -- | shortage events per year [2019-2025] | `1914.0` | `260.1` | `260.1` | `121.2` | 180-250 (not gated) |
| -- | line stops per year [2019-2025] | `20.3` | `24.3` | `24.3` | `16.6` | 15-25 (not gated) |
| -- | expedites per year [2019-2025] | `1914.0` | `143.6` | `143.6` | `95.9` | 100-150 (not gated) |
| ok | month-of-year seasonality in demand [2019-2025] | `1.45x peak/trough` | `1.43x peak/trough` | `1.43x peak/trough` | `1.03x peak/trough` | > 1.10 |

**DENSITY**

| | check | run 4 | run 3 | run 2 | run 1 | expected |
|---|---|---|---|---|---|---|
| ok | zero-order channel-weeks | `82.1%` | `86.0%` | `80.2%` | `0.0%` | 60%-99% |
| ok | weekly store is a panel | `100.0%` | `100.0%` | `100.0%` | `0.0%` | >= 90% |
| -- | staleness since last trade | `P50 28d / P90 77d / max 441d` | `P50 35d / P90 105d / max 406d` | `P50 28d / P90 77d / max 406d` | _skip: no idle weeks_ | reported |
| ok | PO lines per channel per year | `39.73` | `5.42` | `5.42` | `4.15` | >= 3.0 |
| -- | training rows per snapshot | `5` | `60` | `60` | `30` | reported |
| ok | zero-order channel-weeks [2019-2025] | `82.1%` | `86.0%` | `80.2%` | `0.0%` | 60%-99% |
| ok | weekly store is a panel [2019-2025] | `100.0%` | `100.0%` | `100.0%` | `0.0%` | >= 90% |
| -- | staleness since last trade [2019-2025] | `P50 28d / P90 77d / max 441d` | `P50 35d / P90 105d / max 406d` | `P50 28d / P90 77d / max 406d` | _skip: no idle weeks_ | reported |
| ok | PO lines per channel per year [2019-2025] | `7.45` | `7.63` | `7.63` | `4.15` | >= 3.0 |
| -- | training rows per snapshot [2019-2025] | `5` | `60` | `60` | `27` | reported |

---

## 11. Comparison against `csv_full_seed1`

Control figures are from run 4's own control run, which is byte-identical to run 3's on every
check line.

| Tier 3 figure | run 4 | run 3 | run 2 | run 1 | `csv_full_seed1` | band | run 4 verdict |
|---|---|---|---|---|---|---|---|
| PO lines with fill < 1.0 | **43.9%** | 15.9% | 20.0% | 72.7% | 9.5% | 8–15% | **FAIL** |
| PO lines arriving late | **61.9%** | 19.8% | 19.8% | 82.3% | 12.8% | 15–25% | **FAIL** |
| fill-rate mass at exactly 1.0 | **56.1%** | 84.1% | 80.0% | 27.3% | 90.5% | ≥ 60% | **FAIL** |
| fill-rate mass at exactly 0 | **0.0%** | 2.0% | 2.0% | 31.8% | 1.4% | 1–3% | **FAIL** |
| right-censored tail (× span-implied) | **0.23×** | 0.38× | 0.38× | 0.48× | 0.56× | 0.30–1.20× | **FAIL** |
| supplier-months constrained | **49.0%** | 97.6% | 97.6% | 88.5% | 24.1% | 20–30% | **FAIL** |
| capacity unobservable | **51.0%** | 2.4% | 2.4% | 11.5% | 75.9% | ≥ 70% | **FAIL** |
| lead-time skewness | **+0.30** | +0.88 | +0.88 | +0.05 | +2.36 | ≥ 0.50 | **FAIL** |
| lead time median / P90 / P99 (d) | **41 / 68 / 86** | 63 / 96 / 132 | 63 / 96 / 132 | 50 / 78 / 93 | 31 / 67 / 139 | — | — |
| shortage events / year | **10,213.3** | 235.6 | 235.6 | 120.5 | 214.7 | 180–250 | **FAIL** |
| line stops / year | **108.3** | 21.7 | 21.7 | 14.7 | 20.0 | 15–25 | **FAIL** |
| expedites / year | **10,213.3** | 131.6 | 131.6 | 93.7 | 124.8 | 100–150 | **FAIL** |
| demand peak/trough ratio | 1.45× | 1.46× | 1.46× | 1.03× | 1.30× | > 1.10 | ok |
| zero-order channel-weeks | 82.1% | 80.2% | 80.2% | 0.0% | 81.4% | 60–99% | ok |
| channels with a contiguous panel | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | ≥ 90% | ok |
| channel-weeks per channel | 376 | 365 | 365 | 172 | 387 | — | — |
| PO lines per channel per year | 39.73 (7.45 windowed) | 5.42 | 5.42 | 4.15 | 7.02 | ≥ 3.0 | ok |
| staleness P50 / P90 / max (d) | 28 / 77 / 441 | 35 / 105 / 406 | 28 / 77 / 406 | none | 42 / 1085 / 3563 | — | — |
| training rows per snapshot | **5** | 60 | 60 | 30 | 44,007 | — | — |

Run 4 is **further from `csv_full_seed1` than run 3 on ten of the twelve gated Tier 3 figures**,
and closer on none of them except the three that were already passing. The three figures that
now sit closest to the control — zero-order channel-weeks (82.1% vs 81.4%), panel contiguity and
channel-weeks per channel — are the three §3.4 identifies as tuned via the 0.19 activity
probability.

Two figures worth flagging:

- **`training rows per snapshot` = 5**, against 44,007 in the control and 60 in run 3. The
  generator emits 1,200 snapshots × 5 tasks × 1 row = 6,000 labels, i.e. exactly one entity per
  task per snapshot. Every correlation in §8.5 is measured on ~1,200 rows, which is ample to
  distinguish |r| ≈ 0.02 from a real effect, but it is a very thin training set.
- **Rare-event rates are ~47× the control** purely because 13,394 shortages sit in the ~15 months
  the event tables span rather than the seven years the panel claims.

---

## 12. Gate readiness

### Delivery risk / arrival timing — **not trainable**

Blocking: `arrival_week` **does not exist** in the dataset (emitted as `arrival_timing`);
`arrival_timing: something is censored` = 0.0% (band 1–60%); `PO lines arriving late` = 61.9%
(band 15–25%); `channel store conserves received units` +3.24%; `derived store buckets on
visible week` = `neither`; `channel_performance_weekly` fails Tier 1 with 165,014 duplicate PKs;
`po_lines: two timestamps differ` = 0.0%, so there is no as-of displacement to read.

Beyond the gates: the label correlates **−0.013** with `otd_rate_last13`.

### Fill rate — **not trainable**

Blocking: `PO lines with fill < 1.0` = 43.9% (band 8–15%, and 28 points worse than run 3);
`fill-rate mass at exactly 1.0` = 56.1% (band ≥ 60%); `mass at exactly 0` = 0.0% (band 1–3%);
`fill_rate: entity_type matches the spec` = 1,200/1,200 wrong; `fill_rate: something is
censored` = 0.0%; the same conservation and Tier 1 failures on the store it would read.

Beyond the gates: the label correlates **+0.014** with `fill_rate_last13` — the channel's own
trailing fill rate.

### Part shortage — **not trainable**

Blocking: `supplier-months constrained` = 49.0% (band 20–30%); `capacity unobservable` = 51.0%
(band ≥ 70%); `shortage events per year` = 10,213.3 (band 180–250); `shortage_qty: entity_type
matches the spec` = 1,200/1,200 wrong; `shortage_qty: something is censored` = 0.0%;
`training_labels` and `shortage_events` both fail Tier 1.

Beyond the gates: `days_since_last_short` — the obvious predictive feature — is **identically
zero across all 5,866,280 channel-weeks**, so the correlation is not merely low, it is
undefined.

### Summary

| task | run 1 | run 2 | run 3 | run 4 | blocking |
|---|---|---|---|---|---|
| Delivery risk / arrival timing | not trainable | not trainable | not trainable | **not trainable** | task absent; 0% censored; late 61.9%; store PKs; label r = −0.013 |
| Fill rate | not trainable | not trainable | not trainable | **not trainable** | fill<1 43.9%; masses out of band; wrong entity; label r = +0.014 |
| Part shortage | not trainable | not trainable | not trainable | **not trainable** | capacity pair; 10,213 shortages/yr; wrong entity; feature all-zero |

No task has been trainable in any of the four runs, and run 4 is the furthest from it. Run 3's
blocking set was three specific defects per head; run 4's is a superset of run 3's plus the
entire as-of and Tier 1 layer that runs 2 and 3 had fixed.

---

## 13. `V4_Corrections.md` acceptance checklist

### Sections 1–8

| § | requirement | status | evidence |
|---|---|---|---|
| **0** | labels are consequences; PO→ACK→ASN→GRN spine; conservation as a target | **partially met** | the spine exists and shortages descend from it (§8.3); labels are not consequences (§8.5); conservation regressed to +12.2% (§9 Q2) |
| **1** | recording lag right-skewed, state-dependent | **partially met** | `lag()` is log-normal with a stress multiplier (§3.1); but P50 = 1 h on `po_lines` so no fact crosses a week boundary — `two timestamps differ` = 0.0% |
| **1** | lag never negative | **not met** | 5,063 negative lags across `supplier_audits` (4,584), `expedite_events` (297), `shortage_events` (180), `line_stop_events` (2) |
| **1** | some records never entered | **not met** | no suppression step exists; `label_censored` False on 6,000/6,000 rows; all five censoring checks fail at 0.0% |
| **2** | disruptions enter at supplier capacity and transit | **met** | `supply` feeds `fill` and `lead_actual`, not material requirement (§3.2) |
| **2** | demand-side shocks a separate class | **not met** | demand is a sine + trend + noise; no shock of any kind (§3.2) |
| **3** | shortage → expedite | **met** | `finish_remaining.py:13` |
| **3** | shortage → PO line revisions | **not met** | revisions driven by `late` only (§3.3) |
| **3** | shortage → alternate sourcing / allocation | **not met** | `alternate_sources` static from 2018-01-01; `supplier_allocation` constant 0.5 on one date |
| **3** | **alternate sourcing → receiving supplier's utilisation** | **not met** | absent from all 632 lines — the arc §3 calls "the only path a GNN reads" |
| **3** | line stop → production plan | **not met** | plan generated independently |
| **3** | sustained performance → allocation | **not met** | allocation never changes |
| **4** | mechanism parameters set, outcomes measured | **not met** | eight outcome literals found in source (§3.4), including capacity observability set by `ratio.quantile(.75)` — the exact anti-pattern §4 names |
| **5** | 16,072 channels / 15,334 traded / 738 cold-start | **partially met** | configured correctly; realised 15,621 distinct ids / 15,069 traded / 552 cold-start, with 165,014 duplicate PKs (§3.5) |
| **5** | channel-weeks per channel ≈ 386.8; PO lines/channel/yr ≈ 7.02 | **partially met** | 365 median (max 730); 7.45 windowed but 39.73 over the actual PO span |
| **6** | two seeds + per-metric variance band | **not met** | one seed (1001); no variance band published |
| **6** | `code_commit` consistent and matching the source hash | **partially met** | 1 distinct value, verified to match SHA-256 of six generator files — but six others are excluded, including `fix_capacity_only.py` (§3.6) |
| **7** | report what came out, including failures | **met** | `local_validation.json` states 15.3% constrained is "below reference band and must be rechecked externally" and that no external pass is claimed |
| **8** | capacity binds at PO→GRN; recording layer orthogonal | **partially met** | capacity binds at the right node; the recording layer exists but is constant on 7 tables and negative on 4 |

### Section 9 acceptance criteria

| # | criterion | status | evidence |
|---|---|---|---|
| 1 | generator source shipped | **met** | `db/generator/`, 12 files, 632 lines |
| 2 | parameter file separating mechanism from measured outcomes | **partially met** | `v4_parameters.json` exists with a `mechanism_parameters` block, but one entry is `capacity_calibration_factor: 0.3` — a measured outcome listed as a mechanism |
| 3 | two seeds and the per-metric variance band | **not met** | one seed, no band |
| 4a | late ↔ short correlation, clearly positive | **partially met** | φ = **+0.040**, continuous r = +0.138 — positive but weak |
| 4b | fill in constrained vs unconstrained months, materially lower | **not met** | **undefined** — 0 unconstrained supplier-months in the matched set (§8.2) |
| 4c | share of shortages with an upstream cause, near 100% | **met on the number, not the intent** | 98.73%; placebo 96.89%, so the gap over chance is 1.84 points |
| 4d | staleness ↔ lead-time variance, positive | **met on the number, not the intent** | +0.461 / +0.642, produced by `rewrite_stale.py` writing staleness from lead time |
| 4e | **each label ↔ its most predictive feature — decisive** | **not met** | +0.014, −0.013, +0.023, −0.043, undefined (§8.5) |
| 4f | KS of each label vs Uniform(0,1), reject at p < 0.01 | **met** | all five reject, p ≤ 1.5e-14 |
| 5 | a statement of what failed and the mechanism for each | **partially met** | `local_validation.json` flags the capacity shortfall honestly; no statement covers the 73 failures, the label detachment, or the duplicate keys |

`V4_Corrections.md` says of criterion 4e: *"Criterion 4's fifth bullet is the one to check first.
Every other number can be right while that one is zero, and if it is zero the dataset is version
1 again in better clothes."*

**It is zero.**

---

## Appendix — files

| file | contents |
|---|---|
| `docs/external_dataset_validation.md` | notice recording the loss of run 1's report during run 3 (**not modified**) |
| `docs/validator_run_external.txt` | run 1 stdout, external — source for all run-1 figures (**not modified**) |
| `docs/validation2.md`, `docs/validation3.md` | run 2 and run 3 reports (**not modified**) |
| `docs/validator_run2_*.txt`, `docs/validator_run3_*.txt` | run 2 and run 3 transcripts (**not modified**) |
| `docs/validation4.md` | this report |
| `docs/validator_run4_external.txt` | run 4 stdout, external (678 lines) |
| `docs/validator_run4_control_csv_full_seed1.txt` | run 4 stdout, control |
| `docs/validator_run4_joint_probes.py` | the §8 probe script (read-only, separate from the validator) |
| `docs/validator_run4_schema_reconciliation.txt` | the §2b three-way reconciliation output |

Validator SHA-256 before and after run 4:
`c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` — unchanged, and identical to
the SHA recorded in `docs/validation2.md` and `docs/validation3.md`. **`db/validator.py` was not
edited.** Every run used `--no-report` so that no prior report could be overwritten.

No CSV, `dataset_structure.md`, `schema.sql`, `V4_Corrections.md` or file under `db/generator/`
was modified.
