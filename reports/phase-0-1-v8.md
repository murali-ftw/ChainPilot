# Phase 0 and Phase 1 on v8

**Audience:** whoever runs the next phase on v8, and whoever owns the serving path.
**Scope:** Stage 1 Phase 0 (loaders, panel width, supplier-store audit, graph, leak assertions),
Stage 2 Phase 1 (first training, 3-seed bands), Stage 3 what v8 newly unblocks.
**Measured on:** `db/gen_v8/seed_1001`, fixed split (train ≤ 2023, validation 2024, test 2025).
**Companion:** `reports/v8-clearance.md`, whose gate rule this phase corrects.

---

## 0. Verdict, up front

**Phase 0 is clean. Ten leak assertions run, ten pass, none disabled or weakened.**

The single most consequential result of this phase is **not** about v8. Running arrival on v8
fired the Phase 11 as-of assertion on the first cell. I investigated before treating it as a v8
finding, and it is not one: **arrival training has been unrunnable on v6, v7 and v8 alike since
commit `ef3048d`** — 244,000 of 244,000 rows on every world. Phase 11 reported its feature path
as "default off, every existing configuration bit-identically unchanged"; the code does not do
that. Fixed by gating the *attachment* of line-level columns, never the assertion. **Deviation 59.**

| | |
|---|---|
| Panel width | **25** = 15 value + 10 indicator (v6/v7: 24). Derived from the world, asserted, falsified three ways |
| Graph | **unchanged from v6/v7**: 4 node types, R = 6, 17,119 nodes, 48,216 edges, median channel degree 3, one component |
| Supplier store | 7 identically-zero columns — and **no panel feature reads it**, so nothing dead is carried |
| Leak assertions | **10 run, 10 pass, 0 fired unexpectedly** |
| B1 re-check | **100.000000%** on seed 1001 *and* seed 1005, through the loader path |
| Phase 1 | see §3 — running at the time of writing |
| Newly unblocked | Phase 9.1 in full; 10.1's stock-balance and safety-stock constraints; 10.2's `SimulationScorer` in quantity terms |

---

## 1. The gate correction, recorded

`reports/v8-clearance.md` made **B9 blocking** and stopped Phase 0. That rule was an error in the
brief, not in the measurement, and stopping against it was correct. **B9 describes the same
condition v6 and v7 carried**, and Phases 1–11 trained on those worlds throughout.

**The corrected rule, which governs this phase:**

- **B1 and B12 CLEARED → Phase 0/1 may proceed.** They are.
- **B9 NOT CLEARED →**
  - (a) the as-of assertion in the training path **stays armed and is never weakened**;
  - (b) **no line-level feature** — line age, `promise_week`, original or current promise date,
    `po_created` — is added to any head;
  - (c) the promise-date comparison is **reported as RETIRED**, not as a loss.

B9 blocks only guide 11.2 (deviation 45), any arrival ranking claim (deviation 32), and any
deployable reading of the promise-date baseline (deviation 48) — all three already closed
questions. Recorded as **deviation 58** in the clearance report.

**Deviation 55 is withdrawn.** Verified against `db/gen_v6/seed_1001/training_labels.pre_fix`:

| file | shortage rows | zeros | positives | positive rate |
|---|---|---|---|---|
| v6 `training_labels.pre_fix` (**pre-repair**) | 45,942 | **0** | 45,942 | **100.00%** |
| v6 `training_labels.csv` (post-repair) | 124,500 | 107,705 | 16,795 | **13.49%** |
| **v8** | 124,500 | 95,090 | 29,410 | **23.62%** |

**The zero-negative defect was real and was repaired** in the Phase 1 label fix. v8 is **not**
credited with fixing it. v8's genuine improvement is the positive rate, **13.49% → 23.62%**.

---

## 2. Stage 1 — Phase 0

### 2.1 What the schema diff forced, and what it did not

The diff against v6/v7 is **0 tables / 0 columns / 0 retypes**, so almost nothing needed changing.

**Did not need changing:** `ml/data/loader.py` (readers, as-of sweep, node tables, edge and
`HeteroData` construction, A1 and A2 assertions), `ml/data/sequences.py`, `ml/train/folds.py`,
`ml/models/*`, `ml/eval/phase5_metrics.py`, and `ml/configs/shipped.json` — **untouched**, with
the world passed on the command line.

**Did need changing**, three places, all reported:

| file | change | why |
|---|---|---|
| `ml/config.py` | `v8` added to `WORLDS` / `WORLDS_SEED2`, plus `WORLDS_ALL_SEEDS` (v8 ships five seeds) and `EXPECTED_PANEL_D` | a third world has to be nameable |
| `ml/data/cache.py` | panel columns **derived by measurement** instead of hardcoded, + width assertion | §2.2 |
| `ml/train/temporal_share.py`, `phase5_heads.py`, `loop.py` | `row_features` threaded through `labels_for` | §2.5 — a pre-existing defect, not a v8 one |

### 2.2 Panel width — measured per world, asserted, and falsified

v6/v7 hold four `channel_performance_weekly` columns at constant zero (deviation 4). **v8 holds
three**: `revision_count` is populated (3.33% non-zero, max 5, 6 distinct).

```
v8 : 15 value + 10 indicator = d_in 25      (32 with the spec's calendar block)
v6 : 14 value + 10 indicator = d_in 24      (31 with it)
```

The calendar block is not in the built panel on any world — deviation 2 records `d_in = 24` as
built against the spec's 38 — so **25 is the number that reaches the model on v8**.

`cache.build_panel` now measures the constant-zero set on the CSVs it is reading and asserts the
result against `config.EXPECTED_PANEL_D`. **Dropped on v8:** `days_since_last_short`,
`weeks_since_last_activity`, `weeks_since_last_receipt`. Panel `[16072, 535, 15]`, complete, 535
weeks from 2016-01-04, 8,598,520 rows.

**The assertion fires (standing rule 1), three ways:**

| mutation | result |
|---|---|
| expect 14 (v6/v7's width, borrowed) | **FIRES** — "measured 15 value channels, config says 14" |
| expect 25 (`d_in` mistaken for `d`) | **FIRES** |
| undeclared world `v99` | **FIRES** — "not declared in EXPECTED_PANEL_D; declare it rather than defaulting" |

**Regression: v6 rebuilds byte-identically.** `panel.npy`, `miss.npy` and `active.npy` all
`np.array_equal` against the shipped cache, same 14 columns in the same order. **No Phase 2–11
number moves.**

Masters are identical across all three worlds: 16,072 channels, 420 suppliers, 620 parts,
7 plants, 83 snapshots, 535 weeks.

### 2.3 Supplier-store audit (deviation 54)

`supplier_performance_weekly`, 224,700 rows, 19 numeric columns. **Seven are identically zero:**

| column | non-zero |
|---|---|
| `lead_time_actual_days` | **0.000%** |
| `lead_time_ratio` | **0.000%** |
| `otd_rate_last13` | **0.000%** |
| `days_since_last_short` | **0.000%** |
| `reporting_lag_days` | **0.000%** |
| `weeks_since_last_activity` | **0.000%** |
| `weeks_since_last_receipt` | **0.000%** |

**Which store does each panel feature read? All fifteen read `channel_performance_weekly`.
Not one reads `supplier_performance_weekly`.** So nothing identically zero is carried as a dead
input, and **no feature needed dropping** — the dead columns were never wired in.

**The Level-3 arrival probe reads the CHANNEL store.** The same four column names carry real
values there and zeros in the supplier store:

| column | channel store | supplier store |
|---|---|---|
| **`otd_rate_last13`** — §16's decisive arrival probe | **75.145%** non-zero | **0.000%** |
| `lead_time_actual_days` | 95.305% | 0.000% |
| `lead_time_ratio` | 95.305% | 0.000% |
| `reporting_lag_days` | 95.305% | 0.000% |

The probe passes on v8 because it is evaluated against the channel store. **Any future work that
reaches for supplier-weekly performance reads zeros** — that is the live risk deviation 54 names,
and it is not realised in the current pipeline.

*(The supplier store also carries two columns the channel store does not exercise the same way:
`revision_count` at 47.449% non-zero and `active_channel_count` at 100%.)*

### 2.4 The graph

**Unchanged from v6/v7 in every respect** — the masters are shared across worlds, so this is
expected rather than surprising:

| | v8 | v6/v7 |
|---|---|---|
| node types | **4** (channel, supplier, part, plant) | 4 |
| nodes | **17,119** (16,072 / 420 / 620 / 7) | 17,119 |
| core edges | **48,216** over 3 relations + 3 reverse | 48,216 |
| **R** | **6** | 6 |
| median channel degree | **3** | 3 |
| components | **1**, 17,119 nodes (100%) | 1 |
| isolated nodes | **0** | 0 |

Extended relations: `supplier→depends_on→supplier` 260, `part→alt_sourced_from→supplier` 16,072,
`part→stocked_at→plant` 4,340.

**Degree distribution — what a node actually chooses between:**

| entity | channels per node, median | range |
|---|---|---|
| supplier | **38** | 21–55 |
| part | **26** | 12–45 |
| plant | **2,285** | 2,227–2,362 |

**The "attention pays only where a node has neighbours to choose between" principle rests on this
distribution, and the distribution is identical to v6/v7's.** So the principle transfers — but
transferring the principle is not the same as transferring a *result*. Whether attention pays on
v8 is a v8 measurement, and it is not made here.

### 2.5 Every leak assertion — and the one that fired

**Ten assertions run. Ten pass. None disabled, none weakened, none fired unexpectedly.**

| assertion | source | result |
|---|---|---|
| A1 — `recorded_ts >= event_ts` on 11 transactional tables | `loader.py` | **PASS** — 0 negative on every table |
| A2 — store buckets on `max(event_week, recorded_week)`, exact integer conservation | `loader.py` | **PASS** — ordered 304,313,857, received 240,258,195, both exact |
| A3 — `weeks_since_last_activity` never reads the future (brute force + perturbation) | `staleness.py` | **PASS** — 16,384 brute-force checks, 12 perturbation cuts |
| fixed-split no-leak, `arrival_week` | `folds.py` | **PASS** — 176,000 / 32,000 / 36,000 |
| fixed-split no-leak, `capacity_strain` | `folds.py` | **PASS** — 110,000 / 20,000 / 22,500 |
| fixed-split no-leak, `fill_rate` | `folds.py` | **PASS** — 176,000 / 32,000 / 36,000 |
| fixed-split no-leak, `shortage_qty` | `folds.py` | **PASS** — 66,000 / 12,000 / 13,500 |
| fixed-split no-leak, `demand_drift` | `folds.py` | **PASS** — 7,920 / 1,440 / 1,260 |
| rolling origins 1–8 no-leak | `folds.py` | **PASS** — all 8 origins, max(train) < min(evaluate) |
| Phase 11 `po_line` as-of | `phase5_heads.py`, `loop.py` | **ARMED, not reached** — see below |

**Label-window overlap** (measured, not asserted, as `folds.py` requires): 8,000 training rows
whose label window ends in or after validation; **0** ending in or after test; 8,000 validation
rows ending in or after test; max window 90 days. Same property the fixed split has always had.

#### The Phase 11 assertion fired, and it is not a v8 finding — deviation 59

The first arrival cell died in 26 seconds:

```
File "ml/train/phase5_heads.py", line 196, in batches
    assert (rows.line_recorded_ts <= rows.snapshot_date).all(), \
AssertionError: a po_line is not yet recorded at its own snapshot -- as-of violation
```

I added no line-level feature. Investigating before reporting it as a v8 property:

| world | `line_age_weeks` attached by default | as-of violations |
|---|---|---|
| v6 | **yes** | **244,000 / 244,000 (100.00%)** |
| v7 | **yes** | **244,000 / 244,000 (100.00%)** |
| v8 | **yes** | **244,000 / 244,000 (100.00%)** |

**Identical on all three worlds.** `git log -S` places the cause precisely: commit `ef3048d`
(Phase 11 Stage 2) introduced **both** the assertion in `batches()` **and** the unconditional
attachment of `line_age_weeks` / `line_recorded_ts` in `temporal_share.labels_for`. The assertion
guards on `"line_age_weeks" in rows`, so **attaching the column is indistinguishable from using
it**, and the assertion fires for every arrival cell on every world.

Phase 11's report states the path is "default off" and "every existing configuration is
bit-identically unchanged" (deviation 45). **Neither holds.** No baseline arrival cell was re-run
after that commit, so arrival training has been silently unrunnable ever since.

**The fix stops ATTACHING what must not be USED** — which is exactly clause (b) of the corrected
rule. `labels_for` takes `row_features=False` by default and attaches the line-level columns only
when the pilot path is explicitly requested. **The assertion is not touched, not weakened, not
bypassed.** Verified both directions:

| | result |
|---|---|
| `row_features=False` (the shipped path), v6 / v7 / v8 | labels frame carries **no line-level column at all** |
| `row_features=True`, v6 and v8 | assertion **still FIRES**, 244,000 / 244,000 rows |
| `loop.train(row_features=True)` end-to-end | **still raises** the as-of violation |

`promise_week` stays attached: it is the lateness ROC-AUC's **scoring reference**, never a head
input. It is privileged information and is labelled as such wherever it appears.

A second, latent hole was closed with it: `loop.train()` called `P5.labels()` without threading
`row_features`, so a pilot cell would have trained **silently without the features and without
ever reaching the assertion**. Threaded, with an explicit guard.

### 2.6 B1 through the loader path, and seed 1005

| | rows | exact | part-plants fully exact | max abs diff |
|---|---|---|---|---|
| v8 seed 1001 | 2,253,420 | **100.000000%** | 4,212 / 4,212 | **0** |
| v8 seed 1005 | 2,255,025 | **100.000000%** | 4,215 / 4,215 | **0** |

**Clearance-report open item 7 is closed, and the shortfall was mine, not the data's.** The
earlier 99.9944% came from an exact-week merge: the opening posting sits at `event_ts`
2015-12-28, in the week *before* the store's first week (2016-01-04), so the merge drops it and a
forward-fill inside the merged frame cannot recover it. Three part-plants in seed 1005 have no
transaction in the store's first week and so read a cumulative of 0. Re-run as an **as-of join** —
the last cumulative level at or before each store week — both seeds reconcile exactly.
`ml/train/phase0_v8.py` carries the corrected join; the clearance report is corrected.

---

## 3. Stage 2 — Phase 1, first training

*Running at the time of writing. This section is completed when the queue lands; see §7.*

**What is being run:** `ml/configs/shipped.json` **unmodified**, world passed on the command
line, fixed split, 3 seeds (7 / 17 / 27), shipped depth **and** h0 — 21 cells. No learning rate
is retuned and nothing is tuned.

**fill's configured model is `b5flat22` and the serving path has no LightGBM loader
(deviation 46).** That is unresolved, and this phase does not repair it. **Fill's neural h0 is
trained instead** — `arch none / depth 0 / lr 1.25e-4`, which is `shipped.json`'s own
`_superseded` entry — and the config is left exactly as it stands.

### 3.1 How arrival is reported (corrected rule, clause c)

**The promise-date C-index comparison is RETIRED on v8.** It is not printed as a model result and
the sentence "the head loses to the promise date" does not appear in this report.

The reason is structural, from `generator_v8.py` l.1469–1486:

```python
fut = (pt > t0) & (pt <= t1)      # pt = the PO line's PLACEMENT week
li  = np.where(fut)[0]            # the label population: lines placed AFTER the snapshot
```

**The arrival label population is lines placed strictly after `t0`, by construction.** The
promise date belongs to a line that will not be raised for another ~7 weeks (B9: median −47 days,
nearest row −7). So **arrival ranking is untestable on this data** — not because the head is
weak, and not because the promise date is strong, but because the two quantities are not
available at the same instant. No feature change to the head can make it a fair comparison.

What is reported instead: **the head's own C-index**, and **lateness ROC-AUC**. Where the
promise-date figure appears at all it is labelled *privileged baseline, not a fair comparison* in
the same line, and the scorer carries it under a `PRIVILEGED__` prefix so it cannot be quoted by
accident.

**One caveat I will not paper over:** the lateness metric itself uses the promise date as its
reference — the label is `arrival > promise` and the score is `prediction − promise`. So lateness
ROC-AUC is measured *against a privileged reference*, exactly as Phase 8 §5 framed it ("the head
adds lateness information beyond the promise date"). It is the head's measured value on this
data; it is not a claim that a planner holding only `t0` information could reproduce it.

### 3.2 How shortage is reported

**v8's shortage positive rate is 23.62%** against a real-world operating base rate near 3%
(`synthetic_rules.md` §9), and v8's event rates run **4×–6× over** Rane's stated bands
(deviation 56: shortage 940.8/yr vs 180–250, line stops 153.6/yr vs 15–25, expedites 640.4/yr vs
100–150).

**Any shortage probability from v8 is therefore mis-calibrated in level, by construction.**
ROC-AUC is a *ranking* metric and is unaffected, so ROC-AUC is what is reported. **No shortage
probability is quoted anywhere in this report**, and the caveat travels with the number as a
field in the scorer's output rather than as a footnote.

Shortage is `diagnostic_only` in `shipped.json` and is not shipped.

---

## 4. Stage 3 — what v8 newly unblocks

**B1 and B2 clearing is the most consequential result in the clearance audit, and it is not about
the model.**

### 4.1 Now startable, with the evidence

| work | unblocked by | evidence |
|---|---|---|
| **Phase 9.1 stock roll-forward** | **B1** | opening balance reconciles at **100.000000%** on 2,253,420 rows, 4,212/4,212 part-plants, max diff 0 — on seed 1005 too. All 18 of 9.1's named input columns are present and populated; `qty_available = qty_on_hand − qty_reserved − qty_blocked` holds on **100.0000%** of rows; `safety_stock_qty > 0` on **100%** |
| **9.1's verify gate** | **B1** | the gate is *"with fill fixed at 1.0 and timing at the promise date, the simulation reproduces `inventory_position_weekly.qty_available` exactly."* On v6/v7 that store was entirely zero, so the gate could not run at all. On v8 it is populated and the gate **is runnable** |
| **Guide 10.1's real stock-balance and safety-stock-floor constraints** | **B1 + B2** | deviation 39 dropped both for want of an opening level. v8 supplies it. B2 supplies the multi-period horizon they need: **12–26 periods per plan version** (median 19), **100%** forward, refreshed every **56 days** |
| **A genuine multi-week delivery schedule** | **B2** | deviation 38's "the deliverable schedule is single-period" no longer describes v8 — `production_plan` is 100% forward with 7 plan versions |
| **10.2's `SimulationScorer`, in quantity terms** | **B1** | it raised for want of inventory; inventory now exists per part-plant-week (2,253,420 rows, 4,212 part-plants). It can now measure expected **shortage** against a real balance instead of `ReducedScorer`'s unmet-demand-in-period. **It still cannot price it** — see §4.2 |
| **10.2's incumbent baseline** | **B8** | **97.05%** of part-plants carry a recorded split (v6 19.5%, v7 25.7%), 89.15% of versions with more than one supplier, splits summing to 100 on 100% of versions. Deviation 47's "nothing to recommend against" no longer applies |
| **A capacity ceiling in 10.2, at supplier grain** | **B7 (partial)** | `evidence_strength` 100% populated, `revealed_capacity_est` populated on 100% of constrained months — **but supplier-grain only**, see §4.2 |

### 4.2 What stays blocked, and why

| blocked | why | can a generator fix it? |
|---|---|---|
| **10.1's objective** — holding cost, ordering cost, truck capacity | **B3: 0 of 4 exist**, explicitly `null` under `parameters_v8.json`'s `EXCLUDED__by_design`. With `c_hold = 0` the schedule's *timing* stays unidentified and deviation 37's gate stays vacuous | **No — CLIENT ASK.** Deviations 36, 37 |
| **10.2's shortage-cost objective** | **B3**: no shortage cost. `SimulationScorer` can now measure expected shortage *quantity*; it cannot convert that to money without a rate | **No — CLIENT ASK.** Deviation 41 |
| **The min-volume constraint and its penalty** | **B5**: neither a commitment period nor a penalty basis exists as a column. `min_volume_commitment` now varies (12–1,087) and `penalty_clause_inr` varies (₹1,670–₹1.8 m), but a quantity with no window cannot be enforced and a penalty with no basis cannot be priced | **No — CLIENT ASK** |
| **Any constraint keyed on `sourcing_channels`** | **B6 partial**: `is_approved` constant `true` and `approval_status` constant `approved` on all 16,072 channels. Deviation 40's INERT verdict stands for that path. The **alternate-source** path *can* now bind — 16.7% of alternate sources are unqualified | Partly — v8 fixed half of it |
| **A per-supplier-part capacity ceiling** | **B7 partial, deviation 60**: `revealed_capacity_monthly` holds **420 supplier-part pairs for 420 suppliers** — one nominal part each — while a supplier serves a median of **37** parts. `supplier_capacity.part_id` is 100% NULL. 10.2 enumerates supplier-**part** candidates, so one shared ceiling must be attributed across 37 parts — a declared modelling choice, not a constraint the data supplies | Yes, but it is not fixed in v8 |
| **Anything reading `supplier_performance_weekly`** | **B11**: 7 identically-zero columns including `otd_rate_last13`. Not currently realised — no panel feature reads that store (§2.3) | Yes |
| **Phase 9.2's G10 copula gate** | **`db/ground_truth.py` does not exist in this repo.** The gate calls `ground_truth.realised_correlation_matrix()`; there is no such module, and no `db/prove_claims.py` either. The copula can be *built*; its acceptance gate cannot be *run* | Not a data issue — a missing module |
| **The copula's shared-checkpoint grouping arm** | `logistics_lanes.via_checkpoint` is **100% NULL** (B11). Grouping can use `supplier_group_id` (84 groups) and `supplier_upstream` (260 rows); the lane-checkpoint arm the guide names cannot be formed | Yes |

**A note on the BOM**, corrected from the clearance report: `bom.parent_part_id` is 100% NULL, but
that is **benign** — `bom_level` is constant 1, so the BOM is single-level (180 products, median
21.5 parts each) and `parent_part_id` is only meaningful at level ≥ 2. The milder real
observation is that **the BOM is flat**, so multi-level explosion is never exercised.

### 4.3 Phase 9 cost estimate — NOT started

The simulation grid on v8, measured:

```
4,212 part-plants x 83 snapshots x 13-week horizon x N = 1,000 paths
  = 4,544,748,000 path-weeks;  per-snapshot state 219 MB float32
```

| component | estimate | basis |
|---|---|---|
| 9.1 roll-forward **arithmetic** | **~0.2 min per seed** (0.14 s per snapshot) | timed at full scale on this machine |
| 9.1 **head inference** (draw f, T, C per path) | **dominant term** — hours, not minutes | it is Phase 1's inference cost repeated per snapshot over the full channel population, not the 36,000-row test fold |
| Part-plant ↔ channel mapping + BOM explosion | ~1 h to build, then cheap | single-level BOM, 4,340 `part_plant` rows, 16,072 channels |
| 9.2 copula: ρ from pre-`t0` co-occurrence | ~1 h | 84 groups; must be estimated from data before `t0`, never from generator parameters |
| 9.2 **G10 acceptance gate** | **cannot run** | `db/ground_truth.py` absent |
| **Implementation** (`ml/sim/montecarlo.py`, `ml/sim/copula.py` do not exist) | **the real cost** — neither file exists | `ml/sim/` is not in the tree |

**Honest summary: Phase 9 is now *possible* on v8 where it was impossible on v6/v7, and it is
mostly unwritten.** The blocker was never compute — the roll-forward is a second per snapshot.
It was the opening balance, and that is what B1 delivers. Budget the work as implementation plus
one inference sweep, not as simulation compute. **Phase 9 was not started.**

---

## 5. Deviations — continuing the index

| # | Spec / prior report says | Measured | Where |
|---|---|---|---|
| **58** | B9 blocks Phase 0/1 | **it does not.** Same condition v6/v7 carried; Phases 1–11 trained on those worlds. B9 blocks only guide 11.2 (45), arrival ranking claims (32) and deployable promise-date readings (48). Corrected rule: B1 + B12 → proceed; assertion stays armed; no line-level feature; promise comparison retired | §1 |
| **59** | Phase 11's row-feature path is "default off" and "every existing configuration is bit-identically unchanged" (deviation 45) | **neither holds.** `ef3048d` attached `line_age_weeks` / `line_recorded_ts` unconditionally while the assertion guards on their presence, so **arrival training has been unrunnable on v6, v7 and v8 alike** — 244,000/244,000 rows on each. Fixed by gating the attachment; the assertion is untouched and still fires | §2.5 |
| **60** | `revealed_capacity_est` is a per-supplier-part ceiling, per §6's K(supplier, part, month) | **supplier-grain only**: 420 supplier-part pairs for 420 suppliers, one nominal part each, against a median 37 parts served. B7 → PARTIAL | `v8-clearance.md` §2 B7 |
| **61** | the panel is 14 value + 10 indicator (deviation 2/4) | **world-dependent.** v8 is **15 + 10 = 25**; v6/v7 stay 14 + 10 = 24. Derived by measurement and asserted per world, never borrowed; v6 rebuilds byte-identically | §2.2 |
| **62** | `loop.py` exposes the Phase 11 pilot path | **it does not.** `loop.py:130` reads `args.row_features`, which argparse never defines — there is no `--row-features` flag, so the pilot is reachable only through `backtest.py`. Not fixed here | §2.5 |
| **63** | Phase 9.2's G10 gate validates the simulation | **it cannot be run**: the gate calls `ground_truth.realised_correlation_matrix()` and **`db/ground_truth.py` does not exist** in this repo, nor does `db/prove_claims.py`. The copula can be built; its acceptance gate cannot | §4.2 |

---

## 6. Open items

1. **Arrival results on v6 and v7 should be re-confirmed** now that arrival training runs again
   (deviation 59). Nothing suggests the *numbers* changed — the defect made cells crash, not
   mis-train — but no arrival cell has completed on any world since `ef3048d`, so that is an
   assumption until one does.
2. **`--row-features` has no CLI flag** (deviation 62). The pilot path is reachable only through
   `backtest.py`. Wire it or delete the dead `getattr` in `loop.py:130`.
3. **`db/ground_truth.py` is missing** (deviation 63), so Phase 9.2's G10 gate cannot run. Find
   it or replace the gate with one that can fail on this repo.
4. **The capacity grain** (deviation 60) needs a decision before 10.2 quotes anything: either
   attribute the supplier ceiling across its parts and say so, or ask for per-part capacity.
5. **v8's event rates** (deviation 56) remain 4×–6× over Rane's stated bands. Not a Phase 0/1
   blocker; it bears on every Phase 9/10 claim about how often anything happens.
6. **`supplier_performance_weekly`'s seven zero columns** (deviation 54) are harmless today
   because nothing reads that store. Decide whether it is live before anything does.
7. **B3 and B5 are client asks** and have been for three worlds. No generator can close them.

---

## 7. What was run, and what was not

Phase 0 ran in full. Stage 3's analysis is complete and required no training.

Stage 2's queue — 21 cells, `ml/configs/shipped.json` unmodified — was running when this section
was written; §3 records the protocol and the two reporting rules in force, and the tables are
completed when it lands.

**Nothing was tuned. No learning rate was retuned. `ml/configs/shipped.json` is untouched. No
assertion was disabled, weakened or bypassed** — the one change near an assertion stops
line-level columns being *attached*, which is the standing rule itself, and the assertion was
verified still firing in both directions afterwards. `db/gen_v6/**`, `db/gen_v7/**`,
`db/gen_v8/**`, `db/validator.py`, `docs/specs/synthetic_rules.md`, `db/dataset_structure.md`,
`docs/specs/model_plan.md` and every pre-existing report in `reports/` are unmodified;
`reports/v8-clearance.md` is this session's own and is corrected in place.

**No v8 number in this report is compared to a v6 or v7 number as though it were the same
measurement.** Where v6/v7 figures appear they are labelled contrast.
