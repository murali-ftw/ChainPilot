# HADES v4 / ChainPilot — Final Report

**A consolidated account of what was built, what it can do, what it cannot, and what a real deployment would need.**

This document supersedes nothing. Every phase report remains authoritative for its own detail and is cited throughout
as `report §section`. Where two reports disagree, the later correction wins and both are named (§0.2). Every number
here appears in a named report section; nothing was re-run, re-scored or re-measured to write it.

`inventory_position_weekly` was not read in the writing of this report, nor in any modelling phase of the project.

---

## 0. How to read this document

### 0.1 What this project is

A supply-chain forecasting system built against a **synthetic** dataset of an Indian auto-components manufacturer,
intended as a proof of concept for a real extract from Rane. Two synthetic worlds (`gen_v6`, `gen_v7`) were generated
and validated, four forecasting tasks were built and evaluated, and two optimisers were attempted.

**No number in this document is a measurement on Rane's data.** Every score is within-world: the same generator
produced the training and the test data, separated only by time (`phase1_2.md` header). A synthetic world has no
schema drift, no ERP migration, no relabelled part numbers. These are an optimistic upper bound.

### 0.2 Corrections — where a later phase overturned an earlier claim

Seven claims were made and later corrected. Each is marked **SUPERSEDED** where it appears, and none was silently
dropped.

| # | the original claim | where | the correction | where |
|---|---|---|---|---|
| 1 | Arrival lateness ROC-AUC 0.7561 / 0.7559 for the old head | `phase1_2.md` §7 | **Units error** — a standardised prediction minus a promise in weeks. In consistent units: **0.7620 / 0.7604**, and the promise week alone scores 0.7482 / 0.7498 | `phase-5.md` §7, §11 item 7 |
| 2 | Fill CRPS ≈ 0.059 / 0.099, "right on average" | `phase1_2.md` §7 and Phases 2–4 | **The legacy formula steps the CDF at 20 bin edges and cannot see where inside the top bin the mass sits.** Exact (point-mass-aware) CRPS is 0.0752 / 0.1141 for the same old head — the point masses are worth 18% / 10% | `phase-5.md` §5 |
| 3 | Fill calibration reaches "parity with LightGBM on v6" | `phase-5.md` Addendum A | Downgraded to "not distinguishable" (`phase-6.md` §6), then **withdrawn**: under an identical protocol every head seed is worse than every LightGBM-22 seed on v6, by 0.0024 ECE with disjoint ranges | `phase-7.md` §5 |
| 4 | Capacity v6's shipped depth is not distinguishable from the strongest deployable baseline (Phase 7 binding statement 3) | `phase-7.md` §6 | **True where measured (2025), contradicted on origins 1–6.** h⁴ beats B5 in 11 of 12 pre-2025 windows | `phase-8.md` §3.1, §4 3a |
| 5 | Phase 8's interim counts: origins 1–6 h⁴ 10/1/1; h⁴ beats B5 in 10 of 12; coverage below nominal in 13 of 16; exceedance 4–12% | `phase-8-interim.md` §1 | **Four miscounts**, recounted in code: **9/2/1**, **11 of 12**, **12 of 16**, **4–15%** | `phase-8.md` Appendix A, deviation 26 |
| 6 | Capacity's 2025 losses reflect a late-period regime | `phase-8-interim.md` §1 | **Withdrawn.** No trend (slope +0.004/+0.010 per year, p = 0.42/0.38), no late step (best change point is COVID 2020-03), no ceiling; the only dated generator terms end 2022-06-30 | `phase-8.md` §4 3a, deviation 25 |
| 7 | The promise-date baseline is deployable — "a planner already holds it" | `phase-7.md` §7 | **False on this data.** The promise belongs to a PO line created 7–89 days *after* the snapshot (median 49), in 244,000 of 244,000 rows in both worlds. Its 0.87–0.89 C-index uses future information | `phase-11-pilot.md` §3, deviation 48 |

### 0.3 A note on discipline

Two rules govern every number: **selection happens on validation, never on an evaluation window**, and **a margin
inside its band is "not distinguishable", never "better"**. Bands are measured per metric, per configuration, per
world, per origin, and never borrowed (`phase-6.md` §6). Where this document says "not distinguishable" it means the
seed ranges overlap, not that the difference is small.

---

## 1. Executive summary

A forecasting system for four supply-chain questions was built end to end on synthetic data, evaluated across eight
half-year windows, and taken as far as two optimisers. **Two of the four forecasts work and are worth showing a
client. One is untestable on this data for a structural reason. One was never the product path.** Both optimisers are
blocked by missing data rather than by any modelling difficulty — and that, rather than the models, is the most
useful output of the project.

The honest one-line summary: **the modelling is sound and the data is not sufficient to deploy it.**

### The capability table

| capability | status | what it means |
|---|---|---|
| **Fill-rate distributions** (will this order line arrive complete, and if not, how much?) | **WORKS** | Best proper score of any forecaster tested, in both worlds and across all 8 backtest windows (`phase-9a.md` §A.3). Marginal calibration is **worse than a LightGBM's** in 12 of 16 windows, which is why a switch to that LightGBM is proposed and unapproved (§5.2) |
| **Arrival lateness** (which orders will miss their promised date) | **WORKS** | The head adds information beyond the promise date in **8 of 8** windows, every seed, beating LightGBM every time (`phase-8.md` §4 3d). The size varies 20-fold: quote **+0.02 median**, range +0.001 to +0.031 |
| **Arrival ranking** (which order arrives first) | **UNTESTABLE HERE** | Not "the model loses" — the comparison is unwinnable by construction. The promise date is built from the same line-level quantity as the label and is unavailable at prediction time (§5.1) |
| **Capacity strain — the median** (which suppliers will be over-utilised) | **WORKS, UNSTABLY** | Depth-4 beats depth-0 in 10 of 16 windows and loses 4, and **nothing measured predicts which** — not the direction, magnitude, range or season of the label shift (`phase-8.md` §4 3a) |
| **Capacity strain — the intervals** | **NOT QUOTABLE** | 80% intervals cover 0.72–0.81, below nominal in 12 of 16 windows, in every model class (`phase-8.md` §3.2) |
| **Allocation recommendations** (how to split volume between suppliers) | **NO QUOTABLE OUTPUT** | Only 25% of part-plants have a recommendation that survives both the shortage-cost assumption and the seed bands; a third of those say "change nothing" (`phase-11-pilot.md` §4.2) |
| **Delivery schedule** (when to place orders) | **SINGLE-PERIOD ONLY** | The MILP solves and is verified, but the data carries a one-week forward horizon and no holding cost, so there is no timing decision to make (`phase-10.md` §4) |
| **Inventory / Monte Carlo shortage** | **BLOCKED** | No opening stock level exists in the dataset in any usable form (`phase3_closeout.md` §1c) |

### What works, and why it is credible

- **Fill and arrival-lateness are genuine results**, measured through one scorer on asserted-identical rows, with
  3-seed bands, across 8 rolling origins in 2 worlds. Fill's proper score beats every baseline in 15 of 16 windows
  against LightGBM-22 and 16 of 16 against a model-free histogram (`phase-9a.md` §A.3).
- **The evaluation discipline is the most transferable thing here** (§4, §7). Ten "gates that cannot fail" were
  caught and named; several corrected results that had already reached a report.

### What does not work, stated plainly

- **The graph — the architectural premise of the project — contributes nothing on the validation harness.** G4 fails
  on both worlds: the graph's share of above-chance signal is ≈0% on arrival, −1.0% on shortage and +3.6% of a small
  gain on fill (`validation7.md` §3, §11). Inside the trained models the picture is better but not uniform (§5).
- **Capacity depth is window-dependent with no identified predictor.** It cannot be quoted as a property of the
  configuration (`phase-8.md` §4 3a).
- **Both optimisers are data-blocked**, not solver-blocked (§6).
- **One shipped configuration change is declarative only.** `shipped.json` names a LightGBM for fill; the serving
  path cannot load one and silently serves the superseded neural head (`phase-11-pilot.md` §2.2).

### The single most important thing to fix

**The opening stock balance.** It blocks the Phase 9 simulation, the delivery schedule's stock-balance form, and the
allocation optimiser's real objective — three of the four things a client would most want (§8.1).

---

## 2. What was built

### 2.1 In plain terms

A model that reads **one weekly history per sourcing channel** — a channel being one supplier supplying one part to
one plant — and predicts four things about the next 90 days. The weekly history is 52 weeks of order quantities,
receipts, fill rates, lead times, utilisation and reporting lags.

Three pieces stack:

1. **A temporal encoder** reads each channel's 52-week history and compresses it to a state vector.
2. **A graph encoder** lets channels that share a supplier, a part or a plant exchange information.
3. **Four task heads** turn that state into a prediction — a distribution over arrival weeks, a distribution over
   fill rates, three utilisation quantiles, or a shortage probability.

The model never sees an entity identifier. Every feature is an attribute an unseen entity would also have, because a
model shipped to Rane meets only entities it has never seen (`loader.py` module docstring; `phase1_2.md` §6).

### 2.2 Temporal-SHARE, and why it is not an STGNN

The architecture is a **TCN front-end feeding a relational graph encoder**, applied per snapshot. It is not a
spatio-temporal GNN in the usual sense: **time is handled entirely by the TCN, and the graph sees only one instant**.
There is no temporal edge, no recurrence across snapshots, and no message passing through time. One optimiser step
consumes one whole snapshot of all 16,072 channels (`phase1_2.md` §5).

A second structural point: **entity nodes carry no sequence of their own.** A supplier has no weekly row, so it
enters the graph as the mean of its member channels' TCN states — content-derived, which keeps the model inductive. A
learned per-supplier embedding would key on identity, which the rules forbid (`phase1_2.md` §6).

**The TCN** — kernel 2, dilations 1/2/4/8/16/32, 6 layers, hidden 64, receptive field 64 weeks, 46,144 parameters.
**Residual connections are a required element, not an option**: built literally as six `Conv1d`+ReLU layers the
encoder scored C-index **0.5000**, exactly chance, while a ridge probe on the identical tensors scored 0.6464. Adding
a 1×1 residual projection per layer, changing nothing else, took it to 0.6524 (`phase1_2.md` §1.1; deviation 1).
Causality is asserted, not assumed: perturbing the last timestep moves no earlier position by a single bit
(`phase1_2.md` §6).

### 2.3 The three graph encoders, with real parameter counts

| | **SHARE** | **SHARE-lite** | **HeteroMP** | h⁰ (no graph) |
|---|---|---|---|---|
| mechanism | basis-decomposed per-relation weights (B = 10) + relation-blind attention | free per-relation weights + the same attention | learned per-relation means + a gate | — |
| hidden | 128 | 128 | 64 | — |
| depth | 4 layers | 4 layers | 2 rounds = 4 layers | — |
| **encoder parameters** | **730,992** | **468,608** | h¹ 33,216 · h⁴ 66,432 | 0 |
| **full model** | **793,777** | — | h¹ 83,585 · h⁴ 116,801 | **50,369** |

Sources: `phase1_2.md` §6; SHARE-lite from `phase-4.md` §4.

**The basis decomposition is dead weight on this graph, and it should be said plainly.** It saves parameters only
when the number of relations R exceeds the number of bases B. Here R = 6 against B = 10, so per layer the basis
scheme costs 163,840 + 60 parameters against 98,304 for the free per-relation weights it replaces — **1.67× more**
(`phase1_2.md` §6; deviation 6). Removing it (SHARE-lite) cost nothing measurable across four arrival cells, largest
deviation 0.0018 against a 0.0028 seed band, and saved 36% of the encoder (`phase-4.md` §4). **So whatever SHARE buys
on this graph comes from the attention, not the basis trick.**

**h⁰ is architecture-independent — verified, not asserted.** At depth 0 no graph module is constructed at all, so
"SHARE h⁰" and "HeteroMP h⁰" build the identical object; trained three times under three labels the maximum gap was
0.0004 (`phase1_2.md` §6). Every graph-versus-no-graph comparison is therefore a clean encoder comparison.

### 2.4 The graph's actual shape, against the specification

| | specification | **measured** |
|---|---|---|
| node types | 6 | **4** — `supplier_group` and `po_line` are not in the graph |
| relations R | 20 | **6** (3 undirected: channel↔supplier, ↔part, ↔plant) |
| nodes | — | **17,119** (420 supplier, 620 part, 7 plant, 16,072 channel) |
| edges | — | **48,216** undirected (96,432 directed) |
| median channel degree | — | **3** |
| components | — | **1**, covering 100% of nodes; 0 isolated |

Sources: `validation5.md` §B4 and §3 (step 4); `phase1_2.md` §6; deviation 7.

The gap between R = 20 and R = 6 is the root of the basis-decomposition finding above: **the specification's
parameter economics assumed a graph four times as relationally rich as the one that exists.**

### 2.5 The four heads, and why each was chosen

| head | task | form | why |
|---|---|---|---|
| **Hazard** | arrival week | independent per-week sigmoids, censored negative log-likelihood, 12 weeks + ">12" | 43.6–46.2% of arrival labels are right-censored (the order has not arrived at the horizon). The old MSE head used **only uncensored rows — 55.7%**; the hazard head puts **all 176,000 rows** into the loss, asserted every epoch |
| **22-cell CDF** | fill rate | 22 outputs with explicit point masses at 0 and 1, RPS loss | The label is ~91% a point mass at exactly 1.0. The specification's "20 bins" list is arithmetically 22 cells; uniform bins cannot represent a point mass |
| **Monotone quantile** | capacity strain | 3 quantiles via softplus increments | Guarantees P10 ≤ P50 ≤ P90 structurally — **0 crossings on every test row** and on 51,000 synthetic rows |
| **Binary** | shortage | BCE | **Diagnostic only.** The product path is Monte Carlo simulation, which is blocked |

Sources: `phase-5.md` §3, §5, §7, §8, §9.

Every head was built at h⁰ and verified **depth-agnostic**: constructed on every encoder configuration and run
forward and backward on the real v6 graph, **28 of 28 combinations pass** (`phase-5.md` §5).

### 2.6 What ships today

`ml/configs/shipped.json`, version **`phase10-shipped-1`** (previously `phase6-shipped-1`):

| task | configuration | recalibration |
|---|---|---|
| arrival_week | SHARE-lite h⁴ @ 2.5e-4 for **ranking**; h⁰'s distribution **served always** (drift gate removed) | 13-cell moment matching, fitted per run on validation |
| fill_rate | **`b5flat22`** — LightGBM-22 on flat graph features, no identity keys *(declarative only — see §5.2)* | 22-cell MM or VS, selected on validation log score |
| capacity_strain | HeteroMP h⁴ @ 2.5e-4 | none |
| shortage_qty | HeteroMP h¹ @ 1.25e-4 | none — **diagnostic** |

Common: staleness gate **off**, reconstructed feature **off**, 3 seeds, fixed split, fit window 2019–2025.

---

## 3. The data — the honest account

### 3.1 Seven generations to get two usable worlds

| run | what it was | validator result (seed 1001) | verdict |
|---|---|---|---|
| 1–4 | the `db/` lineage | 88/74/20 → 99/73/22 | **not usable** throughout |
| 5 (Part A) | the generator as shipped | **not run — no dataset** | aborts at `po_lines`: 9 columns where the schema mandates 12 |
| 5 (Part B) | `generator_v5.py`, a rebuild | **171 / 1 / 4** | usable with fixes |
| 6 | `generator_v6.py` | **170 / 2 / 4** (restated 171/1/4) | usable with fixes |
| 7 | `generator_v7.py` | **163 / 9 / 4** | **NOT USABLE** |

Sources: `validation5.md` §1 and Part B; `validation6.md` §1; `validation7.md` §1, §9.

**Runs 1–4 were fitted, not generated.** Seven §13 outcome values appeared as literal inputs: the zero-order week
share was set at 0.19, the seasonality ratio hard-coded as four month bumps, censoring a flat 8% coin flip. The
signature is unmistakable — *the two outcomes that were set land in band; every outcome left to emerge lands far
outside it* (`validation5.md` §2.8).

The rebuild inverted that discipline: only mechanism parameters are set, and outcomes are measured. The literals
audit is clean on v5, v6 and v7 — **no band value appears anywhere in the source** (`validation6.md` §2).

### 3.2 What v6 and v7 are

Both are ten full years (2016–2025) plus a partial trailing quarter, cut at **2026-03-31** — the Indian fiscal year
end, chosen because it is the most natural date for a real extract and because its trailing window sits in a
seasonal trough (`validation6.md` §4). Both pass exact integer conservation, the inventory ledger identity, and the
as-of/leakage tier of the validator (`validation6.md` §11).

**They are two different regimes, not two seeds.** v7 changed the demand process: the per-channel demand level was
split into a persistent component and an AR(1) transient with a 66-week autocorrelation, preserving the marginal
variance exactly (`validation7.md` §4). The consequence is that v7 is **more learnable and worse validated** than v6:
the part-shortage GBM moved from *losing* to a naive baseline to beating it by 83%, while six of fourteen band
outcomes regressed on seed 1001 (`validation7.md` §11, §6). That is the opposite of the usual trade, and it is why
both worlds are carried through every comparison rather than one being chosen.

### 3.3 The v6 label repair, and its determinism proof

v6's `shortage_qty` labels held 45,942 rows with **zero negatives**, so PR-AUC was 1.0000 by construction and
measured nothing (`phase1_2.md` §11).

The repair was done properly, and the proof matters more than the fix:

- **Determinism first.** `generator_v6.py --seed 1001` was re-run into a scratch directory and hashed table by table:
  **48 of 49 byte-identical**; the exception is `dataset_coverage.csv`, whose 9 substantive columns are identical and
  which differs only in a wall-clock stamp. `training_labels.csv` was byte-identical (`phase1_2.md` §3.1).
- **The method was a splice, not a regeneration.** A corrected rebuild emits more shortage rows, which advances the
  `label_id` counter and consumes different RNG draws — 19 downstream tables would have moved. So corrected shortage
  rows were spliced in, with non-shortage rows copied through as **the original raw bytes** (`phase1_2.md` §3.2).
- **Proof nothing else moved:** non-shortage rows hash identically before and after (946,200 rows), 0 duplicate
  `label_id`, all 47 other tables unchanged, and **0 of 234 validator checks changed status** (`phase1_2.md` §3.3, §3.5).
- **Result:** 16,795 positive / 107,705 negative, a 13.49% positive rate against v7's 12.8% reference.

The original is preserved as `training_labels.pre_fix`.

### 3.4 What the two worlds do and do not establish

**They establish** that a result is not an artefact of one demand process: every headline in this project is reported
per world, and several reverse between them (capacity depth, fill calibration, arrival's lateness margin).

**They do not establish generalisation to Rane.** `synthetic_rules.md` §17 is explicit that seed- and world-to-world
agreement is not a proxy for a real extract, and within-world scores are weaker evidence still (`phase1_2.md` §12
item 6). Two worlds from one generator family share every structural assumption the generator makes.

### 3.5 The defects that bounded the work

| defect | measurement | real extract too? |
|---|---|---|
| **29 all-zero numeric columns** across 5 tables, identical in both worlds — 13 of them every numeric column of `inventory_position_weekly` (`phase1_2.md` §2) | 8.6M rows | **Artefact.** Generator placeholders never populated |
| **No opening stock balance anywhere.** `inventory_position_weekly` all zero and only 12 distinct dates over nine years; `inventory_snapshots.qty_on_hand` **97.5% negative** in v6 (54.9% v7); the `inventory_transactions` cumsum agrees on **0.33%** of rows (`phase3_closeout.md` §1c) | 518k rows | **Artefact**, but the *consequence* is real: without a stock level no shortage simulation is possible anywhere |
| **Constant contract terms** — MOQ 10, lot size 25 (and 10 in a second table, disagreeing), planning lead time 30 d, max volume cap 5,000, min volume commitment 100, penalty ₹10,000, `is_shutdown` 100% zero (`phase-10.md` §2) | all rows | **Artefact.** A real contract set varies |
| **No holding cost, no ordering cost, no truck capacity** anywhere in the 49-table schema (`phase-10.md` §2.1) | — | **Partly real.** Many ERPs do not carry a carrying-cost rate; it is a finance parameter, not a transaction |
| **Single-period forward plan.** `part_demand_weekly` carries one week per part-plant, 30 days out, at 12 as-of dates ~9.5 months apart; `production_plan` is **entirely retrospective** — `target_period` is 1–8 days *before* `recorded_ts`, 0.00% forward in both worlds (`phase-10.md` §2, deviation 38) | 52,080 rows | **Artefact.** A real MRP publishes a multi-period forward plan |
| **Arrival labels attach to lines not yet raised.** In **244,000 of 244,000** rows in both worlds the `po_line` is created 7–89 days *after* its own snapshot, median 49 (`phase-11-pilot.md` §3.2) | all arrival rows | **Artefact of the label design.** A real "when will this order arrive" question is asked about an *open* order |
| **`qualification_status` constant "qualified"; `is_approved` constant 1; `revealed_capacity_est` 100% NULL** (`phase-10.md` §2) | 16,072 / 51,660 rows | **Artefact.** Real sourcing has unqualified and in-qualification suppliers |

**The distinction matters for what transfers.** The all-zero columns, the constant contracts and the retrospective
plan are generator gaps — a real extract would very likely carry these fields properly. The *consequences* recorded
in this report (no stock simulation, no multi-period schedule, no allocation objective) are therefore **not
predictions about Rane**; they are statements about what could be built here. §8 turns each into a specific ask.

One defect is worth separating: **the arrival label design is a modelling choice, not a population gap.** It makes
the arrival task a channel-level forecast of orders *not yet raised*, which is coherent — but it makes the
promise-date comparison unwinnable (§5.1).

---

## 4. Method — the discipline

This is the most transferable output of the project. The models are ordinary; the evaluation discipline is not.

### 4.1 As-of rules

Every feature is built from rows whose **`recorded_ts <= t0`**, never `event_ts <= t0`. The distinction is the whole
point: a fact that happened before the snapshot but was keyed in afterwards was not knowable then. The generators
implement a per-table recording-lag process, and the validator confirms the derived stores bucket on
`max(event_week, recorded_week)` at 100% on the cells where the two rules disagree (`validation6.md` §11).

**The as-of gate is binding, and it was tested.** G5 shuffles `recorded_ts` within each table and rebuilds: the
arrival head loses **12.2% / 14.6% of above-chance signal** (`phase-2.md` §6, §gate).

Causality is re-asserted on the inference path of all four shipped models: perturbing every panel value, lag and flag
after t₀ leaves predictions bitwise identical where the model is deterministic, and within the model's own
repeat-forward floor where it is not, while one week of in-window change moves the output by 10⁻² to 10⁰
(`phase-6.md` §5).

### 4.2 Selection on validation only

Configurations are chosen on a validation fold and never on an evaluation window — **even when the evaluation answer
is clearer, and even when validation cannot settle the question.** Two consequences the project lived with:

- **Capacity v6's shipped depth rests on validation against a real, seed-stable test disagreement.** HeteroMP h⁴
  beats h⁰ on validation by 0.0031 and loses on test by 0.0030, with no overlap on either fold, in opposite
  directions. The configuration was not changed, because the only evidence against it is test data (`phase-6.md` §6, §8).
- **When validation cannot separate two options, that is the finding.** The recalibrated validation ECE is ≈0 for
  every fill arm by construction, so the comparison moved to the raw pre-recalibration figure rather than to the
  evaluation fold (`phase-9b.md` §3.1).

### 4.3 Bands, never borrowed

A verdict is given only when two seed ranges are **disjoint**. Bands are per metric, per configuration, per world,
per origin.

**Why this is not pedantry:** arrival SHARE-lite h⁴'s C-index spread is **0.00197 on v6 and 0.00026 on v7** — a 7.6×
difference for one configuration. A band borrowed across worlds is wrong in both directions, too narrow for v6 and
too wide for v7 (`phase-6.md` §6). An earlier failure (R2) was recorded against a band **4.9× too narrow** for the
configuration it judged, and re-read at 0.39× against its own (`phase-6.md` §4).

### 4.4 The rolling-origin backtest, and the overlap it carries

Eight origins, each a training cut plus a half-year evaluation window, both worlds. The specification gives no
validation slice and the loop needs one, so **the 12 months before each cut are carved out of training**; evaluation
windows are untouched, and train + validation is asserted equal to the specification's cut. All 16 origin × world
combinations pass 24 of 24 assertions (`phase-8.md` §2; deviation 18).

**Deviation 20 bounds every number in the backtest, and is stated rather than asserted away.** Labels look 90 days
ahead, so at every origin the last two validation snapshots' outcomes land inside the evaluation window — **8,000
arrival rows, 8,000 fill, 5,000 capacity, 3,000 shortage**. Early stopping, recalibration and the drift baseline are
all fitted on validation, so every origin's model is selected with a sliver of evaluation-window information. **No
training row's outcome reaches the evaluation window.** The fixed split every earlier phase used has exactly the same
overlap, so asserting it away would have excluded every prior number; it is measured and reported instead
(`phase-8.md` §2, §7).

### 4.5 Artifact identity asserted at write time

Names for bundles, prediction files and index keys derive from **one module** (`ml/artifact_identity.py`), uniqueness
is asserted across the whole set before anything is written, and each write is guarded against an overwrite whose
recorded identity differs (`phase-9b.md` §4).

This rule was earned: `loop.bundle_dir` and `backtest.export` each built the same name independently, they drifted,
and a truncated-history cell **silently overwrote a full-history cell's predictions**. It was caught by noticing that
a number had moved — luck, not a control (`phase-9a.md` deviation 28). The sweep after the fix found no other
collision: 198 bundles, no duplicate keys, 1,560 prediction files all accounted for (`phase-9b.md` §4).

### 4.6 GATES THAT CANNOT FAIL

**A gate that cannot fail is not a gate.** Before writing a check, state what would make it fire and confirm that
outcome is reachable on this data. Ten instances were found and named across this project — several after the gate
had already "passed" in a report.

| # | where | the gate | why it could not do its job |
|---|---|---|---|
| 1 | `validation6.md` §2 (run 1) | line-stop test requiring *every* channel of a part-plant to be empty at once | could never fire — 0 events |
| 2 | implementation guide §5.2 | fill band-coverage G5 | cannot fail: 91.4% of fill labels are exactly 1.0, so the empirical CDF jumps at the top bin and coverage error pins at 0.5 for **any** forecaster, including LightGBM (`phase1_2.md` §7) |
| 3 | `phase3_closeout.md` §9.1 | "the simulation reproduces `qty_available` exactly" | vacuous against an all-zero column — satisfied by a simulation that outputs zero |
| 4 | `validation7.md` §2 (amendment 01) | the censored-tail check | a gate that could not **pass**: a 1.47× structural floor against a 0.3–1.2× band, for any world with non-zero lead time |
| 5 | `validator_amendment_02.md` | a constant-column gate firing on all three buckets | would be ignored within a week; narrowed to all-zero numerics only (`phase1_2.md` §2) |
| 6 | `phase-8.md` §4 3c | arrival's drift gate at 1.0 pp | its output flips with the training seed in **8 of 16** windows, and the two branches are never distinguishable |
| 7 | `phase-9b.md` §3.1 (deviation 33) | comparing recalibrated calibration on validation | the recalibrator is fitted on that fold, so marginal ECE is ≈0.0000 for **every** arm — head, LightGBM-22, B5-flat-22 and B2 alike |
| 8 | `phase-10.md` §2.2 (deviation 37) | guide 10.1's verify gate | "zero holding cost" and "no capacity constraint" are **both true of every part**, and with c_hold = 0 ordering first costs exactly what ordering last costs — it tests the solver's tie-breaking |
| 9 | `phase-10.md` §5.3 (deviation 42) | 10.2's seed-band check | keyed to the capacity **P50**, which is below 1.0 for 98% of suppliers, so the term was inert, every seed scored identically and the check could not fail |
| 10 | `phase-10.md` §5.5 (deviation 43) | 10.2's tooling verify gate | first "passed" while blocking **nothing**: the frozen suppliers were not qualified at the plants tested, so no candidate ever offered them volume |

#### The counter-example: the gate that could fail, did, and prevented a leaking model

Phase 11 added a per-batch as-of assertion to the arrival feature path: *every row used must already be recorded at
its own snapshot.* It **fired on the first training batch of the first cell**, before any bundle was written
(`phase-11-pilot.md` §3.1).

What it caught was not a small bug. In **244,000 of 244,000** arrival label rows, in both worlds, the PO line is
created 7–89 days after its own snapshot. Had the assertion not been there, the pilot would have trained a model on
`promise_week` and line age — quantities that do not exist at prediction time — and reported a C-index that beat the
promise-date baseline. **That number would have been leakage, and it would have gone to a client as arrival's
headline result.**

The contrast with the ten failures above is the lesson: nine of them were checks that passed while measuring
nothing. This one was a check that could fail, and the single time it fired it paid for every other assertion in the
project.

---

## 5. Results per task

Every figure below is test-fold, through one scorer, on asserted-identical rows. "Windows" means origin × world cells
in the rolling-origin backtest; the fixed split (train ≤ 2023, validate 2024, test 2025) is named separately.

### 5.1 Arrival — the task with the most interesting result

**What it predicts:** for the PO lines that will be raised on a channel in the next 90 days, the week each arrives,
as a distribution over weeks 2–12 plus ">12". 43.6–46.2% of labels are right-censored.

**What ships:** SHARE-lite h⁴ @ 2.5e-4 for ranking; h⁰'s recalibrated distribution served **always** since Phase 10
removed the drift gate.

#### The headline numbers

| measurement | v6 | v7 | windows behind it |
|---|---|---|---|
| C-index, shipped head (fixed split) | 0.6734 | 0.6773 | 3 seeds, 1 window (`phase-7.md` §6) |
| **Lateness ROC-AUC margin over the promise date** | +0.0332 | +0.0251 | fixed split (`phase-7.md` §7) |
| Same margin, backtested | **+0.0049 to +0.0306** | **+0.0014 to +0.0277** | **8 windows** (origins 1, 2, 6, 7 × 2 worlds) (`phase-8.md` §4 3d) |
| Head beats LightGBM B5 on lateness | 8 of 8 windows, +0.0041 to +0.0236 | ″ | 8 windows (`phase-8.md` §4 3d) |
| **C-index vs the promise date** | −0.2074 to −0.2170 | −0.1836 to −0.2144 | 8 of 8 windows (`phase-9b.md` §2) |

#### The lateness claim — quotable, with its range

**The head adds lateness information beyond the promise date in 8 of 8 windows, every seed, beating LightGBM every
time.** The margin exceeds the head's own seed spread in all 8 (1.2× to 22.7×).

**But the size is not stable, and the 2025 headline is the best window, not the typical one.** The margin runs
**+0.0014 to +0.0306, median +0.02** — a 20-fold range. The two largest are the 2025 windows, which is where the
fixed-split headline of +0.033 / +0.025 came from. Under a stricter test — requiring the head's worst seed to clear
the promise baseline's **upper** bootstrap bound — **3 of 8 windows fail** (`phase-8.md` §4 3d).

**And it is not driven by training history.** Origin 7 retrained with history cut from 44 snapshots to 18 — origin
1's size — keeps its margin: +0.0306 → +0.0303 (v6) and +0.0277 → +0.0257 (v7), bands overlapping, travelling 1% and
8% of the way to origin 1's value. The driver is the evaluation window, not how much history a model has
(`phase-9a.md` §Stage B). **So a client with several years of history has no entitlement to the +0.03 windows.**

#### The ranking claim — untestable, and why that is the right word

**SUPERSEDED: `phase-7.md` §7's description of the promise-date baseline as deployable — "a planner already holds
it" — is false on this data.**

The generator builds both quantities from the same two upstream terms:

```
contracted   = integers(15, 70) per channel, drawn once, constant over time
promise_date = po_created + contracted[channel]
lead         = lognormal(log(0.51 × contracted[channel]), 0.34) × stress
arrival_week = po_created_week + ceil(lead / 7)
```

Neither construction reads the other. But both are offsets from the same **line-level** anchor, `po_created`, and
both carry the same channel-level scale, `contracted` (`phase-9b.md` §2.1).

Three facts then settle it:

1. **The head is a channel-level forecaster.** It receives nothing about the individual PO line — not its age, not
   its contracted lead, not its promise. Measured directly: **25,569 evaluation rows across eight cells carry
   bit-identical predictions** in groups of up to 4 — the lines that share a channel within a snapshot
   (`phase-9b.md` §2.3).
2. **The promise week is line-level and orders the label at ρ = 0.75–0.78**; the head manages ρ = 0.26–0.37
   (`phase-9b.md` §2.3).
3. **The promise date does not exist at prediction time.** The line is created a median of 49 days *after* the
   snapshot (`phase-11-pilot.md` §3.2).

**So the comparison was never winnable, in either direction.** It is not that the head loses; it is that a
channel-level model cannot be ranked against a line-level quantity that is unavailable when the forecast is made, and
**no feature change can fix it** — supplying those features would be supplying the future. Guide step 11.2, which
Phase 10 called the project's highest-expected-value model change, is **withdrawn as specified** for exactly this
reason (`phase-11-pilot.md` §5; deviation 45).

**One consequence runs the other way, and it is favourable.** The head's lateness win is measured against a
**future-informed** baseline while using only as-of channel data. Beating a privileged baseline is a stronger result
than beating a fair one (`phase-11-pilot.md` §1).

#### Why lateness survives while ranking does not

Lateness is `arrival > promise`, i.e. `lead > contracted`, i.e.

```
0.51 × lognormal(0, 0.34) × [1 + 0.55·supplier_state⁺ + 0.9·transit_state + 0.85·clip(load_prev − 0.70, 0, 2) + 0.55·regime] > 1
```

The lognormal is line-level noise the head cannot see. **The bracket is not noise** — it is supplier latent state,
lane transit state, the supplier's previous-month utilisation and the regime calendar, and two of those are directly
in the head's panel (`load_ratio` *is* the supplier utilisation in the congestion term). **So the head's 8-of-8
lateness win is mechanistically expected**, not a fluke: the channel-level state that inflates a lead beyond its
contracted value is exactly what a channel-level encoder reads, while the promise date — a constant per line —
cannot express it at all (`phase-11-pilot.md` §3.4).

#### Open items

- Origins 3, 4 and 5 were never trained for arrival, so 2023 and 2024 H1 are untested (`phase-8.md` §7).
- Whether `promise_week` becomes a head input is now **closed on this data** and open only for a real extract where
  orders are raised before the forecast (`phase-11-pilot.md` §5).
- The drift gate is removed and the statistic still logged; the served distribution no longer depends on it, verified
  by a 14-value sweep (`phase-11-pilot.md` §2.1).

### 5.2 Fill rate — the best-evidenced task, and the one with a live decision

**What it predicts:** the fraction of an ordered quantity that will be received within 90 days, as a 22-cell
distribution with explicit point masses at 0 and 1.

**What ships:** `b5flat22` per `shipped.json` — **but see the provenance failure below.**

#### The headline numbers, across the full backtest

| measurement | result | windows |
|---|---|---|
| **Exact CRPS vs LightGBM-22** | head better in **15 of 16** (loses v7 o8 by 0.0005) | 16 (`phase-9a.md` §A.3) |
| **Exact CRPS vs B2 as-of histogram** | head better in **16 of 16** | 16 |
| **Marginal ECE-22 vs LightGBM-22** | **head worse in 12 of 16**, better in 2, tied 2 | 16 |
| Same, per world | v6: 1/2/5 · **v7: 1/0/7** | 16 |
| **Marginal ECE vs B2** | head better in 9, worse in 5, tied 2 | 16 |
| Recalibrated ECE-22 range across windows | **0.0111 to 0.1017 — a factor of 9** | 16 |
| **Raw validation ECE vs every baseline** | **head worse in 16 of 16**, by 2.15×–8.17× | 16 (`phase-9b.md` §3) |
| Validation CRPS advantage | head better in **16 of 16**, by 0.84%–2.60% | 16 |

#### The decision, and its basis

**Phase 7 binding statement 2 does not merely hold — it extends.** Phase 7 found LightGBM-22 better calibrated on v6
only and claimed parity for v7 (`phase-7.md` §5). Across eight windows per world, **LightGBM wins calibration in 5 of
8 on v6 and 7 of 8 on v7** — the world where the parity claim had survived (`phase-9a.md` §A.3).

Because selection must happen on validation, the decision was redone there. The recalibrated validation ECE is ≈0 for
every arm by construction (gate 7 in §4.6), so the comparison uses **raw** validation ECE, where nothing has been
fitted: **the head loses 16 of 16 windows to LightGBM-22, B5-flat-22 and even B2**, with disjoint bands, median 3.5×
(`phase-9b.md` §3).

**Validation agrees with the evaluation windows, so the evidence supports switching fill to B5-flat-22** — deployable
(no identity keys), deterministic, no seed band, and no per-window recalibration temperature to maintain. **The cost
is 0.8–2.6% of exact CRPS**, where the head wins 16 of 16 (`phase-9b.md` §3.4).

**A reversal condition is recorded in the config itself:** if the allocation optimiser becomes the consuming path,
re-open this. An optimiser consumes the whole distribution and therefore pays the proper score, where the head wins
(`phase-10.md` §3).

#### The provenance failure — the switch is not real

`shipped.json` names `model: b5flat22`, but:

- **no `b5flat22` bundle exists**,
- **`loop` has no code path that can materialise a LightGBM**, and
- **`loop.predict` silently serves the superseded neural head** (`arch=none, depth=0, lr=0.000125`).

Both worlds, verified (`phase-11-pilot.md` §2.2; deviation 46). **The Phase 10 fill switch is declarative only.**
Anyone reading the config believes fill ships a LightGBM; anyone calling the serving path receives the depth-0 head.
It must be made real or reverted before fill is served; it was reported and not repaired, because that is a decision.

#### Two further corrections worth carrying

**SUPERSEDED — the legacy CRPS formula.** Phases 2–4 measured fill CRPS with a formula that steps the CDF at 20 bin
edges and **cannot tell a mass at exactly 1.0 from one at 0.96**. On the exact integral the same old head scores
0.0752 / 0.1141 rather than 0.0595 / 0.0995 — the point masses are worth 18% / 10%. "Right on average" survived three
phases because of it (`phase-5.md` §5).

**SUPERSEDED — Addendum A's v6 parity.** Claimed in `phase-5.md` Addendum A, downgraded to "not distinguishable" in
`phase-6.md` §6, and **withdrawn** in `phase-7.md` §5: under an identical protocol every head seed is worse than every
LightGBM-22 seed on v6, by 0.0024 ECE. The "±0.003 refit noise" that justified the downgrade **does not exist** —
LightGBM is deterministic at a fixed seed, and across five different seeds its ECE moves by 0.0009 (`phase-7.md` §4).

### 5.3 Capacity strain — works, unstably

**What it predicts:** a supplier's mean utilisation over the next 90 days, clipped at 3.0, as P10/P50/P90. 12.3% (v6)
/ 29.3% (v7) of labels exceed 1.0, so "above 1 means trouble" is a live reading (`phase-5.md` §8).

**What ships:** HeteroMP h⁴ @ 2.5e-4, unchanged.

#### Depth — the finding, and what was ruled out

| split | h⁴ wins | ties | h⁰ wins |
|---|---|---|---|
| **origins 1–6** (2022 H1 → 2024 H2), 12 windows | **9** | 2 | 1 |
| **origins 7–8** (2025), 4 windows | **1** | 0 | **3** |
| **all 16** | **10** | 2 | 4 |

Source: `phase-8.md` §3.1 (recounted in code; supersedes the interim's 10/1/1 — correction 5).

**Four hypotheses were tested and all four failed to explain it:**

1. **The sign of the label shift** — h⁴ wins and loses in both rising and falling regimes (7/0/3 rising, 3/2/1
   falling) (`phase-8.md` §4 3a).
2. **The proposed anchoring mechanism** — the windows where h⁴'s median moves *against* the label shift are not the
   windows where it loses; origin 8 moves *with* the shift on every seed in both worlds and h⁴ loses both.
3. **Magnitude beyond the training range** — Spearman +0.10, −0.02 and −0.05 against the margin for three different
   extrapolation measures; no evaluation window extrapolates at all (0.85–2.36% of labels outside the training
   [p1, p99], about what an unchanged distribution gives) (`phase-8.md` §4 3a).
4. **A late-period regime** — **withdrawn.** No trend (slope +0.004 / +0.010 per year, p = 0.42 / 0.38), no late step
   (best change point is COVID, 2020-03), no ceiling; 2025's yearly mean matches 2020–2022; the only dated generator
   terms end 2022-06-30. Origin 8 looks high because the specification truncates it to **two Aug–Sep peak snapshots**,
   and its mean equals the all-years Aug–Sep average (`phase-8.md` §4 3a; deviations 24, 25).

**So the defensible statement is:** h⁴'s advantage is **window-dependent with no measured predictor**. It held in 10
of 16 half-year windows, and nothing about a window tells you in advance whether it will hold. That is quotable as a
limitation; "h⁴ is better" and "h⁴ fails when utilisation rises" are both unsupported.

#### The intervals — not quotable

| model class | 80% coverage range | windows below nominal (of 16) |
|---|---|---|
| h⁴ | 0.72–0.81 | **12** |
| h⁰ | 0.76–0.83 | 11 |
| B5 LightGBM | 0.73–0.86 | 10 |

P90 exceedance reaches 20–22% on origins 7–8 for h⁴ and 17–20% on origins 6–8 for h⁰, against a nominal 10%
(`phase-8.md` §3.2).

**The mechanism is identified.** Exceedance tracks how far the evaluation window's level sits **above the window the
intervals were fitted on** — Spearman **+0.80 (h⁴), +0.68 (h⁰), +0.82 (B5)** against the evaluation-mean z-score. From
origin 6 onward every evaluation window is above its own validation year in both worlds, which is why the
deterioration starts there (`phase-8.md` §4 3a, 8D.4).

**This is the sharpest dissociation in the project:** the window's level predicts under-coverage strongly (ρ ≈ 0.8,
every model class) and predicts the depth margin not at all (ρ ≈ 0). A fixed widening factor will not fix it; the
correction has to be level-aware (guide §11.1).

#### Open items

- The capacity fixed-split result: the shipped head is **not distinguishable** from B5 LightGBM on v6 (+0.00014,
  inside its 0.0034 spread) and beats it on v7 by 0.0032 (`phase-7.md` §6). Binding statement 3 — **SUPERSEDED**
  as a claim about the configuration (correction 4).
- Capacity v6's validation/test disagreement was real on all six seeds and twice as large as first reported
  (`phase-6.md` §6).

### 5.4 Shortage — diagnostic only, never the product path

**What it predicts:** P(any shortage in the 90-day window) for a part × plant.

**It is not the shortage model.** `model_plan.md` puts shortage on Monte Carlo simulation, which is Phase 9 and
blocked for want of an opening stock level. The supervised head is a **diagnostic** of whether the encoder carries
shortage signal, and is labelled as such in every output row (`phase-5.md` §9).

| forecaster | v6 PR-AUC | v7 PR-AUC | base rate |
|---|---|---|---|
| head HeteroMP h¹ (1 seed) | 0.6616 | 0.5071 | 0.1967 / 0.1258 |
| B5 LightGBM (5 seeds) | 0.5376 | 0.4486 | |
| naive per-part-plant rate | 0.4896 | 0.2487 | |

Source: `phase-7.md` §6.

The head has **one seed**, so its band is unmeasured and no difference is claimed (`phase-7.md` §6). Two deviations
from the Phase 0–3 closeout remain open: the learning-rate sweep optimum sits on its bottom boundary and was never
extended, and both h⁰ cells are floors capped at 119/120 epochs (`phase-5.md` §11).

### 5.5 What the graph contributes — the architectural premise

This deserves its own statement because it is the premise the project was built on.

**On the validation harness, G4 fails in both worlds.** With mean-pooled neighbour aggregates handed to the same GBM
head, the graph's share of above-chance signal is **≈0% on arrival, −1.0% on shortage, +3.6% of a small gain on
fill** (`validation7.md` §3, §11). Strengthening the propagation arc from −0.195 to −0.324 between v6 and v7 did not
move it.

**Inside the trained neural models the picture is better but inconsistent:**

| task | what depth bought | source |
|---|---|---|
| arrival | SHARE-lite h⁴ over h⁰: **+0.0127 / +0.0154** C-index (4.5× / 5.5× the wider band), confirmed on validation and test in both worlds | `phase-5.md` Addendum B |
| capacity | HeteroMP h⁴ over h⁰: validation prefers it in both worlds; test confirms on v7 (14.5× band) and **contradicts on v6** | `phase-5.md` Addendum B; `phase-6.md` §6 |
| fill | **h⁰ has the best test CRPS in every row**; graph cells' distributions collapse under input drift | `phase-5.md` Addendum B |

**The mechanism for why the graph adds so little is identified:** the arcs propagate through supplier capacity, which
every channel of that supplier shares — so the neighbourhood mean a GNN aggregates is nearly collinear with the
channel's own `load_ratio` feature, which h⁰ already has. The graph would only add signal if neighbours carried
information a channel's own features do not (`validation7.md` §12 item 3).

**And a consistent cost was measured twice:** graph encoders **rank better and extrapolate worse**. They aggregate
neighbour states, so an input shift moves every prediction together; validation and test disagreed whenever 2025
drifted, in both capacity and fill (`phase-5.md` Addendum B).

---

## 6. The optimisers

Both were built, both are verified on constructed inputs, and **both are limited by data rather than by the solver.**
The MILP solves in milliseconds; what is missing is a stock balance, a holding cost, a forward requirement and four
constraint parameters.

### 6.1 Guide 10.1 — the delivery-schedule MILP

**Built:** `ml/opt/schedule_lp.py` on `scipy.optimize.milp` (HiGHS). Integer lot quantities and a binary
order-placed indicator per week, every input loaded as-of.

**What had to be dropped or assumed** (`phase-10.md` §4.1):

| guide term | treatment |
|---|---|
| stock balance `I_w` | **DROPPED** — no opening level exists |
| `I_w ≥ safety_stock` | **DROPPED** — needs the balance |
| `c_hold · I_w`, `c_order · y_w` | **INJECTED**, default 0.0 — neither exists in the schema |
| `c_freight · ceil(x_w / truck_cap)` | **SUBSTITUTED** with per-unit freight — no truck capacity exists |
| coverage, MOQ, lot multiples, weekly cap, feasible weeks | kept, all data-backed |

**What it produces:** 200 of 200 parts optimal in each world, on a **one-week** horizon, where the answer is
arithmetic — round the requirement up to a lot multiple — and only two constraints ever bind (`phase-10.md` §4.2).

**Why it is single-period is a data fact, not a design choice.** `part_demand_weekly` carries one forward week per
part-plant at 12 as-of dates ~9.5 months apart, and `production_plan` is entirely retrospective (deviation 38).

**And with no holding cost the timing is not identified at all:** ordering everything in the first week costs exactly
what ordering everything in the last does. **Nothing in this module should be read as advice about when to order.**

**Verify gates:** the guide's own gate is vacuous (§4.6, gate 8) and was replaced by five gates, **each demonstrated
failing** against a deliberately broken solver (`phase-10.md` §4.3).

### 6.2 Guide 10.2 — allocation

**Built:** candidate enumeration, the constraint layer, and two scorers behind one interface (`ml/opt/allocation.py`).

**Scope is real:** 90.6% of part-plants have more than one qualified supplier. **But only 19.5% (v6) / 25.7% (v7)
have a recorded incumbent split**, so for four in five there is nothing to compare a recommendation against
(`phase-10.md` §5.1).

**The constraint layer — what the guide calls "the product" — is roughly half enforceable** (`phase-10.md` §5.2):

| constraint | status |
|---|---|
| **Tooling** (`is_transferable`, `duplicate_exists`) | **ENFORCED and binding** — blocks 77 (v6) / 88 (v7) candidate moves |
| **Ramp rate** | ENFORCED, but its baseline period is **assumed** |
| **Minimum-volume commitment** | **DEGENERATE** — constant 100 with no period |
| **Qualification lead time** | **INERT** — nothing in this dataset is ever unqualified |
| **Capacity ceiling** | **BLOCKED** — `revealed_capacity_est` is 100% null |

**The objective:** `SimulationScorer` implements the guide's call signature and **raises** — it is not stubbed with a
fake number. `ReducedScorer` runs in its place and measures **expected unmet demand in period, not stockout against
inventory**, so it is weaker than the guide's objective and says so wherever a number is quoted (`phase-10.md` §5.3).
Its capacity term consumes a P90 quantile and therefore **inherits intervals that are not quotable** (§5.3).

**The ranking is not a recommendation.** Sweeping the assumed shortage cost across two orders of magnitude
(₹100–₹10,000) on 120 part-plants (`phase-11-pilot.md` §4):

| | v6 | v7 | combined |
|---|---|---|---|
| winner stable across the sweep | 50 (83%) | 47 (78%) | 97 (81%) |
| margin survives the seed bands | 23 (38%) | 17 (28%) | 40 (33%) |
| **both — quotable** | **18 (30%)** | **12 (20%)** | **30 (25%)** |
| of those, "keep the incumbent" | 6 | 3 | 9 |

**So roughly 18% of part-plants yield an actionable, defensible recommendation**, and the assumed ₹1,000 sits inside
the band where winners flip. **The shortage cost is not a tuning constant** — it sets the exchange rate between unmet
demand and purchase price, which is the entire trade-off the allocator exists to make. Choosing it ourselves means
choosing the answer.

**Verify gate:** unlike 10.1's, the tooling gate binds on 41 part-plants and **fails when the check is removed** —
demonstrated in both worlds (`phase-10.md` §5.5).

---

## 7. What we learned that transfers

These are findings about method or mechanism rather than about this dataset.

### 7.1 Interval coverage fails with the level, not uniformly

P90 exceedance tracks how far the evaluation window's level sits above the window the intervals were calibrated on —
**Spearman ≈ 0.8 in every model class**, neural and gradient-boosted alike (`phase-8.md` §4 3a, 8D.4). Coverage is
close to nominal when the level is stable and collapses when it moves.

**Transfers as:** a fixed widening factor cannot fix interval coverage. The correction must be level-aware —
conformal widening keyed to recent drift, or recalibration refitted on a trailing window. And the diagnostic is
cheap: compare the evaluation window's label level to the calibration window's before trusting any interval.

### 7.2 Graph depth can be window-dependent with no available predictor

Depth-4 beat depth-0 in 10 of 16 windows and lost 4, and four separate hypotheses — shift sign, anchoring mechanism,
extrapolation magnitude, late-period regime — all failed to predict which (`phase-8.md` §4 3a).

**Transfers as:** a depth chosen on one validation year is a claim about that year. Architecture choices need several
validation windows, and where they cannot get them, the instability is the finding. A corollary measured twice here:
**graph encoders rank better and extrapolate worse**, because aggregating neighbour states moves every prediction
together when inputs shift (`phase-5.md` Addendum B).

### 7.3 The gates discipline

Ten gates that could not fail, against one assertion that could and did (§4.6). The pattern in all ten is the same:
**a check whose premise is universally satisfied, or whose thresholded quantity has noise larger than the threshold,
looks exactly like a passing check.**

**Transfers as:** before writing a check, state what would make it fire and confirm that outcome is reachable on the
data you have. Then show it failing on a constructed input. Phases 10 and 11 did this routinely — five replacement
gates for the schedule MILP, each demonstrated failing; a drift sweep demonstrated catching a gated implementation.

### 7.4 Generator forensics — reading the source to explain the scores

Several of the project's most consequential findings came from reading the generator's definitions rather than from
running another model:

- **The promise-date reframe** (§5.1) — reading `promise = po_created + contracted` and
  `lead = lognormal(log(0.51·contracted), 0.34)` explained a C-index gap four phases of modelling had treated as a
  model deficiency.
- **The capacity "regime" withdrawal** (§5.3) — the generator has no dated term after 2022-06-30, so a 2025 regime
  could not exist; the label series confirmed it.
- **The literals audit** (`validation5.md` §2.8) — grepping the source for band values is what identified runs 1–4 as
  fitted rather than generated.

**Transfers as:** when a model's score is surprising, read how the target was constructed before concluding anything
about the model. On real data the equivalent is the ETL and the business process that writes the column.

### 7.5 Check whether a comparison is winnable before concluding from it

The strongest single lesson. Phases 5, 7 and 8 all reported "the head does not out-rank the promise date" as a result
about the head. It was a result about the feature set, and then — once Phase 11 measured the label timing — a result
about a baseline that uses information from the future (§5.1).

**Transfers as:** for any baseline you lose to, ask what information it has that the model does not, and whether the
model could legitimately have it at prediction time. If it could not, the comparison measures the information gap,
not the model. This is worth asking **before** the comparison reaches a report — the signal was there in
`phase-5.md` §11, which noted that "arrival and fill predict for PO lines that do not exist at the snapshot", and its
consequence for the baseline went undrawn for six phases.

### 7.6 Two smaller ones

- **Validation cannot compare a quantity that was fitted on validation** (§4.6, gate 7). Obvious once stated, and it
  silently produced a "16 of 16 not distinguishable" result that meant nothing.
- **Artifact identity must be asserted at write time, never assumed from a naming convention** (§4.5). Two code paths
  building the same name will drift, and the second silently destroys the first.

---

## 8. What Rane must supply

*This section is written to be handed to a client directly. Each item names what is needed, why, and what it
unblocks.*

### 8.1 Blocking — nothing downstream works without these

**1. Opening stock position per part-plant, as of a stated timestamp.**
One number per part-plant: units on hand, with the as-of date and, if possible, the safety-stock target in force.

*What it unblocks:* the Phase 9 stock roll-forward and Monte Carlo shortage model; the delivery schedule's
stock-balance and safety-stock constraints; and the allocation optimiser's real objective (expected shortage rather
than expected unmet demand).

*Why we ask:* the supplied `inventory_position_weekly` is entirely zero and carries only 12 distinct dates over nine
years; `inventory_snapshots.qty_on_hand` is 97.5% negative in one world; and reconstructing the balance from
`inventory_transactions` reproduces the stated figure on 0.33% of rows. **This is the single highest-value item on
this list — it is the only one that blocks an entire phase.**

**2. A forward requirement plan.**
The MRP or planning horizon as actually run: weekly or monthly buckets, several periods ahead, with the date each
version was published.

*What it unblocks:* any multi-week delivery schedule at all.

*Why we ask:* the current extract's forecast is one week, 30 days out, refreshed roughly every 9.5 months, and the
production plan is recorded 1–8 days *after* the period it describes.

**3. A shortage or stockout cost per unit — or a service-level target to price it against.**
*What it unblocks:* any allocation recommendation. **This has moved from an assumption to blocking:** with the cost
swept across two orders of magnitude, only 25% of part-plants keep the same winning split, and the assumed ₹1,000
sits inside the band where the answer changes. Until this is supplied, **no allocation output should be shown.**

### 8.2 Cost parameters — the delivery-schedule objective is currently assumed

**4. Inventory holding / carrying cost** (per unit per week, or an annual carrying rate). No such column exists.
*Consequence today:* the schedule's **timing** is not identified — every schedule with the same number of order weeks
costs the same.

**5. Ordering / setup cost per purchase order.** Absent. Without it there is no economic order quantity, only lot
rounding.

**6. Truck capacity and freight tariff structure.** Absent, so freight is priced per unit rather than per truck.
Step-shaped freight is what makes consolidation worth anything.

### 8.3 Constraint parameters — columns exist but carry no usable content

**7. Minimum-volume commitment: over what period?** Constant 100 for all 3,000 contracts with no window. **And how is
the penalty (constant ₹10,000) applied** — per breach, per unit short, or pro-rata?

**8. Ramp-rate baseline.** `ramp_rate_pct_per_month` is real (5–39%), but a percentage *of what*, measured from which
month?

**9. Real qualification status.** Every row reads "qualified" and every channel is approved, so the PPAP lead time —
which the guide calls the heart of the allocation product — can never bind. We need the genuine mix of qualified,
in-qualification and unqualified, with target completion dates.

**10. Supplier capacity ceilings.** `revealed_capacity_est` and `evidence_strength` are 100% null. Declared capacity
per supplier-part per period, or enough delivery history to estimate it.

**11. Current sourcing splits for the remaining ~80% of part-plants.** Without the incumbent there is no baseline to
recommend against.

**12. Per-supplier-part pricing including the alternate's delta.** `cost_delta_pct` exists (−4% to +12%); confirm it
is the right basis and whether price breaks apply.

### 8.4 For the forecasts specifically

**13. Purchase-order lines that are open at the forecast instant, with their promise dates and creation dates.**
*What it unblocks:* the arrival ranking question, which is untestable on the current data because every labelled line
is created after the snapshot it is labelled at (§5.1). On a real extract, a planner forecasting an **open** order
does hold its promise date and its age — and that is the one experiment with a known-high ceiling.

**14. A second extract at a later date**, or the history to construct one.
*What it unblocks:* the only measurement that would support a production accuracy number — a rolling-origin backtest
on Rane's own data. Nothing in this project is that.

### 8.5 What we can build the moment items 1–3 arrive

The solver, the constraint layer, both verify-gate suites, the candidate enumeration and all four forecast heads are
written and tested. With a real opening balance and a real forward plan, the delivery schedule becomes a genuine
multi-period optimisation and the allocation optimiser's simulation objective replaces the reduced one without
touching the interface.

---

## 9. If this ran again

Honest retrospectives, in the order they cost the most.

### 9.1 Sequencing errors

**1. Output-format work happened before the data was validated.** The guide specifies a `model_outputs` contract down
to a per-row join on `(snapshot_id, entity_id, task)`. The table ships with **0 rows and no `task` column**, so the
verify step as written could not be run and the loop had to substitute `model_name = hades-<task>` (`preflight.md`
§A1.2; `phase-6.md` §2 row 8). More broadly, modelling phases 0–3 proceeded while validation run 7 had marked v7
**NOT USABLE** and G4 — the gate asking whether the graph contributes at all — was failing in both worlds
(`validation7.md` §1, §14).
*Do differently:* clear G4 before building an architecture whose premise it tests.

**2. Four phases of fill CRPS were measured with a point-mass-blind formula.** The legacy formula cannot distinguish
a mass at exactly 1.0 from one at 0.96, against a label that is ~91% a point mass at 1.0. "Right on average" survived
three phases because of it, and the exact figure is 18% / 10% worse (`phase-5.md` §5).
*Do differently:* derive the metric from the label's shape before measuring anything. A degenerate metric is worse
than no metric, because it produces numbers.

**3. The promise-date structural check was specified in Phase 9A and silently skipped.** It was not run and not
recorded as skipped; Phase 11 found it, ran it, and it overturned a binding statement that had stood since Phase 7
(`phase-11-pilot.md` deviation 31).
*Do differently:* a specified check that is not run is a deviation, and belongs in the deviations index the moment it
is skipped.

**4. "Phase 10" was used for four different pieces of work the guide never scoped.** Removing the drift gate,
switching fill, the capacity interval fix and the arrival feature change were all filed under "belongs to Phase 10"
across Phases 8 and 9. The guide's Phase 10 is the two optimisers; none of the four is an optimiser
(`phase-10.md` deviation 34).
*Do differently:* when deferring work, name the phase that will do it or create one. "Later" is not a schedule.

### 9.2 Method choices worth revisiting

**5. Three seeds, not the specification's five.** Every band in the project is a 3-seed range, and several verdicts
sit close to band edges (`phase-8.md` §7). Whether five seeds would move any verdict is **not measured**.

**6. One learning rate tuned on arrival and frozen across tasks** cost 14 of 34 cells their convergence in the first
grid — every fill and shortage number from that phase is a lower bound (`phase1_2.md` §4).
*Do differently:* freeze the protocol, not the value; tune per task within one protocol.

**7. The graph was carried for six phases before G4 was taken as decisive.** The measurement existed in
`validation7.md` §3 and the architecture continued as specified.
*Do differently:* let a failing premise gate the next phase, not annotate it.

### 9.3 What to keep

The bands-never-borrowed rule; validation-only selection including its uncomfortable cases; the deviations index as a
living document that later phases correct; the "gates that cannot fail" discipline; and reading the generator source
when a score is surprising. All five caught real errors that had already reached a report.

---

## 10. Appendices

### Appendix A — Full deviations index

Reproduced from `docs/implementation_guide.md` §Known deviations from spec, all 48 rows, with the phase that found
each. **Verified: this index matches the guide exactly — 48 rows, numbered 1–48, no mismatch.**

| # | found in | the specification / guide says | measured |
|---|---|---|---|
| 1 | Phase 1 | a 6-layer TCN of the given geometry trains | it does not without residual connections: C-index 0.5000 vs a ridge probe's 0.6464 |
| 2 | Phase 1 | `d_in = 38` | **24** as built (14 value + 10 indicator); 31 with the calendar block |
| 3 | Phase 1 | `fill_rate`/`ack_gap_ratio` 81.4% null, `load_ratio` 81.8% | 5.52% / 5.52% / 0.00% (v6); 5.95% / 5.95% / 0.00% (v7) |
| 4 | Phase 1 | 18 numeric columns, 13 nullable | **four columns are constant zero**, leaving 14 and 10 |
| 5 | Phase 1 | idle weeks are null, `is_active_week` mean ≈ 0.19 | idle weeks are forward-filled; mean 0.137 (v6) / 0.114 (v7) |
| 6 | Phase 2 | basis decomposition (B = 10) saves parameters | it **costs 1.67× more** than the per-relation weights it replaces, because R = 6 |
| 7 | Phase 1 | 6 node types, R = 20 relations | **4 node types, R = 6** — `supplier_group` and `po_line` are not in the graph |
| 8 | Phase 5 | the staleness gate has two inputs; the guide's `w = b = 0` init | **one input**; lag forward-filled (94.6% observed); that init receives zero gradient |
| 9 | Phase 5 | capacity label is fill capped at 1, three horizons | supplier utilisation clipped at 3.0, one horizon (90 d), censor flag is noise |
| 10 | Phase 5 | 20 fill bins including two point masses | **22 cells** — the 20-bin list is arithmetically short |
| 11 | Phase 5 | arrival 1–13 from promise, 8.4% censored | **2–12 from the snapshot, 43.6–46.2% censored** |
| 12 | Phase 6 | 115 snapshots at 28 days; covid excluded | **83 at 6-week spacing**; covid (5 of 44 training snapshots) not excluded |
| 13 | Phase 6 | bit-for-bit reproduction with deterministic algorithms | strict mode raises on MPS; identical forwards differ by 3e-8 |
| 14 | Phase 6 | a checkpoint stamped with five identifiers | the shippable unit is a **bundle** with recalibration and drift baseline |
| 15 | Phase 7 | B3 "always predict w = 1" covers 82.2% of arrivals | **week 1 never occurs**; B3 replaced by the training marginal and promise-date-only |
| 16 | Phase 7 | B5 flattens graph structure; channels per supplier median 12 | earlier LightGBMs used identity codes; median is **38** |
| 17 | Phase 7 | baselines compared directly against models | baselines go through the same recalibration protocol and the bundle scorer |
| 18 | Phase 8 | a rolling origin is a training cut plus an evaluation window | the loop needs validation: the 12 months before each cut are carved from training |
| 19 | Phase 8 | ≥ 5 seeds × 8 folds; calibration fitted on earlier folds | **3 seeds**; recalibration refitted on each origin's own validation slice |
| 20 | Phase 8 | a fold is leak-free when max(train) < min(evaluate) | 90-day outcome windows cross every boundary, in the fixed split as in the origins |
| 21 | Phase 8 | the backtest runs every fold for every task | measured **51.5 h**; run under a 6 h budget stop and two follow-up stages |
| 22 | Phase 8 | the 120-epoch cap is slack | arrival h⁰ needs 77–116 epochs; cap raised to 200 in Stage 8E; no 8E cell reached it |
| 23 | Phase 8 | the drift thresholds can be calibrated from the backtest | **0 of 48** arrival and **0 of 48** fill observations exceed 4.5 pp |
| 24 | Phase 8 | the eight origins are comparable half-year windows | origin 8 holds **2 snapshots, both at the seasonal peak** |
| 25 | Phase 8 | capacity's 2025 result reflects a late-period regime | **there is no late-period change in the label** — no trend, no step, no ceiling |
| 26 | Phase 8 | Phase 8 reports the backtest | its interim carried **four miscounts and one misreading**, recounted in code |
| 27 | Phase 9A | fill's rolling-origin backtest is deferred | **run**: origins 2–8, both worlds, 3 seeds, 42 cells |
| 28 | Phase 9A | a bundle's predictions and index entry are uniquely identified | **they were not** — a truncated cell overwrote a full-history cell's predictions |
| 29 | Phase 9A | B2 is the as-of 52-week histogram of active weeks | rows with no active week fall back to that origin's training-fold global CDF |
| 30 | Phase 9A | training history and the calendar can be separated by truncation | truncation separates **quantity** only; recency moves with the window |
| 31 | Phase 9B | Phase 9A Stage 0 ran the promise-date structural check | **it did not** — specified, never run, never recorded as skipped |
| 32 | Phase 9B | arrival's C-index measures how well a model orders arrivals | **it cannot, for a channel-level model** |
| 33 | Phase 9B | validation can compare recalibrated calibration | **it cannot** — marginal ECE is ≈0.0000 for every arm |
| 34 | Phase 9B | "belongs to Phase 10" names work inside Phase 10's scope | **it does not** — Phase 10 is the two optimisers |
| 35 | Phase 9B | a uniqueness assertion protects correct data | the first version fired on two legitimate bundles at different origins |
| 36 | Phase 10 | 10.1's inputs include holding cost, ordering cost, truck capacity | **none of the three exists** in the 49-table schema |
| 37 | Phase 10 | 10.1's verify gate discriminates | **it cannot fail** — both premises are true of every part |
| 38 | Phase 10 | a rolling plan gives a forward requirement horizon | one week per part-plant; `production_plan` is entirely retrospective |
| 39 | Phase 10 | the stock balance is a 10.1 constraint | **dropped** — no opening level exists in any usable form |
| 40 | Phase 10 | 10.2's qualification lead time is a binding constraint | **INERT** — nothing is ever unqualified |
| 41 | Phase 10 | 10.2 ranks candidates by expected shortage cost | **blocked** — `SimulationScorer` raises; `ReducedScorer` measures unmet demand |
| 42 | Phase 10 | a seed-band check protects the ranking | keyed to the P50 the term was inert and the check could not fail |
| 43 | Phase 10 | a verify gate that passes is a verify gate that works | 10.2's tooling gate first passed while blocking nothing |
| 44 | Phase 11 | arrival labels describe lines observable at the snapshot | **they do not** — 244,000 of 244,000 rows, created 7–89 days after |
| 45 | Phase 11 | guide 11.2 gives the head `promise_week` and line age | **withdrawn as specified** — neither exists at prediction time |
| 46 | Phase 11 | `shipped.json` names the model that is served | **it does not** — the fill switch is declarative only |
| 47 | Phase 11 | 10.2's candidate ranking is a recommendation | only **25%** survive both the cost sweep and the seed bands |
| 48 | Phase 11 | the promise-date baseline is deployable | **false on this data** — it uses information from the future |

### Appendix B — Timeline

Wall-clock is as recorded in each report; where a report does not state one, this says so rather than estimating.

| phase | what it asked | what it found | wall-clock |
|---|---|---|---|
| **Validation 1–4** | is this dataset usable? | no — fitted by construction; 7 outcome values set as inputs | not stated |
| **Validation 5** | can the generator be rebuilt? | yes — 171/1/4; conservation exact; all 9 probes pass | ~123 s per seed |
| **Validation 6** | make escalation a consequence; choose a cut date | worked on seed 1001, exposed a seed sensitivity on 1002 | 169.9 s per seed |
| **Validation 7** | flatten the seed cascade | cascade flattened 3.18× → 1.57×, but 6 band regressions → **v7 NOT USABLE**; **G4 fails in both worlds** | 169.4 s per seed |
| **Preflight** | inventory the repo | 3 pre-existing defects recorded; `model_outputs` 0 rows | run 2026-09-09 |
| **Phase 0** | environment, guide correction | guide phases vs task overlay reconciled | ≈ 9 min |
| **Phase 1** | data loading; guide and label fixes | 5 guide corrections; validator amendment 02; **v6 shortage labels repaired** with determinism proof | not stated separately |
| **Phase 2** | sequence assembly, baselines, G5 | as-of gate is binding (−12.2% / −14.6% on arrival) | 15,251 s = 4.24 h (36 trainings) |
| **Phase 3** | temporal encoder, cross-world | annotated: the fit window was specified and never applied | (see closeout) |
| **Phase 0–3 closeout** | inventory decision; retune fill/shortage | **Phase 9 blocked — no stock level anywhere** | 23.1 h wall / ~4.8 h compute |
| **Phase 4** | graph depth and encoder | ship h¹, SHARE-lite; basis decomposition buys nothing | not stated |
| **Phase 5** | the four task heads | hazard head +0.0118 C-index at h⁰; fill point masses; capacity's first measurement; **Addendum A** recalibration; **Addendum B** re-derived depth | ≈ 6.0 h |
| **Phase 6** | the training loop | 6 of 7 reproduction checks pass; bands measured per world; drift gate implemented | ≈ 5.4 h elapsed |
| **Phase 7** | baselines | fill parity **withdrawn** on v6; the promise-date baseline formalised | ≈ 12 min |
| **Phase 8** | rolling-origin backtest | capacity depth window-dependent; 3b closed; drift gate removed; arrival lateness 8 of 8 | 51.5 h projected, budget-stopped; Stage 8E 8.9 h |
| **Phase 9A** | fill's backtest; the arrival confound | LightGBM better calibrated in 12 of 16; margin **not** history-driven | 7.7 h + 1.0 h |
| **Phase 9B** | close two decisions; the Phase 10 gate | arrival ranking **untestable**; fill switch recommended; gate passes | not stated |
| **Phase 10** | the two optimisers | both data-blocked; 8th gate that cannot fail; Phase 11 scoped | not stated |
| **Phase 11 pilot** | smoke test, arrival pilot, sensitivity | fill switch **not honoured by the serving path**; arrival pilot **blocked by as-of**; allocation not quotable | not stated |

### Appendix C — Glossary

| term | meaning |
|---|---|
| **as-of** | Building features only from rows whose `recorded_ts` has passed at the snapshot. Not `event_ts`: a fact keyed in late was not knowable early |
| **band** | The spread (max − min) of a metric across training seeds for one configuration, in one world, on one origin. A margin inside a band is not a difference |
| **bundle** | The shippable unit: checkpoint + normaliser + validation and test predictions + recalibration + drift baseline + metrics + stamps. A directory holding only weights is not a model |
| **C-index** | Concordance index: the probability the model orders a random comparable pair correctly. 0.5 is chance |
| **CRPS** | Continuous ranked probability score — a proper score for a distributional forecast. **Exact** CRPS is point-mass-aware; the **legacy** form steps at bin edges and cannot see inside the top bin |
| **channel** | One supplier supplying one part to one plant. The unit of the weekly panel and the node type the model reads |
| **drift (label-free)** | Movement of a model's own predicted statistic between the validation inputs and new inputs, computed without any labels — so it can run in production |
| **ECE** | Expected calibration error: Σ over bins of |predicted probability − observed frequency|. **Marginal** ECE checks the overall histogram; **conditional reliability** checks whether rows given 95% actually occur 95% of the time |
| **G0–G7** | The acceptance ladder: is this a dataset (G0), does the graph exist (G1), what is the number to beat (G2), is there signal (G3), **does the graph contribute (G4)**, is the as-of gate binding (G5), are the folds sound (G6), will it survive production (G7) |
| **h⁰ / h¹ / h⁴** | Graph readout depth: no graph at all (h⁰), one hop, four hops. h⁰ is architecture-independent — the identical model whichever encoder requested it |
| **HeteroMP** | Heterogeneous message passing: learned per-relation means plus a gate. 84k–117k parameters |
| **origin** | One rolling-origin fold: a training cut, a 12-month validation slice carved from its end, and a half-year evaluation window |
| **pinball loss** | The quantile loss. Averaged over P10/P50/P90 it is this project's capacity metric |
| **PR-AUC** | Area under the precision-recall curve. The honest metric at a 13–20% base rate, where ROC-AUC flatters |
| **recalibration** | A correction fitted on the validation fold and applied unchanged to the evaluation fold — moment matching (per-cell weights) or vector scaling (one temperature + per-cell biases), selected on validation log score. Refitted every retraining |
| **RPS** | Ranked probability score: the distance-aware loss used for the fill head's 22 cells |
| **SHARE / SHARE-lite** | The relational graph encoder: basis-decomposed per-relation weights with relation-blind attention (731k parameters), and the same without the basis decomposition (469k) |
| **TCN** | Temporal convolutional network — dilated causal convolutions with residual connections; the per-channel sequence encoder |
| **v6 / v7** | The two synthetic worlds. Different demand processes, not different seeds |
| **world** | One complete generated dataset (49 tables). Two worlds × two seeds exist; all modelling uses seed 1001 |

### Appendix D — Artifact inventory

| artifact | count | stamp state |
|---|---|---|
| **Fixed-split bundles** (Phase 6) | **22** | **`…-7f04a3c+dirty`** — built before the Phase 6 code was committed. The stamp is honest; it records that the code state was uncommitted (`phase-6.md` §8). **None was regenerated**; Phase 8 verified instead that every `config.json` hash was unchanged since Phase 7 read them (`phase-8.md` Stage 0b) |
| **Backtest bundles** (Phases 8–9A) | **198** | **0 dirty.** 84 at `15a480e`, 36 at `9768e4f` (Stage 8E), 30 at `a0e50d6`, 48 at `a259af6` (Phase 9A) |
| — of which capacity | 8 origins × 2 worlds × (h⁴ 3 seeds + h⁰ 1 seed) | clean |
| — of which arrival | origins 1, 2, 6, 7 × 2 worlds × (h⁴, h⁰) × 3 seeds, plus 6 truncated-history cells | clean |
| — of which fill | 8 origins × 2 worlds × 3 seeds | clean |
| **Prediction files** | 1,560 | 600 bundle-owned, 944 baseline, 16 promise-only; **0 unclaimed** (`phase-9b.md` §4) |
| **Worlds** | 2 (`gen_v6`, `gen_v7`), seed 1001 used throughout; seed 1002 generated and validated, not modelled on | — |
| **Origins** | 8 defined and leak-asserted; capacity ran all 8, arrival 4, fill 8, shortage 0 | — |
| **Cap-bound cells** | **1** — arrival h⁰ v6 seed 7 at origin 1, 120 epochs, flagged `stop: "CAP (floor)"` in its own bundle | — |

**Why the 22 Phase 6 bundles are `+dirty` and were not rebuilt:** they predate the commit that contains their code.
Regenerating them would change every number the reproduction gate checks. Phase 8 instead measured that every
`config.json` was last written before Phase 7's scorer ran, and that every SHA-256 equalled the value read at the
start of Phase 8 (`phase-8.md` Stage 0b).

---

## Verification statement

As required before finishing, I verified the following and state the results plainly:

1. **Every number cited appears in a named report section.** Each figure in this document carries its source in the
   form `report §section`. Where a report does not record a quantity — Phase 9B's, Phase 10's and Phase 11's
   wall-clock, and whether five seeds would move any verdict — this document says **"not stated"** or **"not
   measured"** rather than estimating.

2. **Every superseded claim is marked as superseded, not silently dropped.** All seven are tabulated in §0.2 with
   both the original and the correction cited, and each is re-flagged where it appears in §5.

3. **The deviations index in Appendix A matches `docs/implementation_guide.md`** — 48 rows, numbered 1–48, no gaps,
   no additions, no renumbering, and every row's substance preserved. Checked programmatically. **One presentational
   difference, reported rather than hidden:** four rows (5, 12, 36, 44) have their "specification says" clause
   abbreviated here for table width — e.g. row 12's "115 snapshots at 28 days; regime flag available; covid excluded"
   is shortened to "115 snapshots at 28 days; covid excluded". No measured value differs. **No substantive mismatch to
   report.** (Two numbering inconsistencies existed earlier in the
   project and were reconciled in the guide before this report: `reports/phase-8.md` §6's row 25 was renumbered to 26
   to match the guide, and deviation 35 had been recorded in `reports/phase-9b.md` but never added to the guide
   index. Both are now consistent; the affected phase reports were not edited for this report.)

4. **§1 contains no claim that §5 does not support.** The capability table's eight rows each trace to §5 or §6:
   fill (§5.2), arrival lateness and ranking (§5.1), capacity median and intervals (§5.3), allocation (§6.2),
   delivery schedule (§6.1), inventory (§3.5, §6).

5. **`inventory_position_weekly` was not read** in producing this report. It was never read in any modelling phase of
   the project; the audit and both optimiser modules refuse the filename at runtime, and no file under `ml/`
   references it.

**One scope note on sources:** the brief listed `reports/validation5.md`, `validation6.md`, `validation7.md` and
`reports/phase1fixes.md`. The validation reports live in **`docs/`**, not `reports/`, and the fixes report is
`reports/phase-1-fixes.md`. All four were read at those paths. No existing report, `docs/synthetic_rules.md`,
`docs/dataset_structure.md`, `docs/model_plan.md`, `db/gen_v6/**`, `db/gen_v7/**` or `db/validator.py` was modified.
