# Phase 11A — graph control, baselines, seeds, and the Phase 9 gate

**Audience:** whoever decides what ships off v8, and whoever builds Phase 9.
**Measured on:** `db/gen_v8/seed_1001` (dataset seed fixed throughout), fixed split — train ≤ 2023,
validation 2024, test 2025. Selection on validation only.
**Companions:** `reports/part2/phase-0-1-v8.md`, `reports/part2/v8-clearance.md`.
**Path note:** the brief names `reports/phase-11a.md`; `reports/` was reorganised into `part1/`
(v6–v7 era) and `part2/` (v8 era) before this phase, so this sits with the other v8 reports.

---

## 1. Verdict per stage

### Stage 1 — the shuffled-graph control: **THE EDGES CARRY INFORMATION** (verdict i)

On `capacity_strain`, with a degree-preserving permuted neighbourhood, **the real graph beats the
shuffled graph outside the seed bands on both folds.** 71.7% of the h4-over-h0 advantage is
attributable to the edges; 28.3% to depth and parameters alone, and on the test fold that
remainder is not distinguishable from zero.

**This pipeline does not corroborate v8's own G4.** That is stated without softening, and so is
its limit: the instruments differ, so this is the answer *on this pipeline*, not a refutation of
that measurement. §2.4.

### Stage 2 — baselines: **the head loses on two of four tasks, and those losses come first**

- **arrival lateness: NOT DISTINGUISHABLE from a naive constant predictor.** The head's 3-seed
  band straddles it. The lateness advantage is not established on v8.
- **fill calibration: the head loses ECE-22 to every LightGBM arm and to a naive as-of
  histogram**, the latter by 3.4×. v8 reproduces the v6/v7 pattern the brief asked about, more
  starkly.
- **capacity intervals: the head loses 80% coverage** to `naive_global` and to B5.
- The head wins arrival C-index, fill exact CRPS, capacity pinball, and shortage outright. §3.

### Stage 3 — five model seeds *(running at the time of writing; §4)*

### Stage 4 — deviation 59 reproduction *(running at the time of writing; §5)*

### Stage 5 — the Phase 9 gates: **both specified and demonstrated failing; neither existed before**

`db/ground_truth.py` **never existed in this repo**. A replacement G10 is built against the CSVs
and fires in both directions; 9.1's identity gate is now discriminating on v8 and is shown to be
**vacuous on v6**, where it passes a simulation that omits the opening balance entirely. §6.

---

## 2. The shuffle control

### 2.1 What was permuted

`W["rel"][j]` maps each channel to its entity under relation *j*. The control applies a
permutation to the **channel axis**, `rel_shuffled[j][c] = rel[j][pi_j(c)]`, which leaves the
multiset of entity assignments untouched — so **every entity keeps exactly the degree it had** —
while changing which channels sit in its neighbourhood. **Each relation gets its own
permutation**: sharing one would move a channel's whole (supplier, part, plant) triple together
and preserve the co-occurrence structure the control exists to destroy.

Not shuffled: `pp_of_chan` / `pp_uniq`, which are entity binding for the shortage readout, not
graph structure. Shuffling them would change which label a row is scored against.

The shuffled arm is a **distinct configuration**: `graph_shuffle` enters
`ml/artifact_identity.py`'s `config_name` and `identity_of`, so it cannot share a bundle,
prediction file or index key with the real arm (standing rule 2). `device_inputs` keys its cache
on it, and **both** evaluation paths pass it through — scoring a shuffled bundle against the real
graph would silently compare a model to inputs it never saw.

### 2.2 Falsifying the control before trusting it (Stage 1.2)

| relation | edges | entities | **endpoint changed** | chance-level unchanged | degree distribution | degree min/median/max |
|---|---|---|---|---|---|---|
| supplier | 16,072 | 420 | **99.72–99.80%** | 0.244% | **identical** | 21 / 38 / 55 |
| part | 16,072 | 620 | **99.80–99.88%** | 0.167% | **identical** | 12 / 26 / 45 |
| plant | 16,072 | 7 | **85.68–86.01%** | 14.293% | **identical** | 2,227 / 2,285 / 2,362 |
| **total** | 48,216 | — | **95.08–95.19%** | — | **identical** | — |

The plant relation's 86% is exactly chance for 7 plants (100 − 14.29), which is what a proper
random permutation should give.

**The falsification itself can fail**, demonstrated on three degenerate shuffles:

| mutation | result |
|---|---|
| identity shuffle (nothing moves) | **FIRES** — "only 0.0% of edges changed endpoint" |
| 90% of edges preserved | **FIRES** — degree distribution changed |
| all edges to one entity | **FIRES** — degree distribution changed |

Cross-contamination was checked explicitly: after building the shuffled world, the cached
real-graph world is byte-unchanged.

### 2.3 The three arms — `capacity_strain`, 3 model seeds

**Validation pinball (the selection surface; lower is better):**

| arm | mean | band | seeds |
|---|---|---|---|
| **real graph** | **0.06961** | [0.06879, 0.07035] | 0.06879 / 0.06970 / 0.07035 |
| **shuffled graph** | 0.07352 | [0.07306, 0.07400] | 0.07306 / 0.07350 / 0.07400 |
| **h0** (no graph) | 0.07506 | [0.07475, 0.07544] | 0.07475 / 0.07499 / 0.07544 |

**All three pairwise DISJOINT.**

**Gap decomposition — the h4-over-h0 advantage is 0.00545 pinball:**

| source | pinball | share |
|---|---|---|
| **the EDGES** (real − shuffled) | **0.00391** | **71.7%** |
| depth and parameters alone (shuffled − h0) | 0.00154 | 28.3% |

**Test fold (reported; not the selection surface):**

| arm | pinball | 80% coverage | MAE P50 |
|---|---|---|---|
| real | **0.07248** [0.07213, 0.07279] | 0.78391 | 0.22800 |
| shuffled | 0.07659 [0.07599, 0.07692] | 0.78221 | 0.24272 |
| h0 | 0.07671 [0.07652, 0.07694] | 0.80234 | 0.24373 |

real vs shuffled **disjoint**; **shuffled vs h0 NOT disjoint**. On the evaluation fold a permuted
neighbourhood buys nothing over no neighbourhood at all, so there essentially **the whole
advantage is the edges** and the depth/parameter share falls to nothing measurable.

### 2.4 Against v8's own G4 — stated plainly, and bounded

`docs/v8/v8_report.md` §3.1 reports the graph contributing nothing, with a shuffled control that
scored the same as the real graph on all five dataset seeds. **This pipeline gets the opposite
answer.** Both measurements are real; they are not the same measurement:

| | v8's G4 | this measurement |
|---|---|---|
| model | `validator_v8.py`'s own | this repo's TCN + HeteroMP, shipped config |
| metric | MAE | pinball (quantile head) |
| split | validator's | fixed split 2023 / 2024 / 2025 |
| seeds varied | **dataset** seeds 1001–1005 | **model** seeds 7/17/27, dataset seed 1001 fixed |
| shuffled control | yes | yes |

**What is defensible:** on this pipeline, with the shipped configuration, the real neighbourhood
beats a degree-preserving shuffle on both folds, disjointly, and the shuffle is sound. **What is
not defensible from this alone:** that v8's G4 is wrong. The axes differ — most importantly G4
varied the dataset seed while this varies the model seed, so a dataset-seed effect would not show
here. **Deviation 64**, and open item 1: running this control across v8's five dataset seeds is
the measurement that would reconcile them.

### 2.5 Stage 1.5 — what this says about Phase 8 §4 3a

Phase 8 found h4's capacity advantage window-dependent across 16 windows, with five mechanism
tests all failing to explain it. "The edges carry nothing and depth does the work" was an
untested sixth candidate.

**Verdict (i) rules that candidate out on v8**: depth and parameters alone account for 28.3% of
the advantage on validation and nothing distinguishable on test. Had the verdict been (ii), it
would have explained Phase 8's window-dependence neatly — a depth effect with no structural
content is exactly the kind of thing that would drift with the window.

**It does not settle Phase 8's finding**, which was measured on v6/v7. Those are different worlds
and their bands are not borrowed. What it does is remove the sixth candidate *on v8* and leave
Phase 8's question open on its own world.

---

## 3. Baselines on v8

54 baseline configurations, fitted on v8's fixed split and scored under **the head's own
recalibration protocol** (`loop.fit_recalibration`, fitted on validation, never carried).
**Row identity asserted on all 324 prediction files.** LightGBM is deterministic at a fixed seed,
so its five fits differ only through the sampler seed and the spread is near zero — that is the
expected result, not a defect.

**Disjoint bands are required for any claim of difference.** Overlapping bands are reported as
NOT DISTINGUISHABLE, which is itself a finding.

### 3.1 Where the head LOSES — first, as instructed

#### arrival lateness: not distinguishable from a naive constant

| arm | ROC-AUC (lateness) | band |
|---|---|---|
| **head h4** | 0.75304 | [0.74990, 0.75626] |
| **naive_global_median** | **0.75181** | — |
| `PRIVILEGED__promise_only` | **0.75181** | — |
| b5flat LightGBM | 0.74594 | [0.74547, 0.74624] |

**The naive constant and the promise-only baseline score identically — 0.75181 — and that is not
a coincidence.** Lateness ROC-AUC ranks `prediction − promise`; with a *constant* prediction that
is a monotone function of the promise, so ranking by it **is** ranking by the promise date. The
head's band straddles that value.

**On v8's fixed split the arrival head's lateness advantage over the promise date is not
established.** Phase 8 §5 recorded it holding in 8 of 8 windows on v6/v7; that is a different
world and is not carried here. **Deviation 65.**

A caveat that cuts both ways and is already on the record
(`reports/part2/phase-0-1-v8.md` §3.1): the lateness metric is *defined* against the promise
date, so every arm in this table — the head included — is scored against privileged information.

#### fill calibration: the head loses to everything, including a naive histogram

| arm | ECE-22 (lower better) | band | vs head |
|---|---|---|---|
| **b2_rolling52_cdf** (naive as-of histogram) | **0.01651** | — | **beats head 3.4×** |
| recal_b5flat22 | 0.03139 | [0.03107, 0.03169] | beats head |
| recal_lgbm22_id | 0.03226 | [0.03175, 0.03259] | beats head |
| b5flat22 | 0.05179 | [0.05102, 0.05270] | beats head |
| lgbm22_id | 0.05237 | [0.05151, 0.05315] | beats head |
| **head h0, recalibrated** | **0.05583** | [0.05417, 0.05759] | — |
| naive_channel_cdf | 0.07273 | — | head wins |

All five losses are **disjoint**. **Stage 2.3's question answered: yes, v8 reproduces the v6/v7
pattern, and more starkly.** There the neural head lost marginal calibration to LightGBM in 12 of
16 windows; here it loses to *every* LightGBM arm and to a rolling-52 as-of histogram.

#### capacity intervals: the head loses coverage

| arm | 80% coverage (nominal 0.80) | vs head |
|---|---|---|
| naive_global | 0.82849 | beats head |
| b5flat_q | 0.80344 | beats head |
| naive_supplier | 0.78791 | NOT DISTINGUISHABLE |
| **head h4** | **0.78391** | — |
| naive_channel | 0.61191 | head wins |

### 3.2 Where the head wins

| task | metric | head | strongest baseline | disjoint |
|---|---|---|---|---|
| arrival | C-index | **0.67438** | 0.64891 (b5flat_reg) | yes |
| fill | **exact CRPS** (point-mass-aware) | **0.13827** | 0.13942 (recal_lgbm22_id) | yes |
| capacity | pinball | **0.07248** | 0.07990 (b5flat_q) | yes |
| shortage | ROC-AUC | **0.81587** | 0.77257 (b5flat_bin) | yes |
| shortage | PR-AUC | **0.65875** | 0.58889 (b5flat_bin) | yes |

`naive_global_rate` scores ROC-AUC exactly 0.50000, which is the sanity check passing.

### 3.3 The shipped head against the strongest DEPLOYABLE baseline, per task

| task | verdict |
|---|---|
| **arrival** | **Wins on ranking** (C-index, disjoint). **Does NOT win on lateness** — not distinguishable from a constant predictor, i.e. from the promise date's own ordering |
| **fill** | **Wins the proper score, loses the marginal calibration.** Exact CRPS disjointly better; ECE-22 disjointly worse than four LightGBM arms and a naive histogram |
| **capacity** | **Wins point accuracy and pinball, loses interval coverage.** B5 quantile LightGBM is better calibrated at the 80% interval |
| **shortage** | **Wins outright** on both ranking metrics. Diagnostic only; not shipped, and no probability is quotable (§3.4) |

### 3.4 Shortage's level caveat, carried with the number

v8's shortage positive rate on the test fold is **25.70%** against a real operating base rate near
3% (`synthetic_rules.md` §9), and v8's event rates run 4–6× over Rane's stated bands (deviation
56). **Any shortage probability from v8 is mis-calibrated in level by construction.** ROC-AUC and
PR-AUC are ranking metrics and are unaffected, which is why they are the numbers reported. **No
shortage probability is quoted anywhere in this report.**

---

## 4. Five model seeds

*Running at the time of writing.*

**These are MODEL seeds (7/17/27/37/47), not dataset seeds.** v8 ships five dataset seeds
(1001–1005) and that is a different axis; **dataset seed 1001 is held fixed throughout this
phase**, including in Stage 1's control.

---

## 5. Deviation 59 — the reproduction check

*Running at the time of writing.*

---

## 6. Phase 9's gates

### 6.1 `db/ground_truth.py` never existed (Stage 5.1)

Searched the working tree, all refs, and every blob in history. **`db/ground_truth.py` and
`db/prove_claims.py` have never existed in this repository.** What does exist in history is a
pair of modules from a different project:

| file | commits | era |
|---|---|---|
| `ml/counterfactual_ground_truth.py` | d0c3866, 087d056, f81a216 | Aug 2026 |
| `ml/graph/hidden_dependency_ground_truth.py` | same | Aug 2026 |

Both are built on `db/generate_dataset.py` — a generator that **is itself gone** — and both belong
to commits labelled "Failiure" / "Greatest Stepping Stone". The v4 reboot (`ab4ad00`,
2026-09-09) introduced `gen_v6`, `gen_v7`, `schema.sql` and `dataset_structure.md` fresh, and did
not port them. The guide's G10 text arrived in that same commit, referencing a module from the
superseded era.

**G10 has been uncallable for the entire life of this codebase.** Deviation 63 stands, with its
cause now established.

### 6.2 The replacement G10, and what it cannot do (Stage 5.2)

The reference moves from generator internals to **the CSVs**, which is what a modelling team
actually sees:

```
reference = realised within-supplier-group correlation of OBSERVED weekly shortfalls,
            from channel_performance_weekly over a window before t0
gate      = | simulated within-group correlation - reference | <= 0.05
```

**Demonstrated firing in both directions**, and passing where it should:

| constructed simulation | simulated r | \|Δ\| | result |
|---|---|---|---|
| independent sampling (no shared factor) | −0.0019 | 0.2121 | **FIRES** |
| one global common factor, too strong | +0.5074 | 0.2972 | **FIRES** |
| week-permuted real data | +0.2101 | 0.0000 | PASS |

**The limit, reported and deliberately NOT gated.** On v8 (2019–2025, 420 suppliers × 365 weeks):

| | mean r |
|---|---|
| within-group pairs (840) | **+0.2101** |
| cross-group pairs (87,150) | **+0.2055** |
| **separation** | **+0.0046** |

Per-group means span +0.1455 to +0.2729 across all 84 groups. **The separation is an order of
magnitude inside the gate's own ±0.05 tolerance.** Every supplier shares a common factor near
r = 0.21 and `supplier_group_id` adds essentially nothing on top of it.

So G10 tests that a simulation reproduces the **correlation level**. It **cannot** test that a
copula recovered **group structure**, because a per-group ρ and a single global ρ are
indistinguishable on this world. Gating on the separation would be a check that cannot fail —
instance 11 avoided. This corroborates the guide's own §9.2 warning ("no result from this dataset
supports the independence-understates-exposure claim") from an independent direction.

### 6.3 9.1's own gate is now discriminating — and was vacuous before (Stage 5.3)

The gate: *with fill fixed at 1.0 and timing at the promise date, the simulation reproduces
`inventory_position_weekly.qty_available` exactly.*

**On v8** (`qty_available` mean 1,029.4, max 22,045, populated — B1 cleared it):

| constructed simulation | mismatched rows | result |
|---|---|---|
| perfect | 0 | PASS |
| **one unit wrong on one row** | **1** | **FIRES** |
| scrap netted into receipts (0.1% shift) | 2,245,648 | **FIRES** |
| opening balance omitted | 2,245,648 | **FIRES** |

**On v6**, where `qty_available` is entirely zero, the *same* gate:

| constructed simulation | mismatched rows | result |
|---|---|---|
| perfect (also zero) | 0 | PASS |
| **opening balance omitted entirely** | **0** | **PASS** ← vacuous |

That is instance 3 on this project's record, shown rather than asserted: on v6/v7 the gate would
have accepted a simulation that discarded the entire opening balance. **On v8 it catches a single
unit in 2.25 million rows.**

### 6.4 Revised Phase 9.1 estimate — NOT started

The grid on v8: 4,212 part-plants × 83 snapshots × 13-week horizon × N = 1,000 paths =
**4,544,748,000 path-weeks**, 219 MB float32 of state per snapshot.

| component | estimate | basis |
|---|---|---|
| **Implementation** — `ml/sim/montecarlo.py`, `ml/sim/copula.py` | **the dominant cost**; neither file exists | `ml/sim/` contained only the gates written in this phase |
| roll-forward **compute** | **~0.2 min per seed** (0.14 s/snapshot) | timed at full scale on this machine |
| **one inference sweep** (draw f, T, C per path) | hours, not minutes | Phase 1's inference cost repeated per snapshot over the full channel population, not the 36,000-row test fold |
| part-plant ↔ channel mapping + BOM explosion | ~1 h to build, then cheap | single-level BOM (`bom_level` constant 1), 4,340 part-plants |
| ρ estimation for the copula | ~1 h | 84 groups, estimated from data before t0 — never from generator parameters |
| **G10 acceptance** | **now runnable** (§6.2) | was uncallable |
| **9.1 acceptance** | **now discriminating** (§6.3) | was vacuous on v6/v7 |

**The blocker was never compute.** It was the opening balance, which B1 delivered, and the two
gates, which this phase supplies. Budget Phase 9.1 as implementation plus one inference sweep.

---

## 7. Deviations and open items

*Completed after Stages 3 and 4 land.*
