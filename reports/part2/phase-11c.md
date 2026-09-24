# Phase 11C — close the scorer defect, re-band the evidence, adopt the metric, run Phase 9.1

**Audience:** whoever owns the client-facing claims, and whoever finishes Phase 9.
**Measured on:** `db/gen_v8/seed_1001` unless stated; v6/v7 rolling origins where Phase 8 is
re-banded. Selection on validation only.
**Companions:** `reports/part2/phase-11b.md`, `phase-11a.md`, `phase-0-1-v8.md`,
`reports/part1/phase-8.md`.

---

## 1. Verdict per stage

### Stage A — **Phase 8's arrival claim was measured against a naive constant** (form ii)

`phase7_score.py`'s `arrival_promise` branch substituted a constant for the promise arm's own
prediction before computing lateness. On all eight Phase 8 cells the recorded `promise_roc` is
that substituted value; the promise arm's own prediction scores **0.5000**. Phase 8 §5's *"the
head adds lateness information beyond the promise date — 8 of 8 windows, margin +0.0014 to
+0.0306"* is a margin over **a naive constant**, and the v6/v7 ship table's arrival row has been
quoting the wrong comparison since Phase 8. The companion claim *"beating LightGBM in all 8"*
is sound. §2.

### Stage B — **19 of 20 re-banded comparisons HOLD at five seeds**

This stage exonerates more than it weakens. **All eight capacity comparisons hold exactly.** One
verdict moves: v6 o8 fill ECE-22 head-vs-b5flat22, from *head wins* to **UNDETERMINED**. §3.

### Stage C — **the t0 metric is adopted and the privileged one is now unquotable by construction**

Reference (a) is wired in, recorded in `docs/specs/lateness_metric.md`, and
`assert_no_privileged_headline` makes the `PRIVILEGED__` prefix raise on selection *and* on
headline reporting. Arrival's claim, restated for a client, is in §4.3. §4.

### Stage D — **the machinery is sound; THE DRAWS ARE NOT CALIBRATED**

Full grid ran 3,441,620,000 cells in **5.16 min**. The gate passes and all five constructed
failures still fire at full scale. But validated against held-out 2025 outcomes the simulation
**over-projects shortage by 5.29×** — 44.36% of part-plant-weeks below safety stock against 8.38%
observed. **No quantity from this simulation is usable as a forecast yet.** §5.

### THE DEMAND CAVEAT — stated here, not buried

The consumption stream is `part_demand_weekly`'s **point forecast with uniform jitter**, because
**no demand head exists in the shipped set**. Every interval this simulation produces reflects
**supply uncertainty only**. And the jitter is itself too narrow: measured on v8, actual/planned
production has **sd 0.1265** against the jitter's **0.0865**, so the placeholder **understates
real demand dispersion by 1.46×**. Both facts belong beside every number the simulation emits.

---

## 2. Stage A — deviation 72's blast radius

### 2.1 Every call site (A.1)

| location | context | constant used |
|---|---|---|
| `phase7_score.py:111` (`kind_of`) | Phase 7 fixed-split path; applies to `promise_only` | median of `naive_global_median` (line 375) |
| `phase7_score.py:334` | **Phase 8 backtest path**, files prefixed `PROMISE_` | **1.0** (line 339) |
| `phase7_score.py:93` | the branch itself | — |

It has been in the scorer since Phase 7. Every promise-referenced lateness number in
`reports/part1/phase-8.md` §3.3 and §5, and in `phase8e_arrival.json`, comes through it.

### 2.2 What was actually scored (A.2 / A.3)

Rescored from stored predictions, no retraining:

| cell | recorded `promise_roc` | promise arm's **own** prediction | const = 1.0 | const = 9.0 |
|---|---|---|---|---|
| v6 o1 | 0.7449 | **0.5000** | 0.7449 | 0.7449 |
| v6 o2 | 0.7469 | **0.5000** | 0.7469 | 0.7469 |
| v6 o6 | 0.7439 | **0.5000** | 0.7439 | 0.7439 |
| v6 o7 | 0.7600 | **0.5000** | 0.7600 | 0.7600 |
| v7 o1 | 0.7561 | **0.5000** | 0.7561 | 0.7561 |
| v7 o2 | 0.7401 | **0.5000** | 0.7401 | 0.7401 |
| v7 o6 | 0.7384 | **0.5000** | 0.7384 | 0.7384 |
| v7 o7 | 0.7561 | **0.5000** | 0.7561 | 0.7561 |

The recorded value matches the substituted constant in every cell, **and is identical for any
constant** — which is the mechanism: with `pl = ET − AUX`, a constant `ET` ranks on `−AUX`. The
promise arm's own prediction yields one distinct lateness value across 36,000 rows, hence 0.5000.

### 2.3 The claim, restated (A.4 — form ii)

> **Phase 8 §5's arrival row compares the head against a NAIVE CONSTANT, not against the promise
> date.** The quoted margins — +0.0014 to +0.0306, median +0.02, 8 of 8 windows — are margins over
> a constant predictor. Reproduced here exactly: +0.0014 to +0.0306, median +0.0193.

A naive constant is a legitimate baseline. It is **not** the promise date, and the promise date as
an independent forecaster scores 0.5 on this metric. **The v6/v7 ship table's arrival row has been
quoting the wrong comparison since Phase 8.**

**What survives unchanged:** *"beating LightGBM in all 8"*. `b5flat_reg` is scored through
`arrival_point` on its own prediction, and the head's band clears B5's disjointly in 8 of 8 cells
(head − B5 from +0.0041 to +0.0236).

### 2.4 The fix and its test (A.5)

`ml/tests/test_no_lateness_substitution.py` fails if any arm's `roc_auc_late` differs from the
value its own prediction produces.

| | result |
|---|---|
| against the **pre-fix** scorer | **FAILS on all 8 promise cells**, deltas +0.2384 to +0.2600 |
| against the **fixed** scorer | **PASSES**, 156 arrival prediction files checked |

The branch now scores each arm's own prediction. The constant ranker is **kept** — it is a useful
baseline — but emitted as `roc_auc_late_CONSTANT_RANKER` with a note, so it can never again be
read as "the promise date".

---

## 3. Stage B — re-banding at five model seeds

### 3.1 Scope, and the asymmetry (B.1)

The brief's ten cells cost ~26 runs (~7 h) because a capacity h⁴-vs-h⁰ cell needs **both** arms at
five seeds. Against ~3.5 h remaining, **six cells × 2 seeds = 12 runs** were taken: the h⁴/head arm
widened to five, h⁰ left at three, LightGBM baselines at their five deterministic fits.
**That asymmetry is carried in every row below**, not hidden.

**Deferred, with their m/s ratios:** fill v7 o2 (0.43), v6 o1 (0.44), v6 o5 (0.46), v7 o8 (0.48),
v6 o2 (0.58), v7 o1 (0.72), v7 o6 (1.02), v7 o5 (1.35), v7 o3 (1.37), v6 o3 (1.46); capacity
v7 o7 (0.68), v6 o8 (0.93), v7 o4 (1.02), v7 o6 (1.30), v6 o5 (1.35), v6 o6 (1.94). **Arrival was
excluded by decision** — Stage A showed its count measures head-vs-constant, so re-banding would
sharpen a mislabelled comparison.

### 3.2 Per cell (B.2)

| cell | comparison | 3-seed spread | 5-seed spread | ratio | @3 | @5 | changed |
|---|---|---|---|---|---|---|---|
| v6 o7 | h⁴ vs h⁰ *(h⁰ n=3)* | 0.00377 | 0.00377 | 1.00 | h⁰ | h⁰ | no |
| v6 o7 | h⁴ vs B5 | 0.00377 | 0.00377 | 1.00 | undetermined | undetermined | no |
| v7 o3 | h⁴ vs h⁰ *(h⁰ n=3)* | 0.00328 | 0.00328 | 1.00 | undetermined | undetermined | no |
| v7 o3 | h⁴ vs B5 | 0.00328 | 0.00328 | 1.00 | undetermined | undetermined | no |
| v6 o3 | h⁴ vs h⁰ *(h⁰ n=3)* | 0.00199 | 0.00199 | 1.00 | undetermined | undetermined | no |
| v6 o3 | h⁴ vs B5 | 0.00199 | 0.00199 | 1.00 | **h⁴** | **h⁴** | no |
| v6 o4 | h⁴ vs h⁰ *(h⁰ n=3)* | 0.00210 | 0.00211 | 1.01 | **h⁴** | **h⁴** | no |
| v6 o4 | h⁴ vs B5 | 0.00210 | 0.00211 | 1.01 | **h⁴** | **h⁴** | no |
| v6 o8 | ECE-22 vs lgbm22 | 0.00144 | 0.00196 | 1.36 | undetermined | undetermined | no |
| **v6 o8** | **ECE-22 vs b5flat22** | 0.00144 | 0.00196 | 1.36 | **head** | **UNDETERMINED** | **YES** |
| v6 o8 | ECE-22 vs B2 | 0.00144 | 0.00196 | 1.36 | head | head | no |
| v6 o8 | CRPS vs all three | 0.00017 | 0.00037 | 2.25 | head | head | no |
| v6 o7 | ECE-22 vs lgbm22 | 0.00207 | 0.00269 | 1.30 | lgbm22 | lgbm22 | no |
| v6 o7 | ECE-22 vs b5flat22 | 0.00207 | 0.00269 | 1.30 | b5flat22 | b5flat22 | no |
| v6 o7 | ECE-22 vs B2 | 0.00207 | 0.00269 | 1.30 | undetermined | undetermined | no |
| v6 o7 | CRPS vs all three | 0.00002 | 0.00012 | **4.97** | head | head | no |

**20 comparisons re-banded, 1 changed, 19 held.**

Capacity spreads barely move (1.00–1.01) because only h⁴ was widened and both new seeds landed
inside the existing band. Fill spreads widen 1.30–4.97×; even the 4.97× case holds, because its
margin was large relative to the widened spread.

### 3.3 A scoring error caught before it reached this table

The first version of `phase11c_reband.py` scored the fill head **raw** against **recalibrated**
baselines: head ECE-22 0.09166 where Phase 9A recorded **0.02404**, manufacturing a 0.067 margin
and a false "verdict changed". It was exposed because the recorded at-risk margin disagreed by
three orders of magnitude. Fixed to apply each bundle's own recalibration; the numbers now
reconcile with Phase 9A exactly (v6 o8 head 3-seed [0.02318, 0.02461] → 5-seed
[0.02318, 0.02513]). **Deviation 80.**

### 3.4 Restated headline counts (B.3)

| Phase 8 / 9A count | status after re-banding |
|---|---|
| capacity **"h⁴ wins 10 of 16"** | **unmoved by this sample.** The 4 tightest of 16 cells were re-banded and all 8 comparisons held. 12 cells unexamined |
| fill **"LightGBM better in 12 of 16"** | **11 of 16 supported, 1 UNDETERMINED, 4 unexamined.** The undetermined cell is v6 o8 vs b5flat22 |
| arrival **"8 of 8"** | **superseded, not re-banded.** Stage A shows it counts head-vs-constant. It is not a promise-date result and should not be quoted as one |

Where a cell moved it is **undetermined**, not *tied*: an overlap licenses no claim of equality.

### 3.5 Proposed corrected ship table (B.4) — a PROPOSAL; `shipped.json` and `phase-8.md` untouched

| task | Phase 8 §5 row | proposed at five seeds |
|---|---|---|
| **capacity_strain** | h⁴, moderate for ranking, low for intervals | **unchanged.** The tightest cells held; the interval caveat stands and 11A added that h⁰ and B5 are better calibrated at the 80% interval on v8 |
| **arrival_week** | "adds lateness information beyond the promise date, 8 of 8" | **REWORD REQUIRED.** Beyond a *naive constant*, 8 of 8. Beyond the promise date: **not established** — the promise arm scores 0.5. Against a *t0-available* reference the claim **is** established (§4.3) |
| **fill_rate** | h⁰ recalibrated, unbacktested | **unchanged in direction**, with one fewer supporting cell: 11 of 16 rather than 12, one undetermined |
| shortage_qty | not shipped | unchanged |

---

## 4. Stage C — the adopted lateness metric

### 4.1 What was wired in (C.1)

`ml/eval/lateness_metric.py` adopts reference (a), the **as-of channel median observed lead**.
`assert_no_privileged_headline` makes the `PRIVILEGED__` prefix **load-bearing**:

| use | result |
|---|---|
| report the adopted metric as a headline | passes |
| select on the adopted metric | passes |
| **report a `PRIVILEGED__` metric as a headline** | **FIRES** |
| **select on a `PRIVILEGED__` metric** | **FIRES** |

### 4.2 The as-of assertion, all four modes (C.2)

| input | result |
|---|---|
| correctly filtered set (322,357 rows) | passes |
| unfiltered whole history | **FIRES** — 707,489 receipts recorded after t0 |
| filtered set + **one** future-dated row | **FIRES** |
| **empty set** | **FIRES** — the guard added in 11B after a falsification passed vacuously |

### 4.3 Arrival under the adopted metric, five model seeds (C.3)

| arm | seeds | ROC-AUC |
|---|---|---|
| **head h⁴** | 5 | **0.70905** [0.70645, 0.71298] |
| b5flat LightGBM | 5 | 0.70535 [0.70476, 0.70573] |
| head h⁰ | 3 | 0.69999 [0.69842, 0.70133] |
| naive constant | 1 | 0.68697 |
| reference-only | 1 | **0.50000** |

All three comparisons **disjoint**. `reference-only` at exactly 0.5 is the health check passing.

**Arrival's claim, in one sentence a client could read:**

> On v8, ranking purchase orders by how much later they will arrive than their channel's recent
> average — a reference computable from what was recorded by the forecast date — the model
> separates late from on-time arrivals with ROC-AUC 0.709, ahead of a gradient-boosted model on
> the same features (0.705) and of a no-graph variant (0.700), on five training seeds with no
> overlap between them.

**Deviation 65 stays narrowed (C.5).** Both sentences travel with any arrival lateness claim, and
are a module constant so they cannot be separated:

> Against the promise date the claim is **retired**: that reference is not available at t0.
> Against a t0-available reference it is **established** at five model seeds.

### 4.4 Recorded for the next phase (C.4)

`docs/specs/lateness_metric.md` carries the definition, the training-fitted offset (**+4.571
weeks, never refitted**), the cold-start path, the four as-of failure modes, and the measured
table. **`docs/specs/**` is on the protected list while C.4 asks for a file there; this ADDS a new
file and modifies no existing one — verified by `git status` (`??`) and an empty diff for that
directory. Deviation 81.**

---

## 5. Stage D — Phase 9.1 at full grid

### 5.1 The run (D.1)

```
4,340 part-plants x 61 snapshots x 13 weeks x N = 1,000 paths = 3,441,620,000 cells
WALL CLOCK 5.16 min        (estimate was 8.4 min)
```

**61 snapshots, not 83.** The as-of assertion fired on the 2016 snapshots: no label snapshot
exists at or before them, so the heads cannot be read as-of there. That is the assertion working,
and the 2019–2025 modelling window is the correct scope anyway.

### 5.2 The gate at full scale (D.2)

| constructed simulation | mismatched rows | result |
|---|---|---|
| correct replay | 0 | **PASS** |
| **one unit wrong on one row** | **1** | **FIRES** |
| scrap netted into receipts (0.1%) | 1,971,853 | **FIRES** |
| opening balance omitted | 1,971,853 | **FIRES** |
| safety stock used in place of the level | 1,978,220 | **FIRES** |

A gate that discriminated on 200 part-plants still discriminates on 4,340.

### 5.3 THE VALIDATION — the simulation is NOT calibrated (D.3)

Like-for-like, on the fraction of part-plant-**weeks** below safety stock, 2025:

| | value |
|---|---|
| **simulated** (path-weeks, 9 snapshots × 200 paths) | **44.36%** [33.25, 55.53] |
| **observed** (store part-plant-weeks) | **8.38%** |
| **ratio** | **5.29× OVER-projection** |

Magnitude too: simulated mean shortfall when short **564 units**, against recorded shortage events
averaging **95** (median 1, total 39,895 across 2025).

**The driver, named.** Over the 13-week horizon, arrivals replace only **85.4%** of consumption
(arrivals 1,258 vs consumption 1,474 per part-plant, against an opening level of 1,238). The
horizon drains by construction. The cause is the **order quantity**, which is a placeholder
heuristic — "order roughly the horizon's requirement" — rather than the reorder-point / lot-size /
MOQ policy the data actually carries in `part_plant`. Arrivals landing beyond week 13 are also
dropped rather than carried.

**An earlier framing of this number was not like-for-like and is corrected here**: comparing
"part-plants with *any* shortfall path" (91.09%) against recorded escalated events (8.66%) gave
10.5×, but with 13,000 path-weeks per part-plant almost anything trips "any", and recorded events
are a strict subset of shortage conditions by design (`synthetic_rules.md` §9). **The path-week
fraction is the comparable quantity and 5.29× is the number that stands.**

**Consequence, stated plainly: no quantity from this simulation is usable as a forecast.** The
machinery, the gate and the as-of discipline are sound; the draws are not.

### 5.4 The demand caveat, quantified (D.4)

| | |
|---|---|
| actual/planned production, v8, 65,700 product-plant-weeks | mean **0.9967**, sd **0.1265**, CV 0.127 |
| p05 / p95 | 0.8033 / 1.2181 |
| the simulation's `uniform(0.85, 1.15)` jitter | sd **0.0865** |
| **understatement factor** | **1.46×** |

So the placeholder is not merely *a* placeholder — it is **narrower than observed plan error by
1.46× in sd**. Widening it to match would widen every projected interval by roughly that factor,
and would still capture only the dispersion of the *plan*, not genuine demand uncertainty, because
**no demand head exists**. Every interval this simulation produces reflects **supply uncertainty
only**. This is carried in the module docstring and in §1.

### 5.5 The orphan part-plants (D.5)

**113 part-plants have no channel and are CARRIED, not dropped** — they contribute **0.0133%** of
total projected shortfall. An earlier subset filter (`n_channels > 0`) excluded them; that was
found and fixed before the full run, because dropping them would remove their shortage from the
totals, which is the opposite of conservative. **Deviation 82.**

### 5.6 The 9.2 decision (D.6)

**Recommendation: DEFER the copula.**

| option | cost | what it buys |
|---|---|---|
| **build it, gate the level only** | ~1 day: ρ estimation from pre-t0 co-occurrence, sampling, G10 wiring | correlated sampling that reproduces the correlation **level**. It **cannot** be validated as recovering group structure on v8 |
| **defer until a world separates group structure** | 0 now | nothing lost that can be demonstrated on v8 |

The reasoning: on v8 within-group correlation is **+0.2101** and cross-group is **+0.2055** — a
separation of **+0.0046**, an order of magnitude inside G10's own ±0.05 tolerance. A per-group ρ
and a single global ρ are **indistinguishable** here. So the copula's central claim — that
correlated failure within a supplier group matters — is untestable on this world, and building it
would produce a component whose correctness cannot be demonstrated.

**And the prior question is now more urgent than the copula.** Phase 9.1's own draws over-project
by 5.29×. Correlated sampling on top of mis-calibrated marginals would make the output worse, not
better. **Fix the order policy and re-validate 9.1 before any copula work.**

---

## 6. Deviations and open items

### 6.1 Deviations

| # | Prior statement | Measured | Where |
|---|---|---|---|
| **79** | Phase 8 §5: "the head adds lateness information beyond the promise date — 8 of 8 windows, margin +0.0014 to +0.0306" | **the comparison is against a NAIVE CONSTANT.** The scorer substituted a constant for the promise arm on all 8 cells; the promise arm's own prediction scores 0.5000, identically for any constant. The margins are reproduced exactly but are margins over a constant. The v6/v7 ship table's arrival row has quoted the wrong comparison since Phase 8. *"Beating LightGBM in all 8" is sound* | §2 |
| **80** | a re-banding scorer can compare stored predictions directly | **not for fill**: Phase 9A's head figure is the RECALIBRATED ECE. Scoring raw against recalibrated baselines gave head ECE-22 0.09166 where 0.02404 was recorded, and a false "verdict changed". Caught because the recorded at-risk margin disagreed by three orders of magnitude | §3.3 |
| **81** | `docs/specs/**` is read-only while C.4 requires a file there | resolved by **adding** `docs/specs/lateness_metric.md` and modifying no existing file — verified `??` status and an empty diff for that directory | §4.4 |
| **82** | 11B §5.2: the 113 orphan part-plants are carried with zero arrivals | **the subset run carried them; the first full-grid run did not.** A `n_channels > 0` filter excluded them. Fixed before the reported run; they contribute 0.0133% of projected shortfall | §5.5 |
| **83** | Phase 9.1's roll-forward, gated and running, is a result | **it is a mechanism, not a result.** Validated against held-out 2025 outcomes it over-projects shortage by **5.29×** (44.36% of part-plant-weeks vs 8.38% observed), driven by a placeholder order quantity that replaces only 85.4% of consumption over the horizon | §5.3 |
| **84** | the simulation's demand jitter is a neutral placeholder | **it is narrower than reality.** Actual/planned production on v8 has sd 0.1265 against the jitter's 0.0865 — the placeholder **understates** demand dispersion by **1.46×**, on top of representing supply uncertainty only | §5.4 |

### 6.2 Open items

1. **Fix Phase 9.1's order policy and re-validate.** The 5.29× over-projection is the single thing
   blocking any quantity from this simulation. `part_plant` carries `reorder_point_qty`,
   `min_order_qty` and `lot_size`; the current heuristic uses none of them. Arrivals beyond the
   13-week horizon should be carried rather than dropped.
2. **A demand head does not exist.** Until one does, every interval is supply-only, and the
   consumption jitter should at minimum be widened to the measured sd 0.1265 (deviation 84).
3. **Phase 8's ship table needs the arrival row reworded** (deviation 79). The proposal is in §3.4;
   `shipped.json` and `phase-8.md` are untouched and this phase makes no edit to either.
4. **16 at-risk cells remain unexamined** (§3.1 lists them with m/s ratios). The 6 re-banded held
   19 of 20, which is evidence the remainder will largely hold — but it is evidence, not a
   measurement.
5. **h⁰ arms are still at three seeds** wherever a capacity comparison was decided against them.
   Four of the eight capacity comparisons in §3.2 are h⁴-vs-h⁰ and carry that asymmetry.
6. **The copula is deferred**, not cancelled (§5.6). Revisit when either 9.1 is calibrated or a
   world exists whose supplier groups are separable.
7. **Phase 7's fixed-split `arrival_promise` path** used a different constant (the median of
   `naive_global_median`) than the backtest's 1.0. Both are now fixed, but any Phase 7 figure
   quoting `promise_only` lateness carries the same relabelling as deviation 79.

---

## 7. Compliance

- **`ml/configs/shipped.json` is unchanged.** Stage A invalidated a ship-table claim and Stage B
  moved a verdict; neither was allowed to edit the config, and §3.4 is a **proposal**.
- **`reports/part1/phase-8.md` is unchanged.** Its correction is recorded here, per the standing
  rule that a prior report is never edited.
- **No assertion was disabled or weakened.** One was **added** (`assert_no_privileged_headline`)
  and one **restored to honesty** (the lateness branch now scores each arm's own prediction, with
  a test that fails against the old behaviour). The as-of assertion fired correctly during the
  full-grid run and was obeyed, not bypassed.
- **Every gate demonstrated failing:** the substitution test (8 cells pre-fix), the privileged
  guard (both uses), the as-of receipt assertion (three failure modes), 9.1's identity gate (five
  constructed failures at full scale).
- **Five model seeds** for everything newly quoted; where an arm is at three, the row says so.
- **`inventory_position_weekly` was read** in `ml/sim/montecarlo.py`'s `opening_position` (the
  simulation's opening balance) and `run_gate` (9.1's acceptance reference), and in §5.3's
  observed-condition comparison. It is no longer forbidden on v8 — B1 reconciled it at
  100.000000% — and it is read only on those paths, never as a model feature.
- **Phase 9.2's copula was not started.**
- **Commit before running**; nothing stamped `+dirty`.
