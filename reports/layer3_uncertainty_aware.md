# Layer 3 — Calibration, Confidence, and Two Sanity Checks

**Date:** 2026-08-14
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisites read before starting:** `reports/phase1_latent_state.md` (Model A, the
recoverability record) and `reports/phase_modelB_concat_baseline.md` (this session's audit
conclusion: Model C is gated and the gate did not open).

**New code, three files, no inherited file modified:** `ml/layer3_sufficiency.py` (Step 1),
`ml/identifiability_check.py` (Step 2A), `ml/layer3_uncertainty.py` (Steps 3–5).

**Scale caveat carried from every prior phase:** everything below is the `v1` preset (800
suppliers, 15 snapshots), five dataset seeds 42–46. Per `docs/14_Project_Roadmap.md` §3.5 this
is a **pilot, not a benchmark finding**. Spec-scale `db/csv/` holds only seeds 42–43, so the
five-seed minimum is not satisfiable there without regeneration.

---

## Verdict, up front

| State | Step 1 | Step 2 case | Step 3 | Step 4 | Step 5 | Final |
|---|---|---|---|---|---|---|
| **Supply Stress** (A) | **PASS** | **Case 1** | **STOP — G** | (STOP — G) | no threshold | ❌ excluded at Step 3 |
| **Supply Stress** (E) | **PASS** | **Case 1** | **STOP — G** | (STOP — G) | no threshold | ❌ excluded at Step 3 |
| **Recovery Capability** (E) | **STOP — E** | Case 1 | not run | not run | not run | ❌ excluded at Step 1 |
| **Supplier Reliability** (A, E) | n/a — no head | **Case 3** | closed | closed | closed | ❌ closed at Step 2, C |

**No state ended this prompt with a calibrated, confidence-aware, threshold-gated output.**
Zero states reached Step 5 with a frozen threshold. The ladder is reported step by step below
with the taxonomy classification for each STOP, and — because a bare "it failed" is not a
finding — with the measurement that distinguishes each failure from its neighbours.

The one genuinely new positive result is Step 2's: **Supply Stress and Recovery Capability are
identifiable** under a real do(Z) intervention, and **Supplier Reliability is not**, which
closes its open failure category as C rather than A/B/D.

---

## Scope notes and two documentation defects found on the way in

Stated here rather than buried, because both concern artefacts this prompt instructed the work
to reuse.

1. **`docs/decision_support_build.md` §2.3 does not exist in this repository.** `docs/` holds
   exactly two files (`14_Project_Roadmap.md`, `HADES-v3_init.md`). The live implementation of
   that protocol is `ml/uncertainty_calibrate.py`, whose own docstring cites
   `reports/layer3_testing.md` §10.3 — also absent here, an inherited V2 reference. **Step 3
   therefore matches the code, not the document**: `evaluate_fold`'s exact rotation (fit the
   isotonic map on one world, evaluate on a different one, rotate over all ordered seed pairs)
   and its exact fit-floor / member-floor pair.

2. **`ml/test_hypothesis_calibration.py` does not exist in this repository.** Phase 0's port
   removed it as part of the fifteen scrapped V2 Layer-3 modules
   (`reports/phase0_audit.md`, "Repository cleanup"). The prompt cites it as what validates the
   PAVA/ECE machinery, so the gap was closed rather than noted and skipped:
   `ml/hypothesis_ranker.py` is byte-identical to V2's, and V2's suite was executed against
   V3's module in V3's environment. **All 12 checks pass**, including the two that matter here
   — ECE 0.00163 on a predictor calibrated by construction, and ECE 0.5996 (expected 0.600) on
   a 0.9-always predictor of a 30% class.

   ```
   $ PYTHONPATH=. venv/bin/python <HADES_v2>/ml/test_hypothesis_calibration.py
   [PASS] roc_auc matches sklearn (0.682234 vs 0.682234)
   ... 12 checks ...
   ALL CHECKS PASSED
   ```

3. **The 5×5 grid's second axis is the head init seed, not the backbone seed.** Only `m0`
   backbone checkpoints exist in `out/ds_ckpt/`; building `m1..m4` would mean *training SHARE*,
   which the ground rules forbid. The head init seed is also the axis Phase 1 measures its
   reproduction floor along, so the grid sits on the same axis as every prior number. Flagged
   because it means Step 4's confidence signal estimates the *smaller* of the two variance
   sources — precisely the caveat `ml/uncertainty_ensemble.py`'s docstring raises — and that
   turns out to matter for how Step 4's STOP is read.

---

## STEP 1 — Downstream usefulness / sufficiency

**What this step does not establish, restated because the result is positive for one state.**
Nothing here is evidence that Z is a correct or faithful representation of the true latent
state. A downstream-informative Z can be a confounded proxy or a partial shadow of something
else. Correctness is Step 2's question. This section's PASS must not be read as evidence of
identifiability or causal correctness — and, as it happens, Step 2 goes on to show that Model
A's recoverability rests partly on upstream correlates that a do(Z) severs.

**Design.** `Y` is `impact`, the only one of the three benchmark tasks whose label sits on the
**Supplier** entity (`ml/models/depth.py::TASK_ENTITY_TYPE`) and therefore the only one that
can be aligned row-for-row with a per-supplier latent state. `X` is the raw observable supplier
feature vector `data["Supplier"].x` — the same 14 numbers h⁰ projects. `Z` is Model A's head
output at the state's Phase-1 best depth.

Three-way split so no arm is scored on its own training data: the Z head is fitted on `tr`
(40%), the downstream head on `va` (20%) using Z values that are **out of sample** for it, and
everything is scored on `te` (40%). Training the downstream head on `tr` would feed it
in-sample Z, whose error distribution is not the one it meets at test time — a leak in the only
direction that flatters this step.

**Model A's head is reused, not reimplemented, and that is checked numerically.**
`fit_probs()` is `train_head()` with the return value changed from a scalar AUC to
probabilities and nothing else touched. `--verify-model-a` re-derives Phase 1's per-init-seed
Z-head AUC from those probabilities and diffs it against `out/phase1/heads_{A,E}.json`:

```
Model A equivalence: 25 checks, 0 mismatches      (variant A)
Model A equivalence: 50 checks, 0 mismatches      (variant E)
```

All 75 agree to < 1e-9. The pooled Z-head means come back at **0.6426** (A, stress),
**0.6509** (E, stress), **0.6530** (E, resilience) — Phase 1's numbers to four decimals.

### Results — 5 dataset seeds × 5 init seeds

| variant | state | arm | AUC | init floor | dset floor | **floor** | above .5 | clears | sign-consistent |
|---|---|---|---|---|---|---|---|---|---|
| A | supply_stress | **P(Y\|Z)** | **0.7075** | 0.0243 | 0.1804 | **0.1804** | **+0.2075** | **YES** | yes |
| A | supply_stress | P(Y\|X) | 0.7832 | 0.0171 | 0.0257 | 0.0257 | +0.2832 | YES | yes |
| A | supply_stress | P(Y\|X,Z) | 0.7914 | 0.0438 | 0.0485 | 0.0485 | +0.2914 | YES | yes |
| E | supply_stress | **P(Y\|Z)** | **0.7669** | 0.0191 | 0.1179 | **0.1179** | **+0.2669** | **YES** | yes |
| E | supply_stress | P(Y\|X) | 0.8124 | 0.0241 | 0.0618 | 0.0618 | +0.3124 | YES | yes |
| E | supply_stress | P(Y\|X,Z) | 0.8209 | 0.0291 | 0.0676 | 0.0676 | +0.3209 | YES | yes |
| E | recovery_capability | **P(Y\|Z)** | **0.5711** | 0.0755 | 0.1391 | **0.1391** | **+0.0711** | **no** | yes |
| E | recovery_capability | P(Y\|X) | 0.8124 | 0.0241 | 0.0618 | 0.0618 | +0.3124 | YES | yes |
| E | recovery_capability | P(Y\|X,Z) | 0.8090 | 0.0236 | 0.0615 | 0.0615 | +0.3090 | YES | yes |

Per dataset seed:

```
A  supply_stress        P(Y|Z)  [0.7245, 0.5793, 0.7181, 0.7597, 0.7558]
E  supply_stress        P(Y|Z)  [0.7264, 0.7734, 0.8077, 0.8225, 0.7046]
E  recovery_capability  P(Y|Z)  [0.6646, 0.5387, 0.5672, 0.5594, 0.5255]
```

### The gap, reported and not gated

| variant / state | P(Y\|X,Z) − P(Y\|Z) | reading |
|---|---|---|
| A / supply_stress | **+0.0839** | Z carries most of what X carries for this task, but not all |
| E / supply_stress | **+0.0539** | same, narrower |
| E / recovery_capability | **+0.2379** | Z carries almost nothing X does not |

A second reading that the gate does not use but which belongs on the record: **adding Z to X
buys +0.0082 (A) and +0.0084 (E) for supply stress, and −0.0034 for recovery capability.** Both
positive figures are far inside their own floors (0.0485, 0.0676). So the honest statement is
that the estimated state is *downstream-informative on its own* and is **not** demonstrated to
add anything on top of the observed features it was derived from. That is a weaker claim than
"the latent state is useful downstream", and it is the one the measurement supports.

### Gate

- **Supply Stress, Variant A — PASS.** +0.2075 against a floor of 0.1804, sign-consistent 5/5.
  Noted honestly: the dataset-seed floor is large (0.1804), driven by seed 43 at 0.5793 against
  seed 45's 0.7597. It clears, but on a wide floor.
- **Supply Stress, Variant E — PASS.** +0.2669 against 0.1179, sign-consistent 5/5.
- **Recovery Capability, Variant E — STOP.** +0.0711 against a floor of 0.1391. Does not clear.

### Recovery Capability's STOP — classified **E (measurement/label failure)**

The classification is derived, not chosen. `RESILIENCE` is **static per supplier**, while
`impact` is a label on `(supplier, t0)` that varies across the six test snapshots. A
constant-per-supplier predictor can rank suppliers and cannot rank supplier-snapshots at all,
so the estimand and the label granularity are mismatched — the 24,000 test rows are ~800
independent suppliers per seed replicated six times with identical Z, exactly the
non-independence Phase 1 already flagged and carried forward as qualification 4.

Two things this is **not**, both checked rather than assumed. It is not D (data/observability):
Step 2 measures the observations directly and finds Recovery Capability **identifiable**. And
it is not a failure of the state itself: Model A recovers it at 0.6530, the strongest
recoverability number in the project. The failure is that this benchmark offers no
supplier-level downstream label at a granularity a static state can be scored against.

**Consequence:** Recovery Capability does not proceed to Step 3, per the ladder, *regardless of
its Step 2 case assignment* — and its Step 2 assignment turns out to be Case 1. That is
recorded plainly in the summary rather than smoothed over.

---

## STEP 2 — Identifiability and recoverability

The two questions are kept separate throughout and never conflated.

### 2A — Identifiability: does changing the true Z produce a distinguishable observational change?

**The intervention is a real `do(Z)`, isolated to one supplier at a time.** For a target
supplier `s`, exactly one latent quantity is overridden inside the generator's own namespace,
with structure, co-parents, schedule, dispatch times, eta, lead times, carriers and every other
supplier's Z held fixed:

| state | `do()` | channel |
|---|---|---|
| supply_stress | `stress(s,t) := z_lo` vs `:= z_hi` | `p_delay = 0.025 + 0.38·st·absorption` |
| supplier_reliability | `IDIO[s] := None` vs factual outage | `own_stress` → `stress` → `p_delay` |
| recovery_capability | `RESILIENCE[s] := r_lo` vs `:= r_hi` | `absorption(s)` → `p_delay` |

Isolation is not cosmetic. `stress(s,t)` reads its co-parents' `own_stress`, so a *global*
`IDIO := None` would move every partner too and the measured effect would be contaminated by a
second causal path — which could only ever flatter identifiability. Since a supplier's shipments
depend on that supplier's own `stress` and `absorption` and nothing else, overriding one
supplier at a time is exact.

`z_lo`/`z_hi` are the medians of the two halves of a median split — deliberately the *same
contrast Model A is scored on*, since Phase 1 binarises every continuous state at its train
median. Using quartiles or extremes would have made identifiability answer a strictly easier
question than the recoverability number it is being compared against.

**Re-simulation is replaced by re-realisation under common random numbers**, for the reason
`ml/counterfactual_ground_truth.py` measured rather than assumed: the generator draws from one
sequential stream, so an intervention that flips a single shipment desynchronises every later
draw and 100% of shipments end up differing for reasons unrelated to the intervention. Here each
shipment carries a fixed `(u_delay, g_lateness, e_early, j_jitter)` tuple that **both** worlds
consume, so the only thing that can move an outcome is Z. The observation is then read out by
calling the generator's own `sup_features()` — so the observation vector *is*
`supplier_temporal_features.csv`, the entire time-varying observable channel
`ml/data/loader.py` reads for a Supplier.

**The harness is validated before it is trusted.** Run at *factual* Z with a fresh RNG, it must
reproduce the factual observable distribution or every do(Z) number below would be measuring
the harness:

Means over the five dataset seeds:

| variant | factual on_time_180d | re-realised | factual late rate | re-realised | worst-seed abs diff |
|---|---|---|---|---|---|
| A | 0.8472 | 0.8471 | 0.1740 | 0.1733 | 0.0060 / 0.0062 |
| E | 0.9232 | 0.9214 | 0.0867 | 0.0865 | 0.0037 / 0.0030 |

This check earned its place: its first run showed a **+2.6 point** on-time discrepancy, which
traced to the generator's `J()` timestamp jitter (`db/generate_dataset.py:207`). It looks
cosmetic and is not — a not-delayed shipment is recorded at `J(eta − randint(0,30) hours)`, so
the 1-in-31 that draw 0 hours land at `eta + up to 45 min` and are counted **late** by
`sup_features`' `delivered_at <= eta` test. Reproducing `J()` closed the gap to < 0.006.

#### Results

| variant | state | rows | divergence rate | classifier AUC | **null AUC** | floor | above .5 | clears | sign | **identifiable** |
|---|---|---|---|---|---|---|---|---|---|---|
| A | supply_stress | 46,380 | **0.582** | **0.6262** | 0.4911 | 0.0539 | +0.1262 | YES | yes | **YES** |
| A | supplier_reliability | 487 | **0.010** | 0.5000 | 0.5040 | 0.0909 | −0.0000 | no | NO | **NO** |
| E | supply_stress | 45,975 | **0.305** | **0.5478** | 0.4908 | 0.0296 | +0.0478 | YES | yes | **YES** |
| E | supplier_reliability | 475 | **0.006** | 0.4997 | 0.5079 | 0.0386 | −0.0003 | no | NO | **NO** |
| E | recovery_capability | 45,975 | **0.174** | **0.5392** | 0.4898 | 0.0297 | +0.0392 | YES | yes | **YES** |

The **null is measured, not assumed to be 0.5**: it is the identical pipeline run on two worlds
with the *same* Z and different realisation seeds, which is the only way to know how much
apparent separation the pipeline manufactures from realisation noise alone. It lands at
0.4898–0.5079 across all five arms, and its excursion from 0.5 is folded into the floor
alongside the init-seed and dataset-seed spreads; on both `supplier_reliability` arms it *is*
the binding floor.

**Divergence rate is the number that cannot be argued with**, and it is exact rather than
statistical: the fraction of `(supplier, t0)` observation rows that differ **at all** between
the two worlds. Supply stress moves 58.2% / 30.5% of rows; recovery capability 17.4%; supplier
reliability **1.0% / 0.6%**.

**Supply Stress and Recovery Capability clear, but the margins deserve stating plainly.** On
Variant E they clear their floors by 0.0182 and 0.0095 respectively — real, sign-consistent
across 5/5 seeds, and small. Variant A's supply stress is the only comfortable margin (+0.1262
against 0.0539).

#### Supplier Reliability — the diagnostic case, resolved

The scoped arm above asks whether the outage is visible *at the snapshot where the label says
it is active*. A null there could equally mean "visible, but later", so a strictly **more
generous** arm was added: widen the scope to every t0 in the timeline for every outaged
supplier, so an outage whose observable consequence lands two snapshots after the window still
counts.

| arm | variant | rows | divergence | AUC |
|---|---|---|---|---|
| scoped to active-outage t0 (Model A's target) | A | 487 | **0.010** | 0.5000 |
| all t0, all outaged suppliers | A | 7,305 | **0.081** | 0.5020 |
| scoped to active-outage t0 | E | 475 | **0.006** | 0.4997 |
| all t0, all outaged suppliers | E | 7,125 | **0.033** | 0.5032 |

Flat on the generous arm too. And the mechanism is measurable, not inferred — `IDIO` gives 25%
of suppliers one ~25-day outage across a 21-month timeline, and:

```
mean shipments dispatched INSIDE the outage window:  1.10 (A) / 1.09 (E)
suppliers with at least one such shipment:            47% (A) /  43% (E)
```

**53% (A) and 57% (E) of outaged suppliers dispatch nothing at all during their own outage.** For
the rest, the intervention gets roughly one shipment through which to express itself, against a
`Δp_delay ≈ 0.05`, and the observable feature is a trailing 30/90/180-day rate over deliveries.
The information is not diluted; it is very close to absent.

### 2B — Recoverability: reported from Phase 1, not remeasured

Verbatim from `reports/phase1_latent_state.md`, at each state's best depth. Nothing in this
section was re-fitted for the purpose of reporting it; the numbers Step 1's `--verify-model-a`
independently reproduced (to < 1e-9) are these same numbers.

| variant | state | depth | AUC | init floor | dset floor | **floor** | above .5 | clears | sign-consistent |
|---|---|---|---|---|---|---|---|---|---|
| A | supply_stress | h² | **0.6426** | 0.0066 | 0.0390 | 0.0390 | +0.1426 | **YES** | yes |
| E | supply_stress | h¹ | **0.6509** | 0.0148 | 0.0256 | 0.0256 | +0.1509 | **YES** | yes |
| E | recovery_capability | h¹ | **0.6530** | 0.0204 | 0.0310 | 0.0310 | +0.1530 | **YES** | yes |
| A | supplier_reliability | h⁴ | 0.4870 | 0.0260 | 0.0439 | 0.0439 | −0.0130 | no | **NO** |
| E | supplier_reliability | h⁴ | 0.5131 | 0.0249 | 0.1017 | 0.1017 | +0.0131 | no | **NO** |

### Both on a comparable footing

The identifiability classifier and Model A's head solve different tasks, so both are put on the
same **AUC-above-its-own-floor** axis rather than compared by eye:

| variant / state | identifiability AUC (floor) | margin over floor | recoverability AUC (floor) | margin over floor |
|---|---|---|---|---|
| A / supply_stress | 0.6262 (0.0539) | **+0.0723** | 0.6426 (0.0390) | **+0.1036** |
| E / supply_stress | 0.5478 (0.0296) | **+0.0182** | 0.6509 (0.0256) | **+0.1253** |
| E / recovery_capability | 0.5392 (0.0297) | **+0.0095** | 0.6530 (0.0310) | **+0.1220** |
| A / supplier_reliability | 0.5000 (0.0909) | **−0.0909** | 0.4870 (0.0439) | **−0.0569** |
| E / supplier_reliability | 0.4997 (0.0386) | **−0.0389** | 0.5131 (0.1017) | **−0.0886** |

**One asymmetry in that table needs an explanation rather than a shrug: recoverability is
*higher* than identifiability on every arm.** A model estimating Z from a single un-intervened
world beats a classifier distinguishing two do(Z) worlds. That is not a contradiction — it is
informative about what Model A is actually reading.

`do(Z)` severs Z from its causes. In the factual world `own_stress(s,t)` begins at
`(1 − base_rel)·0.5` and picks up event-membership terms, so Z is *correlated with observable
supplier attributes and graph position* that remain predictive whether or not Z's own
downstream consequences are visible. Model A, reading a 128-dim representation built by message
passing over the whole graph, can exploit those upstream correlates. The identifiability
classifier cannot: under `do(Z)`, every supplier is forced to the same z, and the only remaining
signal is Z's marginal effect on realised deliveries.

So the two numbers measure genuinely different things, and the gap between them is the size of
the part of Model A's performance that comes from **reading the causes of Z rather than its
observable consequences**. That is legitimate for prediction and is *not* causal recovery. It
is stated here because Step 1's PASS could otherwise be over-read.

### Case assignment

| state | identifiable | recoverable | **Case** | classification | disposition |
|---|---|---|---|---|---|
| **Supply Stress** (A) | YES | YES | **Case 1** | — | → Step 3 |
| **Supply Stress** (E) | YES | YES | **Case 1** | — | → Step 3 |
| **Recovery Capability** (E) | YES | YES | **Case 1** | — | blocked at Step 1 (E); does **not** reach Step 3 |
| **Supplier Reliability** (A) | NO | NO | **Case 3** | **C** | **STOP — closed** |
| **Supplier Reliability** (E) | NO | NO | **Case 3** | **C** | **STOP — closed** |

**Supplier Reliability is Case 3, not Case 4.** Case 4's trigger is explicit and conjunctive —
identifiability *confirmed* **and** a thin positive rate. Its positive rate is thin (1.1–1.2%,
the codebase's own reference point for thin), but identifiability is **not** confirmed: it is
measured and rejected on two variants with different mechanisms, on both the scoped and the
generous arm, against a measured null. The first conjunct fails, so the assignment is Case 3.

Per the Case 3 rule, this is closed. **Do not compensate with more capacity, temporal modelling,
calibration or a confidence mechanism — there is nothing to calibrate.** This also supersedes
Phase 1's recommendation to "change the sampling, not the head": reweighting cannot manufacture
an observable consequence that 57% of outaged suppliers never generate.

**Case 2 was assigned to nothing**, so no architecture improvement is scientifically justified
by this prompt. That is consistent with, and independent of, `reports/phase_modelB_concat_baseline.md`'s
conclusion that Model C is not justified.

#### Stated limitation on every identifiability number

The schedule is held fixed — that is what makes common random numbers work. `RESILIENCE` also
acts through `mitigation_level()`, which changes the replenishment trigger and order quantity
and therefore the schedule itself; `stress` perturbs arrival timing similarly. That channel is
outside this measurement, so **every identifiability figure here is a lower bound**. It cannot
be recovered by re-simulation on this benchmark, for the reason
`ml/counterfactual_ground_truth.py` already measured. This bound direction matters most for
Recovery Capability (which clears by only 0.0095) and least for the Supplier Reliability
negative, where the scoped arm is 0/487 rows and no plausible schedule effect closes that.

---

## STEP 3 — Calibration (Case 1 states only)

In scope: **Supply Stress on Variants A and E**. Recovery Capability is excluded by its Step 1
STOP; Supplier Reliability is closed at Step 2.

**Protocol.** Isotonic PAVA and equal-count ECE imported from `ml/hypothesis_ranker.py`, nothing
reimplemented. The map is fitted on one world and evaluated on a **different** world, rotating
over all 20 ordered seed pairs. Both floors from `ml/uncertainty_calibrate.py` are reproduced:
the *fit floor* (refit PAVA on identical inputs — deterministic, expected to be exactly 0) and
the *member floor* (vary which head init seeds compose the ensemble), which is the operative one.

### Results — 20 leave-one-world-out folds

| variant | arm | ECE | Brier | AUC |
|---|---|---|---|---|
| A | single_raw | 0.0403 | 0.2350 | 0.6430 |
| A | single_cal | 0.0410 | 0.2350 | 0.6414 |
| A | **ens_raw** | **0.0407** | 0.2346 | 0.6435 |
| A | **ens_cal** | **0.0461** | 0.2349 | 0.6403 |
| E | single_raw | 0.0365 | 0.2311 | 0.6539 |
| E | single_cal | 0.0421 | 0.2323 | 0.6510 |
| E | **ens_raw** | **0.0377** | 0.2314 | 0.6525 |
| E | **ens_cal** | **0.0406** | 0.2326 | 0.6494 |

| variant | ECE improvement (raw → cal) | folds improved | fit floor | **member floor** | gate |
|---|---|---|---|---|---|
| A | **−0.0054** | 7/20 | 0.000000 | **0.0132** | **STOP** |
| E | **−0.0029** | 8/20 | 0.000000 | **0.0117** | **STOP** |

Per-fold improvement ranges from −0.0288 to +0.0194 (A) and −0.0295 to +0.0240 (E).
Calibration is a coin flip that loses on average.

### The STOP is classified **G**, and the diagnosis distinguishes three different things it could have been

"Calibration failed" has at least three meanings and they call for different responses, so both
discriminating measurements were run.

**Was there anything to calibrate?** Equal-count ECE is a mean of |empirical − confidence| over
bins, and each bin's empirical rate is a binomial estimate from n/10 samples, so even a
*perfectly* calibrated predictor posts non-zero ECE at finite n. Resampling labels from the
model's own predicted probabilities — making it calibrated by construction — gives the floor:

| variant | binomial ECE floor | [p05, p95] | observed raw ECE | ratio | raw ECE at the floor? |
|---|---|---|---|---|---|
| A | 0.0184 | [0.0122, 0.0252] | 0.0407 | **2.22×** | **No** |
| E | 0.0173 | [0.0111, 0.0259] | 0.0377 | **2.18×** | **No** |

So the miscalibration is **real** — about 0.020–0.022 of it, roughly twice the noise floor.
This is not a case of nothing to fix.

**Can isotonic fix it at all?** Fitting the map on the *same* world it is evaluated on is
optimistically biased and is never gated on — reported purely as an upper bound:

| variant | within-world ECE raw → cal | improvement |
|---|---|---|
| A | 0.0407 → **0.0205** | **+0.0202** |
| E | 0.0377 → **0.0153** | **+0.0224** |

Isotonic removes essentially all of the real miscalibration and lands the score **at the
binomial floor**. So the machinery works and the score is fixable.

**Therefore the failure is specific and nameable: the calibration map does not transfer between
worlds.** Each dataset seed's head is miscalibrated in a *different direction*, so a map fitted
on world A actively harms world B — leaving cross-world ECE worse than raw in 13/20 and 12/20
folds. This is the dataset-seed dominance this project has documented everywhere else
(`docs/14_Project_Roadmap.md` §4 risk 4; §5.6's 10.3× ratio), now surfacing inside the
calibration map itself.

Fitting within-world is leakage, and the prompt's protocol forbids it explicitly. **There is
therefore no legitimate route to a calibrated Layer-3 output on this benchmark at five worlds.**
Classified **G (calibration/uncertainty failure)** — not F: the underlying fits are stable
(the fit floor is exactly 0.000000, and Phase 1's `assert_backbone_frozen` never fired across
any of this session's head fits).

---

## STEP 4 — Per-entity confidence

**Formally out of scope** — Step 4 admits only states that passed Step 3, and none did. It was
run and is reported anyway, clearly labelled as beyond the STOP, because it is an independent
gate on a different claim (Step 3 asks whether probabilities are honest in aggregate; Step 4
asks whether the confidence signal orders error per entity) and because its result is a clean
negative that would otherwise have to be re-derived later.

Confidence is `−var` across the five head init seeds within a world, calibrated first with a
different world's map. Bins are equal-count, **least confident first**.

### Variant A (h²), pooled across 5 worlds

| bin | n | mean var | Brier | \|err\| | AUC | base rate |
|---|---|---|---|---|---|---|
| 1 (least confident) | 4,108 | 1.95e-03 | 0.2358 | 0.4638 | 0.6335 | 0.5565 |
| 2 | 4,108 | 3.36e-04 | 0.2377 | 0.4665 | 0.6328 | 0.5166 |
| 3 | 4,108 | 1.68e-04 | 0.2300 | 0.4574 | 0.6604 | 0.5397 |
| 4 | 4,107 | 8.60e-05 | 0.2335 | 0.4650 | 0.6391 | 0.5401 |
| 5 (most confident) | 4,107 | 3.18e-05 | 0.2379 | 0.4667 | **0.6148** | 0.5435 |

### Variant E (h¹), pooled across 5 worlds

| bin | n | mean var | Brier | \|err\| | AUC | base rate |
|---|---|---|---|---|---|---|
| 1 (least confident) | 4,800 | 1.49e-03 | 0.2347 | 0.4601 | 0.6496 | 0.5123 |
| 2 | 4,800 | 4.60e-04 | 0.2280 | 0.4543 | 0.6605 | 0.5042 |
| 3 | 4,800 | 2.58e-04 | 0.2285 | 0.4543 | 0.6682 | 0.4877 |
| 4 | 4,800 | 1.35e-04 | 0.2341 | 0.4590 | 0.6422 | 0.5185 |
| 5 (most confident) | 4,800 | 4.24e-05 | 0.2369 | 0.4662 | **0.6142** | 0.5262 |

| variant | monotone? | worlds monotone | least− minus most-confident Brier | member floor on bin Brier | gate |
|---|---|---|---|---|---|
| A | **False** | 0/5 | **−0.0021** | 0.0079 | **STOP** |
| E | **False** | 0/5 | **−0.0022** | 0.0097 | **STOP** |

**Classified G.** Not monotone in any world, and the total spread across all five bins is
*negative* and lies inside the member floor — so there is no confidence ordering here to
salvage, in either direction.

The shape is worth naming because it is the opposite of the intended one: **the most-confident
bin is the least discriminative** (AUC 0.6148 vs 0.6335 in bin 1 on A; 0.6142 vs 0.6496 on E).
Low init-seed variance marks predictions where all five heads converged — which reflects
optimisation insensitivity, not certainty about the label. This is `ml/uncertainty_ensemble.py`'s
own warning arriving in practice: model-init ensembling estimates the *smaller* variance source,
and here it estimates something that is not error-relevant at all. The mean variance (3e-05 to
2e-03) against a Brier of ~0.23 says the same thing in one number — the signal is not weak, it
is essentially absent.

Whether a **dataset-seed** confidence axis would do better is not answerable on this benchmark:
entities do not persist across dataset seeds, so per-entity dataset-seed variance is not
identifiable, exactly as `ml/uncertainty_ensemble.py` states. This module does not fabricate it.

---

## STEP 5 — Insufficient-evidence threshold

**Formally out of scope** — Step 5 admits only states that passed Step 4. Run and reported for
the same reason as Step 4, and because the prompt requires the coverage-vs-reliability trade-off
be shown plainly rather than hidden.

**Selection method, documented before evaluation and not revised afterwards.** The threshold is
a cut on the validated confidence signal (init-seed variance). Selection uses the **validation
partitions of the four selection worlds only**; the held-out world's test partition is untouched
until the threshold is frozen. The criterion:

> Take the **smallest** rejection rate in the grid (5%, 10%, …, 50%) whose accepted-set Brier
> improves on the all-predictions Brier by more than the Step-3 member-composition floor.

Two properties are deliberate. It **prefers coverage** — the grid is walked from 5% upward and
the first qualifying candidate wins, so the rule never rejects more than it must. And it can
return **nothing**: if no candidate clears the floor, that is reported as the answer rather than
falling back to whichever cut happened to score best.

### The coverage-vs-reliability trade-off, in full

Mean over the 5 rotations, on the selection worlds' validation partitions. Member floor to
clear: **0.0132** (A), **0.0117** (E).

| reject | variant A: accepted / total | Brier all | Brier accepted | improvement | variant E: accepted / total | Brier all | Brier accepted | improvement |
|---|---|---|---|---|---|---|---|---|
| 5% | 7,804 / 8,215 | 0.2339 | 0.2330 | **+0.00093** | 9,120 / 9,600 | 0.2337 | 0.2338 | −0.00006 |
| 10% | 7,393 / 8,215 | 0.2339 | 0.2329 | **+0.00096** | 8,640 / 9,600 | 0.2337 | 0.2341 | −0.00041 |
| 15% | 6,983 / 8,215 | 0.2339 | 0.2332 | +0.00068 | 8,160 / 9,600 | 0.2337 | 0.2348 | −0.00108 |
| 20% | 6,572 / 8,215 | 0.2339 | 0.2339 | −0.00000 | 7,680 / 9,600 | 0.2337 | 0.2355 | −0.00174 |
| 25% | 6,161 / 8,215 | 0.2339 | 0.2337 | +0.00016 | 7,200 / 9,600 | 0.2337 | 0.2358 | −0.00210 |
| 30% | 5,751 / 8,215 | 0.2339 | 0.2343 | −0.00043 | 6,720 / 9,600 | 0.2337 | 0.2363 | −0.00258 |
| 35% | 5,340 / 8,215 | 0.2339 | 0.2346 | −0.00075 | 6,240 / 9,600 | 0.2337 | 0.2366 | −0.00288 |
| 40% | 4,929 / 8,215 | 0.2339 | 0.2344 | −0.00052 | 5,760 / 9,600 | 0.2337 | 0.2380 | −0.00431 |
| 50% | 4,108 / 8,215 | 0.2339 | 0.2351 | −0.00116 | 4,800 / 9,600 | 0.2337 | 0.2395 | −0.00582 |

**No candidate qualifies in any rotation, on either variant.** The best improvement available
anywhere is **+0.00096** (A, 10% rejection) against a floor of 0.0132 — short by a factor of
**14**. On Variant E every candidate is negative and the trade-off is **monotonically worse**
the more it rejects: throwing away half the predictions makes the survivors *less* reliable.

### Required reporting

| quantity | Variant A | Variant E |
|---|---|---|
| total test predictions available | 20,538 | 24,000 |
| usable predictions | **20,538** (all) | **24,000** (all) |
| insufficient-evidence predictions | **0** | **0** |
| percentage rejected | **0.0%** | **0.0%** |
| performance before rejection (AUC / Brier / ECE) | 0.6435 / 0.2346 / 0.0407 | 0.6525 / 0.2314 / 0.0377 |
| performance on accepted predictions | identical — nothing was rejected | identical |
| does rejection improve reliability? | **No** | **No** |

The before-rejection row is the Step-3 **`ens_raw`** pipeline (ensemble mean over the five head
init seeds, uncalibrated), not `ens_cal` — because the calibrated arm was itself rejected at
Step 3, and quoting a calibrated number here would present as the operating point something the
ladder already stopped.

Because no threshold was frozen, no rejection was applied to any test partition and the
matched-coverage random-rejection control never ran. That control remains implemented
(`apply_threshold`, 200 repeats at matched coverage) and would be the first thing to report if a
future arm ever produces a qualifying threshold — a rejection rule is worth something only to
the extent it beats dropping the same number of predictions at random.

**This is not a "high rejection rate" outcome, which the prompt rightly says would not be
automatic failure.** It is the opposite: the confidence signal cannot identify *any* subset,
at any coverage, that is more reliable than the whole. Nothing is being hidden and no cases are
being suppressed — the rejected set is empty because no cut earned one.

---

## STEP 6 — Integrity: SHARE, the Layer-2 readout, and Model A's heads are untouched

**Ground rule:** SHARE, the Layer-2 Markov readout, and Model A's existing heads are not
retrained or modified anywhere in this prompt.

**Interpretation, stated because it is load-bearing.** Steps 3–5 require per-entity
*probabilities* from Model A's head; a calibration and confidence pipeline cannot be built on a
scalar AUC, and `train_head` returns only an AUC. The head was therefore re-instantiated from
its own unmodified definition with byte-identical hyperparameters, never edited and never
re-tuned. That claim is verified numerically rather than asserted: all **75** re-fits reproduce
`out/phase1/heads_{A,E}.json`'s per-init-seed AUCs to < 1e-9, and the pooled means come back as
Phase 1's 0.6426 / 0.6509 / 0.6530 exactly. SHARE itself was never trained — every backbone
loaded from cache.

```
$ git status --porcelain
?? ml/identifiability_check.py        <- new, this session
?? ml/layer3_sufficiency.py           <- new, this session
?? ml/layer3_uncertainty.py           <- new, this session
?? ml/modelB_concat_head.py           <- pre-existing, prior session
?? reports/phase_modelB_concat_baseline.md

$ git status --porcelain db/          # empty — the generator was executed, never edited

$ shasum -a 256 ...
1b85a68f9e9705be...  ml/models/rgcn_attn_encoder.py         IDENTICAL to V2
9e7919e6bfcf39ca...  ml/models/rgcn_attn_markov_encoder.py  IDENTICAL to V2
cbd40aadc8570745...  ml/models/heads.py                     IDENTICAL to V2
8f669a1683117280...  ml/models/encoder.py                   IDENTICAL to V2
8fa14bc270946507...  ml/models/depth.py                     IDENTICAL to V2
4cf67f7d4f25b3a8...  ml/models/model.py                     IDENTICAL to V2
689962edd1315599...  ml/ds_backbone.py                      IDENTICAL to V2
c2e15e39cdc101d4...  ml/hypothesis_ranker.py                IDENTICAL to V2
3a8c888040446aa2...  ml/uncertainty_ensemble.py             IDENTICAL to V2
24ccd9d58eb44b69...  ml/uncertainty_calibrate.py            IDENTICAL to V2
4e9a4db9d2d4ef3a...  ml/data/loader.py                      IDENTICAL to V2

8e4a0803d8f09cd0...  ml/latent_state_head.py
```

`ml/latent_state_head.py`'s digest **`8e4a0803d8f09cd0…` matches the value recorded at the close
of `reports/phase_modelB_concat_baseline.md`** exactly, so Model A's methodology is the same
code that produced Phase 1's numbers.

All ten backbone checkpoints in `out/ds_ckpt/` carry mtimes of **2026-08-13 22:03–22:44**,
predating this session (2026-08-14). Every fit loaded from cache; **no backbone was trained**.
Backbone health on first load matched Phase 1's recorded values exactly
(`delay=0.8046 shortage=0.7835 impact=0.9340`, variant A seed 42).

`assert_backbone_frozen()` ran after head construction and again after training on **every**
head fit in this session — Step 1's **300** fits (A: 1 state × 5 dataset × 5 init × (1 Z head +
3 downstream arms) = 100; E: 2 states × 5 × 5 × 4 = 200) and Steps 3–5's **50** grid fits (2
variants × 5 dataset × 5 init) — and never fired. Step 2's classifier fits call `train_head`
without a model handle, since they never touch the backbone at all: that arm runs entirely
inside the generator's namespace and reads no SHARE representation.

---

## Final summary

### States with a calibrated, confidence-aware, threshold-gated output

**None.**

### Excluded — at which step, and why

| state | excluded at | classification | reason |
|---|---|---|---|
| **Supplier Reliability** (A, E) | **Step 2** | **C** (identifiability) | Case 3. `do(IDIO)` moves 1.0%/0.6% of the labelled observation rows and 8.1%/3.3% on the generous all-t0 arm; classifier AUC 0.5000/0.4997 against a measured null of 0.5040/0.5079. 57% of outaged suppliers dispatch nothing during their own outage. **Closed — do not compensate.** |
| **Recovery Capability** (E) | **Step 1** | **E** (measurement/label) | P(Y\|Z) = 0.5711, +0.0711 against a floor of 0.1391. A *static* per-supplier state cannot rank the time-varying `(supplier, t0)` `impact` label; the effective sample is ~800 suppliers per seed, not 24,000 rows. Identifiable and recoverable (Case 1) — the benchmark offers no downstream label at a granularity it can be scored against. |
| **Supply Stress** (A) | **Step 3** | **G** (calibration) | Case 1, Step 1 PASS. Cross-world isotonic makes ECE **worse** (−0.0054, 7/20 folds improved) against a member floor of 0.0132. Miscalibration is real (2.22× the binomial floor) and within-world isotonic fixes it (+0.0202) — the map does not transfer between worlds. |
| **Supply Stress** (E) | **Step 3** | **G** (calibration) | As above: −0.0029, 8/20 folds, member floor 0.0117; 2.18× the binomial floor; within-world +0.0224. |

### What this prompt establishes that was not known before

1. **Supply Stress and Recovery Capability are genuinely identifiable** under a real do(Z) with
   the schedule pinned by common random numbers — measured against a measured null, on five
   dataset seeds, sign-consistent. Margins over floor are +0.0723 / +0.0182 / +0.0095.
2. **Supplier Reliability's open failure category is resolved as C, not A/B/D.** Phase 1 could
   only say the head failed; this says the observations do not carry the distinction, on both
   the labelled-snapshot arm and a strictly more generous one, with the mechanism quantified.
   Phase 1's "change the sampling, not the head" recommendation is superseded.
3. **Model A's recoverability is not primarily causal.** Recoverability exceeds identifiability
   on every arm, and the gap is the part of Model A's performance that comes from reading the
   *causes* of Z rather than its observable consequences.
4. **The calibration failure is a transfer failure, not an absence of miscalibration.** Raw ECE
   sits at 2.2× the binomial noise floor and within-world isotonic removes essentially all of
   it; only the cross-world map fails. Naming it that way is what makes it actionable.
5. **Head-init-seed variance is not a usable confidence signal for Layer 3** — the
   most-confident bin is the *least* discriminative on both variants.

### What must not be built on this evidence

- No architecture work for any state: **Case 2 was assigned to nothing**.
- Nothing further for Supplier Reliability: Case 3 is closed, including reweighting/resampling.
- No confidence or abstention mechanism built on init-seed variance: Step 4 measured it and it
  does not order error.
- No within-world calibration fit, which is the one change that would make Step 3 "pass" and is
  leakage.

### The defensible follow-ups this report supports

1. **More dataset seeds, for the calibration transfer question specifically.** The member floor
   (0.0132 / 0.0117) and the cross-world variance are both estimated from five worlds. The
   diagnosis says the map is world-specific; more worlds would establish whether a map *pooled*
   across many worlds transfers, which is a different and untested proposition from the
   pairwise rotation run here. This is a re-measurement, not a licence for new architecture.
2. **A time-varying downstream label at supplier granularity** would let Recovery Capability's
   Step 1 be re-asked properly. `mitigation_level` — Phase 1's highest-value instrumentation
   candidate, now recorded per supplier per week — is the dynamic form of Recovery Capability
   and is the obvious candidate. That is instrumentation plus a new Step 1, not a rescue of
   this one.
3. **Spec-scale confirmation (Phase 5)** of every number above, all of which are `v1`-preset
   pilots.

---

## Reproduction

```
$ venv/bin/python -u ml/layer3_sufficiency.py --variant A --states supply_stress \
      --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 --verify-model-a \
      --out out/layer3/step1_A.json
$ venv/bin/python -u ml/layer3_sufficiency.py --variant E \
      --states supply_stress,recovery_capability --seeds 42,43,44,45,46 \
      --init-seeds 0,1,2,3,4 --verify-model-a --out out/layer3/step1_E.json

$ venv/bin/python -u ml/identifiability_check.py --variant A \
      --states supply_stress,supplier_reliability --seeds 42,43,44,45,46 \
      --init-seeds 0,1,2,3,4 --out out/layer3/step2_A.json
$ venv/bin/python -u ml/identifiability_check.py --variant E --seeds 42,43,44,45,46 \
      --init-seeds 0,1,2,3,4 --out out/layer3/step2_E.json

$ venv/bin/python -u ml/layer3_uncertainty.py --variant A --state supply_stress \
      --out out/layer3/step345_A.json
$ venv/bin/python -u ml/layer3_uncertainty.py --variant E --state supply_stress \
      --out out/layer3/step345_E.json
```

Raw results: `out/layer3/step1_{A,E}.json`, `out/layer3/step2_{A,E}.json`,
`out/layer3/step345_{A,E}.json`.
Model A comparison numbers: `out/phase1/heads_{A,E}.json`.
