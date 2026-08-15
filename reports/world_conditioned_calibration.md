# World-Conditioned Calibration — following up Layer 3 Step 3's STOP — G

**Date:** 2026-08-15
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisite read in full before starting:** `reports/layer3_uncertainty_aware.md` STEP 3 — the
STOP this session follows up. Every number below is anchored to it.

**New code, five files, no inherited file modified:** `ml/calibration_grid_cache.py` (shared
prediction cache), `ml/calibration_curve_compare.py` (Step 4),
`ml/calibration_variance_decompose.py` (Step 5), `ml/world_summary_features.py` (7b's
conditioning variable), `ml/world_conditioned_calibration.py` (Steps 1–3 reproduction, 7a, 7b,
gating). `ml/hypothesis_ranker.py`'s isotonic/ECE machinery is imported unmodified, as Step 3
did; no calibration primitive is reimplemented.

**Scale caveat, carried from every prior phase:** everything below is the `v1` preset (800
suppliers, 15 snapshots), five dataset seeds 42–46. Per `docs/14_Project_Roadmap.md` §3.5 this is
a **pilot, not a benchmark finding**.

---

## Verdict, up front

| question | answer |
|---|---|
| Do Steps 1–3 reproduce in this environment? | **Yes — bit-exact**, every figure to 0.000e+00 |
| Step 4 — do the per-world calibration curves differ, and how? | **Variant A: not distinguishably** (0.80–1.02× a matched null). **Variant E: yes** (1.48–1.83×), and the difference is **non-affine** |
| Step 5 — does world identity explain the differences? | **Not demonstrated.** `pct_world` lands at 22.6–92.0%, in the band of the v2 precedent — but **9 of 10** metric×variant cells fail a null in which world identity explains *nothing* |
| Step 6 — does a one-world map transfer? | **No** — cited from Step 3, not re-run |
| 7a — does seed-identity conditioning recover the ceiling? | Per-world isotonic yes (+0.0202/+0.0224, = Step 3's leaky bound). A per-world *affine* head recovers only +0.0121 / +0.0022. **Diagnostic only — not deployable, and weaker evidence than anticipated** |
| 7b — does an observable-proxy conditional map generalise to an unseen world? | **No.** On no variant, on no arm, does conditioning add anything over the same four worlds pooled *without* it |
| Does Supply Stress move off STOP — G? | **No. It stays at STOP — G**, with a sharper diagnosis and one new gated candidate that is not conditioning |

**The one unanticipated positive**, found by a control rather than by the experiment: on Variant
A, an **unconditional** calibration map pooled over four worlds improves ECE by **+0.0081**
against a freshly measured floor of 0.0030 — where Step 3's one-world map *harmed* it by −0.0054.
It is not sign-consistent (4/5 folds) and it **fails on Variant E** (−0.0010), so it does not
clear this project's bar. It is reported as the highest-value follow-up, not as a fix.

---

## A documentation defect, restated because this is the third session it has cost

`reports/decision_support_build.md` — which the build prompt instructs this session to read at
§2.2 — **does not exist in this repository**. `reports/` holds seven files and that is not one of
them. This is the same class of gap `reports/layer3_uncertainty_aware.md` §"Scope notes" recorded
for `docs/decision_support_build.md` §2.3 and `ml/test_hypothesis_calibration.py`: the V2 report
was not ported in Phase 0.

**The gap was closed rather than noted and skipped.** The three figures this session needs from
§2.2 survive verbatim in `docs/14_Project_Roadmap.md` §"Confidence / uncertainty", which quotes
them and cites §2.2 as their source:

> Dataset-seed variance dominates total AUC variance: **98.7% (delay), 67.9% (shortage), 70.8%
> (impact)** (`reports/decision_support_build.md` §2.2).

Step 5 below uses those, from that surviving record, and pins them next to the code that cites
them (`ml/calibration_variance_decompose.py::V2_PCT_WORLD`) so the comparison cannot drift. What
is **not** recoverable is §2.2's own methodology beyond the two estimator formulas the prompt
quotes — in particular whether it gated its percentages against any null. That matters for how
Step 5's result is read, and is flagged there rather than assumed either way.

---

## STEPS 1–3 — reproduction, not new measurement

**This section re-runs Step 3 and measures nothing new.** Its only purpose is to confirm this
session's environment is the one Step 3 measured, before four new experiments are built on top of
its numbers. The grid was rebuilt from scratch — same checkpoints, same
`ml/layer3_uncertainty.py::build_grid`, same `ml/hypothesis_ranker.py` isotonic/ECE code — and
diffed against `out/layer3/step345_{A,E}.json`.

| quantity | Variant A | Variant E | Δ vs Step 3 (both) |
|---|---|---|---|
| `single_raw` ECE | 0.040329 | 0.036469 | **0.000e+00** |
| `single_cal` ECE | 0.041037 | 0.042102 | **0.000e+00** |
| `ens_raw` ECE | 0.040684 | 0.037697 | **0.000e+00** |
| `ens_cal` ECE | 0.046078 | 0.040563 | **0.000e+00** |
| ECE improvement (raw → cal) | −0.005393 | −0.002866 | **0.000e+00** |
| folds improved | 7/20 | 8/20 | identical |
| member floor | 0.013194 | 0.011674 | **0.000e+00** |
| fit floor | 0.00000000 | 0.00000000 | identical |
| binomial ECE floor | 0.018368 (2.215×) | 0.017304 (2.179×) | **0.000e+00** |
| within-world isotonic improvement | +0.020182 | +0.022443 | **0.000e+00** |

Every figure reproduces **exactly**, not merely within the member floor. Backbone health on first
load matched Phase 1's recorded values (`delay=0.8046 shortage=0.7835 impact=0.9340`, Variant A
seed 42). Nothing drifted, so the four new experiments proceed.

**One implementation note that earned its place**, because it nearly broke exactness. The
prediction grid is cached to `.npz` so five modules do not each pay 11 s/world to rebuild it.
The first version widened the arrays to `float64` on write. The head emits `float32`, so the
ensemble mean is a `float32` reduction; widening moved it in the ~1e-9 place, which was enough to
flip a tie in PAVA's stable sort and in `ece`'s equal-count bin edges — and on Variant E world 45
that moved ECE by **8e-4**. Far inside the member floor and completely immaterial to any
conclusion, but it would have meant the cached grid was not the object Step 3 measured. The cache
now preserves dtype exactly, and the round-trip is asserted (`np.array_equal` plus dtype) on
every load in the check above.

---

## STEP 4 — how the per-world calibration curves differ

**New.** Step 3 showed transfer fails in aggregate (ECE). ECE is a scalar, so it cannot say
whether the worlds' curves are parallel, stretched, or genuinely differently shaped — and those
call for different fixes. Each pair of per-world isotonic curves is decomposed on a shared grid:

    g_i(x) − g_j(x)  =  shift  +  slope·(x − x̄)  +  residual(x)

A pure **shift** would be fixable by one global offset; a pure **slope** by one global
temperature; a large **residual** is the part *no* affine recalibration can remove. Components
are reported as RMS over the grid, so they partition the total and "which dominates" is a
comparison rather than a judgement call. The grid is the **overlap** of the five worlds'
prediction ranges (97.9% / 99.7% of predictions), because `isotonic_apply` clamps outside its
fitted support and evaluating beyond the overlap would manufacture residual out of extrapolation.

### The floor is the number that changes the reading

The null this step must reject is *"all five worlds share one calibration curve, and what we see
is label noise at these sample sizes."* It is simulated directly — one pooled isotonic curve
treated as common truth, each world's labels resampled from it at that world's own n and
prediction distribution, curves refitted, pairs decomposed, 200 times. This is the same
construction Step 3's binomial ECE floor used, applied to curve shape instead of to ECE, and it
is matched on n, on the prediction distribution, and on the 10-pair aggregation.

### Results — 10 world pairs per variant

| | shift | slope·x | **residual** | total | residual share |
|---|---|---|---|---|---|
| **Variant A** observed | 0.0162 | 0.0164 | **0.0542** | 0.0612 | 78.8% |
| same-curve null, mean | 0.0204 | 0.0252 | 0.0532 | 0.0650 | — |
| same-curve null, p95 ← **the gate** | 0.0337 | 0.0401 | 0.0652 | 0.0823 | — |
| **observed / null** | **0.80×** | **0.65×** | **1.02×** | **0.94×** | — |
| clears? | no | no | **no** | no | |
| **Variant E** observed | 0.0217 | 0.0298 | **0.0710** | 0.0834 | 74.4% |
| same-curve null, mean | 0.0122 | 0.0163 | 0.0479 | 0.0539 | — |
| same-curve null, p95 ← **the gate** | 0.0201 | 0.0258 | 0.0559 | 0.0642 | — |
| **observed / null** | **1.79×** | **1.83×** | **1.48×** | **1.55×** | — |
| clears? | **YES** | **YES** | **YES** | **YES** | |

Two further floors are reported for continuity and are deliberately **not** gated on. The
**member floor** (3-of-5 init-seed subsets, same world, 225 comparisons) is 0.0367 / 0.0361 on
the residual — it holds the sample fixed, so it cannot answer the sampling question. The
**split-half floor** (disjoint halves of one world, 50 comparisons) is 0.0820 / 0.0684, measured
at n/2 where isotonic noise is larger; inheriting an n/2 floor for an n-sized comparison would be
the mirror image of the leakage Step 3 refused.

### What this says

**The residual share is not the finding, and reading it as one would be the mistake this step
exists to prevent.** Residual dominates 10/10 pairs on both variants and takes 78.8% / 74.4% of
the difference energy — but it takes ~82% of the *null's* energy too. Isotonic curves are step
functions; they are non-affine against each other even when they estimate the same truth. The
share is structural. The **excess over the null** is the evidence, and it splits by variant:

- **Variant A: the five calibration curves are not distinguishable from a single common curve.**
  Every component sits at or below its null (0.80×, 0.65×, 1.02×, 0.94×); 0/10 pairs have a
  residual above the null p95. At five worlds and ~4,100 rows each, there is no measurable
  per-world curve difference to condition on.
- **Variant E: they do differ, and the difference is non-affine.** All four components clear;
  7/10 pairs have a residual above the null p95; 10/10 pairs' curves cross. So for E, no single
  global shift, temperature, or other affine recalibration could reconcile the worlds — only a
  world-specific map could.

This is what decides the *shape* of Step 7's fix, and it already predicts the split seen there:
a per-world **affine** head should recover much of the ceiling on A (where an affine story
suffices) and little of it on E (where it does not). Step 7a measures exactly that.

---

## STEP 5 — does world identity explain the curve differences

**Hypothesis test against a named precedent, not a fresh guess.** `decision_support_build.md`
§2.2's two-way split is applied unchanged, so the numbers are comparable:

    var_model   = mean_d Var_m(metric)        var_dataset = Var_d(mean_m metric)
    pct_world   = var_dataset / (var_dataset + var_model)

Five metrics describe each (world × init-seed) cell's calibration curve: raw ECE, the ECE its own
map achieves, and Step 4's shift / slope / residual measured against a pooled reference curve.

### Results, with the v2 figures alongside

| metric | var_model | var_world | **% world** | % bias-corrected | null mean | null p95 | clears? |
|---|---|---|---|---|---|---|---|
| **Variant A** | | | | | | | |
| `ece_raw` | 1.51e-05 | 1.03e-04 | **87.3%** | 86.9% | 67.1% | 90.1% | no |
| `ece_own_map` | 4.56e-05 | 1.33e-05 | **22.6%** | 8.4% | 25.2% | 48.5% | no |
| `shift` | 5.46e-05 | 8.68e-05 | **61.4%** | 58.1% | 74.1% | 92.3% | no |
| `slope` | 1.35e-03 | 2.50e-03 | **65.0%** | 62.3% | 68.4% | 91.4% | no |
| `rms_residual` | 2.07e-05 | 5.02e-05 | **70.8%** | 69.0% | 53.6% | 85.7% | no |
| **Variant E** | | | | | | | |
| `ece_raw` | 1.88e-05 | 4.27e-05 | **69.4%** | 67.4% | 61.8% | 87.3% | no |
| `ece_own_map` | 2.67e-05 | 1.30e-05 | **32.8%** | 22.4% | 24.3% | 49.4% | no |
| `shift` | 4.59e-05 | 2.23e-04 | **82.9%** | 82.3% | 68.1% | 88.8% | no |
| `slope` | 5.93e-04 | 6.86e-03 | **92.0%** | 91.9% | 71.4% | 90.3% | **YES** |
| `rms_residual` | 1.99e-05 | 3.89e-05 | **66.1%** | 63.6% | 53.6% | 84.3% | no |

> **v2 §2.2 precedent, for comparison: 98.7% (delay), 67.9% (shortage), 70.8% (impact)** — AUC,
> on the backbone-seed axis.

### The comparison invites a wrong conclusion, so the caveats come before the reading

**Taken at face value, the agreement is striking**: `ece_raw` at 87.3% / 69.4%, `rms_residual` at
70.8% / 66.1%, sitting right on top of §2.2's 67.9% and 70.8%. That reading does not survive
three things, all measured rather than argued.

1. **Nine of ten cells fail their null.** The null asks what `pct_world` looks like when world
   identity explains *nothing* — labels resampled from one pooled curve, once per world, shared
   across that world's five members exactly as real labels are. It comes back at a **mean of
   24–74% and a p95 of 48–92%**. A `pct_world` in the 60–90% band is simply what five worlds and
   five init seeds produce on their own. Only Variant E's `slope` clears (92.0% against a p95 of
   90.3%) — 1 of 10 tests, marginally, which is close to what 10 tests at α=0.05 yield by chance.
2. **The second axis is the head init seed, not the backbone seed** — carried verbatim from
   `layer3_uncertainty_aware.md` §3 and forced by the same constraint (only `m0` checkpoints
   exist; building `m1..m4` means training SHARE). §2.2's `var_model` is backbone-seed variance,
   the **larger** of the two model-side sources. Substituting the smaller one inflates
   `pct_world`, so **every percentage above is an upper bound on the world share as §2.2 would
   have measured it**, and is not substitutable into the 98.7/67.9/70.8 series.
3. **`Var_d(mean_m ·)` is biased upward** by `var_model / M`. §2.2 used it, so it is reported
   first for comparability; the corrected column is beside it. The correction is small here
   (≤ 3 points except on `ece_own_map`) precisely *because* the init-seed axis is so small —
   which is caveat 2 showing up again.

### What Step 5 establishes

**The world-variance hypothesis is not confirmed at this scale.** The percentages are consistent
with world identity dominating; they are equally consistent with world identity explaining
nothing, and the measurement cannot separate those at five worlds. This is a **non-detection at
this scale, not an established absence** — with five worlds, the between-world term is a variance
estimated from five points, and no amount of care in the estimator fixes that.

It also flags, without resolving, a question about the precedent: §2.2's 98.7/67.9/70.8 were
computed with this estimator on this benchmark family, and whether they were gated against a null
of this kind is not recoverable, because the report is not in this repository (see above). The
98.7% figure is high enough that a null is unlikely to touch it; 67.9% and 70.8% sit inside the
band this session's null produces from pure noise. That is an observation about what can be
checked here, not a claim about §2.2's conclusion.

**Consequence for Step 7, stated before Step 7 is read:** Step 5 does **not** motivate
world-conditioning. Combined with Step 4 — where Variant A's curves are indistinguishable from a
common curve — the honest prior going into Step 7 is that conditioning has little or nothing to
capture on A, and possibly something real on E.

---

## STEP 6 — cited, not re-run

Step 6 is exactly Step 3's 20 leave-one-world-out cross-world transfer test, and it is
**referenced rather than repeated**, per the build prompt:

> Cross-world isotonic makes ECE **worse**: **−0.0054** (A), 7/20 folds improved; **−0.0029**
> (E), 8/20 folds improved, against member floors of **0.0132** / **0.0117**. Per-fold
> improvement ranges −0.0288…+0.0194 (A) and −0.0295…+0.0240 (E). Classified **STOP — G**.
> — `reports/layer3_uncertainty_aware.md` STEP 3

That baseline is reproduced inside this session's own leave-one-world-out harness as the
`step3_pairwise` arm (the four single-world maps applied to the held-out world, averaged), which
aggregates to exactly those figures — see the 7b table below. It is carried as the comparison
point, not measured afresh.

---

## STEP 7 — world-conditioned calibration

Two readings of "world-conditioned", built separately because they have completely different
status.

### 7a — conditioned on dataset-seed identity. **DIAGNOSTIC ONLY. NOT A DEPLOYABLE FIX.**

This is stated here in the body, not in a footnote, because it is the sentence a reader skimming
for *"did world-conditioning work?"* must not miss: **7a cannot be shipped, and nothing
downstream may treat it as if it could.** It conditions on which of the five known dataset seeds
a sample came from. A sixth, unseen world has no seed ID, so 7a has **no defined behaviour on it
at all** — not a degraded behaviour, an undefined one. It can only ever be evaluated in-sample
over these five worlds.

| arm | mean ECE (A) | improvement (A) | mean ECE (E) | improvement (E) |
|---|---|---|---|---|
| raw (uncalibrated) | 0.0407 | — | 0.0377 | — |
| **per-world isotonic** | **0.0205** | **+0.0202** | **0.0153** | **+0.0224** |
| **pooled head, one-hot world ID** | **0.0285** | **+0.0121** | **0.0355** | **+0.0022** |

**The per-world isotonic row reproduces Step 3's within-world leaky bound exactly** (+0.0202 /
+0.0224) — as it must; it is the same object, recomputed here so 7a and 7b sit on one table.

**Two readings, and the second is the honest one.**

The reading the build prompt anticipated is that 7a recovering the ceiling *confirms the Step 5
world-variance hypothesis*. It does not, and the reason is structural rather than a matter of
degree. A per-world isotonic map fitted and evaluated on the same world removes **all** of that
world's miscalibration — the part that is a per-world property and the part that is ordinary
in-sample overfitting to n≈4,100 rows. It does not isolate a world effect, and it would post the
same +0.02 if every world shared one calibration curve. Step 3 already labelled this arm
"optimistically biased on purpose; an upper bound, never a gate", and that label is what governs
here. With Step 5's percentages failing their null and Step 4's Variant A curves
indistinguishable from a common curve, **the confirmation the prompt was looking for is not
available from 7a, and this report does not claim it.**

What 7a *does* establish is new and worth having: **the one-hot head row measures how much of the
ceiling an affine per-world map can reach**, and it splits by variant exactly as Step 4 predicted
it would. On Variant A a per-world Platt-form map recovers **60%** of the isotonic ceiling
(+0.0121 of +0.0202) — an affine story largely suffices. On Variant E it recovers **10%**
(+0.0022 of +0.0224) — an affine story does not suffice, which is the same conclusion Step 4
reached from curve shape by an entirely independent route. Two different measurements agreeing on
the affine/non-affine split is the most solid thing in this section.

### 7b — conditioned on an observable graph-computed proxy. **Deployable, and tested as such.**

`ml/world_summary_features.py` computes five summary statistics **off the graph itself**, at the
point in the pipeline a deployment would have them — the test-split snapshots. Nothing reads
`resolved_config.json`, a seed, the generator namespace, or any label.

| feature | reads | Variant A range | Variant E range |
|---|---|---|---|
| `mean_on_time_180d` | Supplier `x`, trailing 180d on-time rate | 0.4007–0.4601 | 0.3644–0.4363 |
| `std_on_time_90d` | Supplier `x`, trailing 90d on-time rate | 0.3861–0.4039 | 0.3879–0.4182 |
| `mean_lateness_variance` | Supplier `x`, lateness variance | 4.617–6.309 | 1.759–2.635 |
| `upstream_edge_density` | `(Supplier, UPSTREAM_OF, Supplier)` / n | 1.965–1.994 | **absent** |
| `supplier_shipment_degree` | `(Shipment, SHIPS_FROM, Supplier)` / n | 19.55–21.29 | 17.07–17.43 |

Two findings fell out of building this, both worth recording:

- **A negative control validates the column indexing.** `capacity_score` and `lead_time_days`
  look like obvious world summaries and are useless as such: the loader z-scores them *within
  each world* (`ml/data/loader.py::_prepare_static_features`), so their world-level mean is 0 by
  construction. Measured: `max|mean_capacity_score_z| = 8.75e-09` (A), `7.15e-09` (E). The
  pipeline is reading the columns it thinks it is.
- **Variant E emits no `UPSTREAM_OF` relation at all** — `supplier_upstream.csv.gz` does not
  exist in any `vE_seed*` directory, so the disruption-propagation edge density is structurally
  absent, not merely constant. The feature is dropped on E (4 features used, not 5) rather than
  carried as an unidentified coefficient.

The conditional calibrator is a Platt-form head on `[1, z, φ, z·φ]` with `z = logit(p)` and φ the
standardised summary vector. The `z·φ` interaction block is what makes it *conditional* rather
than a global map with extra offsets — without it the features could only shift the curve, and
Step 4 says shift is not what the between-world difference is made of. The ridge penalty is not
optional: **four training worlds supply at most four distinct values of φ**, so the conditioning
block is near rank-deficient by construction (measured: the centred 5×5 world-feature matrix has
rank 4; a fit on four worlds can identify at most three centred directions). The penalty is
selected by **inner leave-one-world-out over the four training worlds only** — the held-out
world's predictions, labels, and features are untouched during selection.

**Evaluated under Step 3's own protocol:** fit on four worlds, test on the held-out fifth, rotate
over all five.

#### Two controls the comparison cannot do without

7b differs from Step 3's baseline in **two** ways at once — it conditions on features, *and* it
pools four worlds instead of fitting on one. So the same four pooled worlds are also run with **no
conditioning at all** (`pool4_isotonic`, `pool4_platt`). Without those, any improvement over Step
3 could be entirely the pooling.

#### Results — 5 leave-one-world-out folds

Test rows per held-out world: A — 3,924 / 4,230 / 4,074 / 4,146 / 4,164 (positives 2,082 / 2,312 /
2,191 / 2,212 / 2,278); E — 4,800 each (positives 2,446 / 2,460 / 2,441 / 2,450 / 2,438).

| arm | ECE | Brier | AUC | impr. vs raw | folds impr. |
|---|---|---|---|---|---|
| **Variant A** | | | | | |
| raw | 0.0407 | 0.2346 | 0.6435 | — | — |
| `step3_pairwise` (Step 3's baseline) | 0.0461 | 0.2349 | 0.6403 | **−0.0054** | 1/5 |
| `pool4_isotonic` *(control, unconditional)* | 0.0341 | 0.2337 | 0.6420 | +0.0066 | 3/5 |
| `pool4_platt` *(control, unconditional)* | **0.0325** | 0.2338 | 0.6435 | **+0.0081** | 4/5 |
| **`cond_full`** (5 features) | 0.0346 | 0.2339 | 0.6435 | +0.0061 | 3/5 |
| **`cond_single`** (`mean_on_time_180d`) | 0.0374 | 0.2343 | 0.6435 | +0.0033 | 4/5 |
| **Variant E** | | | | | |
| raw | 0.0377 | 0.2314 | 0.6525 | — | — |
| `step3_pairwise` (Step 3's baseline) | 0.0406 | 0.2326 | 0.6494 | **−0.0029** | 2/5 |
| `pool4_isotonic` *(control, unconditional)* | 0.0354 | 0.2312 | 0.6514 | +0.0023 | 3/5 |
| `pool4_platt` *(control, unconditional)* | 0.0387 | 0.2316 | 0.6525 | −0.0010 | 2/5 |
| **`cond_full`** (4 features) | 0.0576 | 0.2371 | 0.6525 | **−0.0200** | 1/5 |
| **`cond_single`** | 0.0414 | 0.2319 | 0.6525 | −0.0037 | 2/5 |

`step3_pairwise` aggregates to −0.0054 and −0.0029, reproducing Step 3's 20-fold figures exactly,
which confirms the harness is measuring the same object.

#### Does the conditioning add anything? — the decisive contrast

Each conditioned arm against each unconditioned control, on the same folds, with the difference
given **its own** floor over the same 3-of-5 init-seed subsets:

| contrast | Δ improvement | own floor | conditioning adds? |
|---|---|---|---|
| **A** `cond_full` − `pool4_platt` | **−0.0021** | 0.0022 | **no** |
| **A** `cond_full` − `pool4_isotonic` | −0.0005 | 0.0047 | **no** |
| **A** `cond_single` − `pool4_platt` | −0.0049 | 0.0024 | **no** |
| **A** `cond_single` − `pool4_isotonic` | −0.0033 | 0.0045 | **no** |
| **E** `cond_full` − `pool4_platt` | **−0.0189** | 0.0145 | **no** (significantly worse) |
| **E** `cond_full` − `pool4_isotonic` | −0.0222 | 0.0178 | **no** (significantly worse) |
| **E** `cond_single` − `pool4_platt` | −0.0027 | 0.0014 | **no** (significantly worse) |
| **E** `cond_single` − `pool4_isotonic` | −0.0060 | 0.0060 | **no** |

**Every contrast is negative.** On Variant A conditioning is indistinguishable from doing nothing
(−0.0021 against a floor of 0.0022); on Variant E it is measurably *harmful*, catastrophically so
on one fold (`cond_full`, held-out world 46: **−0.0888**, despite the inner selection raising the
penalty to λ=30 for that fold — the instability the rank deficiency predicts).

#### Which of the two failure modes is it?

7b failing has two possible causes needing different follow-ups: *the features carry no
information about calibration*, or *four worlds cannot identify a dependence on them even if one
exists*. Asked directly — correlate each feature, across the five worlds, with three descriptions
of that world's calibration, with **exact** permutation p-values over all 120 orderings:

| | Variant A | Variant E |
|---|---|---|
| tests run | 15 | 12 |
| nominally significant at p<0.05 | **1** (`supplier_shipment_degree` vs `ece_raw`, r=+0.955, p=0.025) | **0** |
| survives Bonferroni (α/15 = 0.0033) | **no** | — |
| replicates on the other variant | **no** — r = −0.305 (p=0.592), opposite sign | — |

At n=5 worlds, |r| must reach ~0.878 for p<0.05, so this test has very little power; one nominal
hit in 15 is close to what chance delivers, it does not survive multiplicity correction, and it
reverses sign on the other variant. **The two failure modes are therefore not separated by this
measurement** — which is itself the reportable result. Both remain live, and both are
consequences of having five worlds.

---

## Adopt-only-if-gated

Per the standing rule, an improvement over Step 3's cross-world baseline is **not** adoption.
7b's metric gets its own freshly measured floor; Step 3's member floor (0.0132 / 0.0117) and the
within-world ceiling from Step 3 or 7a are answers to different questions and are **not**
inherited.

**Refit floor — 10 identically configured refits, nothing changed: `max|dev| = 0.00000000`, both
variants.** This is reported as a fact, not as a result: IRLS is deterministic given `(X, y, λ)`,
with no random initialisation, so an exactly-zero floor is a property of the solver. Reading it
as "the metric is stable" would be exactly the error Step 3 avoided when it reported its own fit
floor at 0.000000 and then gated on the member floor instead. Same here.

**Member floor — the operative one.** The entire 5-fold 7b pipeline re-run once per 3-of-5
head-init-seed subset (10 refits), floor = maximum pairwise deviation of the resulting
improvement.

| arm | impr. vs raw | vs Step 3 baseline | **own member floor** | **clears own floor** | beats Step 3 |
|---|---|---|---|---|---|
| **Variant A** | | | | | |
| `cond_full` | +0.0061 | +0.0115 | 0.0036 | **YES** | YES |
| `cond_single` | +0.0033 | +0.0087 | 0.0036 | no | YES |
| `pool4_isotonic` *(control)* | +0.0066 | +0.0120 | 0.0058 | **YES** | YES |
| `pool4_platt` *(control)* | **+0.0081** | +0.0135 | 0.0030 | **YES** | YES |
| **Variant E** | | | | | |
| `cond_full` | −0.0200 | −0.0171 | 0.0145 | no | no |
| `cond_single` | −0.0037 | −0.0009 | 0.0012 | no | no |
| `pool4_isotonic` *(control)* | +0.0023 | +0.0052 | 0.0059 | no | YES |
| `pool4_platt` *(control)* | −0.0010 | +0.0018 | 0.0007 | no | YES |

### The gated verdict on 7b

**7b is not a usable calibration fix, and the reason is not the one the floor test alone would
suggest.** On Variant A `cond_full` does clear its own freshly measured floor (+0.0061 vs 0.0036)
and does beat Step 3's baseline (+0.0115). Taken alone that would read as a pass. It is not one,
for two reasons that the controls and the second variant make unavoidable:

1. **The conditioning contributes nothing.** The same four worlds pooled with **no conditioning
   at all** score +0.0081 — better than `cond_full` — and the contrast between them (−0.0021
   against a floor of 0.0022) says conditioning is indistinguishable from doing nothing. Whatever
   is working on Variant A, it is the pooling, not the world features.
2. **It fails on Variant E**, where every conditioned arm is worse than raw and the conditioning
   contrast is significantly negative. Nothing here is sign-consistent across variants, and this
   project does not adopt on one variant.

So the outcome is the combination the build prompt named in advance, with one amendment.
**No observable proxy tested here captures the regime that matters, and finding one is
explicitly flagged as open, not silently dropped.** The amendment is to the other half: the
prompt's fallback phrasing was "7a succeeds while 7b fails", and 7a's success does not carry the
diagnostic weight that phrasing assumes, for the reason given in the 7a section — a within-world
isotonic fit would post the same +0.02 whether or not the miscalibration is a per-world property,
and Step 5 could not demonstrate that it is.

### The one thing that did clear, reported as a follow-up and not as a fix

**`pool4_platt` on Variant A: +0.0081 against a freshly measured floor of 0.0030, 4/5 folds
improved** — an unconditional map pooled over four worlds, where Step 3's one-world map *harmed*
ECE by −0.0054. This was a control, not the experiment, and it is precisely what Step 3's own
defensible-follow-up #1 predicted:

> more worlds would establish whether a map **pooled** across many worlds transfers, which is a
> different and untested proposition from the pairwise rotation run here.

This session tested that proposition at four worlds. **It does not clear this project's bar**: it
is not sign-consistent (4/5 folds, not 5/5), and on Variant E the same arm returns −0.0010. It is
reported as the highest-value follow-up available from this work, and adopting it on the Variant
A number alone would be exactly the single-variant, single-arm inference this report has declined
to make everywhere else.

---

## Gate classification

| state | Case | Step | **Gate** | change from `layer3_uncertainty_aware.md` |
|---|---|---|---|---|
| **Supply Stress** (A) | Case 1 | Step 3 | **STOP — G** | **unchanged** — clearer diagnosis, no route opened |
| **Supply Stress** (E) | Case 1 | Step 3 | **STOP — G** | **unchanged** — clearer diagnosis, no route opened |

**Supply Stress stays at STOP — G on both variants.** No state ends this session with a
calibrated Layer-3 output. Classification remains **G (calibration/uncertainty failure)** and not
F: the underlying fits are stable — the refit floor is exactly 0.000000, Steps 1–3 reproduce
bit-exactly, and `assert_backbone_frozen()` never fired.

What has changed is the diagnosis. Step 3 named the failure "the calibration map does not
transfer between worlds" and attributed it to per-world miscalibration in different directions.
This session tested that attribution directly and **could not confirm it**:

- Variant A's five per-world curves are **not distinguishable from a single common curve** at
  0.80–1.02× a matched null (Step 4). Variant E's are (1.48–1.83×), and are non-affine.
- World identity's share of curve-position variance (22.6–92.0%) **fails a null in which world
  identity explains nothing** in 9 of 10 cells (Step 5).
- Conditioning on world identity (7a) recovers the ceiling only in the one form that is
  optimistically biased by construction, and an affine per-world map recovers 60% / 10%.
- Conditioning on an observable proxy (7b) **adds nothing over unconditional pooling on either
  variant**, and is measurably harmful on E.

**All of this is non-detection at this scale, not an established absence** — the same framing
every other result in this project carries. Five worlds is the binding constraint on every
negative above, and it is the same constraint Step 3 identified. The per-world differences may be
real and simply unmeasurable at five worlds; the summary features may be informative and simply
unidentifiable from four training points. What this session establishes is that **neither can be
demonstrated at this scale**, and that the sharper hypothesis worth testing next is not
conditioning at all.

### What must not be built on this evidence

- **7a is not a deployable calibrator** and must not be reported, reused, or benchmarked as one.
  It has no defined behaviour on an unseen world.
- **No world-conditioned calibration head**, on these features or a richer set, at five worlds.
  The conditioning block cannot be identified from four training worlds, and the contrast test
  says it adds nothing when it can be fitted at all.
- **No adoption of `pool4_platt`** on the Variant A number. It fails on E and is not
  sign-consistent.
- **No within-world calibration fit**, which remains the one change that would make Step 3
  "pass" and remains leakage.

### The defensible follow-up this report supports

**One, and it is a re-measurement, not new architecture.** Generate more dataset seeds and re-ask
two questions that are now sharply posed and both currently answerable only as non-detections:

1. **Does an *unconditional* map pooled over many worlds transfer?** `pool4_platt` at four worlds
   gives +0.0081 on A (floor 0.0030) and −0.0010 on E. The pooling arm, not the conditioning arm,
   is where the signal is. This is Step 3's follow-up #1, now with a measured four-world data
   point and its own floor.
2. **Do the per-world curves actually differ?** Step 4's same-curve null and Step 5's variance
   null are both sample-size-limited in the same way. More worlds raise the power of both
   directly, and both are re-runnable as-is — `ml/calibration_curve_compare.py` and
   `ml/calibration_variance_decompose.py` take `--seeds`.

Conditioning on observable world summaries is **not** among the follow-ups this report supports
at any scale reachable from here, unless (2) first shows the per-world differences are real.

---

## STEP 6 (integrity) — SHARE, the Markov readout, and Model A's heads are untouched

**Ground rule:** SHARE and the Layer-2 Markov readout are not retrained or modified by this
session. Confirmed by `git diff` at the close, as required.

```
$ git diff --stat                     # EMPTY — no tracked file modified
$ git status --porcelain db/          # EMPTY — the generator was read, never edited
$ git status --porcelain
?? ml/calibration_curve_compare.py        <- new, this session
?? ml/calibration_grid_cache.py           <- new, this session
?? ml/calibration_variance_decompose.py   <- new, this session
?? ml/world_conditioned_calibration.py    <- new, this session
?? ml/world_summary_features.py           <- new, this session
?? ml/identifiability_check.py            <- prior session
?? ml/layer3_sufficiency.py               <- prior session
?? ml/layer3_uncertainty.py               <- prior session
?? ml/modelB_concat_head.py               <- prior session
?? reports/layer3_uncertainty_aware.md
?? reports/phase_modelB_concat_baseline.md
```

Every file this session touched is **new and untracked**. Nothing inherited was modified — the
five reused modules carry digests identical to those recorded at the close of
`reports/layer3_uncertainty_aware.md` STEP 6:

```
c2e15e39cdc101d4...  ml/hypothesis_ranker.py       MATCHES layer3_uncertainty_aware.md
24ccd9d58eb44b69...  ml/uncertainty_calibrate.py   MATCHES
3a8c888040446aa2...  ml/uncertainty_ensemble.py    MATCHES
8e4a0803d8f09cd0...  ml/latent_state_head.py       MATCHES
4e9a4db9d2d4ef3a...  ml/data/loader.py             MATCHES
689962edd1315599...  ml/ds_backbone.py             MATCHES
1b85a68f9e9705be...  ml/models/rgcn_attn_encoder.py         MATCHES
9e7919e6bfcf39ca...  ml/models/rgcn_attn_markov_encoder.py  MATCHES
cbd40aadc8570745...  ml/models/heads.py                     MATCHES
8f669a1683117280...  ml/models/encoder.py                   MATCHES
8fa14bc270946507...  ml/models/depth.py                     MATCHES
4cf67f7d4f25b3a8...  ml/models/model.py                     MATCHES

e25b3357ac81a896...  ml/layer3_uncertainty.py      unmodified this session (Step 3's module)
```

All ten backbone checkpoints in `out/ds_ckpt/` carry mtimes of **2026-08-13 22:03–22:44**,
predating both this session (2026-08-15) and the previous one. Every grid build loaded from
cache; **no backbone was trained**. `assert_backbone_frozen()` runs inside `build_grid` after
head construction and after training on every head fit, and never fired across the **50** head
fits (2 variants × 5 dataset seeds × 5 init seeds) this session performed. All other work in this
session — Steps 4, 5, 7a, 7b, every floor and every null — reads the cached prediction grid and
performs no model fitting of any kind.

**Timing, per the standing ground rule to measure before committing to a grid.** The first
single-world grid build was timed at **11.1 s** before the full 5×2 grid was launched (total
2 min per full rebuild). The new fitting loop (7b's IRLS) was timed at **2.4 s** for both
variants' 5 folds before the 20-refit floor campaign was committed to.

---

## Reproduction

```
$ venv/bin/python -u ml/world_summary_features.py       --out out/wcc/world_features.json
$ venv/bin/python -u ml/calibration_curve_compare.py    --out out/wcc/step4_curves.json
$ venv/bin/python -u ml/calibration_variance_decompose.py --out out/wcc/step5_variance.json
$ venv/bin/python -u ml/world_conditioned_calibration.py --out out/wcc/step7.json
```

The Steps 1–3 reproduction check rebuilds the grid with `load_grid(..., rebuild=True)` and diffs
`step3` / `step3_diagnostics` against `out/layer3/step345_{A,E}.json`; its output is
`out/wcc/repro_step3_{A,E}.json`.

Raw results: `out/wcc/step4_curves.json`, `out/wcc/step5_variance.json`, `out/wcc/step7.json`,
`out/wcc/world_features.json`, `out/wcc/repro_step3_{A,E}.json`.
Cached prediction grid: `out/wcc/grid_{A,E}_supply_stress.npz`.
Step 3 comparison numbers: `out/layer3/step345_{A,E}.json`.
