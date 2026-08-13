# Decision-Support Build — Counterfactual Reasoning, Uncertainty Estimation, Explainable HADES

**What this session did.** Built and trained real components for the three decision-support
capabilities `findings/evolution.md` §5 names as the direction after Layer 3 was scrapped, in
three independently-gated phases, and reported each against its own stop condition. This is a
build-and-measure session, not a design document — `docs/Decision_Support_Architecture_Spec.md`
is the design; nothing here inherits its numbers, and where a measurement contradicts an
assumption made there, the measurement wins and is called out.

**The architecture was not touched.** SHARE and the Markov Blanket depth readout were reused
exactly as trained — `delay -> h^1`, `shortage -> h^3`, `impact -> h^4`, zero-parameter
index-select, no depth adaptivity anywhere. `ml/ds_backbone.py::freeze` asserts every backbone
parameter has `requires_grad = False` before any phase runs, and `git diff` reports no change to
`ml/models/rgcn_attn_encoder.py`, `ml/models/rgcn_attn_markov_encoder.py`, `ml/models/heads.py`
or `ml/train.py` at the close of the session.

**Scale and cost.** Variant 0 at the `v1` preset (`sup_n=800`, 15 snapshots, 40/20/40 temporal
split) — the cheap, already-validated anchor the brief specifies. One training is **251 s**
uncontended (347 s at 2-way concurrency), **752,211 parameters**, which matches
`HADES_v1/reports/info.md` §4's recorded count for this arm exactly and is the first sanity
anchor of the session. The 5x5 backbone grid is **25 trainings ≈ 2.4 CPU-hours**; every phase
reads from that one cache rather than retraining.

---

## Headline results, per phase

| phase | stop condition | outcome |
|---|---|---|
| **1 — Counterfactual** | naive predicted deltas fail to track true deltas → report and open Phase 1b | **HIT (clean miss).** Naive sign agreement **0.478 over 15 runs** against chance 0.500 — *below* chance, and the single-seed 0.524 sits inside a measured model-seed floor of **0.324**, 13x the apparent effect. **Phase 1b built and also failed**: the interventionally-supervised head scores **0.402**, losing 13 of 15 paired folds to the naive baseline. |
| **2 — Uncertainty** | none (report the variance split) | **Delivered.** Dataset-seed variance is **98.7% / 67.9% / 70.8%** of total AUC variance on delay / shortage / impact. §5.6's delay finding replicates (8.7x vs 10.3x); its shortage figure is revised. |
| **3 — Explainability** | explanations failing the ablation test are reported as failures, not findings | **CLEARED, modestly.** Faithfulness pass rate **0.166** against a chance level of 0.100, above chance on **5/5** seeds — but measured only on the **3.4%** of predictions that are not saturated. |

---

## 1. Phase 1 — Counterfactual Reasoning

### 1.1 What was built

- **`ml/counterfactual_edit.py`** — applies a structural edit to a snapshot's `HeteroData` and
  re-runs the **existing, unmodified** forward pass. No new architecture: SHARE's parameters are
  indexed by node type and relation type, never by node identity or edge count, so adding or
  deleting edges changes only which messages are built and how the per-destination softmax
  normalises. Edits are mirrored into the `ToUndirected` twin relation, and a validity pass
  checks for dangling indices after every edit.
- **`ml/counterfactual_ground_truth.py`** — the generator-side truth. Executes
  `db/generate_dataset.py` in-process, then recomputes its own causal chain
  (`stress -> p_delay -> p_impact`) under the intervened structure.
- **`ml/run_counterfactual_phase1.py`** — pairs predicted against true deltas, partitioned.
- **`ml/counterfactual_delta_head.py`** — Phase 1b, built because the stop condition was hit.

Three interventions, chosen because on Variant 0 they all act through the one live structural
channel: with `MECHS = ()`, `hp_coupling`, `upstream_stress` are zero and `recv_atten =
absorption = 1`, leaving `stress = own_stress + 0.35 x SUM own_stress(coparents)`. The co-parent
graph *is* the causal lever, and all three interventions move it — `add_dual_source`,
`remove_supplier`, `substitute_supplier`.

### 1.2 Why the ground truth is computed in closed form — measured, not assumed

The obvious approach is to re-run the generator with the intervention applied and diff the
labels. Before relying on that, it was tested: insert **one extra random draw** before the
shipment simulation — the smallest possible perturbation, standing in for what any intervention
does — and diff the emitted labels. Variant 0, `v1` preset, seed 42:

| quantity | effect of ONE extra draw |
|---|---|
| shipments differing in status or eta | **100.0%** (39,184 of 39,184) |
| **`delay`** labels flipped | **797 of 5,527 = 14.4%**, i.e. **179% of the 444 true positives** |
| `shortage` labels flipped | **0** |
| `impact` labels flipped | **0** |

**The answer is task-dependent, and this corrected an assumption carried into the session.** For
`delay`, re-simulation is unusable — a single draw's divergence flips nearly twice as many labels
as there are positives, so any intervention's realised effect would be buried in it. But for
`impact` and `shortage` the realised labels are **robust** to total schedule divergence, because
both are coarse aggregates: impact is "at least one in-flight shipment is delayed in the horizon",
which survives reshuffling for any supplier with several shipments. The blanket claim that
re-simulation cannot work here is wrong, and the module's rationale was rewritten to say so.

The closed-form route is still used, for two reasons that survive that correction. It is **exact**
rather than robust-by-saturation, and it yields a **continuous** probability delta rather than a
binary label flip — which matters decisively, because the true effects are a few points of risk on
a handful of suppliers, and a binary target at that effect size would have almost no power.

**The estimand, stated plainly.** Holding the realised schedule fixed (same shipments, same
dispatch times, same eta) and recomputing only what the intervention changes is common random
numbers in exact form. It measures the effect of the structural change on delay risk *through the
stress channel*, and excludes the second-order path where a structural change alters which
supplier is chosen for future replenishment. `substitute_supplier` and `remove_supplier` reassign
shipment ownership explicitly, capturing the dominant part of that path; the residual is a stated
limitation.

### 1.3 The result — a clean miss

900 interventions (60 of each kind x 5 dataset seeds), scored on the `impact` task over the
6-snapshot test block. **2,695 entity-snapshot pairs carried a nonzero true effect.**

| intervention | affected | sign agreement | excl. silent | silent | Spearman rho | magnitude ratio | false-effect rate |
|---|---|---|---|---|---|---|---|
| `add_dual_source` | 857 | **0.465** | 0.465 | 0.149 | −0.022 | 0.161 | 0.00079 |
| `remove_supplier` | 1,006 | 0.572 | 0.638 | 0.501 | −0.036 | 0.081 | 0.00018 |
| `substitute_supplier` | 832 | 0.546 | 0.577 | 0.394 | +0.203 | 0.158 | 0.00060 |
| **ALL** | **2,695** | **0.524** | 0.536 | 0.358 | +0.068 | **0.132** | 0.00052 |

*Chance is 0.500. "Silent" is the fraction of truly-affected entities where the model's predicted
delta is below 1e-4 — it said nothing at all. "Magnitude ratio" is mean |predicted| / mean |true|.
"False-effect rate" is over the bystander population, whose true effect is exactly zero.*

Per-seed sign agreement, and this is what decides the gate:

| kind | d42 | d43 | d44 | d45 | d46 | above chance |
|---|---|---|---|---|---|---|
| `add_dual_source` | 0.539 | 0.358 | 0.586 | 0.518 | 0.323 | **3/5** |
| `remove_supplier` | 0.449 | 0.544 | 0.633 | 0.529 | 0.704 | 4/5 |
| `substitute_supplier` | 0.543 | 0.531 | 0.528 | 0.636 | 0.493 | 4/5 |
| **ALL** | 0.502 | 0.471 | 0.589 | 0.554 | 0.507 | **4/5** |

**This is a clean miss, and it is the stop condition.** Four readings agree at this model seed,
and §1.3.1's floor then shows the residual edge is noise:

1. **Sign agreement is 0.524 against a chance level of 0.500** — a 2.4-point edge on the
   decision-relevant quantity, and §1.3.1 shows even that vanishes under model-seed averaging.
2. **Never 5/5 sign-consistent**, on any intervention type or pooled — the standard every other
   result in this project is held to. `add_dual_source` is *below* chance at 0.465, i.e. the
   model gets that intervention's direction wrong more often than right.
3. **Spearman is ~0**, and negative for two of the three types. The model does not rank *who* is
   affected, let alone by how much.
4. **Magnitude ratio 0.13** — predicted deltas are about **8x too small** — and the model is
   **silent on 36%** of truly-affected entities, rising to 50% on `remove_supplier`.

One thing the naive approach does get right: the **false-effect rate is ~0.0005**. Message passing
does spread the edit beyond the truly-affected set, but only barely; the model is not inventing
effects everywhere, it is failing to find the real ones.

### 1.3.1 The reproduction floor — which changes the headline

Sign agreement is a new metric, so its floor was measured rather than assumed: the identical
900-intervention grid was re-run on **two further model-init seeds**, giving 15 independent
measurements of the same quantity (3 model seeds x 5 dataset seeds, `out/cf_phase1_floor.json`).

| dataset seed | m0 | m1 | m2 | spread |
|---|---|---|---|---|
| 42 | 0.502 | 0.385 | 0.627 | 0.242 |
| 43 | 0.471 | 0.273 | 0.563 | 0.290 |
| 44 | 0.589 | 0.275 | 0.599 | **0.324** |
| 45 | 0.554 | 0.387 | 0.507 | 0.168 |
| 46 | 0.507 | 0.472 | 0.457 | 0.050 |

**Mean spread 0.215, max spread 0.324, from changing nothing but the model-init seed.** Across all
15 runs: min 0.273, max 0.627, **mean 0.4778**, std 0.109.

Two consequences, and the second supersedes §1.3's reading:

1. **The apparent edge is 13x smaller than the floor.** The 0.0245 excess over chance sits against
   a max-abs model-seed floor of 0.324. By the standard `reports/layer3_testing.md` §4 set — a
   delta below the floor is not evidence, and sign consistency across dataset seeds does not
   rescue it, because all seeds share the floor — there is nothing here.
2. **Averaged over model seeds, sign agreement is 0.4778 — *below* chance.** The 0.524 in §1.3 was
   an artifact of which model seed happened to be trained first. The honest statement is not
   "marginally above chance"; it is **at chance, with the apparent excess entirely inside
   model-init noise**.

Spearman behaves the same way and flips sign with the model seed: pooled rho is +0.068 at m0,
−0.31 to −0.17 at m1, and +0.16 to +0.38 at m2. A rank correlation whose *sign* depends on the
model seed is not measuring a relationship.

**This is the strongest single result in Phase 1**, and it only exists because the floor was
measured. Without it, the reported finding would have been "0.524 vs 0.500, weakly above chance",
which is wrong in substance as well as in confidence.

### 1.4 What this means

SHARE and the Markov readout were trained as ordinary supervised predictors, by focal loss on
observational snapshots, with no interventional signal anywhere. Handed an edited graph they
produce *an* answer — the forward pass runs perfectly well, no architectural change is needed —
but that answer does not track what the generator says would actually happen. The model estimates
`P(Y | X)` and the intervention asks for `P(Y | do(X))`, and on this benchmark those differ enough
that the associational answer is barely better than a coin flip on direction and uncorrelated on
magnitude.

**Non-detection at this scale, not an established absence.** This is Variant 0 at `sup_n=800`,
one task, three intervention types, 2,695 affected entity-snapshots. A larger or differently
structured world might behave differently, and the true effects here are small — mean |true delta|
is a few points of probability — which is exactly the regime where an associational model has
least to go on.

### 1.5 Phase 1b — the delta head, built because the miss was clean

Per the brief, a clean miss opens Phase 1b: a small head supervised **directly** on
generator-produced `(factual, intervened, true-delta)` triples, rather than relying on the frozen
forward pass to imply the effect. `ml/counterfactual_delta_head.py` reads the frozen Markov
embedding of each entity before and after the edit, their difference, the naive prediction, five
co-parent-graph reach features and a one-hot over intervention type; it emits a gate and a
magnitude, scored by a zero-inflated loss so a regressor cannot win by predicting zero everywhere.

Splits are leave-one-dataset-seed-out, and target-disjointness is **asserted** rather than assumed
— suppliers are world-specific, so a leave-one-world-out split shares no target entity by
construction, which is the property that stops the head memorising which suppliers matter.

**Result: the delta head is *worse* than the naive baseline it was built to improve on.**
15 folds (5 held-out worlds x 3 head-init seeds), 58,818 head parameters, ~270,000 training rows
per fold of which ~800 carry a nonzero effect.

| arm | mean sign agreement | range | wins |
|---|---|---|---|
| naive (frozen forward pass) | **0.5085** | 0.458 – 0.549 | 13/15 |
| CF delta head (interventional supervision) | **0.4016** | 0.112 – 0.545 | 2/15 |
| **paired delta (head − naive)** | **−0.1069** | | |

Per held-out world, averaged over the three head seeds:

| test world | d42 | d43 | d44 | d45 | d46 |
|---|---|---|---|---|---|
| head sign agreement | 0.483 | 0.490 | 0.463 | 0.419 | **0.155** |
| head-seed spread | 0.117 | 0.100 | 0.172 | 0.023 | 0.092 |

Magnitude is worse still: the ratio of predicted to true effect size has median **21.7** and ranges
over nine orders of magnitude (7e-06 to 1,344). Phase 1's naive predictions were 8x too *small*;
the head's are typically ~20x too *large* and occasionally absurd. Nothing about the regression is
converging to a stable scale.

**Why this is a stronger negative than Phase 1's, not a weaker one.** The head was given exactly
what Phase 1 established the frozen model lacks — a training signal that is interventional by
construction, the generator's own true effects — plus the frozen embeddings before and after the
edit, their difference, the naive prediction as a feature, and explicit co-parent reach features
that make the affected set nearly identifiable. It still does not transfer to a held-out world.

The most likely reason is visible in the corpus shape: **~800 nonzero-effect rows against ~270,000
total**, spread over three training worlds whose suppliers are unrelated to the test world's
(§2.2). The head has a few hundred examples of what a real effect looks like, and no entity
identity it can carry across worlds — which is the same wall `reports/layer3_testing.md` §9.8
described from the other side, that this benchmark's structure is distinguishable largely by
identity and identity does not generalise.

**Phase 1 verdict, both tiers together.** The naive approach is at chance
(0.478 over 15 runs, floor 0.324). The interventionally-supervised head is **below** it
(0.402, losing 13 of 15 paired folds). Counterfactual effect prediction does not work here by
either route at this scale — and unlike Phase 1's result, this one cannot be attributed to the
absence of a causal training signal, because that signal was supplied.

---

## 2. Phase 2 — Uncertainty Estimation

### 2.1 What was built

- **`ml/uncertainty_ensemble.py`** — the 5x5 grid (5 dataset seeds x 5 model-init seeds) and its
  two-way variance decomposition, with hash integrity checks that the config matches across all
  members, the data hash *differs* across worlds, and *matches* within a world. Without those,
  an ensemble silently containing a mis-seeded member produces a variance estimate that means
  nothing.
- **`ml/uncertainty_calibrate.py`** — reuses the isotonic-regression and ECE machinery already
  written and validated in `ml/hypothesis_ranker.py`, rather than reimplementing it. That code is
  checked by `ml/test_hypothesis_calibration.py` (13 checks, including ECE returning exactly 0.60
  on a 0.9-always predictor of a 30% class), so the metric layer arrives pre-validated.

### 2.2 The variance decomposition — the number this phase exists to produce

`reports/phase7_training_results.md` §5.6 measured dataset-seed and model-seed variance
*separately*, one axis at a time. The 5x5 grid allows the proper two-way decomposition by the law
of total variance: `var_model = mean_d Var_m(AUC)`, `var_dataset = Var_d(mean_m AUC)`.

| task | grand mean AUC | std_model | std_dataset | ratio | **dataset share of total variance** |
|---|---|---|---|---|---|
| delay | 0.7759 | 0.0047 | **0.0407** | **8.7x** | **98.7%** |
| shortage | 0.7877 | 0.0061 | 0.0089 | 1.5x | **67.9%** |
| impact | 0.9333 | 0.0076 | 0.0119 | 1.6x | **70.8%** |

Against §5.6's figures:

| task | std_model (§5.6 → here) | std_dataset (§5.6 → here) | ratio (§5.6 → here) | verdict |
|---|---|---|---|---|
| delay | 0.0037 → 0.0047 | 0.0376 → **0.0407** | 10.3x → **8.7x** | **confirms** |
| shortage | 0.0019 → **0.0061** | 0.0090 → 0.0089 | 4.8x → 1.5x | **revises** |
| impact | 0.0102 → 0.0076 | 0.0128 → 0.0119 | 1.3x → 1.6x | confirms |

**§5.6's headline replicates and its shortage figure is revised.** On delay the finding is solid:
dataset-seed std is 8.7x model-seed std, and **98.7% of all AUC variance is which world was
generated**. The shortage revision is instructive rather than contradictory — the dataset-seed std
matches almost exactly (0.0089 vs 0.0090), but the model-seed std is 3x larger here (0.0061 vs
0.0019), because §5.6 estimated it from a *single* dataset seed while this grid averages
within-world variance across five. A one-world estimate of model variance was optimistic.

**The conclusion the brief asked for, stated plainly: on every task, most predictive variance is
dataset-level, ranging from 68% to 99%.** An uncertainty estimate built from model-init ensembling
or MC Dropout alone would be measuring the smaller term throughout, and on delay it would be
capturing roughly one part in seventy-five of the total variance.

**One identifiability limit, and it is a trap rather than an obvious gap.** The dataset-seed term
is **not identifiable per entity**: it is a population-level quantity here, `var_model` is
per-entity and `var_dataset` is not, and `ml/uncertainty_ensemble.py` returns that caveat as a
field rather than letting a consumer assume otherwise.

What makes it a trap is that the data *looks* pairable and is not. The generator derives primary
keys as UUIDv5 from stable keys (`docs/08_Backend_Design.md` §3), and a supplier's stable key is
its index — so **supplier ids are 100% shared across dataset seeds**. Joining seed 42 to seed 43
on `id` succeeds for all 800 suppliers and yields a per-entity "dataset variance" that means
nothing, because the entity behind the id is not the same business. Measured on seeds 42 vs 43:

| attribute | agrees across worlds |
|---|---|
| `country` | 19.5% (chance is 16.7% for six countries) |
| `lead_time_days` | 2.2% |
| `capacity_score` | 0.0% |
| `reliability_history` | 0.0% |

The identifier persists; the supplier does not. This was found because Phase 1b's split assertion
— written on the assumption that suppliers are world-specific — failed, and the assumption turned
out to be backwards. The assertion was replaced with one on the property that actually holds
(attribute divergence below chance), which is also the check that would catch genuine leakage if
a future generator change made entities persist.

### 2.3 Calibration — and which component is doing the work

Protocol matches `reports/layer3_testing.md` §10.3: the isotonic map is fitted on a **held-out
world** and evaluated on a **different** held-out world, rotated over all 20 ordered
(calibration, test) pairs. Fitting within-world would be optimistic in exactly the place §5.6 says
the variance lives.

| task | single raw | single + isotonic | ensemble raw | ensemble + isotonic | member floor (max abs) |
|---|---|---|---|---|---|
| delay | 0.2527 | **0.0277** | 0.2505 | 0.0291 | 0.0102 |
| shortage | 0.4362 | 0.0321 | 0.4297 | **0.0259** | 0.0120 |
| impact | 0.1166 | 0.0102 | 0.1206 | **0.0087** | 0.0039 |

*ECE, equal-count bins, mean over 20 folds.*

**Two findings, and the second is the more useful one.**

*Recalibration is transformative.* Raw ECE is 0.12–0.44 — focal loss, which deliberately
down-weights easy negatives, leaves probabilities badly miscalibrated. Isotonic drops it by
**9x to 17x**, far beyond any floor. Discrimination is essentially preserved (ensemble raw AUC
0.780 / 0.792 / 0.939), with the small loss from isotonic's tie-merging behaving exactly as
`ml/test_hypothesis_calibration.py` asserts it must.

*Ensembling does not clear its own floor on calibration.* Ensemble-plus-isotonic beats
single-plus-isotonic by **+0.0062** on shortage (floor 0.0120), **+0.0015** on impact (floor
0.0039), and is **worse** on delay (−0.0014, floor 0.0102). Not one of those clears. **On this
benchmark, at this scale, recalibration is doing the work and model-init ensembling is not** —
which is consistent with §2.2, since a model-init ensemble is averaging over the axis that carries
1.3%–32% of the variance.

**Floors, and which kind is operative.** The *fit* floor — refitting isotonic on fixed member
predictions — is **exactly 0.000000** on all three tasks, reproducing the pattern
`reports/layer3_testing.md` §10.3.1 found: PAVA is deterministic, so repeating it proves nothing.
The operative floor is **member composition** (which model seeds compose the ensemble), measured
over all 3-of-5 subsets at 0.0039–0.0120. That distinction is the difference between a floor that
looks impressive and one that means something.

---

## 3. Phase 3 — Explainable HADES

### 3.1 What was built

- **`ml/explain_prediction.py`** — occlusion attribution over the model's own input features:
  set feature `i` of the target entity to a neutral baseline (the per-node-type **median**, not
  zero, since these features are z-scored and clipped counts where zero is a specific and often
  extreme value), re-run the frozen forward pass, and take the drop in predicted risk.
- **`ml/test_explanation_faithfulness.py`** — the mandatory gate: ablate the cited factor, confirm
  the prediction moves in the claimed direction, by the claimed magnitude, and by more than a null.

Attention is **not used at all**. SHARE's shared, relation-blind scorer produces a per-edge weight
distribution that looks exactly like an explanation, and this project has flagged repeatedly that
such weights are not one until checked. The candidate factors are input features only.

### 3.2 Two corrections that were the substance of this phase

**(a) The highest-risk predictions cannot be explained by occlusion at all.** The first
implementation targeted the top-K entities by predicted risk — what a planner would actually open
— and returned **zero explanations across every seed**. The cause is not a bug in the attribution:
**96.5% of `delay` predictions sit at p ≥ 0.999**, and a saturated sigmoid does not move when one
input feature is occluded, so every attribution there is identically zero. The model is most
confident exactly where occlusion can say least about it. Targeting was changed to a stratified
draw across the non-saturated band, and the saturation fraction is reported as a first-class
limitation: **only 3.4% of predictions are in a range where this method can operate.**

**(b) The null in a faithfulness test does all the work, and two natural choices are degenerate.**
Three formulations were tried before one was defensible:

| null formulation | measured pass rate | why it is wrong |
|---|---|---|
| other features of the **same** entity | — | **circular**: the cited factor is the argmax of occlusion attribution and the ablation is the same operation, so it wins by construction |
| pooled 99th percentile across **all** entities | **0.004** | mis-scaled: one global threshold set by the most ablation-sensitive entities, which an insensitive prediction can never clear however faithful its explanation |
| same feature on **random** other entities | **1.000** | vacuous: the pool is 96% saturated, so the null collapses to ~0 and everything passes |
| same feature on other **non-saturated** entities | **0.166** | matched population, non-circular — the one reported |

The first three would each have produced a confident headline (0.4% or 100%) that was a property
of the threshold rather than of the explanations. The saturation finding in (a) is what makes the
third fail, so the two corrections are connected.

### 3.3 The result

`delay` task, Variant 0, 5 dataset seeds, 40 stratified targets per snapshot,
**1,142 explanations**:

| seed | 42 | 43 | 44 | 45 | 46 | **mean** |
|---|---|---|---|---|---|---|
| faithfulness pass rate | 0.125 | 0.150 | 0.229 | 0.209 | 0.117 | **0.166** |

**Chance is 0.100** — a 90th-percentile null is exceeded 10% of the time by construction. The pass
rate is above chance on **5 of 5 seeds**, which is this project's standard, and at 1,142
explanations the binomial standard error at p = 0.1 is 0.009, so 0.166 sits about 7 SE above
chance. **The gate is cleared, and modestly: 1.66x chance.**

Direction and magnitude both pass at **100%**, and that is close to tautological rather than
impressive — the attribution and the faithfulness ablation are the same operation, differing only
in that attribution occludes the target set jointly while the ablation occludes one entity. The
specificity test is the only informative component, which is why the null formulation mattered
so much.

**Which factors get cited, and which survive:**

| feature | cited | passed | pass rate |
|---|---|---|---|
| `days_since_dispatch` | 841 | 105 | 12.5% |
| `status_in_transit` | 215 | 51 | 23.7% |
| `carrier_on_time_rate_90d` | 47 | 3 | 6.4% |
| `status_scheduled` | 27 | 21 | **77.8%** |
| `status_delayed` | 6 | 6 | 100% |
| `status_delivered` | 6 | 1 | 16.7% |

**The most-cited factor is among the least specific.** `days_since_dispatch` accounts for 74% of
all citations — it is the dominant driver of delay risk, which is mechanically sensible, a shipment
long dispatched and not yet delivered is late — but it passes the specificity test only 12.5% of
the time, because occluding it moves *most* comparable shipments too. The rare citations are the
specific ones. A system that reported only the top-attributed factor without this gate would be
telling users "days since dispatch" for three quarters of predictions while that statement
distinguishes the entity from its peers in one case in eight.

### 3.4 What this means

The pipeline works and the gate discriminates, but the honest headline is narrow: **on the 3.4% of
predictions occlusion can touch, one explanation in six is specific to its entity.** The remaining
five in six cite a factor that is genuinely driving the prediction (direction and magnitude both
hold) but is driving everyone else's too — true, and not informative.

**Non-detection at this scale, not an established absence.** One task, one variant, one attribution
method, `sup_n=800`. Occlusion is the weakest of the attribution families and was chosen precisely
because it is the same operation as the verification, which makes attribution and its check
commensurable. A gradient- or mask-based method might cite different factors; it would have to
clear the same gate.

---

## 4. What it would take to extend

**Phase 1.** The blocking limitation is the construction-time estimand: the closed-form ground
truth holds the replenishment schedule fixed, so it measures the effect through the stress channel
and not the path where a structural change alters who ships next. Lifting it needs a generator
that can apply a structural edit at an arbitrary `t0` and simulate forward — Mechanism C's
existing pre-scheduled rewire machinery is the natural template. Extending to `shortage` needs
that too, since shortage outcomes depend on the inventory/agent feedback loop a single forward
pass cannot represent. Extending to more variants is cheap by comparison and would test whether
the miss is specific to Variant 0's single causal channel.

**Phase 2.** The cheapest material improvement is **more worlds**: `var_dataset` rests on 5, where
a variance estimate carries ~63% relative error at 4 degrees of freedom. Generation is ~20 s per
world at mid scale, so `S = 10-20` is nearly free and would tighten every number in §2.2. The
identifiability limit (§2.2, last paragraph) needs a generator mode that holds the supplier
population fixed across seeds and varies only the stochastic simulation; that would make
per-entity dataset-seed uncertainty measurable for the first time. Extending to shortage/impact
serving is already done — the decomposition covers all three tasks.

**Phase 3.** Extending to `shortage` and `impact` is mechanical (the modules are task-parameterised
and were run on `delay` only per the brief's scoping) but will meet the same saturation question,
which should be measured per task before anything else — impact's base rate is much lower, so its
saturated fraction will differ. The more valuable extension is a second attribution family
(integrated gradients or a learned edge mask) held to the identical specificity gate, which would
separate "this method is weak" from "these predictions are not specifically explainable". And §10's
calibrated-attribution machinery could supply a *semantic* layer on top of the feature layer, but
only for `impact`, since it is supplier-pair-based.

---

## 5. Provenance

**Trainings: 25**, all Variant 0 at the `v1` preset, `rgcn_attn_markov`, hidden=128, num_bases=10,
num_layers=4, dropout=0.2, 100 epochs, focal gamma=2, CPU, 5 dataset seeds x 5 model-init seeds.
251 s each uncontended, ~2.4 CPU-hours total, cached in `out/ds_ckpt/` and shared by every phase.
Timing was measured on the first run before any grid was committed.

**No SHARE or Markov readout retraining, and no modification.** `git diff` reports no change to
`ml/models/rgcn_attn_encoder.py`, `ml/models/rgcn_attn_markov_encoder.py`, `ml/models/heads.py`,
`ml/models/depth.py` or `ml/train.py`. `ml/ds_backbone.py::freeze` asserts every backbone
parameter is frozen and raises if not — the failure would otherwise be silent.

**New code:** `ml/ds_backbone.py` (shared frozen-backbone harness and checkpoint cache),
`ml/counterfactual_edit.py`, `ml/counterfactual_ground_truth.py`,
`ml/run_counterfactual_phase1.py`, `ml/counterfactual_delta_head.py` (Phase 1/1b);
`ml/uncertainty_ensemble.py`, `ml/uncertainty_calibrate.py` (Phase 2);
`ml/explain_prediction.py`, `ml/test_explanation_faithfulness.py` (Phase 3).

**Reused rather than reimplemented:** `ml/hypothesis_ranker.py`'s isotonic PAVA, ECE,
reliability-curve and ROC-AUC implementations, already validated by
`ml/test_hypothesis_calibration.py` (13 checks, re-run this session, all passing);
`ml/data/loader.py`'s snapshot construction and 40/20/40 split; `ml/train.py`'s objective,
unchanged; `ml/evaluate.py::collect_predictions`, so every AUC here sits on the same axis as the
rest of the project's record.

**Raw results:** `out/cf_phase1.json` (Phase 1 grid), `out/cf_phase1_floor.json` (model-seed
floor), `out/cf_phase1b.json` (Phase 1b), `out/uncertainty_grid.json` +
`out/uncertainty_grid_preds.npz` (5x5 predictions), `out/uncertainty_calibration.json`,
`out/explain_faithfulness.json`. Checkpoints in `out/ds_ckpt/`. All of `out/` is gitignored, so
this report is the durable record.

**Standing caveat, unchanged from every other result in this project.** Everything here rests on
Variant 0 at `sup_n=800` with 15 snapshots and 5 dataset seeds, on CPU-class hardware — not the
benchmark's own configuration. And every number describes a synthetic generator; whether any of it
transfers to a real supply chain is untested and untestable here.
