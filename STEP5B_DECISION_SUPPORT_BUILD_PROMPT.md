# Build and Test — Counterfactual Reasoning, Uncertainty Estimation, Explainable HADES

> Paste into a fresh session. Opus, high effort. This session builds and trains real components,
> not documentation. Read `findings/evolution.md` in full before starting — it is the record of
> what SHARE and the Markov Blanket readout actually are, and everything below reuses them as-is.
> Three independent build phases, each with its own cost check and stop condition, same discipline
> as every prior session in this project. Do not skip a stop to build all three back to back.

## Ground rules, carried from every prior session

Time the first 2–3 runs of any new training loop before committing to a full grid. Measure a
reproduction floor for any new metric before trusting a delta against it — retrain 8–10
identically-configured runs, nothing changed, and report how much the metric moves on its own,
the same methodology `ml/reproduction_floor.py` and `ml/hypothesis_ranker.py`'s calibration checks
already established. Report positive counts beside every AUC. Five seeds, sign-consistency, same
as every other result in this project. **SHARE and the Markov readout are not retrained or
modified by any of the three phases below** unless a phase's own results say otherwise — confirm
with `git diff` at the close of each phase, the same check every prior session ran.

---

## Phase 1 — Counterfactual Reasoning, built on the generator's own re-simulatability

**What actually gets built.** A graph-editing utility (`ml/counterfactual_edit.py`) that takes a
trained SHARE + Markov readout checkpoint, a snapshot's `HeteroData` graph, and an intervention
spec (remove a supplier and its edges; add a new `SUPPLIES` or dual-sourcing edge; substitute one
supplier's edges for another's), applies the edit, and re-runs the existing forward pass —
**unmodified** — to get a "naive counterfactual" prediction. This requires no new neural
architecture: it reuses the encoder and readout exactly as trained, and tests whether they
produce sensible answers to an edited input at all.

**The ground truth this project actually has and most counterfactual work doesn't.** Build a
matching generator-side utility (`ml/counterfactual_ground_truth.py`) that applies the *same*
structural intervention directly to `db/generate_dataset.py`'s config or edge construction,
regenerates the world with the same random seed apart from the intervention, and reads the *true*
simulated outcome for every entity that exists in both the original and intervened worlds. This
is the actual test: does the naive model's predicted *change* in risk (intervened prediction minus
original prediction) agree in sign and rough magnitude with the *true* simulated change from
re-running the generator itself.

**Start small — 3 intervention types, 1 variant, before any grid.** Pick the three interventions
easiest to define cleanly at the generator level: remove a supplier, add a dual-sourcing edge, and
substitute one supplier for another of the same type. Run on Variant 0 at the `v1` preset (cheap,
already-validated anchor), 5 dataset seeds, and report sign-agreement between predicted delta and
true delta per intervention type, same discipline as every delta-of-delta comparison already used
in this project (`reports/layer2_testing.md` §7 is the template).

**The hard question this phase must answer plainly, not assume the answer to:** SHARE and the
Markov readout were trained as ordinary supervised predictors, with no interventional training
signal. If the naive approach's predicted deltas do not correlate with the true simulated deltas —
a real, live possibility, and the reason this is Phase 1 and not a foregone conclusion — **stop
and report that as the finding.** It would mean an associational model cannot answer a true
intervention question without dedicated causal training signal, which motivates Phase 1b: fit a
small delta-prediction head, supervised directly on generator-produced (original, intervened,
true-delta) triples, rather than relying on the frozen forward pass. Build Phase 1b only if Phase
1's naive result is a clean miss; if the naive approach already tracks true deltas reasonably
well, report that and stop — a dedicated head would be solving a problem that doesn't exist.

---

## Phase 2 — Uncertainty Estimation, built around the variance source this project already measured

**The finding that must shape this, not be treated as background:** `reports/
phase7_training_results.md` §5.6 measured dataset-seed variance at up to **10.3x** model-seed
variance on this benchmark. An uncertainty estimate built only from model-init ensembling or MC
Dropout would be measuring the smaller of the two sources. Build the ensemble on **both axes**:
reuse the 5 dataset-seed-trained checkpoints this project already has on disk wherever possible
(`out/` directories from prior sessions), and train 5 additional model-init seeds on one dataset
seed, giving a 5x5 grid that can report how much of total predictive variance comes from which
axis — the direct, quantified answer to what §5.6 only measured on one metric.

**What gets built:** `ml/uncertainty_ensemble.py` (predictive mean and variance from the 5x5 grid,
decomposed by axis) and `ml/uncertainty_calibrate.py`, reusing the isotonic-regression-plus-ECE
machinery already written and validated in `ml/hypothesis_ranker.py` — don't reimplement
calibration from scratch, that code already exists and is already tested against sklearn and
known-good synthetic cases. Calibrate the ensemble's predictive interval against true empirical
coverage the same way `reports/layer3_testing.md` §10.3 validated `regional_logistics`'s
confidence numbers: a reliability curve over equal-count bins, reported before and after
calibration, on held-out seeds never used to fit the calibration.

**Cost check before committing.** The 5x5 grid is 25 trainings at whatever this session's
per-run cost turns out to be — time the first 3 before scheduling the rest, same as every prior
sweep in this project.

**Deliverable check specific to this phase:** report explicitly what fraction of total predictive
variance is dataset-seed vs. model-seed, per task — this number either confirms §5.6's finding
generalizes to a proper decomposition or revises it, and either result is worth stating plainly.

---

## Phase 3 — Explainable HADES, built on the one attribution mechanism already validated

**Reuse, don't rebuild.** `reports/layer3_testing.md` §10 already built and validated a calibrated
attribution mechanism — it correctly attributes correlated supplier behavior to observable causes
(AUC 0.817, ECE 0.003) and correctly declines to claim what it has no evidence for. This phase
extends that same machinery (`ml/hypothesis_features.py`'s observable feature set, `ml/
hypothesis_ranker.py`'s calibrated head design) from "why are these two suppliers correlated" to
"why is this specific delay/shortage/impact prediction elevated" — build `ml/explain_prediction.py`
using the same observable feature families already proven to calibrate well (lead time, capacity,
reliability trend, degree, snapshot history) as the candidate explanatory factors for a single
prediction, rather than starting an explainability method from a generic literature survey.

**The mandatory step this phase cannot skip, per this project's own history:** attention weights
or raw feature-importance scores are not an explanation until they're checked for faithfulness.
Build `ml/test_explanation_faithfulness.py`: for the top-attributed factor on a sample of
predictions, perturb or ablate exactly that factor (holding everything else fixed) and confirm the
prediction actually moves in the direction and rough magnitude the explanation implied. **An
explanation that fails this test is not reported as a finding — it is reported as a failed
explanation**, the same standard this project already applied to attention-weight retrieval
confidence in `reports/layer3_testing.md` §9's contrastive arm, which looked confident and was
shown to be memorization.

**Scope for this session:** delay predictions only, Variant 0, 5 seeds — the cheapest, most
data-rich task, to get a working faithfulness-tested pipeline before spending the same build on
shortage and impact. Report the faithfulness pass rate (what fraction of sampled explanations
survive the ablation check) as the headline number, not accuracy or plausibility.

---

## Deliverable

New report, `reports/decision_support_build.md`, one section per phase, each stating what was
built, what the cost check found, the actual results against the stop conditions above, and — for
whichever phases produced a working component — what it would take to extend to the remaining
tasks and variants. State plainly, per phase, whether the phase's own stop condition was hit
(Phase 1's naive-approach miss, Phase 3's faithfulness failure) or cleared, using the same
"non-detection at this scale, not an established absence" framing every other result in this
project carries.
