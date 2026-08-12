# Layer 3, Reframed — Behavioural Hypothesis Generation, Not Hidden-Cause Discovery

> Paste into a fresh session. Opus, high effort. Read `reports/layer3_testing.md` in full,
> especially §9.7 (the observable co-failure result and the base-pool confound) and §9.8 (the
> depth probe — the finding that governs this entire session). This is not a retry of hidden-
> dependency discovery. It is a different, narrower claim, and the session's job is to build and
> honestly evaluate that narrower claim, not to quietly re-ask the question §9.8 already answered.

## Why this session exists, and what it is not

§9.8 found that at **no depth of SHARE's stack — including the raw input features — does true
co-membership separate from a size-matched random regrouping.** No retrieval objective, however
designed, can reliably answer "do these two suppliers share a hidden parent" from this
representation. That question is closed, per that session's own findings, pending a generator
change this session does not attempt.

This session builds something else: instead of committing to one hidden cause, detect unusual
correlated behaviour between suppliers and produce a **ranked, calibrated set of plausible
explanations** — including an explicit "unknown" option — and leave the final call to a human
reviewer. This is a weaker, more defensible claim than discovery, and it is allowed to be weaker
without being a failure, provided it is evaluated honestly against what it actually claims.

**State this prediction before building anything, so a specific result later isn't
misread as a bug:** two of the candidate hypotheses below rest on fundamentally different
evidence. "Regional logistics disruption" and similar categories map onto the base dataset's
shared factor pools (`H_PORT`, `H_TRUCK`, `H_CUSTOMS`), which are largely **observable** — two of
three are defined by `country`, the third by shipping mode, both already features in the graph.
"Shared upstream dependency" is exactly the hypothesis §9.8 showed has **zero** encodable signal
anywhere. A correctly calibrated model should therefore rank "shared upstream dependency" low or
push it toward "unknown" for most detected patterns. **If that happens, it is the module working
correctly, not failing** — say so explicitly in the report rather than treating it as a
disappointing result.

## 1. Ground truth — build it from mechanisms that already exist, not an invented taxonomy

Do not invent unsupervised categories. Construct multi-class ground truth per candidate pattern
from what the generator already tracks, read the same privileged way `HP_GROUPS` was read in the
last session (`ml/extract_hidden_state.py`) — **training-time and evaluation-time only, never a
model input**, and disclose this the same way §9.1 disclosed Stage 1's privileged supervision:

- **Shared upstream dependency** — `HP_GROUPS` Type A or B co-membership (Type C excluded — it is
  the decoy).
- **Regional / logistics factor** — shared membership in `H_PORT`, `H_TRUCK`, or `H_CUSTOMS`
  (locate the exact assignment structure in `db/generate_dataset.py`'s `own_stress` factor pools
  before assuming the names or membership rule carry over from §9.7.1's summary — confirm live).
- **Supplier switching** — Mechanism C rewiring events. **Confirm before building this category
  at all**: §9.7.1 found `CS_REWIRES = 0` on both Variant B and Variant D. If the primary test
  variants carry no Mechanism C activity, this category has zero true positives there by
  construction — either extend the variant set to include Mechanism C, or explicitly report this
  hypothesis as untestable on the current data and drop it from the trained categories rather than
  training on a class with no positive examples.
- **Unknown / coincidental** — pairs whose observable co-degradation clears the detection
  threshold below but share **no** privileged mechanism at all. This is a real, expected category,
  not a leftover bucket — most detected patterns will likely land here or in the regional/logistics
  category, per the prediction above.

A pair may legitimately carry more than one true label (e.g., a Type A pair that also shares
`H_PORT`) — build this as **multi-label**, not forced single-class, and say so in the report.

## 2. Behavioural Pattern Detection — reuse §9.7's instrument, don't rebuild it

This stage already exists in substance. Reuse `ml/observable_cofailure.py`'s `on_time_rate_90d`
correlation scoring (the exact column §2.2's generator-side check validated) to produce a
candidate set of **pattern instances** — pairs whose correlation clears a threshold or ranks in
the top-K per supplier. **Expect this set to be thin and confound-heavy, per §9.7.3's own
numbers**: only 704–708 of 2,000 suppliers carried a usable trajectory at mid scale, and
coincidental base-pool pairs outnumbered real Type A/B pairs 363:1. Report the candidate set's
size and composition (how many carry each ground-truth label, including multi-label overlap)
before training anything on top of it — this determines how badly imbalanced the ranking task
is, and that imbalance needs to be handled explicitly (class weighting or resampling), not
ignored.

## 3. Hypothesis Ranking — the model, and how to evaluate it honestly

A classification head over the detected pattern instances, trained against the multi-label ground
truth in §1, producing a distribution (not a single argmax) over: shared upstream dependency,
regional/logistics, supplier switching (if retained per §1), unknown. Input features: whatever
observable signal the pair's correlation was built from, plus each supplier's own observable
attributes (country, transport mode, degree) — **not** privileged mechanism membership, which is
the training target, not a feature.

**Calibration is the actual deliverable, not accuracy.** Build a reliability curve and report
Expected Calibration Error per class — does "41% shared upstream" actually correspond to roughly
41% of such-labelled instances being true, over enough instances to say so. A model that always
predicts a plausible-looking but uncalibrated number has reproduced the attention-weight-as-
explanation trap this project has already flagged once; do not let a confident-looking percentage
substitute for a validated one.

**Measure a reproduction floor for these metrics before trusting any of them**, same discipline
§9.2 established for retrieval quality: retrain 8–10 identically-configured pairs and report how
much per-class calibration error and ranking AUC move with nothing changed. A result that doesn't
clear this floor is not a finding.

**Report per-class performance separately, not one pooled number** — given the prediction above,
the "shared upstream dependency" class is expected to perform near chance or be systematically
down-ranked; conflating it with the regional/logistics class (expected to perform well, since it's
close to observable) would hide the one result this session most needs to state plainly.

## 4. Fusion and AUC — a separate, secondary experiment, not the success criterion

**Decouple this explicitly.** Every version of Transformer 2 fusion tested across three prior
sessions — original additive, confidence-weighted, trust-gated, cross-attention, and the
contrastive-retrieval variant in §9 — came back flat-to-strongly-negative on impact AUC. Nothing
about reframing retrieval as hypothesis ranking changes that history, and this session should not
assume it will. If time allows after §3's ranking quality is reported and calibrated, reintroduce
Confidence-Aware Fusion using the hypothesis module's own confidence as the fusion weight, and
report the AUC delta against this session's own measured reproduction floor (re-measure it on
whatever device this session runs on, per §9.10's finding that the floor is device- and
arm-dependent, not a fixed constant carried over from earlier sessions). **State up front in the
final report that a null or negative result here would not undermine §3's findings** — hypothesis
quality and downstream AUC are different claims, and this project has already learned what happens
when they get conflated.

## 5. Human-in-the-loop deliverable

Produce the ranked-hypothesis output in the format this module is meant to actually deliver: for a
sample of detected pattern instances (worth reviewing a few dozen by hand), the supplier pair, the
detected behavioural pattern, and the ranked hypothesis list with calibrated confidences —
including cases where "unknown" wins, which is expected and should be shown, not filtered out.
This is what a supply-chain reviewer would actually see; include it in the report as a concrete
artifact, not just aggregate metrics.

## Cost discipline

Same as every prior session: confirm which variants actually have usable Mechanism C activity
before deciding whether to generate new mid-scale data or drop that category; time the first few
runs of pattern detection and ranking-model training before committing to the full grid; reuse
Variant B and D's existing mid-scale data (`db/csv_mid/`) rather than regenerating unless the
supplier-switching category requires a new variant.

## Deliverable

Append a new section to `reports/layer3_testing.md` — **§10, Behavioural Hypothesis Generation**
— continuing the same report rather than starting a new one, per this project's standing
convention. Structure it around: the ground-truth construction and its disclosure (§1), the
detected pattern set's size and composition (§2), per-class ranking quality and calibration
against a measured reproduction floor (§3), the human-facing example output (§5), and — clearly
separated, clearly secondary — the fusion/AUC result (§4). Close with one paragraph stating
whether the "shared upstream dependency" class landed where predicted, and what this does and
does not settle about Layer 3's value as a decision-support tool versus a discovery mechanism, at
this label volume, same caveat every other result in this project carries.
