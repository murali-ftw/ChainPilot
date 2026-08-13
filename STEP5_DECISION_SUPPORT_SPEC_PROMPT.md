# Decision-Support Architecture Specification — Counterfactual Reasoning, Uncertainty Estimation, Explainable HADES

> Paste into a fresh session. Opus, high effort, expect this to run long — it is a pure
> documentation task (no code, no implementation) but a large one; let it continue across
> multiple responses rather than compressing. Read `findings/evolution.md` in full before
> writing anything — it is the authoritative record of what HADES's architecture actually is
> today and why Layer 3 was scrapped, and this specification must be grounded in that record,
> not in a generic or idealized version of it.

## Role

You are acting as lead AI systems architect for HADES. Your task is to produce a complete
technical design specification for three new capabilities, built on top of HADES's validated
architecture. **You are not implementing anything. Do not write code.** The document must be
detailed enough that another engineer could implement the system directly from it — full
architecture, mathematics, training and inference pipelines, APIs, database changes, evaluation
protocol, ablations, failure modes, limitations — for all three features, with nothing skipped
or summarized away. If the response would exceed the context window, stop naturally and continue
in the next response until the specification is complete.

## Correction to carry through the entire document, stated once here so it isn't repeated as a caveat in every section

**HADES's Layer 2 is not adaptive.** It is a fixed-depth Markov Blanket readout — a
zero-parameter, per-task lookup (delay→h¹, shortage→h³, impact→h⁴), the winner of an extensive,
two-project investigation into whether per-node adaptive depth selection could beat it. It never
did: eight distinct gate designs, roughly 130 training runs across two independently-built
datasets, and the two most-likely-looking adaptive designs (Rung 5, Variant A) were shown to
either be mathematically incapable of deviating from the fixed prior at any tested setting, or to
collapse into picking one global depth per random seed rather than genuine per-node adaptivity.
`findings/evolution.md` §3 has the full account. **Refer to this component as "the Markov Blanket
depth readout" or "the fixed-depth Layer 2 readout" throughout this specification, never
"Adaptive Markov Blanket."** All three features below build on top of this fixed, validated
component — none of them should assume or reintroduce per-node depth adaptivity, which is a
closed question, not an open one.

**Current validated architecture, for reference throughout:**

- **Layer 1 — SHARE** (`ml/models/rgcn_attn_encoder.py`): relation-aware R-GCN with a shared,
  relation-blind attention scorer. Outputs `h⁰...h⁴` per node — the representation at every depth
  from the raw input projection through four rounds of message passing.
- **Layer 2 — the Markov Blanket readout** (`ml/models/rgcn_attn_markov_encoder.py`): selects
  the fixed depth per task described above, zero learned parameters.
- **Layer 3 — frozen.** Hidden-relationship discovery was tested through five independent
  approaches (similarity retrieval, confidence-aware fusion, a trust gate, cross-attention,
  contrastive retrieval under privileged supervision, and observable co-failure retrieval) and
  found the limitation is in the *data* — this benchmark's hidden-parent mechanism spreads
  influence across a group average, which destroys the pairwise attribution any retrieval method
  needs — not in any architecture tried. `findings/evolution.md` §4–§5 has the full account, one
  retained capability (calibrated attribution to *observable* causes), and the specific generator
  change that would reopen discovery if ever prioritized.

Every feature below must state explicitly, in its own architecture section, which parts reuse
SHARE's existing `h⁰...h⁴` outputs and the Markov readout unmodified, and which parts require new
components — per the brief's own instruction, this distinction must be explicit, not implied.

---

## Feature 1 — Counterfactual Reasoning

**Research question:** "What would happen if the supply chain changed?" — remove a supplier, add
a backup, change a logistics provider, add dual sourcing, increase inventory, substitute a
supplier.

**Project-specific grounding to build the specification around, not a generic counterfactual-GNN
design:** HADES's benchmark is a synthetic generator with a known, re-executable data-generating
process (`db/generate_dataset.py`). This is an unusual asset for counterfactual ML — most
counterfactual-prediction work has no ground truth to validate against at all. Specify how the
generator itself can produce **interventional ground truth**: rerun the generator with a
specified structural change applied (a supplier's edges removed, a new sourcing edge added) and
compare the resulting simulated outcomes against the trained model's counterfactual prediction for
the same intervention. This makes the evaluation protocol for this feature fundamentally different
from — and more rigorous than — most published counterfactual-GNN work, and the specification
should build the evaluation section around that asset specifically, not around a generic
proxy-metric approach.

Also specify, explicitly, whether SHARE and the Markov readout — trained purely as standard
supervised predictors, with no interventional or causal training signal — can be trusted to
answer a true intervention question at all, or whether the architecture needs a counterfactual-
specific training signal (e.g., training on generator-produced intervention/outcome pairs
directly) to avoid the standard failure mode where an associational model produces a
plausible-looking but causally wrong answer to "what if." State a position and justify it.

Cover, in full: motivation, research gap, scientific contribution, system architecture, new
layers required (state explicitly whether any are needed beyond a new head/module on SHARE's
outputs), data flow, mathematical formulation, counterfactual graph generation and intervention
representation, graph editing strategy, whether and how message passing needs to change to
support edited graphs, training objectives and loss functions (grounded in the generator-produced
ground truth above), training and inference pipelines, API design, database schema changes,
dashboard integration, evaluation metrics (built around the generator re-simulation asset),
ablation studies, failure modes, limitations, complexity analysis, and future extensions.

---

## Feature 2 — Uncertainty Estimation

**Research question:** "How confident is this prediction?"

**Project-specific grounding, not a generic Deep-Ensembles-vs-MC-Dropout survey:** two findings
from this project's own history must shape this design, not be treated as background literature.

First, `reports/phase7_training_results.md` §5.6 measured that **dataset-seed variance is up to
10.3x larger than model-init-seed variance** on this benchmark's delay task (0.0376 vs. 0.0037
AUC std) — meaning most published uncertainty-estimation techniques (Deep Ensembles varying model
initialization, MC Dropout, evidential heads trained on one dataset) would systematically capture
the *smaller* source of variance and understate the *larger* one, which is variance in which
world was generated, not variance in how the model was trained. The specification must address
this directly: does the proposed method's uncertainty estimate account for dataset-level
variance, or only model-level variance, and if only the latter, say so plainly as a limitation
rather than presenting the method as if it captures "confidence" in the everyday sense.

Second, this project already has a fully worked reproduction-floor methodology (`reports/
layer2_testing.md` §4, `reports/layer3_testing.md` §9.2/§10.3.1) — measuring how much a metric
moves under identically-configured retraining, with nothing changed, as the baseline any claimed
effect must clear. Specify how this exact methodology is reused to validate that a calibration
claim (e.g., an Expected Calibration Error number) is itself a real result and not noise, before
any confidence output is presented as trustworthy.

Cover, in full: motivation, scientific contribution, a reasoned comparison of Bayesian vs.
frequentist approaches, Deep Ensembles, Monte Carlo Dropout, and Evidential Deep Learning against
this project's specific data (multi-seed training already standard practice here, per the
established sweep methodology — state whether the existing 5-dataset-seed infrastructure can be
reused directly as ensembling material, and whether that alone addresses the dataset-variance
problem above or only partially does), which approach is recommended and why, required
architecture modifications (state explicitly whether this needs new neural layers or only
additional prediction heads on SHARE's existing outputs), mathematical formulation, confidence
calibration, prediction intervals, risk calibration, reliability diagrams, Expected Calibration
Error, Brier Score, Negative Log-Likelihood, training objectives and loss functions, integration
with the existing delay/shortage/impact prediction heads, dashboard visualization, API design,
database schema changes, evaluation protocol (incorporating the reproduction-floor discipline
above), ablation studies, failure modes, and computational overhead.

---

## Feature 3 — Explainable HADES

**Research question:** "Why did the model make this prediction?"

**Project-specific grounding, not a generic GNNExplainer/PGExplainer survey:** this project
already has one working, validated attribution mechanism to build from and one documented failure
mode to explicitly design against.

The working mechanism: `reports/layer3_testing.md` §10's hypothesis-generation module correctly
and calibratedly attributes correlated supplier behavior to *observable* causes (AUC 0.817,
Expected Calibration Error 0.003, validated over 13,000+ instances) and correctly declines to
claim a cause it has no evidence for. Specify how this same calibrated-attribution machinery
generalizes from "why are these two suppliers correlated" to "why is this specific delay/
shortage/impact prediction elevated" — attributing a single prediction to observable contributing
factors rather than producing a plausible-looking but unvalidated importance score.

The failure mode to design against, flagged explicitly multiple times across this project's
history: attention weights presented as explanations without validating that they are faithful —
that the "explained" factor, when removed or perturbed, actually changes the prediction the way
the explanation implies. Any explanation method proposed here must specify a concrete faithfulness
test built on exactly that logic (ablate the cited factor, confirm the prediction moves as
claimed) as a mandatory validation step, not an optional one, before any explanation is presented
to a user as trustworthy.

Cover, in full: motivation, scientific contribution, an explainability taxonomy (local, global,
counterfactual explanations, feature attribution, graph attribution), a reasoned comparison of
GNNExplainer, PGExplainer, GraphMask, Integrated Gradients, and attention rollout against this
project's architecture and data, which method or combination is recommended and why, faithfulness
and sufficiency and comprehensiveness as concrete, measured properties (not asserted ones — tie
directly to the ablation-based faithfulness test above), human interpretability, explanation
validation protocol, dashboard visualization, API design, database schema changes, evaluation
metrics, a human evaluation protocol, failure modes, and computational complexity. State
explicitly whether explanation generation happens during inference (as part of the forward pass)
or as a post-processing module over a frozen trained model, and justify the choice against this
project's existing deployment pattern (SHARE and the Markov readout run inference once per
snapshot; state how that constrains or doesn't constrain the explanation approach).

---

## Required structure, for every feature, in this order

1. Motivation
2. Problem Statement
3. Research Gap
4. Scientific Contribution
5. System Architecture
6. Component Diagram (ASCII)
7. Data Flow Diagram (ASCII)
8. Mathematical Formulation
9. Neural Network Changes — explicit on what reuses SHARE/the Markov readout unmodified vs. what
   is new
10. Loss Functions
11. Training Procedure
12. Inference Procedure
13. Algorithms
14. Pseudocode (structure and logic only — no runnable code)
15. API Design
16. Database Schema Changes
17. Dashboard / UI Design
18. Computational Complexity
19. Evaluation Metrics
20. Ablation Studies
21. Failure Modes
22. Limitations
23. Expected Research Contributions
24. Future Extensions

## Output requirements

Markdown, ASCII diagrams, equations where appropriate. No code, no implementation. Do not
summarize or compress any section — completeness over brevity throughout. Continue across
multiple responses if needed rather than shortening.

## Deliverable

Write to `docs/Decision_Support_Architecture_Spec.md`. This is a specification document, joining
`docs/00_Benchmark_Specification.md` as an authoritative design reference for this project, not
an experimental report — do not place it under `reports/` or `findings/`.
