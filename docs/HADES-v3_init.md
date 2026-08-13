# HADES V3 — Initiation Document: A Neuro-Symbolic Decision-Support Architecture

**Status:** verified proposal, corrected in three places against HADES V2's actual measured
results, with one open validity caveat flagged before any implementation work begins. This
document is the authoritative starting reference for HADES V3, in the same role
`docs/00_Benchmark_Specification.md` served for V2.

**Provenance:** everything below is grounded in `HADES_v2/findings/evolution.md` (the full
architecture history through V2) and `HADES_v2/reports/layer3_testing.md` (the hidden-dependency
discovery investigation this proposal responds to). Two components — SHARE and the Markov
Blanket depth readout — are inherited unchanged from V2, validated, and not re-opened here.

---

## 1. Executive summary

HADES V3 evolves HADES from a purely predictive graph neural network into a neuro-symbolic
decision-support system. It keeps V2's two validated components (SHARE, the Markov Blanket
readout) and replaces the abandoned hidden-relationship-discovery layer with two new components:
an estimator for **latent operational state** (not latent supplier *identity*, which V2 spent
five independent sessions proving is not recoverable from this benchmark's observable data), and
an **explicit structural causal model** built directly from the benchmark generator's own known
equations, feeding a counterfactual engine capable of answering genuine intervention questions —
something no purely observational model in this project has ever been shown able to do.

This is a change of *kind*, not just of degree. Every architecture V2 built — SHARE, Markov,
Transformer 2 in all its forms — was trained to answer `P(Y|X)`: what will happen, given what is
observed. None of them were ever trained or evaluated on `P(Y|do(X))`: what would happen if the
world were deliberately changed. V3's central claim is that this second question requires a
structurally different component, not a bigger or better version of the first.

---

## 2. Why V1 and V2 were limited, precisely

### 2.1 Original HADES

```
Graph → Encoder → Prediction
```

Learned statistical association between observed features and future disruption:
`P(Y|X)`, `X` = observed graph, `Y` = future disruption. Never trained or capable of answering an
intervention question, because nothing in its training signal ever represented "the world changed
because of an action," only "the world is, and here is what followed."

### 2.2 HADES V2's Layer 3, and precisely why it failed

`Graph → SHARE → Markov → Hidden Relationship Discovery → Prediction`

Five independent approaches were tried, across two independently-built datasets: similarity-based
retrieval, privileged contrastive supervision (a direct ceiling test with the true labels handed
to the loss), observable behavioral (co-failure) retrieval, and a reframed hypothesis-ranking
formulation. **The precise finding, not the general one:** a same-strength hidden coupling (0.35)
was recoverable at AUC 0.78 when it was a *direct, pairwise* relationship between two named
suppliers (the `component_suppliers` co-parent relationship, used as a positive control), and
**not recoverable at all**, at any pool width, on either dataset variant tested, when the
identical-strength coupling was spread as an average across a group of four or five members
(`HADES_v2/reports/layer3_testing.md` §10.3.4). A representation probe confirmed this at every
depth of the encoder, including the raw input features, against a permutation-matched null: no
cue anywhere distinguishes two suppliers secretly sharing a hidden parent from two that do not.

This is a sharper diagnosis than "multiple hidden causes produce identical observable behaviour,
therefore the inverse mapping is non-identifiable" (true as a general statement, and also a real,
separately-measured confound — unrelated shared factor pools outnumbered true hidden-parent pairs
363:1 in observable co-degradation data, `layer3_testing.md` §9.7.1). The more specific and more
useful finding is mechanistic: **this benchmark's hidden-parent mechanism dilutes its effect
across a group mean, which destroys the pairwise attribution any retrieval method or classifier
needs.** That is a property of how the generator currently constructs hidden-group influence, not
a property of retrieval architecture, and it is why five different architectural answers to "how
should retrieval work" all hit the identical wall.

### 2.3 The core insight V3 is built around

Prediction and intervention are different mathematical problems.

```
Prediction:      P(Y | X)         "What happens, given what I observe?"
Intervention:    P(Y | do(X))     "What happens if I deliberately change the world?"
```

A predictive encoder cannot automatically learn intervention semantics from observational training
data alone — this is not specific to HADES, it is a general property of the gap between
association and intervention (Pearl's causal hierarchy: association is rung 1, intervention is
rung 2, and no amount of rung-1 data alone identifies rung-2 quantities without either randomized
interventional data or explicit structural assumptions). V2's architecture never had either. V3
supplies both: the benchmark's generator provides re-simulatable interventional data (§5.1 below),
and the structural causal model supplies the explicit assumptions.

---

## 3. Proposed architecture

```
Supply Chain Graph
        ↓
SHARE                          (Layer 1 — representation learning, UNCHANGED from V2)
        ↓
Markov Blanket                 (Layer 2 — local predictive neighbourhood, UNCHANGED from V2)
        ↓
Latent State Estimation        (Layer 3 — NEW: operational state, not hidden identity)
        ↓
Structural Causal Model        (Layer 4 — NEW: explicit causal equations)
        ↓
Counterfactual Engine          (Layer 5 — NEW: intervention evaluation)
        ↓
Prediction Heads               (delay / shortage / impact / confidence / explanation / counterfactual outcomes)
```

**Note on a diagram discrepancy in the proposal this document formalizes:** an earlier sketch of
this architecture placed a "Risk Prediction" stage before the Counterfactual Engine and a separate
"Explanation Engine" after it. This document adopts the single-`Prediction Heads` version above,
with confidence, explanation, and counterfactual outcomes as parallel outputs of one final stage
rather than sequential engines — simpler, and consistent with how V2's existing prediction heads
already attach directly to a shared representation. Revisit this choice explicitly before
implementation if a sequential design turns out to be needed.

---

## 4. Layer 1 — SHARE (inherited, unchanged)

Relation-aware R-GCN with a shared, relation-blind attention scorer. Transforms the heterogeneous
supply-chain graph into `h⁰...h⁴` per node — representations at every depth from the raw input
projection through four rounds of message passing. Validated as best-or-tied-for-best on all
three prediction tasks across V2's twelve mechanism variants. **Not retrained or modified by any
V3 component** unless a specific V3 finding calls for it — confirm with `git diff` at every stage
of implementation, the discipline every V2 session applied to this exact question.

---

## 5. Layer 2 — the Markov Blanket readout (inherited, unchanged)

Zero-parameter, per-task fixed depth (delay→h¹, shortage→h³, impact→h⁴). The winner of an
exhaustive two-project investigation into per-node adaptive depth selection — eight distinct gate
designs, roughly 130 training runs, never beaten, with the two closest contenders mechanically
shown to be either incapable of deviating from the fixed prior at any tested setting or collapsing
into a single global depth per random seed rather than genuine per-node adaptivity
(`findings/evolution.md` §3). **This is not "Adaptive Markov Blanket."** Adaptive depth selection
is a closed question in this project, not an open one, and no V3 component should reopen it.

---

## 6. Layer 3 — Latent State Estimation

**Purpose.** Estimate hidden *operational conditions* — not hidden supplier *identities*, which
§2.2 established is not recoverable from this benchmark's observable data by any method tried.

**Why this is expected to succeed where identity discovery did not — precedent, not analogy.**
This is not solely a design intuition ("real companies can observe effects without observing
causes"). It has a partial, already-measured empirical precedent in this project's own history:
Mechanism E's true hidden resilience state was checked for observable recoverability during V2's
dataset calibration and found recoverable at **AUC ≈ 0.599** — meaningfully above chance, in
sharp contrast to hidden-parent *identity*, which sat at exactly chance at every depth of the
network including raw input features, under every method tried (`docs/phase2_coverage_recheck.md`;
`layer3_testing.md` §9.8). One data point is not a proof, but it is real evidence, already in
hand, that operational-state estimation and identity discovery behave differently on this
benchmark — cite this precedent directly rather than presenting the design choice as untested.

**Candidate latent states, and what must be confirmed before building each one.** Six were
proposed: Supply Stress, Inventory Health, Logistics Stability, Capacity Pressure, Supplier
Reliability, Recovery Capability. **Not all six are equally supported today.** Two map directly
onto generator-tracked internal state already used elsewhere in this project as privileged,
training-time-only supervision (`own_stress` ≈ Supply Stress; `RESILIENCE` ≈ Recovery Capability).
The other four do not have a confirmed generator-side ground truth as of this document. Before
building an estimation head for any of the six, confirm — the same way `HP_GROUPS` and
`RESILIENCE` were confirmed to exist and to have the claimed structure before being used
(`layer3_testing.md` §9.1, §10.1) — whether the generator tracks a corresponding internal
variable, or whether it would need to be newly instrumented, or whether it should be dropped from
the initial build. Do not assume all six are equally buildable.

**Supervision discipline.** Any generator-tracked latent used to supervise this layer is
privileged, training-time (and evaluation-time) only, disclosed explicitly, and never a model
input at inference — the same standard this project has applied to every privileged read since
`ml/extract_hidden_state.py` was first built.

**Recommended testbed.** V2's Mechanism A (Partial Visibility) already truncates hidden-tier
suppliers out of the observable graph, and Variant A is the dataset already built around exactly
the "Tier-1 visibility only" premise motivating this layer. Use it as the first testbed rather
than constructing a new scenario — the data already exists.

---

## 7. Layer 4 — Structural Causal Model

**Purpose.** Convert estimated latent operational states into explicit causal reasoning —
e.g. `Supply Stress → Inventory Reduction → Production Delay → Shipment Delay → Customer Impact`
— as stated equations, not a learned black-box function.

**The recommended approach, and why it is stronger than the open question the original proposal
left it as.** The proposal's own "Open Research Questions" ask how SCM equations should be
learned, implying causal discovery from data — a notoriously hard, brittle problem in general.
**This benchmark does not require that.** `HADES_v2/db/generate_dataset.py` *is* a structural
causal model already — it contains the true, known equations relating stress, resilience,
absorption, and downstream outcomes (`own_stress`, `COPARENT_COUPLING`, `HP_ALPHA`, the resilience
absorption terms). **Extract the SCM's equations directly from the generator's source rather than
learning them from data.** This removes the single hardest open problem in the original proposal
and gives V3 a correctness guarantee no learned-SCM approach could match on this benchmark.

**The validity caveat that must not be glossed over.** Extracting a correct SCM this way works
*because* this is a synthetic benchmark with known, coded ground-truth equations. It does not, by
itself, demonstrate that this architecture works for a real deployment, where no such generator
exists and the true causal structure would need to be hand-specified by domain experts or learned
via causal discovery from real operational data — a different and substantially harder problem
this project has not attempted. **Keep these two claims separate in every future document:**
"validated on this benchmark via generator-extracted equations" is achievable now; "ready for real
industrial deployment" is not established by that validation and should not be asserted as though
it were.

---

## 8. Layer 5 — Counterfactual Engine

**Purpose.** Evaluate interventions — remove a supplier, add dual sourcing, reroute logistics,
substitute a supplier, increase inventory — by simulating outcomes under the SCM from Layer 4,
rather than by re-running a purely observational forward pass on an edited graph and hoping the
result is causally meaningful.

**Ground truth this project already has, and should use.** Because the generator is
re-executable, real interventional ground truth can be produced directly: apply the same
structural intervention to the generator's configuration or edge construction, regenerate the
world with the same seed apart from the intervention, and read the true simulated outcome. This
is the same asset and the same method already scoped for the counterfactual-reasoning build in
`HADES_v2/STEP5B_DECISION_SUPPORT_BUILD_PROMPT.md` Phase 1 — this layer should be built as the
production version of that same mechanism, not a separate design.

**The question this layer must answer honestly, not assume.** Whether a naive approach (edit the
graph, re-run SHARE and the Markov readout unmodified) tracks true simulated intervention effects
at all is an open, testable question — if it does not, the SCM layer becomes load-bearing rather
than optional, and the counterfactual engine must route through Layer 4's explicit equations
rather than the encoder's raw forward pass.

---

## 9. Prediction heads

Delay risk, shortage risk, impact risk, confidence (see uncertainty-estimation work already
scoped in `STEP5B` Phase 2), explanation (see the calibrated-attribution work already validated
in `layer3_testing.md` §10 and scoped for extension in `STEP5B` Phase 3), and counterfactual
outcomes from Layer 5. These reuse existing, already-validated machinery wherever V2 built it,
rather than each being designed from scratch.

---

## 10. Why this is a better fit for real deployment than V2's Layer 3 was — and what remains
unproven about that claim

Real companies typically observe Tier-1 suppliers, orders, inventory, logistics, and shipments,
and typically do not observe Tier-2 or Tier-3 suppliers directly. V2's Layer 3 asked a model to
recover hidden supplier *identities* from Tier-1-visible behavior — exactly the question five
sessions of experimentation showed is not answerable on this benchmark. V3 does not attempt that.
It estimates hidden operational *conditions* from observable evidence instead, which is both a
better match to what a real company could plausibly validate against its own history, and a
question this project already has one positive empirical data point for (§6 above).

**What is not yet demonstrated:** that the SCM layer, RL-free counterfactual reasoning, and the
latent-state estimator generalize beyond this synthetic benchmark to real operational data, where
the generator-extraction shortcut in §7 is not available. This document records that gap
explicitly so it is not silently assumed away in later work.

---

## 11. Comparison

| | Original HADES | HADES V2 | HADES V3 (this document) |
|---|---|---|---|
| Pipeline | Graph → Prediction | Graph → SHARE → Markov → Hidden Relationship Discovery → Prediction | Graph → SHARE → Markov → Latent State Estimation → SCM → Counterfactual Engine → Prediction |
| Answers | `P(Y\|X)` only | `P(Y\|X)` only (discovery never converted to accuracy) | `P(Y\|X)` and, via the SCM, `P(Y\|do(X))` |
| Hidden-structure approach | None | Recover hidden supplier *identity* — closed, not recoverable | Estimate hidden operational *state* — has partial empirical precedent |
| Causal equations | None | None | Extracted directly from the generator's known structure |

---

## 12. Open research questions, carried forward from the original proposal, with this document's
position stated where evidence already exists

- **Which latent states are universally useful?** Not yet known; §6 above requires confirming
  generator-side ground truth per candidate state before building it, rather than assuming all
  six are equally supported.
- **How should SCM equations be learned?** Not an open question for this benchmark — extract them
  from `db/generate_dataset.py` directly (§7). Remains genuinely open for any future real-data
  application.
- **Can latent states be partially supervised?** Plausible, following the same privileged-signal
  discipline already established for `HP_GROUPS` and `RESILIENCE` — worth a dedicated build
  session, not assumed here.
- **How should uncertainty propagate through the SCM?** Open. Connects directly to the
  uncertainty-estimation work already scoped in `STEP5B` Phase 2, which should be built to
  propagate through this layer rather than only covering the prediction heads.
- **Can causal rules adapt online?** Open, and out of scope for an initial build.

---

## 13. Status and next step

This document establishes the architecture and its evidentiary basis. It does not implement
anything. The next step is a build-and-test prompt, in the same style as
`HADES_v2/STEP5B_DECISION_SUPPORT_BUILD_PROMPT.md`, scoped to: (1) confirming which of the six
candidate latent states have real generator-side ground truth, (2) extracting the SCM's equations
from the generator directly, and (3) testing whether the naive graph-edit counterfactual approach
tracks true re-simulated outcomes before deciding whether Layer 4 is load-bearing or an
enhancement — mirroring the phased, gated discipline every V2 session in this project's history
has used before committing to a full build.
