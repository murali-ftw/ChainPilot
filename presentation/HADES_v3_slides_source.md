# HADES v3
## A Validation-First Neuro-Symbolic Architecture for Supply-Chain Risk

Predicting shipment **delay**, product **shortage** and supplier **impact** — and being honest about which of those answers we are allowed to explain.

Three generations. Five layers. Four gates. Every claim carries the test that earned it.

*Measured on the HADES-Bench synthetic generator, Variant 0, 800 suppliers × 15 snapshots, 5 dataset seeds × 5 model-init seeds.*

---

# The Finalised Architecture

![HADES v3 overall architecture](https://s3-alpha.figma.com/thumbnails/40915307-48e7-4dbc-aadb-12aa698b31a1?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073400Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=8890bcfe1c4f6e7f2da9d59fd6cf2595323a3177757d9b14db434b24db43b4e4)

| Layer | What it is | Status |
|---|---|---|
| **0 — Input** | Heterogeneous temporal graph, 8 node types, strict as-of semantics | Closed |
| **1 — SHARE** | RGCN basis sharing + one shared relation-blind attention scorer | Closed since V1 |
| **2 — Markov readout** | Zero-parameter fixed depth: delay→h¹, shortage→h³, impact→h⁴ | Closed since V1 |
| **3 — Validation gates** | Four ordered admissibility tests, not a feature generator | Final form |
| **4 — Output schema** | Six fields that may never substitute for one another | Final form |
| **5 — Action policy** | Evidence tier decides what language is allowed | Final form |

**The one idea underneath all of it:** a component that fails its own acceptance test is *excluded from user-facing output* rather than kept because it looks interpretable.

---

# Layer 1 — SHARE: why it never changed

**The finding.** Seven encoders were built at matched parameter budgets and tested on all three tasks across five seeds. SHARE was **best-or-tied-best on all three simultaneously** — the only architecture that was.

| Encoder | Idea | Result |
|---|---|---|
| GraphSAGE | one transform for every relation | never wins, sets the floor |
| GAT | attention inside a relation | weakest (shortage 0.6560) |
| HGT (the original) | fully independent per-relation parameters | best impact **0.9335**, loses shortage every round |
| RGCN | basis-decomposition sharing | worst impact **0.8509** |
| **SHARE** | RGCN transform **+ one shared attention scorer** | **best or tied on all three** |
| SHARP | + per-relation embedding | tied, costs more |
| SHARK | + per-relation attention matrix | same accuracy, **10× less stable** (impact std 0.056 vs 0.006) |

**Current AUC (V3 grid, retrained independently):** delay **0.7774** (old graph) · shortage **0.9069** (bucketed-Carrier graph) · impact **0.9302** (bucketed-Carrier graph)

**Why it hasn't changed:** re-validated on a second, independently-built dataset across **all twelve** mechanism variants, then SHA-256 frozen. 752,211 parameters, byte-identical from V2 to V3. This is a closed question, not an unexamined assumption.

**A newer graph, the same frozen SHARE.** Shortage's and impact's numbers moved this generation (0.7871→0.9069, 0.9339→0.9302) because the *graph SHARE reads* gained new node and relation types — not because SHARE's equations changed. Parameter count grows with schema size (752,211→956,157 on the enriched schema) purely from the extra per-type input projections and basis coefficients every new node/relation type costs; the shared-basis, shared-attention mechanism itself is untouched and still SHA-256-identical. See the next slide for what changed and why.

---

# How Layer 1 works

![Layer 1 SHARE](https://s3-alpha.figma.com/thumbnails/9182eb46-5156-4ed8-b421-3882c1507ec5?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073420Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0739e4fcc79730f660af75ec28817689dfa815f8c36c98a682d6a068964b1218)

**1. Share the transforms.** Each relation type gets a *mix* of ten shared basis matrices instead of its own private set:

$$W_r = \sum_{b=1}^{10} a_{rb} V_b$$

*Why:* four of the ten relations have only hundreds of edges. Give them private parameters and they memorise; make them borrow from a shared pool and they cannot.

**2. Add one attention scorer that is blind to relation type.**

$$\alpha_{ij} = \mathrm{softmax}_j\big(\mathrm{LeakyReLU}(a^\top[W_r h_i \Vert W_r h_j])\big)$$

*Why:* sharing transforms costs you the ability to single out one alarming neighbour. A single shared scorer buys that back without re-introducing per-relation parameters.

**3. Update and stack four times.**

$$h_i^{(l+1)} = \sigma\Big(W_0 h_i^{(l)} + \sum_r \sum_{j} \alpha_{ij} W_r h_j^{(l)}\Big)$$

Output: five views of every node, $h^0 \dots h^4$, where $h^0$ has seen no neighbours at all and acts as the control.

---

# Layer 2 — Markov readout: why it never changed

**The finding.** Each task's causal chain has its own length. Delay is about *this* shipment. Shortage needs one more hop. Impact aggregates a supplier's whole downstream footprint. So depth is a property of the **task**, not something to learn per node.

**Eight learned alternatives were tried across two projects and ~130 training runs. None ever beat simply fixing it.**

| Alternative | Why it failed |
|---|---|
| Learned Task Gate | mechanistically correct, only 1 of 9 cells significant |
| Rung 4 / Rung 5 | unstable, never beat the fixed choice |
| Variant A (capped gate) | its correction term was **mathematically incapable of moving** — 100% agreement was a restatement, not evidence |
| Variant E (weight averaging) | catastrophic, worse than random |

**What the graph is actually worth (5×5 grid) — updated after the Carrier-hub fix:**

| Task | Features only | SHARE + Markov | Gain | Floor | Clears? |
|---|---|---|---|---|---|
| delay | 0.7416 | **0.7774** | +0.0357 | 0.0874 | no |
| shortage | 0.8727 | **0.9069** | **+0.0343** | 0.0335 | **yes** — margin 0.0008, thin, stated plainly |
| impact | 0.8202 | **0.9302** | **+0.1100** | 0.0252 | **yes** |

**The uncomfortable row from the previous generation is resolved, not walked back.** Shortage's graph used to cost 8.6 AUC points against features alone (0.8727 → 0.7871). Diagnosis found two separate facts, not one: a `Product—REPLENISHED_BY→Shipment` relation genuinely carries shortage's signal (live inbound resupply that features-only cannot see), but a companion `Carrier` node — 5 nodes absorbing the entire shipment population, in-degree growing to 7,136 — destabilised the encoder badly enough to erase the gain on 4 of 5 dataset seeds. Re-granularising that one node type into 312 `(carrier, lane)` buckets (62× lower fan-in, same information — every shipment's true carrier stays exactly recoverable, verified on all 39,184 shipments) fixed it: **shortage passes for the first time in this project's record**, and impact recovers the pass the instability had knocked out. Neither SHARE nor the Markov readout changed — the graph they read did.

**Depth, re-checked rather than assumed.** Because the graph changed, the fixed depth choices were re-tested against two alternative depths apiece, on the frozen backbone, with no retraining. Neither alternative cleared its own stability floor, and both scored below the depth already in use — h¹ remains the best available depth for delay, h³ for shortage. The zero-parameter readout stays exactly as designed; testing it harder is what earns the word "closed."

---

# How Layer 2 works

![Layer 2 Markov readout](https://s3-alpha.figma.com/thumbnails/f4dfdb00-6cbd-45f3-9383-6a56c3818ba5?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073426Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=244f7a27f939b62adb28addd2e9702804f3445267c43aba6499cfe94df9b1a07)

**The whole layer is one lookup — zero parameters, no gradient:**

$$r(\text{task}) = \begin{cases}1 & \text{delay}\\ 3 & \text{shortage}\\ 4 & \text{impact}\end{cases} \qquad \hat h_v = h_v^{\,r(\text{task})}$$

Then a small per-task head turns that one view into a probability:

$$\hat y_v = \sigma(W_{\text{task}}\hat h_v + b_{\text{task}})$$

trained with **focal loss** ($\gamma = 2.0$).

*Why focal loss:* impact is only 2.13% positive. Ordinary cross-entropy would let the model win by always saying "no". Focal loss down-weights the easy negatives so the rare positives still count.

*Why zero parameters is a feature, not a shortcut:* it cannot overfit, it cannot drift between seeds, and it made the "adaptive depth" question falsifiable — three specific reopening conditions were named and none has been met.

---

# Layer 2 — are the probabilities honest?

**A separate question from accuracy, measured directly for the first time this generation.** AUC (previous slides) measures whether the model ranks risky cases above safe ones. It says nothing about whether the number it prints — "36% chance" — is anywhere close to true.

| task | AUC (ranking) | states, on average | actually happens | overstatement |
|---|---|---|---|---|
| shortage | 0.9069 | 36.1% | 5.4% | **6.7×** |
| impact | 0.9302 | 16.2% | 3.3% | **5.0×** |
| delay | 0.7774 | 36.9% | 12.4% | **3.0×** |

**Not a mystery — a known, mechanistic cause.** The training loss (`FocalLoss`, `alpha = 1 − true rate`) provably rebalances *every* task's effective training target to exactly 0.5, no matter how rare the real event is. A single-signed, closed-form-explainable bias is close to the ideal case for a post-hoc fix rather than trial and error.

**The fix: undo the known shift first, then validate any further fit against a new world.** A Bayes prior-shift correction — using each world's own *training-split* positive rate, never the test rate, so it needs no held-out fitting — removed **86–89% of the calibration error on all three tasks for free**. Anything proposed beyond that was fit on 4 dataset-seed worlds and scored only on a 5th it never saw, rotated through all 5, and rejected outright wherever it improved one binned statistic while quietly hurting the proper one — the same transfer failure this project has flagged before with fitted calibration maps.

| task | final correction shipped | final ECE (was) | total reduction | reads as a literal probability up to |
|---|---|---|---|---|
| impact | analytic + temperature + small top-up | 0.0086 (0.1296) | **93.4%** | p < 0.40 — 99.7% of its predictions |
| shortage | analytic only — top-up tried, failed its held-out test | 0.0343 (0.3074) | **88.8%** | p < 0.20 — 99.0% of its predictions |
| delay | analytic only — top-up passed but changed nothing | 0.0281 (0.2461) | **88.6%** | p < 0.20 — 89.6% of its predictions |

**Ranking is untouched — provably, not just presumably.** Every correction applied is strictly monotone in the raw score, so the AUC on the previous slide is unchanged by any of this (max difference 0.0000 for delay and shortage; a floating-point rounding artifact, five orders of magnitude below noise, for impact). Past each task's threshold above, the model still puts its riskiest case first — read that number as a rank, not as a percentage.

---

# The Layer 3 problem: detection is easy, explanation is not

![Detection vs explanation](https://s3-alpha.figma.com/thumbnails/ce258c09-ed16-4b3b-83d2-a82ab3abdcb7?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073556Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=123fb2c8633e5b7d6f6e68b3c312ff86c0692d856be7b2045dc6ae74eb18d793)

**Detection needs only correlation. Explanation needs a number to *mean* something.** We reached 0.78–0.93 AUC on detection and could not turn any internal number into a defensible "because".

Four specific ways numbers lied to us:

1. **Occlusion cites `days_since_dispatch` on 69% of delay explanations.** Elapsed time is *exposure*, not a cause — the generator draws the ETA independently of stress. A dominant mechanical driver looks exactly like an explanation.
2. **A learned node-type gate names one feature for 59% of impact predictions.** An amortised gate with an L1 penalty optimises to a population-level ranking, and a constant answer can never pass a per-instance specificity test.
3. **Every observable is a *descendant* of the real cause.** `stress`, `RESILIENCE` and the event calendar are never emitted. Agreement with verified causal parents is **0.000** for every feature-based method on impact.
4. **The one method that talked about the right kind of object still wasn't faithful.** `operational_cf` cites a genuine co-parent — 1.000 structural agreement — and scores 0.093 faithfulness.

**The lesson that shaped the final design: a high attribution value is not a causal effect, and no amount of post-hoc reading makes it one.**

---

# Layer 3 — the evolution

![Layer 3 evolution](https://s3-alpha.figma.com/thumbnails/38fba77b-1842-4f78-a909-66eb233389bf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073503Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=a70c7d273f8166cb7da598a1bfceba081ea984966cf786f1d7ca1059d3e3faaf)

| Generation | What Layer 3 was | Result | Why it moved on |
|---|---|---|---|
| **V1** | Transformer 2 — global attention to discover hidden supplier links | Discovery **worked**: rank 0.7255 vs 0.500 chance | The hidden link was **causally redundant** — four fusion designs all flipped sign across seeds |
| **V2** | Hidden-coupling retrieval, five independent redesigns | All five hit the **same wall** | Group-mean dilution: a hidden parent split across 4–5 members leaves no pair attributable. A data-generation constraint, not a modelling one |
| **V2 reframe** | Calibrated hypothesis generation | Survived, honestly: AUC **0.817** on observable causes; never claimed the hypothesis it couldn't support (<0.13% confidence) | Kept in the narrower, honest form |
| **V3 Phase 1** | Latent State Estimation — 7 candidate hidden states | Only **2 of 7** cleared: Supply Stress 0.64–0.65, Recovery Capability 0.65 | Supplier Reliability collapsed from a promising 0.5444 single seed to below chance on five |
| **V3 Steps 1–6** | Sufficiency → identifiability → calibration → confidence → threshold | **Zero states finished.** Calibration doesn't transfer across worlds; the confidence signal's *most*-confident bin had the *lowest* AUC | The estimator route was exhausted |
| **V3 Final** | **A validation gate stack** | — | If you can't produce a trustworthy explanation, produce a trustworthy statement about *whether* you have one |

---

# The finalised Layer 3

![Layer 3 four gates](https://s3-alpha.figma.com/thumbnails/abbfc68f-2fc9-4dc1-9cd6-524eb34b9379?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073448Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9aa732f53046d93a4b926654d3e6922f024348b252af14a03895074c5d469072)

**Layer 3 stopped being a thing that produces explanations and became a thing that decides which explanations may be spoken.** Four ordered gates; a later gate is never evaluated for a task whose earlier gate failed.

| Gate | The claim it guards | Verdict as measured |
|---|---|---|
| **0 — Identifiability** | *this effect is visible in observations at all* | delay **passed** (by 0.0013) · impact failed (by 0.0006) · shortage **not computable** |
| **1 — Entity relevance** | *this entity is why the prediction is elevated* | **failed on all 5 methods, both tasks** |
| **2 — Symbolic mechanism** | *this pathway is a real mechanism* | **passed** — 16/16 rules on 111,375 comparisons |
| **3 — Causal effect** | *this action changes the outcome by X* | **failed** — blocked at Gate 0, and the estimator is below chance |

**Per task:**

| Task | Gate 0 | Gate 1 | Gate 2 | Gate 3 | Tier reached |
|---|---|---|---|---|---|
| delay | passed | failed | passed | failed | Prediction + verified mechanism catalogue |
| shortage | not computable | failed | passed (no path ends here) | failed | Prediction only |
| impact | failed | failed | passed | blocked | Prediction only |

**`shortage` is "not computable", not "failed"** — and that distinction is carried into the output, because "we tested it and it didn't hold" and "we could not test it" are different facts.

---

# How Layer 3 works

![Verified SCM](https://s3-alpha.figma.com/thumbnails/ce42d633-003f-4b91-8dd2-e2129d506852?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073548Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=4bd68cc542265e03f0d94417726a6aa2eb15dde2fbd013d614cbed1d959ea55a)

**Gate 0 — does intervening actually change anything?** Fix every random draw per shipment (common random numbers) so the *only* thing that can move an outcome is the intervention, then test the **residual**, not the raw label:

$$\text{residual AUC on } \; Y - g(X, X_{\text{nbr}}) \;\; \text{vs a measured null}$$

*Why the residual:* the raw observable shift ran 0.57–0.71 everywhere and would have passed **every** arm. The residual asks whether anything is left after the ordinary predictor has spoken.

**The floor is the whole trick:**

$$\text{floor} = \max(\text{init-seed spread},\ \text{dataset-seed spread},\ \text{null excursion})$$

*Why:* the single largest signal in the table (0.5494) **fails**, because its spread across worlds is 0.0719. One world would have called it the best result of the build.

**Gate 2 — verify the mechanism, don't trust the narrative.** Every rule was re-implemented standalone and checked against the generator:

$$\text{stress}(s,t) = \min\big(0.95,\ \text{own\_stress}(s,t) + 0.35\textstyle\sum_{\text{co-parents}}\text{own\_stress}\big)$$
$$p_{\text{delay}} = \min(0.80,\ 0.025 + 0.38\cdot \text{stress}\cdot \text{absorption})$$

16/16 rules verified to 0.0 error. The plausible business story *"supplier failure → delay → inventory drop → shortage"* is **rejected** — one of its four links has no verified equation, and three quarters of a story is a hypothesis.

---

# Layer 3 — what's still wrong, and what a real dataset fixes

**Every negative verdict in this build is a *floor* verdict, not a *signal* verdict:** 0.162 against 0.281 · 0.304 against 0.261 · 0.4614 against 0.4341. The signals exist; we cannot yet prove they aren't seed noise.

| Issue now | Why it exists | What a real dataset changes |
|---|---|---|
| Gates pass or fail by 0.0013 and 0.0006 | five worlds is the resolution limit | real supply chains have thousands of suppliers over years — the floor shrinks with real volume, not synthetic reruns |
| Latent state unrecoverable ($R^2$ = 0.001–0.007) | the generator never emits `stress`; every observable is a descendant | real firms **do** observe proxies — audit scores, financial health, capacity utilisation, quality history — the generator simply never had them |
| Shortage not testable | forcing stress changes the *entity set*: only 0.2477% of the ledger keeps its identity | real history is a fixed record; you observe what happened, you don't re-roll the world |
| 96.4% of shipment nodes are already delivered | a synthetic ledger accumulates history it never prunes | a live operational feed is dominated by in-flight shipments, which is where the decisions actually are |

**The honest bound:** Gate 2 passes *because* this world's equations are source code. In a real deployment there is no generator — **the part of this architecture that works best is the part that would not exist.**

---

# The finalised Layer 4

![Layer 4 schema](https://s3-alpha.figma.com/thumbnails/7a44b125-8a24-4c5e-9100-cf5e12d3ab1d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073509Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=dd0696e531362a4972dd90680e358577181c2707b4d459f96924589c35ee3023)

**Every prediction leaves the system as six fields that may never substitute for one another.**

| # | Field | Always present? | Source |
|---|---|---|---|
| 1 | Prediction probability | yes | forward pass — raw **and** calibrated, both always shown |
| 2 | **Evidence status** | yes | a summary of which gates passed, and of nothing else |
| 3 | Entity relevance | only if Gate 1 passed | the passing method's cited factor |
| 4 | Symbolic consistency | only if Gate 2 passed | verified pathways + rule-set version string |
| 5 | Causal effect | only if Gate 3 passed | the passing estimator's Δ with its interval |
| 6 | Confidence | yes | the seed grid — never the probability |

**Fields 3–5 print an explicit "Unsupported" rather than disappearing.** A missing field reads as "not applicable here"; the truth is "not established anywhere".

**Field 2 has three values, not two:** `passed`, `failed`, and `not_computable`. On this build shortage is `not_computable` and impact is `failed`, and merging them would be a lie.

---

# How Layer 4 works

**The three collapses that a normal system makes silently, enforced in code by `assert_distinctions`:**

$$\text{Gate value} \neq \text{causal effect} \qquad \text{Probability} \neq \text{Evidence status} \neq \text{Confidence}$$

Every field carries a `kind` and a `derived_from` provenance string, so the check is a **runtime rejection**, not a code review convention:

1. **Field 5 may only come from a Gate-3-passing estimator.** Writing an attention weight or a relevance score into "causal effect" is the commonest form of this error — rejected by provenance, not by inspection.
2. **Field 6 must come from the seed grid.** A `max(p, 1−p)` "confidence" is just field 1 wearing a hat.
3. **Field 2 must be a function of gate outcomes only.**

**Confidence is measured on two axes, and only one is per-entity:**

$$\text{Var}_{\text{total}} = \underbrace{\mathbb{E}_d[\text{Var}_m(\cdot)]}_{\text{model seed, per-entity}} + \underbrace{\text{Var}_d(\mathbb{E}_m[\cdot])}_{\text{dataset seed, population only}}$$

*Why this matters:* dataset-seed variance is **68–99%** of the total. An uncertainty estimate built only from model-init ensembling would be measuring the smaller source throughout — on delay, about one part in seventy-five.

**One artefact reported rather than hidden:** the isotonic calibrator's top bin steps to exactly 1.000 while the raw probability is ~0.92. That is PAVA behaving as specified — so both columns are always emitted.

---

# Layer 4 — what's still wrong, and what a real dataset fixes

| Issue now | Why it exists | What a real dataset changes |
|---|---|---|
| Fields 3 and 5 are permanently "Unsupported" | no gate they depend on has passed | a real dataset gives real interventions — pilots, dual-sourcing decisions, supplier switches actually executed — which is the only evidence Gate 3 was ever designed to accept |
| Calibrated probability saturates at 1.000 | PAVA tie-merging on a held-out synthetic world | real outcome rates vary continuously; the top bin stops being degenerate |
| Confidence is only per-entity on the smaller axis | "world identity" is a synthetic construct with no real-world analogue | in production there is one world. The dominant variance term stops being an unidentifiable nuisance and becomes ordinary temporal drift you can monitor |
| Evidence status has never once said `passed` for fields 3/5 | five worlds, three model seeds | volume alone moves several of these verdicts — every one of them failed on its floor, not on its signal |

**The schema itself is the deliverable and it is real-world ready today.** It is deliberately built so that a company plugging in its own data changes the *values*, not the contract.

---

# The finalised Layer 5

![Layer 5 policy](https://s3-alpha.figma.com/thumbnails/ae52b80e-241e-4677-998c-86e0a2a750bf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073523Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ec34d9c212879979e637c31b29249afbf9842a3889cbc2c58fbbca83ecc8f691)

**What the system is allowed to recommend is a function of which claims were validated — never of how high the probability is or how confident the model feels.**

| Tier | Admitted when | What may be said |
|---|---|---|
| **Prediction only** | Gate 1 and Gate 2 did not both pass | monitor / contingency language. Options may be *held ready*; no evidence is offered about what acting would do |
| **Relevance + mechanism** | Gates 1 and 2 passed, Gate 3 did not | investigate the named candidates **through their verified path**; effect size not established |
| **Validated causal effect** | Gate 3 passed *for that effect* | intervention recommendation, with interval and business constraints |

**Confidence never promotes a tier.** A high-confidence prediction with no validated relevance is still prediction-only — because recalibration moved ECE by 9×–17× while model ensembling cleared no floor at all. A confident number here is a statement about calibration, not about evidence.

---

# How Layer 5 works

**Each tier's wording is fixed in code, not composed at call time**, so a stronger tier's phrasing cannot leak into a weaker one's output.

**`assert_no_causal_language`** runs a regex over anything emitted below the top tier and raises on:

> *will reduce · reduces · caused by · root cause · prevents · mitigates · attributable to · responsible for*

*Why a regex is not a hack:* the entire gate architecture is defeated the moment the prose promises a causal effect the numbers never established. Phrasing is the last mile, and it is the easiest thing to get wrong by accident.

**What the system actually emits today, for its highest-risk shipment:**

```
MONITOR. delay risk for a09ac6d5-… is estimated at 1.000 calibrated
(0.879 raw, model-seed sd 0.0449). No validated explanation is
available: Gate 1 (entity relevance) did not clear; Gate 3 (causal
effect) did not clear. Contingency options such as activating an
alternate source may be held ready as an operational hedge; this
system offers no evidence about what such an action would do.
```

That paragraph is the product. It is less than we hoped for and more than most systems can defend.

---

# Layer 5 — what's still wrong, and what a real dataset fixes

| Issue now | Why it exists | What a real dataset changes |
|---|---|---|
| Only the bottom tier is ever reached | Gates 1 and 3 fail upstream | the tiers are not aspirational — they are already implemented and tested; they are waiting on evidence, not on code |
| Ranking must be restricted to the "scored" population | 96.4% of shipment nodes are delivered history with no label; ranking over everything returns only those | a live feed ranks in-flight shipments by construction |
| Contingency language can't be costed | no business constraints exist in a synthetic world | real procurement data carries lead times, contract penalties, alternate-source pricing — the missing half of a genuine recommendation |
| Tier 3 has never been exercised on real output | Gate 3 needs interventional evidence | firms run interventions constantly and record them. That log **is** the interventional dataset this layer was designed for |

---

# The architecture end to end

![End to end](https://s3-alpha.figma.com/thumbnails/c1f804d1-7b21-4766-a1fc-7be3d7301322?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAQ4GOSFWC2FGOZM5C%2F20260822%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20260822T073529Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=1a44ed8afa5724cea0a70f9c05244a5c51fb5b3e96ddb53d547f70fadb4d09f6)

**World 42, last test snapshot, highest-risk shipment.**

| Step | What happens | Outcome |
|---|---|---|
| 1 | Layers 0/1/2 encode the snapshot; the delay head fires | 0.8795 raw → 1.0000 calibrated |
| 2 | Gate 0 — identifiability | **passed** (residual 0.5263, null 0.5017, floor 0.0250) |
| 3 | Gate 1 — five relevance methods run | **failed** (best 0.304 vs floor 0.261) |
| 4 | Gate 2 — three pathways traced to verified equations | **passed** (rule set `dc62a005…`) |
| 5 | Gate 3 — counterfactual estimator scored | **failed** (0.4614 vs floor 0.4341) |
| 6 | Layer 4 emits six fields, three carrying "Unsupported" | record admitted |
| 7 | Layer 5 emits MONITOR language, causal phrasing blocked | action issued |

**Two explanations a plausibility-based system would have offered were rejected here:** the 90-day carrier rate (an aggregate of realised outcomes, not a mechanism) and `days_since_dispatch` (exposure, not cause) — **the factor occlusion cites on 69% of delay explanations.**

---

# How the architecture solves the problem

**The problem was never "can we predict supply-chain risk". It was "can anyone act on a prediction they cannot interrogate".**

- **It separates four questions that every commercial risk tool merges into one number:** how likely · how confident · why · what to do. Each has its own test and its own field.
- **It makes the negative result usable.** "No validated explanation is available, here is exactly which test failed and by how much" is an actionable statement. "Risk: 87%, key driver: days since dispatch" is a confident guess dressed as a finding.
- **It refuses at the right layer.** A gate value can never leak into a causal-effect field, a probability can never masquerade as confidence, and causal verbs are blocked by regex below the tier that earned them.
- **It converts the hardest failure mode of ML deployment — quiet over-claiming — into a loud, typed, testable one.**

**What HADES v3 delivers today:** calibrated risk, an honest uncertainty estimate, a verified mechanism catalogue — and no explanation it cannot defend.

---

# This is not a supply-chain architecture

**Nothing in Layers 3–5 mentions shipments.** The pattern is: *a high-accuracy predictor over relational data, where the decision requires a reason, and the reason cannot be read off the model.* That describes most high-stakes ML.

| Domain | Same shape | What the gates would guard |
|---|---|---|
| **Clinical risk** | patient graph, deterioration prediction, latent physiology never charted | "this lab value is why" is exactly Gate 1; "this treatment changes the outcome by X" is exactly Gate 3 |
| **Credit & fraud** | transaction graph, regulator demands adverse-action reasons | Gate 1 is the legal requirement; a saturated-population problem identical to our 96.4% delivered shipments |
| **Predictive maintenance** | equipment graph, latent wear never sensed directly | the descendant-observable problem verbatim — every sensor is downstream of the true state |
| **Grid & network ops** | topology, cascade propagation, co-parent coupling | our co-parent bleed term is the same structure as shared-corridor failure |
| **Public health & logistics** | mobility graph, intervention questions | Gate 0's "is this even identifiable from what we observe" is the first question in every policy evaluation |

**The transferable object is the discipline, not the domain:** an ordered gate stack, a reproduction floor computed on *two* axes, a typed output schema with provenance, and a phrasing policy tied to evidence tier. Swap the SCM and the feature set; everything else survives.

---

# Why this is a benchmark, not just a model

**Most published explainability work reports a number. This reports a number *and* the floor that number has to beat — and then throws away its own best results when they don't.**

- **The dual-seed floor is the contribution.** Measuring across model-init seeds *and* dataset seeds turned three "successes" into failures: an occlusion baseline that cleared at one seed (0.166 vs 0.100), the largest residual signal in the build (0.5494), and a counterfactual edge of 0.524 that was 13× smaller than its own noise. Every one would be publishable at a single seed.
- **It gives commercial prediction networks a vocabulary they currently lack.** Today a model outputs a score and a SHAP chart. HADES outputs a score, an evidence status, and a refusal — machine-readable, provenance-checked, and impossible to collapse by accident.
- **It makes over-claiming a test failure instead of a style choice.** `assert_distinctions` and `assert_no_causal_language` are the first executable answers to "prove your explanation isn't decoration".
- **It draws the honest boundary that the field keeps blurring:** association, intervention, and mechanism are three different products. A model trained only on observation can sell the first, sometimes the third, and never the second — no matter how good its AUC is.

**The next phase of commercial prediction isn't higher accuracy. It's systems that know the difference between what they measured and what they'd like to say.**

---

# What's still missing — and why only real data closes it

**Every negative verdict in this build failed its floor, not its signal.** That is a very specific kind of "not yet".

| Open issue | Measured constraint | The fix a real dataset provides |
|---|---|---|
| Gate 0 margins at the resolution limit | delay passes by **0.0013**, impact misses by **0.0006** | volume: a real chain gives thousands of suppliers over years, not five synthetic worlds |
| Gate 1 fails on all five methods | best is 0.304 against a **0.261** init-seed floor | more seeds *and* real heterogeneity — synthetic worlds are too similar to each other to separate signal from world identity |
| Gate 3 unreachable | the frozen representation, not data scale: **1.0000** with the true latent state, **0.4614** without | real firms observe proxies for stress the generator never emits, and they *log real interventions* — the only true source of rung-2 evidence |
| Shortage untestable | intervening changes the entity set; **0.2477%** identity retention | real history is fixed. You observe the world that happened |
| Calibration doesn't transfer across worlds | each seed's head is miscalibrated in a different direction | there is only one world in deployment; cross-world transfer stops being the binding problem and becomes temporal drift |
| Gate 2's success wouldn't survive the move | 16/16 verified **because the equations are source code** | this is the honest reversal: real data fixes four gates and takes one away. Structure would have to be elicited from domain experts or learned by causal discovery |

**The candidates already scouted:** SupplyGraph (real FMCG heterogeneous graph, no risk labels) · FreshRetailNet-LT (real hourly stockout labels, not graph-structured) · DataCo (real late-delivery labels, flat) · FactSet Revere (the real firm-to-firm network, institutional access only). **None is a drop-in.** That is stated as a gap, not a solved problem — which is, in the end, the whole point of the architecture.
