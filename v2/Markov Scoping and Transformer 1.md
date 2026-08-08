# Markov Scoping vs. Transformer 1 — How Deep Should the Model Look?

*A decision guide for ChainPilot's Layer 2. Companion to `GNN Theory.md` and `Types Of GNNs and Hybrids.md`.*

---

## What this document answers

Layer 2 has to decide one thing that turns out to matter enormously: **how many hops of the supply-chain graph should influence a given prediction?**

There are two ways to decide it.

- **The Markov way** — work it out from the shape of the graph, before training. Structural, free, fixed.
- **The Transformer 1 way** — let the model learn it, per node, during training. Adaptive, expensive, learned.

This document explains both, shows how the first shrinks the second, compares using each alone against using both, and gives a recommendation with the numbers behind it.

---

# PART 1 — THE MARKOV IDEA

## 1.1 The intuition

The original theory, in plain words:

> *If five people have annoyed me today, the odds I'm sweet to the sixth are low. You don't need to know who the five were, or why. My current mood already contains the accumulated damage.*

Applied to suppliers: a tier-1 supplier who is stressed by its own upstream problems will *show* that stress — late shipments, degraded reliability, capacity complaints. So you can read the upstream damage off the tier-1 supplier directly, without tracing the chain back to tier-2, tier-3, tier-n.

This is a **sufficient statistic** argument, and it's the same assumption behind Hidden Markov Models and Kalman filters: *the current observed state screens off the history that produced it.*

## 1.2 What a Markov blanket actually is

In a probabilistic graphical model, the **Markov blanket** of a node `X` is the smallest set of nodes such that, once you know them, nothing else in the graph tells you anything more about `X`.

For a directed model it's three things:

1. **Parents** — what directly causes `X`
2. **Children** — what `X` directly causes
3. **Co-parents (spouses)** — the *other* causes of `X`'s children

**Analogy:** to predict whether a student passes an exam, you need their study habits (parent), their mock-test scores (child), and the difficulty of the paper (co-parent — the other thing that determines the mock scores). Knowing their grandparents' study habits adds nothing once you know the student's own.

That third category is the one people forget, and in ChainPilot it turns out to be the interesting one.

## 1.3 The blanket of a Supplier node in ChainPilot

Taking edges in their causal direction (supplier stress → component availability):

| Blanket component | In our graph | Hops away | Already captured? |
|---|---|---|---|
| **Children** | Components it supplies (`SUPPLIES`) | 1 | Yes |
| **Co-parents** | *Other suppliers of those same components* | 2 | Yes, incidentally |
| **Parents** | Upstream suppliers (`SUB_SUPPLIES`) | 1 | Sparse / absent |
| **Moralised co-parents** | Suppliers sharing an *unobserved* upstream source | ∞ (no path exists) | **No** |

Two consequences fall straight out of this table.

**First: the blanket gives a principled floor of L = 2, not L = 1.** Two suppliers of the same component are in each other's blanket even though no supplier→supplier edge exists between them, and reaching one from the other takes exactly two hops. So "tier-1 is enough" translates to *two* message-passing layers, not one. That number is now derived rather than guessed.

**Second: the last row is Transformer 2's exact job.** Coupling co-parents of an *unobserved* shared parent is what statisticians call **moralisation**. It is the one part of the blanket that message passing structurally cannot reach, because there is no path to walk. This is the sharpest possible statement of why Transformer 2 exists — much better than "global attention finds hidden dependencies."

## 1.4 Correction one — tier is not the same as layer

This is the most common mistake when applying the theory, and it would cut the wrong axis.

"Tier-1 with maximum weight, tier-2 with minimal weight" is a claim about **supplier-to-supplier upstream distance**. GNN layer count `L` counts **hops in the heterogeneous graph**, and most of those hops aren't supplier-to-supplier at all.

Look at the BOM path:

```
Supplier → Component → Product → Order → Customer
```

That's four hops and contains **zero** tier traversal. Capping `L = 2` wouldn't limit how far upstream you look — it would stop a supplier from ever seeing the customer orders its components end up in, which is precisely the signal shortage prediction needs.

**The correct implementation is a relation-scoped depth budget:**

- Full depth along BOM relations (`SUPPLIES`, `USED_IN`, `ORDERED`, `PLACED_BY`)
- Capped or exponentially decayed along `SUB_SUPPLIES` — scale by `α^tier`, α ≈ 0.2–0.3

In PyTorch Geometric this is a per-relation layer mask, or simply a scaling factor on HGT's existing per-meta-relation prior `μ`. **Cost: zero new parameters.** HGT already has that prior; you're just setting it deliberately instead of letting it float.

## 1.5 Correction two — the blanket is per-target, not global

This is the observation that turns the Markov idea from a constraint into a design.

We have three prediction heads, and their targets sit at **different positions in the graph**:

| Head | Target lives on | Blanket radius | Should read |
|---|---|---|---|
| Delay | Supplier / Shipment | ~2 hops | `h²` |
| Shortage | Product / Warehouse | ~3 hops | `h³` |
| Impact / order-at-risk | Order / Customer | ~4 hops (full BOM) | `h⁴` |

A single global `L` is wrong for all three at once — too deep for delay (over-smoothing, wasted compute, unreadable explanations) and too shallow for impact.

**So the Markov design is: run the encoder to L = 4, retain every intermediate layer, and give each head its own readout depth.**

This is the whole mechanism. There's no new module, no new parameters — just a decision about which tensor each head reads from.

## 1.6 Two honest caveats

**The screening is only approximate.** Claim A requires `P(risk | tier-1 state) = P(risk | tier-1 state, upstream)`. That holds exactly only if the tier-1 observation is complete and noise-free. `reliability_history` is lagged, noisy and partial, so some upstream information survives. *How much* is an empirical question — which is exactly why it should be tested rather than asserted.

**It's weakest where the system claims most value.** The screening works for stress that has **already surfaced** in behaviour. But the pitch is early warning — catching disruption "before it fully unfolds." A tier-2 failure that hasn't yet degraded tier-1 observables is exactly the case where deep visibility would pay, and exactly the case Claim A discards.

That doesn't kill the theory, but it changes the honest framing: **we cap at tier-1+ε because tier-2 data is sparse-to-absent, and Claim A is the argument for why that costs less than it sounds.** Which is a stronger position than presenting it as a free choice.

## 1.7 Prior art — this is well-trodden

Claim A is a supply-chain instantiation of an established result. Citing it properly makes the project stronger, not weaker.

| Work | What it establishes |
|---|---|
| Li, Han & Wu (2018); Oono & Suzuki (2020) | **Over-smoothing** — deep GNNs converge to indistinguishable node representations. Bounded depth is more *accurate*, not just cheaper. |
| GraphSAGE — Hamilton et al. (2017) | Neighbourhood sampling to bound receptive field, motivated by tractability and speed |
| **APPNP — Klicpera et al. (2019); PPRGo — Bojchevski et al. (2020)** | Personalised-PageRank propagation, influence decaying as `α(1−α)^k`. This is *literally* "tier-1 max weight, tier-2 minimal weight" as a closed form. **The closest formal match.** |
| SGC — Wu et al. (2019) | Fixed low-order propagation matches deep GNNs on many tasks |
| Jumping Knowledge — Xu et al. (2018) | Per-node receptive field; already cited in our docs |
| Graph Information Bottleneck — Wu et al. (2020); GSAT — Miao et al. (2022) | Modern formalisation of "minimal sufficient subgraph" — the blanket idea for GNNs, and directly relevant to explainability |
| Koller & Sahami (1996); Aliferis et al. (HITON/MMMB) | Classical result that the blanket is the minimal sufficient feature set |

**Claim B (dyadic risk) is a different story** and is covered separately — it maps to the *preferred customer status* literature (Schiele, Calvi & Gibbert 2012; Steinle & Schiele 2008), which is descriptive and has essentially never been formalised into a computed risk score. That gap is where the originality lives.

---

# PART 2 — HOW MARKOV REDUCES THE NEED FOR TRANSFORMER 1

## 2.1 What Transformer 1 was for

Per the original spec (`updates/Supplier_Risk_Prediction.md` §6.2), T1 is a small Transformer that reads each node's four-token depth sequence `[h¹, h², h³, h⁴]` and learns an attention-weighted blend, producing:

- a fused embedding `z_i` per node
- four depth weights per node, summing to 1

It exists to answer: *how deep should this particular node look?*

## 2.2 What the blanket replaces

| T1's job | Markov equivalent | Cost |
|---|---|---|
| Choose depth per prediction task | Per-task readout depth from blanket geometry (§1.5) | 0 params |
| Downweight deep hops for `SUB_SUPPLIES` | Relation-scoped decay `α^tier` on HGT's `μ` prior | 0 params |
| Provide evidence about depth sufficiency | Nested L-sweep with significance testing | 4 runs instead of 6 |
| Mitigate over-smoothing | Shallower effective readout does the same thing | 0 params |

That covers most of T1's stated purpose, structurally, for nothing.

## 2.3 The one thing it does *not* replace

**Per-node adaptivity.** The blanket gives one depth per *task*. It cannot express that a single-product leaf supplier and a multi-division hub supplier need different depths.

And critically, it cannot express **conditional** behaviour like:

> *If tier-1 and tier-2 are both risky, the chain is already clearly risky — stop, tier-3 adds nothing. But if tier-1 looks clean while tier-2 is risky, that's ambiguous — expand to tier-3 to find out whether tier-1 is genuinely buffered or just hasn't broken yet.*

A Markov blanket is **structural** — defined by topology, not by node values. A supplier's blanket is identical whether that supplier is clean or on fire. So it will *strictly* enforce a fixed radius and will never do this.

What the above describes has a different name: **context-specific independence** (Boutilier et al., 1996) — independence that holds for some *values* of the conditioning variables but not others.

Three ways to get it, cheapest first:

1. **A deterministic gate in the Risk Intelligence Layer.** The rule is already written in words; write it in code. Run the shallow readout for everything, flag the ambiguous nodes, re-score those with deeper propagation. Zero parameters, fully auditable, matches the weighted-risk-formula precedent.
2. **A confidence-triggered cascade.** Same shape, but triggered by the existing confidence signal (softmax margin / MC-dropout variance) rather than a hand-written rule. The classic cascaded-classifier pattern.
3. **Transformer 1** — the learned, general, expensive version. Prior art: Spinelli et al.'s Adaptive Propagation GNN (2020) does per-node halting; DAGNN (Liu et al., 2020) does adaptive receptive field.

**Blocking constraint:** every version presumes tier-3 data exists to expand *into*. `SUB_SUPPLIES` is currently sparse-to-absent and not committed. Adaptive expansion into an empty edge set returns nothing. Specify it now; gate it on data later.

## 2.4 Does it also remove Transformer 2?

**No — the opposite.** Moralisation over the visibility frontier (§1.3, last row) is a blanket concept with no other implementation. Message passing cannot create an edge where no path exists. The blanket framing is the best justification T2 has ever had, and it also tells you how to bound its cost: restrict the attention pool to structurally motivated pairs (co-suppliers within a `component_type`, frontier-flagged nodes) instead of dense N².

**Summary: the blanket subtracts the component we could least validate, and sharpens the one we could least justify.**

---

# PART 3 — SUBSTITUTING T1 WITH MARKOV

## 3.1 What the substitution looks like

**Before:** encoder → retain `h¹–h⁴` → T1 learns per-node fusion → single `z_i` → all three heads read `z_i`.

**After:** encoder → retain `h¹–h⁴` → delay head reads `h²`, shortage head reads `h³`, impact head reads `h⁴`.

Note that `h¹–h⁴` retention (§6.1) survives either way, so nothing you build now blocks adding T1 back later.

## 3.2 Pros of substituting

- **Zero new parameters** — decisive when you have only a few hundred positive disruption labels
- **Fewer training runs** — 4 instead of 6, since the dual-run protocol and CV stability gate disappear with T1
- **Defensible in one sentence** — *"delay's target sits on the Supplier node, its blanket is 2 hops, so the delay head reads layer 2"* is checkable by an examiner; a stability score over learned attention weights is not
- **No methodological failure modes** — no circular reasoning to guard against, no "inconclusive" outcome to explain
- **Smaller explanations** — the GNNExplainer computation subgraph shrinks dramatically (see §4.3)
- **Works with the data we actually have** — needs no tier-2 edges to exist
- **Fixes a real modelling error** — one global depth was wrong for three heads with three different target positions

## 3.3 Cons of substituting

- **No per-node adaptivity** — every supplier of a given type gets the same depth
- **Asserted, not measured** — unless the L-sweep validates it, it's a design claim rather than an experimental result
- **No research novelty** — sound engineering, but nobody calls "we chose readout depth per task" a contribution
- **Purely structural** — the conditional tier-3 behaviour needs the bolt-on gate from §2.3
- **Fixed per task** — if BOM path length varies a lot across product families, one number per head is wrong for some of them

## 3.4 Why T1 could not do the job it was assigned anyway

Worth stating plainly, because it's the strongest single argument.

T1 was specified as the **empirical test of Markov Claim A**. It cannot be. Attention weights measure what the model **used**, not what was **sufficient** — and those come apart in both directions:

- High weight on `h³` because hop-3 carries *redundant but easier-to-extract* signal → conditional information zero
- Near-zero weight on `h⁴` because of over-smoothing or optimisation dynamics → the hop might still carry non-redundant signal a better-conditioned model would find

A blanket claim is a **conditional independence** statement, and it's tested by a nested model comparison: train L = 1, 2, 3, 4 under identical splits and test whether each ΔAUC is significant (DeLong, or bootstrap CIs). The depth at which improvement stops **is** the empirical blanket radius.

The dual-run methodology and the 5-fold stability gate in §6.2 are careful, well-motivated stabilisation — of the wrong estimator.

---

# PART 4 — THE THREE CONFIGURATIONS COMPARED

All figures at `d = 64`, `L = 4`, 8 node types, 10 meta-relations, 5,000 nodes / 20,000 edges. Derivations in the appendix.

## 4.1 Headline numbers

| | Markov only | T1 only | Both |
|---|---|---|---|
| **Parameters** | 634,091 | 734,059 (+16%) | 734,059 (+16%) |
| **Forward pass** | 1.97 GFLOP | 5.90 GFLOP (**+200%**) | 5.90 GFLOP (+200%) |
| **Training runs** | 4 | 6 | 5 |
| **Adapts per supplier** | No | Yes | Yes |
| **Can test the depth theory** | Yes (L-sweep) | No | Yes (L-sweep) |
| **Risk of circular reasoning** | None | High | None *if ordered correctly* |
| **Labels needed to be worth it** | Any | ~1,000+ | ~1,000+ |
| **Research novelty** | Low | Medium | High |
| **Can fail and yield nothing** | No | Yes | Unlikely |

**Note that T1-only and Both have identical parameter counts.** The Markov prior is a regularisation and initialisation choice — it costs nothing. Everything "Both" adds over "T1 only" is free.

## 4.2 The surprise — T1 triples inference cost

T1 looks tiny at ~100K parameters against HGT's 634K. But it costs **twice as much to run as the entire 4-layer encoder.**

The reason: HGT's cost is dominated by *edges* (sparse — 20K of them), while T1 runs dense `12d²` projections on **every node × 4 depth tokens × 2 layers**. Sparse structure is what makes GNNs cheap; T1 discards it and does dense work everywhere.

**Parameter count badly understates T1's cost.** Judge it on FLOPs.

*Fairness note:* this is full-graph batch scoring. For NFR-01's single-entity request path, T1 only runs over that request's subgraph, so the 2-second latency target is not threatened either way. The 3× hits batch inference and training.

## 4.3 Effect on output quality

**Improves under Markov:**

- **Explainability — the biggest win.** At L = 2 the GNNExplainer computation subgraph is ~73 nodes; at L = 4 it's ~4,700, i.e. ~94% of the graph. Only one of those is checkable by a procurement manager. This also directly reduces ML-03 (explanation instability), since a smaller search space yields more stable subgraphs.
- **Generalisation** — fewer parameters against scarce positives means less memorisation.
- **Calibration** — less over-smoothing means less homogenised output. This matters specifically because the chatbot's "78% delay probability" has to be interpretable at face value.
- **Inductive scoring** for new suppliers (F-07) — bounded receptive field is what makes cold-start scoring possible at all.

**Degrades under Markov:**

- **Recall on correlated / systemic failure** — the cliff-edge case where a clean-looking tier-1 breaks because a shared invisible upstream failed. This is the real cost, and it's why Transformer 2 should be kept even when T1 is dropped.

**Not affected, contrary to intuition:** affected-order tracing (FR-GNN-04) is done by **graph reachability**, not by the model's receptive field. Capping readout depth does not shorten the trace.

### The k-hop reach that motivates all of this

| L | Unsaturated reach | Share of a 5,000-node graph |
|---|---|---|
| 1 | ~9 | 0.2% |
| 2 | ~73 | 1.5% |
| 3 | ~585 | 12% |
| 4 | ~4,681 | **~94%** |

At L = 4 the receptive field is effectively the whole graph — which makes HGT's "type-aware **local** structure" justification close to vacuous at that depth. Supply chains are hub-heavy, so real saturation arrives faster than this uniform-branching estimate, not slower.

## 4.4 The measurement caveat that governs everything

At prototype scale the differences between these configurations may be **statistically invisible**. With AUC ≈ 0.80:

| Positive labels | 95% CI on AUC | Smallest detectable model-vs-model gap |
|---|---|---|
| 200 | ±0.037 | ~0.037 |
| 500 | ±0.024 | ~0.024 |
| 1,000 | ±0.017 | ~0.017 |

If GraphSAGE scores 0.79 and HGT scores 0.81 on 500 positives, **no difference has been measured.** Report bootstrap or DeLong confidence intervals on every ablation row, never point estimates. "The architectures were not distinguishable at this label volume" is a legitimate, defensible finding — and reporting it is what makes the honesty commitment in ML-05 operational rather than aspirational.

---

# PART 5 — WHY "BOTH" IS STRONGER THAN EITHER

## 5.1 The residual is the finding

Running both isn't averaging two depth mechanisms. It's setting a **structural prior** (blanket-derived per-task depth) and letting T1 learn the **deviation** from it.

That produces something neither gives alone: *where does the model disagree with the blanket, and why?*

> *"Depth attention tracked blanket-predicted depth for 94% of suppliers, and deviated upward specifically for multi-division hub suppliers and frontier-flagged nodes."*

Compare that to what each option produces on its own:

- **T1 alone:** "mean depth was 1.8" — a measurement with no reference point
- **Markov alone:** "we chose depth from graph geometry" — an assertion with no test
- **Both:** a tested structural hypothesis *with a characterised failure mode*

The third is the strongest epistemic position, and it's the one the rest of the project's documentation already aims for.

## 5.2 It also fixes T1's worst practical weakness

Unbiased T1 has to learn depth preference from scratch on a few hundred labels — which is precisely why ML-11 warns the weights may be noise rather than a learned preference. A structurally motivated prior makes it far more sample-efficient and makes the stability gate much more likely to pass.

## 5.3 Three conditions — the first is non-negotiable

**1. Claim A must move to the L-sweep *first*.**

This is Fix B applied to ourselves. Biasing T1's depth attention toward a blanket prior and then citing its weights as evidence about graph structure is the *exact* circularity that Fix B was written to prevent — the conclusion would be engineered into the training objective.

Composing them is legitimate **only because** Claim A has moved out of T1. Once T1 stops being a measuring instrument, priors on it are just regularisation, and regularisation is fine. Order matters: Markov-as-prior is illegal while T1 is the Claim A test, and legal the moment it isn't.

**2. Prior and deviation, not two voters.** Markov sets the centre; T1 learns departure from it. Not two independent mechanisms whose outputs get combined — that recreates the T1/T2 functional-overlap problem §10 already flags.

**3. Enough labels.** Below ~500 positives the deviation can't be distinguished from noise. Above ~1,000 it becomes measurable.

---

# PART 6 — SHRINKING T1: THE LADDER

## 6.1 Why the prior justifies a smaller T1

T1 is sized for learning depth preference **from scratch**. That's a hard job: discover, unaided, a four-way distribution per node from a few hundred labels.

Give it a structural starting point and the job changes completely: **learn a small correction to a known answer.** Most of those corrections are near zero. That is a far smaller function, and it needs far less machinery.

There's a second, sharper argument. Read the original output spec:

> *(a) a fused embedding `z_i` per node; (b) per-node depth attention weights (**4 values summing to 1**)*

A convex combination of `h¹–h⁴` with learned weights summing to 1 **is a gate, not a Transformer.** The multi-head QKV machinery and the feed-forward network are doing considerably more than the stated output requires.

Without a prior you could argue the extra capacity is needed, since the model must discover the whole depth structure unaided. **With the prior, that argument evaporates.** You've handed it the answer's neighbourhood; it only needs enough capacity to say *"this hub supplier needs half a hop more than the blanket predicts."*

## 6.2 The ladder

| Rung | Version | Params | GFLOP | vs. as-specced |
|---|---|---|---|---|
| 1 | As specced — 2-layer, ff=4 | 99,968 | 3.93 | — |
| 2 | 1-layer, ff=4 | 49,984 | 1.97 | 2× smaller |
| 3 | 1-layer, ff=2 | 33,472 | 1.31 | 3× smaller |
| 4 | 1-layer, attention only (no FFN) | 16,960 | 0.66 | 6× smaller |
| 5 | **Prior-initialised gate (d→32→4)** | **2,212** | **0.022** | **45× fewer params, 181× fewer FLOPs** |
| 6 | Fixed depth (Markov only) | 0 | 0 | — |

## 6.3 Pros and cons of each rung

### Rung 1 — 2-layer Transformer, ff=4 (as specced)

- **Pros:** maximum expressiveness; can model complex interactions between depth levels; matches the "GNN × Transformer" framing most literally.
- **Cons:** 100K parameters against a few hundred positives — capacity spent in the worst possible place; triples inference FLOPs; almost entirely redundant once a prior exists; the highest overfitting risk on the ladder.
- **Verdict:** only justified with no prior and >2,000 positive labels. Not our situation.

### Rung 2 — 1-layer, ff=4

- **Pros:** halves everything for very little expressiveness loss; a single attention layer over four tokens is already plenty to blend them; keeps a full FFN for non-linear mixing.
- **Cons:** still 50K parameters and a full extra encoder's worth of FLOPs; the FFN is the largest single component and is doing the least useful work.
- **Verdict:** a reasonable compromise if you want to keep something genuinely Transformer-shaped. Not the best value on the ladder.

### Rung 3 — 1-layer, ff=2

- **Pros:** the FFN expansion drops from 4× to 2×, which is where most of the remaining cost lives; 33K parameters is defensible; still a real Transformer block.
- **Cons:** the reduced FFN is a somewhat arbitrary cut — you're shrinking a component without a principled reason for that specific size.
- **Verdict:** fine, but you're now paying 33K for machinery you can't clearly justify. If you're cutting this far, keep going.

### Rung 4 — 1-layer, attention only (no FFN)

- **Pros:** keeps the genuinely useful part — attention between the four depth tokens, so `h¹` and `h³` can interact when setting weights — and drops the part that mostly adds capacity; 17K parameters; a clean conceptual story ("depth attention, nothing more").
- **Cons:** an attention block without an FFN is an unusual configuration and would need justifying in a write-up; still 30× more expensive than rung 5 for a benefit that is not demonstrated.
- **Verdict:** the best rung *if* you can show the four depth tokens genuinely need to interact. Only climb here with evidence.

### Rung 5 — Prior-initialised gate (d→32→4) ★

- **Pros:** 2,212 parameters and 0.022 GFLOP — **effectively free**, ~1% overhead rather than 200%; produces exactly the output the spec asks for (four weights summing to 1); **keeps the entire research artifact** — the residual between blanket-predicted and learned depth survives untouched; initialise its output at the blanket depth and the untrained model reproduces Markov-only behaviour exactly, so training can only improve on it; if the weights never move meaningfully, *that is itself the Claim A evidence*, cleanly and for 2K parameters.
- **Cons:** no attention between depth tokens — the gate reads the node once and emits weights, rather than letting `h¹` and `h³` interact. *Largely fixable:* feed the gate a concatenation of pooled `h¹–h⁴` instead of raw node features, still under 5K parameters. Also slightly weakens the "GNN × Transformer" narrative — though Transformer 2 and the global prediction head both still carry genuine Transformer content, so the framing survives.
- **Verdict:** **the recommended rung.**

### Rung 6 — Fixed depth (Markov only)

- **Pros:** absolutely free; nothing to train, tune, persist or explain; the simplest thing that could work.
- **Cons:** no adaptivity at all; **no residual, so no research artifact** — you lose the single strongest reason to have done any of this; hub vs. leaf suppliers treated identically.
- **Verdict:** the correct Phase-1 baseline. But rung 5 costs ~1% over it and buys back the entire finding, which is an extremely good trade.

## 6.4 Which is best

**Rung 5 — the prior-initialised gate.**

The reasoning in one paragraph: it costs about 1% overhead instead of 200%, it produces exactly the output the specification asks for, it preserves the residual that makes "both" worth doing at all, and it degrades gracefully — initialised at the blanket prior, the worst case is that it learns nothing and you're back to Markov-only behaviour, having lost 2,212 parameters.

**Climb the ladder only on evidence.** If the gate's learned weights are consistently pinned at the edges of what a gate can express, that's a signal you need real attention between depth tokens, and rung 4 becomes justified. Climbing *with* evidence is defensible. Starting at rung 1 and hoping is exactly what produced a 100K-parameter component for a problem with a few hundred labels.

---

# PART 7 — RECOMMENDATION

## 7.1 Sequence, don't choose

**Phase 1 — Markov only.**

- Encoder at L = 4, retain `h¹–h⁴` (nothing is burned; the upgrade path stays open)
- Per-task readout depth: delay ← `h²`, shortage ← `h³`, impact ← `h⁴`
- Relation-scoped decay on `SUB_SUPPLIES` via HGT's existing `μ` prior
- Claim A answered by a 4-point L-sweep with confidence intervals, not by attention weights
- Optional: deterministic ambiguity gate in the Risk Intelligence Layer for the conditional tier-3 behaviour
- **Cost: zero new parameters, 4 training runs**

**Phase 2 — add the gate, gated on positives clearing ~1,000.**

- Rung 5: prior-initialised depth gate, output initialised at the blanket depth
- Report the **residual** as the result
- **Cost: +2,212 parameters (~1% FLOP overhead), +1 training run**

## 7.2 The one thing that must not slip

**Move Claim A to the L-sweep before adding any prior to a learned depth mechanism.** Get that order wrong and every subsequent result is circular — the same trap Fix B was written to catch, arriving from a different direction.

## 7.3 Summary table

| Decision | Answer |
|---|---|
| Does Markov remove Transformer 1? | Mostly — replaces its depth-selection role for free. Doesn't replace per-node adaptivity. |
| Does Markov remove Transformer 2? | **No** — it strengthens the case for it (moralisation over the visibility frontier). |
| Does Markov enforce a strictly fixed depth? | **Yes.** Adaptive behaviour needs context-specific independence, not a blanket — use the deterministic gate. |
| Best configuration right now? | **Markov only.** |
| Best configuration at ~1,000+ positives? | **Both, with a rung-5 gate.** |
| Is T1 alone ever right? | **No.** Most expensive, most likely to yield nothing, and it can't test the claim it was built to test. |
| Does "both" reduce computational cost? | Not vs. Markov alone. Yes vs. the current spec (5 runs not 6, no dual-run protocol). And the prior *enables* the 45× shrink. |

---

## Appendix — Assumptions behind the numbers

Nothing here was measured; all figures are derived analytically from the specification, since no code exists yet.

- **Hidden dimension** `d = 64`; attention heads `h = 4`. Neither is specified anywhere in the current documentation — that is itself a gap worth closing.
- **8 node types** (Doc 6 §5), feature widths from Doc 10 §7.
- **10 Phase-1 meta-relations**, counting `SHIPS_FROM` twice since it targets both Supplier and Factory.
- **Graph scale:** 5,000 nodes (NFR-02 ceiling), ~20,000 edges (average undirected degree ≈ 8).
- **HGT per-layer parameters:** `4T·d²` (per-type K/Q/V/A) + `2R·d²/h` (per-relation attention and message) + per-relation prior + skip terms.
- **Transformer per-layer parameters:** `4(d²+d)` for QKVO + `2·ff·d²` for the FFN + layer-norm terms.
- **FLOPs** counted as 2 × MACs; HGT split into a node-projection term and an edge term; T1 evaluated over 4 tokens per node across all 5,000 nodes.
- **AUC confidence intervals** via the Hanley–McNeil standard error at AUC = 0.80.
- **k-hop reach** uses uniform branching, which *understates* saturation on hub-heavy graphs.

Recompute all of these once real dimensions and a real graph exist. The relative orderings are robust; the absolute values are not.
