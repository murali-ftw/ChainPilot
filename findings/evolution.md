# HADES Architecture Evolution — From HGT + Short Head + Global Head to SHARE + Markov Blanket

**Purpose:** a single, chronological record of how HADES's architecture arrived at its current
form — SHARE as the encoder, the Markov Blanket as the per-task depth readout, and Layer 3
(hidden-dependency discovery) scrapped as a discovery mechanism after exhaustive testing across
two projects. This document exists so the decision to drop Layer 3 does not read as an
unexplained retreat: it is the end of a long, evidenced line of experiments, each one closing off
a specific possibility rather than a change of direction taken lightly.

**Sources.** Everything below is drawn from measured results, not estimates: `HADES_v1/reports/
info.md`, `layer1.md`, `layer2.md`, `layer3.md` (V1), and `HADES_v2/reports/layer2_testing.md`,
`layer3_testing.md` (V2). Numbers are quoted as reported in those documents.

---

## 1. The starting point — HGT + short head + global head

HADES's architecture began with three components decided up front, before any of the empirical
work that follows:

- **HGT (Heterogeneous Graph Transformer) as the encoder.** Every one of the graph's 20
  meta-relations gets its own fully independent attention and message matrices — the most
  expressive, most parameter-heavy encoder design available, and the project's original choice
  precisely because it made no simplifying assumption about which relations mattered more than
  others.
- **A short, fixed-depth prediction head.** Every task read from the same final encoder layer,
  with no reasoning about whether a shallower or task-specific read might serve a given
  prediction better. Depth was not yet a design question — the encoder's last layer was simply
  "the" representation.
- **A global head.** The architecture always intended a second, non-local component — a global
  attention mechanism that could connect two entities with no path between them in the graph,
  aimed specifically at hidden dependencies the schema could never encode as an edge (two
  suppliers secretly sharing an unmodeled upstream plant, for instance). This is what later
  became Transformer 2 / Layer 3.

Three questions followed from this starting point, and they became the project's three layers:
which encoder actually deserves to be the default (Layer 1), whether every task should really
read the same depth (Layer 2), and whether the global head's hidden-dependency ambition actually
pays off (Layer 3). Each is addressed in turn below, in the order the evidence accumulated.

---

## 2. Layer 1 — choosing the encoder

**The question:** does HGT's full per-relation independence actually earn its cost, or does a
cheaper design do as well or better?

Seven encoders were built and matched at comparable parameter budgets, each tested across five
seeds on delay, shortage, and impact AUC.

| Architecture | Idea | Verdict |
|---|---|---|
| GraphSAGE | One shared transform for every relation — the cheapest possible baseline | Never wins outright; establishes the floor |
| GAT | Attention within each relation, no relation-type awareness | Weakest architecture across every round (shortage AUC 0.6560) |
| **HGT** | Full per-relation independence — the original design | Decisively best on impact (0.9335) but loses to the RGCN family on shortage every round |
| RGCN | Basis-decomposition sharing instead of full independence | Confirms the thin-relation-overfitting hypothesis on shortage, but worst on impact (0.8509) |
| **SHARE** (RGCN + shared attention) | RGCN's transform, plus one shared, relation-agnostic attention scorer | **Best or tied-for-best on all three tasks simultaneously** |
| SHARP (RGCN + relation embedding) | SHARE plus a small per-relation tag fed into the scorer | Statistically tied with SHARE, costs more, no measurable gain |
| SHARK (RGCN + per-relation attention matrix) | The most expressive attention design tested | Same accuracy range as SHARE/SHARP, but an order of magnitude more seed-to-seed instability (impact std 0.056 vs SHARE's 0.006) |

**Why SHARE won.** Four of the graph's ten relations are structurally thin (hundreds to a few
thousand edges against tens of thousands for the rest). HGT gives each of them a fully
independent parameter set anyway, which overfits exactly those thin relations — the mechanism
that explains RGCN's win over HGT on shortage. But RGCN's shared-transform simplicity costs it
impact, where a genuinely standout neighbor needs to be singled out regardless of which relation
it arrived through. SHARE keeps RGCN's thin-relation protection (shared basis pool, no
per-relation matrices) and adds one shared, relation-blind attention scorer on top — recovering
impact (0.9365–0.9393, tying HGT) without giving back shortage (beating HGT by +0.0147–0.0167).
It is the only architecture that never shows elevated seed-to-seed instability, and it is cheaper
to build and train than SHARP or SHARK, which added complexity without any measurable return.
**SHARE became the production encoder — the original "HGT" component was replaced entirely.**

This finding replicated on V2's benchmark with a fresh, independently-generated dataset
(`reports/phase7_training_results.md`): SHARE remained best-or-tied-for-best on all three tasks
across all twelve of V2's mechanism variants, and SHARP's extra cost was confirmed to buy nothing
across the full twelve-variant sweep, not just the one dataset V1 had access to.

---

## 3. Layer 2 — from a fixed final-layer read to the Markov Blanket, and why per-node adaptivity never won

**The question:** should every task really read the encoder's last layer, or does depth deserve
to be a decision — either per task, or per node?

### 3.1 V1 — five independent mechanisms, one winner

| Mechanism | Idea | Outcome |
|---|---|---|
| Learned Task Gate | One learned depth-blend per task, KL-anchored to a theory prior | Correctly recovers each task's known depth preference, but only 1 of 9 cells clears significance — mechanistically correct, practically inert |
| **Markov Floor** | Zero-parameter, per-task *fixed* depth derived from graph structure (delay→h¹, shortage→h³, impact→h⁴) | Delay +0.0072 AUC, consistent across all 5 seeds, at zero cost — **became the production baseline** |
| Rung 4 / Rung 5 | Per-node MLP or attention gate, free to deviate from Markov per node | Never beat Markov on AUC; wildly unstable — some runs stayed pinned to Markov, others let 75–98% of nodes drift, with no accuracy difference between the two outcomes |
| **Variant A** | Rung 5 + a hard cap on how far the gate's correction can drift from Markov | **100% match rate, every seed, every cap tested — zero deviating nodes, and no accuracy cost.** Adopted as a strict, safety-hardened upgrade over plain Markov |
| Variant B | Rung 5 + explicit structural features (degree, node type) fed to the gate | Stabilized shortage, left delay just as unstable |
| Variant C | Rung 5, gate frozen early then unfrozen at a low learning rate | Large stability gain, not a complete fix |
| Variant D | Rung 5 with per-depth projection instead of mean-pooling | No improvement — the instability was never caused by the information loss this fixed |
| Variant E | Post-hoc weight-averaging across independently-trained seeds | **Catastrophic failure** — scored worse than random guessing |

V1's conclusion: no depth-selection mechanism ever produced a statistically defensible accuracy
gain over the fixed Markov Floor. Variant A's contribution was safety, not accuracy — a
structural guarantee that a per-node gate could never drift arbitrarily far from the trusted
prior, which V1 adopted as the production default specifically *because* it never showed the
bimodal instability every unconstrained gate design did.

### 3.2 V2 — the same question asked on a dataset finally capable of answering it

V1's dataset had structurally uniform chain length, so its five depth-gate variants never had
genuine per-node heterogeneity to adapt *to* — every negative result could, in principle, have
been an artifact of that flatness. V2's Mechanism J varies chain length per product, which made
this the first dataset in either project where adaptive depth could actually be tested against
real heterogeneity.

The result did not change, and it became *stronger*, not weaker: all twelve delta-of-delta cells
(testing whether any gated arm improved more under real heterogeneity than without it) flipped
sign across five seeds. Base Rung 5, Variant A, B, and C were all reproduced byte-identical from
V1 and re-tested; none beat Markov consistently anywhere.

A follow-up diagnostic then explained *why*, mechanically, rather than leaving it as an
unexplained negative:

- **Variant A was never actually being tested.** Its logit is `fixed_prior + λ·tanh(residual)`,
  where the prior gap is 8.0 and the maximum possible correction is `2λ`. Every λ ever tested in
  either project (0.1–1.0) is 8 to 40 times too small to ever flip the decision — Variant A's
  predictions were Markov's predictions, node for node, on every one of 195,029 scored rows,
  every seed, every variant. Its "100% stability" in V1 was a restatement of an inequality, not
  evidence of a working, cautious gate.
- **Base Rung 5, which *could* move, didn't do what it was built to do.** It never learned
  per-node preferences; it collapsed into picking one global depth for the *entire task*, and
  which depth it picked changed randomly by seed — a per-node gate behaving as a per-task
  constant, traced to the gate mean-pooling a node's five depth-token embeddings into one vector
  before ever scoring them.
- **When Variant A's cap was widened past the point where movement became mathematically
  possible (λ=6.0, 10.0), the gate moved on 80–90% of nodes — and accuracy got measurably worse,
  not better.**
- **Depth representations genuinely differ from each other** (h⁰ and h⁴ are nearly orthogonal),
  ruling out "the depths all look the same" as the explanation — there was real information to
  choose among. But that difference does not correlate with chain length in any direction that
  holds across seeds, meaning Mechanism J's heterogeneity does not produce the kind of per-node
  variation an adaptive mechanism would need to exploit.

**Conclusion, closed at this scale for this design space:** the fixed Markov Floor remains the
correct production default. It is free, it has never been consistently beaten across roughly 130
training runs and eight gate variants spanning two independently-built datasets, and the specific
reasons every attempted alternative failed are now mechanically understood rather than merely
observed. Three concrete, falsifiable conditions were identified that could reopen the question —
more label volume (spec-scale, not yet run), a gate that scores per-depth-token rather than
mean-pooling first, or a dataset where predictive signal is deliberately placed at different
depths for different nodes — but absent one of those, further gate-design iteration would be
re-running an experiment whose mechanism is now understood.

---

## 4. Layer 3 — the global head, tested five ways, and why it is being scrapped

**The question:** can the "global head" always envisioned in the original architecture — a
non-local attention mechanism connecting suppliers with no graph path between them — actually
discover hidden dependencies, and can that discovery be turned into better predictions?

### 4.1 V1 — the mechanism works as a discovery tool, but has nothing to discover that matters

Transformer 2 (same-type global attention over Supplier embeddings, no adjacency mask, top-64
cosine-similarity candidate pool) was built and validated against a planted scenario: four
suppliers, four countries, sharing one unmodeled upstream polymer plant (`H_POLYMER`). It found
them decisively — mean percentile rank 0.7255 against a chance level of 0.500, discovery rate
5.5–7.5x chance, consistent across every seed.

But four different ways of fusing that discovery into the impact prediction — plain additive
fusion, confidence-weighted fusion, a per-node trust gate, and cross-attention fusion — all came
back with AUC deltas that flipped sign across seeds. The cause was root-caused directly in the
generator: `H_POLYMER` was a real, causally-grounded correlation (all four members genuinely,
simultaneously exposed to the same disruption windows) but **causally redundant** for prediction
— a member's own observable history already fully captured its risk, so nothing about a
partner's state added anything a model could learn to use. No fusion design, however
sophisticated, could win on a target with no incremental predictive information in it.
Confidence-Aware Fusion was adopted as the standing default anyway — not for an accuracy win
(nothing produced one), but because it added zero parameters and preserved discovery quality
best of the three, the least-risk choice while genuinely coupled data didn't exist.

### 4.2 V2 — the missing precondition is built, and discovery breaks instead

V2 was built specifically to close the gap V1 identified: Mechanism B now generates three
group types by the identical structural process — Type A (`α > 0` via Mechanism D, genuinely
causally coupled), Type B (`α = 0`, correlated but redundant — reproducing V1's exact
`H_POLYMER` behavior on purpose), and Type C (a pure decoy). For the first time, a real,
causally-informative hidden-dependency scenario existed to test against.

Retrieval itself failed instead. Percentile rank sat at 0.479–0.514 against a 0.500 chance
baseline and discovery rate at 0.77x–1.11x chance — indistinguishable from the inert Type C decoy,
at both scales tested, across all four fusion designs. V1 had discovery without coupling; V2 had
coupling without discovery. Investigation found the embeddings being compared were never trained
to encode co-membership at all — they were shaped entirely by the main prediction losses, with
zero pressure to preserve the structure retrieval needed.

### 4.3 The retrieval redesign — a ceiling test, an honest attempt at the real thing, and a definitive answer

A follow-up session tested the diagnosis directly, in two stages, with a hard stop between them:

- **Stage 1 (privileged supervision — the ceiling test).** An auxiliary contrastive loss, trained
  directly against the true group labels, was added to see whether the representation could
  encode co-membership at all if explicitly taught to. It worked spectacularly on the groups it
  was shown — Recall@64 rose to 10.9x chance, MRR to 24.3x — and failed completely to generalize:
  on held-out groups from the identical generative process, retrieval stayed at chance, 0 of 20
  cells clearing their own measured reproduction floor. It had memorized which 41 suppliers were
  labeled related, not learned what relatedness looks like. The cost of that memorization was the
  single largest effect measured anywhere in this project: impact AUC fell by 0.145–0.158,
  sign-consistent on 5 of 5 seeds, roughly thirty times the arm's own noise floor.
- **Stage 2 (the honest, observable-only version).** Retrieval built from the same behavioral
  correlation statistic the generator's own validation check uses came back a clean null — real
  Type B groups elevated only 1.19x–1.41x above chance with inconsistent sign, and the Type C
  decoy scored as well or better in most cells. A newly-discovered confound explained part of it:
  the base dataset's own unrelated shared-factor pools (port, trucking, customs) produce 363
  coincidental co-degrading pairs for every one real hidden-parent pair.
- **The representation probe — the finding that closes the question.** A separate check fit the
  same discrimination objective at every depth of the encoder, including the raw input features,
  against a permutation-matched null. At no depth — none — does true co-membership separate from
  a random regrouping by more than the null's own noise. There is no cue anywhere in the model,
  including before any processing happens, that distinguishes two suppliers who secretly share a
  hidden parent from two who do not. Only their individual identities do, and identities can be
  memorized but never generalized from.

### 4.4 The reframe — Behavioural Hypothesis Generation, and its own definitive result

One further redesign was attempted before concluding: instead of committing to a single hidden
cause, generate a ranked, calibrated set of plausible explanations for observed correlated
behavior, with an explicit "unknown" option, and leave the interpretation to a human reviewer.
This tested cleanly:

- For **regional/logistics causes** (shared port, trucking route, customs zone — all close to
  observable via country and shipping mode), the module ranked and calibrated well: AUC 0.817,
  predicted and empirical rates agreeing to within 0.011 across ten bins of over 13,000 instances.
- For **shared upstream dependency** — the hypothesis the whole line of work existed to answer —
  the module never emitted a confidence above 0.13%, on any variant, including on pairs that
  genuinely shared a hidden parent, and on the highest-powered measurement available it ranked at
  chance with the inert decoy scoring *above* it. This is the module working correctly, not
  failing: with no evidence, it correctly declined to manufacture a hypothesis, and its
  reliability curve confirms the near-zero confidence is honest rather than merely small.
- One narrow, real, but bounded finding emerged: among pairs that already show strong observable
  co-degradation, the *redundant* group type (the one whose shared factor leaks into each
  member's own history, matching V1's `H_POLYMER`) is mildly separable (AUC 0.64–0.74), but that
  effect vanishes entirely across the full population of pairs, and the *causally-coupled* type
  never separates from chance in any setting tested.
- **The most precise diagnosis of the whole project came out of this session.** A same-strength
  comparison (coupling coefficient 0.35 in both cases) showed that a *direct, pairwise* hidden
  relationship (two named suppliers exchanging stress directly) is recoverable from behavior alone
  at AUC 0.78 — clearly findable — while the *group-averaged* hidden-parent relationship this
  project has tested throughout is not, at any pool width, on either variant. The obstacle was
  never signal strength. It is that spreading a hidden parent's influence across an average of
  four or five members dilutes any individual pair's share of it below the point where the pair
  can be individually attributed — a structural property of how the dataset generates hidden
  groups, fully explaining every null this entire layer produced, from V1's `H_POLYMER` onward.

---

## 5. Why Layer 3 is being scrapped, stated in full

Hidden-relationship discovery was investigated through five substantially different approaches
across two independently-built datasets: similarity-based retrieval (V1's Transformer 2 and its
three fusion redesigns), privileged contrastive supervision (a ceiling test with the true labels
handed directly to the loss), observable behavioral retrieval (co-failure correlation built from
the same statistic the generator's own validation suite uses), representation probing (a direct
check of every layer of the encoder, including raw input features, against a permutation-matched
null), and a reframed hypothesis-generation formulation that traded discovery for calibrated,
ranked plausibility.

Across all five, the result was the same. No transferable signal sufficient for reliable
hidden-group recovery was found anywhere the search was capable of reaching — not in an emergent
property of task-trained embeddings, not under a model explicitly and directly supervised with
the correct answer (which could only memorize, never generalize), and not at any single depth of
the network, including before any learning happens at all. The one case where a same-strength
hidden relationship *was* recoverable — a direct pairwise dependency, tested as a positive control
— proved the pipeline itself is sound and pinpointed the actual cause: this benchmark's
hidden-parent mechanism spreads its effect across a group average, and that averaging destroys
the pairwise attribution any retrieval method or classifier needs to work with, regardless of how
the method is designed.

**The limitation is therefore a property of the information available in the observable data —
specifically, of how this benchmark's generator currently constructs hidden-group influence —
rather than a property of any retrieval architecture tested.** Five different architectural
answers to "how should retrieval work" all ran into the identical wall, which is the signature of
a data-generation constraint, not a modeling one. A model cannot recover an attribution the
generator's own mechanism never assigns to an individual pair in the first place.

One genuinely useful capability survives the loss of the discovery ambition: the hypothesis-
generation module correctly and honestly attributes correlated behavior to observable causes
(regional and logistics factors) with validated calibration, and correctly declines to claim a
hidden cause when it has no evidence for one — a real, if narrower, decision-support tool rather
than the discovery engine originally envisioned as the "global head."

A concrete, specific fix exists for the underlying dataset limitation — make hidden-parent
coupling pairwise rather than group-mean-mediated, at unchanged strength — and the positive
control's 0.78 AUC predicts it would become recoverable if built. That is a generator change, not
a model change, and it is out of scope for the current architecture line.

**Future work therefore shifts from hidden-relationship discovery toward decision-support
capabilities built on the validated SHARE and Markov Blanket architecture** — counterfactual
reasoning (what would happen if a supplier were removed or replaced, checkable against this
project's own re-simulatable generator), uncertainty estimation, and calibrated, faithful
explanation — rather than continuing to iterate on a discovery objective five independent
experiments have now shown is not answerable with the information this benchmark currently makes
available.

---

## 6. The architecture today

| Layer | Original concept | Final architecture | Status |
|---|---|---|---|
| 1 — Encoder | HGT, full per-relation independence | **SHARE** — RGCN transform + one shared, relation-blind attention scorer | Production, validated across two datasets and twelve mechanism variants |
| 2 — Depth readout | A short, fixed final-layer head | **Markov Blanket** — zero-parameter, per-task fixed depth (delay→h¹, shortage→h³, impact→h⁴) | Production; every adaptive alternative tested (eight variants, ~130 runs, two datasets) failed to beat it, with the failure mechanically explained |
| 3 — Global head | Hidden-dependency discovery + fusion | **Scrapped as a discovery mechanism.** Retained only as a calibrated, honest hypothesis-generation tool for observable (non-hidden) correlated-risk explanations | Closed at this scale for this design space, pending a generator-level fix not currently in scope |

**HADES's production architecture is SHARE + Markov Blanket.** The global head that was part of
the original concept from the beginning is not being abandoned casually — it was tested, in five
substantially different forms, across two independently-built benchmarks, and every path led to
the same structural wall. What comes next builds on the two components that earned their place,
rather than continuing to search for a signal five experiments agree is not there to find.
