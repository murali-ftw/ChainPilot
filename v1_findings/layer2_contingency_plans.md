# Layer 2 Contingency Plans — Depth-Selection Ideas Reference

**Purpose:** a reference archive of depth-selection / adaptive-receptive-field ideas for Step 6
(Layer 2), split into two sections: **Improvements** — concrete next steps for Variant A (the
current best result, `reports/info.md` Layer 2 §3), proposed but not yet built or tested — and
**Changes** — the original 24-idea brainstorm (17 "halting mechanisms" + 7 "alternative
theories") merged down to 11 distinct mechanisms after deduplication, kept as a longer-range
reference archive.

**Status at time of writing:** Markov Floor (fixed per-task depth) is the validated production
default. Variant A (bounded residual on top of Markov/Rung 5) is a validated, strictly-more-
capable upgrade — 100% stability, no accuracy cost, and the current best foundation to build on
further.

---

# Improvements — concrete next steps for Variant A

Not yet built or tested. Ranked in priority order; motivated by the fact that Variant A already
solved stability completely but did not move accuracy — it's the only variant safe enough to
keep building on without reopening the instability problem.

## 1. Variant A + Variant C combined

**Core idea:** run both fixes together instead of in isolation. A bounds how far the gate's
correction can drift (a hard structural ceiling); C slows down how fast the gate is allowed to
approach that boundary (freeze early, unfreeze at a reduced learning rate). These address two
different halves of the same underlying problem and neither showed any accuracy cost alone.

**Why try it first.** Cheapest, most directly evidenced next step — both ingredients are
already validated individually; the only open question is whether combining them compounds the
(already-strong) stability result or unlocks any accuracy benefit neither showed alone.

## 2. Variant A + Variant B combined (bounded gate + structural features)

**Core idea:** feed Variant B's explicit structural signal (normalized degree, node-type
one-hot, per-relation edge-count histogram) into Variant A's bounded gate, instead of B's
original unbounded one.

**Why try it.** B already found a real, non-noise signal (deviating nodes are consistently
lower-degree) — but it was tested on the unbounded original Rung 5, where any benefit from that
signal was likely muddied by the same instability that broke every other unbounded variant. With
A's safety cap in place, B's structural evidence has a stable foundation to actually express
itself in the AUC numbers, if it's going to at all.

## 3. Weight-average Variant A's seeds (retry Variant E, restricted to Variant A)

**Core idea:** Variant E (post-hoc weight-averaging across independently-trained seeds)
catastrophically failed when applied to the original, unbounded Rung 5 — because those seeds
converged to disconnected, unrelated solutions with no shared trajectory (a well-documented
precondition failure for this technique). Variant A's seeds don't have that problem: they
converge to essentially the same answer (100% match rate, every seed). That's much closer to the
condition under which weight-averaging is expected to actually work.

**Why try it.** Near-free — post-hoc, no new training needed beyond seeds already trained for
Variant A. Directly tests whether Variant E's failure was about the *unbounded* design
specifically, not the averaging technique itself.

## 4. Widen or anneal Variant A's lambda cap

**Core idea:** every cap size tested so far (λ = 0.1, 0.3, 0.5) held perfect stability with zero
deviating nodes — none came close to breaking. Try a larger cap, or start small and grow it
during training (mirroring Variant C's "let things settle first" logic), to see whether more
room for the gate to move starts capturing real per-node signal instead of the model
consistently declining to use the room it's already given.

**Why try it.** Tests whether the current caps are simply conservative rather than at the actual
edge of what's safe — cheap to sweep, since it's a single scalar.

## 5. More seeds on Variant A specifically, to sharpen the statistics

**Core idea:** Variant A's point estimates already beat Markov on every task at every lambda
tested — the reason none of these differences are "confirmed" is standard 5-seed sign-flip
noise, not a real absence of an edge. Running more seeds (10–20) on just this arm could be
enough to clear the significance bar rather than just lean toward it.

**Why try it.** Variant A trains at roughly the same cost as the original Rung 5 (cheapest
per-node gate design tested) — doubling or tripling seed count here is inexpensive relative to
what it could resolve.

**Suggested first move, if only one is built:** #2 (A + B combined) — B is the only variant that
already surfaced a real, non-noise signal (the degree correlation); pairing it with A's bounded,
stable foundation is the most direct way to find out whether that signal can finally show up in
the AUC numbers instead of being lost to instability.

---

# Changes — original 24-idea catalog (deduplicated)

**Originally 24 distinct-named ideas across two brainstorming passes (17 "halting mechanisms" +
7 "alternative theories"). Many turned out to be the same underlying mechanism described with
different names — this section merges those into one entry each, while preserving every
original name so nothing is lost for future reference.**

## 1. Minimal Explanatory Subgraph

**Merges:** Adaptive Causal Subgraph Discovery, Reachability Theory, Structural Change Halting

**Core idea:** instead of deciding how many layers to run, identify which specific upstream
nodes actually explain a prediction. Depth becomes a byproduct of the discovered subgraph, not a
decision made in advance. Purely structural/topological — can reuse existing k-hop reach and
attention-weight data for a first, cheap validation pass before any new training.

**Grounding:** connects directly to Graph Information Bottleneck (Wu et al., 2020) and GSAT
(Miao et al., 2022) — both already cited as prior art in `Markov Scoping and Transformer 1.md`.

---

## 2. New-Information Convergence

**Merges:** Message-Convergence Halting, Novelty-Based Halting, Information Gain Theory,
Residual Information Halting, Vanilla Embedding-Convergence Halting

**Core idea:** watch whether new, useful information is still arriving at a node with each
additional layer (via message content, embedding change, or a more formal information-theoretic
residual) and stop once it plateaus. Fully label-free — computed directly from values the
encoder already produces, no training needed for the signal itself.

**Caveat on record:** raw convergence signals can be confused by over-smoothing (representations
going quiet because they're being homogenized, not because they've genuinely settled) — flagged
as an open risk needing validation against the project's existing per-node-type over-smoothing
data (Shipment's known anti-smoothing behavior is the natural stress test).

---

## 3. Markov Floor / Causal Horizon

**Merges:** Markov-Constrained Halting, Hybrid Markov + Halting, Causal Horizon Learning,
Causal Halting

**Core idea:** derive a minimum required depth per task directly from the graph's causal/
structural shape (Markov blanket reasoning) and never allow a node to stop short of it. This is
not a new idea to build — it *is* the already-implemented, already-validated production design
(`reports/info.md` Layer 2 §3), and serves as the mandatory safety floor for any of the other
ideas in this document, per the lesson learned from Rung 4/5's unconstrained drift.

---

## 4. Diffusion-Based Convergence

**Merges:** Graph Diffusion Halting, Influence Propagation Theory

**Core idea:** frame message passing as a diffusion process with a known, closed-form decay
rate, and stop once that process has mathematically converged rather than empirically
threshold-watching.

**Grounding:** the closest formal match in the literature to the project's own theory —
APPNP (Klicpera et al., 2019) / PPRGo (Bojchevski et al., 2020), personalized-PageRank
propagation with influence decaying as `α(1−α)^k`. Most mathematically rigorous option on this
list; could in principle avoid empirical threshold tuning entirely.

---

## 5. Adaptive Graph Traversal

**Core idea:** don't decide a stopping depth at all — let the model actively choose which
neighbor to visit next, like a guided search, following only the most promising causal paths
until enough evidence is gathered. Depth emerges from the traversal rather than being decided
upfront. The most structurally different idea on this list (active path-following vs. layer-wise
halting) and the most engineering-heavy to build.

---

## 6. Selective Branch Expansion

**Merges:** Adaptive Graph Expansion, Relation-Aware Halting

**Core idea:** don't treat every neighbor or relation type equally — expand only the branches or
relation types that are actually informative, potentially giving different parts of a node's
neighborhood different effective depths. A refinement layered on top of whichever core signal
(1, 2, or 4) is chosen, not a standalone replacement for it. Connects to the original theory
doc's own relation-scoped decay idea (§1.4).

---

## 7. Attention-Entropy Halting

**From:** Entropy-Based Halting (attention-entropy variant)

**Core idea:** use how spread out or concentrated SHARE's own attention weights already are as
the stopping signal — low entropy means the model already looks confident about which neighbors
matter. Notable because this signal is free: SHARE computes attention weights as part of its
normal forward pass already, so no new instrumentation is needed to extract it.

---

## 8. Prediction-Based Halting

**Merges:** Confidence-Based Halting, Prediction Stability Halting, Uncertainty-Based Halting

**Core idea:** attach a prediction head at each depth and stop once the model's own prediction
becomes confident, stable, or low-uncertainty (via softmax margin, successive-prediction
similarity, or MC-Dropout/ensemble variance). Legitimate and standard (early-exit network
literature), but reintroduces partial dependence on the project's scarce labels — the exact
fragility the label-free alternatives (1, 2, 4, 7) were chosen specifically to avoid. Fallback
tier, not primary.

---

## 9. Global Context Halting

**Core idea:** let a future global-attention module (Transformer 2 / the "Global Transformer")
influence the halting decision — global uncertainty could request additional local reasoning at
a specific node. Legitimate idea, but conditional on Transformer 2 existing first, which it does
not yet in this codebase. Purely speculative until that component is built.

---

## 10. Budgeted Halting

**Core idea:** assign each node a computation budget and penalize excess depth in the training
objective (a "ponder cost"), nudging the model toward using only as much depth as it needs.
Standard technique, directly descended from the original Adaptive Computation Time paper
(Graves, 2016). Best used as an add-on term to whichever core mechanism is chosen, not a
standalone approach.

---

## 11. Architecture-Depth Search (kept separate — different scope, not a duplicate)

**From:** Gradient-Based Halting

**Core idea:** as originally defined ("monitor gradients or loss improvement during
propagation"), this is actually a *training-time, whole-architecture* question — how many
layers should the network have at all — rather than a per-node, per-inference adaptive
mechanism like everything else on this list. Closer in spirit to the project's original L-sweep
than to any of the halting ideas above. Kept as its own entry rather than merged, since
collapsing it into the halting family would misrepresent what it actually answers.

---

## Deduplication map (all 24 original names, for lookup)

| Original name | Merged into |
|---|---|
| Vanilla Embedding-Convergence Halting | §2 New-Information Convergence |
| Markov-Constrained Halting | §3 Markov Floor / Causal Horizon |
| Message-Convergence Halting | §2 New-Information Convergence |
| Novelty-Based Halting | §2 New-Information Convergence |
| Entropy-Based Halting | §7 Attention-Entropy Halting |
| Confidence-Based Halting | §8 Prediction-Based Halting |
| Residual Information Halting | §2 New-Information Convergence |
| Gradient-Based Halting | §11 Architecture-Depth Search |
| Prediction Stability Halting | §8 Prediction-Based Halting |
| Relation-Aware Halting | §6 Selective Branch Expansion |
| Causal Halting | §3 Markov Floor / Causal Horizon |
| Uncertainty-Based Halting | §8 Prediction-Based Halting |
| Hybrid Markov + Halting | §3 Markov Floor / Causal Horizon |
| Global Context Halting | §9 Global Context Halting |
| Budgeted Halting | §10 Budgeted Halting |
| Graph Diffusion Halting | §4 Diffusion-Based Convergence |
| Structural Change Halting | §1 Minimal Explanatory Subgraph |
| Causal Horizon Learning | §3 Markov Floor / Causal Horizon |
| Information Gain Theory | §2 New-Information Convergence |
| Reachability Theory | §1 Minimal Explanatory Subgraph |
| Influence Propagation Theory | §4 Diffusion-Based Convergence |
| Adaptive Graph Expansion | §6 Selective Branch Expansion |
| Adaptive Graph Traversal | §5 Adaptive Graph Traversal |
| Adaptive Causal Subgraph Discovery | §1 Minimal Explanatory Subgraph |

**24 original ideas → 11 distinct entries after deduplication.**
