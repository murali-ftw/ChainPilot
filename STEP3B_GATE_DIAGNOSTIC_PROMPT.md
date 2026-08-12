# Gate Disagreement Diagnostic — Does Adaptive Depth Have Anything to Find?

> Paste into a fresh session. Opus, high effort — this is genuine diagnostic design, not a rerun.
> Read `reports/layer2_testing.md` in full first, especially §9 (Limitations) which is the reason
> this session exists, and `HADES_v1/reports/layer2.md` Round 4's "per-node gate analysis" section
> for how V1 originally measured match rate and the degree correlation.

## Why this session exists

Four specific questions, none answerable from the existing sweep because gate weights were never
persisted:

1. How often does the gate disagree with Markov?
2. When it disagrees, which nodes are affected?
3. Do those nodes share properties (degree, chain length, resilience, etc.)?
4. Are the representations at different depths actually different enough for adaptive selection to
   matter?

Questions 1–3 test whether the gate is finding a real, structured pattern or just noise. Question 4
tests something more fundamental underneath all of it — whether there's anything for *any* depth
mechanism to find, independent of whether this particular gate finds it. Answer 4 first if you can;
if the depths aren't meaningfully different from each other, 1–3 are diagnosing a gate that was
never going to matter regardless of its design.

## 1. Persist what training already computes but discards

`Rung5HADESModel` and its variants already stash `_last_gate_weights` during the forward pass —
confirm this against the model code before assuming its shape. Modify the eval path (not the
training loop) to capture, for every node scored on the test split:

- the full softmax distribution over depths (h⁰…h⁴ for the depth-gate family), not just the argmax
- the argmax depth chosen
- the Markov/prior depth for that task
- entity id and entity type (confirm what the per-task readout entity actually is from
  `ml/models/heads.py` — don't assume; delay, shortage, and impact may read out at different entity
  types)

Do this for **Variant A specifically** (the production choice) as the primary target, and for base
Rung 5 as a comparison — does bounding the residual change *how often* the gate disagrees, or just
*how far* it's allowed to drift when it does?

Run across the same five dataset seeds and both variants (0 and J) as the original sweep, so this
plugs directly into the existing results rather than starting a new experimental frame.

## 2. Question 1 — disagreement rate

For each (arm, dataset variant, seed, task): fraction of nodes where `argmax(gate distribution) !=
Markov depth for that task`. Report as V1 did — as a match rate (V1's Variant A: 100% match, zero
deviating nodes) — so this is directly comparable to the V1 number, not a new metric invented for
this session. Report mean and spread across the five seeds; a single-seed number is exactly the
mistake this project corrected in `docs/phase0_power_check.md` §1.

**If Variant A's disagreement rate is at or near zero on V2 too**, that alone is close to a full
explanation for the null result in `reports/layer2_testing.md` — a gate that never disagrees with
the fixed prior cannot produce a different prediction from it, regardless of whether disagreeing
*would* have helped. Say so plainly if that's what's found; it would mean the bounded-residual
design is doing exactly what V1 built it to do (stay safe), at the cost of never being tested on
whether deviation pays off.

## 3. Question 2 — which nodes disagree

For every node where the gate's argmax differs from Markov, record its identity and entity type.
Produce a plain table/list, not just a count — this is the input to question 3, and it's also worth
inspecting directly: are disagreements concentrated in a small, specific population, or scattered
roughly uniformly across all scored entities?

## 4. Question 3 — do disagreeing nodes share properties

Join the disagreement set from §3 against:

- **Degree** — same feature Variant B already computes for the gate input; reuse it rather than
  recomputing. This directly retests V1's own finding (deviating nodes are consistently
  lower-degree) on V2's population.
- **Chain length / tier depth** — from Mechanism J's `supplier_upstream` structure, for whichever
  entities trace to a tiered supplier. This is new to V2; V1 never had it to test.
- **Resilience — two versions, kept explicitly separate, do not conflate them:**
  - **True hidden resilience**, read directly from the generator's internal simulation state (not
    from any emitted CSV — this is privileged, ground-truth-only information no real model could
    ever access). This answers "is there a genuine underlying pattern in the data," a question
    about the dataset, not about what any model could learn.
  - **The observable resilience proxy** built in `docs/phase2_coverage_recheck.md`'s continuous
    estimator (the one that reached AUC 0.599 at λ=1.3). This answers "could a model that had
    access to this signal have found the pattern," which is the more relevant question for whether
    a *better-informed* gate design might work, as distinct from whether the pattern exists at all.

For each property, test whether it predicts disagreement-set membership (a simple logistic
regression or equivalent, with a permutation-null or bootstrap CI, matching this project's standing
statistical discipline — no bare threshold comparisons). Report effect sizes and confidence
intervals per property, per task, per variant, not just "significant/not significant."

## 5. Question 4 — are the depths actually different enough to matter

Independent of anything the gate does: for a sample of scored nodes, extract the raw embeddings at
every depth (h⁰ through h⁴) before the gate ever selects among them. Compute pairwise cosine
similarity between consecutive depths (h⁰-h¹, h¹-h², h²-h³, h³-h⁴) and between the endpoints
(h⁰-h⁴), per node type.

- If similarities are consistently high (representations barely change with depth), there is
  nothing for *any* depth-selection mechanism to exploit at that node type, regardless of how well
  designed the gate is — this would be a structural explanation that applies to Rung 5, Variant A,
  B, C, and any future depth mechanism alike, not a verdict on this specific gate design.
- Check whether this varies with chain length specifically — do longer chains (deeper in Mechanism
  J's tiers) show *more* separation between depths than shorter ones, or is smoothing uniform
  regardless of topology? This is the direct test of whether J's heterogeneity creates the kind of
  representational difference an adaptive mechanism would need to find, separate from whether this
  particular gate found it.
- **This connects to an existing open item worth checking while this instrumentation is already
  built:** `v1_findings/v2.md` item 8 flagged Shipment nodes' anti-smoothing pattern (similarity
  *falling* with depth, opposite of every other node type) as never root-caused in V1. This
  diagnostic is positioned to check whether that pattern still holds on V2 and whether it
  correlates with anything measured here. Report it if it does; don't go looking for it if it adds
  a second session's worth of scope — a one-paragraph check is enough, not a full investigation.

## Deliverable

Append a new section to `reports/layer2_testing.md` (keep the running record in one place, per the
standing convention — this is a direct follow-up to §9's limitations, not a new topic). Structure
the section around the four questions as asked, in order, each with its own answer stated plainly
first, then the supporting measurement. If question 4's answer is "representations barely differ,"
say that up front — it would mean the null result in §7 has a structural explanation that has
nothing to do with gate design, which changes what (if anything) is worth trying next from
`v1_findings/layer2_contingency_plans.md`.
