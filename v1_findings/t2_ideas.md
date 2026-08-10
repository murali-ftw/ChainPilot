# Transformer 2 Fusion — Proposed Improvement Ideas

**Context:** `reports/step7_transformer2_pilot.md` found that Transformer 2 reliably discovers
the planted hidden-dependency signal (validation passed decisively, consistent across 5 seeds)
but that discovery did not translate into a measurable impact-AUC improvement — every
comparison flipped sign across seeds. These are the proposed fixes for that gap, in ranked
order, with enough explanation to build any of them without re-deriving the reasoning.

**Note:** 8 ideas are catalogued below (all that were proposed and discussed) — ranked by
priority, not by when they were proposed.

**Prerequisite, applies to all of them:** before building any of these, verify whether the
H_POLYMER hidden-dependency scenario in `db/generate_dataset.py` is actually wired to influence
disruption/impact labels via a causal coupling parameter, or whether it exists purely to be
discoverable with no real predictive link to the outcome. If no coupling exists, none of the
ideas below can produce a genuine AUC gain from this specific scenario, however well-built —
see the master prompt in the conversation history (Prerequisite 0) for the exact check.

---

## 1. Per-Node Trust Gate — highest priority

**The problem it targets.** The current fusion uses a single global scalar (`t2_scale`) applied
identically to all 800 suppliers, implicitly assuming every discovered "hidden dependency" match
deserves equal trust. This is very likely wrong — most of a supplier's top-64 similarity matches
are coincidental, not real hidden causes; only a handful of suppliers (4, in the planted test
case) have a genuine hidden connection worth trusting strongly.

**The fix.** Replace the single global scalar with a small, per-supplier learned gate — every
node gets its own trust value instead of sharing one setting with all 800. Implemented the same
way Variant A fixed the earlier depth-selection instability: zero-initialized so it starts
numerically identical to the current baseline (`t2_scale=0` behavior), and only learns to trust
Transformer 2 more for specific suppliers if training gives it a real reason to.

**Why it's ranked first.** Directly targets the diagnosed root cause (signal dilution from a
uniform setting), and reuses a mechanism already proven to work in this exact project (Variant
A's bounded, per-node correction pattern, `reports/rung5_types.md` Type A) — not a new class of
risk.

---

## 2. Confidence-Aware Fusion — second priority, pairs naturally with #1

**The problem it targets.** Learning trust from labels alone means the model has almost no
evidence to work with — only a few hundred labeled disruption events total, and only 4 real
hidden-dependency cases. A learned-from-scratch trust signal (like a plain per-node gate) has
very little to learn from.

**The fix.** Before asking the network to *learn* trust, compute a confidence estimate directly
from the retrieval mechanism itself — cosine similarity of the top-64 candidate pool, attention
entropy, or retrieval consistency. These numbers already exist inside Transformer 2's forward
pass; no new labels or supervision needed. Use this confidence score to scale fusion strength
directly, or — better — use it as the fixed starting point a per-node trust gate (#1) learns
small deviations from.

**Why it's ranked second.** The only idea on this list requiring zero labels — sidesteps the
small-label instability pattern that has affected every purely-learned mechanism in this
project's history. Should be validated first with a cheap diagnostic: measure whether retrieval
confidence actually correlates with how much a supplier's prediction changes when Transformer 2
is included, before building the full mechanism.

---

## 3. Cross-Attention Fusion — third priority

**The problem it targets.** The current fusion just adds Transformer 2's output directly into
the impact embedding (`z_impact + t2_scale * t2_output`) — a blunt combination that can't
selectively pull out only the relevant part of what was discovered.

**The fix.** Replace the additive combination with cross-attention: let the impact
representation actively query Transformer 2's retrieved-pool output, so the impact head can
selectively retrieve only the information relevant to that specific prediction, rather than
receiving the whole discovered signal indiscriminately.

**Why it's ranked third, not higher.** Still a "utilization-stage" fix (changes how discovered
information is used, same category as #1/#2) rather than a bigger redesign, but it adds real
parameters and a more complex mechanism than either simpler fix — worth trying only if #1/#2
together prove insufficient. This project has seen richer attention mechanisms fail to beat
simpler designs before (Rung 4's attention-based gate cost 15x more parameters than Rung 5's
plain MLP for no measurable benefit) — a reason for caution, not a reason to skip it entirely.

---

## 4. Contrastive Alignment — fourth priority, currently blocked

**The problem it targets.** Right now Transformer 2 is only ever supervised indirectly, through
the downstream impact-prediction loss. It never gets a direct signal about which suppliers
*should* look similar because they share a real hidden dependency.

**The fix.** Add a contrastive loss term that explicitly pulls the embeddings of genuinely
hidden-dependent suppliers closer together, and pushes unrelated suppliers' embeddings apart —
improving the encoder's representation quality directly, not just the fusion step on top of it.

**Why it's ranked fourth, and currently blocked.** Requires labeled positive/negative
hidden-dependency pairs to train against. Right now there are only 4 confirmed real pairs in the
entire dataset (the H_POLYMER scenario) — training a contrastive objective off 4 positive
examples carries a real, serious risk of just memorizing those specific 4 suppliers rather than
learning anything general. Worth pursuing only after the labeled hidden-dependency case count is
deliberately grown.

---

## 5. Auxiliary Hidden-Link Prediction — fifth priority, same blocker as #4

**The problem it targets.** Transformer 2 currently has no objective of its own — it's trained
entirely through whatever gradient reaches it from the downstream impact-prediction loss, which
is a weak, indirect training signal for a component whose actual job is finding hidden
connections.

**The fix.** Give Transformer 2 an explicit secondary objective: directly predict whether two
suppliers share a hidden dependency, providing a much more direct learning signal than relying
solely on downstream impact labels.

**Why it's ranked fifth, and currently blocked.** Same fundamental limitation as #4 — needs
sufficient hidden-dependency supervision (real labels or reliable pseudo-labels) to be trained
safely. With only 4 known real cases, this is a second-stage improvement, not an immediate next
experiment.

---

## 6. Memory-Augmented Transformer — sixth priority

**The problem it targets.** Transformer 2 currently rediscovers the same hidden dependencies
from scratch on every training iteration, which is wasteful and can produce inconsistent
results run to run.

**The fix.** Maintain an external memory of previously discovered hidden dependencies, so the
model can reuse historical discoveries instead of repeatedly rediscovering identical
relationships — improving consistency and computational efficiency.

**Why it's ranked low.** This addresses training efficiency and consistency, not the diagnosed
accuracy gap — it doesn't explain or fix why the discovered signal fails to move AUC, it just
makes rediscovering that signal cheaper. Worth revisiting later, not now.

---

## 7. Retrieval-Augmented GNN — seventh priority

**The problem it targets.** The idea, borrowed from Retrieval-Augmented Generation in large
language models, is to treat Transformer 2 as a retrieval engine that selects the most relevant
hidden suppliers before prediction, so only retrieved nodes participate in reasoning.

**Why it's ranked low.** The existing top-k=64 cosine-similarity candidate pool already *is*
this retrieval step. Unless retrieval itself becomes adaptive or iterative (multiple rounds of
retrieval-then-reasoning), this idea largely re-describes functionality that's already built,
rather than adding new capability.

---

## 8. Edge Proposal Network — lowest priority, last resort

**The problem it targets.** All the ideas above operate at the embedding/fusion level, after the
graph structure is fixed. This idea instead has Transformer 2 predict entirely new, probabilistic
edges representing hidden supplier relationships, and feeds those edges back into the graph so
RGCN can perform another round of message passing on an enriched structure.

**Why it's ranked last.** The most architecturally ambitious idea on this list, and the most
disruptive — it means modifying the graph structure feeding SHARE, the one component in this
project's entire history that has been validated and stable throughout. It also assumes
discovered hidden relationships are not just correct (already validated) but *causally useful*
for downstream prediction once injected into message passing — a claim the current evidence
doesn't yet support (discovery works; fusion into the impact head doesn't measurably help yet).
Given this project's repeated lesson that adding capacity/complexity before evidence justifies
it tends to introduce new instability rather than fix existing gaps, this should only be
attempted after every simpler utilization-based fix above has been tried and shown insufficient.

---

## Suggested build order

0. Verify the causal coupling exists in the dataset generator (prerequisite, applies to all 8).
1. Measure whether retrieval confidence correlates with downstream usefulness (cheap diagnostic).
2. Confidence-Aware Fusion (#2).
3. Per-Node Trust Gate (#1), anchored to the confidence score from step 2.
4. Cross-Attention Fusion (#3), only if 2+3 together prove insufficient.
5. Contrastive Alignment (#4) and Auxiliary Hidden-Link Prediction (#5), only once the labeled
   hidden-dependency case count is deliberately grown.
6. Memory-Augmented Transformer (#6), if retrieval consistency becomes a bottleneck later.
7. Edge Proposal Network (#8), only as an absolute last resort.
