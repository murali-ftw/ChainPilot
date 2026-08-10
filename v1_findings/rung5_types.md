# Rung 5 Gate Designs — A Technical Catalog

This is a side-by-side reference for every Rung-5-family depth-gate design built in Step 6:
the original Rung 5 (built in Round 4 of `reports/layer_2.md`) plus its five isolated variants
A–E (built in Round 5). It is a companion to `reports/layer_2.md`'s chronological narrative,
not a replacement for it — read `layer_2.md` for the full experimental context, the shared
baselines each design was measured against, and the per-round governance record. This document
exists so the mechanism and findings for any one design can be looked up without wading through
the round-by-round story.

**Shared context, true of every design below unless stated otherwise:**

- All are genuine PER-NODE gates: given a node's five depth-token representations
  `[h^0_i, h^1_i, h^2_i, h^3_i, h^4_i]` (SHARE's encoder, `h^0..h^4`, exposed by
  `ml/models/rgcn_attn_depthgate_encoder.py`), each design produces THAT NODE'S OWN 5-way
  softmax over depths — never one distribution shared across a whole task's node population
  (that was Step 6 Round 2's task-level gate, a different, earlier design).
- All are prior-initialized: at construction, every task's gate is biased so its output
  distribution is a sharp near-one-hot at that task's Markov fixed depth
  (`ml/models/rgcn_attn_markov_encoder.py::MARKOV_READOUT_DEPTH`: delay→h¹, shortage→h³,
  impact→h⁴, `reports/layer_2.md` Round 3) — the layer that reads node content is
  zero-initialized, so `logits = position_bias` for every node at construction, identical and
  content-independent. `softmax(position_bias)` at `sharpness=8.0` puts ~99.9% of its mass on
  the target depth (~0.03% on each other depth). Verified live for every design: 100% argmax
  match rate, zero NaNs, before any training.
- All blend the ORIGINAL (unprojected/untransformed) depth tokens for the final embedding
  `z_i = Σ_d alpha_i[d] · h^d_i` — whatever machinery a design uses to DECIDE `alpha_i` never
  also transforms the values being blended. This keeps every design comparable: they differ
  ONLY in how `alpha` is computed.
- All were trained at SHARE's matched-d config (`hidden=128, num_bases=10, num_layers=4,
  epochs=100`), 5 seeds each (except Variant E, post-hoc, no training), against the same
  `markov` (fixed-depth) and `rung5`-original shared baselines, retrained fresh in the same
  process as whichever variant(s) were being tested.
- "Match rate" = share of nodes whose post-training argmax depth still equals the Markov prior
  depth for that task. "Degree" = total edges touching a node across all relations in its
  snapshot (dst-side sum under `ToUndirected()`), a coarse hub/leaf proxy.

---

## Type 0 — Original Rung 5 (the baseline every variant modifies)

**File:** `ml/models/rgcn_attn_rung5_encoder.py`. **Architecture id:** `rgcn_attn_rung5`.
**Built:** Round 4.

**Mechanism.** No attention between depth tokens — the cheapest rung on
*"Markov Scoping and Transformer 1.md"* Part 6's ladder (§6.4's own recommendation). A small
MLP (`hidden → 32 → 5`) reads a **mean-pooled** representation of the five depth tokens:

```
pooled_i = mean(h^0_i, h^1_i, h^2_i, h^3_i, h^4_i)     # [hidden]
logits_i = MLP(pooled_i) + position_bias                # [5], MLP: hidden->32->5, ReLU
alpha_i  = softmax(logits_i)
z_i      = sum_d alpha_i[d] * h^d_i
```

Mean-pooling was chosen over concatenation specifically to keep the design cheap — concat would
5x the first layer's parameter count and push this rung past Rung 4's cost, defeating the point
of being the "cheap" rung. The documented trade-off: mean-pooling discards each depth's
identity before the gate ever sees it (it infers "which depths are represented in this pooled
summary," not "what does h¹ specifically look like versus h³").

**Parameters:** 4,298/task × 3 tasks = **12,894** total (full model: 765,105, at hidden=128).
~15x cheaper than Rung 4 (the attention-based sibling design, 198,546 gate params) at the same
hidden width.

**AUC** (5-seed mean): delay 0.8150–0.8170 (two separate rounds' retrains), shortage
0.7904–0.7976 (highest seed-to-seed spread of any arm tested, std up to 0.0146), impact
0.9443–0.9450. Never clears 5-seed sign-consistency against markov on any task (both rounds).

**Stability — the headline finding.** Wildly seed-dependent and bimodal: delay match rate
0.124–0.277 (mean across two rounds' retrains), with per-seed values ranging from 0.05 to 0.61
— some seeds barely move from the prior, most drift almost entirely away from it. Shortage is
even more starkly bimodal: per-seed match rates of `[0.15, 0.80, 0.99, 1.00, 1.00]` in one
retrain — three seeds essentially perfect, two nearly total defections. Impact is the most
stable of the three tasks (0.913–0.917 mean) but still ranges 0.73–1.00 across seeds.

**Degree correlation.** Impact: CONSISTENT negative (−6.61 to −6.61 across both rounds'
retrains) — deviating nodes have lower degree than matching ones. Delay/shortage: FLIPS (no
reliable pattern) in most retrains.

**Verdict.** The mechanism works exactly as designed (correct near-one-hot init, genuine
per-node adaptivity, a real degree signal on impact) but is unusable as-is for anything
requiring predictable behavior — the same architecture and task can land in completely
different stability regimes depending only on random seed, with no way to predict which in
advance and no AUC signal distinguishing the two regimes.

---

## Type A — Fixed Prior + Bounded Residual Gate

**File:** `ml/models/rgcn_attn_rung5_variant_a.py`. **Architecture id:** `rgcn_attn_rung5_a`.
**Built:** Round 5. **The one change from Type 0:** position bias becomes a FIXED buffer
(`register_buffer`, never trained — Type 0's was a learned `nn.Parameter` that merely *started*
near one-hot); the MLP's raw output is passed through `tanh` and scaled by a small scalar
`lambda_bound` before being added:

```
logits_i = position_bias_FIXED + lambda_bound * tanh(MLP(pooled_i))
```

Since `tanh ∈ [-1, 1]`, the residual can never exceed `±lambda_bound` in any logit component —
a **hard, structural ceiling** on drift, not a soft discouragement (contrast with Round 2's KL
penalty, which only made drift costly, never impossible). Tested at `lambda_bound ∈ {0.1, 0.3,
0.5}`, 5 seeds each (15 runs total).

**Parameters:** identical structure to Type 0 (765,090 full model — the 15-parameter
difference from Type 0's 765,105 is the position bias moving from `nn.Parameter` to a
non-trainable buffer).

**AUC** (5-seed mean, all three lambdas): delay 0.8176–0.8188 (best point estimate of any Rung
5-family design), shortage 0.7980–0.7994, impact 0.9418–0.9447. Every comparison against
markov/rung5-original still flips sign across seeds — no significant AUC change — but every
point estimate is competitive with or better than the unfixed original.

**Stability — the decisive result.** **100% match rate. Every seed. Every lambda. Every task.
Zero deviating nodes, anywhere, in all 15 runs.** The bounded residual didn't just reduce
drift, it eliminated it entirely — the model never used any of its permitted `±0.1` to `±0.5`
logit headroom to move a single node's argmax away from the prior.

**Degree correlation.** Not computable — no deviating nodes exist to correlate with anything.

**Lambda sensitivity.** None on stability (perfect at all three); minor on AUC (impact dips
slightly at λ=0.5: 0.9418 vs 0.9447 at 0.1/0.3, within noise).

**Verdict.** **The clearest, most decisive result in the whole Rung-5-family investigation.**
Solves the stability problem completely, at essentially zero AUC cost, and with a nominal (if
not significant) point-estimate improvement on delay. The strongest candidate for a
follow-up combination round.

---

## Type B — Structural Features in the Gate Input

**File:** `ml/models/rgcn_attn_rung5_variant_b.py`. **Architecture id:** `rgcn_attn_rung5_b`.
**Built:** Round 5. **The one change from Type 0:** the gate's MLP input widens from `hidden`
(pooled embedding alone) to `hidden + 1 + num_node_types + num_task_relations`, concatenating
per node:

1. Normalized total degree (divided by that entity type's mean degree in the same snapshot).
2. One-hot node type over all node types in the graph metadata — a CONSTANT vector within any
   one task's gate (every node a task's gate ever sees shares that task's fixed entity type),
   implemented as specified regardless and flagged as degenerate-for-this-design rather than
   silently omitted.
3. A per-relation edge-count histogram — one count per relation where the node's entity type
   is the destination (e.g. "how many `SUPPLIES` edges, how many `SHIPS_FROM` edges"), not
   just a pooled total.

Position bias stays LEARNED (unlike Type A) and the learning rate is unchanged — this variant
isolates ONLY the effect of giving the gate explicit structural signal.

**Parameters:** 766,289 full model (+1,184 over Type 0, from the widened first MLP layer per
task).

**AUC** (5-seed mean): delay 0.8144 (the lowest point estimate of any variant), shortage
0.7987, impact 0.9382 (also the lowest). Every comparison flips sign against both baselines.

**Stability — mixed, task-dependent.** Shortage improved substantially: match rate 0.981
(range 0.92–1.00) vs. Type 0's 0.712 (range 0.15–1.00) — a real, much tighter result. Delay did
NOT improve: match rate 0.516, still ranging 0.15 to 1.00 across seeds, statistically
indistinguishable in spread from the unfixed original. Impact: 0.902, similar to Type 0.

**Degree correlation.** Shortage: CONSISTENT negative (−2.06). Impact: FLIPS (one seed showed
an enormous, implausible positive outlier — +230 — suggesting the correlation isn't reliable
here, likely driven by a small number of very-high-degree deviating nodes in that seed).

**Verdict.** A partial, task-specific fix — meaningfully stabilizes shortage, does nothing for
delay's instability. Worth carrying into a combination round narrowly (paired with something
that addresses delay), not worth deploying alone.

---

## Type C — Freeze / Differential Learning Rate

**File:** none (architecturally identical to Type 0 — a training-loop-only change in
`ml/train.py::train_model`). **Architecture id:** `rgcn_attn_rung5_c` (a distinct id despite
reusing Type 0's model class, so its runs are logged separately). **Built:** Round 5. **The
one change from Type 0:** the gate's parameters are frozen (`requires_grad=False`) for the
first 15 of 100 epochs — encoder and heads train normally on the Markov-initialized readout,
behaving exactly like the `markov` arm would during this phase — then unfrozen at `0.1 ×` the
main learning rate for the remaining 85 epochs. Implemented as a single optimizer with two
param groups from the start (frozen params simply produce no gradient, so `AdamW.step()`
skips them for free — no optimizer rebuild needed at the unfreeze boundary). Verified
directly: training for exactly 15 epochs with `freeze_epochs=15` leaves the gate reading
`[0.0003, 0.9987, 0.0003, 0.0003, 0.0003]`-style near-one-hot weights, unchanged to four
decimal places from initialization — the freeze mechanism works exactly as designed.

**Parameters:** identical to Type 0 (765,105) — no architecture change, only a training
schedule change.

**AUC** (5-seed mean): delay 0.8166, shortage 0.8005 (the **best point estimate of any Rung
5-family arm, including markov and Type A** — the only arm to clear 0.80), impact 0.9408. Every
comparison still flips sign against both baselines.

**Stability — the clear second-best result.** Delay: 1.000 match rate, ALL 5 seeds — fully
solved. Shortage: 0.995 (range 0.98–1.00) — nearly solved. Impact: 0.973 (range 0.92–1.00) —
substantially improved over Type 0's 0.917. Not the complete, zero-deviation elimination Type A
achieved, but a large, consistent improvement across all three tasks simultaneously (Type B, by
contrast, only helped one task).

**Degree correlation.** Shortage: CONSISTENT negative (−1.11). Impact: CONSISTENT negative
(−7.70, close to Type 0's own −6.61) — the clearest replication of the degree-correlation
pattern of any variant.

**Verdict.** The best all-around single fix that still permits some genuine per-node
adaptivity (unlike Type A, which suppressed deviation to literally zero). Combined with Type
A's hard bound, this is the round's own top recommendation for a follow-up.

---

## Type D — Depth-Identity-Preserving Projection

**File:** `ml/models/rgcn_attn_rung5_variant_d.py`. **Architecture id:** `rgcn_attn_rung5_d`.
**Built:** Round 5. **The one change from Type 0:** mean-pooling is replaced by a SHARED small
linear projection (`hidden → 8`) applied to each of `h^0..h^4` INDIVIDUALLY (same weights
reused at every depth position — a fair, identical transform per depth, not five separate
ones), then concatenated into a 40-dim vector before the rest of the gate:

```
proj_i = [ W(h^0_i), W(h^1_i), W(h^2_i), W(h^3_i), W(h^4_i) ]   # [40], W: hidden->8
logits_i = MLP(proj_i) + position_bias                           # MLP: 40->32->5
```

Position bias stays LEARNED and the learning rate is unchanged — this variant isolates ONLY
the effect of preserving each depth's individual identity through the gate's input, addressing
the exact limitation Type 0's own docstring names (mean-pooling "discards each depth's
identity before the gate ever sees it").

**Parameters:** 759,753 full model — actually **smaller** than Type 0 (765,105), since the
40-dim projected input to the MLP's first layer is cheaper than the original 128-dim pooled
input despite the extra shared projection layer.

**AUC** (5-seed mean): delay 0.8160, shortage 0.7962, impact 0.9430. Every comparison flips
sign against both baselines.

**Stability — did not help, and arguably regressed.** Delay: 0.113 match rate (range
0.06–0.17) — statistically indistinguishable from Type 0's own 0.124/0.277, still fully
bimodal and unstable. Shortage: 0.545 (range 0.12–0.78) — actually **worse** than Type 0's
0.712, and still bimodal. Impact: 0.960 (range 0.80–1.00) — a modest improvement over Type 0's
0.917.

**Degree correlation.** Shortage: CONSISTENT negative (−0.23, small magnitude). Impact:
CONSISTENT negative (−6.84, close to Type 0's −6.61).

**Verdict.** Preserving depth identity through a per-depth projection, on its own, does not
address the instability — and may make it marginally worse on shortage. Not recommended for
further pursuit on this evidence; the degree correlation pattern still replicates, but that
was already established by Type 0 itself.

---

## Type E — Post-Hoc Weight-Averaging (not a training variant)

**No new architecture file** — reuses Type 0's model class (`Rung5HADESModel`,
`rgcn_attn_rung5`) with averaged weights loaded in. **Built:** Round 5, post-hoc, no training
of its own.

**Mechanism.** Take the FULL parameter set (encoder + gates + heads — not the gate submodule
alone; see interpretation note below) of the 5 already-trained Type-0 seeds from the same
round, average every matching tensor elementwise, load the result into a fresh model instance,
evaluate directly on the test set — no further training.

**Interpretation note.** "Average the gate's parameters (not predictions)" was read as
full-model weight-averaging (the standard SWA/model-soup technique), not gate-submodule-only
averaging — evaluating a gate in isolation requires an encoder to sit under it, and the 5
seeds' encoders differ from each other, so there is no principled way to average only the gate
weights and still get a coherent, evaluable model. "Not predictions" is read as contrasting
this against prediction-space ensembling (averaging output probabilities), which was
deliberately NOT what was done here.

**Parameters:** 765,105 (same shape as Type 0, since it's literally an averaged Type-0 model).

**AUC — catastrophic.** delay 0.7285, shortage 0.3223, impact 0.4548. Every comparison against
every baseline (markov, rung5-original, and specifically the BEST individual rung5 seed) is
**significant in the negative direction** on all three tasks (e.g. vs. best individual seed:
delay Δ=−0.0921 CI[−0.112,−0.072]; shortage Δ=−0.4791 CI[−0.508,−0.451]; impact Δ=−0.4919
CI[−0.537,−0.447]). Shortage and impact land below what a random classifier would be expected
to score.

**Stability.** 100% match rate on all three tasks — but this is a meaningless "stability" here;
it reflects the gate weights averaging to something near the shared initialization region
(all 5 seeds' gates likely converged to similar territory, unlike their encoders/heads) rather
than any property worth calling "stable" given the catastrophic AUC underneath it.

**Verdict.** **Averaging hurt, badly and unambiguously — it did not help and did not make no
difference.** This is the well-documented failure mode of naive parameter-space averaging
across independently-initialized, independently-trained neural networks that lack linear mode
connectivity (Frankle et al. 2020; Wortsman et al. 2022's own "model soup" paper explicitly
scopes its success claims to fine-tunes of a SHARED pretrained checkpoint, not
independently-trained-from-scratch runs like these five seeds). Not a viable technique for
this architecture family as tested. Ensembling PREDICTIONS (averaging output probabilities
rather than weights) was not tested here and remains a separate, untried question.

---

## Side-by-side comparison

| Design | Params (full model) | delay AUC | shortage AUC | impact AUC | delay match | shortage match | impact match | Degree-corr. (CONSISTENT tasks) |
|---|---:|---|---|---|---|---|---|---|
| **markov** (reference, no gate) | 752,211 | 0.8163–0.8172 | 0.7970–0.7974 | 0.9399–0.9424 | — (fixed, no gate) | — | — | — |
| **Type 0 (original)** | 765,105 | 0.8150–0.8170 | 0.7904–0.7976 | 0.9443–0.9450 | 0.124–0.277 | 0.712–0.785 | 0.913–0.917 | impact |
| **Type A (bounded)** | 765,090 | 0.8176–0.8188 | 0.7980–0.7994 | 0.9418–0.9447 | **1.000** | **1.000** | **1.000** | n/a (no deviation) |
| **Type B (structural feat.)** | 766,289 | 0.8144 | 0.7987 | 0.9382 | 0.516 | 0.981 | 0.902 | shortage |
| **Type C (freeze/low-LR)** | 765,105 | 0.8166 | **0.8005** | 0.9408 | 1.000 | 0.995 | 0.973 | shortage, impact |
| **Type D (per-depth proj.)** | 759,753 | 0.8160 | 0.7962 | 0.9430 | 0.113 | 0.545 | 0.960 | shortage, impact |
| **Type E (weight-avg)** | 765,105 | 0.7285 ⚠️ | 0.3223 ⚠️⚠️ | 0.4548 ⚠️⚠️ | 1.000* | 1.000* | 1.000* | n/a (*meaningless given AUC*) |

No design's AUC beats markov or Type 0 with 5-seed statistical significance, on any task —
every pairwise comparison in every round flips sign across seeds, except Type E, which loses
significantly to everything.

---

## Synthesis — which design to use, and what to test next

**For stability alone: Type A, unambiguously.** Complete elimination of the seed-dependent
drift problem, at every lambda tested, at essentially zero AUC cost.

**For a small amount of genuine, apparently-meaningful per-node adaptivity while still being
far more stable than the original: Type C.** It has the best shortage AUC point estimate of
any Rung-5-family design, the strongest and most consistent degree-correlation replication, and
near-total (though not literally 100%) stability.

**Recommended next step:** train **Type A + Type C combined** (bounded residual gate, AND
frozen-then-low-LR training schedule) against the same markov/rung5-original baselines. They
are mechanistically complementary — one bounds how far the gate can ever move, the other slows
how fast it approaches that bound — and neither showed any accuracy downside alone. A
secondary, narrower follow-up worth considering: Type B's structural features layered onto
either A or C, to see if delay's instability (which B alone never fixed) responds better when
paired with a bound or a freeze schedule.

**Not recommended for further investment:** Type D (no stability benefit over the unfixed
original, and a possible shortage regression) and full-model weight-averaging (Type E,
definitively harmful as tested — if ensembling is worth revisiting, prediction-space averaging
would need to be tried as a separate, different technique, not weight-space averaging again).

**The standing open question across ALL six designs, including markov itself:** none has ever
produced a statistically defensible AUC improvement on shortage or impact specifically — five
independent mechanisms (task-gate, markov, Rung 4, Rung 5, and Rung 5's five variants) have all
found the same negative result on those two tasks. Every design's AUC differences are within
what 5-seed noise produces. Only delay (via markov's free fixed depth, or Type A/C's
near-equally-good point estimates) has ever moved reliably. This is worth treating as a
property of the dataset/label volume at this scale, not a reason to keep testing new gate
mechanisms against it without new evidence (e.g. more labels, or a different task formulation)
suggesting otherwise.

---

*Companion document: `reports/layer_2.md` (full chronological narrative, shared-baseline
methodology, and per-round governance record for all designs cataloged here).*
