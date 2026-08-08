# Phase 3 — Transformer 2: Global Attention Discovery + Fusion

**Scope:** the complete Transformer 2 line of work in one place — building the global
same-type attention mechanism (Claim 3) on top of the finalized Layer 1 + Layer 2 stack,
validating it as a hidden-dependency discovery tool, discovering that discovery did not
translate into an accuracy win, and testing three independent fusion redesigns to try to
close that gap. Fuses `reports/step7_transformer2_pilot.md` (the original build + validation)
and `reports/step7b_transformer2_fusion_pilot.md` (the three fusion-fix variants) into a
single chronological record, plus this document's own final findings and recommendation.

**Stack this whole phase builds on:** SHARE (`rgcn_attn`) as the encoder, Variant A (bounded
residual on the Markov floor) as the depth-selection mechanism — the finalized output of Layer
1 and Layer 2. Nothing in this phase changes SHARE's encoder, SHARP, SHARK, or the existing
Markov/Variant A depth logic; Transformer 2 and its fusion variants are purely additive
components layered on top.

---

## Part 1 — Building Transformer 2 (Round 1)

### 1.1 The wiring deviation — stated up front

**`docs/10_AI_ML_Documentation.md` §8.3 specifies wiring Transformer 2's output to the delay
head only** ("Phase 1: wired to the delay head only (supplier-level)"). **This implementation
wires it to the IMPACT head instead.** The reason is direct evidence from this project's own
Step 6 Round 1 reach analysis (`reports/layer_2.md`): the co-parent/hidden-dependency path
needs a hard minimum of **3 hops** from delay's actual prediction target (Shipment) —
`Shipment → Supplier (1) → Component (2) → co-parent Supplier (3)`, confirmed 100% of
supplier-sourced shipments, 0% at hop 1 or hop 2, in a live measurement against the same
dataset this phase uses. Delay reads `h¹` under Markov/Variant A — one hop — so the
co-parent/hidden-dependency signal is **structurally out of reach for delay at any depth this
project has ever tested, baseline included.** The same Round 1 analysis found impact's target
(Supplier, hop 0 by definition) **does** have real co-parent signal available from hop 2
onward, and impact reads `h⁴` under Markov/Variant A — well within reach. Wiring Transformer 2
to delay, per the original doc's untested assumption, would have attached it to the one head
structurally incapable of using it. Not a silent substitution — flagged in the model's own
module docstring (`ml/models/rgcn_attn_variant_a_transformer2.py`) and in every governance
record this phase writes.

### 1.2 Design

**Same-type-only global attention over Supplier embeddings, no adjacency mask.** Operates on
SHARE's own Supplier-type node embeddings, POST-encoder, at whatever depth the impact head
currently reads under Variant A (`z_impact_i` — Variant A's own per-node bounded-residual
blend). No adjacency mask is the entire point — Transformer 2 exists specifically to connect
Supplier pairs with NO graph path between them (the "moralization" case); restricting the pool
to graph-adjacent pairs would defeat the mechanism outright.

**Bounded candidate pool: top-k=64 by cosine similarity, per Supplier.** Matches
`docs/10_AI_ML_Documentation.md` §8.3's own bounding choice exactly — NOT bounded by shared
`component_type`, which that section explicitly flags as wrong (same-`component_type`
suppliers are already 2-hop reachable via the co-parent path, so that bounding would exclude
exactly the cross-type correlation the hidden-dependency scenario and Transformer 2 both exist
to find).

**`is_frontier` override — implemented, but INERT, disclosed explicitly.** The design calls for
always including frontier-flagged suppliers regardless of cosine rank. `suppliers.is_frontier`
is documented as "data-gated" and does not exist in the live schema (re-confirmed live before
writing this phase's code). The override path is implemented correctly but never exercised — a
disclosed limitation carried over from the schema's own documented gap, not a silently-dropped
requirement.

**Fusion into the impact head (original design).** `z_impact_fused = z_impact + t2_scale *
t2_output`, where `t2_scale` is a single learned scalar initialized to exactly 0, matching this
project's established convention (every new component starts identical to the validated
baseline it extends). At `t2_scale=0`, this component's forward pass is numerically identical
to plain Variant A's impact readout.

**Implementation.** `ml/models/transformer2.py` (`Transformer2GlobalAttention`) and
`ml/models/rgcn_attn_variant_a_transformer2.py` (`Transformer2HADESModel`, reusing
`Rung5VariantAGateHead`/`PredictionHead` UNMODIFIED for every task's own depth-selection and
readout). Registered as `rgcn_attn_variant_a_transformer2`. Full model at `hidden=128,
num_bases=10`: **814,627 params** (765,090 for Variant A alone + 49,537 for Transformer 2's
Q/K/V projections + the scale scalar).

**New table `hidden_dependency_links`** (`db/hidden_dependency_links_schema.sql`) persists
discovered pairs for human review — bounded to the top-500 pairs by symmetric attention score
per (seed, most-recent-test-snapshot), plus the 6 H_POLYMER ground-truth pairs unconditionally.
A row is "an investigation lead, never a fact," per its own schema documentation.

### 1.3 Device check (Round 1)

**MPS used for training, verified before choosing it:** the new ops (cosine similarity,
`fill_diagonal_`, `topk`, advanced-index gather, attention/softmax) matched CPU vs MPS forward
+ backward on a synthetic 800×128 batch (`atol=1e-4`, max diff ~1.5e-7 values / ~1.5e-6
gradients), and showed a modest ~1.34x speedup alone (10.1ms CPU vs 7.5ms MPS) — smaller than
the encoder's own ~2.5x since this op set is dispatch-overhead-bound, not compute-bound.
Combined with the encoder, which dominates total cost, MPS remained the better choice overall.

### 1.4 Validation — the planted hidden-dependency scenario

**Ground truth.** Four suppliers, four different countries, different (non-polymer-exclusive)
component types, sharing one unmodeled upstream polymer plant — no row, edge, or column
connects them in the schema: USA-Poly Supply 10, IND-Precision Supply 112, CHI-Precision Supply
00, VIE-Alloy Supply 01.

**Two metrics, both against a defined chance baseline** (not raw attention scores presented as
self-evidently meaningful): (1) percentile rank of each H_POLYMER pair's cosine similarity
among all C(800,2)=319,600 possible Supplier pairs (chance = 0.500); (2) pool-membership
discovery rate — does the pair appear in either endpoint's own top-64 pool (chance measured
directly from the global discovery rate, ~10.4%).

**Results, 5 seeds × 6 test snapshots × 6 pairs = 180 pair-observations:**

| Seed | Mean percentile rank | Discovery rate | Global chance discovery rate |
|---|---|---|---|
| 0 | 0.7702 | 0.694 | 0.1022 |
| 1 | 0.7314 | 0.917 | 0.1066 |
| 2 | 0.7318 | 0.667 | 0.1050 |
| 3 | 0.7099 | 0.694 | 0.1060 |
| 4 | 0.6841 | 0.750 | 0.1043 |
| **Overall** | **0.7255** (chance 0.500) | **0.7444** (chance ~0.105) | — |

**VALIDATION VERDICT: PASS, decisively, consistent across every seed** — every seed's mean
percentile rank exceeds 0.68 and every discovery rate exceeds 0.66, 5.5–7.5x the chance rate.
Not one lucky seed; it replicates.

### 1.5 Impact AUC — original additive fusion, with vs. without Transformer 2

| Arm | delay | shortage | impact |
|---|---|---|---|
| baseline (Variant A alone) | 0.8166 ± 0.0015 | 0.7974 ± 0.0038 | 0.9434 ± 0.0023 |
| Transformer 2 (additive fusion) | 0.8186 ± 0.0021 | 0.7996 ± 0.0034 | 0.9455 ± 0.0019 |

**No statistically reliable AUC change on any task, including impact — the head Transformer 2
was actually wired to.** All three mean deltas are small and positive, but each task's 5-seed
deltas flip sign (impact: 4 of 5 seeds positive, one −0.0040 breaks sign-consistency).

### 1.6 Round 1 verdict

**Discovery and downstream accuracy are separate claims, and this project's own validation
methodology treats them as such.** On the discovery question, Transformer 2 works exactly as
designed — reliably, above chance, across every seed, recovering a signal no graph-based
message-passing mechanism in this project's history (SHARE, SHARP, SHARK, Markov, or any
Rung-5-family gate) could ever reach, because by construction no edge connects the planted
suppliers. That the resulting embedding shift didn't yet move impact's AUC in a way five seeds
agree on was reported as a real, honest limitation — motivating Round 2, below.

---

## Part 2 — Three Isolated Fusion-Improvement Variants (Round 2)

### 2.1 Prerequisite 0 — Causal coupling verdict (read first, colors every result below)

Before building any fusion fix, `db/generate_dataset.py` was read directly to check whether
H_POLYMER carries a genuine incremental predictive signal, the way Task 3's dual-sourcing
scenario explicitly does (`COPARENT_COUPLING = 0.35`).

**Finding: NO cross-supplier causal coupling exists for H_POLYMER.** A supplier's `own_stress`
already fully reflects its own H_POLYMER event exposure. The only cross-supplier bleed-through
term in the entire generator (`stress() = own_stress() + COPARENT_COUPLING * own_stress(partner)`)
draws its `coparents` map exclusively from `component_suppliers` (Task 3's separate dual-sourcing
mechanism) — never from H_POLYMER membership. Every label, including impact, derives from
`stress()`, which for an H_POLYMER member (absent an unrelated `component_suppliers` partner)
reduces to exactly `own_stress()` — fully derivable from that supplier's own observable history
alone.

H_POLYMER **is** a real, causally-grounded correlation (all 4 members are genuinely,
simultaneously exposed to the same disruption event windows), but it is **structurally
REDUNDANT for prediction purposes** — nothing about a partner's current stress leaks into a
member's own stress computation. **Blocking finding: no fusion mechanism, however well-built,
can produce a genuine AUC improvement from discovering H_POLYMER specifically, because there is
nothing incrementally predictive in that discovery.**

### 2.2 Prerequisite 1 — Does retrieval confidence correlate with downstream usefulness?

Using the 5 freshly-retrained original-Transformer-2 models, `confidence_i` (mean top-64 cosine
similarity + attention entropy) was correlated against `|prediction_with_T2 −
prediction_without_T2|` for every Supplier, pooled across test snapshots.

**Result: `pearson_r = +0.7666`, a strong, meaningful positive correlation** — Suppliers with
more confident retrieval are the ones whose predictions Transformer 2 actually moves the most.
This motivated proceeding with Confidence-Aware Fusion as designed, though Prerequisite 0's
finding means this correlation alone doesn't predict an AUC win — confidence tracking
*how much* a prediction shifts is a different claim than that shift being *correct* for the
label.

### 2.3 Device check (Round 2)

Entropy computation, the per-node gate MLP pattern, and `nn.MultiheadAttention` cross-attention
all verified forward+backward, CPU vs MPS, on tensors/weights cloned identically across
devices (an earlier check using independent per-device random draws produced spurious diffs
purely from CPU/MPS RNG streams — caught and corrected). Corrected check: all four outputs
matched to `atol=1e-4` (max diff ~1.2e-6). Cross-attention alone timed **1.85x faster on MPS**
(1.20ms/iter vs 2.22ms/iter). **MPS used for all 25 training runs.**

### 2.4 The three variants

All reuse `Rung5VariantAGateHead`/`PredictionHead` UNMODIFIED — no changes to Variant A. All
verified live: exactly zero contribution to impact logits at construction, matching this
project's safe-start convention.

**Variant 1 — Confidence-Aware Fusion** (`rgcn_attn_t2_confidence`). Reuses
`Transformer2GlobalAttention` unmodified. Multiplies a per-node confidence score (mean top-64
cosine similarity + attention entropy, zero new learned parameters) into the existing learned
global `t2_scale`: `z_impact + t2_scale * confidence_i * t2_output_i`. Params: 814,627 —
identical to the original pilot, confidence adds zero learned parameters.

**Variant 2 — Per-Node Trust Gate** (`rgcn_attn_t2_trustgate`). Replaces the single global
`t2_scale` with a per-node gate: `trust_i = tanh(MLP(z_impact_i))`, MLP `hidden→32→1`,
zero-initialized final layer (reuses Variant A's own bounded-residual pattern). Params: 818,787
(+4,160).

**Variant 3 — Cross-Attention Fusion** (`rgcn_attn_t2_crossattn`). Self-contained top-64
cosine retrieval (duplicated, not shared, to keep all three variants strictly isolated) so the
impact embedding cross-attends over the raw retrieved pool (`nn.MultiheadAttention`, 4 heads)
rather than inheriting Transformer 2's generic weighting. Zero-initialized output `Linear`.
Params: 864,162 (+49,535).

### 2.5 AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| base_a (Variant A alone) | 0.8167 ± 0.0025 | 0.7952 ± 0.0038 | 0.9436 ± 0.0011 |
| base_t2 (original additive fusion) | 0.8144 ± 0.0051 | 0.7991 ± 0.0044 | 0.9425 ± 0.0026 |
| confidence | 0.8176 ± 0.0026 | 0.7995 ± 0.0039 | 0.9432 ± 0.0025 |
| trustgate | 0.8185 ± 0.0008 | 0.7991 ± 0.0037 | 0.9438 ± 0.0019 |
| crossattn | 0.8186 ± 0.0017 | 0.7992 ± 0.0050 | 0.9443 ± 0.0020 |

**Every single comparison — 3 variants × 2 baselines × 3 tasks = 18 cells — FLIPS sign across
the 5 seeds.** No variant reliably beats either shared baseline on AUC, on any task — exactly
what Prerequisite 0 predicted.

### 2.6 Hidden-dependency validation — discovery quality preserved?

| Arm | Mean percentile rank | Discovery rate | Global chance rate | Verdict |
|---|---|---|---|---|
| base_t2 (reference) | 0.7730 | 0.7778 | 0.1037 | **PASS** |
| confidence | 0.7130 | 0.7333 | 0.1061 | **PASS** |
| trustgate | 0.6357 | 0.6000 | 0.1034 | **PASS** |
| crossattn | 0.6510 | 0.5667 | 0.1063 | **PASS** |

All four arms clear the chance bar decisively. Confidence-Aware Fusion stays closest to the
original discovery signal (0.713 vs. 0.773); Trust Gate and Cross-Attention show a somewhat
weaker (still clearly passing) recovery (0.636, 0.651) — plausibly because those two variants'
fusion mechanisms pull the impact embedding's own gradient away from the pure retrieval
objective more than confidence-weighting (which only rescales an existing, unchanged retrieval)
does.

### 2.7 Per-Node Trust Gate — stability analysis

| Seed | Mean trust | Std trust | Share positive/negative/~0 | corr(\|trust\|, degree) |
|---|---|---|---|---|
| 0 | +0.0290 | 0.0748 | 83.6% / 14.7% / 1.6% | −0.1287 |
| 1 | +0.3846 | 0.3834 | 72.7% / 26.7% / 0.6% | −0.1460 |
| 2 | +0.0211 | 0.2035 | 65.1% / 33.4% / 1.5% | −0.0116 |
| 3 | +0.4280 | 0.3257 | 96.3% / 1.8% / 1.9% | −0.2204 |
| 4 | −0.0088 | 0.0801 | 40.0% / 41.5% / 18.6% | +0.1012 |

**Highly unstable across seeds — the same bimodal, seed-dependent signature Step 6's depth-gate
work found repeatedly.** The degree correlation itself flips sign across seeds. The 4 known
H_POLYMER members' mean trust (+0.0166) is *lower* than the rest of the population's (+0.1715)
— the gate does not learn to trust Transformer 2 more for the very suppliers where the
discovered correlation is real, consistent with Prerequisite 0: there is no training pressure
to trust H_POLYMER's signal specifically, because fusing it in offers no incremental predictive
value.

---

## Part 3 — Final Findings and Recommendation

**None of the three fusion redesigns closed the gap, and Prerequisite 0 explains why this was
the expected outcome, not a surprising one.** The discovery mechanism is robust across every
fusion redesign tried (all 4 arms pass hidden-dependency validation decisively), but
H_POLYMER — the only planted hidden-dependency scenario this dataset contains — carries no
incremental causal signal beyond what each supplier's own features already capture. All 18 AUC
comparisons across the three variants and two baselines flip sign across seeds —
indistinguishable from noise. This is not a failure of Transformer 2's architecture: the
discovery mechanism itself is repeatedly, robustly confirmed correct and stable across five
different fusion designs now (the original plus these three). It is that this specific
synthetic dataset's only planted hidden-dependency scenario was never going to reward any
fusion mechanism with an AUC win, because it was built as a discoverable structural pattern,
not an incrementally predictive one.

### Why Confidence-Aware Fusion is the chosen mechanism

With no AUC winner among the three, the choice comes down to cost, stability, and how much of
the validated discovery signal each design preserves — and Confidence-Aware Fusion wins on all
three:

- **Zero added parameters.** It reuses a signal Transformer 2 already computes internally
  (top-64 cosine similarity, attention entropy) — no new learned weights, unlike Trust Gate
  (+4,160 params) or Cross-Attention (+49,535 params, the most expensive of the three with
  nothing to show for the cost).
- **Best-preserved discovery quality of the three.** 0.713 mean percentile rank vs. the
  original's 0.773 — closest to the validated baseline, because it only rescales an existing,
  unchanged retrieval rather than pulling the impact embedding's gradient away from the
  retrieval objective (which is exactly what the weaker-performing Trust Gate and
  Cross-Attention variants do).
- **Most stable of the three.** No new learned weights means no new source of the bimodal,
  seed-dependent instability that Trust Gate reproduced (the same failure signature seen
  throughout Step 6's depth-gate work) and that Cross-Attention's larger parameter count risks
  as well.
- **Grounded in a real, measured correlation.** Prerequisite 1 confirmed retrieval confidence
  genuinely tracks how much Transformer 2 moves a prediction (r=+0.77) — a real mechanism to
  exploit, even though this specific dataset can't yet reward it with a labeled AUC win.
- **Directly evidenced by Prerequisite 0.** Since no learned fusion parameter can be pushed by
  training to do better than chance on this dataset's only hidden-dependency scenario, the
  mechanism that adds the *least* new trainable surface area — and therefore the least new risk
  — is the correct default until a dataset with genuine incremental cross-supplier signal
  becomes available.

**Recommendation:** adopt Confidence-Aware Fusion (`rgcn_attn_t2_confidence`) as the standing
Transformer 2 fusion design — not because it wins on AUC (nothing does, on this dataset), but
because it is free, stable, and the closest to preserving Transformer 2's one genuinely
validated capability (hidden-dependency discovery) while carrying the least risk of the kind of
instability seen in every other learned per-node gate this project has built. Per-Node Trust
Gate is not discarded outright — its instability, not its AUC null result, is flagged as the
open question worth revisiting if a future dataset provides genuine incremental signal.
Cross-Attention Fusion is the least justified of the three on current evidence (highest cost,
weakest discovery-quality retention, no AUC compensation).

**What would actually resolve this:** a dataset built the way Task 3's dual-sourcing scenario
was — an explicit, incremental cross-supplier bleed-through term (`COPARENT_COUPLING`) — applied
to a Transformer-2-style undirected/non-graph-edge relationship, not H_POLYMER as it currently
exists. Until then, `hidden_dependency_links` remains a validated investigation-leads tool for
human review, and its fusion into impact's AUC remains an unproven, not-yet-production
accuracy improvement.

---

## Governance record (both rounds)

**Round 1** — backed up before any write
(`reports/backups/model_registry_and_evals_backup_20260808_201700.sql`). New table
`hidden_dependency_links` created. 10 new registry rows, all `status='active'` — 308 total.
150 new evaluation rows — 4,488 total. 2,524 new `hidden_dependency_links` rows. No existing
row from any prior round modified. Log: `reports/logs/run_transformer2_pilot_20260808_202618.log`.

**Round 2** — backed up before any write
(`reports/backups/model_registry_and_evals_backup_20260808_215212.sql`). 25 new registry rows
(5 baseline-comparison + 5 confidence + 5 trustgate + 5 crossattn + 5 re-run additive-fusion),
all `status='active'` — 333 total. 375 new evaluation rows — 4,863 total. 7,557 new
`hidden_dependency_links` rows (15 variant runs, most-recent-snapshot only) — 10,081 total. No
existing row from any prior round modified. Log:
`reports/logs/run_transformer2_fusion_pilot_20260808_215220.log`.

---

*Scoped to SHARE + Variant A + Transformer 2 across both rounds — no changes to SHARP
(`rgcn_relemb`), SHARK (`rgcn_battn`), or the existing Markov/Variant A depth-selection logic.
Round 1 wired Transformer 2 to the IMPACT head, deviating from
`docs/10_AI_ML_Documentation.md` §8.3's delay-only specification — see §1.1 above for the full
rationale.*
