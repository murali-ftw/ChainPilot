# Phase 16 — Two encoder variants: trajectory-similarity attention and degree-adaptive relational aggregation

Branch `phase16-encoders`, off `90a38ed`. Nothing is merged and **nothing entered `shipped.json`**. Pre-registration:
`phase-16-preregistration.md` (`bdafd27`, committed before any measurement). Stage 0: `phase-16-stage0.md`
(`a2d1a35`). Every Phase 16 artifact is stamped clean:
- Stage 0 `d17289d`;
- training cells `c99af0e` / `e7163e4`, which have identical `ml/` (see deviation 153);
- scores and diagnostics `e7163e4`.

Learning rates are each task's shipped rate, 2.5e-4 for arrival and capacity, with **no retune** (deviation 144).
Selection was on validation only. Every arm has five model seeds {7, 17, 27, 37, 47}, read by exact bundle name.

## Summary

| Candidate | Task | Verdict vs own incumbent | Notes |
|---|---|---|---|
| **A1** share_traj (Δ present) | arrival | **tie** (undetermined on val, test, lateness); PASS | control A2 inside A0's band on every surface → A1/A2 **valid**. The model learned near-zero weights on Δ |
| A2 share_traj (Δ zeroed) — control | arrival | tie; PASS | valid control |
| **B1** relational PNA | arrival | **tie**; PASS | sits low on every surface; validation bands overlap by 0.0006 |
| **B2** low-degree bypass, k = 1 | arrival | **tie**; PASS | k selection degenerate (k = 1 ≡ k = 2) |
| **B3** PNA + bypass, k = 1 | arrival | **WORSE — disjoint on test C-index**; FAIL | val and lateness undetermined |
| **B1** relational PNA | capacity | **tie**; PASS | |
| B2 / B3 | capacity | not trained: identical to B0 / B1 by construction | HeteroMP has no softmax to bypass (deviation 149) |

**No variant beats its incumbent disjointly (P5 right).** Both candidates are **closed negatives** on this graph:
- **A** is a tie whose model barely uses the new term.
- **B** is a tie (B1, B2) or worse (B3).

Under the S1 rule, nothing merges.

---

## 1. Stage 0 — the publishable measurements

Full detail in `phase-16-stage0.md`; the headline figures follow.

**Attention entropy by node type** (figure: `reports/part2/figures/phase16_entropy.svg`). Shipped SHARE-lite arrival
h4, 5 seeds, all 9 test snapshots, reaching paths only. H is normalised by log|N(i)| over the softmax the model
actually computes (relation-blind, whole neighbourhood).

| node type | median H_norm | H_norm < 0.10 | H_norm > 0.90 | |N(i)| = 1 |
|---|---|---|---|---|
| channel (|N| = 3) | 0.107 | **49.1%** (L4: 88.1%) | 5.6% | 0% |
| supplier (median 38) | 0.727 | 0.01% | 0.7% | 0% |
| part (median 26) | 0.822 | 3.1% | 31.3% | 0% |
| plant (median 2,285) | 0.875 | 0% | **35.4%** | 0% |

**Normalised effective message magnitude into channels** (s_r/Σs, 5-seed mean [min, max]):

| layer | supplier→ch | part→ch | plant→ch |
|---|---|---|---|
| 1 | 0.253 [0.050, 0.635] | 0.212 [0.021, 0.614] | 0.535 [0.054, 0.930] |
| 2 | 0.614 [0.001, 0.832] | 0.281 [0.051, 0.994] | 0.105 [0.005, 0.173] |
| 3 | 0.772 [0.424, 1.000] | 0.142 [0.000, 0.523] | 0.085 [0.000, 0.256] |
| 4 | 0.403 [0.000, 0.991] | 0.590 [0.006, 1.000] | 0.007 [0.000, 0.019] |

Entity nodes have one incoming relation each, so their share is 1.0 by construction.

**Degree spectrum (confirmed):**

| relation (target) | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| channel→supplier (420 suppliers) | 21 | 30 | 38 | 46 | 55 |
| channel→part (620 parts) | 12 | 19 | 26 | 32 | 45 |
| channel→plant (7 plants) | 2,227 | 2,232 | 2,285 | 2,359 | 2,362 |
| each entity→channel relation | 1 | 1 | 1 | 1 | 1 |

A channel has 3 neighbours in total. 7 nodes, all plants, aggregate over more than 1,000 neighbours.

## 2. The reachability map and exclusions

**3 of 24** (layer, relation, target) paths do not reach the readout: layer 4, channel→{supplier, part, plant}. The
analytic map equals the gradient map (|∂ readout/∂g| on a per-path message gate) on all five seeds. JSON:
`reports/part2/phase-16-reachability.json`.

Excluded from **every** measurement in this phase (Stage 0 and all diagnostics):
- those 3 paths, i.e. 1,047 entity node-states per snapshot × 9 test snapshots × 5 seeds;
- the **49,152 parameters** of deviation 107 (3 × 128 × 128).

## 3. G1 — inertness, with the gate shown firing

S2 and S3 were followed:
- **No edit to `tcn.py` or `share.py`.**
- **Phase 16 classes are new, in new files** (`ml/models/phase16_encoders.py`, `ml/train/phase16_heads.py`).
- **The TCN per-depth states come from forward hooks** on `res[1..5]` and `norm`. They are registered in share_traj's
  setup and removed on teardown.
- **`loop.py` gained one routing function.** It returns `P5.HeadNet` itself unless a non-default Phase 16 axis is
  set.

G1 ran the same script against a worktree at the branch point `90a38ed` and against the branch (`2e34c8a`). It ran
on **CPU, float32**, because MPS is not bit-repeatable: the incumbent SHARE-lite forward run twice in one process
differs at 1e-8 (deviation 147). Byte hashes:

| check | branch point = branch? | with SHARELayer.forward perturbed by +1e-6 |
|---|---|---|
| (a) TCN output, seed-7 init, one window, all 16,072 channels | **identical** | identical (path not touched) |
| (b) SHARE-lite arrival h4 forward | **identical** | **DIFFERS — gate fires** |
| (b) HeteroMP capacity h4 forward | **identical** | identical (path not touched) |
| (c) 3-epoch seed-7 training, arrival lite h4, all weights | **identical** | **DIFFERS — gate fires** |
| (c) 3-epoch seed-7 training, capacity mp h4, all weights | **identical** | identical |

The perturbation was reverted (it was a run-time monkeypatch flag, `--perturb share`, never a code edit).
Artifacts: `ml/artifacts/phase16_g1_{bp,head,perturbed}.json`.

At initialisation, A2 (Δ zeroed, Δ weights zero-initialised) and A1 reproduce SHARE-lite's forward pass to 1.5e-8 on
MPS, the same magnitude as MPS's own run-to-run noise. The variant draws no extra random numbers.

## 4. Per-candidate arms, bands, verdicts

Verdict rule: "gain"/"worse" only when the 5-seed [min, max] bands are disjoint, otherwise **undetermined** (a tie).
PASS means not disjointly worse on any surface. `ml/eval/phase16_score.py` → `ml/artifacts/phase16_scores.json`.

### 4.1 Candidate A — trajectory-similarity attention (arrival only)

The only change is the logit: e_ij = LeakyReLU(aᵀ[W_r h_i ‖ W_r h_j ‖ Δ_ij]). Here Δ_ij = [cos(z_i^(l), z_j^(l))]
for l = 1..6, the TCN's per-dilation block states at the last week (entity nodes: the mean of members' states, as
SHARE seeds them). Δ is detached. Its 6 weights per layer are zero-initialised, adding 24 parameters in total.
Capacity has no attention, so it has no Candidate A arm.

| arm | val C-index | test C-index | test lateness ROC-AUC | epochs |
|---|---|---|---|---|
| **A0** share_lite (stored, not retrained) | 0.67067 [0.66904, 0.67193] | 0.67442 [0.67344, 0.67551] | 0.70905 [0.70645, 0.71298] | 53–60 |
| **A1** Δ present | 0.67112 [0.67072, 0.67146] | 0.67483 [0.67412, 0.67645] | 0.71083 [0.70760, 0.71416] | 53–66 |
| **A2** Δ zeroed (control) | 0.67067 [0.66880, 0.67175] | 0.67428 [0.67294, 0.67573] | 0.70956 [0.70734, 0.71232] | 50–56 |

- **Control:** A2 is not disjoint from A0 on any surface → the implementation has no side effect; **A1 and A2 are
  valid.**
- **A1 vs A0:** undetermined on all three surfaces. It is higher in the mean everywhere (+0.0005 val, +0.0004 test,
  +0.0018 lateness), with a tighter validation band, but this is not a difference. **Verdict: tie. PASS.**
- **Baseline stop condition:** not triggered. A0 was not retrained; the stored bundles were reused under an identity
  the S4 gate proved unchanged.

**Diagnostics** (`ml/eval/phase16_diag.py` → `ml/artifacts/phase16_diag.json`, test split, reaching paths):

*Learned weights on Δ (per layer: mean |w| over the 6 depths; in brackets, the depth with the largest |w|):*

| seed | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| 7 | 0.048 (d2) | 0.015 (d6) | 0.002 (d2) | 0.014 (d1) |
| 17 | 0.010 (d1) | 0.005 (d1) | 0.012 (d1) | 0.007 (d2) |
| 27 | 0.008 (d1) | 0.005 (d1) | 0.018 (d3) | 0.013 (d1) |
| 37 | 0.007 (d6) | 0.008 (d1) | 0.022 (d1) | 0.006 (d1) |
| 47 | 0.028 (d6) | 0.007 (d3) | 0.005 (d1) | 0.010 (d6) |

Every weight is below 0.06 in magnitude. For scale, the node-attention vectors are initialised at std 0.1, and a
cosine lies in [−1, 1]. The shortest timescale (depth 1, dilation 1) is the most frequent arg-max: 11 of 20
(seed, layer) cells. There is no consistent depth across seeds, so **the model does not use any timescale of Δ in a
reproducible way**.

*What vs scale.* In each trained A1 model, Δ was zeroed at inference and the attention re-computed:

| seed | channel paths whose top-attended entity changes, L1/L2/L3/L4 | mean abs α change, L1/L2/L3/L4 |
|---|---|---|
| 7 | 0.26% / 0.00% / 0% / 0% | 0.0038 / 0.0039 / 0.0001 / 0.0002 |
| 17 | 0.01% / 0.02% / 0% / 0.00% | 0.0002 / 0.0002 / 0.0013 / 0.0001 |
| 27 | 0.10% / 0.00% / 0.19% / 0.01% | 0.0009 / 0.0000 / 0.0019 / 0.0008 |
| 37 | 0.04% / 0.03% / 0.13% / 0% | 0.0013 / 0.0006 / 0.0029 / 0 |
| 47 | 0.11% / 0.02% / 0% / 0% | 0.0015 / 0.0011 / 0 / 0 |

Δ changes **what** attention does on at most 0.26% of channel paths, and its size by at most 0.004 in α.

*Entropy by node type, A1 vs A0* (pooled over 5 seeds; median; < 0.10; > 0.90):

| | channel | supplier | part | plant |
|---|---|---|---|---|
| A0 | 0.107; 49.1%; 5.6% | 0.727; 0.0%; 0.7% | 0.822; 3.1%; 31.3% | 0.875; 0.0%; 35.4% |
| A1 | 0.096; 50.4%; 3.8% | 0.723; 0.0%; 7.4% | 0.895; 1.6%; 48.8% | 0.905; 0.0%; 53.3% |

The channel profile is unchanged. At part and plant nodes, A1's attention is **more uniform**: near-uniform paths go
from 31% to 49% and from 35% to 53%. These figures are pooled, unbanded, and compare different trained models, so
some of the shift may be training variance. Because Δ itself barely moves α (above), this is **not** Δ acting at
inference: if anything, Δ's presence during training left the high-degree attention flatter.

Recorded as the brief asks: **a tie with a changed attention-entropy profile at entity nodes — a smaller finding,
not a failure, and not attributable to Δ at inference.** A learned φ in place of cosines is **not** warranted: the
condition was "only if cosines show an effect", and they do not.

### 4.2 Candidate B — degree-adaptive relational aggregation (relational PNA; Corso et al., 2020)

**B-1:** per relation, AGG_r(i) = [mean ‖ max ‖ min ‖ std] of the messages m_j, times the degree scalers
S(d, α) = (log(d+1)/δ_r)^α for α ∈ {1, 0, −1}. Then one linear map takes 12d back to d.
- The messages are m_j = α_ij W_r h_j for share_pna and m_j = up_r(h_j) for heteromp_pna.
- δ_r is the mean of log(d+1) over relation r's destination nodes. The graph is static, so this equals the training
  mean.

**B-2:** where |N_r(i)| ≤ k, α = 1 and the message passes directly; the remaining edges are softmaxed among
themselves.

**k selection** (rule fixed in `ml/train/phase16_queue.py` before any B2 number existed). Stage 0.3 proved that
k = 1 and k = 2 select the identical 48,216 edges, all into channels, so k = 1. The validation "curve" is two points
at seed 7: k1 0.67009, k2 0.67023. The difference between two runs of the same computation is MPS noise, which is
deviation 149. **Selected k = 1**, frozen before any test number was computed (`ml/artifacts/phase16_k_selection.json`).

**Arrival** (incumbent A0 above):

| arm | val C-index | test C-index | test lateness ROC-AUC | verdict |
|---|---|---|---|---|
| B0 = A0 | 0.67067 [0.66904, 0.67193] | 0.67442 [0.67344, 0.67551] | 0.70905 [0.70645, 0.71298] | — |
| **B1** PNA | 0.66833 [0.66734, 0.66967] | 0.67273 [0.67175, 0.67359] | 0.70627 [0.70256, 0.71096] | tie (all undetermined); PASS |
| **B2** bypass k=1 | 0.66958 [0.66764, 0.67123] | 0.67364 [0.67280, 0.67475] | 0.70618 [0.70242, 0.70842] | tie; PASS |
| **B3** PNA + bypass | 0.66799 [0.66714, 0.66908] | **0.67117 [0.66969, 0.67191]** | 0.70198 [0.69397, 0.70792] | **WORSE (test C-index disjoint); FAIL** |

B1 is below A0's mean on every surface. Its bands overlap A0's by margins of 0.0006 (val) and 0.00015 (test), so
the verdict is a tie, and it is the least comfortable tie in this phase. The bypass does not remove a no-op on this
model. It replaces each channel's learned 3-way selection (§1) with an unweighted sum. Combined with PNA, it is the
one arm disjointly worse.

**Capacity** (mean pinball, lower is better):

| arm | val pinball | test pinball | verdict |
|---|---|---|---|
| **B0** heteromp (stored) | 0.06981 [0.06879, 0.07035] | 0.07277 [0.07213, 0.07408] | — |
| **B1** heteromp_pna | 0.06998 [0.06925, 0.07058] | 0.07349 [0.07301, 0.07377] | tie; PASS |
| B2 / B3 | = B0 / = B1 by construction | | not trained (deviation 149) |

The baseline stop condition was not triggered: B0 was not retrained.

**Diagnostics.**

*Incoming-aggregate norm, B1 / B0* (ratio of 5-seed means; per-seed values in `phase16_diag.json`):

| arrival | channel | supplier | part | plant |
|---|---|---|---|---|
| L1 | **5.34** | 0.52 | 0.09 | 0.35 |
| L2 | 2.42 | 0.75 | 0.65 | **0.04** |
| L3 | 1.22 | 0.87 | 0.09 | **0.13** |
| L4 | 0.68 | (dead) | (dead) | (dead) |

| capacity | channel (what each channel receives) | supplier | part | plant |
|---|---|---|---|---|
| round 1 | 3.66 | 3.95 | 5.57 | **4.67** |
| round 2 | 2.54 | 2.33 | 4.54 | **4.96** |

Measured as |log ratio|:
- **arrival** plant aggregation moves more than channel aggregation at L2 and L3 (3.3 and 2.1 vs 0.9 and 0.2), but
  **less at L1** (1.0 vs 1.7). Pooled over reaching layers, plant moves more: 2.1 vs 0.8;
- **capacity** plant moves more than channel in both rounds (1.54 vs 1.30; 1.60 vs 0.93).

*Plant nodes — share of the projected multi-aggregate's variance by aggregator* (scalers pooled; over the 7 plants ×
test snapshots × layers or rounds):

| | mean | max | min | std |
|---|---|---|---|---|
| arrival (share_pna) | 0.000 | 0.754 | 0.235 | 0.011 |
| capacity (heteromp_pna) | **0.310** | 0.338 | 0.320 | 0.032 |

**The mean does not dominate on either task.** On capacity, where messages are not pre-weighted, max, min and std
carry 69% of the plant aggregate's variance.

On arrival the mean's ~0 share is an **artefact of the brief's message definition**. m_j = α_ij W_r h_j with
α ≈ 1/2,285 makes the mean of the messages O(1/d²), so arrival's variance split says nothing about whether averaging
loses information (deviation 150). Capacity is the valid test.

So multi-aggregation does carry non-mean information at the plants, and does move plant aggregation more than
channel aggregation. **It buys nothing measurable on either task's metric.**

## 5. Pre-registered predictions

| | Prediction | Mark | Evidence |
|---|---|---|---|
| **P1** | Channel nodes show near-zero entropy on the majority of reaching paths (a structural no-op); suppliers do not | **WRONG** | 49.1% < 50% of reaching channel paths are near zero. More fundamentally, a channel's softmax is over 3 entities, not 1; low entropy there is a learned hard selection, not a no-op. (The supplier half holds: 0.01%.) |
| **P2** | Part's normalised message share is the lowest of the three entity relations, consistent with C3 | **WRONG** | Lowest only at L1, and not disjointly. Plant is lowest at L2–L4, and part is largest at the readout layer. Magnitude ≠ ablation (deviation 148) |
| **P3** | Candidate A's effect on arrival is small or null | **RIGHT** (the stated reason is wrong) | Tie on all surfaces; Δ weights below 0.06. The premise "attention is a no-op at the readout, so Δ can act only at supplier and plant nodes" is false (P1). Δ can act at channels; the model simply did not use it |
| **P4** | Candidate B moves plant-node aggregation more than channel-node aggregation | **RIGHT, with one exception** | Capacity: yes in both rounds. Arrival: yes pooled and at L2–L3; **no at L1**. The mean does not dominate the plant multi-aggregate (capacity 31%). The averaging loss was real and measurable, but recovering it did not improve any metric |
| **P5** | Neither variant beats its incumbent disjointly at 5 seeds | **RIGHT** | No disjoint gain anywhere; one disjoint loss (B3, arrival test C-index) |

## 6. Deviations (continuing from 142)

| # | Deviation | Where |
|---|---|---|
| **143** | **A fifth identity axis was added.** `traj_delta` ('zeroed' for the A2 control) was added to the four axes the brief listed; without it A1 and A2 would share one identity. Like the others, it is omitted when default | `ml/artifact_identity.py` |
| **144** | **The brief's learning rate is wrong for these tasks.** "Shipped 1.25e-4 throughout" is fill's rate. Arrival and capacity ship 2.5e-4 (`ml/configs/shipped.json`), and "no retune" was honoured by keeping each task's shipped rate | brief |
| **145** | **The brief names classes and modules that don't exist.** There is no `ShareLiteLayer`: the incumbent is `SHARELayer` with `n_bases=None` (`ml/models/share.py`). `HeteroMP` lives in `tcn.py`, which may not be edited, so `HeteroMPPNA` is in a new file. The TCN has no per-dilation-block module, so the hooks sit on `res[l]` (whose input is block state z_l) and on `norm` (whose input is z_6). No fallback subclass was needed | brief S2/S3 |
| **146** | **The softmax is relation-blind, not per relation.** The brief, and `results/observation1.md:143-144`, write α_ij^r = softmax_j per relation. `SHARELayer.forward` takes the softmax over the whole incoming neighbourhood across relations, so per-relation |N_r(i)| = 1 at channels says nothing about attention. `results/observation1.md:180-184` says the channel softmax "is over a single element and does no work"; it is over three, and 49.1% of channel paths (88% at L4) are a near-hard learned selection. Stage 0.1 measured the softmax the model computes. Prior text is not edited | share.py; observation1.md |
| **147** | **G1 ran on CPU.** MPS is not bit-repeatable even for the unchanged incumbent against itself (1e-8, `index_put_with_accumulate`), so a bit-level gate on MPS would fail for reasons unrelated to Phase 16 | §3 |
| **148** | **Gate 0.2 disagrees with C3.** Effective message magnitude ranks plant lowest into channels at L2–L4, while C3's ablation finds plant live and part dead. Magnitude measures what the next layer receives, not what the readout uses | Stage 0.2 |
| **149** | **B-2 is degenerate on this graph.** k = 1 and k = 2 select identical edge sets, so the "validation curve" is two runs of one computation (seed-7 values differ by 1.4e-4: MPS noise). HeteroMP has no softmax, so its bypass is the identity; capacity B2 ≡ B0 and B3 ≡ B1 were not trained. On arrival the bypass does not skip a deterministic softmax; it replaces a learned 3-way selection with an unweighted sum | Stage 0.3; §4.2 |
| **150** | **Arrival's plant variance-share test is uninformative.** With m_j = α_ij W_r h_j (as the brief specifies), the mean aggregator is O(1/d²) at a plant, so its ~0 share is structural. Capacity's unweighted messages are the valid test | §4.2 |
| **151** | **The queue was killed and relaunched, resetting one stage clock.** The training queue was attached to the agent session, which ended at ~00:18 Thu mid-cell (B1 arrival, s37). It was relaunched detached at 00:44; completed cells were skipped by their full-identity markers, and s37 retrained from scratch. Stage b_arr's 12 h cap clock restarted at the relaunch (cap never reached: b_arr ran 21:04–00:18 before the kill and 00:44–04:32 after, 7 h 02 m in total). No STOPPED marker was written by any stage | `ml/artifacts/phase16_queue.txt` |
| **152** | **Stage 4B's capacity re-run misses its validation band by 1.3e-5.** The capacity s7 re-run lands **outside** the stored validation band: 0.0703657 vs the band's top, 0.0703528. That top edge is the stored s7 run itself. The re-run's test pinball (0.07250) is inside. G1 proved the incumbent path bit-identical on CPU, so this is MPS run-to-run nondeterminism on the band-defining seed, not Phase 16 code, but **the letter of 4B ("both must land inside") is not met for capacity validation** | §8 |
| **153** | **`results/observation1.md` changed on this branch, twice.** (i) A `git commit -am` in this phase swept the user's uncommitted edit to it into a code commit (`6087d28`). It was undone by a soft reset within a minute, before anything ran on it, and replaced by `2e34c8a` (G1 fix only). (ii) The user's own commit `e7163e4` "Observations" (Wed 22:17) is on this branch and modifies `results/observation1.md`; it touches no `ml/` file, so cells stamped `c99af0e` and `e7163e4` ran identical code. No Phase 16 code or report writes to `results/` | §8 |
| **154** | **The end-of-phase identity gate reports 250 raw failures, all by design.** Re-run after training, the S4 gate counts 250 name differences between the branch-point module and the Phase 16 module. **All 250 are the 31 Phase 16 bundles**, whose non-default axes the branch-point code cannot name; **0 of the 340 other configs** (338 Phases 0–15 + 2 audit re-runs) differ | §8 |

## 7. Literature positioning

**share_traj — a variant of published work, not novel.** Adding an edge-level term inside a GAT attention logit,
e_ij = LeakyReLU(aᵀ[W h_i ‖ W h_j ‖ W_e e_ij]), is edge-featured graph attention:
- Gong & Cheng, "Exploiting Edge Features for Graph Neural Networks", CVPR 2019; also EGAT (2021);
- PyTorch Geometric's `GATConv(edge_dim=…)` implements exactly this form.

Making that edge feature a **similarity of the endpoints' learned temporal embeddings** is also published in the
spatio-temporal GNN literature:
- STA-GNN adds a contextual-similarity term computed from temporally encoded node features directly to the
  attention logits before the softmax (Koistinen et al., arXiv:2603.10676, 2026);
- STFGNN builds its graph from DTW similarity of node time series (Li & Zhu, AAAI 2021);
- DSTAGNN derives spatial-temporal aware attention from historical-sequence similarity (Lan et al., ICML 2022).

The temporal-graph attention models are adjacent rather than identical. TGAT (Xu et al., ICLR 2020) puts a
functional time encoding into the attention keys and queries. TGN (Rossi et al., 2020) adds per-node memory with
attention-based aggregation. Neither uses an endpoint-trajectory similarity scalar.

**What is specific here** is one design detail: the similarity is split into one cosine per TCN dilation depth (six
scalars). That is a parameterisation choice, not a contribution, and it showed no effect. **The trajectory-similarity
logit is already published in substance; no novelty is claimed.**

**share_pna / heteromp_pna — an application, not an invention.** This is Principal Neighbourhood Aggregation (Corso,
Cavalleri, Beaini, Liò & Veličković, "Principal Neighbourhood Aggregation for Graph Nets", NeurIPS 2020): the
mean/max/min/std aggregators and the logarithmic degree scalers, used verbatim. The only things specific to this
phase are:
- applying it per relation on a heterogeneous graph;
- the motivation from Stage 0's measured degree spectrum (seven nodes of degree above 2,200 beside 16,072 of
  degree 3).

**SHARE-lite itself.** The graph stage is an R-GAT-family relational attention encoder (Busbridge, Sherburn,
Cavallo & Hammerla, "Relational Graph Attention Networks", 2019):
- GAT's attention form (Veličković et al., ICLR 2018) is used verbatim;
- the free per-relation weights W_r are R-GCN's base model (Schlichtkrull et al., ESWC 2018), without its basis
  decomposition.

**No architectural novelty is claimed for SHARE-lite.**

## 8. Stage 4 — isolation audit

**A. Empty diffs from the branch point `90a38ed` to HEAD:**

| path | diff |
|---|---|
| `ml/models/tcn.py` (TCN **and** the incumbent HeteroMP class) | **empty** |
| `ml/models/share.py` (the incumbent SHARELayer / SHARE classes) | **empty** |
| `ml/configs/shipped.json` | **empty** |
| `db/` (incl. `db/validator.py`, `db/dataset_structure.md`, `db/gen_v6–v8/`) | **empty** |
| `docs/` (incl. `docs/specs/model_plan.md`, `docs/specs/synthetic_rules.md`) | **empty** |
| every prior report under `reports/` | **empty** (only new Phase 16 files added) |
| `results/` | **not empty: the user's own commit `e7163e4`** (deviation 153); no Phase 16 commit touches it |

Files Phase 16 changed:
- modified: `ml/artifact_identity.py` (S4 axes) and `ml/train/loop.py` (routing, CLI flags, teardown). G1 covers
  both;
- added: the Phase 16 scripts, encoders and reports.

**B. End-to-end re-runs on the Phase 16 branch** (stored config, seed 7, separate bundle root
`ml/artifacts/phase16_audit_bundles/`, stamp `e7163e4`):

| config | val | test | lateness (test) | inside stored band? |
|---|---|---|---|---|
| arrival lite h4 s7 | 0.67125 (band [0.66904, 0.67193]) | 0.67436 ([0.67344, 0.67551]) | 0.71187 ([0.70645, 0.71298]) | **yes, all three** |
| capacity mp h4 s7 | 0.0703657 (band [0.06879, 0.0703528]) | 0.07250 ([0.07213, 0.07408]) | — | test **yes**; val **NO, by 1.3e-5** (deviation 152) |

**C. S4 identity gate, restated.**
- **Before any training** (`c5ce3f2`): 338 stored configs, 29 queue markers, 600 manifest entries, 198 index keys,
  2,704 name-function outputs and 1,352 score names. All were recomputed with the Phase 16 axes present and compared
  with the branch-point module and the stored strings: **0 mismatches**.
- **Shown firing:** a non-default `encoder_variant` changed all 8 names; an explicit default changed none; reverting
  restored them.
- **Re-run at the end** of the phase: **0 mismatches across all 340 non-Phase-16 configs**. The 250 raw differences
  are exactly the 31 new Phase 16 bundles (deviation 154).
- Artifacts: `ml/artifacts/phase16_identity_gate.json` (pre-training) and `phase16_identity_gate_end.json`.

## 9. inventory_position_weekly reads

**NONE.** Verified by searching every Phase 16 file and the Phase 16 diff of `loop.py` for the table name, with zero
matches:
- `ml/models/phase16_encoders.py`;
- `ml/train/phase16_heads.py` and `ml/train/phase16_queue.py`;
- `ml/eval/phase16_*.py`;
- `ml/artifact_identity.py`.

Phase 16 reads only the arrival and capacity label frames and the channel panel, through the unchanged
`phase5_heads.device_inputs` / `labels` path, plus stored bundles.

## Sources

- Corso et al., Principal Neighbourhood Aggregation for Graph Nets, NeurIPS 2020.
- [Gong & Cheng, Exploiting Edge Features for Graph Neural Networks, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_Exploiting_Edge_Features_for_Graph_Neural_Networks_CVPR_2019_paper.pdf)
- [Koistinen et al., Spatio-Temporal Attention Graph Neural Network, arXiv:2603.10676](https://arxiv.org/html/2603.10676v1)
- [Li & Zhu, Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16542/16349)
- [Lan et al., DSTAGNN, ICML 2022](https://proceedings.mlr.press/v162/lan22a/lan22a.pdf)
- [Xu et al., Inductive Representation Learning on Temporal Graphs (TGAT), ICLR 2020](https://openreview.net/forum?id=rJeW1yHYwH)
- [Rossi et al., Temporal Graph Networks (TGN), 2020](https://github.com/twitter-research/tgn)
- Busbridge et al., Relational Graph Attention Networks, 2019.
- Veličković et al., Graph Attention Networks, ICLR 2018.
- Schlichtkrull et al., Modeling Relational Data with Graph Convolutional Networks, ESWC 2018.
