# Phase 16 — Stage 0: measurement on the shipped SHARE-lite arrival encoder

Branch `phase16-encoders` (off `90a38ed`). Script `ml/eval/phase16_stage0.py`, run at code commit **`d17289d`**
(clean; stamp in `ml/artifacts/phase16_stage0.json`). Forward passes only, plus one gradient probe for the
reachability map. No training. Pre-registration committed first (`bdafd27`).

Bundles read, by exact name: `ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s{7,17,27,37,47}`. Test split:
all **9** arrival test snapshots. The instrumented forward pass reproduces the incumbent `SHARE.forward` to at most
**7.6e-6** absolute on every seed (MPS float32). A larger gap would have meant measuring a different model.

## Gates and outcomes

| Gate | Result | Consequence |
|---|---|---|
| 0.0 reachability | **3 of 24** (layer, relation, target) paths are dead: layer 4, channel→{supplier, part, plant}. Analytic map = gradient map, identical on all 5 seeds | 1,047 entity node-paths per snapshot and **49,152 parameters** (= deviation 107, confirmed) are excluded from every measurement below |
| **0.1 attention entropy (P1)** | **FIRED — P1 WRONG.** Channel paths with H_norm < 0.10: **49.1%**, which is not a majority. Suppliers: 0.01% | The premise is also wrong in kind (below): channel attention is a **learned selection among three terms**, not a structural no-op. The motivation for the low-degree bypass (B-2) is **gone**; the motivation for Candidate A at channel nodes is **not** |
| **0.2 message magnitude (P2)** | **DISAGREES with C3.** Part's share is lowest at layer 1 only (not disjoint). At layers 2–4 plant is lowest, and 5-seed bands span most of [0, 1] | Deviation **148**: message magnitude and ablation measure different things. P2 is WRONG |
| 0.3 degree spectrum | Recorded medians **confirmed**: channel 3, part 26, supplier 38, plant 2,285. **7 nodes** (all plants) aggregate over more than 1,000 neighbours | B-2's k = 1 and k = 2 select the **identical** 48,216 edges (deviation 149) |

**Neither candidate is cancelled.** Candidate A goes ahead: channel attention is live, and Δ acts there. B-1 (PNA)
goes ahead: attention at high-degree nodes is closest to uniform, i.e. closest to a plain mean (plant nodes: 35%
of paths near uniform, 66% at layer 1). **B-2 (bypass) runs as the brief specifies, but its stated rationale does
not hold.** On this model it does not remove a no-op. It replaces a learned three-way selection at every channel
with an unweighted sum.

## 0.0 The reachability map

The readout is the arrival head applied to every channel's h⁴. Each (layer, relation) path's messages were scaled
by a gate g = 1, and |∂ readout / ∂g| was measured on seed 7's first test snapshot. The zero/non-zero pattern is the
same on all five seeds and equals the analytic map. The JSON is at `reports/part2/phase-16-reachability.json` (a
copy of `ml/artifacts/phase16_reachability.json`).

| layer | channel→supplier | channel→part | channel→plant | supplier→channel | part→channel | plant→channel |
|---|---|---|---|---|---|---|
| 1 | yes | yes | yes | yes | yes | yes |
| 2 | yes | yes | yes | yes | yes | yes |
| 3 | yes | yes | yes | yes | yes | yes |
| 4 | **no** | **no** | **no** | yes | yes | yes |

**Excluded from every Stage 0 measurement:** 3 of 24 paths. That is 1,047 entity nodes per snapshot × 9 snapshots
× 5 seeds of layer-4 entity states, and the 3 × 128 × 128 = **49,152** parameters of deviation 107.

## 0.1 Attention entropy by node type

**What the model actually computes.** `SHARELayer.forward` (`ml/models/share.py`) takes the softmax over each
destination's **whole** incoming neighbourhood, across all relations. A channel's neighbourhood is exactly 3 edges:
one supplier, one part, one plant. So:

- **fraction of nodes with |N(i)| = 1 (softmax deterministic): 0%** for every node type;
- per relation, |N_r(i)| = 1 for **100%** of channels. But α is not computed per relation, so that count says
  nothing about what attention does.

H is normalised by log|N(i)|: log 3 for channels, log(member count) for entities. Pooled over reaching layers,
5 seeds and 9 test snapshots:

| node type | n paths | mean | median | p10 | p90 | H_norm < 0.10 | H_norm > 0.90 (near-uniform) |
|---|---|---|---|---|---|---|---|
| channel | 2,892,960 | 0.265 | 0.107 | 0.000 | 0.804 | **49.1%** | 5.6% |
| supplier | 56,700 | 0.701 | 0.727 | 0.525 | 0.839 | 0.01% | 0.7% |
| part | 83,700 | 0.754 | 0.822 | 0.379 | 0.979 | 3.1% | **31.3%** |
| plant | 945 | 0.808 | 0.875 | 0.507 | 0.953 | 0.0% | **35.4%** |

By layer (median H_norm; fraction < 0.10; 5-seed band of the per-seed median):

| layer | channel | supplier | part | plant |
|---|---|---|---|---|
| 1 | 0.291; 27.7%; [0.084, 0.475] | 0.606; 0%; [0.515, 0.688] | 0.821; 0%; [0.711, 0.993] | 0.920; 0%; [0.891, 0.997] |
| 2 | 0.289; 31.2%; [0.004, 0.636] | 0.755; 0%; [0.697, 0.795] | 0.861; 0%; [0.734, 0.974] | 0.871; 0%; [0.772, 0.949] |
| 3 | 0.105; 49.4%; [0.000, 0.705] | 0.779; 0%; [0.719, 0.826] | 0.601; 9.2%; [0.140, 0.961] | 0.561; 0%; [0.461, 0.870] |
| 4 | **0.005; 88.1%**; [0.000, 0.073] | excluded (dead) | excluded | excluded |

**Figure:** `reports/part2/figures/phase16_entropy.svg`, the H_norm distribution by node type (reaching paths only;
dashed lines at 0.10 and 0.90).

**Gate 0.1 fired; P1 is WRONG.** The threshold was declared in the script before it ran: "near zero" means
H_norm < 0.10, and the gate needs more than 50% of reaching channel paths. The pooled figure is 49.1%. The margin
is thin, but the more important error is **what low channel entropy means**:

- **It is not a structural no-op.** With three terms, H ≈ 0 means attention puts nearly all its weight on **one** of
  the channel's three entities. That is a hard, learned choice of which relation to hear.
- **At the readout layer (L4) that choice is near-total:** 88% of channel paths are below 0.10, and the per-seed
  medians sit in [0.000, 0.073].
- **It is seed-dependent at L2–L3:** the per-seed median ranges from 0.00 to 0.70.

The "listening to your only informant" picture in `results/observation1.md:180-184` does not describe this model
(deviation **146**). **Suppliers do not show near-zero entropy.** That half of P1 holds, but P1 is scored as a whole.

## 0.2 Effective message magnitude per relation

s_r^l = mean_i ‖Σ_{j∈N_r(i)} α_ij W_r h_j‖₂, over reaching paths, averaged over the 9 test snapshots. The share is
normalised within (layer, node type) and shown as the mean over 5 seeds with a [min, max] band. Channels are the
only node type with more than one incoming relation. Every entity node has exactly one, so its share is 1.0 by
construction and carries no comparison.

| layer | supplier→channel | part→channel | plant→channel | lowest |
|---|---|---|---|---|
| 1 | 0.253 [0.050, 0.635] | 0.212 [0.021, 0.614] | 0.535 [0.054, 0.930] | part (not disjoint) |
| 2 | 0.614 [0.001, 0.832] | 0.281 [0.051, 0.994] | 0.105 [0.005, 0.173] | plant |
| 3 | 0.772 [0.424, 1.000] | 0.142 [0.000, 0.523] | 0.085 [0.000, 0.256] | plant |
| 4 | 0.403 [0.000, 0.991] | 0.590 [0.006, 1.000] | 0.007 [0.000, 0.019] | plant |

**Gate 0.2: DISAGREES with C3 → deviation 148.** C3's ablation says part is dead and supplier and plant are live.
By magnitude:

- **plant is the smallest message into channels at layers 2, 3 and 4**, and at the readout layer it is 0.7% of the
  total;
- **part is the largest at the readout layer in the seed mean (0.590)**, reaching 100% on at least one seed;
- the 5-seed bands are so wide (most span more than 0.5) that no ranking is stable across seeds except "plant
  smallest at L4".

So message magnitude and ablation measure different things. A message can be large and carry nothing the readout
needs (a large near-constant part message is one way), and a small message can carry what the ablation finds
indispensable. **P2 is WRONG.** The brief warned against reading ‖W_r‖ or gates. This shows that s_r is not enough
either: it is what the downstream layer receives, not what it uses.

(Context only, one seed: the reachability probe's |∂readout/∂g| on seed 7 ranks supplier→channel highest at every
layer (430 at L3, 288 at L4) and part and plant far below (3.6/0.9 at L3, 5.6/3.3 at L4). Sensitivity puts
supplier first, as C3 does, but cannot separate part from plant. It is one seed and is not used for any gate.)

## 0.3 Degree spectrum, re-verified from the graph

| relation (target) | nodes | min | p10 | median | p90 | max | > 1,000 |
|---|---|---|---|---|---|---|---|
| channel→supplier (supplier) | 420 | 21 | 30 | **38** | 46.1 | 55 | 0 |
| channel→part (part) | 620 | 12 | 19 | **26** | 32 | 45 | 0 |
| channel→plant (plant) | 7 | 2,227 | 2,232 | **2,285** | 2,359 | 2,362 | **7** |
| supplier→channel (channel) | 16,072 | 1 | 1 | 1 | 1 | 1 | 0 |
| part→channel (channel) | 16,072 | 1 | 1 | 1 | 1 | 1 | 0 |
| plant→channel (channel) | 16,072 | 1 | 1 | 1 | 1 | 1 | 0 |
| whole neighbourhood (channel) | 16,072 | 3 | | **3** | | 3 | 0 |

All recorded medians are confirmed. No entity is empty. **Seven nodes, the seven plants, aggregate over more than
1,000 neighbours.** The smallest per-relation entity neighbourhood is 12, so the bypass sets are:

- |N_r(i)| ≤ 1: 48,216 edges, all into channels;
- |N_r(i)| ≤ 2: the same 48,216 edges.

**k = 1 and k = 2 are the same intervention on this graph** (deviation 149).
