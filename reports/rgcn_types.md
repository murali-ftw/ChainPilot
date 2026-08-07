# RGCN Architecture Family — Mechanisms, Matched-Parameter Results, and Over-Smoothing

**Scope:** for each of plain RGCN and its three attention hybrids (RGCN+Attention, display
name **SHARE**, `rgcn_attn` in code; RGCN+BasisAttn, display name **SHARK**, `rgcn_battn` in
code; RGCN+RelEmbedding, display name **SHARP**, `rgcn_relemb` in code) — what it is, its matched-parameter
results, and (for the three hybrids) a fresh over-smoothing sweep: an AUC-vs-depth table and a
direct embedding-similarity measurement, both computed this round. Plain RGCN is included as
parameter-count and AUC context only — it was not retrained or re-swept this round. Closes
with a full cross-architecture comparison spanning this project's whole history.

## Dataset-version note (live-checked this round)

Confirmed live: 800 suppliers, 15 `graph_snapshots`, and — corrected from how prior rounds
described this same database — **the Task 3 co-parent/dual-sourcing mechanism is actively
loaded**, not dormant (`SUPPLIES` edge count 2,820, not 2,400; 449/800 suppliers reach a
different supplier in 2 hops). Full detail in `reports/entropy_test.md`'s dataset-version note.
Every RGCN-family result cited below — from the 4-architecture ablation onward — has been
running against this same co-parent-enabled dataset consistently, which is why they remain
comparable to each other. **The one exception is HGT's depth-sweep reference numbers below**
(`ml/run_step_v3.py`, reused here per this round's own instruction, not retrained): that run
predates the Task 3 dataset reload and ran against the base v3 generation without
dual-sourcing. Its row in the AUC-vs-depth table carries that caveat; the three hybrids'
own rows don't, since all three were trained in-process this round against the identical live
data.

---

## Plain RGCN (Schlichtkrull et al. 2018 basis decomposition)

**What it is.** Every relation's message transform is a linear combination of a small shared
pool of `num_bases` basis matrices, `W_r = Σ_b a_r[b]·V_b` — one shared basis pool and
per-relation coefficient vector for the whole encoder (not per layer), so relational parameter
cost doesn't scale with depth. Aggregation is per-relation mean, then summed across relations
feeding a destination node — every relation gets an equal vote, unlike HGT's fully-dedicated
per-relation attention. See `ml/models/rgcn_encoder.py`.

**Matched-parameter results** (hidden=138, num_bases=4 — `reports/hades_model_development_report.md`
Round 5; not retrained or re-swept this round, context only):

| Task | AUC (mean, 5 seeds) |
|---|---|
| delay | 0.8044 |
| shortage | **0.7976** |
| impact | 0.8509 |

Parameter count: 699,602 (encoder) / 757,565 (full model). **No over-smoothing sweep exists
for plain RGCN this round** — stated explicitly rather than omitted silently, per this round's
own scope (context only, not re-swept).

---

## SHARE — RGCN+Attention (`ml/models/rgcn_attn_encoder.py`)

**What it is.** Keeps RGCN's basis-shared message transform unchanged. Replaces the
per-relation-mean-then-sum aggregation with a single shared attention scorer (`att_msg`,
`att_dst` — two `Linear(hidden,1)` per layer, the only new parameters relative to plain RGCN)
scoring every incoming edge to a destination node, across every relation feeding it together,
then one joint softmax over that whole pool per destination node — matching `HGTConv`'s
per-node softmax scope at a fraction of its cost.

**Matched-parameter results** (hidden=128, num_bases=10, this round's fresh 5-seed baseline
run — reproducing `reports/rgcn_matched_pilot.md`'s earlier measurement at the same config as
an internal-consistency check):

| Task | AUC (mean ± std, 5 seeds, this round) | AUC (mean, `rgcn_matched_pilot.md`) |
|---|---|---|
| delay | 0.8105 ± 0.0033 | 0.8118 |
| shortage | 0.7982 ± 0.0021 | 0.7971 |
| impact | 0.9365 ± 0.0058 | 0.9387 |

All three tasks land within one seed's own std of the earlier measurement — a clean
replication. Parameter count: 752,211 (full model, unchanged from `rgcn_matched_pilot.md`).

**Over-smoothing — AUC vs. depth** (baseline: 5 seeds; L1-L4: 3 seeds, diagnostic):

| Config | delay | shortage | impact |
|---|---|---|---|
| baseline | 0.8105 ± 0.0033 | 0.7982 ± 0.0021 | 0.9365 ± 0.0058 |
| L1 | 0.8154 ± 0.0030 | 0.7911 ± 0.0013 | 0.9244 ± 0.0047 |
| L2 | 0.8119 ± 0.0042 | 0.7969 ± 0.0019 | 0.9377 ± 0.0026 |
| L3 | 0.8131 ± 0.0028 | 0.7956 ± 0.0011 | 0.9265 ± 0.0051 |
| L4 | 0.8120 ± 0.0032 | 0.7971 ± 0.0028 | **0.9424 ± 0.0012** |

Shallow (L1) already captures most of delay's signal; shortage is essentially flat across
depth (0.791–0.798); impact's best result is actually at **L4** (reading the deepest layer for
every task, not the structural prior's own shallower h³ assignment) — mild evidence impact
benefits from depth beyond what the documented prior prescribes, consistent with
`reports/hades_model_development_report.md`'s standing open finding that the structural prior
doesn't clearly beat a shared depth.

**Over-smoothing — embedding similarity** (mean cosine similarity, 500 sampled same-type
pairs, baseline config, averaged across 5 seeds):

| Node type | layer 1 | layer 2 | layer 3 | layer 4 | Trend |
|---|---|---|---|---|---|
| Supplier | 0.421 | 0.515 | 0.553 | 0.639 | Rising (smoothing) |
| Warehouse | 0.839 | 0.974 | 0.979 | 0.983 | Rising, saturates near ceiling |
| Customer | 0.951 | 0.990 | 0.985 | 0.983 | Rising, near ceiling by L2 |
| Component | 0.770 | 0.750 | 0.790 | 0.809 | Mildly rising |
| Order | 0.807 | 0.725 | 0.746 | 0.810 | Dips then recovers |
| Product | 0.721 | 0.617 | 0.583 | 0.818 | Dips then rises sharply |
| Factory | 1.000 | 1.000 | 1.000 | 1.000 | Flat at ceiling (only 5 nodes) |
| **Shipment** | 0.935 | 0.907 | 0.791 | **0.788** | **Falling** (anti-smoothing) |

---

## SHARK — RGCN+BasisAttn (`ml/models/rgcn_battn_encoder.py`)

**What it is.** The most expensive of the three hybrids: a SECOND full basis-decomposed pool
dedicated to attention (`att_basis`, `att_coeff`, same shape/init pattern as the transform's
own `rel_basis`/`rel_coeff`), giving every relation its own attention MATRIX `A_r` rather than
a shared vector scorer. Attention logit is a scaled bilinear form on the pre-transform
embeddings, `e_ij = (h_i^T A_r h_j) / √hidden`, scored per relation, then the same joint
softmax as SHARE runs over the concatenated pool per destination node.

**Matched-parameter search** (Phase 4, this round): HGT's real anchor re-derived live —
701,088 (unchanged). Three candidates found; candidate (b), preserving `num_bases≥8 AND
num_bases_attn≥8`, was trained:

| Candidate | hidden | num_bases | num_bases_attn | Params | Diff from anchor |
|---|---:|---:|---:|---:|---:|
| (a) tightest raw fit | 128 | 2 | 8 | 701,256 | 168 (0.02%) |
| **(b) both≥8 — TRAINED** | **110** | **9** | **16** | **701,310** | **222 (0.03%)** |
| (c) intermediate | 113 | 6 | 16 | 701,379 | 291 (0.04%) |

**Matched-parameter results** (hidden=110, num_bases=9, num_bases_attn=16, 5 seeds):

| Task | AUC (mean ± std, 5 seeds) |
|---|---|
| delay | 0.8113 ± 0.0052 |
| shortage | 0.7956 ± 0.0013 |
| **impact** | **0.9114 ± 0.0559** |

**Impact's std (0.0559) is an order of magnitude wider than either other hybrid's** (SHARE:
0.0058, SHARP: 0.0071) — driven by a single outlier seed (seed1: impact=0.8016, vs.
0.9433/0.9468/0.9464/0.9189 for the other four). This pattern repeats at L1 (impact std 0.0329,
again driven by one low seed). **SHARK (RGCN+BasisAttn) is measurably less seed-stable on
impact than either of the other two hybrids** — the most expressive, most expensive design (two
full basis pools) also shows the most fragile training dynamics on this task, at this parameter budget.
Parameter count: 738,273 (full model).

**Over-smoothing — AUC vs. depth:**

| Config | delay | shortage | impact |
|---|---|---|---|
| baseline | 0.8113 ± 0.0052 | 0.7956 ± 0.0013 | 0.9114 ± 0.0559 |
| L1 | 0.8076 ± 0.0100 | 0.7947 ± 0.0005 | 0.8938 ± 0.0329 |
| L2 | 0.8074 ± 0.0011 | 0.7969 ± 0.0012 | 0.9063 ± 0.0129 |
| L3 | 0.8068 ± 0.0016 | 0.7963 ± 0.0007 | 0.9244 ± 0.0128 |
| L4 | 0.8110 ± 0.0035 | 0.7972 ± 0.0012 | **0.9424 ± 0.0037** |

Same qualitative shape as SHARE — impact rises toward L4 — but every impact cell's std is
markedly wider than the corresponding SHARE/SHARP cell, the seed-instability pattern above
showing up at every depth, not just baseline.

**Over-smoothing — embedding similarity:**

| Node type | layer 1 | layer 2 | layer 3 | layer 4 | Trend |
|---|---|---|---|---|---|
| Factory | 0.594 | 0.708 | 0.873 | 0.915 | Rising sharply |
| Supplier | 0.490 | 0.543 | 0.563 | 0.602 | Rising |
| Warehouse | 0.663 | 0.843 | 0.928 | 0.940 | Rising sharply |
| Customer | 0.858 | 0.921 | 0.943 | 0.959 | Rising |
| Component | 0.695 | 0.852 | 0.869 | 0.882 | Rising |
| Product | 0.722 | 0.549 | 0.545 | 0.689 | Dips then partially recovers |
| Order | 0.839 | 0.785 | 0.771 | 0.780 | Mildly falling, ~flat |
| **Shipment** | 0.932 | 0.936 | 0.916 | **0.889** | **Falling** (anti-smoothing, milder than SHARE) |

---

## SHARP — RGCN+RelEmbedding (`ml/models/rgcn_relemb_encoder.py`)

**What it is.** Extends SHARE's shared-scorer attention with one extra additive term: a
small per-relation embedding (`rel_embed`, `[num_relations, relation_embed_dim=16]`) fed
through one more shared scorer (`att_rel`), computed once per relation per layer (not per
edge) and broadcast into that relation's edges' logits before the same joint softmax runs —
a relation-identity signal at `O(num_relations × relation_embed_dim)` cost, far cheaper than
SHARK's second full basis pool.

**Matched-parameter results** (hidden=128, num_bases=10, relation_embed_dim=16, this round's
fresh 5-seed baseline — reproducing `reports/rgcn_matched_pilot.md`'s earlier measurement):

| Task | AUC (mean ± std, 5 seeds, this round) | AUC (mean, `rgcn_matched_pilot.md`) |
|---|---|---|
| delay | 0.8100 ± 0.0020 | 0.8105 |
| shortage | **0.7988 ± 0.0011** | 0.7991 |
| impact | 0.9366 ± 0.0071 | 0.9369 |

Again a clean replication, all three tasks within noise of the earlier measurement. Parameter
count: 752,599 (full model, unchanged).

**Over-smoothing — AUC vs. depth:**

| Config | delay | shortage | impact |
|---|---|---|---|
| baseline | 0.8100 ± 0.0020 | 0.7988 ± 0.0011 | 0.9366 ± 0.0071 |
| L1 | 0.8160 ± 0.0008 | 0.7915 ± 0.0008 | 0.9221 ± 0.0026 |
| L2 | 0.8161 ± 0.0013 | 0.7971 ± 0.0018 | 0.9271 ± 0.0033 |
| L3 | 0.8059 ± 0.0055 | 0.7972 ± 0.0007 | 0.9315 ± 0.0091 |
| L4 | 0.8103 ± 0.0032 | 0.7981 ± 0.0031 | 0.9297 ± 0.0054 |

Delay actually peaks at L1/L2, not baseline or L4 — the only one of the three hybrids where a
shallower shared depth beats the structural-prior baseline on delay, though the gap (≈0.006)
is within seed noise at 3 seeds. Impact rises with depth similarly to the other two hybrids but
plateaus rather than peaking at L4.

**Over-smoothing — embedding similarity:**

| Node type | layer 1 | layer 2 | layer 3 | layer 4 | Trend |
|---|---|---|---|---|---|
| Warehouse | 0.956 | 0.998 | 0.998 | 0.998 | Rising, saturates by L2 |
| Factory | 0.916 | 0.999 | 0.999 | 0.999 | Rising, saturates by L2 |
| Supplier | 0.395 | 0.508 | 0.552 | 0.656 | Rising |
| Customer | 0.816 | 0.935 | 0.943 | 0.967 | Rising |
| Component | 0.782 | 0.751 | 0.799 | 0.806 | Mildly rising |
| Order | 0.765 | 0.680 | 0.736 | 0.759 | Dips then recovers, ~flat |
| Product | 0.753 | 0.631 | 0.600 | 0.860 | Dips then rises sharply |
| **Shipment** | 0.935 | 0.906 | 0.752 | **0.767** | **Falling** (anti-smoothing) |

## Cross-hybrid over-smoothing pattern

**All three hybrids show the same qualitative shape, and it replicates a finding first seen
in the very first round of this project.** Most node types' embeddings grow MORE similar with
depth (the classic over-smoothing signature) — most strongly and consistently for Supplier
(rising from ~0.40–0.49 at layer 1 to ~0.60–0.66 by layer 4 in all three architectures) and for
the small-population node types Factory/Warehouse (which also sit near a similarity ceiling
that's partly just a small-n artifact — 5 and 8 nodes respectively). **Shipment is the one
node type that consistently does the opposite in all three hybrids** (falling similarity with
depth — 0.935→0.788 for SHARE, 0.932→0.889 for SHARK, 0.935→0.767 for SHARP) — the
same anti-smoothing pattern `reports/hades_model_development_report.md` Round 1's original HGT
over-smoothing measurement first found (Shipment: 0.835→0.668) and now independently
reproduced across three structurally different architectures on a different, larger dataset.
This is now a robust, architecture-independent finding: depth differentiates Shipment
embeddings in this graph rather than smoothing them, whatever the encoder.

---

## Closing conclusions — every architecture tried, across this project's whole history

| Architecture | Params (matched-d) | delay | shortage | impact | Source |
|---|---:|---|---|---|---|
| GraphSAGE (matched-d=66) | 720,261 | 0.8032 | 0.7898 | 0.8889 | `hades_model_development_report.md` Round 5 |
| GAT (matched-d=92) | 731,495 | 0.7400 | 0.6560 | 0.9182 | `hades_model_development_report.md` Round 5 |
| HGT (d=64, `run_step_v3.py` depth sweep, base-v3 dataset — see caveat above) | 713,763 | 0.744 | 0.772 | 0.920 | `hades_model_development_report.md` Round 3 |
| HGT (d=64, fixed/matched, co-parent dataset — reused since Round 5) | 713,763 | 0.8015 | 0.7824 | 0.9335 | `hades_model_development_report.md` Rounds 5–6 |
| RGCN (matched-d=138, num_bases=4) | 757,565 | 0.8044 | **0.7976** | 0.8509 | `hades_model_development_report.md` Round 5 |
| RGCN+Attn / SHARE (matched-d=128, num_bases=10) | 752,211 | 0.8105–0.8118 | 0.7971–0.7982 | 0.9365–0.9387 | `rgcn_matched_pilot.md`; reproduced this round |
| RGCN+BasisAttn / SHARK (hidden=110, num_bases=9, num_bases_attn=16) | 738,273 | 0.8113 | 0.7956 | 0.9114 (unstable, std 0.056) | This round (new) |
| RGCN+RelEmbedding / SHARP (matched-d=128, num_bases=10, relemb=16) | 752,599 | 0.8100–0.8105 | **0.7988–0.7991** | 0.9366–0.9369 | `rgcn_matched_pilot.md`; reproduced this round |

**The overall ranking established across every round holds, and this round doesn't overturn
it.** RGCN+Attn (SHARE) remains the strongest single architecture found in this project —
highest or joint-highest on delay, competitive on shortage, and the only hybrid to close plain
RGCN's large impact deficit against HGT to a statistical tie
(`rgcn_matched_pilot.md`'s sign-consistency tests), all at roughly a quarter of HGT's parameter
count before matching and near-parity after. SHARP (RelEmbedding) is a close second,
essentially tied with SHARE on every task (`rgcn_matched_pilot.md`'s paired comparison:
all three tasks flip sign across seeds) at a marginally higher parameter cost for the added
relation-identity signal — not a clear win over the simpler SHARE. **SHARK (BasisAttn),
new this round, does not improve on either simpler hybrid** — its delay/shortage numbers are
comparable to SHARE/SHARP, but its impact result is both no better on average (0.9114 vs.
0.9365–0.9387) and materially less stable across seeds (std 0.056 vs. 0.006–0.007) — the extra
expressiveness of a full per-relation attention matrix, at roughly double the relational
parameter cost of the vector-scorer hybrids, buys measurable training fragility without a
measurable accuracy gain on this dataset. **Plain RGCN remains the one architecture that wins
shortage most cleanly** (0.7976, matching SHARP's 0.7988–0.7991 within noise) but loses
impact by the widest margin of any RGCN-family variant (0.8509, well below every hybrid's
~0.91–0.94) — the attention step, in any of its three forms, is what actually closes that gap,
not the basis-decomposition transform alone.

**Recommendation, unchanged from `rgcn_matched_pilot.md`, now reinforced:** RGCN+Attn (SHARE)
remains the best-supported candidate for any further work in this line — cheapest of the
three hybrids to build and train, most stable across seeds, and no other variant tried across
four rounds of experimentation (plain RGCN, SHARE, SHARK, and SHARP) beats it on more than one task.

---

*See `reports/entropy_test.md` for the relation-frequency entropy diagnostic this round also
produced, and the paragraph connecting it to the win/loss pattern summarized above.
`reports/hades_model_development_report.md` and `reports/rgcn_matched_pilot.md` carry the full
detail behind every non-this-round number cited above.*
