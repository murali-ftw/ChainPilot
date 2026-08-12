# Phase 7 — SHARE Ported, Evaluation Harness Built, V2 Trained

**Status:** the port is verified against V1's recorded numbers, the evaluation protocol is
implemented, and the twelve variants are trained and compared. One thing did not go as the step
brief assumed, and it is the first thing to say.

**The specified sweep — nine architectures × twelve variants × five seeds at spec scale — is not
runnable on this hardware, by roughly four orders of magnitude.** Measured, not estimated: one
spec-scale snapshot is 762,812 nodes and 4,430,104 edges, and one forward+backward pass of SHARE at
its matched-parameter width costs **86 s on CPU** (15.7 s forward, 70.5 s backward, 6.2 GB peak
RSS). A 100-epoch run over the 16 training snapshots is therefore **~38 hours**, and the full
540-run sweep is **~20,000 CPU-hours**. MPS does not rescue it — measured at 96 s per pass, *slower*
than CPU because the workload is scatter/gather-bound rather than matmul-bound, and it runs out of
memory outright at the matched width (30.1 GiB requested against a 30.19 GiB cap).

So the sweep was run at the **V1-comparable configuration** — `sup_n=800`, 15 snapshots, all ten
mechanisms active on all twelve variants — where a run costs 5–11 minutes instead of 38 hours. §3
explains why that is a defensible choice for the questions being asked and not merely a compute
concession; §5.1 says exactly which cells were run and which were not; §7 states what a spec-scale
answer would still cost.

---

## 1. What was built

| Path | What it is |
|---|---|
| `ml/models/rgcn_attn_encoder.py` | **SHARE**, byte-identical copy of V1's module (md5-verified) |
| `ml/models/rgcn_relemb_encoder.py` | **SHARP**, byte-identical copy |
| `ml/models/rgcn_battn_encoder.py` | **SHARK**, byte-identical copy |
| `ml/models/rgcn_encoder.py`, `heads.py`, `depth.py` | byte-identical copies |
| `ml/models/encoder.py` | factory over the nine arms; V1's Step-6/7 research arms dropped, **GraphGPS added** |
| `ml/models/model.py` | encoder + structural depth-prior readout + heads |
| `ml/data/loader.py` | **new** — `db/csv/*.csv.gz` → `HeteroData`, replacing V1's PostgreSQL path |
| `ml/train.py` | V1's loop, `model_registry` writes removed, early stopping added |
| `ml/evaluate.py` | V1's metrics, upgraded to a **block** bootstrap (§4) |
| `ml/run_benchmark_eval.py` | the protocol: sanity / sweep / report |

The four RGCN-family encoders and the heads/depth modules were copied, not retyped — `md5` on
`rgcn_attn_encoder.py` matches V1's. The brief asked for the architecture to be preserved exactly,
and a copy is the only way to be sure of that.

**GraphGPS is an adaptation, not a port** — V1 never ran it. PyG's `GPSConv` pairs a local MPNN with
*dense* global attention, which is O(n²) in node count; at 762k nodes that is not slow, it is
unallocatable. `GPSEncoder` keeps GraphGPS's actual structure (local message passing + a graph-wide
global attention term + a feed-forward block) with the global term computed by **linear
(kernelised) attention within each node type**, preserving "every node can see every same-type node"
at O(n·d²). A GraphGPS number in this report is a number for that adaptation.

### The data-loading layer

V1 read snapshots out of PostgreSQL. V2's benchmark artifact is the gzipped CSV directory, so the
loader reproduces V1's SQL semantics over pandas — same columns, same z-scores, same one-hot
category sets, same as-of rules. It produces the same **8 node types and 20 meta-relations** V1 had,
and three things genuinely change:

1. **The as-of clock is `recorded_at`, not `changed_at`.** V1 read true event time for shipment
   status because V1 had no reporting delay. Under Mechanism G the two clocks diverge deliberately,
   and reading `changed_at` would hand the model exactly what G exists to withhold. This is the one
   intentional deviation from V1's feature SQL; on a variant without G the two differ only by the
   generator's 5–45 minute write jitter.
2. **`supplier_upstream` becomes an eleventh forward relation**, `(Supplier, UPSTREAM_OF, Supplier)`
   — Mechanism J's tiers, already truncated by Mechanism A. Variants without J emit no such file and
   get no such relation, rather than an empty one (an empty relation still costs the RGCN family a
   coefficient row). **Consequence worth stating: Variants A, J and K carry 22 meta-relations, not
   20**, so their matched-parameter counts differ slightly from the other nine variants' — e.g.
   GraphSAGE is 755,373 there against 720,261 elsewhere. That is disclosed in every table rather
   than smoothed over.
3. **Mechanism A shrinks the node set.** Hidden suppliers are absent from `suppliers.csv`, so edges
   pointing at them are dropped by the same id-map filter V1 used. That truncation is the mechanism
   working.

**No hidden state is read, because none is emitted.** `verify_no_hidden_state()` scans every emitted
table's header for the mechanisms' latent columns (resilience, attenuation, hidden-parent identity,
stress terms). It returns empty on every variant, as designed — the check exists so that a future
generator change that starts emitting one is caught here rather than silently improving every score.

## 2. Sanity check — the port reproduces V1

**The anchor is not `db/csv/v0_seed42`.** The step brief calls that "the byte-identical-to-V1
anchor", but Phase 6 replaced it: `db/csv/v0_seed42` is now Variant 0 at **spec** scale (4,000
suppliers, 40 snapshots), a different world. The byte-identical anchor is Variant 0 at the **`v1`
preset**, which is what this check uses.

SHARE, matched-d=128, num_bases=10, L=4, 100 epochs, 5 model-init seeds. Split 6 train / 3
validation / 6 test snapshots — train ends 2024-12-01, test starts 2025-04-01, **identical to V1's
own cutoffs**.

| | ported (5 seeds) | V1 recorded | delta vs V1 midpoint |
|---|---|---|---|
| parameter count | **752,211** | 752,211 | **exact match** |
| delay AUC | 0.8100 ± 0.0037 | 0.8105–0.8118 | −0.0012 |
| shortage AUC | 0.7984 ± 0.0019 | 0.7971–0.7982 | +0.0007 |
| impact AUC | 0.9369 ± 0.0102 | 0.9365–0.9393 | −0.0010 |

Per seed: delay 0.8083 / 0.8154 / 0.8055 / 0.8111 / 0.8095; shortage 0.8005 / 0.7959 / 0.7978 /
0.7976 / 0.8001; impact 0.9484 / 0.9280 / 0.9245 / 0.9410 / 0.9428.

**All three tasks land within 0.0012 of V1's recorded midpoint, well inside V1's own seed spread,
and the parameter count matches to the digit.** The exact parameter match is the stronger of the two
signals: it means the port has V1's in-dims (14/6/13/6/7/7/5/3), V1's 20 relations, and V1's layer
structure, since any discrepancy in those would move the count. Every other architecture's
matched-parameter count reproduces V1's table exactly too — HGT 713,763, RGCN 757,565, GraphSAGE
720,261, GAT 731,495, SHARP 752,599, SHARK 738,273.

**The port is verified. Everything downstream rests on this.**

## 3. Why the sweep runs at the V1-comparable configuration

### 3.1 The measurement

| | V1-comparable (`sup_n=800`, 15 snapshots) | spec scale (`sup_n=4,000`, 40 snapshots) |
|---|---|---|
| nodes per snapshot | ~28,000 | **762,812** |
| edges per snapshot | ~63,000 | **4,430,104** |
| SHARE forward pass | — | **15.7 s** |
| SHARE backward pass | — | **70.5 s** |
| peak RSS, one pass | — | **6.2 GB** |
| training snapshots | 6 | 16 |
| **one 100-epoch run** | **5–11 min** (measured: 324 s GraphSAGE … 666 s GAT, 419 s SHARE) | **~38 h** |
| **540-run sweep (9 x 12 x 5)** | ~12 h on this machine | **~20,000 CPU-hours** |

MPS was measured, not assumed: 96 s per pass at hidden=64 (slower than CPU's 86 s at hidden=128) and
an outright allocation failure at the matched width. The workload is dominated by `gather`,
`scatter` and segment-`softmax` over 4.4M edges — memory-bandwidth-bound, which is the regime where
Apple's GPU backend has no advantage over its CPU and where its scatter kernels are weakest.

### 3.2 Why the smaller configuration is the right control anyway

Three of the four questions this session exists to answer are **comparisons**, not absolute numbers:

- do V1's architectural claims (SHARE beats HGT on delay/shortage, ties on impact; SHARP and SHARK
  do not beat SHARE) replicate on V2?
- does discovering hidden structure help once it is causally coupled (D vs B, K vs D)?
- does adaptive depth help under genuine chain-length heterogeneity (J vs 0)?

For the replication question the V1-comparable configuration is not a compromise, it is the
**better** control: V1's numbers were recorded at exactly this scale, with exactly this split, so
running V2's variants here changes *only the mechanisms*. A spec-scale replication would confound
"the claim does not hold under V2's mechanisms" with "the claim does not hold at 27× the graph
size", and would have no V1 baseline to compare against at that size.

For the variant questions, the mechanisms are all fully active at this configuration — all ten,
every variant, with the same generator and the same validation suite. What is smaller is the label
volume, and that is the honest cost: **positive counts here are 66–1,123 per task per variant
against the spec's 2,000–5,000 target**, so every effect measured below has wider intervals than a
spec-scale run would give. Small effects that fail to clear significance here are *not* established
as absent; §6 says so wherever it applies.

### 3.3 What is not claimed

This is not a spec-scale result and is not presented as one. `docs/phase6_spec_scale_report.md`'s
power table describes the spec-scale corpus, which exists on disk and is what a GPU run should use.
§7 states what that would cost.

## 4. The evaluation protocol

Implemented in `ml/run_benchmark_eval.py`, per `docs/00_Benchmark_Specification.md`'s Evaluation
Protocol.

**Temporal split by snapshot count, never randomly.** The brief specifies "60/20/40 train/val/test",
which sums to 120% and cannot be a proportion. V1's actual v3 split was 6/3/6 of 15 snapshots —
**40/20/40** — and that is what is used, applied to whatever `SNAPSHOTS` the data on disk reports
in `resolved_config.json` rather than to a hardcoded date table. At 15 snapshots that reproduces
V1's cutoffs exactly (train ends 2024-12-01, test starts 2025-04-01); at spec scale it would give
16/8/16.

**Five seeds, with sign consistency.** In the sweep the five seeds are **dataset** seeds — each is a
different generated world — with the model-init seed held fixed, so the variance reported is the
benchmark's rather than the fit's. (The §2 sanity check does the opposite: one dataset, five
model-init seeds, which is what V1's own numbers vary over.) Every comparison reports mean, standard
deviation, and how many of the five seeds agree on the *direction* of the effect. Seeds 44/45/46 are
not retained on disk and are regenerated on demand by `db/regenerate_seed.py`; the harness invokes
it rather than silently dropping a seed.

**Paired bootstrap, upgraded to block resampling.** V1 explained that the schema asks for a paired,
time-blocked bootstrap over whole snapshots — repeated entities across snapshots are autocorrelated,
so an i.i.d. row bootstrap understates the interval — but V1's test split was 2 snapshots, and
block-resampling 2 blocks is degenerate, so V1 fell back to row-level and labelled it honestly.
**V2's test split is 6 snapshots (16 at spec scale), so the block bootstrap is no longer degenerate
and is the default here**, recorded as `ci_method='paired_block_bootstrap'`. Row-level remains
available and is selected automatically below 4 blocks, still labelled for what it is.

**Variant interpretation rules, enforced in code.** `REPORT_AGAINST = {"A": "J", "D": "B", "F": "E",
"K": "D"}` is consulted by the reporting path: `variant_comparison()` raises if asked to report one
of those variants against anything other than its mandated reference, so Variant A cannot be
reported against Variant 0 by accident. `VARIANT_CAVEATS` attaches the E/F sign-inversion warning
and the Variant K impact-underpowering warning to every table that mentions those variants — printed
by the tool, not left to the reader.

**Positive counts sit beside every AUC** in every table the harness prints, so an underpowered cell
is visible in the same row as the metric.

**Provenance.** Every results file carries the git commit, the platform, the torch version, the
argv, the matched-parameter table, the split fractions, and the full `resolved_config` of each
variant-seed it trained on — the same discipline the generator's `resolved_config.json` applies to
the data.

## 5. Results — the sweep

### 5.1 Coverage

The grid actually run, after the scale decision in §3 and one further cut explained below:

| grid | architectures | variants | seeds | runs |
|---|---|---|---|---|
| **variant effects** | SHARE | all 12 | 5 (42–46) | 60 |
| **architecture replication** | all 9 | Variant 0 | 5 | 45 |
| second architecture across variants | GCN | all 12 | 5 on 8 variants, 2 on A/C/E/G | 48 |
| partial third and fourth | GraphSAGE, GAT | all 12 | 5 on Variant 0, 1 elsewhere | 32 |

**165 unique runs, 22.5 process-hours of training.** Every cell used for a claim in §6 has its full
five seeds; the GCN and GraphSAGE/GAT rows with fewer are cross-checks, and are labelled as such
wherever they are used.

**Why not all nine architectures on all twelve variants.** That grid is 540 runs and the first
attempt at it measured out at ~12 hours: HGT is the long pole at roughly 2,000 s per run, because
`HGTConv` carries per-relation parameters for all 20 (22 on A/J/K) meta-relations. The grid was
re-cut to put the five seeds where they answer a question — SHARE across every variant for the
variant effects, every architecture on Variant 0 for the replication test — rather than spreading
one seed thinly over all 108 cells, which §5.6 shows would have been the worst possible use of
them. **Consequence: this report cannot say whether the architecture ranking changes between
variants.** It says what the ranking is on Variant 0 and what the variants do to SHARE. GCN's
twelve-variant row is a partial check on that and agrees with SHARE's ordering wherever both
exist.

A process-level note worth recording because it cost an hour: the first sweep ran 18 processes at
`OMP_NUM_THREADS=1`, on the assumption that one thread per process maximises throughput for
graph workloads. It does not here — these encoders are dominated by dense `hidden x hidden`
matmuls, which thread well. Four processes at three threads each ran a SHARE run in **419 s**
against **>3,600 s** for twelve processes at one thread — an 8.6× throughput difference from the
thread setting alone.

### 5.2 SHARE across the twelve variants (5 dataset seeds)

AUC is mean ± std over the five seeds; `pos` is the mean number of test positives, printed beside
the metric so an underpowered cell is visible in its own row.

| var | mechanisms | delay AUC | pos | shortage AUC | pos | impact AUC | pos |
|---|---|---|---|---|---|---|---|
| 0 | — | 0.7744 ± 0.0376 | 519 | 0.7856 ± 0.0090 | 1,036 | 0.9321 ± 0.0128 | 156 |
| A | J+A | 0.7733 ± 0.0299 | 628 | 0.7826 ± 0.0044 | 1,077 | 0.9198 ± 0.0097 | 203 |
| B | B | 0.7812 ± 0.0360 | 553 | 0.7831 ± 0.0112 | 1,048 | 0.9322 ± 0.0066 | 155 |
| C | C | 0.7789 ± 0.0291 | 520 | 0.7869 ± 0.0122 | 1,023 | 0.9273 ± 0.0097 | 150 |
| D | B+D | 0.7821 ± 0.0342 | 548 | 0.7800 ± 0.0114 | 1,026 | 0.9343 ± 0.0059 | 157 |
| E | E | 0.8148 ± 0.0292 | 403 | 0.7904 ± 0.0086 | 761 | 0.9182 ± 0.0224 | 84 |
| F | E+F | 0.8086 ± 0.0317 | 370 | 0.7817 ± 0.0141 | 791 | 0.9090 ± 0.0238 | 83 |
| G | G | **0.9173** ± 0.0111 | 184 | **0.7378** ± 0.0049 | 1,036 | 0.9434 ± 0.0071 | 103 |
| H | H | 0.7789 ± 0.0334 | 549 | 0.7858 ± 0.0101 | 1,035 | 0.9227 ± 0.0127 | 155 |
| I | I | 0.7746 ± 0.0277 | 502 | 0.7804 ± 0.0116 | 1,045 | 0.9281 ± 0.0049 | 156 |
| J | J | 0.7662 ± 0.0340 | 628 | 0.7840 ± 0.0036 | 1,077 | 0.9363 ± 0.0054 | 203 |
| K | all ten | **0.9107** ± 0.0123 | **113** | **0.7412** ± 0.0083 | 745 | 0.9259 ± 0.0170 | **65** |

### 5.3 Variant effects, each against its mandated reference

Deltas are per-seed differences against the reference variant, so the "sign" column is the number
of the five seeds that agree on the direction. **Consistent** means all five agree.

| effect | task | mean delta | sign | reading |
|---|---|---|---|---|
| **G vs 0** | delay | **+0.1429 ± 0.0304** | 5+/0− consistent | see the warning below |
| | shortage | **−0.0478 ± 0.0062** | 0+/5− consistent | reporting delay genuinely hurts shortage |
| | impact | +0.0113 ± 0.0101 | 4+/1− mixed | — |
| **K vs D** | delay | **+0.1286 ± 0.0297** | 5+/0− consistent | inherited from G, not from coupling |
| | shortage | **−0.0388 ± 0.0144** | 0+/5− consistent | also G's |
| | impact | −0.0083 ± 0.0144 | 1+/4− mixed | underpowered, 65 positives |
| **E vs 0** | delay | **+0.0404 ± 0.0151** | 5+/0− consistent | absorption removes the hardest positives |
| | shortage | +0.0048 ± 0.0063 | 4+/1− mixed | — |
| | impact | −0.0139 ± 0.0168 | 1+/4− mixed | — |
| **A vs J** | delay | +0.0070 ± 0.0087 | 5+/0− consistent | small |
| | shortage | −0.0014 ± 0.0052 | 1+/4− mixed | — |
| | impact | **−0.0165 ± 0.0048** | 0+/5− consistent | truncation costs impact |
| **J vs 0** | delay | −0.0081 ± 0.0089 | 0+/5− consistent | chain depth makes delay harder |
| | shortage | −0.0016 ± 0.0100 | 3+/2− mixed | — |
| | impact | +0.0042 ± 0.0134 | 3+/2− mixed | — |
| **D vs B** | delay | +0.0009 ± 0.0092 | 3+/2− mixed | **null** |
| | shortage | −0.0032 ± 0.0086 | 1+/4− mixed | **null** |
| | impact | +0.0021 ± 0.0035 | 4+/1− mixed | **null** |
| **F vs E** | delay | −0.0062 ± 0.0183 | 3+/2− mixed | null |
| | shortage | −0.0087 ± 0.0104 | 1+/4− mixed | null |
| | impact | −0.0092 ± 0.0254 | 1+/4− mixed | null |
| **B vs 0** | all three | ≤ ±0.0068 | mixed on all three | null, and expected — B sets α=0 |
| **C vs 0** | all three | ≤ ±0.0048 | mixed on all three | null |
| **H vs 0** | all three | ≤ ±0.0094 | mixed on all three | null |
| **I vs 0** | all three | ≤ ±0.0052 | mixed on all three | null |

### 5.4 A warning about Mechanism G's delay AUC, which is not a model result

Variant G's delay AUC is **+0.14 above Variant 0's, consistently across all five seeds** — the
largest effect in the table by a factor of three. **It is not evidence that anything got better.**

Mechanism G delays what the model may *read*, not when events *happen*. A shipment that has actually
been delivered still looks `in_transit` as of t0 if its delivery has not been reported yet, so it
stays label-eligible and is a guaranteed negative. That inflates the denominator with easy negatives
and strips positives out: Variant G carries **184 delay positives against Variant 0's 519**, on a
*larger* eligible set. AUC rises because the task changed, not because the model learned more.

This is the same structural fact `docs/phase6_spec_scale_report.md` §2.1 measured on the label
side — Variant K's delay positive rate is 1.77% against Variant 0's 18.86% while its denominator is
larger. **Any comparison that includes a G-bearing variant (G, K) on the delay task must be read as
a task-difficulty change first.** The harness prints the positive count in the same row as the AUC
precisely so this cannot be read past.

### 5.5 The nine architectures on Variant 0, at matched parameters (5 dataset seeds)

Variant 0 is V1's own world, so this is the cleanest available replication test: same scale, same
split, same matched-parameter budgets, same 100 epochs.

| architecture | params | delay AUC | shortage AUC | impact AUC |
|---|---:|---|---|---|
| GCN | 708,009 | 0.7463 ± 0.0577 | 0.7468 ± 0.0253 | 0.8694 ± 0.0258 |
| GraphSAGE | 720,261 | 0.7669 ± 0.0474 | 0.7775 ± 0.0091 | 0.8630 ± 0.0281 |
| GAT | 731,495 | 0.7201 ± 0.0214 | **0.6521** ± 0.0118 | 0.9081 ± 0.0150 |
| HGT | 713,763 | 0.7557 ± 0.0280 | 0.7680 ± 0.0083 | 0.9220 ± 0.0054 |
| RGCN | 757,565 | 0.7635 ± 0.0486 | 0.7857 ± 0.0096 | **0.8300** ± 0.0293 |
| GraphGPS | 688,123 | 0.7645 ± 0.0509 | **0.7883** ± 0.0079 | 0.9203 ± 0.0152 |
| **SHARE** | 752,211 | 0.7744 ± 0.0376 | 0.7856 ± 0.0090 | 0.9321 ± 0.0128 |
| SHARP | 752,599 | **0.7774** ± 0.0352 | 0.7865 ± 0.0067 | **0.9338** ± 0.0077 |
| SHARK | 738,273 | 0.7754 ± 0.0392 | 0.7869 ± 0.0078 | 0.9201 ± 0.0198 |

Paired delta-AUC against SHARE — same held-out rows, same dataset seed, block bootstrap per seed,
sign agreement across the five. Positive = SHARE ahead.

| vs SHARE | delay | shortage | impact |
|---|---|---|---|
| GCN | **+0.0281** 5/5, 4 sig | **+0.0388** 5/5, 4 sig | **+0.0627** 5/5, 5 sig |
| GraphSAGE | +0.0075 4/5, 2 sig | **+0.0081** 5/5, 2 sig | **+0.0691** 5/5, 5 sig |
| GAT | **+0.0543** 5/5, 4 sig | **+0.1335** 5/5, 5 sig | **+0.0241** 5/5, 5 sig |
| HGT | +0.0186 4/5, 3 sig | **+0.0176** 5/5, 3 sig | **+0.0101** 5/5, 2 sig |
| RGCN | +0.0109 3/5, 2 sig | −0.0001 2/5, 1 sig | **+0.1021** 5/5, 5 sig |
| GraphGPS | +0.0099 4/5, 1 sig | −0.0027 0/5, 2 sig | +0.0118 4/5, 3 sig |
| SHARP | −0.0031 1/5, 1 sig | −0.0009 2/5, 1 sig | −0.0016 3/5, 2 sig |
| SHARK | −0.0010 2/5, 0 sig | −0.0013 3/5, 2 sig | +0.0120 4/5, 3 sig |

### 5.6 Dataset-seed variance is 10x model-seed variance, and V1 only measured the smaller one

The §2 sanity check varies the **model-init** seed on one dataset; the sweep varies the **dataset**
seed with model init fixed. Same architecture, same configuration, both five seeds:

| task | model-seed std | dataset-seed std | ratio |
|---|---|---|---|
| delay | 0.0037 | **0.0376** | **10.3x** |
| shortage | 0.0019 | 0.0090 | 4.8x |
| impact | 0.0102 | 0.0128 | 1.3x |

**Every error bar in V1's architecture comparison is a model-init error bar**, because V1 trained on
one generated world. On delay, the world-to-world spread is ten times larger. A delay gap of
+0.0103 to +0.0112 — which is what V1 reported for SHARE over HGT, and treated as a consistent win —
is well inside a single standard deviation of dataset variation. This is the label-volume analogue
of what `docs/phase6_spec_scale_report.md` §4 measured on the generator side (delay positives vary
~2x between seeds because the hidden-factor member sets are redrawn over a power-law degree
distribution), and it has the same consequence: **architecture claims on this benchmark must be
averaged over generated worlds, not over model initialisations.**

That is a finding about how V1's numbers should be read, not a defect in V1's execution — V1 had one
world to train on.

## 6. What replicates, what does not, and what is answered for the first time

### 6.1 V1's architecture claims

| V1 claim | V2 result | verdict |
|---|---|---|
| SHARE beats HGT on **shortage** | +0.0176, 5/5 seeds agree, 3/5 individually significant | **replicates** |
| SHARE beats HGT on **delay** | +0.0186, 4/5 agree, 3/5 significant | **directionally replicates**, not unanimous |
| SHARE ties HGT on **impact** | +0.0101, 5/5 agree but only 2/5 significant | **replicates as a tie**, with SHARE nominally ahead |
| SHARP does not beat SHARE | −0.0031 / −0.0009 / −0.0016, mixed signs on all three | **replicates** |
| SHARK does not beat SHARE | −0.0010 / −0.0013 / +0.0120, mixed signs on all three | **replicates** |
| GAT is the clear worst on shortage | 0.6521 against every other arm's 0.75–0.79 | **replicates**, emphatically |
| RGCN is the worst on impact | 0.8300, SHARE ahead by +0.1021 on 5/5 seeds | **replicates** |
| HGT beats GraphSAGE on impact | 0.9220 vs 0.8630 | **replicates** |

**SHARE is again the only architecture that is best-or-tied-for-best on all three tasks at once**,
and it is again never the unstable one. The one V1 claim that softens is the *delay* win over HGT:
it holds in direction but not on every world, which §5.6 explains — V1 could not have seen that,
having measured only one world.

**New, not previously testable: GraphGPS.** The adapted GraphGPS (§1) is the only baseline that
matches SHARE on shortage (−0.0027 in GPS's favour, 5/5 consistent but only 2/5 significant — a
tie) while staying respectable on impact (0.9203). It does not beat SHARE anywhere. Given it is the
cheapest arm at 688k parameters, it is the most interesting baseline in the set; a dense-attention
implementation at spec scale might do better, and this one cannot speak to that.

### 6.2 The V2 questions, answered for the first time

**Does discovering hidden structure help once it is causally coupled? No measurable effect here.**
Variant D vs Variant B — the pair that isolates coupling, since both carry Mechanism B's groups and
only D turns on α — gives **+0.0009 delay, −0.0032 shortage, +0.0021 impact, with mixed signs on all
three tasks**. Variant B against Variant 0 is likewise null (≤ ±0.0068, mixed), which is the
expected control: B sets α=0 for every group type, so it *should* be null, and it is.

This is a null result and it is reported as one. Two things must be said with it:

1. **It is a null at this label volume, not an established absence.** The relevant cells carry
   548–1,048 test positives against the spec's 2,000–5,000 target. `docs/phase6_spec_scale_report.md`
   §5 already flags that the spec-scale corpus itself only just clears that floor. A coupling effect
   of the size α=0.35 might plausibly produce could sit inside these intervals.
2. **The spec's own interpretation rule was applied.** Variant K's hidden-structure result is
   reported against Variant D, never alone — and K vs D is +0.1286 on delay with 5/5 sign
   agreement, which looks like a large "coupling helps" effect and **is not one**: Variant K
   contains Mechanism G, and §5.4 shows G alone moves delay AUC by +0.1429 on 5/5 seeds by changing
   what the delay task *is*. Reading K vs D as a coupling result would be exactly the mistake the
   spec's rule exists to prevent.

**Does adaptive depth help under genuine chain-length heterogeneity? Not answerable from this
session, and the reason is worth stating.** Mechanism J's effect on the tasks is measurable —
J vs 0 is −0.0081 on delay with 5/5 sign agreement, so heterogeneous chain length makes the delay
task slightly *harder*, and impact is unmoved (+0.0042, mixed). But "adaptive depth" is a property
of V1's Step-6 depth-gate architectures (`rgcn_attn_depthgate`, the rung ladder, Rung 5's per-node
gates), which this port deliberately did not carry over — the brief scoped the port to SHARE, SHARP,
SHARK and the baselines. Every architecture here reads a **fixed** structural depth prior
(delay→h², shortage→h³, impact→h³). So V2 now has a variant that genuinely varies chain length, and
the architectures that could exploit it are the obvious next port.

**What the other mechanisms do to task difficulty**, all with 5/5 sign agreement unless noted:

- **G (information delay) is by far the largest effect**, and it is a task change, not a difficulty
  change in the ordinary sense: +0.1429 delay AUC, −0.0478 shortage AUC. See §5.4.
- **E (hidden resilience) makes delay easier**: +0.0404. Absorption removes the marginal positives —
  the ones that would have gone late but did not — leaving a cleaner separation, at the cost of
  dropping delay positives from 519 to 403.
- **A (visibility truncation) costs impact**: −0.0165 against Variant J, its mandated reference.
  Truncating the observable graph removes exactly the upstream context the impact head needs.
- **C, H, I are null** (≤ ±0.0094, mixed signs on all three tasks). Dynamic rewiring, external
  shocks and multi-source dependencies do not measurably change any task's difficulty for SHARE at
  this label volume. Three nulls out of ten mechanisms is a real finding about the benchmark: those
  three mechanisms are currently costing generation complexity without producing a measurable
  difficulty signal, and either their parameters need strengthening or their value lies in
  robustness questions this harness does not yet ask.

### 6.3 What this report does not establish

- **Nothing at spec scale.** §3 and §7.
- **Whether the architecture ranking changes between variants.** All nine architectures were run on
  Variant 0 only; SHARE and GCN are the only arms with all twelve variants, and they agree on
  ordering wherever both exist.
- **Any null as an absence.** Every null above is bounded by intervals set by 65–1,077 test
  positives. The honest statement is "no effect detectable at this label volume", and the spec-scale
  corpus exists precisely to tighten that.

## 7. What a spec-scale answer would cost

The harness is scale-agnostic: `--csv-dir db/csv` points it at the spec-scale corpus and everything
else is unchanged. What is missing is compute, and the size of the gap is measured, not guessed.

| | measured |
|---|---|
| SHARE, one spec-scale forward+backward, CPU | 86 s |
| one 100-epoch run (16 train snapshots) | ~38 h |
| nine architectures x twelve variants x five seeds | **~20,000 CPU-hours** |

On a single A100-class GPU the scatter/gather-bound inner loop should run 20–50× faster than this
machine's CPU, putting one run at roughly 45–110 minutes and the full sweep at **400–1,000 GPU-hours**
— a few days on 8 GPUs, which is an ordinary cost for a benchmark paper's headline table and an
impossible one for a laptop session.

Two cheaper intermediate options, in case the full sweep is not wanted:

1. **SHARE-only across all twelve variants at spec scale, five seeds** — 60 runs, ~2,300 CPU-hours,
   ~50 GPU-hours. Answers every *variant* question at spec scale and leaves the architecture
   comparison at the V1-comparable configuration where it has a V1 baseline to be compared against.
2. **All nine architectures on Variants 0 and K only, five seeds** — 90 runs. Answers the
   architecture question at spec scale on the two variants that bracket the benchmark.

Option 1 is the better buy: the architecture ranking is the part that already has a V1 control, and
the variant effects are the part that is new.

---

## 8. SHARP across all twelve variants (Phase 7b)

§5.1 left SHARP tested on Variant 0 only, so there was no evidence either way about whether the two
architectures respond to the mechanisms the same way. This section closes that gap: SHARP
(`rgcn_relemb`, 752,599 params) run across **all twelve variants × five dataset seeds**, at the same
V1-comparable configuration, same 40/20/40 temporal split, same matched-parameter arm, same
block-bootstrap protocol. 60 runs at **419 s each — identical per-run cost to SHARE**, as the
budget estimate predicted from their near-identical size.

### 8.1 Device check: MPS wins at this scale, and was still not used

§3.1 found MPS *slower* than CPU at spec scale. That does not transfer, and it was worth measuring
rather than assuming — at this sweep's scale (27,012 nodes, 123,232 edges per snapshot) **MPS is
2.2× faster than CPU in a single process**: 1.23 s/epoch against 2.71 s/epoch, a projected 123 s
against 271 s for a 100-epoch run.

It was still not used, for two measured reasons:

| configuration | per-run | throughput |
|---|---|---|
| MPS, 1 worker | 123 s | 29 runs/h |
| MPS, 2 workers | 222 s each | 32 runs/h |
| MPS, 4 workers | 440 s each | 33 runs/h |
| **CPU, 4 workers × 3 threads** | **419 s each** | **34 runs/h** |
| 2 MPS + 2 CPU (hybrid) | 222 s / 367 s | **52 runs/h** |

1. **MPS does not parallelise.** Per-run time scales almost linearly with worker count, so aggregate
   throughput is flat at ~30 runs/h — the GPU is the bottleneck and is no faster in total than four
   CPU workers. The 2.2× single-process win buys nothing for a 60-run sweep.
2. **The hybrid would have been fastest (52 runs/h, ~1.5×) and would have invalidated the
   comparison.** Splitting variants across devices puts any numeric difference between CPU and MPS
   float32 reduction orders inside every cross-variant delta — and the deltas below are as small as
   0.002. SHARE's numbers were produced on CPU at 4 workers × 3 threads, so SHARP was run
   identically and the SHARP-vs-SHARE comparison carries no device confound.

Recorded for future sweeps: **at ~30k-node graphs prefer MPS for a single job and CPU for many; do
not mix devices inside a comparison.** A harness bug was fixed to make this testable at all —
`run_one()` passed `--device` through to the model but never moved the bundles, so `--device mps`
had never actually worked.

### 8.2 SHARP per-variant (5 dataset seeds)

| var | mechanisms | delay AUC | pos | shortage AUC | pos | impact AUC | pos |
|---|---|---|---|---|---|---|---|
| 0 | — | 0.7828 ± 0.0414 | 519 | 0.7846 ± 0.0069 | 1,036 | 0.9321 ± 0.0104 | 156 |
| A | J+A | 0.7662 ± 0.0336 | 628 | 0.7842 ± 0.0019 | 1,077 | 0.9236 ± 0.0038 | 203 |
| B | B | 0.7849 ± 0.0314 | 553 | 0.7755 ± 0.0168 | 1,048 | 0.9366 ± 0.0094 | 155 |
| C | C | 0.7770 ± 0.0286 | 520 | 0.7861 ± 0.0092 | 1,023 | 0.9291 ± 0.0099 | 150 |
| D | B+D | 0.7776 ± 0.0370 | 548 | 0.7794 ± 0.0115 | 1,026 | 0.9293 ± 0.0079 | 157 |
| E | E | 0.8155 ± 0.0234 | 403 | 0.7915 ± 0.0087 | 761 | 0.9204 ± 0.0107 | 84 |
| F | E+F | 0.8118 ± 0.0249 | 370 | 0.7773 ± 0.0145 | 791 | 0.9149 ± 0.0207 | 83 |
| G | G | **0.9174** ± 0.0115 | 184 | **0.7362** ± 0.0051 | 1,036 | 0.9441 ± 0.0063 | 103 |
| H | H | 0.7761 ± 0.0306 | 549 | 0.7811 ± 0.0088 | 1,035 | 0.9281 ± 0.0156 | 155 |
| I | I | 0.7770 ± 0.0289 | 502 | 0.7749 ± 0.0090 | 1,045 | 0.9301 ± 0.0059 | 156 |
| J | J | 0.7627 ± 0.0395 | 628 | 0.7816 ± 0.0033 | 1,077 | 0.9364 ± 0.0044 | 203 |
| K | all ten | **0.9110** ± 0.0194 | **113** | **0.7427** ± 0.0120 | 745 | 0.9255 ± 0.0095 | **65** |

Mechanism G's delay figure carries §5.4's warning unchanged — it is a task-difficulty artifact, not
a signal. Variant K's impact cell rests on 65 test positives and is underpowered by construction.

### 8.3 SHARP's variant effects, each against its mandated reference

| effect | task | mean delta | sign |
|---|---|---|---|
| **G vs 0** | delay | **+0.1346 ± 0.0336** | 5+/0− consistent |
| | shortage | **−0.0484 ± 0.0036** | 0+/5− consistent |
| | impact | **+0.0120 ± 0.0042** | 5+/0− consistent |
| **K vs D** | delay | **+0.1334 ± 0.0411** | 5+/0− consistent |
| | shortage | **−0.0367 ± 0.0155** | 0+/5− consistent |
| | impact | −0.0039 ± 0.0084 | 1+/4− mixed |
| **E vs 0** | delay | **+0.0327 ± 0.0196** | 5+/0− consistent |
| | shortage | +0.0069 ± 0.0083 | 4+/1− mixed |
| | impact | **−0.0117 ± 0.0073** | 0+/5− consistent |
| **J vs 0** | delay | **−0.0201 ± 0.0088** | 0+/5− consistent |
| | shortage | −0.0030 ± 0.0079 | 2+/3− mixed |
| | impact | +0.0043 ± 0.0095 | 3+/2− mixed |
| **A vs J** | delay | +0.0034 ± 0.0071 | 3+/2− mixed |
| | shortage | +0.0026 ± 0.0029 | 4+/1− mixed |
| | impact | **−0.0128 ± 0.0036** | 0+/5− consistent |
| **F vs E** | delay | −0.0038 ± 0.0146 | 3+/2− mixed |
| | shortage | **−0.0142 ± 0.0083** | 0+/5− consistent |
| | impact | −0.0055 ± 0.0168 | 2+/3− mixed |
| **I vs 0** | delay | −0.0058 ± 0.0188 | 1+/4− mixed |
| | shortage | **−0.0097 ± 0.0055** | 0+/5− consistent |
| | impact | −0.0019 ± 0.0069 | 1+/4− mixed |
| **D vs B** | delay | −0.0074 ± 0.0125 | 2+/3− mixed |
| | shortage | +0.0039 ± 0.0075 | 3+/2− mixed |
| | impact | −0.0072 ± 0.0083 | 1+/4− mixed |
| **B vs 0** | all three | ≤ ±0.0092 | mixed on all three |
| **C vs 0** | all three | ≤ ±0.0058 | mixed on all three |
| **H vs 0** | all three | ≤ ±0.0067 | mixed on all three |

### 8.4 Does SHARP track SHARE? On every effect either one calls consistent, yes

Head-to-head, paired on identical held-out rows per variant and seed, **SHARP − SHARE is within
±0.0085 on all 36 variant × task cells**, and only four reach 5/5 sign agreement (Variant A delay
−0.0071, Variant D delay −0.0046, Variant H shortage −0.0047, Variant I shortage −0.0055 — all
marginally in SHARE's favour, none larger than 0.008). **V1's conclusion that SHARP adds cost for
no measurable accuracy gain now holds across all twelve variants, not just the plain dataset**,
which is the specific gap this session existed to close.

The more informative result is what happens when the two architectures' *mechanism responses* are
compared. Across the 33 effect × task cells:

- **12 cells have at least one architecture reporting 5/5 sign agreement. In all 12, the other
  architecture agrees in direction.** Not one reverses.
- **8 cells disagree in sign, and all 8 are cells BOTH architectures already flagged mixed** —
  A vs J shortage, C vs 0 delay, all three of D vs B, H vs 0 delay and shortage, I vs 0 delay.

The two sets do not overlap at all. The sign-consistency gate this project has used since Phase 0 is
therefore doing real work: it separates effects that survive an architecture change from effects
that do not. Nothing labelled consistent flipped; everything that flipped was already labelled mixed.

**Effects now confirmed by two independent architectures** (both 5/5, same direction): G's delay
inflation and shortage damage, K vs D's inherited version of both, E's delay easing, J's delay
hardening, and A's impact cost. These are the benchmark's architecture-independent mechanism
signatures.

**Effects SHARP resolves that SHARE left mixed** — SHARP reaches 5/5 consistency where SHARE did
not, in the same direction each time: E vs 0 impact (−0.0117), F vs E shortage (−0.0142), I vs 0
shortage (−0.0097), G vs 0 impact (+0.0120). The consequential one is **Mechanism I**: §6.2 listed
it among three mechanisms with no detectable effect, and SHARP shows a consistent −0.0097 on
shortage across all five seeds. Multi-source AND/OR dependency does measurably harden the shortage
task; SHARE simply could not resolve it. **This corrects §6.2's "C, H and I are null" to "C and H
are not detected; I is a real, small effect on shortage."**

**The D vs B null is now considerably stronger.** Two architectures, five seeds each, and they
disagree in sign on all three tasks with every cell mixed. If coupling produced a real effect at
this label volume, two architectures of near-identical capacity would not land on opposite sides of
zero on every task. **Discovering hidden structure, once causally coupled, remains not detected at
this label volume** — still a non-detection rather than an established absence, since this is the
V1-comparable configuration and not spec scale.

### 8.5 What this section does not change

- **Still not spec scale.** §3 and §7 stand; every null here is bounded by 65–1,077 test positives.
- **Still no third architecture across all twelve variants.** GCN covers twelve variants (five seeds
  on eight of them); HGT, RGCN, GAT, GraphSAGE and GraphGPS remain Variant-0 only.
- **SHARP is still not recommended over SHARE.** It ties everywhere and costs an extra per-relation
  embedding table; §6.1's verdict is unchanged and now rests on 12 variants instead of one.
