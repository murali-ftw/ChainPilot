# Model B — Simple Full-Representation Baseline (Layer 3 Fusion Pilot)

**Date:** 2026-08-14
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisite:** `reports/phase1_latent_state.md` — Model A, the baseline being tested against.
**Scope:** the cheap falsification test for H1, run *before* any attention/fusion architecture is
built. Model C (level attention + cross-level fusion) is **not** built in this session and is
explicitly gated on this report.

**New code, one file, no inherited file modified:** `ml/modelB_concat_head.py`.

---

## The question

- **H0** — a single task-specific SHARE depth plus a simple head (Model A) is sufficient for
  latent-state estimation.
- **H1** — information distributed across multiple SHARE depths improves it.

Model B is the cheapest possible test of H1: `concat(h⁰,h¹,h²,h³,h⁴) → MLP → state`. Literal
concatenation, no learned attention, no learned fusion. The reasoning for testing H1 this way
first is this project's own precedent — if concatenation-plus-MLP shows nothing, a more expressive
combination mechanism is very unlikely to.

---

## Step 1 — Dimensionality confirmation

Confirmed from the **actual frozen checkpoint's forward pass**, not from `CFG`'s `hidden=128`
claim, on both testbeds:

```
variant A: CFG={'hidden': 128, 'num_bases': 10}  n_depths=5
  h^0 Supplier (654, 128)   h^1 (654, 128)   h^2 (654, 128)   h^3 (654, 128)   h^4 (654, 128)
  widths=[128,128,128,128,128]  all_equal=True  concat_dim=640
  raw Supplier input x: (654, 14)

variant E: CFG={'hidden': 128, 'num_bases': 10}  n_depths=5
  h^0 Supplier (800, 128)   h^1 (800, 128)   h^2 (800, 128)   h^3 (800, 128)   h^4 (800, 128)
  widths=[128,128,128,128,128]  all_equal=True  concat_dim=640
  raw Supplier input x: (800, 14)
```

**All five depths are exactly 128-wide on both variants.** Concatenation is therefore **literal**:
640-dim, no padding, no per-level projection, **no deviation from Model B's definition to flag**.

Note for context: h⁰ is a 14 → 128 projection of the raw supplier input features, so it is 128-wide
by construction rather than by coincidence. `concat_embeddings()` additionally asserts that all
five depths emit the same `(supplier, t0)` rows in the same order before concatenating — misaligned
rows would pair one supplier's h⁰ with another's h³ and would still train and still report a
plausible AUC, so it is checked rather than assumed. It never fired.

---

## Step 2 — What was held fixed

Model A's methodology is **imported, not reimplemented** — `latent_targets`, `binarise`,
`align`, `depth_embeddings`, `train_head`, `LatentStateHead`, `assert_backbone_frozen` all come
from `ml/latent_state_head.py` unmodified. The **only** thing that differs between the two arms is
the width of the feature vector handed to the head (128 → 640):

| held identical | value |
|---|---|
| target construction | `latent_targets()` — same privileged generator lift |
| binarisation | median split, threshold from **train only** |
| split | 40/20/40 temporal, `ds_backbone.load_world` |
| metric | `hypothesis_ranker.roc_auc`, classification |
| head shape | `Linear(in,32) → ReLU → Dropout(0.1) → Linear(32,1)` |
| optimiser | Adam, lr 1e-3, weight_decay 1e-4, 120 epochs, `pos_weight` |
| seeds | dataset 42–46 × init 0–4 |
| scale | `v1` preset (800 suppliers, 15 snapshots) |

This stays a **classification** problem at the same binarisation — deliberately not the regression
framing the original architecture proposal suggested eventually. Changing the metric family at the
same time as the architecture would have made the Model A vs Model B delta uninterpretable. A
regression framing for magnitude estimation remains a separate, explicitly-flagged follow-up.

Independent confirmation that the targets and split really are identical: Model B's pooled test
counts match Phase 1's exactly — **11,075/20,538** (A, stress), **12,235/24,000** (E, stress),
**12,000/24,000** (E, resilience).

**Floors were re-measured fresh.** Model A's per-depth floors (0.0390 / 0.0256 / 0.0310) are floors
for a different quantity — a 128-dim head — and are **not** inherited anywhere below.

---

## Step 3 — Results

Model B, 5 dataset seeds × 5 init seeds, `v1` preset.

| variant | state | AUC | seed std | init floor | dset floor | **floor** | above .5 | clears floor | sign-consistent | pos/test |
|---|---|---|---|---|---|---|---|---|---|---|
| A | supply_stress | 0.6338 | 0.0058 | 0.0304 | 0.0133 | **0.0304** | +0.1338 | **YES** | yes | 11,075/20,538 |
| E | supply_stress | 0.6961 | 0.0233 | 0.0155 | 0.0592 | **0.0592** | +0.1961 | **YES** | yes | 12,235/24,000 |
| E | recovery_capability | 0.6686 | 0.0099 | 0.0145 | 0.0222 | **0.0222** | +0.1686 | **YES** | yes | 12,000/24,000 |

Per dataset seed:

```
A / supply_stress        [0.6367, 0.6396, 0.6263, 0.6373, 0.6291]
E / supply_stress        [0.6706, 0.7299, 0.6873, 0.7086, 0.6843]
E / recovery_capability  [0.6628, 0.6798, 0.6645, 0.6782, 0.6575]
```

### Model A vs Model B

| variant / state | A depth | A AUC | A std | A floor | B AUC | B std | B floor | margin (B−A) | margin > B's floor? |
|---|---|---|---|---|---|---|---|---|---|
| A / supply_stress | h² | 0.6426 | 0.0154 | 0.0390 | 0.6338 | **0.0058** | 0.0304 | **−0.0087** | **NO** |
| E / supply_stress | h¹ | 0.6509 | 0.0099 | 0.0256 | **0.6961** | **0.0233** | 0.0592 | **+0.0452** | **NO** (short by 0.0140) |
| E / recovery_capability | h¹ | 0.6530 | 0.0126 | 0.0310 | 0.6686 | 0.0099 | 0.0222 | **+0.0156** | **NO** (short by 0.0066) |

Model A's std is computed from Phase 1's own per-seed numbers in `out/phase1/heads_{A,E}.json`.

---

## Step 4 — The gate, per state per variant

### Variant A — Supply Stress: **STOP, clean miss**

1. Clears own fresh floor — **PASS** (+0.1338 vs 0.0304)
2. Sign-consistent across 5 seeds — **PASS**
3. Beats Model A by more than its own floor — **FAIL.** Margin is **−0.0087**: Model B is
   *worse* than Model A, before the floor is even consulted.
4. Stability — Model B is **more** stable (std 0.0154 → 0.0058, 0.37×).

Paired per-seed reading (supplementary, not the gate): Model B beats Model A on **1 of 5** seeds.

```
seed:      42        43        44        45        46
B − A:  −0.0176  −0.0185  −0.0116  −0.0060  +0.0099
```

This is an unambiguous miss. Concatenating all five depths does not merely fail to help here — it
costs 0.0087 AUC relative to reading h² alone. The likely reading is dilution: h² carries the
signal, and appending four other depths adds 512 mostly-uninformative dimensions that the head
must learn to ignore from a fixed training budget.

### Variant E — Supply Stress: **STOP, clean miss** *(the interesting one)*

1. Clears own fresh floor — **PASS** (+0.1961 vs 0.0592)
2. Sign-consistent across 5 seeds — **PASS**
3. Beats Model A by more than its own floor — **FAIL.** Margin **+0.0452** against a freshly
   measured floor of **0.0592**. Short by 0.0140.
4. Stability — **Model B is 2.34× LESS stable** (std 0.0099 → 0.0233). **Flagged.**

This is the SHARP/SHARK failure mode reproducing almost exactly, and the gate's fourth criterion
is what makes it visible. Model B posts the largest mean gain anywhere in this test (+0.0452, and
positive on **5 of 5** seeds):

```
seed:      42        43        44        45        46
B − A:  +0.0223  +0.0650  +0.0416  +0.0523  +0.0450
```

and it fails anyway — **because its own seed-to-seed spread grew faster than its mean did.** Model
B's dataset-seed floor (0.0592) is more than double Model A's (0.0256), driven by seed 43 posting
0.7299 against seed 42's 0.6706. The gain and the instability are not two separate findings; the
instability is *mechanically why* the gain does not clear. A mean-only comparison would have
reported this as a clear +0.0452 win and sent Model C into construction.

**This is the strongest signal in the test and it still does not pass, and it is not being passed.**
The paired reading (5/5 seeds positive) is genuinely favourable and is recorded here so it is not
lost. But the gate specified for this session is explicit — the margin must exceed Model B's own
floor, "not just larger than zero" — and switching to the paired criterion *after* seeing which
criterion would let the result through is exactly the goalpost-move this project's reporting
standard exists to prevent. Recorded as a clean miss. If a paired-difference gate is thought to be
the better instrument, that case should be made on its merits and pre-registered, then this arm
re-run against it — not retrofitted here.

### Variant E — Recovery Capability: **STOP, clean miss**

1. Clears own fresh floor — **PASS** (+0.1686 vs 0.0222)
2. Sign-consistent across 5 seeds — **PASS**
3. Beats Model A by more than its own floor — **FAIL.** Margin **+0.0156** against a floor of
   **0.0222**. Short by 0.0066.
4. Stability — Model B is slightly **more** stable (std 0.0126 → 0.0099, 0.78×).

Paired per-seed: Model B beats Model A on **4 of 5** seeds, and the one loss is on seed 42 —
the seed on which Model A posted its own best number.

```
seed:      42        43        44        45        46
B − A:  −0.0121  +0.0287  +0.0206  +0.0285  +0.0120
```

Not sign-consistent as a paired delta, and the mean margin does not clear the floor. Clean miss.

---

## Verdict

| variant | state | Model B verdict | Model C justified? |
|---|---|---|---|
| A | Supply Stress | **STOP — clean miss** (B is 0.0087 *worse* than A) | ❌ **no** |
| E | Supply Stress | **STOP — clean miss** (margin +0.0452 < floor 0.0592; B 2.34× less stable) | ❌ **no** |
| E | Recovery Capability | **STOP — clean miss** (margin +0.0156 < floor 0.0222) | ❌ **no** |

**Model C is not justified for any state on either variant, and must not be built.** No state
cleared all four gate criteria. Every state failed on the same criterion — criterion 3 — and no
state failed criteria 1 or 2, so this is not a case of a broken measurement: Model B works, is
sign-consistent, and clears chance comfortably everywhere. It simply does not beat Model A by more
than its own noise.

**H0 is not rejected.** On this evidence, a single task-specific SHARE depth plus a simple head
remains sufficient for latent-state estimation on this benchmark, and the case for spending
architecture on multi-depth fusion is not made.

---

## Honest qualifications

1. **This is a pilot, not a benchmark finding.** Everything above is at the `v1` preset (800
   suppliers, 15 snapshots), the same caveat every number in Phase 1 carries. Spec-scale `db/csv/`
   holds only seeds 42–43, so the five-seed minimum is not satisfiable there without regeneration.

2. **The E / Supply Stress near-miss is the one result worth revisiting**, and revisiting it means
   changing the *measurement*, not the gate. Its floor is inflated by dataset-seed variance
   (0.0592, from one high seed), which is the dominant variance source this project has already
   documented. More dataset seeds would tighten that floor directly and is the cheapest way to find
   out whether +0.0452 is real. That is a re-measurement of Model B, not a licence for Model C.

3. **Capacity is confounded with multi-depth information, and this test does not separate them.**
   Model B's head has 5× the input weights of Model A's (640 vs 128 in-features, ~20.5k vs ~4.1k
   first-layer parameters). Where Model B gained, some of that gain may be capacity rather than
   distributed-depth information. This does not affect any verdict — every verdict is STOP, and the
   confound could only ever have *flattered* Model B — but it would matter to any future arm, which
   should carry a parameter-matched single-depth control.

4. **Supplier Reliability was not tested.** It failed its Phase 1 gate independently on both
   variants (below chance, not sign-consistent, ~1.1% positive rate). Model B is a test of how to
   *read* a state the representation is known to carry, not a rescue for one it does not. Phase 1's
   recommendation there stands: change the sampling, not the head.

5. **Recovery Capability's static-target non-independence carries over unchanged.** Its 24,000 test
   rows are ~800 independent suppliers per seed replicated across 6 snapshots. Five-seed sign
   consistency carries the result, not the row count.

---

## Backbone integrity at the close

Ground rule: SHARE and the existing Layer-2 Markov readout are not retrained or modified.

```
$ git status --porcelain
?? ml/modelB_concat_head.py          <- the only change: one new file

$ git status --porcelain db/         # empty — the generator was executed, never edited

$ shasum -a 256 ml/models/rgcn_attn_encoder.py ml/models/rgcn_attn_markov_encoder.py \
                ml/models/heads.py ml/models/encoder.py ml/models/depth.py \
                ml/ds_backbone.py ml/latent_state_head.py
1b85a68f9e9705be...  ml/models/rgcn_attn_encoder.py
9e7919e6bfcf39ca...  ml/models/rgcn_attn_markov_encoder.py
cbd40aadc8570745...  ml/models/heads.py
8f669a1683117280...  ml/models/encoder.py
8fa14bc270946507...  ml/models/depth.py
689962edd1315599...  ml/ds_backbone.py
8e4a0803d8f09cd0...  ml/latent_state_head.py

$ diff -q ml/models/{rgcn_attn_markov_encoder,rgcn_attn_encoder,heads}.py \
          /Users/muralik/Documents/Programs/HADES_v2/ml/models/
IDENTICAL (all three, no output)
```

`ml/latent_state_head.py` is **unmodified** — Model A's methodology was imported, never edited, so
the Phase 1 numbers this report compares against remain reproducible from the same code that
produced them.

All ten backbone checkpoints in `out/ds_ckpt/` carry mtimes of 2026-08-13 22:03–22:44, predating
this session (2026-08-14 08:25+). Every fit below loaded from cache; **no backbone was retrained**.
Backbone health on the first-loaded checkpoint matched Phase 1's recorded values exactly
(`delay=0.8046 shortage=0.7835 impact=0.9340`, variant A seed 42).

`assert_backbone_frozen()` ran after head construction and again after training on every one of
the **75 head fits** (A: 1 state × 5 dataset × 5 init = 25; E: 2 states × 5 × 5 = 50) and never
fired.

---

## Reproduction

```
$ venv/bin/python -u ml/modelB_concat_head.py --variant A --seeds 42,43,44,45,46 \
      --config v1 --init-seeds 0,1,2,3,4 --out out/modelB/concat_A.json
$ venv/bin/python -u ml/modelB_concat_head.py --variant E --seeds 42,43,44,45,46 \
      --config v1 --init-seeds 0,1,2,3,4 --out out/modelB/concat_E.json
```

Raw results: `out/modelB/concat_A.json`, `out/modelB/concat_E.json`.
Model A comparison numbers: `out/phase1/heads_A.json`, `out/phase1/heads_E.json`.

---

## Statement for the next session

**Model C is gated and the gate did not open.** Do not build level attention or cross-level fusion
for Supply Stress or Recovery Capability on either variant. The single defensible follow-up this
report supports is a **re-measurement** of Model B on Variant E / Supply Stress with more dataset
seeds, to establish whether its +0.0452 mean gain survives a tightened floor — with the explicit
warning that the same arm is currently **2.34× less stable seed-to-seed** than Model A, which is
the failure mode this project has recorded before and the reason that gain does not currently count.
