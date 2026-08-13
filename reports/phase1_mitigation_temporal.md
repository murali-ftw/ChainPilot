# Phase 1 — Layer 3: Mitigation Level, Static and Temporal (HADES V3)

**Date:** 2026-08-14
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisites:** `reports/phase0_audit.md` (five checks pass); `reports/phase1_latent_state.md`
(Supply Stress and Recovery Capability cleared); `reports/phase1_mitigation_level.md` (Gate 0
inertness, Gate 0.5 depth sweep, and an observability control already run).

**Standing note on scope.** Recovery Capability (0.6530, Variant E) and Supply Stress (0.6426
Variant A / 0.6509 Variant E) cleared their gates in `phase1_latent_state.md` and feed the SCM
**regardless of what happens below**. A STOP in this file means "Mitigation does not get a temporal
channel this round", not "Layer 3 has nothing for Layer 4", and does not block Phase 2.

**New code this session:** `ml/verify_mitigation_recorder.py` (Gate 0 correctness),
`ml/mitigation_raw_signal_gate1.py` (Gate 1), `ml/mitigation_temporal_gates.py` (Gates 2–3).

---

## GATE 0 — Instrument Mitigation Level: inert *and* correct

### 0.1 The instrumentation

`MITIGATION_HISTORY[(sup_id, as_of)]` is written inside `mitigation_level()`
(`db/generate_dataset.py:1213-1226`), on both return paths, before `agent_observe()` can drain
`agent_pending`. It consumes no RNG draw, adds no branch, and writes nothing at all when
Mechanism E is off (the `if not RESILIENCE: return 0.0` guard precedes every write). Full diff and
rationale: `reports/phase1_mitigation_level.md` §Step 1.

### 0.2 Inertness — **PASS** (carried forward)

Baselines were captured from the unmodified generator *before* the edit; the `v1` anchor was first
shown to reproduce the on-disk `db/csv_v1scale/v0_seed42` exactly, so the baseline is trustworthy
rather than merely self-consistent. Before vs. after:

```
variant 0: IDENTICAL   (20/20 .csv.gz files byte-for-byte)
variant E: IDENTICAL   (20/20 .csv.gz files byte-for-byte)
variant K: IDENTICAL   (21/21 .csv.gz files byte-for-byte)
```

Variant 0 is the anchor the brief names, but **E and K are the checks that carry the weight** —
Variant 0 has Mechanism E off and never executes the instrumented path at all.

### 0.3 Determinism — **PASS** (carried forward)

```
$ ./venv/bin/python db/run_benchmark.py --verify-determinism --variant E --config v1
determinism: variant E, 20/20 .csv.gz files byte-identical across two runs
```

### 0.4 Correctness — **PASS** (new this session)

Inertness and correctness are different failure modes: a recorder that writes the wrong value, or
at the wrong key, would still produce byte-identical CSVs and still pass §0.2. This was not
previously tested.

`ml/verify_mitigation_recorder.py` runs the generator with an **in-memory source patch** that
independently captures the value actually consumed at each of the two use sites in the weekly
replenishment loop — the committed generator is not modified:

```
_mit     = max((mitigation_level(s_, week) for s_ in _cand), default=0.0)   :1258   (trigger)
_qty    *= 1.0 + 0.35 * mitigation_level(sup, week)                         :1269   (quantity)
```

The script refuses to run if either use site is not found exactly once, so a future generator edit
produces a loud failure rather than a vacuous pass.

```
$ ./venv/bin/python ml/verify_mitigation_recorder.py --variant E --seed 42 --config v1

recorder entries      : 55,200
use-site observations : 1,560,680  (1,544,128 trigger, 16,552 qty)
checked               : 1,560,680 use-site values against the recorder
missing from recorder : 0
value mismatches      : 0
non-vacuity check     : 1,047,734 of 1,560,680 use-site values are > 0 (67.1%)

GATE 0 CORRECTNESS: PASS — every value the simulation used matches the recorded value exactly
```

Manual trace, covering the computed branch and both use sites (exact equality, not tolerance):

```
site     supplier       week                  used by sim     recorded  exact
trigger  3dffa708-718   2024-01-15 08:00:00      0.287717     0.287717    yes
qty      3dffa708-718   2024-01-15 08:00:00      0.287717     0.287717    yes
```

The earliest observations are all `0.0` (the `seen[0] < 3` branch, no observed history yet), which
would make a trivially-passing trace — so non-zero values are traced separately, and a non-vacuity
check confirms 67.1% of observed values are non-zero. A recorder that captured nothing would
otherwise report zero mismatches and look like a pass.

**GATE 0: PASS** — inert, deterministic, and correct.

---

## GATE 0.5 — Depth sweep, no assumed depth

Full h⁰…h⁴ sweep on Variant E, 5 dataset seeds × 5 init seeds, using
`ml/latent_state_head.py`'s established target construction, median-split binarisation (threshold
from train only) and AUC — identical to `own_stress`/`RESILIENCE`, so results are comparable.

| depth | AUC | init floor | dataset floor | above .5 | clears | sign-consistent |
|---|---|---|---|---|---|---|
| h⁰ | 0.9838 | 0.0018 | 0.0118 | +0.4838 | YES | yes |
| **h¹** | **0.9855** | 0.0022 | 0.0119 | +0.4855 | **YES** | yes |
| h² | 0.9691 | 0.0191 | 0.0803 | +0.4691 | YES | yes |
| h³ | 0.9743 | 0.0103 | 0.0588 | +0.4743 | YES | yes |
| h⁴ *(markov)* | 0.9578 | 0.0082 | 0.1038 | +0.4578 | YES | yes |

**GATE 0.5: PASS — best depth h¹ at AUC 0.9855**, clears its floor and is sign-consistent across
all five seeds. **h¹ is the depth Gate 2's `Δh` uses.** It coincides with Recovery Capability's
best depth, but was determined by sweep, not inherited.

**The static baseline is near ceiling, and that governs how Gates 2–3 must be read.** At 0.9855
there is only 0.0145 of headroom to a perfect score, while the h¹ dataset-seed floor alone is
0.0119. Any temporal lift has to fit into that gap *and* exceed its own floor. This is a structural
constraint on Gates 2–3, and it was known before they ran.

---

## GATE 1 — Deterministic raw-signal check (no model, no SHARE)

`x(s,t) = 1 − on_time_rate_90d`, the supplier's own emitted rolling late rate
(`db/generate_dataset.py:1485`), already a model input feature (`ml/data/loader.py:309-315`). Read
as-of `t0` exactly as the loader reads it. Two exclusions make the test non-circular: `x` is **not**
SHARE's embedding, and **not** mitigation's own lag. `shipment_count_180d` is reported alongside
because the `seen[0] < 3 → 0.0` branch ties mitigation to observed shipment count.

```
$ ./venv/bin/python ml/mitigation_raw_signal_gate1.py --variant E --seeds 42,43,44,45,46 --config v1

  LEVELS  Pearson r(late_rate, mitigation)      : +0.5486
          Spearman rho                          : +0.5411
          Pearson r(shipment_count, mitigation) : +0.0693
          R^2, 2-feature linear fit             : 0.3049
  DELTAS  Pearson r(d late_rate, d mitigation)  : +0.5511
          Spearman rho                          : +0.5068
```

**Reported outcome: a trivial function of already-visible history explains a substantial minority
of mitigation — about 30% of its variance — but not most of it.**

This resolves an apparent tension with `reports/phase1_mitigation_level.md`, which found emitted
columns alone reach **AUC 0.9750** on mitigation. Both are correct, and the difference is the
target:

| target form | emitted-only performance |
|---|---|
| **binarised** (median split) — what every gate here scores | AUC **0.9750** |
| **continuous** value — what Gate 1 regresses | R² **0.3049**, r ≈ 0.55 |

**Mitigation's ordering around its median is nearly free from observables; its magnitude is not.**
Since every gate in this phase scores the binarised target, the operative number is 0.9750 — the
representation is competing for the last ~1% of an already-solved ranking problem. Any future work
wanting mitigation as a *magnitude* faces a genuinely harder and largely unsolved problem, and
would need a regression-trained head rather than these classifiers.

Notably `r(Δx, Δmitigation) = +0.5511` is essentially identical to the level correlation — the
observable delta tracks the latent delta about as well as levels track levels, which is a direct
warning that `Δh` in Gate 2 has little unique information to contribute.

**GATE 1: informational, reported. Does not block Gate 2** (per the brief). It frames Gates 2–3:
they are contesting a small residual on a near-saturated ranking task.

---

## GATE 2 — Deterministic representation delta (`Δh`)

```
$ ./venv/bin/python -u ml/mitigation_temporal_gates.py --variant E --seeds 42,43,44,45,46 \
      --config v1 --depth 1 --init-seeds 0,1,2,3,4 --out out/phase1c/gates23_E.json
```

`Δh(t) = h(t) − h(t−1)` at h¹, concatenated into the **same probe and the same evaluation**
Gate 0.5 used. Zero new trainable parameters.

**Ordered snapshots are assembled post hoc, not a native SHARE capability.** `ml/train.py` and
`ml/data/loader.py` forward every snapshot independently — there is no recurrence anywhere in the
existing pipeline. The sequence is built by collecting SHARE's independent per-`t0` outputs in
order; `ml/ds_backbone.py::load_world` asserts Supplier node ordering is stable across snapshots,
which is what makes that assembly valid.

**First-snapshot handling.** `Δh` is undefined at the first snapshot of a split (`split_bundles`
carves positionally). Those rows are **dropped**, and the number is reported per seed —
**600–641 rows dropped from train and the same from test per dataset seed**, i.e. one snapshot's
worth (~20% of each split's rows, since v1 splits are 6/3/6 snapshots).

**The static baseline is recomputed on the identical reduced row set.** Comparing a temporal model
on *n−k* rows against Gate 0.5's baseline on *n* rows would confound the feature with the sample.
The recomputed static baseline lands at 0.9855 — the same value Gate 0.5 reported on the full row
set, which is a useful consistency check.

| arm | AUC | init floor | dataset floor |
|---|---|---|---|
| static h¹ (recomputed on identical rows) | 0.9855 | 0.0024 | 0.0111 |
| **+ Δh** | **0.9867** | 0.0032 | 0.0125 |

| | value |
|---|---|
| lift | **+0.0012** |
| per-seed lifts | [0.0009, 0.0009, 0.0023, 0.0005, 0.0015] |
| paired floor (init 0.0035, between-seed 0.0018) | **0.0035** → **MISSES** |
| absolute-AUC floor (stricter) | 0.0125 → **MISSES** |
| sign-consistent across 5 seeds | **yes** (all positive) |

### **GATE 2: STOP — the lift does not clear its floor.**

Two floor definitions were computed, because the choice is not obvious for a paired comparison and
the answer should not depend on picking the convenient one:

- **Paired floor (0.0035)** — the variability of the *lift itself*: the init-seed spread of the
  per-init paired delta, and the between-seed spread of the lift. This is the appropriate test,
  since both arms share the same rows, the same world and the same frozen backbone, so the lift
  isolates the feature set.
- **Absolute-AUC floor (0.0125)** — the spread of the absolute AUC, which additionally charges the
  lift for between-world differences it does not contain. Stricter.

**The lift misses under both** (+0.0012 vs 0.0035 and vs 0.0125), so the verdict does not depend on
that methodological choice — the cleanest possible outcome for a STOP.

**Reported honestly: the effect is small, positive, and consistent — and still not evidence.** All
five per-seed lifts are positive, which is not nothing. But every one of them is smaller than the
noise in the quantity being measured, and this project's standard is that a delta under its own
floor is not a finding regardless of sign consistency. Gate 1 predicted this: `r(Δx, Δmitigation)`
was +0.5511, essentially identical to the level correlation, meaning the observable delta carries
almost no information the level did not already carry.

---

## GATE 3 — Learned temporal estimator (GRU)

**Formally, Gate 3 should not have run** — the brief gates it on Gate 2 clearing, and Gate 2
stopped. It was run anyway because it is cheap on cached backbones and because "the deterministic
delta failed, but would a learned model have succeeded?" is worth answering rather than leaving
open. **Its result does not change Gate 2's STOP.**

A single-layer GRU (hidden 32) over the ordered per-supplier snapshot embeddings, per-timestep
output, same binarisation, same masking as Gate 2 (step 0 masked out, matching the dropped rows
exactly). Discrete by design: snapshots are monthly and evenly spaced, so a continuous-time model
would add adjoint solvers and stability tuning to fit a shape unobservable at this resolution.

| arm | AUC | init floor | dataset floor |
|---|---|---|---|
| static h¹ | 0.9855 | 0.0024 | 0.0111 |
| + Δh (Gate 2) | 0.9867 | 0.0032 | 0.0125 |
| **GRU (Gate 3)** | **0.9922** | 0.0011 | 0.0075 |

| | value |
|---|---|
| lift over Δh | **+0.0054** |
| per-seed lifts | [0.0039, 0.0112, 0.0028, 0.0025, 0.0068] |
| paired floor (init 0.0035, between-seed 0.0087) | **0.0087** → **MISSES** |
| absolute-AUC floor | 0.0075 → **MISSES** |
| sign-consistent across 5 seeds | **yes** (all positive) |
| beats Gate 2's deterministic result | yes, numerically (+0.0054) |

### **GATE 3: STOP — beats the deterministic delta numerically, but not beyond its own noise.**

The GRU is the strongest arm in absolute terms (0.9922 vs 0.9867 vs 0.9855) and its lift over Δh is
positive on all five seeds. It still misses both floors. The between-seed spread of its lift
(0.0087) is larger than the lift itself (0.0054), driven by seed 43 (+0.0112) against seed 45
(+0.0025) — the learned model helps substantially in some worlds and barely in others, which is
exactly what an unreliable gain looks like.

**A correction made during this analysis, recorded because it changed a verdict.** The first
implementation of the paired floor contained an operator-precedence bug —
`[a] if cond else [b] + [c]` parses as `[a] if cond else ([b] + [c])`, silently dropping the
between-seed term whenever init spreads existed. That reported Gate 3's paired floor as 0.0035 and
therefore as **CLEARS**. With the floor computed correctly (0.0087) Gate 3 **MISSES**. The bug is
fixed in `ml/mitigation_temporal_gates.py`; both gate verdicts above are the corrected ones.

---

## Layer 3 composition — the input set Phase 2 receives

| state | status | form delivered to the SCM | evidence |
|---|---|---|---|
| **Supply Stress** | ✅ cleared (prior phase) | Layer 3 estimate, best depth **h²** (Variant A) / **h¹** (Variant E) | AUC 0.6426 / 0.6509, sign-consistent 5/5; **+0.0814 over an emitted-columns-only probe**, well above its 0.0256 floor |
| **Recovery Capability** | ✅ cleared (prior phase) | Layer 3 estimate, best depth **h¹**; **Mechanism-E variants only** | AUC 0.6530, sign-consistent 5/5; **+0.1227 over emitted-only**, above its 0.0310 floor |
| **Mitigation Level** | ⛔ **no temporal channel** (Gate 2 STOP) | **Static value supplied from emitted features, not from a Layer 3 head** | Gate 0.5 static 0.9855; Gate 2 Δh +0.0012 (floor 0.0035); Gate 3 GRU +0.0054 (floor 0.0087) |

### Why Mitigation is delivered as an emitted-feature value rather than an estimator output

Gate 0.5's static head reaches 0.9855, which is not in dispute. But
`reports/phase1_mitigation_level.md` showed a linear probe on **emitted columns alone, with no
encoder at all**, reaches **0.9750** on the same binarised target — the whole frozen backbone buys
+0.0105, below its own 0.0119 floor. Gate 1 confirms the mechanism: `r(late_rate, mitigation)`
= +0.5486 on levels, and mitigation's own definition
(`min(1, observed_late_rate × (0.5 + RESILIENCE))`) is dominated by a quantity emitted three ways.

So Mitigation is real, causally load-bearing in the generator, and available to Phase 2 at ~0.975
fidelity — just not as evidence that Layer 3 recovers hidden state, and not with a temporal channel.
Supplying it from emitted features is simpler, cheaper, requires no privileged read, and loses
essentially nothing.

**One caveat that limits how far this conclusion generalises.** Every gate here scores the
**binarised** (median-split) target. Gate 1 found the **continuous** magnitude is only ~30%
explained by observables (R² = 0.3049). If a later phase needs mitigation as a magnitude rather
than a ranking, none of these results transfer, and a regression-trained head — not these
classifiers — would be the right instrument.

### Verdict

**Layer 3 delivers two estimated latent states to Phase 2 — Supply Stress and Recovery Capability —
plus Mitigation Level as an observable-derived input. Mitigation receives no temporal channel this
round.**

Per this report's opening note, that STOP is scoped to Mitigation's temporal channel and **does not
block Phase 2**.

---

## Backbone integrity at the close of Phase 1

```
$ shasum -a 256 ml/models/{rgcn_attn_encoder,rgcn_attn_markov_encoder,heads}.py
IDENTICAL  models/rgcn_attn_encoder.py
IDENTICAL  models/rgcn_attn_markov_encoder.py
IDENTICAL  models/heads.py        (all matching Phase 0's recorded digests)
```

SHARE and the Layer-2 Markov readout were not retrained or modified. `assert_backbone_frozen()` ran
on every probe fit and never fired. `db/generate_dataset.py` carries only the Gate 0 mitigation
recorder, whose inertness (§0.2), determinism (§0.3) and correctness (§0.4) are all confirmed above.
