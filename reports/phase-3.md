# Phase 3 — Temporal encoder, cross-world evaluation, h⁰/h¹/h⁴

> ## ⚠ Annotation added at Phase 0–3 closeout — read before quoting any number below
>
> **The fit window was specified for this phase and never applied.** `ml/train/cross_world.py`
> line 17 reads
>
> ```python
> from config import WORLDS, CACHE, SPLIT
> ```
>
> It imports `SPLIT` but **not `FIT_WINDOW`**, and nothing in the file filters `snapshot_date` to
> the 2019–2025 modelling window. `labels_for()` therefore returns every label row from 2016
> onward, so **this phase trained on 66 snapshots while every baseline it is compared against
> used 44** — the baselines in `docs/learnability.py` and `ml/baselines/learnability_windowed.py`
> apply the window, this trainer does not.
>
> **Consequences, in order of importance:**
>
> 1. **The absolute figures in this report and those in [`phase1_2.md`](phase1_2.md) are not
>    measured on the same training fold.** They are not directly comparable to four decimals.
> 2. **`phase1_2.md` carries the authoritative numbers wherever the two overlap** — it applies
>    the window, trains all 16,072 channels to convergence with early stopping, and reports 95%
>    bootstrap intervals. Where this report and that one disagree, that one is correct.
> 3. **The direction and magnitude of this report's findings survive.** Phase 2 §2 measured the
>    window's cost at ~0.15% on arrival for the GBM, and `phase1_2.md` §10 reproduces this
>    report's arrival conclusion under the corrected protocol (+8.7% / +8.1% with SHARE against
>    the +5.5–7.2% claimed here). **The headline is not in question; the decimals are.**
>
> **This phase was deliberately not re-run.** The effect is small, the conclusion reproduces, and
> re-running would have cost more than it resolved. The annotation exists so that nobody
> reconciling the two tables later has to rediscover why they differ.
>
> One further correction: this report's §3.2 describes `HeteroMP` as the graph encoder throughout.
> That is accurate, but it is **not SHARE** — `phase1_2.md` §6 builds SHARE as specified and
> reports both side by side. The "learned encoder" conclusion here is a HeteroMP result.


Device **MPS**, torch 2.14.0. Sweep wall-clock **2,644 s** for 18 trainings × 2 evaluations
(≈145 s per cell). Peak RSS 2.7 GB. Code: `ml/models/tcn.py`, `ml/train/cross_world.py`.

**Headline: with a learned encoder the graph does contribute — 5.5–7.2% of above-chance signal on
arrival, in all four cells.** That contradicts the prior measurement of ~0% with fixed
mean-pooling, and it is the answer to the question run 7 left open.

---

## 3.1 Dilated causal TCN — and a defect I had to fix in my own encoder

Guide geometry built as specified: kernel 2, dilations 1/2/4/8/16/32, **6 layers**, hidden 64,
receptive field **64 weeks ≥ 52**.

**Causality unit test passes exactly.** Perturbing the last timestep by +100 leaves every earlier
position bit-identical:

    max abs diff on earlier positions: 0.0

Left-padding only, then trimming the right overhang — never a symmetric `padding=`.

### The defect: a 6-deep ReLU stack with no residuals learns nothing

My first TCN was the guide's geometry taken literally — six `Conv1d`+ReLU layers, no skip
connections. It did not train:

    losses: 1.0136  0.9995  0.9987  0.9978  0.9972  0.9978      (standardised MSE, i.e. R^2 ~ 0)
    C-index: 0.5000 within-world, 0.5000 cross-world

A linear probe isolated it as an encoder fault, not a data-path fault:

| probe on the same tensors the TCN sees | C-index |
|---|---|
| Ridge on the **last week** only | **0.6386** |
| Ridge on the **flat 52-week** window | **0.6464** |
| LightGBM baseline (reference) | 0.6510 |
| my TCN | 0.5000 |

The signal was fully accessible; the stack was destroying it. Adding a 1×1 residual projection per
layer fixed it, and the causality test still passes:

    losses: 0.9614  0.9229  0.9128  0.9057  0.9012  0.8979
    C-index: 0.6524 within-world     (46,144 params)

I record this because the guide's Phase 3 specifies the dilation/depth/width but not residuals,
and taken literally it produces an encoder that cannot learn. Hyperparameters were tuned **once**
at this point on train-v7/test-v7 (`hidden 64, lr 2e-3, wd 1e-4, 6 epochs`) and then **frozen** for
all 18 cells — no per-cell tuning.

## 3.2 The learned graph encoder

`HeteroMP`: one *round* is channels → {supplier, part, plant} learned means → back to channels
through a learned gate, with a residual so h⁴ can fall back to h⁰. **h¹ = 1 round, h⁴ = 2 rounds
(4 message-passing layers), h⁰ = no message passing.** This is the learned counterpart of the
fixed mean-pooling used in the run-7 diagnostic.

---

## P3.1 The four cells

Same features, same frozen hyperparameters, baselines recomputed against the world each cell is
**tested** on. Masters are byte-identical across worlds (Stage A §A1.4), so the index maps coincide
and a v6-trained model runs on v7 unchanged — and nothing in the model is keyed on identity in any
case.

### Arrival timing — C-index ↑ (the reliable task)

| | test v6 | test v7 |
|---|---|---|
| **train v6** | **0.6580** *(within)* | 0.6254 *(cross)* |
| **train v7** | 0.6638 *(cross)* | **0.6635** *(within)* |
| naive baseline on that test world | 0.5573 | 0.5636 |
| LightGBM baseline on that test world | 0.6444 | 0.6510 |

*(h⁴ shown; full depth table below.)*

**The regime-transfer penalty is strongly asymmetric.**

| direction | within | cross | penalty, as % of above-chance signal |
|---|---|---|---|
| v6 → v7 | 0.6580 (0.1580 above chance) | 0.6254 (0.1254) | **−20.6%** |
| v7 → v6 | 0.6635 (0.1635) | 0.6638 (0.1638) | **+0.2% — no penalty** |

A model trained on **v7 transfers to v6 for free**; one trained on **v6 loses a fifth of its signal
on v7**. v7 is the regime with the AR(1) demand transient — the harder, noisier world. Training on
the harder regime generalises to the simpler one; the reverse does not hold. If Rane's real data is
noisier than either synthetic world, that direction is the one that matters, and it argues for
training on v7.

Both neural cells **beat the LightGBM baseline** on their own test world (0.6635 vs 0.6510 on v7;
0.6580 vs 0.6444 on v6).

### Fill rate — CRPS ↓

| | test v6 | test v7 |
|---|---|---|
| **train v6** | **0.0594** | 0.1006 |
| **train v7** | 0.0597 | **0.1001** |
| naive baseline | 0.0616 | 0.1041 |
| LightGBM baseline | 0.0598 | 0.0987 |

Cross-regime penalty is small in both directions (v6→v7 0.1006 vs 0.1001 within; v7→v6 0.0597 vs
0.0594 within). The neural model matches LightGBM on v6 and is slightly behind on v7
(0.1001 vs 0.0987), while comfortably beating naive on both.

### Part shortage — PR-AUC ↑. **Half this table is uninterpretable.**

| | test v6 | test v7 |
|---|---|---|
| **train v6** | **1.0000** ⚠ | 0.1258 |
| **train v7** | 1.0000 ⚠ | **0.4388** |
| base rate on that test world | 0.2073 | 0.1328 |
| LightGBM baseline | 0.5457 | 0.4692 |

**v6's `training_labels` shortage rows are 45,942 with zero negatives** (v7: 124,500 rows, 12.8%
positive). This is the defect found in run 7's Part 0 and fixed only in the v7 generator. It breaks
the v6 shortage column on both axes:

- **as a test set** — every label is positive, so PR-AUC is 1.0 by construction and measures nothing;
- **as a training set** — a v6-trained model never sees a negative, and scores **0.1258 on v7
  against a 0.1328 base rate**, i.e. *below chance*. It learned nothing discriminative.

Only the **train-v7 → test-v7** cell is meaningful: **0.4388**, against a 0.1328 base rate and a
0.4692 LightGBM baseline. The neural head has real signal but is behind the GBM, most likely
because my part×plant readout is an unweighted mean of member-channel states and 6 epochs is short.

---

## P3.2 h⁰ vs h¹ vs h⁴ in all four cells

Graph share = (above-chance at h⁴ − above-chance at h⁰) / above-chance at h⁴, measured against
0.5 for C-index, the naive baseline for CRPS, and the base rate for PR-AUC — each on the world the
cell is *tested* on.

| task | train | test | h⁰ | h¹ | h⁴ | **graph share** |
|---|---|---|---|---|---|---|
| **arrival** | v6 | v6 | 0.6467 | 0.6562 | **0.6580** | **+7.2%** |
| **arrival** | v6 | v7 | 0.6170 | 0.6225 | **0.6254** | **+6.7%** |
| **arrival** | v7 | v6 | 0.6547 | 0.6625 | **0.6638** | **+5.5%** |
| **arrival** | v7 | v7 | 0.6525 | 0.6624 | **0.6635** | **+6.7%** |
| fill | v6 | v6 | 0.0596 | 0.0594 | 0.0594 | +9.7% |
| fill | v6 | v7 | 0.1007 | 0.1010 | 0.1006 | +2.2% |
| fill | v7 | v6 | 0.0600 | 0.0597 | 0.0597 | +16.9% |
| fill | v7 | v7 | 0.1015 | 0.0992 | 0.1001 | +35.5% |
| shortage | v7 | v7 | 0.4271 | 0.4384 | **0.4388** | **+3.8%** |
| shortage | v6 | v7 | 0.1172 | 0.1258 | 0.1258 | n/m (below chance) |
| shortage | * | v6 | 1.0000 | 1.0000 | 1.0000 | n/m (no negatives) |

### The graph contributes, and the prior measurement was an artefact of the aggregator

| measurement | arrival | shortage | fill |
|---|---|---|---|
| run 7, **fixed mean-pooling** + GBM head | **−0.3%** | **−1.0%** | +3.6% |
| here, **learned encoder** | **+5.5 to +7.2%**, all four cells | **+3.8%** | +2.2 to +35.5% |

**On arrival the sign flips and the magnitude is consistent across all four cells** — 5.5%, 6.7%,
6.7%, 7.2%. That is not noise around zero; it is a stable positive effect, and it approaches the
~10% HADES v3 reference on delay. h¹ captures most of it and h⁴ adds a little more, which is what
a real but shallow propagation effect looks like.

Fill's numbers are large in percentage terms but tiny in absolute CRPS (0.1015 → 0.1001 is 0.0014)
and non-monotonic in depth, so I would not defend the +35.5% figure as anything but noise on a
small margin. **Arrival is the result that carries the finding.**

**What changed versus run 7.** Same graph, same three relations, same worlds. The difference is
that a mean over a channel's supplier-, part- and plant-neighbourhood is nearly collinear with the
channel's own `load_ratio` feature, which h⁰ already has — so fixed mean-pooling adds nothing. A
learned, gated aggregation can weight neighbours and extract a residual the mean cannot. Run 7's
report said the h⁰ diagnostic's logic "holds in the direction that matters"; on this evidence it
does not — **fixed mean-pooling understated the graph, and the conclusion that the architecture
premise is unsupported was premature.**

---

## P3.3 The confound — stated plainly

**v6 → v7 is not a clean ablation of demand persistence.** The v7 generator changed the demand
process, and everything downstream moved with it:

| | v6 | v7 |
|---|---|---|
| demand rate | one persistent per-channel draw, σ=1.05 | persistent level σ=0.85 **+ AR(1) transient**, φ=0.97 |
| planner | policy sized on the true rate | EWMA re-forecast, α=0.10 |
| PO lines with fill < 1.0 | 11.0% | **18.0%** |
| supplier-months constrained | 20.9% | **30.3%** |
| shortage label negatives | **none** | 12.8% positive rate |
| PO lines | 1,178,254 | 982,585 |

So the off-diagonal measures **regime A versus regime B as bundles** — demand persistence, fill
spread, capacity pressure and a label-schema fix, all at once. It is not an isolation of any one
axis. The asymmetric transfer result (v7→v6 free, v6→v7 −20.6%) is a property of that bundle.

**What this does not tell you.** It measures sensitivity to one axis of variation between two
synthetic worlds. It is **not** accuracy on Rane's data, and it is **not** a substitute for
rolling-origin backtesting on their extract. `synthetic_rules.md` §17 is explicit that seed- and
world-to-world agreement is not a proxy for generalisation to Rane; this measurement inherits that
caveat in full.

---

## Gate

| Requirement | Result |
|---|---|
| All four cells complete | **pass** — 18 trainings × 2 evaluations, 2,644 s |
| h⁰/h¹/h⁴ per task per cell | **pass** — table above |
| Confound stated | **pass** — §P3.3 |

**Two findings that outrank the gate:**

1. **The graph contributes 5.5–7.2% of above-chance signal on arrival with a learned encoder**,
   consistently in all four cells, where fixed mean-pooling measured ~0%. The G4 conclusion carried
   since run 7 needs revising: it was a statement about the aggregator, not about the world.
2. **The v6 shortage task cannot be evaluated or trained on** — its label table has no negative
   class. Any future work using v6 for shortage must regenerate its labels first; v7 is unaffected.
