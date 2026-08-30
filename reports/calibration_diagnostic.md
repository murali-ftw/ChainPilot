# Calibration diagnostic — `delay`, `shortage`, `impact`

**Measurement only. Nothing was retrained and no correction was fitted.** Checkpoints are loaded
from `out/ds_ckpt/`, `freeze()` asserts every parameter is frozen, and the forward pass is
`ml/evaluate.py::collect_predictions` called exactly as every other evaluation calls it. The
output of this run is a decision about *whether* a correction is warranted, per task.

**Why this is not an AUC question.** AUC is invariant to any monotone transform of the scores, so
a model that says 0.999 when it means 0.60 scores identically to one that says 0.60. Every number
this project has recorded for the three tasks is an AUC, so this axis has never been measured for
`shortage` or `impact` at all.

| task | primary arm | why |
|---|---|---|
| `delay` | old V2 graph | the arm its reference number (+0.0357 vs floor 0.0874) comes from |
| `shortage` | bucketed-Carrier ("fixed") | the arm on which it currently passes ([reports/new_nodes_fix1.md](reports/new_nodes_fix1.md)) |
| `impact` | bucketed-Carrier ("fixed") | same |

Both arms produce all three tasks from one forward pass, so the off-arm numbers are reported free,
as a cross-check rather than a headline. 5 dataset seeds × 5 model-init seeds = 25 cells per arm.

Code: [ml/calibration_diagnostic.py](ml/calibration_diagnostic.py) · results:
[out/layer3_v3/calibration_diagnostic.json](out/layer3_v3/calibration_diagnostic.json)

---

## 0. First finding: the premise does not hold for these models

The task framing cites 96.5 % of `delay` predictions saturated at p ≥ 0.999. **That is not what
these checkpoints do.** Measured over all 25 cells:

| task | fraction at p ≥ 0.999 | fraction in the 90–100 % bin |
|---|---|---|
| delay | **0.0000 %** | 0.0 % (39 of 104,330) |
| shortage | 0.0000 % | 0.0 % (0 of 481,080) |
| impact | 0.0017 % | 0.1 % |

This is not a disagreement with the repository's own record — it is corroborated by it. The frozen
Phase 0 run recorded its own saturation census independently
(`out/layer3_v3/phase0_baseline.json`, written by `ml/layer3_baseline.py::_saturation`), and it
says the same thing for every world:

| world | delay p ≥ 0.999 | delay "explainable" band | delay median p |
|---|---|---|---|
| 42 | 0.00 % | 93.5 % | 0.453 |
| 43 | 0.00 % | 91.9 % | 0.488 |
| 44 | 0.00 % | 83.3 % | 0.469 |
| 45 | 0.00 % | 98.9 % | 0.462 |
| 46 | 0.00 % | 99.9 % | 0.463 |

`reports/decision_support_build.md` is **not present in this repository** — it is cited from module
docstrings but the file does not exist here — so the 96.5 % figure cannot be checked against its
own context. Whatever it describes (a different architecture, scale, or an earlier V1 model), it
does not describe the checkpoints under test. `delay`'s median prediction is ≈ 0.46, not ≈ 1.0.

**This matters for what follows**: `delay` does have a serious calibration problem, but it is not
the one the framing anticipated. It is not saturation at the ceiling; it is a uniform upward bias
across the whole probability range. And `shortage` and `impact` turn out to have the same
disease — which could not have been assumed in either direction, and is why it was measured.

---

## 1. Headline (brief item 4)

Metrics computed **within each cell** and then averaged, so each model's honesty is measured on
its own predictions rather than on a 25-model mixture. `*` marks the primary arm.

| task | arm | AUC | ECE | Brier | base rate | mean predicted | gap | top bin | bottom bin |
|---|---|---|---|---|---|---|---|---|---|
| **delay** | old `*` | 0.7774 | **0.2461** | 0.1735 | 0.1240 | 0.3694 | **+0.2454** | 0.0 % | 23.1 % |
| | carrier_lane | 0.7438 | 0.2673 | 0.1828 | 0.1240 | 0.3909 | +0.2668 | 0.0 % | 21.5 % |
| **shortage** | old | 0.7871 | 0.4347 | 0.2394 | 0.0538 | 0.4885 | +0.4347 | 0.0 % | 0.0 % |
| | carrier_lane `*` | 0.9069 | **0.3074** | 0.1491 | 0.0538 | 0.3612 | **+0.3074** | 0.0 % | 0.9 % |
| **impact** | old | 0.9339 | 0.1199 | 0.0769 | 0.0325 | 0.1524 | +0.1199 | 0.1 % | 72.1 % |
| | carrier_lane `*` | 0.9302 | **0.1296** | 0.0816 | 0.0325 | 0.1620 | **+0.1295** | 0.1 % | 70.3 % |

`gap` = mean(predicted) − mean(observed): positive is **overconfident**.

**All three tasks are overconfident, and substantially so.** Restated as a ratio, the mean stated
probability against the rate at which the event actually occurs:

| task | says on average | actually happens | overstatement |
|---|---|---|---|
| **shortage** | 0.361 | 0.054 | **6.7×** |
| **impact** | 0.162 | 0.033 | **5.0×** |
| **delay** | 0.369 | 0.124 | **3.0×** |

Note the inversion against AUC: `shortage` has the **best** discrimination of the three (0.907)
and the **worst** calibration (ECE 0.307). That is precisely the failure AUC cannot see.

### The two ECE definitions came out identical, and that is itself the finding

The reliability diagram uses **equal-width** deciles (what the brief asks for, and the only
binning under which "fraction in the top bin" is meaningful — equal-count binning forces every bin
to hold 10 %). The project's existing validated ECE (`ml/hypothesis_ranker.py::ece`) uses
**equal-count** bins. Both were computed. They agree to ~5e-5 on `shortage` and `impact`, and to
7.6e-4 on `delay`.

That is not a coincidence and not a bug. When every bin errs in the **same direction**,

```
ECE = Σ (n_b/N)·|conf_b − acc_b| = |Σ (n_b/N)·(conf_b − acc_b)| = |mean(p) − mean(y)|
```

independent of the binning scheme. Checked directly: `shortage` and `impact` have a **positive**
gap in *every* populated bin; `delay` has 9 positive bins and 1 negative (80–90 %), which is
exactly why it is the only task where the two definitions differ at all.

**Consequence, and it is the practically important one:** the miscalibration is a single uniform
directional bias, not a mixture of over- and under-confidence in different regions. A monotone
global map (temperature / Platt / isotonic) is therefore the *right shape of instrument* — there
is no region where it would have to push in the opposite direction.

---

## 2. Reliability diagrams (brief item 3)

Predictions pooled over all 25 cells per task, primary arm. `frac` is the share of all predictions
in that bin; `gap` = mean predicted − observed rate.

### `delay` (old graph) — base rate 0.1240

| bin | n | frac | mean pred | obs rate | gap |
|---|---|---|---|---|---|
| **0–10 %** | **24,098** | **23.1 %** | 0.0193 | 0.0005 | +0.0188 |
| 10–20 % | 4,801 | 4.6 % | 0.1469 | 0.0058 | +0.1410 |
| 20–30 % | 3,942 | 3.8 % | 0.2496 | 0.0216 | +0.2280 |
| 30–40 % | 5,938 | 5.7 % | 0.3576 | 0.0621 | +0.2955 |
| 40–50 % | 24,508 | 23.5 % | 0.4594 | 0.1017 | +0.3577 |
| 50–60 % | 36,040 | 34.5 % | 0.5465 | 0.2190 | +0.3275 |
| 60–70 % | 4,260 | 4.1 % | 0.6270 | 0.3566 | +0.2704 |
| 70–80 % | 491 | 0.5 % | 0.7445 | 0.7149 | +0.0297 |
| 80–90 % | 213 | 0.2 % | 0.8419 | 0.9202 | −0.0783 |
| **90–100 %** | **39** | **0.0 %** | 0.9261 | 0.8718 | +0.0543 |

The mass sits in the middle (58 % between 40 % and 60 %), not at the ceiling. Worst error is in
the 40–50 % bin: the model says 46 %, the event happens 10 % of the time. The top two bins are
nearly empty and — interestingly — are the *only* well-calibrated part of the range.

### `shortage` (fixed graph) — base rate 0.0538

| bin | n | frac | mean pred | obs rate | gap |
|---|---|---|---|---|---|
| **0–10 %** | **4,284** | **0.9 %** | 0.0813 | 0.0002 | +0.0811 |
| 10–20 % | 91,448 | 19.0 % | 0.1642 | 0.0009 | +0.1633 |
| 20–30 % | 148,978 | 31.0 % | 0.2476 | 0.0013 | +0.2463 |
| 30–40 % | 59,817 | 12.4 % | 0.3359 | 0.0039 | +0.3320 |
| 40–50 % | 32,734 | 6.8 % | 0.4513 | 0.0207 | +0.4306 |
| 50–60 % | 73,306 | 15.2 % | 0.5592 | 0.1086 | +0.4506 |
| 60–70 % | 63,070 | 13.1 % | 0.6364 | 0.2165 | +0.4199 |
| 70–80 % | 7,069 | 1.5 % | 0.7308 | 0.4101 | +0.3207 |
| 80–90 % | 374 | 0.1 % | 0.8200 | 0.5428 | +0.2772 |
| **90–100 %** | **0** | **0.0 %** | — | — | — |

The worst-calibrated task, and the pattern is unlike `delay`'s. **The top bin is empty** and the
bottom bin nearly so — 62 % of predictions sit between 10 % and 40 %, where the event happens
0.1 % of the time. A stated 25 % that means 0.13 % is a 190× overstatement. `shortage` is not
saturated at either end; it is uniformly inflated across a compressed middle band.

### `impact` (fixed graph) — base rate 0.0325

| bin | n | frac | mean pred | obs rate | gap |
|---|---|---|---|---|---|
| **0–10 %** | **84,328** | **70.3 %** | 0.0101 | 0.0003 | +0.0098 |
| 10–20 % | 4,064 | 3.4 % | 0.1428 | 0.0030 | +0.1399 |
| 20–30 % | 2,159 | 1.8 % | 0.2475 | 0.0083 | +0.2391 |
| 30–40 % | 1,805 | 1.5 % | 0.3492 | 0.0161 | +0.3331 |
| 40–50 % | 2,431 | 2.0 % | 0.4571 | 0.0366 | +0.4205 |
| 50–60 % | 10,114 | 8.4 % | 0.5633 | 0.0910 | +0.4723 |
| 60–70 % | 11,777 | 9.8 % | 0.6405 | 0.1471 | +0.4934 |
| 70–80 % | 2,882 | 2.4 % | 0.7360 | 0.2915 | +0.4445 |
| 80–90 % | 361 | 0.3 % | 0.8318 | 0.5457 | +0.2861 |
| **90–100 %** | **79** | **0.1 %** | 0.9345 | 0.4557 | +0.4788 |

A third distinct signature: **70 % of predictions are confidently low and essentially correct**
there (says 1.0 %, happens 0.03 %). The damage is concentrated in the 20 % of mass above 50 %,
where gaps reach +0.49 — and the sparse top bin is the worst of all: says 93 %, happens 46 %.

**Three tasks, three different shapes.** The assumption that they would behave alike — in any
direction — would have been wrong.

---

## 3. Per-dataset-seed world (brief item 5)

Primary arm; per-world means over that world's 5 init seeds.

| task | metric | d42 | d43 | d44 | d45 | d46 | world spread |
|---|---|---|---|---|---|---|---|
| **delay** | ECE | 0.2046 | 0.2671 | 0.2280 | 0.2436 | 0.2874 | **0.0828** |
| | top-bin frac | 0.1 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.1 % |
| | bottom-bin frac | 25.5 % | 22.4 % | 28.8 % | 23.4 % | 14.5 % | **14.3 %** |
| | AUC | 0.8187 | 0.7313 | 0.7963 | 0.8029 | 0.7377 | 0.0874 |
| **shortage** | ECE | 0.3125 | 0.3312 | 0.2976 | 0.3197 | 0.2759 | **0.0553** |
| | top-bin frac | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| | bottom-bin frac | 0.6 % | 0.1 % | 1.4 % | 1.6 % | 0.8 % | 1.4 % |
| | AUC | 0.9048 | 0.9226 | 0.8933 | 0.9017 | 0.9123 | 0.0293 |
| **impact** | ECE | 0.1338 | 0.1198 | 0.1336 | 0.1353 | 0.1253 | **0.0155** |
| | top-bin frac | 0.3 % | 0.0 % | 0.0 % | 0.1 % | 0.0 % | 0.3 % |
| | bottom-bin frac | 69.2 % | 73.3 % | 67.6 % | 70.6 % | 70.7 % | 5.8 % |
| | AUC | 0.9190–0.9408 | | | | | 0.0219 |

**Calibration is world-dependent for `delay`, mildly so for `shortage`, and stable for `impact`.**

- `delay`'s ECE spread across worlds (0.0828) is almost exactly its AUC spread (0.0874) — the
  established "dataset seed explains most of the variance" finding applies to calibration too.
  Its bottom-bin mass swings 14.5 %–28.8 %, so the *shape* of the curve changes by world, not just
  its level.
- `impact` is the outlier in the good direction: ECE 0.120–0.135 across all five worlds, a spread
  of 0.0155 against a mean of 0.130 — the miscalibration is large but highly consistent.

Init-seed spread of ECE *within* a world (max over worlds): delay 0.0355, shortage 0.1493, impact
0.0849. `shortage`'s is large because one arm cell is an outlier; the world-level means above are
the more reliable view.

---

## 4. Why all three are overconfident — the objective explains the direction

This is not a mysterious pathology; it is a designed consequence of the training loss, and the
arithmetic is exact. `ml/train.py` fits `FocalLoss(gamma=2, alpha=alpha_from_positive_rate(π))`
with **α = 1 − π**, the inverse-class-frequency choice. Under that weighting the effective prior
the model is trained against is

```
π' = α·π / (α·π + (1−α)·(1−π))     with α = 1−π
   = (1−π)π / ((1−π)π + π(1−π))  =  0.5      — for every task, exactly
```

So the objective deliberately rebalances every task to a 50/50 prior. The model's outputs are
honest probabilities **on that reweighted distribution**, and necessarily inflated relative to the
real event rate. Measured against that prediction:

| task | base rate π | mean predicted | rebalanced target | how far toward 0.5 |
|---|---|---|---|---|
| delay | 0.1240 | 0.3694 | 0.5 | **65 %** |
| shortage | 0.0538 | 0.3612 | 0.5 | **69 %** |
| impact | 0.0325 | 0.1620 | 0.5 | **28 %** |

Every task sits between its true prior and the rebalanced 0.5, which is what the mechanism
predicts. It explains the **direction** for all three and the **magnitude** for two; `impact` is
pulled far less because focal's `(1−p_t)^γ` term down-weights easy negatives and `impact`
classifies most negatives confidently (70 % of its mass in the bottom bin). The α weights alone
would have predicted the opposite ordering (impact has the most aggressive α at 29.8:1 vs delay's
7.1:1), so α is the cause of the direction but not the ranking.

**This is good news for correctability**: a known, systematic, single-signed prior shift is close
to the ideal case for a post-hoc monotone map.

---

## 5. Verdicts and recommendation (brief item 6)

**No correction was fitted or applied.** Per task:

### `shortage` — real gap, largest of the three. **Correction warranted, aggressive.**
ECE 0.3074; states 0.361 where the event occurs 0.054 (**6.7×**). Every populated bin is
overconfident, with gaps up to +0.45. 62 % of predictions sit in the 10–40 % band where the true
rate is ~0.1 %. It has the best AUC of the three and the worst calibration, so its ranking is
usable and its probabilities are not. **Highest priority.**

### `delay` — real gap, but not the anticipated one. **Correction warranted, moderate.**
ECE 0.2461; states 0.369 where the event occurs 0.124 (**3.0×**). **The saturation premise is
false for these models** (0.0000 % at p ≥ 0.999; median p ≈ 0.46) — the problem is a uniform
upward bias across a mid-range-concentrated distribution, not a ceiling effect. Its top two bins
are actually the best-calibrated part of its range. A correction is justified, but a
saturation-motivated fix would have been aimed at the wrong thing.

### `impact` — real gap, smallest, and by far the most stable. **Correction warranted, mild — and it is the only task where a shared map is plausible.**
ECE 0.1296; states 0.162 where the event occurs 0.033 (**5.0×**). But 70 % of its predictions are
in the bottom bin and near-correct there; the error is concentrated in the ~20 % of mass above
0.5. Its world-to-world ECE spread is 0.0155, an order of magnitude tighter than `delay`'s.

**All three show a genuine gap, so all three are recommended — but not by default; each is
supported by its own measurement above,** and they differ in severity (0.31 / 0.25 / 0.13), in
shape (compressed mid-band / mid-range bias / sparse-high-tail), and in how aggressive a
correction should be.

### The constraint that governs any follow-up

This project has already established that calibration maps **do not transfer across worlds**:
`reports/world_conditioned_calibration.md` and `reports/layer3_uncertainty_aware.md` STEP 3 found
a within-world isotonic fit removes essentially all miscalibration (+0.0202 / +0.0224) while a map
fitted on one world **harms** another (−0.0054 / −0.0029 over 20 leave-one-world-out folds), and
that fitting within-world is leakage — classified **STOP — G**. That finding constrains this one
directly, and the per-world spreads above say how much:

| task | ECE mean | world spread | spread as % of the gap | shared map plausible? |
|---|---|---|---|---|
| impact | 0.1296 | 0.0155 | 12 % | **yes, most likely** |
| shortage | 0.3074 | 0.0553 | 18 % | probably |
| delay | 0.2461 | 0.0828 | 34 % | **doubtful** |

For `delay`, a third of the gap is world-specific, which is the regime where transfer already
failed. Any correction step must therefore be validated **leave-one-world-out**, exactly as that
prior work was, and not scored on the world it was fitted on.

### Suggested follow-up, in priority order

1. **Fit and LOWO-validate a single global temperature per task.** Section 1 showed the
   miscalibration is uniformly single-signed, so one scalar is the right first instrument — and
   temperature is the most transferable option available, which matters given the constraint
   above. Start with `impact` (tightest world spread → cleanest test of whether transfer works
   here at all), then `shortage`, then `delay`.
2. **Do not fit isotonic first.** It is more flexible and will fit each world's curve better,
   which is exactly how the previous attempt failed to transfer.
3. **Report ECE alongside AUC from now on.** `shortage` having the best AUC and the worst ECE is
   the clearest possible demonstration that one does not stand in for the other.

---

## 6. Notes and limitations

- **Nothing was retrained; no correction was fitted.** This run only scores existing checkpoints.
- **`reports/decision_support_build.md` is absent from this repository**, so its 96.5 % figure
  could not be examined in context. The contradiction is stated against the frozen Phase 0
  saturation record, which is present and agrees with this run.
- **Pooled reliability curves mix 25 models.** Bin-level numbers in §2 are a mixture; all
  summary metrics in §1 and §3 are computed per cell and then averaged, which is the stricter view.
- **One dead head is included in the secondary arm.** `carrier_lane d45 m3 delay` emits a single
  constant value (0.4931, AUC 0.5000, ECE 0.3728) — see `reports/new_nodes_fix1.md` §4. It affects
  only the non-primary `delay` row and is left in rather than silently dropped.
- **`shortage`'s label rows outnumber its nodes.** 481,080 predictions come from
  (product × warehouse) label instances sharing Product embeddings — the documented coarsening in
  `ml/models/heads.py`. Bin counts reflect label rows, not distinct nodes.
- **Test-set only.** No calibration statistic here touches train or validation splits.

## 7. Reproducing

```
python3 ml/calibration_diagnostic.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

~15 min on CPU; loads 50 cached checkpoints and runs forward passes only.
