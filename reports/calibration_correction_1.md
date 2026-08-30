# Calibration correction 1 — analytic prior shift, then a LOWO-validated residual

**Post-hoc only. SHARE and Markov are untouched and nothing was retrained.** This is a correction
layer applied to already-computed probabilities from the same 25-cell grid
(5 dataset seeds × 5 model-init seeds) used by
[reports/calibration_diagnostic.md](reports/calibration_diagnostic.md).

**Headline.** The analytic step alone — zero data fitting — removes **86–89 % of the calibration
error on all three tasks**. A residual scalar temperature then survives leave-one-world-out
validation on `impact` (clear gain) and on `delay` (a gain so small it is not worth shipping), and
**fails LOWO on `shortage`**, which keeps the analytic correction alone.

| task | ECE raw → analytic → +temp | recommendation |
|---|---|---|
| **delay** | 0.2461 → **0.0281** → 0.0253 | analytic alone (temperature passes but is ≈ a no-op) |
| **shortage** | 0.3074 → **0.0343** → *0.0253 rejected* | **analytic alone — residual fit FAILS LOWO** |
| **impact** | 0.1296 → 0.0177 → **0.0120** | **analytic + temperature (T ≈ 1.314)** |

Code: [ml/calibration_correction.py](ml/calibration_correction.py) · results:
[out/layer3_v3/calibration_correction.json](out/layer3_v3/calibration_correction.json)

---

## 1. What was applied, and the one choice that matters

**Step 1 — analytic prior shift, no fitting.** A model trained under effective prior `π'` and
deployed under true prior `π` is corrected on the odds:

```
odds_corrected = odds_raw · (π / (1−π)) · ((1−π') / π')
```

With `π' = 0.5` the second factor is 1, giving the brief's form
`p_corr = p·π / (p·π + (1−p)·(1−π))`. The diagnostic established `π' = 0.5` exactly, for every
task, because `ml/train.py` uses `FocalLoss(gamma=2, alpha=1−π)` and that α cancels the true prior
identically.

**Which π — and why not the one in the brief.** The brief quotes the *test* base rates (0.1240 /
0.0538 / 0.0325). Using those would leak held-out information into a step whose entire claim is
that it needs none. The correction instead uses the **train-split positive rate, per world** —
literally the quantity `ml/train.py::compute_task_alphas` computed α from, with its own docstring
noting "alpha is a training hyperparameter, never tuned against held-out data." They differ enough
to matter:

| task | train π per world (d42…d46) | test base rate |
|---|---|---|
| delay | 0.1906 · 0.1255 · 0.1419 · 0.1454 · 0.1232 | 0.1240 |
| shortage | 0.0870 · 0.0672 · 0.1036 · 0.0930 · 0.0687 | 0.0538 |
| impact | 0.0362 · 0.0354 · 0.0421 · 0.0367 · 0.0333 | 0.0325 |

`shortage`'s train rate is ~1.5× its test rate, so the choice is not cosmetic. Using the train
rate keeps "zero data fitting" literally true.

**Step 2 — residual scalar temperature**, `p' = σ(logit(p_corr)/T)`, one scalar per task, fitted
by NLL (a proper scoring rule, smooth in T; ECE is a step function of the binning and can be
driven to a spurious optimum). Fitted on 4 worlds, scored on the held-out 5th, rotated through all
5 folds. Isotonic was deliberately not used — its flexibility is what failed to transfer in
`reports/world_conditioned_calibration.md` (**STOP — G**).

---

## 2. Step 1 results — analytic correction (brief item 2)

Per-cell metrics averaged, same convention as the diagnostic.

| task | stage | AUC | ECE | Brier | mean pred | base rate | gap | \|gap\|/base |
|---|---|---|---|---|---|---|---|---|
| **delay** | raw | 0.7774 | 0.2461 | 0.1735 | 0.3694 | 0.1240 | +0.2454 | 1.98 |
| | **analytic** | 0.7774 | **0.0281** | **0.0971** | 0.1096 | 0.1240 | **−0.0145** | 0.12 |
| **shortage** | raw | 0.9069 | 0.3074 | 0.1491 | 0.3612 | 0.0538 | +0.3074 | 5.71 |
| | **analytic** | 0.9069 | **0.0343** | **0.0449** | 0.0606 | 0.0538 | **+0.0068** | 0.13 |
| **impact** | raw | 0.9302 | 0.1296 | 0.0816 | 0.1620 | 0.0325 | +0.1295 | 3.98 |
| | **analytic** | 0.9302 | **0.0177** | **0.0287** | 0.0151 | 0.0325 | **−0.0174** | 0.54 |

| task | ECE closed | Brier reduced |
|---|---|---|
| delay | **88.6 %** | 44.0 % |
| shortage | **88.8 %** | 69.9 % |
| impact | **86.3 %** | 64.8 % |

### Per dataset-seed world

| task | metric | d42 | d43 | d44 | d45 | d46 | mean | world spread |
|---|---|---|---|---|---|---|---|---|
| **delay** | ECE raw | 0.2046 | 0.2671 | 0.2280 | 0.2436 | 0.2874 | 0.2461 | 0.0828 |
| | ECE analytic | 0.0544 | 0.0139 | 0.0307 | 0.0215 | 0.0203 | **0.0281** | 0.0405 |
| | Brier analytic | 0.1146 | 0.0878 | 0.0990 | 0.0949 | 0.0892 | **0.0971** | 0.0267 |
| | gap analytic | −0.0162 | −0.0105 | −0.0227 | −0.0129 | −0.0100 | **−0.0145** | 0.0127 |
| **shortage** | ECE raw | 0.3125 | 0.3312 | 0.2976 | 0.3197 | 0.2759 | 0.3074 | 0.0553 |
| | ECE analytic | 0.0359 | 0.0346 | 0.0329 | 0.0451 | 0.0230 | **0.0343** | 0.0222 |
| | Brier analytic | 0.0454 | 0.0356 | 0.0547 | 0.0515 | 0.0374 | **0.0449** | 0.0191 |
| | gap analytic | +0.0092 | +0.0080 | +0.0095 | +0.0087 | −0.0013 | **+0.0068** | 0.0108 |
| **impact** | ECE raw | 0.1338 | 0.1198 | 0.1336 | 0.1353 | 0.1253 | 0.1296 | 0.0155 |
| | ECE analytic | 0.0152 | 0.0177 | 0.0204 | 0.0152 | 0.0200 | **0.0177** | 0.0052 |
| | Brier analytic | 0.0258 | 0.0279 | 0.0334 | 0.0271 | 0.0294 | **0.0287** | 0.0076 |
| | gap analytic | −0.0143 | −0.0177 | −0.0201 | −0.0152 | −0.0199 | **−0.0174** | 0.0058 |

The correction works in **every world for every task** — no world is left worse off, and world
spread shrinks on all three (delay 0.0828→0.0405, shortage 0.0553→0.0222, impact 0.0155→0.0052).

### Reliability deciles after the analytic correction

**`delay`** (was: 58 % of mass in 40–60 %, gaps to +0.36)

| bin | n | frac | mean pred | obs rate | gap |
|---|---|---|---|---|---|
| 0–10 % | 39,719 | 38.1 % | 0.0253 | 0.0153 | +0.0100 |
| 10–20 % | 53,779 | 51.5 % | 0.1442 | 0.1575 | −0.0133 |
| 20–30 % | 10,059 | 9.6 % | 0.2347 | 0.3305 | −0.0959 |
| 30–40 % | 460 | 0.4 % | 0.3373 | 0.6609 | −0.3235 |
| 40–50 % | 168 | 0.2 % | 0.4459 | 0.8393 | −0.3934 |
| 50–100 % | 145 | 0.1 % | — | — | mixed |

90 % of the mass is now within ±0.013 of truth. The large gaps sit in the top 0.7 % of the
distribution, and they are now **under**-confident, not over.

**`shortage`** (was: 62 % in 10–40 % where the true rate is ~0.1 %)

| bin | n | frac | mean pred | obs rate | gap |
|---|---|---|---|---|---|
| 0–10 % | 364,412 | 75.7 % | 0.0365 | 0.0097 | +0.0268 |
| 10–20 % | 111,634 | 23.2 % | 0.1316 | 0.1810 | −0.0495 |
| 20–30 % | 4,649 | 1.0 % | 0.2286 | 0.4201 | −0.1915 |
| 30–40 % | 368 | 0.1 % | 0.3277 | 0.5543 | −0.2267 |
| 40–50 % | 17 | 0.0 % | 0.4142 | 0.6471 | −0.2328 |
| 50–100 % | 0 | 0.0 % | — | — | — |

**`impact`** (was: gaps to +0.49 above 0.5)

| bin | n | frac | mean pred | obs rate | gap |
|---|---|---|---|---|---|
| 0–10 % | 118,367 | **98.6 %** | 0.0134 | 0.0278 | **−0.0144** |
| 10–20 % | 1,501 | 1.3 % | 0.1205 | 0.3638 | −0.2433 |
| 20–30 % | 82 | 0.1 % | 0.2433 | 0.5732 | −0.3299 |
| 30–100 % | 50 | 0.0 % | — | — | mixed |

---

## 3. What is left over — measured, not assumed (brief item 3)

The brief asked specifically whether the analytic step **over-corrects `impact`**. It does, and it
is the largest residual of the three:

| task | mean pred after | true base rate | ratio | direction | sign across 5 worlds |
|---|---|---|---|---|---|
| delay | 0.1096 | 0.1240 | **0.88×** | mild **under**-confidence | 5/5 negative |
| shortage | 0.0606 | 0.0538 | **1.13×** | mild residual **over**-confidence | 4/5 positive, d46 negative |
| **impact** | **0.0151** | **0.0325** | **0.46×** | **over-corrected — now under-confident** | **5/5 negative** |

- **`delay` and `shortage` are close to closed.** Residual |gap| is 12–13 % of base rate, against
  world spreads of 0.0127 / 0.0108 — i.e. the leftover is roughly the size of the world-to-world
  noise. `shortage`'s residual is not even sign-consistent across worlds.
- **`impact` over-corrects, as predicted.** Its residual gap is 54 % of its base rate and negative
  in all 5 worlds. It now states 1.3 % where 2.8 % occurs. This is exactly the direction the
  diagnostic anticipated: `impact`'s realised bias was only 28 % of the way to 0.5, while the
  analytic step assumes the full 100 %, so applying it in full pushes past the target.

---

## 4. Step 2 — residual temperature, leave-one-world-out (brief items 4–5)

Sequenced as the brief specifies, by the diagnostic's transfer-risk ranking: `impact` (world
spread 12 % of its gap), then `shortage` (18 %), then `delay` (34 %). T is fitted on 4 worlds and
**every number below is scored on the held-out 5th only**.

### `impact` — **LOWO PASS (5/5 folds, both metrics)**

| held-out | T | ECE analytic | ECE +temp | Δ | Brier analytic | Brier +temp | Δ |
|---|---|---|---|---|---|---|---|
| d42 | 1.3348 | 0.0152 | 0.0136 | +0.0016 | 0.0258 | 0.0247 | +0.0011 |
| d43 | 1.3111 | 0.0177 | 0.0127 | +0.0050 | 0.0279 | 0.0266 | +0.0014 |
| d44 | 1.3099 | 0.0204 | 0.0153 | +0.0052 | 0.0334 | 0.0319 | +0.0015 |
| d45 | 1.3216 | 0.0152 | 0.0092 | +0.0060 | 0.0271 | 0.0262 | +0.0010 |
| d46 | 1.2929 | 0.0200 | 0.0094 | +0.0106 | 0.0294 | 0.0281 | +0.0013 |
| **mean** | **1.3141** | 0.0177 | **0.0120** | **+0.0057** | 0.0287 | **0.0275** | **+0.0013** |

T is tightly stable across folds (1.293–1.335, spread 0.042) and **> 1**, which softens toward
0.5 — exactly the direction needed to undo the over-correction found in §3. Improves both metrics
on every fold. **Ship it.**

### `shortage` — **LOWO FAIL**

| held-out | T | ECE analytic | ECE +temp | Δ | Brier analytic | Brier +temp | Δ |
|---|---|---|---|---|---|---|---|
| d42 | 0.8730 | 0.0359 | 0.0280 | +0.0079 | 0.0454 | 0.0459 | **−0.0005** |
| d43 | 0.8729 | 0.0346 | 0.0129 | +0.0218 | 0.0356 | 0.0361 | **−0.0005** |
| d44 | 0.8757 | 0.0329 | 0.0365 | **−0.0036** | 0.0547 | 0.0552 | **−0.0005** |
| d45 | 0.8707 | 0.0451 | 0.0313 | +0.0138 | 0.0515 | 0.0520 | **−0.0006** |
| d46 | 0.8554 | 0.0230 | 0.0178 | +0.0052 | 0.0374 | 0.0383 | **−0.0009** |
| **mean** | 0.8695 | 0.0343 | 0.0253 | +0.0090 | 0.0449 | 0.0455 | **−0.0006** |

It improves ECE on 4 of 5 folds — and **degrades Brier on 5 of 5**. That is the signature of a fit
that is not learning calibration but shuffling mass between bins: ECE is a binned statistic and
can be improved by moving probability across a bin edge, whereas Brier is a proper scoring rule
and cannot be gamed that way. **Rejected. `shortage` keeps the analytic correction alone.**

### `delay` — **LOWO PASS (5/5, both metrics), but effectively a no-op**

| held-out | T | ECE analytic | ECE +temp | Δ | Brier analytic | Brier +temp | Δ |
|---|---|---|---|---|---|---|---|
| d42 | 1.0376 | 0.0544 | 0.0520 | +0.0024 | 0.1146 | 0.1144 | +0.0002 |
| d43 | 1.0256 | 0.0139 | 0.0111 | +0.0027 | 0.0878 | 0.0878 | +0.0001 |
| d44 | 1.0159 | 0.0307 | 0.0285 | +0.0021 | 0.0990 | 0.0989 | +0.0001 |
| d45 | 1.0338 | 0.0215 | 0.0170 | +0.0044 | 0.0949 | 0.0948 | +0.0002 |
| d46 | 1.0322 | 0.0203 | 0.0179 | +0.0023 | 0.0892 | 0.0891 | +0.0001 |
| **mean** | **1.0290** | 0.0281 | **0.0253** | **+0.0028** | 0.0971 | 0.0970 | **+0.0001** |

T = 1.029 is within 3 % of the identity and the Brier gain is 0.0001. It passes cleanly, but it
buys almost nothing.

---

## 5. Why `shortage` failed and `impact` passed — and why the risk ranking did not predict it

The diagnostic's transfer-risk ordering (impact safest at 12 %, delay riskiest at 34 %) **did not
predict the outcome**: `delay` passed and `shortage` failed. World spread was the wrong variable.
The right one is whether the residual is still **single-signed**, measured as the share of
prediction mass sitting in positively- vs negatively-biased bins:

| task | stage | mass in +gap bins | mass in −gap bins | single-signed? |
|---|---|---|---|---|
| delay | raw | 99.8 % | 0.2 % | **yes** |
| | analytic | 38.1 % | 61.9 % | **no** |
| shortage | raw | 100.0 % | 0.0 % | **yes** |
| | analytic | 75.7 % | 24.3 % | **no** |
| impact | raw | 100.0 % | 0.0 % | **yes** |
| | **analytic** | **0.0 %** | **100.0 %** | **yes** |

The diagnostic's single-signed finding was about the **raw** probabilities, and that is precisely
what made a global monotone shift the right instrument for step 1 — it removed 86–89 % of the
error on all three. But the analytic step *consumes* the single-signed component. What remains is
**shape** error, and on `delay` and `shortage` that residual is sign-mixed over a large share of
the mass. A single scalar cannot push in two directions at once, so on `shortage` it trades one
bin against another — improving the binned metric while degrading the proper one, exactly as
observed. `impact` is the one task whose residual stays 100 % single-signed (98.6 % of its mass in
one bin, all under-confident), which is why one scalar both fits and transfers there.

This also means the brief's premise for step 4 — "the miscalibration is uniformly single-signed,
so a single scalar is the right-shaped instrument" — holds for step 1 but **stops holding after
it**, for two of the three tasks. That is not an argument for a more flexible fit: isotonic would
fit that sign-mixed shape per world and is exactly what failed to transfer before.

---

## 6. Final verdicts (brief item 6)

| task | original ECE | after analytic (no fitting) | closed by analytic | residual fit | after residual (held-out) | **final ECE** | **final Brier** |
|---|---|---|---|---|---|---|---|
| **delay** | 0.2461 | 0.0281 | **88.6 %** | LOWO pass, +0.0028 — negligible | 0.0253 | **0.0281** | **0.0971** |
| **shortage** | 0.3074 | 0.0343 | **88.8 %** | **LOWO FAIL** (Brier worse 5/5) | *rejected* | **0.0343** | **0.0449** |
| **impact** | 0.1296 | 0.0177 | **86.3 %** | LOWO pass, +0.0057 (5/5) | 0.0120 | **0.0120** | **0.0275** |

**`delay` — ship the analytic correction alone.** It closes 88.6 % of the gap with no fitting. The
temperature passes LOWO on all 5 folds but T = 1.029 and the Brier gain is 0.0001; given this
project's STOP — G history, adding a fitted component for that is not worth the transfer risk it
carries. Residual: mild under-confidence, 12 % of base rate.

**`shortage` — ship the analytic correction alone. The residual fit fails its LOWO check.** ECE
improves on 4/5 folds but Brier degrades on 5/5, which means the apparent ECE gain is bin-boundary
movement rather than better probabilities. The analytic step alone already leaves a residual gap
(+0.0068) that is smaller than the world-to-world spread of that gap (0.0108) and not even
sign-consistent across worlds — there is little real signal left for a correction to capture.

**`impact` — ship the analytic correction plus temperature T ≈ 1.314.** This is the one task where
the analytic step measurably **over**-corrects (to 0.46× the true rate), the one whose residual
stays single-signed, and the one whose fitted T is stable across folds (1.293–1.335) and improves
both ECE and Brier on every held-out fold. Final held-out ECE 0.0120, Brier 0.0275.

---

## 7. AUC and pass/fail status are unchanged (brief item 7)

Both corrections are **strictly monotone in `p`**, so every rank-based statistic is mathematically
invariant. Verified numerically rather than asserted: across all **75 cells** (3 tasks × 5 worlds ×
5 init seeds),

```
max | AUC_raw − AUC_corrected |  =  0.00e+00
```

This is a calibration layer applied *after* the ranking is produced. **It does not change any
task's pass/fail status under the existing floor-based bar**, and must not be reported as doing
so. The standing verdicts from [reports/new_nodes_fix1.md](reports/new_nodes_fix1.md) are
untouched: `shortage` PASS, `impact` PASS, `delay` still failing on its dataset-seed floor.

---

## 8. Limitations

- **Nothing was retrained.** SHARE, Markov and all 75 checkpoints are unmodified; this operates on
  cached forward-pass outputs.
- **π is the train-split rate, per world.** This is leak-free and is the quantity α was derived
  from, but it means the correction is world-specific in a mild sense — it uses a training
  statistic of the world the model was trained on, which is available at deployment for that
  model. It is not fitted to any held-out data.
- **The analytic step is not LOWO-validated because it is not fitted.** Its numbers are full-grid.
  Only the temperature is LOWO-scored.
- **`impact`'s T was fitted on the analytic output**, so it inherits any error in π. A different π
  choice would shift the optimal T.
- **Residual shape error remains on all three tasks**, concentrated in the sparse high-probability
  tail (e.g. `delay`'s 30–50 % bins are under-confident by ~0.32–0.39 over 0.6 % of the mass). No
  instrument used here addresses shape, by design.
- **`shortage`'s prediction count reflects label rows, not distinct nodes** (the documented
  Product×Warehouse coarsening in `ml/models/heads.py`).

## 9. Reproducing

```
python3 ml/calibration_correction.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

First run builds a prediction cache (~15 min of forward passes on frozen checkpoints); subsequent
runs are seconds.
