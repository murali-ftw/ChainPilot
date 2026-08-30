# Calibration correction 2 — is the tail residual real, and does a beta top-up transfer?

**Post-hoc only; nothing retrained; nothing already shipped is removed or downgraded.** This step
can only *add* a validated top-up or confirm the existing correction stands.

**Headline.** The residual is **real on all three tasks** — but it is not where the framing
expected, and it is not fittable on two of them. A 2-parameter beta top-up transfers on `impact`
only. `delay` and `shortage` keep exactly what
[reports/calibration_correction_1.md](reports/calibration_correction_1.md) shipped.

| task | tail residual | beta fit | LOWO | **final shipped chain** |
|---|---|---|---|---|
| **delay** | **real** (under-confident) | attempted | **REJECT** — Brier degrades on d42, ECE on 3/5 | `analytic` (unchanged) |
| **shortage** | **real** (under-confident) | attempted | **REJECT** — Brier degrades on d44 | `analytic` (unchanged) |
| **impact** | **real** (under-confident) | attempted | **ACCEPT** — Brier improves 5/5 | `analytic + temperature + beta` |

Code: [ml/calibration_correction2.py](ml/calibration_correction2.py) · results:
[out/layer3_v3/calibration_correction2.json](out/layer3_v3/calibration_correction2.json)

**Out-of-fold discipline.** `impact` ships a fitted temperature, so measuring its residual against
a T fitted to the same data would flatter it. Every `impact` number here uses the T fitted on the
**other four worlds**. `delay` and `shortage` ship analytic-only, which is unfitted.

---

## 1. Is the residual real? (brief item 1)

95 % Wilson score intervals on the observed rate, pooled across all 25 cells. Wilson rather than
the normal approximation because these bins hold as few as 1 point at rates near 0 or 1, exactly
where the normal interval under-covers.

### `delay` — shipped `analytic`, tail p ≥ 20 %

| bin | n | pos | mean pred | obs rate | 95 % CI on observed | verdict |
|---|---|---|---|---|---|---|
| 20–30 % | 10,059 | 3,325 | 0.2347 | 0.3305 | [0.3214, 0.3398] | **REAL — under-confident** |
| 30–40 % | 460 | 304 | 0.3373 | 0.6609 | [0.6164, 0.7026] | **REAL — under-confident** |
| 40–50 % | 168 | 141 | 0.4459 | 0.8393 | [0.7763, 0.8871] | **REAL — under-confident** |
| 50–60 % | 72 | 61 | 0.5394 | 0.8472 | [0.7468, 0.9125] | **REAL — under-confident** |
| 60–70 % | 49 | 48 | 0.6436 | 0.9796 | [0.8931, 0.9964] | **REAL — under-confident** |
| 70–80 % | 17 | 16 | 0.7472 | 0.9412 | [0.7302, 0.9895] | noise |
| 80–90 % | 6 | 4 | 0.8520 | 0.6667 | [0.3000, 0.9032] | noise |
| 90–100 % | 1 | 0 | 0.9484 | 0.0000 | [0.0000, 0.7935] | *"real" on n=1 — disregard* |
| **≥ 20 % pooled** | **10,832** | 3,899 | **0.2474** | **0.3600** | **[0.3510, 0.3690]** | **REAL — under-confident** |

### `shortage` — shipped `analytic`, tail p ≥ 20 %

| bin | n | pos | mean pred | obs rate | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| 20–30 % | 4,649 | 1,953 | 0.2286 | 0.4201 | [0.4060, 0.4343] | **REAL — under-confident** |
| 30–40 % | 368 | 204 | 0.3277 | 0.5543 | [0.5033, 0.6043] | **REAL — under-confident** |
| 40–50 % | 17 | 11 | 0.4142 | 0.6471 | [0.4130, 0.8269] | noise |
| **≥ 20 % pooled** | **5,034** | 2,168 | **0.2365** | **0.4307** | **[0.4171, 0.4444]** | **REAL — under-confident** |

### `impact` — shipped `analytic + temperature` (out-of-fold T), tail p ≥ 10 %

| bin | n | pos | mean pred | obs rate | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| 10–20 % | 15,788 | 2,709 | 0.1227 | 0.1716 | [0.1658, 0.1775] | **REAL — under-confident** |
| 20–30 % | 293 | 171 | 0.2296 | 0.5836 | [0.5264, 0.6386] | **REAL — under-confident** |
| 30–40 % | 58 | 30 | 0.3393 | 0.5172 | [0.3916, 0.6407] | **REAL — under-confident** |
| 40–50 % | 13 | 6 | 0.4393 | 0.4615 | [0.2321, 0.7086] | noise |
| 50–60 % | 6 | 1 | 0.5488 | 0.1667 | [0.0301, 0.5635] | noise |
| 60–70 % | 2 | 0 | 0.6875 | 0.0000 | [0.0000, 0.6576] | *"real" on n=2 — disregard* |
| 70–80 % | 2 | 0 | 0.7285 | 0.0000 | [0.0000, 0.6576] | *"real" on n=2 — disregard* |
| 90–100 % | 4 | 0 | 0.9630 | 0.0000 | [0.0000, 0.4899] | *"real" on n=4 — disregard* |
| **≥ 10 % pooled** | **16,166** | 2,917 | **0.1262** | **0.1804** | **[0.1746, 0.1864]** | **REAL — under-confident** |

### A factual correction to the premise

The brief describes this residual as sitting in bins holding "well under 1 % of prediction mass
per task". That is not what the data shows:

| task | tail as defined | share of all predictions | mass in CI-excluding bins | mass in bins with n < 50 |
|---|---|---|---|---|
| delay | p ≥ 0.20 | **10.38 %** | 10.36 % | 0.07 % |
| shortage | p ≥ 0.20 | 1.05 % | 1.04 % | 0.004 % |
| impact | p ≥ 0.10 | **13.47 %** | 13.46 % | 0.02 % |

**The real bias is not in the sparse extreme tail — it is in the first bin or two above the
bulk**, which are large (10,059 / 4,649 / 15,788 points) and give tight intervals. The genuinely
sparse bins (n < 50, ≤ 0.07 % of mass) are mostly *noise*, and the three that flag as "real" do so
on n = 1, 2, 2 and 4, where a Wilson interval is formally valid but practically uninformative.
Those are disregarded rather than treated as evidence.

So the answer to "real signal or small-sample noise" is: **real, and with far more mass behind it
than assumed** — which is why a fit was attempted on all three rather than none.

---

## 2. Beta top-up, leave-one-world-out (brief items 2–3)

`p' = sigmoid(a·logit(p) + b)`, fitted by NLL on 4 worlds, scored on the held-out 5th. `a` is
constrained positive so the map cannot invert the ranking. Every number below is held-out.

**Acceptance rule.** Matching the precedent in `reports/calibration_correction_1.md` — where
`impact`'s temperature improved Brier 5/5 and shipped, while `shortage`'s improved ECE 4/5 but
degraded Brier 5/5 and was rejected as bin-boundary gaming — **Brier arbitrates**. Generalised to
the bar this project's STOP — G history justifies: a fitted correction ships only if it improves
the proper scoring rule on **every** held-out world. One world where it hurts is exactly the
transfer failure already recorded twice.

### `delay` (on top of `analytic`) — **REJECT**

| held-out | a | b | ECE shipped | ECE +beta | ΔECE | Brier shipped | Brier +beta | ΔBrier |
|---|---|---|---|---|---|---|---|---|
| d42 | 1.9761 | 1.9063 | 0.0544 | 0.0584 | −0.0040 | 0.1146 | 0.1148 | **−0.0003** |
| d43 | 1.8630 | 1.5766 | 0.0139 | 0.0173 | −0.0034 | 0.0878 | 0.0870 | +0.0008 |
| d44 | 1.6933 | 1.3002 | 0.0307 | 0.0156 | +0.0151 | 0.0990 | 0.0960 | +0.0030 |
| d45 | 1.6990 | 1.3343 | 0.0215 | 0.0236 | −0.0022 | 0.0949 | 0.0922 | +0.0028 |
| d46 | 1.8074 | 1.4766 | 0.0203 | 0.0179 | +0.0024 | 0.0892 | 0.0883 | +0.0009 |
| **mean** | 1.8078 | 1.5188 | 0.0281 | 0.0266 | +0.0016 | 0.0971 | 0.0957 | +0.0015 |

ECE improves on only **2/5** folds and Brier degrades on d42. The mean looks positive, but it is
carried by one fold (d44, +0.0151 ECE). Parameter spread is wide (a: 0.283, b: 0.606). Rejected.

### `shortage` (on top of `analytic`) — **REJECT**

| held-out | a | b | ECE shipped | ECE +beta | ΔECE | Brier shipped | Brier +beta | ΔBrier |
|---|---|---|---|---|---|---|---|---|
| d42 | 2.5117 | 3.1279 | 0.0359 | 0.0158 | +0.0202 | 0.0454 | 0.0431 | +0.0023 |
| d43 | 2.5674 | 3.1797 | 0.0346 | 0.0114 | +0.0232 | 0.0356 | 0.0340 | +0.0016 |
| **d44** | 2.9396 | 4.1483 | 0.0329 | 0.0422 | **−0.0094** | 0.0547 | 0.0577 | **−0.0030** |
| d45 | 2.4463 | 3.0008 | 0.0451 | 0.0220 | +0.0231 | 0.0515 | 0.0482 | +0.0033 |
| d46 | 2.7755 | 3.5299 | 0.0230 | 0.0212 | +0.0018 | 0.0374 | 0.0373 | +0.0001 |
| **mean** | 2.6481 | 3.3973 | 0.0343 | 0.0225 | +0.0118 | 0.0449 | 0.0441 | +0.0009 |

This is the closest call in the report: 4/5 folds improve substantially on **both** metrics, and
the mean ECE gain (+0.0118) is the largest of the three tasks. It fails on **d44 alone**, which
degrades both ECE (−0.0094) and Brier (−0.0030).

**d44 is the same world that broke `shortage`'s temperature fit** in
`reports/calibration_correction_1.md` (the only ECE-degrading fold there too). That is not a
coincidence to wave away — it is a specific world whose calibration curve differs from the other
four, and it is exactly the STOP — G failure mode. Parameter spread is also the widest of the
three (a: 0.493, b: 1.148), with d44 pulling hardest (a = 2.94 vs 2.45–2.78 elsewhere). Rejected.

### `impact` (on top of `analytic + temperature`) — **ACCEPT**

| held-out | a | b | ECE shipped | ECE +beta | ΔECE | Brier shipped | Brier +beta | ΔBrier |
|---|---|---|---|---|---|---|---|---|
| d42 | 2.0174 | 2.3079 | 0.0136 | 0.0093 | +0.0042 | 0.0247 | 0.0241 | +0.0006 |
| d43 | 1.9772 | 2.2722 | 0.0127 | 0.0048 | +0.0079 | 0.0266 | 0.0254 | +0.0011 |
| d44 | 2.0439 | 2.4464 | 0.0153 | 0.0094 | +0.0059 | 0.0319 | 0.0307 | +0.0012 |
| d45 | 2.0129 | 2.3487 | 0.0092 | 0.0089 | +0.0003 | 0.0262 | 0.0254 | +0.0008 |
| d46 | 2.1336 | 2.5878 | 0.0094 | 0.0104 | −0.0010 | 0.0281 | 0.0276 | +0.0004 |
| **mean** | **2.0370** | **2.3926** | 0.0120 | **0.0086** | **+0.0035** | 0.0275 | **0.0266** | **+0.0008** |

**Brier improves on 5/5 folds**; ECE improves on 4/5, with the single exception (d46, −0.0010)
being the *benign* direction — proper rule up, binned metric marginally down, which is the
opposite of the gaming signature. Parameters are by far the most stable of the three
(a: 2.013–2.134, spread 0.156; b: 2.272–2.588, spread 0.316). Accepted.

**Parameter stability tracks transferability.** The ordering of `(a, b)` fold-spread — impact
0.156/0.316, delay 0.283/0.606, shortage 0.493/1.148 — is exactly the ordering of the LOWO
outcome. A fit whose parameters swing between worlds is a fit describing world-specific structure.

---

## 3. Per-task verdicts (brief item 4)

| task | residual real? | beta LOWO ECE | beta LOWO Brier | decision | **final chain** | **final held-out ECE / Brier** |
|---|---|---|---|---|---|---|
| **delay** | **yes** (10.4 % of mass, under-confident) | 0.0281 → 0.0266 (2/5 folds) | 0.0971 → 0.0957 (4/5) | **REJECT** | `analytic` | **0.0281 / 0.0971** |
| **shortage** | **yes** (1.05 % of mass, under-confident) | 0.0343 → 0.0225 (4/5) | 0.0449 → 0.0441 (4/5) | **REJECT** | `analytic` | **0.0343 / 0.0449** |
| **impact** | **yes** (13.5 % of mass, under-confident) | 0.0120 → **0.0086** (4/5) | 0.0275 → **0.0266** (5/5) | **ACCEPT** | `analytic + temperature + beta` | **0.0086 / 0.0266** |

**`delay` — keep `analytic`.** The residual is real but the fit does not transfer: ECE worsens on
3 of 5 held-out worlds. Nothing changes from `reports/calibration_correction_1.md`.

**`shortage` — keep `analytic`.** The most tempting rejection in this report: 4/5 folds improve
both metrics, and the mean ECE gain would be the largest of the three. It is rejected on d44
alone, the same world that broke its temperature fit. Shipping it would mean accepting a known
+0.0030 Brier regression on a world we have already seen this task fail to transfer to, for a
mean gain of +0.0009. Under the established rule that is not a close call, even though the
numbers look close.

**`impact` — ship `analytic + temperature + beta`** (a ≈ 2.037, b ≈ 2.393). This is a genuine
improvement on top of an already-validated correction: held-out ECE 0.0120 → **0.0086** and
Brier 0.0275 → **0.0266**, with Brier improving on every fold and the tightest parameters of the
three. Cumulatively, `impact`'s ECE has now gone **0.1296 → 0.0177 → 0.0120 → 0.0086**, a 93.4 %
total reduction from raw.

---

## 4. Usage guidance (brief item 5)

Computed on each task's **final** shipped chain, out-of-fold. Classification:

- **absolute** — |mean predicted − observed| < 0.05. Read the number as a probability.
- **rank only** — gap ≥ 0.05 **and** predicted outside the 95 % CI. Demonstrated bias.
- **insufficient** — gap ≥ 0.05 but predicted inside the CI. Too few points to tell either way.

| task | chain | band | mass | n | mean pred | obs rate | \|gap\| | use |
|---|---|---|---|---|---|---|---|---|
| **delay** | `analytic` | 0–10 % | 38.1 % | 39,719 | 0.0253 | 0.0153 | 0.010 | **absolute** |
| | | 10–20 % | 51.5 % | 53,779 | 0.1442 | 0.1575 | 0.013 | **absolute** |
| | | 20–30 % | 9.6 % | 10,059 | 0.2347 | 0.3305 | 0.096 | rank only |
| | | 30–40 % | 0.4 % | 460 | 0.3373 | 0.6609 | 0.324 | rank only |
| | | 40–50 % | 0.2 % | 168 | 0.4459 | 0.8393 | 0.393 | rank only |
| | | 50–70 % | 0.1 % | 121 | — | — | 0.31–0.34 | rank only |
| | | 70–90 % | 0.02 % | 23 | — | — | ~0.19 | insufficient |
| | | 90–100 % | 0.001 % | 1 | 0.9484 | 0.0000 | 0.948 | rank only |
| **shortage** | `analytic` | 0–10 % | 75.7 % | 364,412 | 0.0365 | 0.0097 | 0.027 | **absolute** |
| | | 10–20 % | 23.2 % | 111,634 | 0.1316 | 0.1810 | 0.050 | **absolute** |
| | | 20–30 % | 1.0 % | 4,649 | 0.2286 | 0.4201 | 0.192 | rank only |
| | | 30–40 % | 0.1 % | 368 | 0.3277 | 0.5543 | 0.227 | rank only |
| | | 40–50 % | 0.004 % | 17 | 0.4142 | 0.6471 | 0.233 | insufficient |
| **impact** | `analytic+temp+beta` | 0–10 % | 84.6 % | 101,506 | 0.0077 | 0.0076 | **0.0001** | **absolute** |
| | | 10–20 % | 12.1 % | 14,539 | 0.1369 | 0.1336 | **0.003** | **absolute** |
| | | 20–30 % | 2.4 % | 2,897 | 0.2398 | 0.2547 | **0.015** | **absolute** |
| | | 30–40 % | 0.6 % | 699 | 0.3355 | 0.3534 | **0.018** | **absolute** |
| | | 40–50 % | 0.2 % | 189 | 0.4416 | 0.5714 | 0.130 | rank only |
| | | 50–60 % | 0.04 % | 51 | 0.5351 | 0.6863 | 0.151 | rank only |
| | | 60–70 % | 0.05 % | 56 | 0.6400 | 0.5714 | 0.069 | insufficient |
| | | 70–100 % | 0.05 % | 63 | — | — | 0.17–0.90 | rank only |

### Summary

| task | absolute-trustworthy range | mass | rank-only | insufficient |
|---|---|---|---|---|
| **delay** | **p < 0.20** | **89.62 %** | 10.36 % | 0.02 % |
| **shortage** | **p < 0.20** | **98.95 %** | 1.04 % | 0.00 % |
| **impact** | **p < 0.40** | **99.70 %** | 0.25 % | 0.05 % |

`impact`'s accepted beta top-up is visible here: its first four bands are now calibrated to within
0.02 (0.0077 vs 0.0076; 0.1369 vs 0.1336; 0.2398 vs 0.2547; 0.3355 vs 0.3534), extending its
absolute-trustworthy range from p < 0.2 to **p < 0.4**.

### Ranking is unaffected — everywhere, including where absolute values are not trustworthy

**Every correction in this chain — analytic prior shift, temperature, and beta — is strictly
monotone in `p`.** So for any downstream use that depends only on *order* — percentile
thresholding, top-K triage, ranked worklists — **this entire residual is irrelevant**. A
`delay` prediction in the 40–50 % band should not be read as "44 % likely" (it is 84 %), but it is
still correctly ranked above one in the 20–30 % band, and AUC is unchanged from every prior report.

Verified numerically across all 75 cells:

| task | chain | max \|ΔAUC\| vs raw |
|---|---|---|
| delay | `analytic` | **0.00e+00** |
| shortage | `analytic` | **0.00e+00** |
| impact | `analytic + temperature + beta` | **7.69e-06** |

`impact`'s 7.69e-06 is a floating-point artifact, not a real reordering: the logit-based stages
clip at 1e-7, and 14,672 of its predictions sit below that clip, so distinct near-zero values map
to one value and create ties that did not exist. The maps are strictly monotone in exact
arithmetic. The magnitude is **five orders below `impact`'s own reproduction floor (0.0252)** and
cannot move any reported statistic.

**Practical guidance:** use absolute probabilities below each task's threshold above; use rank
above it. `impact` is now usable as an absolute probability over 99.7 % of its distribution;
`shortage` over 99.0 %; `delay` over 89.6 %.

---

## 5. Nothing was downgraded (brief item 6)

- `delay` and `shortage` ship **exactly** what `reports/calibration_correction_1.md` shipped:
  the analytic prior shift, unchanged. Their rejected beta fits are recorded, not applied.
- `impact` **gains** a top-up on top of its existing analytic + temperature correction; neither
  earlier component was refitted or replaced.
- No AUC-based pass/fail status changes. The standing verdicts from
  [reports/new_nodes_fix1.md](reports/new_nodes_fix1.md) — `shortage` PASS, `impact` PASS, `delay`
  still failing on its dataset-seed floor — are untouched, and cannot be affected by a monotone
  post-hoc map.

## 6. Limitations

- **The n ≤ 4 bins that flag as "real over-confident"** (delay 90–100 %, impact 60–70 / 70–80 /
  90–100 %) are formally CI-excluding but carry 1–4 points. They are reported for completeness and
  disregarded in every decision.
- **`shortage`'s rejection rests on one world.** 4/5 folds improve both metrics substantially. If
  d44's behaviour were understood and shown to be an artifact rather than a real world difference,
  this decision should be revisited — it is the highest-value open item here.
- **Residual shape error remains on `delay` and `shortage`** above p ≈ 0.2, by design: no
  instrument that transfers was found for it. It affects 10.4 % and 1.0 % of their mass.
- **`impact`'s beta was fitted on top of its temperature**, so it inherits any error in that T and
  in π. The three stages are not jointly optimised.
- **Bin counts reflect label rows, not distinct nodes** for `shortage` (the documented
  Product×Warehouse coarsening in `ml/models/heads.py`).

## 7. Reproducing

```
python3 ml/calibration_correction2.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

Seconds, from the cached predictions built by `ml/calibration_correction.py`.
