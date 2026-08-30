# Operating points — what happens if you actually act on the model's output

**Read-only.** No retraining, no new forward passes, no change to any shipped calibration
correction or to any AUC-based pass/fail status. This translates results already computed into a
decision-threshold view, from the prediction cache built by `ml/calibration_correction.py`.

Each task is thresholded on its **final shipped probability** from
[reports/calibration_correction_2.md](reports/calibration_correction_2.md):

| task | chain | base rate |
|---|---|---|
| `delay` | analytic prior shift only | 12.44 % |
| `shortage` | analytic prior shift only | 5.38 % |
| `impact` | analytic + temperature + beta | 3.25 % |

Anything fitted is applied **out of fold** — `impact`'s T and (a, b) for a world are the ones
fitted on the other four — so no operating point is read off a correction tuned on the data it is
scored against. Precision and recall are computed inside each of the 25 cells (5 dataset seeds ×
5 model-init seeds) on a shared threshold grid, then averaged.

Code: [ml/operating_points.py](ml/operating_points.py) · results:
[out/layer3_v3/operating_points.json](out/layer3_v3/operating_points.json)

---

## 0. Why there is no accuracy number in this report

At these base rates a model that answers "no" to everything scores:

| task | "always no" accuracy | real cases it catches |
|---|---|---|
| delay | **87.6 %** | 0 |
| shortage | **94.6 %** | 0 |
| impact | **96.75 %** | 0 |

A single accuracy figure would make a useless model look excellent, so none is computed here for
any task — not even informally. The right answer to a request for one is the precision and recall
pair below.

---

## 1. Summary table — three tasks × three operating points

Read every row as: **out of 100 real cases, the model catches N** (recall), and **out of 100 cases
the model flags, M are actually real** (precision).

| task | operating point | threshold | catches N /100 real | M /100 flags real | F1 | % of all cases flagged | band |
|---|---|---|---|---|---|---|---|
| **delay** | balanced (max F1) | 0.1440 | **64.6** | **24.3** | 0.345 | 33.7 % | absolute |
| | high-recall (~80 % caught) | 0.1283 | **80.0** | **21.6** | 0.339 | 46.2 % | absolute |
| | high-precision (~80 % real) | 0.2877 | **4.6** | **80.1** | 0.083 | 0.9 % | ⚠ **RANK-ONLY** |
| **shortage** | balanced (max F1) | 0.1176 | **67.2** | **25.5** | 0.332 | 17.3 % | absolute |
| | high-recall (~80 % caught) | 0.1056 | **80.0** | **21.9** | 0.325 | 22.2 % | absolute |
| | high-precision (*80 % unattainable*) | 0.1999 | **7.1** | **53.9** | 0.097 | 1.0 % | absolute |
| **impact** | balanced (max F1) | 0.1412 | **53.9** | **25.7** | 0.314 | 7.9 % | absolute |
| | high-recall (~80 % caught) | 0.1003 | **80.0** | **18.0** | 0.283 | 15.4 % | absolute |
| | high-precision (*80 % unattainable*) | 0.3950 | **5.3** | **56.8** | 0.092 | 0.3 % | absolute |

Stated in words, at the balanced point:

- **`delay`** — out of 100 real delay cases the model catches **65**; out of 100 cases it flags,
  **24** are actually real. It flags a third of all shipments to do that.
- **`shortage`** — catches **67** of 100 real cases; **26** of 100 flags are real; flags 17 % of
  all cases.
- **`impact`** — catches **54** of 100 real cases; **26** of 100 flags are real; flags 8 % of all
  cases.

### Two things this table says that AUC did not

**Every operating point is dominated by false alarms.** At the balanced point all three tasks sit
at 24–26 % precision, meaning **roughly 3 of every 4 alerts are false**. That is still 2–8× better
than flagging at random (base rates 12.4 / 5.4 / 3.25 %), and `impact`'s AUC of 0.93 is real — but
"ranks well" and "usable as an alert without triage" are different claims, and only the first is
supported.

**80 % precision is unattainable for `shortage` and `impact` at any usable threshold.** The best
available while still catching at least 5 in 100 is 53.9 % and 56.8 %. For `delay` 80 % precision
*is* reachable — but only by catching **4.6 of every 100 real cases**, i.e. missing 95 % of them.
There is no threshold on any of these tasks that is simultaneously precise and thorough.

*(The high-precision fallback is the most precise threshold that still catches ≥ 5 % of real
cases. Unconstrained, the precision maximum sits where a cell flags one or two items — precision
estimated on n≈1, recall ≈ 0 — which is an artifact, not an operating point.)*

---

## 2. World-to-world spread at the balanced point (item 3)

Per-dataset-seed-world means at the single balanced threshold above.

| task | metric | d42 | d43 | d44 | d45 | d46 | mean | **spread** |
|---|---|---|---|---|---|---|---|---|
| **delay** | recall | 0.943 | 0.325 | 0.785 | 0.776 | 0.402 | 0.646 | **0.618** |
| | precision | 0.267 | 0.216 | 0.254 | 0.254 | 0.224 | 0.243 | 0.051 |
| **shortage** | recall | 0.798 | 0.531 | 0.909 | 0.800 | 0.322 | 0.672 | **0.588** |
| | precision | 0.221 | 0.282 | 0.206 | 0.238 | 0.329 | 0.255 | 0.123 |
| **impact** | recall | 0.545 | 0.644 | 0.741 | 0.516 | 0.246 | 0.539 | **0.495** |
| | precision | 0.269 | 0.240 | 0.216 | 0.238 | 0.321 | 0.257 | 0.105 |

**This is the largest caveat in the report.** One fixed threshold produces recall anywhere from
**32 to 94 out of 100** on `delay`, depending only on which world it runs in. Precision is far
more stable (spread 0.05–0.12). So a fixed cutoff buys a fairly predictable *false-alarm rate* and
an almost unpredictable *catch rate*.

The "mean recall 64.6" in §1 is therefore an average over worlds that individually range from 32
to 94 — it should never be quoted as what the model will catch on a new world.

### Does the threshold at least transfer?

Choosing the balanced threshold on 4 worlds and scoring it on the held-out 5th, rotated:

| task | F1 in-sample | F1 held-out | drop | threshold spread across folds |
|---|---|---|---|---|
| delay | 0.345 | 0.322 | +0.023 | 0.0223 |
| shortage | 0.332 | 0.314 | +0.017 | 0.0177 |
| impact | 0.314 | 0.302 | +0.012 | 0.0196 |

The threshold itself transfers reasonably — F1 loses only 0.012–0.023 out of sample, and the
chosen threshold moves by ~0.02 between folds. The instability is in *what a fixed threshold
catches per world*, not in *where the optimum sits*.

---

## 3. Raw vs corrected (item 4)

### At matched percentile cutoffs — identical, as required

Every correction in the chain is strictly monotone, so flagging the top-K % of cases selects the
**same set** whether you rank by the raw score or the corrected probability. Verified per cell at
K ∈ {0.1, 0.5, 1, 2, 5, 10, 20, 50} %, across all 25 cells:

| task | max \|Δ precision\| | max \|Δ recall\| |
|---|---|---|
| delay | **0.00e+00** | **0.00e+00** |
| shortage | **0.00e+00** | **0.00e+00** |
| impact | **0.00e+00** | **0.00e+00** |

Exactly zero. The calibration work changed no ranking and therefore no percentile-based decision.

### At a fixed probability value — different, and this is not a contradiction

| task | operating point | corrected threshold | raw threshold | corrected N/M | raw N/M |
|---|---|---|---|---|---|
| delay | balanced | 0.1440 | 0.5201 | 64.6 / 24.3 | 67.2 / 25.7 |
| | high-recall | 0.1283 | 0.4872 | 80.0 / 21.6 | 80.0 / 23.2 |
| | high-precision | 0.2877 | 0.7000 | 4.6 / 80.1 | 4.3 / 80.2 |
| shortage | balanced | 0.1176 | 0.6058 | 67.2 / 25.5 | 61.8 / 27.2 |
| | high-recall | 0.1056 | 0.5729 | 80.0 / 21.9 | 80.0 / 21.9 |
| | high-precision | 0.1999 | 0.7357 | 7.1 / 53.9 | 5.2 / 53.7 |
| impact | balanced | 0.1412 | 0.6383 | 53.9 / 25.7 | 52.8 / 25.7 |
| | high-recall | 0.1003 | 0.5832 | 80.0 / 18.0 | 80.1 / 17.9 |
| | high-precision | 0.3950 | 0.8042 | 5.3 / 56.8 | 5.7 / 57.4 |

**Why the thresholds differ:** the correction moved which raw score maps to which stated
probability. A raw 0.52 on `delay` now correctly reads as 0.14. Both cutoffs select nearly the
same cases; only the number you write on the dial changed. Reading "the raw threshold is 0.52 and
the corrected one is 0.14" as a performance difference would be a misreading — it is the same
decision expressed in honest units.

**Why the resulting N/M still differ slightly** (e.g. `delay` balanced 64.6/24.3 vs 67.2/25.7),
despite strict monotonicity: the analytic correction uses each **world's own train-set base rate**
(chosen in `reports/calibration_correction_1.md` to keep the step leak-free). That makes the
raw→corrected map *world-specific*, so one fixed corrected threshold corresponds to a slightly
different raw cutoff in each world. Within any single cell the two are identical at matched rank —
that is what the 0.00e+00 above proves. Across worlds at one fixed number, they need not be.

### A side effect of that choice, reported because it is operationally relevant

| task | recall spread across worlds — raw | — corrected | precision spread — raw | — corrected |
|---|---|---|---|---|
| delay | 0.316 | **0.618** | 0.148 | **0.051** |
| shortage | 0.310 | **0.588** | 0.113 | 0.123 |
| impact | 0.316 | **0.495** | 0.043 | 0.105 |

Thresholding the *corrected* probability at a fixed value gives **more** variable recall across
worlds than thresholding the raw score, because the correction deliberately injects each world's
own base rate — worlds differ in π (delay's train π spans 0.123–0.191), so equalising what a
number *means* necessarily unequalises what a fixed cutoff *catches*. `delay`'s precision becomes
much more consistent in exchange (0.148 → 0.051).

This is not an argument against the calibration — the corrected numbers are the honest ones, and
§4 of `calibration_correction_2.md` showed they are accurate over 89–99 % of the distribution. It
is an argument for **thresholding on percentile, not on probability, when consistent alert volume
across worlds matters.** Percentile thresholding is exactly the operation the 0.00e+00 check above
proves is unaffected by any of this.

---

## 4. Rank-only band flags (item 5)

`reports/calibration_correction_2.md` §4 established where each task's probability is trustworthy
as an absolute number: `delay` and `shortage` below p = 0.20, `impact` below p = 0.40.

| task | operating point | threshold | in band? |
|---|---|---|---|
| delay | balanced | 0.1440 | absolute ✓ |
| delay | high-recall | 0.1283 | absolute ✓ |
| **delay** | **high-precision** | **0.2877** | ⚠ **above 0.20 — RANK-ONLY** |
| shortage | balanced | 0.1176 | absolute ✓ |
| shortage | high-recall | 0.1056 | absolute ✓ |
| shortage | high-precision | 0.1999 | absolute ✓ (0.0001 below the boundary) |
| impact | balanced | 0.1412 | absolute ✓ |
| impact | high-recall | 0.1003 | absolute ✓ |
| impact | high-precision | 0.3950 | absolute ✓ (0.005 below the boundary) |

**`delay`'s high-precision point at 0.2877 sits in the rank-only band.** The stated threshold value
carries the same caveat as any absolute reading there: the model's "0.29" in that region
corresponds to a materially higher true rate (§1 of `calibration_correction_2.md` measured the
20–30 % band at 0.2347 predicted vs 0.3305 observed). The *measured* precision and recall at that
cutoff — 80.1 and 4.6 out of 100 — are empirical counts and remain valid; what is not trustworthy
is reading "0.2877" as a probability. The ranking the cutoff relies on is unaffected.

Two others sit just inside the boundary — `shortage`'s high-precision point at 0.1999 (0.0001
below) and `impact`'s at 0.3950 (0.005 below). Both should be treated as borderline rather than
comfortably inside.

---

## 5. What this means in practice

- **For triage / worklists** — use these operating points and expect ~3 in 4 flags to be false at
  the balanced point. That is normal and useful at these base rates, provided a human or a cheap
  second check follows the flag. It is not usable as an automated action trigger.
- **For "don't miss anything"** — the high-recall rows catch 80 in 100, at the cost of flagging
  46 % (`delay`), 22 % (`shortage`) or 15 % (`impact`) of everything.
- **For "don't cry wolf"** — only `delay` reaches 80 % precision, and only by missing 95 % of real
  cases. `shortage` and `impact` cannot reach it at all.
- **If alert volume must be consistent across worlds** — threshold on percentile, not on
  probability. Percentile cutoffs are provably unaffected by any correction in the chain
  (0.00e+00), and avoid the recall instability in §2.
- **Do not quote the mean recall as an expectation for a new world.** The per-world range
  (32–94 on `delay`) is the honest statement.

---

## 6. Scope

This is a read-only translation. It does not change any AUC-based pass/fail status — the standing
verdicts from [reports/new_nodes_fix1.md](reports/new_nodes_fix1.md) (`shortage` PASS, `impact`
PASS, `delay` still failing on its dataset-seed floor) are untouched — and it does not modify any
shipped calibration correction. No accuracy figure is computed for any task.

**Limitations.** Thresholds are selected on test predictions, so the §1 numbers carry the
in-sample optimism quantified in §2 (F1 drop of 0.012–0.023 out of fold); the held-out figures are
the ones to plan against. `shortage`'s counts are label rows, not distinct nodes (the documented
Product×Warehouse coarsening in `ml/models/heads.py`). All figures are cell-averages over the
25-cell grid at the v1 preset.

## 7. Reproducing

```
python3 ml/operating_points.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

Runs from the cached predictions; no forward passes.
