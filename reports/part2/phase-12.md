# Phase 12 — six tests, reordered for gating and parallelism

**Audience:** whoever owns the client-facing claims, and whoever builds Waves B and C.
**Measured on:** `db/gen_v8/seed_1001` unless a world is named. v6/v7 appear only where a stored
v6/v7 artifact is the comparator (A4's regression and like-for-like figure) and, in Wave B, for B1.
Selection on validation only. No training in Wave A.
**Companions:** `reports/part2/phase-11c.md`, `phase-11b.md`, `phase-11a.md`, `phase-0-1-v8.md`,
`reports/part1/phase-8.md`, `phase-5.md`, `phase-11-pilot.md`, `docs/specs/lateness_metric.md`.
**Path note:** the brief names `reports/part2/observation1.md`; it exists at `results/observation1.md`
(untracked). It was read there and not moved.

**Status of this document: WAVES A AND B — AT THE HARD CHECKPOINT.** Wave A was reported at its
boundary (commit `4ef6f51`). Wave B follows in §3. Wave C has **not** started. The brief makes it
conditional on B2's ratio, and that decision is §3.3's.

---

## 1. Verdict per test

| test | verdict | the number that decided it |
|---|---|---|
| **A1 — gate for C1** (seasonal residual in arrival) | **GATE CLOSED → C1 CANCELLED.** A seasonal tilt exists in the arrival-week residual, in the direction the generator's path predicts. It does not reach the lateness metric, and a month-level correction is worth nothing measurable | val-fitted month shift moves test lateness ROC-AUC by **+0.00015** (per-seed −0.0014 … +0.0022), inside a full-fold seed spread of **0.0065**; one month's minimum detectable ROC-AUC effect is **0.042**, larger than the **0.033** the entire remaining headroom would give if it all sat in a single month |
| **A2 — gate for C2** (fill interior error by month) | **GATE CLOSED → C2 CANCELLED; C4 REDIRECTED** to A2.4's proposals (written, not built) | after normalising for where the interior mass lives, the head's error density per month runs **0.83–1.21×**, while one cell ([0.95, 1)) is predicted at **2.9×** its frequency and 4 of 20 cells carry **61%** of the interior error. The failure is per-BIN, and it is present in **every** month at 3.0–6.6× that month's noise floor |
| **A3 — Test 1.2 redesign** | **Step 1 PASSES (re-verified); the ablation is RESPECIFIED** as a part-relation ablation at fixed depth | part degree median **26**, range **12–45** (supplier 38, 21–55). Removing the part relation leaves **131,072** (arrival, 24.6% of the model) and **16,640** (capacity, 14.2%) parameters without gradient. Cost **10 cells, ≈2.5 h on one queue** |
| **A4 — Test 5.1** price_weight = 0 | **BUILT. Invariance to the shortage cost PROVED on data** (0 of 183 part-plants change winner across 100 → 10,000; the same check flags 38 at weight 1). **The unlock is small** | quotable rises **25% → 31.7%** on v6/v7 (3 seeds) and **4.8% → 9.5%** on v8 (5 seeds), but **actionable** recommendations barely move: **17.5% → 19.2%** (v6/v7), **3.2% → 7.9%** (v8) |
| **A5 — one order policy** | **SPECIFIED**, with three corrections to the brief's rule, each measured | the brief's trigger double-counts lead-time demand (ROP − SS is **1.6–2.3×** planning-lead demand); the hazard head's T is **not** a lead time; the simulation omits an open pipeline that already covers **0.55–1.03×** the 13-week requirement |

| **B1 — Test 3.1** (conformal widening, v6/v7) | **FAILS, and the premise does not hold.** CQR hits nominal on its own window (16/16) and makes the next window **worse** for h⁴ (in band 9 → **5** of 16). ρ does not move | ρ(P90 exceedance, level gap) **0.65 → 0.65** (h⁴), 0.80 → 0.80 (h⁰), 0.83 → 0.83 (B5). The median lags the level: P50 moves **51–74%** of the label shift, and that lag explains post-CQR exceedance at **Spearman 0.96** |
| **B3 — inter-plant transfers** (after the checkpoint) | **BUILT, DOES NOT CLOSE THE GAP.** Frequency moves 0.04; magnitude overshoots the other way | ratio **1.435× [1.424, 1.447]** against 1.475× without; shortfall when short 298 → **20** against 91 observed. Part-plant rebalancing cannot reproduce the data's channel-level rescues |
| **B2 — the order policy** | **BUILT, WIRED INTO BOTH CONSUMERS, VALIDATED ON OBSERVED 2025 OUTCOMES. The ratio closes most of the way; it is NOT calibrated** | part-plant-weeks below SS **5.29× → 1.475× [1.464, 1.487]** (5 fill-head seeds); replacement **82.5% → 100.3%**; shortfall when short 564 → 298 against 91 observed. The residual points at **inter-plant transfers**, which the simulation omits (20% of consumption flows) |

**C1 and C2 are both cancelled by their gates. C3 runs regardless** (Wave C). C4 is replaced by
A2.4's proposals. **Wave C has not started:** B2's ratio did not reach 1.0, so the brief's checkpoint
applies (§3.3).

Three items found in Wave A change earlier reports' numbers, not only this phase's:

- **The 0.0445 / 0.0049 fill figures are v6, not v8** (Phase 5 §6.4). v8's head sits at
  0.057–0.084 on its own training marginal (deviation 86).
- **Phase 11's 25% counted duplicate candidates as runners-up.** Every exact top-two tie in that
  sweep, 29 of 60 on v6 and 30 of 60 on v7, is the winning split under another name (deviation 90).
- **On v8, 57 of 120 part-plants have no feasible allocation candidate, the incumbent included.**
  The qualification constraint, inert on v6/v7, rejects the status quo (deviation 91).

---

## 2. Wave A

### 2.1 A1 — does arrival carry a seasonal residual? (gate for C1)

`ml/eval/phase12_a1_seasonal.py` → `ml/artifacts/phase12_a1.json`, commit `f05d755`, clean.
It uses the five shipped v8 h⁴ arrival bundles (`v8_lite_h4_lr0.00025_s{7,17,27,37,47}`) and their
stored test (2025) and validation (2024) predictions, scored under the adopted t0 reference
(`docs/specs/lateness_metric.md`, offset +4.571 wk).

**Reproduction first.** The fold-level headline had to come back before any per-month number was
trusted. It came back exactly: per seed 0.70905 / 0.70904 / 0.71298 / 0.70772 / 0.70645, mean
**0.70905**. The script asserts this.

**The design problem, found before any result.** v8's label snapshots are monthly and sparse. The
2025 test fold is **nine snapshots of 4,000 rows**: 2025-01-06, 02-17, 03-31, 05-12, 06-23, 08-04,
09-15, 10-27 and 12-08, so April, July and November are absent. Validation 2024 has eight. **On one
fold, "calendar month" and "snapshot" are the same variable.** A month effect on the test fold
cannot be told apart from a one-off shock to that snapshot. The only thing that separates
seasonality from a snapshot shock is **repetition across years**: a seasonal residual must recur in
the same calendar month of 2024 and 2025. Five months are shared (Feb, Mar, Jun, Sep, Oct).

Per row, on the uncensored rows (the lateness metric's population):
`resid = Y − P` in weeks; `p_late = P(R < T ≤ 12 | T ≤ 12)` from the survival curve;
`e_late = 1{Y > R} − p_late`.

#### A1.3 — power, stated before the result

| quantity | value |
|---|---|
| test-fold rows / scored (uncensored) | 36,000 / 18,534 |
| **December rows / scored** | **4,000 / 2,058** (the brief's "~3,000": deviation 85) |
| full-fold lateness ROC-AUC seed spread, 5 seeds | **0.0065** (the brief's "~0.003": deviation 85) |
| per-month ROC-AUC bootstrap SE | 0.0105–0.0122 |
| per-month ROC-AUC seed spread | 0.0060–0.0197 |
| **minimum detectable one-month ROC-AUC effect** (80% power, α = 0.05) | **0.042–0.048** (December 0.042) |
| minimum detectable one-month residual, vs the rest | 0.14–0.17 weeks (Bonferroni over 9 months: 0.22) |
| minimum detectable one-month `e_late` | 0.027–0.031 |
| **cross-year replication**: shared months / critical r at 0.05 | **5 / 0.754** |
| its power at a true between-year ρ of 0.5 / 0.7 / 0.9 | **0.28 / 0.49 / 0.87** |
| headroom, h⁴ over b5flat LightGBM | **0.0037** ROC-AUC |
| that headroom if concentrated in 1 / 2 / 3 of 9 months | 0.033 / 0.017 / 0.011 per month |

Every per-month SE is from a row bootstrap. Rows in one snapshot share its shocks, so these are
**lower bounds**, and every minimum detectable effect above is optimistic.

**What the power check says on its own:** a per-month lateness effect worth acting on cannot be
detected by this design. If the entire remaining headroom (0.0037) sat in a single month, it would
be 0.033 there, below the 0.042 this sample can see. The cross-year test, the only one that
separates season from snapshot, has five points and a coin-flip's power at ρ = 0.7. That is a
finding in itself.

#### A1.2 — the result anyway (test fold, seed-mean prediction, centred on the fold mean)

| month | scored | late rate | resid, weeks [95% CI] | `e_late` [95% CI] | ROC-AUC, 5-seed mean [range] | generator lag-season |
|---|---|---|---|---|---|---|
| Jan | 1,989 | 0.498 | −0.054 [−0.161, +0.055] | +0.014 [−0.006, +0.035] | 0.7138 [0.7044, 0.7241] | −0.36 |
| Feb | 2,219 | 0.409 | −0.185 [−0.285, −0.084] | −0.042 [−0.060, −0.023] | 0.7001 [0.6929, 0.7075] | −0.36 |
| Mar | 1,949 | 0.519 | +0.064 [−0.048, +0.163] | +0.018 [−0.002, +0.038] | 0.7351 [0.7301, 0.7389] | 0.00 |
| May | 2,091 | 0.458 | −0.167 [−0.268, −0.067] | −0.008 [−0.028, +0.011] | 0.6989 [0.6934, 0.7026] | 0.00 |
| Jun | 1,919 | 0.500 | +0.150 [+0.042, +0.255] | +0.025 [+0.005, +0.045] | 0.7107 [0.7069, 0.7140] | 0.00 |
| Aug | 1,927 | 0.459 | −0.070 [−0.185, +0.042] | −0.002 [−0.024, +0.018] | 0.6958 [0.6911, 0.7038] | 0.41 |
| Sep | 1,982 | 0.524 | +0.195 [+0.095, +0.302] | +0.028 [+0.009, +0.048] | 0.7000 [0.6947, 0.7071] | 1.41 |
| Oct | 2,400 | 0.418 | −0.061 [−0.152, +0.029] | −0.048 [−0.067, −0.030] | 0.7052 [0.7025, 0.7084] | 1.00 |
| Dec | 2,058 | 0.515 | +0.169 [+0.066, +0.272] | +0.031 [+0.011, +0.051] | 0.7089 [0.7029, 0.7216] | 1.00 |

| test | result |
|---|---|
| null of no **snapshot** structure, one-way ANOVA, test | resid F = 7.50, p = 5×10⁻¹⁰; `e_late` F = 9.23, p = 9×10⁻¹³ — **rejected** |
| same, validation | resid F = 14.2; `e_late` F = 5.16 — rejected |
| seed consistency of the month profile (pairwise r, 5 seeds) | mean 0.76, min 0.58 |
| **cross-year replication, 5 shared months** | resid r = **−0.19** (p = 0.63); `e_late` r = **+0.12** (p = 0.43); ROC-AUC r = **−0.43** (p = 0.78) — **not replicated** |
| **pooled contrast against the generator's lagged seasonal shape**, 17 snapshots | resid r = **+0.58**, permutation p = **0.008**, slope 0.156 wk per unit; `e_late` r = +0.31, p = 0.10 |
| **CEILING: a month-level seasonal shift fitted on VALIDATION, applied to test** | Δ ROC-AUC mean **+0.00015**, per seed −0.0013 / +0.0022 / −0.0014 / +0.0007 / −0.0006; shifted band [0.70580, 0.71159] against base [0.70645, 0.71298] — **overlapping, no effect** |

Snapshots differ from one another, which is certain. Whether that is **seasonality** is only
partly answerable. Against the one shape fixed in advance by the generator, the arrival-week
residual does tilt seasonally (p = 0.008). The lateness error does not reach significance. And
correcting the tilt, fitted on validation, buys nothing on test.

#### A1.4 — consistency with the generator's path

Seasonality reaches arrival only indirectly: demand seasonality (`season()`: festive Aug–Nov,
fiscal Mar, monsoon Jun–Aug, winter Dec–Jan negative) → monthly utilisation `util_hist` →
`sup_load_prev` (**last** month's) → the lead multiplier `+0.85·clip(sup_load_prev − 0.70, 0, 2)`
(`generator_v8.py:576`).

| | measured |
|---|---|
| **raw** observed lead by PO month, 2019–2023, correlation with the lagged shape | **+0.854** (same-month shape +0.683: the lag fits better, as the path predicts) |
| raw seasonal amplitude in lead | **2.60 weeks** (Jan 4.86 → Sep 7.45) |
| between-year correlation of the raw monthly pattern | mean 0.55 (0.16–0.90) |
| model residual's range across months | 0.38 wk (test), 0.56 wk (val) |
| model residual against the lagged shape | +0.57 (test), +0.62 (val), pooled +0.58 (p = 0.008) |

**Consistent with the path, and mostly already absorbed.** The raw signal is strong, lagged by one
month, and in the generator's direction. The encoder reads supplier load directly, and about
0.16 week per unit of the lagged shape survives in the residual, against a raw swing of 2.6 weeks.
The TCN's 64-week receptive field does not need a seasonal branch to reach this signal. It already
has the proximate driver as an input.

**GATE: CLOSED. C1 is CANCELLED.** The monthly structure that exists is in the residual level, not
in lateness ranking. A month-level correction measures +0.00015. Per-month effects worth acting on
sit below this design's detection limit, and the whole extractable headroom over b5flat LightGBM is
0.0037, whatever the architecture.

### 2.2 A2 — is fill's interior error concentrated by month? (gate for C2)

`ml/eval/phase12_a2_fill_months.py` → `ml/artifacts/phase12_a2.json`, commit `7cc20e6`, clean.
The five v8 fill heads (`v8_none_h0_lr0.000125_s{7,…,47}`) are run **in inference** over the
training fold. Training-fold predictions were never stored, and the diagnosis lives there.
**Inference reproduces every bundle's stored test predictions exactly** (max |Δ| = 0.0, asserted),
so the training-fold numbers come from the model that shipped.

**First, a correction.** The brief's "0.0445 against LightGBM's 0.0049" is Phase 5 §6.4's **v6**
measurement (v7 was 0.0537 / 0.0235). v8's own figures:

| raw, un-recalibrated | head, 5 seeds | b5flat22 LightGBM, 5 fits |
|---|---|---|
| interior Σ\|·\|, **training** fold | **0.0572–0.0842** | not measured (see below) |
| total Σ\|·\|, training | 0.0636–0.0949 | — |
| interior Σ\|·\|, validation | 0.0628–0.0921 | **0.0180–0.0183** |
| interior Σ\|·\|, test | 0.0764–0.1039 | 0.0247–0.0257 |

LightGBM's training-fold predictions were never stored. Refitting it is training, so its v8
training marginal is **not** measured in this wave.

#### A2.2 — bin occupancy, actual counts

| fold | rows | at 0 | at 1.0 | interior | rows per interior cell, min / median / max | share per cell, min / max |
|---|---|---|---|---|---|---|
| train | 176,000 | 2.07% | 78.61% | 19.32% | 729 / 968 / 5,745 | 0.41% / 3.26% |
| val | 32,000 | 2.25% | 77.59% | 20.16% | 130 / 196 / 1,156 | 0.41% / 3.61% |
| test | 36,000 | 2.16% | 74.93% | 22.92% | 183 / 242 / 1,387 | 0.51% / 3.85% |

Training counts by cell (0 … 21): 3,644 · 2,295 · **5,745 · 4,875 · 3,203 · 2,427** · 2,024 · 1,546 ·
1,272 · 1,138 · 957 · 978 · 930 · 808 · 861 · 729 · 760 · 798 · 874 · 854 · 929 · 138,353.
"Under 1% per bin" holds for 15 of 20 cells. The low-fill cells [0.05, 0.25) hold 1.4–3.3% each
(deviation 87).

#### A2.1 — decomposition by calendar month

With mean predicted cell mass p_c and observed frequency o_c, the interior error is
E = Σ_{c=1..20} |p_c − o_c|. Each month's contribution
K_m = Σ_c sign(p_c − o_c) · w_m (p_mc − o_mc) **sums exactly to E** (asserted to 10⁻¹⁰; a first run
fired on float32 means, deviation 97). Concentration is judged against a **snapshot-permutation
null**: training snapshots, 3–4 per calendar month, are reassigned to months at random, each
snapshot kept intact.

**The detector, demonstrated both ways before it was believed:**

| constructed input | statistic | null p95 | p | result |
|---|---|---|---|---|
| error injected into October (60% of interior mass moved to the atom) | 0.1202 | 0.0759 | 0.000 | **FIRES** |
| error made exactly flat (each month = its observed freq + the fold-wide error) | 6×10⁻¹⁹ | 0.0266 | 1.000 | **silent** |

**A would-be twelfth vacuous gate, caught.** On the **test** fold each month is exactly one snapshot
with an equal row share, so a snapshot permutation only relabels months and the statistic is
invariant. The first run returned null = observed to 10⁻¹⁶. **It cannot fire, so no test-fold
concentration p-value is reported** (deviation 88).

**Training fold, seed-mean prediction, E = 0.0475:**

| month | rows | row share | **share of E** | interior-labelled share of rows | error density per interior row | own-month Σ\|·\| | its noise floor, mean / p95 |
|---|---|---|---|---|---|---|---|
| Jan | 16,000 | 0.091 | 0.077 | 16.4% | 1.00 | 0.040 | 0.010 / 0.013 |
| Feb | 12,000 | 0.068 | 0.054 | 17.0% | 0.89 | 0.039 | 0.013 / 0.018 |
| Mar | 12,000 | 0.068 | 0.052 | 16.5% | 0.89 | 0.041 | 0.013 / 0.017 |
| Apr | 16,000 | 0.091 | 0.081 | 14.2% | 1.21 | 0.046 | 0.011 / 0.014 |
| May | 16,000 | 0.091 | 0.087 | 15.7% | 1.17 | 0.047 | 0.011 / 0.014 |
| Jun | 12,000 | 0.068 | 0.103 | 27.9% | 1.04 | 0.093 | 0.014 / 0.018 |
| Jul | 16,000 | 0.091 | **0.142** | 27.5% | 1.10 | 0.080 | 0.013 / 0.016 |
| Aug | 16,000 | 0.091 | 0.113 | 27.8% | 0.86 | 0.069 | 0.014 / 0.018 |
| Sep | 12,000 | 0.068 | 0.095 | 27.2% | 0.99 | 0.070 | 0.015 / 0.020 |
| Oct | 16,000 | 0.091 | 0.106 | 21.8% | 1.03 | 0.056 | 0.012 / 0.015 |
| Nov | 16,000 | 0.091 | 0.049 | 11.7% | 0.89 | 0.069 | 0.011 / 0.014 |
| Dec | 16,000 | 0.091 | 0.043 | 11.0% | 0.83 | 0.054 | 0.011 / 0.014 |

| concentration test | statistic | null mean / p95 | p |
|---|---|---|---|
| against **row share** | 0.1185 | 0.0448 / 0.0694 | < 0.002 |
| against **interior-mass share** | 0.0127 | 0.0063 / 0.0101 | 0.002 |

Reading it:

1. **The raw concentration is real and is where the data is.** June to October hold 56% of the error
   on 41% of the rows, because those months have 22–28% interior-labelled rows against 11–17%
   elsewhere. Seasonal supplier strain produces more partial fills.
2. **Normalised for that, the head fails at nearly the same rate everywhere.** Error per interior row
   runs 0.83–1.21× across months. The residual departure from flat is statistically detectable
   (p = 0.002) but small, and not seed-stable. April's share, for example, spans 0.056–0.090 across
   the five seeds against its 0.067 interior share.
3. **Every month is 3.0–6.6× above its own noise floor.** The failure is not located in some months
   and absent from others.
4. **Where it IS concentrated: bins.**

| cell | [0.05, .10) | [.10, .15) | [.15, .20) | … | **[0.95, 1)** |
|---|---|---|---|---|---|
| head, predicted mass (train) | 0.0256 | 0.0222 | 0.0245 | | **0.0154** |
| observed (train) | 0.0326 | 0.0277 | 0.0182 | | **0.0053** |
| difference | −0.0070 | −0.0055 | +0.0063 | | **+0.0101 (2.9×)** |

These four cells carry 0.0289 of E = 0.0474, which is **61%**. The [0.95, 1) cell beside the
complete-fill atom is the same RPS artefact Phase 5 §6.3 found on v6. It reproduces on v8.

**Ceiling of the month-weighting premise.** Suppose weighting brought every month's error density
to exactly 1.0, which it has no mechanism to do. E would fall by the excess above 1.0: April 0.0144,
May 0.0127, June 0.0040, July 0.0124 and October 0.0030 of E's shares, **4.7% of E, ≈ 0.002**. The
head-to-LightGBM interior gap on validation is **≈ 0.045–0.074**.

#### A2.3 — the precedent

Phase 5 §6.5 ablated three losses (RPS, cross-entropy, RPS + CE). Every ECE difference landed inside
band, and CRPS did not move. Sample weighting is the same family of intervention: it changes which
rows dominate the gradient, not the head's ability to allocate mass among sparse neighbouring cells.
**A2.1 gives no reason to expect a different outcome.** It gives a reason to expect the same one.
The error is a per-cell shape failure present in every month, and month weights change neither cell
sparsity nor the early-stopping-on-CRPS mechanism Phase 5 §6.4 identified.

**GATE: CLOSED. C2 is CANCELLED; C4 is REDIRECTED.**

#### A2.4 — what to build instead (PROPOSALS, not built in this phase)

1. **Reparameterise the interior as a continuous density.** Two atoms (0, 1) plus a Beta(α, β) on
   (0, 1), with three softmax weights and two positive shape parameters in place of 20 independent
   cells. It is smooth by construction, so it cannot place 2.9× mass in one cell beside its
   neighbours, and every interior row informs every interior cell's mass. Score with exact CRPS,
   which is closed-form for atom + Beta mixtures. Risk: a unimodal Beta may under-fit the low-fill
   bulk at [0.05, 0.25); a two-component Beta mixture is the fallback.
2. **Per-BIN, not per-month, weighting.** Up-weight RPS terms at the edges bounding the sparse cells,
   or add a marginal-matching penalty `λ·Σ_c |mean_batch p_c − o_c^train|`. This targets exactly the
   cells the decomposition names. It is still the sample-weighting family, so it inherits A2.3's
   warning, and it should be tried only if (1) is not.
3. **Finish the LightGBM serving loader (deviation 46)** and ship the arm that already reproduces
   its marginal: b5flat22 interior Σ|·| **0.018** on validation against the head's 0.063–0.092.
   This is the cheapest option and needs no new model. `shipped.json` already names this arm.
   What is missing is a loader that can serve it.

**Recommended order: 3, then 1.** Option 3 closes a configuration-describes-what-does-not-run defect
regardless of any model result.

### 2.3 A3 — Test 1.2: Step 1 already done; the ablation respecified

**A3.1 — re-verified, not re-derived.** `reports/part2/phase-0-1-v8.md` §2 records part-node degree
(channels per part) **median 26, range 12–45**, against supplier **38 (21–55)** and plant
2,285 (2,227–2,362). The shuffle-control table in `phase-11a.md` §2.2 records the same distribution
independently (part 12 / 26 / 45, supplier 21 / 38 / 55), measured on the graph the model consumes.
**26 clears the 15–20 threshold. Step 1 passes; Step 3 does not apply.**

**A3.2 — the ablation as written confounds path with depth.** "h⁰ vs h¹ at the part node"
compares zero rounds with one round, and h¹ already contains the part path along with every other
one-hop path. It measures depth. **Specified instead: a RELATION ABLATION.** Full h⁴ against h⁴ with
the **part relation removed**, at the same depth (4), same width, same learning rate (not retuned),
same seeds.

| | arrival (SHARE-lite, 4 layers) | capacity (HeteroMP, 2 rounds) |
|---|---|---|
| model parameters, full | 533,004 | 117,123 |
| encoder parameters, full | 468,608 | 66,432 |
| what the part relation owns | 2 directed types (channel→part, part→channel) × 4 layers × 128² | up + down per round × 2 rounds × (64² + 64) |
| **parameters left without gradient when part is removed** | **131,072 (24.6% of the model)** | **16,640 (14.2%)** |

**Implementation, specified so the residual is stated rather than hidden.** Keep `n_rel` unchanged
(6 for SHARE-lite, 3 for HeteroMP) and remove the part edges from the edge list or relation index.
The nominal parameter count is then **identical** and the part weights receive **no gradient**, so
the effective difference is the figure above. **Do not re-index the remaining relations.**
HeteroMP's `zip(rel_index, rel_size)` would silently route plant through the part relation's
weights. Pass part as an empty relation, or mask by relation id.

- `drop_relation` must enter `ml/artifact_identity.py`'s `config_name` / `identity_of`, as
  `graph_shuffle` does (standing rule 2), and both evaluation paths must pass it through.
- **The ablation must be shown capable of failing.** An assertion that relation ids {1, 4} carry
  **zero** edges in the ablated arm must **fire on the full graph**. A second check must confirm
  supplier and plant edge counts are unchanged (48,216 → 32,144 edges, exactly 16,072 fewer per
  direction).
- **A parameter-exact alternative, recorded for C3.2.** A **part-only shuffle**, permuting the part
  relation within type with degree preserved, keeps every parameter live and the depth identical,
  with a residual difference of **0**. It isolates part-*identity* information where removal
  isolates part-*path* information. Removal is the brief's test and stays primary. The part-only
  shuffle is the natural control if removal shows a disjoint gap.

**A3.3 — why the hypothesis is sound, and where it applies.** At the **channel** readout,
attention is a no-op: median channel degree is 3, one neighbour per relation, and SHARE-lite's
softmax runs over each destination's whole neighbourhood. At the **part node**, 26 channels compete
for weight, so attention has something to choose between. The hypothesis needed a sharper test, not
a different one. **One qualification the brief does not state:** capacity's shipped encoder is
**HeteroMP, which has no attention**, only learned per-relation means through a gate. On capacity
the ablation tests whether the part *path* carries information. It cannot test attention. The
attention-at-the-part-node reading applies to **arrival only**.

**A3.4 — cost.** Only the ablated arm trains. The full h⁴ arms exist at five seeds for both tasks
(`v8_lite_h4_…_s{7,…,47}`, `v8_mp_h4_…_s{7,…,47}`).

| | measured per cell (the five full-arm bundles) | ×5 seeds |
|---|---|---|
| arrival h⁴ | 1,030–1,157 s (≈18 min), 53–60 epochs, patience-stopped | ≈ 90 min |
| capacity h⁴ | 655–940 s (≈12 min), 44–64 epochs, patience-stopped | ≈ 62 min |
| **total** | | **10 cells ≈ 2.5 h on one queue, ≈ 1.5 h on two** |

Removing a third of the edges should make the ablated arm slightly faster. That is not claimed until
measured. **No training was run in Wave A.**

### 2.4 A4 — Test 5.1: price_weight = 0, and the consequence

`ml/opt/allocation.py` (`ReducedScorer(price_weight=0.0)`), `ml/opt/sensitivity.py` and
`ml/opt/phase12_a4_price.py` → `ml/artifacts/phase12_a4.json`, commit `51ff706`, clean.

**A4.1 — built.** `ReducedScorer` takes `price_weight`, **default 0.0**.
`score = c_short·unmet + price_weight·purchase`. `purchase_cost` and `price_term` are **logged per
option at every weight**, including 0. Every result carries `decision_basis`, which at weight 0
reads **"price not weighted in this decision — most reliable supplier, not best value"**
(`price_label()`, A4.7). The allocation CLI's `--price-weight` defaults to 0 and prints the
decision basis. `sensitivity.evaluate` keeps `price_weight=1.0` as its own default so that Phase 11's
sweep still reproduces.

**A4.2 — the regression check, and why it is not the one the brief wrote.** The brief asks that
weight-0 recommendations be *identical to current output*. That cannot hold. Current output includes
the purchase term, and its winner flips with `c_short` for 10 to 15 part-plants per world, while at
weight 0 the winner provably cannot flip. The invariant that **can** be checked is the default-
preserving one. **At price_weight = 1.0 the new scorer must reproduce the stored Phase 11 sweep
exactly** (deviation 89).

| world | part-plants × costs × fields compared | differences |
|---|---|---|
| v6 | 60 × 5 × {winner, runner-up, margin, bands_overlap, survives} | **0 — PASS** |
| v7 | 60 × 5 × same | **0 — PASS** |

**Weight 0 against current output at ₹1,000 (a finding, not a defect):** the winner differs on
18 of 60 (v6), 10 of 60 (v7) and 20 of 63 (v8). Many of those are duplicate-tie relabellings (below).

**A4.3 — smoke test: price_weight > 0 behaves.** Across weights 0 → 0.01 → 0.1 → 1 → 10 → 10⁶, the
winning candidate's purchase cost is **non-increasing on 30/30 (v6), 30/30 (v7) and 13/13 (v8)**
part-plants. At 10⁶ the winner is the **cheapest feasible candidate on all of them**.

**A4.4 — the invariance to shortage cost, PROVED on data.** The Phase 11 sweep (₹100 / 300 / 1,000 /
3,000 / 10,000) is re-run at both weights with an invariance check that flags any part-plant whose
winner changes across the sweep. The check is shown capable of firing on the weight-1 sweep, where
winners are known to flip.

| world | seeds | flagged at **weight 1** (check must fire) | flagged at **weight 0** | flagged at weight 0, deduplicated |
|---|---|---|---|---|
| v6 | 3 | **10 of 60 — fires** | **0 of 60** | 0 |
| v7 | 3 | **13 of 60 — fires** | **0 of 60** | 0 |
| v8 | **5** | **15 of 63 — fires** | **0 of 63** | 0 |

**At price_weight = 0 the winning candidate is unchanged at every shortage cost for every
part-plant, on every world: 0 of 183.** The margins scale with `c_short` and so do the seed bands, so
band overlap is invariant too. The parameter that blocked allocation for three datasets drops out
of the ranking.

**A4.5 — what that unlocks.** A recommendation is **quotable** if its winner is stable across the
sweep and its margin over the runner-up survives the seed bands at every cost. **Actionable**
means quotable and not "keep the incumbent".

| world | seeds | | quotable, weight 1 | **quotable, weight 0** | actionable, w1 → **w0** |
|---|---|---|---|---|---|
| v6 | 3 | as Phase 11 counted | 18 / 60 (30%) | **18 / 60 (30%)** | 12 → **10** |
| v7 | 3 | as Phase 11 counted | 12 / 60 (20%) | **20 / 60 (33%)** | 9 → **13** |
| **v6 + v7** | 3 | as Phase 11 counted | **30 / 120 (25%)** — reproduces Phase 11 | **38 / 120 (31.7%)** | **21 → 23 (17.5% → 19.2%)** |
| v6 + v7 | 3 | duplicates collapsed | 53 / 120 (44.2%) | 63 / 120 (52.5%) | 21 → 23 |
| **v8** | **5** | as Phase 11 counted | 3 / 63 (4.8%) | **6 / 63 (9.5%)** | **2 → 5 (3.2% → 7.9%)** |
| v8 | 5 | duplicates collapsed | 7 / 63 (11.1%) | 10 / 63 (15.9%) | 2 → 5 |

**The duplicate-candidate defect (deviation 90).** `candidates()` emits five named splits, which
often coincide. "Cap any supplier at 70%" *is* the incumbent when no supplier exceeds 70%, and
"all to cheapest" *is* the incumbent when the cheapest already holds 100%. The winner's "runner-up"
is then itself under another name: margin 0, bands overlapping by construction, never quotable.
**Every exact top-two tie is such a duplicate:** 29 of 60 (v6), 30 of 60 (v7), 34 of 63 (v8), with
0 distinct splits tied. The sort is stable and the incumbent is listed first, so the incumbent also
"wins" every such tie. `evaluate(dedupe=True)` collapses identical splits and records aliases,
leaving 0 zero-margin ties on every world. `dedupe=False` stays the default so that Phase 11
reproduces. **Deduplication raises the quotable count but not the actionable one.** Every recovered
recommendation is "keep the incumbent".

**The honest reading of the unlock:**

- **Stability is now free.** Every winner is stable because the shortage cost no longer enters the
  ranking.
- **The binding limit is now the seed bands,** i.e. whether fill and strain signals separate
  suppliers beyond seed noise. They mostly do not.
- **Actionable recommendations move by two part-plants of 120 on v6/v7 and three of 63 on v8.**
  Price removal unblocks the *question*. It does not create signal.
- **v8's figure is lower than v6/v7's.** Its bands are 5-seed (wider), and it has the
  qualification problem below.

**A v8 finding outside the test's scope (deviation 91).** v8 has **3,827** multi-supplier
part-plants with forward requirement at t0 = 2025-06-30. Of the 120 selected by Phase 11's rule,
**57 have no feasible candidate at all, the incumbent included.** Example: P00001 @ PL01, incumbent
volume sits with SUP00006 ("not qualified, needs 75 days") and SUP00404 (171 days).
`constraint_report` rejects any split giving *any* share to a supplier `alternate_sources` marks
unqualified, **including volume that supplier already holds**. On v6/v7 every row read "qualified"
and the rule was inert. On v8 it binds and rejects the status quo. Either `alternate_sources` status
does not describe incumbents, or the constraint should apply only to *increased* share, as tooling
and ramp already do. **This is a semantics question for the constraint layer, not fixed here.** The
v8 fractions above are over the 63 part-plants with a feasible candidate. Over all 120 sampled they
are halved: quotable 6/120 = 5.0% at weight 0.

**v8's signals are as-of.** Fill (b5flat22, recalibrated, 5 fits) and strain (mp h⁴ P90, 5 seeds)
come only from test snapshots **recorded on or before t0** (2025-01-06, 02-17, 03-31, 05-12, 06-23):
420 suppliers each. The v6/v7 path (`supplier_signals`, Phase 10/11) averages origin 7's whole
evaluation window, 2025 H1, **which straddles its t0 of 2025-04-27**, so it reads predictions made
after t0. It is left untouched so the regression can hold, and recorded as deviation 92.

**A4.6 — the residual limitation, carried.** `strain_penalty` consumes a capacity-strain **P90**,
and capacity intervals are **not quotable** (80% coverage 0.72–0.81, below nominal in 12 of 16
windows, every model class). A weight-0 ranking turns on `(1 − E[fill])·strain_penalty` alone, so it
inherits that more directly than the weight-1 ranking did. The limitation is in the module docstring
and in the `caveat` field of every output, beside `decision_basis`.

**A4.7 — ship gate: met.** Weight-0 output reads **"price not weighted in this decision"** and is
labelled **"most reliable supplier", not "best value"**, in the scorer's return value, the CLI output
and every JSON row.

### 2.5 A5 — one order policy, not two

**A5.1 — one module.** `ml/opt/order_policy.py`, imported by both `ml/sim/montecarlo.py` and
`ml/opt/schedule_lp.py`. It owns every decision about *when* to order and *how an order arrives*.
Neither consumer may contain trigger, lead-time or pipeline logic of its own. **Enforced by a test
that can fail:** `ml/tests/test_single_order_policy.py` fails if `reorder_point_qty`,
`planning_lead_time_days` or a lead-time quantile is referenced outside `order_policy.py`. It must be
shown failing against today's `montecarlo.py`, whose `ordered = cons.mean(2).sum(1)` is exactly such
logic, before it is believed. B2 builds it; it does not exist yet.

**The measurements behind the rule.** `ml/eval/phase12_a5_policy_facts.py` →
`ml/artifacts/phase12_a5_facts.json`, commit `1f6b803`, clean. v8, all nine 2025 test snapshots,
every table filtered as-of t0.

| quantity (median across the nine t0s, range) | value | consequence |
|---|---|---|
| **(ROP − SS) / (weekly requirement × planning lead weeks)** | **1.9 (1.6–2.3)** | ROP **already contains** lead-time demand plus a service buffer |
| ROP / SS | 5.44 | |
| open pipeline at t0 / 13-week requirement (total) | **0.63 (0.55–1.03)** | today's simulation **omits** this entirely |
| (on hand + pipeline) / 13-week requirement | 1.54 (1.28–1.93) | |
| part-plants with IP = on hand + pipeline < ROP at t0 | 0.31 (0.14–0.34) | the trigger does fire, but not everywhere |
| recorded line quantity (as-of, trailing 26 wk) / MOQ | **2.8** | MOQ-and-lot sizing alone would order ≈ ⅓ of a typical line |
| recorded line quantity / lot size | 7.7 | |
| MOQ / weekly requirement | 0.65 | |

The generator's own rule, read at `generator_v8.py:375–380, 524–527`, explains every row. It keeps
inventory position `ip = on_hand + on_order`, reviews every 2 weeks, orders when `ip < rop`, and
orders **up to** `upto = rop + dhat·cover`, where `rop = dhat·lead + safety + z·dhat·√lead·cv`.

**A5.2 — the rule, with the three corrections the data forces.**

| element | brief | **specified** | why |
|---|---|---|---|
| **trigger** | `stock − E[consumption over lead] ≤ reorder_point_qty` | **`IP_w ≤ reorder_point_qty`**, IP_w = on hand + open pipeline (including arrivals sampled beyond the horizon). Equivalent alternative: the brief's form with **`safety_stock_qty`** as the comparator and the P90 lead in the consumption term | ROP − SS is 1.9× lead-time demand, so subtracting lead-time consumption **and** comparing to ROP counts it twice and triggers far too early (deviation 93). B2 runs the ROP form as primary and the SS form as the sensitivity |
| **review cadence** | not stated | every week, unless the as-of PO-creation cadence per part-plant says otherwise (measured in B2) | the generator reviews every 2 weeks, but that parameter is not in the data |
| **lead time** | the full hazard survival curve; buffer to P90 | the **full lead-from-order distribution**, as-of per channel: the hazard curve **re-anchored** from "weeks from t0" to "weeks from order", with the as-of empirical channel lead pmf (receipts recorded ≤ t0) as the fallback. P90 of that distribution is the buffer in the SS form | the arrival label is **weeks from the snapshot** to first receipt for lines **raised after t0**, so T = order-raise wait + lead, and the lateness spec's own offset (+4.571 wk) is that wait. Using T as a lead delays every policy order by the wait (deviation 94). **Gate, capable of failing:** the re-anchored median must match the as-of empirical channel median within ±1 week on validation for ≥ 80% of channels, and the **un-re-anchored** T must FAIL it |
| **quantity** | `min_order_qty` and `lot_size` | **order-up-to:** Q = max(MOQ, lot · ⌈(target − IP) / lot⌉), target = ROP + cover · weekly requirement, with *cover* estimated as-of from recorded line quantities. The MILP keeps batching; this replaces timing only | recorded lines are 2.8× MOQ. MOQ-and-lot sizing alone would under-replenish and reproduce the drain it is meant to fix |
| **open pipeline at t0** | not stated | every PO line created and recorded ≤ t0 without a final receipt recorded ≤ t0, remaining quantity = ordered − received as-of, arrival sampled from the lead distribution **conditional on age**: P(L = a + k \| L > a). **Identity gate, capable of failing:** Σ remaining = `inventory_position_weekly.open_po_qty` at t0, and it must fire with one line dropped | pipeline covers 0.55–1.03× of the 13-week requirement, and the Phase 11C run had none of it (deviation 95) |
| **beyond the horizon** | carried, not dropped | kept in IP, so the trigger does not re-order what is already coming, and reported per path as post-horizon receipts | |
| **fill per order** | — | sampled per order from the fill distribution | unchanged |

**Data caveat carried into the module (deviation 96).** `part_plant.reorder_point_qty` is the
generator's **end-of-run** ROP. It is written once at line 1675 from the final week's `rop`, while
the generator's ROP moves weekly with its demand forecast `dhat`, and it is stamped
`effective_from = 2016-01-01`. For 2025 t0s it is close to the ROP then in force. For early t0s it
is a later value presented as an old one. The module must say so, and B2's validation is on 2025
only.

**A5.3 — the claim, stated honestly.** A reorder-point rule gives a **feasible** schedule, not an
**optimal** one. It answers "when must I order to avoid a stockout", not "when should I order to
minimise cost". The two coincide only when holding cost is zero, which is exactly the degenerate
case the MILP already has (`schedule_lp.py`: with `c_hold = 0` timing is tie-breaking). The rule
sidesteps the missing cost parameters. It does not solve them. There is one more reason to adopt it
here specifically: B2 validates against *observed* 2025 outcomes, and those outcomes were produced by
a reorder-point / order-up-to process. A simulation that replays that process is the like-for-like
comparison. An optimiser's schedule would not be.

---

## 3. Wave B

### 3.1 B1 — conformalised quantile regression on capacity intervals (v6/v7)

`ml/eval/phase12_b1_conformal.py` → `ml/artifacts/phase12_b1.json`, commit `9da7933`, clean.
No training: stored predictions from the Phase 8 backtest bundles.

**B1.1 — the method, fixed before any result.** CQR (Romano, Patterson & Candès 2019), with
`E_i = max(q_lo − y, y − q_hi)` and `Q̂` = the ⌈(n+1)(1−α)⌉-th smallest, applied as `[q_lo − Q̂, q_hi + Q̂]`.
**α = 0.20** (the P10–P90 interval). **Calibration window: the origin's own 12-month validation
slice**, which ends the day before the evaluation half-year begins. A trailing 6-month sub-window and
a width-normalised CQR (E / (q_hi − q_lo)) are pre-declared **sensitivities**. Nothing is selected on
the evaluation window.

The finite-sample guarantee requires calibration and evaluation rows to be **exchangeable**. Phase 8
§8D.4 is the evidence that they are not whenever utilisation shifts. Two further caveats apply on the
own window: rows are correlated (supplier-months recur across snapshots), and the validation slice
also chose the early-stopping epoch.

**B1.2 — P50 is byte-identical**, asserted on every seed of every arm and cell, validation and
evaluation. The assertion **fires** on a 10⁻¹² change to a single P50.

**The implementation is shown correct before the result is read.** Split each calibration window at
random, calibrate on one half and cover the other. That gives **0.797–0.805 in all 48 arm-cells**
(exchangeable data, as the guarantee predicts). At α = 0.5 the same check gives 0.495–0.509. So the
procedure is sound. What fails below is its premise.

**Seeds.** h⁴ is at **5 seeds in 4 cells** (v6 o3, o4, o7; v7 o3) and **3 seeds in the other 12**,
the only bundles that exist. h⁰ is at **3** everywhere and B5 at **5** fits everywhere. Taking h⁴ to
five everywhere would need about 26 training runs (≈5 h), above B1's 3 h ceiling. **Every figure below
is the seed mean; the h⁴ rows at three seeds are the twelve not listed above.**

**B1.3 / B1.5 — 80% coverage per window, band 0.80 ± 0.02 = [0.78, 0.82], pass ≥ 14 of 16 on the NEXT window.**

| world | origin | h⁴ seeds | raw, next window | **CQR, own window** | **CQR, next window** [seed range] | Q̂ | level gap (eval − calib) |
|---|---|---|---|---|---|---|---|
| v6 | 1 | 3 | 0.807 | 0.800 | 0.834 [0.829, 0.843] | +0.012 | −0.029 |
| v6 | 2 | 3 | 0.800 | 0.800 | **0.807** [0.803, 0.811] | +0.003 | −0.012 |
| v6 | 3 | 5 | 0.789 | 0.800 | 0.776 [0.749, 0.806] | −0.005 | −0.089 |
| v6 | 4 | 5 | 0.768 | 0.800 | 0.777 [0.761, 0.804] | +0.003 | −0.008 |
| v6 | 5 | 3 | 0.718 | 0.800 | 0.774 [0.773, 0.776] | +0.017 | −0.006 |
| v6 | 6 | 3 | 0.806 | 0.800 | 0.829 [0.818, 0.839] | +0.010 | +0.085 |
| v6 | 7 | 5 | 0.761 | 0.800 | 0.770 [0.746, 0.790] | +0.004 | +0.040 |
| v6 | 8 | 3 | 0.770 | 0.800 | 0.779 [0.769, 0.787] | +0.005 | +0.073 |
| v7 | 1 | 3 | 0.804 | 0.800 | 0.821 [0.814, 0.831] | +0.011 | −0.042 |
| v7 | 2 | 3 | 0.798 | 0.800 | **0.818** [0.805, 0.833] | +0.013 | +0.037 |
| v7 | 3 | 5 | 0.751 | 0.800 | 0.754 [0.735, 0.764] | +0.001 | −0.192 |
| v7 | 4 | 3 | 0.735 | 0.800 | 0.759 [0.731, 0.787] | +0.010 | −0.022 |
| v7 | 5 | 3 | 0.800 | 0.800 | **0.786** [0.780, 0.790] | −0.007 | −0.022 |
| v7 | 6 | 3 | 0.808 | 0.800 | **0.818** [0.815, 0.823] | +0.006 | +0.200 |
| v7 | 7 | 3 | 0.744 | 0.800 | 0.767 [0.762, 0.774] | +0.016 | +0.050 |
| v7 | 8 | 3 | 0.783 | 0.800 | **0.781** [0.759, 0.793] | −0.001 | +0.116 |

| arm (seeds) | in band, raw | in band, **own** window | **in band, NEXT window** | below / above band | 6-month window | width-normalised | **pass (≥ 14)** |
|---|---|---|---|---|---|---|---|
| **h⁴** (3–5) | 9 | 16 | **5** | 8 / 3 | 7 | 5 | **FAIL** |
| h⁰ (3) | 12 | 16 | **11** | 2 / 3 | 10 | 11 | **FAIL** |
| B5 (5) | 7 | 16 | **9** | 3 / 4 | 7 | 9 | **FAIL** |

**On its own window CQR is at 0.800 in 48 of 48 arm-cells, by construction** (B1.5). That is not
counted. **On the next window it fails for every model class.** For h⁴ it is worse than doing
nothing (9 → 5 in band): Q̂ is tiny (−0.007 to +0.017) and overshoots in low-level windows as often
as it helps in high-level ones.

**B1.4 — ρ between P90 exceedance and the level gap** (Spearman over 16 windows, seed means):

| arm | raw | CQR (12-month) | CQR (6-month) | width-normalised |
|---|---|---|---|---|
| h⁴ | +0.653 (p = 0.006) | **+0.650** (0.006) | +0.694 (0.003) | +0.641 (0.007) |
| h⁰ | +0.797 (< 0.001) | **+0.803** (< 0.001) | +0.850 | +0.803 |
| B5 | +0.832 (< 0.001) | **+0.826** (< 0.001) | +0.829 | +0.821 |

The raw column **reproduces Phase 8 §8D.4's validation→evaluation shift correlations exactly**
(h⁴ +0.65, h⁰ +0.80, B5 +0.83), which is the check that the level-gap variable is the same one.
**ρ does not fall at all.** This is not even the cosmetic outcome B1.4 warned about, because coverage
did not improve either.

**B1.6 — premise re-checked (well inside the 3 h ceiling).** The brief's premise is that the
intervals are too narrow and widening fixes them. Measured on the same 16 windows:

| | h⁴ | h⁰ | B5 |
|---|---|---|---|
| correlation, model's own P50 shift (eval − calib) vs the realised label shift | 0.72 | 0.82 | 0.90 |
| **slope: P50 moves this fraction of the level shift** | **0.51** | **0.74** | **0.65** |
| P50 bias on the evaluation window vs level gap (slope) | −0.60 | −0.57 | −0.34 |
| **Spearman, post-CQR P90 exceedance vs P50 evaluation bias** | **0.96** | **0.91** | **0.76** |

**The intervals are not too narrow; they are in the wrong place.** Every model's median follows a
level shift only halfway, so when utilisation rises the whole interval sits low. P90 exceedance is
then almost entirely the median's lag (ρ = 0.96 for h⁴). A widening calibrated on the previous level
cannot anticipate a shift it has not seen, and B1.2 forbids moving the median. **Interval-only
conformal widening cannot pass this test on this data.** The fix belongs in level tracking: refit the
quantile heads on a trailing window, which is observation 1 §4.5's alternative (b), or add an
explicit level term to the median. Both retrain the median, so both are out of B1's scope. Proposed,
not built (deviation 98).

### 3.2 B2 — the order policy, built and validated

**Code.** `ml/opt/order_policy.py` is the single policy. `ml/sim/montecarlo.py` (refactored:
`read_heads`, `forward_requirement` and `fill_by_part_plant` are now shared; the Phase 9.1 placeholder
is kept, labelled, only to reproduce 11C) and `ml/opt/schedule_lp.py` (`mode="policy"`) both import it.
`ml/tests/test_single_order_policy.py` **passes, and FIRES** when a reorder-point reference is injected
into `schedule_lp.py` or the placeholder heuristic is copied outside its sanctioned function.
Validation harness: `ml/sim/phase12_b2_validate.py` (commit `077a779`; five-seed arms `9e2d59d`).

#### The comparison had to be rebuilt first (deviation 99)

11C Stage D.3's comparison script was **never committed**; its numbers existed only in the report.
The harness rebuilds it, and the unchanged placeholder must reproduce it before anything else is
believed:

| | Phase 11C reported | **rebuilt, placeholder** |
|---|---|---|
| simulated part-plant-weeks below SS (9 snapshots × 200 paths) | 44.36% [33.25, 55.53] | **44.35% [33.26, 55.52]** |
| observed, 2025 store, on hand < SS | 8.38% | **8.38%** (219,024 part-plant-weeks) |
| ratio | 5.29× | **5.29×** |
| simulated mean shortfall when short | 564 | **564** |
| replacement, arrivals / consumption | 85.4% | **85.4% on the first snapshot; 82.5% over all nine** |

It reproduces. 11C's 85.4% is the 2025-01-06 snapshot alone; 82.5% is the nine-snapshot figure and
is used below. After the `montecarlo` refactor the placeholder still gives 44.348% (to ≈10⁻⁸).

**The reference, stated precisely.** "Observed" is the 2025 store's `qty_on_hand < safety_stock_qty`,
the same quantity the simulation's I₀ and roll-forward represent. Using `qty_available` gives 15.84%,
but that nets reservations the simulation does not model, so it is not like-for-like. A
**horizon-matched** reference (the 13 weeks each snapshot projects) is also reported: 8.00%.

#### The gates, each shown capable of failing

| gate | on the real input | on the constructed failure |
|---|---|---|
| **9.1 identity** (B2.2), full scale | correct replay: **0 mismatched, PASS** | one unit wrong in one row: **FIRES (1)**; scrap netted: FIRES (1,971,853); opening omitted: FIRES (1,971,853); SS in place of level: FIRES (1,978,220) |
| **lead gate** (A5): median within ±1 wk of the as-of empirical lead for ≥ 80% of part-plants, validation snapshot 2024-06-10 | empirical median **4 wk** | raw hazard T: **0.0% within, FIRES** (median 12 wk); **re-anchored T − 4.571: 1.9% within, also FAILS** |
| **pipeline identity** (simulated pipeline at t0 = store `open_po_qty`, per part-plant) | passes on every block of every snapshot | fires if a part-plant's pipeline is dropped |
| **two ledgers**: `montecarlo.roll_forward` on the policy's arrivals must reproduce the policy loop's positions | holds (atol 10⁻²) on every block | — |
| **single policy** (A5.1) | PASS | FIRES on injected logic |
| **schedule timing**: no receipt outside `order_policy`'s weeks | 0 of 200 part-plants | **FIRES** with the timing constraint removed |

**Two corrections to the Wave A specification, both forced by data:**

- **The hazard head cannot supply a lead-from-order, even re-anchored** (deviation 101). Its median is
  12 weeks from the snapshot, against an empirical lead of 4, and subtracting the lateness offset
  leaves only 1.9% of part-plants within a week. The simulation therefore uses the **as-of empirical
  lead distribution**: full pmf, receipts recorded ≤ t0, trailing 104 weeks, per part-plant, shrunk to
  the global pmf below 20 receipts. That was A5's fallback. It is now the only admissible source.
- **The line-level pipeline cannot reproduce the store** (deviation 100). About 2% of PO lines close
  with **zero** delivery and write **no GRN row at all**: there are 0 zero-quantity GRN rows among
  1,029,846. So an old unreceived line is indistinguishable, as-of, from a pending one. Unrestricted,
  the reconstruction totals 8.3 M against the store's 4.1 M; restricted to lines ≤ 26 weeks old, it
  is 3.96 M, with 65% of part-plants exact at 2025-06-23. The pipeline **total** comes from the as-of
  store. **Timing** comes from each part-plant's recent unreceived lines, drawn from P(L = a + k | L > a).

#### B2.3 / B2.4 — the result, against held-out 2025 observed outcomes

v8, all 4,212–4,340 part-plants, nine 2025 snapshots, 200 paths. Consumption is the plan ×
N(1, **0.1265**), which is B2.5's measured dispersion. **It still represents PLAN error, not demand
uncertainty: no demand head exists.** Fill is drawn from the neural fill head, as in 11C.

| arm | below SS | **ratio vs 8.38%** | horizon-matched | per-snapshot ratio | replacement | shortfall when short | orders / pp / 13 wk |
|---|---|---|---|---|---|---|---|
| placeholder (11C) | 44.35% | **5.29×** | 5.54× | 3.56–10.56 | 82.5% | 564 | 1 lump |
| policy, no pipeline | 23.30% | 2.78× | 2.91× | 1.99–4.77 | 92.5% | 377 | 5.4 |
| **policy, ROP form (primary)** | **12.47%** | **1.49×** | 1.56× | **1.14–2.11** | **100.3%** | **298** | 4.1 |
| policy, SS form (sensitivity) | 12.98% | 1.55× | 1.62× | 1.16–2.47 | 92.6% | 299 | 3.6 |
| *observed, 2025 store* | *8.38% (8.00% horizon-matched)* | | | | *receipts / issue 101.2%* | *91* | |

**Five fill-head seeds** (B2 primary arm; the arrival head no longer enters the simulation):

| fill seed | 7 | 17 | 27 | 37 | 47 | **band** |
|---|---|---|---|---|---|---|
| below SS | 12.47% | 12.27% | 12.28% | 12.45% | 12.35% | **[12.27%, 12.47%]** |
| ratio | 1.487 | 1.464 | 1.465 | 1.485 | 1.472 | **1.475× [1.464, 1.487]** |

**Attribution** (each step is a measured arm):

- **5.29× → 2.78×:** order rule + lead-from-order + widened jitter.
- **2.78× → 1.49×:** the open pipeline at t0.
- On a log scale the policy closes **77%** of the gap to 1.0.
- Replacement moves from 82.5% to **100.3%**, against the store's receipts/issue of 101.2%. **The
  horizon no longer drains by construction.** That was 11C's proximate cause, and it is gone.
- Shortfall when short falls 564 → 298 against an observed 91. The **magnitude is still 3.3× over**,
  worse than the frequency.

**B2.2 — the full grid under the policy.** 4,340 part-plants × 61 snapshots (2019–2025) × 13 weeks ×
1,000 paths = **3,441,620,000 cells in 11.67 min**, against 5.16 min for the placeholder. The policy
loop is the cost. The 9.1 gate passes and all five failures fire at full scale. The 113 orphan
part-plants are carried with zero arrivals: 0.042% of projected shortfall.
(`ml/artifacts/phase12_b2_fullgrid.json`, commit `9e2d59d`, clean. `montecarlo.main`'s D.1/D.4
subset-timing block still exercises the placeholder; only the grid uses `--policy`, see deviation 103.)

**B2.6 — `schedule_lp.py` on the same module.** `mode="policy"`, v8, t0 = 2025-06-23, 200 random
part-plants:

- **Timing from `order_policy.receipt_plan`.** The expected path from the as-of opening level and
  pipeline, with receipts at placement + median lead.
- **The MILP owns only lot multiples, MOQ, capacity and a penalised safety-stock floor.**
- **200 of 200 solved; 175 receive at least one order; 0 receipts outside the policy's weeks.**
- **43 of 200 cannot hold safety stock at the policy's timing.** Their slack is reported, not hidden.
- **The MILP batches to a median 0.59× the policy's own order-up-to quantity.** It buys only what the
  13-week floor needs, because it has no end-of-horizon target.
- **This is a FEASIBLE schedule, not an OPTIMAL one** (A5.3). There is still no holding or ordering
  cost.

#### B2.7 — verdict: the ratio closes most of the way; this is NOT yet a calibrated forecast

**5.29× → 1.475× [1.464, 1.487]**, disjoint from 1.0 on every seed and above 1.0 on every one of the
nine snapshots (1.14–2.11). The first calibrated forecast in the project **has not been reached**.

**What the residual implicates**, measured in the held-out 2025 ledger over the same horizons
(`ml/sim/phase12_b2_residual.py`, commit `9e2d59d`):

| flow per part-plant per 13-week horizon | observed 2025 | simulation |
|---|---|---|
| consumption (issue to production) | 1,485 | 1,542 (the plan): **+3.8%** |
| supplier receipts | 1,503 | 1,547: +2.9% |
| **inter-plant transfers IN** | **301 (20% of consumption)** | **0: not modelled** |
| inter-plant transfers OUT | 292 | 0 |
| expedited quantity | 10.5 | 0 |
| **part-plant-weeks above SS only because of that week's transfer-in** | **2.7%** | — |
| observed below SS if that week's transfer-in is removed | **≥ 10.7%** | 12.5% |

1. **Inter-plant transfers (primary).** Net transfer is small (+9.6 per part-plant), but the gross
   flow is 20% of consumption, and generator §4 sends it from surplus plants **to short ones**. It is
   a replenishment path aimed exactly at the part-plant-weeks this metric counts. Removing only the
   same-week transfers already lifts observed below-SS to ≥ 10.7%, i.e. a ratio **≤ 1.16×**. Earlier
   weeks' transfers help too, so that bound is loose.
2. **Plan above actual consumption by 3.8%.** The plan is the only consumption input, and no demand
   head exists.
3. **The magnitude gap (3.3×) is consistent with (1).** Transfers top up the deepest shortfalls first.

**This is a more informative failure than 11C's**, as B2.7 anticipated. The drain is gone. What
remains is a named mechanism the data carries (`inventory_transactions.transfer_in/out`), not a
placeholder.

### 3.3 The hard checkpoint

The brief: *"If B2's ratio does not close, Wave C is deprioritised."* It closed from 5.29× to 1.48×
but not to 1.0. **Wave C has not been started, and the decision is yours.** My recommendation:

1. **Next: model inter-plant transfers in `order_policy`** as a rebalancing step from as-of surplus
   (on hand above 1.25 × SS, the generator's own threshold) to short part-plants of the same part.
   Then re-validate. It is estimated at 2–3 h. It is the only item whose expected effect is known to
   be large (≤ 1.16× from the same-week bound alone). The same run should carry the 3.8% plan bias as
   a stated residual.
2. **C3 (part-relation ablation, ≈2.5 h)** is independent of B2 and can run in parallel on the other
   queue. Its ceiling is the 0.004-scale headroom. It is worth running because it is cheap and fully
   specified, not because it moves anything a planner reads.
3. **C1 and C2 stay cancelled** (§2.1, §2.2).

### 3.4 B3 — inter-plant transfers (run after the checkpoint, on "run all")

**What was built.** `order_policy.rebalance`, running every simulated week per **part** across its plants
after receipts, follows generator §4's form:

- surplus_p = max(0, I − 1.25·SS), deficit_p = max(0, SS − I);
- moved = rate × min(Σ surplus, Σ deficit), taken and given in proportion;
- the part total is conserved exactly (unit-tested on a constructed case).

Two parameters, stated separately:

- **LEND = 1.25** comes from the generator source. It is a stated policy parameter and a client ask.
- **The rate is estimated as-of from the ledger**: realised net transfer-in over min(Σ surplus, Σ deficit),
  computed on pre-transfer positions over the trailing 52 weeks, with only rows recorded ≤ t0.

Net transfers enter the returned arrivals, so the two-ledgers check still holds. The work was built in a
separate worktree (`p12-transfers`, `5716e95`, merged as `dc68562`), so that no C3 training cell could stamp
`+dirty`.

**A correction to §3.2 found on the way (deviation 105).** B2's residual table used **gross** transfer-in
(301 per part-plant). The generator moves stock between **channels**, and 30% of gross transfer-in is
shuffling between channels of the *same* part-plant, which nets to zero there. Re-measured on **net**
transfers:

- 2.61% of 2025 part-plant-weeks were lifted above SS by that week's net transfer (gross gave 2.7%);
- pre-transfer below-SS is 10.95%, so the bound becomes 12.47 / 10.95 ≈ **1.14×**.

The conclusion stands; the number is corrected.

**Second finding: the realised rate exceeds what a part-plant model can move.** Realised net inflow runs
**1.45–1.98×** min(Σ surplus, Σ deficit) at part-plant level, at every t0 tested. At that granularity, with
that lending threshold, there is not enough visible surplus to carry the flows the ledger shows, so the rate
clips at 1.0.

**Result** (9 snapshots × 200 paths; five fill-head seeds for the primary transfer arm):

| arm | below SS | ratio vs 8.38% | shortfall when short |
|---|---|---|---|
| B2 primary, no transfers (5 seeds) | 12.27–12.47% | **1.475× [1.464, 1.487]** | 298 |
| **+ transfers, LEND 1.25, as-of rate (5 seeds)** | 11.94–12.13% | **1.435× [1.424, 1.447]** | **19.5–19.9** |
| + transfers, LEND 1.0 (seed 7; bound on lending) | 12.45% | 1.485× | 15.2 |
| observed 2025 | 8.38% | — | 91 |

**Verdict: transfers at part-plant granularity do NOT close the gap.** The bands are disjoint, but the ratio
moves by only 0.04. The mechanism also **distorts the magnitude**: proportional top-ups turn deep shortfalls
into shallow ones (298 → 20) and rarely lift a part-plant back **above** SS, so the magnitude goes from 3.3×
over to **4.6× under**. Lending more (LEND 1.0) does not help either; it pushes the lenders below their own
SS. In the data, transfers rescue weeks outright, because the generator tops up individual **channels**.
A simulation whose state is the part-plant cannot represent that (deviation 106).

**What this means for the residual.** The 1.4–1.5× over-projection is now localised to **sub-part-plant
(channel-level) dynamics**: channel stocks, channel-level safety stock, channel-level rebalancing. The
simulation aggregates all of those away. Fixing it would mean re-stating the simulation at channel
granularity, a structural change and not a parameter. **B2's ROP arm without transfers stays the primary
figure (1.475×).** The transfer arm is a labelled sensitivity. It is not adopted, because it trades a
0.04 frequency gain for a magnitude error.

## 4. Wave C

*Not started — held at the hard checkpoint (§3.3).* **C3 runs** per §2.3's specification when released. **C1 is cancelled** (§2.1) and **C2 is
cancelled** (§2.2). **C4 is replaced** by A2.4's proposals (§2.2), which are not built in this phase.

## 5. Proposed ship-table changes — PROPOSALS; `shipped.json` untouched

From Wave A only:

| task | proposal |
|---|---|
| fill_rate | **no model change.** Build the b5flat22 serving loader (A2.4 option 3, deviation 46) so the configuration describes what runs. The interior reparameterisation is a next-phase build |
| allocation (not in `shipped.json`) | if allocation is shown at all, show it at **price_weight = 0** with the label "most reliable supplier — price not weighted in this decision". Quote only deduplicated, band-surviving recommendations, and the strain-P90 caveat travels with every row. **Do not show v8 allocation** until the qualification-constraint semantics (deviation 91) are settled |
| arrival_week | unchanged |
| capacity_strain | **unchanged, and the interval caveat now has a mechanism:** the median tracks only 51–74% of a level shift, and interval-only conformal widening does not fix coverage on the next window (B1). "Do not quote a capacity interval" stands. Proposed next: refit the quantile heads on a trailing window (retrains the median) |
| shortage_qty (simulation) | **not shipped.** It moved from "unusable (5.29×)" to "over-projects 1.48× [1.46, 1.49], mechanism named". No quantity from it is quotable until the transfer path is modelled and the ratio re-validated. If a figure is shown internally it carries: ratio 1.48×, plan-error-only intervals, transfers not modelled |

## 6. Deviations and open items

### 6.1 Deviations, continuing the index from 85

| # | Prior statement | Measured | Where |
|---|---|---|---|
| **85** | brief A1.3: "~3,000 December rows against a full-fold seed spread of ~0.003" | **4,000** December rows, **2,058** scored (uncensored); full-fold 5-seed spread **0.0065**. The test fold is **nine monthly snapshots** (Apr, Jul, Nov absent), so month and snapshot are the same variable on one fold | §2.1 |
| **86** | brief / observation 1 §3.4, quoted for v8: "Σ\|error\| 0.0445 against LightGBM's 0.0049" | those are **Phase 5 §6.4's v6** figures. v8 head, training fold, raw: interior **0.057–0.084**, total 0.064–0.095 (5 seeds). b5flat22's v8 training marginal is not measured (predictions never stored) | §2.2 |
| **87** | brief A2.2: "roughly 18% spreads over 20 interior bins — under 1% per bin" | interior **19.3%** (train); 15 of 20 cells under 1%, but [0.05, 0.25) holds **1.4–3.3%** per cell; median cell 0.55% (968 rows) | §2.2 |
| **88** | a snapshot-permutation null tests month concentration on any fold | **vacuous on the test fold**: one snapshot per month with equal row shares, so the statistic is permutation-invariant (null = observed to 10⁻¹⁶). Caught before use; no test-fold concentration p-value is reported | §2.2 |
| **89** | brief A4.2: weight-0 recommendations "must be IDENTICAL to current output" | **cannot hold.** Current output includes the purchase term and flips with `c_short`; weight 0 provably cannot. Regression run at **weight 1** against the stored Phase 11 sweep: 0 differences on v6 and v7 | §2.4 |
| **90** | Phase 11 Stage 3: 25% of part-plants carry a quotable recommendation | **the count treated duplicate candidate splits as runners-up.** Every exact top-two tie, 29/60 (v6) and 30/60 (v7), is the winner under another name, with margin 0 and never quotable. Deduplicated: 53/120 (44.2%). **The actionable count is unchanged (21/120)**: every recovered recommendation is "keep the incumbent" | §2.4 |
| **91** | the allocation constraint layer is correct on v8's data | **57 of 120 v8 part-plants have NO feasible candidate, the incumbent included.** The qualification check rejects any share to a supplier `alternate_sources` marks unqualified, including volume it already holds. Inert on v6/v7, binding on v8 | §2.4 |
| **92** | Phase 10/11 allocation signals are as-of t0 | `supplier_signals` averages origin 7's whole 2025 H1 evaluation window, which **straddles** its t0 (2025-04-27), so it uses predictions made after t0. Left unchanged so the regression holds; the v8 path is restricted to snapshots ≤ t0 | §2.4 |
| **93** | brief A5.2 trigger: `stock − E[consumption over lead] ≤ reorder_point_qty` | **double-counts lead-time demand.** (ROP − SS) = **1.9×** (1.6–2.3) planning-lead demand on v8; the generator's ROP is `dhat·lead + SS + z·…`. Specified: `IP ≤ ROP`, or the brief's form with SS | §2.5 |
| **94** | brief A5.2 / observation 1 §6.1: the hazard head gives an order's lead time | the arrival label is **weeks from the snapshot** for lines **raised after t0**: T = order-raise wait + lead (the lateness offset +4.571 wk is that wait). T is not a lead from order and must be re-anchored, with a gate that the raw T fails | §2.5 |
| **95** | observation 1 §6.1: "for each open order, week T is drawn from the hazard head" | `montecarlo.draw_from_heads` draws **one lump order** of the horizon's requirement per part-plant and models **no open pipeline at t0**. The pipeline already covers **0.55–1.03×** of the 13-week requirement | §2.5 |
| **96** | `part_plant.reorder_point_qty` is a static policy parameter effective from 2016 | it is the generator's **end-of-run** ROP (`generator_v8.py:1675`), a snapshot of a weekly-varying policy stamped `effective_from 2016-01-01` — a later value presented as an early one for any t0 before 2025 | §2.5 |
| **97** | A2's month decomposition sums exactly to E | **it did not on the first run**: float32 means broke the 10⁻⁹ identity check. The check fired and the decomposition now runs in float64. Recorded because the check is what caught it | §2.2 |

| **98** | observation 1 §4.5: conformal widening keyed to recent drift is "the one deferred model change whose premise still holds" | **the premise does not hold.** CQR is exact on its own window (48/48) and fails on the next for every class (h⁴ 5/16, h⁰ 11/16, B5 9/16 in [0.78, 0.82]); ρ unchanged (0.65 / 0.80 / 0.83). Each model's P50 follows only **51–74%** of a level shift, and that lag explains post-CQR exceedance at Spearman 0.96 (h⁴). The fault is the median's level tracking, which an interval-only method may not touch | §3.1 |
| **99** | Phase 11C D.3's 44.36% / 8.38% / 5.29× / 85.4% | **the script that produced them was never committed.** Rebuilt; reproduces 44.35% [33.26, 55.52], 8.38%, 5.29×, 564. **85.4% is the first snapshot's replacement rate**; over all nine it is 82.5% | §3.2 |
| **100** | A5's pipeline identity: Σ unreceived line quantity = `open_po_qty` | **unattainable line by line:** ~2% of PO lines close with zero delivery and write no GRN (0 zero-quantity rows in 1,029,846), so they look open forever. The total comes from the as-of store; timing from recent unreceived lines. The identity gate is re-pointed at the simulated total, and it can still fail | §3.2 |
| **101** | A5: the hazard curve, re-anchored by the lateness offset, can be the lead-from-order | **it cannot.** Re-anchored T is within ±1 wk of the empirical lead for 1.9% of part-plants (raw T: 0.0%; median 12 wk vs 4). The simulation's lead is the as-of empirical pmf | §3.2 |
| **102** | observation 1 §6.1: "the board, the dice and the scorekeeping are all correct. One player instruction is wrong" | **three instructions were wrong** (the trigger/quantity, the lead definition, the missing pipeline) **and one mechanism is missing**: inter-plant transfers, 20% of consumption flows, aimed at short part-plants. With the three fixed the ratio is 1.48×; the transfers are the named residual | §3.2 |
| **103** | `montecarlo.py --policy rop` runs the policy end to end | **the grid does; the D.1/D.4 subset-timing block in `main()` still calls the placeholder `draw_from_heads`.** Its timing lines describe the placeholder. The grid and every validation number are the policy's | §3.2 |
| **105** | §3.2 (B2.7): transfers IN are 301 per part-plant per horizon, and 2.7% of weeks are above SS only because of them | **gross, not net.** 30% of gross transfer-in moves between channels of the SAME part-plant and nets to zero there. On NET transfers: 2.61% of weeks rescued, pre-transfer below-SS 10.95%, bound ≈ 1.14×. The conclusion survives; the number is corrected | §3.4 |
| **106** | §3.3: modelling inter-plant transfers is the step "whose expected effect is known to be large" | **it was not, at part-plant granularity.** The ratio moves 1.475× → 1.435×, and the magnitude goes from 3.3× over to 4.6× under. The realised rate is 1.45–1.98× the part-plant-visible transferable pool. The residual is channel-level dynamics, which a part-plant state cannot carry | §3.4 |
| **104** | Phase 11C: the full grid takes 5.16 min | under the policy it takes **11.67 min**: the endogenous order loop and the pipeline scheduling. Still minutes | §3.2 |

### 6.2 Open items

1. **Allocation's qualification semantics on v8** (deviation 91). Decide whether `alternate_sources`
   status applies to incumbents. Until then, no v8 allocation output is quotable.
2. **`supplier_signals`' as-of looseness on v6/v7** (deviation 92). Fix when v6/v7 allocation is next
   re-run, and accept that the regression baseline moves when it is.
3. **b5flat22's v8 training marginal** is unmeasured (deviation 86). It needs a refit, which is
   training.
4. **A2.4's three proposals** are written, not built. Option 3 (the loader) is also deviation 46.
5. **A1's residual seasonal tilt** (+0.16 wk per unit of the lagged shape, p = 0.008) is real and
   small. If a future metric scores arrival-week *level* rather than lateness ranking, revisit it.
   Under the adopted metric it is worth nothing measurable.
6. `results/observation1.md` is untracked and sits outside `reports/`. Moving it is the owner's call.
7. **Model inter-plant transfers** in `order_policy` and re-validate B2 (§3.3, item 1). This is the gate on
   the first calibrated simulation.
8. **The simulation's fill is the neural head**, while `shipped.json` names b5flat22 (deviation 46).
   B2's five-seed band is over the neural head's seeds.
9. **Capacity level tracking** (deviation 98): a trailing-window refit of the quantile heads, or a level
   term in the median. It retrains the median, so it is a next-phase test.
10. **h⁴ capacity backtest bundles are at 3 seeds in 12 of 16 cells.** B1's verdict does not depend on
    them (it fails for every class, including B5 at five fits), but those rows are three-seed rows.
11. **The magnitude gap** (298 simulated vs 91 observed shortfall when short) needs its own check after
    transfers are modelled.

---

## 7. Compliance (Waves A and B)

- **`ml/configs/shipped.json` is unchanged.** Verified by an empty `git diff` on the file. §5 is
  proposals only.
- **No assertion was disabled or weakened.** The Phase 11 as-of assertion (`assert_asof_receipts`)
  ran armed inside A1's reference build. Assertions **added**: A1's reproduction of 0.70905; A2's
  inference-reproduces-stored-predictions (max |Δ| < 10⁻⁴, observed 0.0) and the decomposition
  identity; A4's regression. No line-level feature was added; A1 and A2 read predictions only.
- **Every new gate was shown capable of failing before its result was believed.** A2's concentration
  test (injected October fires; flat construction silent); A4's invariance check (fires on weight 1:
  10 / 13 / 15 flagged); A4's regression (it compares against stored output and would report any
  difference); A2's decomposition identity (it fired, deviation 97). **One gate was found vacuous and
  not used:** A2's test-fold permutation (deviation 88). A5's three gates (lead re-anchoring,
  pipeline identity, single-policy test) are specified with the input that must make each fire. They
  are built and demonstrated in B2.
- **Five model seeds for everything newly quoted on v8:** A1 (h⁴ arrival × 5), A2 (fill head × 5,
  b5flat22 × 5 fits), A4 v8 (b5flat22 × 5, mp h⁴ × 5). **A4's v6/v7 rows are at three seeds**, the
  only seeds the Phase 10/11 signals exist at, and each such row says so.
- **Selection on validation only.** A1's seasonal-shift ceiling is fitted on 2024 and applied to
  2025. Nothing was selected on test.
- **No training ran.** A2 ran inference over the training fold with existing checkpoints.
- **Commit before running.** Every artifact carries `code_dirty: false`: A1 `f05d755`, A2 `7cc20e6`,
  A4 `51ff706`, A5 facts `1f6b803`. `phase12_common.require_clean()` refuses to run on a dirty `ml/`.
- **`inventory_position_weekly` was read in exactly one Wave A place:**
  `ml/eval/phase12_a5_policy_facts.py`, for `qty_on_hand`, `open_po_qty` and `qty_in_transit`
  (plus `safety_stock_qty`), filtered as-of t0. It was read **to specify the order policy**, i.e. to
  measure the open pipeline and the inventory position the trigger will use. That is the same
  sanctioned use the simulation already makes (B1 reconciled the store at 100.000000%). It was never
  read as a model feature, and never in A1–A4.
- **Wave B additions.**
  - **`shipped.json` is still unchanged.**
  - **No assertion was weakened.** The 9.1 identity gate ran armed at full scale. The as-of assertions
    in `montecarlo` (`assert_asof`) and in `order_policy` (lead history, line quantities, pipeline)
    all require recorded ≤ t0 and a non-empty set.
  - **Assertions added:** P50 byte-identity (B1); pipeline identity; the two-ledgers check; the lead
    gate; the single-policy test; the schedule timing gate. **Each is shown firing in §3.1–§3.2.**
  - **No learning rate retuned. No training ran in Wave B.** B1 is post-hoc on stored predictions; B2
    runs inference on the shipped heads.
  - **Seeds.** B1's h⁴ and h⁰ rows are at three seeds except the four h⁴ cells at five; B5 is at five.
    **B2's headline ratio is at five fill-head seeds.**
  - **Every Wave B artifact is stamped clean:** B1 `9da7933`; B2 validation `077a779`; seeds and
    residual `9e2d59d`; the full grid ran at `9e2d59d` with a clean tree. The `montecarlo` CLI does
    not stamp its JSON; the commit is recorded here.
  - **`inventory_position_weekly` in Wave B** was read only by `montecarlo.opening_position` and
    `run_gate` (opening level; the 9.1 reference), by `order_policy.Tables` / `pipeline` /
    `receipt_plan` (`open_po_qty` and `qty_on_hand`, as-of t0, for the pipeline total and opening
    level), and by `phase12_b2_validate.observed` (2025 weeks, as the **held-out evaluation
    reference**). It is the store B1 reconciled at 100.000000%, and in none of these is it a model
    feature. `inventory_transactions` and `expedite_events` were read by the residual diagnostic,
    as evaluation references only.
- **Protected paths untouched:** `db/gen_v6/**`, `db/gen_v7/**`, `db/gen_v8/**` (the generator was
  *read*, not modified), `db/validator.py`, `docs/specs/**`, `db/dataset_structure.md`, and every
  existing report.
