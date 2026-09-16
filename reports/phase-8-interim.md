# Phase 8 interim — capacity depth, the drift pre-pass, and what to run next

**Status.** The full backtest stopped at the 6-hour budget check after origin 1 (51.5 h projected, 48 h remaining).
This interim report covers **Stage 8A** (inference-only drift pre-pass), **Stage 8B** (capacity, both worlds,
origins 2–8) and the decision gate **8C**. **Arrival and fill were not run. Fill is deferred and unscheduled** — it has
no depth question of its own (the shipped configuration *is* h⁰), so it buys nothing until the capacity and arrival
questions are settled. The full `reports/phase-8.md` stays open until they are.

Every standing rule held: selection on validation only, bands per metric per configuration per world and never
borrowed, no learning rate retuned, no protected file touched, `inventory_position_weekly` never read.

<!-- P8I_S1 -->

## 2. Stage 8A — the label-free drift pre-pass, and whether 3b can be refitted at all

**What this is.** Origin 1's 30 trained checkpoints — every task, both worlds, all three seeds — run over the input
windows of **all eight origins**. Inference only: no training, no refitting, no recalibration refit, and no label
value on the prediction path. Each model's own label-free statistic against its own validation baseline, exactly as
`loop.predict` computes it; `ml/artifacts/backtest/phase8a_drift_prepass.json`, 240 (bundle × window) inferences.

**What it is not (8A.4).** These are **origin-1 models scored on other windows**, not each origin's own model. A model
run on a window five years past its training period drifts partly because it is stale — a reason Stage 3b would never
see, because 3b trains a fresh model per origin. **This bounds the input drift the eight windows contain. It is not a
substitute for 3b, and no threshold is refitted from it.**

### Drift per origin (mean over 3 seeds; excess = |shipped − h⁰| on the same inputs, which is what the bands are defined on)

| task | world | origin | shipped drift pp | h⁰ drift pp | excess pp (min–max) | band(s) across seeds |
|---|---|---|---|---|---|---|
| arrival_week | v6 | 1 | +0.36 | +0.99 | 0.69 (0.02–1.97) | usable, degraded |
| arrival_week | v6 | 2 | -4.94 | -1.52 | 3.42 (3.00–3.68) | degraded |
| arrival_week | v6 | 3 | -1.77 | -2.46 | 0.81 (0.16–1.37) | usable, watch, degraded |
| arrival_week | v6 | 4 | -2.46 | -2.14 | 0.41 (0.06–1.03) | usable, degraded |
| arrival_week | v6 | 5 | -1.54 | -1.85 | 0.74 (0.58–1.00) | watch, degraded |
| arrival_week | v6 | 6 | -3.91 | -2.15 | 1.76 (0.62–2.92) | watch, degraded |
| arrival_week | v6 | 7 | -2.90 | -1.75 | 1.15 (0.39–1.70) | usable, degraded |
| arrival_week | v6 | 8 | -3.42 | -1.84 | 1.58 (1.18–2.01) | degraded |
| arrival_week | v7 | 1 | -0.43 | +0.72 | 1.15 (0.78–1.44) | watch, degraded |
| arrival_week | v7 | 2 | -4.39 | -1.11 | 3.28 (2.79–3.93) | degraded |
| arrival_week | v7 | 3 | -1.59 | -1.75 | 0.28 (0.15–0.52) | usable, watch |
| arrival_week | v7 | 4 | -2.43 | -2.14 | 0.69 (0.60–0.76) | watch |
| arrival_week | v7 | 5 | -3.04 | -1.65 | 1.39 (0.05–2.15) | usable, degraded |
| arrival_week | v7 | 6 | -3.96 | -1.86 | 2.10 (1.57–2.92) | degraded |
| arrival_week | v7 | 7 | -3.89 | -1.24 | 2.65 (1.82–3.22) | degraded |
| arrival_week | v7 | 8 | -3.37 | -1.58 | 1.79 (1.42–2.00) | degraded |
| capacity_strain | v6 | 1 | -2.73 | -1.62 | 1.12 (0.02–1.70) | usable, degraded |
| capacity_strain | v6 | 2 | -2.42 | +2.03 | 4.45 (4.32–4.66) | degraded, **unusable** |
| capacity_strain | v6 | 3 | -6.06 | -5.32 | 0.74 (0.29–1.40) | usable, watch, degraded |
| capacity_strain | v6 | 4 | -2.42 | -2.55 | 0.63 (0.47–0.75) | usable, watch |
| capacity_strain | v6 | 5 | -6.91 | -7.26 | 0.43 (0.13–0.74) | usable, watch |
| capacity_strain | v6 | 6 | -1.56 | -1.88 | 0.81 (0.51–1.19) | watch, degraded |
| capacity_strain | v6 | 7 | -5.36 | -2.76 | 2.60 (2.34–2.88) | degraded |
| capacity_strain | v6 | 8 | +3.50 | +5.44 | 1.94 (1.48–2.65) | degraded |
| capacity_strain | v7 | 1 | -4.61 | -6.21 | 1.61 (0.68–2.38) | watch, degraded |
| capacity_strain | v7 | 2 | +1.70 | +9.38 | 7.68 (6.12–8.65) | **unusable** |
| capacity_strain | v7 | 3 | -11.76 | -12.15 | 0.67 (0.21–1.59) | usable, degraded |
| capacity_strain | v7 | 4 | -1.30 | -0.97 | 0.33 (0.00–0.88) | usable, watch |
| capacity_strain | v7 | 5 | -11.42 | -14.46 | 3.04 (2.27–3.60) | degraded |
| capacity_strain | v7 | 6 | +1.07 | +3.66 | 2.59 (1.51–3.86) | degraded |
| capacity_strain | v7 | 7 | -9.66 | -5.19 | 4.47 (3.27–5.46) | degraded, **unusable** |
| capacity_strain | v7 | 8 | +11.20 | +12.90 | 1.70 (0.82–2.26) | watch, degraded |
| fill_rate | v6 | 1 | -0.17 | -0.17 | — (no h⁰ twin: fill ships **as** h⁰) | usable |
| fill_rate | v6 | 2 | -0.32 | -0.32 | — (no h⁰ twin: fill ships **as** h⁰) | usable |
| fill_rate | v6 | 3 | +1.01 | +1.01 | — (no h⁰ twin: fill ships **as** h⁰) | watch, degraded |
| fill_rate | v6 | 4 | +0.38 | +0.38 | — (no h⁰ twin: fill ships **as** h⁰) | usable |
| fill_rate | v6 | 5 | +1.64 | +1.64 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v6 | 6 | +0.77 | +0.77 | — (no h⁰ twin: fill ships **as** h⁰) | watch |
| fill_rate | v6 | 7 | +0.64 | +0.64 | — (no h⁰ twin: fill ships **as** h⁰) | watch |
| fill_rate | v6 | 8 | -0.76 | -0.76 | — (no h⁰ twin: fill ships **as** h⁰) | watch |
| fill_rate | v7 | 1 | +1.29 | +1.29 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v7 | 2 | -2.86 | -2.86 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v7 | 3 | +2.73 | +2.73 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v7 | 4 | -1.15 | -1.15 | — (no h⁰ twin: fill ships **as** h⁰) | watch, degraded |
| fill_rate | v7 | 5 | +3.62 | +3.62 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v7 | 6 | -1.68 | -1.68 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v7 | 7 | +1.24 | +1.24 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |
| fill_rate | v7 | 8 | -3.32 | -3.32 | — (no h⁰ twin: fill ships **as** h⁰) | degraded |

### 8A.3 — how many observations would land in each band if all eight origins were run

| task | usable ≤ 0.5 | watch 0.5–1.0 | degraded 1.0–4.5 | **unusable > 4.5** | largest observed |
|---|---|---|---|---|---|
| arrival_week | 9 | 9 | 30 | **0** | 3.93 pp |
| capacity_strain | 9 | 9 | 24 | **6** | 8.65 pp |
| fill_rate | 9 | 12 | 27 | **0** | 3.90 pp |

Each row is 8 origins × 2 worlds × 3 seeds = 48 observations (fill: 48 raw-drift observations, since fill has no
h⁰ reference to exceed — the shipped fill configuration *is* h⁰).

**Plainly:**

- **Arrival: zero observations above 4.5 pp.** The largest excess anywhere in the eight windows is **3.93 pp**, and
  30 of 48 sit in the wide `degraded` 1.0–4.5 band. **3b cannot be refitted for arrival from this backtest at any
  budget.** Running all eight origins for 28 h would not produce a single observation in the band that separates
  `degraded` from `unusable`, because the windows do not contain that much drift. The 4.5 pp threshold stays where
  Addendum B put it, and the `watch` band (0.5–1.0, 9 observations here) stays **interpolated**.
- **Capacity is the exception: 6 observations above 4.5 pp** (v6 origin 2 at 4.3–4.7 pp, v7 origin 2 at 6.1–8.7 pp,
  v7 origin 7 at 3.3–5.5 pp). Capacity has **no calibrated threshold at all** (Phase 6 removed arrival's bands from it:
  the statistic is mean P50 utilisation, whose "pp" are hundredths of a utilisation unit, not a probability). So these
  are the first observations that could *start* a capacity-specific curve — but they come from origin-1 models, so they
  bound the input drift and no more.
- **Fill: zero above 4.5 pp**, largest 3.90 pp, and no excess statistic exists at all.

**Consequence for 3b.** The damage side of the curve still requires each origin's own model. What 8A settles is the
*input* side: **the eight windows do not span the upper band for arrival or fill**, so the 28 h of arrival training
cannot calibrate the threshold that matters. That is the single most useful thing this pre-pass bought.

## 3. Recommendation on arrival — do not spend the 28 hours

Arrival (shipped h⁴ **and** h⁰, 3 seeds, both worlds) costs **27.2 h** for origins 2–8 on two queues,
scaled from origin 1's measured cell times by training snapshots:

| origin | training snapshots | arrival cost (h⁴ + h⁰, 3 seeds, both worlds) | 8A excess range across seeds and worlds |
|---|---|---|---|
| 2 | 22 | 2.4 h | 2.79–3.93 pp |
| 3 | 26 | 2.9 h | 0.15–1.37 pp |
| 4 | 31 | 3.4 h | 0.06–1.03 pp |
| 5 | 35 | 3.9 h | 0.05–2.15 pp |
| 6 | 39 | 4.3 h | 0.62–2.92 pp |
| 7 | 44 | 4.9 h | 0.39–3.22 pp |
| 8 | 48 | 5.3 h | 1.18–2.01 pp |
| **2–8 total** | | **27.2 h** | 0.05–3.93 pp |

**Recommendation: no — not for 3b, which was the reason the 28 h was on the table.** 8A shows the eight windows
contain **no arrival observation above 4.5 pp** and only 9 of 48 in the `watch` band. Training every origin's own
arrival model would fill in the *damage* axis for drift the windows do contain (1–4 pp), but it cannot place a single
observation where the `degraded`/`unusable` boundary sits. **The threshold that gates the fallback would stay
interpolated after 28 h of training.**

**What arrival would still buy, and what it costs:**

- **3d across origins** (lateness beyond the promise). Origin 1 already shows the margin shrinking sharply: +0.005 on
  v6 and +0.001 on v7, against +0.033 / +0.025 on the 2025 fixed split. Whether the head's lateness advantage is a
  2025 artefact is a real question, and it is the strongest remaining reason to run arrival.
- **Per-origin C-index and week-ECE tables** for the report's §3.
- **3c across origins** — but see §4: the recommendation there is to remove the gate, which makes its calibration moot.

**If you want a subset, run origins 2, 6 and 7.** They span the arrival drift range the windows do contain — origin 2
is the high-drift extreme in both worlds (3.00–3.93 pp), origins 6 and 7 sit at 1.5–3.2 pp, and origin 1 (already
trained) covers the low end at 0.02–1.44 pp. Origins 3, 4 and 5 add only more of the 0.1–2 pp middle that origin 1
already samples. **Cost: 11.7 h** for the three, against 27.2 h for all seven.

**My order of preference:** (1) stop arrival here and take 3d as provisional on origin 1 plus the fixed split;
(2) if 3d matters for the client deliverable, run origins 2, 6 and 7 only (11.7 h);
(3) the full 28 h buys tables, not answers.

## 4. Recommendation on 3c — remove the drift gate. I agree, and the pre-pass strengthens the case

**Agreed: remove it.** Serve h⁰'s distribution always; keep SHARE-lite h⁴ for ranking. Four pieces of evidence, then
the one argument against, then what removal does not fix.

### 1. The two policies are not distinguishable where they differ

| origin 1 | gated (shipped, threshold 1.0 pp) | always-h⁰ | verdict |
|---|---|---|---|
| v6 week-ECE | 0.0392 (spread 0.0114) | 0.0448 (spread 0.0038) | **not distinguishable** — ranges overlap |
| v7 week-ECE | 0.0397 (spread 0.0075) | 0.0409 (spread 0.0039) | **not distinguishable** |

On the 2025 fixed split the two policies agree wherever the gate engaged (Phase 6 §7: v7 engaged on all three seeds,
so always-h⁰ would have served the same 0.0146), and **always-h⁰ would have been better on v6**, where the gate served
the model on two seeds of three (0.0247 / 0.0221) against h⁰'s 0.0166. So: equal in v7-2025, better in v6-2025, not
distinguishable on origin 1 in either world.

### 2. The gate's output flips with the seed — now measured on 16 windows, not one

Origin 1 split its own decision: **1 of 3 seeds engaged on v6, 2 of 3 on v7**. Phase 6 saw the same on the fixed
split (excess 0.92–1.10 pp across seeds, straddling the 1.0 pp threshold). Stage 8A now puts three seeds against
**16 origin × world windows**:

| the three seeds, in one window | count of the 16 |
|---|---|
| **straddle the threshold — the gate's answer depends on which seed trained** | **8** |
| all three above 1.0 pp (engage) | 6 |
| all three below 1.0 pp (do not engage) | 2 |

Straddling windows: v6 o1 (0.02–1.97 pp), v6 o3 (0.16–1.37 pp), v6 o4 (0.06–1.03 pp), v6 o5 (0.58–1.00 pp), v6 o6 (0.62–2.92 pp), v6 o7 (0.39–1.70 pp), v7 o1 (0.78–1.44 pp), v7 o5 (0.05–2.15 pp).

**A decision procedure whose output changes when you re-run training with a different seed is not a decision
procedure.** The quantity it thresholds — excess drift — has a seed spread comparable to the threshold itself.

### 3. Validation cannot settle it, by construction

The drift statistic is defined as movement *away from the validation baseline*, so on the validation fold the excess
is ≈ 0 and the gate never engages. **No validation-fold experiment can compare the two policies.** That is why this
is a design simplification rather than a selection: adopting it is not choosing a configuration on test, it is
removing a branch that validation cannot exercise. The change itself belongs to Phase 10, not here.

### 4. h⁰'s distribution has been at least as good in every cell tested since Phase 5 — with one honest exception

Phase 6 §7 found h⁰ at least as well calibrated in all six fixed-split cells. **Origin 1 does not reproduce that:**
the model's recalibrated week-ECE (0.0385) is *better* than h⁰'s (0.0448) in point estimate on v6, though the ranges
overlap. The case for removal therefore does not rest on "h⁰ is always better" — it rests on the two policies being
indistinguishable while one of them carries a seed-dependent branch.

### The argument against, stated fairly

On v7-2025 the gate demonstrably worked: it served a distribution 3.1–3.9× better calibrated than the model's own
(Phase 6 §7). Removing the gate keeps that benefit — always-h⁰ serves the same distribution — but it also gives up
the *ability to notice*. A live drift statistic that no longer switches anything still has monitoring value, so
**keep computing and logging drift and excess; stop letting them switch the output.**

### What removal does not fix

h⁰ must still be trained and shipped for every arrival model, because it now serves the distribution. The h⁰ arrival
cells are also the ones that hit the epoch cap (§5). And 8A shows the threshold could not have been calibrated from
this backtest anyway (§2) — removing the gate and leaving the threshold interpolated are the same decision seen from
two sides.

### Running list: "gates that cannot fail" — this is the 6th

| # | where | the gate | why it could not do its job |
|---|---|---|---|
| 1 | `docs/validation6.md` run 1 | line-stop test requiring *every* channel of a part-plant to be empty at once | could never fire — 0 events |
| 2 | `docs/implementation_guide.md` §5.2 | fill band-coverage G5 | cannot fail: the empirical CDF jumps at the top bin, coverage error pins at 0.5 for any forecaster |
| 3 | `reports/phase3_closeout.md` §9.1 | "the simulation reproduces `qty_available` exactly" | vacuous against an all-zero column: satisfied by a simulation that outputs zero |
| 4 | `docs/validation7.md` amendment 01 | a gate that could not *pass* | the companion failure mode |
| 5 | `docs/validator_amendment_02.md` | a bucket gate that would have fired on all three buckets | would be ignored within a week |
| **6** | **this phase** | **arrival's drift gate at 1.0 pp** | **cannot decide: its output flips with the training seed in 8 of 16 windows, and the two branches are not distinguishable** |

`docs/synthetic_rules.md` records that the name was coined after three earlier generator-side instances; the five
above are the ones written up in this repository's reports and guides.

## 5. Two recordings from origin 1

### Per-fold fill recalibration temperature

Recalibration is refitted on each origin's own validation slice and never carried across folds. Origin 1 selected
**vector scaling on all six cells** (3 seeds × 2 worlds):

| world | seed 7 | seed 17 | seed 27 |
|---|---|---|---|
| v6 | T = 1.134 | T = 1.181 | T = 1.107 |
| v7 | T = 1.205 | T = 1.231 | T = 1.176 |

The range **1.107–1.231** is wider than the fixed split's (1.008–1.089, Phase 6 §6), and every fold needed T > 1:
the raw head is over-confident on this window and the correction is larger than it was on 2024.

### Arrival h⁰ v6 seed 7 hit the 120-epoch cap — it is a floor, but a shallow one

| cell | stop | epochs | best epoch | validation gain over the last 10 epochs | training loss still falling |
|---|---|---|---|---|---|
| **v6 h⁰ seed 7** | **CAP (floor)** | **120** | **116** | **8.2e-5** | yes |
| v6 h⁰ seed 17 | patience | 103 | 94 | 9.4e-5 | yes |
| v6 h⁰ seed 27 | patience | 102 | 93 | 2.6e-5 | yes |
| v7 h⁰ seed 7 | patience | 113 | 104 | 2.6e-5 | yes |
| v7 h⁰ seed 17 | patience | 96 | 87 | 6.6e-5 | yes |
| v7 h⁰ seed 27 | patience | 86 | 77 | 1.7e-5 | yes |

**Is it a floor? Yes, by the project's definition** — the run was bound by the cap, not by patience, and its training
loss was still falling. **How much it costs is another matter:** validation C-index moved by **8.2e-5 over the final ten
epochs**, against arrival's measured 3-seed C-index spread of 0.0006–0.0023 on these origins. The cap is holding back
something an order of magnitude smaller than the seed band, and the five sibling cells stopped on patience at 86–113
epochs. **h⁰ arrival simply trains slowly at 2.5e-4**: every cell needs 77–116 epochs to reach its best, against 31–41
for capacity and 57–75 for fill.

**If arrival is run later the cap must be raised** — 200 would clear every observed best epoch with the same margin
patience gives — **and recorded as a deviation**. Raising an epoch cap is not learning-rate retuning; the rate stays at
2.5e-4 as shipped. Until then, every arrival h⁰ number in this phase carries the floor caveat, and the one cell that hit
the cap is flagged in its own bundle (`stop: "CAP (floor)"`).

## 6. Deviations

Appended to the known-deviations index in `docs/implementation_guide.md` (rows 18–23; additions only, nothing removed):

| # | the guide / specification says | measured, and what Phase 8 did |
|---|---|---|
| 18 | a rolling origin is a training cut plus an evaluation window | it has no validation slice, and the loop needs one; the **12 months before each cut** are carved out of training. Evaluation windows untouched, and train + validation asserted equal to the specification's cut |
| 19 | ≥ 5 seeds × 8 folds; calibration fitted on earlier folds | **3 seeds** (the Phase 6 triple); recalibration and the drift baseline refitted on **each origin's own validation slice**, never carried across origins |
| 20 | a fold is leak-free when max(train) < min(evaluate) | 90-day outcome windows cross every boundary: at each origin the last two validation snapshots' outcomes land in the evaluation window (8,000 arrival rows). **No training row's outcome reaches it.** The fixed split has the identical overlap, so it is measured and reported, not asserted |
| 21 | the backtest runs every fold for every task | measured **51.5 h** for the full grid. Under the 6 h budget stop this phase ran **capacity on origins 1–8 and arrival + fill on origin 1 only**; **fill is deferred and unscheduled** |
| 22 | the 120-epoch cap is slack | arrival h⁰ at 2.5e-4 reaches its best at **77–116 epochs**; one cell (v6 seed 7) stopped at the cap. Raising the cap to 200 if arrival is re-run is permitted and is **not** learning-rate retuning |
| 23 | the drift thresholds can be calibrated from the backtest | **zero** arrival and fill observations above 4.5 pp in all eight windows (§2). The thresholds stay where Addendum B put them, and the `watch` band stays interpolated |

**Deviations from this brief:** none. Stage 8A ran inference only and finished inside its 30-minute budget; Stage 8B
runs capacity for origins 2–8 in both worlds with h⁴, h⁰ and B5; no arrival or fill cell was trained beyond origin 1.

**Not done, and deliberately:** fill is deferred and unscheduled (no depth question — the shipped fill configuration
*is* h⁰); shortage stays diagnostic and unqueued; `inventory_position_weekly` was never read, so Phase 9 stays blocked.
