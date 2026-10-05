# Phase 24 — pre-registration

**Committed before any Phase 24 measurement. Never edited afterwards.** Predictions are marked right or wrong in
`reports/part2/phase-24.md`; wrong ones are kept.

**Base:** `HADES-v4-ml-pipeline` tip **`e5ecb3fdbc099d81c0d367d0cff2cccff12e0f0b`** (`reports/part2/merge-23b-report.md` present
at it). **Work branch:** `phase24`, a temporary git worktree (`../HADES_v4_phase24`) cut from that tip, merged back `--no-ff`.

**Machine:** Apple M4 Pro (CPU + MPS).
- Stages 1–5: inference with stored bundles only.
- Stage 6: neural training only, concurrency 1, the incumbent architecture and learning rate.

**Wall-clock stop:** 8 h from 00:00:51 IST on 2026-10-06, i.e. **08:00:51 IST**. Past it, write
`reports/part2/phase24/STOPPED.json`.

**Deviations** are numbered from 264.

## Phase 15 classification rule

Copied verbatim from `reports/part2/phase-15.md` (its reading rules) and from `ml/eval/phase15.py` (the rule itself), as the
Phase 22 and 23B pre-registrations quoted them:

> - **Recall at p** is the TEST recall at the threshold chosen on validation to reach precision p. The TEST precision
>   actually achieved is in brackets. **✓** means the bar held on test for **every** seed with ≥ 50 alerts.
> - **Max precision** is the highest test precision at any coverage with ≥ 50 alerts. It is descriptive, not an
>   operating point.
> - **The base rate is on every row.** "REACHABLE" follows the brief's definition literally, and lift is added beside
>   it, because the definition ignores base rate (deviation 134).

```
STAGE B VERDICT RULES, fixed here before any curve was computed (committed with this file):
  precision is read on a 40-point geometric coverage grid 0.1%..100%, cells with < 50 alerts dropped.
  rho  = Spearman(coverage, precision)   (tunable scores have rho strongly NEGATIVE: tighter coverage, higher precision)
  lift = max precision - base rate;  drop = peak precision - precision at the tightest usable coverage
  TUNABLE        rho <= -0.7 AND drop <= 0.05 AND lift >= 0.10
  NOT TUNABLE    lift < 0.05 (flat)  OR  rho >= 0 (inverted)  OR  drop > 0.10 (peaks early, then falls)
  WEAKLY TUNABLE anything else
  Computed on VALIDATION per seed (the verdict is the majority over seeds) and on TEST (confirmation only).
REACHABLE (Stage G), as the brief defines it: YES if a validation-chosen point gives precision >= 0.85 with recall >= 0.10
and >= 50 alerts on TEST for EVERY seed; PARTIAL if 0.70-0.85; NO, TUNING if Stage B says NOT TUNABLE; NO, CEILING otherwise.
```

**Class mapping:** `phase20_decisions.reachable_and_class`, imported unchanged:
- **ALERT** = REACHABLE YES / PARTIAL with lift ≥ 1.5;
- **WATCHLIST** = not ALERT, and Stage B TUNABLE / WEAKLY TUNABLE;
- **RETIRED** = Stage B NOT TUNABLE.

## What "the simulation's numbers" are (fixed here, from the reports that published them)

The harness is unchanged: `ml/sim/montecarlo.py`, plus `ml/opt/order_policy.py` pinned at `9e2d59d` (deviation 122), driven
exactly as `ml/sim/phase15_sim.py` drives it. Same settings:
- seeds `default_rng(11)` per fill seed;
- N = 200 paths, W = 13 weeks;
- the 9 test (2025) and 9 validation (2024) snapshots;
- fill seeds 7–47; arrival seed 7.

| quantity | published | definition (unchanged) | source |
|---|---|---|---|
| **Q1** pre-rescue match ratio | **1.129×** [1.121, 1.139] | per fill seed: mean over the 9 test snapshots of the simulated below-SS fraction ÷ the PRIVILEGED pre-rescue reference (`phase13_s1.reference(pw, +1)`); 5-seed mean [min, max] | `phase-13.md` S1, `phase13_s1.json` |
| **Q2** rescued weeks missed | **66%** (8,230 / 8,249 / 8,155 / 8,170 / 8,063 of 12,401) | rescued = pre-rescue below SS and observed not below; missed = simulation score < that seed's τ, fitted on VALIDATION (max F1 against the observed reference, `phase14_score.fit_tau`) | `phase-14.md` §2.6, `phase14_sim.json` |
| **Q3** rescue-detection precision (the "simulation proxy") | **0.619** [0.619, 0.620] at test recall 0.194 | score = P_sim(level < SS); label = a transfer-in recorded that week (`acted`); per seed, the highest-precision VALIDATION threshold with validation recall ≥ 0.20, applied unchanged to TEST (Phase 17 `prec_at_min_recall`); detection recall = the TEST recall there | `phase-17.md` §2.1 |
| **Q4** below-SS detector ceiling | **0.28** (0.284 = max over seeds of the test peak) | the highest TEST precision at any coverage cell with ≥ 50 alerts (UC5a, observed reference), `phase15.analyse` | `phase-15.md` §1, `phase15.json` |

**Clean run.** Identical, except that the two neural heads the harness reads are Phase 22's clean bundles:
- `phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7`;
- `phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{7,17,27,37,47}`.

These are the paths `reports/part2/phase23b/stage3_simulation.md` names.

**Intervals:**
- snapshot-block bootstrap: 1,000 resamples of whole test snapshots, seed 2024, paired across the leaky and clean arms on
  identical rows;
- per resample, each quantity is recomputed per fill seed at that seed's fixed threshold and averaged over the five seeds.

**Phase 15 class:** for Q3 and for the UC5a curve, computed with the rule above on both arms.

## Predictions (the owner's, verbatim)

| # | prediction |
|---|---|
| **P1** | the simulation's pre-rescue match ratio stays within 1.129x ± 0.10 on clean model inputs |
| **P2** | its rescue-detection precision drops (it was leaky-fed) but stays below the clean B1a LightGBM (0.733) and below the 0.28 detector ceiling's reading, so predict-the-rescue remains the better rescue answer |
| **P3** | it still misses more than half of the rescued weeks |
| **P4** | the neural B1b on clean inputs, full 5 seeds, TIES the clean LightGBM B1a (disjoint gain is not expected) |
| **P5** | B1b hits its training cap on at least one seed (so figures are floors) |

## Decision rules (fixed here)

**P1 RIGHT** iff the clean Q1 five-seed mean is in [1.029, 1.229].

**P2 RIGHT** iff all three hold. "Stays below 0.733" is read as **Q3 clean < 0.733**: B1a is the rescue model being compared,
quoted from `phase-23b.md`. "Below the 0.28 ceiling's reading" is read as **Q4 clean ≤ 0.284**, the published ceiling.
1. **Q3 drops:** the block interval of (clean − leaky) excludes 0, below 0.
2. **Q3 clean < 0.733** (five-seed mean).
3. **Q4 clean ≤ 0.284.**

**P3 RIGHT** iff the clean Q2 missed share is > 0.5 on every fill seed.

**P4.** Same operating point as Q3 (Phase 17's rule), on the B1 rows.
- **GAIN:** the B1b clean five-seed band lies wholly above B1a clean's published band [0.7325, 0.7336].
- **WORSE:** wholly below.
- **TIE:** the bands overlap.
- **Caveat:** B1a clean's predictions are on the other machine (`ml/artifacts/phase23b` absent here). If they are absent, the
  paired block interval cannot be formed, and the verdict rests on the bands, labelled.
- **P4 RIGHT** iff TIE.

**P5 RIGHT** iff at least one B1b seed stops at its cap: the epoch cap (120, `shipped.json` common) or the wall-clock cap below.

## Stage 6 (B1b clean) settings, fixed here

The run is `ml/train/phase17_b1.py run_neural`, reproduced in a new file with exactly two changes:
- the panel world is **`v8clean`** (Phase 22's clean cache) in place of `v8`;
- the bundle root is `ml/artifacts/phase24/bundles`.

Everything else is unchanged:
- the shipped shortage head `mp h1`, lr 1.25e-4;
- `max_epochs` 120, patience 8;
- one optimiser step per snapshot;
- the one-hot week offset;
- the orphan zero row;
- seeds 7, 17, 27, 37, 47, run in that order, at concurrency 1.

**Rows.** The same construction as `phase17_b1.build_rows`, written to `ml/artifacts/phase24/b1_rows.npz`
(`ml/artifacts/phase17/` is absent on this machine). They are gated element for element against the stored
`phase15_sim/{val,test}_s7.npz` rows.

**Wall-clock cap (deviation 264).** 40 minutes per seed, so that five seeds fit the 8 h stop. Phase 17's cells ran on a
different machine with no wall cap.
- A seed that reaches 40 minutes keeps its best-validation state and is recorded **"CAP (floor)"**, exactly like the
  epoch cap.
- If the stop would be passed before all five seeds finish, the remaining seeds are `not_run` in `STOPPED.json`, and no
  verdict is scored (the scorers require five seeds).

**Stage −1 handshake:** Phase 22's procedure, unchanged.
- The stored config `arrival lite h4 s7` on `v8` is retrained for 3 epochs.
- Each epoch's validation score is compared with the stored 5-seed band at that epoch.
- **INSIDE:** proceed.
- **OUTSIDE:** Stage 6 results are labelled "within-machine comparison".

## Gates and their constructed failing cases (every case is run by the code)

| gate | passes when | constructed failing case (must fire) |
|---|---|---|
| G1 wrapper identity | each bundle the wrapper loads has the expected task, world, seed, arch, depth and lr; a directory named by its identity module; and a `checkpoint.pt` SHA-1 equal to the one pinned in Stage 1 | **(a)** the fill seed-17 bundle placed in the fill seed-7 slot, and the leaky arrival bundle placed in the clean arrival slot, must each RAISE |
| G2 wrapper reproduction | with the ORIGINAL leaky paths, the wrapper reproduces all 45 stored test-snapshot below-SS fractions exactly, the stored `phase15_sim` rows (score, labels, keys) bit-exactly, and Q1–Q4 to their published precision | **(b)** is this gate itself. Its own failing case: the comparator fed one perturbed score (+1e-6 on one row) must report NOT EQUAL; and the CLEAN run must NOT reproduce the stored rows (else the swap did nothing) |
| G3 handshake | Stage −1 inside the stored band | **(c)** the same values shifted by +0.05 must report OUTSIDE |
| G4 five seeds | every scorer refuses an arm without all of seeds 7, 17, 27, 37, 47 | an arm with one seed dropped must RAISE |
| G5 no training in Stages 1–5 | no Stage 1–5 module calls an optimiser step, `.fit(`, `lgb.train(` or `.backward(` | a constructed offender source must be FLAGGED |
| G6 B1 rows identity | rebuilt val / test rows equal the stored `phase15_sim` rows (part-plant, week, label) | one label flipped must make the gate fire |
| G7 clean cache | Stage 6 reads only a cache with `clean_of` set and `replaced_columns == clean_panel.LEAKING` | the leaky cache `v8` must be REFUSED |

## Stops

- Wrapper reproduction (G2) fails: **STOP**. The wrapper is invalid, and no clean simulation number is reported.
- A Phase 22 clean bundle is missing: Stage 2's clean arm writes "MISSING: <file>" and is not run.
- `phase-23b.md` absent: the B1a comparison is not made.
