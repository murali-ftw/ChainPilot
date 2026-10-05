# Phase 24 Stage 2 — the shortage simulation restated on clean model inputs

**Audience:** whoever quotes the shortage simulation, or decides between it and predict-the-rescue.

**Measured on:** v8 seed 1001, fixed split (thresholds on validation 2024, scored on TEST 2025). RAW. Fill seeds 7, 17, 27, 37
and 47; arrival seed 7.

**Device:** Apple M4 Pro, MPS, for both arms. The leaky arm reproduces the stored simulation exactly on MPS. The stored runs
recorded no device; their bit-exact reproduction is the evidence that the device matches.

**Instruments:**
- `ml/eval/phase24_sim_clean.py` (`cda8074`): `run --arm leaky`, `run --arm clean`, `score`.
- Outputs: `ml/artifacts/phase24/sim/{leaky,clean}/` (rows, `run_log.json`) and `ml/artifacts/phase24/stage2_sim.json`.
- Logs, one per arm: `sim_leaky.log`, `sim_clean.log`, `sim_score.log`.
- Run time: each arm 10.1 min.

## The wrapper: what it changes, and nothing else

The harness is imported unchanged:
- `ml/sim/montecarlo.py`;
- `ml/opt/order_policy.py` pinned at `9e2d59d` (deviation 122);
- `phase14_sim`'s snapshots and labels;
- the driver loop of `ml/sim/phase15_sim.py`.

Only its UC5 rows are kept, plus a snapshot index for the block bootstrap. The UC8 universes are not needed.

**The only change.** The source of `montecarlo.read_heads` is taken with `inspect`, and its two hard-coded bundle literals are
replaced:

| head | leaky (as published) | clean |
|---|---|---|
| arrival (13-cell) | `bundles/arrival_week/v8_lite_h4_lr0.00025_s7` | `phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7` |
| fill (22-cell), per fill seed | `bundles/fill_rate/v8_none_h0_lr0.000125_s{s}` | `phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}` |

The result is executed in `montecarlo`'s own namespace. The code asserts that every other character of the function is
unchanged.

**What is held fixed:**
- the random draws: `default_rng(11)` per fill seed, snapshots in order;
- the opening positions, the plan and the ROP policy;
- N = 200 paths, W = 13 weeks.

## Gates (each with its constructed failing case, run by the code)

| gate | result | failing case |
|---|---|---|
| **G1 identity** | 12 / 12 bundles match their slot: config fields, identity-module directory name, and `checkpoint.pt` SHA-1 pinned in Stage 1. The heads actually cached by `read_heads` are exactly this arm's six bundles, on both arms | (a) fill s17 in the fill-s7 slot: **RAISED** (seed 17 ≠ 7). The leaky arrival bundle in the clean slot: **RAISED** (world v8 ≠ v8clean) |
| **G2 reproduction** | **PASS.** Leaky through the wrapper reproduces all **45 / 45** stored test-snapshot below-SS fractions exactly. It reproduces all **10 / 10** stored `phase15_sim` row files bit-exactly (score, three labels, keys) | (b) one score perturbed by 1e-6: reported **NOT EQUAL**. The clean arm reproduces **0 / 45** snapshots and **0 / 10** row files, so the swap took effect |
| Q1–Q4 published | Q1 per seed \|Δ\| = **0.0** (1.1291); Q2 rescued / flagged / missed counts **equal** on all five seeds (12,401; missed 8,230 / 8,249 / 8,155 / 8,170 / 8,063); Q3 **0.619** [0.6186, 0.6199] at recall 0.194; Q4 curve, test peaks and Stage B verdict **equal** to `phase15.json` | — |
| G4 five seeds | the scorer loads only complete five-seed arms | one seed dropped: **RAISED** |

## Results: clean vs published (the same rows, the same draws)

| quantity | published (LEAKY (superseded, Phase 22)) | **clean** | clean − leaky | P |
|---|---|---|---|---|
| **Q1** pre-rescue match ratio, 5-seed mean [min, max] | 1.129× [1.121, 1.139] | **1.137×** [1.134, 1.139] | +0.007, **undetermined** (block [−0.019, +0.032]) | P1 |
| **Q2** share of rescued weeks the simulation missed | 65.9% [65.0, 66.5] | **66.2%** [65.8, 66.6] | +0.3 pt, **undetermined** (block [−0.8, +1.3] pt) | P3 |
| … missed rescued weeks per seed (of 12,401) | 8,230 / 8,249 / 8,155 / 8,170 / 8,063 | 8,253 / 8,166 / 8,222 / 8,243 / 8,182 | — | |
| **Q3** rescue-detection precision (Phase 17 operating point) | 0.619 [0.619, 0.620] | **0.612** [0.612, 0.613] | **−0.007** (block **[−0.008, −0.004]**, lower) | P2 |
| … detection recall there | 0.194 [0.192, 0.196] | **0.197** [0.196, 0.198] | +0.003, undetermined | |
| … precision @ 1 / 5 / 10 / 20% coverage | 0.593 / 0.625 / 0.625 / 0.607 | 0.590 / 0.622 / 0.621 / 0.601 | — | |
| **Q4** below-SS detector ceiling (peak test precision, ≥ 50 alerts) | 0.284 | **0.282** | −0.003 | P2 |
| Phase 15 class: rescue detector / below-SS detector | WATCHLIST (NO, CEILING) / WATCHLIST (NO, CEILING); Stage B WEAKLY TUNABLE | **WATCHLIST (NO, CEILING) / WATCHLIST (NO, CEILING)**; Stage B WEAKLY TUNABLE | unchanged | |

**Interval kinds.**
- Bracketed ranges in the clean and published columns are 5-seed [min, max].
- "block" is the paired snapshot-block bootstrap of the difference: 1,000 resamples of the 9 test snapshots, each seed at its
  own validation threshold, averaged over the seeds.

**Snapshot-block intervals of the points:**
- Q1 clean [0.979, 1.307] (published [0.958, 1.303]);
- Q3 clean [0.572, 0.643] (published [0.577, 0.650]).

**Per test snapshot, Q3 precision (5-seed mean):**
- clean 0.576 / 0.495 / 0.544 / 0.579 / 0.629 / 0.646 / 0.673 / 0.655 / 0.592;
- published 0.579 / 0.499 / 0.551 / 0.582 / 0.632 / 0.652 / 0.679 / 0.665 / 0.595.

Clean is lower in every one of the 9 snapshots, by 0.003–0.010.

## Plain answers

**Is the simulation's picture still true on clean inputs? Yes.** Replacing the leaky arrival and fill heads with their clean
retrains changes the simulation by less than its own snapshot-to-snapshot noise:
- **Pre-rescue match:** 1.137× vs 1.129×, a difference the snapshot resampling cannot tell from zero.
- **Rescued weeks missed:** still two-thirds (66.2%).
- **Below-safety-stock ceiling:** 0.28.

The simulation never leaned on the leak much. Its forecast is driven by the opening stock, the plan and the reorder policy; the
heads only shape arrival timing and fill, and the clean heads differ little on those (Phase 22). The leak cost the arrival head
0.012 AUC and the fill head 0.02 AUC.

**Is predict-the-rescue still the better rescue answer? Yes, clearly.** At Phase 17's operating point, against a 0.4535 base:

| | precision |
|---|---|
| the simulation as a rescue detector, clean (RECOMPUTED here) | 0.612 |
| clean B1a LightGBM (QUOTED, `reports/part2/phase-23b.md` §1, `535361d`) | **0.733** [0.7325, 0.7336] |

Clean B1a is still GO, but marginal under block resampling: ensemble [0.697, 0.773], and 3 of 9 snapshots fall below 0.70.

**Phase 23B's paired comparison** (QUOTED, its own block bootstrap): clean B1a minus the simulation proxy (0.619) =
+0.115 [+0.089, +0.148]. Against the clean simulation, 0.612, the gap is about 0.12. That figure is not paired here, because the
B1a predictions are on the other machine.

## Predictions

- **P1 RIGHT.** The clean Q1 five-seed mean is 1.137, inside [1.029, 1.229].
- **P2 RIGHT.**
  1. Q3 drops: −0.007, block [−0.008, −0.004], excluding 0.
  2. Q3 clean 0.612 < 0.733.
  3. Q4 clean 0.282 ≤ 0.284.
- **P3 RIGHT.** The missed share is 65.8–66.6% on every fill seed, above 0.5.
