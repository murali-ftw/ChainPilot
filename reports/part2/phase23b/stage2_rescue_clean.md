# Phase 23B Stage 2 — predict-the-rescue on clean inputs

**Measured on:** world 1 `v8` (seed 1001) and world 2 `v8w1002`, fixed split, TEST, RAW (B1a is not recalibrated).
**CPU only.** LightGBM 4.7.0, frozen `phase7_fit.GBM` (hash `9c0275e581be`, n_jobs 6), seeds 7 17 27 37 47, one CPU job.
**Instruments:** `ml/eval/phase23b_rescue.py` (`ea44b63` fit; `3da031c` score) → `ml/artifacts/phase23b/stage2_fit.json`,
`stage2_score.json`, per-(world, arm) logs `logs/fit_{world}_{arm}.json`, predictions `preds/{world}_{arm}_s{seed}.npz`.

## a. Clean caches

`ml/artifacts/phase22` is **absent** here, so the clean caches were **rebuilt** with Phase 22's builder, unchanged:
- world 2's base cache, by `ml/data/phase20_world.py cache`: 0.3 min;
- `ml/data/clean_panel.py build --world v8` → `cache/v8clean`: 0.3 min;
- `ml/data/clean_panel.py build --world v8w1002` → `cache/v8w1002clean`: 0.2 min.

World 2's CSVs: `data_worlds/v8_seed1002/seed_1002` was absent, and was **copied** from the stored `db/gen_v8/seed_1002`, which
Phase 20 recorded as byte-identical to the generated world (deviation 180). The 7 key tables verify equal by SHA-1
(`cache_builds.json`); nothing was regenerated.

**Phase 22's clean-panel tests**, re-run unchanged through `ml/eval/phase23b_cleancheck.py` on both clean worlds, **PASS**, with
Phase 22's own counts:
- future poison: 0 clean-column changes; the order-keyed offender moves 1,502 / 7,135 channels;
- self-exclusion: 0; the offender admits 1,461–1,546 rows.

## b. LightGBM B1a, five seeds

**Rows:** Phase 17's, re-gated (G2): val 438,048 and test 492,804 rows equal the stored `phase15_sim` rows on part-plant, week
and label; a flipped label fires. **Base rate: test 0.4535.**

**Arms:**
- **PUBLISHED** = the stored leaky cache. **G3: it reproduces the stored Phase 17 `b1a_lgbm_s{s}.npz` bit-exactly on all five
  seeds** (max |Δ| 0.0); its constructed failing case (seed 8 instead of 7) does not reproduce (max |Δ| 0.21).
- **CLEAN** = `v8clean`, the nine columns replaced by Phase 22's as-of-safe columns.
- **NL** = the 18 features from the nine columns removed (19 features).

**Operating point** = Phase 17's: on validation, the highest-precision threshold with validation recall ≥ 0.20, applied to
test.

| arm | precision @ op (5 seeds) | test recall | lift | P @ 1 / 5 / 10 / 20% coverage | 5-seed ensemble @ op, **block 95%** | per-snapshot min / median / max | Phase 15 class |
|---|---|---|---|---|---|---|---|
| sim proxy (Phase 14/15) | 0.619 [0.619, 0.620] | 0.194 | 1.37 | 0.593 / 0.625 / 0.625 / 0.607 | 0.619 [0.580, 0.649] | 0.500 / 0.594 / 0.680 | WATCHLIST (NO, CEILING) |
| own history (as-of 52-wk transfer frequency) | 0.631 | 0.376 | 1.39 | 0.781 / 0.745 / 0.708 / 0.655 | 0.631 [0.583, 0.675] | 0.516 / 0.599 / 0.737 | WATCHLIST (NO, CEILING) |
| **PUBLISHED** (leaky) | **0.823** [0.822, 0.824] | 0.246 | 1.81 | 0.938 / 0.883 / 0.842 / 0.788 | 0.825 [0.791, 0.854] | 0.738 / 0.813 / 0.884 | **ALERT** (PARTIAL @ 0.85, lift 1.84) |
| **CLEAN** | **0.733** [0.7325, 0.7336] | **0.254** | **1.62** | 0.881 / 0.812 / 0.772 / 0.711 | **0.733 [0.697, 0.773]** | 0.653 / 0.728 / 0.840 | **ALERT** (PARTIAL @ 0.80, lift 1.76) |
| NL (columns removed) | 0.712 [0.711, 0.713] | **0.189** | 1.57 | 0.844 / 0.769 / 0.724 / 0.672 | 0.713 [0.669, 0.751] | 0.604 / 0.710 / 0.798 | ALERT (PARTIAL @ 0.70, lift 1.58) |

Majority baseline on the same test rows: accuracy 0.546 ("never acted").

**Paired differences** (5-seed ensembles, each at its own validation-chosen threshold, snapshot-block bootstrap, 1,000
resamples of the 9 test snapshots):

| comparison | Δ precision @ op | block 95% | |
|---|---|---|---|
| clean − published | **−0.091** | [−0.104, −0.075] | worse |
| NL − published | −0.113 | [−0.127, −0.098] | worse |
| clean − NL | +0.022 | [+0.015, +0.031] | better |
| **clean − sim proxy (0.619)** | **+0.115** | **[+0.089, +0.148]** | better |
| **clean − own history (0.631)** | **+0.103** | **[+0.081, +0.125]** | better |

**Per snapshot (precision at the ensemble's operating point, test 2025):**

| snapshot | 01-06 | 02-17 | 03-31 | 05-12 | 06-23 | 08-04 | 09-15 | 10-27 | 12-08 |
|---|---|---|---|---|---|---|---|---|---|
| published | 0.808 | 0.813 | 0.777 | 0.831 | 0.849 | 0.884 | 0.882 | 0.810 | 0.738 |
| **clean** | 0.695 | 0.730 | 0.671 | 0.728 | 0.728 | 0.840 | 0.828 | 0.728 | 0.653 |
| sim proxy | 0.578 | 0.500 | 0.551 | 0.582 | 0.632 | 0.653 | 0.680 | 0.665 | 0.594 |
| own history | 0.583 | 0.579 | 0.592 | 0.599 | 0.646 | 0.714 | 0.737 | 0.659 | 0.516 |

**Verdict against the Phase 17 rule** (GO = precision > 0.70 at recall ≥ 0.20, disjoint from the proxy at five seeds):

- **CLEAN: GO.** 0.733 at recall 0.254, five-seed band [0.7325, 0.7336]. It is disjoint from the proxy and from own history by
  block interval, and it beats both in **9 of 9** test snapshots.
- **It is less comfortable than the published figure.**
  - The snapshot-block interval of the clean ensemble, [0.697, 0.773], reaches **0.697**, just under the bar.
  - Three of nine snapshots sit below 0.70 (January, March, December).
  - The leak was worth **0.09** of precision.
- **NL does not meet the rule:** its test recall at the validation-chosen threshold is 0.189 < 0.20. The as-of-safe
  replacements are worth **+0.022** over dropping the columns.

## c. B1b (neural)

**Not re-scored: needs the Mac.** No GPU is used in this phase.

**What it would need:**
- Phase 17's `ml/train/phase17_b1.py neural` on the clean world (`v8clean` cache; `P5.device_inputs("v8clean", …)` with the
  world registered as Phase 22 does), 5 seeds.
- On the Mac's MPS, roughly 20–40 min per seed at Phase 22's per-epoch rates. Phase 17's B1b ran to the 120-epoch cap here.
- The 3 Phase 17 seeds (27, 37, 47) were never trained (Phase 17 deviation 152), so B1b has no complete published band either.

**B1b read all nine leaking columns (Stage 1); its published 0.859 (2 seeds) is not quotable.**

## d. World 2 (`v8w1002`), clean LightGBM B1a

Own rows, built by the Phase 17 construction: 2,408,120 / 437,840 / 492,570 (train / val / test), 44 / 8 / 9 snapshots,
4,340 part-plants. **Own base rate: test 0.292** (world 1: 0.454). Own bands; nothing borrowed.

| arm | precision @ op | recall | lift | P @ 1 / 5 / 10 / 20% | ensemble block 95% | per-snapshot min / med / max | class |
|---|---|---|---|---|---|---|---|
| own history | 0.454 | 0.193 | 1.55 | 0.571 / 0.494 / 0.463 / 0.425 | [0.414, 0.500] | 0.379 / 0.423 / 0.563 | WATCHLIST |
| **clean B1a** | **0.596** [0.594, 0.597] | 0.191 | **2.04** | 0.752 / 0.653 / 0.589 / 0.515 | [0.554, 0.640] | 0.521 / 0.574 / 0.708 | **WATCHLIST** (NO, CEILING) |

**clean − own history: +0.142 [+0.129, +0.154] → REPLICATES** in sign. It beats own history in 9 of 9 snapshots, and its lift
is **higher** than on world 1 (2.04 vs 1.62).

The absolute bar is a different matter. On world 2's lower base rate the clean model reaches 0.596 at recall 0.191: below
GO's 0.70 and below 0.20 recall. Its Phase 15 class is WATCHLIST, not ALERT. The **direction** replicates; the **alert
level** does not transfer to a world with fewer transfers.

## e. Control

**Cross-snapshot permuted-label control** (training labels permuted uniformly across all training rows, clean features, 5
seeds): precision at the operating point **0.468** against the 0.4535 base rate. Lift 1.03; it collapses (within 0.03). Best
iterations were 1–93.

**Constructed failing case:** the leaked (published) arm beats the control by **+0.355** (block [+0.320, +0.366]) and the
clean arm by +0.254. Both fire.

**G5** (all five seeds required): an arm with one seed dropped is refused.
