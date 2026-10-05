# Phase 24 Stage 6 — the neural rescue head (B1b) on clean inputs, all five seeds

**Audience:** whoever decides which predict-the-rescue model ships.

**Measured on:** v8 seed 1001 on the clean panel `v8clean` (Phase 22's cache), fixed split; thresholds on validation 2024,
scored on TEST 2025; RAW.

**Machine and run:**
- Apple M4 Pro, **MPS**, concurrency 1;
- seeds 7, 17, 27, 37 and 47, run in that order from 00:36 to 02:58 IST.

**Instruments:**
- `ml/train/phase24_b1b_clean.py` (`632eeb8`): `rows`, `handshake`, `neural --seed s`, `score`.
- Outputs: `ml/artifacts/phase24/b1_rows.json`, `handshake.json`, `bundles/rescue_week/v8clean_mp_h1_lr0.000125_s{seed}/`
  and `b1b_score.json`.
- Logs, one per cell: `b1b_s{seed}.log`.

## Setup: Phase 17's B1b, with only the panel and the bundle root changed

- **Model and training:** the shipped shortage head (`mp h1`, lr 1.25e-4), `max_epochs` 120, patience 8, one optimiser step per
  snapshot.
- **Inputs:** the one-hot week offset through the existing per-row input; a zero orphan row.
- **Selection:** best-validation-PR-AUC state.
- **Wall cap (deviation 264, pre-registered):** 40 minutes per seed. No seed reached it.

| check | result | constructed failing case |
|---|---|---|
| **G6 rows** | rebuilt val (438,048) and test (492,804) rows equal the stored `phase15_sim` rows on part-plant, week and label; train 2,409,264 rows, 44 snapshots. Base rate train 0.455, val 0.405, **test 0.4535** (Phase 17's) | one label flipped: **fires** |
| **G7 clean cache** | `v8clean`: `clean_of = v8`; `replaced_columns` = the nine leaking columns | the leaky cache `v8`: **REFUSED** |
| **G3 Stage −1 handshake** | stored `arrival lite h4 s7` on `v8`, 3 epochs on MPS (83 s): **INSIDE** the stored band every epoch (\|Δ\| vs stored s7 ≤ 1.8e-5). **Proceed: not a within-machine comparison** | (c) +0.05: **OUTSIDE** |
| **G4 five seeds** | the scorer loads only all five seeds | one seed dropped: **RAISED** |

## Training

| seed | epochs | best epoch | best val PR-AUC | stop | minutes |
|---|---|---|---|---|---|
| 7 | 120 | 119 | 0.7066 | **CAP (floor): 120 epochs** | 29.1 |
| 17 | 120 | 116 | 0.7018 | **CAP (floor): 120 epochs** | 28.9 |
| 27 | 120 | 119 | 0.7045 | **CAP (floor): 120 epochs** | 28.9 |
| 37 | 96 | 87 | 0.6918 | patience | 23.1 |
| 47 | 120 | 119 | 0.7054 | **CAP (floor): 120 epochs** | 29.1 |

**Four of five seeds stopped at the 120-epoch cap,** with validation still rising, exactly as Phase 17's two seeds did. Their
figures are **floors**. Each epoch took 14.4–14.6 s.

## Results (TEST 2025, base rate 0.4535)

| arm | precision at the Phase 17 operating point | test recall there | lift | P @ 1 / 5 / 10 / 20% coverage | PR-AUC / ROC-AUC | Phase 15 class | mark |
|---|---|---|---|---|---|---|---|
| **B1b clean, 5 seeds** | **0.791** [0.747, 0.817] | 0.274 [0.205, 0.377] | 1.74 [1.65, 1.80] | 0.916 / 0.864 / 0.824 / 0.769 | 0.715 / 0.765 | **ALERT** (PARTIAL @ 0.85, lift 1.80; Stage B TUNABLE) | RECOMPUTED |
| B1a clean LightGBM, 5 seeds | 0.733 [0.7325, 0.7336] | 0.254 | 1.62 | 0.881 / 0.812 / 0.772 / 0.711 | — | ALERT (PARTIAL @ 0.80, lift 1.76) | QUOTED, `phase-23b.md` §1 (`535361d`) |
| simulation proxy, clean heads (Stage 2) | 0.612 [0.612, 0.613] | 0.197 | 1.35 | 0.590 / 0.622 / 0.621 / 0.601 | 0.545 / 0.621 | WATCHLIST (NO, CEILING) | RECOMPUTED |
| own history (as-of 52-week transfer rate) | 0.631 | 0.376 | 1.39 | — | — | — | RECOMPUTED (Phase 17's rule) |
| LEAKY (superseded, Phase 22): B1b published, 2 of 5 seeds | 0.859 [0.858, 0.859] | 0.238 | — | 0.947 / 0.909 / 0.874 / — | 0.757 / 0.797 | not classified | QUOTED, `phase-17.md` §2.1 |
| LEAKY (superseded, Phase 22): B1a published | 0.823 [0.822, 0.824] | 0.246 | 1.81 | 0.938 / 0.883 / 0.842 / 0.788 | 0.729 / 0.772 | ALERT | QUOTED |

**5-seed ensembles, each at its own validation-chosen operating point** (snapshot-block bootstrap, 1,000 resamples of the 9
test snapshots, paired on identical rows):

| | precision | block 95% |
|---|---|---|
| **B1b clean ensemble** | **0.805** | [0.791, 0.824] |
| clean simulation proxy ensemble | 0.613 | [0.572, 0.643] |
| own history | 0.631 | [0.580, 0.679] |
| **B1b − simulation proxy** | **+0.192** | **[+0.159, +0.236]** |
| **B1b − own history** | **+0.174** | **[+0.131, +0.223]** |
| B1a clean ensemble (QUOTED, not paired) | 0.733 | [0.697, 0.773] |

**Per test snapshot** (B1b ensemble precision at its operating point):
- 2025-01-06 0.857, 02-17 0.817, 03-31 0.796, 05-12 0.789, 06-23 0.791, 08-04 0.853, 09-15 0.807, 10-27 0.784, 12-08 0.788;
- min / median / max **0.784 / 0.797 / 0.857**.
- **Every snapshot is above 0.70.** Clean B1a had 3 of 9 below.

## Verdict vs clean B1a: **GAIN** (by the pre-registered band rule)

- **The band rule:** B1b's five-seed band [0.747, 0.817] lies wholly above B1a clean's [0.7325, 0.7336].
- **Higher on every axis:**
  - precision at every coverage (1, 5, 10, 20%);
  - a higher recall at the operating point (0.274 vs 0.254);
  - the ensemble's block interval [0.791, 0.824] does not overlap B1a's QUOTED [0.697, 0.773].
- **The limit:** B1a clean's predictions are on the other machine, so **no paired block interval of B1b − B1a exists**
  (deviation 270). The seeds also land at different recalls (0.205–0.377), because Phase 17's rule picks the best
  validation-precision point with recall ≥ 0.20 per seed.
- **Taken together:** the comparison at fixed coverage, where both sides use the same rule, agrees with the operating-point
  verdict.

## What it means

**The neural head adds something on clean inputs.**
- It is right about **8 times in 10** at the operating point (0.79 per seed, 0.80 as an ensemble), against 7.3 for the
  LightGBM, 6.1 for the simulation and 4.5 by chance.
- It also catches more of the rescued weeks: about 27% against 25%.
- **It clears the 0.70 bar in every test snapshot,** which removes the "marginal" caveat on clean B1a.
- **Its figures are floors:** four seeds were still improving at the 120-epoch cap.

**Not servable yet.**
- Phase 17's B1b has no serving path.
- Phase 23B's `ml/serve/rescue.py` serves the LightGBM B1a, not this head.
- The bundles exist under `ml/artifacts/phase24/bundles/rescue_week/` (checkpoint, predictions, config with its
  `artifact_identity`), but `shipped.json` is not edited.
- Adopting B1b needs a serving module with an identity guard, and the owner's decision.

## Predictions

- **P4 WRONG.** B1b does not tie B1a; it is a GAIN by the band rule. The wrong prediction is kept.
- **P5 RIGHT.** 4 of 5 seeds stopped at the 120-epoch cap (seeds 7, 17, 27, 47); seed 37 stopped on patience.
