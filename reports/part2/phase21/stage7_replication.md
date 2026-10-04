# Phase 21 Stage 7 — replication on the second world (v8w1002)

**Audience:** whoever asks whether a Phase 21 result is a property of the method or of one world.
**Measured on:** v8w1002 = `data_worlds/v8_seed1002/seed_1002` (Phase 20's world from v8 seed 1002; **not
regenerated**). TEST, RAW, 5 seeds, its own BASE and its own bands. LightGBM proxy only.
**Instruments:** `grpstats.py build --world v8w1002` (`c2a756c`); `phase21_select.py --world v8w1002` (k: arrival 300,
fill 100; T 45); `phase21_proxy.py` (`cbb1167`, `7169503`); `phase21_score.py snap --world v8w1002` (`676ae0c`);
`phase21_replicate.py` (`dc0ad62`) → `ml/artifacts/phase21/stage7_replication.json`.

**Bytes verified:** `po_lines.csv`, `grn_lines.csv`, `training_labels.csv` and `sourcing_channels.csv` re-hashed equal to
`ml/artifacts/phase20/world2_generation.json`. Phase 20 recorded no hash for the acknowledgement, lane or revision
tables; their hashes are now in `grpstats_v8w1002.json`. **BASE:** seed 7 refitted reproduces Phase 20's stored world-2
BASE **bit-exactly** (arrival and fill); the 5-seed band is Phase 20's stored files.

**Arms added beyond the pre-registered list (deviation 201):** ack and its cross-channel shuffle (v8's only passing
arm: P8 cannot be scored without it), and L5's cross-channel shuffle (control b).

## Verdicts (gain = mean arm − mean BASE, better = positive; each world against its own BASE band)

| task × arm | metric | v8-1001 gain (vs BASE) | v8w1002 gain (vs BASE) | verdict | ratio w2 / v8 |
|---|---|---|---|---|---|
| arrival × L4 | lateness AUC | −0.0015 (worse) | −0.0035 (worse) | the **loss** replicates | 2.28 |
| | A3 | −0.06 d (undet.) | −0.08 d (undet.) | UNDETERMINED | 1.22 |
| arrival × L5 | lateness AUC | −0.0012 (undet.) | −0.0034 (worse) | the loss replicates (disjoint on w2 only) | 2.94 |
| | A3 | −0.04 d (undet.) | −0.09 d (worse) | the loss replicates (w2 only) | 2.31 |
| arrival × L5 permuted month | lateness AUC | −0.0015 (worse) | −0.0036 (worse) | the loss replicates | 2.48 |
| fill × L4 | CRPS | −0.0027 (**worse**) | **+0.0012 (better)** | **DOES NOT** (opposite sign) | −0.43 |
| | P(full) AUC | +0.024 (better) | +0.040 (better) | REPLICATES | 1.69 |
| | UC2b P @ 5% | +0.020 (better) | +0.047 (better) | REPLICATES | 2.37 |
| fill × L5 | CRPS | −0.0030 (worse) | +0.0012 (better) | **DOES NOT** | −0.40 |
| | P(full) AUC | +0.024 (better) | +0.041 (better) | REPLICATES | 1.70 |
| | UC2b P @ 5% | +0.026 (better) | +0.051 (better) | REPLICATES | 1.97 |
| fill × L5 permuted month | all three | as L5 | as L5 | as L5 | |
| **fill × ack-gap** | CRPS | +0.0017 (better) | +0.0008 (better) | **REPLICATES** | **0.497** |
| | P(full) AUC | +0.028 (better) | +0.027 (better) | **REPLICATES** | 0.96 |
| | UC2b P @ 5% | +0.032 (better) | +0.031 (better) | **REPLICATES** | 0.98 |

Gates on world 2:
- **ack PASS**: better on all three; its shuffle better nowhere.
- **L5 FAIL**: better on all three, but both controls gain too. The permuted month gains on all three; the
  cross-channel shuffle gains on AUC and P@5% (+0.007 and +0.020, smaller).
- L5 vs L4: undetermined on every metric (the month adds nothing, as on v8).

**Standalone rules** (one predictor each, block vs that world's BASE ensemble):

| rule | v8-1001 | v8w1002 |
|---|---|---|
| arrival group rule, lateness AUC | 0.680 vs 0.706: worse | 0.661 vs 0.684: worse |
| fill group distribution, CRPS / AUC / P@5% | 0.1400 / 0.600 / 0.355: undetermined vs BASE | 0.0951 / 0.585 / 0.243: **worse** on all three |

## Reading

- **The acknowledgement gap is the one Phase 21 pass, and it replicates** on all three fill metrics at nearly the same
  size (×0.96–0.98 on AUC and precision; ×0.50 on CRPS).
- **The group block's arrival loss replicates**, larger on world 2.
- **The group block's fill ranking gain replicates** (AUC, materially-short precision). Its effect on CRPS **flips
  sign** between worlds, so its calibration effect is world-specific.
- **The order month is inert in both worlds.**
- **What is not replicated:** the neural half of every hybrid (Stage 6), since no neural model exists on world 2 and
  this phase is CPU-only; the at-placement stage; season + cadence + L5 for fill.
- Both worlds come from **one generator** (`generator_v8.py`, self-hash 71de78a). This is within-generator
  replication, not real-world evidence.
