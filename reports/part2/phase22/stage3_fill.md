# Phase 22 Stage 3 — fill consolidation on clean inputs

**Audience:** whoever owns the fill product (arrives-in-full list, materially-short list, the fill distribution).
**Measured on:** v8 seed 1001, world `v8clean` (every leaking panel column replaced). Snapshot rows as in Phases 19–21
(test 36,000; UC2 base 0.755, UC2b base 0.245). Replication: `v8w1002clean`. TEST, RAW, 5 seeds.
**Instruments:**
- LightGBM arms: `ml/baselines/phase22_proxy.py` (`625d0cc`).
- Family tables: `ml/data/phase22_rows.py` (`cdcb9af`; torch-free copy `75e77d7`).
- Neural: `ml/train/phase22_train.py bind` (season + cadence cells at `2cee433`).
- Scorer: `ml/eval/phase22_fill.py` (gates `83351fa` → `fill_gates.json`; final `2cee433` → `fill_final.json`).

## a. Families on clean BASE (LightGBM, 5-seed bands)

The acknowledgement table covers every PO line: one row per line (1,030,336), `recorded_ts` on each. A snapshot row's
own line has none at t0 (not yet raised), so the gap enters as a channel / supplier statistic as-of t0 (asserted by
`grpstats.Builder`).

| arm | exact CRPS | P(fill = 1) AUC | UC2b P @ 5% | UC2 P @ 5% |
|---|---|---|---|---|
| **BASE_clean** | **0.1407** [0.1406, 0.1408] | **0.5879** | **0.368** | 0.856 |
| + season + cadence | 0.1374 | 0.6354 | 0.435 | 0.904 |
| + ack-gap | 0.1390 | 0.6182 | 0.391 | 0.928 |
| + L4 channel statistics | 0.1437 | 0.6182 | 0.373 | 0.929 |
| + season + cadence + ack-gap | 0.1351 | 0.6641 | 0.461 | 0.945 |
| **+ season + cadence + ack-gap + L4 (consolidated)** | **0.1348** [0.1346, 0.1349] | **0.6664** | 0.455 | 0.945 |
| control: season permuted across snapshots + cadence shuffled across channels | 0.1410 | 0.5811 | 0.347 | 0.859 |
| control: ack-gap, same channel at a donor snapshot | **0.1367** | **0.6418** | 0.419 | 0.945 |
| control: ack-gap, another channel at the same snapshot | 0.1416 | 0.5833 | 0.358 | 0.848 |
| control: L4, another channel at the same snapshot | 0.1451 | 0.5802 | 0.338 | 0.848 |

Gate self-test: equal bands FAIL, and a control that also gains FAILs. Both constructed cases fire.

| family | verdict | why | what the failing control preserves |
|---|---|---|---|
| **season + cadence** | **PASS** | better on CRPS, AUC, UC2b; its control better nowhere | the control keeps only BASE |
| **ack-gap** | **FAIL** | better on all three, but the **snapshot-permuted control (same channel, wrong time) is also better** on CRPS and AUC | keeps the channel's identity, so its typical gap |
| **L4** | **FAIL** | CRPS disjointly worse (AUC better) | — |

**Two Phase 21 fill results are withdrawn on clean inputs.**
- **The acknowledgement gap is a static channel trait, not current information.** Read at the wrong time for the right
  channel, it helps as much. Phase 21's pass used only the cross-channel control.
- **The L4 block** still sharpens the ranking but worsens the distribution.

The consolidated LightGBM arm still beats BASE_clean and season + cadence on all three metrics, but two of its four
families fail their gate.

## b. Neural (MPS, concurrency 1; incumbent fill h0, lr 0.000125, bound through `n_row_feats`)

**Only season + cadence passed**, so the neural consolidated arm *is* the clean Phase 19 fill arm (one set of five cells
serves both, deviation 218). The handshake from Stage −1 applies (inside the stored band).

| | clean incumbent | **clean + season + cadence** | verdict |
|---|---|---|---|
| exact CRPS | 0.1397 [0.1396, 0.1399] | **0.1375** [0.1370, 0.1377] | better |
| P(fill = 1) AUC | 0.6007 [0.5976, 0.6050] | **0.6402** [0.6390, 0.6419] | better |
| UC2b P @ 5% | 0.401 [0.392, 0.412] | **0.464** [0.459, 0.470] | better |
| **verdict** | | | **GAIN** (self-test: the incumbent against itself is TIE on every metric) |

Per seed (minutes, epochs run and best epoch):

| s7 | s17 | s27 | s37 | s47 | s/epoch |
|---|---|---|---|---|---|
| 10.6 (41, 32) | 10.6 (41, 32) | 10.8 (42, 33) | 9.6 (37, 28) | 12.5 (49, 40) | 14.5 |

A first queue, cut by a separator bug, trained **season only** (`_rfseason`, 9.6–14.2 min/seed); it is kept as a
diagnostic and not used (deviation 215).

## c. Blends (weights on validation CRPS; snapshot-block bootstrap, 9 test snapshots)

| predictor (all clean) | CRPS | P(full) AUC | UC2 P @ 5% | UC2b P @ 5% | UC2b per-snapshot min / median / max |
|---|---|---|---|---|---|
| clean incumbent ensemble | 0.1396 | 0.6030 | 0.869 | 0.407 | 0.22 / 0.35 / 0.55 |
| neural season + cadence ensemble | 0.1373 | 0.6417 | 0.913 | 0.476 | 0.23 / 0.40 / 0.55 |
| LightGBM consolidated ensemble | 0.13448 | 0.6685 | 0.944 | 0.459 | 0.23 / 0.41 / 0.54 |
| **consolidated blend (w_neural = 0.02)** | **0.13444** | **0.6688** | **0.944** | 0.461 | 0.23 / 0.42 / 0.54 |
| **clean Phase 19 blend** (neural season + cadence + LightGBM fwd_load, w 0.49) | 0.13576 | 0.6636 | 0.927 | **0.511** | 0.32 / 0.48 / 0.62 |

Consolidated blend minus the clean Phase 19 blend (block, better = positive):

| metric | difference | verdict |
|---|---|---|
| CRPS | **+0.0013** [0.0003, 0.0025] | better |
| AUC | +0.005 [−0.001, +0.011] | undetermined |
| UC2 P @ 5% | **+0.030** [0.008, 0.072] | better |
| UC2b P @ 5% | **−0.048** [−0.081, −0.019] | **worse** |

Validation puts almost no weight on the neural half, so the consolidated blend is the LightGBM consolidated arm.

**Published, reported apart (never compared):** the published Phase 19 blend had CRPS 0.1354, AUC 0.664, UC2 0.927,
UC2b **0.530**. Its clean restatement has UC2b 0.511 (CRPS 0.1358).

**Phase 15 classes.** UC2 is **WATCHLIST** for every predictor: YES at 0.90, but lift ≤ 1.20 on a 0.755 base. UC2b is
**WATCHLIST** (NO, CEILING) for every predictor. **No class changes.**

## d. Replication on world 2 (LightGBM only, own BASE_clean, own bands)

| arm | metric | v8clean gain | v8w1002clean gain | verdict | ratio |
|---|---|---|---|---|---|
| ack-gap | CRPS / AUC / UC2b @ 5% | +0.0017 / +0.030 / +0.022 | +0.0010 / +0.038 / +0.025 | **REPLICATES** (all three) | 0.60 / 1.26 / 1.13 |
| ack-gap gate | | FAIL (snapshot-permuted control gains) | **FAIL** (the same control gains on CRPS and AUC) | **the failure replicates** | |
| consolidated | CRPS / AUC / UC2b @ 5% | +0.0060 / +0.079 / +0.087 | +0.0026 / +0.079 / +0.076 | **REPLICATES** (all three) | 0.43 / 1.00 / 0.87 |

The neural half is not replicated (no neural model on world 2; CPU-only there). One generator: within-generator
replication only.

## e. Persistence

**Declared NOT SERVABLE in this phase.**
- The winning fill predictor (the LightGBM consolidated arm, and the Phase 19-style blend) needs the **season family**.
- Its as-of builder (`ml/data/fwd_load.py`, Phase 18) reads `part_demand_weekly`, which constraint 4 forbids in model
  and serve code.
- A servable fill model needs either a forward-plan source other than `part_demand_weekly`, or an exception to
  constraint 4 that records deviation 162's as_of + 2 days bound.
- **The order-time arrival product (Stage 2e) is servable;** fill is not.
