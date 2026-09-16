# Phase 8 — Evaluation and rolling-origin backtest (guide 8.1–8.2)

<!-- P8_S1 -->

## Header

| | |
|---|---|
| **Commits** | Stage 0a: **`174a2e1`** — the Phase 7 working tree (code, guide, report) committed before anything ran. Phase 8 code: **`ffdcfa2`**, **`2b9bd1c`**, **`c665339`**, **`a0e50d6`**, each committed **before** the cells that use it, so `git status -- ml` was empty when every Phase 8 bundle was stamped. The one untracked file, `docs/Rane_Presentation (1).pdf`, is not project code, lies outside `ml/`, and does not enter the stamp; it was left uncommitted. |
| Device | **MPS**, Apple Silicon, float32, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `num_workers=0`, torch 2.14.0; LightGBM on CPU in a torch-free process |
| Configuration | `ml/configs/shipped.json`, version `phase6-shipped-1`, **unchanged**; no learning rate retuned |
<!-- P8_HEADER -->

### Stage 0b — the Phase 6 bundles

All **22** Phase 6 bundles carry `code_commit = 7f04a3c` and `code_dirty = true` (model version `…-7f04a3c+dirty`): they
were built before `d1ede34` existed. **None was regenerated.**

Phase 7 recorded no hashes, so "unchanged since Phase 7 read them" rests on two measurements
(`ml/artifacts/backtest/phase6_bundle_hashes.json`):

- **every `config.json` was last written between 08:07 and 13:27 on the day, before Phase 7's scorer ran (20:59)**;
- **every SHA-256 equals the value read at the start of Phase 8.**

| bundle | config.json SHA-256 (prefix) | last written | stamp |
|---|---|---|---|
| arrival_week/v6_lite_h4_lr0.00025_s7 | b18cd0fdc18c691e | 08:58 | …-7f04a3c+dirty |
| arrival_week/v6_lite_h4_lr0.00025_s17 | d621b7a6cb0b9003 | 11:50 | …-7f04a3c+dirty |
| arrival_week/v6_lite_h4_lr0.00025_s27 | 11a00c0c9d2cf50e | 10:59 | …-7f04a3c+dirty |
| arrival_week/v6_none_h0_lr0.00025_s7 | 7b0d8dceeebd571e | 10:20 | …-7f04a3c+dirty |
| arrival_week/v6_none_h0_lr0.002_s7 | 68d7b7d54fe9d267 | 09:19 | …-7f04a3c+dirty |
| arrival_week/v7_lite_h4_lr0.00025_s7 | 0c22b2a0cb5cdff8 | 11:00 | …-7f04a3c+dirty |
| arrival_week/v7_lite_h4_lr0.00025_s17 | f0a3410eee21c0fa | 12:25 | …-7f04a3c+dirty |
| arrival_week/v7_lite_h4_lr0.00025_s27 | 777bb94c0f07f171 | 11:41 | …-7f04a3c+dirty |
| arrival_week/v7_none_h0_lr0.00025_s7 | f94497f5d1a4ddd9 | 08:09 | …-7f04a3c+dirty |
| capacity_strain/v6_mp_h4_lr0.00025_s7 | afdcc9206fbc0120 | 10:19 | …-7f04a3c+dirty |
| capacity_strain/v6_mp_h4_lr0.00025_s17 | d542700e0b431bcb | 12:00 | …-7f04a3c+dirty |
| capacity_strain/v6_mp_h4_lr0.00025_s27 | 8c34961b37bf03d3 | 12:24 | …-7f04a3c+dirty |
| capacity_strain/v6_none_h0_lr0.00025_s7 | ceb7d7947fa2288e | 08:40 | …-7f04a3c+dirty |
| capacity_strain/v7_mp_h4_lr0.00025_s7 | be37beccd9dec3c8 | 09:30 | …-7f04a3c+dirty |
| capacity_strain/v7_mp_h4_lr0.00025_s17 | 401058c90f4e7e98 | 12:59 | …-7f04a3c+dirty |
| capacity_strain/v7_mp_h4_lr0.00025_s27 | f77baa4ec1fffacf | 12:57 | …-7f04a3c+dirty |
| fill_rate/v6_none_h0_lr0.000125_s7 | 5d13105238515c22 | 08:26 | …-7f04a3c+dirty |
| fill_rate/v6_none_h0_lr0.000125_s17 | d899b902c3b89a0c | 08:07 | …-7f04a3c+dirty |
| fill_rate/v6_none_h0_lr0.000125_s27 | de1765bc03cce2b4 | 08:07 | …-7f04a3c+dirty |
| fill_rate/v7_none_h0_lr0.000125_s7 | 6708d3e1215003f6 | 08:08 | …-7f04a3c+dirty |
| fill_rate/v7_none_h0_lr0.000125_s17 | 7ff8f2eebbdf0a02 | 13:27 | …-7f04a3c+dirty |
| fill_rate/v7_none_h0_lr0.000125_s27 | 91f5fd30c07bbabc | 13:27 | …-7f04a3c+dirty |

**The fixed-split path is unchanged by the Phase 8 code.** `loop.predict` was re-run on two Phase 6 bundles after the
rolling-origin split was added: fill h⁰ v6 seed 7 serves a distribution **identical** (max |Δ| = 0.0) to its bundle's
recalibrated predictions, drift −0.688 pp (Phase 6: −0.69); arrival SHARE-lite h⁴ v6 seed 7 with its h⁰ reference gives
drift −0.583, h⁰ +0.350, **excess 0.933 pp, `watch`, distribution from the model** — Phase 6's figures — with the
ranking score `allclose` (2.4e-6) to the bundle.

---

## 2. Origins — run and excluded

### How an origin is split

Specification §9.2 gives each origin a training cut and an evaluation window, **and no validation slice**. The loop
cannot run without one: early stopping, recalibration and the drift baseline are all fitted on validation. **The 12
months before each cut are carved out of training as validation** — the length of the fixed split's 2024, so
recalibration sees a full seasonal cycle. The evaluation window is the specification's, untouched, and the assertions
below check that train + validation is exactly the specification's training cut.

### Every origin, both worlds — windows, rows, assertions

Snapshots are 6 weeks apart. Row counts are identical in v6 and v7 (the label tables share their sampling design), so
one table carries both worlds; the assertion column is per world.

| origin | train | validation | evaluation (test) | snapshots train / val / test | arrival & fill rows train / val / test | capacity rows | shortage rows | covid snapshots in train | assertions passed v6 | v7 | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2019-01-01 → 2020-12-31 | > 2020-12-31 → 2021-12-31 | 2022-01-01 → 2022-06-30 | 18 / 8 / 5 | 72,000 / 32,000 / 20,000 | 45,000 / 20,000 / 12,500 | 27,000 / 12,000 / 7,500 | 5 | 24/24 | 24/24 | **runs** |
| 2 | 2019-01-01 → 2021-06-30 | > 2021-06-30 → 2022-06-30 | 2022-07-01 → 2022-12-31 | 22 / 9 / 4 | 88,000 / 36,000 / 16,000 | 55,000 / 22,500 / 10,000 | 33,000 / 13,500 / 6,000 | 5 | 24/24 | 24/24 | **runs** |
| 3 | 2019-01-01 → 2021-12-31 | > 2021-12-31 → 2022-12-31 | 2023-01-01 → 2023-06-30 | 26 / 9 / 4 | 104,000 / 36,000 / 16,000 | 65,000 / 22,500 / 10,000 | 39,000 / 13,500 / 6,000 | 5 | 24/24 | 24/24 | **runs** |
| 4 | 2019-01-01 → 2022-06-30 | > 2022-06-30 → 2023-06-30 | 2023-07-01 → 2023-12-31 | 31 / 8 / 5 | 124,000 / 32,000 / 20,000 | 77,500 / 20,000 / 12,500 | 46,500 / 12,000 / 7,500 | 5 | 24/24 | 24/24 | **runs** |
| 5 | 2019-01-01 → 2022-12-31 | > 2022-12-31 → 2023-12-31 | 2024-01-01 → 2024-06-30 | 35 / 9 / 4 | 140,000 / 36,000 / 16,000 | 87,500 / 22,500 / 10,000 | 52,500 / 13,500 / 6,000 | 5 | 24/24 | 24/24 | **runs** |
| 6 | 2019-01-01 → 2023-06-30 | > 2023-06-30 → 2024-06-30 | 2024-07-01 → 2024-12-31 | 39 / 9 / 4 | 156,000 / 36,000 / 16,000 | 97,500 / 22,500 / 10,000 | 58,500 / 13,500 / 6,000 | 5 | 24/24 | 24/24 | **runs** |
| 7 | 2019-01-01 → 2023-12-31 | > 2023-12-31 → 2024-12-31 | 2025-01-01 → 2025-06-30 | 44 / 8 / 5 | 176,000 / 32,000 / 20,000 | 110,000 / 20,000 / 12,500 | 66,000 / 12,000 / 7,500 | 5 | 24/24 | 24/24 | **runs** |
| 8 | 2019-01-01 → 2024-06-30 | > 2024-06-30 → 2025-06-30 | 2025-07-01 → 2025-09-30 | 48 / 9 / 2 | 192,000 / 36,000 / 8,000 | 120,000 / 22,500 / 5,000 | 72,000 / 13,500 / 3,000 | 5 | 24/24 | 24/24 | **runs** |

**Assertions, per origin × world × task (6 × 4 = 24 each):**

1. `non_empty_folds`
2. `no_leak (train < validation < test, fit window, disjoint)`
3. `spec §9.2 max(train incl. validation) < min(evaluate)`
4. `evaluation window equals the spec's`
5. `train + validation equals the spec's training cut`
6. `folds.assert_rolling_origins (all 8, as defined)`

**Result: 16 of 16 origin × world combinations pass 24 of 24. No origin is excluded.** Full per-task output:
`ml/artifacts/backtest/phase8_origins.json` (printed by `python ml/eval/backtest.py plan`).

**Origin 8 evaluates on 2 snapshots** (2025-08-04, 2025-09-15 — 8,000 arrival rows); every other origin on 4–5.
The fit window runs to 2025-12-08, but the specification ends origin 8 at 2025-09-30, so the last two snapshots are
never evaluated.

### Outcome windows cross the fold boundaries — measured, not asserted

| split | task | training rows whose label window ends in or after validation | validation rows whose label window ends in or after test | training rows ending in or after test | longest label window |
|---|---|---|---|---|---|
| fixed split (every prior phase) | arrival_week | 8,000 | 8,000 | 0 | 90 d |
| every rolling origin | arrival_week | 8,000 | 8,000 | 0 | 90 d |
| fixed split (every prior phase) | fill_rate | 8,000 | 8,000 | 0 | 90 d |
| every rolling origin | fill_rate | 8,000 | 8,000 | 0 | 90 d |
| fixed split (every prior phase) | capacity_strain | 5,000 | 5,000 | 0 | 90 d |
| every rolling origin | capacity_strain | 5,000 | 5,000 | 0 | 90 d |
| fixed split (every prior phase) | shortage_qty | 3,000 | 3,000 | 0 | 90 d |
| every rolling origin | shortage_qty | 3,000 | 3,000 | 0 | 90 d |

Identical at every origin in both worlds: **True**. Labels look 90 days ahead, so the last two snapshots of
each fold have outcomes that land in the next fold. **No training row's outcome reaches the evaluation window.** Validation
rows' outcomes do, which touches early stopping and recalibration, not training. The fixed split has exactly the same
overlap, so asserting it would exclude the split every earlier number was measured on; it is recorded as deviation 20.

---

## 3. Per-task results

<!-- P8_S3 -->

---

## 4. The four questions

<!-- P8_S4 -->

---

## 5. Recommended ship configuration

<!-- P8_S5 -->

---

## 6. Deviations

<!-- P8_S6 -->

---

## 7. Open items carried to Phase 10

<!-- P8_S7 -->
