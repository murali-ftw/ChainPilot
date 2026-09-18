# Phase 8 — Evaluation and rolling-origin backtest (guide 8.1–8.2)

> **Verdict.** The backtest was run under a 6-hour budget stop and two follow-up stages: **capacity on all 8 origins
> in both worlds, arrival on origins 1, 2, 6 and 7, fill on origin 1 only, shortage not at all.**
>
> **Capacity depth (3a): h⁴ wins 10 of 16 windows, ties 2, loses 4 — and nothing measured predicts which.** Not the
> sign of the label shift (it wins and loses in both regimes), not the magnitude beyond the training range
> (Spearman ≈ 0 against the margin), not the season. Splitting by period, h⁴ takes 9 of 12 windows before 2025 and 1 of
> 4 in 2025 — but Stage 8D.3 shows **there is no late-period change in the label to hang that on**: no trend, no step,
> no ceiling, and the only dated generator terms end in mid-2022. **So the 2025 result is not a generator artefact, and
> it is not a regime.** What the data support is that **h⁴'s advantage is unstable across windows**, which is a
> limitation a client can be told and a configuration claim that cannot.
> **Capacity intervals remain not quotable:** 80% coverage 0.72–0.81, below nominal in 12 of 16 windows, in every model
> class, driven by where the evaluation window sits relative to the calibration year (ρ ≈ 0.8).
>
> **Drift thresholds (3b): CLOSED as infeasible.** Zero of 48 arrival and zero of 48 fill observations reach the
> `unusable` band in any of the eight windows, so no amount of training calibrates the threshold that gates the
> fallback.
>
> **The drift gate (3c): remove it.** Gated and always-h⁰ are **not distinguishable in 8 of 8** arrival windows, and
> the gate's own decision flips with the training seed in 4 of those 8. Keep computing and logging drift; stop letting
> it switch the output. This is the **6th** "gate that cannot fail" in the project's record.
>
> **Arrival lateness (3d): the direction survives, the size does not.** The head adds lateness information beyond the
> promise date in **8 of 8** windows, every seed, beating LightGBM every time — but the margin runs **+0.0014 to
> +0.0306** (median +0.02), the two largest are the 2025 windows, and in 3 of 8 it sits inside the promise baseline's
> own bootstrap interval. **The head still does not out-rank the promise date in any window** (−0.18 to −0.22 C-index),
> reproducing Phase 7's binding statement 1 eight more times.
>
> **Fill has no rolling-origin backtest and is deferred and unscheduled. Phase 9 stays blocked:**
> `inventory_position_weekly` was never read.

<!-- P8_S1 -->

## Header

| | |
|---|---|
| **Commits** | Stage 0a: **`174a2e1`** — the Phase 7 working tree (code, guide, report) committed before anything ran. Phase 8 code: **`ffdcfa2`**, **`2b9bd1c`**, **`c665339`**, **`a0e50d6`**, **`90aae35`** (Stage 8A), **`9768e4f`** (Stage 8D's analysis and Stage 8E's epoch-cap flag, per-origin arrival B5 and scorer change), each committed **before** the cells that use it, so `git status -- ml` was empty when every Phase 8 bundle was stamped. **All 36 Stage 8E bundles carry `…-9768e4f` with `code_dirty = false`**; origin 1's arrival and fill bundles carry `…-a0e50d6`, also clean. The one untracked file, `docs/Rane_Presentation (1).pdf`, is not project code, lies outside `ml/`, and does not enter the stamp; it was left uncommitted. |
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
The fit window runs to 2025-12-08, but the specification ends origin 8 at 2025-09-30, so the last two snapshots
(2025-10-27, 2025-12-08) are never evaluated. **Both of origin 8's snapshots fall in the August–September seasonal
peak, and the two it never reaches fall in the autumn trough** — which is why its label shift is the largest of the
eight and why that shift is composition rather than a level change (§4 3a, deviation 24).

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

Every comparison below follows one rule: **a difference exists only when the two seed ranges are disjoint.** Otherwise
the verdict is "not dist.". Bands are per metric, per configuration, per world and per origin, and are never pooled or
borrowed. Figures are mean (spread = max − min across seeds). h⁴ and h⁰ have 3 seeds; LightGBM B5 has 5 fits.

### 3.1 Capacity — all 16 origin × world cells

Mean pinball over P10/P50/P90 (lower is better); `ml/artifacts/backtest/phase8b_capacity.json`.

| world | origin | evaluates | label shift train→test (hundredths) | h⁴ | h⁰ | B5 | evaluation: h⁴ vs h⁰ | validation: h⁴ vs h⁰ | h⁴ vs B5 |
|---|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 2022 H1 | +1.59 | 0.0426 (0.0005) | 0.0452 (0.0008) | 0.0472 (0.0001) | **h⁴** | **h⁴** | **h⁴** |
| v6 | 2 | 2022 H2 | +3.66 | 0.0454 (0.0007) | 0.0499 (0.0005) | 0.0518 (0.0002) | **h⁴** | **h⁴** | **h⁴** |
| v6 | 3 | 2023 H1 | −7.72 | 0.0389 (0.0020) | 0.0397 (0.0004) | 0.0444 (0.0003) | not dist. | not dist. | **h⁴** |
| v6 | 4 | 2023 H2 | −3.52 | 0.0399 (0.0021) | 0.0411 (0.0003) | 0.0453 (0.0001) | **h⁴** | **h⁴** | **h⁴** |
| v6 | 5 | 2024 H1 | −6.25 | 0.0427 (0.0008) | 0.0415 (0.0004) | 0.0463 (0.0001) | **h⁰** | **h⁰** | **h⁴** |
| v6 | 6 | 2024 H2 | +4.42 | 0.0429 (0.0022) | 0.0471 (0.0004) | 0.0493 (0.0001) | **h⁴** | not dist. | **h⁴** |
| v6 | 7 | 2025 H1 | +3.84 | 0.0496 (0.0038) | 0.0471 (0.0007) | 0.0501 (0.0002) | **h⁰** | **h⁴** | not dist. |
| v6 | 8 | 2025 Q3 | +12.00 | 0.0553 (0.0033) | 0.0506 (0.0005) | 0.0523 (0.0002) | **h⁰** | **h⁰** | **B5** |
| v7 | 1 | 2022 H1 | +2.57 | 0.0661 (0.0007) | 0.0683 (0.0010) | 0.0704 (0.0002) | **h⁴** | **h⁴** | **h⁴** |
| v7 | 2 | 2022 H2 | +13.04 | 0.0715 (0.0023) | 0.0799 (0.0005) | 0.0818 (0.0002) | **h⁴** | **h⁴** | **h⁴** |
| v7 | 3 | 2023 H1 | −14.26 | 0.0609 (0.0033) | 0.0617 (0.0007) | 0.0621 (0.0003) | not dist. | **h⁴** | not dist. |
| v7 | 4 | 2023 H2 | −4.19 | 0.0632 (0.0035) | 0.0668 (0.0005) | 0.0733 (0.0001) | **h⁴** | **h⁴** | **h⁴** |
| v7 | 5 | 2024 H1 | −12.07 | 0.0584 (0.0006) | 0.0630 (0.0001) | 0.0670 (0.0002) | **h⁴** | **h⁴** | **h⁴** |
| v7 | 6 | 2024 H2 | +13.24 | 0.0738 (0.0044) | 0.0795 (0.0009) | 0.0835 (0.0004) | **h⁴** | not dist. | **h⁴** |
| v7 | 7 | 2025 H1 | +6.82 | 0.0758 (0.0022) | 0.0816 (0.0009) | 0.0773 (0.0003) | **h⁴** | **h⁴** | **h⁴** |
| v7 | 8 | 2025 Q3 | +22.33 | 0.0949 (0.0042) | 0.0812 (0.0010) | 0.0791 (0.0003) | **h⁰** | **h⁰** | **B5** |

**Counts** (recounted in code by `ml/eval/phase8d_extrapolation.py`; the interim's origins 1–6 row was wrong, see
Appendix A):

| cells | evaluation: h⁴ / tie / h⁰ | h⁴ vs B5: h⁴ / tie / B5 |
|---|---|---|
| origins 1–6 (2022 H1 → 2024 H2), 12 cells | **9 / 2 / 1** (ties v6 o3, v7 o3; loss v6 o5) | **11 / 1 / 0** (tie v7 o3) |
| origins 7–8 (2025), 4 cells | **1 / 0 / 3** (win v7 o7) | **1 / 1 / 2** (tie v6 o7; losses v6 o8, v7 o8) |
| all 16 | **10 / 2 / 4** | **12 / 2 / 2** |

Validation and the evaluation window agree in **12 of 16** cells. A validation-only rule would ship h⁰ on v6 o5, v6 o8
and v7 o8. It would keep h⁴ on **v6 o7**, where validation prefers h⁴ and the evaluation window prefers h⁰.

### 3.2 Capacity intervals — P90 exceedance (nominal 0.10) and 80% coverage (nominal 0.80)

| world | origin | P90 exceedance h⁴ | h⁰ | B5 | 80% coverage h⁴ | h⁰ | B5 |
|---|---|---|---|---|---|---|---|
| v6 | 1 | 0.0814 (0.0178) | 0.0549 (0.0035) | 0.1065 (0.0062) | 0.8073 (0.0067) | 0.8069 (0.0033) | 0.8200 (0.0082) |
| v6 | 2 | 0.1151 (0.0256) | 0.0738 (0.0029) | 0.1089 (0.0024) | 0.7995 (0.0213) | 0.7642 (0.0073) | 0.7578 (0.0079) |
| v6 | 3 | 0.0642 (0.0259) | 0.0658 (0.0028) | 0.0751 (0.0046) | 0.7937 (0.0762) | 0.8126 (0.0027) | 0.7815 (0.0110) |
| v6 | 4 | 0.0631 (0.0188) | 0.0689 (0.0058) | 0.0833 (0.0084) | 0.7677 (0.0450) | 0.7841 (0.0126) | 0.7708 (0.0141) |
| v6 | 5 | 0.0752 (0.0046) | 0.1236 (0.0068) | 0.1083 (0.0006) | 0.7177 (0.0154) | 0.7829 (0.0096) | 0.7633 (0.0099) |
| v6 | 6 | 0.1148 (0.0153) | 0.1684 (0.0116) | 0.1188 (0.0020) | 0.8057 (0.0336) | 0.7977 (0.0153) | 0.7929 (0.0039) |
| v6 | 7 | 0.2209 (0.0590) | 0.1681 (0.0181) | 0.1457 (0.0078) | 0.7601 (0.0467) | 0.8077 (0.0206) | 0.7945 (0.0068) |
| v6 | 8 | 0.2157 (0.0516) | 0.1732 (0.0116) | 0.1467 (0.0096) | 0.7698 (0.0490) | 0.7897 (0.0214) | 0.8074 (0.0118) |
| v7 | 1 | 0.1173 (0.0157) | 0.0755 (0.0066) | 0.1042 (0.0144) | 0.8036 (0.0124) | 0.8285 (0.0117) | 0.8552 (0.0161) |
| v7 | 2 | 0.1460 (0.0532) | 0.0722 (0.0014) | 0.1254 (0.0059) | 0.7976 (0.0227) | 0.7649 (0.0088) | 0.7406 (0.0026) |
| v7 | 3 | 0.0533 (0.0149) | 0.0630 (0.0012) | 0.0600 (0.0035) | 0.7505 (0.0343) | 0.8013 (0.0085) | 0.8097 (0.0079) |
| v7 | 4 | 0.0414 (0.0095) | 0.0501 (0.0050) | 0.0888 (0.0035) | 0.7354 (0.0435) | 0.7787 (0.0154) | 0.7315 (0.0049) |
| v7 | 5 | 0.1107 (0.0322) | 0.1095 (0.0083) | 0.1122 (0.0048) | 0.7998 (0.0231) | 0.7893 (0.0039) | 0.7833 (0.0079) |
| v7 | 6 | 0.1269 (0.0192) | 0.1702 (0.0062) | 0.1430 (0.0172) | 0.8083 (0.0384) | 0.7919 (0.0076) | 0.7685 (0.0187) |
| v7 | 7 | 0.2196 (0.0286) | 0.2035 (0.0055) | 0.1422 (0.0078) | 0.7437 (0.0186) | 0.7822 (0.0070) | 0.8150 (0.0063) |
| v7 | 8 | 0.2043 (0.0126) | 0.1717 (0.0104) | 0.1319 (0.0044) | 0.7825 (0.0108) | 0.7883 (0.0234) | 0.8263 (0.0066) |

- **Origins 1–5: P90 exceedance 4–15%** (h⁴ 4.1–14.6%, h⁰ 5.0–12.4%, B5 6.0–12.5%). It straddles nominal.
- **Origins 6–8: h⁴ 11–22%, h⁰ 17–20%, B5 12–15%.** Every model class exceeds nominal, including the LightGBM that
  shares none of the neural architecture.
- **80% coverage by class:**

  | class | range | windows below nominal (of 16) |
  |---|---|---|
  | h⁴ | 0.72–0.81 | 12 |
  | h⁰ | 0.76–0.83 | 11 |
  | B5 | 0.73–0.86 | 10 |

  Every model class is below nominal in most windows.
- **What drives it** (§4, 8D.4): the evaluation window's level relative to the window the intervals were fitted on.
  It is not depth, and it is not a late-period regime.

### 3.3 Arrival — origins 1, 2, 6 and 7

**Stage 8E: origins 2, 6 and 7 trained for question 3d only** (h⁴ and h⁰, 3 seeds, both worlds, 36 cells,
17.8 GPU-hours / 8.9 h wall on two queues), with origin 1 from the interim stage alongside. Origins 3, 4 and 5 were
not run: 8A settled 3b, and the gate 3c is being removed, so they would add 0.1–2 pp drift the other windows already
sample. **This is a subset, not the backtest the specification asks for.**

**Lateness beyond the promise (ROC-AUC, higher is better).** A row is late when the observed arrival week exceeds the
promise week. The promise-date baseline is one deterministic forecaster, so its band is a bootstrap interval, not a
refit band; the head's is a 3-seed range.

| world | origin | evaluates | late rate | promise date alone | head h⁴ (3 seeds) | h⁰ (3 seeds) | B5 LightGBM (5 fits) | head − promise | × head spread | every head seed > promise | head vs B5 | head vs h⁰ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 2022 H1 | 0.170 | 0.7449 [0.7356, 0.7551] | **0.7498 (0.0015)** | 0.7411 (0.0010) | 0.7444 (0.0012) | **+0.0049** | 3.3× | yes | head | head |
| v6 | 2 | 2022 H2 | 0.119 | 0.7469 [0.7351, 0.7577] | **0.7705 (0.0022)** | 0.7594 (0.0017) | 0.7547 (0.0014) | **+0.0237** | 10.8× | yes | head | head |
| v6 | 6 | 2024 H2 | 0.120 | 0.7439 [0.7314, 0.7562] | **0.7661 (0.0095)** | 0.7410 (0.0040) | 0.7483 (0.0007) | **+0.0222** | 2.3× | yes | head | head |
| v6 | 7 | 2025 H1 | 0.114 | 0.7600 [0.7479, 0.7704] | **0.7906 (0.0013)** | 0.7685 (0.0038) | 0.7670 (0.0008) | **+0.0306** | 22.7× | yes | head | head |
| v7 | 1 | 2022 H1 | 0.195 | 0.7561 [0.7468, 0.7664] | **0.7575 (0.0012)** | 0.7488 (0.0011) | 0.7529 (0.0006) | **+0.0014** | 1.2× | yes | head | head |
| v7 | 2 | 2022 H2 | 0.183 | 0.7401 [0.7298, 0.7514] | **0.7529 (0.0058)** | 0.7540 (0.0021) | 0.7488 (0.0007) | **+0.0128** | 2.2× | yes | head | not dist. |
| v7 | 6 | 2024 H2 | 0.176 | 0.7384 [0.7274, 0.7490] | **0.7548 (0.0068)** | 0.7366 (0.0021) | 0.7453 (0.0007) | **+0.0164** | 2.4× | yes | head | head |
| v7 | 7 | 2025 H1 | 0.145 | 0.7561 [0.7464, 0.7665] | **0.7838 (0.0031)** | 0.7612 (0.0030) | 0.7612 (0.0011) | **+0.0277** | 8.8× | yes | head | head |

**C-index (ranking arrivals, higher is better).**

| world | origin | promise date alone | head h⁴ | h⁰ | head − promise | head out-ranks promise |
|---|---|---|---|---|---|---|
| v6 | 1 | **0.8907** | 0.6833 (0.0007) | 0.6753 (0.0006) | -0.2074 | **no** |
| v6 | 2 | **0.8831** | 0.6661 (0.0030) | 0.6619 (0.0008) | -0.2170 | **no** |
| v6 | 6 | **0.8805** | 0.6749 (0.0011) | 0.6668 (0.0014) | -0.2056 | **no** |
| v6 | 7 | **0.8834** | 0.6703 (0.0013) | 0.6574 (0.0026) | -0.2131 | **no** |
| v7 | 1 | **0.8871** | 0.6727 (0.0023) | 0.6659 (0.0008) | -0.2144 | **no** |
| v7 | 2 | **0.8726** | 0.6628 (0.0046) | 0.6649 (0.0009) | -0.2098 | **no** |
| v7 | 6 | **0.8730** | 0.6894 (0.0038) | 0.6799 (0.0001) | -0.1836 | **no** |
| v7 | 7 | **0.8803** | 0.6735 (0.0009) | 0.6611 (0.0008) | -0.2068 | **no** |

**Training, all 36 Stage 8E cells (deviation 22).** The cap was raised from 120 to 200 for arrival. **Every cell
stopped on patience; not one reached the cap.**

| configuration | epochs run | best epoch | stop reason |
|---|---|---|---|
| h⁴, origins 2/6/7 | 30–90 | 21–81 | patience, 18 of 18 |
| h⁰, origins 2/6/7 | 72–105 | 63–96 | patience, 18 of 18 |

**No Stage 8E cell is a floor, and none is flagged.** The one cap-bound cell in this phase remains **arrival h⁰ v6
seed 7 at origin 1** (120 epochs, best 116), trained before the cap was raised and not re-run; it stays flagged
`stop: "CAP (floor)"` in its own bundle. Every 8E bundle carries a clean `…-9768e4f` stamp with no `+dirty`.

### 3.4 Fill — no rolling-origin results. Deferred and unscheduled.

**Fill has no rolling-origin backtest.** A backtest compares windows, and fill has one. Its shipped configuration was
trained on **origin 1 only** (3 seeds × 2 worlds), while the full backtest's cost was being measured and before the
6-hour budget stop (deviation 21). No other origin was trained, none is queued, and **no statement in this report rests
on fill across windows.**

The one window is recorded so the bundles are not orphaned. It is not evidence of stability:

| world | exact CRPS, recalibrated (3 seeds) | ECE-22, recalibrated | ECE-22, raw | recalibration selected | temperature per seed |
|---|---|---|---|---|---|
| v6 | 0.0574 (0.0000) | 0.0310 (0.0036) | 0.1175 (0.0162) | vector scaling × 3 | 1.134 / 1.181 / 1.107 |
| v7 | 0.0965 (0.0002) | 0.0212 (0.0026) | 0.1173 (0.0420) | vector scaling × 3 | 1.205 / 1.231 / 1.176 |

The temperature range, **1.107–1.231**, is wider than the fixed split's 1.008–1.089 (Phase 6 §6). Every fold needed
T > 1.

**Fill has no depth question.** Its shipped configuration *is* h⁰. The earlier phases' fill claims rest on the 2025
fixed split alone: Phase 7 §5 on calibration parity, and Phase 6 on the served head. **Fill's rolling-origin backtest is
deferred and unscheduled** (§7).

### 3.5 Shortage

Diagnostic only, as in every earlier phase. Not queued, and no result.

---

## 4. The four questions

### 3a — Capacity depth: does h⁴ beat h⁰ outside the band, and on what does it depend?

**Answer: h⁴ beats h⁰ in 10 of 16 windows, ties 2 and loses 4. The advantage is not stable across windows, and no
property of the label that we measured predicts where it fails.**

**1. Not sign-dependent** (Stage 8B.3, stratifying variable recorded before any score existed):

| label shift, train → test | cells | h⁴ / tie / h⁰ |
|---|---|---|
| rising (origins 1, 2, 6, 7, 8 in both worlds) | 10 | 7 / 0 / 3 |
| falling (origins 3, 4, 5 in both worlds) | 6 | 3 / 2 / 1 |

h⁴ wins in both regimes and loses in both. The mechanism proposed in Phases 6 and 7, the shipped head's median moving
*against* the label shift, does not track the outcome either:

- The all-against windows are v6 o4, v6 o7, v7 o4 and v7 o7. h⁴ wins three of the four.
- Origin 8 moves *with* the shift on every seed in both worlds, and h⁴ loses both.
- Totals: h⁴ moves against on 14 of 48 seed-windows, h⁰ on 8 of 48.

**2. The period split (corrected).**

| evaluation period | cells | h⁴ | tie | h⁰ |
|---|---|---|---|---|
| origins 1–6 (2022 H1 → 2024 H2) | 12 | **9** | 2 | **1** |
| origins 7–8 (2025) | 4 | **1** | 0 | **3** |

Against B5, h⁴ wins 11 of 12 on origins 1–6. On 2025 it wins 1, ties 1 and loses 2.

**Phase 7 binding statement 3** was that capacity v6's shipped depth is not distinguishable from the strongest
deployable baseline. It holds where it was measured (2025). It is contradicted on every origin from 1 to 6.

**3. Stage 8D.2: magnitude beyond the training range does not predict the margin.** The test here is magnitude, where
point 1 tested sign. The inputs are computed from the label tables alone, with no model, per origin × world. The
training window is the fold the model's weights were fitted on; validation is excluded.

| world | origin | train min | train p1 | train mean | train p99 | train max | eval mean | z (eval mean vs train) | eval outside train min–max | eval outside train [p1, p99] | above p99 | below p1 | h⁴ − h⁰ pinball | depth verdict | h⁴ − B5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 0.121 | 0.203 | 0.6509 | 1.777 | 2.819 | 0.6668 | +0.05 | 0.05% | 1.35% | 0.66% | 0.69% | −0.0027 | h⁴ | −0.0047 |
| v6 | 2 | 0.121 | 0.207 | 0.6508 | 1.720 | 3.000 | 0.6875 | +0.11 | 0.00% | 1.83% | 1.30% | 0.53% | −0.0045 | h⁴ | −0.0063 |
| v6 | 3 | 0.121 | 0.209 | 0.6647 | 1.732 | 3.000 | 0.5875 | −0.24 | 0.00% | 1.73% | 0.64% | 1.09% | −0.0008 | tie | −0.0055 |
| v6 | 4 | 0.114 | 0.212 | 0.6650 | 1.720 | 3.000 | 0.6298 | −0.11 | 0.00% | 2.07% | 0.93% | 1.14% | −0.0012 | h⁴ | −0.0054 |
| v6 | 5 | 0.114 | 0.214 | 0.6676 | 1.726 | 3.000 | 0.6050 | −0.19 | 0.00% | 2.20% | 0.66% | 1.54% | +0.0011 | **h⁰** | −0.0037 |
| v6 | 6 | 0.114 | 0.213 | 0.6593 | 1.719 | 3.000 | 0.7035 | +0.14 | 0.00% | 2.15% | 1.81% | 0.34% | −0.0042 | h⁴ | −0.0065 |
| v6 | 7 | 0.114 | 0.212 | 0.6560 | 1.719 | 3.000 | 0.6944 | +0.12 | 0.00% | 1.68% | 1.19% | 0.49% | +0.0025 | **h⁰** | −0.0006 |
| v6 | 8 | 0.114 | 0.211 | 0.6517 | 1.715 | 3.000 | 0.7717 | **+0.38** | 0.00% | 1.82% | 1.66% | 0.16% | +0.0047 | **h⁰** | +0.0030 |
| v7 | 1 | 0.090 | 0.208 | 0.8319 | 2.496 | 3.000 | 0.8577 | +0.05 | 0.00% | 0.85% | 0.34% | 0.51% | −0.0022 | h⁴ | −0.0042 |
| v7 | 2 | 0.090 | 0.212 | 0.8264 | 2.480 | 3.000 | 0.9568 | +0.28 | 0.00% | 1.88% | 1.15% | 0.73% | −0.0084 | h⁴ | −0.0103 |
| v7 | 3 | 0.090 | 0.220 | 0.8528 | 2.539 | 3.000 | 0.7102 | −0.30 | 0.00% | 2.03% | 0.31% | 1.72% | −0.0009 | tie | −0.0012 |
| v7 | 4 | 0.090 | 0.221 | 0.8536 | 2.470 | 3.000 | 0.8117 | −0.09 | 0.00% | 2.36% | 0.96% | 1.40% | −0.0035 | h⁴ | −0.0101 |
| v7 | 5 | 0.090 | 0.221 | 0.8654 | 2.480 | 3.000 | 0.7447 | −0.26 | 0.00% | 1.74% | 0.31% | 1.43% | −0.0047 | h⁴ | −0.0086 |
| v7 | 6 | 0.090 | 0.218 | 0.8495 | 2.457 | 3.000 | 0.9819 | +0.28 | 0.00% | 1.50% | 1.25% | 0.25% | −0.0057 | h⁴ | −0.0097 |
| v7 | 7 | 0.090 | 0.216 | 0.8452 | 2.457 | 3.000 | 0.9134 | +0.15 | 0.00% | 1.45% | 1.32% | 0.13% | −0.0058 | h⁴ | −0.0015 |
| v7 | 8 | 0.090 | 0.216 | 0.8368 | 2.422 | 3.000 | 1.0601 | **+0.49** | 0.00% | 1.80% | 1.68% | 0.12% | +0.0137 | **h⁰** | +0.0158 |

Correlation with the h⁴ − h⁰ pinball margin, all 16 cells:

| predictor | Spearman ρ (p) | Pearson r (p) | within v6, Spearman | within v7, Spearman |
|---|---|---|---|---|
| fraction outside train [p1, p99] | +0.10 (0.71) | +0.08 (0.77) | −0.17 | +0.19 |
| fraction above train p99 | −0.02 (0.95) | +0.26 (0.32) | −0.17 | −0.16 |
| fraction above train p99 of train + validation | −0.05 (0.85) | +0.14 (0.60) | −0.19 | −0.10 |
| z-score of the evaluation mean | −0.05 (0.85) | +0.31 (0.25) | +0.10 | −0.17 |
| eval p99 − train p99 | −0.13 (0.63) | +0.17 (0.54) | −0.05 | −0.26 |
| *for reference:* signed label shift, train → test | −0.12 (0.66) | +0.31 (0.25) | +0.10 | −0.17 |

The fraction outside the training min–max has no variance to correlate. It is 0.00% in 15 cells and 0.05% in one,
because the training maximum is the generator's 3.0 label clip in 15 of 16 cells.

**Plainly: the fraction of evaluation labels beyond the training range does not predict the margin.** The prior was
this: v7 o8 (eval mean 1.0601 against train 0.8368) and v6 o8 (0.7717 against 0.6517) are the highest-level windows in
their worlds and the only two cells where h⁴ loses to both h⁰ and B5.

- **The level half of the prior is true.** Origin 8 has the largest evaluation-mean z in both worlds: +0.38 on v6,
  +0.49 on v7.
- **It does not generalise.**
  - The Pearson r of +0.31 is carried almost entirely by those two points, and Spearman is −0.05.
  - v7 o2 and v7 o6 sit at z = +0.28, the next-highest level, and are **h⁴'s two largest wins in v7** (−0.0084 and
    −0.0057).
  - On the range measure itself, origin 8 is unremarkable: 1.82% and 1.80% of labels fall outside the training
    [p1, p99], ranking 5th of 8 in v6 and 4th of 8 in v7.
- **No evaluation window extrapolates in label space.** 0.85–2.36% of evaluation labels fall outside the training
  [p1, p99], about what an unchanged distribution would give (2%).

**The interim's period-based framing therefore stands as a description.** It does not stand as an explanation (next).

**4. Stage 8D.3: the late window is neither a generator artefact nor a regime.**

The mean `capacity_strain` per snapshot across the whole fit window (2019-01-14 → 2025-12-08, 61 snapshots per world)
includes the two snapshots after 2025-09-30 that origin 8 never evaluates.

*Yearly means of the per-snapshot mean:*

| world | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| v6 | 0.603 | 0.698 | 0.696 | 0.676 | 0.611 | 0.654 | **0.696** |
| v7 | 0.748 | 0.916 | 0.900 | 0.902 | 0.767 | 0.863 | **0.908** |

*The seasonal cycle, as the mean over all years of the snapshots falling in each month:*

| world | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 0.593 | 0.604 | 0.646 | 0.623 | 0.648 | 0.747 | 0.745 | **0.772** | 0.742 | 0.670 | 0.591 | 0.558 |
| v7 | 0.708 | 0.718 | 0.802 | 0.734 | 0.825 | 1.054 | 1.050 | **1.126** | 1.049 | 0.878 | 0.705 | 0.629 |

*The late window, snapshot by snapshot. The last two rows are never evaluated:*

| snapshot | v6 mean | v7 mean | in which evaluation window |
|---|---|---|---|
| 2024-07-22 | 0.745 | 1.093 | origin 6 |
| 2024-09-02 | 0.776 | 1.143 | origin 6 |
| 2024-10-14 | 0.665 | 0.892 | origin 6 |
| 2024-11-25 | 0.628 | 0.799 | origin 6 |
| 2025-01-06 | 0.643 | 0.805 | origin 7 |
| 2025-02-17 | 0.644 | 0.795 | origin 7 |
| 2025-03-31 | 0.705 | 0.924 | origin 7 |
| 2025-05-12 | 0.712 | 0.944 | origin 7 |
| 2025-06-23 | 0.769 | 1.098 | origin 7 |
| 2025-08-04 | 0.793 | 1.135 | **origin 8** |
| 2025-09-15 | 0.750 | 0.986 | **origin 8** |
| 2025-10-27 | 0.636 | 0.792 | never evaluated |
| 2025-12-08 | 0.613 | 0.692 | never evaluated |

*Jul–Sep peak snapshots by year, for comparison:*
- **v6:** 2019 0.671–0.708, 2020 0.800–0.837, 2021 0.750–0.799, 2022 0.715–0.821, 2023 0.702–0.726,
  2024 0.745–0.776, 2025 0.750–0.793.
- **v7:** 2019 0.899–1.001, 2020 1.125–1.215, 2021 1.064–1.165, 2022 1.053–1.251, 2023 1.020–1.027,
  2024 1.093–1.143, 2025 0.986–1.135.

**Neither a trend continued from 2019, nor a step, nor a climb toward a ceiling.**

- **No trend.** A linear fit over all 61 snapshots gives:

  | world | slope per year | 95% CI | p | R² |
  |---|---|---|---|---|
  | v6 | +0.004 | −0.007 to +0.015 | 0.42 | 0.011 |
  | v7 | +0.010 | −0.013 to +0.034 | 0.38 | 0.013 |

- **No late step.** The best single change point in either world is **2020-03-09 (COVID)**. It explains 9% / 7% of the
  variance; nothing after it comes close.
- **No ceiling.** 2025's yearly mean matches 2020–2022 in both worlds. 2025's peak snapshots sit *inside* the range of
  the 2020–2022 peaks, not above them. The two post-origin-8 snapshots fall back to the ordinary Oct–Dec trough.
- **What makes origin 8 look high is season plus truncation.** The specification ends origin 8 at 2025-09-30, so its
  evaluation window is **two snapshots, both in the Aug–Sep peak**. Its evaluation mean equals the all-years Aug–Sep
  average: 0.772 on v6 against a 0.757 Aug/Sep average, and 1.060 on v7 against 1.088. Its train → test shift of
  +12 / +22 hundredths is window composition. Every other origin averages 4–5 snapshots across a half-year.

**Generator config** (`db/gen_v6/generator_v6.py` and `db/gen_v7/generator_v7.py`, parameter block and the capacity
and demand definitions only; no generated data read):
- **Demand** carries `trend_per_year = 0.021`. It compounds from 2016-01-04 at a constant rate, with no break, and
  multiplies weekly demand.
- **Capacity** `K` is drawn i.i.d. lognormal per supplier-month. It is reduced by the supplier's latent state, and by
  two regime windows that are the only dated terms: COVID, 2020-03-01 → 2020-09-30, weight 1.0; the chip shortage,
  2021-04-01 → 2022-06-30, weight 0.55. **Nothing is dated after 2022-06-30.**
- **Saturation.** Previous-month overload reduces effective capacity (a carry term, the same in every month), and
  labels are clipped at 3.0. Neither is time-dependent.
- **Seasonality** is fixed amplitude and phase: festive Aug–Nov +0.29, monsoon Jun–Aug +0.12, fiscal March +0.21,
  winter Dec–Jan −0.10.
- **v7 differs from v6** only in the demand-rate process (an AR(1) level component) and a planner re-forecast, and in
  neither is anything time-dated.
- **The configured 2.1%/yr demand trend** would add about 0.014 (v6) or 0.018 (v7) utilisation per year if it passed
  straight through. That lies **inside** the measured slope's confidence interval, so the backtest can neither detect
  nor exclude it. Either way it is a smooth, constant-rate term: it cannot make the last year structurally different
  from the six before it.

**Verdict (8D.3).**
- **The 2025 capacity result is not bound to anything in this generator's late window.** No generator term changes
  after mid-2022. The realised label has no trend, step or ceiling. The 2025 evaluation windows are ordinary seasonal
  windows at levels the training data already contains.
- **So the result is not a generator artefact.** A real extract could show it. It is also **not a "2025 regime"**: the
  interim's reading of it as one is withdrawn.
- **What the data support is weaker, and more useful to state:**
  - h⁴'s advantage over h⁰ is window-dependent.
  - It failed in 4 of 16 half-year windows, 3 of them in 2025.
  - Neither the sign, the magnitude, the range nor the season of the label movement predicts which windows fail.
- **That instability is quotable to a client, as a limitation.** Neither of these is quotable: "h⁴ is better", or
  "h⁴ fails in rising-utilisation periods". This is a synthetic benchmark, so no statement about Rane follows from any
  of it without a Rane backtest.

**5. Stage 8D.4: the exceedance deterioration from origin 6.** The level relative to the fitting windows explains it.
8D.3's answer does not, and needs to: there is no late regime to point to.

| P90 exceedance of | Spearman ρ with the eval-mean z-score | with the validation → evaluation shift | with the train → evaluation shift |
|---|---|---|---|
| h⁴ | **+0.80** (p < 0.001) | +0.65 (0.006) | +0.79 (< 0.001) |
| h⁰ | +0.68 (0.004) | **+0.80** (< 0.001) | +0.66 (0.005) |
| B5 | **+0.82** (< 0.001) | **+0.83** (< 0.001) | +0.79 (< 0.001) |

- **Exceedance tracks how far the evaluation window sits above the windows the model and its intervals were fitted on.**
  From origin 6 onward, every evaluation window is above its own validation year, in both worlds: +8.5 / +20.0 (o6),
  +4.0 / +5.0 (o7), +7.3 / +11.6 (o8) hundredths. That is why the deterioration starts at origin 6 and not origin 7.
- **Origin 2 on v7 fits the same pattern.** It is +0.28 z above training: h⁴ exceeds at 14.6%, B5 at 12.5%. h⁰ does
  not (7.2%); its validation → test shift there is a modest +3.7.
- **The contrast is the finding.** Level predicts under-coverage strongly (ρ ≈ 0.8, every class) and does not predict
  depth at all (ρ ≈ 0).
- **Like 8D.3, this transfers to real data as a mechanism:** intervals under-cover whenever utilisation rises above the
  calibration year. It does not transfer as a period.

### 3b — Drift thresholds: CLOSED as infeasible from this backtest

Stage 8A ran origin 1's 30 trained checkpoints over the inputs of all eight origins: 240 inferences, no training, no
labels on the prediction path. It bounds the drift these windows contain (`phase8a_drift_prepass.json`):

| task | usable ≤ 0.5 pp | watch 0.5–1.0 | degraded 1.0–4.5 | **unusable > 4.5** | largest |
|---|---|---|---|---|---|
| arrival_week (excess over h⁰) | 9 | 9 | 30 | **0 of 48** | 3.93 pp |
| capacity_strain (excess over h⁰) | 9 | 9 | 24 | 6 of 48 | 8.65 pp |
| fill_rate (raw drift; no h⁰ twin) | 9 | 12 | 27 | **0 of 48** | 3.90 pp |

**Closed.**
- **Arrival and fill thresholds cannot be refitted from this backtest at any budget.** The eight windows contain no
  observation in the `unusable` band, so training every origin's own model would fill in the damage axis for 1–4 pp of
  drift and still place nothing where the `degraded`/`unusable` boundary sits.
- **Arrival's thresholds stay where Addendum B put them.** The `watch` band stays interpolated.
- **Capacity has no calibrated threshold to refit.** Its statistic is in utilisation hundredths, not probability. Its
  6 observations above 4.5 pp (v6 o2, v7 o2, v7 o7) come from origin-1 models scored on later windows, so they bound
  input drift and no more.
- Deviation 23.

### 3c — The drift gate: remove it. Keep the statistic; stop letting it switch the output.

**Recommendation: serve h⁰'s recalibrated distribution always, keep SHARE-lite h⁴ for ranking, and keep computing and
logging drift and excess.** The change is Phase 10's to make.

1. **The branches are not distinguishable where they differ.**
   - **Origin 1, week-ECE** (both "not dist."):

     | world | gated | always-h⁰ |
     |---|---|---|
     | v6 | 0.0392 (spread 0.0114) | 0.0448 (0.0038) |
     | v7 | 0.0397 (0.0075) | 0.0409 (0.0039) |

   - **2025 fixed split** (Phase 6 §7): always-h⁰ is identical where the gate engaged (v7, all seeds) and better on v6
     (0.0166 against 0.0247 / 0.0221).
   - **8E's origins.** Now measured on all four arrival origins, not one. **Gated and always-h⁰ are not distinguishable in 8 of 8
   origin × world windows**, and the gate's decision still splits across seeds within a window in 4 of the 8:

     | world | origin | excess pp, 3 seeds | seeds engaging | gated week-ECE | always-h⁰ | verdict |
     |---|---|---|---|---|---|---|
     | v6 | 1 | 1.97 / 0.10 / 0.02 | 1 of 3 | 0.0392 (0.0114) | 0.0448 (0.0038) | not dist. |
     | v6 | 2 | 1.31 / 0.99 / 0.85 | 1 of 3 | 0.1182 (0.0118) | 0.1103 (0.0061) | not dist. |
     | v6 | 6 | 3.30 / 3.06 / 2.69 | 3 of 3 | 0.0422 (0.0064) | 0.0422 (0.0064) | identical — all engaged |
     | v6 | 7 | 0.47 / 0.42 / 0.44 | 0 of 3 | 0.0273 (0.0028) | 0.0259 (0.0045) | not dist. |
     | v7 | 1 | 1.44 / 0.78 / 1.22 | 2 of 3 | 0.0397 (0.0075) | 0.0409 (0.0039) | not dist. |
     | v7 | 2 | 1.39 / 2.00 / 2.08 | 3 of 3 | 0.1001 (0.0030) | 0.1001 (0.0030) | identical — all engaged |
     | v7 | 6 | 0.42 / 1.00 / 2.60 | 2 of 3 | 0.0363 (0.0056) | 0.0360 (0.0045) | not dist. |
     | v7 | 7 | 2.31 / 1.99 / 2.62 | 3 of 3 | 0.0198 (0.0026) | 0.0198 (0.0026) | identical — all engaged |

   In the three windows where every seed engaged, the two policies are **the same object**: the gate served h⁰ anyway.
   In the other five they are not distinguishable. **No window tested separates the two policies.**
2. **The gate's decision flips with the training seed.** Stage 8A, three seeds against the 1.0 pp threshold in 16
   windows:
   - **8 straddle it**: v6 o1, o3, o4, o5, o6, o7; v7 o1, o5;
   - 6 engage on every seed;
   - 2 never engage.

   The thresholded quantity has a seed spread comparable to the threshold.
3. **Validation cannot exercise it, by construction.** Excess drift is movement away from the validation baseline, so
   it is ≈ 0 on validation. Removing the branch is a simplification, not a selection on test.
4. **The argument against, stated fairly.** On v7-2025 the gate served a distribution 3.1–3.9× better calibrated than
   the model's own. Always-h⁰ keeps that benefit. What it gives up is the switch, not the ability to notice, which is
   why the statistic stays logged.

**This is the 6th "gate that cannot fail"** in this project's record:
1. `docs/validation6.md` run 1: the line-stop test that could never fire.
2. Implementation guide §5.2: fill band-coverage G5, whose coverage error pins at 0.5 for any forecaster.
3. `reports/phase3_closeout.md` §9.1: "reproduces `qty_available` exactly", vacuous against an all-zero column.
4. `docs/validation7.md` amendment 01: a gate that could not pass.
5. `docs/validator_amendment_02.md`: a bucket gate that would fire on all three buckets.
6. **Arrival's drift gate at 1.0 pp.** Its output depends on the seed in half the windows, and its two branches are
   not distinguishable.

**What removal does not fix.** h⁰ must still be trained and shipped for every arrival model, because it now serves the
distribution. h⁰ arrival is the configuration that needed the epoch cap raised (deviation 22).

### 3d — Arrival: is the head's lateness advantage beyond the promise date stable across windows?

**Answer: the direction is stable and the size is not.** The head adds lateness information beyond the promise
date in **8 of 8** origin × world windows. Every seed beats the promise date in every window, and the margin is larger
than the head's own seed spread in every window (1.2× to 22.7×). **Arrival's remaining claim is supported by the
backtest in sign — and the +0.033 / +0.025 headline from 2025 is not a typical window.**

**1. The margin is not inside the seed spread anywhere.**

| window | head − promise | × head seed spread |
|---|---|---|
| v7 o1 (2022 H1) | +0.0014 | 1.2× |
| v7 o2 (2022 H2) | +0.0128 | 2.2× |
| v6 o6 (2024 H2) | +0.0222 | 2.3× |
| v7 o6 (2024 H2) | +0.0164 | 2.4× |
| v6 o1 (2022 H1) | +0.0049 | 3.3× |
| v7 o7 (2025 H1) | +0.0277 | 8.8× |
| v6 o2 (2022 H2) | +0.0237 | 10.8× |
| v6 o7 (2025 H1) | +0.0306 | 22.7× |

**2. It varies more than twenty-fold, and the largest margins are the 2025 ones.**

| window | v6 | v7 |
|---|---|---|
| origin 1 — 2022 H1 | +0.0049 | +0.0014 |
| origin 2 — 2022 H2 | +0.0237 | +0.0128 |
| origin 6 — 2024 H2 | +0.0222 | +0.0164 |
| origin 7 — 2025 H1 | +0.0306 | +0.0277 |
| **2025 fixed split** (Phase 7 §7) | **+0.0332** | **+0.0251** |
| median of the four origins | +0.0229 | +0.0146 |

The fixed split's +0.033 / +0.025 is reproduced by origin 7, the other window that evaluates 2025, and is **6 to 20
times** the origin-1 margin. **A client quoted "+3 points of lateness AUC" would be quoted the best window measured,
not the typical one.** The honest range is **+0.001 to +0.031, median +0.02**.

**3. A stricter test, and where it fails.** The promise-date baseline is a single forecaster, so it carries a bootstrap
interval rather than a refit band. Requiring the head's *worst* seed to clear the promise baseline's **upper** bootstrap
bound is stricter than the project's disjointness rule and answers "could this be sampling noise in the baseline?":

- **5 of 8 windows pass:** v6 o2, v6 o6, v6 o7, v7 o6, v7 o7.
- **3 of 8 do not:** **v6 o1 (+0.0049), v7 o1 (+0.0014) and v7 o2 (+0.0128)** — in those the head's advantage is inside
  the promise baseline's own sampling interval.
- Both failures on origin 1 are the earliest window tested (2022 H1).

**4. Against the learned baselines, the head is clean.** It beats B5 LightGBM on lateness in **8 of 8** windows
(+0.0041 to +0.0236, disjoint bands) and h⁰ in 7 of 8 (v7 o2 not distinguishable). B5's own margin over the promise
date is −0.0032 to +0.0087: **the head adds 2 to 5 times what LightGBM adds**, which reproduces Phase 7 §7.

**5. Phase 7 binding statement 1 holds in every window.** The head does not out-rank the promise date on C-index in
**0 of 8** windows; the gap is −0.18 to −0.22, the same as the fixed split's −0.21 / −0.20. Arrival's C-index is still
not evidence that the head orders arrivals better than a planner already can.

**What this means for the claim that would go to the client.**
- **Supported:** the head carries lateness information the promise date does not, in every window tested, in both
  worlds, on 3 seeds, beating LightGBM every time.
- **Not supported:** any particular size. The margin is 20× larger in the best window than the worst, the two 2025
  windows are the largest, and in the earliest window it is inside the baseline's own noise.
- **Still contradicted:** any ranking claim.
- **Untested:** 2023 and 2024 H1 (origins 3, 4, 5), and fill-style calibration of the arrival distribution across
  windows.

---

## 5. Recommended ship configuration

The shipped configuration file (`ml/configs/shipped.json`, `phase6-shipped-1`) is **unchanged by this phase.** The table
states what each task's claim may be defended over. "Period" is the set of synthetic evaluation windows the evidence
covers. "Range" is the condition under which the claim held.

| task | ship | confidence | period the claim is defensible over | range / restriction |
|---|---|---|---|---|
| **capacity_strain** | **h⁴ (message-passing, depth 4, lr 2.5e-4)**, unchanged | **Moderate for ranking the median; low for intervals** | **2022 H1 → 2024 H2 (origins 1–6): h⁴ 9 / tie 2 / h⁰ 1, and beats B5 in 11 of 12.** On **2025** (origins 7–8) it loses to h⁰ in 3 of 4 and to B5 in 2 of 4 | **Restriction:** "h⁴ beats h⁰" may not be quoted as a property of the configuration. It held in 10 of 16 half-year windows, and **nothing measured about a window predicts whether it will hold**: not sign, magnitude, range or season (8D.2–8D.3). Every window tested had its labels inside the training distribution (≤ 2.4% outside [p1, p99]); behaviour on labels outside the training range is **untested**. **Intervals are not quotable**: 80% coverage 0.72–0.81, below nominal in 12 of 16 windows, and P90 exceedance up to 22% whenever utilisation sits above the calibration year (8D.4). Validation-only selection would have demoted h⁴ in 3 cells and kept it wrongly in 1. There is no evidence for a global demotion, so none is made |
| **arrival_week** | **h⁴ SHARE-lite (lr 2.5e-4) for ranking, h⁰ (lr 2.5e-4) recalibrated for the served distribution**, unchanged. **Remove the drift gate in Phase 10 (3c): serve h⁰ always** | **Moderate for the direction of the lateness claim; low for its size; none for ranking** | **Origins 1, 2, 6, 7 (2022 H1, 2022 H2, 2024 H2, 2025 H1) in both worlds, plus the 2025 fixed split.** 2023 and 2024 H1 are untested (origins 3–5 not run) | **Quotable:** the head adds lateness information beyond the promise date — 8 of 8 windows, every seed, beating LightGBM in all 8. **Not quotable:** the size. It ranges **+0.0014 to +0.0306** (median +0.02) and the largest values are the 2025 windows; in 3 of 8 windows it is inside the promise baseline's bootstrap interval. **Never quotable:** that the head ranks arrivals better than the promise date — it loses by 0.18–0.22 C-index in 8 of 8 windows (Phase 7 binding statement 1). The served distribution's calibration across windows is unmeasured |
| **fill_rate** | **h⁰ (lr 1.25e-4), recalibrated**, unchanged | **Unbacktested** | **The 2025 fixed split only** (Phases 6–7), plus one recorded origin-1 window that supports no across-window claim | v7 calibration parity with LightGBM-22 on marginal ECE only; on v6 LightGBM-22 is better calibrated (Phase 7 binding statement 2). The recalibration temperature moved from 1.01–1.09 (fixed split) to 1.11–1.23 (origin 1): **the correction is window-dependent**, and with one window there is no band on how much |
| shortage_qty | not shipped | — | — | diagnostic only |

---

## 6. Deviations

Appended to the known-deviations index in `docs/implementation_guide.md`. Rows 18–23 were written at the interim; rows 21
and 22 are updated for Stage 8E; rows 24–25 are new. Additions only, nothing removed.

| # | the guide / specification says | measured, and what Phase 8 did |
|---|---|---|
| 18 | a rolling origin is a training cut plus an evaluation window | it has no validation slice, and the loop needs one; the **12 months before each cut** are carved out of training. Evaluation windows untouched; train + validation asserted equal to the specification's cut |
| 19 | ≥ 5 seeds × 8 folds; calibration fitted on earlier folds | **3 seeds** (the Phase 6 triple); recalibration and the drift baseline refitted on **each origin's own validation slice**, never carried across origins |
| 20 | a fold is leak-free when max(train) < min(evaluate) | 90-day outcome windows cross every boundary: at each origin the last two validation snapshots' outcomes land in the evaluation window (**8,000 arrival rows, 8,000 fill, 5,000 capacity, 3,000 shortage**). **No training row's outcome reaches it.** The fixed split has the identical overlap, so it is measured and reported, not asserted. Consequence in §7 |
| 21 | the backtest runs every fold for every task | measured **51.5 h** for the full grid. Run under the 6 h budget stop and two follow-up stages: **capacity on origins 1–8; arrival on origins 1, 2, 6 and 7 (Stage 8E, 3d only); fill on origin 1 only — fill's backtest is deferred and unscheduled**; shortage not queued |
| 22 | the 120-epoch cap is slack | arrival h⁰ at 2.5e-4 needs 77–116 epochs at origin 1, where one cell stopped at the cap. **Stage 8E raised the arrival cap to 200** (`backtest.py queue --arrival-max-epochs 200`, written into each bundle's config) for origins 2, 6 and 7. Origin 1's arrival cells keep the 120 cap they were trained under and were not re-run; its one cap-bound cell stays flagged as a floor. The learning rate is 2.5e-4 as shipped: not retuning. **Outcome: no Stage 8E cell reached 200** (h⁴ best epoch 21–81, h⁰ 63–96; all 36 stopped on patience), so the raised cap removed the floor rather than moving any number |
| 23 | the drift thresholds can be calibrated from the backtest | **0 of 48** arrival and **0 of 48** fill observations exceed 4.5 pp across all eight windows (largest 3.93 / 3.90). 3b is **closed as infeasible**; the thresholds stay where Addendum B put them, and the `watch` band stays interpolated |
| 24 | the eight origins are comparable half-year evaluation windows | origin 8's window ends at the specification's 2025-09-30 and holds **2 snapshots, both at the Aug–Sep seasonal peak**; the fit window's last two snapshots (2025-10-27, 2025-12-08) are never evaluated. Origin 8's +12.0 / +22.3 label shift is **seasonal composition** (its mean equals the all-years Aug–Sep mean), not a level change (8D.3). Reported as is; no origin redefined |
| 25 | Phase 8 reports the backtest | the interim (`reports/phase-8-interim.md`) carried four miscounts and one misreading, recounted in code in Stage 8D and corrected in both documents (Appendix A). The misreading, "a 2025 regime", is withdrawn |

**Deviations from the closeout brief:** three, all recorded rather than silent.
1. **The brief's §7 figure "80% coverage below nominal in 13 of 16" is the interim's miscount.** The recount is **12 of
   16** for h⁴ (11 for h⁰, 10 for B5). §7 carries the corrected figure.
2. **Fill is not quite "no rolling-origin results".** One window (origin 1, 3 seeds × 2 worlds) was trained during the
   interim stage and is recorded in §3.4. There is no *across-window* fill result, no fill claim in this report rests
   on it, and fill's backtest stays deferred and unscheduled — but the window exists and is reported rather than
   suppressed.
3. **Stage 8D read one generated table.** 8D.2 requires the capacity labels, so
   `db/gen_v{6,7}/seed_1001/training_labels.csv` was read for `task == capacity_strain` — the label table every phase
   reads, through the same fold code. The generator files were read as configuration only (the parameter block and the
   capacity, demand and label definitions). No other table under `db/gen_v6/**` or `db/gen_v7/**` was read or modified,
   and **`inventory_position_weekly` was never read**.

---

## 7. Open items carried to Phase 10

1. **Deviation 20 bounds the whole backtest.** Labels look 90 days ahead. At every origin, the last two validation
   snapshots' outcomes land inside the evaluation window: **8,000 arrival rows, 8,000 fill, 5,000 capacity, 3,000
   shortage.**
   - **What it touches.** Early stopping selects the checkpoint on those rows. Recalibration and the drift baseline are
     fitted on them. So every origin's model is selected and recalibrated with a sliver of evaluation-window outcome
     information.
   - **What it does not touch.** No training row's outcome reaches the evaluation window.
   - **Why comparisons stay fair.** The fixed split every earlier phase used has exactly the same overlap, and every
     configuration and baseline in a comparison sees the same rows.
   - **Why it still matters.** It bounds every absolute number here, not only the comparisons. A purged split with a
     90-day gap between validation and evaluation would remove it. It was not built in Phase 8 because it would change
     the population behind every earlier number.
2. **Open item 3 — capacity interval coverage — stays open, and capacity intervals are still not quotable.**
   - **Coverage.** 80% coverage is 0.72–0.81 for h⁴, below nominal in 12 of 16 windows; h⁰ 0.76–0.83 (11 of 16);
     B5 0.73–0.86 (10 of 16).
   - **Exceedance.** P90 exceedance reaches 20–22% on origins 7–8 for h⁴ and 17–20% on origins 6–8 for h⁰, in every
     model class.
   - **Cause.** 8D.4 identifies the driver: the evaluation window's level above the fitting windows (ρ ≈ 0.8). So a
     fix has to be level-aware, for example conformal widening keyed to recent drift, or a recalibration refitted on a
     trailing window. Widening at a fixed factor would not do it.
   - *(The brief's "13 of 16" is the interim's miscount; the recount is 12.)*
3. **Seeds: 3 per configuration against the specification's 5** (deviation 19).
   - Every band in this report is a 3-seed range, and several verdicts sit close to band edges (v6 o7 capacity against
     B5, −0.0006).
   - Whether 5 seeds would move any verdict is unmeasured.
4. **Fill has no rolling-origin backtest at all.** One window (origin 1) was recorded and supports no across-window
   claim. Fill's calibration claims rest on the 2025 fixed split alone, and its recalibration temperature already
   differs between the two windows measured. **Deferred and unscheduled.**
5. **Capacity depth is window-dependent, with no identified predictor** (§4 3a). Before any client-facing capacity
   claim, a real extract needs its own rolling-origin backtest. The synthetic finding transfers as "check it per
   window", not as a configuration.
6. **3c is decided but not implemented.** Remove the switch, keep logging drift and excess, ship h⁰ for every arrival
   model.
7. **Arrival's lateness margin is window-dependent, and 3 of 8 windows do not clear the promise baseline's
   bootstrap interval** (§4 3d).
   - The claim's **direction** is backed by 8 windows; its **magnitude** is not, and the 2025 windows are the largest.
   - **Origins 3, 4 and 5 were never run for arrival**, so 2023 and 2024 H1 are untested.
   - Phase 7 open item 1 stays open: whether `promise_week` becomes a head input, or arrival's claim is permanently
     restated as "lateness beyond the promise". This backtest supports the restatement and gives no support to a
     ranking claim.
   - The served arrival distribution's calibration was compared across policies (3c) but not across windows as a
     stability claim.
8. **Phase 9 is still blocked.** `inventory_position_weekly` was never read in Phase 8, in any stage.
9. **Inherited and unchanged:** the Phase 6 bundles' `+dirty` stamps (Stage 0b), and covid snapshots not excluded
   (deviation 12).

---

## Appendix A — corrections to `reports/phase-8-interim.md`

Recounted in code (`ml/eval/phase8d_extrapolation.py`, `recount()`) from `phase8b_capacity.json`; the interim is
corrected in place with a note at its top.

| where | interim said | correct | why |
|---|---|---|---|
| §1 period split, origins 1–6 | h⁴ 10 / tie 1 / h⁰ 1 | **h⁴ 9 / tie 2 / h⁰ 1** | v6 o3 and v7 o3 are both "not dist."; the all-16 row (10/2/4) was right and confirms it |
| §1, h⁴ vs B5 on origins 1–6 | 10 of 12 | **11 of 12** | only v7 o3 is not distinguishable |
| §1 recommendation | "10 of 12 before 2025" | **9 of 12** (2 ties, 1 loss) | follows from row 1 |
| §1.4 80% coverage below nominal | 13 of 16 | **12 of 16** (h⁴) | the table itself shows four windows ≥ 0.80 |
| §1.4 exceedance, origins 1–5 | 4–12% | **4–15%** | v7 o2 h⁴ is 14.6% |
| §1 closing, origin 8 | "the last two snapshots in the fit window"; "whether 2025 is a permanent change" | origin 8 is the **two Aug–Sep peak snapshots**; the fit window's last two are never evaluated; **no late change exists** in the label | 8D.3 |
