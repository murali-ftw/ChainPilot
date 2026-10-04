# Decision record — shipping after Phase 20 (`phase20-shipped-1`)

**Date:** 2026-10-04 (integration phase). **Configuration:** `ml/configs/shipped.json` → key `phase20_shipped`, version
`phase20-shipped-1`. Appended; every earlier key (`tasks`, `common`, `drift_bands_pp`, `version: phase10-shipped-1`) is
unchanged, and `ml/train/loop.py` still reads them.
**Served by:** `ml/serve/` only. Its identity guard refuses any bundle whose bundle name, `identity_of` and `row_family`
differ from a member listed in the configuration (`ml/tests/test_serve.py`).
**Sources:** `reports/part2/phase-20.md` §5 (shipping table) and the reports each row cites. No new measurement.

## The rule this record follows

A shipped item must be **loadable by the serving path**. If it is not, it is recorded **"declared, not servable"** and is
never served or silently replaced. Where a servable item measured on the same decision is available, it is named as the
**stand-in** and labelled as such.

**Why three items are not servable:** every LightGBM fit in Phases 7–20 stored its **predictions only**. No LightGBM model
file exists, and producing one would be a refit, i.e. training, which the integration phase does not allow. The Phase 17
predict-the-rescue artifacts also exist only on the Windows/CUDA machine.

## Decisions

| item | status | members (bundle identity) | decision metric | reversal condition |
|---|---|---|---|---|
| **arrival.ranked_late_list** (UC1, watchlist) | **servable** | incumbent SHARE-lite h⁴, `arrival_week/v8_lite_h4_lr0.00025_s{7,17,27,37,47}`; mean of S and pT; score P(late) | UC1 precision @ 5% **0.969**, beating the Phase 19 blend 0.943 by +0.027 [0.018, 0.036] (snapshot-block), 9 of 9 snapshots | replace if a challenger beats it at 5% coverage with a block interval excluding 0 |
| **arrival.interval** | **servable** | the same five bundles; split-conformal offsets on the ensemble's expected week: **q10 −31.05 d, q90 +13.68 d** | coverage **0.80** (0.791–0.806 per seed) at 44.7 d wide, vs raw P10–P90 0.93 at 52 d | refit the offsets on validation only if a later year's coverage leaves 0.78–0.82 |
| **arrival.point_estimate** | **declared, not servable** | Phase 19 blend: 0.52 × neural (fwdload + season + cadence) ensemble + 0.48 × LightGBM + fwd_load ensemble | median abs error **12.34 d** vs 13.07 (−0.74 d [0.54, 0.97], block); lateness AUC 0.731 | servable once LightGBM + fwd_load is persisted with an identity and reproduces Phase 18's stored predictions |
| arrival.point_estimate_neural (**stand-in**) | **servable** | `phase19/bundles/arrival_week/v8_lite_h4_lr0.00025_s{…}_rffwdload+season+cadence` | median abs error **12.90 d** vs 13.07 (+0.18 d better [0.01, 0.39], block); lateness AUC 0.717 vs 0.714 | superseded by the declared blend once servable |
| **fill.materially_short_list** (UC2b, watchlist) | **declared, not servable** | Phase 19 blend: 0.50 × neural (season + cadence) + 0.50 × LightGBM + fwd_load | UC2b precision @ 5% **0.530** vs incumbent 0.464 (+0.063 [0.043, 0.090], block); stays WATCHLIST | servable once its LightGBM half is persisted |
| fill.materially_short_list_neural (**stand-in**) | **servable** | `phase19/bundles/fill_rate/v8_none_h0_lr0.000125_s{…}_rfseason+cadence` | Phase 19 **GAIN**: CRPS 0.1371 vs 0.13876, P(full) AUC 0.643 vs 0.620 (disjoint 5-seed bands); UC2b @ 5% 0.478 vs 0.464 (undetermined) | superseded by the declared blend once servable |
| fill.b5flat22 (Phase 10's shipped fill model) | **declared, not servable (superseded)** | LightGBM `b5flat22` (`tasks.fill_rate`, unchanged) | the weakest fill arm on every Phase 20 decision metric: UC2b @ 5% **0.384**, UC2 recall @ p 0.85 **0.189** | see the re-read below |
| **capacity.alert** (UC3, ALERT) | **servable** | incumbent mp h⁴, `capacity_strain/v8_mp_h4_lr0.00025_s{…}`; mean of quantiles; score P(strain > 1) | ALERT: PARTIAL at p 0.85, lift 1.98; precision @ 5% 0.865; **per-snapshot minimum 0.456 (Dec 2025)** | review if precision at the operating point falls below 0.70 on two consecutive quarters, or a challenger beats it on ≥ 3 of 6 UC3 points |
| **rescue.predict_the_rescue** | **declared, not servable** | Phase 17 B1a LightGBM | GO: precision **0.823** [0.822, 0.824] at recall 0.246 (proxy 0.619) | servable once persisted and reproduced |

Every member's `identity` (the `identity_of` dict) is written into `shipped.json` from the bundle's own `config.json`.

## Re-reading the Phase 10 reversal condition for `b5flat22`

Phase 10 shipped LightGBM `b5flat22` for fill on calibration grounds. Its reversal condition, verbatim from
`shipped.json` (`tasks.fill_rate._reversal_condition`):

> *"RE-OPEN THIS if guide 10.2's allocation optimiser becomes the consuming path. An optimiser consumes the whole
> distribution, so it pays the proper score, where the neural head wins 16 of 16; the calibration advantage that justified
> this switch is a marginal-probability property that matters when a planner reads P(fill complete) directly. See
> phase-9b.md B.5."*

**Against the Phase 20 numbers:**
- **The consuming path is not the optimiser.** It is a **ranked materially-short list** (UC2b) and a displayed P(full)
  (UC2), both judged on discrimination at the top of the list. The condition's literal trigger has not fired.
- **Its premise has, in substance.** `b5flat22`'s advantage was calibration that matters "when a planner reads P(fill
  complete) directly". Phase 15–20 showed no fill decision is an alert: P(full) is never used as a yes/no (UC2 lift ≤ 1.28).
  On the list a planner actually reads, `b5flat22` is the **weakest** arm: UC2b precision @ 5% 0.384, against 0.464 for the
  incumbent neural ensemble and 0.530 for the Phase 19 blend (`phase20/stage1_decisions.md`).
- **The calibration argument itself has narrowed.** The Phase 19 blend is better than the incumbent ensemble on CRPS, AUC
  **and** ECE under the snapshot-block bootstrap. LightGBM + fwd_load alone is still the best calibrated (ECE 0.033;
  `phase19/stage4_intervals.md`).
- **And it was never served.** `loop.predict` cannot load it (deviation 46). The neural head was served while the
  configuration named LightGBM.

**Decision:** `b5flat22` is recorded as **superseded and not servable**. Its `tasks.fill_rate` entry is left untouched for the
record. The fill item that ships is the materially-short watchlist (declared blend; served stand-in: the Phase 19 neural
season + cadence ensemble).

## What this record does not do

- It does **not** change `tasks.*`, which `loop.predict` reads. Old serving behaviour is unchanged for any caller of `loop.py`.
- It does **not** train or refit anything. The declared items need a phase that is allowed to fit and **persist** LightGBM
  models.
- Every figure is from the synthetic v8 world (seed 1001) and must be re-measured on Rane's data (`results/observation2.md` §10).
