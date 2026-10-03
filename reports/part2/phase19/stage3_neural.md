# Phase 19 Stage 3 — neural binding of the families that passed Gate v2

**Audience:** whoever decides whether the forward-plan and cadence inputs go into the shipped neural models.
**Measured on:** v8 seed 1001, fixed split, TEST, RAW, 5 seeds (7/17/27/37/47) per arm, against the stored incumbents'
5-seed bands.
**Machine:** Apple M4 Pro, torch 2.14.0, **MPS**, float32. **Concurrency level 1**: one neural cell at a time, recorded
in every bundle's `train_log.json`. Stage 4a's CPU bootstrap ran beside the first capacity cell (wall-clock only).
**Code:** `ml/train/phase19_bind.py` + `ml/train/phase19_identity.py` (`b9b1060`; every cell refused a dirty `ml/`, and
every cell's commit is in its train log), scorer `ml/eval/phase19_neural_score.py` (`11f0c93` → `ml/artifacts/phase19/neural_score.json`).
**Status:** complete. 15 / 15 cells finished at 00:07 IST, before the 06:26 stop, so there is no STOPPED file.
**Verdicts: fill GAIN · arrival TIE · capacity TIE.**

## Handshake (Stage −1): INSIDE the stored band

Stored `arrival_week v8_lite_h4_lr0.00025_s7`, retrained 3 epochs here (`ml/artifacts/phase19/handshake.json`):

| epoch | val C-index here | stored s7 | \|Δ\| | stored 5-seed band |
|---|---|---|---|---|
| 0 | 0.4874943 | 0.4874936 | 6.8e-7 | [0.4801, 0.5115] |
| 1 | 0.5005151 | 0.5005219 | 6.8e-6 | [0.4944, 0.5340] |
| 2 | 0.5554819 | 0.5555247 | 4.3e-5 | [0.5377, 0.5640] |

Training loss agrees to ≤ 3.5e-5. **The stored incumbents are reused as the baseline; none was retrained.**

## What was bound

Through the existing per-row input (`n_row_feats`: per-row features concatenated to the channel encoding before the
head), by a NEW subclass `RowFamilyHeadNet` and a NEW batch wrapper, both bound into `phase5_heads` for the training
process only. The incumbent's own `loop.train` and `loop.finish_bundle` ran unchanged, with the incumbent's architecture,
depth, LR, max epochs (120) and patience (8). Columns were z-scored with the training rows' mean / sd, clipped ±5, NaN → 0,
plus a missing-indicator per column that is ever missing on training rows (all pre-registered).

| use case | families (passed Gate v2) | width | bundle (under `ml/artifacts/phase19/bundles/`) |
|---|---|---|---|
| arrival | fwd_load + fwd_season + cadence | 43 (35 + 8 indicators) | `arrival_week/v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence` |
| fill | fwd_season + cadence | 23 (20 + 3) | `fill_rate/v8_none_h0_lr0.000125_s{s}_rfseason+cadence` |
| capacity | fwd_load | 20 (15 + 5) | `capacity_strain/v8_mp_h4_lr0.00025_s{s}_rffwdload` |

Identity: axis `row_family`, omitted at default. `artifact_identity.py` is untouched; names come from
`phase19_identity`, which appends `_rf…` only when the axis is set, under a separate root (deviation 173).

## Results (RAW, test, 5-seed bands; all five seeds reach every operating point)

| use case · metric | stored incumbent | **Phase 19 bound** | vs incumbent |
|---|---|---|---|
| **fill** · exact CRPS | 0.13876 [0.13866, 0.13882] | **0.1371** [0.1366, 0.1374] | **better** |
| **fill** · P(fill = 1) ROC-AUC | 0.6204 [0.6200, 0.6211] | **0.6431** [0.6424, 0.6440] | **better** |
| fill · ECE-22 (not primary) | 0.124 [0.110, 0.141] | 0.124 [0.095, 0.160] | undetermined |
| **→ fill verdict** | | | **GAIN** |
| arrival · lateness ROC-AUC | 0.7090 [0.7064, 0.7130] | 0.7135 [0.7111, 0.7185] | undetermined |
| arrival · A3 median abs err (days) | 13.06 [12.86, 13.30] | 12.91 [12.76, 13.03] | undetermined |
| arrival · C-index (not primary) | 0.6744 [0.6734, 0.6755] | **0.6776** [0.6771, 0.6785] | better |
| **→ arrival verdict** | | | **TIE** |
| capacity · precision @ 1 / 5 / 10% | 0.904 / 0.851 / 0.810 | 0.923 / 0.886 / 0.847 | all undetermined |
| capacity · recall @ p 0.70 / 0.80 / 0.85 | 0.515 / 0.318 / 0.192 | 0.506 / 0.298 / 0.168 | all undetermined |
| **→ capacity verdict** | | | **TIE** |

Capacity bands in full: precision @ 5% 0.851 [0.832, 0.860] vs **0.886 [0.829, 0.908]**; recall @ 0.70
0.515 [0.459, 0.584] vs 0.506 [0.380, 0.545].

### Per seed

| use case | seed | minutes | epochs (best) | s/epoch | stop | device | concurrency | commit |
|---|---|---|---|---|---|---|---|---|
| capacity | 7 | 4.0 | 14 (5) | 15.0 | patience | mps | 1 | `b9b1060` |
| capacity | 17 | 13.1 | 54 (45) | 14.5 | patience | mps | 1 | `b9b1060` |
| capacity | 27 | 13.5 | 56 (47) | 14.4 | patience | mps | 1 | `b9b1060` |
| capacity | 37 | 13.3 | 55 (46) | 14.4 | patience | mps | 1 | `b9b1060` |
| capacity | 47 | 9.0 | 37 (28) | 14.4 | patience | mps | 1 | `b9b1060` |
| arrival | 7 | 15.1 | 46 (37) | 19.4 | patience | mps | 1 | `b9b1060` |
| arrival | 17 | 17.0 | 52 (43) | 19.4 | patience | mps | 1 | `b9b1060` |
| arrival | 27 | 16.6 | 51 (42) | 19.3 | patience | mps | 1 | `b9b1060` |
| arrival | 37 | 19.9 | 61 (52) | 19.3 | patience | mps | 1 | `b9b1060` |
| arrival | 47 | 17.7 | 54 (45) | 19.4 | patience | mps | 1 | `b9b1060` |
| fill | 7 | 11.2 | 46 (37) | 14.2 | patience | mps | 1 | `b9b1060` |
| fill | 17 | 13.1 | 54 (45) | 14.2 | patience | mps | 1 | `b9b1060` |
| fill | 27 | 12.6 | 52 (43) | 14.2 | patience | mps | 1 | `b9b1060` |
| fill | 37 | 11.2 | 46 (37) | 14.2 | patience | mps | 1 | `b9b1060` |
| fill | 47 | 14.1 | 58 (49) | 14.3 | patience | mps | 1 | `b9b1060` |

The incumbents' stored s/epoch on this machine: arrival 19.2–19.7, fill 14.1–14.2, capacity 14.5–14.9. The per-row input
costs no measurable time.

## Reading

- **Fill: the first disjoint gain over a stored incumbent on these three use cases in Phases 16–19** (every encoder change tied or lost).
  The forward season and the channel's cadence lift P(fill = 1) ROC-AUC by +0.023 and cut exact CRPS by 0.0017, both
  outside the bands. It is an information gain: architecture, LR and depth are the incumbent's.
- **Arrival: tie on the primary metrics.** Every mean moves the right way: lateness AUC +0.0045, A3 −0.15 days, and C-index
  disjointly better (+0.003). But the incumbent's band is wide (0.706–0.713) and the Phase 19 band sits inside its
  upper half. The LightGBM proxy's gain from the same inputs (+0.013 over its own base) does not carry through whole to the
  neural model, which already reads more of the channel history than LightGBM's flat row does.
- **Capacity: tie, with a seed that stopped early.** Seed 7 stopped at epoch 14 (best epoch 5; validation pinball
  0.0692 against 0.0637–0.0652 for the other four). Patience is the frozen incumbent setting and was not changed. The
  5-seed means sit above the incumbent on precision (0.886 vs 0.851 at 5%) and slightly below on recall, but the bands
  span the incumbent's. The seed-ensemble in Stage 4b averages out the single-seed instability.
