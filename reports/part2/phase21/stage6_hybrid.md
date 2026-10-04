# Phase 21 Stage 6 — hybrids with the stored neural ensembles, classes, and slices

**Audience:** whoever decides whether group statistics change what ships.
**Measured on:** v8 seed 1001, TEST, RAW. Weights fitted on VALIDATION only and frozen. Intervals: snapshot-block
bootstrap (1,000 resamples of the 9 test snapshots, paired).
**Instruments:** `ml/eval/phase21_hybrid.py` (`d31947a`) → `ml/artifacts/phase21/stage6_hybrid.json`; slices from
`phase21_score.py snap` (`676ae0c`). Stored parents: incumbent neural ensembles (`bundles/arrival_week/v8_lite_h4…`,
`bundles/fill_rate/v8_none_h0…`), Phase 19 neural ensembles (`phase19/bundles/…_rffwdload+season+cadence`,
`…_rfseason+cadence`), Phase 18 LightGBM + fwd_load (`phase18/preds`, for the Phase 19 blend).

## 6a. Arrival hybrid

Best LightGBM + group arm by **validation** lateness AUC (5-seed means): BASE + season + cadence + L5 (0.7144; L4
0.7066, L5 0.7068). Simplex weights fitted on validation: **incumbent 0.00, Phase 19 neural 0.55, LightGBM + group
0.45** (val lateness AUC 0.7275).

| predictor | lateness AUC | A3 (days) |
|---|---|---|
| **hybrid** | **0.7291** | **12.41** |
| incumbent neural ensemble | 0.7138 | 13.07 |
| Phase 19 neural ensemble | 0.7170 | 12.90 |
| LightGBM + season + cadence + L5 ensemble | 0.7151 | 13.00 |
| **Phase 19 blend** (0.52 neural + 0.48 LightGBM + fwd_load) | **0.7313** | **12.34** |

Hybrid minus each comparator (block 95% interval, better = positive):

| vs | lateness AUC | A3 (days) |
|---|---|---|
| incumbent ensemble | +0.015 [0.013, 0.017] **better** | +0.66 [0.46, 0.86] **better** |
| Phase 19 neural ensemble | +0.012 [0.010, 0.014] better | +0.48 [0.29, 0.66] better |
| LightGBM + group ensemble | +0.014 [0.010, 0.017] better | +0.59 [0.38, 0.78] better |
| **Phase 19 blend** | **−0.002 [−0.004, −0.0002] worse** | **−0.08 [−0.14, −0.03] worse** |

The hybrid is the Phase 19 blend recipe with the LightGBM half carrying season + cadence + L5 instead of fwd_load,
and it is slightly **worse** than that blend. The group block does not replace forward load.

**UC1 decision level** (late vs contract, censoring resolved, base 0.730; precision in the top of the list reported
apart from the AUC, as Phase 20 asked):

| predictor | class | REACHABLE (bar, lift) | P @ 1% / 5% / 10% / 20% | per-snapshot P @ 5% min / median / max |
|---|---|---|---|---|
| hybrid (pred − R) | WATCHLIST | YES (0.90, 1.25) | 0.974 / 0.944 / 0.916 / 0.874 | 0.898 / 0.934 / 0.974 |
| Phase 19 blend | WATCHLIST | YES (0.90, 1.25) | 0.977 / 0.943 / 0.917 / 0.875 | 0.908 / 0.938 / 0.964 |
| Phase 19 neural ensemble (expected week − R) | WATCHLIST | YES (0.90, 1.25) | 0.977 / 0.953 / 0.929 / 0.895 | 0.918 / 0.939 / 0.974 |
| LightGBM + group ensemble | WATCHLIST | YES (0.85, 1.17) | 0.957 / 0.903 / 0.875 / 0.836 | 0.857 / 0.903 / 0.964 |
| **incumbent ensemble P(late)** | WATCHLIST | YES (0.90, 1.25) | **0.991 / 0.969 / 0.952 / 0.912** | 0.939 / 0.969 / 0.985 |

The hybrid vs the incumbent's P(late), paired block at coverage:
- 1%: −0.014 [−0.028, 0.000], undetermined.
- **5%: −0.026 [−0.038, −0.017], worse.**
- 10%: −0.035 [−0.042, −0.028], worse.

**AUC gain again does not mean a better top of the list** (Phase 20's finding, reproduced).

## 6b. Fill hybrid

Best LightGBM + group fill arm by **validation** CRPS: BASE + season + cadence + L5 (0.12185; L5 + ack 0.12528, L5
0.12576, L4 0.12588, ack 0.12668). Weight on the Phase 19 neural season + cadence ensemble, fitted on validation CRPS:
**w = 0.01**. The hybrid **is** the LightGBM arm.

| predictor | exact CRPS | P(full) AUC | UC2b P @ 5% |
|---|---|---|---|
| **hybrid** (≈ LightGBM + season + cadence + L5) | **0.13431** | **0.6701** | 0.478 |
| incumbent neural ensemble | 0.13852 | 0.6240 | 0.464 |
| Phase 19 neural ensemble | 0.13693 | 0.6447 | 0.478 |
| **Phase 19 blend** (0.50 / 0.50) | 0.13537 | 0.6639 | **0.530** |

Hybrid vs the Phase 19 blend (block):
- **CRPS +0.0010 [0.0002, 0.0020], better.**
- AUC +0.006 [−0.002, 0.013], undetermined.
- **UC2b P @ 5%: −0.041 [−0.075, −0.008], worse.**

Vs the incumbent ensemble: CRPS +0.0042 [0.0027, 0.0059] and AUC +0.045 [0.034, 0.061] better; UC2b undetermined.

| list | hybrid | Phase 19 blend | incumbent | block (hybrid − other) |
|---|---|---|---|---|
| UC2 (arrives in full), P @ 5% | **0.943** | 0.927 | 0.881 | vs blend +0.025 [0.003, 0.067] **better**; vs incumbent +0.064 [0.030, 0.095] better |
| UC2b (materially short), P @ 5% | 0.478 | **0.530** | 0.464 | vs blend −0.041 [−0.075, −0.008] **worse**; vs incumbent +0.022 [−0.019, 0.063] undetermined |
| UC2b per-snapshot P @ 5% min / median / max | 0.250 / 0.405 / 0.585 | 0.300 / 0.455 / 0.640 | 0.205 / 0.380 / 0.550 | |

## 6c. Phase 15 classification, old beside new

| use case | old class (Phase 20) | best arm here | new class | changes? |
|---|---|---|---|---|
| UC1 arrival late vs contract | WATCHLIST | incumbent P(late) still best at 5% (0.969); hybrid 0.944 | WATCHLIST (every arm) | **no** |
| UC2 fill = 1 | WATCHLIST | hybrid / LightGBM + season + cadence + L5: YES at 0.90, lift 1.20 | WATCHLIST (lift < 1.5) | **no** |
| UC2b fill < 0.95 | WATCHLIST | Phase 19 blend (0.530) still best at 5% | WATCHLIST (NO, CEILING) | **no** |
| UC1-P at placement (new decision point, Stage 4) | — (not classified before) | flat BASE (no-leak) | **ALERT** (PARTIAL at 0.80, lift 2.2; one month at 0.66) | new; **no group arm improves it** |

**No use case changes class.**

## 6d. Slices (arrival, snapshot; thresholds fixed on validation: T = 47 receipts at L4)

P5's arm is BASE + L5 (no arrival family passed). An earlier scorer run selected the slice arm on **test**; that
selection was removed before any slice was read into a verdict (deviation 202).

| slice | rows (observed) | lateness AUC BASE / BASE + L5 | A3 BASE / L5 (days) | UC1 P @ 5% BASE / L5 | verdict |
|---|---|---|---|---|---|
| cold (n4 = 0) | **0** | — | — | — | empty: no snapshot row has a channel without receipts |
| thin (0 < n4 < 47) | 3,511 (1,624) | 0.7587 / 0.7581 | 12.84 / 12.98 | 0.858 / 0.864 | **tie** on all three |
| normal (n4 ≥ 47) | 32,489 (16,910) | 0.6989 / 0.6976 | 13.28 / 13.31 | 0.897 / **0.879** | tie, tie, **worse** |

Context, no verdict: BASE + season + cadence + L5 vs BASE + season + cadence, lateness AUC.
- Thin: 0.7661 vs 0.7653.
- Normal: 0.7083 vs 0.7096.

There is no thin-channel effect.

By calendar month of t0, the test months carry no seasonal artefact that favours L5. BASE / BASE + L5 lateness AUC
(5-seed means):

| month of t0 | Jan | Feb | Mar | May | Jun | Aug | Sep | Oct | Dec |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 0.725 | 0.706 | 0.718 | 0.714 | 0.701 | 0.704 | 0.703 | 0.695 | 0.693 |
| BASE + L5 | 0.722 | 0.704 | 0.716 | 0.714 | 0.701 | 0.702 | 0.700 | 0.695 | 0.693 |

L5 is level with or below BASE in every month.
