# Phase 21 Stage 5 — fill: group fill distributions and the acknowledgement gap

**Audience:** whoever owns the fill product (P(full), the materially-short list).
**Measured on:** v8 seed 1001, TEST (36,000 rows; UC2b base rate 0.245), RAW, 5 seeds. The 22-cell binning is unchanged.
**Instruments:** `phase21_proxy.py --task fill` (fits `6248fae`; season + cadence controls `7169503`);
`phase21_score.py snap` (`676ae0c`) → `score_snap_v8.json`. k = 100 (validation).
**Anchor:** BASE seed 7 reproduces the stored `b5flat22_s7` bit-exactly. BASE + season + cadence is Phase 20's stored
`p20_lgbm_rf` (v8).

## The acknowledgement gap is as-of and exists

- 1,030,336 acknowledgement rows, **one per PO line**: full 807k / partial 202k / rejected 21k.
- Each row has its own `recorded_ts`; the builder admits a row only if it is recorded ≤ t0 (future-poison test PASS).
- A snapshot row's own line has no acknowledgement at t0 (the line is not yet raised), so the gap enters only as a
  **group** statistic: mean gap and share not-full over the channel's / supplier's acknowledged lines, and over those
  still open.

## Arms (5-seed bands)

| arm | exact CRPS | P(fill = 1) AUC | UC2b precision @ 5% | vs BASE |
|---|---|---|---|---|
| **BASE** (shipped `b5flat22`) | **0.13970** [0.1396, 0.1398] | **0.6049** [0.6047, 0.6053] | **0.384** [0.376, 0.389] | — |
| BASE + L4 | 0.1424 [0.1421, 0.1429] | 0.6289 [0.6270, 0.6299] | 0.404 [0.389, 0.422] | CRPS **worse**; AUC and P@5% better (P@5% by 0.0005) |
| BASE + L4, cross-channel shuffle | 0.1435 | 0.5993 | 0.358 | worse |
| **BASE + L5** | 0.1427 [0.1422, 0.1431] | 0.6289 [0.6260, 0.6307] | 0.410 [0.406, 0.415] | CRPS **worse**; AUC and P@5% better |
| BASE + L5, permuted month | 0.1427 | 0.6258 | 0.400 | CRPS worse; AUC better |
| BASE + L5, cross-channel shuffle | 0.1436 | 0.6000 | 0.362 | worse |
| **BASE + ack-gap** | **0.1380** [0.1379, 0.1381] | **0.6333** [0.6324, 0.6342] | **0.416** [0.400, 0.421] | **better on all three** |
| BASE + ack-gap, cross-channel shuffle | 0.1403 | 0.6042 | 0.384 | CRPS worse; the rest undetermined |
| BASE + L5 + ack-gap | 0.1419 [0.1415, 0.1424] | 0.6307 | 0.402 | CRPS worse |
| BASE + season + cadence (Phase 20) | 0.13667 [0.1366, 0.1368] | 0.6433 | 0.440 | better |
| **BASE + season + cadence + L5** | **0.13461** [0.1345, 0.1347] | **0.6681** [0.6674, 0.6693] | **0.472** [0.462, 0.477] | vs season + cadence: **better on all three** |
| BASE + season + cadence + L4 | 0.13459 | 0.6677 | 0.471 | vs season + cadence: better on all three |
| BASE + season + cadence + L5, permuted month | 0.13473 | 0.6665 | 0.466 | vs season + cadence: better on all three |
| BASE + season + cadence + L5, cross-channel shuffle | 0.13692 | 0.6408 | 0.429 | vs season + cadence: undetermined on all three |
| **standalone group distribution** (no ML) | 0.14004 | 0.6002 | 0.355 | block vs BASE ensemble: undetermined on all three |

## Gate verdicts

| family | verdict | why |
|---|---|---|
| **L4** | **FAIL** | CRPS disjointly **worse** (+0.0027), despite AUC +0.024 |
| **L5** | **FAIL** | CRPS disjointly worse (+0.0030), despite AUC +0.024 and P@5% +0.026 |
| control (c): L5 vs L4 | undetermined on all three | the month adds nothing |
| **ack-gap** | **PASS** | CRPS −0.0017, AUC +0.028, P@5% +0.032; its cross-channel shuffle is better than BASE nowhere |
| L5 + ack-gap | FAIL | CRPS worse (the L5 block's calibration cost dominates) |
| season + cadence + L5 vs season + cadence (deviation 201) | **FAIL as pre-registered** | better on all three, but control (a), the **permuted month**, is also better on all three |

**Why season + cadence + L5 fails but is not empty.** The L5 block contains the L4 (channel) columns. So control (a)
removes only the month, and it gains just as much. The rule's reading is exact: **the month is not the information**.
Control (b) shuffles the whole block across channels, and it gains **nothing** (all three undetermined against season + cadence: CRPS 0.13692 vs 0.13667, AUC 0.6408 vs 0.6433).
The season + cadence + **L4** arm equals the L5 arm (0.13459 vs 0.13461).

So the channel-level KM/fill history **does** add to the Phase 19 season + cadence features (CRPS −0.0021, AUC +0.025,
UC2b P@5% +0.032), and the gain disappears when the history is moved to the wrong channel. This is recorded as a
measured result, not a pre-registered pass. On BASE alone the same L4 block hurts CRPS: it sharpens the ranking (AUC)
while miscalibrating the distribution, and season + cadence repairs the calibration.

Gain importance: in BASE + L5 the group block carries **72%** (L5 columns 50%, L1 8%, L2 6%, L4 4%, L3 4%). In
BASE + L5 + ack the acknowledgement block carries 18%.

Snapshot-block intervals (seed ensembles vs the BASE ensemble):
- **ack:** CRPS +0.0017 [0.0014, 0.0020]; AUC +0.029 [0.024, 0.034]; UC2b P@5% +0.033 [0.001, 0.062], all better.
- **season + cadence + L5:** CRPS +0.0053 [0.0029, 0.0082]; AUC +0.062 [0.040, 0.088]; P@5% +0.092 [0.029, 0.156], all better.

## Decision level (Phase 15)

| arm | UC2 (fill = 1, base 0.755) class / REACHABLE / bar / lift / P@5% | UC2b (fill < 0.95, base 0.245) class / REACHABLE / P @ 1% / 5% / 10% / 20% | UC2b per-snapshot P@5% min |
|---|---|---|---|
| BASE, per seed | WATCHLIST / YES / 0.85 / 1.14 / 0.864 | WATCHLIST / NO, CEILING / 0.437 / 0.384 / 0.366 / 0.345 | 0.216 |
| BASE + L5, per seed | WATCHLIST / YES / 0.90 / 1.16 / 0.932 | WATCHLIST / NO, CEILING / 0.441 / 0.410 / 0.383 / 0.355 | 0.246 |
| BASE + ack, per seed | WATCHLIST / YES / 0.90 / 1.20 / 0.933 | WATCHLIST / NO, CEILING / 0.481 / 0.416 / 0.389 / 0.364 | 0.255 |
| BASE + season + cadence + L5, per seed | WATCHLIST / YES / 0.90 / 1.20 / 0.941 | WATCHLIST / NO, CEILING / 0.536 / 0.472 / 0.441 / 0.402 | 0.229 |
| standalone group distribution | WATCHLIST / PARTIAL / 0.85 / 1.12 / 0.907 | WATCHLIST / NO, CEILING / 0.411 / 0.355 / 0.337 / 0.323 | 0.155 |

No fill use case changes class. UC2 reaches the 0.90 bar with every group-carrying arm, but its lift stays ≤ 1.20
(< 1.5) on a 75% base. UC2b stays capped below any alert bar: the best UC2b list here (0.472 at 5%) is still below the
Phase 19 blend's 0.530 (Stage 6). One snapshot ran at 0.23–0.26 for every arm.
