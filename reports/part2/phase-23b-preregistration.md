# Phase 23B — pre-registration

**Committed before any Phase 23B measurement. Never edited afterwards.** Predictions are marked right or wrong in
`reports/part2/phase-23b.md`; wrong ones are kept.

**Branch:** `phase23b`, a new git worktree cut from **BASE_SHA = `545e282cb1f5f0a4b3c9168ba379de49f820a3ad`** (`origin/HADES-v4-ml-pipeline`,
"Merge phase22 into HADES-v4-ml-pipeline"; `reports/part2/phase-22.md` and `reports/part2/phase22/stage1_leak.md` present at it).
The owner's paste was left as a placeholder; the owner instructed "run this prompt on the pulled branch", and the pulled
`HADES-v4-ml-pipeline` tip carrying Phase 22 is `545e282`. The checked-out `integration/ml-pipeline-20261004` is an ancestor of it.
**Machine:** Windows 11, NVIDIA RTX 3050 Ti Laptop (4 GB). **CPU only** in this phase: no neural training, no GPU use
(`HADES_DEVICE=cpu` on every run). Python 3.10.11, torch 2.14.0+cu126 (imported, never on CUDA), LightGBM 4.7.0.
Concurrency: one CPU job at a time unless stated. Wall-clock stop: 6 h from 21:19 on 2026-10-05, i.e. **03:19 on 2026-10-06**.

**Data.** World 1 `v8` = `db/gen_v8/seed_1001` (junction to the main checkout, read-only). World 2 `v8w1002` =
`data_worlds/v8_seed1002/seed_1002`, which is absent here; it is materialised by **copying** (not regenerating) the stored
`db/gen_v8/seed_1002`, which Phase 20 recorded as **byte-identical** to the generated world (Phase 20 deviation 180). The copy is
verified table by table against `db/gen_v8/seed_1002` before use.

## Phase 15 classification rule

Copied verbatim from `reports/part2/phase-15.md` (its reading rules) and from `ml/eval/phase15.py` (the rule itself), as Phase 22's
pre-registration quoted them:

> - **Recall at p** is the TEST recall at the threshold chosen on validation to reach precision p. The TEST precision
>   actually achieved is in brackets. **✓** means the bar held on test for **every** seed with ≥ 50 alerts.
> - **Max precision** is the highest test precision at any coverage with ≥ 50 alerts. It is descriptive, not an
>   operating point.
> - **The base rate is on every row.** "REACHABLE" follows the brief's definition literally, and lift is added beside
>   it, because the definition ignores base rate (deviation 134).

```
STAGE B VERDICT RULES, fixed here before any curve was computed (committed with this file):
  precision is read on a 40-point geometric coverage grid 0.1%..100%, cells with < 50 alerts dropped.
  rho  = Spearman(coverage, precision)   (tunable scores have rho strongly NEGATIVE: tighter coverage, higher precision)
  lift = max precision - base rate;  drop = peak precision - precision at the tightest usable coverage
  TUNABLE        rho <= -0.7 AND drop <= 0.05 AND lift >= 0.10
  NOT TUNABLE    lift < 0.05 (flat)  OR  rho >= 0 (inverted)  OR  drop > 0.10 (peaks early, then falls)
  WEAKLY TUNABLE anything else
  Computed on VALIDATION per seed (the verdict is the majority over seeds) and on TEST (confirmation only).
REACHABLE (Stage G), as the brief defines it: YES if a validation-chosen point gives precision >= 0.85 with recall >= 0.10
and >= 50 alerts on TEST for EVERY seed; PARTIAL if 0.70-0.85; NO, TUNING if Stage B says NOT TUNABLE; NO, CEILING otherwise.
```

Class mapping: `phase20_decisions.reachable_and_class`, imported unchanged (ALERT = REACHABLE YES / PARTIAL with lift ≥ 1.5;
WATCHLIST = not ALERT and Stage B TUNABLE / WEAKLY TUNABLE; RETIRED = Stage B NOT TUNABLE).

## Predictions (the owner's, verbatim)

| # | prediction |
|---|---|
| **P1** | the predict-the-rescue LightGBM B1a read at least one of the nine Phase 22 leaking panel columns |
| **P2** | clean, its precision at recall ~0.246 falls by < 0.05 (stays above the 0.619 simulation proxy and the 0.631 own-history baseline, block-disjoint) and stays GO |
| **P3** | the shortage simulation's inputs (arrival and fill predictions) were leaky, and re-running it on clean predictions leaves its pre-rescue match within 1.129x ± 0.10 but lowers its detection precision |
| **P4** | the clean rescue model replicates on world 2 in sign |
| **P5** | the clean LightGBM bundle serves bit-exactly |

## Decision rules (fixed here)

- **Operating point** ("precision at recall ~0.246"): Phase 17's procedure, unchanged. Per seed, on VALIDATION, the highest-precision
  threshold whose validation recall is ≥ 0.20; applied unchanged to TEST (Phase 17 published 0.823 [0.822, 0.824] at test recall 0.246).
- **GO** (Phase 17 B1 rule): precision > 0.70 at recall ≥ 0.20, disjoint from the proxy at 5 seeds. **0.63–0.70** marginal;
  **≤ 0.63** NO-GO.
- **Clean vs published:** Δ = clean − published, at the operating point and at 1 / 5 / 10 / 20 % coverage. A margin inside the 5-seed
  band is UNDETERMINED. Intervals: snapshot-block bootstrap (1,000 resamples of whole test snapshots) on the 5-seed ensemble score
  (each seed's score averaged), paired across arms on identical rows.
- **P2 resolves RIGHT** iff (i) published − clean < 0.05 at the operating point (5-seed means), (ii) clean > 0.619 and > 0.631 with the
  block interval of (clean − baseline) excluding 0, and (iii) clean meets GO.
- **World 2 (P4):** own rows, own base rate, own bands, never borrowed. World 2 has no stored simulation proxy, so the sign test is
  against its own as-of trailing-transfer-frequency baseline: **REPLICATES** if clean B1a − baseline is positive at the operating point
  with the block interval excluding 0; **DOES NOT** if negative and excluding 0; **UNDETERMINED** otherwise.
- **P3:** RIGHT iff (a) the simulation's arrival and fill inputs are LEAKY by the Stage 1 audit, and (b) the clean re-run's pre-rescue
  match is within 1.129 ± 0.10 and its rescue-detection precision is lower with the block interval excluding 0. If (b) cannot be
  measured here (the clean neural bundles are on the Mac), P3 is recorded **PENDING** for part (b) and judged on (a) only, labelled as such.
- **P5:** RIGHT iff the served predictions equal the stored clean-arm predictions **bit-exactly** on 200 test rows, for every seed.

## Gates and their constructed failing cases (every case is run by the code)

| gate | passes when | constructed failing case (must fire) |
|---|---|---|
| G1 leak audit of a feature list | no feature derives from a LEAKING column, and the future-poison test leaves the arm's own features at t0 unchanged | **(a) a deliberately leaked arm:** the NL feature list plus `lead_time_actual_days_mean` re-added must be FLAGGED (by name and by poison) |
| G2 rows identity | rebuilt world-1 val/test rows equal the stored `phase15_sim` rows (part-plant, week, label) element for element | one label flipped must make the gate fire |
| G3 published reproduction | re-fitting PUBLISHED B1a with the stored code, config and seeds reproduces the stored `b1a_lgbm_s{s}.npz` predictions (bit-exactly if possible) | a refit with the wrong seed must NOT reproduce |
| G4 permuted-label control | precision at the operating point under cross-snapshot permuted training labels collapses to the base rate (within 0.03) | the leaked (published) arm must beat the control by > 0.10 |
| G5 five seeds | every scorer refuses to score an arm without all of seeds 7, 17, 27, 37, 47 | an arm with one seed dropped must raise |
| G6 serving identity | `ml/serve/rescue.py` loads a bundle only if its identity (config hash, commit, feature list) matches | **(b) a mismatched bundle** (feature list or config hash altered) must raise |
| G7 serving AST scan | `ml/serve/rescue.py` reads no `inventory_position_weekly`, no `part_demand_weekly` and none of the nine LEAKING columns | a constructed offender source reading `inventory_position_weekly` and `load_ratio` must be flagged |
| G8 serving as-of | every source row used by the feature builder has `recorded_ts <= t0` | a constructed future row must raise |

## Stops

- BASE_SHA absent from this clone, or `reports/part2/phase-22.md` absent at it: STOP (checked: both present).
- World 1 or world 2 data absent and not recoverable as above: STOP and list what must be copied from the Mac.
- Clean arrival / fill predictions for the simulation absent: Stage 3 is **PENDING** with the exact file list; nothing is guessed.
