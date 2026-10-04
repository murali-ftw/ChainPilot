# Phase 20 — pre-registration

**Committed before any Phase 20 measurement.** Nothing below is edited after a result is known. Each prediction is
marked right or wrong in `reports/part2/phase-20.md`; wrong ones are kept.

**Measured on:** v8 seed 1001 (primary) and a second world generated from v8 seed 1002 (Stage 2); fixed split
(train ≤ 2023, val 2024, test 2025); seeds 7 / 17 / 27 / 37 / 47.
**Branch / base:** `phase20`, from `phase19` at `e9c0862` (isolation audit `7e64261`). Not merged.
**Machine:** Apple M4 Pro (14 cores, 24 GB), macOS. **CPU only**: no neural training, no MPS. LightGBM 4.7.0.
**Wall-clock stop:** 10 h from the start at 00:41 IST 2026-10-04, i.e. **10:41 IST 2026-10-04**.
**Deviations** are numbered from 178.

## Predictions (from the brief, verbatim)

| # | Prediction |
|---|---|
| **P1** | Decision level: the Phase 19 arrival blend beats the incumbent neural ensemble on precision at 5% coverage (block interval excludes 0) |
| **P2** | Decision level: fill stays WATCHLIST under the Phase 15 classification even with the Phase 19 blend |
| **P3** | Replication: every (use case × family) that PASSED Gate v2 in v8-1001 shows a same-sign, disjoint gain over BASE on the second world |
| **P4** | Replication: the second world's gain for each passing arm is within a factor of 2 of v8-1001's gain on the same metric |
| **P5** | Load forecaster recovers < 25% of the hindsight gap on arrival and ≥ 15% on capacity |
| **P6** | The forecast feature passes its gate on capacity only |
| **P7** | The privileged reorder-trigger probe recovers ≥ 50% of the timing headroom (lateness AUC ≥ 0.81 from BASE 0.7053 toward 0.9104) |
| **P8** | No observed (non-privileged) stock source in v8 is a legitimate as-of reorder signal, so the client ask stands as the only route |

## Rules (from the brief, verbatim)

- Replicates = same sign AND disjoint 5-seed band on the same primary metric, in a world with its own BASE.
- Forecast feature PASSES a use case if BASE+fwd_load+fwd_pred beats BASE+fwd_load with disjoint bands on a primary
  metric, none disjointly worse, AND the snapshot-permuted fwd_pred arm is not disjointly better than BASE+fwd_load.
  Capacity: ≥ 3 of 6 points better, none worse.
- Recovery share = (arm − (BASE+fwd_load)) / (hindsight_load − (BASE+fwd_load)) on the same metric.
- Probe recovery share = (BASE+true_position − BASE) / (BASE+true_creation_week − BASE) on lateness AUC.

## Phase 15 rules, copied verbatim from `ml/eval/phase15.py` (committed `23d2477`)

```
OPERATING POINTS ARE CHOSEN ON VALIDATION, PER SEED, AND APPLIED TO THAT SEED'S TEST. The test curves in Stage A are
DESCRIPTIVE -- no test curve is ever used to pick a point (asserted: `pick_on_val` takes only validation arrays).

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

`pick_on_val` (verbatim behaviour): "VALIDATION ONLY: the lowest threshold (max recall) whose validation precision ≥ p with
≥ 50 alerts." Precision bars p = 0.70 / 0.80 / 0.85 / 0.90; MIN_ALERTS = 50.

## Rules Phase 15 does not state, fixed here before any Phase 20 number is read

**1. The ALERT / WATCHLIST / RETIRED classification.** Phase 15 wrote its recommendations by judgment (its §8). The
mechanical rule used here:

- **ALERT** = REACHABLE is YES or PARTIAL (Phase 15 rule) **and** test lift (precision ÷ base rate) at the operating
  point that makes it reachable is **≥ 1.5**. That point is the highest bar in {0.90, 0.85, 0.80, 0.70} whose
  validation-chosen point holds on test with ≥ 50 alerts on every seed, 5-seed mean lift.
- **WATCHLIST** = not ALERT, and Stage B (validation majority) is TUNABLE or WEAKLY TUNABLE.
- **RETIRED** = Stage B is NOT TUNABLE.

On Phase 15's published numbers this rule gives UC1 WATCHLIST (lift 1.19), UC2b WATCHLIST (CEILING, TUNABLE) and UC3
ALERT (PARTIAL, lift 2.0), matching Phase 15 §8. It gives **UC2 WATCHLIST** (PARTIAL, lift 1.10, TUNABLE), where
Phase 15 §8 wrote "retire as a yes/no; keep P(full) as a displayed probability". That disagreement is known now and
recorded, not tuned away. **P2 is scored on UC2b (fill < 0.95)**, the fill decision Phase 15 and Phase 17 called a
watchlist; UC2 is reported beside it.

**2. Deterministic arms** (seed-ensembles, blends): the rule is applied to the one predictor ("every seed" = that
predictor), and every interval is a snapshot-block bootstrap (1,000 resamples of whole test snapshots, paired across arms).

**3. Accuracy beside the majority baseline:** accuracy at the threshold that maximises F1 on validation, beside the
accuracy of always predicting the majority class (test).

**4. Stage 1 use cases and scores** (Phase 15's definitions, unchanged):
- UC1 arrival late vs the **contracted** lead reference, censoring resolved (Phase 15 `arrival_arrays`, `lab_c`).
  Score: P(late) read off the survival curve for distribution arms; prediction − reference for point arms (LightGBM,
  the blend), exactly as Phase 15 scored LightGBM.
- UC2 fill = 1 (score P(fill = 1)); UC2b fill < 0.95 (score = mass below 0.95); UC3 strain > 1 (score P(strain > 1)
  from P10 / P50 / P90).
- Neural ensembles average the distributions (arrival S and pT; fill P22; capacity quantiles). The Phase 19 blends use
  Phase 19's validation-fitted weights: arrival 0.52 × Phase 19 neural ensemble + 0.48 × LightGBM + fwd_load ensemble;
  fill 0.50 / 0.50.
- Per-snapshot spread of precision at 5% coverage: the top 5% of **each test snapshot's** rows, per snapshot.

**5. Stage 2 (second world).** The world is generated from the unmodified generator with seed 1002 into `data_worlds/`
(outside `db/`) and registered in-process by a new thin wrapper. P3 / P4 are scored on the six cells that passed
Gate v2 in v8-1001 (arrival: fwd_season, fwd_load, cadence; fill: fwd_season, cadence; capacity: fwd_load), on each
primary metric that was disjointly better there. P3 compares the family arm with BASE (for fwd_load: BASE + fwd_load
vs BASE); the Gate v2 reference comparison (fwd_load vs fwd_season) is reported beside it. P4: ratio of mean gains
(world 2 ÷ v8-1001) in [0.5, 2.0]. The LightGBM-only blend recipe: per use case, the LightGBM arm carrying the families
the Phase 19 neural arm carried (arrival fwd_load + season + cadence; fill season + cadence; capacity fwd_load), blended
with LightGBM + fwd_load, with the weight fitted on that world's validation.

**6. Stage 3 forecaster.** One channel-level LightGBM regressor per window (weeks 1–4 / 5–8 / 9–13), target log1p of
the realised ordered quantity of the channel's PO lines created in that window (an observed label). Features: fwd_load,
fwd_season, cadence, the channel's trailing ordered quantity over 4 / 13 / 52 weeks and line count over 52 weeks (as-of,
asserted), and the flat static channel features. **Supplier-level predictions = the sum of the supplier's channel
predictions** (not a second model). Purged blocked K-fold, K = 5 contiguous blocks of train snapshots; each fold's
training rows exclude every snapshot within 13 weeks of the held-out block; only snapshots whose 13-week window ends
by the train end are training targets. Val / test predictions come from a model fitted on all eligible train rows.
Forecaster seed = proxy seed. `fwd_pred` = 6 columns (channel × 3 windows, supplier × 3 windows, predicted quantities).
Accuracy on test: MAE (raw quantity) and R² (raw and log1p), against the naive fwd_load channel requirement.

**7. Stage 4 probe.** The simulator's channel-level inventory position (on hand + on order) and reorder point at the
end of each snapshot week are captured by exec'ing the generator's source **with a read-only hook** (no RNG draw added),
and the run is accepted only if every `_sim.npz` array equals the stored world's. Probe columns: (position − reorder
point) ÷ planner forecast (weeks above the trigger), position ÷ reorder point, the planner forecast, and weeks to the
channel's next review. A legitimate arm, if the audit finds an observed source: the part-plant's as-of on hand from the
`inventory_transactions` ledger (rows recorded ≤ t0) + open PO quantity − `part_plant.reorder_point_qty`, divided by the
trailing 13-week ledger issues; the gate as in Phase 19 (disjoint gain, snapshot-permuted control not better).

**8. Deadline.** A stage that cannot finish before 10:41 writes `reports/part2/phase-20-STOPPED.json` (completed,
not_run, reason) and is reported as partial.
