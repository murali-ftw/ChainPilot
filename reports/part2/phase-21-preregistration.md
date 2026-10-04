# Phase 21 — pre-registration

**Committed before any Phase 21 measurement.** Nothing below is edited after a result is known. Each prediction is
marked right or wrong in `reports/part2/phase-21.md`; wrong ones are kept.

**Question:** the user's 5-D group idea. Split deliveries into groups keyed by (supplier, part, plant, order month,
lane cluster), take censoring-aware, as-of group statistics of lead time and fill, and predict lateness (arrival) and
the received fraction (fill) from them, standalone and as a hybrid with the stored models.
**Measured on:** v8 seed 1001 (primary). Replication on **v8w1002**, the second world Phase 20 generated from v8 seed 1002
(Stage 7). Fixed split: train ≤ 2023, val 2024, test 2025. Seeds 7 / 17 / 27 / 37 / 47.
**Branch / base:** `phase21`, cut from `HADES-v4-ml-pipeline` at `124be5b` (the merged pipeline). Worked in a separate git
worktree (`../HADES_v4_phase21`); the user's working tree is not touched. Not merged, not pushed.
**Machine:** Apple M4 Pro (14 cores, 24 GB), macOS. **CPU only**: no neural training, no MPS. LightGBM 4.7.0, Python 3.14.
**Wall-clock stop:** 8 h from the start at 23:37 IST 2026-10-04, i.e. **07:37 IST 2026-10-05**. A stage that cannot
finish writes `reports/part2/phase-21-STOPPED.json` (completed, not_run, reason) and is reported as partial.
**Deviations** are numbered from **197** (the highest in `reports/part2/integration-report.md` is 196).

## Predictions (from the brief, verbatim)

| # | Prediction |
|---|---|
| **P1** | Standalone group rule (expected date = shrunk KM median; lateness = expected minus promise; no ML) gets arrival lateness AUC < 0.70 (below the stored incumbent 0.709) on test. |
| **P2** | BASE + group features (LightGBM, snapshot rows) beats BASE on arrival lateness AUC by < 0.02 and the gain is NOT disjoint on the 5-seed band. |
| **P3** | The L5 (with order month) features add < 0.01 AUC over the L4 (no month) features on arrival at snapshots (the month key at snapshots is only a seasonal proxy). |
| **P4** | AT-PLACEMENT arm: the true order month adds a disjoint gain over the L4 features on lateness AUC (month is legitimate there and carries the seasonal / queue-at-creation signal). |
| **P5** | Gains concentrate in thin channels (n below the validation-chosen threshold at L4); normal channels tie. |
| **P6** | Fill: group fill features add < 0.01 P(fill=1) AUC and < 0.001 exact CRPS over BASE; the acknowledged-vs-ordered gap, if available, is the only fill feature that passes. |
| **P7** | The hybrid (LightGBM + group features blended with the stored neural ensembles, weight fitted on validation) improves arrival median error (A3) but still ranks the top 5% worse than the incumbent neural ensemble P(late), i.e. no change of Phase 15 class. |
| **P8** | Replication on the second world: any passing arm replicates in sign; effect sizes differ by more than a factor of 2. |

How each is scored, fixed now:
- **P1** right iff the standalone rule's test lateness AUC (one deterministic predictor) is < 0.70.
- **P2** right iff (mean BASE+L5 − mean BASE) < 0.02 **and** the bands are not disjoint. The arm scored is BASE + L5
  (the user's 5-D idea); BASE + L4 is reported beside it.
- **P3** right iff mean(BASE+L5) − mean(BASE+L4) < 0.01 on snapshot lateness AUC.
- **P4** right iff the at-placement BASE+L5 (true month) band is disjointly above BASE+L4 on at-placement lateness AUC.
- **P5** right iff, for the best passing arrival arm (or BASE+L5 if none passes), the 5-seed band of the gain over BASE
  is disjointly positive in the thin slice and **not** disjoint (tie) in the normal slice. Per-slice lateness AUC.
- **P6** right iff (i) every group-fill arm (L4, L5) adds < 0.01 P(fill=1) AUC and improves exact CRPS by < 0.001 (mean
  vs mean), and (ii) BASE + ack-gap passes its gate while BASE + L4 and BASE + L5 do not. If ack data is not as-of usable,
  clause (ii) is NOT TESTABLE and P6 is scored on clause (i) with that stated.
- **P7** right iff the Stage 6a hybrid's A3 block interval vs the incumbent neural ensemble excludes 0 in its favour
  **and** its UC1 precision at 5% coverage is lower than the incumbent ensemble's (point), **and** no use case changes class.
- **P8** right iff every arm that PASSES in v8-1001 has the same-sign mean gain on world 2 **and** at least one passing
  arm's gain ratio (world 2 ÷ v8-1001) is outside [0.5, 2.0]. If nothing passes in v8-1001, P8 is NOT TESTABLE.

## Rules (from the brief, verbatim)

- A feature family PASSES a use case if BASE+family beats BASE with disjoint 5-seed bands on a primary metric, none
  disjointly worse, AND its permuted control (below) is not disjointly better than BASE.
- Controls: (a) order-month key permuted across snapshots/rows within the same split; (b) group statistics assigned from
  a different random channel at the same level (cross-channel shuffle); (c) L5 vs L4 comparison to isolate the month
  dimension; (d) the no-ML group rule as the floor.
- Primary metrics: arrival = lateness ROC-AUC (docs/specs/lateness_metric.md) and A3 median abs error; fill = exact CRPS,
  P(fill=1) ROC-AUC and the Phase 15 UC2b materially-short precision at 5% coverage.
- Decision level (Phase 15 scorer): precision at 1/5/10/20% coverage, recall at the Phase 15 precision bars, lift,
  accuracy beside the majority baseline, per-snapshot spread of precision at 5% coverage (one good quarter cannot carry
  a verdict).
- Thin / cold / normal slice thresholds are chosen on validation and written to the report before test is scored.

Which control gates which family (fixed now): **L4**: control (b), BASE + L4 cross-channel shuffle. **L5**: both (a)
BASE + L5 permuted month and (b) BASE + L5 cross-channel shuffle; either being disjointly better than BASE on any primary
metric fails the gate. **ack-gap**: control (b) applied to the ack columns (cross-channel shuffle). Fill's UC2b precision
at 5% is a per-seed metric read off Phase 15's coverage curve (the seed's test scores), banded over the five seeds like
the other two. A metric unreachable on some seed is UNREACHABLE, never a band (Phase 19 deviation 171).

## Phase 15 rules

Copied verbatim from `ml/eval/phase15.py` (the rule exists only as this docstring; Phase 20 deviation 178):

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

The ALERT / WATCHLIST / RETIRED mapping is Phase 20's, as pre-registered there (`phase-20-preregistration.md` rule 1)
and implemented in `ml/eval/phase20_decisions.py::reachable_and_class`, imported unchanged:
**ALERT** = REACHABLE YES or PARTIAL **and** test lift ≥ 1.5 at the highest bar in {0.90, 0.85, 0.80, 0.70} whose
validation-chosen point holds on test with ≥ 50 alerts on every seed; **WATCHLIST** = not ALERT and Stage B (validation
majority) TUNABLE or WEAKLY TUNABLE; **RETIRED** = Stage B NOT TUNABLE. Deterministic arms (seed ensembles, blends,
the no-ML rule) are one predictor ("every seed" = that predictor); their intervals are snapshot-block bootstraps (1,000
resamples of whole test snapshots, paired across arms). Old classes (Phase 20): UC1 WATCHLIST, UC2 WATCHLIST, UC2b
WATCHLIST.

## Definitions, fixed before any number is read

### D1. Snapshot rows (Stages 3, 5, 6, 7) — the same rows as Phase 19 / 20

`training_labels.csv`, task `arrival_week` (arrival) or `fill_rate` (fill), snapshot dates in the fit window
2019-01-01 … 2025-12-31, merged to `po_lines` on `entity_id = po_line_id` (`phase7_fit.labels`, unchanged); split by
snapshot date (`folds.fixed_split`); row order `phase7_fit.ordered`. One row = (snapshot t0, PO line). By the generator's
§14 (`generator_v8.py`, labels block: `fut = (pt > t0) & (pt <= t1)`, t1 = t0 + 13 weeks), the line is one **raised in
the 13 weeks after t0**, up to 4,000 sampled per snapshot. So **the row's own line does not yet exist at t0**.
- **Arrival label:** `label_value` = weeks from t0 to the line's first receipt (`pa − t0`), right-censored
  (`label_censored`) if no receipt by t1. It contains the wait until the line is raised plus the lead once raised.
- **Fill label:** `label_value` = delivered ÷ ordered for the line (`pd_ / pq`), the 22-cell target `phase7_fit.fill_cell`.
- **Row identity** is asserted against the stored `phase7_preds` BASE files (Y, EV, entity, order) for every arm.

### D2. Order-month key

- Snapshot rows: the line is never raised at t0 (D1), so its creation month is **unknown as-of for every row**. The
  key is the **calendar month of t0**, a seasonal proxy and **not** the true order month. Stage 1 verifies the share
  of snapshot rows whose line was created ≤ t0 (expected 0%); if any are, their key is their creation month.
- At-placement rows (Stage 4): the key is the line's **true creation month**, legitimately known when it is raised.
- "Month" is the **month of the year (1–12)**: a group pools the same calendar month across years.

### D3. Group levels (finest to coarsest)

- **L5** = (supplier, part, plant, order month, lane cluster)
- **L4** = (supplier, part, plant, lane cluster) — the earlier 4-D idea
- **L3** = (supplier, order month, lane cluster)
- **L2** = (supplier)
- **L1** = global

**Lane cluster** = (transport mode, transit class), where transit class is the tercile of the lane's
`standard_transit_days` (cut points from the lane table, which is static). Candidate lane / channel columns, verified
in Stage 1: `sourcing_channels.transport_mode`, `transport_distance_km`; `logistics_lanes.transport_mode`,
`standard_transit_days`, `carrier_id`, `via_checkpoint`; `supplier_sites.country` / `state`; `purchase_orders.incoterm`
and `po_type` (PO header, as-of when the PO is recorded). A column enters the cluster only if Stage 1 finds it varies and
is as-of; otherwise it is listed and dropped. Phase 18 found one lane per channel and `via_checkpoint` empty, so the
lane cluster is constant within a channel and L4 = the channel; the lane enters only through L3, as a backoff for thin
groups.

The levels are not all nested (L4 ⊄ L3). **Backoff chains:** L5 → L4 → L2 → L1 and L3 → L2 → L1 (no-month chain: L4 →
L2 → L1).

### D4. As-of source rows (every statistic)

At an as-of instant τ (a snapshot date t0 at 00:00, or the Monday 00:00 on or before a line's creation for Stage 4):
- a PO line enters a group's history only if `po_lines.recorded_ts ≤ τ`;
- a receipt counts only if its `grn_lines.recorded_ts ≤ τ`; acknowledgements only if `supplier_acknowledgements.recorded_ts ≤ τ`;
- the zero-fill signals (supplier rejection ack, qty → 0 revision) only if `recorded_ts ≤ τ` (Phase 13 `fill_history`'s
  closed-line rule);
- `part_demand_weekly`, if read, uses `as_of_date + 2 days` (deviation 162). It is not planned to be read.
Every source row that feeds a statistic carries the latest `recorded_ts` it depended on, and the builder **asserts** that
it is ≤ τ, and that the history is non-empty. A failing assertion is a **STOP**.

### D5. Expected delivery and the survivorship fix

- **Lead** of a historical line = first receipt `event_ts` − `created_ts` (days), using the earliest receipt recorded
  ≤ τ. A line with no receipt recorded ≤ τ is **censored** at τ − `created_ts`.
- **Lateness vs contract** = lead − `contracted_lead_time_days` (the line's `original_promise_date` is created +
  contracted lead in v8; Stage 1 verifies), with the same censoring.
- **Group medians and quantiles are Kaplan–Meier estimates** over every line in the group (receipted or open-and-censored
  as of τ). Never the median of completed deliveries only. Where KM never falls to the quantile (heavy censoring), the
  value is the group's largest observed time (a lower bound) and the row is counted.
- **Expected delivery rule:** where `original_promise_date` is available (at placement: always), it is the promise.
  Where it is not (every snapshot row, D1), expected = order date + shrunk group KM median lead.

### D6. Shrinkage

Empirical-Bayes backoff down each chain: `shrunk(L) = w·raw(L) + (1 − w)·shrunk(parent(L))`, `w = n / (n + k)`, with n the
group's **number of receipted lines as-of** (KM events). Quantiles are shrunk as values. Fill histograms are shrunk as
`(counts + k·parent) / (n + k)`. **k is one constant per task, chosen on VALIDATION only** from {1, 3, 10, 30, 100, 300}:
- arrival snapshot arms: max validation lateness AUC of the standalone group rule (D9);
- at-placement arms: max validation at-placement lateness AUC of the at-placement standalone rule;
- fill arms: min validation exact CRPS of the standalone group distribution.
Every statistic carries n at each level and the **resolved level** (the finest level of its chain with n ≥ k, else L1).

### D7. Features per row (as-of the row's τ)

Arrival / at-placement group block, per level of the arm: shrunk KM median, P10, P90 and IQR of lead (days) and of
lateness vs contract; shrunk KM P(late vs contract) = S_excess(0); n (receipted lines) and n_all (all lines) at each
level; resolved level. The **L4 arm** carries L4, L2, L1. The **L5 arm** carries the L4 arm's columns **plus** L5 and
L3 (so L5 − L4 isolates the month dimension, control (c)).
Fill group block: the shrunk 22-bin histogram at the arm's finest level; P(fill = 1), mean shortfall (1 − fill) and n at
each level; resolved level. Fill history = lines **closed** as-of τ (Phase 13's rule: final receipt recorded ≤ τ, or a
zero signal with no receipt), fill = min(received, ordered) ÷ ordered. Fill has no KM: open lines are not in the fill
history, and that bias is stated, not corrected.
**ack-gap block** (fill): per group (L4 and L2), over lines with an acknowledgement recorded ≤ τ: mean of
1 − ack_qty ÷ qty_ordered (clipped to [0, 1]), the share not acknowledged `full`, and the same two over the group's
lines still **open** as-of τ, with their counts. The row's own line has no acknowledgement at t0 (D1), so at snapshots
the gap can only be a group statistic. Stage 1 verifies the acknowledgement rows exist and are as-of.

### D8. Arms

LightGBM arms are Phase 7's guide-B5 fit, unchanged (`phase7_fit.lgbm_fit`: frozen GBM config, same labels, rows,
objective, early stopping on validation; arrival L2 on observed rows, fill 22-class), columns appended. BASE for v8 is
refitted for seed 7 and must reproduce the stored `phase7_preds` LightGBM-flat predictions **bit-exactly** before any arm
is trusted; the 5-seed BASE band is the stored files.
- **Stage 3 (arrival, snapshot):** BASE; BASE+L4; BASE+L4 cross-channel shuffle; BASE+L5; BASE+L5 permuted month;
  BASE+L5 cross-channel shuffle; standalone group rule (no ML); group-only LightGBM (L5 block alone, no BASE columns);
  BASE + Phase 19 fwd_season + cadence, and the same + L5.
- **Stage 4 (at placement):** BASE (flat as-of channel features at τ + log1p qty_ordered); BASE+L4; BASE+L5 (true month);
  BASE+L5 permuted month; standalone group rule.
- **Stage 5 (fill, snapshot):** BASE; BASE+L4; BASE+L4 cross-channel shuffle; BASE+L5; BASE+L5 permuted month; BASE+L5
  cross-channel shuffle; BASE+ack-gap; BASE+ack-gap cross-channel shuffle; BASE+L5+ack-gap; standalone group
  distribution (no ML); BASE + season + cadence (Phase 20's stored `lgbm_rf` for fill) and the same + L5.
- **Controls' randomness:** permuted month: within each split, the month keys of the rows are permuted (rng 70000 +
  seed), and the row reads its own channel's statistics at the donor month, as-of its own τ. Cross-channel shuffle: within
  each snapshot (Stage 4: each creation week), the group block of each row is taken from a uniformly drawn other row's
  channel (rng 80000 + seed).

### D9. Standalone group rules (no ML)

- **Snapshot arrival:** predicted arrival week = a + shrunk KM median lead (weeks) at the L5 chain's resolved level,
  a = the median of (Y − KM median) over **validation** observed rows (the wait until the order is raised; b = 1 fixed).
  Lateness is scored by the spec (`pl = prediction − R`). The spec's reference replaces "the promise", which no
  snapshot row has (D1). One deterministic predictor; snapshot-block intervals.
- **At placement:** predicted lead = shrunk KM median lead at the resolved level (true month), lateness score =
  expected − promise (the contract). A3 after the same validation offset a.
- **Fill:** the shrunk 22-bin histogram at the L5 chain's resolved level is the predictive distribution, scored by exact
  CRPS and P(fill = 1) AUC.

### D10. Stage 4 rows, labels and metrics (a different decision point)

Unit = a PO line, scored on the day it is raised. Rows = the **distinct** PO lines in v8's `arrival_week` label rows in
the fit window, each once, split by **creation date** (train ≤ 2023-12-31, val 2024, test 2025). τ = the Monday 00:00 on
or before `created_ts`. BASE features = `World.channel_features` at τ (flat, no identity codes, exactly as a snapshot on
that Monday is featurised) + log1p(qty_ordered). Label: lead in weeks = (first receipt `event_ts` − `created_ts`) / 7,
**censored** if the line has no receipt in the world. Metrics (the Phase 15 / lateness definitions cannot be applied
unchanged, because the reference is defined from t0; the departure is):
- lateness AUC at placement: on observed rows, yl = lead > contracted lead (= receipt after `original_promise_date`),
  score = predicted lead − contracted lead (expected minus promise);
- A3: median |7·pred − 7·Y| days on observed rows;
- decision level (UC1-P): late vs contract with censoring resolved (Phase 15's `lab_c` with R = contracted lead in weeks:
  censored lines count as late if the contract is shorter than their observed time), score = pred − contract.
Its absolute numbers are not compared with the snapshot arms.

### D11. Slices (Stage 6d)

n4 = receipted lines at L4 (the channel) as-of the row's τ. **cold** = n4 = 0; **thin** = 0 < n4 < T; **normal** = n4 ≥ T,
with **T = the 20th percentile of n4 over validation rows with n4 > 0**, computed in Stage 2 and written to the
Stage 2 file before any test score. Month-of-order slices: test rows by calendar month of the key (D2).

### D12. Intervals and comparisons

5-seed bands = [min, mean, max]; better / worse only when DISJOINT, otherwise UNDETERMINED. Deterministic predictors
(standalone rules, seed ensembles, blends) use the snapshot-block bootstrap (1,000 resamples of whole test snapshots,
paired across arms; Stage 4: whole creation weeks). No row bootstrap. RAW predictions only (no recalibration) in every
comparison. Bands are never borrowed across worlds.

### D13. Hybrid (Stage 6)

Arrival: point blend of three expected-week predictors, w on the simplex in steps of 0.05, fitted on **validation**
lateness AUC (Phase 19's blend criterion), frozen for test: the incumbent neural ensemble (h4, mean S / pT → expected
week), the Phase 19 neural ensemble (`…_rffwdload+season+cadence`), and the best LightGBM + group arm's 5-seed mean
(best by validation lateness AUC). Fill: w·(Phase 19 neural season + cadence ensemble P22) + (1 − w)·(best LightGBM +
group fill arm ensemble P22), w on a 0.01 grid by validation exact CRPS. Compared with both parents and the Phase 19
blend, with snapshot-block intervals, and through the Phase 15 classification.

### D14. Replication (Stage 7)

World v8w1002 = `data_worlds/v8_seed1002/seed_1002` (Phase 20), not regenerated; its key-table SHA-1s are re-hashed and
must equal `ml/artifacts/phase20/world2_generation.json`. BASE = Phase 20's stored world-2 BASE predictions (five seeds),
with seed 7 refitted and asserted bit-exact. Own bands. Verdict per arm: REPLICATES (same sign and disjoint vs its own
BASE), DOES NOT (opposite sign disjoint, or v8 passed and world 2 is disjoint the wrong way), UNDETERMINED (bands overlap).
k is re-chosen on world 2's validation by the same rule.

### D15. Isolation

No module under `ml/train`, `ml/models`, `ml/data`, `ml/baselines` imports a `PRIVILEGED__` path or reads
`inventory_position_weekly` (AST scan with constructed-offender self-test). The future-poison test (every receipt,
acknowledgement and line recorded after τ corrupted: features unchanged) and the self-exclusion test (a row's own
outcome never in its own statistic) run before any feature is used and again in the isolation audit.
