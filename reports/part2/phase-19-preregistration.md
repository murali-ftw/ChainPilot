# Phase 19 — pre-registration

**Committed before any Phase 19 measurement.** Nothing below is edited after a result is known. Each prediction is
marked right or wrong in `reports/part2/phase-19.md`; wrong ones are kept.

**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, val 2024, test 2025), seeds 7 / 17 / 27 / 37 / 47.
**Branch / base:** `phase19`, from `phase18` at `62bb607` (Phase 18 content `1212dd3`; not merged).
**Machine:** Apple M4 Pro (14 cores, 24 GB), macOS; torch 2.14.0 (MPS), LightGBM 4.7.0. The machine that trained
every stored v8 bundle.
**Wall-clock stop:** 10 h from the start at 20:26 IST 2026-10-03, i.e. **06:26 IST 2026-10-04**.
**Deviations** are numbered from 171.

Phase 18's numbers are reused as stored context only. Its FAIL verdicts stand; nothing here re-labels them.

## Feature families (all as-of; every source row asserted `recorded_ts ≤ t0`)

| family | definition |
|---|---|
| `fwd_load` | Phase 18's 15 supplier- and channel-specific forward-plan columns, `ml/data/fwd_load.py` **unchanged** (`part_demand_weekly` recorded time = `as_of_date` + 2 days, deviation 162) |
| `fwd_season` | **only** the snapshot-level network mean (NaN-aware, over all channels) of each of the 15 `fwd_load` columns: 15 values per snapshot, identical for every row of the snapshot, no supplier or channel identity. New module `ml/data/fwd_season.py` |
| `cadence` | per channel at t0, from `po_lines` with `recorded_ts ≤ t0` (and `grn_lines` with `recorded_ts ≤ t0` for openness): (1) days since the last line was raised (`t0 − max created_ts`); (2) median and (3) IQR, in days, of the gaps between consecutive `created_ts` among lines raised in the 52 weeks to t0 (missing if fewer than 2 gaps); (4) open line count and (5) age in days of the oldest open line, where open = raised in the 26 weeks to t0 with no final receipt recorded by t0 (Phase 18's openness rule). New module `ml/data/cadence.py` |

## Arms (LightGBM proxy, frozen GBM, the same rows as Phase 18)

| arm | definition |
|---|---|
| BASE | the stored LightGBM-flat arms; the proxy runner must again reproduce seed 7 **bit-exactly** before any new arm is fitted |
| BASE + fwd_load | **reused** from Phase 18 (stored predictions); its feature arrays are rebuilt and asserted equal to Phase 18's, and its stored scores are re-scored and asserted equal |
| BASE + fwd_season | new. Consistency check: its predictions must equal Phase 18's diagnostic `fwd_load_netmean` arm (same values, renamed columns) |
| BASE + fwd_load, snapshot-permuted | each snapshot's whole `fwd_load` block (all channels × 15 columns) is taken from a **different snapshot of the same split** (a derangement within train, within val, within test; `rng(30000 + seed)`); channel c receives channel c's row of the donor snapshot |
| BASE + fwd_season, snapshot-permuted | the same derangement applied to the 15-value season vectors |
| BASE + cadence | new |
| BASE + cadence, shuffled | the cadence rows permuted across channels within each snapshot (`rng(40000 + seed)`) |
| PRIVILEGED: BASE + true creation week | arrival and fill only (their rows are PO lines): the line's creation week minus t0, **no state**. Never a feature. Code and outputs under `reports/part2/phase19/PRIVILEGED__*` |
| PRIVILEGED: BASE + hindsight load | **reused** from Phase 18 (stored predictions) |

## Gate v2 (replaces the Phase 18 gate)

Bands are [min, mean, max] over the 5 seeds; "better / worse" means **disjoint** bands.

- **fwd_season PASSES** a use case if BASE + fwd_season beats BASE on ≥ 1 primary metric, is worse on none, AND the
  snapshot-permuted fwd_season arm is not better than BASE on any primary metric.
- **fwd_load's supplier-specific content PASSES** a use case if BASE + fwd_load beats **BASE + fwd_season** on ≥ 1
  primary metric, is worse than it on none, AND the snapshot-permuted fwd_load arm is not better than BASE on any
  primary metric.
- **cadence PASSES** if BASE + cadence beats BASE on ≥ 1 primary metric, is worse on none, AND the shuffled cadence arm
  is not better than BASE on any primary metric.
- **Capacity** has six points (precision at 1 / 5 / 10% coverage, recall at p = 0.70 / 0.80 / 0.85): a pass needs ≥ 3 of 6
  better and none worse.
- Primary metrics: arrival = lateness ROC-AUC (`docs/specs/lateness_metric.md`) and A3 median absolute error;
  fill = exact CRPS and P(fill = 1) ROC-AUC; capacity = the six UC3 points, operating points chosen on validation per seed.
- Scorer: Phase 18's `ml/eval/phase18_score.py` metric functions, unchanged.

## Stage 2 timing rule (from the brief)

Forecasting the order-raising time is **worth building** only if the PRIVILEGED creation-week arm adds **≥ 0.05**
lateness ROC-AUC over BASE (5-seed means) **AND** cadence recovers **≥ 30%** of that addition,
i.e. (cadence − BASE) / (creation-week − BASE) ≥ 0.30 on lateness ROC-AUC means. Otherwise the report records
"arrival headroom is future supplier state, not reachable from as-of data".

## Stage 3 neural binding (only if something passes Gate v2)

- **What is bound:** for each use case, the union of the columns of every family that PASSED Gate v2 for that use case
  (fwd_load's 15 columns if its supplier-specific content passed; fwd_season's 15 if it passed; cadence's 5 if it passed).
  A use case with no pass gets no neural run.
- **Input path:** the existing per-row input (`n_row_feats`: per-row features concatenated to the channel encoding before
  the head), through a NEW `HeadNet` subclass and a NEW batch builder bound into `phase5_heads` for that process only.
  No existing class is edited.
- **Scaling, fixed now:** each column is z-scored with the mean and sd of the **training rows** (NaN-aware), clipped to
  ±5, NaN → 0, and one missing-indicator column is added for every column that has any NaN on training rows.
- **Identity:** a new axis `row_family` (e.g. `fwd_load`), omitted at its default (absent). `artifact_identity.py` is
  protected, so names come from a new wrapper that appends `_rf{family}` only when the axis is set. Bundles are
  written under a separate root, `ml/artifacts/phase19/bundles/`. A test shows that every stored config is unchanged
  and that a set axis gives a distinct name.
- **Training:** incumbent architecture, depth, LR, max epochs and patience (arrival SHARE-lite h⁴ 2.5e-4; fill h⁰
  22-cell 1.25e-4; capacity mp h⁴ 2.5e-4), the incumbent's own `loop.train` / `loop.finish_bundle`, 5 seeds,
  **concurrency 1** (no other job runs during a neural cell).
- **Handshake first:** retrain stored `arrival_week v8_lite_h4_lr0.00025_s7` for 3 epochs. If every epoch's validation
  C-index is inside the stored 5-seed band at that epoch, proceed. Otherwise label every result "within-machine
  comparison" and retrain the incumbents.
- **Neural verdicts** against the stored incumbent's 5-seed band: **GAIN** = disjoint better on ≥ 1 primary metric and
  worse on none (capacity: ≥ 3 of 6); **WORSE** = disjoint worse on any primary metric and better on none; **TIE**
  otherwise; **MIXED** if disjoint better on one and worse on another.

## Stage 4 snapshot-block bootstrap

1,000 resamples of **whole test snapshots** with replacement (there are 9 test snapshots). Each resample recomputes
every metric on the rows of the drawn snapshots, keeping duplicates. Paired: every arm is scored on the same resamples.
"Survives" = the 95% interval of the difference still excludes 0 in the same direction. For the conformal intervals,
coverage is reported with its block-bootstrap 95% interval. Blend weights are refitted on validation only.

## Predictions

| # | Prediction |
|---|---|
| **P1** | fwd_season beats BASE on fill and capacity (disjoint), and its snapshot-permuted arm does not |
| **P2** | Supplier-specific fwd_load beats fwd_season on all three use cases |
| **P3** | Snapshot-permuted fwd_load is not disjointly better than BASE on any use case |
| **P4** | BASE + true creation week alone (no state) adds < 0.02 lateness AUC (timing is not the lever by itself) |
| **P5** | cadence adds < 0.01 lateness AUC |
| **P6** | Neural + fwd_load beats the neural incumbent (disjoint) on fill and on capacity recall at p = 0.70, and ties on arrival |
| **P7** | The snapshot-block bootstrap widens the Phase 18 intervals, but the arrival blend's lateness gain still excludes 0 |
| **P8** | A blend of neural + fwd_load with LightGBM + fwd_load beats the neural + fwd_load ensemble on arrival only |

P1 is scored on its own words: "beats BASE (disjoint) on fill and capacity" means at least one primary metric is
disjointly better in each; capacity's gate needs 3 of 6 points, but P1 does not. P6 and P8 are scored only on the arms
that run. A prediction about an arm that never ran (Gate v2 did not pass) is recorded **NOT TESTABLE**, not right or wrong.
