PROPOSED - NOT ACTIVE until the owner approves in writing

# Proposed exception to constraint 4: the production plan (`part_demand_weekly`) as an as-of serving input

**Status:** proposed in Phase 23AC Track T2. Not approved. Nothing in `ml/configs/shipped.json` references this
document, the dormant reader, or the dormant fill service; no shipped item changes.
**Measured on:** v8 seed 1001 (world `v8clean`), as quoted below. Companions: `reports/part2/phase-23ac-preregistration.md`
(T2, deviation 225), `reports/part2/phase22/stage3_fill.md` (section e), `reports/part2/phase-22.md`.
**Code (dormant):** `ml/serve/_plan_reader.py`, `ml/serve/fill_consolidated.py`; measurement `ml/eval/phase23ac_t2_serve.py`;
tests `ml/tests/test_phase23ac_t2_isolation.py`, `ml/tests/test_phase23ac_t2_serve.py`.

## What is asked

Constraint 4 forbids `part_demand_weekly` in model and serve code. Phase 22 Stage 3 (section e) declared the clean fill
predictors **not servable** for that reason only: their season family is built from the plan. This document asks the owner
to allow **one** reader of that table, under the controls below, so the neural season + cadence fill bundles can be served.

## Why the plan is a legitimate as-of input

- `part_demand_weekly` is the plan of record: recorded forward gross requirements per part x plant x target week, a new
  version every 8 weeks (`as_of_date`). At a snapshot t0 the planner has already published every version dated before t0.
  It is information a buyer has at t0, not an outcome.
- The published table has **no `recorded_ts`** column (the schema drops the generator's). The generator draws the recorded
  time as `as_of_date + 0..2 days` (`generator_v8.py` l. 1418). The reader therefore uses the conservative upper bound
  **`as_of_date + 2 days`** as the recorded time and admits a version only when that bound is `<= t0`
  (deviation 162, `ml/data/fwd_load.PDW_MAX_LAG`).
- Of each (part-plant, target week) the latest admissible version is used; target weeks are t0 + 1 .. t0 + 13.

## What could go wrong

1. **A later plan version leaking into an earlier t0.** A version published after t0 revises the same target weeks with
   knowledge t0 did not have. If admitted, the season family carries future information.
2. **The bound is wrong on real client data.** The 2-day lag is a property of this generator. A client's planning system
   may publish late, back-date `as_of_date`, or overwrite versions in place. Then `as_of_date + 2 d` is not an upper bound
   and the assertion passes on leaking rows.
3. **A second reader bypassing the assertion.** Any other module that opens the table (or calls a pre-existing builder
   without the reader) can feed unbounded rows to a model while the allow-listed reader stays correct.

## Controls

- **A single allow-listed reader:** `ml/serve/_plan_reader.py` is the only new module that names the table. Before any row
  reaches `fwd_load.snapshot_features` (pure computation), it asserts `as_of_date + 2 days <= t0` on **every row it passes
  on**, computed from `as_of_date` itself (not from the derived `recorded_ts`, so a tampered `recorded_ts` cannot pass).
  A violation raises `fwd_load.AsOfViolation` and serving stops.
- **Poison test:** every plan row with `as_of_date + 2 d > t0` is replaced by random values; the season rows built through the
  reader must not change (0 of N). Constructed offender: a reader without the bound must change N of N, so the test can fail.
- **AST allow-list test:** every new Phase 23AC module under `ml/` is scanned (docstrings excluded); a string naming the table
  anywhere but `ml/serve/_plan_reader.py` fails. A constructed second reader must be flagged. The pre-existing readers
  (`ml/data/fwd_load.py`, Phase 18 build time; `ml/data/phase20_world.py`; `ml/serve/features.py` through `fwd_load`) are
  listed as found, not changed.
- **On real client data** (risk 2) the bound must be re-established from the client's own publication log before the
  exception applies; until then this exception covers the synthetic worlds only.

## What is lost if declined

All figures below are **QUOTED**, not recomputed, v8 seed 1001, world `v8clean`, test 2025, 5 seeds:

| predictor | figure | source |
|---|---|---|
| neural season + cadence (the five dormant bundles) | CRPS **0.1375**, P(fill = 1) AUC **0.640**, UC2b precision@5% **0.464** | QUOTED, `reports/part2/phase-22.md` (headline table) and `reports/part2/phase22/stage3_fill.md` section b |
| LightGBM consolidated (blend w_neural = 0.02) | CRPS **0.1344**, UC2 precision@5% **0.944**, UC2b precision@5% **0.461** | QUOTED, `reports/part2/phase-22.md` and `reports/part2/phase22/stage3_fill.md` section c |
| clean Phase 19 recipe (comparison for UC2b) | UC2b precision@5% **0.511** (vs 0.461 for the consolidated blend) | QUOTED, `reports/part2/phase-22.md` and `reports/part2/phase22/stage3_fill.md` section c |

Declining leaves fill served by the clean incumbent without season + cadence (QUOTED, `stage3_fill.md` section b: CRPS 0.1397,
AUC 0.6007, UC2b 0.401), or not served.

**Even if approved, only the neural member is servable.** The LightGBM consolidated model was never persisted (Phase 22
saved predictions only). Persisting it would need a refit, which Phase 23AC forbids (deviation 225), so the dormant path
serves the five neural season + cadence bundles
`ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{7,17,27,37,47}_rfseason+cadence` only.

## What approval would change

Nothing automatically. The reader and the service stay dormant: no entry in `ml/configs/shipped.json` points to them. An
approval in writing would let a later phase add a shipped item that names `ml/serve/fill_consolidated.py`.
