# Phase 23AC T2 — the fill serving exception: PROPOSED and DORMANT

**Audience:** the owner, who decides whether `part_demand_weekly` may be read by serving code.
**Measured on:** v8 seed 1001 (`v8clean`), test 2025. CPU only.
**Instruments:**
- Code (`d401cb3`): `ml/serve/_plan_reader.py`, `ml/serve/fill_consolidated.py`, `ml/tests/test_phase23ac_t2_isolation.py`,
  `ml/tests/test_phase23ac_t2_serve.py`.
- `ml/eval/phase23ac_t2_serve.py`, gate fix `006ebc9` → `ml/artifacts/phase23ac/t2/serve.json`, `reports/part2/phase23ac/t2/serve.csv`.

**Decision record:** `docs/decisions/proposed_exception_part_demand_weekly.md`, headed "PROPOSED - NOT ACTIVE until the
owner approves in writing". **Nothing imports the dormant modules except the tests; `shipped.json` is untouched.**

## What was built

- **`_plan_reader.py`:**
  - the **only** new module that names `part_demand_weekly`;
  - asserts as_of_date + 2 days ≤ t0 (deviation 162's bound) on every plan row it passes on;
  - computes the bound from `as_of_date` itself, so a forged `recorded_ts` cannot pass it.
- **`fill_consolidated.py`** serves the five Phase 22 clean neural bundles (`…_rfseason+cadence`, seeds 7–47).
  - Season and cadence are built from source through the reader.
  - Cadence reuses the reader's single load, rather than `cadence.load_sources`, which would read the plan a second time
    without the bound.
- **Identity guard** (the bundles' checkpoint and train-log SHA-1s are pinned). It raises on:
  - a missing, incomplete or wrong-name bundle;
  - a wrong world or wrong family;
  - a tampered checkpoint or train log.
- **Not built:** the LightGBM consolidated model (CRPS 0.1344) was never persisted, and persisting it needs a refit that
  this phase forbids (deviation 225). **Only the neural member is servable.**

## Results

| check | result |
|---|---|
| served features vs the training-time `RowStore22` features, 200 rows | **identical** (max \|Δ\| 0.0, NaN pattern equal) |
| served 5-seed fill distribution vs stored predictions, 200 test rows (5 snapshots × 40) | **max \|Δ\| 2.38e-7**; per seed 2.98e-7 – 4.17e-7, 3,800–3,900 of 4,400 cells nonzero |
| **P3** (exactly bit-identical) | **WRONG.** The stored predictions were made on MPS; this phase serves on CPU (no MPS allowed). The difference is float32 rounding in the network, not the inputs |
| season poison: plan rows with as_of + 2 d > t0 corrupted | bounded reader changes **0 of 4,000** rows at every snapshot; the constructed offender (no bound) changes **4,000 of 4,000** where future plan versions exist (2025-01-06, 2025-06-23). 2025-12-08 has no future-recorded version: n/a for the offender (deviation 228) |
| reader allow-list (AST over every new module) | one reader (`_plan_reader.py`); a constructed second reader is flagged; docstrings pass |
| guard | missing bundle, tampered config, tampered checkpoint, wrong name, the season-only sibling bundle and an incomplete bundle each raise |
| `inventory_position_weekly` in new modules | 0 |

**Pre-existing `part_demand_weekly` readers, found by the scan and not changed (a finding):**
- `ml/data/fwd_load.py` (Phase 18; built the stored season arrays);
- `ml/data/phase20_world.py` (hashes the table);
- `ml/eval/phase12_a5_policy_facts.py`;
- `ml/opt/allocation.py`, `ml/opt/audit_inputs.py`, `ml/opt/schedule_lp.py`;
- `ml/sim/montecarlo.py`;
- `ml/serve/features.py` (Phase 20 serving) reads it through `fwd_load`.

If the exception is granted, these are the reads to bring under the same bound.

## The numbers the path would give (QUOTED, not re-measured)

| | value | source |
|---|---|---|
| neural season + cadence ensemble, exact CRPS | 0.1375 (AUC 0.640, UC2b 0.464) | `phase22/stage3_fill.md` b |
| LightGBM consolidated (not servable: never persisted) | CRPS 0.1344; UC2 precision @ 5% 0.944 | `phase-22.md` §1 |
| UC2b (materially short) @ 5% | consolidated 0.461 vs clean Phase 19 recipe 0.511 | `phase-22.md` §1 |

**Owner's decision:**
- **If granted:** the neural fill model with season + cadence can be served, after re-establishing the as_of + 2 d bound on
  the client's own plan data.
- **If declined:** the clean fill distribution falls back to the clean incumbent (CRPS 0.1397, AUC 0.601).
