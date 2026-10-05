# Phase 24 Stage 1 — artifact and input check

**Audience:** whoever reads the Phase 24 numbers and needs to know what they rest on.

**Measured on:** this machine's `ml/artifacts/` (the Mac main checkout, read through the `phase24` worktree).

**Instrument:** `ml/eval/phase24_inputs.py` (`cda8074`) → `ml/artifacts/phase24/inputs.json`. Read-only: no CSV is read and no
model is run.

## a. The model bundles the simulation loads

**Leaky bundles.** The harness hard-codes these in `ml/sim/montecarlo.read_heads`, lines 176–177, and the harness is not
edited:
- `ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s7`;
- `ml/artifacts/bundles/fill_rate/v8_none_h0_lr0.000125_s{7,17,27,37,47}`.

**Clean bundles.** These are the paths named in `reports/part2/phase23b/stage3_simulation.md`, from Phase 22:
- `ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7`;
- `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{7,17,27,37,47}`.

| bundle | present, complete | config fields (task, world, seed, arch, depth, lr) | directory = identity name | pinned `checkpoint.pt` SHA-1 |
|---|---|---|---|---|
| leaky arrival s7 + fill s7…s47 (6) | **6 / 6** | match (`v8`) | **6 / 6** (`artifact_identity.bundle_name`) | in `inputs.json` |
| clean arrival s7 + fill s7…s47 (6) | **6 / 6** | match (`v8clean`) | **6 / 6** (`phase19_identity.bundle_name`) | in `inputs.json` |

The pinned SHA-1s are what the Stage 2 wrapper's identity gate (G1) checks before it loads anything.

## b. `ml/artifacts/phase23b`

**ml/artifacts/phase23b ABSENT: Phase 23B's clean rescue figures are QUOTED from reports/part2/phase-23b.md, not recomputed.**
- The B1a predictions and bundles (`rescue_configured.json`, `bundles/rescue_lgbm_v8clean_s{seed}/`) are on the Windows machine.
- `ml/artifacts/phase17/` is also absent here, so B1a published and B1b published are QUOTED from `reports/part2/phase-17.md`.

## c. The stored files Stage 2 must reproduce, and the clean cache

| file | status |
|---|---|
| `phase13_s1.json` (Q1 1.129×), `phase14_sim.json` (Q2 counts), `phase15.json` (Q4 curve), `phase15_sim_gate.json` | OK |
| `phase12_b2_validate.json`, `phase12_b2_seeds.json` (the 45 stored per-snapshot below-SS fractions) | OK |
| `phase15_sim/{val,test}_s{7,17,27,37,47}.npz` (the 10 stored row files) | OK, 10 / 10 |
| `order_policy.py @ 9e2d59d` (deviation 122) | git SHA-1 `922a0fcd…`; the stored cache copy is equal to it. Stage 2 writes its own copy under `phase24/` and does not touch the stored one |
| `cache/v8clean/meta.json` | `clean_of = v8`; `replaced_columns` = the nine leaking columns |

## What can run

| stage | runs? | why |
|---|---|---|
| 2: leaky reproduction | **yes** | all six leaky bundles and every stored file present |
| 2: clean arm | **yes** | all six clean bundles present, identities match |
| 3: rescue row | **QUOTED** | the rescue models' predictions are on the other machine |
| 3: simulation row | **RECOMPUTED** | from Stage 2 |
| 6: B1b clean | **yes** | the clean cache is present; the B1 rows are rebuilt and gated against the stored `phase15_sim` rows |
