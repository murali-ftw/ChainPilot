# Phase 24 — the shortage simulation restated on clean inputs, the pending rows refreshed, a Wednesday one-pager, and the clean neural rescue head

## Header

**Audience:** the project owner, presenting on Wednesday, and whoever decides which rescue model ships.

**Measured on:** v8 seed 1001 (`v8` and `v8clean`), fixed split (train ≤ 2023, validation 2024, **test 2025**), RAW. Model
seeds 7, 17, 27, 37 and 47.

**Machine:** Apple M4 Pro.
- Stages 1–5: inference only (the simulation on MPS).
- Stage 6: neural training on MPS, concurrency 1.

**Branch:** temporary `phase24` worktree from base **`e5ecb3f`**.
- Stages 1–5 were merged into `HADES-v4-ml-pipeline` as `b64069d` **before any training**.
- Stages 6–7 are merged at the end.

**Wall-clock:** 00:00:51 – 03:10 IST on 2026-10-06, inside the 8 h stop (08:00:51). No `STOPPED.json`.

**Companions:**
- Pre-registration: `reports/part2/phase-24-preregistration.md` (`f144053`).
- `reports/part2/phase24/`:
  - `stage1_inputs.md`, `stage2_simulation.md`, `stage6_b1b_clean.md`;
  - `metrics_pack_v2.md`, `progress_v2.md`, `charts/rescue_published_vs_clean.png`;
  - `wednesday_summary.md`.
- `results/observation3.1.md`.

**Code:**
- `cda8074`: inputs and the simulation wrapper;
- `3a2a40f`: metrics v2;
- `cb6590e`: isolation;
- `632eeb8`: the B1b trainer.

**Number marks:** every number is RECOMPUTED (this phase, with its artifact) or QUOTED (report and commit). LEAKY figures
are labelled "LEAKY (superseded, Phase 22)".

**Status:** complete. Predictions: 4 right, 1 wrong.

## 1. Decisions (clean vs published)

| question | published (LEAKY (superseded, Phase 22)) | **clean** | decision | mark / source |
|---|---|---|---|---|
| simulation: pre-rescue match ratio | 1.129× [1.121, 1.139] | **1.137×** [1.134, 1.139]; Δ +0.007, block [−0.019, +0.032] | **the mechanism holds** (undetermined change) | RECOMPUTED, `phase24/stage2_sim.json` |
| simulation: rescue-detection precision (Phase 17 point) | 0.619 [0.619, 0.620] at recall 0.194 | **0.612** [0.612, 0.613] at recall 0.197; Δ −0.007, block [−0.008, −0.004] | slightly lower; **WATCHLIST** (NO, CEILING), unchanged | RECOMPUTED |
| simulation: share of rescued weeks missed | 65.9% | **66.2%** | still two-thirds | RECOMPUTED |
| simulation: below-SS detector ceiling | 0.284 | **0.282** | unchanged; WATCHLIST | RECOMPUTED |
| predict-the-rescue, LightGBM B1a | 0.823 | **0.733** [0.7325, 0.7336] at recall 0.254; ensemble block [0.697, 0.773] | still GO; ALERT (PARTIAL @ 0.80), marginal | QUOTED, `phase-23b.md` §1 (`535361d`) |
| **predict-the-rescue, neural B1b** | 0.859 (2 of 5 seeds, capped) | **0.791** [0.747, 0.817] at recall 0.274; ensemble **0.805** [0.791, 0.824]; every snapshot ≥ 0.784 | **GAIN over clean B1a** (bands disjoint); ALERT (PARTIAL @ 0.85, lift 1.80); floors (4 of 5 capped); **not servable yet** | RECOMPUTED, `phase24/b1b_score.json` |
| B1b vs the clean simulation proxy / own history | — | +0.192 [+0.159, +0.236] / +0.174 [+0.131, +0.223] (paired block) | better than both | RECOMPUTED |
| PENDING rows in `observation3.md` and the 23AC metrics pack | PENDING PHASE 23B | filled in `observation3.1.md`, `metrics_pack_v2.md` and `progress_v2.md` (the originals are not edited) | done | — |

## 2. Prediction scorecard

| # | prediction (the owner's) | verdict | evidence |
|---|---|---|---|
| P1 | the pre-rescue match ratio stays within 1.129 ± 0.10 on clean inputs | **RIGHT** | 1.137 (five-seed mean) |
| P2 | rescue-detection precision drops, stays below clean B1a (0.733) and below the 0.28 ceiling's reading | **RIGHT** | 0.612 vs 0.619 (block Δ [−0.008, −0.004]); 0.612 < 0.733; ceiling 0.282 ≤ 0.284 |
| P3 | it still misses more than half of the rescued weeks | **RIGHT** | 65.8–66.6% on every fill seed |
| P4 | B1b on clean inputs, 5 seeds, TIES clean B1a | **WRONG** | GAIN: B1b band [0.747, 0.817] wholly above B1a's [0.7325, 0.7336]; higher at every coverage |
| P5 | B1b hits its training cap on at least one seed | **RIGHT** | 4 of 5 seeds at the 120-epoch cap |

**4 right, 1 wrong.** The wrong one is kept.

## 3. Deviations, continuing from 264

| # | rule | what was done instead, and why |
|---|---|---|
| **264** | Stage 6: "same caps as Phase 17 B1b" | **A 40-minute-per-seed wall cap was added beside the 120-epoch cap**, pre-registered, so that five seeds fit the 8 h stop. No seed reached it (max 29.1 min). Four reached the epoch cap: floors |
| **265** | charts with matplotlib | **Rendered by the scratch virtual environment with matplotlib** (Phase 23AC deviation 224's arrangement). `ml/eval/phase24_chart.py` reads only the CSV the measurement script writes |
| **266** | metrics pack v2 as `.md` and `.csv` | **`metrics_pack_v2.csv` and `charts/data/rescue_published_vs_clean.csv` are written but not committed**: `*.csv` is gitignored repo-wide (Phase 23AC deviation 229). The `.md` carries every number |
| **267** | "import the Phase 13/15 harness UNCHANGED" | **`phase15_sim.main` writes into the stored `ml/artifacts/phase15_sim/`, so it is not called.** Its driver loop is reproduced in the wrapper (UC5 rows plus a snapshot index), writing under `phase24/sim/`. `phase14_sim.b2_order_policy` would rewrite the stored `cache/order_policy_9e2d59d.py`, so the identical git source is written under `phase24/`. **G2 proves equivalence:** 10 / 10 stored row files reproduce bit-exactly, and 45 / 45 snapshots exactly |
| **268** | Q3 "reproduce to the published precision" | **Phase 17's `prec_at_min_recall` is nested inside `phase17_b1.score`, so it is reproduced line for line.** `phase17/b1_score.json` is absent here, so 0.619 [0.619, 0.620] is QUOTED from `phase-17.md`; it reproduces at 3 dp |
| **269** | constraint 3: "part_demand_weekly … not in this phase" | **The unchanged simulation harness reads `part_demand_weekly`** (`montecarlo.forward_requirement`: the plan as of t0, `assert_asof` on `as_of_date`), as every simulation run since Phase 12 has. Stage 2 requires the harness unchanged. **No new Phase 24 module names the table** (AST scan) |
| **270** | P4: B1b vs B1a | **B1a clean's predictions are absent here** (`ml/artifacts/phase23b`), so the verdict uses the five-seed bands (pre-registered caveat), with no paired B1b − B1a block interval |
| **271** | new git worktree | **The worktree reads the main checkout's `venv`, `ml/artifacts` and world data through symlinks**, hidden from `git status` by a scratch excludes file passed through `GIT_CONFIG_COUNT`; no repository config changed. Outputs land under the main checkout's `ml/artifacts/phase24/` (gitignored) |
| **272** | `observation3.1.md`: "the two PENDING rows replaced" | **Three sentences that the restated rescue figure contradicts were changed with them**: §0 "No use case is an alert any more", §10 "no alert", §10 "two use cases pending". Each is listed in the v4.1 changelog and Appendix B |
| **273** | Stages 1–4 committed before training | **`wednesday_summary.md` gets a clearly marked "Update after Phase 24 Stage 6" paragraph** after Stage 6. Its table and the Stage 1–5 numbers are unchanged |

## 4. Every `inventory_position_weekly` / `part_demand_weekly` read

| reader | table | purpose |
|---|---|---|
| `ml/sim/montecarlo.opening_position`, `order_policy@9e2d59d` (unchanged harness, run by `phase24_sim_clean.py`) | `inventory_position_weekly`, as of t0 | the simulation's opening stock and pipeline: the sanctioned simulation path since Phase 9 |
| `ml/sim/montecarlo.forward_requirement` (unchanged harness) | `part_demand_weekly`, as of t0 | the simulation's consumption plan (deviation 269) |
| `phase14_sim.store_labels`, `phase13_s1.store_2025` (unchanged, called by the scorer) | `inventory_position_weekly`, 2024–2025 weeks | **evaluation references only** (observed / pre-rescue below SS) |
| `ml/train/phase24_b1b_clean.py build_rows` | `inventory_position_weekly` (`part_id`, `plant_id`, `week_start` only) | **row filter only**: which part-plant-weeks exist, Phase 17's convention. The label comes from `inventory_transactions` |

**Model and serve code read neither table.** The B1b features are the clean channel panel (`v8clean`) and the masters.
Phase 24 adds nothing under `ml/serve` or `ml/models`.

## 5. What each result changes for the product

- **The simulation's picture is restated, not overturned.** Quote it with clean figures:
  - "within 14% of the pre-rescue world" (1.137×);
  - "misses two-thirds of rescued weeks";
  - "a weak week-by-week detector (0.61 at a 0.45 base)".
  The LEAKY-UNCONFIRMED labels can come off.
- **Predict-the-rescue is the shortage product,** and the neural head is now the better of the two on clean inputs:
  - 0.79–0.80 vs 0.73, with every 2025 snapshot above 0.78;
  - that removes clean B1a's "marginal" caveat;
  - but B1b has **no serving path**: it needs a serving module (identity guard, clean-cache guard, as-of asserts, like
    `ml/serve/rescue.py`) and the owner's decision.
  - Until then, **the servable answer is clean B1a** (Phase 23B's bundle, on the other machine).
  - `shipped.json` is not edited.
- **B1b's figures are floors.** Four seeds were still improving at 120 epochs. A longer cap is a separate, pre-registered
  decision.
- **For Wednesday:**
  - `reports/part2/phase24/wednesday_summary.md` (one page, plain language);
  - `results/observation3.1.md` (the full write-up);
  - `metrics_pack_v2.md` (every table).

## 6. Plain-language answers

**Is the shortage simulation's picture still true on clean inputs? Yes.**
- It was re-run with the same random draws, changing only the two models it reads.
- The original models first reproduced every published number exactly.
- With the clean models:
  - it still matches the stock picture before planners' rescues to within 14% (1.137× vs 1.129×; the difference is noise);
  - it still misses two-thirds of the rescued weeks;
  - its flags are a hair less precise (0.612 vs 0.619).
- The simulation is driven by stock, plan and policy, and barely leaned on the leak.

**Is predict-the-rescue still the better rescue answer, and does the neural head add anything? Yes, and yes.**
- Clean, the simulation's flags are right 6 times in 10 (against 4.5 by chance).
- The LightGBM is right 7.3 times in 10.
- The neural head, retrained clean on all five seeds, is right 8 times in 10, and above 7.8 in every test period.
- The neural head is better than the LightGBM on every measure compared here, by the pre-registered rule. It is a floor
  (training was capped), and it is not servable yet.

## 7. Isolation audit

`ml/tests/test_phase24_isolation.py` → `ml/artifacts/phase24/isolation_audit_stage5.json` (before training) and
`isolation_audit_stage7.json` (after). **Both: 8 / 8 PASS.**

| check | stage 5 | stage 7 | its failing case |
|---|---|---|---|
| protected paths vs `e5ecb3f` (every file existing under ml/, results/, db/, docs/specs/, reports/ + `shipped.json`) | 336 checked; **0 modified**; 13 added | 336; **0 modified**; 15 added (16 with this report) | the refusal fires on `e7163e4^..e7163e4` (it modified `results/observation1.md`); Phase 23AC's additions are seen |
| db/ (Phase 23AC's check, imported) | clean; seeds 1001–1005; world-2 hashes equal | same | — |
| stored artifacts (Phase 23AC's 188-file manifest) | **0 changed** | **0 changed** | — |
| stored identities (Phase 23AC's check) | **371 / 371**; Phase 19 bundles 15; Phase 22 bundles 25 | same; **Phase 24 bundles 5 / 5** at `artifact_identity.bundle_name` | — |
| AST scan of Phase 24 modules | no ml/serve or ml/models read; the scanner's own lists only | plus `phase24_b1b_clean.py` (the row filter, §4) | offender strings (plain and f-string) flagged; a docstring not counted |
| leak guard (`scripts/run_leak_guard.sh`) | exit 0 | exit 0 | (its own known-answer check) |
| no training | **0 training calls** in Phase 24 modules | only the declared Stage 6 trainer | 4 constructed offenders flagged |

**Merges:**
- Phase 24 adds new files only.
- Stages 1–5 merged as `b64069d`; Stages 6–7 merged `--no-ff` as recorded in the merge commit.
- Pushed to `origin/HADES-v4-ml-pipeline` with a plain fast-forward push.
- `main` is not touched.
