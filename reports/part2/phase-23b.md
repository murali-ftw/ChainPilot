# Phase 23B — predict-the-rescue and shortage-simulation leak audit, clean re-score, servable bundle

**Audience:** whoever decides whether predict-the-rescue ships, and whoever quotes the shortage simulation.
**Measured on:** world 1 `v8` (seed 1001) and world 2 `v8w1002`, fixed split (train ≤ 2023, val 2024, test 2025). Thresholds on
validation only. TEST, RAW.
**Companions:** `phase-23b-preregistration.md` (`fb998c0`), `phase23b/stage1_audit.md`, `phase23b/stage2_rescue_clean.md`,
`phase23b/stage3_simulation.md`; `phase-17.md`, `phase-22.md`, `phase22/stage1_leak.md`, `phase-13.md`, `phase-15.md`.
**Branch:** `phase23b` in a new git worktree, cut from **BASE_SHA `545e282cb1f5f0a4b3c9168ba379de49f820a3ad`** (the pulled
`HADES-v4-ml-pipeline` tip, "Merge phase22"; deviation 240). Not merged, not pushed.
**Machine:** Windows 11, RTX 3050 Ti Laptop (4 GB), **CPU only**: no neural training, no GPU use (`HADES_DEVICE=cpu`).
Python 3.10.11, LightGBM 4.7.0, torch 2.14.0+cu126 (imported, never on CUDA). One CPU job at a time, except the three cache builds,
which ran beside the Stage 1 audit. Every artifact carries its commit SHA; every run refused a dirty `ml/`.
**Wall-clock:** started 21:19, finished 21:48 on 2026-10-05, inside the 6 h stop (03:19).

## 1. Decisions

| arm | clean vs published | verdict | the number | interval |
|---|---|---|---|---|
| **B1a predict-the-rescue, LightGBM (world 1)** | precision at the Phase 17 operating point **0.733** clean vs **0.823** published (recall 0.254 vs 0.246) | **still GO** (> 0.70 at recall ≥ 0.20, disjoint from the proxy); Phase 15 class **ALERT** (PARTIAL), unchanged | Δ **−0.091**; lift 1.62 (was 1.81); beats the 0.619 sim proxy by +0.115 and the 0.631 own-history baseline by +0.103 | 5-seed band [0.7325, 0.7336]; block Δ [−0.104, −0.075]; vs proxy [+0.089, +0.148]; vs own history [+0.081, +0.125]; clean ensemble block [**0.697**, 0.773] |
| B1a, NL (leaking columns removed) | 0.712 at recall **0.189** | does not meet GO (recall < 0.20); clean replacements worth +0.022 [+0.015, +0.031] | Δ vs published −0.113 | [−0.127, −0.098] |
| **B1a clean, world 2** | 0.596 at recall 0.191 vs own history 0.454 (own base 0.292) | **REPLICATES in sign**; **not GO** there (below 0.70 and 0.20 recall); class WATCHLIST | +0.142; lift 2.04 | [+0.129, +0.154] |
| B1b predict-the-rescue, neural | — | **not re-scored: needs the Mac** (no GPU here); it read all nine leaking columns, so its published 0.859 (2 seeds) is not quotable | — | — |
| **Shortage simulation** | inputs: arrival s7 and fill s7–s47 neural heads | **inputs LEAKY**; the clean re-run is **PENDING: needs the clean neural bundles from the Mac** | 1.129× / 0.28 / two-thirds missed stand **unrestated** | — |
| **Servable bundle** (clean B1a, 5 seeds) | `ml/serve/rescue.py` + `ml/artifacts/phase23b/bundles/rescue_lgbm_v8clean_s{seed}` | **SERVABLE, bit-exact** | 200 test rows × 5 seeds: features and probabilities equal, max \|Δ\| 0.0 | — |

## 2. Scorecard

| # | prediction (the owner's) | **right / wrong** | the number |
|---|---|---|---|
| **P1** | B1a read at least one of the nine leaking columns | **RIGHT** | all nine: 18 of its 37 features. Every one of the 18 moves under future poison (27–100% of part-plants), and none of the other 19 does |
| **P2** | clean, precision at recall ~0.246 falls by < 0.05, stays above 0.619 and 0.631 (block-disjoint), stays GO | **WRONG** | it falls by **0.091** (block [−0.104, −0.075]). The other two clauses hold: +0.115 / +0.103 over the baselines, block-disjoint, and still GO |
| **P3** | the simulation's inputs were leaky; clean, the pre-rescue match stays within 1.129 ± 0.10 but detection precision falls | **part (a) RIGHT; part (b) PENDING** | both inputs (arrival s7, fill s7–s47) read all nine columns; the clean re-run needs the Mac's clean bundles |
| **P4** | the clean rescue model replicates on world 2 in sign | **RIGHT** | +0.142 [+0.129, +0.154] over world 2's own-history baseline, 9 of 9 snapshots (the alert level does not transfer: 0.596, WATCHLIST) |
| **P5** | the clean LightGBM bundle serves bit-exactly | **RIGHT** | 5 of 5 seeds bit-exact on 200 test rows; four constructed mismatches raise |

Three right (P1, P4, P5), one wrong (P2), one half-pending (P3). The wrong one is kept.

## 3. What each result changes for the product

- **Predict-the-rescue stays the shortage product, at a lower number.** Quote **0.73 precision at a quarter of the weeks
  planners act on**, against a 45% base rate (lift 1.6), and retire Phase 17's 0.82: it was 0.09 of leak. It still beats the
  simulation and the part-plant's own history in every 2025 snapshot.
- **The margin over the 0.70 bar is now thin.** Three of nine snapshots are under 0.70, and the snapshot-block interval
  reaches 0.697. **Present it as an ALERT at the PARTIAL (0.80) bar with a watched precision**, not as a comfortable 0.82.
- **On a world with fewer transfers (world 2) the same model ranks better than history (lift 2.0) but does not clear an alert
  bar.** The alert threshold has to be set per site from its own base rate. It cannot be carried over.
- **It is servable now:** a loadable, identity-guarded, bit-exact bundle closes Phase 17's "no LightGBM model was saved" gap.
  `shipped.json` is not edited; adopting it is a decision for the owner.
- **The shortage simulation's figures are on leaky inputs.** Until the Mac re-run lands, the 1.129× pre-rescue match, the
  0.28 ceiling and the two-thirds of rescues missed must carry "pending a clean re-run". That includes Phase 17 A2's profile of
  the missed rescues, which was measured on the same simulation rows.
- **B1b (neural) cannot be quoted** until it is retrained clean on the Mac.

## 4. Plain language

**Is predict-the-rescue still a GO on clean inputs?** **Yes, but by less.**
- Remove the leak and the model still picks out the weeks planners move stock: 73 times in 100 at the chosen operating point,
  where the simulation manages 62 and the part-plant's own history 63. It beats both in every month of 2025.
- About nine points of the old 82 were the leak.
- It now sits close to the 70 bar, so it should be watched, not assumed.
- On a second world it still beats history, but there are fewer transfers there to find, so it is a watchlist rather than an
  alert.

**Is the simulation's picture of the project still true?** **Not yet known.**
- The simulation's two inputs (the arrival and fill predictions) were trained on the leaking columns.
- Phase 22 showed both lose skill when the leak is removed.
- The clean re-run needs the clean models on the Mac. Until it runs, the simulation's numbers are the leaky ones, labelled as
  such.

## 5. Phase 17 artifacts on this machine

All present, except what was never produced:
- `ml/artifacts/phase17/` (`b1_rows.npz/json`, `b1a_lgbm.json`, `b1a_lgbm_s{7,17,27,37,47}.npz`, `b1_score.json`, `a1–a3.json`,
  `b2/b3/b4_score.json`, `b4_equivalence.json`, `logs/`);
- the 22 Phase 17 bundles: `rescue_week` s7 and s17, 10 `_tgt*`, 5 `_headband5`, 5 `_lean`;
- `ml/artifacts/phase15_sim/` (all 10 files) and `phase14_sim.json`, `phase15.json`, `phase15_sim_gate.json`, `phase13_s1.json`.

**Missing:** `bundles/rescue_week/v8_mp_h1_lr0.000125_s{27,37,47}`, never trained (Phase 17 deviation 152); and all of
`ml/artifacts/phase22/` (Mac only).

## 6. Every `inventory_position_weekly` / `part_demand_weekly` read

| reader | table | purpose |
|---|---|---|
| `ml/eval/phase23b_rescue.py build_rows` (line 84) | `inventory_position_weekly` (`part_id`, `plant_id`, `week_start` only) | **evaluation row filter** for world 2's rows: which part-plant-weeks exist, Phase 17's convention. The label comes from `inventory_transactions` |
| stored `phase15_sim` rows (world 1 rows and proxy) | built by `phase14_sim.store_labels` (Phase 15) | read here as stored arrays |

**Model and serve code read neither table.** The B1a features are `part_plant_features` (panel and masters).
`ml/serve/rescue.py` passes an AST scan with an offender self-test. The other hits of those names in Phase 23B code are text:
- the audit's description of the simulation's inputs (`phase23b_audit.py` lines 79–80);
- the scanners' forbidden lists (`phase23b_isolation.py`, `test_phase23b_serve.py`);
- the constructed offender.

`part_demand_weekly` is read by nothing in Phase 23B.

## 7. Isolation audit

`ml/eval/phase23b_isolation.py` (`cd3622c`) → `ml/artifacts/phase23b/isolation.json`. **ALL PASS.**

| check | result | its failing case |
|---|---|---|
| protected paths (118: db/gen_v6–v8, db/validator.py, docs/specs, dataset_structure.md, shipped.json, results/, reports/part1, every report at BASE, share.py, tcn.py, artifact_identity.py, phase19_identity.py, the Phase 23A paths) | **empty** vs BASE and in the worktree | the same diff over BASE^1..BASE is non-empty (6 files): fires |
| every EXISTING file under `ml/serve/` and `ml/tests/` | **unmodified**; added only: `ml/serve/rescue.py`, `ml/tests/test_phase23b_serve.py`, `ml/tests/test_phase23b_identity.py` | — |
| db/ | worktree clean; main checkout: **no content change** (seed_1001–1005 present; its pre-existing line-ending-only modifications are untouched) | — |
| stored configs | **393 / 393 recompute identically** against the module of the branch that created each: 340 Phase 0–15 vs `90a38ed`, 31 Phase 16 vs `caf613c`, 22 Phase 17 vs `a2301ef` (`test_phase23b_identity.py`) | a Phase 17 axis on a Phase 0–15 config changes the name; `90a38ed` cannot name it: fires |
| AST scan of the serve code | PASS | offender reading `inventory_position_weekly` and `load_ratio` flagged |
| serve tests | PASS (bit-exact ×5; mismatch, leaky-cache and future-row cases raise) | four mismatched bundles raise |
| constructed failing cases | G1 leaked arm flagged; G2 flipped label fires; G3 wrong seed does not reproduce; G4 leaked arm beats the permuted control by +0.355; G5 four seeds refused; G6–G8 in the serve tests | all fire |

## 8. Deviations, continuing from 240

| # | prior statement | measured / done | where |
|---|---|---|---|
| **240** | brief: BASE_SHA = the owner's paste | the paste was left as a placeholder; on the owner's instruction ("run this prompt on the pulled branch") BASE = the pulled `origin/HADES-v4-ml-pipeline` tip `545e282` (Phase 22 merged; `phase-22.md` present). The checked-out `integration/ml-pipeline-20261004` is its ancestor | header |
| **241** | world 2 at `data_worlds/v8_seed1002` (do not regenerate) | absent here. **Copied** from `db/gen_v8/seed_1002`, which Phase 20 recorded byte-identical to the generated world (deviation 180); 7 key tables verified equal by SHA-1 | stage 2a |
| **242** | Phase 22 clean caches | `ml/artifacts/phase22` absent. Rebuilt with Phase 22's builder unchanged (world-2 base cache 0.3 min, `v8clean` 0.3 min, `v8w1002clean` 0.2 min). Phase 22's clean-panel tests re-run unchanged → PASS with Phase 22's own counts | stage 2a |
| **243** | `phase20_decisions.reachable_and_class` imported unchanged | the module loads a Mac-only Phase 19 artifact at import, so the function's **source text** is executed unchanged (source SHA-1 `8ebeece6b80e`) | stage 2b |
| **244** | "all stored configs recompute identically" | `ml/tests/test_integration_identity.py` **STOPs here** on the first Phase 17 bundle (`…_s17_lean`). Its Phase 0–15 bucket compares with `90a38ed`, and the Phase 17 bundles exist only on this machine. A new test applies its own rule with the creating branch `a2301ef` for Phase 17: **393 / 393** | §7 |
| **245** | GO "at recall ≥ 0.20" | the operating point is chosen on validation (recall ≥ 0.20) and judged on test. NL's test recall is 0.189 and world 2's 0.191, so both miss; clean's is 0.254 | stage 2b, 2d |
| **246** | GO as a five-seed rule | the clean five-seed band is [0.7325, 0.7336], comfortably > 0.70, but the **snapshot-block** interval of the ensemble is [**0.697**, 0.773] and 3 of 9 snapshots are < 0.70. GO holds by the pre-registered rule and is **marginal under snapshot resampling** | stage 2b |
| **247** | P4 "replicates on world 2" | world 2 has no stored simulation proxy; the sign test is against its own-history baseline, as pre-registered | stage 2d |
| **248** | B1b re-score | not done (neural, no GPU). B1b also has no complete published band (seeds 27 / 37 / 47 never trained, Phase 17 deviation 152) | stage 2c |
| **249** | Stage 3: re-run the harness UNCHANGED with clean predictions | the clean bundles are absent (PENDING). Also, `read_heads` hard-codes the leaky bundle paths, so an "unchanged" harness cannot read clean bundles. The Mac run needs a new-file wrapper that changes only which bundle is read | stage 3 |
| **250** | the poison test "on the actual columns B1a reads" | run on B1a's own part-plant features at 6 sampled snapshot weeks, H_week only (Phase 22 used 12 weeks × 2 horizons for the columns). The rebuild reproduces the stored panel on 100.000% of values | stage 1b |
| **251** | serve: "never reads the leaking columns" | B1a CLEAN uses Phase 22's as-of-safe replacements, which keep the column names. The guarantee is enforced as: the serve path reads **only a clean cache** (`clean_of` set, `replaced_columns == LEAKING`; the leaky cache raises NotClean), and the AST scan forbids naming the columns or the two tables. The scan skips docstrings | stage 4 |
| **252** | Phase 17 A2 (the missed-rescue profile) | measured on the same simulation rows whose inputs are now LEAKY. It inherits Stage 3's PENDING | §3 |

## Five-line summary

1. **B1a read all nine Phase 22 leaking columns** (18 of 37 features; every one moves under future poison).
2. **Clean precision 0.733 vs published 0.823** at the Phase 17 operating point (Δ −0.091 [−0.104, −0.075]); still beats the
   proxy (0.619) and own history (0.631).
3. **Still GO** (Phase 15 class ALERT); marginal under snapshot resampling (block 0.697). Replicates on world 2 in sign, not at
   the alert level.
4. **Simulation: inputs LEAKY; restatement PENDING** (needs Phase 22's clean neural bundles from the Mac).
5. Report: `reports/part2/phase-23b.md`.
