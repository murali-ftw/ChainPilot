# Merge of Phase 23B (`origin/phase23b`) into `HADES-v4-ml-pipeline`

**Audience:** the project owner, and whoever copies Phase 23B's artifacts to the Mac.

**Machine:** the Mac main checkout, CPU only (`HADES_DEVICE=cpu`). No model was trained, fitted or re-scored.

**Date:** 2026-10-05, 23:21–00:20 IST, inside the 3 h stop.

**Status: MERGED.** Every Step 3 check passed. The Phase 23B serve tests are SKIPPED-NO-ARTIFACTS (§5). Nothing was merged
into or pushed to `main`.

## 1. SHAs

| what | SHA |
|---|---|
| local `HADES-v4-ml-pipeline` before | `4f0bd5af5adc95d0e42ee9f309853feef2e0f90f` |
| `origin/HADES-v4-ml-pipeline` before | `4f0bd5af5adc95d0e42ee9f309853feef2e0f90f` (already equal: nothing unpushed; deviation 262) |
| `origin/phase23b` | `542d324cfe9eb84588346a8be47e66afafa651b9` (not altered) |
| merge-base | `545e282cb1f5f0a4b3c9168ba379de49f820a3ad` (as expected) |
| left / right count | 6 / 11 (as expected) |
| integration merge (worktree `../HADES_v4_merge23b`, branch `integration/phase23b-merge-20261005`) | `9ade122` |
| `HADES-v4-ml-pipeline` after (`--no-ff` of the integration branch) | `11c428e` |
| `origin/main`, recorded at Step 0e | `cb7d3d1a36689d2aea0f2d5bd805b27731a976b6` |
| backup tag `backup/pre-23b-merge-20261005` | `4f0bd5a` |
| backup tag `backup/phase23b-20261005` | `542d324` |

## 2. Commits unique to each side

**Local only (6):**
- `4f0bd5a` Phase 23AC final report and `observation3.md`;
- `ec9c68b` track reports;
- `10f88da` T4 charts;
- `006ebc9` T2 gate;
- `d401cb3` Step 3 code;
- `1a73f75` pre-registration.

**`origin/phase23b` only (11):**
- `542d324`, `535361d` (`phase-23b.md`);
- `cd3622c` (identity vs the creating branch);
- `6593b57` (Stage 5 isolation);
- `3da031c`, `3f52a1c`, `4e351fe`;
- `13d5bd0` (Stage 4 serve);
- `ea44b63` (Stage 2 rescue);
- `ee62f46` (Stage 1 audit);
- `fb998c0` (pre-registration).

## 3. Files, intersection and safety scan

**Phase 23B side: 12 files, all added (A):**
- `ml/eval/phase23b_audit.py`, `phase23b_cleancheck.py`, `phase23b_isolation.py`, `phase23b_rescue.py`;
- `ml/serve/rescue.py`;
- `ml/tests/test_phase23b_identity.py`, `test_phase23b_serve.py`;
- `reports/part2/phase-23b-preregistration.md`, `phase-23b.md`;
- `reports/part2/phase23b/stage1_audit.md`, `stage2_rescue_clean.md`, `stage3_simulation.md`.

**Local side:** 34 files, all added (Phase 23AC).

**Intersection: none.**

**Safety scan of what the merge brings in:**
- no file over 5 MB (largest: `phase23b_rescue.py`, 21,887 bytes);
- no path under `ml/artifacts`, `data_worlds/` or a cache;
- no `*.pt`, `*.pth`, `*.ckpt`, `*.db` or `*.parquet`;
- no modification or deletion of any existing file;
- no change to `db/`, the specs, `shipped.json`, `share.py`, `tcn.py`, `artifact_identity.py` or `phase19_identity.py`.

**Step 0d (also pushed):** no unpushed commits. `origin/HADES-v4-ml-pipeline` is an ancestor of (equal to) the local tip.

**Step 3f:** the diff `4f0bd5a..9ade122` is 12 additions and 0 modifications. **Step 4:** the files the merge changes and
`git status --porcelain` have no file in common (the tree was clean).

## 4. Tests: before and after the merge

**Runs:**
- **Before:** the main checkout at `4f0bd5a` for the full suite and the Phase 23AC isolation; a detached worktree at
  `542d324` for the Phase 23B tests.
- **After:** the integration worktree at `9ade122`.

Every `ml/tests/test_*.py` was run standalone; `test_phase7_bands` via `unittest`.

| test | before | after | reading |
|---|---|---|---|
| test_artifact_identity, test_integration_anchors, test_integration_identity, test_leak_guard, test_no_drift_switch, test_no_lateness_substitution, test_no_substitution_path, test_phase21_grpstats, test_phase22_clean_panel, test_phase22_serve, test_phase23ac_t1_card, test_phase23ac_t2_isolation, test_phase23ac_t2_serve, test_phase23ac_t4_metrics, test_phase7_bands, test_single_order_policy | PASS | PASS | — |
| test_phase23b_identity | PASS at `542d324`: 371 / 371 (Phase 0–15 340, Phase 16 31, Phase 17 0) | PASS, same counts | Phase 17 bundles live on the Windows machine (Phase 23B counted 393 there) |
| test_phase23b_serve | FileNotFoundError: `ml/artifacts/phase23b/rescue_configured.json` | same | **SKIPPED-NO-ARTIFACTS** (§5) |
| test_phase23ac_isolation | **PASS 6 / 6** | **FAIL** at `test_protected_paths_unchanged`: "Phase 23B paths written by this phase" | **LEGITIMATE.** It asserts that nothing since its base `545e282` touched the 23B paths; the merge brings them in. Its other five checks were run one by one after the merge: **all PASS** |
| test_phase17_identity | FAIL ("identity changed" on a Phase 16 `encheteromp` config) | same | **pre-existing**, not merge-caused: it compares with `90a38ed`, which cannot name Phase 16 variants |
| test_phase18/19/20/21/22_isolation | FAIL ("protected paths changed": `results/observation*.md` and later reports since each base) | same | **pre-existing**, legitimate: each asserts an empty diff against its own phase base |
| test_serve | FAIL: Phase 22 fill bundle `…_s7_rfseason+cadence` not bit-exact, 4.17e-7 | same | **pre-existing**: CPU serving vs MPS-stored predictions (Phase 23AC P3) |
| `scripts/run_leak_guard.sh` | exit 0 | **exit 0** | the nine fail poison; all clean columns pass; no Phase 23B feature path flagged |
| `ml/eval/phase23b_isolation.py` | not run | not run | deviation 260 |

**Result:** the merge broke nothing. Every failure after the merge either failed identically before it, or (Phase 23AC
isolation) fails because its base moved. **No fix commit was needed.**

**Import smoke** (after the merge, each module in its own process):
- **159 OK, 0 FAIL.**
- **27 not imported**, because their top level runs code (scripts without a main guard). Not importing them keeps the
  "no model run" rule.
- **Includes:** all seven new Phase 23B modules, each OK.

**Identity recompute** (Phase 23AC `test_identities`, after the merge):
- **371 / 371** stored configs recompute identically (named by the base: 371);
- Phase 19 bundles **15**, Phase 22 bundles **25**, all at their identity names;
- **Phase 23B bundles: 0 on this machine.** No bundle renamed; no mismatch.

**Stored artifacts:** the 188-file Phase 23AC manifest is unchanged (0 changed). `db/` is clean; five seed directories are
present; world-2 hashes are equal to Phase 20.

## 5. SKIPPED-NO-ARTIFACTS: what to copy

`ml/artifacts/` is gitignored and `ml/artifacts/phase23b/` does not exist on the Mac. Copy the whole directory:

- **from (Windows):** `C:\Users\0rayc\Documents\GitHub\ChainPilot-phase23b\ml\artifacts\phase23b\`
- **to (Mac):** `/Users/muralik/Documents/Programs/HADES_v4/ml/artifacts/phase23b/`

It must contain at least:
- `rescue_configured.json`;
- `bundles/rescue_lgbm_v8clean_s{7,17,27,37,47}/` (each with `model.txt`, `identity.json`, `test_rows_200.npz`).

The Mac's `ml/artifacts/cache/v8` and the Phase 22 clean caches are already present. Then run:

```
HADES_DEVICE=cpu ./venv/bin/python ml/tests/test_phase23b_serve.py
```

The guard refuses any copy whose booster SHA-1 or identity hash differs from `rescue_configured.json`.

## 6. Deviations, from 260

| # | rule | what was done |
|---|---|---|
| 260 | "run test_phase23b_* isolation" | `ml/eval/phase23b_isolation.py` was **not run**. Its G3 check refits a LightGBM (wrong-seed failing case), which this merge forbids, and it reads a hard-coded Windows path (`MAIN = C:\Users\0rayc\...`). Its non-fitting checks were reproduced instead: protected-path diff (git: at `542d324` empty; after the merge only `results/observation3.md`, added by Phase 23AC and in its own `P23A` list, so legitimate), identities (`test_phase23b_identity.py`), serve (skipped, §5), reads (`test_phase23ac_isolation` AST scan) |
| 261 | verify in a new worktree | the worktree has no `venv`, `ml/artifacts` or world data (all gitignored). Symlinks to the main checkout's copies were added, hidden from `git status` via `GIT_CONFIG_COUNT` / `core.excludesFile` pointing to a scratch file: no repository config changed. Test outputs written under `ml/artifacts/` therefore landed in the main checkout's (gitignored) artifacts. The last write of `phase23ac/isolation_audit.json` is the pre-merge PASS run. No bundle, prediction or cache was written |
| 262 | Step 0d expects unpushed Phase 22 / 23AC commits | `origin/HADES-v4-ml-pipeline` already equalled the local tip `4f0bd5a`; the push carries only `9ade122`, `11c428e`, the Phase 23B commits and this report |
| 263 | "fix only merge-caused breakage" | seven failures exist before the merge (`test_phase17_identity`, Phase 18–22 isolation, `test_serve`'s CPU rounding). They are recorded, not fixed: none is merge-caused, and each is a phase-scoped assertion or the known CPU vs MPS difference |

## 7. What Phase 23B resolved (QUOTED from `reports/part2/phase-23b.md`)

- **Predict-the-rescue B1a read all nine leaking columns** (18 of its 37 features; each moves under future poison).
- **Clean precision 0.733 vs 0.823 published** at the Phase 17 operating point (Δ −0.091 [−0.104, −0.075]). It still beats
  the simulation proxy (0.619) and own history (0.631).
- **Still GO** (Phase 15 class ALERT, PARTIAL), but **marginal under block resampling** (snapshot-block interval
  [0.697, 0.773]; 3 of 9 snapshots below 0.70).
- **Replicates on world 2 in sign** (+0.142 [+0.129, +0.154] over own history), not at the alert level (0.596, WATCHLIST).
- **The simulation's inputs are LEAKY.** Its restatement is **PENDING**: it needs Phase 22's clean neural bundles, which
  are on this Mac (`ml/artifacts/phase22/`), plus a new-file wrapper (Phase 23B deviation 249).
- **The clean B1a bundle serves bit-exactly** (5 seeds, 200 test rows, max |Δ| 0.0, on the Windows machine).
- B1b (neural rescue) is not quotable until retrained clean.

**Not edited by this merge:**
- `results/observation3.md` and the Phase 23AC metrics pack (`reports/part2/phase23ac/t4/metrics_pack.md`) **still show
  predict-the-rescue and the simulation as PENDING PHASE 23B**.
- A follow-up refreshes them: rescue becomes 0.733 clean; the simulation stays PENDING until its clean re-run.

## 8. Rollback

Nothing below is run here. A rollback of a pushed branch needs the owner's decision, and must be a revert, not a force-push.

```
# local only, before pushing:
git switch HADES-v4-ml-pipeline && git reset --hard backup/pre-23b-merge-20261005
# after pushing (no history rewrite): revert the merge commit, keeping the pipeline side
git revert -m 1 11c428e
# Phase 23B's own tip stays at backup/phase23b-20261005 (= origin/phase23b 542d324)
git worktree remove ../HADES_v4_merge23b ; git worktree remove ../HADES_v4_23btip
```
