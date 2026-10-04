# Integration report — Phases 16–20 into `HADES-v4-ml-pipeline` (2026-10-04)

**Audience:** the repository owner, deciding whether to push the merged `HADES-v4-ml-pipeline`.
**Machine:** Apple M4 Pro, macOS, torch 2.14.0 (MPS), LightGBM 4.7.0. **No training of any kind.**
**Window:** started 11:25 IST, stop 19:25 IST (8 h). Finished well inside it.
**Status:** merged into `HADES-v4-ml-pipeline` **locally**. **Not pushed. `main` untouched.** Nothing was rebased,
force-pushed or deleted. Your working tree (`HADES_v4`, on `phase20`) was not touched: all work ran in separate git
worktrees (`../HADES_v4_integration`, and a scratch `../HADES_v4_tiptest` used to run each phase's tests at its own tip).

## Stage 0 — plan

| branch | tip | merge-base with `origin/HADES-v4-ml-pipeline` | commits ahead | on the remote? |
|---|---|---|---|---|
| `phase16-encoders` | `caf613c` | `90a38ed` | 10 | `origin/phase16-encoders` is at `e7163e4`; **local is 2 ahead** (`8a0d9c9` Phase 16 report, `caf613c` "Phase 17": a byte-identical copy of the three Phase 17 report files) |
| `phase17` | `a2301ef` | `90a38ed` | 24 | yes (`origin/phase17`) |
| `phase18` | `62bb607` | `90a38ed` | 34 (= phase17 + 10) | **no** (local only) |
| `phase19` | `e9c0862` | `90a38ed` | 47 (= phase18 + 13) | yes (`origin/phase19`) |
| `phase20` | `b5a65a5` | `90a38ed` | 64 (= phase19 + 17) | **no** (local only) |

- **Graph.** Every branch forks from `90a38ed` (the pipeline tip, Phase 15). `phase17 → phase18 → phase19 → phase20` is a
  linear chain, each containing the one before. `phase16-encoders` is a separate fork. No branch fails to descend from the
  expected base.
- **Windows/CUDA commits on `phase17`:** `167eafb` (device: CUDA, psapi RSS, `HADES_DEVICE`), `11c6019` (low-VRAM mode),
  `b2c7c31` (PowerShell queue).
- **Files touched by both forks:** `ml/artifact_identity.py`, `ml/train/loop.py`, and the three Phase 17 report files
  (byte-identical on both sides).
- **Phase 16:** branch name `phase16-encoders`; report `reports/part2/phase-16.md` exists (`8a0d9c9`).
- **Merge order, from the graph:** `phase16-encoders` → `phase17` → `phase18` → `phase19` → `phase20`. The fork goes first
  so its identity axes are present before Phase 17 adds its own. The chain then merges in descent order, each merge adding
  only its own commits.

**Backup tags** (local; `git tag -l "backup/*"`):

| tag | commit |
|---|---|
| `backup/pre-integration-20261004` | `90a38ed` (HADES-v4-ml-pipeline before integration) |
| `backup/pre-integration-20261004-phase16-encoders` | `caf613c` |
| `backup/pre-integration-20261004-phase17` | `a2301ef` |
| `backup/pre-integration-20261004-phase18` | `62bb607` |
| `backup/pre-integration-20261004-phase19` | `e9c0862` |
| `backup/pre-integration-20261004-phase20` | `b5a65a5` |

(Per-branch tags use a `-<branch>` suffix: git cannot hold `backup/pre-integration-20261004` as both a tag and a prefix.)

## Stage 1 — merge

### 1a. Each branch's own audit at its own tip (before merging)

| tip | `test_artifact_identity` | phase audit | result |
|---|---|---|---|
| phase16-encoders `caf613c` | PASS (6) | `ml/eval/phase16_identity_gate.py` | **FAIL**: 280 mismatches over 386 checks, **all from the 31 Phase 16 encoder-variant configs**, whose names legitimately differ from the branch-point module; all 340 other configs identical. The gate was written before those bundles existed (deviation 188) |
| phase17 `a2301ef` | PASS (6) | `test_phase17_identity` | PASS |
| phase18 `62bb607` | PASS (6) | `test_phase18_isolation` | PASS |
| phase19 `e9c0862` | PASS (6) | `test_phase19_isolation` | PASS (371 identities; 15 Phase 19 bundles) |
| phase20 `b5a65a5` | PASS (6) | `test_phase20_isolation` | PASS |

### 1b–1c. Merges (all `--no-ff`, on `integration/ml-pipeline-20261004`, cut from `origin/HADES-v4-ml-pipeline`)

| step | merge commit | conflicts | identity recompute after |
|---|---|---|---|
| phase16-encoders | `fa90a4f` | none | 371 / 371 (340 Phase 0–15 + 31 Phase 16) |
| *(identity test added)* | `2371f3c` | — | — |
| phase17 | `f5e79f9` | **`ml/artifact_identity.py`** (2 hunks) | 371 / 371 |
| phase18 | `f909b30` | none | 371 / 371 |
| phase19 | `2b1832f` | none | 371 / 371; 15 Phase 19 bundles; axes fire |
| phase20 | `d676377` | none | 371 / 371; 15 Phase 19 bundles; axes fire |

`git log --oneline --first-parent 90a38ed..integration/ml-pipeline-20261004` lists every integration commit (anchor test `149d7bc`, client request `f80c8ae`, serving + shipped.json `27a13e4`, observation2 + decision record `41a7f9a`, this report).

**The one conflict.** Phase 16 and Phase 17 each appended identity axes in the same two places (`config_name`,
`identity_of`). Resolution: **union**, Phase 16's encoder axes first and then Phase 17's (`cap_target`, `lean_encoder`), each
omitted at its default. No stored bundle carries both kinds of axis, so the order changes no existing name.
`ml/train/loop.py` auto-merged. `ml/train/phase19_identity.py` keeps working (15 bundles named correctly; a set
`row_family` gives a distinct name).

**Identity end state (new test `ml/tests/test_integration_identity.py`).** Every stored config recomputes to the same
`config_name`, `bundle_name`, `bundle_path_key` and `identity_of` as on the branch that created it (Phase 0–15 vs `90a38ed`,
Phase 16 variants vs `caf613c`), and sits in the directory the merged code names: **371 / 371, including the 31 Phase 16
variants, which are now nameable.** The brief's "402 total" assumed 371 + 31; on this machine the 371 already include the 31
(earlier reports: "371 recomputed; named by this base 340; cannot be named 31"). Phase 17's B-arm bundles live only on the
Windows machine and are not here (deviation 189).

**Device check (Windows/CUDA commits).** `device.get_device()` on this Mac: **`mps`, float32, before (`90a38ed`) and after
the merge.** CUDA is unavailable here; `HADES_PANEL_HOST` / `HADES_TCN_CHUNK` default off (`PANEL_HOST False`,
`TCN_CHUNK 0`). The Mac path is unchanged.

### 1d. Full test suite and import smoke (final integration tip)

| test | result |
|---|---|
| test_artifact_identity | PASS (6) |
| **test_integration_identity** (new) | **PASS**: 371 / 371; Phase 19 15; axes fire |
| **test_integration_anchors** (new) | **PASS** (below) |
| **test_serve** (new) | **PASS** (Stage 5) |
| test_no_drift_switch | PASS (4) |
| test_no_lateness_substitution | PASS |
| test_no_substitution_path | PASS |
| test_phase7_bands | PASS |
| test_single_order_policy | PASS |
| test_phase17_identity | FAIL, **phase-scoped**: compares every config against `90a38ed`, and the 31 Phase 16 variants now exist on the tree |
| test_phase18_isolation / 19 / 20 | FAIL, **phase-scoped**: each asserts `git diff <its own base> HEAD` is empty on protected paths, and the merged tree carries later phases' files, Phase 16's identity axes (`c5ce3f2`), your "Observations" commit to `results/observation1.md` (`e7163e4`) and, for Phase 20, the authorised `shipped.json` change. Each **passed at its own tip** (1a) |

**Import smoke:** 122 `ml/` modules (excluding tests), each imported in its own process: **115 import**. The 7 that fail
(`learnability_windowed`, `phase5_predict_fold`, `phase5_verify`, `phase6_verify`, `run10_closeout`, `run8_grid`,
`run9_grid`) are scripts that run on import (argument parsing, v6/v7 worlds). **They fail identically at `90a38ed`**:
pre-existing, not merge-caused. No merge-caused breakage was found, so no fix commit was needed.

### 1e. Behaviour anchors (read-only, stored bundles, merged code)

| anchor | recomputed | published | equal |
|---|---|---|---|
| arrival C-index (5-seed mean) | 0.674416 | 0.67442 | yes |
| arrival lateness ROC-AUC | 0.709048 | 0.70905 | yes |
| fill exact CRPS | 0.138762 | 0.13876 | yes |
| capacity precision @ 1 / 5 / 10% | 0.904000 / 0.850844 / 0.810400 | 0.904 / 0.851 / 0.810 | yes |
| capacity recall @ p 0.70 / 0.80 / 0.85 | 0.515025 / 0.317833 / 0.191489 | 0.515 / 0.318 / 0.191 | yes |
| Phase 18 arrival blend lateness AUC (w 0.59) | 0.724749 | 0.7247 | yes |

### 1f. Into `HADES-v4-ml-pipeline`

`git merge --no-ff integration/ml-pipeline-20261004` on `HADES-v4-ml-pipeline` → **`ca1185f`** (Stage 1 state).
Stages 2–6 were then committed on the integration branch and merged into `HADES-v4-ml-pipeline` a second time with
`--no-ff`. That final merge commit is the tip: `git log -1 --oneline HADES-v4-ml-pipeline`. **Not pushed.**

## Stages 2–5 — what was added

| stage | file | note |
|---|---|---|
| 2 | `results/observation2.md` | v3.0, supersedes and does not edit `observation1.md`. Placed in `results/` on your instruction (`observation1.md` lives there too) |
| 3 | `docs/client/data_request_v2.md` | five asks with measured value and honest limit; gate F0 checklist |
| 4 | `docs/decisions/shipped_phase20.md`; `ml/configs/shipped.json` → `phase20_shipped` (version `phase20-shipped-1`) | appended; every earlier key unchanged (one existing line gained a trailing comma). 9 items: 5 servable, 4 declared |
| 5 | `ml/serve/guard.py`, `features.py`, `service.py`; `ml/tests/test_serve.py` | identity guard, as-of feature builder from source, neural / ensemble loader |

**What Stage 4 changed in `shipped.json`.** A new top-level key `phase20_shipped` listing, per item, its status, members
(bundle root, task, name, `identity_of`, `row_family`), aggregation, decision metric, source and reversal condition.
**Servable:** `arrival.ranked_late_list`, `arrival.interval`, `arrival.point_estimate_neural` (stand-in),
`fill.materially_short_list_neural` (stand-in), `capacity.alert`. **Declared, not servable:** `arrival.point_estimate`
(the Phase 19 blend), `fill.materially_short_list` (blend), `fill.b5flat22` (superseded), `rescue.predict_the_rescue`.
**No LightGBM model has ever been persisted** (Phases 7–20 stored predictions only), and the B1a artifacts are on the Windows
machine, so these cannot be loaded without a refit, which this phase forbids (deviation 191).

**Serving tests:**
- (i) all 25 members of the 5 servable items, on 2025-01-06 and 2025-12-08. The 5 no-graph fill members reproduce the
  stored predictions **bit-exactly**. That also proves the from-source feature builder equals the training one, since those
  members consume season + cadence. The 20 graph members are within **4.3e-6** (the tolerance is 1e-5); see deviation 190.
- (ii) the guard **fires** on a wrong seed's bundle, on a Phase 19 row-family bundle standing in for its incumbent (they share
  `identity_of`: the exact silent-substitution case), on a missing bundle, and on a declared item (`NotServable`). The correct
  member passes.
- (iii) AST scan of `ml/serve/*.py`: 0 references to `inventory_position_weekly` or privileged paths; the scanner flags
  constructed offenders and passes a docstring mention.

## Deviations, continuing from 188

| # | statement | what happened | where |
|---|---|---|---|
| **188** | 1a: each branch's audit at its own tip | Phase 16's own S4 gate fails at its tip, by construction: it compares every stored config to the branch point, and its 31 encoder-variant bundles (created after the gate was written) legitimately differ. All 340 other configs are identical | 1a |
| **189** | 1c: "402 total (371 + 31)" | the 371 stored configs already include the 31 Phase 16 variants. End state: **371 / 371, all nameable**. Phase 17's B-arm bundles are only on the Windows machine | 1c |
| **190** | 5d(i): "serving reproduces the stored predictions bit-exactly" | true for the **no-graph** members (5 / 5, bit-exact). **Impossible for graph members on MPS**: the message-passing scatter-add (`index_put_with_accumulate_mps`) is nondeterministic. Every bundle's `train_log.json` records it, and two serving runs of the same model differ by up to 2.4e-7. The stored predictions are one draw. Graph members are asserted within 1e-5; observed ≤ 4.3e-6 | Stage 5 |
| **191** | 4/5: "a LightGBM loader … reads stored bundles" | **no LightGBM model file exists anywhere**: every fit stored predictions only. The loader cannot be built without a refit (training, not allowed). Every LightGBM-containing item is *declared, not servable*, each with a servable stand-in named. Deviation 46 is therefore **addressed, not removed**: the service refuses instead of silently substituting | Stage 4 |
| **192** | Stage 2: "write observation2.md next to observation1.md" | done: both are in `results/`. Written there on your explicit instruction, although `results/**` is otherwise protected; `observation1.md` untouched | Stage 2 |
| **193** | 1f then "commit on HADES-v4-ml-pipeline" | the 1f merge (`ca1185f`) was made after Stage 1's verification; Stages 2–6 were merged in a second `--no-ff` merge | 1f |
| **194** | "the user's working tree is never touched" | your tree was not touched. Two side effects outside the worktrees: (a) the data (`ml/artifacts`, `db/gen_v8/seed_*`, `data_worlds`, `venv`) was **symlinked** into the worktrees, so test runs wrote small JSON files into your `ml/artifacts` (gitignored): `integration_anchors.json`, `integration/serve_tests.json`, and **`phase16_identity_gate.json`, overwritten** by re-running Phase 16's gate at its tip; (b) six temporary symlink patterns were added to `.git/info/exclude` and **removed at the end** | 1a, 1d |
| **195** | 1d: "full ml/tests suite" | the four phase-scoped isolation audits fail on the merged tree, as the brief anticipated; each passed at its own tip | 1d |
| **196** | 4: "you may push the integration branch and the tags" | **nothing was pushed**: neither the integration branch, the tags, nor `HADES-v4-ml-pipeline`. Commands below | — |

## Not done

- Nothing pushed (by design for the pipeline; by choice for the integration branch and the tags).
- No LightGBM item is servable; that needs a phase allowed to fit and **persist** LightGBM models.
- Deviation 122 (simulation random stream) remains open; see `results/observation2.md` Appendix D.

## Commands

```bash
cd /Users/muralik/Documents/Programs/HADES_v4

# (1) inspect the merged pipeline
git log --oneline --graph -25 HADES-v4-ml-pipeline
git diff --stat origin/HADES-v4-ml-pipeline HADES-v4-ml-pipeline
git log --oneline origin/HADES-v4-ml-pipeline..HADES-v4-ml-pipeline | wc -l

# (2) push it (and, optionally, the integration branch and the backup tags)
git push origin HADES-v4-ml-pipeline
git push origin integration/ml-pipeline-20261004
git push origin 'refs/tags/backup/pre-integration-20261004*'

# (3) roll back the local pipeline branch to its pre-integration state (before pushing)
git branch -f HADES-v4-ml-pipeline backup/pre-integration-20261004
#     after pushing, revert instead of rewriting published history:
#     git revert -m 1 <merge-commit>

# worktree used for the integration (safe to remove; the branch remains)
git worktree remove --force ../HADES_v4_integration
```
