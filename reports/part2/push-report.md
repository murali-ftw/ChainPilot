# Push report — Phases 16–20 to `origin/HADES-v4-ml-pipeline` (2026-10-04)

**STATUS: PUSHED.** Plain pushes only: no `--force`, no `--force-with-lease`, no `+refspec`, no `--delete`. `main` not
pushed and unchanged. The local-only branch `HADES-v4` was not pushed. Nothing in the code was measured or changed. This
report is the only commit added after verification.
**Remote:** `origin` = https://github.com/murali-ftw/ChainPilot.git. **Pushed from:** `/Users/muralik/Documents/Programs/HADES_v4`
(working tree clean, on `phase20`, untouched; the tests ran in a temporary detached worktree, since removed).

## Refs before and after

| ref | origin before | origin after | how |
|---|---|---|---|
| `HADES-v4-ml-pipeline` | `90a38ed` (Phase 15) | **`dd5971a`** (+87, fast-forward), then this report's commit | `git push origin HADES-v4-ml-pipeline` |
| `phase18` | *(absent)* | `62bb607` | new branch |
| `phase16-encoders` | `e7163e4` | `caf613c` (+2, fast-forward) | plain push |
| `integration/ml-pipeline-20261004` | *(absent)* | `1d5637a` | new branch |
| `phase17` | `a2301ef` | `a2301ef` (unchanged; not pushed) | — |
| `phase19` | `e9c0862` | `e9c0862` (unchanged; not pushed) | — |
| `phase20` | `b5a65a5` | `b5a65a5` (already on origin before this push; not pushed) | — |
| `main` | `cb7d3d1` | **`cb7d3d1` (unchanged)** | never pushed |

| tag (pushed one explicit refspec each) | commit |
|---|---|
| `backup/pre-integration-20261004` | `90a38ed` |
| `backup/pre-integration-20261004-phase16-encoders` | `caf613c` |
| `backup/pre-integration-20261004-phase17` | `a2301ef` |
| `backup/pre-integration-20261004-phase18` | `62bb607` |
| `backup/pre-integration-20261004-phase19` | `e9c0862` |
| `backup/pre-integration-20261004-phase20` | `b5a65a5` |

None of the tags existed on origin before (no SHA conflict to skip).

## Step 1 — verification (before any push)

| check | result |
|---|---|
| a. fetch; SHAs | local `dd5971ac887679ba5c1e839262b41d3f77bb5149`; origin `90a38eda267c8b45624640d908346344f4caf3b3`; origin/main `cb7d3d1a36689d2aea0f2d5bd805b27731a976b6` |
| b. origin pipeline is an ancestor of local | **yes**: fast-forward, 87 ahead / 0 behind |
| c. phase tips contained | `phase16-encoders` (caf613c), `origin/phase17` (a2301ef), `phase18` (62bb607), `origin/phase19` (e9c0862), `phase20` (b5a65a5): **all ancestors** |
| d. identity / anchors / tests / shipped.json | quoted below |
| e. size and path scan | 110 files changed, 15,838 insertions, 438 deletions. **No added file over 5 MB** (largest added: `results/observation2.md`, 39,765 bytes). **No added path** under `ml/artifacts`, `data_worlds/`, bundles, caches, or `*.db *.pt *.pth *.ckpt *.parquet *.npz *.npy *.csv` |
| f. protected paths | db/gen_v6, gen_v7, gen_v8, db/validator.py, synthetic_rules.md, dataset_structure.md, model_plan.md, docs/specs: **no change**. `ml/configs/shipped.json`: the authorised Stage 4 change (append-only; one existing line gained a trailing comma; 752 lines added under the new key `phase20_shipped`). **`results/`: two changes, authorised by the owner before pushing** (below) |
| g. backup tags | all six present locally (listed above) |
| h. tests on a clean checkout of `dd5971a` | `ml/tests`: **9 pass, 4 fail**. The 4 are the phase-scoped isolation audits the integration report lists (`test_phase17_identity`, `test_phase18_isolation`, `test_phase19_isolation`, `test_phase20_isolation`): not blockers. Import smoke: **119 of 126** `ml/` modules import; the 7 failures are the pre-existing import-time scripts the integration report lists (identical at `90a38ed`) |

**The `results/` exception (1f).** The brief expected no change under `results/`. Two changes are present, both from the
owner: `results/observation1.md` was modified by the owner's own commit `e7163e4` ("Observations", murali k, 2026-09-30) on
`phase16-encoders`, a branch this push publishes; `results/observation2.md` was added at the owner's explicit request in the
integration phase. The literal check failed, so the push was **paused and the owner asked**. The owner answered:
*"Authorised, push"*. No other protected path changed.

### 1d — quoted verbatim from `reports/part2/integration-report.md` at `dd5971a`

**Identity recompute:**
> **Identity end state (new test `ml/tests/test_integration_identity.py`).** Every stored config recomputes to the same
> `config_name`, `bundle_name`, `bundle_path_key` and `identity_of` as on the branch that created it (Phase 0–15 vs `90a38ed`,
> Phase 16 variants vs `caf613c`), and sits in the directory the merged code names: **371 / 371, including the 31 Phase 16
> variants, which are now nameable.** The brief's "402 total" assumed 371 + 31; on this machine the 371 already include the 31
> (earlier reports: "371 recomputed; named by this base 340; cannot be named 31"). Phase 17's B-arm bundles live only on the
> Windows machine and are not here (deviation 189).

**371 vs 402, stated plainly:** the 31 Phase 16 encoder variants **were** recomputed, and identically. They are part of the
371 stored configs (340 Phase 0–15 + 31 Phase 16), not in addition to them. "402" was a miscount in the integration brief, not
a coverage gap, and there is **no identity mismatch**. Not an open item. (Phase 17's own B-arm bundles are not on this
machine and so are not in the 371.)

**Behaviour anchors:**
> | anchor | recomputed | published | equal |
> |---|---|---|---|
> | arrival C-index (5-seed mean) | 0.674416 | 0.67442 | yes |
> | arrival lateness ROC-AUC | 0.709048 | 0.70905 | yes |
> | fill exact CRPS | 0.138762 | 0.13876 | yes |
> | capacity precision @ 1 / 5 / 10% | 0.904000 / 0.850844 / 0.810400 | 0.904 / 0.851 / 0.810 | yes |
> | capacity recall @ p 0.70 / 0.80 / 0.85 | 0.515025 / 0.317833 / 0.191489 | 0.515 / 0.318 / 0.191 | yes |
> | Phase 18 arrival blend lateness AUC (w 0.59) | 0.724749 | 0.7247 | yes |

**Test results:**
> | test_artifact_identity | PASS (6) |
> | **test_integration_identity** (new) | **PASS**: 371 / 371; Phase 19 15; axes fire |
> | **test_integration_anchors** (new) | **PASS** (below) |
> | **test_serve** (new) | **PASS** (Stage 5) |
> | test_no_drift_switch | PASS (4) |
> | test_no_lateness_substitution | PASS |
> | test_no_substitution_path | PASS |
> | test_phase7_bands | PASS |
> | test_single_order_policy | PASS |
> | test_phase17_identity | FAIL, **phase-scoped**: compares every config against `90a38ed`, and the 31 Phase 16 variants now exist on the tree |
> | test_phase18_isolation / 19 / 20 | FAIL, **phase-scoped**: […] Each **passed at its own tip** (1a) |

**The shipped.json change:**
> **What Stage 4 changed in `shipped.json`.** A new top-level key `phase20_shipped` listing, per item, its status, members
> (bundle root, task, name, `identity_of`, `row_family`), aggregation, decision metric, source and reversal condition.
> **Servable:** `arrival.ranked_late_list`, `arrival.interval`, `arrival.point_estimate_neural` (stand-in),
> `fill.materially_short_list_neural` (stand-in), `capacity.alert`. **Declared, not servable:** `arrival.point_estimate`
> (the Phase 19 blend), `fill.materially_short_list` (blend), `fill.b5flat22` (superseded), `rescue.predict_the_rescue`.

## Step 3 — remote verified after the push

| check | result |
|---|---|
| a. heads on origin | `HADES-v4-ml-pipeline` dd5971a · `phase16-encoders` caf613c · `phase17` a2301ef · `phase18` 62bb607 · `phase19` e9c0862 · `phase20` b5a65a5 · `integration/ml-pipeline-20261004` 1d5637a. **origin pipeline == local** (`dd5971a`) |
| b. tags on origin | all six backup tags, SHAs as listed |
| c. `main` | `cb7d3d1` before and after: **unchanged** |
| d. `git rev-list --left-right --count HADES-v4-ml-pipeline...origin/HADES-v4-ml-pipeline` | **`0 0`** |

## Not pushed, and why

- `main`: never, by rule.
- `HADES-v4` (local-only branch): not the target.
- `phase17`, `phase19`, `phase20`: already on origin at the same SHA as local; nothing to push.
- `--tags`: not used; only the six backup tags, one explicit refspec each.

## Rollback

```bash
# where origin/HADES-v4-ml-pipeline was before this push: 90a38ed (Phase 15)
git rev-parse backup/pre-integration-20261004            # 90a38ed...
git log --oneline -1 backup/pre-integration-20261004
git tag -l "backup/pre-integration-20261004*"            # per-phase tips before integration

# safe undo of the merged content, WITHOUT rewriting published history (a new commit on top):
git revert -m 1 dd5971a          # the second integration merge (Stages 2-6)
git revert -m 1 ca1185f          # the Stage 1 merge (Phases 16-20)
```

**Restoring `origin/HADES-v4-ml-pipeline` to `90a38ed` itself would need a force-push.** This prompt never performs one.
Whether to do that is the owner's separate decision. `git revert` above is the history-preserving alternative.
