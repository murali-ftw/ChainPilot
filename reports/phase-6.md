# Phase 6 — Training loop (guide 6.1–6.3)

**Verdict.** The loop reproduces Phase 5: **6 of 7 gate checks pass**. The one failure — arrival SHARE-lite h⁴, 1.9× a
band borrowed from h⁰ — sits at **0.39× its own measured band** and is traced to run-to-run MPS non-determinism that
Phase 5's own code shows. The seed triples confirm every Addendum B verdict under its own band, with one refinement:
**capacity v6's validation/test disagreement is real on all six seeds and twice as large as reported**. The arrival
fallback switch is implemented, verified in both states, and engages on its own on all three v7 seeds. **Phase 7 may
start.**

## 1. Header

| | |
|---|---|
| Device | **MPS**, Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32, `num_workers=0`; seeds set for torch, numpy, python and MPS; deterministic algorithms requested with `warn_only=True` (strict mode raises — §2). No CPU fallback was needed. |
| torch | 2.14.0 |
| Population | **all 16,072 channels**; fixed split train ≤ 2023 / val 2024 / test 2025; fit window 2019–2025; inductive — no entity IDs |
| Shipped configuration | `ml/configs/shipped.json` — arrival SHARE-lite h⁴ @ 2.5e-4, fill h⁰ @ 1.25e-4, capacity HeteroMP h⁴ @ 2.5e-4, shortage HeteroMP h¹ @ 1.25e-4 (diagnostic); gate and reconstructed feature **off**. No learning rate retuned. |
| **Cells** | **22 bundles**: **18 trained** by the loop and **4 converted** from Phase 5 checkpoints (steps 3–8, reload verified with `allclose`). All 18 trained cells stopped on **patience**; **none reached the cap**. Plus: 6 two-epoch runs for the §4 repeat test, and 6 smoke bundles (fill trained, fill converted, arrival h⁰ converted, arrival SHARE-lite h⁴, capacity, shortage) in a scratch root. |
| **Wall-clock** | queues **≈ 5.4 h elapsed** (08:05 → 13:27 IST), two queues sharing MPS; **10.66 h summed** across trained cells, 2.5 min across conversions. Per cell: arrival SHARE-lite h⁴ 34–50 min, capacity HeteroMP h⁴ 19–50 min, fill h⁰ 20–30 min, arrival h⁰ 20–61 min, capacity h⁰ 15 min. |
| Projection before Stage 4 | printed at 10:10 from measured cells: **≈ 4.7 h** remaining; actual 3.3 h (capacity cells ran shorter than projected). **No cells cut** — the brief's cut order (fill v7, then capacity v7) was not needed. |
| **Peak RSS** | **3.12 GB** per training process (highest bundle) |
| Nondeterministic ops recorded | `index_put_with_accumulate_mps` — in every trained bundle |
| New files | `ml/train/loop.py`, `ml/train/folds.py`, `ml/configs/shipped.json`, `ml/train/phase6_queue.py`, `ml/train/phase6_verify.py`, `ml/eval/phase6_collect.py`, this report |
| Edited | `docs/implementation_guide.md` (6.1–6.3 annotations, deviation rows 12–14); `ml/train/phase5_heads.py` (additive: `device_inputs` returns the normaliser) |
| Not touched | `db/gen_v6/`, `db/gen_v7/`, `validator.py`, `synthetic_rules.md`, `dataset_structure.md`, `model_plan.md`, every prior report |
| `inventory_position_weekly` | **not read** |
| Artifacts | `ml/artifacts/bundles/<task>/<world>_<arch>_h<d>_lr<lr>_s<seed>/`; `phase6_verify.json`, `phase6_repeat_test.json`, `phase6_fallback_test.json`, `phase6_fallback_seeds.json`, `phase6_drift_arrival_v7.json`, `phase6_collect.json`, `phase6_tables.md`, `phase6_queue_{A,B}.json` |

---

## 2. Guide reconciliation

Guide steps 6.1–6.3 predate Phase 5, and the generated worlds differ from the ones they describe. Each disagreement was
resolved the way Phases 1–5 resolved theirs: change the code, or annotate the guide, and say which.

| # | topic | guide 6.x specifies | Phase 5 / the data requires | resolution |
|---|---|---|---|---|
| 1 | **folds** | 8 rolling origins (spec §9.2); `folds.py`; verify max(train) < min(test) | the brief carries forward **one fixed split** (train ≤ 2023, val 2024, test 2025) — every number the reproduction gate checks was measured on it | **code**: `ml/train/folds.py` — `fixed_split` with the verify **asserted** on every task and world; the eight rolling origins implemented with the same assert, **defined, not run** (Phase 8.2). **Guide annotated** |
| 2 | **snapshots** | 83 snapshots, 2016-12-26 → 2025-09-22 (spec: 115 at 28 days) | measured: **83 at 6-week spacing, 2016-07-04 → 2025-12-08**; 61 in the fit window | **guide annotated** |
| 3 | **covid regime** | exclude `regime_flag = 'covid'` from primary training (spec §9.4) | `snapshots.csv` has **no regime flag**; `calendar` flags covid 2020-03-01 → 2020-09-30 at every plant — **5 of 44 training snapshots**. Phases 2–5 never excluded them; excluding changes the population behind every reproduced number | **code**: `folds.covid_mask`, **off by default**. **Guide annotated**; regime robustness recorded as open (§8) |
| 4 | **seeding** | seed torch / numpy / python; `torch.use_deterministic_algorithms(True)` | measured on MPS: **strict mode raises** (`index_put_with_accumulate_mps` has no deterministic kernel) | **code**: seeds all three + MPS, requests deterministic algorithms with `warn_only=True`, records the ops that warned in every bundle. **Guide annotated** |
| 5 | **reproduction** | "the same seed and fold reproduce the metric **bit-for-bit**" | measured: same seed → **bitwise-identical initial weights**; two identical forward passes differ by **up to 3.0e-8** | **guide annotated**: reproduction is `allclose` on predictions and **within the per-metric seed band** on metrics, never `equal` |
| 6 | **reproduction floor** | ≥ 5 seeds × 8 folds per configuration | the brief runs **3 seeds on the fixed split** | **guide annotated**: the band measured here has no year-to-year component, and says so where used |
| 7 | **stamps** | five identifiers with fixed values (`hades-v4-synth-1.0`, `1.0`, …) | `snapshots.csv` carries `rane-v5-seed1001`, `v5.0`, `labels-v5`, and the **generator's** commit (per world) | **code**: every bundle stamps the file's values plus the repository commit and a dirty flag. **Guide annotated** |
| 8 | **`model_outputs`** | write predictions in its shape; verify the join on `(snapshot_id, entity_id, task)` | the table has **no `task` column** | **code**: `model_name = hades-<task>`; each bundle writes `model_outputs.csv.gz` and **asserts** a one-to-one join with an actual on every row. Written under `ml/artifacts/bundles/`, never into `db/`. **Guide annotated** |
| 9 | **early stopping** | on the fold's own validation slice | same | **matches** |
| 10 | **recalibration is part of the model** | *silent* | Phase 5 Addendum A: fitted on validation, **refitted every retraining** (fill T = 1.008 / 1.089 / 1.030 across seeds) | **code**: pipeline steps 5–6, per run. **Guide annotated** |
| 11 | **validation predictions + checkpoints by default** | "checkpoint" only | Phase 5 saved neither; Addendum A retrained four cells to run a designed fix | **code**: steps 3–4, always. **Guide annotated** |
| 12 | **drift monitor** | *silent* | Addendum B: label-free drift predicts calibration damage; arrival needs an h⁰ fallback | **code**: step 7 saves the baseline; `predict()` emits drift per batch and overall, and switches arrival's distribution to h⁰'s. **Guide annotated** |
| 13 | **the normaliser** | *silent* | a model evaluated with a refitted normaliser is a different model | **code**: `normaliser.npz` in every bundle; `predict()` asserts the rebuilt normaliser matches |

**Guide diff.** Annotation blocks after the verify steps of **6.1**, **6.2** and **6.3**; three rows (**12–14**) added
to the known-deviations index. No other guide text changed.

---

## 3. The pipeline

One entry point, `ml/train/loop.py`, takes `(task, world, encoder, depth, rate, seed, gate, wsla)` and runs all eight
steps. The shipped configuration is **`ml/configs/shipped.json`**, not a branch in the code: arrival SHARE-lite h⁴ at
2.5e-4, fill h⁰ at 1.25e-4, capacity HeteroMP h⁴ at 2.5e-4, shortage HeteroMP h¹ at 1.25e-4 (diagnostic); staleness
gate and reconstructed feature **off** — the loop asserts both.

| step | what happens | saved | why it cannot be skipped |
|---|---|---|---|
| **1** | labels, fixed split, **no-leak assert**, device-side panel and graph, windows sliced as-of | — | the as-of property is asserted per run, not trusted from Phase 5 |
| **2** | AdamW, one step per snapshot, patience 8, cap 120, **restore best**; **every training row enters the loss, asserted each epoch** | `train_log.json` (losses, validation curve, stop reason, rows per epoch, warned nondeterministic ops, peak RSS) | — |
| **3** | restored-best weights | `checkpoint.pt`, `normaliser.npz` | Phase 5 saved no checkpoint; Addendum A had to retrain |
| **4** | validation- and test-fold predictions | `preds_val.npz`, `preds_test.npz`, `model_outputs.csv.gz` | every validation-fitted step needs validation predictions |
| **5** | recalibration **fitted on the validation fold**: arrival 13-cell moment matching; fill 22-cell MM or VS, selected on validation log score; capacity and shortage none | — | the uncalibrated fill head is 2–5× worse on calibration |
| **6** | its parameters | `recalibration.json` | refitted every run — the temperature differs across seeds |
| **7** | **label-free drift baseline**: mean predicted P(T > 12) (arrival), P(complete) (fill), P50 (capacity), P(short) (shortage) on validation inputs | `drift_baseline.json` | the monitor has nothing to compare against without it |
| **8** | test metrics, invariants (survival monotone, zero quantile crossings, fill partition), the `model_outputs` join, config and stamps | `metrics.json`, `join_check.json`, `config.json` (`complete: true`) | a bundle without `complete: true` is refused by `predict()` |

**A bundle is the unit.** `predict()` refuses anything without `complete: true`. A directory holding only
`checkpoint.pt` is not a model.

**Two ways in.** `train` runs all eight steps. `convert` runs steps 3–8 on a Phase 5 cell that already has a
restored-best checkpoint and validation predictions — it reloads the checkpoint, **recomputes the validation
predictions and asserts they match the saved ones with `allclose`**, then builds the bundle. Four Phase 5 cells enter
this way rather than being retrained.

---

## 4. Reproduction gate

The most important check in this phase: a refactored loop that silently changes numbers is worse than no refactor. The
new loop was run, from scratch, on five configurations Phase 5 had already measured. Bands are the per-metric seed bands
the brief fixes (arrival C-index 0.0004, fill CRPS 0.0001, capacity pinball 0.0004), plus Addendum A's ECE bands.

| id | configuration | metric | target | **achieved** | \|Δ\| | Δ / band | verdict | epochs (best) new vs Phase 5 | best validation new vs Phase 5 |
|---|---|---|---|---|---|---|---|---|---|
| R1 | arrival h⁰ v6 @ 2e-3 | C-index | 0.6602 | **0.66025** | 0.00005 | 0.12× | **PASS** | 34 (25) vs 35 (26) | 0.66787 vs 0.66775 |
| R2 | arrival SHARE-lite h⁴ v6 @ 2.5e-4 | C-index | 0.6731 | **0.67234** | 0.00076 | **1.91×** | **FAIL** | 80 (71) vs 79 (70) | 0.67968 vs 0.67949 |
| R3 | fill h⁰ v6 @ 1.25e-4 | exact CRPS | 0.0618 | **0.06185** | 0.00005 | 0.50× | **PASS** | 51 (42) vs 52 (43) | 0.05262 vs 0.05262 |
| R3 | ″ | ECE 20-bin, raw | 0.0448 | **0.04476** | 0.00004 | 0.01× (band 0.0073) | **PASS** | | |
| R3 | ″ | ECE 20-bin, recalibrated | 0.0203 | **0.02028** | 0.00002 | 0.01× (band 0.0021) | **PASS** | | |
| R4 | capacity h⁰ v6 @ 2.5e-4 | mean pinball | 0.0464 | **0.04643** | 0.00003 | 0.08× | **PASS** | 35 (26) vs 35 (26) | 0.04477 vs 0.04477 |
| R5 | capacity HeteroMP h⁴ v7 @ 2.5e-4 | mean pinball | 0.0707 | **0.07066** | 0.00004 | 0.09× | **PASS** | 74 (65) vs 81 (72) | 0.06467 vs 0.06466 |

**6 of 7 checks pass. R2 fails at 1.9× its band.** Every secondary figure moved with its primary: R3's recalibration
again selects vector scaling at T = 1.008 (Addendum A: 1.008); R4 reproduces coverage 0.8205 and Spearman 0.7589 exactly;
R5's coverage 0.7576 and Spearman 0.7473 sit beside Addendum B's 0.750 and 0.745; R1's lateness ROC-AUC is 0.762 against
0.7618.

### Finding the cause of R2 — before proceeding, as the brief requires

A failure is a loop bug until shown otherwise. Three measurements, in order.

**(1) Epoch-by-epoch validation curves against Phase 5's stored runs.**

| configuration | \|Δ val\| at epoch 0 | max over epochs 0–9 | max over the whole run | first epoch with \|Δ\| > 1e-6 |
|---|---|---|---|---|
| R4 capacity h⁰ | **0** | **0** | **0 — bitwise identical, all 35 epochs** | never |
| R3 fill h⁰ | 7.7e-10 | 1.9e-7 | 3.1e-6 | 27 |
| R5 capacity HeteroMP h⁴ | 3.7e-7 | 1.1e-5 | 2.3e-4 | 2 |
| R1 arrival h⁰ | 1.3e-6 | 8.4e-5 | 3.2e-4 | 0 |
| **R2 arrival SHARE-lite h⁴** | **2.3e-5** | **1.8e-4** | **5.0e-3** | **0** |

The loop's data path, loss, early stopping and scoring reproduce Phase 5 **bitwise** where the model itself is
deterministic (R4). Everywhere else the curves drift, and R2 starts furthest out at epoch 0 — 18× R1's gap and 60× R5's. That
pattern fits device non-determinism compounding through training — but it would also fit a loop difference that only
shows on the hazard + SHARE-lite path, so (2) tests it directly.

**(2) A controlled repeat, same configuration, same seed, same process** (R2's configuration, two epochs each;
`ml/artifacts/phase6_repeat_test.json`):

| run | \|Δ val\| vs harness run 1, epoch 0 | epoch 1 | vs Phase 5's stored curve, epoch 0 | epoch 1 |
|---|---|---|---|---|
| Phase 5 harness, run 1 | 0 | 0 | 2.5e-6 | 2.7e-5 |
| **Phase 5 harness, run 2 — identical code, identical seed** | **1.8e-5** | **1.0e-4** | 2.1e-5 | 1.3e-4 |
| loop, run 1 | 1.3e-6 | 1.1e-4 | 3.8e-6 | 1.4e-4 |
| loop, run 2 | 1.1e-5 | 1.2e-4 | 1.3e-5 | 1.4e-4 |
| harness, deterministic-algorithms flag left on by the loop | 2.0e-5 | 4.2e-5 | 2.3e-5 | 1.5e-5 |
| harness, flag reset off | 6.3e-7 | 1.5e-5 | 1.9e-6 | 1.3e-5 |

- **Phase 5's own code does not reproduce itself on this configuration.** Run twice with the same seed, it differs by
  1.8e-5 after one epoch — as large as R2's gap to the stored curve (2.3e-5).
- **The loop sits inside that spread** in both of its runs.
- **The one global setting the loop adds, `use_deterministic_algorithms(True, warn_only=True)`, moves nothing
  systematically**: with it on and with it off, the harness lands on both sides of its own repeat.

**(3) The other graph configuration passes.** R5 (HeteroMP h⁴, same non-deterministic scatter ops) reproduces within
0.09× its band while stopping 7 epochs earlier than Phase 5 did.

**Verdict on R2: not a loop bug.** The loop reproduces Phase 5's harness to within the harness's own same-seed spread.
R2's miss is **run-to-run non-determinism of the SHARE-lite h⁴ arrival path on MPS**, present in Phase 5's code and
compounding over 80 epochs from ~2e-5 at epoch 0 to 5e-3 on the validation curve. The 0.0004 band it was held to was
**borrowed from h⁰ cells at 2e-3**, whose own run-to-run noise is near zero, so it cannot contain that. **R2 stays
recorded as a FAIL against the band the brief set.** It is re-read against SHARE-lite h⁴'s own measured band in §6.

### R2 re-read against SHARE-lite h⁴'s own band

Stage 4 trained two more seeds of exactly R2's configuration. Test C-index, arrival SHARE-lite h⁴ v6 @ 2.5e-4:

| seed 7 (R2) | seed 17 | seed 27 | mean | **measured spread** | borrowed band |
|---|---|---|---|---|---|
| 0.67234 | 0.67363 | 0.67431 | 0.67343 | **0.00197** | 0.0004 |

- **The band R2 was held to was 4.9× too narrow for this configuration.** It came from h⁰ cells, whose run-to-run
  noise is near zero; SHARE-lite h⁴'s seed spread is 0.00197.
- **Against its own band R2 is 0.39× — inside.** Phase 5's 0.6731 lies within the three-seed range
  [0.6723, 0.6743].
- **R2 therefore stays recorded as a FAIL against the band the brief fixed, and is not a difference against the band
  measured for its configuration.** The loop reproduces Phase 5 on every configuration, within that configuration's
  own noise.

---

## 5. Re-verification — the loop rebuilds the data path, so every invariant is measured again

From `ml/train/phase6_verify.py` (output `ml/artifacts/phase6_verify.json`) and from each bundle's own asserts.

### Folds

| check | v6 | v7 |
|---|---|---|
| fixed split, no leak — **asserted** for arrival, fill, capacity, shortage | pass | pass |
| rows train / validation / test — arrival and fill | 176,000 / 32,000 / 36,000 | 176,000 / 32,000 / 36,000 |
| rows — capacity; shortage | 110,000 / 20,000 / 22,500; 66,000 / 12,000 / 13,500 | same |
| the eight rolling origins, max(train) < min(evaluate) — **asserted** | pass, all 8 (fold 8 evaluates on 2 snapshots) | pass, all 8 |
| covid training snapshots the exclusion would remove | 2020-03-09, 04-20, 06-01, 07-13, 08-24 — **5 of 44** | identical |

### Determinism on MPS

| check | result |
|---|---|
| `torch.use_deterministic_algorithms(True)` | **raises** — `index_put_with_accumulate_mps does not have a deterministic implementation` |
| same, `warn_only=True` | runs; the warned ops are recorded in every bundle's `train_log.json` |
| same seed → identical initial weights | **bitwise identical** |
| two identical forward passes, SHARE-lite h⁴ | differ by **3.0e-8** — not bit-deterministic |

### Causality, end to end — on the loop's inference path, all four shipped models

Every panel value, reporting lag and its observed flag **after t₀** was perturbed (+100, flags inverted). On MPS a graph
encoder's repeated forward already differs from itself, so the test demands the perturbed output stay **within the
model's own repeat-forward floor** — and bitwise identity where the model is deterministic:

| model | repeat-forward floor | future perturbation, max \|Δ\| | within floor | bitwise | last in-window week, max \|Δ\| |
|---|---|---|---|---|---|
| fill h⁰ | **0.0** | **0.0** | yes | **yes** | 0.914 |
| arrival SHARE-lite h⁴ | 3.0e-8 | 3.0e-8 | yes | no — device | 0.0100 |
| capacity HeteroMP h⁴ | 2.4e-7 | 2.4e-7 | yes | no — device | 0.602 |
| shortage HeteroMP h¹ | 1.2e-7 | 6.0e-8 | yes | no — device | 0.437 |

**Causal in all four.** Where the model is deterministic (h⁰), the future cannot move a single bit; where it is not, the
future moves the output no more than re-running the identical input does, while one week of in-window change moves it
by 10⁻² to 10⁰ — the test is sensitive.

### Fill partition

The 22-cell partition refines the legacy 20 bins **exactly on all 244,000 labels in each world**, re-checked on every
fill bundle's labels as well.

### Invariants asserted inside every bundle

Generated by `ml/eval/phase6_collect.py` from each bundle's `train_log.json`, `metrics.json` and `join_check.json`. A bundle is only written `complete` if every assert passed.

| bundle | trained by | rows entering the loss per epoch (training population) | survival violations | quantile crossings | fill partition | join: predictions / with actual / one-to-one | nondeterministic ops warned | peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| arrival_week/v6_lite_h4_lr0.00025_s17 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| arrival_week/v6_lite_h4_lr0.00025_s27 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 2.78 |
| arrival_week/v6_lite_h4_lr0.00025_s7 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| arrival_week/v6_none_h0_lr0.00025_s7 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| arrival_week/v6_none_h0_lr0.002_s7 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| arrival_week/v7_lite_h4_lr0.00025_s17 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| arrival_week/v7_lite_h4_lr0.00025_s27 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 2.78 |
| arrival_week/v7_lite_h4_lr0.00025_s7 | loop | [176000] (176,000) | 0 | — | — | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| arrival_week/v7_none_h0_lr0.00025_s7 | converted | — (converted) | 0 | — | — | 36,000 / 36,000 / True | — | 2.94 |
| capacity_strain/v6_mp_h4_lr0.00025_s17 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 2.78 |
| capacity_strain/v6_mp_h4_lr0.00025_s27 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 2.78 |
| capacity_strain/v6_mp_h4_lr0.00025_s7 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 2.78 |
| capacity_strain/v6_none_h0_lr0.00025_s7 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 2.78 |
| capacity_strain/v7_mp_h4_lr0.00025_s17 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 3.12 |
| capacity_strain/v7_mp_h4_lr0.00025_s27 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 2.78 |
| capacity_strain/v7_mp_h4_lr0.00025_s7 | loop | [110000] (110,000) | — | 0 | — | 22,500 / 22,500 / True | index_put_with_accumulate_mps | 2.78 |
| fill_rate/v6_none_h0_lr0.000125_s17 | converted | — (converted) | — | — | True | 36,000 / 36,000 / True | — | 2.77 |
| fill_rate/v6_none_h0_lr0.000125_s27 | converted | — (converted) | — | — | True | 36,000 / 36,000 / True | — | 2.77 |
| fill_rate/v6_none_h0_lr0.000125_s7 | loop | [176000] (176,000) | — | — | True | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 2.78 |
| fill_rate/v7_none_h0_lr0.000125_s17 | loop | [176000] (176,000) | — | — | True | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 3.12 |
| fill_rate/v7_none_h0_lr0.000125_s27 | loop | [176000] (176,000) | — | — | True | 36,000 / 36,000 / True | index_put_with_accumulate_mps | 2.78 |
| fill_rate/v7_none_h0_lr0.000125_s7 | converted | — (converted) | — | — | True | 36,000 / 36,000 / True | — | 2.94 |

---

## 6. Seed bands, measured

Addendum B's depth cells were one seed each, with bands borrowed from h⁰ on v6 — and, for arrival, from a different
learning rate. Stage 4 trained seeds 7 / 17 / 27 of every shipped configuration in both worlds. All figures are test 2025.

### Every band, generated from the bundles

| task | world | metric | seed 7 | seed 17 | seed 27 | mean | **spread** | sd | borrowed band used in Addendum B |
|---|---|---|---|---|---|---|---|---|---|
| arrival_week | v6 | cindex | 0.6723 | 0.6736 | 0.6743 | 0.6734 | **0.0020** | 0.0010 | 0.0004 |
| arrival_week | v6 | roc_auc_late | 0.7800 | 0.7830 | 0.7812 | 0.7814 | **0.0029** | 0.0015 | — |
| arrival_week | v6 | ece_week | 0.1369 | 0.1473 | 0.1231 | 0.1358 | **0.0242** | 0.0122 | — |
| arrival_week | v6 | ece_week (recal) | 0.0247 | 0.0221 | 0.0274 | 0.0247 | **0.0053** | 0.0026 | — |
| arrival_week | v7 | cindex | 0.6773 | 0.6775 | 0.6772 | 0.6773 | **0.0003** | 0.0001 | 0.0004 |
| arrival_week | v7 | roc_auc_late | 0.7748 | 0.7758 | 0.7741 | 0.7749 | **0.0017** | 0.0008 | — |
| arrival_week | v7 | ece_week | 0.2303 | 0.1984 | 0.1399 | 0.1895 | **0.0904** | 0.0458 | — |
| arrival_week | v7 | ece_week (recal) | 0.0564 | 0.0458 | 0.0543 | 0.0521 | **0.0106** | 0.0056 | — |
| capacity_strain | v6 | pinball_mean | 0.0477 | 0.0496 | 0.0511 | 0.0495 | **0.0034** | 0.0017 | 0.0004 |
| capacity_strain | v6 | coverage80 | 0.7852 | 0.7624 | 0.7435 | 0.7637 | **0.0418** | 0.0209 | — |
| capacity_strain | v6 | spearman_p50 | 0.7765 | 0.7834 | 0.7832 | 0.7810 | **0.0069** | 0.0039 | — |
| capacity_strain | v7 | pinball_mean | 0.0707 | 0.0712 | 0.0720 | 0.0713 | **0.0013** | 0.0007 | 0.0004 |
| capacity_strain | v7 | coverage80 | 0.7576 | 0.7477 | 0.7587 | 0.7547 | **0.0110** | 0.0061 | — |
| capacity_strain | v7 | spearman_p50 | 0.7473 | 0.7508 | 0.7500 | 0.7494 | **0.0035** | 0.0018 | — |
| fill_rate | v6 | crps_exact | 0.0618 | 0.0618 | 0.0619 | 0.0618 | **0.0001** | 0.0000 | 0.0001 |
| fill_rate | v6 | ece20 | 0.0448 | 0.0520 | 0.0486 | 0.0484 | **0.0073** | 0.0036 | 0.0073 |
| fill_rate | v6 | ece20 (recal) | 0.0203 | 0.0224 | 0.0211 | 0.0212 | **0.0021** | 0.0011 | — |
| fill_rate | v6 | rel_one (recal) | 0.0085 | 0.0108 | 0.0096 | 0.0097 | **0.0022** | 0.0011 | — |
| fill_rate | v7 | crps_exact | 0.1025 | 0.1027 | 0.1028 | 0.1027 | **0.0003** | 0.0002 | 0.0001 |
| fill_rate | v7 | ece20 | 0.0739 | 0.0876 | 0.0855 | 0.0824 | **0.0137** | 0.0074 | 0.0073 |
| fill_rate | v7 | ece20 (recal) | 0.0143 | 0.0150 | 0.0137 | 0.0143 | **0.0013** | 0.0007 | — |
| fill_rate | v7 | rel_one (recal) | 0.0119 | 0.0115 | 0.0105 | 0.0113 | **0.0014** | 0.0007 | — |

The task-by-task reading follows.

### Arrival — SHARE-lite h⁴ @ 2.5e-4

| world | metric | seed 7 | seed 17 | seed 27 | **spread** | vs borrowed 0.0004 |
|---|---|---|---|---|---|---|
| v6 | C-index | 0.67234 | 0.67363 | 0.67431 | **0.00197** | **4.9× wider** |
| v6 | lateness ROC-AUC | 0.7800 | 0.7830 | 0.7812 | 0.0029 | |
| v6 | week-ECE raw → recalibrated | 0.1369 → 0.0247 | 0.1473 → 0.0221 | 0.1231 → 0.0274 | 0.0242 → **0.0053** | |
| v7 | C-index | 0.67731 | 0.67746 | 0.67719 | **0.00026** | **0.7× — narrower** |
| v7 | lateness ROC-AUC | 0.7748 | 0.7758 | 0.7741 | 0.0017 | |
| v7 | week-ECE raw → recalibrated | 0.2303 → 0.0564 | 0.1984 → 0.0458 | 0.1399 → 0.0543 | 0.0904 → **0.0106** | |

**One band per task does not describe both worlds.** The same configuration's C-index spread is **7.6× larger on v6 than
on v7**. A band borrowed across worlds is wrong in both directions: too narrow for v6, too wide for v7.

**Addendum B's arrival verdicts hold under their own bands**:

| world | Addendum B margin, SHARE-lite h⁴ over h⁰ | vs borrowed 0.0004 | **vs own band** | h⁰ below every h⁴ seed? | verdict moves? |
|---|---|---|---|---|---|
| v6 | +0.0127 | 32× | **6.4×** (three-seed mean over h⁰: +0.0129, 6.6×) | yes | **no** |
| v7 | +0.0154 | 39× | **58×** (three-seed mean: +0.0147, 56×) | yes | **no — strengthens** |

**v7's damaged distribution is not a one-seed accident.** After recalibration, week-ECE is **0.046–0.056 on all three
v7 seeds**, against h⁰'s 0.0146. The fallback of §7 is needed for every seed, not just the one Addendum B trained. On v6
the recalibrated figure is 0.022–0.027, against h⁰'s 0.0166.

**Recalibration also shrinks the seed band.** Raw week-ECE spreads 0.0242 (v6) and 0.0904 (v7) across seeds;
recalibrated, 0.0053 and 0.0106. This matches Addendum A's finding on fill: the per-seed variation is largely per-cell
bias, and bias is what recalibration removes.

### Capacity v6 — the configuration the brief said to watch

Addendum B: validation chose HeteroMP h⁴, test preferred h⁰ by 0.0013 — 3.3× a borrowed 0.0004 band. The brief asked
whether a wider real band would dissolve the disagreement.

| capacity v6, mean pinball ↓ | seed 7 | seed 17 | seed 27 | **spread** |
|---|---|---|---|---|
| HeteroMP h⁴ (Phase 6 bundles) — test | 0.04771 | 0.04959 | 0.05108 | **0.00337 — 8.4× the borrowed band** |
| HeteroMP h⁴ — 80% coverage | 0.7852 | 0.7624 | 0.7435 | 0.0418 |
| HeteroMP h⁴ — Spearman P50 | 0.7765 | 0.7834 | 0.7832 | 0.0069 |

**Part one — against its own band, Addendum B's single-seed margin dissolves.** Seed 7, the cell Addendum B compared,
is HeteroMP h⁴'s **best** seed. Its +0.00128 over h⁰ is **0.4× the measured band**.

**Part two — the disagreement is nonetheless real, and larger than Addendum B stated.** Phase 5 had already trained h⁰ at
the same three seeds. Putting all six cells side by side:

| capacity v6, mean pinball ↓ | h⁰ (Phase 5, seeds 7/17/27) | HeteroMP h⁴ (seeds 7/17/27) | ranges overlap? | better, by (means) |
|---|---|---|---|---|
| **validation 2024** | 0.04470 – 0.04508 | **0.04175 – 0.04183** | **no** — gap 0.0029 | **HeteroMP h⁴**, 0.0031 |
| **test 2025** | **0.04633 – 0.04672** | 0.04771 – 0.05108 | **no** — gap 0.0010 | **h⁰**, 0.0030 |

**Every seed of each configuration falls on the far side of every seed of the other, on both folds, in opposite
directions.** This is not seed noise. It is the 2025 utilisation drift Addendum B diagnosed, now shown to hold across
initialisations, and the gap on test is **0.0030, not 0.0013**: Addendum B's single cell understated it by more than
half. The label-free monitor sees the same thing (§7): under the 2025 inputs, h⁰'s P50 rises +4.19 points with the drift
while HeteroMP h⁴'s falls −2.78.

**The shipped configuration is not changed.** Selection is on validation, and validation still separates the two
cleanly in HeteroMP h⁴'s favour. Switching to h⁰ would be selection on test, which the rules forbid. This is recorded as
**the strongest open risk in the shipped configuration** (§8). It is exactly what Phase 8.2's rolling origins exist to
decide: the first check that asks a depth to win across more than one validation year.

### Capacity v7 — validation and test agree

| capacity v7, HeteroMP h⁴ | seed 7 | seed 17 | seed 27 | **spread** |
|---|---|---|---|---|
| test mean pinball ↓ | 0.07066 | 0.07119 | 0.07199 | **0.00132 — 3.3× the borrowed band** |
| validation mean pinball ↓ | 0.06467 | 0.06518 | 0.06530 | 0.00063 |
| 80% coverage | 0.7576 | 0.7477 | 0.7587 | 0.0110 |
| Spearman P50 | 0.7473 | 0.7508 | 0.7500 | 0.0035 |

**Addendum B's v7 verdict holds.** HeteroMP h⁴ over h⁰ was +0.0058 — **4.4× its own band** (three-seed mean over h⁰:
+0.0053, 4.0×). **Every seed beats h⁰ on both folds**: on test the worst h⁴ seed scores 0.0720 against h⁰'s 0.0765; on
validation, 0.0653 against 0.0741. On v7 there is no disagreement to explain. Caveat: h⁰ v7 is a single Phase 5 cell, so
the h⁰ side of this comparison has no band of its own.

**Coverage is below nominal on every seed**, 0.748–0.759 against 80% — the 2025 under-coverage §8 records, now shown to
hold across initialisations.

### Fill — h⁰ @ 1.25e-4, point-mass head with validation recalibration

| world | metric | seed 7 | seed 17 | seed 27 | **spread** | vs the band Phases 5 used |
|---|---|---|---|---|---|---|
| v6 | exact CRPS | 0.06185 | 0.06180 | 0.06188 | **0.00008** | 0.8× the 0.0001 CRPS band |
| v6 | ECE 20-bin, raw → recalibrated | 0.0448 → 0.0203 | 0.0520 → 0.0224 | 0.0486 → 0.0211 | 0.0073 → **0.0021** | identical to Addendum A's |
| v6 | reliability P(f = 1), recalibrated | 0.0085 | 0.0108 | 0.0096 | 0.0022 | |
| v6 | recalibration selected on validation, temperature | vs, 1.008 | vs, 1.089 | vs, 1.030 | | |

**The v6 fill band is, by construction, Addendum A's**: seed 7 was retrained by the loop and reproduced exactly (§4), and
seeds 17 and 27 were converted from Addendum A's own checkpoints. It is the one band here that was already measured,
and the loop confirms it.

**Addendum B's fill verdict holds.** h⁰ over validation's HeteroMP h⁴ on test CRPS, 0.0618 against 0.0622: a margin of
0.0004, **4.8× its own band**. Fill stays at h⁰.

**Addendum A's "parity with LightGBM on v6" is borderline, not settled.** Recalibrated ECE against LightGBM-22 put
through the same protocol (0.0187):

| comparison | margin | vs own band 0.0021 |
|---|---|---|
| seed 7 — the figure Addendum A quoted | +0.0016 | 0.76× — inside |
| **three-seed mean, 0.0212** | **+0.0025** | **1.2× — just outside** |

LightGBM's 0.0187 is itself a single fit, and Phase 5 measured LightGBM's ECE refit noise at about ±0.003, which is
larger than the gap. Whether the head matches LightGBM on v6 **cannot be decided with one LightGBM fit**. Addendum A's
parity claim is downgraded to *not distinguishable*, and noted in §8.

| world | metric | seed 7 | seed 17 | seed 27 | **spread** | |
|---|---|---|---|---|---|---|
| v7 | exact CRPS | 0.10253 | 0.10273 | 0.10284 | **0.00031** | |
| v7 | ECE 20-bin, raw → recalibrated | 0.0739 → 0.0143 | 0.0876 → 0.0150 | 0.0855 → 0.0137 | 0.0137 → **0.0013** | |
| v7 | reliability P(f = 1), recalibrated | 0.0119 | 0.0115 | 0.0105 | 0.0014 | |
| v7 | recalibration selected on validation, temperature | mm, 1.000 | vs, 1.048 | vs, 1.024 | | |

**Addendum B's fill v7 verdict holds.** h⁰ over validation's HeteroMP h⁴, 0.1025 against 0.1038: **4.2× its own band**,
and every h⁰ seed scores below 0.1038.

**Addendum A's "closed on v7" holds across seeds.** Recalibrated ECE is **0.0137–0.0150, mean 0.0143**, against
LightGBM's 0.0141 as published and 0.0163 refitted, and far below LightGBM-22 put through the same protocol (0.0271).
The spread after recalibration is 0.0013.

**The recalibration method is chosen per retraining, and it does change.** Validation log score picked moment matching
on seed 7 and vector scaling on seeds 17 and 27. This is why recalibration is refitted inside the pipeline instead of
carried forward as a fixed transform.

---

## 7. The drift monitor

### How it is computed

The statistic is **label-free** — it is computed from a model's predictions alone, so it can run on production inputs
that have no outcomes yet:

| task | statistic (`drift_statistic` in the config) | why this one |
|---|---|---|
| arrival | mean predicted **P(T > 12)**, raw hazard output | Addendum B: its drift ranked the post-recalibration week-ECE of all nine arrival cells |
| fill | mean predicted **P(complete)**, raw | Addendum B: +4.5 to +5.4pp on the v7 graph cells whose calibration collapsed |
| capacity | mean predicted **P50** | no Phase 5 threshold; recorded so the baseline exists |
| shortage | mean predicted **P(short)** | diagnostic only |

### Where it is stored

Step 7 computes the statistic on the **validation fold's inputs** at training time and writes `drift_baseline.json`
into the bundle: the statistic's name, its value (raw and recalibrated), the row count and the validation snapshots.
It is refitted with every training run, like the recalibration.

### How `predict()` emits it

`loop.predict(bundle, fold, h0_bundle=None)` loads a **complete** bundle, rebuilds the model, **asserts the rebuilt
normaliser matches the bundle's**, applies the bundle's recalibration, and returns — alongside the predictions:

- **per batch** (one snapshot of incoming inputs): the statistic, the baseline, and `drift_pp` = (statistic − baseline) × 100;
- **overall**, row-weighted across batches — the figure decisions are taken on, because Addendum B's thresholds were
  measured on whole-fold means, not single snapshots;
- with an h⁰ reference bundle, the same for h⁰ on the **same inputs**, and **`excess_pp = |drift − h⁰ drift|`** — the
  quantity Addendum B's thresholds are defined on;
- a **status** from `drift_bands_pp` in `ml/configs/shipped.json`:

| excess drift beyond h⁰ | status | evidence |
|---|---|---|
| ≤ 0.5 pp | `usable` | Addendum B: recalibrated week-ECE 0.015–0.022 |
| 0.5 – 1.0 pp | `watch` | **interpolated — not measured** |
| 1.0 – 4.5 pp | `degraded` | 1.5–2.0 pp measured at 0.043–0.054 |
| > 4.5 pp | `unusable` | 4.5–5.2 pp measured at 0.105–0.124 |

A model whose shipped configuration *is* h⁰ (fill) has no reference to exceed; its raw drift is reported with status
`no h0 reference`, because h⁰'s own drift largely tracks genuine label shift (Addendum B: −0.7 / −0.9 pp against an
observed fall in complete fills).

### The arrival fallback — implemented, not prose

`shipped.json → tasks.arrival_week.fallback.threshold_pp = 1.0`. When `predict()` is given arrival's h⁰ bundle and the
overall excess drift exceeds it, **the returned `distribution` — the one the Monte Carlo consumes — is h⁰'s recalibrated
13-cell distribution**, while `ranking_score` stays SHARE-lite h⁴'s expected week. `overall.distribution_source` says
which was served.

**Smoke evidence, non-engaged path** (1-epoch SHARE-lite h⁴ v7 against the converted h⁰ v7 bundle): drift +0.20 pp,
h⁰ drift +0.27 pp, excess 0.07 pp → `usable`, distribution served from the model. The engaged path is exercised on the
real bundles below.

### Measured on the real bundles — 2025 test inputs, no labels read on the prediction path

| task | world | model bundle (seed 7) | h⁰ reference | drift pp | h⁰ drift pp | excess pp | status | distribution served |
|---|---|---|---|---|---|---|---|---|
| arrival | v6 | SHARE-lite h⁴ | h⁰ @ 2.5e-4 (Stage 5) | **−0.58** | +0.35 | **0.93** | `watch` | model — below the 1.0 pp threshold |
| arrival | v7 | SHARE-lite h⁴ | h⁰ @ 2.5e-4 (converted, Addendum B) | **−2.11** | +0.27 | **2.38** | `degraded` | **h⁰ — fallback engaged** |
| fill | v6 | h⁰ | — (is h⁰) | **−0.69** | — | — | `no h0 reference` | model |
| fill | v7 | h⁰ | — (is h⁰) | **−0.89** | — | — | `no h0 reference` | model |
| capacity | v6 | HeteroMP h⁴ | h⁰ @ 2.5e-4 (R4) | **−2.78** | **+4.19** | 6.96 | `no calibrated threshold` | — |
| capacity | v7 | HeteroMP h⁴ | — (no v7 h⁰ bundle) | −1.66 | — | — | `no h0 reference` | — |

**The loop's monitor reproduces Phase 5's drift figures.** Fill h⁰ drifts −0.69 / −0.89 pp against Addendum B's −0.7 /
−0.9. SHARE-lite h⁴ v6 drifts −0.58 pp against Addendum B's −0.5.

**A bug found and fixed on the way: the status bands were first applied to capacity.** They labelled capacity v6
`unusable` at 6.96 "pp". The bands were measured on arrival's P(T > 12), a probability. Capacity's statistic is mean P50
*utilisation*, so its "pp" are hundredths of a utilisation unit, and arrival's thresholds do not transfer to it. The
config now carries `drift_bands_apply`, true for arrival only; every other task reports its drift and excess with status
`no calibrated threshold`.

**What capacity's drift does show is Addendum B's mechanism, detected without labels.** Under the identical 2025 inputs,
**h⁰'s P50 rises +4.19 points — matching the observed +4.2-point rise in utilisation** (Addendum B: 0.654 → 0.696) —
while **HeteroMP h⁴'s P50 falls −2.78 points, against it.** This is the anchoring Addendum B blamed for the v6 test
disagreement, now visible from predictions alone, and on exactly the configuration where test preferred h⁰. It is not a
calibrated alarm, but it is the right signal.

### The fallback engages on its own on v7

On v7 no override was needed. SHARE-lite h⁴'s predicted P(T > 12) moves **−2.11 pp** under the 2025 inputs while h⁰'s
moves +0.27 pp; the excess of **2.38 pp** crosses the 1.0 pp threshold, and `predict()` serves h⁰'s distribution
(`ml/artifacts/phase6_drift_arrival_v7.json`):

| arrival v7 | week-ECE on 2025 test | source |
|---|---|---|
| **distribution actually served** | **0.0146** | h⁰, recalibrated (`allclose` to h⁰'s bundle) |
| what would have been served without the switch | 0.0564 | SHARE-lite h⁴, recalibrated |
| ranking — unchanged | C-index 0.6773 | SHARE-lite h⁴ |

**The switch buys a 3.9× better-calibrated distribution for the Monte Carlo at no cost to ranking**, and it does so on
exactly the world where Addendum B measured the damage (drift −2.0 pp, recalibrated week-ECE 0.0537 — reproduced here as
−2.11 pp and 0.0564). Per-snapshot excess on v7 ran 0.08–5.52 pp, seven of nine snapshots above 1 pp.

### The fallback switch, verified in both states

`ml/artifacts/phase6_fallback_test.json` — the real v6 bundles (measured excess 0.93 pp), run once at the shipped
threshold and once with the threshold forced to 0.5 pp:

| | shipped threshold 1.0 pp | forced threshold 0.5 pp |
|---|---|---|
| `distribution_source` | **model** | **h0 (fallback engaged)** |
| served distribution equals | SHARE-lite h⁴'s recalibrated distribution (`allclose`) | **h⁰'s recalibrated distribution (`allclose`)** |
| week-ECE of the served distribution | 0.0247 | **0.0166** |
| ranking score | SHARE-lite h⁴'s expected week (`allclose` to its bundle) | **unchanged — still SHARE-lite h⁴** |

The switch does what the config says, in both directions: it changes the distribution the Monte Carlo consumes and
leaves the ranking alone.

### Is the switch's decision stable across retrainings?

The same `predict()` call, shipped threshold, on all three seeds of the shipped arrival configuration in both worlds
(`ml/artifacts/phase6_fallback_seeds.json`):

| world | seed | drift pp | excess pp | status | served | week-ECE served | week-ECE if not switched |
|---|---|---|---|---|---|---|---|
| v6 | 7 | −0.58 | 0.93 | `watch` | model | 0.0247 | 0.0247 |
| v6 | 17 | −0.57 | 0.92 | `watch` | model | 0.0221 | 0.0221 |
| v6 | 27 | −0.75 | **1.10** | `degraded` | **h⁰ — engaged** | **0.0166** | 0.0274 |
| v7 | 7 | −2.11 | 2.38 | `degraded` | h⁰ — engaged | **0.0146** | 0.0564 |
| v7 | 17 | −1.59 | 1.86 | `degraded` | h⁰ — engaged | **0.0146** | 0.0458 |
| v7 | 27 | −2.13 | 2.41 | `degraded` | h⁰ — engaged | **0.0146** | 0.0543 |

- **On v7 the decision is stable and matters.** All three seeds engage, and the switch serves a distribution
  **3.1–3.9× better calibrated** than the model's own.
- **On v6 the decision is a coin flip, and matters little.** Excess drift across seeds spans **0.92–1.10 pp, straddling
  the 1.0 pp threshold**, so one retraining in three flips the switch. The stake is small: the model's own recalibrated
  distribution is 0.022–0.027 there, against h⁰'s 0.0166.
- **h⁰'s distribution was at least as well calibrated in all six cells.** That is a test-set observation and cannot move
  the configuration — but it means Phase 8's backtest should weigh a plain "always serve h⁰'s distribution" policy
  against the drift gate.

The 1.0 pp threshold is **not changed**: it comes from Addendum B and nothing calibrated says where else to put it. It
sits inside v6's seed-to-seed spread of excess drift, which is recorded in §8 as needing a hysteresis band or a
multi-year calibration.

**Why decisions use the overall figure, not a batch.** Per-snapshot excess on these nine 2025 snapshots ran **0.16,
1.29, 1.04, 0.67, 1.38, 0.81, 0.74, 2.40, 0.23 pp** — four of nine above 1 pp on their own — while the row-weighted
overall is 0.93. A per-batch rule would flap the Monte Carlo's input between models snapshot to snapshot. `predict()`
therefore emits every batch's figure for monitoring and **switches on the overall one**, which is the scale Addendum B's
thresholds were measured at.

---

## 8. Still open, with mechanisms

### Closed in this phase

**Arrival h⁰ v6 at 2.5e-4 now has validation predictions and a recalibrated week-ECE** (Addendum B item 5). Retrained
under the loop: test C-index **0.6605** (Addendum B 0.6604), week-ECE raw **0.0606** (0.0594) → recalibrated
**0.0166**. Arrival h⁰'s recalibrated week-ECE therefore now exists in both worlds — **0.0166 (v6), 0.0146 (v7)** — and
the v6 bundle is the h⁰ reference the arrival drift monitor needs. The cell trained 100 epochs against Addendum B's 82:
the same run-to-run drift of the arrival path measured in §4, with the test score unmoved.

**The loop runs shortage** (a one-epoch HeteroMP h¹ bundle is complete and joins 13,500 of 13,500 rows), so the
diagnostic has a pipeline when Phase 9 unblocks. Its deviations are listed below, still open.

### Still open

**Capacity v6: validation and test disagree on every seed — the strongest open risk in the shipped configuration.**
Across three seeds each, HeteroMP h⁴ beats h⁰ on validation by 0.0031 and loses on test by 0.0030; neither pair of
ranges overlaps, in opposite directions (§6). Mechanism, measured twice: 2025 utilisation rises ≈ 0.04, and the graph
anchors each channel to its supplier's history, so its P50 moves −2.78 points against the drift while h⁰'s moves +4.19
with it (§7). The configuration is not changed, because the only evidence against HeteroMP h⁴ is test data. The
decision belongs to Phase 8.2's rolling origins, which ask a depth to win across several validation years.

**Arrival's fallback threshold sits inside its own seed noise on v6.** Excess drift across three v6 seeds spans
0.92–1.10 pp around a 1.0 pp threshold, so the switch engages for one retraining in three (§7). On v7 it engages for all
three. The threshold came from one Phase 5 cell and has never been calibrated. Two fixes are available: a hysteresis
band (engage above 1.25, release below 0.75), or calibrating the threshold against week-ECE across rolling origins. h⁰'s
distribution was at least as well calibrated in all six cells tested, which is worth checking in the backtest as a plain
policy.

**Addendum A's fill parity with LightGBM on v6 is not distinguishable, rather than established.** Addendum A quoted
seed 7's recalibrated ECE, 0.0203, against LightGBM-22's same-protocol 0.0187: 0.76× the band. Across three seeds the
mean is 0.0212, which is 1.2× the band. LightGBM's figure comes from one fit with about ±0.003 refit noise, so the
comparison can be settled only with several LightGBM fits under the same protocol. That is cheap, a few minutes on CPU,
but it belongs to Phase 7's baselines.

**Seed bands are per world, not per task.** Arrival SHARE-lite h⁴'s C-index spread is 0.00197 on v6 and 0.00026 on v7,
a 7.6× difference for one configuration. Every future margin must be read against the band of its own configuration in
its own world, and a band borrowed across worlds is wrong in both directions.

**Covid is not excluded from training.** Specification §9.4 says to; Phases 2–6 did not. It removes 5 of 44 training
snapshots, which changes the population behind every number the reproduction gate checks — so it could not be applied
and reproduced at once. `folds.covid_mask` is implemented and off. Measuring it is one extra seed triple per task.

**The rolling origins are defined, not run.** All eight are implemented and leak-asserted. Every Phase 6 band still
comes from one validation year and one test year. This is the Phase 8.2 backtest, and Addendum B showed it is where
graph depth choices are most at risk: validation and test disagreed whenever 2025 drifted.

**Three seeds on one split, not ≥ 5 seeds × 8 folds.** The bands in §6 contain model-initialisation noise only. They
have no year-to-year component, so they are a floor on the real uncertainty.

**Bit-for-bit reproduction is unattainable on MPS.** Strict deterministic algorithms raise, and identical forward passes
differ by up to 3e-8 (2.4e-7 for HeteroMP h⁴). Reproduction is `allclose` plus the seed band. On CPU the question stays
open and has not been measured.

**Bundles are stamped `+dirty`** because the Phase 6 code was uncommitted when they were built. The stamp is honest, but
a bundle can only be tied to an exact code state once the commit exists.

**The drift thresholds come from arrival only.** The `watch` band is interpolated. Fill's evidence is Addendum B's h¹
and h⁴ cells, and capacity has no threshold at all. Calibrating the thresholds per task, over several validation years,
needs the rolling origins.

**Capacity's 2025 bands under-cover for every model** (Addendum B: P90 exceeded on 17–25% of rows). The loop ships
capacity without recalibration, as configured, so it inherits this. A validation-fitted correction would not fix it,
because 2024 does not show the drift.

**Shortage's two closeout deviations remain open**: the sweep optimum on its bottom boundary, and both h⁰ cells capped
at 119/120. No compute was spent on them. The loop runs shortage end to end: a one-epoch HeteroMP h¹ bundle is
complete, its join is one-to-one on 13,500 rows, and every output row is labelled *DIAGNOSTIC — not the shortage product
path*. Phase 9 remains blocked.

---

## 9. Gate

| requirement | result |
|---|---|
| Stage 1 — 6.1–6.3 read and reconciled with Phase 5's three findings, as a table; guide diff | **pass** — 13 rows, each resolved in code or by annotation (§2) |
| 2a — one config-driven entry point; the shipped configuration is a file | **pass** — `loop.py` + `shipped.json`; gate and reconstructed feature asserted off |
| 2b — the eight-step pipeline; a bundle is the unit | **pass** — 22 complete bundles; `predict()` refuses incomplete ones |
| 2c — inference path emits drift with every batch; arrival fallback implemented | **pass** — per-batch and overall drift; switch verified forced (§7) and engaging on its own on all three v7 seeds |
| 2d — seeding recorded; reproduction by `allclose`, never `equal` | **pass with a stated deviation** — strict deterministic algorithms raise on MPS; `warn_only`, ops recorded per bundle |
| 2e — split, fit window, all 16,072 channels, inductive | **pass** — no-leak asserted per run |
| **Stage 3 — reproduction gate** | **6 of 7 checks pass.** R2 (arrival SHARE-lite h⁴) **FAILS against its given band** (1.9×). Cause found, **not a loop bug**: Phase 5's code does not reproduce itself on that configuration; R2 is **0.39× its own measured band** (§4) |
| re-verification — causality, hazard rows and survival, quantile crossings, fill partition | **pass** — all four shipped models causal; asserted inside every bundle (§5) |
| **Stage 4 — seed triples, both worlds** | **pass — complete, nothing cut.** Bands measured per task per world (§6) |
| Addendum B verdicts re-read against their own bands | arrival v6 / v7, capacity v7, fill v6 / v7 **hold**; **capacity v6's disagreement is real on every seed and ≈ 2× as large** (0.0030, not 0.0013); Addendum A's v6 fill parity downgraded to **not distinguishable** |
| Stage 5 — arrival h⁰ v6 validation predictions; shortage runnable | **pass** — recalibrated week-ECE 0.0166; shortage bundle complete, deviations still open |
| budget and cut order | **pass** — projection printed before Stage 4; no cut needed |
| files that must not change | **pass** — see header |
| `inventory_position_weekly` untouched | **pass** |

### May Phase 7 start?

**Yes.** The loop reproduces Phase 5 within every configuration's own noise, produces shippable bundles, and monitors
drift with a working fallback. Three things Phase 7 inherits:

1. **Commit the Phase 6 code before building bundles that must be traceable.** Every bundle here is stamped `+dirty`.
2. **Fit LightGBM baselines several times under the same recalibration protocol**, so the fill-parity question in §6
   can be answered rather than left "not distinguishable".
3. **Capacity v6's shipped depth rests on validation alone, against a real and seed-stable test disagreement.** Phase 8.2's
   rolling origins, not Phase 7, should decide it — but no one should quote capacity v6 2025 bands before then.
