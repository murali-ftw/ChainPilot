# Phase 13 — Stage 0 gate outcomes

**Measured on:** v8 seed 1001, fixed split. Representative t0 = **2025-01-06** (the first 2025 test snapshot),
closed-line lookback 104 weeks. `ml/eval/phase13_stage0.py` → `ml/artifacts/phase13_stage0.json`, commit `1fc1d29`,
clean. **No test or arm is cancelled by any Stage 0 gate.** Pre-registration: `phase-13-preregistration.md`
(`1cc38db`), committed before this ran.

## Preflight

| item | outcome |
|---|---|
| **P1** constant substitution (deviation 72) | **Already fixed in Phase 11C** (`ce58df5`). The brief's premise is stale: `phase7_score.py`'s `arrival_promise` branch scores each arm's own prediction, and the constant appears only as `roc_auc_late_CONSTANT_RANKER`. The outcome test (`test_no_lateness_substitution.py`) still passes. **Added** `ml/tests/test_no_substitution_path.py`, which checks the *code*: an AST check that no constant-fed value is written to `roc_auc_late`. It passes, and it **fires** when the pre-11C line is reinserted |
| **P2** identity | `config_name` / `identity_of` now carry fill head, ratio key, estimator, shrinkage and ratio use. `score_name()` names score files from the full identity, including calibration state (raw\|recal, with anything else refused). `marker_name()` names completion markers from the full bundle path. The tests pass (6/6), including a new Phase 13 axis test |
| **P3** clean tree + SHA | `require_clean()` refuses to run on a dirty `ml/`; every artifact carries `code_commit` / `code_dirty` |
| **P4** pre-registration | committed at `1cc38db`, before any measurement |

## Gates

### 0.1 — schema: **PASS (F2 can be built), but not via the field the brief named**

- **(a) A channel is exactly (part, supplier, plant).** 16,072 channels = 16,072 unique triples; `site_id` is 1:1
  with supplier.
- **(b) `purchase_orders.status` cannot identify anything.** It takes **one value, `OPEN`, on all 1,063,256 POs.**
  Two other fields identify a zero-delivery closure, **as-of**:

| signal (recorded ≤ t0) | lines raised before 2025-06 with **no GRN ever** (19,721) | lines **with** a GRN |
|---|---|---|
| `supplier_acknowledgements.ack_status = 'rejected'` or `po_line_revisions` qty → 0 | **19,696** (99.87%) | **0** |
| neither signal | 25 | — |
| recording lag after the line is raised | ack: median 5 d, p95 24 d; qty→0: median 2 d, p95 23 d | |

**"Closed, nothing arrived" := a rejection or a qty→0 revision recorded ≤ t0, with no GRN recorded ≤ t0.** There are
zero false positives. This **contradicts Phase 12 deviation 100** (`reports/part2/phase-12.md`, deviation row 100,
"an old unreceived line is indistinguishable, as-of, from a pending one"). That claim was made after looking only at
GRN and PO status. Recorded as a new deviation in `phase-13.md`.

### 0.2 — unified baseline table: pinned (full table in `phase-13.md` §2)

| quoted figure | what it actually is | pinned |
|---|---|---|
| **0.018** "b5flat22 interior Σ\|err\|, TRAIN marginal" | **mislabelled**: b5flat22's RAW interior error on **VALIDATION** (Phase 12 A2). Its training marginal was never stored | val raw **0.01816 [0.01803, 0.01832]**; test raw 0.02515 [0.02467, 0.02568] |
| **0.063–0.092** "the head" | the head's RAW interior error on **VALIDATION** | val raw **0.07685 [0.06284, 0.09210]**; train raw 0.07286 [0.05724, 0.08420]; test raw 0.08781 [0.07639, 0.10390] |
| **0.13827** exact CRPS | the head's **RECALIBRATED TEST** CRPS; `phase-0-1-v8.md:311` gave it a 3-seed band | **0.13826 [0.13820, 0.13834]** at 5 seeds; raw test 0.13876 [0.13866, 0.13882] |
| **0.0165** ECE | rolling-52 histogram, RAW, **TEST** ECE-22, **single deterministic run** | test 0.01651; **validation 0.07765.** The same arm is 4.7× worse on validation |

- **Invalid-criterion flag (GATE 0.2):** any F1 pass criterion stated against the "0.018 train" figure is **invalid**
  (wrong split and wrong label). F1 is judged against the head's **same-split** raw band.
- **The rolling-52 histogram has no seeds.** Its figures are single-run, and no band may be claimed for it.
  Comparisons against it use a row-bootstrap interval, labelled as such (deviation in `phase-13.md`).
- **Recalibrated VALIDATION interior error and ECE are ≈ 0 for every arm** (0.000013–0.000025). That is the
  deviation-80 trap made visible: the recalibrator is fitted on that fold. They are never used for selection.

### 0.3 — S1 arithmetic: **PASS → S1 and S2 proceed**

| quantity (2025 store, part-plant-weeks) | value |
|---|---|
| observed shortfall when short (SS − on hand) | 91.4 |
| **brief's sum: 91 + mean net transfer-in over the observed-short weeks** | **178.3** |
| pre-rescue shortfall when short (on hand − that week's net transfer) | 149.6 |
| window, within 2× of 298 | [149, 596] |

Both versions fall inside the window (the pre-rescue one only just). **My pre-registered prediction 7 (gate closes)
is wrong.** Pre-rescue below-SS frequency is **10.95%**, against 8.38% observed.

**Stated in advance for S1's falsification F-b (wrong sign), before S1 runs.** Subtracting transfers moves the
reference to about 8.38 − (10.95 − 8.38) ≈ **5.8%** below SS. The ratio then moves **away** from 1.0, to about
12.36 / 5.8 ≈ **2.1×** (the 5-seed simulated mean is 12.36%). F-b passes if its ratio is ≥ 1.8×.

### 0.4 — b5flat22 scoreability: **SCOREABLE**

Stored raw and recalibrated predictions exist for 5 fits on val and test, and they score through the same
`phase5_metrics` path as the head. It is **not servable** (deviation 46), and no loader is built. b5flat22 stays in
F1 as arm 2, **scored from stored predictions, not retrained.**

### 0.5 — variance decomposition: **plant is ALIVE after supplier, but tiny → no key dropped**

182,442 closed lines as-of t0 (2.25% at fill 0, 80.1% at fill 1). Additive R² was fitted by backfitting. The noise
floor for each added term permutes that factor **within** the conditioning factor's groups (200 permutations,
p95).

| term | ΔR² | noise floor p95 | verdict |
|---|---|---|---|
| **plant after supplier** | **0.000175** | 0.000074 | **above floor (2.4×)** |
| supplier × plant interaction | 0.0183 | 0.0141 | above floor |
| triple after (part, supplier) cells | 0.0032 | 0.0026 | above floor |
| part after supplier | 0.0054 | — | |
| supplier after part | 0.0253 | — | |
| (part × supplier) interaction over additive | 0.1015 | not computed | **in-sample; cell means over ~12 lines each over-fit** |

**GATE 0.5 does not fire.** Plant's marginal share is real but minute (0.02% of variance). (part, supplier, plant)
and (supplier, plant) stay as F2 keys. Supplier carries almost all of the main-effect variance (2.6%). The large
in-sample R² of fine cells (0.13) is an over-fit upper bound; F2 is the out-of-sample test.

### 0.6 — cell counts at t0: **raw triple KEPT**

| key | cells with history | closed lines per cell, median / p10 / p25 | scoring rows with ≥ 1 / 3 / 5 / 10 / 20 / 50 |
|---|---|---|---|
| (part, supplier, plant) | 15,061 | 12 / 6 / 9 | 99.8% / 98.9% / 96.7% / **76.7%** / 9.8% / 0% |
| (part, supplier) | 14,661 | 12 / 6 / 9 | 99.8% / 99.0% / 96.9% / 78.0% / 13.3% / 0% |
| (supplier, plant) | 2,921 | 60 / 26 / 41 | 100% / 100% / 99.98% / 99.9% / 98.5% / 80.9% |
| supplier | 420 | 434 / 346 / 383 | 100% at every n |

76.7% ≥ 50%, so **raw (part, supplier, plant) stays as arm K3.** Note: (part, supplier) is almost the same key as
the triple (14,661 vs 15,061 cells), because a part-supplier pair almost always serves one plant. **Cold start is
nearly absent on v8:** 0.25% of scoring rows (10 of 4,000) have no triple history at t0, so any cold-start
breakdown will rest on a handful of rows and will be reported with its n.

## Arms carried forward

- **F1:** all six arms. Arm 2 (b5flat22) is scored from stored predictions. Arm 5 is the **RPS boundary reweight**,
  with its weight selected on raw validation.
- **F2:** keys K1 (part, supplier), K2 (supplier, plant), K3 raw triple, and K4 the shrunk hierarchy; each with
  arms (a) and (c), and ratio_of_sums vs mean_of_ratios.
- **S1, S2:** proceed.
- **2A:** proceeds if F1 arm 4 trains.
