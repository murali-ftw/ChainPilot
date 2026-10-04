# Phase 19 — forward load re-tested, the timing claim tested, the neural models taught

**Audience:** whoever decides what the arrival, fill and capacity products ship next, and which data gaps block them.
**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, val 2024, test 2025), seeds 7 / 17 / 27 / 37 / 47. TEST
unless marked; RAW throughout. Thresholds and blend weights fitted on validation only.
**Companions:** `phase-19-preregistration.md` (`4d533b4`); `phase19/stage1_gate.md`; `phase19/stage2_timing.md`;
`phase19/stage3_neural.md`; `phase19/stage4_intervals.md`; Phase 18: `reports/phase18.md`.
**Branch / base:** `phase19`, from `phase18` at `62bb607` (Phase 18 content `1212dd3`). Not merged.
**Status:** complete. All 15 neural cells finished at 00:07 IST, inside the 10-hour window (20:26 → 06:26), so there is
no STOPPED file.

## Header

| | |
|---|---|
| machine | Apple M4 Pro, 14 cores, 24 GB, macOS (Darwin 27.0): the machine that trained every stored v8 bundle |
| software | Python 3.14, torch 2.14.0 (**MPS**, float32), LightGBM 4.7.0, NumPy / pandas / scikit-learn from `venv/` |
| concurrency level | **1 neural cell at a time** (recorded in every Phase 19 bundle). LightGBM fits ran on the CPU before the neural queue. Stage 4a's CPU bootstrap ran beside the first capacity cell, affecting wall-clock only (deviation 176) |
| handshake | **inside** the stored band (\|Δ val C-index\| ≤ 4.3e-5 over 3 epochs); stored incumbents reused, none retrained |
| dirty-tree rule | every run refused a dirty `ml/` (`phase12_common.require_clean`); every artifact records `code_dirty = false` |

| artifact | commit |
|---|---|
| pre-registration | `4d533b4` |
| fwd_season / cadence features; proxy fits (BASE bit-exact); PRIVILEGED timing fits | `da6fee1` |
| Gate v2 scores (`gate_v2.json`) | first run `8599179`; five-seed rule `e203a02` |
| timing scores (`PRIVILEGED__timing_scores.json`) | `528f9a1`, wording `f5b2f46` |
| neural cells (15) | `b9b1060` |
| neural scores (`neural_score.json`) | `11f0c93` |
| Stage 4a / 4b block bootstrap | `7c9ad7a` / `11f0c93` |
| isolation audit (+ `test_artifact_identity.py` 6 / 6) | run at `7e64261`, all PASS |

**Anchors reproduced first:** BASE re-fitted bit-exactly on every task. Phase 18's fwd_load arrays rebuilt equal, and its
stored scores re-scored equal. fwd_season equals Phase 18's network-mean diagnostic exactly. The incumbents' bands
equal Phase 18's (arrival lateness 0.7090 [0.7064, 0.7130], fill CRPS 0.13876, capacity precision @ 5% 0.851).

---

## 1. Decisions

| use case × question | verdict | the number | band / interval |
|---|---|---|---|
| arrival × fwd_season (Gate v2) | **PASS** | lateness AUC 0.7067 vs BASE 0.7053; A3 13.07 vs 13.25 d; permuted 0.7020 | [0.7062, 0.7071] vs [0.7048, 0.7057] |
| arrival × fwd_load, supplier-specific | **PASS** | 0.7186 vs fwd_season 0.7067; permuted 0.7039 (not > BASE) | [0.7183, 0.7191] |
| arrival × cadence | **PASS** | 0.7142 vs 0.7053; shuffled 0.7050 | [0.7141, 0.7144] |
| fill × fwd_season | **PASS** | CRPS 0.1369 vs 0.1397; AUC 0.641 vs 0.605; permuted no gain | [0.1368, 0.1369] |
| fill × fwd_load, supplier-specific | **FAIL** | beats fwd_season, but the **snapshot-permuted** fwd_load also beats BASE (CRPS 0.1387, AUC 0.617): a static descriptor, not a forecast | [0.1383, 0.1391] vs [0.1396, 0.1398] |
| fill × cadence | **PASS** | CRPS 0.1395 vs 0.1397; AUC 0.607 vs 0.605 (small) | [0.1395, 0.1396] |
| capacity × fwd_season | **FAIL** | worse at 1% coverage (0.717 vs 0.780) | [0.698, 0.747] vs [0.751, 0.804] |
| **capacity × fwd_load, supplier-specific** | **PASS** (as coded: FAIL; deviation 171) | 6 / 6 points above fwd_season; permuted not above BASE on any comparable point | precision @ 5% 0.813 [0.802, 0.828] vs 0.758 [0.742, 0.766] |
| capacity × cadence | **FAIL** | worse at 10% coverage | |
| **arrival × order-timing forecast** | **not worth building from as-of cadence** | true creation week alone **+0.205** lateness AUC (0.910); cadence recovers **4%** | [0.9102, 0.9105] |
| **fill × neural + season + cadence** | **GAIN** | CRPS **0.1371** vs 0.13876; P(full) AUC **0.643** vs 0.620 | [0.1366, 0.1374]; [0.6424, 0.6440] |
| arrival × neural + fwd_load + season + cadence | **TIE** | lateness 0.7135 vs 0.7090; A3 12.91 vs 13.06; C-index disjointly better (0.6776 vs 0.6744) | [0.7111, 0.7185] |
| capacity × neural + fwd_load | **TIE** | precision @ 5% 0.886 vs 0.851; recall @ 0.70 0.506 vs 0.515 (seed 7 stopped at epoch 14) | [0.829, 0.908] vs [0.832, 0.860] |
| **arrival × Phase 19 blend** (block bootstrap) | **better than the incumbent ensemble** | lateness **0.731** vs 0.714; A3 **12.34** vs 13.07 d | +0.017 [0.015, 0.020]; −0.74 d [0.54, 0.97] |
| **fill × Phase 19 blend** (block) | **better than the incumbent ensemble** on CRPS, AUC, ECE | CRPS 0.1354, AUC 0.664, ECE 0.060 | CRPS +0.0032 [0.0023, 0.0041] |
| capacity × Phase 19 ensemble (block) | better at 10% coverage only | 0.867 | +0.052 [0.009, 0.098] |
| Phase 18 arrival blend (block re-test) | **survives** | +0.0084 to +0.0136 lateness AUC | |
| Phase 18 fill CRPS comparisons between families (block) | **do not survive** (now undetermined) | | |

## 2. The timing test (Stage 2)

| arm | lateness AUC | A3 (days) |
|---|---|---|
| BASE | 0.7053 | 13.25 |
| BASE + cadence (as-of) | 0.7142 | 13.08 |
| **PRIVILEGED BASE + true creation week, no state** | **0.9104** | **6.42** |
| Phase 18 PRIVILEGED: creation week + state | 0.9229 | 5.63 |
| Phase 18 PRIVILEGED: state, no creation week | 0.7092 | 13.24 |

**Phase 18's claim stands: the arrival headroom is order timing.** Timing alone reaches 94% of the timing + state oracle,
and state alone reaches almost nothing. But the order-raising time is not visible in as-of PO history: cadence recovers
4%. The pre-registered rule says **not worth building**. The brief's fallback sentence ("future supplier state") is
contradicted by the measurement and is replaced by the measured one (deviation 172).

## 3. Prediction scorecard

| # | prediction | **right / wrong** | the number |
|---|---|---|---|
| **P1** | fwd_season beats BASE on fill and capacity (disjoint), and its snapshot-permuted arm does not | **RIGHT** | fill CRPS / AUC disjoint; capacity better at 5%, 10%, R 0.70, R 0.80. Permuted arms better nowhere. (Capacity still FAILs its gate: worse at 1%) |
| **P2** | supplier-specific fwd_load beats fwd_season on all three use cases | **RIGHT** | arrival lateness AUC; fill CRPS and AUC; capacity 6 / 6 (fill's gate fails on its control, not on this comparison) |
| **P3** | snapshot-permuted fwd_load is not disjointly better than BASE on any use case | **WRONG** | fill: CRPS 0.1387 [0.1383, 0.1391] vs 0.1397 [0.1396, 0.1398]; AUC 0.617 vs 0.605 |
| **P4** | true creation week alone adds < 0.02 lateness AUC | **WRONG** | **+0.205** (0.9104 vs 0.7053) |
| **P5** | cadence adds < 0.01 lateness AUC | **RIGHT** | +0.0089 (0.7142 vs 0.7053; disjoint) |
| **P6** | neural + fwd_load beats the incumbent on fill and on capacity recall @ 0.70, and ties on arrival | **WRONG** | arrival TIE (right); capacity recall @ 0.70 0.506 vs 0.515, a TIE (wrong); fill's bound arm carries season + cadence, not fwd_load (its fwd_load failed Gate v2), so the fill clause is NOT TESTABLE as worded. The season + cadence arm GAINs |
| **P7** | the block bootstrap widens the Phase 18 intervals, but the arrival blend's lateness gain still excludes 0 | **RIGHT** | 48 / 53 intervals wider (median 1.9×); blend gain [+0.0084, +0.0136]. Five intervals, all on arrival, are 0.94–0.99× as wide, including that one (0.94×) |
| **P8** | a blend of neural + fwd_load with LightGBM + fwd_load beats the neural + fwd_load ensemble on arrival only | **WRONG** | it beats on arrival (+0.014 lateness) **and on fill** (CRPS, AUC, ECE); capacity identical (w = 1.00) |

**4 right, 4 wrong.**

## 4. Deviations, continuing from 171

| # | prior statement | measured / done | where |
|---|---|---|---|
| **171** | pre-registration: "bands are [min, mean, max] over the 5 seeds"; scorer "Phase 18's, unchanged" | Phase 18's `band()` silently dropped seeds whose recall-at-p was unreachable on validation, so one- and three-seed values were compared as bands. Found **after** the first gate run because a verdict hinged on it. The Phase 19 scorer requires all five seeds (Phase 15's UNREACHABLE rule); the as-coded verdict is kept beside it. **One verdict changes: capacity fwd_load supplier-specific FAIL → PASS.** Phase 18's verdicts are unaffected (each failed on a point all seeds reach) | stage 1 |
| **172** | brief, Stage 2: otherwise record "arrival headroom is future supplier state, not reachable from as-of data" | the timing-only arm shows the headroom **is** timing (0.910 vs state-only 0.709), so that sentence is false here. The rule's outcome, not worth building, is kept, and the sentence replaced by the measured one | stage 2 |
| **173** | constraint 2: new axis "omitted-at-default in the identity" | `ml/artifact_identity.py` is protected, so the axis `row_family` lives in a new wrapper (`ml/train/phase19_identity.py`, suffix `_rf…` only when set) with a separate bundle root `ml/artifacts/phase19/bundles/`. The audit shows every stored config is named identically and a set axis is distinct | stage 3, §6 |
| **174** | Stage 4a: "redo Phase 18's seed-ensemble comparisons" | capacity's ensemble is scored from the mean of the quantiles (as the blend is built), where Phase 18 used the mean of per-seed P(strain > 1). Capacity ensemble rows therefore compare slightly different predictors, and Phase 18's ±0.001 capacity blend-vs-ensemble differences vanish by construction | stage 4 |
| **175** | P6 / P8: "neural + fwd_load" | fill's bound arm carries fwd_season + cadence, because fill's supplier-specific fwd_load failed Gate v2. Stage 4b's LightGBM side is LightGBM + fwd_load for all three use cases, as P8 states | stages 3–4 |
| **176** | "concurrency 1" | one neural cell at a time throughout. Stage 4a (CPU bootstrap, ~3 min) ran during the first capacity cell, which affects wall-clock only (that cell ran 15.0 s/epoch, as the smoke run did) | header |
| **177** | Stage 1: "fit only the NEW arms" | BASE + fwd_load and the hindsight load are Phase 18's stored predictions (`94cfdc5` / `9ce7007`), re-scored and asserted equal, not refitted. BASE was re-fitted for seed 7 only, as the reproduction gate | stage 1 |

## 5. Every `inventory_position_weekly` read

**None.** No Phase 19 module reads it (search of every Phase 19 file: the only hit is `cadence.py`'s docstring saying
it is never read). The PRIVILEGED timing arm reads `_sim.npz`'s `pt`, `pa`, `pq` and `pd` arrays only. Phase 18's
listed reads (an identity check during its generator re-run) were not repeated.

## 6. Isolation audit

`ml/tests/test_phase19_isolation.py` → `ml/artifacts/phase19/isolation_audit.json`, run at `7e64261`, every check PASS:

| check | result |
|---|---|
| `git diff 62bb607 HEAD` and the working tree on db/gen_v6, gen_v7, gen_v8, db/validator.py, docs/specs/**, db/dataset_structure.md, ml/configs/shipped.json, results/**, reports/part1, **every report that existed at the base**, ml/models/share.py, ml/models/tcn.py, ml/artifact_identity.py | **empty**. `git diff 62bb607 HEAD -- ml/models/share.py ml/models/tcn.py` is empty |
| stored configs under `artifact_identity` at the base and now | **371 identical**; 340 in the directory `bundle_name` gives; **31 cannot be named by this base** (Phase 16 encoder variants on the unmerged `phase16-encoders` branch; deviation 149) |
| `phase19_identity` on every stored config | same name as `artifact_identity` (axis omitted at default); a set `row_family` gives a distinct name under the Phase 19 root; all 15 Phase 19 bundles sit at the name the wrapper gives them |
| AST scan of `ml/train`, `ml/models`, `ml/data`, `ml/baselines` | **0 violations**, with Phase 18's forbidden strings extended to Phase 19's (`PRIVILEGED__`, `phase19/PRIVILEGED`, `phase19/preds`). Self-test: Phase 18's three constructed offenders **and** three Phase 19 ones are flagged; a docstring mention is not. Fresh interpreters importing `fwd_season`, `cadence`, `phase19_proxy` (torch-free) and `phase19_bind`, `phase19_identity` load nothing from `reports/`. Pre-existing `_sim.npz` reader `ml/baselines/learnability_windowed.py` (not Phase 19) listed |
| feature names | no Phase 19 column carries `PRIVILEGED__` |

## 7. What this changes for the product

- **Arrival: ship the Phase 19 blend and the expected-week interval.** The blend (0.52 × neural with forward load,
  forward season and cadence; 0.48 × LightGBM with forward load) lifts lateness AUC from the incumbent ensemble's 0.714
  to **0.731** and cuts the median error from 13.07 to **12.34 days**, under the honest (snapshot-block) bootstrap. The
  remaining headroom is **when the planner raises the order**, worth +0.2 AUC, and the PO history cannot see it.
  **Blocked on data:** a signal of the reorder trigger (inventory position against the reorder point). It is the one
  input that would move arrival, and it is excluded by standing rule. Whether arrival should be re-scoped to lines
  already raised (counted from creation) is a product decision.
- **Fill: the first disjoint gain over a stored incumbent in Phases 16–19, from information, not architecture.** Season + cadence lifts the
  neural head disjointly (CRPS 0.1371, P(full) AUC 0.643). The blend with LightGBM + forward load is better again
  (0.1354 / 0.664) and better than the incumbent ensemble on CRPS, AUC **and** calibration. LightGBM + forward load alone
  remains best calibrated (ECE 0.033). **Recorded for a decision, not applied:** `shipped.json` ships LightGBM
  `b5flat22`, is protected here, and its reversal condition (Phase 10) should be re-read against these numbers.
- **Capacity: forward load is real but the neural model already had most of it.** Supplier-specific forward load
  passes the fair gate on the LightGBM proxy (6 / 6 points), but in the neural head it **ties** the incumbent. Its
  5-seed ensemble is better only at 10% coverage. **Blocked on data:** the monthly capacity draw (Phase 18), which no
  as-of source tracks.
- **What is closed:** supplier-specific forward load for fill (it acts as a static descriptor, not a forecast);
  forward season and cadence for capacity; the order-timing forecast from PO cadence; Phase 18's fill CRPS claims between
  model families (they do not survive the block bootstrap).
