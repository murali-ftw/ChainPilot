# Phase 13 — fill-head reparameterisation (F1, F2) and simulation re-reference (S1, S2)

**Audience:** whoever owns the client-facing claims, and whoever decides the next fill and simulation work.
**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, validation 2024, test 2025) unless stated. Selection on
validation only. No learning rate retuned.
**Companions:** `phase-13-preregistration.md` (`1cc38db`, committed before any measurement),
`phase-13-stage0.md` (`97720e6`), `phase-12.md`, `results/observation1.md`.

---

## 1. Verdicts, and what the gates cancelled

**No Stage 0 gate cancelled any test or arm** (§2.2). **Two 12-hour caps fired** at 10:22 on 2026-09-29
(`ml/artifacts/phase13_STOPPED_f1a.json`, `phase13_STOPPED_f2.json`). They cut:

- **the regression control to 3 of 5 seeds**, which also hit the 120-epoch ceiling on every seed, so it is a floor;
- **F2 key (part, supplier) to 4 of 5 seeds**;
- **F2 keys (supplier, plant) and the raw triple to 0 of 5.**

No arm below 5 seeds is quotable (deviation 67), and each such row says so.

| test | verdict | the number that decided it |
|---|---|---|
| **F1** — does a smooth interior fix the fill head? | **NO ARM PASSES. Every head arm is UNDETERMINED against the head.** The Beta head fixes [0.95, 1) and moves the error to the low-fill cells, with ROC-AUC disjointly **degraded**. The cheap boundary fix is the best head arm (the only one with a disjoint ROC-AUC gain), but its interior gain is not disjoint | test interior Σ\|err\|: head 0.0878 [0.0764, 0.1039]; boundary 0.0820 [0.0614, 0.1007]; Beta 0.0932 [0.0906, 0.1011]. **The rolling-52 histogram scores 0.0100 and b5flat22 0.0252: 3.5–9× better than any head arm** |
| **F2** — history ratios | **FAIL.** Arm (a), the ratio alone, cannot compete on CRPS. Arm (c), the ratio as a head input, is indistinguishable from the head without it. The ECE guard is degraded against the histogram | (c) shrunk hierarchy, test CRPS 0.13884 [0.13858, 0.13902] vs head 0.13876 [0.13866, 0.13882] (overlap); ECE-22 0.132 vs histogram 0.0165 |
| **2A** — shrunk ratio as the Beta centre | **FAIL.** Disjointly worse than the head on CRPS and ROC-AUC, and [0.95, 1) is worse | CRPS 0.13904 [0.13900, 0.13908]; ROC-AUC 0.6053 vs 0.6204; [0.95, 1) at 3.4–3.8× its frequency |
| **S1** — the simulation as a pre-rescue forecast | **PASS**, with every falsification behaving | ratio **1.129× [1.121, 1.139]** against the PRIVILEGED pre-rescue reference; shuffled transfers give 1.38–1.40×, never in band |
| **S2** — transfer recommendations | **Reproduces planners' DONOR choice better than naive and than random, at every θ**; not better at choosing recipients | donor agreement at matched k: policy **0.556** vs naive 0.527 vs random 0.520 |

---

## 2. Stage 0 and the unified baseline table

### 2.1 Preflight

- **P1 — already fixed; the premise is stale** (deviation 109). The constant substitution behind deviation 72 was
  removed in Phase 11C (`ce58df5`). `phase7_score.py:93` now scores each arm's own prediction. `:104` emits the
  constant only as `roc_auc_late_CONSTANT_RANKER`. `:111` (`kind_of`) and `:334` (the backtest path) pass the
  constant only to that ranker. Added `ml/tests/test_no_substitution_path.py`: an AST check that no constant-fed
  value is written to `roc_auc_late`. It **passes**, and **fires** when the pre-11C line is reinserted.
- **P2.** Every Phase 13 axis enters `config_name` / `identity_of`: fill head, ratio key, estimator, shrinkage and
  use. **So does `fill_loss`, which was missing** (deviation 110): a boundary-reweighted run would have shared the
  plain head's bundle path. `score_name` covers calibration state (raw|recal only) and every axis; `marker_name`
  keys completion markers on the full bundle path. Tests: 6/6.
- **P3.** `require_clean()`. Every Phase 13 artifact and bundle is stamped clean; the scorer refuses a `+dirty`
  bundle.
- **P4.** Pre-registration committed at `1cc38db`, before any measurement.

### 2.2 Gates (full detail in `phase-13-stage0.md`)

| gate | measured | outcome |
|---|---|---|
| **0.1** channel / zero-delivery closure | channel = (part, supplier, plant), 16,072 = 16,072. `purchase_orders.status` has **one value (OPEN)**. But **19,696 of 19,721** never-received lines carry an as-of supplier rejection or qty→0 revision, and **0** received lines do (median recording lag 2–5 days) | **PASS** via acknowledgements/revisions. **Contradicts** `reports/part2/phase-12.md:939` (deviation 100); deviation 111 |
| **0.2** baseline table | §2.3; four loose numbers pinned | two invalid-criterion flags (deviations 112, 114) |
| **0.3** S1 arithmetic | 91 + net transfer in the short weeks = **178**; pre-rescue 150; window [149, 596] | **PASS** |
| **0.4** b5flat22 scoreable | stored raw + RECAL predictions, 5 fits, val + test | **scoreable**, not servable; no loader built |
| **0.5** plant after supplier | ΔR² **0.000175** against a within-supplier permutation floor of 0.000074 | **alive** (2.4× floor), though 0.02% of variance: no key dropped |
| **0.6** raw-triple coverage | **76.7%** of scoring rows have ≥ 10 closed lines | raw triple kept |

### 2.3 The unified baseline table, verbatim (v8, fixed split, `phase13_stage0.json`, `1fc1d29`)

Seed-banded arms are shown as `mean [min, max]` over 5 seeds. The histogram is **single-run, no band**. Training-split
predictions for the LightGBM arms and the histogram were never stored.

| arm | split | calib | n rows | interior Σ\|err\| | exact CRPS | ECE-22 | ROC-AUC P(fill=1) |
|---|---|---|---|---|---|---|---|
| 22-cell head (h⁰) | train | raw | 176,000 | 0.0729 [0.0572, 0.0842] | 0.11994 [0.11963, 0.12015] | 0.0826 [0.0636, 0.0949] | 0.6574 [0.6557, 0.6605] |
| 22-cell head | train | recal | 176,000 | 0.0145 [0.0141, 0.0153] | 0.12014 [0.11982, 0.12033] | 0.0242 [0.0224, 0.0260] | 0.6574 [0.6557, 0.6606] |
| 22-cell head | val | raw | 32,000 | 0.0768 [0.0628, 0.0921] | 0.12667 [0.12652, 0.12682] | 0.0868 [0.0697, 0.1043] | 0.6344 [0.6322, 0.6356] |
| 22-cell head | val | recal | 32,000 | 0.00001 (**≈0 by construction**) | 0.12663 [0.12647, 0.12677] | 0.00002 (**≈0 by construction**) | 0.6344 [0.6321, 0.6355] |
| 22-cell head | test | raw | 36,000 | 0.0878 [0.0764, 0.1039] | 0.13876 [0.13866, 0.13882] | 0.1235 [0.1103, 0.1414] | 0.6204 [0.6200, 0.6211] |
| 22-cell head | test | recal | 36,000 | 0.0279 [0.0262, 0.0290] | **0.13826 [0.13820, 0.13834]** | 0.0549 [0.0518, 0.0576] | 0.6205 [0.6198, 0.6212] |
| b5flat22 | val | raw | 32,000 | **0.01816 [0.01803, 0.01832]** | 0.12851 [0.12845, 0.12855] | 0.0359 [0.0353, 0.0366] | 0.6078 [0.6070, 0.6081] |
| b5flat22 | val | recal | 32,000 | ≈0 | 0.12842 [0.12838, 0.12845] | ≈0 | 0.6078 |
| b5flat22 | test | raw | 36,000 | 0.02515 [0.02467, 0.02568] | 0.13969 [0.13962, 0.13976] | 0.0518 [0.0510, 0.0527] | 0.6049 [0.6047, 0.6053] |
| b5flat22 | test | recal | 36,000 | 0.01750 [0.01741, 0.01761] | 0.13943 [0.13939, 0.13947] | 0.0314 [0.0311, 0.0317] | 0.6049 |
| lgbm22_id | val / test | raw | | 0.0179 / 0.0254 | 0.12865 / 0.13968 | 0.0351 / 0.0524 | 0.6053 / 0.6060 |
| lgbm22_id | test | recal | 36,000 | 0.0177 | **0.13942** | 0.0323 | 0.6060 |
| rolling-52 as-of histogram | val | raw | 32,000 | 0.0372 | 0.14673 | **0.0776** | 0.5845 |
| rolling-52 as-of histogram | test | raw | 36,000 | 0.0100 | 0.15825 | **0.0165** | 0.6054 |
| every LightGBM arm, histogram | train | — | 176,000 | not stored (a refit is training) | | | |

**The four loose numbers, pinned:**
- **0.018** is b5flat22's **validation** interior error, not "train marginal". Its training marginal was never
  measured (deviation 112).
- **0.063–0.092** is the head's raw **validation** band.
- **0.13827** is the head's **recalibrated test** CRPS: **0.13826 [0.13820, 0.13834]** at five seeds.
  `phase-0-1-v8.md:311` quoted it with a 3-seed band.
- **0.0165** is the histogram's **test** ECE-22 from a **single** run. The same arm scores **0.0776 on validation**
  (deviation 114).

---

## 3. Per test

### 3.1 F1 — fill head parameterisation

**Arms:**

1. the 22-cell head;
2. b5flat22 (scored from stored predictions, not retrained);
3. plain regression control;
4. atom + atom + Beta interior, trained with the **same** RPS on the same 22 cells, so only the parameterisation
   changes;
5. **the RPS boundary term between [0.95, 1) and the atom at 1.0 up-weighted**, chosen over a widened atom
   because it leaves the target and the scoring partition unchanged. The weight was selected **once on raw
   validation** interior error from {3, 10, 30} at seed 7, giving **w = 3** (0.0804, against 0.0857 / 0.0973 for
   10 / 30 and the head's 0.0870). Frozen before any test number (`phase13_f1_bw_selection.json`);
6. the rolling-52 as-of histogram.

**TABLE A — RAW. This table decides.** `mean [min, max]`, 5 seeds unless marked.

| arm | split | exact CRPS | interior Σ\|err\| | ROC-AUC P(fill=1) | vs head (test) |
|---|---|---|---|---|---|
| 1 head | val | 0.12667 [0.12652, 0.12682] | 0.0768 [0.0628, 0.0921] | 0.6344 [0.6322, 0.6356] | — |
| 1 head | test | 0.13876 [0.13866, 0.13882] | 0.0878 [0.0764, 0.1039] | 0.6204 [0.6200, 0.6211] | — |
| 2 b5flat22 | test | 0.13969 [0.13962, 0.13976] | **0.0252** [0.0247, 0.0257] | 0.6049 [0.6047, 0.6053] | interior disjointly better; CRPS and ROC disjointly worse |
| 3 regression **(3 seeds, cap-cut, epoch-ceiling floor)** | test | 0.2465 [0.2425, 0.2543] | 1.142 | 0.5959 [0.5590, 0.6151] | **FAILS**, not quotable at 3 seeds |
| 4 Beta interior | val | 0.12669 [0.12654, 0.12696] | 0.0849 [0.0804, 0.0930] | 0.6338 [0.6302, 0.6366] | |
| 4 Beta interior | test | 0.13877 [0.13864, 0.13894] | 0.0932 [0.0906, 0.1011] | **0.6188 [0.6177, 0.6195]** | **UNDETERMINED**: interior and CRPS overlap; ROC **disjointly degraded** |
| 5 boundary w=3 | val | 0.12671 [0.12653, 0.12686] | 0.0709 [0.0526, 0.0879] | 0.6352 [0.6324, 0.6370] | |
| 5 boundary w=3 | test | 0.13882 [0.13873, 0.13892] | 0.0820 [0.0614, 0.1007] | **0.6222 [0.6216, 0.6226]** | **UNDETERMINED**: interior and CRPS overlap; ROC **disjointly better** |
| 6 histogram (bootstrap) | test | 0.15825 [0.15581, 0.16039] | **0.0100** [0.0102, 0.0181] | 0.6054 [0.5989, 0.6122] | interior far better; CRPS far worse |

**TABLE B — RECALIBRATED** (own recalibrator, fitted on validation). **For completeness only.** Validation is ≈0 by
construction and is omitted.

| arm | test CRPS | test interior | test ECE-22 | test ROC-AUC |
|---|---|---|---|---|
| 1 head | 0.13825 [0.13820, 0.13834] | 0.0279 [0.0262, 0.0290] | 0.0549 | 0.6205 |
| 2 b5flat22 | 0.13943 [0.13939, 0.13947] | 0.0175 | 0.0314 | 0.6049 |
| 3 regression | **N/A**: a point forecast cannot be recalibrated | | | |
| 4 Beta | 0.13822 [0.13815, 0.13834] | 0.0269 [0.0246, 0.0300] | 0.0531 | 0.6189 |
| 5 boundary | 0.13824 [0.13817, 0.13832] | 0.0278 [0.0266, 0.0295] | 0.0548 | 0.6222 |
| 6 histogram | **N/A**: stored raw only | | | |

**Diagnostics — where the error sits (test, raw).** Top-4 interior cells are shown for seed 7; the share and the
[0.95, 1) ratio are per seed.

| arm | top-4 cells | share of interior error | [0.95, 1) predicted / observed | atom at 1.0 pred / obs |
|---|---|---|---|---|
| 1 head | 3, 2, 1, **20** | 0.47–0.62 | **1.85–3.61×** | 0.778 / 0.749 |
| 4 Beta | **1, 2, 3, 4** | 0.68–0.80 | 0.14–1.22× (erratic) | 0.779 / 0.749 |
| 5 boundary | 2, 5, 4, 3 | 0.47–0.70 | **0.98–1.90×** | 0.789 / 0.749 |
| 2 b5flat22 | 3, 4, 2, 5 | 0.62 | 0.91× | 0.775 / 0.749 |
| 6 histogram | 4, 8, 3, 2 | 0.56 | 0.97× | 0.754 / 0.749 |

**Reading.**
- **Neither fix removes the failure; each moves it.** The boundary fix removes most of the [0.95, 1) excess.
  The Beta head removes it erratically (from 0.14× to 1.22× by seed) and pushes the dominant error **into the
  low-fill cells [0, 0.2)**, which then carry 68–80% of its interior error.
- **The same four low cells dominate every arm, the baselines included.** The low-fill region is where the
  interior error lives once [0.95, 1) is corrected.
- **The control loses by a mile** (CRPS 0.246 against 0.139), so **the metric works**.
- **The histogram beats every head arm on interior error, 3.5–9×. The whole head programme is in question on
  this metric**, as the brief anticipated. The head's advantage is CRPS (0.1388 against 0.1583), and b5flat22
  sits between. **No head arm reaches b5flat22's raw interior error; recalibration brings them near it** (Table B,
  0.027 against 0.018).

**Verdict: F1 FAILS its pass criterion** (interior disjointly below the head's same-split band). Arms 4 and 5 are
**UNDETERMINED** against the head. Arm 5 is the only arm with a disjoint gain anywhere (ROC-AUC +0.002), which
supports pre-registration 1. **The Beta build is not needed**: the cheap boundary arm matches or beats it on every
metric.

### 3.2 F2 — history-ratio features

**Build.** `ml/data/fill_history.py`. Closed lines are taken as-of: a final GRN recorded ≤ t0, or a zero-delivery
closure (rejection or qty→0 recorded ≤ t0, no GRN). Estimators: ratio_of_sums and mean_of_ratios per key. K4 is the
Beta-binomial shrunk hierarchy: triple → (part, supplier) → supplier → global. κ is fitted per tier by method of
moments, with **units as trials**:

| tier | κ at t0 = 2025-01-06 |
|---|---|
| supplier | 28.7 |
| (part, supplier) | 7.3 |
| triple | 741 |

This is **a shrinkage device, not a correctly specified likelihood**: fill has atoms at 0 and 1, and units are
bundled in lines. It is validated empirically below.

**As-of gates, both shown firing:**

| gate | result |
|---|---|
| **G1**: counts built from lines recorded after t0 | **FIRES** ("rows recorded 2026-04-01 (> t0)") |
| **G2**: μ and κ fitted on full history | **FIRES** ("PRIOR … rows recorded 2027-03-20 (> t0)"). **A first version did not fire.** The leaked prior was built from an empty set, and the assertion accepted an undated prior. The assertion now requires a dated, non-empty prior (deviation 115) |
| empty history | **FIRES** |

**Arm (a) — the ratio alone, as a point forecast (deterministic; row-bootstrap 95%, test):**

| key | ratio_of_sums CRPS | mean_of_ratios CRPS | ROC-AUC (ros / mor) |
|---|---|---|---|
| (part, supplier) | 0.2488 [0.2463, 0.2513] | **0.2436 [0.2412, 0.2460]** | 0.586 / 0.587 |
| (supplier, plant) | 0.2628 [0.2604, 0.2649] | **0.2523 [0.2499, 0.2544]** | 0.539 / 0.544 |
| raw triple | 0.2482 [0.2456, 0.2505] | **0.2431 [0.2406, 0.2455]** | 0.588 / 0.588 |
| K4 shrunk hierarchy (ros by construction) | 0.2485 [0.2460, 0.2508] | — | 0.587 |
| rolling-52 histogram | **0.1583 [0.1558, 0.1604]** | | 0.605 |

A point forecast cannot place an atom at 1.0, so **arm (a) cannot win on CRPS by construction**: it loses by
0.09 (deviation 116). **mean_of_ratios beats ratio_of_sums on CRPS disjointly for every key**, and ROC-AUC is
indistinguishable.

**Arm (c) — ratio as an input to the current head.** TABLE A (raw) decides; TABLE B is beside it.

| key | seeds | split | A: CRPS | A: ECE-22 | A: ROC-AUC | B: CRPS | B: ECE-22 |
|---|---|---|---|---|---|---|---|
| **K4 hierarchy** | **5** | val | 0.12668 [0.12645, 0.12690] | 0.100 [0.059, 0.140] | 0.6341 [0.6317, 0.6377] | 0.12666 | ≈0 |
| **K4 hierarchy** | **5** | test | **0.13884 [0.13858, 0.13902]** | **0.132 [0.103, 0.172]** | 0.6205 [0.6192, 0.6240] | 0.13833 [0.13806, 0.13852] | 0.0542 |
| K1 (part, supplier) | **4, cap-cut, NOT QUOTABLE** | test | 0.13890 [0.13877, 0.13903] | 0.134 | 0.6196 | 0.13839 | 0.0550 |
| K2 (supplier, plant) | **0: not run (cap)** | | | | | | |
| K3 raw triple | **0: not run (cap)** | | | | | | |
| *head without ratio* | 5 | test | *0.13876 [0.13866, 0.13882]* | *0.1235* | *0.6204* | | |
| *histogram (bootstrap)* | — | test | *0.1583 [0.1558, 0.1604]* | *0.0165 [0.0143, 0.0275]* | *0.605 [0.599, 0.612]* | | |

**Against the histogram**, the brief's baseline to beat:

- **CRPS disjointly better** (0.1388 vs 0.1583).
- **ECE-22 guard disjointly DEGRADED** (0.132 vs 0.0165).
- ROC-AUC guard better.

**→ FAIL** on the guard. **Against the head without the ratio:** every metric **overlaps**, so the ratio adds
nothing measurable. The CRPS win over the histogram belongs to the head, not the ratio. The plain head wins it too
(deviation 117).

**Per-tier breakdown, K4, test (mandatory):**

| tier that fired | rows | share | CRPS (A, 5 seeds) | ROC-AUC |
|---|---|---|---|---|
| (part, supplier, plant) | 35,915 | **99.76%** | 0.13889 [0.13864, 0.13909] | 0.6205 |
| supplier (no triple or pair history) | 84 | 0.23% | 0.1149 [0.1142, 0.1158] | 0.599 |
| (part, supplier) | 1 | 0.003% | — | — |

**The triple fires on essentially every row.** No fallback is carrying the result, and so there is no risk of
"shipping the fallback believing it was the ratio". **Cold start**, meaning no own history, is **85 rows** (CRPS
0.1136). Every cold-start comparison rests on that n, and none is a finding.

**Verdict: F2 FAILS.** Arm (a) loses on CRPS by construction. Arm (c) fails the ECE guard and is indistinguishable
from the head without the ratio. K2 and K3 were not run (cap), and K1 is at 4 seeds.

### 3.3 Stage 2A — the shrunk ratio as the Beta head's centre (F2 arm (b))

The interior mean is `sigmoid(logit(r̂_K4) + offset)`, a Beta with learned concentration. Five seeds; the triple
tier fires on 99.76% of test rows.

| split | A: CRPS | A: interior | A: ROC-AUC | B: CRPS | [0.95, 1) pred / obs |
|---|---|---|---|---|---|
| val | 0.12716 [0.12703, 0.12727] | 0.113 [0.110, 0.117] | 0.6215 [0.6185, 0.6237] | 0.12832 | |
| test | **0.13904 [0.13900, 0.13908]** | **0.1226 [0.1185, 0.1272]** | **0.6053 [0.6037, 0.6078]** | 0.13907 | **3.37–3.81×** |

It is **disjointly worse than the head on every Table A metric.** Anchoring the interior on a history ratio makes
the near-1 excess **worse** (3.4–3.8× against the head's 1.9–3.6×), because the ratio (~0.84) pulls interior mass
toward the top. **FAIL.**

### 3.4 S1 — the simulation as a pre-rescue forecast

`ml/sim/phase13_s1.py` (`3d57191`). No model change. The reference is 2025 store positions with each week's **net**
transfer added back, named **`PRIVILEGED__prerescue_below_ss`**. `assert_no_privileged_headline` fires if that
reference is used for selection. The simulated side is Phase 12 B2's five fill-seed runs: **N = 200 paths, held
fixed**, all paths retained, fractions taken at the end.

| falsification (run before the headline) | result |
|---|---|
| **F-a** identity: zero transfers added | reproduces B2's per-seed ratios **exactly** (1.4870 / 1.4639 / 1.4647 / 1.4848 / 1.4725) |
| **F-b** wrong sign: pre-stated ≥ 1.8× (`phase-13-stage0.md`) | **6.55× [6.50, 6.61]**: moves away as required. My point estimate (≈2.1×) was 3× too small (deviation 118) |
| **F-c** shuffled: 50 permutations, total preserved | **1.38–1.40×, never in [1.0, 1.2]**: attribution to the specific rescued weeks holds |

**Headline:** pre-rescue reference **10.95%** below SS → **ratio 1.129× [1.121, 1.139]**, five fill seeds, **PASS**.

**What it means, stated narrowly.** The simulation forecasts **the world before inter-plant rescue** to within
13–14%. It does **not** forecast the observed world: that is still 1.475×. The planners' own transfers are the
difference, and the added-back transfers are **future information** relative to every t0. **S1 is a statement
about what the simulation represents, not a better forecast.**

### 3.5 S2 — transfer recommendations

`ml/opt/phase13_s2.py` (`6fe2244`).

**Precondition (deviation 91):** the donor search does **not** reuse allocation's qualification predicate.
Inter-plant transfers involve no supplier, so **0%** of v8 is excluded.

**Ground-truth defect (deviation 119):** `inventory_transactions.from_plant_id` equals the **receiving** plant on
all 1,482,011 transfer-in rows, so the donor is never recorded. A **donor hit** is inferred: a transfer-out of the
same part at the recommended donor in the same week.

**Rules.** "Projected short" is a fixed definition: P_sim(level < SS) ≥ 0.5.

- **Policy:** the donor whose simulated shortage chance after giving is lowest, recommended only if that chance
  is below θ.
- **Naive:** the nearest same-part plant by haversine distance with positive as-of surplus.
- **Random:** a uniformly random same-part plant with simulated surplus. This is the base rate.

All three are ranked by the recipient's shortage chance and compared at **matched k**. θ was fitted on
validation, maximising F1 = **0.5**. That is **the edge of the grid**: validation F1 rose monotonically in θ
(deviation 120). θ is then swept on test.

**Test, five fill seeds, `mean [min, max]`:**

| θ | k (matched) | **policy donor precision** | naive | random | recipient precision, policy / naive / random | policy count / naive count | recall (recipient), policy / naive |
|---|---|---|---|---|---|---|---|
| 0.02 | 17,335 | **0.556** [0.554, 0.558] | 0.521 | 0.515 | 0.595 / 0.616 / 0.616 | 17,335 / 25,436 | 0.046 / 0.071 |
| 0.10 | 21,215 | **0.550** [0.549, 0.552] | 0.524 | 0.519 | 0.597 / 0.620 / 0.620 | 21,215 / 25,436 | 0.057 / 0.071 |
| 0.30 | 23,812 | **0.555** [0.553, 0.556] | 0.526 | 0.520 | 0.609 / 0.623 / 0.623 | 23,812 / 25,436 | 0.065 / 0.071 |
| **0.50 (fitted)** | 24,662 | **0.556** [0.555, 0.558] | 0.527 [0.526, 0.528] | 0.520 [0.519, 0.522] | 0.616 / 0.624 / 0.624 | 24,662 / 25,436 | 0.068 / 0.071 |

- **Donor verdict: policy, at every θ in the sweep. The verdict does not flip.** The policy beats random by +0.036,
  disjointly. **Naive beats random by only +0.007**: "nearest with surplus" is barely better than chance.
- **Recipient level:** naive and random are **identical**, because they cover the same short weeks. The policy's
  lower recipient precision comes from its θ filter dropping weeks planners did rescue, **not** from donor choice.
  Recall is low for every rule (0.05–0.07): most real transfers go to weeks the simulation does not call short.
- **Before and after**, recorded per recommendation in `phase13_s2.json`. Example: P00001@PL04 ← PL01; recipient
  shortage chance 0.535 → 0.500; donor 0.000 → 0.000. **The recommended quantity only lifts the recipient's
  median to SS**, so its shortage chance falls to about 0.5 by construction. The layer names *who*, not *enough*.

> **Label (carried in every output):** freight cost and transfer lead time not weighed; validated against planner
> ACTION, not planner INTENT; the donor safety check inherits an uncalibrated simulation (see S1).

**Claim, in the brief's words:** it **reproduces experienced planners' donor choice better than a naive rule**,
0.556 against 0.527, with random at 0.520. **This is not evidence of shortage avoided.**

---

## 4. Pre-registered predictions, marked

| # | prediction | outcome |
|---|---|---|
| 1 | F1: the Beta head will NOT clearly beat a cheaper boundary fix; the [0.95, 1) problem is a loss/boundary problem | **RIGHT.** The boundary arm fixes most of [0.95, 1), has the only disjoint gain (ROC-AUC), and matches the Beta head everywhere else; the Beta head degrades ROC-AUC |
| 2 | F2: (part, supplier) beats the raw triple; the shrunk hierarchy beats both; (supplier, plant) close, may win on cold start | **UNTESTED / WRONG.** Raw triple and (supplier, plant) were not run (cap). The hierarchy and (part, supplier) **overlap** (the latter at 4 seeds), so "the hierarchy beats both" is **not supported**. Cold start is 85 rows |
| 3 | F2 arm (c) looks best on validation and worst on test | **WRONG.** On validation (c) sits inside the head's band (0.12668 vs 0.12667). The worst test arm is 2A, not (c) |
| 4 | ratio_of_sums will beat mean_of_ratios | **WRONG.** mean_of_ratios is disjointly better on CRPS for every key; ROC-AUC indistinguishable |
| 5 | S1: ratio lands in [1.0, 1.2] | **RIGHT**: 1.129× [1.121, 1.139] |
| 6 | S2 beats naive on agreement with planners, and this is not evidence of shortage avoided | **RIGHT on donor choice** (0.556 vs 0.527, every θ); **WRONG on recipients** (naive higher, 0.624 vs 0.616). The second clause stands |
| 7 | *(Claude)* Gate 0.3 closes | **WRONG**: 178 inside [149, 596] |
| 8 | *(Claude)* Gate 0.1 closes: zero-delivery closures not identifiable | **WRONG**: acknowledgements and revisions identify them, 19,696 / 19,721, zero false positives |
| 9 | *(Claude)* the boundary fix removes most of the [0.95, 1) excess but not the low-fill cells | **RIGHT**: [0.95, 1) goes from 1.85–3.61× to 0.98–1.90×, and cells 2–5 then dominate |

---

## 5. Deviations, continuing from 108

| # | Prior statement | Measured | Where |
|---|---|---|---|
| **109** | Phase 13 brief P1: `phase7_score.py` lines 93, 111, 334 still carry the constant substitution | **already fixed in Phase 11C (`ce58df5`).** The brief's premise is stale. A static unreachability test is added and shown firing | §2.1 |
| **110** | P2: identity carries every axis | **`fill_loss` was absent from artifact identity.** A reweighted-RPS run would have shared the plain head's bundle path. Fixed before any Phase 13 training | §2.1 |
| **111** | `reports/part2/phase-12.md:939` (deviation 100): "an old unreceived line is indistinguishable, as-of, from a pending one" | **contradicted.** `supplier_acknowledgements.ack_status = 'rejected'` or a qty→0 revision identifies 19,696 of 19,721 zero-delivery closures, with 0 false positives, recorded a median 2–5 days after the line. Phase 12's pipeline total (taken from the store) is unaffected; its reasoning was incomplete | §2.2 |
| **112** | brief 0.2: "0.018 (b5flat22, TRAIN MARGINAL)" | it is b5flat22's **validation** interior error. Its training marginal was never stored. Any criterion using it as a train target is invalid | §2.3 |
| **113** | observation 1 §3.3 and `phase-0-1-v8.md:311`: 0.13827 | a 3-seed band. At 5 seeds **0.13826 [0.13820, 0.13834]**, unchanged in substance | §2.3 |
| **114** | brief 1B: the histogram is the baseline to beat at ECE 0.0165, with 5-seed disjointness | the histogram is **deterministic** (no seeds), and its ECE is **0.0776 on validation**: 4.7× worse than on test. Comparisons against it use a row bootstrap, labelled. A 5-seed band for it cannot exist | §2.3, §3.2 |
| **115** | G2 fires on a leaked prior | **a first version passed vacuously.** The leaked construction produced an empty prior and the assertion accepted an undated one. Fixed to require a dated, non-empty prior, then shown firing. Also, κ's first method-of-moments solver bisected with the sign reversed and sat at the 10⁹ ceiling; fixed | §3.2 |
| **116** | brief F2 arm (a): "ratio alone" can pass on exact CRPS | **it cannot by construction.** A point forecast is a step CDF with no atom at 1.0 on a target with 75–80% mass there. It loses to the histogram by 0.09 | §3.2 |
| **117** | brief F2 PASS: beating the histogram's CRPS shows the ratio's value | **the plain head beats the histogram's CRPS too** (0.1388 vs 0.1583). The ratio's contribution is measured against the head without it, where it is nil. Also: the scorer's first glob pooled the 2A bundles into the K4 arm as a 7-"seed" band (a reader-side twin of deviations 28/73); fixed to exact per-seed names before any number was reported | §3.2 |
| **118** | S1 F-b pre-stated: ≈2.1× | **6.55×.** The criterion (≥ 1.8×) passes, but the symmetric-shift reasoning was wrong: transfers land on the weeks at SS, so subtracting them clears nearly every short week (reference 1.9%) | §3.4 |
| **119** | `inventory_transactions` records the donor of an inter-plant transfer | **`from_plant_id` equals the RECEIVING plant on all 1,482,011 transfer-in rows.** The donor is never recorded. S2's donor hits are inferred from same-week transfer-outs | §3.5 |
| **120** | S2's θ is fitted on validation | fitted at **0.5, the edge of the grid.** Validation F1 rises monotonically in θ, so the optimum may lie beyond it. The test verdict is the same at every θ, so this does not change the conclusion | §3.5 |
| **121** | Phase 13 budget: every arm at 5 seeds | **the 12 h caps cut three arms**: regression 3/5 (which also hit the 120-epoch ceiling on every seed, a floor), K1 4/5, K2 and K3 0/5. STOPPED markers committed; no silent overrun. Contention between three concurrent GPU jobs (~45 min per cell against ~21 alone) was the cost | §1 |

## 6. Every `inventory_position_weekly` read in this phase

| reader | fields | purpose |
|---|---|---|
| `ml/eval/phase13_stage0.py::stage03` | qty_on_hand, safety_stock_qty (2025 weeks) | Gate 0.3 arithmetic: held-out evaluation reference |
| `ml/sim/phase13_s1.py::store_2025` | qty_on_hand, safety_stock_qty (2025 weeks) | S1's reference (observed / pre-rescue / wrong-sign / shuffled): evaluation only |
| `ml/opt/order_policy.py` (Tables, `pipeline`, `transfer_frames`), `ml/sim/montecarlo.py::opening_position` | open_po_qty, qty_on_hand, safety_stock_qty, **as-of t0** | S2's simulation: opening level and open pipeline, the sanctioned Phase 9 path |

**None is a model feature.** The F1/F2 heads and `fill_history` never read it. `inventory_transactions` (transfers)
was read by Stage 0.3, S1, S2 and `order_policy.transfer_frames`, as evaluation references or as-of rate estimates.

## 7. Open items, continuing Phase 12's list (1–13)

14. **The fill head's interior error is 3.5–9× a non-neural baseline's, whichever parameterisation is used.** The
    head wins CRPS by ~1%. Either accept recalibrated calibration (Table B: 0.028, against b5flat22 raw 0.025) or
    ship b5flat22 through a real loader (deviation 46). The head-shape programme should stop here.
15. **The boundary fix (w = 3) is the one head change with a disjoint gain** (ROC-AUC +0.002). A candidate for a
    shipped-config proposal, not a change: it needs its own backtest.
16. **F2 keys (supplier, plant) and the raw triple are unrun, and (part, supplier) is at 4 seeds.** Given K4 ≈ K1
    ≈ the head, the expected value of finishing them is low. Record, do not prioritise.
17. **The low-fill cells [0, 0.25) are where interior error lives once [0.95, 1) is fixed**, for every arm and
    baseline. Any further fill work targets them.
18. **S1's meaning:** the simulation is calibrated to the pre-rescue world. A **transfer model at channel
    granularity** (Phase 12 open item 12) is still what would make it forecast the observed one.
19. **S2's quantity rule lifts recipients only to the median** (shortage chance ~0.5 after). Sizing to a stated
    service level, and weighing freight and lead time, are needed before it recommends anything a planner would
    act on.
20. **`from_plant_id` never records the donor** (deviation 119). It is a data defect the generator's authors
    should know about.
21. **mean_of_ratios beats ratio_of_sums as a point summary**, the opposite of the pre-registration. That bears
    on how any history ratio is quoted to a client.

## 8. Compliance

- **`ml/configs/shipped.json` unchanged.** Every protected path is unchanged: an empty `git diff` from Phase 12's
  final commit for `db/`, `docs/`, `reports/part1/`, every prior `reports/part2/` report, and `results/`.
  Generators were **read as configuration** (to find the transfer threshold); `phase13_s1.py` and
  `order_policy.py` cite them as a client ask, not data.
- **Selection on validation only:** the boundary weight (raw val interior error), S2's θ (val F1), and the F2 κ
  (fitted as-of, no hyperparameter). Nothing was selected on test. Recalibrated validation figures were never used.
- **No learning rate retuned**; every arm at the shipped 1.25×10⁻⁴.
- **Five model seeds** for every quoted arm. Cap-cut arms are marked not quotable. Deterministic arms carry
  labelled bootstrap intervals, never seed bands.
- **Every new gate was shown capable of failing:** the static substitution test; identity (6/6); G1; G2 (after the
  vacuous first version was fixed); the empty-history guard; S1's F-a/F-b/F-c; the privileged guard; the scorer's
  `+dirty` refusal.
- **Commit before running.** Artifacts are stamped clean: stage 0 `1fc1d29`; S1 `3d57191`; S2 `6fe2244`;
  scorer `ce4fc15`. Every Phase 13 bundle stamp is clean, which the scorer asserts.
- **No assertion weakened.** One was *strengthened* (G2's prior provenance).

---

## Plain-language summary

This phase tried to make the fill forecast's detailed shape match reality, and to give the stock simulation an
honest reference. **Reshaping the fill model did not work.** A smooth interior, history ratios, and ratios as the
curve's centre each either moved the error somewhere else or made it worse. The one small real gain came from a
cheap loss tweak. The detailed shape of a simple historical histogram, and of the existing gradient-boosted model,
is still 3–9× closer to reality than any neural variant, though the neural model remains slightly better overall.
**On the simulation, two things are now quotable, each with its label.** The simulation matches the
**pre-rescue** world, before planners move stock between plants, to within about 13%, but it still over-projects
the observed world by about 1.48×. A transfer-suggestion layer picks the donor plant planners actually used 55.6%
of the time, against 52.7% for "nearest plant" and 52% by chance. **It measures agreement with planners, not
shortages avoided, and ignores freight and lead time.** **Not quotable:** any improvement to fill calibration from
this phase; any shortage quantity as a forecast of the observed world; any transfer recommendation as a cost or
service benefit; and any figure from the arms the time cap cut (regression control, and two of four history keys).
