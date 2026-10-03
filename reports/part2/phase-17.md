# Phase 17 — four decisions and four trained arms

**Audience:** whoever decides what the shortage, rescue, capacity and fill products become next.
**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, val 2024, test 2025). Thresholds fitted on validation only.
**Companions:** `phase-17-preregistration.md` (`c7294b7`), `phase-17-STOPPED.json`, `phase-15.md`, `phase-14.md`, `phase-12.md` §C3.
**Branch / base:** `phase17`, from `origin/HADES-v4-ml-pipeline` at `90a38ed` (Phase 15), plus three Windows/CUDA commits
(`167eafb`, `11c6019`, `b2c7c31`). Every artifact carries its commit SHA; every run refused to start on a dirty `ml/`.

## Environment (Stage −1) — read this before any number

| | |
|---|---|
| machine | Windows 11, **NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM**, driver 581.95 (the brief assumes 32 GB) |
| torch / CUDA / cuDNN | 2.14.0+cu126 / 12.6 / 9.10.02; Python 3.10.11; float32, **TF32 off** |
| memory mode | `HADES_PANEL_HOST=1`, `HADES_TCN_CHUNK=4096` (panel in host RAM, TCN checkpointed in channel chunks; arithmetic unchanged — one step's loss bit-identical, `11c6019`) |
| **concurrency level** | **1 for every Phase 17 cell** (two concurrent cells measured ~10× slower each on 4 GB: VRAM spills to shared memory) |
| GPU clock | the laptop firmware held the GPU in **software thermal slowdown / power cap** for most of the run (~350–830 MHz of 2,100, ~30 W; full clock only briefly after a reboot), with the Alienware profile on Performance. It affects wall-clock only |
| stored artifacts | the Mac's `ml/artifacts` (copied); `cache/v8` **byte-identical** to this machine's rebuild (panel / miss / active SHA-1 equal) |

**Handshake outcome: INSIDE the stored band.** Stored `arrival_week v8_lite_h4_lr0.00025_s7` (`b1794c0`), retrained 3 epochs here:
val C-index per epoch 0.4875004 / 0.5005191 / 0.5555158 against stored 0.4874936 / 0.5005219 / 0.5555247 (|diff| ≤ 8.8e-6;
the stored 5-seed spread at those epochs is 0.025–0.040). Training loss agrees to ≤ 1.6e-5. **Stored baselines are reused
and the halt-and-diagnose rule is a genuine signal.** Not one baseline was retrained, so it never fired.

**Time:** the brief's per-stage caps applied to B1 (B1b stopped at 2 of 5 seeds). On the user's instruction the caps were
then lifted for B2–B4 and the stopped B1b seeds, under a single wall-clock stop at 10:00 2026-10-03 (deviation 152).

---

## 1. The four Track A decisions

| | question | decision rule (pre-registered) | measured | **resolved** |
|---|---|---|---|---|
| **A1** | does the shortage head survive a 3% base rate? | precision @ 5% coverage: ≥ 0.60 survives · 0.40–0.60 watchlist · < 0.40 retire | **0.224** [0.223, 0.226] importance-weighted; **0.223** [0.222, 0.225] subsampled (5 seeds) | **RETIRE** |
| **A2** | why does the simulation miss two-thirds of rescues? | missed part-plants have more channels → build channel granularity · miss uniform → do not | channels 3.90 vs 3.84 (median 4 vs 4); P(missed > flagged) **0.51** rows / **0.504** part-plants | **granularity is NOT the explanation — do not start that build.** The miss is not uniform either: it is horizon- and starting-position-structured (§1.2) |
| **A3** | signed-days lateness from the hazard head | (P3) beats the channel-median constant on median abs error | **13.06** [12.86, 13.30] days vs **14.00** | **beats the constant** (disjoint); undetermined vs LightGBM (13.25) |
| **A4** | do capacity labels exist at 30 / 60 / 120 days? | gate only | **30 no · 60 no · 90 yes · 120 no** | horizon stacking is **not schedulable** without new labels |

### 1.1 A1 — the shortage head at a 3% base rate

`ml/eval/phase17_a.py a1` → `phase17/a1.json` (`ee6ed7e`). Shipped `mp h¹`, 5 seeds, **test**, stored predictions. (a)
weight k ≈ 11.2 on every negative so the weighted base rate is 3%; (b) all negatives kept, positives subsampled to 3%, 100
resamples per seed. Cells with < 50 alerts suppressed (none were).

| coverage | v8 base 25.7%: precision | **3%, importance-weighted**: precision / recall / F1 | **3%, subsampled**: precision (seed-7 95% range) | ceiling at 3% | Δ precision |
|---|---|---|---|---|---|
| 0.5% | 0.979 | 0.545 / 0.092 / 0.157 | 0.550 [0.413, 0.635] | 1.00 | −0.43 |
| 1% | 0.961 | 0.457 / 0.153 / 0.229 | 0.455 [0.385, 0.538] | 1.00 | −0.50 |
| 2% | 0.946 | 0.352 / 0.235 / 0.282 | 0.352 [0.292, 0.404] | 1.00 | −0.59 |
| **5%** | **0.890** | **0.224** / 0.374 / 0.280 | **0.223** [0.199, 0.256] | **0.60** | **−0.67** |
| 10% | 0.806 | 0.151 / 0.503 / 0.232 | 0.150 [0.137, 0.171] | 0.30 | −0.66 |

PR-AUC **0.660 → 0.234** (both methods). Always-majority on the same rows: accuracy 0.743 at v8's base, **0.970** at 3%,
F1 0, no alert raised. Both re-weightings agree to the third decimal.

**Retire.** 0.224 is far under 0.40, and under the 0.60 a perfect ranker could reach (deviation 144), so the ceiling is not
what decides it. Stop spending on the shortage BCE head.

### 1.2 A2 — the rescued weeks the simulation misses

`phase17_a.py a2` → `phase17/a2.json`. **Gate first:** Phase 14's counts reproduced exactly on all five fill seeds (12,401
rescued; flagged 4,171 / 4,152 / 4,246 / 4,231 / 4,338; missed 8,230 / 8,249 / 8,155 / 8,170 / 8,063), each seed's τ refitted
on validation with Phase 14's rule.

| profile (5-seed mean) | **missed** (~8,170/seed) | flagged (~4,230/seed) | all test rows |
|---|---|---|---|
| channels per part-plant, row-weighted: mean / median | **3.90 / 4** | 3.84 / 4 | 3.81 |
| … P(missed row has more channels than a flagged row) | **0.51** (p up to 0.05) | | |
| … same, distinct part-plants | **0.504** (p up to 0.86) | | |
| concentration, part-plant: top-decile share / Gini | 0.459 / 0.706 | 0.664 / 0.830 | 0.103 / 0.029 |
| … part | 0.234 / 0.358 | 0.305 / 0.492 | 0.103 / 0.025 |
| … plant | 0.174 / 0.071 | 0.179 / 0.098 | 0.145 / 0.005 |
| … supplier (a week split across its part-plant's suppliers) | 0.164 / 0.191 | 0.211 / 0.282 | 0.133 / 0.104 |
| opening on-hand / safety stock (median, as of t0) | **1.69** | 1.20 | 2.02 |
| open pipeline / safety stock (median) | **1.93** | 2.38 | 1.60 |
| weeks ahead 1 → 13 (seed 7 counts) | **rising** 516 → 714 | **falling** 473 → 226 | flat |

Uniform expectation: top-decile share 0.10, Gini 0.

**What the miss is not:** channel count (no measurable difference), plant (near uniform), supplier (weak).
**What it is:** concentrated on a minority of part-plants (though less so than the flagged weeks), **late in the 13-week
horizon**, and starting from **comfortable stock** with a **thinner pipeline**. The simulation catches drawdowns that are
close and already visible at t0. It misses drawdowns that develop later from a healthy start.

### 1.3 A3 — signed-days lateness

`phase17_a.py a3` → `phase17/a3.json`. `E[lateness_days] = 7(Σₖ P(T=k)·k + 13·P(T>12)) − 7R`. R = as-of channel median observed
lead + c (c = +4.571 wk, training rows), never the promise date. Test: 36,000 rows, **18,534 uncensored** (scored),
**17,466 censored** (no observed lateness).

| arm | median abs error (days) | MAE | within ±3 d | within ±7 d |
|---|---|---|---|---|
| **h⁴ expected lateness, raw (5 seeds)** | **13.06** [12.86, 13.30] | 14.68 | 0.127 | 0.285 |
| h⁴ P50, raw | 14.00 | 16.38 | 0.121 | 0.286 |
| h⁴ expected, recalibrated | 15.92 [15.12, 16.45] | 17.94 | 0.102 | 0.236 |
| channel-median constant | 14.00 | 15.84 | 0.098 | 0.284 |
| reference only (lateness 0) | 14.00 | 15.74 | 0.104 | 0.278 |
| b5flat LightGBM point arm (5 seeds) | 13.25 [13.21, 13.27] | **14.07** | 0.115 | 0.266 |

- **The head beats the channel-median constant on median error**, disjointly. It ties LightGBM on median error (bands
  overlap) and loses to it on MAE.
- **For a point error the reference cancels:** (pred − R) − (Y − R) = pred − Y. R only places the number on a lateness scale
  and decides late / on time (deviation 150).
- The P10–P90 interval covers **93%** of outcomes against a nominal 80% (raw): honest but wide. Recalibration worsens
  every point metric; raw and recalibrated are not mixed.

**One row in the LLM-explanation shape** (seed 7, first uncensored test row; the LLM may restate only these fields):

| field | value |
|---|---|
| prediction | expected lateness **+9.9 days**; median **+21.0 days** (= the "later than 12 weeks" cap: more than half the mass is beyond the horizon) |
| interval | P10 −14.0 to P90 +21.0 days |
| reference | **70.0 days**, as-of channel median observed lead + c, **as of 2025-01-06** |
| data support | **50 deliveries** behind the reference |
| top signals (elevated vs the training normaliser) | `lead_time_ratio` z = 3.88 · `lead_time_actual_days` z = 1.80 · `ack_gap_ratio` z = 1.67 |
| caveat | correlational; the signals are elevated inputs, not causes; no action is prescribed |
| actual (not shown to the LLM) | −7.0 days |

### 1.4 A4 — capacity horizons

`training_labels`: all 207,500 `capacity_strain` rows have `horizon_days = 90` and a 90-day window; the generator emits one
horizon (`db/gen_v8/generator_v8.py:1485`). **No** labels at 30, 60 or 120 days. Nothing was built.

---

## 2. Track B — per arm

| arm | seeds | the number that decides it | disjoint? | **verdict** |
|---|---|---|---|---|
| **B1a** predict-the-rescue, LightGBM | **5 / 5** | precision **0.823** [0.822, 0.824] at recall 0.246 (proxy 0.619) | yes | **GAIN → GO** |
| B1b predict-the-rescue, neural | **2 / 5** (STOPPED, cap) | 0.859 [0.858, 0.859] at recall 0.238 | not decidable at 2 seeds | **gain over the proxy; vs B1a undetermined** |
| B2b Δ-strain capacity | 5 / 5 | max val precision 0.569; never reaches p = 0.70 (B2a: recall 0.191 at 0.810) | yes (below at every coverage) | **WORSE** |
| B2c level + Δ, alert on either | 5 / 5 | max val precision 0.662; never reaches p = 0.70 | yes | **WORSE** |
| B3b 5-band ordinal fill | 5 / 5 | raw exact CRPS 0.13945 vs 0.13876 (worse); recalibrated 0.13829 vs 0.13825 | raw yes / recal no | **WORSE (raw) · TIE (recal)** |
| B4b lean encoder | 5 / 5 | test C-index 0.67429 vs 0.67442; lateness 0.70747 vs 0.70905 | no | **TIE (a win per the brief)** |

### 2.1 B1 — predict-the-rescue

`ml/train/phase17_b1.py` (`95642a9`…`255050d`) → `phase17/b1_score.json`.

**Rows and label.** Phase 14/15's UC5 universe: part-plant × the 13 weeks after each snapshot. Validation (438,048) and test
(492,804) rows were **rebuilt and gated** against the stored `phase15_sim` rows: part-plant, week and label equal element by
element. Train: 2,409,264 rows, 44 snapshots. **Base rate: train 0.455, val 0.405, test 0.4535** (the brief's ≈ 0.454,
confirmed). Label = a `transfer_in` recorded that week; features as of t0; no part-plant without a channel occurs.

**Arms.** B1a = LightGBM on the existing `part_plant_features` (the b5flat_bin set) plus the week offset, frozen GBM config,
5 seeds. B1b = the shipped shortage head (`mp h¹`, lr 1.25e-4), part-plant pooled, one-hot week offset through the existing
per-row input, one optimiser step per snapshot. **No feature was engineered** (deviation 146 on the week offset).

| arm (seeds) | test PR-AUC | test ROC-AUC | **precision at recall ≥ 0.20** (val-chosen, test) | test recall there | P @ 1% / 5% / 10% coverage | val precision at the proxy's recall 0.418 / 0.183 |
|---|---|---|---|---|---|---|
| Phase 14 crude proxy (5 fill seeds) | 0.547 | 0.623 | 0.619 [0.619, 0.620] | 0.194 | 0.593 / 0.625 / 0.625 | 0.561 / 0.600 |
| as-of trailing transfer frequency (reference) | 0.605 | 0.663 | 0.631 | 0.376 | 0.781 / 0.745 / 0.708 | 0.530 / 0.575 |
| **B1a LightGBM (5)** | **0.729** [0.7285, 0.7289] | 0.772 | **0.823 [0.822, 0.824]** | 0.246 | 0.938 / 0.883 / 0.842 | **0.746 / 0.843** |
| B1b neural (**2 of 5**: seeds 7, 17) | 0.757 [0.7567, 0.7577] | 0.797 | 0.859 [0.858, 0.859] | 0.238 | 0.947 / 0.909 / 0.874 | 0.785 / 0.874 |

Majority baseline on the same test rows: accuracy 0.546 ("never acted"). At the validation max-F1 threshold B1a gives
precision 0.596 / recall 0.841 / F1 0.698 / accuracy **0.669 vs 0.546 majority**.

- **GO.** B1a reaches **0.823** precision at recall 0.246, against > 0.70 at ≥ 20% recall. It is disjoint from the proxy at
  five seeds, and the validation-chosen point holds on test (validation 0.835).
- **It is not persistence.** A part-plant's own as-of transfer frequency gets 0.631 at that recall, barely above the proxy.
  The learned model adds 0.19 of precision over both.
- **B1b's seeds stopped at the 120-epoch ceiling** with validation PR-AUC still rising (best epoch 119), so its figures are a
  floor (deviation 147). Its two seeds sit above B1a's band (0.859 vs 0.822–0.824), but **two seeds are not the five
  the rule requires: B1b vs B1a is undetermined.** The other three seeds were stopped by the B1 cap. After the cap was
  lifted they were queued last and did not fit before 10:00 (deviation 152). **The GO rests on B1a.**

### 2.2 B2 — the Δ-strain capacity head

`ml/train/phase17_b2.py` (`b6aefe2`, scorer `30d80b1`) → `phase17/b2_score.json`. B2a = the stored `mp h⁴` bundles (reused;
handshake inside band). B2b/B2c = the incumbent architecture and LR, **target changed**. Δ = strain − m, where m is the
channel's median capacity label whose 90-day window **ended** in the 52 weeks to t0 (as of t0 by construction; asserted).
Decision = Phase 15's UC3 (strain > 1), operating points chosen on validation per seed. Test base rate **0.405**.

| arm (5 seeds each) | Stage B (val) | max val precision | recall @ p = 0.70 / 0.80 / 0.85 (test precision) | precision @ 1% / 5% / 10% coverage | **vs B2a** |
|---|---|---|---|---|---|
| **B2a incumbent mp h⁴** | TUNABLE | — | **0.515** (0.681) / **0.318** (0.762) / **0.191** (0.810) | 0.904 / 0.851 / 0.810 | — (reproduces Phase 15 exactly) |
| B2b Δ only | **NOT TUNABLE** | **0.569** [0.564, 0.573] | unreachable at every bar | 0.529 / 0.636 / 0.625 | **WORSE** |
| B2c level + Δ, alert on either | **NOT TUNABLE** | **0.662** [0.654, 0.673] | unreachable at every bar | 0.582 / 0.689 / 0.701 | **WORSE** |
| *diagnostic, not an arm:* B2c's level head alone | TUNABLE | — | 0.578 (0.651) / 0.382 (0.734) / 0.280 (0.764) | 0.910 / 0.860 / 0.806 | no verdict |

- **Fail.** Neither arm reaches even 0.70 precision on validation, so there is no recall at fixed precision to compare. Both
  are worse than the incumbent, disjointly at every coverage point.
- **Why: the target carries noise, not movement.** m correlates with the realised 90-day strain at **0.08 (train), 0.004
  (val), 0.09 (test)**. A label is known only ~90 days after its snapshot, and each snapshot samples a different subset of
  channels, so 26–30% of rows fall back to a single older label (deviation 148). Predicting y − m is predicting y plus
  noise: B2b's P50 correlates 0.317 with the outcome against the incumbent's 0.533.
- **B2c: the encoder is fine, the alert rule is not.** Its level head alone tracks the incumbent (P50 correlation 0.525;
  same precision-at-coverage frontier, 0.806 vs 0.810 at 10%). "Alert on either crossing" lets the noisy Δ head's confident
  errors fill the top of the ranking. The diagnostic row's higher recall at p = 0.85 comes with lower test precision
  (0.764 vs 0.810). That is a slide along the same curve, as in Phase 15 Stage F, not a gain, and it is not claimed.
- Validation pinball per arm (5 seeds): B2b 0.0988–0.0996, B2c 0.0843–0.0852, against about 0.069–0.070 for the incumbent.

### 2.3 B3 — five-band ordinal fill head

`heads.FillBand5Head` (`38d14fd`, a new class: softmax over {0} | (0, .50) | [.50, .95) | [.95, 1) | {1}, ordinal RPS over the 4
band boundaries, mass spread evenly over each band's cells so every 22-cell scorer applies), trained by `loop.py
--fill-head band5` with the incumbent's arch / depth / LR. Scorer `ml/eval/phase17_b3.py` (`7118129`) →
`phase17/b3_score.json`. B3a = the stored 22-cell h⁰ bundles. Test, 5 seeds each.

| | B3a incumbent, raw | **B3b 5-band, raw** | raw verdict | B3a, recalibrated | **B3b, recalibrated** | recal verdict |
|---|---|---|---|---|---|---|
| exact CRPS | 0.13876 [0.13866, 0.13882] | 0.13945 [0.13923, 0.13978] | **WORSE (disjoint)** | 0.13825 [0.13820, 0.13834] | 0.13829 [0.13807, 0.13843] | undetermined |
| interior Σ\|err\| (20 cells) | 0.0878 [0.0764, 0.1039] | 0.0951 [0.0924, 0.0978] | undetermined | 0.0279 | 0.0275 | undetermined |
| ECE-22 | 0.1235 [0.1103, 0.1414] | 0.1307 [0.1274, 0.1328] | undetermined | 0.0549 | 0.0537 | undetermined |
| P(fill = 1) ROC-AUC | 0.6204 [0.6200, 0.6211] | 0.6220 [0.6189, 0.6257] | undetermined | 0.6205 | 0.6220 | undetermined |

**Predicted vs actual frequency per band (raw, 5-seed mean):**

| band | actual | B3a predicted (ratio) | B3b predicted (ratio) |
|---|---|---|---|
| {0} | 0.022 | 0.023 (1.05) | 0.022 (1.03) |
| (0, 0.50) | 0.172 | 0.142 (**0.83**) | 0.142 (**0.83**) |
| [0.50, 0.95) | 0.051 | 0.041 (**0.80**) | 0.042 (**0.82**) |
| [0.95, 1) | 0.006 | 0.016 (**2.55**) | 0.010 (**1.56**) |
| {1} | 0.749 | 0.779 (1.04) | 0.784 (1.05) |

- **Worse raw, tie recalibrated.** The band cut halves the `[0.95, 1)` over-prediction (2.55× → 1.56×), the cell Phase 13
  named. But it buys nothing in aggregate: raw exact CRPS is disjointly worse, and after recalibration every metric ties.
- **The same bands dominate.** The two interior bands are still under-predicted by 17–18% (0.83×, 0.82×), unchanged, and
  {1} is still over-predicted. In absolute mass these errors (−0.030, −0.009, +0.035) are 4–9× the `[0.95, 1)` error the
  boundary was built to fix (+0.004). An aggregate that does not improve while the same bands dominate has fixed nothing.
- As expected, this does not change the decision: fill stays a watchlist (measured precision ceiling 0.577, Phase 15).

### 2.4 B4 — the lean encoder

`ml/models/share_lean.py` (`6c27972`; **new classes** `SHARELeanLayer` / `SHARELean`), trained by `ml/train/phase17_b4.py`
through the incumbent's own `loop.train` and `loop.finish_bundle`. The encoder is swapped by a subclass bound in-process; no
incumbent class was edited (**`git diff 90a38ed HEAD -- ml/models/share.py ml/models/tcn.py` is empty**; `heads.py` only
gained `FillBand5Head`). Equivalence first (`phase17/b4_equivalence.json`): with copied weights the lean encoder equals C3's
`drop_relation part` incumbent to **3e-8** on the real graph, and the check fails against a supplier-dropped variant (0.018).
Scorer → `phase17/b4_score.json` (`f130084`). Arrival, test, as-of lateness reference (`docs/specs/lateness_metric.md`).

| | B4a incumbent SHARE-lite h⁴ (stored, 5 seeds) | **B4b lean (5 seeds)** | verdict |
|---|---|---|---|
| test C-index | 0.67442 [0.67344, 0.67551] | 0.67429 [0.67293, 0.67527] | **TIE** |
| test lateness ROC-AUC (as-of R) | 0.70905 [0.70645, 0.71298] | 0.70747 [0.70061, 0.71980] | **TIE** |
| validation C-index | 0.67067 [0.66904, 0.67193] | 0.67004 [0.66789, 0.67160] | **TIE** |
| model parameters | 533,004 | **369,164** (−163,840, −30.7%) | |
| epochs to stop (best / run) | 44–51 / 53–60 | 35–57 / 44–66 | |
| s / epoch | 19.2–19.7 (Mac, MPS: not comparable); **25.7–46.8 on this machine** (3 seeds, level 1) | **27.5–43.5** (this machine, level 1) | undetermined |

- **Tie, and per the brief a tie is a win.** The incumbent's lateness figures reproduce the spec of record exactly
  (0.70905 [0.70645, 0.71298]), so the comparison is anchored. Lean ties on every metric.
- **Wall-clock is undetermined.** Both encoders' per-epoch time on this machine is dominated by the laptop firmware's
  thermal throttling (GPU clock 350–1,965 MHz), which moves a cell by ±40%, more than the graph encoder's share of the
  step. The parameter cut (−30.7%) is certain; a timing win has to be measured on an unthrottled machine (deviation 151).

---

## 3. Pre-registered predictions

| # | prediction | **right / wrong** | the number |
|---|---|---|---|
| **P1** | shortage head < 0.40 @ 5% coverage at a 3% base | **RIGHT** | 0.224 (both methods) |
| **P2** | missed rescues concentrated **and** missed part-plants have more channels | **WRONG** | concentrated, yes (Gini 0.71); more channels, no (P = 0.51 / 0.504) |
| **P3** | signed-days lateness beats the channel-median constant on median abs error | **RIGHT** | 13.06 [12.86, 13.30] vs 14.00 |
| **P4** | predict-the-rescue > 0.63 precision at matched recall on validation | **RIGHT** | B1a 0.746 at the proxy's recall 0.418, 0.843 at 0.183 (proxy 0.561 / 0.600) |
| **P5** | Δstrain head does **not** beat the incumbent disjointly | **RIGHT** | both Δ arms worse: max val precision 0.569 (B2b) and 0.662 (B2c), never reaching p = 0.70; incumbent recall 0.191 at 0.810 |
| **P6** | lean encoder **ties** the full encoder | **RIGHT** | test C-index 0.67429 vs 0.67442, lateness 0.70747 vs 0.70905, all bands overlapping (5 seeds each) |

## 4. Concurrency per cell

**Every Phase 17 cell ran at concurrency level 1** (one GPU job at a time), recorded in each new bundle's `train_log.json`
(`concurrency_level`). Within every comparison all new cells share that level. **The stored baseline cells (B2a, B3a, B4a)
were trained on the Mac and their logs do not record a concurrency level**, so the brief's "same level within a comparison"
cannot be verified for the stored side (deviation 143). The level affects wall-clock, not arithmetic: the handshake
reproduced a stored trajectory to ≤ 8.8e-6. B4's wall-clock comparison therefore uses this machine's own incumbent runs.

| arm | seed | commit | device | concurrency level | best / run | stop | s/epoch | minutes |
|---|---|---|---|---|---|---|---|---|
| B1a LightGBM rescue | 7 | `0a19487` | cpu | **1** | 398/400 trees | early-stop |  | 0.6 |
| B1a LightGBM rescue | 17 | `0a19487` | cpu | **1** | 400/400 trees | early-stop |  | 0.6 |
| B1a LightGBM rescue | 27 | `0a19487` | cpu | **1** | 400/400 trees | early-stop |  | 0.6 |
| B1a LightGBM rescue | 37 | `0a19487` | cpu | **1** | 399/400 trees | early-stop |  | 0.6 |
| B1a LightGBM rescue | 47 | `0a19487` | cpu | **1** | 400/400 trees | early-stop |  | 0.6 |
| B1b neural rescue head | 7 | `3adea7c` | cuda | **1** | 119/120 | CAP (floor) | 37.2 | 75 |
| B1b neural rescue head | 17 | `d2d79c7` | cuda | **1** | 119/120 | CAP (floor) | 40.2 | 81 |
| B2b Delta-strain | 7 | `255050d` | cuda | **1** | 50/59 | patience | 40.4 | 40 |
| B2b Delta-strain | 17 | `d992600` | cuda | **1** | 47/56 | patience | 40.2 | 38 |
| B2b Delta-strain | 27 | `fea0741` | cuda | **1** | 49/58 | patience | 37.6 | 36 |
| B2b Delta-strain | 37 | `30d80b1` | cuda | **1** | 50/59 | patience | 37.6 | 37 |
| B2b Delta-strain | 47 | `30d80b1` | cuda | **1** | 50/59 | patience | 37.7 | 37 |
| B2c level+Delta | 7 | `d992600` | cuda | **1** | 49/58 | patience | 40.3 | 39 |
| B2c level+Delta | 17 | `d992600` | cuda | **1** | 44/53 | patience | 38.3 | 34 |
| B2c level+Delta | 27 | `30d80b1` | cuda | **1** | 51/60 | patience | 37.6 | 38 |
| B2c level+Delta | 37 | `30d80b1` | cuda | **1** | 51/60 | patience | 38.0 | 38 |
| B2c level+Delta | 47 | `30d80b1` | cuda | **1** | 45/54 | patience | 37.9 | 34 |
| B3b 5-band fill | 7 | `30d80b1` | cuda | **1â€ ** | 96/105 | patience | 36.3 | 63 |
| B3b 5-band fill | 17 | `30d80b1` | cuda | **1â€ ** | 82/91 | patience | 36.1 | 55 |
| B3b 5-band fill | 27 | `30d80b1` | cuda | **1â€ ** | 93/102 | patience | 30.6 | 52 |
| B3b 5-band fill | 37 | `30d80b1` | cuda | **1â€ ** | 91/100 | patience | 33.3 | 55 |
| B3b 5-band fill | 47 | `30d80b1` | cuda | **1â€ ** | 80/89 | patience | 34.7 | 51 |
| B4b lean encoder | 7 | `30d80b1` | cuda | **1** | 43/52 | patience | 27.5 | 24 |
| B4b lean encoder | 17 | `30d80b1` | cuda | **1** | 57/66 | patience | 37.5 | 41 |
| B4b lean encoder | 27 | `30d80b1` | cuda | **1** | 44/53 | patience | 33.6 | 30 |
| B4b lean encoder | 37 | `30d80b1` | cuda | **1** | 35/44 | patience | 43.5 | 32 |
| B4b lean encoder | 47 | `30d80b1` | cuda | **1** | 52/61 | patience | 36.9 | 38 |

â€  trained through `loop.py`, which does not write `concurrency_level`; level 1 from `ml/artifacts/phase17/chain.out` (the queue runs one cell at a time and logs each START/END).

## 5. Identity check

`ml/tests/test_phase17_identity.py` (`2db9004`): **371 stored configs** recompute to identical `identity_of`, bundle path and
config name under the Phase 17 axes (`cap_target`, `lean_encoder`, each omitted at its default), against `artifact_identity`
as of `90a38ed`. The falsification half passes: non-default values get distinct names, default = absent. 31 stored bundles
are Phase 16 encoder variants whose axes live on the unmerged `phase16-encoders` branch; this base cannot name them
(deviation 149).

## 6. Deviations, continuing from 142

| # | prior statement | measured / done | where |
|---|---|---|---|
| **143** | brief: one GPU with 32 GB; "at most 3 training jobs at once", same level within a comparison | **4 GB**. Two concurrent cells measured ~10× slower each (VRAM spills to shared memory), so **every Phase 17 cell ran at level 1**. The stored baselines' logs (B2a/B3a/B4a, Mac) **record no concurrency level**, so "same level within a comparison" cannot be verified for the stored side. Level affects wall-clock, not arithmetic (handshake ≤ 8.8e-6) | header, §4 |
| **144** | A1 decision rule: "≥ 0.60 precision @ 5% coverage → survives" | at a 3% base and 5% coverage **even a perfect ranker reaches 0.03 / 0.05 = 0.60**: the top band is reachable only by perfection. The head's 0.224 is far below 0.40, so the ceiling does not decide the verdict | §1.1 |
| **145** | A2 decision rule: "more channels → granularity; uniform → not" | **neither branch fits**: channel count does not differ (P = 0.51), but the miss is not uniform. It is concentrated, late in the horizon and from comfortable starting stock. Recorded as "granularity is not the explanation", plus what the miss is | §1.2 |
| **146** | B1: "engineer no new features" | the **week offset** (0–12) is passed as the row key (a LightGBM column; a one-hot through B1b's existing per-row input). Without it the 13 weekly rows of a part-plant-snapshot are indistinguishable. B1b keeps the shortage head's batching (one step per snapshot), but a step now holds ~54,756 rows, because the target universe is part-plant × week | §2.1 |
| **147** | B1 arms trained to convergence | **B1a hit the frozen GBM's 400-tree cap on every seed** (best iteration 398–400); **B1b hit the 120-epoch cap on both seeds** (best epoch 119, validation PR-AUC still rising). Both are **floors**; nothing was retuned | §2.1 |
| **148** | B2: "predict movement directly rather than inferring it from two stale levels" | the as-of trailing median correlates with the realised 90-day strain at **0.08 / 0.004 / 0.09** (train / val / test). 26–30% of rows fall back to the last ended label (labels sample a subset of channels per snapshot), 407 training rows to the global median. Δ is the level plus noise | §2.2 |
| **149** | brief: "every stored config from Phases 0–15 recomputes to its stored identity" | **371 verified unchanged**. **31 stored bundles are Phase 16 encoder variants** (`_encshare_pna…`, `_encshare_traj…`) whose axes exist only on the unmerged `phase16-encoders` branch; this base cannot name them. Not caused by Phase 17 | §5 |
| **150** | A3: point lateness against the as-of reference | **the reference cancels in a point error** ((pred − R) − (Y − R) = pred − Y). It sets the displayed lateness and the late / on-time call only. P50 is week-quantised (median error exactly 14.00 days). The P10–P90 interval covers 93% against a nominal 80% | §1.3 |
| **151** | B4: "per-cell wall-clock" against the incumbent | the stored B4a timings are from the Mac (MPS) and are **not comparable**; B4b is compared with this machine's own full-length incumbent runs (`ml/artifacts_win`, 3 seeds, level 1). The laptop firmware varied the GPU clock between ~350 and 1,965 MHz (software thermal slowdown), so wall-clock is noisy. Track A ran on the CPU alongside the first B1b cell | §2.4 |
| **152** | per-stage caps; "on cap, commit partials with a STOPPED marker" | the B1 cap applied (B1b **2 of 5**, `d992600`). **On the user's instruction the caps were then lifted** for B2–B4 and the stopped B1b seeds, under one wall-clock stop at **10:00 2026-10-03**. A **reboot at 01:17** cut B3 seed 27 and left the GPU idle until 04:02 (~3 h 45 min lost). Cells that did not fit by 10:00 are in `phase-17-STOPPED.json` | header, §4 |
| **153** | B2 pass rule: "a disjoint gain in recall at fixed precision" | **undefined for both Δ arms**: neither reaches even p = 0.70 on validation (max 0.569 / 0.662), so no fixed-precision point exists. They are recorded **WORSE** on the coverage frontier, below the incumbent at every coverage | §2.2 |
| **154** | B4: "drop the part relation and the 49,152 structurally dead parameters" | the two sets overlap in the last layer's part slice; removed in total: **163,840** (468,608 → 304,768). The lean encoder's function equals C3's `drop_relation part` arm (max \|diff\| 3e-8 on the real graph with copied weights; the supplier-dropped variant differs by 0.018) | §2.4 |
| **155** | B3: "report raw and recalibrated separately" | B3b's recalibrated figures use the project's 22-cell fill recalibrator (`fit_recalibration`, MM / VS chosen on validation log score) **on the spread 22-cell distribution**. That can move mass between cells inside a band, so "recalibrated 5-band" is not strictly 5-band | §2.3 |
| **156** | A2 / B1 / A1: scored from stored artifacts | the stored `phase15_sim` rows hold no snapshot column. Snapshots were recovered by **rebuilding the row keys and gating them equal** (B1 `rows`), which also identifies each row's t0 for A2 | §1.2, §2.1 |
| **157** | Stage C: "worktree clean" | `ml/` was clean for every run (enforced: `require_clean`). The worktree carries line-ending-only modifications to `db/gen_v8/*` and `db/schema.sql` (content identical; `generator_v8.py` SHA-1 equals the manifests' `71de78a`), left untouched because `db/gen_*` may not be modified | header |

## 7. Every `inventory_position_weekly` read

| reader | purpose |
|---|---|
| `phase17_b1.py rows` | **row filter only**: which part-plant-weeks exist (the `ok` filter `phase15_sim.py` applied). The label comes from `inventory_transactions` |
| `phase17_a.py a2` → `asof_position` | opening on-hand, safety stock and `open_po_qty` **as of each t0** (recorded_ts ≤ t0, asserted), to profile already-labelled rescued weeks |
| stored `phase15_sim` rows (A2, B1 scoring) | their labels were built by `phase14_sim.store_labels` (Phase 15); read here as stored arrays |

**Never a model feature.** B1a's features are `part_plant_features` (channel panel and graph counts). B1b reads the channel
panel. A1, A3, B2, B3 and B4 do not read the table.

## 8. What to build next, and what is now closed

**Build next: predict-the-rescue (B1) as the shortage product.** It is the only arm this phase that moved a decision. It
gives 0.82 precision at a quarter of the weeks planners actually acted on, against 0.62 for the simulation-based proxy
and 0.63 for a part-plant's own transfer history, on a 45% base rate, disjoint at five seeds. Its two known limits are
ceilings, not flaws: LightGBM hit its tree cap and the neural head its epoch cap, so both are floors. So the next step is
the three missing neural seeds and an un-capped LightGBM. That is budget, not design.

**Adopt the lean encoder (B4) for arrival work**: same metrics, 30.7% fewer parameters, and the same function as C3's ablation.
Its speed claim waits for an unthrottled machine.

**Closed:**
- **The shortage BCE head (A1):** 0.22 precision at 5% coverage on a realistic 3% base. Retire it; stop spending on it.
- **The Δ-strain capacity head (B2):** the trailing level is uncorrelated with the next 90 days on this data, so a Δ target
  adds noise, and capacity stays on the incumbent `mp h⁴`.
- **The 5-band fill head (B3):** it fixes the one cell it targeted and nothing in aggregate; fill stays a watchlist.
- **The channel-granularity simulation rebuild (A2):** missed rescues do not come from part-plants with more channels. The
  simulation's blind spot is drawdowns that develop late in the horizon from healthy starting stock. Any simulation fix
  should target that, not granularity.
- **Capacity horizon stacking (A4):** no 30 / 60 / 120-day labels exist.

**Signed-days lateness (A3)** is ready to feed an LLM explanation in the stated shape. It beats the channel-median constant,
ties LightGBM on median error, and its interval is honest but wide.
