# Phase 9A — fill's rolling-origin backtest and the arrival training-size confound

> **Verdict.**
>
> **Fill is not stable across windows, and on marginal calibration LightGBM-22 beats it.** Across 16 origin × world
> windows the recalibrated head's ECE-22 ranges **0.0111 to 0.1017 — a factor of 9**. Against LightGBM-22 under the
> identical recalibration protocol the head loses **12 of 16** windows, wins 2 and ties 2; against the deployable
> B5-flat-22 it loses 12 of 16. **Phase 7 binding statement 2 does not merely hold — it extends.** Phase 7 found
> LightGBM better calibrated on v6 only; across windows LightGBM wins **5 of 8 on v6 and 7 of 8 on v7**.
>
> **On the proper score the head wins almost everywhere:** exact CRPS beats LightGBM-22 in **15 of 16** windows and
> the model-free B2 histogram in **16 of 16**. Fill's case rests on the proper score, not on marginal calibration.
>
> **B2 is no longer the cheap replacement it looked like on the fixed split.** Phase 7 found it better than the head on
> marginal ECE in both worlds; across windows the head beats it in **9 of 16** and loses **5**.
>
> **The recalibration temperature is not transferable.** Vector scaling was selected in all 48 cells, with temperatures
> from **0.961 to 1.274**. The correction changes direction between windows — two windows need sharpening (T < 1) while
> the rest need softening — so **a temperature fitted on one year cannot be trusted for the next**.
>
> **The arrival margin is NOT driven by training history.** Origin 7 retrained with history cut from 44 snapshots to
> origin 1's 18 keeps its margin: **+0.0306 → +0.0303 (v6)** and **+0.0277 → +0.0257 (v7)**, bands overlapping the
> full-history cell, travelling **1% and 8%** of the way to origin 1's +0.0049 / +0.0014. **Phase 8's calendar reading
> stands and its median (+0.02) remains the honest forward-looking number** — the high end is not the figure to quote.
>
> `inventory_position_weekly` was never read. Phase 9 proper stays blocked.

## Header

| | |
|---|---|
| Commits | Code **`a259af6`**, committed before any cell ran: history truncation, per-origin fill baselines, baseline recalibration in the backtest scorer. **`cb24b17`** fixes the export naming defect found mid-phase (deviation 28) and adds the two analysis scripts. |
| Device | **MPS**, Apple Silicon, float32, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `num_workers=0`, torch 2.14.0; LightGBM on CPU in a torch-free process |
| Configuration | `ml/configs/shipped.json`, `phase6-shipped-1`, **unchanged**; no learning rate retuned |
| Cost | Stage A fill **42 cells, 7.7 h** wall on two queues, plus 176 baseline fits (45 min CPU). Stage B **6 cells, 1.0 h**. Scoring 1,552 score sets |
| Stamps | every Phase 9A bundle carries `code_dirty = false`: 42 fill bundles at `…-a259af6`, 6 truncated arrival bundles at `…-a259af6` |
| Rules | selection on validation only; bands per metric per configuration per world per origin, never borrowed; no LR retuned; protected files untouched; **`inventory_position_weekly` never read** (no file under `ml/` references it) |

---

## Stage A — fill across the rolling origins

Shipped fill: point-mass head, RPS loss, h⁰, lr 1.25e-4, 3 seeds. Origins 2–8 were trained here; origin 1 came from
Phase 8. Every baseline was refitted per origin on that origin's own folds, and **every arm — head and baseline alike —
was recalibrated by the head's own protocol, fitted on that origin's validation fold and never carried across origins**
(`ml/artifacts/backtest/phase9a_baseline_recal.json`, 176 files).

### A.1 — per origin, per world

Exact CRPS is the point-mass-aware integral, never the legacy formula. "Reliability of P(f=1)" is the conditional
reliability of the complete-fill point mass. Spread is max − min across 3 seeds.

| world | origin | evaluates | exact CRPS (recal) | ECE-22 raw | ECE-22 recalibrated | reliability of P(f=1) | method | temperature per seed |
|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 2022 H1 | 0.0574 (0.0000) | 0.1174 (0.0161) | **0.0310 (0.0036)** | 0.0162 (0.0015) | vs | 1.134 / 1.181 / 1.107 |
| v6 | 2 | 2022 H2 | 0.0617 (0.0002) | 0.0894 (0.0110) | **0.0186 (0.0023)** | 0.0096 (0.0021) | vs | 1.115 / 1.189 / 1.124 |
| v6 | 3 | 2023 H1 | 0.0465 (0.0001) | 0.0994 (0.0214) | **0.0361 (0.0010)** | 0.0157 (0.0013) | vs | 1.043 / 1.105 / 1.082 |
| v6 | 4 | 2023 H2 | 0.0516 (0.0001) | 0.0830 (0.0180) | **0.0128 (0.0013)** | 0.0087 (0.0030) | vs | 1.041 / 1.112 / 1.074 |
| v6 | 5 | 2024 H1 | 0.0442 (0.0000) | 0.0794 (0.0070) | **0.0111 (0.0011)** | 0.0060 (0.0007) | vs | 1.013 / 1.067 / 1.033 |
| v6 | 6 | 2024 H2 | 0.0610 (0.0002) | 0.0580 (0.0143) | **0.0403 (0.0016)** | 0.0210 (0.0022) | vs | 0.997 / 1.064 / 1.048 |
| v6 | 7 | 2025 H1 | 0.0624 (0.0000) | 0.0692 (0.0153) | **0.0341 (0.0021)** | 0.0170 (0.0015) | vs | 1.010 / 1.089 / 1.030 |
| v6 | 8 | 2025 Q3 | 0.0703 (0.0002) | 0.0867 (0.0158) | **0.0240 (0.0014)** | 0.0097 (0.0042) | vs | 0.961 / 1.037 / 0.983 |
| v7 | 1 | 2022 H1 | 0.0965 (0.0002) | 0.1173 (0.0420) | **0.0212 (0.0026)** | 0.0094 (0.0015) | vs | 1.205 / 1.231 / 1.176 |
| v7 | 2 | 2022 H2 | 0.1093 (0.0007) | 0.1206 (0.0514) | **0.0340 (0.0037)** | 0.0204 (0.0056) | vs | 1.226 / 1.274 / 1.238 |
| v7 | 3 | 2023 H1 | 0.0732 (0.0004) | 0.1331 (0.0372) | **0.0832 (0.0071)** | 0.0354 (0.0038) | vs | 1.075 / 1.088 / 1.092 |
| v7 | 4 | 2023 H2 | 0.0873 (0.0001) | 0.1169 (0.0408) | **0.0389 (0.0073)** | 0.0181 (0.0033) | vs | 0.997 / 1.042 / 1.039 |
| v7 | 5 | 2024 H1 | 0.0750 (0.0002) | 0.0972 (0.0341) | **0.0387 (0.0066)** | 0.0161 (0.0020) | vs | 0.973 / 1.025 / 0.988 |
| v7 | 6 | 2024 H2 | 0.1154 (0.0005) | 0.0992 (0.0172) | **0.1017 (0.0062)** | 0.0471 (0.0042) | vs | 0.992 / 1.051 / 1.025 |
| v7 | 7 | 2025 H1 | 0.1047 (0.0002) | 0.1151 (0.0181) | **0.0482 (0.0034)** | 0.0242 (0.0005) | vs | 0.997 / 1.050 / 1.018 |
| v7 | 8 | 2025 Q3 | 0.1190 (0.0005) | 0.1255 (0.0141) | **0.0418 (0.0007)** | 0.0201 (0.0053) | vs | 1.101 / 1.164 / 1.099 |

### A.2 — is fill's calibration stable, and is the temperature's window-dependence bounded?

**No to both.**

**Calibration is not stable.** Recalibrated ECE-22 spans **0.0111 (v6 o5) to 0.1017 (v7 o6)** — a **9.1×** range across
windows of one configuration. The worst windows are not the 2025 ones: v7 o6 (2024 H2) and v7 o3 (2023 H1) are the two
worst, at 0.1017 and 0.0832.

**Recalibration usually helps a lot, and once it hurt.** Raw ECE-22 runs 0.058–0.133 and recalibration improves it in
**15 of 16** windows, often by 3–6×. In **v7 o6 it made marginal calibration worse** (0.0992 raw → 0.1017 recalibrated):
the vector scaling that minimises validation log score can cost marginal calibration on the evaluation window.

**The temperature is not transferable.**

| world | o1 | o2 | o3 | o4 | o5 | o6 | o7 | o8 |
|---|---|---|---|---|---|---|---|---|
| v6 | 1.140 | 1.143 | 1.076 | 1.076 | 1.038 | 1.036 | 1.043 | **0.994** |
| v7 | 1.204 | 1.246 | 1.085 | 1.026 | **0.995** | 1.023 | 1.022 | 1.121 |

(per-origin mean of 3 seeds; full per-seed values in A.1)

- **Full range across all 8 origins: 0.961–1.274** (v6 0.961–1.189, v7 0.973–1.274). Origin 1's 1.107–1.231 was **not**
  the extreme, and the fixed split's 1.008–1.089 was a narrow sample of a much wider spread.
- **Vector scaling was selected in all 48 cells.** The method is stable; its parameter is not.
- **The correction reverses direction.** v6 o8 and v7 o5 need **T < 1** — sharpening — while every other window needs
  softening. A stale temperature does not merely under-correct there; it corrects the wrong way.
- **Year-to-year movement is larger than seed noise.** The largest change between consecutive origins is 0.066 (v6) and
  **0.161** (v7, o2 → o3), against a within-window seed spread of at most 0.079 / 0.066.

**Answer: a temperature fitted on one year cannot be trusted for the next.** It must be refitted per window, which is
what the shipped loop already does on each origin's own validation slice — this quantifies why that matters.

### A.3 / A.4 — against LightGBM-22 and the model-free B2, same protocol

Marginal ECE-22, recalibrated, all arms:

| world | origin | head (3 seeds) | LightGBM-22, identity (5 fits) | B5-flat-22 (5 fits) | B2 as-of | head vs LGBM-22 | head vs B5-flat | head vs B2 |
|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 0.0310 (0.0036) | 0.0265 (0.0005) | 0.0261 (0.0006) | 0.0294 | **baseline** | **baseline** | not dist. |
| v6 | 2 | 0.0186 (0.0023) | 0.0232 (0.0007) | 0.0236 (0.0008) | 0.0199 | **head** | **head** | **head** |
| v6 | 3 | 0.0361 (0.0010) | 0.0346 (0.0008) | 0.0347 (0.0007) | 0.0485 | **baseline** | **baseline** | **head** |
| v6 | 4 | 0.0128 (0.0013) | 0.0098 (0.0005) | 0.0097 (0.0003) | 0.0084 | **baseline** | **baseline** | **baseline** |
| v6 | 5 | 0.0111 (0.0011) | 0.0106 (0.0008) | 0.0106 (0.0002) | 0.0188 | not dist. | not dist. | **head** |
| v6 | 6 | 0.0403 (0.0016) | 0.0347 (0.0007) | 0.0357 (0.0009) | 0.0449 | **baseline** | **baseline** | **head** |
| v6 | 7 | 0.0341 (0.0021) | 0.0279 (0.0006) | 0.0277 (0.0008) | 0.0335 | **baseline** | **baseline** | not dist. |
| v6 | 8 | 0.0240 (0.0014) | 0.0241 (0.0010) | 0.0250 (0.0008) | 0.0378 | not dist. | **head** | **head** |
| v7 | 1 | 0.0212 (0.0026) | 0.0193 (0.0005) | 0.0193 (0.0003) | 0.0297 | **baseline** | **baseline** | **head** |
| v7 | 2 | 0.0340 (0.0037) | 0.0245 (0.0010) | 0.0247 (0.0015) | 0.0324 | **baseline** | **baseline** | **baseline** |
| v7 | 3 | 0.0832 (0.0071) | 0.0930 (0.0011) | 0.0953 (0.0005) | 0.1344 | **head** | **head** | **head** |
| v7 | 4 | 0.0389 (0.0073) | 0.0196 (0.0005) | 0.0192 (0.0009) | 0.0175 | **baseline** | **baseline** | **baseline** |
| v7 | 5 | 0.0387 (0.0066) | 0.0293 (0.0005) | 0.0282 (0.0009) | 0.0298 | **baseline** | **baseline** | **baseline** |
| v7 | 6 | 0.1017 (0.0062) | 0.0935 (0.0010) | 0.0954 (0.0016) | 0.1391 | **baseline** | **baseline** | **head** |
| v7 | 7 | 0.0482 (0.0034) | 0.0202 (0.0008) | 0.0200 (0.0006) | 0.0276 | **baseline** | **baseline** | **baseline** |
| v7 | 8 | 0.0418 (0.0007) | 0.0294 (0.0005) | 0.0289 (0.0009) | 0.0588 | **baseline** | **baseline** | **head** |

| comparison | head better | not distinguishable | baseline better |
|---|---|---|---|
| vs **LightGBM-22 (identity codes)** — Phase 7 binding statement 2's object | **2** | 2 | **12** |
| vs B5-flat-22 (no identity keys, deployable) | 3 | 1 | **12** |
| vs **B2 as-of histogram** (model-free) | **9** | 2 | 5 |

Per world, against LightGBM-22: **v6 head 1 / tie 2 / LightGBM 5**, **v7 head 1 / tie 0 / LightGBM 7**.

**Phase 7 binding statement 2 holds and extends, and this is the result that favours the baseline.** Phase 7 found
LightGBM-22 better calibrated on v6 and claimed parity only for v7. Across 8 windows per world, **LightGBM-22 is better
calibrated than the shipped head in 12 of 16 windows, and on v7 it wins 7 of 8** — the world where Phase 7's Addendum A
claim survived. The head wins marginal calibration in exactly two windows (v6 o2, v7 o3). **Fill's marginal-calibration
claim is not defensible in either world.**

**B2 has moved the other way.** On the fixed split the model-free histogram beat the head on marginal ECE in both
worlds. Across windows the head wins 9 of 16 and loses 5 (v6 o4; v7 o2, o4, o5, o7). **B2 is still not dominated**, but
"the cheapest thing that could replace the head" is no longer supported as a general statement — it is a v7-leaning
result.

**On the proper score the ordering reverses.** Exact CRPS, same protocol:

| world | origin | head exact CRPS | LightGBM-22 | B5-flat-22 | B2 as-of | head vs LGBM-22 | head vs B2 |
|---|---|---|---|---|---|---|---|
| v6 | 1 | **0.0574 (0.0000)** | 0.0584 (0.0001) | 0.0581 (0.0002) | 0.0601 | **head** | **head** |
| v6 | 2 | **0.0617 (0.0002)** | 0.0626 (0.0001) | 0.0626 (0.0002) | 0.0647 | **head** | **head** |
| v6 | 3 | **0.0465 (0.0001)** | 0.0466 (0.0001) | 0.0467 (0.0000) | 0.0478 | **head** | **head** |
| v6 | 4 | **0.0516 (0.0001)** | 0.0525 (0.0001) | 0.0525 (0.0000) | 0.0549 | **head** | **head** |
| v6 | 5 | **0.0442 (0.0000)** | 0.0445 (0.0001) | 0.0446 (0.0000) | 0.0458 | **head** | **head** |
| v6 | 6 | **0.0610 (0.0002)** | 0.0617 (0.0001) | 0.0617 (0.0001) | 0.0644 | **head** | **head** |
| v6 | 7 | **0.0624 (0.0000)** | 0.0635 (0.0001) | 0.0634 (0.0001) | 0.0660 | **head** | **head** |
| v6 | 8 | **0.0703 (0.0002)** | 0.0710 (0.0001) | 0.0713 (0.0002) | 0.0743 | **head** | **head** |
| v7 | 1 | **0.0965 (0.0002)** | 0.0980 (0.0001) | 0.0979 (0.0002) | 0.1010 | **head** | **head** |
| v7 | 2 | **0.1093 (0.0007)** | 0.1114 (0.0006) | 0.1112 (0.0003) | 0.1191 | **head** | **head** |
| v7 | 3 | **0.0732 (0.0004)** | 0.0740 (0.0002) | 0.0740 (0.0002) | 0.0783 | **head** | **head** |
| v7 | 4 | **0.0873 (0.0001)** | 0.0888 (0.0001) | 0.0889 (0.0002) | 0.0969 | **head** | **head** |
| v7 | 5 | **0.0750 (0.0002)** | 0.0761 (0.0001) | 0.0760 (0.0000) | 0.0793 | **head** | **head** |
| v7 | 6 | **0.1154 (0.0005)** | 0.1187 (0.0002) | 0.1184 (0.0002) | 0.1292 | **head** | **head** |
| v7 | 7 | **0.1047 (0.0002)** | 0.1053 (0.0001) | 0.1054 (0.0001) | 0.1099 | **head** | **head** |
| v7 | 8 | **0.1190 (0.0005)** | 0.1185 (0.0001) | 0.1187 (0.0001) | 0.1288 | **baseline** | **head** |

The head wins exact CRPS in **15 of 16** against LightGBM-22 (losing only v7 o8, by 0.0005) and **16 of 16** against B2.
**So fill's defensible claim is the proper score, not calibration**: the head's distribution is sharper where it counts,
while its marginal bin probabilities are worse calibrated than a LightGBM's in three windows out of four.

---

## Stage B — the arrival training-size confound

Phase 8 §4 3d read arrival's margin variation as calendar-driven. Later origins also carry more training history, so
across the 8 existing cells **year and history length are collinear at ρ = 1.00** — the reading was not separable from
the data that produced it.

### B.1 / B.2 — origin 7 retrained with origin 1's history length

Training truncated to the **most recent 18 snapshots** (72,000 rows, exactly origin 1's size), validation and evaluation
windows unchanged, 3 seeds, both worlds, epoch cap 200. A separate configuration with its own bundle, its own
prediction files and its own band — it never borrows origin 7's.

| world | cell | training snapshots | lateness ROC-AUC | promise date | margin | × seed spread | clears promise CI | C-index |
|---|---|---|---|---|---|---|---|---|
| v6 | origin 7, full | 44 | 0.7906 (0.0013) | 0.7600 | **+0.0306** | 22.7× | yes | 0.6703 (0.0013) |
| v6 | **origin 7, truncated** | **18** | 0.7903 (0.0016) | 0.7600 | **+0.0303** | 18.4× | yes | 0.6660 (0.0006) |
| v6 | origin 1 | 18 | 0.7498 (0.0015) | 0.7449 | +0.0049 | 3.3× | no | 0.6833 (0.0007) |
| v7 | origin 7, full | 44 | 0.7838 (0.0031) | 0.7561 | **+0.0277** | 8.8× | yes | 0.6735 (0.0009) |
| v7 | **origin 7, truncated** | **18** | 0.7818 (0.0024) | 0.7561 | **+0.0257** | 10.6× | yes | 0.6683 (0.0012) |
| v7 | origin 1 | 18 | 0.7575 (0.0012) | 0.7561 | +0.0014 | 1.2× | no | 0.6727 (0.0023) |

- **The margin does not collapse.** It keeps **99%** of the full-history margin on v6 and **93%** on v7, moving only
  **1%** and **8%** of the way toward origin 1's value.
- **Truncated and full are not distinguishable on lateness ROC-AUC.** The seed ranges overlap in both worlds
  (v6 0.7894–0.7910 against 0.7901–0.7915; v7 0.7802–0.7827 against 0.7819–0.7851).
- **Both truncated cells still clear the promise baseline's bootstrap interval**, the stricter test that origin 1 failed
  in both worlds.
- **History does buy something, and it is not this.** C-index falls by a small but **disjoint** amount when history is
  cut — 0.6703 → 0.6660 (v6) and 0.6735 → 0.6683 (v7). More history helps rank arrivals slightly; it does not enlarge
  the lateness margin.

### B.3 — the two correlations, and why neither is evidence

Across the 8 existing full-history cells:

| predictor of the margin | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| training snapshots | +0.80 (0.017) | +0.88 (0.004) |
| calendar year | +0.80 (0.018) | +0.88 (0.004) |
| **training snapshots vs calendar year** | **+1.00** | **+1.00** |

**Both correlations are the same correlation.** The two predictors are perfectly rank-collinear by construction, so
neither of these numbers is evidence for either explanation, and the near-identical coefficients are an artefact of the
design rather than a finding. **B.1 is the test**, and it separates them: holding the evaluation window fixed and
removing 26 snapshots of history changed the margin by 1–8%.

### B.4 — verdict, and what it does to the client number

**The driver is the window, not training history. Phase 8's calendar reading stands, and its median remains the
forward-looking number.**

- **The margin is a property of the evaluation window.** An 18-snapshot model evaluated on 2025 H1 earns +0.030 / +0.026;
  an 18-snapshot model evaluated on 2022 H1 earns +0.005 / +0.001. Same history length, 6–20× different margin.
- **Consequence for the client figure: quote the median, +0.02, not the high end.** More history does not buy a larger
  lateness margin, so an extract with several years of history has no entitlement to the +0.03 windows. §5's arrival row
  below keeps the median and the full +0.001 to +0.031 range.
- **What Stage B does not separate.** Truncation holds history *length* fixed while keeping the most *recent* 18
  snapshots, so "the window the model is evaluated on" and "the era the training data comes from" remain bundled. Both
  are ruled out as *quantity* effects; neither is ruled in as the mechanism. Distinguishing them needs an origin-7 model
  trained on origin 1's *calendar* window, which is a different experiment and was not run (deviation 30).

---

## Updated Phase 8 §5 rows

Replacing the fill and arrival rows of `reports/phase-8.md` §5. Capacity and shortage are unchanged.

| task | ship | confidence | period the claim is defensible over | range / restriction |
|---|---|---|---|---|
| **fill_rate** | **h⁰ (lr 1.25e-4), recalibrated per window**, unchanged | **Moderate for the proper score; none for marginal calibration** | **All 8 origins × 2 worlds (2022 H1 → 2025 Q3), now backtested** | **Quotable:** exact CRPS beats LightGBM-22 in 15 of 16 windows and the model-free B2 in 16 of 16. **Not quotable:** marginal calibration — **LightGBM-22 is better calibrated in 12 of 16 windows** (v7 7 of 8), so Phase 7 binding statement 2 extends to both worlds; and B2 beats the head in 5 of 16. **Recalibrated ECE-22 varies 9× across windows (0.011–0.102)**, so no single calibration figure may be quoted. **The recalibration temperature (0.961–1.274, vector scaling in all 48 cells) must be refitted per window** and reverses direction between windows |
| **arrival_week** | **h⁴ SHARE-lite (lr 2.5e-4) for ranking, h⁰ recalibrated for the served distribution**, unchanged; remove the drift gate in Phase 10 | **Moderate for the direction of the lateness claim; low for its size; none for ranking** | Origins 1, 2, 6, 7 in both worlds plus the 2025 fixed split; 2023 and 2024 H1 untested | **Quotable:** the head adds lateness information beyond the promise date — 8 of 8 windows, every seed, beating LightGBM in all 8. **The forward-looking figure is the median, +0.02, over the range +0.001 to +0.031** — Phase 9A Stage B shows the size is set by the evaluation window, **not** by how much training history an extract has, so more history does not entitle a client to the +0.03 windows. In 3 of 8 windows the margin is inside the promise baseline's bootstrap interval. **Never quotable:** that the head ranks arrivals better than the promise date (it loses by 0.18–0.22 C-index in 8 of 8; truncating history costs a further 0.004–0.005) |

---

## Deviations

Appended to the known-deviations index in `docs/implementation_guide.md`, continuing from row 26 (the guide's index is authoritative; `reports/phase-8.md` §6 row 25 was renumbered to 26 to match it).

| # | the guide / specification says | measured, and what Phase 9A did |
|---|---|---|
| 27 | fill's rolling-origin backtest is deferred and unscheduled (deviation 21) | **run here**: origins 2–8, both worlds, 3 seeds, 42 cells, 7.7 h. Deviation 21's fill clause is **discharged**; fill now has all 8 origins. Shortage remains unqueued |
| 28 | a bundle's predictions and index entry are identified by task, world, origin, architecture, depth, learning rate and seed | **they were not unique.** `backtest.py export` built the prediction filename and index key without the truncation suffix that `loop.bundle_dir` uses, so Stage B's truncated cells **overwrote origin 7's full-history arrival predictions and index entries**. Caught by comparing against Phase 8's recorded values (head ROC 0.7906 → 0.7903). Fixed in `export`, all 198 bundles re-exported and all 1,552 score sets recomputed; **the full-history origin-7 figures reproduce Phase 8 exactly** (+0.0306 / +0.0277). No Phase 8 conclusion changes |
| 29 | B2 is the as-of 52-week histogram of the channel's active weeks | at a rolling origin a channel may have **no** active week in the trailing 52; those rows fall back to **that origin's training-fold global 22-cell CDF**, the same rule the fixed split used |
| 30 | Stage B separates training history from the calendar | it separates **quantity** only. The truncated cell keeps the most **recent** 18 snapshots, so the era of the training data moves with the evaluation window. "More history" is ruled out; "recent history vs the evaluation window itself" is **not separated**, and would need an origin-7 model trained on origin 1's calendar window |

---

## Open items

1. **Fill's marginal calibration is worse than a LightGBM's in three windows out of four.** Either the fill claim is
   restated as a proper-score claim, or the head's binned probabilities need work. Not a configuration change to make
   from evaluation evidence; it belongs with the Phase 10 decisions.
2. **Recalibration can hurt marginal calibration** (v7 o6). The selection rule minimises validation log score, which is
   not the metric being quoted. Worth a per-window selection between MM and VS on the metric that will be reported.
3. **Fill's recalibrated ECE varies 9× across windows.** Any fill calibration number given to a client must name its
   window.
4. **Stage B's remaining ambiguity** (deviation 30): quantity of history is ruled out; recency versus the evaluation
   window is not separated.
5. **Arrival origins 3, 4 and 5 remain untrained**, so 2023 and 2024 H1 are still untested for 3d.
6. **Unchanged from Phase 8 §7:** deviation 20's validation-outcome overlap bounds every number here too; 3 seeds
   against the specification's 5; capacity intervals not quotable; the drift gate decided but not removed.
7. **Phase 9 proper stays blocked.** `inventory_position_weekly` was not read in Phase 9A — no file under `ml/`
   references it, in code or in any artifact written by this phase.
