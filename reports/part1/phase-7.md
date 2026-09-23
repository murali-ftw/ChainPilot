# Phase 7 — Baselines (guide 7.1–7.3)

> **Verdict.** The baseline set is complete, scored by the bundle scorer on asserted-identical rows, with refit bands measured
> per configuration per world. **Fill:** on v7 the served head beats LightGBM-22 on marginal ECE and exact CRPS; on v6
> LightGBM-22 is better calibrated (+0.0024 ECE under the same protocol, disjoint) while the head wins exact CRPS —
> **Addendum A's v6 parity is withdrawn**. On marginal ECE the head does not beat the model-free as-of histogram (B2) in either
> world. **Arrival:** the head beats every learned and naive baseline on C-index and lateness ROC-AUC, but **the promise
> date alone orders arrivals far better — C-index 0.883 / 0.877 against 0.673 / 0.677**; the head adds +0.033 / +0.025
> lateness ROC-AUC over it. **Capacity v6:** the shipped depth-4 head is not distinguishable from B5 LightGBM or from the
> non-deployable per-supplier floor; on v7 it beats both. **Gate: Phase 8 may start**, carrying §9 forward.

## 1. Header

| | |
|---|---|
| **Commit** | **`d1ede34`** — *Phase 6: training loop, bundles, drift monitor*, on `HADES-v4-ml-pipeline`. Made at Stage 0a, before any Phase 7 measurement. **The 22 Phase 6 bundles predate it and keep their `+dirty` stamps**; they were not rebuilt. |
| Device | LightGBM on CPU (`n_jobs=6`); scoring, recalibration and drift in a separate torch process on MPS/CPU. The two libraries are never in one process: on this machine whichever runs second segfaults (Phase 5, re-confirmed in Phase 6). |
| Population | fixed split train ≤ 2023 / val 2024 / test 2025; fit window 2019–2025; **row identity asserted against the Phase 6 bundles for every baseline prediction file** |
| Wall-clock | fitting **581 s** (108 baselines → 216 prediction files, one process, torch never imported); scoring **122 s** (row identity on 216 files, 20 recalibration fits, 128 test-fold score sets with intervals, drift, ablation); band, C-index decomposition and interval read-outs under a minute each. **Total ≈ 12 min** |
| LightGBM fit time — guide 7.2 verify (< 120 s) | **11.0–11.1 s** per LightGBM-22 fill fit (the instrumented raw arm, 10 fits). Every fit early-stops on the 2024 fold (patience 40, at most 400 trees): fill 21–38 trees, arrival 86–151, shortage 91–140. **Pass** |
| Scorer | `phase5_metrics` — the functions `loop.test_metrics` used for the Phase 6 bundles; exact CRPS for fill; 95% percentile bootstrap, 1,000 resamples |
| Row identity | asserted on all 216 files: labels equal the Phase 6 reference bundle's on the same fold (arrival, fill, capacity) and entity order equals `model_outputs`; shortage labels equal Phase 5 run10's. Test rows 36,000 / 36,000 / 22,500 / 13,500 |
| Fix during the phase | `bands()` in the scorer silently dropped every metric (scores are tuples; it tested for lists). Fixed, and bands regenerated from the stored scores — **no score was recomputed** |

---

## 2. Inventory — what already existed, before any code was written

Phase 5 built the capacity floors and the LightGBM-20/22 fill fits; `run9_preds` holds six older baseline prediction
sets whose generating script **is not in the repository**. Every row below was scored earlier on the same test rows
the models use (36,000 arrival and fill, 22,500 capacity, 13,500 shortage — re-asserted this phase).

| task | baseline | where | script in repo? | test score (v6 / v7) | features |
|---|---|---|---|---|---|
| arrival | "naive" | `run9_preds/BASE_*_arrival_week_naive` | **no** — definition not recorded in any report | C-index 0.5507 / 0.5574; lateness ROC-AUC 0.7064 / 0.7097 | unknown |
| arrival | LightGBM regressor | `run9_preds/BASE_*_arrival_week_lightgbm` | **no** | C-index 0.6428 / 0.6481; ROC-AUC 0.7570 / 0.7568 | as-of channel + static + **identity codes** (per `learnability_windowed.py`, the only surviving template) |
| arrival | promise week alone | measured ad hoc in Phase 5 §7 | — | ROC-AUC **0.7482 / 0.7498** | promise date |
| fill | "naive" | `run9_preds/BASE_*_fill_rate_naive` | **no** — definition not recorded | exact CRPS 0.0792 / 0.1229; ECE20 0.0152 / 0.0324 | unknown |
| fill | LightGBM-20 (the "0.0107" fit) | `run9_preds/BASE_*_fill_rate_lightgbm` | **no** | exact CRPS 0.0759 / 0.1140; ECE20 **0.0107 / 0.0141** | as above |
| fill | LightGBM-20, refit | `phase5_preds/BASE_*_fill_rate_lightgbm20` | `phase5_baselines.py` | exact CRPS 0.0758 / 0.1138; ECE20 0.0144 / 0.0161 | as-of + static + identity codes |
| fill | LightGBM-22 | `phase5_preds/BASE_*_fill_rate_lightgbm22` | `phase5_baselines.py` | exact CRPS 0.0623 / 0.1025; ECE20 0.0129 / 0.0163 | as-of + static + identity codes |
| fill | LightGBM-22 + validation MM/VS | `phase5_preds_recal/` | `phase5_recal.py` | ECE20 **0.0187 / 0.0271** (VS selected) | same, **one fit** |
| capacity | naive global / per-supplier / per-channel P10–P50–P90 | `phase5_preds/BASE_*_capacity_strain_naive_*` | `phase5_baselines.py` | mean pinball 0.0726 / 0.1084; 0.0515 / 0.0954; 0.0587 / 0.1079 | — (two identity-keyed) |
| capacity | LightGBM quantile ×3 | `phase5_preds/BASE_*_capacity_strain_lightgbm` | `phase5_baselines.py` | mean pinball 0.0520 / 0.0770 | as-of + static, **no identity codes** |
| shortage | naive per-part-plant historical rate | `run9_preds/BASE_*_shortage_qty_naive` | **no** (named in `phase1_2.md` §11) | PR-AUC 0.4896 / 0.2487 | identity-keyed |
| shortage | LightGBM | `run9_preds/BASE_*_shortage_qty_lightgbm` | **no** | PR-AUC 0.5287 / 0.4438 | unknown |

### What was genuinely missing

1. **Any refit band.** Every LightGBM number above is **one fit**. The fill-parity claim Phase 6 downgraded rests on one.
2. **The same-protocol arm as a band** — Addendum A recalibrated a single LightGBM-22 fit.
3. **Promise-date-only as a formal baseline**, scored on both arrival metrics.
4. **A calibration floor for arrival.** Guide B3 ("always week 1") is vacuous here; nothing replaced it.
5. **Guide B2** (as-of rolling fill) and **per-channel / per-lane arrival medians**, explicitly defined.
6. **Guide B5** — LightGBM on **flattened graph features without identity keys**, for any task. Every fill, arrival and
   shortage LightGBM above used identity codes, which a model shipped to an unseen Rane entity cannot use.
7. **Scripts** for the six `run9` baselines. They stay in the tables as unscripted legacy entries; every variant is
   rebuilt explicitly rather than guessed at.
8. **Validation predictions for almost every baseline**, so no baseline drift had ever been measured.
9. **The guide 7.3 ablation ratio** — the h⁰ and full cells exist; the ratio was never computed.

**Built in this phase:** `ml/baselines/phase7_fit.py` (every fit and floor, torch-free), `ml/eval/phase7_score.py`
(row identity, scoring, recalibration arms, bands, drift, ablation).

---

## 3. Guide reconciliation

| # | step | the guide specifies | the data / Phases 5–6 require | resolution |
|---|---|---|---|---|
| 1 | 7.1 B3 | "always predict w = 1", covering 82.2% of arrivals | **week 1 never occurs**: observed arrivals are 2–12 weeks from the snapshot | **code**: two replacement floors — the training-fold marginal 13-cell distribution, and promise-date-only. **Guide annotated** |
| 2 | 7.1 B2, capacity | point = `fill_rate_last13` | capacity's label is **utilisation** clipped at 3.0, not fill | **code**: empirical P10/P50/P90 floors (global, per-supplier, per-channel; latter two non-deployable). **Guide annotated** |
| 3 | 7.1 B2, fill | band = spread of the last 52 weekly `fill_rate` values | `fill_rate` is forward-filled across idle weeks, so those values are mostly repeats | **code**: as-of 22-cell distribution of the channel's own **active-week** fill in the 52 weeks to t₀. **Guide annotated** |
| 4 | 7.1 B4 | drift quantile buckets | no demand-drift head ships | **out of scope; guide annotated** |
| 5 | 7.2 statistics | channels per supplier median 12, P90 56, max 168; per part 2; per group 4 | measured **38 / 46 / 55**; **26**; **5**; 9.4% of part × plant sole-sourced | **guide annotated** |
| 6 | 7.2 features | flatten the graph into scalars | earlier LightGBMs used **identity codes** instead | **code**: B5 built as specified, no identity keys; the identity-keyed LightGBM-22 kept and labelled non-deployable, because Addendum A's claim was made against it. **Guide annotated** |
| 7 | 7.2 losses | "the same losses" | holds for fill (22-class) and capacity (quantile); **arrival's LightGBM is a point regressor on observed rows** | labelled wherever compared. **Guide annotated** |
| 8 | 7.2 process | — | LightGBM and torch cannot share a process | **code**: fits in a torch-free process, scoring in another; torch absence asserted |
| 9 | recalibration | *silent* | a baseline compared to a recalibrated head must go through the **same** fit-and-select-on-validation protocol | **code**: `loop.fit_recalibration` applied to every LightGBM-22 fit (§4). **Guide annotated** |
| 10 | scoring | *silent* | the **bundle scorer**, identical rows, 1,000-resample intervals; **exact CRPS** for fill | **code**: `phase5_metrics` (what `loop.test_metrics` uses); row identity asserted per file. **Guide annotated** |
| 11 | 7.3 ablation | run the full pipeline with zero layers; publish (full − h⁰) / (full − random) | the h⁰ and full cells already exist (Addendum B, Phase 6); retraining is forbidden | **computed from existing cells**; "random" = the naive global floor for loss metrics. **Guide annotated** |
| 12 | file names | `rolling.py`, `lgbm.py`, `ablation.py` | one torch-free fitting process and one scoring process | **code**: `phase7_fit.py`, `phase7_score.py` |

**Guide diff.** Annotation blocks after the verify steps of **7.1**, **7.2** and **7.3**; three rows (**15–17**) in
the known-deviations index. No other guide text changed.

---

## 4. The LightGBM-22 refit band — five fits × three arms × two worlds

LightGBM-22 on **Addendum A's exact feature set** (as-of channel row, static attributes, identity codes — the model its
parity claim was made against), test 2025, 36,000 rows per world. Each arm is five fits:

| arm | what it isolates |
|---|---|
| **raw** | seed 7, refitted five times — is the figure earlier reports quoted even reproducible? |
| **refit-only** | seeds 7 / 17 / 27 / 37 / 47, no recalibration — fit noise alone |
| **validation-fitted** | the same five fits put through **the neural head's own protocol**: moment matching or vector scaling fitted on 2024, method selected on validation log score (`loop.fit_recalibration`) |

Guide B5 — the same model with **no identity codes** and flattened graph counts instead — is reported beside it.

### v6

| arm | ECE 20-bin, five fits | mean | **spread** | sd | ECE 22-cell mean | exact CRPS mean (spread) | reliability P(f=1) mean |
|---|---|---|---|---|---|---|---|
| raw | 0.01292 · 0.01292 · 0.01292 · 0.01292 · 0.01292 | 0.01292 | **0.00000** | 0 | 0.01477 | 0.06230 (0) | 0.0077 |
| refit-only | 0.01292 · 0.01250 · 0.01338 · 0.01306 · 0.01323 | 0.01302 | **0.00088** | 0.00034 | 0.01493 | 0.06235 (0.00013) | 0.0070 |
| **validation-fitted** | 0.01874 · 0.01909 · 0.01927 · 0.01877 · 0.01850 | **0.01887** | **0.00077** | 0.00030 | 0.01986 | 0.06248 (0.00008) | 0.0115 |
| B5, refit-only | 0.01215 · 0.01320 · 0.01290 · 0.01299 · 0.01288 | 0.01282 | 0.00105 | 0.00040 | 0.01479 | 0.06232 (0.00004) | 0.0075 |
| B5, validation-fitted | 0.01900 · 0.01919 · 0.01936 · 0.01896 · 0.01926 | 0.01915 | 0.00040 | 0.00017 | 0.02011 | 0.06246 (0.00002) | 0.0120 |

### v7

| arm | ECE 20-bin, five fits | mean | **spread** | sd | ECE 22-cell mean | exact CRPS mean (spread) | reliability P(f=1) mean |
|---|---|---|---|---|---|---|---|
| raw | 0.01633 × 5 | 0.01633 | **0.00000** | 0 | 0.01825 | 0.10251 (0) | 0.0096 |
| refit-only | 0.01633 · 0.01572 · 0.01609 · 0.01592 · 0.01568 | 0.01595 | **0.00065** | 0.00027 | 0.01795 | 0.10249 (0.00007) | 0.0082 |
| **validation-fitted** | 0.02715 · 0.02648 · 0.02677 · 0.02624 · 0.02612 | **0.02655** | **0.00102** | 0.00042 | 0.02679 | 0.10256 (0.00007) | 0.0110 |
| B5, refit-only | 0.01637 · 0.01619 · 0.01675 · 0.01631 · 0.01618 | 0.01636 | 0.00057 | 0.00023 | 0.01824 | 0.10262 (0.00013) | 0.0074 |
| B5, validation-fitted | 0.02656 · 0.02655 · 0.02623 · 0.02632 · 0.02625 | 0.02638 | 0.00033 | 0.00016 | 0.02661 | 0.10269 (0.00013) | 0.0106 |

### What the band shows

**1. The "±0.003 refit noise" that justified "not distinguishable" does not exist.** Five refits at seed 7 produce
**bitwise-identical predictions** in both worlds: LightGBM is deterministic at a fixed seed. Across five *different*
seeds its ECE moves by **0.0009 (v6) and 0.0007 (v7)**, a quarter of what was assumed. The 0.0107 → 0.0144 gap that
Phases 5 and 6 read as refit noise was two different models — the unscripted run-9 LightGBM-20 and
`phase5_baselines.py`'s — not one model refitted.

**2. Validation recalibration worsens LightGBM on every one of ten fits.** ECE 20-bin rises by **+0.0059 on v6** and
**+0.0106 on v7**, while exact CRPS barely moves (+0.0001). This confirms Addendum A's single-fit observation as a
property of the model, not an accident of one seed. §8 measures why.

**3. Identity codes buy nothing.** B5, with no identity keys and flattened graph counts instead, lands inside
LightGBM-22's own spread in every arm in both worlds. The deployable baseline is as strong as the non-deployable one.

**4. Seed-7 anchor.** Refit-only seed 7 is exactly the raw figure (0.01292 / 0.01633), and it is Addendum A's
LightGBM-22 (0.0129 / 0.0163) — so this band sits on the model the earlier claims used.

---

## 5. The fill parity answer — supersedes Addendum A's claim and Phase 6 §8's downgrade

The neural point-mass head is h⁰ with its validation recalibration; three seeds per world (Phase 6 bundles). LightGBM-22
is five seeds per arm. **Every margin is read against the spread measured for that configuration, in that world, and
against both sides' spreads.**

### v6 — **the head does not match LightGBM-22 on calibration. LightGBM is better.**

| comparison, ECE 20-bin ↓ | head (3 seeds) | LightGBM-22 (5 seeds) | gap | vs LightGBM spread | vs head spread | ranges overlap? |
|---|---|---|---|---|---|---|
| **same protocol** (both validation-fitted) | **0.02124** [0.0203–0.0224] | **0.01887** [0.0185–0.0193] | **+0.00237** | 3.1× | 1.1× | **no** |
| head recalibrated vs LightGBM as it comes | 0.02124 | 0.01302 [0.0125–0.0134] | +0.00822 | 9.4× | 3.9× | no |

| comparison, exact CRPS ↓ | head (3 seeds) | LightGBM-22 (5 seeds) | gap | ranges overlap? |
|---|---|---|---|---|
| **served head (recalibrated) vs LightGBM, same protocol** | **0.06161** [0.06159–0.06163] | 0.06248 [0.06243–0.06250] | **−0.00087** | **no — head better** |
| served head vs LightGBM as it comes | 0.06161 | 0.06235 [0.06230–0.06243] | −0.00074 | no — head better |
| head before recalibration vs LightGBM as it comes | 0.06184 [0.06180–0.06188] | 0.06235 | −0.00051 | no — head better |

| comparison, conditional reliability of P(f = 1) ↓ | head | LightGBM-22 | gap | ranges overlap? |
|---|---|---|---|---|
| same protocol | 0.00965 [0.0085–0.0108] | 0.01147 [0.0104–0.0127] | −0.0018 | **yes — not distinguishable** |
| served head vs LightGBM as it comes | 0.00965 | 0.00697 [0.0063–0.0077] | +0.0027 | no — LightGBM better |

Under the identical protocol, **every head seed is less well calibrated than every LightGBM-22 seed**, and the gap
clears both spreads. It is small — 0.0024 of ECE — but it is a difference, not noise. **Addendum A's "parity with LightGBM
on v6" is withdrawn, and Phase 6's "not distinguishable" is resolved: on v6 LightGBM-22 is the better-calibrated fill
distribution**, by 0.0024 under the same protocol and 0.0082 as it comes. **On the proper score the head wins**: its
exact CRPS is lower than every LightGBM seed's.

### v7 — **Addendum A's "closed" survives, and the head beats LightGBM-22.**

| comparison, ECE 20-bin ↓ | head (3 seeds) | LightGBM-22 (5 seeds) | gap | vs LightGBM spread | vs head spread | ranges overlap? |
|---|---|---|---|---|---|---|
| **same protocol** | **0.01434** [0.0137–0.0150] | **0.02655** [0.0261–0.0272] | **−0.01221** | 11.9× | 9.2× | **no** |
| head recalibrated vs **LightGBM's best arm** (as it comes) | 0.01434 | 0.01595 [0.0157–0.0163] | **−0.00160** | 2.5× | 1.2× | **no** |

| comparison, exact CRPS ↓ | head (3 seeds) | LightGBM-22 (5 seeds) | gap | ranges overlap? |
|---|---|---|---|---|
| **served head (recalibrated) vs LightGBM, same protocol** | **0.10208** [0.10197–0.10218] | 0.10256 [0.10253–0.10260] | **−0.00048** | **no — head better** |
| served head vs LightGBM as it comes | 0.10208 | 0.10249 [0.10245–0.10251] | −0.00041 | no — head better |
| head before recalibration vs LightGBM as it comes | 0.10270 [0.10253–0.10284] | 0.10249 | +0.00021 | no — LightGBM better |

| comparison, conditional reliability of P(f = 1) ↓ | head | LightGBM-22 | gap | ranges overlap? |
|---|---|---|---|---|
| same protocol | 0.01130 [0.0105–0.0119] | 0.01102 [0.0108–0.0114] | +0.0003 | **yes — not distinguishable** |
| served head vs LightGBM as it comes | 0.01130 | 0.00819 [0.0067–0.0096] | +0.0031 | no — LightGBM better |

**On v7 the head is better calibrated than LightGBM-22 under every arm** — by 0.0122 under the same protocol, and by
0.0016 even against LightGBM with no recalibration at all, with every head seed below every LightGBM seed. Against the
three reference points the brief lists: 0.0143 against 0.0141 *published* (a different, unscripted LightGBM-20),
**0.0160 refit** and **0.0266 same-protocol**. On exact CRPS the **served** head is better than every LightGBM seed,
by 0.0004–0.0005; only the head *before* recalibration is marginally worse (+0.0002).

### Plainly

| world | marginal calibration (ECE 20), same protocol | ECE 20 vs LightGBM's best arm | proper score (exact CRPS), served head | conditional reliability of P(f = 1) |
|---|---|---|---|---|
| **v6** | **LightGBM-22 better** (+0.0024, disjoint) | LightGBM-22 better (+0.0082) | **head better** (−0.0009 same protocol / −0.0007, disjoint) | not distinguishable under the same protocol; LightGBM as it comes better (+0.0027) |
| **v7** | **head better** (−0.0122, disjoint) | **head better** (−0.0016, disjoint) | **head better** (−0.0005 / −0.0004, disjoint) | not distinguishable under the same protocol; LightGBM as it comes better (+0.0031) |

**On v7 the served head beats LightGBM-22 on marginal calibration and on the proper score. On v6 it wins the proper score
and loses marginal calibration.** On conditional reliability of P(f = 1) — what a planner reading one row relies on — the
two are not distinguishable under the same protocol, and LightGBM without recalibration is better in both worlds. The
shipped configuration does not change — this phase measures, it does not select — but **"the calibration gap is closed"
is true on v7 only, and only for marginal ECE.** §6 shows that marginal ECE is also beaten by a model-free floor.

---

## 6. The full baseline table

Test 2025, every row through the same scorer on the same rows. **Seeded rows:** mean over fits (spread = max − min; seed-7
95% interval). **Deterministic rows:** point [95% interval]. **Identity-keyed** = keyed on a supplier, channel, lane or
part-plant id learned from training data, so it has nothing to say about an entity it has never seen (not deployable to
an unseen Rane entity). The head rows are the shipped Phase 6 configurations, three seeds each.

### Arrival

| forecaster | deployable | fits | v6 C-index ↑ | v6 lateness ROC-AUC ↑ | v7 C-index ↑ | v7 lateness ROC-AUC ↑ |
|---|---|---|---|---|---|---|
| **head** — SHARE-lite h⁴ (shipped) | yes | 3 seeds | **0.6734** (0.0020; [0.6644, 0.6812]) | **0.7814** (0.0029; [0.7715, 0.7894]) | **0.6773** (0.0003; [0.6691, 0.6860]) | **0.7749** (0.0017; [0.7663, 0.7826]) |
| **promise date only** | yes | — | **0.8831** [0.8776, 0.8885] | 0.7482 [0.7403, 0.7566] | **0.8770** [0.8718, 0.8829] | 0.7498 [0.7424, 0.7575] |
| B5 LightGBM regressor, flat graph, no identity codes | yes | 5 seeds | 0.6433 (0.0003; [0.6357, 0.6530]) | 0.7558 (0.0007; [0.7476, 0.7640]) | 0.6490 (0.0008; [0.6411, 0.6585]) | 0.7555 (0.0005; [0.7482, 0.7633]) |
| LightGBM regressor, identity codes | identity-keyed | 5 seeds | 0.6437 (0.0005) | 0.7556 (0.0008) | 0.6494 (0.0007) | 0.7560 (0.0009) |
| naive per-channel median | identity-keyed | — | 0.5507 [0.5422, 0.5585] | 0.7064 [0.6973, 0.7164] | 0.5574 [0.5483, 0.5667] | 0.7097 [0.7010, 0.7178] |
| naive per-lane median | identity-keyed | — | 0.5179 [0.5107, 0.5257] | 0.7365 [0.7276, 0.7450] | 0.5244 [0.5156, 0.5314] | 0.7392 [0.7311, 0.7471] |
| naive global median | yes | — | 0.5000 (all ties) | 0.7482 (= promise) | 0.5000 | 0.7498 (= promise) |
| training-fold marginal 13-cell distribution | yes | — | 0.5000 | 0.7482; P(T > promise) 0.7531 [0.7454, 0.7611] | 0.5000 | 0.7498; P(T > promise) 0.7543 [0.7470, 0.7620] |
| *legacy* run9 "naive" (unscripted) | — | 1 | 0.5507 | 0.7064 | 0.5574 | 0.7097 |
| *legacy* run9 LightGBM (unscripted) | identity-keyed | 1 | 0.6428 | 0.7570 | 0.6481 | 0.7568 |

Lateness ROC-AUC ranks *prediction − promise*, so every point forecaster's lateness score already contains the promise
date, and a constant forecaster's lateness score **is** promise-only's. The legacy run9 "naive" matches the per-channel
median to four decimals on both metrics in both worlds: **it was the per-channel median**.

**Head against the strongest learned baseline (B5):** C-index +0.0301 on v6 (15× the head's spread) and +0.0283 on v7
(105×); lateness ROC-AUC +0.0256 on v6 (8.8×) and +0.0194 on v7 (11.5×). **The head beats every learned and naive
baseline on both ranking metrics in both worlds — except the promise date on C-index (§7).**

**Arrival-week distribution** (week-ECE ↓ over 12 weeks + ">12"; reliability of P(arrive ≤ 12) ↓)

| forecaster | fits | v6 week-ECE | v6 reliability | v7 week-ECE | v7 reliability |
|---|---|---|---|---|---|
| head, recalibrated (MM13), no fallback | 3 seeds | 0.0247 (spread 0.0053) | *before recalibration:* 0.0608 (0.0127) | 0.0522 (0.0106) | *before recalibration:* 0.0929 (0.0421) |
| head as served (drift fallback to h⁰ where it engaged) | 3 seeds | 0.0247 / 0.0221 / **0.0166** (seed 27 fell back) | — | **0.0146** × 3 (fell back on every seed; h⁰ seed 7) | — |
| training-fold marginal 13-cell | — | 0.0548 [0.0463, 0.0655] | 0.0281 [0.0216, 0.0319]† | 0.0430 [0.0351, 0.0541] | 0.0286 [0.0165, 0.0264]† |

- **v6:** the recalibrated head beats the constant marginal floor by 0.030 week-ECE (5.7× its spread).
- **v7: the recalibrated SHARE-lite head without fallback is not distinguishable from the constant marginal floor**
  (+0.0092, inside its own 0.0106 spread). The served output, the h⁰ fallback, is 0.028 below the floor; its band is
  unmeasured, because the h⁰ reference has one seed.
- **Reliability of P(arrive ≤ 12) before recalibration is worse than the constant floor in both worlds.** The bundle
  metrics do not record it after recalibration (§9).

### Fill

| forecaster | deployable | fits | v6 exact CRPS ↓ | v6 ECE 20 ↓ | v6 ECE 22 ↓ | v6 rel. P(f=1) ↓ | v7 exact CRPS ↓ | v7 ECE 20 ↓ | v7 ECE 22 ↓ | v7 rel. P(f=1) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| **head** h⁰, served (recalibrated) | yes | 3 seeds | **0.06161** (0.00004; [0.0597, 0.0635]) | 0.02124 (0.00211; [0.0173, 0.0265]) | 0.02389 | 0.00965 | **0.10208** (0.00021; [0.0999, 0.1041]) | 0.01434 (0.00133; [0.0126, 0.0214]) | 0.01493 | 0.01130 |
| head h⁰ before recalibration | yes | 3 seeds | 0.06184 | 0.04845 (0.00727) | 0.06188 | 0.01182 | 0.10270 | 0.08235 (0.01369) | 0.08677 | 0.02564 |
| LightGBM-22, identity codes, validation-fitted | identity-keyed | 5 seeds | 0.06248 | 0.01887 (0.00077) | 0.01986 | 0.01147 | 0.10256 | 0.02655 (0.00102) | 0.02679 | 0.01102 |
| LightGBM-22, identity codes, refit-only | identity-keyed | 5 seeds | 0.06235 | 0.01302 (0.00088) | 0.01493 | **0.00697** | 0.10249 | 0.01595 (0.00065) | 0.01795 | **0.00819** |
| LightGBM-22, identity codes, raw (seed 7 × 5) | identity-keyed | 5 refits | 0.06230 | 0.01292 (0; [0.0104, 0.0200]) | 0.01477 | 0.00773 | 0.10251 | 0.01633 (0; [0.0131, 0.0240]) | 0.01825 | 0.00962 |
| B5 LightGBM-22, flat graph, validation-fitted | yes | 5 seeds | 0.06246 | 0.01915 (0.00040) | 0.02011 | 0.01201 | 0.10269 | 0.02638 (0.00033) | 0.02661 | 0.01063 |
| B5 LightGBM-22, flat graph, refit-only | yes | 5 seeds | 0.06232 | 0.01282 (0.00105) | 0.01479 | 0.00754 | 0.10262 | 0.01636 (0.00057) | 0.01824 | 0.00737 |
| LightGBM-20, identity codes | identity-keyed | 5 seeds | 0.07581 (0.00007)‡ | 0.01297 (0.00213) | — | — | 0.11386 (0.00017)‡ | 0.01589 (0.00040) | — | — |
| **B2** as-of 52-week active-week histogram | yes (the channel's own history) | — | 0.0671 [0.0652, 0.0690] | **0.0134** [0.0102, 0.0201] | 0.0138 | 0.0550 | 0.1156 [0.1133, 0.1177] | **0.0110** [0.0094, 0.0198] | 0.0110 | 0.0877 |
| naive global empirical CDF | yes | — | 0.0658 [0.0638, 0.0679] | 0.0152 [0.0124, 0.0223] | 0.0162 | 0.0136† | 0.1120 [0.1097, 0.1143] | 0.0324 [0.0277, 0.0420] | 0.0349 | 0.0316† |
| naive per-channel CDF | identity-keyed | — | 0.0666 [0.0647, 0.0684] | 0.0175 [0.0146, 0.0237] | 0.0175 | 0.0483 | 0.1117 [0.1095, 0.1140] | 0.0345 [0.0297, 0.0441] | 0.0364 | 0.0645 |
| naive per-channel mean (point mass) | identity-keyed | — | 0.1081 [0.1061, 0.1102] | 0.7802 | 0.9373 | 0.4479 | 0.1793 [0.1770, 0.1817] | 1.0987 | 1.2347 | 0.5431 |
| *legacy* run9 LightGBM-20 (unscripted) | identity-keyed | 1 | 0.0759‡ | 0.0107 | — | — | 0.1140‡ | 0.0141 | — | — |
| *legacy* run9 "naive" (unscripted) | — | 1 | 0.0792‡ | 0.0152 | — | — | 0.1229‡ | 0.0324 | — | — |

‡ a 20-bin forecaster cannot place mass at exactly f = 1, which exact CRPS penalises. † point estimate outside its own
percentile interval: reliability of a constant forecast is a non-smooth statistic, and the percentile bootstrap does not
bracket it. The legacy run9 "naive" matches the global CDF's ECE 20 exactly in both worlds, with the higher CRPS of a
20-bin histogram. **It was very likely a global 20-bin histogram.**

- **Exact CRPS: the served head is the best row in both worlds.** It is 0.0007 below the next best on v6 (B5 refit) and
  0.0004 below on v7 (LightGBM-22 refit), with disjoint ranges.
- **Marginal ECE 20: the head does not beat the model-free floors on v6.** B2 (0.0134), the global CDF (0.0152) and the
  per-channel CDF (0.0175) all sit below every head seed (≥ 0.0203): margins 3.7×, 2.8× and 1.8× the head's spread.
  **On v7 B2 (0.0110) beats the head (0.0143) by 2.5× its spread**; the global and channel CDFs do not.
- **What marginal ECE does not measure.** It rewards matching the marginal histogram. B2's conditional reliability of
  P(f = 1) is 0.055 / 0.088, 6–8× worse than the head's, and its CRPS is 0.005 / 0.013 worse. Both are true; the metric
  as defined is beaten.
- **Deployable versus identity-keyed LightGBM:** B5 equals identity-keyed LightGBM-22 inside its spread in every arm.

### Capacity

| forecaster | deployable | fits | v6 mean pinball ↓ | v6 80% coverage | v6 Spearman P50 ↑ | v7 mean pinball ↓ | v7 80% coverage | v7 Spearman P50 ↑ |
|---|---|---|---|---|---|---|---|---|
| **head** HeteroMP h⁴ (shipped) | yes | 3 seeds | **0.04946** (0.00337; [0.0470, 0.0484]) | 0.764 (0.042) | **0.781** (0.007) | **0.07128** (0.00133; [0.0697, 0.0717]) | 0.755 (0.011) | **0.749** (0.004) |
| head h⁰ (not shipped; reference) | yes | 3 / 1 seeds | 0.04649 (0.00039) | — | — | 0.07654 (1 seed) | — | — |
| B5 LightGBM quantile, flat graph | yes | 5 seeds | 0.04932 (0.00016; [0.0486, 0.0500]) | 0.795 | 0.702 | 0.07449 (0.00016; [0.0734, 0.0755]) | 0.810 | 0.695 |
| LightGBM quantile, as-of + static (Phase 5) | yes | 1 | 0.0520 | — | — | 0.0770 | — | — |
| naive per-supplier P10/P50/P90 | **identity-keyed** | — | 0.0515 [0.0509, 0.0521] | 0.788 | 0.701 | 0.0954 [0.0940, 0.0968] | 0.781 | 0.525 |
| naive per-channel P10/P50/P90 | **identity-keyed** | — | 0.0587 [0.0579, 0.0595] | 0.616 | 0.653 | 0.1079 [0.1063, 0.1096] | 0.610 | 0.464 |
| naive global P10/P50/P90 | yes | — | 0.0726 [0.0717, 0.0736] | 0.838 | — (constant) | 0.1084 [0.1069, 0.1099] | 0.846 | — |

- **v6: the shipped head is not distinguishable from B5 LightGBM** (+0.00014, inside its 0.0034 spread). **Nor is it
  distinguishable from the non-deployable per-supplier floor** (−0.0020, 0.6× its spread).
- **The 0.0051 margin is h⁰'s.** The brief's "per-supplier gets within 0.0051 of the head" is the depth-0 head's margin
  (0.0515 − 0.0465), 12.8× h⁰'s spread; h⁰ also beats B5 by 0.0028, with disjoint ranges.
- **Why per-supplier is strong on v6:** the capacity label is supplier-level utilisation, so a supplier-keyed quantile
  already holds most of it. On v7 it collapses (0.0954).
- **v7: the shipped head beats B5 by 0.0032** (2.4× its spread, disjoint) and per-supplier by 0.024.
- **The head ranks clearly better than every baseline in both worlds** (Spearman 0.78 against 0.70; 0.75 against 0.70),
  but under-covers its 80% interval (0.74–0.79).
- **Not acted on.** Capacity v6's depth is Phase 8.2's question; the shipped configuration is unchanged.

### Shortage — DIAGNOSTIC

| forecaster | deployable | fits | v6 PR-AUC ↑ | v6 ROC-AUC ↑ | v7 PR-AUC ↑ | v7 ROC-AUC ↑ |
|---|---|---|---|---|---|---|
| head HeteroMP h¹ (Phase 5 run10) | yes | 1 seed | 0.6616 [0.6440, 0.6794] | 0.8647 [0.8566, 0.8720] | 0.5071 [0.4845, 0.5317] | 0.8342 [0.8240, 0.8452] |
| head h⁰ (floor; capped at 120 epochs) | yes | 1 seed | 0.5840 [0.5655, 0.6037] | 0.8146 | 0.4691 [0.4456, 0.4940] | 0.7983 |
| B5 LightGBM binary, flat graph | yes | 5 seeds | 0.5376 (0.0035; [0.5167, 0.5545]) | 0.7845 (0.0016) | 0.4486 (0.0073; [0.4250, 0.4754]) | 0.8027 (0.0012) |
| naive per-part-plant historical rate | identity-keyed | — | 0.4896 [0.4696, 0.5102] | 0.7695 | 0.2487 [0.2319, 0.2668] | 0.6960 |
| naive global rate (= base rate) | yes | — | 0.1967 | 0.5000 | 0.1258 | 0.5000 |
| *legacy* run9 LightGBM (unscripted) | — | 1 | 0.5287 | — | 0.4438 | — |

The legacy run9 "naive" (0.4896 / 0.2487) is the per-part-plant rate, as Phase 1–2 §11 says. The head has one seed, so
its band is unmeasured and no difference is claimed; its interval is disjoint from B5's seed-7 interval in both
worlds. Diagnostic only.

---

## 7. The promise-date baseline, and the model's margin over it

**Definition.** For each (PO line, snapshot) test row:

- **Score:** weeks from the snapshot to `po_lines.original_promise_date`, uncapped, in the label's units. No promise
  date is missing in either world.
- **C-index:** ranks rows on that score, so an earlier promise means an earlier arrival.
- **Lateness:** a row is late when its observed arrival week exceeds its promise week. A forecaster that knows only the
  promise ranks lateness by −promise, since an earlier promise is more likely to be missed. It is scored by passing a
  constant forecast through the same `arrival_scores` call, so the number is exactly what the model scorer gives a
  forecaster that knows nothing but the promise.
- **Training data:** none. **Deployable:** yes — a planner already holds it.

**The head never sees it.** In the model code `promise_week` appears only as the scoring offset `AUX`.

### C-index, split by the kind of pair it compares

3,000,000 sampled pairs; the scorer's own values agree to ±0.0004. "Observed–observed" pairs both arrive inside 12
weeks; "observed–censored" pairs are one inside and one beyond. The share of observed–observed pairs is 38.3% (v6) and
35.4% (v7).

| forecaster | v6 all | v6 observed–observed | v6 observed–censored | v7 all | v7 observed–observed | v7 observed–censored |
|---|---|---|---|---|---|---|
| **promise date only** | **0.8835** | **0.8260** | **0.9192** | **0.8770** | **0.8130** | **0.9121** |
| head, seeds 7 / 17 / 27 | 0.672 / 0.673 / 0.674 | 0.633 / 0.635 / 0.637 | 0.696 / 0.697 / 0.697 | 0.677 / 0.678 / 0.677 | 0.619 / 0.619 / 0.620 | 0.709 / 0.710 / 0.709 |
| B5 LightGBM, seed 7 | 0.6440 | 0.6048 | 0.6685 | 0.6501 | 0.5990 | 0.6781 |
| per-channel median | 0.5505 | 0.5309 | 0.5627 | 0.5577 | 0.5359 | 0.5697 |

Why the promise date is this strong:

- On observed rows, promise week and arrival week correlate at 0.78 (v6) and 0.76 (v7).
- A line promised within 12 weeks arrives inside the horizon 92.5% / 90.2% of the time; a line promised later, 32.0% /
  27.9%.
- 56.4% / 56.0% of test rows are promised beyond week 12.

### Margins

| comparison | v6 | v7 |
|---|---|---|
| **C-index: head − promise** | **−0.2097** (head spread 0.0020) | **−0.1997** (head spread 0.0003) |
| **lateness ROC-AUC: head − promise** | **+0.0332** (11.3× head spread 0.0029) | **+0.0251** (14.9× head spread 0.0017) |
| lateness ROC-AUC: B5 LightGBM − promise | +0.0076 (spread 0.0007) | +0.0057 (spread 0.0005) |
| lateness ROC-AUC: per-channel median − promise | −0.0418 | −0.0401 |

**Plainly: the head does not beat the promise date at ordering arrivals.**

- **The gap is large and not only censoring.** It is 0.21 / 0.20 C-index, and it holds within pairs that both arrive
  inside the horizon (0.635 against 0.826).
- **No learned model here is given the promise date or the line's age.** The head reads the channel's weekly history.
- **What the head adds is lateness information beyond the promise:** +0.033 / +0.025 ROC-AUC, 11× / 15× its own
  spread, and four times what LightGBM adds.
- **Phase 5 §7's "promise week on its own 0.7482 / 0.7498" is confirmed through this scorer.**
- **Consequence:** arrival's C-index is not evidence that the head orders arrivals better than a planner can. Whether
  the promise date should become an input is a model change, not made here (§9).

---

## 8. Baseline drift, and the LightGBM recalibration asymmetry

Each forecaster's label-free statistic on validation (2024) inputs against test (2025) inputs: the same shift Phase 6's
monitor sees. The observed label shift needs labels and is shown only to read the drift against. Head rows are seeds
7 / 17 / 27; seeded baselines are the mean of five fits.

### Arrival — observed censoring rate P(T > 12): v6 41.52% → 41.64% (+0.13 pp); v7 44.18% → 44.72% (+0.54 pp)

| forecaster | statistic | v6 val → test | v7 val → test |
|---|---|---|---|
| head SHARE-lite h⁴ | P(T > 12), raw | −0.58 / −0.57 / −0.75 pp | −2.11 / −1.59 / −2.13 pp |
| head h⁰ reference (2.5e-4) | P(T > 12), raw | +0.35 pp | +0.27 pp |
| head SHARE-lite h⁴ | expected week, raw | −0.020 / −0.002 / −0.018 wk | −0.155 / −0.109 / −0.139 wk |
| B5 LightGBM regressor | mean predicted week | +0.042 wk | +0.032 wk |
| LightGBM regressor, identity codes | mean predicted week | +0.043 wk | +0.034 wk |
| per-channel median / per-lane median | mean predicted week | +0.012 / +0.001 wk | +0.004 / +0.012 wk |
| promise date only | mean promise week | −0.003 wk | −0.033 wk |
| global median, marginal 13-cell | — | 0 by construction | 0 |

**Censoring rose in both worlds.** h⁰ and both LightGBMs move toward later arrival, the same direction. **The shipped
SHARE-lite head moves the other way**, about 3× harder on v7 than on v6; that is the excess drift Phase 6's fallback measures, and it
engages the fallback on every v7 seed. No baseline drifts as far as the head.

### Fill — observed P(complete): v6 91.51% → 89.97% (−1.54 pp); v7 82.88% → 81.75% (−1.13 pp)

| forecaster | v6 P(complete) val → test | v7 P(complete) val → test |
|---|---|---|
| head h⁰, raw | −0.69 / −0.58 / −0.61 pp | −0.89 / −0.75 / −0.88 pp |
| head h⁰, recalibrated | −0.72 / −0.58 / −0.68 pp | −1.02 / −0.80 / −1.01 pp |
| LightGBM-22 refit-only / validation-fitted | −1.00 / −0.88 pp | −2.15 / −2.20 pp |
| B5 refit-only / validation-fitted | −0.98 / −0.86 pp | −2.12 / −2.19 pp |
| B2 as-of 52-week histogram | −1.78 pp | −3.97 pp |
| per-channel CDF / per-channel mean | −0.02 / +0.02 pp | −0.17 / −0.53 pp |
| global CDF | 0 | 0 |

### Capacity — observed mean utilisation: v6 0.654 → 0.696 (+4.2 hundredths); v7 0.863 → 0.908 (+4.5)

| forecaster | v6 mean P50 val → test | v7 mean P50 val → test |
|---|---|---|
| head HeteroMP h⁴ | −2.8 / −3.4 / −3.8 | −1.7 / −1.1 / −2.4 |
| head h⁰ reference (seed 7) | **+4.2** | — (no v7 h⁰ capacity bundle) |
| B5 LightGBM quantile | +2.3 | +3.0 |
| per-supplier / per-channel | +0.09 / +0.08 | −0.53 / −0.56 |
| global | 0 | 0 |

**The shipped capacity head moves its median against the label shift on all six seeds**, while h⁰ (v6) and B5 move with
it. Phase 6 did not see this: the drift bands are arrival-only, and capacity has no fallback. This bears directly on the
Phase 8.2 depth question; it is recorded, not acted on.

### Shortage (diagnostic) — observed rate: v6 13.37% → 19.67% (+6.31 pp); v7 12.39% → 12.58% (+0.19 pp)

B5 +2.27 / +2.01 pp; per-part-plant −0.15 / 0.00 pp; global 0. There is no shortage bundle, so no head drift is recorded.

### The recalibration asymmetry, quantified (fill, ECE on the 22 cells)

Recalibration fits away the **validation** marginal error and carries that correction into 2025. Two quantities bound
what it can do:

- **the removable bias:** the forecaster's own validation ECE 22 before recalibration;
- **the imported shift:** the 2024 → 2025 distance between the observed 22-cell marginals.

The imported shift is the same for every forecaster: 0.0324 on v6, 0.0267 on v7.

| world | forecaster | removable bias (val ECE 22) | ratio to imported shift | test ECE 22 before | test ECE 22 after | **effect of recalibration** | label-free P(complete) drift | observed shift |
|---|---|---|---|---|---|---|---|---|
| v6 | **head h⁰** (3 seeds) | 0.0477 | **1.47** | 0.0619 | 0.0239 | **−0.0380** | −0.62 pp | −1.54 pp |
| v6 | LightGBM-22 (5 seeds) | 0.0102 | **0.31** | 0.0149 | 0.0199 | **+0.0049** | −1.00 pp | −1.54 pp |
| v6 | B5 (5 seeds) | 0.0104 | 0.32 | 0.0148 | 0.0201 | +0.0053 | −0.98 pp | −1.54 pp |
| v6 | B2 as-of histogram | 0.0167 | 0.51 | 0.0138 | — | — | −1.78 pp | −1.54 pp |
| v7 | **head h⁰** (3 seeds) | 0.0825 | **3.09** | 0.0868 | 0.0149 | **−0.0718** | −0.84 pp | −1.13 pp |
| v7 | LightGBM-22 (5 seeds) | 0.0325 | **1.22** | 0.0180 | 0.0268 | **+0.0088** | −2.15 pp | −1.13 pp |
| v7 | B5 (5 seeds) | 0.0321 | 1.20 | 0.0182 | 0.0266 | +0.0084 | −2.12 pp | −1.13 pp |
| v7 | B2 as-of histogram | 0.0552 | 2.07 | 0.0110 | — | — | −3.97 pp | −1.13 pp |

**The asymmetry is +0.0049 / +0.0088 test ECE 22 for LightGBM against −0.038 / −0.072 for the head.** How the numbers
read:

- **The head** carries a large validation error (1.5× / 3.1× the imported shift), and its inputs under-track the shift
  (−0.62 against −1.54 pp; −0.84 against −1.13 pp). Removing its validation error is mostly a real correction.
- **LightGBM on v6** has almost nothing to remove (0.31×), so the fit mainly transplants 2024's marginal.
- **LightGBM on v7** has a larger validation error, but its test error is already *smaller* than its validation error
  (0.018 against 0.032). Its inputs already carry the shift — label-free drift −2.15 pp against an observed −1.13 — so
  correcting 2024's error counts the shift twice.
- **B2 is the limiting case:** the largest validation error on v7 and the smallest test error, because an as-of
  histogram follows the shift directly.

This is a reading consistent with every row, not a controlled test.

### Guide 7.3 ablation — graph contribution from existing cells (no retraining)

Ratio = (full − h⁰) / (full − random) for rising metrics and (h⁰ − full) / (random − full) for losses, so a positive
value means the graph helps. "Random" is 0.5 for C-index, the base rate for PR-AUC, and the naive global floor for loss
metrics. **Full** is the shipped configuration (seed mean).

| task | world | metric | full | h⁰ | random | ratio | margin read against the measured band |
|---|---|---|---|---|---|---|---|
| arrival | v6 | C-index | 0.6734 | 0.6605 | 0.5 | **0.075** | +0.0130, 6.6× SHARE-lite's spread; h⁰ band unmeasured (1 seed) |
| arrival | v7 | C-index | 0.6773 | 0.6626 | 0.5 | **0.083** | +0.0147, 55× SHARE-lite's spread; h⁰ band unmeasured |
| capacity | v6 | mean pinball | 0.04946 | 0.04649 (3 seeds) | 0.0726 | **−0.128** | **h⁰ better** by 0.0030: 0.9× h⁴'s spread, 7.6× h⁰'s |
| capacity | v7 | mean pinball | 0.07128 | 0.07654 (1 seed) | 0.1084 | **+0.142** | h⁴ better by 0.0053, 4× h⁴'s spread |
| fill | v6 | exact CRPS (before recalibration) | 0.06184 | = full | 0.0658 | **0** by construction (ships at h⁰) | best graph variant 0.06192 → −0.020 |
| fill | v7 | exact CRPS | 0.10270 | = full | 0.1120 | **0** by construction | best graph variant 0.10312 → −0.045 |
| shortage (DIAGNOSTIC) | v6 | PR-AUC | 0.6616 | 0.5840 | 0.1967 | 0.167 | single seeds; h⁰ is a floor |
| shortage (DIAGNOSTIC) | v7 | PR-AUC | 0.5071 | 0.4691 | 0.1258 | 0.099 | single seeds; h⁰ is a floor |

**Where the graph earns its place:** arrival gains 7–8% of its lift over random; capacity v7 14%, but capacity v6
**−13%**. Fill's graph variants are worse than h⁰ in both worlds. Against the promise date instead of random, arrival's
C-index has no lift to share (§7).

---

## 9. Still open — carried forward, not acted on

1. **Arrival ranking against the promise date.** Every learned arrival model is 0.20–0.24 C-index below promise-only,
   and none sees the promise date or line age. Whether `promise_week` becomes a head input — or arrival's ranking claim
   is restated as "lateness beyond the promise" — is a model decision for Phase 8. Nothing was changed here.
2. **Fill calibration on v6.** LightGBM-22 is better calibrated under the same protocol (+0.0024). The head wins the
   proper score, and the model-free B2 histogram beats both on marginal ECE in both worlds. "Parity" or "closed" may
   be claimed only for v7, and only for marginal ECE.
3. **Capacity v6 (Phase 8.2).** The shipped h⁴ head is not distinguishable from B5 or from the non-deployable
   per-supplier floor, h⁰ beats both, and h⁴'s median drifts against the label shift on all six seeds. The shipped
   configuration is unchanged.
4. **Arrival v7 without fallback.** SHARE-lite's recalibrated week-ECE is not distinguishable from the constant marginal
   floor. The served h⁰ fallback beats the floor, but its band is unmeasured (one h⁰ seed at 2.5e-4 per world).
5. **Arrival reliability of P(arrive ≤ 12)** before recalibration is worse than a constant, and the bundle metrics do
   not record it after recalibration.
6. **Percentile-bootstrap intervals** for reliability statistics of constant forecasters can exclude the point estimate
   (marked †). They are reported as computed.
7. **Shortage** head rows are single seeds with no band. Diagnostic only; no claim is made.
8. **Legacy run9 baselines** were identified by exact score match: arrival naive = per-channel median, shortage naive
   = per-part-plant rate, fill naive ≈ global 20-bin histogram. The run9 LightGBMs remain unscripted; the rebuilt
   equivalents above supersede them.
9. **Phase 6 bundles** keep their `+dirty` stamps, which predate `d1ede34`. Guide B4 (demand-drift buckets) is out of
   scope. **`inventory_position_weekly` is untouched; Phase 9 remains blocked.**

---

## 10. Gate — may Phase 8 start?

| condition | status |
|---|---|
| every guide 7.1–7.3 baseline built, or its replacement recorded and annotated | **met** (§2, §3) |
| one scorer — the bundle scorer — with identical test rows asserted | **met** (216 files) |
| baselines under the same recalibration protocol as the head | **met** (§4) |
| refit bands measured per configuration, per world; no borrowed bands | **met** (§4–§6); unmeasured bands are named, not borrowed |
| fill parity answered | **met** (§5) |
| promise-date baseline formal, both metrics, margin stated | **met** (§7) |
| baseline drift and the asymmetry quantified | **met** (§8) |
| shipped configuration, learning rates, bundles, protected files untouched | **met** |

**Yes — Phase 8 may start.** It inherits three statements it may not contradict without new evidence:

1. The arrival head does not out-rank the promise date; its measured value is lateness beyond the promise.
2. On v6, fill calibration favours LightGBM-22, and on marginal ECE the model-free B2 histogram beats the head in both
   worlds.
3. Capacity v6's shipped depth is not distinguishable from the strongest deployable baseline.
