# Phase 22 — the leak removed and every baseline restated; the order-time arrival product; fill consolidated; capacity diagnosed

**Audience:** the owner, and whoever decides what ships for arrival, fill and capacity, and quotes any number from Phases 1–21.
**Measured on:** v8 seed 1001 (primary), with leaking panel columns replaced in the clean world `v8clean`. Replication on
v8w1002 (`data_worlds/v8_seed1002`, Phase 20; recorded hashes re-verified). Fixed split (train ≤ 2023, val 2024,
test 2025); seeds 7 / 17 / 27 / 37 / 47; TEST unless marked; RAW throughout.
**Companions:** `phase-22-preregistration.md` (`b529946`); `phase22/stage1_leak.md`, `stage2_order_time.md`,
`stage3_fill.md`, `stage4_capacity.md`. No PRIVILEGED file.
**Branch / base:** `phase22`, cut from local `HADES-v4-ml-pipeline` at `cbc7088`, in a separate worktree. Not merged, not
pushed. The user's working tree was not touched.
**Status:** complete. Started 00:53:40 IST; all neural cells finished by 06:37; wall-clock stop 14:53:40 IST. No STOPPED file.
**Deviations** are numbered from **209**.

## Header

| | |
|---|---|
| machine | Apple M4 Pro, 14 cores, 24 GB, macOS (Darwin 27.0) |
| software | Python 3.14.6, torch 2.14.0 (MPS, float32), LightGBM 4.7.0, NumPy 2.5.3, pandas 2.3.3 |
| neural concurrency | **1** in every recorded cell (`train_log.json`). Exception, deviation 215: two copies of one queue ran side by side for ~4 minutes; one was killed before any bundle was written |
| CPU beside MPS | LightGBM fits (≤ 3 processes × 6 threads) and scorers ran beside neural cells. Wall-clock only: arrival s/epoch 19.6–20.3 vs the stored 19.2–19.7 |
| dirty-tree rule | every fit, build, cell and score ran from a clean `ml/`; every artifact records its commit and `code_dirty = false` |
| Stage −1 handshake | stored `arrival lite h4 s7`, 3 epochs: **INSIDE** the stored band (\|Δ val C-index\| ≤ 2.0e-5); its constructed failing case (+0.05) reports OUTSIDE |

| artifact | commit |
|---|---|
| pre-registration | `b529946` |
| leak scan / leak rows / clean check | `c6d96d2` / `95a8dde` / `e025965` |
| clean caches (`v8clean`, `v8w1002clean`) and tests | `29f2054` |
| LightGBM arms (restatement, order time, fill families; world 2) | `625d0cc`, `2b602b5`, `cbb1167`… (per-arm logs in `ml/artifacts/phase22/logs/`) |
| clean neural incumbents (15 cells) | `10cb9f9`, `75e77d7`, `5f9fc7d`, `e79327f` (per cell in its `train_log.json`) |
| neural fill season + cadence (5 cells) | `2cee433` |
| order-time scores / product bundle / serve tests | `2b602b5` (v8), `f2e1f85` (w2) / `ml/artifacts/phase22/serve/order_time_v8clean/` / `serve_tests.json` |
| restatement / fill / capacity | `e79327f` / `2cee433` / `5f9fc7d` (clean), `f2e1f85` (published) |
| isolation audit | §7 |

---

## 1. Decisions (published = leaky, stored; clean = this phase; never compared across the two columns)

| use case × question | verdict | published | **clean** | interval (clean − published, block) |
|---|---|---|---|---|
| **leak scan** | **9 leaking panel columns**, not 3 | — | the fill family, ack_gap_ratio and load_ratio added to the known three | the scan reproduces the stored panel 100%; its constructed failing case fires |
| **arrival** incumbent neural, lateness AUC | **worse clean** | 0.7090 [0.7064, 0.7130] | **0.6951** [0.6817, 0.7041] | −0.0117 [−0.0169, −0.0057] |
| arrival neural, A3 | undetermined | 13.06 d | 13.15 d | +0.03 d [−0.16, +0.27] |
| arrival neural, UC1 P @ 5% (P(late)) | worse clean | 0.969 | 0.948 | −0.021 [−0.034, −0.009] |
| arrival LightGBM-flat, lateness AUC | worse clean | 0.7053 | 0.6991 | −0.0063 |
| arrival Phase 19 recipe (re-fitted on validation) | lower | 0.7305 / 12.41 d (w 0.49) | **0.7217 / 12.57 d** (w 0.52) | — |
| **arrival class (UC1)** | **no change** | WATCHLIST | WATCHLIST | |
| **fill** incumbent neural, P(full) AUC | worse clean | 0.6204 | **0.6007** | −0.021 [−0.044, −0.0002] |
| fill neural, UC2b P @ 5% | worse clean | 0.451 | **0.401** | −0.058 [−0.078, −0.014] |
| fill neural, CRPS | undetermined | 0.13876 | 0.1397 | +0.0011 [−0.0006, +0.0030] |
| **fill classes (UC2, UC2b)** | **no change** | WATCHLIST, WATCHLIST | WATCHLIST, WATCHLIST | |
| **capacity** incumbent neural, UC3 P @ 5% | **much worse clean** | 0.851 | **0.735** | −0.108 [−0.212, −0.023] |
| **capacity class (UC3)** | **ALERT → WATCHLIST** | ALERT (PARTIAL at 0.85, lift 2.0) | **WATCHLIST** (NO, CEILING) | |
| **order time: expected date** | **the shrunk-KM median wins** (clean, strictly as-of) | — | A3 **10.02 d** vs LightGBM 12.62 d | +2.60 d [2.18, 3.04]; **replicates** (+2.81 d on world 2) |
| order time: KM + LightGBM blend | degenerates to KM | — | w_KM = 1.00 on validation | — |
| order time: 80% interval (month-specific conformal) | in band | — | **0.803** [0.797, 0.808]; months 0.767–0.829 | replicates (0.811) |
| **order time: late flag, τ-week row (pre-registered)** | **ALERT** | Phase 21: ALERT (0.79, still leaky, deviation 210) | 0.794 at 5%, lift 2.25 | world 2: WATCHLIST (0.712) |
| **order time: late flag, STRICTLY as-of** | **WATCHLIST** | — | 0.615 at 5%; lateness AUC 0.623 | world 2 the same (0.623) |
| order time: L4 group statistics | FAIL (no gain; the control, which keeps the week's L1/L2 trend, gains) | — | | replicates (no gain) |
| order-time product persisted and servable | **yes** | — | bit-exact on 200 test lines; mismatches raise | |
| **fill families** (clean LightGBM gates) | **season + cadence PASS**; ack-gap FAIL; L4 FAIL | — | ack-gap's same-channel snapshot-permuted control also gains | the ack failure replicates on world 2 |
| fill neural + season + cadence | **GAIN** over the clean incumbent | — | CRPS 0.1375, AUC 0.640, UC2b 0.464 | disjoint on all three |
| **fill consolidated blend vs the clean Phase 19 blend** | better CRPS and UC2; **worse UC2b** | Phase 19 blend 0.1354 / UC2b 0.530 | **0.13444** / UC2 0.944 / UC2b 0.461 vs 0.13576 / 0.927 / 0.511 | CRPS +0.0013 [0.0003, 0.0025]; UC2 +0.030 [0.008, 0.072]; UC2b −0.048 [−0.081, −0.019] |
| fill servable | **no** | — | its season family's builder reads part_demand_weekly (constraint 4) | |
| **capacity Dec 2025** | **ranking degradation in a quiet month** (not regime, not threshold drift) | AUC 0.637 < val min 0.657 | AUC 0.601 < val min 0.606; base rate 0.20 inside 0.11–0.68 | |
| capacity rolling recalibration | **none ADOPTED** | | schemes that raise precision flag 0.7–1.8% (outside 5–15%) | |
| capacity intervals (Phase 18 level-aware conformal) | hold | quarters 0.783–0.806 | quarters 0.764–0.829 | |

## 2. Prediction scorecard

| # | prediction | **right / wrong** | the number |
|---|---|---|---|
| **P1** | leak removal lowers neural arrival lateness AUC by 0.004–0.015; A3 worsens 0.1–0.5 d | **WRONG** | lateness −0.0117 (in range), but A3 +0.03 d (undetermined, below 0.1) |
| **P2** | fill and capacity not materially affected | **WRONG** | the audit shows both read leaking columns. Fill AUC −0.021, UC2b −0.058; capacity precision @ 5% **−0.108** |
| **P3** | no class changes for UC1, UC2, UC2b, UC3; the order-time flag stays ALERT | **WRONG** | **UC3 ALERT → WATCHLIST**. UC1, UC2, UC2b unchanged; the pre-registered (τ-week) order-time flag stays ALERT, but strictly as-of it is WATCHLIST |
| **P4** | the scan finds a further as-of violation | **RIGHT** | six more columns: fill_rate (+ 4 / 13 / 52), ack_gap_ratio, load_ratio |
| **P5** | best date = KM + LightGBM blend, A3 ≤ 10.0 d, block-better than LightGBM; flag lateness AUC ≥ 0.69 | **WRONG** | validation chose the KM median (blend weight on LightGBM 0); A3 **10.02** (block-better by 2.60 d, but above 10.0); flag 0.6825 (strict 0.623) |
| **P6** | conformal 80% covers 0.77–0.83 overall, undercovers > 5 points in some month | **WRONG** | 0.803 overall; worst month 0.767 (Jan), 3.6 points under: month-specific quantiles held every month |
| **P7** | consolidated fill beats the Phase 19 blend on CRPS (block) and UC2, stays WATCHLIST, UC2b does not beat 0.530 | **RIGHT** | vs the clean Phase 19 blend: CRPS +0.0013 [0.0003, 0.0025]; UC2 0.944 vs 0.927; WATCHLIST; UC2b 0.461 |
| **P8** | rolling recalibration raises the worst quarter > 0.05 but lowers the mean; Dec 2025 per diagnosis | **WRONG** | none adopted; the best defined by validation (flag-rate) **lowers** the worst quarter (0.594 vs 0.667). The diagnosis is ranking degradation, the case P8 does not name |
| **P9** | capacity stays ALERT with a documented floor | **WRONG** | clean UC3 is **WATCHLIST**; floor (worst quarter, precision @ 5%) 0.553 mean [0.492, 0.624] |

**2 right, 7 wrong.**

## 3. Deviations, continuing from 209

| # | prior statement | measured / done | where |
|---|---|---|---|
| **209** | constraint 5: every feature strictly as-of t0 | the stored panel row read for snapshot t0 covers the week **starting** at t0 (up to 6 days after the snapshot date); the generator's labels start the week after. The scan's verdict uses that week horizon (H_week, pre-registered D1); the strict horizon is reported per column and measured: at snapshots worth ≈ 0.005 lateness AUC; **at order time decisive** (flag ALERT → WATCHLIST) | 1, 2 |
| **210** | Phase 21: an "honest" leak-free order-time flag is ALERT (BASE_nl) | BASE_nl removed 3 of the 9 leaking columns; on clean inputs the ranking arms are WATCHLIST, and only the dedicated flag with the τ-week row is ALERT | 2 |
| **211** | `load_ratio` is an observable load | it is the supplier's whole-month ordered quantity over declared capacity; at t0 it holds orders not yet placed, and the capacity label is built from the same monthly ordered totals over the forward months. Capacity's ALERT was mostly this | 1, 4 |
| **212** | constraint 4: no PRIVILEGED / generator state | the leak scan and leak-rows audit read `_sim.npz` to reproduce the generator exactly; they live in `ml/eval/` and feed no model. The clean replacements use CSVs only | 1 |
| **213** | D8: windows {2, 4, 6, 8} chosen on validation walk-forward | stored capacity predictions exist only for val and test, so after the 90-day purge only 2–4 validation snapshots are scorable; windows 6 and 8 are infeasible. A first scorer counted a quarter with no flag as precision 0, which made `flag_rate` "ADOPTED"; corrected to undefined before any verdict was written | 4 |
| **214** | D3: the clean Phase 19 arrival blend | its neural half (the Phase 19 rf arm) tied the incumbent and was not retrained; the recipe is restated with the incumbent in its place in both columns. The Phase 21 hybrid is restated the same way | 1 |
| **215** | concurrency 1 | (a) two copies of the bound-fill queue ran for ~4 minutes; the duplicate was killed before writing. (b) a comma separator split the bind families: the first five bound cells trained **season only** (`_rfseason`, correctly named, kept as a diagnostic); the intended season + cadence cells were re-run | 3 |
| **216** | D6: Stage 2 arms | strict arms added (`base_lag1`, `flag_lag1`, `L4_lag1`): panel row before τ. The persisted product uses the strict flag | 2 |
| **217** | never overwrite a stored artifact | a first build of world 2's placement group store would have rewritten Phase 21's `grpstats_v8w1002.json` stamp and `k_select_v8w1002.json`; it was killed before writing (verified) and rebuilt into `ml/artifacts/phase22/`. World 2's placement k (3) was chosen inside the Stage 2 scorer by the Phase 21 rule | 2 |
| **218** | D7: neural bound with the passing families; the clean Phase 19 fill arm | only season + cadence passed, so both are the same five cells | 3 |
| **219** | Stage 3e: persist the best fill model | declared **not servable**: the season family's as-of builder reads `part_demand_weekly`, which constraint 4 forbids in model / serve code. The season family itself comes from Phase 18's stored arrays (`fwd_load.py`, deviation 162's as_of + 2 d bound), a pre-existing reader | 3 |
| **220** | 1c: "verify equality with the stored columns where no leak applies" | point-in-time columns agree on 95–100% of no-leak rows; the rolled columns (4–52-week windows) cannot, because order- and receipt-keying place past values in different weeks | 1 |
| **221** | "402 / 371 identities expected" | **real count in §7** | 7 |
| **222** | Stage 3a: ack-gap control "permuted across snapshots" | implemented exactly as the same channel at a donor snapshot of the same split (a derangement), which keeps channel identity; this is the control that fails it | 3 |

## 4. Every `inventory_position_weekly` / `part_demand_weekly` read

**`inventory_position_weekly`: none.** **`part_demand_weekly`: no Phase 22 module reads it**, as the AST scan with self-test
confirms. The season family used in Stage 3 comes from Phase 18's **stored** `fwd_load` arrays, which `ml/data/fwd_load.py`
built from `part_demand_weekly` with deviation 162's as_of + 2 days bound. That pre-existing read is listed, not hidden.

## 5. In plain language

**(1) How much of the published performance was the leak?**
- **Arrival: a little.**
  - The incumbent's lateness ranking loses about 0.012 AUC (0.709 → 0.695), and the top of its late list loses about
    2 points (0.969 → 0.948).
  - Its median date error does not move.
- **Fill: more.**
  - The incumbent's arrives-in-full ranking loses 0.021 AUC.
  - Its materially-short list loses 6 points (0.451 → 0.401).
- **Capacity: most of it.**
  - The one alert in the project read the current month's eventual ordered quantity, which is part of its own label.
  - Clean, its top-5% precision falls from 0.85 to 0.74. **It is no longer an alert.**
- The leak touched about **95% of every model's input rows**.

**(2) Is the order-time arrival product real, and what would it say?** **The date is real; the late flag is a
watchlist.**
- **The date.** For a line raised today, the channel's survival-corrected median lead (shrunk toward its supplier) misses
  the receipt by a median of **10 days**, against 12.6 for the best model on the same as-of information. That gain
  replicates on the second world.
- **What it says:** *"Expected receipt around <date>; in 8 cases out of 10 between 16 days early and 40 days late
  (narrower for lines raised in Jan–Feb, wider in Aug–Nov)."*
- **The late-risk score.** It is right about **6 times in 10 at its top 5%**, against a 3.5-in-10 base, so it belongs on
  a watchlist, not as an alert.
- It reaches alert quality (8 in 10) only when it is allowed to see the rest of the week the line was raised in, which
  a planner does not have.
- The product is persisted and servable.

**(3) Is the consolidated fill model worth retraining the stored incumbent for?** **Yes, for the fill distribution and
the arrives-in-full list. No, for the materially-short list.**
- **Gains on clean inputs:**
  - The forward season plus the channel's ordering cadence gives the neural head a disjoint gain on every fill metric.
  - The LightGBM consolidated model gives the best fill distribution measured (CRPS 0.1344).
  - It gives the best arrives-in-full list (0.944 at 5%).
- **Where it loses:** the materially-short list is better served by the Phase 19 recipe with forward load (0.511 vs 0.461).
- **What does not carry:** the acknowledgement gap and the channel group statistics, Phase 21's candidates, do not carry
  real information: the gap is a fixed trait of the channel.
- **The obstacle to shipping it:** its season input needs the planning table this phase may not read in serving code.

**(4) Is the December 2025 capacity drop fixable or a regime shift?** **Neither: it is a ranking failure in a quiet month.**
- The share of strained channels (20%) was normal for the season, but the model ordered channels worse than in any
  validation month (AUC 0.60).
- Recalibration cannot fix a ranking. Every scheme that raised precision did it by flagging almost nothing.
- The answer is a **monitor**: a trailing precision and base-rate gauge, with alerts suppressed when it trips
  (`stage4_capacity.md` §e).

## 6. Updated shipping table (only verdicts that survived a 5-seed band or the block bootstrap; recommend only)

`ml/configs/shipped.json` and `docs/decisions` are not edited.

| use case | ships now | watchlist | blocked on data | closed |
|---|---|---|---|---|
| **arrival (snapshot)** | the clean-retrained Phase 19 recipe as the point estimate (A3 12.57 d); the clean incumbent's P(late) as the ranked late list (0.948 at 5%). **Replace every stored arrival model with its clean retrain** | the late list (WATCHLIST, unchanged) | order timing (Phase 19), reorder state (Phase 20) | every leaky stored arrival model |
| **arrival (order time, NEW)** | **the expected-date product**: shrunk-KM median + month-specific 80% interval, servable (`ml/serve/order_time.py`) | the strict late-risk score (top 5%, precision 0.62) | — | the KM + LightGBM blend (no weight); L4 group statistics at order time |
| **fill** | the clean incumbent retrained **with season + cadence** (GAIN) as the fill distribution; the clean Phase 19 recipe as the materially-short list (0.511) | the LightGBM consolidated model (best CRPS and UC2; **not servable** under constraint 4) | a forward-plan source other than `part_demand_weekly`, or a recorded exception to constraint 4 | ack-gap and L4 as fill inputs; every leaky stored fill model |
| **capacity** | **nothing as an alert.** The strain quantiles with the level-aware interval (≈ 80% per quarter) | the clean incumbent's top list (WATCHLIST, 0.735 at 5%) behind the trailing precision monitor | the supplier's declared-capacity history and any observed capacity evidence (client ask) | rolling recalibration (no scheme adopted); the leaky alert |

### Errata for a future `observation3.md` (figures in `results/observation2.md` superseded here)

| observation2 | published figure | superseded by (clean) |
|---|---|---|
| §summary row 3 "capacity ships as an alert, 81% right" (l. 31) | ALERT | **WATCHLIST**; precision @ 5% 0.735 |
| arrival table (l. 124–129, 397–407): lateness AUC 0.709, UC1 @ 5% 0.969, blend 0.731 / 12.34 d | leaky inputs | 0.695; 0.948; Phase 19 recipe (incumbent) 0.722 / 12.57 d |
| fill table (l. 151–155, 408–409): P(full) AUC 0.620 → 0.643; CRPS 0.13876 → 0.1371 → 0.1354; UC2b 0.464 → 0.530 | leaky inputs | incumbent 0.601; + season + cadence 0.640 / CRPS 0.1375; clean Phase 19 blend CRPS 0.1358, UC2b 0.511 |
| capacity table (l. 176–180): 0.904 / 0.851 / 0.810; recall 0.515 / 0.318 / 0.192; ensemble 0.865 | leaky inputs | 0.828 / 0.735 / 0.679; recall 0.308 / 0.116 / UNREACHABLE |
| oracle headroom table (l. 231–233) | measured against leaky incumbents | to be re-measured against the clean incumbents |
| any order-time ALERT quoted from Phase 21 | ALERT (0.79) | τ-week flag 0.794 (ALERT); strictly as-of 0.615 (WATCHLIST) |

## 7. Isolation audit

`ml/tests/test_phase22_isolation.py` → `ml/artifacts/phase22/isolation_audit.json`: *(run below; result recorded in the
commit that follows this report)*.
