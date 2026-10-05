# Observation 3.1 — What ChainPilot predicts once the leak is removed, what ships, and what it would take to do better

**Version 4.1 (DRAFT).** Written after Phases 21, 22, 23AC, 23B and 24 (2026-10-06). A full copy of `observation3.md` (v4.0,
not edited) with its two PENDING PHASE 23B rows filled. It **supersedes `observation3.md` and `observation2.md` (v3.0)**, and
stands on its own. Appendix B lists every superseded figure.

**Changelog v4.0 → v4.1** (every changed row, and its source):

| where | v4.0 said | v4.1 says | source |
|---|---|---|---|
| top of file | PENDING PHASE 23B box | removed: both rows are restated | this table |
| §0 row 4b, §3.6, §8 note, Appendix A, C | predict-the-rescue PENDING; 0.823 at recall 0.246 (LEAKY-UNCONFIRMED) | **clean B1a 0.733** [0.7325, 0.7336] at recall 0.254, base 0.4535; **still GO, ALERT (PARTIAL @ 0.80), marginal** under block resampling ([0.697, 0.773]); 0.823 is LEAKY (superseded, Phase 22) | `reports/part2/phase-23b.md` §1 (`535361d`), QUOTED |
| §0 row 5, §3.7, §8 note, Appendix A, C | simulation PENDING; 1.129× and ceiling 0.28 (LEAKY-UNCONFIRMED) | **restated on clean heads:** pre-rescue match **1.137×** [1.134, 1.139]; rescue-detection precision **0.612** (was 0.619); still misses **66%** of rescued weeks; below-SS ceiling **0.28**; WATCHLIST | `reports/part2/phase24/stage2_simulation.md`, RECOMPUTED |
| §0 "what changed", line 2 | "No use case is an alert any more" | one alert remains, and it is marginal: predict-the-rescue | Phase 23B |
| §10 | "no alert, honestly"; "two use cases pending a restatement" | one marginal alert (predict-the-rescue); nothing pending a restatement | Phases 23B, 24 |
| "Where the numbers come from" | nothing new measured | the simulation's clean restatement is the one new measurement | Phase 24 |
| Appendix B | — | extended with the v4.0 rows this version supersedes | — |

The metrics pack with these rows filled is `reports/part2/phase24/metrics_pack_v2.md`.

**Who this is for:** a client lead or product owner who wants to know what the system can be trusted to do, without
reading twenty-three phase reports. Each section says *what it does, how well, why it is limited, and what ships*.

**Where the numbers come from.** Every number is copied from a committed report or a metrics pack. The Phase 23AC pack
**recomputed 79 of 79 published figures from stored predictions and reproduced every one** (`phase23ac/t4/metrics_pack.md`).
The Phase 24 pack (`phase24/metrics_pack_v2.md`) adds the rescue and simulation rows. The only new measurement is the
simulation's clean restatement (Phase 24 Stage 2), which first reproduced the published simulation exactly. Each number is
named beside it and registered in Appendix A.

Unless marked otherwise:
- **World:** the synthetic **v8 world, dataset seed 1001**.
- **Split:** fixed (train up to 2023, validation 2024, **test 2025**).
- **Seeds:** **five model seeds** where a band `[min, max]` is shown.
- **RAW** straight from the model; the two are never mixed with recalibrated figures in one comparison.

**Read this first: the leak (§1) and the standing caveat (§10).**
- **The leak.** Every figure published before Phase 22 was computed with **nine panel columns that carried future
  information**. Only the CLEAN figures below are current. The old ones appear only in §1 and Appendix B, labelled LEAKY.
- **The caveat.** Every figure comes from one synthetic generator and must be re-measured on Rane's own data.

---

## 0. One-page summary

ChainPilot looks at each **sourcing channel** (one supplier supplying one part to one plant; 16,072 of them) and each
purchase-order line, and forecasts eight things. **CLEAN figures only.**

| # | use case | the question | **status (clean)** | the number that decides it |
|---|---|---|---|---|
| 1 | **Arrival timing (at a forecast date)** | when will the open order arrive, and will it be late? | **ships**: point estimate, and the late list as a ranked **watchlist** | median date error **12.57 days**; the late list is right **9.5 times in 10** in its top 5%, against **7.3** by chance (lift 1.30: a watchlist) |
| 1b | **Arrival at order time** *(new, Phase 22)* | the moment a PO line is raised: when will it arrive? | **ships: expected date + 80% interval** (servable); late-risk score as a **watchlist** | median error **10.0 days** vs **14.0** for the promise date; 80% interval covers **0.80**; late-risk right **6.2 in 10** in its top 5% vs **3.5** by chance |
| 2 | **Fill rate** | how much of the order will arrive? | **watchlist** | "materially short" list right **5.1 in 10** in its top 5%, on a **2.4 in 10** base; best fill distribution CRPS **0.1344** |
| 3 | **Supplier strain (capacity)** | will demand exceed what the supplier can deliver in the next 90 days? | **watchlist** (it was the one alert; the alert was the leak) | right **7.3 in 10** in its top 5% against **4.1** by chance; one snapshot (Dec 2025) at **0.45** |
| 4 | **Shortage risk (head)** | which part-plants will run short? | **retired** | 0.224 precision in its top 5% at a 3% base (LEAKY-UNCONFIRMED; retired regardless) |
| 4b | **Predict-the-rescue** | which weeks will planners move stock to save? | **ALERT, marginal** (still GO on clean inputs; not servable on this machine yet) | right **7.3 in 10** at a quarter of the rescued weeks, against **4.5** by chance (clean B1a 0.733, lift 1.62); the simulation manages 6.1 |
| 5 | **Shortage simulation** | how much short, in which week? | **mechanism holds; detector WATCHLIST** | matches the pre-rescue world to within 14% (1.137×); misses two-thirds of rescued weeks; below-safety-stock ceiling 0.28 |
| 6–8 | **MILP schedule, allocation, transfer** | how much, from whom, from where? | **blocked** (costs; a rule) / retired as an alert | no model metric |

**What changed since v3.0, in five lines.**
1. **Nine input columns leaked the future** (§1). Removing them cost a little on arrival, more on fill, and **most of
   capacity, which is no longer an alert**.
2. **Only one alert remains, and it is marginal: predict-the-rescue** (clean 0.733 at a quarter of the rescued weeks; its
   snapshot-resampled interval reaches 0.697). Every other decision list is a ranked watchlist, better than chance by 1.2–2.1×.
3. **A new product works:** the order-time expected date beats the promise date and the channel's own average lead.
4. **The forward plan still helps fill on clean inputs** (season + cadence: a disjoint gain). The acknowledgement gap and
   channel group statistics, briefly promising, are closed.
5. **A standing leak guard now checks every model input** on every run (§7).

---

## 1. The leak: nine columns, how they were found, what they changed

**What it was.** The weekly channel table (`channel_performance_weekly`) is built by the data generator. For nine of its
fifteen columns, the generator wrote each order line's **eventual outcome** into the week the line was **ordered**:

| columns | what they carried, too early |
|---|---|
| `lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13` | the line's eventual delivery lead |
| `fill_rate`, `fill_rate_last4`, `fill_rate_last13`, `fill_rate_last52`, `ack_gap_ratio` | the line's eventual delivered fraction |
| `load_ratio` | the supplier's **whole-month** ordered quantity, including orders not yet placed |

In plain terms, the forecast was being made with tomorrow's newspaper folded into today's.

> **In plain terms — the tailor.** A tailor who is told your final measurements before you have been measured will make a
> suit that fits perfectly. Take the tape away and the fit is what it really is.

**How it was found.**
- **Phase 21** spotted three columns while building an order-time product: a model there scored 0.99, which is
  impossible on the day an order is placed.
- **Phase 22** rebuilt all fifteen columns from the raw tables, line for line. It reproduced the stored table exactly, then
  hid every row recorded after each week, to see which columns move. **Nine moved; six did not**
  (`phase22/stage1_leak.md`).
- The leak reached **94–96% of every model's input rows**.
- An earlier check (Phase 0–1, maximum correlation with the label 0.319) passed. A forward-filled future value
  correlates weakly with the label but **equals** another line's outcome exactly.

**What removing it changed** (each model retrained on clean inputs; clean minus published, 5-seed ensembles,
snapshot-block 95% intervals; `phase-22.md` §1):

| use case | measure | published (LEAKY) | **clean** | change |
|---|---|---|---|---|
| arrival late list | lateness ranking (AUC) | 0.709 | **0.695** | −0.012 [−0.017, −0.006] |
| arrival late list | right in the top 5% | 0.969 | **0.948** | −0.021 [−0.034, −0.009] |
| arrival date | median error | 13.06 d | **13.15 d** | +0.03 d, undetermined |
| fill | ranking of "arrives in full" (AUC) | 0.620 | **0.601** | −0.021 [−0.044, −0.0002] |
| fill | materially-short list, top 5% | 0.451 | **0.401** | −0.058 [−0.078, −0.014] |
| capacity | right in the top 5% | 0.851 | **0.735** | **−0.108** [−0.212, −0.023] |
| **capacity class** | | **ALERT** | **WATCHLIST** | **the one alert is withdrawn** |

**Why capacity fell furthest.** Its label is the supplier's ordered quantity over capacity in the next 90 days, starting
with the current month. `load_ratio` had already counted the rest of that month's orders, so the model was partly reading
its own answer.

**What is clean now.**
- **Replacement columns:** as-of-safe columns rebuilt from the tables (an outcome counts only from the week it is
  recorded). They pass a future-poison test and a self-exclusion test, each of which has a planted leak it must catch
  (`phase22/stage1_leak.md` §c).
- **Retraining:** every arrival, fill and capacity model was retrained on them.
- **The standing guard (§7):** now checks every model input on every run.

---

## 2. Two ways to read every number: metric space and decision space

A model can be *better* without being *useful*.
- **Metric space** (AUC, CRPS, pinball loss) answers "is A better than B at ordering or describing outcomes?".
- **Decision space** (precision, recall, lift at an operating point) answers "if a planner acts on this, how often are they
  right?".

| decision | base rate (v8 test) | source |
|---|---|---|
| line arrives late against its contract (forecast date) | **0.730** | `phase-15.md` §1 |
| line raised today arrives late against its contract (order time) | **0.349** | `phase22/stage2_order_time.md` |
| line arrives exactly in full | **0.749** | `phase-15.md` §1 |
| demand exceeds supplier capacity (strain > 1) | **0.405** | `phase-15.md` §1 |
| line is materially short (fill < 0.95) | **0.245** | `phase-15.md` §1 |
| a part-plant week is rescued by a planner transfer | **0.4535** | `phase-17.md` §2.1 |

> **In plain terms — the weather forecaster in the desert.** If it is sunny 73 days in 100, always saying "sunny" is right
> 73% of the time and tells you nothing. So **no accuracy figure here appears without the always-guess-the-majority
> figure beside it**, and the honest measures are **lift** and **precision at coverage** ("if it only speaks about its
> most confident 5%, how often is it right?"). The metrics pack checks this by feeding the scorer a predictor that always
> says the majority answer: its accuracy equals the baseline while its precision and recall expose it
> (`phase23ac/t4/metrics_pack.md`, sanity check iii).

**An AUC gain is not a decision.**
- The Phase 19 arrival blend ranked lines better overall (0.731 vs 0.714) but was right less often in its top 5% (0.943
  vs 0.969) (`phase-20.md`).
- On clean inputs the same pattern holds: the clean recipe ranks at 0.722 but is right 0.919 at the top, against the clean
  incumbent's 0.948 (`phase22/stage1_leak.md`).

---

## 3. The use cases (clean)

### 3.1 Arrival timing, at a forecast date

**What it does.** For each open purchase-order line, a distribution over its arrival week (a *hazard* model that uses the
48.5% of lines still in transit at the horizon instead of discarding them). From it come a date and a ranked list of the
lines most likely to arrive late against their contract.

**How well (clean):**

| | measure | value | source |
|---|---|---|---|
| point estimate (clean Phase 19 recipe: neural ensemble 0.52 + LightGBM with the forward plan 0.48) | median error | **12.57 d** [12.22, 13.05] | `phase23ac/t4/metrics_pack.md` |
| late list (clean neural ensemble P(late)) | right in top 5% | **0.948** [0.930, 0.965] vs base 0.730 (lift 1.30) | same |
| late list | lateness ranking AUC | 0.702 (ensemble); 0.695 [0.682, 0.704] per seed | `phase22/stage1_leak.md` |
| class | | **WATCHLIST** (unchanged) | |

**Why it is limited.** Three-quarters of lines are late against contract, so even a very good list is only 1.3× better
than chance. The real headroom is **when the order will be raised** (§4), which data available at the forecast date
barely reveals.

### 3.2 Arrival at order time (new)

**What it does.** The moment a planner raises a PO line, it gives:
- the **expected receipt date**: the channel's survival-corrected median lead (a Kaplan–Meier estimate that counts
  still-open lines, shrunk toward the supplier when the channel is thin);
- an **80% interval**, calibrated per creation month;
- a **late-risk score**, built from information held strictly before the day the line is raised.

It is **persisted and servable** (`ml/serve/order_time.py`, `ml/serve/order_time_card.py`): the identity guard refuses a
wrong or tampered model, and serving reproduces the stored predictions bit-exactly.

**How well, against what Rane has today** (31,794 lines raised in 2025; creation-week block intervals;
`phase23ac/t1/t1_order_time.md`):

| predictor | median error | mean error | within ±7 days | average bias |
|---|---|---|---|---|
| **the product** | **10.0 d** | **17.3 d** | **38%** | **−8.4 d** (early) |
| promise date | 14.0 d | 18.4 d | 27% | +1.7 d |
| channel's trailing mean lead | 12.3 d | 19.5 d | 32% | −0.4 d |
| channel's trailing median lead | 11.0 d | 19.2 d | 35% | −3.8 d |

The product is better than all three on every measure, with the block interval excluding zero (e.g. median error vs the
promise date: 3.7 days better [2.7, 4.3]).

**The interval.** It covers **80%** of receipts (0.803 [0.797, 0.808]), 0.77–0.83 in every creation month, at 37–70
days wide (narrow for lines raised in Jan–Feb, wide in Aug–Nov).

**The late-risk score.** Strictly as-of, it is right **0.615** in its top 5% against a 0.349 base: a **watchlist**
(`phase22/stage2_order_time.md`). It reaches alert quality (0.79) only if it may see the rest of the week the line was
raised in, which a planner does not have.

**Why it is limited.**
1. **Bias.** It aims at the typical line, so on average it is **early** (−8.4 days): lead times have a long late tail. The
   interval, not the point, covers that tail.
2. **The second world.** It still beats the promise date and the trailing mean, but only **ties** the channel's trailing
   median on median error, and is slightly worse at hitting ±7 days (Appendix C, claim 2 label).

**What a planner would see:**

> *"Expected around 3 March (week 10); 8 in 10 such lines arrive between 13 February and 25 March. Late-risk 0.39:
> watchlist score."*

That line arrived on 12 April: it is in the sample on purpose (`phase23ac/t1/sample_cards.json`).

### 3.3 Fill rate

**What it does.** For each order line, a 22-cell distribution of the fraction that will arrive. From it come "will it
arrive in full?" (UC2) and "will it be materially short?" (UC2b).

**How well (clean;** `phase22/stage3_fill.md`, `phase23ac/t4/metrics_pack.md`**):**

| | measure | value |
|---|---|---|
| clean incumbent neural | CRPS / P(full) AUC | 0.1397 / 0.601 |
| **neural + forward season + cadence** | CRPS / AUC / short list @ 5% | **0.1375 / 0.640 / 0.464**: a disjoint gain over the clean incumbent on all three |
| **best distribution and "arrives in full" list:** LightGBM consolidated (+ season, cadence, ack-gap, channel statistics) ≈ the consolidated blend | CRPS / UC2 @ 5% | **0.1344 / 0.944** (base 0.749) |
| **best materially-short list:** clean Phase 19 recipe (neural season + cadence 0.49 + LightGBM with forward load 0.51) | UC2b @ 5% | **0.511** [0.446, 0.549] (base 0.245; lift 2.1) |
| classes | UC2 / UC2b | **WATCHLIST / WATCHLIST** |

**Why it is limited.**
- No arm reaches an alert bar on the short list.
- The real headroom is the supplier's queue in the week of delivery (§4).
- **Not servable yet:** the forward-season input needs the production plan table, which serving code may not read.
  Phase 23AC prepared a **dormant** serving path with a single, bounded reader of that table. It awaits the owner's
  written decision (`docs/decisions/proposed_exception_part_demand_weekly.md`). It serves within 2.4e-7 of the stored
  predictions on CPU, not bit-exactly (`phase23ac/t2/t2_fill_exception.md`).

### 3.4 Supplier strain (capacity)

**What it does.** P10, P50 and P90 of supplier strain (demand ÷ capacity) over the next 90 days, and a ranked list of
channels likely to exceed capacity.

**How well (clean;** `phase22/stage4_capacity.md`**):**

| measure | value |
|---|---|
| right in top 1 / 5 / 10% | **0.828 / 0.735 / 0.679** (base 0.405) |
| recall at a validation-chosen 70% / 80% precision | 0.308 / 0.116; 85% unreachable on some seeds |
| per-snapshot precision @ 5%, min / median / max | **0.45 (2025-12-08)** / 0.64 / 0.89 |
| 80% interval coverage per quarter (level-aware conformal) | 0.76–0.83 |
| class | **WATCHLIST** (was ALERT on leaky inputs) |

**December 2025.**
- Strained channels were normal for the season (20%), but the model ordered them worse than in any validation month
  (within-month AUC 0.60 vs a validation low of 0.61).
- That is a **ranking failure in a quiet month**, not a new regime and not a stale threshold.
- **No recalibration scheme was adopted.** The ones that raised precision did it by flagging almost nothing.

**What it needs:**
- the product answer is a **monitor**: trailing precision and base rate on resolved snapshots, with the alert flag
  suppressed when precision falls below 0.60;
- the data answer is the supplier's **true** capacity (§9).

### 3.5 Shortage risk (head): retired

**Retired.** Re-weighted to a realistic 3% shortage rate, its top-5% precision is **0.224**, far below a 0.40 floor
(`phase-17.md` §1.1).

This figure was measured with the leaky panel (LEAKY-UNCONFIRMED). It is not re-measured, because the head is retired
regardless.

### 3.6 Predict-the-rescue: restated by Phase 23B (clean)

**What it does.** For each part-plant and each of the next 13 weeks, the chance that a planner will move stock in to save
it. It ranks the weeks a planner should look at first.

**How well (clean;** `phase-23b.md` §1, QUOTED: its predictions are on the other machine**):**

| | measure | value |
|---|---|---|
| clean LightGBM B1a, at Phase 17's operating point | precision / recall | **0.733** [0.7325, 0.7336] / **0.254** (base 0.4535, lift 1.62) |
| … 5-seed ensemble | snapshot-block 95% | [0.697, 0.773]; 3 of 9 snapshots below 0.70 |
| … top 1 / 5 / 10 / 20% | precision | 0.881 / 0.812 / 0.772 / 0.711 |
| beats the simulation proxy / the part-plant's own history | paired block | +0.115 [+0.089, +0.148] / +0.103 [+0.081, +0.125] |
| class | | **ALERT** (PARTIAL @ 0.80, lift 1.76): **marginal** |
| second world (same generator) | | beats its own history (+0.142), but at 0.596 it is a WATCHLIST there |

**Why it is limited.**
- About 0.09 of the published 0.823 was the leak: LEAKY (superseded, Phase 22).
- The margin over the 0.70 bar is thin, so present it as an alert with a watched precision.
- The alert threshold has to be set per site from its own base rate.

**Servable.** Phase 23B saved an identity-guarded, bit-exact bundle (`ml/serve/rescue.py`). Its files are on the other
machine; copy `ml/artifacts/phase23b/` to serve it here (`reports/part2/merge-23b-report.md` §5).

The neural version (B1b) is retrained clean in Phase 24 Stage 6 (`reports/part2/phase24/stage6_b1b_clean.md`).

### 3.7 Shortage simulation: restated on clean inputs (Phase 24)

*The board game*: plays the next 13 weeks forward 200 times per seed, drawing arrivals and fill from the models.

Phase 24 re-ran it with the same draws, swapping only the two models it reads (arrival and fill) for their clean retrains.
The wrapper first reproduced every published simulation number exactly with the original models
(`reports/part2/phase24/stage2_simulation.md`).

| | published (LEAKY (superseded, Phase 22)) | **clean** |
|---|---|---|
| matches the pre-rescue world | 1.129× | **1.137×** [1.134, 1.139]; the difference is undetermined |
| rescue-detection precision (Phase 17's point) | 0.619 | **0.612** (lower in every snapshot, by 0.003–0.010) |
| rescued weeks it misses | 66% | **66%** |
| below-safety-stock ceiling | 0.28 | **0.28** |
| class | WATCHLIST | **WATCHLIST** |

**The picture holds.** The simulation leaned on the stock, the plan and the reorder policy, not on the leak.
- **As a mechanism** it is good: within 14% of the pre-rescue world.
- **As a detector** it is weak.
- **Predict-the-rescue** (0.733) is the better rescue answer.

### 3.8 Delivery schedule (MILP), allocation, transfer

No model metric. One line each, as in `observation2.md` §3.6:

| use case | status | why |
|---|---|---|
| Delivery schedule (MILP) | blocked | needs holding, ordering, freight and shortage costs; 0 of 4 exist. *Grocery shopping with a small fridge: with no price on the fridge, the solver can say how much but not when* |
| Supplier allocation | blocked | 57 of 120 part-plants have no allowed split; on the 63 feasible it ties the status quo |
| Transfer recommendation | retired as an alert; tie-break only | its confidence is NOT TUNABLE (lift 0.03); "always yes" beats it on F1 |

---

## 4. The ceiling: how much better could anything get? (PARTLY STALE)

Phase 18 built an **oracle**: the same model allowed to read the generator's hidden state. It is never a feature.
**PARTLY STALE:** the oracle and its comparison model were measured on the **leaky** inputs. The *gaps* below are
therefore understated, and the "what it is made of" findings stand. Not re-measured here.

| use case · metric | model (LEAKY) | oracle | what the headroom is made of |
|---|---|---|---|
| arrival · lateness AUC | 0.709 | 0.9385 | **when the order will be raised**: the creation week alone gives 0.910 (`phase-19.md` §2) |
| fill · P(full) AUC | 0.620 | 0.918 | **the supplier's queue** that week |
| capacity · precision @ 5% | 0.851 | 1.000 | **the monthly capacity draw**, redrawn almost independently each month |

**Can stock data reach arrival's timing headroom?** At most 17% of it, even with the simulator's exact stock position
(0.705 → 0.739; `phase-20.md`).

> **In plain terms — the weather forecast and the picnic.** You can forecast tomorrow's weather. You cannot forecast when
> your friend will decide to have a picnic, even knowing their fridge, because that also depends on how hungry they get
> between now and then. Arrival lateness from today is mostly the picnic question. The order-time product (§3.2) answers
> the weather question instead: once the picnic is planned, when will the food arrive?

---

## 5. The forward-plan method, and replication scope

**The idea.** The client's forward requirement plan, recorded before the forecast date, says how much will be needed in
the next 13 weeks. Three families were built from it, all strictly as-of (`reports/phase18.md`, `phase-19.md`):
- **forward season:** the network-wide requirement ahead;
- **supplier-specific forward load;**
- **cadence:** each channel's ordering rhythm.

**On clean inputs (Phase 22):** forward season + cadence **passes** for fill against a control that scrambles the dates,
and gives the neural fill model a disjoint gain. The acknowledgement gap and the channel group statistics **fail**: read
at the wrong time for the right channel, they help just as much, so they are fixed channel traits, not news.

**Replication.** A second world from the same generator (seed 1002), each scored against its own baseline:
- The forward-plan gains replicated in direction in Phase 20, with effect sizes moving ×0.3 to ×2.6.
- On clean inputs, the fill consolidated gain replicates (AUC ×1.00, CRPS ×0.43), and so does the acknowledgement gap's
  *failure* (`phase22/stage3_fill.md` d).
- The order-time date's advantage over the promise date replicates; its edge over the channel's trailing median does not
  (§3.2).

**This is replication within one generator, not real-world evidence.**

---

## 6. What did not work, and is closed

| idea | closed by | the number | source |
|---|---|---|---|
| capacity as an alert | Phase 22 (clean restatement) | 0.735 at 5%; NO, CEILING | `phase-22.md` §1 |
| acknowledgement gap as a fill input | Phase 22 Stage 3 | its same-channel, wrong-date control gains as much | `phase22/stage3_fill.md` |
| channel group statistics (5-D groups) for arrival | Phases 21–22 | no gain at snapshots or at order time | `phase-21.md`, `phase22/stage2_order_time.md` |
| the order month as a key | Phase 21 | adds nothing at snapshots or at order time | `phase-21.md` |
| KM + LightGBM date blend | Phase 22 | validation gives the LightGBM date no weight | `phase22/stage2_order_time.md` |
| rolling recalibration for capacity | Phase 22 Stage 4 | no scheme adopted; precision rises only by flagging < 2% | `phase22/stage4_capacity.md` |
| shortage head as a product | Phase 17 | 0.224 at a 3% base | `phase-17.md` §1.1 |
| smarter graph encoders; five-band fill head; Δ-strain head | Phases 16–17 | ties or losses | `phase-16.md`, `phase-17.md` |
| global pulse; lane / checkpoint node | Phase 18 | worse; blocked (one lane per channel) | `reports/phase18.md` |
| cadence as a timing forecast; learned load forecaster; ledger stock arm | Phases 19–20 | 4%; 3–14%; +0.0016 | `phase-19.md`, `phase-20.md` |

---

## 7. Scorecards and lessons

| phase | right | wrong | source |
|---|---|---|---|
| 16 | 3 | 2 | `phase-16.md` |
| 17 | 5 | 1 | `phase-17.md` |
| 18 | 3 | 5 | `reports/phase18.md` |
| 19 | 4 | 4 | `phase-19.md` |
| 20 | 2 | 6 | `phase-20.md` |
| 21 | 5 | 3 | `phase-21.md` |
| 22 | 2 | 7 | `phase-22.md` |
| 23AC | 1 | 4 | `phase-23ac.md` |

**Lessons:**

1. **A gate that cannot fail is not a gate.** Every check carries a constructed input that must make it fire. The leak
   guard's own known-answer test (the nine columns must fail, their clean replacements must pass) is what made it
   trustworthy. Its first alignment rule failed that test and was amended in the open (Phase 23AC deviation 226).
2. **Check what a control preserves.** A shuffle that keeps the season is not a null (Phase 18). A cross-channel shuffle
   that keeps the network trend is not a null either (Phase 22, order time).
3. **Scorers need all five seeds.** A point unreachable on some seeds is UNREACHABLE, never a band (Phase 19).
4. **Leak guards.** A weak correlation with the label does not mean a column is safe: a forward-filled future value is
   *equal* to another line's outcome and only weakly correlated with this line's. Test by poisoning the future and by
   matching values to still-pending outcomes, on every run (`docs/standards/leak_guard.md`, `scripts/run_leak_guard.sh`).
5. **As-of means the day, not the week.** A weekly table read "this week" includes days after the decision. At order time
   that turned an alert into a watchlist (Phase 22).

---

## 8. Ship table (identical to `reports/part2/phase-22.md` §6)

| use case | ships now | watchlist | blocked on data | closed |
|---|---|---|---|---|
| **arrival (snapshot)** | the clean-retrained Phase 19 recipe as the point estimate (A3 12.57 d); the clean incumbent's P(late) as the ranked late list (0.948 at 5%). **Replace every stored arrival model with its clean retrain** | the late list (WATCHLIST, unchanged) | order timing (Phase 19), reorder state (Phase 20) | every leaky stored arrival model |
| **arrival (order time, NEW)** | **the expected-date product**: shrunk-KM median + month-specific 80% interval, servable (`ml/serve/order_time.py`) | the strict late-risk score (top 5%, precision 0.62) | — | the KM + LightGBM blend (no weight); L4 group statistics at order time |
| **fill** | the clean incumbent retrained **with season + cadence** (GAIN) as the fill distribution; the clean Phase 19 recipe as the materially-short list (0.511) | the LightGBM consolidated model (best CRPS and UC2; **not servable** under constraint 4) | a forward-plan source other than `part_demand_weekly`, or a recorded exception to constraint 4 | ack-gap and L4 as fill inputs; every leaky stored fill model |
| **capacity** | **nothing as an alert.** The strain quantiles with the level-aware interval (≈ 80% per quarter) | the clean incumbent's top list (WATCHLIST, 0.735 at 5%) behind the trailing precision monitor | the supplier's declared-capacity history and any observed capacity evidence (client ask) | rolling recalibration (no scheme adopted); the leaky alert |

**Since Phase 22, nothing ships automatically.** Phase 23AC added:
- a planner card for the order-time product;
- a **dormant, proposed** fill serving path (awaiting the owner's decision);
- the standing leak guard.

Predict-the-rescue is restated (clean 0.733, ALERT, marginal; Phase 23B) and the simulation is restated (Phase 24); neither is in
the Phase 22 ship table above, which is reproduced unchanged.

---

## 9. What we need from the client, and what each is worth

Full request: `docs/client/data_request_v2.md`. Values are synthetic-world measurements, to be re-measured on Rane's data.

| ask | why | measured value | the honest limit |
|---|---|---|---|
| dated plan / forecast versions (Ask 1) | the forward-plan features need the plan as it stood on each past date | fill: forward season + cadence gives a disjoint neural gain on clean inputs (CRPS 0.1397 → 0.1375, AUC 0.601 → 0.640) | effect sizes moved ×0.3–×2.6 between two worlds of one generator |
| channel stock position and as-of reorder points (Ask 2) | arrival's headroom is order timing | at most +0.034 lateness AUC (an upper bound, measured on leaky inputs) | the other 83% is the future demand path |
| holding, ordering, freight and shortage costs (Ask 3) | MILP, allocation and scheduling have no objective without them | not measurable without them | — |
| declared-capacity history and observed capacity evidence (Ask 4) | capacity's headroom is the true monthly capacity, and its alert was lost with the leak | true capacity was worth +0.09 precision @ 5% in principle (oracle, PARTLY STALE) | no source in v8 tracks it |
| PO line creation history with recorded timestamps, and reorder rules (Ask 5) | the order-time product needs as-of receipts; arrival's headroom is order timing | the order-time date beats the promise date by 3.7 days median error | measured in one synthetic generator |

---

## 10. Honest scope and the standing caveat

Twenty-three phases in, on clean inputs, ChainPilot is:
- **one alert, and it is marginal**: predict-the-rescue (clean 0.733, interval down to 0.697);
- **one new product that works**: the order-time expected date and interval;
- **two improved forecasts:** arrival's point estimate, and fill's distribution driven by the forward plan;
- **ranked watchlists** for lateness, fill and capacity;
- **a shortage simulation** that matches the pre-rescue world (1.137×) but is a weak detector;
- **use cases blocked on client data:** two on things the client can supply (costs; the allocation rule), and three on
  data that may not exist (order timing, the supplier's queue, true capacity).

**The standing caveat.** Every figure is measured within one synthetic world. Phase 20's second world comes from the same
generator, so it tests stability, not reality. **Every absolute figure must be re-measured on Rane's extract** before it is
quoted outside the project. What transfers is the **method**:
- forecasts strictly as-of the day;
- censoring modelled;
- controls checked for what they preserve;
- five-seed bands;
- a leak guard on every input.

---

## Appendix A — Numbers register

CLEAN unless marked LEAKY. RAW. v8 seed 1001, test 2025, unless marked.

| metric | value | band / interval | seeds | RAW/RECAL | split / world | source |
|---|---|---|---|---|---|---|
| base rate late vs contract (forecast date) | 0.730 | — | — | — | test | `phase-15.md` §1 |
| base rate late vs contract (order time) | 0.349 | — | — | — | test | `phase22/stage2_order_time.md` |
| base rates fill = 1 / fill < 0.95 / strain > 1 / rescue | 0.749 / 0.245 / 0.405 / 0.4535 | — | — | — | test | `phase-15.md` §1; `phase-17.md` §2.1 |
| leaking columns | 9 of 15 | rebuild reproduces 100% | — | — | v8 | `phase22/stage1_leak.md` |
| share of label rows the leak reaches | 93.0–95.7% | — | — | — | train / val / test | `phase22/stage1_leak.md` |
| arrival lateness AUC, clean neural | 0.6951 | [0.6817, 0.7041] | 5 | RAW | test | `phase22/stage1_leak.md` |
| arrival UC1 precision @ 5%, clean neural ensemble | 0.948 | [0.930, 0.965] block | ensemble | RAW | test | `phase23ac/t4/metrics_pack.md` |
| arrival A3, clean Phase 19 recipe | 12.57 d | [12.22, 13.05] block | ensemble | RAW | test | `phase23ac/t4/metrics_pack.md` |
| arrival A3, clean neural | 13.15 d | — | 5 | RAW | test | `phase22/stage1_leak.md` |
| order-time A3 (product / promise / trailing mean / trailing median) | 10.0 / 14.0 / 12.3 / 11.0 d | product vs promise +3.75 [2.73, 4.31] | det. | — | test | `phase23ac/t1/t1_order_time.md` |
| order-time share within ±7 d | 37.8% / 27.1% / 32.0% / 35.4% | vs promise +10.7 pt [9.1, 12.3] | det. | — | test | same |
| order-time signed bias, product | −8.4 d | — | det. | — | test | same |
| order-time 80% interval coverage / width | 0.803 / 54 d | [0.797, 0.808]; months 0.767–0.829 | det. | RECAL (conformal) | test | same |
| order-time A3 world 2 (product / trailing median) | 8.8 / 9.0 d | difference undetermined [−0.05, +0.39] | det. | — | test, v8w1002 | same |
| order-time strict late flag precision @ 5% | 0.615 | [0.607, 0.619] | 5 | RAW | test | `phase23ac/t4/metrics_pack.md` |
| order-time τ-week flag precision @ 5% | 0.794 | — | 5 | RAW | test | `phase22/stage2_order_time.md` |
| fill CRPS / AUC, clean incumbent | 0.1397 / 0.6007 | [0.1396, 0.1399] / [0.5976, 0.6050] | 5 | RAW | test | `phase22/stage1_leak.md` |
| fill neural + season + cadence CRPS / AUC / UC2b | 0.1375 / 0.640 / 0.464 | [0.1370, 0.1377] / [0.6390, 0.6419] / [0.459, 0.470] | 5 | RAW | test | `phase22/stage3_fill.md` |
| fill consolidated blend CRPS / UC2 @ 5% | 0.13444 / 0.944 | vs clean Phase 19 blend +0.0013 [0.0003, 0.0025] / +0.030 [0.008, 0.072] | ensemble | RAW | test | same |
| fill UC2b @ 5%, clean Phase 19 recipe | 0.511 | [0.446, 0.549] | ensemble | RAW | test | `phase23ac/t4/metrics_pack.md` |
| dormant fill path vs stored | max \|Δ\| 2.38e-7 | per seed ≤ 4.17e-7 | 5 | — | test, 200 rows | `phase23ac/t2/t2_fill_exception.md` |
| capacity precision @ 1 / 5 / 10%, clean | 0.828 / 0.735 / 0.679 | @5% [0.703, 0.784] | 5 | RAW | test | `phase22/stage1_leak.md` |
| capacity recall @ p 0.70 / 0.80 / 0.85, clean | 0.308 / 0.116 / UNREACHABLE | — | 5 | RAW | test | same |
| capacity worst snapshot precision @ 5% | 0.45 (2025-12-08) | — | 5 | RAW | test | `phase23ac/t4/progress.md` |
| capacity Dec 2025 within-month AUC | 0.601 | val min 0.606 | 5 | RAW | test | `phase22/stage4_capacity.md` |
| capacity interval coverage per quarter | 0.764–0.829 | — | 5 | RECAL (conformal) | test | same |
| LEAKY: capacity precision @ 5% (published) | 0.851 | [0.832, 0.860] | 5 | RAW | test | `phase-15.md` §1 |
| LEAKY: arrival UC1 @ 5% (published ensemble) | 0.969 | — | ensemble | RAW | test | `phase20/stage1_decisions.md` |
| rescue B1a precision at Phase 17's point, clean (QUOTED) | 0.733 at recall 0.254 | [0.7325, 0.7336]; ensemble block [0.697, 0.773] | 5 | RAW | test | `phase-23b.md` §1 |
| LEAKY: rescue B1a precision @ recall 0.246 (published) | 0.823 | [0.822, 0.824] | 5 | RAW | test | `phase-17.md` §2.1 |
| simulation rescue-detection precision, clean / published | 0.612 / 0.619 | [0.612, 0.613] / [0.619, 0.620]; Δ block [−0.008, −0.004] | 5 fill | RAW | test | `phase24/stage2_simulation.md` |
| simulation rescued weeks missed, clean / published | 66.2% / 65.9% | Δ block [−0.8, +1.3] pt | 5 fill | — | test | same |
| simulation below-SS ceiling, clean / published | 0.282 / 0.284 | — | 5 fill | RAW | test | same |
| simulation vs pre-rescue, clean / published | 1.137× / 1.129× | [1.134, 1.139] / [1.121, 1.139]; Δ block [−0.019, +0.032] | 5 fill | — | test | `phase24/stage2_simulation.md`; `phase-13.md` |
| LEAKY-UNCONFIRMED: shortage head @ 5%, 3% base | 0.224 | [0.223, 0.226] | 5 | RAW | test | `phase-17.md` §1.1 |
| PARTLY STALE: oracle arrival / fill / capacity | 0.9385 / 0.918 / 1.000 | — | 5 | RAW | test | `phase18/oracle/PRIVILEGED__stage2_oracle.md` |
| recomputed vs published figures | 79 of 79 reproduce | — | — | — | — | `phase23ac/t4/metrics_pack.md` |

## Appendix B — Errata: every `observation2.md` (v3.0) figure superseded

Starting from the Phase 22 errata table (`phase-22.md` §6).

| v3.0 said (location) | now | superseded by |
|---|---|---|
| summary row 3: "capacity ships as an alert … 81% right" (l. 31) | **WATCHLIST**; 0.735 at 5% | `phase-22.md` §1 |
| §3.3 heading "the one alert"; §10 "one alert: supplier capacity strain" | **no alert** | same |
| arrival lateness AUC 0.709 (l. 124, 397) | 0.695 (clean neural) | `phase22/stage1_leak.md` |
| UC1 precision @ 5% 0.969 / blend 0.943 (l. 128, 407) | 0.948 / clean recipe 0.919 | same |
| Phase 19 blend 0.731 / 12.34 d (l. 126, 403–404) | clean recipe 0.722 / 12.57 d | same |
| fill P(full) AUC 0.620 → 0.643 (l. 151–152, 408–409) | 0.601 → 0.640 (clean, + season + cadence) | `phase22/stage3_fill.md` |
| fill CRPS 0.13876 → 0.1371 → 0.1354 (l. 153, 410) | 0.1397 → 0.1375 → consolidated 0.1344 | same |
| UC2b 0.464 → 0.530 (l. 154, 411) | 0.401 → clean Phase 19 recipe 0.511 | same |
| capacity 0.904 / 0.851 / 0.810; recall 0.515 / 0.318 / 0.191 (l. 177, 412–413) | 0.828 / 0.735 / 0.679; recall 0.308 / 0.116 / UNREACHABLE | `phase22/stage1_leak.md` |
| capacity per-snapshot min 0.46 (l. 414) | 0.45 (clean), same snapshot | `phase23ac/t4/progress.md` |
| oracle table (l. 231–233) | PARTLY STALE: measured against leaky models | §4 here |
| any order-time ALERT quoted from Phase 21 | τ-week flag 0.794 (ALERT); strictly as-of 0.615 (WATCHLIST) | `phase22/stage2_order_time.md` |
| quotable claim 1 (capacity alert) | withdrawn | `phase-22.md` |
| quotable claims 2 and 4 (12.3 d; 53% materially short) | replaced: 12.6 d; 51% | Appendix C here |
| quotable claims 5 and 6 (rescue 82%; simulation 13%) | rescue 73% (clean); simulation 14% (clean) | Appendix C here |
| **v4.0** summary row 4b, §3.6: predict-the-rescue PENDING, 0.823 (LEAKY-UNCONFIRMED) | clean 0.733, ALERT, marginal | `phase-23b.md` §1 |
| **v4.0** summary row 5, §3.7: simulation PENDING, 1.129× (LEAKY-UNCONFIRMED) | clean 1.137×; detection 0.612; misses 66% | `phase24/stage2_simulation.md` |
| **v4.0** §0 "No use case is an alert any more"; §10 "no alert" | one marginal alert: predict-the-rescue | `phase-23b.md` §1 |
| **v4.0** Appendix A: rescue and simulation rows LEAKY-UNCONFIRMED | clean rows added; old rows labelled LEAKY | Appendix A here |

## Appendix C — Quotable claims (each survives a five-seed band or the block bootstrap; clean)

1. *"When a planner raises a purchase order, our expected receipt date is off by a median of 10 days, against 14 for the
   promise date and 11 for the channel's own average."* (creation-week block bootstrap; `phase23ac/t1/t1_order_time.md`)
   Label: on a second simulated world it ties the channel average; it is early on average by about a week.
2. *"Our 80% receipt-date range contains the actual date 80% of the time, in every month of the year."* (`phase23ac/t1`)
   Label: January 77%.
3. *"Of the top 5% of order lines we flag as materially short, 51% are, against a background of 25%."* (block bootstrap;
   `phase23ac/t4/metrics_pack.md`)
4. *"Our predicted delivery date for open orders is off by a median of 12.6 days."* (block bootstrap; same)
5. *"Adding your forward plan improves our fill forecast, and the improvement held on a second simulated world."*
   (5-seed bands; `phase22/stage3_fill.md`) Label: same generator.
6. *"We check every input every run for information from the future, and we found and removed nine such columns."*
   (`phase22/stage1_leak.md`, `docs/standards/leak_guard.md`)

7. *"Of the weeks our model says a planner will need to move stock in, 73% are, against a background of 45%."* (5-seed band
   and block bootstrap; `phase-23b.md` §1) Label: it catches a quarter of such weeks, and the interval reaches 70%.
8. *"Our shortage simulation reproduces the stock picture before planners' rescues to within 14%."* (5 fill seeds;
   `phase24/stage2_simulation.md`) Label: as a week-by-week detector it is weak (it misses two-thirds of rescued weeks).

## Appendix D — Open deviations

| # | what | status | source |
|---|---|---|---|
| 122 | Phase 12 B3 changed the default simulation's random stream | **open**; worked around (pinned `order_policy.py`) | `phase-14.md` |
| 46 | `shipped.json` names a LightGBM the serving path cannot load | addressed, not removed (no LightGBM model was ever saved) | `integration-report.md` |
| 209 | the weekly panel row includes days after the snapshot date | **open as a convention**; worth ≈ 0.005 AUC at snapshots, decisive at order time | `phase-22.md` §3 |
| 219 | the best fill model is not servable (forward-plan table forbidden in serving code) | **open**: proposed exception awaiting the owner | `phase-22.md`; `docs/decisions/proposed_exception_part_demand_weekly.md` |
| 226 | the leak guard's label-alignment rule amended after a smoke run (0 / 1 atoms excluded) | **open**: reported beside the literal rule | `phase-23ac.md` |
| 180 | the "second world" equals the stored seed-1002 world | **stands** as a scope limit (one generator) | `phase-20.md` §3 |
| 249 | the simulation harness hard-codes its model paths | **resolved** by a new-file wrapper that swaps only those two paths and reproduces the published run exactly | `phase24/stage2_simulation.md` |
| 264 | B1b clean runs under a 40-minute-per-seed wall cap (Phase 17's had none) | **open**: a capped seed is a floor | `phase-24-preregistration.md` |
