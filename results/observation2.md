# Observation 2 — What ChainPilot predicts, what ships, and what it would take to do better

**Version 3.0.** Written after Phases 16–20 and their integration into `HADES-v4-ml-pipeline` (2026-10-04). It
**supersedes `observation1.md` (v2.0, after Phase 15)** and stands on its own: you do not need v2.0 to read it.
Appendix B lists every v2.0 figure or conclusion that a later report corrected.

**Who this is for:** a client lead or product owner who wants to know what the system can be trusted to do, without
reading twenty phase reports. Each section says *what it does, how well, why it is limited, and what ships*.

**Where the numbers come from.** Nothing new was measured for this document. Every number is copied from a committed
report, named beside it, and registered in Appendix A. Unless a world is named, every figure is from the synthetic
**v8 world, dataset seed 1001, fixed split** (train up to 2023, validation 2024, test 2025), on the **test** year, with
**five model seeds** where a band `[min, max]` is shown. "RAW" means straight from the model; "recalibrated" means
corrected on the validation year. The two are never mixed in one comparison.

**Read this first: the standing caveat (§10).** Every figure is measured inside one synthetic world made by one
generator. Effect sizes moved by a factor of 0.3 to 2.6 between two worlds from that same generator (§5). The figures
are an upper bound on what a real extract will give, and must be re-measured on Rane's own data.

---

## 0. One-page summary

ChainPilot looks at each **sourcing channel**, one supplier supplying one part to one plant (16,072 of them), and
forecasts eight things.

| # | use case | the question | **status** | the number that decides it |
|---|---|---|---|---|
| 1 | **Arrival timing** | when will the order arrive, and will it be late? | **ships: a point estimate + an 80% interval; the late-list stays a ranked watchlist** | the Phase 19 blend's median error 12.34 days (vs 13.07) is declared but not yet servable; its servable neural stand-in gives 12.90; the late-list keeps the incumbent's ranking, which is better at the top (0.969 vs 0.943) |
| 2 | **Fill rate** | how much of the order will arrive? | **watchlist** (better than before, still below any alert bar) | "materially short" list: 53% right in the top 5%, on a 24.5% base |
| 3 | **Supplier strain (capacity)** | will demand exceed what the supplier can deliver in the next 90 days? | **ships as an alert**, the one alert in the project | 81% right, catches 19%, flags about 1 in 10; one quarter (Dec 2025) ran at 46% |
| 4 | **Shortage risk (head)** | which part-plants will run short? | **retired** | at a realistic 3% shortage rate it is right 22% of the time in its top 5% |
| 4b | **Predict-the-rescue** *(new)* | which weeks will planners have to move stock to save? | **GO** (LightGBM), declared but not yet servable | 82% right at a quarter of the rescues, vs 62% for the simulation it replaces |
| 5 | **Shortage simulation** | how much short, in which week? | **calibrated as a mechanism, weak as a detector**; its fix (channel granularity) is **closed** | 1.129× the pre-rescue world; misses two-thirds of rescued weeks |
| 6 | **Delivery schedule (MILP)** | how much to order, in which week? | **blocked on costs** | 0 of 4 cost parameters exist |
| 7 | **Supplier allocation** | which supplier gets what share? | **blocked on a rule** | 57 of 120 part-plants have no allowed split |
| 8 | **Transfer recommendation** | which plant should send stock to which? | **retired as an alert** | its confidence carries no information (lift 0.03) |

**What changed since v2.0, in four lines.**
1. **The models were not short of architecture, they were short of information.** Every encoder change in Phases 16–17
   tied. Feeding the models the client's **forward plan** moved arrival and fill, and gave the first disjoint neural gain
   over a stored incumbent since Phase 15 (fill).
2. **The gains are real but did not change any decision class.** Arrival's better score lives in the middle of the
   ranking, not the top. Fill's better list is still a watchlist.
3. **We now know what the remaining headroom is made of**, and most of it is out of reach of data available at forecast
   time: arrival's is *when the order will be raised*, fill's is *the supplier's queue that week*, capacity's is *the
   supplier's true capacity that month*.
4. **Two use cases closed** (shortage head, simulation granularity fix), **one opened** (predict-the-rescue).

---

## 1. Two ways to read every number: metric space and decision space

A model can be *better* without being *useful*. Two kinds of number in this document answer different questions:

- **Metric space** (AUC, CRPS, pinball loss): *"is model A better than model B at ordering or describing outcomes?"*
- **Decision space** (precision, recall, lift at an operating point): *"if a planner acts on this, how often are they right?"*

**The base rates decide almost everything.** These are the share of cases where the answer is "yes":

| decision | base rate (v8 test) | source |
|---|---|---|
| line arrives late against its contract | **0.730** | `phase-15.md` §1 |
| line arrives exactly in full | **0.749** | `phase-15.md` §1 |
| demand exceeds supplier capacity (strain > 1) | **0.405** | `phase-15.md` §1 |
| line is materially short (fill < 0.95) | **0.245** | `phase-15.md` §1 |
| a part-plant week is rescued by a planner transfer | **0.4535** | `phase-17.md` §2.1 |

> **In plain terms — the weather forecaster in the desert.** If it is sunny 73 days in 100, a forecaster who always says
> "sunny" is right 73% of the time and has told you nothing. That is why **no accuracy figure in this document appears
> without the always-guess-the-majority figure beside it**, and why the honest measures are **lift** (how much better than
> the base rate are the cases the model flags?) and **precision at coverage** (if the model only speaks about its most
> confident 5%, how often is it right?).

**An AUC gain is not a decision.** Phase 20 is the clearest example. The Phase 19 arrival blend lifts lateness AUC from
0.709 to 0.731, a real improvement in ordering lines overall. But on the actual decision, *"flag the lines most likely to
be late"*, the blend's top 5% is right **0.943** of the time against the incumbent's **0.969**, and it is worse in **9 of
9** test snapshots (`phase-20.md` §1; `phase20/stage1_decisions.md`). The blend is better in the middle of the list and
worse at the top, which is where a planner looks.

---

## 2. The shared foundation: a trajectory reader and a huddle

Three of the learned models share one encoder, in two stages: **time first, then structure** (unchanged since v2.0).

1. **The trajectory reader (TCN).** Each channel's last 52+ weeks of measured behaviour (deliveries, lead times, fill,
   load) are compressed into one summary. *Like a doctor reading a long chart: the last weeks day by day, the last year
   month by month.* It only reads the past; the future pages are physically covered.
2. **The huddle (graph pass).** Channels exchange that summary with their supplier, part and plant, for one or four rounds.
   *Like a morning huddle: a channel learns that its supplier is struggling on eleven other parts.*

**Does the huddle help?** Yes, and it was tested by sabotage (v2.0, Phase 11A): scramble *who* reports to whom while
keeping every group's size, and forecasts got worse on all ten world-task combinations (`observation1.md` §1.3,
`phase-11a.md`). The help comes from the **supplier and plant** links; removing the **part** link changes nothing
measurable (C3 ablation, `phase-12.md`).

**What Phases 16–17 added to this picture:**
- **Two smarter huddles tied.** Trajectory-similarity attention ("listen harder to channels that move like you") and
  degree-aware aggregation (PNA) both tied the incumbent; one variant was worse on test C-index (`phase-16.md`). Neither
  shipped.
- **The lean encoder ties and is 30.7% smaller.** Dropping the part link and 49,152 dead parameters gives test C-index
  0.67429 vs 0.67442 and lateness AUC 0.70747 vs 0.70905, all bands overlapping, with **369,164 parameters instead of
  533,004** (`phase-17.md` §2.4). It is the same function as the C3 ablation.

> **Lesson.** Five phases of encoder work tied. The bottleneck was never how the huddle talks; it was what the room knew.

---

## 3. The use cases

### 3.1 Arrival timing

**What it does.** For a purchase-order line, a distribution over the week it will arrive, from which come (a) how late it
is likely to be against its channel's normal lead, in days, and (b) a ranked list of the lines most likely to run late.

**How the incumbent works.** A *hazard* model: for each week it predicts "given it has not arrived yet, will it arrive this
week?" — which uses the 48.5% of lines still in transit when the 90-day window closes instead of discarding them
(`observation1.md` §2.2).

**How well.**
| measure | value | source |
|---|---|---|
| lateness ranking (AUC, as-of reference), incumbent | **0.709** [0.706, 0.713] | `phase-17.md` §2.4 |
| median error of the predicted lateness, incumbent | **13.06 days** [12.86, 13.30] vs 14.00 for "use the channel's usual lead" | `phase-17.md` §1.3 |
| the same, Phase 19 blend (neural + LightGBM, both fed the forward plan) | **12.34 days**; AUC **0.731** | `phase-19.md` §1, §7 |
| P10–P90 interval coverage | raw 93% at 52 days wide; **re-centred on the expected week: 80% at 45 days** | `phase-19.md` §1 (stage 4) |
| late-list (UC1: late vs contract), top 5% | incumbent ensemble **0.969**, blend 0.943 (worse, 9 of 9 snapshots) | `phase-20.md` §1 |
| late-list lift | at most 1.36× (base 0.73) → **WATCHLIST** | `phase20/stage1_decisions.md` |

**Why it is limited.** Three-quarters of lines run late against contract anyway, so a yes/no alert has almost no room
(lift ≤ 1.36×). And the remaining headroom is **when the order will be raised**: the label counts weeks from the forecast
date, and most of the uncertainty is the wait before the order exists (§4).

**What ships** (`docs/decisions/shipped_phase20.md`).
- **Point estimate:** the Phase 19 blend (12.34 days median error) is the decision, but it is **declared, not servable**:
  its LightGBM half was never saved as a model. Until it is, the servable **stand-in** is the Phase 19 neural ensemble
  (12.90 days, still better than the incumbent's 13.07 under the snapshot-block bootstrap).
- **Interval:** the **expected-week conformal interval** on the incumbent ensemble (≈ 80% coverage at ≈ 45 days).
- **Ranked late-list:** the **incumbent neural ensemble's P(late)**, as a watchlist, not an alert.

### 3.2 Fill rate

**What it does.** A distribution over the fraction of an order that will arrive, with explicit spikes at "nothing" and
"exactly complete". *Like predicting a batsman's score: most innings end at 0 or not out, and a smooth curve cannot put a
spike on exactly zero.*

**How well.**
| measure | value | source |
|---|---|---|
| P(fill = 1) ranking (AUC), incumbent neural head | **0.620** | `phase-17.md` §2.3 |
| same, **neural head + forward season + cadence** (Phase 19) | **0.643** [0.642, 0.644], **disjoint GAIN** | `phase-19.md` §1 |
| exact CRPS (lower better): incumbent → Phase 19 neural → Phase 19 blend | 0.13876 → **0.1371** → **0.1354** | `phase-19.md` §1, `phase19/stage4_intervals.md` |
| "materially short" list (UC2b), precision in top 5% | incumbent 0.464 → **Phase 19 blend 0.530** (+0.063 [0.043, 0.090]); top 1% 0.644, 2.6× base | `phase20/stage1_decisions.md` |
| shipped `b5flat22` on the same list | **0.384**, the weakest fill arm on every decision metric | `phase20/stage1_decisions.md` |

**Why it is limited.** No arm can hold 70% precision with even 10% recall on the short-list (`phase-20.md` §2, P2). The
oracle (§4) says the missing information is the **supplier's queue that week**: how much else it was asked for. That is
not visible at forecast time.

**What ships.** The Phase 19 blend is the decision for the **materially-short watchlist**, but it is **declared, not
servable** (its LightGBM half was never saved). The servable **stand-in** is the Phase 19 neural season + cadence ensemble,
the arm with the disjoint gain. "Arrives in full" (UC2) stays a displayed probability, not a yes/no. LightGBM `b5flat22`,
which the configuration named since Phase 10 but could never serve, is recorded as superseded; the Phase 10 reversal
condition is re-read in `docs/decisions/shipped_phase20.md`.

### 3.3 Supplier strain (capacity) — the one alert

**What it does.** Three quantiles (P10, P50, P90) of supplier strain (demand ÷ capacity) over the next 90 days, and from
them the probability that demand exceeds capacity. *The tailor cutting cloth: deliberately cut the P90 long, because
falling short costs more than overshooting.*

**How well.**
| measure | value | source |
|---|---|---|
| alert at the validation-chosen p = 0.85 point | **0.810 precision at 0.191 recall**, ~10% of channel-horizons flagged, lift 2.0× | `phase-15.md` §1 |
| precision at 1 / 5 / 10% coverage | 0.904 / 0.851 / 0.810 | `phase-15.md`, re-confirmed `phase-17.md` §2.2 |
| 5-seed ensemble, precision at 5% | 0.865 | `phase20/stage1_decisions.md` |
| **per-snapshot spread at 5% coverage** | **0.46 (Dec 2025)** to 0.91; median 0.81 | `phase20/stage1_decisions.md` |
| forward plan (supplier-specific) in the neural head | **tie** (precision 0.886 vs 0.851, recall 0.506 vs 0.515, bands overlap) | `phase-19.md` §1 |

**Why it is limited.** About 60% of the gap to the oracle is the **supplier's true capacity that month**, which the
generator draws almost independently each month. The declared capacity tracks it at a correlation of 0.11. No forecast-time
data can see it (§4).

**What ships.** The **incumbent neural ensemble as the UC3 alert**, labelled *"81% right, catches 19%, flags about 1 in
10"*, with the caveat that one quarter ran at 46%.

### 3.4 Shortage risk (retired) and predict-the-rescue (GO)

**Shortage head — retired.** v2.0 kept it as an internal watchlist pending a re-measure at a realistic base rate. Phase 17
did that re-measure: re-weighted to a 3% shortage rate, its precision in the top 5% is **0.224** (both re-weighting methods
agree), far below the 0.40 floor (`phase-17.md` §1.1). Stop spending on it.

**Predict-the-rescue — new, GO.** Instead of predicting shortages, predict the **weeks planners will act on** (a
`transfer_in` recorded). LightGBM (B1a) reaches **0.823 precision [0.822, 0.824] at recall 0.246**, against **0.619** for
the simulation-based proxy and 0.631 for a part-plant's own transfer history, on a 45.4% base (`phase-17.md` §2.1). A
neural version reached 0.859 on **2 of 5 seeds only**: undetermined against LightGBM, and the GO rests on LightGBM. Both
arms hit their training caps, so these are floors. **Not yet servable:** no LightGBM model was saved, and the Phase 17
artifacts live only on the Windows machine.

### 3.5 Shortage simulation

**What it does.** Plays the next 13 weeks forward a thousand times (*the board game*), drawing arrivals and fill from the
models, and counts how often stock falls below safety.

**Status (unchanged since v2.0, plus one closure).** It reproduces the world *before* planner rescues to within 13%
(1.129×, falsified three ways; `phase-13.md`, `observation1.md` §6.2). As a detector it is weak: it never flags two-thirds
of the weeks planners rescued, and its precision ceiling is 0.28 (`phase-15.md`). v2.0's planned fix, **restating it at
channel granularity, is closed**: missed rescues do not come from part-plants with more channels (P = 0.51); they come from
drawdowns that develop **late in the horizon from healthy starting stock** (`phase-17.md` §1.2). Predict-the-rescue (§3.4)
answers the planner's question more directly.

### 3.6 Delivery schedule (MILP), allocation, transfer recommendation

| use case | status | why | source (last report to touch it) |
|---|---|---|---|
| Delivery schedule (MILP) | **blocked** | needs holding, ordering, freight and shortage costs; 0 of 4 exist. *Grocery shopping with a small fridge: with no price on the fridge, the solver can say how much but not when.* | `phase-15.md` (UC6) |
| Supplier allocation | **blocked** | 57 of 120 part-plants have no allowed split (the qualification rule rejects even the incumbent); on the 63 feasible it ties the status quo | `phase-15.md` (UC7), `phase-14.md` |
| Transfer recommendation | **retired as an alert**; tie-break suggestion only | its confidence is NOT TUNABLE (lift 0.03); "always yes" beats it on F1 | `phase-15.md` (UC8) |

---

## 4. The ceiling: how much better could anything get?

Phase 18 built an **oracle**: the same model, but allowed to read the generator's hidden state. It is never a feature; it
measures how much signal exists in principle (`reports/phase18.md` §2, `phase18/oracle/PRIVILEGED__stage2_oracle.md`).

| use case · metric | model | oracle | headroom | what the headroom is made of |
|---|---|---|---|---|
| arrival · lateness AUC | 0.709 | **0.9385** | +0.23 | **when the order will be raised**: knowing only the creation week (no other state) gives **0.910** (`phase-19.md` §2); state without timing gives 0.709 |
| fill · P(fill = 1) AUC | 0.620 | **0.918** | +0.30 | **the supplier's queue**: true capacity and supplier state alone +0.13; knowing what else was ordered that week another +0.17 |
| capacity · precision @ 5% | 0.851 | **1.000** | +0.15 | **the monthly capacity draw K**: true K alone gives 0.944; K is redrawn almost independently each month (correlation 0.09), and the declared figure tracks it at 0.11 |

**Can stock data reach arrival's timing headroom?** Phase 20 tested the best case: the simulator's **exact** inventory
position against its **exact** reorder point, per channel. It lifts lateness AUC only from **0.705 to 0.739**, i.e.
**17%** of the timing headroom (`phase-20.md` §1, `phase20/stage4_reorder.md`). When an order is raised also depends on
demand that has not happened yet. The stock ledger v8 already publishes is worth **+0.0016**.

> **In plain terms — the weather forecast and the picnic.** You can forecast tomorrow's weather well. You cannot forecast
> *when your friend will decide to have a picnic*, even if you know how full their fridge is, because that also depends
> on how hungry they get between now and then. Arrival lateness, measured from today, is mostly the picnic question.

---

## 5. The forward-plan method

**The idea (Phase 18).** The labels are about orders raised **after** the forecast date, but the models only saw the past.
The client's **forward requirement plan**, recorded before the forecast date, says how much will be needed in the next
13 weeks. Three families were built from it, all strictly as-of (every source row recorded before the forecast date,
asserted):

- **forward season**: the network-wide total requirement ahead, the same number for every channel (*"the whole network is
  heading into a busy quarter"*);
- **supplier-specific forward load**: each supplier's and channel's share of that requirement against its recent
  throughput;
- **cadence**: each channel's ordering rhythm (days since its last order, typical gap between orders).

**The fair test (Gate v2, Phase 19).** Phase 18's first test failed because its control (shuffling suppliers within a
date) kept the network-wide season intact, and that was real information (`phase18/stage4_forward_load.md`). Phase 19
re-tested with a control that moves each date's plan to a **different date** (`phase19/stage1_gate.md`):

| use case | forward season | supplier-specific forward load | cadence |
|---|---|---|---|
| arrival | PASS | PASS | PASS |
| fill | PASS | FAIL (its gain survives the date scramble: a fixed channel trait, not a forecast) | PASS |
| capacity | FAIL (worse at 1% coverage) | PASS* | FAIL |

\* Passes only once the scorer requires all five seeds for a band (Phase 19 deviation 171; see §7).

**Replication on a second world (Phase 20).** A world generated from seed 1002, scored against **its own** baseline:

- **5 of 6 passing cells replicate** (same direction, separated bands). Supplier-specific forward load on arrival
  replicates at the same size (×0.99). **Cadence on fill does not replicate** (`phase20/stage2_replication.md`).
- **Effect sizes moved ×0.3 to ×2.6**, and that world's base rates differ (strain > 1: 20% vs 41%; full fill: 84% vs 75%).
- **The neural half of the Phase 19 blends was not replicated** (that phase was CPU-only). With LightGBM alone, blending
  never beat its better parent.
- **Honest scope:** the "fresh" world is byte-identical to the stored seed-1002 world and comes from **the same
  generator**. This is replication **within one generator**, not real-world evidence.

**A learned load forecaster does not pay.** Predicting each channel's future orders and feeding that in passes on
arrival and fill but recovers only **3–4%** (arrival) and **10–14%** (fill) of the gap to perfect hindsight
(`phase20/stage3_forecaster.md`).

---

## 6. What did not work, and is closed

| idea | closed by | the number | source |
|---|---|---|---|
| shortage head as a product | realistic base-rate re-measure | precision @ 5% **0.224** at a 3% base (floor 0.40) | `phase-17.md` §1.1 |
| Δ-strain capacity head (predict movement) | Phase 17 B2 | never reaches 0.70 precision on validation (max 0.569 / 0.662); the trailing level correlates **0.004–0.09** with the next 90 days | `phase-17.md` §2.2 |
| five-band fill head | Phase 17 B3 | raw CRPS worse (0.13945 vs 0.13876, disjoint); recalibrated tie | `phase-17.md` §2.3 |
| channel-granularity simulation rebuild | Phase 17 A2 | missed rescues no more channel-heavy (P = 0.51) | `phase-17.md` §1.2 |
| capacity horizon stacking (30/60/120 days) | Phase 17 A4 | only 90-day labels exist | `phase-17.md` §1.4 |
| smarter graph encoders (trajectory attention, PNA) | Phase 16 | ties; one disjoint loss | `phase-16.md` |
| global pulse (network-wide trailing statistics) | Phase 18 Stage 5 | worse on arrival AUC, fill CRPS, capacity top precision | `phase18/stage5_pulse.md` |
| lane / checkpoint node | Phase 18 Stage 6 | **blocked**: one lane per channel; `via_checkpoint` 100% empty | `phase18/stage6_lane.md` |
| order-cadence as a timing forecast | Phase 19 Stage 2 | recovers **4%** of the timing headroom (rule needs 30%) | `phase19/stage2_timing.md` |
| learned load forecaster | Phase 20 Stage 3 | 3–4% (arrival), 10–14% (fill) recovery; fails capacity in both worlds | `phase20/stage3_forecaster.md` |
| ledger-based stock arm | Phase 20 Stage 4 | **+0.0016** lateness AUC | `phase20/stage4_reorder.md` |
| trailing-window conformal for capacity | Phase 18 Stage 3d | no tighter per-snapshot coverage than a static offset | `phase18/stage3_squeeze.md` |

---

## 7. Predictions, and what being wrong taught

Each phase wrote its predictions down before measuring.

| phase | right | wrong | the wrong ones |
|---|---|---|---|
| 16 | 3 | 2 | P1 (channel attention is a no-op: it is a learned 3-way selection), P2 (part carries the least message share) |
| 17 | 5 | 1 | P2 (missed rescues have more channels) |
| 18 | 3 | 5 | P2 (fill near its ceiling), P5 (forward load does not help fill), P6 (ensembling + blend tie or beat everywhere), P7 (global pulse ties: it was worse), P8 (P50-centred conformal hits 0.78–0.82) |
| 19 | 4 | 4 | P3 (date-scrambled forward load gives nothing: it helps fill), P4 (timing alone adds < 0.02: it adds 0.205), P6 (neural + forward load beats on capacity recall), P8 (the blend helps on arrival only: also on fill) |
| 20 | 2 | 6 | P1 (arrival blend wins at 5% coverage), P3 (every pass replicates), P4 (gains within ×2), P6 (forecaster passes capacity only), P7 (exact reorder state recovers ≥ 50%), P8 (no observed stock source exists) |

Sources: each phase's report §"prediction scorecard" (`phase-16.md` §5, `phase-17.md` §3, `reports/phase18.md` §3,
`phase-19.md` §3, `phase-20.md` §2).

**Lessons the wrong predictions taught:**
1. **A gate that cannot fail is not a gate.** Every check in the project now carries a constructed input that makes it fire
   (the falsification half). Phase 20's own isolation scanner is tested on planted violations.
2. **Check what a control preserves.** Phase 18's shuffle was meant to destroy the information and kept the network-wide
   season, so a real signal was recorded as FAIL. The control, not the feature, was wrong; Phase 19 re-tested under a new
   pre-registration and never re-labelled the old verdict.
3. **A band must use all five seeds.** Phase 19 found the scorer quietly building "five-seed" bands from one seed when the
   others could not reach an operating point. It flipped one verdict (Phase 19 deviation 171).
4. **Row bootstraps are optimistic.** Rows in one snapshot share its shocks. Resampling whole snapshots widened 48 of 53
   intervals (median 1.9×) and overturned several fill comparisons (`phase19/stage4_intervals.md`).

---

## 8. Ship table

This table matches the decision record `docs/decisions/shipped_phase20.md` exactly. **"Declared"** means decided but
**not servable** yet (its LightGBM part was never saved as a model); the **stand-in** is what is actually served meanwhile.

| use case | **ships now** (served) | declared, not servable | watchlist | blocked on data | closed |
|---|---|---|---|---|---|
| **Arrival** | ranked late-list (UC1): incumbent neural ensemble P(late) · interval: expected-week conformal (~80% at ~45 d) · point estimate: **stand-in** Phase 19 neural ensemble (12.90 d) | point estimate: Phase 19 blend (12.34 d) | the late-list itself (lift ≤ 1.36) | when the order is raised (+0.205 AUC in principle); channel stock + reorder points worth at most +0.034 | encoder variants; load forecaster; ledger stock arm; cadence timing |
| **Fill** | materially-short list: **stand-in** Phase 19 neural season + cadence ensemble | materially-short list: Phase 19 blend; `b5flat22` (superseded) | UC2b list; P(full) as a displayed probability (UC2) | the supplier's weekly queue | five-band head; supplier-specific forward load for fill; cadence for fill (one world only) |
| **Capacity** | UC3 alert: incumbent neural ensemble (81% / 19%, ~10% flagged; Dec 2025 at 0.46) | — | supplier-specific forward load (replicates in the proxy; ties in the neural head) | the monthly capacity draw K | Δ-strain; horizon stacking; forecaster for capacity |
| **Predict-the-rescue** | — | LightGBM B1a (0.823 at recall 0.246) | neural B1b (2 of 5 seeds) | — | — |
| **Shortage head** | — | — | — | — | **retired** (0.224 at a 3% base) |
| **Simulation** | as a pre-rescue reference only | — | — | — | channel-granularity rebuild |
| **MILP / allocation** | — | — | — | costs (MILP); qualification rule (allocation) | — |
| **Transfer** | tie-break suggestion only | — | — | — | as an alert |

---

## 9. What we need from the client, and what each is worth

Full request: `docs/client/data_request_v2.md`.

| ask | why | measured value in the synthetic world | the honest limit |
|---|---|---|---|
| dated plan / forecast versions, each with the date it was recorded | the forward-plan features need the plan **as it stood** on each past date | supplier-specific forward load: arrival AUC +0.013, capacity precision @ 5% +0.12 (proxy); replicates on a second world | effect sizes moved ×0.3–×2.6 between two worlds of one generator; the value must be measured on Rane's data |
| channel-level stock position and as-of reorder points, timestamped | arrival's headroom is order timing, and the reorder trigger is the timing rule | **at most +0.034** lateness AUC (0.705 → 0.739), 17% of the timing headroom | an upper bound (the simulator's position is exact and current); the other 83% is the future demand path |
| holding, ordering, freight and shortage costs | the MILP, allocation and delivery scheduling have no objective without them | not measurable without them | — |
| supplier-declared capacity history and any observed capacity evidence | capacity's headroom is the true monthly capacity | true K is worth +0.09 precision @ 5% in principle | no source in v8 tracks K; real data may or may not |
| PO line creation history and reorder rules | arrival's headroom is when orders are raised | the creation week alone is worth +0.205 AUC in principle | that is an oracle figure; how much a reorder rule recovers is unmeasured |

---

## 10. Honest scope and the standing caveat

Twenty phases in, ChainPilot is:
- **one alert**: supplier capacity strain;
- **one new GO**: predict-the-rescue;
- **two improved forecasts with a ship path**: arrival's point estimate and fill's short-list, both driven by the
  client's forward plan;
- **ranked watchlists** for lateness and fill;
- **two use cases blocked on the client** (costs; the allocation rule) and **three on data that may not exist** (order
  timing, the supplier's queue, true capacity).

**The standing caveat.** Every figure is measured **within one synthetic world**: one generator produced the training and
test data, separated only by time. Phase 20's second world comes from the **same generator**, so it tests stability, not
reality. **Effect sizes did not transfer** even between those two worlds (×0.3 to ×2.6). What transfers is the
**mechanisms**: censoring must be modelled; forecasts must be as-of; the forward plan carries information the past does
not; controls must be checked for what they preserve; every band needs all its seeds. **Every absolute figure here must be
re-measured on Rane's extract** before it is quoted to anyone outside the project.

---

## Appendix A — Numbers register

Every number quoted above. "RAW" unless marked. Split = v8 seed 1001 fixed split, test year, unless marked.

| metric | value | band | seeds | RAW / RECAL | split / world | source |
|---|---|---|---|---|---|---|
| base rate late vs contract | 0.730 | — | — | — | test | `phase-15.md` §1 |
| base rate fill = 1 | 0.749 | — | — | — | test | `phase-15.md` §1 |
| base rate strain > 1 | 0.405 | — | — | — | test | `phase-15.md` §1 |
| base rate fill < 0.95 | 0.245 | — | — | — | test | `phase-15.md` §1 |
| base rate rescue week | 0.4535 | — | — | — | test | `phase-17.md` §2.1 |
| arrival lateness AUC, incumbent h⁴ | 0.70905 | [0.70645, 0.71298] | 5 | RAW | test | `phase-17.md` §2.4 (v2.0 same) |
| arrival C-index, incumbent h⁴ | 0.67442 | [0.67344, 0.67551] | 5 | RAW | test | `phase-17.md` §2.4 (v2.0 quoted 0.67438; later used) |
| arrival lean encoder C-index / lateness | 0.67429 / 0.70747 | overlapping | 5 | RAW | test | `phase-17.md` §2.4 |
| lean encoder parameters | 369,164 vs 533,004 (−30.7%) | — | — | — | — | `phase-17.md` §2.4 |
| arrival A3 median abs error, incumbent | 13.06 d | [12.86, 13.30] | 5 | RAW | test, uncensored | `phase-17.md` §1.3 |
| arrival A3, channel-median constant | 14.00 d | — | — | — | test | `phase-17.md` §1.3 |
| arrival A3, Phase 19 blend | 12.34 d | block CI vs incumbent ensemble −0.74 d [0.54, 0.97] | ensemble | RAW | test | `phase19/stage4_intervals.md` |
| arrival lateness AUC, Phase 19 blend | 0.731 | +0.017 [0.015, 0.020] vs incumbent ensemble | ensemble | RAW | test | `phase19/stage4_intervals.md` |
| arrival P10–P90 coverage raw | 0.93 (0.921–0.943) | — | 5 | RAW | test, uncensored | `phase18/stage3_squeeze.md` |
| arrival expected-week conformal coverage / width | 0.80 (0.791–0.806) / 45 d | block CI ≈ ±0.012 | 5 | RECAL | test, uncensored | `phase19/stage4_intervals.md` |
| UC1 precision @ 5%, incumbent ensemble / blend | 0.969 / 0.943 | diff +0.027 [0.018, 0.036] | ensemble | RAW | test | `phase20/stage1_decisions.md` |
| fill P(full) AUC, incumbent | 0.6204 | [0.6200, 0.6211] | 5 | RAW | test | `phase-17.md` §2.3 |
| fill P(full) AUC, neural + season + cadence | 0.6431 | [0.6424, 0.6440] | 5 | RAW | test | `phase19/stage3_neural.md` |
| fill exact CRPS: incumbent / Phase 19 neural / Phase 19 blend | 0.13876 / 0.1371 / 0.1354 | [0.13866, 0.13882] / [0.1366, 0.1374] / block CI | 5 / 5 / ensemble | RAW | test | `phase-17.md`, `phase19/stage3_neural.md`, `phase19/stage4_intervals.md` |
| UC2b precision @ 5%: incumbent ensemble / Phase 19 blend / b5flat22 | 0.464 / 0.530 / 0.384 | blend vs incumbent +0.063 [0.043, 0.090] | ens / ens / 5 | RAW | test | `phase20/stage1_decisions.md` |
| capacity precision @ 1 / 5 / 10% | 0.904 / 0.851 / 0.810 | 5-seed | 5 | RAW | test | `phase-15.md` §1, `phase-17.md` §2.2 |
| capacity recall @ p 0.70 / 0.80 / 0.85 | 0.515 / 0.318 / 0.191 | 5-seed | 5 | RAW | test | same |
| capacity per-snapshot P @ 5%, min / median | 0.46 / 0.81 | — | ensemble | RAW | test | `phase20/stage1_decisions.md` |
| capacity Phase 19 neural + fwd_load P @ 5% / R @ 0.70 | 0.886 / 0.506 | [0.829, 0.908] / [0.380, 0.545] | 5 | RAW | test | `phase19/stage3_neural.md` |
| shortage head P @ 5% at a 3% base | 0.224 | [0.223, 0.226] | 5 | RAW (re-weighted) | test | `phase-17.md` §1.1 |
| rescue B1a LightGBM precision @ recall 0.246 | 0.823 | [0.822, 0.824] | 5 | RAW | test | `phase-17.md` §2.1 |
| rescue B1b neural | 0.859 | [0.858, 0.859] | **2** | RAW | test | `phase-17.md` §2.1 |
| rescue proxy / own history | 0.619 / 0.631 | — | 5 / 1 | RAW | test | `phase-17.md` §2.1 |
| simulation vs pre-rescue | 1.129× | [1.121, 1.139] | 5 | — | test | `phase-13.md`, `observation1.md` §6.2 |
| oracle arrival / fill / capacity | 0.9385 / 0.9184 / 1.000 | 5-seed | 5 | RAW | test | `phase18/oracle/PRIVILEGED__stage2_oracle.md` |
| oracle state-only fill / capacity | 0.7506 / 0.944 | 5-seed | 5 | RAW | test | same |
| arrival timing-only (creation week) | 0.9104 | [0.9102, 0.9105] | 5 | RAW | test | `phase19/stage2_timing.md` |
| arrival exact reorder probe | 0.7394 | [0.7390, 0.7397] | 5 | RAW | test | `phase20/stage4_reorder.md` |
| ledger stock arm | 0.7069 vs 0.7053 | [0.7065, 0.7072] | 5 | RAW | test | `phase20/stage4_reorder.md` |
| K month-to-month correlation; declared vs true | 0.094; 0.110 | — | — | — | generator | `phase18/stage1_generator.md` |
| forward-load gain, arrival (v8 / world 2) | +0.0133 / +0.0131 | disjoint in both | 5 | RAW | test | `phase20/stage2_replication.md` |
| replication effect-size range | ×0.3 – ×2.6 | — | — | — | two worlds | `phase20/stage2_replication.md` |
| load forecaster recovery, arrival / fill | 3–4% / 10–14% | — | 5 | RAW | test, both worlds | `phase20/stage3_forecaster.md` |
| block vs row bootstrap widening | 48 of 53, median 1.9× | — | — | — | test | `phase19/stage4_intervals.md` |

## Appendix B — Errata: what in `observation1.md` (v2.0) was later corrected or superseded

| v2.0 said | now | corrected by |
|---|---|---|
| shortage head: "internal watchlist only … re-measure at a realistic base rate before investing" | re-measured: 0.224 at a 3% base → **retired** | `phase-17.md` §1.1 |
| simulation: "restate at channel granularity: the only remaining modelling item with a case behind it" | **closed**: misses are not channel-driven (P = 0.51); they are late-horizon drawdowns from healthy stock | `phase-17.md` §1.2 |
| arrival: "nothing; it is at its ceiling" | **not at its ceiling**: oracle headroom +0.23, almost all order timing, mostly unreachable | `reports/phase18.md` §2, `phase-19.md` §2 |
| fill serving: "do not swap the model; b5flat22 is worse on CRPS and AUC" | still true, and stronger: on decisions, b5flat22 is the weakest fill arm; the Phase 19 blend is better on CRPS, AUC **and** calibration than the incumbent ensemble | `phase20/stage1_decisions.md`, `phase19/stage4_intervals.md` |
| "a leaner encoder … is a legitimate simplification to test" | tested: **tie, −30.7% parameters** | `phase-17.md` §2.4 |
| capacity: "the one shippable alert" | stands; adds a per-snapshot caveat (Dec 2025 at 0.46 at 5% coverage) | `phase20/stage1_decisions.md` |
| arrival C-index 0.67438 | 0.67442 (Phase 17's recomputation; the later figure is used) | `phase-17.md` §2.4 |
| "Phase 15 ceiling for materially short is 0.577" | the Phase 19 blend reaches 0.644 at 1% coverage; still NO, CEILING at the bars | `phase20/stage1_decisions.md` |
| fill: "head-shape programme closed" | stands; the five-band head (Phase 17) also failed | `phase-17.md` §2.3 |
| §10.2 plan items 3 (channel granularity) and 6 (re-measure shortage) | both done and closed | `phase-17.md` |

## Appendix C — Quotable claims (each survives a five-seed band or the snapshot-block bootstrap)

1. *"For supplier capacity, we alert on about one horizon in ten and are right about 8 times in 10, against a background of
   4 in 10."* (5-seed band; `phase-15.md`). Label: one quarter ran at 46%.
2. *"Our predicted delivery date is off by a median of 12.3 days, against 13.1 for the previous model."* (block bootstrap;
   `phase19/stage4_intervals.md`)
3. *"Our 80% delivery-date interval contains the actual date 80% of the time, at about 45 days wide."* (5 seeds, block CI;
   `phase19/stage4_intervals.md`)
4. *"Of the top 5% of order lines we flag as materially short, 53% are, against a background of 25%."* (block bootstrap;
   `phase20/stage1_decisions.md`)
5. *"We predict which weeks your planners will need to move stock with 82% precision, catching a quarter of them."*
   (5-seed band; `phase-17.md` §2.1)
6. *"The simulation reproduces what would happen without your planners' intervention to within about 13%."* (5-seed band,
   falsified three ways; `phase-13.md`)
7. *"Adding your forward plan improves delivery-lateness ranking, and the improvement held on a second simulated world."*
   (5-seed bands in both; `phase20/stage2_replication.md`). Label: same generator; effect size must be re-measured.

## Appendix D — Open deviations

| # | what | status | source |
|---|---|---|---|
| **122** | Phase 12 B3's block regrouping silently changed the default simulation's random stream; today's code does not reproduce Phase 12 B2 | **open.** Worked around: Phase 14 reloads `order_policy.py` pinned at `9e2d59d` (gated 45/45 exact), and Phase 15 stored the regenerated per-row paths. No later report restores the default stream | `phase-14.md` deviations table (122), `phase-15.md` header |
| 46 | `shipped.json` names a LightGBM the serving path cannot load | **addressed in the integration, not removed**: no LightGBM model was ever saved, so `b5flat22` and every LightGBM-containing item are now recorded *declared, not servable*, and `ml/serve/` refuses (identity guard) rather than substituting. Serving a LightGBM item needs a phase allowed to fit and save one | `docs/decisions/shipped_phase20.md`, `reports/part2/integration-report.md` |
| 149 | 31 Phase 16 bundles not nameable by the pre-Phase-16 identity code | **closed in the integration**: all 371 stored configs (including the 31) recompute identically on the merged code | `reports/part2/integration-report.md` |
| 167 | Phase 18 intervals used a row bootstrap (optimistic) | **closed**: redone with a snapshot-block bootstrap in Phase 19 | `phase19/stage4_intervals.md` |
| 171 | scorer built bands from fewer than five seeds | **closed** for Phases 19–20 scorers; Phase 18's verdicts unaffected | `phase19/stage1_gate.md` |
| 180 | the "fresh" second world equals the stored seed-1002 world | **stands** as a scope limit (same generator) | `phase-20.md` §3 |
| 119 | `inventory_transactions.from_plant_id` records the receiving plant; the donor is never stored | **open** (generator defect) | `observation1.md` §9.2 |
