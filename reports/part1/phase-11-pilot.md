# Phase 11 pilot — config smoke test, arrival feature pilot, allocation sensitivity

## 1. Verdict

| stage | verdict |
|---|---|
| **1 — config smoke test** | **FAILS for fill in both worlds.** Arrival's provenance is correct; the fill switch is **declarative only** — `shipped.json` names `b5flat22`, the serving path cannot load it, and `loop.predict` silently serves the superseded neural head. Reported, not repaired, per 1.3. |
| **2 — arrival feature pilot** | **BLOCKED, and not by budget.** `promise_week` and line age **do not exist at prediction time**: every arrival label row describes a `po_line` created **7 to 89 days after its own snapshot** (median 49), in 244,000 of 244,000 rows in both worlds. Supplying them would feed the head the future. No cell was trained. |
| **3 — allocation sensitivity** | **Run, and the answer is negative.** Across 120 part-plants only **30 (25%)** have a recommendation that is both stable across the shortage-cost sweep and survives the seed bands — and a third of those say "keep the incumbent". **10.2 produces no quotable recommendation on this data.** |

**The Stage 2 blocker is the most consequential finding in this pilot, and it corrects an earlier claim.** Phase 7 §7 described the promise-date baseline as *"Training data: none. Deployable: yes — a planner already holds it."* **On this data a planner does not hold it.** The promise date belongs to a purchase-order line that will not be raised for another seven weeks on average. Its C-index of 0.87–0.89 is achieved with information from the future relative to the prediction instant.

That changes the reading of the arrival comparison in three ways:

1. **Phase 9B's "ranking is untestable" was right, and the reason is stronger than stated.** It is not merely that the head is never given the line-level scalar — the scalar **cannot legitimately be given to it**, because it does not exist at t0. The gap is unclosable by any as-of model, not just by this one.
2. **The head's lateness win is better than reported, not worse.** The head reads only as-of channel data; the promise baseline it beats by +0.001 to +0.031 ROC-AUC is future-informed. Beating a privileged baseline is a stronger result than beating a fair one.
3. **Guide step 11.2 — which Phase 10 called the project's highest-expected-value model change — is withdrawn as specified.** Its premise was that giving the head `promise_week` and line age would close the ranking gap. Those inputs are not available at prediction time, so the experiment cannot be run honestly. §5 records the correction.

`inventory_position_weekly` was never read.

---

## 2. Stage 1 — provenance of the served distribution

`ml/eval/phase11_smoke.py`. The test is provenance, not execution: does the served distribution come from the source the configuration names?

### 2.1 arrival_week — PASS in both worlds

| check | v6 | v7 |
|---|---|---|
| configured | `serve_distribution: h0_always`, `fallback: null` | same |
| `serve_source()` returns | `h0` | `h0` |
| `distribution_source` in the output | `h0 (always, gate removed)` | same |
| served distribution **byte-for-byte equals h⁰'s own** | **True** | **True** |
| ranking still produced by h⁴ | True | True |
| drift / h⁰ drift / excess present | −0.5831 / +0.3498 / **0.9329** pp | −2.1115 / +0.2724 / **2.3838** pp |
| every batch carries drift and excess | True | True |
| `serve_source` can see a drift value | **False** (asserted on its signature) | **False** |

**v7's excess is 2.38 pp — well over the removed gate's 1.0 pp threshold.** Under the old configuration the fallback would have engaged; now the number is computed, logged and ignored. That is the clearest possible demonstration that the switch is gone while monitoring survives. v6's 0.9329 pp reproduces Phase 8's recorded 0.933 pp.

### 2.2 fill_rate — MISMATCH in both worlds

| check | result |
|---|---|
| configured model | **`b5flat22`**, family `lightgbm`, `identity_keys: false` |
| bundle the serving path loaded | `{world}_none_h0_lr0.000125_s7` → `arch=none, depth=0, lr=0.000125` — **the superseded neural head** |
| a `b5flat22` bundle exists | **False** |
| `loop` has any code path that can materialise a LightGBM | **False** |
| drift still computed | yes (−0.6884 pp v6, −0.8856 pp v7, 9 batches, 36,000 rows) |

**The Phase 10 fill switch is recorded in configuration and honoured by nothing in the serving path.** `loop.predict` materialises neural bundles only; there is no loader for a LightGBM, no bundle to load, and nothing asserted the gap — so the call returns the old model's distribution while the config claims the new one. Anyone reading `shipped.json` today would believe fill ships B5-flat-22; anyone calling the serving path would receive the depth-0 head.

Per instruction 1.3 I stopped and did **not** repair the configuration. The fix is a decision, not a patch: either the serving path gains a LightGBM loader and a `b5flat22` bundle is built, or the switch is reverted until it can be. Recorded as deviation 46.

This does not invalidate Stage 3 — `ReducedScorer` reads the `b5flat22` prediction files directly rather than through `loop.predict`, so the allocation scorer does consume the configured model.

---

## 3. Stage 2 — the arrival feature pilot is blocked by as-of

### 3.1 What the assertion caught

The brief's own precondition was that *"line age must be computable at t0 from recorded data"*. It is not. The per-batch as-of assertion fired on the first training batch of the first cell, before any bundle was written:

```
AssertionError: a po_line is not yet recorded at its own snapshot -- as-of violation
```

### 3.2 The measurement

| | v6 | v7 |
|---|---|---|
| arrival label rows | 244,000 | 244,000 |
| rows whose `po_line` is recorded **after** its own snapshot | **244,000 (100.00%)** | **244,000 (100.00%)** |
| rows whose line is created **before** its snapshot | **0** | **0** |
| `created_ts − snapshot_date`, days | min 7, p25 26, **median 49**, p75 70, max 89 | same shape |
| `recorded_ts − created_ts`, days | median 0 | median 0 |
| origin-7 train / validation / evaluation folds affected | 100% / 100% / 100% | 100% / 100% / 100% |

A sample row makes it concrete: snapshot `2019-01-14`, line created `2019-04-04`, promise date `2019-06-10`, label 13 weeks. At the snapshot the line does not exist, has no promise date and has no age.

### 3.3 Why this is not a bug to be worked around

The arrival task is **not** "when will this open order arrive". It is a channel-level forecast: *for the orders that will be raised on this channel over the next 90 days, when will they arrive.* That is why the shipped head is channel-level by construction, and why it emits one distribution per channel-snapshot — the 25,569 bit-identical rows Phase 9B measured are a consequence of the task's shape, not a deficiency.

Three responses were possible. Two are wrong:

- **Drop the assertion.** It would train on the future and every subsequent number would be leakage.
- **Drop the offending rows.** There are none left — it is 100%, in both worlds, in every fold.
- **Report it.** Taken.

### 3.4 Verdict (2.4), stated plainly

**Neither.** The head does not out-rank the promise date, and it cannot be made to by this route — the information the pilot was designed to give it is not available at prediction time.

**What that implies:** line age was not "the missing information" in any usable sense. It is genuinely predictive — the generator builds both promise and arrival from `po_created` — but it belongs to an event that has not happened when the forecast is made. **Arrival ranking against the promise date is not a fair comparison on this dataset in either direction**, and no feature change to the head can make it one.

Per 2.5 no `shipped.json` change is proposed. `shipped.json` is untouched by this stage.

### 3.5 What was built, and what state it is in

The feature path is implemented and committed, default **off**, and every existing configuration is bit-identically unchanged: a `row_features` cell carries a `_rowfeat` suffix through `ml/artifact_identity.py`, so it can never share a bundle, prediction file or index key with the channel-only cell (standing rule 2). It is retained because the as-of assertion it carries is the check that caught this, and because the same machinery would be correct on a real extract where orders are raised before the forecast is made. **No cell was trained; no bundle exists.**

---

## 4. Stage 3 — the shortage cost decides the recommendation

`ml/opt/sensitivity.py`. Shortage cost swept 100 / 300 / 1,000 / 3,000 / 10,000 Rs per unit across 60 part-plants per world, 3 seeds, the same fill and capacity signals Phase 10 used. No training, no inference.

### 4.1 The Phase 10 §5.4 rows, swept

Winner at each cost, left to right: 100 / 300 / 1,000 / 3,000 / 10,000.

| world | part-plant | req | qualified | winner across the sweep | stable? | survives bands at every cost? | quotable? |
|---|---|---|---|---|---|---|---|
| v6 | P00001 @ PL01 | 124 | 5 | incumbent | incumbent | incumbent | incumbent | shift 20pp | **NO** | yes | **no** |
| v6 | P00001 @ PL02 | 322 | 4 | cap at 70% | cap at 70% | cap at 70% | cap at 70% | cap at 70% | **yes** | yes | **yes** |
| v6 | P00001 @ PL03 | 82 | 2 | incumbent | incumbent | incumbent | incumbent | incumbent | **yes** | **no** | **no** |
| v6 | P00001 @ PL04 | 247 | 3 | incumbent | incumbent | incumbent | incumbent | incumbent | **yes** | **no** | **no** |
| v6 | P00001 @ PL05 | 241 | 3 | incumbent | incumbent | incumbent | incumbent | incumbent | **yes** | **no** | **no** |
| v6 | P00001 @ PL06 | 195 | 5 | cap at 70% | cap at 70% | incumbent | incumbent | incumbent | **NO** | yes | **no** |
| v7 | P00001 @ PL01 | 132 | 5 | incumbent | incumbent | incumbent | cap at 70% | cap at 70% | **NO** | **no** | **no** |
| v7 | P00001 @ PL02 | 249 | 4 | incumbent | incumbent | incumbent | incumbent | incumbent | **yes** | **no** | **no** |
| v7 | P00001 @ PL03 | 82 | 2 | incumbent | incumbent | cap at 70% | cap at 70% | cap at 70% | **NO** | **no** | **no** |
| v7 | P00001 @ PL04 | 217 | 3 | incumbent | incumbent | incumbent | incumbent | incumbent | **yes** | **no** | **no** |
| v7 | P00001 @ PL05 | 206 | 3 | incumbent | incumbent | incumbent | cap at 70% | cap at 70% | **NO** | **no** | **no** |
| v7 | P00001 @ PL06 | 188 | 5 | shift 20pp | shift 20pp | incumbent | incumbent | incumbent | **NO** | yes | **no** |

### 4.2 The whole sample

| | v6 | v7 | combined |
|---|---|---|---|
| part-plants evaluated | 60 | 60 | 120 |
| winner **stable** across the sweep | 50 (83%) | 47 (78%) | 97 (81%) |
| margin **survives the seed bands** at every cost | 23 (38%) | 17 (28%) | 40 (33%) |
| **BOTH — a quotable recommendation** | **18 (30%)** | **12 (20%)** | **30 (25%)** |
| of those, "keep the incumbent" (no action) | 6 of 18 | 3 of 12 | 9 of 30 |

**Where the winners flip.** 10 of 60 (v6) and 13 of 60 (v7) change winner across the sweep, and the flips cluster exactly where the assumption sits: v7 flips 4 times between 300 and 1,000 and 7 times between 1,000 and 3,000; v6 flips 6 times between 3,000 and 10,000. **The assumed Rs 1,000 is not a neutral placeholder — it is inside the band where the answer changes.**

### 4.3 Verdict (3.3), unsoftened

**Only 30 of 120 part-plants (25%) carry a recommendation that is stable against the shortage-cost assumption and distinguishable from the runner-up given the seed bands. Of those, 9 recommend changing nothing. So roughly 18% of part-plants yield an actionable, defensible allocation recommendation.**

**10.2 produces no quotable recommendation on this data, and no allocation output should be shown to a client until the shortage cost is supplied by them.** It is not a tuning constant: it sets the exchange rate between unmet demand and purchase price, which is the entire trade-off the allocator exists to make. Choosing it ourselves means choosing the answer.

**This must be read alongside Phase 10 §6.** That list is going to a client, and nothing in it should imply the allocator is ready. It is not: its objective is a reduced proxy for the guide's (no inventory state), its capacity term inherits intervals that are not quotable, its qualification constraint is inert, its capacity ceiling is absent, and now its ranking is shown to turn on a number nobody has supplied. **Item 6 of §6.2 — shortage/stockout cost — moves from "currently an assumption" to blocking.**

---

## 5. Correction to guide step 11.2

Phase 10 Stage 4 wrote step 11.2 (*"Arrival: give the head `promise_week` and line age"*) and called it the project's highest-expected-value model change, estimated at 6–7 h. **That estimate is withdrawn.** Its premise — that these are inputs a model could legitimately hold — is false on this dataset, where the line is raised on average seven weeks after the forecast.

The step is not deleted, because it may be correct on a real extract: at Rane, a planner forecasting an **open** order does hold its promise date and its age. The guide now records the precondition — the pilot may only run where labels attach to lines already raised at t0 — and the measurement that fails it here. Recorded as deviation 45.

---

## 6. Deviations

Continuing the guide's index from row 44.

| # | the guide / specification says | measured, and what Phase 11 did |
|---|---|---|
| 44 | arrival labels describe purchase-order lines observable at the snapshot | **they do not.** In 244,000 of 244,000 rows in both worlds the line is created **7–89 days after** its own snapshot (median 49); `recorded_ts` equals `created_ts` to the day. The task is a channel-level forecast of orders **not yet raised**, which is why the head is channel-level and emits one distribution per channel-snapshot |
| 45 | guide 11.2 — giving the head `promise_week` and line age is the highest-value model change | **withdrawn as specified**: neither input exists at prediction time on this data, so the pilot cannot run without leakage. No cell trained. The step is kept with a precondition for a real extract |
| 46 | `shipped.json` names the model that is served | **it does not.** The Phase 10 fill switch to `b5flat22` is declarative only: no such bundle exists, `loop` cannot materialise a LightGBM, and `loop.predict` silently serves the superseded depth-0 head. Reported, not repaired |
| 47 | 10.2's candidate ranking is a recommendation | **only 25% of part-plants** survive both a shortage-cost sweep over two orders of magnitude and the 3-seed bands, and 9 of those 30 say "keep the incumbent". The assumed Rs 1,000/unit sits inside the band where winners flip |
| 48 | Phase 7 §7 — the promise-date baseline is deployable, "a planner already holds it" | **false on this data**: the promise date belongs to a line raised ~7 weeks after the forecast instant. Its 0.87–0.89 C-index uses future information, so the head's lateness margin is measured against a **privileged** baseline |

---

## 7. Open items

**Carried forward from Phase 10 §8**, unchanged except where noted: 10.1 is single-period until a forward plan exists; 10.2's constraint layer is half enforceable; `ReducedScorer`'s reduced objective; capacity intervals not quotable; Phase 9 blocked; fill's marginal calibration versus the proper score.

**Resolved or changed:**

1. **Phase 10 open item 1 (the config change unexercised) is now closed, with a failure.** Arrival passes; fill does not. See §2.2 — it needs a decision.
2. **Phase 10 open item 4 (shortage-cost sensitivity) is now measured** and the answer is negative (§4.3). It escalates to a blocking client ask.

**New:**

3. **The fill switch must be made real or reverted** (deviation 46). Two options, both decisions for you: build a `b5flat22` serving artifact and teach `loop.predict` to load it, or revert `shipped.json` to the neural head until that exists. I did neither.
4. **A provenance assertion belongs in the serving path itself**, not only in a smoke test: `loop.predict` should refuse to serve when the loaded bundle does not match the configured model. Today it serves the wrong model silently. This is the same class as deviation 28 — an identity that nothing asserted at the point of use.
5. **Arrival's task definition should be stated explicitly wherever the head is described.** It forecasts orders not yet raised. Every prior report compares it against a baseline that implicitly assumes the opposite.
6. **The 10.2 ranking needs a client-supplied shortage cost before any output is shown** (§4.3), and Phase 10 §6.2 item 6 should be re-labelled blocking before that list is sent.
7. **Guide 11.1 (capacity intervals, ~3 h) is untouched** and remains the one deferred model change whose premise still holds.

**`inventory_position_weekly` was never read in Phase 11** — the smoke test, the sensitivity sweep and both optimiser modules refuse the filename at runtime, and no file under `ml/` references it.
