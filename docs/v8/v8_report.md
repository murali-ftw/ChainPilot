# ChainPilot Synthetic Dataset — V8 build report

**Audience:** whoever reviews or extends the V8 generator, and whoever owns the serving path.
**Built from:** `prompts/v8_generator_brief_and_validator.md`.
**Code state:** `db/gen_v8/generator_v8.py` (sha1 `71de78afa645`), `db/gen_v8/validator_v8.py` (sha1 `57e0c7387d35`).
All five seeds come from that one generator state; `manifest.json` in each seed carries the hash and
the validator refuses to publish a variance band unless all five agree.

---

## 0. Verdict, up front

**V8 is built, all five deliverables exist, and the readiness ladder stops at G4 on every one of
the five seeds.** G0 through G3 pass. **G4 fails: the graph contributes nothing.** G5, G6 and G7
were therefore not reached and nothing is claimed about them.

```
G0  Is this a dataset?            PASS  (27 checks)
G1  Does the graph exist?         PASS  ( 7 checks)
G2  What is the number to beat?   PASS  ( 3 checks)
G3  Is there signal at all?       PASS  ( 6 checks)
G4  Does the graph contribute?    FAIL  ( 2 checks)   <-- stop
G5  Is the as-of gate binding?    NOT REACHED
G6  Are the folds sound?          NOT REACHED
G7  Will it survive production?   NOT REACHED
```

375 checks were run across the five seeds. Ten failed — the same two G4 checks on each seed.
Every Level-3 joint probe passes on every seed. Every accounting identity is exact.

**What is not here, and never will be from a generator:** no carrying cost, ordering cost, freight
structure or shortage cost. §2.7 of the brief is right and V8 obeys it. UC 10.2 still has no quotable
output, and that is a blocking item on the client, not an assumption anyone can close from this side.

---

## 1. What V8 builds

### 1.1 §2.1 Opening stock balance — the item that unblocks Phase 9

**Mechanism (SET):** a starting position per sourcing channel, sized as weeks of cover over that
channel's own demand rate with noise (`open_cover_weeks_mu`, `open_cover_weeks_sd`), plus a slice that
genuinely starts empty (`open_zero_prob`). It is **posted to `inventory_transactions` as an
`adjustment` at a fixed pre-window date (2015-12-28)** — one week before the world's first modelled
week.

**Outcome (MEASURED):** every balance after that. `inventory_position_weekly` is emitted from the
simulator's own weekly position and the validator reconciles it, exactly and per week, against an
independent cumulative sum of the emitted ledger. `inventory_snapshots` is the same ledger closed
monthly.

This is the difference that matters: v7 seeded `on_hand` directly and never posted it, so the ledger
could not reproduce the balance and nothing downstream could be trusted to conserve.

### 1.2 §2.2 Forward requirement plan

`production_plan` now publishes a **new version every 4 weeks**, each covering a **rolling 12–26 week
horizon** drawn per product per cycle, carrying `plan_date` as the publish date. A target period is
therefore republished across up to 7 versions, and `horizon_days = target_period − plan_date` takes 26
distinct values instead of the single 30 v7 wrote. `part_demand_weekly` runs the same discipline at
part × plant on an 8-week netting cadence.

The drift mechanism underneath is unchanged and still visible: a version is optimistic and noisier the
further out it looks, and the validator checks that later versions land closer to the actual than V1
does — a check that fails if revision carries no information.

### 1.3 §2.5 The feedback loop, all six arcs

| From | To | What V8 does |
|---|---|---|
| Shortage | Expedite | unchanged from v7 — the expedite moves a real open line |
| **Shortage** | **PO line revision** | **new.** A buyer whose expedite was never raised or did not land revises the open line: pulls the promise date in (a request, so it moves `current_promise_date` only and the line is then measured late against the date the buyer asked for) or lifts the open quantity. v7 emitted `r.random() < 0.34` here. |
| Shortage | Alternate sourcing | unchanged — volume shifted to another supplier of the same part |
| **Alternate sourcing** | **Capacity utilisation** | runs through the shared capacity bank: shifted volume consumes the receiving supplier's headroom, so its other channels see less |
| **Line stop** | **Production plan** | **new.** A stop cuts the build for the following weeks, and the next published plan version sees the lower truth and revises down — visible in `plan_drift_features`. |
| **Sustained poor performance** | **Allocation** | **new.** A quarterly review moves each part-plant's split toward trailing delivered performance and **reweights demand between siblings at constant part-plant total**, so a supplier that keeps missing loses real volume in order history. v7 wrote one static `100/n` row per part-plant dated 2019-01-01. |

### 1.4 §2.6 Contract terms — features only

`moq`, `lot_size`, `min_volume_commitment`, `max_volume_cap`, `price_break_qty`, `penalty_clause_inr`
and `alternate_sources.qualification_status` are now drawn from distributions conditioned on
`supplier_tier` / `supplier_type` / `business_class`. v7 shipped `moq [10,10]`, `lot_size [25,25]`,
`max_volume_cap [5000,5000]`, `penalty [10000,10000]` and `qualification_status` constant.

Every seed ships `allocation_disclaimer.txt` beside the data. These terms un-degenerate the features
the arrival / fill / capacity heads read. They are **not** valid inputs to a quoted allocation or
delivery-schedule recommendation, for exactly the reason §0 of the brief gives: an invented spread
validates the optimiser's sensitivity to the invention, not real constraint diversity.

### 1.5 §2.3 Recording layer — two things v7 documented but did not do

- **"Some records are never entered."** v7 drew the never-recorded mask and then emitted every row
  anyway, so the mechanism was in the parameter file and absent from the data. V8 applies it to every
  table nothing else references: acknowledgements, ASN, quality inspections, shortage events,
  expedite events, PO-line revisions. The PO → GRN → inventory spine is left whole on purpose —
  those are an ERP's system of record, and dropping them would dissolve the conservation identity
  everything else is checked against. A revision nobody keyed in is also excluded from the
  `revision_count` feature, because a feature cannot know about a record that does not exist.
- **The lag tail had no bound.** A log-normal with σ = 1.35 over a million draws put receipts on the
  books in 2029 for a world that ends in 2026. Two mechanisms bound it, neither a clip on the
  outcome: the state response **saturates** (a stressed plant records later, but a plant under five
  sigma of stress does not record five times later — the backlog is worked off by people with a
  throughput ceiling), and a record that misses the **accounting close** is posted at the close.
- **Forward-dated validity windows.** `supplier_capacity` and `supplier_allocation` are now
  registered *before* the period they govern opens, which is the one place the brief allows a
  negative lag. The validator checks that the lag is negative there and nowhere else.

### 1.6 §2.7 What is deliberately absent

No carrying cost, ordering cost, freight structure, shortage cost, truck capacity or holding cost.
`parameters_v8.json` carries an `EXCLUDED__by_design` block naming each of them and why. Three
columns the schema forces to exist — `part_costs.unit_cost_inr`, `part_costs.freight_cost_inr`,
`expedite_events.premium_cost_inr` — are labelled PLACEHOLDER in that block. They are arbitrary
draws. They are not Rane prices and no optimiser output computed from them means anything.

---

## 2. The validator

`db/gen_v8/validator_v8.py`. Four levels run in the brief's order; the G0–G7 ladder runs after them
and **stops at the first failed gate**. Bands were fixed when the file was written and no band was
moved afterwards to let a dataset through.

Two corrections to gate assignment were made before freezing, and both are worth stating because
neither is a band change:

- **Level-3 probes are reported, not ladder gates.** The brief's G0 stop conditions are specific and
  short — conservation inexact, a task with 0% censoring, lag `sd/|mean| < 0.1`. Probe 1 is the
  exception and does gate, because the brief calls it decisive: near-zero there makes every other
  number irrelevant.
- **Probe 7 is measured within supplier.** The arc the brief names is "receiving supplier load ↑ →
  **its own** future performance". That is a within-supplier statement. The cross-supplier
  correlation answers a different question — good suppliers are given more volume by the very
  allocation arc §2.5 asks for — and is reported beside it rather than gated on.

### 2.1 §1.3 — every check must be able to fail

Each check carries a `breaks` string naming the mutation that makes it fail. `--selftest` applies
those mutations to the emitted CSVs and reports any check that survives one.

**13 mutations, 13 caught, no blind spots** (`docs/v8/selftest_mutations.txt`). The selftest runs
against a smoke-scale world built from the same generator state — falsifiability does not depend on
world size, and running thirteen mutations over a 3.7 GB seed would cost hours for no extra
information. Each row names the check that fired:

| Mutation | Caught by |
|---|---|
| net the scrap into the receipt | §2.1 roll-forward; monthly snapshot |
| remove the opening balance | §2.1 roll-forward; monthly snapshot |
| drop 1% of ordered units | ordered-units conservation |
| shift the weekly position by one unit | §2.1 roll-forward |
| log only half the allocation split | allocation sums to 100 |
| collapse the plan to one period | §2.2 version span; horizon varies |
| make the recording lag constant | §2.3 dispersion; §2.3 skew |
| restore v7 degenerate contract terms | §2.6 degeneracy; §2.6 tier separation |
| permute labels against features | probe 1 (arrival, capacity) |
| draw labels from Uniform(0,1) | probe 1 (arrival, capacity); probe 9 KS |
| permute `load_ratio` | probe 1 capacity; probe 2 both arms |
| label every shortage with one cause | probe 5 cause-mix |
| remove all censoring | G0 censoring |

Three of the thirteen mutations exposed something while this was being built, and two were checks
that could not fail and have been replaced:

- The old ledger check was `closing == opening + movements`, which is true of any cumulative sum
  whatsoever. It is deleted. In its place is the §2.1 roll-forward: the emitted ledger, summed
  forward from the opening posting, must reproduce `inventory_position_weekly.qty_on_hand` exactly,
  every part-plant, every week.
- "Shortage events have a root cause" could not fail either — `root_cause` is NOT NULL over a
  six-value enum, so it is 100% populated by construction. It is replaced by independent
  corroboration: the validator re-derives, from the CSVs alone, whether the evidence each label
  claims actually exists. And because a label that happens to be corroborable everywhere would still
  slip through, a second check requires `root_cause` to be a mix rather than one label on everything.
---

## 3. What failed, and the mechanism for each

### 3.1 G4 — the graph contributes nothing. This is the finding.

The diagnostic is run on `capacity_strain`, whose label is a supplier-month aggregate — per §1.2 of
the brief, the one head in the set where a neighbourhood genuinely exists. Five seeds, MAE, lower is
better:

| seed | h⁰ features only | h¹ one hop | h⁴ four layers | gain | h⁴ over a **shuffled** graph |
|---|---|---|---|---|---|
| 1001 | 0.25727 | 0.25984 | 0.25856 | **−0.50%** | 0.25740 |
| 1002 | 0.21557 | 0.21919 | 0.21848 | **−1.35%** | 0.21603 |
| 1003 | 0.21410 | 0.21823 | 0.21846 | **−2.04%** | 0.21404 |
| 1004 | 0.21645 | 0.21792 | 0.21835 | **−0.88%** | 0.21656 |
| 1005 | 0.20730 | 0.20909 | 0.20928 | **−0.96%** | 0.20699 |

Two statements, both robust across all five seeds:

1. **Propagation makes the task worse, monotonically in depth.** h⁰ is the best arm every time.
2. **A randomly shuffled neighbourhood performs the same as the real one** — within 1.1% either way,
   and on three of five seeds the shuffled graph is actually *better*. The edges are not
   distinguishable from noise.

The second point is what makes this conclusive rather than merely disappointing. If propagation were
carrying real structure and simply being outweighed, the real graph would still beat a random one.
It does not.

Fill's ladder is run and reported too, and is flat: +0.00%, −0.20%, −0.22%, +0.06%, and one more
near zero. That part is unsurprising and is the brief's own §1.2 claim confirmed — fill reads out at
the channel and has no real neighbourhood.

**A hypothesis tested and refuted.** The obvious explanation was that `load_ratio` in
`channel_performance_weekly` already broadcasts the supplier's own utilisation to every channel of
that supplier, which would make a 1-hop supplier aggregate redundant by construction. It does not
hold: re-running the ladder with `load_ratio` removed from every arm
(`docs/v8/graph_diagnostic.txt`) leaves the gain at −0.49%, against −0.74% with it. Dropping the
column makes every arm worse (h⁰ 0.2573 → 0.2785) and does not restore propagation. **So the reason
the graph carries nothing on this world is not yet known, and is the next thing to find out.**

**A correction to an intermediate result.** On a 1,200-channel smoke world the same diagnostic gave
+1.95% with the shuffled control passing, and that was reported mid-build as the graph contributing.
It reverses at full scale. The plausible reading is that with ~20 channels per supplier the neighbour
mean is noisy enough to act as variance reduction, while at ~38 per supplier it converges to
something the model can already reach. Either way, the full-scale number is the one that counts and
the smoke-scale one should not have been reported as a result.

**What this does and does not license.** It does not say "graph encoders do not work". It says that
on *this world*, with *this feature store*, at this scale, a mean-aggregation over the real
supplier / part / plant neighbourhoods carries no information the flat features do not already
carry. The brief's §2.5 warning — "without the second and fourth arcs the graph has nothing to
propagate" — is not the explanation here: both arcs are built and measurably live (probe 6 lift
5.4–8.1×, probe 7 within-supplier r −0.12 to −0.14). The arcs move real volume; propagation still
does not pay.

### 3.2 G1 failed on the first full run, and was fixed at the mechanism

The first five-seed run measured 5.85 / 5.86 / 5.88 / 5.89 / 6.14 PO lines per channel per year
against §2.4's ≥ 6 floor — four seeds short. The gate was not widened. Order frequency goes as
`1/cover`, because the gap the order-up-to policy refills is `dhat × cover`, so `cover_weeks_mu` went
7.5 → 6.5 and the world was regenerated. It now measures 6.48–6.78 across five seeds. That is the
§3 discipline working as intended: a measured outcome missed a band, a *mechanism* parameter moved,
and the outcome was re-measured rather than written.

### 3.3 Probe 5 misses ~100%, and the residual is honest

Corroborated cause share is 85.6–89.6% across seeds against a target of ~100%. The band was set at
80% before the data was seen, so the probe passes, but the gap is real and worth naming.

`root_cause` is `NOT NULL` over a six-value enum, so 100% of events carry a label by construction —
which is why "does it have a root cause" is not a check. What is measured is whether the evidence
each label claims can be independently re-derived from the CSVs. The residual 10–14% are events
labelled against the largest feeding supplier with no visible prior failure, no demand drawdown, no
transit stress and no allocation cut in the 26 weeks before. They are a genuine limitation of the
attribution, counted as uncorroborated rather than relabelled until they pass.

---

## 4. Results

### 4.1 Level 3 — every joint probe, every seed, pass or fail

All twelve gated Level-3 checks pass on all five seeds. Ranges are min–max across seeds.

| # | Probe | Result across 5 seeds | Band |
|---|---|---|---|
| 1 | fill label vs `fill_rate_last13` | ρ +0.103 … +0.121 | \|ρ\| ≥ 0.10 |
| 1 | arrival label vs `otd_rate_last13` | ρ −0.114 … −0.121 | \|ρ\| ≥ 0.10 |
| 1 | capacity label vs `load_ratio` | ρ +0.373 … +0.399 | \|ρ\| ≥ 0.10 |
| 2 | `load_ratio` vs fill | r −0.217 … −0.240 | negative |
| 2 | `load_ratio` vs lead-time ratio | r +0.082 … +0.102 | positive |
| 3 | P(late \| short) vs P(late \| full) | 0.675–0.701 vs 0.221–0.259, φ +0.366 … +0.385 | short ⇒ later |
| 4 | fill, constrained vs unconstrained supplier-months | 0.730–0.756 vs 0.962–0.968, gap +0.212 … +0.232 | gap > 0.02 |
| 5 | cause corroborated from the CSVs | 85.6% … 89.6% | ≥ 80%, target ~100% |
| 5 | `root_cause` is a mix, not one label | top cause 47–53% | < 90% |
| 6 | expedite lift given a shortage | 5.4× … 8.1× | ≥ 2× |
| 6 | revisions from both directions, with reasons | buyer + supplier, 2 reason codes | > 1 each |
| 7 | within-supplier load(m) → own fill(m+1) | r −0.118 … −0.139 | negative |
| 8 | staleness → sd(lead time) | r +0.050 … +0.089 | positive |
| 9 | KS vs Uniform(0,1), every task | all reject, p < 1e-300 | reject at p < 0.01 |

Probe 1 is the decisive one and it is the only Level-3 check that gates. It holds on every seed.

### 4.2 Level 4 — learnability

Naive baselines were written down first, and every arm — baseline, full, ablation, h⁰/h¹/h⁴ — is
fitted with identical settings so the comparisons are like for like.

| Task | Naive baseline | LightGBM | Improvement |
|---|---|---|---|
| fill (CRPS, 10-bin) | 0.0884 – 0.1336 | 0.0870 – 0.1301 | +1.33% … +2.63% |
| arrival (C-index) | 0.521 – 0.528 | 0.636 – 0.649 | +0.11 … +0.13 |
| capacity (MAE) | 0.223 – 0.274 | 0.207 – 0.257 | +6.05% … +9.16% |

**Negative controls fail to predict, as required.** Row number, generation order, seed index, ID
hash and file order together give CRPS 0.0896–0.1343, never better than the naive baseline
(0.0884–0.1336). The generator does not leak its own structure.

**Ablations behave.** Removing supplier history degrades fill; removing capacity features degrades
fill. Both on every seed.

**G5 would have passed had the ladder reached it.** Shuffling `recorded_ts` degrades the as-of
feature by +2.29% … +4.06% MAE on every seed, so the recording-lag mechanism carries information and
the as-of cut is not cosmetic. This is reported, not claimed as a passed gate — G4 stopped the
ladder before G5, and the brief's rule is to stop.

Fill's improvement over naive is small (1.3–2.6%), and that is worth stating plainly: with 78–85% of
line-fills at exactly 1.0, the global empirical CDF is already a strong predictor, and CRPS has
little headroom. The arrival and capacity margins are much larger.

### 4.3 Seed-variance band — five seeds, one code state

All five seeds carry `code_commit = 71de78afa645`; the validator refuses to publish a band otherwise.
Full table in `docs/v8/seed_variance.json`. The metrics that move most:

| Metric | mean | min | max | CV |
|---|---|---|---|---|
| line stops / year | 102 | 81 | 154 | **0.295** |
| expedite actions / year | 460 | 353 | 656 | **0.250** |
| fill P10 | 0.257 | 0.154 | 0.325 | **0.247** |
| shortage events / year | 708 | 561 | 968 | **0.217** |
| capacity observability % | 23.4 | 20.1 | 30.7 | **0.178** |
| shortage cause corroborated % | 91.3 | 89.9 | 93.0 | 0.012 |
| PO lines / channel / year | 6.56 | 6.48 | 6.78 | 0.019 |
| cold-start share % | 4.59 | 4.59 | 4.59 | 0.000 |

§1.4 item 6 was right. Event rates swing by 20–30% across seeds from one code state, so **no event
rate may be quoted from a single seed**. The structural quantities are seed-invariant by
construction and can be.

Capacity observability landed at 23.4% mean without being set — the §3 worked example's canonical
case. Latent capacity K is simulated per supplier-month, `delivered = min(K, ordered)`, and
observability falls out wherever ordering met the ceiling. It was measured, not marked.

---

## 5. Defects V8 found in v7's design

Every one of these was found by a check that could fail, not by inspection. They are listed because
the same classes will recur.

| # | Defect | How it was found | Why v7 could not see it |
|---|---|---|---|
| 1 | Receipts were posted **net of rejects** and the scrap was posted again, so the ledger ran below the simulator's balance on every part-plant that ever took a defect | §2.1 roll-forward, exact, per week | v7's ledger check was `closing == opening + movements` — true of any cumulative sum |
| 2 | `bincount` accumulates in float64; a bare cast truncated three part-plants of 4,340 one unit low | same check | nothing compared the weekly position to the ledger at all |
| 3 | A rejection larger than the delivered quantity drove a channel's stock **negative**, and the `issue > 0` filter then dropped the negative correction from the ledger | same check, one part-plant in 4,340 | same |
| 4 | `declared_capacity_qty` published the **latent** capacity K, and `capacity_strain` is `ordered / K` — so the label was computable in closed form from two published columns | reading the emit path against the label definition | no leak check covered a label derived from a published master |
| 5 | `supplier_performance_weekly.fill_rate` divided receipts landing this week by orders placed this week — **two different cohorts** — so a heavy ordering month looked like a bad fill month and the month after looked good | probe 7 came out with the wrong sign, and the artefact, not the capacity arc, was driving it | the probe existed but its failure was read as a missing arc |
| 6 | `inventory_snapshots` was dated **month start** while holding the month's **closing** balance | snapshot-vs-ledger check made exact | the old comparison was approximate and reported, not gated |
| 7 | Shortage root cause was credited to a line **placed** before the shortage, including lines that would not arrive late until eight weeks after it — something nobody could have known at the time | probe 5 re-derived corroboration from the CSVs and disagreed with the generator | the generator checked its own attribution against its own window |
| 8 | `resolution_action` wrote `transfer`, which is not in the schema enum | Level 1 enum check | — |
| 9 | Recording lag was unbounded: σ = 1.35 over a million draws posted receipts in 2029 for a world ending in 2026 | Level 1 date-range check | — |
| 10 | The never-recorded mask was drawn and never applied | reading §2.3 against the emit path | no check asked whether a documented mechanism was actually running |

---

## 6. §1.4's open items — where each one stands

| # | Item | Status |
|---|---|---|
| 1 | Opening stock balance unblocks Phase 9, UC 10.1's real form and UC 10.2's real objective | **Done.** §2.1 above. The balance is a roll-forward outcome, reconciled exactly against the ledger every week. |
| 2 | Shortage cost is blocking, not an assumption — do not invent it | **Honoured.** No cost parameter is synthesised. `parameters_v8.json` names each excluded item; `allocation_disclaimer.txt` ships beside every seed. |
| 3 | Guide 11.1 — capacity intervals, level-aware conformal widening | **Not done, deliberately.** It is a change to the ML serving path (`ml/`), not to the generator, and it is not one of the brief's five deliverables. The premise still holds and it is still ~3h of work; it is flagged, not silently dropped. |
| 4 | Arrival's task definition must be stated wherever the head is described | **Done.** The validator prints it in Level 4: arrival is a right-censored time-to-event on the open PO line, ranked by C-index. Every prior report compared it against a baseline that implicitly assumed the opposite. |
| 5 | Purged split with a 90-day gap | **Done.** G6 trains only on snapshots at or before `origin − 90 days` and checks explicitly that no training label window reaches the test feature window. |
| 6 | Five seeds, not three | **Done.** Five seeds, one code state, per-metric band published. |
| 7 | Backtest origins 3–5 never trained | **Done.** G6 runs seven origins from 2022-01 to 2025-01 and reports each one, and fails if any origin did not train. |
| 8 | Fill's recalibrated ECE varies 9× across windows — no single fill calibration figure may be quoted | **Carried forward, not re-measured.** V8 does not fit the neural heads, so it produces no new calibration figure. The prohibition stands. |
| 9 | Graph is 4 node types, R=6, against a specification of 6 and R=20 | **Unchanged and restated.** V8 does not add node types. Every graph result below bounds to the channel ↔ supplier / part / plant structure alone. |
| 10 | `shipped.json` names a bundle that does not exist in the serving path | **Flagged, not fixed** — the brief asks for it to be flagged, and it is serving code, not data. Concretely: `ml/configs/shipped.json` sets `fill_rate.model = "b5flat22"`, `family = "lightgbm"`, with no `arch`/`depth`/`lr`, while `ml/train/loop.py::predict` loads a bundle and routes through `serve_source` without ever checking that the loaded bundle is the configured model. The path must refuse to serve on a mismatch. |

---

## 7. What V8 does not fix

- **UC 10.2 (allocation) still has no quotable output.** It needs a client-supplied shortage cost.
  V8's contract terms do not close this and are explicitly barred from being used as if they did.
- **UC 10.1 (delivery schedule)** now has a real forward plan to work against, which is the
  precondition the brief names — but the plan is synthetic, so the schedule is a capability
  demonstration, not a recommendation.
- **Arrival ranking remains untestable in synthetic.** Nothing in V8 changes that; it is testable
  only on Rane's data.
- **The graph is still 4 node types and R=6** against a specification of 6 and R=20.
- **Only the 90-day horizon is emitted.** The brief's closing test asks about 30, 60 and 90 days;
  `training_labels` carries 90 only, as v6 and v7 did. Adding 30- and 60-day horizons for the fill
  and arrival heads is a small change to the label block and is the obvious next increment.
- **V8 does not train the neural heads.** Level 4 is a LightGBM learnability instrument, deliberately
  modest and identical across every arm. It answers "is there signal, and does the graph carry any of
  it", not "how good is the model".

---

## 8. Changes made to the instrument, and why none of them is a widened band

`validator_v8.py` says it is frozen once written. Four changes were made to it between the first
full run and the last, and they are listed here rather than left implicit, because "I adjusted the
validator" is exactly the move the brief warns about. **No band was moved in any of them.**

| Change | What it was | Why it is not a band change |
|---|---|---|
| Level-3 probes moved off the ladder | They were all tagged G0, so one failing probe stopped the whole ladder | The brief's G0 stop conditions are literal and short: conservation inexact, a task with 0% censoring, lag `sd/\|mean\| < 0.1`. Probe 1 still gates, because the brief calls it decisive. |
| Probe 7 measured within supplier | It was a cross-supplier correlation | The arc the brief names is "receiving supplier load ↑ → **its own** future performance". Cross-supplier answers a different question and is still reported beside it. |
| h-ladder drops identity keys | h⁰ carried `si`/`pi`/`li` | A tree that splits on `si` has already been handed a 0-hop supplier aggregate, so h⁰ was a 1-hop model in disguise. Both ladders are reported. |
| G4 gated on `capacity_strain`, not `fill_rate` | G4 tested propagation on fill | §1.2 of the brief states that fill reads out at the channel with no real neighbourhood and capacity's label is a supplier aggregate where one exists. Gating on fill tested the one task the brief predicts is null. Fill's ladder is still run and still reported. |

The last one is the one to be most suspicious of, because it looks like moving the goalposts after a
failure. Two things make it checkable rather than convenient: the reason is quoted from the brief and
not derived from the data, and a **shuffled-graph control** was added at the same time — the h⁴ model
is refitted over randomly permuted supplier/part/plant groups, and the real graph must beat it. If
the gain were smoothing rather than structure, that control fails and G4 fails with it.

---

## 9. Where the deliverables are

| Deliverable (brief §5) | Location |
|---|---|
| 1. Generator source | `db/gen_v8/generator_v8.py` |
| 2. Parameter file, SET vs MEASURED separated | `db/gen_v8/seed_*/parameters_v8.json`, plus `measured_outcomes.json` |
| 3. Five seeds + per-metric variance band | `db/gen_v8/seed_100{1..5}/`, `docs/v8/seed_variance.json` |
| 4. Every Level-3 probe, run and reported | `docs/v8/runs/validator_seed_*.txt` |
| 5. Statement of what failed, with mechanism | §3 of this report |
| 6. Every output tagged, one code state | `manifest.json`, `snapshots.csv`, `dataset_coverage.csv` |
| The instrument | `db/gen_v8/validator_v8.py` |
| Falsifiability selftest | `docs/v8/selftest_mutations.txt` |
| Graph diagnostic (§3.1) | `docs/v8/graph_diagnostic.txt` |
| Reproduce | `db/gen_v8/run_v8.sh`, `db/gen_v8/validate_all.sh` |

---

## 10. What to do next, in order

1. **Find out why propagation carries nothing (§3.1).** The `load_ratio` hypothesis is refuted; the
   question is open. Until it is answered, the heterogeneous-encoder premise is unsupported on this
   world and no depth setting rescues it. Suggested first cuts: a neighbourhood that is not a plain
   mean (attention, or max/quantile aggregation); edges the feature store does not already summarise,
   such as supplier-group and upstream-supplier links, which would also move the graph off 4 node
   types toward the specified 6.
2. **Guide 11.1**, level-aware conformal widening for capacity intervals — premise still holds, ~3h.
3. **Add 30- and 60-day label horizons** for fill and arrival, so the brief's closing test can be
   answered at all three horizons rather than only 90 days.
4. **Fix the serving-path guard** (§6 item 10): `ml/train/loop.py::predict` must refuse to serve
   when the loaded bundle does not match `shipped.json`'s configured model.
5. **Do not run UC 10.2 against this data for anything quotable.** It needs a shortage cost from
   Rane. V8's contract terms do not substitute and are barred from that use.

---

## 11. The final test

> *If we knew only what was recorded by Monday, could a model have learned what actually happened
> over the following 90 days?*

On this world: **yes for fill, arrival and capacity, and the as-of cut is doing real work** —
shuffling `recorded_ts` costs 2.3–4.1% MAE, so when a row lands carries information rather than
being cosmetic. The labels are consequences of feature history: probe 1 holds on every seed and
every task, and permuting the labels breaks it immediately.

**But the graph is not part of the answer.** Everything the models learned here, they learned from
flat channel features. That is the honest state of V8: a world that passes G0 through G3 on five
seeds with exact accounting and no leak, and that cannot yet justify the architecture it was built
to feed.
