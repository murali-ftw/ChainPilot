# synthetic_rules.md
## ChainPilot × Rane — project-specific rules for generating the synthetic world

**Supersedes:** the generic `synthetic.md`. Where the two differ, this file wins.
**Reads alongside:** `dataset_structure.md` (schema, authoritative), `model_plan.md` (what is
learned and what is not), `V4_Corrections.md` (what the last generator got wrong),
`data_plan.md`, and `validator.py` (the acceptance instrument).

**Why this file exists.** The generic rules are correct and they are not enough. Three
generated datasets satisfied them in spirit and failed anyway, because the rules did not carry
the numbers, the entity counts, the table names, or the specific failure modes this project has
already hit. Everything below is either a concrete target measured on a world that passes, or a
prohibition traceable to a defect that actually shipped. Nothing here is illustrative.

---

# 0. The customer and the shape of their world

Rane Group, Indian automotive components. **7 plants.** Their own stated operational rates,
which the synthetic world must reproduce because they are the sanity anchor:

| Rane's stated figure | Per year, 7 plants |
|---|---|
| Line stops | ~18 |
| Expedites | ~120 |

Indian auto carries strong calendar effects — festive-season build-ahead, March fiscal year-end,
monsoon logistics. A flat year is a generator that forgot them, not a plausible alternative.

**Scope discipline.** Of the eight use cases, only three are GNN tasks: **delivery risk, fill
rate, part shortage**. Part demand over three months is BOM arithmetic, not forecasting.
Allocation is a solver. Production exposure and delivery split are simulation. The synthetic
world must serve all eight, but do not build "signal" into the arithmetic ones to make them look
learnable — that is a leak, not a feature.

---

# 1. Time architecture

**Full world:** `2016-01-01 → 2026-09-07` — ten full years plus a partial current year.
**Primary modelling window:** `2019-01-01 → 2025-12-31`.
**Regime block, retained not discarded:** 2016–2018.

The partial trailing year is deliberate: it is what a real extract looks like, and it is where
right-censoring comes from naturally.

A previous version ran from 2012. The extra four years bought nothing — the recommended
modelling window excluded them anyway, and the density per week fell. **Length is not a
substitute for density.** See §5.

**Grain.**

- Persistent sequence entity: **sourcing channel = supplier × part × plant**, observed weekly.
  This is what Temporal-SHARE reads. Channels have ten years of history.
- Instance entities: PO line, PO schedule, ASN, GRN line, inspection. These do **not** have
  seven-year sequences and must never be treated as if they do.
- Everything transactional stays at event grain. Weekly stores are **derived** afterwards, never
  sampled.

---

# 2. Entity population — non-negotiable minimums

A previous version shipped 320 channels. At that size the heterogeneous graph is smaller than a
4-layer encoder's receptive field, layers 2–4 do nothing, and the architecture cannot be
evaluated at all. This is readiness gate G1 and it fails before any causal work matters.

| Quantity | Target | Reference world (passes) | V3 (failed) |
|---|---|---|---|
| Plants | 7 | 7 | 7 |
| Sourcing channels | ≥ 15,000 | 16,072 | 320 |
| Channels that ever traded | ≥ 14,000 | 15,334 | 320 |
| Channels never traded (cold-start slice) | 3–6% | 738 (4.6%) | 0 |
| Channel-week rows | ≥ 5M | 5,937,771 | 55,000 |
| Weeks in the weekly store | ≥ 500 | 523 | 401 |
| Channel-weeks per traded channel | ≥ 350 | 386.8 | 171.9 |
| PO lines per channel per year | ≥ 6 | 7.02 | 4.15 |

**The cold-start slice is mandatory.** Channels with no trading history are exactly what
production hands the model on day one with Rane. A world containing none of them cannot test the
behaviour that most determines whether the POC survives contact with real data.

---

# 3. The as-of contract — the rule that broke three times

## 3.1 The gate is on `recorded_ts`, never `event_ts`

For snapshot `t0`, a fact enters the feature set iff `recorded_ts <= t0`. For effective-dated
records: `effective_from <= t0 AND (effective_to IS NULL OR effective_to > t0)`.

Gating on `event_ts` is the leak that makes a model look excellent and be unbuildable.

## 3.2 The recording layer is a process, not an offset

**What went wrong, three times:** `recorded_ts = event_ts + constant`. Reporting lag came back
with `sd/|mean| = 0.000` on nearly every table — +1.00d, +0.25d, +0.17d, +0.08d depending on the
table. On three tables the constant was negative, producing 12,201 inventory transactions and
4,227 acknowledgements recorded before they happened.

**Requirements:**

- Lag is drawn from a **right-skewed** distribution (log-normal or gamma), per table, with a
  genuinely long tail. Not constant, not symmetric, not shared across tables.
- Lag is **state-dependent**. A plant in a disrupted week records later. A supplier about to miss
  a commitment acknowledges late or never. A receipt during a shutdown is back-entered weeks
  afterwards. The correlation between lag and trouble is itself signal; an i.i.d. lag destroys it.
- Some records are **never entered**. This is what makes a fact unobservable rather than merely
  late, and it is the mechanism behind genuine censoring.
- Lag is **never negative**, except where the anchor is a forward-dated validity-window start
  (`effective_from` on contracts, capacity, allocation) — agreeing a figure before it takes
  effect is normal and must be exempted explicitly rather than silently.

**Reference distribution from the world that passes:** staleness P50 12 days, P90 838, max 3,569
measured across channel-weeks. A per-table lag with 14–45% of rows recorded in a *later week*
than the event, with `sd/|mean| ≥ 0.5`, is the shape to hit. One table in V3 got this right —
`po_line_revisions`, at 44% later-week, P50 2d / P90 18d / max 29d, `sd/|mean| = 7.485` — and it
is the template for the rest.

## 3.3 Visible week

A derived weekly store buckets each row on `max(event_week, recorded_week)` — the week in which
the fact was both true and knowable.

Bucketing on the event week while gating on `recorded_ts` is a contradiction: a row whose event
falls in week 10 and whose entry falls in week 12 is not visible in week 10, and by week 12 its
event week no longer matches. It lands in neither and is silently dropped, reading as
`qty_ordered = 0` — indistinguishable from a genuinely idle week. This cost 4.45% of ordered
units and 6.86% of received units in one version, and survived three audits.

---

# 4. Conservation — exact, not approximate

For every derived weekly store, every unit in the source table appears in **exactly one** week
of the store. Integer equality, not a tolerance:

```
source_qty(table, quantity) == emitted_qty(store, quantity)
```

Rows excluded from one side must be excluded from the other: rows whose visible week falls
outside the store's span, and rows whose key is absent from the master. Both sides filtered
identically, so the identity stays exact.

**This is the single most valuable check in the suite.** It is the only thing that distinguishes
"sparse because the world is sparse" from "sparse because the builder dropped rows," and those
two are indistinguishable by eye. V3 emitted 138,615,579 ordered units against a source of
4,114,197 — a factor of 33.7, with received at 37.0. Not even a consistent multiple, so not a
units bug: the store had no relationship to its source at all.

Applies to: `channel_performance_weekly`, `supplier_performance_weekly`, and the PO/GRN header
↔ line identities.

**`part_demand_weekly` is exempt from unit conservation** and this exemption is deliberate: the
store is a BOM explosion scaled by a drift quantile, rounded, and thresholded at half a unit, so
it is not unit-preserving by construction. A tolerance band wide enough to survive the rounding
could not fail on a broken builder. Three exact identities the spec *does* state are asserted
instead: `horizon_days = week_start − as_of_date`, `p90 >= p50`, and `week_start > as_of_date`.

**Inventory ledger identity**, per part × plant, per period, exact:

```
closing = opening + receipts + transfer_in + returns
                  − consumption − transfer_out − scrap ± adjustments
```

Snapshots are computed from transactions. Never generate both independently. Inter-plant
transfers must exist — they are a hidden supply/consumption stream and a real mitigation lever.

---

# 5. Sparsity — the world is mostly idle

A procurement calendar is not a row per channel per week. Most channel-weeks have no order.

| Property | Target | Reference world |
|---|---|---|
| Channel-weeks with `qty_ordered = 0` | 80–93% | 81.4% of the panel; 90.1% of emitted rows |
| Channels with a contiguous weekly panel | ≥ 95% | 100% |
| Duplicate PKs in any weekly store | 0 | 0 |

V3 produced 0.0% idle weeks and 0.0% contiguous panels, with 10,210 duplicate PKs on 55,000 rows
across 320 channels — the signature of sampling (channel, week) pairs with replacement rather
than walking a calendar. A rolling 13- or 52-week window cannot be computed over a scattered
sample of channel-weeks, which means the entire feature store is unbuildable.

The panel must be **complete and contiguous**: every channel has a row for every week between its
first and last activity, most of them idle.

---

# 6. Supplier capacity — latent, and observability emerges

Define latent `K(supplier, part, month)` evolving over time through machine additions,
breakdowns, maintenance, labour, shift changes, competing customers, stress, seasonality,
disruption, and recovery.

```
delivered = min(available capacity, ordered requirement)
```

subject to fulfilment behaviour, transport, quality and noise.

**Observability is an outcome, never a setting.** Where `ordered << K`, the month carries no
information about K. Where `ordered` approaches or exceeds K, capacity reveals itself.

| Property | Target band | Reference world |
|---|---|---|
| Supplier-months constrained (measured on months with activity) | 20–30% | 24.1% |
| Capacity unobservable | ≥ 70% | 75.9% |

If the realised share misses, adjust the **order policy** or the **capacity distribution** and
re-run. Never adjust the observability. Empty months are not evidence either way and must be
excluded from the denominator — including them inflates it until the check cannot fail.

---

# 7. Fill rate — the distribution must be produced, not manufactured

Fill emerges from fulfilment behaviour driven by persistent supplier state, capacity pressure,
recent performance, demand pressure and disruption.

| Property | Target band | Reference world |
|---|---|---|
| PO lines with fill < 1.0 | 8–15% | 9.5% |
| Mass at exactly 1.0 | ≥ 60% | 90.5% |
| Mass at exactly 0 | 1–3% | 1.4% |

The point masses matter more than the band. A continuous haircut puts every line just *below* 1;
the absence of a spike at exactly 1.0 is the signature of a naively generated set, and the binned
CDF head has no bin to put its mass in. V3 returned 27.3% at 1.0 and 31.8% at 0 — a distribution
with no structure at either end.

**Persistence requirement.** A supplier that has degraded for four consecutive months must carry
a different future risk profile from one that is consistently mediocre, even at equal mean fill.
Without this, the temporal encoder has nothing to encode and h⁰ ≈ h⁴.

---

# 8. Arrival timing — a survival process with real censoring

Arrival depends on supplier, origin site, destination plant, mode, distance, shared logistics
checkpoints, historical performance, capacity stress and disruption.

| Property | Target band | Reference world |
|---|---|---|
| PO lines arriving late | 15–25% | 12.83% — **known miss, see below** |
| Lead-time skewness | ≥ 0.5 | +2.36 |
| Lead time median / P90 / P99 | right-skewed | 31d / 67d / 139d |
| Right-censored tail | 0.3–1.2× span-implied | 0.56× |

For an open PO at `t0` the world knows `T > current observed age` and nothing more. That is
**right censoring**, and it must be represented as such in the label, never deleted. Open lines
are the population a hazard model exists to consume.

A Gaussian lead time (V3: skew +0.05, median 50 / P90 78 / P99 93) is a generator artefact, not
a supply chain.

> **Known open issue.** Our own reference world reads 12.83% late against a 15–25% band, on the
> normal-regime slice. This is a genuine property of that world, reproduced line-for-line by an
> independent re-implementation. It has not been fixed by widening the band, and it should not be
> in the new generator either. Fix the mechanism or report the miss.

---

# 9. Shortage — emerges from stock balance

```
Shortage = max(0, safety_stock − projected_inventory)
```

rolled forward from demand, inventory, open POs, expected receipts, and the fill / arrival /
capacity uncertainty around them.

**Two distinct populations, and conflating them is a defect:**

1. **Shortage conditions** — part × plant × week observations below safety stock. Target 2–5%.
2. **Recorded shortage events** — the escalated subset that reaches `shortage_events.csv`.
   Strictly a subset. Many conditions are absorbed by inter-plant transfer, expedite, supplier
   recovery, allocation change, rescheduling or alternate sourcing.

| Property | Target, 7 plants | Reference world |
|---|---|---|
| Shortage events / year | 180–250 | 214.7 |
| Line stops / year | 15–25 | 20.0 |
| Expedites / year | 100–150 | 124.8 |

Every recorded event must carry a `root_cause` that comes from the mechanism that actually
produced it, and a `responsible_supplier_id` that traces to a real channel. **Target: 100% of
shortage events have an identifiable upstream cause** — a fill miss or late arrival on a feeding
channel in the preceding weeks. A shortage with no antecedent anywhere in the receipt history was
inserted, not caused.

**Base-rate note for modelling:** the shortage head's operating base rate is ~3%. The world's
natural rate may sit above that; reaching 3% is done by downsampling positives at training time,
**not** by suppressing shortages in the generator.

---

# 10. Feedback and propagation — what makes the graph earn its place

A chain that runs strictly downward and stops at shortage produces a dataset where
`expedite_events`, `po_line_revisions`, `alternate_sources`, `supplier_allocation` and `tooling`
conform to schema and carry no information.

Mandatory arcs:

| From | To | Mechanism |
|---|---|---|
| Shortage / projected shortage | Expedite | expedite raised against the open PO line |
| Shortage | PO line revision | quantity or date revised on open lines |
| Shortage | Alternate sourcing | volume shifted, changing `supplier_allocation` |
| **Alternate sourcing** | **Receiving supplier's utilisation** | **its load rises** |
| **Receiving supplier's load** | **Its other channels' fill and lead time** | **degradation propagates** |
| Line stop | Production plan | subsequent plan revised, feeding demand drift |
| Sustained poor performance | Allocation | share shifted away over months |

**The two bold rows are the whole reason a GNN is in this architecture.** They are the only path
by which supplier A's failure reaches a *different* channel through supplier B. Without them the
h⁰ diagnostic will show the graph contributing nothing, and a per-entity tabular model dominates.
For reference, on HADES v3 the graph contributed roughly **10% of the above-chance signal** on
delay — small, real, and measurable. That is the number the new world has to at least match.

Disruptions must also **recover**: `normal → shock → degradation → mitigation → recovery →
normal`. A permanently broken supplier is not a model of supplier behaviour.

**Tier-2 propagation.** `Tier-1 A → Tier-2 X ← Tier-1 B` means one Tier-2 disruption hits
multiple Tier-1s simultaneously — this is the correlated-failure structure the Monte Carlo
shortage simulation and the Gaussian copula exist to estimate. Visibility must be imperfect: some
dependencies known, some partial, some unknown.

---

# 11. Disruptions — two classes, entering at the right nodes

**Supply-side** enters at **supplier capacity** and at **transit**, never at material requirement:
capacity loss, shutdown, transport disruption, upstream material constraint propagated through
`supplier_upstream`.

**Demand-side** enters at the **demand generator**: programme pull-in, cancellation, model-year
change, production surge, volume reduction.

A previous plan routed COVID / strike / delay into Material Requirement. That is wrong twice over:
MRP is BOM arithmetic off the plan and has no mechanism by which a disruption could reach it; and
routing supply shocks through a demand-side node entangles `demand_drift` with `capacity_strain`,
which are precisely the two things two of the heads exist to separate.

The 2019–2025 window deliberately contains COVID and the chip shortage. Event-rate bands describe
**normal operation**, so gate them on the normal-regime slice and report the disrupted slice
separately rather than widening the band to cover both.

---

# 12. Demand and plan

Demand carries trend, seasonality, product mix, plant differences, part usage, operational noise
and shocks. Not a sine wave, and not one global seasonal pattern.

| Property | Target | Reference world |
|---|---|---|
| Month-of-year peak/trough ratio | > 1.10 | 1.30 |

Production plans are versioned `V1 → V2 → V3 → … → Actual`, with firm and tentative periods
behaving differently. Actual production depends on plan, demand conditions, **material
availability**, plant capacity and disruptions — never generated independently. The revision
sequence is model history, and horizon-dependent error is the demand-drift signal.

---

# 13. Mechanism parameters vs. measured outcomes

**This is the discipline that the previous three attempts failed.** You set mechanism parameters.
You measure everything else. When a measured outcome misses its band, you adjust a *mechanism
parameter* and re-run. You never adjust the outcome.

| You SET these | You MEASURE these — never set them |
|---|---|
| Capacity distribution per supplier tier | % of supplier-months where capacity is observable |
| Order policy: safety stock, lot size, reorder point, aggressiveness | Fill distribution and its masses at 0 and 1.0 |
| Lead-time process: lane base, variance, disruption multiplier | Lead-time median / P90 / P99 / skewness |
| Recording-lag distribution per table and its state dependence | Staleness P50 / P90 / max; later-week share |
| Disruption calendar: onset, duration, severity, which suppliers | Shortage / line-stop / expedite counts per year |
| Demand seasonality amplitude and phase | Demand peak/trough ratio |
| Entity population and channel activity rates | Zero-order week share; channel-weeks per channel |
| Supplier quality and reliability priors | Late↔short correlation; constrained-month fill gap |

**Source audit.** If any value in the right-hand column appears in the generator as a literal, a
parameter, a clip, a post-hoc adjustment or an assertion, the dataset is fitted rather than
generated — however well it validates. Grep the source for 0.25, 0.87, 0.90, 1.30, 0.15–0.25 and
the other band values before shipping.

---

# 14. Labels

Built **after** the world, per snapshot `t0`: freeze the world, build features from what is known
by `t0`, look forward into the simulated future, record the actual outcome.

| Task | Entity | Derived from |
|---|---|---|
| `arrival_week`, `arrival_delay`, `label_censored` | `po_line` | PO → ASN → GRN actual arrival, right-censored if unresolved |
| `fill_rate` | `po_line` | ordered vs cumulative GRN receipts |
| `capacity_strain_30/60/90d` | `channel` | supplier utilisation over rolling months |
| `demand_drift` | `product_plant` | production plan vs actual |
| `shortage_qty` | `part_plant` | inventory balance after demand and receipts |

**Entity binding is one-to-one.** V1 drew `entity_type` uniformly across five types per task, and
drew `entity_id` from the wrong pool — rows labelled `channel` carried `PART00010`, rows labelled
`supplier` carried `CH00062`, 2,252 of 2,292 unresolved. Every `entity_id` must resolve into the
master for its declared type.

**Labels must not be uniform noise.** V1's five tasks returned mean 0.4941–0.5034 with sd
0.2872–0.2927 and ~2,100 distinct values on ~2,400 rows. Uniform(0,1) has mean 0.5 and
sd 1/√12 = 0.28868. Those were `random()` with task names attached. `arrival_week` — a week index
— and `shortage_qty` — a part count — were both bounded in [0,1], which is a stronger tell than
any test. **A KS test against Uniform(0,1) must reject at p < 0.01 for every task.**

**Censoring must be non-zero and non-total** on every task: 1–60%. At exactly 0.0% nothing is
withheld and observability is untested — and a gate that only asks whether the column exists
passes on 837,318 uncensored rows all equal to exactly 1.0, which is what happened.

---

# 15. Never do these

Traceable to defects that shipped:

```
Never generate a label independently of the features.
Never use a row counter, record ID, generation order, seed index, or sin/cos of an index as a target.
Never draw entity_type or entity_id from a pool other than the task's declared entity.
Never set capacity observability; let min(K, ordered) produce it.
Never tune staleness, seasonality, fill mass, or an event rate after observing it.
Never use a constant recording lag, and never allow a negative one.
Never bucket a derived store on event week while gating on recorded_ts.
Never patch a derived store after generation to force conservation.
Never generate shortage events independently of inventory.
Never generate production actual independently of plan and material availability.
Never generate inventory snapshots independently of transactions.
Never generate GRNs independently of shipments and POs.
Never change allocation without changing the receiving supplier's load.
Never delete censored observations.
Never write a gate that cannot fail: `stddev > 0` passes at 1e-5, and `len(distinct) <= 2` is not a test for binary labels.
```

The last one has a name in this project — **"gates that can't fail"** — after three separate
instances. Any new check must be able to fail on a plausibly-broken dataset, or it is deleted and
the deletion is explained.

---

# 16. Acceptance — four levels plus the readiness ladder

`validator.py` is the instrument. It is **frozen**: a band adjusted until the data passes has
measured itself, not the data. Path discovery may change; nothing else may.

**Level 1 — structural.** PK uniqueness, FK integrity, entity resolution, nullability, enums,
types, date ranges.

**Level 2 — accounting.** PO and GRN conservation, inventory balance, allocation sums, production
balance. Exact integer identities.

**Level 3 — causal (joint distributions, not marginals).** These are what a band-tuner does not
think to satisfy, and they are checked in this order of importance:

1. **Each label vs. its most predictive feature** in the corresponding weekly store — fill label
   against `fill_rate_last13`, arrival against `otd_rate_last13`, and so on. **Decisive.**
   Near-zero means the labels are detached, and every other number is irrelevant.
2. Capacity pressure ↑ → fill ↓, and → lead time ↑
3. Late and short co-occur on the same PO lines
4. Fill in constrained vs. unconstrained supplier-months, materially lower in constrained
5. Share of shortage events with an identifiable upstream cause — target ~100%
6. Shortage → expedite / revision / alternate-sourcing probability ↑
7. Alternate sourcing → receiving supplier load ↑ → its future performance may deteriorate
8. Staleness ↑ → lead-time variance ↑ across channels
9. KS test of every label against Uniform(0,1), rejecting at p < 0.01

Direction and mechanism are the goal, not a particular coefficient.

**Level 4 — learnability.** Before any neural model:

- **Naive baselines, written down first**: per-channel historical mean fill, per-lane median lead
  time, naive seasonal demand, marginal shortage base rate. Without these there is no way to tell
  a working model from a broken one.
- **LightGBM on tabular features only**, per task, on a proper time split. If it cannot beat the
  naive baselines, stop. Do not put a neural network on a broken world.
- **Negative controls**: row number, generation order, seed index, ID hash, file order. These must
  *fail* to predict. If one of them works, the generator leaked its own structure.
- **Ablations**: remove supplier history → supplier-risk performance degrades; remove capacity
  features → capacity prediction degrades; remove network relationships → propagation performance
  degrades. If removing a supposedly important feature changes nothing, its causal role is not
  actually in the generator.
- **h⁰ diagnostic**: features-only vs 1-hop vs 4-hop. If h⁰ ≈ h⁴ the graph carries nothing and
  the architecture premise is unsupported on this data. Reference: ~10% of above-chance signal on
  delay in HADES v3.

**Readiness ladder** — run in order, stop on failure, never widen a gate to proceed:

| Gate | Asks | Stop if |
|---|---|---|
| G0 | Is this a dataset? | conservation inexact, any task 0% censored, lag `sd/\|mean\| < 0.1` |
| G1 | Does the graph exist? | median channel degree 1, or components smaller than the receptive field |
| G2 | What is the number to beat? | naive baselines not computed |
| G3 | Is there signal at all? | LightGBM cannot beat G2 |
| G4 | Does the graph contribute? | h⁰ ≈ h⁴ |
| G5 | Is the as-of gate binding? | shuffling `recorded_ts` does not degrade performance |
| G6 | Are the folds sound? | label window in train overlaps feature window in test; too few positives per fold |
| G7 | Will it survive production? | no cold-start slice; no seed-variance band |

---

# 17. Reproducibility

Two seeds minimum under identical code — **1001 and 1002** — with similar distributions,
different individual histories, identical mechanisms. Publish the per-metric variance band
between them. Any later claim that a change improved a number must clear that band.

Every output carries: seed, generator version, parameter version, `code_commit`, generation
timestamp, dataset version. If multiple scales are generated, all of them come from **one** code
state and share a `code_commit`.

Seed-to-seed variance is not a proxy for generalisation to Rane's real world. For Rane, the
evidence is **rolling-origin backtesting** on their extract, not multi-world agreement.

---

# 18. What ships

1. The **generator source**, not only the CSVs. The dataset is not reviewable without it.
2. A **parameter file** listing every mechanism parameter and its value, explicitly separated from
   measured outcomes, in the shape of §13's table.
3. **Two seeds** and the per-metric variance band.
4. The **Level 3 joint probes** run and reported with their statistics, whether they pass or fail.
5. A **statement of what failed**, with the mechanism explanation for each.

**On the commitment.** Do not promise zero validation failures in advance. Committing to that
against a validator whose bands are already published is exactly the incentive that produced three
fitted datasets. A world that reports "conservation exact, labels causal, all joint probes
positive, seasonality came out at 1.08 against a 1.10 band" is far more trustworthy than one
reporting 0/0/0 — the first describes a world, the second describes a fit.

---

# 19. The final test

> **If we knew only what was recorded by Monday, could a model have learned what actually
> happened over the following 30, 60 and 90 days?**

If the honest answer is instead *"the distributions are right, the validator passes, but the
labels are not consequences of the feature history"* — it is not training data. It is a
collection of plausible CSVs, and this project has now produced three of those.

Generation order is always:

```
REALISTIC MECHANISM → REALISTIC RELATIONSHIP → REALISTIC OUTCOME → REALISTIC DISTRIBUTION
```

Never the reverse. The generator is part of the ML experiment, not an input to it.
