# HADES-Bench
## A Hypothesis-Driven Benchmark for Reasoning Under Incomplete, Dynamic and Partially Observable Supply Chains

**Status:** authoritative source of truth for the V2 generator and V2 documentation set. Both
`V2_MASTER_PROMPT.md` Part 1 and Part 2 read this file directly. Sections marked **[V2
clarification]** are corrections to the original spec draft and take precedence over the prose
around them where the two conflict. Sections marked **[drafted]** fill gaps the original spec
left open; they are proposed defaults, not yet validated against a generated dataset, and should
be revisited after Phase 0 (power check) and Phase 6 (generate/validate/report) of the build.

---

## Objective

HADES-Bench is designed as a controlled benchmark for evaluating graph learning architectures
under realistic supply-chain uncertainty. Unlike conventional synthetic datasets that primarily
evaluate message passing on static graphs, HADES-Bench models supply chains as adaptive economic
systems where companies operate with incomplete information, hidden upstream dependencies,
changing supplier relationships, delayed observations, and autonomous decision making.

The benchmark is not intended to maximize realism through arbitrary complexity. Instead, every
mechanism is deliberately introduced to evaluate a specific reasoning capability while remaining
independently configurable and experimentally isolatable. Every benchmark mechanism answers one
scientific question.

---

## Core Design Principles

### 1. Hypothesis-Driven Design

Every source of uncertainty introduced into the benchmark must correspond to a measurable
hypothesis. No mechanism should exist solely to make the benchmark more difficult. Performance
degradation should always be attributable to an identifiable cause.

### 2. Independent Economic Agents

Every supplier is modeled as an independent economic and legal entity. Companies actively respond
to disruptions through:

- supplier substitution
- inventory management
- logistics adaptation
- contractual flexibility
- operational mitigation

Consequently, disruptions should not propagate uniformly through the graph. Instead, disruption
influence may be attenuated, redirected, amplified, or absorbed, depending on each supplier's
hidden operational state.

### 3. Time-Causal Simulation

Every supplier decision must depend only on information available at time *t*. The simulator must
never leak future information into present decisions. Examples of prohibited behavior include:

- switching suppliers before disruption occurs
- replenishing inventory using future knowledge
- reacting to events that have not yet happened

Labels are generated from true event time while the model only observes delayed operational
information.

**[V2 clarification — two clocks, executable, not documented]** The dataset carries two
timelines: **true event time**, used for label generation, and **observed time**, equal to true
event time plus a per-source reporting delay, which is everything the model reads. This extends
V1's existing bitemporal fields (`observed_at` / `recorded_at` on `inventory_history` and
`shipment_status_history`) rather than introducing a parallel mechanism. The no-leakage rule must
be implemented as an executable assertion inside the simulation loop — every decision function
takes an explicit `as_of` timestamp and asserts every input it reads is timestamped at or before
it — not merely documented as a convention. This is the constraint most likely to erode silently
as later mechanisms stack on top of it.

### 4. Statistical Power

Every benchmark variant must contain enough positive examples to distinguish genuine
architectural improvements from random seed variation. Target approximately:

- 2,000–5,000 shortage positives
- 2,000–5,000 delay positives
- 2,000–5,000 impact positives

for every benchmark variant, **including the combined benchmark (Variant K)**.

**[V2 clarification]** Absorption (Mechanisms E/F) and visibility truncation (Mechanism A) both
suppress positive rates, so Variant K is the variant most likely to miss this target. Do not lock
config defaults (`SUP_N`, timeline length, snapshot count) until a power check has been run
against a provisional Variant K configuration. See Build Phase 0 in `V2_MASTER_PROMPT.md`.

---

## Benchmark Mechanisms

### Mechanism A — Partial Supply Chain Visibility

Organizations rarely possess complete visibility beyond Tier-1 or Tier-2 suppliers. Observable
graphs should therefore terminate at varying upstream depths. The hidden upstream network
continues to exist inside the simulator but remains invisible to the model.

**Purpose:** evaluate reasoning under incomplete graph observations.

### Mechanism B — Hidden Shared Structure

Multiple observable suppliers secretly depend on the same hidden upstream entity. The hidden
supplier never appears in the observable graph.

**[V2 clarification — supersedes the single-type description]** Mechanism B generates **three
types** by the identical structural process (same group-size distribution, same
member-selection rule, same edge structure, same member degree distribution), differing only in
downstream coupling:

| Type | Coupling | Expected model behavior |
|---|---|---|
| **A** | `alpha > 0` (see Mechanism D) | Discovery should improve prediction — hidden parent stress causally drives member risk. |
| **B** | `alpha = 0`, members' own observable features are correlated (redundant) | Discovery should succeed; prediction should **not** improve. This is V1's `H_POLYMER` behaviour. |
| **C** | `alpha = 0`, no correlation with labels at all (decoy) | A robust model should ignore this structure entirely. |

All three types live in the **same** dataset variant — the model must discriminate among them,
not be evaluated on them in isolation. A dedicated validation check must confirm that a simple
classifier cannot separate the three types from structural features alone; if it can, the
generation process is not actually identical across types and must be fixed.

**Purpose:** evaluate whether models can discover latent structural relationships, and whether
they can tell useful latent structure from irrelevant or redundant latent structure, without
being told which is which.

### Mechanism C — Dynamic Supplier Relationships

Supplier relationships evolve over time due to:

- supplier substitution
- emergency sourcing
- pricing changes
- logistics rerouting
- geopolitical events
- contractual updates

**Purpose:** evaluate robustness under evolving graph topology.

### Mechanism D — Hidden Dependency Coupling

Starting from Mechanism B (Type A groups only), hidden shared suppliers influence downstream
disruption probability:

```
Supplier Risk = Own Risk + α × Hidden Parent Stress
```

where `α` is configurable.

**Purpose:** evaluate whether discovering hidden dependencies provides genuine incremental
predictive information. Unlike HADES V1, hidden relationships are now both discoverable and
causally relevant for at least one of the three types generated under Mechanism B.

### Mechanism E — Hidden Supplier Resilience

Each supplier possesses a latent resilience state. This hidden state governs:

- recovery speed
- inventory depletion
- emergency sourcing capability
- operational flexibility
- disruption absorption

The resilience state is never directly observable. Instead, it must be inferred from historical
disruption-response behavior of neighboring suppliers.

**Purpose:** evaluate reasoning under partially observable economic behavior.

### Mechanism F — Adaptive Risk Transmission

Risk transmission is not randomly assigned. Instead, transmission behavior is generated
deterministically from the hidden supplier resilience defined in Mechanism E.

```
High resilience  → strong attenuation → disruptions disappear rapidly
Low resilience   → weak attenuation   → disruptions continue propagating
```

The model must infer transmission behavior indirectly from observable disruption histories
rather than reading it directly from node attributes.

**[V2 clarification — E and F share one latent variable]** Mechanisms E and F must not introduce
two independent hidden quantities. Attenuation along an edge is a deterministic function of the
*same* hidden `resilience ∈ [0,1]` value defined in Mechanism E, never a second latent variable.
Resilience must be recoverable, above chance, from a supplier's observed past
(disruption → outcome) pairs as they appear in `shipment_status_history` and
`supplier_temporal_features`. This recoverability must be checked with a dedicated validation
step (e.g. a simple estimator fit on observed history, confirmed to beat chance at recovering
resilience). **If resilience is not recoverable from observables, Mechanism F is irreducible
noise and any depth-reasoning result built on it is uninterpretable** — this is the single most
important validation check in the V2 build, because a depth mechanism failing on unlearnable
noise is empirically indistinguishable from genuine depth-indifference, which is the exact
ambiguity V2 exists to resolve.

**Purpose:** evaluate adaptive locality. The optimal reasoning depth should emerge from inferred
transmission behavior instead of graph distance alone.

**[V2 measurement — recoverability direction is variant-dependent]** Adding Mechanism F **inverts
the sign** of resilience's observable signature relative to Mechanism E alone. Measured with the
same estimator on the same world: recoverability is **direct in Variant E (AUC 0.561)** and
**inverted in Variant F (AUC 0.396)**.

The cause is structural, not an artifact. Under E alone, a resilient supplier absorbs its own stress,
so it looks better than its stress level predicts. Under F it *additionally* receives less
transmitted stress from upstream — which lowers the very stress variable an estimator regresses
against, flipping the direction of the observed relationship.

Three consequences bind on any submission:

1. **No baseline or evaluation harness may assume a fixed sign** for resilience recovery. The
   information is present in both variants — an inverted predictor is recovered by inverting it —
   but a model or probe hard-coded to one direction will read the other variant as noise.
2. **Variant E and Variant F results must never be pooled**, or compared as though they measure the
   same directional effect. They measure the same latent through opposite observable signatures.
3. **The Clarification 3 recoverability check therefore gates on `|AUC − 0.5|` clear of zero**, not
   on `AUC > 0.5`. (Contrast Mechanism B's indistinguishability check, where the requirement is the
   *absence* of information and a reliably-below-chance AUC is a failure.)

Underlying measurement: `docs/PHASE2_PHASE6_IMPLEMENTATION.md` §8 item 2.

### Mechanism G — Information Delay

Operational information is delayed. Examples include:

- supplier reporting delays
- ERP synchronization lag
- customs processing
- shipment visibility latency

The model observes delayed information while labels are generated from the true event timeline.
See the two-clocks clarification under Principle 3.

**Purpose:** evaluate reasoning under incomplete operational visibility.

### Mechanism H — External Shock Events

Introduce correlated disruptions through:

- earthquakes
- floods
- wars
- sanctions
- pandemics
- port closures

These events simultaneously affect multiple suppliers regardless of graph connectivity.

**Purpose:** evaluate robustness to correlated global disruptions.

### Mechanism I — Multi-Source Dependencies

Suppliers depend simultaneously on multiple upstream entities. Dependency behavior follows
deterministic AND/OR functions.

**Example:**

```
Battery Manufacturer depends on:
  - Lithium Mine
  - Nickel Mine
  - Electronics Supplier
```

Failure of upstream entities influences downstream production according to configurable
dependency weights.

**Purpose:** evaluate reasoning over realistic multi-source supply chains.

### Mechanism J — Chain-Length Heterogeneity

Supply chains should exhibit varying upstream depths — some industries naturally contain OEM →
Tier 1 → Tier 2, while others extend to OEM → Tier 1 → Tier 2 → Tier 3 → Tier 4 → Tier 5+.

Importantly, chain length and disruption attenuation are **independent** mechanisms. A long chain
may exhibit strong attenuation, while a short chain may transmit disruption almost unchanged. This
independence must be checked numerically (near-zero correlation between chain length and realised
attenuation) during generation.

**Purpose:** evaluate whether architectures can adapt reasoning depth according to topology
rather than assuming a globally optimal propagation distance.

---

## Configuration Table

Every benchmark mechanism must expose reproducible generation parameters. Every parameter must
have a name, a numeric default, and a valid range — no parameter may be left as a qualitative
label.

**[V2 clarification — replaces the two unusable rows in the original draft with numeric
parameters]**

| Mechanism | Parameter | Default | Range |
|---|---|---|---|
| World scale | `SUP_N` — simulated supplier population | 4,000 | 800–10,000 |
| World scale | `SNAPSHOTS` — monthly `t0` count | 40 | 6–60 |
| World scale | `T_START` — simulation start | 2024-01-01 | any date ≤ `FIRST_T0 − WARMUP_DAYS` |
| World scale | `WARMUP_DAYS` — `T_START` → first `t0` | 182 | 180–365 |
| World scale | `HORIZON_DAYS` — label horizon per snapshot | 14 | 7–28 |
| World scale | `SETTLE_DAYS` — last `t0` → `T_END`, beyond the horizon | 44 | 14–90 |
| World scale | `SHORTAGE_SAMPLE_RATE` — fraction of `inv_pairs` carrying shortage labels | 0.08 | 0.01–1.00 |
| World scale | `DELAY_SAMPLE_RATE` — fraction of eligible shipments carrying delay labels | 1.00 | 0.01–1.00 |
| Visibility (A) | Maximum Visible Tier | 2 | 1–5 |
| Hidden Structure (B) | Hidden Parent Rate | 0.20 | 0–0.50 |
| Hidden Structure (B) | Type A : B : C Mix | 1 : 1 : 1 | any 3-way split summing to 1 |
| Dynamic Relationships (C) | Edge Rewire Probability | 0.10 | 0–0.50 |
| Coupling (D) | α | 0.35 | 0–1.00 |
| Hidden Resilience (E) | Mean Resilience | 0.50 | 0–1.00 |
| Hidden Resilience (E) | Resilience Std Dev | 0.15 | 0.05–0.30 |
| Hidden Resilience (E) | `resilience_lambda` — coupling into observable outcome | 1.3 | 0.0–3.0 |
| Risk Transmission (F) | Attenuation Coefficient — High Resilience | 0.15 | 0.05–0.30 |
| Risk Transmission (F) | Attenuation Coefficient — Medium Resilience | 0.55 | 0.40–0.70 |
| Risk Transmission (F) | Attenuation Coefficient — Low Resilience | 0.90 | 0.75–0.98 |
| Information Delay (G) | Delay Distribution (mean) | 2 weeks | 0–8 weeks |
| Information Delay (G) | Delay Distribution (shape) | log-normal, σ=0.5 | configurable |
| External Shocks (H) | Shock Arrival Rate | 12 / year | 0–52 / year |
| External Shocks (H) | Blast Radius (suppliers per shock) | 5 | 1–50, drawn from shared-infrastructure grouping, not uniform-random |
| Multi-Source (I) | Dependency Count | 3 | 2–6 |
| Multi-Source (I) | AND vs OR Mix | 70% AND / 30% OR | configurable |
| Chain Length (J) | Tier Depth | 3 (mode) | 2–6, drawn independent of attenuation regime |

Every published experiment must report its full resolved benchmark configuration (all rows
above, as generated), not just which mechanisms were enabled. The generator emits this as JSON
alongside each variant's CSVs.

**Timeline derivation.** `T_START` and `T_END` are not free parameters — only `T_START`,
`WARMUP_DAYS`, `SNAPSHOTS`, `HORIZON_DAYS` and `SETTLE_DAYS` are, and the rest follow:

```
FIRST_T0 = T_START + WARMUP_DAYS          = 2024-01-01 + 182d = 2024-07-01
LAST_T0  = FIRST_T0 + (SNAPSHOTS - 1) months               = 2027-10-01
T_END    = LAST_T0 + HORIZON_DAYS + SETTLE_DAYS            = 2027-11-28
```

At the defaults that yields `t0` = **Jul 1 2024 … Oct 1 2027**, matching the split table above and
`docs/phase0_power_check.md`. `WARMUP_DAYS = 182` exists so the *first* `t0` already has its full
180-day trailing feature history, per the backfill-honesty rule; `SETTLE_DAYS` gives the last
snapshot's in-flight shipments room to resolve past their label horizon. **`T_START` is the
simulation start, not the first snapshot** — the six months between them are warm-up that the model
never sees a snapshot for, which is why the two dates differ from the `t0` range everywhere else.

Attenuation Coefficient values above represent the fraction of upstream stress transmitted per
hop; a supplier's realized coefficient is interpolated from its continuous hidden `resilience`
value against these three anchor points, not selected from three discrete buckets.

**`SUP_N` population definition [resolved in Phase 0].** `SUP_N` counts the **full simulated
supplier population**, including tier-3+ suppliers that Mechanism A renders invisible to the model.
It is *not* the visible, label-bearing subset. Two consequences that must be respected wherever
`SUP_N` is used:

1. **Label volume scales with the visible subset, not with `SUP_N`.** Impact is one label per
   supplier per snapshot and only visible suppliers carry usable labels, so a variant with
   Mechanism A enabled yields fewer impact positives than one without at the same `SUP_N`. Power
   targets must be computed against the visible population.
2. **Every Phase 0 measurement uses this definition**, and V1's flat topology makes the two
   readings numerically identical there (no tiers ⇒ visible = total), so the co-degradation sweep
   and the label-volume power calculation in `docs/phase0_power_check.md` are on the same footing.
   Once Mechanism J introduces tiers, they diverge and must be reported separately.

**Why two sampling rates exist.** The three tasks scale on different denominators — impact is one
label per supplier per snapshot, delay one per eligible shipment, shortage one per (product,
warehouse, snapshot) — so no single world size puts all three in range at once, and the two large
denominators have to be sampled down while impact is scaled up.

Both sampled sets must be drawn from a fixed seed and held **identical across all variants and all
seeds**, so cross-variant comparison is never confounded by which shipments or pairs were sampled.
Sample the *entities* (`inv_pairs`, eligible shipments), not the label rows, so each sampled
entity keeps its full time series.

**[Amended in Phase 6 — both rates re-derived at spec scale against the real generator.]** The
Phase 0 §4 figures these rates were originally set from (Variant K: impact 2,049, delay 6,266,
shortage 46,966) used the `ABSORB=0.533` approximation, before Mechanisms E and F were built and
before G existed. Measured with all three live, Variant K is **impact 1,518, delay 3,642, shortage
35,169** — 26%, 42% and 25% below the projections. Consequences, in full in
`docs/phase6_spec_scale_report.md`:

- `SHORTAGE_SAMPLE_RATE` moves from 0.07 to **0.08.** Shortage density spans only 1.42× across
  variants, so a single global rate does exist — but 0.07 was derived from one seed, and across all
  12 × 5 runs it puts Variant K at **1,891 on seed 46, under the floor**. The 60-run feasible
  window is [0.074, 0.095]; 0.08 clears the floor by 8% on the worst measured seed and the ceiling
  by 16% on the best.
- `DELAY_SAMPLE_RATE` changes to **1.00**. Delay density spans **9.0×** across variants, and no
  single rate satisfies both ends: the 5,000 ceiling binds at ≤ 0.152 (Variants A, J) while the
  2,000 floor binds at ≥ 0.549 (Variant K). The floor is the constraint with statistical meaning
  and the ceiling is a cost bound worth ~10 MB per variant-seed, so the floor wins. Eleven of
  twelve variants therefore exceed 5,000 delay positives, by up to 6.6×; this is recorded as a
  deliberate ceiling violation, not an oversight.
- **Impact misses the floor on the combined benchmark.** Variant K averages **1,896** over five
  seeds (in band on 3 of 5) and Variant F 2,256 (4 of 5); impact has no sampling lever. Closing it
  needs `SNAPSHOTS` ≈ 53 — measured, since post-2025 snapshots carry 90% of the in-calendar
  positive density — or a larger `SUP_N`, which `docs/phase0_power_check.md` §1 rules out. An open
  configuration decision, not a settled default. Impact also *overshoots* on Variants A and J
  (5,412), so it is out of band at both ends of the variant set.
- **Read power flags across seeds, never from one run.** Delay positives vary ~2× and impact ~40%
  between seeds, because the hidden-factor member sets are re-drawn per seed over a power-law
  degree distribution. This is the same conclusion `docs/phase0_power_check.md` §2–§3 reach for the
  gating checks, now measured for label volume too.


### Mechanism E coupling — `resilience_lambda` **[added in Phase 2]**

Resilience enters the observable world through
`p_delay = 0.025 + 0.38 · stress · max(0, 1 − resilience_lambda · resilience)`.

The default of **1.3** was selected empirically in `docs/phase2_coverage_recheck.md` §9, not
chosen a priori. It is the value that makes resilience recoverable from observable history — AUC
0.599 [0.577, 0.619] — **without distorting the world**: mean supplier stress stays at 0.131,
identical to the spec default at every value of the coefficient. The alternative route to comparable
recoverability, escalating Mechanism H's shock regime, reached AUC 0.622 only by driving mean stress
to 0.598, and was rejected as a different world rather than a tuning change.

Saturation is the constraint on raising it further: the multiplier clamps at 0, so suppliers with
`resilience_lambda · resilience ≥ 1` become perfectly immune and mutually indistinguishable —
3.0% of the population at 1.3, 20.3% at 1.6, and 49.7% at 2.0, where measured AUC turns *down*.

---

## Benchmark Variants

| Variant | Composition |
|---|---|
| **0** | Base dataset |
| **A** | Base + Mechanism J + Mechanism A (Partial Visibility — requires J's multi-tier topology) |
| **B** | Base + Mechanism B (Hidden Shared Structure — Types A/B/C, α = 0 for all) |
| **C** | Base + Mechanism C (Dynamic Supplier Relationships) |
| **D** | Base + Mechanism B + Mechanism D (Hidden Dependency Coupling — requires B's Type A groups) |
| **E** | Base + Mechanism E (Hidden Supplier Resilience) |
| **F** | Base + Mechanism E + Mechanism F (Adaptive Risk Transmission — requires E's resilience state) |
| **G** | Base + Mechanism G (Information Delay) |
| **H** | Base + Mechanism H (External Shock Events) |
| **I** | Base + Mechanism I (Multi-Source Dependencies) |
| **J** | Base + Mechanism J (Chain-Length Heterogeneity) |
| **K** | Combined — all mechanisms enabled simultaneously |

**[V2 clarification — three dependency pairs, not two]** Variants A, D and F are not
single-mechanism additions; each depends on a prerequisite mechanism:

| Variant | Composition | Prerequisite exists because |
|---|---|---|
| **D** | `B + D` | coupling needs B's Type A groups to couple through |
| **F** | `E + F` | transmission attenuation is derived from E's resilience state |
| **A** | `J + A` | truncation needs upstream tiers to truncate |

`A = J + A` was added after Phase 0. V1's topology is Supplier → Component → Product → Order →
Customer — a **single** supplier tier (`grep -n tier db/generate_dataset.py db/schema.sql` returns
only `customer_priority_tier`, a customer segment; `db/supplier_dyadic_risk_schema.sql` records that
the V1 supplier tier amendment was deliberately not built). Mechanism A's `Maximum Visible Tier`
therefore has nothing to truncate against the base topology, and Mechanism J is what creates
upstream supplier tiers. Composing `A = J + A` keeps **Variant 0 identical to V1**, which every
cross-variant comparison is anchored against; adding tiers to the base instead would break that
anchor.

This must be stated wherever variant results are reported, so that D's effect is not misattributed
to coupling alone when it also carries B's hidden-structure effect, and likewise for F/E and A/J.

**Paired comparison for A.** Because Variant J is `Base + J` and Variant A is `Base + J + A`, the
pair (J, A) isolates truncation exactly — J held constant, visibility the only difference. Variant A
results must be reported as a delta against **Variant J**, never against Variant 0, which would
confound truncation with chain-length heterogeneity.

---

## Benchmark Evaluation Protocol

Every submission must evaluate using:

- identical temporal train/validation/test splits
- identical benchmark configuration
- five independent random seeds
- paired bootstrap confidence intervals
- sign consistency across seeds
- mean and standard deviation

Architectural improvements are considered significant only when they remain statistically
distinguishable from confidence intervals and maintain directional consistency across seeds.

### Variant K interpretation rule **[added after Phase 0]**

Any *"does discovering hidden structure help"* result reported from **Variant K must be interpreted
as a delta against Variant D**, never read in isolation.

Phase 0 measured that absorption (Mechanisms E/F) suppresses the observable co-degradation signal
that Mechanism B's discoverability depends on, and that the suppression does **not** diminish with
scale (`docs/phase0_power_check.md` §3). Variant K runs B, D, E and F simultaneously. A null result
in K therefore does **not** license the conclusion that coupling does not help — the signal may have
been absorbed before K could observe it. Variant D (`B + D`, absorption-free) is the clean baseline
for that specific claim, and K measures how much of D's effect survives absorption.

The same reasoning applies to the other prerequisite pairs: report A against J, and F against E.
Variant 0 is the anchor for the benchmark as a whole, not for any individual mechanism that has a
prerequisite.

### Resilience recoverability has no fixed sign **[added in Phase 2]**

When evaluating anything that probes hidden resilience, note that its observable signature is
**direct in Variant E (AUC 0.561) and inverted in Variant F (AUC 0.396)** — see Mechanism F above.
Evaluation harnesses must not assume a direction, and E/F results must not be pooled or compared as
if they measured the same directional effect. Report `|AUC − 0.5|` with the direction stated
alongside it.

### Temporal split definition **[rescaled after Phase 0]**

The original spec requires "identical temporal splits" without defining them. Phase 0 set the
timeline to **40 monthly snapshots**, `t0` = Jul 1 2024 … Oct 1 2027, 14-day label horizon per
snapshot. The V1 60/20/20 proportions are preserved:

| Split | Snapshot range (`t0`) | Count |
|---|---|---|
| Train | Jul 2024 – Jun 2026 | 24 (snapshots 1–24) |
| Validation | Jul 2026 – Feb 2027 | 8 (snapshots 25–32) |
| Test | Mar 2027 – Oct 2027 | 8 (snapshots 33–40) |

These cutoffs must be identical across all twelve variants so cross-variant comparisons are
valid — a variant may change *what* happens inside a snapshot window, never *which* snapshots
fall in which split.

**The disruption event calendar must cover all 40 months.** V1's `EVENTS` list in
`db/generate_dataset.py` ends Sep 2025; Phase 0 measured that snapshots past the calendar produce
only ~55% of a calibrated month's positives, because only the per-supplier `IDIO` events still
fire. Extending the timeline without extending `EVENTS` silently under-powers the validation and
test splits specifically — which is where it does the most damage.

### Baseline specification **[drafted — confirm after Phase 0/6]**

The original spec lists "HADES" as a mandatory baseline without specifying which variant. Every
submission must report against:

| Baseline | Notes |
|---|---|
| GCN | standard config, layer count and hidden dim held constant across all baselines below |
| GraphSAGE | mean aggregator |
| GAT | single-head unless multi-head is the object of study |
| RGCN | relation-specific weights per V1 schema's edge types |
| HGT | V1's persisted comparison anchor per `v1_findings/v2.md` item 7, once built |
| GraphGPS | |
| HADES — SHARE | V1's primary hybrid architecture (`v1_findings/README.md`) |
| HADES — SHARP | V1 variant |
| HADES — SHARK | V1 variant; report per-seed variance explicitly given V1's documented instability on this architecture |

All baselines must be run at the same layer count and hidden dimension within a given experiment,
with any hyperparameter that is *not* the subject of the study held constant across all baselines
and reported in the experiment's configuration output.

---

## Out of Scope for HADES-Bench V2

The following research questions remain intentionally excluded from Version 2 and will be
investigated separately. These are deferred, not omitted, and should not be silently dropped from
future documentation:

- Thin-relation parameter-sharing analysis (SHARK)
- Degree-randomization robustness
- Paired HGT checkpoint comparisons
- Shipment anti-smoothing 2×2 experiments

---

## Philosophy

HADES-Bench is designed to evaluate reasoning rather than memorization. Instead of asking which
architecture achieves the highest score on a single synthetic graph, the benchmark isolates the
individual mechanisms that make real-world supply chains difficult to reason about. Each
benchmark mechanism corresponds to one scientific hypothesis, allowing researchers to determine
not only whether an architecture succeeds, but precisely **why** it succeeds or fails.

---

## Validation Workflow **[added in Phase 2/6]**

Validation runs **inside** `db/generate_dataset.py` and executes on every generation, before any
dataset is usable. The suite exits non-zero on any failure, so a variant that fails validation
cannot silently become a benchmark artifact.

| Mechanism | Validation utility | Statistic |
|---|---|---|
| A — Visibility | truncation percentage; no feature/label row references a hidden supplier; hidden network still transmits | set containment, exact |
| B — Hidden structure | coverage vs `hidden_parent_rate`; type mix; Type A/B/C indistinguishable from topology; Type B co-degrades; Type C inert | permutation null over group membership |
| C — Dynamic relationships | rewire rate vs `edge_rewire_prob`; validity windows close; edge **set** and **count** both move across snapshots | exact counts |
| D — Coupling | Type A stress exceeds own-history stress; coupling reaches **only** Type A | direct measurement of `hp_coupling()` |
| E — Resilience | distribution matches config; never appears in any emitted table; saturation share; recoverable from observed history | held-out AUC + bootstrap CI |
| F — Transmission | attenuation is an exact function of resilience; spans configured regimes; monotone decreasing | exact equality, monotonicity |
| G — Information delay | recorded ≥ true time; mean lag matches config; as-of status uses recorded time; the two clocks demonstrably diverge | distributional + exact |
| H — External shocks | shock count vs `shock_rate_per_year`; blast radius; hits are infrastructure-correlated, not uniform | grouping test vs uniform baseline |
| I — Multi-source | source count; AND/OR mix; no duplicate sources; AND and OR resolve differently | exact counts |
| J — Chain length | depth heterogeneity; depths within range; depth independent of per-hop attenuation | permutation null over attenuation |
| Phase 2 loop | leakage assertions enabled; agent clock never ran ahead | executable assertion |

### Gate-versus-report rule

Several checks are **statistically under-powered at small configurations**. Rather than weaken a
threshold to make it pass, each such check **gates when it has the power to mean something and
reports its numbers otherwise**, printing the sample size it would need:

| Check | Gates when | Reports otherwise because |
|---|---|---|
| Legacy H_POLYMER co-degradation | Variant 0 only | a fixed 4-member statistic; Phase 0 §1 showed it degenerates to n=1–2 at scale |
| Type B / Type C co-degradation | ≥ 20 measurable groups | `sup_n=800` yields 6–9 groups; `sup_n=4,000` yields 26–34 |
| Type B under absorption | Mechanism E absent | absorption compresses the signal ~15% by design; compare Variant K against Variant D |
| Resilience recoverability | ≥ 800 estimable suppliers | CI width scales with the estimable population |

A reported check still prints its effect size and null band. It never silently passes.

### Statistical conventions

Three conventions were adopted after specific measurement errors, and apply to every check:

1. **Test against the correct null, not against a round number.** Cross-validated AUC on a few
   hundred rows is systematically biased *below* 0.50, so indistinguishability checks compare
   against a permutation null (measured at [0.398, 0.570]) rather than against 0.50.
2. **Never pool one-vs-rest scores.** Scores from separately fitted models carry different
   intercepts; multi-class discrimination is reported as a macro average.
3. **Widen per-run bands for repeated gates.** A gate evaluated once per variant per seed runs 60
   times in a full sweep, so a 95% band would be expected to exclude ~3 times by chance. Such gates
   use a 99% band.

---

## Evaluation Workflow **[added in Phase 6]**

`db/benchmark_eval.py` implements the protocol; `db/run_benchmark.py` generates the matrix.

```
generate  →  validate  →  load  →  split  →  train  →  evaluate  →  compare
```

1. **Generate** — `run_benchmark.py` produces the twelve variants across five seeds. Every dataset
   carries `resolved_config.json` recording benchmark version, generation timestamp, seed, variant,
   mechanisms enabled, the fully resolved configuration, and realised label/row counts.
2. **Validate** — the in-generator suite runs automatically and exits non-zero on failure.
3. **Split** — `temporal_splits()` divides snapshots 60/20/20 **positionally**, so cutoffs are
   identical across variants by construction. A variant may change what happens inside a snapshot
   window, never which snapshots fall in which split.
4. **Compare** — `significant()` applies the spec's full criterion: a paired bootstrap CI that
   excludes zero **and** sign consistency ≥ 0.8 across seeds. A delta whose CI excludes zero but
   whose sign flips across seeds is not an improvement.

Reproducibility is verified by regenerating a variant twice and diffing the **compressed** bytes
(`run_benchmark.py --verify-determinism`) — the stronger check, since matching decompressed
contents would not catch a timestamp leaking into a gzip header.
