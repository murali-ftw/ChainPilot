# Phase 2 Gate — Resilience-Coverage Recheck with Real Mechanism H

**Verdict: the gate FAILS. Coverage is 5.4%, against a >50% requirement.**

Mechanism H at its spec defaults makes **no measurable difference** to resilience coverage — 5.4%
with H versus 5.6% without it. The sequencing note in `V2_MASTER_PROMPT.md` anticipated this and
offered two remedies: raise `Shock Arrival Rate`, or diagnose the shipment-cadence bottleneck. This
recheck did both. Raising the shock rate **cannot** clear the gate at any parameterisation. The
bottleneck is real and is diagnosed below, and it points somewhere neither remedy anticipated: the
estimator, not the world.

Configuration: `SUP_N = 4,000`, 40 snapshots, `shock_rate_per_year = 12`, `blast_radius = 5` — the
recommended configuration from `docs/phase0_power_check.md` §6. Reproduce with
`python3 db/phase2_coverage_recheck.py`.

**Per instruction, Mechanisms E/F were not built in this session regardless of outcome.**

---

## 1. The headline numbers

| | shipments | shocks | suppliers hit | estimable | coverage | gate |
|---|---|---|---|---|---|---|
| Variant 0 (no H) | 437,257 | 0 | 0 | 223 / 4,000 | **5.6%** | FAIL |
| Variant H (real H) | 437,246 | 47 | 224 (5.6%) | 217 / 4,000 | **5.4%** | FAIL |

The 0.2-point difference is RNG drift, not signal. H changes nothing.

"Estimable" uses `docs/phase0_power_check.md` §3's definition unchanged: a supplier needs **≥5
shipments dispatched at stress ≥ 0.35 and ≥5 below it**, so that its disruption-response behaviour
is observable at all. Coverage is a property of the generated world — it does not depend on how
resilience is assigned, which is what makes it usable as a gate.

## 2. Why H does nothing

At defaults, 47 shocks × blast radius 5 = **224 distinct suppliers touched, 5.6% of the
population.** Blast radius 5 was specified against no particular world size; against 4,000
suppliers it is a rounding error. The arrival *rate* was never the binding parameter.

But the deeper constraint is cadence, and it is visible in one number:

```
high-stress shipments per supplier: median 0, p90 4, need >= 5
```

**The 90th-percentile supplier still falls short of the threshold.** Each shipping supplier averages
152 shipments across 1,427 days ≈ 0.107/day. A 25–60 day shock window therefore yields 2.7–6.4
shipments, and a supplier needs ≥5 *inside stress windows* while also retaining ≥5 outside them.
Most suppliers never accumulate enough stressed shipments to be observed under stress at all.

## 3. Raising the shock rate does not fix it — and eventually makes it worse

Sweeping H's parameters on the generated world:

| shock_rate/yr | blast_radius | duration (d) | suppliers hit | coverage |
|---|---|---|---|---|
| 12 | 5 | 25–60 | 224 | 5.4% |
| 12 | 50 | 25–60 | 1,697 | 7.1% |
| 52 | 50 | 25–60 | 3,478 | 13.1% |
| 52 | 200 | 25–60 | 3,986 | **29.6%** |
| 52 | 400 | 60–180 | 4,000 | 27.6% |
| 104 | 400 | 60–180 | 4,000 | 10.9% |

**Coverage is non-monotonic in shock intensity, and the peak is 29.6% — well short of 50%.** This is
the most important structural finding here. The estimability criterion needs evidence on *both*
sides of the stress threshold. Too few shocks and there is no high-stress evidence; too many and
every supplier is stressed all the time, so the low-stress contrast disappears. At 104 shocks/year
with radius 400, coverage collapses to 10.9% despite every supplier being hit.

There is no shock configuration that clears 50%. Raising `Shock Arrival Rate`, the sequencing note's
first remedy, is not available as a fix.

## 4. A hard ceiling that no shock rate can lift

```
1,131 of 4,000 suppliers (28.3%) ship NOTHING and can never be estimable at any shock rate.
Maximum achievable coverage is therefore 71.7%.
```

These are the component-less suppliers — the same population Mechanism J promotes to tiers 2+ and
Mechanism A would hide. A supplier with no shipments has no `(disruption → outcome)` history by
construction, so resilience is unrecoverable for it no matter what.

> **This number was measured on Variants 0 and H, neither of which includes Mechanism A** — the
> sentence above describes who those suppliers are, not a mechanism that was applied. §11 recomputes
> the ceiling per variant and finds it differs sharply once A is present: 71.7% for Variant F,
> **86.3%** for Variant K. Do not carry "71.7%" forward as a single global number.

## 5. The actual bottleneck is the estimator, not the world

The 5–9% figure is substantially an artifact of §3's prototype, which **bins** stress into high/low
and demands ≥5 observations per bin. A continuous estimator — per-supplier regression of on-time
outcome against dispatch-time stress, using every shipment — has no binning requirement. Both run
on the same world, with the same hidden resilience:

| estimator | coverage | AUC | 95% CI | |
|---|---|---|---|---|
| binned (≥5 each side) | 5.4% (217 suppliers) | 0.498 | [0.424, 0.567] | **spans chance** |
| continuous (all shipments) | **58.4%** (2,336 suppliers) | 0.536 | [0.511, 0.558] | above chance |

Switching estimator moves coverage from 5.4% to **58.4%** — clearing the 50% bar — and is the only
change tested here that does. It also recovers 2,336 suppliers out of the 2,869 that ship at all,
i.e. 81% of the reachable population.

**This does not mean the gate is cleared, and it should not be read that way.** The continuous
estimator clears the *coverage* bar while revealing the more serious problem: recoverability is
**0.536**, barely above chance. Clarification 3 requires resilience to be recoverable
*meaningfully* above chance, and an AUC of 0.536 is not meaningfully anything. The binned estimator
was hiding a weak signal behind a coverage problem; removing the coverage problem exposes it.

## 6. What would make Phase 2 viable

Signal strength tracks how much stress each supplier actually experiences, because resilience only
manifests *under* stress. Holding the estimator continuous and varying the shock regime:

| shock regime | mean stress | AUC | 95% CI |
|---|---|---|---|
| 12/yr, radius 5 (default) | 0.128 | 0.555 | [0.530, 0.576] |
| 52/yr, radius 200 | 0.235 | 0.574 | [0.549, 0.597] |
| 52/yr, radius 400, 60–180d | 0.598 | **0.622** | [0.601, 0.646] |

*(Shocks are swapped onto the already-generated shipment set, so this ignores feedback from shocks
onto shipment cadence. Indicative, not a substitute for a full regeneration.)*

Recoverability rises with mean stress, and reaches 0.622 only under a regime — 52 shocks/year,
blast radius 400, 60–180 day durations — that is **~4× the arrival rate, 80× the blast radius, and
3× the duration of H's spec defaults**. That is not a parameter tweak; it is a different world, and
one where the mean supplier sits at stress 0.598 more or less permanently. Whether a benchmark that
disrupted is still the benchmark anyone wants is a design question, not a tuning question.

## 7. Recommendation

Do not build Mechanism F on the current configuration. Three options, in the order I would take
them:

1. **Replace the estimability criterion before doing anything else.** The binned ≥5/≥5 rule is not
   in the spec — it was my own prototype convenience in Phase 0, and it has been setting the agenda
   ever since. Adopt the continuous estimator as the definition of the Clarification 3 check, and
   restate the coverage target against the 71.7% ceiling. This alone takes coverage from 5.4% to
   58.4% and costs nothing.
2. **Then confront the real question: is AUC 0.536 good enough?** It is not, on my reading of
   Clarification 3. Either strengthen the coupling between resilience and observable outcome (the
   prototype uses `p_delay = 0.025 + 0.38·stress·(1 − 0.7·resilience)`; raising that 0.7 makes
   resilience bite harder without touching the shock regime), or accept a heavier shock regime, or
   both. This is a spec decision about how discoverable the latent should be, and it belongs to you.
3. **Re-run this check after whichever change you choose, with a full regeneration** rather than the
   swapped-shock approximation in §6, before Phase 2 starts.

The Phase 0 finding that motivated this gate stands, but its cause was misattributed. It was never
mainly about disruption density — it was the estimator, plus a structural ceiling nobody had
measured, plus a genuinely weak latent signal that the coverage problem was masking.

## 8. Status of the sequencing note's condition

`V2_MASTER_PROMPT.md` says: *"Do not start Phase 2 until the H-coverage recheck confirms >50%
coverage at the recommended scale."*

- With the criterion as written in Phase 0 §3: **5.4% — not confirmed, and unreachable via H.**
- With a continuous estimator: **58.4% — the coverage condition is met**, but the recoverability it
  exposes (AUC 0.536) does not satisfy Clarification 3's "meaningfully above chance".

Phase 2 remains blocked pending your decision on §7.2. Mechanisms E/F were not built.

---

# Follow-up — Coefficient Sweep, Stratified AUC, Per-Variant Ceiling

Three items flagged as untested above. All three run against the same world
(`SUP_N = 4,000`, 40 snapshots, Mechanism H at spec defaults). Reproduce with
`python3 db/phase2_gate_followup.py`. Still disposable prototype code — E/F are not built.

**Headline: there is a non-distorting lever, and it was never Mechanism H.** Raising the
resilience→outcome coupling coefficient from 0.7 to 1.3 takes recoverability from AUC 0.547 to
0.599 **without changing the world's stress profile at all**.

## 9. Coefficient sweep — Mechanism H held at spec defaults

| λ | coverage | AUC | 95% CI | mean stress | median stress | saturated |
|---|---|---|---|---|---|---|
| 0.7 (current) | 58.4% | 0.547 | [0.524, 0.569] | 0.131 | 0.121 | 0.0% |
| 1.0 | 58.4% | 0.572 | [0.550, 0.591] | 0.131 | 0.121 | 0.0% |
| **1.3** | 58.4% | **0.599** | **[0.577, 0.619]** | **0.131** | **0.121** | **3.0%** |
| 1.6 | 58.4% | 0.610 | [0.586, 0.632] | 0.131 | 0.121 | 20.3% |
| 2.0 | 58.4% | 0.584 | [0.562, 0.608] | 0.131 | 0.121 | 49.7% |

**Mean stress is 0.131 at every value.** This is the entire point of the contrast with the
H-escalation path in §6: that route bought AUC 0.622 by driving mean stress from 0.128 to 0.598 —
permanent ambient crisis. The coefficient buys nearly the same recoverability while leaving the
world's disruption profile *bit-for-bit* as the spec intends. It changes how hard resilience bites
on an outcome, not how often outcomes are stressed.

"Saturated" is the fraction of suppliers with `λ·resilience ≥ 1`, whose delay multiplier
`max(0, 1 − λ·res)` clamps to zero — perfectly immune, and therefore mutually indistinguishable.
That is why AUC turns down at λ = 2.0 despite the stronger coupling: half the population collapses
into one indistinguishable block. **λ = 1.6 is the arithmetic peak but costs 20.3% saturation;
λ = 1.3 gets 98% of the AUC for 3.0%.**

**Answer to the question posed: yes.** λ = 1.3 reaches AUC 0.599 with a CI of [0.577, 0.619] — clear
of 0.50 by more than three CI half-widths, not a boundary case — with no distortion of mean stress.
Note also that the 5-fold CV estimator carries the systematic *negative* bias documented in
`docs/phase0_power_check.md` §2, so these are conservative.

## 10. AUC stratified by shipment volume

| quartile | n | shipments | AUC @ λ=0.7 | 95% CI | AUC @ λ=1.3 | 95% CI |
|---|---|---|---|---|---|---|
| Q1 | 584 | 8–14 | 0.472 | [0.425, 0.525] *spans* | 0.514 | [0.460, 0.567] *spans* |
| Q2 | 584 | 14–25 | 0.519 | [0.471, 0.564] *spans* | 0.551 | [0.502, 0.599] |
| Q3 | 584 | 25–49 | 0.473 | [0.429, 0.521] *spans* | 0.592 | [0.544, 0.636] |
| Q4 | 584 | 49–37,281 | 0.601 | [0.555, 0.638] | **0.709** | [0.665, 0.747] |
| pooled | 2,336 | 8–37,281 | 0.547 | [0.524, 0.569] | 0.599 | [0.577, 0.619] |

**At λ = 0.7 the hypothesis holds exactly: the signal lives entirely in Q4.** Q1–Q3 all span chance;
the pooled 0.547 was a strong top-quartile signal diluted by three quartiles of noise. (Q4's
49–37,281 range reflects the Pareto degree distribution — a handful of dominant suppliers carry
enormous volume.)

**But λ = 1.3 does not merely amplify Q4 — it lifts Q2 and Q3 clear of chance too.** That is the
distinction that decides the fix, and it means the two levers are complementary rather than
alternatives. At λ = 1.3, resilience is recoverable for Q2–Q4 = **1,752 suppliers**, 75% of the
estimable population, 61% of all shippers, 43.8% of the simulated population. Only Q1 — suppliers
with 8–14 shipments — remains at chance.

**Does the `fulfilment_preference_weight` precedent apply?** *Partially, and only to the residue.*
`v1_findings/v2.md` item 6 accepted a signal computable for 6.7% of pairs, reported honestly rather
than forced to full coverage. That precedent is the right treatment for **Q1 at λ = 1.3** — one
quartile, 14.6% of the population, where evidence is genuinely too thin. It is *not* the right frame
for the signal as a whole, because at λ = 1.3 three quartiles clear chance. Reporting resilience as
a 14.6%-coverage curiosity would understate it as badly as claiming 58.4% uniform recoverability
would overstate it. The correct treatment is a **confidence-weighted signal whose confidence tracks
shipment volume**, with Q1 explicitly flagged as below the evidence floor.

## 11. Zero-shipper ceiling, per variant

Neither Variant F nor Variant K can be generated until Phase 2 builds E/F. But **E and F alter
transmission and hidden state only — which suppliers ship is fixed by BOM structure (`prod_bom_sup`),
which E/F never touch.** So Variant 0 is an exact proxy for Variant F's shipper set, and Variant A
(`= J + A`, buildable today) is an exact proxy for Variant K's visibility structure.

| | simulated | emitted | shipping & visible | ceiling / simulated | ceiling / emitted |
|---|---|---|---|---|---|
| **Variant F** (via Variant 0) | 4,000 | 4,000 | 2,869 | 71.7% | **71.7%** |
| **Variant K** (via Variant A) | 4,000 | 3,323 | 2,868 | 71.7% | **86.3%** |

**Correcting the record: Mechanism A was *not* active when 71.7% was first measured.** §4 above
derived it from Variants 0 and H, neither of which includes A; the accompanying prose describing
those suppliers as "the population J promotes and A hides" was an aside about who they are, not a
statement that A was applied. The number itself is A-free and stands.

The two variants differ in what the residue *means*, which is the substantive point:

- **Variant F (no A):** all 1,131 zero-shippers are emitted. They appear in `suppliers.csv`, carry
  impact labels, and are scored — but resilience is unrecoverable for every one of them. They are
  **visible-but-unrecoverable** nodes, and a model evaluated on Variant F is asked about them
  anyway. The ceiling against what the model sees is 71.7%.
- **Variant K (with A):** 677 of them sit above `max_visible_tier` and are never emitted. Of 3,323
  emitted suppliers, 2,868 ship, so **86.3%** are potentially recoverable and the visible-but-
  unrecoverable residue falls to 455 (13.7% of emitted).

**These two numbers must not be conflated going forward.** Mechanism A *improves* the recoverable
fraction of the observable graph — it removes precisely the nodes that could never have been
estimated. Variant F is the harder case for resilience, not K.

## 12. Recommendation

**Proceed to Phase 2 with the continuous estimator and coupling coefficient λ = 1.3, reporting
resilience as a confidence-weighted signal.** λ = 1.3 delivers AUC 0.599 [0.577, 0.619] — clear of
chance by more than three CI half-widths, and conservative given the estimator's known negative bias
— while leaving mean supplier stress at 0.131, identical to the spec default. That is the property
the Mechanism-H escalation path could not offer: it fixes recoverability without rebuilding the
world into permanent crisis. Choose 1.3 over the arithmetic peak at 1.6 because saturation rises
from 3.0% to 20.3% there, and a benchmark in which one supplier in five is perfectly immune to
disruption is both unrealistic and self-defeating (AUC already turns down by λ = 2.0). Retire the
binned ≥5/≥5 estimability rule entirely — it was a Phase 0 convenience, it is not in the spec, and
it was the single largest cause of the original 5–9% finding. Carry two caveats into Phase 2: the
thinnest volume quartile (8–14 shipments, 14.6% of suppliers) stays at chance even at λ = 1.3 and
should be flagged below the evidence floor rather than scored, following the
`fulfilment_preference_weight` precedent; and state the coverage ceiling per variant — 71.7% for
Variant F, 86.3% for Variant K — never as one number.

**Not started in this session:** Mechanisms E and F remain unbuilt, per instruction.
