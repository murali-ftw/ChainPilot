# 01 — Benchmark Requirements

> **V2 reframing.** This document was a product requirements document for the V1 ChainPilot/HADES
> application. HADES-Bench V2 is a research benchmark, so the product claims have been replaced by
> the research questions the benchmark exists to answer. The V1 document is preserved in git
> history; nothing accurate was discarded.

**Authoritative spec:** `00_Benchmark_Specification.md`. Where this document and the spec disagree,
the spec wins.

---

## 1. Purpose

HADES-Bench V2 evaluates whether graph learning architectures can *reason* about supply chains, as
opposed to fitting a static graph. Every mechanism in the benchmark exists to test one hypothesis,
is independently switchable, and is isolated by its own dataset variant so that a performance
change is attributable to an identifiable cause.

The benchmark deliberately does not maximise realism through complexity. A mechanism that makes the
task harder without answering a question does not belong in it.

## 2. The seven research questions

The spec defines ten mechanisms, each with a stated purpose. Grouped by the reasoning capability
they probe, they answer seven questions. Each question names the variants that isolate it and the
comparison that must be reported.

| # | Research question | Mechanisms | Isolating comparison |
|---|---|---|---|
| **RQ1** | Can an architecture reason when the observable graph *terminates before the cause*? | A | Variant A vs **Variant J** |
| **RQ2** | Can it discover latent shared structure — and distinguish useful from redundant from decoy structure, without being told which is which? | B | Variant B vs Variant 0 |
| **RQ3** | Does discovering hidden structure yield *incremental predictive information*, or only discoverability? | D | Variant D vs **Variant B** |
| **RQ4** | Can it infer hidden operational state (resilience) and the adaptive transmission behaviour it induces, from observable history alone? | E, F | Variant E vs Variant 0; Variant F vs **Variant E** |
| **RQ5** | Can it reason when observation *lags* truth — labels on true event time, features on reported time? | G | Variant G vs Variant 0 |
| **RQ6** | Is it robust to non-stationary topology and to correlated, infrastructure-clustered failure? | C, H | Variants C and H vs Variant 0 |
| **RQ7** | Can it adapt reasoning depth to topology, and handle deterministic AND/OR redundancy? | I, J | Variants I and J vs Variant 0 |

**RQ1, RQ3 and RQ4 have prerequisite mechanisms** (`A = J + A`, `D = B + D`, `F = E + F`), so their
comparison baseline is *not* Variant 0. Reporting Variant D against Variant 0 attributes B's
hidden-structure effect to coupling. This is a correctness requirement, not a stylistic preference.

## 3. Mechanism-to-hypothesis map

| Mechanism | Hypothesis under test |
|---|---|
| A — Partial visibility | Performance degrades gracefully when upstream causes are unobservable, rather than collapsing |
| B — Hidden shared structure | Latent groups are discoverable from correlated behaviour with no connecting edge |
| C — Dynamic relationships | Predictions survive topology that changes between snapshots |
| D — Hidden dependency coupling | Discovered structure carries information not already in a node's own history |
| E — Hidden resilience | A latent operational state is inferable from `(disruption → outcome)` history |
| F — Adaptive transmission | Optimal propagation depth emerges from inferred behaviour, not graph distance |
| G — Information delay | As-of reasoning is possible when the reporting clock lags the event clock |
| H — External shocks | Correlated multi-supplier failure is distinguishable from independent failure |
| I — Multi-source dependencies | AND and OR redundancy are distinguishable from source count alone |
| J — Chain-length heterogeneity | Reasoning depth adapts per-chain rather than assuming a global optimum |

## 4. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Twelve variants (0, A–K), each generated independently and reproducibly |
| FR-2 | Every mechanism independently switchable via configuration |
| FR-3 | Every parameter carries a name, numeric default, valid range and description |
| FR-4 | Byte-identical regeneration from the same seed and configuration |
| FR-5 | Labels derive from true event time; features from observed time only |
| FR-6 | Hidden state (resilience, hidden parents, truncated tiers) never emitted to any model-visible table |
| FR-7 | Validation executes before any dataset is usable, and fails loudly |
| FR-8 | Each dataset is self-describing: version, seed, mechanisms, resolved config, realised counts |

## 5. Non-functional requirements

| ID | Requirement | Status |
|---|---|---|
| NFR-1 | Generator is stdlib-only, no runtime dependencies | met |
| NFR-2 | Variant 0 reproduces V1 byte-for-byte | met — verified every phase |
| NFR-3 | Full 12 × 5 sweep completes in reasonable wall time | ~2 h at spec scale |
| NFR-4 | Output storage within ~15 GB | **not met** — ~24 GB; see `PHASE2_PHASE6_IMPLEMENTATION.md` §7 |

## 6. Out of scope

Carried forward from the spec, deliberately deferred rather than dropped:

- Thin-relation parameter-sharing analysis (SHARK)
- Degree-randomization robustness
- Paired HGT checkpoint comparisons
- Shipment anti-smoothing 2×2 experiments

## 7. Current status

**The benchmark is implemented but not yet generated at spec scale.** All ten mechanisms and all
twelve variants exist and validate, but the 12 × 5 sweep has only run at the small `v1` fixture
(800 suppliers, 15 snapshots), label subsampling is configured but not wired into emission, and
power targets are met for no task at any scale actually run. See
`PHASE2_PHASE6_IMPLEMENTATION.md` §7 for the full open-items list.
