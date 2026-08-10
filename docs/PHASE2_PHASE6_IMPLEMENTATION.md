# Phase 2 + Phase 6 — Implementation Summary

**Status: all ten mechanisms implemented, all twelve variants generate, 12 variants × 5 seeds = 60
runs with zero validation failures, Variant 0 byte-identical to V1.**

```
total validation failures across all runs: 0
determinism: variant K, 21/21 .csv.gz files byte-identical across two runs
Variant 0: BYTE-IDENTICAL to pre-Phase-1 V1 output
```

---

## 1. Files created

| File | Purpose |
|---|---|
| `db/run_benchmark.py` | Phase 6 generation matrix, report table, determinism verification |
| `db/benchmark_eval.py` | Phase 6 evaluation protocol: splits, paired bootstrap, sign consistency |
| `db/phase2_gate_followup.py` | Phase 2 gate: coefficient sweep, stratified AUC, per-variant ceiling |
| `docs/PHASE2_PHASE6_IMPLEMENTATION.md` | this document |

(`db/phase2_coverage_recheck.py`, `docs/phase0_power_check.md` and
`docs/phase2_coverage_recheck.md` were created in earlier sessions.)

## 2. Files modified

| File | Change |
|---|---|
| `db/generate_dataset.py` | Mechanisms E and F; time-causal agent loop; E/F validation; benchmark metadata; statistical corrections to the B, J and legacy co-degradation checks |
| `db/README.md` | repository structure, variant table, generation/evaluation pipeline, V2 mechanism descriptions, known limitations |
| `docs/00_Benchmark_Specification.md` | `resilience_lambda` config row + rationale; Validation Workflow; Evaluation Workflow; gate-versus-report rule; statistical conventions |
| `docs/phase0_power_check.md` | §8.1 amended with the Phase 5 coverage recheck outcome (earlier session) |
| `docs/phase2_coverage_recheck.md` | §9–§12 appended: coefficient sweep, stratified AUC, ceilings (earlier session) |

## 3. Benchmark mechanisms implemented

All ten. A, B, C, D, G, H, I, J landed in Phases 3–5; **E and F are this session's work.**

- **E — Hidden Supplier Resilience.** One latent per supplier, Beta-distributed to the configured
  mean/std, never emitted. Governs disruption absorption (`p_delay` suppression) and, through the
  agent loop, inventory depletion and mitigation aggressiveness.
- **F — Adaptive Risk Transmission.** Attenuation is a deterministic piecewise-linear interpolation
  of the *same* resilience value against the three configured anchors — not a second latent, not
  random. Applies at the **downstream** node of every inbound edge (co-parent, hidden-parent, and
  upstream chain alike), per Clarification 3.

Two composition fixes were needed and are worth recording: F initially touched only J's upstream
chains, making Variant F label-identical to Variant E; and `upstream_stress` attenuated at the
*upstream* node rather than the downstream one. Both are corrected.

## 4. Configuration parameters added

| Parameter | Default | Range | Description |
|---|---|---|---|
| `resilience_lambda` | 1.3 | 0.0–3.0 | Coupling of hidden resilience into observable outcome |

The full 31-parameter table lives in `Config` in `db/generate_dataset.py` and in the spec's
Configuration Table; every parameter carries a default, an executable range in `RANGES`, and a
description. Out-of-range and unknown parameters abort before generation.

The default of 1.3 is empirical, not chosen a priori — see `docs/phase2_coverage_recheck.md` §9.
It is the value that makes resilience recoverable (AUC 0.599 [0.577, 0.619]) while leaving mean
supplier stress at 0.131, identical to the spec default.

## 5. Validation utilities added

Per-mechanism coverage is tabulated in the spec's Validation Workflow. New in this session:

- **E**: distribution matches config; **never appears in any emitted table** (exact per-supplier
  match across 154,400 emitted cells); saturation share; recoverability from observed history
  (held-out AUC + bootstrap CI, continuous estimator).
- **F**: attenuation is an exact function of resilience (zero deviation from the interpolation);
  spans the configured regimes; strictly monotone decreasing in resilience.
- **Agent loop**: leakage assertions enabled; observation frontier never ran ahead of its clock.

Four existing checks were made statistically correct rather than left to pass by luck:

1. **Legacy H_POLYMER co-degradation** — gates for Variant 0 (the V1 reproduction guarantee);
   reported for V2 variants, where Phase 0 §1 showed the fixed 4-member statistic degenerates to
   n=1–2 under truncation and absorption.
2. **Type B / Type C co-degradation** — was measured as a whole-timeline average, diluting a
   110-day event ~5×. Now measured **inside each group's own event windows**, as a mean effect
   against a permutation null over random member sets.
3. **J depth-vs-attenuation independence** — a fixed `|corr| < 0.15` bar replaced by a permutation
   null, because chains share upstream suppliers and the sampling distribution is not zero-centred.
4. **Recoverability directionality** — the gate is `|AUC − 0.5|` clear of zero, not `AUC > 0.5`.

## 6. Documentation updated

`docs/00_Benchmark_Specification.md` (config table, Validation Workflow, Evaluation Workflow,
gate-versus-report rule, statistical conventions), `db/README.md` (structure, variants, pipeline,
mechanisms, limitations), plus this summary. `docs/phase0_power_check.md` and
`docs/phase2_coverage_recheck.md` carry the empirical results the defaults rest on.

## 7. Remaining TODOs

1. **Run the full matrix at spec scale.** Everything reported here is the `v1` preset (800
   suppliers, 15 snapshots), which runs in ~10 s per variant-seed. The spec configuration
   (`sup_n=4,000`, 40 snapshots) costs ~115 s and ~443 MB per variant-seed — roughly **2 hours and
   ~24 GB** for the full 12 × 5 sweep. Storage exceeds the ~15 GB target; the recommended lever is
   retaining 2 of 5 seeds on disk and regenerating the rest (`docs/phase0_power_check.md` §7).
2. **Re-derive `DELAY_SAMPLE_RATE` and `SHORTAGE_SAMPLE_RATE` at spec scale.** Both were set from
   Phase 0 measurements taken *before* Mechanism G existed. G suppresses delay positives ~3×, so
   0.50 is very likely wrong for any variant including G.
3. **Neither sampling rate is wired into label emission yet.** They are configured, validated and
   documented, but `generate_dataset.py` does not yet subsample. Every variant currently emits all
   labels — which is why the report table shows shortage `ok` and delay `LOW` rather than both in
   band.
4. **Power targets are not met at the `v1` preset** (delay and impact both `LOW` for every variant).
   Expected: that preset is the V1 fixture, not a benchmark configuration.
5. **`supplier_upstream` has no DDL and is not loaded.** Mechanism J emits
   `supplier_upstream.csv.gz` (its upstream edge list), but `db/schema.sql` has no matching table
   and `load_data.py`'s `LOAD_ORDER` does not include it, so the file is silently skipped on load.
   The CSV is correct and Mechanism J's in-simulator behaviour is unaffected; only the PostgreSQL
   path is incomplete. Needs a `CREATE TABLE` plus a `LOAD_ORDER` entry.
6. **Docs 01–14 are not yet rewritten.** `V2_MASTER_PROMPT.md` Part 2 specifies a per-document
   treatment for the fourteen numbered docs, including archiving several under `docs/v1_archive/`
   and writing `docs/CHANGELOG_V1_to_V2.md`. That is a separate session's work and the prompt asks
   for confirmation before archiving anything.

## 8. Potential implementation risks

1. **Variant K's Type B signal is below detectability.** Measured at spec scale: Variant D (no
   absorption) clears at −0.0428 against a null low of −0.0332; Variant K sits at −0.0366 against
   −0.0389. Absorption removes ~15% of the signal. This is the interaction Phase 0 §3b predicted and
   is exactly why the spec requires Variant K to be read against Variant D. **A null result on
   hidden-structure discovery in Variant K is uninterpretable in isolation.**
2. **Resilience recoverability inverts under Mechanism F.** In Variant E the signature is direct
   (AUC 0.561); in Variant F it is inverted (0.396). A resilient supplier both absorbs its own
   stress *and* receives less transmitted stress, which flips the sign of the observable
   relationship. Information is preserved — but any baseline that assumes a fixed direction will
   read Variant F as noise.
3. **Several checks report rather than gate at small configurations.** The thresholds are stated in
   the output and in the spec, but a reader skimming for `[PASS]` could mistake a reported check for
   a passed one. Every reported check prints its effect size and null band; none silently passes.
4. **Mechanism A improves rather than degrades recoverable coverage.** It removes precisely the
   zero-shipper nodes that could never be estimated, so Variant F (no A) is the *harder* case for
   resilience, not Variant K. Easy to get backwards when interpreting results.
5. **Truncation broke referential integrity, and the narrow check missed it.** Mechanism A
   truncates `suppliers.csv`, but `component_suppliers` and `risk_scores` still emitted rows
   pointing at hidden suppliers — a live FK violation that `load_data.py` caught only when Variant K
   was actually loaded into PostgreSQL. Fixed, and the Mechanism A check now verifies **every**
   emitted supplier reference rather than just features and labels. The lesson generalises: a
   validation suite that reasons over in-memory structures can miss what the emitted artifact
   actually contains, so the loader is a necessary second gate, not a formality.
6. **The `v1` preset is load-bearing for regression.** Variant 0 at that preset is the only
   byte-identity guarantee against V1. Changing it silently breaks the anchor that every
   cross-variant comparison depends on.

## 9. Suggested next phase

**Run the spec-scale sweep and wire the label sampling, in that order.** Concretely:

1. Wire `DELAY_SAMPLE_RATE` / `SHORTAGE_SAMPLE_RATE` into label emission (TODO 3), sampling
   *entities* rather than label rows so each sampled entity keeps its full time series, with the
   sampled set held identical across variants and seeds.
2. Re-derive both rates from a spec-scale Variant K run, since Mechanism G changed the delay
   denominator after those defaults were set.
3. Execute the full 12 × 5 sweep at spec scale with a seed-retention policy, and publish the report
   table with power flags per variant per task.
4. Then proceed to `V2_MASTER_PROMPT.md` Part 2 — the fourteen numbered docs — which should be
   written against measured spec-scale numbers rather than the `v1` fixture's.
