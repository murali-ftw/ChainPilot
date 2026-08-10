# 14 — Project Roadmap (V2)

> **V2 update.** Rewritten around the V2 build phases. The out-of-scope list is carried forward as
> **deferred work**, not silently dropped.

---

## 1. Build phases — status

Phases ran in a revised order. Phase 2 was **gated** after Phase 0 found resilience recoverable for
only 5–9% of suppliers, so Mechanisms E/F were deliberately built last, after Mechanism H existed to
be tested against.

Executed order: **0 → 1 → 3 → 4 → 5 → H-coverage recheck → 2 → 6.**

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| **0** | Power check; entity/time scale | ✅ complete | `phase0_power_check.md` |
| **1** | Config object, variant manifest, CLI, per-variant output | ✅ complete | Variant 0 byte-identical |
| **3** | Mechanisms B, D | ✅ complete | topology indistinguishability vs permutation null |
| **4** | Mechanisms A, G, J | ✅ complete | depth ⊥ attenuation (+0.010 at spec scale) |
| **5** | Mechanisms C, H, I | ✅ complete | rewire rate, blast-radius correlation, AND/OR |
| **gate** | H-coverage recheck | ✅ complete | `phase2_coverage_recheck.md` |
| **2** | Mechanisms E, F, time-causal agent loop | ✅ complete | recoverability, attenuation determinism |
| **6** | Generation matrix, evaluation protocol, reporting | 🟡 **partial** | matrix + protocol built; **spec-scale sweep not run** |

## 2. What is genuinely done

- All **ten mechanisms** implemented and independently switchable.
- All **twelve variants** generate; 12 × 5 = 60 runs with **zero validation failures**.
- **Variant 0 byte-identical** to V1 through every phase.
- Determinism verified at the compressed-byte level, including across `PYTHONHASHSEED`.
- Evaluation protocol implemented and self-tested.
- Full PostgreSQL load path working for every variant.

## 3. What is not done

**The benchmark is not ready for submissions.** Specifically:

| # | Open item | Blocks |
|---|---|---|
| 1 | Spec-scale sweep (`sup_n=4000`, 40 snapshots) never run — everything so far is the 800-supplier `v1` fixture | any published result |
| 2 | `DELAY_SAMPLE_RATE` / `SHORTAGE_SAMPLE_RATE` not wired into label emission | power targets |
| 3 | Both rates need re-deriving at spec scale — they predate Mechanism G, which suppresses delay positives ~3× | power targets |
| 4 | Power targets met for no task at any scale actually generated | any published result |
| 5 | Storage ~24 GB against a ~15 GB target | full-sweep retention |

Source of record: `PHASE2_PHASE6_IMPLEMENTATION.md` §7.

## 4. Next phase

1. Wire label subsampling into emission (entities, not label rows; sampled set fixed across variants
   and seeds).
2. Re-derive both sampling rates from a spec-scale Variant K run.
3. Execute the full 12 × 5 sweep at spec scale with a seed-retention policy.
4. Publish the report table with power flags per variant per task.
5. Re-run the Clarification 3 recoverability check at spec scale, where ~2,300 suppliers are
   estimable rather than ~390.

## 5. Deferred — out of scope for V2

Deliberately excluded, to be investigated separately. **Deferred, not dropped.**

| Question | Why deferred |
|---|---|
| Thin-relation parameter-sharing analysis (SHARK) | needs a densified-relation variant V2 does not build |
| Degree-randomization robustness | needs supplier degree manipulable independently of every other property |
| Paired HGT checkpoint comparisons | needs an explicit exception to the project's no-checkpoint rule |
| Shipment anti-smoothing 2×2 | needs a synthetic node type with position and features varied independently |

## 6. Known risks carried into the next phase

1. **Variant K's Type B signal is below detectability** (−0.0366 vs null −0.0389 at spec scale,
   against Variant D's −0.0428 vs −0.0332). A null discovery result in K is uninterpretable alone.
2. **Resilience recoverability inverts between Variant E and Variant F.** No probe may assume a
   fixed sign; E and F must never be pooled.
3. **Several checks report rather than gate** at small configurations. A reported check is not a
   passed check.
4. **Mechanism A improves recoverable coverage** rather than degrading it — Variant F is the harder
   case, not K. Ceilings differ: 71.7% (F) vs 86.3% (K).
5. **The in-generator suite cannot see the emitted artifact.** The PostgreSQL load is a necessary
   second gate, not a formality.
6. **The `v1` preset is load-bearing.** Variant 0 at that preset is the only byte-identity guarantee
   against V1; changing it silently breaks the anchor.
