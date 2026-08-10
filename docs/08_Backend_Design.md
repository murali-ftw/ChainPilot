# 08 — Generation and Evaluation Pipeline

> **V2 reframing.** Previously the V1 application backend (services, queues, inference API).
> Reduced to what V2 actually has: a generation pipeline and an evaluation harness. There is no
> server, no API and no runtime service in this repository.

---

## 1. Components

| Component | File | Responsibility |
|---|---|---|
| Generator | `db/generate_dataset.py` | simulation, ten mechanisms, validation suite, CSV + manifest emission |
| Benchmark runner | `db/run_benchmark.py` | variant × seed matrix, report table, determinism verification |
| Evaluation harness | `db/benchmark_eval.py` | temporal splits, paired bootstrap, sign consistency |
| Loader | `db/load_data.py` | schema application, FK-safe COPY, verification queries, rollback |
| Phase harnesses | `db/phase0_*.py`, `db/phase2_*.py` | disposable prototypes that produced the empirical defaults |

Everything is stdlib-only except `load_data.py`, which needs `psycopg2-binary`.

## 2. Generator structure

A single deterministic script, executed top to bottom:

```
config resolution ──▶ validate ranges ──▶ seed RNG
        │
        ▼
 world construction   suppliers → components → hidden structure (B) → tiers (J)
        │             → shocks (H) → multi-sourcing (I) → rewiring (C)
        ▼
 stress model         own_stress()  = base + shared factors + shocks + idiosyncratic
                      stress()      = own + co-parent + hidden-parent (D) + upstream (J)
                                      each inbound path attenuated by recv_atten() under F
        ▼
 simulation           shipments (delay ~ stress × absorption(E))
                      weekly inventory walk + time-causal agent loop
        ▼
 snapshots            features (as-of, recorded clock) → labels (true clock)
        ▼
 emission             gzip CSV + resolved_config.json
        ▼
 validation suite     exits non-zero on failure
```

**Mechanism isolation** is the load-bearing property: each mechanism draws from a dedicated RNG
stream, so enabling it consumes zero draws from the main simulation. That is what allows Variant 0
to stay byte-identical to V1 while ten mechanisms exist in the same file.

## 3. Determinism

| Source of nondeterminism | Mitigation |
|---|---|
| RNG | single seeded stream + dedicated per-mechanism streams |
| Primary keys | UUIDv5 from stable keys, never random |
| Set iteration order | `sorted()` at every set-to-sequence boundary |
| gzip header | `GzipFile(mtime=0, filename="")` |
| Wall clock | no clock reads in the data path; the manifest timestamp is excluded from diffs |

Verified by `run_benchmark.py --verify-determinism`, which diffs the **compressed** bytes across two
runs — the stronger check, since matching decompressed contents would not catch a timestamp leaking
into a gzip header.

## 4. Time-causal agent loop

Each supplier is an agent deciding once per weekly timestep how hard to mitigate, from its
**recorded** delivery outcomes and its own hidden resilience. The leakage constraint is an
**executable assertion**, not a convention:

- `agent_observe(sup_id, as_of)` moves outcomes across a monotonic frontier, asserting each row's
  recorded time is ≤ `as_of`, and rejects a clock that moves backwards.
- `assert_not_future()` raises `TemporalLeak` on violation.
- `LEAK_ASSERTIONS = True` by default; disabling it is a deliberate act.

Decisions feed back into the world through the reorder trigger and replenishment quantity, which is
how Mechanism E reaches inventory depletion as well as delay probability.

## 5. Evaluation harness

`benchmark_eval.py` supplies the protocol, not baselines. No model is trained anywhere in this
repository; baseline implementation is a submission's responsibility.

```python
temporal_splits(t0s)        # positional 60/20/20, identical across variants
paired_bootstrap(deltas)    # percentile CI on a paired delta
sign_consistency(deltas)    # majority-sign fraction across seeds
significant(deltas)         # CI excludes zero AND consistency >= 0.8
```

## 6. Storage

~443 MB gzipped per variant-seed at spec scale; **~24 GB** for a full 12 × 5 sweep, against a
~15 GB target. Recommended lever: retain 2 of 5 seeds on disk and regenerate the rest on demand
(~115 s each, exact by determinism). See `PHASE2_PHASE6_IMPLEMENTATION.md` §7.
