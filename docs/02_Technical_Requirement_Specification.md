# 02 — Technical Requirement Specification (V2 Generator)

> **V2 reframing.** Previously the technical spec for the V1 application stack. Now the technical
> requirements for the V2 benchmark *generator*: configuration schema, variant manifest,
> determinism, leakage constraints and power targets.

**Authoritative spec:** `00_Benchmark_Specification.md`. Implementation: `db/generate_dataset.py`.

---

## 1. Non-negotiable invariants

| ID | Invariant | How it is enforced |
|---|---|---|
| INV-1 | **Determinism.** Same seed + config ⇒ byte-identical `.csv.gz` | UUIDv5 keys, seeded RNG, `sorted()` over every set iteration, `gzip.GzipFile(mtime=0, filename="")`. Verified by diffing *compressed* bytes |
| INV-2 | **No temporal leakage.** Every value observable at *t* is computable from events at ≤ *t* | Executable assertions in the agent loop (`assert_not_future`, `TemporalLeak`), plus as-of reconstruction of status and features |
| INV-3 | **Stdlib only.** No runtime dependencies | `import csv, gzip, io, json, math, os, random, uuid, argparse, dataclasses` |
| INV-4 | **Validation must pass.** Never weaken a check to make a mechanism pass | Suite runs at the end of every generation and exits non-zero |

INV-4 has a documented corollary added during the build: where a check is statistically
under-powered at a given configuration, it **reports its numbers rather than gating**, printing the
sample size it would need. See the spec's gate-versus-report rule. It never silently passes.

## 2. Configuration schema

A `Config` dataclass holds all 31 parameters; `RANGES` makes the spec's Range column executable.

```
--variant   0 | A..K            which benchmark variant
--seed      int                 RNG seed (default 42)
--config    spec | v1 | <path>  preset name, or JSON file of overrides
--out-dir   <path>              default: db/csv/v<variant>_seed<seed>/
```

Requirements:

- **CFG-1** Every parameter has a name, numeric default, valid range, description.
- **CFG-2** Out-of-range values abort before generation, listing every violation at once.
- **CFG-3** Unknown parameter names abort (catches typos silently ignored otherwise).
- **CFG-4** Cross-field constraints validated (`type_mix` sums to 1; `tier_depth_min ≤ mode ≤ max`).
- **CFG-5** The fully resolved config is emitted as `resolved_config.json` beside the CSVs.
- **CFG-6** `spec` preset = the spec's defaults. `v1` preset = the V1 reproduction fixture.

### Timeline derivation

`T_START` and `T_END` are **derived, not free**:

```
FIRST_T0 = T_START + WARMUP_DAYS
LAST_T0  = FIRST_T0 + (SNAPSHOTS - 1) months
T_END    = LAST_T0 + HORIZON_DAYS + SETTLE_DAYS
```

`WARMUP_DAYS = 182` exists so the *first* snapshot already has its full 180-day trailing feature
history. `T_START` is the simulation start, **not** the first snapshot.

## 3. Variant manifest

```python
VARIANTS = {"0": (), "A": ("J","A"), "B": ("B",), "C": ("C",), "D": ("B","D"),
            "E": ("E",), "F": ("E","F"), "G": ("G",), "H": ("H",), "I": ("I",),
            "J": ("J",), "K": ("A","B","C","D","E","F","G","H","I","J")}
```

- **VAR-1** Three variants carry prerequisites: `A = J + A`, `D = B + D`, `F = E + F`.
- **VAR-2** A variant whose mechanisms are unimplemented is **refused**, not silently emitted as
  Variant 0 under another name.
- **VAR-3** Variant 0 with the `v1` preset must remain byte-identical to the V1 dataset. This is
  the regression anchor every cross-variant comparison depends on.

## 4. Mechanism isolation requirements

- **ISO-1** Every mechanism draws from a **dedicated RNG stream**, so enabling it consumes zero
  draws from the main simulation. This is what makes INV-1 and VAR-3 hold simultaneously.
- **ISO-2** A disabled mechanism must reduce its formulas to their pre-mechanism form exactly
  (`absorption()` returns 1.0 without E; `recv_atten()` returns 1.0 without F; `hp_coupling()`
  returns 0.0 without D).
- **ISO-3** Hidden state is never emitted. Enforced by an exact per-supplier scan of every emitted
  cell against that supplier's own resilience.

## 5. Leakage constraints

| Constraint | Implementation |
|---|---|
| Labels use **true** event time | `training_labels.event_at` from the true transition timestamp |
| Features use **observed** time | `sup_features` filters on `delivered_rec`; `asof_status` on recorded time |
| Agent decisions use only ≤ `as_of` | `agent_observe()` moves rows across a monotonic frontier, asserting each |
| Feature windows end at `t0` | by construction; asserted in the suite and again by `load_data.py` |

Mechanism G widens the gap between the two clocks rather than introducing a second timeline — it
extends V1's existing `observed_at` / `recorded_at` bitemporal pair.

## 6. Power targets

Target 2,000–5,000 positives per task per variant, **including Variant K**.

- **PWR-1** Impact is the binding task; it is the only one that reaches the range by scaling.
- **PWR-2** Delay and shortage sit on much larger denominators and must be **subsampled down**.
  `DELAY_SAMPLE_RATE` and `SHORTAGE_SAMPLE_RATE` exist for this.
- **PWR-3** Sampling must select *entities*, not label rows, so each sampled entity keeps its full
  time series, and the sampled set must be identical across all variants and seeds.

> **Not yet met.** PWR-2/PWR-3 are configured, range-checked and documented but **not wired into
> label emission**. Every variant currently emits all labels. See
> `PHASE2_PHASE6_IMPLEMENTATION.md` §7 items 2–3.

## 7. Output requirements

- **OUT-1** One `<table>.csv.gz` per table, gzip level 9, per-variant directory.
- **OUT-2** `resolved_config.json` carrying benchmark version, generation timestamp, seed, variant,
  mechanisms enabled, resolved config, and realised label/row counts.
- **OUT-3** Compression must not weaken INV-1 — verified by diffing compressed bytes, including
  across differing `PYTHONHASHSEED`.
- **OUT-4** `load_data.py` decompresses transparently and still accepts a plain `.csv` directory.

Measured ratio: **3.18×** (124.7 MB → 39.2 MB at V1 scale). Modest because roughly half the corpus
is UUID-keyed `inventory_history`, and random hex does not compress.
