# 04 — Benchmark Workflow

> **V2 reframing.** Previously the V1 application's request/screen flow. Replaced by the benchmark
> workflow: generate → validate → load → split → train → evaluate → compare.

---

## 1. End-to-end flow

```
  configure          generate            validate           load
 ┌──────────┐      ┌──────────┐       ┌──────────┐      ┌──────────┐
 │ Config   │─────▶│ simulate │──────▶│ in-gen   │─────▶│ Postgres │
 │ +variant │      │ 10 mechs │       │  suite   │      │ (optional)│
 └──────────┘      └──────────┘       └──────────┘      └──────────┘
                         │                  │
                         ▼                  ▼
                  <table>.csv.gz     exit non-zero on
                  resolved_config     any failure
                         │
        split            ▼            train        evaluate        compare
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ 60/20/20 │──▶│  5 seeds │──▶│ baseline │──▶│  paired  │──▶│ variant  │
   │positional│   │ per var. │   │  + cand. │   │ bootstrap│   │  deltas  │
   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## 2. Stages

### 1. Configure
Pick a variant and a config source: the `spec` preset (the spec's defaults), the `v1` preset (the
V1 reproduction fixture), or a JSON file of overrides. Every parameter is range-checked at startup;
out-of-range values and unknown parameter names abort before any work is done.

### 2. Generate
`db/generate_dataset.py` simulates the world: latent supplier stress drives shipment delays, which
drive inventory shortages, which drive labels. The ten mechanisms layer on top, each from a
dedicated RNG stream so that enabling one consumes no draws from the main simulation.

A variant whose mechanisms are not implemented is **refused**, never silently emitted as Variant 0.

### 3. Validate
The suite runs automatically at the end of generation and exits non-zero on failure. See
`13_Testing_Documentation.md`.

### 4. Load *(optional but recommended)*
`db/load_data.py` applies `schema.sql`, COPYs the tables in FK-safe order, runs verification
queries, and **rolls back everything if any check fails**. This is the only layer that inspects the
emitted artifact rather than in-memory structures, and it has already caught a referential-integrity
bug the in-generator suite missed.

### 5. Split
`temporal_splits()` divides snapshots 60/20/20 positionally. Cutoffs are identical across variants
by construction — the property that makes cross-variant comparison valid.

### 6. Train
Out of scope for this repository. A submission trains its own baselines and candidate per
`10_AI_ML_Documentation.md` §8, across five seeds, at fixed layer count and hidden dimension.

### 7. Evaluate and compare
`significant()` applies the spec's full criterion: a paired bootstrap CI excluding zero **and** sign
consistency across seeds. Deltas are taken against the correct baseline variant — which for A, D, F
and K is **not** Variant 0 (see `10_AI_ML_Documentation.md` §3).

## 3. Commands

```bash
# single variant
python3 db/generate_dataset.py --variant K --seed 42
python3 db/generate_dataset.py --variant 0 --config v1      # V1 reproduction fixture

# the matrix (counts only; the full sweep at spec scale is ~24 GB)
python3 db/run_benchmark.py --stats-only --report report.json

# reproducibility (diffs compressed bytes)
python3 db/run_benchmark.py --verify-determinism --variant K

# load and verify
createdb hades && python3 db/load_data.py --dsn "postgresql:///hades" --drop \
    --csv-dir db/csv/vK_seed42

# split composition and label availability
python3 db/benchmark_eval.py --dataset db/csv/vK_seed42 --check-splits
```

## 4. What a generated dataset contains

```
db/csv/v<variant>_seed<seed>/
├── resolved_config.json      version, timestamp, seed, variant, mechanisms, config, counts
├── suppliers.csv.gz          truncated by Mechanism A when enabled
├── supplier_upstream.csv.gz  Mechanism J only; absent otherwise
├── ...                       19 further tables
└── training_labels.csv.gz    delay / shortage / impact
```

Every dataset is self-describing: `resolved_config.json` records exactly what produced it, so a
published experiment can report its full configuration rather than only which mechanisms it enabled.
