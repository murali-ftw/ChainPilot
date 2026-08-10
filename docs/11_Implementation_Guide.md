# 11 — Build and Run Guide (V2)

> **V2 rewrite.** Previously the V1 application setup guide. Now the guide to generating,
> reproducing and evaluating HADES-Bench V2 datasets.

---

## 1. Requirements

- **Python 3.9+**, standard library only, for generation and evaluation.
- **PostgreSQL** and `psycopg2-binary`, only if you want to load datasets into a database.

```bash
python3 -m venv venv
venv/bin/pip install psycopg2-binary       # loader only; generation needs nothing
```

## 2. Generate a variant

```bash
python3 db/generate_dataset.py --variant K --seed 42
```

Writes to `db/csv/vK_seed42/`: one `<table>.csv.gz` per table plus `resolved_config.json`. The
validation suite runs automatically and exits non-zero on failure.

| Flag | Meaning |
|---|---|
| `--variant` | `0`, `A`…`K` (default `0`) |
| `--seed` | RNG seed (default `42`) |
| `--config` | `spec` (spec defaults), `v1` (V1 reproduction fixture), or a path to a JSON overrides file |
| `--out-dir` | output directory (default `csv/v<variant>_seed<seed>`) |

### Configuration presets

- **`spec`** — the spec's defaults: `sup_n=4000`, 40 snapshots. **The benchmark configuration.**
  ~115 s and ~443 MB per run.
- **`v1`** — `sup_n=800`, 15 snapshots, no subsampling. **The V1 reproduction fixture**, not a
  benchmark configuration. ~10 s per run. Several validation checks report rather than gate at this
  size; see `13_Testing_Documentation.md` §4.

### Overriding parameters

```bash
echo '{"sup_n": 2000, "snapshots": 24, "alpha": 0.5}' > my.json
python3 db/generate_dataset.py --variant D --config my.json
```

Every parameter is range-checked; out-of-range values and unknown names abort before generation and
list every violation at once.

## 3. Reproduce the V1 dataset

```bash
python3 db/generate_dataset.py --variant 0 --config v1
```

This must be **byte-identical** to the V1 dataset. It is the regression anchor every cross-variant
comparison depends on; if it drifts, stop and find out why before trusting any result.

## 4. Generate the matrix

```bash
# counts only, no CSVs — the full sweep at spec scale is ~24 GB
python3 db/run_benchmark.py --stats-only --report report.json

# a subset, with datasets written
python3 db/run_benchmark.py --variants 0,D,K --seeds 42,43 --config spec
```

Prints a per-variant report table with positive counts, rates, node/edge counts and power flags
(`LOW` / `ok` / `HIGH`), plus total validation failures.

## 5. Verify reproducibility

```bash
python3 db/run_benchmark.py --verify-determinism --variant K --config v1
```

Regenerates twice and diffs the **compressed** files. Expect
`21/21 .csv.gz files byte-identical`. Manual equivalent:

```bash
python3 db/generate_dataset.py --variant K --config v1 && cp -r db/csv/vK_seed42 /tmp/run1
python3 db/generate_dataset.py --variant K --config v1
diff -r --brief -x resolved_config.json /tmp/run1 db/csv/vK_seed42     # expect no output
```

`resolved_config.json` is excluded because it carries a generation timestamp by design.

## 6. Load into PostgreSQL

```bash
createdb hades
python3 db/load_data.py --dsn "postgresql:///hades" --drop --csv-dir db/csv/vK_seed42
```

Applies `schema.sql`, COPYs in FK-safe order, runs verification queries, and **rolls back
everything if any check fails**. `--verify-only` re-runs the checks alone.

**Run the loader before trusting a variant.** It is the only layer that inspects the emitted
artifact rather than in-memory structures, and it has already caught a referential-integrity bug the
in-generator suite missed.

## 7. Inspect splits

```bash
python3 db/benchmark_eval.py --dataset db/csv/vK_seed42 --check-splits
```

Prints the temporal split boundaries and per-task, per-split label counts. `--check-splits` exits
non-zero if any task has no test-split positives.

## 8. Reproduction checklist for a published experiment

1. Record `benchmark_version` and the full `config` block from `resolved_config.json` — not just
   which mechanisms were enabled.
2. Use **five seeds** per variant.
3. Compare against the **correct baseline variant**: A→J, D→B, F→E, K→D for hidden-structure
   claims. Not Variant 0.
4. Report paired bootstrap CI **and** sign consistency, plus mean and standard deviation.
5. State which validation checks gated and which reported at your configuration.

## 9. Current limitations

- The 12 × 5 sweep has only been run at the `v1` fixture. **No spec-scale sweep exists yet.**
- `DELAY_SAMPLE_RATE` / `SHORTAGE_SAMPLE_RATE` are configured and validated but **not wired into
  label emission**; every variant currently emits all labels.
- Power targets are met for no task at any scale actually generated.

See `PHASE2_PHASE6_IMPLEMENTATION.md` §7 for the full open-items list.
