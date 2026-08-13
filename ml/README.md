# `ml/` — architectures, data loading, and the evaluation harness

The models are ported from HADES V1 (`HADES_v1/ml/`); the data layer and the evaluation protocol are
new, because V2's benchmark artifact is a directory of gzipped CSVs rather than a live PostgreSQL
database. Results and reasoning: `reports/phase7_training_results.md`.

```
ml/
├── data/loader.py           db/csv/*.csv.gz -> HeteroData bundles + labels + temporal split
├── models/
│   ├── rgcn_attn_encoder.py     SHARE  — byte-identical copy of V1's module
│   ├── rgcn_relemb_encoder.py   SHARP  — byte-identical copy
│   ├── rgcn_battn_encoder.py    SHARK  — byte-identical copy
│   ├── rgcn_encoder.py          RGCN   — byte-identical copy
│   ├── heads.py, depth.py       byte-identical copies (focal loss, structural depth prior)
│   ├── encoder.py               factory: GCN/GraphSAGE/GAT/HGT/RGCN/GraphGPS + SHARE/SHARP/SHARK
│   └── model.py                 encoder + depth-prior readout + per-task heads
├── train.py                 V1's loop (AdamW, focal loss, best-val checkpoint) + early stopping
├── evaluate.py              metrics, block bootstrap, paired delta-AUC, sign consistency
└── run_benchmark_eval.py    the protocol: sanity / sweep / report
```

## The nine architectures

| id | name | matched-d arm | params |
|---|---|---|---|
| `gcn` | GCN | hidden=78 | 708,009 |
| `graphsage` | GraphSAGE | hidden=66 | 720,261 |
| `gat` | GAT | hidden=92 (single-head) | 731,495 |
| `hgt` | HGT | hidden=64 | 713,763 |
| `rgcn` | RGCN | hidden=138, num_bases=4 | 757,565 |
| `gps` | GraphGPS | hidden=40 | 688,123 |
| `rgcn_attn` | **SHARE** | hidden=128, num_bases=10 | 752,211 |
| `rgcn_relemb` | SHARP | hidden=128, num_bases=10, relemb=16 | 752,599 |
| `rgcn_battn` | SHARK | hidden=110, num_bases=9, num_bases_attn=16 | 738,273 |

Every count except GCN's and GraphGPS's reproduces `HADES_v1/reports/info.md` §4 exactly — V1 ran
neither of those two arms. Counts are for a 20-meta-relation variant; Variants A, J and K carry
Mechanism J's `UPSTREAM_OF` relation and so run 22, which moves the per-relation architectures'
counts slightly (disclosed in every results table).

## Running it

```bash
# 1. the gate: does the port reproduce V1 on the byte-identical anchor?
python3 ml/run_benchmark_eval.py sanity --seeds 0,1,2,3,4 --out out/sanity.json

# 2. the sweep. Four worker processes at THREE threads each is the measured optimum
#    (~419 s/run); one thread per process is 8.6x slower, and MPS does not parallelise
#    -- reports/phase7_training_results.md §5.1 and §8.1.
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python3 ml/run_benchmark_eval.py sweep \
    --csv-dir db/csv_v1scale --variants I,J,K --seeds 42,43,44,45,46 \
    --epochs 100 --out out/sweep_IJK.json

# spec scale: point at db/csv instead (see reports/phase7_training_results.md §3 first)
python3 ml/run_benchmark_eval.py sweep --variants 0 --seeds 42 --out out/spec_0.json

# 3. tables, variant effects, and the mandated caveats
python3 ml/run_benchmark_eval.py report --results 'out/sweep_*.json'
```

`--auto-regenerate` calls `db/regenerate_seed.py` for the three seeds that are not retained on disk
rather than silently skipping them.

## Things worth knowing before reading a number

- **The split is temporal, by snapshot count, 40/20/40**, read from `resolved_config.json` rather
  than a hardcoded date table. At 15 snapshots it reproduces V1's cutoffs exactly.
- **In `sweep` mode the five seeds are DATASET seeds** — five different generated worlds, model init
  held fixed — so the reported variance is the benchmark's. In `sanity` mode they are model-init
  seeds, which is what V1's own numbers vary over.
- **CIs are block bootstraps over whole snapshots** where the split has ≥4 of them, which V2's does
  and V1's did not; `ci_method` records which was used in every row.
- **Variant interpretation rules are enforced, not documented.** `REPORT_AGAINST` makes Variant A
  reportable only against J, D only against B, F only against E, K only against D. Asking for
  anything else raises.
- **The loader reads `recorded_at`, not `changed_at`**, for as-of shipment status — Mechanism G's
  whole purpose is that those differ.
- **Snapshot assembly is cached** to `ml/.cache/` per (variant-seed, feature spec, clock), because
  nine architectures × five seeds read the same bundles 45 times for each time they are built.
- **Do not mix devices inside a comparison.** `--device mps` works (and is 2.2x faster than CPU for
  a single job at ~30k nodes) but CPU/MPS float32 reduction orders differ, and the cross-variant
  deltas this harness reports are as small as 0.002.
