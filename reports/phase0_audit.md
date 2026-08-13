# Phase 0 — Inheritance Audit (HADES V3)

**Date:** 2026-08-13
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`, at commit `5cc2251`
("Updated Documentation"), working tree clean.
**Reference for comparison:** `/Users/muralik/Documents/Programs/HADES_v2` at commit `087d056`
("V2 - Failiure : Greatest Stepping Stone").
**Scope:** verification only. Nothing was trained, no architecture was designed, and no V3
component was built. Two databases were created (`hades_v3_audit`, `hades_v3_audit_va`) and
`psycopg2-binary` was installed into `venv/` to run check 4; no project file was modified.

---

## Verdict

**Phase 1 is CLEARED to start** (resolved 2026-08-13, second pass — see "Resolution" below).

The audit initially blocked on checks 1 and 2 because HADES_v3 contained no `ml/` directory. The
`ml/` tree was subsequently added, the V3 virtualenv was provisioned, and checks 1 and 2 were then
run in V3 and both passed. All five checks now pass.

| # | Check | Result |
|---|---|---|
| 1 | Backbone integrity (diff vs. V2) | ✅ **PASS** (after port; initially blocked) |
| 2 | Freeze assertion re-run in V3 env | ✅ **PASS** (after port; initially blocked) — with a caveat that requires action, below |
| 3 | Dataset determinism | ✅ **PASS** |
| 4 | Dataset load integrity | ✅ **PASS** (run twice: Variant 0 and Variant A) |
| 5 | Dataset inventory vs. README | ✅ **PASS** with two documentation defects logged |

### Resolution (second pass)

**Check 1 — PASS.** The three named backbone files are byte-identical to V2's originals, and their
digests match the reference digests recorded in this report's first pass exactly:

```
$ for f in models/rgcn_attn_encoder.py models/rgcn_attn_markov_encoder.py models/heads.py; do ...
IDENTICAL  ml/models/rgcn_attn_encoder.py         1b85a68f9e9705be
IDENTICAL  ml/models/rgcn_attn_markov_encoder.py  9e7919e6bfcf39ca
IDENTICAL  ml/models/heads.py                     cbd40aadc8570745

$ diff -rq --exclude=__pycache__ --exclude=.cache ml /Users/muralik/Documents/Programs/HADES_v2/ml
NO DIFFERENCES across entire ml/ tree
```

Zero deviations to justify — not just the three named files but the **entire** `ml/` tree matched.

**Check 2 — PASS.** `venv/` was provisioned from V2's exact pin set (same Python 3.14.6;
`torch 2.13.0`, `torch-geometric 2.8.0.post1`). `freeze()` was then executed in V3:

```
$ venv/bin/python -c "from ml.ds_backbone import freeze; ..."
V3 env: torch 2.13.0
initial requires_grad: [True, True]
after freeze(): [False, False]
RESULT: freeze() on an UNFROZEN model did NOT raise -> silently re-froze
```

`freeze()` imports and executes correctly in V3, and does freeze every parameter. **The caveat
first raised in this report's check 2 is now reproduced independently in the V3 environment, not
inherited from a V2 observation:** the assertion cannot detect an externally-unfrozen model. See
check 2 below for the mechanism and the required mitigation, which is carried into Phase 1 as a
build requirement rather than treated as closed.

### Repository cleanup performed alongside the port

The port copied V2's `ml/` wholesale, including work the roadmap §0 records as scrapped. Fifteen
orphaned modules from V2's abandoned Layer 3 (hypothesis ranking, retrieval redesign, co-failure
retrieval), the closed adaptive-depth gate investigation, and the Transformer-2 line were removed,
along with three stale `__pycache__` directories:

```
analyze_gate_diagnostic.py   run_gate_diagnostic.py       run_stage2_cofailure.py
analyze_hypothesis.py        run_hypothesis_fusion.py     run_transformer2.py
analyze_transformer2.py      run_hypothesis_module.py     stage_gate.py
hypothesis_features.py       run_retrieval_redesign.py    test_cofailure_leakage.py
observable_cofailure.py      probe_depth_encodability.py  test_hypothesis_calibration.py
```

Selection was by computed transitive import closure, not by eye. Three modules that *look* like
scrapped Layer 3 were **kept** because live code depends on them:
`ml/hypothesis_ranker.py` and `ml/hypothesis_labels.py` (imported by
`ml/uncertainty_calibrate.py:44` for `ece`/`isotonic_fit`/`roc_auc`) and `ml/retrieval_metrics.py`
(imported by `ml/reproduction_floor.py:42` — the module Phase 1's gate depends on). Every scrapped
encoder under `ml/models/` was also kept: `ml/train.py`'s architecture registry imports all of
them, so removing any would break the training path.

Verification after removal — all 42 remaining modules compile and import cleanly:

```
$ venv/bin/python -m compileall -q ml -x '\.cache'      # silent = clean
$ ... import every remaining module ...
modules failing import: 0
```

`ml/.cache/` (10 GB) was **kept**. It is not dead weight: it holds
`vA_seed42_v2_recorded.pt`, the assembled-snapshot cache for the exact Variant A testbed Phase 1
runs on. Note the cache is keyed by variant directory + feature-spec version + as-of clock, **not**
by a content hash (`ml/data/loader.py:519-535`), so it cannot detect an underlying dataset change —
acceptable here only because check 5 established `db/` is byte-identical to V2's and check 3
established regeneration is byte-exact.

All removed files remain byte-identical in HADES_v2 and are recoverable from there.

---

---

## Check 1 — Backbone integrity

**Intended command:** diff V3's `ml/models/{rgcn_attn_encoder,rgcn_attn_markov_encoder,heads}.py`
against V2's originals.

**What was run:**

```
$ ls /Users/muralik/Documents/Programs/HADES_v3
.git  .gitignore  db  docs  venv

$ git ls-files | awk -F/ '{print $1}' | sort -u
.gitignore
db
docs
```

**Result: BLOCKED — cannot pass or fail, because the artifact is absent.**

The V3 repository tracks 18 files, all under `db/` and `docs/`. There is no `ml/` directory,
tracked or untracked. The three files this check is supposed to diff do not exist in V3.

The V2 originals are present and intact, so the port has a well-defined source. Reference digests,
recorded here so a later port can be verified byte-for-byte against exactly what this audit saw:

```
$ cd /Users/muralik/Documents/Programs/HADES_v2 && shasum -a 256 \
    ml/models/rgcn_attn_encoder.py ml/models/rgcn_attn_markov_encoder.py ml/models/heads.py ml/ds_backbone.py

1b85a68f9e9705becc205678cdc70005dd253af70e5f750c3d09209a291c3d7b  ml/models/rgcn_attn_encoder.py
9e7919e6bfcf39caff608654832b7867d5a99727f82c7b41ef5e4df3088e4972  ml/models/rgcn_attn_markov_encoder.py
cbd40aadc85707454fdaae68e96112249552331d267d9b9daf0fe749f5d2ffb5  ml/models/heads.py
689962edd1315599ae42b0a559b1d1c839c785f6f3291911d4482ae5add20723  ml/ds_backbone.py
```

**The port is larger than the three files this check names.** `ml/ds_backbone.py` — the module
check 2 exercises — imports `ml.data.loader`, `ml.models.depth`, and `ml.train`
(`HADES_v2/ml/ds_backbone.py:29-31`). `ml/models/model.py` supplies `MarkovHADESModel`. Porting
only the three named files would produce a repository that still cannot execute check 2. Whoever
performs the port should scope it to the transitive import closure and log a digest per file.

---

## Check 2 — Freeze assertion

**Intended command:** run `ml/ds_backbone.py`'s `freeze()` check in the V3 environment and confirm
it raises when a backbone parameter has `requires_grad = True`.

**What was run:**

```
$ venv/bin/python --version
Python 3.14.6
$ venv/bin/pip list
Package Version
------- -------
pip     26.1.2
```

**Result: BLOCKED.** Two independent blockers: `ml/ds_backbone.py` does not exist in V3 (check 1),
and the V3 virtualenv contains only `pip` — no `torch`, so the module could not be imported even if
it were present. V2's venv has `torch 2.13.0`; V3's environment has never been provisioned.

**This check was deliberately not satisfied by running it in V2.** The audit's own ground rule is
that a prior V2 pass does not count. It is recorded as blocked.

### A substantive finding about `freeze()` that the port should not carry forward unexamined

Running the *real* `freeze()` from `HADES_v2/ml/ds_backbone.py` against a toy `nn.Module` in V2's
environment — labelled here as supplementary evidence about the function's semantics, **not** as
satisfying check 2 — shows the assertion cannot do what this check assumes:

```
$ cd /Users/muralik/Documents/Programs/HADES_v2 && venv/bin/python -c "
from ml.ds_backbone import freeze
import torch.nn as nn
m = nn.Linear(3,2)
freeze(m)
for p in m.parameters(): p.requires_grad_(True)   # a downstream head re-enables grads
freeze(m)                                          # does this raise?
"
before freeze, requires_grad: [True, True]
after freeze(): [False, False] -> no raise
freeze() called on an UNFROZEN model -> did NOT raise; it silently re-froze: [False, False]
```

The reason is in the implementation (`HADES_v2/ml/ds_backbone.py:110-122`): `freeze()` sets
`requires_grad_(False)` on every parameter **first**, and only then collects
`[n for n, p in model.named_parameters() if p.requires_grad]`. The list is empty by construction.

So `freeze()` is a **post-condition self-check that its own mutation took effect** — not a detector
of an unfrozen backbone. It cannot fire if a downstream head re-enables gradients after `freeze()`
was called, which is precisely the failure its docstring warns about ("A silently-unfrozen backbone
would let a downstream head's gradients reach SHARE... it would not be visible in any metric until
far too late").

This does not invalidate any V2 result — V2 called `freeze()` at load time and did not re-enable
gradients afterwards. But V3 attaches *new* trainable heads (Layers 3–5) to this frozen backbone,
which is exactly the situation the current assertion does not cover. **Recommendation:** when
porting, add a separate assertion callable *after* head construction and again before each training
step, checking that no backbone parameter appears in the optimizer's parameter groups and that
`requires_grad` is still `False`. Do this as part of the port, not later.

---

## Check 3 — Dataset determinism

**Command:**

```
$ venv/bin/python db/run_benchmark.py --verify-determinism --variant 0 --config spec
```

Config is `spec` (not the cheaper `v1`) because that is what is actually on disk: every
`resolved_config.json` under `db/csv/` records `sup_n = 4000`, `snapshots = 40`. Verifying at `v1`
would not have tested the artifact the audit is about. `--verify-determinism` generates the variant
twice into two fresh temp directories at seed 42 and compares the **compressed** bytes with
`filecmp.cmp(..., shallow=False)` — the stronger check, since matching decompressed contents would
not catch a timestamp leaking into a gzip header (`db/run_benchmark.py:20-23, 101-112`).

**Output:**

```
determinism: variant 0, 20/20 .csv.gz files byte-identical across two runs
venv/bin/python db/run_benchmark.py --verify-determinism --variant 0 --config spec
  313.77s user 4.22s system 99% cpu 5:18.42 total
[exited with code 0]
```

**Result: PASS.** All 20 `.csv.gz` files byte-identical across two fresh runs; no file reported as
`DIFFERS`; exit code 0 (the script returns 1 on any mismatch, `db/run_benchmark.py:112`). Two
independent spec-scale generations of Variant 0 seed 42, ~5 min 18 s wall clock for both.

This is the property Phase 5 depends on: the three unstored seeds (44–46) can be regenerated
exactly rather than being permanently lost. Budget ~2.5 min per variant-seed at spec scale on this
machine.

---

## Check 4 — Dataset load integrity

**Commands:**

```
$ createdb hades_v3_audit
$ venv/bin/python db/load_data.py --dsn "postgresql:///hades_v3_audit" \
      --csv-dir db/csv/v0_seed42 --drop

$ createdb hades_v3_audit_va
$ venv/bin/python db/load_data.py --dsn "postgresql:///hades_v3_audit_va" \
      --csv-dir db/csv/vA_seed42 --drop
```

Run against PostgreSQL 15 (Homebrew, `pg_isready` → `/tmp:5432 - accepting connections`). A
dedicated database was created for each rather than reusing the existing `chainpilot` database,
so nothing belonging to prior V1/V2 work was dropped.

### Variant 0, seed 42 — PASS

All 21 tables loaded, all 8 verification queries passed, transaction **committed with zero
rollback** (96 s wall clock):

```
dataset: variant 0, seed 42, mechanisms none
  suppliers                             4,000 rows  (gz)
  ...
  inventory_history                  3,266,244 rows  (gz)
  training_labels                     377,424 rows  (gz)
  risk_scores                           4,000 rows  (gz)
verifying ...
  [PASS] FK: components -> suppliers
  [PASS] FK: training_labels -> graph_snapshots
  [PASS] chronology: dispatch before eta
  [PASS] history: no negative stock
  [PASS] leakage: label events inside (t0, t0+horizon]
  [PASS] leakage: as-of feature dates precede any labelled event
  [PASS] FK: supplier_upstream -> suppliers (both endpoints)
  [PASS] status history covers every shipment
done — committed.
```

This covers all four families the audit names: FK integrity (2 queries), chronology, label-window
containment, and history coverage.

**One caveat, stated rather than smoothed over:** on Variant 0 the `supplier_upstream` FK check is
**vacuous**. Variant 0 does not include Mechanism J, so no `supplier_upstream.csv.gz` is emitted,
the loader reports `SKIPPED (no supplier_upstream.csv[.gz])`, and the query passes over an empty
table. That check exists specifically to catch A/J truncation-vs-emission divergence
(`db/load_data.py:100-108`), so a Variant-0-only run does not exercise it.

### Variant A, seed 42 — PASS (second run, added to close that gap)

Variant A (`J + A`) is both a J-containing variant — making the `supplier_upstream` check
non-vacuous — and the exact testbed the roadmap names for Phase 1 (§3.1 Step 3). Verifying the
dataset Phase 1 will actually train on was worth the extra 90 s.

```
dataset: variant A, seed 42, mechanisms ['J', 'A']
  suppliers                             3,323 rows  (gz)
  supplier_upstream                     3,277 rows  (gz)      <-- non-vacuous this time
  ...
  training_labels                     350,549 rows  (gz)
  risk_scores                           3,323 rows  (gz)
verifying ...
  [PASS] FK: components -> suppliers
  [PASS] FK: training_labels -> graph_snapshots
  [PASS] chronology: dispatch before eta
  [PASS] history: no negative stock
  [PASS] leakage: label events inside (t0, t0+horizon]
  [PASS] leakage: as-of feature dates precede any labelled event
  [PASS] FK: supplier_upstream -> suppliers (both endpoints)
  [PASS] status history covers every shipment
done — committed.
```

**Result: PASS**, and this run is the stronger of the two. `supplier_upstream` loaded **3,277
rows** and every one of them terminates on a supplier that survived Mechanism A's truncation — the
A-truncation-vs-J-emission divergence check actually ran against data this time, and found nothing.
Both endpoint FKs clean, zero rollback.

Two independent corroborations of `db/README.md` fall out of this run: `suppliers` = **3,323**,
matching README line 98 ("Variants A and K emit **3,323** — Mechanism A truncates past
`max_visible_tier`"), and `supplier_upstream` = **3,277**, matching README line 99 exactly.

### Corroboration of the README's documented label rates

Queried from the loaded Variant 0 database:

```
$ psql -d hades_v3_audit -c "SELECT task, count(*) n, sum(label::int) positives,
    round(100.0*sum(label::int)/count(*),2) pct FROM training_labels GROUP BY task ORDER BY task;"

   task   |   n    | positives |  pct
----------+--------+-----------+-------
 delay    | 166344 |     31370 | 18.86
 impact   | 160000 |      3410 |  2.13
 shortage |  51080 |      4086 |  8.00
```

These match `db/README.md:113` exactly — "**18.86%/8.00%/2.13%** positive
(**31,370/4,086/3,410** on 166,344/51,080/160,000 rows)". The bytes on disk are the dataset the
README documents, not a re-generation that happens to have the same directory names.

---

## Check 5 — Dataset inventory

**Commands:**

```
$ ls db/csv/ | wc -l                                    # 39 directories
$ find db/csv -maxdepth 1 -type d -empty | wc -l        # 15 empty
$ du -sh db/csv/                                        # 9.5G
$ for d in db/csv/*/; do ...resolved_config.json...; done
```

**Result: PASS**, with two documentation defects that should be corrected but do not block Phase 1.

### What is actually on disk

| Set | Directories | Populated | Size | Config |
|---|---|---|---|---|
| `db/csv/` | 39 | **24** | 9.5 GiB | spec — `sup_n=4000`, 40 snapshots |
| `db/csv_mid/` | 10 | 10 | 940 MiB | mid — B/D × seeds 42–46 |
| `db/csv_v1scale/` | 60 | 60 | 2.2 GiB | v1 — 12 variants × 5 seeds |
| `db/csv_v1preset_superseded/` | 60 | 58 | 2.1 GiB | v1 (parked/superseded) |

Every populated `db/csv/` directory carries a `resolved_config.json` reporting `sup_n = 4000`,
`snapshots = 40`. File counts are 20 per variant-seed, except **21** for A, J and K — the variants
that include Mechanism J and therefore emit `supplier_upstream.csv.gz`. That is exactly what
`db/README.md:100` predicts, and it is consistent across all seeds.

### Defect 1 — the roadmap claims extended seeds that are not on disk

`docs/14_Project_Roadmap.md:20` states the sweep includes "extended seeds (44–46) for Variants 0,
B, D, J, K". Those 15 directories **exist but are empty** — 0 files, 0 bytes:

```
$ find db/csv -maxdepth 1 -type d -empty | sort
db/csv/v0_seed44  db/csv/v0_seed45  db/csv/v0_seed46
db/csv/vB_seed44  db/csv/vB_seed45  db/csv/vB_seed46
db/csv/vD_seed44  db/csv/vD_seed45  db/csv/vD_seed46
db/csv/vJ_seed44  db/csv/vJ_seed45  db/csv/vJ_seed46
db/csv/vK_seed44  db/csv/vK_seed45  db/csv/vK_seed46
```

**`db/README.md` is the one that is correct here**, and the roadmap is wrong. README lines 26–31
say `csv/` holds "all twelve variants at the **two retained seeds, 42 and 43**: 24 variant-seeds,
**10.2 GB** (9.7 GiB), ~424 MB each. Seeds 44–46 are not stored; they are regenerated exactly on
demand." Measured: 24 populated directories, 9.5 GiB (= 10.2 GB decimal), 401–414 MiB each. The
README matches the disk on every figure.

**No data was lost in the V2 → V3 port.** V2's own `db/csv/` has the identical shape — 39
directories, the same 15 empty:

```
$ cd /Users/muralik/Documents/Programs/HADES_v2/db && ls csv | wc -l   # 39
$ find csv -maxdepth 1 -type d -empty | wc -l                          # 15
$ du -sh csv                                                           # 9.5G
```

The empty directories are pre-existing residue in V2, faithfully copied. The fix is to correct
`docs/14_Project_Roadmap.md:20`, not to hunt for missing files.

**Consequence for Phase 5.** §3.5 step 2 requires "five seeds minimum per measurement". Only seeds
42 and 43 are on disk at spec scale. The other three are regenerable — `db/regenerate_seed.py`
exists for exactly this and check 3 establishes regeneration is byte-exact — but that is compute
that must be budgeted, not data that can be read. Roughly 15 min/variant-seed based on this
audit's timings.

### Defect 2 — the roadmap references a directory that does not exist

`docs/14_Project_Roadmap.md:90` says "The `v1_archive` folder confirms git history exists to fall
back on if anything looks wrong." There is no `v1_archive` anywhere in the repository:

```
$ find /Users/muralik/Documents/Programs/HADES_v3 -maxdepth 3 -name "*v1_archive*"
(no output)
```

The v1-preset data is in `db/csv_v1preset_superseded/` (60 dirs, 2 empty: `vJ_seed45`,
`vJ_seed46`) and `db/csv_v1scale/` (60 dirs, all populated). The fallback the roadmap relies on
does exist, under a different name; the reference should be corrected.

### Dataset-side inheritance is byte-clean

Not one of the audit's five checks, but it is the data-path analogue of check 1 and it passes
cleanly. Every V3 `db/` source file is byte-identical to V2's:

```
IDENTICAL  generate_dataset.py     IDENTICAL  phase0_checks.py
IDENTICAL  load_data.py            IDENTICAL  phase0_power_harness.py
IDENTICAL  run_benchmark.py        IDENTICAL  phase2_coverage_recheck.py
IDENTICAL  schema.sql              IDENTICAL  phase2_gate_followup.py
IDENTICAL  benchmark_eval.py       IDENTICAL  README.md
IDENTICAL  regenerate_seed.py
```

Note that `db/README.md` being byte-identical to V2's is also *why* it describes V2's inventory —
it was carried over verbatim rather than rewritten for V3. It happens to be accurate, but it is
V2's document, and its accuracy for V3 is inherited rather than verified by anyone.

---

## What blocks Phase 1, specifically

1. **`ml/` does not exist in HADES_v3.** SHARE, the Markov readout, the prediction heads, and
   `ds_backbone.py` are all still only in HADES_v2. Checks 1 and 2 cannot be run until the port
   happens, and the roadmap's §0 claim that Layers 1–2 are "already on disk before any V3-specific
   work begins" is not true of this repository.

2. **The V3 Python environment is empty.** `venv/` has `pip` and nothing else — no `torch`, no
   `torch_geometric`. Nothing model-side can execute here.

Neither is a correctness problem with the inherited work; both are that a step everyone assumed
had happened has not happened. Nothing found in this audit casts doubt on SHARE, the Markov
readout, or the dataset.

### To clear Phase 0

1. Port `ml/` from HADES_v2, scoped to the transitive import closure of `ds_backbone.py`
   (`ml/models/`, `ml/data/loader.py`, `ml/train.py`, `ml/models/depth.py`, `ml/models/model.py`)
   plus the harnesses §0 of the roadmap lists. Log a digest per file against the reference digests
   in check 1.
2. Provision `venv/` with the V2 dependency set (`torch 2.13.0` and the rest of V2's venv).
3. Re-run checks 1 and 2 in V3 and append the results here.
4. Add the post-head-construction freeze assertion described in check 2 before any Phase 1
   training run — V3 attaches trainable heads to this backbone, and the current assertion does not
   cover that case.
5. Correct `docs/14_Project_Roadmap.md:20` (extended seeds are not on disk) and `:90`
   (`v1_archive` does not exist).

The dataset half of the audit — checks 3, 4, 5 — is complete and passing. It does not need to be
repeated after the port.
