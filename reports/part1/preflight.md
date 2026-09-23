# Stage A — Preflight

Run on 2026-09-09. Working tree `/Users/muralik/Documents/Programs/HADES_v4`.
Commands are inline throughout; raw inventory JSON in the session scratchpad.

**Gate result: A1 PASSED, A2 archived (nothing deleted outside caches/OS cruft), A3 verified.
Phase 0 may start.** Three pre-existing defects are recorded in §A1.6 that I could not repair
and that are not caused by this stage.

---

## A1.1 Directory map

| Path | Purpose |
|---|---|
| `db/` | Datasets and the acceptance instrument. 12 GB. |
| `db/gen_v6/`, `db/gen_v7/` | The two worlds. Each holds `seed_1001/`, `seed_1002/` and its generator source. |
| `db/gen_v*/seed_100*/` | 49 spec CSVs + `_sim.npz`, `_events.npz`, `mechanism_parameters.json`. |
| `db/validator.py` | Acceptance instrument, amended once (amendment 01). Runnable. |
| `db/validate.py` | Original modular validator. **Not runnable** — imports `config`, `generators`, `ground_truth`, none of which exist in the repo. Retained: it is on the protected list. |
| `db/schema.sql` | Authoritative column/type/NOT NULL/enum definitions (49 tables). |
| `db/dataset_structure.md` | Authoritative schema prose. |
| `docs/` | Reports, validator transcripts, probe scripts, learnability harness. 1.8 MB. |
| `docs/archive/` | **Created by A2.** Orphaned run-7 intermediates. |
| `reports/` | **Created by this stage.** Output of Stages A and Phases 0–3. |
| `venv/` | Active Python 3.14 environment (`which python3` → `venv/bin/python3`). |
| `.git/` | Git repository, branch `HADES-v4`. |

**Directory-name correction.** The task refers to `db/gen_v6/seed1001`. The actual directories are
`seed_1001` and `seed_1002` (underscore). All paths below use the real names.

## A1.2 Per-CSV inventory

Produced by `wc -l` per file plus a header read for the column count; date span is the min/max of
the first date-like column, parsed with `pd.to_datetime(errors='coerce')`.

| # | table | cols | v6·1001 rows | v7·1001 rows | v6·1002 | v7·1002 | v7 MB | date span (v7·1001) | same? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `alternate_sources` | 9 | 16,072 | 16,072 | 16,072 | 16,072 | 0.9 | 1970-01-01→1970-01-01 `ramp_rate_pct_per_month` | differs |
| 2 | `asn` | 8 | 1,143,708 | 953,123 | 1,247,675 | 915,795 | 91.0 | 2016-01-06→2026-03-30 `dispatch_ts` | differs |
| 3 | `bom` | 10 | 4,000 | 4,000 | 4,000 | 4,000 | 0.2 | 2016-01-01→2016-01-01 `effective_from` | differs |
| 4 | `business_units` | 3 | 4 | 4 | 4 | 4 | 0.0 | — | IDENTICAL |
| 5 | `calendar` | 13 | 26,201 | 26,201 | 26,201 | 26,201 | 1.5 | 2016-01-01→2026-03-31 `date` | IDENTICAL |
| 6 | `channel_performance_weekly` | 20 | 8,598,520 | 8,598,520 | 8,598,520 | 8,598,520 | 813.4 | 2016-01-04→2026-03-30 `week_start` | differs |
| 7 | `customers` | 6 | 20 | 20 | 20 | 20 | 0.0 | — | IDENTICAL |
| 8 | `dataset_coverage` | 10 | 48 | 48 | 48 | 48 | 0.0 | 2016-01-01→2016-01-01 `earliest_available_date` | differs |
| 9 | `expedite_events` | 10 | 1,403 | 1,397 | 4,993 | 855 | 0.1 | 2016-03-01→2026-03-26 `event_ts` | differs |
| 10 | `goods_receipts` | 6 | 1,143,708 | 953,123 | 1,247,675 | 915,795 | 77.2 | 2016-01-09→2026-04-04 `receipt_ts` | differs |
| 11 | `grn_lines` | 9 | 1,143,708 | 953,123 | 1,247,675 | 915,795 | 93.9 | 2016-01-09→2026-04-04 `event_ts` | differs |
| 12 | `inventory_position_weekly` | 18 | 52,080 | 52,080 | 52,080 | 52,080 | 3.7 | 2017-02-27→2026-03-30 `week_start` | IDENTICAL |
| 13 | `inventory_snapshots` | 10 | 517,996 | 518,032 | 517,651 | 517,665 | 30.9 | 2016-01-01→2026-03-01 `snapshot_date` | differs |
| 14 | `inventory_transactions` | 9 | 12,703,783 | 13,543,769 | 13,725,807 | 12,732,551 | 1425.3 | 2016-01-04→2026-04-04 `event_ts` | differs |
| 15 | `line_stop_events` | 13 | 206 | 147 | 508 | 66 | 0.0 | 2016-01-15→2026-03-18 `stop_start_ts` | differs |
| 16 | `logistics_lanes` | 8 | 16,072 | 16,072 | 16,072 | 16,072 | 0.7 | — | differs |
| 17 | `model_outputs` | 14 | 0 | 0 | 0 | 0 | 0.0 | — | IDENTICAL |
| 18 | `part_costs` | 6 | 3,980 | 3,980 | 3,965 | 3,965 | 0.2 | 2019-01-01→2019-01-01 `effective_from` | differs |
| 19 | `part_demand_weekly` | 8 | 52,080 | 52,080 | 52,080 | 52,080 | 2.7 | 2017-02-27→2026-03-30 `week_start` | differs |
| 20 | `part_plant` | 11 | 4,340 | 4,340 | 4,340 | 4,340 | 0.2 | 2016-01-01→2016-01-01 `effective_from` | differs |
| 21 | `parts` | 13 | 620 | 620 | 620 | 620 | 0.0 | — | IDENTICAL |
| 22 | `plan_drift_features` | 15 | 15,120 | 15,120 | 15,120 | 15,120 | 1.4 | 2019-01-01→2025-12-01 `target_period` | differs |
| 23 | `plants` | 11 | 7 | 7 | 7 | 7 | 0.0 | 2015-01-01→2015-01-01 `commissioned_date` | IDENTICAL |
| 24 | `po_line_revisions` | 10 | 34,441 | 53,171 | 48,000 | 38,710 | 5.0 | 2016-01-04→2026-04-04 `event_ts` | differs |
| 25 | `po_line_schedules` | 7 | 400,000 | 400,000 | 400,000 | 400,000 | 33.5 | 2016-01-19→2020-06-05 `schedule_date` | differs |
| 26 | `po_lines` | 12 | 1,178,254 | 982,585 | 1,283,024 | 941,808 | 124.4 | 2016-01-19→2026-06-12 `original_promise_date` | differs |
| 27 | `product_economics` | 5 | 180 | 180 | 180 | 180 | 0.0 | 2019-01-01→2019-01-01 `effective_from` | differs |
| 28 | `production_actual` | 9 | 15,120 | 15,120 | 15,120 | 15,120 | 0.9 | 2019-01-01→2025-12-01 `period` | differs |
| 29 | `production_plan` | 11 | 15,120 | 15,120 | 15,120 | 15,120 | 1.3 | 2019-01-01→2025-12-01 `plan_date` | differs |
| 30 | `products` | 9 | 180 | 180 | 180 | 180 | 0.0 | — | IDENTICAL |
| 31 | `purchase_orders` | 11 | 1,178,254 | 982,585 | 1,283,024 | 941,808 | 108.1 | 2016-01-04→2026-04-04 `created_ts` | differs |
| 32 | `quality_inspections` | 9 | 1,143,708 | 953,123 | 1,247,675 | 915,795 | 77.5 | 2016-01-09→2026-04-04 `event_ts` | differs |
| 33 | `revealed_capacity_monthly` | 11 | 51,660 | 51,660 | 51,660 | 51,660 | 2.7 | 2016-01-01→2026-03-01 `month` | differs |
| 34 | `shortage_events` | 11 | 1,911 | 1,793 | 6,072 | 1,140 | 0.2 | 2016-01-11→2026-03-21 `shortage_start_ts` | differs |
| 35 | `snapshots` | 11 | 83 | 83 | 83 | 83 | 0.0 | 2016-07-04→2025-12-08 `as_of_ts` | differs |
| 36 | `sourcing_channels` | 13 | 16,072 | 16,072 | 16,072 | 16,072 | 1.4 | 2016-01-01→2016-01-01 `ppap_date` | IDENTICAL |
| 37 | `supplier_acknowledgements` | 7 | 1,178,254 | 982,585 | 1,283,024 | 941,808 | 85.5 | 2016-01-05→2026-04-08 `ack_date` | differs |
| 38 | `supplier_allocation` | 9 | 2,300 | 3,008 | 6,411 | 1,786 | 0.2 | 2019-01-01→2019-01-01 `effective_from` | differs |
| 39 | `supplier_audits` | 8 | 2,000 | 2,000 | 2,000 | 2,000 | 0.2 | 2019-01-01→2025-11-04 `audit_date` | differs |
| 40 | `supplier_capacity` | 8 | 51,660 | 51,660 | 51,660 | 51,660 | 4.0 | 1970-01-01→1970-01-01 `capacity_qty_per_month` | differs |
| 41 | `supplier_contracts` | 11 | 3,000 | 3,000 | 3,000 | 3,000 | 0.2 | — | differs |
| 42 | `supplier_financials` | 6 | 3,360 | 3,360 | 3,360 | 3,360 | 0.2 | 2019-12-31→2026-12-31 `period` | differs |
| 43 | `supplier_performance_weekly` | 21 | 224,700 | 224,700 | 224,700 | 224,700 | 23.0 | 2016-01-04→2026-03-30 `week_start` | differs |
| 44 | `supplier_quality_ppm` | 7 | 3,360 | 3,360 | 3,360 | 3,360 | 0.2 | 2019-01-01→2026-01-01 `period` | differs |
| 45 | `supplier_sites` | 9 | 420 | 420 | 420 | 420 | 0.0 | — | IDENTICAL |
| 46 | `supplier_upstream` | 8 | 260 | 260 | 260 | 260 | 0.0 | — | differs |
| 47 | `suppliers` | 13 | 420 | 420 | 420 | 420 | 0.0 | 2015-01-01→2015-01-01 `onboarded_date` | IDENTICAL |
| 48 | `tooling` | 7 | 2,500 | 2,500 | 2,500 | 2,500 | 0.1 | — | differs |
| 49 | `training_labels` | 15 | 992,142 | 1,070,700 | 1,018,824 | 1,070,700 | 169.8 | 2016-07-04→2025-12-08 `snapshot_date` | differs |

## A1.3 Completeness — both worlds carry all 49 tables

| | v6·1001 | v6·1002 | v7·1001 | v7·1002 |
|---|---|---|---|---|
| CSV files | **49** | **49** | **49** | **49** |
| total rows | 31,943,035 | 33,747,282 | 31,531,523 | 30,431,321 |
| on-disk | 3.1 GB | 3.3 GB | 3.1 GB | 3.0 GB |
| `_sim.npz`, `_events.npz`, `mechanism_parameters.json` | present | present | present | present |

**No table is missing.** Two carry no data, and both are expected rather than truncation:

- `model_outputs` — **0 rows in all four**. It is a model-output table with nothing written yet;
  the validator accepted it as structurally clean in runs 5–7.
- `inventory_position_weekly` — 52,080 rows, but **every numeric column is zero**
  (`[c for c in d.columns if is_numeric(d[c]) and (d[c]!=0).any()]` → `[]`). It is a placeholder the
  generator emits to satisfy the 49-table spec. Identical across both worlds for that reason.

Everything else is populated and consistent in width with `db/schema.sql`.

## A1.4 The two worlds are not the same data

SHA-256 of every table in `seed_1001`, both worlds:

    tables with IDENTICAL content across worlds: 11 / 49
    differing:                                   38 / 49

The 11 identical tables are exactly the deterministic masters and the two placeholders —
`business_units`, `calendar`, `customers`, `parts`, `plants`, `products`, `sourcing_channels`,
`supplier_sites`, `suppliers`, plus `inventory_position_weekly` and `model_outputs`. Both
generators draw the master population from the same seed with unchanged code, so byte-identity
there is correct and expected.

Every table that carries simulated behaviour differs, and the row counts differ materially —
`po_lines` 1,178,254 (v6) vs 982,585 (v7), `inventory_transactions` 12.7 M vs 13.5 M. **The worlds
are distinct.**

## A1.5 Seeds present

Both seeds are present for both worlds: `seed_1001` and `seed_1002` in each of `db/gen_v6/` and
`db/gen_v7/`. **None missing.** Phases 0–3 use `seed_1001` per the task; `seed_1002` is available
for a seed-variance check if a later phase wants one.

## A1.6 Python inventory, and three pre-existing defects

| File | What it is | Parses | Imports resolve | Referenced by |
|---|---|---|---|---|
| `db/validator.py` | Acceptance instrument (amendment 01 applied) | yes | yes | validation2–7 |
| `db/validate.py` | Original modular validator | yes | **no** — `config`, `generators`, `ground_truth` missing | protected list |
| `db/gen_v6/generator_v6.py` | Run-6 world generator | yes | yes | validation6 |
| `db/gen_v7/generator_v7.py` | Run-7 world generator | yes | yes | validation7 |
| `docs/learnability.py` | G2/G3/G4 harness | yes | yes | baseline, validation7 |
| `docs/joint_probes_v2.py` | §16 probes, corrected | yes | yes | validation7 |
| `docs/validator_run5_joint_probes.py` | §16 probes v1 | yes | yes | validation5–7 |
| `docs/validator_run4_joint_probes.py` | §16 probes, run 4 | yes | yes | validation4 |
| `docs/validator_run3_joint_probes.py` | §16 probes, run 3 | yes | yes | validation3 |

No file imports another; every script is standalone and takes its dataset path as `argv[1]`.

**Defect 1 — `db2/generator_source.py` is gone, and 40 links in `validation5.md` point at it.**
The restructure deleted `db2/`. A filesystem-wide search finds no surviving copy of
`generator_source.py`, `generator_runnable.py` or `generator_v5.py`. Run 5's entire source audit —
every line-numbered citation in `validation5.md` Part A — now points at a file that does not exist.
**I cannot repair this**: the artefact is gone, and the reporting rules forbid editing
`validation*.md`. Recorded so the loss is on the record rather than discovered later.

**Defect 2 — `V4_Corrections.md` is missing.** It is on the task's protected list and was at
`db/V4_Corrections.md` as recently as run 5. Not present anywhere in the tree.

**Defect 3 — `db/validate.py` cannot run**, as above. `validator.py` is the self-contained
replacement and carries a docstring explaining why the constants were copied rather than imported.

**Stale defaults, harmless but worth noting**: both generators default `--out gen_v5`, and
`validator_run5_joint_probes.py` defaults its dataset path to `gen_v5/seed_1001`. All three accept
an explicit path, so nothing breaks in use — but see the incident below.

### Incident during A1 — disclosed

To test runnability I executed each module with `importlib`, which runs module-level code. Both
generators do their work at module level, so `generator_v6.py` began generating into its default
`--out gen_v5` and wrote **2.9 GB** before I killed it. I removed the stray `gen_v5/` directory
immediately. **No protected data was touched** — the newest mtime inside all four world directories
is 09:57, from the original generation runs, and this session began at 12:50. The runnability check
was redone statically with `ast.parse` plus `importlib.util.find_spec`, which is what §A1.6 reports.

## A1.7 Broken references

    43 broken markdown links, all pre-existing:
      40  validation5.md -> ../db2/generator_source.py#L…   (defect 1)
       1  validation4.md -> ../db/generator/fix_capacity_only.py
       2  false positives: implementation_guide.md `data[nt].x` (code, not a link)

No Python import resolves to a deleted path. The three stale `gen_v5` defaults above are argument
defaults, not imports.

## A1.8 Disk

    free on volume: 255 GB
    db/  12 GB     docs/ 1.8 MB

**A1 gate: PASSED.** Both worlds complete at 49/49 tables, both seeds present, worlds verified
distinct.

---

## A2 Cleanup — proposal, then archive

The proposal was printed in full before anything moved. Summary:

**Deleted (regenerable or OS-generated only):**

| Path | Size | Reason |
|---|---|---|
| `docs/__pycache__`, `db/__pycache__`, `db/gen_v6/__pycache__` | 472 KB | regenerable bytecode |
| `./.DS_Store`, `db/.DS_Store` | 16 KB | macOS Finder metadata |
| `training_reports/` | 0 | empty, unused |
| **total reclaimed** | **~500 KB** | |

**Archived to `docs/archive/` (moved, not removed) — 11 files, 44 KB:**
`gen_v7_1001.log`, `gen_v7_1002.log`, `learn_gt_gen_v6_seed_1001.log`,
`learn_gt_gen_v7_seed_1001.log`, `learn_v7_1001.log`, `probes_v1_v7_seed_100{1,2}.txt`,
`probes_v2_v6_seed_100{1,2}.txt`, `probes_v2_v7_seed_100{1,2}.txt`.

These are the only files in `docs/` that **no** surviving `.md` mentions by name, so moving them
breaks nothing. They remain the raw evidence behind `validation7.md` §7 and §11 and are preserved
in full at their new path.

**Not touched, and why.** The disposable set here is genuinely tiny, because `docs/` is almost
entirely live audit trail:

- **Every** `validator_run*.txt` is referenced by a surviving report. A strict markdown-link scan
  called several of them orphans; a full-text scan showed run 1's transcripts
  (`validator_run_external.txt`, `validator_run_control_csv_full_seed1.txt`) are cited **in prose
  with backticks** by `external_dataset_validation.md`, which exists precisely because run 1's
  report was destroyed and those transcripts are its only surviving record. Link-scanning alone
  would have archived them.
- `validator_run{3,4}_joint_probes.py` are superseded but cited by validation3/4.
- `venv/` is the active interpreter, not a stale environment.
- I deliberately archived **nothing** referenced by a `validation*.md`, because repairing such a
  link would require editing those files, which the reporting rules forbid.

**Link re-check after the move:**

    newly broken by cleanup: 0
    pre-existing (db2/ etc, cannot repair): 43

**No protected file appears in either list.** Verified item by item against the task's list.

---

## A3 `.gitignore`

**This is a git repository**: `git rev-parse --git-dir` → `.git`, branch `HADES-v4`.
45 commits are reachable from remote refs (`origin/main`, `origin/HADES-v{1,2,3}`, `origin/GAT`,
`origin/dev`) but **branch `HADES-v4` has no commits and `git ls-files` returns 0**. Nothing is
tracked, so **no `git rm --cached` is needed**. Sampling the most recent reachable tree confirms
**0 CSVs were ever committed**.

`.gitignore` written as specified, with one deliberate change and one addition:

- **Changed** `db/gen_v6/` → `db/gen_v6/seed_*/` (and the same for v7). The blanket rule also
  excluded `db/gen_v7/generator_v7.py`, verified with
  `git check-ignore -v db/gen_v7/generator_v7.py` → `.gitignore:3:db/gen_v7/`. Losing the generator
  from version control contradicts `synthetic_rules.md` §18 — *"the generator source, not only the
  CSVs; the dataset is not reviewable without it."* Scoping to the seed directories keeps every
  dataset byte ignored and the source tracked.
- **Added** `*.npz`, for the `_sim.npz` / `_events.npz` simulation state that ships beside each
  world (up to 40 MB each).

### Verification

    $ git check-ignore -v db/gen_v6/seed_1001/po_lines.csv
    .gitignore:6:db/gen_v6/seed_*/   db/gen_v6/seed_1001/po_lines.csv
    $ git check-ignore -q db/gen_v7/generator_v7.py ; echo $?
    1                                    # not ignored — tracked, as intended

    $ git status --porcelain -uall | wc -l
    63                                   # files git would track
    $ git status --porcelain -uall | grep -cE '\.csv$|\.npz$'
    0                                    # no dataset file would be staged

**63 files would be tracked**: `.gitignore`, both generators, `db/validator.py`, `db/validate.py`,
`db/schema.sql`, `db/dataset_structure.md`, and the 56 documents, transcripts and scripts in
`docs/` (11 of them now under `docs/archive/`).

One consequence worth stating: `mechanism_parameters.json` sits inside the seed directories and is
therefore ignored. §18 asks for a shipped parameter file; the parameters are also present verbatim
in the tracked generator source as the `P = dict(...)` block, so the requirement is still met by
the tracked tree.

**A3 gate: PASSED.**

---

## Stage A verdict

| Gate | Result |
|---|---|
| A1 structure and completeness | **PASS** — 49/49 tables in all four seed directories, both seeds present, worlds verified distinct (38/49 tables differ) |
| A2 cleanup | **PASS** — ~500 KB deleted (caches and `.DS_Store` only), 11 files archived, 0 links newly broken |
| A3 `.gitignore` | **PASS** — verified, 63 files tracked, 0 dataset files |

Carried forward as findings, not blockers: `db2/generator_source.py` and `V4_Corrections.md` were
deleted before this session and cannot be recovered; `db/validate.py` does not run.
