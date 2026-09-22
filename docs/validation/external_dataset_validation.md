# ⚠️ This file no longer contains run 1's report

**Run 1's external-dataset validation report was overwritten on 2026-09-08 and could not be
recovered.**

## What happened

`validator.py` writes `../docs/external_dataset_validation.md` on **every** run by default
(`REPORT_PATH`, validator.py line 94). Only `--no-report` suppresses it. Run 3's control run
against `csv_full_seed1` was launched without that flag, so the validator replaced this file
with the *control* dataset's report before the behaviour was noticed.

Recovery was attempted and failed: this project is not a git repository, `tmutil` lists no
local snapshots, `~/.Trash` is empty, and `~/Documents` is local rather than iCloud-synced.
The original bytes are gone.

This file has deliberately **not** been reconstructed into something that looks like the
original. A reconstruction would be indistinguishable from the real run-1 report at a glance
and would be a fabrication of a primary record.

## No measurement was lost

Run 1's complete stdout is intact and is the authoritative record of that run:

- **`docs/validator_run_external.txt`** — run 1, external dataset, 606 lines, every check with
  its status, measured value, expected band and rationale.
- **`docs/validator_run_control_csv_full_seed1.txt`** — run 1, control.

Every run-1 figure in the trend table of `docs/validation3.md` (§2) and in its full tier tables
(§8) was parsed from `validator_run_external.txt`, not from this file. `docs/validation2.md`
also carries run-1 columns throughout. The three-way comparison is unaffected.

If a rendered run-1 report is wanted, it can be regenerated from the transcript above and
should be labelled as reconstructed.

## Note for future runs

Always pass `--no-report` when running the validator against any dataset, or the next run will
overwrite this file again:

    python3 <dataset>/validator.py <dataset> --no-report
