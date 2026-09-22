# docs/

```
specs/        the standing specification. What the dataset and the models are supposed to be.
validation/   the written validation record, one document per run, newest last.
  runs/       raw validator output those documents were written from.
tools/        the probe and learnability scripts used by a particular validation run.
logs/         generation and training logs. gen_*.log is gitignored (reproducible from
              the generator plus the seed); everything else here is kept.
v8/           the V8 build: report, seed-variance band, falsifiability selftest.
  runs/       per-seed validator output.
archive/      superseded run output from v6/v7, kept for reference.
```

## Where to start

| If you want | Read |
|---|---|
| what the world is meant to contain | `specs/synthetic_rules.md`, `specs/data_plan.md` |
| what the models are meant to do | `specs/model_plan.md`, `specs/benchmark_specification.md` |
| how it is all put together | `specs/implementation_guide.md` |
| the current state of the dataset | `v8/v8_report.md` |
| whether a validator check can actually fail | `v8/selftest_mutations.txt` |
| how a number varies across seeds | `v8/seed_variance.json` |
| the history of how the world got here | `validation/validation2.md` … `validation7.md` |

## Conventions

- A `runs/` folder holds **raw instrument output**. It is evidence, not prose; the prose
  that cites it sits one level up.
- A validation document is written once and not edited afterwards. Corrections go in a
  later document or in `validation/validator_amendment_*.md`, so the record stays
  auditable.
- `specs/` is the only directory that describes intent. Everything else describes what
  was measured.

## Code that writes here

- `db/validator.py` writes `validation/external_dataset_validation.md`.
- `db/gen_v8/validate_all.sh` writes `v8/runs/` and `v8/seed_variance.*`.

Move either of those directories and update the writer with it.
