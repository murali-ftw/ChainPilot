# Validator amendment 03 — the v8 clearance checks (B1–B13)

**The third authorised change to the validation instrument, and it is additive only.**

| | |
|---|---|
| Form | **new file `db/validator_amendment_03.py`** — 462 insertions |
| `db/validator.py` | **byte-identical, untouched.** SHA `188180bbb6127bea788b191348353c895f16f7866ccbedcac8fbd63f7dc95201` before and after |
| Diff against the frozen instrument | `0 files changed` — nothing inserted, altered or removed in `validator.py` |
| Checks added | 13 (one per clearance block), of which 3 are gate-blocking |
| Checks changed | **none** |
| Bands moved | **none** |
| New file SHA | `15d445608cc6374a48612172c54dff4b07f840b4f83b2565a518897cd610d31f` |

## Why this amendment is a separate file, when 01 and 02 edited `validator.py` directly

The v8 brief's standing rules list `db/validator.py` as **protected, do not modify**, while its
Stage 2.2 asks for an additive amendment covering the B1–B13 checks. Those two instructions point
in opposite directions on the same file.

A standalone module satisfies both without weakening either. `validator.py` keeps its exact bytes,
so its 229 checks return precisely what they returned before — the strongest possible reading of
"nothing existing altered or removed". The clearance checks still become repeatable, which is what
Stage 2.2 exists to secure. **This is deviation 49**; if the project would rather have the checks
inline, moving them is a mechanical edit and the verdicts do not change.

## What it adds, and why the frozen instrument could not already answer these

`validator.py` predates v8's rebuilt stores. It is thorough about structure, accounting and
marginal distributions, and it has no check at all for four things v8 was built to fix:

1. whether an opening stock level **reconciles** against the transaction ledger (B1) — the frozen
   ledger check tests an identity that is true of any cumulative sum;
2. whether a requirement plan is **forward-looking** (B2);
3. whether a label's entity is **observable at its own snapshot** (B9) — the as-of leak that
   Phase 11 caught by assertion, at training time, after the pilot had already been queued;
4. whether contract terms, qualification status and capacity ceilings carry **variation** rather
   than a documented constant (B4, B6, B7).

Each of those was a blocking item in `reports/phase-10.md` §6. A check that can confirm them is
worth more than a check that confirms the column exists, which was true of every defect this
project has found.

## Standing rule 1 — every check carries the mutation that makes it fire

Each check declares a `breaks` string naming the mutation that fires it, and every one has been
demonstrated firing. **23 gates, 23 demonstrated capable of failing.**

One of them was **inert on first writing and is recorded here because of it**: B8's coverage gate
was first falsified by thinning `supplier_allocation` to a 20% sample of *rows*. Coverage did not
move — 97.05% before, 97.05% after — because each part-plant carries ~149 allocation rows, so a
row sample still touches nearly every part-plant. The mutation had to thin the **part-plant**
population instead, at which point the gate fires correctly (19.5% → `NOT CLEARED`). The gate was
fine; the falsification was wrong, and a falsification that cannot fail is the same defect one
level up.

A second, stronger demonstration runs the whole suite against `db/gen_v6/seed_1001`, read-only:

| block | v8 | v6 | deciding number on v6 |
|---|---|---|---|
| B1 | CLEARED | **NOT CLEARED** | roll-forward exact 4.58% |
| B2 | CLEARED | **NOT CLEARED** | plan forward fraction 0.0% |
| B4 | CLEARED | **NOT CLEARED** | every term single-valued |
| B6 | PARTIAL | **NOT CLEARED** | non-modal qualification 0.0% |
| B7 | CLEARED | **NOT CLEARED** | revealed capacity 0% populated |
| B8 | CLEARED | **NOT CLEARED** | coverage 18.99% |
| B9 | NOT CLEARED | NOT CLEARED | 0.0% both worlds |
| B12, B13 | CLEARED | CLEARED | — |

Eight blocks change verdict between two real worlds. These gates discriminate on data, not only
under synthetic mutation.

## Blocking structure

B1, B9 and B12 are marked `blocking=True`. The runner exits non-zero unless all three read
`CLEARED`, and prints the gate outcome explicitly. B2–B8, B10, B11 and B13 are recorded with a
verdict and do not gate.

## Usage

```bash
python db/validator_amendment_03.py db/gen_v8/seed_1001
python db/validator_amendment_03.py db/gen_v8/seed_1001 --only B1 B9 B12 --sample 400
python db/validator_amendment_03.py db/gen_v8/seed_1001 --json out.json
```

`--sample N` reconciles N part-plants in B1 instead of all 4,212, for a fast pass. The full B1 run
and the B11 sweep each read the whole seed; budget about five minutes for the complete suite on a
3.7 GB seed.

The module never writes into a dataset directory. `db/gen_v6/**`, `db/gen_v7/**` and
`db/gen_v8/**` are read-only to it.

## One caution about the frozen instrument's side effect

`db/validator.py` writes its report to `docs/validation/external_dataset_validation.md` on every
run, overwriting whatever was there. Running it against v8 therefore silently replaced the
existing v6/v7 report; it was restored from git. Amendment 03 writes nothing unless `--json` is
passed. Anyone running the frozen validator against a new world should check `git status`
afterwards.
