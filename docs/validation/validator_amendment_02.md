# Validator amendment 02 — constant-column check

**The second authorised change to `db/validator.py`.** It is an **addition only**: 104 lines
inserted, **zero lines removed or altered**. No existing band, tolerance, threshold, gate
condition, skip rule or exclusion moves. Every check that existed before this amendment
returns exactly what it returned before it.

| | |
|---|---|
| SHA before | `ed4742e0bef9d60af7c5b2a2550b590b27e6b217574df5b17877ba99be57ac56` |
| SHA after | `4dbb229836de2f3bcfd39ad4ff343b4c9b568f2d6c7e1b10591d167360dfb2ce` |
| Diff | `1 file changed, 104 insertions(+)` — no deletions |
| Checks added | 1 gate, 1 warning, 1 informational line per table with a constant column |
| Checks changed | **none** |
| Functions touched | `TableReport.__init__` (one new field), `tier1` (census + report), new module-level `_constant_bucket` |

## Why

A column holding one distinct value across an entire table carries **zero information**. It is
either a placeholder the generator declared and never wrote, or a bug. The validator had no
check for it at any tier, and the cost of that gap is on the record: Phase 2 of run 8 found
four such columns in `channel_performance_weekly` by hand, after building features on them —
`revision_count`, `days_since_last_short`, `weeks_since_last_activity`,
`weeks_since_last_receipt`, all zero in all 8,598,520 rows of both worlds. The specification
still lists all four as live features, and one of them (`weeks_since_last_activity`) as half
the replacement for `staleness_days`.

Nothing in Tier 1 could see this. Type checks pass on a zero. NOT NULL passes on a zero. The
PK and FK checks do not look at the column. A constant column is structurally perfect and
semantically empty, and that is precisely the shape of defect this check exists to catch.

## What it does

During the existing Tier 1 row pass, every **declared** column that is present accumulates a
distinct-value set capped at two entries. Cost is O(1) per cell after a column has seen its
second value, which for a real column is the second row. Tables with fewer than two rows are
exempt: there, constancy is arithmetic, not a defect.

Each column with exactly one distinct value is sorted into one of three buckets.

| bucket | definition | severity | why |
|---|---|---|---|
| **`zero`** | a declared numeric column that is `0` in every row | **FAIL** | there is no reading under which this is correct |
| `blank` | every value is the empty string | warn | often the documented open-ended convention — empty `effective_to` means *still effective* |
| `literal` | one distinct non-zero, non-blank value | warn | often a genuinely uniform dimension — one country, one currency, one contract start |

**Only `zero` is gated, and the narrowness is deliberate.** A `DECIMAL` column that is 0.0 in
every one of eight million rows was not written. An empty nullable column and a uniform
dimension both have readings under which they are correct, and a gate that fired on all three
would be ignored within a week — which is this project's "gate that cannot fail" failure mode
run in reverse. Both non-gated buckets are still **reported in full**, per table and in
aggregate, so nothing is hidden; they simply do not decide the verdict.

## Output

Three additions to the Tier 1 section:

```
CONSTANT COLUMNS
  [FAIL] channel_performance_weekly     4 all-zero, 0 all-blank, 0 literal    0 all-zero
         ALL-ZERO (unpopulated): days_since_last_short, revision_count,
         weeks_since_last_activity, weeks_since_last_receipt
  [warn] suppliers                      0 all-zero, 0 all-blank, 3 literal    0 all-zero
         single literal: country, is_active, onboarded_date
  ...
TABLES
  [FAIL] no all-zero numeric column                 29 column(s) in 5 table(s)    0
  [warn] other constant columns (warning)   21 all-blank, 85 single-literal       0
```

`res.facts["constant_columns"]` carries the machine-readable `{zero, blank, literal}` lists.

## What it finds on the existing worlds

`gen_v6/seed_1001`, and identically on `gen_v7/seed_1001`: **29 all-zero numeric columns
across 5 tables**, plus 21 all-blank and 85 single-literal reported as warnings.

| table | all-zero columns |
|---|---|
| `inventory_position_weekly` | 13 — `qty_on_hand`, `qty_blocked`, `qty_reserved`, `qty_in_transit`, `qty_available`, `open_po_qty`, `safety_stock_qty`, `days_of_supply`, `consumption_4w`, `consumption_13w`, `inter_plant_in_4w`, `inter_plant_out_4w`, `staleness_days` |
| `supplier_performance_weekly` | 8 — `lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13`, `revision_count`, `days_since_last_short`, `reporting_lag_days`, `weeks_since_last_activity`, `weeks_since_last_receipt` |
| `channel_performance_weekly` | 4 — `revision_count`, `days_since_last_short`, `weeks_since_last_activity`, `weeks_since_last_receipt` |
| `inventory_snapshots` | 3 — `qty_blocked`, `qty_in_transit`, `qty_reserved` |
| `quality_inspections` | 1 — `qty_deviation_accepted` |

**The `inventory_position_weekly` result is the one to act on.** Every numeric column in that
table is zero: it has 52,080 rows and no content at all. Any Phase 9 stock roll-forward built
against it would be rolling zeros forward. That table is now known-empty rather than assumed
populated, which is the whole point of the check.

## Effect on prior verdicts

**Both worlds now fail this new gate.** They failed it before it existed too; the check did not
create the defect, it surfaced it. Prior validation reports (`validation2.md` …
`validation7.md`) were computed before the check existed and are **not** restated — they remain
correct as records of what the frozen instrument measured at the time, and none of their
numbers moves. Reports from run 8 onward carry the check.

Because this amendment adds a failing gate to every dataset, the run-8 surgical-patch
comparison in [`reports/phase-1-fixes.md`](../reports/phase-1-fixes.md) is run **amendment-02
against amendment-02** — the pre-patch and post-patch validator runs on `gen_v6/seed_1001`
both include it, so the diff between them isolates the label change and nothing else.

## Conditions

Per the run-8 brief this is the only change authorised to `validator.py`. It is strictly
additive and strictly stricter: no dataset that passed before passes less easily on any
pre-existing check, and one class of defect that could not previously be detected now fails.
