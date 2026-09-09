# Phase 1 — Loader and graph builder

Device MPS / torch 2.14.0 (graph construction is CPU-side; MPS is not used until Phase 3).
Run on **both worlds, one at a time**. Code: `ml/data/loader.py`, `ml/data/cache.py`.

**Gate: PASSED.** Both worlds load, both graphs reproduce the run-7 profile exactly, all three
as-of assertions pass.

| | v6 | v7 |
|---|---|---|
| as-of assertions (A1, A2) | **pass** | **pass** |
| core graph nodes / edges | **17,119 / 48,216** | **17,119 / 48,216** |
| median channel degree | **3** | **3** |
| components / coverage | **1 / 100.00%** | **1 / 100.00%** |
| as-of check wall-clock | 11.1 s | 10.9 s |
| graph build wall-clock | 0.6 s | 0.1 s |
| cache build wall-clock | 6.2 s | 5.6 s |
| peak RSS | 2.07 GB (5.58 GB incl. cache build) | 3.02 GB (5.95 GB incl. cache build) |

---

## 1.1 Reader — format-agnostic

`table_path()` resolves `.csv` then `.csv.gz`, so the loader works against either without a flag.
`read_rows()` streams via `csv.DictReader`; `read_df()` goes through pandas for the large tables,
as the guide advises.

## 1.2 As-of discipline — asserted, not assumed

Three invariants, each a hard `assert` in the loader. The task's instruction was to assert rather
than assume, so a failure stops the phase.

### A1 — `recorded_ts >= event_ts` on every transactional table

```
assert_a1_no_negative_lag(csv_dir)
```

Checks 11 tables (`po_lines`, `purchase_orders`, `grn_lines`, `asn`, `goods_receipts`,
`quality_inspections`, `inventory_transactions`, `supplier_acknowledgements`, `shortage_events`,
`expedite_events`, `po_line_revisions`), resolving the event column per table
(`created_ts` / `dispatch_ts` / `receipt_ts` / `shortage_start_ts` / `event_ts`).

**Result: PASS on 11/11 tables, 0 violations, in both worlds. 0 tables skipped.**

### A2 — the weekly store buckets on `max(event_week, recorded_week)`

Asserted the only way that actually proves it: recompute the visible week from the source tables
and require **exact integer conservation** against the store.

```python
vw = max( week(created_ts), week(recorded_ts) )        # weeks from the store's first Monday
assert po_lines.qty_ordered[in_span].sum() == cpw.qty_ordered.sum()
assert grn_lines.qty_received[in_span].sum() == cpw.qty_received.sum()
```

| world | ordered units | received units | weeks |
|---|---|---|---|
| v6 | **206,279,157** exact | **182,173,072** exact | 535 (2016-01-04 → 2026-03-30) |
| v7 | **265,126,832** exact | **209,163,226** exact | 535 (2016-01-04 → 2026-03-30) |

These reproduce `validation7.md` §13 to the unit — the loader and the frozen validator agree
independently. Bucketing on `event_week` alone would not balance; the assertion would fire.

### A3 — features are gated on `recorded_ts`, never `event_ts`

Implemented as a **sweep**, per the guide, so a caller cannot forget it: `AsOfSweep` sorts once by
`recorded_ts` and admits rows only as the cutoff reaches them. Phase 2's feature join uses
`week_start <= t0` on the store, which A2 has just proven is itself a `recorded_ts`-respecting
quantity.

## 1.3 Node tables — inductive only

Contiguous integer index per node type. **No entity id, index, hash or embedding-by-identity is a
feature.** This is enforced by construction: the feature builders take only attribute columns.

| node type | count | feature dim | composition |
|---|---|---|---|
| channel | 16,072 | 11 | contracted lead time, log distance, transport mode (4), approval status (4), is_approved |
| supplier | 420 | 11 | tier (2), type (3), business class (3), payment terms, msme flag, is_active |
| part | 620 | 12 | category (5), material (4), is_critical, log standard lead time, log shelf life |
| plant | 7 | 3 | log capacity/day, latitude, longitude |
| supplier_group | 84 | — | index only; no attribute table exists (guide O1) |

Categorical cardinalities are pinned in `CAT`, not inferred from the loaded batch.
`parts.criticality_reason` is excluded as the guide instructs (constant).

Why this matters beyond hygiene: 738 channels (4.59%) never trade, every Rane entity will be
unseen, and the Phase 3 cross-world evaluation requires a model trained on v6 to run on v7 —
impossible if anything is keyed on identity.

**Correction to the guide.** Its step 1.3 lists `parts.std_cost_inr`; the column is actually
`standard_lead_time_days`. The loader raised `AttributeError` on first run and I fixed the loader,
not the data.

## 1.4 Edges and `HeteroData`

**Core relations — the three that define the run-5/7 profile**, each also added in reverse:

    channel --sourced_from--> supplier      16,072 edges
    channel --supplies-->     part          16,072
    channel --delivers_to-->  plant         16,072
                                     total  48,216

**Verification against the run-7 figures the task supplied:**

| | expected | v6 measured | v7 measured |
|---|---|---|---|
| nodes | 17,119 | **17,119** ✓ | **17,119** ✓ |
| edges | 48,216 | **48,216** ✓ | **48,216** ✓ |
| median channel degree | 3 | **3** ✓ | **3** ✓ |
| components | 1 | **1** ✓ | **1** ✓ |
| largest component | 100% | **100.00%** ✓ | **100.00%** ✓ |
| isolated nodes | 0 | **0** ✓ | **0** ✓ |

Degree distribution of the other node types, identical in both worlds because the masters are
byte-identical across them (Stage A §A1.4):

| node type | median | min | max | isolated |
|---|---|---|---|---|
| supplier | 38 | 21 | 55 | 0 |
| part | 26 | 12 | 45 | 0 |
| plant | 2,285 | 2,227 | 2,362 | 0 |

**Extended relations** from the guide's 1.4, built and reported separately so the core profile
stays comparable to prior runs:

    supplier --depends_on--> supplier            260   (supplier_upstream)
    part     --alt_sourced_from--> supplier   16,072   (alternate_sources)
    part     --stocked_at--> plant             4,340   (part_plant)

Two guide expectations I could not meet, and why — both are the guide being stale rather than the
builder being wrong:

- **`po_line` nodes.** The guide wants snapshot-local open PO lines (`< 50,000`). Our `po_lines`
  carries no `status` column and open-ness must be derived from absence in `grn_lines`; I have left
  `po_line` nodes out of Phase 1 and will add them in Phase 3 only if the h⁰/h⁴ measurement needs
  them. Recorded as not-done rather than quietly skipped.
- **`part --used_in--> product` from `bom`.** `bom` has 4,000 rows against the guide's asserted
  18,532, and its `parent_part_id` is empty throughout, so the multi-level explosion the guide
  describes does not exist in these worlds. Not built.

## 1.5 Snapshot cache

**Finding that changes the design.** `sourcing_channels` is time-invariant in both worlds —
every row `effective_from = 2016-01-01`, `effective_to` NULL, `approval_status = 'approved'`. The
**core graph topology is therefore identical at all 83 snapshots**, and the guide's per-snapshot
effective-date filter is a no-op for it. Caching 83 copies of the same graph would be pure waste,
so the cache stores **one graph per world** plus the time-varying features.

**Second finding.** The weekly store is a complete contiguous panel — 535 weeks present for every
one of the 16,072 channels, `panel_complete=True` — so it densifies exactly into `[NCH, T, d]`
with no ragged padding. The guide's left-padding instruction (written for a panel with min 19
weeks per channel) is correct but inert here.

    ml/artifacts/cache/{v6,v7}/
        panel.npy    [16072, 535, 14] float32   482 MB    the 14 per-timestep channel features
        miss.npy     [16072, 535, 10] float32   344 MB    observed-indicator for the 10 nullables
        active.npy   [16072, 535]     float32    34 MB    is_active_week
        graph.pt                                          HeteroData, one per world
        meta.json                                         column order, week0, T, completeness

Round-trip verified: both worlds reload at the stated shapes with `panel_complete=True`.
Arrays are memory-mapped on load, so Phase 2 and 3 do not re-parse a CSV.

---

## What is deliberately not done yet

| Item | Status | Reason |
|---|---|---|
| `po_line` nodes | not done | no `status` column; deferred to Phase 3 if the ablation needs them |
| `bom` part→product edges | not done | 4,000 rows, `parent_part_id` empty — no explosion exists here |
| Per-snapshot graph cache | **not applicable** | topology is static; one graph per world is the correct cache |
| Normalisation | Phase 2 | must be fitted on the training fold only |

**Gate: both worlds load, both graphs match the expected profile to the digit, A1/A2 assertions
pass. Phase 2 may start.**
