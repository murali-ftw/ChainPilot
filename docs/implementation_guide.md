# HADES v4 / ChainPilot — Implementation Guide

**Build reference for the models specified in
[`benchmark_specification.md`](benchmark_specification.md).**

This document is written to be **queried, not read front to back**. Jump to the step you
need; each is self-contained and states its own inputs and dependencies.

Every step gives the same six fields:

| Field | Meaning |
|---|---|
| **What** | One sentence |
| **Files** | Exact paths created or modified |
| **Depends on** | Which earlier steps must be complete |
| **Inputs** | Exact CSV paths and column names |
| **Implementation** | The approach, with code sketches where non-obvious |
| **Verify** | A concrete check with an expected result |

Counts and column names throughout were measured from `db/gen_v7/seed_1001/`. Do not trust a
column name from a planning document without checking it against the file.

---

## Contents

- [Dependency graph](#dependency-graph)
- [Phase 0 — Environment](#phase-0--environment)
- [Phase 1 — Data loading](#phase-1--data-loading) ← **start here, nothing trains until this is right**
- [Phase 2 — Sequence assembly](#phase-2--sequence-assembly)
- [Phase 3 — Temporal encoder](#phase-3--temporal-encoder)
- [Phase 4 — Graph encoder (SHARE)](#phase-4--graph-encoder-share)
- [Phase 5 — Heads](#phase-5--heads)
- [Phase 6 — Training loop](#phase-6--training-loop)
- [Phase 7 — Baselines](#phase-7--baselines)
- [Phase 8 — Evaluation](#phase-8--evaluation)
- [Phase 9 — Simulation](#phase-9--simulation)
- [Phase 10 — Optimisers](#phase-10--optimisers)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)

---

## Dependency graph

```
Phase 0  Environment
   │
   ▼
Phase 1  Data loading  ◄──────── everything depends on this
   │
   ├─────────────┬──────────────┬─────────────────┐
   ▼             ▼              ▼                 ▼
Phase 2      Phase 7        Phase 9          Phase 10
Sequences    Baselines      Simulation       Optimisers
   │         (needs 8's     (needs 5's       (needs 9)
   ▼          metrics)       outputs)
Phase 3
Temporal enc.
   │
   ▼
Phase 4  Graph encoder
   │
   ▼
Phase 5  Heads ──────────► Phase 6  Training loop
                                │
                                ▼
                           Phase 8  Evaluation
```

**Can run in parallel once Phase 1 is done:**

- **Phase 7 (baselines)** — no neural code at all. Build these *first*; §7 of the
  specification requires the comparison to exist before the neural model, so it cannot be
  retrofitted.
- **Phase 10 (optimisers)** — the LP and the enumeration phase need only `part_demand_weekly`,
  `supplier_contracts`, `logistics_lanes` and `part_plant`. Neither needs a learned model.
- **Phase 9 (simulation)** — the stock roll-forward and copula machinery can be built and
  tested against *baseline* fill/timing distributions, then swapped to the neural outputs.

**Strictly sequential:** 1 → 2 → 3 → 4 → 5 → 6 → 8.

**Suggested order for one person:** Phase 0, 1, 7, 2, 3, 4, 5, 6, 8, 9, 10. Baselines early
because they are cheap, they surface data problems, and they may make the rest unnecessary.

---

## Phase 0 — Environment

### Step 0.1 — Dependencies

**What** — Install the runtime.

**Files** — `ml/requirements.txt`

**Depends on** — nothing

**Inputs** — none

**Implementation**
```
torch>=2.2              # TCN + SHARE
torch-geometric>=2.5    # HeteroData, message passing
numpy>=1.26
pandas>=2.2             # loading only; not in the training loop
lightgbm>=4.3           # baselines B5, and UC9
scikit-learn>=1.4       # metrics, calibration
ortools>=9.9            # CP-SAT for the MILP (Phase 10); HiGHS also fine
pyarrow>=15             # Parquet cache (step 1.5)
```
The generator itself needs none of this — it is stdlib plus numpy. Keep the two environments
separate so a modelling dependency can never affect data reproducibility.

**Verify** — `python -c "import torch, torch_geometric, lightgbm, ortools; print('ok')"`

---

### Step 0.2 — Directory layout

**What** — Fix where model code and artefacts live.

**Files** — creates `ml/`

**Depends on** — 0.1

**Inputs** — none

**Implementation**
```
ml/
  data/
    loader.py         # Phase 1: as-of reads, HeteroData construction
    sequences.py      # Phase 2: [T x d] tensors, masking, normalisation
    cache.py          # Phase 1.5: Parquet/torch snapshot cache
  models/
    tcn.py            # Phase 3
    share.py          # Phase 4
    heads.py          # Phase 5
    staleness.py      # Phase 5.0: the gating layer
  train/
    loop.py           # Phase 6
    folds.py          # Phase 6.1: time splits
  baselines/
    rolling.py        # B2, B3, B4
    lgbm.py           # B5
    ablation.py       # B6, h0
  eval/
    metrics.py        # Phase 8
    backtest.py       # Phase 8.2: rolling origin
  sim/
    montecarlo.py     # Phase 9
    copula.py         # Phase 9.2
  opt/
    schedule_lp.py    # Phase 10.1
    allocation.py     # Phase 10.2
  artifacts/          # checkpoints, metrics, predictions — gitignored
```

**Verify** — `find ml -name '*.py' | wc -l` returns the expected file count; `ml/artifacts/`
is in `.gitignore`.

---

### Step 0.3 — Choose the world to develop against

**What** — Decide which generated dataset each activity uses.

**Files** — `ml/config.py`

**Depends on** — 0.2

**Inputs** — `db/gen_v6/seed_1001/`, `db/gen_v7/seed_1001/` (and the `seed_1002` twin of each)

**Implementation**

The three size presets (`small` / `mid` / `full`) are **gone**. There are now two *worlds*, each
generated by its own committed generator, each at full scale. They are not sizes — they are
different regimes, and the difference between them is the experiment.

| World | Path | Size | Generator | Use for |
|---|---|---|---|---|
| `v6` | `db/gen_v6/seed_1001/` | 3.1 GB, 31.9M rows | `db/gen_v6/generator_v6.py` | The run-6 regime. Persistent per-channel demand. |
| `v7` | `db/gen_v7/seed_1001/` | 3.1 GB, 31.5M rows | `db/gen_v7/generator_v7.py` | The run-7 regime. Demand split into a persistent level plus an AR(1) transient. |

Both carry a `seed_1002` twin for seed-variance checks. All 49 spec tables are present in each,
flat — there is **no `derived/` subdirectory**; the weekly stores sit alongside the raw tables.

Every figure that leaves the team is quoted from a named world and seed, because the two differ
materially: `po_lines` is 1,178,254 in v6 and 982,585 in v7, and 38 of 49 tables differ by content
hash. Quoting "the dataset" without naming the world is now meaningless.

There is no regeneration shortcut. To rebuild a world:
```bash
python3 db/gen_v7/generator_v7.py --seed 1001 --out db/gen_v7
```
It takes ~170 s and peaks near 11 GB.

**Verify**
```bash
for w in gen_v6 gen_v7; do
  echo "$w: $(ls db/$w/seed_1001/*.csv | wc -l) tables"   # 49 each
done
```

---

## Phase 1 — Data loading

> **This is step 1 of the whole project. Nothing trains until it is correct.** Every leakage
> guarantee in the specification is enforced here or not at all.

### Step 1.1 — CSV reader (format-agnostic)

**What** — Read the emitted CSVs. Plain `.csv` today; accept `.csv` too so this
never has to be revisited.

**Files** — `ml/data/loader.py`

**Depends on** — 0.3

**Inputs** — any `db/gen_v7/seed_1001/*.csv`

**Implementation**

Every file is a plain CSV with a header, `NULL` encoded as the empty string, dates as
`YYYY-MM-DD` and timestamps as `YYYY-MM-DD HH:MM:SS`. Earlier presets were gzipped; resolve the
extension rather than assuming either one.

```python
import csv, gzip, os

def table_path(csv_dir: str, table: str) -> str:
    """Return the readable path for `table`, plain or gzipped."""
    for ext in (".csv", ".csv"):
        p = os.path.join(csv_dir, table + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{table}[.csv|.csv] not under {csv_dir}")

def _open(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") \
           else open(path, "rt", newline="")

def read_rows(csv_dir: str, table: str):
    with _open(table_path(csv_dir, table)) as fh:
        yield from csv.DictReader(fh)
```

`pandas` reads either transparently, so `pd.read_csv(table_path(d, t))` needs no special case.

For the large tables (`po_lines` 982,585 rows; `grn_lines` 953,123;
`channel_performance_weekly` 8,598,520) prefer pandas with explicit dtypes — the
`DictReader` path is ~4× slower and materialises a dict per row.

Do **not** use `pd.read_csv(..., parse_dates=...)` on the timestamp columns at load time; it
is slow and the as-of comparisons in 1.2 work correctly on ISO strings. Parse only what you
need, when you need it.

**Verify**
```python
# measured on db/gen_v7/seed_1001 (run 7). gen_v6 differs: po_lines 1_178_254.
assert sum(1 for _ in read_rows(D, "po_lines")) == 982_585
assert sum(1 for _ in read_rows(D, "channel_performance_weekly")) == 8_598_520
```

---

### Step 1.2 — The as-of filter

**What** — The single helper every raw read goes through.

**Files** — `ml/data/loader.py`

**Depends on** — 1.1

**Inputs** — every raw table with a `recorded_ts` column

**Implementation**

The rule, from `data_plan.md`:

> x(t₀) = g({rows : `recorded_ts` ≤ t₀}) — **never `event_ts` ≤ t₀**

Implement it as a **sweep**, not a filter, so a caller cannot forget it. Sort once by
`recorded_ts`; admit rows only as the sweep reaches them. `db/derive/build_features.py`
contains a working reference implementation (`AsOfSweep`) — port it rather than rewriting.

```python
class AsOfSweep:
    """Rows become visible only when their recorded_ts is reached."""
    def __init__(self, rows, recorded_index=0):
        self._rows = sorted(rows, key=lambda r: r[recorded_index])
        self._ix, self._pos = recorded_index, 0

    def advance(self, cutoff):
        out = []
        while self._pos < len(self._rows) and self._rows[self._pos][self._ix] <= cutoff:
            out.append(self._rows[self._pos]); self._pos += 1
        return out
```

Two additional rules that are **not** covered by `recorded_ts`:

1. **Effective-dated master data** — `bom`, `sourcing_channels`, `part_plant`,
   `supplier_allocation`, `part_costs`, `supplier_capacity`:
   ```sql
   WHERE effective_from <= t0 AND (effective_to IS NULL OR effective_to > t0)
   ```
2. **The BOM explosion is the exception.** Its validity window must contain the **target
   period**, not t₀. Using today's BOM to explode a 2019 plan silently mis-states every
   historical requirement.

Additionally bound every query by `snapshots.data_cutoff_ts` for the snapshot being built.

**Verify** — reconstruct one channel's trailing-4-week received quantity at a past t₀ both
ways. On the reference world `csv_full_seed1` (outside this repo) the two disagree on **9.8% of channel-weeks**, and the as-of view
is missing **6.9%** of receipt quantity that had physically arrived but was not yet posted.
If they agree exactly, the filter is reading `event_ts`.

That 6.9% is *displaced*, not lost: those units belong to the later week in which they were
posted. `validate.py` asserts both halves separately — `reporting lag displaces receipts`
(reported, ungated) and `channel store conserves ordered/received units` (must be exactly
zero). A pipeline that drops them instead of deferring them passes the first and fails the
second.

---

### Step 1.3 — Node tables

**What** — Load the five persistent node types and build index maps.

**Files** — `ml/data/loader.py`

**Depends on** — 1.2

**Inputs**
```
db/gen_v7/seed_1001/sourcing_channels.csv
    channel_id, supplier_id, site_id, part_id, plant_id, is_approved, ppap_date,
    approval_status, contracted_lead_time_days, transport_mode,
    transport_distance_km, effective_from, effective_to
db/gen_v7/seed_1001/suppliers.csv
    supplier_id, supplier_group_id, country, state, city, supplier_tier,
    supplier_type, business_class, onboarded_date, is_active, msme_flag,
    payment_terms_days
db/gen_v7/seed_1001/parts.csv
    part_id, part_category, material_type, uom, is_critical, criticality_reason,
    standard_lead_time_days, shelf_life_days, is_active, introduced_date,
    discontinued_date
db/gen_v7/seed_1001/plants.csv
    plant_id, bu_id, state, latitude, longitude, capacity_units_per_day,
    commissioned_date, is_active
```

**Implementation**

Feature composition is specified in
[benchmark_specification §2.2](benchmark_specification.md#22-static-feature-composition).
Build a contiguous integer index per node type; `HeteroData` requires it.

Two things that will bite:

- **Exclude `parts.criticality_reason`.** Only `single_source` occurs in the data — the
  column is constant and contributes nothing. See specification §11.1.
- **`supplier_group` has no attribute table.** 225 groups exist as a column value only.
  See **O1** in the specification; pick an approach before Phase 4.

Categorical cardinalities, measured — size the one-hots from these, do not infer them from
the batch you happen to load:
```
suppliers.country          6     parts.part_category        6
suppliers.state           10     parts.material_type       15
suppliers.supplier_tier    2     parts.uom                  2
suppliers.supplier_type    3     plants.state               4
suppliers.business_class   3     sourcing_channels.approval_status  4
                                 sourcing_channels.transport_mode   4
```

**Verify**
```python
# measured on db/gen_v7/seed_1001; identical in gen_v6 (the masters are byte-identical
# across the two worlds -- only the simulated tables differ).
assert len(channel_index) == 16_072
assert len(supplier_index) == 420     # 46 tier-2 + 374 tier-1
assert len(part_index) == 620
assert len(plant_index) == 7
assert len(group_index) == 84         # supplier_group_id, 5 suppliers per group
```

---

### Step 1.4 — Edges and `HeteroData` construction

**What** — Build the heterogeneous graph for one snapshot.

**Files** — `ml/data/loader.py`

**Depends on** — 1.3

**Inputs** — the node tables from 1.3, plus
```
db/gen_v7/seed_1001/supplier_upstream.csv    supplier_id, upstream_supplier_id, part_id,
                                              dependency_type, criticality, is_sole_source,
                                              confidence, known_since
db/gen_v7/seed_1001/bom.csv                  product_id, part_id, qty_per_unit,
                                              scrap_factor, bom_level, parent_part_id,
                                              effective_from, effective_to, is_phantom
db/gen_v7/seed_1001/part_plant.csv           part_id, plant_id, safety_stock_qty, ...,
                                              effective_from, effective_to
db/gen_v7/seed_1001/alternate_sources.csv    part_id, supplier_id, plant_id,
                                              qualification_status, ...
db/gen_v7/seed_1001/logistics_lanes.csv      lane_id, origin_site_id, dest_plant_id,
                                              transport_mode, via_checkpoint, ...
db/gen_v7/seed_1001/po_lines.csv             po_line_id, po_id, part_id, channel_id, ...
```

**Implementation**

Ten forward relations, each also added in reverse → **R = 20**. Cardinalities from
[specification §2.3](benchmark_specification.md#23-edge-types-with-measured-cardinality).

```python
from torch_geometric.data import HeteroData

d = HeteroData()
d['channel'].x = channel_x                      # [16072, 12]
d['supplier'].x = supplier_x                    # [902, 27]
...
d['channel', 'sourced_from', 'supplier'].edge_index = ...   # [2, 16072]
d['supplier', 'rev_sourced_from', 'channel'].edge_index = ...
```

Filter `bom` and `part_plant` to rows valid **at the target period** and **at t₀**
respectively (step 1.2). Exclude `bom.is_phantom = 'true'` — phantom assemblies pass through
without being stocked.

`po_line` nodes are snapshot-specific: include only lines open at t₀ (created and recorded by
t₀, not yet fully received). ~15k of the 1,169,215, not all of them.

**Verify**
```python
assert d['channel','sourced_from','supplier'].edge_index.shape[1] == 16_072
assert d['supplier','depends_on','supplier'].edge_index.shape[1] == 260   # supplier_upstream
assert d['part','used_in','product'].edge_index.shape[1] <= 4_000         # bom, non-phantom
assert d['po_line'].num_nodes < 50_000                                    # not 982,585
```

---

### Step 1.5 — Snapshot cache

**What** — Materialise each snapshot once; never re-parse CSVs in the training loop.

**Files** — `ml/data/cache.py`

**Depends on** — 1.4

**Inputs** — `db/gen_v7/seed_1001/snapshots.csv`
(`snapshot_id`, `as_of_ts`, `data_cutoff_ts`, `horizon_days`, `feature_spec_version`,
`dataset_version`, `label_version`, `code_commit`, `created_ts`)

**Implementation**

**83 snapshots**, 2016-07-04 → 2025-12-08, all with `horizon_days = 90`.

Serialise each to `ml/artifacts/cache/{dataset_version}/{snapshot_id}.pt` with
`torch.save`. Key the cache directory on `dataset_version` **and** `feature_spec_version` so
a feature change cannot silently reuse stale tensors.

Rebuilding all 83 for a world is the expensive step — budget for it once, then iterate.

**Verify** — cache directory holds 83 files; loading one and re-deriving it from CSV gives
identical tensors (`torch.equal`).

---

## Phase 2 — Sequence assembly

### Step 2.1 — Per-node `[T × d]` tensors

**What** — Turn the weekly stores into padded, masked sequence tensors.

**Files** — `ml/data/sequences.py`

**Depends on** — 1.5

**Inputs**
```
db/gen_v7/seed_1001/channel_performance_weekly.csv
    channel_id, week_start, qty_ordered, qty_received, is_active_week,
    fill_rate, fill_rate_last4, fill_rate_last13, fill_rate_last52,
    lead_time_actual_days, lead_time_ratio, otd_rate_last13, ack_gap_ratio,
    revision_count, load_ratio, active_weeks_in_52, days_since_last_short,
    reporting_lag_days, weeks_since_last_activity, weeks_since_last_receipt
db/gen_v7/seed_1001/supplier_performance_weekly.csv
    ... same 18 features + active_channel_count
    (reporting_lag_days is always NULL at supplier grain -- see spec 5.1)
db/gen_v7/seed_1001/calendar.csv
    date, plant_id, is_working_day, shift_count, is_holiday, is_shutdown,
    month_of_year, regime_flag, ...
```

**Implementation**

Target shape per persistent node: `[52, 38]`, the trailing 52 weeks ending at t₀, plus a
`[52]` `is_active_week` mask carried alongside. d = 38 is composed in
[specification §3.3](benchmark_specification.md#33-per-timestep-feature-vector).

> **Columns renamed in Phase C — `_Nw` → `_lastN`.** `fill_rate_13w` is now
> `fill_rate_last13`, and so on. The rename is not cosmetic: these average the last N weeks
> the channel *ordered*, not the last N calendar weeks, and at this cadence `_last13` spans a
> median of 85 calendar weeks. `staleness_days` is **gone**, replaced by `reporting_lag_days`
> and `weeks_since_last_activity`. See [specification §3.1](benchmark_specification.md#31-input).

Measured facts that drive the implementation:

| | |
|---|---|
| Channels present | **15,334** of 16,072 — 738 never traded (cold-start slice, 4.59%) |
| Distinct weeks | 535 (`gen_v7`, 2016-01-04 → 2026-03-30); `gen_v6` also 535 |
| Weeks per channel | 535 for every channel — the store is a complete contiguous panel |
| Channels with < 52 weeks | 0 |

**Left-pad** short sequences to 52 with zeros and carry a validity mask. Because the
convolution in Phase 3 is causal, left padding cannot leak — a padded position is only ever a
*past* input to a real one.

**Verify**
```python
X, mask = build_sequence("S0001-P00001-PL01", t0)
assert X.shape == (52, 38)          # d_in = 38 since Phase C, not 31
assert mask.dtype == torch.bool and mask.sum() <= 52
```

---

### Step 2.2 — Missing-value masking

**What** — Give every nullable column a paired missing-indicator; never impute.

**Files** — `ml/data/sequences.py`

**Depends on** — 2.1

**Inputs** — as 2.1

**Implementation**

**A channel is idle in four weeks out of five.** Null rates are in
[specification §3.2](benchmark_specification.md#32-the-sequence-is-sparse--this-dominates-the-implementation);
`fill_rate` and `ack_gap_ratio` are 81.4% null, `load_ratio` 81.8%, the lead-time pair 72.6%.

Thirteen nullable columns → thirteen indicator channels. Fill the value channel with 0.0
**and** set the indicator to 0, so the model can distinguish "no activity" from "activity,
value zero".

```python
NULLABLE = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52",
            "lead_time_actual_days", "lead_time_ratio", "otd_rate_last13",
            "ack_gap_ratio", "load_ratio", "days_since_last_short",
            "reporting_lag_days", "weeks_since_last_activity",
            "weeks_since_last_receipt"]
```

**Mean-imputing `fill_rate` is the single most damaging thing you can do here** — it
fabricates 82% of the activity in the sequence.

**`weeks_since_last_receipt` NULL means *never received*, not "long ago".** Zero-filling it
merges a channel that has never taken a delivery with one that took its last delivery this
week — the two extremes of the same axis.

**Two masks, and they are not the same thing.** `is_active_week` marks a real row in which
nothing was ordered; the padding mask marks positions before the channel's first week where
no row exists. Exclude padding from the loss; **do not** exclude inactive weeks, which are
observations. See [specification §3.5](benchmark_specification.md#35-variable-length-sequences).

The rolling columns (`fill_rate_last4/13/52`, `otd_rate_last13`) are the dense signal; the
instantaneous ones are a sparse event stream. Expect the model to lean on the rolling ones —
and remember they are long-run averages, not recent ones.

**Verify** — for a known idle week, the `fill_rate` channel is 0.0 **and** its indicator is
0, and `is_active_week` is 0 while the padding mask is 1. Across a full batch, indicator
means should reproduce §3.2 (e.g. `fill_rate` indicator mean ≈ 0.181).

---

### Step 2.3 — Normalisation

**What** — Standardise numeric channels using statistics from training folds only.

**Files** — `ml/data/sequences.py`

**Depends on** — 2.2, 6.1 (fold definition)

**Inputs** — as 2.1

**Implementation**

Fit mean and standard deviation on the **training fold only** and persist them alongside the
checkpoint. Fitting on all data leaks the future through the scaler — a subtle leak that
`validate.py` cannot catch because it never sees your tensors.

Compute statistics over **valid positions only** (mask = 1). Including padded and null
positions drags every mean toward zero and makes the scale meaningless.

Skewed columns (`qty_ordered`, `qty_received`, `lead_time_actual_days`,
`days_since_last_short`) benefit from `log1p` before standardising —
`sourcing_channels.contracted_lead_time_days` alone spans 4 to 206.

**Verify** — on the training fold, each standardised channel has mean ≈ 0 and sd ≈ 1 over
valid positions. On the **test** fold the means will *not* be 0 — that is correct and is the
point.

---

## Phase 3 — Temporal encoder

### Step 3.1 — Dilated causal TCN

**What** — Encode each persistent node's 52-week sequence to a 64-d state.

**Files** — `ml/models/tcn.py`

**Depends on** — 2.3

**Inputs** — `[B, 52, 38]` tensors from Phase 2

**Implementation**

Full specification in
[benchmark_specification §3.4](benchmark_specification.md#34-architecture--dilated-causal-tcn).

| | |
|---|---|
| Kernel | 2 |
| Dilations | 1, 2, 4, 8, 16, 32 |
| Layers | **6** (not 5 — see below) |
| Receptive field | 64 weeks ≥ the 52-week window |
| Hidden | 64 |
| Parameters | ≈ 48,640 |

`model_plan.md` specifies 5 layers, giving a 32-week receptive field. **Use 6.** With 5, the
earliest 20 weeks of the input window are unreachable by the top layer.

Causality is enforced by **left-padding only**, then trimming the right:
```python
pad = (kernel - 1) * dilation
z = F.pad(z, (pad, 0))          # left only
z = conv(z)[:, :, :T]           # trim any overhang
```
A symmetric `padding=` argument in `nn.Conv1d` will silently make the model non-causal and
leak the future. This is the most common bug in TCN implementations and it produces
suspiciously good results — see [Troubleshooting](#result-looks-too-good).

Take the **final position only**: `s_c = z[:, :, -1]` → `[B, 64]`.

**Verify** — the causality unit test. Perturb the input at the last timestep; the output at
every earlier position must be unchanged.
```python
x2 = x.clone(); x2[:, -1, :] += 100.0
assert torch.allclose(model.all_positions(x)[:, :, :-1],
                      model.all_positions(x2)[:, :, :-1])
```

---

## Phase 4 — Graph encoder (SHARE)

### Step 4.1 — Port or adapt v3's encoder

**What** — Decide whether HADES v3's `rgcn_attn_encoder.py` can be reused.

**Files** — `ml/models/share.py`

**Depends on** — 3.1

**Inputs** — v3 source, if available

**Implementation**

The **mechanism** is unchanged from v3 — basis decomposition with B = 10 plus one
relation-blind attention scorer. The **schema is not**. What must change:

| | v3 | v4 |
|---|---|---|
| Node types | v3's set | `channel`, `supplier`, `part`, `plant`, `supplier_group`, `po_line` |
| Relations R | v3's set | **20** (10 forward + 10 reverse) |
| Node input | v3 features | persistent nodes enter with the TCN state **s**_c; `po_line` with a linear projection of 18 current-state features |

The basis-decomposition and attention code is **relation-count agnostic** and should port
directly. The parts that will not port: node-type registration, the input projection layer,
and anything that assumes a fixed node set across snapshots (this graph churns —
`po_line` nodes appear and vanish).

**If v3 source is unavailable**, `torch_geometric.nn.RGCNConv` with `num_bases=10` gives the
basis decomposition; the relation-blind attention has to be written by hand — it is not a
standard PyG layer.

**Verify** — parameter count within 5% of the ≈ 45,384 per layer computed in
[specification §4](benchmark_specification.md#4-graph-encoder--share). A large discrepancy
means the basis decomposition is not active and you have per-relation weights.

---

### Step 4.2 — Layer stack and readout depth

**What** — Assemble the layers and make readout depth a per-task parameter.

**Files** — `ml/models/share.py`

**Depends on** — 4.1

**Inputs** — `HeteroData` from 1.4, node states from 3.1

**Implementation**

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}_0^{(l)}\mathbf{h}_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \alpha_{ij}\, \mathbf{W}_r^{(l)} \mathbf{h}_j^{(l)}\right)$$

**Return all intermediate representations** — h⁰, h¹, h², h³ — not just the last. Each head
selects its own depth, and gate **G7** requires h⁰ to be available as the ablation.

```python
def forward(self, data):
    h = {nt: self.input_proj[nt](data[nt].x) for nt in data.node_types}
    states = [h]                       # h0 = no message passing at all
    for layer in self.layers:
        h = layer(h, data.edge_index_dict)
        states.append(h)
    return states                      # [h0, h1, h2, h3]
```

**Readout depth must be derived, not inherited.** v3 fixed delay→h¹, shortage→h³, impact→h⁴,
but its own Layer 3 work found h⁴ was the *worst* depth for both latent states that passed
the gate. Sweep h¹…h³ per task on `mid` and record the result (**O6**).

**Verify** — `len(states) == n_layers + 1`, and `states[0]` is identical to the input
projection (no message passing has occurred).

---

## Phase 5 — Heads

### Step 5.0 — Staleness gating layer

**What** — Learned per-feature trust decay on `reporting_lag_days`.

**Files** — `ml/models/staleness.py`

**Depends on** — 2.2

**Inputs** — `reporting_lag_days` from `channel_performance_weekly`; on the reference world `csv_full_seed1`
**median 1.3 days, P90 5.7, P99 14, max 96**.

> **This step used to read `staleness_days`, and that column is gone.** It conflated posting
> lag with channel inactivity: 90% of its rows were empty weeks, so its "median 12, P90 838"
> described how long channels had been *idle*, not how late records were posted. (The "never
> exceeds 6 days" figure this note used to quote for the genuine lag was itself an artefact of
> a builder bug, now fixed — the real range is 0–96 days, P90 5.7.) Feed the gate
> `reporting_lag_days`; feed
> `weeks_since_last_activity` in as a **plain feature only** — never into the gate, because a
> quiet channel is giving you a true reading, not a stale one
> ([specification §5.2](benchmark_specification.md#52-inactivity-is-a-feature-not-a-trust-signal)).

**Implementation**

$$\tilde{x}_i = g_i \cdot x_i^{\text{obs}} + (1-g_i)\cdot x_i^{\text{prior}}, \qquad g_i = \exp\big(-\max(0,\ w_i \Delta t_i + b_i)\big)$$

```python
class StalenessGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))     # start at no decay
        self.b = nn.Parameter(torch.zeros(d))
        self.prior = nn.Parameter(torch.zeros(d)) # learned fallback per feature
    def forward(self, x, dt):                     # x [B,T,d], dt [B,T,1]
        g = torch.exp(-F.relu(self.w * dt + self.b))
        return g * x + (1 - g) * self.prior
```

Three rules: **per feature** (not per node — `country` never changes); **continuous** (a hard
30-day cutoff makes 29 and 31 days behave completely differently); and **Δt also passed as a
plain feature**, because staleness decides trust *and* may itself be predictive.

Apply to the **time-varying** channels only — never to static node features.

Note the range: `reporting_lag_days` spans **0–96 days**, P50 1.3, P90 5.7, P99 14.
Initialise `w` for that scale — one sized for the old 0–3000 day range leaves `g ≈ 1`
everywhere and the gate learns nothing. The distribution is heavy-tailed, not bounded: an
earlier revision of this line said the range was tiny (0–6 days) and that `log1p(dt)` was
unnecessary. That range was a measurement artefact; with a 96-day max against a 1.3-day
median, `log1p(dt)` is now worth trying rather than ruled out.

`dt` is NULL on weeks with nothing posted (53.5% of rows). Pass its missing-indicator through
and leave `x` untouched where it is missing — an absent lag is not a lag of zero.

**Verify** — at `dt = 0`, `g == 1` and the layer is the identity. As `dt → ∞`, output → the
learned prior. Initialising `w = 0` makes the layer start as a no-op, so it cannot hurt early
training.

---

### Step 5.1 — Quantile head (capacity strain)

**What** — 9 outputs: 3 quantiles × 3 horizons, monotone by construction.

**Files** — `ml/models/heads.py`

**Depends on** — 4.2, 5.0

**Inputs** — `db/gen_v7/seed_1001/training_labels.csv` filtered to
`task = 'capacity_strain'` — **1,005,609 rows**, `entity_type = 'channel'`, `label_value` ∈
[0.000, 1.000], median 1.000, **31.7% censored**

**Implementation**

Predict the lowest quantile, then non-negative increments:
```python
q10 = self.lin_q10(h)                      # [B, 3] one per horizon
q50 = q10 + F.softplus(self.lin_d1(h))
q90 = q50 + F.softplus(self.lin_d2(h))
```
This makes monotonicity **structural**. Sorting post-hoc hides the pathology instead.

```python
def pinball(y, yhat, q):
    d = y - yhat
    return torch.maximum(q * d, (q - 1) * d).mean()
```

Days-to-exceed is **derived, not learned**: `DTE = min{h : q50(h) >= 1}`, linearly
interpolated on the 30/60/90 grid.

> **O3 — the label is a fill rate, not a utilisation.** `label_value` caps at 1.000, whereas
> `model_plan.md` §5 describes utilisation, which can exceed 1 and where ">1 means trouble"
> is the whole reading. `revealed_capacity_monthly.csv` supplies
> `capacity_utilisation_observed` and `revealed_capacity_est` if utilisation is wanted.
> **Decide before writing this head; the two targets are not interchangeable.**

All 83 snapshots carry `horizon_days = 90`, so the 30- and 60-day targets must be derived
from inside the 91-day window (specification §11.5).

**Verify** — zero quantile crossings on held-out data (gate G6). Pinball loss at q=0.5 should
be lower than at q=0.1 and q=0.9 on a well-fit model.

---

### Step 5.2 — Binned CDF head (fill rate)

**What** — A 20-bin distribution over [0,1] with dedicated endpoint bins.

**Files** — `ml/models/heads.py`

**Depends on** — 4.2, 5.0

**Inputs** — `training_labels` where `task = 'fill_rate'` — **1,284,907 rows**,
`entity_type = 'po_line'`, **12.3% censored**

**Implementation**

Bins: `b0 = {0}`, `b1 = (0, 0.05]`, …, `b18 = (0.95, 1)`, `b19 = {1}`.

Measured label distribution over the **uncensored** population (1,127,491 rows): **91.5% at
exactly 1.0**, **1.8% at exactly 0.0**, 6.7% interior. Bins 0 and 19 exist to hold those point masses; a Gaussian head is confidently
wrong at both ends, which are the two outcomes a planner cares about.

```python
def crps_loss(logits, y_bin_index):
    pi = F.softmax(logits, dim=-1)             # [B, 20]
    F_hat = torch.cumsum(pi, dim=-1)
    step = (torch.arange(20) >= y_bin_index.unsqueeze(1)).float()
    return ((F_hat - step) ** 2).sum(-1).mean()
```

CRPS, not cross-entropy: cross-entropy treats bins as unordered, so predicting bin 2 when the
truth is bin 19 costs the same as predicting bin 18. CRPS penalises by distance and is a
*proper* scoring rule.

> ### `label_censored` is correct on this table — filter on it directly
>
> **This block previously told you the opposite.** It was right at the time; the defect is
> fixed in the generator and the workaround it prescribed is now itself a bug. See
> [specification §11.7](benchmark_specification.md#117-training_labelslabel_censored--fixed-was-unusable-for-the-fill-rate-task).
>
> ```python
> settled  = labels[~labels.label_censored]   # 15.1% censored at full; usable target
> censored = labels[labels.label_censored]    # genuinely still open at window end
> ```
>
> **Do NOT recompute censoring from `grn_lines.is_final_receipt`.** That was the old advice.
> 7.09% of PO lines settle with no final receipt at all — the supplier shipped nothing, or the
> line timed out short — and are recorded by a `po_line_revisions` short-close row instead.
> Filtering on `is_final_receipt` alone marks every one of them censored and reintroduces a
> milder version of the original defect.
>
> **The one thing you must not do yourself:** `fill_rate` rows are deliberately **absent** for
> buyer-cancelled lines (28.1% of short closes). The supplier was never given the chance to
> deliver, so the outcome is undefined, not zero. If you build the population from `po_lines`
> rather than reading `training_labels`, exclude them explicitly:
> ```python
> cancelled = {r["po_line_id"] for r in read_rows(D, "po_line_revisions")
>              if r["reason_code"] == "buyer_cancellation"}
> ```
> Scoring those as `fill = 0` teaches the model that the supplier failed when Rane changed its
> mind.
>
> Treat `censored` rows as interval-censored (true value ∈ [observed, 1]) with a CRPS that only
> penalises predicted mass **below** the observed value, or drop them.

Downstream consumers use **E[f] = Σ π_b · mid(b)**, and report
`qty_ordered × (1 − E[f])` — a quantity, not a probability, so it adds correctly across the
four-level rollup.

**Verify** — first, that the target is not constant:
```python
assert settled.label_value.nunique() > 100      # 8,661 at full
assert settled.label_value.std() > 0.05         # 0.1609 at full
assert 0.05 < (settled.label_value < 0.999).mean() < 0.20
```
`validate.py`'s `check_labels` asserts the same property for every task on every run, so a
regression here fails the dataset build rather than surfacing as a suspiciously good model.
Then, on held-out data, predicted P(bin 19) ≈ the observed rate of `fill_rate = 1.0` among
**settled** rows. Band coverage: the P10–P90 interval contains 80% ± 5pp of outcomes
(gate G5).

---

### Step 5.3 — Discrete-time hazard head (arrival timing)

**What** — 12 conditional hazards, one per forward week.

**Files** — `ml/models/heads.py`

**Depends on** — 4.2, 5.0

**Inputs** — `training_labels` where `task = 'arrival_week'` — **1,310,934 rows**,
`entity_type = 'po_line'`, `label_value` ∈ 1…13, **8.4% censored**, `censor_time` carries
weeks observed so far

**Implementation**

**Independent sigmoids, not a softmax.** λ_w is a conditional probability, not a
distribution over w:
```python
lam = torch.sigmoid(self.lin(h))            # [B, 12]
S = torch.cumprod(1 - lam, dim=-1)          # survival
p_T = lam * torch.cat([torch.ones(B,1), S[:, :-1]], dim=-1)
```

$$\mathcal{L} = -\sum_\ell \sum_{w=1}^{\min(T_\ell,\,C_\ell)} \Big( \mathbb{1}[T_\ell = w]\log\hat\lambda_w + \mathbb{1}[T_\ell > w]\log(1-\hat\lambda_w) \Big)

**Censoring is the point here, not an edge case.** An open PO contributes
`log(1 - λ̂_w)` for every week it has survived — real information. Dropping censored rows
keeps only orders that already landed and makes every forecast optimistic.

```python
# w < T: survived.  w == T and not censored: the event.
surv = (torch.arange(12) < (T - 1).unsqueeze(1))
event = (torch.arange(12) == (T - 1).unsqueeze(1)) & (~censored).unsqueeze(1)
loss = -(event * torch.log(lam + 1e-8) + surv * torch.log(1 - lam + 1e-8)).sum(-1).mean()
```

**Measured arrival distribution:** w=1 **82.2%**, w=2 4.5%, w=3 3.1%, w=4 2.5%, w=5 2.1%,
thin tail to w=13. The mass at w=1 dominates, so **gate G4 requires beating a
"always predict w=1" baseline** — otherwise the head has learned the marginal and nothing
else.

**Verify** — `p_T.sum(-1) + S[:, -1] ≈ 1.0` (arrival probabilities plus survival past week 12
sum to one). Weekly calibration error < 5pp for w = 1…6 (gate G5).

---

## Phase 6 — Training loop

### Step 6.1 — Time-based folds

**What** — Splits by time, never randomly.

**Files** — `ml/train/folds.py`

**Depends on** — 1.5

**Inputs** — `snapshots.csv` — 83 snapshots,
2016-12-26 → 2025-09-22

**Implementation**

Eight rolling origins, listed in
[specification §9.2](benchmark_specification.md#92-snapshots-and-rolling-origins). A random
split leaks the future and will produce results that cannot be reproduced in deployment.

Two exclusions:
- **`regime_flag = 'covid'` rows out of primary training** (specification §9.4). Report on
  them separately.
- **Snapshots before 2019-01-01** are outside the modelling window; 2016–2018 is the
  held-out sanity block.

**Verify** — for every fold, `max(train.snapshot_date) < min(test.snapshot_date)`. Assert it;
do not eyeball it.

---

### Step 6.2 — Seeding and determinism

**What** — Make a run reproducible and the reproduction floor measurable.

**Files** — `ml/train/loop.py`

**Depends on** — 6.1

**Inputs** — none

**Implementation**
```python
torch.manual_seed(s); np.random.seed(s); random.seed(s)
torch.use_deterministic_algorithms(True)
```
Run **≥ 5 model-init seeds × 8 time folds** per configuration. The spread across those 40
runs *is* the reproduction floor (see [Glossary](#glossary)); a claimed improvement smaller
than it is not an improvement.

**Verify** — the same seed and fold reproduce the metric bit-for-bit. If not, find the
nondeterminism before measuring anything.

---

### Step 6.3 — Loop and checkpointing

**What** — Train, checkpoint, and stamp every artefact.

**Files** — `ml/train/loop.py`

**Depends on** — 5.1–5.3, 6.2

**Inputs** — cached snapshots from 1.5

**Implementation**

Stamp every checkpoint and prediction file with all five identifiers from
`snapshots.csv`: `dataset_version` (`hades-v4-synth-1.0`), `feature_spec_version`
(`1.0`), `label_version` (`1.0`), `model_version`, `code_commit`. This is what supports *"if
this had been deployed in Jan-2023, here is what it would have predicted"* — unverifiable
without them (gate G11).

Write predictions in the `model_outputs` table's shape:
`output_id, snapshot_id, model_name, model_version, entity_type, entity_id, horizon_days,
point_estimate, p10, p90, distribution, confidence_basis, evidence_strength, created_ts`.
The generator writes this table **empty on purpose** — it is yours to fill.

Early stopping on validation loss from the *fold's own* validation slice, never the test
slice.

**Verify** — `model_outputs` joined to `training_labels` on
`(snapshot_id, entity_id, task)` returns one row per prediction with its actual. This join is
the backtest, and it is only safe because predictions were kept out of the feature store.

---

## Phase 7 — Baselines

> **Build these before the neural model.** Specification §7 requires the comparison to exist
> first so it cannot be retrofitted. They are also cheap, and they surface data problems the
> neural model would only obscure.

### Step 7.1 — Rolling-statistics baselines (B2, B3, B4)

**What** — Non-learned baselines read straight off the derived stores.

**Files** — `ml/baselines/rolling.py`

**Depends on** — 1.2

**Inputs**
```
channel_performance_weekly.csv   fill_rate_last13, fill_rate_last52
plan_drift_features.csv          drift_ratio, horizon_days, plant_id, regime_flag
training_labels.csv                      task, label_value, label_censored
```

**Implementation**

- **B2 (fill, capacity):** point prediction = `fill_rate_last13` at t₀; band = empirical spread
  of the last 52 weekly `fill_rate` values.
- **B3 (timing):** always predict w = 1. Covers **82.2%** of observed arrivals.
- **B4 (drift):** bucket `drift_ratio` by (horizon band, plant, product family) and read
  percentiles directly. **This is UC1's incumbent**, per `data_plan.md` — LightGBM is an
  escalation only if buckets prove thin. `db/derive/build_features.py::drift_quantiles` is a
  working implementation to port.

**Verify** — B3 achieves C-index ≈ 0.5 by construction (no discrimination) but a strong
Brier score. If the neural head cannot beat it on C-index, gate G4 fails.

---

### Step 7.2 — LightGBM on flat features (B5)

**What** — The baseline the neural model must beat.

**Files** — `ml/baselines/lgbm.py`

**Depends on** — 1.4, 2.2

**Inputs** — the same features, with graph structure flattened to scalars

**Implementation**

Flatten the network: `channels_per_supplier` (median 12, P90 56, max 168),
`channels_per_part` (median 2), `suppliers_per_group` (median 4), degree counts,
`is_sole_source` from `supplier_upstream`. Use the **same losses** — pinball and a
multiclass proxy for CRPS — so the comparison measures the same thing.

> **What this actually measures.** Flattening the network into scalars *is* the baseline.
> Once structure is collapsed into a handful of columns the graph has nothing left to
> contribute, so **a tie proves the flattening was adequate, not that the graph is useless.**
> Keep both labelled as what they are.

**Verify** — trains in under two minutes on `mid`. If it takes much longer, the feature
matrix is being rebuilt per fold instead of cached.

---

### Step 7.3 — h⁰ ablation (B6) — mandatory

**What** — Measure what the graph contributes over input quality alone.

**Files** — `ml/baselines/ablation.py`

**Depends on** — 4.2, 5.1–5.3

**Inputs** — the full pipeline

**Implementation**

Run the complete model with **zero message-passing layers** — node features and the temporal
encoder, nothing else. Step 4.2 returns `states[0]` for exactly this purpose.

Report per task: `(full − h⁰) / (full − random)`.

HADES v3 measured the graph's contribution on the delay task at **~10% of above-chance
signal** (raw features 0.7516 AUC, full encoder 0.7802). Input quality dominated architecture
by a wide margin.

**Verify** — h⁰ must be strictly worse than the full model on at least one task, or the
graph layers are not connected. Gate **G7** requires this number published per task; there is
no minimum value.

---

## Phase 8 — Evaluation

### Step 8.1 — Metric suite

**What** — Implement the metrics in specification §8.

**Files** — `ml/eval/metrics.py`

**Depends on** — 6.3

**Inputs** — `model_outputs` joined to `training_labels`

**Implementation**

CRPS and band coverage (fill), C-index and weekly calibration (timing), pinball and
event-recall-with-lead-time (capacity), WMAPE per horizon band (demand), MAE in units and
days (shortage).

**No accuracy figure for any task** (gate G12). At a 3.9% shortage base rate a
constant "no" scores 96% and is worthless.

**O4 — "significant capacity event" is undefined.** The lead-time recall metric cannot be
computed until it is defined with Rane. Do not invent a threshold.

**Verify** — CRPS of a perfect forecast is 0. For a uniform 20-bin forecast the value
depends on where the truth sits: **3.325 averaged over all 20 bins**, 6.175 when the truth is
bin 19. Assert against the bin-specific value, not a single constant.

---

### Step 8.2 — Rolling-origin backtest

**What** — Run all folds, all seeds, and report spread.

**Files** — `ml/eval/backtest.py`

**Depends on** — 6.1, 8.1

**Inputs** — cached snapshots, trained checkpoints

**Implementation**

≥ 5 model-init seeds × 8 folds = 40 runs per configuration. **Report per-fold spread every
time, never just the mean.** v3's hardest lesson: one fixed threshold produced recall from
32 to 94 out of 100 depending only on which world it ran in, and the mean of 64.6 described
nothing.

Calibration fitted on earlier folds, evaluated only on later ones.

**Verify** — the output table has a row per (fold, seed), and the summary reports min /
median / max. A summary with only a mean fails gate G8.

---

## Phase 9 — Simulation

### Step 9.1 — Stock roll-forward

**What** — Monte Carlo shortage per (part, plant, week).

**Files** — `ml/sim/montecarlo.py`

**Depends on** — 5.2, 5.3 (or baselines, for early development)

**Inputs**
```
inventory_position_weekly.csv
    part_id, plant_id, week_start, as_of_date, qty_on_hand, qty_blocked,
    qty_reserved, qty_in_transit, qty_available, open_po_qty, safety_stock_qty,
    days_of_supply, consumption_4w, consumption_13w, inter_plant_in_4w,
    inter_plant_out_4w, staleness_days, recorded_ts
    -- NOTE: this staleness_days is NOT the one removed from
    -- channel_performance_weekly. Here it is the age of the inventory reading
    -- feeding the row, is bounded (P50 14, P90 28, max 126 days), and carries
    -- one meaning only. It is fine to use as-is.
part_demand_weekly.csv
    part_id, plant_id, week_start, as_of_date, gross_requirement_p50,
    gross_requirement_p90, horizon_days, driving_products
```

**Implementation**

$$I_w^{(n)} = I_{w-1}^{(n)} + A_w^{(n)} - C_w^{(n)}, \qquad \text{Short}_w^{(n)} = \max\big(0,\ \text{safety\_stock} - I_w^{(n)}\big)$$

Shortage is measured against **safety stock**, not zero — hitting zero is already a line
stop and the warning must arrive earlier.

Draw per pass: f from the binned head, T from the hazard head, C from the drift quantiles.
N = 1000. **Keep all N sample paths** — the UC6 rollup needs them.

**Never sum P90s.** P90(Σ Short_p) ≠ Σ P90(Short_p); the right side assumes everything goes
wrong at once. Carry paths up through every level and take the percentile **at the top**.

**Verify** — with fill fixed at 1.0 and timing fixed at the promise date, the simulation
reproduces `inventory_position_weekly.qty_available` exactly. Any drift is a bug in the
roll-forward, not in the sampling.

---

### Step 9.2 — Gaussian copula

**What** — Correlated sampling across channels sharing structure.

**Files** — `ml/sim/copula.py`

**Depends on** — 9.1

**Inputs** — `suppliers.supplier_group_id`, `supplier_upstream`,
`logistics_lanes.via_checkpoint`

**Implementation**

$$z_g^{(n)} \sim \mathcal{N}(0,1) \text{ per group } g, \qquad f_\ell^{(n)} = \Pi_\ell^{-1}\Big(\Phi\big(\rho\, z_{g(\ell)}^{(n)} + \sqrt{1-\rho^2}\,\varepsilon_\ell^{(n)}\big)\Big)$$

Π⁻¹ is the inverse CDF from the binned head; ρ is estimated from historical co-occurrence of
shortfalls within each group. Estimate ρ **from data before t₀**, not from the generator's
parameters — a modelling team would never see those.

**Verify (gate G10)** — the realised within-group correlation of simulated shortfalls matches
`db/ground_truth.realised_correlation_matrix()` within ±0.05.
```python
import sys; sys.path.insert(0, "db")
import ground_truth
labels, M = ground_truth.realised_correlation_matrix()
```
This validates the *simulation*, not the generator.

> **Read this before quoting any exposure number.** On the reference world `csv_full_seed1`, assuming
> independence understates P90 production exposure by only **1.4%**, because a product's
> median 16 BOM parts come from **19 distinct supplier groups**. The copula is correct and
> gate G10 will pass, but **no result from this dataset supports the "independence
> understates exposure" business claim** (specification §11.6). Reproduce with
> `db/prove_claims.py`.

---

## Phase 10 — Optimisers

### Step 10.1 — Delivery schedule LP

**What** — Weekly receipt quantities minimising holding, ordering and freight.

**Files** — `ml/opt/schedule_lp.py`

**Depends on** — 1.2 (no learned model needed)

**Inputs**
```
part_demand_weekly.csv    gross_requirement_p50, gross_requirement_p90
part_plant.csv                    safety_stock_qty, min_order_qty, lot_size
supplier_contracts.csv            moq, lot_size, min_volume_commitment, max_volume_cap
logistics_lanes.csv               standard_transit_days, distance_km, transport_mode
calendar.csv                      is_working_day, is_shutdown
```

**Implementation**

$$\min_{x} \sum_w \Big( c_{\text{hold}} I_w + c_{\text{order}} y_w + c_{\text{freight}} \lceil x_w / \text{truck\_cap}\rceil \Big)$$

subject to stock balance, `I_w ≥ safety_stock`, `x_w ≥ MOQ·y_w`, `x_w ≡ 0 (mod lot_size)`,
`Σ x_w =` monthly requirement, `y_w ∈ {0,1}`.

The binary `y_w` is what makes this integer rather than a plain LP. Small enough to solve
exactly in milliseconds.

**Verify** — on a part with no capacity constraint and zero holding cost, the solver orders
exactly the requirement in the last feasible week. If it orders earlier, the holding cost is
not wired up.

---

### Step 10.2 — Allocation: enumeration, then MILP

**What** — Recommended supplier split with the constraints that make it actionable.

**Files** — `ml/opt/allocation.py`

**Depends on** — 9.1

**Inputs**
```
supplier_allocation.csv    allocation_pct, effective_from, changed_by, change_reason
alternate_sources.csv      qualification_status, qualification_lead_days,
                              ramp_rate_pct_per_month, cost_delta_pct
sourcing_channels.csv      is_approved, approval_status
supplier_contracts.csv     min_volume_commitment, max_volume_cap, moq, lot_size
tooling.csv                is_transferable, duplicate_exists, transfer_lead_days
part_costs.csv             unit_cost_inr, freight_cost_inr
revealed_capacity_monthly.csv   revealed_capacity_est, evidence_strength
```

**Implementation**

**Phase 1 ships first: enumerate 5 candidate splits**, re-run the Phase 9 simulation under
each, rank by expected shortage cost plus purchase cost. Every number is explainable line by
line.

**Phase 2, MILP,** only when the scenario space outgrows enumeration. `E[Short(x)]` cannot be
embedded in a MILP — fit a piecewise-linear approximation from simulation runs on a grid,
solve, then **re-simulate the chosen solution properly to verify**. The approximation guides
the search; the simulation validates the answer.

> **The constraints are the product, not the objective.** Anyone can write a solver that says
> "move 40% to Supplier B". What makes it usable is encoding why you often cannot: PPAP takes
> months (`alternate_sources.qualification_lead_days`, 45–300 days), the dies sit in Supplier
> A's plant (`tooling.is_transferable`), there are minimum-volume penalties
> (`supplier_contracts`), and B cannot ramp 30% → 45% next week
> (`ramp_rate_pct_per_month`).

Solver: OR-Tools CP-SAT or HiGHS.

**Verify** — a part whose tooling has `is_transferable = false` and `duplicate_exists =
false` must return the incumbent split unchanged, whatever the cost difference. If the
optimiser moves it, the tooling constraint is not bound.

---

## Troubleshooting

### Result looks too good

**Symptom** — a head reaches near-perfect validation on the first run; or the metric is far
better than the LightGBM baseline immediately.

**Likely causes, in order:**

1. **Non-causal convolution.** A symmetric `padding=` in `nn.Conv1d` makes the TCN see the
   future. Run the causality test in step 3.1 — it is three lines and catches this instantly.
2. **Feature read on `event_ts`.** Grep the feature pipeline for `event_ts`. Every hit must
   be justified in writing. Re-run the step 1.2 verification: as-of and naive views must
   differ on ~8.4% of channel-weeks.
3. **Normalisation fitted on all data**, including the test fold (step 2.3).
4. **Label in the features.** `qty_received`, `qty_accepted` and any `grn_lines` aggregate
   are label components. The raw `po_lines.csv` deliberately has no `qty_received` column
   — if it appears in your feature frame, you joined it in.
5. **Random split instead of time split** (step 6.1).

### All-NaN sequences

**Symptom** — loss is NaN from the first batch, or a node's `[52, 31]` tensor is entirely
null.

**Likely causes:**

1. **The channel is one of the 738 with no sequence at all.**
   `channel_performance_weekly` covers 15,333 of 16,072 channels. Filter these out or handle
   cold start explicitly.
2. **Mean-imputing `fill_rate`**, which is null 90.1% of the time — the mean over an empty
   set is NaN. Use the missing-indicator scheme in step 2.2.
3. **Normalisation over padded positions**, giving sd = 0 and a division by zero. Compute
   statistics over valid positions only.
4. **`log1p` on a negative value.** `inventory_transactions.qty` is signed;
   `days_since_last_short` can be null. Clip before transforming.

### Exploding memory on `full`

**Symptom** — OOM when loading, or when building `HeteroData` for a snapshot.

**Likely causes:**

1. **All 1,169,215 `po_line` nodes in one graph.** Only lines *open at t₀* belong in a
   snapshot — about 15k. Check `d['po_line'].num_nodes` (step 1.4 verify).
2. **All 83 snapshots in memory.** Cache to disk (step 1.5) and load one at a time.
3. **`channel_performance_weekly` as one DataFrame.** 5,930,225 rows × 20 columns. Read it
   once, pivot to per-channel arrays, cache, then never read the CSV again in the loop.
4. **`part_demand_weekly` is 8,712,265 rows** — the largest derived store. Only the current
   snapshot's `as_of_date` is ever needed. Filter on read.

### A head will not converge

**Symptom** — loss plateaus immediately, or the head predicts the marginal for every input.

**Per head:**

- **Fill rate, predicting 1.0 for everything** — check this **first**. On a current dataset
  the uncensored target has sd 0.1609 across 8,661 distinct values, so a constant prediction
  means either a stale dataset (regenerate; see specification §11.7) or that you rebuilt the
  population yourself and scored buyer-cancelled lines as `fill = 0` (step 5.2).
- **Fill rate** — predicting bin 19 for everything is also the marginal (91.5% of uncensored
  labels), so a head that has learned nothing and a head that has learned the base rate look
  alike. Check censored rows were handled as intended (step 5.2), and that CRPS is implemented
  on the **cumulative** distribution, not per-bin.
- **Arrival timing** — predicting λ₁ ≈ 0.82 and λ_{2..12} ≈ 0 is the marginal (82.2% at w=1).
  This is what gate G4 exists to catch. Check the survival term is in the loss; without it
  the head only ever sees events and never the "survived another week" signal.
- **Capacity strain** — if all three quantiles collapse to the same value, the softplus
  increments have died. Check the initialisation; increments starting near zero with a
  saturating activation will not recover.
- **All heads** — check the staleness gate is not saturating. Initialise `w = 0` so it starts
  as the identity (step 5.0).

### A baseline beats the neural model

**This is a valid outcome, not a bug.** Specification §7 anticipates it.

**What to do:**

1. **Confirm it is real.** Is the margin larger than the reproduction floor across ≥ 5 seeds
   × 8 folds? A difference inside the floor is not a difference (step 6.2).
2. **Check the h⁰ ablation.** If h⁰ ≈ full model, the graph is contributing nothing and the
   result is consistent — v3 measured the graph at ~10% of above-chance signal on delay.
3. **Then ship the baseline.** Gate G3 says so explicitly. Keep the graph only where it is
   structurally required: multi-tier propagation through `supplier_upstream`, and cold-start
   channels (204 have < 52 weeks of history).
4. **Do not** add layers, widen the hidden dim, or train longer to rescue the comparison.
   Input quality dominated architecture in v3 by a wide margin; spend the effort on features.
5. **Record it.** "LightGBM matched temporal-SHARE on fill rate at one tenth the training
   cost" is a finding worth keeping, not a failure to hide.

### Simulation and ground truth disagree on correlation

**Symptom** — gate G10 fails; realised within-group correlation is far from
`ground_truth.realised_correlation_matrix()`.

**Likely causes:**

1. **ρ estimated on the wrong grain.** Estimate from monthly shortfall per supplier, and
   remove the cross-sectional week effect first — otherwise the market-wide regime factor
   swamps the structural signal and every pair looks correlated.
2. **The copula applied to the wrong marginal.** Π⁻¹ must be the *predicted* CDF for that
   line, not a pooled one.
3. **Groups keyed incorrectly.** `supplier_group_id` lives on `suppliers.csv`, not on
   `sourcing_channels.csv` — join through `channel → supplier → group`.
4. **A missing observation zero-filled.** Check this before the other three. The shortfall
   series is sparse — at `full` only 71.7% of supplier-month bins carry an observation — and
   filling the rest with `0.0` does not mean "unknown", it means *this supplier had no
   shortfall*. That single substitution reported the realised within-group correlation as
   0.246 when it is 0.403. Correlate pairwise-complete, over the bins where **both**
   suppliers were active, and return NaN below a minimum overlap rather than guessing.
   The mirror-image failure appears at low event rates: when almost nothing happens,
   subtracting the cross-sectional week mean makes every series track its negative and
   *every* pair correlates toward 1, unrelated pairs included. Always read within-group
   against unrelated, never on its own (specification §11.6.1, §10.1).

---

## Glossary

**as-of** — Reconstructing state as it was known at a past time t₀, filtering on
`recorded_ts ≤ t₀` and never `event_ts ≤ t₀`. An event that happened on the 5th but was
entered on the 12th was not knowable on the 8th. On this dataset the two views differ on 8.4%
of channel-weeks.

**censoring** — An outcome not yet observed at the end of its window. Dropping censored rows
keeps only orders that already arrived, biasing every forecast optimistic; the hazard head
consumes them as real information about survival. `label_censored` is per task and is
trustworthy on all five: `fill_rate` 15.1%, `arrival_week` 8.9%, `capacity_strain` 32.2%,
`demand_drift` 20.5%, `shortage_qty` 44.3%. It previously encoded "did not reach fill 1.0"
for `fill_rate` and was unusable; that is fixed (specification §11.7). Note that **settled ≠
has a final receipt**: 7.09% of lines settle via a short-close revision with no final receipt
at all.

**short close** — A `po_line_revisions` row cutting a line's qty to what was received, marking
settlement for a line that never got a final receipt. `reason_code` carries the cause:
`supplier_no_ship`, `short_close_timeout`, or `buyer_cancellation`. The last has **no
fill-rate label** — the supplier was never given the chance to deliver (specification
§11.7.2).

**channel** — A `(supplier × part × plant)` triple; the core modelling entity. 16,072 exist,
9,026 currently active. Has a multi-year track record, which is what the temporal encoder
runs over.

**h⁰** — The model with **zero** message-passing layers: node features and the temporal
encoder only. The mandatory ablation that isolates the graph's contribution. v3 measured it
at ~10% of above-chance signal on delay.

**instance node** — A node with no meaningful history: `po_line`. Created, lives weeks, gone.
Gets current-state features only, never a sequence.

**persistent node** — A node with a history worth encoding: `channel`, `supplier`, `part`,
`plant`, `supplier_group`. Gets the temporal encoder.

**regime** — A labelled structural break in `calendar.regime_flag`: `normal` (69.9% of
plant-days), `covid` (15.2%), `chip_shortage` (12.5%), `demonetisation` (2.3%). A regime
shifts the *parameters* of the process, not just the noise, so it must be conditioned on
rather than averaged through. `chip_shortage` is part-scoped — electronic parts only.

**reproduction floor** — The spread of a metric across ≥ 5 model-init seeds × 8 time folds
with no intervention. A claimed improvement smaller than the floor is not an improvement.
Replaces v3's cross-world variance, which does not exist when Rane is one world.

**snapshot** — A t₀ at which features are frozen and a prediction is made. 83 exist, 42 days
apart, each with `horizon_days = 90`. `data_cutoff_ts` is the hard bound: no row with a later
`recorded_ts` may enter.

**strain** — What UC2 predicts *instead of* capacity. Capacity is censored — `delivered =
min(K, ordered)` — so in any month you ordered less than the supplier could make, the data
says nothing about K. On this dataset ~73% of supplier-months are unconstrained and carry no
capacity information at all. Strain (fill decay, lead-time stretch) is observable; capacity is
not.
