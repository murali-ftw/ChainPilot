# ChainPilot / HADES — Synthetic Dataset

A deterministic, leakage-free synthetic ERP world for the HADES architecture,
generated per `Dataset.md` (HADES Synthetic World Specification) against the
schema in `docs/05_Database_Design.md` and the temporal contract in
`architecture/project_HADES.md` §2.6.

## Contents

```
db/
├── schema.sql            DDL only — enums, 19 tables, indexes, constraints
├── generate_dataset.py   the simulation engine (seeded; re-run = identical CSVs)
├── load_data.py          loads schema + CSVs into PostgreSQL, then verifies
├── README.md
└── csv/                  one file per table, FK-safe naming
```

| Table | Rows | Notes |
|---|---|---|
| suppliers | 50 | 6 countries, power-law degree, `reliability_history` display-only |
| components | 150 | each belongs to exactly one supplier; fasteners/electronics reused widely |
| products | 80 | 6 categories |
| product_components | 450 | BOM with validity windows; 4 substitutions + 4 pure additions spread Feb–Nov |
| factories / product_factories | 5 / 119 | per-product capacity, `is_primary` |
| warehouses / inventory | 8 / 194 | current state only |
| inventory_history | ~10,280 | weekly observations, bitemporal (`observed_at`+`recorded_at`) |
| customers | 100 | strategic 15 / standard 70 / low 15 |
| orders / order_items | 1,000 / ~2,490 | seasonal spike Sep–Oct, 5% enterprise orders |
| shipments | ~1,440 | supplier→warehouse replenishment + factory fulfilment |
| shipment_status_history | ~4,460 | full transition log — the delay-label source |
| supplier_temporal_features | 300 | 6 monthly `as_of_date`s × 50 suppliers, windows end at t0 |
| carrier_performance_snapshots | 30 | 5 carriers × 6 t0s |
| graph_snapshots | 6 | t0 = Jul 1 … Dec 1 2024, horizon 14d, counts JSON; `USED_IN` edge count grows 442→446 across snapshots as BOM additions land |
| training_labels | ~1,774 | tasks `delay` / `shortage` / `impact`; ~11% / ~7% / ~4% positive (within Dataset.md Appendix A: 5–15% / 2–8%) |
| risk_scores | 50 | seed rows, `weighted_formula` — **not** model output |

Exact counts vary slightly on regeneration-affecting edits (RNG draw order shifts downstream), but are always internally consistent and re-validated by the generator's own suite — see `Regenerating` below.

## How the world works (causality, not randomness)

Nothing is sampled independently. A latent **stress** value per supplier per
day drives everything observable:

1. **Disruption events** (Chapter 10): a NA trucking strike (May), German
   customs friction (Feb), port congestion hitting sea-freight suppliers
   (Aug–Oct), a factory outage (Jun), and a **hidden upstream polymer
   shortage** (Sep–Dec). Each ramps up to a peak and decays — gradual recovery.
2. Stress raises per-shipment delay probability and lateness magnitude at
   dispatch time → `shipment_status_history` transitions → delay labels.
3. Delayed replenishment shipments postpone warehouse arrivals → stock drifts
   below reorder thresholds → `inventory_history` shortage episodes →
   shortage labels.
4. Seasonal demand (Sep–Oct spike) accelerates draw-down, compounding 2–3.

### The hidden-dependency scenario (Transformer 2's target)

Four suppliers in **different countries supplying different component types**
share an unmodelled upstream polymer plant. There is **no row, no edge, no
column** connecting them — only the correlated degradation it causes:
at the Dec 1 snapshot their mean 90-day on-time rate is well below the fleet
mean (checked and asserted by the generator's validation suite every run,
margin > 0.10). A structural model cannot see this; an embedding-similarity
model can. That gap is the Transformer 2 acceptance test.

Shared port and shared trucking factors are implemented the same way
(members correlated, mechanism never emitted), per Dataset.md Chapter 9.

## Temporal integrity (the leakage contract)

- **Bitemporal history**: `inventory_history` and `shipment_status_history`
  carry `observed_at`/`changed_at` (valid time) *and* `recorded_at` (system
  time), so as-of queries can exclude retroactive corrections.
- **Features end at t0**: every `supplier_temporal_features` window and
  carrier snapshot is computed strictly from events at or before its
  `as_of_date`.
- **Labels start after t0**: every `training_labels.event_at` lies in
  `(t0, t0 + 14d]` — asserted by the generator's validation suite and again
  by `load_data.py` against the loaded database.
- **Eligibility**: delay labels exist only for shipments whose *as-of* status
  (reconstructed from the transition log) is `scheduled` or `in_transit`.
- **Display-only columns**: `suppliers.reliability_history`,
  `shipments.status`, `shipments.delivered_at`, and `inventory.stock_level`
  are current-state values. Reading them as model features is label leakage;
  the schema carries comments saying so.

## Realism audit

A statistical pass over the generated CSVs (duplicate/PK checks, per-supplier-week
delay-outcome mixing, single-feature AUC on the strongest historical signal,
ID-hash leakage check, timestamp-granularity histograms) found the causal
structure sound — no full-row or PK duplicates, no ID-encoded leakage, delay
outcomes are *not* trivially predictable from one historical feature
(in-sample AUC ≈ 0.59–0.63 on `on_time_rate_90d` alone) — but surfaced one
systemic "looks synthetic" tell and three secondary ones, all now fixed in
`generate_dataset.py`:

- **Zero timestamp jitter.** Every recorded-event timestamp in every table
  landed on an exact minute:second, and several "system" columns
  (`created_at`/`computed_at`/`scored_at`) were a single identical instant
  shared by every row in the table — most visibly `inventory_history`, where
  all ~10k rows were stamped at exactly Monday 14:00:00 UTC. Fixed with a
  `J()` jitter helper (0–45 min) applied to every *observed/recorded* event
  timestamp. Defined analytical boundaries — `t0`, BOM validity dates,
  `qualified_at` — are deliberately left clean; those are business dates, not
  recorded ERP events, and jittering them would be the wrong kind of "realistic."
- **Static graph topology across snapshots.** All 8 BOM changes deactivated
  on exactly 2024-07-01 (the first snapshot t0), so `USED_IN` edge count was
  identical at every one of the 6 monthly snapshots. Fixed by spreading 4
  substitutions (Feb–May, count-neutral swaps) and 4 pure additions (Jul–Nov,
  net +1 edge each) across the year, so both the active edge *set* and the
  active edge *count* now evolve snapshot to snapshot (442→446).
- **Label rates ran hot vs. Dataset.md Appendix A** (delay 5–15%, shortage
  2–8% guidance; actual was ~17%/~23%). Fixed by recalibrating the
  stress→delay-probability formula and the inventory reorder trigger; now
  ~11% / ~7%, with the hidden-dependency margin still comfortably intact.
- **A floor clamp on `reliability_history`** (`max(0.55, beta_sample)`)
  produced an artificial spike of suppliers sitting at exactly 0.5500. Fixed
  by replacing the clamp with an affine rescale of the same Beta draw, which
  keeps the same source distribution shape with no discontinuity.

## Loading

```bash
pip install psycopg2-binary
createdb chainpilot
python load_data.py --dsn "postgresql://user:pass@localhost:5432/chainpilot"
```

`load_data.py` applies `schema.sql`, COPYs the CSVs in FK order, runs seven
verification queries (FK integrity, chronology, label-window containment,
history coverage), and **rolls back everything if any check fails**.
`--drop` recreates the schema first; `--verify-only` re-runs checks alone.

## Regenerating

```bash
python generate_dataset.py
```

Fully deterministic: fixed RNG seed + UUIDv5 keys. The validation suite
(FK/PK integrity, chronology, non-negative stock, label windows, graph
connectivity, hidden-dependency observability) runs at the end and exits
non-zero on any failure.

## Scope notes

- `schema.sql` covers the model-relevant tables only; application tables
  (users, alerts, chat, MCP, audit) live in `docs/05_Database_Design.md`.
- `orders.customer_name` is omitted — the schema reflects the post-migration
  state (Doc 5 §7.1 step 6).
- `risk_scores` rows are deterministic weighted-formula seeds for dashboard
  demos. Real rows are written by the inference service after training.
- `impact` labels are supplier-level ("any of this supplier's in-flight
  shipments went late in the horizon") — a simple but causal definition;
  refine when the impact head's target is finalised.
- First usable t0 is 2024-07-01: earlier months lack the 90/180-day feature
  history, per the backfill-honesty rule (Doc 5 §7.2).
