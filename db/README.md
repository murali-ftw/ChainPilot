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

**v2 (current)** — scaled up post-Steps-0-5 (`reports/steps_0-5_findings_v2.md`'s "Step C"):
`docs/01_Product_Requirement_Document.md` §8 assumes "hundreds to low thousands" positive
labels; the original 50-supplier world produced single/low-double-digit positives for
delay/impact. `SUP_N` went 50→180 with every other world-size count (`SCALE = SUP_N/50 =
3.6`) scaled proportionally.

| Table | Rows | Notes |
|---|---|---|
| suppliers | 180 | 6 countries, power-law degree, `reliability_history` display-only |
| components | 540 | each belongs to exactly one supplier; fasteners/electronics reused widely |
| products | 288 | 6 categories |
| product_components | 1,580 | BOM with validity windows; 4 substitutions + 4 pure additions spread Feb–Nov |
| factories / product_factories | 5 / 427 | per-product capacity, `is_primary` (factory count NOT scaled — physical DCs, not supply-chain-proportional) |
| warehouses / inventory | 8 / 723 | current state only (warehouse count NOT scaled, same reason) |
| inventory_history | ~38,320 | weekly observations, bitemporal (`observed_at`+`recorded_at`) |
| customers | 360 | strategic ~15% / standard ~70% / low ~15% |
| orders / order_items | 3,600 / ~8,970 | seasonal spike Sep–Oct, 5% enterprise orders |
| shipments | ~5,130 | supplier→warehouse replenishment + factory fulfilment |
| shipment_status_history | ~15,840 | full transition log — the delay-label source |
| supplier_temporal_features | 1,080 | 6 monthly `as_of_date`s × 180 suppliers, windows end at t0 |
| carrier_performance_snapshots | 30 | 5 carriers × 6 t0s |
| graph_snapshots | 6 | t0 = Jul 1 … Dec 1 2024, horizon 14d, counts JSON; `USED_IN` edge count grows across snapshots as BOM additions land |
| training_labels | ~6,640 | tasks `delay` / `shortage` / `impact`; ~11.4% / ~9.1% / ~4.1% positive (140/396/44 total positives — delay and shortage now solidly in the "low hundreds" the PRD assumes; impact reached the low 40s, a 4x improvement over v1's 11 but still short of "hundreds" — pushing it further would need supplier counts well beyond the 150-200 range this scale-up targeted, see `reports/steps_0-5_findings_v2.md`) |
| risk_scores | 180 | seed rows, `weighted_formula` — **not** model output |

Exact counts vary slightly on regeneration-affecting edits (RNG draw order shifts downstream), but are always internally consistent and re-validated by the generator's own suite — see `Regenerating` below.

**v1 (superseded)** — the original 50-supplier world: suppliers 50, components 150,
products 80, product_components 450, factories/product_factories 5/119,
warehouses/inventory 8/194, inventory_history ~10,280, customers 100, orders/order_items
1,000/~2,490, shipments ~1,440, shipment_status_history ~4,460,
supplier_temporal_features 300, training_labels ~1,774 (delay/shortage/impact ~11%/~7%/~4%,
34/84/11 total positives). Kept here for the record; `db/csv/` and the database now hold v2.

## How the world works (causality, not randomness)

Nothing is sampled independently. A latent **stress** value per supplier per
day drives everything observable:

1. **Disruption events** (Chapter 10): a NA trucking strike (Jul–Aug), German
   customs friction (Aug–Sep), port congestion hitting sea-freight suppliers
   (Sep–Nov), a factory outage (Jun), and a **hidden upstream polymer
   shortage** (Sep–Dec). Each ramps up to a peak and decays — gradual recovery.
   Spread across Jul–Dec (v2 — see below) rather than concentrated in the
   back half of the year, so the chronological train (Jul–Sep) / test
   (Nov–Dec) split sees a comparable mix of quiet and disrupted periods on
   both sides.
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

**v2 fix:** in v1, the 4 members were chosen unconstrained, but polymer
*components* were then deliberately concentrated onto those same 4 suppliers
(18 of 25 polymer parts), making `component_type='polymer'` an accidental
giveaway of membership — the exact observable column the scenario is
supposed to be invisible through. v2 removes that concentration and instead
picks the 4 members deterministically to span **4 distinct component_types**
(never `polymer` as any member's dominant type). Current composition
(countries: Germany, Vietnam, USA, Vietnam — still cross-country as a group;
each member's own component types, collectively spanning all 5 categories,
no member polymer-only):

| Supplier (country) | Component types supplied |
|---|---|
| Germany | electronic, fastener, mechanical |
| Vietnam | electronic, fastener, mechanical, polymer, specialty |
| USA | electronic, fastener, mechanical, polymer |
| Vietnam | specialty |

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

## v2 fixes (post-Steps-0-5, `reports/steps_0-5_findings_v2.md`'s "Step C")

Four targeted structural problems identified in `reports/steps_0-5_findings.md`,
addressed in `generate_dataset.py`:

- **World scale.** `SUP_N` 50→180 (`SCALE = SUP_N/50` applied to
  components/products/customers/orders proportionally; factories/warehouses
  left fixed as physical infrastructure). Moves delay/shortage positive
  counts from single/low-double digits into the low hundreds, per
  `docs/01_Product_Requirement_Document.md` §8's own stated assumption.
  Impact only reached the low 40s (4x improvement, not "hundreds") —
  further scaling would need supplier counts well beyond 150-200.
- **Disruption timeline concentrated in H2.** All 4 events used to cluster
  Aug-Dec, so a chronological train (Jul-Sep) / test (Nov-Dec) split trained
  on a quiet regime and tested on a disrupted one. Redistributed (see the
  causality section above) so every month Jul-Dec has at least one active
  event, on both sides of the split.
- **Shortage labels dropped the warehouse dimension.** `entity_id` for
  `task='shortage'` is a `product_id`, but a product can be stocked at
  multiple warehouses with independent outcomes — without a disambiguator,
  71 (product, snapshot) pairs in v1 carried conflicting true/false rows.
  Fixed by adding `training_labels.warehouse_id` (populated for `shortage`
  only); the validation suite now asserts zero conflicts once grouped by
  `(snapshot_id, entity_id, warehouse_id)` (see `docs/05_Database_Design.md`
  §6.18 for the column's exact semantics and the model-pipeline caveat).
- **Hidden-dependency scenario confounded with `component_type`.** v1 picked
  H_POLYMER's 4 members unconstrained but then forced 18 of 25 `polymer`
  components onto those same suppliers, making `component_type='polymer'` a
  giveaway. Fixed by removing that concentration and instead choosing the 4
  members deterministically to span 4 distinct component_types (see "The
  hidden-dependency scenario" above for the resulting composition) — the
  causal mechanism (shared stress via the hidden factor) is unchanged, only
  its observability via `component_type` is removed, per `project_HADES.md`
  §5.4's requirement that Transformer 2's eventual validation mean anything.

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
- `shortage` labels' `entity_id` is a `product_id`, disambiguated by the
  `warehouse_id` column (v2) since a product can be stocked at multiple
  warehouses with independent outcomes in the same snapshot — see
  `docs/05_Database_Design.md` §6.18.
