# 05 — Database Design (V2)

> **V2 update.** The V1 schema is unchanged and still correct. This revision adds Mechanism J's
> upstream edge table, records the two-clock observability contract, documents variant metadata,
> and — most importantly — marks explicitly **which state is model-visible and which is
> simulator-only ground truth**.

**DDL:** `db/schema.sql`. **Loader:** `db/load_data.py`.

---

## 1. Visibility classification

This is the most important table in this document. Getting it wrong means training on ground truth.

### Model-visible (emitted to CSV, loaded to PostgreSQL)

| Table | Notes |
|---|---|
| `suppliers` | **truncated by Mechanism A** — tiers above `max_visible_tier` are never emitted |
| `components`, `products`, `product_components`, `product_factories` | BOM structure |
| `component_suppliers` | dual/multi-sourcing; rewired over time by Mechanism C; truncated by A |
| `supplier_upstream` | **new in V2.** Mechanism J's upstream chain; a hop is emitted only when *both* endpoints survive A's truncation |
| `factories`, `warehouses`, `customers`, `inventory` | master and current state |
| `orders`, `order_items`, `shipments` | operational |
| `inventory_history`, `shipment_status_history` | bitemporal; see §2 |
| `supplier_temporal_features`, `carrier_performance_snapshots` | as-of features, windows end at `t0` |
| `graph_snapshots`, `training_labels` | snapshot registry and labels |
| `risk_scores` | deterministic seed rows, **not** model output; truncated by A |

### Simulator-only ground truth (never emitted anywhere)

| State | Mechanism | Why it must stay hidden |
|---|---|---|
| `RESILIENCE[sup_id]` | E | the latent the benchmark asks models to infer. Enforced by an exact per-supplier scan of every emitted cell |
| `SUP_ATTEN[sup_id]` | F | a deterministic function of resilience; emitting it would leak resilience |
| `HP_GROUPS` / `HP_OF` (hidden parents, types A/B/C) | B, D | the latent structure models must discover; no row, no edge, no column connects members |
| `H_PORT`, `H_TRUCK`, `H_CUSTOMS`, `H_POLYMER` | V1 | shared hidden factors; only their correlated effects are observable |
| `SHOCK_EVENTS` | H | shocks are observable only through their consequences |
| Hidden-tier suppliers and their edges | A | continue to exist and transmit inside the simulator |
| `agent_seen` / mitigation decisions | Phase 2 | the agent's internal state |

### Display-only columns (present but leakage if used as features)

`suppliers.reliability_history`, `shipments.status`, `shipments.delivered_at`,
`inventory.stock_level` are **current-state** values. Reading them as model features is label
leakage; the schema carries comments saying so.

## 2. Two-clock observability

V1 already carried a bitemporal pair. Mechanism G widens the gap rather than adding a second
timeline.

| Table | Valid time (when it happened) | System time (when it was known) |
|---|---|---|
| `inventory_history` | `observed_at` | `recorded_at` |
| `shipment_status_history` | `changed_at` | `recorded_at` |

Contract:

- **Labels use valid time.** A delay that occurred inside the horizon is a positive regardless of
  when the feed reported it.
- **Features and as-of state use system time.** `asof_status()` and `sup_features()` filter on the
  recorded clock, so an event that happened before `t0` but was reported after it is genuinely
  unknown.
- The supplier-agent is blinded by the same clock — it cannot react to an outcome it has not yet
  been told about.

Consequence worth knowing: under Mechanism G, already-delivered shipments can still look in-transit
and remain label-eligible as guaranteed negatives, which dilutes the delay positive rate ~3×.

## 3. New in V2 — `supplier_upstream`

```sql
CREATE TABLE supplier_upstream (
    id                    UUID PRIMARY KEY,
    supplier_id           UUID NOT NULL REFERENCES suppliers(id),  -- downstream (buyer)
    upstream_supplier_id  UUID NOT NULL REFERENCES suppliers(id),  -- upstream (seller)
    upstream_tier         SMALLINT NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL,
    deactivated_at        TIMESTAMPTZ,                             -- NULL = still active
    UNIQUE (supplier_id, upstream_supplier_id, created_at),
    CHECK  (supplier_id <> upstream_supplier_id)
);
```

One row per hop, directed downstream → upstream. Only suppliers owning **no** components are
eligible for tier ≥ 2, which is what lets Mechanism A hide them without dangling a component FK.
Upstream suppliers are **shared** across chains — one mill serving many fabricators — because the
component-less pool is only ~20% of the population and exclusive chains would cap mean depth near
1.2. The table is absent entirely in variants without Mechanism J.

## 4. Variant metadata — `resolved_config.json`

Emitted beside the CSVs, not a database table:

```json
{
  "benchmark_version": "HADES-Bench-V2.0",
  "generated_at": "...", "generation_seed": 42, "variant": "K",
  "mechanisms_enabled": ["A","B","C","D","E","F","G","H","I","J"],
  "variant_dependencies": {"A": "J + A", "D": "B + D", "F": "E + F"},
  "derived": { "t_start": "...", "first_t0": "...", "last_t0": "...", "t_end": "...",
               "timeline_days": 1427, "snapshot_count": 40, "scale": 80.0 },
  "config": { "...all 31 parameters as resolved..." },
  "label_counts": { "delay": {...}, "shortage": {...}, "impact": {...} },
  "row_counts": { "suppliers_emitted": 3323, "suppliers_simulated": 4000, "..." : 0 }
}
```

`suppliers_emitted` vs `suppliers_simulated` is the auditable record of Mechanism A's truncation.

## 5. Referential integrity under truncation

Mechanism A removes rows from `suppliers`, so **every** table referencing a supplier must be
filtered consistently. This was a real defect: `component_suppliers` and `risk_scores` continued
emitting rows pointing at hidden suppliers, producing a foreign-key violation that only surfaced
when Variant K was loaded into PostgreSQL.

Both the generator (`A: no EMITTED table references a truncated supplier`) and the loader
(`FK: supplier_upstream -> suppliers (both endpoints)`) now check this. Run the loader before
trusting a variant — the in-generator suite reasons over in-memory structures and cannot see what
actually landed on disk.

## 6. Label semantics

- `training_labels.event_at` lies in `(t0, t0 + horizon]`, asserted by the generator and again by
  the loader.
- `shortage` labels key on `entity_id = product_id` **disambiguated by `warehouse_id`**, since one
  product can be stocked at several warehouses with independent outcomes in the same snapshot.
- `delay` labels exist only for shipments whose *as-of* status is `scheduled` or `in_transit`.
- `impact` labels exist only for **visible** suppliers.

## 7. Scope notes

- `schema.sql` covers model-relevant tables only. V1 application tables (users, alerts, chat, MCP,
  audit) are archived with the V1 application docs under `docs/v1_archive/`.
- `risk_scores` rows are deterministic weighted-formula seed rows for dashboard demos, not model
  output.
- First usable `t0` is `T_START + WARMUP_DAYS`; earlier months lack the 180-day feature history.
