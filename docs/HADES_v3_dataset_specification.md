# HADES v3 / ChainPilot — Data Specification for Sourcing a Real-World Dataset

**Purpose of this document.** HADES v3 is currently trained on a synthetic (simulated) supply-chain
dataset that was deliberately built to mirror the shape of a real ERP/logistics environment — the
same node types, the same columns, the same timing rules. This document is a complete map of that
structure: every node (entity) the model reasons over, the exact table and columns each node's
features are built from, what each feature is used for, and the relationships (edges) connecting
nodes into one graph. Use it as the request checklist for what to ask a company for when sourcing a
real dataset to replace the synthetic one.

**How to read this.** Each node section lists the *raw columns* (what a database/ERP export needs
to contain) and the *derived features* the model actually trains on (what those columns get turned
into). A company will have the raw columns in some form; the derived-feature step is something this
project's own pipeline already does automatically — it does not need to be requested from them.

---

## 1. The big picture — what kind of model this is

HADES predicts three outcomes from a single shared graph of a supply chain, rebuilt fresh at each
monthly snapshot date (`t0`):

| Task | What it predicts | Which node carries the prediction |
|---|---|---|
| **Delay** | Will this shipment arrive late within the next 14 days? | Shipment |
| **Shortage** | Will this product run below its reorder threshold at this warehouse within 14 days? | Product (disambiguated by warehouse) |
| **Impact** | Will any of this supplier's in-flight shipments go late within 14 days? | Supplier |

Every prediction is made **as of a specific date** (`t0`), using only information that existed
*and had already been recorded* by that date. This "as-of" discipline is the single most important
requirement to carry into any real-world data request — see §5.

---

## 2. Node types — overview

The graph has 11 node types. The first 8 are the core structure; the last 3 are a 2025 enrichment
that added shipping-logistics detail (carriers, ports, routes) specifically to fix the shortage and
impact tasks' accuracy, described further in §6.

| # | Node type | Represents | Core or enrichment |
|---|---|---|---|
| 1 | Supplier | A company that supplies components | Core |
| 2 | Component | A raw part sourced from a supplier | Core |
| 3 | Product | A finished/assembled item made of components | Core |
| 4 | Factory | An assembly site | Core |
| 5 | Warehouse | A stocking/distribution location | Core |
| 6 | Customer | A buyer of finished products | Core |
| 7 | Shipment | One physical movement of goods | Core |
| 8 | Order | A customer purchase order | Core |
| 9 | Carrier / CarrierLane | A shipping company (or a carrier-route pairing) | Enrichment |
| 10 | Port | A sea or inland freight gateway | Enrichment |
| 11 | Route | A specific origin→destination shipping lane | Enrichment |

---

## 3. Core nodes — detailed feature specification

### 3.1 Supplier

**Source tables:** `suppliers`, `supplier_temporal_features`

| Column | Table | Type | Used as a feature? | Significance for training |
|---|---|---|---|---|
| `id` | suppliers | UUID (PK) | Identity only | Links every other table to this supplier |
| `country` | suppliers | text | Yes — one-hot encoded | Geography is a strong proxy for regional disruption exposure (customs, ports, regional shocks) |
| `capacity_score` | suppliers | numeric | Yes — z-scored | How much volume this supplier can realistically handle; low capacity relative to demand is a leading shortage/delay indicator |
| `lead_time_days` | suppliers | integer | Yes — z-scored | Baseline shipping time; a key driver of how much buffer exists before a delay becomes a shortage |
| `name` | suppliers | text | No | Identity/display only |
| `reliability_history` | suppliers | numeric (0–1) | **No — excluded on purpose** | Mutable, whole-history figure; using it as a feature would leak information from after the prediction date. **Do not substitute this for the temporal features below.** |
| `on_time_rate_30d` / `_90d` / `_180d` | supplier_temporal_features | numeric (0–1) | Yes | Rolling on-time delivery rate over 3 windows — the single strongest signal for a supplier's current reliability trend |
| `trend_slope` | supplier_temporal_features | numeric | Yes | Whether the supplier's reliability is improving or degrading recently, not just its current level |
| `lateness_variance` | supplier_temporal_features | numeric | Yes | How *erratic* the supplier is — a consistently mediocre supplier is more predictable (and lower risk) than a volatile one with the same average |
| `days_since_last_late` | supplier_temporal_features | integer | Yes | Recency of the last problem — a supplier "due for" another issue reads differently than one that just failed |
| `as_of_date` | supplier_temporal_features | date | Used for filtering | Every one of the fields above must be computed **fresh at each snapshot date**, using only data available up to that date (see §5) |

**Required cadence:** the temporal features need to be recomputed periodically (monthly, in this
project) so the model always sees a supplier's *recent* behavior, not a static lifetime average.

---

### 3.2 Component

**Source table:** `components`

| Column | Type | Used as a feature? | Significance |
|---|---|---|---|
| `id` | UUID (PK) | Identity only | |
| `supplier_id` | UUID (FK) | Defines the Supplier→Component edge | Which supplier makes this component |
| `component_type` | text | Yes — one-hot encoded | Different component classes (electronic, mechanical, polymer, fastener, specialty) carry different baseline risk profiles |
| `unit_cost` | numeric | Yes — z-scored | Higher-cost components often correlate with longer/more specialized lead times |
| `name` | text | No | Display only |

---

### 3.3 Product

**Source tables:** `products`, `product_components` (bill of materials), `inventory_history`

| Column | Table | Used as a feature? | Significance |
|---|---|---|---|
| `id` | products | Identity only | |
| `category` | products | Yes — one-hot encoded | Product category has its own demand-seasonality and shortage patterns |
| `sku`, `name`, `is_active` | products | No | Display/identity only |
| BOM composition (from `product_components`) | product_components | Yes — derived | `bom_component_count`: how many distinct components this product needs — more components means more places a shortage can originate; `bom_mean_quantity_required`: average units of each component needed per product unit |
| Stock position (from `inventory_history`, latest reading per warehouse as of the snapshot date) | inventory_history | Yes — derived | `min_stock_ratio` / `avg_stock_ratio` (stock ÷ reorder threshold, across the warehouses stocking this product): the direct precursor signal to a shortage; `total_stock`, `total_reorder_threshold`, `warehouse_count` |

**Important:** the live `inventory.stock_level` column (current snapshot table) is **not** what
feeds the model — only the **historical, timestamped** `inventory_history` table is used, so that
"stock level as of `t0`" can be reconstructed honestly instead of reading today's number. See §5.

**BOM validity windows:** `product_components` needs a `created_at` / `deactivated_at` pair per row
so a component substitution or addition can be dated — the model must only see the BOM as it stood
at each snapshot date, not the current one.

---

### 3.4 Factory

**Source table:** `factories`

| Column | Used as a feature? | Significance |
|---|---|---|
| `id` | Identity only | |
| `capacity_units_per_day` | Yes — z-scored | Manufacturing throughput ceiling |
| `location` | Yes — one-hot encoded (by country/region bucket) | Regional exposure, same rationale as Supplier country |
| `name`, `is_active` | No | Display only |

---

### 3.5 Warehouse

**Source table:** `warehouses` (core); `warehouse_role_features` (enrichment, §6)

| Column | Table | Used as a feature? | Significance |
|---|---|---|---|
| `id` | warehouses | Identity only | |
| `capacity_units` | warehouses | Yes — z-scored | Storage ceiling |
| `location` | warehouses | Yes — one-hot encoded | Regional exposure |
| `inbound_share`, `inbound_30d`, `outbound_30d`, `open_inbound`, `open_outbound` | warehouse_role_features | Yes (enrichment) | Distinguishes a warehouse acting mainly as a replenishment point vs. mainly as a fulfilment point, and how much open (in-flight) traffic it currently has in each direction |

---

### 3.6 Customer

**Source table:** `customers`

| Column | Used as a feature? | Significance |
|---|---|---|
| `id` | Identity only | |
| `priority_tier` (strategic / standard / low) | Yes — one-hot encoded | A strategic customer's order being at risk carries different downstream business weight, and often different service commitments |
| `name`, `contract_terms`, `is_active` | No | Display/contractual only |

---

### 3.7 Shipment — the delay-prediction entity

**Source tables:** `shipments`, `shipment_status_history`, `carrier_performance_snapshots`

| Column | Table | Used as a feature? | Significance |
|---|---|---|---|
| `id` | shipments | Identity only | |
| `supplier_id`, `factory_id`, `warehouse_id`, `order_id` | shipments | Define edges (§4) | Connects this shipment into the rest of the graph |
| `carrier` | shipments | Joins to carrier performance | Which shipping company is handling this leg |
| `eta` | shipments | Yes — derived (`days_to_eta`) | How much runway remains before the promised arrival |
| `dispatched_at` | shipments | Yes — derived (`days_since_dispatch`, only once actually dispatched) | How long this shipment has already been in motion |
| **`status` (live column)** | shipments | **No — excluded** | Current/terminal value; using it directly would leak the outcome. The model instead reconstructs "status as of `t0`" from the history table below |
| **`delivered_at`** | shipments | **No — excluded, this is part of the label itself** | |
| `status` (each transition) | shipment_status_history | Yes — one-hot encoded, **as-of `t0` only** | scheduled / in_transit / delivered / delayed, reconstructed from the transition log up to the snapshot date |
| `on_time_rate_90d` | carrier_performance_snapshots | Yes | The handling carrier's recent track record, joined in by carrier name and as-of date |

**This is the task-1 (delay) prediction target's home node.** The label itself comes from
`shipment_status_history`: a shipment is a delay-positive case if it transitions to `delayed` with
the transition's true event time inside the 14-day prediction window.

---

### 3.8 Order

**Source table:** `orders`

| Column | Used as a feature? | Significance |
|---|---|---|
| `id` | Identity only | |
| `status` (open / fulfilled / cancelled / at_risk) | Yes — one-hot encoded, as-of `t0` | Current fulfillment state |
| `due_at` | Yes — derived (`days_to_due`) | Urgency |
| `customer_id` | Defines Order→Customer edge | |
| `placed_at` | Used for as-of filtering | Only orders placed by `t0` exist in that snapshot |

---

## 4. Relationships (edges) — how the nodes connect

A graph model needs the connections as much as the node features. These are built directly from
foreign keys and junction tables already present in a normal ERP schema — no extra data collection
beyond what's listed above.

| Relationship | From table | Extra info carried on the edge |
|---|---|---|
| Supplier → supplies → Component | `components.supplier_id` (+ `component_suppliers` for secondary/qualified sources) | — |
| Component → used in → Product | `product_components` (BOM, as-of active rows only) | quantity required |
| Product → stocked at → Warehouse | `inventory_history` (latest reading per pair) | stock level, reorder threshold |
| Product → manufactured at → Factory | `product_factories` (as-of active rows) | site capacity for that product |
| Shipment → ships from → Supplier / Factory | `shipments.supplier_id` / `.factory_id` | — |
| Shipment → ships to → Warehouse | `shipments.warehouse_id` | days to ETA |
| Shipment → fulfills → Order | `shipments.order_id` | — |
| Order → ordered → Product | `order_items` | quantity |
| Order → placed by → Customer | `orders.customer_id` | — |
| Supplier → upstream of → Supplier | `supplier_upstream` (optional; multi-tier supply chains only) | tier depth |

### Enrichment relationships (§6)

| Relationship | From table | Extra info carried on the edge |
|---|---|---|
| Shipment → handled by → Carrier / CarrierLane | `shipment_routes` | — |
| Shipment → moves on → Route | `shipment_routes` | — |
| Route → passes through → Port | `route_ports` | role (origin/transit/destination), sequence |
| Route → depends on → Route | `route_dependencies` | shared channel (port/trucking/customs), group size |
| Shipment → originates from → Supplier | `shipments` + `eta`/`dispatched_at` | days-to-eta, days-since-dispatch |
| Shipment → replenishes → Warehouse | `shipments` (inbound leg, has a supplier) | — |
| Shipment → delivers to → Warehouse | `shipments` (outbound leg, no supplier — factory-origin) | — |
| **Product → replenished by → Shipment** | `product_replenishment` | days-to-eta, days-since-ordered, warehouse index |
| Warehouse → fulfills → Customer | `warehouse_customers` | order count |

The **Product → replenished by → Shipment** edge is the single most important enrichment edge — it
is what lets the model see *which specific inbound shipment is currently resupplying a given
product/warehouse position*, which was structurally missing from the original graph and is
discussed in §6.

---

## 5. Non-negotiable data requirements — the "leakage-free" contract

These are the requirements to raise explicitly with the company's data/engineering team, because
most operational databases are **not** built this way by default, and getting this wrong silently
produces a model that looks accurate in testing but fails in production.

**1. Every changing fact needs a history, not just a current value.** Stock levels, shipment
status, and supplier reliability must be logged as a **time series of past readings**, not just
overwritten in place. A single "current stock level" column is not sufficient — the model needs to
be able to ask "what was the stock level as of *this past date*," which requires every change to
have been recorded with a timestamp, never overwritten.

**2. Every historical record needs two timestamps, not one — this is the single most important
ask.** Every logged event needs both:
- **when it actually happened** (e.g., the shipment status genuinely changed, the stock was
  genuinely counted), and
- **when it was recorded/reported into their system** (which is very often *later* than when it
  happened — data entry lag, batch syncs, carrier reporting delays).

Without the second timestamp, it's impossible to honestly reconstruct "what did we know as of this
date" versus "what turned out to be true" — and training on the second (what turned out to be true)
instead of the first is the most common, and most dangerous, mistake in building a model like this.

**3. Labels must be scoped to a clear time window.** A prediction made "as of" a date should only
be scored against outcomes that occurred in a defined window afterward (this project uses 14 days).
Get clarity from the company on how quickly they'd want a warning before an event, since that sets
the label window.

**4. Any column that only holds today's/current state should be flagged, not silently used.**
Columns like a live "status" field, a live "stock level" field, or a "last known reliability" figure
are fine for the company's own dashboards, but are exactly the kind of column that causes leakage if
fed directly into training — they need their historical equivalents instead (see §5.1 above).

---

## 6. Why the enrichment nodes (Carrier / Port / Route) were added

The original graph structure (§3) was missing the shipping-logistics layer — nothing represented
*how* a shipment physically travels, only its endpoints. Two problems this caused, both fixed by
adding Carrier/Port/Route:

- **Shortage prediction lacked a "what's currently coming to restock this" signal.** Adding the
  `Product → replenished by → Shipment` edge gave the model direct visibility into the inbound
  shipment resolving a given shortage risk.
- **The initial single "Carrier" node design overloaded one node with too many connections** (a
  handful of major carriers, each linked to thousands of shipments), which destabilized training.
  The fix was re-grouping carrier information into many smaller **(carrier, lane)** nodes — same
  underlying carrier performance data, just organized so no single node in the graph is
  overwhelmingly overconnected. If sourcing this from a real company, ask for carrier performance
  broken out **by lane** (i.e., by origin–destination pair), not just by carrier overall, since that
  finer breakdown is what the fix actually depends on.

If a real-world dataset can only supply carrier-level (not lane-level) shipping data, the model can
still be built — it would simply use the original, coarser Carrier node design, with the
understanding that it is more prone to the instability already observed and fixed on the synthetic
data.

---

## 7. Consolidated data request checklist

A single list to bring into the conversation — every file/table to ask for, grouped by whether it's
essential (core model) or valuable-but-optional (enrichment).

**Essential — the model cannot be built without these:**

- Supplier master data + a **rolling history** of on-time delivery performance (not just a current score)
- Component master data (type, cost, which supplier makes it)
- Product master data + bill-of-materials, with change history
- Factory and Warehouse master data (capacity, location)
- Customer master data (priority tier)
- Shipment records (supplier/factory/warehouse/order links, carrier, ETA, dispatch time)
- **Full shipment status transition history**, each with both an event time and a recorded time
- Order and order-line data
- **Inventory history** — periodic stock readings over time, with both an event time and a recorded time
- Carrier-level on-time performance history

**Valuable — improves shortage/impact accuracy specifically, per §6:**

- Carrier performance broken out by lane (origin–destination), not just overall
- Port/gateway identification for shipment origins and destinations
- Route-level transit-time and condition history
- Any known shared-infrastructure dependencies between lanes (shared ports, shared customs
  jurisdictions, shared trucking corridors) — even an approximate list is useful
- A direct link between a specific inbound shipment and the (product, warehouse) position it is
  currently replenishing

**Not needed / do not request:** anything the company considers a true "hidden" or unobservable
factor (e.g., an internal risk score they can't explain the derivation of) is not useful as a raw
input — the model is specifically designed to *infer* that kind of signal from the observable
history above, not to be handed a pre-computed guess.
