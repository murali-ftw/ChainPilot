# 06 — Graph Design (V2)

> **V2 update.** Node and edge types extended for the new mechanisms, with every edge classified as
> **observable** or **simulator-internal**. An edge in the second class exists and transmits risk
> but is never emitted — that gap is what several research questions measure.

---

## 1. Node types

| Node | Source table | Observable? |
|---|---|---|
| Supplier | `suppliers` | **partially** — Mechanism A truncates tiers above `max_visible_tier` |
| Component | `components` | yes |
| Product | `products` | yes |
| Factory / Warehouse | `factories`, `warehouses` | yes |
| Customer | `customers` | yes |
| Order / Shipment | `orders`, `shipments` | yes |
| **Hidden parent** | *none* | **no** — Mechanism B; has no table, no id, no row |

The hidden parent is not a suppressed node; it is a node that never existed in any emitted form.
Only the correlated behaviour of its members is observable.

## 2. Edge types

### Observable

| Edge | From → To | Source | Notes |
|---|---|---|---|
| `SUPPLIES` | Supplier → Component | `components.supplier_id` | primary source |
| `ALSO_SUPPLIES` | Supplier → Component | `component_suppliers` | multi-sourcing; **validity windows**, rewired by C |
| `UPSTREAM_OF` | Supplier → Supplier | `supplier_upstream` | **new in V2** — Mechanism J; emitted only when *both* endpoints survive A |
| `USED_IN` | Component → Product | `product_components` | validity windows; evolves across snapshots |
| `MANUFACTURED_AT` | Product → Factory | `product_factories` | |
| `STOCKED_AT` | Product → Warehouse | `inventory` | |
| `SHIPPED_BY` | Shipment → Supplier/Factory | `shipments` | |
| `ORDERED_BY` | Order → Customer | `orders` | |

### Simulator-internal (never emitted)

| Edge | Mechanism | What it does | Why it is hidden |
|---|---|---|---|
| hidden-parent membership | B | groups suppliers under a shared upstream entity | the structure models must **discover**; discoverable only through correlated degradation |
| hidden-parent coupling | D | `α × mean(co-members' own stress)` | the incremental signal — unreachable from a member's own history |
| truncated upstream hops | A | tier > `max_visible_tier` links | the hidden network keeps transmitting; ~5,400 such links at spec scale |
| shared-factor membership | V1 | `H_PORT`, `H_TRUCK`, `H_CUSTOMS`, `H_POLYMER` | only correlated effects are observable |
| shock incidence | H | which suppliers a shock hit | inferable only from consequences |

## 3. Transmission and attenuation

Risk reaches a supplier through three inbound paths, all attenuated at the **downstream** node when
Mechanism F is active (Clarification 3: a resilient buyer absorbs an upstream shock):

| Path | Edge class | Coefficient |
|---|---|---|
| co-parent bleed | observable (`ALSO_SUPPLIES` co-parents) | `COPARENT_COUPLING × recv_atten(node)` |
| hidden-parent coupling | **internal** | `α × recv_atten(node)`, Type A groups only |
| upstream chain | observable topology, **internal beyond tier 2** | per-hop `SUP_ATTEN[downstream]` |

Without Mechanism F, `recv_atten()` returns 1.0 and every variant keeps its pre-F transmission
exactly. With F, the coefficient is a deterministic interpolation of hidden resilience against the
three configured anchors — never a second latent, never random.

## 4. Snapshot semantics

`graph_snapshots` registers one row per `t0` with node counts, edge counts and label counts. Two
properties matter for graph construction:

- **Edges carry validity windows.** `component_suppliers`, `product_components` and
  `supplier_upstream` all have `created_at` / `deactivated_at`. A snapshot's graph is the set of
  edges live at its `t0` — not the whole table.
- **Topology moves between snapshots.** Mechanism C mixes substitutions (count-neutral swaps) with
  emergency-sourcing additions (net +1), so both the active edge **set** and its **count** change
  snapshot to snapshot. A static edge count across snapshots is a synthetic tell that V1's audit
  specifically fixed; do not reintroduce it by materialising the graph once.

## 5. Depth heterogeneity

Mechanism J gives each product-facing supplier its own chain depth, drawn from a triangular
distribution (`tier_depth_min` / `mode` / `max`). Realised distribution at spec scale:

```
depth 2: 197   3: 1,360   4: 1,144   5: 509   6: 67
```

**Chain depth is independent of per-hop attenuation** — a long chain may attenuate strongly and a
short one weakly. Verified against a permutation null rather than a fixed threshold, because chains
share upstream suppliers and the sampling distribution of that correlation is not zero-centred.
Measured correlation at spec scale: **+0.010**.

End-to-end attenuation *is* negatively correlated with depth (−0.59), which is geometry — more hops
multiply more coefficients — not a design flaw.

## 6. Construction guidance

1. Build per-snapshot, respecting validity windows; do not materialise one static graph.
2. Treat `suppliers` as already truncated — absent upstream nodes are the task, not missing data.
3. `supplier_upstream` is absent entirely without Mechanism J; handle its absence rather than
   assuming an empty table.
4. Never join back to simulator-internal state. It is not in the CSVs, and reconstructing it from
   the generator source would invalidate the result.
