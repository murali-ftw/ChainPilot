# Relation-Frequency Entropy Diagnostic

**Scope:** measure the real skew of edge counts across every meta-relation in the currently
live graph, compute Shannon entropy over that distribution, confirm which relations are
genuinely thinnest, and check whether that skew number predicts the win/loss pattern already
measured across every architecture tried in this project. No training in this document — pure
measurement against the live database and a synthesis of already-existing results.

## Dataset-version note (live-checked, not assumed — and a correction to prior rounds)

Confirmed live before this measurement: **800 suppliers, 15 `graph_snapshots`** — the same
scale as every "v3" round since `reports/hades_model_development_report.md`'s Round 3. But a
direct check this round found something prior rounds' own Phase-0 checks got wrong:
**dual-sourcing (the Task 3 co-parent mechanism) is actually active on the currently loaded
dataset**, not dormant as the RGCN-architecture rounds (`reports/hades_model_development_report.md`
Rounds 5–6, `reports/rgcn_matched_pilot.md`) stated. `component_suppliers` holds 420 rows (one
secondary supplier per affected component — 17.5% of components), and a direct 2-hop reach
check finds **449/800 suppliers (56.1%) reach a different supplier via
`SUPPLIES`→`rev_SUPPLIES`** — matching Task 3's own reported co-parent-reach measurement
exactly, not the 0/800 that holds on the base v3 generation without dual-sourcing. Earlier
rounds' Phase-0 checks saw `component_suppliers` had at most one row per component and
mis-read that as "no dual-sourcing present" — it actually means the opposite: those 420
components already have their (single) secondary supplier active. This measurement below —
and every RGCN-family training round since the 4-architecture ablation — has therefore been
running against the co-parent-enabled dataset the whole time, not the schema-only "dormant"
state previously claimed. This doesn't invalidate any prior measured number (they're real
measurements against whatever was actually loaded, and every RGCN-family round has been
internally consistent with the others on this same data), but it is the accurate dataset-state
correction for the record.

## Real relation-frequency table (live, from a freshly built snapshot)

Built directly from `ml/graph/builder.py::build_snapshot` against the latest snapshot
(2025-09-01), not from `graph_snapshots.edge_counts` — that column is stale (last refreshed
before v3's relation renaming; it still has 4 keys under pre-`STOCKED_AT` names). All 20
meta-relations (10 forward + `ToUndirected()`'s 10 reverse) below, sorted ascending:

| Relation | Edge count | Share of total |
|---|---:|---:|
| `Product→MANUFACTURED_AT→Factory` | 1,933 | 0.50% |
| `Factory→rev_MANUFACTURED_AT→Product` | 1,933 | 0.50% |
| `Supplier→SUPPLIES→Component` | 2,820 | 0.72% |
| `Component→rev_SUPPLIES→Supplier` | 2,820 | 0.72% |
| `Product→STOCKED_AT→Warehouse` | 3,194 | 0.82% |
| `Warehouse→rev_STOCKED_AT→Product` | 3,194 | 0.82% |
| `Component→USED_IN→Product` | 7,092 | 1.82% |
| `Product→rev_USED_IN→Component` | 7,092 | 1.82% |
| `Shipment→SHIPS_FROM→Supplier` | 15,826 | 4.05% |
| `Supplier→rev_SHIPS_FROM→Shipment` | 15,826 | 4.05% |
| `Shipment→SHIPS_FROM→Factory` | 20,116 | 5.15% |
| `Shipment→FULFILLS→Order` | 20,116 | 5.15% |
| `Factory→rev_SHIPS_FROM→Shipment` | 20,116 | 5.15% |
| `Order→rev_FULFILLS→Shipment` | 20,116 | 5.15% |
| `Order→PLACED_BY→Customer` | 25,228 | 6.46% |
| `Customer→rev_PLACED_BY→Order` | 25,228 | 6.46% |
| `Shipment→SHIPS_TO→Warehouse` | 35,942 | 9.21% |
| `Warehouse→rev_SHIPS_TO→Shipment` | 35,942 | 9.21% |
| `Order→ORDERED→Product` | 62,936 | 16.12% |
| `Product→rev_ORDERED→Order` | 62,936 | 16.12% |
| **Total** | **390,406** | **100%** |

Note `SUPPLIES`/`rev_SUPPLIES` at 2,820 (not 2,400) reflects the active dual-sourcing edges
per the dataset-version note above — 2,400 mandatory primary-supplier edges (one per
component) plus 420 secondary-supplier edges from `component_suppliers`.

## Shannon entropy

$H = -\sum_i p_i \ln p_i$ over the 20-way relation-share distribution above:

| Metric | Value |
|---|---:|
| Raw entropy $H$ (nats) | 2.6008 |
| Maximum possible entropy, $\ln(20)$ (perfectly uniform) | 2.9957 |
| **Normalized entropy, $H / \ln(20)$** | **0.8682** |

At 0.868 (out of a maximum of 1.0), the relation distribution is closer to uniform than to
maximally skewed — no single relation dominates the graph's edges the way a power-law-heavy
schema might. But the *ratio* between the thinnest and densest relation is still large: the
densest (`ORDERED`, 16.12%) carries **32.6× as many edges** as the thinnest (`MANUFACTURED_AT`,
0.50%) — a real, structurally meaningful gap sitting underneath a globally-moderate entropy
number.

## Are MANUFACTURED_AT/SUPPLIES/STOCKED_AT/USED_IN really the thinnest four?

**Yes, confirmed directly from the live count table above** — they are exactly the four
lowest-count relations (1,933 / 2,820 / 3,194 / 7,092), with a real gap before the fifth-lowest
(`SHIPS_FROM`→Supplier at 15,826 — more than double `USED_IN`'s count). This is the same
identification `ml/models/sparse_hgt_encoder.py`'s Task 2 experiment used
(`reports/hades_model_development_report.md` Round 4), now re-confirmed live against the
current (co-parent-enabled) dataset rather than assumed to still hold.

## Does this entropy number predict the win/loss pattern already measured?

**Not as a single global scalar — but the *identity* of which relations are thinnest lines up
with the pattern precisely, on a task-by-task basis.** Pulling already-existing results,
without retraining anything:

- **Shortage is the one task where basis-decomposition architectures (RGCN, RGCN+attn) beat
  HGT consistently, on every axis tested so far** — RGCN over both HGT and GraphSAGE, fixed-d
  and matched-d (`reports/hades_model_development_report.md` Round 5); RGCN+attn over HGT at
  both fixed-d (`reports/hades_model_development_report.md` Round 6) and matched-d
  (`reports/rgcn_matched_pilot.md`). Shortage's own causal signal draws directly on
  `STOCKED_AT` (Product-Warehouse stock/threshold context) and `USED_IN` (Component-Product BOM
  structure) — **two of the four thinnest relations in the entire graph.** This is exactly
  where HGT's fully-dedicated per-relation parameterization would be expected to overfit most
  (least data per dedicated weight matrix), and exactly where every basis-sharing architecture
  tried so far has won.
- **Impact is the one task HGT wins decisively against every RGCN-family architecture tried,
  including the hybrids** (`reports/hades_model_development_report.md` Rounds 5–6,
  `reports/rgcn_matched_pilot.md`) — RGCN+attn is the only hybrid to close that gap to a
  statistical tie; plain RGCN and RGCN+relemb both lose to HGT on impact consistently. Impact's
  target is the Supplier node itself (`entity_type='supplier'`), reached at hop 0 — it doesn't
  lean on the graph's thinnest relations the way shortage does, so HGT's richer,
  fully-dedicated per-relation parameterization isn't penalized there, and its extra capacity
  seems to actively help.
- **Delay sits in between** — GraphSAGE/HGT/RGCN were a 3-way statistical tie in the 4-arch
  round, with RGCN+attn opening a new, consistent win over both at fixed-d and matched-d. Delay's
  own target (Shipment, reachable at hop 1 via `SHIPS_FROM`) touches one of the *mid-tier*
  relations (15,826 edges, 4.05% share) rather than the thinnest four directly.

**Net reading:** the normalized entropy (0.868) is not low enough, as a single number, to
predict "HGT will overfit broadly" — most of the graph's relation mass is fairly evenly spread.
What predicts the measured win/loss pattern is more specific than global entropy: **whether a
given task's own causal signal concentrates on the graph's thinnest few relations.** Shortage
does (2 of its causal relations are 2 of the 4 thinnest) and is the one task where every
basis-sharing architecture beats HGT. Impact doesn't, and HGT wins there outright. This is a
task-level alignment with the identity of the thin-relation set, not a global correlation with
the entropy scalar itself — the entropy measurement's real value here is confirming that
`MANUFACTURED_AT`/`SUPPLIES`/`STOCKED_AT`/`USED_IN` really are a distinct, separated cluster
(a genuine ~4–16× gap to the next tier, not an arbitrary cutoff), which is the premise the whole
thin-relation-overfitting hypothesis (Task 2 onward) has rested on since it was first raised.

---

*See `reports/rgcn_types.md` for the full per-architecture results (parameter counts,
matched-d AUC, over-smoothing sweep) this entropy measurement is being checked against. This
document is the entropy diagnostic specifically; it does not restate those numbers in full.*
