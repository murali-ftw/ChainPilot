# Document 10 — AI/ML Documentation

## HADES: Hierarchical Attention with Dual Encoders and Structure

Version: 2.0 — rescoped to a standalone model-development prototype
Status: Baseline
Supersedes: v1.0 (dashboard/chatbot/MCP-integrated framing)
Authoritative architecture spec: `project_HADES.md` (supersedes `architecture.md` where they disagree)

---

## 1. Purpose

This project is not a product build. It is a **research prototype whose only goal is to train the HADES architecture on a leakage-safe supply-chain graph and find out whether its four design claims actually hold** — not to ship a dashboard, a chatbot, an MCP execution layer, or an optimization engine. Everything in this document exists in service of that one goal: get from raw records to a trained model, and get honest, statistically-defensible evidence for or against each claim below.

**The four claims under test**, each owned by one HADES component (`project_HADES.md` Part 0):

1. **Type-aware local structure beats type-blind local structure** — HGT should outperform GraphSAGE and GAT on the same graph, at matched parameter count, once label-volume noise is accounted for.
2. **Depth should be chosen per task, not fixed globally** — a structural depth prior derived from graph connectivity (not learned) should beat a single shared readout depth, and a small learned gate should be able to improve on the prior only where the data supports it.
3. **Correlated risk can exist with no recorded edge between two suppliers** — a global, same-type attention stage should be able to surface it, where message-passing at any depth structurally cannot.
4. **A supplier's global fragility is not the same number as this firm's exposure to it** — a deterministic dyadic reweighting should reorder risk in a way the encoder alone did not already encode.

## 2. Scope

- **In scope:** feature engineering, the leakage contract, graph construction, the HGT encoder, depth selection (structural prior + learned gate), Transformer 2 (global same-type attention), Claim B (dyadic reweighting), training protocol, evaluation protocol (including the architecture ablation and the layer-depth sweep), explainability, and model-governance metadata for every training run.
- **Out of scope, explicitly:** the frontend dashboard, the interactive chatbot, RAG/LLM explanation generation, the MCP execution/approval layer, the OR-Tools optimization engine, alerting, and the transparent weighted business-risk formula. None of these are needed to test the four claims above, and none are being built for this prototype. If any of that work resumes later, `architecture.md` and `project_HADES.md` remain its design reference — this document does not need to change to support that.
- **Not this document's job:** REST API shape, authentication, UI screens — see Section 4 for what was deliberately not carried forward from the earlier product-scoped version of this document.

## 3. Assumptions

- Training runs offline/batch against a fixed dataset snapshot; there is no serving layer, no online inference endpoint, and no retraining trigger to design.
- The synthetic dataset (`db/`) stands in for real historical data. It is deliberately constructed to contain a genuine hidden-dependency signal (Claim 3) and genuine label rarity, so the evaluation is a real test of the claims and not a foregone conclusion (`db/README.md`).
- GNNExplainer (or an equivalent perturbation-based explainer) is sufficient for the qualitative explanation-fidelity check in Section 10; explainability research rigor beyond that is out of scope.
- All figures in `project_HADES.md` are analytical estimates pending a real graph (`project_HADES.md` Appendix, "Final note to the reader") — this document treats them as planning inputs, not settled facts, and Section 9's measurement protocol is how they get replaced with real numbers.

## 4. What Changed From the Product-Scoped Version of This Document

The previous version of this document specified an ML layer feeding a six-layer product pipeline (RAG, LLM explanation, chatbot, MCP execution, OR-Tools optimization, frontend dashboard) and treated HADES's depth gate / Transformer 2 / Claim B as a distant, loosely-specified "Not committed — Phase 2" upgrade pointing at a draft document. Two things have changed:

1. **The product scope is gone.** This is now purely a model-development effort. Sections describing the Risk Intelligence Layer's business-facing weighted formula, threshold/categorization for a dashboard, the alternative-supplier recommender, and the what-if simulator have been removed — they were downstream *consumers* of Layer 2's output for a UI that is not being built.
2. **HADES is now fully specified**, in `project_HADES.md`, correcting several things the old draft got wrong or left vague: message passing needs the *reverse* of every relation to reach a Supplier at all (Section 6.2); a naive feature set leaks the label directly (Section 5.5); "Markov blanket" was an overstated claim, now correctly named a *structural depth prior* and tested by held-out comparison rather than asserted (Section 8.2); the global-attention bounding rule in the old draft was wrong in a way that would have excluded the exact cross-type correlation it exists to find (`project_HADES.md` §5.4); and the architecture ablation was missing a matched-parameter control arm without which "HGT won" is confounded with "HGT has more parameters" (Section 9.5).

## 5. Data Sources

The source-of-truth schema for everything below is `db/schema.sql` (19 tables, DDL only, model-relevant tables exclusively — application tables from the earlier product scope are not part of this schema) and the dataset it is populated by (`db/generate_dataset.py`, `db/README.md`). See `05_Database_Design.md` for the full table-by-table specification.

| Table group | Tables | Role |
|---|---|---|
| Master entities | `suppliers`, `components`, `products`, `factories`, `warehouses`, `customers` | Graph nodes |
| Structural / junction | `product_components`, `product_factories` | BOM and factory-qualification edges, both with validity windows |
| Operational (current-state, **display only**) | `inventory`, `orders`, `order_items`, `shipments` | Never read as model features directly — see Section 6.4 |
| History (leakage-safe features and labels derive from these) | `inventory_history`, `shipment_status_history`, `supplier_temporal_features`, `carrier_performance_snapshots` | The actual feature source |
| Snapshot / label infrastructure | `graph_snapshots`, `training_labels` | One row per `t0`; labels windowed to `(t0, t0+14d]` |
| Model output (write target, not an input) | `risk_scores` | Written by inference after training — the seed rows currently in `db/csv/risk_scores.csv` are deterministic weighted-formula placeholders for schema testing, not model output |

**Governance tables this document depends on but that do not yet exist in `schema.sql`** — `model_registry` and `model_evaluation_runs` (Section 11) — are scoped as an early roadmap step (`14_Model_Development_Roadmap.md`), not aspirational documentation with no path to being built.

## 6. Feature Engineering and the Leakage Contract

### 6.1 The contract, restated precisely (`project_HADES.md` §2.5)

For every training example, at prediction timestamp `t0` and forecast horizon `H = 14 days`:

```
FEATURES  ← computed ONLY from rows with timestamp ≤ t0
LABEL     ← did the event occur within (t0, t0 + H] ?
```

This is not a style preference — it is the difference between measuring forecasting skill and measuring whether the model can read its own answer off a column. `db/README.md`'s realism audit found the synthetic dataset's single-feature AUC on its strongest historical signal (`on_time_rate_90d` alone) sits at 0.59–0.63; if any solo feature scored above ~0.9, that would be a leak, and Step 1 of the roadmap re-runs this exact test before trusting anything downstream.

### 6.2 The exclusion list — columns that must never be read as features

| Column | Why it leaks | Use instead |
|---|---|---|
| `shipments.status`, `shipments.delivered_at` | Terminal/current state; the delay label is partly defined from these | Reconstruct as-of status from `shipment_status_history` at `t0` |
| `inventory.stock_level` | Current state; the shortage label is defined from stock breaching threshold | `inventory_history` as-of `t0` |
| `suppliers.reliability_history` | A single scalar rolled over *all* history, including the future relative to any given `t0` | `supplier_temporal_features` windows, all ending at `t0` |

### 6.3 Feature derivation

| Feature | Derivation | Node/Edge |
|---|---|---|
| `lead_time_days` (scaled) | Min-max or z-score over `suppliers.lead_time_days` | Supplier |
| On-time rate, 30/90/180-day windows | Rolling windows over `shipment_status_history`, ending at `t0`, from `supplier_temporal_features` | Supplier |
| Trend slope, lateness variance, days since last late | Derived the same way, same table | Supplier |
| `capacity_score` (scaled) | Normalized supplier-reported/inferred capacity | Supplier |
| `days_to_eta` / `days_to_due` | `eta - t0` / `due_at - t0`, recomputed per snapshot | Shipment / Order |
| `days_since_dispatch`, carrier/route on-time rate, seasonal index | From `shipment_status_history` + `carrier_performance_snapshots`, as-of `t0` | Shipment |
| `stock_level`, `reorder_threshold` (as-of `t0`) | `inventory_history` at or before `t0` | `STOCKED_AT` edge |
| `quantity_required` | `product_components.quantity_required`, filtered to BOM edges active as-of `t0` (`created_at ≤ t0 < deactivated_at`) | `USED_IN` edge |
| One-hot categoricals | `component_type`, `category`, `location` bucket, as-of `status` | Various |
| Label: `delayed` | Shipment transitioned to a late/delayed state within `(t0, t0+H]`, per `shipment_status_history` | Supplier/Shipment target |
| Label: `shortage` | Stock breached `reorder_threshold` within `(t0, t0+H]`, per `inventory_history` | Product/Warehouse target |
| Label: `impact` | Any of a supplier's in-flight shipments went late within the horizon (supplier-level; a simple, causal starting definition — `db/README.md` scope note) | Supplier target |

## 7. Graph Construction

### 7.1 Node types and feature dimensionality

Matches `project_HADES.md` §2.1, after the multi-window temporal additions:

| Node type | Features | Dim |
|---|---|---|
| Supplier | lead time, capacity, country one-hot (6 countries in the current dataset), **+6 temporal window features** | 21 (project figure; recompute against the real one-hot width) |
| Component | unit cost, `component_type` one-hot | 10 |
| Product | `category` one-hot (6 categories) | 8 |
| Factory | capacity, location one-hot | 10 |
| Warehouse | capacity, location one-hot | 10 |
| Shipment | `days_to_eta`, as-of status one-hot, **+4 temporal features** | 10 |
| Order | `days_to_due`, status one-hot | 6 |
| Customer | `priority_tier` one-hot | 4 |

Every type is projected into a shared hidden dimension `d = 64` before layer 1 (`project_HADES.md` §2.1): `h⁰_v = W_in[τ(v)] · x_v + b_in[τ(v)]`.

### 7.2 Meta-relations — forward and reverse

PyTorch Geometric message passing flows source → target only. The original 10-relation schema left a Supplier with almost no in-edges — it would never receive information about the components it supplies or the orders at risk downstream. Every relation is therefore paired with its reverse via `torch_geometric.transforms.ToUndirected()`, doubling the relation count to **20** (`project_HADES.md` §2.2):

| Forward | Reverse |
|---|---|
| `Supplier → SUPPLIES → Component` | `Component → rev_SUPPLIES → Supplier` |
| `Component → USED_IN → Product` | `Product → rev_USED_IN → Component` |
| `Product → STOCKED_AT → Warehouse` | `Warehouse → rev_STOCKED_AT → Product` |
| `Product → MANUFACTURED_AT → Factory` | `Factory → rev_MANUFACTURED_AT → Product` |
| `Shipment → SHIPS_FROM → Supplier` | `Supplier → rev_SHIPS_FROM → Shipment` |
| `Shipment → SHIPS_FROM → Factory` | `Factory → rev_SHIPS_FROM → Shipment` |
| `Shipment → SHIPS_TO → Warehouse` | `Warehouse → rev_SHIPS_TO → Shipment` |
| `Shipment → FULFILLS → Order` | `Order → rev_FULFILLS → Shipment` |
| `Order → ORDERED → Product` | `Product → rev_ORDERED → Order` |
| `Order → PLACED_BY → Customer` | `Customer → rev_PLACED_BY → Order` |

**Do not assume — measure.** After building the graph, compute the actual set of node types reachable in `k` hops per target type and use that measured table, not the theoretical one above, to justify the depth prior (Section 8.2).

### 7.3 As-of graph assembly

The graph for a given `t0` is assembled from the as-of feature set (Section 6.3) plus the BOM/factory-qualification edges active at `t0` (`product_components`/`product_factories` validity windows). One `HeteroData` object is built per `t0` in the snapshot schedule (`graph_snapshots`, currently monthly, `t0 = 2024-07-01 … 2024-12-01`), with `training_labels` attached from `(t0, t0+14d]`. There is no separate "ML-only" construction path and no serving-time construction path — this document only needs one.

## 8. Model Architecture

The full mathematical specification, worked numerical examples, parameter/FLOP budgets, and the reasoning behind every design choice live in `project_HADES.md` Parts 3–7; this section is the implementation-facing summary, organized around the four claims (Section 1).

### 8.1 Structural Encoder — Claim 1

**Production architecture: SHARE** (Shared Heterogeneous Attention over Relational
Edges, `rgcn_attn` in code — the identifier is unchanged from the ablation runs
below). Type-aware local structure via a basis-decomposed relation transform plus
one shared attention scorer with a joint softmax across every relation feeding a
destination node (`project_HADES.md` §3). `L = 4` layers, all four intermediate
outputs `h¹…h⁴` retained rather than only the last (Section 8.2 explains why).
HGT (below) was the originally-specified architecture and is retained in
`project_HADES.md` §3A as the mathematical reference SHARE's transform builds on
— it remains the strongest single-task performer, by raw mean, on the impact task.

**Committed architecture ablation, Claim 1's actual test — six architectures, not
three, run across four rounds of ablation** (full detail and all numbers:
`reports/rgcn_types.md`; this section is the implementation-facing summary, not a
restatement):

| Stage | Architecture | Why this stage | Outcome |
|---|---|---|---|
| 1 | GraphSAGE | Baseline — does neighborhood aggregation beat row-by-row models at all, with no attention | Beats HGT on shortage; type-blind everywhere else |
| 2 | GAT | Adds attention — *which* neighbors matter | Still type-blind; weakest architecture overall on delay/shortage |
| 3 | HGT | Graph is natively multi-typed (8 node types, 20 meta-relations); type-aware attention models that directly | Wins impact outright; loses shortage to GraphSAGE and (eventually) to every basis-sharing architecture |
| 4 | RGCN | Basis-decomposition transform (Schlichtkrull et al. 2018) — relation cost shared, not fully dedicated per relation like HGT | Wins shortage consistently; HGT's worst-impact challenger |
| 5 | **SHARE (`rgcn_attn`) — selected for production** | RGCN's transform + one shared attention scorer, joint softmax across relations | Wins delay and shortage consistently vs. HGT; ties impact — best net result of any architecture tried |
| 6 | Two further RGCN+attention hybrids — **SHARP** (`rgcn_relemb`) and **SHARK** (`rgcn_battn`) | Tested whether relation-identity signal or a second attention-dedicated basis pool improves on SHARE | Neither beats SHARE; SHARK is markedly less seed-stable |

All architectures trained/evaluated on the same held-out split, same loss and
regularization, under a distinct `model_version` each (Section 11), **at both a
fixed-`d` and a matched-parameter arm** (`d`/`num_bases` raised so every
architecture has a comparable parameter count) — without the matched arm, a
win is confounded between "the mechanism helped" and "more capacity helped"
(`project_HADES.md` §8.3).

### 8.2 Depth Selection — Claim 2

**The problem:** a single final-layer readout forces every prediction task through the same number of hops, but delay/shortage/impact targets sit at different distances from a Supplier (roughly 2, 3, and 3–4 hops respectively — `project_HADES.md` §4.2), and going deep enough for the farthest target over-smooths the nearest one.

**The fix, in two parts:**

1. **Structural depth prior (zero parameters)** — each prediction head reads a *specific* retained layer (`h²` for delay, `h³` for shortage/impact), chosen from graph connectivity, not learned. This alone is a real fix to a real modelling error and costs nothing.
2. **Learned depth gate (small, optional)** — a per-head gate (`d → 32 → 4`, ~2,212 parameters) initialized to reproduce the prior exactly, so training can only move away from it deliberately, anchored by a KL term (`λ = 0.01`) so departure has to be earned by the data. This is a *gate*, not a full Transformer — `project_HADES.md` §4.4 derives why a 2-layer attention block over four depth tokens would cost 45× the parameters and 180× the FLOPs of the gate for the same nominal output shape ("4 values summing to 1").

**`L = 4`, not 3, is deliberate**, even though the deepest prior is `h³`: one layer of headroom lets the gate express *upward* deviation (hubs, frontier nodes), not only downward. Stopping at `L=3` would censor the residual at the top — see `project_HADES.md` §4.6.

**Naming discipline:** a true Markov blanket is a property of a fitted joint distribution, not of a schema; graph distance alone proves no conditional independence. This document and `project_HADES.md` therefore call it a *structural depth prior* and settle its validity only by the held-out layer-depth sweep (Section 9.3), never by reading the gate's own weights as evidence (`project_HADES.md` §4.2, §4.4 "ordering rule").

### 8.3 Transformer 2 — Claim 3

**The problem it exists for:** two suppliers can fail together through a shared, unrecorded upstream source. No edge chain connects them, so message passing at any depth is structurally blind to it — this is the one gap in the whole architecture that cannot be closed by making HGT bigger (`project_HADES.md` §5.1). The synthetic dataset is built with exactly this scenario (`db/README.md`, "the hidden-dependency scenario") specifically so this claim has something real to be tested against.

**The mechanism:** dense self-attention over Supplier embeddings only, no adjacency mask (`project_HADES.md` §5.2–5.3):

- **Same-type constraint is mandatory, not a tuning choice** — attending across all node types would partially undo the type separation the structural encoder spends a substantial share of its parameters building (76% under HGT's own accounting, `project_HADES.md` §3A.5; SHARE's cost structure differs but the same type-separation principle holds).
- **Bounded candidate pool** — top-k by embedding cosine similarity (`k = 64`), always including frontier nodes (suppliers with no recorded upstream). An earlier draft bounded the pool by shared `component_type`, which is wrong: same-`component_type` suppliers are already 2-hop reachable via the co-parent path, so that bounding would have excluded exactly the cross-type correlation (different countries, different component types) the synthetic scenario and Transformer 2 both exist to find.
- **Output scope, Phase 1:** wired to the delay head only (supplier-level). Product/Order-level heads are not fed T2's output yet — extending that is a real interface decision, not a default, and is deferred until T2 has demonstrated it finds anything real (`project_HADES.md` §5.6).

**Validation, never by attention weight alone:** a high attention score means "similar within this candidate pool," not "shares an upstream supplier." Validated by (1) held-out `SUB_SUPPLIES` edge recovery where such edges exist, (2) future co-disruption rate among flagged pairs in the holdout period, and (3) cross-checking against a link-prediction head if one is built. Discovered pairs are persisted as investigation leads with their attention weight and `model_version`, never asserted as fact (`project_HADES.md` §5.8).

### 8.4 Prediction Heads

```
delay_v     = σ( MLP(z_delay_v) )       Supplier, Shipment
shortage_v  = σ( MLP(z_shortage_v) )    Product × Warehouse
impact_v    = σ( MLP(z_impact_v) )      Supplier (current label definition, Section 6.3)
```

Loss: focal loss per head (`γ=2`, `α` from inverse class frequency — disruptions are a minority class), plus the depth-gate's KL-anchor term. Total budget (`d=64`, `L=4`, all components, HGT-as-encoder design-time target): **773,311 parameters, 1.445 GFLOP full-graph forward pass** — see `project_HADES.md` Part 7 for the full breakdown and its production note (SHARE, the deployed encoder, is 752,211 real measured params at matched-`d` in place of that budget's HGT row; recompute once the gate and T2 are actually built).

### 8.5 Claim B — Dyadic Risk (Claim 4)

**The problem:** "Supplier X: 80% failure probability" conflates X's own fragility with *this firm's* exposure to it. A fragile supplier for whom this firm is a top-priority customer will be served first; a moderate-risk sole-source supplier is worse than the number suggests.

**Why it is deliberately not a model component:** the node is the structural encoder's unit of prediction (true of SHARE exactly as it was of HGT) — there is no architectural slot for a `(supplier, buyer)` pair, and reweighting by order-volume share and contract priority is auditable business arithmetic that a network would only make opaque for no representational benefit (`project_HADES.md` §6.4):

```
dyadic_risk = global_risk × f(order_volume_share, contract_priority_weight, fulfilment_preference_weight)
```

**The double-counting risk, and the required test before trusting it:** `Customer.priority_tier` is already a graph node feature the encoder can see via `PLACED_BY`, so Claim B may partly reweight by a signal the encoder already learned. Before this claim is reported as validated: train the encoder with and without `priority_tier` and compare Claim B's reordering rate; if the rates are close, the signals are independent, if the rate collapses without it, the encoder had already learned it. **`fulfilment_preference_weight` (historical rationing behavior, nowhere else in the schema) is the signal to lead on**, precisely because it is the one the encoder structurally cannot already know.

## 9. Training and Evaluation Protocol

### 9.1 Split

Time-based only, by `t0`: train on earlier snapshots, validate/test on later ones, never a random split — the same entity appearing across many snapshots would leak across a random split. Full run inventory, hyperparameters, and pipeline steps: `project_HADES.md` Part 8.

### 9.2 Runs required for a credible result

| Purpose | Runs |
|---|---|
| Layer-depth sweep (`L = 1,2,3,4`) | 4 |
| Architecture ablation (GraphSAGE, GAT, HGT at fixed `d`) | 3 |
| Matched-parameter arm | 3 |
| Depth gate on/off | 1 |
| Transformer 2 on/off | 1 |
| Claim B double-counting test | 1 |
| **Total** | **13** |

### 9.3 Metrics and confidence intervals — mandatory on every row

| Metric | Applies to | Target |
|---|---|---|
| AUC-ROC | Delay, shortage classification | ≥ 0.80 held-out (a target, not a pass/fail gate for the underlying claims) |
| Precision / Recall | Both classification heads | Reported per class |
| Calibration | Reliability diagram | A predicted probability should mean roughly what it says |
| Explanation stability | GNNExplainer across seeds | Subgraph overlap |
| Inference time, parameter count | Per architecture | Direct cost input to the ablation, not just an accuracy comparison |

At realistic label volumes, most comparisons here will not clear statistical significance:

| Positives | 95% CI on AUC | Smallest detectable gap |
|---|---|---|
| 200 | ±0.037 | ~0.037 |
| 500 | ±0.024 | ~0.024 |
| 1,000 | ±0.017 | ~0.017 |

**Report DeLong or bootstrap CIs on every ablation row.** "The architectures were not distinguishable at this label volume" is a legitimate, expected finding at this dataset's scale (current `training_labels` positive rates: ~11% delay, ~7% shortage, ~4% impact on ~1,774 rows — see `db/README.md`) and must be stated plainly, not obscured by reporting only point estimates.

### 9.4 Per-claim validation — how each claim is actually settled

| Claim | Component | Validated by | Never validated by |
|---|---|---|---|
| 1 | HGT vs. GraphSAGE/GAT | Matched-parameter ablation, CIs on every row | An unmatched comparison, or a single run |
| 2 | Depth prior / gate | Held-out layer-depth sweep; gate adopted only if it beats the prior baseline by more than the CI | Reading the gate's own learned weights as if they were proof |
| 3 | Transformer 2 | Held-out edge recovery, future co-disruption rate, cross-validation against a link-prediction head | A single high attention score in isolation |
| 4 | Claim B | Double-counting ablation (with/without `priority_tier`), then reviewer plausibility check on reordering | Assuming independence without running the test |

### 9.5 The single-feature leakage test (run before believing anything else)

Train on one feature at a time. Any solo feature scoring AUC > 0.9 leaks; fix and repeat. `db/README.md`'s realism audit already ran a version of this over the synthetic CSVs (max solo-feature AUC ≈ 0.63) — re-run it against the actual assembled `HeteroData` features, not just the raw CSVs, since the failure mode is in feature *assembly*, not only in the source columns.

## 10. Explainability

- **Mechanism:** GNNExplainer (or equivalent perturbation-based explainer) identifies the minimal subgraph responsible for a given prediction.
- **Role here:** a diagnostic and communication tool for presenting *why* the model flagged something during evaluation and thesis defense — not a served product feature, and not the instrument used to validate Claim 2 (Section 9.4 — attention weights measure usage, not sufficiency).
- **Output:** computed and logged per high-risk held-out prediction during evaluation; no dashboard rendering, no chatbot integration.

## 11. Model Governance

Every training run should write one row each to two tables not yet in `schema.sql` (roadmap: `14_Model_Development_Roadmap.md`, Step 5):

| Table | Fields | Purpose |
|---|---|---|
| `model_registry` | `model_version`, `architecture`, `training_dataset`, `training_timestamp`, `experiment_id`, `git_commit`, `hyperparameters`, `parameter_count`, `status` | Reproducibility — which code and data produced this exact model |
| `model_evaluation_runs` | `model_version`, `architecture`, `task`, metrics (AUC, precision, recall, F1, CI bounds, inference time) | Queryable, comparable results across the ablation and the layer-depth sweep |

Written by the training pipeline directly, never entered by hand, so these records cannot drift from what was actually run. This is intentionally lightweight — a version, dataset reference, experiment ID, git commit, and hyperparameters — not a full MLOps registry with automated promotion/rollback, which is out of scope for a prototype whose purpose is to answer a research question, not to serve traffic.

## 12. Known Limitations

| Limitation | Detail |
|---|---|
| Statistical power | At realistic label volumes, most architecture/component comparisons will be within the noise floor (Section 9.3) — report this honestly rather than only the point estimate that looks best |
| `SUB_SUPPLIES` / tier data | Uncommitted and near-empty in the current dataset; the relation-splitting design for tier×confidence bands (`project_HADES.md` §2.3) is specified but not buildable until this data exists |
| Snapshot, not temporal | The model reasons over discrete monthly snapshots, not a continuous temporal graph; multi-window features (Section 6.3) recover most of the practical benefit cheaply, a full temporal-sequence model does not |
| Correlation ≠ causation | Transformer 2's discoveries are statistical leads for a human reviewer, never proof of a shared supplier relationship |
| Single focal firm | Claim B's formulation assumes one buyer's perspective; documented, not solved |
| Not a novel architecture | HGT (2020), Jumping Knowledge (2018), and global attention (2021–22) are each established; the claim being tested here is that this specific composition, applied to a partially-observed heterogeneous supply-chain graph, is the right one for this problem class — not that any individual component is new |
| Analytical figures | All parameter/FLOP numbers in `project_HADES.md` are derived, not measured, until a real graph exists (Section 7 measurement note) |

---

## Document Control

- **Purpose:** Section 1. **Scope:** Section 2. **What changed from the product-scoped version:** Section 4. **Data sources:** Section 5. **Architecture:** Section 8 (full spec: `project_HADES.md`). **Training/evaluation protocol:** Section 9. **Limitations:** Section 12.
- Feeds `14_Model_Development_Roadmap.md` directly — this document specifies *what* is being built and why; the roadmap specifies the *order* to build and validate it in.
