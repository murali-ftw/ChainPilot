# HADES — Hierarchical Attention with Dual Encoders and Structure

*A complete technical reference for ChainPilot's Layer 2. Every component, every equation, every number — with the analogies that make them stick. Written to be read once, thoroughly, before implementation begins.*

**Supersedes** the configuration in `architecture.md`, which predates the correctness audit. Where the two disagree, this document wins.

> **Production architecture note (2026-08-07).** SHARE (Shared-basis Heterogeneous
> Attention Relational Encoder, `rgcn_attn` in code) is ChainPilot's production
> architecture as of this date, selected after a four-round ablation against HGT,
> GraphSAGE, GAT, and three other RGCN-family hybrids. See `reports/rgcn_types.md`
> and `reports/info.md` for the full evidence trail. **The code-level identifier is
> unchanged** — every call site, the `model_registry.architecture` column, and every
> already-logged historical row still read `rgcn_attn`; SHARE is a documentation name
> only, introduced here and used throughout this document from Part 3 onward. HGT's
> full derivation is retained in Part 3A as the original baseline architecture and as
> the mathematical reference SHARE's own transform builds on.

---

## Table of contents

| Part | Contents |
|---|---|
| 0 | What HADES is — the name, the thesis, the pipeline |
| 1 | Notation and dimensions |
| 2 | The graph — types, reverse relations, features, the leakage contract |
| 3 | **SHARE** — the structural encoder (production) |
| 3A | **HGT** — the original baseline encoder (superseded; math reference) |
| 4 | **Depth selection** — Markov prior + learned gate |
| 5 | **Transformer 2** — global attention across the visibility frontier |
| 6 | Prediction heads, Risk Intelligence, Claim B |
| 7 | Complete parameter and FLOP budget |
| 8 | Training protocol |
| 9 | Evaluation protocol |
| 10 | Known issues, including what has no fix |
| 11 | Implementation checklist |
| A | All formulae and constants, collected |

---

# PART 0 — WHAT HADES IS

## 0.1 Decoding the name

| Letter | Stands for | Which component |
|---|---|---|
| **H**ierarchical | Depth is chosen per task and per node, not fixed globally | Markov prior + depth gate |
| **A**ttention | Three distinct attention mechanisms, each with a different scope | SHARE (typed local, production; HGT originally), gate (depth), T2 (global same-type) |
| **D**ual **E**ncoders | Two encoding stages with different information sources | SHARE encodes *structure* (§3; HGT, §3A, is the original baseline it superseded); T2 encodes *statistical similarity* |
| **S**tructure | Graph geometry supplies the prior, not just the data | Blanket-derived depth, relation-scoped tier decay |

## 0.2 The thesis, in one paragraph

> Supply chain risk lives in a **partially-observed heterogeneous network**. The types matter (a supplier is not a warehouse), the depth matters differently per prediction (delay is local, impact is far), the couplings that cause correlated failure are frequently **not recorded as edges**, and the resulting score belongs to a *relationship*, not to a node. HADES uses one type-aware encoder for what the graph records, one global-attention stage for what it doesn't, structural geometry to decide how far each prediction should look, and a deterministic reweighting to convert supplier fragility into buyer exposure.

## 0.3 The four design principles

1. **Every component names one thing the previous one structurally cannot do.** Nothing is decorative. If you can't name the gap, cut the component.
2. **Prefer structure over learning when structure is available.** Depth comes from graph geometry (free) before it comes from learned attention (expensive) — against a few hundred positive labels, every parameter must justify itself.
3. **Persist every stage's output.** Depth weights, dependency links, dyadic scores are all written to disk. A reviewer can inspect *why* a prediction looks the way it does at each stage, rather than facing one opaque score.
4. **Never let a mechanism double as its own evidence.** Attention weights measure what the model *used*, not what was *sufficient*. Depth claims are tested by held-out ablation, never by reading the gate.

## 0.4 The pipeline

```
   PostgreSQL  ──►  snapshot at t₀  (leakage contract, §2.5)
                          │
                          ▼
              HeteroData: 8 node types, 20 meta-relations
                          │
        ┌─────────────────▼─────────────────┐
        │  SHARE ENCODER         §3          │   type-aware local structure
        │  L = 4 layers, all retained        │   752,211 params (matched-d)
        └─────────────────┬─────────────────┘   `rgcn_attn` in code — HGT's
                                                  original §3A box: 698,448 params
                          │  h¹ h² h³ h⁴
        ┌─────────────────▼─────────────────┐
        │  DEPTH SELECTION      §4          │   how far should each
        │  Markov prior + gate ×3           │   prediction look?
        └─────────────────┬─────────────────┘   6,636 params · 0.065 GFLOP
                          │  z_delay  z_short  z_impact
        ┌─────────────────▼─────────────────┐
        │  TRANSFORMER 2        §5          │   who is coupled without
        │  suppliers only, bounded pool     │   a recorded edge?
        └─────────────────┬─────────────────┘   49,984 params · 0.069 GFLOP
                          │
        ┌─────────────────▼─────────────────┐
        │  PREDICTION HEADS     §6          │   delay · shortage · impact
        └─────────────────┬─────────────────┘   12,675 params
                          │
        ┌─────────────────▼─────────────────┐
        │  RISK INTELLIGENCE    §6.3        │   confidence, category,
        │  incl. Claim B (dyadic)           │   dyadic reweighting
        └───────────────────────────────────┘   0 params — deterministic

                TOTAL: 773,311 params · 1.445 GFLOP
```

**Read the right column downward.** Each box exists because the one above it cannot do that job. That chain is the architecture's entire argument.

---

# PART 1 — NOTATION AND DIMENSIONS

Fix these once; every formula below uses them.

| Symbol | Meaning | Value | Where it comes from |
|---|---|---|---|
| `d` | Hidden dimension | **64** | Chosen — see §7.4 for why not 128 |
| `h` | Attention heads | **4** | Standard; `d/h = 16` per head |
| `d/h` | Per-head dimension | **16** | |
| `L` | Encoder layers | **4** | §4.6 — deepest blanket + 1 headroom |
| `T` | Node types | **8** | Doc 6 §5 |
| `R` | Meta-relations | **20** | 10 forward + 10 reverse (§2.2) |
| `V` | Nodes | **22,155 – 66,973** | Measured, `ml/graph/builder.py` against the v3 dataset (15 snapshots, Jul 2024 - Sep 2025) — grows snapshot to snapshot as orders/shipments accumulate; NFR-02's ~5,000 ceiling was a pre-graph planning estimate, superseded (`reports/step5_result_v3.md`) |
| `E` | Directed edges (20 meta-relations, forward+reverse) | **122,594 – 388,242** | Measured, same source; ~40,000 was the planning estimate |
| `N_s` | Supplier nodes | **800** | Measured (`db/generate_dataset.py`'s `SUP_N`) — no longer an estimate |
| `k` | T2 candidate pool size | **64** | §5.4 |
| `τ` | Prior width | **0.8** | §4.3 |
| `λ` | Prior KL strength | **0.01** | §4.3 |
| `t₀` | Prediction timestamp | per snapshot | §2.5 |
| `H` | Forecast horizon | **14 days** | §2.5 |

**Notation conventions**

- `x_v` — raw feature vector of node `v`
- `h^ℓ_v` — embedding of node `v` after layer `ℓ`
- `τ(v)` — the *type* of node `v` (Supplier, Product, …). Distinct from prior width `τ` — context disambiguates.
- `φ(e)` — the *relation* of edge `e`
- `⟨τ(s), φ(e), τ(t)⟩` — a **meta-relation** triple
- `N(t)` — the in-neighbours of `t`, i.e. nodes with an edge pointing *at* `t`
- `z_v` — the fused embedding after depth selection

---

# PART 2 — THE GRAPH

## 2.1 Node types and features

Feature dimensions **after** the multi-window temporal additions (§2.4):

| Node type | Features | `F_τ` |
|---|---|---|
| Supplier | lead time, reliability, capacity, country one-hot (12), **+6 temporal** | **21** |
| Component | unit cost, component_type one-hot | 10 |
| Product | category one-hot | 8 |
| Factory | capacity, location one-hot | 10 |
| Warehouse | capacity, location one-hot | 10 |
| Shipment | days_to_eta, status one-hot, **+4 temporal** | **10** |
| Order | days_to_due, status one-hot | 6 |
| Customer | priority_tier one-hot | 4 |

Every type is projected into the shared `d = 64` space before layer 1:

```
h⁰_v  =  W_in[τ(v)] · x_v  +  b_in[τ(v)]           W_in[τ] ∈ ℝ^{64 × F_τ}
```

**Parameters:** `Σ_τ (F_τ · 64 + 64) = 5,568`

## 2.2 Meta-relations — and why reverse edges are mandatory

**This is the fix that most changes the numbers.** PyTorch Geometric message passing flows **source → target only**. Look at which edges point *into* a Supplier under the original schema:

- `SHIPS_FROM` (Shipment → Supplier) ✓
- `SUB_SUPPLIES` (Supplier → Supplier) — uncommitted, near-empty

**That's it.** A Supplier would *never* receive information about the components it supplies, the products those feed, or the orders at risk. The co-parent path that justifies the L=2 floor —

```
Supplier_A  ──SUPPLIES──►  Component  ◄──SUPPLIES──  Supplier_B
```

— requires traversing the second edge **backwards**. Without a reverse relation, that path does not exist for message passing.

> **Data status of this path — corrected, active as of `reports/entropy_test.md`.**
> For most of this project's history, this path existed structurally but returned
> nothing: `components.supplier_id` was a single not-null FK, so `Supplier_B` above
> was always the same node as `Supplier_A` (0/800 suppliers measured reaching a
> *different* supplier this way). That changed once dual-sourcing shipped
> (`component_suppliers`, the co-parent mechanism) — the currently loaded database
> has it **active**: 420 components (17.5%) carry a second qualified supplier, and
> **449/800 suppliers (56.1%) now measurably reach a different supplier in 2 hops**
> (`reports/entropy_test.md`'s dataset-version finding). This is NOT the same thing
> as `SUB_SUPPLIES` below, which remains a separate, genuinely unbuilt relation —
> see the note there.

```python
from torch_geometric.transforms import ToUndirected
data = ToUndirected()(data)      # adds rev_* for every relation
```

| | Forward only | With reverse |
|---|---|---|
| Meta-relations `R` | 10 | **20** |
| Directed edges `E` | 20,000 | **40,000** |
| HGT params / layer | 154,122 | **174,612** (+13.3%) |
| HGT GFLOP (L=4) | 0.655 | **1.311** (+100%) |

The 20 meta-relations:

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

> **Do not assume — measure.** After building the graph, for each target node type compute the set of node types actually reachable in `k` hops. Print the table. That measured table, not the theoretical argument in §4, is what justifies the depth prior.

## 2.3 Tier and confidence — how they enter the model

**The problem:** HGT's learnable relation prior `μ` is **one scalar per meta-relation**. Tier and confidence are **per-edge** attributes. And `HGTConv` in PyG accepts `(x_dict, edge_index_dict)` — no `edge_attr`. You cannot express *"this particular edge has confidence 0.4"* through a relation-level scalar.

**The fix — make the edge property a relation property by splitting:**

```
SUB_SUPPLIES_t2_high    (tier 2, confidence ≥ 0.7)
SUB_SUPPLIES_t2_low     (tier 2, confidence < 0.7)
SUB_SUPPLIES_t3_high    (tier 3, confidence ≥ 0.7)
SUB_SUPPLIES_t3_low     (tier 3, confidence < 0.7)
```

Now `μ` expresses tier decay and confidence weighting natively, at **~4,096 params/layer** and zero new machinery. Initialise:

```
μ[SUB_SUPPLIES_t2_*]  ←  α¹  ≈ 0.30
μ[SUB_SUPPLIES_t3_*]  ←  α²  ≈ 0.09
```

and let training adjust from there. The model *learns* per-band reliability rather than being told it.

**Status:** `SUB_SUPPLIES` — a direct, tiered Supplier→Supplier edge — is still uncommitted and near-empty; no such table or edge type exists in the current schema. This section remains a design for when that data exists. **Do not confuse this with the co-parent path** (§2.2's `Supplier→SUPPLIES→Component→rev_SUPPLIES→Supplier`), which is a different, already-built mechanism now confirmed active — `reports/entropy_test.md`.

## 2.4 Multi-window temporal features

A single scalar `reliability_history` cannot encode *"this supplier is currently degrading."* Two suppliers both sliding downward look identical to two suppliers who are both chronically mediocre. Add to **Supplier** (+6 dims):

| Feature | Encodes |
|---|---|
| On-time rate, 30-day window | Recent state |
| On-time rate, 90-day window | Medium-term state |
| On-time rate, 180-day window | Baseline state |
| Trend slope over last *N* shipments | **Degrading vs. stable** |
| Variance of lateness | Erratic vs. consistently poor |
| Days since last late shipment | Recency |

And to **Shipment** (+4): days_since_dispatch, carrier rolling on-time rate, route rolling on-time rate, seasonal index.

**Cost: +640 parameters** (input projection only). This is the cheap 80% of temporal modelling — enough to make "both suppliers are currently degrading" *representable*, which is what T2's correlation story actually requires. A full temporal graph (snapshot sequences, `HeteroData` → `List[HeteroData]`) is a genuine architectural change and is **out of scope**.

## 2.5 The leakage contract — read this twice

**The single most important section in this document.** Without it, every number you produce is meaningless.

**The problem, concretely:** `shipments.status` is listed as a Shipment node feature. The `delayed` label is defined as `status = 'delayed' OR delivered_at > eta`. **The label is sitting in the input.** A model can score AUC 0.99 by reading one one-hot slot, and learn nothing. The same structure holds for shortage: `stock_level` and `reorder_threshold` are `STOCKED_AT` edge features, and the shortage label is stock breaching threshold.

**Analogy.** You're predicting exam results. Your feature list includes "final grade." The model achieves perfect accuracy and has learned nothing about studying. Every downstream conclusion — which architecture is better, whether depth matters, whether T2 finds anything — is decided by that one column.

### The contract

```
For each training example:
    t₀  =  prediction timestamp
    H   =  forecast horizon (14 days)

    FEATURES ← computed ONLY from rows with timestamp ≤ t₀
    LABEL    ← did the event occur within (t₀, t₀ + H] ?
```

### The exclusion list

| Field | Currently | Must become |
|---|---|---|
| `shipments.status` | terminal status — **LEAKS** | status **as of t₀**: `pending` / `in_transit` only |
| `shipments.delivered_at` | in features — **LEAKS** | **excluded entirely** — it *is* the label |
| `shipments.eta` | in features | keep — known at `t₀` |
| `inventory.stock_level` | current — **LEAKS** | level **as of t₀** |
| `reliability_history` | rolling over all history — **LEAKS** | rolling over `[t₀ − N, t₀]` only |
| Temporal windows (§2.4) | — | all windows end at `t₀` |
| Delay label | `status='delayed'` | shipment became late **within `(t₀, t₀+H]`** |
| Shortage label | stock < threshold | stock breached threshold **within `(t₀, t₀+H]`** |

### The one architectural consequence

You now need **many snapshots**, one per `t₀`, not one:

```
t₀ = 2024-01-15  →  snapshot₁ + labels from (01-15, 01-29]
t₀ = 2024-02-15  →  snapshot₂ + labels from (02-15, 02-29]
...
```

Roll `t₀` forward monthly (or weekly) across your history. This is a **training-pipeline** change, not a model change — but it's the largest single piece of work in the project.

**Split by time, never at random:** train on `t₀ ≤ T_train`, validate on `T_train < t₀ ≤ T_val`, test on `t₀ > T_val`. A random split leaks across snapshots because the same entity appears in many.

### The leakage test

Before believing any result: **train on a single feature at a time.** If any solo feature reaches AUC > 0.9, it leaks. Fix and repeat. This test costs an hour and has saved more projects than any architecture choice.

---

# PART 3 — SHARE: THE STRUCTURAL ENCODER (PRODUCTION)

*SHARE — Shared-basis Heterogeneous Attention Relational Encoder, `rgcn_attn` in
code — is ChainPilot's production structural encoder, selected over HGT (§3A),
GraphSAGE, GAT, and two other RGCN-family hybrids after a four-round ablation
(`reports/rgcn_types.md`). Everything below is real, measured behaviour — not the
theoretical derivation §3A uses — because SHARE's own numbers were never worked
out on paper first; they came directly from `ml/models/rgcn_attn_encoder.py`
trained against the live database.*

## 3.1 The idea in one picture

HGT (§3A) gives every meta-relation its own **fully dedicated** pair of
`16×16` weight matrices — one language lesson per conversation type, paid for
in full every time. SHARE keeps the "different node types speak different
languages" half of that idea (§3.2 below) but replaces the "every conversation
gets its own bespoke etiquette" half with something cheaper and, on this data,
better: every relation's transform is a **linear combination of a small shared
phrasebook**, and instead of scoring attention separately within each
conversation, everyone in the room competes for the same, single attention
budget regardless of which language they're speaking.

| Model | Behaviour in that room |
|---|---|
| GraphSAGE | Everyone talks at once; you average the noise |
| GAT | You listen *harder* to some — but they're still speaking languages you don't know |
| HGT (§3A) | Every profession gets a **dedicated translator and its own etiquette** |
| **SHARE** | Every profession draws its translator from a **small shared phrasebook** (§3.3) — cheaper than a dedicated one — and **one attention rule, not per-profession rules**, decides who gets heard |

## 3.2 The meta-relation — unchanged from HGT

SHARE's message *transform* still parameterises by the triple
`⟨τ(source), φ(edge), τ(target)⟩` — the same 20 meta-relations §3A.2 describes,
`⟨Shipment, SHIPS_FROM, Supplier⟩` and `⟨Shipment, SHIPS_FROM, Factory⟩` still
kept apart. What changes is *how* each relation's own parameters are built, not
which relations exist.

## 3.3 The mathematics

### Step 1 — the transform: basis decomposition, not a dedicated matrix

For relation `r`, instead of HGT's own dedicated `W_MSG_i[φ]` per relation:

```
W_r  =  Σ_b  a_r[b] · V_b                    V_b ∈ ℝ^{d × d},  b = 1..num_bases
                                              a_r ∈ ℝ^{num_bases}  (one vector per relation)
msg(s→t)  =  h_s · W_r
```

`V_b` (the basis pool, `rel_basis` in code) and `a_r` (`rel_coeff`) are **shared
across every layer** — a single set for the whole encoder, so this part of the
parameter cost never scales with depth `L`. Every relation still gets its own
`W_r`, but it's built from `num_bases` shared components rather than paid for
from scratch — this is the entire cost saving relative to HGT's dedicated
`W_ATT`/`W_MSG` per relation (§3A.3).

### Step 2 — attention: one shared scorer, one joint softmax

Where HGT computes `K·W_ATT[φ]·Qᵀ` — a dedicated bilinear form per relation —
SHARE scores every edge with the **same two vectors for every relation and
every node type**:

```
logit(s→t)  =  LeakyReLU( att_msg · msg(s→t)  +  att_dst · h_t )

α(s→t)  =  ─────────exp(logit(s→t))─────────
            Σ_{s' → t, ANY relation} exp(logit(s'→t))
```

`att_msg`, `att_dst` ∈ ℝ^d, one pair **per layer**, shared across all 20
relations — not per-relation like HGT's `μ`. The softmax's sum runs over
**every** in-edge to `t`, mixed across relations, not per relation then
recombined — the same "one softmax, type-awareness lives elsewhere" principle
HGT's own §3A.3 Step 3 already established, just with the competition now
spanning relations too, not only neighbours within one relation.

### Step 3 — aggregate and update

```
h_t'  =  self_loop(h_t)  +  Σ_{s → t}  α(s→t) · msg(s→t)
```

`self_loop` is a full, NOT basis-shared, `d×d` matrix per node type **per
layer** — this is what plays HGT's residual role (§3A.3 Step 5), carrying each
node's own identity forward rather than overwriting it.

## 3.4 Parameter counts — real, measured, not derived on paper

| Arm | hidden | num_bases | Params (full model: encoder + 3 task heads) |
|---|---:|---:|---:|
| Fixed-`d` | 64 | 8 (default) | **170,984** |
| **Matched-`d` (production)** | **128** | **10** | **752,211** |

Matched-`d` was chosen by grid search to land near HGT's own real, measured
`d=64` anchor (**713,763** full-model params — not the theoretical 704,016 §3A.5
derives on paper; feature dimensions turned out to differ slightly from the
doc's placeholder estimates once measured against the real database, same
correction §3A itself flags at the end of §3A.7). At fixed-`d=64`, SHARE costs
**170,984 — less than a quarter of HGT's** — because the basis-shared transform
and the two-vector attention scorer are both far cheaper than a dedicated
matrix per relation.

## 3.5 Real results — the ablation verdict

Five seeds, paired bootstrap significance, both fixed-`d` and matched-`d`
(`reports/hades_model_development_report.md` Round 6, `reports/rgcn_matched_pilot.md`):

| Task | vs. HGT, fixed-`d` | vs. HGT, matched-`d` |
|---|---|---|
| delay | **SHARE wins**, consistent across all 5 seeds (+0.0112) | **SHARE wins**, consistent (+0.0103) |
| shortage | **SHARE wins**, consistent (+0.0162) | **SHARE wins**, consistent (+0.0147) |
| impact | Statistical tie (sign flips across seeds, +0.0058 mean) | Statistical tie (sign flips, +0.0052 mean) |

**SHARE wins two of three tasks outright, at both parameter budgets, and never
loses to HGT on any task.** The fixed-`d` win survives matching parameters
almost unchanged (`reports/rgcn_matched_pilot.md`'s before/after table: all
three tasks move by under 0.0015) — not a smaller-model-regularises-better
artifact. Impact remains HGT's own best task by raw mean
(`reports/rgcn_types.md`'s closing comparison), which is why §3A is retained in
full rather than deleted.

## 3.6 Why SHARE, not HGT — the ablation in one line

HGT pays for a **fully dedicated** per-relation parameter set on all 20
meta-relations, including the four thinnest (`MANUFACTURED_AT`, `SUPPLIES`,
`STOCKED_AT`, `USED_IN` — `reports/entropy_test.md`'s relation-frequency
measurement). Shortage's own causal signal draws directly on two of those four
— exactly where a fully-dedicated parameterisation is most exposed to
overfitting on sparse data, and exactly the task where every basis-sharing
architecture tried (plain RGCN, SHARE, and SHARE's own two further hybrids)
beats HGT consistently. SHARE adds a shared attention step on top of that
basis-sharing, closing HGT's other advantage (impact) to a tie without giving
back the shortage or delay wins — the best net result of every architecture
tried across four rounds of ablation (`reports/rgcn_types.md`'s closing table).

---

# PART 3A — HGT: THE ORIGINAL BASELINE ENCODER (SUPERSEDED)

**This part is retained, unmodified, for two reasons:** SHARE's own transform (§3)
is a direct simplification of the per-relation math derived below, so this is the
reference those simplifications are measured against; and HGT remains the strongest
single-task performer of any architecture tried, by raw mean, on the impact task
(`reports/rgcn_types.md`). Every number and derivation below describes HGT
specifically and is unchanged from before SHARE was selected — nothing here was
retroactively edited to fit SHARE's numbers.

## 3A.1 The idea in one picture

Your graph is a meeting where people speak different professional languages. Suppliers speak logistics. Products speak engineering. Orders speak sales.

| Model | Behaviour in that room |
|---|---|
| GraphSAGE | Everyone talks at once; you average the noise |
| GAT | You listen *harder* to some — but they're still speaking languages you don't know |
| **HGT** | Every profession gets a **translator**, and every *kind of conversation* gets its own **etiquette** |

Everything below implements those two ideas.

## 3A.2 The meta-relation

HGT parameterises by the triple, not the edge:

```
⟨ τ(source) , φ(edge) , τ(target) ⟩
```

`⟨Shipment, SHIPS_FROM, Supplier⟩` and `⟨Shipment, SHIPS_FROM, Factory⟩` are **different meta-relations** — same edge type, different target. GAT would collapse them; HGT keeps them apart.

## 3A.3 The mathematics

For target node `t`, source node `s`, edge `e = (s → t)`, at layer `ℓ`, for each head `i ∈ {1..h}`:

### Step 1 — type-specific projections (the translators)

```
K_i(s)  =  K_Lin_i[τ(s)] · h^{ℓ-1}_s          K_Lin_i[τ] ∈ ℝ^{16 × 64}
Q_i(t)  =  Q_Lin_i[τ(t)] · h^{ℓ-1}_t          Q_Lin_i[τ] ∈ ℝ^{16 × 64}
M_i(s)  =  M_Lin_i[τ(s)] · h^{ℓ-1}_s          M_Lin_i[τ] ∈ ℝ^{16 × 64}
```

Different node type ⇒ different weight matrix. A Component and an Order **cannot** be confused, because they were never projected by the same function.

### Step 2 — relation-specific attention (the etiquette)

```
                    ⎛                                    ⎞
ATT_i(s,e,t)  =     ⎜ K_i(s) · W_ATT_i[φ(e)] · Q_i(t)ᵀ  ⎟ ·  μ[⟨τ(s),φ(e),τ(t)⟩]
                    ⎝                                    ⎠   ─────────────────────
                                                                     √(d/h)

     W_ATT_i[φ] ∈ ℝ^{16 × 16}          μ[·] ∈ ℝ   (one learnable scalar per meta-relation)
```

`W_ATT` sits **between** key and query — this is exactly what GAT cannot express. `μ` is the per-relation importance dial, and where tier decay is imposed (§2.3).

### Step 3 — softmax over ALL in-neighbours

```
                       exp( ATT_i(s,e,t) )
α_i(s,e,t)  =  ──────────────────────────────────
                Σ_{s' ∈ N(t)}  exp( ATT_i(s',e',t) )
```

**One softmax across every in-neighbour, regardless of type.** A Component and an Order compete for the same attention budget. Type-awareness lives in the *projections*, not in separate competitions. This is a common misreading — worth pinning down now.

### Step 4 — messages

```
MSG_i(s,e,t)  =  M_i(s) · W_MSG_i[φ(e)]              W_MSG_i[φ] ∈ ℝ^{16 × 16}
```

Same two-part structure: the **source type's** matrix extracts what that kind of node has to say; the **relation's** matrix reshapes it for that conversation.

### Step 5 — aggregate and update

```
h̃_t  =  ‖_{i=1..h}  Σ_{s ∈ N(t)}  α_i(s,e,t) · MSG_i(s,e,t)          ‖ = concat over heads

h^ℓ_t  =  A_Lin[τ(t)] · σ( h̃_t )  +  h^{ℓ-1}_t                        A_Lin[τ] ∈ ℝ^{64 × 64}
                                        ↑
                                    residual — why depth is survivable
```

The residual is why HGT tolerates depth better than plain GNNs: each node keeps its own identity, and layers **add refinements** rather than overwriting.

## 3A.4 Worked numerical example

**Product P** updating at layer 1. In-neighbours (with reverse relations active):

| Neighbour | Type | Relation |
|---|---|---|
| C1 | Component | `USED_IN` |
| C2 | Component | `USED_IN` |
| O1 | Order | `ORDERED` |
| W1 | Warehouse | `rev_STOCKED_AT` |

**Head 1, step by step:**

```
Q₁(P)  = Q_Lin₁[Product]   · h⁰_P     →  ℝ¹⁶
K₁(C1) = K_Lin₁[Component] · h⁰_C1    →  ℝ¹⁶      ← Component matrix
K₁(O1) = K_Lin₁[Order]     · h⁰_O1    →  ℝ¹⁶      ← DIFFERENT matrix
K₁(W1) = K_Lin₁[Warehouse] · h⁰_W1    →  ℝ¹⁶      ← DIFFERENT again
```

Raw scores through relation-specific matrices, scaled by `√16 = 4` and each meta-relation's `μ`:

| Neighbour | `K·W_ATT·Qᵀ` | `μ` | scaled score | `exp` | **α** |
|---|---|---|---|---|---|
| C1 | 8.4 | 1.20 | 2.52 | 12.43 | **0.47** |
| C2 | 6.0 | 1.20 | 1.80 | 6.05 | **0.23** |
| O1 | 5.2 | 0.95 | 1.24 | 3.46 | **0.13** |
| W1 | 6.8 | 0.85 | 1.45 | 4.26 | **0.16** |
| | | | | Σ = 26.20 | Σ = 1.00 |

Then:

```
h̃_P  =  0.47·MSG(C1) + 0.23·MSG(C2) + 0.13·MSG(O1) + 0.16·MSG(W1)
h¹_P  =  A_Lin[Product] · σ(h̃_P)  +  h⁰_P
```

**What GAT would do:** one shared `W` for all four neighbours. It could still learn those same α values — but it has **no way to express** that *"a component supplying me"* and *"an order demanding me"* are fundamentally different kinds of information. Listening harder without understanding the language.

## 3A.5 Parameter derivation

Per layer:

| Group | Formula | Count |
|---|---|---|
| Node projections (K, Q, M, A) | `T · 4 · (d² + d)` = `8 · 4 · (4096 + 64)` | 133,120 |
| Relation matrices (ATT, MSG) | `R · 2 · h · (d/h)²` = `20 · 2 · 4 · 256` | 40,960 |
| Relation priors `μ` | `R` | 20 |
| Skip/norm | `T · d` | 512 |
| **Per layer** | | **174,612** |

```
Input projections           5,568
HGT encoder (L=4)         698,448      = 4 × 174,612
                        ─────────
Encoder subtotal          704,016
```

**The 76 / 24 split.** Node-type parameters are 76% of each layer; relation parameters are 24%. Practical consequence: **adding a node type costs ~4× more than adding a relation.** Splitting `SUB_SUPPLIES` into four tier×confidence bands costs 4,096/layer. Adding a `Contract` node type would cost 16,640/layer plus its input projection.

## 3A.6 FLOP derivation

Counting `FLOPs = 2 × MACs`, dominant terms only:

```
Node projections:   2 · 4 · V · d²        = 2 · 4 · 5,000 · 4,096   = 163.8 MFLOP
Edge ATT + MSG:     2 · E · 2 · h · (d/h)² = 2 · 40,000 · 2 · 1,024  = 163.8 MFLOP
                                                                     ─────────────
                                                    per layer          327.7 MFLOP
                                                    L = 4            1,310.7 MFLOP
```

**HGT encoder: 1.311 GFLOP.**

Note the near-perfect balance between node and edge terms at these dimensions. Doubling edges (reverse relations) doubled the edge term; the node term is unaffected.

## 3A.7 Depth and reach — measured, not assumed

With reverse relations, average undirected degree ≈ `2E/V` = **16**.

| L | Params | GFLOP | Uniform-branching reach |
|---|---|---|---|
| 1 | 192,855 | 0.328 | ~0% |
| 2 | 367,467 | 0.655 | 5% |
| 3 | 542,079 | 0.983 | **87%** |
| **4** | **716,691** | **1.311** | **100%** |
| 5 | 891,303 | 1.638 | 100% — nothing new |

**Two things to internalise.**

*First,* reverse relations moved saturation from L=4 to **L=3**. That is a large change and it comes entirely from a correctness fix, not a design choice.

*Second — and this trips people up — reach and information are not the same thing.* At L=4 nearly every node is *reachable*, which means `h⁴` is heavily **over-smoothed**: closer to a graph-wide average than to a description of any particular node. So `h⁴` isn't useless because it's uninformative in principle; it's low-information because repeated neighbourhood averaging has washed out the differences.

**This is why L=4 is still correct in HADES** — see §4.6. `h⁴` exists as *headroom* for the depth gate, and the expected outcome is that the gate assigns it near-zero weight almost everywhere. **That outcome is the result, not a failure.** It is the empirical confirmation of the depth prior.

> These reach figures assume uniform branching. Real supply chain graphs are hub-heavy (one component supplied by one supplier, used in many products), so actual saturation arrives **faster**. Replace this table with measurements from your real graph before citing it.

## 3A.8 Pros

| Advantage | Why it matters here |
|---|---|
| Type-aware by construction | Your graph genuinely has 8 kinds of thing |
| No hand-designed meta-paths | HAN/metapath2vec need you to specify `Supplier→Component→Product`; HGT learns which paths matter |
| Parameters scale as `(T + R)`, not `T²R` | Makes a rich schema affordable |
| Residual connections | Trainable at depth without immediate collapse |
| Inspectable attention | Diagnostic hooks for free |
| First-class PyG support | `HGTConv` exists |

## 3A.9 Cons

| Limitation | Consequence |
|---|---|
| Most expensive of the three ablation stages | ~3× GraphSAGE at equal `d` |
| Fixed receptive field | `L` layers = exactly `L` hops |
| Over-smoothing at depth | Residuals delay it; they don't prevent it |
| No sense of time | One snapshot — mitigated but not solved by §2.4 |
| Rare relations get badly-estimated parameters | `SUB_SUPPLIES` bands would train on very few examples |
| No native `edge_attr` | Confidence must be encoded via relation splitting (§2.3) |

## 3A.10 Why HGT alone is not enough — the five gaps

| # | Gap | Root cause | Filled by |
|---|---|---|---|
| 1 | Cannot couple nodes with **no path** | Message passing walks edges; no edge, no message — at any depth | **T2** (§5) |
| 2 | One depth for three differently-positioned targets | Single final-layer output | **Markov prior** (§4) |
| 3 | Deep enough for impact ⇒ over-smoothed for delay | Repeated averaging | **Intermediate retention + gate** (§4) |
| 4 | Produces a **global** score; risk is **relational** | The node is the prediction unit; no slot for a (supplier, buyer) pair | **Claim B** (§6.3) |
| 5 | Cannot distinguish *"no upstream"* from *"unknown upstream"* | Absent edges carry no signal | **`is_frontier` + T2** (§5.5) |

**HGT is the right encoder and the wrong complete system.** Every other component in HADES exists because HGT solves exactly one problem — type-aware local structure — and nothing else.

---

# PART 4 — DEPTH SELECTION

## 4.1 The problem

The structural encoder (SHARE in production, §3; originally HGT, §3A — both share
this property, since HADES's own convention is to retain every layer, §0.4)
computes `h¹…h⁴` and, by default, discards three of them:

```
h⁰ →[1]→ h¹ →[2]→ h² →[3]→ h³ →[4]→ h⁴
          ✗        ✗        ✗        ✓ keep only this
```

That assumes every node and every task needs exactly 4 hops. **Both assumptions are wrong.**

## 4.2 The structural prior — where depth comes from

### The blanket concept

A node's **Markov blanket** is the smallest set such that, knowing it, nothing else in the graph adds information. Three parts: **parents**, **children**, **co-parents** (the *other* causes of its children).

**Analogy.** To predict whether a student passes an exam: their study habits (parent), their mock scores (child), and the paper's difficulty (co-parent — the *other* thing that determined the mocks). Their grandparents' habits add nothing once you know the student.

> **Naming discipline.** A true Markov blanket is a property of a *fitted joint distribution*, not of a schema. Graph distance alone establishes no conditional independence. **Call this a "task-specific structural depth prior."** Use blanket reasoning as *motivation*, and validate the choice by held-out performance (§9.3) — never by asserting the theory.

### The blanket of a Supplier node

| Part | In your graph | Hops | Requires |
|---|---|---|---|
| Children | Components it supplies | 1 | `SUPPLIES` |
| **Co-parents** | Other suppliers of those components | **2** | `SUPPLIES` + **`rev_SUPPLIES`** |
| Parents | Upstream suppliers | 1 | `SUB_SUPPLIES` (sparse) |
| Moralised co-parents | Suppliers sharing an *unobserved* upstream | ∞ | **no path — T2's job** |

The co-parent term gives a **derived floor of L = 2** — counted, not guessed. Note it depends entirely on the reverse relation existing (§2.2) **and on dual-sourced data actually being present**, which is now the case (§2.2's data-status note; `reports/entropy_test.md` measures 449/800 suppliers, 56.1%, reaching a co-parent this way on the currently loaded database).

### The prior is per-target

| Head | Target sits on | Path from a Supplier | Prior |
|---|---|---|---|
| **Delay** | Supplier / Shipment | children + co-parents | `h²` |
| **Shortage** | Product / Warehouse | `Supplier→Component→Product` + warehouse context | `h³` |
| **Impact** | Order / Customer | `…→Product→rev_ORDERED→Order` | `h³` |

**A single global depth is wrong for all three simultaneously.** That is the modelling error the prior fixes, for zero parameters.

### Move 2 — relation-scoped tier decay

```
BOM relations (SUPPLIES, USED_IN, ORDERED, PLACED_BY, + reverses)  →  full depth
SUB_SUPPLIES_t2_*                                                   →  μ init at α¹ ≈ 0.30
SUB_SUPPLIES_t3_*                                                   →  μ init at α² ≈ 0.09
```

> **Tier ≠ layer — the mistake that would cut the wrong axis.** *"Tier-1 and tier-2 only"* is a claim about **supplier-to-supplier upstream distance**. `L` counts hops in the heterogeneous graph, and the BOM path `Supplier→Component→Product→Order→Customer` is 4 hops containing **zero tier traversal**. Capping `L=2` wouldn't limit upstream reach — it would stop a supplier from ever seeing the orders its components end up in. Cap tier on the `SUB_SUPPLIES` axis; leave `L` to the blanket.

## 4.3 The prior, numerically

Prior weights decay away from the blanket depth:

```
                exp( −|k − b_head| / τ )
p_head[k]  =  ─────────────────────────────                k ∈ {1,2,3,4}
               Σ_j exp( −|j − b_head| / τ )
```

With `τ = 0.8`:

| Head | `b` | `p[1]` | `p[2]` | `p[3]` | `p[4]` |
|---|---|---|---|---|---|
| Delay | 2 | 0.173 | **0.604** | 0.173 | 0.050 |
| Shortage | 3 | 0.050 | 0.173 | **0.604** | 0.173 |
| Impact | 3 | 0.050 | 0.173 | **0.604** | 0.173 |

Implemented as the gate's **output bias**:

```
b_out[head]  =  log( p_head )

delay     →  [ −1.75, −0.50, −1.75, −3.00 ]
shortage  →  [ −3.00, −1.75, −0.50, −1.75 ]
```

Initialise the gate's output *weights* near zero, and the untrained network reproduces the prior **exactly**. Training can only move away from it deliberately.

**`τ` controls prior width.** `τ = 0.4` gives `[0.070, 0.854, 0.070, 0.006]` — sharp, nearly a point prior. `τ = 2.0` is nearly uniform, i.e. no prior at all. `τ = 0.8` spans three layers with ~95% of the mass, tolerating an off-by-one error in either direction.

## 4.4 The gate — the minimal learned deviation

**Analogy — a camera zoom.** `h¹` is a macro shot (just me and my neighbours); `h⁴` is a wide shot (nearly the whole graph). Vanilla HGT forces every node to use the wide shot. The prior says which zoom each *task* wants. The gate lets each *node* nudge that.

**Architecture** — one per head:

```
u_v        =  [ h¹_v ‖ h²_v ‖ h³_v ‖ h⁴_v ]                 ∈ ℝ²⁵⁶   (optional: concat mode)
              or simply  h⁰_v                                ∈ ℝ⁶⁴    (node mode, default)

g_v        =  ReLU( W₁ · u_v + b₁ )                          W₁ ∈ ℝ^{32 × 64}
ℓ_v        =  W₂ · g_v + b_out[head]                         W₂ ∈ ℝ^{4 × 32}
w_v        =  softmax( ℓ_v )                                 ∈ Δ³   (4 values, sum 1)

z^head_v   =  Σ_{k=1..4}  w_v[k] · h^k_v                     ∈ ℝ⁶⁴
```

**Parameters per gate:** `64·32 + 32 + 32·4 + 4 = 2,212`. Three gates = **6,636**.

**FLOPs:** `2 · (64·32 + 32·4) · 5,000 · 3 heads` = **0.065 GFLOP**.

### Why a gate and not a full Transformer

The specification asks for *"4 values summing to 1."* **A convex combination with learned weights is a gate.** Multi-head QKV over four tokens plus a feed-forward network does far more than that output requires — and once the prior supplies the starting point, the gate only learns a *correction*, which is a much smaller function.

**And per-head instantiation triples whatever you choose:**

| Version | ×1 params | ×1 GFLOP | ×3 params | ×3 GFLOP |
|---|---|---|---|---|
| 2-layer Transformer, ff=4 | 99,968 | 3.93 | **299,904** | **11.80** |
| 1-layer, ff=4 | 49,984 | 1.97 | 149,952 | 5.90 |
| 1-layer, attention only | 16,960 | 0.66 | 50,880 | 1.97 |
| **Gate** ★ | **2,212** | **0.02** | **6,636** | **0.07** |

Three full Transformers would cost **11.8 GFLOP — nine times the entire HGT encoder** — to fuse four vectors. The gate does the same job for 0.6% of that.

**A note on why T1 looks cheap but isn't.** HGT's cost is dominated by *edges* (sparse). A depth Transformer runs dense projections on **every node × 4 tokens × 2 layers**. Sparse structure is what makes GNNs cheap; a depth Transformer discards it. **Judge it on FLOPs, never parameters.**

### The KL anchor

```
L_total  =  L_task  +  λ · Σ_heads  E_v [ KL( w_v ‖ p_head ) ]           λ = 0.01
```

The data must actively earn any departure from the prior.

> **Ordering rule — non-negotiable.** Biasing the gate toward a structural prior and then citing its weights as evidence *about that structure* is circular. **Depth claims are settled by the L-sweep (§9.3), never by reading the gate.** Get this order wrong and every downstream result is invalid.

## 4.5 Worked numerical example

**Shortage head**, prior `p = [0.050, 0.173, 0.604, 0.173]`, bias `b_out = [−3.00, −1.75, −0.50, −1.75]`.

**Supplier A** — one component, one product, one customer. The gate's learned contribution: `W₂·g_A = [+1.10, +0.85, −0.20, −0.60]`.

```
ℓ_A  =  [−3.00+1.10, −1.75+0.85, −0.50−0.20, −1.75−0.60]  =  [−1.90, −0.90, −0.70, −2.35]
exp  =  [ 0.150,  0.407,  0.497,  0.095 ]        Σ = 1.149
w_A  =  [ 0.130,  0.354,  0.432,  0.083 ]
```

*Deviation from prior:* `+0.080, +0.181, −0.172, −0.090`. **This node pulled shallower** than the prior predicted — consistent with a leaf supplier whose deep context is mostly noise.

**Supplier B** — twelve products, three factories. `W₂·g_B = [−0.90, −0.30, +0.25, +1.05]`.

```
ℓ_B  =  [−3.90, −2.05, −0.25, −0.70]
w_B  =  [ 0.017,  0.107,  0.649,  0.227 ]
```

*Deviation:* `−0.033, −0.066, +0.045, +0.054`. **This node pulled deeper.**

**That difference is the research finding.** Aggregate it across the evaluation set:

> *"The learned depth tracked the structural prediction for 94% of suppliers, deviating upward specifically for multi-division hubs and frontier-flagged nodes."*

Compare with what each alternative produces alone — *"mean depth was 1.8"* (a measurement with no reference point) or *"we chose depth from graph geometry"* (an assertion with no test). **The residual is a tested structural hypothesis with a characterised failure mode**, which is the strongest of the three.

## 4.6 Why L = 4 with a deepest prior of h³

**The headroom argument.** If the deepest prior is `h³` and the encoder stops at 3, the gate **sits on the ceiling**: it can learn *"look shallower"* but never *"look deeper."* The residual would be **censored at the top** — and upward deviation (hubs, frontier nodes) is exactly the interesting direction.

```
L  =  (deepest prior)  +  1  =  3 + 1  =  4          capped at 4
```

**Cost: +174,612 params (+32% over L=3), +0.33 GFLOP.**

**The honest tension:** at L=4 the receptive field is ~100% saturated (§3A.7), so `h⁴` is heavily over-smoothed. You are paying for a layer you expect the gate to mostly ignore.

**Why it's still correct:** the gate assigning `h⁴` near-zero weight almost everywhere **is a result**, and one you cannot obtain if you never compute `h⁴`. It's an empirical confirmation of the depth prior at a cost of 32% of the encoder. **If your budget is tight, L=3 is a defensible fallback — you simply lose the ability to detect upward deviation, and must report the residual as one-sided.**

## 4.7 Blend levels — the dial

Prior strength (`τ`, `λ`) and gate capacity move **together**. A strong prior means less to learn, so less capacity is needed. Mismatching them is the failure mode.

| Level | Configuration | Blend params | Blend GFLOP |
|---|---|---|---|
| **L0** | Fixed readout, no gate | 0 | 0 |
| **L1** | Gate ×3, `τ=0.4`, `λ=0.1` — prior-locked | 6,636 | 0.065 |
| **L2** ★ | Gate ×3, `τ=0.8`, `λ=0.01` — prior-anchored | 6,636 | 0.065 |
| **L2+** | Concat-gate ×3 (reads `h¹…h⁴`), `λ=0.01` | 25,068 | 0.250 |
| **L3** | Attention-only Transformer ×3, `λ=0` | 50,880 | 1.966 |
| **L4** | Full 2-layer Transformer ×3, no prior | 299,904 | 11.796 |

**This is the bias–variance tradeoff, and label count picks the optimum.**

```
More prior  →  more bias, less variance   →  better when labels are scarce
More gate   →  less bias, more variance   →  better when labels are abundant
```

**The measurement caveat that decides it:**

| Positives | 95% CI on AUC | Smallest detectable gap |
|---|---|---|
| 200 | ±0.037 | ~0.037 |
| 500 | ±0.024 | ~0.024 |
| 1,000 | ±0.017 | ~0.017 |

**At ~500 positives, L1 through L3 will most likely be statistically indistinguishable.** When you cannot measure the accuracy difference, choose on cost — and L2 costs 4% of inference while L4 costs 900%. That is not a compromise; it is the correct decision under the evidence you will actually have.

**Build it as configuration, not architecture:**

```python
DepthGate(
    prior      = blanket_prior[head],   # [w1..w4]
    tau        = 0.8,                   # prior width      ← knob A
    kl_lambda  = 0.01,                  # prior strength   ← knob A
    hidden     = 32,                    # gate capacity    ← knob B
    input_mode = "node",                # "concat" for L2+
)
```

L0 → L2+ are reachable by changing four values.

## 4.8 Pros and cons

**Pros:** near-free; fixes a genuine modelling error; defensible in one sentence; no methodological failure modes; smaller explanation subgraphs; degrades gracefully to L0 behaviour if it learns nothing; produces the residual finding.

**Cons:** the prior is a heuristic until the L-sweep validates it; gate weights measure *usage*, not sufficiency; three gates require a precise interface with T2 (§5.6); `λ` and `τ` are new hyperparameters.

---

# PART 5 — TRANSFORMER 2

## 5.1 The problem — the one gap that cannot be closed by scale

Two suppliers fail in the same week. **No edge chain between them** — different components, different products, different customers. Both quietly buy from the same tier-2 source you never recorded.

**Message passing cannot help at any depth.** It walks edges. No path, no message. A 4-layer encoder and a 40-layer encoder are equally blind. This is a **structural** limit, not a capacity limit — the one gap you cannot close by making HGT bigger.

## 5.2 The mechanism

```
Q = W_Q · z_s        K = W_K · z_s        V = W_V · z_s              z_s ∈ ℝ^{N_s × 64}

A = softmax( Q Kᵀ / √(d/h) )                    ← NO adjacency mask

out = A · V   →   W_O   →   FFN(64 → 256 → 64)  with residual + LayerNorm
```

**That missing mask is the entire idea.**

| | Question it asks |
|---|---|
| Message passing | *"Who are you connected to?"* |
| Global attention | *"Who do you resemble?"* |

**Analogy.** HGT knows people through the **org chart** — who reports to whom. T2 notices two people in unrelated departments always take the **same days off**. No line between them on the chart, but you infer a shared cause. Geographically: message passing follows **roads**; T2 looks at a **satellite photo** and sees two towns with identical architecture — probably built by the same company — with no road between them.

## 5.3 The same-type constraint — mandatory, not optional

T2 attends **only Supplier ↔ Supplier**. Never Supplier ↔ Product.

**Why mandatory:** HGT spends 76% of its parameters keeping each node type's embedding space *distinct*. Unconstrained global attention recombines every type into one shared space — **partially undoing the work that makes HGT worth its cost.** The constraint also matches the goal: the coupling you care about is supplier-to-supplier. Supplier-to-product coupling is already fully described by the edges you have.

## 5.4 Bounding the candidate pool

Dense attention is `O(N_s²·d)`:

| Suppliers | Dense | Bounded (k=64) | Speedup |
|---|---|---|---|
| 200 | 0.030 GFLOP | 0.023 | 1.3× |
| **600** | **0.151** | **0.069** | **2.2×** |
| 2,000 | 1.221 | 0.229 | 5.3× |
| 5,000 | 6.892 | 0.573 | 12.0× |
| 20,000 | 104.366 | 2.294 | **45.5×** |

### How to bound — and one way NOT to

> **Correction worth flagging.** An earlier draft suggested bounding by *"co-suppliers within the same `component_type`."* **That is wrong.** Co-suppliers of the same component are already 2-hop reachable (the co-parent term), so restricting to them would confine T2 to couplings HGT can already see — defeating its entire purpose. Worse, the canonical example (a German connector supplier and a Malaysian casings supplier coupled through a shared polymer plant) involves **different component types**, and would be excluded.

**The correct bounding uses no graph structure at all:**

| Rule | Why |
|---|---|
| **Top-k by embedding cosine**, `k = 64` | The pre-filter criterion is the *same* criterion attention would use to score the pair — so nothing genuinely coupled is excluded |
| **Always include frontier nodes** | They are the reason T2 exists; never let them drop out of a candidate set |
| **Threshold before persisting** | Keeps `hidden_dependency_links` reviewable rather than drowning reviewers in weak pairs |

Use approximate nearest-neighbour (FAISS / HNSW) and the pre-filter is `O(N log N · d)`, not quadratic.

## 5.5 Frontier nodes — the second job

Nodes with `is_frontier = true` (no recorded upstream) enter the pool **on equal footing**.

In message passing a frontier node is an **orphan** — no in-edges means no messages to aggregate, so HGT has literally nothing to say about it. In embedding attention it is a full participant, because similarity requires no edges. **Frontier nodes are precisely the ones T2 exists to serve.**

## 5.6 The output contract — an unresolved interface

**The problem, stated plainly:** T2 attends Supplier↔Supplier and therefore produces refined **supplier** embeddings only. It does **not** produce updated Product, Warehouse, Order, or Customer embeddings. So how does its output reach the shortage head (Product/Warehouse) or the impact head (Order/Customer)?

**It doesn't.** Three options:

| Option | Mechanism | Cost | Verdict |
|---|---|---|---|
| **1 — Scope to delay only** ★ | Wire T2 exclusively to the supplier-level delay head | 0 | **Take this now.** Honest about what T2 does; defers the interface until T2 has proven it finds anything |
| **2 — One propagation layer after T2** | `HGT ×4 → gate → T2 → HGT ×1 → heads` | +174,612 params | Principled. Does **not** violate §6.3 — that clause forbids feeding *back into* the encoder; a new forward layer extends the pipeline without creating a cycle |
| 3 — Same-type T2 per target type | Product↔Product, Order↔Order modules | ×N cost | No motivating story; the hidden-coupling problem is specifically supplier-to-supplier |

**HADES v1 takes Option 1.** Revisit only if T2 demonstrably finds real couplings *and* you need them downstream.

## 5.7 Worked example

Suppliers **S1** (Germany, connectors) and **S2** (Malaysia, casings). No shared component, product, or customer. Graph distance: effectively infinite.

With multi-window temporal features (§2.4), both embeddings encode *"90-day on-time rate falling, trend slope negative, lateness variance rising."*

```
cosine(z_S1, z_S2) = 0.71     → S2 enters S1's top-64 pool
attention(S1 → S2) = 0.31     → unusually high for unrelated nodes
```

Persisted to `hidden_dependency_links` with its weight and `model_version`. A reviewer investigates and finds both source a specialty polymer from one plant.

> **Note this example only works because of §2.4.** With a single scalar `reliability_history`, "both currently degrading" is **not representable** — you cannot distinguish a supplier sliding downward from one that has always been mediocre. An earlier draft of this example claimed temporal correlation the feature set could not support. The multi-window features are what make it honest.

## 5.8 Validation — similarity is not a dependency

A high attention weight means *"similar within this candidate pool."* It does **not** mean *"shares an upstream supplier."* Three validation routes, in increasing strength:

1. **Held-out edges** — mask known `SUB_SUPPLIES` edges; does T2 rank those pairs highly?
2. **Future co-disruption** — do flagged pairs co-fail more than chance in the holdout period?
3. **F-09 as validator** — you already plan a link-prediction head. Score T2's discoveries against it. **Two components validating each other for free.**

**Persist links as investigation leads, never as facts.** Surface with attention weight and `model_version` attached, so a reviewer sees the strength of the claim.

## 5.9 Parameters and FLOPs

```
QKVO projections   4 · (d² + d)      = 4 · 4,160     = 16,640
FFN (64→256→64)    2 · 4 · d² + 4d   = 32,768 + 256  = 33,024
LayerNorms         2 · 2d            =                    256
                                                     ─────────
                                        Transformer 2   49,984
```

```
Projections   2 · (4·N_s·d² + 8·N_s·d²)   = 59.0 MFLOP
Attention     2 · 2 · N_s · k · d          =  9.8 MFLOP    (bounded, k=64)
                                            ───────────
                                              68.8 MFLOP  =  0.069 GFLOP
```

## 5.10 Pros and cons

**Pros:** the only mechanism that closes Gap 1; serves frontier nodes; cheap at your scale (+5% FLOPs); produces auditable findings; closes the BOM reach gap without a 5th hop; fails gracefully — if no correlation exists, the pipeline still works as HGT + depth selection.

**Cons:** similarity ≠ causation (ML-12); rests on the *untested* assumption that supplier-to-supplier correlation exists in your data at all; quadratic beyond ~2,000 suppliers without bounding; attention scores are directional and pool-dependent; supplier-only output cannot reach non-supplier heads (§5.6); can reinforce spurious correlation and historical bias; counterintuitive findings can erode user trust even when correct.

---

# PART 6 — HEADS, RISK INTELLIGENCE, CLAIM B

## 6.1 Prediction heads

```
delay_v     =  σ( w_d · MLP(z_delay_v)   + b_d )         Supplier, Shipment
shortage_v  =  σ( w_s · MLP(z_short_v)   + b_s )         Product × Warehouse
impact_v    =  σ( w_i · MLP(z_impact_v)  + b_i )         all entity types
```

Each head: `(d² + d) + (d + 1) = 4,225` params. Three heads = **12,675**.

## 6.2 Loss

```
L_task  =  Σ_heads  w_head · FocalLoss( pred, label ; γ = 2, α = class_weight )
L_total =  L_task  +  λ · Σ_heads E_v[ KL( w_v ‖ p_head ) ]
```

Focal loss because disruptions are a minority class. `α` from inverse class frequency; `γ = 2` is the standard focusing parameter.

## 6.3 Risk Intelligence — deterministic, zero parameters

| Step | Mechanism |
|---|---|
| Confidence | Softmax margin, or MC-dropout variance over ~10 stochastic passes, normalised to [0,1] |
| Business aggregation | `impact_score` via `gnn_native` (direct) or `weighted_formula` |
| Threshold evaluation | Compare against configurable thresholds |
| Categorisation | `low` / `medium` / `high` / `critical` |
| Method tagging | Every row records `scoring_method` |

```
Overall Risk = 0.30·Supplier + 0.25·Shipment + 0.20·Inventory + 0.15·Demand + 0.10·Financial
```

These five are **not model heads** — they are inputs to a deterministic aggregation downstream of the GNN. Weights are illustrative starting points, tuned against validation data.

## 6.4 Claim B — dyadic risk

**The insight:** *"Supplier X: 80% failure probability"* conflates X's **own fragility** with **your exposure** to it. A fragile supplier for whom you are a top-priority customer will serve you first. A moderate supplier who is sole-source for your strategic line is worse than the number suggests.

**Why it isn't in the model:** the **node** is the structural encoder's unit of prediction — one node, one embedding, one score, true of SHARE (§3) exactly as it was of HGT (§3A). There is no architectural slot for a *(supplier, buyer)* pair. And reweighting by order-volume share and contract priority is deterministic arithmetic; burying it in a network would make an auditable business rule opaque for zero representational benefit.

```
dyadic_risk  =  global_risk  ×  f( order_volume_share,
                                   contract_priority_weight,
                                   fulfilment_preference_weight )
```

### The double-counting problem — test before you trust

`Customer` is a first-class node whose **only** feature is `priority_tier`, and `PLACED_BY` connects orders to customers. **So the encoder may already be learning part of what Claim B reweights by** — applying the same signal twice.

**The test:**

```
1. Train encoder WITH customer priority_tier    → Claim B reordering rate R₁
2. Train encoder WITHOUT it                     → Claim B reordering rate R₂

R₁ ≈ R₂  →  independent, no double-count
R₁ ≪ R₂  →  the encoder already learned it
```

**Then reselect signals:**

| Signal | Visible to the encoder? | Verdict |
|---|---|---|
| `order_volume_share` | **Yes** — derivable from `ORDERED` edges | Likely double-counts |
| `contract_priority_weight` | **Yes** — `Customer.priority_tier` is the only Customer feature | Likely double-counts |
| `fulfilment_preference_weight` | **No** — historical rationing behaviour is nowhere in the schema | **Lead with this** |

**Lead Claim B on fulfilment preference.** It is the one signal the GNN structurally cannot see, and it makes the contribution cleaner: *"we add information the graph doesn't contain"* rather than *"we reweight by information the graph already has."*

**Present it as an exposure score, never a calibrated failure probability.**

---

# PART 7 — COMPLETE BUDGET

## 7.1 Parameters

| Component | Params | Share |
|---|---|---|
| Input projections (8 types) | 5,568 | 0.7% |
| **HGT encoder** (L=4 × 174,612) | **698,448** | **90.3%** |
| Depth gates (3 × 2,212) | 6,636 | 0.9% |
| Transformer 2 | 49,984 | 6.5% |
| Prediction heads (3) | 12,675 | 1.6% |
| Risk Intelligence / Claim B | 0 | 0% |
| **TOTAL** | **773,311** | 100% |

**The encoder is 90% of the model.** Every argument about T1's size, T2's cost, and gate capacity is an argument about the remaining 10%. If you need a large saving, the levers are `d` and `L` — nothing else moves the needle.

> **Production note.** This table is the design-time target budget (T2 and the
> depth gate remain unbuilt — §10.3) and its HGT row is the original theoretical
> derivation (§3A.5), not a live measurement. The encoder actually deployed in
> production is SHARE (§3), at 752,211 real measured params (matched-`d`,
> `rgcn_attn` in code) — swap that in for the HGT row above if estimating today's
> real total; FLOPs for SHARE haven't been derived on paper the way §3A.6 does
> for HGT, only wall-clock training time has been measured
> (`reports/rgcn_matched_pilot.md`).

## 7.2 FLOPs (full-graph forward pass)

| Component | GFLOP | Share |
|---|---|---|
| HGT encoder (L=4) | 1.311 | 90.7% |
| Depth gates (3, all 5,000 nodes) | 0.065 | 4.5% |
| Transformer 2 (bounded, 600 suppliers) | 0.069 | 4.8% |
| Heads + Risk Intelligence | ~0.001 | 0.1% |
| **TOTAL** | **1.445** | 100% |

**Well inside NFR-01's 2-second budget** — tens of milliseconds on CPU. And note the single-entity request path touches only a subgraph, so per-request latency is far lower still.

## 7.3 What each fix cost

| Fix | Params | FLOPs |
|---|---|---|
| Reverse relations (§2.2) | +81,960 | +0.655 GFLOP |
| Multi-window temporal features (§2.4) | +640 | negligible |
| Tier/confidence relation split (§2.3) | +16,384 (when built) | small |
| L=4 headroom over L=3 (§4.6) | +174,612 | +0.328 GFLOP |
| Gate instead of full T1 (§4.4) | **−293,268** | **−11.73 GFLOP** |
| Bounded instead of dense T2 (§5.4) | 0 | **−0.082 GFLOP** |

**Every correctness fix combined costs less than the two efficiency choices save.**

## 7.4 Why `d = 64` (HGT) — and why SHARE's production `d = 128` doesn't reopen this argument

For HGT specifically, `d = 128` quadruples the dominant term — L=4 would go from
698K to ~2.8M parameters against a few hundred positive labels, with no
documented justification anywhere in the project. That reasoning is unchanged
and still correct **for HGT's own fully-dedicated-per-relation parameterisation**,
where every unit of `d` is spent `T+R` times over (§3A.5's 76/24 split).

It does not transfer to SHARE. SHARE's basis-shared transform and two-vector
attention scorer cost far less per unit of `d` — its own `d=128` matched-parameter
arm (752,211 params, chosen specifically to land near HGT's `d=64` anchor, §3.4)
lands at roughly the SAME total budget HGT's `d=64` does, not 4× it. The
parameters-per-positive-label risk (ML-09) that made raising `d` unaffordable for
HGT was always a property of HGT's parameterisation, not a universal ceiling on
`d` itself — SHARE's own ablation (§3.5) is the direct empirical demonstration
that a wider `d` is affordable once the per-relation cost structure changes.

---

# PART 8 — TRAINING PROTOCOL

## 8.1 The pipeline

```
1. For each t₀ in the snapshot schedule:
     build HeteroData with features ≤ t₀    (§2.5)
     attach labels from (t₀, t₀+H]
2. Split by t₀: train / val / test — NEVER randomly
3. Train with focal loss + KL anchor        (§6.2)
4. Early stop on validation AUC
5. Persist to model_registry + model_evaluation_runs
```

## 8.2 Hyperparameters

| Setting | Value | Note |
|---|---|---|
| Optimizer | AdamW | |
| Learning rate | 1e-3, cosine decay | |
| Weight decay | 1e-4 | |
| Dropout | 0.2 (structural encoder layers — SHARE in production, HGT originally), 0.1 (gate, T2) | |
| Edge dropout | 0.1 | Regularises against sparse-relation overfit |
| Batch | Full-graph | 5,000 nodes fits comfortably |
| Early stopping | Patience 20 on val AUC | |
| Gradient clipping | 1.0 | Attention stacks can spike |
| Focal `γ`, `α` | 2, inverse class frequency | |
| KL `λ` | 0.01 | §4.3 |
| Prior `τ` | 0.8 | §4.3 |

## 8.3 Training runs required

| Purpose | Runs |
|---|---|
| L-sweep (L = 1,2,3,4) | 4 |
| Architecture ablation (GraphSAGE, GAT, HGT, RGCN at fixed `d`) | 4 |
| **Matched-parameter arm** (SAGE `d=86`, GAT `d=116`, HGT `d=64`, RGCN `d`+`num_bases` per below) | 4 |
| Gate on/off | 1 |
| T2 on/off | 1 |
| Claim B double-counting test | 1 |
| **Total** | **15** |

**The matched-parameter arm is not optional.** GAT has *fewer* parameters than GraphSAGE at equal `d` (`d²` vs `2d²` per relation), so a Stage-2 win is ambiguous between *"attention helped"* and *"less capacity regularised better on scarce labels."* Without the matched arm you cannot distinguish them, and your central ablation claim is confounded.

**RGCN (`ml/models/rgcn_encoder.py`) — a fourth arm, added post-v3.** RGCN
(Schlichtkrull et al. 2018 basis decomposition, hand-written over
`edge_index_dict` since `torch_geometric.nn.RGCNConv`'s public API expects a
homogeneous graph, not this project's `HeteroData` convention) gives every
relation its own transform like HGT, but as a linear combination of a small
shared pool of `num_bases` basis matrices rather than a fully dedicated
`d x d` matrix per relation — its own hyperparameter, alongside `hidden`
(shared with the other three arms) and GAT's `heads`: **`num_bases`**
(default 8; the matched-parameter arm's value is found by grid search, same
method as SAGE/GAT's matched-`d`). Against this round's live-recomputed
feature dims, HGT's real fixed-`d=64` encoder anchors at 701,088 parameters;
grid-searching `num_bases in {4, 8, 12, 16}` at `hidden=64` alone landed
nowhere close (self-loop + input-projection dominate RGCN's count at that
width, unlike HGT's per-relation attention), so the search widened `hidden`
alongside `num_bases` — `hidden=138, num_bases=4` lands at 699,602 params,
0.21% off HGT's anchor. See `ml/run_step_v4_4arch.py` for the full 4-arch,
5-seed run matrix this round added.

**Where this ended up: SHARE, not RGCN or HGT.** Two more RGCN-family hybrids
were added and ablated after the round documented above — RGCN+Attention
(`rgcn_attn` in code, this is **SHARE**, §3) and two further variants
(`rgcn_relemb`, `rgcn_battn`). Across all six architectures at matched
parameters, SHARE won delay and shortage consistently against HGT and tied on
impact, beating every other architecture tried on more than one task
(`reports/rgcn_types.md`'s closing comparison table). **SHARE is the production
architecture selected from this whole ablation line** — see the production note
at the top of this document and §3.

## 8.4 Governance — write it, don't type it

Every run writes one `model_registry` row: `training_dataset`, `training_timestamp`, `experiment_id`, `git_commit`, `hyperparameters`, `parameter_count`, `status`. **Written by the pipeline, never entered by hand**, so it cannot drift from what was actually run.

---

# PART 9 — EVALUATION PROTOCOL

## 9.1 Metrics

| Metric | Applies to | Target |
|---|---|---|
| AUC-ROC | delay, shortage | ≥ 0.80 held-out |
| Precision / Recall | at the alert operating threshold | Reported per class |
| **Calibration** | reliability diagram | *"78%"* must mean 78% — the chatbot states it at face value |
| Inference time | per architecture | Cost input to the ablation |
| Explanation stability | GNNExplainer across seeds | Subgraph overlap; addresses ML-03 |

## 9.2 Confidence intervals — mandatory on every row

| Positives | 95% CI on AUC | Smallest detectable gap |
|---|---|---|
| 200 | ±0.037 | ~0.037 |
| 500 | ±0.024 | ~0.024 |
| 1,000 | ±0.017 | ~0.017 |

**If GraphSAGE scores 0.79 and HGT scores 0.81 on 500 positives, you have not measured a difference.** Report DeLong or bootstrap CIs on every ablation row. *"The architectures were not distinguishable at this label volume"* is a legitimate finding, and it is the one most projects quietly avoid stating.

## 9.3 Per-component validation

| Component | How it earns inclusion |
|---|---|
| Reverse relations | Measured k-hop reach table per target type |
| Depth prior | **L-sweep** — nested comparison, `ΔAUC` vs CI |
| Depth gate | Beats fixed-readout baseline by more than the CI |
| T2 | Held-out `SUB_SUPPLIES` edges, future co-disruption rate, or F-09 agreement |
| Claim B | Double-counting test, then reviewer assessment of reordering plausibility |
| Explanations | GNNExplainer (perturbation-based), **never attention weights** |

## 9.4 Over-smoothing test

Compute mean pairwise cosine similarity between node embeddings at each layer. A sharp rise marks your empirical depth ceiling — this replaces the theoretical reach table in §3A.7 with a measurement.

---

# PART 10 — KNOWN ISSUES

## 10.1 Fixed in this document

| Issue | Fix | §|
|---|---|---|
| Label leakage | Forecasting contract, exclusion list, multi-snapshot training | 2.5 |
| Graph directionality | Reverse relations, `R` 10→20 | 2.2 |
| Tier & confidence unusable via `μ` | Relation splitting by tier × confidence band | 2.3 |
| T2 output contract | Scoped to delay head (Option 1) | 5.6 |
| Temporal claim unsupported | Multi-window features | 2.4 |
| "Markov blanket" overstated | Renamed *structural depth prior* | 4.2 |
| Attention ≠ explanation | GNNExplainer only; L-sweep for depth | 9.3 |
| Wrong bounding for T2 | Cosine top-k, not component-type | 5.4 |
| Ablation confound | Matched-parameter arm | 8.3 |
| Claim B double-counting | Ablation test, lead on fulfilment preference | 6.4 |

## 10.2 No fix available

| Issue | Why unfixable | What to do |
|---|---|---|
| **Statistical power** | At ~500 positives, most comparisons are below the detection threshold | Report CIs; pre-register which comparisons you expect to be decisive; state plainly when a difference is not significant |
| **Exogenous shocks** (ML-14) | No architecture can produce a signal the graph never recorded | External risk feeds only — a data problem, not an architecture problem |
| **Single focal firm** (ML-13) | Claim B's formulation assumes one buyer | Out of scope; documented |
| **Snapshot, not temporal** | Full temporal modelling is an architectural change | §2.4 buys most of the benefit; note the limitation |
| **Not a novel architecture** | HGT (2020) + JK (2018) + global attention (2021–22); GraphGPS already templates local+global | Claim a *novel configuration for a specific structural problem*. That claim is bulletproof; a novelty claim is not. |

## 10.3 Status — this is a target, not Phase 1

`is_frontier`, `SUB_SUPPLIES`, the depth gate and Transformer 2 are all **"Not committed — Phase 2 (early April 2027) or later"** in the project's own phasing.

**HADES is the target configuration.** The Phase-1 deliverable is §11 steps 1–5 only.

---

# PART 11 — IMPLEMENTATION CHECKLIST

## Step 0 — Dataset *(blocks everything)*

- [ ] Secure a dataset with **real timestamps** on shipments, orders, inventory
- [ ] Public option: the benchmark in *Graph Neural Networks in Supply Chain Analytics* (arXiv 2411.08550)
- [ ] Count positive labels for delay and shortage. **This number determines your blend level and your entire evaluation design.**

## Step 1 — Leakage contract *(blocks everything downstream)*

- [ ] Choose `H` (14 days) and the `t₀` schedule (monthly)
- [ ] Implement as-of feature computation with a hard timestamp cutoff
- [ ] Apply the exclusion list (§2.5)
- [ ] Build multi-snapshot training data
- [ ] **Run the single-feature leakage test.** Any solo feature above AUC 0.9 leaks.

## Step 2 — Graph correctness

- [ ] Add reverse relations (`ToUndirected`)
- [ ] **Measure** the actual k-hop reach table per target node type
- [ ] Add multi-window temporal features (§2.4)
- [ ] Verify feature dims: Supplier 21, Shipment 10

## Step 3 — Baseline

- [ ] HGT encoder, `L=4`, retain `h¹…h⁴`
- [ ] **Fixed** per-task readout (blend level L0 — no gate yet)
- [ ] Three prediction heads, focal loss
- [ ] Time-based split
- [ ] AUC, precision/recall, calibration, **all with CIs**

## Step 4 — Validate the depth prior

- [ ] L-sweep at L = 1, 2, 3, 4
- [ ] Nested significance test on each `ΔAUC`
- [ ] Over-smoothing measurement per layer
- [ ] Set `τ` from the result: sharp peak → 0.4, flat curve → 0.8

## Step 5 — Architecture ablation

- [ ] GraphSAGE, GAT, HGT at fixed `d = 64`
- [ ] **Matched-parameter arm:** SAGE `d=86`, GAT `d=116`, HGT `d=64`
- [ ] Report both axes, with CIs
- [ ] Persist all runs to `model_evaluation_runs` + `model_registry`

*— Phase 1 ends here —*

## Step 6 — Depth gate

- [ ] Implement `DepthGate` with `prior`, `tau`, `kl_lambda`, `hidden`, `input_mode`
- [ ] Initialise output bias at `log(p_head)`, weights near zero
- [ ] Verify: untrained model reproduces L0 behaviour **exactly**
- [ ] Train; compare against step 3 baseline
- [ ] **Adopt only if it beats the baseline by more than the CI**
- [ ] Report the residual: blanket-predicted vs. learned depth

## Step 7 — Transformer 2

- [ ] Implement with same-type constraint (suppliers only)
- [ ] Cosine top-k pre-filter, `k = 64`, always including frontier nodes
- [ ] **Wire to the delay head only** (Option 1, §5.6)
- [ ] Persist `hidden_dependency_links` with weights and `model_version`
- [ ] Validate against held-out edges / future co-disruption / F-09
- [ ] **Adopt only if validation shows better-than-chance discovery**

## Step 8 — Claim B

- [ ] Run the double-counting test (§6.4)
- [ ] Reselect signals — lead on `fulfilment_preference_weight`
- [ ] Persist to `supplier_dyadic_risk`, never overwriting `risk_scores`
- [ ] Surface both figures side by side on the dashboard
- [ ] Validate reordering against real fulfilment outcomes

---

# APPENDIX A — FORMULAE AND CONSTANTS

## A.1 Every equation, collected

**Input projection**
```
h⁰_v = W_in[τ(v)] · x_v + b_in[τ(v)]
```

**SHARE layer (production, §3)**
```
W_r        = Σ_b a_r[b] · V_b                          (basis-shared transform, §3.3 Step 1)
msg(s→t)   = h_s · W_r

logit(s→t) = LeakyReLU( att_msg · msg(s→t) + att_dst · h_t )
α(s→t)     = softmax_{s' → t, ANY relation} logit(s'→t)   (§3.3 Step 2 — joint across relations)

h_t'       = self_loop[τ(t)] · h_t  +  Σ_{s→t} α(s→t) · msg(s→t)
```

**HGT layer** (per head `i` — §3A, original baseline, superseded in production)
```
K_i(s) = K_Lin_i[τ(s)] · h^{ℓ-1}_s
Q_i(t) = Q_Lin_i[τ(t)] · h^{ℓ-1}_t
M_i(s) = M_Lin_i[τ(s)] · h^{ℓ-1}_s

ATT_i(s,e,t) = ( K_i(s) · W_ATT_i[φ(e)] · Q_i(t)ᵀ ) · μ[⟨τ(s),φ(e),τ(t)⟩] / √(d/h)
α_i(s,e,t)   = softmax_{s ∈ N(t)} ATT_i(s,e,t)
MSG_i(s,e,t) = M_i(s) · W_MSG_i[φ(e)]

h̃_t   = ‖_i Σ_{s∈N(t)} α_i · MSG_i
h^ℓ_t = A_Lin[τ(t)] · σ(h̃_t) + h^{ℓ-1}_t
```

**Depth prior**
```
p_head[k] = exp(−|k − b_head| / τ) / Σ_j exp(−|j − b_head| / τ)
b_out[head] = log(p_head)
```

**Depth gate**
```
g_v      = ReLU(W₁ · u_v + b₁)
ℓ_v      = W₂ · g_v + b_out[head]
w_v      = softmax(ℓ_v)
z^head_v = Σ_k w_v[k] · h^k_v
```

**Transformer 2**
```
pool(s) = top-k by cos(z_s, z_s') ∪ {frontier nodes}
A       = softmax(Q Kᵀ / √(d/h))      over pool(s) only
out     = FFN(LN(A·V·W_O + z))
```

**Loss**
```
L_total = Σ_heads w_head · Focal(pred, label; γ=2, α) + λ · Σ_heads E_v[KL(w_v ‖ p_head)]
```

**Claim B**
```
dyadic_risk = global_risk × f(order_volume_share, contract_priority, fulfilment_preference)
```

## A.2 Constants

```
d = 64          h = 4           d/h = 16        L = 4
T = 8           R = 20          V = 22,155-66,973 (measured)   E = 122,594-388,242 (measured)
N_s = 800 (measured)
k = 64          τ = 0.8         λ = 0.01        H = 14 days
γ = 2 (focal)   lr = 1e-3       wd = 1e-4       dropout = 0.2 / 0.1
```

**V/E/N_s recomputed from the real v3 graph** (`reports/step5_result_v3.md`, Step 2) — this is
the update Part 1's own closing note (below) asks for: "the absolute values depend on ...
real graph size and real degree distribution, and should be recomputed the moment you have a
graph." They're no longer a pre-code estimate. Everything else in this budget (parameter counts,
FLOP ratios) still assumes `d=64` and the planning-time node/edge counts it was derived against;
recomputing the parameter/FLOP budget itself against the real per-snapshot graph size is not yet
done and is a fair next step, not claimed here.

## A.3 Budget at a glance

```
PARAMETERS (design-time target,       FLOPs (full-graph forward,
  HGT encoder — §3A, superseded)        HGT encoder — not derived for SHARE)
  input projections      5,568        HGT encoder        1.311 GFLOP
  HGT encoder (L=4)    698,448        depth gates        0.065
  depth gates (×3)       6,636        Transformer 2      0.069
  Transformer 2         49,984        heads              0.001
  heads (×3)            12,675      ───────────────────────────
─────────────────────────────        TOTAL              1.445 GFLOP
  TOTAL                773,311
```

**Production encoder swap:** the deployed encoder is SHARE (§3), 752,211 real
measured params at matched-`d` (`rgcn_attn` in code) in place of the 698,448-param
HGT row above — see §7.1's production note for the same caveat. SHARE's own FLOP
cost hasn't been derived on paper the way HGT's was; only wall-clock training time
is measured (`reports/rgcn_matched_pilot.md`).

## A.4 Citations

- **HGT** — Hu, Dong, Wang & Sun (2020), *Heterogeneous Graph Transformer* (§3A, original baseline)
- **RGCN / basis decomposition** — Schlichtkrull et al. (2018), *Modeling Relational Data with Graph Convolutional Networks* — SHARE's transform (§3.3) is built directly on this
- **Jumping Knowledge** — Xu et al. (2018)
- **GraphSAGE** — Hamilton, Ying & Leskovec (2017)
- **GAT** — Veličković et al. (2018)
- **Over-smoothing** — Li, Han & Wu (2018); Oono & Suzuki (2020)
- **APPNP / PPRGo** — Klicpera et al. (2019); Bojchevski et al. (2020)
- **GraphGPS** — Rampášek et al. (2022)
- **GraphTrans** — Wu et al. (2021)
- **Graph Information Bottleneck / GSAT** — Wu et al. (2020); Miao et al. (2022)
- **Context-specific independence** — Boutilier et al. (1996)
- **Markov blanket feature selection** — Koller & Sahami (1996); Aliferis et al.
- **Preferred customer status** — Schiele, Calvi & Gibbert (2012); Steinle & Schiele (2008)
- **Focal loss** — Lin et al. (2017)
- **DeLong test** — DeLong, DeLong & Clarke-Pearson (1988)

---

## Final note to the reader

Everything numerical in this document was originally **derived analytically from the specification**, because no code existed yet. The relative orderings are robust — the gate really is 45× smaller than a full depth Transformer, the encoder really is 90% of the model, reverse relations really do double the edge term. The **absolute** values depend on `d`, `L`, real graph size and real degree distribution, and should be recomputed the moment you have a graph.

**That graph now exists.** Part 1's `V`/`E`/`N_s` and Appendix A.2's constants block have been
recomputed against it (`reports/step5_result_v3.md`, Step 2) — this fulfills the instruction
above. `V`/`E` turned out to be an order of magnitude larger than the ~5,000/~40,000 planning
estimate (22,155–66,973 nodes / 122,594–388,242 edges, growing snapshot to snapshot as the
15-month simulated timeline accumulates orders and shipments) and `N_s` grew from an estimated
~600 to a measured 800. The relative-ordering claims in the paragraph above are unaffected by
this — they were never claims about the absolute node/edge counts.

The lineage claim to make, and to make precisely:

> *A GraphTrans-style sequential hybrid with a heterogeneous local stage (SHARE —
> basis-shared relational transform + joint attention, `rgcn_attn` in code;
> originally HGT, §3A, superseded after a four-round ablation,
> `reports/rgcn_types.md`), a structurally-primed depth gate between the stages,
> and a type-constrained, frontier-aware global stage — specialised for
> partially-observed heterogeneous networks with rare adverse events.*

That claim survives review. *"A new architecture"* would not.
