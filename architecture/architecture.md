# ChainPilot Layer 2 — The Architecture, Explained

*How HGT, the two Transformers, and the Markov scoping principle fit together — what each does, why each is needed, and what happens without them. Companion to `GNN Theory.md`, `Types Of GNNs and Hybrids.md`, and `Markov Scoping and Transformer 1.md`.*

---

## How to read this document

Layer 2 is not one model. It's four components, each solving exactly one problem that the others structurally cannot. This document explains each in isolation, then shows how they compose — and, importantly, which ones you can drop.

All figures assume `d = 64`, 4 attention heads, 8 node types, 10 Phase-1 meta-relations, a 5,000-node / 20,000-edge graph with ~600 suppliers. Derivations are in the appendix.

---

# PART 0 — THE ARCHITECTURE AT A GLANCE

## 0.1 The pipeline

```
        HeteroData snapshot (8 node types, 10 edge types)
                          ↓
        ┌─────────────────────────────────────────┐
        │  HGT ENCODER  (L layers, all retained)  │   ← type-aware local structure
        └─────────────────────────────────────────┘
                          ↓  h¹  h²  h³  h⁴
        ┌─────────────────────────────────────────┐
        │  DEPTH SELECTION  (blended)             │   ← how far should each
        │  Markov sets the prior                  │      prediction look?
        │  Transformer 1 learns the deviation     │
        └─────────────────────────────────────────┘
                          ↓  fused z_i
        ┌─────────────────────────────────────────┐
        │  TRANSFORMER 2  (same-type global attn) │   ← who is coupled without
        └─────────────────────────────────────────┘      a recorded edge?
                          ↓
              Prediction heads: delay / shortage / impact
                          ↓
        ┌─────────────────────────────────────────┐
        │  RISK INTELLIGENCE LAYER                │   ← whose risk is it?
        │  confidence, formula, Claim B (dyadic)  │      (no learned parameters)
        └─────────────────────────────────────────┘
```

## 0.2 What each component does

| Component | One-line job | Learned? |
|---|---|---|
| **HGT encoder** | Learn type-aware local structure by message passing over typed nodes and edges | Yes |
| **Markov readout** | Set the *prior* — how deep each prediction task should look, from graph geometry | **No — zero parameters** |
| **Transformer 1 / gate** | Learn the *deviation* from that prior, per node — and measure where the structure was wrong | Yes (size is your choice — see Part 6.5) |
| **Transformer 2** | Find suppliers that behave together despite having no recorded connection | Yes |
| **Risk Intelligence (Claim B)** | Reweight a supplier's global score by *your firm's* specific exposure | **No — deterministic** |

## 0.3 How they complement each other

The design principle: **each component names one specific thing the previous one structurally cannot do.** Nothing is decorative.

| Component | Blind spot it covers | Blind spot it still has |
|---|---|---|
| HGT | Typed local structure — the thing every simpler model gets wrong | One depth for all tasks; can't reach unconnected nodes |
| Markov readout | Per-task depth | Same depth for every node of a given task |
| Transformer 1 / gate | Per-node depth, over-smoothing | Still edge-bound — can't see past the graph |
| Transformer 2 | Couplings with no recorded path; frontier nodes | Produces a *global* score per supplier |
| Claim B | Supplier fragility ≠ your exposure | Single focal firm; needs contract data |

Read down the right column and each row's limitation is answered by the next. That chain is the architecture's actual argument.

**Four distinct information types:**

- HGT encodes **structure**
- Depth selection encodes **how much structure matters, per node**
- Transformer 2 encodes **unstructured statistical correlation**
- Claim B encodes **business relationship context**

These are genuinely different signals, not four ways of re-deriving the same thing.

**Graceful degradation.** If Transformer 2 finds nothing (tier-2 correlation genuinely doesn't exist in your data), the pipeline still works as HGT + depth selection. No component's failure is fatal to the others — that's a property of the layered design, not an accident.

---

# PART 1 — HGT

## 1.1 The core idea in one picture

Your graph has 8 kinds of node and 10 kinds of edge. Think of it as a meeting where people speak different professional languages — Suppliers speak logistics, Products speak engineering, Orders speak sales.

| Architecture | What it does in that room |
|---|---|
| **GraphSAGE** | Everyone talks at once, you average the noise |
| **GAT** | You listen *harder* to some people — but they're still speaking languages you don't know |
| **HGT** | Every profession gets a **translator**, and every *kind of conversation* gets its own **etiquette** |

Everything below is how those two ideas — translators and etiquette — are implemented.

## 1.2 Meta-relations: the key concept

HGT doesn't think in terms of *edges*. It thinks in **meta-relations** — a triple:

```
⟨ source node type , edge type , target node type ⟩
```

From your graph:

- `⟨Supplier, SUPPLIES, Component⟩`
- `⟨Component, USED_IN, Product⟩`
- `⟨Shipment, SHIPS_FROM, Supplier⟩`
- `⟨Shipment, SHIPS_FROM, Factory⟩` ← **same edge type, different meta-relation**

That last pair matters. `SHIPS_FROM` means something different pointing at a Supplier than at a Factory, and HGT gives each its own parameters. GAT would collapse them.

## 1.3 One HGT layer — three questions

### Q1: "Who should I listen to?" — attention

Say **Product P** is updating itself. Its neighbours are Components and Orders.

**Step 1 — everyone speaks through their own translator.**

```
P's question    →  Q_Linear[Product]   · h(P)
Component's ID  →  K_Linear[Component] · h(C)
Order's ID      →  K_Linear[Order]     · h(O)
```

Three *different weight matrices*. A Component and an Order cannot be confused, because they were never projected by the same function. **This is the translator.**

**Step 2 — the relationship type shapes the comparison.**

```
score(C→P) = K(C) · W_ATT[USED_IN] · Q(P)ᵀ
score(O→P) = K(O) · W_ATT[ORDERED] · Q(P)ᵀ
```

`W_ATT` sits *between* key and query. It's the etiquette. **This is what GAT cannot do** — GAT has one matrix for every edge in the graph.

**Step 3 — a learned dial per relationship type.**

```
score = score × μ[⟨Component, USED_IN, Product⟩] / √d
```

`μ` is one learnable number per meta-relation: *"conversations of this type generally matter this much."* Cheap — and this is exactly where you impose the `SUB_SUPPLIES` decay.

**Step 4 — softmax across all neighbours.** All types compete for one attention budget. Type-awareness lives in the projections, not in separate competitions.

All of this runs in `h` parallel heads, each in a `d/h`-dimensional subspace — so one head can specialise in timing while another specialises in capacity.

### Q2: "What do they actually say?" — message

```
Message(C→P) = M_Linear[Component] · h(C) · W_MSG[USED_IN]
```

Same two-part structure: the **source type's** matrix extracts what that kind of node has to say; the **edge type's** matrix reshapes it for that relationship. A Supplier tells a Component about *reliability*; a Shipment tells a Warehouse about *timing*.

### Q3: "How do I update?" — aggregate

```
h̃(P) = Σ  attention(s→P) × Message(s→P)

h_new(P) = A_Linear[Product] · σ(h̃(P))  +  h_old(P)
                                            ↑ residual
```

`A_Linear` is **per target type** — a Product integrates information differently than a Warehouse. The **residual** is why HGT tolerates depth better than plain GNNs: each node keeps its identity, and layers *add refinements* instead of overwriting.

## 1.4 Worked example

**Product P** updating, with neighbours C1, C2 (via `USED_IN`) and O1 (via `ORDERED`):

| Step | What happens |
|---|---|
| 1 | P forms its query using the **Product** Q-matrix |
| 2 | C1, C2 form keys using the **Component** K-matrix |
| 3 | O1 forms its key using the **Order** K-matrix — *a completely different matrix* |
| 4 | C1's score goes through `W_ATT[USED_IN]`; O1's through `W_ATT[ORDERED]` |
| 5 | Each scaled by its meta-relation's `μ` |
| 6 | Softmax over all three → e.g. `C1: 0.5, C2: 0.2, O1: 0.3` |
| 7 | Messages built via type-specific `M_Linear`, reshaped by edge matrices |
| 8 | Weighted sum → `A_Linear[Product]` → add P's old embedding |

**What GAT would do:** one shared matrix for all three. It could still learn those same weights — but it has no way to express that *"a component supplying me"* and *"an order demanding me"* are fundamentally different kinds of information. Listening harder without understanding the language.

## 1.5 The genuinely clever bit — parameter factorisation

**Naive approach:** one matrix per meta-relation triple. With 8 types and 10 relations that's hundreds of `d×d` matrices, most trained on too few examples.

**HGT factorises:**

| Parameters | Count | Shared across |
|---|---|---|
| `K, Q, M, A` | 4 per **node type** = 32 | every edge touching that type |
| `W_ATT, W_MSG` | 2 per **edge type** = 20 | every node pair using that edge |
| `μ` | 1 scalar per meta-relation | — |

**52 matrices instead of hundreds.** At `d = 64`:

```
Node-type params:  8 × 4 × 64×64      = 131,072   (85%)
Edge-type params:  10 × 2 × 64×64/4   =  20,480   (15%)
                                      ──────────
                          per layer   ≈ 154,000
```

| Encoder depth | Parameters | GFLOP | k-hop reach |
|---|---|---|---|
| L = 1 | 171,725 | 0.49 | ~0% |
| L = 2 | 325,847 | 0.98 | 1.5% |
| L = 3 | 479,969 | 1.47 | 12% |
| L = 4 | 634,091 | 1.97 | **94%** |
| L = 5+ | 788,213+ | 2.46+ | 100% — nothing new |

**Practical consequence:** adding a node type costs ~6× more than adding an edge type. `SUB_SUPPLIES` is cheap; a `Contract` node type would not be.

## 1.6 Pros

| Advantage | Why it matters here |
|---|---|
| **Type-aware by construction** | Your graph genuinely has 8 kinds of thing; every simpler model must pretend otherwise |
| **No hand-designed meta-paths** | Older models (HAN, metapath2vec) need you to specify `Supplier→Component→Product` manually. HGT learns which paths matter. |
| **Parameters scale as (types + relations)** | Not their product — makes rich schemas affordable |
| **Residual connections** | Trainable at depth without immediate collapse |
| **Attention weights are inspectable** | Explainability hooks for free |
| **First-class PyG support** | `HGTConv` exists — you're not implementing from scratch |

## 1.7 Cons

| Limitation | Consequence |
|---|---|
| **Most expensive of the three** | ~326K params at L=2 vs GAT's 103K |
| **Fixed receptive field** | L layers = exactly L hops |
| **Over-smoothing at depth** | Residuals delay it; they don't prevent it |
| **No sense of time** | One snapshot, no event history |
| **Rare relations get badly-trained parameters** | `SUB_SUPPLIES` is sparse — its dedicated matrices would be estimated from very few examples. Per-relation parameters are a liability when a relation is rare. |
| **Treats every edge as equally real** | Your `supplier_relationships.confidence` column exists *because* tier-2 data is self-reported. HGT has no way to consume that. |

## 1.8 Why HGT alone is not suitable here

Five gaps. Each is **structural**, not a tuning problem.

### Gap 1 — It cannot connect nodes with no path

Two suppliers share an unrecorded tier-2 source and fail together. They have **no edge chain between them.**

Message passing walks edges. No edge, no message — regardless of layer count, hidden dimension, or training time. Adding layers doesn't help because there is nothing to walk.

→ **Filled by Transformer 2.**

### Gap 2 — One depth for three different targets

| Head | Target node | Needs |
|---|---|---|
| Delay | Supplier / Shipment | ~2 hops |
| Shortage | Product / Warehouse | ~3 hops |
| Impact | Order / Customer | ~4 hops |

Vanilla HGT produces **one** final-layer embedding that all heads read. You're forced to pick one `L` that is simultaneously too deep for delay and too shallow for impact.

→ **Filled by the Markov readout (or Transformer 1).**

### Gap 3 — Going deep enough for impact destroys the shallow predictions

"Just use L=4" doesn't work. At L=4 the receptive field covers **~94% of the graph**. `h⁴` is close to a graph-wide average — every node looks the same.

→ **Filled by retaining `h¹–h⁴` and reading the appropriate one per task.**

### Gap 4 — It produces a global score, but risk is relational

*"Supplier X: 80% failure probability"* conflates X's own fragility with **your** exposure to it. A fragile supplier for whom you're a top-priority customer will serve you first.

The **node** is HGT's unit of prediction. One node, one embedding, one score. There is no architectural slot for a *(supplier, buyer)* pair.

→ **Filled by Claim B — deterministically, outside the model.**

### Gap 5 — It doesn't know it's blind

A supplier with genuinely no upstream dependencies and a supplier whose upstream you simply *failed to record* look **identical** — both have zero incoming `SUB_SUPPLIES` edges. Absence of an edge carries no information in message passing.

→ **Filled by `is_frontier` + frontier-aware attention.**

### Summary

| Gap | Root cause | Filled by |
|---|---|---|
| Can't couple unconnected suppliers | Message passing needs a path | **Transformer 2** |
| One depth, three targets | Single final-layer output | **Markov readout** |
| Deep enough = over-smoothed | Repeated averaging | **Intermediate-layer retention** |
| Global score ≠ your risk | Node is the prediction unit | **Claim B** |
| Can't distinguish "none" from "unknown" | Absent edges carry no signal | **`is_frontier` + T2** |

**HGT is the right encoder and the wrong complete system.** It solves exactly one problem extremely well — learning type-aware local structure — and every other component exists because it solves *that one problem and nothing else*.

---

# PART 2 — TRANSFORMER 1: The Zoom Lens

## 2.1 The problem it inherits

HGT runs 4 layers and, by default, throws away three:

```
h⁰ → [1] → h¹ → [2] → h² → [3] → h³ → [4] → h⁴
             ✗          ✗          ✗          ✓ keep only this
```

That assumes **every node needs exactly 4 hops**. A supplier selling one component to one product needs almost none. A supplier feeding twelve product lines needs a lot. Forcing both through the same depth means one is wrong.

## 2.2 What it receives

Stop discarding, and every node carries a **4-token sequence**:

```
[ h¹ , h² , h³ , h⁴ ]
```

**The conceptual leap:** this sequence is not time, and not word order. It's **depth**.

**Analogy — a camera zoom:**

| Token | Shot | Sees |
|---|---|---|
| `h¹` | Macro | Just me and my direct neighbours |
| `h²` | Close-up | My neighbourhood |
| `h³` | Medium | My region |
| `h⁴` | Wide | ~94% of the graph |

Vanilla HGT forces every node to use the wide shot. **T1 learns the right zoom per node.**

## 2.3 The mechanism

```
Q = W_Q · [h¹,h²,h³,h⁴]     K = W_K · [...]     V = W_V · [...]
A = softmax( Q Kᵀ / √d )                # a 4×4 matrix per node
z_i = Σ_k w_k · h^k                     # fused embedding → Transformer 2
[w₁..w₄]                                # 4 weights summing to 1 → persisted
```

Note: **one shared set of weights for all nodes** — unlike HGT, no per-type parameters. And note that second output: most components produce only a representation. T1 also produces a **measurement**.

## 2.4 Worked example

**Supplier A** — one component, one product, one customer:

| | `h¹` | `h²` | `h³` | `h⁴` |
|---|---|---|---|---|
| Learned weight | **0.72** | 0.21 | 0.05 | 0.02 |

*"My immediate situation is my situation. Beyond two hops I'm absorbing the graph average — noise, not signal."*

**Supplier B** — twelve products, three factories, recorded tier-2:

| | `h¹` | `h²` | `h³` | `h⁴` |
|---|---|---|---|---|
| Learned weight | 0.18 | **0.34** | **0.36** | 0.12 |

*"My risk genuinely depends on structure two to three hops out."*

**Vanilla HGT gives both nodes `h⁴` and nothing else.** For Supplier A that's actively harmful.

## 2.5 Cost — a counterintuitive fact

| Version | Params | GFLOP |
|---|---|---|
| Full T1 (2 layers, ff=4) | 99,968 | **3.93** |
| Prior-initialised gate | 2,212 | 0.022 |

**T1 costs twice as much to run as the entire 4-layer HGT encoder** (1.97 GFLOP), despite being 16% of its parameters.

Why: HGT's cost is dominated by *edges* — sparse, 20,000. T1 runs dense projections on **every node × 4 tokens × 2 layers** = 40,000 dense operations. Sparse structure is what makes GNNs cheap; T1 discards it.

**Judge T1 on FLOPs, not parameters.**

## 2.6 Pros

| Advantage | Why |
|---|---|
| **Per-node adaptivity** | Hubs and leaves handled without enumerating cases |
| **Over-smoothing mitigation** | Deep layers become *optional* — nodes that don't benefit assign near-zero weight |
| **Produces a measurement** | The depth profile is a reportable finding |
| **Cheap in parameters** | 100K on a 634K model |
| **Well-grounded** | Jumping Knowledge (Xu et al., 2018) is established and consistently improves GCN/SAGE/GAT |

## 2.7 Cons

| Limitation | Why it bites |
|---|---|
| **3× inference FLOPs** | The dominant cost, invisible if you only count parameters |
| **Measures usage, not sufficiency** | Attention weights ≠ conditional independence — cannot test the depth hypothesis however carefully run |
| **No labels to validate against** | You can check stability, never correctness |
| **6 training runs** | Dual-run + 5-fold CV, if used as a scientific instrument |
| **Overlaps with T2** | Both attention over related representations — hard to attribute credit |
| **Mostly redundant given a structural prior** | Machinery sized for learning depth from scratch |

## 2.8 Without T1, what breaks?

- Every node forced to `h⁴` — a representation covering ~94% of the graph
- Leaf suppliers get maximally over-smoothed representations
- **All three heads read the same depth**, despite targets at 2, 3 and 4 hops
- No depth measurement, so the sufficiency question is unanswerable from the model

**But:** the Markov readout fixes the second and third for **zero parameters**. Only per-*node* adaptivity genuinely requires T1 — and a 2,212-parameter gate delivers that at 1/45th the cost.

---

# PART 3 — TRANSFORMER 2: The Satellite Photo

## 3.1 The problem it inherits

Two suppliers fail in the same week. **No edge chain between them** — different components, different products, different customers. But both quietly buy from the same tier-2 source you never recorded.

**Message passing cannot help at any depth.** A 4-layer encoder and a 40-layer encoder are equally blind. This is a structural limit, not a capacity limit — the one gap you cannot close by making HGT bigger.

## 3.2 The mechanism

```
Q, K, V  ←  all from z_i
A = softmax( Q Kᵀ / √d )       ← NO adjacency mask
```

**That missing mask is the entire idea.**

| | Question asked |
|---|---|
| Message passing | *"Who are you connected to?"* |
| Global attention | *"Who do you resemble?"* |

**Analogy.** HGT knows people through the **org chart** — who reports to whom. T2 notices two people in unrelated departments always take the **same days off**. No line between them on the chart, but you infer a shared cause.

Geographically: message passing follows **roads**. T2 looks at a **satellite photo** and sees two towns with identical architecture — probably built by the same company — with no road between them.

## 3.3 Worked example

Suppliers **S1** (Germany, connectors) and **S2** (Malaysia, casings). No shared component, product, or customer. Graph distance: effectively infinite.

But their learned embeddings encode similar patterns — both late in Q2 2024, both with capacity dips in the same window.

```
attention(S1 → S2) = 0.31        ← unusually high for unrelated nodes
```

Persisted to `hidden_dependency_links` with its weight and `model_version`. A reviewer investigates and finds both source a specialty polymer from one plant.

**HGT could never surface that** — not for lack of capacity, but because there was no path to walk.

## 3.4 The mandatory constraint (Fix A)

T2 attends **only between same-type pairs**. Supplier↔Supplier. Never Supplier↔Product.

**Why mandatory, not optional:** HGT works hard to keep each node type's embedding space *distinct* via type-specific parameters. Unconstrained global attention recombines every type into one shared space — partially undoing exactly the work that makes HGT worth its cost.

The constraint also matches the goal. The coupling you care about is supplier-to-supplier. Supplier-to-product coupling is already fully described by the edges you have.

## 3.5 Frontier nodes — the second job

Nodes flagged `is_frontier = true` enter the attention pool **on equal footing**.

In message passing, a frontier node is an **orphan** — no incoming edges means no messages to aggregate, so HGT has literally nothing to say about it. In embedding attention it's a full participant, because similarity requires no edges.

**Frontier nodes are precisely the ones T2 exists to serve.**

## 3.6 Cost — the other counterintuitive fact

| | Params | GFLOP |
|---|---|---|
| T2 (1 layer, ff=4, 600 suppliers) | 49,984 | **0.112** |

**T2 adds ~6% to inference cost. T1 adds 200%.**

This inverts intuition. T2 is *quadratic* — sounds terrifying — but runs on **600 suppliers**, not 5,000 nodes. T1 is *linear* — sounds safe — but runs on **all 5,000 nodes × 4 tokens**.

At your scale, quadratic-on-a-subset beats linear-on-everything. The quadratic term only dominates past ~2,000 suppliers:

| Suppliers | T2 cost | % quadratic |
|---|---|---|
| 200 | 12 MFLOP | 44% |
| 600 | 66 MFLOP | 70% |
| 2,000 | 578 MFLOP | 89% |
| 5,000 | 3,364 MFLOP | 95% |

Bounding the candidate pool (same `component_type`, frontier-flagged, or top-k by cosine) keeps the capability and drops cost to `O(N·k·d)`.

## 3.7 Pros

| Advantage | Why |
|---|---|
| **The only mechanism that closes Gap 1** | Nothing else can couple nodes with no path — not depth, not width, not training |
| **Serves frontier nodes** | Turns orphans into participants |
| **Cheap at your scale** | +6% FLOPs |
| **Produces reviewable findings** | `hidden_dependency_links` is an auditable artifact |
| **Closes the BOM reach gap** | Global gathering sidesteps needing a 5th message-passing hop |
| **Graceful failure** | Finds nothing → pipeline still works |

## 3.8 Cons

| Limitation | Why it bites |
|---|---|
| **Correlation ≠ causation** | A discovered pair may be coincidence (ML-12); the architecture can't tell |
| **Rests on an untested assumption** | That genuine supplier-to-supplier correlation exists in your data |
| **Type-flattening pressure** | Fix A mitigates but doesn't eliminate the tension |
| **Quadratic beyond ~2,000 nodes** | 95% of cost is quadratic at 5,000 suppliers |
| **Counterintuitive findings erode trust** | A user who can't see *why* two unrelated suppliers were coupled may distrust the system even when it's right |
| **Needs a persistence threshold** | Too low and reviewers drown in spurious pairs |

## 3.9 Without T2, what breaks?

- **Correlated systemic failure becomes invisible** — the failure mode that actually destroys supply chains
- **Frontier nodes stay orphans** — no signal for exactly the suppliers you know least about
- `is_frontier` becomes decorative
- Your architecture has no answer to *"what about the tier-2 you can't see?"* — the first question any practitioner asks

**Nothing else can substitute.** T1 can be replaced by a 2K gate. T2 cannot be replaced at all.

---

# PART 4 — TRANSFORMER WHAT-IF SCENARIOS

| Scenario | Params | GFLOP | vs. HGT alone |
|---|---|---|---|
| **A — HGT alone** | 634,091 | 1.97 | baseline |
| **B — HGT + T1** | 734,059 | 5.90 | +16% params, **+200% FLOPs** |
| **C — HGT + T2** | 684,075 | 2.08 | +8% params, **+6% FLOPs** |
| **D — HGT + T1 + T2** | 784,043 | 6.01 | +24%, +206% |
| **E — HGT + gate + bounded T2** | 690,711 | 2.14 | **+9%, +9%** |

### A — HGT alone

**Works:** typed local structure, all three predictions, explanations, embeddings. A complete, defensible system.

**Breaks:** one depth for three differently-positioned targets. Frontier suppliers invisible. Correlated failure undetectable.

**Verdict:** the correct Phase-1 baseline. Everything else must earn its place against this.

### B — HGT + T1 only

**Fixes:** per-node depth, over-smoothing, a depth measurement.

**Still broken:** frontier suppliers still orphans. Correlated failure still invisible. **You've made the reachable parts sharper while remaining blind to what you can't reach.**

**Verdict:** worst value on the table — triples inference for the gap with the cheapest alternative.

### C — HGT + T2 only

**Fixes:** frontier coupling, correlated failure detection, BOM reach.

**Still broken:** all heads read the same over-smoothed `h⁴` — but the Markov readout fixes that for **zero parameters**, no T1 required.

**Verdict:** **the best single addition, by a wide margin.** Highest value per FLOP in the design.

### D — HGT + T1 + T2 (full spec)

**Fixes:** everything except Gap 4 (deliberately outside the model).

**Costs:** 784K parameters against a few hundred positive labels, plus the T1/T2 functional overlap.

**Verdict:** technically complete, poor value.

### E — HGT + gate + bounded T2 ★

**Fixes:** all four addressable gaps. **Costs:** +9% params, +9% FLOPs.

**Verdict:** **recommended.** Same capability as D at roughly one-twentieth the marginal compute.

---

# PART 5 — THE MARKOV APPROACH

## 5.1 The problem it inherits

Identical to T1's. **T1 and Markov are two answers to the same question.** T1 says *"let the model learn the right depth."* Markov says *"work it out from the shape of the graph before training starts."*

## 5.2 The concept — a Markov blanket

A node's **Markov blanket** is the smallest set of nodes such that, once you know them, nothing else in the graph tells you anything more. Three parts: **parents**, **children**, and **co-parents** (the *other* causes of its children).

**Analogy.** To predict whether a student passes an exam you need their study habits (parent), their mock scores (child), and the paper's difficulty (co-parent — the *other* thing that determined the mocks). Their grandparents' habits add nothing once you know the student.

That third category is the one people forget, and here it's decisive.

## 5.3 The blanket in your graph

For a **Supplier** node:

| Part | In your graph | Hops | Captured? |
|---|---|---|---|
| Children | Components it supplies | 1 | Yes |
| **Co-parents** | **Other suppliers of those same components** | **2** | Yes, incidentally |
| Parents | Upstream suppliers (`SUB_SUPPLIES`) | 1 | Sparse/absent |
| Moralised co-parents | Suppliers sharing an *unobserved* upstream | ∞ | **No** → T2's job |

**Two consequences.** The co-parent term gives a **derived floor of L = 2** — not guessed, counted. And the last row is precisely Transformer 2's justification.

## 5.4 The key insight — the blanket is per-target

| Head | Target lives on | Blanket radius | Reads |
|---|---|---|---|
| Delay | Supplier / Shipment | ~2 hops | `h²` |
| Shortage | Product / Warehouse | ~3 hops | `h³` |
| Impact | Order / Customer | ~4 hops | `h⁴` |

**One global depth is wrong for all three simultaneously.**

## 5.5 The mechanism — two moves, zero parameters

**Move 1 — per-task readout depth.** No module, no training, no weights. You choose **which tensor each head reads**.

**Move 2 — relation-scoped depth budget.**

```
BOM relations (SUPPLIES, USED_IN, ORDERED, PLACED_BY)  →  full depth
SUB_SUPPLIES                                            →  scaled by α^tier, α ≈ 0.2–0.3
```

Implemented by setting HGT's existing `μ` prior. **Also zero new parameters.**

> **Critical detail — tier ≠ layer.** "Tier-1 and tier-2 only" must be capped on the `SUB_SUPPLIES` axis, *not* by capping `L`. Your BOM path is 4 hops and contains **zero** tier traversal. Capping `L=2` wouldn't limit upstream reach — it would stop a supplier from ever seeing the orders its components end up in.

## 5.6 Worked example

**Supplier A** flagged for delay risk:

| Head | Reads | Sees | Why |
|---|---|---|---|
| Delay | `h²` | A, its components, other suppliers of those components | Substitutability and co-parent load predict A's lateness |
| Shortage | `h³` | + products and warehouse stock positions | Shortage is a property of stock, not of A |
| Impact | `h⁴` | + orders and customers downstream | Impact is measured in commitments at risk |

**Vanilla HGT gives all three `h⁴`** — the delay head, which needed a tight 2-hop view, gets a graph-wide average.

## 5.7 Cost

**Zero parameters. Zero FLOPs.** You compute `h¹…h⁴` either way.

## 5.8 Pros

| Advantage | Why |
|---|---|
| **Free** | Decisive at a few hundred positive labels |
| **Fixes a real modelling error** | One depth for three differently-positioned targets was simply wrong |
| **Defensible in one sentence** | *"Delay's target sits on the Supplier node, its blanket is 2 hops, so the delay head reads layer 2"* |
| **No methodological failure modes** | Nothing to bias, no circularity, no "inconclusive" outcome |
| **Smaller explanations** | 2-hop subgraph ≈ 73 nodes vs 4-hop ≈ 4,700 |
| **Fewer training runs** | 4 (L-sweep) vs 6 (dual-run + CV) |
| **Works with the data you have** | Needs no tier-2 edges |

## 5.9 Cons

| Limitation | Why it bites |
|---|---|
| **Fixed per task, not per node** | A leaf and a hub supplier get identical depth |
| **Asserted, not measured** | Unless the L-sweep validates it, it's a design claim |
| **No research novelty** | Sound engineering, not a contribution |
| **Purely structural** | Cannot express *"look deeper only when tier-1 and tier-2 disagree"* — that's context-specific independence, needing a separate gate |
| **Blanket ≠ readout layer exactly** | Layer *k* mixes information from ≤ *k* hops, so the mapping has slack |

## 5.10 An important limitation of the theory itself

Claim A requires `P(risk | tier-1 state) = P(risk | tier-1 state, upstream)`. That holds exactly only if the tier-1 observation is complete and noise-free. `reliability_history` is lagged, noisy and partial — so some upstream information survives.

**And it's weakest where the system claims most value.** The screening works for stress that has *already surfaced* in behaviour. But the pitch is early warning. A tier-2 failure that hasn't yet degraded tier-1 observables is exactly the case where deep visibility would pay.

**Honest framing:** you cap at tier-1+ε because tier-2 data is sparse-to-absent, and Claim A is the argument for why that costs less than it sounds. That's a stronger position than presenting it as a free choice.

---

# PART 6 — MARKOV SCENARIOS

| # | Scenario | L | Params | GFLOP |
|---|---|---|---|---|
| 1 | **T1 only** (no Markov) | 4 | 734,059 | 5.90 |
| 2 | **Markov only** | 3 | **479,969** | **1.47** |
| 3 | **Full T1 + Markov** (3 instances) | 4 | 933,995 | 13.76 |
| 3b | Full T1 + Markov (shared trunk) | 4 | 734,059 | 5.90 |
| 4 | **Minimal T1 (gate) + Markov** | 4 | 640,727 | 2.03 |

## Scenario 1 — T1 only

**Gets you:** per-node adaptivity, a depth measurement, over-smoothing mitigation.

**Fails to get you:** per-*task* depth. A single T1 produces **one** fused `z_i` that all three heads read — so you've solved node heterogeneity while leaving the target-position problem untouched. Delay and impact still share a representation.

**Also:** learning depth from a few hundred positives is exactly the setup ML-11 warns produces noise. And 6 training runs to make the measurement credible.

**Verdict:** most expensive, least grounded, and doesn't fix what Markov fixes. **Weakest of the four.**

## Scenario 2 — Markov only

**Gets you:** per-task depth, free, defensible, with the smallest explanations and cheapest inference — 24% fewer parameters and 25% fewer FLOPs than anything else here.

**Fails to get you:** per-node adaptivity; no residual finding.

**Verdict:** **the correct Phase-1 configuration.** Zero risk, zero cost, fixes the actual modelling error.

## Scenario 3 — Full T1 + Markov

Here's the trap, and it's the most important number in this document.

**Per-task readout forces you to instantiate the fusion mechanism once per head.** One shared T1 produces one `z_i` — which is exactly what destroys per-task depth. To keep it, you need three.

| Rung | ×1 | ×3 (one per head) |
|---|---|---|
| 2-layer T1, ff=4 | 99,968 / 3.93 GFLOP | **299,904 / 11.80 GFLOP** |
| Gate | 2,212 / 0.02 | **6,636 / 0.07** |

Three full T1s is **13.76 GFLOP — seven times the entire HGT encoder** — for a depth-fusion mechanism.

*Scenario 3b* (shared attention trunk, three lightweight output heads) claws it back to 5.90 GFLOP, but constrains the trunk to serve all three depth decisions simultaneously.

**Verdict:** technically complete, terrible value. Avoid.

## Scenario 4 — Minimal T1 (gate) + Markov ★

Three prior-initialised gates, one per head, each centred on its own blanket depth.

**Gets you:** everything. Per-task depth (Markov), per-node adaptivity (gates), and the **residual finding** — *where does the model disagree with the blanket, and why?*

**Costs:** +6,636 parameters and +0.07 GFLOP over Markov-only at the same `L`. Under 1%.

**Versus Scenario 3:** **293,268 fewer parameters and 6.8× less compute, for the same capability.**

**Verdict:** **recommended.**

## 6.1 The best scale of T1 in the hybrid

**Answer: the prior-initialised gate, `d → 32 → 4`, one per head. 2,212 parameters each.**

### The full ladder

| Rung | Version | ×1 params | ×1 GFLOP | ×3 params | ×3 GFLOP |
|---|---|---|---|---|---|
| 1 | 2-layer, ff=4 (as specced) | 99,968 | 3.93 | 299,904 | 11.80 |
| 2 | 1-layer, ff=4 | 49,984 | 1.97 | 149,952 | 5.90 |
| 3 | 1-layer, ff=2 | 33,472 | 1.31 | 100,416 | 3.93 |
| 4 | 1-layer, attention only | 16,960 | 0.66 | 50,880 | 1.97 |
| 5 | **Prior-initialised gate** ★ | **2,212** | **0.02** | **6,636** | **0.07** |
| 6 | Fixed depth (Markov only) | 0 | 0 | 0 | 0 |

### Pros and cons per rung

**Rung 1 — 2-layer, ff=4.** *Pros:* maximum expressiveness; matches the "GNN × Transformer" framing most literally. *Cons:* 300K parameters at ×3 against a few hundred positives; 6× the encoder's compute; almost entirely redundant once a prior exists. *Verdict:* only justified with no prior and >2,000 positives.

**Rung 2 — 1-layer, ff=4.** *Pros:* halves everything for little expressiveness loss; one attention layer over four tokens is plenty. *Cons:* still 150K at ×3, and the FFN — the largest component — does the least useful work. *Verdict:* reasonable compromise, not the best value.

**Rung 3 — 1-layer, ff=2.** *Pros:* cuts where most remaining cost lives; still a real Transformer block. *Cons:* the reduced FFN is an arbitrary size with no principled justification. *Verdict:* if you're cutting this far, keep going.

**Rung 4 — 1-layer, attention only.** *Pros:* keeps the genuinely useful part (attention *between* depth tokens, so `h¹` and `h³` can interact) and drops the part that mostly adds capacity; clean conceptual story. *Cons:* unusual configuration needing justification; still 30× rung 5 for an undemonstrated benefit. *Verdict:* the right rung *if* you can show the depth tokens need to interact. Climb here only with evidence.

**Rung 5 — Prior-initialised gate. ★** *Pros:* effectively free; produces exactly the specified output; **keeps the entire research artifact** (the residual); initialise at the blanket depth and the untrained model reproduces Markov-only behaviour exactly, so training can only improve on it; if weights never move meaningfully, *that is itself the Claim A evidence*. *Cons:* no attention between depth tokens — largely fixable by feeding the gate a concatenation of pooled `h¹–h⁴` (still under 5K); slightly weakens the "Transformer" narrative, though T2 and the prediction head still carry Transformer content. *Verdict:* **recommended.**

**Rung 6 — Fixed depth.** *Pros:* absolutely free, nothing to train or explain. *Cons:* no adaptivity; **no residual, so no research artifact**. *Verdict:* correct Phase-1 baseline, but rung 5 costs ~1% over it and buys back the entire finding.

### Five reasons the gate wins in the hybrid

1. **The prior makes the job small.** Unhybridised, T1 must discover a four-way distribution per node unaided. Hybridised, it learns a *correction to a known answer* — and most corrections are near zero.
2. **The output spec is already a gate.** §6.2 asks for *"4 values summing to 1."* A convex combination with learned weights **is** a gate. QKV plus an FFN does far more than that requires.
3. **Per-head instantiation triples whatever you choose.** 3 × 2,212 = 6,636 is nothing. 3 × 99,968 = 299,904 is a third of your encoder.
4. **Your label volume can't support the alternative.** ~500 positives against 300K parameters of depth machinery is a ratio nothing justifies — ML-09 flags exactly this.
5. **It degrades gracefully.** Worst case: it learns nothing and you've spent 6,636 parameters. There is no failure mode where you're worse off than Markov-only.

### When to climb

| Signal | Action |
|---|---|
| Gate weights consistently pinned at the edge of expressible range | Climb to rung 4 |
| Positives exceed ~2,000 **and** rung 4 still saturates | Consider rung 3 |
| Neither | **Stay at rung 5** |

Climbing *with* evidence is defensible. Starting at rung 1 and hoping is how the original spec reached a 100K-parameter component for a problem with a few hundred labels.

### One non-negotiable ordering rule

**Claim A must move to the L-sweep before any prior is placed on a learned depth mechanism.** Biasing the gate toward a blanket prior and then citing its weights as evidence about graph structure is the exact circularity Fix B was written to prevent — arriving from a new direction. Get the order wrong and every downstream result is invalid.

---

# PART 6.5 — HOW MUCH TO BLEND

You've decided to blend Markov and Transformer 1. The remaining question is *how far* — and that's a dial, not a switch.

## 6.5.1 The two knobs

Blending has two independent controls, and "extent of blend" means moving both together.

**Knob A — prior strength.** How hard the Markov answer pulls.

```
w_k ∝ exp( −|k − blanket| / τ )        ← prior shape, τ controls width
loss += λ · KL( learned ‖ prior )      ← λ controls how hard it pulls
```

| Setting | Effect |
|---|---|
| `τ = 0.4`, `λ = 0.1` | Narrow, strong — T1 can barely move |
| `τ = 0.8`, `λ = 0.01` | Medium, light — T1 moves where evidence supports |
| `τ = 2.0`, `λ = 0` | Effectively uniform — no prior at all |

**Knob B — T1 capacity.** How much machinery learns the deviation. The shrink ladder from §6.1.

**They move together.** A strong prior means less to learn, which means less capacity is needed. A weak prior means T1 must discover more, which needs more capacity. Mismatching them is the failure mode: heavy capacity under a strong prior is wasted parameters; light capacity under no prior can't learn what it needs to.

## 6.5.2 The five blend levels

| Level | Configuration | Blend params | Blend GFLOP | Total | Total GFLOP |
|---|---|---|---|---|---|
| **L0** | Fixed readout — pure Markov | 0 | 0 | 634,091 | 1.97 |
| **L1** | Gate ×3, `τ=0.4`, `λ=0.1` — prior-locked | 6,636 | 0.065 | 640,727 | 2.03 |
| **L2** | Gate ×3, `τ=0.8`, `λ=0.01` — prior-anchored ★ | 6,636 | 0.065 | 640,727 | 2.03 |
| **L2+** | Concat-gate ×3 (reads pooled `h¹–h⁴`), `λ=0.01` | 25,068 | 0.250 | 659,159 | 2.22 |
| **L3** | Attention-only T1 ×3, init-only prior, `λ=0` | 50,880 | 1.966 | 684,971 | 3.93 |
| **L4** | Full 2-layer T1 ×3, uniform init — pure T1 | 299,904 | 11.796 | 933,995 | 13.76 |

**The span:** L4 costs **45× the parameters and 181× the compute** of L2 for the same nominal capability. L2 adds **1.05% parameters and 3.3% FLOPs** over pure Markov.

## 6.5.3 What each level buys and costs

### L0 — Fixed readout (blend = 0%)

*Accuracy:* baseline. Correct per-task depth, no per-node adaptation.
*Compute:* free.
*Pros:* nothing can go wrong; smallest explanations; fewest training runs.
*Cons:* leaf and hub suppliers treated identically; **no residual, so no research finding**.
*Pick if:* you're running out of time, or positives are under ~300 and you want zero risk.

### L1 — Prior-locked (blend ≈ 20%)

*Accuracy:* marginal gain over L0. The gate can nudge but not depart.
*Compute:* +1% params, +3% FLOPs.
*Pros:* maximum stability — weights will pass the CV check comfortably; a residual exists, even if small; safest way to get a measurement.
*Cons:* if the blanket estimate is genuinely wrong, the strong prior suppresses the correction and you'd never know. You get a confidently-wrong answer rather than a noisy right one.
*Pick if:* positives under ~500 and you mainly want the residual as a *sanity check* rather than a finding.

### L2 — Prior-anchored ★ (blend ≈ 50%)

*Accuracy:* best expected at moderate label volume. Prior handles the common case; gate corrects where data insists.
*Compute:* +1% params, +3% FLOPs.
*Pros:* the residual is genuinely informative — deviations mean something; initialised at the prior, so the worst case is L0 behaviour; both correction directions observable; cheapest configuration that produces a real finding.
*Cons:* `λ` is a hyperparameter you now have to tune; weaker prior means slightly noisier weights than L1.
*Pick if:* positives between ~500 and ~2,000. **Default recommendation.**

### L2+ — Concat-gate (blend ≈ 55%)

Same as L2, but the gate reads a concatenation of pooled `h¹–h⁴` rather than node features alone.

*Accuracy:* recovers most of the depth-token interaction that a plain gate loses.
*Compute:* +4% params, +13% FLOPs. Still trivial.
*Pros:* addresses the single legitimate criticism of the gate — that it can't let `h¹` and `h³` interact — for 18K parameters.
*Cons:* slightly more to explain; the benefit is plausible but undemonstrated.
*Pick if:* L2's weights look under-expressive, or you want to pre-empt the "a gate can't do what attention does" objection cheaply.

### L3 — Prior-seeded, free to move (blend ≈ 75%)

*Accuracy:* better *if* the blanket estimate is meaningfully wrong. Otherwise equal to L2 with more variance.
*Compute:* +8% params, **+100% FLOPs** — doubles inference.
*Pros:* genuine attention between depth tokens; can express corrections a gate cannot; still benefits from the prior as initialisation.
*Cons:* doubles inference cost for an unproven gain; with no KL pull, weights can drift on scarce labels (exactly ML-11); you're back to needing a stability check.
*Pick if:* positives exceed ~2,000 **and** L2/L2+ weights are visibly saturating at the edges of what a gate can express.

### L4 — Pure T1, no prior (blend = 0%, other end)

*Accuracy:* highest ceiling, highest variance. Needs the most data to realise its ceiling.
*Compute:* **+47% params, +600% FLOPs.**
*Pros:* maximum expressiveness; the depth measurement is unbiased and citable as Claim A evidence without the ordering caveat.
*Cons:* 300K parameters against a few hundred positives; six training runs; the dual-run protocol; a real chance the result comes back "inconclusive" after all that.
*Pick if:* you have several thousand positives and depth is genuinely the research question. **Not your situation.**

## 6.5.4 The tradeoff, stated properly

This is the **bias–variance tradeoff**, and your label count picks the optimum.

```
More Markov  →  more bias, less variance   →  better when labels are scarce
More T1      →  less bias, more variance   →  better when labels are abundant
```

- **Bias risk:** the blanket estimate is off by a layer, and a strong prior stops the model from correcting it.
- **Variance risk:** the model learns fold-specific noise instead of a real depth preference (ML-11).

At ~500 positives, **variance is the larger danger**. That argues for L1–L2. As positives grow past ~2,000, bias becomes the binding constraint and L3 starts to pay.

## 6.5.5 The measurement caveat that decides it

Here's the honest constraint on this entire decision:

| Positive labels | 95% CI on AUC | Smallest detectable difference |
|---|---|---|
| 200 | ±0.037 | ~0.037 |
| 500 | ±0.024 | ~0.024 |
| 1,000 | ±0.017 | ~0.017 |

**At your label volume, L1 through L3 will most likely be statistically indistinguishable.** A depth-fusion mechanism beating another by more than 0.024 AUC would be a surprising result.

**If you cannot measure the accuracy difference, choose on cost.** And on cost the answer is unambiguous: L2 costs 3% of inference; L4 costs 600%.

That's not a compromise — it's the correct decision under the evidence you'll actually have.

## 6.5.6 Decision table

| Your situation | Blend level | Why |
|---|---|---|
| Positives < 300, or time-constrained | **L0** | Zero risk, zero cost |
| Positives 300–500 | **L1** | Residual as a sanity check, maximum stability |
| **Positives 500–2,000** | **L2** ★ | Best value; informative residual; 3% overhead |
| L2 weights look under-expressive | **L2+** | Cheapest fix for the one real gate criticism |
| Positives > 2,000 and L2+ saturating | **L3** | Only now does attention earn its 2× compute |
| Positives > 5,000 and depth *is* the research question | L4 | Not your situation |

## 6.5.7 How to move between levels without re-architecting

Design for L2 and make the blend a **configuration**, not a rewrite:

```python
depth_gate = DepthGate(
    prior      = blanket_prior[head],   # [w1..w4] per head
    tau        = 0.8,                   # prior width      ← knob A
    kl_lambda  = 0.01,                  # prior strength   ← knob A
    hidden     = 32,                    # gate capacity    ← knob B
    input_mode = "node",                # or "concat" for L2+
)
```

L0 through L2+ are then reachable by changing four values. Only L3 and L4 require swapping the module — and by the time you'd want them, you'll have the label count to justify the work.

**Start at L2. Tune `λ` and `τ` against validation. Climb only if the weights tell you to.**

---

# PART 7 — COMPARED WITH GraphGPS AND GraphTrans

## 7.1 What GraphGPS actually is

One GraphGPS layer runs **two branches in parallel**, then merges:

```
        ┌─ MPNN branch (local) ──┐
h_in ───┤                        ├── merge ── h_out    ← repeat × L
        └─ Attention branch ─────┘
              (global)
```

Plus **positional/structural encodings** (Laplacian eigenvectors, random-walk features) injected before layer 1.

**GraphTrans** is the other template: run a full MPNN stack **first**, then feed its output through a Transformer **once**, sequentially.

## 7.2 Exactly the same

| Aspect | Shared with GraphGPS |
|---|---|
| Core thesis | Local message passing + global attention beats either alone |
| Problem addressed | Over-smoothing, over-squashing, long-range dependencies |
| Ingredient classes | An MPNN branch and a Transformer-attention branch |
| Modularity claim | "Swap the local branch for any MPNN" — GraphGPS says it; you do it |
| Global branch math | Standard multi-head self-attention, no adjacency mask |
| Residuals + layer norm | Standard in both |

## 7.3 Completely different

| # | Aspect | GraphGPS | Yours |
|---|---|---|---|
| 1 | **Topology** | Parallel branches merged **every layer** | **Sequential**: all local first, then fuse, then global **once** |
| 2 | **Interleaving** | Local/global mix L times; global feeds back into local | Global never feeds back; one-directional |
| 3 | **Positional encodings** | **Central** — LapPE/RWSE are half the recipe | **Absent** |
| 4 | **Heterogeneity** | Homogeneous; type-blind attention | Fully heterogeneous; even the *global* branch is type-aware |
| 5 | **Global attention scope** | All nodes ↔ all nodes | **Same-type pairs only** (Fix A) |
| 6 | **Depth fusion (JK)** | None — each layer overwrites the last | Retains `h¹–h⁴`, fuses per node and per task |
| 7 | **Readout** | One representation feeds the head | **Per-task readout depth** |
| 8 | **Frontier awareness** | No concept of data incompleteness | `is_frontier` nodes deliberately included |
| 9 | **Purpose of attention weights** | Mechanism only | Mechanism **and** persisted evidence |
| 10 | **Scalability** | Performer/BigBird for linear attention | Dense N², bounded by top-k |

## 7.4 The audit

**It is not an improvement of GraphGPS. It is not the same. It isn't GraphGPS-shaped at all — it's GraphTrans-shaped.**

GraphGPS was partly a *response* to GraphTrans, arguing interleaving beats sequencing. Your topology is GraphTrans's. So the accurate lineage statement is:

> *A GraphTrans-style sequential hybrid with a heterogeneous (HGT) local stage, JK-style depth fusion between the stages, and a type-constrained, frontier-aware global stage.*

## 7.5 Where yours is weaker — two real deficits

**1. No interleaving.** In GraphGPS a node can incorporate globally-gathered information and then propagate it locally next layer — global and local *compound*. In yours, global attention is a one-shot post-processor. This is a genuine expressiveness loss, and precisely the loss GraphGPS was designed to fix.

**2. No positional encodings.** Bare self-attention over embeddings has no idea where nodes sit in the graph — GraphGPS spends half its recipe fixing that. Your partial defence: T2's input is HGT's output, so position is already implicitly baked into the embeddings (GraphTrans's defence too). It works, but it's weaker than explicit PE. **Expect "why no LapPE/RWSE?" and have the answer ready:** at 5,000 nodes with typed structure doing most of the positional work, PE is plausible future work — say it, don't hope it isn't asked.

## 7.6 Where yours is stronger — with an asterisk

Rows 4–8: heterogeneity, type-constrained attention, depth fusion, per-task readout, frontier awareness. GraphGPS has none of them.

**The asterisk:** they're stronger **for your problem class** — typed graphs with a visibility frontier and targets at different depths. On a generic benchmark (molecules, citation networks), most are dead weight and GraphGPS wins comfortably.

## 7.7 Why yours is the right choice here

Both GraphGPS and GraphTrans are **homogeneous**. Fed your graph as-is, they'd have to flatten it — one node type, one edge type, `SUPPLIES` = `SHIPS_TO` = `PLACED_BY`. That's exactly the limitation your Stage-2 → Stage-3 ablation exists to demonstrate.

| Your requirement | GraphGPS | GraphTrans | Yours |
|---|---|---|---|
| 8 node / 10 edge types | after modification | after modification | native |
| Scarce labels | worst — global attention every layer | moderate | lightest |
| Targets at different depths | no | no | **yes** |
| Visibility frontier | no | no | **yes** |
| Explainability mandate | hardest — interleaving smears credit | stages separable | every stage persisted |
| Long-range mixing power | **best** | moderate | moderate |
| Positional encodings | **yes** | no | no |
| Off-the-shelf code | **yes** | partial | you build it |

**Steal the one thing worth stealing:** if T2 underperforms, a cheap RWSE positional encoding (random-walk features, computed once, a few extra input dims) is the first fix to try — GraphGPS's best idea at near-zero parameter cost, slotting into your input projection without touching anything else.

## 7.8 The honest cons of your design

1. **Not a new architecture.** All three components are off-the-shelf (HGT 2020, JK 2018, global attention 2021–22), and GraphGPS already templates local+global combination. A reviewer will read it as *GraphTrans with a heterogeneous local stage plus JK*. **Don't claim novelty you don't have** — claim a *novel configuration for a specific structural problem*, which is bulletproof.
2. **Specialisation is a trade, not an upgrade.** Strictly better on graphs like yours; strictly worse on complete, homogeneous, label-rich graphs.
3. **The supply-chain HGT space is crowded.** 2025–26 work already applies HGT to supply chain risk, and the frontier has moved to **temporal** models (TG-RRNet, DynSupplyNet). Your snapshot design is defensible for a prototype but expect the question.
4. **Statistical power may not support the comparison.** At ~500 positives, the 95% CI on AUC is ±0.024. Differences smaller than that are noise. Report bootstrap or DeLong CIs on every ablation row.
5. **Your genuine novelty isn't architectural.** It's **Claim B** — dyadic risk. Lead with that.

---

# PART 8 — RECOMMENDED CONFIGURATION

## Phase 1

| Decision | Value |
|---|---|
| Encoder | HGT, **L = 4**, retain `h¹–h⁴` — one layer of headroom above the deepest blanket readout, so the gate can express upward deviation |
| Depth prior | **Markov per-task readout** — delay ← `h²`, shortage ← `h³`, impact ← `h⁴` |
| Depth deviation | **Blend level L2** — three prior-anchored gates, `τ = 0.8`, `λ = 0.01` |
| Tier budget | `SUB_SUPPLIES` decayed by `α^tier`, α ≈ 0.2–0.3, via HGT's `μ` |
| Global attention | **Transformer 2, bounded candidate pool** |
| Depth evidence | **4-point L-sweep with confidence intervals** — not attention weights |
| Dyadic risk | Claim B in the Risk Intelligence Layer |
| **Cost** | **~641K params, ~2.03 GFLOP, 5 training runs** |

**Why `L = 4` rather than 3.** In a pure-Markov design `L = 3` suffices, because the global attention prediction head carries the long-range BOM reach. But once you blend, the gate learns a *deviation* from the blanket — and if the deepest prior is `h³` and the encoder stops at 3, the gate sits on the ceiling and can only ever detect *downward* deviation. **The residual would be censored at the top**, and upward deviation (hub suppliers, frontier nodes) is exactly the interesting case. One layer of headroom costs 154,122 parameters and buys an uncensored measurement.

**The rule:** `L = (deepest blanket readout) + 1`, capped at 4.

## Phase 2 — as evidence accumulates

| Trigger | Action |
|---|---|
| L2 gate weights look under-expressive | Move to **L2+** (concat-gate) — +18K params |
| Positives exceed ~2,000 **and** L2+ saturates | Consider **L3** (attention-only) — doubles inference |
| L-sweep still improving at `L = 4` | You're at the structural ceiling; report the residual as **one-sided** |

## What not to build

- **Transformer 1 in its specified 2-layer form (blend level L4)** — 45× the parameters and 181× the compute of L2 for the same capability
- **`L = 5+`** — reach is already 100% at L=4; nothing to gain
- **Unconstrained global attention** — undoes HGT's type separation
- **Depth attention as the Claim A instrument** — it measures usage, not sufficiency
- **A heavy gate under a strong prior, or a light gate under no prior** — the two knobs must move together (§6.5.1)

---

## Appendix — assumptions behind the numbers

Nothing here was measured; all figures are derived analytically from the specification.

- `d = 64`, `h = 4` attention heads. Neither is specified in the current documentation — that is itself a gap worth closing.
- 8 node types (Doc 6 §5), feature widths from Doc 10 §7.
- 10 Phase-1 meta-relations, counting `SHIPS_FROM` twice (it targets both Supplier and Factory).
- Graph scale: 5,000 nodes (NFR-02 ceiling), ~20,000 edges (average undirected degree ≈ 8), ~600 suppliers.
- HGT per-layer parameters: `4T·d²` (per-type K/Q/V/A) + `2R·d²/h` (per-relation attention and message) + per-relation prior + skip terms ≈ 154,122 at `d=64`.
- Transformer per-layer parameters: `4(d²+d)` for QKVO + `2·ff·d²` for the FFN + layer-norm terms.
- FLOPs counted as 2 × MACs. HGT split into a node-projection term and an edge term; T1 evaluated over 4 tokens per node across all 5,000 nodes; T2 over 600 suppliers with the quadratic attention term included.
- AUC confidence intervals via the Hanley–McNeil standard error at AUC = 0.80.
- k-hop reach uses uniform branching, which *understates* saturation on hub-heavy graphs.

Recompute once real dimensions and a real graph exist. The relative orderings are robust; the absolute values are not.

---

## Citations

- **HGT** — Hu, Dong, Wang & Sun (2020), *Heterogeneous Graph Transformer*
- **Jumping Knowledge** — Xu et al. (2018), *Representation Learning on Graphs with Jumping Knowledge Networks*
- **GraphSAGE** — Hamilton, Ying & Leskovec (2017)
- **GAT** — Veličković et al. (2018)
- **Over-smoothing** — Li, Han & Wu (2018); Oono & Suzuki (2020)
- **APPNP / PPRGo** — Klicpera et al. (2019); Bojchevski et al. (2020)
- **GraphGPS** — Rampášek et al. (2022), *Recipe for a General, Powerful, Scalable Graph Transformer*
- **GraphTrans** — Wu et al. (2021), *Representing Long-Range Context for Graph Neural Networks with Global Attention*
- **Graph Information Bottleneck / GSAT** — Wu et al. (2020); Miao et al. (2022)
- **Context-specific independence** — Boutilier et al. (1996)
- **Markov blanket feature selection** — Koller & Sahami (1996); Aliferis et al. (HITON/MMMB)
- **Preferred customer status** — Schiele, Calvi & Gibbert (2012); Steinle & Schiele (2008)
