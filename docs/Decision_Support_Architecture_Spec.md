# Decision-Support Architecture Specification

**Counterfactual Reasoning · Uncertainty Estimation · Explainable HADES**

> **Document class.** This is a *specification*, not an experimental report. It joins
> `docs/00_Benchmark_Specification.md` as an authoritative design reference. Nothing in it has
> been implemented or measured; every number quoted from prior work is cited to the report that
> measured it, and every number introduced here as a *target* is labelled as a target.
>
> **Status:** design complete, implementation not started.
> **Depends on:** `findings/evolution.md` (architecture record), `docs/00_Benchmark_Specification.md`
> (generator contract), `reports/phase7_training_results.md`, `reports/layer2_testing.md`,
> `reports/layer3_testing.md`.

---

## §0. Preface — the ground this specification stands on

### 0.1 How to read this document

Three features are specified independently and in full. Each follows the identical 24-section
structure, so §F1.19 (Feature 1's evaluation metrics) has the same meaning as §F3.19. Sections are
numbered `F<feature>.<section>` throughout.

The three features are **separable**: each can be built, evaluated and shipped without the other
two. They share infrastructure, and §4 (Cross-Cutting Concerns) specifies what that shared
infrastructure is and the order in which it should be built to avoid duplicated work. Where a
feature depends on another, the dependency is stated explicitly and an independent fallback is
given.

Sections that require it state a **position** rather than surveying options neutrally. Where a
position could reasonably go the other way, the alternative and the reason for rejecting it are
recorded, so that a future engineer disagreeing with a choice can see what it was traded against.

### 0.2 The Layer 2 naming correction, stated once

**HADES's Layer 2 is not adaptive.** It is a fixed-depth Markov Blanket readout: a zero-parameter,
per-task index-select into SHARE's retained layer stack —

    delay -> h^1,  shortage -> h^3,  impact -> h^4

— implemented as `layers[MARKOV_READOUT_DEPTH[task]][entity_type]` with no `nn.Parameter` anywhere
in the readout path (`ml/models/rgcn_attn_markov_encoder.py`).

It is called **"the Markov Blanket depth readout"** or **"the fixed-depth Layer 2 readout"**
throughout this document, never "Adaptive Markov Blanket." Per-node adaptive depth selection is a
**closed question, not an open one**: eight distinct gate designs across roughly 130 training runs
on two independently-built datasets never beat it, and the two most promising designs were shown to
be either mathematically incapable of deviating from the prior at any tested setting (Variant A:
prior gap 8.0 against a maximum correction of 2λ, with every λ ever tested 8–40x too small) or to
collapse into one global depth per random seed rather than genuine per-node adaptivity (Rung 5,
traced to the gate mean-pooling depth tokens before scoring). `findings/evolution.md` §3 has the
full account.

**No feature specified here may assume, require, or reintroduce per-node depth adaptivity.** All
three build on the fixed readout as a frozen, validated component. Where a feature needs
information from a depth other than its task's Markov depth — Feature 3 does — it reads that depth
*additionally and read-only*, and never changes which depth the prediction itself comes from.

### 0.3 The validated architecture these features build on

```
                          ┌────────────────────────────────────────────┐
   snapshot t0            │  LAYER 1 — SHARE                           │
   HeteroData             │  ml/models/rgcn_attn_encoder.py            │
   8 node types      ────▶│                                            │
   ~20 meta-relations     │  per-node-type input projection  lin_in    │
   (10 forward,           │  shared basis pool  rel_basis (B=10)       │
    ToUndirected)         │  per-relation coeffs rel_coeff             │
                          │      W_r = sum_b a_r[b] · V_b              │
                          │  per-node-type per-layer self-loop Linear  │
                          │  shared attention scorer att_msg, att_dst  │
                          │      one softmax per DESTINATION node,     │
                          │      over its ENTIRE incoming edge set,    │
                          │      across relations, not within them     │
                          │                                            │
                          │  emits h^0, h^1, h^2, h^3, h^4             │
                          └───────────────────┬────────────────────────┘
                                              │  (all five depths retained)
                          ┌───────────────────▼────────────────────────┐
                          │  LAYER 2 — Markov Blanket depth readout    │
                          │  ml/models/rgcn_attn_markov_encoder.py     │
                          │  ZERO parameters. Pure index-select.       │
                          │    z_delay    = h^1["Shipment"]            │
                          │    z_shortage = h^3["Product"]             │
                          │    z_impact   = h^4["Supplier"]            │
                          └───────────────────┬────────────────────────┘
                                              │
                          ┌───────────────────▼────────────────────────┐
                          │  PREDICTION HEADS — ml/models/heads.py     │
                          │  one per task, 2-layer MLP                 │
                          │    Linear(d,d) -> ReLU -> Linear(d,1)      │
                          │  raw logits; FocalLoss(gamma=2,            │
                          │  alpha = 1 - positive_rate) applies sigmoid│
                          └───────────────────┬────────────────────────┘
                                              │
                                     delay / shortage / impact
                                       (per-node probabilities)

   LAYER 3 — FROZEN. Not used by any feature in this document as a discovery
   mechanism. One capability is retained and IS reused, by Feature 3:
   calibrated attribution to OBSERVABLE causes (reports/layer3_testing.md §10).
```

**Production configuration**, as used by every result cited here:
`hidden=128`, `num_bases=10`, `num_layers=4`, `dropout=0.2`, focal `gamma=2`.
Task-to-entity mapping: `delay -> Shipment`, `shortage -> Product`, `impact -> Supplier`.

**Graph relations** (forward; `ToUndirected()` adds the reverse of each, giving the ~20
meta-relations SHARE's basis decomposition is sized for):

| # | relation | edge attributes |
|---|---|---|
| 1 | `Supplier -SUPPLIES-> Component` | — (primary ∪ active secondary from `component_suppliers`) |
| 2 | `Component -USED_IN-> Product` | `quantity_required` |
| 3 | `Product -STOCKED_AT-> Warehouse` | `stock_level`, `reorder_threshold` |
| 4 | `Product -MANUFACTURED_AT-> Factory` | `capacity_units_per_day` |
| 5 | `Shipment -SHIPS_FROM-> Supplier` | — |
| 6 | `Shipment -SHIPS_FROM-> Factory` | — |
| 7 | `Shipment -SHIPS_TO-> Warehouse` | days-to-ETA |
| 8 | `Shipment -FULFILLS-> Order` | — |
| 9 | `Order -ORDERED-> Product` | `quantity` |
| 10 | `Order -PLACED_BY-> Customer` | — |
| 11 | `Supplier -UPSTREAM_OF-> Supplier` | — (Mechanism J only; absent otherwise, never an empty relation) |

**Temporal protocol**: snapshots are monthly `t0`s (15 at mid scale, 40 at spec scale); the split
is **40/20/40 by snapshot count**, never random (`ml/data/loader.py::split_bundles`). Features are
as-of `t0`; labels are on true event time within `(t0, t0 + horizon]`.

### 0.4 What does not exist today, and must therefore be built

`docs/08_Backend_Design.md` states it plainly: **"There is no server, no API and no runtime service
in this repository."** HADES-Bench V2 is a generation pipeline plus an evaluation harness, driven
from the command line.

This has direct consequences for three of the required sections in every feature:

- **§F*.15 API Design** specifies a *new* service. It is not an extension of an existing API,
  because there is none. Each feature's API section therefore also states whether that feature
  *requires* the service or can be delivered as an offline artifact, and every feature is
  specified so that its research contribution is reachable **without** building the service.
- **§F*.16 Database Schema Changes** are additive DDL against `db/schema.sql`, which today is
  applied by `db/load_data.py` into PostgreSQL as an *optional* step in the pipeline
  (`docs/04_Application_Flow.md` §4: "Load *(optional but recommended)*"). No feature may make the
  database a hard dependency of its training or evaluation path; the database is a serving and
  audit surface, and the file system remains the system of record for experiments.
- **§F*.17 Dashboard / UI Design** specifies a *new* surface. It is scoped as a thin read-only
  view over the API, and every feature states its non-UI fallback (a static HTML or Markdown
  artifact produced by an analysis script, exactly as `ml/analyze_hypothesis.py` produces §10's
  tables today).

One useful pre-existing hook: `risk_scores` already carries a `confidence NUMERIC(5,4)` column and
a `scoring_method` enum with a `gnn_native` value, both unused legacy from V1. Feature 2 adopts
them rather than adding parallel columns (§F2.16).

### 0.5 Two evidentiary disciplines that are mandatory in all three features

These are not methodological preferences. They are the two findings that most often invalidate a
plausible-looking result on this benchmark, and every feature below is required to apply both.

**(a) The reproduction floor.** `reports/layer3_testing.md` §4 measured how far two *identically
configured* training runs land apart — same architecture, same dataset, same model seed, same
budget, differing only in process environment (thread count, and therefore CPU float reduction
order; `scatter`/`index_add` in message passing is not order-deterministic):

| task | mean signed | mean abs | max abs | std |
|---|---|---|---|---|
| delay | +0.0002 | 0.0024 | **0.0121** | 0.0037 |
| shortage | −0.0001 | 0.0007 | **0.0031** | 0.0010 |
| impact | −0.0009 | 0.0026 | **0.0082** | 0.0031 |

*(CPU, 16 identically-configured pairs. `reports/layer3_testing.md` §9.2 re-measured it on MPS and
§10.3.1 measured it for a non-GNN head, where it came back at exactly 0.0000 — the head was
bit-deterministic, so the operative floor there became the model-init-seed spread instead. The
lesson generalises: **measure the floor for the specific estimator, on the specific device, before
believing any delta from it.**)*

The rule this imposes: **any claimed effect must clear the max-abs floor for its own estimator on
its own device, measured in the same session.** Sub-floor deltas are not evidence, and — the
sharper form of the rule, from the same section — **sign consistency across dataset seeds does not
rescue a sub-floor delta**, because the floor is a property of the training process that all seeds
share. Five same-signed sub-floor deltas are five draws from one biased-by-noise process, not five
independent confirmations.

**(b) Dataset-seed variance dominates model-seed variance.**
`reports/phase7_training_results.md` §5.6:

| task | model-seed std | dataset-seed std | ratio |
|---|---|---|---|
| delay | 0.0037 | **0.0376** | **10.3x** |
| shortage | 0.0019 | 0.0090 | 4.8x |
| impact | 0.0102 | 0.0128 | 1.3x |

The variance that matters most on this benchmark is **which world was generated**, not how the
model was initialised. Every claim must be averaged over generated worlds, not over model
initialisations. This has a specific and severe consequence for Feature 2, where most published
uncertainty methods estimate exactly the smaller of these two quantities; §F2.3 and §F2.22 treat
it as the central design problem rather than a caveat.

### 0.6 One citation correction, recorded so it is not propagated

The brief for this document cites `reports/layer2_testing.md` §4 for the reproduction-floor
methodology. That section is *"Cost — the gate is not free, and the estimate was re-checked"*, and
concerns per-run wall-clock, not the noise floor. The reproduction-floor methodology is
`reports/layer3_testing.md` **§4**, extended in **§9.2** (retrieval metrics, MPS) and **§10.3.1**
(a bit-deterministic estimator, where the identical-config floor was exactly zero and the
model-seed floor became operative). This document cites the layer3 sections throughout. The
correction is noted here rather than silently applied, because an authoritative reference that
quietly disagrees with its own brief is worse than one that says why.

### 0.7 Notation

| symbol | meaning |
|---|---|
| `G_t = (V, E, X_t)` | the heterogeneous snapshot graph as of `t0 = t`; `V` typed nodes, `E` typed edges, `X_t` as-of node features |
| `h^l_v` | SHARE's representation of node `v` after `l` rounds of message passing, `l ∈ {0..4}` |
| `d(τ)` | the Markov depth for task `τ`: `d(delay)=1`, `d(shortage)=3`, `d(impact)=4` |
| `z^τ_v = h^{d(τ)}_v` | the readout embedding for task `τ` at node `v` |
| `f_τ` | task `τ`'s prediction head (2-layer MLP), `ŷ^τ_v = σ(f_τ(z^τ_v))` |
| `Φ` | the frozen composite `G_t ↦ {ŷ^τ_v}`: SHARE + Markov readout + heads |
| `a` | an **intervention** (Feature 1), a structural edit to the world |
| `G_t^{do(a)}` | the snapshot graph after applying `a` |
| `Y^{do(a)}` | the outcome under intervention `a` — *simulated* when produced by the generator, *predicted* when produced by a model |
| `θ` | model parameters; `s_m` model-init seed; `s_d` dataset seed |
| `𝔈` | an ensemble; `\|𝔈\|` its member count |
| `ECE` | Expected Calibration Error, equal-count bins (§F2.19) |
| `φ_j` | attribution weight assigned to explanatory factor `j` (Feature 3) |

---

# PART I — FEATURE 1: COUNTERFACTUAL REASONING

*"What would happen if the supply chain changed?"*

---

## F1.1 Motivation

Every prediction HADES currently produces is a statement about a world that already exists. A
planner reading `impact_score = 0.87` for a supplier learns that this supplier is likely to cause
downstream damage in the next horizon. What the planner actually needs to decide is **what to do
about it** — dual-source the component, hold more inventory, qualify a replacement, switch the
logistics provider — and none of those decisions is answerable by a risk score. They are
answerable only by comparing outcomes across worlds: the one that exists, and the one that would
exist after the change.

This is the difference between a *monitoring* system and a *decision-support* system, and it is
the reason this feature is specified first. The three levers a supply-chain planner actually
controls are structural (who supplies what, through which route, with what redundancy) and
inventory-level. A model that scores risk but cannot evaluate a structural change leaves the entire
decision to the human, and gives them no more than a ranked list of things to worry about.

There is also a project-specific reason this feature is worth building *here* rather than
anywhere else, and it is the strongest asset this benchmark has. **HADES owns its
data-generating process.** `db/generate_dataset.py` is a deterministic simulator with dedicated
per-mechanism RNG streams, byte-identical reproducibility from `(seed, config)`, and a documented
causal stress model. That means an intervention can be **actually executed** — the world rebuilt
with a supplier removed, the simulation re-run, the true outcomes observed — and the model's
counterfactual prediction scored against a real interventional outcome rather than a proxy. Almost
no published counterfactual-GNN work has this. It is the difference between validating a
counterfactual method and merely demonstrating one.

## F1.2 Problem Statement

Given a snapshot `G_t` and an intervention `a` drawn from a defined action space `A`, predict the
post-intervention outcome distribution

    Ŷ^{do(a)} = { ŷ^τ_v : τ ∈ {delay, shortage, impact}, v ∈ V_τ }

over the same horizon the factual prediction covers, together with the **effect**

    Δ^τ_v(a) = ŷ^τ_v(do(a)) − ŷ^τ_v(factual)

and do so accurately enough, and with honest enough uncertainty, that a planner can rank candidate
interventions and act on the ranking.

Three properties distinguish this from ordinary prediction and each one drives a design decision
below:

1. **The input graph changes.** `a` edits `E` (and sometimes `V` and `X`). SHARE must run over an
   edited graph, and §F1.9 establishes that it can do so unmodified — but also what that does and
   does not buy.
2. **The training distribution does not contain the edited world.** Every snapshot the model has
   ever seen is a factual observation. This is the associational/interventional gap, and §F1.4 and
   §F1.9 take an explicit position on it rather than assuming a graph edit is sufficient.
3. **The generator's world has feedback that a single forward pass cannot represent.** Removing a
   supplier changes shipment schedules, which change inventory trajectories, which change the
   time-causal agent loop's reorder decisions (`docs/08_Backend_Design.md` §4), which change
   shortage outcomes. A GNN forward pass on an edited snapshot propagates information across the
   graph but not *forward through time*. §F1.21 treats this as a first-class failure mode, not a
   footnote.

**Explicitly out of scope.** Optimal intervention *search* (choosing the best `a` from a large
combinatorial space) is not specified here; this feature scores a supplied candidate set. §F1.24
sketches the extension. Interventions that change the generator's *parameters* rather than its
*structure* (e.g. globally raising `alpha`) are also out of scope: they are not actions a planner
can take.

## F1.3 Research Gap

**In the counterfactual-GNN literature.** Methods for counterfactual reasoning on graphs
(CF-GNNExplainer and its descendants, graph counterfactual generation, structural-causal-model
approaches to relational data) are overwhelmingly evaluated in one of two ways: against a
*proxy* — did the prediction change in the direction a human would expect — or against a
*simulator built specifically for the paper*, whose fidelity to anything is unestablished. The
recurring criticism is that a counterfactual predictor is validated on the same associational data
that produced it, which cannot distinguish "correctly predicted the intervention's effect" from
"produced a plausible number."

**What is different here.** HADES-Bench's generator is not a proxy and not built for this feature:
it is the same simulator that produced every training label the model was fitted on, with a
documented and mechanistically-isolated causal structure. Re-running it under an intervention
produces the **actual** counterfactual outcome under the same data-generating process the factual
data came from. This makes it possible to measure, rather than argue about, the size of the
associational-to-interventional gap for a standard supervised GNN — which is the gap most of the
literature asserts exists without quantifying.

**The gap this feature fills** is therefore twofold: a counterfactual capability for HADES, and a
*measurement* — on a benchmark with genuine interventional ground truth — of how far a
conventionally-trained graph model's answer to "what if" is from the truth, and how much of that
distance is closed by training on interventional pairs.

## F1.4 Scientific Contribution

1. **Interventional ground truth for a graph-learning benchmark.** A protocol
   (§F1.11, §F1.19) for producing `(G_t, a, Y^{do(a)})` triples by re-executing a deterministic
   generator under structural edits, with **common random numbers** (§F1.8.4) so that the factual
   and counterfactual worlds differ *only* by the intervention and not by RNG divergence. This is
   the technical core of the contribution and the part most easily got wrong.

2. **A measured answer to "can a supervised GNN answer an intervention question?"** The
   specification mandates that the associational baseline (§F1.9, Tier 1) be built and evaluated
   *alongside* the counterfactual-trained model (Tier 2), against the same re-simulated truth. The
   deliverable is a number: how wrong graph-editing-plus-forward-pass is, per task, per
   intervention class.

3. **An intervention taxonomy grounded in a real mechanism set.** The action space (§F1.8.1) maps
   one-to-one onto structures the generator actually produces — `component_suppliers` edges,
   `supplier_upstream` tiers, inventory levels, carrier assignment — rather than onto abstract
   "node deletion" and "edge addition," which makes every intervention executable as a real
   re-simulation.

4. **A negative-result-resistant evaluation.** The reproduction-floor discipline (§0.5a) applied to
   counterfactual error, so that a claimed improvement from counterfactual training must clear the
   noise of the training process itself.

## F1.5 System Architecture

The feature is a **new module over a frozen backbone**, plus a **new offline data-generation
pipeline**. It adds no changes to SHARE and none to the Markov readout.

Two tiers, both specified, both built, and evaluated against each other:

- **Tier 1 — Associational Graph Edit (`CF-Assoc`).** Apply `a` to `G_t`, run frozen `Φ`, read the
  difference. Zero new parameters, zero new training. **This is a control, not the deliverable.**
  It exists to quantify what a conventionally-trained model does when handed an edited graph, and
  is labelled as associational everywhere it surfaces.
- **Tier 2 — Counterfactual Effect Head (`CF-Head`).** A new head trained on generator-produced
  interventional triples to predict the *effect* `Δ^τ_v(a)` directly, reading SHARE's frozen
  embeddings of both the factual and edited graphs plus an explicit encoding of the intervention.

```
 ┌─────────────────────────── OFFLINE (once per dataset config) ──────────────────────────┐
 │                                                                                        │
 │  db/generate_dataset.py --variant V --seed s            ──▶  FACTUAL world  W_0        │
 │            │                                                  (CSV + labels)           │
 │            │  --intervene <spec>   (NEW: §F1.8.2)                                      │
 │            ├──────────────────────▶ COUNTERFACTUAL world W_a1  (CSV + labels)          │
 │            ├──────────────────────▶ COUNTERFACTUAL world W_a2                          │
 │            └──────────────────────▶ ...                                                │
 │                                                                                        │
 │  Common random numbers (§F1.8.4) guarantee W_0 and W_a differ ONLY by the intervention │
 │                                                                                        │
 │            ▼                                                                           │
 │   INTERVENTION CORPUS   (G_t, a, Y_factual, Y^{do(a)})   -> out/cf_corpus/             │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
 ┌──────────────────────────────────────▼─────────────────── ONLINE (training + serving) ─┐
 │                                                                                        │
 │   G_t ──▶ [ SHARE (FROZEN) ] ──▶ h^0..h^4 ──▶ [ Markov readout ] ──▶ z^τ  ──▶ ŷ^τ      │
 │             │                                                                          │
 │   a ──▶ [ Graph Editor ] ──▶ G_t^{do(a)} ──▶ [ SHARE (FROZEN, same weights) ]          │
 │             │                                    │                                     │
 │             │                                    ▼                                     │
 │             │                              h̃^0..h̃^4 ──▶ [ Markov readout ] ──▶ z̃^τ    │
 │             │                                                     │                    │
 │             ▼                                                     ▼                    │
 │      [ Intervention Encoder ψ(a) ]  ────────────────▶  [ CF-Head g_τ ]  ──▶ Δ̂^τ_v      │
 │             (NEW, small)                                   (NEW)                       │
 │                                                                                        │
 │   Tier 1 output (control):  Δ̂^τ_v = σ(f_τ(z̃^τ_v)) − σ(f_τ(z^τ_v))                     │
 │   Tier 2 output (deliverable): Δ̂^τ_v = g_τ( z^τ_v , z̃^τ_v , ψ(a) , ρ_v(a) )           │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

`ρ_v(a)` is a small vector of **structural reach features** for node `v` relative to intervention
`a` — graph distance to the nearest intervened node, whether `v` lies within the Markov depth of
any intervened node, and how many intervened nodes are within that depth. It is computed on the
edited graph by breadth-first search, costs nothing to train, and gives the head an explicit signal
for the single strongest prior available: *an intervention cannot affect a node it cannot reach in
`d(τ)` hops*. §F1.8.5 defines it.

## F1.6 Component Diagram (ASCII)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ EXISTING, UNMODIFIED                                                          │
│                                                                               │
│  ┌───────────────────┐   ┌───────────────────────┐   ┌────────────────────┐   │
│  │ db/generate_      │   │ ml/models/            │   │ ml/models/heads.py │   │
│  │  dataset.py       │   │  rgcn_attn_encoder.py │   │  PredictionHead x3 │   │
│  │  (extended, §F1.8)│   │  SHARE — FROZEN       │   │  FROZEN            │   │
│  └───────────────────┘   └───────────────────────┘   └────────────────────┘   │
│  ┌───────────────────┐   ┌───────────────────────┐                            │
│  │ ml/data/loader.py │   │ ml/models/rgcn_attn_  │                            │
│  │  (extended, §F1.8)│   │  markov_encoder.py    │                            │
│  │                   │   │  Markov readout       │                            │
│  └───────────────────┘   │  ZERO PARAMS, FROZEN  │                            │
│                          └───────────────────────┘                            │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼─────────────────────────────────────────┐
│ NEW COMPONENTS                                                                │
│                                                                               │
│  ┌──────────────────────────┐  produces the corpus; pure data, no model       │
│  │ cf_corpus_builder        │  - resolves an intervention spec                │
│  │  (ml/cf/corpus.py)       │  - invokes the generator twice under CRN        │
│  │                          │  - aligns entities across the two worlds        │
│  │                          │  - emits (G_t, a, Y_f, Y_cf) shards             │
│  └────────────┬─────────────┘                                                 │
│               │                                                               │
│  ┌────────────▼─────────────┐  ┌──────────────────────────┐                   │
│  │ GraphEditor              │  │ InterventionEncoder ψ    │                   │
│  │  (ml/cf/edit.py)         │  │  (ml/cf/encode.py)       │                   │
│  │  - typed edit ops        │  │  - action type embedding │                   │
│  │  - validity checks       │  │  - target node pooling   │                   │
│  │  - feature recomputation │  │  - magnitude scalars     │                   │
│  │  - returns HeteroData    │  │  -> R^{d_a}, d_a = 32    │                   │
│  └────────────┬─────────────┘  └────────────┬─────────────┘                   │
│               │                             │                                 │
│  ┌────────────▼─────────────────────────────▼─────────────┐                   │
│  │ ReachFeatures rho_v(a)   (ml/cf/reach.py)              │                   │
│  │  BFS on edited graph; 5 scalars per (node, action)     │                   │
│  └────────────┬───────────────────────────────────────────┘                   │
│               │                                                               │
│  ┌────────────▼───────────────────────────────────────────┐                   │
│  │ CF-Head g_tau   (ml/cf/head.py)                        │                   │
│  │  one per task; 3-layer MLP; predicts the EFFECT        │                   │
│  │  plus a gate that can output exactly zero (§F1.10)     │                   │
│  └────────────┬───────────────────────────────────────────┘                   │
│               │                                                               │
│  ┌────────────▼───────────────────────────────────────────┐                   │
│  │ CF evaluation harness (ml/cf/evaluate.py)              │                   │
│  │  - re-simulation scoring (§F1.19)                      │                   │
│  │  - reproduction floor for CF error (§F1.19.6)          │                   │
│  │  - Tier 1 vs Tier 2 paired comparison                  │                   │
│  └────────────────────────────────────────────────────────┘                   │
│                                                                               │
│  ┌────────────────────────────────────────────────────────┐                   │
│  │ OPTIONAL SERVING: cf_service (API §F1.15), cf tables    │                  │
│  │ (§F1.16), dashboard panel (§F1.17)                      │                  │
│  └────────────────────────────────────────────────────────┘                   │
└───────────────────────────────────────────────────────────────────────────────┘
```

## F1.7 Data Flow Diagram (ASCII)

**Offline corpus construction** (run once per `(variant, dataset seed, intervention set)`):

```
  intervention spec file            generator config
  (JSON, §F1.8.2)                   (variant, seed)
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
        ┌───────────────────────────────┐
        │ resolve targets against the   │   target ids must exist in the
        │ FACTUAL world's entity ids    │   factual world, else reject
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐        ┌────────────────────────────┐
        │ run generator, no intervention│        │ run generator, intervention│
        │ CRN streams pinned (§F1.8.4)  │        │ applied at construction    │
        │        -> W_0                 │        │ CRN streams pinned         │
        └───────────────┬───────────────┘        │        -> W_a              │
                        │                        └──────────────┬─────────────┘
                        └──────────────┬────────────────────────┘
                                       ▼
                     ┌──────────────────────────────────┐
                     │ ENTITY ALIGNMENT (§F1.8.6)       │
                     │ UUIDv5 ids are stable under      │
                     │ intervention for surviving       │
                     │ entities; removed/added entities │
                     │ recorded explicitly              │
                     └────────────────┬─────────────────┘
                                      ▼
                     ┌──────────────────────────────────┐
                     │ per snapshot t0, per task tau:   │
                     │   Y_f[t0,tau,v], Y_cf[t0,tau,v]  │
                     │   effect  = Y_cf - Y_f           │
                     │ emit shard  out/cf_corpus/...    │
                     └────────────────┬─────────────────┘
                                      ▼
                          INTERVENTION CORPUS on disk
```

**Training** (Tier 2):

```
 corpus shard ──▶ load factual snapshot G_t  ──▶ SHARE (frozen, no_grad) ──▶ h^0..h^4
        │                                                                      │
        │                                                          Markov ──▶ z^tau
        │
        ├──▶ GraphEditor(a) ──▶ G_t^{do(a)} ──▶ SHARE (frozen, no_grad) ──▶ h~^0..h~^4
        │                                                                      │
        │                                                          Markov ──▶ z~^tau
        │
        ├──▶ InterventionEncoder ──▶ psi(a) in R^32
        │
        └──▶ ReachFeatures ──▶ rho_v(a) in R^5
                        │
                        ▼
        CF-Head g_tau( z^tau_v , z~^tau_v , psi(a) , rho_v(a) )  ──▶  Delta^hat
                        │
                        ▼
        loss vs. TRUE effect from re-simulation (§F1.10)
                        │
                        ▼
        gradients flow ONLY into g_tau, psi, and the reach projection.
        SHARE, the Markov readout and the three prediction heads are frozen.
```

**Inference**: identical, minus the loss; the two SHARE passes can be batched together, and the
factual pass is shared across all candidate interventions for the same `t0` (§F1.18).

## F1.8 Mathematical Formulation

### F1.8.1 The action space

An intervention `a` is a typed, parameterised structural edit. Six classes, each chosen because
the generator produces the structure it edits:

| id | action | formal edit | generator structure touched |
|---|---|---|---|
| `A1` | **remove supplier** `s` | delete `s` from `V_Supplier`; delete all incident edges; every component whose only source was `s` becomes unsourced | `components.supplier_id`, `component_suppliers` |
| `A2` | **add backup source** `(c, s')` | add `component_suppliers` row `(c, s')`, hence edge `s' -SUPPLIES-> c` | `component_suppliers`, `coparents` |
| `A3` | **substitute supplier** `(c, s_old -> s_new)` | `A1`-on-edge then `A2`; the incumbent's edge to `c` is deactivated | the `CS_REWIRES` "swap" kind, Mechanism C |
| `A4` | **add dual sourcing** `(c)` | choose `s'` by policy (cheapest qualified / most reliable / most distant), then `A2` | `dual_source_fraction` |
| `A5` | **increase inventory** `(p, w, Δq)` | `inventory.stock_level += Δq` for `(p, w)`; optionally raise `reorder_threshold` | `inventory`, and the agent loop's reorder trigger |
| `A6` | **change logistics provider** `(s, carrier)` | reassign the carrier attribute governing `s`'s shipment lead-time distribution | `carrier_performance_snapshots`, lead-time draw |

`A1`–`A4` are edge-set edits; `A5` is a feature edit; `A6` is a parameter edit on a *specific
entity* (and is therefore in scope, unlike a global parameter change). Composite interventions are
sequences `a = (a_1, ..., a_k)` applied in order; the corpus includes `k ≤ 2` composites so that
§F1.20's additivity ablation is possible.

### F1.8.2 Intervention specification format

A spec is JSON, resolvable against a factual world, and is the single artifact shared by the
generator extension, the graph editor and the API:

```
{
  "id": "a_0417",
  "base": {"variant": "B", "seed": 42, "config": "db/config_mid.json"},
  "ops": [
    {"type": "A1", "target": {"entity": "supplier", "id": "<uuid>"}},
    {"type": "A5", "target": {"entity": "inventory", "product_id": "<uuid>",
                              "warehouse_id": "<uuid>"}, "delta_units": 500}
  ],
  "effective_from": "2024-01-01T00:00:00Z",
  "policy": null
}
```

`effective_from` is pinned to `T_START` for the corpus (§F1.8.3 explains why), and is retained in
the schema so that mid-timeline interventions become expressible later without a format change.

### F1.8.3 Where the intervention is applied in generator time

**Position: interventions are applied at world-construction time, before the simulation loop
runs, for the corpus used to train and evaluate this feature.**

The alternative — applying an intervention at some `t0` in the middle of the timeline and
simulating forward from there — is what a deployed planner actually wants, and it is *more*
realistic. It is deliberately not the corpus design, for a reason specific to this generator: the
world-construction phase (suppliers → components → hidden structure → tiers → shocks →
multi-sourcing → rewiring) happens once, top to bottom, before any simulation, and the stress model
and agent loop then read that structure at every timestep. Injecting a structural change mid-loop
requires the generator to support a *dynamic* structural edit, which today exists only for
Mechanism C's pre-scheduled rewires. Building a general mid-timeline intervention facility is a
substantially larger generator change than this feature needs.

The consequence is stated plainly and carried into §F1.22: **the corpus answers "what would this
world have looked like had it always been different", not "what happens if I change this world
now."** These coincide only when the intervention's effect reaches steady state within the
timeline. For `A1`–`A4` (structural sourcing changes) the difference is small after the 182-day
warm-up, because sourcing structure is static in the factual world too. For `A5` (inventory) it is
**large** — a one-time stock injection at `T_START` is almost entirely washed out by the weekly
inventory walk long before the first `t0`, whereas an injection at `t0` is exactly what a planner
means. §F1.22 therefore restricts `A5` claims, and §F1.24 specifies the generator change that
lifts the restriction.

### F1.8.4 Common random numbers — the technical core

Naively re-running the generator with a supplier removed produces a world that differs from the
factual world **everywhere**, not just at the intervention. The reason is RNG consumption: removing
a supplier changes how many draws the construction loop makes, which shifts every subsequent draw
in the stream, which re-randomises component assignment, hidden-group membership, idiosyncratic
outage windows and every shipment in the simulation. The measured "effect" would then be dominated
by world-level divergence — precisely the 10.3x dataset-seed variance of §0.5b — and would be
uninterpretable.

The generator already has the mechanism needed to prevent this, and it is one of its best
properties: **mechanism isolation**, in which each mechanism draws from a dedicated RNG stream so
that enabling it consumes zero draws from the main simulation (`docs/08_Backend_Design.md` §2). The
intervention facility must extend the same pattern.

Requirements on the generator extension:

1. **A dedicated intervention RNG stream.** Any randomness the intervention itself needs (e.g.
   `A4`'s policy choice of a backup supplier, `A6`'s new lead-time draws) comes from
   `random.Random(0x1470 + CFG.seed)`, never from the main stream.
2. **Draw-count invariance in construction.** Applying an intervention must not change the number
   of draws taken from the main stream during world construction. Where an intervention removes an
   entity, the entity is **constructed and then marked as intervened**, not skipped — so the draws
   that built it still occur. The removal takes effect at edge-assembly and simulation time.
3. **Draw-count invariance in simulation, where achievable.** Per-shipment and per-timestep draws
   must be keyed by entity id and timestep, not consumed sequentially from a shared stream. Where
   the current implementation consumes sequentially, the intervention facility must convert those
   call sites to keyed streams (`random.Random(hash((stream_id, entity_id, t)))` or equivalent).
   **This is the largest single implementation task in Feature 1 and it must be verified, not
   assumed** — see the null-intervention test below.
4. **The null-intervention test, mandatory.** Running the generator with an intervention whose op
   list is empty must produce output **byte-identical** to running it without the intervention flag
   at all, verified by the existing compressed-byte diff
   (`run_benchmark.py --verify-determinism` methodology). If the null intervention perturbs a single
   byte, common random numbers are not established and no counterfactual number produced by the
   corpus means anything. This test gates the entire feature.
5. **The locality test, mandatory.** For a single-supplier `A1` intervention, the set of entities
   whose simulated trajectories differ between `W_0` and `W_a` must be *contained in* the set
   reachable from the intervened supplier through the causal structure. A difference appearing at
   an unreachable entity is proof of RNG leakage. The test reports the reachable-set size and the
   differing-set size; the differing set being a strict subset is the pass condition.

Formally, writing `ω` for the generator's random state and `W(ω, a)` for the produced world, CRN
requires that `W(ω, ∅)` and `W(ω, a)` be coupled on the same `ω`, so that

    Y^{do(a)} − Y^{factual}  =  [ W(ω, a) − W(ω, ∅) ]  evaluated at the SAME ω

is a within-world contrast rather than a between-world one. This is the variance-reduction
technique that makes an effect measurable at all at this benchmark's noise levels: without it the
per-intervention effect (typically a few points of AUC on a handful of nodes) sits far below the
world-to-world spread.

### F1.8.5 Reach features

For node `v`, task `τ`, intervention `a` with intervened node set `T(a)`:

    rho_v(a) = [ min_{u in T(a)} dist(u, v)  clipped at 6,
                 1[ min dist <= d(tau) ],
                 |{ u in T(a) : dist(u,v) <= d(tau) }|,
                 |T(a)|,
                 1[ v in T(a) ] ]

`dist` is undirected hop distance on the edited graph (SHARE runs on the `ToUndirected` graph, so
undirected distance is the right notion of reach). The second component is the load-bearing one:
**message passing at depth `d(τ)` cannot transmit any information from `u` to `v` if
`dist(u,v) > d(τ)`**, so the true effect at such a node is exactly zero *as far as the model can
possibly know*. §F1.10's gate uses this to make the zero prediction exactly representable, and
§F1.19.3 uses it to partition the evaluation into reachable and unreachable nodes — a partition that
turns out to matter, because a large majority of nodes are unreachable from any single intervention
and pooling them in would let a model score well by predicting zero everywhere.

### F1.8.6 Entity alignment across worlds

The generator assigns primary keys as **UUIDv5 from stable keys, never random**
(`docs/08_Backend_Design.md` §3). This is what makes alignment tractable: an entity constructed
from the same stable key in both worlds has the same id in both. Alignment is therefore an id join,
with three explicitly recorded categories:

- **surviving** — present in both `W_0` and `W_a`; effect is defined and enters the corpus.
- **removed** — present in `W_0`, absent in `W_a` (an `A1` target and any entity whose existence
  depended on it). Effect is undefined; recorded with a `removed` flag and **excluded from the
  regression loss**, since predicting an outcome for a node that no longer exists is not a
  meaningful task. Their disappearance is itself a prediction target for a later extension
  (§F1.24).
- **added** — absent in `W_0`, present in `W_a` (`A2`/`A4` may create `component_suppliers` rows;
  no new *nodes* are created by the specified action space, so this set is empty for `A1`–`A6` and
  the category exists for forward compatibility).

Alignment must be verified, not assumed: the corpus builder asserts that the surviving-entity id
sets match exactly and that node feature vectors for entities *outside* the intervention's reach
are identical between worlds at `t0` — a second, cheaper form of the locality test.

## F1.9 Neural Network Changes

### F1.9.1 What is reused, unmodified

| component | reuse | note |
|---|---|---|
| **SHARE** (`rgcn_attn_encoder.py`) | **unmodified, frozen** | runs twice per training example (factual, edited). No architectural change is required to run it on an edited graph — see §F1.9.3. |
| **Markov Blanket depth readout** | **unmodified, frozen** | the edited graph's embeddings are read at the *same* per-task depths. No adaptivity is introduced. |
| **Prediction heads** (`heads.py`) | **unmodified, frozen** | used to produce the Tier 1 control and the factual baseline probability that Tier 2's effect is added to. |
| **Loader** (`ml/data/loader.py`) | **extended, not modified in behaviour** | a new entry point builds a `HeteroData` from an *edited* world; the existing snapshot path is untouched and its outputs remain byte-comparable. |
| `verify_no_hidden_state()` | **reused as-is** | must continue to return empty on every counterfactual world; an intervention must not become a channel for privileged state to enter features. |

### F1.9.2 What is new

| component | parameters | shape |
|---|---|---|
| `InterventionEncoder ψ` | action-type embedding `6 × 32`, target pooling `Linear(128, 32)`, magnitude MLP `Linear(4, 32)`, fusion `Linear(96, 32)` | ≈ 8.4k |
| `ReachProjection` | `Linear(5, 16) → ReLU` | ≈ 0.1k |
| `CF-Head g_τ` (×3) | input `128 + 128 + 32 + 16 = 304`; `Linear(304,128) → ReLU → Dropout → Linear(128,64) → ReLU → Linear(64,2)` | ≈ 48k each |
| **total new** | | **≈ 153k**, against SHARE's ~0.8M — under 20% added, and none of it in the backbone |

The head's two outputs are `(δ̂, g)`: a raw effect and a gate logit. §F1.10 explains why.

### F1.9.3 Does SHARE need to change to run on edited graphs? — No, and here is why

SHARE consumes `x_dict` and `edge_index_dict`. Its parameters are indexed by *node type* and
*relation type*, never by node identity or by edge count:

- `lin_in` is per node type;
- `rel_basis`/`rel_coeff` produce `W_r` per relation type, shared across layers;
- self-loops are per node type per layer;
- `att_msg`/`att_dst` are shared across all relations and node types.

Adding or deleting edges therefore changes only which messages are constructed and how the
per-destination softmax normalises, not which parameters apply. Deleting a node is representable as
deleting its rows and all incident edges. **No architectural change is required.**

Two mechanical cautions that must be handled by the editor, not by SHARE:

1. **Isolated nodes.** A node with no incoming edges after an edit receives only its self-loop
   contribution. That is well-defined and is the correct behaviour, but it is a distribution shift:
   the factual training set contains few such nodes. §F1.21 lists it as a failure mode and
   §F1.19.3 partitions the evaluation so it cannot be hidden.
2. **Empty relations.** If an edit removes the last edge of a relation type, the relation must be
   *dropped from the dict*, not passed as an empty tensor — the loader's existing convention
   ("absent for variants without J — no empty relation") must be preserved, or the softmax
   normalisation will see an empty group.

### F1.9.4 Position — can SHARE + Markov, trained as supervised predictors, answer an intervention question?

**Position: no, not reliably, and the specification therefore requires Tier 2. But the claim is
made falsifiable rather than asserted: Tier 1 is built and measured, and if it matches Tier 2
within the reproduction floor, Tier 2 should be dropped.**

The argument, in four parts, each grounded in this project rather than in general causal-inference
principle:

**(i) The model estimates `P(Y | X)`, and the intervention asks for `P(Y | do(X))`.** SHARE is
fitted by focal loss on observational snapshots. Nothing in that objective distinguishes a feature
that *causes* an outcome from one that merely *co-occurs* with it. Editing the graph and re-running
the forward pass produces `P(Y | X = edited)` — the model's belief about a world in which it
*observed* this graph, which is a different quantity from the world in which someone *made* this
graph so. The two coincide only when there is no confounding between the edited structure and the
outcome, and in this generator there certainly is: hidden-parent group membership, base-pool
membership (`H_PORT`/`H_TRUCK`/`H_CUSTOMS`, covering 50% of suppliers at mid scale) and resilience
all influence both which suppliers look risky and what happens downstream, and none of them is
observable.

**(ii) The training distribution contains no edited worlds at all.** Every graph SHARE has seen is
a factual snapshot generated by an unmodified construction phase. An `A1` edit produces
configurations — an unsourced component, a supplier with no `SUPPLIES` edges — that occur rarely or
never in training. This is straightforward covariate shift, and the model has no calibration
against it.

**(iii) This project has already measured what happens when a model is asked about structure it was
not trained to represent.** The entire Layer 3 arc is that measurement. The most direct instance:
under privileged contrastive supervision, the representation achieved Recall@64 at 10.9x chance on
the groups it was shown and **chance** on held-out groups from the identical generative process
(`findings/evolution.md` §4.3) — it had memorised specific identities, not learned a transferable
relation. A model that memorises identity-specific structure is exactly the model whose response to
"remove this specific supplier" is least trustworthy, because the removal invalidates the memorised
association without the model knowing it should.

**(iv) A one-shot forward pass cannot represent the generator's temporal feedback.** The generator's
outcome for `shortage` depends on an inventory trajectory that depends on the agent loop's reorder
decisions that depend on recorded delivery outcomes. Removing a supplier changes deliveries →
changes reorder decisions → changes inventory → changes shortage. SHARE propagates information
across the graph at a single `t0`; it has no mechanism at all for propagating a change *forward
through time*. Tier 1's shortage counterfactual is therefore structurally incapable of being right
for the right reason, whatever number it produces.

**Why Tier 2 fixes the fixable part.** Training `g_τ` on generator-produced `(a, Δ)` pairs gives the
model a signal that is *interventional by construction*: the target is the true effect of an actual
intervention under the true data-generating process. That directly addresses (i) and (ii). It
addresses (iv) only partially — the head learns the *statistical* consequences of temporal feedback
as observed across the corpus, without representing the mechanism — which is why §F1.22 states the
shortage-task limitation explicitly and §F1.24 specifies a recurrent extension.

**Why Tier 1 is nonetheless built.** Three reasons. It is free. It is the honest control: without
it, any Tier 2 result is unanchored, and the field's habit of reporting counterfactual methods
without an associational baseline is precisely what §F1.3 identifies as the gap. And there is a real
possibility it is adequate for `A1`/`A3` on the `impact` task, where the effect is dominated by
graph-structural reachability that a forward pass *does* represent; if the measurement says so, the
simpler system should win.

## F1.10 Loss Functions

### F1.10.1 Targets

For a corpus example `(G_t, a)` and node `v` with task `τ`, the generator supplies:

- `y_f ∈ {0,1}` — the factual label,
- `y_cf ∈ {0,1}` — the counterfactual label under `do(a)`,
- and the derived **true effect on the label**, `Δ_v = y_cf − y_f ∈ {−1, 0, +1}`.

Regressing a three-valued target directly is a poor fit for what the head must produce: a planner
needs a *probability* change, not a class change, and the label is a Bernoulli realisation of an
underlying risk. The corpus therefore aggregates: for each `(v, τ, a)` the true effect target is the
**empirical rate difference across the snapshots in which `v` is scored**,

    Delta_v(a)  =  ( 1/|T_v| ) * sum_{t in T_v} [ y_cf(v,t) − y_f(v,t) ]     in  [-1, +1]

with `|T_v|` the number of test-split snapshots in which `v` carries a label for `τ`. Where a single
snapshot is the unit of prediction, `|T_v| = 1` and the target is the raw `{−1,0,+1}`; the head is
trained on both granularities and §F1.20 ablates the choice.

### F1.10.2 The zero-inflation problem, and the gated loss

The effect distribution is overwhelmingly zero. Most nodes are outside `d(τ)` hops of any single
intervention, and even inside that radius most labels do not flip. A plain regression loss on this
target is minimised almost perfectly by predicting zero everywhere, and would report an excellent
MSE while being useless — the same trap §10 of `reports/layer3_testing.md` navigated with the
`unknown` class, and the same remedy applies: **make "no effect" an explicit, separately-scored
decision rather than a value the regressor happens to output.**

The head emits `(δ̂_v, g_v)` and the prediction is

    Delta_hat_v  =  sigma(g_v) * delta_hat_v

so an exactly-zero effect is representable by `σ(g)→0` without the regressor being pushed toward
zero. The loss is a sum of three terms:

    L  =  L_gate  +  lambda_reg * L_effect  +  lambda_dir * L_direction

**(a) Gate term** — binary cross-entropy on "is there any effect at all":

    L_gate = BCEWithLogits( g_v , 1[ |Delta_v| > eps ] ) , with per-batch pos_weight

`eps = 0.005` (half a percentage point of rate change), chosen to sit above the corpus's own
re-simulation noise (§F1.19.6) rather than at zero. `pos_weight` is the negative/positive ratio in
the batch, capped at 50 — the same capped-weighting convention `ml/hypothesis_ranker.py` uses, and
for the same reason: an uncapped weight at this imbalance makes the loss surface almost entirely
about a handful of positives.

**(b) Effect term** — Huber (smooth-L1) regression, **computed only on nodes with a true nonzero
effect**:

    L_effect = ( 1/|N+| ) * sum_{v in N+} Huber_beta( delta_hat_v − Delta_v ) ,  beta = 0.05

Restricting to `N+ = { v : |Δ_v| > ε }` is what stops the regressor from being dominated by the
zeros the gate is already responsible for. Huber over MSE because the effect distribution has heavy
tails — a removed sole-source supplier can move a downstream product's shortage rate by a large
amount — and MSE would let those few nodes dominate the gradient.

**(c) Direction term** — a sign-agreement penalty on the same subset:

    L_direction = ( 1/|N+| ) * sum_{v in N+} max( 0 , m − sign(Delta_v) * delta_hat_v ) ,  m = 0.01

This exists because **a counterfactual that gets the sign wrong is worse than one that abstains**: a
planner told an intervention helps when it hurts is actively misled. The hinge makes sign error
costly independently of magnitude error, which Huber alone does not.

Defaults `λ_reg = 1.0`, `λ_dir = 0.5`, both ablated in §F1.20.

### F1.10.3 What is *not* in the loss, deliberately

- **No consistency term tying `Δ̂` to the factual head's output.** A term like
  `|σ(f_τ(z)) + Δ̂ − σ(f_τ(z̃))|` would regularise Tier 2 toward Tier 1, which is exactly the
  associational answer this feature exists to improve on.
- **No loss on `removed` entities.** Their outcome is undefined (§F1.8.6).
- **No auxiliary loss on SHARE.** The backbone is frozen; any gradient into it would invalidate
  every cited factual result and re-open the entire Layer 1/2 validation.

## F1.11 Training Procedure

### F1.11.1 Corpus construction (offline, once per configuration)

```
for variant in {B, D, 0, J}:                  # §F1.11.4 explains the variant choice
  for dataset_seed in {42..46}:
    W0 <- generate(variant, seed, no intervention)          # cached; already on disk
    assert null_intervention_byte_identical(W0)             # §F1.8.4 gate
    for a in sample_interventions(W0, N_per_world):
      Wa <- generate(variant, seed, intervene=a)            # CRN-coupled
      assert locality(W0, Wa, a)                            # §F1.8.4 gate
      align(W0, Wa) -> surviving / removed / added
      for t0 in snapshots(W0):
        emit shard (t0, a, labels_f, labels_cf, reach, alignment)
```

**Intervention sampling.** `N_per_world = 200`, stratified so that the corpus is not dominated by
interventions with no measurable effect:

| stratum | share | rationale |
|---|---|---|
| high-degree suppliers (top decile by `SUPPLIES` count) | 25% | largest effects; where the method must work |
| sole-source components' suppliers | 25% | the case a planner most wants scored |
| random suppliers | 20% | unbiased coverage, mostly null effects |
| inventory (`A5`) on stocked-out-prone `(p,w)` | 15% | tests the feature-edit path |
| logistics (`A6`) | 10% | tests the parameter-edit path |
| composites (`k=2`) | 5% | enables §F1.20's additivity ablation |

At 200 interventions × 5 seeds × 4 variants = **4,000 counterfactual world generations**. At the
measured ~20 s per mid-scale world (§F1.18) this is ~22 CPU-hours, parallelisable across processes
and bounded by the memory ceiling recorded for this project (~2 concurrent heavy processes). This is
the dominant cost of Feature 1 and it is paid once.

### F1.11.2 Training loop (Tier 2)

```
freeze SHARE, Markov readout, prediction heads          # assert requires_grad == False
precompute and cache factual embeddings z^tau per (variant, seed, t0)   # §F1.18
for epoch in 1..E:
  for batch of corpus examples:
    z~  <- SHARE(edited graph)      under no_grad
    psi <- InterventionEncoder(a)
    rho <- ReachFeatures(v, a)      precomputed with the shard
    (delta_hat, g) <- g_tau( z, z~, psi, rho )
    L <- L_gate + lambda_reg*L_effect + lambda_dir*L_direction
    backprop into { g_tau, psi, ReachProjection } ONLY
  validate on held-out interventions (§F1.11.3)
  early stop on validation gate-AUC + effect-MAE composite, patience 10
```

`E = 60`, AdamW, `lr = 1e-3`, `weight_decay = 1e-4`, batch = 64 interventions.

### F1.11.3 The split — held-out **interventions**, held-out **worlds**, and held-out **time**

This is the part most easily got wrong and it determines what the evaluation means. Three
independent axes, and the specification requires all three:

- **Held-out interventions.** No intervention in validation or test may appear in training, and no
  *target entity* may either. Splitting by intervention while sharing targets would let the head
  memorise "supplier `X` is important," which is the Layer 3 memorisation failure in a new costume.
- **Held-out dataset seeds.** Leave-one-seed-out, exactly as `reports/layer3_testing.md` §10.3 did,
  because §0.5b makes world-level variance the dominant term and a within-world split would
  understate it badly.
- **Held-out time.** The existing 40/20/40 snapshot split is preserved; counterfactual effects are
  scored on test snapshots only.

The reported headline number uses all three simultaneously (unseen intervention, unseen world,
unseen time). §F1.20 ablates relaxing each one, and the gap between the relaxed and strict numbers
is itself a reportable result about how much of the model's apparent skill is memorisation.

### F1.11.4 Which variants

`B` and `D` are the primary pair — the same pair the Layer 3 work used, differing only in
`HP_ALPHA` (0 vs 0.35), which makes them a controlled contrast for whether counterfactual accuracy
depends on hidden coupling being live. `0` is the clean baseline with no mechanisms. `J` is included
because Mechanism J's multi-tier topology is the only one that makes `dist(u,v) > d(τ)` common for
supplier-to-supplier paths, which is where the reach features earn their place. Variant `K` is
excluded on the standing cost discipline.

## F1.12 Inference Procedure

```
INPUT: snapshot t0, candidate intervention set A_cand = {a_1..a_m}
1. Load G_t; run SHARE ONCE; cache h^0..h^4 and z^tau.          [shared across all m]
2. For each a_j:
     2a. G_edit <- GraphEditor(G_t, a_j)          [typed edit, validity-checked]
     2b. h~ <- SHARE(G_edit); z~^tau <- Markov(h~)
     2c. psi <- InterventionEncoder(a_j); rho <- ReachFeatures(., a_j)
     2d. (delta_hat, g) <- g_tau(...); Delta_hat <- sigmoid(g) * delta_hat
     2e. y_cf_hat <- clip( y_f_hat + Delta_hat , 0, 1 )
3. Rank A_cand by an objective the CALLER supplies (§F1.15), not by a fixed one.
4. Attach provenance: model version, corpus version, reachability flag per node,
   and the abstention flag (§F1.12.1).
OUTPUT: per-intervention, per-task, per-node effect + ranked summary
```

### F1.12.1 Mandatory abstention rule

The system **must refuse to answer** rather than extrapolate, in three cases:

1. **Out-of-support intervention.** The action type or magnitude falls outside the corpus's
   sampled range (e.g. an `A5` delta larger than any trained on). Return `abstain:
   out_of_support`.
2. **Unreachable node.** `dist(v, T(a)) > d(τ)`. The model cannot know of any effect; the honest
   output is exactly `0.0` with an `unreachable` flag, **not** a small nonzero number. This is
   enforced structurally, not learned: the serving path masks these to zero.
3. **Low corpus density.** Fewer than a threshold count of corpus examples with a comparable
   `(action type, target degree decile)`. Return `abstain: sparse_support`.

Abstention is a first-class output, following §10's finding that a module which correctly declines
to claim a cause it has no evidence for is more useful than one that always answers.

## F1.13 Algorithms

**Algorithm 1 — CRN-coupled counterfactual world generation.**
*Input:* base config `(variant, seed, config)`, intervention `a`.
*Output:* world `W_a` coupled to `W_0` on the same random state.
1. Resolve `a`'s targets against `W_0`'s entity ids; reject if any target is absent.
2. Seed the main stream from `CFG.seed`; seed the intervention stream from `0x1470 + CFG.seed`.
3. Run world construction **unchanged**, constructing every entity including intervened ones, so
   draw counts are invariant.
4. Apply `a` to the constructed structures: mark removed entities, insert added edges, adjust
   feature values. Draw any randomness `a` needs from the intervention stream only.
5. Run the simulation with keyed per-entity/per-timestep randomness.
6. Emit CSVs and the manifest, with `intervention` recorded in `resolved_config.json`.

**Algorithm 2 — Graph edit on a built snapshot** (serving path; no generator involved).
1. Copy `HeteroData`; never mutate the cached factual object.
2. For each op, apply the typed edit: for `A1`, compute the incident-edge mask per relation and
   drop those columns; for `A2`/`A4`, append columns to `Supplier -SUPPLIES-> Component` and its
   reverse; for `A5`, update `Product` and `Warehouse` feature columns and the `STOCKED_AT` edge
   attributes; for `A6`, update the `Supplier` feature columns that carry lead-time statistics.
3. Recompute any node feature that is a deterministic function of the edited structure (degree
   counts, source counts). **Do not** recompute features that are functions of *history* — those
   would require re-simulation and are exactly what Tier 1 cannot do; leave them and record that
   they are stale, because pretending otherwise is the silent-failure mode here.
4. Drop any relation whose edge set became empty.
5. Re-apply `ToUndirected` consistency: every forward edit must be mirrored in the reverse relation.
6. Validate: no dangling indices, no negative counts, node counts unchanged except for `A1`.

**Algorithm 3 — Reach feature computation.** Multi-source BFS from `T(a)` on the undirected edited
graph, truncated at depth 6, producing `dist` per node; then the five scalars of §F1.8.5. Cost
`O(|V| + |E|)` per intervention, computed once and cached with the shard.

**Algorithm 4 — Effect evaluation against re-simulation.** For each held-out intervention: read the
true `Δ_v` from the corpus; read the model's `Δ̂_v`; partition nodes by reachability; compute the
metrics of §F1.19 on each partition separately and pooled.

## F1.14 Pseudocode

*Structure and logic only.*

```
FUNCTION build_corpus(variant, seed, spec_set):
    W0 <- load_or_generate(variant, seed)
    ASSERT null_intervention_is_byte_identical(variant, seed)
    FOR a IN spec_set:
        IF NOT targets_exist(a, W0): SKIP and record rejection reason
        Wa <- generate_with_intervention(variant, seed, a)      # CRN
        diff <- entities_whose_trajectory_differs(W0, Wa)
        reach <- causal_reachable_set(a, W0)
        ASSERT diff SUBSET-OF reach                             # locality gate
        align <- align_entities(W0, Wa)
        FOR t0 IN snapshots:
            FOR tau IN {delay, shortage, impact}:
                yf  <- labels(W0, t0, tau)
                ycf <- labels(Wa, t0, tau)
                effect <- aggregate_effect(yf, ycf, align.surviving)
                WRITE shard(t0, tau, a, effect, reach_features, align)
    RETURN corpus_manifest


FUNCTION cf_forward(G, a, tau, frozen_backbone, cf_head):
    z      <- markov_readout(frozen_backbone(G), tau)           # no_grad
    G_edit <- graph_edit(G, a)                                  # Algorithm 2
    z_edit <- markov_readout(frozen_backbone(G_edit), tau)      # no_grad
    psi    <- intervention_encoder(a)
    rho    <- reach_features(G_edit, a, tau)
    delta_raw, gate <- cf_head(concat(z, z_edit, psi, rho))
    delta  <- sigmoid(gate) * delta_raw
    delta  <- MASK_TO_ZERO where rho.unreachable                # §F1.12.1 rule 2
    RETURN delta, gate


FUNCTION cf_loss(delta_raw, gate, delta_true, eps, lambdas):
    is_effect <- ABS(delta_true) > eps
    L_gate    <- bce_with_logits(gate, is_effect, pos_weight=capped_ratio(is_effect))
    IF ANY(is_effect):
        L_eff <- huber(delta_raw[is_effect] - delta_true[is_effect], beta)
        L_dir <- mean(relu(margin - sign(delta_true[is_effect]) * delta_raw[is_effect]))
    ELSE:
        L_eff <- 0 ; L_dir <- 0
    RETURN L_gate + lambdas.reg * L_eff + lambdas.dir * L_dir


FUNCTION evaluate_counterfactual(model, held_out_corpus):
    results <- {}
    FOR a IN held_out_corpus.interventions:
        pred  <- cf_forward for every scored node
        truth <- corpus effect
        FOR partition IN {reachable, unreachable, all}:
            results[a][partition] <- {
                sign_accuracy, effect_mae, effect_rank_corr,
                gate_auc, gate_ece, top_k_intervention_agreement }
    results.floor <- reproduction_floor(model_config, n_replicates)   # §F1.19.6
    RETURN results
```

## F1.15 API Design

**Precondition (§0.4): no service exists today.** Feature 1's research contribution is fully
reachable offline — corpus, training and evaluation are all file-system-based, and
`ml/cf/evaluate.py` produces the tables. The API below is specified for the deployed decision-support
case and is **optional**.

Proposed service: `cf_service`, HTTP/JSON, versioned under `/v1`.

```
POST /v1/counterfactual/score
  request:
    { "snapshot_t0": "2025-04-01T08:00:00Z",
      "world": {"variant": "B", "seed": 42},
      "interventions": [ <intervention spec, §F1.8.2>, ... ],
      "tasks": ["impact", "shortage"],
      "return_nodes": "affected" | "all" | "top:<k>" }
  response:
    { "model_version": "cf-head-1.0.0",
      "corpus_version": "cf-corpus-2026-08",
      "backbone_version": "share-markov-1.4.2",
      "results": [
        { "intervention_id": "a_0417",
          "status": "ok" | "abstain",
          "abstain_reason": null | "out_of_support" | "sparse_support",
          "per_task": {
            "impact": {
              "n_scored": 812, "n_reachable": 96,
              "summary": {"mean_effect": -0.031, "n_improved": 74, "n_worsened": 4},
              "nodes": [ {"entity_id": "...", "y_factual": 0.81,
                          "delta": -0.24, "y_counterfactual": 0.57,
                          "gate_confidence": 0.93, "reachable": true} ] } } } ] }

POST /v1/counterfactual/rank
  As above, plus "objective": {"task": "impact", "aggregate": "sum"|"max"|"count_above:<p>",
                               "scope": "all"|"entity_ids:[...]"}
  Returns interventions ordered by the caller-supplied objective.
  The service does NOT define a default objective — see below.

GET  /v1/counterfactual/action-space
  Returns the six action types, their parameters, and the corpus's supported ranges,
  so a caller can construct in-support interventions without trial and error.

GET  /v1/counterfactual/corpus-coverage?variant=B&seed=42
  Returns support density by (action type, target degree decile) — the data behind
  the sparse_support abstention, exposed so callers can see why they were refused.
```

**Design decisions worth stating:**

- **No default ranking objective.** "Best intervention" depends on cost, feasibility and risk
  appetite, none of which the model knows. The service ranks by an objective the caller supplies and
  refuses to invent one; a default would be a policy decision smuggled in as a technical one.
- **`backbone_version` is returned separately from `model_version`.** A counterfactual result is
  only valid against the backbone it was trained over; surfacing both makes a mismatch detectable.
- **Abstention is a 200, not a 4xx.** "I cannot answer this" is a valid answer, and callers must
  handle it as data rather than as an error.
- **No write endpoints.** The service never mutates world state. Persisting a scored scenario is
  §F1.16's concern and is done by the caller.

## F1.16 Database Schema Changes

Additive DDL against `db/schema.sql`. The database remains optional (§0.4); the corpus is
file-system-native and these tables are for serving and audit.

```sql
CREATE TYPE cf_action_type AS ENUM
    ('remove_supplier','add_backup_source','substitute_supplier',
     'add_dual_sourcing','increase_inventory','change_logistics_provider');

CREATE TYPE cf_status AS ENUM ('ok','abstain_out_of_support','abstain_sparse_support');

-- One row per intervention scenario evaluated.
CREATE TABLE cf_scenarios (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id       VARCHAR(64),                    -- caller's spec id, e.g. 'a_0417'
    world_variant     VARCHAR(8)  NOT NULL,
    world_seed        INTEGER     NOT NULL,
    snapshot_t0       TIMESTAMPTZ NOT NULL,
    spec              JSONB       NOT NULL,           -- the §F1.8.2 document, verbatim
    effective_from    TIMESTAMPTZ NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_cf_scenarios_world ON cf_scenarios (world_variant, world_seed, snapshot_t0);

-- One row per (scenario, op) so the action space is queryable, not buried in JSONB.
CREATE TABLE cf_scenario_ops (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id   UUID NOT NULL REFERENCES cf_scenarios(id) ON DELETE CASCADE,
    op_index      SMALLINT NOT NULL,
    action_type   cf_action_type NOT NULL,
    target_type   entity_type,
    target_id     UUID,
    magnitude     NUMERIC(12,4),                      -- delta_units for A5, NULL otherwise
    params        JSONB,
    UNIQUE (scenario_id, op_index)
);
CREATE INDEX idx_cf_ops_target ON cf_scenario_ops (target_type, target_id);

-- One row per (scenario, task, entity) prediction.
CREATE TABLE cf_predictions (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id           UUID NOT NULL REFERENCES cf_scenarios(id) ON DELETE CASCADE,
    task                  VARCHAR(50) NOT NULL,
    entity_type           entity_type NOT NULL,
    entity_id             UUID NOT NULL,
    y_factual             NUMERIC(5,4) NOT NULL CHECK (y_factual BETWEEN 0 AND 1),
    delta                 NUMERIC(6,4) NOT NULL CHECK (delta BETWEEN -1 AND 1),
    y_counterfactual      NUMERIC(5,4) NOT NULL CHECK (y_counterfactual BETWEEN 0 AND 1),
    gate_confidence       NUMERIC(5,4) CHECK (gate_confidence BETWEEN 0 AND 1),
    reachable             BOOLEAN NOT NULL,           -- dist <= markov depth for this task
    hop_distance          SMALLINT,                   -- NULL when unreachable beyond clip
    status                cf_status NOT NULL DEFAULT 'ok',
    tier                  SMALLINT NOT NULL,          -- 1 = associational control, 2 = CF head
    model_version         VARCHAR(50) NOT NULL,
    backbone_version      VARCHAR(50) NOT NULL,
    corpus_version        VARCHAR(50),
    scored_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_cf_pred_scenario ON cf_predictions (scenario_id, task);
CREATE INDEX idx_cf_pred_entity   ON cf_predictions (entity_type, entity_id, scored_at DESC);
CREATE INDEX idx_cf_pred_delta    ON cf_predictions (task, delta);

-- Ground truth from re-simulation. Populated ONLY for corpus scenarios, never for
-- user-supplied ones -- a user scenario has no re-simulated world to compare against,
-- and a NULL here is the honest representation of that.
CREATE TABLE cf_ground_truth (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id      UUID NOT NULL REFERENCES cf_scenarios(id) ON DELETE CASCADE,
    task             VARCHAR(50) NOT NULL,
    entity_type      entity_type NOT NULL,
    entity_id        UUID NOT NULL,
    true_delta       NUMERIC(6,4) NOT NULL,
    alignment_class  VARCHAR(16) NOT NULL,            -- surviving | removed | added
    UNIQUE (scenario_id, task, entity_type, entity_id)
);
```

**Three schema decisions worth defending.** `tier` is a column rather than two tables, so the
associational control and the counterfactual head are queryable side by side and a report cannot
accidentally show one while claiming the other. `cf_ground_truth` is separate from `cf_predictions`
and is explicitly nullable-by-absence, so there is no column that is "sometimes truth and sometimes
estimate." And `spec` is retained verbatim as JSONB *in addition to* the normalised
`cf_scenario_ops` rows, because the normalised form will drift as the action space grows and the
verbatim document is what makes an old scenario re-runnable.

## F1.17 Dashboard / UI Design

**Fallback first (§0.4):** the non-UI deliverable is a static artifact from `ml/cf/report.py` —
Markdown or self-contained HTML with the §F1.19 tables and per-intervention effect distributions,
in the same style as `ml/analyze_hypothesis.py`'s output. Everything below is optional.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  SCENARIO PLANNER                             world: Variant B / seed 42          │
│                                               snapshot: 2025-04-01                │
├─────────────────────────┬─────────────────────────────────────────────────────────┤
│  INTERVENTIONS          │  PREDICTED EFFECT — impact                              │
│                         │                                                         │
│  [+] add intervention   │   worse ◀───────────────────┼───────────────────▶ better │
│                         │                             0                           │
│  ● A1 remove supplier   │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                            │
│    SUP-8f3a  (12 comps) │   mean −0.031   74 improved / 4 worsened / 734 unchanged │
│    ⚠ sole source for 3  │                                                         │
│                         │  AFFECTED ENTITIES (reachable only, 96 of 812)          │
│  ○ A4 dual-source       │  ┌────────────┬────────┬────────┬────────┬────────────┐ │
│    COMP-19c2            │  │ entity     │ before │ after  │ Δ      │ confidence │ │
│                         │  ├────────────┼────────┼────────┼────────┼────────────┤ │
│  ○ A5 +500 units        │  │ PROD-4a1f  │  0.81  │  0.57  │ −0.24  │ ▓▓▓▓▓▓▓ .93│ │
│    PROD-4a1f @ WH-2     │  │ PROD-9c03  │  0.66  │  0.52  │ −0.14  │ ▓▓▓▓▓░░ .71│ │
│                         │  │ SUP-2b8e   │  0.44  │  0.46  │ +0.02  │ ▓▓░░░░░ .28│ │
│  ─────────────────────  │  └────────────┴────────┴────────┴────────┴────────────┘ │
│  RANK BY                │                                                          │
│  ▸ objective: [sum ▾]   │  716 entities are OUTSIDE this intervention's reach at   │
│    task:      [impact▾] │  depth 4 and are reported as exactly 0, not estimated.   │
│    scope:     [all   ▾] │                                                          │
│                         │  ┌──────────────────────────────────────────────────────┐│
│  ⚠ ranking uses YOUR    │  │ TIER: ② counterfactual head                          ││
│    objective. The model │  │ Tier ① (graph edit + forward pass) would say −0.019.  ││
│    does not define one. │  │ Divergence 0.012 — within this model's measured band. ││
│                         │  └──────────────────────────────────────────────────────┘│
└─────────────────────────┴──────────────────────────────────────────────────────────┘
```

**UI rules that are requirements, not styling:**

1. **Unreachable nodes are shown as a *count*, never as tiny nonzero bars.** Rendering 716 near-zero
   effects invites a user to read noise as signal.
2. **Both tiers are always visible.** Showing only Tier 2 hides the associational control that makes
   it interpretable.
3. **Confidence is the gate probability, labelled as "is there any effect", not as "is this number
   right."** These are different claims and conflating them is the §10 trap.
4. **Abstention renders as an explicit panel** with the reason and a link to corpus coverage — never
   as an empty result, which is indistinguishable from "no effect."
5. **Sole-source and other structural warnings are computed from the graph, not the model**, and are
   marked as such.

## F1.18 Computational Complexity

**Corpus construction (offline, dominant).** Per counterfactual world: one full generator run. The
measured mid-scale (`sup_n=2,000`) figure from this session's work is **~20 s per world**
single-process. Spec scale is ~115 s (`docs/08_Backend_Design.md` §6). At the §F1.11.1 grid of 4,000
worlds: **~22 CPU-hours at mid scale**, ~128 at spec scale. Storage is the harder constraint —
~443 MB gzipped per variant-seed at spec scale means the full counterfactual corpus must **not**
retain worlds. The specification therefore requires the corpus builder to extract labels, reach
features and alignment, then **discard the counterfactual CSVs**, keeping only the derived shards
(kilobytes per intervention) plus the intervention spec, which is sufficient to regenerate any world
exactly by determinism — precisely the trade `db/regenerate_seed.py` already makes.

**Training.** Per example, two frozen SHARE forward passes plus a small MLP. The factual pass is
**cacheable per `(variant, seed, t0)`** and shared across every intervention on that snapshot, so
the marginal per-intervention cost is one edited forward pass:
`O(L · (|E| · d + |V| · d²))` with `L=4`, `d=128`. The head itself is negligible.
For a batch of 64 interventions on one snapshot: 1 cached + 64 edited passes.

**Inference.** Same structure. For `m` candidate interventions on one snapshot: **1 + m** SHARE
passes, `m` BFS traversals (`O(|V| + |E|)` each), `m` head evaluations. Fully parallel across `j`.
An important practical note: the edited graphs differ from the factual graph by a handful of edges,
so a future optimisation could recompute only the `d(τ)`-hop neighbourhood of `T(a)` rather than the
whole graph — reducing per-intervention cost by orders of magnitude. This is specified as a
**future** optimisation (§F1.24) and not required for correctness.

**Memory.** Two `HeteroData` snapshots plus two layer stacks resident. At mid scale this is well
inside the ~2-concurrent-heavy-process budget this project operates under; at spec scale the edited
graph must be built by masking rather than copying to stay inside it.

## F1.19 Evaluation Metrics

The whole evaluation is built on the re-simulation asset. Every metric below compares against a
**true** effect obtained by re-running the generator, not against a proxy.

### F1.19.1 Primary metrics (per task, per action class, never pooled into one number)

| metric | definition | why |
|---|---|---|
| **Sign accuracy** | fraction of truly-affected nodes where `sign(Δ̂) = sign(Δ)` | the decision-relevant quantity: does the intervention help or hurt |
| **Effect MAE** | `mean \|Δ̂ − Δ\|` over truly-affected nodes | magnitude fidelity |
| **Effect rank correlation** | Spearman `ρ(Δ̂, Δ)` over all scored nodes | whether the *ordering* of who is affected is right |
| **Gate AUC** | ROC AUC of `σ(g)` against `1[\|Δ\| > ε]` | can it tell affected from unaffected at all |
| **Gate ECE** | equal-count-bin ECE of `σ(g)` | is "93% confident there is an effect" honest |
| **Top-k intervention agreement** | overlap between model-ranked and truth-ranked top-`k` interventions under a fixed objective | the end-to-end decision quality |

### F1.19.2 The comparison that is the point

Tier 1 vs Tier 2 on every metric, **paired by intervention**, with a paired bootstrap over
interventions and sign consistency across dataset seeds — the protocol
`db/benchmark_eval.py` already supplies. The deliverable is the *gap*, per task and per action
class, and the honest possibilities include "no gap," in which case Tier 1 ships.

### F1.19.3 Mandatory partitions

Metrics are reported separately for:

- **reachable** (`dist ≤ d(τ)`) vs **unreachable** nodes. Pooling them lets a model score superbly by
  predicting zero for the large unreachable majority; the reachable partition is the real test.
- **truly-affected** (`|Δ| > ε`) vs **unaffected** nodes.
- **per action class** `A1`–`A6`, because these are different problems: `A1` is structural and
  well-represented; `A5` is a feature edit whose corpus is compromised by §F1.8.3's construction-time
  limitation and whose numbers must be read with that attached.
- **per task**, because `shortage` uniquely depends on the temporal feedback loop neither tier
  represents (§F1.9.4 iv), and a pooled number would let `impact` carry it.

### F1.19.4 Isolated-node and out-of-distribution audit

Report the count and the metric breakdown for nodes that became isolated or near-isolated under the
edit (§F1.9.3). If accuracy on this subgroup is materially worse, the failure is a distribution
shift in the *backbone*, not in the head, and the correct fix is to include such configurations in
factual training — which is a Layer 1 change and therefore out of scope, so it must be reported as a
limitation rather than patched in the head.

### F1.19.5 Corpus validity gates (run before any metric is believed)

1. Null-intervention byte-identity (§F1.8.4 item 4).
2. Locality containment (§F1.8.4 item 5).
3. Feature identity for out-of-reach entities at `t0` (§F1.8.6).
4. `verify_no_hidden_state()` empty on every counterfactual world.

A failure in any of these invalidates the corpus. They are gates, not diagnostics.

### F1.19.6 Reproduction floor for counterfactual error

Per §0.5a, and this is not optional. Train the CF-head **`n ≥ 8` times with identical
configuration** and report how far each metric moves with nothing changed. If the head turns out to
be bit-deterministic on the target device — as the comparable head in
`reports/layer3_testing.md` §10.3.1 did, where the identical-config floor was exactly 0.0000 — then
report that fact and shift the replicate budget to **model-init-seed** variation, which becomes the
operative floor. Additionally, because §0.5b makes world variance dominant, report the
**dataset-seed spread** of every headline metric alongside its floor. A Tier 2 − Tier 1 gap that
does not clear the larger of (floor, seed spread) is not a finding.

## F1.20 Ablation Studies

| # | ablation | question it answers | expected direction |
|---|---|---|---|
| A1 | **Tier 1 only** (no CF head) | how much does counterfactual training actually buy? | the headline comparison; genuinely uncertain for `impact` |
| A2 | **No reach features** (`ρ` removed) | is explicit reachability necessary, or does SHARE's edited embedding already carry it? | expect degradation concentrated on unreachable nodes |
| A3 | **No edited embedding** (`z̃` removed; head sees `z, ψ, ρ` only) | can the effect be predicted from the intervention description alone, without re-encoding? | a *small* drop here would be a warning that the graph edit is contributing little |
| A4 | **No gate** (plain Huber regression) | is the zero-inflation machinery earning its place? | expect much worse effect-MAE on affected nodes |
| A5 | **λ_dir = 0** | does the direction hinge improve sign accuracy at acceptable MAE cost? | expect sign accuracy down, MAE flat or slightly better |
| A6 | **Snapshot-level vs aggregated targets** (§F1.10.1) | which target granularity trains better? | unknown; genuine open question |
| A7 | **Relaxed splits** — share targets / share worlds / share time, one at a time | how much apparent skill is memorisation? | expect a large jump when target entities are shared, per §F1.9.4 (iii) |
| A8 | **Composite additivity**: predict `a1+a2` from models trained only on singletons | do effects compose? | expect additivity to fail where interventions share a reach neighbourhood |
| A9 | **Variant B vs D** | does counterfactual accuracy depend on hidden coupling being live (`HP_ALPHA` 0 vs 0.35)? | a controlled contrast on the exact pair Layer 3 used |
| A10 | **Backbone unfrozen** (diagnostic only, never shipped) | how much is lost by freezing? | run once, to quantify the cost of the freeze constraint; shipping it would invalidate every factual result |

## F1.21 Failure Modes

| # | failure | detection | mitigation |
|---|---|---|---|
| 1 | **RNG divergence masquerading as effect** — the single most dangerous failure; produces large, confident, entirely spurious effects | null-intervention byte-identity and locality gates (§F1.19.5) | gates block the corpus; no metric is computed until they pass |
| 2 | **Temporal feedback unmodelled** — `shortage` effects mispredicted because inventory/agent-loop dynamics are not represented (§F1.9.4 iv) | per-task metric split; expect `shortage` worst | reported as a limitation; recurrent extension in §F1.24 |
| 3 | **Construction-time vs now-time mismatch** for `A5` (§F1.8.3) | per-action-class split; `A5` effects will be implausibly small | `A5` claims restricted (§F1.22); generator change specified in §F1.24 |
| 4 | **Zero-prediction collapse** — head predicts no effect everywhere and scores well on pooled metrics | reachable/affected partitions (§F1.19.3); gate AUC near 0.5 | gated loss (§F1.10.2); partitioned reporting makes collapse visible |
| 5 | **Target memorisation** — head learns "supplier X matters" rather than what makes a supplier matter | target-disjoint split (§F1.11.3) and ablation A7 | strict split is the default; A7 measures the gap |
| 6 | **Isolated-node distribution shift** — edited graphs contain configurations the backbone never trained on | §F1.19.4 audit | reported, not patched; a backbone fix is out of scope |
| 7 | **Stale history features** — Tier 1's edited graph carries history-derived features from the unedited world (Algorithm 2 step 3) | explicit staleness flag on those columns | flagged in output; it is the fundamental reason Tier 1 is a control and not the deliverable |
| 8 | **Out-of-support extrapolation** — a user asks for an intervention magnitude far outside the corpus | support check at serving | abstention (§F1.12.1) |
| 9 | **Silent action-space drift** — the editor and the generator's intervention implementation diverge, so the trained-on edit and the served edit differ | a shared conformance test: for a sampled intervention, the editor's `HeteroData` must equal the loader's `HeteroData` built from the generator's intervened CSVs | conformance test in CI; this is the counterfactual analogue of the byte-identity check |
| 10 | **Effect below re-simulation noise** — the true effect is smaller than the generator's own stochastic variation | `ε` chosen above measured re-simulation noise; effects below it are labelled "no detectable effect," not "no effect" | honest labelling; the distinction matters to a planner |

## F1.22 Limitations

1. **Construction-time interventions only.** The corpus answers "what if this world had always been
   different," not "what if I change it now" (§F1.8.3). For `A1`–`A4` these largely coincide after
   warm-up; for **`A5` (inventory) they do not**, and no `A5` claim in this feature may be presented
   as a now-time recommendation.
2. **`shortage` counterfactuals are structurally handicapped.** Neither tier represents the
   inventory/agent feedback loop that determines shortage outcomes (§F1.9.4 iv). Expect the weakest
   results here and do not average them away.
3. **The action space is the generator's, not the world's.** Six action classes chosen because the
   generator produces the structures they edit. Real supply-chain interventions (renegotiating
   terms, qualifying a new part, changing safety stock policy) are outside it.
4. **Ground truth is synthetic.** The re-simulation asset makes the *evaluation* rigorous with
   respect to the data-generating process; it says nothing about whether that process resembles a
   real supply chain. This is the standing caveat on every result in this project and it is not
   weakened by the strength of the evaluation protocol.
5. **Frozen backbone.** Effects the backbone's representation cannot express are unreachable for the
   head regardless of training. A10 quantifies the cost; lifting the freeze is out of scope because
   it would invalidate the Layer 1/2 validation.
6. **Composite interventions are barely covered.** 5% of the corpus at `k=2`. Anything beyond pairs
   is unsupported and must abstain.
7. **Scale.** Everything specified is sized for mid scale (`sup_n=2,000`, 15 snapshots). Spec scale
   multiplies corpus cost ~6x and forces the masking-based editor of §F1.18.
8. **No causal identification claim.** This feature predicts interventional outcomes by *learning
   from simulated interventions*. It does not identify causal effects from observational data, and
   nothing here should be read as doing so.

## F1.23 Expected Research Contributions

1. **A counterfactual-GNN evaluation protocol with genuine interventional ground truth**, including
   the common-random-numbers coupling that makes the contrast within-world rather than
   between-world — the methodological core, and the part transferable to any benchmark with a
   re-executable generator.
2. **A quantified associational-to-interventional gap** for a standard supervised heterogeneous GNN,
   per task and per intervention class, on a benchmark where the truth is knowable.
3. **A negative result of known size, if that is what it is.** If Tier 2 does not clear Tier 1 by
   more than the reproduction floor, that is a publishable and useful finding about how much
   counterfactual-specific training actually buys, and the specification is built so that outcome is
   reportable rather than embarrassing.
4. **The reachability partition as a methodological requirement.** Demonstrating that pooled
   counterfactual metrics on graphs are dominated by structurally-unaffected nodes, and that a
   depth-bounded reachability partition is necessary for the metric to mean anything.

## F1.24 Future Extensions

1. **Mid-timeline interventions.** Extend the generator to apply a structural edit at an arbitrary
   `t0` and simulate forward, using Mechanism C's existing pre-scheduled rewire machinery as the
   template. This is the single highest-value extension: it lifts limitation 1 and makes `A5`
   meaningful.
2. **Recurrent counterfactual head.** Give the head a temporal state across snapshots so inventory
   and agent-loop dynamics are representable, addressing limitation 2.
3. **Localised recomputation.** Recompute only the `d(τ)`-hop neighbourhood of `T(a)` instead of the
   whole graph (§F1.18) — a large constant-factor win that makes interactive scenario exploration
   feasible.
4. **Intervention search.** Given a budget and an objective, search the action space rather than
   scoring a supplied set. Requires 3 to be tractable.
5. **Uncertainty on counterfactual effects.** Feature 2's machinery applied to `Δ̂`, giving intervals
   rather than point effects. This is the most natural cross-feature composition and is specified in
   §4.2.
6. **Explaining a counterfactual.** Feature 3 applied to the effect: *why* does removing this
   supplier help this product. §4.3.
7. **Pairwise-coupling generator variant.** `findings/evolution.md` §5 identifies the change that
   would reopen hidden-dependency discovery (pairwise rather than group-mean-mediated coupling). A
   counterfactual corpus built on such a variant would test whether interventions on *hidden*
   structure are predictable, which the current variants cannot.

---

# PART II — FEATURE 2: UNCERTAINTY ESTIMATION

*"How confident is this prediction?"*

---

## F2.1 Motivation

`impact_score = 0.87` is currently presented as a fact. It is an estimate, and the size of the
error bar around it varies enormously across entities: a supplier with 40 recorded shipments and a
dense component neighbourhood is estimated far more precisely than one with three shipments and a
single downstream part. A planner who cannot tell those apart will act on both with equal
conviction, and will be wrong more often on the second — which is exactly the population where
being wrong is most likely.

Two project-specific facts make this feature more than a standard add-on, and both of them cut
*against* the obvious implementation.

**First, this benchmark's dominant variance is not the variance that standard methods measure.**
`reports/phase7_training_results.md` §5.6 measured **dataset-seed std at up to 10.3x model-seed
std** on delay (0.0376 vs 0.0037). Deep Ensembles over model initialisations, MC Dropout, and
evidential heads fitted on one dataset all estimate the *model* term. On delay, that term is under
a tenth of the total. A system that reported model-seed uncertainty as "confidence" would be
reporting the small term while the large one went unmentioned — and would be most confidently wrong
precisely when the world happens to be unusual, which is the failure a planner most needs
protection from.

**Second, this project already knows how to tell a real calibration result from noise**, and that
discipline has to be applied to the confidence numbers themselves. §10 of
`reports/layer3_testing.md` reports ECE 0.0032–0.0036 against a *measured* model-seed floor of
0.0021–0.0023 — the claim is credible because the floor was measured, not because the number was
small. An ECE quoted without a floor is an unvalidated number wearing the costume of a validated
one, which is the same failure this project flagged for attention weights.

## F2.2 Problem Statement

For each task `τ` and node `v`, produce not a point estimate `ŷ^τ_v` but a **calibrated predictive
distribution** — at minimum a probability with an honest interval, at best a decomposition into
sources:

    p(y^tau_v | G_t)  ,  with  Var[y] = E_{world}[ Var_model[y] ]  +  Var_{world}[ E_model[y] ]
                                        \_______________/            \_________________/
                                          model / epistemic            world / dataset
                                          (what ensembles measure)     (10.3x larger on delay)

and state, for every number surfaced, **which of those two terms it contains.**

The system must satisfy three requirements:

1. **Calibration**: among predictions assigned probability `p`, the empirical event rate is `p`.
   Measured by ECE with equal-count bins, reliability diagrams, Brier score and NLL.
2. **Discrimination preserved**: adding uncertainty must not degrade AUC beyond the reproduction
   floor. A perfectly calibrated but less discriminative model is a worse decision tool.
3. **Honest scoping**: any interval presented to a user must be labelled with what it covers. An
   interval covering model variance only must say so.

## F2.3 Research Gap

**In the uncertainty literature.** The canonical comparison (Deep Ensembles vs MC Dropout vs
evidential methods) is conducted almost entirely on benchmarks with a single fixed dataset, where
*epistemic* uncertainty is operationally defined as variation across model initialisations or
dropout masks. This is a reasonable convention for ImageNet, where there is one ImageNet. It is
misleading for any benchmark generated by a stochastic process, where "which world" is a real and
often dominant source of variance — and it is quantitatively wrong here by a measured factor of up
to 10.3x.

The literature does have the vocabulary for this: it is the distinction between epistemic
uncertainty about *parameters* and about *the data-generating process*, and hierarchical Bayesian
treatments handle it in principle. What is missing is any empirical study that *measures both
terms on the same benchmark* and reports how badly the standard methods understate the total. This
project can, because it has 12 variants × 5 seeds of independently generated worlds and has already
measured the ratio.

**The gap this feature fills**: an uncertainty method whose decomposition is explicitly
world-aware, and a measurement of how much a conventional ensemble understates predictive
uncertainty on a generator-based benchmark — with the reproduction-floor discipline applied so the
calibration claim is itself validated.

## F2.4 Scientific Contribution

1. **A two-level uncertainty decomposition matched to the benchmark's actual variance structure**,
   separating within-world (model) from across-world (dataset) uncertainty, with both measured
   rather than one assumed negligible.
2. **A quantification of the understatement** — how far a standard Deep Ensemble's interval falls
   short of the total predictive variance, per task, on a benchmark where the world-level term is
   directly measurable by regenerating worlds.
3. **Reuse of existing 5-dataset-seed sweep infrastructure as ensembling material**, turning a
   methodological practice this project already follows (average over worlds, not initialisations)
   into an uncertainty estimator at near-zero marginal cost.
4. **Calibration claims validated against a measured reproduction floor**, extending the §10
   protocol from a hypothesis-ranking head to the main prediction heads.

## F2.5 System Architecture

**Recommended method: a two-level ensemble with post-hoc recalibration — "Hierarchical Ensemble +
Isotonic" (HEI).** §F2.5.3 justifies this against the alternatives.

### F2.5.1 The two levels

```
   LEVEL 2 — ACROSS WORLDS (dataset seeds)  ......... the DOMINANT term (10.3x on delay)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  world s_d = 42      world s_d = 43     ...     world s_d = 46           │
   │  ┌───────────────┐   ┌───────────────┐         ┌───────────────┐         │
   │  │ LEVEL 1 —     │   │ LEVEL 1 —     │         │ LEVEL 1 —     │         │
   │  │ ACROSS MODELS │   │ ACROSS MODELS │         │ ACROSS MODELS │         │
   │  │ s_m = 0..M-1  │   │ s_m = 0..M-1  │         │ s_m = 0..M-1  │         │
   │  │  M frozen     │   │  M frozen     │         │  M frozen     │         │
   │  │  SHARE+Markov │   │  SHARE+Markov │         │  SHARE+Markov │         │
   │  │  replicas     │   │  replicas     │         │  replicas     │         │
   │  └───────┬───────┘   └───────┬───────┘         └───────┬───────┘         │
   │          │ mu_42, var_42     │ mu_43, var_43           │ mu_46, var_46   │
   └──────────┼───────────────────┼─────────────────────────┼─────────────────┘
              └───────────────────┴─────────────────────────┘
                                  ▼
            mu_bar   = mean_d mu_d                       point estimate
            var_model = mean_d var_d                     WITHIN-world (epistemic, model)
            var_world = var_d( mu_d )                    ACROSS-world (the big one)
            var_total = var_model + var_world            law of total variance
                                  ▼
                    ┌──────────────────────────────┐
                    │ ISOTONIC RECALIBRATION       │  fit on a held-out
                    │ (per task; §F2.8.4)          │  calibration split
                    └──────────────┬───────────────┘
                                   ▼
                       calibrated p, interval, and a
                       LABELLED decomposition of the variance
```

### F2.5.2 What this means operationally

The **critical caveat, and it is a hard one**: `var_world` is computable only when the same entity
exists across worlds. It does not, in general — dataset seed 42 and seed 43 generate *different
suppliers*. So `var_world` is **not** a per-entity quantity here. It is estimable at two levels:

- **Population level** (always available): the spread of a metric or of the predicted-probability
  distribution across worlds. This is what §5.6 measured and it is genuinely informative — it says
  "predictions of this kind carry this much world-level uncertainty."
- **Stratum level** (the useful compromise): entities are bucketed by observable covariates that
  *do* transfer across worlds — degree decile, country, shipment-count decile, lead-time tercile,
  label base rate of the stratum. `var_world` is then estimated per stratum and attached to
  entities in that stratum.

**Position: per-entity world uncertainty is not identifiable on this benchmark, and the
specification does not pretend otherwise.** The system reports a per-entity *model* interval and a
per-stratum *world* inflation, combined into a total, with the composition made explicit in the API
and UI. Claiming a per-entity world-variance number would be fabricating identifiability that does
not exist.

### F2.5.3 Why not the alternatives — a reasoned comparison

| method | what it measures here | verdict |
|---|---|---|
| **MC Dropout** | variation from dropout masks at inference — a crude posterior approximation over *weights only* | **Rejected as primary.** Measures a subset of the smaller term. Its one merit is cost (no retraining), so it is retained as a **cheap baseline** in §F2.20's ablation to quantify what it misses. `dropout=0.2` already exists in the encoder, so this costs nothing to add. |
| **Deep Ensembles (model-init only)** | `var_model` exactly | **Rejected as primary, retained as Level 1.** It is the right estimator for the within-world term and it is well-validated. It is simply not the whole answer here: on delay it captures ~9% of the total std. This project's own §5.6 is the evidence. |
| **Evidential Deep Learning** | a parametric second-order distribution from a single model | **Rejected.** Attractive on cost (one model, one pass), but three problems here. It requires changing the prediction heads and the loss, which touches the validated stack. Its epistemic estimate is known to be poorly identified without OOD training signal. And most decisively, it is fitted on *one world*, so it structurally cannot see the dominant term — it would produce a confident-looking single number of exactly the kind §0.5 warns against. |
| **Bayesian (variational / Laplace / SWAG)** | an approximate weight posterior | **Rejected as primary.** Same structural limitation as evidential — a posterior over weights given *one* dataset is a within-world object. Laplace or SWAG on top of the existing checkpoints is cheap and is listed as a §F2.24 extension for improving Level 1's estimate, not for reaching Level 2. |
| **Frequentist: bootstrap over training data** | resampling variance | **Rejected.** Resampling snapshots breaks the temporal split; resampling nodes breaks graph structure. Neither is coherent on this data. |
| **Two-level ensemble (recommended)** | `var_model` and `var_world` separately, plus their sum | **Adopted.** It is the only option that reaches the dominant term, and the material for Level 2 already exists. |

### F2.5.4 Can the existing 5-dataset-seed infrastructure be reused directly?

**Yes for Level 2, and this is the feature's biggest cost saving — but it only partially addresses
the dataset-variance problem, and the partiality must be stated.**

What it gives directly: the standard sweep already trains the same architecture on 5 independently
generated worlds per variant. Those checkpoints *are* a Level-2 ensemble; no new training is
required to obtain `var_world` at population and stratum level.

What it does **not** give, and why the problem is only partly addressed:

1. **Five worlds is a small sample for a variance.** The sampling error of a variance estimated
   from `n=5` is large (the χ² relative error is ~63% at 4 d.o.f.). §F2.19.5 therefore requires
   `var_world` to be reported with a bootstrap CI, and §F2.22 lists the small-`n` limitation
   plainly. Extending to 10 seeds is cheap (generation is ~20 s/world at mid scale) and is the
   recommended first upgrade.
2. **It cannot be made per-entity** (§F2.5.2). Reusing the sweep does not solve identifiability.
3. **It conflates world variance with label-volume variance.** `docs/phase6_spec_scale_report.md` §4
   (cited via §5.6) found delay positives vary ~2x between seeds because hidden-factor member sets
   are redrawn over a power-law degree distribution. Part of `var_world` is therefore variance in
   *how many positives exist*, not in how predictable they are. §F2.20's ablation A5 separates these
   by conditioning on realised base rate.
4. **Level 1 does need new training**: `M` model-init replicas per world. At `M=5` and 5 worlds that
   is 25 trainings per variant — the dominant new cost, quantified in §F2.18.

## F2.6 Component Diagram (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ EXISTING, UNMODIFIED                                                         │
│  SHARE encoder · Markov depth readout · PredictionHead x3 · FocalLoss        │
│  ml/train.py (used as-is to produce replicas; no objective change)           │
│  db/benchmark_eval.py  (paired bootstrap, sign consistency)                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │  M x S checkpoints (M model seeds, S world seeds)
┌──────────────────────────────▼───────────────────────────────────────────────┐
│ NEW COMPONENTS                                                               │
│                                                                              │
│  ┌────────────────────────────┐   ┌──────────────────────────────────────┐   │
│  │ EnsembleRegistry           │   │ StratumMap                           │   │
│  │  (ml/unc/registry.py)      │   │  (ml/unc/strata.py)                  │   │
│  │  - indexes checkpoints by  │   │  - observable covariates that        │   │
│  │    (variant, s_d, s_m)     │   │    transfer across worlds            │   │
│  │  - integrity: same config  │   │  - degree decile x country x         │   │
│  │    same split, same data   │   │    shipment decile x lead tercile    │   │
│  └─────────────┬──────────────┘   └──────────────┬───────────────────────┘   │
│                │                                  │                          │
│  ┌─────────────▼──────────────────────────────────▼───────────────────────┐  │
│  │ VarianceDecomposer   (ml/unc/decompose.py)                             │  │
│  │   mu_bar, var_model (within-world), var_world (per stratum), var_total │  │
│  │   + bootstrap CIs on every variance component                          │  │
│  └─────────────┬──────────────────────────────────────────────────────────┘  │
│                │                                                             │
│  ┌─────────────▼──────────────┐   ┌──────────────────────────────────────┐   │
│  │ Recalibrator               │   │ CalibrationEvaluator                 │   │
│  │  (ml/unc/calibrate.py)     │   │  (ml/unc/evaluate.py)                │   │
│  │  - isotonic (reuse the     │   │  - ECE (equal-count), reliability,   │   │
│  │    PAVA impl already in    │   │    Brier, NLL, interval coverage     │   │
│  │    ml/hypothesis_ranker.py)│   │  - REPRODUCTION FLOOR for each       │   │
│  │  - temperature (baseline)  │   │  - reuses ml/test_hypothesis_        │   │
│  │  - fit on held-out world   │   │    calibration.py's validated metrics│   │
│  └────────────────────────────┘   └──────────────────────────────────────┘   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ OPTIONAL SERVING: risk_scores.confidence (EXISTING COLUMN, §F2.16),    │  │
│  │ uncertainty API (§F2.15), dashboard bands (§F2.17)                     │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Deliberate reuse worth noting**: the isotonic PAVA implementation and the ECE/reliability
machinery already exist and are **already validated** in `ml/hypothesis_ranker.py` and
`ml/test_hypothesis_calibration.py` (13 checks, including an ECE that must return exactly 0.60 on a
0.9-always predictor of a 30% class). Feature 2 reuses both rather than reimplementing, which means
its metric layer arrives pre-validated.

## F2.7 Data Flow Diagram (ASCII)

```
 TRAINING / FITTING (offline)
 ────────────────────────────
  for s_d in {42..46}:                      # world seeds — EXISTING sweep material
    for s_m in {0..M-1}:                    # model seeds — NEW replicas
        train SHARE+Markov+heads  ─────▶ checkpoint(variant, s_d, s_m)
                                              │
                                              ▼
                                     EnsembleRegistry
                                              │
  held-out CALIBRATION world (one s_d) ───────┤
                                              ▼
                                   collect probabilities on
                                   calibration split, all members
                                              │
                          ┌───────────────────┴──────────────────┐
                          ▼                                      ▼
                  VarianceDecomposer                       Recalibrator
                  mu_bar, var_model,                       isotonic curve
                  var_world per stratum                    per task
                          └───────────────────┬──────────────────┘
                                              ▼
                                    UncertaintyModel artifact
                                    (curves + stratum table + registry ref)

 INFERENCE (per snapshot)
 ────────────────────────
   G_t ──▶ for each member (s_d, s_m):  SHARE ▶ Markov ▶ head ▶ p_{d,m}
                                              │
                                              ▼
              mu_bar = mean over all members            (point estimate)
              var_model = mean_d [ var_m( p_{d,m} ) ]   (within-world)
              var_world = stratum_table[ stratum(v) ]   (across-world, per stratum)
                                              │
                                              ▼
                        isotonic( mu_bar )  ──▶  p_calibrated
                        interval from var_total, labelled by composition
                                              │
                                              ▼
                    { p, [lo, hi], var_model, var_world, n_members, stratum }
```

## F2.8 Mathematical Formulation

### F2.8.1 The decomposition

Let `p_{d,m}(v,τ)` be the predicted probability from the member trained on world seed `d` with model
seed `m`. With `S` world seeds and `M` model seeds:

    mu_d(v)    = (1/M) * sum_m p_{d,m}(v)                     within-world mean
    var_d(v)   = (1/(M-1)) * sum_m ( p_{d,m}(v) - mu_d(v) )^2 within-world variance

    mu_bar(v)  = (1/S) * sum_d mu_d(v)                        pooled point estimate
    var_model  = (1/S) * sum_d var_d(v)                       E_world[ Var_model ]
    var_world  = (1/(S-1)) * sum_d ( mu_d(v) - mu_bar(v) )^2  Var_world[ E_model ]

    var_total  = var_model + var_world                        law of total variance

For entity-level scoring, `mu_bar` and `var_model` are computed per entity **within** its own world
(an entity exists in exactly one world), so in practice:

    per-entity:   mu(v) = (1/M) * sum_m p_{d(v),m}(v) ,   var_model(v) as above with M members
    per-stratum:  var_world(k) estimated across worlds on stratum k  (§F2.8.2)
    total(v)   =  var_model(v) + var_world( stratum(v) )

This is the operational form and it is what the API returns. The pooled `mu_bar` form above is used
for *metric-level* reporting (§F2.19), where averaging over worlds is the correct protocol per
§0.5b.

### F2.8.2 Stratum-level world variance

Partition entities by an observable, world-transferable key
`k(v) = (degree_decile, country, shipment_count_decile, lead_time_tercile)`. For task `τ` and
stratum `k`, let `m_d(k)` be the mean predicted probability over entities of stratum `k` in world
`d`. Then

    var_world(k) = (1/(S-1)) * sum_d ( m_d(k) - mean_d m_d(k) )^2

with a bootstrap CI over worlds. Strata with fewer than `n_min = 30` entities in any world are
merged upward (dropping the finest covariate first) until the threshold is met; the merge path is
recorded so a user can see how coarse their entity's stratum actually is.

**Why these covariates.** They are observable (no privileged state), they exist in every world, and
each is already known to relate to predictability on this benchmark: degree governs how much
neighbourhood evidence exists, `country` is what defines two of the three base event pools
(`H_TRUCK` = USA/Mexico, `H_CUSTOMS` = Germany, per `reports/layer3_testing.md` §9.7.1), shipment
count governs observable history density — the very quantity §10.2 found determines whether an
entity is analysable at all — and lead time is the emitted proxy for the sea-freight flag behind
`H_PORT`.

### F2.8.3 Intervals

The predictive quantity is a probability, so a Gaussian interval is wrong near 0 and 1. Intervals
are constructed on the **logit** scale and mapped back:

    l(v)      = logit( mu(v) )
    sd_l(v)   = sqrt( var_total(v) ) / ( mu(v) * (1 - mu(v)) )      delta-method
    interval  = sigmoid( l(v) ± z * sd_l(v) ) ,  z = 1.96 for 95%

with `mu` clipped to `[1e-6, 1-1e-6]`. The delta-method approximation degrades at extreme `mu`; the
implementation must fall back to a **bootstrap percentile interval over ensemble members** when
`mu < 0.02` or `mu > 0.98`, and the API records which method produced the interval.

### F2.8.4 Recalibration

An ensemble mean is not automatically calibrated, and focal loss makes this worse: focal loss
deliberately down-weights easy negatives, which systematically shifts predicted probabilities away
from base rates. Recalibration is therefore **required, not optional**.

The recommended map is **isotonic regression**, reusing the validated PAVA implementation in
`ml/hypothesis_ranker.py`:

    p_cal = Iso_tau( mu(v) ) ,  Iso_tau fitted on a HELD-OUT world's calibration split

**Temperature scaling** is specified as the baseline comparator (one parameter, cannot overfit, but
cannot fix non-monotone miscalibration). §F2.20 ablation A2 decides between them on measured ECE
with the floor attached.

**The fitting split is a whole held-out world, not a random subset.** Fitting the calibration map on
data from the same world the model was trained on would produce a map that corrects within-world
miscalibration only, and would be optimistic exactly where §5.6 says the variance is. This mirrors
the leave-one-seed-out protocol `reports/layer3_testing.md` §10.3 used.

### F2.8.5 Metrics, stated formally

    ECE = sum_b ( n_b / N ) * | acc_b - conf_b |            equal-COUNT bins, B = 10
    Brier = (1/N) sum_v ( p_v - y_v )^2
    NLL = -(1/N) sum_v [ y_v log p_v + (1-y_v) log(1-p_v) ]
    Coverage(alpha) = (1/N) sum_v 1[ y_v in interval_alpha(v) ]

Equal-count rather than equal-width bins, for the reason `ml/hypothesis_ranker.py` documents: these
predictions pile up near zero (base rates ~11%/~7%/~4%), and equal-width binning would put a handful
of points behind most of the reliability curve.

## F2.9 Neural Network Changes

### F2.9.1 What is reused, unmodified

| component | reuse |
|---|---|
| **SHARE** | unmodified. Ensemble members are independent trainings of the *same* architecture; no architectural change whatsoever. |
| **Markov readout** | unmodified, zero parameters, per member. |
| **Prediction heads** | unmodified. |
| **FocalLoss** | unmodified — deliberately. See §F2.9.3. |
| **`ml/train.py`** | used as-is with different `seed` values. Feature 2 requires **no change to the training objective or loop.** |
| **isotonic / ECE / reliability** | reused from `ml/hypothesis_ranker.py`, already validated by `ml/test_hypothesis_calibration.py`. |

### F2.9.2 What is new

**No new neural layers, and no new prediction heads.** This is the single most important
architectural statement in Feature 2: the recommended method is entirely *post-hoc* over
independently trained replicas of the existing model. Everything new is analysis code plus a small
fitted artifact:

| new artifact | size |
|---|---|
| isotonic calibration curve per task | ≤ a few hundred `(x,y)` knots |
| stratum world-variance table | `\|strata\| × 3` floats |
| ensemble registry (checkpoint index + integrity manifest) | metadata only |

The alternatives rejected in §F2.5.3 would each have required real architectural change — evidential
heads replace the output layer and the loss; variational Bayes replaces every weight with a
distribution — and that is part of why they were rejected. Touching the validated stack is a cost,
not a neutral choice.

### F2.9.3 Why the training objective is deliberately left alone

It would be natural to add a calibration-aware term to the loss. The specification declines, for
three reasons. Changing the objective invalidates comparability with every result cited in this
document and in `findings/evolution.md`. Focal loss's miscalibration is *systematic and monotone*,
which is exactly the case isotonic recalibration handles completely. And a calibration term would
trade discrimination for calibration inside the model, whereas post-hoc recalibration is provably
monotone and therefore **cannot change AUC at all** — which makes requirement 2 of §F2.2 satisfiable
by construction rather than by measurement. (`ml/test_hypothesis_calibration.py` already asserts
this property of the isotonic implementation out-of-sample.)

## F2.10 Loss Functions

**There is no new training loss.** Members are trained with the existing focal objective,
`gamma = 2`, `alpha = 1 − positive_rate` per task, unchanged.

Two *fitting* objectives exist, neither of which is a neural loss:

**(a) Isotonic fit** — minimise squared error subject to monotonicity, solved exactly by PAVA:

    min over g non-decreasing:  sum_v ( g( mu(v) ) - y_v )^2

**(b) Temperature fit** (baseline only) — minimise NLL over one scalar `T > 0`:

    min over T:  - sum_v [ y_v log sigma( l(v)/T ) + (1-y_v) log( 1 - sigma( l(v)/T ) ) ]

Both are fitted on the held-out calibration world's calibration split and evaluated on a different
held-out world's test split.

**Optional diagnostic objective (not part of the shipped path):** proper scoring-rule decomposition
of the Brier score into reliability, resolution and uncertainty, reported in §F2.19 to show *which
component* recalibration improved. It is a decomposition, not an objective, and is listed here
because it is the natural place a reader will look for it.

## F2.11 Training Procedure

### F2.11.1 Producing the ensemble

```
FIXED: variant, config, architecture (rgcn_attn_markov), hidden=128, num_bases=10,
       num_layers=4, dropout=0.2, epochs, temporal split 40/20/40.
VARIED: s_d in {42,43,44,45,46}   (world seed  -> Level 2)
        s_m in {0,1,2,3,4}        (model seed  -> Level 1)

for s_d in world_seeds:
    data <- load_or_generate(variant, s_d)          # existing sweep material
    for s_m in model_seeds:
        ckpt <- train_model('rgcn_attn_markov', data, seed=s_m, **CFG)
        register(ckpt, variant, s_d, s_m, config_hash, split_hash, data_hash)
```

**Integrity requirements on the registry**, because an ensemble silently containing a member trained
on a different split or config produces a variance estimate that means nothing:

- `config_hash` must match across all members.
- `split_hash` (the snapshot index boundaries) must match across all members of the same world.
- `data_hash` must differ across worlds and match within a world — this is what proves Level 2 is
  actually varying the world and Level 1 actually is not.
- Any mismatch is a hard error, not a warning.

### F2.11.2 Role assignment of world seeds

Five worlds cannot simultaneously be ensemble members, the calibration set, and the test set. The
protocol is **leave-two-out, rotated**:

| role | worlds | purpose |
|---|---|---|
| ensemble members | 3 | produce `mu`, `var_model` |
| calibration | 1 | fit isotonic / temperature |
| test | 1 | report ECE, Brier, NLL, coverage |

rotated over all 20 ordered `(calibration, test)` pairs, or a fixed 5-fold rotation if cost binds.
Results are averaged over folds and the fold spread is reported — it is a direct read on
world-level sensitivity and is expected to be large per §5.6.

**A consequence that must be stated:** with only 3 worlds contributing members, the `var_world`
estimate in any single fold rests on `S=3`, which is very thin. §F2.19.5 requires the `var_world`
CI to be reported and §F2.22 lists it as the principal limitation. Extending to 10 world seeds
(~20 s of generation each at mid scale) is the cheapest material improvement available to this
feature and is the recommended first upgrade.

### F2.11.3 Fitting the calibration and stratum artifacts

```
1. Score the calibration world with all members -> mu(v), var_model(v).
2. Fit Iso_tau on ( mu(v), y_v ) for each task, on the calibration SPLIT of that world.
3. Build strata from observable covariates (§F2.8.2); merge sparse strata upward; record merges.
4. Estimate var_world per stratum ACROSS the member worlds; bootstrap CI over worlds.
5. Persist { iso curves, stratum table, registry ref, config hashes } as the UncertaintyModel.
```

### F2.11.4 Cost, and how to avoid paying it twice

`M × S = 25` trainings per variant is the dominant cost. Two mitigations, both real:

- **The `S` axis already exists.** The standard sweep trains one model per world per variant. Those
  are `s_m = 0` for every world, so `S` of the 25 are already on disk.
- **`M` need not be 5.** `M = 3` is defensible for a within-world variance whose magnitude is known
  to be small (§5.6), and §F2.20 ablation A6 measures the sensitivity of every downstream number to
  `M`. Start at `M = 3`, raise only if A6 shows the estimate is still moving.

## F2.12 Inference Procedure

```
INPUT: snapshot G_t, entity set (or all), tasks
1. For each registered member (s_d, s_m) applicable to this world:
       run SHARE ▶ Markov ▶ head  -> p_{d,m}(v, tau)
   [members are independent; fully parallel; each is one standard forward pass]
2. mu(v)        = mean over model seeds within the entity's own world
   var_model(v) = unbiased variance over those model seeds
3. stratum(v)   = lookup from observable covariates; var_world = stratum table
4. var_total(v) = var_model(v) + var_world(stratum(v))
5. p_cal(v)     = Iso_tau( mu(v) )
6. interval     = logit-scale delta method, or member bootstrap at extreme mu (§F2.8.3)
7. EMIT { p_cal, interval, var_model, var_world, var_total, n_members,
          stratum_id, stratum_merge_depth, interval_method, coverage_target }
OUTPUT: calibrated probability + LABELLED uncertainty decomposition
```

### F2.12.1 The labelling requirement

Every uncertainty number leaving the system carries a **scope label**, and this is a requirement of
the specification rather than a presentational nicety:

| field | scope | meaning |
|---|---|---|
| `var_model` | this entity, this world | how much the estimate moves under retraining |
| `var_world` | this entity's **stratum**, across worlds | how much estimates of this *kind* move across generated worlds |
| `var_total` | combined | the honest total, subject to the stratum approximation |
| `n_members` | — | how many models the estimate rests on |
| `stratum_merge_depth` | — | how coarse the stratum had to become to have `n ≥ 30` |

A consumer that displays a single "confidence" number without these is misusing the output, and
§F2.17 specifies the UI accordingly.

## F2.13 Algorithms

**Algorithm 1 — Two-level variance decomposition.** Given member probabilities indexed by
`(world, model)`: compute per-world means and unbiased variances over model seeds; pool means to
`mu_bar`; `var_model` is the mean of within-world variances; `var_world` is the unbiased variance of
per-world means. Report each with a bootstrap CI resampling **worlds** (not entities) for
`var_world`, and resampling **model seeds** for `var_model`.

**Algorithm 2 — Stratum construction with sparsity merging.** Build the finest key; count entities
per stratum per world; while any stratum has `n < 30` in any world, drop the least informative
remaining covariate (order: lead-time tercile → shipment decile → degree decile → country) and
rebuild; record the final key and the merge depth.

**Algorithm 3 — Isotonic recalibration** (reuse). PAVA over `(mu, y)` sorted by `mu`; store as a
step function; apply by interpolation with clamped ends. The implementation and its validation
already exist in `ml/hypothesis_ranker.py` / `ml/test_hypothesis_calibration.py`.

**Algorithm 4 — Interval construction with fallback.** If `0.02 ≤ mu ≤ 0.98`, use the logit
delta-method interval; else compute the empirical percentile interval over member probabilities.
Record which was used.

**Algorithm 5 — Calibration reproduction floor.** Repeat the entire fit-and-evaluate pipeline `n ≥ 8`
times with identical configuration; report mean-abs and max-abs deviation of ECE, Brier and NLL. If
the pipeline is deterministic given fixed checkpoints — it will be, since no training occurs in the
fit — then the floor of the *fitting* step is zero and the operative floor is the one induced by
retraining the members. In that case, retrain the ensemble `n ≥ 8` times and floor the metrics
against that. **This distinction must be resolved by measurement, exactly as
`reports/layer3_testing.md` §10.3.1 resolved it, not assumed.**

## F2.14 Pseudocode

```
FUNCTION build_uncertainty_model(variant, world_seeds, model_seeds, roles):
    registry <- EMPTY
    FOR s_d IN world_seeds:
        data <- load_or_generate(variant, s_d)
        FOR s_m IN model_seeds:
            ckpt <- train_existing_model(data, seed=s_m)     # UNCHANGED objective
            registry.add(ckpt, s_d, s_m, hashes(data, config, split))
    ASSERT registry.config_hashes_all_equal()
    ASSERT registry.data_hash_differs_across_worlds()
    ASSERT registry.data_hash_equal_within_world()

    member_worlds, calib_world, test_world <- roles
    P_cal <- score(registry.members(member_worlds), calib_world)
    mu, var_model <- within_world_stats(P_cal)
    strata <- build_strata(calib_world)                       # Algorithm 2
    var_world <- across_world_variance(registry, strata)      # Algorithm 1
    iso <- FOR EACH task: fit_isotonic(mu, labels(calib_world))
    RETURN UncertaintyModel(iso, strata, var_world, registry_ref)


FUNCTION predict_with_uncertainty(model, G_t, v, tau):
    p_members <- [ member.forward(G_t)[tau][v] FOR member IN model.registry ]
    mu        <- mean(p_members within v's world)
    var_m     <- var(p_members within v's world, ddof=1)
    k         <- model.strata.lookup(v)
    var_w     <- model.var_world[tau][k]
    p_cal     <- model.iso[tau].apply(mu)
    lo, hi    <- interval(mu, var_m + var_w)                  # Algorithm 4
    RETURN { p: p_cal, lo, hi, var_model: var_m, var_world: var_w,
             n_members: len(p_members), stratum: k,
             merge_depth: model.strata.merge_depth[k] }


FUNCTION evaluate_calibration(model, test_world):
    preds <- predict_with_uncertainty for all scored entities
    out <- { ece: ece_equal_count(preds.p, labels, B=10),
             reliability: reliability_curve(preds.p, labels, B=10),
             brier: brier(preds.p, labels),
             nll: nll(preds.p, labels),
             coverage_95: coverage(preds.lo, preds.hi, labels),
             auc: roc_auc(preds.p, labels) }             # must not degrade
    out.floor <- calibration_reproduction_floor(...)     # Algorithm 5, MANDATORY
    out.by_stratum <- SAME METRICS per stratum           # §F2.19.4
    RETURN out
```

## F2.15 API Design

Optional (§0.4). Offline fallback: a static calibration report from `ml/unc/report.py` containing
reliability diagrams and the §F2.19 tables.

```
POST /v1/predict/with-uncertainty
  request: { "snapshot_t0": "...", "world": {"variant":"B","seed":42},
             "tasks": ["impact"], "entity_ids": [...] | null,
             "interval": 0.95 }
  response:
    { "model_version": "...", "uncertainty_model_version": "...",
      "n_members": 15, "n_member_worlds": 3, "n_member_model_seeds": 5,
      "predictions": [
        { "entity_id": "...", "task": "impact",
          "p": 0.87, "interval": [0.79, 0.92], "interval_method": "logit_delta",
          "uncertainty": {
            "var_model": 0.0009,   "var_model_scope": "this entity, this world",
            "var_world": 0.0141,   "var_world_scope": "stratum d7|Germany|s4|lt2, across worlds",
            "var_total": 0.0150,
            "world_share": 0.94 },
          "stratum": {"id": "d7|Germany|s4|lt2", "merge_depth": 0, "n_entities": 118},
          "caveat": "var_world is a per-stratum estimate; per-entity world uncertainty
                     is not identifiable on this benchmark (spec §F2.5.2)." } ] }

GET  /v1/uncertainty/calibration
  Returns the current reliability curve, ECE, Brier, NLL, coverage, AND the measured
  reproduction floor for each — so a consumer can see whether the calibration claim
  itself clears noise.

GET  /v1/uncertainty/strata
  The stratum table with per-stratum var_world, its bootstrap CI, entity counts and
  merge depth. Exposed because a user is entitled to know how coarse the estimate
  attached to their entity actually is.

GET  /v1/uncertainty/ensemble
  Registry composition: member count, world seeds, model seeds, config hash.
  Makes "how many models is this resting on" answerable without reading a report.
```

**Design decisions.** `world_share = var_world / var_total` is returned as a first-class field
because it is the number that tells a consumer whether the interval is dominated by the term the
system can only estimate coarsely. The `caveat` string is part of the contract, not documentation —
the identifiability limit travels with the number. And `/calibration` returns the floor alongside
the metric, so the API cannot be used to report an ECE without its noise context.

## F2.16 Database Schema Changes

**Existing columns are adopted rather than duplicated.** `risk_scores` already has
`confidence NUMERIC(5,4)` and `scoring_method` with a `gnn_native` value — both unused legacy from
V1. `confidence` is defined here as `1 − 2·sqrt(var_total)` clipped to `[0,1]` (a bounded,
monotone-decreasing function of total predictive sd) **for backward compatibility only**; every new
consumer should read the decomposition table below instead, because a scalar "confidence" cannot
carry the scope labels §F2.12.1 requires.

```sql
-- Additive: the decomposition behind each scored prediction.
CREATE TABLE prediction_uncertainty (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_score_id        UUID REFERENCES risk_scores(id) ON DELETE CASCADE,
    task                 VARCHAR(50) NOT NULL,
    entity_type          entity_type NOT NULL,
    entity_id            UUID NOT NULL,
    snapshot_t0          TIMESTAMPTZ NOT NULL,
    p_calibrated         NUMERIC(6,5) NOT NULL CHECK (p_calibrated BETWEEN 0 AND 1),
    p_raw_ensemble_mean  NUMERIC(6,5) NOT NULL,
    interval_lo          NUMERIC(6,5) NOT NULL,
    interval_hi          NUMERIC(6,5) NOT NULL,
    interval_level       NUMERIC(4,3) NOT NULL DEFAULT 0.95,
    interval_method      VARCHAR(24) NOT NULL,      -- logit_delta | member_bootstrap
    var_model            NUMERIC(10,8) NOT NULL,    -- within-world (per entity)
    var_world            NUMERIC(10,8),             -- across-world (per STRATUM; NULL if none)
    var_total            NUMERIC(10,8) NOT NULL,
    n_members            SMALLINT NOT NULL,
    n_member_worlds      SMALLINT NOT NULL,
    stratum_id           VARCHAR(64),
    stratum_merge_depth  SMALLINT,
    uncertainty_model_version VARCHAR(50) NOT NULL,
    scored_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (interval_lo <= interval_hi)
);
CREATE INDEX idx_pu_entity ON prediction_uncertainty (entity_type, entity_id, scored_at DESC);
CREATE INDEX idx_pu_task   ON prediction_uncertainty (task, snapshot_t0);

-- The stratum table, versioned, so a historical prediction can be re-interpreted.
CREATE TABLE uncertainty_strata (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uncertainty_model_version VARCHAR(50) NOT NULL,
    task                VARCHAR(50) NOT NULL,
    stratum_id          VARCHAR(64) NOT NULL,
    covariate_key       JSONB NOT NULL,
    merge_depth         SMALLINT NOT NULL,
    n_entities_min      INTEGER NOT NULL,      -- min across member worlds
    var_world           NUMERIC(10,8) NOT NULL,
    var_world_ci_lo     NUMERIC(10,8),
    var_world_ci_hi     NUMERIC(10,8),
    n_worlds            SMALLINT NOT NULL,
    UNIQUE (uncertainty_model_version, task, stratum_id)
);

-- Calibration state, INCLUDING its own reproduction floor. A calibration metric
-- stored without its floor is exactly the unvalidated number this project has
-- twice flagged; the schema makes omitting it impossible.
CREATE TABLE calibration_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uncertainty_model_version VARCHAR(50) NOT NULL,
    task                VARCHAR(50) NOT NULL,
    evaluated_on_world  INTEGER NOT NULL,
    n_instances         INTEGER NOT NULL,
    ece                 NUMERIC(8,6) NOT NULL,
    ece_floor_max_abs   NUMERIC(8,6) NOT NULL,
    brier               NUMERIC(8,6) NOT NULL,
    brier_floor_max_abs NUMERIC(8,6) NOT NULL,
    nll                 NUMERIC(10,6) NOT NULL,
    auc                 NUMERIC(6,5) NOT NULL,
    coverage_95         NUMERIC(6,5) NOT NULL,
    reliability_curve   JSONB NOT NULL,        -- [{n, confidence, empirical}, ...]
    floor_n_replicates  SMALLINT NOT NULL,
    floor_kind          VARCHAR(24) NOT NULL,  -- identical_config | model_seed | dataset_seed
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Registry of ensemble members, so an interval is traceable to the models behind it.
CREATE TABLE ensemble_members (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uncertainty_model_version VARCHAR(50) NOT NULL,
    world_variant       VARCHAR(8) NOT NULL,
    world_seed          INTEGER NOT NULL,
    model_seed          INTEGER NOT NULL,
    role                VARCHAR(16) NOT NULL,  -- member | calibration | test
    checkpoint_path     TEXT NOT NULL,
    config_hash         CHAR(64) NOT NULL,
    data_hash           CHAR(64) NOT NULL,
    split_hash          CHAR(64) NOT NULL,
    UNIQUE (uncertainty_model_version, world_variant, world_seed, model_seed)
);
```

`ece_floor_max_abs` and `floor_kind` being `NOT NULL` is the schema-level enforcement of §0.5a: it
is not possible to record a calibration result in this system without also recording the noise floor
it was judged against, and which kind of floor it is.

## F2.17 Dashboard / UI Design

Fallback: static reliability diagrams and tables from `ml/unc/report.py`.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  SUPPLIER RISK — SUP-8f3a                          snapshot 2025-04-01           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   impact risk        0.87   ├────────────●────────┤     95% interval 0.79 – 0.92 │
│                          0.0                    1.0                              │
│                                                                                  │
│   WHERE THE UNCERTAINTY COMES FROM                                               │
│   ┌────────────────────────────────────────────────────────────────────────────┐ │
│   │ across worlds  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  94%           │ │
│   │ across models  ▓▓                                             6%            │ │
│   └────────────────────────────────────────────────────────────────────────────┘ │
│   ⓘ "Across worlds" is estimated for this supplier's STRATUM                     │
│     (degree decile 7 · Germany · shipments d4 · lead-time t2 · 118 entities),    │
│     not for this supplier individually — per-entity world uncertainty is not     │
│     identifiable on this benchmark. It rests on 3 generated worlds.              │
│                                                                                  │
│   BASED ON 15 models  (3 worlds × 5 model seeds)                                 │
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  CALIBRATION — is 0.87 honest?                                                   │
│                                                                                  │
│   1.0┤                                                    ·'                     │
│      │                                              ·''                          │
│   emp│                                        ●''                                │
│      │                                  ●''                                      │
│   0.5┤                            ●''                                            │
│      │                      ●''                                                  │
│      │                ●''                                                        │
│      │          ●''         ● measured    ·· ideal                               │
│   0.0┼────●''──────────────────────────────────────────                          │
│      0.0                  predicted                  1.0                         │
│                                                                                  │
│   ECE 0.0041   (measured floor 0.0023 — clears)      Brier 0.0612   AUC 0.9365   │
│   95% interval coverage: 0.938 on held-out world                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**UI rules that are requirements:**

1. **The two variance sources are always shown separately**, never summed into one bar. The whole
   point of the feature is that they are different and unequally knowable.
2. **The stratum caveat is displayed inline**, not in a tooltip or a footnote. A user reading "94%
   across worlds" must see, in the same glance, that it is a stratum-level estimate.
3. **`n_members` and `n_member_worlds` are always visible.** An interval from 3 worlds and one from
   30 are different objects.
4. **ECE is displayed with its floor**, and the word "clears" or "does not clear" appears. A
   calibration number without its floor is not shown at all.
5. **No traffic-light colouring of confidence.** Mapping a continuous, coarsely-estimated quantity
   onto green/amber/red invents precision the estimate does not have.

## F2.18 Computational Complexity

**Training.** `M × S` trainings per variant. At the measured mid-scale cost for this architecture
(`reports/layer2_testing.md` §4: Markov arm 396–456 s per run), `M=5, S=5` is **25 runs ≈ 2.8–3.2
CPU-hours per variant**, of which `S=5` already exist from the standard sweep, so the marginal cost
is 20 runs ≈ 2.2–2.6 h. At `M=3` it is 10 marginal runs ≈ 1.1–1.3 h. This is modest and it is a
one-time cost per variant.

**Fitting.** Isotonic is `O(N log N)` per task; stratum construction is `O(N)`; bootstrap CIs are
`O(B · S)` with `B=1000` over `S` worlds — seconds in total.

**Inference.** `|𝔈|` independent forward passes per snapshot instead of 1 — a **linear** cost
multiple, `15×` at `M=5, S=3`. This is the feature's real operational cost and there is no way
around it within this method family. Three mitigations, in order of preference:

1. **Members are embarrassingly parallel.** Wall-clock is `ceil(|𝔈| / workers)` forward passes.
2. **Inference runs once per snapshot** in this deployment pattern (monthly `t0`s), not per query,
   so the multiple applies to a batch job rather than to a request path.
3. **`M` is tunable** and §F2.20 A6 measures where the estimate stops moving; if `M=3` suffices, the
   multiple drops to `9×`.

Memory is `|𝔈| × ` one model (~0.8M params each — trivial) plus one snapshot resident at a time if
members are run sequentially, which is the recommended default.

## F2.19 Evaluation Protocol

### F2.19.1 Core metrics

Per task, on a held-out **world**: ECE (equal-count, `B=10`), full reliability curve, Brier, NLL,
95% interval coverage, and AUC (which must not degrade — and cannot, since isotonic is monotone;
reporting it is a check that the pipeline is wired correctly, not a hypothesis).

### F2.19.2 The comparison grid

| arm | what it is | why it is in the grid |
|---|---|---|
| **single model, raw** | today's system | the baseline being improved on |
| **single model + isotonic** | recalibration only, no ensemble | isolates how much is recalibration vs ensembling |
| **MC Dropout** | `T=30` stochastic passes, one model | the cheap standard method; measures what dropout-based uncertainty misses |
| **Deep Ensemble (model seeds only)** | Level 1 only | **the key comparator** — this is what the literature would do |
| **HEI (recommended)** | Levels 1+2 + isotonic | the proposal |

The headline result is **HEI vs Deep Ensemble**, because that difference *is* the contribution: it
is the measured understatement of a model-init-only ensemble on this benchmark.

### F2.19.3 Interval coverage as the discriminating test

ECE measures whether the *point* probability is honest; it is insensitive to whether the *interval*
is. The discriminating test between a Level-1 and a Level-1+2 method is therefore **coverage on a
held-out world**: nominal 95% intervals should contain the outcome 95% of the time. The prediction
this specification makes, and which the experiment will confirm or refute, is that a
model-init-only ensemble will show coverage **materially below nominal** on `delay` (where the world
term is 10.3x the model term) and close to nominal on `impact` (where the ratio is 1.3x). If that
pattern does not appear, the two-level design is not earning its cost and §F2.20 A1 should retire
it.

### F2.19.4 Mandatory disaggregation

Never one pooled number. Report per task; per stratum (at least by degree decile and by
shipment-count decile); and separately for entities whose stratum required merging
(`merge_depth > 0`), since their `var_world` is coarser and their coverage is expected to be worse.

### F2.19.5 Variance-estimate uncertainty

`var_world` from `S ∈ {3,5}` worlds is a very noisy statistic. Report a bootstrap CI over worlds for
every `var_world`, and **do not display a `var_world` whose CI spans an order of magnitude** — show
the CI instead, or fall back to the coarser parent stratum.

### F2.19.6 Reproduction floor — mandatory, and the kind must be identified

Per §0.5a and Algorithm 5. Resolve by measurement which floor is operative:

- If the fit is deterministic given fixed checkpoints (expected), the identical-config floor of the
  *fitting* step is 0.0000, exactly as `reports/layer3_testing.md` §10.3.1 found for the comparable
  head. **Report that, and do not present it as a result.**
- The operative floor is then the **member-retraining** floor: retrain the ensemble `n ≥ 8` times and
  measure how far ECE, Brier and coverage move. Record `floor_kind = 'model_seed'`.
- Report the **dataset-seed spread** of each metric alongside, since §0.5b makes it the larger term.

An ECE improvement that does not clear the larger of these is not a finding.

## F2.20 Ablation Studies

| # | ablation | question | expected direction |
|---|---|---|---|
| A1 | **Level 2 removed** (model-seed ensemble only) | how much does world-level ensembling add? | the headline; expect large coverage gains on delay, small on impact |
| A2 | **Isotonic vs temperature vs none** | which recalibration, and does ensembling alone calibrate? | expect isotonic ≥ temperature ≫ none, given focal loss |
| A3 | **MC Dropout instead of ensembling** | is the cheap method adequate? | expect worst coverage; quantifies the cost of the shortcut |
| A4 | **Stratum granularity** (full key → country only → global) | how much does per-stratum world variance beat one global number? | genuinely open; if global is as good, simplify |
| A5 | **Condition on realised base rate** | how much of `var_world` is label-volume variation rather than predictability variation? | expect a substantial share, per §F2.5.4 item 3 |
| A6 | **`M` sweep** (1, 3, 5, 10) | where does `var_model` stop moving? | sets the production `M` |
| A7 | **`S` sweep** (3, 5, 10) | where does `var_world` stop moving? | expect still moving at 5 — the case for more seeds |
| A8 | **Per-task focal `alpha` sensitivity** | does the recalibration map depend on the class weighting? | expect isotonic to absorb it entirely |
| A9 | **Calibration fitted in-world vs held-out world** | how optimistic is in-world calibration fitting? | expect materially optimistic; justifies §F2.8.4's protocol |

## F2.21 Failure Modes

| # | failure | detection | mitigation |
|---|---|---|---|
| 1 | **Reporting model uncertainty as "confidence"** — the headline risk of this whole feature | scope labels are mandatory fields (§F2.12.1); API returns `world_share` | schema `NOT NULL` on the decomposition; UI rule 1 |
| 2 | **Overconfident intervals** — nominal 95% covering far less | coverage on held-out world (§F2.19.3) | two-level variance; if still short, widen via a measured coverage-calibration factor and report it as such |
| 3 | **Stratum too coarse** — `var_world` attached to an entity that does not resemble its stratum | `merge_depth` reported; per-merge-depth coverage (§F2.19.4) | expose merge depth; fall back to global `var_world` and say so |
| 4 | **`var_world` estimated from too few worlds** | bootstrap CI (§F2.19.5) | suppress display when the CI is uninformative; raise `S` |
| 5 | **Ensemble members not actually independent** (shared checkpoint, wrong seed, same data) | registry hash integrity checks (§F2.11.1) | hard error, not a warning |
| 6 | **Calibration drift across snapshots** — the map fitted on old snapshots decays | ECE tracked per test snapshot over time | re-fit on a schedule; version the calibration artifact |
| 7 | **Isotonic overfitting on a small calibration set** | out-of-sample ECE vs in-sample; the validated implementation already checks out-of-sample behaviour | fit on a whole held-out world; monitor knot count vs `N` |
| 8 | **AUC silently degraded** by a pipeline bug | AUC reported every run; monotone maps cannot change it, so any change is a bug | treat any AUC movement beyond the floor as a defect, not a result |
| 9 | **Extreme-probability intervals nonsensical** | interval method recorded; check `lo ≥ 0`, `hi ≤ 1` | bootstrap fallback below 0.02 / above 0.98 (§F2.8.3) |
| 10 | **Cost pressure collapses the ensemble to `M=1`** in deployment | `n_members` is a required output field | make the degradation visible rather than silent; a 1-member "ensemble" must report `var_model = NULL`, not 0 |

## F2.22 Limitations

1. **Per-entity world uncertainty is not identifiable on this benchmark.** Entities do not persist
   across dataset seeds. `var_world` is a stratum-level estimate, and every consumer-facing number
   carries that caveat (§F2.5.2). This is the single most important limitation and it is structural,
   not a matter of effort.
2. **`var_world` rests on 3–5 worlds.** A variance from `n=5` has ~63% relative error at 4 d.o.f.
   Bootstrap CIs are mandatory and will often be wide. Raising `S` is cheap and is the first
   recommended upgrade.
3. **World variance and label-volume variance are conflated** unless A5 is run: part of the
   across-world spread is variation in how many positives the generator drew, not in how predictable
   they are.
4. **Calibration is snapshot-averaged.** A single map per task is fitted across the calibration
   world's snapshots; genuine temporal drift in calibration is not modelled (failure mode 6).
5. **Uncertainty is over the model's own predictive distribution, not over the world's realism.**
   None of this addresses whether the generator resembles a real supply chain — the standing caveat.
6. **Inference cost is `|𝔈|×`.** There is no free version of this within the ensemble family.
7. **No OOD detection.** The system does not flag inputs unlike anything trained on; a confidently
   wrong prediction on a genuinely novel structure remains possible. This is the natural
   composition point with Feature 1's out-of-support abstention and is noted in §4.2.
8. **Coverage is validated on synthetic held-out worlds** drawn from the same generator — an
   in-distribution test of an in-distribution claim.

## F2.23 Expected Research Contributions

1. **A measurement of how much a standard Deep Ensemble understates predictive uncertainty on a
   generator-based benchmark** — per task, with the 10.3x/4.8x/1.3x variance ratio as the a-priori
   prediction and interval coverage as the test.
2. **A two-level uncertainty decomposition and the identifiability boundary that comes with it** —
   including the negative result that per-entity world uncertainty is *not* identifiable when
   entities do not persist across worlds, which is a general property of generator-based benchmarks
   and is rarely stated.
3. **Calibration claims validated against a measured reproduction floor**, and a schema that makes
   reporting one without the other impossible.
4. **Evidence on whether recalibration or ensembling is doing the work** (A2), which is a question
   the applied literature usually answers by assumption.

## F2.24 Future Extensions

1. **More worlds.** `S = 10–20`. The cheapest, highest-value improvement: it directly attacks
   limitations 2 and 3.
2. **Persistent-entity worlds.** A generator mode that holds the supplier population fixed across
   seeds and varies only the stochastic simulation would make per-entity `var_world` identifiable,
   lifting limitation 1 entirely. This is a moderate generator change and is the single most
   valuable extension for this feature.
3. **SWAG / Laplace on top of existing checkpoints** to improve Level 1 at near-zero cost.
4. **Conformal prediction** for distribution-free coverage guarantees, using a held-out world as the
   conformal calibration set. It would give a *guarantee* rather than an estimate, at the cost of
   less informative intervals — a genuinely attractive alternative worth measuring against HEI.
5. **Temporal calibration.** Per-snapshot or rolling recalibration to address drift.
6. **Uncertainty-aware ranking.** Rank entities by an upper confidence bound rather than by the
   point estimate, which changes which entities a planner sees first.
7. **Uncertainty on counterfactual effects** (Feature 1 × Feature 2, §4.2).
8. **OOD detection** via ensemble disagreement on structurally novel inputs, addressing limitation 7
   and composing with Feature 1's abstention rule.

---

# PART III — FEATURE 3: EXPLAINABLE HADES

*"Why did the model make this prediction?"*

---

## F3.1 Motivation

A planner shown `impact_score = 0.87` for a supplier cannot act on it without knowing what drove
it. Is it this supplier's own deteriorating on-time rate? Its position as sole source for three
components in a high-volume product? A shared customs exposure with four other suppliers that are
all degrading together? Each of those implies a different response, and the score implies none of
them.

This project has two assets that make its answer to this different from a generic explainability
add-on, and one is a warning rather than a capability.

**The asset.** `reports/layer3_testing.md` §10 built and validated a calibrated attribution module.
It attributes correlated supplier behaviour to observable causes at **AUC 0.817/0.818** with
**ECE 0.0032–0.0036** against a measured model-seed floor of 0.0021–0.0023, verified across ten
equal-count bins of **13,297 instances each**, predicted and empirical rates agreeing to within
0.011 everywhere. Just as importantly, it **declines to claim causes it has no evidence for**: it
never emitted a confidence above 0.126% for `shared_upstream` across twelve variant-by-configuration
settings, including on pairs that genuinely shared a hidden parent, and its reliability curve
confirms that near-zero is honest rather than merely small. That combination — calibrated when it
speaks, silent when it should be — is what an explanation system needs and almost never has.

**The warning.** This project has flagged, repeatedly and across multiple reports, the failure mode
of **presenting attention weights as explanations without validating faithfulness**. It is not a
hypothetical concern here: SHARE's defining architectural feature *is* a shared attention scorer
that produces a clean per-edge weight distribution over each node's entire incoming edge set. Those
weights are sitting right there, they look exactly like an explanation, and reading them as one
would be unvalidated. Feature 3 is designed against that temptation: **no explanation is presented
to a user until an ablation test confirms the cited factor actually moves the prediction.**

## F3.2 Problem Statement

Given a trained, frozen `Φ`, a snapshot `G_t`, a task `τ` and a target node `v` with prediction
`ŷ^τ_v`, produce an explanation

    E(v, tau) = { (j, phi_j, c_j) }  over factors j,

where `φ_j` is an attribution weight, `c_j` is a **calibrated confidence** that factor `j` is a
genuine contributor, and the factor set spans three levels the architecture makes natural:

1. **Input features** of `v` and of nodes in its Markov blanket;
2. **Graph structure** — which edges and which neighbours carried the signal;
3. **Semantic causes** — the observable mechanism categories §10's module already ranks
   (`regional_logistics`, `shared_sourcing`, `unknown`), lifted from pair-level to
   prediction-level.

Subject to three hard requirements:

- **Faithfulness is measured, not asserted.** Every explanation must pass an ablation test:
  removing or perturbing the cited factor must move the prediction in the claimed direction by a
  claimed amount. Explanations failing the test are **not shown**.
- **Calibrated abstention.** "I cannot explain this prediction" is a valid, first-class output, and
  the confidence attached to a factor must mean what it says.
- **No change to the prediction.** Explanation must not alter `ŷ`. It is a read-only analysis of a
  frozen model.

## F3.3 Research Gap

**In graph explainability.** GNNExplainer, PGExplainer, GraphMask and their successors optimise a
mask over edges or features to maximise mutual information with the prediction. They are almost
universally evaluated by (a) synthetic motif recovery — does the mask find the planted house/cycle
— or (b) qualitative inspection. Two structural problems follow. Motif recovery tests whether the
explainer finds a structure the *dataset designer* planted, not whether the *model* used it — a
model that ignores the motif entirely can still be "explained" correctly by a mask that finds it.
And neither evaluation produces a **calibrated confidence**: an importance score of 0.8 has no
defined frequentist meaning, so a user cannot know how often such a claim is right.

**Attention-as-explanation** has been challenged extensively in the NLP literature, and the
conclusion — attention weights are not reliably faithful — transfers directly to SHARE, whose
attention is explicitly relation-blind and shared, i.e. optimised for prediction, with no pressure
whatsoever toward interpretability.

**What is different here.** This project has an attribution module that is already *calibrated and
validated* over 13,000+ instances, with a documented abstention behaviour, and it has an established
discipline (the reproduction floor) for deciding whether a calibration number is real. The gap this
feature fills is the combination the literature lacks: **graph explanations carrying calibrated
confidences, with faithfulness measured by ablation as a gate rather than reported as a metric.**

## F3.4 Scientific Contribution

1. **Calibrated graph explanations.** Extending §10's validated pair-level attribution machinery to
   single-prediction explanation, so that "62% likely regional/logistics" carries the same
   empirically-verified meaning it does in §10 — measured with equal-count-bin ECE against a
   reproduction floor.
2. **Ablation-gated explanation delivery.** A design in which faithfulness is a *precondition for
   display*, not a reported statistic. Explanations that fail their own ablation test are suppressed
   and counted, and the suppression rate is itself a headline metric.
3. **A direct, measured comparison of attention-as-explanation against ablation ground truth** on an
   architecture whose attention is shared and relation-blind — quantifying, on this benchmark, how
   unfaithful the tempting free explanation actually is.
4. **Depth-aware structural attribution.** Explanations bounded by the Markov depth of the task, so
   a `delay` explanation cannot cite a node 3 hops away that `h^1` provably cannot see. This is a
   correctness constraint the fixed-depth readout makes exactly checkable, and it is unavailable to
   architectures with learned or unbounded receptive fields.

## F3.5 System Architecture

**Position: explanation is a post-processing module over a frozen trained model, not part of the
forward pass.** §F3.5.3 justifies this against the alternative.

Three explainers, composed, each answering a different question, plus one gate that decides whether
any of them is allowed to speak.

```
                    frozen Φ (SHARE + Markov readout + heads)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────────┐    ┌───────────────────────┐    ┌──────────────────────┐
│ E1  FEATURE       │    │ E2  STRUCTURAL        │    │ E3  SEMANTIC         │
│ ATTRIBUTION       │    │ ATTRIBUTION           │    │ ATTRIBUTION          │
│                   │    │                       │    │                      │
│ Integrated        │    │ edge/subgraph masking  │    │ REUSES §10's         │
│ Gradients over    │    │ within d(tau) hops,    │    │ calibrated hypothesis│
│ node features     │    │ learned mask           │    │ head, lifted from    │
│ (frozen model,    │    │ (GNNExplainer-style)   │    │ pair-level to        │
│ gradients w.r.t.  │    │ + attention readout    │    │ prediction-level     │
│ INPUT, not params)│    │ as a COMPARATOR only   │    │ (§F3.8.4)            │
└─────────┬─────────┘    └───────────┬───────────┘    └──────────┬───────────┘
          │                          │                           │
          └──────────────┬───────────┴───────────────────────────┘
                         ▼
          ┌──────────────────────────────────────────┐
          │  FAITHFULNESS GATE  (§F3.8.5)            │
          │  For every candidate factor:             │
          │    ablate it → re-run frozen Φ →         │
          │    does ŷ move as the explanation says?  │
          │  PASS → may be shown                     │
          │  FAIL → suppressed, counted, logged      │
          └──────────────┬───────────────────────────┘
                         ▼
          ┌──────────────────────────────────────────┐
          │  CALIBRATION LAYER                       │
          │  isotonic map from raw attribution score │
          │  to P(factor is genuinely contributing), │
          │  fitted on a held-out world against      │
          │  ablation-derived ground truth           │
          └──────────────┬───────────────────────────┘
                         ▼
              ranked, calibrated, faithfulness-gated
              explanation  +  explicit "unexplained"
              residual share
```

### F3.5.1 Why three explainers rather than one

They answer genuinely different questions and a user needs different ones at different times.
*"The on-time rate for this supplier fell"* (E1) is actionable in a different way from *"this
supplier is the only source for a component in your highest-volume product"* (E2), which is
different again from *"four suppliers you depend on share a customs exposure"* (E3). Collapsing
them into one importance vector destroys the distinction. They are ranked jointly by calibrated
confidence, with the level labelled.

### F3.5.2 Why §10's module is the semantic layer, and what has to change

§10's module answers a *pair-level* question: given two suppliers co-degrading, which observable
mechanism explains it. Feature 3 needs a *prediction-level* answer: why is this node's risk
elevated. The lift (§F3.8.4) runs the existing head over `(v, u)` pairs for `u` in `v`'s
Markov-depth neighbourhood and aggregates. Two things transfer unchanged: the observable feature
construction, and the calibrated `unknown` class that makes honest abstention possible. One thing
must be re-validated rather than assumed: the calibration was measured for the pair task, and
aggregation to node level changes the estimand, so **the lifted head requires its own ECE and its
own floor** (§F3.19.3). Reusing a calibration claim across a changed estimand would be the exact
error this project keeps documenting.

### F3.5.3 Post-processing vs in-forward-pass — the position, justified against the deployment pattern

**Position: post-processing over a frozen model.**

The deployment pattern makes this straightforward. SHARE and the Markov readout **run inference once
per snapshot** — monthly `t0`s, a batch job, not a request path. That has three consequences:

1. **There is no latency budget to protect.** Explanation cost need not fit inside a forward pass
   because the forward pass is not on a user's critical path. The main constraint an in-forward-pass
   design would relieve does not exist here.
2. **Explanations are needed for a small subset.** A planner examines the top-ranked entities, not
   all 2,000+. Computing explanations for every node on every snapshot would waste almost all of the
   work; on-demand post-processing computes only what is asked for.
3. **In-forward-pass explanation requires changing the model.** Any architecture emitting
   explanations as part of its forward pass — attention regularised for interpretability, a
   built-in mask head, a self-explaining layer — changes the trained artifact and therefore
   invalidates every result in `findings/evolution.md`, re-opening the Layer 1 and Layer 2
   validation. That cost is not remotely justified by a latency benefit that does not exist.

The one genuine cost of post-processing is that explanations are **not free**: each requires
gradients or repeated forward passes (§F3.18). Given the batch deployment pattern, this is the right
trade, and it is the same trade Feature 2 makes.

**The single exception**, and it is a cheap one: SHARE's attention weights `α` *are* produced during
the forward pass and can be cached at negligible cost. They are cached — but as a **comparator for
§F3.20's faithfulness study**, not as an explanation. The distinction is enforced in the API by
placing them behind a field explicitly named as unvalidated (§F3.15).

## F3.6 Component Diagram (ASCII)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ EXISTING, UNMODIFIED / REUSED                                                  │
│  SHARE (frozen)   Markov readout (frozen)   heads (frozen)                     │
│  ml/hypothesis_features.py   — observable pair features, assert_no_privileged  │
│  ml/hypothesis_ranker.py     — head, isotonic PAVA, ECE, reliability, roc_auc  │
│  ml/hypothesis_labels.py     — class set, decoy exclusion discipline           │
│  ml/test_hypothesis_calibration.py — 13 validated metric checks                │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────────────┐
│ NEW COMPONENTS                                                                 │
│                                                                                │
│  ┌──────────────────────────┐  ┌───────────────────────────┐                   │
│  │ FeatureAttributor  (E1)  │  │ StructuralAttributor (E2) │                   │
│  │  ml/xai/feature_attr.py  │  │  ml/xai/struct_attr.py    │                   │
│  │  - Integrated Gradients  │  │  - d(tau)-bounded subgraph│                   │
│  │  - baseline: per-type    │  │  - learned edge mask      │                   │
│  │    feature median        │  │  - attention read (COMPAR-│                   │
│  │  - completeness check    │  │    ATOR ONLY, flagged)    │                   │
│  └────────────┬─────────────┘  └────────────┬──────────────┘                   │
│               │                             │                                  │
│  ┌────────────▼─────────────────────────────▼──────────────┐                   │
│  │ SemanticAttributor (E3)   ml/xai/semantic_attr.py       │                   │
│  │  - lifts §10's pair head to node level (§F3.8.4)        │                   │
│  │  - RE-CALIBRATED for the new estimand, not inherited    │                   │
│  └────────────┬────────────────────────────────────────────┘                   │
│               │                                                                │
│  ┌────────────▼────────────────────────────────────────────┐                   │
│  │ FaithfulnessGate     ml/xai/faithful.py                 │                   │
│  │  - ablate cited factor, re-run frozen Φ                  │                  │
│  │  - sufficiency / comprehensiveness (§F3.19.1)           │                   │
│  │  - PASS/FAIL per factor; suppression is mandatory       │                   │
│  └────────────┬────────────────────────────────────────────┘                   │
│               │                                                                │
│  ┌────────────▼────────────────────────────────────────────┐                   │
│  │ ExplanationCalibrator  ml/xai/calibrate.py              │                   │
│  │  - isotonic: raw score -> P(genuine contributor)        │                   │
│  │  - ground truth = ablation outcome                      │                   │
│  │  - reuses the VALIDATED PAVA + ECE implementation       │                   │
│  └────────────┬────────────────────────────────────────────┘                   │
│               │                                                                │
│  ┌────────────▼────────────────────────────────────────────┐                   │
│  │ ExplanationAssembler   ml/xai/assemble.py               │                   │
│  │  - ranks across E1/E2/E3 by calibrated confidence       │                   │
│  │  - computes the UNEXPLAINED residual share              │                   │
│  │  - renders natural-language templates (§F3.17)          │                   │
│  └─────────────────────────────────────────────────────────┘                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

## F3.7 Data Flow Diagram (ASCII)

```
 EXPLANATION REQUEST: (snapshot t0, task tau, node v)
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ 1. Load G_t; run frozen Φ once; cache h^0..h^4, alpha, y_hat      │
 └──────────────────────────────┬───────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌─────────────┐        ┌───────────────┐       ┌────────────────┐
 │ E1 features │        │ E2 structure  │       │ E3 semantic    │
 │ IG over     │        │ mask over the │       │ §10 head over  │
 │ x_v and     │        │ d(tau)-hop    │       │ (v,u) pairs in │
 │ blanket x_u │        │ subgraph      │       │ the blanket    │
 └──────┬──────┘        └───────┬───────┘       └───────┬────────┘
        │  raw scores           │  raw scores           │  raw scores
        └───────────────┬───────┴───────────────────────┘
                        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ 2. CANDIDATE FACTOR SET  (top-N per explainer, N ~ 10)            │
 └──────────────────────────────┬───────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ 3. FAITHFULNESS GATE — for each candidate factor j:               │
 │      ablate j  ->  G'  ->  frozen Φ  ->  y_hat'                    │
 │      delta_j = y_hat - y_hat'                                     │
 │      PASS iff sign(delta_j) matches claim AND |delta_j| >= tau_f   │
 │      (tau_f measured, not chosen — see §F3.8.5)                   │
 └──────────────────────────────┬───────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ 4. CALIBRATION — isotonic( raw_score_j ) -> c_j                   │
 │    fitted offline on a HELD-OUT WORLD against ablation truth      │
 └──────────────────────────────┬───────────────────────────────────┘
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ 5. ASSEMBLE: rank by c_j; compute unexplained residual;            │
 │    if no factor passes -> "unexplained", with the reason           │
 └──────────────────────────────┬───────────────────────────────────┘
                                ▼
                     explanation record (§F3.16)
```

## F3.8 Mathematical Formulation

### F3.8.1 E1 — Integrated Gradients over input features

For target node `v`, task `τ`, input feature vector `x_u` of a node `u` in `v`'s `d(τ)`-hop
blanket, and a baseline `x̄_u`:

    IG_i(u) = ( x_u,i - xbar_u,i ) * INTEGRAL_{s=0..1}
                  d f_tau( Phi( X(s) ) )_v  /  d x_u,i   ds

approximated by a Riemann sum over `K = 50` steps, where `X(s) = X̄ + s·(X − X̄)` interpolates the
**whole feature tensor** (interpolating only `x_u` would break the path-integral's completeness
property).

**Baseline choice.** The per-node-type **feature median** over the training split, not zeros. Zero
is not a neutral input here — features are z-scored and clipped counts, so a zero vector is a
specific, sometimes extreme, entity. The median is the closest available notion of "a typical node
of this type." The baseline is a stated assumption, and §F3.20 ablation A3 tests sensitivity to it.

**Completeness check** (a correctness gate, not a metric): IG satisfies
`Σ_{u,i} IG_i(u) ≈ f_τ(Φ(X))_v − f_τ(Φ(X̄))_v`. The implementation asserts agreement within 5%; a
larger gap means too few integration steps or a discontinuity, and the explanation is rejected
rather than reported.

**Why IG rather than raw gradients or SHAP.** Raw gradients are local and saturate — a feature that
has already pushed a prediction to 0.99 has near-zero gradient despite being the reason. IG's path
integral fixes exactly that. SHAP is preferable in principle but its graph-structured variants
require an exponential number of coalitions or strong independence assumptions that a graph
violates by construction; the cost is not justified given the ablation gate is the real arbiter of
faithfulness anyway.

### F3.8.2 E2 — depth-bounded structural attribution

The candidate edge set is **bounded by the task's Markov depth**:

    S_v(tau) = { e = (a,b) in E : dist(v, a) < d(tau)  or  dist(v, b) < d(tau) }

Edges outside `S_v(τ)` **cannot** influence `ŷ^τ_v`, because `h^{d(τ)}_v` is the result of exactly
`d(τ)` rounds of message passing. This is not a heuristic prune — it is a proof, and it is available
precisely because the readout depth is *fixed and known*. An architecture with a learned or
unbounded receptive field could not make this guarantee. For `delay`, `d = 1`, so the candidate set
is just the shipment's immediate edges; for `impact`, `d = 4`.

A continuous mask `m ∈ [0,1]^{|S_v(τ)|}` is fitted per explanation by minimising

    L(m) = | f_tau( Phi( G ⊙ m ) )_v  -  f_tau( Phi(G) )_v |
           + lambda_1 * ||m||_1                    (sparsity)
           + lambda_2 * H(m)                       (entropy, drives m toward 0/1)

with `G ⊙ m` scaling each edge's message contribution by `m_e` before the attention softmax. This is
GNNExplainer's formulation, restricted to the provably-relevant subgraph. `λ_1 = 0.005`,
`λ_2 = 0.1`, 100 Adam steps at `lr = 0.01`, defaults ablated in §F3.20.

**Attention weights are computed but are not an explanation.** SHARE's per-edge `α` from the
destination-node softmax is cached and reported **only** as `attention_comparator`, and §F3.20 A6
measures its agreement with the ablation-derived ground truth. If the agreement turns out to be
high, that is a finding worth reporting — and would justify a much cheaper explainer. If it is low,
that is the measured version of the warning this project has been issuing on principle. Either way
it is measured, and until it is, `α` is never surfaced as a reason.

### F3.8.3 Factor scores are made comparable before ranking

E1, E2 and E3 produce scores on different scales (a signed feature contribution, a mask weight in
`[0,1]`, a class probability). They are **not** combined by normalising them onto a common scale —
that would invent comparability. Instead each is mapped through its **own** isotonic calibrator to a
common, meaningful quantity: `c_j = P(factor j is a genuine contributor)`, defined by the ablation
ground truth of §F3.8.5. Ranking is on `c_j`, which is the only scale on which the three levels are
legitimately comparable.

### F3.8.4 E3 — lifting §10's pair-level attribution to prediction level

For target node `v` (task `impact`, entity type `Supplier` — the task where semantic attribution is
meaningful), let `N_d(v)` be the suppliers within `v`'s Markov-depth neighbourhood that carry a
usable observable trajectory. For each `u ∈ N_d(v)` run §10's calibrated head to obtain
`P(class | v, u)` over `{regional_logistics, shared_sourcing, shared_upstream, unknown}`.

Aggregate to node level by an evidence-weighted mean over the pairs that pass §10's own detection
criterion:

    P_node(class | v) = ( sum_{u in D(v)} w_u * P(class | v,u) ) / ( sum_{u in D(v)} w_u )
    w_u = |corr_90d(v,u)|        detection strength as the weight
    D(v) = { u : u is in v's detected co-degradation set }

with `P_node(unknown | v) = 1` and all other classes zero when `D(v) = ∅` — the honest output when
there is no detected behavioural pattern to explain.

**Three constraints carried over from §10, non-negotiable:**

- `shared_upstream` is retained as a class **and is expected to be near zero**. §10 measured it at
  chance (AUC 0.4928/0.5088 on 328 positives, decoy scoring above it) and never above 0.126%
  confidence. It stays in the output because removing it would hide an honest abstention; it must
  not be presented as a finding when it appears.
- The `unknown` class is first-class, not a leftover.
- Privileged mechanism state is never an input; `assert_no_privileged_features` runs on the lifted
  feature set exactly as it does on the pair-level one.

**And one that is new**: the aggregation changes the estimand, so **the lifted head is
re-calibrated and re-floored at node level** (§F3.19.3). §10's ECE does not transfer.

### F3.8.5 The faithfulness gate — the mechanism, and how its threshold is set

For a candidate factor `j` with claimed direction `s_j ∈ {+1, −1}`, define the ablated world `G_{−j}`:

| factor type | ablation |
|---|---|
| E1 feature `(u, i)` | set `x_{u,i}` to the baseline median |
| E2 edge set | delete those edges (and their `ToUndirected` mirrors) |
| E3 semantic class | delete the edges/features carrying that class's evidence (e.g. for `shared_sourcing`, the co-parent edges linking `v` to the cited partners) |

Then

    delta_j = f_tau( Phi(G) )_v  -  f_tau( Phi(G_{-j}) )_v

and factor `j` **passes** iff

    sign(delta_j) == s_j    AND    |delta_j| >= tau_f

**`τ_f` is measured, not chosen.** It is set to the **99th percentile of |Δ| under null ablations** —
ablating randomly selected factors of the same type and cardinality that the explainer did *not*
cite. This makes the threshold a statement about the model's own sensitivity noise, and makes
"passes the gate" mean "moves the prediction more than a random ablation of the same size would."
`τ_f` is estimated per task and per factor type on a held-out world and stored with the explanation
model version. This is the same discipline as the reproduction floor, applied to ablation
sensitivity.

**Failures are suppressed and counted.** The suppression rate is a headline metric (§F3.19.2), not
an internal detail: a system suppressing 80% of its explanations is telling the user something
important about itself.

### F3.8.6 The unexplained residual

    residual(v) = 1  -  ( sum over PASSING factors j of  |delta_j| )  /  |y_hat_v - y_baseline_v|

where `y_baseline_v` is the prediction under full ablation of the candidate set. The residual is
**always displayed**. An explanation accounting for 30% of a prediction's elevation is a partial
explanation, and presenting a ranked list without the residual implies a completeness that has not
been established.

## F3.9 Neural Network Changes

### F3.9.1 What is reused, unmodified

| component | reuse |
|---|---|
| **SHARE** | unmodified, **frozen**. Gradients are taken w.r.t. *inputs* for IG, never w.r.t. parameters. No parameter is ever updated. |
| **Markov readout** | unmodified. It also *supplies the depth bound* that makes E2's candidate set provably complete — the fixed depth is load-bearing here, not incidental. |
| **Prediction heads** | unmodified; called repeatedly under ablation. |
| **§10 hypothesis module** | head architecture, observable features, isotonic PAVA, ECE, reliability, `assert_no_privileged_features` — all reused. Re-calibrated for the new estimand (§F3.8.4). |
| **`ml/test_hypothesis_calibration.py`** | reused wholesale; Feature 3's calibration metrics arrive pre-validated. |
| **SHARE's attention `α`** | read-only, cached, used **only** as a comparator (§F3.8.2). |

### F3.9.2 What is new

| component | parameters | note |
|---|---|---|
| E1 Integrated Gradients | **0** | pure analysis over the frozen model |
| E2 edge mask | `\|S_v(τ)\|` **per explanation**, discarded after | not model parameters; a per-instance optimisation variable |
| E3 node-level aggregation | **0** new; reuses §10's ~7,684-parameter head | the head is re-fitted at node level, not enlarged |
| Faithfulness gate | **0** | repeated frozen forward passes |
| Explanation calibrators | isotonic curves, ~hundreds of knots per (task, factor type) | fitted artifacts, not layers |

**Feature 3 adds no neural layers to the production model and changes no trained weights.** The only
fitted objects are per-instance masks (transient) and isotonic curves (small artifacts).

### F3.9.3 The one architectural property being exploited

The fixed-depth readout is not merely compatible with this feature — it is what makes E2's
completeness claim provable. Because `z^τ_v = h^{d(τ)}_v` with `d(τ)` fixed and known, the set of
edges that *can* influence the prediction is exactly `S_v(τ)`, and an explanation is guaranteed not
to be missing a relevant edge outside it. Under a learned or per-node-adaptive depth this guarantee
would evaporate: the receptive field would be input-dependent, and the explainer would have to
either over-approximate it or risk unsoundness. This is a concrete, previously-unremarked benefit
of the Layer 2 decision recorded in `findings/evolution.md` §3, and it is worth stating because the
adaptive-depth line of work was closed on accuracy grounds alone.

## F3.10 Loss Functions

**No training loss for the production model.** Three optimisation objectives exist, none of which
updates `Φ`:

**(a) E2 per-instance mask objective** (§F3.8.2), optimised per explanation and discarded:

    L(m) = | f_tau(Phi(G ⊙ m))_v - f_tau(Phi(G))_v |  +  lambda_1 ||m||_1  +  lambda_2 H(m)

with `H(m) = −Σ_e [ m_e log m_e + (1−m_e) log(1−m_e) ]`. The fidelity term uses the *prediction*
rather than the label: the object being explained is what the model said, not what was true.

**(b) E3 node-level head fitting.** The §10 head's multi-label objective, unchanged in form:
`BCEWithLogits` per class with capped `pos_weight`, fitted against node-level class labels derived
from the generator's privileged mechanism state — **training-and-evaluation-time only, never an
input**, with the same disclosure §10 made.

**(c) Explanation calibrator fitting.** Isotonic (PAVA) from raw factor score to the binary
ablation-derived outcome `1[factor passes the faithfulness gate]`, fitted on a held-out world.

Nothing here backpropagates into SHARE, the Markov readout, or the prediction heads. An assertion
that `requires_grad` is `False` on every backbone parameter runs at explainer start-up, for the same
reason the byte-identity checks exist elsewhere in this project: the failure would be silent.

## F3.11 Training Procedure

Only two things are fitted, and neither is the production model.

### F3.11.1 Fitting the node-level semantic head (E3)

```
FOR each world seed s_d:
    build detected co-degradation sets D(v) using §10's instrument, unchanged
    build observable pair features (ml/hypothesis_features.py, unchanged)
    derive NODE-level class labels from privileged mechanism state:
        class c is TRUE for node v iff v participates in >= 1 pair carrying c
    aggregate pair predictions to node level (§F3.8.4)
LEAVE-ONE-WORLD-OUT: train on 3 worlds, calibrate on 1, test on 1
```

The split is by **world**, exactly as §10.3 did and for the same reason: §0.5b makes world variance
dominant, and §9.8 established that this benchmark's signal is distinguishable largely by *identity*,
so a within-world split invites memorisation.

### F3.11.2 Fitting the explanation calibrators

This is the step that makes confidences mean something, and it needs ground truth. That ground truth
is the **ablation outcome**, which is expensive but obtainable:

```
FOR a sample of (v, tau) on the calibration world:            # N ~ 2,000 nodes per task
    run E1, E2, E3 -> candidate factors with raw scores
    FOR each candidate factor j:
        run the faithfulness ablation -> delta_j
        label_j <- 1[ sign matches AND |delta_j| >= tau_f ]
    accumulate (raw_score_j, label_j) per (task, factor_type)
FIT isotonic per (task, factor_type) on those pairs
ESTIMATE tau_f itself from NULL ablations on the same world (§F3.8.5)
```

Cost is dominated by ablation forward passes: `N_nodes × N_candidates` per task. At
`N_nodes = 2,000`, `N_candidates = 10`, that is 20,000 forward passes per task per world — hours,
not days, and it is a one-time offline fit. §F3.18 gives the reduction that makes it cheaper.

### F3.11.3 What must be re-measured rather than inherited

Stated explicitly because inheriting would be the easy error: **§10's ECE of 0.0032–0.0036 does not
transfer to Feature 3.** It was measured for a pair-level estimand on a detected-pair candidate set.
Feature 3's estimand is node-level and its candidate set is different. The lifted head gets its own
ECE, its own reliability curve, and its own reproduction floor (§F3.19.3). The *machinery* is
inherited and pre-validated; the *numbers* are not.

## F3.12 Inference Procedure

```
INPUT: (t0, tau, v), optional max_factors
1. Frozen forward pass on G_t; cache h^0..h^4, alpha, y_hat_v.       [1 pass]
2. Compute the depth-bounded candidate set S_v(tau) by BFS to d(tau).
3. E1: Integrated Gradients, K=50 steps.                            [K passes]
   Assert completeness within 5%; else reject E1 for this node.
4. E2: fit the edge mask, 100 steps.                                [100 passes on
                                                                     the SUBGRAPH]
5. E3: run §10's head over pairs (v,u), u in D(v); aggregate.       [cheap, no Φ]
6. Take top-N candidates per explainer (N = 10).
7. FAITHFULNESS GATE: ablate each, re-run Φ.                        [<= 3N passes]
   Suppress failures; count them.
8. CALIBRATE surviving raw scores -> c_j.
9. Rank by c_j; compute the unexplained residual (§F3.8.6).
10. If NO factor passes -> return status "unexplained" WITH the reason
    (all-failed / no-candidates / completeness-rejected), never an empty list.
OUTPUT: ranked calibrated factors + residual + suppression count + status
```

### F3.12.1 Abstention, and why it is the design's most important behaviour

Three distinct honest outcomes, and they must not be collapsed:

| status | meaning |
|---|---|
| `explained` | ≥1 factor passed the gate and carries a calibrated confidence |
| `unexplained_no_faithful_factor` | candidates were generated, none moved the prediction more than a random ablation would |
| `unexplained_no_candidates` | nothing to cite — e.g. `D(v) = ∅` and the node has a near-baseline feature vector |

This mirrors the §10 behaviour that is the strongest evidence the underlying machinery is honest:
that module declined to claim `shared_upstream` even on pairs that genuinely had one, because it had
no evidence. **An explanation system that always produces an explanation is not trustworthy**, and
the suppression rate is reported precisely so that a user can see how often the system is choosing
silence.

## F3.13 Algorithms

**Algorithm 1 — Depth-bounded candidate subgraph.** BFS from `v` on the undirected graph to depth
`d(τ)`; collect all edges with at least one endpoint at depth `< d(τ)`. Complexity
`O(|V_local| + |E_local|)`. Correctness: every edge influencing `h^{d(τ)}_v` is included, by induction
on message-passing rounds.

**Algorithm 2 — Integrated Gradients with completeness gate.** Interpolate the full feature tensor
over `K` steps; accumulate `∂ŷ_v/∂X` at each; multiply by `(X − X̄)`; assert
`|Σ IG − (ŷ(X) − ŷ(X̄))| ≤ 0.05·|ŷ(X) − ŷ(X̄)|`; on failure raise `K` once and re-run, then reject.

**Algorithm 3 — Edge-mask fitting.** Initialise `m` at 0.5 (logit 0); Adam 100 steps on §F3.10(a);
threshold at 0.5 for the reported edge set; report both the continuous mask and the thresholded set,
since the thresholding is a choice and hiding it would obscure sensitivity.

**Algorithm 4 — Null-ablation threshold estimation.** For each `(task, factor_type)`, sample 1,000
random ablations of the same type and cardinality as typical explanations on a held-out world;
compute `|Δ|`; set `τ_f` to the 99th percentile. Store with the model version.

**Algorithm 5 — Faithfulness gate.** For each candidate: construct `G_{−j}`, run `Φ`, compute
`Δ_j`, apply the sign-and-magnitude test. Batch the ablations where independent.

**Algorithm 6 — Sufficiency / comprehensiveness.** *Comprehensiveness*: remove the top-`k` cited
factors, measure the drop. *Sufficiency*: keep **only** the top-`k` (ablate everything else in the
candidate set), measure how much of the original prediction remains. Both reported as curves over
`k = 1..N`, not as single numbers (§F3.19.1).

**Algorithm 7 — Explanation calibration.** PAVA over `(raw_score, gate_outcome)` per
`(task, factor_type)`; apply by clamped interpolation. Reused, pre-validated implementation.

## F3.14 Pseudocode

```
FUNCTION explain(v, tau, G, frozen_model, xai_model, max_factors):
    ASSERT all backbone parameters have requires_grad == False
    y_hat <- frozen_model(G)[tau][v]
    S     <- depth_bounded_subgraph(G, v, markov_depth(tau))        # Algorithm 1

    cand  <- []
    ig, ok <- integrated_gradients(frozen_model, G, v, tau, K=50)   # Algorithm 2
    IF ok: cand += top_n(ig, N, type='feature')
    ELSE : record completeness_rejected

    mask  <- fit_edge_mask(frozen_model, S, v, tau)                 # Algorithm 3
    cand += top_n(mask, N, type='structure')

    IF tau == 'impact':
        sem <- semantic_node_level(v, G, hypothesis_head)           # §F3.8.4
        cand += classes_above_base_rate(sem, type='semantic')

    passed, suppressed <- [], 0
    FOR j IN cand:
        G_ablated <- ablate(G, j)
        delta_j   <- y_hat - frozen_model(G_ablated)[tau][v]
        IF sign(delta_j) == j.claimed_sign AND abs(delta_j) >= xai_model.tau_f[tau][j.type]:
            j.delta <- delta_j
            j.confidence <- xai_model.iso[tau][j.type].apply(j.raw_score)
            passed.append(j)
        ELSE:
            suppressed += 1

    IF passed IS EMPTY:
        RETURN { status: reason_for_emptiness(cand), suppressed: suppressed }

    ranked   <- sort(passed, by=confidence, desc)[:max_factors]
    residual <- 1 - sum(abs(j.delta) for j in passed) / abs(y_hat - baseline_prediction)
    RETURN { status: 'explained', factors: ranked, residual: residual,
             suppressed: suppressed, attention_comparator: cached_alpha(v) }


FUNCTION fit_explanation_calibrators(calib_world, tasks, n_nodes):
    tau_f <- FOR EACH (task, factor_type): null_ablation_percentile(99)   # Algorithm 4
    pairs <- []
    FOR v IN sample(calib_world, n_nodes):
        FOR j IN generate_candidates(v):
            delta <- ablation_delta(v, j)
            pairs.append( (j.type, j.raw_score, 1 IF passes(delta, tau_f) ELSE 0) )
    iso <- FOR EACH (task, factor_type): fit_isotonic(pairs)
    RETURN XaiModel(iso, tau_f, version)


FUNCTION evaluate_explanations(xai_model, test_world):
    out <- {}
    out.faithfulness    <- { comprehensiveness_curve, sufficiency_curve }   # Algorithm 6
    out.calibration     <- { ece, reliability, brier } on gate outcomes
    out.calibration.floor <- reproduction_floor(...)                        # MANDATORY
    out.suppression_rate<- fraction of candidates failing the gate
    out.abstention_rate <- fraction of nodes returning 'unexplained'
    out.attention_agreement <- rank_corr(cached_alpha, ablation_delta)      # the comparator
    out.by_task <- SAME, disaggregated
    RETURN out
```

## F3.15 API Design

Optional (§0.4). Offline fallback: `ml/xai/report.py` emits per-entity explanation cards plus the
§F3.19 aggregate tables as a static artifact.

```
POST /v1/explain
  request: { "snapshot_t0": "...", "world": {"variant":"B","seed":42},
             "task": "impact", "entity_id": "...",
             "max_factors": 5, "include_comparator": false }
  response:
    { "status": "explained" | "unexplained_no_faithful_factor"
                             | "unexplained_no_candidates",
      "prediction": 0.87,
      "explanation_model_version": "xai-1.0.0",
      "backbone_version": "share-markov-1.4.2",
      "factors": [
        { "level": "structure",
          "statement": "Sole source for COMP-19c2, used in PROD-4a1f (highest order volume)",
          "confidence": 0.81,
          "faithfulness": { "delta": 0.19, "threshold": 0.021, "passed": true },
          "evidence": {"edges": [["SUP-8f3a","SUPPLIES","COMP-19c2"]], "hops": 1} },
        { "level": "feature",
          "statement": "on_time_rate_90d is 0.62, well below this supplier's own history",
          "confidence": 0.64,
          "faithfulness": { "delta": 0.08, "threshold": 0.021, "passed": true },
          "evidence": {"node": "SUP-8f3a", "feature": "on_time_rate_90d",
                       "value": 0.62, "baseline": 0.89} },
        { "level": "semantic",
          "statement": "Co-degrading with 4 suppliers sharing a customs exposure",
          "confidence": 0.58,
          "faithfulness": { "delta": 0.05, "threshold": 0.021, "passed": true },
          "evidence": {"class": "regional_logistics", "partners": ["...","..."]} } ],
      "unexplained_residual": 0.31,
      "suppressed_candidates": 7,
      "attention_comparator": null,
      "caveats": [
        "Residual 0.31: roughly a third of this prediction's elevation is not accounted
         for by any factor that passed the faithfulness test.",
        "7 candidate factors were suppressed for failing the ablation test." ] }

GET  /v1/explain/faithfulness-report
  Comprehensiveness and sufficiency curves, suppression and abstention rates,
  calibration + its reproduction floor, and attention-vs-ablation agreement.

GET  /v1/explain/thresholds
  The measured tau_f per (task, factor type), with the null-ablation distribution
  it came from — so a user can see what "passed the faithfulness test" required.
```

**Design decisions.** `attention_comparator` defaults to `null` and is populated only when
explicitly requested, with a response-level note that it is **unvalidated**; making it opt-in is the
API-level enforcement of §F3.8.2. `unexplained_residual` and `suppressed_candidates` are required
fields, not optional metadata — a response cannot omit how much it failed to explain. And `status`
distinguishes the two kinds of silence, because "we found nothing faithful" and "there was nothing
to look at" mean different things to a planner.

## F3.16 Database Schema Changes

```sql
CREATE TYPE xai_factor_level  AS ENUM ('feature','structure','semantic');
CREATE TYPE xai_status        AS ENUM ('explained',
                                       'unexplained_no_faithful_factor',
                                       'unexplained_no_candidates');

CREATE TABLE explanations (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_score_id         UUID REFERENCES risk_scores(id) ON DELETE CASCADE,
    task                  VARCHAR(50) NOT NULL,
    entity_type           entity_type NOT NULL,
    entity_id             UUID NOT NULL,
    snapshot_t0           TIMESTAMPTZ NOT NULL,
    prediction            NUMERIC(6,5) NOT NULL,
    status                xai_status NOT NULL,
    unexplained_residual  NUMERIC(6,5),          -- NULL only when status <> 'explained'
    suppressed_candidates SMALLINT NOT NULL DEFAULT 0,
    explanation_model_version VARCHAR(50) NOT NULL,
    backbone_version      VARCHAR(50) NOT NULL,
    generated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (status <> 'explained' OR unexplained_residual IS NOT NULL)
);
CREATE INDEX idx_expl_entity ON explanations (entity_type, entity_id, generated_at DESC);
CREATE INDEX idx_expl_status ON explanations (status, task);

CREATE TABLE explanation_factors (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    explanation_id     UUID NOT NULL REFERENCES explanations(id) ON DELETE CASCADE,
    rank               SMALLINT NOT NULL,
    level              xai_factor_level NOT NULL,
    statement          TEXT NOT NULL,
    raw_score          NUMERIC(12,8) NOT NULL,
    confidence         NUMERIC(6,5) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    -- Faithfulness is stored per factor and is NOT NULL. A factor row cannot exist
    -- without the ablation evidence that let it be shown.
    faithfulness_delta NUMERIC(8,6) NOT NULL,
    faithfulness_threshold NUMERIC(8,6) NOT NULL,
    evidence           JSONB NOT NULL,
    UNIQUE (explanation_id, rank)
);

-- Suppressed candidates are RETAINED, not discarded. They are the record of what the
-- system considered and rejected, and the suppression rate is a headline metric.
CREATE TABLE explanation_suppressed (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    explanation_id     UUID NOT NULL REFERENCES explanations(id) ON DELETE CASCADE,
    level              xai_factor_level NOT NULL,
    raw_score          NUMERIC(12,8) NOT NULL,
    faithfulness_delta NUMERIC(8,6) NOT NULL,
    reason             VARCHAR(32) NOT NULL   -- sign_mismatch | below_threshold | completeness
);

-- The measured null-ablation thresholds, versioned.
CREATE TABLE xai_thresholds (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    explanation_model_version VARCHAR(50) NOT NULL,
    task               VARCHAR(50) NOT NULL,
    level              xai_factor_level NOT NULL,
    tau_f              NUMERIC(8,6) NOT NULL,
    null_p50           NUMERIC(8,6) NOT NULL,
    null_p99           NUMERIC(8,6) NOT NULL,
    n_null_samples     INTEGER NOT NULL,
    estimated_on_world INTEGER NOT NULL,
    UNIQUE (explanation_model_version, task, level)
);

-- Calibration of explanation confidences, with its floor. Same discipline as F2.16.
CREATE TABLE xai_calibration_reports (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    explanation_model_version VARCHAR(50) NOT NULL,
    task               VARCHAR(50) NOT NULL,
    level              xai_factor_level NOT NULL,
    n_instances        INTEGER NOT NULL,
    ece                NUMERIC(8,6) NOT NULL,
    ece_floor_max_abs  NUMERIC(8,6) NOT NULL,
    floor_kind         VARCHAR(24) NOT NULL,
    brier              NUMERIC(8,6) NOT NULL,
    reliability_curve  JSONB NOT NULL,
    suppression_rate   NUMERIC(6,5) NOT NULL,
    abstention_rate    NUMERIC(6,5) NOT NULL,
    attention_rank_corr NUMERIC(6,5),          -- the comparator finding
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`explanation_factors.faithfulness_delta NOT NULL` is the schema-level enforcement of this feature's
central rule: **a factor cannot be stored, and therefore cannot be displayed, without the ablation
evidence that justified it.** `explanation_suppressed` exists so the system's rejections are
auditable — an explanation system whose failures are invisible cannot be evaluated.

## F3.17 Dashboard / UI Design

Fallback: static explanation cards from `ml/xai/report.py`.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  WHY IS SUP-8f3a AT 0.87 IMPACT RISK?                     snapshot 2025-04-01     │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ① STRUCTURE                                              confidence  81%         │
│     Sole source for COMP-19c2, used in PROD-4a1f                                  │
│     ┌───────────────────────────────────────────────────────────────────────────┐ │
│     │  SUP-8f3a ──SUPPLIES──▶ COMP-19c2 ──USED_IN──▶ PROD-4a1f                  │ │
│     │            (only source)          (qty 4)     (highest order volume)      │ │
│     └───────────────────────────────────────────────────────────────────────────┘ │
│     ✓ verified: removing this path drops the prediction by 0.19                   │
│       (random ablations of the same size move it by ≤ 0.021)                      │
│                                                                                   │
│  ② FEATURE                                                confidence  64%         │
│     on_time_rate_90d = 0.62, against this supplier's 0.89 baseline                │
│     ✓ verified: restoring it to baseline drops the prediction by 0.08             │
│                                                                                   │
│  ③ SEMANTIC                                               confidence  58%         │
│     Co-degrading with 4 suppliers sharing a customs exposure (Germany)            │
│     ✓ verified: removing those links drops the prediction by 0.05                 │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  NOT EXPLAINED                                                     31%      │  │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░                  │  │
│  │  About a third of this prediction's elevation is not accounted for by any   │  │
│  │  factor that passed the faithfulness test.                                  │  │
│  │                                                                             │  │
│  │  7 further candidates were considered and REJECTED — they did not move the  │  │
│  │  prediction more than a random change of the same size would.  [see all]    │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  Every factor above was verified by removing it and re-running the model.         │
│  Nothing is shown on the basis of attention weights alone.                        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**UI rules that are requirements:**

1. **Every displayed factor carries its verification inline** — the measured `Δ` and the threshold
   it had to beat. Not in a tooltip.
2. **The unexplained residual is always shown, at the same visual weight as the factors.** It is not
   a footnote; it is the honesty of the whole card.
3. **Suppressed candidates are surfaced as a count with a drill-down.** Hiding rejections would make
   the system look more certain than it is.
4. **Attention weights are never shown on this card.** They live behind an explicitly-labelled
   diagnostic view for researchers, never in the planner's path.
5. **`unexplained` states render as a full card with the reason**, not as an empty panel — an empty
   panel is indistinguishable from a loading failure.
6. **No percentage is shown for a factor that did not pass the gate**, at all, anywhere.

## F3.18 Computational Complexity

Per explanation, in frozen forward passes over the **`d(τ)`-hop subgraph** (not the whole graph —
the depth bound is a real computational saving as well as a correctness property):

| stage | passes | note |
|---|---|---|
| base prediction | 1 | cached per snapshot, shared across all explanations |
| E1 Integrated Gradients | `K = 50` | forward + backward each; gradients w.r.t. inputs only |
| E2 mask fitting | 100 | on the subgraph; the dominant term for `impact` (`d=4`) |
| E3 semantic | ~0 | reuses cached §10 pair predictions; no `Φ` pass |
| faithfulness gate | ≤ `3N = 30` | batchable |
| **total** | **≈ 180** | subgraph passes per explained node |

For `delay` (`d = 1`) the subgraph is tiny and the whole procedure is milliseconds. For `impact`
(`d = 4`) on a dense supplier the 4-hop neighbourhood can approach the full graph, and this is the
worst case.

**Three properties make this acceptable given the deployment pattern:**

1. **On-demand.** Explanations are computed for the entities a planner opens, not for all of them.
2. **No latency budget.** Inference is a monthly batch job (§F3.5.3).
3. **Offline fitting is the only large cost** — ~20,000 ablation passes per task per world for
   calibrator fitting (§F3.11.2), paid once per model version.

**Reductions available if needed:** cache the base forward pass per snapshot (already specified);
reduce `K` to 20 with a completeness re-check; cap E2's mask steps at 50 with a fidelity check; and
restrict E2's candidate set to edges with non-negligible attention as a *pre-filter* — legitimate,
because narrowing the candidate set does not make an unfaithful factor pass the gate, it only risks
missing a faithful one, and that risk is measurable via §F3.20 A7.

## F3.19 Evaluation Metrics

### F3.19.1 Faithfulness — measured, as curves

| metric | definition | reported as |
|---|---|---|
| **Comprehensiveness** | `ŷ(G) − ŷ(G ∖ top-k factors)` | curve over `k = 1..N`, per task |
| **Sufficiency** | `ŷ(G with ONLY top-k retained) / ŷ(G)` | curve over `k`, per task |
| **Gate pass rate** | fraction of candidates passing the ablation test | per task, per level |
| **Sign fidelity** | fraction of passing factors whose `Δ` sign matched the claim | per task, per level |

Curves rather than single numbers because a single `k` is an arbitrary choice, and comprehensiveness
at `k=1` and `k=10` answer different questions.

### F3.19.2 Honesty metrics — the ones that make the system trustworthy

| metric | why it matters |
|---|---|
| **Suppression rate** | how often the system rejects its own candidate. A very low rate suggests the gate is too permissive; a very high one suggests the explainers are poor. Both are informative and neither is automatically bad. |
| **Abstention rate** | fraction of nodes with no faithful explanation at all. This is the direct analogue of §10's willingness to say `unknown`, and a system with an abstention rate of zero should be distrusted. |
| **Unexplained residual distribution** | how much of a typical prediction goes unaccounted for |

### F3.19.3 Calibration of explanation confidence — with its own floor

ECE (equal-count, `B=10`), reliability curve and Brier for `c_j` against the ablation outcome, per
task and per factor level, using the pre-validated implementation. **Its own reproduction floor is
mandatory** (§0.5a): fit the calibrators `n ≥ 8` times; if the fit is deterministic given fixed
inputs, report that and shift to the floor induced by re-fitting the underlying models, recording
`floor_kind` exactly as §F2.19.6 requires.

**§10's ECE is not inherited** (§F3.11.3). The lifted node-level semantic head is measured afresh.

### F3.19.4 The attention-as-explanation comparison

Rank correlation between SHARE's cached per-edge `α` and the ablation-derived `|Δ|` on the same
edges, per task. This is a **headline result of the feature**, in either direction: high agreement
would be a genuine and useful finding (attention *is* faithful in this architecture, and a cheap
explainer is available); low agreement is the measured confirmation of the warning this project has
been issuing on principle since Layer 3. Reported with a bootstrap CI over nodes and disaggregated
by task, since `delay` (`d=1`, one softmax) and `impact` (`d=4`, four composed softmaxes) are very
different cases and attention rollout is known to degrade with depth.

### F3.19.5 Mandatory disaggregation

Per task; per factor level; and by **prediction magnitude** (high-risk vs low-risk nodes) — a system
that explains confident predictions well and uncertain ones badly is behaving very differently from
one with uniform quality, and pooling hides it.

### F3.19.6 What is deliberately *not* used as a metric

**Motif recovery.** This benchmark plants known structures (hidden-parent groups, base event pools),
and it would be easy to score an explainer on whether it recovers them. That metric is rejected on
the §F3.3 grounds: it measures whether the explainer finds a structure the *generator* planted, not
whether the *model* used it — and §9.8 established that the model demonstrably does **not** use
hidden-group membership, since it cannot detect it at any depth. An explainer scored well on
recovering hidden groups would be scored well for describing something the model provably ignores.
This is a concrete instance of why ablation-based faithfulness is the right ground truth here, and it
is worth recording as a lesson the Layer 3 work paid for.

## F3.20 Ablation Studies

| # | ablation | question | expected direction |
|---|---|---|---|
| A1 | **No faithfulness gate** (show everything) | how many unfaithful explanations would reach a user? | expect a substantial fraction; quantifies the gate's value |
| A2 | **Each explainer alone** (E1 / E2 / E3) | which level carries the most faithful explanations? | expect E2 strongest for `impact`, E1 for `delay` |
| A3 | **IG baseline choice** (median / zero / random node) | how baseline-sensitive is E1? | a large sensitivity would undermine E1's reliability |
| A4 | **IG steps `K`** (10/20/50/100) | where does completeness stabilise? | sets production `K` |
| A5 | **Mask regularisation** `λ_1, λ_2` | sparsity vs fidelity trade | sets defaults |
| A6 | **Attention as the explainer** | can the free explanation replace the expensive one? | **the key comparison** (§F3.19.4) |
| A7 | **Attention pre-filter for E2's candidates** | does narrowing by attention lose faithful factors? | expect a small loss; if small, adopt for speed |
| A8 | **Depth bound removed** (allow edges beyond `d(τ)`) | sanity check: factors beyond the Markov depth must never pass the gate | **any pass is a bug**, not a finding — this is a correctness test |
| A9 | **`τ_f` at p95 / p99 / p99.9** | how does threshold strictness trade suppression against faithfulness? | sets the operating point |
| A10 | **Node-level semantic head vs §10's pair head applied naively** | does the lift need re-calibration, or would inheriting have been fine? | expect re-calibration to matter; documents why §F3.11.3 exists |

A8 deserves emphasis: it is not an ablation in the usual sense but a **falsification test of the
architecture's own guarantee**. If an edge outside `d(τ)` hops ever produces a `|Δ|` above `τ_f`,
either the depth bound is wrong, the Markov readout is not doing what it claims, or the ablation is
leaking. All three are defects.

## F3.21 Failure Modes

| # | failure | detection | mitigation |
|---|---|---|---|
| 1 | **Attention presented as explanation** — the named failure this feature exists to prevent | opt-in API field; UI rule 4; A6 measures it | structurally prevented: `α` cannot reach the planner's card |
| 2 | **Plausible but unfaithful explanation** | the gate itself; A1 quantifies what it catches | suppression is mandatory, not advisory |
| 3 | **Explaining a wrong prediction convincingly** — the model is wrong, the explanation faithfully explains *why it is wrong*, and the user believes both | Feature 2's uncertainty on the same card (§4.3) | display prediction uncertainty alongside the explanation; a confident explanation of a low-confidence prediction must be visually marked |
| 4 | **Ablation off-manifold** — removing an edge creates a graph the model never trained on, so `Δ` reflects distribution shift rather than the factor's contribution | compare edge-deletion against edge-*reweighting* (soft mask) on the same factors | report both; where they disagree materially, flag the factor as `off_manifold_uncertain` |
| 5 | **Isolated-node artifacts** — ablation isolates the target node entirely | detect zero-degree after ablation | fall back to soft masking for that factor |
| 6 | **Inherited calibration** — §10's ECE quoted for the lifted head | §F3.11.3 requires re-measurement; A10 quantifies the difference | separate `xai_calibration_reports` rows per level |
| 7 | **Explanation instability** — near-identical inputs producing very different explanations | stability audit: perturb a feature by ±1%, measure rank correlation of the factor list | report a stability score; unstable explanations are suppressed |
| 8 | **Residual ignored by users** | it is a required field and a required UI element at equal visual weight | design-level, and enforced by the schema `CHECK` |
| 9 | **Semantic layer over-claiming `shared_upstream`** | §10 measured it at chance; any nonzero confidence here is suspicious | retain the class, expect ~0, alert if it ever exceeds a small threshold — a regression indicator, not a discovery |
| 10 | **Cost pressure disables the gate** in deployment | `faithfulness_delta` is `NOT NULL` in the schema | a factor without ablation evidence cannot be persisted or served |

## F3.22 Limitations

1. **Explanations are of the model, not of the world.** A faithful explanation says why *the model*
   predicted what it did. If the model is wrong, the explanation faithfully explains an error. This
   is the fundamental limit of all post-hoc explainability and it is not solved here.
2. **Ablation is off-manifold.** Deleting an edge produces a graph outside the training
   distribution, so `Δ` mixes the factor's contribution with distribution shift (failure mode 4).
   Soft masking mitigates but does not eliminate it. This is a known, unsolved problem in the field.
3. **The semantic layer only covers `impact`.** §10's machinery is supplier-pair-based; `delay`
   (Shipment) and `shortage` (Product) have no equivalent, so they get E1 and E2 only.
4. **`shared_upstream` will effectively never be citable**, because §9.8 established the model
   cannot detect it. The explanation system inherits that limit exactly, and correctly.
5. **Explanations are per-prediction, not global.** No global model summary is produced; §F3.24
   lists it as an extension.
6. **Human interpretability is asserted, not yet measured.** The natural-language templates are
   designed for a supply-chain planner but no human study has been run. §F3.19 measures faithfulness
   and calibration, which are necessary but not sufficient for usefulness — §F3.23's protocol is
   specified but unexecuted.
7. **Cost scales with `d(τ)`.** `impact` at `d=4` on a dense supplier approaches whole-graph cost.
8. **Suppression could hide real factors.** A factor genuinely contributing less than `τ_f` is
   correctly suppressed but really did contribute; the system is biased toward silence, which is the
   right bias for trust and the wrong one for completeness. The residual makes it visible.

## F3.23 Expected Research Contributions

1. **Ablation-gated explanation delivery** — treating faithfulness as a precondition for display
   rather than a reported metric, with the suppression rate as a first-class output.
2. **Calibrated graph explanation confidences**, validated by equal-count ECE against a measured
   reproduction floor — extending §10's validated protocol to a new estimand and re-measuring rather
   than inheriting.
3. **A measured verdict on attention-as-explanation** for a shared, relation-blind attention
   architecture, per task and per depth (§F3.19.4).
4. **Depth-bounded structural attribution with a completeness guarantee**, showing a concrete benefit
   of the fixed-depth readout that the adaptive-depth investigation never had reason to look for.
5. **A methodological argument, with evidence, against motif-recovery evaluation** (§F3.19.6): this
   benchmark contains planted structures the model provably cannot detect, so recovering them would
   score an explainer for describing something the model ignores.

**Human evaluation protocol** (specified, not yet run): 8–12 participants with supply-chain
operational experience; 30 explanation cards each, balanced across tasks and confidence levels;
three measures — *decision accuracy* (can the participant identify the correct mitigating action),
*calibration of trust* (does self-reported confidence track the system's calibrated confidence), and
*detection of unfaithfulness* (synthetic cards with deliberately unfaithful factors inserted; can
participants tell?). The third is the important one, and it is rarely tested: an explanation system
whose users cannot distinguish its good explanations from its bad ones has not achieved
interpretability regardless of its metrics.

## F3.24 Future Extensions

1. **Global explanations** — model-level summaries (which relations matter most for each task,
   aggregated over entities) rather than per-prediction ones.
2. **Counterfactual explanations** — "this supplier would drop below 0.5 risk if it had two more
   qualified sources." This is Feature 1 × Feature 3 and is the most natural composition (§4.3).
3. **Contrastive explanations** — "why this supplier rather than that one," which is often what a
   planner actually asks.
4. **On-manifold ablation** — replace deleted structure with generator-plausible substitutes rather
   than deleting outright, attacking limitation 2 directly. The generator makes this feasible in a
   way most projects cannot match.
5. **Uncertainty-aware explanations** — Feature 2's interval displayed on the same card, addressing
   failure mode 3 (§4.3).
6. **Explanation for `shortage` and `delay` at the semantic level** — requires building the
   product-level and shipment-level analogues of §10's pair module.
7. **Execute the human evaluation** in §F3.23.
8. **Learned explainer amortisation** (PGExplainer-style) — train a model to predict masks in one
   pass instead of optimising per instance, trading fidelity for a large speed-up. Worth it only if
   §F3.18's reductions prove insufficient, and it must be held to the same gate.

---

# PART IV — CROSS-CUTTING CONCERNS

---

## §4.1 Shared infrastructure, and the order to build it in

The three features overlap more than their independence suggests. Building them in the wrong order
duplicates work; building the shared pieces first makes each feature cheaper than the last.

| shared component | F1 | F2 | F3 | build once in |
|---|---|---|---|---|
| **Frozen-backbone harness** (load checkpoint, assert `requires_grad=False`, run `Φ`, expose `h^0..h^4` and `α`) | ✓ | ✓ | ✓ | Feature 2 (needs it first and simplest) |
| **Isotonic / ECE / reliability** (already exists and is validated in `ml/hypothesis_ranker.py` + `ml/test_hypothesis_calibration.py`) | ✓ (gate calibration) | ✓ | ✓ | **already built** — reuse, do not reimplement |
| **Reproduction-floor runner** (repeat a fit `n≥8`, report mean/max abs deviation, identify `floor_kind`) | ✓ | ✓ | ✓ | Feature 2 |
| **Ensemble/checkpoint registry with hash integrity** | ✓ (versioning) | ✓ (core) | ✓ (versioning) | Feature 2 |
| **Depth-bounded BFS neighbourhood** (`d(τ)`-hop reach) | ✓ (reach features) | — | ✓ (candidate set) | Feature 1 |
| **Graph editor** (typed, validity-checked edits on `HeteroData`) | ✓ (core) | — | ✓ (ablation) | Feature 1 |
| **Observable feature builder + `assert_no_privileged_features`** (exists) | ✓ | ✓ (strata) | ✓ | **already built** — reuse |
| **Static report renderer** (the non-UI fallback) | ✓ | ✓ | ✓ | Feature 2 |

**Recommended build order: Feature 2 → Feature 3 → Feature 1.**

This is deliberately *not* the order the features are specified in, and the reason is cost and risk:

- **Feature 2 first.** Lowest risk (no new neural layers at all), lowest new cost (the `S` axis
  already exists), and it builds three of the shared components. It also delivers the calibration
  discipline the other two depend on. If it fails, it fails cheaply.
- **Feature 3 second.** Needs the frozen-backbone harness and the floor runner from F2, reuses §10's
  already-validated machinery, and needs the graph editor only in its simplest form (ablation, which
  is edge deletion — a strict subset of F1's editor). It also delivers the attention-faithfulness
  verdict, which is independently valuable.
- **Feature 1 last.** Highest cost (4,000 world generations), highest risk (the CRN work in §F1.8.4
  is the single hardest engineering task in this document and gates everything downstream), and it
  benefits most from the editor and reach machinery being already exercised by F3.

A reader who intends to build only one should build Feature 2. A reader who wants the most novel
result should build Feature 1, and should budget the CRN work generously.

## §4.2 Composition: Feature 1 × Feature 2

**Uncertainty on counterfactual effects.** The natural composition, and the one a planner most
needs: `Δ̂ = −0.24` is far less useful than `Δ̂ = −0.24 [−0.31, −0.09]`.

Mechanically this is straightforward — train `|𝔈|` CF-heads, one per backbone ensemble member, and
decompose `Var[Δ̂]` exactly as §F2.8.1 does. **Two cautions specific to this composition:**

1. **The variance decomposition changes meaning.** `var_world` for a counterfactual effect is
   variance across *generated worlds*, but a counterfactual is already defined relative to one
   specific world. The quantity is "how much would this effect estimate differ had the world been
   drawn differently," which is coherent but is **not** the same as an interval on the effect in
   *this* world. It must be labelled accordingly, and the §F2.5.2 identifiability caveat applies
   with equal force.
2. **Feature 1's abstention and Feature 2's uncertainty must not be conflated.** "Out of support"
   (§F1.12.1) and "wide interval" are different states: the first says the question is unanswerable,
   the second says the answer is imprecise. Presenting a wide interval where an abstention is
   warranted would be a regression in honesty.

## §4.3 Composition: Feature 1 × Feature 3, and Feature 2 × Feature 3

**Counterfactual explanations (F1 × F3).** "This supplier's risk would fall below 0.5 with two more
qualified sources" is an explanation *and* an intervention, and it is the most useful single output
this document's three features can jointly produce. The implementation is F3's explainers run over
F1's effect rather than over the prediction: which parts of the intervention drove the effect. The
faithfulness gate transfers directly — ablate the cited part of the intervention, re-score the
effect, confirm it moves.

**Uncertainty-aware explanations (F2 × F3).** This composition addresses F3's failure mode 3
(convincingly explaining a wrong prediction) and it should be treated as close to mandatory rather
than optional. A high-confidence explanation attached to a high-uncertainty prediction is the most
dangerous single artifact this system can produce, because the explanation's specificity lends
unearned credibility to the number. The UI requirement: **when `var_total` places a prediction in
the top decile of uncertainty, the explanation card must be visually marked**, and the marking must
say that the prediction being explained is itself poorly determined.

## §4.4 Versioning and provenance across all three features

Every artifact produced by any feature carries four versions, and any mismatch is a hard error:

```
backbone_version        — SHARE + Markov + heads checkpoint set (or ensemble registry hash)
feature_model_version   — cf-head / uncertainty-model / xai-model
corpus_version          — F1 only: the intervention corpus the head was trained on
config_hash             — architecture + training config, must match the backbone's
```

The rule is that **a downstream artifact is invalid against a backbone it was not fitted over.** A
CF-head trained on backbone `v1.4.2` scoring predictions from `v1.5.0` is not a degraded result, it
is a meaningless one, and the serving path must refuse rather than warn.

## §4.5 What this document does not address

Stated so that absence is not read as oversight:

- **Retraining cadence and drift monitoring** for the production backbone. Out of scope; these
  features assume a frozen, versioned backbone.
- **Authentication, authorisation, multi-tenancy, rate limiting** on the proposed APIs. All three API
  sections specify shape and semantics, not operational concerns.
- **Real-data transfer.** Every claim in this document is scoped to HADES-Bench's synthetic
  generator. Whether any of it transfers to a real supply chain is untested and untestable here —
  the standing caveat on this project's every result.
- **Optimal intervention search** (§F1.24 item 4), **global explanations** (§F3.24 item 1) and
  **conformal prediction** (§F2.24 item 4) are named as extensions, not specified.
- **The Layer 3 discovery question.** Closed per `findings/evolution.md` §5. No feature here reopens
  it, and the one capability retained from that line of work — calibrated attribution to *observable*
  causes — is used by Feature 3 for exactly what it was validated for and nothing more.

## §4.6 Consolidated risk register

| risk | feature | severity | the thing that catches it |
|---|---|---|---|
| CRN not established; RNG divergence read as effect | F1 | **critical** | null-intervention byte-identity gate (§F1.8.4); blocks the corpus |
| Model-only uncertainty presented as "confidence" | F2 | **critical** | mandatory scope labels; `world_share` as a required field |
| Attention surfaced as explanation | F3 | **critical** | opt-in API field, UI rule 4, schema `NOT NULL` on ablation evidence |
| Calibration claimed without a floor | F2, F3 | high | `NOT NULL` floor columns in both calibration tables |
| Explanation faithful to a wrong prediction | F3 | high | F2 × F3 composition (§4.3); uncertainty marking |
| Effect below re-simulation noise reported as effect | F1 | high | `ε` set above measured noise; "no detectable effect" wording |
| `var_world` from 3–5 worlds over-trusted | F2 | medium | bootstrap CI; suppression when uninformative |
| Off-manifold ablation | F3 | medium | soft-mask comparison; `off_manifold_uncertain` flag |
| Version mismatch between head and backbone | all | medium | §4.4 hard error |
| Cost pressure degrading a safeguard (ensemble to 1 member, gate disabled) | F2, F3 | medium | make degradation *visible* in required output fields rather than silent |

---

## Appendix A — Requirements traceability

Confirmation that every element the brief required appears, with its location.

| requirement | F1 | F2 | F3 |
|---|---|---|---|
| Motivation | §F1.1 | §F2.1 | §F3.1 |
| Problem Statement | §F1.2 | §F2.2 | §F3.2 |
| Research Gap | §F1.3 | §F2.3 | §F3.3 |
| Scientific Contribution | §F1.4 | §F2.4 | §F3.4 |
| System Architecture | §F1.5 | §F2.5 | §F3.5 |
| Component Diagram (ASCII) | §F1.6 | §F2.6 | §F3.6 |
| Data Flow Diagram (ASCII) | §F1.7 | §F2.7 | §F3.7 |
| Mathematical Formulation | §F1.8 | §F2.8 | §F3.8 |
| Neural Network Changes — explicit reuse vs new | §F1.9 | §F2.9 | §F3.9 |
| Loss Functions | §F1.10 | §F2.10 | §F3.10 |
| Training Procedure | §F1.11 | §F2.11 | §F3.11 |
| Inference Procedure | §F1.12 | §F2.12 | §F3.12 |
| Algorithms | §F1.13 | §F2.13 | §F3.13 |
| Pseudocode (no runnable code) | §F1.14 | §F2.14 | §F3.14 |
| API Design | §F1.15 | §F2.15 | §F3.15 |
| Database Schema Changes | §F1.16 | §F2.16 | §F3.16 |
| Dashboard / UI Design | §F1.17 | §F2.17 | §F3.17 |
| Computational Complexity | §F1.18 | §F2.18 | §F3.18 |
| Evaluation Metrics | §F1.19 | §F2.19 | §F3.19 |
| Ablation Studies | §F1.20 | §F2.20 | §F3.20 |
| Failure Modes | §F1.21 | §F2.21 | §F3.21 |
| Limitations | §F1.22 | §F2.22 | §F3.22 |
| Expected Research Contributions | §F1.23 | §F2.23 | §F3.23 |
| Future Extensions | §F1.24 | §F2.24 | §F3.24 |

**Feature-specific requirements from the brief:**

| requirement | where |
|---|---|
| Layer 2 named "Markov Blanket depth readout" throughout, never "Adaptive" | §0.2, and consistently thereafter |
| No feature assumes or reintroduces per-node depth adaptivity | §0.2; F3.9.3 explicitly benefits from the *fixed* depth |
| Each feature explicit on what reuses SHARE/Markov unmodified vs. what is new | §F1.9.1–2, §F2.9.1–2, §F3.9.1–2 |
| **F1:** generator produces interventional ground truth; evaluation built around re-simulation | §F1.4.1, §F1.8.4, §F1.11.1, §F1.19 |
| **F1:** explicit position on whether a supervised model can answer an intervention question | §F1.9.4 — position stated, four-part justification, falsifiable via Tier 1 |
| **F1:** counterfactual graph generation, intervention representation, graph editing, message-passing changes | §F1.8.1–2, §F1.13 Alg. 2, §F1.9.3 (no change needed, with two cautions) |
| **F2:** address dataset-seed variance being up to 10.3x model-seed variance | §F2.1, §F2.3, §F2.5.1–2, §F2.19.3, §F2.22.1 |
| **F2:** state plainly if a method captures only model-level variance | §F2.5.3 (per method), §F2.12.1 (scope labels), §F2.21.1 |
| **F2:** whether existing 5-dataset-seed infrastructure is reusable as ensembling material, and whether that alone suffices | §F2.5.4 — yes for Level 2, with four stated reasons it only partly suffices |
| **F2:** Bayesian vs frequentist, Deep Ensembles, MC Dropout, Evidential compared against this data | §F2.5.3 |
| **F2:** reproduction-floor methodology reused to validate calibration claims | §F2.19.6, §F2.13 Alg. 5, schema `NOT NULL` floor columns §F2.16 |
| **F3:** generalise §10's calibrated attribution from pair-level to prediction-level | §F3.5.2, §F3.8.4, §F3.11.3 (re-calibration required) |
| **F3:** concrete mandatory ablation-based faithfulness test | §F3.8.5, gate in §F3.12, schema enforcement §F3.16 |
| **F3:** GNNExplainer/PGExplainer/GraphMask/IG/attention-rollout compared | §F3.3, §F3.5, §F3.8.1–2, §F3.19.4, §F3.24.8 |
| **F3:** faithfulness/sufficiency/comprehensiveness measured, not asserted | §F3.19.1 (curves), §F3.20 |
| **F3:** explanation during forward pass or post-processing, justified against the deployment pattern | §F3.5.3 — post-processing, justified on the once-per-snapshot batch pattern |
| **F3:** human evaluation protocol | §F3.23 |

---

## Appendix B — Open questions this specification does not settle

Recorded because a specification that pretends to have resolved everything is less useful than one
that marks its own uncertainty.

1. **Will Tier 2 beat Tier 1 on `impact`?** §F1.9.4 argues it should on principle, but `impact`'s
   effect may be dominated by graph reachability that a forward pass already represents. Genuinely
   uncertain, and §F1.20 A1 is designed to settle it.
2. **Is `var_world` per stratum stable enough to be useful?** With `S ∈ {3,5}` it may be too noisy
   to act on. §F2.19.5 will show this; if not, the honest fallback is a single global inflation
   factor per task.
3. **Is SHARE's attention faithful?** Unknown. §F3.19.4 measures it. A high-agreement result would
   simplify Feature 3 substantially and would be a genuine surprise given the literature.
4. **What suppression rate is "right"?** §F3.19.2 measures it, but there is no principled target.
   Too low means a permissive gate; too high means weak explainers. Judgement will be required.
5. **Does the `A5` (inventory) construction-time limitation make that action class useless, or merely
   attenuated?** §F1.8.3 predicts the effect is largely washed out; if it is *entirely* washed out,
   `A5` should be dropped until the mid-timeline extension exists.
6. **Do counterfactual effects compose?** §F1.20 A8. If they do not, single-intervention scoring is
   the only defensible product and combinatorial planning is out of reach without much more corpus.

---

*End of specification.*
