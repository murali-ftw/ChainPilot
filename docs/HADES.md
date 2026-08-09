# HADES — Original Proposed Architecture (Pre-Pivot Reference)

**Status: historical/reference document.** This describes HADES exactly as it was originally
proposed in `docs/project_HADES.md` — HGT as the structural encoder, a full Transformer
("T1"/"short head") for per-node depth selection, and a global attention module ("T2"/"long
head") for cross-supplier hidden-dependency discovery — **before** any of the four claims were
actually tested. It is preserved here as the design the project's real, measured results (Layer
1/2/3, `reports/info.md`) are compared against, not as a description of what was ultimately
built. Where the production system diverged (SHARE replacing HGT, the gate replacing the full
T1 Transformer, Confidence-Aware Fusion as T2's chosen fusion mechanism, Claim B validated as
independent rather than scoped to `fulfilment_preference_weight` alone), that is Layer 1/2/3's
story, not this one.

---

## 1. Why this shape — the three-gap framing

The original proposal starts from five gaps a plain type-aware GNN cannot close on its own,
and assigns each to one HADES component:

| # | Gap | Why message passing can't close it | Closed by |
|---|---|---|---|
| 1 | Cannot couple nodes with no path between them | Message passing walks edges; no edge, no message, at any depth | **T2** |
| 2 | One depth serves three differently-positioned prediction targets | A single final-layer output is used for every task | **Structural prior + depth gate** |
| 3 | Deep enough for one task is over-smoothed for another | Repeated averaging blurs local detail the shallow task needs | **Intermediate-layer retention + gate** |
| 4 | The model produces a global score; real risk is relational (this firm's exposure, not the supplier's fragility in the abstract) | The node is the encoder's unit of prediction — no architectural slot for a (supplier, buyer) pair | **Claim B (dyadic reweighting)** |
| 5 | Cannot distinguish "no upstream" from "unknown upstream" | Absent edges carry no signal either way | **`is_frontier` + T2** |

**"HGT is the right encoder and the wrong complete system."** Every other component exists
because HGT solves exactly one problem — type-aware local structure — and nothing else.

---

## 2. Component 1 — HGT (the original structural encoder)

**What it is.** Heterogeneous Graph Transformer (Hu, Dong, Wang & Sun, 2020) — every one of
the graph's meta-relations (a `(source type, relation, target type)` triple, not just a
relation name — `⟨Shipment, SHIPS_FROM, Supplier⟩` and `⟨Shipment, SHIPS_FROM, Factory⟩` are
different meta-relations under HGT, even though GAT would collapse them) gets its own fully
independent attention matrix `W_ATT_r` and message matrix `W_MSG_r`.

**Why it was the original choice.** No hand-designed meta-paths required (unlike
HAN/metapath2vec) — HGT learns which paths matter directly. First-class PyTorch Geometric
support (`HGTConv`). A residual connection at every layer means each node keeps its own
identity rather than being overwritten, which is why HGT tolerates depth better than plain
GNNs.

**Cost, as originally specified.** `d=64` (the design-time anchor), `L=4`: **698,448 params**
for the encoder alone (154,122 per layer at `d=64`; would rise to 174,612/layer, +13.3%, once
the leakage-safe feature/label contract was finalized), **1.311 GFLOP** for a full-graph
forward pass — 90.7% of the entire originally-specified model's total compute.

**Why it was later superseded (not this document's story, but flagged for context).** A
four-round ablation found HGT paying for a fully dedicated per-relation parameter set on all 20
meta-relations, including four data-thin relations where that dedication overfits. SHARE (RGCN
basis-sharing + one shared attention scorer) closed HGT's own remaining advantage (impact) to a
statistical tie while winning delay and shortage outright, at a fraction of the parameter cost
— see `reports/info.md` Layer 1 for the full, real-numbers account. This document keeps HGT as
originally specified; it is not updated to describe SHARE.

---

## 3. Component 2 — T1, the full depth Transformer ("short head")

**The problem it was proposed to solve.** HGT's encoder computes `h¹…h⁴` and, by default, only
the final layer `h⁴` is used by every task — implicitly assuming every node and every task
needs exactly four hops. The original proposal's diagnosis: that assumption is wrong, and wrong
in a way that matters, because delay's target sits close (a supplier's own local neighborhood)
while impact's target sits far (spanning to Order/Customer).

**The structural prior — where the original design gets its starting depth from.** A node's
Markov blanket (parents, children, co-parents — the *other* causes of its children) is,
loosely, the smallest set of neighbors such that knowing them adds no more information than
knowing the whole graph. Applied to a Supplier node: children (components it supplies, 1 hop),
co-parents (other suppliers of those same components, 2 hops — requires the reverse `SUPPLIES`
edge and dual-sourced data), parents (upstream suppliers, 1 hop if `SUB_SUPPLIES` data exists),
and "moralised" co-parents (suppliers sharing an *unobserved* upstream — infinite hops, not
reachable by any depth — this is explicitly T2's job, not T1's). The co-parent term gives a
derived floor of `L=2`, counted directly from the graph rather than guessed.

**Per-task prior weights**, `p_head[k] = exp(−|k − b_head|/τ) / Σ_j exp(−|j − b_head|/τ)`,
`τ=0.8`:

| Head | Blanket depth `b` | `p[1]` | `p[2]` | `p[3]` | `p[4]` |
|---|---|---|---|---|---|
| Delay | 2 | 0.173 | **0.604** | 0.173 | 0.050 |
| Shortage | 3 | 0.050 | 0.173 | **0.604** | 0.173 |
| Impact | 3 | 0.050 | 0.173 | **0.604** | 0.173 |

**The "short head" itself — a full 2-layer Transformer, one per task**, treating each node's
four saved depths `[h¹, h², h³, h⁴]` as a 4-token sequence and running multi-head QKV attention
plus a feed-forward network over it to produce the task's blended representation. This is what
"short head" refers to: short not in depth of the *graph*, but in sequence length — 4 tokens
per node, a genuinely small Transformer relative to a language model, but still dense
projections applied to every node, discarding the sparsity that makes GNNs cheap in the first
place.

**Cost, as originally specified** — the reason this component did not survive to production
unmodified:

| Version | Params (×1 head) | GFLOP (×1) | Params (×3 heads) | GFLOP (×3) |
|---|---|---|---|---|
| 2-layer Transformer, ff=4 (**the original "short head" proposal**) | 99,968 | 3.93 | **299,904** | **11.80** |
| 1-layer, ff=4 | 49,984 | 1.97 | 149,952 | 5.90 |
| 1-layer, attention only | 16,960 | 0.66 | 50,880 | 1.97 |
| Gate (the eventual, cheaper substitute — not this document's subject) | 2,212 | 0.02 | 6,636 | 0.07 |

Three full T1 Transformers, one per task, would cost 11.8 GFLOP to fuse four vectors —
**roughly nine times the entire HGT encoder's own cost** — judged correctly by FLOPs (dense
projections on every node × every token × every layer), not by the comparatively modest
parameter count.

**The KL anchor (part of the original design regardless of gate-vs-Transformer choice).**
`L_total = L_task + λ · Σ_heads E_v[KL(w_v ‖ p_head)]`, `λ=0.01` — a training penalty pulling
the learned depth-blend weights back toward the structural prior above, so training can only
move away from the theorized depth deliberately, not drift arbitrarily.

---

## 4. Component 3 — T2, the global attention module ("long head")

**The problem it was proposed to solve — the one gap scale cannot close.** Two suppliers fail
in the same week, sharing no edge chain between them — different components, different
products, different customers, both quietly buying from the same unrecorded tier-2 source.
Message passing cannot help at any depth here: it walks edges, and a 4-layer encoder and a
40-layer encoder are equally blind to a pair with no path connecting them at all. This is a
structural limit, not a capacity limit.

**Why "long head."** Where T1 operates over a short, fixed 4-token sequence per node, T2
operates over the *entire* population of same-type nodes — the "long" end of the pairing, a
genuinely global attention pass, computing `A = softmax(QKᵀ/√(d/h))` over Supplier embeddings
with **no adjacency mask** at all. Message passing asks "who are you connected to?" — T2 asks
"who do you resemble?"

**The same-type constraint — mandatory, not optional.** T2 attends Supplier↔Supplier only,
never Supplier↔Product. HGT spends the majority of its parameters keeping each node type's
embedding space distinct; unconstrained global attention across all types would partially undo
that work. The coupling the mechanism cares about is specifically supplier-to-supplier.

**Bounding the candidate pool — and one proposed bounding rule flagged as wrong even at design
time.** Dense attention is `O(N_s²·d)` — at 20,000 suppliers, a 45.5x cost difference between
dense and a bounded top-k=64 pool. An earlier draft suggested bounding by "co-suppliers within
the same `component_type`" — the original document itself flags this as wrong, since
same-`component_type` co-suppliers are already 2-hop reachable via the co-parent term (T1's
job), so restricting T2 to them would confine it to couplings HGT can already see, defeating
its purpose. The corrected rule: top-k=64 by embedding cosine similarity (the same criterion
attention itself would use to score a pair), always include frontier nodes regardless of rank,
and threshold before persisting to keep `hidden_dependency_links` reviewable.

**Frontier nodes — T2's second job.** A node with `is_frontier=true` (no recorded upstream) is
an orphan to message passing — no in-edges, nothing to aggregate. In embedding attention it is
a full participant, since similarity requires no edges at all. Frontier nodes are precisely the
ones T2 exists to serve.

**The output contract — the original document's own unresolved interface question.** T2
attends Supplier↔Supplier and therefore only ever refines Supplier embeddings — it produces
nothing for Product, Warehouse, Order, or Customer, so how does its output reach the shortage
or impact heads at all? Three options were considered:

| Option | Mechanism | Cost | Original verdict |
|---|---|---|---|
| **1 — Scope to delay only** ★ | Wire T2 exclusively to the supplier-level delay head | 0 | **"HADES v1 takes Option 1."** Honest about what T2 does; defers the interface question until T2 has proven it finds anything |
| 2 — One propagation layer after T2 | `HGT ×4 → gate → T2 → HGT ×1 → heads` | +174,612 params | Principled, does not violate the no-feedback-into-the-encoder rule, but unbuilt in the original spec |
| 3 — Same-type T2 per target type | Separate Product↔Product, Order↔Order modules | ×N cost | No motivating story — the hidden-coupling problem is specifically supplier-to-supplier |

**Worked example from the original document.** Supplier S1 (Germany, connectors) and S2
(Malaysia, casings) — no shared component, product, or customer, effectively infinite graph
distance. With multi-window temporal features, both embeddings independently encode "90-day
on-time rate falling, trend slope negative, lateness variance rising" — a resemblance T2 can
find and message passing structurally cannot.

---

## 5. Component 4 — Claim B (mentioned for completeness, not this document's focus)

Reweights a supplier's global risk score by this specific customer's exposure to it
(`dyadic_risk = global_risk × f(order_volume_share, contract_priority_weight,
fulfilment_preference_weight)`) — deliberately **not** a network component, since the node
remains the encoder's unit of prediction with no architectural slot for a (supplier, buyer)
pair. Full original specification: `docs/10_AI_ML_Documentation.md` §8.5. Real, tested result:
`reports/step8_claim_b.md`.

---

## 6. What this document is not

This is the architecture **as proposed**, derived analytically before any code existed against
a real graph. It does not describe what was actually built and measured — `docs/project_HADES.md`
itself notes the original numbers were "derived analytically from the specification... the
relative orderings are robust... the absolute values depend on `d`, `L`, real graph size and
real degree distribution, and should be recomputed the moment you have a graph," which is
exactly what Layers 1–3 (`reports/info.md`, `reports/phase3.md`) did. For the real, measured
architecture — SHARE in place of HGT, the bounded-residual gate (Markov Floor / Variant A) in
place of the full T1 Transformer, and Confidence-Aware Fusion as T2's adopted fusion design —
see `reports/info.md`.
