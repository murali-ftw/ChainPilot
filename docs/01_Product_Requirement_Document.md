# Document 1 — Product Requirement Document (PRD)

## HADES Model-Development Prototype

Version: 2.0 — rescoped from a full product build to a standalone model-development prototype
Status: Baseline

---

## 1. Purpose

This is not a product. It is a **research prototype whose only deliverable is a trained HADES model and honest, statistically-grounded evidence for or against its four design claims** (Section 4). Everything a product build would need — a dashboard, a chatbot, an approval workflow, an execution layer — is out of scope and not documented here. This document exists to keep that scope explicit and to specify what the prototype actually has to do to answer its research question.

## 2. Scope

### 2.1 In Scope

- A leakage-safe synthetic dataset and its as-of feature/label contract (`db/`, `05_Database_Design.md`).
- Graph construction: an 8-node-type, 20-meta-relation (10 forward + 10 reverse) heterogeneous graph, assembled per snapshot (`06_Graph_Database_Design.md`).
- The HADES architecture (`project_HADES.md`, restated for implementation in `10_AI_ML_Documentation.md`): HGT encoder, structural depth prior + learned depth gate, Transformer 2 global attention, Claim B dyadic reweighting.
- Training: the layer-depth sweep, the architecture ablation (GraphSAGE/GAT/HGT with a matched-parameter control arm), and, as time allows, the depth gate / Transformer 2 / Claim B builds.
- Evaluation: metrics with confidence intervals, per-claim validation routes, a leakage test, and explainability as a diagnostic tool.
- Model governance: `model_registry` and `model_evaluation_runs` so every run is reproducible and comparable.

### 2.2 Out of Scope

Everything downstream of a trained model in the earlier product framing: the interactive chatbot, the what-if simulator, the visual explainability overlay (as a UI feature — GNNExplainer computation itself is in scope, its rendering is not), the alternative-supplier recommender, the risk trend timeline, proactive MCP-based alerts, the OR-Tools optimization engine, customer allocation, the transparent weighted business-risk formula, and the frontend dashboard. None of these are needed to test the four claims in Section 4, and none are being built. `architecture.md` remains their design reference if this work resumes as a product later — this document does not track that possibility.

### 2.3 Not Part of This Project At All

Authentication, authorization, a served API, RAG/LLM explanation generation, and security hardening — see `03_UI_UX_Documentation.md`, `07_Vector_Database_Design.md`, `08_Backend_Design.md`, `09_REST_API_Documentation.md`, and `12_Security_Documentation.md`, each of which is now a short out-of-scope notice rather than a specification.

## 3. Vision

Show, with evidence a reviewer can check rather than take on faith, that a heterogeneous, depth-adaptive, partially-global graph architecture is a materially better way to reason about supply-chain risk than the alternatives — and be equally willing to report where the evidence doesn't support that.

## 4. The Four Claims This Prototype Exists to Test

Each is owned by one HADES component; each has a defined validation route (`10_AI_ML_Documentation.md` §9.4) that is not "read the model's own internals and believe them":

| # | Claim | Owned by | Validated by |
|---|---|---|---|
| 1 | Type-aware local structure beats type-blind local structure, not just because it has more parameters | HGT encoder | Matched-parameter architecture ablation (GraphSAGE / GAT / HGT), with confidence intervals |
| 2 | Depth should be chosen per task, and a structural prior derived from graph geometry is a better starting point than a single global depth | Depth selection | Held-out layer-depth sweep (`L=1..4`) |
| 3 | Correlated risk can exist between suppliers with no recorded edge, and can be surfaced without one | Transformer 2 | Held-out edge recovery, future co-disruption rate, optional link-prediction cross-check |
| 4 | A supplier's global fragility and this firm's specific exposure to it are different numbers | Claim B (dyadic reweighting) | Double-counting ablation (with/without `priority_tier`), then reviewer plausibility check |

Claim 1 plus the leakage contract and graph correctness underneath it (reverse edges, temporal features) are the **core deliverable**. Claims 2–4 are validated as time and evidence allow, in that order — see `14_Model_Development_Roadmap.md`.

## 5. Success Criteria

- The leakage test passes (no single feature scores AUC > 0.9) before any other result is trusted.
- The architecture ablation runs to completion with a matched-parameter control arm, and every row carries a confidence interval.
- The result for each claim is reported honestly, including "not distinguishable at this label volume" if that is what the evidence shows — a null result is a valid outcome of this project, not a failure of it.
- Every training run is reproducible from `model_registry` alone (dataset, code commit, hyperparameters).

## 6. Functional Requirements

### 6.1 Data and Leakage Contract

| ID | Requirement |
|---|---|
| FR-DATA-01 | Every feature used in a training example is computed only from source rows with a timestamp ≤ that example's `t₀` |
| FR-DATA-02 | Delay/shortage/impact labels are derived only from events in `(t₀, t₀+14d]` |
| FR-DATA-03 | A single-feature leakage test (AUC > 0.9 on any solo feature fails it) is run and passes before any model result is trusted |
| FR-DATA-04 | Train/validation/test splits are by `t₀`, never random |

### 6.2 Graph Construction

| ID | Requirement |
|---|---|
| FR-GC-01 | The graph has 8 node types and 20 meta-relations (10 forward + 10 reverse via `ToUndirected()`) |
| FR-GC-02 | Structural edges (`USED_IN`, `MANUFACTURED_AT`) are filtered to those active as-of `t₀` using validity-window columns, not inferred from `shipments` |
| FR-GC-03 | One `HeteroData` snapshot is built per `t₀` in the snapshot schedule and registered in `graph_snapshots` with node/edge/label counts |

### 6.3 Model Architecture (`10_AI_ML_Documentation.md` §8)

| ID | Requirement |
|---|---|
| FR-MODEL-01 | HGT encoder, `L=4`, all four intermediate layer outputs retained |
| FR-MODEL-02 | Delay, shortage, and impact heads each read a task-specific structural-prior layer, not a single shared final layer |
| FR-MODEL-03 | GraphSAGE and GAT baselines are trained on the identical graph/split/loss/regularization as HGT, for the ablation |
| FR-MODEL-04 (stretch) | A learned depth gate, initialized to reproduce the structural prior exactly, anchored by a KL term |
| FR-MODEL-05 (stretch) | Transformer 2: same-type-only, no adjacency mask, bounded candidate pool by embedding cosine top-k, always including frontier nodes |
| FR-MODEL-06 (stretch) | Claim B: a deterministic (non-learned) dyadic reweighting, validated by the double-counting test before being reported |

### 6.4 Training and Evaluation

| ID | Requirement |
|---|---|
| FR-EVAL-01 | Every architecture/configuration is evaluated on the same held-out split with AUC-ROC, precision/recall, and calibration |
| FR-EVAL-02 | Every `model_evaluation_runs` row carries a confidence interval and the method used to compute it |
| FR-EVAL-03 | The architecture ablation includes a matched-parameter control arm |
| FR-EVAL-04 | GNNExplainer output is computed and logged for held-out high-risk predictions, as a diagnostic, never as the instrument that validates the depth claim |
| FR-EVAL-05 | Every training run writes one `model_registry` row, written by the pipeline, never entered by hand |

## 7. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-01 | **Reproducibility over performance.** A worse-but-reproducible result is preferred to a better-but-unreproducible one; every run is traceable to a dataset snapshot, git commit, and hyperparameter set |
| NFR-02 | **Statistical honesty.** No metric is reported without its confidence interval; a comparison inside the noise floor is reported as such, not as a win |
| NFR-03 | **Compute stays modest.** The full architecture at `d=64`, `L=4` is ~773K parameters / ~1.4 GFLOP full-graph forward pass (`project_HADES.md` Part 7) — trainable on a single consumer GPU or CPU; no distributed training infrastructure is required or planned. The parameter/FLOP count was derived against a ~5,000-node planning estimate; the measured v3 graph (`reports/step5_result_v3.md`, Step 2) is an order of magnitude larger (22,155–66,973 nodes), and one 100-epoch training run measured ~7 minutes on CPU (was ~80s on the original 50-supplier scale) — "reasonable time" still holds (still single-machine, still no distributed infrastructure), but the magnitude moved from single-digit minutes to a multi-hour full ablation matrix (55 runs, 4.61h measured) |
| NFR-04 | **No production concerns.** No uptime, latency (beyond training-time budgets), authentication, or multi-user requirement applies — this is a single-researcher offline pipeline |
| NFR-05 | Dataset remains fully synthetic; no real supplier/customer data enters this repository |

## 8. Assumptions

- The synthetic dataset (`db/`) is a sufficient stand-in for real historical data to test the four claims, including a genuine (constructed) hidden-dependency signal for Claim 3 (`db/README.md`).
- Label volumes will be modest (hundreds to low thousands of positives per task); Section 5 and `10_AI_ML_Documentation.md` §9.3 treat this as a known constraint to report honestly, not a problem to hide.
- A single researcher, working intermittently, is the intended user of this entire pipeline — there is no team-coordination or handoff requirement.

## 9. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-01 | Label volume is too low to distinguish any architecture/component from its alternative | Report confidence intervals on every comparison (NFR-02); a null result is treated as a legitimate finding, not a failed project |
| R-02 | Claim B's double-counting test shows the dyadic signal is redundant with what the encoder already learned | Reframe around `fulfilment_preference_weight` specifically, the one signal not already visible to the encoder (`10_AI_ML_Documentation.md` §8.5) |
| R-03 | Transformer 2 finds no correlated pairs above the noise floor | The architecture is designed to degrade gracefully to HGT + depth selection if this happens (`project_HADES.md` Part 0.3) — not a pipeline failure |
| R-04 | Scope creep back toward the dashboard/product framing mid-project | This document and `10_AI_ML_Documentation.md` §2 name the out-of-scope list explicitly; re-litigate scope only via a deliberate decision, not incrementally |

## 10. Future Extension

If the product-scoped build (dashboard, chatbot, MCP execution, optimization engine) is resumed later, `architecture.md` is its design reference. This document does not attempt to keep that possibility "warm" — it specifies only the research prototype actually being built.

---

## Document Control

- **Purpose:** Section 1. **Scope:** Section 2. **The four claims:** Section 4. **Functional Requirements:** Section 6. **Non-Functional Requirements:** Section 7. **Risks:** Section 9. **Future Extension:** Section 10.
- Baseline for `10_AI_ML_Documentation.md` and `14_Model_Development_Roadmap.md`.
