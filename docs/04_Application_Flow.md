# Document 4 — Application Flow

## HADES Model-Development Prototype

Version: 2.0 — rescoped to the ML pipeline only
Status: Baseline

---

## 1. Purpose

This document describes the three flows that make up this project's entire runtime behavior: building a graph snapshot, training a model, and evaluating it. There is no user-facing application, so there are no auth, dashboard, chat, approval, alert, or execution flows — those existed only in the earlier product-scoped version of this document.

## 2. Scope

- **In scope:** Graph Snapshot Construction Flow (Section 5), Training Flow (Section 6), Evaluation Flow (Section 7), the Leakage Test (Section 8).
- **Out of scope:** every flow that only a served application would need — see `01_Product_Requirement_Document.md` §2.2/§2.3.

## 3. Assumptions

Same as `01_Product_Requirement_Document.md` §8: a single researcher runs each flow as an offline script against the database in `05_Database_Design.md`; nothing here is triggered by a user action or a schedule.

## 4. Dependencies

`05_Database_Design.md` (source tables), `06_Graph_Database_Design.md` (graph assembly), `10_AI_ML_Documentation.md` (feature engineering, model, training/evaluation protocol).

## 5. Graph Snapshot Construction Flow

**Trigger:** A researcher runs the snapshot-building script for a given `t₀` (or the full snapshot schedule).

```mermaid
flowchart TD
    A["Read master tables as-of t0\n(suppliers, components, products, ...)"] --> B["Read history tables as-of t0\n(inventory_history, shipment_status_history,\nsupplier_temporal_features, carrier_performance_snapshots)"]
    B --> C["Filter structural edges to those\nactive at t0 (product_components,\nproduct_factories validity windows)"]
    C --> D["Encode node/edge features\n(06_Graph_Database_Design.md §7)"]
    D --> E["Assemble HeteroData\n(8 node types, 10 forward relations)"]
    E --> F["ToUndirected()\nadd 10 reverse relations -> 20 total"]
    F --> G["Write graph_snapshots row\n(node/edge/label counts, git_commit)"]
    G --> H["Attach training_labels\nfrom (t0, t0+14d]"]
    H --> I["Snapshot ready for training/evaluation"]
```

- **Steps:** read master + history tables as-of `t₀` → filter structural edges by validity window → encode features → assemble `HeteroData` → add reverse relations → register the snapshot → attach labels.
- **Failure handling:** a snapshot whose `t₀` predates the trustworthy-from date of any history table (`05_Database_Design.md` §8, DB-05) is refused rather than built from approximated history.
- **Repeats once per `t₀`** in the snapshot schedule — this is a batch process over the dataset's history, not a single live graph kept up to date.

## 6. Training Flow

**Trigger:** A researcher runs the training script for a given architecture/configuration.

```mermaid
sequenceDiagram
    participant SNAP as Graph Snapshots (Section 5)
    participant TRAIN as Training Loop
    participant MODEL as HADES Model (10_AI_ML_Documentation.md §8)
    participant PG as model_registry

    TRAIN->>SNAP: Load snapshots for the time-based train/val split
    TRAIN->>TRAIN: Assemble batches (full-graph per snapshot)
    loop Each epoch
        TRAIN->>MODEL: Forward pass (delay/shortage/impact heads)
        MODEL-->>TRAIN: Predictions
        TRAIN->>TRAIN: Focal loss + KL anchor (if depth gate active)
        TRAIN->>TRAIN: Backward pass, optimizer step
        TRAIN->>TRAIN: Evaluate on validation split, check early-stopping
    end
    TRAIN->>PG: Write model_registry row\n(model_version, architecture, dataset, git_commit,\nhyperparameters, parameter_count, status)
```

- **Steps:** load the time-based split → train with early stopping on validation AUC → write governance metadata directly from the pipeline, never by hand (`01_Product_Requirement_Document.md` FR-EVAL-05).
- **Split discipline:** train/validation/test are split by `t₀`, never randomly — the same entity appears across many snapshots, and a random split would leak across them.
- **Applies identically** to every architecture in the ablation (GraphSAGE, GAT, HGT, matched-parameter arm) and to every `L` in the layer-depth sweep — only the model definition and `L` change; the flow itself does not.

## 7. Evaluation Flow

**Trigger:** A training run completes.

```mermaid
sequenceDiagram
    participant TRAIN as Training Loop (Section 6)
    participant EVAL as Evaluation Step
    participant PG as model_evaluation_runs

    TRAIN->>EVAL: Trained model + held-out test snapshots
    EVAL->>EVAL: Compute AUC-ROC, precision/recall, calibration per task
    EVAL->>EVAL: Compute confidence interval (paired, time-blocked bootstrap or DeLong)
    EVAL->>PG: Insert model_evaluation_runs row(s)\n(model_version, architecture, task, metric, value, ci_lower, ci_upper, ci_method)
    EVAL->>EVAL: Compare against other rows for the same claim\n(10_AI_ML_Documentation.md §9.4)
    EVAL->>EVAL: Report result, including "not distinguishable\nat this label volume" if that is what the CIs show
```

- **Steps:** compute metrics on the held-out split → compute and persist confidence intervals → compare against the relevant claim's other runs → report honestly.
- **Never skips the interval.** A `model_evaluation_runs` row with no `ci_lower`/`ci_upper` is treated as incomplete, not as a valid point estimate to compare on its own — this is the schema decision from `05_Database_Design.md` §6.21 enforced at the process level.
- **Repeats** for every run in `10_AI_ML_Documentation.md` §9.2's run inventory (layer-depth sweep, ablation, matched-parameter arm, and — as built — depth-gate on/off, Transformer 2 on/off, Claim B double-counting test).

## 8. The Leakage Test

**Trigger:** Run once before trusting any other result (`10_AI_ML_Documentation.md` §9.5), and again after any change to feature engineering.

```mermaid
flowchart TD
    A["Take the assembled HeteroData\nfeature set (not raw CSVs)"] --> B["Train a minimal model\non one feature at a time"]
    B --> C{"Any solo feature\nAUC > 0.9?"}
    C -->|Yes| D["Leak found -- trace to source column,\nfix feature derivation, repeat"]
    C -->|No| E["Proceed to Section 6 (Training Flow)"]
```

- **Why this flow exists separately:** `db/README.md`'s realism audit already ran a version of this over the raw synthetic CSVs (max solo-feature AUC ≈ 0.63). This flow re-runs it against the actual *assembled* graph features, since the failure mode that matters is in feature assembly (e.g., accidentally joining a post-`t₀` row), not only in the source columns.

## 9. Risks

| ID | Risk | Mitigation |
|---|---|---|
| AF-01 | A snapshot is built with a `t₀` earlier than a history table's trustworthy-from date | Refused at construction time (Section 5), not caught later at training time |
| AF-02 | An evaluation run is compared against another without checking both have confidence intervals attached | `model_evaluation_runs` rows without CI columns populated are treated as incomplete (Section 7) |
| AF-03 | The leakage test (Section 8) is skipped after a feature-engineering change because "it already passed once" | Re-run required by process after any feature change, not only once at project start |

---

## Document Control

- **Purpose:** Section 1. **Scope:** Section 2. **Assumptions:** Section 3. **Dependencies:** Section 4. **Risks:** Section 9.
