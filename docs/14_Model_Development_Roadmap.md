# Document 14 — Model Development Roadmap

## HADES Model-Development Prototype: How to Build and Train It

Version: 1.0
Status: Baseline
Restates and sequences `project_HADES.md` Parts 8 and 11 as an executable roadmap, cross-referenced to the rescoped Documents 1–13.

---

## 1. How to Read This Document

This is the order to actually do the work in — not a restatement of what each component is (`10_AI_ML_Documentation.md` already does that) or why it's designed the way it is (`project_HADES.md` already does that). Each step below names: what you're doing, what it depends on, how you know it's done, and what claim (if any) it moves forward. Steps are sequential — later steps assume earlier ones are complete and correct, especially the leakage contract, which nothing downstream is trustworthy without.

**The prototype has a real stopping point.** Steps 0–5 are the core deliverable: a leakage-safe graph, a trained and validated HGT baseline, and evidence for or against Claim 1. Steps 6–8 (depth gate, Transformer 2, Claim B) are pursued in that order, as far as time allows, and each is independently useful even if the ones after it don't happen — this is by design (`project_HADES.md` Part 0.3, "graceful degradation").

## 2. Current State

As of this document, the following are already built and should not be redone:

| Done | Where |
|---|---|
| Schema for the 19 model-relevant tables (Group A/B + `risk_scores`) | `db/schema.sql` |
| A deterministic, leakage-audited synthetic dataset, including a constructed hidden-dependency scenario for Claim 3 | `db/generate_dataset.py`, `db/README.md` |
| A loader with FK/chronology/label-window verification | `db/load_data.py` |
| The full architecture specification | `project_HADES.md` |
| Rescoped documentation set (this one included) | `01`–`13`, this document |

**Not yet built:** any actual model code, the graph-assembly pipeline, `model_registry`/`model_evaluation_runs`/`explanation_subgraphs` tables, and anything from Step 3 onward below.

## 3. Step 0 — Environment and Data Audit

**Do:**
- Set up the Python environment (`11_Implementation_Guide.md` §5–6).
- Load the dataset (`11_Implementation_Guide.md` §7): `createdb`, `python db/load_data.py`.
- Re-read `db/README.md`'s realism audit findings and confirm they still hold (max solo-feature AUC ≈ 0.59–0.63, label rates ~11%/~7%/~4% for delay/shortage/impact).

**Done when:** the database loads cleanly, all of `load_data.py`'s verification queries pass, and you have the actual current row counts and label rates in front of you — not the numbers in this document, which will drift as the generator is tuned.

**Blocks:** everything.

## 4. Step 1 — The Leakage Contract

**Do:**
- Implement as-of feature computation with a hard `t₀` cutoff (`10_AI_ML_Documentation.md` §6.1–6.3).
- Apply the exclusion list (`10_AI_ML_Documentation.md` §6.2) — never read `shipments.status`, `shipments.delivered_at`, or `inventory.stock_level` as features.
- Build the multi-snapshot structure: one feature/label set per `t₀` in the snapshot schedule.
- **Run the single-feature leakage test** (`10_AI_ML_Documentation.md` §9.5, `13_Testing_Documentation.md` §6): train on one feature at a time; any solo feature above AUC 0.9 is a leak.

**Done when:** the leakage test passes against the *assembled* feature set (not just the raw CSVs, which already passed this in `db/README.md`'s audit — re-running it here catches leaks introduced during assembly, e.g. an accidental join to a post-`t₀` row).

**Blocks:** every downstream training result. Do not proceed past this step on a failure.

## 5. Step 2 — Graph Correctness

**Do:**
- Assemble `HeteroData` per snapshot (`06_Graph_Database_Design.md` §11): 8 node types, 10 forward relations.
- Apply `ToUndirected()` to add the 10 reverse relations (`06_Graph_Database_Design.md` §6) — 20 meta-relations total.
- **Measure, don't assume, the actual k-hop reach per target node type** (`06_Graph_Database_Design.md` §11) — this is what the depth prior in Step 4 will actually be justified by, not the planning table in `project_HADES.md` §3.7.
- Verify feature dimensionality matches Section 7 of `06_Graph_Database_Design.md` — recompute against the real one-hot cardinalities, don't assume the placeholder numbers. Done: measured values are Supplier 14, Component 6, Product 13, Factory 6, Warehouse 7, Shipment 7, Order 5, Customer 3 (not the original 21/10/8/10/10/10/6/4 planning estimate — `06_Graph_Database_Design.md` §7 now carries the measured table, confirmed unchanged across the v2 and v3 dataset scale-ups since only population sizes grew, not category cardinalities).
- Write one `graph_snapshots` row per `t₀` with real node/edge/label counts.

**Done when:** a `HeteroData` object exists for every `t₀` in the schedule, reverse relations are confirmed present (spot-check: a Supplier node has nonzero in-degree from `rev_SUPPLIES`), and the measured k-hop reach table is in hand.

## 6. Step 3 — HGT Baseline

**Do:**
- Implement the HGT encoder, `L=4`, retaining all four intermediate layer outputs (`10_AI_ML_Documentation.md` §8.1).
- Implement the **fixed structural depth-prior readout** (blend level L0 — no learned gate yet): each head reads its designated layer (`h²` delay, `h³` shortage/impact), chosen from the measured reach table in Step 2, not learned.
- Implement the three prediction heads with focal loss (`10_AI_ML_Documentation.md` §8.4).
- Implement the time-based train/val/test split (`10_AI_ML_Documentation.md` §9.1).
- **Before training runs on real results, build `model_registry`, `model_evaluation_runs`, and `explanation_subgraphs`** (`05_Database_Design.md` §6.20–6.22) — a schema with nowhere to store a confidence interval guarantees a point estimate gets reported instead (`05_Database_Design.md` §6.21). Do this now, not after the first run.
- Train the baseline; compute AUC-ROC, precision/recall, calibration, all with confidence intervals.

**Done when:** one `model_registry` row and a full set of `model_evaluation_runs` rows (with CIs) exist for the HGT baseline at `L=4`, fixed readout.

## 7. Step 4 — Validate the Depth Prior (Claim 2, Part 1)

**Do:**
- Run the layer-depth sweep: train and evaluate at `L=1,2,3,4` (`10_AI_ML_Documentation.md` §9.2, §9.4).
- Compute the over-smoothing measurement (mean pairwise embedding cosine similarity per layer) — this replaces the theoretical reach table with a real empirical depth ceiling.
- Run a nested significance test on each `ΔAUC` between adjacent `L` values.
- Report the result honestly: at the label volumes typical of this dataset, some `L` values will likely be statistically indistinguishable (`10_AI_ML_Documentation.md` §9.3) — say so.

**Done when:** four `model_evaluation_runs` rows exist for the sweep, each with a CI, and you can state — with evidence, not assertion — whether the structural depth prior (`h²`/`h³` split by task) actually improves on a single shared depth.

**Moves forward:** Claim 2, the "structural prior beats a single shared depth" half of it. (The "learned gate improves further" half is Step 6.)

## 8. Step 5 — Architecture Ablation (Claim 1)

**Do:**
- Train GraphSAGE and GAT baselines on the identical graph, split, loss, and regularization as HGT (`10_AI_ML_Documentation.md` §8.1).
- **Train the matched-parameter control arm** — GraphSAGE and GAT with `d` raised so their parameter counts are comparable to HGT's. This is not optional: without it, a Stage 3 win is confounded between "attention helped" and "more capacity helped" (`10_AI_ML_Documentation.md` §8.1, `project_HADES.md` §8.3).
- Evaluate all six runs (3 architectures × fixed-`d`/matched-`d`) on the identical held-out split, all with confidence intervals.
- Report the comparison on both axes (fixed-`d` and matched-`d`), honestly, including a null result if the CIs overlap.

**Done when:** all six runs are in `model_evaluation_runs`, and Claim 1 has a stated, evidenced answer — "HGT wins," "not distinguishable at this label volume," or something in between.

**— The core prototype deliverable ends here. Steps 0–5 are what "a trained, evaluated HADES baseline with evidence for Claim 1" means. Everything below is pursued as time and evidence allow, each with its own honest stopping point. —**

## 9. Step 6 — The Learned Depth Gate (Claim 2, Part 2)

**Do:**
- Implement the per-head depth gate (`10_AI_ML_Documentation.md` §8.2): `d → 32 → 4`, initialized to reproduce the Step 4 structural prior exactly, anchored by a KL term (`λ=0.01`).
- Build `node_depth_attention` (`05_Database_Design.md` §6.24) to persist per-node weights.
- Train with the gate active; compare against the Step 3/4 fixed-readout baseline.
- **Adopt the gate only if it beats the fixed-prior baseline by more than the confidence interval** — not "the gate ran and produced numbers."
- If adopted, report the residual: which suppliers/nodes deviated from the prior, and in which direction (`10_AI_ML_Documentation.md` §8.2's worked example pattern).

**Done when:** you can state, with a CI-backed comparison, whether the learned gate earns its (small) added cost over the free structural prior.

## 10. Step 7 — Transformer 2 (Claim 3)

**Do:**
- Implement same-type-only global attention over Supplier embeddings, no adjacency mask (`10_AI_ML_Documentation.md` §8.3).
- Implement the cosine top-k candidate pool (`k=64`), always including `is_frontier` nodes.
- Wire the output to the delay head only, per the current interface scope (`10_AI_ML_Documentation.md` §8.3, Option 1).
- Build `hidden_dependency_links` (`05_Database_Design.md` §6.25); persist discovered pairs with attention weight, `model_version`, `snapshot_t0`.
- Validate: check discovered pairs against the dataset's constructed hidden-dependency scenario (`db/README.md` — the four cross-country, cross-component-type suppliers sharing an unmodeled polymer plant). This dataset was built specifically to make this validation possible — use it.
- **Adopt only if validation shows better-than-chance discovery**, not merely "attention scores were computed."

**Done when:** you can state whether Transformer 2 actually recovered the constructed hidden-dependency signal in the synthetic data — the single most concrete pass/fail check available anywhere in this roadmap.

## 11. Step 8 — Claim B (Dyadic Reweighting)

**Do:**
- Implement the deterministic reweighting formula (`10_AI_ML_Documentation.md` §8.5) — not a learned component.
- **Run the double-counting test first:** train the encoder with and without `Customer.priority_tier`; compare Claim B's reordering rate between the two. If the rate collapses without the feature, the encoder already knew it.
- Lead the reported formula on `fulfilment_preference_weight` specifically — the one signal the encoder cannot already see (`10_AI_ML_Documentation.md` §8.5).
- Build `supplier_dyadic_risk` (`05_Database_Design.md` §6.26); persist reweighted scores separately from `risk_scores`, never overwriting the global score.

**Done when:** the double-counting test has actually run and its result is reported alongside the reweighting, not omitted.

## 12. Step 9 — Write-Up

**Do:**
- Assemble the final report from what actually happened, not from what `project_HADES.md` predicted would happen — the whole point of Steps 4/5/7/8's validation routes is that the answer isn't known in advance.
- For each of the four claims (Section 1, `01_Product_Requirement_Document.md` §4): state the claim, the validation route used, the result, and the confidence interval or equivalent evidence.
- State plainly which claims were tested to completion, which were tested but inconclusive at this label volume, and which were not reached given the time available — all three are legitimate outcomes of a research prototype.

## 13. Summary Table

| Step | What | Claim | Status if skipped |
|---|---|---|---|
| 0 | Environment + data audit | — | Nothing else is possible |
| 1 | Leakage contract | — | Every downstream number is meaningless |
| 2 | Graph correctness (reverse edges, measured reach) | — | Depth prior in Step 4 is unfounded |
| 3 | HGT baseline + governance tables | — | No reproducible result exists at all |
| 4 | Layer-depth sweep | 2 (prior half) | Depth-prior claim is asserted, not evidenced |
| 5 | Architecture ablation + matched-param arm | 1 | No answer to "does type-aware structure help" |
| **— core deliverable boundary —** | | | |
| 6 | Learned depth gate | 2 (gate half) | Fine — Step 4's fixed prior still stands on its own |
| 7 | Transformer 2 | 3 | Fine — HGT + depth selection is a complete, defensible system without it |
| 8 | Claim B | 4 | Fine — a documented future extension, not a gap in the core result |
| 9 | Write-up | all | Report only what was actually reached |

## 14. Risks to the Roadmap Itself

| Risk | Mitigation |
|---|---|
| Label volume turns out too low even for Step 5's core ablation to be decisive | Report the null result (Step 5) — it is still evidence, and `10_AI_ML_Documentation.md` §9.3 predicts this is likely at this dataset's scale |
| Time runs out before Step 7/8 | The stopping points in Section 13 are real — Steps 0–5 alone are a complete, honest deliverable |
| Temptation to skip Step 1's leakage test because "the dataset was already audited" | The audit in `db/README.md` covers the raw CSVs; Step 1 covers the assembled tensor features, a different failure surface — both are required |
| Temptation to skip the matched-parameter arm in Step 5 because it's "extra work for a control" | It's the difference between a real Claim 1 result and a confounded one — not optional (`10_AI_ML_Documentation.md` §8.1) |

---

## Document Control

- Sequences `project_HADES.md` Parts 8 (training protocol) and 11 (implementation checklist) against the rescoped `01`–`13` documents.
- Supersedes any earlier product-scoped project roadmap/sprint plan.
