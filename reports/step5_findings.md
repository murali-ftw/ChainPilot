# HADES Model-Development Prototype — Combined Report (Desired Result → Findings → Conclusions → Diagnosis → Proposed Changes)

**Scope:** Steps 0–5 of `docs/14_Model_Development_Roadmap.md`, plus the Step A follow-up.
**Combines:** `reports/1. steps_0-5_findings.md` (findings and conclusions) and `reports/2. changes1.md` (desired result, diagnosis, and proposed changes) into a single reading order: what we set out to get → what we did and found → what it means → what's wrong → what we're changing and why.

---

## 1. Desired Result

What Steps 0–5 were supposed to deliver, per the roadmap's own success criteria (`01_Product_Requirement_Document.md` §5, `14_Model_Development_Roadmap.md` §8):

| Requirement | Source |
|---|---|
| Label volumes in the "hundreds to low thousands of positives per task" | `01` §8 |
| The architecture ablation resolves Claim 1 with a stated, evidenced answer — "HGT wins," "not distinguishable," or an in-between, on **each** task | `14` §8 |
| The layer-depth sweep resolves Claim 2's prior-half with evidence, not assertion | `14` §7 |
| Confidence intervals narrow enough that a real effect (if one exists) is detectable, not swamped by the CI table's own noise floor | `10_AI_ML_Documentation.md` §9.3 |
| Depth-prior assignment per head justified by the **measured** k-hop reach table, not the theoretical one | `06_Graph_Database_Design.md` §11, §6.1 |
| Train/validation/test snapshots drawn from comparable underlying conditions, so a held-out result measures generalization, not regime transfer | implicit in `10` §9.1's "split by t₀, never randomly" — the point of a time split is to test forward generalization, not transfer across disjoint regimes |
| Labels are unambiguous — one truth value per (entity, snapshot) | implicit correctness requirement of `05_Database_Design.md` §6.18 |
| A hidden-dependency scenario for Transformer 2 that is recoverable **only** through learned similarity, not through any single recorded categorical column | `project_HADES.md` §5.4's own stated reasoning for rejecting `component_type` bounding |

In short: a dataset and baseline strong enough that Claim 1 and Claim 2 each get a real verdict, and a foundation solid enough that starting Step 6 (the learned depth gate) is a justified next step rather than a guess.

---

## 2. Findings by Step

### 2.1 Step 0 — Environment and Data Audit

- **Environment:** System Python was 3.9.6 (below the 3.11+ floor); installed Python 3.11.15 via Homebrew and rebuilt `venv/` on it. PostgreSQL was not installed at all; installed PostgreSQL 15.18 via Homebrew, running as a background service. Wrote `requirements.txt` (`torch` 2.13, `torch-geometric` 2.8, `pandas` 3.0.5, `numpy` 2.4.6, `scikit-learn` 1.9, `scipy` 1.17, `psycopg2-binary` 2.9, `pytest` 9.1, `ruff` 0.16).
- **Database load:** `db/load_data.py --drop` loads all 19 tables and passes all 7 verification queries (FK integrity, chronology, non-negative stock history, both leakage checks, status-history coverage) cleanly, every time this was re-run across the whole session.
- **Realism audit re-confirmed:** label rates from the loaded DB — delay 10.97% (310 rows, 34 positive), shortage 7.22% (1,164 rows, 84 positive), impact 3.67% (300 rows, 11 positive) — match `db/README.md`'s stated ~11%/~7%/~4% exactly, and reproduce identically on every reload.
- **Open discrepancy (not resolved, flagged):** two independent attempts to reproduce `db/README.md`'s claimed single-feature AUC (~0.59–0.63 for `on_time_rate_90d` on delay) using different join methodologies both landed near chance (~0.47–0.48) instead. The original audit script isn't in the repo, so this isn't a like-for-like reproduction failure — noted as a soft finding to double-check before leaning on that number, not a blocker (Step 0's hard gate — clean load + 7/7 checks + real row counts in hand — was met).

### 2.2 Step 1 — The Leakage Contract

- Built `ml/data/features.py`, `ml/data/snapshots.py`, `ml/tests/test_leakage.py`.
- **Exclusion list enforced structurally**, not by convention: `shipments.status`, `shipments.delivered_at`, `inventory.stock_level`, `suppliers.reliability_history` never appear in a live SQL `SELECT` anywhere in `ml/data/features.py` (verified by grep — the only remaining matches are `inventory_history.stock_level`, the sanctioned as-of replacement).
- **A real bug caught and fixed:** the first pass filtered suppliers/products by `created_at <= t0`, which silently zeroed out both tables. `created_at` on reference tables turned out to be dataset-*generation* metadata (clustered at 2025-01-02, after every snapshot t0), not a business timestamp — unlike `orders.placed_at` / `shipments.created_at`, which are genuine 2024 business dates. Fixed by dropping the filter on reference tables only.
- **Single-feature leakage test result — no leak found** (gate: AUC > 0.90):

  | Task | n | positives | strongest solo feature | AUC |
  |---|---|---|---|---|
  | delay (Shipment) | 310 | 34 (10.97%) | `days_to_eta` | 0.7244 |
  | shortage (Product) | 1,164 | 84 (7.22%) | `min_stock_ratio` | 0.8808 |
  | impact (Supplier) | 300 | 11 (3.67%) | `country_India` | 0.7122 |

  `min_stock_ratio` at 0.88 is close to the gate but not a leak: it's the as-of "how close is this product's worst-stocked warehouse to its reorder threshold" — causally the most direct legitimate predictor of a future shortage, not future information.

### 2.3 Step 2 — Graph Correctness

- Built `ml/graph/builder.py` (HeteroData assembly, `ToUndirected()`) and `ml/graph/reach.py` (empirical k-hop reach).
- **Real feature dimensions, not the doc's placeholder estimates** (exactly as the roadmap instructed — "recompute against real one-hot cardinalities, don't assume the placeholder"):

  | Node type | Doc estimate | Actual | Why |
  |---|---|---|---|
  | Supplier | 21 | **14** | 6 countries, not more |
  | Component | 10 | **6** | 5 `component_type` values |
  | Product | 8 | **13** | added BOM/stock aggregates beyond plain category one-hot |
  | Factory | 10 | **6** | 5 factories → 5 location buckets |
  | Warehouse | 10 | **7** | 8 warehouses → 6 location buckets |
  | Shipment | 10 | **7** | route-level carrier data isn't populated in this dataset (checked directly — blank in `carrier_performance_snapshots`), so no fabricated `route_on_time_rate_90d` |
  | Order | 6 | **5** | 4 statuses + `days_to_due` |
  | Customer | 4 | **3** | only 3 tiers exist |

- **20 meta-relations confirmed** (10 forward + `ToUndirected()`'s 10 reverse), `graph_snapshots` refreshed with real node/edge/label counts and the actual git commit (the seed rows had a placeholder-looking hash and pre-`STOCKED_AT`-naming edge-count keys).
- **A documented-claim correction, not silently assumed true:** `docs/06_Graph_Database_Design.md` §6.1 claims a supplier's "co-parents" (other suppliers of the same component) are reachable in 2 hops via `SUPPLIES` → `rev_SUPPLIES`. Measured directly: **this does not hold**. `components.supplier_id` is a single not-null FK — each component belongs to exactly one supplier — so that 2-hop round trip can only return to the *same* supplier. Checked all 50 suppliers: 0 reached a different supplier in 2 hops. Flagged for Step 4 to justify the depth prior from the measured reach table, not this specific claim.
- **`USED_IN` edge count** grows 442→443→444→445→445→446 across the 6 persisted snapshots — matches `db/README.md`'s stated BOM evolution exactly.

### 2.4 Step 3 — HGT Baseline + Governance Tables

- Applied `db/governance_schema.sql` (`model_registry`, `model_evaluation_runs`, `explanation_subgraphs`) before any training ran, per `docs/05_Database_Design.md` §6.20–6.22.
- Built `ml/models/{depth,heads,encoder,model}.py`: HGT via `torch_geometric.nn.HGTConv` (L=4, d=64, heads=4), fixed structural depth-prior readout (`delay→h²`, `shortage/impact→h³`), 2-layer MLP heads, focal loss (γ=2, α from inverse train-split class frequency).
- **Shortage head design note:** `training_labels` for the shortage task carries **194 rows for 80 products** — one row per (product, warehouse) stocking instance, not one per product (confirmed by direct query: every product has 1–3 rows, matching the 194 `inventory` rows). There's no Product×Warehouse *node* in this graph (`STOCKED_AT` is an edge), so the shortage head reads the Product node's own embedding for every one of that product's labelled instances — a documented coarsening, not a bug.
- **Time split (strict, never random):** train Jul–Sep 2024 (3 snapshots), validation Oct 2024 (1), test Nov–Dec 2024 (2).
- Time budget: ~80s to fully train 100 epochs on CPU (dataset is small enough that this was never a constraint across 11 total model runs).

**Baseline (`hgt-baseline-v1`) test-split results:**

| Task | AUC | 95% CI | n | positives |
|---|---|---|---|---|
| delay | 0.6734 | [0.535, 0.811] | 142 | 17 |
| shortage | 0.7258 | [0.638, 0.801] | 388 | 37 |
| impact | 0.4010 | [0.031, 0.899] | 100 | 4 |

`impact`'s CI spans almost the entire [0,1] range — a legitimate "inconclusive at this label volume" result (only 4 test positives), not hidden.

### 2.5 Step 4 — Validate the Depth Prior (Claim 2, Part 1)

Layer-depth sweep (`L=1,2,3,4`, single shared readout — every task reads the same layer), each trained 100 epochs and compared against the Step 3 structural-prior baseline on the identical test split.

**Over-smoothing measurement** (mean pairwise cosine similarity per layer, L=4 model):

| Layer | Supplier | Component | Product | Factory | Warehouse | Shipment | Order | Customer |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.570 | 0.583 | 0.970 | 0.721 | 0.991 | 0.835 | 0.809 | 0.778 |
| 2 | 0.625 | 0.797 | 0.915 | 0.812 | 0.992 | 0.731 | 0.725 | 0.882 |
| 3 | 0.705 | 0.827 | 0.909 | 0.920 | 0.997 | 0.702 | 0.819 | 0.913 |
| 4 | 0.666 | 0.822 | 0.935 | 0.924 | 0.997 | 0.668 | 0.813 | 0.946 |

Most types rise toward collapse with depth (Factory 0.72→0.92, Customer 0.78→0.95). **Shipment does the opposite** (0.835→0.668) — an unpredicted, genuine finding, reported as measured rather than fitted to expectation.

**Structural prior vs. single shared depth** (paired bootstrap ΔAUC, 95% CI):

| Task | vs L=1 | vs L=2 | vs L=3 | vs L=4 |
|---|---|---|---|---|
| delay | ns | ns | ns | ns |
| shortage | prior worse (-0.049, sig.) | ns | ns | ns |
| impact | prior better (+0.115, sig.) | prior worse (-0.177, sig.) | ns | ns |

**Nested significance test, adjacent L values** (paired bootstrap ΔAUC, 95% CI):

| Task | L2 vs L1 | L3 vs L2 | L4 vs L3 |
|---|---|---|---|
| delay | +0.079 [-0.069, +0.228] ns | +0.019 [-0.051, +0.089] ns | -0.004 [-0.069, +0.063] ns |
| shortage | -0.017 [-0.066, +0.032] ns | -0.044 [-0.121, +0.033] ns | +0.039 [+0.002, +0.080] sig. |
| impact | +0.292 [+0.201, +0.384] sig. | -0.292 [-0.388, -0.196] sig. | +0.010 [-0.026, +0.057] ns |

**Honest conclusion:** the structural depth prior does **not** clearly beat a single shared depth. Delay shows no distinguishable difference anywhere. Shortage/impact show a few CIs excluding zero, but they flip sign between adjacent comparisons (impact: +0.29 then -0.29) — the signature of single-seed model-fit variance at very low positive counts (3–4 test positives for impact), not a real depth effect. **Claim 2 (prior half) is inconclusive at this label volume** — exactly what `10_AI_ML_Documentation.md` §9.3 predicts would happen, stated plainly rather than papered over.

### 2.6 Step 5 — Architecture Ablation (Claim 1)

Six runs: {GraphSAGE, GAT, HGT} × {fixed-d=64, matched-parameter}.

- **Engineering correction made before running:** GAT with multi-head attention (heads=4) had *more* parameters than GraphSAGE at equal `d`, contradicting the ablation's own premise (`project_HADES.md` §8.3: "GAT has fewer parameters than GraphSAGE at equal d"). Switched to single-head GAT so the documented capacity ordering (GAT < SAGE < HGT at fixed d) actually holds.
- **Matched-`d` values recomputed against real dimensions, not assumed:** the doc's `d=86/116` (GraphSAGE/GAT) were derived against placeholder feature dims. A parameter-count search against this dataset's real dims landed on **GraphSAGE d=66, GAT d=92** — all three architectures within ~1% of HGT's 701K-parameter encoder.

**Test-split ROC-AUC, all six arms:**

| Task | graphsage-fixed-d | gat-fixed-d | hgt-fixed-d | graphsage-matched-d | gat-matched-d | hgt-matched-d |
|---|---|---|---|---|---|---|
| delay | 0.640 [0.51,0.76] | 0.655 [0.53,0.78] | 0.673 [0.54,0.81] | 0.721 [0.57,0.86] | 0.642 [0.49,0.78] | 0.673 [0.54,0.81] |
| shortage | 0.677 [0.59,0.75] | 0.480 [0.40,0.56] | **0.726 [0.64,0.80]** | 0.527 [0.44,0.62] | 0.681 [0.60,0.75] | **0.726 [0.64,0.80]** |
| impact | 0.823 [0.73,0.91] | 0.776 [0.45,0.96] | 0.401 [0.03,0.90] | 0.826 [0.72,0.92] | 0.732 [0.52,0.90] | 0.401 [0.03,0.90] |

**Claim 1, stated plainly:**

- **delay:** not distinguishable — every pairwise CI overlaps zero, both axes.
- **shortage:** **HGT wins, robustly** — beats GraphSAGE and GAT significantly at fixed-d, and still beats GraphSAGE at matched-d (GAT closes the gap once matched, CI then includes zero).
- **impact:** HGT **loses** to both baselines at fixed-d (nominally significant vs GAT, -0.375) — but that gap **disappears** once GAT is parameter-matched (CI includes zero). This is the matched-parameter arm doing exactly its documented job: an apparent "GAT beats HGT" result is at least partly a capacity confound (GAT had fewer params, regularized better on 4 test positives), not a clean architecture effect.

### 2.7 Independent Reproducibility Audit

Two full audits were run (Steps 0–2, then Steps 0–5), each from a genuinely clean slate (`db/load_data.py --drop`, which also erases the governance tables — reapplied `db/governance_schema.sql` and retrained all 11 models on top of it both times).

**Result: every retrained model reproduced its original metrics bit-for-bit** — identical AUCs, identical confidence intervals, identical best-epoch selections, identical over-smoothing profiles, identical significance-test outcomes, across all 11 models.

**One durability gap found and fixed:** `model_evaluation_runs.model_version`'s foreign key was missing `ON UPDATE CASCADE` — it had only been patched live on the running database (to allow renaming `model_version` identifiers to match an exact naming spec), not in the source `db/governance_schema.sql`. A fresh `--drop` would have silently lost that fix. Added to the source file so it survives future reloads.

**Not touched, flagged instead of silently fixed:** `db/generate_dataset.py` / `db/load_data.py` carry pre-existing `ruff` style findings (unsorted imports, `dict()`-as-literal, etc.). Left alone both times: nothing there is functionally failing (all 7 load checks pass every run), and `generate_dataset.py` is a seeded deterministic simulation not written in this work — rewriting its control flow for lint compliance risks altering its stochastic behavior for zero functional benefit.

### 2.8 Current Governance State

11 `model_registry` rows, all `status='active'`, 195 `model_evaluation_runs` rows:

| model_version | architecture | parameters | purpose |
|---|---|---|---|
| `hgt-baseline-v1` | heterogeneous_graph_transformer | 713,763 | Step 3 baseline — L=4, fixed structural depth prior |
| `hgt-lsweep-L1` | heterogeneous_graph_transformer | 191,259 | Step 4 sweep, L=1 |
| `hgt-lsweep-L2` | heterogeneous_graph_transformer | 365,427 | Step 4 sweep, L=2 |
| `hgt-lsweep-L3` | heterogeneous_graph_transformer | 539,595 | Step 4 sweep, L=3 |
| `hgt-lsweep-L4` | heterogeneous_graph_transformer | 713,763 | Step 4 sweep, L=4 |
| `hgt-fixed-d` | heterogeneous_graph_transformer | 713,763 | Step 5 ablation, fixed arm |
| `graphsage-fixed-d` | graphsage | 677,571 | Step 5 ablation, fixed arm |
| `gat-fixed-d` | gat | 360,131 | Step 5 ablation, fixed arm |
| `hgt-matched-d` | heterogeneous_graph_transformer | 713,763 | Step 5 ablation, matched arm |
| `graphsage-matched-d` | graphsage | 720,261 | Step 5 ablation, matched arm |
| `gat-matched-d` | gat | 731,495 | Step 5 ablation, matched arm |

(Two more rows — `hgt-baseline-v1` retrained and `hgt-baseline-v2-impact-h2` — were added by the Step A follow-up; see §6.)

### 2.9 Code Inventory

```
db/
├── governance_schema.sql        model_registry, model_evaluation_runs, explanation_subgraphs
ml/
├── data/
│   ├── db.py                    connection helper
│   ├── features.py               as-of feature extraction, exclusion-list enforcement
│   └── snapshots.py              multi-snapshot feature/label assembly
├── graph/
│   ├── builder.py                HeteroData assembly, ToUndirected, graph_snapshots persistence
│   └── reach.py                  empirical k-hop reach measurement
├── models/
│   ├── depth.py                  structural depth prior + shared-depth override (L-sweep)
│   ├── heads.py                  prediction heads + focal loss
│   ├── encoder.py                 HGT / GraphSAGE / GAT encoders
│   └── model.py                  HADESModel (encoder + depth readout + heads composed)
├── train.py                      time split, training loop, model_registry logging
├── evaluate.py                   metrics, bootstrap CIs, over-smoothing, significance tests
├── run_step3.py / run_step4.py / run_step5.py   reproducible entrypoints
└── tests/
    ├── test_leakage.py           Step 1 gate (3 tests)
    ├── test_graph.py             Step 2 gate (21 tests)
    └── test_model.py             Step 3 unit tests (12 tests)
```

**Test suite: 36/36 passing. `ruff check ml/`: clean.**

### 2.10 Open Items Carried Forward (as of the Steps 0–5 report)

- **`db/README.md`'s solo-feature AUC claim (~0.59–0.63)** could not be reproduced against the loaded DB by two independent methodologies (both landed ~0.47–0.48) — worth a closer look, not yet resolved.
- **`docs/06_Graph_Database_Design.md` §6.1's co-parent claim** is measured false under the current schema — Step 4's depth-prior justification should cite the measured reach table, not this claim as written.
- **Claim 2 (depth-prior half) is inconclusive** at this dataset's label volume — not a negative result, but not a positive one either.
- **`impact` task (11 total positives, 3–4 per split)** makes every CI on that task very wide; several "significant" findings there should be read as fragile until more labelled snapshots exist.
- **CI method is row-level i.i.d. bootstrap**, not the schema's specified paired time-blocked bootstrap — documented compromise, since the test split is only 2 snapshots (block-resampling 2 blocks is degenerate). Recorded as `ci_method='row_bootstrap'`, not mislabelled.

---

## 3. Conclusions of Steps 0–5

*(This section is the original Part I conclusion from `1. steps_0-5_findings.md`. It is, as expected, materially the same as "Current Result" in `2. changes1.md` — the Step A finding, run after that conclusion was written, is appended at the end of this section so the two documents' pictures line up exactly.)*

### Bottom line

The pipeline is real, leakage-clean, and reproducible. Claim 1 has a genuine, task-dependent answer. Claim 2 does not. Claims 3 and 4 have not been reached.

### Claim 1 — Does type-aware attention (HGT) beat type-blind baselines?

**Yes, but only on one of the three tasks — and one apparent win for a baseline turned out to be a capacity confound, not a real effect.**

| Task | Verdict |
|---|---|
| delay | Not distinguishable. GraphSAGE, GAT, and HGT are statistically indistinguishable at both fixed and matched parameter counts. |
| shortage | **HGT wins, robustly.** Beats both baselines at fixed-d; still beats GraphSAGE once parameters are matched. The only task/comparison where the ablation gives a clean, confound-free win. |
| impact | **HGT loses** to GAT at fixed parameter count — but that loss **disappears** once GAT is parameter-matched to HGT. The fixed-d result was a capacity confound (GAT had fewer parameters and simply regularized better on 4 test-set positives), not evidence that attention hurts. |

**Conclusion:** Claim 1 is true for shortage, unresolved for delay, and the one result that looked like a loss for HGT does not survive the matched-parameter control. There is no task where a type-blind baseline beats HGT once parameters are controlled for.

### Claim 2 — Does the structural depth prior (h²/h³ split) beat a single shared depth?

**No clear answer — inconclusive at this dataset's label volume, not a negative result.**

- Delay: no significant difference between the prior and any single shared depth L=1–4.
- Shortage/impact: a handful of comparisons have CIs excluding zero, but the sign **flips between adjacent L values** (e.g. impact: prior beats L=2 by a wide margin, then loses to L=3 by a similar margin) — the signature of single-seed training variance at very low positive counts (3–4 test positives for impact), not a real depth effect.
- The over-smoothing measurement shows most node types drift toward embedding collapse with depth (Factory 0.72→0.92 cosine similarity, Customer 0.78→0.95) but **Shipment does the opposite** (0.84→0.67) — depth differentiates Shipment embeddings rather than smoothing them.

**Conclusion:** the data does not currently support adopting the structural prior over a shared depth, nor does it support rejecting it. This is a genuinely open question at this label volume, not a finding either way.

### Claims 3 and 4 — not reached

Transformer 2 (hidden-dependency discovery) and Claim B (dyadic reweighting) are Steps 7–8, not attempted in this scope. No conclusion exists for either yet.

### Pipeline integrity — what can be trusted about the numbers above

- **No label leakage.** The single-feature test (gate: AUC > 0.90) found nothing above 0.88 across all three tasks; the closest (`min_stock_ratio` on shortage, 0.88) is a legitimate direct predictor of shortage, not a leaked column.
- **Fully deterministic, independently verified twice.** A from-scratch database reload followed by retraining all 11 models reproduced every metric, every confidence interval, and every significance-test outcome bit-for-bit, on two separate occasions.
- **Feature dimensions are real, not assumed.** Every node type's dimensionality was recomputed against the actual data rather than the design docs' placeholder estimates, and differs from those estimates in every case (e.g. Supplier 14 not 21, Product 13 not 8).
- **One documented architectural claim was checked and found false.** The graph docs assert a supplier's "co-parents" are reachable in 2 hops; measured directly, this cannot happen under the current schema (each component has exactly one supplier). This does not by itself explain Claim 2's inconclusive result, but it means the depth prior's justification needs to rest on the measured reach table, not that specific claim.
- **The matched-parameter control mattered.** Without it, Step 5 would have reported "GAT beats HGT on impact" as if it were an architecture finding. It wasn't.

### Caveats that limit confidence in the above

- **`impact` has only 11 labelled positives total** (3–4 per split). Every conclusion touching this task should be read as provisional until more labelled snapshots exist — several of the "significant" findings above are fragile at this volume.
- **Confidence intervals use row-level bootstrap, not the schema's specified time-blocked bootstrap** — a deliberate, documented compromise, since the test split is only 2 snapshots (block resampling 2 blocks is statistically degenerate). Intervals are recorded as `ci_method = 'row_bootstrap'` and likely understate true uncertainty, per the same caveat the schema itself raises about i.i.d. methods.
- **`db/README.md`'s claimed single-feature AUC (~0.59–0.63) could not be reproduced** by two independent join methodologies against the loaded database (both landed ~0.47–0.48). Not resolved; doesn't change the leakage-test conclusion above (which used the assembled graph tensors directly, not this reproduction attempt), but is an open discrepancy worth resolving before citing that number elsewhere.

### Governance record (at the time of the Steps 0–5 report)

11 `model_registry` rows (`hgt-baseline-v1`, 4-way L-sweep, 6-way architecture ablation), all `status='active'`, 195 `model_evaluation_runs` rows with metric values and 95% CIs. This is the complete, reproducible evidentiary basis for the conclusions above.

### Addendum — Step A (run after the above was written)

To directly test the impact-head depth-prior question flagged in Claim 2 above, the prior was corrected (`impact: h³ → h²`, matching the L-sweep's observed peak) and retrained as a controlled A/B (`hgt-baseline-v2-impact-h2`) against a freshly retrained `hgt-baseline-v1`.

**Result: not significant** — ΔAUC = −0.044, CI [−0.106, +0.013]. The corrected prior does not beat the original at this label volume.

This confirms the "Current Result" picture in `2. changes1.md` exactly: correcting the known depth-assignment mismatch alone does not resolve the impact task. The binding constraint is the label volume underneath it (4 test positives), not just which layer the head reads.

---

## 4. Issues With Current Findings

| # | Issue | Evidence |
|---|---|---|
| 1 | **Label volume is far below the "hundreds to low thousands" target.** | Totals: delay 34, shortage 84, impact 11. Test split alone: 17 / 37 / **4**. |
| 2 | **Train and test snapshots sample different regimes.** | Train (Jul–Sep) has 9/23/4 positives; test (Nov–Dec) has 17/37/4 — more positives in test than train, because `db/generate_dataset.py`'s disruption events (port congestion, polymer shortage, trucking strike) are concentrated in H2 2024. A chronological split across a lopsided event calendar measures regime transfer, not generalization. |
| 3 | **Shortage labels lose the warehouse dimension.** | `entity_id = product_id` for a label that is actually per `(product, warehouse)`. 1,164 rows collapse to 480 distinct keys, of which **71 carry contradictory true/false labels** for the identical input. |
| 4 | **Supplier count (50) makes Transformer 2's future bounding a no-op.** | `project_HADES.md` §5.4 specifies `k=64` candidates as a *bound* on a larger pool; at `N_s=50`, `k ≥ N_s` always, so the "bounded" pool is dense attention regardless. |
| 5 | **The documented co-parent 2-hop path cannot exist under the current schema.** | `components.supplier_id` is a single NOT-NULL FK — `Supplier → SUPPLIES → Component → rev_SUPPLIES → Supplier` always returns to the same supplier. Measured directly: 0/50 suppliers reach a different supplier in 2 hops. This is part of the derivation `docs/06` §6.1 uses to justify `delay → h²`. |
| 6 | **The impact head's depth-prior mismatch is confirmed, but correcting it doesn't fix the result.** | L-sweep peaks at L2 for impact; Step A retrained with impact→h² and found the correction **not significant** (§3 addendum above) — meaning the binding constraint isn't just the wrong layer, it's the label volume underneath it (4 test positives). |
| 7 | **The planted Transformer-2 validation scenario is confounded with an observable column.** | The 4 hidden-dependency suppliers are the fleet's heaviest polymer-component suppliers — `component_type` alone would flag them, defeating the scenario's purpose per `project_HADES.md` §5.4's own argument against exactly this kind of bounding. |
| 8 | **Confidence intervals use row-level bootstrap, not the schema's specified time-blocked bootstrap.** | Documented compromise — the test split is only 2 snapshots, so block resampling is degenerate. Intervals likely understate true uncertainty. |
| 9 | **Multi-window temporal features are 52–90% NULL.** | `on_time_rate_90d` missing for 60–70% of suppliers at most snapshot dates; 50 suppliers and ~625 supplier-linked shipments/year is too thin to populate 30/90/180-day windows reliably. |
| 10 | **All Step 4/5 significance findings are from a single training seed.** | No seed-variance decomposition exists yet to separate "real effect" from "this particular initialization." |
| 11 | **`db/README.md`'s claimed solo-feature AUC (~0.59–0.63) could not be reproduced** (two independent attempts landed ~0.47–0.48) | Open, unresolved — plausibly explained by issue 9 (NULL handling differs between the two methodologies), but not confirmed. |

---

## 5. How These Issues Hinder Moving to Step 6

`14_Model_Development_Roadmap.md` §9 states the gate's adoption gate explicitly: *"Adopt the gate only if it beats the fixed-prior baseline by more than the confidence interval — not 'the gate ran and produced numbers.'"* Every issue above works against that condition being checkable at all right now:

- **No validated foundation to deviate from.** The gate (HADES §4.4) is architecturally a *small, KL-anchored correction* to the structural prior — by design a minor learned perturbation, not an independent model. Step 4 didn't show the prior beating a shared depth on any task, and Step A shows correcting the prior's known misassignment for impact still isn't detectable. There is no demonstrated-good prior to anchor a correction to; building the gate now means training a small deviation from a starting point of unknown quality.
- **The gate's expected effect size is smaller than what the pipeline can currently resolve.** HADES §4.7's own bias-variance table says a strong prior + small gate is the right choice specifically *because* label counts are low — but even that table's floor (200 positives) is 18× larger than impact's total count (11) and 5× larger than delay's (34). A mechanism explicitly designed to produce a *small* deviation cannot be distinguished from noise inside CIs this wide.
- **`node_depth_attention` (05 §6.24) is a diagnostic for "why did this node deviate from the prior," which presumes the prior is a meaningful reference point.** If the prior itself isn't validated, a reviewer inspecting the residual has nothing trustworthy to compare it against.
- **Regime mismatch would contaminate gate training specifically.** The gate learns per-node deviations from held-out performance; if train and test come from different disruption regimes, the gate risks learning to correct for the *train* regime's idiosyncrasies rather than a genuine structural pattern, and Step 6's own adoption test (compare against Step 3/4 baseline, same split) would inherit that same bias in both arms — masking the problem rather than exposing it.
- **Rework risk.** Any dataset fix made after Step 6 is built invalidates the gate's learned weights along with everything else, doubling engineering cost for no benefit over fixing the data first.

Conclusion: nothing about the current findings says the gate would fail if built — it says **the pipeline cannot currently tell you whether it succeeded or failed.** That is the specific condition the roadmap's own adoption criterion is designed to catch.

---

## 6. Updates Made — Status

| Step | Description | Status |
|---|---|---|
| **A** | Retrain impact head with prior corrected `h³ → h²`, controlled A/B against a freshly retrained `hgt-baseline-v1`, same paired bootstrap test as Step 4. | **Done.** `ml/models/depth.py` (`STRUCTURAL_DEPTH_PRIOR_V2`), `ml/run_step_a.py`, `ml/models/model.py`, `ml/train.py` updated; result logged to `model_registry`/`model_evaluation_runs`. Outcome: not significant (§3 addendum, §4 issue 6). |
| **B** | Multi-seed (5×) reruns of the shortage/impact L-sweep and baseline, to separate real effects from single-seed variance. | **Proposed, not yet run.** |
| **C** | Regenerate `db/generate_dataset.py`: scale suppliers/shipments up, spread disruption events across the full year, add the warehouse dimension to shortage labels, decouple the hidden-dependency members from `component_type`. | **Proposed, not yet run.** |
| **D** | Rebuild the database and rerun Steps 0–5 end to end against the regenerated dataset. | **Proposed, not yet run — depends on C.** |
| **E** | Write `steps_0-5_findings_v2.md` comparing v1 vs. v2 dataset results side by side. | **Proposed, not yet run — depends on D.** |

Only Step A has produced a result so far, and that result is itself informative: it rules out "the depth assignment was the whole problem" and points at label volume as the binding constraint, which is exactly what Steps C/D are aimed at.

---

## 7. How the Proposed Changes Would Affect the Model — Theoretically, Mathematically, and In Practice

### B — Multi-seed variance decomposition

- **Theoretical:** separates two distinct sources of an observed ΔAUC — genuine structural signal, and optimization/initialization noise. Right now these are conflated in every "significant" L-sweep finding.
- **Mathematical:** the bootstrap CI already reported captures *sampling* variance (uncertainty from which held-out examples happened to land in test); it does not capture *training* variance (a different random init converging to a different local optimum). Reporting `min/max/std` of ΔAUC across 5 seeds gives an empirical estimate of the second source, which the current single-seed numbers cannot distinguish from the first.
- **Real-world:** a result that only holds for one lucky initialization is not a result a stakeholder, reviewer, or production deployment can rely on. This is a cheap (~80s × 5 per cell) way to avoid reporting noise as a finding.

### C.1 — Scale up suppliers and shipments

- **Theoretical:** restores the conditions the architecture was designed for. HADES's central structural argument — co-parent reasoning, tiered `SUB_SUPPLIES` decay, T2's bounded candidate pool — assumes a supplier population large enough that these mechanisms do real work. At `N_s=50`, several of those mechanisms are vacuous by construction (see §4, issue 4).
- **Mathematical:** confidence-interval half-width for a binomial-style AUC estimate shrinks roughly with `1/√n_positive` (the exact relationship behind `project_HADES.md`'s own CI table: 200 positives → ±0.037, 1,000 → ±0.017). Moving impact's 11 total positives toward even the table's lowest rung (200) would shrink its CI by roughly `√(200/11) ≈ 4.3×` — enough to make the L-sweep's flip-signed findings interpretable instead of ambiguous.
- **Real-world:** a 50-supplier network is smaller than most real firms' actual supplier base, and undersized specifically in the direction that removes the hub structure and upstream redundancy the model needs to reason about — a company evaluating this architecture on their real supplier graph would not encounter this data regime.

### C.2 — Spread disruption events across the full year

- **Theoretical:** corrects a covariate-shift problem. A held-out generalization claim requires train and test to be drawn from comparable conditions; right now they aren't (§4, issue 2).
- **Mathematical:** with events concentrated in H2, the train and test splits are effectively different distributions over the label-generating process, not just different time windows of the same one. Any AUC difference measured across architectures or depths is partially explained by "which regime did this run's random variation land closer to," inflating apparent variance and biasing point estimates in an unpredictable direction depending on which task's signal happens to correlate with which event.
- **Real-world:** real supply chains do have uneven disruption calendars, but a benchmark where literally every major event lands in the test window (by construction, not by realistic seasonality) is an artifact of the generator's event schedule, not a realistic difficulty. Spreading events restores the ability to say "the model generalized" rather than "the model happened to see something similar to test during training."

### C.3 — Fix the shortage label's warehouse collapse

- **Theoretical:** removes an irreducible-error floor that was self-inflicted, not inherent to the task. Currently 71 of 480 distinct label instances have both a true and a false example sharing the identical input vector (the Product node's embedding, with no warehouse distinction) — a coin-flip target that no model, however good, can resolve.
- **Mathematical:** with ~6% of shortage instances structurally unresolvable, the achievable AUC ceiling is capped below 1.0 by an amount roughly proportional to that contradiction rate — HGT's measured 0.726 is being compared against an unknown, artificially lowered ceiling rather than the task's true difficulty.
- **Real-world:** shortage risk is inherently location-specific — a product can be critically low in one distribution center and fully stocked in another. A model that cannot represent this distinction (because the current schema doesn't attach a warehouse to the prediction target) would be operationally close to useless for an ops team trying to decide *which* warehouse needs an expedited shipment.

### C.4 — Decouple the hidden-dependency scenario from `component_type`

- **Theoretical:** restores the scenario to what it's supposed to test — whether learned embedding similarity can recover a coupling that has **no recorded edge and no shared categorical feature.** As built, the scenario is recoverable by a one-line `GROUP BY component_type`, which is precisely the failure mode `project_HADES.md` §5.4 explicitly warns against ("an earlier draft bounded by component_type... would have excluded exactly the cross-type correlation the scenario exists to find" — except here it's not excluded, it's *identical to* the scenario).
- **Mathematical:** any future validation of Transformer 2 against this scenario (`10_AI_ML_Documentation.md` §9.4, held-out edge recovery / co-disruption rate) would be measuring whether T2 rediscovers information already present in a one-hot input feature — a trivially achievable baseline, not evidence that global attention finds anything message-passing and simple feature lookups cannot.
- **Real-world:** the entire commercial argument for Transformer 2 in `project_HADES.md` §5.1 is "these two suppliers fail together and there's no recorded link between them, no shared component type, nothing an analyst's spreadsheet would surface." A confounded scenario cannot demonstrate that value proposition — a reviewer would rightly ask why a SQL query didn't already solve this.

---

## 8. How These Changes Improve the Model

Taken together, the changes in §7 do not alter the architecture at all — they alter the conditions under which the architecture is being judged. Their combined effect:

- Claims 1 and 2 get enough statistical power to resolve to an actual verdict on delay and impact, not just shortage — closing two of the three currently-open questions instead of leaving them permanently in the noise floor.
- The train/test comparison starts measuring **generalization**, which is what the roadmap's time-based-split requirement (`10` §9.1) is for — not regime transfer, which is what it currently, unintentionally measures.
- The shortage head's measured ceiling rises toward its true achievable value once contradictory labels are removed, making HGT's win in that task cleaner and more citable.
- Any future work on Transformer 2 inherits a validation scenario that can actually falsify the mechanism, rather than one that would "pass" regardless of whether T2 works.
- Multi-seed evidence turns "this L-sweep finding might be noise" into either a confirmed effect or an honestly discarded one — removing the single largest source of ambiguity in the current Step 4 results.

None of this requires new model code. It is entirely upstream of the architecture, which is the correct place to spend effort before adding the next architectural component (the gate).

---

## 9. How These Changes Get Us to the Desired Result

Mapping directly back to §1's checklist:

| Desired result | How the proposed changes get there |
|---|---|
| Label volumes in the hundreds | C.1 (scale suppliers/shipments) directly targets this |
| Claim 1 resolved per task | C.1 + C.2 (power + regime match) give delay and impact a real chance to resolve instead of sitting in the noise floor |
| Claim 2's prior-half resolved with evidence | C.1 + B (power + seed-variance decomposition) let the L-sweep's flip-signed findings be confirmed or discarded rather than left ambiguous; Step A already shows the fix isn't just about depth assignment — it's about volume, which C.1 addresses directly |
| CIs narrow enough to detect a real effect | C.1, mathematically, via the `1/√n` relationship in §7 |
| Depth-prior assignment justified by measured reach | Already partly done (Step 2's reach measurement, Step A's targeted test); C.1 gives it enough data to be conclusive rather than suggestive |
| Comparable train/test regimes | C.2, directly |
| Unambiguous labels | C.3, directly |
| A falsifiable hidden-dependency scenario for T2 | C.4, directly |
| A justified basis for starting Step 6 | The sum of the above — once Claim 2's prior-half has an actual answer and the CIs can resolve a small effect, the roadmap's own adoption criterion ("beats baseline by more than the CI") becomes checkable for the first time |

The path to Step 6 is not blocked by anything about the architecture or the gate design — it's blocked by the evidentiary base underneath it not yet being strong enough to apply the roadmap's own stated adoption test. Steps B–E close that gap without touching Step 6 itself; Step D's rerun is the checkpoint at which it becomes possible to say, with evidence, whether Step 6 is worth building.
