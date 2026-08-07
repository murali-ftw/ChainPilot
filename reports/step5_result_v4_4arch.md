# HADES Model-Development Prototype — RGCN Fourth-Architecture Ablation (v4)

**Scope:** add RGCN (Schlichtkrull et al. 2018 basis decomposition) as a fourth encoder
architecture alongside HGT/GraphSAGE/GAT, and re-run Steps 3-5's architecture ablation as a
4-way matrix — 5 seeds × 4 architectures × 2 param-arms (fixed-d=64, matched-d) = 40 training
runs, multi-seed from the start. Run against the same dataset as `reports/step5_result_v3.md`
and its follow-up (v3: 800 suppliers, 15 monthly snapshots Jul 2024 - Sep 2025) — re-verified
live against the database before any code was written, not assumed. This is an **addendum**:
v3's own conclusions and numbers are unchanged and are not restated here except where a new
measurement bears directly on them. Same structure as the prior two reports: **Part I —
Conclusions**, **Part II — Findings by Phase**.

---

# Part I — Conclusions

## Bottom line

**No architecture dominates across all three tasks — the four-way matrix sharpens Claim 1
rather than resolving it.** HGT still wins impact outright, replicating every prior round's
finding. RGCN is a genuine new result: it beats both HGT and GraphSAGE on shortage,
**consistently across all 5 seeds, on both the fixed-d and matched-d arms** — the cleanest,
most reproducible finding in this entire round, and independent support (via a different,
more principled sharing mechanism) for the thin-relation-overfitting hypothesis Task 2's
sparse-merged-HGT experiment first raised in the v3 follow-up. Delay remains statistically
undecided among HGT/GraphSAGE/RGCN on both arms — the sign of the delta flips across seeds
every time those three are compared. GAT is the clear architecture-level loser on delay and
shortage in both arms, and only beats another architecture (GraphSAGE) on impact — never HGT.

## Dataset version actually used (Phase 0 pre-flight)

Confirmed live against the database, not assumed, before any code was written: **v3** (800
suppliers, 2,400 components, 1,280 products, 15 `graph_snapshots` rows, Jul 2024 - Sep 2025).
The `component_suppliers` junction table added by Task 3's v3-followup experiment **is
present** in the live schema (420 rows) but every component still has exactly one supplier
(`max(count) = 1` across `component_suppliers`) — i.e. this is v3 plus Task 3's schema
addition with dual-sourcing dormant, **not the separate v4/co-parent dataset generation Task 3
used**. `ml/graph/builder.py`'s `SUPPLIES` union-all against `component_suppliers` is a no-op
on this data, same as it has been since Task 3 shipped. Real per-node-type feature dims
(re-derived live against this database, not reused from any prior report's numbers):
Supplier=14, Component=6, Product=13, Factory=6, Warehouse=7, Shipment=7, Order=5, Customer=3
— unchanged from v3.

`model_registry`/`model_evaluation_runs` were backed up via `pg_dump` to
`reports/backups/model_registry_and_evals_backup_20260806_230544.sql` (81 + 1,083 rows) before
any write this round, regardless of the fact that no `--drop` reload was needed this time —
the v3 `--drop` incident (`reports/step5_result_v3_followup.md`'s governance record) is why
this happens unconditionally now. Post-run: 121 + 1,683 rows — exactly +40 registry rows and
+600 evaluation rows, no prior row touched or modified.

## RGCN — design and the bet it tests

`ml/models/rgcn_encoder.py` implements basis-decomposition R-GCN, hand-written over
`edge_index_dict` rather than using `torch_geometric.nn.RGCNConv` directly — the same reason
`ml/models/sparse_hgt_encoder.py` (Task 2) is also a bespoke module: `RGCNConv`'s public API
expects a homogeneous graph (one feature matrix + integer edge-type tensor), not this
codebase's per-relation `HeteroData` convention. Every relation gets its own transform, like
HGT, but drawn from a small shared pool of `num_bases` basis matrices (`W_r = Σ_b a_r[b] · V_b`)
rather than a fully dedicated `hidden × hidden` matrix — a single basis pool and coefficient
set for the whole encoder, reused identically at every layer, so the relational parameter cost
does not scale with `num_layers`. A separate, NOT basis-shared, per-node-type self-loop matrix
per layer carries identity across layers, playing the role HGT's residual connection and
GraphSAGE's self+neighbor combine play. If HGT's fully-dedicated per-relation parameterization
is overfitting on the thinnest relations (the hypothesis Task 2 tested by literally sharing one
`SAGEConv` across the 4 thinnest relations), RGCN's structured, cheaper form of sharing —
every relation still gets its own coefficients, just drawn from a shared basis — is a second,
more principled way to probe the same question at a matched parameter budget.

## Parameter counts

Matched-parameter search target: HGT's real encoder-only anchor, **701,088 params**
(re-derived live against this run's real feature dims — unchanged from v3, confirming dims
haven't shifted). Grid-searching `num_bases ∈ {4, 8, 12, 16}` at `hidden=64` (the first step
specified) landed nowhere close (max 203,392 params at `num_bases=16`) — self-loop and
input-projection cost dominates RGCN's parameter count at that width, unlike HGT's per-relation
attention, so the search widened `hidden` alongside `num_bases`, as instructed for exactly this
case. Best fit found: **`hidden=138, num_bases=4` → 699,602 encoder params, 0.21% off HGT's
anchor** — the closest matched-arm fit of any of the four architectures (GraphSAGE: 0.81% off;
GAT: 0.64% off).

| Arm | Encoder-only params | Full model params (incl. 3 task heads) |
|---|---:|---:|
| HGT fixed-d=64 | 701,088 | 713,763 |
| GraphSAGE fixed-d=64 | 664,896 | 677,571 |
| GAT fixed-d=64 | 347,456 | 360,131 |
| **RGCN fixed-d=64** (`num_bases=8`, default) | 170,464 | 183,139 |
| HGT matched-d=64 (anchor, identical to fixed-d by construction) | 701,088 | 713,763 |
| GraphSAGE matched-d=66 | 706,794 | 720,261 |
| GAT matched-d=92 | 705,548 | 731,495 |
| **RGCN matched-d=138** (`num_bases=4`) | 699,602 | 757,565 |

Full-model counts (what's actually logged to `model_registry.parameter_count`) exceed the
encoder-only search target by a constant +12,675 in the fixed-d arm (3 task heads at d=64,
identical across all four architectures there) and by an architecture-specific amount in the
matched-d arm, since `hidden` — and therefore each `PredictionHead`'s own d²+d+d+1 cost — differs
per architecture in that arm. This mirrors the existing GraphSAGE=66/GAT=92 precedent exactly.

## AUC — mean/std/min/max across 5 seeds, per (task, arm)

| Arm | delay | shortage | impact |
|---|---|---|---|
| hgt-fixed-d | 0.8015 ± 0.0106 [0.7834, 0.8132] | 0.7824 ± 0.0057 [0.7745, 0.7920] | **0.9335 ± 0.0022** [0.9307, 0.9372] |
| graphsage-fixed-d | **0.8053** ± 0.0040 [0.7996, 0.8116] | 0.7857 ± 0.0047 [0.7782, 0.7916] | 0.8912 ± 0.0046 [0.8872, 0.8999] |
| gat-fixed-d | 0.7467 ± 0.0080 [0.7325, 0.7549] | 0.6728 ± 0.0248 [0.6422, 0.6975] | 0.9138 ± 0.0050 [0.9093, 0.9231] |
| rgcn-fixed-d | 0.8018 ± 0.0016 [0.8001, 0.8048] | **0.7942** ± 0.0037 [0.7871, 0.7975] | 0.8676 ± 0.0122 [0.8498, 0.8876] |
| hgt-matched-d | 0.8015 ± 0.0106 [0.7834, 0.8132] | 0.7824 ± 0.0057 [0.7745, 0.7920] | **0.9335 ± 0.0022** [0.9307, 0.9372] |
| graphsage-matched-d | 0.8032 ± 0.0038 [0.7957, 0.8062] | 0.7898 ± 0.0026 [0.7870, 0.7939] | 0.8889 ± 0.0079 [0.8764, 0.8983] |
| gat-matched-d | 0.7400 ± 0.0089 [0.7237, 0.7495] | 0.6560 ± 0.0238 [0.6220, 0.6926] | 0.9182 ± 0.0064 [0.9069, 0.9259] |
| rgcn-matched-d | **0.8044** ± 0.0031 [0.7993, 0.8079] | **0.7976** ± 0.0021 [0.7956, 0.8004] | 0.8509 ± 0.0119 [0.8418, 0.8728] |

HGT's fixed-d and matched-d rows are identical by construction — `hidden=64` is the anchor for
both arms, unchanged from the existing GraphSAGE/GAT precedent (`run_step5.py`/
`run_step_v3.py`).

## Sign-consistency across seeds — every architecture vs HGT, plus GAT/RGCN vs GraphSAGE

**Fixed-d axis:**

| Task | Comparison | mean ΔAUC | Verdict |
|---|---|---:|---|
| delay | HGT vs GraphSAGE | −0.0038 | FLIPS (noise) |
| delay | HGT vs GAT | +0.0548 | **CONSISTENT** (HGT wins) |
| delay | HGT vs RGCN | −0.0004 | FLIPS (noise) |
| delay | GraphSAGE vs GAT | +0.0586 | **CONSISTENT** (GraphSAGE wins) |
| delay | GraphSAGE vs RGCN | +0.0035 | FLIPS (noise) |
| shortage | HGT vs GraphSAGE | −0.0033 | FLIPS (noise) |
| shortage | HGT vs GAT | +0.1095 | **CONSISTENT** (HGT wins) |
| shortage | HGT vs RGCN | **−0.0118** | **CONSISTENT** (RGCN wins) |
| shortage | GraphSAGE vs GAT | +0.1129 | **CONSISTENT** (GraphSAGE wins) |
| shortage | GraphSAGE vs RGCN | **−0.0085** | **CONSISTENT** (RGCN wins) |
| impact | HGT vs GraphSAGE | +0.0423 | **CONSISTENT** (HGT wins) |
| impact | HGT vs GAT | +0.0197 | **CONSISTENT** (HGT wins) |
| impact | HGT vs RGCN | +0.0659 | **CONSISTENT** (HGT wins) |
| impact | GraphSAGE vs GAT | −0.0226 | **CONSISTENT** (GAT wins) |
| impact | GraphSAGE vs RGCN | +0.0236 | FLIPS (noise) |

**Matched-d axis:**

| Task | Comparison | mean ΔAUC | Verdict |
|---|---|---:|---|
| delay | HGT vs GraphSAGE | −0.0017 | FLIPS (noise) |
| delay | HGT vs GAT | +0.0615 | **CONSISTENT** (HGT wins) |
| delay | HGT vs RGCN | −0.0029 | FLIPS (noise) |
| delay | GraphSAGE vs GAT | +0.0632 | **CONSISTENT** (GraphSAGE wins) |
| delay | GraphSAGE vs RGCN | −0.0012 | FLIPS (noise) |
| shortage | HGT vs GraphSAGE | −0.0074 | FLIPS (noise) |
| shortage | HGT vs GAT | +0.1264 | **CONSISTENT** (HGT wins) |
| shortage | HGT vs RGCN | **−0.0152** | **CONSISTENT** (RGCN wins) |
| shortage | GraphSAGE vs GAT | +0.1338 | **CONSISTENT** (GraphSAGE wins) |
| shortage | GraphSAGE vs RGCN | **−0.0078** | **CONSISTENT** (RGCN wins) |
| impact | HGT vs GraphSAGE | +0.0447 | **CONSISTENT** (HGT wins) |
| impact | HGT vs GAT | +0.0153 | **CONSISTENT** (HGT wins) |
| impact | HGT vs RGCN | +0.0827 | **CONSISTENT** (HGT wins) |
| impact | GraphSAGE vs GAT | −0.0294 | **CONSISTENT** (GAT wins) |
| impact | GraphSAGE vs RGCN | +0.0380 | **CONSISTENT** (GraphSAGE wins) |

Per-seed delta arrays and the paired-bootstrap CI method (`ml/evaluate.py::paired_delta_auc_ci`,
same as every prior round) are in the raw run log,
`reports/logs/run_step_v4_4arch_20260806_231446.log`.

## Who wins each task, at each arm

| Task | Fixed-d winner | Matched-d winner | Notes |
|---|---|---|---|
| **delay** | 3-way statistical tie: HGT / GraphSAGE / RGCN | Same 3-way tie | GAT loses consistently to both HGT and GraphSAGE on both arms. RGCN vs GAT was not directly tested (the comparison set anchors on HGT and GraphSAGE only, per the task's own scope), but RGCN's raw mean (~0.80) clears GAT's (~0.74) by several times either side's own seed-to-seed spread, so the same ranking almost certainly holds. |
| **shortage** | **RGCN**, consistently beating both HGT and GraphSAGE | **RGCN**, same consistent win over both | The cleanest, most reproducible result in the whole round: RGCN's shortage win over HGT and over GraphSAGE holds sign across all 5 seeds, on both arms. New evidence — via a different, more principled mechanism than Task 2's ad hoc `SAGEConv` merge — for the thin-relation-overfitting hypothesis the v3 follow-up first raised. GAT is the clear loser on shortage in both arms. |
| **impact** | **HGT**, decisively | **HGT**, decisively | HGT beats all three other architectures consistently on both arms, by the largest margins in the whole matrix (up to +0.0827 vs RGCN, matched-d). RGCN is impact's *worst* performer on both arms — the same basis-shared relation weights that win it shortage appear to cost it here, where the task apparently needs more per-relation specificity than a shared basis preserves. |

---

# Part II — Findings by Phase

## Phase 0 — Pre-flight

Live DB check (not assumption) confirmed v3 (800 suppliers / 15 snapshots) with Task 3's
`component_suppliers` table present but dormant — see Part I. `pg_dump` backup of
`model_registry` + `model_evaluation_runs` taken unconditionally before any write, per the v3
`--drop` incident's standing lesson.

## Phase 1-2 — RGCN implementation and factory wiring

`ml/models/rgcn_encoder.py` (new) implements `RGCNEncoder`, mirroring `HGTEncoder`/
`HeteroGNNEncoder`'s exact contract (`__init__(metadata, in_dims, hidden, num_layers,
num_bases, dropout)`, `forward(x_dict, edge_index_dict) -> list[dict]`, returning every layer's
per-node-type dict). `"rgcn"` added to `ARCHITECTURES` and dispatched in `build_encoder`
(`ml/models/encoder.py`) alongside the existing `"hgt_sparse"` branch pattern; `num_bases`
threaded through as an optional `build_encoder` kwarg (default 8) that only applies to `rgcn`,
leaving every other call site's signature unaffected — then further threaded through
`HADESModel.__init__` (`ml/models/model.py`) and `train_model`/`run_training_job`
(`ml/train.py`) so the run script can set it per arm, and recorded in each run's
`hyperparameters` JSON (`null` for non-RGCN architectures).

## Phase 3 — Tests

`"rgcn"` added to `test_encoder_forward_pass_shape_and_no_nan`'s parametrize list
(`ml/tests/test_model.py`). New test
`test_rgcn_parameter_count_reflects_basis_sharing` locks in the property the whole
architecture bet rests on directly against the `rel_basis`/`rel_coeff` parameter tensors:
exact-formula assertions (`num_bases·hidden²` and `num_relations·num_bases`), a strict
inequality against a naive fully-dedicated-per-relation matrix count, and — separately —
that `rel_basis`/`rel_coeff` are identically sized between a 1-layer and an 8-layer encoder
(basis sharing is *not* per-layer) even though total parameter count still grows with
`num_layers` (self-loop is per-layer). Full suite: **40/40 passing** — every pre-existing test
plus the new ones, not just the new tests in isolation.

## Phase 4 — Matched-parameter search

See Part I's Parameter counts section for the full table and the search's outcome
(`hidden=138, num_bases=4`, 0.21% off HGT's re-derived 701,088-param anchor).

## Phase 5 — 4-architecture run matrix

`ml/run_step_v4_4arch.py` (new; does not touch `run_step5.py` or `run_step_v3.py`) ran 5 seeds
× 4 architectures × 2 param-arms = 40 training runs, multi-seed built in from the start for
every arm — no repeat of the v1/v2 single-seed architecture-ablation gap
`run_step_v3.py`'s own docstring flags. No depth-config sweep included (Step 6 remains on hold
pending the reach/over-smoothing issues the v3 follow-up already found — this round is
architecture-only, per the task's explicit scope). Completed in **2.49h (8,950s)** wall-clock,
zero errors, logged in full to `reports/logs/run_step_v4_4arch_20260806_231446.log`. Every run
wrote its own `model_registry` row suffixed `-4arch-seed{n}` — verified post-run: 40 new
registry rows (10 per architecture, all `status='active'`) and 600 new evaluation-run rows,
with the pre-run row counts (81 registry / 1,083 evaluation) intact underneath (121 / 1,683
after). No `-v3-seed`, `-sparserelation-v3-seed`, or `-coparent-v4-seed` row was touched.

## Phase 6 — Docs

`docs/project_HADES.md` §8.3's training-runs-required table extended to 4 architectures per
arm (13 → 15 total runs) with a new paragraph documenting RGCN's `num_bases` hyperparameter
and its matched-arm search result. `docs/14_Model_Development_Roadmap.md`'s Step 5 section
extended to describe the 4-architecture, 40-run matrix and RGCN's own bespoke-module rationale.

## Governance record

Backed up before any write (`reports/backups/model_registry_and_evals_backup_20260806_230544.sql`).
40 new `-4arch-seed{n}` registry rows (10 gat / 10 graphsage / 10 heterogeneous_graph_transformer
/ 10 rgcn), all `status='active'` — 121 total registry rows (81 + 40). 600 new evaluation rows —
1,683 total (1,083 + 600). No existing row from any prior round modified.

## Test suite

`ml/tests/` — **40/40 pytest pass** after all changes (Phase 3 above), including the new
basis-sharing lock-in test.

---

*This addendum does not modify `reports/step5_result_v3.md`'s or
`reports/step5_result_v3_followup.md`'s existing conclusions, numbers, or governance rows. Their
Claim 1 and Claim 2 findings stand as reported there; this round adds a fourth architecture arm
on top, with one new qualitative result (RGCN's consistent shortage win) and no reversal of any
prior finding.*
