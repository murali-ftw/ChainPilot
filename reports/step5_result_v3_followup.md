# HADES Model-Development Prototype — Post-v3 Follow-up: Reach, Sparse-Relation, and Real Co-Parent Signal (Tasks 1–3)

**Scope:** three follow-up tasks to `reports/step5_result_v3.md`, run in priority order. Tasks 1–2 are
the core work, against the existing v3 dataset (800 suppliers, 15 monthly snapshots Jul 2024 – Sep
2025). Task 3 is explicitly a separate, optional, larger experiment against a new fourth dataset
generation — it does not touch or revise v3's own numbers. This is an **addendum**: v3's conclusions
in `reports/step5_result_v3.md` are unchanged and are not restated here except where a new
measurement bears directly on them. Same structure as that report: **Part I — Conclusions**, **Part
II — Findings by Task**.

---

# Part I — Conclusions

## Bottom line

Every one of v3's open questions that these three tasks could test came back **negative or
non-committal, never "HGT wins."** Task 1 shows that two of the three depth priors' own theoretical
derivations (`docs/project_HADES.md` §4.2) don't describe either the measured graph or the model
actually built — independent evidence for Claim 2's negative finding, not a repeat of it. Task 2
shows HGT's consistent shortage loss to GraphSAGE can be made to disappear (become a statistical tie)
by merging its thinnest relations' parameters — but not reversed into a win. Task 3, a genuinely
different experiment giving the graph a real causal co-parent signal that provably didn't exist
before, produces the **same qualitative outcome as Task 2** (tie, not win) by a completely different
mechanism. Stated plainly, per the task's own instruction: **no result in this round shows HGT
winning under the 5-seed sign-consistency standard used throughout this project.**

## Task 1 — Were h² / h³ ever the empirically-correct targets, independent of the co-parent question?

**No, not for two of the three tasks, on their own theoretical terms — independent of whether the
co-parent mechanism works.** `docs/project_HADES.md` §4.2 derives each task's structural-prior depth
from "path from a Supplier": delay = h² (children + co-parents), shortage = h³
(`Supplier→Component→Product` + warehouse context), impact = h³ (`…→Product→rev_ORDERED→Order`,
i.e. an Order/Customer target).

| Task | Theoretical target/mechanism (§4.2) | What Task 1 measured | Verdict |
|---|---|---|---|
| **delay** | h², via "children + co-parents" | Shipment (delay's *actual* target) is reachable at **hop 1** directly via `SHIPS_FROM` — the co-parent mechanism isn't needed to reach it at all, and that mechanism is independently confirmed broken (0/800, see below) | h²'s own justification doesn't match either the measured path *or* the measured hop-count needed |
| **shortage** | h³, via Product + warehouse context | Product coverage jumps from sparse (hop 2, mean 8.4/1,280) to majority (hop 3, mean 795/1,280) | **The one prior with solid measured support** — consistent with genuinely needing the extra hop |
| **impact** | h³, target = Order/Customer | Order first appears at hop 3 (mean 378/25,193, 75% of sources) — matches the theoretical derivation precisely | Measures out correctly, **but the impact task actually implemented in Steps 3–5 targets Supplier itself** (`entity_type='supplier'`), not Order/Customer. This validates a prior for a task that was never built; the Supplier-target task actually evaluated has no analogous reach-based derivation — its target coincides with the BFS source, at hop 0. |

**Stated as the task explicitly requires:** h² and h³ were the empirically-correct targets for
**at most one of three tasks** (shortage) on the theory's own terms. Delay's h² justification cites a
mechanism (co-parents) that isn't required to reach its actual target and is separately broken.
Impact's h³ justification describes a task (Order/Customer-targeted) that differs from the one
actually built and evaluated (Supplier-targeted). Even shortage's h³ — the one prior with genuine
measured support — did not survive 5-seed testing in the main v3 round (`reports/step5_result_v3.md`
Claim 2). Net: this independently reinforces Claim 2 from a completely different angle (reach
structure, not depth-sweep AUCs) — the disconnect looks like it's between *"information becomes
available at hop k"* and *"reading from hop k specifically improves prediction,"* not a matter of
picking the wrong hop-count.

**Co-parent finding, reconfirmed:** the base v3 graph still shows **0/800** suppliers reaching a
different supplier in 2 hops via `SUPPLIES`/`rev_SUPPLIES` — `components.supplier_id` is a single
not-null FK, so a round trip can only return to the same supplier. Unchanged from the original v3
round; this is what Task 3 (below) addresses directly, as a separate experiment.

## Task 2 — Does merging HGT's thinnest relations close the shortage gap?

**The consistent loss disappears — it becomes a statistical tie, not a win.** The four thinnest
meta-relations (`MANUFACTURED_AT` 1,933 edges, `SUPPLIES` 2,400, `STOCKED_AT` 3,194, `USED_IN`
7,092 — a real gap before the next-thinnest at 15,608) were merged (not dropped, since two of them
carry shortage's own causal signal) onto one shared `SAGEConv` parameter set, while the other 12
meta-relations kept full per-relation HGT parameterization. Direct test of whether HGT's dedicated
thin-relation parameters were overfitting relative to GraphSAGE's relation-agnostic aggregation.

| Axis | Original hgt-vs-graphsage gap (`reports/step5_result_v3.md`) | hgt_sparse-vs-graphsage, this task |
|---|---|---|
| fixed-d | −0.0123, **CONSISTENT** loss (all 5 seeds negative) | per-seed dAUC `[+0.009, +0.0063, −0.0019, +0.0042, −0.0023]`, mean +0.0031 → **FLIPS (noise)** |
| matched-d | −0.0137, **CONSISTENT** loss | per-seed dAUC `[+0.0058, +0.0018, −0.001, +0.0015, 0.000]`, mean +0.0016 → **FLIPS (noise)** |

Reported exactly as the task specified: **the consistent loss disappeared / became a statistical
tie — not "HGT now wins."** The delta itself isn't seed-consistent in either direction. This is
evidence in favor of the thin-relation-overfitting hypothesis as *part* of the explanation for HGT's
shortage weakness, but not a demonstration that HGT is the better architecture once fixed.

## Task 3 (separate, optional experiment) — Does a real co-parent signal let HGT win?

**No — same qualitative result as Task 2 (tie, not win), reached by an entirely different
mechanism.** Flagged, per the task's own framing, as a genuinely different experiment: a new
`component_suppliers` junction table gives ~17.5% of components (420/2,400) a second qualified
supplier, with a **real causal coupling** (a co-parent partner's own latent stress bleeds into its
partner's — `COPARENT_COUPLING = 0.35` — discoverable only by traversing the co-parent edge, not
from either supplier's own history). This is not present in the base v1/v2/v3 dataset at all; testing
it required a new dataset generation, DB reload, and a distinct `-coparent-v4-seed{n}` set of runs.

Precondition re-verified first: co-parent 2-hop reach on this new graph is **449/800 (56%)**, versus
0/800 on the base v3 graph — the structural condition docs/06 §6.1 always assumed is now genuinely
true.

| Task | hgt-coparent vs graphsage-coparent, per-seed ΔAUC (paired bootstrap, 5 seeds) | Mean | Consistency |
|---|---|---|---|
| shortage | `[+0.0048, −0.0171, −0.0037, +0.0089, −0.0095]` | −0.0033 | **FLIPS (noise)** |
| delay | `[−0.0090, −0.0195, −0.0017, +0.0016, +0.0095]` | −0.0038 | **FLIPS (noise)** |

Neither task shows a consistent HGT advantage even with a real, reachable co-parent signal newly
present in the graph. **Per the explicit instruction: this is not reframed as "HGT wins"** — the
result is inconclusive/noise on both tasks, the same qualitative category as Task 2's finding, not a
repeat of the original consistent loss either (the loss itself also disappeared here). Two
independent interventions — merging thin relations (Task 2) and adding a real relational signal HGT
alone should be positioned to exploit (Task 3) — both land in the same place: a tie, not a win. This
is the strongest evidence yet in this project that HGT's shortage weakness isn't a data-availability
problem (missing signal, wrong parameterization) so much as it's a genuine architecture-vs-task
mismatch that these interventions don't resolve.

**Reminder of scope:** Task 3's dataset is a separate, fourth generation — not "v3 plus one extra
table." Because the co-parent coupling feeds directly into shipment-delay branching (a
control-flow-dependent random draw), the entire generated world differs from base v3 from that point
forward, not just the newly-added edges. It is still fully deterministic (byte-identical across
reruns) and passes every existing validation check plus three new ones. v3's own findings and numbers
are unaffected — they live on in `reports/step5_result_v3.md` and their governance rows, both intact.

---

# Part II — Findings by Task

## Task 1 — Restricted k-hop reach (Supplier/Order-sourced)

The full all-node-type reach BFS from every node type died on Shipment's 35,687-node sweep in the
original v3 round. Restricted here to exactly the two sources the depth priors cite
(`docs/project_HADES.md` §4.2's "path from a Supplier" table): Supplier-sourced (all 800) and
Order-sourced (sample of 200/25,193). Completed in 37.8s.

**Supplier-sourced (all 800), hop ≤ k:**

| k | Component | Product | Factory | Warehouse | Shipment | Order | Supplier | Customer |
|---|---|---|---|---|---|---|---|---|
| 1 | mean 3.00, 80.5% reached | — | — | — | mean 19.51, 73.9% reached | — | — | — |
| 2 | 3.00, 80.5% | mean 8.39, 75.0% | — | mean 3.69, 73.9% | 19.51, 73.9% | — | — | — |
| 3 | mean 37.53, 80.5% | mean 794.93, 75.0% | mean 2.81, 75.0% | mean 4.54, 75.0% | mean 16,457.03, 73.9% | mean 378.21, 75.0% | — | — |
| 4 | mean 1,363.85, 80.5% | mean 932.30, 75.0% | mean 3.71, 75.0% | mean 5.94, 75.0% | mean 22,193.11, 75.0% | mean 17,284.49, 75.0% | mean 400.97, 75.0% | mean 238.32, 75.0% |

**Order-sourced (sample 200/25,193), hop ≤ k:** Product/Customer reach 100% at hop 1; Component
reaches 100% at hop 2 (mean 13.59); Supplier first appears at hop 3 (mean 11.96, 100%); Shipment
climbs from 0.72 (hop 1, 72%) to a near-complete sweep by hop 4.

**Co-parent check:** 0/800 suppliers reach a different supplier in 2 hops (unchanged from the
original v3 round; `components.supplier_id` is a single not-null FK). See Part I for the per-task
comparison against `docs/project_HADES.md` §4.2's theoretical derivations, and the plain statement
on whether h²/h³ were ever the correct empirical targets.

## Task 2 — Sparse-relation-merged HGT encoder

**Thinnest meta-relations identified** (edge count at the largest snapshot): `MANUFACTURED_AT`
1,933, `SUPPLIES` 2,400, `STOCKED_AT` 3,194, `USED_IN` 7,092 — a real gap before the next-thinnest,
`SHIPS_FROM`→Supplier at 15,608 (more than double `USED_IN`'s count).

**Encoder design** (`ml/models/sparse_hgt_encoder.py`): merging required a custom encoder, not a
`metadata` relabel — `HGTConv`'s per-relation weights are keyed strictly by the full
`(src_type, rel, dst_type)` triple, and the four thin relations don't even share a common source or
destination type. The actual merge mechanism: one `SAGEConv(hidden, hidden)` instance per layer,
passed into `HeteroConv` under all 8 thin-relation keys (4 forward + 4 reverse) — `nn.ModuleDict`
registers it under multiple string keys, but since it's the same Python object, its parameters and
every gradient update are genuinely shared across all 8. The other 12 meta-relations keep full
per-relation HGT parameterization via a second `HGTConv` restricted to just those 12. Verified via a
direct parameter-count comparison: 668,448 (sparse) vs. 701,088 (plain hgt) encoder parameters, plus
a passing NaN/shape integration test (`test_sparse_hgt_encoder_forward_pass_and_param_reduction`,
added to `ml/tests/test_graph.py`).

**Training:** 5 seeds, 1 run each (`hgt`'s own fixed-d/matched-d entries were identical in the
original ablation — d never changes for the anchor architecture — so `hgt_sparse`'s single d=64 run
serves both comparisons).

| Task | hgt_sparse AUC (mean±std, 5 seeds) |
|---|---|
| delay | 0.7220±0.0170 |
| shortage | 0.7874±0.0024 (tighter than plain hgt's 0.772±0.009) |
| impact | 0.9148±0.0083 |

See Part I for the shortage-vs-GraphSAGE delta table and verdict.

## Task 3 — Real causal co-parent signal (separate, optional experiment)

**Schema** (`db/schema.sql`): new `component_suppliers` junction table, mirroring
`product_components`'s validity-window pattern — secondary qualified suppliers for a component,
beyond the mandatory `components.supplier_id` primary.

**Generator** (`db/generate_dataset.py`): 420/2,400 components (17.5%, within the requested 15–20%
range) get a second qualified supplier, chosen via a dedicated local RNG (`_coparent_rng`) isolated
from the main simulation's shared `random` stream. The causal mechanism: `stress()` was split into
`own_stress()` (a supplier's own base reliability + shared-hidden-factor events + idiosyncratic
outage — the pre-Task-3 body, unchanged) and a new `stress()` that adds `COPARENT_COUPLING = 0.35`
of each co-parent partner's `own_stress` (never the partner's *coupled* stress, which would create a
mutual A-depends-on-B-depends-on-A recursion). Suppliers with no co-parent partner (~82.5%) see
identical behavior to base v3.

Three new validation checks added to the generator's own validation suite (all pass): dual-sourced
fraction in the requested [0.15, 0.20] range (measured 17.5%), FK integrity with secondary ≠ primary,
and mutual co-parent partnerships. All 22 checks (19 pre-existing + 3 new) pass; the dataset is
byte-identical across reruns (verified directly).

**Builder** (`ml/graph/builder.py`): the `SUPPLIES`/`rev_SUPPLIES` edge SQL now unions the mandatory
primary (`components.supplier_id`) with any `component_suppliers` rows valid as of `t0`. No-op on a
graph with an empty `component_suppliers` table (base v3); only adds edges on this new dataset.

**Reach re-verified:** co-parent 2-hop reach is 449/800 (56.1%) on this graph, vs. 0/800 on base v3.

**Training** (`ml/run_task3_coparent.py`): 5 seeds × 2 arms (hgt, graphsage) at matched hidden=64,
fixed-d only (the restricted comparison requested, not a repeat of the full 6-arm ablation matrix).
Both trained in-process, so shortage/delay deltas use the same paired-bootstrap method
(`ml/evaluate.py::paired_delta_auc_ci`) as Steps 4/5 and the main v3 round.

| Arm | delay (mean±std) | shortage (mean±std) | impact (mean±std) |
|---|---|---|---|
| hgt-coparent | 0.8015±0.0106 | 0.7824±0.0057 | 0.9335±0.0022 |
| graphsage-coparent | 0.8053±0.0040 | 0.7857±0.0047 | 0.8912±0.0046 |

See Part I for the paired per-seed delta table and verdict (shortage and delay both FLIP — no
consistent HGT advantage).

## Governance record

**Task 2:** backed up `model_registry`/`model_evaluation_runs` (pg_dump + in-DB copy tables) before
any write. 5 new `-sparserelation-v3-seed{n}` rows, no existing row touched — 71 total registry rows
(66 + 5).

**Task 3:** backed up governance tables (pg_dump + in-DB copy tables) before the `--drop` reload this
experiment required. `--drop`'s `DROP SCHEMA public CASCADE` wiped the in-DB copy tables too (same
schema, same cascade) — restored both tables from the pg_dump file immediately after, confirmed
71 registry / 933 evaluation rows intact before re-creating fresh in-DB copies for the rest of the
experiment. 10 new `-coparent-v4-seed{n}` rows, no existing row touched — 81 total registry rows
(71 + 10).

## Test suite

38/38 `pytest` pass after both Task 2 and Task 3. `ruff check ml/` clean throughout. `db/`'s
pre-existing, unenforced lint style (21 `C408`-class notes before this round) gained exactly one more
instance from Task 3's new code, consistent with the surrounding file's established pattern — not a
new category of issue.

---

*This addendum does not modify `reports/step5_result_v3.md`'s existing conclusions, numbers, or
governance rows. v3's Claim 1 and Claim 2 stand as reported there; Tasks 1–3 add independent
evidence on top, none of it reversing the direction of any v3 finding.*
