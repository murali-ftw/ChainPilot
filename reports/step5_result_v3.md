# HADES Model-Development Prototype — Findings & Conclusions, Steps 0–5 (v3)

**Scope:** rerun Steps 0–5 against the v3 dataset (800 suppliers, 15 monthly snapshots Jul 2024 –
Sep 2025), with multi-seed evaluation built into **both** Step 4 (depth sweep) and Step 5
(architecture ablation) from the start — the prior round only did this for Step 4 (`ml/run_step_b.py`),
which is exactly what caught three single-seed "significant" findings that were noise. This
report has two parts: **Part I — Conclusions** and **Part II — Findings by Step**.

---

# Part I — Conclusions

## Bottom line

The multi-seed-from-the-start design paid for itself immediately: it caught that **v2's own
cleanest, most-cited Claim 2 result — "the structural depth prior beats shared depth on
shortage" — does not survive proper 5-seed scrutiny.** That finding was reported as decisive in
`reports/step5_result_v2.md` on the strength of a single-seed bootstrap CI; under 5 seeds, the
sign flips on all but one of the seven prior-vs-shared-depth comparisons across all three tasks.
Claim 1 (architecture ablation), by contrast, comes out of this round *more* decisive than
before — several comparisons are now sign-consistent across all 5 seeds, including one (HGT
beats GAT on delay) that v2's single seed reported as "not distinguishable."

## Claim 1 — Architecture ablation: v2 vs v3, now seed-backed

v2 used single-seed bootstrap-CI significance. v3 uses 5-seed sign-consistency (a comparison is
"real" only if all 5 seeds' point-estimate deltas agree in sign — the same standard
`reports/step5_result_v2.md`'s Step B established).

| Task | v2 fixed-d | **v3 fixed-d (5-seed)** | v2 matched-d | **v3 matched-d (5-seed)** |
|---|---|---|---|---|
| delay | not distinguishable (both) | HGT≈SAGE (flips); **HGT beats GAT (consistent, +0.061)** | not distinguishable | HGT≈SAGE (flips); **HGT beats GAT (consistent, +0.065)** |
| shortage | HGT wins both (sig) | **HGT loses to SAGE (consistent, -0.012)**; HGT beats GAT (consistent, +0.119) | HGT loses SAGE, beats GAT | HGT loses to SAGE (consistent, -0.014); HGT beats GAT (consistent, +0.121) |
| impact | HGT beats SAGE, ties GAT | HGT beats SAGE (consistent, +0.068); HGT≈GAT (flips) | HGT beats SAGE, ties GAT | HGT beats SAGE (consistent, +0.080); HGT≈GAT (flips) |

**What's new and real:** HGT beating GAT on delay is a genuinely new, seed-robust finding — v2's
single seed called delay "not distinguishable" everywhere. **What got more honest, not more
favorable:** v2's fixed-d axis showed "HGT wins shortage on both baselines"; v3 shows that
win only holds against GAT — against GraphSAGE, HGT is a small but *consistent* loser on
shortage at **both** fixed-d and matched-d, not just matched-d as v2 found. The direction v2
reported (HGT winning shortage vs SAGE at fixed-d) does not survive 5-seed averaging.

**Stated plainly, per task, both axes:**
- **delay:** HGT ≈ GraphSAGE (genuinely tied); **HGT reliably beats GAT** — real, new this round.
- **shortage:** **HGT reliably beats GAT** (large effect, ~+0.12); **HGT reliably loses to
  GraphSAGE** (small effect, ~-0.01 to -0.014, but consistent across every seed on both axes).
- **impact:** **HGT reliably beats GraphSAGE**; HGT ≈ GAT (genuinely tied, not a confound this
  time — it flips both ways with no directional pattern).

## Claim 2 — Depth prior: v2's decisive result does not replicate

| Task | v2 (single-seed, prior vs L1-L4) | **v3 (5-seed sign-consistency)** |
|---|---|---|
| delay | all not significant | all **FLIP (noise)** |
| shortage | worse vs L1 (sig); **better vs L2, L3, L4 (all sig)** | worse vs L1: **flips**; vs L2/L3/L4: **all flip** — only **L3-vs-L2 itself** (not prior-vs-L3) is consistent, and it's small (-0.011) |
| impact | all not significant (clean null) | all **FLIP (noise)** |

**This is the single most important finding of the v3 round.** v2's report called shortage
"the clearest win from the dataset fix... the structural prior earns its place." Under 5-seed
sign-consistency, **none of the four prior-vs-shared-depth comparisons hold up** — every one
flips sign across seeds, exactly like the three findings Step B caught in v1. v2's result wasn't
wrong to compute — the single bootstrap CI was numerically correct for *that one trained
model* — but it was never a safe basis for "the prior earns its place." **Claim 2 remains
unresolved for all three tasks, now on stronger evidence than either prior round had.** The one
survivor (shortage L3 < L2, small effect) is a shared-depth-vs-shared-depth comparison, not a
finding about the structural prior at all.

## The dataset fix: sufficient for stability, not for every open question

- **delay:** test n=4,107, 339 positives. AUC stable across seeds (std 0.007–0.015 at every
  config) — a real, trustworthy point estimate, even though several architecture/depth
  comparisons on it still don't resolve.
- **shortage:** test n=19,164, 957 positives — comfortably above any CI-table floor. Stable
  (std 0.004–0.011) and now the best-powered task for architecture comparisons; its depth-prior
  story, though, is the one that got *walked back* this round.
- **impact:** test n=4,800, **114 positives — does not clear 200**, the CI-table floor for a
  clearly detectable effect. But **impact's seed-to-seed AUC std (0.004–0.010) is now the
  tightest of the three tasks** — tighter than delay's or shortage's. Positive *count* and
  model-fit *stability* are not the same thing: v2's impact already fixed the wild v1 instability
  (std 0.11–0.26 there); v3's much larger supplier pool (800, not 180) stabilized it further even
  though the test-split label count only grew from 21 to 114, well short of 200. **Verdict:**
  impact now supports a real, stable point-estimate verdict on architecture (HGT beats SAGE,
  ties GAT, consistently) even though its raw positive count alone wouldn't predict that.

**Is further scaling needed before Steps 6–8?** For delay/shortage, no — both are well-powered
and stable now. For impact, the *positive count* is still short of the nominal 200 floor, but
the *actual measured stability* says the task is usable as-is; citing both numbers together
(114 positives, std ≈0.005–0.01) is the honest way to report it, rather than either number alone.

## Split boundaries: why Jul–Dec 2024 / Jan–Mar 2025 / Apr–Sep 2025

The per-snapshot label table (see Part II) shows an extreme spike in Oct–Nov 2024 (delay 248/194
vs. a 26–74 baseline every other month). Reusing v1/v2's "last N snapshots as test" convention
literally would have put that spike inside validation, isolated from both train and test. Chosen
instead: **train = Jul–Dec 2024** (6 snapshots, includes the spike so the model trains on the
disrupted regime), **val = Jan–Mar 2025** (3), **test = Apr–Sep 2025** (6). This was checked
explicitly, not assumed — full table in Part II.

## Graph construction and training cost: the real bottleneck is training, not assembly

Building the largest v3 snapshot (66,973 nodes, 388,242 edges) takes **0.68s** — construction
was never the constraint. Training is: one real 100-epoch HGT run on the actual split took
**421.8s (~7 min)**, about **5.3x** v2's ~80s. This was measured directly (not extrapolated)
before committing to the 55-run matrix, flagged to the user as an explicit checkpoint, and run
as approved. The full 55-run matrix (25 depth-sweep + 30 architecture-ablation, 5 seeds each)
completed in **4.61 hours**.

## Operational note: a governance-table mistake, disclosed and partially repaired

Step 0's mandatory `db/load_data.py --drop` does `DROP SCHEMA public CASCADE`, which destroys
`model_registry`/`model_evaluation_runs` along with the data tables. This round's 55 new runs
were correctly given distinct `-v3-seed{n}`-suffixed `model_version`s (never upserted over v2's
names), but the `--drop` itself destroyed v2's, Step A's, and Step B's existing rows as a side
effect — the same end state the "don't repeat the v1→v2 mistake" instruction was trying to
prevent, reached by a different mechanism. This was caught, disclosed immediately, and partially
repaired: v2's 11 core rows (Steps 3–5) were reconstructed from this session's own verbatim run
output — cross-checked against the parameter counts already published in
`reports/step5_result_v2.md` — and re-inserted with `status='archived'` and an explicit
`RECONSTRUCTED` marker in `git_commit` and `hyperparameters`, so they're queryable again but
never mistakable for a live rerun. Step A's (2 rows) and Step B's (25 rows) were not
reconstructed — they remain report-only, per the accepted lower-priority option. **Lesson for
next time:** back up governance tables before any `--drop`, not just avoid name collisions in the
new run.

---

# Part II — Findings by Step

## Step 0 — Load

`db/load_data.py --drop` against the v3 CSVs: 19/19 tables, 7/7 verification checks pass. Real
DB label rates (not assumed from generation): delay 1,066/11,168 (9.55%), shortage
2,875/47,910 (6.00%), impact 279/12,000 (2.33%) — matches generation-time output exactly, no
loader-side discrepancy.

## Step 1 — Leakage

3/3 pass. No leak (gate: AUC > 0.90). Strongest solo features: `days_to_eta` 0.7282 (delay),
`min_stock_ratio` 0.8929 (shortage — closer to the gate than v2's 0.86, still under it),
`days_since_last_late` 0.6898 (impact).

## Step 2 — Graph Correctness

- Feature dimensions **recomputed, not assumed**: identical to v2 (Supplier 14, Component 6,
  Product 13, Factory 6, Warehouse 7, Shipment 7, Order 5, Customer 3) — category cardinalities
  didn't change, only population sizes did. 20 meta-relations confirmed.
- All 15 `graph_snapshots` rows persisted with real node/edge counts (22,155 nodes at the first
  snapshot growing to 66,973 at the last; edges 122,594 → 388,242).
- Co-parent 2-hop reach finding reproduces: still structurally false (0/800 suppliers reach a
  different supplier in 2 hops via SUPPLIES/rev_SUPPLIES — `components.supplier_id` is still a
  single FK). The full 4-hop reach table (all node types) was not completed — BFS from all
  35,687 Shipment nodes at this scale ran long enough that it was killed in favor of running just
  the bounded co-parent check (Supplier-sourced, 2 hops), which is what Step 2 actually requires;
  the broader diagnostic table is not reported this round.
- One test fixture needed updating: `ml/tests/test_graph.py`'s BOM-substitution unit test
  referenced component UUIDs that shifted under the new RNG draw order (same product, same date,
  different component id) — re-derived from the live database, not weakened.

## Step 3/4 — Baseline + Depth-Config Sweep (5 seeds x 5 configs = 25 runs)

Per-snapshot positive-label table, pulled from the loaded DB (the basis for the split decision):

| t0 | delay | shortage | impact |
|---|---|---|---|
| 2024-07 | 26 | 132 | 13 |
| 2024-08 | 26 | 119 | 12 |
| 2024-09 | 55 | 206 | 14 |
| **2024-10** | **248** | **459** | 28 |
| **2024-11** | **194** | **383** | 28 |
| 2024-12 | 74 | 216 | 25 |
| 2025-01 | 27 | 119 | 11 |
| 2025-02 | 43 | 139 | 20 |
| 2025-03 | 34 | 145 | 14 |
| 2025-04 | 50 | 179 | 18 |
| 2025-05 | 51 | 144 | 21 |
| 2025-06 | 61 | 138 | 23 |
| 2025-07 | 64 | 133 | 14 |
| 2025-08 | 59 | 139 | 16 |
| 2025-09 | 54 | 224 | 22 |

Split: train Jul–Dec 2024 (delay 623, shortage 1,515, impact 120), val Jan–Mar 2025 (delay 104,
shortage 403, impact 45), test Apr–Sep 2025 (delay 339, shortage 957, **impact 114**).

**AUC spread across 5 seeds** (full per-seed values in `ml/run_step_v3.py`'s log):

| Config | delay (mean±std) | shortage (mean±std) | impact (mean±std) |
|---|---|---|---|
| baseline (fixed prior) | 0.744±0.013 | 0.772±0.009 | 0.920±0.009 |
| L1 | 0.739±0.007 | 0.780±0.006 | 0.908±0.004 |
| L2 | 0.734±0.015 | 0.782±0.004 | 0.912±0.010 |
| L3 | 0.731±0.010 | 0.771±0.011 | 0.923±0.010 |
| L4 | 0.730±0.015 | 0.765±0.010 | 0.916±0.006 |

Every config/task cell has std well under 0.02 — a dramatic improvement over v2's impact std of
0.004–0.010 was already fine, but v1's 0.11–0.26 (Step B) is not even in the same regime. The
dataset fix's biggest stability win is visible here directly.

**Significance, per-seed sign-consistency (7 comparisons x 3 tasks = 21 total):** only **1 of 21**
is consistent (shortage L3-vs-L2, mean -0.011). All 4 prior-vs-shared-depth comparisons flip for
all 3 tasks. See Part I for the comparison against v2's claimed shortage result.

## Step 5 — Architecture Ablation (5 seeds x 6 arms = 30 runs)

Matched-`d` parameter counts reverified against v3's real dimensions: HGT d=64 → 701,088
(encoder only), GraphSAGE d=66 → 706,794 (+5,706), GAT d=92 → 705,548 (+4,460) — unchanged from
v2's search, as expected (dimensions didn't change).

**AUC spread across 5 seeds:**

| Arm | delay (mean±std) | shortage (mean±std) | impact (mean±std) |
|---|---|---|---|
| hgt-fixed-d | 0.744±0.013 | 0.772±0.009 | 0.920±0.009 |
| graphsage-fixed-d | 0.749±0.007 | 0.784±0.003 | 0.852±0.010 |
| gat-fixed-d | 0.683±0.007 | 0.654±0.022 | 0.908±0.006 |
| hgt-matched-d | 0.744±0.013 | 0.772±0.009 | 0.920±0.009 |
| graphsage-matched-d | 0.747±0.011 | 0.786±0.001 | 0.840±0.012 |
| gat-matched-d | 0.678±0.009 | 0.651±0.013 | 0.904±0.004 |

(hgt-fixed-d and hgt-matched-d are identical by construction — HGT's own d=64 is the anchor for
both axes, same as every prior round.)

**Significance, per-seed sign-consistency (6 comparisons per axis x 2 axes = 12 total):** **8 of
12 consistent** — see Part I's table for the full breakdown per task/axis. This is the opposite
pattern from Claim 2: architecture comparisons got *more* decisive with proper seeding, not less.

## Governance record (v3, current DB state)

66 `model_registry` rows total: 55 new (`status='active'`, this round's runs) + 11 reconstructed
v2 rows (`status='archived'`, explicitly marked `RECONSTRUCTED`). 858 `model_evaluation_runs`
rows. Every v3 run is queryable by its exact `model_version` (e.g. `hgt-baseline-v3-seed0`
through `hgt-baseline-v3-seed4`).

## Test suite

37/37 `pytest` pass. `ruff check ml/` clean.

---

*Part 2 (documentation updates to `db/README.md`, `project_HADES.md`, the PRD, the TRS,
`06_Graph_Database_Design.md`, and `14_Model_Development_Roadmap.md`) follows in the same
commit as this report, using the measured numbers above as the source.*
