# HADES Model-Development Prototype — Combined Report (v1 → v5)

**This is a merge of six prior reports into one chronological document, replacing all of
them:** `steps_0-5_findings.md` ("v1"), `step5_result_v2.md`, `step5_result_v3.md`,
`step5_result_v3_followup.md` (Tasks 1–3), `step5_result_v4_4arch.md` (RGCN 4-arch), and
`rgcn_attn_pilot.md` (RGCN+attn pilot, "v5"). Nothing in this merge changes any number,
verdict, or conclusion from any original report — it reorganizes six separately-written
documents into one continuous narrative, trims duplicated process boilerplate (the same
"how to read this report" scaffolding, repeated code-inventory blocks, etc.), and adds a
short synthesis at the top connecting the rounds. Where a later round revised or overturned
an earlier round's finding, both are shown side by side, exactly as the original follow-up
reports did — this document does not retroactively "correct" v1's or v2's own numbers.

**Reading order:** an Executive Summary (the final state of every claim as of the most
recent round), a Master Timeline (which dataset/codebase state each round ran against), then
one section per round in the order they actually happened, then a single consolidated
governance/test-suite record and open-items list at the end.

---

# Executive Summary — Final State as of the Most Recent Round (RGCN+attn Pilot)

## Claim 1 — Does type-aware attention beat type-blind baselines?

**Task-dependent, and the answer changed materially across rounds as label volume, seed
count, and the architecture roster all grew.** As of the latest evidence (5-seed, v3 dataset,
now with RGCN and RGCN+attn added as 4th/5th arms):

| Task | Verdict (latest evidence) |
|---|---|
| delay | Historically undecided (v1–v3: not distinguishable, or HGT-vs-GAT only). **RGCN+attn's pilot is the first architecture to win delay consistently against both HGT and plain RGCN** — a new result, fixed-d only, not yet matched-d validated. |
| shortage | v1: HGT won cleanly. v2 (matched-d): reversed — GraphSAGE beat HGT. v3 (5-seed): confirmed HGT loses to GraphSAGE consistently, both axes. v3-followup: merging HGT's thinnest relations closes the loss to a tie (not a win). **v4: RGCN beats both HGT and GraphSAGE consistently, on both axes — the first architecture-level win to survive every check.** v5: RGCN+attn keeps that shortage win over HGT. |
| impact | v1: confounded by capacity, resolved to a tie once matched. v2/v3: HGT reliably beats GraphSAGE, ties GAT. **v4: HGT beats all three other architectures (including RGCN, RGCN's worst task) by the largest margins in the whole matrix. v5: RGCN+attn closes RGCN's impact deficit against HGT entirely** (from a consistent large loss to a statistical tie). |

**Net effect of the whole architecture line of work:** HGT's early "wins everywhere" story
(v1) narrowed under better statistics (v2/v3) to "wins impact, loses shortage to GraphSAGE."
Two independent interventions on *why* HGT loses shortage — merging its thinnest relations'
parameters (v3-followup Task 2) and giving the graph a real causal co-parent signal
(v3-followup Task 3) — both turned that loss into a tie, never a win, suggesting the
weakness is a genuine architecture-vs-task mismatch, not a fixable data or parameterization
gap. RGCN (v4) then won shortage outright via a structurally different form of sharing
(basis decomposition), and RGCN+attn (v5) — a shared-attention hybrid on top of RGCN — is
the first architecture in this entire project to have the best mean AUC on all three tasks
simultaneously, at roughly a quarter of HGT's parameter count. That result is fixed-d only
and needs a matched-parameter follow-up before it can be called a resolved win.

## Claim 2 — Does the structural depth prior (h²/h³) beat a single shared depth?

**Unresolved, on steadily improving evidence, trending toward "no."** v1: inconclusive,
single seed, low label volume. v2: appeared resolved for shortage (prior beats every shared
depth L2–L4). v3 (5-seed sign-consistency): **that shortage result did not replicate** —
every prior-vs-shared-depth comparison flips sign across seeds, for all three tasks; the one
survivor is a shared-depth-vs-shared-depth comparison, not a finding about the prior at all.
v3-followup Task 1 independently reinforced the negative finding from a completely different
angle (measured k-hop reach vs. the prior's own theoretical derivation): the theory's own
justification doesn't hold for two of the three tasks' targets as actually built. **Claim 2
remains open, now on its strongest evidence yet that the prior does not earn its place over a
shared depth.** Step 6 (the learned depth gate) has been on hold since the v3-followup for
exactly this reason — there is no validated prior to anchor a small learned correction to.

## Architecture roster and parameter-count landscape (current)

| Architecture | Encoder params (hidden=64) | Introduced | Status |
|---|---:|---|---|
| HGT | 701,088 | v1 | Baseline throughout; wins impact consistently from v2 onward |
| GraphSAGE | 664,896 (677,571 matched@66) | v1 | Beats HGT on shortage from v2 onward |
| GAT | 347,456 (705,548 matched@92) | v1 | Consistently weakest architecture on delay/shortage across every round |
| RGCN (basis decomposition) | 170,464 | v4 | Wins shortage consistently vs. both HGT and GraphSAGE; worst on impact |
| **RGCN+attn ("Option 1")** | 170,984 | v5 (pilot) | Best mean AUC on all 3 tasks simultaneously, fixed-d only; matched-d not yet run |

---

# Master Timeline — Dataset and Codebase State Per Round

| Round | Report section below | Dataset | Suppliers | Snapshots | What changed |
|---|---|---|---|---|---|
| v1 | Round 1 | Original synthetic dataset | 50 | 6 (Jul–Dec 2024) | Steps 0–5 built from scratch; Step A follow-up (corrected impact prior) |
| v2 | Round 2 | Regenerated: scale + 3 targeted fixes | 180 | 6 (Jul–Dec 2024) | Step B (multi-seed depth sweep only), Step C (dataset regeneration), Step D (full Steps 0–5 rerun) |
| v3 | Round 3 | Regenerated again: much larger scale, longer horizon | 800 | 15 (Jul 2024–Sep 2025) | Multi-seed built into **both** Step 4 and Step 5 from the start |
| v3-followup | Round 4 | Same v3 dataset (Tasks 1–2); a **separate, fourth dataset generation** for Task 3 only (co-parent signal) | 800 (Tasks 1–2); 800 w/ dual-sourcing (Task 3 only) | 15 | Task 1: restricted k-hop reach measurement. Task 2: sparse-relation-merged HGT encoder. Task 3: real causal co-parent signal via new `component_suppliers` table |
| v4 (4-arch) | Round 5 | Same v3 dataset, `component_suppliers` table present but dormant (no dual-sourcing) | 800 | 15 | RGCN added as 4th architecture arm; full 40-run multi-seed matrix (4 arch × 2 param-arms × 5 seeds) |
| v5 (RGCN+attn pilot) | Round 6 | Same v3 dataset, confirmed unchanged | 800 | 15 | RGCN+attn ("Option 1") added as 5th architecture; 15-run fixed-d-only pilot (3 arch × 5 seeds) |

Every round from v3 onward ran against the same underlying dataset generation (v3: 800
suppliers / 15 snapshots) — only the encoder architecture roster and analysis methodology
changed from Round 3 onward, not the data itself (except Task 3's own separate, optional
co-parent dataset, explicitly scoped as not touching v3's own numbers).

---

# Round 1 — v1 (Steps 0–5, original dataset)

**Dataset:** 50 suppliers, 6 monthly snapshots (Jul–Dec 2024). **Scope:** build the pipeline
end to end for the first time and get an initial read on Claims 1 and 2.

## Findings by step

**Step 0 — Environment and data audit.** Installed Python 3.11.15 and PostgreSQL 15.18 from
scratch. `db/load_data.py --drop` loads all 19 tables and passes all 7 verification queries.
Label rates matched `db/README.md` exactly: delay 10.97% (34 positive/310), shortage 7.22%
(84/1,164), impact 3.67% (11/300). **Open discrepancy, never resolved:** two independent
attempts to reproduce `db/README.md`'s claimed single-feature AUC (~0.59–0.63 for
`on_time_rate_90d`) both landed near chance (~0.47–0.48) — flagged, not blocking.

**Step 1 — Leakage contract.** Exclusion list (`shipments.status`, `shipments.delivered_at`,
`inventory.stock_level`, `suppliers.reliability_history`) enforced structurally, verified by
grep. **A real bug caught and fixed:** the first feature-extraction pass filtered
suppliers/products by `created_at <= t0`, which silently zeroed both tables — `created_at` on
reference tables turned out to be dataset-generation metadata, not a business timestamp.
Single-feature leakage test (gate: AUC > 0.90) found no leak: `min_stock_ratio` on shortage
was the closest at 0.88, a legitimate direct predictor, not leaked information.

**Step 2 — Graph correctness.** Real per-node-type feature dimensions recomputed against the
actual data, differing from the design docs' placeholder estimates in every case (e.g.
Supplier 14 not 21, Product 13 not 8). 20 meta-relations confirmed (10 forward +
`ToUndirected()`'s 10 reverse). **A documented architectural claim checked and found false:**
`docs/06_Graph_Database_Design.md` §6.1 claims a supplier's "co-parents" are reachable in 2
hops via `SUPPLIES → rev_SUPPLIES` — measured directly across all 50 suppliers, 0 reached a
different supplier in 2 hops, because `components.supplier_id` is a single not-null FK. This
finding persisted, unchanged, through every subsequent round.

**Step 3 — HGT baseline + governance tables.** `db/governance_schema.sql` applied
(`model_registry`, `model_evaluation_runs`, `explanation_subgraphs`). HGT via
`torch_geometric.nn.HGTConv` (L=4, d=64, heads=4), fixed structural depth-prior readout
(delay→h², shortage/impact→h³), 2-layer MLP heads, focal loss. Strict time split: train
Jul–Sep 2024 (3 snapshots), val Oct 2024 (1), test Nov–Dec 2024 (2).

| Task | AUC | 95% CI | n | positives |
|---|---|---|---|---|
| delay | 0.6734 | [0.535, 0.811] | 142 | 17 |
| shortage | 0.7258 | [0.638, 0.801] | 388 | 37 |
| impact | 0.4010 | [0.031, 0.899] | 100 | 4 |

impact's CI spans almost the entire [0,1] range — a legitimate "inconclusive at this label
volume" result, not hidden.

**Step 4 — Depth-prior sweep (Claim 2, part 1).** Layer-depth sweep L=1..4 against the fixed
structural-prior baseline, single seed. Over-smoothing measurement showed most node types
drift toward embedding collapse with depth (Factory 0.72→0.92, Customer 0.78→0.95) but
**Shipment did the opposite** (0.835→0.668), an unpredicted genuine finding. Nested
significance tests found a handful of "significant" comparisons, but they flip sign between
adjacent L values (impact: +0.292 then −0.292) — the signature of single-seed variance at
very low positive counts, not a real depth effect. **Claim 2 (prior half) inconclusive at
this label volume.**

**Step 5 — Architecture ablation (Claim 1, part 1).** Six runs: {GraphSAGE, GAT, HGT} ×
{fixed-d=64, matched-parameter}. Engineering correction made before running: GAT's default
multi-head attention gave it *more* parameters than GraphSAGE at equal d, contradicting the
ablation's own premise — switched to single-head GAT. Matched-d values recomputed against
real dimensions (not the doc's placeholder-derived 86/116): GraphSAGE d=66, GAT d=92, all
within ~1% of HGT's 701K-parameter encoder.

| Task | graphsage-fixed-d | gat-fixed-d | hgt-fixed-d | graphsage-matched-d | gat-matched-d | hgt-matched-d |
|---|---|---|---|---|---|---|
| delay | 0.640 | 0.655 | 0.673 | 0.721 | 0.642 | 0.673 |
| shortage | 0.677 | 0.480 | **0.726** | 0.527 | 0.681 | **0.726** |
| impact | 0.823 | 0.776 | 0.401 | 0.826 | 0.732 | 0.401 |

**delay:** not distinguishable, every pairwise CI overlaps zero. **shortage: HGT wins,
robustly**, beats both baselines at fixed-d, still beats GraphSAGE at matched-d. **impact:**
HGT loses to GAT at fixed-d, but that loss disappears once GAT is parameter-matched — the
matched-parameter control catching a capacity confound, not a real architecture effect.

**Step A follow-up (run after the above was written).** Tested the corrected impact prior
(h³→h², motivated by the L-sweep's L2 peak) head-to-head against the original, controlled
A/B. **Result: not significant** (ΔAUC = −0.044, CI [−0.106, +0.013]) — correcting the depth
assignment alone does not fix impact. This pointed at label volume, not depth assignment, as
the binding constraint — directly motivating Round 2's dataset regeneration.

## Issues identified (carried into Round 2's proposed changes)

1. Label volume far below the "hundreds to low thousands" PRD target (delay 34, shortage 84,
   impact 11 total positives).
2. Train/test snapshots sampled different regimes — all 4 disruption events concentrated in
   H2 2024, so a chronological split measured regime transfer, not generalization.
3. Shortage labels lost the warehouse dimension: `entity_id = product_id` for a label that's
   actually per (product, warehouse) — 71 of 480 distinct keys carried contradictory
   true/false labels.
4. Supplier count (50) made Transformer 2's future k=64 candidate bounding a no-op (k ≥ N_s
   always).
5. The documented co-parent 2-hop path cannot exist under the schema (confirmed above).
6. The impact head's depth-prior mismatch was confirmed but correcting it alone didn't fix
   the result (Step A) — the real constraint is label volume.
7. The planted Transformer-2 hidden-dependency scenario was confounded with `component_type`
   (18 of 25 polymer components concentrated on the same 4 suppliers the scenario needed
   hidden).
8. CI method was row-level i.i.d. bootstrap, not the schema's specified time-blocked
   bootstrap (documented compromise — 2-snapshot test split makes block resampling
   degenerate).
9. Multi-window temporal features were 52–90% NULL at this supplier count.
10. All Step 4/5 significance findings came from a single training seed — no variance
    decomposition existed yet.

**Conclusion driving Round 2:** nothing about the findings said the depth gate (Step 6) would
fail if built — they said the pipeline couldn't yet tell whether it succeeded or failed. Five
follow-up steps were proposed: **B** (multi-seed variance check), **C** (regenerate the
dataset with four targeted fixes), **D** (rebuild and rerun Steps 0–5), **E** (write the
comparison report) — plus Step A above, already done.

---

# Round 2 — v2 (Steps A–D, regenerated dataset)

**Dataset:** 180 suppliers, 6 monthly snapshots (Jul–Dec 2024), regenerated with 4 targeted
fixes. **Scope:** finish Steps A/B (on the v1 dataset) and Steps C/D (regenerate and rerun).

## Step B — Multi-seed variance check (v1 dataset, before regeneration)

Retrained the 5 Step-4 configurations across 5 seeds — 25 runs. **All three of v1's flagged
"significant" L-sweep findings flip sign across seeds:**

| Finding | v1 single-seed | Per-seed ΔAUC across 5 seeds |
|---|---|---|
| impact L2 vs L1 | +0.292 (significant) | [+0.292, +0.008, −0.070, −0.089, −0.294] |
| impact L3 vs L2 | −0.292 (significant) | [−0.292, −0.247, +0.479, −0.083, +0.487] |
| shortage L4 vs L3 | +0.039 (significant) | [+0.039, −0.077, −0.062, −0.172, −0.032] |

impact's AUC point estimate swung from 0.19 to 0.93 across seeds at fixed L (std 0.11–0.26) —
essentially unevaluable with a single training run. delay/shortage were far more seed-stable
(std 0.006–0.06). This is the clearest demonstration in the whole project that a single-seed
bootstrap CI does not capture model-fit variance, and it's why Step C (more label volume)
mattered more than Step B (more seeds) for actually resolving anything.

## Step C — Dataset regeneration (four fixes)

1. **Scale.** `SUP_N` 50→180 (proportional scale to components/products/customers/orders;
   factories/warehouses left fixed). Delay/shortage positives: 34→140, 84→396. Impact: 11→44
   (4×, short of "hundreds" — would need supplier counts well beyond 150–200).
2. **Disruption-timeline redistribution.** v1 concentrated all 4 events Aug–Dec; redistributed
   so every month Jul–Dec has at least one active event on both sides of the split.
3. **Shortage label warehouse dimension.** Added `training_labels.warehouse_id`, populated
   for shortage only — the 71-conflict problem (worse in absolute terms at the larger scale,
   303–340 conflicts before the fix) resolved to zero once grouped by
   `(snapshot_id, entity_id, warehouse_id)`. Caveat carried forward: still a data-level fix
   only, no Product×Warehouse graph node exists, so the shortage head still reads the
   Product node's own embedding.
4. **Hidden-dependency/`component_type` decoupling.** Removed the accidental concentration of
   18/25 polymer components onto the 4 hidden-dependency suppliers; the 4 members now span 4
   distinct component types.

Generator's own validation suite: `ALL CHECKS PASSED`, including two new checks.

## Step D — Rebuild and rerun Steps 0–5

Feature dimensions recomputed and found identical to v1 (scale-up grew population sizes, not
category cardinalities). 20 meta-relations confirmed; co-parent finding reproduces unchanged.
37/37 tests pass.

## Claim 1 — v1 vs v2

| Task | v1 fixed-d | v2 fixed-d | v1 matched-d | v2 matched-d |
|---|---|---|---|---|
| delay | not distinguishable | not distinguishable | not distinguishable | not distinguishable |
| shortage | **HGT wins** (both) | **HGT wins** (both) | HGT wins SAGE, ties GAT | **HGT wins GAT, LOSES to SAGE** |
| impact | HGT loses to GAT | HGT beats SAGE, ties GAT | ties both (gap closed) | HGT beats SAGE, ties GAT |

**The important reversal:** v1's cleanest, most-cited result — "HGT wins shortage on both
axes" — did not fully survive. At v2's matched-parameter arm, GraphSAGE (d=66) significantly
*beats* HGT (d=64) on shortage (ΔAUC = −0.037, CI [−0.062, −0.011]). HGT's shortage advantage
over GraphSAGE specifically turned out to be capacity-sensitive — the matched-parameter
control working against HGT this time, not for it. Impact's story flipped the other way: from
"HGT has a problem vs GAT" (v1, a capacity confound) to "HGT and GAT are genuinely tied,
GraphSAGE lags" (v2, clean and consistent).

## Claim 2 — v1 vs v2

| Task | v1 (prior vs L1–L4) | v2 (prior vs L1–L4) |
|---|---|---|
| delay | all not significant | all not significant |
| shortage | worse vs L1 (sig); ns elsewhere | worse vs L1 (sig); **better vs L2, L3, L4 (all sig)** |
| impact | flip-flopping (shown noise by Step B) | all not significant (clean, stable) |

v2's shortage result read as "the clearest win from the dataset fix — the structural prior
earns its place." (Round 3, below, found this did not survive proper 5-seed scrutiny.)

## Was the dataset fix sufficient?

Partially. Delay/shortage label volume (140/396 total positives) reached the PRD's "low
hundreds" target and visibly changed what the statistics could show. Impact (44 total, 21 in
test) improved 4× and became individually usable (AUC 0.85, CI width ~0.17) but its supplier-
count-bounded label volume would need supplier counts well beyond 150–200 to reach the same
volume as delay/shortage — directly motivating Round 3's further scale-up to 800 suppliers.

---

# Round 3 — v3 (800 suppliers, 15 snapshots, multi-seed from the start)

**Dataset:** 800 suppliers, 15 monthly snapshots (Jul 2024 – Sep 2025). **Scope:** rerun
Steps 0–5 with multi-seed built into *both* Step 4 and Step 5 from the start — the prior
round only did this for Step 4, which is exactly what caught v1's noise.

## Bottom line

The multi-seed-from-the-start design paid for itself immediately: it caught that **v2's own
cleanest, most-cited Claim 2 result — "the structural depth prior beats shared depth on
shortage" — does not survive proper 5-seed scrutiny.** Claim 1, by contrast, came out *more*
decisive than before, with several comparisons now sign-consistent across all 5 seeds,
including one (HGT beats GAT on delay) v2's single seed had called "not distinguishable."

## Claim 1 — v2 vs v3, now seed-backed

v3 uses 5-seed sign-consistency (a comparison counts as real only if all 5 seeds' point
estimates agree in sign):

| Task | v2 fixed-d | v3 fixed-d (5-seed) | v2 matched-d | v3 matched-d (5-seed) |
|---|---|---|---|---|
| delay | not distinguishable | HGT≈SAGE (flips); **HGT beats GAT (consistent, +0.061)** | not distinguishable | HGT≈SAGE (flips); **HGT beats GAT (consistent, +0.065)** |
| shortage | HGT wins both (sig) | **HGT loses to SAGE (consistent, −0.012)**; HGT beats GAT (consistent, +0.119) | HGT loses SAGE, beats GAT | HGT loses to SAGE (consistent, −0.014); HGT beats GAT (consistent, +0.121) |
| impact | HGT beats SAGE, ties GAT | HGT beats SAGE (consistent, +0.068); HGT≈GAT (flips) | HGT beats SAGE, ties GAT | HGT beats SAGE (consistent, +0.080); HGT≈GAT (flips) |

**Stated plainly:** delay — HGT ≈ GraphSAGE (genuinely tied), HGT reliably beats GAT (new
this round). shortage — HGT reliably beats GAT (large, ~+0.12), **HGT reliably loses to
GraphSAGE** (small but consistent, ~−0.01 to −0.014, on both axes — the finding that drove
Round 4's RGCN work). impact — HGT reliably beats GraphSAGE, HGT ≈ GAT (genuinely tied, not a
confound this time).

## Claim 2 — v2's decisive result does not replicate

| Task | v2 (single-seed, prior vs L1–L4) | v3 (5-seed sign-consistency) |
|---|---|---|
| delay | all not significant | all **FLIP (noise)** |
| shortage | worse vs L1 (sig); better vs L2, L3, L4 (all sig) | worse vs L1: flips; vs L2/L3/L4: all flip — only L3-vs-L2 itself is consistent (small, −0.011) |
| impact | all not significant | all **FLIP (noise)** |

**The single most important finding of the v3 round.** None of the four prior-vs-shared-depth
comparisons hold up under 5-seed sign-consistency, for any task. Claim 2 remains unresolved,
now on the strongest evidence yet that the structural prior does not earn its place.

## Dataset sufficiency, task by task

- **delay:** test n=4,107, 339 positives, AUC std 0.007–0.015 — stable, trustworthy.
- **shortage:** test n=19,164, 957 positives, std 0.004–0.011 — best-powered task.
- **impact:** test n=4,800, 114 positives (short of the 200-positive CI-table floor), but
  seed-to-seed AUC std (0.004–0.010) is the *tightest* of the three tasks — positive count and
  model-fit stability are not the same thing.

## Split boundaries

Per-snapshot label counts showed an extreme spike in Oct–Nov 2024 (delay 248/194 vs. a 26–74
baseline elsewhere). Chosen split: **train Jul–Dec 2024** (6 snapshots, includes the spike),
**val Jan–Mar 2025** (3), **test Apr–Sep 2025** (6) — checked explicitly against the full
per-snapshot table, not assumed.

## Cost

Largest v3 snapshot (66,973 nodes, 388,242 edges) builds in 0.68s — never the bottleneck. One
100-epoch HGT run took ~421.8s (~7 min), 5.3× v2's ~80s. Full 55-run matrix (25 depth-sweep +
30 architecture-ablation) completed in 4.61 hours.

## Operational note — governance-table mistake, disclosed

`db/load_data.py --drop`'s `DROP SCHEMA public CASCADE` destroyed `model_registry`/
`model_evaluation_runs` (v2's, Step A's, and Step B's rows) as a side effect, despite this
round's 55 new runs correctly using distinct `-v3-seed{n}` names that never collided with
v2's. Caught, disclosed, partially repaired: v2's 11 core rows reconstructed from this
session's own run output and re-inserted `status='archived'` with an explicit
`RECONSTRUCTED` marker; Step A's/B's rows were not reconstructed (accepted lower priority).
**Lesson adopted for every round since:** back up governance tables before any `--drop`,
unconditionally — this is why every round from v3-followup onward opens with a `pg_dump`
before any write, regardless of whether a `--drop` is planned that round.

## Governance record (v3)

66 `model_registry` rows (55 new `status='active'` + 11 reconstructed `status='archived'`),
858 `model_evaluation_runs` rows. 37/37 tests pass.

---

# Round 4 — v3 Follow-up (Tasks 1–3: Reach, Sparse-Relation, Real Co-Parent Signal)

**Dataset:** same v3 dataset for Tasks 1–2; a **separate, optional, fourth dataset
generation** for Task 3 only. **Scope:** three follow-ups to v3's open questions, run in
priority order.

## Bottom line

Every one of v3's open questions that these three tasks could test came back **negative or
non-committal, never "HGT wins."** Task 1 shows two of the three depth priors' own
theoretical derivations don't describe either the measured graph or the model actually
built — independent evidence for Claim 2's negative finding, from a completely different
angle (reach structure, not depth-sweep AUCs). Task 2 shows HGT's consistent shortage loss to
GraphSAGE can be made to disappear (become a statistical tie) by merging its thinnest
relations' parameters — but not reversed into a win. Task 3, a genuinely different experiment
giving the graph a real causal co-parent signal that provably didn't exist before, produces
the same qualitative outcome as Task 2 by a completely different mechanism. **No result in
this round shows HGT winning shortage under the 5-seed sign-consistency standard.**

## Task 1 — Were h²/h³ ever the empirically-correct targets?

**No, not for two of the three tasks, on their own theoretical terms.**
`docs/project_HADES.md` §4.2 derives delay=h² ("children + co-parents"), shortage=h³
(Product + warehouse context), impact=h³ (Order/Customer target).

| Task | Theoretical target/mechanism | What Task 1 measured | Verdict |
|---|---|---|---|
| delay | h², via co-parents | Shipment (delay's actual target) reachable at **hop 1** directly via `SHIPS_FROM` — the co-parent mechanism isn't needed at all, and it's independently confirmed broken (0/800) | h²'s own justification doesn't match either the measured path or the measured hop-count |
| shortage | h³, Product + warehouse | Product coverage jumps sparse (hop 2, mean 8.4/1,280) → majority (hop 3, mean 795/1,280) | **The one prior with solid measured support** |
| impact | h³, target = Order/Customer | Order first appears at hop 3 (mean 378/25,193, 75%) — matches precisely | Measures out correctly, but the task **actually implemented** targets Supplier itself, not Order/Customer — validates a prior for a task that was never built |

Co-parent finding reconfirmed: still 0/800 suppliers reach a different supplier in 2 hops.

## Task 2 — Does merging HGT's thinnest relations close the shortage gap?

The four thinnest meta-relations (`MANUFACTURED_AT` 1,933 edges, `SUPPLIES` 2,400,
`STOCKED_AT` 3,194, `USED_IN` 7,092 — a real gap before the next-thinnest at 15,608) were
merged onto one shared `SAGEConv` parameter set (`ml/models/sparse_hgt_encoder.py`), while
the other 12 relations kept full per-relation HGT parameterization.

| Axis | Original hgt-vs-graphsage gap (v3) | hgt_sparse-vs-graphsage (this task) |
|---|---|---|
| fixed-d | −0.0123, **CONSISTENT** loss | mean +0.0031 → **FLIPS (noise)** |
| matched-d | −0.0137, **CONSISTENT** loss | mean +0.0016 → **FLIPS (noise)** |

**The consistent loss disappears — becomes a statistical tie, not a win.** Evidence for the
thin-relation-overfitting hypothesis as *part* of the explanation, not a demonstration HGT is
better once fixed.

## Task 3 (separate, optional experiment) — Does a real co-parent signal let HGT win?

A new `component_suppliers` junction table gave ~17.5% of components (420/2,400) a second
qualified supplier with a real causal coupling (`COPARENT_COUPLING = 0.35`), discoverable
only by traversing the co-parent edge. Required a new dataset generation, DB reload, and
distinct `-coparent-v4-seed{n}` runs. Co-parent 2-hop reach on this new graph: 449/800 (56%),
vs. 0/800 on base v3.

| Task | hgt-coparent vs graphsage-coparent, mean ΔAUC | Consistency |
|---|---|---|
| shortage | −0.0033 | **FLIPS (noise)** |
| delay | −0.0038 | **FLIPS (noise)** |

**Same qualitative result as Task 2 (tie, not win), by an entirely different mechanism.**
Two independent interventions — merging thin relations and adding a real relational signal
HGT alone should be positioned to exploit — both land in the same place. Strongest evidence
yet that HGT's shortage weakness is a genuine architecture-vs-task mismatch, not a
data-availability or parameterization problem.

**Scope note:** Task 3's dataset is a separate, fourth generation, not "v3 plus one table" —
the co-parent coupling feeds into shipment-delay branching, so the whole generated world
differs from base v3 from that point forward. v3's own findings and governance rows are
unaffected.

## Governance record

Task 2: 5 new `-sparserelation-v3-seed{n}` rows, 71 total registry rows. Task 3: backed up
before the required `--drop` reload (restored after `--drop`'s cascade wiped the backup
copies too, confirmed intact before proceeding); 10 new `-coparent-v4-seed{n}` rows, 81 total
registry rows. 38/38 tests pass.

---

# Round 5 — v4: RGCN Fourth-Architecture Ablation

**Dataset:** confirmed live before any code was written — v3 (800 suppliers, 15 snapshots),
`component_suppliers` present but dormant (max 1 supplier/component — not the separate
Task-3/co-parent dataset). **Scope:** add RGCN (Schlichtkrull et al. 2018 basis
decomposition) as a fourth encoder architecture and rerun Steps 3–5 as a 4-way matrix — 5
seeds × 4 architectures × 2 param-arms = 40 training runs, multi-seed from the start.

## Bottom line

**No architecture dominates across all three tasks — the four-way matrix sharpens Claim 1
rather than resolving it.** HGT still wins impact outright, replicating every prior round.
**RGCN is a genuine new result: it beats both HGT and GraphSAGE on shortage, consistently
across all 5 seeds, on both arms** — the cleanest, most reproducible finding in this round,
and independent support (via a different, more principled mechanism than Task 2's ad hoc
merge) for the thin-relation-overfitting hypothesis. Delay remains statistically undecided
among HGT/GraphSAGE/RGCN. GAT is the clear architecture-level loser on delay and shortage.

## RGCN design

Hand-written over `edge_index_dict` (`ml/models/rgcn_encoder.py`) rather than using
`torch_geometric.nn.RGCNConv` directly, since its public API expects a homogeneous graph, not
this codebase's per-relation `HeteroData` convention — same reason `hgt_sparse` is also
bespoke. Every relation's transform is a linear combination of a small shared pool of
`num_bases` basis matrices (`W_r = Σ_b a_r[b]·V_b`) — one shared basis pool + one coefficient
vector per relation for the *whole* encoder, reused identically at every layer, so relational
parameter cost doesn't scale with `num_layers`. A separate, NOT basis-shared, per-node-type
self-loop matrix per layer carries identity across layers (HGT's residual/GraphSAGE's
self+neighbor role).

## Parameter counts

Matched-parameter search target: HGT's real encoder-only anchor, re-derived live: 701,088.
Grid-searching `num_bases ∈ {4,8,12,16}` at `hidden=64` landed nowhere close (max 203,392) —
self-loop/input-projection dominates RGCN's count at that width — so the search widened
`hidden` alongside `num_bases`: **hidden=138, num_bases=4 → 699,602, 0.21% off HGT's anchor**,
the closest matched-arm fit of any of the four architectures.

| Arm | Encoder-only | Full model (incl. 3 task heads) |
|---|---:|---:|
| HGT fixed/matched-d=64 | 701,088 | 713,763 |
| GraphSAGE fixed-d=64 / matched-d=66 | 664,896 / 706,794 | 677,571 / 720,261 |
| GAT fixed-d=64 / matched-d=92 | 347,456 / 705,548 | 360,131 / 731,495 |
| RGCN fixed-d=64 (num_bases=8) / matched-d=138 (num_bases=4) | 170,464 / 699,602 | 183,139 / 757,565 |

## AUC — mean across 5 seeds, per (task, arm)

| Arm | delay | shortage | impact |
|---|---|---|---|
| hgt-fixed/matched-d | 0.8015 | 0.7824 | **0.9335** |
| graphsage-fixed-d | **0.8053** | 0.7857 | 0.8912 |
| graphsage-matched-d | 0.8032 | 0.7898 | 0.8889 |
| gat-fixed-d | 0.7467 | 0.6728 | 0.9138 |
| gat-matched-d | 0.7400 | 0.6560 | 0.9182 |
| rgcn-fixed-d | 0.8018 | **0.7942** | 0.8676 |
| rgcn-matched-d | **0.8044** | **0.7976** | 0.8509 |

## Sign-consistency verdicts (both axes, HGT/GraphSAGE anchors)

| Task | Fixed-d winner | Matched-d winner |
|---|---|---|
| delay | 3-way tie: HGT/GraphSAGE/RGCN (GAT loses consistently to both) | Same 3-way tie |
| shortage | **RGCN**, consistently beating both HGT and GraphSAGE | **RGCN**, same consistent win |
| impact | **HGT**, decisively (beats all three, up to +0.083 vs RGCN) | **HGT**, decisively |

RGCN is impact's *worst* performer on both arms — the same basis-shared relation weights that
win it shortage appear to cost it here.

## Governance record

40 new `-4arch-seed{n}` rows (10 per architecture, all active) — 121 total registry rows.
600 new evaluation rows — 1,683 total. 40/40 tests pass. Wall-clock: 2.49h for all 40 runs.

---

# Round 6 — v5: RGCN+Attn ("Option 1") Pilot

**Dataset:** re-confirmed live, unchanged v3. **Scope:** implement, wire, and
unit/shape-verify a fifth architecture — RGCN + shared (relation-agnostic) attention — then
run a narrow pilot: fixed-d=64 only, 5 seeds × 3 architectures (HGT, RGCN, RGCN+attn) = 15
training runs. No matched-d arm, no doc updates, no GraphSAGE/GAT re-run — explicitly scoped
narrow.

## Bottom line

**RGCN+attn has the highest mean AUC on all three tasks simultaneously — something no single
architecture achieved across any prior round.** It keeps RGCN's consistent shortage advantage
over HGT, closes RGCN's large, consistent impact deficit against HGT entirely (from a
consistent −0.0636 loss to a statistically-tied +0.0058), and opens a new, genuinely
consistent delay win over both HGT and plain RGCN individually. All at **4.1× fewer
parameters than HGT**, and only 520 more than plain RGCN.

## Design — what changed from plain RGCN

The message *transform* is identical to `RGCNEncoder` (`lin_in`, shared `rel_basis`/
`rel_coeff` basis pool, the `W_r` computation, per-layer self-loops). What changes is purely
how transformed messages combine: plain RGCN aggregates each relation separately (mean) then
sums — every relation gets an equal vote. RGCN+attn instead scores every incoming edge to a
destination node — across every relation feeding that node, mixed together — with one shared
attention scorer (`att_msg`, `att_dst`: two `Linear(hidden,1)` layers per layer, the ONLY new
learnable parameters), then takes a single joint softmax over that whole pool per destination
node (`torch_geometric.utils.softmax`), matching `HGTConv`'s per-node softmax scope at a
fraction of its parameter cost. The scorer is deliberately shared across every relation and
node type — the cheapest of the candidate hybrid designs, lowest overfitting risk.

## Parameter counts (hidden=64, recomputed live)

| Architecture | Params | vs HGT |
|---|---:|---:|
| HGT | 701,088 | — |
| RGCN (num_bases=8) | 170,464 | 4.11× fewer |
| **RGCN+attn** (num_bases=8) | 170,984 | 4.10× fewer |

RGCN+attn adds exactly **520 params** over plain RGCN — `2 × (hidden+1) × num_layers`.

## AUC — mean across 5 seeds, fixed-d=64

| Architecture | delay | shortage | impact |
|---|---|---|---|
| hgt | 0.8015 | 0.7824 | 0.9335 |
| rgcn | 0.8018 | 0.7944 | 0.8699 |
| **rgcn_attn** | **0.8127** | **0.7985** | **0.9393** |

## Sign-consistency

| Comparison | delay | shortage | impact |
|---|---|---|---|
| rgcn_attn vs hgt | +0.0112 **CONSISTENT** (attn wins) | +0.0162 **CONSISTENT** (attn wins) | +0.0058 FLIPS (tied) |
| rgcn_attn vs rgcn | +0.0109 **CONSISTENT** (attn wins) | +0.0041 FLIPS (tied) | +0.0694 **CONSISTENT** (attn wins, large) |
| rgcn vs hgt (re-measured) | +0.0003 FLIPS (tied) | +0.0120 **CONSISTENT** (rgcn wins) | −0.0636 **CONSISTENT** (hgt wins, large) |

The re-measured `rgcn vs hgt` row replicates Round 5's findings exactly — a clean
internal-consistency check.

## Caveat — this is not yet a resolved win

This is a fixed-d=64 pilot, not a matched-parameter comparison. RGCN+attn's real parameter
count (170,984) is far below HGT's anchor (701,088), so the result could partly reflect
regularization from a much smaller model on a label-scarce dataset rather than the attention
mechanism itself. A matched-d follow-up (widening `hidden`/`num_bases` toward HGT's budget,
same method as Round 5's search) is the natural next step, flagged here as the clearest open
item this project's architecture line of experiments has produced.

## Governance record

15 new `-rgcnattn-pilot-seed{n}` rows (5 per architecture, all active) — 136 total registry
rows. 225 new evaluation rows — 1,908 total. 43/43 tests pass. Wall-clock: 0.95h for all 15
runs.

---

# Consolidated Governance and Test-Suite Record (current, live state)

As of this report:

| Metric | Value |
|---|---|
| `model_registry` rows | 136 (125 active, 11 archived — the reconstructed v2 rows from Round 3) |
| `model_evaluation_runs` rows | 1,908 |
| Architectures represented | hgt (62 rows incl. `heterogeneous_graph_transformer` label), graphsage (27), gat (22), rgcn (15), rgcn_attn (5), hgt_sparse (5) |
| `ml/tests/` suite | 43/43 passing |

Every round's runs are queryable by their distinct `model_version` suffix — `-lsweep-`,
`-fixed-d`/`-matched-d`, `-v3-seed{n}`, `-sparserelation-v3-seed{n}`, `-coparent-v4-seed{n}`,
`-4arch-seed{n}`, `-rgcnattn-pilot-seed{n}` — no round has ever upserted over a prior round's
rows (the one exception, Round 3's `--drop`-triggered loss of Round 2's/Step A's/Step B's
rows, was disclosed and partially repaired in Round 3 itself, and is the reason every round
since backs up governance tables unconditionally before any write).

## Code inventory (current state)

```
db/
├── governance_schema.sql        model_registry, model_evaluation_runs, explanation_subgraphs
ml/
├── data/
│   ├── db.py                    connection helper
│   ├── features.py              as-of feature extraction, exclusion-list enforcement
│   └── snapshots.py             multi-snapshot feature/label assembly
├── graph/
│   ├── builder.py               HeteroData assembly, ToUndirected, graph_snapshots persistence
│   └── reach.py                 empirical k-hop reach measurement
├── models/
│   ├── depth.py                 structural depth prior + shared-depth override (L-sweep)
│   ├── heads.py                 prediction heads + focal loss
│   ├── encoder.py               HGT / GraphSAGE / GAT / RGCN / RGCN+attn factory (build_encoder)
│   ├── sparse_hgt_encoder.py    Task 2's sparse-relation-merged HGT variant
│   ├── rgcn_encoder.py          RGCN (basis decomposition)
│   ├── rgcn_attn_encoder.py     RGCN + shared attention ("Option 1")
│   └── model.py                 HADESModel (encoder + depth readout + heads composed)
├── train.py                     time split, training loop, model_registry logging
├── evaluate.py                  metrics, bootstrap CIs, over-smoothing, significance tests
├── run_step3.py / run_step4.py / run_step5.py / run_step_a.py / run_step_b.py
├── run_step_v3.py               v3 multi-seed Steps 3-5 matrix
├── run_task2_sparse_relation.py / run_task3_coparent.py
├── run_step_v4_4arch.py         4-architecture (incl. RGCN) matrix
├── run_rgcn_attn_pilot.py       RGCN+attn pilot
└── tests/
    ├── test_leakage.py          Step 1 gate
    ├── test_graph.py            Step 2 gate
    ├── test_dyadic.py
    └── test_model.py            encoder/heads/depth-prior unit tests (43 tests total across suite)
```

---

# Open Items — Current State

Carried forward from earlier rounds, with resolution status as of the latest round:

| Item | Origin | Status |
|---|---|---|
| `db/README.md`'s claimed solo-feature AUC (~0.59–0.63) unreproduced | Round 1 | **Still open**, never revisited |
| Label volume below PRD target | Round 1 | **Resolved** by Round 3's 800-supplier scale-up (delay/shortage now well-powered; impact's positive count is still short of the 200 floor but seed-to-seed stability is now the tightest of the three tasks) |
| Train/test regime mismatch | Round 1 | **Resolved** by Round 2's disruption-timeline redistribution |
| Shortage label warehouse collapse | Round 1 | **Resolved** by Round 2's `warehouse_id` addition (data-level only — no Product×Warehouse graph node exists; the shortage head still reads the Product node's own embedding) |
| Co-parent 2-hop path doesn't exist under the schema | Round 1 | **Confirmed unchanged** through every round on the base dataset; Round 4's Task 3 built a separate dataset with a real coupling and found it still didn't produce an HGT win |
| Transformer 2 scenario confounded with `component_type` | Round 1 | **Resolved** by Round 2's decoupling fix (T2 itself remains unbuilt — Steps 7-8 not yet reached) |
| CI method is row-level, not time-blocked bootstrap | Round 1 | **Still a documented compromise** — test splits remain too short for block resampling to be non-degenerate |
| Single-seed significance findings | Round 1 | **Resolved as a practice** from Round 3 onward — multi-seed built into every round's methodology since |
| Claim 2 (structural depth prior) | Rounds 1-4 | **Still unresolved**, on progressively stronger negative evidence — v3's 5-seed result and Round 4's independent reach-based reinforcement both point toward "no," not settled either way |
| Step 6 (learned depth gate) | Roadmap | **On hold**, pending Claim 2 resolution — there is no validated prior to anchor a small learned correction to |
| RGCN+attn matched-parameter arm | Round 6 | **Not yet run** — the clearest immediate next step; needed to separate "attention helped" from "smaller model regularized better" |
| Whether RGCN+attn's delay/impact gains hold beyond this pilot's 5 seeds at full arm scope | Round 6 | **Open** — this was a fixed-d-only pilot, not a full ablation arm |

---

*This document supersedes `steps_0-5_findings.md`, `step5_result_v2.md`, `step5_result_v3.md`,
`step5_result_v3_followup.md`, `step5_result_v4_4arch.md`, and `rgcn_attn_pilot.md`, which
have been removed from `reports/` as part of this merge. Raw per-seed run logs referenced
above remain in `reports/logs/`; governance-table backups remain in `reports/backups/`.*
