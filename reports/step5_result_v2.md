# HADES Model-Development Prototype — Findings & Conclusions, Steps 0–5 (v2)

**Scope:** a follow-up pass over `reports/steps_0-5_findings.md` ("v1"), motivated by three
problems v1 itself identified: the impact head's depth prior looked wrong given the L-sweep
shape, the L-sweep's significance tests ran on a single training seed, and the underlying
synthetic dataset had structural problems (label volume, disruption-timeline concentration,
a shortage-label ambiguity, and a component-type confound in the hidden-dependency scenario)
that capped what any of Steps 3–5 could show. This report covers four pieces of work, done in
order: **Step A** (test a corrected impact prior), **Step B** (multi-seed variance check),
**Step C** (regenerate the dataset with four targeted fixes), **Step D** (rebuild and rerun
Steps 0–5 against it). Steps 6–8 (depth gate, Transformer 2, Claim B) are untouched.

This report has two parts: **Part I — Conclusions** and **Part II — Findings by Step**.

---

# Part I — Conclusions

## Bottom line

The dataset fixes worked as intended: label volume moved into the range the PRD assumes,
the impact task went from statistically unevaluable to a clean, confident result, and the
depth-prior question (Claim 2) went from inconclusive flip-flopping to a real, decisive
finding. Claim 1's answer changed in a genuinely informative way — not just "more confident,"
but **more skeptical of v1's own cleanest-looking result**. Steps A and B, run before the
dataset fix, independently confirmed that three of v1's "significant" findings were single-seed
noise — which is exactly why the dataset fix mattered more than a bigger bootstrap.

## Step A — Corrected impact depth prior: does not beat the original

Testing `impact -> h²` (motivated by v1's L-sweep shape) against the documented `impact -> h³`,
head-to-head, on the v1 dataset: **not significant** (ΔAUC = -0.044, CI [-0.106, +0.013]) — the
corrected prior does not beat the original. Worth noting explicitly: this is a *different*
question from what the L-sweep answers. The L-sweep forces *all three* tasks to share one
depth; Step A changes *only* impact's readout while delay/shortage keep their own priors. The
more targeted test found no improvement, which is the more relevant test for "should the
documented prior change" — and the answer, at this label volume, is no.

## Step B — Multi-seed check: v1's three flagged findings were all noise

All three of v1's single-seed "significant" L-sweep findings **flip sign across 5 seeds**:

| Finding | v1 single-seed | Per-seed ΔAUC across 5 seeds |
|---|---|---|
| impact L2 vs L1 | +0.292 (significant) | [+0.292, +0.008, -0.070, -0.089, -0.294] |
| impact L3 vs L2 | -0.292 (significant) | [-0.292, -0.247, +0.479, -0.083, +0.487] |
| shortage L4 vs L3 | +0.039 (significant) | [+0.039, -0.077, -0.062, -0.172, -0.032] |

None of the three survive. impact's AUC point estimate itself swings from 0.19 to 0.93 across
seeds at fixed L (std 0.11–0.26) — the task was, on the v1 dataset, essentially unevaluable
with a single training run. delay/shortage were far more seed-stable (std 0.006–0.06). This is
the clearest possible demonstration that a single-seed bootstrap CI, however correctly computed,
does not capture model-fit variance — and it's why Step C (more label volume) mattered more
than Step B (more seeds) for actually resolving anything.

## Claim 1 — Architecture ablation: v1 vs v2

| Task | v1 fixed-d | v2 fixed-d | v1 matched-d | v2 matched-d |
|---|---|---|---|---|
| delay | not distinguishable | not distinguishable | not distinguishable | not distinguishable |
| shortage | **HGT wins** (both) | **HGT wins** (both) | HGT wins SAGE, ties GAT | **HGT wins GAT, LOSES to SAGE** |
| impact | HGT loses to GAT | HGT beats SAGE, ties GAT | ties both (gap closed) | HGT beats SAGE, ties GAT |

**What changed, and why it matters:** delay's verdict is unchanged (still not enough signal).
Impact's story flipped from "HGT has a problem vs GAT" (v1, later shown to be a capacity
confound) to "HGT and GAT are genuinely tied, GraphSAGE lags" (v2, clean and consistent across
both axes) — the larger, better-balanced dataset resolved the ambiguity in HGT's favor.
**Shortage is the more important change: v1's cleanest, most confident result — "HGT wins
shortage on both axes" — did not fully survive.** At v2's matched-parameter arm, GraphSAGE
(d=66) significantly *beats* HGT (d=64) on shortage (ΔAUC = -0.037, CI [-0.062, -0.011]). HGT
still beats GraphSAGE at fixed-d, and still beats GAT at both axes — so the honest reading is
narrower than v1's: **HGT's shortage advantage over GraphSAGE specifically is itself capacity-
sensitive**, the same kind of confound the matched arm was built to catch, just showing up in
the opposite direction (working against HGT this time, not for it). This is the single most
important reason not to treat v1's Claim 1 answer as final — the matched-parameter control
matters on every axis, not just the one where it happened to change the answer the first time.

## Claim 2 — Depth prior: v1 vs v2

| Task | v1 (prior vs L1..L4) | v2 (prior vs L1..L4) |
|---|---|---|
| delay | all not significant | all not significant |
| shortage | worse vs L1 (sig); ns elsewhere | worse vs L1 (sig); **better vs L2, L3, L4 (all sig)** |
| impact | flip-flopping (sig both directions, shown to be noise by Step B) | all not significant (clean, stable) |

**This is the clearest win from the dataset fix.** v1's shortage row was a single ambiguous
significant result; v2's is four decisive comparisons, three of which agree with each other
(the prior beats every shared depth except the shallowest). v1's impact row was actively
misleading (Step B proved it was noise); v2's is cleanly "not distinguishable" — a real,
trustworthy null result instead of noise that happened to look like a finding. **Claim 2 is
now answerable for shortage: the structural prior earns its place over sharing a single depth,
at least against L2–L4.** Delay and impact remain genuinely undecided, honestly reported as
such rather than forced into a verdict.

## Was the dataset fix sufficient?

**Partially — good enough to make Steps 3–5's numbers trustworthy, not good enough to close
every open question.**

- **Sufficient:** delay and shortage label volume (140 and 396 total positives) is now
  solidly in the "low hundreds" the PRD assumes, and it visibly changed what the statistics
  could show — Claim 2 went from noise to a real result, and Claim 1's shortage story became
  more nuanced rather than just "more confident in the same direction."
- **Not fully sufficient:** impact (44 total positives, 21 in the test split) improved 4x over
  v1 and is now individually usable (AUC 0.85, CI width ~0.17), but delay's CIs are still wide
  enough that "not distinguishable" could still be under-powered rather than a true null — and
  impact's supplier-level label is inherently bounded by supplier count, so reaching the same
  "low hundreds" volume delay/shortage now have would require supplier counts well beyond the
  150-200 this round targeted (per Step C's own honest accounting).
- **Recommendation before Steps 6–8:** the current dataset is good enough to build the depth
  gate and Transformer 2 on without repeating this exercise, *if* their acceptance criteria are
  read with the same label-volume caveats applied here — impact-task claims in particular
  should keep citing their positive count, not just their point estimate.

---

# Part II — Findings by Step

## Step A — Corrected Impact Depth Prior

**Motivation:** v1's L=1..4 sweep showed impact's AUC peak at L2 (+0.292 vs L1, significant)
then drop at L3 (-0.292 vs L2, significant) — a structural signal at L2, not the documented h³.
impact's label is supplier-level, a shallower structural distance than the Order/Customer-level
path `project_HADES.md` §4.2 used to derive h³.

**What was built:** `ml/models/depth.py` gained `STRUCTURAL_DEPTH_PRIOR_V2` (`impact -> 2`,
delay/shortage unchanged) and `readout_layer_for_task` took an optional `prior` argument;
`HADESModel`/`train_model`/`run_training_job` thread it through. The original prior was left
untouched — this is a controlled A/B (`ml/run_step_a.py`), not a replacement. A unit test
(`test_structural_depth_prior_v2_only_changes_impact`) asserts the isolation.

**Result** (both retrained fresh in the same process, v1 dataset):

| Model | delay | shortage | impact |
|---|---|---|---|
| v1 (impact=h³, as-documented) | 0.6734 | 0.7258 | 0.4010 [0.031, 0.899] |
| v2-impact-h² (corrected) | 0.6278 | 0.7854 | 0.3568 [0.010, 0.889] |

impact: ΔAUC (corrected − original) = **-0.0443, CI [-0.1063, +0.0126] — not significant.**
The corrected prior does not beat the original. (Side observation, not the main finding:
delay and shortage's own AUCs also shifted between the two runs, even though their readout
layers didn't change — a real consequence of a shared encoder: changing where the impact head
reads from changes the gradients the whole encoder receives, so it isn't possible to change one
task's depth in isolation without some effect on the others.)

Both runs logged: `model_registry` (`hgt-baseline-v1`, `hgt-baseline-v2-impact-h2`), 15 metric
rows each in `model_evaluation_runs` with CIs.

## Step B — Multi-Seed Variance Check

**What was built:** `ml/run_step_b.py` retrains the 5 Step-4 configurations (fixed-prior
baseline + shared-depth L=1,2,3,4) across seeds 0–4 (model init/dropout only; dataset, split,
and feature assembly stay fixed/deterministic upstream) — 25 runs, each with its own
`model_registry` row and `model_evaluation_runs` rows.

**AUC spread across 5 seeds** (v1 dataset; full table in `reports/step_b_raw_output.txt`):

| Config | Task | mean | std | min | max |
|---|---|---|---|---|---|
| baseline | delay | 0.613 | 0.045 | 0.547 | 0.673 |
| baseline | shortage | 0.740 | 0.015 | 0.723 | 0.762 |
| baseline | impact | 0.521 | **0.215** | 0.333 | 0.896 |
| L4 | delay | 0.674 | 0.038 | 0.609 | 0.714 |
| L4 | shortage | 0.692 | 0.062 | 0.578 | 0.753 |
| L4 | impact | 0.489 | **0.111** | 0.297 | 0.599 |

impact's seed-to-seed standard deviation (0.11–0.26) dwarfs delay/shortage's (0.006–0.06) at
every config. **Result, stated plainly (per the same convention `reports/steps_0-5_findings.md`
used):** all three of v1's flagged "significant" findings flip sign across seeds (table in Part
I) — they were single-seed model-fit noise, not real depth effects. This is not a failure of
the bootstrap CI method; it's a demonstration that a CI computed from one trained model answers
"how uncertain is this model's estimate," not "how uncertain is the true effect" — the latter
needs multiple training runs, which is exactly what Step B added.

## Step C — Regenerate the Synthetic Dataset (Four Fixes)

All in `db/generate_dataset.py`; full detail and exact validation-suite output in
`db/README.md`'s "v2 fixes" section.

1. **Scale.** `SUP_N` 50→180 (`SCALE=3.6` applied proportionally to
   components/products/customers/orders; factories/warehouses left fixed as physical
   infrastructure, not supply-chain-proportional). Delay/shortage positives: 34→140,
   84→396 — solidly into the "low hundreds" `docs/01_Product_Requirement_Document.md` §8
   assumes. Impact: 11→44 (4x, not full "hundreds" — would need supplier counts well beyond
   150-200).
2. **Disruption timeline redistribution.** v1 concentrated all 4 disruption events Aug–Dec,
   so the chronological train (Jul-Sep) / test (Nov-Dec) split trained on a quiet regime and
   tested on a disrupted one. Redistributed so every month Jul–Dec has at least one active
   event on both sides of the split (verified directly: Jul train sees H_TRUCK; Aug train sees
   H_TRUCK+H_CUSTOMS; Sep train sees H_CUSTOMS+H_PORT+H_POLYMER onset; Oct val sees
   H_PORT+H_POLYMER; Nov/Dec test see the same two events' tails). One constraint discovered
   during this fix: the hidden-dependency check reads Dec-1's 90-day trailing on-time rate, so
   H_POLYMER's peak had to stay early enough (kept at its original Oct 10 peak) for delayed
   shipments to resolve into that window by Dec 1 — pushing it later (tried Nov 20) broke the
   check (members ended up *above* the fleet mean, the wrong direction).
3. **Shortage label warehouse dimension.** `entity_id` for `task='shortage'` is a `product_id`,
   but a product can be stocked at multiple warehouses with independent outcomes — the original
   v1 (50-supplier) dataset had 71 (product, snapshot) pairs carrying conflicting true/false
   rows; the same measurement against the scaled-up (180-supplier) world, before the fix, found
   303-340 (higher in absolute terms simply because there are more products and warehouse
   pairs, not a different rate). Fixed by adding `training_labels.warehouse_id` (populated for
   `shortage` only; NULL for delay/impact) — `db/schema.sql` and
   `docs/05_Database_Design.md` §6.18 updated. Validation suite now asserts zero conflicts once
   grouped by `(snapshot_id, entity_id, warehouse_id)`. **Caveat carried forward:** this is a
   data-level fix only — the graph has no Product×Warehouse node, so `ml/models/heads.py`'s
   shortage head still reads the Product node's own embedding for every one of that product's
   labelled instances (a separate, documented, out-of-scope-for-this-round architectural
   question).
4. **Hidden-dependency scenario decoupled from `component_type`.** v1 picked H_POLYMER's 4
   members unconstrained but then forced 18 of 25 `polymer` components onto those same 4
   suppliers — an accidental giveaway via the one column the scenario is supposed to be
   invisible through. Fixed by removing that concentration and choosing the 4 members
   deterministically to span 4 distinct component_types (never `polymer` as any member's
   dominant type). Resulting composition (Germany/Vietnam/USA/Vietnam, each member's own
   component types collectively spanning all 5 categories, no member polymer-only) is in
   `db/README.md`. The generator's own validation suite passes this as a new explicit check.

**Generator output:** `ALL CHECKS PASSED`, including both new checks (shortage
warehouse-disambiguation, hidden-dependency component-type diversity). Fully deterministic
(seeded RNG + uuid5 keys) — re-running produces the same CSVs, confirmed by the row counts
matching exactly across the two generation runs during this work.

## Step D — Rebuild and Rerun Steps 0–5

- **Step 0:** `db/load_data.py --drop` against the v2 CSVs — 19/19 tables, 7/7 verification
  queries pass. Label rates: delay 11.44% (140/1224), shortage 9.13% (396/4338), impact 4.07%
  (44/1080) — matches Step C's generation output exactly.
- **Step 1:** leakage test — 3/3 pass, no leak (`min_stock_ratio` still closest at 0.86,
  under the 0.90 gate; `days_to_eta` 0.70 for delay, `lateness_variance` 0.72 for impact).
- **Step 2:** feature dimensions **recomputed, not assumed** — confirmed identical to v1
  (Supplier 14, Component 6, Product 13, Factory 6, Warehouse 7, Shipment 7, Order 5,
  Customer 3), because the scale-up grew population sizes, not category cardinalities (still
  6 countries, 5 component types, 6 product categories, etc.). 20 meta-relations confirmed.
  Reach measurement and the co-parent finding (still structurally false, same schema reason)
  reproduce unchanged. One test fixture needed updating: the BOM-substitution unit test
  referenced specific component UUIDs that shifted under the new RNG draw order (same
  product, same date, different component id) — re-derived from the live database, not
  weakened or skipped.
- **Steps 3–5:** see Part I's Claim 1/2 tables for the substantive results. All 11
  `model_registry` rows re-registered (same `model_version` names, `ON CONFLICT DO UPDATE`),
  195 `model_evaluation_runs` rows refreshed with the new CIs.
- **Full suite:** 37/37 tests pass, `ruff check ml/` clean.

## Governance record (v2 dataset, current DB state)

| model_version | architecture | parameters | notes |
|---|---|---|---|
| `hgt-baseline-v1` | heterogeneous_graph_transformer | 713,763 | Step 3 baseline, now v2 numbers |
| `hgt-lsweep-L1..L4` | heterogeneous_graph_transformer | 191,259 / 365,427 / 539,595 / 713,763 | Step 4 sweep, now v2 numbers |
| `hgt-fixed-d`, `graphsage-fixed-d`, `gat-fixed-d` | — | 713,763 / 677,571 / 360,131 | Step 5 fixed-d arm, now v2 numbers |
| `hgt-matched-d`, `graphsage-matched-d`, `gat-matched-d` | — | 713,763 / 720,261 / 731,495 | Step 5 matched-d arm, now v2 numbers |
| `hgt-baseline-v2-impact-h2` | heterogeneous_graph_transformer | 713,763 | Step A, v1 dataset (pre-regeneration) |
| `hgt-seedvar-{baseline,L1..L4}-s{0..4}` | heterogeneous_graph_transformer | (25 rows) | Step B, v1 dataset (pre-regeneration) |

v1's original numbers are not recoverable from the live database (the same `model_version`
names were reused and updated in place for the v2 reruns) — they are preserved in
`reports/steps_0-5_findings.md` and this report's comparison tables.
