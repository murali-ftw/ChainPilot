# Step 7 — Transformer 2 (Global Attention, Claim 3) on SHARE + Variant A

**Scope:** implement Transformer 2 (global same-type attention, Claim 3) on top of the
finalized Layer 1 + Layer 2 stack — SHARE (RGCN+Attention, `rgcn_attn`) as the encoder,
Variant A (bounded residual on the Markov floor, `reports/rung5_types.md` Type A) as the
depth-selection mechanism. A new, purely additive component — no changes to SHARE's encoder,
SHARP, SHARK, or the existing Markov/Variant A depth logic.

---

## The wiring deviation — stated up front, per this round's own instruction

**`docs/10_AI_ML_Documentation.md` §8.3 specifies wiring Transformer 2's output to the delay
head only** ("Phase 1: wired to the delay head only (supplier-level)"). **This implementation
wires it to the IMPACT head instead.** The reason is direct evidence from this project's own
Step 6 Round 1 reach analysis (`reports/layer_2.md`): the co-parent/hidden-dependency path
needs a hard minimum of **3 hops** from delay's actual prediction target (Shipment) —
`Shipment → Supplier (1) → Component (2) → co-parent Supplier (3)`, confirmed 100% of
supplier-sourced shipments, 0% at hop 1 or hop 2, in a live measurement against the same
dataset this round uses. Delay reads `h^1` under Markov/Variant A — one hop — so the
co-parent/hidden-dependency signal is **structurally out of reach for delay at any depth this
project has ever tested, baseline included.** The same Round 1 analysis found impact's target
(Supplier, hop 0 by definition) **does** have real co-parent signal available from hop 2
onward, and impact reads `h^4` under Markov/Variant A — well within reach. Wiring Transformer 2
to delay, per the original doc's untested assumption, would have attached it to the one head
structurally incapable of using it. This is not a silent substitution — it is flagged here, in
the model's own module docstring (`ml/models/rgcn_attn_variant_a_transformer2.py`), and in
every governance record this round writes.

---

## Device check (done first, per this project's protocol)

**MPS was used for training — verified before choosing it, not assumed:**

1. **Correctness.** The new ops this component needs — cosine similarity (normalize + matmul),
   `fill_diagonal_`, `topk`, advanced-index gather (`K[topk_idx]`), and the resulting
   attention/softmax — ran forward+backward on a synthetic 800×128 batch on CPU vs MPS:
   candidate indices identical, output values allclose to `atol=1e-4` (max diff ~1.5e-7),
   gradients allclose (max diff ~1.5e-6).
2. **Speed.** This component alone showed a modest ~1.34x speedup on MPS (10.1ms/iter CPU vs
   7.5ms/iter MPS) — smaller than the encoder's own ~2.5x (`reports/layer_2.md` Round 5), since
   this op set is cheap and dominated by per-op dispatch overhead rather than compute. Combined
   with the encoder, which dominates total forward/backward cost, MPS remained the better choice
   overall, consistent with every recent round's device check.

---

## Design

**Same-type-only global attention over Supplier embeddings, no adjacency mask.** Operates on
SHARE's own Supplier-type node embeddings, POST-encoder, at whatever depth the impact head
currently reads under Variant A (`z_impact_i` — Variant A's own per-node bounded-residual
blend, not a raw fixed `h^4` read, since that blend is literally what "the impact head
currently reads" under the finalized stack). No adjacency mask is the entire point — Transformer
2 exists specifically to connect Supplier pairs with NO graph path between them (the
"moralization" case, `reports/rung5_types.md`'s own citation of
"Markov Scoping and Transformer 1.md" §1.3/§2.4); restricting the pool to graph-adjacent pairs
would defeat the mechanism outright.

**Bounded candidate pool: top-k=64 by cosine similarity, per Supplier.** Matches
`docs/10_AI_ML_Documentation.md` §8.3's own bounding choice exactly — NOT bounded by shared
`component_type`, which that section explicitly flags as wrong (same-`component_type`
suppliers are already 2-hop reachable via the co-parent path, so that bounding would exclude
exactly the cross-type correlation the hidden-dependency scenario and Transformer 2 both exist
to find).

**`is_frontier` override — implemented, but INERT this round, disclosed explicitly.** The
design calls for always including frontier-flagged suppliers regardless of cosine rank.
`docs/05_Database_Design.md` §6.23 documents `suppliers.is_frontier` as "data-gated" and
confirms it does NOT exist in the live schema — re-confirmed live before writing this round's
code (`\d suppliers` has no such column). The override path is implemented correctly
(frontier candidates displace each row's lowest-scoring cosine candidates, keeping pool width
fixed) but is never exercised (`is_frontier=None` throughout this round's training), since
there is nothing to flag. A disclosed limitation carried over from the schema's own documented
gap, not a silently-dropped requirement.

**Fusion into the impact head.** `z_impact_fused = z_impact + t2_scale * t2_output`, where
`t2_scale` is a single learned scalar initialized to exactly 0 — matching this project's
established convention throughout Step 6 (every new component starts identical to the
validated baseline it extends, so training can only move away deliberately). At `t2_scale=0`,
this component's forward pass is numerically identical to plain Variant A's impact readout.

**Implementation.** New files: `ml/models/transformer2.py` (`Transformer2GlobalAttention`)
and `ml/models/rgcn_attn_variant_a_transformer2.py` (`Transformer2HADESModel`, which imports
and reuses `Rung5VariantAGateHead` and `PredictionHead` UNMODIFIED for every task's own
depth-selection and readout — no changes to `ml/models/rgcn_attn_rung5_variant_a.py`).
Registered as a new architecture, `rgcn_attn_variant_a_transformer2`, in `ml/models/encoder.py`
/ `ml/train.py`. Full model at `hidden=128, num_bases=10`: **814,627 params** (765,090 for
Variant A alone + 49,537 for Transformer 2's Q/K/V projections + the scale scalar — confirmed
live, exact match to the formula `3*(128²+128) + 1`).

**New table `hidden_dependency_links`** (`db/hidden_dependency_links_schema.sql`, per
`docs/05_Database_Design.md` §6.25 exactly — new `validation_status` ENUM plus the table, both
newly created this round). **Logging is bounded, not literal "every discovered pair"**: the
top-500 pairs by symmetric attention score, plus the 6 H_POLYMER ground-truth pairs
unconditionally (even when they don't clear that cut), for the most recent test snapshot only,
across all 5 trained seeds. Logging every discovered pair across 5 seeds × 6 snapshots would
be up to ~25,600 unique pairs per (seed, snapshot) — not a reviewable "investigation leads"
list per §6.25's own framing ("a row is an investigation lead, never a fact"). A disclosed
bounding decision.

---

## Validation — the planted hidden-dependency scenario

**Ground truth.** `db/README.md`'s "hidden-dependency scenario": four suppliers, in four
different countries, supplying different (non-polymer-exclusive) component types, share an
unmodeled upstream polymer plant — no row, no edge, no column connects them in the schema.
`db/generate_dataset.py`'s `H_POLYMER` selection is fully deterministic (first-generated
supplier, by component insertion order, for each of 4 distinct non-polymer component types) —
reconstructed read-only from `db/csv/components.csv`'s generation-order rows
(`ml/graph/hidden_dependency_ground_truth.py`), never by re-running the generator (which would
have reloaded the live dataset this whole project's history runs against — far outside this
round's scope). Cross-checked against the live database and matches `db/README.md`'s own
published table exactly: USA-Poly Supply 10, IND-Precision Supply 112, CHI-Precision Supply
00, VIE-Alloy Supply 01 — 4 countries, India's member specifically lacking `fastener` (matching
the README's table cell-for-cell).

**The chance baseline, concrete, not raw attention scores presented as self-evidently
meaningful.** Two metrics, both compared against a defined null:

1. **Percentile rank** of each H_POLYMER pair's cosine similarity among ALL C(800,2)=319,600
   possible Supplier pairs in that snapshot. Chance expectation under a null of no structure:
   uniform, mean percentile = 0.500.
2. **Pool-membership discovery rate**: does the pair appear in either endpoint's own top-64
   candidate pool? The chance baseline here is not simulated — it's measured directly and
   exactly: the **global discovery rate**, i.e. what fraction of ALL 319,600 possible pairs are
   discovered this way, purely from the pool-size mechanics of the trained embeddings.

**Results — 5 seeds × 6 test snapshots × 6 pairs = 180 pair-observations:**

| Seed | Mean percentile rank | Discovery rate | Global chance discovery rate |
|---|---|---|---|
| 0 | 0.7702 | 0.694 (25/36) | 0.1022 |
| 1 | 0.7314 | 0.917 (33/36) | 0.1066 |
| 2 | 0.7318 | 0.667 (24/36) | 0.1050 |
| 3 | 0.7099 | 0.694 (25/36) | 0.1060 |
| 4 | 0.6841 | 0.750 (27/36) | 0.1043 |
| **Overall** | **0.7255** (chance: 0.500) | **0.7444** (chance: **~0.105**) | — |

`P(all 6 H_POLYMER pairs discovered by chance alone) ≈ global_rate^6 ≈ 0.000001` (a crude
independence-assuming supporting statistic, not the primary metric, reported because it's a
simple additional sanity check, not because the 6 pairs' discovery events are actually
independent of each other).

**VALIDATION VERDICT: PASS.** The planted hidden-dependency scenario is recovered well above
chance on both metrics, consistently across all 5 seeds — every single seed's mean percentile
rank exceeds 0.68 (vs. 0.50 chance) and every seed's discovery rate exceeds 0.66 (vs. ~0.10
chance). This is not one lucky seed; it replicates.

---

## Impact AUC — with vs. without Transformer 2

| Arm | delay | shortage | impact |
|---|---|---|---|
| baseline (Variant A alone) | 0.8166 ± 0.0015 | 0.7974 ± 0.0038 | 0.9434 ± 0.0023 |
| Transformer 2 (fused into impact) | 0.8186 ± 0.0021 | 0.7996 ± 0.0034 | 0.9455 ± 0.0019 |

**Paired bootstrap + 5-seed sign-consistency:**

| Task | Per-seed ΔAUC | Mean ΔAUC | Verdict |
|---|---|---|---|
| delay | `[-0.0001, +0.0022, +0.0050, -0.0012, +0.0040]` | +0.0020 | FLIPS (noise) |
| shortage | `[+0.0053, +0.0004, +0.0041, +0.0047, -0.0035]` | +0.0022 | FLIPS (noise) |
| impact | `[-0.0040, +0.0055, +0.0074, +0.0007, +0.0010]` | +0.0021 | FLIPS (noise) |

**No statistically reliable AUC change on any task, including impact — the head Transformer 2
was actually wired to.** All three mean deltas are small and positive, and impact's own 4-of-5
seeds are positive, but one negative seed (−0.0040) breaks 5-seed sign-consistency. Delay and
shortage move by a similar small amount despite receiving zero direct input from Transformer 2
(expected — they share the same encoder/gates, so run-to-run training variance alone produces
comparable-sized noise on every task).

---

## Verdict — adopt or not?

Per this round's own instruction: **"Adopt Transformer 2 only if this validation clears the
better-than-chance bar. If it doesn't, report that plainly as a negative result."** The
validation DID clear the bar, decisively (percentile rank 0.73 vs. 0.50, discovery rate 0.74
vs. ~0.10, both consistent across all 5 seeds). But the mechanism's OTHER stated purpose —
improving impact AUC by fusing the discovered signal into the impact head — did not
materialize; every AUC comparison flips sign across seeds.

**These are not the same question, and this project's own validation methodology
(`docs/10_AI_ML_Documentation.md` §9.4) explicitly treats them as separate claims requiring
separate evidence** — "Transformer 2 [validated by] held-out edge recovery... never by a single
high attention score in isolation" is about whether the mechanism finds real structure, not
whether that structure currently helps the specific downstream task it's fused into. On that
narrower, correctly-scoped question: **Transformer 2 works exactly as designed** — it
reliably, above chance, across every seed tested, recovers a hidden-dependency signal that no
graph-based message-passing mechanism in this project's entire history (SHARE, SHARP, SHARK,
Markov, or any Rung-5-family gate) could ever have reached, because by construction no edge
connects the planted suppliers. That the resulting embedding shift doesn't yet move impact's
AUC in a way five seeds agree on is a real, honestly-reported limitation, not a reason to
discard the component — consistent with the depth-gate's own precedent in this project
(`reports/layer_2.md` Round 2: "the gate does exactly what it was mechanistically designed to
do... that mechanistic success does not, however, convert into a broad AUC win").

**Recommendation:** keep Transformer 2 as a validated discovery/investigation-leads mechanism
(the `hidden_dependency_links` table is real, working, and populated with a genuine
better-than-chance signal a human reviewer can act on) while treating its fusion into impact's
AUC as unproven and not yet a production accuracy improvement. A natural follow-up, not
attempted here: does the fusion help more at a higher `t2_scale` than training alone found, or
does giving the gate more capacity/epochs to exploit the signal change the AUC picture — the
same "test one isolated change at a time" discipline this project used throughout Step 6.

---

## Governance record

Backed up before any write
(`reports/backups/model_registry_and_evals_backup_20260808_201700.sql`). New table
`hidden_dependency_links` created (`db/hidden_dependency_links_schema.sql`) — 0 rows before
this round, since it didn't exist. 10 new registry rows — 5
`rgcn_attn_rung5_a-t2pilot-baselinecmp-seed{n}` (architecture=`rgcn_attn_rung5_a`) + 5
`rgcn_attn_variant_a_transformer2-t2pilot-seed{n}`
(architecture=`rgcn_attn_variant_a_transformer2`), all `status='active'` — **308 total registry
rows** (298 + 10). **150 new evaluation rows** (10 runs × 15 metric rows) — **4,488 total**
(4,338 + 150). **2,524 new `hidden_dependency_links` rows** (5 seeds × ~500–506 each: 500
top-ranked + the 6 H_POLYMER ground-truth pairs, minus however many of those 6 already ranked
inside the natural top-500 for that seed — 0 already-present for seeds 0–2, 3 already-present
for seeds 3–4). No existing row from any prior round modified. Full run log:
`reports/logs/run_transformer2_pilot_20260808_202618.log`.

---

*Scoped to SHARE + Variant A only, per this round's own framing — no changes to SHARP
(`rgcn_relemb`), SHARK (`rgcn_battn`), or the existing Markov/Variant A depth-selection logic.
Wired to the IMPACT head, deviating from `docs/10_AI_ML_Documentation.md` §8.3's delay-only
specification — see the deviation section above for the full rationale.*
