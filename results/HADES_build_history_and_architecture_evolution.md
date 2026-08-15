# HADES — Everything Built and Tested So Far, and How the Architecture Evolved

*Compiled from: `HADES_v2/docs/{01_PRD,02_TRS,11_Implementation_Guide,14_Project_Roadmap}.md`,
`HADES_v2/findings/evolution.md`, `HADES_v2/ml/{README.md,train.py,data/loader.py}`,
`HADES_v2/STEP5B_DECISION_SUPPORT_BUILD_PROMPT.md`, `HADES_v2/reports/decision_support_build.md`,
`HADES_v3/docs/HADES-v3_init.md`, `HADES_v3/docs/14_Project_Roadmap.md`,
`HADES_v3/reports/{phase0_audit,phase1_latent_state,layer3_uncertainty_aware}.md`,
`HADES_v3/db/README.md`. Every number below is sourced from one of these files; nothing here is
inferred or estimated.*

---

## 1. How to read this

This project has gone through three architecture generations (V1, V2, V3), and within each
generation individual layers went through their own bake-offs, failures, and pivots. The
organizing discipline that emerges — and gets stricter every generation — is: **never trust a
single number without a measured reproduction floor, and never let a phase's failure get silently
patched over; report it, diagnose it, and let the diagnosis decide what gets built next.** Section
14 pulls that discipline out explicitly since it's the throughline connecting every pivot below.

Section 2 is a one-screen timeline. Sections 3–13 are the full chronological build history.
Section 15 is the current planned architecture with a per-layer status table. Section 16 is a
file/component inventory of what's actually implemented on disk.

---

## 2. Timeline at a glance

| Era | What happened | Headline result | Outcome |
|---|---|---|---|
| **V1** | Original HADES/ChainPilot application: HGT encoder, fixed-depth head, planned global attention head | — | Superseded; V1 PRD/TRS preserved only in git history |
| **V1 encoder bake-off** | 7 encoders tested, 5 seeds each | SHARE best-or-tied on all 3 tasks | **SHARE replaces HGT** as production encoder |
| **V1 depth investigation** | Learned Task Gate, Rung 4/5, Variants A–E tested | Markov Floor (fixed per-task depth) never beaten | **Markov Blanket readout adopted** |
| **V1 Layer 3 (Transformer 2)** | Global attention head vs. planted `H_POLYMER` scenario | Discovery worked; no fusion design produced accuracy gain | Relationship was causally redundant; kept as zero-cost default only |
| **V2 — HADES-Bench rebuild** | 10 mechanisms, 12 variants, determinism + leakage invariants | Phase 0 found resilience recoverable for only 5–9% of suppliers | **Mechanisms E/F deliberately built last**, after Mechanism H |
| **V2 architecture re-validation** | SHARE and Markov Floor re-tested on 12 new variants, incl. genuinely heterogeneous chain length (Mechanism J) | SHARE still best-or-tied; Markov Floor still unbeaten across ~130 runs, 8 gate variants | **Both closed as production defaults** |
| **V2 Layer 3, take two** | Genuine hidden coupling (Mechanism B) built; 5 independent retrieval redesigns tried | All 5 hit the same wall: group-mean dilution | **Layer 3 scrapped as discovery mechanism**; kept only as calibrated hypothesis-generation tool |
| **V2 decision-support build** | Counterfactual reasoning, uncertainty estimation, explainability — 3 phases on the frozen V2 backbone | Counterfactual: clean miss (0.478 vs chance 0.500, floor 0.324). Uncertainty: dataset-seed variance = 68–99% of total. Explainability: 0.166 faithfulness pass rate vs 0.100 chance | Counterfactual failure **directly motivates V3's SCM layer** |
| **V3 — architectural pivot** | New init doc: V1/V2 only ever answered `P(Y\|X)`; V3 adds layers to answer `P(Y\|do(X))` | — | SHARE + Markov inherited unchanged; 3 new layers planned (Latent State, SCM, Counterfactual Engine) |
| **V3 Phase 0 — audit** | Inheritance/environment audit before any V3 training | `ml/` and torch venv didn't exist yet in V3 repo; froze()-check bug found | Ported cleanly; 5/5 checks pass; 2 roadmap doc defects found |
| **V3 Phase 1 — latent state** | 6 candidate hidden states + 1 bonus tested for ground truth and recoverability | Only 2 of 7 cleared: Supply Stress (AUC 0.64–0.65), Recovery Capability (AUC 0.65, E-variants only) | Supplier Reliability failed; Inventory Health/Capacity Pressure dropped (no real latent variable) |
| **V3 Layer 3 continued — Steps 1–6** | Sufficiency, identifiability, calibration, per-entity confidence, threshold gating | Supply Stress & Recovery Capability genuinely identifiable via real `do(Z)`. Calibration fails to transfer across worlds (STOP–G). Confidence signal doesn't order error at all (STOP–G) | **Zero states reach a calibrated, confidence-aware, threshold-gated output** — currently the project's frontier |
| **Now / queued** | World-Conditioned Calibration build prompt drafted (this session); real-world supply-chain datasets scouted for future external validation | — | Not yet run |

---

## 3. V1 — the original HADES (ChainPilot)

The project began as a product, not a benchmark: `01_Product_Requirement_Document.md` refers to it
as "the V1 ChainPilot/HADES application," and `02_Technical_Requirement_Specification.md` was
"previously the technical spec for the V1 application stack." Neither the original product
architecture nor its full requirements survive in the current docs — they were superseded and are
"preserved in git history" only. What we know for certain about V1's architecture comes from
`findings/evolution.md`'s account of what it started with and immediately began testing against
alternatives:

> *"Original 3-part design: HGT encoder + short fixed-depth head + planned global head (Transformer
> 2)."* — `findings/evolution.md` §1

That is: a Heterogeneous Graph Transformer encoder (every one of the graph's ~20 meta-relations
gets a fully independent attention/message-matrix set), a fixed-depth readout, and a planned
non-local attention mechanism meant to catch relationships the graph schema itself couldn't encode.
All three became open questions almost immediately, and the next three sections are the record of
answering them.

---

## 4. V1 → V2: the first architecture bake-off

### 4.1 Layer 1 — choosing an encoder (SHARE is born)

Seven encoders were built at matched parameter budgets and tested on delay/shortage/impact AUC
across five seeds (`findings/evolution.md` §2):

| Architecture | Idea | Result |
|---|---|---|
| GraphSAGE | one shared transform for all relations | never wins; establishes the floor |
| GAT | attention within relation, relation-blind | weakest overall (shortage AUC 0.6560) |
| HGT (the original) | full per-relation independence | best on impact (0.9335), loses to RGCN family on shortage every round |
| RGCN | basis-decomposition sharing | confirms thin-relation-overfitting hypothesis on shortage; worst on impact (0.8509) |
| **SHARE** (RGCN + shared attention) | RGCN transform + one shared, relation-agnostic attention scorer | **best or tied-for-best on all three tasks simultaneously** |
| SHARP (SHARE + relation embedding) | small per-relation tag fed to scorer | statistically tied with SHARE, costs more, no gain |
| SHARK (SHARE + per-relation attention matrix) | most expressive attention tested | same accuracy as SHARE/SHARP but ~10× more seed-to-seed instability (impact std 0.056 vs SHARE's 0.006) |

Why: four of the graph's ten relations are structurally thin (hundreds to a few thousand edges vs.
tens of thousands for the rest). HGT gives each a fully independent parameter set anyway and
overfits them — which is why RGCN beats HGT on shortage. But RGCN's shared-transform simplicity
costs it on impact, where a genuinely standout neighbor must be singled out regardless of relation
type. SHARE keeps RGCN's thin-relation protection and adds one shared attention scorer, recovering
impact (0.9365–0.9393, tying HGT) without giving back shortage (beating HGT by +0.0147–0.0167), with
no elevated seed instability, and cheaper than SHARP/SHARK.

**SHARE became the production encoder — the original HGT component was replaced entirely.**
Confirmed later on V2's independently-generated benchmark: best-or-tied-for-best on all three
tasks, across **all twelve** V2 mechanism variants.

### 4.2 Layer 2 — should readout depth be a decision? (the Markov Blanket readout)

The question: should every task read the encoder's last layer, or should depth be chosen — per
task, or per node? Five independent mechanisms were tried in V1 (`evolution.md` §3.1):

| Mechanism | Idea | Outcome |
|---|---|---|
| Learned Task Gate | one learned depth-blend per task, KL-anchored to a theory prior | recovers each task's known depth preference, but only 1 of 9 cells clears significance — "mechanistically correct, practically inert" |
| **Markov Floor** | zero-parameter, per-task **fixed** depth from graph structure: delay→h¹, shortage→h³, impact→h⁴ | delay +0.0072 AUC, consistent across all 5 seeds, zero cost — **became production baseline** |
| Rung 4 / Rung 5 | per-node MLP or attention gate, free to deviate from Markov | never beat Markov; wildly unstable, no accuracy difference between stable and unstable runs |
| Variant A | Rung 5 + hard cap on gate's drift from Markov | 100% match rate, every seed, every cap — adopted as a safety-hardened upgrade |
| Variants B, C, D | structural features / freeze-then-unfreeze / per-depth projection | partial or no stability fix |
| Variant E | post-hoc weight-averaging across independently-trained seeds | **catastrophic failure** — worse than random guessing |

V1's dataset had structurally uniform chain length, so no variant ever had real per-node
heterogeneity to adapt to. V2 later built Mechanism J specifically to give adaptive depth a fair
shot (varying chain length per product) — the first dataset in either project capable of genuinely
testing it. The result held, and got a mechanistic explanation (§5 below).

### 4.3 Layer 3, attempt 1 — Transformer 2, the global attention head

Built to catch hidden relationships the graph schema can't encode as an edge — e.g., two suppliers
secretly sharing an unmodeled upstream plant. Tested against a planted scenario (four suppliers,
four countries, sharing one unmodeled `H_POLYMER` plant): discovery **worked decisively** — mean
percentile rank 0.7255 vs. chance 0.500, discovery rate 5.5–7.5× chance, consistent across seeds.
But four fusion designs (plain additive, confidence-weighted, per-node trust gate, cross-attention)
all had accuracy deltas that flipped sign across seeds. Root cause: `H_POLYMER` was real and
causally grounded, but **causally redundant** for prediction — each member's own observable history
already fully captured its own risk. Confidence-Aware Fusion was adopted as the standing default
anyway: not for an accuracy win (none produced one), but because it added zero parameters and
best-preserved discovery quality while genuinely coupled data didn't yet exist to test against.

---

## 5. V2 — HADES-Bench: rebuilding the benchmark itself

V2 reframed HADES from a product into a benchmark purpose-built to answer whether graph learning
architectures can *reason* about supply chains rather than fit a static graph
(`01_PRD.md` §1). It introduced a synthetic-data generator with hard invariants:

- **Determinism** — identical seed+config produces byte-identical `.csv.gz` output (verified by
  diffing compressed bytes, including across differing `PYTHONHASHSEED`).
- **No temporal leakage** — every value observable at time t must be computable from events ≤ t;
  enforced by executable assertions (`assert_not_future`, `TemporalLeak`).
- **Stdlib only**, no runtime dependencies for generation.
- **Validation must pass, never be weakened to pass a mechanism** — a check that's statistically
  under-powered at a given config *reports* its numbers rather than silently gating.

Ten independently-switchable mechanisms (A–J) compose into twelve benchmark variants (0, A–K), each
isolating one hypothesis: partial visibility (A), hidden shared structure (B), dynamic
relationships (C), hidden dependency coupling (D), hidden resilience (E), adaptive transmission
(F), information delay (G), external shocks (H), multi-source dependencies (I), chain-length
heterogeneity (J). Three variants have prerequisites that must not be skipped when attributing an
effect: A = J+A, D = B+D, F = E+F.

### 5.1 The plan changed almost immediately: Phase 2 got deferred

V2's build phases ran in a **revised order**, not the nominal one, because of an early finding:

> *"Phase 2 was gated after Phase 0 found resilience recoverable for only 5–9% of suppliers, so
> Mechanisms E/F were deliberately built last, after Mechanism H existed to be tested against."*
> — `HADES_v2/docs/14_Project_Roadmap.md` §1

Executed order: **0 → 1 → 3 → 4 → 5 → H-coverage recheck → 2 → 6.** By the time E/F (resilience,
adaptive transmission) were finally built, coverage had been re-verified via a dedicated gate step
(`phase2_coverage_recheck.md`), and a genuine empirical surprise turned up along the way: Mechanism
A was found to *improve* recoverable coverage rather than degrade it — Variant F, not Variant K, is
the harder case (coverage ceilings 71.7% vs. 86.3%).

### 5.2 What V2's benchmark build actually completed

All ten mechanisms implemented and independently switchable; all twelve variants generate with
zero validation failures across 12×5 = 60 runs; Variant 0 stays byte-identical to V1 through every
phase; a PostgreSQL load path catches at least one referential-integrity bug the in-generator
suite alone missed. Phase 6 (the full spec-scale generation matrix) was initially left partial — the
spec-scale sweep hadn't been run yet — but by the time `HADES_v3/db/README.md` was written, the
benchmark had reached its full intended scale: `SUP_N = 4,000`, 40 monthly snapshots, twelve
variants at two retained seeds (42, 43; 10.2 GB), with seeds 44–46 regenerable on demand rather
than stored (a full 12×5 sweep would run ~25 GB against a ~15 GB target).

Along the way the dataset itself went through its own realism/scale evolution — worth noting since
it's exactly the same "find a problem, fix it, re-verify determinism" discipline the model
architecture went through: world scale grew 50 → 180 → 800 → 4,000 suppliers across versions;
realism fixes included adding timestamp jitter (fixing a "looks synthetic" zero-jitter tell),
making topology evolve between snapshots, recalibrating label rates against spec guidance, and
fixing a genuine determinism bug where Python's `set` iteration order was silently
`PYTHONHASHSEED`-dependent (fixed with `sorted()`, reverified byte-identical across four runs).

### 5.3 SHARE and the Markov Floor, re-validated — not assumed

V2 didn't just carry SHARE and the Markov readout forward on faith. Both were re-tested on the new,
independently-built dataset. SHARE reconfirmed best-or-tied-for-best across all three tasks and all
twelve mechanism variants. The Markov Floor was re-tested specifically against Mechanism J's
genuinely heterogeneous chain lengths — the first fair test either project had run — and the result
didn't just hold, it got a mechanistic diagnosis (`evolution.md` §3.2):

- **Variant A's "100% stability" in V1 was never real evidence.** Its logit formula
  (`fixed_prior + λ·tanh(residual)`) had a prior gap of 8.0, and every λ ever tested (0.1–1.0, in
  either project) was 8–40× too small to ever flip the decision. Its predictions equaled Markov's
  predictions node-for-node on all 195,029 scored rows, every seed, every variant tested — "a
  restatement of an inequality, not evidence."
- **Base Rung 5, which could actually move, never learned per-node preferences** — it collapsed
  into picking one global depth for the entire task, changing randomly by seed. Traced to the gate
  mean-pooling a node's five depth-token embeddings into one vector before scoring them, destroying
  the per-node information it needed.
- When Variant A's cap was deliberately widened past mathematical possibility (λ=6.0, 10.0), the
  gate did move — on 80–90% of nodes — and accuracy got measurably worse, not better.

Conclusion, closed after ~130 training runs across eight gate variants and two independently-built
datasets: the fixed Markov Floor stands as production default. Three specific, falsifiable
reopening conditions were named (more label volume at spec scale, a gate that scores per-depth-token
rather than mean-pooling first, or a dataset placing predictive signal at different depths per
node) — but this is a closed question until one of them is met, "not an open one" (carried forward
verbatim into the V3 init document).

---

## 6. V2 Layer 3, take two — genuine coupling, and why it was scrapped for good

V1's Layer 3 test used a coupling (`H_POLYMER`) that turned out to be causally redundant — not a
fair test of whether hidden-relationship discovery could ever pay off. V2 built Mechanism B
specifically to generate **genuinely** causally-coupled hidden groups (Type A: real coupling via
Mechanism D; Type B: correlated but redundant, reproducing `H_POLYMER`; Type C: pure decoy). Five
independent approaches were then tried, each hitting a wall in a different way (`evolution.md` §4):

1. **Retrieval itself broke.** Percentile rank 0.479–0.514 vs. 0.500 chance — indistinguishable
   from the inert decoy, at both scales tested, across all four fusion designs. The compared
   embeddings had never been trained to encode co-membership; they were shaped entirely by main
   prediction losses.
2. **Privileged contrastive supervision (ceiling test).** An auxiliary loss trained directly
   against true group labels worked spectacularly on the groups it was shown — Recall@64 rose to
   10.9× chance, MRR to 24.3× — but failed completely to generalize to held-out groups from the
   *identical* generative process: 0 of 20 cells cleared their own reproduction floor. It had
   memorized which suppliers were labeled related, not learned relatedness — and it cost impact AUC
   0.145–0.158, sign-consistent on 5/5 seeds, "roughly thirty times the arm's own noise floor" —
   the single largest effect measured anywhere in the project.
3. **Observable-only behavioral retrieval.** Built from the generator's own validation statistic;
   came back a clean null. A confound was found along the way: the dataset's own unrelated
   shared-factor pools (port, trucking, customs) produce 363 coincidental co-degrading pairs for
   every one real hidden-parent pair.
4. **Representation probe.** Fit the same discrimination objective at every encoder depth,
   including raw input features, against a permutation-matched null. At no depth does true
   co-membership separate from a random regrouping by more than the null's own noise.
5. **Reframe — Behavioral Hypothesis Generation.** Instead of committing to one hidden cause,
   generate a ranked, calibrated set of plausible explanations with an explicit "unknown" option.
   This one *worked*, but honestly: regional/logistics causes scored AUC 0.817 (predicted vs.
   empirical rates agreeing to within 0.011 across ten bins of 13,000+ instances); the target
   hypothesis (shared upstream dependency) was correctly and honestly never claimed — confidence
   never exceeded 0.13% on any variant, even on pairs genuinely sharing a hidden parent.

**The diagnosis that ended it:** at the same coupling strength (0.35), a *direct pairwise* hidden
relationship is recoverable from behavior alone at AUC 0.78 — the retrieval pipeline itself is
sound. But the *group-averaged* hidden-parent relationship tested throughout is not recoverable at
any pool width, on either variant. Spreading a hidden parent's influence across an average of four
or five members dilutes any individual pair's share of it below the point where the pair can be
individually attributed — a property of how the generator produces hidden groups, not of any tested
architecture.

**Why scrapped rather than fixed:** five substantially different approaches across two
independently-built datasets all hit the identical wall — "the signature of a data-generation
constraint, not a modeling one." A concrete fix exists (make hidden-parent coupling pairwise rather
than group-mean-mediated), but that's a generator change, out of scope for the model architecture
line, and would reopen a question this project closed with a clean diagnosis. Layer 3 as a
discovery mechanism was scrapped; the hypothesis-generation module survives as a calibrated, honest
decision-support tool for *observable* correlated-risk explanations — and the project's attention
redirected toward counterfactual reasoning, uncertainty estimation, and calibrated explanation,
built on top of the now-closed SHARE + Markov Blanket foundation.

---

## 7. V2 decision-support build — counterfactual reasoning, uncertainty, explainability

Built entirely on the frozen V2 backbone (SHARE + Markov readout), verified untouched by `git diff`
at every stage. Scale: Variant 0, `v1` preset (800 suppliers, 15 snapshots).

### 7.1 Phase 1 — Counterfactual Reasoning: a clean miss, and why it matters

Two components were built: `ml/counterfactual_edit.py` (structural graph edit + unmodified frozen
forward pass — the "naive" approach) and `ml/counterfactual_ground_truth.py` (re-derives the true
effect in closed form from the generator's own causal chain, `stress → p_delay → p_impact`).

**Why closed-form, not full re-simulation:** inserting one extra random draw before shipment
simulation flipped 100% of shipment statuses and 14.4% of delay labels (179% of the true positive
count — unusable), but left shortage and impact labels completely unchanged (both robust, coarse
aggregates). Closed-form was used anyway for its continuous probability delta rather than a binary
flip — but this holds the realized schedule fixed, so it measures effect *only* through the stress
channel, explicitly excluding the second-order path where structural change alters who gets chosen
for future replenishment.

**Result — a clean miss, three interventions (remove supplier, add dual-sourcing, substitute
supplier), 900 interventions, `impact` task:**

| | pooled sign agreement | vs. chance (0.500) |
|---|---|---|
| single run | 0.524 | looked like a small edge |
| **15-run reproduction floor** (3 model-init seeds × 5 dataset seeds) | **mean 0.4778**, spread up to 0.324 per dataset seed | **below chance** |

The apparent 0.524 edge was 13× smaller than the measured floor — an artifact of which model seed
happened to train first, not evidence. Magnitude ratio (predicted/true effect size) was 0.132 —
predictions roughly 8× too small — and the model stayed silent (predicted delta < 1e-4) on 36% of
truly-affected entities, rising to 50% on the `remove_supplier` intervention.

**Phase 1b — because the miss was clean, a dedicated delta-prediction head was built and tested:**
`ml/counterfactual_delta_head.py` reads frozen embeddings before/after the edit plus reach features
and intervention type, trained on generator-produced (original, intervened, true-delta) triples with
a proper leave-one-dataset-seed-out split. Result: **worse than the naive baseline** — mean sign
agreement 0.4016 vs. naive's 0.5085, losing 13 of 15 paired folds; magnitude ratio median 21.7,
ranging seven orders of magnitude out to 1,344. Diagnosed cause: only ~800 nonzero-effect training
rows spread across three worlds unrelated to the test world's suppliers — the identical
identity-generalization wall Layer 3 hit.

**Verdict:** "Counterfactual effect prediction does not work here by either route at this scale."
SHARE and the Markov readout were trained as ordinary supervised predictors with no interventional
signal — they produce *an* answer to an edited graph, but that answer doesn't track the generator's
true effect. **This finding is the direct evidentiary basis for V3's SCM layer** (Section 8) — it's
cited verbatim in the V3 roadmap as the reason Layer 4 becomes load-bearing, not optional.

### 7.2 Phase 2 — Uncertainty Estimation: dataset-seed variance dominates

Built the first proper two-way variance decomposition (law of total variance,
`var_model = mean_d Var_m(AUC)`, `var_dataset = Var_d(mean_m AUC)`) over a 5×5 grid (5 dataset
seeds × 5 model-init seeds = 25 trainings):

| task | grand mean AUC | std_model | std_dataset | ratio | **dataset share of total variance** |
|---|---|---|---|---|---|
| delay | 0.7759 | 0.0047 | 0.0407 | 8.7× | **98.7%** |
| shortage | 0.7877 | 0.0061 | 0.0089 | 1.5× | **67.9%** |
| impact | 0.9333 | 0.0076 | 0.0119 | 1.6× | **70.8%** |

Against a prior single-axis estimate (§5.6 of an earlier report): delay confirms (10.3×→8.7×),
impact confirms, shortage revises upward (4.8×→1.5×) because the earlier estimate came from a
single dataset seed and was optimistic about model variance. Bottom line: on every task, most
predictive variance is dataset-level (68–99%) — an uncertainty estimate built only from
model-init ensembling would be measuring the smaller source throughout, on delay capturing roughly
one part in seventy-five of total variance.

**A trap found along the way — "supplier ID persists, the supplier doesn't":** the dataset-seed
term is not identifiable per-entity, and the data *looks* pairable when it isn't, because the
generator derives primary keys as UUIDv5 from stable keys (a supplier's stable key is its index) —
so supplier IDs are 100% shared across dataset seeds. Joining seed 42 to seed 43 on `id` succeeds
for all 800 suppliers and produces a per-entity "dataset variance" that means nothing, since the
entity behind the ID isn't the same business:

| attribute | agrees across worlds |
|---|---|
| `country` | 19.5% (chance ≈ 16.7%) |
| `lead_time_days` | 2.2% |
| `capacity_score` | 0.0% |
| `reliability_history` | 0.0% |

This was found because an earlier split assertion (written assuming suppliers *are* world-specific)
failed and the assumption turned out to be backwards — the assertion was replaced with one checking
attribute divergence below chance.

**Calibration:** raw ECE was badly miscalibrated (0.12–0.44, from focal loss) — isotonic
recalibration dropped it 9× to 17×, far beyond any floor, with discrimination essentially
preserved. But ensembling itself did *not* clear its own floor on any task (+0.0062 shortage,
+0.0015 impact, −0.0014 delay, against floors of 0.0120/0.0039/0.0102) — recalibration was doing
the work, not model-init ensembling, consistent with the variance decomposition above.

### 7.3 Phase 3 — Explainable HADES: modest, honest, and mostly unmeasurable

Built `ml/explain_prediction.py` (occlusion attribution: set one input feature to its per-node-type
median, re-run the frozen forward pass, measure the drop in predicted risk — attention weights were
deliberately not used, since they "look exactly like an explanation" without being checked for
faithfulness) and `ml/test_explanation_faithfulness.py` (ablate the cited factor, confirm the
prediction actually moves as claimed, against a matched null).

Two corrections were the substance of this phase:

1. **96.5% of `delay` predictions sit at p ≥ 0.999** — completely saturated, so occlusion of a
   single feature can't move them at all. Targeting was changed to a stratified draw across the
   remaining 3.4% non-saturated band, and that limitation is reported as first-class, not smoothed
   over.
2. **Three null formulations for the faithfulness test were tried**, and two were degenerate (a
   pooled 99th-percentile null was mis-scaled by ablation-sensitive entities; same-feature-on-random
   entities was vacuous because the pool is 96% saturated). The one reported —
   same-feature-on-other-non-saturated-entities — is matched and non-circular.

**Result, 1,142 explanations across 5 seeds:**

| seed | 42 | 43 | 44 | 45 | 46 | mean |
|---|---|---|---|---|---|---|
| faithfulness pass rate | 0.125 | 0.150 | 0.229 | 0.209 | 0.117 | **0.166** |

Chance is 0.100 by construction. Above chance on 5/5 seeds, ~7 standard errors above chance at this
sample size — **cleared, modestly, at 1.66× chance.** `days_since_dispatch` drove 74% of all
citations but only passed specificity 12.5% of the time (a dominant mechanical driver, not a
specific explanation); `status_scheduled` and `status_delayed` were rarer but far more specific
(77.8% and 100% pass rates).

**Overall verdict table for this build session:**

| phase | outcome |
|---|---|
| 1 — Counterfactual | **HIT (clean miss)** — naive 0.478 below chance; interventionally-supervised head 0.402, worse still |
| 2 — Uncertainty | **Delivered** — dataset-seed variance 98.7% / 67.9% / 70.8% of total |
| 3 — Explainability | **Cleared, modestly** — 0.166 vs. chance 0.100, but only on 3.4% of predictions |

---

## 8. The pivot — HADES V3, from `P(Y|X)` to `P(Y|do(X))`

The V3 init document states the reframe directly: V1 and V2 both trained architectures that only
ever answer *association* — `P(Y|X)` — because Pearl's causal hierarchy places prediction at rung 1
and intervention at rung 2, and no amount of rung-1 data alone identifies rung-2 quantities without
either interventional data or explicit structural assumptions. V2 had neither. The Phase 1
counterfactual clean-miss (Section 7.1) is cited as the direct empirical confirmation of exactly
this gap — SHARE and the Markov readout produce *an* answer to an edited graph, but it doesn't track
the true effect, because they were never trained or structured to.

**V3's architecture adds three new layers on top of the unchanged V1/V2 foundation:**

```
Supply Chain Graph
  → SHARE                          (Layer 1, UNCHANGED — inherited, closed question)
  → Markov Blanket readout         (Layer 2, UNCHANGED — inherited, closed question)
  → Latent State Estimation        (Layer 3 in V3 — NEW; estimates hidden operational
                                     STATE, explicitly not hidden identity — V2's Layer 3
                                     question stays closed and is not reopened)
  → Structural Causal Model        (Layer 4 — NEW; explicit causal equations, extracted
                                     directly from the generator's known code rather than
                                     learned from data)
  → Counterfactual Engine          (Layer 5 — NEW; evaluates interventions by simulating
                                     through Layer 4's SCM, not a raw forward pass)
  → Prediction Heads               (delay/shortage/impact + confidence + explanation +
                                     counterfactual outcomes, as parallel outputs)
```

Layer 4's design choice — extract equations from the generator's known code
(`own_stress`, `COPARENT_COUPLING`, `HP_ALPHA`, resilience absorption terms) rather than discover
them from data — sidesteps causal discovery, explicitly named as "the single hardest open problem
the original proposal would otherwise have inherited." The init document is careful to flag the
resulting validity caveat itself: this works *because* it's a synthetic benchmark with known coded
equations, and does **not** by itself establish real-deployment readiness, where no generator exists
and structure would need hand-specification or causal discovery from real data — "a different and
substantially harder problem this project has not attempted."

**What V3 inherits without re-doing:** the full V2 dataset (already generated), SHARE and the
Markov readout (both explicitly closed questions, confirmed unchanged via `git diff` at every
stage), and the prototype decision-support machinery from Section 7 — not V3 architecture per se
(V2's Layer 3 stays scrapped), but its harnesses, the frozen-backbone loading pattern, and Phase 1's
already-answered naive-counterfactual question carry forward directly.

---

## 9. V3 Phase 0 — the inheritance audit

Before any V3-specific training, five checks were run (`reports/phase0_audit.md`):

1. **Backbone integrity.** Initially **blocked** — the V3 repo had no `ml/` directory at all, only
   `db/` and `docs/`. After porting `ml/` (cleaning 15 orphaned V2 modules by computed transitive
   import closure — abandoned Layer 3 hypothesis-ranking/retrieval-redesign work, the closed
   adaptive-depth-gate line, the Transformer-2 line), the entire `ml/` tree matched V2 with zero
   differences.
2. **Freeze assertion.** Initially blocked (`ds_backbone.py` didn't exist yet; V3's `venv/` had no
   torch). After provisioning, a real bug was found: `freeze()` sets `requires_grad=False` on every
   parameter *first*, then checks whether any remain `True` — a post-condition self-check of its own
   mutation, not a detector of an externally-unfrozen backbone. Not a problem for V2 (which never
   re-enabled grads after freezing), but it matters for V3 because V3 attaches new trainable heads
   to the frozen backbone. **Recommendation carried forward: add `assert_backbone_frozen()`**,
   callable after head construction and before every training step. (Confirmed implemented and
   exercised — and never fired — across every subsequent V3 training run; see Sections 10–11.)
3. **Dataset determinism.** Pass — 20/20 `.csv.gz` files byte-identical on a fresh two-run
   comparison at spec scale.
4. **Dataset load integrity.** Pass on Variant 0 and Variant A (Phase 1's testbed) — all
   verification queries pass, zero rollback, label rates cross-checked exactly against the README
   (delay 18.86%, shortage 8.00%, impact 2.13%).
5. **Dataset inventory vs. README.** Pass, but with **two documentation defects logged**: the
   roadmap's claim that extended seeds 44–46 exist on disk for several variants was false (the
   directories exist but are empty — no data lost, just a stale claim); and a `v1_archive` folder
   the roadmap references doesn't exist anywhere (the actual fallback data lives under different,
   correctly-named directories).

Verdict: "Phase 1 is CLEARED to start." Nothing in this audit cast doubt on SHARE, the Markov
readout, or the dataset itself — the issues were entirely about the V3 *repository* not yet having
the environment V2 had, which is now fixed.

---

## 10. V3 Phase 1 — Latent State Estimation

This is V3's Layer 3, and it is explicitly **not** a re-opening of V2's scrapped Layer 3 question
(hidden supplier identity). It asks a different, previously untested question: can hidden
*operational conditions* (not identities) be estimated from observable behavior?

**Ground-truth confirmation came first, before any estimator was built** — a discipline directly
inherited from the mistake V2's Layer 3 made early on (assuming a hidden state existed without
confirming it). Six candidates were proposed, plus one found along the way:

| Candidate | Verdict | Why |
|---|---|---|
| **Supply Stress** | confirmed, ground truth exists | `own_stress()`/`stress()` in the generator |
| **Recovery Capability** | confirmed, ground truth exists (Mechanism-E variants only) | `RESILIENCE[sup_id]` |
| **Supplier Reliability** | confirmed as a state, but ultimately fails downstream | `IDIO[sup_id]` (outage state) |
| Inventory Health | **dropped** | the "hidden" quantities (`stock`, `threshold`) are already emitted to CSV *and* already used as model input features — an estimator would just read its own input back |
| Capacity Pressure | **dropped** | no time-varying latent capacity variable exists; the one hidden capacity event (`FACTORY_OUTAGE`) is one factory of five for one fixed 21-day window — too sparse to supervise |
| Logistics Stability | needs new instrumentation | a real latent effect exists (port events add +0.10 to sea-shipment stress) but it's computed inline and never stored — no readable per-entity target exists yet |
| Mitigation Level (bonus, not originally proposed) | needs new instrumentation | genuinely latent, but currently unrecoverable post-hoc because the agent loop destructively drains its own pending-action queue |

**Estimator heads were built for the three confirmed states, tested on 5 dataset seeds × 5
model-init seeds, with a reproduction-floor gate that takes the max of an init-seed floor and a
dataset-seed floor** — a design choice made specifically because Section 7.2's variance finding
(dataset-seed variance dominates) had already shown a single-axis floor isn't trustworthy on this
benchmark:

| Candidate | Best AUC (depth) | Reproduction floor | Sign-consistent | Verdict |
|---|---|---|---|---|
| Supply Stress (Variant A) | 0.6426 (h²) | 0.0390 | 5/5 seeds | **PASS** |
| Supply Stress (Variant E) | 0.6509 (h¹) | 0.0256 | 5/5 seeds | **PASS** |
| Recovery Capability (Variant E) | 0.6530 (h¹) | 0.0310 | 5/5 seeds | **PASS — strongest result in the phase** |
| Supplier Reliability (Variant A) | 0.4870 (best depth) | 0.0439 | not sign-consistent | **FAIL — below chance** |
| Supplier Reliability (Variant E) | 0.5131 (best depth) | 0.1017 | not sign-consistent | **FAIL — below chance** |

Two findings inside this phase are worth flagging on their own:

- **Supplier Reliability's single-seed pilot had looked promising** (AUC 0.5444 at seed 42 alone —
  "superficially reportable") and then **collapsed to 0.4763 and lost sign consistency** once all
  five dataset seeds were run — an in-phase, live demonstration of exactly the dataset-seed-variance
  risk Section 7.2 first measured. Diagnosed cause: only 1.1–1.2% positive rate, because the
  underlying outage mechanism gives just 25% of suppliers one ~25-day outage across a 21-month
  timeline, and 53–57% of outaged suppliers dispatch nothing at all during their own outage window.
- **The Markov readout's fixed depths (h¹ for shortage-analog tasks, h⁴ for impact-analog tasks) are
  not the right depths for latent-state estimation** — h⁴ is in fact the *worst*-performing depth for
  both states that passed. This is explicitly **not** treated as grounds to reopen the closed
  adaptive-depth question (Section 5.3) — the fixed depths were tuned for the three original
  prediction tasks, not for latent-state estimation, and Phase 2 was instructed to read from each
  state's own best depth rather than inherit h⁴ by default.

**Deliverable:** Phase 2 (the SCM) is cleared to start, using Supply Stress and Recovery Capability
as its estimated latent inputs — Recovery Capability restricted to Mechanism-E variants — with every
number above explicitly carried forward as a `v1`-preset (800-supplier) pilot, pending spec-scale
confirmation.

---

## 11. V3 Layer 3 continued — the six-step gate pipeline (Steps 1–6)

This is the most recent completed work (`reports/layer3_uncertainty_aware.md`), and it is where the
project currently sits. It takes the two states cleared in Phase 1 and runs them through a
six-step, explicitly-gated pipeline toward a calibrated, confidence-aware, threshold-gated output —
and, notably, **none of the three candidate states makes it all the way through.**

| State | Step 1 (sufficiency) | Step 2 (identifiability + recoverability) | Step 3 (calibration) | Step 4 (confidence) | Step 5 (threshold) | Final |
|---|---|---|---|---|---|---|
| Supply Stress (A) | PASS | Case 1 (identifiable + recoverable) | **STOP — G** | (out of scope, run anyway) | no threshold set | excluded at Step 3 |
| Supply Stress (E) | PASS | Case 1 | **STOP — G** | (out of scope, run anyway) | no threshold set | excluded at Step 3 |
| Recovery Capability (E) | **STOP — E** | Case 1 (never reached) | not run | not run | not run | excluded at Step 1 |
| Supplier Reliability (A, E) | n/a — no head built | Case 3 → classification **C** | closed | closed | closed | closed at Step 2 |

**Step 1 — sufficiency.** Tests whether the estimated latent state Z adds anything to predicting
`impact` beyond the raw supplier features X already give you. Supply Stress clears comfortably on
both variants (+0.2075 and +0.2669 above chance, against floors of 0.1804 and 0.1179). Recovery
Capability **fails** — not because the state is unrecoverable (it has the project's strongest
recoverability number, 0.6530), but because `RESILIENCE` is static per supplier while `impact`
varies per supplier-snapshot: a constant predictor can rank suppliers but can't distinguish a
supplier's own snapshots from each other, and the 24,000 "test rows" are really only ~800
independent suppliers replicated six times. Classified **STOP — E**, a measurement/label-granularity
failure, not a state-recoverability failure — and it stops Recovery Capability from proceeding
regardless of what Step 2 would have found.

**Step 2 — identifiability (via real interventions) and recoverability (from Phase 1).** This step
does something Phase 1 didn't: it isolates a genuine `do(Z)` intervention on one supplier at a time
(structure, co-parents, schedule, dispatch times all held fixed via common random numbers) and asks
whether the resulting true observational change is detectable at all — a stricter, causally-grounded
question than "can a model recover Z from features." Supply Stress and Recovery Capability both
clear this on real intervention data (identifiability margins of +0.0723 to +0.0095 above a measured
empirical null); Supplier Reliability does not (classifier AUC ~0.50 on both the strict and a
generous arm — because 53–57% of outaged suppliers ship nothing during their own outage window, the
signal a classifier would need simply isn't present in the observations, regardless of model
capacity). This **closes** Supplier Reliability's earlier open question from Phase 1 (which had
recommended "change the sampling, not the head") with a sharper diagnosis: it's not a sampling
problem, it's a genuine identifiability failure — classification **C**, "there is nothing to
calibrate," no further work justified.

One more finding here matters for how Model A's results should be read: **recoverability exceeds
identifiability on every single arm** — e.g., Supply Stress on Variant A clears identifiability by
+0.0723 but recoverability by +0.1036. The gap is explained mechanistically: a true `do(Z)`
intervention severs Z from its own causes, but the factual `own_stress` value correlates with
observable attributes and graph position that a 128-dimensional message-passing model can exploit —
so part of what looks like "recovering the hidden state" is actually reading the causes of the
state, which is legitimate for prediction but is explicitly **not** causal recovery.

**Step 3 — calibration, the current sticking point.** For the two states that reached this step
(Supply Stress on both variants), isotonic calibration works perfectly *within* a world — it removes
essentially all of the real miscalibration (raw ECE is 2.18–2.22× the binomial noise floor, so this
isn't a case of nothing to fix) and lands right at the noise floor (+0.0202 and +0.0224 improvement).
But fit on one world and evaluated on a different world — the only legitimate protocol, since fitting
within-world is leakage — the map **actively harms** the held-out world on average: ECE gets worse
in 13/20 and 12/20 of 20 leave-one-world-out folds, against member-composition floors of 0.0132 and
0.0117. Classified **STOP — G**: each dataset seed's head is miscalibrated in a different direction,
so no single map transfers — the same dataset-seed dominance pattern found in Section 7.2, now
surfacing inside the calibration map itself.

**Step 4 — per-entity confidence (formally out of scope, since nothing passed Step 3, but run
anyway** as an independent check on a different claim). Confidence, built from variance across
5 head-init seeds, was found to be actively backwards: on both variants, the *most*-confident bin
has the *lowest* AUC of any bin (0.6148 vs. 0.6335 in the least-confident bin, on Variant A) — the
opposite of the shape a working confidence signal should have. Classified **STOP — G**: the
model-init-seed variance is estimating exactly the axis Section 7.2 already showed carries the
smaller share of total variance, and here it turns out to carry essentially none of the
error-relevant signal at all.

**Step 5 — insufficient-evidence thresholding (also formally out of scope, run anyway).** Tested
whether rejecting the least-confident predictions at any coverage level (5% to 50%) produces a more
reliable accepted set. **No candidate at any coverage, on either variant, clears its own
member-composition floor** — the best case (Variant A, 10% rejection) improves Brier by 0.00096
against a floor of 0.0132, short by a factor of 14. Not framed as "a high rejection rate didn't
help" — framed more precisely as "the confidence signal cannot identify any subset, at any coverage,
that is more reliable than the whole."

**Step 6 — integrity check.** Confirms the backbone was genuinely never touched: all 75 head re-fits
from Step 1 reproduce Phase 1's per-init-seed AUCs to under 1e-9; SHA-256 checksums on 11 core files
match V2 exactly; `assert_backbone_frozen()` ran across 350 separate fits this session and never
fired once.

**Bottom line for this section, stated plainly in the report itself:** *"No state ended this prompt
with a calibrated, confidence-aware, threshold-gated output. Zero states reached Step 5 with a
frozen threshold."* The genuinely new positive result buried inside that negative: Supply Stress and
Recovery Capability are now **confirmed identifiable under a real intervention**, not just
recoverable by a model — and Supplier Reliability's open failure category from Phase 1 is now closed
with a specific, quantified mechanism rather than left as "needs more sampling."

---

## 12. Where things stand now, and what's queued next

**The project's current frontier is Step 3's calibration-transfer failure.** Earlier in this session
a follow-up build prompt was drafted (not yet run) — *World-Conditioned Calibration* — that treats
this failure as the thing to diagnose rather than patch:

- Steps 1–3 of that new plan re-verify the existing numbers above as a **reproduction check**, not
  new work.
- A genuinely new diagnostic step decomposes *why* the calibration curves differ between worlds
  (shift vs. scale vs. non-monotonic difference), and tests directly whether dataset-seed/world
  identity explains the difference — benchmarked against the 98.7%/67.9%/70.8% variance figures from
  Section 7.2 as the precedent hypothesis.
- The one genuinely new experiment splits "world-conditioned calibration" into two variants with
  different status: a **diagnostic** variant conditioned on dataset-seed identity directly (can only
  confirm the hypothesis in-sample over the 5 known worlds — not deployable, since a real new world
  has no seed ID), and a **deployable** variant conditioned on an observable summary statistic
  computed from the graph itself (aggregate stress level, disruption density, etc.) — the only
  version that could plausibly generalize to a truly unseen world.
- Any improvement from the deployable variant must clear its **own** freshly-measured reproduction
  floor before being called a fix — explicitly not inheriting Step 3's floor or the within-world
  ceiling, following the same discipline every other result in this document was held to.

Separately, this session scouted real-world (non-synthetic) supply-chain datasets as candidates for
future external validation — since everything above, without exception, is measured on the
synthetic generator. The most relevant candidates found: **SupplyGraph** (a real heterogeneous
graph from an FMCG company, closest structural match but no native delay/shortage/impact labels),
**FreshRetailNet-LT** (real, verified hourly stockout labels — the strongest real shortage ground
truth found, but not graph-structured), **DataCo Smart Supply Chain** (real, ready-made "late
delivery risk" label, but flat/tabular), and **FactSet Revere Supply Chain Relationships** (the real
firm-to-firm buyer-supplier network the academic literature actually uses, but institutional access
only and no disruption labels of its own). None is a drop-in replacement for the synthetic
generator's entity-level ground truth — this is flagged as an honest gap, not a solved problem.

---

## 13. The architecture as currently planned, with a status per layer

```
Supply Chain Graph
  │
  ├─ Layer 1  SHARE (RGCN + shared relation-blind attention)          [CLOSED / production, unchanged since V1→V2]
  │
  ├─ Layer 2  Markov Blanket readout (fixed depth: delay→h¹,          [CLOSED / production, unchanged since V1]
  │           shortage→h³, impact→h⁴, zero parameters)
  │
  ├─ Layer 3  Latent State Estimation                                  [PARTIAL — 2 of 7 candidate states cleared
  │           (Supply Stress, Recovery Capability)                     recoverability; NEITHER has a calibrated,
  │                                                                     confidence-aware, threshold-gated output yet —
  │                                                                     calibration fails to transfer across worlds
  │                                                                     (Step 3, STOP–G); confidence signal doesn't
  │                                                                     order error at all (Step 4, STOP–G).
  │                                                                     World-Conditioned Calibration queued as the
  │                                                                     next attempt to unblock this.]
  │
  ├─ Layer 4  Structural Causal Model                                  [NOT YET BUILT — planned: extract equations
  │           (generator-extracted causal equations)                   directly from generator code. Explicitly
  │                                                                     load-bearing, not optional, because the
  │                                                                     naive alternative (Layer 5 without an SCM)
  │                                                                     already failed — see below.]
  │
  ├─ Layer 5  Counterfactual Engine                                    [NOT YET BUILT as SCM-routed. The naive
  │           (SCM-routed intervention evaluation)                     version — raw graph edit + frozen forward
  │                                                                     pass, no SCM — was tested in the V2
  │                                                                     decision-support build and is a clean miss
  │                                                                     (0.478 vs chance, floor 0.324). A dedicated
  │                                                                     delta-prediction head was also tried and was
  │                                                                     worse (0.402). This negative result is why
  │                                                                     Layer 4 exists.]
  │
  └─ Prediction Heads
      ├─ delay / shortage / impact risk                                 [production, from SHARE + Markov readout]
      ├─ confidence / uncertainty                                       [built (`ml/uncertainty_ensemble.py`,
      │                                                                  `ml/uncertainty_calibrate.py`); dataset-seed
      │                                                                  variance dominates (68–99% of total) on the
      │                                                                  original 3 tasks; on Layer 3's latent states
      │                                                                  the analogous confidence signal fails
      │                                                                  outright (Step 4 above)]
      ├─ explanation                                                    [built (`ml/explain_prediction.py`,
      │                                                                  `ml/test_explanation_faithfulness.py`);
      │                                                                  faithfulness pass rate 0.166 vs chance 0.100,
      │                                                                  cleared modestly, but only measurable on 3.4%
      │                                                                  of `delay` predictions (the rest saturate)]
      └─ counterfactual outcomes                                        [pending Layer 5]
```

---

## 14. Cross-cutting lessons — the discipline that shaped every pivot above

These aren't separate findings; they're the methodology that emerged from repeated failure and now
gates every phase in the project, and they explain *why* the architecture kept changing shape rather
than just getting patched:

1. **A single-seed or single-run number is not evidence — a measured reproduction floor is
   mandatory before any delta is trusted.** This is the single most repeated lesson in the project.
   It first bit hard in Phase 1's counterfactual test (a 0.524 single-seed result that was actually
   13× smaller than the noise floor once measured properly), and it recurred verbatim in Phase 1
   latent-state estimation (Supplier Reliability's promising single-seed AUC of 0.5444 collapsing to
   below-chance once five seeds were run). Every phase since computes its floor before reporting a
   delta.
2. **Dataset-seed variance, not model-init variance, is usually the dominant source of uncertainty
   on this benchmark (68–99% of total, task-dependent).** This was measured once (Section 7.2) and
   then treated as a standing constraint everywhere downstream — it's why Phase 1 latent-state
   estimation gates on `max(init_seed_floor, dataset_seed_floor)` rather than either alone, and it's
   the direct explanation offered for why Step 3's calibration map doesn't transfer between worlds.
3. **A failure gets diagnosed to a specific, nameable cause before the project decides what to build
   next — never patched blindly.** V2's Layer 3 wasn't just abandoned when it failed; five different
   approaches were tried specifically to isolate *why*, and the diagnosis (group-mean dilution) is
   what determined the response (scrap discovery, keep hypothesis-generation). The same pattern
   repeats in Step 3's calibration failure: the STOP isn't "calibration doesn't work," it's "the map
   doesn't transfer between worlds, and here's the three-way breakdown (real miscalibration vs.
   binomial floor, within-world upper bound, cross-world transfer) that proves it."
4. **A capability that fails at its stated goal but produces a useful, honestly-scoped byproduct is
   kept in that narrower form rather than discarded entirely.** Layer 3's discovery mechanism failed,
   but its reframe into calibrated hypothesis generation (AUC 0.817 on observable causes, honestly
   near-zero confidence on the hidden-parent hypothesis it can't support) survived as a real tool.
5. **Privileged/generator-side ground truth is fine for supervision or evaluation, but is
   explicitly firewalled from ever reaching the model as an input** — enforced with denylists and
   assertion functions (`verify_no_hidden_state()`, `HIDDEN_STATE_COLUMNS`) at every point a hidden
   generator variable is used, from V2's base feature loader through V3's latent-state confirmation
   scripts.
6. **Every architectural claim distinguishes "validated on this synthetic benchmark" from "ready for
   real deployment," and never lets the first stand in for the second.** This is stated most
   explicitly in the V3 init document's discussion of the SCM layer (generator-extracted equations
   work *because* the generator's equations are known and coded — this is not evidence a real-data
   SCM would be buildable the same way), and it's the reason this session's dataset-scouting work
   (Section 12) was undertaken at all.

---

## 15. File / component inventory — what's actually implemented

| Component | File(s) | Status |
|---|---|---|
| SHARE encoder | `ml/models/rgcn_attn_encoder.py` | production, closed |
| SHARP / SHARK (rejected alternatives) | `ml/models/rgcn_relemb_encoder.py`, `ml/models/rgcn_battn_encoder.py` | kept for architecture-registry compatibility only |
| Markov Blanket readout | `ml/models/rgcn_attn_markov_encoder.py` | production, closed |
| Rejected depth-gate variants (Rung 4/5, Variants A–E) | subclasses in `ml/train.py`'s model dispatch | historical/reference only, not in production path |
| Dataset generator | `db/generate_dataset.py` | production; 10 mechanisms, 12 variants, spec scale reached |
| Determinism/validation harness | `db/run_benchmark.py`, `db/regenerate_seed.py`, `db/benchmark_eval.py` | production |
| PostgreSQL load path | `db/load_data.py`, `db/schema.sql` | production |
| Training loop | `ml/train.py` | production (AdamW, focal loss, best-val checkpoint, 100 epochs, optional early stopping) |
| Data loader | `ml/data/loader.py` | production; reproduces V1's SQL semantics over pandas, `HIDDEN_STATE_COLUMNS` denylist enforced |
| Frozen-backbone loader | `ml/ds_backbone.py` | production; `freeze()` bug found and worked around with `assert_backbone_frozen()` |
| Counterfactual (naive) | `ml/counterfactual_edit.py`, `ml/counterfactual_ground_truth.py`, `ml/run_counterfactual_phase1.py` | built, tested, result is a clean miss |
| Counterfactual (delta head) | `ml/counterfactual_delta_head.py` | built, tested, worse than naive |
| Uncertainty ensemble/calibration | `ml/uncertainty_ensemble.py`, `ml/uncertainty_calibrate.py` | built, delivered (variance decomposition + calibration both working on the original 3 tasks) |
| Explanation + faithfulness testing | `ml/explain_prediction.py`, `ml/test_explanation_faithfulness.py` | built, cleared modestly (0.166 vs. 0.100) |
| Hypothesis ranking (Layer 3 reframe) | `ml/hypothesis_ranker.py`, `ml/hypothesis_features.py`, `ml/hypothesis_labels.py` | built, working (AUC 0.817 on observable causes) |
| Latent-state ground-truth confirmation | `ml/confirm_latent_states.py` | built, run, 3 of 7 candidates confirmed |
| Latent-state estimator heads | `ml/latent_state_head.py` | built, run; Supply Stress & Recovery Capability pass, Supplier Reliability fails |
| Layer 3 gate pipeline (sufficiency/identifiability/calibration/confidence/threshold) | `ml/layer3_sufficiency.py`, `ml/identifiability_check.py`, `ml/layer3_uncertainty.py` | built, run; zero states clear the full pipeline yet |
| Structural Causal Model (Layer 4) | — | **not yet built** |
| Counterfactual Engine, SCM-routed (Layer 5) | — | **not yet built** (naive precursor exists and is known to fail — see above) |
| World-Conditioned Calibration | — | **build prompt drafted this session, not yet run** |

---

*This document was compiled by reading every source file listed at the top in full and
cross-checking numbers where they're repeated across files (e.g., the 98.7%/67.9%/70.8% variance
figures, and the Supply Stress/Recovery Capability AUCs, both of which appear consistently across
three or more separate reports). Where a source file itself flagged a documentation defect (the
false extended-seed claim, the missing `v1_archive` folder, the Variant A / RESILIENCE
incompatibility), that defect is reported here rather than silently corrected, matching this
project's own stated discipline of not smoothing over known gaps.*
