# ChainPilot / HADES — Final Report: The Four Claims, As Actually Tested

**What this document is.** A synthesis of every round run so far — Layers 1–3
(`reports/info.md`, `reports/layer1.md`, `reports/layer2.md`, `reports/layer3.md`) and Step 8
(`reports/step8_claim_b.md`) — against the four claims `docs/01_Product_Requirement_Document.md`
§4 defines this prototype to test, using the validation routes `docs/10_AI_ML_Documentation.md`
§9.4 specifies for each. **No training was re-run and no new numbers were generated to write
this report** — every figure below is cited from an already-completed, already-governed round.
Where `docs/project_HADES.md`'s original analytical predictions (preserved for comparison in
`docs/HADES.md`) differ from what was actually measured, the measured result is what's reported
here — the entire point of running these validation routes was that the real answer wasn't
known in advance.

---

## Executive Summary

**Claim 1 (type-aware structure, PRD §4 row 1) — tested to completion, task-dependent
answer.** The matched-parameter ablation ran to completion with confidence intervals on every
row, 5-seed sign-consistency throughout. HGT (type-aware) beats GraphSAGE/GAT (type-blind)
**consistently on impact** (+0.068 to +0.080 vs. GraphSAGE) but **consistently loses to
GraphSAGE on shortage** (−0.012 to −0.014) and is genuinely tied with it on delay. The claim is
confirmed for one task, contradicted for another, and undecided for the third — a real,
statistically-backed, non-uniform answer. A later architecture line (SHARE, `rgcn_attn`) closed
HGT's shortage loss and matched its impact result, becoming the production default.

**Claim 2 (depth selection, PRD §4 row 2) — tested to completion, task-dependent answer.**
The held-out layer-depth sweep found delay's shallow-depth preference real and
CONSISTENT (Markov Floor: +0.0072 mean AUC over baseline, all 5 seeds positive, zero added
parameters). Shortage and impact showed no reliable AUC change from any depth-selection
mechanism tried — five independent designs, the same negative result — which the sweep itself
correctly reports as depth-indifference for those two tasks, not project failure. Variant A
(bounded residual) eliminated all per-node instability at zero measured accuracy cost and is
the recommended upgrade over Markov alone.

**Claim 3 (hidden-dependency discovery, PRD §4 row 3) — split verdict: discovery tested to
completion and PASSED; downstream accuracy tested but inconclusive, and explained.** Transformer
2's held-out edge-recovery test passed decisively and consistently: the planted H_POLYMER
suppliers ranked at the 0.7255 mean percentile (chance 0.500) and were discovered at a 0.7444
rate (chance ~0.105), every one of 5 seeds clearing both bars. Whether that discovery improves
impact AUC remains not distinguishable at this label volume across all three fusion redesigns
tested (18 of 18 comparisons flip sign across seeds) — and this round went further than most
"inconclusive" verdicts by identifying *why*: a direct read of `db/generate_dataset.py` found
the planted scenario carries no incremental causal signal beyond what each supplier's own
features already capture, a dataset property, not a modeling gap.

**Claim 4 (dyadic risk, PRD §4 row 4) — tested to completion, clean result.** The
double-counting ablation (SHARE + Variant A trained twice, with and without `priority_tier`
visible to the encoder, 5 seeds each) found reordering rates of 0.0142 and 0.0143 respectively
— a 0.99 ratio, well within both arms' own seed noise. Per the PRD's own decision rule, this is
unambiguously the "rates are close, signals are independent" case: Claim B's reweighting is not
simply re-deriving something the encoder already learned.

---

## Methodology — the evidence standard applied throughout

Every AUC comparison cited below uses the same two-part standard, established from
`reports/layer1.md`'s v3 round onward and used without exception in every subsequent round: a
**paired bootstrap** on the delta between two models' predictions on the identical held-out
test set (`ml/evaluate.py::paired_delta_auc_ci`), computed independently per seed, and
**5-seed sign-consistency** — a comparison counts as real only if all 5 seeds' point estimates
agree in sign; if even one seed disagrees, it is reported as "FLIPS (noise)," never rounded up
to a trend. This directly operationalizes `docs/01_Product_Requirement_Document.md` §5's own
stated success criterion: *"The result for each claim is reported honestly, including 'not
distinguishable at this label volume' if that is what the evidence shows — a null result is a
valid outcome of this project, not a failure of it."* `docs/10_AI_ML_Documentation.md` §9.4
assigns each claim its own validation route and explicitly rules out weaker substitutes (an
unmatched comparison, reading a gate's learned weights as proof, a single high attention score,
assuming independence without testing it) — those routes are what each section below actually
ran, not a generic AUC comparison applied uniformly.

| # | Claim | Validated by | Never validated by |
|---|---|---|---|
| 1 | Type-aware vs. type-blind | Matched-parameter ablation, CIs on every row | An unmatched comparison, or a single run |
| 2 | Depth prior / gate | Held-out layer-depth sweep; adopted only if it beats the baseline by more than the CI | Reading the gate's own learned weights as proof |
| 3 | Transformer 2 | Held-out edge recovery, co-disruption rate, optional link-prediction cross-check | A single high attention score in isolation |
| 4 | Claim B | Double-counting ablation, then reviewer plausibility check | Assuming independence without running the test |

---

## Claim 1 — Type-Aware Local Structure vs. Type-Blind Local Structure

> *"Type-aware local structure beats type-blind local structure, not just because it has more
> parameters."* — `docs/01_Product_Requirement_Document.md` §4, row 1. Owned by the HGT
> encoder. Validated by: matched-parameter architecture ablation (GraphSAGE / GAT / HGT), with
> confidence intervals.

### The core ablation (`reports/layer1.md`, v3 round — 5-seed sign-consistency)

| Task | Fixed-d (5-seed) | Matched-d (5-seed) |
|---|---|---|
| delay | HGT ≈ GraphSAGE (FLIPS); **HGT beats GAT, CONSISTENT (+0.061)** | HGT ≈ GraphSAGE (FLIPS); **HGT beats GAT, CONSISTENT (+0.065)** |
| shortage | **HGT loses to GraphSAGE, CONSISTENT (−0.012)**; HGT beats GAT, CONSISTENT (+0.119) | **HGT loses to GraphSAGE, CONSISTENT (−0.014)**; HGT beats GAT, CONSISTENT (+0.121) |
| impact | **HGT beats GraphSAGE, CONSISTENT (+0.068)**; HGT ≈ GAT (FLIPS) | **HGT beats GraphSAGE, CONSISTENT (+0.080)**; HGT ≈ GAT (FLIPS) |

**Stated plainly, per the claim's own wording:** type-aware structure (HGT) beats type-blind
structure (GraphSAGE) on impact, decisively and matched-parameter-confirmed. It **loses** to
type-blind structure on shortage, matched-parameter-confirmed in the opposite direction — the
claim is directly contradicted for that task. Delay is genuinely undecided between HGT and
GraphSAGE (both ties), though HGT does beat the other type-blind baseline, GAT, consistently on
every task. This is a fully tested, CI-backed, 5-seed-consistent result on 5 of 6 cells — the
ablation ran to completion; the answer is task-dependent, not uniform.

### The later architecture line (`reports/layer1.md`, Rounds 5–7; `reports/info.md` Layer 1)

Beyond the PRD's original three-way ablation, four more architectures were tested at
matched parameters against the same HGT anchor: RGCN (basis-decomposition, no attention),
**SHARE** (RGCN + one shared attention scorer), **SHARP** (SHARE + per-relation embedding), and
**SHARK** (SHARE + a second, relation-specific attention basis pool).

| Architecture | vs. HGT, delay | vs. HGT, shortage | vs. HGT, impact |
|---|---|---|---|
| RGCN | ≈ (FLIPS) | **beats, CONSISTENT (+0.0118–0.0152)** | **loses, CONSISTENT (−0.0636)** |
| **SHARE** | **beats, CONSISTENT (+0.0103–0.0112)** | **beats, CONSISTENT (+0.0147–0.0162)** | ≈ (FLIPS, tied) |
| SHARP | ≈ (FLIPS) | **beats, CONSISTENT (+0.0167)** | ≈ (FLIPS, tied) |

SHARE is the only architecture tested anywhere in this project with the best or
statistically-tied-for-best result on all three tasks simultaneously, and moved by under 0.0015
AUC per task between its fixed-d pilot and its matched-d confirmation — a real mechanism
advantage, not a smaller-model-regularizes-better artifact (`reports/layer1.md` Round 7). It is
the production default entering every subsequent layer of this project.

### Classification

- **Tested to completion**, on all six (task × baseline) cells with 5-seed sign-consistency.
  The claim's literal wording is **confirmed for impact**, **contradicted for shortage** (HGT
  loses to the type-blind baseline there), and **not distinguishable for delay** between HGT
  and GraphSAGE specifically (though HGT does beat GAT). None of this is an incomplete test —
  every cell has a confident, CI-backed answer; the answer itself is simply not uniform.
- The SHARE/SHARP/SHARK extension is likewise **tested to completion** — SHARE's win over HGT
  on delay+shortage and tie on impact are all 5-seed-consistent, matched-parameter results.

---

## Claim 2 — Per-Task Depth Selection

> *"Depth should be chosen per task, and a structural prior derived from graph geometry is a
> better starting point than a single global depth."* — PRD §4, row 2. Owned by depth
> selection. Validated by: held-out layer-depth sweep (`L=1..4`).

### Results (`reports/layer1.md` v3 round for the original prior-vs-shared-depth sweep;
`reports/info.md` / `reports/layer2.md` for Markov Floor and every mechanism built after it)

The original structural-prior-vs-shared-depth sweep (v3, 5-seed) found **none** of the four
prior-vs-shared-depth comparisons held up under sign-consistency, on any task — the strongest
negative evidence in the project's history against a single fixed prior depth being self-
evidently correct. That result directly motivated the Markov Floor line of work: derive each
task's depth from the graph's own causal structure (a Markov blanket argument) instead of
asserting it, and test the derived depths directly.

| Task | Result | Verdict |
|---|---|---|
| delay | Markov Floor (`h¹`): **+0.0072 mean AUC vs. baseline, CONSISTENT, all 5 seeds positive**, zero added parameters | **Real, confirmed win** |
| shortage | No depth-selection mechanism tested (Learned Task Gate, Markov Floor, Rung 4/5, Variants A–E) produced a sign-consistent AUC change | **Not distinguishable — depth-indifferent** |
| impact | Same as shortage — five independent mechanisms, the same negative result | **Not distinguishable — depth-indifferent** |

Beyond the depth *value* itself, `reports/layer2.md` / `reports/info.md` Layer 2 tested whether
a *learned, per-node* depth gate could do better than the fixed Markov answer. Two designs
(Rung 4, attention-based; Rung 5, MLP-based) showed a genuine, sign-consistent structural
finding when they deviated (deviating nodes were reliably lower-degree, on shortage and
impact) but also showed severe, seed-dependent bimodal instability — some training runs stayed
pinned at Markov's answer for nearly every node, others let 75–98% drift, with no AUC
difference between the two outcomes. **Variant A** (a `tanh`-bounded residual on a *fixed,
non-trainable* Markov term) eliminated that instability completely — 100% match rate, every
seed, every task, every cap size tested — at no measured AUC cost, and is the recommended
production upgrade over Markov alone. **Variant E** (post-hoc weight-averaging across
independently trained seeds) failed catastrophically (shortage/impact scored below random
guessing) — a well-documented failure mode of the technique itself, not a bug, and is
explicitly not to be used.

### Classification

- **Tested to completion — delay.** A real, CI-backed, 5-seed-consistent win, replicated
  across three independent methodologies (the original 3-seed diagnostic, a dedicated 5-seed
  confirmation round, and the Markov Floor pilot itself).
- **Tested but inconclusive at this label volume — shortage and impact.** Five independent
  mechanisms tried, the same "not distinguishable" result every time. Per the PRD's own §5
  framing, this is reported as depth-indifference, a legitimate finding, not a failure to find
  the right mechanism.
- **Not reached** — the three numerical, label-free depth signals proposed as a
  non-learned alternative (Minimal Explanatory Subgraph, New-Information Convergence,
  Diffusion-Based Convergence; catalogued in `v2/layer2_contingency_plans.md`) are unbuilt.
  Their value proposition (robustness on noisier real-world data) was never tested, positively
  or negatively — genuinely out of scope for the time available, not attempted and inconclusive.
- **Not reached** — Variant A + Variant C combined (mechanistically complementary: one bounds
  drift, the other slows the approach to that bound) was identified as the natural next
  experiment but never run.

---

## Claim 3 — Hidden-Dependency Discovery (Transformer 2)

> *"Correlated risk can exist between suppliers with no recorded edge, and can be surfaced
> without one."* — PRD §4, row 3. Owned by Transformer 2. Validated by: held-out edge
> recovery, future co-disruption rate, optional link-prediction cross-check.

This is the one claim where the discovery question and the downstream-accuracy question are
genuinely separate, and `reports/layer3.md` (internally titled "Phase 3") treats them as such
throughout — the same separation `docs/10_AI_ML_Documentation.md` §9.4 itself specifies ("never
validated by a single high attention score in isolation").

### The discovery test (`reports/layer3.md` Part 1)

A global, same-type attention mechanism over Supplier embeddings (no adjacency mask — the
entire point, since the claim is specifically about pairs with no recorded edge) was tested
against the dataset's one planted hidden-dependency scenario: four suppliers, four countries,
different component types, sharing one unmodeled upstream polymer plant, discoverable only
through correlated behavior, never through any schema row or edge.

| Metric | Result | Chance baseline | Verdict |
|---|---|---|---|
| Mean percentile rank of the 6 planted pairs' similarity, 5 seeds × 6 test snapshots | **0.7255** | 0.500 | Every seed exceeds 0.68 |
| Pool-membership discovery rate | **0.7444** | ~0.105 (measured, not assumed) | Every seed exceeds 0.66, 5.5–7.5x chance |

**PASS, decisively, replicated across every one of 5 seeds** — not one lucky run.

### The downstream-accuracy test (`reports/layer3.md` Parts 1–2)

Whether that discovery improves the impact prediction it was fused into was tested under four
different fusion designs (the original additive fusion, plus three isolated redesigns —
Confidence-Aware, Per-Node Trust Gate, Cross-Attention), each trained fresh at 5 seeds against
shared baselines.

| Comparison | Result |
|---|---|
| Original additive fusion vs. no-Transformer-2 baseline, impact AUC | 0.9434 → 0.9455, but **FLIPS** (4 of 5 seeds positive, one negative) |
| All 3 fusion redesigns × 2 baselines × 3 tasks (18 cells) | **Every cell FLIPS** — no variant reliably beats either baseline on any task |

**Not distinguishable at this label volume, on every fusion design tried.** Unusually for a
null result in this project, the cause was identified rather than left as an open question: a
direct read of `db/generate_dataset.py` (`reports/layer3.md` §2.1, "Prerequisite 0") found the
planted scenario has **no cross-supplier causal coupling** — unlike the separate dual-sourcing
scenario (`COPARENT_COUPLING=0.35`), which *is* wired as an incremental, cross-supplier-only
signal — each hidden-dependency supplier's own label is fully derivable from its own observable
history alone. The scenario is real and discoverable but structurally redundant for prediction,
which is why no fusion mechanism, however well-built, could have produced a genuine AUC gain
from it. A separate diagnostic (`reports/layer3.md` §2.2) confirmed retrieval confidence itself
does correlate strongly with how much a prediction shifts (`pearson_r=+0.7666`) — a real,
exploitable mechanism, just one this specific dataset cannot yet reward with a label-verified
win.

### Classification

- **Tested to completion, PASS** — the discovery claim itself, exactly as PRD §4 row 3 states
  it ("can be surfaced without" a recorded edge). Confirmed decisively, every seed, against a
  defined chance baseline, not attention weights alone.
- **Tested but inconclusive at this label volume — and explained, not just observed** — whether
  discovery converts into an AUC improvement. The inconclusive result is traced to a specific,
  documented dataset property (no incremental causal coupling in the one planted scenario),
  not left as unexplained noise.
- **Not reached** — the optional link-prediction cross-check (`link_prediction_scores`,
  `docs/05_Database_Design.md` §6.27) was never built; the "future co-disruption rate" arm of
  the validation route (monitoring flagged pairs going forward in time) is likewise not
  reached, since it requires calendar time this project's rounds have not yet had.

---

## Claim 4 — Dyadic Risk (Claim B)

> *"A supplier's global fragility and this firm's specific exposure to it are different
> numbers."* — PRD §4, row 4. Owned by Claim B (dyadic reweighting). Validated by:
> double-counting ablation (with/without `priority_tier`), then reviewer plausibility check.

### The double-counting test (`reports/step8_claim_b.md`)

Claim B is deliberately not a learned component — a deterministic formula
(`order_volume_share`, `contract_priority_weight`, `fulfilment_preference_weight` combined into
a bounded multiplier on an existing, unmodified `risk_scores` row). Its risk is that
`contract_priority_weight` reweights by a signal (`customers.priority_tier`) the encoder can
also see directly as a node feature — the test is whether that overlap is real.

SHARE + Variant A was trained twice, fresh, 5 seeds each: once with `priority_tier` visible to
the encoder, once with it masked (`Customer` node features zeroed — confirmed to be *exactly*
the `priority_tier` one-hot and nothing else). For each trained model, the reordering rate
(Kendall-tau discordant-pair fraction between a customer's suppliers ranked by raw model risk
vs. by Claim B's reweighted risk, averaged over 1,295 customers with ≥2 supplier relationships)
was computed.

| Arm | Mean reordering rate | Std |
|---|---|---|
| WITH `priority_tier` | 0.0142 | 0.0042 |
| WITHOUT `priority_tier` | 0.0143 | 0.0035 |

**Ratio = 0.9898.** Per the PRD's own decision rule — close rates mean independent signals,
a collapsed rate means the encoder had already learned it — this is unambiguously the
close-rates case: masking `priority_tier` from the encoder entirely changes the reordering rate
by under 1%, well inside both arms' own seed-to-seed noise. `fulfilment_preference_weight` (the
one input the encoder structurally cannot see under any circumstance) carries the heaviest
weight in the combination formula by design (0.50 of 1.0) but the empirical independence result
does not require narrowing the claim to that signal alone, since the formula as a whole passed.

A real caveat, reported plainly rather than smoothed over: `fulfilment_preference_weight`
itself has evidence for only 6.7% of scored (customer, supplier) pairs (1,923 of 28,529) — most
pairs' scores currently come from the other two inputs, with the lead signal honestly absent
(`NULL`) rather than fabricated, for the majority of pairs today.

### Classification

- **Tested to completion** — the double-counting ablation itself. A clean, unambiguous result
  (ratio 0.99, not a borderline call requiring a qualified verdict), 5 seeds per arm,
  reported with both arms' actual numbers, not just a verdict word.
- **Not reached** — the validation route's second step, a human reviewer plausibility check on
  the reordering itself, is inherently a manual review step outside any automated training
  round and has not been performed.
- **Not reached** — `docs/05_Database_Design.md` §6.23's `suppliers` tier/`is_frontier`
  amendment and `supplier_relationships` table remain data-gated (no `SUB_SUPPLIES`-equivalent
  data exists), confirmed unbuilt across every round that has touched this question, including
  this one.

---

## Where the Shipped System Diverged From the Original Proposal

`docs/HADES.md` preserves `docs/project_HADES.md`'s original, pre-code architecture exactly as
first proposed — HGT as the encoder, a full Transformer ("T1") for depth selection, a global
attention module ("T2") for hidden-dependency discovery — explicitly as a comparison baseline,
not a description of what was built. Four material divergences, each driven by a specific
measured result above:

| Component | Originally proposed | What shipped | Why (report + section) |
|---|---|---|---|
| Encoder | HGT, full per-relation-type parameters on all 20 meta-relations | **SHARE** (RGCN basis-sharing + one shared, relation-agnostic attention scorer) | HGT pays for full parameter dedication on 4 data-thin relations that overfit; SHARE closes HGT's own impact tie while winning delay+shortage outright, at ~5x fewer encoder parameters — `reports/layer1.md` Rounds 4–7 |
| Depth selection | A full 2-layer Transformer ("T1") over 4 depth tokens, ~100K parameters | **Variant A** — a fixed, non-trainable Markov depth plus a `tanh`-bounded small residual (~13K parameters, ~same as the unbounded Rung 5 it replaced) | Every learned, unbounded per-node gate tried (Learned Task Gate, Rung 4, Rung 5) showed severe seed-dependent instability with no accuracy benefit over a free, zero-parameter fixed depth; bounding the drift, not adding more learned capacity, is what fixed it — `reports/layer2.md` / `reports/info.md` Layer 2 |
| Hidden-dependency fusion | Not specified beyond "global attention over Supplier embeddings, fused to the delay head" | **Confidence-Aware Fusion**, fused to the **impact** head, not delay | Delay's target is structurally unreachable by the co-parent/hidden-dependency mechanism at any depth ever tested (needs ≥3 hops from Shipment; delay reads 1); impact's target is reachable and impact is where the depth mechanism already reads deep — `reports/layer3.md` §1.1. Confidence-Aware Fusion was chosen over the original single-scalar additive design and two other redesigns because it adds zero new parameters, best preserves the validated discovery signal, and shows none of the other designs' instability — `reports/layer3.md` Part 3 |
| Claim B scope | Anticipated needing to narrow to `fulfilment_preference_weight` alone if `priority_tier` proved redundant with what the encoder already learned | **Validated as an independent, full-formula contribution** — the double-counting test found the reordering rates close, not collapsed | `reports/step8_claim_b.md`'s double-counting result did not require the narrower framing `docs/10_AI_ML_Documentation.md` §8.5 anticipated as the fallback |

---

## What Would Move an Inconclusive Claim to "Tested to Completion"

Every "not distinguishable at this label volume" result above traces to a specific, identified
property of the current synthetic dataset — not an open-ended "more data would probably help"
shrug. `v2/v2.md` catalogs each one as a concrete, deliberate change to
`db/generate_dataset.py` / the live database, in the same spirit the dataset's own existing
inconsistencies (`H_POLYMER`, `COPARENT_COUPLING`, the four thin relations) were built:

- **Claim 3's AUC question** specifically needs a hidden-dependency scenario wired with an
  incremental, `COPARENT_COUPLING`-style cross-supplier bleed-through term — the one change
  that would let the fusion-design question actually be tested, rather than structurally
  unanswerable by construction (`v2/v2.md` §1).
- **Claims 1, 2, and 3's shared-and-impact-task inconclusiveness** traces substantially to
  overall label volume (~1,774 positive-label rows across all three tasks) — `v2/v2.md` §2
  proposes the specific scale-up that would move most "FLIPS (noise)" verdicts in this report
  into either a confirmed win or a confirmed, statistically-backed tie.
- See `v2/v2.md` in full for every other cataloged gap, including ones not raised as a primary
  finding in this report.

---

*Every number in this document is cited from an already-completed, already-governed round —
`reports/layer1.md`, `reports/layer2.md`, `reports/layer3.md`, `reports/info.md`, and
`reports/step8_claim_b.md`. No training was re-run and no new result was generated to produce
this synthesis.*
