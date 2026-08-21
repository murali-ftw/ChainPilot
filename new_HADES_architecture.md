# HADES — Validation-First Neuro-Symbolic Architecture (as measured)

**Status of this document.** This is the **as-measured** version of the Layer 3 neuro-symbolic
architecture: every gate section carries what was actually found on 2026-08-21, not what was
hypothesised. It is intended to be readable alone — a reader should be able to finish it knowing
exactly what Layer 3 can and cannot currently claim, without the build log. The build log is
`reports/layer3_redesign_2_build.md`.

**A source-document defect, stated once and then not repeated.** The build prompt specifies this
document as an update of `new_HADES.md`, and names `final_architecture_v1.md` as the reference
design. **Neither file exists in this repository, and neither has ever been committed** (checked
against `git log --all --diff-filter=A`). `reports/layer3_v2_build.md` recorded the same defect
for `final_architecture_v1.md` one build earlier. Two substitutions were made rather than
skipping the deliverable:

- the section structure below is reconstructed from the build prompt's own description of
  `new_HADES.md` — its purpose statement, §2's claim-strength policy, the five layers, §6's four
  gates, §7's worked examples, §8's evidence-tier action table, §9's end-to-end example and §11's
  acceptance standard;
- Layers 0–2 are taken from `results/final_architecture_v3.md`, which is the live architecture
  document in this repository.

**Scope of every number here.** Variant 0 at the `v1` preset (`sup_n=800`, 15 snapshots), five
dataset seeds (42–46) × five model-init seeds, CPU. Per `docs/14_Project_Roadmap.md` §3.5 this is
a **pilot, not a benchmark finding**. And everything describes a synthetic generator: whether any
of it transfers to a real supply chain is untested and untestable here.

---

## 1. Purpose

HADES predicts three supply-chain risks — shipment `delay`, product `shortage`, supplier
`impact` — and is meant to say something useful about *why*, and about *what to do*. The earlier
neuro-symbolic proposal offered eight node-type relevance gates, symbolic reasoning over paths,
and causal validation as **available capabilities**. This architecture replaces that with an
**admissibility process**: each class of claim is gated behind an empirical acceptance test, and a
component that fails its test is **excluded from user-facing output** rather than kept for
interpretability.

The reason is specific and is not hypothetical. Two of the three Layer 3 claim types correspond to
capabilities this project has already built and measured once, under different names, and both
failed:

| prior component | measured result | gate it maps to |
|---|---|---|
| `ml/explain_prediction.py` (occlusion attribution) | faithfulness **0.166** vs chance 0.100, computable on **3.4%** of predictions | Gate 1 — entity relevance |
| `ml/counterfactual_edit.py` + `ml/counterfactual_delta_head.py` | naive sign agreement **0.478** vs chance 0.500 (below chance); the interventionally-supervised head **0.402**, losing 13/15 paired folds | Gate 3 — causal effect |

An architecture that presents either as a feature is making a claim its own measurements refute.

---

## 2. Claim-strength policy — and the current status of every claim

The policy: **a claim may be made only at the strength its own acceptance test supports**, and the
tests are ordered, so a later gate is never evaluated for a task whose earlier gate failed.

| # | claim | acceptance test | **current status** |
|---|---|---|---|
| 0 | *this effect is identifiable from observations at all* | `P(X\|do(Z=z₁)) ≠ P(X\|do(Z=z₂))` on the residual of `Y` given `(X, X_nbr)`, against a measured null, above `max(init, dataset, null)` floor, sign-consistent 5/5 | **`delay`: Supported. `impact`: Unsupported. `shortage`: Not testable** (CRN inapplicable — measured) |
| 1 | *this entity/feature is why this prediction is elevated* | coverage, matched-null faithfulness above chance, informative in the high-confidence regime, above the dual seed floor | *(§6.2)* |
| 2 | *this pathway is a real mechanism* | every edge traceable to a generator equation verified against the **current** generator source | **Supported** — 16/16 rules verified, path validator retains known-valid and rejects known-invalid paths |
| 3 | *this action would change the outcome by X* | identifiable effect, chance + dual-floor margin, dual-world reproduction, paired improvement on the prior baselines, calibrated, SCM-valid intervention | *(§6.4)* |
| — | *how likely* (probability) | AUC + cross-world isotonic ECE | **Supported** — 0.7774 / 0.7871 / 0.9339 AUC; ECE 9×–17× better after cross-world isotonic |
| — | *how confident* (uncertainty) | two-axis seed-grid decomposition | **Supported, with a stated identifiability limit** — per-entity model-seed spread is measurable; the dataset-seed term is population-level only |

**Per task, at a glance:**

| task | Gate 0 | Gate 1 | Gate 2 | Gate 3 | evidence tier reached |
|---|---|---|---|---|---|
| `delay` | **passed** | failed | **passed** | failed | *Prediction + verified mechanism catalogue only* |
| `shortage` | **not computable** | failed | passed (no path terminates here) | failed | *Prediction only — effect not identifiable* |
| `impact` | failed | failed | **passed** | failed (blocked at Gate 0) | *Prediction only — effect not identifiable* |

**The distinction the whole architecture turns on:**

$$\text{Gate value} \neq \text{causal effect} \qquad\qquad \text{Probability} \neq \text{Evidence status} \neq \text{Confidence}$$

`ml/layer4_integration.py::assert_distinctions` enforces all three in code — it refuses to emit a
record whose causal-effect field was derived from anything but a Gate-3-passing estimator, whose
confidence field was derived from the probability, or whose evidence status was derived from
anything but gate outcomes.

---

## 3. Layer 0 — Input representation

A heterogeneous temporal graph $\mathcal{G}_{t_0} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$ over
eight node types (Supplier, Component, Product, Factory, Warehouse, Shipment, Order, Customer),
rebuilt per snapshot $t_0$ with strict as-of semantics: a node's features may only use records
whose *recorded* time is $\le t_0$, while labels use *true* event times. Nothing in this build
touches Layer 0.

**A property of this representation that bounds every explanation claim downstream, measured in
Phase 0.** The `Shipment` node set is dominated by **history**: 96.438% of shipment nodes across
the five test worlds sit at p ≥ 0.999, and **99.8–99.98% of those are already `delivered`**
(mean `days_to_eta` = −308 days). The generator's own label walk skips any shipment whose as-of
status is not `scheduled` or `in_transit`, so **not one of those 935,348 saturated nodes carries a
`delay` label** — measured, 0 of 935,348. The model is confidently and correctly stating that
shipments delivered months ago were late.

`reports/decision_support_build.md` §3.2a's "96.5% of `delay` predictions are saturated, so
occlusion is computable on only 3.4%" is a statement about **that** population. On the 20,866
scored rows there is **no saturation at all** (0.000%) and occlusion's coverage is **93%**. Both
Layer 4 and Layer 5 therefore rank within the scored population by default; ranking over all nodes
returns delivered shipments and nothing else.

**What it hands upward, and what it does not.** The emitted observables are
`supplier_temporal_features` (on-time rates at 30/90/180d, trend slope, lateness variance, days
since last late, shipment count), shipment schedule and status, product inventory ratios and BOM
counts, and static attributes. **The generator's latent state — `stress`, `RESILIENCE`, the
hidden-factor event calendar — is never emitted.** That asymmetry is the reason Gates 0 and 3
exist in the form they do: every causal parent in §5's SCM is latent, and every observable is a
*descendant* of one.

## 4. Layer 1 — SHARE, and Layer 2 — Markov Blanket readout

**SHARE** is an RGCN with basis-decomposed per-relation transforms plus **one shared,
relation-blind attention scorer** — parameter sharing to protect thin relations, attention to
recover the ability to single out a standout neighbour. It is validated best-or-tied-best across
twelve independent benchmark variants and **is not modified by this build**; `git diff` and a
sha256 re-check confirm `ml/models/rgcn_attn_encoder.py`,
`ml/models/rgcn_attn_markov_encoder.py`, `ml/models/heads.py`, `ml/models/depth.py` and
`ml/train.py` are byte-identical at the close of the build.

**Layer 2** is a deterministic zero-parameter index-select: `delay → h¹`, `shortage → h³`,
`impact → h⁴`. Eight learned alternatives were tried across two projects and none beat fixing it.

**What Layer 2 is worth, measured on a full 5×5 grid (this build's Phase 0):**

| task | depth | `P(Y\|X)` — features only | `P(Y\|X,H)` — SHARE + Markov | graph gain | dual floor | clears |
|---|---|---|---|---|---|---|
| delay | h¹ | 0.7416 | 0.7774 | +0.0357 | 0.0874 | **no** |
| shortage | h³ | **0.8727** | 0.7871 | **−0.0856** | 0.0231 | **no** |
| impact | h⁴ | 0.8202 | **0.9339** | **+0.1137** | 0.0284 | **YES** |

`P(Y|X,H)` reproduces `reports/decision_support_build.md` §2.2's grand means (0.7759 / 0.7877 /
0.9333) on an independently retrained grid, which is this build's sanity anchor. The
uncomfortable row is `shortage`: a features-only head beats the full graph model by 8.6 AUC
points, so **no Layer 3 component built on the graph representation can recover what the graph
costs on that task**.

---

## 5. The verified structural causal model

Every mechanism below was transcribed from `db/generate_dataset.py` by `ml/scm.py` — a
**standalone re-implementation that never calls the generator**, which is what makes agreement
evidence rather than tautology — and re-verified against the generator **as it stands today** by
`ml/gate2_symbolic.py`.

$$
\begin{aligned}
\mathrm{ramp}(e,t) &= \mathrm{mag}\cdot\mathrm{clamp}_{01}\!\left(\tfrac{t-s_0}{pk-s_0}\ \text{if}\ t\le pk\ \text{else}\ 1-\tfrac{t-pk}{s_1-pk}\right)\\
\mathrm{own\_stress}(s,t) &= \min\!\Big(0.95,\ (1-\mathrm{base\_rel}_s)\cdot 0.5 + \textstyle\sum \mathrm{ramp}\Big)\\
\mathrm{stress}(s,t) &= \min\!\Big(0.95,\ \mathrm{own\_stress}(s,t) + \underbrace{r_x(s)\cdot 0.35\!\!\sum_{p\in \mathrm{coparents}(s)}\!\!\mathrm{own\_stress}(p,t)}_{\text{co-parent bleed}} + r_x(s)\,\mathrm{hp}(s,t) + \mathrm{up}(s,t)\Big)\\
\mathrm{absorption}(s) &= \max\!\big(0,\ 1-\lambda_{\mathrm{res}}\cdot \mathrm{RESILIENCE}_s\big)\qquad (=1 \text{ when Mechanism E is off})\\
p_{\mathrm{delay}}(sh) &= \min\!\big(0.80,\ 0.025 + 0.38\cdot \mathrm{st}\cdot \mathrm{absorption}(s)\big)\\
\mathrm{impact}(s,t_0) &= \mathbb{1}\big[\exists\, sh \in \mathrm{InFlight}(s,t_0):\ sh \text{ logs } \texttt{delayed} \text{ in } (t_0, t_0+H]\big]
\end{aligned}
$$

with `st += 0.10` for a SEA carrier dispatched inside a `PORT_EVENT` window and `st += 0.35`
inside the `FACTORY_OUTAGE` window.

**On Variant 0 the co-parent graph is the only live structural lever.** `MECHS = ()` sets
`hp = up = 0` and `r_x = absorption = 1`, leaving
$\mathrm{stress} = \mathrm{own\_stress} + 0.35\sum_{\text{co-parents}}\mathrm{own\_stress}$.
Every intervention this architecture supports acts through it.

**Verification result (Gate 2): 16 of 16 rules VERIFIED.** Bit-for-bit 0.0 error on **111,375**
supplier×snapshot comparisons across Variants 0 and K × seeds 42–46. Variant K is the arm that
matters: Variant 0 exercises only four terms, so a fidelity pass there says nothing about the
other five; K activates all of them, and a rule is promoted only where a world that actually
exercises it was checked.

**Two rules carry provenance drift.** `replenish_trigger` and `replenish_qty` were cited at
`db/generate_dataset.py:1221–1225` and `:1233–1235`; they now live at **`:1246` and `:1256`**.
The equations still hold exactly — the drift is only detectable by the provenance check, which is
precisely why that check is separate from equation fidelity.

**The rule set is versioned** against generator sha256 `dc62a0051fd8689d…` / `GITC
a3f9c2e8b1d4470a9e6c5f2b8d7a1c3e5f9b2d40`, and every Gate 2 claim in Layer 4 carries that string.

**Standing validity caveat, carried verbatim from `ml/scm.py`'s own docstring.** These equations
are exact *because* the world is synthetic and its structural equations are source code. That
licenses "validated on this benchmark via generator-extracted equations" and **nothing beyond
it**. In a real deployment no generator exists; the structure would have to be hand-specified by
domain experts or learned by causal discovery — a substantially harder problem this project has
not attempted.

---

## 6. The four gates — actual dispositions

### 6.1 Gate 0 — Identifiability

**Question.** Does changing the thing we want to explain produce an observational change that is
distinguishable from a measured empirical null, reproducibly across independent worlds?

**Protocol.** `ml/identifiability_check.py`'s CRN protocol, reused unmodified through
`ml/task_identifiability_gate.py`: each shipment gets a fixed draw tuple
$(u_{\text{delay}}, g_{\text{lateness}}, e_{\text{early}}, j_{\text{jitter}})$ so the *only* thing
that can move an outcome is the intervention. Three tiers: **A** `do(stress)` isolated per
supplier; **B** the joint upstream set moved globally; **C** (new, and the one Gate 3 needs)
`do(coparents[s] = ∅)` against the factual set. The gate is the **residual** test — can the world
be recovered from $Y - g(X, X_{\text{nbr}})$ — not the raw label shift, and the floor is
`max(init-seed spread, dataset-seed spread, null excursion)`.

The label rule was checked **row-for-row against the generator's own `label_rows`**: 0 mismatches
on every seed, both tasks, all five runs.

| task | tier | rows | Y differs | label AUC | obs AUC | **residual** | null | floor | **disposition** |
|---|---|---|---|---|---|---|---|---|---|
| delay | A / B | 7,956 | 203 | 0.5137 | 0.7089 | **0.5263** | 0.5017 | 0.0250 | **IDENTIFIABLE** |
| delay | C | 6,311 | 218 | 0.5184 | 0.6949 | 0.5494 | 0.5062 | 0.0719 | not identifiable |
| impact | A | 9,234 | 173 | 0.5085 | 0.6014 | 0.5252 | 0.5027 | 0.0258 | not identifiable |
| impact | B | 9,234 | 173 | 0.5085 | 0.6459 | 0.5249 | 0.5022 | 0.0252 | not identifiable |
| impact | C | 5,967 | 109 | 0.5081 | 0.5665 | 0.5119 | 0.5042 | 0.0285 | not identifiable |
| shortage | — | — | — | — | — | — | — | — | **NOT TESTABLE** |

**`shortage` is not "failed", it is "not computable", and that was measured.** Forcing
`own_stress` to a constant changes the shipment **entity set** (39,184 → 39,165) because the
weekly inventory walk calls `new_shipment()` inside its own loop whenever stock crosses a trigger
the intervention itself moves. Only **0.2477%** of the ledger keeps the same identity at the same
index. Common random numbers pin per-entity draws; they cannot pin an entity set. This reproduces
`reports/layer3_v2_build.md`'s finding on a different variant, which makes it a property of the
generator rather than of one testbed.

**Three things worth stating plainly about this table.**

1. **`delay` passes by 0.0013 and `impact` misses by 0.0006.** Both sit at the resolution limit of
   a five-world grid. `delay` lands on the passing side; the architecture treats it as such
   because the rule was fixed before the numbers were read, but no one should read that margin as
   a comfortable verdict.
2. **The largest raw signal in the table fails.** `delay` Tier C's residual AUC is 0.5494, the
   highest number here, and its dataset-seed spread is 0.0719 (seed 42: 0.5987; seed 46: 0.5268).
   A single-world reading would have called this the strongest result in the build.
3. **The `observable_shift` column is why the gate is the residual.** It runs 0.57–0.71
   everywhere. Gating on it would have passed every arm on both tasks.

### 6.2 Gate 1 — Entity relevance

**Question.** Does the cited entity, feature, block or path actually drive *this* prediction —
more than it drives a comparable one?

**Protocol.** Five methods, one shared verification harness, `ml/explain_prediction.py` reused as
the occlusion baseline and `ml/test_explanation_faithfulness.py`'s validated specificity null
(same factor, other entities from the **same** population, 90th percentile — chance = **0.100**).
Two populations, never averaged: the **band** (decision-relevant) and the **high-confidence** top
decile. 5 dataset seeds × 3 model seeds × 2 tasks, 600 band explanations per method per task.

| task | pop | method | coverage | faith (p) | faith (logit) | **floor** | > chance | clears | 5/5 |
|---|---|---|---|---|---|---|---|---|---|
| delay | band | occlusion | 0.930 | 0.162 | 0.241 | 0.281 | +0.062 | no | NO |
| delay | band | grad_input | 0.870 | 0.162 | 0.178 | 0.108 | +0.062 | no | yes |
| delay | band | learned_gate | 1.000 | 0.037 | 0.048 | 0.075 | −0.063 | no | NO |
| delay | band | **group_perturb** | 0.950 | **0.304** | 0.385 | 0.261 | +0.204 | no | yes |
| delay | high_conf | occlusion | **0.031** | 0.987 | 0.240 | 0.067 | +0.887 | (yes) | NO |
| delay | high_conf | group_perturb | **0.025** | 0.867 | 0.400 | 0.267 | +0.767 | (yes) | NO |
| impact | band | occlusion | 0.907 | 0.231 | 0.440 | 0.330 | +0.131 | no | yes |
| impact | band | grad_input | 0.877 | 0.237 | 0.438 | 0.253 | +0.137 | no | yes |
| impact | band | learned_gate | 1.000 | 0.070 | 0.103 | 0.100 | −0.030 | no | NO |
| impact | band | group_perturb | 0.902 | 0.136 | 0.345 | 0.158 | +0.036 | no | yes |
| impact | band | operational_cf | 0.397 | 0.093 | 0.224 | 0.188 | −0.007 | no | NO |
| impact | high_conf | occlusion | 1.000 | 0.239 | 0.236 | 0.180 | +0.139 | no | yes |
| impact | high_conf | operational_cf | 0.429 | 0.280 | 0.271 | 0.558 | +0.180 | no | yes |

**Disposition: UNSUPPORTED, on every method and both tasks.** The acceptance rule requires
better-than-chance faithfulness **and** useful coverage **and** dual-floor reproducibility;
nothing satisfies all three.

- **The occlusion baseline reproduces (0.162 vs the prior 0.166) and then fails its own floor.**
  Per-world `[0.233, 0.176, 0.108, 0.217, 0.075]`, init-seed floor **0.281** against a 0.062
  excess. The prior build called this gate cleared at one model seed. Adding the second seed axis
  closes it.
- **Group- and path-level perturbation is the best method and does not clear.** 0.304 at 95%
  coverage, sign-consistent 5/5, against a 0.261 floor. Its modal citation is an **edge set**
  (`edges[SHIPS_FROM]`, 40%) — the most mechanism-shaped statement any method produced.
- **The learned node-type gate `σ(W_k h_v + b_k)` is the worst, and fails structurally.** Below
  chance everywhere despite 100% coverage, because an amortised gate with an L1 penalty optimises
  to a **population-level feature ranking** — it names `lead_time_days_z` for **59%** of `impact`
  predictions — and a constant answer cannot pass a specificity null by construction.
- **Logit-space faithfulness is uniformly better and sometimes nearly double** (0.440 vs 0.231 on
  `impact` occlusion). It changes no verdict, but a future attempt should measure there by
  default.
- **The two rows that clear arithmetically are the prior build's coverage problem, relocated.**
  3.1% and 2.5% coverage — 23 and 19 explanations from 750 targets, drawn from a population that
  is ~100% already-delivered shipments. Neither is sign-consistent; neither is admitted.

**Agreement with Gate 2's verified mechanisms is a finding about the generator, not the methods.**
`scm_parent` is **0.000 for every feature-based method on `impact`**, because impact has **no
observable causal parent**: `stress`, `RESILIENCE` and the event calendar are never emitted, and
every supplier observable is a *descendant*. `operational_cf` — which cites a co-parent entity, a
live SCM edge — scores 1.000 agreement and 0.093 faithfulness. **Talking about the right kind of
object did not make it faithful.**

```text
Entity-level explanation: Unsupported
Reason: relevance signal did not meet coverage, faithfulness, or reproducibility criteria.
```

### 6.3 Gate 2 — Symbolic mechanism

**Disposition: PASSED.** Full detail in §5 and in `reports/layer3_redesign_2_build.md` Phase 3.
Summary: 16/16 rules verified exactly on 111,375 comparisons across two variants and ten worlds;
two rules carry 21-line provenance drift; the path validator retains the three known-valid paths
and rejects all three known-invalid ones, including the prompt's own "Supplier Failure → Shipment
Delay → Inventory Reduction → Shortage", which is blocked at an edge (`replenish_arrival_lag`)
that has **no extracted generator equation** — three quarters of a plausible business narrative
with one unverifiable link is still a hypothesis.

**One distinction this gate makes that a simpler one would not: `VERIFIED` is not `live`.**
`absorption` is verified exactly on all five Variant K worlds and is **inert on Variant 0**, where
`RESILIENCE` is empty. A rule set reporting only "verified" would license a Recovery-Capability
explanation in a world that has no Recovery Capability. The validator requires both.

### 6.4 Gate 3 — Causal effect

**Step 1 stops this gate before any estimator is scored.** Gate 0 Tier C for `impact` is **not
identifiable** (residual 0.5119, null 0.5042, floor 0.0285), so **Gate 3 cannot be passed for
this effect whatever an estimator scores**. Everything below is therefore a **diagnosis**, and it
is reported in full because the prompt requires the failure source to be located, not merely
recorded.

**Step 2 — ground-truth integrity.** All five transcribed constructs still exist in the current
generator. `CausalWorld.stress` and `.p_delay` agree with the generator's own functions to
**1.11e-16** over 12,000 and 16,848 comparisons — exact to float rounding, not bit-identical,
because `CausalWorld` stores co-parents as a **set** so an intervention can edit them, which
reorders a non-associative float sum. `p_impact` scores AUC **0.7782** against the generator's
emitted impact labels with a calibration gap of 0.0135. **The ground truth is still valid.**

**Steps 3–4 — five arms, same interventions, same held-out set.** 5 worlds × 3 model seeds,
5,850 affected entity-snapshots.

| arm | sign | **floor** | > chance | clears | 5/5 | Spearman | ρ floor | magnitude | false-effect |
|---|---|---|---|---|---|---|---|---|---|
| **naive** (frozen SHARE) | **0.4614** | **0.4341** | −0.0386 | no | NO | 0.0388 | 0.6042 | 0.185 | 0.00050 |
| A3_oracle (SCM + true state) | 1.0000 | 0.0000 | +0.5000 | — | yes | 1.0000 | 0.0000 | 1.000 | 0.00000 |
| A3_grid (SCM + true state, t₀ grid) | 0.9697 | 0.0245 | +0.4697 | — | yes | 0.9141 | 0.0343 | 1.002 | 0.00001 |
| A3_est (SCM + estimated state) | 0.8865 | 0.0816 | +0.3865 | see below | yes | 0.8415 | 0.0683 | 0.890 | 0.00003 |
| A3_const (SCM + **constant** state) | 0.7750 | 0.0927 | +0.2750 | control | yes | 0.8452 | 0.0388 | 0.847 | 0.00002 |

**Which diagnostic explained the result: A, decisively; B, rejected on three probes of four.**

*A — representational limit, confirmed by direct comparison.* The naive arm reproduces the prior
build on four independent quantities (0.4614 vs 0.478; floor 0.4341 vs 0.324; magnitude 0.185 vs
0.132; false-effect 0.00050 vs 0.00052), and its Spearman still **flips sign with the model
seed** (−0.234 … +0.455 per cell). Against that, the same interventions run through Gate 2's
verified equations with the true latent state score **exactly 1.0000 on every metric**. The
equations are right, the ground truth is right, the interventions are right — **the frozen
associational representation is what fails**.

*B — data-scale limit, rejected.* Sign agreement is flat at 0.450–0.487 across a **27× range** of
true effect size; amplifying the lever 1.7× makes it slightly **worse** (0.466 → 0.434); the
largest effect the generator can produce moves it to 0.452. Only the training-set-size curve for
the causal-consistency head supports B (0.382 → 0.419 → 0.456 as rows quadruple, 895 nonzero
training rows at 100%), and that trend sits inside the head-seed floor and never reaches chance.

*The arm that looked like a recovery, and the control that says it is not.* `A3_est`'s
`own_stress` estimator has **R² = 0.001–0.007** against the latent it regresses. `A3_const` — the
same equations driven by a single constant with no per-entity information — matches it on
Spearman (0.8452 vs 0.8415) and calibration slope (1.023 vs 0.994). Per intervention kind:

| arm | `add_dual_source` | `remove_supplier` | `substitute_supplier` |
|---|---|---|---|
| naive | 0.4434 | 0.4537 | 0.4932 |
| A3_est | 1.0000 | 0.9967 | **0.6739** |
| A3_const | 1.0000 | 0.9967 | **0.3427** |

The first two kinds have a true effect whose **sign is fixed by construction**, so any
constant-signed estimator scores 1.0 on them. `substitute_supplier` is the only mixed-sign kind,
and there the control is **below chance**. The state estimate contributes ≈ +0.33 on that one
kind and nothing measurable to ranking or scale.

**Step 5 — calibration**, run only for arms that tracked: `A3_est` slope 0.994 / max bin gap
0.0059 / R² 0.667; `A3_const` slope 1.023 / 0.0063 / 0.662. Indistinguishable, which is the
per-kind verdict reached by a second route. `naive` did not qualify.

```text
Causal validation: Unsupported
Causal-effect and root-cause claims withheld pending validated intervention-effect estimation.
```

---

## 7. Layer 4 — the six-field output schema

Every prediction leaves the system as six fields that are **never** allowed to substitute for one
another. `ml/layer4_integration.py` gives each field a `kind` and a `derived_from` provenance
string so the separation is a runtime check rather than a convention.

| # | field | always present? | source |
|---|---|---|---|
| 1 | prediction probability | yes | SHARE + Markov forward pass, raw and cross-world-isotonic-calibrated |
| 2 | **evidence status** | yes | a categorical summary of **which gates passed**, and of nothing else |
| 3 | entity relevance | only if Gate 1 passed | the passing method's cited factor |
| 4 | symbolic consistency | only if Gate 2 passed | the consistent pathways in **this** world + the rule-set version |
| 5 | causal effect | only if Gate 3 passed | the passing estimator's Δ with its interval |
| 6 | confidence / uncertainty | yes | the seed grid — per-entity model-seed spread + the population-level cross-world term |

Fields 3–5 carry an explicit **Unsupported** message when their gate did not pass, rather than
being omitted: a missing field reads as "not applicable to this entity", and the finding is "not
established anywhere".

**`assert_distinctions` refuses to emit a record that has collapsed two fields**, with one check
per way this actually goes wrong:

1. *A gate value is not a causal effect.* Field 5 may only be derived from a Gate-3-passing
   estimator. Writing a relevance score, an attention weight or a learned gate value into field 5
   is the commonest form of this error and is rejected by provenance, not by inspection.
2. *Confidence is not the probability.* Field 6 must come from the seed grid. A `max(p, 1−p)`
   "confidence" is a restatement of field 1.
3. *Evidence status is not confidence and not probability.* Field 2 must be a function of gate
   outcomes.

**Field 2 distinguishes three failure modes that a boolean would merge**: `passed`, `failed`
("we tested it and it did not hold") and `not_run` / `not_computable` ("we did not, or could
not, test it"). On this build `shortage` is `not_computable` at Gate 0 and that is materially
different from `impact`'s `failed`.

**One honest artefact of field 1.** The cross-world isotonic map is fitted by PAVA on a held-out
world, and its top bin is a step to 1.0 — so the highest-risk entities report a calibrated
probability of exactly 1.000 while their raw probability is ~0.92. That is PAVA behaving as
specified (`ml/test_hypothesis_calibration.py` asserts the tie-merging), not a bug, but a
consumer reading only the calibrated column would over-read the top of the ranking. Both columns
are therefore always emitted.

---

## 8. Layer 5 — the evidence-tier action table

The action a decision-support system may recommend is a function of **which claims were
validated**, never of how high the probability is or how confident the model feels.
`ml/layer5_policy.py` fixes the language of each tier rather than composing it at call time, so a
stronger tier's phrasing cannot leak into a weaker one's output.

| tier | admitted when | what may be said |
|---|---|---|
| **prediction-only** | Gate 1 and Gate 2 did not both pass | monitor / contingency language only. Contingency options may be *held ready*; the system offers no evidence about what acting on them would do |
| **relevance + mechanism** | Gate 1 and Gate 2 passed, Gate 3 did not | investigate the named candidates **through their verified path**; the size of any effect is not established |
| **validated causal effect** | Gate 3 passed **for that effect** | intervention-oriented recommendation, with a stated interval and business constraints |

`assert_no_causal_language` runs a regex over the emitted string below the top tier and raises on
*will reduce / reduces / caused by / root cause / prevents / mitigates / attributable to /
responsible for*. The failure mode this guards against is a **phrasing** failure — the whole gate
architecture is defeated if the prose promises a causal effect the numbers never established —
and phrasing is easy to get wrong by accident.

**Confidence never promotes a tier.** A high-confidence prediction with no validated relevance is
still prediction-only. `reports/decision_support_build.md` §2.3 is the reason to be strict:
recalibration moved ECE by 9×–17× while model-init ensembling did not clear its own floor on any
task, so a confident-looking number here is a statement about calibration, not about evidence.

---

## 9. End-to-end, with the actual gate outcomes substituted at each step

World 42, last test snapshot, highest-risk entity in the **scored** population for each task.
This replaces the design document's "no validated evidence" / "fully validated evidence" pair of
hypothetical examples: the first is what the system actually emits, and the second is currently
unreachable.

**Step 1 — Layer 0/1/2.** The snapshot graph is encoded; each task reads its retained depth. The
`delay` head returns **0.8795** raw for shipment `a09ac6d5-…`, **1.0000** after cross-world
isotonic calibration.

**Step 2 — Gate 0.** `delay` is identifiable (residual 0.5263 vs null 0.5017, floor 0.0250,
5/5 sign-consistent). Downstream gates may be attempted for it. → **passed**

**Step 3 — Gate 1.** Five relevance methods are run. The best on this task, `group_perturb`,
reaches 0.304 faithfulness at 95% coverage — three times chance — and its init-seed floor is
0.261 against a 0.204 excess. → **failed**. Field 3 carries `Entity-level explanation:
Unsupported`.

**Step 4 — Gate 2.** The three pathways terminating at `p_delay` — `own_stress_to_delay`,
`coparent_to_delay`, `port_event_to_delay` — each resolve edge-by-edge to rules verified exactly
against generator sha256 `dc62a0051fd8689d…`. → **passed**. Field 4 carries those three path
names and the rule-set version.

Two candidate explanations that a plausibility-based system would have offered are **rejected
here**: `carrier_rate_to_delay` (the observed 90-day carrier rate is an aggregate of realised
outcomes; the generator's only carrier mechanism is `port_bump`, gated on SEA membership and a
`PORT_EVENT` window) and `days_since_dispatch_to_delay` (elapsed time is *exposure*; `eta =
dispatch + lead_time` is drawn independently of stress). **`days_since_dispatch` is the factor
occlusion cites on 69% of `delay` explanations.**

**Step 5 — Gate 3.** The naive counterfactual scores 0.4614 sign agreement against a 0.4341
floor, below chance, with a rank correlation that flips sign with the model seed. → **failed**.
Field 5 carries `Causal validation: Unsupported`.

**Step 6 — Layer 4 emits:**

```text
entity: a09ac6d5-bf19-5036-b7c5-6e94a4dc7398   task: delay
1. Prediction probability : 0.8795 raw / 1.0000 calibrated
2. Evidence status        : Prediction + verified mechanism catalogue only
                            gate0=passed  gate1=failed  gate2=passed  gate3=failed
3. Entity relevance       : Entity-level explanation: Unsupported
4. Symbolic consistency   : own_stress_to_delay, coparent_to_delay, port_event_to_delay
                            (16 verified rules, rule set dc62a0051fd8689d)
5. Causal effect          : Causal validation: Unsupported
6. Confidence / uncertainty: model-seed sd 0.04492; cross-world population sd 0.00725
```

**Step 7 — Layer 5 emits:**

```text
MONITOR. delay risk for a09ac6d5-… is estimated at 1.000 calibrated (0.879 raw)
(model-seed sd 0.0449). No validated explanation is available: Gate 1 (entity relevance) did
not clear; Gate 3 (causal effect) did not clear. Contingency options such as activating an
alternate source may be held ready as an operational hedge; this system offers no evidence
about what such an action would do to the risk.
```

**The same walk for the other two tasks** stops earlier and says so differently: `impact` reads
`gate0=failed` → *"Prediction only — effect not identifiable"*; `shortage` reads
`gate0=not_computable` and an **empty** consistent-path list, because its only candidate pathway
is blocked at `replenish_arrival_lag`, an edge with no extracted generator equation.

---

## 10. What each layer would need to move

| layer | binding constraint, measured | what would move it |
|---|---|---|
| 0 | latent state is never emitted; all observables are descendants | a generator mode emitting a noisy proxy for `own_stress` |
| 1–2 | dataset-seed variance is 68–99% of total AUC variance | more worlds (~20 s each); `S = 15–20` tightens every floor here |
| 3 / Gate 0 | `delay` passes by 0.0013, `impact` misses by 0.0006 | more worlds, same reason |
| 3 / Gate 1 | every method's binding constraint is the **init-seed floor** | more model seeds; and relation-level explanation as a first-class object, since `edges[SHIPS_FROM]` was the most mechanism-shaped citation observed |
| 4 / Gate 2 | one missing equation (`replenish_arrival_lag`) leaves `shortage` with no verified pathway | extract and verify the arrival → stock → shortage transfer from the inventory walk |
| 5 / Gate 3 | the representation, **not** data scale — measured on four probes | a latent-state estimator that actually recovers `own_stress` (current R² = 0.001–0.007) |

---

## 11. Final acceptance standard — the verdict

The standard: HADES may present a causal explanation only when the effect is **identifiable**,
the relevance signal is **faithful, covering and reproducible**, the pathway is **verified against
the generator's own equations**, and the effect estimate **exceeds chance and its reproduction
floor across independent worlds, improves on the prior baselines under paired evaluation, and is
calibrated**.

**Currently: one of those four conditions holds, and it is the one that does not depend on a
learned model.** Gate 2 holds outright — 16 of 16 mechanisms verified bit-exactly across two
variants and ten worlds, with a path validator that retains the three known-valid pathways per
task and rejects every known-invalid one, including two that this build's own relevance methods
cite most often. Gate 0 holds for `delay` alone, by 0.0013 against a floor of 0.0250, which is
the resolution limit of a five-world grid rather than a comfortable margin. Gate 1 fails on all
five methods and both tasks — the best, `group_perturb` at 0.304 against chance 0.100, is
defeated by a 0.261 model-init floor. Gate 3 cannot be reached for the effect it was built to
estimate, and the estimator it would have used is below chance with a floor eleven times its
distance from chance. So **HADES currently produces calibrated risk with an honest uncertainty
estimate and a verified mechanism catalogue, and no validated entity-level or causal
explanation.**

**What would have to change, in the order the evidence points.** The single highest-value change
is *not* a new explanation method: it is **more seeds on both axes**, because every negative
verdict in this build is a floor verdict rather than a signal verdict — 0.162 against 0.281,
0.304 against 0.261, 0.4614 against 0.4341 — and three model seeds and five worlds is a thin
estimate of both floors. After that, the evidence points at Layer 0 rather than Layer 3: this
build measured that the frozen representation, not the data volume, is what defeats causal-effect
estimation (Diagnostic A: 1.0000 with the true latent state, 0.4614 without), and that the
emitted observable channel carries essentially nothing about that latent (R² = 0.001–0.007). A
generator mode that emits a noisy proxy for `own_stress` would convert the open question from *"is
the latent recoverable at all"* — which this project has now answered negatively three times under
three different names — into *"how much estimation noise can the verified SCM tolerate"*, which
the `A3_oracle` / `A3_grid` / `A3_est` / `A3_const` ladder built here already answers on demand.

**And the standing caveat that bounds all of it.** Every equation above is exact *because* this
world is synthetic and its structural equations are source code. That licenses "validated on this
benchmark via generator-extracted equations" and nothing further. In a real deployment there is no
generator: the structure would have to be elicited from domain experts or learned by causal
discovery, and **the hardest part of this architecture — the part Gate 2 passes — is the part
that would not exist.**
