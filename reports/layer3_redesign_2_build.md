# Layer 3 Redesign 2 — Validation-First Neuro-Symbolic Build

**Date:** 2026-08-21
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Testbed:** Variant 0, `v1` preset (`sup_n=800`, 15 snapshots, 40/20/40 temporal split),
dataset seeds 42–46 × model-init seeds 0–4. Variant 0 rather than A or E **because every prior
number this build is measured against was taken on Variant 0** (`reports/decision_support_build.md`);
gating a capability against baselines from a world it was never measured in would make the
closing comparison meaningless.

**New code, seven files, no inherited file modified:** `ml/layer3_baseline.py` (Phase 0),
`ml/gate0_identifiability.py` (Phase 1), `ml/gate1_relevance.py` (Phase 2),
`ml/gate2_symbolic.py` (Phase 3), `ml/gate3_causal.py` (Phase 4),
`ml/layer4_integration.py` (Phase 5), `ml/layer5_policy.py` (Phase 6).

**Reused unmodified**, per the build's ground rule: `ml/explain_prediction.py` and
`ml/test_explanation_faithfulness.py` (Gate 1 baseline); `ml/counterfactual_edit.py`,
`ml/counterfactual_ground_truth.py`, `ml/counterfactual_delta_head.py` (Gate 3 baselines);
`ml/identifiability_check.py`'s CRN protocol and `ml/task_identifiability_gate.py`'s residual
tiers (Gate 0); `ml/scm.py` and `ml/run_scm_validation.py`'s fidelity checks (Gate 2);
`ml/uncertainty_ensemble.py`, `ml/hypothesis_ranker.py`'s isotonic/ECE machinery,
`ml/latent_state_head.py`'s head, `ml/ds_backbone.py`, `ml/evaluate.py::collect_predictions`.

**Scale caveat, carried from every prior phase.** `v1` preset, five dataset seeds. Per
`docs/14_Project_Roadmap.md` §3.5 this is a **pilot, not a benchmark finding**. Every number
below describes a synthetic generator; whether any of it transfers to a real supply chain is
untested and untestable here.

---

## Two source documents named by the prompt do not exist in this repository

`new_HADES.md` and `final_architecture_v1.md` are cited as the specification and the reference
design. Neither is present, and neither has ever been committed (checked against `git log --all
--diff-filter=A`). This is the fifth and sixth such dangling reference in as many sessions;
`reports/layer3_v2_build.md` recorded the same defect for `final_architecture_v1.md`.

**Closed rather than skipped.** Two substitutions were made and are used consistently:

- For `new_HADES.md`: the build prompt itself specifies its structure in enough detail to
  reconstruct — §2's claim-strength table, §6's four gate subsections, §7's two worked examples,
  §8's evidence-tier action table, §9's end-to-end example, §11's acceptance standard. The
  deliverable `new_HADES_architecture.md` is written against that structure and says so in its
  own opening.
- For `final_architecture_v1.md`: `results/final_architecture_v3.md` is the live architecture
  document in this repository and is what Layers 0–5 are read from.
- `reports/decision_support_build.md` — the source of every prior number the prompt cites — was
  **deleted** at commit `087d056` and was recovered from git history (`git show
  087d056:reports/decision_support_build.md`). Every "prior build" figure below is quoted from
  that recovered file.

---

## Verdict, up front

| phase | gate | stop condition | outcome |
|---|---|---|---|
| **0** | — | freeze `P(Y\|X)` and `P(Y\|X,H)` and both floors | **Delivered.** Only `impact`'s graph gain (+0.1137) clears its dual floor (0.0284). `delay`'s +0.0357 does not clear 0.0874. On `shortage` the graph is **actively harmful**: `P(Y\|X)` 0.8727 vs `P(Y\|X,H)` 0.7871. |
| **1** | Gate 0 | not identifiable → no downstream work for that task/effect | **`delay` PASSES** (residual 0.5263 vs null 0.5017, floor 0.0250, 5/5 sign-consistent). **`impact` FAILS** on all three tiers, by 0.0003 / 0.0006 / 0.0166. **`shortage` NOT COMPUTABLE** — CRN cannot be applied (measured). |
| **2** | Gate 1 | no method clears coverage + faithfulness + dual floor → `Entity-level explanation: Unsupported` | **HIT — no method passes, on either task.** Best is `group_perturb` at **0.304** (chance 0.100, 95% coverage, 5/5 sign-consistent) against a model-init floor of **0.261**. The occlusion baseline reproduces the prior build (**0.162** vs 0.166) and fails a floor of 0.281. The learned node-type gate is **below chance** and degenerate. |
| **3** | Gate 2 | unverified rule is a hypothesis, never validation | **PASSES.** 16 of 16 rules VERIFIED against the current generator, exactly 0.0 error on 111,375 supplier×snapshot comparisons across Variants 0 and K. **Two rules carry 21-line provenance drift** — caught only by the provenance check, which is the check that exists for it. |
| **4** | Gate 3 | no estimator clears chance + dual floor → `Causal validation: Unsupported` | **HIT, and stopped at step 1** — Gate 0 failed for this effect, so Gate 3 is diagnostic only. The naive baseline reproduces the prior build (**0.4614** vs 0.478; floor **0.4341** vs 0.324; ρ still flips sign with the model seed). **Diagnostic A confirmed** (verified SCM + true state scores **1.0000**), **Diagnostic B rejected** (flat across a 27× effect-size range). |
| **5–6** | — | wire in only what passed | **Delivered.** Fields 1, 2, 4 and 6 wired; fields 3 and 5 emit their `Unsupported` text. All three tasks land in the **prediction-only** policy tier, with three different reasons. `assert_distinctions` and `assert_no_causal_language` are both live and self-tested. |


### The five results worth having

1. **The prior build's `delay` explainability gate does not survive its own model-init floor.**
   0.162 here against 0.166 there — the number reproduces, the verdict does not, because the prior
   figure was measured at one model seed and the init-seed floor is **0.281**.
2. **The "96.5% saturated / 3.4% coverage" limitation is real, reproduces to 96.438%, and
   describes a population the benchmark never scores.** 99.8%+ of the saturated set are
   already-delivered shipments, and **zero of 935,348** carries a `delay` label. On the scored
   population there is no saturation at all and occlusion's coverage is **93%**.
3. **The counterfactual failure is representational, not data scarcity — measured, not surmised.**
   Gate 2's verified equations on the same interventions and the same held-out set score sign
   agreement, Spearman and magnitude ratio of exactly **1.0000**, against the frozen encoder's
   0.4614 / 0.0388 / 0.185. Diagnostic B is rejected on three probes of four.
4. **A no-information control was necessary and it changed a headline.** The SCM driven by an
   *estimated* latent scores 0.8865 sign agreement; the same SCM driven by a **constant** scores
   0.7750 and beats it on Spearman. Without that control this build would have reported a
   recovery that is not there.
5. **Group- and relation-level perturbation is a better relevance method than anything the prior
   build tried, and still fails.** 0.304 at 95% coverage, sign-consistent 5/5, citing an **edge
   set** rather than a feature — the most mechanism-shaped explanation observed anywhere in this
   project — and defeated by a 0.261 model-init floor.

---

## PHASE 0 — Freeze the baseline

**File:** `ml/layer3_baseline.py`. Full 5 dataset seeds × 5 model-init seeds = **25 cells**, all
25 backbones trained for this build (521 s each at 5-way concurrency, ≈ 3.6 CPU-hours, cached in
`out/ds_ckpt/`). `P(Y|X,H)` is SHARE + the Markov readout at its retained depths — `delay → h¹`,
`shortage → h³`, `impact → h⁴`, zero-parameter index-select, untouched. `P(Y|X)` is
`ml/latent_state_head.py`'s head, unmodified, on the entity's own feature row with no message
passing, so the two arms differ in the representation they are handed and in nothing else.

| task | depth | `P(Y\|X)` | `P(Y\|X,H)` | graph gain | init floor | dataset floor | **FLOOR** | gain > floor |
|---|---|---|---|---|---|---|---|---|
| delay | h¹ | 0.7416 | 0.7774 | **+0.0357** | 0.0140 | 0.0874 | **0.0874** | **no** |
| shortage | h³ | **0.8727** | 0.7871 | **−0.0856** | 0.0101 | 0.0231 | 0.0231 | **no** |
| impact | h⁴ | 0.8202 | 0.9339 | **+0.1137** | 0.0261 | 0.0284 | 0.0284 | **YES** |

Per-world `P(Y|X,H)` means: delay `[0.8187, 0.7313, 0.7963, 0.8029, 0.7377]`; shortage
`[0.8014, 0.7889, 0.7866, 0.7802, 0.7784]`; impact `[0.9449, 0.9447, 0.9329, 0.9303, 0.9165]`.
Within-world model-seed std / across-world std: **0.0041 / 0.0400** (delay), 0.0031 / 0.0091
(shortage), 0.0042 / 0.0118 (impact).

**Three findings, and two of them change how later phases have to be read.**

1. **`P(Y|X,H)` reproduces `reports/decision_support_build.md` §2.2's grand means almost
   exactly** — 0.7774 / 0.7871 / 0.9339 here against 0.7759 / 0.7877 / 0.9333 there, on an
   independently retrained 5×5 grid in a different repository. That agreement is the sanity
   anchor for everything that follows: the two builds are measuring the same model on the same
   world.
2. **The graph earns its place on exactly one task.** `impact` gains +0.1137 over a features-only
   head and clears its floor comfortably. `delay`'s +0.0357 is real in sign — positive on 5/5
   worlds — but sits inside a **dataset-seed floor of 0.0874**, which is itself the §2.2 finding
   restated (98.7% of delay's AUC variance is which world was generated). On `shortage` the
   features-only head is **8.6 points better** than SHARE + Markov; message passing costs
   accuracy there, and no Layer 3 component built on top of it can recover that.
3. **The saturation limitation reproduces exactly — and it is measuring a population no
   planner would ever open.** This is the single most consequential finding in the build, and it
   took two measurements that disagreed to get to it.

   `reports/decision_support_build.md` §3.2a reported **96.5% of `delay` predictions at
   p ≥ 0.999** and occlusion coverage of **3.4%** as the direct consequence. Phase 0's own
   saturation report, computed over the **scored** population (`ml/evaluate.py::collect_predictions`,
   i.e. the labelled rows every AUC in this project comes from), says **0.0% on all three tasks**.
   Both numbers are correct and they are measuring different populations:

   | population | delay nodes | p ≥ 0.999 | of those, `status_delivered` |
   |---|---|---|---|
   | every `Shipment` node in the snapshot | 969,898 | **935,348 = 96.438%** | **99.8–99.98%** |
   | the **scored** population (labelled rows) | 20,866 | **0 = 0.000%** | — |

   Per world: 97.247% / 96.302% / 96.974% / 96.132% / 95.538% over all Shipment nodes.
   **96.438% pooled, against the prior build's 96.5% — an exact reproduction.**

   **The saturated set is almost entirely already-delivered shipments.** On seed 42's last test
   snapshot the saturated group has `status_delivered` = 0.999, `days_to_eta` = **−308 days** and
   `days_since_dispatch` = +320, against +7.2 and +11.9 for the non-saturated group. The
   generator's own label walk skips any shipment whose as-of status is not `scheduled` or
   `in_transit`, so **not one saturated shipment carries a `delay` label** — measured, 0 of
   935,348 across all five worlds.

   So the model is emitting p ≈ 1.0 on hundreds of thousands of shipments that were delivered
   months ago, which is trivially correct and entirely uninformative, and the prior build's
   "occlusion can only touch 3.4% of predictions" is a statement about **that** population. On
   the population the benchmark actually scores, and that a planner would actually open, there is
   **no saturation at all** and occlusion's coverage in Gate 1 below is **93.0%**, not 3.4%.

   `impact` is saturated at the **low** end instead — 25–47% of suppliers at p ≤ 1e-4, median
   predicted probability 0.0004 — and there the scored and full populations coincide, because
   every supplier is labelled.

   | world | delay scored p≥0.999 | delay scored band | impact p≤1e-4 | impact band |
   |---|---|---|---|---|
   | 42 | 0.0% | 93.5% | 42.2% | 57.8% |
   | 43 | 0.0% | 91.9% | 25.2% | 74.8% |
   | 44 | 0.0% | 83.3% | 44.3% | 55.7% |
   | 45 | 0.0% | 98.9% | 46.6% | 53.4% |
   | 46 | 0.0% | 99.9% | 0.8% | 99.3% |

   **Consequence for Gate 1, applied literally.** The `band` population (the decision-relevant
   one) and the `high_conf` population (the top decile of *all* nodes) are reported separately and
   never averaged, because on `delay` the second is ~100% delivered shipments and is therefore not
   a decision-relevant population at all — a fact the numbers there make visible rather than hide.

---

## PHASE 1 — Gate 0: identifiability, before any relevance or causal work

**File:** `ml/gate0_identifiability.py`. Tiers A and B are `ml/task_identifiability_gate.py`'s,
imported unmodified — which in turn imports `ml/identifiability_check.py`'s `Realiser`,
`draw_table`, `_median_split_levels` and `featurise` unmodified. **Tier C is new**, and it is the
tier Gate 3 actually needs: `do(coparents[s] = {})` against the factual set, one supplier at a
time, which is the same co-parent lever `add_dual_source` / `remove_supplier` /
`substitute_supplier` move. On Variant 0 `MECHS = ()`, so `hp_coupling` and `upstream_stress` are
zero and `recv_atten = absorption = 1`, leaving `stress = own_stress + 0.35 · Σ own_stress(co-parents)`
— the co-parent graph is the only live structural lever, which is what makes Tier C the right
question.

The label rule is checked **row-for-row against the generator's own `label_rows`** before any
intervention number is trusted: **0 mismatches on every seed, both tasks, all five runs**
(11,051–11,434 delay rows, 12,000 impact rows per seed).

### Results

| task | tier | rows | Y differs | label AUC | obs AUC | **residual** | **null** | floor | clears | sign 5/5 | **IDENT** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| delay | **A** | 7,956 | 203 | 0.5137 | 0.7089 | **0.5263** | 0.5017 | 0.0250 | **YES** | yes | **YES** |
| delay | **B** | 7,956 | 203 | 0.5137 | 0.7089 | **0.5263** | 0.5017 | 0.0250 | **YES** | yes | **YES** |
| delay | C | 6,311 | 218 | 0.5184 | 0.6949 | 0.5494 | 0.5062 | 0.0719 | no | yes | **NO** |
| impact | A | 9,234 | 173 | 0.5085 | 0.6014 | 0.5252 | 0.5027 | 0.0258 | no | yes | **NO** |
| impact | B | 9,234 | 173 | 0.5085 | 0.6459 | 0.5249 | 0.5022 | 0.0252 | no | yes | **NO** |
| impact | C | 5,967 | 109 | 0.5081 | 0.5665 | 0.5119 | 0.5042 | 0.0285 | no | yes | **NO** |

Per-seed residual AUCs — delay A/B `[0.5358, 0.5336, 0.5127, 0.5287, 0.5205]`, delay C
`[0.5987, 0.5325, 0.5448, 0.5442, 0.5268]`, impact A `[0.5096, 0.5318, 0.5265, 0.5343, 0.5238]`,
impact C `[0.5056, 0.5129, 0.5129, 0.5117, 0.5165]`. The floor is
`max(init-seed spread, dataset-seed spread, null excursion)`, the same three-way maximum
`ml/identifiability_check.py::summarise` uses.

**Four readings, all measured.**

- **`delay` is identifiable and `impact` is not, by margins that are almost identical in size.**
  delay clears by 0.0263 against a floor of 0.0250 — a margin of **0.0013**. impact A misses by
  0.0006 (0.0252 against 0.0258). Neither of those is a comfortable verdict, and the report does
  not present them as one: the honest statement is that both sit at the resolution limit of a
  five-world grid, and delay lands on the passing side of it.
- **Tier C carries the largest raw signal on `delay` and still fails, and the reason is the
  floor's whole purpose.** Its residual AUC is 0.5494 — the highest number in the table, 0.043
  above chance — but its dataset-seed spread is **0.0719**, driven by seed 42's 0.5987 against
  seed 46's 0.5268. A single-world reading would have called this the strongest result in the
  build. It is the least reproducible one.
- **Tiers A and B coincide exactly for both tasks**, and that is a property of Variant 0 rather
  than a coding accident: with Mechanism E off, `RESILIENCE` is empty and `absorption()` returns
  1.0, so the "joint upstream set" degenerates to the stress perturbation alone and a global move
  and a per-supplier move produce the same per-entity result. The module records
  `degenerate_to_stress_only: true` rather than leaving the reader to infer it from equal numbers.
- **The `observable_shift` column is why the gate is the residual and not the raw effect.** It
  runs at **0.57–0.71** everywhere: the intervention moves the observables a great deal. The
  residual sits at 0.51–0.55. Gating on observable shift would have passed every arm on both
  tasks.

### `shortage` — the gate is not computable, and that is measured

`ml/task_identifiability_gate.py::shortage_feasibility`, run unmodified on Variant 0 seed 42:
the generator is executed twice, once factual and once with `own_stress` forced to a constant,
and the two shipment ledgers compared index-for-index.

```
shipments 39,184 -> 39,165  (-19)
identical identity tuple at the same ledger index: 0.2477%
shortage-event key Jaccard: 0.7139
CRN APPLICABLE: False
```

The weekly inventory walk **creates shipments inside its own loop** — a replenishment order is
issued, with a freshly drawn carrier, supplier and lead time, whenever stock crosses a trigger
that the intervened quantity itself moves. Common random numbers pin per-entity draws; they
cannot pin an entity set whose size and composition move. This reproduces
`reports/layer3_v2_build.md`'s finding on a **different variant** (there: +24 shipments, 0.13%
identity retention on Variant A), which upgrades it from a property of one testbed to a property
of the generator.

**Consequence for the gating order, applied literally.** `shortage` gets no Gate 1 and no Gate 3
work: not because it failed them, but because the precondition is not computable. Layer 4
reports that as a distinct status from "failed".

---

## PHASE 2 — Gate 1: entity-relevance diagnostics, five methods

**File:** `ml/gate1_relevance.py`. Five methods, **one shared verification harness**, so the
numbers are commensurable. `ml/explain_prediction.py::occlusion_attribution` is the baseline,
called unmodified; the null formulation is `ml/test_explanation_faithfulness.py`'s validated one
(same cited factor, other entities from the **same** population, 90th percentile — so chance is
**0.100** by construction). The two degenerate alternatives that build measured are not reused.

| # | method | what it cites | new? |
|---|---|---|---|
| 1 | `occlusion` | one input feature | reused unmodified |
| 2 | `grad_input` | one input feature, `x·∂logit/∂x` — on the **logit**, so the `p(1−p)` saturation factor is absent | new |
| 3 | `learned_gate` | one input feature, via `g^(k)(v) = σ(W_k h_v + b_k)` trained to preserve the prediction under an L1 penalty | new |
| 4 | `group_perturb` | a feature **block** or a whole incident **relation** | new |
| 5 | `operational_cf` | a co-parent **entity** — the SCM-valid lever Gate 0 Tier C tested | new |

**Scale:** 5 dataset seeds × 3 model-init seeds × 2 tasks = 30 cells, 3 evenly-spaced test
snapshots per cell (2 for `impact`), 25 targets per population, 20 null draws per distinct cited
factor. The null is computed **once per (snapshot, population, factor) and shared across all five
methods** — exact, since the null depends on the factor and the pool and not on who cited it.
**600 explanations per method per task on the band population** (the prior build reported 1,142
on one task at one model seed; this trades per-cell depth for a measurable dual floor).

### Results

| task | population | method | coverage | faith (p) | faith (logit) | init floor | dset floor | **FLOOR** | > chance | clears | 5/5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| delay | band | occlusion | 0.930 | **0.162** | 0.241 | 0.281 | 0.158 | 0.281 | +0.062 | no | NO |
| delay | band | grad_input | 0.870 | 0.162 | 0.178 | 0.108 | 0.096 | 0.108 | +0.062 | no | yes |
| delay | band | learned_gate | 1.000 | **0.037** | 0.048 | 0.075 | 0.075 | 0.075 | −0.063 | no | NO |
| delay | band | group_perturb | 0.950 | **0.304** | 0.385 | 0.261 | 0.133 | 0.261 | **+0.204** | no | yes |
| delay | high_conf | occlusion | **0.031** | 0.987 | 0.240 | 0.067 | 0.022 | 0.067 | +0.887 | (yes) | NO |
| delay | high_conf | grad_input | 0.865 | 0.033 | 0.170 | 0.350 | 0.137 | 0.350 | −0.067 | no | NO |
| delay | high_conf | learned_gate | 1.000 | 0.000 | 0.068 | 0.000 | 0.000 | 0.000 | −0.100 | no | NO |
| delay | high_conf | group_perturb | **0.025** | 0.867 | 0.400 | 0.267 | 0.200 | 0.267 | +0.767 | (yes) | NO |
| impact | band | occlusion | 0.907 | 0.231 | 0.440 | 0.330 | 0.144 | 0.330 | +0.131 | no | yes |
| impact | band | grad_input | 0.877 | 0.237 | 0.438 | 0.253 | 0.132 | 0.253 | +0.137 | no | yes |
| impact | band | learned_gate | 1.000 | 0.070 | 0.103 | 0.100 | 0.083 | 0.100 | −0.030 | no | NO |
| impact | band | group_perturb | 0.902 | 0.136 | 0.345 | 0.158 | 0.017 | 0.158 | +0.036 | no | yes |
| impact | band | operational_cf | **0.397** | 0.093 | 0.224 | 0.188 | 0.124 | 0.188 | −0.007 | no | NO |
| impact | high_conf | occlusion | 1.000 | 0.239 | 0.236 | 0.180 | 0.100 | 0.180 | +0.139 | no | yes |
| impact | high_conf | grad_input | 0.996 | 0.226 | 0.226 | 0.180 | 0.095 | 0.180 | +0.126 | no | yes |
| impact | high_conf | learned_gate | 1.000 | 0.092 | 0.100 | 0.140 | 0.053 | 0.140 | −0.008 | no | NO |
| impact | high_conf | group_perturb | 1.000 | 0.125 | 0.127 | 0.160 | 0.093 | 0.160 | +0.025 | no | NO |
| impact | high_conf | operational_cf | 0.429 | 0.280 | 0.271 | 0.558 | 0.302 | 0.558 | +0.180 | no | yes |

Chance = 0.100. `(yes)` marks the two rows that clear the floor arithmetically and are
disqualified on coverage and consistency — see below.

**Gate 1 verdict: NO METHOD PASSES, on either task.** The acceptance rule requires materially
better-than-chance faithfulness **and** useful coverage **and** dual-floor reproducibility. Nothing
satisfies all three.

### Six findings

**1. The occlusion baseline reproduces the prior build almost exactly, and then fails its floor.**
0.162 here against **0.166** in `reports/decision_support_build.md` §3.3, on an independently
retrained grid, on the same task, with the same null formulation. The prior build stopped at
"1.66× chance, above chance on 5/5 seeds, gate cleared". Adding the model-init axis changes the
verdict: per-world faithfulness is `[0.233, 0.176, 0.108, 0.217, 0.075]` and the **init-seed
floor is 0.281** — larger than the entire 0.062 excess over chance. The prior conclusion was not
wrong about the number; it was measured on one model seed, and the number is not stable across
that axis. *This is `reports/decision_support_build.md` §1.3.1's own lesson, applied to §3.3.*

**2. The most-cited factor reproduces too, and remains the least specific.**
`days_since_dispatch` is cited on **69%** of delay explanations here against **74%** there. The
cited-factor distributions are:

| task | method | modal factor | share |
|---|---|---|---|
| delay | occlusion | `days_since_dispatch` | 69% |
| delay | grad_input | `days_since_dispatch` | 49% |
| delay | learned_gate | `days_to_eta` | 32% |
| delay | group_perturb | `edges[SHIPS_FROM]` | 40% |
| impact | occlusion | `days_since_last_late` | 26% |
| impact | grad_input | `days_since_last_late` | 20% |
| impact | learned_gate | **`lead_time_days_z`** | **59%** |
| impact | group_perturb | `edges[SHIPS_FROM]` | 43% |

**3. Group- and path-level perturbation is the best method measured and still does not pass.**
0.304 on `delay` — 3× chance, the highest faithfulness of any arm on the decision-relevant
population, sign-consistent on 5/5 worlds (`[0.285, 0.280, 0.385, 0.252, 0.321]`), 95% coverage.
Its init-seed floor is **0.261** against a 0.204 excess. It is the closest any method comes, and
"closest" is not "clears". Notably, its modal citation is an **edge set** (`SHIPS_FROM`, 40%),
not a feature — occluding the link to the origin supplier moves this shipment's prediction more
than it moves a comparable shipment's. That is the most mechanism-like statement any method in
this comparison produced.

**4. The learned node-type relevance gate is the worst method, and it fails in a diagnosable
way.** It is *below chance* on every population (0.000–0.092) despite 100% coverage. The reason
is visible in its citations: on `impact` it names `lead_time_days_z` — a **static** attribute, the
same one — for **59%** of predictions. An amortised gate `σ(W_k h_v + b_k)` with an L1 penalty has
a global optimum that is a **population-level feature ranking**, and a population-level ranking
cannot be a per-prediction explanation: the specificity null is precisely "does this factor move
*this* entity more than a comparable one", and a constant answer fails it by construction. The
proposal this build was asked to test names exactly this parameterisation. On this benchmark it
is the weakest of the five, and the failure is structural rather than a tuning problem.

**5. Logit-space faithfulness is uniformly better than probability-space, and on `impact` it is
nearly double.** 0.440 vs 0.231 (occlusion), 0.438 vs 0.237 (grad_input), 0.345 vs 0.136
(group_perturb) on the band population. Reading relevance in probability space on a task whose
median prediction is 0.0004 is measuring float headroom. **None of the logit numbers clears its
floor either**, so this changes no verdict — but it does mean a future attempt should measure in
logit space by default, and it is the one methodological improvement in this phase that carried.

**6. The two rows that clear the floor are the prior build's coverage problem, relocated.**
`delay / high_conf / occlusion` scores 0.987 and `group_perturb` 0.867 — huge margins over
chance. Coverage is **3.1%** and **2.5%**: 23 and 19 computable explanations out of 750 targets.
That population is the top decile of *all* Shipment nodes, which Phase 0 established is ~100%
already-delivered shipments. So the arithmetic "pass" is measured on a handful of explanations
drawn from a population no planner would open, neither is sign-consistent across worlds, and
neither is admitted. **3.1% coverage against the prior build's 3.4% is the same phenomenon
arriving by a different route**, and it is what the coverage criterion exists to catch.

### Agreement with Gate 2's verified mechanisms

`ml/gate2_symbolic.py::scm_tier` classifies each cited **feature** as `scm_parent` (a component of
a direct parent in a verified equation), `scm_descendant` (an emitted observable produced by a
verified mechanism on the path) or `scm_unrelated`.

| task | method | SCM agreement (parent + descendant) | `scm_parent` alone |
|---|---|---|---|
| delay | occlusion | 0.329 | 0.046 |
| delay | grad_input | 0.373 | 0.093 |
| delay | learned_gate | 0.658 | 0.267 |
| impact | occlusion | 0.530 | 0.000 |
| impact | grad_input | 0.489 | 0.000 |
| impact | learned_gate | 0.252 | 0.000 |
| impact | operational_cf | **1.000** | **1.000** |

**The shape of this table is itself the finding, and it is a property of the generator rather
than of any method.** On `impact`, `scm_parent` is **0.000 for every feature-based method** —
not because the methods are citing nonsense, but because **impact has no observable causal
parent**. Its parents are `stress`, `RESILIENCE` and the hidden event calendar, none of which is
ever emitted; every supplier observable (`on_time_rate_*`, `trend_slope`, `lateness_variance`,
`days_since_last_late`) is a *descendant* of those parents. A feature-attribution method on this
benchmark can at best cite a symptom.

`operational_cf` is the one arm that cites a verified **parent** — a co-parent supplier, which is
a live edge in the SCM — and it scores 1.000 on agreement by construction. Its faithfulness is
0.093 (band) and its coverage is 0.397, because roughly one supplier in six has a co-parent at
all. **Being the only method that talks about the right kind of object did not make it faithful.**

Two limits of this table, stated rather than buried: the tier map is defined over individual
features, so `group_perturb`'s block and edge factors fall outside its domain and are reported as
0.000 where they should read *not mapped*; and `learned_gate`'s high agreement (0.658 on delay) is
an artefact of its degeneracy — it cites `carrier_on_time_rate_90d`, a genuine `scm_parent`, for
27% of delay predictions, while being below chance on faithfulness. **High SCM agreement and low
faithfulness together mean "names a real mechanism, but not this entity's".**

### Failure rule, applied

```text
Entity-level explanation: Unsupported
Reason: relevance signal did not meet coverage, faithfulness, or reproducibility criteria.
```

---

## PHASE 3 — Gate 2: SCM-derived symbolic rule verification

**File:** `ml/gate2_symbolic.py`. **Imported and verified, not re-extracted.** `ml/scm.py`'s
`SupplyChainSCM` — the standalone re-implementation the V3 Phase 1/2 work transcribed from
`db/generate_dataset.py`, which never calls into the generator, which is what makes agreement
evidence rather than tautology — and `ml/run_scm_validation.py`'s `equation_fidelity` and
`outcome_fidelity`, both unmodified.

Three independent checks, because they fail in different ways.

### 3a — Provenance against the *current* generator

Each rule carries the line range the original extraction recorded. The construct is looked up in
the generator **as it stands today**. This is the check `ml/scm.py`'s design makes necessary:
because it never calls the generator, a construct that has merely *relocated* still computes
identically and equation fidelity would pass in silence.

| rule | cited | found | drift |
|---|---|---|---|
| event_ramp | 839–856 | 841, 845, 849 | 0 |
| own_stress | 827–857 | 827 | 0 |
| coparent_bleed | 869–872 | 872 | 0 |
| hp_coupling | 812–824 | 823 | 0 |
| upstream_chain | 634–650 | 634 | 0 |
| recv_atten | 585–592 | 585 | 0 |
| stress | 860–880 | 860 | 0 |
| absorption | 536–547 | 547 | 0 |
| p_delay | 1085 | 1085 | 0 |
| port_bump | 1080–1082 | 1082 | 0 |
| factory_bump | 1078–1079 | 1079 | 0 |
| sourcing_stress | 961–971 | 961 | 0 |
| **replenish_trigger** | 1221–1225 | **1246** | **+21** |
| **replenish_qty** | 1233–1235 | **1256** | **+21** |
| impact_aggregation | 1355–1391 | 1366 | 0 |
| shortage_event | 1259–1262 | 1262 | 0 |

**The generator has moved since the extraction, and two rules' citations are now wrong by 21
lines.** Both are Mechanism E's replenishment equations. Neither is a computation error — the
equations still hold exactly — but a reader following `reports/phase2_scm.md`'s citation would
land on unrelated code, and a future extraction that trusted the line number would transcribe the
wrong thing. The prompt asks for this step precisely because *"the generator may have moved since
the equations were extracted"*; it had.

### 3b — Equation fidelity, and why Variant K is the arm that matters

| variant | seeds | comparisons/seed | max abs err `own_stress` | max abs err `stress` | exact |
|---|---|---|---|---|---|
| 0 | 42–46 | 12,000 | **0.0** | **0.0** | ✅ 5/5 |
| K | 42–46 | 9,810–10,575 | **0.0** | **0.0** | ✅ 5/5 |

**111,375 supplier×snapshot comparisons, bit-for-bit zero.** Variant 0 exercises only
`['coparent', 'factory', 'port', 'sourcing']`; on Variant 0 alone the extraction could be wrong
in four places and still score 0.0. Variant K activates
`['coparent', 'f_on', 'factory', 'hp_alpha', 'port', 'resilience', 'sourcing', 'upstream']`, and
`finalise_status` promotes a rule to `VERIFIED` **only where a world that actually exercises it
was checked** — a rule cannot be promoted by a variant that never runs it. All 16 clear on that
standard.

### 3c — Outcome fidelity

The SCM run forward to `p_delay` on real shipments, against the generator's own `delayed`
transitions (not `delivered_at > eta`, which the `J()` jitter inflates by ~0.03).

| variant | Brier | AUC vs realised late | max bucket gap | mean predicted | observed rate |
|---|---|---|---|---|---|
| 0 (5 seeds) | 0.0798–0.1112 | 0.6724–0.7382 | 0.0097–0.0298 | 0.0928–0.1470 | 0.0926–0.1460 |
| K (5 seeds) | 0.0526–0.0703 | 0.6385–0.7121 | 0.0131–0.0841 | 0.0607–0.0800 | 0.0573–0.0803 |

Mean predicted tracks the observed rate to within 0.004 on 9 of 10 worlds. The AUC is *not* 1.0
and cannot be: the generator realises a Bernoulli draw from `p_delay`, so the ceiling is the
Bayes rate, not perfect separation.

### 3d — Path validation

A candidate path is symbolically consistent only if **every** edge resolves to a rule that is
`VERIFIED` *and* live in the world being reasoned about.

| path | consistent | expected | verdict | blocked at |
|---|---|---|---|---|
| co-parent stressed → partner's stress → partner's shipment delayed → partner flagged | **True** | True | OK | — |
| hidden-factor event → own stress → shipment delayed → supplier flagged | **True** | True | OK | — |
| port event + SEA carrier → shipment delayed → supplier flagged | **True** | True | OK | — |
| **Supplier Failure → Shipment Delay → Inventory Reduction → Shortage** | **False** | False | OK | `replenish_arrival_lag` |
| Recovery Capability absorbs stress → lower delay probability → not flagged | **False** | (variant-dependent) | OK | `absorption` (verified on K, **inert on Variant 0**) |
| SHARE's attention weight on an edge indicates that edge is causal | **False** | False | OK | `attention_weight` |

**The fourth row is the prompt's own example and it is rejected, which is the point of the
gate.** "Supplier Failure → Shipment Delay → Inventory Reduction → Shortage" is a perfectly good
business narrative and three of its four edges are verified. The third is not: the generator's
inventory walk consumes a replenishment shipment's realised arrival time, but **no equation for
the arrival → stock → shortage transfer was ever extracted or verified**, so that edge has no
verified referent and the path is a hypothesis. It is labelled as one rather than dropped.

**The fifth row is the distinction between "verified" and "applicable".** `absorption` is
`VERIFIED` — it was exercised exactly on all five Variant K worlds. It is **not live on Variant
0**, where `RESILIENCE` is empty and `absorption()` returns 1.0. A rule set that reported
"verified" without "live in this world" would license a Recovery-Capability explanation in a
world that has no Recovery Capability.

**Rule set version:** generator sha256 `dc62a0051fd8689d…`, `GITC
a3f9c2e8b1d4470a9e6c5f2b8d7a1c3e5f9b2d40`, verified against variants `['0','K']` × seeds 42–46,
16 rules. Every Gate 2 claim in Layer 4 carries that version string.

**Gate 2 verdict: PASSED.** With the standing caveat carried verbatim from `ml/scm.py`'s own
docstring: these equations are exact *because* the world is synthetic and its structural
equations are source code. That licenses "validated on this benchmark via generator-extracted
equations" and nothing beyond it. In a real deployment no generator exists and the structure
would have to be elicited or discovered — a problem this project has not attempted.

---

## PHASE 4 — Gate 3: causal-effect diagnostics

**File:** `ml/gate3_causal.py`. The mandated order was followed literally; each step is reported
whether or not it changed the verdict.

### Step 1 — Identifiability

**Gate 0 Tier C for `impact` is NOT identifiable** (residual 0.5119, null 0.5042, above chance by
0.0077 against a floor of 0.0285). Per the acceptance rule, **Gate 3 therefore cannot be passed
for this effect regardless of what any estimator scores**, and the whole of Phase 4 runs as
**diagnostic only**. `ml/gate3_causal.py` reads this from `out/layer3_v3/gate0.json` and says so
in its first line of output. Everything below is a diagnosis of *why* the prior build's
counterfactual work failed, not a candidate for deployment.

### Step 2 — Ground-truth integrity against the current generator

| check | result |
|---|---|
| provenance — do the constructs `CausalWorld` transcribes still exist? | **all 5 found** (`own_stress` :827, `coparent_bleed` :872, `p_delay` :1085, `impact_walk` :1366, `delayed_transition` :1097) |
| `CausalWorld.stress` vs the generator's `stress()` | **1.11e-16** over 12,000 comparisons |
| `CausalWorld.p_delay` vs the generator's own formula | **1.11e-16** over 16,848 shipments |
| `p_impact` vs the generator's **emitted** impact labels | AUC **0.7782**, mean predicted 0.1558 vs observed rate 0.1693, gap **0.0135** |

**The residual is one ulp, and its cause is worth recording rather than rounding away.**
`ml/scm.py` scores exactly 0.0 on the same comparison (Gate 2) because it matches the generator's
summation order. `CausalWorld` stores each supplier's co-parents as a **set** — it has to, so an
intervention can add and remove members — which reorders the bleed summation, and float addition
is not associative. So the ground truth is exact to float rounding but **not bit-identical**, and
the module asserts against `1e-15` rather than against zero and reports both flags.

**Verdict: the generator-derived effect calculation is still valid.** It was correct by
construction in the prior build and it still is against the current generator.

### Step 3 — Diagnostic A vs Diagnostic B

Five arms on the same held-out intervention set. 40 interventions of each of the three kinds ×
5 worlds × 3 model seeds = **15 cells, 328–446 affected entity-snapshots each, 5,850 in total**
(the prior build: 2,695 at one model seed).

| arm | what it is |
|---|---|
| `naive` | frozen SHARE under an SCM-valid structural edit — the prior build's baseline, `ml/counterfactual_edit.py` unmodified |
| `A3_oracle` | verified SCM equations + **true** `own_stress` at dispatch time — a ceiling |
| `A3_grid` | verified SCM equations + true `own_stress` read off the **t₀ snapshot grid** — isolates time discretisation |
| `A3_est` | verified SCM equations + `own_stress` **estimated** from emitted observables (ridge, fitted leave-one-world-out) |
| `A3_const` | verified SCM equations + a **constant** `own_stress` — the control that decides whether `A3_est` is a method |

| arm | sign agreement | **floor** | > chance | clears | 5/5 | Spearman | ρ floor | magnitude ratio | false-effect rate |
|---|---|---|---|---|---|---|---|---|---|
| **naive** | **0.4614** | **0.4341** | **−0.0386** | no | NO | 0.0388 | **0.6042** | 0.185 | 0.00050 |
| A3_oracle | 1.0000 | 0.0000 | +0.5000 | yes | yes | 1.0000 | 0.0000 | 1.000 | 0.00000 |
| A3_grid | 0.9697 | 0.0245 | +0.4697 | yes | yes | 0.9141 | 0.0343 | 1.002 | 0.00001 |
| A3_est | 0.8865 | 0.0816 | +0.3865 | yes | yes | 0.8415 | 0.0683 | 0.890 | 0.00003 |
| A3_const | 0.7750 | 0.0927 | +0.2750 | yes | yes | 0.8452 | 0.0388 | 0.847 | 0.00002 |

Per-world naive sign agreement `[0.4721, 0.4187, 0.4096, 0.4593, 0.5472]`; per-cell,
`0.150 … 0.692`, with d44 spanning **0.529 / 0.150 / 0.549** across three model seeds of the
same world.

**Diagnostic A is confirmed and Diagnostic B is rejected.** The reasoning, in order.

**A — the frozen representation is the binding constraint, and the prior result reproduces.**

| quantity | prior build | this build |
|---|---|---|
| naive sign agreement | **0.478** (15 runs) | **0.4614** (15 cells) |
| model-init floor on that metric | 0.324 | **0.4341** |
| Spearman | ~0, sign flips with model seed (+0.068 / −0.31 / +0.16) | 0.0388 pooled; **per cell −0.234 … +0.455**, floor **0.6042** |
| magnitude ratio | 0.132 (≈8× too small) | 0.185 (≈5× too small) |
| false-effect rate on bystanders | 0.00052 | 0.00050 |

Four independent quantities land within noise of the prior build's, in a different repository on
an independently retrained grid. **The naive counterfactual is below chance, its floor is nine
times its distance from chance, and its rank correlation still flips sign with the model seed.**
That last one remains the more damning failure: a model that cannot rank *who* is most affected is
not estimating an effect even when it occasionally gets a direction right.

**Against that, `A3_oracle` scores exactly 1.0000.** Same interventions, same ground truth, same
held-out set — the difference is that it consumes the generator's true latent state and applies
Gate 2's verified equations instead of asking a frozen associational encoder what it thinks.
**The equations are right and the intervention is right; the representation is what fails.** That
is Diagnostic A, established by a direct comparison rather than inferred from a null result.

`A3_grid` at 0.9697 prices time discretisation at **0.03**: reading `own_stress` off the t₀
snapshot grid instead of at the shipment's dispatch instant costs almost nothing, which matters
because any deployable estimator pays that cost.

**But `A3_est` is not a method, and the control says so.** Its `own_stress` estimator has
**R² = 0.001–0.007** — the emitted observable channel carries essentially nothing about the latent
it is regressing. `A3_const`, which replaces the estimate with a single pooled constant carrying
no per-entity information at all, scores **0.7750** sign agreement, **0.8452** Spearman (higher
than `A3_est`'s 0.8415) and a calibration slope of 1.023. Nearly all of `A3_est`'s apparent
success is the **structure** of the intervention, not state estimation.

**Per intervention kind, which is where that becomes unambiguous:**

| arm | `add_dual_source` | `remove_supplier` | `substitute_supplier` |
|---|---|---|---|
| naive | 0.4434 | 0.4537 | 0.4932 |
| A3_oracle | 1.0000 | 1.0000 | 1.0000 |
| A3_grid | 1.0000 | 0.9937 | 0.9158 |
| **A3_est** | 1.0000 | 0.9967 | **0.6739** |
| **A3_const** | 1.0000 | 0.9967 | **0.3427** |

Adding a co-parent raises stress by `0.35 · own_stress(partner) ≥ 0` whatever the partner's state
is, and removing a supplier removes its bleed, so **both of those kinds have a true effect whose
sign is fixed by construction** and any estimator emitting a constant-signed delta scores 1.0 on
them. `substitute_supplier` is the only genuinely mixed-sign kind, and there the constant control
is **0.3427 — well below chance**, while `A3_est` is 0.6739. So the state estimate contributes
**+0.33 of real signal on the one kind that discriminates**, and nothing measurable to ranking or
scale. Reporting `A3_est`'s pooled 0.8865 without this table would have been the single most
misleading number available in this build.

**B — data scale is not the binding constraint. Four probes, none of which supports it.**

*B1 — sign agreement by decile of |true effect|, naive, pooled over 15 cells:*

| bin | mean \|true Δ\| | sign agreement | Spearman |
|---|---|---|---|
| 1 | 0.0032 | 0.4557 | 0.0050 |
| 2 | 0.0079 | 0.4496 | −0.0061 |
| 3 | 0.0135 | 0.4602 | 0.0136 |
| 4 | 0.0245 | 0.4542 | −0.0233 |
| 5 | **0.0853** | **0.4870** | 0.1473 |

A 27× increase in effect size buys **+0.031** of sign agreement, and the top decile is still below
chance.

*B2 — amplified interventions (the same verified lever applied k times):*

| k | mean \|true Δ\| | affected | naive sign | naive ρ | A3_est sign |
|---|---|---|---|---|---|
| 1 | 0.0198 | 64 | 0.4660 | −0.026 | 1.0000 |
| 2 | 0.0271 | 82 | 0.4515 | +0.087 | 1.0000 |
| 4 | 0.0340 | 150 | **0.4335** | +0.100 | 0.9974 |

Making the effect 1.7× larger and the affected set 2.3× bigger makes the naive estimator
**slightly worse**. Spearman rises from −0.026 to +0.100, which is directionally what
Diagnostic B predicts, but +0.100 sits inside a measured ρ floor of 0.6042.

*B4 — the maximum-effect intervention the generator can produce through the verified channel*
(couple a supplier to the world's most stressed supplier): naive **0.4521**, `A3_est` and
`A3_const` both 1.0000. The largest available effect does not move the frozen model.

*B3 — training-set size curve for the causal-consistency head (A2)*, `ml/counterfactual_delta_head.py`
reused unmodified, leave-one-world-out, 30 interventions of each kind per world:

| arm | train rows | train affected | head sign | naive (same rows) | paired Δ | head-seed spread | magnitude ratio (median) | head wins |
|---|---|---|---|---|---|---|---|---|
| 25% | 63,650 | 234 | 0.3816 | 0.4060 | −0.0244 | 0.120 | 7.3 | 5/10 |
| 50% | 127,301 | 444 | 0.4188 | 0.4060 | +0.0128 | 0.098 | 18.7 | 4/10 |
| 100% | 254,604 | **895** | **0.4558** | 0.4060 | +0.0498 | 0.039 | 10.9 | 6/10 |
| balanced | 4,479 | 895 | **0.4786** | 0.4060 | +0.0726 | 0.140 | 123.3 | 6/10 |

**This is the one probe that gives Diagnostic B partial support, and it is not enough.** Sign
agreement rises monotonically with training data — 0.382 → 0.419 → 0.456 — and nonzero-balanced
sampling adds a further +0.023 on 1/57th of the rows. But every arm is **below chance**, the
per-step increment (~0.037) is smaller than the head-seed spread at 25% and 50%, the head wins
only 4–6 of 10 paired folds against a baseline that is itself below chance, and the magnitude
ratio ranges over 7×–123× too large. Extrapolating the curve, reaching *chance* would need
roughly another doubling of affected examples, and chance is not a capability.

*Comparison with the prior build's Phase 1b.* There the head scored **0.402** and **lost 13 of 15
paired folds** to a naive baseline at 0.5085. Here the head is at 0.4558 and the naive baseline
has fallen to 0.4060, so the head now *wins* the pairing (+0.0498, 6/10 folds). **The ordering
flipped and the conclusion did not**: both arms are below chance in both builds, and the corpus
shape is nearly identical (895 nonzero rows in ~255,000 here; ~800 in ~270,000 there). A pairing
that reverses when both arms are noise is a description of the noise.

### Steps 4 and 6 — metrics and dual-floor replication

Reported in the table above. **`naive` fails on every axis**: below chance, floor 11× its
distance from chance, Spearman inside a 0.60 floor, magnitude 5× too small, never sign-consistent
across worlds. The A3 arms clear the floor arithmetically, but `A3_oracle` and `A3_grid` consume
the generator's true latent state and are **ceilings, not methods**, and `A3_est` is not
separable from `A3_const` on anything except `substitute_supplier`'s sign.

### Step 5 — Calibration, gated on step 4

Run only for arms that showed valid effect tracking, per the mandated ordering.

| arm | calibration slope | intercept | max bin gap | R² | residual sd |
|---|---|---|---|---|---|
| A3_oracle | 1.000 | −0.00000 | 0.00000 | 1.000 | 0.00000 |
| A3_grid | 0.876 | +0.00113 | 0.00889 | 0.769 | 0.02292 |
| A3_est | **0.994** | +0.00119 | 0.00590 | 0.667 | 0.02800 |
| A3_const | 1.023 | +0.00096 | 0.00632 | 0.662 | 0.02818 |

`naive` does not appear because it did not qualify — which is the ordering working. Note that
`A3_est` and `A3_const` are indistinguishable on every calibration quantity, which is the same
verdict the per-kind table reached by a different route.

### Acceptance rule, applied

An effect may appear in user-facing output only if it is **identifiable (Gate 0)**, exceeds chance
and the dual floor, reproduces across worlds, **improves on the prior naive and delta-head
baselines under paired evaluation**, is calibrated, and uses SCM-valid interventions (Gate 2).
Gate 0 fails for this effect at the first condition, and `naive` fails the second, third and
fourth independently.

```text
Causal validation: Unsupported
Causal-effect and root-cause claims withheld pending validated intervention-effect estimation.
```

---

## PHASES 5–6 — Layer 4 integration and Layer 5 decision policy

**Files:** `ml/layer4_integration.py`, `ml/layer5_policy.py`.

**What got wired in: fields 1, 2, 4 and 6. What stayed out: fields 3 and 5.**

| field | wired? | why |
|---|---|---|
| 1 prediction probability | yes | always available; raw **and** cross-world-isotonic-calibrated, both emitted |
| 2 evidence status | yes | a function of gate outcomes only, enforced by `assert_distinctions` |
| 3 entity relevance | **no** | Gate 1 failed on every method and task |
| 4 symbolic consistency | **yes** | Gate 2 passed; the field carries the pathways consistent **for this task in this world**, plus the rule-set version |
| 5 causal effect | **no** | Gate 3 failed — and for `impact` could not be reached, since Gate 0 failed for that effect |
| 6 confidence / uncertainty | yes | per-entity model-seed spread + the population-level cross-world term, labelled as the different objects they are |

### The actual output, world 42, last test snapshot, highest-risk scored entity per task

```text
entity: a09ac6d5-bf19-5036-b7c5-6e94a4dc7398   task: delay
1. Prediction probability : 0.8795 raw / 1.0000 calibrated
2. Evidence status        : Prediction + verified mechanism catalogue only
                            gate0=passed  gate1=failed  gate2=passed  gate3=failed
3. Entity relevance       : Entity-level explanation: Unsupported
                            Reason: relevance signal did not meet coverage, faithfulness, or reproducibility criteria.
4. Symbolic consistency   : {"consistent_paths": ["own_stress_to_delay", "coparent_to_delay",
                             "port_event_to_delay"], "n_verified_rules": 16,
                             "n_unverified_rules": 0, "rule_set_version": "dc62a0051fd8689d"}
5. Causal effect          : Causal validation: Unsupported
                            Causal-effect and root-cause claims withheld pending validated intervention-effect estimation.
6. Confidence / uncertainty: model-seed sd 0.04492; cross-world population sd 0.00725

entity: b3288dd6-e3cd-5ec3-a951-39986285505b   task: impact
1. Prediction probability : 0.9188 raw / 1.0000 calibrated
2. Evidence status        : Prediction only — effect not identifiable
                            gate0=failed  gate1=failed  gate2=passed  gate3=failed
...
4. Symbolic consistency   : {"consistent_paths": ["coparent_to_impact", "own_stress_to_impact",
                             "port_event_to_impact"], ...}
6. Confidence / uncertainty: model-seed sd 0.05915; cross-world population sd 0.00086

entity: 72d650a2-1cb5-5045-9692-29777fa6530c   task: shortage
1. Prediction probability : 0.7420 raw / 0.7143 calibrated
2. Evidence status        : Prediction only — effect not identifiable
                            gate0=not_computable  gate1=failed  gate2=passed  gate3=failed
4. Symbolic consistency   : {"consistent_paths": [], "n_verified_rules": 16, ...}
6. Confidence / uncertainty: model-seed sd 0.01375; cross-world population sd 0.00986
```

**Four things this output does that a simpler schema would not.**

1. **`delay` and `impact` get different evidence statuses despite both being "no explanation
   available".** delay is *"Prediction + verified mechanism catalogue only"* — its effect is
   identifiable and a verified pathway exists, so a user has somewhere to look. impact is
   *"Prediction only — effect not identifiable"*, which is a materially weaker position.
2. **`shortage`'s Gate 0 reads `not_computable`, not `failed`.** "We tested it and it did not
   hold" and "the test cannot be run on this generator" are different claims and `Field.kind`
   keeps them apart all the way to the emitted string.
3. **Field 4 is task-scoped.** `shortage`'s consistent-path list is **empty**: the only candidate
   shortage pathway is blocked at `replenish_arrival_lag`. Listing the three verified *impact*
   pathways under a shortage prediction would be precisely the "plausible business narrative"
   failure Gate 2 exists to block, and an earlier version of this module did exactly that before
   the terminal-rule filter was added.
4. **Both probabilities are always emitted.** The isotonic map's top bin is a PAVA step to 1.0, so
   the highest-risk entities read 1.000 calibrated against 0.879–0.919 raw. That is PAVA behaving
   as `ml/test_hypothesis_calibration.py` asserts it must, and a calibrated-only column would
   over-read the top of the ranking.

**A ranking decision that follows directly from Phase 0's saturation finding.** Both modules rank
within the **scored** population by default. Ranking over all `Shipment` nodes returns delivered
shipments at p ≈ 1.0 and nothing else — 96.4% of that population, 99.8%+ of it delivered.
`--all-nodes` reproduces the unscoped behaviour for anyone who wants to see it.

### Layer 5 — every task lands in the prediction-only tier

```text
gates: gate0=passed  gate1=failed  gate2=passed  gate3=failed
evidence status : Prediction + verified mechanism catalogue only
policy tier     : prediction_only

MONITOR. delay risk for a09ac6d5-… is estimated at 1.000 calibrated (0.879 raw)
(model-seed sd 0.0449). No validated explanation is available: Gate 1 (entity relevance) did
not clear; Gate 3 (causal effect) did not clear. Contingency options such as activating an
alternate source may be held ready as an operational hedge; this system offers no evidence
about what such an action would do to the risk.
```

`impact` and `shortage` produce the same tier with their own gate reasons — `impact` adds "Gate 0
(identifiability) did not clear", `shortage` adds "Gate 0 was **not computable** on this
generator".

**"Activate the alternate supplier" is exactly the sentence this phase exists to police**, and it
appears only as a contingency hedge, with an explicit disclaimer that the system has no evidence
about the action's effect. `assert_no_causal_language` is a real check, not decoration —
self-tested at the close of the build:

```text
prediction_only            REJECTED — contains causal language: 'will reduce'
relevance_plus_mechanism   REJECTED — contains causal language: 'caused by'
validated_causal_effect    ACCEPTED
```

The **relevance + mechanism** and **validated causal effect** tiers are implemented, tested and
**unreachable on this benchmark at this scale**. That is the correct outcome given the gate
results, and the code paths exist so that a future build which clears Gate 1 or Gate 3 does not
have to re-derive the policy.

---

## Comparison against `reports/decision_support_build.md`

The build prompt asks, plainly: did this build **recover** capability the prior one did not, or
**reproduce** the same result under a different name? The answer is different for the two gates,
and neither answer is the one the prompt's framing anticipated.

### Gate 1 vs the prior Phase 3 (explainability)

| quantity | prior build | this build | verdict |
|---|---|---|---|
| occlusion faithfulness, `delay` | **0.166** | **0.162** | **reproduced** |
| chance level | 0.100 | 0.100 | same null formulation |
| above chance on all 5 dataset seeds | yes | yes | reproduced |
| **model-init floor on that metric** | **not measured** | **0.281** | **new — and it reverses the verdict** |
| modal cited factor | `days_since_dispatch`, 74% | `days_since_dispatch`, 69% | reproduced |
| saturated fraction, all `Shipment` nodes | 96.5% | **96.438%** | reproduced almost exactly |
| **of the saturated set, already delivered** | not measured | **99.8–99.98%** | **new** |
| **saturated fraction, scored population** | not measured | **0.000%** (0 of 20,866) | **new** |
| occlusion coverage | **3.4%** | **93.0%** on the scored population | **the same model, a different population** |

**This build reproduced the prior number and overturned the prior conclusion, twice, in opposite
directions.**

*The gate the prior build cleared does not survive its own floor.* `reports/decision_support_build.md`
§3.3 reported "the gate is cleared, and modestly: 1.66× chance", above chance on 5/5 seeds. It was
measured at one model-init seed. Across three, per-world faithfulness runs `[0.233, 0.176, 0.108,
0.217, 0.075]` and the init-seed floor is 0.281 — larger than the whole 0.062 excess over chance.
The prior build's own §1.3.1 established exactly this failure mode for the counterfactual work and
did not apply it to §3.3. **Applying it closes the gate.**

*The limitation the prior build reported is real but was attributed to the wrong thing.* "Only
3.4% of predictions are in a range where this method can operate" reads as a statement about
occlusion. It is a statement about **which shipments were being ranked**: 96.4% of `Shipment`
nodes are historical deliveries the benchmark never scores, and the model is confidently and
correctly calling them late. On the population that carries labels there is **no saturation at
all**, occlusion is computable on 93% of predictions, and the coverage objection to occlusion
dissolves. It is replaced by a harder one — the method's faithfulness does not reproduce across
model seeds.

*And a better method was found that also fails.* `group_perturb` — block- and relation-level
perturbation, which the prior build never tried — reaches **0.304**, nearly double occlusion's, at
95% coverage and sign-consistent on 5/5 worlds, citing an **edge set** rather than a feature. Its
init floor is 0.261 against a 0.204 excess. It is the closest anything has come and it does not
clear.

### Gate 3 vs the prior Phase 1 / 1b (counterfactual)

| quantity | prior build | this build | verdict |
|---|---|---|---|
| naive sign agreement | **0.478** (15 runs) | **0.4614** (15 cells) | **reproduced** |
| model-init floor on that metric | 0.324 | **0.4341** | reproduced and larger |
| Spearman | ~0, **sign flips with model seed** | 0.0388 pooled; per cell −0.234 … +0.455 | **reproduced** |
| magnitude ratio | 0.132 | 0.185 | reproduced (still ~5× too small) |
| false-effect rate | 0.00052 | 0.00050 | **reproduced to two significant figures** |
| delta head (A2) sign agreement | 0.402 | 0.4558 | both **below chance** |
| head vs naive, paired | head loses 13/15 | head wins 6/10 | **ordering flipped, conclusion unchanged** |
| head magnitude ratio | median 21.7, 9 orders of magnitude | median 7.3–123 | reproduced |
| **effect-size sensitivity (Diagnostic B)** | not measured | **flat across a 27× range** | **new — and it rejects B** |
| **verified-SCM ceiling (Diagnostic A)** | not measured | **1.0000** | **new — and it confirms A** |

**This build reproduced the prior result to two significant figures on four independent
quantities and then explained it.** The prior build could say only that the naive counterfactual
failed and that supplying an interventional training signal did not fix it. It could not
distinguish "the representation is wrong" from "there is not enough signal in the world".

Running Gate 2's verified equations on the same interventions and the same held-out set gives
sign agreement **1.0000**, Spearman **1.0000**, magnitude ratio **1.000**. The equations are
right, the ground truth is right, the interventions are right. **What fails is the frozen
associational representation**, and the comparison that shows it is direct rather than inferential
— that is Diagnostic A, and it is this build's principal addition to the prior record.

Diagnostic B is rejected on three of four probes: sign agreement is flat (0.450–0.487) across a
27× range of true effect size, gets *worse* under 1.7× amplification, and does not move on the
largest effect the generator can produce. Only the training-set-size curve supports it (0.382 →
0.419 → 0.456 as rows quadruple), and that trend sits inside the head-seed floor and never reaches
chance. **The limitation is representational, not data-scarcity** — which is the opposite of the
conclusion the prior build's §1.5 leaned toward, and it is now measured rather than surmised.

### One caution about the arm that looked like a recovery

`A3_est` — verified equations driven by `own_stress` estimated from observables — scores 0.8865
sign agreement, 0.8415 Spearman and a calibration slope of 0.994. Taken alone that reads as a
recovered capability. It is not, and the control is what says so: `A3_const`, the same equations
driven by a **single constant**, scores 0.7750 / 0.8452 / 1.023. The estimator's own R² against
the latent it is regressing is **0.001–0.007**. On the only mixed-sign intervention kind, the
constant control is **below chance (0.343)** while `A3_est` reaches 0.674 — so there is roughly
+0.33 of real signal there and essentially none anywhere else. Without `A3_const` in the table,
this build would have reported a false recovery, and the reason it is in the table is
`reports/decision_support_build.md` §3.2b's lesson about nulls doing all the work.

---

## Provenance

**Trainings: 25**, all Variant 0 at the `v1` preset, `rgcn_attn_markov`, hidden=128, num_bases=10,
num_layers=4, dropout=0.2, 100 epochs, focal γ=2, CPU, 5 dataset seeds × 5 model-init seeds.
518–595 s each at 5-way concurrency, ≈ 3.6 CPU-hours, cached in `out/ds_ckpt/`.

**No SHARE or Markov readout retraining, and no modification.** `git status` reports **only new,
untracked files** under `ml/`; a sha256 re-check against the hashes taken at the start of the
build confirms `ml/models/rgcn_attn_encoder.py`, `ml/models/rgcn_attn_markov_encoder.py`,
`ml/models/rgcn_attn_depthgate_encoder.py`, `ml/models/heads.py`, `ml/models/depth.py`,
`ml/train.py`, `db/generate_dataset.py`, `ml/explain_prediction.py`,
`ml/test_explanation_faithfulness.py`, `ml/counterfactual_edit.py`,
`ml/counterfactual_ground_truth.py`, `ml/counterfactual_delta_head.py`,
`ml/identifiability_check.py`, `ml/uncertainty_calibrate.py`, `ml/uncertainty_ensemble.py` and
`ml/hypothesis_ranker.py` are **byte-identical**. `ml/ds_backbone.py::freeze` asserts every
backbone parameter is frozen before any phase runs.

**New code:** `ml/layer3_baseline.py`, `ml/gate0_identifiability.py`, `ml/gate1_relevance.py`,
`ml/gate2_symbolic.py`, `ml/gate3_causal.py`, `ml/layer4_integration.py`, `ml/layer5_policy.py`.

**Raw results:** `out/layer3_v3/phase0_baseline.json`, `gate0.json`, `gate1.json`, `gate2.json`,
`gate3.json`, `gate3_b3.json`, `layer4_records_{delay,impact,shortage}.json`,
`layer5_policy_{delay,impact,shortage}.json`, `obs_cache/`. All of `out/` is gitignored, so these
two documents are the durable record.

**Standing caveat, unchanged from every other result in this project.** Everything here rests on
Variant 0 at `sup_n=800` with 15 snapshots and 5 dataset seeds, on CPU-class hardware — not the
benchmark's own configuration. And every number describes a synthetic generator; whether any of it
transfers to a real supply chain is untested and untestable here.

## What it would take to extend

**Gate 0.** `delay` passes by 0.0013 and `impact` misses by 0.0006 against floors of ~0.025. Both
verdicts are at the resolution limit of a five-world grid, and the cheapest material improvement
is **more worlds**: generation is ~20 s each, and `S = 15–20` would tighten every floor in this
build. `shortage` needs a generator mode that applies an intervention without changing the
shipment entity set — Mechanism C's pre-scheduled rewire machinery is the natural template.

**Gate 1.** Two directions, in order of expected value. First, **more model-init seeds**: every
method's binding constraint is the init floor, and three seeds is a thin estimate of it — five
would tell us whether `group_perturb`'s 0.261 floor is real or an artefact of one bad seed.
Second, **relation-level explanation as a first-class object**: `group_perturb`'s modal citation
is `edges[SHIPS_FROM]`, which is the only mechanism-shaped statement any method produced, and it
was included here almost as an afterthought.

**Gate 2.** The one gap the path validator found is the missing `replenish_arrival_lag`
equation — extracting and verifying the arrival → stock → shortage transfer would give `shortage`
its first verified pathway and is a bounded piece of work in the generator's inventory walk.

**Gate 3.** The blocking limitation is **not** data scale, which this build measured. It is that
the effect has to be estimated from a representation trained associationally. The two routes that
follow from Diagnostic A are (a) a latent-state estimator good enough to drive the verified
equations — which requires an observable channel that carries `own_stress`, and this build
measured R² = 0.001–0.007 for the current one — or (b) a generator mode that emits a noisy proxy
for the latent, which would turn the question from "is it recoverable" into "how much noise can
the SCM tolerate", a question the `A3_grid` / `A3_est` / `A3_const` ladder is already built to
answer.
