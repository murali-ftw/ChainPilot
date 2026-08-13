# 14 — Project Roadmap (V3)

> **V3 initiation.** HADES V3 is a change of *kind*, not degree: V1 and V2 both trained
> architectures to answer `P(Y|X)`; V3 adds three new layers whose job is to answer `P(Y|do(X))`.
> This roadmap is written the same way `HADES_v2/docs/14_Project_Roadmap.md` was — phases with a
> stated status, evidence cited by file, and a distinction between what is done, what is gated, and
> what is deferred. Authoritative architecture reference: `docs/HADES-v3_init.md`. Where this
> roadmap and that document disagree, the init document wins; this file sequences the work, it does
> not re-litigate the design.

---

## 0. Starting position — what V3 inherits, not what it builds first

V3 does **not** start from zero. Three things are already on disk before any V3-specific work
begins:

| Asset | Source | State |
|---|---|---|
| **Dataset** | `HADES_v2/db/` → copied into `db/` | The full V2 spec-scale sweep is already present: all 12 variants (0, A–K) at seeds 42/43, plus extended seeds (44–46) for Variants 0, B, D, J, K, plus a mid-scale set for B/D. **No regeneration needed to start.** This is "the previous dataset generated in HADES v2" — use it as-is; regenerate only if a V3 layer needs a config the existing sweep doesn't cover (see §3.1). |
| **Layer 1 — SHARE** | `HADES_v2/ml/models/rgcn_attn_encoder.py` | Validated best-or-tied-for-best on all three tasks, all twelve V2 variants, two independently-built datasets (`findings/evolution.md` §2, §6). **Inherited unchanged.** Not retrained, not modified, unless a specific V3 finding calls for it — confirm with `git diff` at every stage, same discipline as every prior session. |
| **Layer 2 — Markov Blanket readout** | `HADES_v2/ml/models/rgcn_attn_markov_encoder.py` | Zero-parameter, fixed per-task depth (delay→h¹, shortage→h³, impact→h⁴). Winner of an eight-variant, ~130-run, two-project investigation into adaptive depth; every adaptive alternative lost (`findings/evolution.md` §3). **Closed question — do not reopen.** |
| **Prototype decision-support machinery** | `HADES_v2/ml/{ds_backbone,counterfactual_edit,counterfactual_ground_truth,run_counterfactual_phase1,counterfactual_delta_head,uncertainty_ensemble,uncertainty_calibrate,explain_prediction,test_explanation_faithfulness}.py` | Built and run in one prior session (`reports/decision_support_build.md`). Not V3 architecture — V2's Layer 3 was scrapped — but the harnesses, the frozen-backbone loading pattern, and one **already-answered question critical to V3's design** (below) all carry forward. |

**What was scrapped, and why it stays scrapped.** V2's Layer 3 (hidden-*relationship* discovery)
was tested five independent ways across two datasets and failed for a diagnosed, structural reason:
the benchmark's hidden-parent mechanism dilutes its effect across a group mean, which destroys the
pairwise attribution any retrieval method needs (`findings/evolution.md` §5). V3 does not reopen
this. V3's Layer 3 estimates hidden **operational state**, a different and previously
untested question — see §3.2.

---

## 1. A question this roadmap does not need to ask, because V2 already answered it

`docs/HADES-v3_init.md` §8 posed an open question for Layer 5: "whether a naive approach (edit the
graph, re-run SHARE and the Markov readout unmodified) tracks true simulated intervention effects
at all... if it does not, the SCM layer becomes load-bearing rather than optional."

**This was tested in `reports/decision_support_build.md` Phase 1, before V3 existed as a project,
and it is a clean miss:**

| approach | sign agreement | vs. chance (0.500) | vs. measured floor |
|---|---|---|---|
| naive graph-edit + frozen forward pass | 0.478 (15 runs) | at chance | floor is 0.324 (max spread across model-init seeds) — the apparent edge is smaller than the noise |
| interventionally-supervised delta head (Phase 1b) | 0.402 | **below** chance | loses 13 of 15 paired folds to the naive approach |

Both routes were tried — an unmodified forward pass on an edited graph, and a small head trained
**directly** on generator-produced true deltas — and both failed on `impact`, Variant 0, at the
cheap `v1` (800-supplier) fixture.

**Consequence for this roadmap: Layer 4 (the Structural Causal Model) is load-bearing, not an
optional enhancement, from Phase 2 onward.** No V3 phase should re-attempt naive graph-editing or
embedding-difference counterfactuals as a first approach — that ground is already covered and lost.
Any future re-test of the naive approach belongs in §5 (extending the negative result to more
variants/scale), not in the critical path.

---

## 2. Build phases — status

| Phase | Scope | Status |
|---|---|---|
| **0** | Inheritance audit — confirm SHARE/Markov port cleanly, confirm dataset integrity, freeze discipline in place | 🟡 **partial** — assets exist (§0); formal audit not yet run in the V3 repo |
| **1** | Latent State Estimation (Layer 3) — confirm ground truth per candidate state, build estimator heads | ⬜ not started |
| **2** | Structural Causal Model (Layer 4) — extract equations from the generator, validate against real generator output | ⬜ not started |
| **3** | Counterfactual Engine (Layer 5) — SCM-routed intervention evaluation, extended intervention/task coverage | ⬜ not started |
| **4** | Prediction heads — confidence, explanation, counterfactual outcomes wired to a shared representation | ⬜ not started |
| **5** | Spec-scale evaluation — re-run Phases 1–4's findings at `sup_n=4000` (all prior decision-support numbers are at the 800-supplier fixture) | ⬜ not started |
| **6** | Reporting and documentation — V3 doc suite, comparison table vs. V1/V2, deployment-readiness caveat | ⬜ not started |

No phase is complete. Phase 0's assets exist because they were produced by V2 sessions and copied
forward; V3-specific work has not started.

---

## 3. Phase detail

### 3.0 Phase 0 — Inheritance audit

Before Phase 1 starts:

1. Confirm `git diff` between V2's `ml/models/{rgcn_attn_encoder,rgcn_attn_markov_encoder,heads}.py`
   and whatever V3 imports or copies — must be clean, or the deviation must be justified and logged.
2. Re-run `ml/ds_backbone.py::freeze`'s assertion (every backbone parameter `requires_grad = False`)
   in the V3 environment, not just trust that it passed once in V2.
3. Spot-check dataset integrity on the carried-forward `db/csv/` sweep: re-run
   `db/run_benchmark.py --verify-determinism` on at least one variant, and `db/load_data.py` against
   at least one variant-seed, before trusting any of it as V3's testbed. The `v1_archive` folder
   confirms git history exists to fall back on if anything looks wrong.
4. Confirm `db/csv/` still holds what `db/README.md` claims (12 variants × seeds 42/43 at spec
   scale, plus the extended seeds for 0/B/D/J/K) — this file lists directories present at the time
   this roadmap was written; treat it as a starting inventory, not a standing guarantee.

**Gate:** do not start Phase 1 training against a dataset or backbone that hasn't cleared this
audit — every downstream number in Phases 1–5 is only as trustworthy as this step.

### 3.1 Phase 1 — Latent State Estimation (Layer 3)

**Purpose** (`docs/HADES-v3_init.md` §6): estimate hidden *operational conditions*, not hidden
supplier *identity* — the question V2's Layer 3 answered is closed differently from the question
V3's Layer 3 asks.

**Step 1 — confirm ground truth before building anything.** Six candidate states were proposed:
Supply Stress, Inventory Health, Logistics Stability, Capacity Pressure, Supplier Reliability,
Recovery Capability. Two already have a confirmed generator-side variable (`own_stress` ≈ Supply
Stress; `RESILIENCE` ≈ Recovery Capability, and Mechanism E's resilience is separately confirmed
recoverable at **AUC ≈ 0.599** — `docs/phase2_coverage_recheck.md`, cited in the init doc §6). The
other four do **not** have confirmed ground truth as of this roadmap. For each of the four, before
building an estimator head: check `db/generate_dataset.py` for a corresponding internal variable,
using the same confirmation discipline `layer3_testing.md` §9.1/§10.1 applied to `HP_GROUPS` and
`RESILIENCE`. A state with no generator-side signal is either dropped from this build or explicitly
scoped as "would need new instrumentation" — never assumed.

**Step 2 — build estimator heads only for confirmed states.** Privileged-supervision discipline
throughout: any generator-tracked latent used as a training/eval target is disclosed, never a model
input at inference — the standard already applied to `ml/extract_hidden_state.py`.

**Step 3 — testbed.** Use Variant A (`J + A`, tier-truncated observable graph) — it already exists
in the carried-forward dataset and is the scenario the init doc names as the natural fit
(Tier-1-only visibility). No new dataset variant is required to start this phase.

**Gate:** an estimator head that does not clear a measured reproduction floor (same discipline as
§1's model-seed floor — retrain several times, changing only init seed, before trusting any AUC
delta) does not proceed to Phase 2. Report per-state AUC against the 0.599 resilience precedent as
the reference point, not an assumed target.

### 3.2 Phase 2 — Structural Causal Model (Layer 4)

**Purpose** (`docs/HADES-v3_init.md` §7): convert Phase 1's estimated states into explicit,
stated causal equations, extracted from `db/generate_dataset.py`'s actual code — not learned from
data. This sidesteps causal discovery entirely, which the init doc correctly notes is the single
hardest problem the original proposal would otherwise have inherited.

**Step 1 — extraction.** Pull the true equations directly from the generator: `own_stress`,
`COPARENT_COUPLING`, `HP_ALPHA`, the resilience absorption terms (§ "How the world works" and "V2
mechanisms" in `db/README.md`), and the stress→delay/shortage/impact conversion chain.

**Step 2 — validate the extraction, not just the design.** Before this SCM is used by anything
downstream: run it forward on real generator inputs and confirm it reproduces the generator's own
simulated outputs exactly (or within a stated, measured tolerance). An SCM that is "extracted" but
silently wrong is worse than no SCM, because Phase 3 depends on it being load-bearing (§1). This
validation step is new — it is not covered by any V2 report — and belongs in this phase, not
deferred to Phase 3's evaluation.

**Step 3 — record the validity caveat explicitly**, per the init doc §7: this SCM is correct
*because* the generator's equations are known and coded. That does not establish anything about
real deployment, where no such generator exists. State this in the SCM's own documentation, not
just in this roadmap — every V3 report that cites SCM correctness should carry the same caveat V2's
reports carry about synthetic-vs-real.

### 3.3 Phase 3 — Counterfactual Engine (Layer 5)

**Purpose:** evaluate interventions by simulating outcomes through Phase 2's SCM — **not** by
re-running SHARE/Markov's forward pass on an edited graph, which §1 already showed does not track
true effects, and **not** by a small head trained on frozen embeddings, which §1 showed is worse
still.

**Step 1 — route through the SCM.** Build the SCM-based counterfactual path as the primary
mechanism. `ml/counterfactual_edit.py` and `ml/counterfactual_ground_truth.py` remain useful as the
generator-side re-simulation and ground-truth harness (`run_counterfactual_phase1.py`'s pairing
logic is directly reusable), but the *prediction* side must read from Layer 4, not from SHARE's raw
forward pass.

**Step 2 — extend task coverage, addressing the known blocker.**
`reports/decision_support_build.md` §1.2 found the closed-form ground truth is usable for `impact`
and `shortage` (both coarse aggregates, robust to schedule divergence) but **not for `delay`**,
where a single extra RNG draw flips 179% as many labels as there are true positives. Extending to
`delay` needs the generator to apply a structural edit at an arbitrary `t0` and simulate forward —
`§4` of that report names Mechanism C's pre-scheduled rewire machinery as the natural template.
Build that extension before claiming counterfactual coverage of all three tasks.

**Step 3 — extend beyond Variant 0.** All of Phase 1 of the prior work was Variant 0 only. Testing
whether the SCM-routed approach's performance is specific to Variant 0's single causal channel (no
mechanisms enabled beyond the co-parent bleed) or generalizes is comparatively cheap and should
happen before or alongside spec-scale evaluation (Phase 5).

**Gate:** measure a reproduction floor for SCM-routed sign agreement the same way §1 measured one
for the naive approach (0.324 max spread from model-init seed alone) before reporting any delta as
a finding. A result that doesn't clear its own floor is not evidence, regardless of which layer
produced it.

### 3.4 Phase 4 — Prediction heads

Per `docs/HADES-v3_init.md` §9, these are parallel outputs of one final stage, not sequential
engines. Reuse validated V2 machinery rather than rebuilding:

| Head | Reuse from V2 | What's already known |
|---|---|---|
| Confidence / uncertainty | `ml/uncertainty_ensemble.py`, `ml/uncertainty_calibrate.py` | Dataset-seed variance dominates total AUC variance: **98.7% (delay), 67.9% (shortage), 70.8% (impact)** (`reports/decision_support_build.md` §2.2). Isotonic recalibration drops ECE 9–17×; model-init ensembling does **not** clear its own calibration floor on any task. Design uncertainty propagation through the SCM (init doc §12's open question) with this asymmetry in mind — an uncertainty estimate that only ensembles model-init seeds is measuring the smaller source. |
| Explanation | `ml/explain_prediction.py`, `ml/test_explanation_faithfulness.py` | Occlusion-based attribution clears its faithfulness gate at **1.66× chance** (0.166 vs 0.100), but only on the **3.4%** of `delay` predictions that aren't saturated (96.5% sit at p ≥ 0.999). Extending to `shortage`/`impact` is mechanical but will hit a different saturation rate per task — measure it before reporting. A second attribution family (integrated gradients, learned edge mask) held to the same specificity gate would separate "this method is weak" from "these predictions genuinely aren't explainable." |
| Counterfactual outcomes | Phase 3's SCM-routed engine | Feeds directly once Phase 3 clears its gate. |

**Gate:** no head ships into the integrated Layer 5–6 stack until its own component-level gate
(above) is cleared independently — do not let a weak head's failure hide inside an aggregate metric.

### 3.5 Phase 5 — Spec-scale evaluation

**Everything in `reports/decision_support_build.md` — the naive-counterfactual miss, the variance
decomposition, the explanation faithfulness rate — was measured at the `v1` preset (800 suppliers,
15 snapshots), the cheap fixture, not the benchmark's actual spec configuration (`sup_n=4000`, 40
snapshots).** V2's own roadmap flagged the identical gap for the base benchmark (§4 of
`HADES_v2/docs/14_Project_Roadmap.md`: "everything so far is the 800-supplier fixture"). V3 should
not repeat it silently.

1. Re-run Phase 1's estimator gates, Phase 2's SCM validation, and Phase 3's counterfactual gate at
   spec scale, using the already-generated `db/csv/` sweep — no new dataset generation required for
   the variants already at spec scale (0, A–K at seeds 42/43).
2. Five seeds minimum per measurement, sign-consistency reported, paired bootstrap CI — the standard
   this project has applied everywhere (`STEP5B_DECISION_SUPPORT_BUILD_PROMPT.md`'s ground rules,
   `HADES_v2/docs/11_Implementation_Guide.md` §8).
3. State explicitly, per number, whether it confirms or revises the `v1`-preset finding — the same
   discipline `decision_support_build.md` §2.2 used when its own uncertainty decomposition revised
   `phase7_training_results.md` §5.6's single-dataset-seed estimate.

**Gate:** no V3 result is publishable as a benchmark finding until it has been checked at spec
scale. A `v1`-preset-only result is a pilot, not a finding — same standard V2's roadmap held itself
to for the base benchmark sweep.

### 3.6 Phase 6 — Reporting and documentation

1. Rewrite the V2 doc suite's counterparts for V3 (`01_Product_Requirement_Document.md` →
   research questions this architecture answers that V2 couldn't; `02_Technical_Requirement_Specification.md`;
   an implementation guide analogous to `11_Implementation_Guide.md`; a testing document analogous
   to `13_Testing_Documentation.md`).
2. Publish the comparison table `docs/HADES-v3_init.md` §11 already drafted, filled in with
   measured numbers instead of architecture claims: does V3 answer `P(Y|do(X))` at a rate
   distinguishable from chance, at spec scale, with a measured floor under it.
3. Update `findings/evolution.md`'s style of record — a chronological account of what was tried and
   why — with a new section for V3, regardless of which way the SCM-routed counterfactual result
   goes. A negative result here (SCM-routed counterfactuals also failing to clear their floor) is as
   reportable as a positive one; §1 already shows this project reports clean misses in full rather
   than dropping them.
4. Keep the deployment-readiness caveat (init doc §10) in every document that reports a positive
   result: validated-on-this-benchmark-via-generator-extracted-equations is achievable; ready for
   real deployment is a separate, unestablished claim.

---

## 4. Known risks carried into Phase 1

1. **The SCM-load-bearing finding (§1) rests on Variant 0 at `sup_n=800`, one task (`impact`), three
   intervention types.** It is a strong result, but not yet an established absence at spec scale or
   on other variants — Phase 3 §step 3 exists specifically to test whether it generalizes.
2. **`delay` has no usable closed-form counterfactual ground truth** (§3.3, step 2). Any claim of
   full three-task counterfactual coverage without addressing this is premature.
3. **Four of six candidate latent states have no confirmed generator-side ground truth.** Phase 1
   must not assume all six are buildable; building an estimator for an unconfirmed state risks
   training against noise and reporting a spurious AUC.
4. **Dataset-seed variance dominates total predictive variance (68–99%, §3.4).** Any V3 result
   evaluated on a single dataset seed is not trustworthy on this benchmark — this is a stronger
   version of a risk V2's own roadmap already carried ("no single-seed reading establishes that a
   variant is in band," `db/README.md`, Known limitations).
5. **Explanation faithfulness is only measured on 3.4% of predictions** (the non-saturated band).
   The other 96.5% currently have no faithfulness-checked explanation at all — a gap that should be
   stated in any user-facing decision-support output, not smoothed over.
6. **Resilience/state coverage ceilings differ by variant and are bounded below 100%** — the ~28%
   of suppliers who never ship cannot be estimated under Mechanism E/F's existing coverage limits
   (`db/README.md`, Known limitations; `docs/phase2_coverage_recheck.md`). Any Layer 3 estimator
   inherits this ceiling; it is a data property, not a modeling failure to be optimized away.
7. **Variant K's Type B signal sits below detectability** and **E/F's recoverability sign inverts**
   (carried forward verbatim from `HADES_v2/docs/14_Project_Roadmap.md` §6) — still true of the
   dataset V3 is using, since it is the same dataset.

---

## 5. Deferred — explicitly out of scope for this roadmap

| Item | Why deferred |
|---|---|
| Causal discovery from real (non-synthetic) operational data | The generator-extraction shortcut (§3.2) is only valid on this benchmark; real-data SCM construction is a substantially harder, separate project (init doc §7, §10) |
| Online-adaptive causal rules | Explicitly named open and out of scope in the init doc §12 |
| A generator-level fix to make hidden-parent coupling pairwise rather than group-mean-mediated | Would reopen V2's Layer 3 question; the positive control already predicts recoverability (0.78 AUC) if built, but it is a generator change, out of scope for V3's architecture line (`findings/evolution.md` §5) |
| V2's original base-benchmark open items (spec-scale label-rate re-derivation, storage target) | Carried in `HADES_v2/docs/14_Project_Roadmap.md` §3–4; V3 inherits the dataset as generated, and does not re-open base-benchmark calibration unless a V3 finding specifically requires it |

---

## 6. Immediate next step

Run Phase 0 (§3.0). It is cheap, has not been formally executed inside the V3 repository yet, and
every later number in this roadmap is conditional on it passing. Do not start Phase 1 training runs
before the audit's four checks are logged.
