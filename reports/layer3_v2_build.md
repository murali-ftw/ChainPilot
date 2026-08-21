# Layer 3 Redesign v2 — Task-Specific Latent Encoders

**Date:** 2026-08-20
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisites read in full before starting:** `results/final_architecture_v3.md` §4 (Layer 3),
`reports/layer3_uncertainty_aware.md`, `reports/phase1_latent_state.md`, and — because it is the
immediately preceding build against the alternative design — `reports/layer3_redesign_build.md`.

**New code, five files, no inherited file modified:** `ml/task_identifiability_gate.py` (Phase 0),
`ml/task_latent_encoder.py` (Phase 1), `ml/task_incremental_value.py` (Phase 2),
`ml/task_confidence.py` (Phase 4 + the shared prediction grid), `ml/task_evidence_status.py`
(Phase 5). **Phase 3 runs `ml/uncertainty_calibrate.py` unmodified, through its own CLI.**
`ml/identifiability_check.py`'s CRN protocol, `ml/uncertainty_ensemble.py`'s variance
decomposition and `ml/hypothesis_ranker.py`'s isotonic/ECE machinery are imported unmodified.

**Scale caveat, carried from every prior phase:** `v1` preset (800 suppliers, 15 snapshots), five
dataset seeds 42–46. Per `docs/14_Project_Roadmap.md` §3.5 this is a **pilot, not a benchmark
finding**. Both testbeds are reported: Variant A (`J+A`) and Variant E (Mechanism E).

---

## Verdict, up front

| task | Phase 0 tier + result | Phase 1 | Phase 2 `Δ_t` (floors) | Phase 3 | Phase 4 | **Status** |
|---|---|---|---|---|---|---|
| **delay** | **A and B — identifiable (A)**, not on E | h¹ fixed | **+0.0002** (init 0.0141 / dset 0.0017) | transferred **20/20** | **rejected** | **Unidentifiable** |
| **shortage** | **gate not computable** (measured) | h³ fixed | **+0.0224** (init 0.0346 / dset 0.0294) | transferred **20/20** | **rejected** | **Unidentifiable** |
| **impact** | **A — identifiable (A)**, B no, neither on E | h⁴ fixed | **−0.0005** (init 0.0047 / dset 0.0019) | transferred **20/20** | **rejected** | **Unidentifiable** |

**No task earns Layer 3 under this variant.** All three fall back to the existing Layer-2-only
(SHARE + Markov) path, unchanged. `Z_t` is not wired into any risk head on either testbed.

Four results are worth having regardless, and three of them are new to this project:

1. **`Z_t = f_t(X, H)` cannot, even in principle, carry information beyond `(X,H)`** — it is a
   deterministic function of the conditioning set, so `I(Y;Z|X,H) ≡ 0`. The v2 design's §2
   definition and its §5 test are in direct contradiction, and the measured `Δ_t ≈ 0` on the two
   well-conditioned tasks is the predicted consequence, not a bug. §"A definitional finding".
2. **`shortage`'s identifiability gate is not computable on this generator**, and that is
   *measured*: perturbing the latent set changes the shipment **entity set** (+24 shipments; only
   **0.13%** of the ledger keeps the same identity at the same index), which common random numbers
   cannot pin.
3. **The confidence signal fails specifically on the axis the brief said to check.** Pooled across
   worlds the Low→High Brier spread looks strong (+0.094 to +0.169). Per world it is monotone in
   **0/5, 3/5, 3/5** worlds, and the band edges transfer across worlds in **3/20, 13/20, 7/20**
   ordered pairs. A pooled-only check would have accepted all three.
4. **Cross-world calibration transfers cleanly here, where it failed for the v1 latent-state
   heads.** ECE 0.2504 → 0.0238, 0.3155 → 0.0195, 0.1817 → 0.0141, **20/20 folds improved** on
   every arm, against member floors of 0.0092 / 0.0046 / 0.0025. This does **not** rescue Layer 3
   — it is a property of the Layer-2 pathway's task heads — but it resolves an open question from
   `reports/layer3_uncertainty_aware.md` STEP 3.

---

## A documentation defect, restated once

`final_architecture_v1.md` — which this prompt names as the reference design — **does not exist in
this repository**, and `results/` has never contained a `v1` or `v2` (checked against git history).
Its Layer-3 section numbers (`§4.4.3`, `§4.4.5`, `§4.4.6`) do not resolve against
`final_architecture_v3.md`, whose Layer 3 is `§4.1`–`§4.6`. This is the same gap
`reports/layer3_redesign_build.md` recorded, and the fourth such dangling reference in four
sessions.

**Closed rather than skipped:** the v1 *redesign* it points at is fully realised in this repository
as `reports/layer3_redesign_build.md` (2026-08-19), which contains every artefact this prompt cites
it for — the Model A→B→C progression, the state-to-task sufficiency matrix, and the three-way
evidence-status table. That report is used as the reference, and the closing comparison section
below is drawn from it.

---

## A definitional finding that governs how every Phase 2 number must be read

Stated before the results because it is a property of the v2 design, not of this implementation.

§2 defines `Z_t = f_t(X, H^{k_t})`. §5 tests the underlying question `I(Y_t; Z_t | X, H) > 0`.
**These are incompatible.** Conditional mutual information of a deterministic function of the
conditioning set is exactly zero:

    Z = f(X,H)   ⟹   I(Y; Z | X, H) = 0

for any `f`, at any capacity, at any sample size, in the population limit. So the quantity §5 names
as its target is identically zero by construction for every encoder of this form.

This is not a reason to stop, and the build did not stop. The test Phase 2 actually runs is a
*downstream performance* comparison at finite sample and finite capacity, where a compressed,
explicitly-supplied non-linear feature can genuinely help a small head that would otherwise have to
learn the same non-linearity from limited data. That is a **representation/optimisation** effect,
not an information one.

The consequence for reporting is specific and is enforced throughout: a positive `Δ_t` here may
**not** be described as "`Z_t` contains information the Layer-2 pathway does not provide", because
it provably cannot. It may only be described as "an explicit `Z_t` is easier for the downstream
head to use than the same information implicit in `(X,H)` at this sample size". Phase 2 therefore
carries a **width-matched random-feature control**, without which any `Δ_t` is confounded with
input width — the confound `reports/phase_modelB_concat_baseline.md` §"Honest qualifications" 3
flagged for Model B.

The measured `Δ_t` on the two tasks with clean conditioning sets is **+0.0002** (delay) and
**−0.0005** (impact) — indistinguishable from the zero the identity predicts.

---

## PHASE 0 — Two-tier identifiability gate

**What was built.** `ml/task_identifiability_gate.py`. **One gate invocation per task**, never a
shared gate feeding a three-way fork: identifiability is per-target, and Supplier Reliability
failed this exact test (AUC ≈ 0.50, Case 3 / `STOP-C`) while Supply Stress and Recovery Capability
passed it, so a shared gate would have let one task's outcome leak into another's.

**What this phase deliberately does not ask.** Whether SHARE already contains related information.
That overlap is Phase 2's question. Folding it in here would close a task early because SHARE
carries *related* information even where the task's own residual signal is genuinely identifiable.

### The label rule is validated exactly, not distributionally

Mechanism G is off on both testbeds (`report_delay()` returns `timedelta(0)`), so recorded time
equals true time and `asof_status` reduces to the true-time status. The reimplemented label rule is
therefore checked **row-for-row against the generator's own `label_rows`**, at factual Z with
nothing re-realised — a strictly stronger check than the distributional `harness_check` the
existing CRN module uses:

```
seed 42  label rule [delay] : 11,434 compared, 0 mismatches, exact=True
seed 42  label rule [impact]:  9,810 compared, 0 mismatches, exact=True
```

**Zero mismatches on every seed, both variants, all ten runs.** Every intervention number below is
therefore measuring the world rather than the reimplementation.

### The two tiers, and which applied where

**Tier A — named-mechanism.** `do(stress = z_lo)` vs `do(stress = z_hi)`, **isolated to one
supplier at a time**, `ml/identifiability_check.py`'s protocol exactly: fixed CRN tuples
(`u_delay`, `g_lateness`, `e_early`, `j_jitter`), structure/co-parents/schedule held fixed, levels
taken as the medians of the two halves of a median split. Isolation is exact for both tasks here
because `impact(s,t₀)` reads only `s`'s own shipments and `delay(sh,t₀)` only `sh`.

**Tier B — unconstrained residual.** The joint upstream set moved together, globally over all
suppliers: every supplier's `stress` (which already aggregates `EVENTS`, `IDIO`, co-parent bleed,
hidden-parent coupling and J's upstream chains), `RESILIENCE`/absorption where Mechanism E is on,
and the `PORT_EVENTS` carrier bump.

**The gate is the residual test, and it has two controls:**

| quantity | question |
|---|---|
| `label_shift` | can the world be told apart from `Y` alone? *(raw effect)* |
| `observable_shift` | can it be told apart from `(X, X_nbr)` alone? *(reported, **not** gated on)* |
| **`residual_shift`** | **can it be told apart from `Y − g(X, X_nbr)`?** *(the gate)* |

`g` is fitted on one entity-disjoint third; the world classifier is trained on a second third's
residuals and tested on a third. Every group carries rows from **both** worlds. The null is two
worlds with the *same* latent configuration and a different realisation seed, and its excursion
from 0.5 is folded into the floor alongside the init-seed and dataset-seed spreads.

### Results

| variant | task | tier | rows | `Y` differs | label AUC | obs AUC | **residual** | **null** | floor | clears | sign 5/5 | **IDENT** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | delay | **A** | 7,914 | 288 | 0.5193 | 0.7419 | **0.5322** | 0.5017 | 0.0217 | **YES** | yes | **YES** |
| A | delay | **B** | 7,914 | 288 | 0.5193 | 0.7419 | **0.5322** | 0.5017 | 0.0217 | **YES** | yes | **YES** |
| A | impact | **A** | 9,276 | 228 | 0.5111 | 0.6165 | **0.5260** | 0.4973 | 0.0253 | **YES** | yes | **YES** |
| A | impact | **B** | 9,276 | 228 | 0.5111 | 0.6653 | 0.5268 | 0.4982 | 0.0329 | no | yes | **NO** |
| E | delay | A | 7,712 | 67 | 0.5036 | 0.6398 | 0.5150 | 0.5024 | 0.0277 | no | yes | **NO** |
| E | delay | B | 7,714 | **12** | 0.5003 | 0.6156 | 0.5048 | 0.5001 | 0.0158 | no | yes | **NO** |
| E | impact | A | 9,195 | 62 | 0.5036 | 0.5497 | 0.5120 | 0.5027 | 0.0221 | no | yes | **NO** |
| E | impact | B | 9,195 | **10** | 0.5001 | 0.5411 | 0.5020 | 0.5002 | 0.0088 | no | **NO** | **NO** |

Per-seed residual AUCs (Variant A): delay `[0.5323, 0.5233, 0.5399, 0.5370, 0.5287]`, impact
Tier A `[0.5208, 0.5269, 0.5228, 0.5287, 0.5308]`.

**Three readings, all measured rather than inferred.**

- **Tier A and Tier B coincide exactly for `delay`.** Not a coding accident: a shipment's outcome
  depends on its own supplier's `stress` and `absorption` and nothing else, so there is no
  cross-supplier path into it and per-supplier isolation and a global move produce the identical
  per-entity result. They diverge for `impact` (0.5260 vs 0.5268, and materially in `obs AUC`:
  0.6165 vs 0.6653) because impact's neighbourhood block aggregates co-parents, which the global
  perturbation moves and the isolated one deliberately does not.
- **Variant E does not replicate Variant A's pass, and the mechanism is visible in the row
  counts.** Only 67 / 12 labels differ on E against 288 on A. Mechanism E's
  `p_delay = 0.025 + 0.38·st·absorption` with `absorption = max(0, 1 − 1.3·RESILIENCE)` damps the
  channel the intervention acts through; on Tier B, where `RESILIENCE` is itself driven high, the
  channel is very nearly closed and only **12 of 7,714** labelled rows move at all. So E's
  non-replication is a property of the mechanism, not noise — but it is a non-replication, and this
  project does not carry a result on one variant.
- **The `observable_shift` column is why this gate is separated from Phase 2.** It runs at
  0.55–0.74 everywhere — the intervention moves the observables a great deal — while the residual
  sits at 0.50–0.53. Gating on the observable shift would have passed every arm on both variants.

**Stated limitation, in the direction that matters.** `X_nbr` is the entity's graph-neighbourhood
observable block standing in for what one round of message passing sees; conditioning on the
emitted observables plus their neighbourhood is the tightest conditioning available without
re-running SHARE inside each counterfactual world. Conditioning on *less* than the true `H` makes a
Tier B positive **easier**, so a pass here is licence to attempt the encoder and never confirmation
that it will succeed. Phase 2 is what confirms that — and it did not.

### `shortage` — the gate is not computable, and that is measured

`delay` and `impact` are read off shipment outcomes, and re-realisation rewrites those outcomes
without changing which shipments exist. `shortage` is produced by the weekly inventory walk, which
**calls `new_shipment()` inside its own loop**: a replenishment order — with a freshly drawn
carrier, supplier and lead time — is created whenever stock crosses a trigger that the intervened
quantity itself moves. Common random numbers pin per-entity draws; they cannot pin an entity set
whose size and composition change.

Measured directly, by running the generator factual and with `own_stress` forced to a constant:

| | Variant A | Variant E |
|---|---|---|
| shipments, factual → perturbed | 39,138 → **39,162** (+24) | 38,888 → **38,759** (−129) |
| same shipment **id** at the same ledger index | 100% | 100% |
| **identical identity tuple** (supplier, factory, dispatch, eta) at the same index | **0.13%** | **0.25%** |
| shortage-event key Jaccard | 0.724 | — |
| **CRN applicable** | **False** | **False** |

The ids match because they are `uid("shp", index)`; the *entity at that index* is a different
shipment in 99.87% of cases. This is the same structural obstacle
`ml/counterfactual_ground_truth.py` already measured when it replaced re-simulation with
re-realisation.

**Disposition: `shortage` closes at Phase 0** — and it needs a fifth category, because none of the
four given fits. It is not (A) not-simulated, not (C) too-sparse (96,216 test label rows), and not
(D) unidentifiable-via-intervention, because the intervention was never runnable. It is **"gate not
computable under the available protocol"**, and forcing it into (D) would report a measurement
that was never made. The nearest of the four is **(B) already-an-emitted-feature** — the inventory
state driving shortage is `stock_level`/`reorder_threshold`, emitted and already a model input,
which is `reports/phase1_latent_state.md` Step 1 #4's independent finding for the same quantity.

Per this project's precedent for reporting past a STOP (`layer3_uncertainty_aware.md` Steps 4–5),
`shortage` was nonetheless carried through Phases 1–4 as an **explicitly off-ladder diagnostic**,
because those phases need no CRN and the result is informative about what the closure cost. It is
labelled as such in every table below.

**Cost check:** 39–40 s per dataset seed for both tasks × both tiers; the full 5-seed run took
**3 min 20 s** per variant. The shortage feasibility probe is two generator runs, **15 s**.

---

## PHASE 1 — Task-specific latent encoder, fixed depth

**What was built.** `ml/task_latent_encoder.py`. `Z_t = f_t(X, H^{k_t})`, a bottleneck MLP
`[X ; H^k] → Linear(·,32) → ReLU → Dropout(0.1) → Linear(32, 8)`, with a linear head on the
bottleneck supplying the training signal. `Z_t` is the 8-dim bottleneck; Phase 2 consumes `Z`,
never the head's logit.

| task | depth `k_t` | source |
|---|---|---|
| delay | **h¹** | `MARKOV_READOUT_DEPTH`, this benchmark's existing per-task choice |
| shortage | **h³** | as above |
| impact | **h⁴** | as above |

Every training decision matches `ml/latent_state_head.py::train_head` — same standardisation from
train statistics, same `Adam(lr=1e-3, weight_decay=1e-4)`, same 120 full-batch epochs, same
`pos_weight`, same seeding of torch and numpy — so the encoder sits on the same axis as every other
head in this project. Fitted on `tr`; `Z` on `va`/`te` is therefore out of sample for it.

**Confirmation that no forbidden mechanism was introduced.** No multi-depth concatenation, no
learned depth weighting, no temporal modelling, no reconstruction decoder, no causal-loss coupling.
This is a **runtime check, not a promise in a comment**: `assert_no_forbidden_mechanism()` runs
inside every encoder construction and raises if an encoder ever declares one of them or a
non-fixed depth mode. Its self-test confirms it fires:

```
$ venv/bin/python ml/task_latent_encoder.py --selftest
encoder selftest: Z shape (50, 8), depth map {'delay': 1, 'shortage': 3, 'impact': 4}
forbidden-mechanism guard fires as intended: Phase 1 forbids ['learned_depth_gate'] unless a
preceding gate earned it; run it as a separately-labelled Model A->B->C follow-up instead
```

The project's record against adaptive depth is 8/8, and `reports/phase_modelB_concat_baseline.md`
already ran the concatenation diagnostic (Model B) for the state-family framing — it did not clear
its floor on any of three arms, so Model C was never built. Nothing in Phase 2 below produces the
marginal-but-inconsistent signal that would license reopening that as a separate follow-up.

---

## PHASE 2 — Residual-information / incremental-value evaluation, both seed axes

**What was built.** `ml/task_incremental_value.py`. Four arms on one common row set, downstream
head fitted on `va` and scored on `te`:

| arm | features | role |
|---|---|---|
| `baseline` | `[X ; H^k]` | the bar |
| `layer3` | `[X ; H^k ; Z_t]` | the arm |
| **`noise`** | `[X ; H^k ; N(0,1)^8]` | **width-matched control** — identical input width and first-layer parameter count, carrying no information |
| `z_only` | `[Z_t]` | diagnostic: did the encoder learn anything at all |

**Mutual information is not estimated directly**, per the brief — the downstream performance
comparison is far less noisy in this data regime.

**Both seed axes, and the pass requires both independently.** 5 dataset seeds × 5 init seeds. A
task passes only if `Δ_t` is positive, clears the **init-seed** floor, clears the **dataset-seed**
floor, and is sign-consistent across all five dataset seeds — not merely positive on the averaged
number, which can hide an axis that fails on its own.

**The second grid axis is the head init seed, not the backbone seed**, carried verbatim from
`layer3_uncertainty_aware.md` §3 and forced by the same constraint: only `m0` backbone checkpoints
exist in `out/ds_ckpt/`, and building `m1..m4` would mean training SHARE, which the ground rules
forbid.

**Cost check:** 12 s for a 1-seed × 2-init-seed probe; the full 5 × 5 × 3-task grid ran in
**~2 min 30 s** per variant.

### Results — Variant A

| task | depth | `baseline` | `layer3` | `noise` | `z_only` | **`Δ_t`** | init floor | dset floor | sign 5/5 | `Δ`−noise | **GATE** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| delay | h¹ | 0.7661 | 0.7663 | 0.7642 | 0.7547 | **+0.0002** | 0.0141 | 0.0017 | **NO** | +0.0021 | **STOP** |
| shortage † | h³ | 0.8196 | 0.8420 | 0.8195 | 0.8276 | **+0.0224** | 0.0346 | 0.0294 | yes | +0.0225 | **STOP** |
| impact | h⁴ | 0.9287 | 0.9283 | 0.9266 | 0.9273 | **−0.0005** | 0.0047 | 0.0019 | NO | +0.0017 | **STOP** |

### Results — Variant E

| task | depth | `baseline` | `layer3` | `noise` | `z_only` | **`Δ_t`** | init floor | dset floor | sign 5/5 | `Δ`−noise | **GATE** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| delay | h¹ | 0.8110 | 0.8111 | 0.8085 | 0.8106 | **+0.0001** | 0.0094 | 0.0036 | NO | +0.0026 | **STOP** |
| shortage † | h³ | 0.8273 | 0.8418 | 0.8232 | 0.8416 | **+0.0145** | 0.0483 | 0.0555 | NO | +0.0186 | **STOP** |
| impact | h⁴ | 0.9111 | 0.9101 | 0.9083 | 0.9142 | **−0.0010** | 0.0106 | 0.0013 | yes | +0.0018 | **STOP** |

† *off-ladder: `shortage` closed at Phase 0 (gate not computable). Reported for completeness.*

Per-seed `Δ_t`, Variant A: delay `[+0.0006, −0.0007, −0.0005, +0.0004, +0.0011]`; shortage
`[+0.0341, +0.0102, +0.0311, +0.0320, +0.0047]`; impact `[−0.0010, −0.0001, −0.0004, −0.0014,
+0.0006]`.

**Reading.**

- **`delay` and `impact` land on top of the zero the identity predicts.** `Δ` = +0.0002 and
  −0.0005, and neither is sign-consistent. These are the two tasks where the conditioning set
  `(X,H)` is clean and complete, and there is nothing for an explicit `Z_t` to add.
- **`shortage`'s +0.0224 is the largest effect in the build and it still does not clear.** Its
  floors are 0.0346 (init) and 0.0294 (dataset) — it fails on *both* axes, so the two-axis rule is
  not what stopped it; a single-axis rule would have stopped it too. It is also the one arm where
  `z_only` (0.8276) beats nothing much: the encoder did learn something, just not something the
  baseline lacked. On Variant E the same arm is +0.0145 and **not sign-consistent** (one seed at
  −0.0102), so it does not replicate either.
- **The width control earns its place, and it exonerates `Z_t` rather than indicting it.**
  `Δ_noise` is −0.0000 to −0.0041 across all six arms — appending eight pure-noise dimensions costs
  a little and buys nothing. So `shortage`'s +0.0224 is genuinely attributable to `Z_t`'s content,
  not to input width. It simply is not large enough to clear the floor its own seed-to-seed spread
  produces. Worth recording that a **single-seed** probe of the same cell returned `Δ` = +0.0325
  with the noise arm at **+0.0427** — the control *beating* the real feature — which five seeds
  washes out to −0.0000. Second in-project demonstration in two builds of why the five-seed minimum
  exists.

### The variance decomposition, by the project's own estimator

`ml/uncertainty_ensemble.py::metric_decomposition` imported **unmodified** and applied to this
grid's `layer3` AUCs:

| variant | task | `std_init` | `std_dataset` | ratio | **dataset share** |
|---|---|---|---|---|---|
| A | delay | 0.0035 | 0.0367 | **10.6×** | **99.1%** |
| A | shortage | 0.0056 | 0.0112 | 2.0× | 80.0% |
| A | impact | 0.0006 | 0.0042 | 6.6× | 97.8% |
| E | delay | 0.0016 | 0.0271 | **16.7×** | **99.6%** |
| E | shortage | 0.0056 | 0.0524 | 9.3× | 98.9% |
| E | impact | 0.0021 | 0.0193 | 9.4× | 98.9% |

**This independently reproduces the finding the whole floor discipline rests on** — 68–99%
dataset-seed dominance, and §5.6's 10.3× std ratio — on a grid built for a different purpose, with
the same estimator. It is the clearest confirmation of that figure the project has, and it is why
`floor_t = max(init, dataset)` is not a formality: on `delay`/Variant A the dataset-seed floor is
**10.6×** the init-seed one.

---

## PHASE 3 — Calibration, cross-world only

**`ml/uncertainty_calibrate.py`, run unmodified through its own CLI** (digest `24ccd9d58eb44b69…`,
matching every prior report). `ml/task_confidence.py::emit_pred_grid` writes the prediction grid in
exactly the key layout that module's `load_grid` expects, so nothing about PAVA, ECE or the
rotation is reimplemented here.

The isotonic map is fitted on one dataset seed and evaluated on a **different** one, over the full
20-fold leave-one-world-out sweep. **The fraction of folds improving is reported, not just the
average** — the original Step 3 result (+0.0202/+0.0224 in-world, only 7/20 and 8/20 folds improved
out-of-world) is why this project no longer trusts an in-world-only number.

**Formally off-ladder: Phase 3 admits only tasks that passed Phase 2, and none did.** Run and
reported anyway, per this project's precedent, because it is an independent gate on a different
claim and its result is a clean *positive* that would otherwise have to be re-derived later.

| variant | task | `ens_raw` ECE | `ens_cal` ECE | improvement | **folds improved** | member floor | fit floor | verdict |
|---|---|---|---|---|---|---|---|---|
| A | delay | 0.2504 | **0.0238** | **+0.2266** | **20/20** | 0.0092 | 0.000000 | **transferred** |
| A | shortage † | 0.3155 | **0.0195** | **+0.2960** | **20/20** | 0.0046 | 0.000000 | **transferred** |
| A | impact | 0.1817 | **0.0141** | **+0.1676** | **20/20** | 0.0025 | 0.000000 | **transferred** |
| E | delay | 0.2537 | **0.0269** | **+0.2268** | **20/20** | 0.0015 | 0.000000 | **transferred** |
| E | shortage † | 0.2925 | **0.0152** | **+0.2773** | **20/20** | 0.0026 | 0.000000 | **transferred** |
| E | impact | 0.1680 | **0.0055** | **+0.1626** | **20/20** | 0.0016 | 0.000000 | **transferred** |

Brier and NLL move with ECE throughout (e.g. A/impact: Brier 0.1315 → 0.0380, NLL 0.3852 → 0.1305)
and AUC is essentially unchanged (0.9288 → 0.9249), which is what an order-preserving map should
do. The fit floor is exactly 0.000000 — reported as a fact about a deterministic solver, never as a
stability result; the **member floor** is the operative one, and every improvement clears it by
25–156×.

**Why this transfers when the v1 latent-state calibration did not, and what it does and does not
mean.** `layer3_uncertainty_aware.md` STEP 3 classified Supply Stress as `STOP — G`: its raw ECE
was 0.0407, only **2.2×** its binomial noise floor, and the residual miscalibration was
*world-specific*, so a map from one world actively harmed another (7/20 folds). Here the raw ECE is
0.17–0.32, and the miscalibration is large, systematic and **shared across worlds** — it is
dominated by `fit_probs`'s `pos_weight` reweighting, which inflates minority-class probabilities by
the same mechanism in every world. Isotonic removes a distortion that is a property of the *loss*,
not of the *world*, so it transfers perfectly.

**This is a real result and it belongs to the Layer-2 pathway, not to Layer 3.** Phase 2 found
`Δ_t` inside its floor on every arm, so the head being calibrated here is empirically the
`P(Y|X,H)` baseline. The Phase 5 output spec says so explicitly rather than presenting a calibrated
probability as something `Z_t` delivered.

---

## PHASE 4 — Confidence, kept separate from probability, built cheaply

**What was built.** `ml/task_confidence.py`. **No new learned confidence head, no separate
uncertainty network, no additional multi-objective loss.** The candidate is `−var` across the
existing ensemble members — free, reusing the same 5×5 grid Phase 2 already builds. (A learned
confidence head *was* built and measured under the v1 redesign,
`reports/layer3_redesign_build.md` Phase 3, and did not beat the free `|p − 0.5|` margin on any
arm; that is the empirical reason this phase is specified the cheap way, and the reason the route
is not reopened.)

**Validation, and why the dataset-seed axis is the binding part.** Per-entity dataset-seed variance
is **not identifiable** on this benchmark — dataset seeds generate different suppliers, so there is
no entity to pair across worlds — and this module does not fabricate it. It instead requires the
confidence→error relation to survive the dataset-seed axis in the two ways that *are* identifiable:

- **per-world monotonicity** — the relation must hold in **every** world independently. Pooling can
  manufacture a monotone curve out of a between-world mean shift.
- **cross-world transfer** — the band **edges** fitted on one world must still order error on a
  different world, rotated over all 20 ordered pairs.

| variant | task | **pooled** Low−High Brier | member floor | worlds monotone | cross-world folds monotone | **CONFIDENCE** |
|---|---|---|---|---|---|---|
| A | delay | **+0.0943** | 0.0882 | **0/5** | **3/20** | **REJECTED** |
| A | shortage † | **+0.1141** | 0.0818 | **3/5** | 13/20 | **REJECTED** |
| A | impact | **+0.1685** | 0.0882 | **3/5** | **7/20** | **REJECTED** |
| E | delay | −0.0373 | 0.1554 | 0/5 | 0/20 | **REJECTED** |
| E | shortage † | +0.0417 | 0.1111 | 1/5 | 5/20 | **REJECTED** |
| E | impact | +0.1417 | 0.0762 | 4/5 | 14/20 | **REJECTED** |

**This is the gap the brief's Phase 4 was written to close, and it fired.** On Variant A every
pooled spread is positive and clears its member floor — +0.0943, +0.1141, +0.1685 against
0.0882, 0.0818, 0.0882. **A pooled-only check would have accepted all three and shipped a
High/Moderate/Low band for each.** The per-world check finds the relation holds in 0/5, 3/5 and 3/5
worlds, and the band edges transfer in 3/20, 13/20 and 7/20 ordered pairs. Init-seed disagreement
looks stable while the model remains highly seed-sensitive at the dataset level — exactly the
failure mode named in advance, now measured.

**The confidence field is reported as ABSENT for every task, on both variants.** It is not
backfilled with a number that failed its own validation, and no High/Moderate/Low banding is
emitted.

---

## PHASE 5 — Evidence status and the three-field output

**What was built.** `ml/task_evidence_status.py`. Calibration transfer is operationalised by a rule
fixed before the numbers were read: `transferred` = >50% of ordered folds improve **and** the mean
improvement clears the member floor; `mixed` = ≥50% improve but inside the floor;
`majority_failure` (`STOP-G`) = <50% improve.

| task | Phase 0 | Phase 2 | Phase 3 | Phase 4 | **STATUS** |
|---|---|---|---|---|---|
| delay | pass (A/B, Variant A) | **STOP** | transferred | rejected | **Unidentifiable** |
| shortage | **CLOSED** (not computable) | STOP † | transferred † | rejected † | **Unidentifiable** |
| impact | pass (A, Variant A) | **STOP** | transferred | rejected | **Unidentifiable** |

*Identical on both variants; on Variant E, `delay` and `impact` additionally fail Phase 0.*

**A naming tension in the specified table, flagged rather than silently resolved.**
`Unidentifiable` is doing double duty: it covers both "closed at Phase 0" and "failed Phase 2's
floor outright". Those are materially different findings with different follow-ups — `delay` and
`impact` **did** pass Phase 0's identifiability gate on Variant A and are being labelled
"Unidentifiable" for failing a later, different test. The label follows the table as specified; the
detail is carried alongside it in `phase0_outcome` / `phase2_outcome` and distinguished in prose
throughout.

### The three-field output spec, per task

Identical for all three tasks, on both variants:

| field | value |
|---|---|
| **Prediction** | calibrated `P(Y_t=1)` from Phase 3 — **attributable to the Layer-2 `(X,H)` pathway, not to `Z_t`**, since Phase 2 found `Δ_t` inside its floor |
| **Evidence status** | `Unidentifiable` |
| **Confidence** | **ABSENT** — the signal failed its own validation and is not backfilled |

### What this means for Layer 4 wiring

All three tasks: **fall back to the existing Layer-2-only (SHARE + Markov) path, unchanged.**
No task's `Z_t` is consumed by a risk head. No `Uncertain` task exists to surface for
observability. Layer 4 is not modified by this build in any respect.

---

## Comparison against the closest existing v1-style numbers

The nearest existing measurement is the state-family framing's marginal contribution to `impact`
(`layer3_uncertainty_aware.md` STEP 1, re-entered as cited cells in
`reports/layer3_redesign_build.md` Phase 4):

| framing | arm | marginal contribution | its floor | inside floor? |
|---|---|---|---|---|
| **v1 — state family** | Supply Stress × impact (A) | +0.0082 | 0.0485 | yes |
| **v1 — state family** | Supply Stress × impact (E) | +0.0084 | 0.0676 | yes |
| **v1 — state family** | Recovery Capability × impact (E) | −0.0034 | 0.0615 | yes |
| **v2 — task-specific** | `Z_impact` × impact (A) | **−0.0005** | 0.0047 | yes |
| **v2 — task-specific** | `Z_impact` × impact (E) | **−0.0010** | 0.0106 | yes |
| **v2 — task-specific** | `Z_delay` × delay (A) | **+0.0002** | 0.0141 | yes |
| **v2 — task-specific** | `Z_shortage` × shortage (A) | **+0.0224** | 0.0346 | yes |

**Stated plainly: the task-specific framing did not recover more signal than the state-family
framing. It reproduced the same below-floor result, and on `impact` — the one task both framings
measured — it recovered less.** v1's Supply Stress × impact contributed +0.0082; v2's purpose-built
`Z_impact` contributed −0.0005. Both are inside their floors and neither is a usable effect, so
the ordering between them is not itself meaningful; what is meaningful is that building the latent
*for the task* rather than *for a named operational state* did not help.

Two structural differences make v2's floors much tighter, and they cut in opposite directions:
v2 conditions on `(X, H)` where v1's Step 1 conditioned on `X` alone, so v2's baseline is stronger
and its residual smaller; and v2's `Z_t` is a deterministic function of its own conditioning set,
which v1's `Z_s` — a probe for an independently-defined generator quantity — was not. **v1's
`Z_s` at least could in principle have carried information beyond `X`; v2's `Z_t` provably cannot
carry information beyond `(X, H)`.** That is the sharpest difference between the two variants, and
it is a difference of kind, not of degree.

---

## What must not be built on this evidence

- **No Layer 4 rewiring.** No task reached `Supported`; all three fall back to Layer-2-only.
- **No confidence field**, on any task, on either variant. It was measured and rejected on the
  axis that matters; a pooled-only number would pass and must not be substituted.
- **No adaptive depth, cross-depth fusion, temporal modelling, reconstruction decoder or
  causal-loss coupling.** Phase 2 produced no signal that would license any of them, and
  `assert_no_forbidden_mechanism()` enforces this at runtime rather than by convention.
- **No `shortage` identifiability claim in either direction.** The gate was not run, not run and
  found flat.
- **No claim that a calibrated probability is a Layer-3 product.** Phase 3's result is a property
  of the `(X,H)` pathway's task heads.

## The defensible follow-ups this build supports

1. **If the v2 framing is pursued further, `Z_t` must stop being a function of `(X,H)`.** The
   identity `I(Y;Z|X,H) = 0` is not a finite-sample artefact. A `Z_t` that could carry incremental
   information would have to read something `(X,H)` does not — additional raw fields, a different
   snapshot, or privileged generator state as v1's state heads did.
2. **Instrument the inventory walk to make `shortage` gateable.** Recording the per-`(product,
   warehouse, week)` shortage driver at the point it is computed — the same shape of change
   `reports/phase1_mitigation_level.md` made for `mitigation_level` and proved inert — would let
   the label be recomputed without re-running the replenishment loop, which is the specific thing
   that breaks CRN.
3. **The Phase 3 calibration result is worth adopting on its own terms**, separately from Layer 3:
   a cross-world isotonic map improves task-head ECE by +0.16 to +0.30 with 20/20 folds improving
   on both variants. That is a deployable improvement to the *existing* Layer-2 pathway and needs
   no Layer 3 at all. It should be re-measured at spec scale before adoption.
4. **More dataset seeds**, unchanged from every prior report. `shortage`'s +0.0224 against a
   0.0346 init-seed floor is the only arm close enough that a tighter floor could change its call.

**Every negative here is a non-detection at this scale, not an established absence.** The `v1`
preset, five dataset seeds, and the head-init-seed axis standing in for the backbone-seed axis are
the binding constraints, and they are the same ones every prior result in this project carries.

---

## Integrity — SHARE and the Markov readout are untouched

**Ground rule:** `git diff` check after every phase against the SHARE encoder and the Markov
readout module. Neither changed at any point.

```
$ git diff --stat                     # EMPTY — no tracked file modified
$ git status --porcelain db/          # EMPTY — the generator was executed, never edited
$ git status --porcelain
?? ml/task_confidence.py              <- new, this session (Phase 4 + shared grid)
?? ml/task_evidence_status.py         <- new, this session (Phase 5)
?? ml/task_identifiability_gate.py    <- new, this session (Phase 0)
?? ml/task_incremental_value.py       <- new, this session (Phase 2)
?? ml/task_latent_encoder.py          <- new, this session (Phase 1)
?? ml/evidence_status.py              <- prior session (v1 redesign)
?? ml/inventory_state_families.py     <- prior session (v1 redesign)
?? ml/layer3_interface.py             <- prior session (v1 redesign)
?? ml/state_task_sufficiency.py       <- prior session (v1 redesign)
?? reports/layer3_redesign_build.md   <- prior session (v1 redesign)
```

Reused modules carry digests identical to those recorded at the close of
`reports/world_conditioned_calibration.md` and `reports/layer3_redesign_build.md`:

```
24ccd9d58eb44b69...  ml/uncertainty_calibrate.py            MATCHES   (Phase 3 ran this, unmodified)
3a8c888040446aa2...  ml/uncertainty_ensemble.py             MATCHES   (Phase 2's decomposition)
f1dc41d01a2cd3a8...  ml/identifiability_check.py            MATCHES   (Phase 0's CRN protocol)
c2e15e39cdc101d4...  ml/hypothesis_ranker.py                MATCHES
8e4a0803d8f09cd0...  ml/latent_state_head.py                MATCHES
689962edd1315599...  ml/ds_backbone.py                      MATCHES
1b85a68f9e9705be...  ml/models/rgcn_attn_encoder.py         MATCHES   <- SHARE
9e7919e6bfcf39ca...  ml/models/rgcn_attn_markov_encoder.py  MATCHES   <- Markov readout
cbd40aadc8570745...  ml/models/heads.py                     MATCHES
8fa14bc270946507...  ml/models/depth.py                     MATCHES
```

`ml/identifiability_check.py` is imported and **subclassed**, never edited: `TaskRealiser` extends
`Realiser` to hold many suppliers open at once, because a task label is computed across every
supplier's shipments while `realise_supplier` restores state before returning. Subclassing rather
than editing is what preserves that digest.

All ten backbone checkpoints in `out/ds_ckpt/` carry mtimes of **2026-08-13 22:03–22:44**,
predating this session. Every fit loaded from cache; **no backbone was trained.** Backbone health
on first load matched the recorded values exactly (`delay=0.8046 shortage=0.7835 impact=0.9340`,
Variant A seed 42). `assert_backbone_frozen()` ran after head construction and again after training
on every head and encoder fit and never fired. Phase 0 performs no backbone fits at all — it runs
entirely inside the generator's namespace.

---

## Reproduction

```
$ venv/bin/python -u ml/task_latent_encoder.py --selftest

$ for V in A E; do venv/bin/python -u ml/task_identifiability_gate.py --variant $V \
      --tasks delay,shortage,impact --tiers "delay:A,B;impact:A,B" \
      --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --out out/layer3_v2/phase0_gate_$V.json; done

$ for V in A E; do venv/bin/python -u ml/task_incremental_value.py --variant $V \
      --tasks delay,shortage,impact --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --out out/layer3_v2/phase2_$V.json; done

$ for V in A E; do venv/bin/python -u ml/task_confidence.py --variant $V \
      --tasks delay,shortage,impact --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --emit-preds out/layer3_v2/preds_$V.npz \
      --out out/layer3_v2/phase4_confidence_$V.json; done

$ for V in A E; do venv/bin/python -u ml/uncertainty_calibrate.py \
      --preds out/layer3_v2/preds_$V.npz --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4 \
      --out out/layer3_v2/phase3_calibration_$V.json; done

$ for V in A E; do venv/bin/python -u ml/task_evidence_status.py --variant $V \
      --out out/layer3_v2/phase5_status_$V.json; done
```

Note the run order: Phase 4's module owns the shared prediction grid, so it is executed **before**
Phase 3, which consumes that grid through `ml/uncertainty_calibrate.py`'s own CLI.

Raw results: `out/layer3_v2/phase0_gate_{A,E}.json`, `phase2_{A,E}.json`,
`phase3_calibration_{A,E}.json`, `phase4_confidence_{A,E}.json`, `phase5_status_{A,E}.json`;
prediction grids `preds_{A,E}.npz`; logs `log_phase{0,2,4,5}_{A,E}.txt`.
v1 comparison numbers: `reports/layer3_redesign_build.md`, `out/layer3/step1_{A,E}.json`.
