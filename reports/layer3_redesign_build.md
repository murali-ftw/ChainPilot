# Layer 3 Redesign — Gated, State-Specific Operational-State Estimation

**Date:** 2026-08-19
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisites read in full before starting:** `results/final_architecture_v3.md` §4 (Layer 3),
`reports/layer3_uncertainty_aware.md` (the six-step pipeline this redesign replaces),
`reports/phase1_latent_state.md` (the ground-truth-confirmation process this generalises),
`reports/phase_modelB_concat_baseline.md` (Phase 2's already-run gate) and
`reports/world_conditioned_calibration.md` (checked, per the brief, before Phase 3 assumed
anything about the calibration ceiling).

**New code, four files, no inherited file modified:** `ml/inventory_state_families.py` (Phase 0),
`ml/evidence_status.py` (Phase 3), `ml/state_task_sufficiency.py` (Phase 4),
`ml/layer3_interface.py` (Phase 5). `ml/hypothesis_ranker.py`'s isotonic/ECE machinery,
`ml/ds_backbone.py`'s frozen-loading pattern, `ml/uncertainty_ensemble.py`'s variance
decomposition, `ml/latent_state_head.py`'s Model A head and `ml/layer3_uncertainty.py`'s
`step3`/`step4` are all **imported unmodified**, never reimplemented.

**Scale caveat, carried from every prior phase:** everything below is the `v1` preset (800
suppliers, 15 snapshots), five dataset seeds 42–46. Per `docs/14_Project_Roadmap.md` §3.5 this is
a **pilot, not a benchmark finding**.

---

## Verdict, up front

| Phase | Stop condition | Outcome |
|---|---|---|
| **0** — inventory the four state families | close a family with no candidate generator variable | **HIT for 2 of 4 families.** Logistics and Production closed here as generator-extension requirements; Inventory re-confirmed closed; Supplier survives |
| **1** — observability / identifiability gate | AUC must clear the *measured* null, sign-consistent 5/5 | **No new family reached this gate.** Nothing was run; the three supplier states' Step 2 results are cited |
| **2** — Model A → B → C | Model B must clear its floor over Model A or C is not built | **HIT (cited).** Model B misses on all three arms; **Model C is not built** |
| **3** — split state from confidence, assign status | none — every family gets a status | **All three arms land `Uncertain`.** One genuinely new positive: a learned confidence head *is* monotone where init-seed variance was not |
| **4** — state-to-task sufficiency matrix | none — the completed matrix is the deliverable | **Matrix complete. Zero `Supported` cells** — 2 Redundant, 7 Unsupported |
| **5** — hard interface + ablations | if matrix-validated wiring does not beat the Markov baseline, stop | **HIT.** Matrix-validated wiring beats the Markov baseline on **no task, on either variant**. Layer 5 is not built |

**Headline finding.** *The redesign runs to completion and the completed structure is empty.* Every
surviving family lands `Uncertain`; the state-to-task matrix contains no `Supported` cell; and
because the wiring rule admits only `Supported` cells, the matrix-validated risk head is
**bit-identically the Markov-depth baseline** — its gain over that baseline is `+0.0000` on all
six (task × variant) combinations, not approximately zero but exactly zero, because no column was
wired. That is a complete and valid outcome, and it is this report's headline rather than a reason
to keep iterating past the gates.

The redesign nonetheless earned its cost in three specific places, all of which are new:

1. **Production is closed for a stronger reason than "too sparse."** `FACTORY_OUTAGE`'s window
   ends **before the first snapshot** — 0 of 15 `t0`s intersect it. There is no snapshot at which
   the state is active, so no supervision target exists at any granularity, independent of rate.
2. **Logistics is closed for a reason that would have wasted a full identifiability campaign.** On
   the only variants where `recv_atten` is causally live, it is a deterministic transform of
   `RESILIENCE` — and, at the level a head actually sees, its median-split label is the **exact
   complement** of Recovery Capability's on 800/800 suppliers. A probe for it would have returned
   Recovery Capability's own AUC with the labels flipped.
3. **Per-entity confidence is not the dead end the original pipeline recorded.** Step 4 closed
   confidence as **G** on the basis of init-seed ensemble variance, which was non-monotone in
   0/5 worlds. A learned head `C_s = g_s(H, X)` is monotone **pooled on all three arms**, clearing
   its member floor by 11–17×. The route was wrong, not the question — though the head still does
   not beat the free `|p − 0.5|` margin, which is the honest qualification.

---

## Two documentation defects found on the way in

Stated here rather than buried, because both concern artefacts the build prompt instructs this
work to treat as the specification.

1. **`results/final_architecture_v1.md` does not exist in this repository.** `results/` holds
   `final_architecture_v3.md` and `HADES_build_history_and_architecture_evolution.md`, and git
   history shows no `v1` or `v2` was ever committed. The prompt's section references (§4.4.2,
   §4.4.3, §4.4.4, §4.4.5, §4.4.6, §4.4.8, §4.4.9, §4.5, §6) **do not resolve against v3**, whose
   Layer-3 section is §4.1–§4.6 with no `4.4.x` subsections at all. This is the same class of gap
   `layer3_uncertainty_aware.md` recorded for `docs/decision_support_build.md` §2.3 and
   `world_conditioned_calibration.md` recorded for `reports/decision_support_build.md` §2.2 — the
   third and fourth such reference in three sessions.

   **The gap was closed rather than noted and skipped.** `final_architecture_v3.md` §4 was read in
   full, and the build prompt is itself a complete specification of what each phase must do — every
   gate, floor, protocol and disposition category it names is stated in its own text. **This build
   therefore implements the prompt, cross-checked against v3 §4**, and every place where the two
   could diverge is flagged below rather than silently resolved.

2. **The prompt's seeding instruction for Phase 4 conflicts with the classification rule it also
   states.** It directs that the Supply Stress × impact *and* Recovery Capability × impact cells be
   entered as **Redundant** directly. But it also directs that every cell be classified "exactly per
   the architecture's rule", under which **Redundant** requires that *"Z alone predicts Y"*. For
   Recovery Capability × impact, `P(Y|Z) = 0.5711` against a floor of `0.1391` — it **does not**
   clear, which is precisely why `layer3_uncertainty_aware.md` STOPped it at Step 1. Z alone does
   not predict Y, so the rule gives **Unsupported**, not Redundant.

   **The rule was applied, and the deviation is recorded here rather than resolved silently.**
   Both readings are reported in the matrix so the discrepancy is visible; the classification
   difference changes no downstream decision, because neither `Redundant` nor `Unsupported` is
   wired into a risk head.

---

## PHASE 0 — Inventory the generator against the four proposed state families

**What was built.** `ml/inventory_state_families.py`, a read-only audit script. It executes
`db/generate_dataset.py` in-process with `write()` stubbed — the `ml/confirm_latent_states.py`
mechanism Phase 1 used — and inspects the namespace directly, so every verdict is a measurement
rather than a source reading. It trains nothing and touches no checkpoint.

**Cost check.** One namespace load is **7.5–13.7 s** depending on variant. The full audit over four
variants (A, E, F, K) ran in **41 s** wall clock. Well under the "under an hour" budget.

### The four families

| Family | Candidate generator variable(s) | Already an emitted feature? | Verdict |
|---|---|---|---|
| **Supplier** | `own_stress()`/`stress()`, `IDIO`, `RESILIENCE` | partially | **LIVE** — carries `supply_stress` + `recovery_capability` forward; `supplier_reliability` closed (cited) |
| **Logistics** | `recv_atten()`/`SUP_ATTEN`, `PORT_EVENTS` bump, carrier timing | carrier timing **yes** | **CLOSED** — not-simulated (as an independent per-entity state) / already-a-feature |
| **Inventory** | `ip["stock"]`, `ip["thr"]` | **YES** | **CLOSED** — already-a-feature (re-checked, unchanged since Phase 1) |
| **Production** | `FACTORY_OUTAGE` | capacity quantities **yes** | **CLOSED** — not-simulated at any observed `t0` / too-sparse / already-a-feature |

### Supplier — LIVE, cited not remeasured

Per the brief, no supplier state was re-run. `own_stress`/`stress`, `IDIO` and `RESILIENCE` were
confirmed still present, still latent, and still where Phase 1 left them.

- **Supply Stress** — confirmed, passes. AUC 0.6426 (h², A) / 0.6509 (h¹, E),
  `phase1_latent_state.md`; identifiable under `do(Z)`, Case 1, `layer3_uncertainty_aware.md` §2.
- **Recovery Capability** — confirmed, passes, **Mechanism-E variants only**. AUC 0.6530 (h¹, E);
  identifiable, Case 1.
- **Supplier Reliability** — confirmed, **fails identifiability**. `do(IDIO)` moves 1.0% / 0.6% of
  labelled rows; classifier AUC 0.5000 / 0.4997 against a *measured* null of 0.5040 / 0.5079.
  **Case 3, classification C, closed** — cited from `layer3_uncertainty_aware.md` §2, not rerun.

### Logistics — CLOSED at Phase 0. This is the phase's real new work.

Four independent findings, each measured:

1. **`recv_atten()` is causally inert on both testbeds.** `F_ON = False` on Variants A and E, so
   `recv_atten()` returns exactly `1.0` for every supplier. On Variant A the `SUP_ATTEN` dict is
   nonetheless *populated* (800 entries, by Mechanism J's placeholder draw) and then never read —
   the dict exists and the state does not. On Variant E it is empty. A source reading would very
   plausibly have mistaken the populated dict for a live state.

2. **Where it *is* live, it is not an independent family.** On Variants F and K
   (`F_ON = True`), `attenuation_of()` is a piecewise-linear function of `RESILIENCE` with no
   second draw. Measured: **Spearman = −1.000000, n = 800**, and every distinct resilience maps to
   a distinct attenuation.

3. **And the identity holds at the level a head actually sees.** Phase 1 binarises every continuous
   state at its train median, so what a probe learns is the median-split label. Measured: the
   median-split label for `SUP_ATTEN` is the **exact complement** of `RESILIENCE`'s on
   **800/800 suppliers**, both variants. A Model A probe for "logistics stability" would therefore
   return Recovery Capability's own AUC with the labels flipped — a relabelling, not a new
   measurement. This is the finding that makes the closure decisive rather than merely arguable.

4. **The one genuinely logistics-shaped quantity is still unstored, and running it would break a
   ground rule anyway.** The `PORT_EVENTS` bump (`+0.10` to a sea shipment's stress) reaches
   **2.1–2.2%** of shipments and is evaluated inline in `new_shipment()` and never stored — Phase 1's
   `needs instrumentation` verdict, re-confirmed against the current generator. Separately, no
   frozen backbone checkpoint exists for Variant F or K (`out/ds_ckpt/` holds vA and vE only), so
   running Phase 1 on the live-`F` variants would require **training SHARE**, which the standing
   ground rule forbids.

Meanwhile carrier/route timing is squarely **already-a-feature**: `carrier_on_time_rate_90d` is a
Shipment input in `ml/data/loader.py::_shipment_features`, the same disqualification class as
Inventory Health.

> **Generator extension required:** record the per-shipment `PORT_EVENTS` stress contribution at
> the point it is applied (`db/generate_dataset.py:1080-1082`) — the same shape of change
> `reports/phase1_mitigation_level.md` made for `mitigation_level`, and inert by the same argument.

### Inventory — CLOSED, and re-checked rather than cited blind

The brief asks explicitly whether anything changed since Phase 1's finding, so this was verified
rather than assumed. `stock_level` and `reorder_threshold` are still emitted to **both**
`inventory.csv.gz` and `inventory_history.csv.gz`, and still consumed as model inputs —
`min_stock_ratio`, `avg_stock_ratio`, `total_stock`, `total_reorder_threshold`, plus
`stock_level`/`reorder_threshold` as the `STOCKED_AT` **edge attribute**. Unchanged. **already-a-feature.**

### Production — CLOSED, on a finding stronger than the cited one

`capacity_pressure` was already closed as too-sparse. The redesign proposes Production as a
possibly-different family, so it got its own check — and the check returns something the sparsity
argument did not capture:

| measurement | value |
|---|---|
| factories covered by `FACTORY_OUTAGE` | **1 of 5** |
| outage window | 2024-06-03 → 2024-06-24 (**21 days**) |
| shipments dispatched inside the window | **98 of 39,138 (0.25%)** |
| snapshot grid | 15 `t0`s, **2024-07-01** → 2025-09-01 |
| **`t0`s falling inside the outage window** | **0 of 15** |

**The outage ends five weeks before the first snapshot.** There is no `t0` at which this state is
active, so there is no supervision target to construct at any positive rate — this is
`not-simulated (at any observed t0)`, which is a stronger and different disposition from
`too-sparse`. Every other capacity quantity is a static emitted attribute and already a model
input.

> **Generator extension required:** a time-varying per-factory capacity/outage state whose windows
> intersect the snapshot grid. The present single fixed pre-grid window cannot be instrumented
> into a supervisable state — moving the window is a dataset change, not an instrumentation change.

### Phase 0 stop condition

**HIT, for two of four families.** Logistics and Production are closed **here**, before Phase 1,
and reported above as generator-extension requirements rather than carried forward as if live.
Inventory is closed as already-a-feature. Only the Supplier family survives, carrying exactly the
two states that already passed.

---

## PHASE 1 — Observability / identifiability gate

**Nothing was run, and that is the correct outcome rather than an omission.** The brief scopes this
phase precisely: *"The only new work is running this exact harness against whatever candidate
variable Phase 0 surfaced for Logistics (and Production, if Phase 0 didn't already close it)."*
Phase 0 closed both. Supply Stress, Recovery Capability and Supplier Reliability are explicitly
**not rerun** — their Step 2 results are cited directly.

**Cost check:** zero — no CRN run was launched, because there was no surviving candidate to launch
one against. `ml/identifiability_check.py` is untouched (digest `f1dc41d0…`).

| state | divergence rate | classifier AUC | **measured null** | floor | clears | sign 5/5 | **identifiable** |
|---|---|---|---|---|---|---|---|
| A / supply_stress | 0.582 | 0.6262 | 0.4911 | 0.0539 | YES | yes | **YES** |
| E / supply_stress | 0.305 | 0.5478 | 0.4908 | 0.0296 | YES | yes | **YES** |
| E / recovery_capability | 0.174 | 0.5392 | 0.4898 | 0.0297 | YES | yes | **YES** |
| A / supplier_reliability | 0.010 | 0.5000 | 0.5040 | 0.0909 | no | NO | **NO** |
| E / supplier_reliability | 0.006 | 0.4997 | 0.5079 | 0.0386 | no | NO | **NO** |

*All cited verbatim from `reports/layer3_uncertainty_aware.md` STEP 2A. Not remeasured.*

**The brief's own expectation held.** It stated plainly that the base rate on a newly-proposed
Logistics or Production candidate clearing this gate should be treated as genuinely uncertain, and
that a closed result is what the phase exists to catch cheaply. Both closed — and they closed one
phase *earlier* than budgeted, at 41 s of audit rather than a five-seed common-random-numbers
campaign per family. That is the gate structure working exactly as designed.

**Stop condition, per family: no family reached it.** Surviving set into Phase 2:
`{supply_stress (A, E), recovery_capability (E)}`.

---

## PHASE 2 — Progressive complexity, Model A → B → C

**Also entirely citation, for the same structural reason: no new family survived Phase 1, so there
is no new Model A to build.** Model B was already run on **every** surviving arm — which is what
this phase asks for — in `reports/phase_modelB_concat_baseline.md`.

**Cost check:** zero. `ml/modelB_concat_head.py` is untouched (digest `5d16b17e…`).

| variant / state | A depth | A AUC | B AUC | B's own floor | margin (B−A) | clears B's floor? |
|---|---|---|---|---|---|---|
| A / supply_stress | h² | 0.6426 | 0.6338 | 0.0304 | **−0.0087** | **NO** (B is *worse*) |
| E / supply_stress | h¹ | 0.6509 | 0.6961 | 0.0592 | **+0.0452** | **NO** — short by 0.0140 |
| E / recovery_capability | h¹ | 0.6530 | 0.6686 | 0.0222 | **+0.0156** | **NO** — short by 0.0066 |

**Stop condition: HIT on all three arms. Model C is not built** — no `ml/latent_state_depth_gate.py`
exists in this repository and none was created. The fixed depth is kept for every family, and that
is reported as the result rather than as an unfinished task.

Two points worth carrying forward, both from that report:

- The E / Supply Stress arm posts the largest mean gain anywhere (+0.0452, positive on **5/5**
  seeds) and **fails anyway, because its seed-to-seed spread grew faster than its mean did** —
  Model B is 2.34× *less* stable than Model A there. A mean-only comparison would have sent Model C
  into construction.
- Consequently the brief's guard against the Rung-5 gate collapse — *score `g_s` before any pooling
  step, never mean-pool a node's per-depth tokens* — **was never reached**. It is recorded here so
  that a future session that does open the Model C gate inherits the instruction rather than
  rediscovering the failure.

---

## PHASE 3 — Split state estimate from confidence, assign epistemic status

**What was built.** `ml/evidence_status.py`. Two things happen in it, and only the second is a
thin translation layer.

**(1) The calibration ladder is completed for Recovery Capability — measured here for the first
time.** The original pipeline ran Steps 3–5 for Supply Stress only: Recovery Capability was
excluded at STEP 1 by that ladder's ordering, on a downstream-sufficiency test against `impact`,
*before its calibration was ever measured*. The redesign decouples those — sufficiency is Phase 4's
question, not a gate on Phase 3 — so this is genuinely new. `step3`, `step3_diagnostics` and
`step4` are imported from `ml/layer3_uncertainty.py` **unmodified**, which means Supply Stress's
arms double as a reproduction check on the instrument.

**(2) The confidence output is built as a separate head, which is new.** `C_s = g_s(H, X)`, fit on
`va` against the state head's **out-of-sample** correctness there and scored on `te` — the same
three-way discipline `ml/layer3_sufficiency.py` uses. Fitting it on `tr` would feed it in-sample
correctness, a leak in the only direction that flatters it.

**Cost check.** First arm timed at **53 s** before the other two were launched; all three ran in
**2 min 06 s**. Grid build was timed at **13.6 s / world** before the 5-world build was committed to.

### The instrument reproduces its published figures exactly

| arm | ECE improvement | member floor | folds improved | matches `layer3_uncertainty_aware.md` §3? |
|---|---|---|---|---|
| A / supply_stress | **−0.0054** | **0.0132** | **7/20** | **yes, exactly** |
| E / supply_stress | **−0.0029** | **0.0117** | **8/20** | **yes, exactly** |
| E / recovery_capability | **+0.0020** | **0.0140** | **12/20** | **new — never measured before** |

Recovery Capability's five per-seed state-head AUCs also came back as
`[0.6749, 0.6510, 0.6439, 0.6496, 0.6455]` — `phase1_latent_state.md`'s recorded values to four
decimals, an unforced Model A equivalence check.

**Recovery Capability's calibration fails in a *different shape* from Supply Stress's**, and the
distinction is worth naming. Supply Stress's cross-world map makes ECE **worse** (−0.0054 / −0.0029):
the map actively harms. Recovery Capability's makes it **slightly better** (+0.0020) and simply
does not clear its floor, improving on 12/20 folds rather than 7–8/20. Its miscalibration is also
the largest in the project — raw ECE **0.0419, at 2.42× its binomial noise floor** — and
within-world isotonic removes essentially all of it (0.0419 → **0.0067**, +0.0352, the largest
within-world recovery recorded here). So the machinery works, the score is fixable, and the map
still does not transfer. Same wall, approached from a different side.

*Within-world figures are optimistically biased by construction and are reported as an upper bound,
never as a gate — the label Step 3 attached to them, which governs here too.*

### The world-conditioned-calibration follow-up was checked, per the brief

The brief instructs that if the world-conditioned calibration work has been completed and
validated, its deployable observable-proxy-conditioned map should be substituted for the plain
cross-world isotonic map. **It has been completed (`reports/world_conditioned_calibration.md`,
2026-08-15) and its verdict is that no such deployable map exists**: 7b's conditioned arms add
nothing over the same worlds pooled *without* conditioning on Variant A (−0.0021 against a floor of
0.0022) and are measurably harmful on Variant E. That report's own "what must not be built" section
forbids adopting it. **No substitution was made, and the ceiling is not fixed.**

### The confidence head — the phase's genuine positive, with its qualification

| arm | `C_s = g_s(H,X)` AUC vs correctness | `margin` control | `ens_var` (Step 4's signal) | floor | `C_s` − margin |
|---|---|---|---|---|---|
| A / supply_stress | 0.5819 | 0.5918 | **0.4942** | 0.0594 | −0.0099 |
| E / supply_stress | 0.6069 | 0.5968 | **0.4868** | 0.0452 | +0.0101 |
| E / recovery_capability | **0.6419** | 0.6095 | **0.4893** | 0.0415 | +0.0324 |

**Monotonicity — bins ordered least-confident-first, pooled over 5 worlds:**

| signal | A / stress | E / stress | E / resilience |
|---|---|---|---|
| `C_s` pooled Brier spread (least − most confident) | **+0.0649** | **+0.0647** | **+0.0900** |
| its member floor | 0.0047 | 0.0059 | 0.0052 |
| **monotone, and clears?** | **YES (13.8×)** | **YES (11.0×)** | **YES (17.3×)** |
| worlds monotone | 3/5 | 4/5 | **5/5** |
| *Step 4's `ens_var`, for contrast* | −0.0021 (floor 0.0079) | −0.0022 (floor 0.0097) | −0.0013 (floor 0.0133) |
| *`ens_var` worlds monotone* | **0/5** | **0/5** | **0/5** |

Recovery Capability's pooled bin Briers run `0.2668 → 0.2519 → 0.2391 → 0.2191 → 0.1769` — monotone
across all five bins with a clean 0.09 spread.

**This overturns a `STOP — G` on its stated grounds, and the correction belongs on the record.**
`layer3_uncertainty_aware.md` STEP 4 concluded *"Head-init-seed variance is not a usable confidence
signal for Layer 3"* and classified confidence as **G**. That conclusion was correct **about
init-seed variance** — reproduced here at 0/5 worlds monotone on all three arms — and it was
over-generalised to per-entity confidence as such. A learned head reading the same frozen
representation orders the state head's errors cleanly and reproducibly. The route was wrong, not
the question.

**And the qualification that keeps it from being a fix.** `C_s` never beats the free `|p − 0.5|`
margin by more than the floor (−0.0099 / +0.0101 / +0.0324 against floors of 0.0594 / 0.0452 /
0.0415). The margin control is itself monotone on all three arms. So the deployable confidence
signal is the **margin**, not a learned head that costs parameters and buys nothing measurable —
and Phase 5's interface carries the margin for exactly that reason. Reporting the head as the win
here would be claiming a result its own control refutes.

Confidence-head *calibration* transfers on one arm and not the others (A: **+0.0164** vs floor
0.0092, **20/20** folds; E/stress: +0.0077 vs 0.0099, 17/20; E/resilience: +0.0014 vs 0.0189,
11/20). Not sign-consistent across variants, so it does not clear this project's bar.

### Status assignment

| family | state calibration | confidence calibration | confidence monotone | **STATUS** |
|---|---|---|---|---|
| **Supply Stress** (A) | **STOP** (−0.0054 / 0.0132) | clears | yes | **Uncertain** |
| **Supply Stress** (E) | **STOP** (−0.0029 / 0.0117) | **STOP** | yes | **Uncertain** |
| **Recovery Capability** (E) | **STOP** (+0.0020 / 0.0140) | **STOP** | yes | **Uncertain** |
| Supplier Reliability | — | — | — | **Unidentifiable** (Phase 1, cited) |
| Logistics | — | — | — | **Unidentifiable** (Phase 0) |
| Inventory | — | — | — | **Unidentifiable** (Phase 0) |
| Production | — | — | — | **Unidentifiable** (Phase 0) |

**Stop condition: none, by design — and none was needed.** Every family carries a status. The
brief's stated expectation — *"the default expectation for this phase, absent new evidence, is that
both land as Uncertain, not Supported"* — held, and now covers Recovery Capability too, on evidence
that did not previously exist. `Uncertain` is doing the work it was introduced to do: it labels the
failure rather than hiding it inside a binary pass/fail.

---

## PHASE 4 — State-to-task sufficiency matrix

**What was built.** `ml/state_task_sufficiency.py`. For each (family, task) cell it fits
`P(Y_t|X)`, `P(Y_t|Z_s)` and `P(Y_t|X,Z_s)` and computes `Δ_{s,t} = P(Y|X,Z) − P(Y|X)` against a
floor measured for that specific task.

**Propagating a per-supplier state to a non-Supplier task.** `impact` is the only task labelled on
Supplier — which is exactly why STEP 1 measured that column and no other. `delay` (Shipment) and
`shortage` (Product) required carrying `Z_s` along the graph's own edges:

    delay      Shipment --SHIPS_FROM--> Supplier
    shortage   Product <--USED_IN-- Component <--SUPPLIES-- Supplier

**The aggregator was pre-registered as `max` before any cell was fitted**, because the generator's
own `sourcing_stress()` reduces a product's component sources with `max` on its AND branch —
worst-case sourcing is the generator's semantics for how supplier state reaches a product. `mean`
was computed alongside as a secondary and **changes no verdict in any cell** (0 of 6 fresh cells
reclassify; largest divergence between the two deltas: 0.0008). Choosing the aggregator after seeing which scored better would have
been the goalpost-move this project's standard exists to prevent.

All three arms are scored on the **same** row set — the rows where `Z` is defined — because scoring
`P(Y|X)` on a larger population than `P(Y|X,Z)` would make the delta a comparison between two
different tasks. Coverage is reported per cell: `delay` **72.6–72.7%** (a shipment dispatched from a
factory rather than a supplier has no upstream supplier), `shortage` and `impact` **100%**.

**Cost check.** One cell timed at **19 s** before the grid was launched; the full
3 arms × 2 tasks × 5 dataset seeds × 5 init seeds grid ran in **~8 min**.

### The completed matrix

| family | task | cov | `P(Y\|Z)` | `P(Y\|X)` | `P(Y\|X,Z)` | **Δ** | Δ floor | Δ sign 5/5 | **CLASS** | cited |
|---|---|---|---|---|---|---|---|---|---|---|
| Supply Stress (A) | **impact** | — | 0.7075 | 0.7832 | 0.7914 | **+0.0082** | 0.0485 | — | **Redundant** | **YES** |
| Supply Stress (A) | delay | 72.7% | 0.5615 | 0.7568 | 0.7656 | +0.0087 | 0.0535 | NO | **Unsupported** | no |
| Supply Stress (A) | shortage | 100% | 0.5359 | 0.8684 | 0.8691 | +0.0007 | 0.0333 | NO | **Unsupported** | no |
| Supply Stress (E) | **impact** | — | 0.7669 | 0.8124 | 0.8209 | **+0.0084** | 0.0676 | — | **Redundant** | **YES** |
| Supply Stress (E) | delay | 72.6% | 0.5436 | 0.7116 | 0.7144 | +0.0028 | 0.0900 | NO | **Unsupported** | no |
| Supply Stress (E) | shortage | 100% | 0.5368 | 0.8856 | 0.8861 | +0.0005 | 0.0387 | NO | **Unsupported** | no |
| Recovery Capability (E) | **impact** | — | 0.5711 | 0.8124 | 0.8090 | **−0.0034** | 0.0615 | — | **Unsupported** † | **YES** |
| Recovery Capability (E) | delay | 72.6% | 0.5648 | 0.7116 | 0.7125 | +0.0009 | 0.1133 | NO | **Unsupported** | no |
| Recovery Capability (E) | shortage | 100% | 0.5271 | 0.8856 | 0.8852 | −0.0004 | 0.0387 | NO | **Unsupported** | no |

† *The prompt directs this cell be entered as `Redundant`; the architecture's own rule gives
`Unsupported`, because `Redundant` requires that Z alone predicts Y and `P(Y|Z) = 0.5711` does not
clear its 0.1391 floor. The rule was applied. See "Two documentation defects", item 2. The
difference changes no downstream decision — neither class is wired into a risk head.*

**Supported column: empty. 2 Redundant, 7 Unsupported, 0 Supported.**

The three `impact` cells are **cited, not remeasured**, exactly as instructed. The six fresh cells
share one mechanism: **`P(Y|Z)` fails to clear its own floor on every one of them.** Supply Stress
reaches `P(Y|Z) = 0.7075 / 0.7669` on `impact` — a supplier-level state predicting a
supplier-level label — but only **0.5359–0.5648** once propagated to Shipment or Product rows.
Aggregating a per-supplier scalar up a sourcing edge discards most of what made it informative,
and the numbers say so directly rather than by inference.

### A single-seed false positive, caught in-phase

The cost-check run — one cell, dataset seed 42 only — returned Supply Stress (A) × `delay` at
`Δ = +0.0299` against a floor of 0.0240: **"Supported"**. Across five seeds the per-seed deltas are

```
seed:      42        43        44        45        46
delta:  +0.0299   +0.0163   -0.0099   +0.0003   +0.0071      mean +0.0087, floor 0.0535
```

— not sign-consistent, nowhere near its floor, and **Unsupported**. With one seed the floor is the
init-seed spread alone, which `docs/14_Project_Roadmap.md` §4 risk 4 records as the *smaller* of
the two variance sources. This is a live, in-phase repetition of the Supplier Reliability
single-seed pilot (0.5444 → 0.4763) that `phase1_latent_state.md` recorded, and it is the second
time the five-seed rule has caught the same class of error in this project.

**Stop condition: none, by design.** The completed matrix is the deliverable, and its sparse
Supported column is the finding.

---

## PHASE 5 — Hard Layer 3 → Layer 4 interface, rewire the risk heads

**What was built.** `ml/layer3_interface.py`. Each family is packaged as
`O_s = (Z_s, P_s [Supported only], C_s, status)`, with `P_s` **structurally absent** for a
non-Supported family — `assemble()` does not create the key, so a consumer reaching for a
calibrated probability on an `Uncertain` family gets a `KeyError` rather than a plausible-looking
uncalibrated number. That is the failure mode the hard interface exists to make impossible.

`C_s` is carried as the `|p − 0.5|` margin rather than the learned head, applying Phase 3's own
measurement: the learned head does not beat the margin by more than its floor, so the interface
ships the control.

**The wiring rule**, enforced by `wire_for_task()`: a `Supported` cell's `Z_s` reaches the risk
head; an `Uncertain` family's `Z_s` may travel **only alongside its `C_s`**, never as a clean
number; `Redundant`, `Unsupported` and `Unidentifiable` never reach a head at all.

**Cost check.** One seed × one init seed × three tasks timed at **22 s** before the grid launched;
both variants' full 5 × 5 ablation grids ran in **~16 min**.

**Refit floor.** Two identically-configured reruns of the full Phase 5 pipeline: `max|dev| =
0.0000000000` across all ten arms. Reported as a **fact, not a result** — `fit_probs` is seeded and
the backbone loads from cache, so bit-exactness is a property of the setup. The operative floors
are the init-seed and dataset-seed spreads, exactly as Step 3 gated on the member floor rather than
its own exactly-zero fit floor.

### The consequence of an empty Supported column, stated before the tables

`state_matrix_wired` wires **no columns**, because the matrix has no `Supported` cell. It is
therefore not merely similar to the `markov_readout` arm — it **is** that arm, the identical
feature block fitted at the identical init seed. Its gain is `+0.0000` on all six (task × variant)
combinations by construction, not by measurement. The informative arm is `state_all_wired`, which
wires every family in **regardless** of its cell's classification, and which shows what the matrix
is actually costing by excluding them.

### Variant A

| arm | delay | shortage | impact |
|---|---|---|---|
| `tabular_no_share` | 0.7568 | **0.8684** | 0.7832 |
| `depth_h0` | 0.7678 | 0.7883 | 0.7714 |
| `depth_h1` | 0.7786 | 0.7814 | 0.8002 |
| `depth_h2` | **0.7843** | 0.7739 | 0.7911 |
| `depth_h3` | 0.7806 | 0.7712 | 0.9211 |
| `depth_h4` | 0.7817 | 0.7737 | **0.9295** |
| `markov_readout` (h¹/h³/h⁴) | 0.7786 | 0.7712 | 0.9295 |
| `state_all_wired` | 0.7808 | 0.7718 | 0.9298 |
| **`state_matrix_wired`** | **0.7786** | **0.7712** | **0.9295** |
| `markov_frozen_production` | 0.7671 | 0.7832 | 0.9290 |

### Variant E

| arm | delay | shortage | impact |
|---|---|---|---|
| `tabular_no_share` | 0.7116 | **0.8856** | 0.8124 |
| `depth_h0` | **0.7495** | 0.8028 | 0.7904 |
| `depth_h1` | 0.7280 | 0.7922 | 0.8314 |
| `depth_h2` | 0.7367 | 0.7927 | 0.8272 |
| `depth_h3` | 0.7337 | 0.7715 | 0.8773 |
| `depth_h4` | 0.7367 | 0.7710 | **0.9114** |
| `markov_readout` (h¹/h³/h⁴) | 0.7280 | 0.7715 | 0.9114 |
| `state_all_wired` | 0.7292 | 0.7737 | 0.9121 |
| **`state_matrix_wired`** | **0.7280** | **0.7715** | **0.9114** |
| `markov_frozen_production` | 0.8090 | 0.7931 | 0.9221 |

### The credited comparisons — metric fixed in advance

Test ROC-AUC, pooled over 5 dataset seeds; an arm is credited only if the mean gain exceeds
`max(init-seed spread, dataset-seed spread)` **and** is sign-consistent across all five seeds.

| comparison | A/delay | A/shortage | A/impact | E/delay | E/shortage | E/impact |
|---|---|---|---|---|---|---|
| `matrix_wired` − `markov_readout` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `all_wired` − `markov_readout` | +0.0022 | +0.0005 | +0.0002 | +0.0012 | +0.0022 | +0.0007 |
| *its floor* | 0.1119 | 0.0206 | 0.0084 | 0.1258 | 0.0777 | 0.0425 |
| **`matrix_wired` − Markov production** | +0.0115 | −0.0119 | +0.0005 | **−0.0810** | −0.0216 | −0.0107 |
| **credited?** | **no** | **no** | **no** | **no** | **no** | **no** |
| `markov_readout` − `tabular_no_share` | +0.0218 | **−0.0972** | **+0.1463 ✓** | +0.0164 | **−0.1141** | **+0.0990 ✓** |

**Stop condition: HIT. Matrix-validated wiring does not beat the Markov baseline on any task, on
either variant.** Per the brief, Layer 5 output formatting is **not** built on top of this result.

Three readings the ablation set supports, none of which required Layer 3 to work:

1. **Wiring every family in regardless buys nothing either.** `state_all_wired` gains
   +0.0002…+0.0022 against floors of 0.0084…0.1258 — one to two orders of magnitude inside them on
   every arm. So the empty Supported column is not costing a real improvement that a laxer matrix
   would have captured. **The matrix is excluding columns that carry nothing**, which is the most
   favourable possible reading of a gate that admitted nothing, and it is the measured one.
2. **SHARE earns its place on `impact` and loses it on `shortage`.** `markov_readout` beats the
   raw-feature baseline by **+0.1463 (A)** and **+0.0990 (E)** on impact — credited on both. On
   `shortage` it is **−0.0972 / −0.1141**: every SHARE depth is beaten by the Product node's own
   observable features, and `depth_h0` (the raw projection, no message passing) is the best encoder
   depth there. That is a Layer-1/Layer-2 finding this Layer-3 ablation surfaced incidentally, and
   it is flagged rather than buried — it is not this build's to act on.
3. **The footing caveat is load-bearing on `E/delay`.** `markov_frozen_production` (0.8090) beats
   every probe arm there by a wide margin, while on `A/delay` the probes beat it. The production
   number is an end-to-end model trained on `tr` for 100 epochs; the probe arms are small heads fit
   on `va`. Those are different objects on different budgets. The like-for-like question — *does
   matrix-validated wiring help?* — is `state_matrix_wired` vs `markov_readout`, and that answer is
   `+0.0000` everywhere.

---

## Deferred, explicitly out of scope and not built

- **Layer 5** (evidence cards, decision support, SCM/scenario guardrails) — **not built**, and
  Phase 5's stop condition independently forbids building it on this result.
- **Global state fusion** — **not built**, per the architecture's own deferral. Phase 5's per-task
  direct wiring was not shown insufficient on a task that a family's states could explain; it was
  shown to have nothing to wire, which is a different condition and does not trigger the fusion
  experiment.
- **Model C** — **not built**, gated at Phase 2 and the gate did not open.

---

## What must not be built on this evidence

- **No Layer 5**, per Phase 5's stop condition.
- **No Model C**, per Phase 2's — including for Recovery Capability, whose Phase 3 confidence
  result is the strongest number in this build and is not evidence about depth fusion.
- **No logistics or production state head on this generator.** Both closures are structural
  (a deterministic relabelling of an existing state; a window that misses the snapshot grid
  entirely), not sample-size problems, and neither is fixable by more seeds or more capacity.
- **No learned confidence head in the deployed interface.** It does not beat `|p − 0.5|` by more
  than its floor on any arm; the interface ships the margin.
- **No within-world calibration fit**, which remains the one change that would make Phase 3 "pass"
  and remains leakage.
- **No treatment of `Uncertain` as a soft pass.** No `Uncertain` family's `Z_s` reached a risk head
  in any credited arm, and the interface structurally withholds `P_s` from all three.

## The defensible follow-ups this build supports

1. **Instrument the `PORT_EVENTS` per-shipment contribution** (`db/generate_dataset.py:1080-1082`).
   It is the one Logistics candidate that is a real unstored latent rather than a relabelling, the
   change is the same shape as the already-validated `MITIGATION_HISTORY` recorder, and Phase 0
   closed it *only* for want of a stored variable.
2. **Move or multiply `FACTORY_OUTAGE` so its window intersects the snapshot grid.** Production is
   currently unaskable rather than answered — 0/15 snapshots — and this is a dataset change of a
   few lines that would make the question well-posed for the first time.
3. **Re-ask Phase 4's `delay` and `shortage` columns with an entity-level state**, not a propagated
   supplier scalar. Every fresh cell failed at the same point — `P(Y|Z)` collapsing from ~0.77 to
   ~0.55 under aggregation — which is a statement about the propagation, not about the state.
4. **More dataset seeds, for the calibration transfer question**, carried forward unchanged from
   `layer3_uncertainty_aware.md` and `world_conditioned_calibration.md`. Five worlds remains the
   binding constraint on every calibration negative here, including Recovery Capability's new one.

**Every negative in this report is a non-detection at this scale, not an established absence.** The
`v1` preset, five dataset seeds, and — for Phase 3 — the head-init-seed axis standing in for the
backbone-seed axis are the binding constraints, and they are the same ones every prior result in
this project carries.

---

## Integrity — SHARE, the Markov readout, and Model A's heads are untouched

**Ground rule:** SHARE and the Layer-2 Markov readout are not retrained or modified by any phase
above. Confirmed by `git diff` at the close of each phase, as required.

```
$ git diff --stat                     # EMPTY — no tracked file modified
$ git status --porcelain db/          # EMPTY — the generator was executed, never edited
$ git status --porcelain
?? ml/evidence_status.py              <- new, this session (Phase 3)
?? ml/inventory_state_families.py     <- new, this session (Phase 0)
?? ml/layer3_interface.py             <- new, this session (Phase 5)
?? ml/state_task_sufficiency.py       <- new, this session (Phase 4)
```

Every file this session touched is **new and untracked**. The reused modules carry digests
identical to those recorded at the close of `reports/world_conditioned_calibration.md`:

```
c2e15e39cdc101d4...  ml/hypothesis_ranker.py                MATCHES
24ccd9d58eb44b69...  ml/uncertainty_calibrate.py            MATCHES
3a8c888040446aa2...  ml/uncertainty_ensemble.py             MATCHES
8e4a0803d8f09cd0...  ml/latent_state_head.py                MATCHES
689962edd1315599...  ml/ds_backbone.py                      MATCHES
4e9a4db9d2d4ef3a...  ml/data/loader.py                      MATCHES
1b85a68f9e9705be...  ml/models/rgcn_attn_encoder.py         MATCHES
9e7919e6bfcf39ca...  ml/models/rgcn_attn_markov_encoder.py  MATCHES
cbd40aadc8570745...  ml/models/heads.py                     MATCHES
8f669a1683117280...  ml/models/encoder.py                   MATCHES
8fa14bc270946507...  ml/models/depth.py                     MATCHES
4cf67f7d4f25b3a8...  ml/models/model.py                     MATCHES
e25b3357ac81a896...  ml/layer3_uncertainty.py               MATCHES
f1dc41d01a2cd3a8...  ml/identifiability_check.py            unmodified (Phase 1 ran nothing)
5d16b17e235702aa...  ml/modelB_concat_head.py               unmodified (Phase 2 ran nothing)
```

`ml/latent_state_head.py`'s digest `8e4a0803d8f09cd0…` matches the value recorded at the close of
`phase_modelB_concat_baseline.md` and `layer3_uncertainty_aware.md`, so Model A's methodology is
the same code that produced Phase 1's numbers. Phase 3's `align_keep()` deliberately *reproduces*
`align`'s keep-mask rather than editing `latent_state_head.py` to expose it, precisely to preserve
that digest.

All ten backbone checkpoints in `out/ds_ckpt/` carry mtimes of **2026-08-13 22:03–22:44**,
predating this session (2026-08-19) and every prior one. Every fit loaded from cache; **no backbone
was trained.** Backbone health on first load matched the recorded values exactly
(`delay=0.8046 shortage=0.7835 impact=0.9340`, Variant A seed 42; `delay=0.8397 shortage=0.7929
impact=0.9435`, Variant E seed 42).

`assert_backbone_frozen()` ran after head construction and again after training on **every** head
fit in this session and never fired. Phase 0 performed no fits at all (it never loads a
checkpoint); Phases 1 and 2 performed none (nothing was run).

---

## Reproduction

```
$ venv/bin/python -u ml/inventory_state_families.py --variants A,E,F,K --seed 42 --config v1 \
      --out out/layer3_redesign/phase0_families.json

$ venv/bin/python -u ml/evidence_status.py \
      --arms A:supply_stress,E:supply_stress,E:recovery_capability \
      --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --out out/layer3_redesign/phase3_status.json

$ venv/bin/python -u ml/state_task_sufficiency.py \
      --arms A:supply_stress,E:supply_stress,E:recovery_capability --tasks delay,shortage \
      --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --out out/layer3_redesign/phase4_matrix.json

$ venv/bin/python -u ml/layer3_interface.py --variant A --states supply_stress \
      --tasks delay,shortage,impact --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --out out/layer3_redesign/phase5_ablation_A.json
$ venv/bin/python -u ml/layer3_interface.py --variant E --states supply_stress,recovery_capability \
      --tasks delay,shortage,impact --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 \
      --out out/layer3_redesign/phase5_ablation_E.json
```

Raw results: `out/layer3_redesign/phase0_families.json`, `phase3_status.json`,
`phase4_matrix.json`, `phase5_ablation_{A,E}.json`; logs `log_phase4.txt`,
`log_phase5_{A,E}.txt`.
Cached prediction grids: `out/wcc/grid_{A,E}_supply_stress.npz`,
`out/wcc/grid_E_recovery_capability.npz` (new this session).
Confidence-head inputs: `out/layer3_redesign/confinputs_{variant}_{state}.npz`.
Cited comparison numbers: `out/phase1/heads_{A,E}.json`, `out/layer3/step1_{A,E}.json`,
`out/layer3/step2_{A,E}.json`, `out/layer3/step345_{A,E}.json`, `out/modelB/concat_{A,E}.json`.
