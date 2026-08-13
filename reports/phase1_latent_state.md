# Phase 1 — Latent State Estimation (HADES V3, Layer 3)

**Date:** 2026-08-13
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisite:** `reports/phase0_audit.md` — all five checks pass; Phase 1 cleared to start.
**Scope:** estimate hidden operational **state**, never hidden supplier **identity**. The latter
is closed (`findings/evolution.md` §5) and nothing in this phase reopens it.

**New code, all V3-original — no inherited file was modified:**
`ml/confirm_latent_states.py` (Step 1) and `ml/latent_state_head.py` (Steps 2–4).

---

## Step 1 — Ground-truth confirmation for all six candidate states

**Command:**

```
$ venv/bin/python ml/confirm_latent_states.py --variant A --seed 42 --config v1 \
      --out out/phase1/states_A_42_v1.json          # and --variant E, --variant K
```

**Method.** Not a source reading. The generator is executed in-process with `write()` stubbed
and its namespace inspected directly — the mechanism `ml/extract_hidden_state.py` and
`ml/extract_mechanism_state.py` already use, and the confirmation discipline
`reports/layer3_testing.md` §9.1/§10.1 applied to `HP_GROUPS` and `RESILIENCE`. The generator
ends with `raise SystemExit(1 if fail else 0)` (`db/generate_dataset.py:2318`); the script catches
it but **checks the code** and refuses to report ground truth from a world whose own validation
suite failed, rather than swallowing it the way a bare `except SystemExit: pass` would.

### Verdicts

| # | Candidate state | Verdict | Generator symbol | Latent? |
|---|---|---|---|---|
| 1 | **Supply Stress** | ✅ **confirmed — built** | `own_stress()` / `stress()` (`:827`, `:860`) | yes |
| 2 | **Recovery Capability** | ✅ **confirmed — built** (E-variants only) | `RESILIENCE[sup_id]` (`:518`) | yes |
| 3 | **Supplier Reliability** | ✅ **confirmed — built** | `IDIO[sup_id]` (`:727-733`) | yes (see caveat) |
| 4 | **Inventory Health** | ❌ **dropped** | `ip["stock"]`, `ip["thr"]` (`:1240-1245`) | **no — emitted** |
| 5 | **Logistics Stability** | ⚠️ **needs instrumentation** | `PORT_EVENTS` window test (`:1080-1082`) | partially |
| 6 | **Capacity Pressure** | ❌ **dropped** | `capacity_score` etc. (`:334`, `:984`) | **no — emitted** |
| — | *(found, not proposed)* **Mitigation Level** | ⚠️ **needs instrumentation** | `mitigation_level()` (`:1191`) | yes |

### Reasoning per state

**1. Supply Stress — confirmed.** `own_stress(sup_id, base_rel, t)` and `stress(...)` are the
generator's own description of "latent supplier stress in [0,1] at time t — the causal driver of
everything" (`:861`). Never emitted; both names appear in
`ml/data/loader.py`'s `HIDDEN_STATE_COLUMNS` guard. Verified re-evaluable post-hoc: they are pure
functions of module state that survives the run (`EVENTS`, `HP_B_EVENTS`, `SHOCK_EVENTS`, `IDIO`,
`coparents`, `HP_GROUPS`, `SUP_CHAIN`), so targets can be materialised at every snapshot `t0`
without re-instrumenting the generator. Confirmed by evaluation, not by inspection.

**2. Recovery Capability — confirmed, but not on the prescribed testbed.** `RESILIENCE[sup_id]`,
static per supplier, commented `NEVER emitted` at the point of definition. **It requires Mechanism
E, and Variant A is `J + A`** — so on the testbed `docs/14_Project_Roadmap.md` §3.1 Step 3
prescribes, `RESILIENCE` is an empty dict and this state does not exist:

```
variant A seed 42  mechanisms=['A', 'J']
recovery_capability      unavailable_on_this_variant
variant E seed 42  mechanisms=['E']
recovery_capability      confirmed
```

This is reported as `unavailable_on_this_variant`, deliberately **not** as `dropped` — the state is
real and confirmed, it simply is not present on that variant. Consequence: the one state carrying a
measured precedent (AUC ≈ 0.599) cannot be tested on Variant A at all. Phase 1 therefore runs a
second testbed, **Variant E**, for this state. Flagged as a roadmap defect below.

**3. Supplier Reliability — confirmed, with the level/state distinction made explicit.** The
reliability *level* is **not** latent: `base_rel` is emitted verbatim as
`suppliers.reliability_history` (`db/generate_dataset.py:1407`). What *is* latent is the
idiosyncratic outage **state** — `IDIO[sup_id] = (start, peak, end, magnitude)`, drawn for 25% of
suppliers and never emitted. The head therefore estimates "is this supplier inside an
idiosyncratic outage window at `t0`", a genuine hidden binary state, not the observable level.
Two honest caveats: `IDIO` is *not* on `HIDDEN_STATE_COLUMNS` (that list names stress/resilience
quantities, not this one), and `reliability_history` is emitted-but-unused —
`ml/data/loader.py:169` reads only `id, country, capacity_score, lead_time_days`.

**4. Inventory Health — dropped, and the reason is disqualifying.** `stock_level` and
`reorder_threshold` are emitted to both `inventory.csv` and `inventory_history.csv`
(`:1441`, `:1443`) **and are already consumed as model input features**
(`ml/data/loader.py:237-238`, `:324-336`, `:409`, including a `stock/threshold` ratio). An
"estimator" for this state would be reading its own target through the encoder. The only genuinely
hidden component is Mechanism G's lag between true and recorded stock — and **G is not enabled on
Variant A**, so even that is absent here. Not a latent state on this benchmark.

**5. Logistics Stability — needs instrumentation.** A real latent disturbance exists: port events
add `+0.10` to a sea shipment's stress. But it is evaluated **inline inside `new_shipment()` and
never stored** (`:1080-1082`), so there is no per-entity, per-`t0` variable to read out — unlike
stress, it cannot be recovered post-hoc. Its driver is also observable-adjacent: `H_PORT`
membership is exactly the `sea` flag, a deterministic function of the emitted `country` column —
the confound `layer3_testing.md` §9.7.1 already measured. Building a target needs a small, scoped
generator change recording the per-shipment port-event contribution. **Not built.**

**6. Capacity Pressure — dropped.** No time-varying latent capacity variable exists at all. Every
capacity quantity is a **static attribute emitted to CSV** (`suppliers.capacity_score`,
`factories.capacity_units_per_day`, `product_factories.capacity_units_per_day`,
`warehouses.capacity_units`), and `capacity_score` is already a model input feature
(`ml/data/loader.py:169`, `:279` as `capacity_score_z`). The single hidden capacity event is
`FACTORY_OUTAGE` — **one factory of five, for one fixed 21-day window** (`:1078`) — which is not a
supplier-level quantity and is far too sparse to supervise a per-supplier estimator. Dropped, not
deferred: there is nothing here to instrument that would make it a supplier state.

**Bonus — Mitigation Level, a latent state the init doc did not propose.**
`mitigation_level(sup_id, as_of)` (`:1191`) is a per-supplier, time-varying decision in [0,1]:
"how hard this supplier is currently working to protect its downstream customers", driven by
`RESILIENCE` and the supplier's own observed late rate. It is never emitted, and `mitigation` **is**
on `HIDDEN_STATE_COLUMNS` — the project already classifies it as hidden state. It is arguably the
*dynamic* form of Recovery Capability and a better Layer-3 target than the static draw.

It is nonetheless flagged **needs instrumentation, not built**, for a concrete reason:
`agent_observe()` **destructively drains** `agent_pending` as the weekly walk consumes it
(`:1180-1188`), so by the time the run ends the state it read is gone. Unlike `stress()`, calling
it post-hoc returns a value that is not what the simulation actually used. Capturing it requires
recording `mitigation_level` per supplier per week inside the generator — a small, well-scoped
change, and the single highest-value piece of new instrumentation this phase identified.

### Roadmap defect found by Step 1

`docs/14_Project_Roadmap.md` §3.1 Step 3 prescribes Variant A as the Phase 1 testbed, and §3.1
Step 1 simultaneously names `RESILIENCE` as one of the two states with confirmed ground truth and
cites its AUC ≈ 0.599 precedent as the reference point for the gate. **These are incompatible:
Variant A excludes Mechanism E, so `RESILIENCE` does not exist there.** Phase 1 resolves this by
running Variant A (as prescribed) *and* Variant E (for the one state that needs it), but the
roadmap text should be corrected.

---

## Steps 2–4 — Estimator heads, depth probe, and the reproduction floor

**Commands:**

```
$ venv/bin/python -u ml/latent_state_head.py --variant A --seeds 42,43,44,45,46 \
      --config v1 --init-seeds 0,1,2,3,4 --out out/phase1/heads_A.json
$ venv/bin/python -u ml/latent_state_head.py --variant E --seeds 42,43,44,45,46 \
      --config v1 --init-seeds 0,1,2,3,4 --out out/phase1/heads_E.json
```

### Design, and the three controls that make the numbers interpretable

**Frozen backbone, verified frozen.** Features are `layers[d]["Supplier"]` from the frozen
SHARE + Markov forward pass, extracted under `torch.no_grad()`. `get_backbone()` calls
`freeze()`; on top of that, `assert_backbone_frozen()` runs **after head construction and again
after training**, and additionally checks that no backbone parameter object reached the
optimizer. That is the failure `freeze()` structurally cannot catch (Phase 0, check 2) and it is
the exact risk this phase introduces by attaching a new trainable head. It never fired.

**Depth probe including h⁰.** Every state is estimated separately from h⁰…h⁴. h⁰ is the raw input
projection, so **AUC at h⁰ is what is available from input features without any message passing**.
Quoting only the best depth would let a state that is fully readable from raw features be reported
as a representation finding. This is the control `layer3_testing.md` used when it probed
hidden-parent identity at every depth including raw inputs.

**Two floors, gated on the larger.** `docs/14_Project_Roadmap.md` §4 risk 4 records that
dataset-seed variance dominates total AUC variance (68–99%). An init-seed-only floor — the
definition §1 used — therefore measures the *smaller* source. Both are reported; the gate uses
`max(init_seed_floor, dataset_seed_floor)`.

**Scale.** Run at the `v1` preset (800 suppliers, 15 snapshots) across **all five dataset seeds
42–46**, which `db/csv_v1scale/` holds fully populated. This is a deliberate choice: Phase 0
established that spec-scale `db/csv/` has only seeds 42–43 on disk, so the five-seed minimum the
gate requires is not satisfiable at spec scale without regeneration. Per `docs/14_Project_Roadmap.md`
§3.5, **a `v1`-preset result is a pilot, not a benchmark finding**; Phase 5 exists to re-run these
gates at `sup_n=4000`. Every number below carries that caveat.

Backbone health check, unchanged from V2's known values (seed 42):
`delay=0.8046  shortage=0.7835  impact=0.9340`.

### Variant A (`J + A`) — the prescribed testbed

5 dataset seeds × 5 init seeds. `pos/test` counts are pooled across the five seeds.

| state | depth | AUC | init floor | dataset floor | above .5 | clears floor | sign-consistent | pos/test |
|---|---|---|---|---|---|---|---|---|
| supply_stress | h⁰ | 0.6125 | 0.0222 | 0.0304 | +0.1125 | **YES** | yes | 11,075/20,538 |
| supply_stress | h¹ | 0.6288 | 0.0116 | 0.0158 | +0.1288 | **YES** | yes | 11,075/20,538 |
| supply_stress | **h²** | **0.6426** | 0.0066 | 0.0390 | **+0.1426** | **YES** | yes | 11,075/20,538 |
| supply_stress | h³ | 0.6222 | 0.0087 | 0.0304 | +0.1222 | **YES** | yes | 11,075/20,538 |
| supply_stress | h⁴ *(markov)* | 0.6034 | 0.0102 | 0.0150 | +0.1034 | **YES** | yes | 11,075/20,538 |
| supplier_reliability | h⁰ | 0.4530 | 0.0217 | 0.1161 | −0.0470 | no | **NO** | 240/20,538 |
| supplier_reliability | h¹ | 0.4400 | 0.0278 | 0.1515 | −0.0600 | no | **NO** | 240/20,538 |
| supplier_reliability | h² | 0.4763 | 0.0289 | 0.1710 | −0.0237 | no | **NO** | 240/20,538 |
| supplier_reliability | h³ | 0.4806 | 0.0359 | 0.1317 | −0.0194 | no | **NO** | 240/20,538 |
| supplier_reliability | h⁴ *(markov)* | 0.4870 | 0.0260 | 0.0439 | −0.0130 | no | **NO** | 240/20,538 |

**Supply Stress — clears its gate.** Mean AUC **0.6426** at h², above chance by +0.1426 against a
reproduction floor of 0.0390, and **sign-consistent across all five dataset seeds at every depth**:

```
h² per dataset seed: [0.6543, 0.6581, 0.6379, 0.6433, 0.6192]
```

For reference — not as a target — the Mechanism E resilience precedent is AUC ≈ 0.599. Supply
Stress sits above it. Two honest qualifications: that precedent was measured with hand-built
shipment-summary features in a 5-fold CV logistic regression
(`db/phase2_coverage_recheck.py:220-244`), **not** from a frozen SHARE representation, so the two
numbers are not the same experiment; and the comparison is a reference point, as the phase brief
requires.

**The encoder contributes almost nothing, and this must not be glossed.** h⁰ — the raw input
projection, before any message passing — already reaches **0.6125**. The lift from the best depth
over h⁰ is **+0.0301, which does not exceed the 0.0390 dataset-seed floor**. So the correct claim
is: *Supply Stress is recoverable from this benchmark's observable supplier features*, and it is
**not** established that SHARE's message passing adds anything to that recovery. A weaker claim
than "the frozen representation encodes supply stress," and the one the measurement supports.

**Supplier Reliability — fails its gate, and does not proceed.** Mean AUC is **below chance at
every depth** (0.4400–0.4870), not sign-consistent, and does not clear its floor. The per-seed
spread is the tell:

```
h² per dataset seed: [0.5453, 0.5130, 0.4901, 0.3743, 0.4588]
```

Two causes, both stated rather than optimised away. First, the positive class is tiny: **240
positives in 20,538 test rows (1.2%)** — `IDIO` gives 25% of suppliers a single ~25-day outage in a
21-month timeline, so very few are inside one at any snapshot `t0`. Second, the target is the
*outage state*, whose observable consequence is diluted through the same stress→delay conversion
that Supply Stress already captures. This head is not evidence of anything and is excluded from
Phase 2.

**A methodological note worth recording.** A single-seed pilot of this same head (dataset seed 42
only) returned AUC **0.5444** at h² — above chance and superficially reportable. Across five seeds
it collapses to 0.4763 and loses sign consistency. That single-seed reading would have been a
false positive, and it is a direct, in-phase demonstration of `docs/14_Project_Roadmap.md` §4 risk
4 ("any V3 result evaluated on a single dataset seed is not trustworthy on this benchmark").

### Variant E (Mechanism E) — for Recovery Capability

Run because Recovery Capability does not exist on Variant A (Step 1). Supply Stress and Supplier
Reliability are estimated here too, which makes Variant E an independent replication of both.

| state | depth | AUC | init floor | dataset floor | above .5 | clears floor | sign-consistent | pos/test |
|---|---|---|---|---|---|---|---|---|
| recovery_capability | h⁰ | 0.5784 | 0.0206 | 0.0176 | +0.0784 | **YES** | yes | 12,000/24,000 |
| recovery_capability | **h¹** | **0.6530** | 0.0204 | 0.0310 | **+0.1530** | **YES** | yes | 12,000/24,000 |
| recovery_capability | h² | 0.6002 | 0.0187 | 0.0322 | +0.1002 | **YES** | yes | 12,000/24,000 |
| recovery_capability | h³ | 0.5970 | 0.0142 | 0.0313 | +0.0970 | **YES** | yes | 12,000/24,000 |
| recovery_capability | h⁴ *(markov)* | 0.5627 | 0.0103 | 0.0231 | +0.0627 | **YES** | yes | 12,000/24,000 |
| supply_stress | h⁰ | 0.6174 | 0.0135 | 0.0176 | +0.1174 | **YES** | yes | 12,235/24,000 |
| supply_stress | **h¹** | **0.6509** | 0.0148 | 0.0256 | **+0.1509** | **YES** | yes | 12,235/24,000 |
| supply_stress | h² | 0.6501 | 0.0134 | 0.0542 | +0.1501 | **YES** | yes | 12,235/24,000 |
| supply_stress | h³ | 0.6484 | 0.0122 | 0.0829 | +0.1484 | **YES** | yes | 12,235/24,000 |
| supply_stress | h⁴ *(markov)* | 0.6151 | 0.0076 | 0.0476 | +0.1151 | **YES** | yes | 12,235/24,000 |
| supplier_reliability | h⁰ | 0.4974 | 0.0345 | 0.0906 | −0.0026 | no | **NO** | 274/24,000 |
| supplier_reliability | h¹ | 0.4966 | 0.0200 | 0.0913 | −0.0034 | no | **NO** | 274/24,000 |
| supplier_reliability | h² | 0.4824 | 0.0462 | 0.1116 | −0.0176 | no | **NO** | 274/24,000 |
| supplier_reliability | h³ | 0.4844 | 0.0268 | 0.1736 | −0.0156 | no | **NO** | 274/24,000 |
| supplier_reliability | h⁴ *(markov)* | 0.5131 | 0.0249 | 0.1017 | +0.0131 | no | **NO** | 274/24,000 |

**Recovery Capability — clears its gate, and is the strongest result in this phase.** Mean AUC
**0.6530** at h¹, above chance by +0.1530 against a 0.0310 floor, sign-consistent across all five
dataset seeds and unusually tight:

```
h¹ per dataset seed: [0.6749, 0.6510, 0.6439, 0.6496, 0.6455]
```

**This is the only state where the encoder demonstrably earns its place.** h⁰ reaches 0.5784; h¹
reaches 0.6530; the lift of **+0.0746 exceeds the 0.0310 floor**. One round of message passing adds
real, measurable information about hidden resilience beyond what the supplier's own input features
carry — which is exactly the claim V3's Layer 3 needs and the one V2's Layer 3 could never make for
hidden *identity*.

Note also that h⁰ = 0.5784 lands close to the 0.599 hand-built-feature precedent, which is a
reassuring consistency check: that precedent was a logistic regression on observable
shipment summaries, i.e. roughly the information h⁰ has.

**Limitation, stated because the row count overstates the evidence.** `RESILIENCE` is *static* per
supplier, so the target is replicated across the 6 test snapshots. The 24,000 test rows are
**~800 independent suppliers per dataset seed** (4,000 across five), each appearing six times with
identical labels. Rows are not independent, and the 0.599 precedent was computed one row per
supplier. The five-seed sign consistency is what carries the result, not the row count.

**Supply Stress — replicates, and here the encoder lift does clear.** 0.6509 at h¹, sign-consistent
across five seeds, and lift over h⁰ of +0.0335 against a 0.0256 floor. This is a *different*
outcome from Variant A, where the same lift (+0.0301) fell under a larger floor (0.0390). The
honest reading: the encoder's contribution to Supply Stress is real but small, and sits close
enough to the floor that it clears on one variant and not the other. The state itself is solidly
recoverable on both (0.6426 / 0.6509); only the attribution to message passing is marginal.

**Supplier Reliability — fails again, independently.** 0.4824–0.5131, not sign-consistent, does not
clear its floor, with 274 positives in 24,000 rows (1.1%). Failing the same way on two variants
with different mechanisms makes this a reasonably solid negative rather than a variant-specific
artefact.

### Backbone integrity at the close of the phase

Ground rule: SHARE and the Markov readout are not retrained or modified by this phase.

```
$ diff -rq --exclude=__pycache__ --exclude=.cache ml /Users/muralik/Documents/Programs/HADES_v2/ml
Only in ml: confirm_latent_states.py        <- Phase 1 addition
Only in ml: latent_state_head.py            <- Phase 1 addition
Only in HADES_v2/ml: <the 15 scrapped modules removed during the port>

$ shasum -a 256 ml/models/{rgcn_attn_encoder,rgcn_attn_markov_encoder,heads}.py
IDENTICAL  (all three, matching Phase 0's recorded digests)

$ git status --porcelain db/          # empty — the generator was executed, never edited
```

**Zero files differ in content.** Every change is an addition or one of the documented removals.
`assert_backbone_frozen()` ran after head construction and after training on every one of the 250
head fits (2 variants × 5 dataset seeds × 5 init seeds × 5 depths) and never fired.

---

## Deliverable — which states are cleared to feed Phase 2's SCM

### Final status, all six candidates

| Candidate state | Step 1 verdict | Head built? | Best AUC (floor) | Gate | Cleared to Phase 2? |
|---|---|---|---|---|---|
| **Supply Stress** | confirmed | yes | **0.6426** (0.0390) A · **0.6509** (0.0256) E | **PASS** | ✅ **yes** |
| **Recovery Capability** | confirmed (E-variants only) | yes | **0.6530** (0.0310) E | **PASS** | ✅ **yes** — E-variants only |
| **Supplier Reliability** | confirmed | yes | 0.4870 A · 0.5131 E | **FAIL** | ❌ no — below chance, not sign-consistent |
| **Inventory Health** | dropped | no | — | n/a | ❌ no — emitted, and already a model input feature |
| **Logistics Stability** | needs instrumentation | no | — | n/a | ❌ no — no readable per-entity target exists |
| **Capacity Pressure** | dropped | no | — | n/a | ❌ no — no latent per-supplier correlate exists |
| *(found)* **Mitigation Level** | needs instrumentation | no | — | n/a | ❌ not yet — highest-value instrumentation candidate |

### Cleared to feed Phase 2's SCM: two states

1. **Supply Stress** — the generator's own "causal driver of everything", recoverable at
   **0.6426 (Variant A) / 0.6509 (Variant E)**, sign-consistent across five dataset seeds on both
   variants. This is the state Phase 2 most needs: `own_stress` is the input to the
   stress→delay/shortage/impact chain whose equations Phase 2 extracts.
2. **Recovery Capability** — recoverable at **0.6530**, above the 0.599 precedent, sign-consistent,
   and the only state whose encoder lift clears its floor. **Available only on variants including
   Mechanism E** (E, F, K) — Phase 2 must not assume it on Variant A.

### Carried into Phase 2 as explicit qualifications

1. **This is a pilot, not a benchmark finding.** Everything above is at the `v1` preset (800
   suppliers, 15 snapshots). Phase 0 established spec-scale `db/csv/` holds only seeds 42–43, so
   the five-seed gate is not satisfiable there without regeneration (~2.5 min per variant-seed;
   determinism verified). Per roadmap §3.5 these numbers must be re-run at `sup_n=4000` in Phase 5
   before any of them is publishable.
2. **Message passing is doing less than the architecture claims.** For Supply Stress, h⁰ — raw
   input features, no message passing — reaches 0.6125/0.6174, and the best-depth lift over h⁰
   clears its floor on Variant E (+0.0335 vs 0.0256) but **not** on Variant A (+0.0301 vs 0.0390).
   Only Recovery Capability shows an unambiguous encoder contribution (+0.0746 vs 0.0310). Phase 2
   should not assume Layer 3 needs the full backbone for stress.
3. **The Markov readout depth is not the best depth for either state.** h⁴ — the depth the Markov
   readout uses for Supplier-entity tasks — is the *worst* performing depth for both cleared states
   (stress 0.6034/0.6151; resilience 0.5627). Best depths are h² (A) and h¹ (E). This is not a
   reason to reopen adaptive depth selection, which is closed (`findings/evolution.md` §3) — the
   fixed depths were tuned for the three *prediction* tasks, not for latent-state estimation. It is
   a reason for Phase 2 to read latent states from their own best depth rather than inheriting h⁴.
4. **Static-target non-independence.** Recovery Capability's 24,000 test rows are ~800 independent
   suppliers per seed, replicated across 6 snapshots. Five-seed sign consistency carries that
   result, not the row count.
5. **Supplier Reliability's failure is informative, not merely null.** It failed identically on two
   variants with different mechanisms. The likely cause is measurable and stated: 1.1–1.2% positive
   rate, because `IDIO` gives 25% of suppliers a single ~25-day outage across a 21-month timeline.
   Anyone revisiting it should change the *sampling*, not the head.

### Recommended before or alongside Phase 2

- **Correct `docs/14_Project_Roadmap.md` §3.1**: it prescribes Variant A as the Phase 1 testbed
  while naming `RESILIENCE` as a confirmed state and its 0.599 AUC as the gate reference. Variant A
  is `J + A` and excludes Mechanism E, so `RESILIENCE` does not exist there. (Phase 0 also logged
  two other roadmap defects, at §0 line 20 and §3.0 line 90.)
- **Instrument `mitigation_level`** (`db/generate_dataset.py:1191`) — record it per supplier per
  week. It is genuinely latent, already on `HIDDEN_STATE_COLUMNS`, and is the dynamic form of the
  best-performing state in this phase. It cannot be recovered post-hoc because `agent_observe()`
  destructively drains `agent_pending` (`:1180-1188`). This is a small, well-scoped generator
  change and the single highest-value addition Phase 1 identified.

### Statement for Phase 2

**Phase 2 is cleared to start**, using **Supply Stress** and **Recovery Capability** as its
estimated latent inputs, with Recovery Capability restricted to Mechanism-E variants and every
number above carried forward as a `v1`-preset pilot pending Phase 5 spec-scale confirmation.
