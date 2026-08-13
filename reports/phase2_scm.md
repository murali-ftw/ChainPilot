# Phase 2 — Structural Causal Model (HADES V3, Layer 4)

**Date:** 2026-08-14
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisites:** `reports/phase0_audit.md` (all five checks pass);
`reports/phase1_latent_state.md` (Supply Stress, Recovery Capability cleared);
`reports/phase1_mitigation_level.md` and `reports/phase1_mitigation_temporal.md` (Mitigation
Level: Gate 0 inert/correct, Gate 0.5 static passes, **Gates 2 and 3 STOP** — no temporal channel;
delivered to the SCM as an observable-derived input, not as a latent-state estimate).

**New code:** `ml/scm.py` (the SCM), `ml/run_scm_validation.py` (the three validation checks).

---

## Validity caveat — stated first, and carried in the module itself

This SCM's equations are correct **because HADES is synthetic and its structural equations are
coded in `db/generate_dataset.py`**. They were transcribed from that source, not learned. That is
what makes them exact, and it is the entire limit of the claim:

- *"validated on this benchmark via generator-extracted equations"* — achievable, and what is
  measured below.
- *"ready for real industrial deployment"* — **not** established by this validation, and must not
  be asserted on its basis.

In a real deployment no generator exists; the causal structure would have to be hand-specified by
domain experts or learned by causal discovery from real operational data — a substantially harder
problem this project has not attempted. Per `docs/14_Project_Roadmap.md` §3.2 step 3, this caveat
is recorded **in the SCM module's own documentation**, not only here: it is the opening block of
`ml/scm.py`'s docstring.

---

## Extraction — what was pulled from the generator's source

`ml/scm.py` is a **standalone re-implementation**. It never calls into `db/generate_dataset.py`.
That choice is what makes Gate 4a meaningful: a wrapper around the generator's functions would
agree with them by construction and would validate nothing.

Every constant and functional form is cited to a line in the generator. Nothing is fitted.

| Equation | Extracted form | Source |
|---|---|---|
| event ramp | `mag · clamp01((t−s0)/(pk−s0)` if `t≤pk` else `1−(t−pk)/(s1−pk))` | `:839-856` |
| `own_stress` | `min(0.95, (1−base_rel)·0.5 + Σ ramps over EVENTS, HP_B_EVENTS, SHOCK_EVENTS, IDIO)` | `:827-857` |
| co-parent bleed | `+ recv_atten(s) · COPARENT_COUPLING · Σ_partners own_stress` | `:869-872` |
| hidden-parent coupling | `+ recv_atten(s) · HP_ALPHA · mean_{co-members} own_stress` (Type A only) | `:812-824`, `:876` |
| upstream chain | `Σ_hops (Π downstream attenuations) · own_stress(up)` | `:634-650` |
| `stress` | `min(0.95, own_stress + the three transmission terms)` | `:860-880` |
| resilience absorption | `max(0, 1 − resilience_lambda · RESILIENCE[s])` | `:536-547` |
| F attenuation | piecewise-linear in resilience through `(0,atten_low)`,`(0.5,atten_medium)`,`(1,atten_high)` | `:560-566` |
| **stress → delay** | `min(0.80, 0.025 + 0.38 · st · absorption(s))` | `:1085` |
| port bump | `st += 0.10` for a SEA carrier dispatched inside a `PORT_EVENT` window | `:1080-1082` |
| factory bump | `st += 0.35` inside the `FACTORY_OUTAGE` window | `:1078-1079` |
| stress → shortage | `sourcing_stress(p,t) = max_{BOM suppliers} stress(s,t)` | `:961-971` |

**Mitigation Level is represented because the generator uses it causally**, independent of Phase 1's
finding that it is observable rather than latent — the SCM must model what the world does, not what
Layer 3 can estimate:

| Equation | Extracted form | Source |
|---|---|---|
| replenishment trigger | `1.55 + 0.45 · max_{BOM suppliers} mitigation_level` | `:1221-1225` |
| replenishment quantity | `thr · U(2.0,2.8) · (1 + 0.35 · mitigation_level)` | `:1233-1235` |

**What the SCM deliberately does not reproduce.** The generator's stochastic draws —
`random.random() < p_delay`, the lognormal lateness magnitude, `U(2.0,2.8)` — are noise terms. The
SCM reproduces the *structure* (the probability and the mechanism), not individual Bernoulli
outcomes, which no model can recover. This is why 4a.2's outcome check is a calibration test
rather than an exact-match test.

---

## GATE 4a — Extraction fidelity

### 4a.1 — Equation fidelity (against the generator's own functions)

`ml/scm.py`'s `own_stress`/`stress` compared against the generator's, for every visible supplier at
every snapshot.

```
$ ./venv/bin/python -u ml/run_scm_validation.py --variant E --seeds 42,43,44,45,46 --config v1 --depth 1
$ ./venv/bin/python -u ml/run_scm_validation.py --variant K --seeds 42,43,44,45,46 --config v1 --skip-heads
```

| variant | seeds | comparisons/seed | max abs err `own_stress` | max abs err `stress` | exact? |
|---|---|---|---|---|---|
| E (Mechanism E) | 42–46 | 12,000 | **0.0** | **0.0** | ✅ all 5 |
| K (all ten mechanisms) | 42–46 | 9,810–10,575 | **0.0** | **0.0** | ✅ all 5 |

**Result: exact — not "within tolerance", bit-for-bit zero**, on 111,375 supplier×snapshot
comparisons across two variants and ten dataset seeds. Summation order in `own_stress` was matched
to the generator's deliberately (EVENTS → HP_B_EVENTS → SHOCK_EVENTS → IDIO), because float
addition is not associative and exact agreement is the whole test.

**Variant K is the check that matters, and it is why it was run.** Variant E activates only
Mechanism E, so on E alone the fidelity result would cover just `own_stress` + `resilience
absorption` and would say nothing about the transmission terms — the extraction could have been
wrong in four places and still scored 0.0. The runner therefore records which terms each variant
actually exercises. On Variant K:

```
terms active: ['coparent_coupling', 'f_attenuation', 'hp_alpha_type_a', 'hp_b_events',
               'idio', 'own_stress_events', 'resilience_absorption', 'shock_events',
               'upstream_chain']
```

All nine terms live, still exactly 0.0. Every branch of the extracted equations is covered.

### 4a.2 — Outcome fidelity (against realised, emitted outcomes)

Equation agreement shows the transcription is faithful; it does not show the model predicts the
world. This check runs the SCM forward to `p_delay` on real shipments and compares against
outcomes read from the generated world.

Ground truth is the generator's `delayed` status transition (see the ground-truth error note below). The SCM
predicts a *probability*; the world realises a Bernoulli draw from it, so the test is calibration,
not exact match.

**Variant E, per dataset seed:**

| seed | shipments | mean predicted | observed rate | gap | Brier | AUC(p_delay vs late) |
|---|---|---|---|---|---|---|
| 42 | 15,777 | 0.0670 | 0.0678 | **0.0008** | 0.0599 | 0.7012 |
| 43 | 15,748 | 0.0491 | 0.0467 | 0.0023 | 0.0434 | 0.6580 |
| 44 | 15,769 | 0.0556 | 0.0572 | 0.0016 | 0.0525 | 0.6595 |
| 45 | 15,740 | 0.0632 | 0.0640 | **0.0008** | 0.0560 | 0.7087 |
| 46 | 16,058 | 0.0457 | 0.0466 | **0.0009** | 0.0439 | 0.6211 |
| **mean** | | **0.0561** | **0.0565** | **0.0004** | **0.0511** | **0.6697** |

**Variant K (all ten mechanisms), per dataset seed:**

| seed | mean predicted | observed rate | gap | AUC(p_delay vs late) |
|---|---|---|---|---|
| 42 | 0.0800 | 0.0803 | **0.0003** | 0.6978 |
| 43 | 0.0607 | 0.0573 | 0.0034 | 0.6519 |
| 44 | 0.0696 | 0.0725 | 0.0028 | 0.6786 |
| 45 | 0.0743 | 0.0733 | 0.0010 | 0.7121 |
| 46 | 0.0618 | 0.0629 | 0.0012 | 0.6385 |
| **mean** | **0.0693** | **0.0693** | **0.0000** | **0.6758** |

**Measured tolerance: aggregate predicted-vs-observed agreement is 0.0004 (Variant E) and 0.0000
(Variant K) in absolute delay probability**, on base rates of 5.65% and 6.93%. Per-seed gaps stay
within 0.0034. Sign of the gap is not consistent across seeds, which is what an unbiased estimator
under Bernoulli noise should look like.

Two honest qualifications:

- **Worst single-bucket calibration gap is larger than the aggregate**: 0.1031 (E) and 0.0841 (K).
  These occur in sparse high-probability buckets where few shipments land; the aggregate is the
  well-supported number and the bucket figure is reported rather than hidden.
- **AUC(p_delay vs actual late) ≈ 0.67** is *not* a defect of the SCM. It is near the ceiling
  imposed by Bernoulli noise: even the generator's own `p_delay` cannot rank individual coin flips
  better than the probabilities allow. It is reported to show the SCM orders risk correctly, not as
  a predictive-accuracy claim.

### A ground-truth error this check caught

The first version of this check defined "was the shipment late?" as `delivered_at > eta`. That
produced predicted 0.0670 against observed 0.0977 and looked like systematic SCM
under-prediction of ~31%.

It was not. On the on-time branch the generator computes
`actual = J(eta − timedelta(hours=random.randint(0, 30)))` (`:1088`), and `J()` adds 0–2700 s of
jitter (`:207-212`). Whenever that hours draw is `0` — about **1 in 31** on-time shipments — the
delivery timestamp lands *after* `eta` with no delay having occurred. The timestamp comparison was
counting those as late, inflating the observed rate by ≈0.03.

The generator's actual delay event is the **`delayed` status transition** (`:1096-1098`) — the same
ground truth `training_labels` is built from (`:1355`). Switching to it moved the observed rate to
0.0678 against a predicted 0.0670, and lifted `AUC(p_delay vs late)` from 0.6400 to 0.7012. Shipments
whose `eta` falls past `T_END` are excluded, since the generator only emits the transition when
`eta <= T_END` — they are censored, not on-time.

Worth recording because the failure mode was asymmetric: the wrong ground truth made a **correct**
SCM look broken. Had the extraction been reported against it, Phase 2 would have logged a false
negative.

---

## GATE 4b — Causal fidelity: true states vs. estimated states

The SCM cannot consume privileged signals at inference. Each state Phase 1 delivered was fed into
the SCM twice: **(a)** the true generator state — an upper bound on what correct equations can
achieve — and **(b)** the estimate a real inference-time pipeline would actually have. Variant E,
5 dataset seeds.

**Each state is evaluated through its own causal channel, not a common one.** Supply Stress and
Recovery Capability enter `p_delay`. Mitigation Level does **not** appear in `p_delay` at all — its
causal role is the replenishment trigger (`:1258-1259`) and order-quantity multiplier (`:1269`) —
so it is scored through `replenish_trigger`. Scoring mitigation through `p_delay` would have
measured a channel the generator does not have.

**Scale mapping, and its limitation.** The Layer 3 heads were trained on a *median-split* target,
so they emit a probability, not a value on the stress/resilience scale. Estimates are mapped back
by rank/quantile matching against the true distribution. That mapping is monotone, so it preserves
exactly the ordering the head learned and injects no new information — but it also means these
results measure **how well the head ranks the state**, not how well it regresses it. A head trained
directly as a regressor would be the cleaner instrument and is the obvious follow-up.

| state | SCM output | head AUC | true (mean) | estimated (mean) | MAE | max abs err | Pearson r |
|---|---|---|---|---|---|---|---|
| **Supply Stress** | `p_delay` | 0.6539 | 0.0906 | 0.0906 | **0.0543** | 0.3466 | **0.1595** |
| **Recovery Capability** | `p_delay` | 0.6503 | 0.0481 | 0.0484 | **0.0126** | 0.2501 | **0.7098** |
| **Mitigation Level** | `replenish_trigger` | 0.9853 | 1.5777 | 1.5777 | **0.0204** | 0.3725 | **0.4883** |

**This is the first measurement of what Layer 3's estimation error costs Layer 4, and the three
states behave very differently — in ways head AUC alone does not predict at all.**

**Supply Stress is the expensive one.** MAE 0.0543 on a mean `p_delay` of 0.0906 — a **60% relative
error** — and the correlation between true-state and estimated-state `p_delay` is only **r = 0.16**.
Stress enters `p_delay` as a direct linear multiplier (`0.025 + 0.38·st·absorption`), so ranking
error passes straight through, and the quantile mapping spreads estimates across the full stress
range whether or not the head resolved that ordering. Note the *means* match to four decimals
(0.0906 vs 0.0906) — the mapping preserves the marginal distribution by construction — so **an
aggregate-level check would have shown perfect agreement and concealed a per-supplier correlation
of 0.16.** Aggregate agreement is not evidence here.

**Recovery Capability is cheap.** MAE 0.0126 and **r = 0.71**. Resilience enters only through
`absorption = max(0, 1 − 1.3·r)`, a bounded multiplier that compresses estimation error rather than
amplifying it, and the clamp at 0 folds the entire top tail of resilience onto the same value — so
misestimating a highly-resilient supplier often costs nothing at all downstream. The generator's own
comment at `:541-544` anticipates this ("the top tail saturates into effective immunity and becomes
mutually indistinguishable").

**Mitigation Level is the most instructive case: a 0.9853 head yields only r = 0.4883
downstream.** That is a near-perfect classifier producing a downstream signal barely better than
half-correlated with truth, and the reason is not noise — it is that **the head's AUC and the SCM's
input are measuring different things**. AUC scores the *ranking* of mitigation around its median;
`replenish_trigger = 1.55 + 0.45 · mitigation` consumes its *magnitude*. Phase 1's Gate 1 measured
exactly this split independently: mitigation's binarised ranking is ~0.975 recoverable from emitted
columns, while its continuous magnitude is only R² = 0.3049 explained, with r ≈ 0.55. **Gate 1's
r ≈ 0.55 on magnitude and Gate 4b's r = 0.4883 downstream are two independent measurements of the
same limitation, and they agree.**

The general lesson, which applies to every future Layer 3 head in this project: **a head's AUC does
not predict what it costs Layer 4.** Compare the three rows — 0.6503 → r = 0.71, 0.9853 → r = 0.49,
0.6539 → r = 0.16. The ordering is inverted relative to head quality. What determines the downstream
cost is how the state enters its equation (bounded multiplier vs. direct linear term) and whether
the head was trained on the quantity the equation actually consumes (magnitude vs. rank).

**Implication for Phase 3:** counterfactual results routed through the SCM will inherit Supply
Stress's estimation error far more than Recovery Capability's. Any intervention whose effect runs
through estimated stress should be reported against a floor that includes this error, not only the
model-init floor. For mitigation specifically, Phase 1 recommends supplying it from **emitted
features** (0.9750, no privileged read) rather than a Layer 3 head; since the emitted-only and
head-based estimates differ by only +0.0105 AUC on the same binarised target, the downstream
`replenish_trigger` fidelity would be comparable or marginally worse — the r = 0.4883 ceiling is
set by the rank-vs-magnitude mismatch, not by which estimator produced the ranking.

---

## Gate verdict

**Phase 3 (Counterfactual Engine) is CLEARED to start.**

| requirement (`docs/14_Project_Roadmap.md` §3.2) | result |
|---|---|
| Extract equations from the generator's source, not learned | ✅ done — every term line-cited, nothing fitted |
| Validate the extraction reproduces generator output exactly or within stated tolerance | ✅ **exact (0.0)** on equations; **0.0004 / 0.0000** aggregate calibration on outcomes |
| Cover the full stress→delay/shortage/impact chain | ⚠️ **partial** — see limitation 1 |
| Test with true *and* estimated states, report the gap | ✅ done, all three states — Supply Stress r=0.16, Recovery Capability r=0.71, Mitigation r=0.49 |
| Record the validity caveat in the SCM's own documentation | ✅ opening block of `ml/scm.py` |

**GATE 4a: PASS.** Equations exact (0.0) on Variants E and K; outcome calibration within 0.0004 /
0.0000 aggregate. **GATE 4b: reported, diagnostic** — the gaps are measured per state above and are
large enough for Supply Stress that Phase 3 must carry them into its floors rather than assume
estimated state is a drop-in for true state.

### The state set Phase 3 receives

Per `reports/phase1_mitigation_temporal.md`'s verdict:

| state | form | SCM channel | true→estimated fidelity |
|---|---|---|---|
| **Supply Stress** | Layer 3 estimate (h² Variant A / h¹ Variant E) | `p_delay` | r = 0.16 — **weakest link** |
| **Recovery Capability** | Layer 3 estimate (h¹), Mechanism-E variants only | `p_delay` via `absorption` | r = 0.71 |
| **Mitigation Level** | observable-derived (emitted features), **no temporal channel** | `replenish_trigger`, order quantity | r = 0.49 |

`docs/14_Project_Roadmap.md` §1 established that Layer 4 is **load-bearing, not optional**: naive
graph-edit counterfactuals sat at chance (0.478) and an interventionally-supervised delta head did
worse (0.402). This phase supplies the load-bearing component with an exactly-verified forward
model, so Phase 3 has the SCM-routed path it requires.

### Limitations carried into Phase 3 — stated, not deferred

1. **Delay is validated end-to-end; shortage and impact are not.** `sourcing_stress` (the shortage
   entry point) is extracted and exercised inside `stress`, but the full
   inventory-walk→reorder→shortage-episode chain and the impact aggregation were **not** validated
   against emitted outcomes the way `p_delay` was. Only the delay conversion carries the
   0.0004 measurement. Any Phase 3 claim about shortage or impact needs that chain validated first
   — and this is the same asymmetry `reports/decision_support_build.md` §1.2 already found from the
   other direction (closed-form ground truth usable for impact/shortage but not delay).
2. **`v1` preset only.** Consistent with Phase 1; per roadmap §3.5 these are pilot numbers. Phase 5
   must re-run at `sup_n=4000`. The equation-fidelity result is scale-invariant (it is an algebraic
   identity), but the calibration and true-vs-estimated numbers are not.
3. **Estimated-state results measure ranking, not regression** (see Gate 4b). Every Layer 3 head in
   this project is trained on a median-split target, but the SCM's equations consume *magnitudes*.
   Mitigation makes the cost of that mismatch explicit — 0.9853 AUC → r = 0.4883 — and it is the
   single clearest argument for training the next generation of Layer 3 heads as **regressors**.
   This is now the top recommendation carried out of Phase 2.
4. **Mitigation Level's causal path is represented but not validated forward.** Its two causal uses
   (replenishment trigger, order quantity) sit inside the inventory chain covered by limitation 1,
   so Gate 4b measures the *fidelity of its input to the SCM*, not the fidelity of the shortage
   outcome it eventually drives. Per `reports/phase1_mitigation_temporal.md` it is supplied from
   **emitted features** (~0.975 recoverable, no privileged read) and carries **no temporal channel**
   — Gates 2 and 3 both stopped.
5. **The SCM models structure, not noise.** Individual Bernoulli outcomes are not reproducible by
   anything, which bounds every outcome-level metric here.

### Backbone and inherited-file integrity

```
$ shasum -a 256 ml/models/{rgcn_attn_encoder,rgcn_attn_markov_encoder,heads}.py
IDENTICAL (all three, matching Phase 0's recorded digests)
```

SHARE and the Markov readout were not retrained or modified; `assert_backbone_frozen()` ran on
every head fit in Step 3 and never fired. `db/generate_dataset.py` carries only the Phase 1
mitigation recorder, whose byte-identity was proven in `reports/phase1_mitigation_level.md`; Phase 2
made no generator change.
