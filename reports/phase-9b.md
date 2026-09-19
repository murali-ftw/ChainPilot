# Phase 9B — closing the two open decisions, and the Phase 10 gate

## 1. Verdict — the gate PASSES, and Phase 10 was not started

| gate item | status | basis |
|---|---|---|
| **D.1 arrival's ranking claim** | **SETTLED — untestable on this data** | Stage A: the promise week is a line-level quantity built from the same two upstream terms as the label; the head is channel-level. Its 0.87–0.89 C-index is structural |
| **D.2 fill's model** | **SETTLED — form (i): the evidence supports switching to B5-flat-22** | Stage B: on validation the head loses raw calibration in **16 of 16** windows by 2.2×–8.2×, and wins CRPS by 0.8–2.6%. `shipped.json` **not** edited |
| **D.3 drift gate** | decided (remove), not implemented | Phase 8 §4 3c |
| **D.4 capacity depth** | decided (h⁴ ships, instability recorded as a restriction) | Phase 8 §4 3a |
| **D.5 capacity intervals** | decided (not quotable; the fix is level-aware) | Phase 8 §7 |
| **Stage C** | complete | assertion, 5-test regression suite, clean sweep, standing rule |

**The gate passes.** Phase 10 was nonetheless **not started**, for two reasons given in §7: the
guide's Phase 10 is not the work these reports have been calling "Phase 10", and starting its one
unblocked sub-item would breach this brief's 3-hour budget. **Two decisions are waiting on the user**:
approving the fill switch, and scheduling the four unscheduled changes that no phase owns.

**Two findings changed what earlier reports claimed, both recounted in code:**

1. **Arrival's ranking verdict was never a finding about the head.** Phase 7 binding statement 1 and
   Phase 8 §4 3d reported that the head loses to the promise date at ordering arrivals. It does — but the
   comparison was never winnable, because the generator builds promise and arrival from a shared line-level
   anchor the head is not given. The honest claim is **"arrival ranking is untestable on this data with the
   shipped feature set"**.
2. **Validation cannot compare recalibrated calibration at all.** Recalibration is fitted on the validation
   fold, so marginal ECE-22 is ≈0.0000 there for every arm. That is a gate that cannot fail — the seventh on
   the record — and it is why Stage B decides on the raw pre-recalibration figure.

`inventory_position_weekly` was never read. Phase 9 proper stays blocked.

---

## 2. Stage A — the promise-date structure

**This check was specified for Phase 9A, never run, and never recorded as skipped** (deviation 31).

### A.1 / A.2 — how the two quantities are built

From `db/gen_v6/generator_v6.py` and `db/gen_v7/generator_v7.py` — **configuration and definitions only; no
table under `db/gen_v6/**` or `db/gen_v7/**` was read**. The two files are identical in every term below.

```
contracted   = r.integers(15, 70, NCH)                        # per channel, drawn once, constant over time
promise_date = po_created + contracted[channel]               # days
lead         = exp(normal(log(max(contracted[channel] * 0.51, 3)), 0.34))
lead        *= 1 + 0.55*max(supplier_state,0) + 0.9*transit_state
                 + 0.85*clip(supplier_load_prev - 0.70, 0, 2) + 0.55*regime
arrival_week = po_created_week + ceil(lead / 7)
```

**Neither construction takes the other as an input.** `arrival_week` never reads `promise_date`. But they are
not independent either: they share **two** upstream terms, and both are measured from the snapshot `t0`, so
each label is an offset from the same origin:

| shared term | level | in promise | in arrival |
|---|---|---|---|
| `po_created` — when the line was raised, i.e. its **age** at `t0` | **line** | additive anchor | additive anchor |
| `contracted[channel]` — the contracted lead time | channel | the whole term | sets the **median** of the lead (×0.51) |

So the answer to A.2 is the third option: **both are drawn from common upstream quantities**, one of which is
line-level.

### A.3 — the consequence, stated plainly

**The promise date's C-index is structural, and arrival ranking is untestable on this data with the shipped
feature set.**

The head's inputs are a channel's weekly panel as of `t0`. It receives **nothing** about the individual
`po_line` — not its age, not its contracted lead, not its promise. Measured from the artifacts:

| world | origin | rows | distinct predictions | rows exactly tied | largest tie group | head C-index | promise C-index | head ρ vs label | promise ρ vs label | censored |
|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 20,000 | 18,352 | 3,221 | 4 | 0.6829 | 0.8907 | +0.370 | +0.778 | 47.2% |
| v6 | 2 | 16,000 | 14,669 | 2,608 | 4 | 0.6670 | 0.8831 | +0.347 | +0.782 | 39.9% |
| v6 | 6 | 16,000 | 14,636 | 2,661 | 4 | 0.6752 | 0.8805 | +0.342 | +0.778 | 41.5% |
| v6 | 7 | 20,000 | 18,305 | 3,307 | 4 | 0.6700 | 0.8834 | +0.346 | +0.774 | 42.7% |
| v7 | 1 | 20,000 | 17,961 | 3,964 | 4 | 0.6716 | 0.8871 | +0.276 | +0.761 | 49.8% |
| v7 | 2 | 16,000 | 14,472 | 2,977 | 4 | 0.6597 | 0.8726 | +0.264 | +0.749 | 44.1% |
| v7 | 6 | 16,000 | 14,456 | 3,003 | 4 | 0.6909 | 0.8730 | +0.320 | +0.745 | 45.3% |
| v7 | 7 | 20,000 | 18,041 | 3,828 | 4 | 0.6738 | 0.8803 | +0.294 | +0.759 | 45.4% |

- **The head is a channel-level forecaster**, confirmed by 25,569 rows across the eight cells carrying
  bit-identical predictions in groups of up to 4 — the lines that share a channel within a snapshot.
- **The promise week orders the label at ρ = 0.75–0.78; the head manages ρ = 0.26–0.37.** The promise date is
  not a better model. It is a direct readout of two terms the label is built from, one of which varies per
  line and is invisible to the head by construction.

**What this does to the earlier claims.** Phase 7 binding statement 1 ("the arrival head does not out-rank the
promise date") and Phase 8 §4 3d's "still contradicted: any ranking claim" are **artefacts of the feature set,
not findings about the head**. They remain true as measurements and must not be quoted as model evaluation.
The defensible statement is: *on this data a channel-level model cannot be ranked against a line-level
quantity, so arrival ranking is untestable; the head's measured value is lateness beyond the promise.*

This does **not** say no model could beat the promise date. A model given `promise_week` and line age would
have both the shared terms and the channel state that perturbs the lead — Phase 7 open item 1, still open.

**One caveat I will not paper over.** I first tried to bound this with an "oracle over the head's prediction
groups" and discarded it: with ~1.1 rows per group it measures the censoring ties (40–50% of rows carry one
imputed value), not the structure. It is absent from the tables above and from the artifact's reported fields.

### A.5 — is lateness a noise draw?

**No. Lateness carries channel- and supplier-level signal, and the head can see its drivers.**

A line is late when `arrival > promise`, i.e. when `lead > contracted`, i.e. when

```
0.51 * lognormal(0, 0.34) * [1 + 0.55*supplier_state⁺ + 0.9*transit_state + 0.85*clip(load_prev-0.70,0,2) + 0.55*regime] > 1
```

The multiplicative lognormal is line-level noise. **The bracket is not noise** — it is supplier latent state,
lane transit state, the supplier's previous-month utilisation, and the regime calendar. Two of those are
directly present in the head's panel: `load_ratio` **is** the supplier utilisation that drives the congestion
term, and `lead_time_ratio`, `otd_rate_last13` and `ack_gap_ratio` track supplier state.

**So the head's 8-of-8 lateness win is mechanistically expected, not a fluke**: the channel-level state that
inflates a lead beyond its contracted value is exactly what a channel-level encoder can read, while the
promise date — a constant per line — cannot express it at all. This is the one arrival claim the data
supports, and Stage A explains *why* it holds rather than merely observing that it does.

---

## 3. Stage B — the fill ship decision, on validation

Scored on **each origin's own validation fold**, both arms under the head's recalibration protocol fitted on
that fold, from score sets already on disk. No training, no inference, no new fits.

### B.1 / B.2 — the tables, and the metric that cannot discriminate

**Recalibrated validation ECE-22 is ≈0.0000 for every arm in all 16 windows** — head, LightGBM-22, B5-flat-22
and B2 alike. That is not a tie; it is a measurement that cannot come out any other way, because the
recalibrator is fitted on the fold it is scored on. Reporting it as "not distinguishable, 15 of 16" would be
a gate that cannot fail (deviation 33, standing rule 1). The calibration comparison below therefore uses
**raw, pre-recalibration** validation ECE, which nothing has been fitted to.

| world | origin | window | head CRPS | LightGBM-22 | B5-flat-22 | B2 | head ECE-22 raw | LightGBM-22 raw | B5-flat-22 raw | B2 raw | head reliability P(f=1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v6 | 1 | 2022 H1 | 0.0653 (0.0002) | 0.0665 (0.0001) | 0.0664 (0.0001) | 0.0686 | 0.1015 (0.0153) | 0.0229 (0.0027) | 0.0238 (0.0011) | 0.0119 | 0.0091 (0.0030) |
| v6 | 2 | 2022 H2 | 0.0646 (0.0001) | 0.0656 (0.0000) | 0.0654 (0.0002) | 0.0677 | 0.0798 (0.0078) | 0.0195 (0.0007) | 0.0205 (0.0015) | 0.0129 | 0.0074 (0.0012) |
| v6 | 3 | 2023 H1 | 0.0592 (0.0001) | 0.0600 (0.0001) | 0.0599 (0.0001) | 0.0621 | 0.0814 (0.0265) | 0.0100 (0.0012) | 0.0097 (0.0016) | 0.0152 | 0.0070 (0.0012) |
| v6 | 4 | 2023 H2 | 0.0540 (0.0001) | 0.0545 (0.0000) | 0.0545 (0.0001) | 0.0564 | 0.0801 (0.0170) | 0.0206 (0.0031) | 0.0202 (0.0038) | 0.0274 | 0.0073 (0.0013) |
| v6 | 5 | 2024 H1 | 0.0492 (0.0000) | 0.0498 (0.0001) | 0.0498 (0.0001) | 0.0518 | 0.0852 (0.0093) | 0.0235 (0.0020) | 0.0235 (0.0023) | 0.0193 | 0.0060 (0.0022) |
| v6 | 6 | 2024 H2 | 0.0483 (0.0001) | 0.0489 (0.0001) | 0.0489 (0.0001) | 0.0509 | 0.0587 (0.0112) | 0.0169 (0.0026) | 0.0180 (0.0022) | 0.0089 | 0.0051 (0.0016) |
| v6 | 7 | 2025 H1 | 0.0526 (0.0001) | 0.0530 (0.0001) | 0.0531 (0.0001) | 0.0551 | 0.0475 (0.0120) | 0.0102 (0.0005) | 0.0104 (0.0014) | 0.0167 | 0.0064 (0.0016) |
| v6 | 8 | 2025 Q3 | 0.0614 (0.0001) | 0.0625 (0.0001) | 0.0624 (0.0001) | 0.0650 | 0.0687 (0.0144) | 0.0225 (0.0003) | 0.0228 (0.0019) | 0.0326 | 0.0072 (0.0015) |
| v7 | 1 | 2022 H1 | 0.1020 (0.0005) | 0.1038 (0.0001) | 0.1037 (0.0001) | 0.1078 | 0.1309 (0.0474) | 0.0195 (0.0011) | 0.0192 (0.0009) | 0.0123 | 0.0121 (0.0028) |
| v7 | 2 | 2022 H2 | 0.1067 (0.0002) | 0.1083 (0.0003) | 0.1081 (0.0002) | 0.1121 | 0.0928 (0.0479) | 0.0431 (0.0005) | 0.0440 (0.0009) | 0.0239 | 0.0092 (0.0006) |
| v7 | 3 | 2023 H1 | 0.1017 (0.0004) | 0.1033 (0.0002) | 0.1031 (0.0001) | 0.1090 | 0.0955 (0.0573) | 0.0181 (0.0004) | 0.0189 (0.0017) | 0.0101 | 0.0069 (0.0023) |
| v7 | 4 | 2023 H2 | 0.0904 (0.0003) | 0.0920 (0.0001) | 0.0920 (0.0000) | 0.0987 | 0.0914 (0.0472) | 0.0363 (0.0014) | 0.0364 (0.0017) | 0.0540 | 0.0073 (0.0022) |
| v7 | 5 | 2024 H1 | 0.0805 (0.0002) | 0.0817 (0.0001) | 0.0818 (0.0001) | 0.0880 | 0.1260 (0.0318) | 0.0453 (0.0033) | 0.0483 (0.0051) | 0.0565 | 0.0080 (0.0019) |
| v7 | 6 | 2024 H2 | 0.0815 (0.0001) | 0.0829 (0.0001) | 0.0830 (0.0002) | 0.0890 | 0.0770 (0.0412) | 0.0228 (0.0007) | 0.0242 (0.0011) | 0.0170 | 0.0073 (0.0007) |
| v7 | 7 | 2025 H1 | 0.0942 (0.0002) | 0.0968 (0.0001) | 0.0965 (0.0001) | 0.1037 | 0.0824 (0.0190) | 0.0325 (0.0013) | 0.0321 (0.0014) | 0.0552 | 0.0079 (0.0024) |
| v7 | 8 | 2025 Q3 | 0.1085 (0.0006) | 0.1107 (0.0002) | 0.1106 (0.0002) | 0.1171 | 0.1242 (0.0121) | 0.0570 (0.0011) | 0.0573 (0.0008) | 0.0681 | 0.0098 (0.0023) |

**Counts on validation (disjoint 3-seed / 5-fit bands):**

| comparison | head better | not distinguishable | baseline better |
|---|---|---|---|
| exact CRPS vs LightGBM-22 | **16** | 0 | 0 |
| exact CRPS vs B5-flat-22 | **16** | 0 | 0 |
| exact CRPS vs B2 | **16** | 0 | 0 |
| raw ECE-22 vs LightGBM-22 | 0 | 0 | **16** |
| raw ECE-22 vs B5-flat-22 | 0 | 0 | **16** |
| raw ECE-22 vs B2 | 0 | 0 | **16** |

### B.3 — the two magnitudes side by side

| on validation | vs LightGBM-22 | vs B5-flat-22 | vs B2 |
|---|---|---|---|
| head's CRPS advantage, as % of the baseline's CRPS | **+0.84% to +2.60%** (median +1.57%) | +0.88% to +2.39% (median +1.48%) | +4.23% to +9.12% (median +5.26%) |
| head's raw-ECE disadvantage, as a ratio to the baseline's | **2.15× to 8.17×** (median 3.54×) | 2.11× to 8.35× (median 3.44×) | 1.49× to 10.63× (median 4.15×) |

**Validation reproduces the asymmetry, and sharpens it.** On the evaluation windows Phase 9A measured roughly
0.2–2.8% of CRPS against up to 2.4× of ECE, with LightGBM winning calibration in 12 of 16. On validation the
CRPS margin is the same size (0.8–2.6%) while the calibration gap is **larger and unanimous**: 16 of 16,
median 3.5×. The direction, the ordering and the imbalance all agree across the two folds.

### B.4 — recommendation: form (i)

**Validation agrees with the evaluation windows. The evidence supports switching fill to B5-flat-22.**

- **Why B5-flat-22 and not LightGBM-22-identity:** the two are indistinguishable from each other on both
  metrics, and B5-flat-22 uses **no identity keys**, so it is deployable to an unseen Rane entity. The
  identity-keyed variant is not.
- **What is given up:** the head's CRPS advantage, **0.8–2.6% of the baseline's CRPS** (median 1.57% on
  validation, ~0.2–2.8% on the evaluation windows). That is the entire cost.
- **What is gained:** calibration better by 2.2×–8.2× before any recalibration; a deterministic fit with no
  seed band; no per-window recalibration temperature to maintain — Phase 9A measured that temperature moving
  0.961–1.274 and reversing direction between windows, which is maintenance the baseline does not need.
- **`ml/configs/shipped.json` is unchanged.** This is a Phase 10-class change requiring approval; it is flagged
  in §5 as a proposal, and I have not made it.

### B.5 — which metric a planner consumes

A planner reading "the probability this line fills completely" consumes the **marginal bin probability**, where
calibration is the binding property. A planner feeding the whole distribution into a downstream optimiser — which
is what guide step 10.2 would do — consumes the **proper score**, where the head wins. The two metrics favour
different models, so the choice depends on which output the product actually surfaces. **That is a product
question, and I am not deciding it here.** If the answer is "both", the decision is not the one Stage B
settles, and should come back as a new comparison.

---

## 4. Stage C — the deviation-28 class

**C.1 — the assertion.** Names now come from one module, `ml/artifact_identity.py`; `loop.bundle_dir` and
`backtest.export` both derive from it, which removes the drift that caused deviation 28. Before writing
anything, `export` asserts that no two completed bundles share a bundle path, prediction stem or index key,
comparing full identity (task, world, origin, arch, depth, lr, seed, `train_snapshots`, `max_epochs`). Every
write then passes `guard_write`, which refuses an overwrite whose recorded identity differs, using a manifest
(`preds/_manifest.json`, 600 entries). Files predating the manifest are adopted on first sight, then protected.

**A false positive the assertion found in its first run, and the fix.** It initially compared bare bundle
*names* and flagged `v6_lite_h4_lr0.00025_s17` at origins 1 and 2 — bundles that differ legitimately, since a
name is unique only within its `task/origin` directory. The check now compares full bundle **paths**. Recorded
because a check that fires on correct data would have been disabled within a week (standing rule 1).

**C.2 — the regression test.** `ml/tests/test_artifact_identity.py`, 5 tests, all passing:

| test | what would have caught deviation 28 |
|---|---|
| `test_truncation_suffix_separates_every_name` | the exact pair: same task/world/origin/arch/depth/lr/seed, different `train_snapshots` — bundle name, path, both prediction stems and the index key must all differ |
| `test_assert_unique_accepts_distinct_and_rejects_collisions` | distinct configurations pass; a same-name pair differing in identity raises; same name under a different origin is allowed |
| `test_every_identity_field_either_names_or_collides` | each identity field either changes the name or must collide loudly — no field can change what was trained while leaving the artifact path untouched |
| `test_guard_write_refuses_a_differing_overwrite` | re-export of the same cell is idempotent; a different cell writing the same path raises |
| `test_guard_write_adopts_unmanifested_files` | artifacts written before the manifest are protected from the second write onward |

**C.3 — the sweep.** Clean, with nothing found:

| checked | result |
|---|---|
| 198 completed bundles, pairwise on path / stem / index key | **no collisions** |
| 198 index entries | no duplicate keys; no bundle directory mapped by more than one key |
| completed bundles on disk vs index | 198 of 198 present exactly once |
| 1,560 prediction files | 600 bundle-owned, 944 baseline, 16 promise-only — **0 unclaimed** |

The only collision of this class that ever occurred is deviation 28 itself, already corrected in Phase 9A.

**C.4 — the standing rule.** `docs/implementation_guide.md` now opens with a **Standing rules** section, linked
from the contents, holding three rules: gates that cannot fail; **artifact identity must be asserted unique at
write time, never assumed from a naming convention**; and selection on validation only, including what to do
when validation cannot settle the question.

---

## 5. Proposed Phase 8 §5 rows — a proposal only, `shipped.json` unchanged

`reports/phase-8.md` is not edited: earlier reports are protected. These rows supersede its §5 **if approved**.

| task | ship | confidence | period the claim is defensible over | range / restriction |
|---|---|---|---|---|
| **fill_rate** | **PROPOSED CHANGE: B5-flat-22 LightGBM** (flat graph features, no identity keys), replacing the h⁰ head. Requires approval; not made | **Moderate** | All 8 origins × 2 worlds, agreeing on validation and evaluation folds | **Basis:** raw validation ECE-22 better than the head in 16 of 16 windows (2.1×–8.4× for B5-flat-22), and recalibrated evaluation ECE better in 12 of 16. **Cost:** 0.8–2.6% of exact CRPS, in which the head wins 16 of 16. **Also gained:** determinism, no seed band, no per-window temperature (the head's moved 0.961–1.274 and reversed direction). **Undecided:** which metric the product surfaces (B.5) — if the proper score is what a downstream optimiser consumes, this proposal should be re-opened |
| **arrival_week** | h⁴ SHARE-lite for ranking, h⁰ recalibrated for the served distribution, unchanged; remove the drift gate | **Moderate for lateness; ranking is UNTESTABLE, not lost** | Origins 1, 2, 6, 7 both worlds, plus the 2025 fixed split; 2023 and 2024 H1 untested | **Quotable:** lateness beyond the promise — 8 of 8 windows, every seed, beating LightGBM in all 8; forward-looking figure **+0.02 median**, range +0.001 to +0.031, set by the evaluation window and not by history length (Phase 9A Stage B). **Restated:** the head does not out-rank the promise date, **and cannot be tested against it on this data** — the promise is a line-level readout of the label's own construction (§2). Do not quote the C-index gap as a model result. **Untested:** whether a head given `promise_week` and line age would beat it |

---

## 6. Deviations

Continuing the guide's index from row 31.

| # | the guide / specification says | measured, and what Phase 9B did |
|---|---|---|
| 31 | Phase 9A ran the promise-date structural check | **it did not** — specified for Phase 9A, never run, never recorded as skipped. Run here as Stage A; the omission is recorded rather than quietly absorbed |
| 32 | arrival's C-index measures how well a model orders arrivals | **it cannot, for a channel-level model.** Promise and arrival share a line-level anchor (`po_created`) and a channel-level scale (`contracted`); the head emits one distribution per channel-snapshot (25,569 exactly tied rows, groups up to 4). **Arrival ranking is untestable on this data with the shipped feature set** |
| 33 | validation can compare recalibrated calibration between configurations | **it cannot**: the recalibrator is fitted on that fold, so marginal ECE-22 is ≈0.0000 for every arm. The 7th "gate that cannot fail". Stage B decides on raw pre-recalibration ECE instead |
| 34 | "belongs to Phase 10" names work inside Phase 10's scope | **it does not.** The guide's Phase 10 is 10.1 delivery-schedule MILP and 10.2 allocation. Removing the drift gate, switching fill, fixing capacity intervals and adding `promise_week` are **unscheduled work belonging to no phase** |
| 35 | a uniqueness assertion protects correct data | the first version compared bare bundle names and fired on two legitimate bundles at different origins. Corrected to compare full bundle paths before being relied on; recorded because a check that fires on correct data gets disabled |

---

## 7. Phase 10 — actual scope, estimate, and why it was not started

### E.1 — what the guide's Phase 10 actually says

Read from `docs/implementation_guide.md` §Phase 10. **It is not "implement the decisions from Phases 8 and 9."**
It is two optimiser steps, neither of which is a model change:

| step | what | files | depends on | needs a learned model? |
|---|---|---|---|---|
| **10.1 Delivery schedule LP** | weekly receipt quantities minimising holding, ordering and freight; integer because of the binary order-placed indicator `y_w`, subject to stock balance, safety stock, MOQ, lot-size multiples and a monthly requirement | `ml/opt/schedule_lp.py` | 1.2 | **no** — "no learned model needed" |
| **10.2 Allocation** | recommended supplier split: enumerate 5 candidate splits, re-run the Phase 9 simulation under each, rank by expected shortage plus purchase cost; MILP only later, with any piecewise-linear approximation re-simulated to verify | `ml/opt/allocation.py` | **9.1** | via the simulation |

Its stated principle is that **the constraints are the product, not the objective** — qualification lead times,
non-transferable tooling, minimum-volume penalties and ramp-rate limits are what make a recommendation usable.
Both steps carry a concrete verify: for 10.1, a part with no capacity constraint and zero holding cost must be
ordered in the last feasible week; for 10.2, a part whose tooling is neither transferable nor duplicated must
return the incumbent split unchanged.

### E.2 — estimate

| sub-item | status | estimate |
|---|---|---|
| 10.1 delivery schedule LP | **unblocked**: depends only on 1.2, needs no learned model, no `shipped.json` change, and `scipy.optimize.milp` is already installed (no new dependency) | **1.5–2.0 h** — five input tables to load as-of, the MILP, and the verify gate |
| 10.2 allocation | **BLOCKED**: depends on 9.1, the Phase 9 simulation, which cannot start while `inventory_position_weekly` is unread | not estimable |
| **total startable now** | | **1.5–2.0 h**, under the 4 h threshold |

### E.3 / E.4 — why nothing was started

**Under the 4 h rule 10.1 qualifies to begin. It was not begun, because this brief's hard 3-hour budget governs**
and roughly 2 h were spent on Stages A–D plus this report. Starting a 1.5–2.0 h sub-item would breach it. Stopping
at the boundary is the rule's intent.

**Two things need a decision before Phase 10 is worth entering:**

1. **The fill switch (§3, §5).** Approve, reject, or answer B.5's product question first.
2. **Four changes belong to no phase** (deviation 34): removing the drift gate, switching fill, the level-aware
   interval fix, and whether `promise_week` becomes a head input. Earlier reports filed all four under "Phase 10",
   which the guide does not support. They need a home, and 10.1 does not depend on any of them.

---

## 8. Open items carried forward

Unchanged from Phase 9A §7 — fill's marginal calibration, recalibration hurting marginal ECE on v7 o6, fill's 9×
ECE spread across windows, Stage B's remaining recency ambiguity (deviation 30), arrival origins 3–5 untrained,
deviation 20's validation-outcome overlap, 3 seeds against the specification's 5, capacity intervals not quotable,
the drift gate decided but not implemented — plus:

1. **Arrival ranking is untestable on this data** (§2). Either the claim is permanently restated as "lateness
   beyond the promise", or `promise_week` and line age become head inputs and the question is re-opened as a model
   change. Phase 7 open item 1, now with a mechanism.
2. **Validation cannot compare recalibrated calibration** (deviation 33). Any future calibration selection must
   either use a held-out slice the recalibrator never saw, or compare raw outputs.
3. **The fill switch is proposed and unapproved** (§5). `shipped.json` is untouched.
4. **Four changes have no owning phase** (deviation 34).
5. **B.5's product question is open**: which fill metric the product surfaces decides whether §5's proposal is
   right.

**`inventory_position_weekly` was never read in Phase 9B** — no file under `ml/` references it, and no stage of this
phase opened it. Phase 9 proper stays blocked.
