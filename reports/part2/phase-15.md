# Phase 15 — precision at coverage: can any use case reach a usable alert bar?

**Audience:** whoever decides what a planner is alerted on, and at what bar.
**Measured on:** v8 seed 1001, fixed split. **Nothing was trained.** Sources are Phase 14's stored predictions, plus
the simulation paths regenerated with the pinned `order_policy.py@9e2d59d`, gated **45/45 exact** against Phase 12 B2
(`phase15_sim_gate.json`, `c7c36c5`).

**Every operating point was chosen on VALIDATION, per seed, and applied to that seed's TEST.** `pick_on_val` never
receives test arrays. **Stage G's best arm was chosen by validation recall at p = 0.85** (`phase15_supp.json`,
`9608fce`), because the first pass stored only test recall and choosing on it would have been choosing on test
(deviation 141). The test curves are descriptive.

**The Stage B verdict rules were written into `ml/eval/phase15.py` and committed (`23d2477`) before any curve was
computed.** The classifier was shown to return each verdict on constructed inputs:

| constructed input | expected | got |
|---|---|---|
| informative | TUNABLE | TUNABLE (ρ −0.96) |
| random | NOT TUNABLE | NOT TUNABLE (lift 0.03) |
| inverted | NOT TUNABLE | NOT TUNABLE (ρ +0.98) |
| peaks early, then falls | NOT TUNABLE | NOT TUNABLE (drop 0.64) |

Artifacts: `phase15.json` (`23d2477`), `phase15_supp.json` (`9608fce`), `phase15_sim_gate.json` (`c7c36c5`). All
clean. Exact per-seed filenames throughout.

---

## 1. The answer table (Stage G)

**How to read it:**

- **Recall at p** is the TEST recall at the threshold chosen on validation to reach precision p. The TEST precision
  actually achieved is in brackets. **✓** means the bar held on test for **every** seed with ≥ 50 alerts.
- **Max precision** is the highest test precision at any coverage with ≥ 50 alerts. It is descriptive, not an
  operating point.
- **The base rate is on every row.** "REACHABLE" follows the brief's definition literally, and lift is added beside
  it, because the definition ignores base rate (deviation 134).

**RAW:**

| use case | best arm (by val) | base rate | max precision (any coverage, ≥ 50 alerts) | recall @ p=0.70 | @ 0.80 | @ 0.85 | @ 0.90 | best (precision, recall, coverage) point | Stage B | REACHABLE? |
|---|---|---|---|---|---|---|---|---|---|---|
| **UC1** late vs contract, censoring resolved | h⁴ | **0.730** | 1.000 | 0.999 (0.731) ✓ | 0.584 (0.825) ✓ | **0.396 (0.869) ✓** | 0.244 (0.910) ✓ | (0.869, 0.396, 33%) | TUNABLE | **YES** by the rule, but a **lift of only 1.19×** over "every line is late" |
| UC1b arrives within 4 weeks | h⁴ | 0.034 | 0.346 | UNREACHABLE (max val 0.316) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | TUNABLE | **NO, CEILING** |
| **UC2** arrives in full | boundary w=3 | **0.749** | 0.981 | 1.000 (0.749) ✓ | 0.885 (0.774) ✗ | **0.475 (0.821) ✗** | 0.092 (0.871) ✗ | (0.821, 0.475, 43%) | TUNABLE | **PARTIAL**: 0.82 at recall 0.48, lift 1.10 |
| UC2b fill < 0.95 | lgbm22_id | 0.245 | 0.577 | UNREACHABLE (max val 0.579) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | TUNABLE | **NO, CEILING** |
| UC2b fill < 0.75 | lgbm22_id | 0.223 | 0.500 | UNREACHABLE (max val 0.578) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | TUNABLE | **NO, CEILING** |
| **UC3** demand exceeds capacity (90-day) | mp h⁴ | 0.405 | 0.964 | 0.515 (0.681) ✗ | 0.318 (0.762) ✗ | **0.191 (0.810) ✗** | 0.047 (0.898) ✗ | (0.810, 0.191, 9.6%) | TUNABLE | **PARTIAL**: 0.81 at recall 0.19, lift 2.0 |
| **UC4** part-plant at risk | mp h¹ (shipped) | 0.257 | 1.000 | 0.349 (0.780) ✓ | 0.218 (0.873) ✓ | **0.142 (0.910) ✓** | 0.102 (0.927) ✓ | (0.910, 0.142, 4.0%) | TUNABLE | **YES** by the rule; **diagnostic**, not a client claim (v8's rate is ~8× reality) |
| UC5a below SS, observed | ROP simulation | 0.080 | 0.284 | UNREACHABLE (max val 0.243) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | WEAKLY TUNABLE | **NO, CEILING** |
| UC5b below SS, pre-rescue | ROP simulation | 0.105 | 0.340 | UNREACHABLE (max val 0.305) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | WEAKLY TUNABLE | **NO, CEILING** (and PRIVILEGED) |
| UC6 delivery schedule | — | — | — | — | — | — | — | — | — | **BLOCKED**: 0 of 4 cost parameters |
| UC7 allocation | — | — | — | — | — | — | — | — | — | **BLOCKED**: 63 feasible part-plants (qualification rule, deviation 91); every coverage cell has < 50 alerts |
| UC8 recipient | policy score | 0.625 | 0.651 | UNREACHABLE (max val 0.668) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | **NOT TUNABLE** | **NO, TUNING** |
| UC8 donor (inferred truth) | policy score | 0.514 | 0.727 at 0.1% (140 alerts, test only) | UNREACHABLE (max val 0.526) | UNREACHABLE | UNREACHABLE | UNREACHABLE | — | **NOT TUNABLE** (val) | **NO, TUNING** |

**RECALIBRATED** (arrival and fill only; capacity, shortage, simulation and recommender have no recalibrator):

| use case | best arm | base | recall @ 0.70 | @ 0.80 | @ 0.85 | @ 0.90 | REACHABLE? |
|---|---|---|---|---|---|---|---|
| UC1 resolved | h⁴ | 0.730 | 0.999 (0.731) ✓ | 0.576 (0.827) ✓ | **0.395 (0.870) ✓** | 0.241 (0.911) ✓ | YES by the rule, lift 1.19 |
| UC1b 4 weeks | h⁴ | 0.034 | UNREACHABLE (max val 0.326) | — | — | — | NO, CEILING |
| UC2 full | boundary w=3 | 0.749 | 1.000 (0.749) | 0.885 (0.774) ✗ | 0.476 (0.821) ✗ | 0.092 (0.870) ✗ | PARTIAL |
| UC2b < 0.95 | lgbm22_id | 0.245 | UNREACHABLE (max val 0.583) | — | — | — | NO, CEILING |
| UC2b < 0.75 | lgbm22_id | 0.223 | UNREACHABLE (max val 0.566) | — | — | — | NO, CEILING |

**Recalibration moves no row.** A monotone recalibration only relabels the threshold.

**The answer in one paragraph.**

- **Only two decisions reach 0.85 on test at every seed.** Arrival lateness against the contract does, but it sits
  on a 73%-late base, so 0.87 is a 1.19× lift. The shortage head does, but it is diagnostic.
- **Capacity gets closest among the genuinely informative decisions:** 0.81 at 19% recall, twice its base rate.
- **Fill shortfall and below-safety-stock are tunable, but capped far below any usable bar**: 0.58 and 0.28 at best.
- **The transfer recommender's score does not know when it is right.**

**No decision other than capacity offers a useful alert that clears 0.80.**

---

## 2. Per use case: curve, Stage B, Stage C, planner line

Coverage curves are for the best arm, test, 5-seed means, with cells under 50 alerts suppressed. Full curves with
seed bands and binomial precision CIs are in `phase15.json`.

| coverage → | 0.1% | 0.25% | 0.5% | 1% | 2% | 5% | 10% | 20% | 50% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| UC1 h⁴ (base 0.730) | n<50 | 0.998 | 0.994 | 0.986 | 0.979 | 0.968 | 0.947 | 0.908 | 0.828 | 0.730 |
| UC4 mp h¹ (0.257) | n<50 | n<50 | 0.979 | 0.961 | 0.946 | 0.890 | 0.806 | 0.658 | 0.426 | 0.257 |
| UC3 mp h⁴ (0.405) | n<50 | **0.909** | 0.899 | 0.904 | 0.897 | 0.851 | 0.810 | 0.743 | 0.582 | 0.405 |
| UC2b < 0.95 lgbm22_id (0.245) | n<50 | 0.487 | 0.453 | 0.455 | 0.426 | 0.393 | 0.367 | 0.341 | 0.299 | 0.245 |
| UC5a simulation (0.080) | 0.254 | 0.241 | 0.242 | 0.259 | **0.277** | 0.268 | 0.242 | 0.199 | 0.136 | 0.080 |
| UC8 recipient (0.625) | n<50 | 0.643 | 0.598 | 0.625 | 0.589 | 0.582 | 0.573 | 0.594 | 0.616 | 0.625 |
| UC8 donor (0.514) | 0.727 | 0.596 | 0.568 | 0.549 | 0.578 | 0.561 | 0.545 | 0.565 | 0.537 | 0.514 |

**Stage B, monotonicity (validation, seed 7 shown; the verdict is the majority over 5 seeds):**

| use case | ρ(coverage, precision) | peak precision @ coverage | precision at tightest usable | drop | verdict |
|---|---|---|---|---|---|
| UC3 | **−0.999** | 0.983 @ 0.3% | 0.983 | 0.000 | **TUNABLE** |
| UC1, UC2, UC4, UC1b, UC2b (all arms) | ≤ −0.9 | … | … | ≤ 0.06 | TUNABLE (the rolling-52 histogram only WEAKLY) |
| UC5a | −0.358 | 0.242 @ 4.1% | **0.155** @ 0.1% | **0.087** | **WEAKLY TUNABLE**: peaks early and falls at the tightest coverage (test: peak 0.278 @ 2.4%, 0.256 @ 0.1%) |
| UC8 recipient | **+0.216** | 0.678 @ 3.5% | 0.647 | 0.031 | **NOT TUNABLE**: lift 0.07 over base, ρ positive |
| UC8 donor | **+0.011** | 0.524 @ 0.3% | 0.507 | 0.016 | **NOT TUNABLE** on validation: lift 0.03. The test curve's 0.73 at 0.1% does not exist on validation and cannot be selected (deviation 138) |

**UC5 has the brief's pathology in mild form.** Precision rises to about 0.28 at 2–4% coverage and then **falls**
as coverage tightens further (0.25 at 0.1%). The simulation's most confident weeks are not its most accurate. It
misses the NOT TUNABLE threshold (drop > 0.10) by 0.013, so it is recorded as WEAKLY TUNABLE with the drop stated.

**Stage C, three-way abstention.** Both thresholds are fitted on validation per seed: ALERT at the highest bar met
among {0.80, 0.70, 0.60}, CLEAR at NPV ≥ 0.95. Test figures are 5-seed means.

| use case | ALERT bar met | alerts (precision) | clears (NPV) | no-opinion | recall among spoken | planner line |
|---|---|---|---|---|---|---|
| UC1 h⁴ | 0.80 | 18,183 (**0.825**) | 0 (NPV ≥ 0.95 unreachable) | 48.2% | 1.000 | "Of 35,128 lines, the system alerts on 18,183 as late (right 83%), clears none, and passes 16,945 back." (Base: 73% of all lines are late.) |
| UC1b h⁴ | none | 0 | 35,977 (0.966) | 0.1% | 0 | "Of 36,000 lines it never says 'arrives within 4 weeks'; it clears 35,977 (right 97%, the base rate)." |
| UC2 boundary | 0.80 | 30,833 (0.774) | 0 | 14.4% | 1.000 | "Of 36,000 lines, it alerts 30,833 as 'arrives in full' (right 77%, base 75%) and passes 5,167 back." |
| UC2b < 0.95 lgbm22_id | 0.60, one seed only | 19 (0.51) | 15 (0.853) | 99.9% | — | "Of 36,000 lines it speaks on 34; everything else goes back." |
| **UC3 mp h⁴** | 0.80 | **3,829 (0.762)** | 85 (0.869) | 82.6% | 0.997 | **"Of 22,500 channel-horizons, it alerts on 3,829 as over capacity (right 76%), clears 85 (right 87%), and passes 18,586 back."** |
| UC4 mp h¹ | 0.80 | 865 (0.873) | 6,431 (0.916) | 46.0% | 0.583 | "Of 13,500 part-plants, it flags 865 at risk (right 87%), clears 6,431 (right 92%), and passes 6,204 back." (Diagnostic.) |
| UC5a simulation | none | 0 | 424,533 (0.943) | 13.9% | 0 | "Of 492,804 part-plant-weeks it raises no alert; it clears 424,533 (right 94%, below its 0.95 validation bar)." |
| UC8 recipient | 0.60 (base 0.625) | 25,425 (0.625) | 0 | 0% | 1.000 | "It says 'transfer here' on every short week, right 63% of the time: exactly the base rate." |
| UC8 donor | none | 0 | 0 | 100% | — | "It cannot speak with confidence on any donor." |

---

## 3. Stage D — agreement filter (each member at its own validation-fitted F1 threshold, Phase 14's rule; test, 5 seeds)

| use case | single best member: precision / recall / coverage | ALL 3 agree | 2 of 3 | precision gained per point of recall paid (ALL vs best single) |
|---|---|---|---|---|
| UC2 full | 0.750 / 0.999 / 0.999 | 0.750 / 0.999 / 0.998 | 0.749 / 1.000 / 1.000 | none: every member's F1 threshold is "always full" |
| UC2b < 0.95 | head 0.307 / 0.638 / 0.509 | 0.315 / 0.599 / 0.464 | 0.278 / 0.827 / 0.726 | +0.008 for −0.039: **0.2 points per point** |
| UC2b < 0.75 | head 0.283 / 0.621 / 0.489 | 0.291 / 0.569 / 0.435 | 0.258 / 0.789 / 0.681 | +0.008 for −0.052: 0.15 points per point |
| UC3 | mp h⁴ 0.591 / 0.695 / 0.480 | **0.625** / 0.545 / 0.354 | 0.556 / 0.727 / 0.531 | +0.034 for −0.150: 0.23 points per point |

**Agreement buys almost nothing, and it is dominated by tightening the best model's own threshold.** For capacity,
h⁴ alone at its p = 0.70 point gives 0.681 precision at 0.515 recall. That beats ALL-3's 0.625 at 0.545. The model
families' errors overlap more than Phase 13's F1 parity suggested.

---

## 4. Stage E — coarser grain

**Base rates change under aggregation, so the columns compare LIFT and recall at fixed precision, never accuracy.**
Labels: a unit is positive if ANY constituent row is positive. Scores: mean and max of the constituents, as separate
arms.

| decision / grain | units (test) | **base rate** | best pooling | lift @ 1% coverage | lift @ 10% | max val precision | recall @ p=0.85 (test precision) |
|---|---|---|---|---|---|---|---|
| UC2b < 0.95, row (native) | 36,000 | 0.245 | — | 2.18 | 1.68 | 0.585 | UNREACHABLE |
| … supplier × month | 3,777 | **0.741** | max | n<50 | 1.24 | 0.963 | 0.281 (0.874) ✓, **but on a 74% base** |
| … part × plant × month | 21,694 | 0.344 | **max** | 2.06 | 1.67 | 0.725 | UNREACHABLE |
| … supplier × quarter | 1,680 | **0.911** | either | n<50 | 1.09 | 1.000 | 1.000 (0.911): **trivial**, "always yes" |
| UC5a, row (native) | 492,804 | 0.080 | — | 3.23 | 3.03 | 0.243 | UNREACHABLE |
| … part × plant × month | 63,180 | 0.135 | **mean** | **4.31** | 3.33 | 0.535 | UNREACHABLE |
| … part × plant × quarter (substitute for supplier × quarter) | 21,060 | 0.245 | mean | 3.34 | 2.64 | 0.626 | UNREACHABLE |
| UC3, channel × snapshot (native 90-day) | 22,500 | 0.405 | — | 2.23 | 2.00 | 0.988 | 0.191 (0.810) |
| … supplier × snapshot | 3,771 | 0.401 | mean ≈ max | n<50 | 2.02 | 0.931 | 0.224 (0.801) |

**Is coarser grain what makes capacity work? No.**

- **Pooling capacity from channels to suppliers leaves its lift unchanged** (2.02 against 2.00) and its reachable
  precision unchanged (0.80 against 0.81).
- **Coarsening fill does not raise its lift either** (part × plant × month 2.06 against 2.18 at row level). It
  raises the **base rate**, so "reaching" 0.85 at supplier × month is 74% base rate plus a 1.24× lift, and at
  supplier × quarter it is "always yes".
- **UC5 is the one decision whose lift rises with grain.** Part × plant × month with mean pooling goes from 3.23 to
  4.31 at 1% coverage. Its ceiling still sits at 0.54–0.63, far below any bar.

**What makes capacity work is the task.** Its label is a smooth, persistent utilisation ratio, not a lumpy per-line
event. The native "90-day" grain is incidental.

**The monthly capacity grain (E3) could not be scored:** no monthly capacity prediction exists, and building one is
training. Supplier × snapshot was substituted (deviation 137).

---

## 5. Stage F — capacity level correction (UPPER-BOUND PROXY, not a model)

| variant | scalar | recall @ 0.70 (precision) | @ 0.80 | @ 0.85 | @ 0.90 |
|---|---|---|---|---|---|
| uncorrected | — | 0.515 (0.681) | 0.318 (0.762) | 0.191 (0.810) | 0.047 (0.898) |
| **single scalar fitted on VALIDATION** (the brief's proxy) | 1.014–1.064 per seed | 0.515 (0.682) | 0.316 (0.762) | 0.191 (0.810) | UNREACHABLE |
| **per-period level tracker, PRIVILEGED** (val corrected by its ratio, test by its OWN ratio) | val 1.01–1.06; **test 1.05–1.13** | 0.580 (0.651) | 0.388 (0.735) | 0.253 (0.788) | UNREACHABLE |

**Neither moves the frontier.**

- **The validation scalar changes nothing, for two reasons.** It is fitted where the level gap is small (val 0.910
  against 0.923). And a single scalar is nearly rank-preserving, so a threshold re-fitted on corrected validation
  absorbs it (deviation 139).
- **Even perfect per-period tracking, using the test period's own level (future information), only slides the
  operating point along the same curve.** At matched recall 0.25, uncorrected interpolates to ≈ 0.787 and the
  tracker gives 0.788.

**The level lag costs capacity calibration, meaning where the threshold lands. It does not cost the precision
achievable at a given recall.** **A level-tracking retrain is not justified by this test.** It would fix the
absolute probability a planner reads off, not the alert bar.

---

## 6. Deviations, continuing from 132

| # | Prior statement | Measured | Where |
|---|---|---|---|
| **133** | brief B2: "Phase 14 §2.6 already shows [UC5 precision peaking and falling] across θ = 0.1…0.9" | Phase 14's sweep was over a fixed θ, not coverage. On the coverage axis UC5 does peak (0.28 at 2–4%) and fall (0.25 at 0.1%), with a validation drop of 0.087. That is **WEAKLY TUNABLE** under the pre-committed rules, 0.013 short of NOT TUNABLE. The pathology is real but mild | §2 |
| **134** | brief Stage G: "YES = precision ≥ 0.85 at ≥ 10% recall" | **the rule ignores the base rate.** On a 73–75%-positive decision (UC1, UC2), 0.85 is a 1.14–1.19× lift. UC1 is "YES" by the letter and nearly uninformative by lift. Lift is reported beside every verdict | §1 |
| **135** | brief E1: coarser grain as "the largest expected gain" | under "any constituent positive" the **base rate** rises (fill 0.245 → 0.741 → 0.911), and the lift does not. "Reaching" 0.85 at coarse grain is mostly base rate. The brief's own warning about comparing accuracy applies to precision as well | §4 |
| **136** | brief E1 (i), (iii): supplier grains for UC5 | a part-plant-week has **no single supplier.** Supplier × month and supplier × quarter are undefined for UC5; part × plant × month and × quarter were used | §4 |
| **137** | brief E3: capacity "at a monthly grain" | **no monthly capacity prediction exists** (the label and the model are 90-day), and building one is training. Supplier × snapshot was substituted as the coarsening test | §4 |
| **138** | brief B2: "UC8 donor is flat (0.557 at both extremes)" | on the coverage axis it is flat on **validation** (max 0.526, lift 0.03, NOT TUNABLE), but **test** shows 0.73 at 0.1% coverage. That validation cannot see or select it is exactly why operating points are chosen on validation | §2 |
| **139** | brief F1: a single validation-fitted scalar "upper-bounds what a level-tracking model might deliver" | **it cannot.** It is fitted where there is no gap, and it is nearly rank-preserving. The right upper bound corrects each period by its own level (PRIVILEGED). That was built, and it too leaves the frontier unchanged | §5 |
| **140** | brief Stage C: a CLEAR bar of NPV ≥ 0.95 | **unreachable** on majority-positive decisions (UC1, UC2), whose negatives are the minority class, and **met only at the base rate** on rare-positive ones (UC1b, UC5a: NPV = 1 − base). The three-way split is informative only for UC3 and UC4 | §2 |
| **141** | Stage G: "best arm" | a first pass stored only TEST recall. Choosing the best arm on it would have been a test-set choice. Validation recall at p = 0.85 was computed separately and drives every Stage G row. **It changes UC2's verdict:** validation picks the boundary arm (0.821 on test, PARTIAL), not b5flat22, which would have held 0.856 | §1 |
| **142** | Stage B: the constructed "peaks early then falls" test | **a first construction did not produce the pathology.** It moved confident rows to the bottom, which leaves the curve monotone. Rebuilt with confident wrong answers at the top; the detector then fires. Recorded because the check itself nearly shipped untested | §intro |

## 7. Every `inventory_position_weekly` read

| reader | purpose |
|---|---|
| `ml/sim/phase15_sim.py` through `phase14_sim.store_labels` (2024–2026 weeks) | UC5 labels (a) and (b), UC8 action labels. Evaluation only |
| `montecarlo.opening_position` and `order_policy@9e2d59d` (as-of t0) | regenerating the simulation paths. The sanctioned Phase 9 path |

**Never a feature.** `phase15.py` and `phase15_supp.py` read no store table directly; they read the stored
simulation rows.

## 8. Recommendation — one line per use case

| use case | recommendation | the number that decides it |
|---|---|---|
| **UC1** lateness vs contract | **ship as a ranked watchlist**, not an alert | the top third by P(late) is 87% late, but 73% of all lines are late: a lift of 1.19 |
| UC1b within 4 weeks | **ship as a ranked watchlist** ("likely to land early") | max precision 0.35 on a 3.4% base: about 10× lift, never an alert |
| UC2 arrives in full | **retire as a yes/no**; keep P(full) as a displayed probability | PARTIAL at 0.82 against a 0.75 base (lift 1.10) |
| UC2b materially short | **ship as a ranked watchlist** | ceiling 0.58 precision; top 1% is 2.2× the base rate |
| **UC3 demand exceeds capacity** | **ship as an alert at the validation-chosen p = 0.85 point**, labelled "81% right, catches 19%, flags ~10% of channels". **Do not fund a level-tracking retrain** for this | test 0.810 at recall 0.191; the PRIVILEGED tracker gives the same frontier |
| UC4 part-plant at risk | **internal ranked watchlist only**, never a client alert | 0.91 at 14% recall, but on an ~8×-inflated base rate |
| **UC5 below SS** | **needs a model change: restate the simulation at channel granularity** (Phase 12 open item 12) | ceiling 0.28 on an 8% base; weakly tunable, with precision falling at the tightest coverage |
| UC6 schedule | **blocked**: supply the four cost parameters | 0 of 4 exist |
| UC7 allocation | **blocked**: decide the qualification rule for incumbents | 57 of 120 infeasible; 63 left, too few to curve |
| UC8 transfer recipient | **retire** | NOT TUNABLE: best precision 0.65 against a 0.625 base |
| UC8 transfer donor | **retire as an alert**; keep only as a tie-break suggestion | NOT TUNABLE on validation (lift 0.03); precision@1 0.557 against random 0.520 |
