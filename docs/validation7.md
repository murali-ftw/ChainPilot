# Dataset validation — run 7 (`db2/generator_v7.py`, `gen_v7/`)

**Verdict: v7 is not usable. Fix A did what it was asked to do and cost more than it bought.**
The seed cascade flattened exactly as intended — escalated episodes 3.18× → **1.57×** apart — and
part-shortage learnability improved sharply, from a GBM that *lost* to the naive baseline in v6 to
one that beats it by 83%. But six of the fourteen regression outcomes left band on seed 1001, and
**per this run's own rules a regression on the fourteen is a worse outcome than the fix landing.**
Separately, and independently of v7: **G4 fails on v6 and on v7 — the graph contributes ~0% of
above-chance signal on every GNN task.** That is the finding that matters most in this document.

| | v7 · 1001 | v7 · 1002 | v6 · 1001 restated | v6 · 1002 restated |
|---|---|---|---|---|
| **passed / failed / skipped** | **163 / 9 / 4** | 167 / 4 / 4 | **171 / 1 / 4** | 169 / 3 / 4 |
| verdict | **NOT USABLE** | usable with fixes | usable with fixes | usable with fixes |
| conservation | exact | exact | exact | exact |
| failing gates | fill<1.0 ×2, constrained ×2, unobservable ×2, shortage/yr, line-stops/yr, late [2019-25] | fill<1.0 [2019-25], shortage/expedite/line-stop rates | censored tail [2019-25] | shortage/expedite/line-stop rates |

v6's columns are **restated on the amended instrument**, so the v6 → v7 comparison uses one
instrument throughout. On the pre-amendment instrument v6 read 170/2/4 and 168/4/4.

| | |
|---|---|
| `db/validator.py` **before** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| `db/validator.py` **after** | `ed4742e0bef9d60af7c5b2a2550b590b27e6b217574df5b17877ba99be57ac56` |
| Amendment | **01** — [validator_amendment_01.md](validator_amendment_01.md) |
| **Control** | **172 / 2 / 4 → 171 / 3 / 4** — it moved, as expected |
| Generation, seed 1001 | 169.4 s wall, 10.91 GB peak, 49/49 tables, 3.1 GB |
| Generation, seed 1002 | ~170 s wall, ~10.9 GB peak, 49/49 tables, 3.0 GB |

`db2/generator_v6.py` (`ad7534a8…`), `db2/generator_v5.py`, `gen_v6/` and every prior report are
untouched.

---

# 2. Instrument amendment 01 — the first change to a frozen instrument

Full derivation in [validator_amendment_01.md](validator_amendment_01.md). In brief: the
censored-tail check counted lines whose **promise** date fell in the last 90 days but normalised by
`90 / span` with span measured on **created** dates. Since `promise = created + contracted` and
`contracted ~ U(15,70)`, the censored population spans `90 + 42.5 = 132.5` days of creation against
a 90-day normaliser — **a floor of 1.47× against a 0.3–1.2× band, for any world with non-zero lead
time.** Run 6 predicted 1.47× and measured 1.47×/1.46×, which is what identifies it as an
instrument defect. The denominator is now the width of the censored window as a fraction of the
promise timeline — the same variable the censoring predicate tests.

## Effect

| dataset | full span, before | full span, after | 2019–2025, before | after |
|---|---|---|---|---|
| **control** `csv_full_seed1` | 0.56× ok | **0.20× FAIL** | 0.50× ok | 0.50× ok |
| v6 · 1001 | 1.47× FAIL | **0.84× ok** | 1.20× FAIL | 1.20× FAIL |
| v6 · 1002 | 1.46× FAIL | **0.84× ok** | 1.19× ok | 1.19× ok |
| v7 · 1001 | — | **0.70× ok** | — | 1.19× ok |
| v7 · 1002 | — | **0.70× ok** | — | 1.20× ok |

The windowed figures are unchanged, and correctly so: inside a window the population is selected on
promise dates, so `prom_hi ≈ edge` and the new denominator reduces to the old one. Only the
full-span figure, where promises run past the last creation date, moves.

**The control now fails, and I am reporting that rather than adjusting anything.** But the
diagnosis matters, because it exposes a second weakness in the check that the amendment did not
remove:

| | max promise overhang past last creation | overhang at P99.9 | max lead time |
|---|---|---|---|
| control | **176 days** | **6 days** | 212 d |
| v6 / v7 | 69 days | 53 days | 69 d |

The new denominator uses `max(promise)`, an extreme-value statistic. On the control a handful of
212-day-lead lines stretch it by 176 days, inflating the expected share ≈2.8× and moving the gate
from 0.56× to 0.20×. Substituting a P99.9 promise date for the max would put the control at
≈0.55× — essentially where it was — while still removing the 1.47× floor on v6/v7, whose overhang
is typical rather than extreme. **I have not made that change**: the brief authorised one specific
correction, and changing an estimator to move a result is the behaviour this project exists to
avoid. It is recorded here as a candidate amendment 02.

---

# 3. Part 0 — baseline learnability on v6, measured before any run-7 change

Written up in full, before Part 1 began, in
[learnability_v6_baseline.md](learnability_v6_baseline.md). Harness:
[learnability.py](learnability.py). Split fixed everywhere: **train ≤ 2023, val 2024, test 2025**,
metrics on test only, features joined as-of the snapshot (`merge_asof`, backward).

## G2 → G3

| Task | metric | G2 best | G3 (LightGBM, no graph) | |
|---|---|---|---|---|
| fill_rate | CRPS ↓ | 0.0616 per-channel mean | **0.0598** | pass, +2.9% |
| arrival_week | C-index ↑ | 0.5573 per-channel median | **0.6444** | pass, above-chance 2.5× |
| shortage_qty | PR-AUC ↑ | **0.5517** per-part-plant rate | 0.5457 | **G3 LOSES** |
| demand_drift | MAE ↓ | 0.1533 global mean | **0.0986** | pass, −35.7% |

On v6, a part-plant's own shortage history beats a GBM given the full observable feature set. In a
world where shortage is dominated by a persistent per-channel demand draw, history is close to
sufficient and the features add nothing.

**Leak check.** Heaviest single feature is 24.5% (`seasonal_naive` on drift, which is a naive
baseline). On the three GNN tasks the top features are all as-of aggregates from
`channel_performance_weekly`, whose visible-week provenance the validator confirms at 100.0% on
disagreeing cells. No leak.

## G4 — the decisive measurement, and it fails

One *round* = channels → {supplier, part, plant} means → back to channels; a 4-layer encoder is two
rounds. Aggregates are concatenated to the channel features and the **same** GBM head refitted, so
only the graph's contribution varies.

| Task | metric | G2 | h⁰ | h¹ | h⁴ | **graph share of above-chance signal** |
|---|---|---|---|---|---|---|
| fill_rate | CRPS ↓ | 0.0616 | 0.0598 | 0.0593 | 0.0594 | **+22%** of a 0.0023 gain |
| arrival_week | C-index ↑ | 0.5573 | 0.6444 | 0.6439 | 0.6440 | **−0.3%** |
| shortage_qty | PR-AUC ↑ | 0.5517 | 0.5457 | 0.5400 | 0.5385 | **−2.2%** |

**h⁰ ≈ h⁴.** On arrival — the task where HADES v3 measured the graph at ~10% of above-chance signal
— the graph contributes nothing here. On shortage it is actively harmful. Only fill gains, and 22%
of a 0.0023 CRPS improvement is 0.0005 in absolute terms.

Run 6 established that the propagation arc is real but −0.195/−0.240 rather than run 5's −0.811
artifact. **This answers the question run 6 left open: an arc of that strength does not produce a
measurable h⁴ − h⁰ gap.** The architecture premise is unsupported on this data.

*What this is not.* The ablation uses fixed mean-pooling with a GBM head, not a learned GNN, which
could weight neighbours better than a mean. But the diagnostic's logic holds in the direction that
matters: if a channel's supplier-, part- and plant-neighbourhood carried information about its own
future, a GBM handed those aggregates should find some of it. It finds none.

---

# 4. Fix A result — the cascade flattened

## Premise check first, and it failed

The brief's Fix A rationale was that `cap_sigma_log 0.40` is "drawn once per supplier per month,
independent across suppliers", so "a seed that draws a tight capacity vector stays tight for the
whole run". Measured on v6 before changing anything:

    K = exp(N(mu + tier, 0.40, size=(NMONTH, NS)))        <- generator_v6.py
    month-to-month rho(log K)          +0.145   (only via shared stress/regime modulation)
    variance share persistent          0.074
    variance share transient           0.926
    aggregate K per month   seed 1001  3,388,194     seed 1002  3,394,334    -> 0.18% apart
    aggregate ordered/month seed 1001  1,677,438     seed 1002  1,999,572    -> 19.2% apart

Capacity is **already** 92.6% transient and already averages out over 126 months; its aggregate is
seed-invariant to 0.18%. Applying the prescribed decomposition to capacity would have moved
variance *into* a persistent component — the opposite of the intent.

The persistent draw is on the **demand** side: `rate = exp(N(log 11, 1.05))`, 16,072 heavy-tailed
draws made once and held for all 535 weeks, with the top 1% of channels carrying 12–13.5% of all
ordered units. So I applied the brief's structure — same decomposition, same variance-preservation
constraint, same §13 left-column class — to the quantity that actually carries the variance.

## The parameters, and the arithmetic that preserves the marginal variance

    log rate(ch, t) = log(11) + level(ch) + eps(ch, t)
    level(ch) ~ N(0, sigma_s^2),  sigma_s = 0.85
    eps(ch,t) = phi * eps(ch,t-1) + innov,  phi = 0.97,  sigma_innov = 0.1499

    sigma_s^2                    = 0.85^2            = 0.7225   (65.5% of variance)
    sigma_innov^2 / (1 - phi^2)  = 0.1499^2 / 0.0591 = 0.3800   (34.5%)
    total                                            = 1.1025   = 1.05^2   OK

Cross-sectional spread at any instant is unchanged at 1.05, so capacity-strain heterogeneity is
preserved by construction; what changes is that a channel's identity as "large" is no longer
permanent. Autocorrelation time (1+φ)/(1−φ) = 66 weeks. Set once on that argument and not revisited.

## Did the cascade flatten? Yes.

| | v6 · 1001 | v6 · 1002 | ratio | v7 · 1001 | v7 · 1002 | ratio |
|---|---|---|---|---|---|---|
| supplier-months constrained | 20.9% | 23.1% | 1.10× | 30.3% | 24.7% | 1.23× |
| shortage condition-weeks | 70,164 | 137,203 | **1.96×** | 74,974 | 48,769 | **1.54×** |
| escalated episodes | 1,911 | 6,072 | **3.18×** | 1,793 | 1,140 | **1.57×** |
| shortage events / yr | 186.5 | 592.5 | **3.18×** | 175.0 | 111.2 | **1.57×** |
| expedites / yr | 136.9 | 487.2 | 3.56× | 136.3 | 83.4 | **1.63×** |
| line stops / yr | 20.1 | 49.6 | 2.47× | 14.3 | 6.4 | **2.23×** |

**The amplification cascade is gone.** In v6 a 1.10× difference in the capacity/demand balance
became 1.96× in conditions and 3.18× in episodes. In v7 the condition and episode ratios are both
≈1.55×, i.e. the escalation stage no longer amplifies — it passes the upstream difference through
roughly unchanged, which is what a consequence-based rule should do once the upstream input is
stable.

**But both seeds now sit at or below the lower band edge**: shortage events 175.0 and 111.2 against
180–250; line stops 14.3 and 6.4 against 15–25. v6 overshot upward on one seed; v7 undershoots on
both. Mechanism: with demand mean-reverting on a 66-week scale, a part-plant that drifts below
safety tends to drift back before an episode reaches the 3-week / 0.30-severity escalation bar, so
fewer conditions escalate.

---

# 5. Fix A regression table — the six capacity-sensitive checks

| Check | band | v6 · 1001 | v7 · 1001 | v6 · 1002 | v7 · 1002 | |
|---|---|---|---|---|---|---|
| supplier-months constrained | 20–30% | 20.9% ✅ | **30.3% ❌** | 23.1% ✅ | 24.7% ✅ | **1001 regressed** |
| capacity unobservable | ≥70% | 79.1% ✅ | **69.7% ❌** | 76.9% ✅ | 75.3% ✅ | **1001 regressed** |
| probe 1 `capacity_strain` vs `load_ratio` | non-zero | +0.6699 | **+0.6687** | +0.6748 | +0.6560 | **held** |
| probe 2 capacity pressure → fill | negative | −0.3081 | **−0.4238** | −0.3562 | −0.3778 | **stronger** |
| probe 2 capacity pressure → lead time | positive | +0.1411 | **+0.1898** | +0.1477 | +0.1708 | **stronger** |
| probe 4 fill constrained vs unconstrained | materially lower | +0.0954 | **+0.1216** | +0.0942 | +0.1159 | **stronger** |

**The thing Fix A was meant to protect held.** Probe 1's `capacity_strain` coefficient moved
+0.6699 → +0.6687 on seed 1001 — 0.0012, well inside seed noise — and probes 2 and 4 all
strengthened. Variance preservation did its job: cross-sectional heterogeneity survived.

The two band failures are a level shift, not a loss of signal: seed 1001's constrained share moved
from mid-band to 0.3 pp over the top, dragging `capacity unobservable` 0.3 pp under its floor. They
are the same number twice.

---

# 6. Full regression table — all fourteen

Both columns measured by the amended validator, so like-for-like.

| Outcome | band | v6·1001 | v7·1001 | v6·1002 | v7·1002 | |
|---|---|---|---|---|---|---|
| PO lines with fill < 1.0 | 8–15% | 11.0 ✅ | **18.0 ❌** | 13.2 ✅ | 14.2 ✅ | **1001 regressed** |
| fill mass at exactly 1.0 | ≥60% | 89.0 ✅ | 82.0 ✅ | 86.8 ✅ | 85.8 ✅ | held |
| fill mass at exactly 0 | 1–3% | 1.9 ✅ | 1.9 ✅ | 1.7 ✅ | 1.8 ✅ | held |
| PO lines arriving late | 15–25% | 18.8 ✅ | 23.8 ✅ | 19.5 ✅ | 20.3 ✅ | held |
| lead-time skewness | ≥0.5 | +3.03 ✅ | +2.79 ✅ | +2.97 ✅ | +2.94 ✅ | held |
| lead median/P90/P99 | right-skew | 26/60/138 | 27/66/146 | 26/59/133 | 26/62/140 | held |
| supplier-months constrained | 20–30% | 20.9 ✅ | **30.3 ❌** | 23.1 ✅ | 24.7 ✅ | **1001 regressed** |
| capacity unobservable | ≥70% | 79.1 ✅ | **69.7 ❌** | 76.9 ✅ | 75.3 ✅ | **1001 regressed** |
| zero-order channel-weeks | 60–99% | 86.3 ✅ | 88.6 ✅ | 85.1 ✅ | 89.1 ✅ | held |
| seasonality peak/trough | >1.10 | 1.396 ✅ | 2.104 ✅ | 1.397 ✅ | 2.09 ✅ | held |
| shortage events / yr | 180–250 | 186.5 ✅ | **175.0 ❌** | 592.5 ❌ | **111.2 ❌** | **1001 regressed** |
| expedites / yr | 100–150 | 136.9 ✅ | 136.3 ✅ | 487.2 ❌ | **83.4 ❌** | 1002 crossed the band |
| line stops / yr | 15–25 | 20.1 ✅ | **14.3 ❌** | 49.6 ❌ | **6.4 ❌** | **1001 regressed** |
| upstream-cause share | ~100% | 97.6 | 97.9 | 98.0 | 94.7 | held |

**Six regressions on seed 1001** (fill<1.0, constrained, unobservable, shortage/yr, line-stops/yr,
and `arriving late [2019-2025]`), and on seed 1002 the three event rates crossed from above the
band to below it. Under this run's rules that outranks Fix A landing, and it is why the verdict is
*not usable*.

---

# 7. Probes — both scripts, all datasets

v1 = [validator_run5_joint_probes.py](validator_run5_joint_probes.py), unchanged, for continuity
with runs 5–6. v2 = [joint_probes_v2.py](joint_probes_v2.py), the measurement of record.

| # | Probe | v6·1001 v1 | v6·1001 **v2** | v7·1001 v1 | v7·1001 **v2** | v7·1002 **v2** |
|---|---|---|---|---|---|---|
| **1** | `fill_rate` vs `fill_rate_last13` | +0.1627 | +0.1627 | +0.2037 | **+0.2037** | +0.1789 |
| **1** | `arrival_week` vs `otd_rate_last13` | −0.1508 | −0.1508 | −0.1301 | **−0.1301** | −0.1247 |
| **1** | `capacity_strain` vs `load_ratio` | +0.6699 | +0.6699 | +0.6687 | **+0.6687** | +0.6560 |
| 2 | pressure → fill | −0.3081 | −0.3081 | −0.4238 | **−0.4238** | −0.3778 |
| 2 | pressure → lead time | +0.1411 | +0.1411 | +0.1898 | **+0.1898** | +0.1708 |
| 3 | late & short, full population | +0.0105 | **+0.2235** | +0.0560 | **+0.2020** | +0.2051 |
| 3 | late & short, delivered only | — | **+0.0684** | — | **+0.1048** | +0.0896 |
| 4 | fill constrained vs unconstrained | +0.0954 | +0.0954 | +0.1216 | **+0.1216** | +0.1159 |
| 5 | upstream cause | 97.59% | 97.59% | 97.94% | **97.94%** | 94.74% |
| **7** | utilisation(m) → fill(m) | — | **−0.5553** | — | **−0.7373** | −0.6618 |
| **7** | utilisation(m) → fill(m+1) | −0.0447 | **−0.1950** | −0.0972 | **−0.3244** | −0.2756 |
| 7 | utilisation persistence | +0.2870 | +0.2870 | +0.3650 | +0.3650 | +0.3444 |
| 8 | staleness → lead-time variance | +0.1748 | +0.1748 | +0.1529 | +0.1529 | +0.1650 |
| 9 | KS vs Uniform(0,1) ×5 | reject | reject | reject | **reject** | reject |

**Probe 1 held on all three tasks.** `capacity_strain` +0.6699 → +0.6687 is the number Fix A was
most at risk of destroying, and it did not move.

**Probe 7 strengthened materially** — within-month −0.5553 → **−0.7373**, next-month −0.1950 →
**−0.3244**. The propagation arc is stronger in v7 than in v6, on the corrected ordered-weighted
measurement. The v1 script reads −0.045 / −0.097 for the same quantity because it averages a
forward-filled, unweighted weekly series.

**Probe 3's correction is worth its own line.** v1 scores every undelivered line as *on time*
(`NaT > date` is `False`, so `.fillna(True)` never fires), which suppressed φ from +0.2235 to
+0.0105 on v6 — a factor of 21. On the corrected measurement the association was always real, and
in v7 it is stronger on the delivered population (+0.0684 → +0.1048, P(late|short) 0.373 vs
P(late|full) 0.242).

---

# 8. Censored tail on the amended check

Covered in §2. Summary: v6 1.47×/1.46× → **0.84×/0.84×** (both now pass on full span); v7
**0.70×/0.70×**; control 0.56× → **0.20×** (now fails). The 2019–2025 window is unchanged
everywhere by construction, sitting at 1.19–1.20× on the generated worlds and 0.50× on the control.

---

# 9. Seven-run trend

| | r1 | r2 | r3 | r4 | r5·1001 | r6·1001 | **r7·1001** | r7·1002 |
|---|---|---|---|---|---|---|---|---|
| passed / failed / skipped | 88/74/20 | 152/24/13 | 167/8/5 | 99/73/22 | 171/1/4 | 170/2/4 | **163/9/4** | 167/4/4 |
| verdict | not usable | not usable | not usable | not usable | usable w/ fixes | usable w/ fixes | **NOT USABLE** | usable w/ fixes |
| conservation | fail | fail | fail | fail | exact | exact | **exact** | exact |
| probe 1 ×3 | ≈0 | ≈0 | ≈0 | ≈0 | non-zero | non-zero | **non-zero** | non-zero |
| probe 7 (ordered-weighted) | — | — | — | — | — | −0.195 | **−0.324** | −0.276 |
| shortage events / yr | fail | fail | fail | 10,213 | 202.9 | 186.5 | **175.0** | 111.2 |
| **G3 shortage vs baseline** | — | — | — | — | — | **loses** | **+83%** | — |
| **G4 graph share** | — | — | — | — | — | **≈0%** | **≈0%** | — |

Runs 1–4 measured the `db/` lineage; 5–7 the rebuilt generator. Runs 1–6 used the pre-amendment
instrument; run 7's v6 column is restated.

---

# 10. Seed variance band

| Metric | v6 spread | v7 spread | |
|---|---|---|---|
| passed / failed | 171/1 vs 169/3 → 2 checks | 163/9 vs 167/4 → 5 checks | wider |
| **escalated episodes** | **3.18×** | **1.57×** | **narrowed — Fix A's target** |
| shortage condition-weeks | 1.96× | 1.54× | narrowed |
| shortage events / yr | 3.18× | 1.57× | **narrowed** |
| expedites / yr | 3.56× | 1.63× | **narrowed** |
| line stops / yr | 2.47× | 2.23× | narrowed |
| supplier-months constrained | 20.9 ↔ 23.1 → 2.2 pp | 30.3 ↔ 24.7 → 5.6 pp | wider |
| fill < 1.0 | 11.0 ↔ 13.2 → 2.2 pp | 18.0 ↔ 14.2 → 3.8 pp | wider |
| late rate | 18.8 ↔ 19.5 → 0.7 pp | 23.8 ↔ 20.3 → 3.5 pp | wider |
| probe 1 `capacity_strain` | 0.005 | 0.013 | comparable |
| probe 7 next-month | 0.045 | 0.049 | comparable |

**The event-rate spread — the number Fix A exists to move — fell from 3.18× to 1.57×.** The
capacity and fill spreads widened, which is the same level-shift instability that produced the six
band regressions.

---

# 11. Learnability, v7 vs the v6 baseline

Identical split, identical harness. Shortage positives derived from the simulation's condition
series for both, so the two are comparable.

| Task | metric | v6 G2 best | v6 G3 | v7 G2 best | v7 G3 | |
|---|---|---|---|---|---|---|
| fill_rate | CRPS ↓ | 0.0616 | 0.0598 | 0.1041 | 0.0987 | gain 0.0018 → **0.0054** |
| arrival_week | C-index ↑ | 0.5573 | 0.6444 | 0.5636 | **0.6510** | above-chance 0.1444 → **0.1510** |
| shortage_qty | PR-AUC ↑ | **0.5517** | 0.5457 *(loses)* | 0.3163 | **0.4692** | **loses → +83%** |
| demand_drift | MAE ↓ | 0.1533 | 0.0986 | 0.1517 | 0.0983 | unchanged |

| Task | v6 h⁰ | v6 h⁴ | v7 h⁰ | v7 h⁴ | graph share v6 → v7 |
|---|---|---|---|---|---|
| fill_rate | 0.0598 | 0.0594 | 0.0987 | 0.0986 | +22% → **+3.6%** |
| arrival_week | 0.6444 | 0.6440 | 0.6510 | 0.6508 | −0.3% → **−0.13%** |
| shortage_qty | 0.5457 | 0.5385 | 0.4692 | 0.4659 | −2.2% → **−1.0%** |

**G3 improved, decisively on the task that mattered.** In v6 a part-plant's own shortage history
beat the GBM; in v7 the GBM beats that history by 83% of above-chance signal, because the transient
demand component means observable features — load ratio, fill history, lead times — now carry
information that history alone does not. Arrival improved. Drift is unchanged.

Fill's absolute CRPS is worse (0.0598 → 0.0987), but that is mechanical: v7's fill distribution is
wider (18.0% below 1.0 versus 11.0%), and CRPS grows with outcome spread. The comparison that
controls for this — the gain over the best naive baseline on the same data — **tripled**, 0.0018 →
0.0054.

**G4 is unchanged and still fails.** The graph contributes ~0% on arrival and negative on shortage
in both worlds. Strengthening the propagation arc from −0.195 to −0.324 did not move it. So the
combination the brief warned about — better bands, worse learning — did **not** occur; v7 learns
*more* and validates *worse*. That is the opposite trade, and it is the more interesting one.

---

# 12. Still failing — mechanism, not number

**1. Six band regressions on seed 1001, all traceable to one mechanism.** The AR(1) transient
demand component (σ_e = 0.616, 66-week autocorrelation) is only partly absorbed by a 10-week EWMA
re-forecast. The residual mismatch between realised demand and the planner's forecast raises
short-run capacity pressure (constrained 30.3%), widens the fill distribution (18.0% below 1.0),
and simultaneously *shortens* shortage episodes, because demand mean-reverts before an episode
reaches the 3-week escalation bar (shortage events 175.0, line stops 14.3).

*The mechanism I would change:* match the forecast horizon to the process. A 10-week EWMA chasing a
66-week process is a deliberate lag; a forecast with memory closer to the autocorrelation time, or
an order-up-to level that includes a forecast-error term, would absorb the transient without
touching its variance. That is a fourth mechanism and I have not made it — the brief forbids
iterating a parameter until a number lands, and `forecast_alpha` is a parameter.

**2. The control now fails the amended censored-tail check at 0.20×.** Diagnosed in §2: driven by
`max(promise)` as an extreme-value statistic, with the control's 176-day overhang set by a handful
of 212-day-lead lines against a P99.9 of 6 days. Candidate amendment 02, not made.

**3. G4 — the graph contributes nothing, in both worlds.** This is not a v7 regression; it is the
standing state of the project, now measured. §10's arcs exist and are strong on the joint probes
(−0.7373 within month), but they do not translate into a measurable h⁴ − h⁰ gap on any GNN task.

*The mechanism I would change:* the arcs propagate through supplier **capacity**, which every
channel of that supplier shares — so the neighbourhood mean that a GNN aggregates is nearly
collinear with the channel's own `load_ratio` feature, which h⁰ already has. The graph would only
add signal if neighbours carried information a channel's own features do not: part-level substitution,
shared logistics checkpoints, or Tier-2 correlated failure, all of which §10 and §11 describe and
none of which the generator currently propagates as an observable.

---

# 13. Tier tables, compliance, prohibitions

Full transcripts: [v7·1001](validator_run7_v7_1001.txt), [v7·1002](validator_run7_v7_1002.txt),
[v6·1001 restated](validator_run7_v6restated_1001.txt),
[v6·1002 restated](validator_run7_v6restated_1002.txt), [control](validator_run7_control.txt).

**Tier 1 — structural.** 49/49 tables discovered and structurally clean on both v7 seeds. No
missing or forbidden column, type or enum violation, NOT NULL breach, duplicate PK, or orphan FK.

**Tier 2 — as-of and leakage.** All pass on both seeds:

    [ok] channel store conserves ordered units    265,126,832 vs 265,126,832   exact
    [ok] supplier store conserves ordered units   265,126,832 vs 265,126,832   exact
    [ok] derived store buckets on visible week    max(event_week, recorded_week)
    [ok] label window opens after the snapshot
    [ok] weekly store is a panel                  100.0%
    [ok] PO lines per channel per year            5.79 (validator gate >= 3.0); 6.26 per traded channel (§2 >= 6)

**Tier 3.** The fourteen in §6.

**4 skips**, unchanged from runs 5–6 — skipped is not passed: `inventory_position_weekly` and
`plan_drift_features` two-timestamp checks (no `event_ts` column in the spec), and `entity_id`
resolution for the composite `part_plant` and `product_plant` ids.

## `synthetic_rules.md` compliance

| § | Status |
|---|---|
| §0 7 plants; ~18 line stops/yr; ~120 expedites/yr | **partially met** — expedites 136.3 ✅, line stops 14.3 ❌ |
| §1 ten years + partial trailing year; stores derived | **met** (535 weeks, cut 2026-03-31) |
| §2 entity minimums | **met** (all nine) |
| §3.1–3.3 as-of gate, recording layer, visible week | **met** |
| §4 exact conservation, ledger, transfers | **met** (exact both seeds; ledger holds) |
| §5 sparsity, contiguity, no dup PKs | **met** (88.6%, 100%, 0) |
| §6 `min(K, ordered)`; observability emergent | **partially met** — 30.3% constrained on 1001 (band 20–30%) |
| §7 fill masses | **partially met** — mass at 1.0 82.0% ✅, at 0 1.9% ✅, below 1.0 18.0% ❌ |
| §8 survival process, censoring, skew | **met** (skew +2.79, censoring 45.2% emergent, tail 0.70×) |
| §9 shortage from stock balance; two populations; ~100% traced | **partially met** — 97.9% traced ✅, event rates below band ❌ |
| §10 feedback arcs | **met and strengthened** (probe 7 −0.7373 / −0.3244) |
| §11 two disruption classes | **met** |
| §12 demand trend/seasonality | **met** (ratio 2.104) |
| §13 mechanism params set, outcomes measured | **met** — literals audit clean on v7 |
| §14 labels forward, 1:1 binding, censoring 1–60% | **met, and improved** — the shortage label table now carries negatives |
| §15 prohibitions | **all 15 pass** |
| §16 ladder | **G0–G3 pass, G4 FAILS** |
| §17 two seeds + variance band | **met** (§10) |
| §18–19 | met |

**§15 prohibitions — all 15 pass**, unchanged from run 6, with one strengthened: *"never write a
gate that cannot fail"* now has a companion finding — amendment 01 removed a gate that could not
*pass*.

---

# 14. Readiness ladder and gate readiness per task

| Gate | Status | Note |
|---|---|---|
| **G0** | **pass** both seeds | conservation exact, censoring 2.9–45.2% on all five tasks |
| **G1** | **pass** | median channel degree 3, one component, 100% coverage |
| **G2** | **pass — computed** | §3 and §11 |
| **G3** | **pass on v7, all four tasks** | and shortage moved from losing to the naive baseline to beating it by 83% |
| **G4** | **FAIL** | graph share ≈0% on arrival, −1.0% on shortage, +3.6% of a small gain on fill. **h⁰ ≈ h⁴** |
| G5 | not yet run | runnable — `recorded_ts` present and non-degenerate |
| G6 | not yet run | label window opens strictly after the snapshot |
| G7 | **partial** | cold-start 4.59% present; seed band published and narrowed on events, widened on capacity |

**The ladder now stops at G4, not G1.** That is the substantive change this run produced: two gates
were opened and the one that matters is now measured and failing.

## Gate readiness per task

| Task | Trainable? | Note |
|---|---|---|
| **Delivery risk** | **Yes, on both seeds** | C-index 0.6510 vs 0.5636 naive; censoring 45.2% emergent from genuinely open POs; lead skew +2.79. Fit on 2019–2025 where the censored tail is 1.19×. |
| **Fill rate** | **Yes, on both seeds** — but the band regressed | CRPS gain over baseline tripled (0.0018 → 0.0054). `PO lines with fill < 1.0` is 18.0% on seed 1001 against an 8–15% band, so the world is learnable but out of spec. |
| **Part shortage** | **Trainable on both seeds for the first time, and no longer band-compliant on either** | The label table now carries negatives (124,500 rows, 12.8%/9.2% positive), the GBM beats per-part-plant history by 83%, and the seed spread in event rates fell 3.18× → 1.57×. But the rates themselves sit below band on both seeds (175.0 and 111.2 against 180–250). The blocker moved from *seed dependence* to *level*. |

**Answering the question the brief put:** part shortage is no longer seed-dependent in the way run 6
described — the cascade that made seed 1002 unusable is gone, and the task is genuinely learnable on
both seeds for the first time. It is not band-compliant on either, and v7 as a whole is not usable
until the six seed-1001 regressions in §12.1 are paid down.
