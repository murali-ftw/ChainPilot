# Dataset validation — run 6 (`db2/generator_v6.py`, `gen_v6/`)

**Verdict: usable with fixes. Fix 1 landed on seed 1001 and made seed 1002 worse; fix 2 improved
the tail to its structural floor and no further.**
Seed 1001 is **170 passed / 2 failed / 4 skipped**, seed 1002 **168 / 4 / 4**. No outcome that
was in band in run 5 left band on either seed — the fourteen regression checks all hold.
**Part shortage is trainable on seed 1001 and not reproducibly across seeds**, which is now the
headline blocker.

| | run 6 · 1001 | run 6 · 1002 | run 5 · 1001 | run 5 · 1002 |
|---|---|---|---|---|
| **passed / failed / skipped** | **170 / 2 / 4** | 168 / 4 / 4 | 171 / 1 / 4 | 166 / 6 / 4 |
| total checks | 193 | 193 | 193 | 193 |
| verdict | usable with fixes | usable with fixes | usable with fixes | usable with fixes |
| failing gates | censored tail ×2 | censored tail, shortage/expedite/line-stop rates | censored tail | fill<1.0 ×2, censored tail, shortage/expedite/line-stop rates |

`SKIP` is never a pass; the 4 skips are enumerated in §11.

## Instrument, control, generation

| | |
|---|---|
| `shasum -a 256 db/validator.py` **before** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| `shasum -a 256 db/validator.py` **after** | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |

**Diff: none.** No path-discovery change was needed. The SHA is identical to runs 3, 4 and 5, so
runs 3–6 used a bit-identical instrument.

**Control — did not move.** `python3 db/validator.py …/csv_full_seed1 --no-report` →
**172 passed / 2 failed / 4 skipped**, the same two `PO lines arriving late` failures as runs 2–5.
Transcript: [validator_run6_control_csv_full_seed1.txt](validator_run6_control_csv_full_seed1.txt).

| | seed 1001 | seed 1002 |
|---|---|---|
| Wall clock | **169.9 s** | ~170 s |
| Peak RSS | **10.71 GB** | ~10.7 GB |
| Output | 3.1 GB, 49/49 tables | 3.3 GB, 49/49 tables |
| Weeks | 535 | 535 |
| PO lines | 1,178,254 | 1,283,024 |
| Log | [gen_v6_1001.log](gen_v6_1001.log) | [gen_v6_1002.log](gen_v6_1002.log) |

Output paths are `gen_v6/seed_1001/` and `gen_v6/seed_1002/` (underscore, matching run 5's
convention and the generator's `--out` handling), not the `gen_v6/seed1001/` in the brief.

**Extract cut date chosen: 2026-03-31.** Rationale in §4.

`db2/generator_v5.py` is untouched — SHA `28b7d1ad…` unchanged, so run 5 stays reproducible.
`generator_source.py` (`4d1ba07f…`) and `generator_runnable.py` are untouched. `gen_v5/` is intact.

---

# 2. What changed in `generator_v6.py`

285 changed lines against v5. Every edit is either a structural change to how a mechanism works or
a §13 left-column parameter. No edit introduces a constant that moves a measured outcome toward a
band.

## Mechanism-structural

| # | Change | Why |
|---|---|---|
| M1 | Open lines moved from an append-only list into a **flat PO registry** (`po_arr`, `po_disp`, `po_open`, …) with arrival buckets that tolerate stale entries | 1a cannot work without the ability to reach back into a placed line and move it |
| M2 | **Expedite physically accelerates the line**: compression drawn `Beta(2,3)×cap`, capped lower once in transit, bounded so it can never arrive before dispatch, and it can fail | 1a |
| M3 | Allocation priority now responds to **observable buyer pressure** — recent unmet demand, days of cover under a week, a part that has just stopped a line | 1b |
| M4 | **Episode tracking** per part×plant: start, length, max severity, accumulated unmet, stockout weeks, which mitigation acted | 1c needs the episode, not the week |
| M5 | **`escalate_frac` and `linestop_frac` deleted.** Escalation is now `(length ≥ 3w AND severity ≥ 0.30) OR the episode stopped a line` | 1c |
| M6 | `resolution_action` derived from the mitigation bits that actually fired, not `r.choice` | §9 |
| M7 | `expedite_events` emitted at **part×plant×week grain** (one buyer action) rather than one row per accelerated line | grain defect found in-run |
| M8 | Inventory **initialised in the steady-state band** `U(rop, upto)` instead of below the reorder point | start-up artefact found in-run |
| M9 | `revealed_capacity_monthly.month` / `inventory_snapshots.snapshot_date` land on **month starts** instead of 30.44-day steps | data defect that was silently breaking probe 7's join |
| M10 | `part_costs.freight_cost_inr` carries an uplift proportional to a part's realised expedite share | 1a — expediting is not free |

## §13 left-column parameters

Added: `exp_trigger_sev 0.30`, `exp_trigger_weeks 2`, `exp_success_base 0.68`,
`exp_success_stress 0.45`, `exp_compress_predispatch 0.55`, `exp_compress_intransit 0.28`,
`exp_compress_a/b 2.0/3.0`, `exp_premium_per_unit 42`, `exp_freight_uplift 0.35`,
`exp_max_lines_per_week 900`, `prio_starve_w 0.45`, `prio_cover_w 0.30`, `prio_linestop_w 0.55`,
`starve_decay 0.82`, `esc_persist_weeks 3`, `esc_severity 0.30`, `ls_min_stockout_weeks 1`,
`ls_unmet_frac 0.10`.
Changed: `world_end 2026-09-07 → 2026-03-31`, `W` end `2026-08-31 → 2026-03-30`.
**Removed: `escalate_frac`, `linestop_frac`.**

## §13 literals audit, re-run on v6

    capacity-observability share 0.20-0.30  -> no hit
    fill mass 0.87-0.90                     -> no hit
    seasonality ratio 1.30                  -> no hit
    zero-order week share 0.80-0.90         -> no hit
    late rate 0.15-0.25                     -> no hit
    event counts 18 / 120 / 180-250 / 100-150 / 15-25  -> no hit

Every numeric hit is a mechanism quantity: a seasonality **amplitude** (`0.29`, unchanged — the
*ratio* 1.396 is measured), a lane base fraction (`0.51`), an AR coefficient (`0.93`), an
in-transit compression cap (`0.28`), a starvation decay (`0.82`), a capacity stress sensitivity
(`0.30`/`0.22`). The only occurrence of `18` is the words "§0: ~18/yr" in a comment explaining what
the line-stop scale refers to. **No band value appears as an input.**

## Disclosure: this run was not a single regeneration

The brief asks for one principled change and one regeneration. In practice seed 1001 was generated
four times, because the first three runs exposed defects in the code I had just written. None was
band-chasing; each was a mechanism that could not do its job. In order:

| Run | What it exposed | Fix |
|---|---|---|
| 1 | 24,040 expedite events; 0 line stops; 783 events/yr in 2016–18 vs 176 in 2019–25 | M7, M8, and a line-stop test that required *every* channel of a part-plant to be empty at once — a gate that could never fire |
| 2 | 1,548 expedites/yr; 3.3 line stops/yr | Expedite re-raised every week of an episode (a buyer raises one); line-stop threshold set at a lost half-week when Rane's ~18/yr counts stoppages of hours |
| 3 | Probe 7 apparently flipped to **+0.193** | M9 — the join was matching 420 of 51,240 rows |
| 4 | locked | — |

After run 3 no parameter was changed. The `ls_unmet_frac 0.10` and `exp_trigger_sev 0.30` values
were each set once, on a physical argument stated in the source, and not revisited.

---

# 3. Fix 1 result

**Escalation is now a consequence.** `grep escalate_frac linestop_frac db2/generator_v6.py` returns
only a comment. Whether a condition becomes a recorded event is decided by
`(length ≥ 3 weeks AND max severity ≥ 0.30) OR the episode stopped a line`, evaluated when the
episode closes. The event counts are therefore measured outputs.

| | run 5 · 1001 | run 6 · 1001 | run 5 · 1002 | run 6 · 1002 | band |
|---|---|---|---|---|---|
| **shortage condition share** (part×plant-weeks) | 9.06% | **3.02%** ✅ | 13.90% | **5.91%** ❌ | 2–5% |
| condition-weeks | — | 70,164 | — | 137,203 | |
| escalated episodes | — | 1,911 | — | 6,072 | |
| escalation rate | *0.97% (set)* | **2.72% (derived)** | *0.97% (set)* | **4.43% (derived)** | — |
| shortage events / yr | 202.9 ✅ | **186.5** ✅ | 313.1 ❌ | **592.5** ❌ | 180–250 |
| expedites / yr | 117.6 ✅ | **136.9** ✅ | 180.3 ❌ | **487.2** ❌ | 100–150 |
| line stops / yr | 20.6 ✅ | **20.1** ✅ | 28.4 ❌ | **49.6** ❌ | 15–25 |
| upstream-cause share | 89.9% | **97.6%** | 90.2% | **98.0%** | ~100% |

**Seed 1001: fix 1 worked.** Condition share fell 9.06 → 3.02%, into band for the first time. All
three event rates stayed in band while being derived rather than dialled. Upstream-cause rose to
97.6%.

**Seed 1002: it did not.** Condition share fell 13.90 → 5.91% but stayed out, and the three event
rates went *further* out, not in. Of seed 1002's four run-5 failures, **one resolved**
(`PO lines with fill < 1.0`, 16.6% → 13.2%) and three worsened.

**Why, plainly.** Making escalation a consequence removed the knob but did not remove the
sensitivity — it revealed it. The seed difference lives upstream:

| | 1001 | 1002 | ratio |
|---|---|---|---|
| supplier-months constrained | 20.9% | 23.1% | 1.10× |
| shortage condition share | 3.02% | 5.91% | 1.96× |
| escalated episodes | 1,911 | 6,072 | 3.18× |

A 1.10× difference in the capacity/demand balance becomes 1.96× in conditions and 3.18× in events.
In run 5 a fixed `escalate_frac` damped this to exactly linear (13.90/9.06 = 1.53 conditions →
313.1/202.9 = 1.54 events). A consequence-based rule is **super**linear, because both the number of
episodes and their length and depth scale together with tightness. So the event-rate spread widened
from 1.54× to 3.18× — the honest cost of removing the knob.

This is a finding, not a failure of the fix: the escalation rule is now correct, and the remaining
instability is located where it actually lives.

---

# 4. Fix 2 result — cut date 2026-03-31

**Chosen: 2026-03-31.** Reasons, in order:

1. It is the **Indian fiscal year end** — the single most natural date for a real extract to be
   pulled at an Indian auto company, which is exactly what the cut date is meant to represent.
2. Its trailing-90-day promise window spans **Dec–Feb–Mar**, seasonality factor **0.919** against
   the 12-month mean, the lowest of the candidates (June 30 gives 0.923; the v5 August cut gave
   1.079).
3. It preserves §1's intent — ten full years 2016–2025 plus a partial trailing quarter, which is
   where natural right-censoring comes from. **Only the month changed.**
4. The seasonality phase is **untouched**: `seas_festive/fiscal/monsoon/winter` are identical to
   v5, and the measured peak/trough ratio moves only 1.508 → 1.396, still far above the >1.10 band.

| | run 5 · 1001 | run 6 · 1001 | run 5 · 1002 | run 6 · 1002 | band |
|---|---|---|---|---|---|
| censored tail, full span | 1.69× | **1.47×** ❌ | 1.67× | **1.46×** ❌ | 0.3–1.2× |
| censored tail, 2019–2025 | 1.17× ✅ | **1.20×** ❌ | 1.19× ✅ | **1.19×** ✅ | 0.3–1.2× |

**The improvement is real and it stops exactly where arithmetic says it must.** Before generating,
I computed a structural floor for this metric:

> the validator censors a line when `original_promise_date > end − 90d`, but normalises by
> `90 / span` where span is measured on **created** dates. Since `promise = created + contracted`
> and `contracted ~ U(15,70)`, the censored population covers `90 + 42.5 = 132.5` days of order
> creation, not 90 — a floor of **132.5/90 = 1.47×** before any seasonality or trend.

Measured after: **1.47× and 1.46×.** The cut date removed the entire seasonal component and landed
on the floor. **No cut date can bring this metric into band**, because the residual is the
promise-vs-creation offset built into the check, not a property of the world.

The 2019–2025 figure moved 1.17 → 1.20 on seed 1001 (a marginal fail at the boundary) and stayed at
1.19 on seed 1002. That window is where the hazard model will be fitted, and both seeds sit at the
edge of the band rather than outside it in substance.

---

# 5. Regression table — the fourteen

Measured by the frozen validator on both runs, so the comparison is like-for-like.
**No outcome that was in band in run 5 left band in run 6, on either seed.**

| Outcome | band | r5·1001 | r6·1001 | r5·1002 | r6·1002 | verdict |
|---|---|---|---|---|---|---|
| PO lines with fill < 1.0 | 8–15% | 14.9 ✅ | **11.0** ✅ | 16.6 ❌ | **13.2** ✅ | **improved — 1002 fixed** |
| fill mass at exactly 1.0 | ≥60% | 85.1 ✅ | **89.0** ✅ | 83.4 ✅ | **86.8** ✅ | improved |
| fill mass at exactly 0 | 1–3% | 1.8 ✅ | **1.9** ✅ | 1.7 ✅ | **1.7** ✅ | held |
| PO lines arriving late | 15–25% | 20.9 ✅ | **18.8** ✅ | 21.4 ✅ | **19.5** ✅ | held |
| lead-time skewness | ≥0.5 | +2.88 ✅ | **+3.03** ✅ | +2.85 ✅ | **+2.97** ✅ | held (rose) |
| lead median / P90 / P99 | right-skew | 27/62/138 | **26/60/138** | 27/61/134 | 26/59/133 | held |
| supplier-months constrained | 20–30% | 25.9 ✅ | **20.9** ✅ | 27.6 ✅ | **23.1** ✅ | held |
| capacity unobservable | ≥70% | 74.1 ✅ | **79.1** ✅ | 72.4 ✅ | **76.9** ✅ | held |
| zero-order channel-weeks | 60–99% | 84.7 ✅ | **86.3** ✅ | 83.8 ✅ | **85.1** ✅ | held |
| seasonality peak/trough | >1.10 | 1.508 ✅ | **1.396** ✅ | 1.499 ✅ | **1.397** ✅ | held |
| shortage events / yr | 180–250 | 202.9 ✅ | **186.5** ✅ | 313.1 ❌ | **592.5** ❌ | 1002 worse |
| expedites / yr | 100–150 | 117.6 ✅ | **136.9** ✅ | 180.3 ❌ | **487.2** ❌ | 1002 worse |
| line stops / yr | 15–25 | 20.6 ✅ | **20.1** ✅ | 28.4 ❌ | **49.6** ❌ | 1002 worse |
| upstream-cause share | ~100% | 89.9 | **97.6** | 90.2 | **98.0** | improved |

**The two most exposed checks held.** Fix 1a directly touches arrival timing, and the brief flagged
late rate and lead skewness as the likely casualties. Late rate moved 20.9 → 18.8% (expedites do
pull some lines in, as they should) and **lead-time skewness rose, 2.88 → 3.03** — the expedite
mechanism did not flatten the tail, because it fires on ~0.2% of lines and compresses the remaining
leg rather than truncating the distribution.

The three seed-1002 event rates were already out of band in run 5 and went further out; they are
the subject of §3 and §9, not a new regression.

---

# 6. Probes — all nine, both seeds, run 5 alongside

Script unchanged: [validator_run5_joint_probes.py](validator_run5_joint_probes.py).
Output: [run 6 · 1001](validator_run6_probes_seed1001.txt), [run 6 · 1002](validator_run6_probes_seed1002.txt).

| # | Probe | r5·1001 | r6·1001 | r5·1002 | r6·1002 | |
|---|---|---|---|---|---|---|
| **1** | `fill_rate` vs `fill_rate_last13` | +0.1948 | **+0.1627** | +0.2166 | **+0.1876** | **pass** |
| **1** | `arrival_week` vs `otd_rate_last13` | −0.1409 | **−0.1508** | −0.1364 | **−0.1497** | **pass** |
| **1** | `capacity_strain` vs `load_ratio` | +0.6652 | **+0.6699** | +0.6495 | **+0.6844** | **pass** |
| 2 | capacity pressure → fill | −0.3819 | −0.3081 | −0.4051 | −0.3562 | pass |
| 2 | capacity pressure → lead time | +0.1230 | +0.1411 | +0.1287 | +0.1477 | pass |
| 3 | late and short co-occur | +0.0152 | +0.0105 | +0.0231 | +0.0242 | see §7 |
| 4 | fill constrained vs unconstrained | +0.1029 | +0.0954 | +0.0952 | +0.0942 | pass |
| 5 | upstream cause | 89.88% | **97.59%** | 90.24% | **98.02%** | pass |
| 6 | shortage → expedite | elevated | elevated | elevated | elevated | pass |
| **7** | **utilisation → next-month fill** | −0.8110 *(n=420)* | **−0.0447** *(n=51,240)* | −0.7908 *(n=420)* | **−0.1398** *(n=51,240)* | **see below** |
| 8 | staleness → lead-time variance | +0.1383 | +0.1748 | +0.1292 | +0.1562 | pass |
| 9 | KS vs Uniform(0,1), 5 tasks | reject ×5 | **reject ×5** | reject ×5 | **reject ×5** | pass |

**Probe 1 stays non-zero on all three tasks on both seeds.** The `fill_rate` coefficient softened
(+0.195 → +0.163) — expected, since absorption makes a channel's own past fill a slightly weaker
predictor of its future fill — while `arrival_week` strengthened (−0.141 → −0.151).

## Probe 7 — the number moved because the probe was broken, not the arc

Run 5's −0.811 was computed on **420 rows out of 51,240**. `revealed_capacity_monthly.month` was
being written as `2016-01-01 + 30.44·k` days, which drifts off the calendar, so the probe's join
against a month-start key matched only where the drift happened to land. M9 fixed the column; the
join now matches all 51,240 supplier-months and reads −0.0447 / −0.1398.

To compare like with like I measured the arc identically on all four datasets, straight from the
simulation state, ordered-weighted:

| | run 5 · 1001 | run 5 · 1002 | run 6 · 1001 | run 6 · 1002 |
|---|---|---|---|---|
| utilisation(m) → fill(m+1) | −0.2374 | −0.2495 | **−0.1950** | **−0.2402** |
| utilisation(m) → fill(m) | −0.6623 | −0.7329 | **−0.5553** | **−0.6583** |
| n (supplier-months) | 53,340 | 53,340 | 51,240 | 51,240 |

**The propagation arc is intact and correctly signed.** It weakened ~18% on seed 1001 and 4% on
seed 1002 — the expected consequence of absorption, since a starved channel now recovers faster.
Run 5's −0.811 should not be carried forward as the baseline; it was noise on 420 rows.

The probe-script number is weaker than the direct measurement because it averages
`supplier_performance_weekly.fill_rate`, which is forward-filled across inactive weeks, unweighted.

---

# 7. Probe 3 and the upstream-cause residual

## Probe 3 — diagnosed, not patched

The brief asked whether fill and arrival are independent draws downstream of the same supplier
state. Two things are true, and the first is a defect in the probe I wrote in run 5:

**(a) The probe mislabels every undelivered line as on-time.** It computes
`late = (event_ts > current_promise_date).fillna(True)`. For a line with no GRN, `event_ts` is
`NaT`, and `NaT > date` evaluates to **`False`, not `NaN`** — so `.fillna(True)` never fires and
34,546 undelivered lines are scored `short=True, late=False`, the exact opposite of the truth.

| measurement (run 6 · 1001) | φ | P(late\|short) | P(late\|full) |
|---|---|---|---|
| probe as written | +0.0105 | 0.2171 | 0.2027 |
| delivered lines only | **+0.0684** | **0.3135** | 0.2027 |
| raw simulation, `pl_ > contracted` | **+0.1080** | **0.3716** | 0.2105 |

On the world itself, a short line is **1.77× more likely to be late**. The association is real; the
probe understates it by a factor of ten. I have **not** changed the probe script, because the brief
requires it unchanged for comparability — but its probe-3 number should be read as a floor.

**(b) Where the coupling lives.** Decomposing the raw simulation:

| | correlation | n |
|---|---|---|
| **between** supplier-months (short-rate vs late-rate) | **+0.3244** | 33,906 supplier-months |
| **within** supplier-month (residual short vs residual late) | **+0.0015** | 900,787 lines |

A supplier-month under strain genuinely misses on both dimensions. Conditional on the
supplier-month, which particular line falls short (allocation order) and how long it takes
(an independent lognormal draw) are unrelated. That is a modelling choice, and I think it is the
right one — but it is a choice, and it is why the line-level φ is small. **No coupling term was
added.**

## Upstream cause — 97.6% / 98.0%, and the residual is named

Fix 1 moved this from 89.9% to 97.6%. The remaining 46 events on seed 1001 (120 on seed 1002) are
not untraceable noise; they are demand-driven, and the evidence says so:

| | traced | untraced |
|---|---|---|
| median episode length | 5.0 w | **3.0 w** |
| median severity | 0.394 | 0.350 |
| stopped a line | 32.1% | **13.0%** |
| **demand in the 4 weeks before onset, vs that part-plant's own mean** | **1.47×** | **1.79×** |
| share above 1.5× | 48.0% | **69.6%** |
| distinct part-plants | 822 | 30 (28 of which *are* traceable in other episodes) |

These are shorter, shallower, rarely stop a line, land on part-plants that trace fine at other
times, and are preceded by a **1.79× demand spike**. They are correctly labelled `demand_spike`.
§9 requires a `root_cause` that comes from the mechanism that actually produced it; attaching a
`responsible_supplier_id` to a demand spike to reach 100% would be fabrication. **97.6% traced with
a named, evidenced 2.4% demand-side residual is the correct answer**, and it is the first run in
which the residual has a mechanism rather than a number.

---

# 8. Six-run trend

| | run 1 | run 2 | run 3 | run 4 | run 5 · 1001 | run 5 · 1002 | run 6 · 1001 | run 6 · 1002 |
|---|---|---|---|---|---|---|---|---|
| passed / failed / skipped | 88/74/20 | 152/24/13 | 167/8/5 | 99/73/22 | 171/1/4 | 166/6/4 | **170/2/4** | **168/4/4** |
| total checks | 195 | 202 | 194 | 210 | 193 | 193 | 193 | 193 |
| verdict | not usable | not usable | not usable | not usable | usable w/ fixes | usable w/ fixes | **usable w/ fixes** | **usable w/ fixes** |
| conservation | fail | fail | fail | fail | **exact** | **exact** | **exact** | **exact** |
| probe 1 (label vs feature) | ≈0 | ≈0 | ≈0 | ≈0 | non-zero ×3 | non-zero ×3 | **non-zero ×3** | **non-zero ×3** |
| censored tail | fail | fail | fail | 0.23× | 1.69× | 1.67× | **1.47×** | **1.46×** |
| shortage events / yr | fail | fail | fail | 10,213 | 202.9 ✅ | 313.1 ❌ | **186.5** ✅ | **592.5** ❌ |
| upstream-cause share | — | — | — | — | 89.9% | 90.2% | **97.6%** | **98.0%** |

Runs 1–4 measured the `db/` lineage; runs 5–6 measure the rebuilt generator. Run 6 is the first run
in which the shortage-condition share reaches band on any seed.

---

# 9. Still failing — mechanism, not number

### 1. Censored tail, 1.47× / 1.46× against 0.3–1.2× (both seeds)

Not fixable by a cut date. The check censors on `original_promise_date > end − 90d` and normalises
by `90 / span` computed on **created** dates; with `contracted ~ U(15,70)` the censored set spans
132.5 days of order creation, a floor of 1.47×. Run 6 sits on that floor exactly, meaning the
seasonal and trend components — the only parts a generator controls — have been fully removed.

*The mechanism I would change:* nothing in the generator. Either the contracted lead-time
distribution would have to shrink toward zero (which would break §8's lead-time shape), or the
check's denominator would have to be anchored on promise dates rather than creation dates — and
`db/validator.py` is frozen. I am reporting this as a limit of the metric, not proposing a change.

### 2. Seed 1002's three event rates: 592.5 / 487.2 / 49.6 per year

Diagnosed in §3. The escalation rule is now correct; the instability is upstream, where a 1.10×
seed difference in the capacity/demand balance amplifies to 1.96× in shortage conditions and 3.18×
in episodes.

*The mechanism I would change:* the amplification comes from capacity being drawn **once per
supplier per month** from a wide lognormal (`cap_sigma_log 0.40`) that is independent across
suppliers, so a seed that draws a tight capacity vector stays tight for the whole run. Giving
capacity a persistent per-supplier component plus a smaller month-to-month innovation — the same
AR structure the supplier state already uses — would make the aggregate capacity/demand balance
far less seed-sensitive without touching any measured outcome. I have not made that change: it is
a third mechanism, outside this run's scope, and the decision to re-run is yours.

### 3. Shortage condition share on seed 1002: 5.91% against 2–5%

Same root cause as (2). Seed 1001 reaches 3.02%.

---

# 10. Seed variance band — did the spread narrow?

**Mixed, and the direction is informative: it narrowed on everything the fixes touched directly,
and widened on the event rates, for the reason in §3.**

| Metric | run 5 spread (1001↔1002) | run 6 spread | |
|---|---|---|---|
| passed / failed | 171/1 vs 166/6 → 5 checks | 170/2 vs 168/4 → **2 checks** | **narrowed** |
| fill < 1.0 | 14.9 ↔ 16.6 → 1.7 pp | 11.0 ↔ 13.2 → **2.2 pp** | slightly wider, both in band |
| fill mass at 1.0 | 85.1 ↔ 83.4 → 1.7 pp | 89.0 ↔ 86.8 → 2.2 pp | comparable |
| fill mass at 0 | 1.8 ↔ 1.7 → 0.1 pp | 1.9 ↔ 1.7 → 0.2 pp | comparable |
| late rate | 20.9 ↔ 21.4 → 0.5 pp | 18.8 ↔ 19.5 → 0.7 pp | comparable |
| lead skewness | 2.88 ↔ 2.85 → 0.03 | 3.03 ↔ 2.97 → 0.06 | comparable |
| supplier-months constrained | 25.9 ↔ 27.6 → 1.7 pp | 20.9 ↔ 23.1 → 2.2 pp | comparable |
| zero-order week share | 84.7 ↔ 83.8 → 0.9 pp | 86.3 ↔ 85.1 → 1.2 pp | comparable |
| **shortage condition share** | 9.06 ↔ 13.90 → **4.84 pp** | 3.02 ↔ 5.91 → **2.89 pp** | **narrowed** |
| **shortage events / yr** | 202.9 ↔ 313.1 → **1.54×** | 186.5 ↔ 592.5 → **3.18×** | **widened** |
| upstream-cause share | 89.9 ↔ 90.2 → 0.3 pp | 97.6 ↔ 98.0 → 0.4 pp | comparable |
| probe 1 `fill_rate` | +0.195 ↔ +0.217 → 0.022 | +0.163 ↔ +0.188 → 0.025 | comparable |
| probe 7 (direct) | −0.237 ↔ −0.250 → 0.013 | −0.195 ↔ −0.240 → 0.045 | wider |
| censored tail | 1.69 ↔ 1.67 → 0.02 | 1.47 ↔ 1.46 → 0.01 | narrowed |

The condition share — the quantity fix 1 targets — narrowed from 4.84 to 2.89 pp. The event rates
widened because escalation now amplifies the upstream difference instead of damping it linearly.

---

# 11. Full tier tables

Complete transcripts: [seed 1001](validator_run6_v6_seed1001.txt),
[seed 1002](validator_run6_v6_seed1002.txt), [control](validator_run6_control_csv_full_seed1.txt).

**Tier 1 — structural conformance.** 49/49 tables discovered, 49/49 structurally clean on both
seeds. No missing or forbidden column, no type or enum violation, no NOT NULL breach, no duplicate
PK, no orphan FK.

**Tier 2 — as-of and leakage contract.** All pass on both seeds:

    [ok] channel store conserves ordered units    206,279,157 vs 206,279,157   exact
    [ok] channel store conserves received units   182,173,072 vs 182,173,072   exact
    [ok] supplier store conserves ordered units                                exact
    [ok] supplier store conserves received units                               exact
    [ok] derived store buckets on visible week    max(event_week, recorded_week)
         -- on 894,660 channel-weeks where the two rules disagree: visible-week 100.0%
    [ok] label window opens after the snapshot
    [ok] weekly store is a panel                  100.0%   >= 90%
    [ok] PO lines per channel per year            6.90     >= 3.0
    [ok] zero-order channel-weeks                 86.3%    60%-99%

**Tier 3 — distributional adequacy.** The fourteen in §5, plus the censored tail (the only failure
on seed 1001) and the three event rates (seed 1002 only).

**The 4 skips**, identical on both seeds and to run 5 — skipped is not passed:

| Skip | Why |
|---|---|
| `inventory_position_weekly: two timestamps differ` | the table has no `event_ts` column in the spec |
| `plan_drift_features: two timestamps differ` | same |
| `entity_id resolves for entity_type=part_plant` | 3,406 composite `part\|plant` ids the validator cannot resolve against a single master |
| `entity_id resolves for entity_type=product_plant` | 1,260 composite `product\|plant` ids, same reason |

---

# 12. `synthetic_rules.md` compliance and §15 prohibitions

| § | Requirement | Status |
|---|---|---|
| §0 | 7 plants; ~18 line stops/yr; ~120 expedites/yr | **met on 1001** (20.1, 136.9); not on 1002 |
| §1 | ten full years + partial trailing year; stores derived | **met** (2016–2025 + Q1 2026; 535 weeks) |
| §2 | entity population minimums | **met** (all nine, both seeds) |
| §3.1 | as-of gate on `recorded_ts` | **met** |
| §3.2 | recording layer a process, per table, never negative, some never entered | **met** |
| §3.3 | bucket on `max(event_week, recorded_week)` | **met** (validator: 100.0% on disagreeing cells) |
| §4 | exact conservation; ledger identity; transfers exist | **met** (exact on both seeds; ledger holds) |
| §5 | 80–93% idle; ≥95% contiguous; 0 dup PKs | **met** (86.3%, 100%, 0) |
| §6 | `delivered = min(K, ordered)`; observability emergent | **met** (20.9% / 23.1% constrained) |
| §7 | fill masses at 0 and 1.0 | **met** (89.0% / 1.9%) |
| §8 | survival process; real censoring; skew ≥0.5 | **partially met** (skew +3.03, censoring emergent at 42.96%; tail 1.47× — see §9.1) |
| §9 | shortage from stock balance; two populations; ~100% traced | **met on 1001** (3.02% conditions, events in band, 97.6% traced); **not on 1002** |
| §10 | feedback arcs incl. the two bold rows | **met** (arc −0.195 / −0.240 direct) |
| §11 | two disruption classes at correct nodes | **met** |
| §12 | demand trend/seasonality; plans versioned | **met** (ratio 1.396) |
| §13 | mechanism params set, outcomes measured | **met, and strengthened** — `escalate_frac` and `linestop_frac` removed; literals audit clean |
| §14 | labels forward; 1:1 binding; censoring 1–60% | **met** (all five bindings ok; censoring 2.98–42.96%) |
| §15 | prohibitions | **all 15 pass** |
| §16 | acceptance ladder | **G0, G1 pass**; G2–G7 not yet run |
| §17 | two seeds + variance band | **met** (§10) |
| §18 | what ships | **met** |
| §19 | final test | **met in structure** |

### §15 prohibitions

| Prohibition | |
|---|---|
| Never generate a label independently of the features | **pass** (probe 1 non-zero ×3, both seeds) |
| Never use a counter / ID / seed index / sin·cos as a target | **pass** |
| Never draw `entity_type`/`entity_id` from the wrong pool | **pass** |
| Never set capacity observability; let `min(K, ordered)` produce it | **pass** |
| Never tune staleness, seasonality, fill mass or an event rate after observing it | **pass** — and `escalate_frac`, the last remaining rate knob, is now deleted |
| Never use a constant recording lag; never a negative one | **pass** |
| Never bucket a derived store on event week while gating on `recorded_ts` | **pass** |
| Never patch a derived store after generation to force conservation | **pass** |
| Never generate shortage events independently of inventory | **pass** — strengthened: events now descend from inventory *episodes* |
| Never generate production actual independently of plan | **pass** |
| Never generate inventory snapshots independently of transactions | **pass** |
| Never generate GRNs independently of shipments and POs | **pass** |
| Never change allocation without changing the receiving supplier's load | **pass** |
| Never delete censored observations | **pass** |
| Never write a gate that cannot fail | **pass** — and run 6 caught one of mine: the first line-stop test required every channel of a part-plant to be empty simultaneously, and produced 0 events. It was rewritten, not widened. |

---

# 13. Readiness ladder and gate readiness per task

| Gate | Asks | Status | Note |
|---|---|---|---|
| **G0** | Is this a dataset? | **PASS** both seeds | conservation exact, censoring 2.98–42.96% on all five tasks, lag `sd/\|mean\|` passes on every table |
| **G1** | Does the graph exist? | **PASS** | median channel degree 3, one component, 100% coverage, 0 isolated nodes |
| G2 | What is the number to beat? | not yet run | naive baselines still uncomputed — the next step |
| G3 | Is there signal at all? | not yet run | blocked on G2 |
| G4 | Does the graph contribute? | not yet run | the arc is present at −0.195/−0.240; h⁰ vs h⁴ still has to be measured |
| G5 | Is the as-of gate binding? | not yet run | runnable — `recorded_ts` present and non-degenerate everywhere |
| G6 | Are the folds sound? | not yet run | label window opens strictly after the snapshot |
| G7 | Will it survive production? | **partially** | cold-start slice 4.59%; seed-variance band published, but it widened on event rates |

## Gate readiness per task

| Task | Trainable? | Blocker |
|---|---|---|
| **Delivery risk** | **Yes** | `arrival_week` on `po_line`, forward-computed, censored at 42.96% from genuinely open POs. Late 18.8%, skew +3.03, probe 1 −0.151. Expedites now shorten real transit, so the label reflects a mitigation the model can learn. Fit on 2019–2025, where the censored tail is 1.20×/1.19× rather than 1.47×. |
| **Fill rate** | **Yes** | `fill_rate` on `po_line`; masses 89.0% at 1.0 and 1.9% at 0, 11.0% below 1.0 — all in band on **both** seeds for the first time (seed 1002's run-5 failure resolved). Probe 1 +0.163/+0.188, probe 2 −0.308, probe 4 gap +0.095. |
| **Part shortage** | **On seed 1001 only** | Much improved: conditions descend from the stock balance at 3.02% (in band, first time), events derive from episode length, depth and consequence rather than a fraction, 97.6% trace to a real feeding channel, and the 2.4% residual is evidenced as demand-driven. **But seed 1002 gives 5.91% conditions and 592.5 events/yr.** The blocker is no longer the escalation rule — it is the seed sensitivity of the capacity/demand balance described in §9.2. A shortage head trained on seed 1001 would not transfer to seed 1002's base rate. |

Two of three tasks are trainable and reproducible across seeds. The third is trainable on one seed
and not the other, and the mechanism that would fix it is named in §9.2.
