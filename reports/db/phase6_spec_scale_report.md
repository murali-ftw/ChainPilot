# Phase 6 — Spec-Scale Generation, Re-derived Sample Rates, Measured Power

**Status:** the benchmark now exists at its spec configuration (`SUP_N = 4,000`, 40 snapshots).
Label sampling is wired into emission, both rates are re-derived against the real generator with
Mechanisms E, F and G live, and the power targets are **measured** per variant rather than
projected. Closes `docs/PHASE2_PHASE6_IMPLEMENTATION.md` §7 TODOs 1–3 and answers TODO 4.

Three findings dominate and none is cosmetic:

1. **A rate derived from one seed is not derived.** `SHORTAGE_SAMPLE_RATE = 0.07` looks correct on
   seed 42 and puts Variant K at **1,891 — under the floor — on seed 46**. Delay positives vary
   ~2× between seeds because the hidden-factor member sets are re-drawn per seed over a power-law
   degree distribution. Every rate here is derived against all 12 variants × 5 seeds, and the
   shortage rate moves to **0.08** as a result (§2.3).
2. **No single `DELAY_SAMPLE_RATE` puts all twelve variants in 2,000–5,000.** Raw delay density
   spans **9.0×** across variants (Variant J 32,953 positives, Variant K 3,642, both at seed 42).
   The rate that respects the 5,000 ceiling everywhere is ≤ 0.15; the rate that clears the 2,000
   floor everywhere is ≥ 0.55. The two constraints are incompatible by 3.6×. §2.4 resolves this in
   favour of the floor — `DELAY_SAMPLE_RATE = 1.00` — and says why a per-variant rate table, the
   obvious alternative, is the wrong fix.
3. **Impact misses the 2,000 floor on the combined benchmark and cannot be sampled up.** Variant K
   averages **1,896** over five seeds, in band on only 3 of 5; Variant F averages 2,256, in band on
   4 of 5. Phase 0 projected 2,049 for K. Impact has no sampling lever, so this is a configuration
   decision — §5 measures what would close it.

Every number below is output from the real `db/generate_dataset.py`. Nothing is extrapolated.

---

## 1. Wiring the sample rates into label emission

`DELAY_SAMPLE_RATE` and `SHORTAGE_SAMPLE_RATE` existed in `Config` and `RANGES` since Phase 1 but
were never read by the label loop. They are now, on the terms the spec's configuration table sets.

**Entities are sampled, not label rows.** A sampled shipment (delay) or `(product, warehouse)` pair
(shortage) carries a label at **every** t0 it is eligible for; an unsampled one carries none at any
t0. Dropping individual rows instead would hand a model histories with holes punched through them,
which no real label store looks like. Two validation checks enforce it: emitted entities are a
subset of the sampled set, and every sampled shortage pair carries exactly `len(T0S)` rows.

**The draw is a hash, not an RNG stream.** `sample_u(task, *key)` is
`blake2b(SAMPLE_SALT | task | entity_key)` reduced to `[0, 1)`, and membership is `u < rate`. It
reads no configuration and consumes no `random` state, so it cannot drift when a mechanism changes
how much RNG it burns. Entity identities are `uuid5` over stable keys — `uid("shp", idx)`,
`uid("prod", i)`, `uid("wh", i)` — so an entity that exists in two variants, or at two seeds, lands
on the same side of the threshold in both. The spec's *"sampled sets held identical across all
variants and all seeds"* is therefore satisfied **by construction**, not by remembering to reseed;
a validation check re-derives membership from entity identity alone and compares.

A second property falls out of the hash-threshold form and matters in §2.4: sampled sets at
different rates are **nested**, `S(0.10) ⊂ S(0.55)`.

**Impact is not subsampled.** Per `docs/phase0_power_check.md` §4 it is the task that reaches the
target by scaling *up*. Its ground truth is also read off **every** eligible shipment, sampled or
not — the delay sample must not silently thin the impact task with it. A check asserts impact emits
exactly `visible_suppliers × snapshots` rows.

**Deriving rates without a run per candidate.** Because sampling is a pure filter on emission and
changes nothing upstream in the simulation, one un-sampled run yields the exact post-sampling count
at *every* rate simultaneously. `HADES_RATE_CURVE=1` makes the generator emit that curve — rows and
positives on a 0.01–1.00 grid — into the run manifest. Every rate table in §2 comes from twelve such
runs (one per variant, seed 42), not from a sweep of candidate rates.

```bash
HADES_RATE_CURVE=1 python3 db/run_benchmark.py --stats-only \
    --variants 0,A,B,C,D,E,F,G,H,I,J,K --seeds 42 \
    --config unsampled.json --manifest-dir raw/     # unsampled.json: both rates at 1.0
```

## 2. Re-deriving both rates at spec scale, with real E/F/G

### 2.1 The Phase 0 §4 figures no longer hold

Phase 0 measured Variant K under the `ABSORB=0.533` **approximation**, before Mechanisms E and F
were built and before G existed. All three are now live (`resilience_lambda = 1.3`). Measured, at
`SUP_N=4,000 × 40`, Variant K, seed 42, un-sampled:

| Task | Phase 0 §4 (approximated) | Measured now (real E/F/G) | Change |
|---|---|---|---|
| delay | 6,266 | **3,642** | −42% |
| shortage | 46,966 | **35,169** | −25% |
| impact | 2,049 | **1,518** | −26% |

**They differ meaningfully, and the real numbers are used from here on.** The direction is
consistent — the `ABSORB=0.533` scalar understated how much real absorption plus real reporting
delay suppress positives — but the size is not a rounding difference. On delay it inverts the
conclusion: Phase 0 had Variant K 25% *over* the ceiling and prescribed subsampling to 0.50;
Variant K is in fact comfortably *inside* the band with no subsampling at all.

Where the delay suppression comes from is worth recording, because it is not what Phase 0 assumed.
Variant K's delay **denominator is larger** than Variant 0's (205,651 vs 166,344 eligible
shipment-snapshots), not smaller: Mechanism G's reporting lag leaves shipments in a `scheduled` /
`in_transit` as-of state past the point where they have actually completed, so *more* rows qualify.
The positive **rate** is what collapses — 1.77% against Variant 0's 18.86%. G dilutes the
denominator and E/F absorb the numerator, and the two compound.

### 2.2 Raw per-variant positives, spec scale, un-sampled (seed 42)

| var | delay n | delay pos | rate | shortage n | shortage pos | rate | impact n | impact pos | rate |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 166,344 | 31,370 | 18.86% | 640,440 | 49,459 | 7.72% | 160,000 | 3,410 | 2.13% |
| A | 166,549 | 32,953 | 19.79% | 640,440 | 49,910 | 7.79% | 132,920 | 4,400 | 3.31% |
| B | 166,163 | 31,638 | 19.04% | 640,440 | 49,556 | 7.74% | 160,000 | 3,541 | 2.21% |
| C | 166,424 | 31,731 | 19.07% | 640,440 | 49,321 | 7.70% | 160,000 | 3,472 | 2.17% |
| D | 166,328 | 31,809 | 19.12% | 640,440 | 49,813 | 7.78% | 160,000 | 3,617 | 2.26% |
| E | 162,570 | 23,405 | 14.40% | 640,440 | 35,159 | 5.49% | 160,000 | 2,038 | 1.27% |
| F | 162,412 | 19,077 | 11.75% | 640,440 | 35,598 | 5.56% | 160,000 | 1,867 | 1.17% |
| G | 208,804 | 10,957 | 5.25% | 640,440 | 49,459 | 7.72% | 160,000 | 2,336 | 1.46% |
| H | 165,949 | 31,458 | 18.96% | 640,440 | 49,544 | 7.74% | 160,000 | 3,498 | 2.19% |
| I | 165,206 | 23,265 | 14.08% | 640,440 | 49,286 | 7.70% | 160,000 | 3,564 | 2.23% |
| J | 166,549 | 32,953 | 19.79% | 640,440 | 49,910 | 7.79% | 160,000 | 4,400 | 2.75% |
| K | 205,651 | 3,642 | 1.77% | 640,440 | 35,169 | 5.49% | 132,920 | 1,518 | 1.14% |

### 2.3 Shortage — a single global rate works, but it is 0.08, not 0.07

Shortage raw density spans only 1.42× across variants (49,910 down to 35,159), so unlike delay a
single global rate does exist. Deriving it from **seed 42 alone** reproduces the old default: the
feasible window is [0.057, 0.100] and 0.07 sits near its log-midpoint, with all twelve variants
between 2,570 and 3,695 positives. On that evidence 0.07 survives.

**It does not survive five seeds.** Generating all 12 × 5 at 0.07 and measuring:

| | worst observed | best observed |
|---|---|---|
| lowest shortage positives | **Variant K, seed 46: 1,891** — *below the 2,000 floor* | |
| second lowest | Variant F, seed 46: **2,000** — exactly on it | |
| highest | | Variant H, seed 42: 3,695 |

So 0.07 puts Variant K out of band on 1 of 5 seeds and Variant F on the boundary of another. The
seed-42-only derivation missed it because seed 42 happens to be the *densest* seed of the five on
this task — the same power-law member-draw effect described in §4.

Re-deriving against the 60-run envelope instead of one seed: the floor constraint is
`0.07 × 2,000 / 1,891 = 0.0740` and the ceiling constraint is `0.07 × 5,000 / 3,695 = 0.0947`, so
the feasible window is **[0.074, 0.095]** and its log-midpoint is 0.084. **`SHORTAGE_SAMPLE_RATE`
is set to 0.08**, the balanced choice at two decimal places. Regenerating all 60 runs at 0.08
confirms it: every one of the 60 readings lands in band, the tightest at **2,239** (Variant K, seed
46 — 12% above the floor) and the loosest at **4,142** (Variant B, seed 42 — 17% below the ceiling).
Full table in §4.

### 2.4 Delay — no single rate works, and the fix is not a per-variant table

Each variant's feasible rate window (the rates at which its delay positives land in 2,000–5,000),
from the seed-42 rate curves. Five-seed data only widens the spread — §4 shows delay positives
varying ~2× across seeds — so no window below is optimistic in a way that would rescue a global
rate:

| var | raw positives | min rate | max rate |
|---|---|---|---|
| 0 | 31,370 | 0.07 | 0.15 |
| A | 32,953 | 0.07 | 0.15 |
| B | 31,638 | 0.07 | 0.15 |
| C | 31,731 | 0.07 | 0.15 |
| D | 31,809 | 0.07 | 0.15 |
| E | 23,405 | 0.09 | 0.21 |
| F | 19,077 | 0.11 | 0.25 |
| G | 10,957 | 0.19 | 0.45 |
| H | 31,458 | 0.07 | 0.16 |
| I | 23,265 | 0.09 | 0.21 |
| J | 32,953 | 0.07 | 0.15 |
| K | 3,642 | **0.55** | 1.00 |

**The intersection is empty.** The ceiling constraint binds at `5,000 / 32,953 = 0.152` (Variants A
and J); the floor constraint binds at `2,000 / 3,642 = 0.549` (Variant K). They are incompatible by
3.6×. The best a single rate can do is **10 of 12**: any rate in `[0.105, 0.152]` puts every variant
except G and K in band. Eleven is not reachable — admitting G needs ≥ 0.183, past the ceiling.

Phase 0's framing said Variant K is the worst case for suppression, and it is — but for *sampling*
that inverts which end binds, and the inversion is easy to get backwards. K's suppression makes it
the variant that sets the **lower** bound on the rate, while the un-suppressed variants set the
**upper** bound. The intuitive worry is that a rate tuned to K would push low-density variants under
the floor; what actually happens is the reverse — a rate tuned to the ceiling pushes K **73%** under
it.

**Resolution — `DELAY_SAMPLE_RATE = 1.00`.** The 2,000 floor and the 5,000 ceiling are not the same
kind of constraint. The spec's own words are *"enough positive examples to distinguish genuine
architectural improvements from random seed variation"* — that is a **floor**, and violating it
makes a variant's delay result uninterpretable. The ceiling is a cost bound, and at spec scale the
cost it is bounding is negligible — §3.2 measures it: `training_labels.csv.gz` is ~4% of a
424 MB variant-seed and delay rows are ~44% of that, so cutting the rate to 0.15 would save
**~6 MB per variant-seed, 1.5%**. Given a choice between violating a floor that carries statistical
meaning and a ceiling that carries 6 MB, the floor wins. At 1.00 every variant clears 2,000 on
delay; eleven of twelve exceed 5,000, the largest at 6.6× (Variant J, 32,953 at seed 42).

A second argument points the same way: subsampling at generation time is **irreversible for anyone
who only has the CSVs**. Emitting the full delay label set lets a consumer impose any power target
they like downstream; emitting 15% of it does not.

**Why not a per-variant rate table.** A table targeting ~3,300 positives each (0 → 0.11, A/B/C/D/H/J
→ 0.10, E/I → 0.14, F → 0.17, G → 0.30, K → 0.90) does put all twelve inside the band, and because
the sampler is a hash threshold the resulting sets are cleanly *nested* rather than arbitrary. It
is still the wrong fix, for a reason that is easy to miss: **it does not actually buy power for the
comparisons the benchmark exists to make.** Every headline claim in the spec is a cross-variant
delta — K against D, A against J, D against B, F against E — and a paired comparison across two
different entity sets has to run on their intersection, which under nesting *is the lower-rate
variant's entire set*. Comparing K (0.90) against D (0.10) on the common entities gives K ~370 delay
positives, worse than any global rate. The per-variant table converts an in-range number in a report
table into an out-of-range number in the analysis that matters, and it does so while breaking the
spec's identical-sets requirement. Recorded here so it is not proposed again as an improvement.

**What remains open.** With `DELAY_SAMPLE_RATE = 1.00` the parameter is inert at the spec default —
it is wired, validated and exercised (shortage at 0.08 is live), but delay emits everything. If the
5,000 ceiling is later judged to be a hard requirement rather than a cost bound, the only levers
that close the 9× spread are structural: reduce it at the source (Mechanism G's eligibility rule
is what makes Variant K's positive *rate* 10× lower than Variant 0's), or accept per-variant rates
together with an evaluation protocol that computes deltas on the intersection and reports the
intersection's size. Both are spec decisions, not generator decisions.

## 3. The spec-scale sweep, storage and timing

### 3.1 Seed retention policy — 2 of 5, seeds 42 and 43

Implemented before the sweep ran, so the sweep never generated what it would then delete. `db/csv/`
retains the **two lowest seeds, 42 and 43**, of each variant; seeds 44, 45 and 46 are regenerated
on demand by `db/regenerate_seed.py`. Seed 42 is the generator's default and the seed the V1
byte-identity anchor is pinned to; 43 is the next in `run_benchmark.py`'s canonical
`DEFAULT_SEEDS`. The convention itself is arbitrary — what matters is that it is written down, in
`db/README.md`, so "the retained seeds" means one thing to everybody.

```bash
python3 db/regenerate_seed.py --list                 # what is on disk vs regenerable
python3 db/regenerate_seed.py --variant K --seed 44  # ~4 min, byte-exact
python3 db/regenerate_seed.py --all                  # all 36 non-retained variant-seeds
```

### 3.2 What the sweep cost

12 variants × 2 retained seeds = **24 runs, zero validation failures**, four worker processes in
parallel on 14 cores / 24 GB. The 36 non-retained variant-seeds were then run `--stats-only` (no
CSVs) for §4's power table, which is why that table covers five seeds while only two are on disk.

| | measured | Phase 0 §7 projection |
|---|---|---|
| wall clock, 24 runs, 4-way parallel | **22 min 56 s** | — |
| CPU time, summed over runs | 77.0 min (mean **192 s**/run) | ~115 s/run |
| per variant-seed, cheapest → dearest | 149 s (Variant C, seed 43) → 468 s (Variant K, seed 42) | — |
| peak RSS, single run | 4.4 GB | ~3.9 GB |
| on disk, per variant-seed | 420–433 MB, mean **424 MB** | ~443 MB |
| **on disk, 24 retained runs** | **10.17 GB** (9.7 GiB) | ~10.6 GB |
| compression | 1.299 GB → 423 MB, **3.07×** (`v0_seed42`) | ~3.18× |
| projected full 12 × 5 | ~25.4 GB | ~27 GB |

(Sizes are decimal GB summed from file sizes, as `regenerate_seed.py --list` reports them; `du -sh`
shows the same corpus as 9.7 GiB.)

**It lands under the ~15 GB target with ~4.8 GB to spare**, so no further lever is needed. Phase 0's
storage projection was 7% high and its timing estimate 1.7× low — the timing gap is real work, not
contention: Variant K costs 3× Variant 0 because Mechanisms E, F and G and their validation checks
(resilience recoverability with a bootstrap CI, permutation nulls) did not exist when the ~115 s
figure was taken.

`inventory_history` remains 54% of the compressed corpus (233 MB of 427 MB in `vK_seed42`).
Phase 0 §7's lever 2 — exporting it only for sampled `inv_pairs` — would roughly halve the corpus
again and is now *more* attractive than it was, since `SHORTAGE_SAMPLE_RATE = 0.08` is live. It is
still a spec decision (it removes context features for unsampled pairs), so it stays untaken.

`training_labels` is 16.9–18.6 MB per variant-seed, about **4%** of the corpus, and delay rows are
~44% of it. That is the measurement behind §2.4's claim that the 5,000 ceiling is cheap: dropping
`DELAY_SAMPLE_RATE` from 1.00 to 0.15 would save roughly **6 MB per variant-seed, 1.5%** of it.

## 4. Measured power targets, per variant, per task

Real spec scale, real sampling active (`DELAY_SAMPLE_RATE = 1.00`, `SHORTAGE_SAMPLE_RATE = 0.08`),
all twelve variants × all five seeds — 60 runs. `docs/PHASE2_PHASE6_IMPLEMENTATION.md` §7 said this
number had never been confirmed at real scale for any variant. It is now, and it does not all pass.

Flags are on the **mean over five seeds**. The last column counts how many individual seeds land
inside 2,000–5,000, which is the more honest reading given the spread.

| var | mechanisms | delay (mean) | range over seeds | | shortage (mean) | range | | impact (mean) | range | | seeds in band d/s/i |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — | 20,864 | 15,072–31,370 | **HIGH** | 3,613 | 2,964–4,086 | **ok** | 4,174 | 3,410–4,800 | **ok** | 0/5/5 |
| A | J,A | 23,604 | 18,103–32,953 | **HIGH** | 3,654 | 3,004–4,092 | **ok** | 5,412 | 4,400–6,177 | **HIGH** | 0/5/2 |
| B | B | 21,314 | 15,638–31,638 | **HIGH** | 3,627 | 3,040–4,142 | **ok** | 4,249 | 3,541–4,736 | **ok** | 0/5/5 |
| C | C | 20,858 | 14,777–31,731 | **HIGH** | 3,591 | 2,913–4,080 | **ok** | 4,187 | 3,472–4,736 | **ok** | 0/5/5 |
| D | B,D | 21,376 | 15,786–31,809 | **HIGH** | 3,621 | 2,937–4,135 | **ok** | 4,271 | 3,617–4,799 | **ok** | 0/5/5 |
| E | E | 16,492 | 12,003–23,405 | **HIGH** | 2,733 | 2,365–3,115 | **ok** | 2,414 | 2,038–2,663 | **ok** | 0/5/5 |
| F | E,F | 14,161 | 9,730–19,077 | **HIGH** | 2,730 | 2,311–3,034 | **ok** | 2,256 | 1,867–2,479 | **ok** | 0/5/4 |
| G | G | 6,941 | 4,969–10,957 | **HIGH** | 3,613 | 2,964–4,086 | **ok** | 2,797 | 2,336–3,245 | **ok** | 1/5/5 |
| H | H | 20,937 | 15,110–31,458 | **HIGH** | 3,615 | 3,001–4,113 | **ok** | 4,161 | 3,498–4,685 | **ok** | 0/5/5 |
| I | I | 18,762 | 14,561–23,265 | **HIGH** | 3,612 | 2,937–4,009 | **ok** | 4,158 | 3,564–4,668 | **ok** | 0/5/5 |
| J | J | 23,604 | 18,103–32,953 | **HIGH** | 3,654 | 3,004–4,092 | **ok** | 5,412 | 4,400–6,177 | **HIGH** | 0/5/2 |
| K | all ten | 3,858 | 3,315–4,432 | **ok** | 2,679 | 2,239–2,998 | **ok** | **1,896** | 1,518–2,207 | **LOW** | 5/5/3 |

**Reading it, task by task.**

- **Shortage passes outright.** All twelve variants, all five seeds, every one of the 60 readings
  inside the band. The tightest reading is Variant K seed 46 at **2,239** (12% above the floor) and
  the loosest is Variant B seed 42 at **4,142** (17% below the ceiling). This is the payoff from
  deriving the rate against the 60-run envelope instead of one seed (§2.3): at 0.07 the same
  envelope produced a 1,891.
- **Delay clears the floor everywhere and the ceiling almost nowhere**, by construction (§2.4).
  Variant K is the one variant in band on the mean and on all five seeds — the combined benchmark
  is the *only* one that needed no help. Variant G is in band on 1 seed of 5. The other ten sit
  6,900–23,600 above the floor and over the ceiling; the counts are in the table so nobody has to
  take "HIGH" on trust.
- **Impact is out of band at both ends.** Variant K averages **1,896 — under the floor**, in band on
  3 seeds of 5. Variant F is in band on 4 of 5 (seed 42 gives 1,867). Variants A and J average
  5,412 and are in band on 2 of 5, over the *ceiling* on the other three. §5 is about this.

**Seed spread is the other headline.** Delay positives have a coefficient of variation of **20–33%**
across seeds and swing about 2× (Variant 0: 15,072 at seed 46, 31,370 at seed 42); impact runs
10–14%; shortage is the steadiest at ~10%. The cause is structural, not noise in the label
definition: the hidden-factor member sets (`H_PORT`, `H_TRUCK`, `H_CUSTOMS`) are re-drawn per seed
over a **power-law supplier degree distribution**, so how much shipment volume the scripted event
calendar actually hits varies a lot from seed to seed. Seed 42 happens to be the densest of the five
on delay and shortage and the *sparsest* on impact — a seed where disruption concentrates in a few
high-volume suppliers produces many delayed shipments spread over few suppliers.

This is the label-volume analogue of what `docs/phase0_power_check.md` §2–§3 concluded for the
gating checks, and it carries the same rule: **no power claim about a variant should be made from a
single seed.**

**Validation: 59 of 60 runs clean.** The one failure is Variant D, seed 44 —
`B: Type C groups do NOT co-degrade (decoy is inert)`, mean delta **+0.0262** against a permutation
null band of **[−0.0191, +0.0231]**, so 13% outside the edge of the band. That check accepts a null
hypothesis, so its false-alarm rate is the band's tail mass by construction; 1 excursion in 60 runs
is *below* what a 95% band predicts (~3), and the effect size is marginal. It is not evidence of a
defect in Mechanism B, and it is recorded rather than suppressed because a reader counting `[FAIL]`
lines across a sweep deserves to know which one and why. Seed 44 is a regenerated seed, so **all 24
variant-seeds actually on disk validate clean.**

## 5. Impact: the combined benchmark is under the floor, and there is no sampling lever

Impact is not subsampled — by design, since it is the task that reaches the range by scaling up —
so the counts in §4 are the world's, not a rate's. Two things are wrong with them.

**Variant K averages 1,896 across five seeds, 5% under the 2,000 floor, and is in band on only 3 of
5** — per seed 42–46: **1,518 / 2,207 / 1,683 / 2,064 / 2,007**. Variant F averages 2,256 (1,867 /
2,479 / 2,030 / 2,479 / 2,423), in band on 4 of 5. Phase 0 projected 2,049 for Variant K; the
measured mean is 8% below that projection and the worst seed is **26% below**. Variant K is the
combined benchmark — the spec calls it out by name in §4 as the variant that must clear this target
— so this is the finding that matters most in this document.

**And impact overshoots at the other end on Variants A and J**, which average 5,412 (4,400 / 6,177 /
4,867 / 5,701 / 5,914) and exceed 5,000 on 3 of 5 seeds. Mechanism J's upstream chains raise
supplier stress enough to lift the impact rate from 2.13% (Variant 0) to 2.75%. Nothing is broken by
this — the ceiling is a cost bound (§2.4) — but it means impact is out of band at *both* ends of the
variant set, in opposite directions, which no single configuration change fixes.

(A and J report **identical** counts on all three tasks, at every seed. That is correct, not a bug:
`A = J + A`, and Mechanism A only truncates *emission*. The suppliers it hides are deep-tier nodes
that ship nothing directly, so every impact row it removes was a guaranteed negative — which is why
A's impact denominator drops from 160,000 to 132,920 while its positives do not move at all. The
A-vs-J contrast lives in the observable graph, not in the label counts.)

### What would close the Variant K gap

Impact positives are one label per visible supplier per snapshot, so the only levers are
`SNAPSHOTS`, `SUP_N`, and the label definition itself.

- **`SNAPSHOTS = 53`** (from 40). Measured, not assumed: Variant K's later snapshots — the ones past
  the V1 event calendar, which ends Sep 2025 — carry **36.4 impact positives each against 40.5 for
  the in-calendar snapshots, i.e. 90%** (measured on `vK_seed42`), so the linear extrapolation
  holds: `1,896 × 53/40 = 2,512`, and even the worst seed reaches `1,518 × 53/40 = 2,011`.
  Phase 0 §4 warned that months past the event calendar would yield ~55% of a calibrated month;
  measured at spec scale it is 85–90% (Variant 0: 80.0 vs 94.1), because Mechanism H's shocks and
  the stochastic stress process keep generating events after the scripted calendar runs out. The
  warning was right in direction and roughly twice too pessimistic in size. Costs: the timeline
  extends to Nov 2028, the corpus grows ~30% (~12.9 GB retained), and A/J move further over the
  ceiling. 53 also clears Variant F (2,256 → 2,989).
- **`SUP_N ≈ 5,300`.** Rejected: `docs/phase0_power_check.md` §1 measured the co-degradation check
  losing measurability above 4,000 (only 1–2 of H_POLYMER's 4 members qualify at 5,000), and 4,000
  was chosen precisely because it is the largest scale where that check still has n=4.
- **Widen the impact label definition.** Currently "any of this supplier's in-flight shipments went
  late in the horizon". A weaker predicate would raise the rate, but it changes what the task
  *means* and would invalidate comparison against V1 and against every earlier phase's numbers.

**Recommendation: `SNAPSHOTS = 53`, as a spec decision, together with extending `EVENTS`** — Phase 0
§4's rule that extending the timeline requires extending the event calendar in the same commit still
applies, and the 90% figure above is the measured cost of *not* doing it. Not taken in this session:
it changes the spec's headline configuration, and this session was scoped to dataset work.

**Until it is taken, Variant K's impact results are under-powered and must be reported as such.**
That is a stronger caveat than the Variant K interpretation rule already in the spec (which is about
reading K against D on hidden-structure claims); this one says the impact task on K does not have
the positive count the spec asks for, on any reading.

## 6. Determinism through the whole pipeline

Sampling adds a new source of run-to-run variation, and the seed-retention policy makes determinism
load-bearing rather than merely nice — the three non-retained seeds *are* the regeneration
guarantee. Three checks, all diffing **compressed** bytes:

```
determinism: variant K seed 42, 21/21 .csv.gz byte-identical to the on-disk copy (450s, 0 failures)
determinism: variant 0 seed 43, 20/20 .csv.gz byte-identical to the on-disk copy (140s, 0 failures)
```

1. **Regenerate-and-diff at spec scale, against what is actually on disk.** `regenerate_seed.py
   --verify` re-runs the generator into a temp directory and compares every `.csv.gz` byte-for-byte
   with the retained copy. Variant K seed 42 (all ten mechanisms, 21 tables including
   `supplier_upstream`) and Variant 0 seed 43 both match completely. This is the stronger form of
   the check than the earlier sessions' two-fresh-runs comparison, because it validates the files
   a consumer would actually load, not just two throwaway runs against each other.
2. **The V1 byte-identity anchor survives everything in this session.** `--config v1 --variant 0`
   was generated before any change was made and again after all of them — sampling wired in, both
   rates re-derived, the manifest extended — and the outputs are byte-identical, and identical to
   the `v1`-preset fixture that was in `db/csv/` at the start. The `v1` preset pins both rates to
   1.0, so the new sampling code short-circuits and Variant 0 is untouched. That anchor is what
   every cross-variant comparison is measured from, so it is checked, not assumed.
3. **Sampling itself is checked for seed- and variant-invariance inside the generator.** Membership
   is re-derived from entity identity alone on ~500 probe entities per run and compared against the
   sampled set (§1), on every one of the 60 runs.
4. **The identical-sets requirement is confirmed from the emitted CSVs, not just from the code.**
   Reading the shortage entities back out of `training_labels.csv.gz` and intersecting with the
   `(product, warehouse)` pairs the two datasets share:

   | pair | shared pairs | sampled∩shared identical? |
   |---|---|---|
   | `v0_seed42` vs `vK_seed42` (across variants) | 16,011 | **yes**, 1,277 each |
   | `vF_seed43` vs `vK_seed43` (across variants) | 16,057 | **yes**, 1,364 each |
   | `v0_seed42` vs `vJ_seed43` (across seeds *and* variants) | 5,060 | **yes**, 429 each |

   Different seeds generate different `inv_pairs` populations, so the comparison is over the pairs
   they have in common — and on those, membership agrees exactly. That is the property the spec's
   configuration table asks for, verified against the artifact a consumer would actually load.

## 7. Reproducing

```bash
# 1. raw per-variant rate curves (one un-sampled run per variant gives the exact
#    post-sampling count at every rate on a 0.01-1.00 grid)
echo '{"delay_sample_rate": 1.0, "shortage_sample_rate": 1.0}' > unsampled.json
HADES_RATE_CURVE=1 python3 db/run_benchmark.py --stats-only --seeds 42 \
    --config unsampled.json --manifest-dir raw/

# 2. the retained sweep, as it is on disk: 12 variants x seeds 42,43
python3 db/run_benchmark.py --seeds 42,43 --manifest-dir manifests/ --report report.json

# 3. the other three seeds, counts only
python3 db/run_benchmark.py --stats-only --seeds 44,45,46 --manifest-dir manifests/

# 4. determinism, against the copy already on disk
python3 db/regenerate_seed.py --verify --variant K --seed 42
```

Steps 2 and 3 were run as four parallel worker processes over three variants each; the wall-clock
figures in §3.2 are for that arrangement on 14 cores / 24 GB.

## 8. What this session did not do

- **Did not change `SNAPSHOTS`.** §5 recommends 53 and shows the measurement behind it, but that is
  the spec's headline configuration and a session scoped to dataset work should not move it.
- **Did not take Phase 0 §7 lever 2** (`inventory_history` for sampled pairs only). Still a spec
  decision; storage came in under target without it.
- **Did not touch the `v1` preset.** Both sample rates stay pinned to 1.0 there, which is what keeps
  Variant 0 byte-identical to V1; re-verified during this session (§6). The old `v1`-preset contents
  of `db/csv/` are parked at `db/csv_v1preset_superseded/` (2.2 GB, gitignored) rather than deleted;
  they are regenerable in ~10 s per variant-seed and can be removed at will.
- **Did not proceed to training or the ML harness.**
