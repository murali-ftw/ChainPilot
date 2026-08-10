# Phase 0 — Power Check

**Status:** complete. **Recommended configuration: `SUP_N = 4,000`, 40 monthly snapshots**
(Jul 2024 – Oct 2027), with `DELAY_SAMPLE_RATE = 0.50` and `SHORTAGE_SAMPLE_RATE = 0.07`.

Gating checks at that exact configuration: co-degradation **+0.2147** on n=4 with its whole
leave-one-out range above threshold (§1); Type A/B/C indistinguishability **0.521 [0.472, 0.573]**,
CI spanning chance, against a positive control at **0.866** (§2). Resilience recoverability is
**not** cleared — see §3; it is gated on Mechanism H in Phase 2, not on scale.

Phase 1 (config object and variant harness) is unblocked. **Phase 2 is not** until §3 is resolved.

Supersedes `docs/phase0_power_analysis.md` (deleted; this file is its corrected successor).

Every number here is shell output from running the real `db/generate_dataset.py`. Nothing is
extrapolated except the storage projection in §7, which is marked as such.

---

## 0. Provenance, and a correction to the first pass

The first pass reported the co-degradation margin "narrowing from 0.09 to 0.03 as suppliers scaled
800→3,200" and treated that as monotonic erosion.

**Provenance:** those numbers came from actual generated worlds, not analytical estimation — the
real generator, run at each scale, with its own validation suite reporting the margin. The
suppression factors (delay ×0.57, impact ×0.62, shortage ×0.96) likewise came from real generated
label counts over thousands of labels, and **those stand**.

**The correction:** the two margin readings were point estimates of a statistic computed on
**four suppliers**, read at the two-decimal precision the generator prints. Measured exactly across
four scales and two absorption regimes (§1), the margin does not erode monotonically and does not
approach zero. It wanders between +0.11 and +0.30 with no trend. Two draws from a 4-sample
estimator were over-read as a trend.

The finding that replaces it is more consequential, not less — see §1.

Two other first-pass figures are corrected here: `db/csv/` is **124.7 MB**, not 254 MB (the earlier
figure misread `ls`'s 512-byte block total), and this corpus compresses at **3.18×**, not the ~6×
assumed — measured in §7.

## 1. Co-degradation margin at the recommended scale

`SNAPS=36`. Margin = fleet mean 90-day on-time rate at 2024-12-01 minus H_POLYMER member mean;
the check requires > 0.10. "LOO range" is the margin recomputed with each member dropped in turn.

| `SUP_N`                  | `ABSORB=1.0` margin | n           | LOO range        | `ABSORB=0.533` margin | n           | LOO range                  |
| -------------------------- | --------------------- | ----------- | ---------------- | ----------------------- | ----------- | -------------------------- |
| 800                        | +0.1780               | 4           | [+0.157, +0.218] | +0.1393                 | 4           | [+0.111, +0.168]           |
| 3,200                      | +0.1742               | 4           | [+0.144, +0.223] | +0.2671                 | 4           | [+0.140, +0.360]           |
| 4,000                      | **+0.3018**     | **4** | [+0.196, +0.363] | **+0.1949**       | **4** | [+0.157, +0.224]           |
| 5,000                      | +0.2464               | **2** | [+0.185, +0.307] | +0.1136                 | **1** | n/a                        |
| **4,000 @ 40 snaps** | —                    | —          | —               | **+0.2147**       | **4** | **[+0.117, +0.291]** |

**The check passes at every tested scale under both absorption regimes.** It is not monotonic, it
does not reach ≤ 0, and the recommendation is not invalidated by it.

**But the check stops being measurable before it stops passing.** H_POLYMER has a **fixed 4
members at every `SUP_N`** — the selection loop stops at four — while the fleet it is compared
against grows to 1,346. Worse, a member only enters the statistic if it has ≥ 3 deliveries inside
the 90-day window: at `SUP_N=5,000` only **2 of 4** members qualified in the base regime and
**1 of 4** under absorption. A margin computed on one supplier is not evidence of anything.

The leave-one-out ranges show the same fragility at scales where n=4 holds: at `SUP_N=3,200` under
absorption, dropping a single member moves the margin from +0.140 to +0.360.

**Consequences:**

1. **`SUP_N = 4,000` is the largest tested scale at which this check remains measurable
   (n=4) in both regimes, with the whole leave-one-out range above the 0.10 threshold** — including
   at the recommended 40-snapshot timeline (last row: +0.2147, LOO [+0.117, +0.291]). That is the
   deciding reason it is recommended over 5,000, which is otherwise equivalent on power.
2. **The V1 co-degradation check must not be carried into V2 as a gate.** Phase 3 should replace it
   with a Mechanism-B-wide check: at Hidden Parent Rate 0.20 and `SUP_N=4,000` that is ~800 members
   across ~180 groups instead of 4, and the replacement must report a confidence interval rather
   than a bare threshold comparison.
3. The first pass's §3 conclusion — that absorption suppresses the co-degradation signal — **still
   holds** and is visible in the table (absorbed margins are lower at 3 of 4 scales, and the
   absorbed regime is where the member count collapses fastest). What does **not** hold is that
   scale drives it. §5's Variant K interpretation rule rests on the suppression, not on the trend,
   and is unaffected.

## 2. Clarification 1 — Type A/B/C topology indistinguishability

Mechanism B does not exist yet, so this is a **prototype**: hidden-parent groups are drawn over the
real generated supplier population at Hidden Parent Rate 0.20 with a 1:1:1 type mix, by one
identical process, with type assigned by seeded shuffle — independent of every structural property.
Ten group-level structural features (size, member degree mean/std/min/max, shipment volume,
dual-source count, country spread, sea fraction) feed one-vs-rest logistic regressions; the score is
**5-fold held-out** AUC with a percentile bootstrap 95% CI.

`ABSORB=1.0`. The **positive control** repeats everything with type assigned *by group size*, so
structure genuinely does predict type.

| `SUP_N` | groups | pooled AUC      | 95% CI         | spans 0.50?   | control AUC | control CI     |
| --------- | ------ | --------------- | -------------- | ------------- | ----------- | -------------- |
| 800       | 35     | 0.420           | [0.311, 0.528] | yes           | 0.862       | [0.792, 0.926] |
| 3,200     | 145    | 0.435           | [0.383, 0.489] | **no**  | 0.882       | [0.849, 0.911] |
| 4,000     | 180    | **0.521** | [0.472, 0.574] | **yes** | 0.867       | [0.836, 0.898] |
| 5,000     | 223    | 0.461           | [0.413, 0.507] | yes           | 0.861       | [0.837, 0.890] |

(The `ABSORB=0.533` sweep is within ±0.002 of every row — these features are structural, so
absorption barely touches them.)

**The check does not degrade with scale — it improves.** The CI narrows from 0.217 wide at 800 to
0.094 at 5,000, so larger worlds make the check *more* sensitive to a real leak, the opposite of
§1. The positive control sits at 0.86–0.88 with a CI far above 0.50 at every scale, so the check
demonstrably has the power to detect a leak when one exists.

### Correction — the statistic and the null above are both wrong [amended in Phase 3]

Phase 3 built this check for real and, at 186 groups, it **failed**: macro AUC 0.442, CI
[0.385, 0.495], excluding 0.50 on the low side. Diagnosing that exposed two methodological errors in
the table above. Both are fixed in `db/generate_dataset.py`'s permanent check; the table is left
as-run, with its readings now understood as follows.

**Error 1 — pooled one-vs-rest AUC is not a valid statistic.** The "pooled AUC" column ranks scores
from three *separately fitted* one-vs-rest models in one list. Each carries its own intercept and
scale, so pooling them measures the models' relative calibration as much as their discrimination.
The permanent check reports the **macro** average of the three AUCs instead, never a pooled ranking.

**Error 2 — the null is not 0.50.** Cross-validated AUC on a few hundred rows carries a systematic
*negative* bias: excluding a fold's positives from training shifts the fitted direction against
them. Phase 3 measured the size of it by permuting the type labels and refitting 200 times — the
null band came out **[0.398, 0.570]**, centred near 0.48, not 0.50. Every below-0.50 reading in the
table above sits comfortably inside that band.

So the correct reading of the sweep is that **all four scales pass**, including 3,200. There was no
marginal exclusion to explain; the estimator was simply biased low and the gate was set at the wrong
place. The permanent check compares the observed macro AUC against a **200-permutation null**, which
reproduces the bias exactly and is therefore the right reference distribution.

What survives unchanged: the check does not degrade with scale, and the positive control (type
assigned *by group size*) sits at 0.86–0.88 across every scale, so the check has real power to
detect a leak. What is withdrawn: the claim that a single run's CI is an insufficient gate *because
of chance exclusion* — the exclusion was bias, not chance. Cross-seed evaluation is still worth
doing, but for the ordinary reason, not this one.

A methodological note worth preserving, because it cost a wrong result before it was caught: an
earlier version assigned types by `k % 3` and CV folds by `i % 5`. Those two modulos interact, and
the check reported a systematic **0.39–0.44** AUC on data carrying no signal at all. Both are now
seeded shuffles. That fix was necessary but, as the above shows, not sufficient.

## 3. Clarification 3 — resilience recoverability

Also a prototype, since Mechanism E does not exist yet. Hidden `resilience ∈ [0,1]` per supplier
attenuates the stress→delay conversion; outcomes are re-simulated over the generator's **real**
shipment set, using each shipment's real dispatch time and its supplier's real stress trajectory, so
evidence volume per supplier is whatever the world actually provides. Recovery uses only what a
model could compute from `shipment_status_history`: on-time rate in high-stress windows, in
low-stress windows, the gap, and volume. Scored as 5-fold held-out AUC for "resilience above
median".

| `SUP_N`                  | AUC (base) | 95% CI         | AUC (absorbed)  | 95% CI                   | n estimable | coverage  |
| -------------------------- | ---------- | -------------- | --------------- | ------------------------ | ----------- | --------- |
| 800                        | 0.642      | [0.499, 0.772] | 0.675           | [0.556, 0.798]           | 67–72      | 8.4–9.0% |
| 3,200                      | 0.698      | [0.630, 0.754] | 0.668           | [0.606, 0.737]           | 244–251    | 7.6–7.8% |
| 4,000                      | 0.650      | [0.587, 0.722] | 0.614           | [0.538, 0.680]           | 208–217    | 5.2–5.4% |
| 5,000                      | 0.621      | [0.559, 0.680] | 0.706           | [0.655, 0.760]           | 311–320    | 6.2–6.4% |
| **4,000 @ 40 snaps** | —         | —             | **0.549** | **[0.472, 0.631]** | 232         | 5.8%      |

**Recoverability does not degrade with scale, but it is not reliably established either.** AUC sits
at 0.61–0.71 across most of the sweep with no trend, and the CI excludes 0.50 at every scale ≥ 3,200
— **except at the recommended configuration**, where a single draw gave 0.549 [0.472, 0.631],
spanning chance. At `SUP_N=800` the base-regime lower bound is 0.499, so V1's scale cannot establish
this check at all.

Run-to-run spread of 0.53–0.71 on the same mechanism means **this check must be evaluated across the
five seeds with sign consistency, never from a single draw** — the same conclusion §2 reaches, for
the same reason.

**The real problem is coverage, not accuracy.** Only **5–9% of suppliers** have enough
(disruption → outcome) evidence to be estimable at all — the rest lack ≥5 shipments on both sides of
the stress threshold. So resilience is recoverable *for one supplier in fifteen* and, on present
disruption density, **irreducible noise for the other fourteen**. Clarification 3 states plainly
that if resilience is not recoverable, Mechanism F is irreducible noise and any depth result on it
is uninterpretable — that verdict currently applies to ~94% of the population.

This is the same shape as `v1_findings/v2.md` item 6 (`fulfilment_preference_weight` computable for
only 6.7% of pairs), and it has a plausible fix already in the spec: Mechanism H adds shocks at
12/year against V1's ~10 events across 21 months. **Phase 2 must re-measure this coverage with
Mechanism H enabled, treat coverage as a gating metric targeting > 50% of suppliers, and raise
`Shock Arrival Rate` if H alone does not reach it.** Do not build F on top of a 6%-observable
latent variable.

## 4. Label volume and the recommended scale

Carried forward from the first pass and unchanged — these came from real generated label counts.

- V1 baseline: delay **1,959** / shortage **3,161** / impact **378**. Only shortage clears 2,000.
- Absorption suppression at the mean regime: delay ×0.57, impact ×0.62, shortage ×0.96.
- Impact holds at ~2.1% of supplier-snapshots under mean absorption:
  `impact_positives ≈ 0.021 × SUP_N × SNAPSHOTS`.
- Timeline past the V1 event calendar (ends Sep 2025) yields ~55% of a calibrated month, so
  **extending the timeline requires extending `EVENTS` in the same commit.**

**Measured** Variant K positives (`ABSORB=0.533`, Mechanism A at 60% visible), not projected:

| Task     | `SUP_N=4,000 × 36` | `SUP_N=4,000 × 40` | In 2,000–5,000 range? | Action                                |
| -------- | --------------------- | --------------------- | ---------------------- | ------------------------------------- |
| impact   | 1,943                 | **2,049**       | yes, at 40             | scale**up** — the binding task |
| delay    | 5,607                 | 6,266                 | no — 25% over         | subsample to ~0.50                    |
| shortage | 41,904                | 46,966                | no — 9× over         | subsample to ~0.07                    |

**36 snapshots is not enough**: impact lands at 1,943, 3% short of the floor. 40 snapshots clears it
at 2,049. This is why the recommendation is 40, not 36.

**Impact is the only task that reaches the range by scaling.** Delay and shortage have to be
subsampled *down* while impact is scaled *up* — they sit on much larger denominators. Both
`DELAY_SAMPLE_RATE = 0.50` and `SHORTAGE_SAMPLE_RATE = 0.07` are now in the spec's configuration
table, sampling *entities* rather than label rows so each sampled entity keeps its full time series,
with the sampled sets seeded and held identical across all variants and seeds.

The first pass reported only shortage as overshooting; at the recommended scale **delay overshoots
too**, which was not visible at V1 scale.

Measured cost at `4,000 × 40`: ~**115 s** and ~**3.9 GB** peak RSS per variant-seed.

## 5. Variant dependencies and the Variant K interpretation rule

Both are now written into `docs/00_Benchmark_Specification.md`.

**Three dependency pairs, not two.** `grep -n tier db/generate_dataset.py db/schema.sql` returns
only `customer_priority_tier`, a customer segment; `db/supplier_dyadic_risk_schema.sql:10` records
that V1's supplier tier amendment was "deliberately" not built. V1 therefore has a **single**
supplier tier, and Mechanism A's `Maximum Visible Tier` has nothing to truncate. Locked in as
option (a):

| Variant     | Composition         | Report against      |
| ----------- | ------------------- | ------------------- |
| D           | `B + D`           | Variant B           |
| F           | `E + F`           | Variant E           |
| **A** | **`J + A`** | **Variant J** |

`A = J + A` keeps **Variant 0 identical to V1**, preserving the anchor every cross-variant
comparison uses. Because Variant J is `Base + J`, the pair (J, A) isolates truncation exactly.

**Variant K interpretation rule.** Any "does discovering hidden structure help" result from Variant
K must be read as a delta against **Variant D**, never in isolation. Absorption suppresses the
signal Mechanism B's discoverability depends on (§1), so a null in K may mean the signal was
absorbed before K could observe it — not that coupling does not help.

## 6. `SUP_N` population definition — resolved

**`SUP_N` counts the full simulated supplier population, including tier-3+ suppliers that Mechanism
A hides from the model.** It is not the visible subset. Recorded in the spec's configuration table
with both consequences: label volume scales with the *visible* subset, so power targets are computed
against that; and V1's flat topology makes the two readings numerically identical, so **every
measurement in this document — the §1 margin sweep and the §4 label-volume calculation alike — is on
the same footing.** They diverge only once Mechanism J introduces tiers, at which point they must be
reported separately.

## 7. Gzip output — measured

`write()` now emits `<table>.csv.gz` via `gzip.GzipFile(mtime=0, filename="", compresslevel=9)`.

**Determinism holds at the compressed-byte level.** `mtime=0` pins the header timestamp;
`filename=""` prevents `GzipFile` deriving and embedding the source filename from the underlying
fileobj (the non-obvious trap here); CPython always writes `0xFF` for the OS byte. Verified by
diffing the **compressed** files, not their contents:

```
$ python3 generate_dataset.py && cp -r csv $S/run1
$ python3 generate_dataset.py && diff -r --brief $S/run1 csv
IDENTICAL: all .csv.gz byte-for-byte equal across two runs
$ PYTHONHASHSEED=12345 python3 generate_dataset.py && diff -r --brief $S/run1 csv
IDENTICAL under PYTHONHASHSEED=12345 too
```

**Achieved ratio: 3.18×** — 124.7 MB → 39.2 MB at V1 scale, across all 20 tables.

| file                    | raw MB          | gz MB          | ratio          | % corpus |
| ----------------------- | --------------- | -------------- | -------------- | -------- |
| inventory_history       | 62.0            | 20.3           | 3.05           | 49.7%    |
| shipment_status_history | 19.0            | 5.7            | 3.36           | 15.3%    |
| training_labels         | 13.0            | 2.8            | 4.62           | 10.4%    |
| shipments               | 11.2            | 2.9            | 3.90           | 8.9%     |
| order_items             | 9.6             | 4.0            | 2.42           | 7.7%     |
| **total**         | **124.7** | **39.2** | **3.18** |          |

The ratio is modest because half the corpus is `inventory_history` and every row is UUID-keyed —
random hex does not compress.

`load_data.py` decompresses transparently and still accepts plain `.csv`. Verified end-to-end
against PostgreSQL: all 20 tables COPY'd from `.csv.gz`, all 7 verification queries passed,
committed.

### Projected total — over the ~15 GB target

Extrapolated (the one extrapolation in this document) from the 124.7 MB base by table-class scaling
factors to `SUP_N=4,000 × 40` snapshots:

|         | per variant-seed  | × 12 variants × 5 seeds |
| ------- | ----------------- | ------------------------- |
| raw     | ~1.41 GB          | ~85 GB                    |
| gzipped | **~443 MB** | **~27 GB**          |

**~27 GB, about 12 GB over the ~15 GB target.** gzip alone does not get there, because the corpus
compresses at 3.18× rather than the ~6× the first pass assumed. Next levers, best first:

1. **Retain 2 of 5 seeds on disk, regenerate the other 3 on demand.** 12 × 2 × 443 MB = **~10.6 GB**,
   comfortably under target. Regeneration costs ~115 s per seed and determinism guarantees the
   result is exact, so this is storage-for-compute at a very favourable rate. **Recommended.**
2. **Export `inventory_history` only for sampled `inv_pairs`.** It is half the corpus and the worst
   compressor, and `SHORTAGE_SAMPLE_RATE = 0.07` (§4) already means the model consumes shortage
   labels for only ~7% of pairs. Cuts the corpus ~47% → ~14 GB standalone, ~5.6 GB combined with
   (1). This is a **spec decision, not a mechanical one** — full inventory history may still be
   wanted as context features — so it is flagged, not taken.
3. Higher compression is not a lever: level 9 is already in use, and stdlib `lzma` would reach only
   ~4.5× (≈17 GB) at several times the CPU cost.

## 8. Still open

**Blocking Phase 2 (not Phase 1):**

1. **Resilience coverage** (§3). ~~Must reach > 50% coverage with Mechanism H enabled before Phase 2
   builds Mechanism F on it.~~ **Rechecked in Phase 5 — see `docs/phase2_coverage_recheck.md`.**
   Mechanism H at spec defaults changes coverage not at all (5.4% with H vs 5.6% without), and no
   shock parameterisation clears 50%: coverage is non-monotonic in shock intensity and peaks at
   29.6%. The cause was misattributed here. It is mainly the *binned* ≥5/≥5 estimability rule this
   document invented — a continuous estimator gives 58.4% — plus a structural ceiling of 71.7%
   (28.3% of suppliers ship nothing and can never be estimable). Removing the coverage problem
   exposes the real one: recoverability is only AUC 0.536, barely above chance. **Phase 2 stays
   blocked**, but on signal strength, not coverage.

**Decisions, not blockers:**

2. **`inventory_history` export scope** (§7 lever 2). Storage decision with a modelling consequence:
   halves the corpus, but removes context features for unsampled pairs.
3. **Seed retention policy** (§7 lever 1). Recommended at 2-on-disk; needs sign-off since it trades
   ~115 s of regeneration for ~16 GB.

**Recorded for Phase 3, so they are not rediscovered:**

4. Replace the V1 co-degradation check with a Mechanism-B-wide equivalent reporting a CI (§1).
5. Evaluate both the indistinguishability and resilience checks **across five seeds with sign
   consistency**, never from a single run's CI (§2, §3).
6. **Re-examine §3's resilience numbers with the same lens as §2's correction.** That prototype used
   the identical pooled-OvR-vs-0.50 machinery, so its AUCs carry the same negative bias — meaning
   recoverability is probably *understated* there, and the single sub-chance draw at the recommended
   configuration (0.549 [0.472, 0.631]) may be an artifact rather than a finding. This does not
   change §8.1: coverage of 5–9% is measured by counting estimable suppliers, not by AUC, and that
   is what gates Phase 2.

## 9. Reproducing

```bash
# label-volume power sweep
python3 db/phase0_power_harness.py                                     # V1 baseline
P0_SUP_N=4000 P0_SNAPS=40 P0_ABSORB=0.533 P0_VISRET=0.6 \
  python3 db/phase0_power_harness.py                                   # Variant K worst case

# the three gating checks, at the recommended configuration
P0_SUP_N=4000 P0_SNAPS=40 P0_ABSORB=1.0   python3 db/phase0_checks.py
P0_SUP_N=4000 P0_SNAPS=40 P0_ABSORB=0.533 python3 db/phase0_checks.py
```

Both harnesses patch `generate_dataset.py` by regex and exit if a match count is not 1. They now
stub `write()` with an early `return`, leaving its body as unreachable code, so they track only its
*signature* — the gzip change did not require touching them beyond that. **Phase 1 will still break
them** when the config object lands; at that point delete both and drive the same sweeps through
real config parameters.
