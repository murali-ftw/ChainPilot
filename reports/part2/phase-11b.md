# Phase 11B — settle the premise, re-band the evidence, fix the metric, start Phase 9

**Audience:** whoever decides what ships off v8, and whoever finishes Phase 9.
**Measured on:** `db/gen_v8/seed_1001` unless a dataset seed is named; fixed split — train ≤ 2023,
validation 2024, test 2025. Selection on validation only.
**Companions:** `reports/part2/phase-11a.md`, `reports/part2/phase-0-1-v8.md`,
`reports/part2/v8-clearance.md`.

---

## 1. Verdict per stage

### Stage A — the dataset-seed control: **VERDICT (i) GENERALISES**

On **both** `capacity_strain` and `arrival_week`, across **all five dataset seeds** with the model
seed fixed at 7, the real graph beats the degree-preserving shuffle — **10 of 10 world-task
combinations, every paired difference strictly positive.** On arrival the shuffled arm is *worse
than h⁰* on 3 of 5 worlds. **v8's G4 is therefore not reproduced on this pipeline on either
axis** — 11A varied model seeds, this varies the dataset seeds G4 itself used, and the answer is
the same. §2.

### Stage B — re-banding: **the at-risk list is delivered; the re-banding is DEFERRED**

26 cells carry a margin under 2× their 3-seed spread — above the brief's ~20 threshold. On an
explicit budget decision the re-banding was deferred in favour of Stage A in full. **Phase 8's
headline counts therefore remain on 3-seed bands and are NOT restated here**; §3 gives the
at-risk list so the next phase can run it directly. This is a deliberate omission, not an
oversight, and it is open item 1.

### Stage C — the lateness metric: **FIXED, and the old one was HIDING the head's advantage**

Two t0-available references built and asserted as-of. Under both, the naive constant no longer
scores identically to the reference-only arm — the defect is gone. **And the result reverses
11A's finding:** under the privileged promise reference the head was *not distinguishable* from a
naive constant; under an as-of reference it beats the naive constant, LightGBM **and** h0,
disjointly at five model seeds. **Reference (a), the as-of channel median lead, is named as the
lateness metric going forward.** §4.

A second finding, recorded because it corrects an explanation I gave in 11A: the
promise/naive identity at 0.75181 was **not** the metric's algebra. `phase7_score.py` substitutes
a constant for the promise arm before computing lateness. **Deviation 72.**

### Stage D — Phase 9.1: **built, gated, and the cost estimate CORRECTED**

The roll-forward runs, drawing from the shipped heads, with 9.1's gate passing the correct replay
and firing on five constructed failures including one unit wrong in one row of ~2 M. **Full-grid
compute measured at ~9.7 minutes**, which corrects 11A §6.4's "hours, not minutes" — I had
assumed inference scales with the part-plant grid; it scales with snapshots. Full-grid Monte
Carlo not run; Phase 9.2 copula not started. §5.

---

## 2. Stage A — the dataset-seed control

**Why this stage exists.** 11A varied MODEL seeds with dataset seed 1001 fixed and reached
verdict (i) — the edges carry information. v8's own G4 varied DATASET seeds 1001–1005 and reached
the opposite. **A dataset-seed effect could not have appeared in 11A at all.** This runs the three
arms on the axis G4 varied.

### 2.1 The four other dataset seeds as addressable worlds

`v8` remains dataset seed 1001, so every Phase 0–11A number keeps its meaning. Seeds 1002–1005
are added as `v8s1002`…`v8s1005`. **A dataset seed is a different world and its bands are its
own**, exactly as v6, v7 and v8 are never borrowed from each other.

**Panel width measured per dataset seed, not assumed:**

| dataset seed | `revision_count` non-zero | other three candidates | panel `d` | `d_in` |
|---|---|---|---|---|
| 1001 | 3.326% | all constant zero | 15 | 25 |
| 1002 | 2.385% | all constant zero | 15 | 25 |
| 1003 | 2.152% | all constant zero | 15 | 25 |
| 1004 | 2.407% | all constant zero | 15 | 25 |
| 1005 | 2.300% | all constant zero | 15 | 25 |

Each is declared individually in `config.EXPECTED_PANEL_D` so the width assertion still fires on a
mismatch rather than defaulting. All four caches build clean at T = 535, panel complete.

### 2.2 A.1 / A.4 — capacity across all five dataset seeds

Three arms per world, **model seed fixed at 7**, validation pinball (the selection surface, lower
is better):

| dataset seed | real | shuffled | h0 | real < shuffled | shuffled < h0 | edge share |
|---|---|---|---|---|---|---|
| **v8 (1001)** | **0.07035** | 0.07400 | 0.07475 | ✓ | ✓ | 83.1% |
| **v8s1002** | **0.05946** | 0.06254 | 0.06546 | ✓ | ✓ | 51.4% |
| **v8s1003** | **0.05912** | 0.06225 | 0.06460 | ✓ | ✓ | 57.1% |
| **v8s1004** | **0.06123** | 0.06349 | 0.06693 | ✓ | ✓ | 39.6% |
| **v8s1005** | **0.05616** | 0.06032 | 0.06458 | ✓ | ✓ | 49.4% |

**Real beats shuffled on 5 of 5. Shuffled beats h0 on 5 of 5.**

Each dataset seed is its own world, so the comparison that matters is the **paired difference
within a world**, never a band pooled across them:

```
shuffled - real, per world:  [0.00365, 0.00309, 0.00313, 0.00226, 0.00416]
all strictly positive: True    min 0.00226    mean 0.00326
```

The absolute levels vary widely between worlds — 0.056 to 0.070, a 25% spread — which is exactly
why bands are never borrowed. **The ordering does not vary at all.** The edge share of the
h⁴-over-h⁰ gap runs **39.6% to 83.1%, mean 56.2%**: substantial on every world, and its size is
world-dependent in a way the direction is not.

### 2.2b A.1 — arrival across all five dataset seeds

Three arms per world, **model seed fixed at 7**, validation C-index (higher is better):

| dataset seed | real | shuffled | h⁰ | real > shuffled | shuffled vs h⁰ | edge share |
|---|---|---|---|---|---|---|
| **v8 (1001)** | **0.67114** | 0.65507 | 0.65344 | ✓ | shuf better | 90.8% |
| **v8s1002** | **0.66855** | 0.65303 | 0.65378 | ✓ | **shuf WORSE** | 105.1% |
| **v8s1003** | **0.67270** | 0.65757 | 0.65791 | ✓ | **shuf WORSE** | 102.3% |
| **v8s1004** | **0.67284** | 0.65577 | 0.65810 | ✓ | **shuf WORSE** | 115.8% |
| **v8s1005** | **0.66451** | 0.65253 | 0.65073 | ✓ | shuf better | 86.9% |

**Real beats shuffled on 5 of 5.** Paired differences within world:

```
real - shuffled, per world:  [0.01607, 0.01552, 0.01513, 0.01707, 0.01199]
all strictly positive: True    min 0.01199    mean 0.01516
```

**Edge share mean 100.2%, range 86.9%–115.8%**, and on **3 of 5 worlds the shuffled arm is WORSE
than h⁰** — a randomised neighbourhood costs more than having none at all. That reproduces on the
dataset-seed axis what 11A found on seed 1001's test fold: forcing the encoder to aggregate over
misleading neighbours is actively harmful, not merely uninformative.

Arrival's effect is larger and less variable than capacity's: the edge share sits near 100% on
every world (86.9–115.8%) where capacity's ranged 39.6–83.1%.

### 2.3 A.2 — the control re-falsified on every dataset seed

Soundness on seed 1001 does not imply soundness on 1005, so the shuffle was falsified per world:

| world | supplier | part | plant | total | degree distributions |
|---|---|---|---|---|---|
| v8 (1001) | 99.764% | 99.795% | 86.007% | **95.188%** | identical |
| v8s1002 | 99.776% | 99.876% | 85.509% | **95.053%** | identical |
| v8s1003 | 99.733% | 99.851% | 85.378% | **94.987%** | identical |
| v8s1004 | 99.770% | 99.801% | 85.932% | **95.168%** | identical |
| v8s1005 | 99.708% | 99.764% | 85.335% | **94.935%** | identical |

(percentage of edges whose endpoint changed; plant's ~85.5% is chance for 7 plants, 100 − 14.29)

**All three degenerate mutations — identity shuffle, 90% preserved, all-to-one — fire on all five
dataset seeds.** The control is sound on every world it was used on, not merely on the first.

### 2.4 A.4 — the verdict, in the brief's own terms

**Form (i): real beats shuffled outside the bands ACROSS DATASET SEEDS, so verdict (i)
generalises.** It is not a property of dataset seed 1001.

**On BOTH tasks, on all five dataset seeds, without exception:**

| task | real beats shuffled | shuffled worse than h⁰ | edge share |
|---|---|---|---|
| capacity_strain | **5 / 5** | 0 / 5 | 39.6–83.1%, mean 56.2% |
| arrival_week | **5 / 5** | **3 / 5** | 86.9–115.8%, mean 100.2% |

Ten world-task combinations, ten in the same direction, every paired difference strictly
positive.

**v8's own G4 is therefore not reproduced by this pipeline on either axis.** 11A varied model
seeds with the dataset fixed; this stage varies the dataset seeds G4 itself varied, with the
model seed fixed. Both return the opposite of G4.

**Deviation 64's caveat narrows accordingly.** In 11A I wrote that the disagreement could not be
called a refutation because, among other differences, *"G4 varied dataset seeds where this varies
model seeds, so a dataset-seed effect would not show here."* **That specific escape is now
closed** — the dataset-seed axis has been run and it agrees. What remains of the caveat is the
instrument difference alone: `validator_v8.py`'s own model rather than this repo's quantile head,
MAE rather than pinball, and a different split. Those are real and I am not claiming to have
eliminated them; I am recording that the seed-axis explanation is gone.

### 2.5 A.3 / A.5 — the shuffled arm widened to five model seeds, and the corrected edge share

11A quoted its edge shares against a **3-seed** shuffled band while the real arms were already at
five (11A open item 2). Dataset seed 1001's shuffled capacity arm is now at five model seeds:

| arm | seeds | mean | band |
|---|---|---|---|
| real | 5 | 0.06981 | [0.06879, 0.07035] |
| **shuffled** | **5** | 0.07367 | **[0.07306, 0.07420]** (was [0.07306, 0.07400] at 3) |
| h⁰ | 3 | 0.07506 | [0.07475, 0.07544] |

**real vs shuffled remains DISJOINT at 5 against 5**, and shuffled vs h⁰ remains disjoint.

**The corrected decomposition of the 0.00525 h⁴-over-h⁰ gap:**

| source | 11A (3-seed shuffled) | **corrected (5-seed shuffled)** |
|---|---|---|
| **the EDGES** | 71.7% | **73.6%** |
| depth and parameters | 28.3% | **26.4%** |

The correction is **+1.9 points** and changes no verdict. 11A's figure was quoted against a
narrower band than it should have been; the direction and magnitude survive.

**Arrival, same treatment:**

| arm | seeds | mean | band |
|---|---|---|---|
| real | 5 | 0.67067 | [0.66904, 0.67193] |
| **shuffled** | **5** | 0.65319 | **[0.65075, 0.65507]** |
| h⁰ | 3 | 0.65364 | [0.65313, 0.65435] |

real vs shuffled remains **DISJOINT**. **shuffled vs h⁰ is NOT disjoint** — the bands overlap.

| source | 11A (3-seed shuffled) | **corrected (5-seed shuffled)** |
|---|---|---|
| **the EDGES** | 99.2% | **102.6%** |
| depth and parameters | 0.8% | **−2.6%, not distinguishable from zero** |

The edge share exceeds 100% because the shuffled mean now sits marginally *below* h⁰'s. **That is
not a claim that depth hurts**: the two bands overlap, so the depth-and-parameters contribution is
**not distinguishable from zero** and is reported as such. What the five-seed measurement
sharpens is the positive claim — on arrival, essentially the entire h⁴-over-h⁰ advantage is
attributable to the edges, with nothing measurable left for depth.

**h⁰ is still at three seeds** on both tasks. On capacity its band [0.07475, 0.07544] sits clear
of the shuffled maximum 0.07420, so that comparison is not in doubt. On arrival it overlaps the
shuffled band, which is precisely why the depth share there is reported as indistinguishable
rather than as a number. Widening h⁰ is open item 2.

### 2.6 A defect in the queue, caught on the first world

The first run of `ml/train/phase11b_stageA.sh` named every arm's log
`{tag}_{task}_{world}_s{seed}.log` — **identical for the real, shuffled and h0 arms of a world**.
The first arm's `.done` marker then skipped the other two, and **the control silently lost two of
its three arms.**

This is the artifact-identity failure standing rule 2 exists for, one layer up: not two bundles
colliding on a path, but two *queue entries* colliding on a marker. It was caught on the first
world with one bundle built (~13 minutes lost), the log name now carries arch, depth and the
shuffle flag, and on restart the already-built arm skipped correctly via its own bundle. **No
result was contaminated** — the collision suppressed work rather than mixing it. Recorded as
**deviation 73**.

---

## 3. Stage B — the at-risk list, and what deferring it costs

### 3.1 The criterion and the list

A comparison is **at risk** when `margin < 2 × spread`, where the margin is the gap between arm
means and the spread is the larger of the two arms' 3-seed spreads. Computed from recorded
artifacts only; **nothing was trained for this stage**.

```
comparisons examined : 152   (capacity 32, arrival 24, fill 96)
AT RISK              :  42
distinct cells       :  26   (above the brief's ~20 threshold)
```

**The fifteen tightest:**

| task | world | origin | comparison | margin | spread | m/s | verdict @3 |
|---|---|---|---|---|---|---|---|
| fill | v6 | 8 | vs lgbm22 ECE-22 | 0.00002 | 0.00144 | **0.01** | not dist. |
| capacity | v6 | 7 | h⁴ vs B5 | 0.00058 | 0.00377 | **0.15** | not dist. |
| arrival | v7 | 2 | head vs h0 lateness | 0.00112 | 0.00577 | **0.19** | not dist. |
| capacity | v7 | 3 | h⁴ vs h⁰ | 0.00089 | 0.00328 | 0.27 | not dist. |
| fill | v6 | 7 | vs B2 ECE-22 | 0.00057 | 0.00207 | 0.28 | not dist. |
| capacity | v7 | 3 | h⁴ vs B5 | 0.00120 | 0.00328 | 0.37 | not dist. |
| capacity | v6 | 3 | h⁴ vs h⁰ | 0.00081 | 0.00199 | 0.41 | not dist. |
| fill | v7 | 2 | vs B2 ECE-22 | 0.00157 | 0.00368 | 0.43 | B |
| fill | v6 | 1 | vs B2 ECE-22 | 0.00160 | 0.00360 | 0.44 | not dist. |
| fill | v6 | 5 | vs lgbm22 ECE-22 | 0.00051 | 0.00111 | 0.46 | not dist. |
| fill | v7 | 8 | vs b5flat22 CRPS | 0.00026 | 0.00053 | 0.48 | B |
| fill | v6 | 5 | vs b5flat22 ECE-22 | 0.00055 | 0.00111 | 0.49 | not dist. |
| capacity | v6 | 4 | h⁴ vs h⁰ | 0.00119 | 0.00210 | 0.57 | A |
| fill | v6 | 2 | vs B2 ECE-22 | 0.00135 | 0.00231 | 0.58 | A |
| capacity | v6 | 7 | h⁴ vs h⁰ | 0.00251 | 0.00377 | 0.67 | B |

**Arrival needed its own handling** and is the most robust of the three: `head_roc` is a 3-seed
band while `promise_roc` is a scalar with a bootstrap CI, so the margin is taken over the head's
spread alone — which is what Phase 8 itself recorded as `head_minus_promise_over_head_spread`.
**Only one arrival lateness cell is at risk** (v7 o1, m/s 1.23); three more sit just above the
threshold at 2.22–2.41. The other 22 arrival comparisons clear it comfortably, up to m/s 22.74.

### 3.2 What deferring this costs, stated plainly

The re-banding was **deferred by an explicit budget decision** in favour of running Stage A in
full. The consequence, without softening:

- **Phase 8's headline counts are NOT restated.** Capacity's "h⁴ wins 10 of 16", fill's
  "LightGBM better in 12 of 16" and arrival's "8 of 8" all still rest on 3-seed disjointness.
- **Deviation 67 says that is not safe.** Going 3 → 5 seeds in 11A widened six metrics by more
  than 1.5× and flipped two verdicts — both the narrowest margins in their table. Eleven of the
  cells above are tighter than either of those two.
- **The ship table in Phase 8 §5 is therefore not re-validated here**, and no proposal to change
  it is made. `shipped.json` is untouched.

**Open item 1.** The list is machine-readable and the work is ~10 cells × 2 seeds.

---

## 4. Stage C — a lateness metric available at t0

### 4.1 The defect, and a second one found while fixing it

The shipped metric is `arrival_week > promise_week`, scored on `prediction − promise_week`.
`promise_week` comes from a `po_line` raised ~7 weeks **after** the forecast instant (B9 = 0.00%
on every v8 dataset seed), so **every arm including the head is ranked against information no
planner holds at t0.**

**The second defect (deviation 72), which corrects an explanation in 11A §3.1.** I wrote there
that `promise_only` and `naive_global_median` both scored 0.75181 because "ranking
`prediction − promise` with a constant prediction *is* ranking by the promise". That is true of
the naive arm. It is **not** why the two agree. `ml/eval/phase7_score.py`'s `arrival_promise`
branch does not score the promise arm's own prediction for lateness — it substitutes a constant:

```python
late = M.arrival_scores(np.full(len(Y), float(spec["const"])), Y, EV, AUX)
out["roc_auc_late"] = late["roc_auc_late"]
```

So the two entries are **literally the same computation**. Scored honestly, `promise_only`'s
lateness score is identically zero (one distinct value across 36,000 rows) and its ROC-AUC is
**0.5**. 11A's conclusion stands; the mechanism I gave for it was wrong. 11A is a protected
report, so the correction lives here.

### 4.2 The two references, and the as-of assertion

| reference | construction | distinct values | late rate | t0-available |
|---|---|---|---|---|
| **(a) as-of channel lead** | channel's median observed lead (receipt − order, weeks) over GRN lines **recorded ≤ t0**, plus a global offset fitted on TRAINING rows only | **208** | 47.5% | **yes** |
| **(b) contracted lead** | `sourcing_channels.contracted_lead_time_days` / 7, same construction | 55 | 48.9% | **yes** |
| promise *(privileged)* | `original_promise_date − snapshot` | 137 | 20.1% | **no** |

**Why an offset term.** `arrival_week` is measured from t0 and contains two parts: the wait until
the order is raised, and the lead once it is. Only the second varies by channel; the first is a
property of the label's sampling window and is common to every channel. So the channel-varying
part of the reference is the historical lead, and the constant part is fitted **once, on training
rows, never refitted** (+4.571 weeks for (a), +3.286 for (b)). It shifts every reference equally,
so it sets the label's balance and cannot affect the ranking within a reference.

**The as-of assertion fires on all three failure modes:**

| input | result |
|---|---|
| correctly filtered set (322,357 rows) | PASSES |
| unfiltered whole history | **FIRES** — 707,489 receipts recorded after t0 |
| filtered set + **one** future-dated row | **FIRES** — 1 receipt |
| **empty set** | **FIRES** — "would silently fall back to the global constant" |

The emptiness guard was added **after a falsification passed vacuously**: shifting every
`recorded_ts` into the future leaves nothing ≤ t0, and the check then passed on an empty set while
the reference collapsed to the cold-start fallback for every channel. A check that passes hardest
when it has nothing to look at is the defect this project keeps finding.

### 4.3 The results — five model seeds, test fold, ROC-AUC

| arm | seeds | **(a) as-of channel lead** | (b) contracted lead | promise *(privileged)* |
|---|---|---|---|---|
| **head h⁴** | 5 | **0.70905** [0.70640, 0.71300] | **0.77151** [0.76940, 0.77390] | 0.75306 [0.74990, 0.75626] |
| head h⁰ | 3 | 0.69999 [0.69840, 0.70130] | 0.76436 [0.76360, 0.76500] | 0.72993 |
| b5flat LightGBM | 5 | 0.70535 [0.70480, 0.70570] | 0.76578 [0.76530, 0.76610] | 0.74594 |
| naive constant | 1 | 0.68697 | 0.75281 | 0.75181 |
| **reference-only** | 1 | **0.50000** | **0.50000** | **0.50000** |

### 4.4 C.3 — the decisive check PASSES

Under the promise definition the naive constant and the reference arm were indistinguishable
because the scorer made them the same computation. **Under both t0-available references they are
not**: the naive constant scores 0.68697 / 0.75281 while the reference-only arm scores exactly
0.50000, which is what a constant lateness score must give. **The defect is removed and neither
new reference inherits it.**

### 4.5 C.4 — the verdict, and it reverses 11A

**Reference (a), the as-of channel median lead, is named as the lateness metric going forward.**
It is dynamic, business-meaningful ("will this channel run slower than it has been running"),
carries 208 distinct values against (b)'s 55, is balanced at 47.5% late, and is computable from
receipts recorded on or before t0 with the assertion armed.

**Under it, the head's lateness advantage is real and separable — all disjoint at five seeds:**

| comparison | head h⁴ | other | disjoint |
|---|---|---|---|
| vs **naive constant** | min 0.70640 | 0.68697 | **yes** |
| vs **b5flat LightGBM** | min 0.70640 | max 0.70570 | **yes** (margin 0.0007) |
| vs **h⁰** | min 0.70640 | max 0.70130 | **yes** |

**The privileged metric was concealing this, not inflating it.** The promise date is so
informative that ranking by −promise alone reaches 0.75181, leaving almost no headroom — which is
why 11A found the head not distinguishable from a naive constant (deviation 65). Measured against
a reference a planner actually holds at t0, the head's channel-level signal separates.

**Deviation 65 is therefore narrowed, not withdrawn:** the head's lateness advantage over *the
promise date* remains unestablished on v8, and that comparison stays retired. Its advantage over
a *t0-available* reference is established, disjointly, at five seeds. These are different claims
and only the second is deployable.

**C.5 compliance:** this is a measurement-definition change only. No head was retrained against a
new target; the label was recomputed and existing predictions rescored.

---

## 5. Stage D — Phase 9.1, built to a defined stop

### 5.1 What is built

`ml/sim/montecarlo.py`: the roll-forward `I_w = I_{w-1} + A_w − C_w`, shortfall against **safety
stock** (not zero), all N paths retained so percentiles are taken at the top and never summed from
below. Exercised at **N = 50 on a 200 part-plant subset** — the machinery and its cost, not the
full grid.

**Draws come from the shipped heads:** arrival week sampled from the hazard head's recalibrated
13-cell distribution, fill fraction from the CDF head's 22 cells, both read as-of at the last
snapshot ≤ t0 with that bound asserted. **Consumption comes from `part_demand_weekly`'s forward
requirement as of t0, not from a head** — no demand head exists in the shipped set and inventing
one would fabricate an input.

### 5.2 D.3 — mapping and BOM, coverage asserted

| | |
|---|---|
| part-plants | 4,340 |
| with ≥ 1 channel | **4,227 (97.40%)** |
| **with no channel** | **113** |
| channels per part-plant | median 4, max 12 |
| BOM | **single-level asserted** (`bom_level` constant 1), 180 products, 618 parts |

The 113 orphans **are not dropped**. They cannot receive anything, so they are counted, reported
and carried with zero arrivals, which means their projected shortage is still produced rather than
silently missing.

### 5.3 D.2 — 9.1's gate, passing and firing

| constructed simulation | mismatched rows | result |
|---|---|---|
| correct replay | 0 | **PASS** |
| **one unit wrong on one row** | **1** | **FIRES** |
| scrap netted into receipts (0.1%) | 1,971,853 | **FIRES** |
| opening balance omitted | 1,971,853 | **FIRES** |
| safety stock used in place of the level | 1,978,220 | **FIRES** |

### 5.4 D.4 — the full-grid cost, measured, and 11A's estimate corrected

| component | measured | over 83 snapshots |
|---|---|---|
| roll-forward compute | 0.27 s / snapshot | **0.4 min** |
| **head inference** | **6.72 s / snapshot** | **9.3 min** |
| **total full-grid compute** | | **≈ 9.7 min** |

**This corrects `reports/part2/phase-11a.md` §6.4 (deviation 74).** I estimated there that head
inference would run to "hours, not minutes". It does not. The head pass runs **once per snapshot
over all 16,072 channels** and does not scale with the part-plant grid at all — only the
per-part-plant sampling does, and that is negligible. My estimate assumed the wrong scaling axis.

**The conclusion 11A drew from that estimate still holds, for a different reason:** compute was
never the blocker. It is not hours, it is ten minutes — so the remaining cost is implementation
and validation, not machine time.

### 5.5 D.5 / D.6 — the stop, and the limit carried forward

**Full-grid Monte Carlo NOT run. Phase 9.2's copula NOT started.**

G10's limit is carried in the module docstring so it cannot be lost: **G10 tests the correlation
LEVEL only.** Within-group r = 0.2101 against cross-group 0.2055 — a separation of 0.0046, an
order of magnitude inside its own ±0.05 tolerance — so **the copula's group structure is
untestable on v8 and must never be claimed as validated.**

---

## 6. Deviations and open items

### 6.1 Deviations, continuing the index

| # | Prior statement | Measured | Where |
|---|---|---|---|
| **72** | 11A §3.1: `promise_only` and `naive_global_median` both score 0.75181 because ranking `prediction − promise` with a constant prediction *is* ranking by the promise | **the mechanism is wrong.** `ml/eval/phase7_score.py`'s `arrival_promise` branch SUBSTITUTES a constant for the promise arm's own prediction before computing lateness, so the two entries are **literally the same computation**. Scored honestly, `promise_only`'s lateness score is identically zero across 36,000 rows and its ROC-AUC is **0.5**. 11A's conclusion stands; its explanation does not | §4.1 |
| **73** | a queue script cannot produce an artifact-identity collision | **mine did.** `phase11b_stageA.sh` named every arm's log `{tag}_{task}_{world}_s{seed}.log`, identical for the real, shuffled and h⁰ arms, so the first arm's `.done` marker silently skipped the other two and the control lost two of its three arms. Standing rule 2 one layer up — not two bundles on one path, but two queue entries on one marker. Caught on the first world, ~13 min lost, **no result contaminated** (the collision suppressed work rather than mixing it) | §2.6 |
| **74** | 11A §6.4: Phase 9.1's head inference will run to "hours, not minutes" | **9.3 minutes.** The head pass runs once per snapshot over all 16,072 channels and does **not** scale with the part-plant grid; only the per-part-plant sampling does, and that is negligible. My estimate assumed the wrong scaling axis. Full-grid compute measured at **~9.7 min** total | §5.4 |
| **75** | 11A deviation 64: the G4 disagreement cannot be called a refutation partly because "G4 varied dataset seeds where this varies model seeds, so a dataset-seed effect would not show here" | **that escape is closed.** The dataset-seed axis has now been run — 5 seeds × 3 arms × 2 tasks — and it agrees with the model-seed axis: **10 of 10 world-task combinations, real beats shuffled, every paired difference strictly positive.** What remains of the caveat is the instrument difference alone (validator_v8's model, MAE vs pinball, a different split) | §2.4 |
| **76** | 11A's edge shares (71.7% capacity, 99.2% arrival) | **quoted against a 3-seed shuffled band while the real arms were at five.** Corrected against a 5-seed shuffled band: capacity **73.6%** (+1.9), arrival **102.6%** (+3.4). real-vs-shuffled disjointness holds at 5 v 5 on both. On arrival the depth share becomes −2.6% and the shuffled/h⁰ bands OVERLAP, so it is reported as **not distinguishable from zero**, not as depth being harmful | §2.5 |
| **77** | 11A deviation 65: the arrival head's lateness advantage is not established on v8 | **narrowed, not withdrawn.** Against the *promise date* it remains unestablished and that comparison stays retired. Against a **t0-available** reference the head beats a naive constant, b5flat LightGBM **and** h⁰, all disjoint at five model seeds. The privileged metric was **concealing** the advantage, not inflating it | §4.5 |
| **78** | a randomised neighbourhood is at worst uninformative | **on arrival it is harmful on 3 of 5 dataset seeds**, where the shuffled arm scores *below* h⁰ — so the edges account for **more than the whole** h⁴-over-h⁰ gap (edge share up to 115.8%) | §2.2b |

### 6.2 Open items

1. **Stage B's re-banding was deferred** and is the largest outstanding item. 26 cells carry a
   margin under 2× their 3-seed spread; the 10 tightest are ~20 training runs. Until they are
   run, **Phase 8's headline counts — capacity 10/16, fill 12/16, arrival 8/8 — rest on 3-seed
   bands that deviation 67 showed to be optimistic by an unpredictable factor.** The list is in
   `ml/eval/phase11b_atrisk.py` and its JSON output.
2. **h⁰ arms are still at three model seeds** while real and shuffled are at five on dataset seed
   1001. The bands do not overlap so no verdict is in doubt, but the asymmetry should be closed
   before the edge shares are quoted to more precision than they are here.
3. **The dataset-seed control used one model seed per world.** That is the right design for
   isolating the dataset axis, but it means each world's number is a single draw. The direction
   is unanimous across ten combinations, which is strong; a per-world band would be stronger.
4. **v8's G4 remains unexplained rather than refuted.** Both seed axes now agree against it, so
   the disagreement is down to the instrument. Running `validator_v8`'s own diagnostic against
   this pipeline's panel — or this pipeline's quantile head against G4's MAE target — would
   isolate which of model, metric or split is responsible.
5. **The new lateness metric needs adopting formally.** §4.5 names reference (a); nothing in
   `shipped.json` or the scorers has been changed to use it, and `phase7_score.py`'s
   `arrival_promise` substitution (deviation 72) is still in place for anyone who reruns it.
6. **Phase 9.1 is built but not run.** The full grid is ~9.7 min of compute; what remains is
   validating the draws against held-out outcomes and wiring 9.2's copula — whose group
   structure is **untestable on v8** (§5.5) and must never be claimed as validated.
7. **`part_demand_weekly` is the consumption stream** in the simulation because no demand head
   exists in the shipped set. If Phase 9 is to carry a demand-uncertainty claim, that head has to
   exist first; the current construction propagates a point forecast with a uniform jitter, which
   is a placeholder and is labelled as one in the module.

---

## 7. Compliance

- **`ml/configs/shipped.json` is unchanged.** Stage A found the graph carries information and
  Stage C found the arrival head has a real lateness advantage under a corrected metric; **neither
  was allowed to edit the config**, as the brief requires.
- **No assertion was disabled or weakened.** The Phase 11 as-of assertion stays armed and no
  line-level feature was added to any head. The shuffle falsification, its per-dataset-seed
  repeat, the as-of receipt assertion (three failure modes), the panel-width assertion and both
  Phase 9 gates were each demonstrated capable of failing.
- **Selection on validation only.** Every Stage A verdict is decided on validation pinball or
  validation C-index; test figures appear beside them and never choose.
- **No learning rate was retuned.** Every cell took its rate from `shipped.json`.
- **Five model seeds is the floor for anything newly quoted** (deviation 67). Stage C's arms are
  at five. Stage A's dataset-seed control is one model seed per world by design — that is the
  axis under test — and is reported as such, not as a five-seed band.
- **`inventory_position_weekly` was read**, and here is where and why: `ml/sim/montecarlo.py`'s
  `opening_position` (the simulation's opening balance) and `run_gate` (9.1's acceptance
  reference), plus §5.3's gate demonstration. It is no longer forbidden on v8 — B1 reconciled it
  at 100.000000% on 2,253,420 rows — and it is read **only** on those sanctioned Phase 9 paths,
  never as a model feature.
- **Protected paths untouched:** `db/gen_v6/**`, `db/gen_v7/**`, `db/gen_v8/**`,
  `db/validator.py`, `docs/specs/**`, `db/dataset_structure.md`, and every pre-existing report
  including `phase-11a.md` — whose §3.1 error is corrected **here** rather than edited there.
- **Full-grid Monte Carlo not run. Phase 9.2 copula not started.**
- **Commit before running**; nothing stamped `+dirty`.
