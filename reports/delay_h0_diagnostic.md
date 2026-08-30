# `delay` at h⁰ — is the bottleneck Shipment's own features, or the first neighbour-blend?

**Scope.** `delay` only. `shortage` and `impact` are out of scope here: their depths and status
are settled, and neither is touched by anything below.

**Question.** `reports/new_task_depth.md` compared `delay`'s readout at h¹ (retained), h² and h³
and found no swap that beats h¹. Every depth in that table has already been through at least one
round of message passing, so it cannot separate two very different explanations for why `delay`
underperforms:

1. **Thin raw features** — `Shipment.x` is **7 columns**; there simply isn't much to work with,
   and neighbour aggregation is neither helping nor hurting much because little is reaching it.
2. **Lossy aggregation** — the raw signal is there, and the first blend sums it into an
   undifferentiated 128-slot vector, destroying the distinctions the readout needs.

The control that separates them is **h⁰**: `RGCNAttnDepthGateEncoder.forward()`'s
`{nt: self.lin_in[nt](x)}` — Shipment's own 7 raw columns under a learned 7→128 linear
projection and **nothing else**. No neighbour term, no aggregation, no attention. h⁰ vs h¹
isolates exactly the one step the existing table holds fixed.

**No retraining. SHARE untouched.** Same 25 frozen checkpoints in `out/ds_ckpt/`
(`v0_seed4{2..6}_v0_d4{2..6}_m{0..4}_e100.pt`), same 5 dataset seeds × 5 model-init seeds grid,
variant 0, `db/csv_v1scale/`. SHARE emits the full `{h⁰…h⁴}` for every node in one forward pass;
the only thing that varies is which index of that already-computed list the readout reads.
`assert_backbone_frozen()` runs before *and* after every cell, including the check that no
backbone parameter reached an optimizer.

Code: [ml/layer3_h0_diagnostic.py](ml/layer3_h0_diagnostic.py) · results:
[out/layer3_v3/delay_h0_diagnostic.json](out/layer3_v3/delay_h0_diagnostic.json)

---

## 0. Methodology

### The two arms are `ml/layer3_depth_swap.py`'s, unchanged

| | what is fitted | what the gain measures |
|---|---|---|
| **Arm A** — `frozen_head` | nothing at all | whether h¹'s head *transfers* to another depth's geometry |
| **Arm B** — `refit_head` | the readout head only, on frozen `h^d` | whether `h^d` *carries the signal* |

Arm A applies `model.heads["delay"]` to `layers[d]["Shipment"]` — the literal zero-parameter
index-select. Arm B hands the frozen `h^d` to `ml/latent_state_head.py::train_head`: the same
`in → 32 → 1` MLP, the same positive-weighted BCE, the same 120 epochs, the same seeding by
model-init seed that Phase 0's `P(Y|X)` arm uses. Arm B trains no SHARE parameter — it fits a
~4k-parameter probe on a frozen cached tensor. Because both sides of Arm B's gain then share one
readout protocol, `mean AUC(h^d) − mean AUC(X)` isolates the representation and nothing else.

`train_head` gained one optional keyword (`return_preds=True`) so the saturation diagnostic in
§3 is computed on the very predictions each reported AUC was scored on. The default return is
unchanged and every existing caller is byte-for-byte unaffected.

`P(Y|X)` is not recomputed: it is read from `out/layer3_v3/phase0_baseline.json`'s per-cell grid,
because features-only does not depend on depth.

### Both reproduction checks hold exactly

| check | what it asserts | result |
|---|---|---|
| **1** | Arm A at h¹ reproduces Phase 0's cached `P(Y\|X,H)` | **25/25 cells, max \|diff\| = 0.00e+00** |
| **2** | every h¹/h²/h³ cell, both arms, reproduces `out/layer3_v3/phase0_depth_swap.json` | **150/150 cells, max \|diff\| = 0.00e+00** |

Check 2 is the stronger statement: this harness *is* the previous one extended, not a
re-implementation that happens to agree. h⁰ has no prior value by construction — that is the
entire point of the run — so it is trusted on the strength of the other 150 cells reproducing bit
for bit.

### One sanity property of h⁰ worth stating before the numbers

h⁰ is a **rank-≤7 linear map of exactly the 7 columns `P(Y|X)` reads**. It therefore cannot
contain information `P(Y|X)` doesn't, and Arm B's h⁰ gain should sit at ≈ 0. It does (+0.0100,
see §1) — which is a positive control that h⁰ genuinely carries no neighbour information, and
that the small residue is the probe finding a better-conditioned basis rather than new signal.

---

## 1. Requested table — `delay` at h⁰, h¹, h², h³ side by side

25 cells each. FLOOR = max(init-seed floor, dataset-seed floor). PASS requires
**gain > FLOOR AND 5/5 worlds positive**.

### ARM A — pure index-select, nothing fitted

| depth | P(Y\|X) | P(Y\|X,H) | graph gain | init floor | dset floor | **FLOOR** | gain > FLOOR | worlds + | **PASS** |
|---|---|---|---|---|---|---|---|---|---|
| **h⁰ (NEW)** | 0.7416 | 0.6343 | **−0.1074** | 0.4313 | 0.0785 | **0.4313** | no | 0/5 | **FAIL** |
| *h¹ (retained)* | *0.7416* | *0.7774* | *+0.0357* | *0.0140* | *0.0874* | *0.0874* | *no* | *5/5* | *FAIL* |
| h² | 0.7416 | 0.4886 | −0.2531 | 0.5053 | 0.2103 | 0.5053 | no | 0/5 | FAIL |
| h³ | 0.7416 | 0.5016 | −0.2401 | 0.5827 | 0.2061 | 0.5827 | no | 0/5 | FAIL |

The h¹, h² and h³ rows are `reports/new_task_depth.md`'s numbers, reproduced exactly.

**Arm A's h⁰ row is not evidence about h⁰**, for the same reason the h²/h³ rows were not
evidence about depth: that `PredictionHead` was trained jointly with the encoder against h¹'s
activations, so feeding it any other geometry measures head transfer. Its init-seed floor of
**0.4313** — a 43-point AUC swing from nothing but a different seed — says so directly. The one
readable thing here is comparative: h⁰ transfers *better* (0.6343) than h² or h³ (0.4886,
0.5016), i.e. h¹'s head is closer to h⁰'s geometry than to any deeper one. That is a remark, not
a finding.

### ARM B — readout head refit on the frozen `h^d`

| depth | P(Y\|X) | P(Y\|X,H) | graph gain | init floor | dset floor | **FLOOR** | gain > FLOOR | worlds + | **PASS** |
|---|---|---|---|---|---|---|---|---|---|
| **h⁰ (NEW)** | 0.7416 | 0.7516 | **+0.0100** | 0.0038 | 0.0767 | **0.0767** | no | **5/5** | **FAIL** |
| *h¹ (retained)* | *0.7416* | *0.7802* | *+0.0386* | *0.0115* | *0.0778* | *0.0778* | *no* | *5/5* | *FAIL* |
| h² | 0.7416 | 0.7662 | +0.0245 | 0.0782 | 0.0784 | 0.0784 | no | 5/5 | FAIL |
| h³ | 0.7416 | 0.7293 | −0.0123 | 0.3038 | 0.1722 | 0.3038 | no | 1/5 | FAIL |

**No depth passes**, h⁰ included, and for the same reason as always: every gain is smaller than
its own dataset-seed floor (0.077–0.078 at the shallow depths). The monotone decay from
`new_task_depth.md` extends cleanly in both directions:

```
   h⁰ +0.0100  →  h¹ +0.0386  →  h² +0.0245  →  h³ −0.0123
        (rises with the first blend, then decays with every further one)
```

### The paired h¹ − h⁰ contrast (the actual question)

The unpaired table above compares each depth to `P(Y|X)` and so inherits the full dataset-seed
floor. The question "does the first blend help?" is a **within-cell** comparison — same world,
same checkpoint, same probe seed, different index — and is much more sensitive:

| statistic (Arm B, h¹ − h⁰) | value |
|---|---|
| mean paired delta | **+0.0286** |
| per-world deltas (d42…d46) | +0.0543 · +0.0269 · +0.0123 · +0.0350 · +0.0145 |
| cells positive | **25/25** |
| worlds positive | **5/5** |
| range across cells | +0.0045 … +0.0561 |
| init-seed floor **of the delta** (max per-world spread) | 0.0140 |
| dataset-seed floor **of the delta** | 0.0420 |

So: the direction is unambiguous (every one of 25 cells, effect 2× the init-seed floor of the
delta), but the magnitude is **still below the dataset-seed floor of the delta** (+0.0286 vs
0.0420). The honest statement is *"h¹ is reliably better than h⁰ by a small amount that this
5-world grid cannot certify."*

---

## 2. Per-world detail

`P(Y|X)` per world — 42: 0.7486 · 43: 0.7081 · 44: 0.7791 · 45: 0.7588 · 46: 0.7136.

### ARM A

| row | d42 | d43 | d44 | d45 | d46 | worlds + |
|---|---|---|---|---|---|---|
| h⁰ — P(Y\|X,H) | 0.6760 | 0.6521 | 0.5975 | 0.6260 | 0.6198 | |
| h⁰ — gain | −0.0726 | −0.0561 | −0.1816 | −0.1328 | −0.0938 | **0/5** |
| h¹ — P(Y\|X,H) | 0.8187 | 0.7313 | 0.7963 | 0.8029 | 0.7377 | |
| h¹ — gain | +0.0701 | +0.0232 | +0.0172 | +0.0441 | +0.0241 | **5/5** |
| h² — gain | −0.2558 | −0.3044 | −0.3177 | −0.2878 | −0.0996 | **0/5** |
| h³ — gain | −0.2605 | −0.2529 | −0.2670 | −0.3356 | −0.0843 | **0/5** |

### ARM B

| row | d42 | d43 | d44 | d45 | d46 | worlds + |
|---|---|---|---|---|---|---|
| h⁰ — P(Y\|X,H) | 0.7617 | 0.7130 | 0.7897 | 0.7699 | 0.7237 | |
| h⁰ — gain | +0.0131 | +0.0049 | +0.0106 | +0.0111 | +0.0101 | **5/5** |
| h¹ — P(Y\|X,H) | 0.8160 | 0.7399 | 0.8019 | 0.8049 | 0.7383 | |
| h¹ — gain | +0.0674 | +0.0318 | +0.0229 | +0.0461 | +0.0246 | **5/5** |
| h² — gain | +0.0439 | +0.0069 | +0.0132 | +0.0346 | +0.0240 | **5/5** |
| h³ — gain | +0.0545 | −0.0076 | −0.0211 | −0.0048 | −0.0827 | **1/5** |

### Every Arm B cell, h⁰ / h¹

| world | m0 | m1 | m2 | m3 | m4 |
|---|---|---|---|---|---|
| d42 | 0.7619 / 0.8169 | 0.7616 / 0.8132 | 0.7619 / 0.8178 | 0.7615 / 0.8176 | 0.7618 / 0.8145 |
| d43 | 0.7126 / 0.7463 | 0.7136 / 0.7387 | 0.7151 / 0.7348 | 0.7118 / 0.7408 | 0.7119 / 0.7390 |
| d44 | 0.7912 / 0.7958 | 0.7902 / 0.8050 | 0.7897 / 0.8029 | 0.7875 / 0.8030 | 0.7897 / 0.8030 |
| d45 | 0.7707 / 0.8055 | 0.7700 / 0.8063 | 0.7697 / 0.7994 | 0.7696 / 0.8099 | 0.7696 / 0.8035 |
| d46 | 0.7244 / 0.7437 | 0.7247 / 0.7418 | 0.7238 / 0.7341 | 0.7226 / 0.7384 | 0.7232 / 0.7333 |

h¹ > h⁰ in all 25. Note also how *stable* the h⁰ column is within a world (spread ≤ 0.0037):
a rank-7 linear map leaves the probe almost nothing to disagree about across init seeds, which
is why h⁰'s init-seed floor (0.0038) is the lowest number in this entire report.

---

## 3. Saturation diagnostic (`reports/new_nodes_fix1.md`'s dead-head test)

Unique predicted probability values, prediction std, and the fraction at p ≥ 0.999, on Arm B's
own test predictions. Reported **per dataset seed** rather than pooled, because the earlier
collapse (`d45 m3`, 1 unique value, std 3.0e-08) was isolated to one cell in one seed and a
pooled mean would dilute it to ~4% of its size. `uniq min` / `std min` / `p≥.999 max` are the
worst single init seed in that world, so a one-cell collapse cannot hide behind the mean.

| depth | world | n test | uniq mean | **uniq min** | std mean | **std min** | p≥.999 mean | **p≥.999 max** |
|---|---|---|---|---|---|---|---|---|
| **h⁰** | 42 | 4,085 | 4054.6 | 4,045 | 0.2778 | 0.2768 | 0.00% | 0.00% |
| **h⁰** | 43 | 3,765 | 3739.6 | 3,729 | 0.2537 | 0.2527 | 0.00% | 0.00% |
| **h⁰** | 44 | 4,616 | 4333.8 | 4,264 | 0.2838 | 0.2816 | 0.00% | 0.00% |
| **h⁰** | 45 | 4,373 | 4299.0 | 4,262 | 0.2799 | 0.2775 | 0.00% | 0.00% |
| **h⁰** | 46 | 4,027 | 4022.8 | 4,021 | 0.2529 | 0.2522 | 0.00% | 0.00% |
| **h¹** | 42 | 4,085 | 4080.8 | 4,077 | 0.2951 | 0.2937 | 0.16% | 0.24% |
| **h¹** | 43 | 3,765 | 3746.2 | 3,741 | 0.2578 | 0.2530 | 0.01% | 0.03% |
| **h¹** | 44 | 4,616 | 4587.4 | 4,551 | 0.2796 | 0.2772 | 0.07% | 0.19% |
| **h¹** | 45 | 4,373 | 4370.4 | 4,369 | 0.2747 | 0.2693 | 0.00% | 0.00% |
| **h¹** | 46 | 4,027 | 4022.6 | 4,016 | 0.2592 | 0.2515 | 0.00% | 0.02% |

Pooled across all five worlds: h⁰ **20,450 / 20,866** unique (98.0%), std **0.2696**, p ≥ 0.999
**0.00%**; h¹ **20,807 / 20,866** unique (99.7%), std **0.2733**, p ≥ 0.999 **0.05%**.

Three things to read off this:

- **Neither depth is saturated, and neither has a dead head.** The worst single cell in 50 is
  `d43` h⁰ at 3,729 unique values out of 3,765 rows — 99.0% distinct. The `d45 m3` collapse does
  **not** reproduce here; that cell was on the *enriched* graph arm, and on the original graph
  with a refit probe, `d45 m3` is perfectly healthy — 4,262 unique at h⁰ and 4,370 at h¹, 0.00% saturated at both, AUC 0.7696 / 0.8099.
- **h¹ is not more collapsed than h⁰ — it is marginally *less*.** More unique values (99.7% vs
  98.0%), slightly higher std, and a p ≥ 0.999 fraction that is 0.05% rather than 0.00% — a
  difference of ~10 rows in 20,866, not a saturation event. This is the direct falsification of
  the "summed into an undifferentiated card" hypothesis at the first blend.
- **The saturation that the project has been chasing lives deeper.** For reference, the same
  diagnostic at h³ pools to 96.1% unique and **1.63%** at p ≥ 0.999, with individual cells as bad
  as **18.68%** (`d42 m1`, 3,313 unique) and 13.24% (`d44 m1`, 3,618 unique) — and h³ is exactly
  the depth whose AUC goes negative. Saturation and depth track each other, starting at h³, not
  at h¹.

---

## 4. Verdict — one paragraph per depth

**h⁰.** Arm B's h⁰ gain is **+0.0100** against a FLOOR of 0.0767, positive in 5/5 worlds — a
**FAIL** on the bar, and expected to be one: h⁰ is a rank-≤7 linear reparametrisation of the very
features `P(Y|X)` already sees, so it cannot add information, and its near-zero gain is the
positive control confirming that. Its value in this report is not its own score but what it
establishes: `delay`'s representation *before any neighbour contact whatsoever* scores 0.7516,
carries no saturation at all (0.00% at p ≥ 0.999, 98% unique values), and is extraordinarily
stable across init seeds (floor 0.0038). Arm A's h⁰ row (−0.1074, floor 0.4313) is head-transfer
noise and is not evidence about h⁰.

**h¹ (retained).** Unchanged from Phase 0 and reproduced exactly: **+0.0386** against a FLOOR of
0.0778, 5/5 worlds positive, **FAIL** on the bar. What is new is that it is now bracketed from
below: h¹ beats h⁰ by **+0.0286 in all 25 cells and all 5 worlds**, with the init-seed floor of
that paired delta at 0.0140 — half the effect. The first round of neighbour-blending is
**reliably, if modestly, net-helpful**, and h¹ remains the best of the four depths tested for
`delay`.

**h².** +0.0245 against a FLOOR of 0.0784, 5/5 worlds positive, **FAIL** — unchanged from
`new_task_depth.md`. Paired within-cell, h¹ beats h² by **+0.0140** (5/5 worlds, 20/25 cells),
so the second blend gives back roughly half of what the first one won.

**h³.** −0.0123 against a FLOOR of 0.3038, only 1/5 worlds positive, **FAIL** and the worst of
the four — unchanged. It is also the only depth showing real saturation (1.63% pooled at
p ≥ 0.999, up to 18.68% in a single cell), which is the first time in this project's record that
the saturation diagnostic and the AUC degradation have been observed moving together on the same
axis.

### Which of the three patterns does the data show?

**Pattern 3: h¹ is measurably better than h⁰.** Not "essentially the same" (pattern 1) and
definitively not "worse" (pattern 2).

The evidence is one-directional and not close. On AUC, h¹ beats h⁰ in **25 of 25 cells** and 5 of
5 worlds by a mean of +0.0286, an effect 2× the init-seed floor of that paired delta. On the
saturation diagnostic, h¹ is *less* collapsed than h⁰ on all three measures, not more. There is
no reading of these numbers in which the first blend destroys signal — the
"summed-into-an-undifferentiated-64-slot-card" hypothesis is **not supported at h¹**, and this
is the arm that was designed to detect it if it were true.

Two qualifications belong with that, and neither is a reason to soften the direction:

1. **The effect is small and this grid cannot certify it.** +0.0286 is below the dataset-seed
   floor of the paired delta (0.0420), just as +0.0386 is below the unpaired floor of 0.0778.
   "Reliably positive in direction, uncertified in magnitude" is the precise claim; it is not a
   pass and is not reported as one.
2. **The pattern is not monotone in depth.** The first blend helps (+0.0286), the second gives
   half of it back (−0.0140), the third is actively destructive (h¹ beats h³ by +0.0509 in 25/25 cells). So
   pattern 3's tail clause holds as well: "deeper hurts" is a *separate* phenomenon from the
   first blending step, and the two must stop being described as one thing.

The hypothesis this run does **not** rule out is the thin-features one. h⁰ scores 0.7516 from 7
columns; one hop of neighbourhood buys +0.0286 on top; every further hop erodes it. Delay's
entire graph budget is one hop wide and worth under three AUC points — which is a statement about
**how little the 1-hop neighbourhood carries**, not about the aggregation destroying what reaches
it.

---

## 5. What this points to next

The three eliminations now on record for `delay` — depth (`new_task_depth.md`), readout weighting
(the depth gate), and missing schema (`new_nodes_result.md`) — are joined by a fourth: **the
first neighbour-aggregation step is not the problem.** It helps, consistently, in every cell.
That redirects the next fix rather than merely narrowing it:

1. **Widen `Shipment.x` — this is now the highest-value move, and it is the one the data names.**
   h⁰ = 0.7516 is what 7 columns are worth, and measured above chance it is **90%** of everything
   the full 4-layer encoder delivers (0.2516 vs h¹'s 0.2802 above 0.5). The representation is thin because the *input* is thin, and
   `reports/new_nodes_result.md` §7 already reached the same doorway from the other side. The
   specific test: add per-shipment observable columns (age-in-transit at t0, distance-to-promise,
   quantity/value, mode, historical on-time rate of *this* lane) and re-run Phase 0's grid. If
   `delay` moves, it moves here.

2. **Put new information on the 1-hop neighbourhood or on the edges, never deeper.** The gain
   profile (+0.0286 at hop 1, −0.0140 at hop 2, −0.0509 at hop 3) says any signal that must
   travel more than one hop to reach a Shipment will be eroded before it arrives. This is the
   concrete reason `new_nodes`' Carrier/Port/Route additions failed for `delay` while helping
   `shortage`: `shortage` reads at h³ and can wait for distant information; `delay` reads at h¹
   and cannot. Enrichment aimed at `delay` has to land as **edge attributes on Shipment's
   immediate relations**, not as new node types two hops out.

3. **Use paired within-cell contrasts as the default design for every future `delay`
   experiment.** The dataset-seed floor on the unpaired gain is 0.0767; on the paired h¹−h⁰
   delta it is 0.0420 — nearly half, for free, from holding the world and the checkpoint fixed.
   No `delay` effect of realistic size (≤ 0.03) can clear an unpaired floor of 0.077 on five
   worlds, so unpaired designs on this grid are structurally incapable of producing a pass. Any
   experiment intended to *pass* rather than to diagnose needs either the paired design or more
   dataset seeds.

4. **Stop attributing `delay`'s ceiling to saturation.** At h⁰ and h¹ the p ≥ 0.999 fraction is
   0.00% and 0.05%; `reports/decision_support_build.md` §3.2's 96.5% figure does not describe
   this pipeline at the depth `delay` actually reads (consistent with `ml/gate1_relevance.py`'s
   own note that it re-measures at 0.0%). Saturation is real at **h³** (1.63%, worst cell 18.68%)
   and should be investigated there, on `shortage`'s and `impact`'s behalf, not `delay`'s.

**Not recommended:** changing `MARKOV_READOUT_DEPTH["delay"]`. h¹ is now the best of four
measured depths rather than three, and h⁰ — the only untested alternative remaining — is worse
by every measure in this report.

---

## 6. Reproducing

```
python3 ml/layer3_h0_diagnostic.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

~3 min on CPU, all 25 cells from the checkpoint cache; trains nothing but the Arm B probe heads.
Writes `out/layer3_v3/delay_h0_diagnostic.json`. Both reproduction checks print before any table
and must read `max |diff| = 0.00e+00`.

For the prior depth table this extends:

```
python3 ml/layer3_depth_swap.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```
