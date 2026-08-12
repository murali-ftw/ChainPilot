# Layer 3 on V2 — Transformer 2 Where Hidden Dependencies Are Causal

**Status:** all five Transformer 2 files ported byte-identically, parameter counts exact, port
verified against V1's recorded numbers on all four fusion arms, and the brief's full targeted
sweep run to completion — **5 arms × 2 variants × 5 dataset seeds = 50 mid-scale runs**, plus 20
sanity runs and 8 v1-scale runs.

**The answer, stated first: the question still cannot be answered, and the reason has flipped.**
V1 could not test whether discovery converts to accuracy because its planted structure was
discoverable but **causally redundant** (Prerequisite 0). V2 supplies the causal coupling V1 asked
for — Mechanism D's alpha=0.35 on Type A groups is real and confirmed — but **Transformer 2's
retrieval does not discover V2's Mechanism B groups at all**, at either scale tested. Percentile
rank sits at 0.479–0.514 against a 0.500 chance baseline and pool discovery rate at
**0.77x–1.11x chance**, against V1's 0.73 rank and **7x chance**. With no discovery there is
nothing for any fusion design to fuse, and the AUC deltas behave accordingly.

**V1 had discovery without coupling. V2 has coupling without discovery.** Neither project has yet
had both at once — which is exactly what Prerequisite 0 said was required.

**The session's headline number, answered directly: no.** On Variant D — the one condition in
either project where a discovery-driven gain was possible — **not one of the four fusion arms
shows a sign-consistent AUC improvement on impact**. The deltas are −0.0005, −0.0042, +0.0003 and
+0.0004, and the only sign-consistent result among them is confidence-aware fusion consistently
**negative** (0+/5−). Prerequisite 0's prediction is not confirmed.

**And a second finding constrains how much any of that can be claimed.** Re-training sixteen
identically-configured runs (§4) puts this codebase's run-to-run reproduction floor at **0.0026
mean absolute AUC on impact, 0.0082 at worst**. Every arm-vs-baseline delta in this report is at
or below that floor. Sign consistency across dataset seeds does not rescue them, because all
seeds share the floor.

---

## 1. Port and verification

### 1.1 The five files, byte-identical

| file | md5 vs V1 | role |
|---|---|---|
| `ml/models/transformer2.py` | **identical** | `Transformer2GlobalAttention` — same-type, no adjacency mask, top-k=64 |
| `ml/models/rgcn_attn_variant_a_transformer2.py` | **identical** | plain additive fusion (`t2_scale` scalar, init 0) |
| `ml/models/transformer2_confidence.py` | **identical** | confidence-weighted fusion, zero new parameters |
| `ml/models/transformer2_trustgate.py` | **identical** | per-node learned trust gate |
| `ml/models/transformer2_crossattn.py` | **identical** | cross-attention over the retrieved pool |

Parameter counts reproduce V1's table **exactly**: plain 814,627, confidence 814,627, trustgate
818,787, crossattn 864,162, over Variant A's 765,090. All four reuse `Rung5VariantAGateHead` and
`PredictionHead` unmodified — nothing in Layer 1 or Layer 2 changed.

### 1.2 Sanity gate — the port reproduces V1

Variant 0 at the `v1` preset (the byte-identical anchor), 5 model-init seeds, 100 epochs,
hidden=128 / num_bases=10. V1's Round 1/Round 2 figures are used as a band, per the brief.

| arm | params | delay | shortage | impact | verdict |
|---|---|---|---|---|---|
| T2 plain | 814,627 | 0.8160 ± 0.0028 (V1 0.8144–0.8186) | 0.7976 ± 0.0031 (0.7991–0.7996) | 0.9420 ± 0.0011 (0.9425–0.9455) | **PASS** |
| T2 confidence | 814,627 | 0.8159 ± 0.0018 (0.8176) | 0.7972 ± 0.0031 (0.7995) | 0.9410 ± 0.0027 (0.9432) | **PASS** |
| T2 trustgate | 818,787 | 0.8169 ± 0.0017 (0.8185) | 0.7964 ± 0.0039 (0.7991) | 0.9428 ± 0.0034 (0.9438) | **PASS** |
| T2 crossattn | 864,162 | 0.8153 ± 0.0021 (0.8186) | 0.7998 ± 0.0024 (0.7992) | 0.9419 ± 0.0013 (0.9443) | **PASS** |

Every cell is inside ±0.02 of V1's band, largest gap −0.0033. **The port is verified; everything
below is a property of V2's data, not of the code.**

### 1.3 Ground truth — Type A/B/C membership is privileged, as required

`HP_GROUPS` in `db/generate_dataset.py` holds `dict(members=[...], type='A'|'B'|'C')` and is **never
emitted**: no CSV carries group id, type or membership, and `verify_no_hidden_state()` returns empty
on every variant. `ml/extract_hidden_state.py` was extended to read it the same way it reads
Mechanism E's resilience for Layer 2 — validation only, never a model input. At the v1 preset:
**38 groups (13 A / 13 B / 12 C), 160 members**; at mid scale, **90 groups (30/30/30)**, identical
between Variants B and D at the same seed, which is what makes them a clean matched pair.

### 1.4 The impact-head wiring still holds — but the reason has changed

V1 wired Transformer 2 to impact because the co-parent path measured 3 hops from Shipment (out of
reach at delay's h1) and 2 hops from Supplier (in reach at impact's h4). **That measurement does not
transfer to V2**, and re-running it was necessary. Measured on V2's own graphs:

| variant | Supplier -> Type A co-member | Shipment -> its supplier's Type A co-member |
|---|---|---|
| B | **4 hops** (76/90 pairs within 5) | 3 hops (44), 4 (9), 5 (5) |
| D | **4 hops** (64), 5 hops (12) | 3 hops (60), 4 (10), 5 (10) |
| K | 2 hops (4), **4 hops** (62), 5 (10) | 3 hops (47), 4 (19), 5 (9) |

V1's 2-hop figure came from the `component_suppliers` dual-sourcing path, which connects suppliers
that **share a component**. V2's Mechanism B members are drawn by shuffle from the whole supplier
pool and share no component, so the shortest path is `Supplier -> Component -> Product -> Component
-> Supplier` = **4 hops**. Only Variant K has any 2-hop pairs, via Mechanism J's `UPSTREAM_OF` edges.

**Conclusion: keep the impact wiring.** Delay reads h1 and needs 3+ hops — still structurally out of
reach. Impact reads h4 and needs 4 — reachable, but with **zero margin** at L=4, where V1 had two
layers of slack. Worth stating plainly: on V2 a 4-layer message-passing encoder can in principle
reach Type A co-members, so Transformer 2's "no graph path exists" premise is weaker here than in
V1 — though 14 of 90 pairs are still unreachable within 5 hops, and 4-hop propagation through mean
and attention aggregation is heavily diluted.

## 2. The discovery check — it fails, at both scales and in all four arms

This is the precondition the brief says to stop on. It is now measured across the full grid rather
than the partial one, and the verdict does not change.

### 2.1 At the v1 preset, discovery is at chance

| type | percentile rank (chance 0.500) | discovery rate | chance | ratio |
|---|---|---|---|---|
| A | 0.4670 | 0.1185 | 0.1045 | 1.13x |
| B | 0.5160 | 0.0879 | 0.1045 | **0.84x** |
| C (decoy) | 0.4896 | 0.1161 | 0.1045 | 1.11x |

V1's H_POLYMER on the same two metrics: **rank 0.7255, rate 0.7444 = 7.1x chance.** V2's Type B —
the group type explicitly built to reproduce H_POLYMER's behaviour — sits at 0.84x, below chance and
indistinguishable from the Type C decoy.

### 2.2 The cause is not the port, and only partly the scale

The v1 preset cannot support the check. Measured straight from the CSVs with no model involved:
**only 168 of 800 suppliers have usable observable trajectories**, leaving **9 measurable Type B
co-member pairs** — on which the *decoy* Type C scores higher (+0.53 vs +0.22, on 4 pairs). That is
what noise looks like. The generator says the same in its own comment: at `sup_n=800` the Type B
co-degradation check yields "6-7 groups, which is a power limit of the preset, not a property of the
mechanism" — and at that preset **the check does not execute at all**.

So the sweep was rebuilt at `sup_n=2,000`, where it does execute:

```
[PASS] B: Type B groups co-degrade inside their own event windows
       (mean delta -0.0924 over 20 groups, null [-0.0393, +0.0429]; must sit BELOW it)
```

**The co-degradation is verifiably present at 2,000 suppliers. Discovery still is not** — now
across all four fusion arms, 5 seeds each, ~1,500 ground-truth pairs per group type per arm:

| variant | arm | Type A rank / rate ratio | Type B rank / rate ratio | Type C rank / rate ratio |
|---|---|---|---|---|
| B | T2 plain | 0.5083 / 0.80x | 0.4861 / 1.08x | 0.4870 / 1.00x |
| B | T2 confidence | 0.5089 / 0.87x | 0.4859 / 1.10x | 0.4834 / 0.95x |
| B | T2 trustgate | 0.5141 / 0.85x | 0.4838 / 1.09x | 0.4837 / 0.90x |
| B | T2 crossattn | 0.5121 / 0.77x | 0.4790 / 1.07x | 0.4790 / 0.83x |
| D | T2 plain | 0.5068 / 0.90x | 0.4908 / 0.94x | 0.4998 / 1.11x |
| D | T2 confidence | 0.5098 / 0.90x | 0.4895 / 0.94x | 0.4936 / 0.97x |
| D | T2 trustgate | 0.5053 / 0.90x | 0.4953 / 1.01x | 0.4933 / 1.10x |
| D | T2 crossattn | 0.5091 / 0.81x | 0.4838 / 1.02x | 0.4863 / 0.95x |

**Every cell is at chance.** Rank spans 0.479–0.514 against a 0.500 baseline; every discovery-rate
ratio lands between 0.77x and 1.11x. The Type C **decoy** — the structure a healthy model should
ignore entirely — scores as well as Type A and Type B throughout, and on Variant D under plain and
trustgate fusion it scores *highest of the three*. That is the signature of a retrieval mechanism
responding to nothing.

The one consistent pattern is Type A's percentile rank sitting just above 0.500 (0.505–0.514 in
every arm on both variants) while its discovery **rate** sits at or *below* chance (0.77x–0.90x,
the lowest of the three types). A signal that small orders pairs marginally but never puts them
inside a top-64 pool — which is the only thing that would let fusion act on them.

### 2.3 Why V2's groups are harder to discover than V1's H_POLYMER

Not a defect, but it determines what a future dataset would need:

- **Uniqueness.** V1 planted **one** hidden group: 4 suppliers of 800 (0.5%), sharing a single
  dominant factor, in a world where nothing else was planted. Co-membership was the *only*
  explanation for two suppliers co-degrading. V2 plants **20% of all suppliers** into 38–90 groups
  with overlapping event windows, alongside Mechanism H's shocks and Mechanism C's rewiring. Shared
  degradation is now common, so it no longer identifies a group.
- **Retrieval budget.** top-k=64 is 8% of an 800-supplier pool but 3.2% of a 2,000-supplier pool.
  The bar rises with scale while the signal does not.
- **Magnitude.** V1's H_POLYMER factor ran at magnitude 0.95 over 110+ days; V2's Type B events draw
  `uniform(0.55, 0.95)` twice per group — individually weaker and spread thinner.

## 3. Downstream AUC — the full grid, with the precondition failure attached

Mid scale (`sup_n=2,000`), Variants B and D, **5 dataset seeds**, 40 epochs, all arms on an
identical budget so arm-vs-baseline is internally valid. Cross-scale AUC comparisons are not.

| var | arm | params | delay | pos | shortage | pos | impact | pos |
|---|---|---|---|---:|---|---|---|---|
| B | Variant A (no T2) | 765,090 | 0.7920 ± 0.0198 | 1,458 | 0.7776 ± 0.0074 | 2,737 | 0.9361 ± 0.0124 | 388 |
| B | T2 plain | 814,627 | 0.7929 ± 0.0198 | 1,458 | 0.7757 ± 0.0063 | 2,737 | 0.9393 ± 0.0075 | 388 |
| B | T2 confidence | 814,627 | 0.7925 ± 0.0185 | 1,458 | 0.7748 ± 0.0056 | 2,737 | 0.9416 ± 0.0058 | 388 |
| B | T2 trustgate | 818,787 | 0.7942 ± 0.0196 | 1,458 | 0.7765 ± 0.0080 | 2,737 | 0.9402 ± 0.0069 | 388 |
| B | T2 crossattn | 864,162 | 0.7922 ± 0.0183 | 1,458 | 0.7759 ± 0.0068 | 2,737 | 0.9359 ± 0.0095 | 388 |
| D | Variant A (no T2) | 765,090 | 0.7922 ± 0.0199 | 1,451 | 0.7807 ± 0.0067 | 2,747 | 0.9385 ± 0.0102 | 385 |
| D | T2 plain | 814,627 | 0.7914 ± 0.0213 | 1,451 | 0.7805 ± 0.0068 | 2,747 | 0.9380 ± 0.0098 | 385 |
| D | T2 confidence | 814,627 | 0.7922 ± 0.0198 | 1,451 | 0.7805 ± 0.0060 | 2,747 | 0.9343 ± 0.0118 | 385 |
| D | T2 trustgate | 818,787 | 0.7933 ± 0.0202 | 1,451 | 0.7800 ± 0.0067 | 2,747 | 0.9388 ± 0.0103 | 385 |
| D | T2 crossattn | 864,162 | 0.7912 ± 0.0206 | 1,451 | 0.7805 ± 0.0066 | 2,747 | 0.9389 ± 0.0091 | 385 |

### 3.1 Delta against the no-T2 baseline, paired by dataset seed

Sign column is positive/negative seed count; "sig" counts seeds whose own paired block bootstrap CI
excludes zero (2,000 resamples, 6 test snapshots as blocks, `paired_block_bootstrap` throughout).

| var | arm | delay | sign | sig | shortage | sign | sig | impact | sign | sig |
|---|---|---|---|---|---|---|---|---|---|---|
| B | T2 plain | +0.0009 | 4+/1− | 2/5 | −0.0019 | **0+/5−** | 1/5 | +0.0033 | 2+/3− | 2/5 |
| B | T2 confidence | +0.0005 | 3+/2− | 2/5 | −0.0028 | 1+/4− | 2/5 | +0.0055 | 4+/1− | 1/5 |
| B | T2 trustgate | +0.0023 | **5+/0−** | 3/5 | −0.0011 | **0+/5−** | 1/5 | +0.0041 | 3+/2− | 2/5 |
| B | T2 crossattn | +0.0002 | 3+/2− | 0/5 | −0.0017 | 2+/3− | 2/5 | −0.0002 | 2+/3− | 1/5 |
| D | T2 plain | −0.0008 | 1+/4− | 2/5 | −0.0002 | 2+/3− | 2/5 | −0.0005 | 2+/3− | 1/5 |
| D | T2 confidence | −0.0001 | 2+/3− | 2/5 | −0.0002 | 2+/3− | 1/5 | **−0.0042** | **0+/5−** | 1/5 |
| D | T2 trustgate | +0.0011 | 3+/2− | 1/5 | −0.0007 | 3+/2− | 1/5 | +0.0003 | 1+/4− | 1/5 |
| D | T2 crossattn | −0.0010 | 2+/3− | 1/5 | −0.0002 | 2+/3− | 3/5 | +0.0004 | 2+/3− | 1/5 |

### 3.2 The headline number

**On Variant D, does any fusion arm show a real, sign-consistent AUC improvement on impact that
Variant B does not show for the same arm?**

**No.** Variant D's four impact deltas are −0.0005, −0.0042, +0.0003, +0.0004. None is
sign-consistent positive; three of four are 2+/3− or 1+/4−, i.e. coin-flips. The only
sign-consistent impact result on Variant D is confidence-aware fusion at **0+/5− — consistently
negative**, the opposite of Prerequisite 0's prediction.

The larger impact means sit on **Variant B, the alpha=0 control** (+0.0033, +0.0055, +0.0041) — the
variant where discovery-driven gain is *definitionally impossible*, since Type A groups there have
no causal coupling at all. A gain that appears on the control and not on the treatment is not a
discovery effect.

**And those Variant B means are one seed.** Per-seed impact deltas on Variant B:

| arm | seed 42 | 43 | 44 | 45 | **46** | mean w/o 46 |
|---|---|---|---|---|---|---|
| T2 plain | −0.0016 | −0.0030 | +0.0019 | −0.0012 | **+0.0202** | −0.0010 |
| T2 confidence | −0.0001 | +0.0011 | +0.0009 | +0.0027 | **+0.0229** | +0.0012 |
| T2 trustgate | −0.0027 | −0.0024 | +0.0015 | +0.0015 | **+0.0228** | −0.0005 |

On seed 46 the no-T2 baseline itself came in at impact 0.9139 against 0.93–0.95 on every other
seed — an underperforming baseline, which every arm then beats by ~0.022 at once. Drop that one
seed and Variant B's apparent advantage disappears entirely. Symmetrically, Variant D's
confidence result is carried by seed 45 (−0.0130, the single largest-magnitude delta in the grid
and the only one whose own CI excludes zero); without it the mean is −0.0021.

**Both of this session's largest effects are single-seed artifacts pointing in opposite
directions.** The bootstrap agrees: across 24 arm/task cells, no cell has more than 3 of 5 seeds
with a CI excluding zero, and the cells that do (B trustgate delay 3/5, D crossattn shortage 3/5)
are not the cells with consistent signs. Significance scattered across a minority of seeds with
disagreeing directions is what noise looks like, not an effect.

## 4. The reproduction floor — how large a delta has to be before it means anything

This was not in the brief. It emerged because seeds 42/43 had to be retrained to capture the
per-row predictions the paired bootstrap needs, which produced **16 pairs of identically
configured runs** — same architecture, same dataset, same model seed, same 40-epoch budget, the
only difference being the process environment (thread count, and therefore CPU float reduction
order; `scatter`/`index_add` in message passing is not order-deterministic either).

| task | mean signed | mean abs | max abs | std |
|---|---|---|---|---|
| delay | +0.0002 | 0.0024 | 0.0121 | 0.0037 |
| shortage | −0.0001 | 0.0007 | 0.0031 | 0.0010 |
| impact | −0.0009 | **0.0026** | **0.0082** | 0.0031 |

Individual pairs range from +0.0059 to −0.0082 on impact for configurations that differ in
nothing at all.

**Every impact delta in §3.1 is at or below this floor** — the largest, Variant B confidence at
+0.0055, is smaller than the worst single reproduction gap. So is the largest delay delta
(+0.0023 against a 0.0024 floor). Only shortage, whose floor is 0.0007, has deltas that clearly
exceed it — and those are the uniformly **negative** ones (B plain −0.0019 at 0+/5−, B trustgate
−0.0011 at 0+/5−), i.e. the clearest real effect in this grid is Transformer 2 mildly *hurting*
the task it is not wired to.

**Sign consistency across dataset seeds does not rescue a sub-floor delta.** The floor is a
property of the training process, not of the dataset seed, so all five seeds share it; five
same-signed sub-floor deltas are five draws from the same biased-by-noise process, not five
independent confirmations. This is a stronger caveat than "n=5 is small," and it applies to every
comparison in this report — including, symmetrically, the negative ones.

## 5. Trust-gate stability — V1's instability reproduces, and worsens with more seeds

| variant | seed | mean trust | std | pos / neg / ~0 | corr(abs trust, degree) |
|---|---|---|---|---|---|
| B | 42 | +0.1447 | 0.1459 | 75.2% / 23.5% / 1.3% | −0.1618 |
| B | 43 | +0.0014 | 0.1821 | 16.1% / 72.5% / 11.4% | −0.1930 |
| B | 44 | +0.0213 | 0.0550 | 69.4% / 23.6% / 7.0% | −0.0646 |
| B | 45 | +0.2364 | 0.1850 | 79.0% / 19.2% / 1.9% | −0.2960 |
| B | 46 | **−0.0873** | 0.0860 | 22.9% / **75.8%** / 1.3% | −0.2145 |
| D | 42 | +0.1860 | 0.1268 | 90.2% / 9.4% / 0.3% | −0.1206 |
| D | 43 | +0.0436 | 0.0252 | **98.7%** / 0.0% / 1.3% | −0.1189 |
| D | 44 | **−0.1449** | 0.3121 | 26.3% / **71.5%** / 2.1% | −0.1343 |
| D | 45 | +0.1581 | 0.1303 | 74.0% / 3.4% / 22.7% | −0.3015 |
| D | 46 | +0.1663 | 0.2570 | 66.0% / 32.8% / 1.2% | −0.2743 |

**The same bimodal, seed-dependent signature V1 found, now across ten runs instead of four.** Mean
trust spans −0.145 to +0.236. The share of suppliers the gate trusts spans **16% to 98.7%** — on
Variant D, seed 43 trusts Transformer 2 for essentially every supplier while seed 44 distrusts it
for 72%, on the same dataset variant with the same architecture. The degree correlation is
consistently negative here (−0.06 to −0.30), where V1's flipped sign across seeds; that is the one
respect in which V2 is *more* stable than V1.

**The question V1 could not ask, now asked properly.** V2's typing lets trust be compared across a
causally coupled group (Type A), a redundant one (Type B), a decoy (Type C) and the general
population, in the same run:

| variant | Type A | Type B | Type C | non-members | A − non-member |
|---|---|---|---|---|---|
| B (mean over 5 seeds) | +0.0690 | +0.0594 | +0.0630 | +0.0632 | **+0.0058** |
| D (mean over 5 seeds) | +0.0844 | +0.0845 | +0.0918 | +0.0806 | **+0.0038** |

**The gate does not trust Transformer 2 more for the causally coupled groups.** Type A members sit
within 0.006 of the general population on both variants, and on Variant D the **decoy** Type C
receives the highest mean trust of the three types — the exact opposite of the ordering a working
mechanism would produce. This is the same null V1 reported for H_POLYMER, now measured against a
group that genuinely *is* causally informative, which makes it a materially stronger null. It is
also exactly what §2 predicts: with retrieval at chance, a Type A supplier's pool contains no more
of its co-members than anyone else's does, so there is nothing for the gate to learn to trust.

## 6. What was run

| grid | scale | arms | variants | seeds | epochs | runs |
|---|---|---|---|---|---|---|
| sanity gate | v1 preset | 4 T2 arms | Variant 0 | 5 model-init | 100 | 20 |
| discovery + AUC | v1 preset | T2 arms + baseline | B, D | partial | 100 | 8 |
| **discovery + AUC** | **sup_n=2,000** | **5 (4 T2 + baseline)** | **B, D** | **5** | **40** | **50** |

The mid-scale grid is the brief's stated minimum, complete: 5 arms × 2 variants × 5 seeds. Three
things differ from the brief, all stated:

1. **40 epochs at mid scale, not 100.** A mid-scale run costs ~6.5 min at 40 epochs against
   ~16 min at 100. Every mid-scale arm shares the identical budget, so arm-vs-baseline is
   internally valid; cross-scale AUC comparisons are not.
2. **Variant K was not run.** The spec's comparison rule frames K-vs-D as testing whether
   Mechanism E/F absorption *erodes an accuracy gain found on D*. No gain was found on D, so
   there is nothing for absorption to erode, and the comparison cannot produce an interpretable
   result. It would also require generating mid-scale K datasets that do not exist on disk. This
   is a reasoned omission, not a time constraint.
3. **The v1-preset AUC grid was abandoned mid-run** once §2.2 established that discovery there is
   at chance and that the preset has only 9 measurable Type B pairs. The 8 completed runs are kept
   as §2.1's evidence.

**A harness bug was found and fixed.** `ml/data/loader.py`'s snapshot cache keyed on the directory
*basename*, so `db/csv_mid/vB_seed42` and `db/csv_v1scale/vB_seed42` collided and one configuration
silently served the other's bundles. Caught because mid-scale runs finished at suspiciously
v1-scale timings. The key now includes the parent directory. No earlier result in this project is
affected — only one build directory per basename existed before this session.

**A memory limit was hit and characterised.** Four concurrent training processes on `sup_n=2,000`
data exceed this machine's 24 GB and are killed by the OS. One process holds ~3.5 GB steady and
peaks ~7.7 GB building a snapshot cache. Two concurrent processes are both safe and *faster* —
~390 s per run at 2-way against ~1,240 s at 4-way, so the wider parallelism was thrashing.

## 7. What this settles about Transformer 2's production readiness

**It does not overturn V1's recommendation, and it does not confirm it either.** Confidence-Aware
Fusion remains defensible on V1's own grounds — zero added parameters, best-preserved discovery
quality, least new trainable surface. Nothing here contradicts that, and nothing here supports it
on accuracy: on Variant D it is the one arm that is sign-consistently *negative* on impact. The
accuracy case for Transformer 2 remains unproven; this session moved the *reason*, not the verdict.

**Cross-attention fusion, run here across all 5 seeds at mid scale, remains the least justified
arm** — 864,162 parameters against the plain arm's 814,627 for the weakest discovery retention of
the four (Type A rate ratio 0.77x–0.81x, the lowest in the grid) and impact deltas of −0.0002 and
+0.0004. V1's judgement on it holds on V2's data.

**What was learned that V1 could not have learned:**

- V2 **does** supply the causal coupling Prerequisite 0 demanded — Mechanism D's alpha=0.35 on Type
  A groups, with Type B as a matched alpha=0 control and Type C as a decoy, in one dataset. That
  half of the gap is genuinely closed.
- V2 **does not** supply discoverability. Mechanism B's groups are not separable by same-type
  embedding similarity at 800 or 2,000 suppliers, in any of four fusion designs, even where their
  co-degradation is confirmed by the generator's own permutation-null check. **Planting a structure
  that is causally relevant is a different problem from planting one that is findable, and V2
  solved the first while regressing on the second.**
- The trust gate's instability is not V1-specific: it reproduces on a different generator across
  ten runs, and now demonstrably fails to concentrate trust on causally coupled suppliers even when
  they exist — ranking the decoy above them on Variant D.
- **This codebase's AUC deltas have a floor of ~0.003 on impact** (§4). That is a general result
  about every comparison in this project, not just Layer 3, and it is the first time it has been
  measured directly.

**What would actually resolve this**, stated as concretely as V1 stated its own version: a variant
in which hidden-parent groups are both **causally coupled** (as Mechanism D already does) **and
distinctive** — few enough, with a strong enough shared factor, that co-membership is the dominant
explanation for two suppliers' correlated history, the way V1's single 4-member H_POLYMER group was.
Concretely: a low `hidden_parent_rate` with a high-magnitude shared factor, or a dedicated Type A+
group planted at V1's proportions alongside V2's existing type mix. That is a generator change, not
a model change, and it is the one experiment that would let Prerequisite 0 finally be tested end to
end.

**Standing caveat, as everywhere else in this project:** none of this is at spec scale. The AUC
cells rest on 385–2,747 test positives at `sup_n=2,000`, and the discovery result is measured at two
scales, neither of which is the benchmark's own configuration. The spec-scale corpus on disk is
where this should ultimately be settled — pending the GPU run every other null in this project is
already waiting on. A GPU run would also re-open the reproduction floor question, which was
measured on CPU and would need re-measuring on different hardware before any delta is believed
there either.

## 8. Provenance

**94 trainings, 78 reported configurations.** 20 sanity (v1 preset, 5 model-init seeds, 100
epochs), 8 v1-scale discovery/AUC, and 50 mid-scale (5 arms × 2 variants × 5 seeds, 40 epochs). A
further 16 mid-scale trainings are not separate configurations: seeds 42/43 were trained a second
time to capture the per-row predictions the paired bootstrap needs, and their first trainings are
retained in `out/t2mid_prerun/` as the §4 reproduction-floor measurement. All reported mid-scale
numbers come from the second set, so the whole grid shares one process environment.

New code: `ml/run_transformer2.py` (trains and captures discovery quality per group type, trust-gate
stability, and — added this session — per-row test predictions with snapshot blocks via
`--save-preds`), `ml/analyze_transformer2.py` (aggregation, sign consistency, paired block
bootstrap), `ml/reproduction_floor.py` (§4), extensions to `ml/extract_hidden_state.py` (privileged
`HP_GROUPS` read), and the four T2 arms wired into `ml/train.py` and `ml/models/encoder.py`.

Mid-scale datasets (`sup_n=2,000`, Variants B and D, 5 seeds) generated to `db/csv_mid/`, gitignored.
Raw per-run results in `out/t2mid/` and `out/t2_v1scale/`; aggregated tables in
`out/t2mid_summary.json`; ground truth in `out/hidden_mid/`. CPU throughout, 2 concurrent processes
at 6 threads each; MPS not used, per `reports/phase7_training_results.md` section 8.1's measurement
that it does not parallelise and would put a device difference inside these deltas.

**All five ported model files remain md5-identical to V1's**, re-verified at the close of this
session.

## 9. Retrieval Redesign — the contrastive ceiling, and what it does not buy

This section is the direct continuation of §7's own recommendation. §2 found Transformer 2's
retrieval at chance on every group type at both scales, and §7 diagnosed why: the embeddings
being compared were never trained to encode co-membership. This session tested that diagnosis
by pointing a loss straight at the retrieval geometry (Stage 1), then asked whether the same
signal is recoverable without privileged supervision, the way a deployment would have to work
(Stage 2).

**The answer, stated first: the diagnosis in §7 was right, and it does not help.** Contrastive
supervision moves retrieval enormously — Recall@64 from 0.75x chance to **10.9x**, MRR to
**24.3x** — but *only on the groups it was shown*. On held-out groups, drawn from the same
generator by the same process and differing in nothing but whether the loss saw them, retrieval
sits at **0.74x–1.50x chance, and 0 of 20 cells clear the reproduction floor** on either variant.
(The 1.50x is the single highest held-out cell — Variant B, Type B, MRR — and it beats chance on
2 of 5 seeds with a gap of 0.0063 against a floor of 0.0153, i.e. it is noise, not a residual
effect.) The representation memorised the 41 groups it was shown; it learned nothing transferable
about co-membership. And it paid for that memorisation with **impact AUC −0.145 on
Variant B and −0.158 on Variant D, sign-consistent 5/5 on both** — roughly thirty times the
arm's own AUC floor, and the largest effect this project has measured anywhere.

**Stage 2 is a clean null.** Retrieval built from the same observable column the generator's own
co-degradation check uses does not clear chance: Type B, the one type that co-degrades by
design, reaches 1.15x–1.42x on Recall@64 but on only 3–5 of 5 seeds, while the Type C **decoy**
— which must be inert — also lands above chance on 2–3 of 5 seeds and outscores Type B on MRR
twice. The decoy-controlled contrast is positive on 2–4 of 5 seeds and swings from −0.41 to
+1.32.

**And a probe explains why, which neither stage could on its own.** Fitting the same objective on
each frozen encoder depth against a **size-matched permuted-label null** shows that at *no* depth
— not `h^0`, not `h^4`, not `z_impact`, not even the raw input features — does co-membership
separate from a random regrouping by more than the probe's own null (gaps −0.077 to +0.265
against a chance level of 7.600). Two suppliers sharing a hidden parent are distinguished from
two that do not by nothing the model can see, only by their individual identities. That is one
finding, not three: it is why Stage 1 could memorise and not generalise, and why Stage 2 found
nothing to recover.

**Stop condition hit: the third one.** The signal is teachable under privileged supervision and
not recoverable from observable data alone at this label volume. Stage 3, Stage 4 and Stage 5
were not built. Stage 6 remains blocked.

### 9.1 Disclosed deviation — privileged ground truth shaped gradients, for the first time

Every earlier privileged read in this project was **eval-only**: Mechanism E's `resilience` for
the Layer 2 gate diagnostic, `HP_GROUPS` for §2's discovery check. Both were read to *score* a
model, never to fit one. Stage 1 is the first component in either project where the generator's
internal `HP_GROUPS` reaches a **gradient**, as the target of an auxiliary InfoNCE loss
(`ml/models/contrastive.py`).

This is stated here, in the same section that reports the result, and not as a footnote, because
it changes what the number means. **"Can this representation encode co-membership if explicitly
taught to" is a different and strictly easier claim than "does this representation discover
co-membership."** Every Stage 1 figure below is a ceiling under privileged supervision. It is
never evidence of emergent discovery.

Two further properties bound the claim, and both make it *weaker* than it looks:

- The supervision covers the **same suppliers scored at test time** — the setting is
  transductive, so memorisation is permitted, and it is exactly what the model did.
- `HP_GROUPS` is never a model *input*. The group labels enter only through the loss term; the
  forward pass, the parameter count (**814,627**, identical to the plain T2 arm) and the emitted
  data are unchanged. `verify_no_hidden_state()` returns empty on all ten mid-scale
  variant-seeds, re-checked this session.

To separate memorisation from generalisation, supervision covered a random **70% of the Type
A/B groups**; the rest were held out of the loss entirely and scored separately. On the seed-42
pair that is 41 groups supervised and 17 held out, of 87 groups (29 A / 29 B / 29 C). *En
passant*, a small correction to §1.3: it records mid scale as "**90 groups (30/30/30)**", which
is seed 44's count specifically — across the five dataset seeds the figure ranges **85 to 90**
(85/87/88/88/90), identical between Variants B and D at the same seed, on 399–400 members. The
matched-pair property §1.3 rests on is unaffected; only the single headline count was seed-specific.
The supervised split is the ceiling the brief asks for; the held-out split is the one that would
matter in deployment.

### 9.2 The metrics, and the floor §7 said was missing

`ml/analyze_transformer2.py` now reports **Recall@K, Precision@K and MRR** alongside §2's
percentile rank and pool discovery rate, per group type, per variant, per seed
(`ml/retrieval_metrics.py`). Every metric carries its own chance baseline: Recall@K and
Precision@K analytically, MRR by Monte Carlo over random rankings, percentile rank at 0.500.
The implementation was validated end to end against a perfect retriever (1.0 on all three) and a
random one at realistic scale (rank 0.493–0.512, Recall@64 0.034 against a 0.032 baseline).

**§4 measured a floor for downstream AUC but never for retrieval quality itself.** That gap is
closed here. Every configuration was trained **twice**, identically — same architecture, same
dataset, same model seed, same 40-epoch budget, same device, same serial process — giving **10
replicate pairs per arm, 20 in total**.

| arm | group set | pct rank | discovery rate | Recall@64 | MRR |
|---|---|---|---|---|---|
| T2 plain (control) | A | 0.0047 / **0.0132** | 0.0062 / 0.0153 | 0.0056 / **0.0121** | 0.0025 / **0.0081** |
| T2 plain (control) | B | 0.0040 / 0.0062 | 0.0056 / 0.0113 | 0.0044 / 0.0093 | 0.0020 / 0.0049 |
| T2 contrastive | A\|supervised | 0.0214 / **0.0699** | 0.0451 / 0.0727 | 0.0418 / **0.0847** | 0.0526 / **0.0821** |
| T2 contrastive | A\|held_out | 0.0327 / **0.0683** | 0.0130 / 0.0282 | 0.0110 / **0.0231** | 0.0072 / **0.0181** |

*(mean abs / max abs, over 10 pairs each; contrastive rows at final-epoch weights.)*

**The floor is not small, and it is arm-dependent.** The control reproduces to within 0.013 on
percentile rank; the contrastive arm moves by up to **0.070** on the same metric and **0.085**
on Recall@64 with nothing changed at all. Any Stage 1 claim has to clear the larger number, and
the gate below uses **max abs**, not mean abs — the same conservative choice §4 made when it
judged deltas against 0.0082 rather than 0.0026.

The AUC floor was re-measured on this session's device rather than carried over from §4:

| arm | delay | shortage | impact |
|---|---|---|---|
| T2 plain (control) | 0.0023 / 0.0105 | 0.0007 / 0.0023 | **0.0032 / 0.0053** |
| T2 contrastive | 0.0018 / 0.0043 | 0.0040 / 0.0136 | **0.0182 / 0.0363** |
| §4, CPU, for reference | 0.0024 / 0.0121 | 0.0007 / 0.0031 | 0.0026 / 0.0082 |

The control's MPS floor (0.0032 mean abs on impact) is close to §4's CPU floor (0.0026), so
**running serially on one device did not remove the floor** — it is not merely an artifact of
the thread-count variation §4 attributed it to. The contrastive arm's impact floor is 5.7x
larger, which is itself a finding: that objective is a far less stable optimisation.

### 9.3 The control — §2 replicates on a different device, and on three new metrics

| variant | type | pct rank | Recall@64 | ratio | MRR | ratio | discovery | ratio |
|---|---|---|---|---|---|---|---|---|
| B | A | 0.5116 | 0.0241 | 0.75x | 0.0091 | 0.75x | 0.0313 | 0.75x |
| B | B | 0.4811 | 0.0327 | 1.02x | 0.0120 | 0.96x | 0.0429 | 1.03x |
| B | C | 0.4863 | 0.0299 | 0.93x | 0.0118 | 0.97x | 0.0408 | 0.97x |
| D | A | 0.5052 | 0.0248 | 0.78x | 0.0087 | 0.72x | 0.0337 | 0.80x |
| D | B | 0.4904 | 0.0293 | 0.92x | 0.0136 | 1.09x | 0.0403 | 0.96x |
| D | C | 0.4913 | 0.0349 | 1.09x | 0.0127 | 1.04x | 0.0437 | 1.04x |

Percentile ranks land within 0.006–0.009 of §2's CPU figures (§2: A 0.5083/0.5068, B
0.4861/0.4908, C 0.4870/0.4998) and every ratio sits between 0.72x and 1.09x. **§2's null is
device-independent, and the three new IR metrics agree with it** — which also means Recall@K,
Precision@K and MRR were not needed to *find* the null, but they are what makes Stage 1's
ceiling measurable at all, since percentile rank saturates far earlier than they do.

### 9.4 Stage 1 — the ceiling, and the split that matters

Final-epoch weights, 5 dataset seeds, mid scale, `top_k=64`:

| variant | group set | pct rank | Recall@64 | chance | ratio | MRR | chance | ratio |
|---|---|---|---|---|---|---|---|---|
| B | A\|supervised | 0.7410 | 0.2696 | 0.0320 | **8.42x** | 0.2246 | 0.0123 | **18.28x** |
| B | B\|supervised | 0.7285 | 0.2496 | 0.0320 | **7.80x** | 0.2170 | 0.0122 | **17.76x** |
| B | A\|held_out | 0.4960 | 0.0242 | 0.0320 | 0.76x | 0.0134 | 0.0115 | 1.17x |
| B | B\|held_out | 0.5025 | 0.0296 | 0.0320 | 0.93x | 0.0190 | 0.0127 | 1.50x |
| B | C (decoy) | 0.5145 | 0.0302 | 0.0320 | 0.94x | 0.0149 | 0.0122 | 1.22x |
| D | A\|supervised | 0.7741 | 0.3485 | 0.0320 | **10.89x** | 0.2989 | 0.0123 | **24.33x** |
| D | B\|supervised | 0.7252 | 0.2619 | 0.0320 | **8.18x** | 0.2431 | 0.0122 | **19.89x** |
| D | A\|held_out | 0.5020 | 0.0249 | 0.0320 | 0.78x | 0.0084 | 0.0115 | 0.74x |
| D | B\|held_out | 0.4917 | 0.0300 | 0.0320 | 0.94x | 0.0114 | 0.0127 | 0.90x |
| D | C (decoy) | 0.5094 | 0.0403 | 0.0320 | 1.26x | 0.0172 | 0.0122 | 1.41x |

Three things in this table, in order of importance:

1. **The ceiling is real and very high.** On supervised groups every one of the five metrics
   clears chance by more than the floor on **20 of 20** variant/metric cells, 5/5 seeds each,
   with gaps 3–8x the worst reproduction gap. The representation *can* be made to encode
   co-membership. §7's diagnosis was correct.
2. **Generalisation is exactly zero.** Held-out groups clear on **0 of 20** cells. They are the
   same kind of object — Mechanism B draws all groups by one identical process, and the only
   difference is which side of a random 70/30 split they landed on — so this is not a
   distribution-shift result. It is memorisation with no structure learned.
3. **The decoy behaves correctly**, which is what makes the rest trustworthy. Type C, never
   supervised, stays at 0.94x–1.26x. The contrastive loss did not simply inflate every
   similarity.

**The reported ceiling is a lower bound.** The contrastive term was still descending at the
40-epoch budget (54.1 to 20.3 per epoch against a chance value of 6·ln(1999) = 45.6), so a
longer budget would raise the supervised numbers further. It would not touch the held-out ones,
which is the point.

At the **best-validation-AUC checkpoint** — the weights every AUC number in this report rests on
— the ceiling is materially lower (Variant B A-supervised: Recall@64 4.86x rather than 8.42x)
and three cells fall under the floor, because that checkpoint is selected on task AUC and often
predates the contrastive term taking effect. Both checkpoints are recorded per run; the
final-epoch figures are used above because they are the honest measurement of what the objective
achieves.

**One setup finding worth recording, because it nearly produced a false null.** With the
encoder's `dropout=0.2` active, the contrastive term does not move at all — it sits pinned at
exactly ln(N−1) = 7.60 at every loss weight tried (1, 10 and 50 give indistinguishable
trajectories). The identical run with dropout off reaches 1.98 in 300 steps. Dropout noise
swamps the cosine ordering the loss is trying to impose over 2,000 negatives at temperature 0.1.
The contrastive term is therefore computed on a **dropout-free forward pass**, which is also the
correct object: retrieval runs at inference, where dropout is off. Had this not been chased
down, this session would have reported the Stage 1 hard stop — and it would have been an
artifact of the training setup, not a property of the representation.

### 9.5 What the ceiling costs

| variant | arm | delay | shortage | impact |
|---|---|---|---|---|
| B | T2 plain (control) | 0.7927 ± 0.0191 | 0.7767 ± 0.0067 | **0.9406 ± 0.0056** |
| B | T2 contrastive | 0.7809 ± 0.0196 | 0.7670 ± 0.0070 | **0.7956 ± 0.0158** |
| D | T2 plain (control) | 0.7908 ± 0.0208 | 0.7807 ± 0.0069 | **0.9395 ± 0.0095** |
| D | T2 contrastive | 0.7823 ± 0.0199 | 0.7719 ± 0.0046 | **0.7814 ± 0.0329** |

Paired by dataset seed, the impact delta is **−0.1450 on Variant B (5/5 negative)** and
**−0.1580 on Variant D (5/5 negative)**, per-seed range −0.116 to −0.193. Against the
contrastive arm's own impact floor of 0.0363 max abs, that is a factor of about thirty — by a
wide margin the largest and most sign-consistent effect measured anywhere in this project, and
it points the wrong way.

This matters beyond Stage 1. **Stage 4 would have reintroduced Confidence-Aware Fusion on top of
this retriever**, and the retriever only reaches its ceiling by degrading the very head that
fusion feeds. Even had Stage 2 succeeded, a mechanism that buys retrieval quality at −0.15 impact
AUC has no path to a net gain.

### 9.6 Stage 1 gate — the decision, computed rather than eyeballed

`ml/stage_gate.py` applies the brief's bar in code. A metric clears only when all three hold:
it beats chance in the right direction by more than the **worst** single reproduction gap for
that same metric and group set; the per-seed values agree in sign; and it does so on at least
4 of 5 seeds individually. Sign consistency alone is explicitly not sufficient, per §4 — the
floor is a property of the training process, so all seeds share it.

| group set | cells clearing |
|---|---|
| supervised A/B (the ceiling the brief defines) | **20 of 20** |
| held-out A/B (generalisation) | **0 of 20** |
| Type C decoy | 0 of 10 |

**Gate: OPEN.** Contrastive retrieval clears chance by far more than its own reproduction floor,
so the brief's first hard stop was not triggered and Stage 2 was built. Recorded plainly: the
gate opened on the *supervised* number, which is the definition the brief set, while the
held-out result already forecast what Stage 2 would find.

### 9.7 Stage 2 — observable co-failure

#### 9.7.1 The confound, confirmed rather than assumed

The brief asks whether the Type A/B co-failure confound the audit raised actually applies to
Variants B and D. **It does not, in the form the audit described.** `VARIANTS` in
`db/generate_dataset.py` gives B = `("B",)` and D = `("B", "D")`; executing the generator
in-process at mid scale confirms `SHOCK_EVENTS = 0` and `CS_REWIRES = 0` on both. Neither
Mechanism H's shocks nor Mechanism C's rewiring is in the B/D generation path.

**But a larger confound is live, and it is not Mechanism H.** The base world every variant
inherits from V1 carries its own shared hidden-factor event pools inside `own_stress`, and at
`sup_n=2,000` they are big:

| pool | suppliers | share |
|---|---|---|
| H_PORT | 480 | 24.0% |
| H_TRUCK | 320 | 16.0% |
| H_CUSTOMS | 200 | 10.0% |
| H_POLYMER | 4 | 0.2% |
| **any base pool** | **1,000** | **50.0%** |

That yields **185,906 co-degrading pairs from a shared base event pool against 512 real Type
A/B pairs — a ratio of 363:1**. Only 8.2% of A/B pairs also share a base pool, so group
membership is not redundant with pool membership; the problem is one of sheer numbers. Any
retriever ranking on co-failure correlation is swamped by base-pool companions before it reaches
a hidden-parent co-member. This is a materially bigger obstacle than the audit anticipated, and
it is a property of the dataset, not of the retriever.

Because two of the three pools are defined by `country` (H_TRUCK = USA/Mexico, H_CUSTOMS =
Germany) and the third by sea freight — all observable, or closely proxied by observable lead
time — Stage 2 reports a **cohort-residualised** retriever alongside the raw one, removing what
observable covariates explain before correlating. That is still observable-only, and it is the
fair version of the test rather than a handicapped one.

#### 9.7.2 The observable, and the leakage guardrail

§2.2's co-degradation result — the evidence that the signal is present at all — came from the
generator's own suite reading `stf_rows` column 4, which is **`on_time_rate_90d` in
`supplier_temporal_features.csv`**, compared against the fleet mean at the same `as_of_date`
inside each group's event windows (`_windowed_delta`). Stage 2 builds its retrieval feature from
**that same column, detrended the same way**, so a null here cannot be dismissed as having aimed
at the wrong signal.

**The guardrail ran before any Stage 2 number was believed**, and it is adversarial rather than
declarative (`ml/test_cofailure_leakage.py`, re-run inline on the exact matrices used):

| corruption of every post-`t0` observation | max score delta | retrieved pool |
|---|---|---|
| shuffled across suppliers | 3.91e-15 | identical |
| blanked | 3.91e-15 | identical |
| replaced with garbage | 3.91e-15 | identical |
| zeroed | 3.91e-15 | identical |
| **shuffling the PAST (must change)** | **1.00e+09** | **changed** |

The residual 4e-15 is float64 reduction-order noise from pandas re-blocking, thirteen orders of
magnitude below what a real peek would produce; the retrieved pool is identical index for index.
The converse check is included because a test that only asserts invariance passes trivially
against a retriever that ignores its input — the failure mode Phase 0's `k % 3` bug already cost
this project once. **PASSED on every variant-seed.**

#### 9.7.3 The result

Suppliers carrying a usable trajectory (N = 704–708 of 2,000), 5 seeds, pooled:

| variant | mode | type | pct rank | Recall@64 | ratio | MRR | ratio | seeds > chance |
|---|---|---|---|---|---|---|---|---|
| B | raw | A | 0.5097 | 0.0839 | 0.88x | 0.0115 | 0.76x | 1/5 |
| B | raw | **B** | 0.4678 | 0.1335 | **1.41x** | 0.0193 | 1.21x | 4/5 |
| B | raw | C (decoy) | 0.5350 | 0.1095 | 1.15x | 0.0209 | **1.33x** | 3/5 |
| B | cohort-resid. | **B** | 0.4618 | 0.1195 | **1.26x** | 0.0192 | 1.21x | 3/5 |
| B | cohort-resid. | C (decoy) | 0.5462 | 0.0892 | 0.94x | 0.0218 | **1.39x** | 2/5 |
| D | raw | **B** | 0.5163 | 0.1139 | **1.19x** | 0.0195 | 1.19x | 4/5 |
| D | raw | C (decoy) | 0.4969 | 0.1014 | 1.06x | 0.0132 | 0.83x | 2/5 |
| D | cohort-resid. | **B** | 0.5365 | 0.1300 | **1.36x** | 0.0323 | 1.96x | 4/5 |
| D | cohort-resid. | C (decoy) | 0.5035 | 0.1000 | 1.05x | 0.0171 | 1.07x | 3/5 |

**This does not clear chance.** Type B — the one type that co-degrades by design, and the one
§2.2's check detects — is elevated to 1.19x–1.41x on Recall@64, in the right direction, but:

- **sign consistency is 3–5 of 5 seeds, never the 5/5 this project's standard requires**;
- the **Type C decoy, which must be inert, is also above chance on 2–3 of 5 seeds** and beats
  Type B on MRR in two of the four variant/mode cells (1.33x and 1.39x against 1.21x);
- the **decoy-controlled contrast (B − C) is positive on only 2–4 of 5 seeds**, ranging from
  −0.41 to +1.32 — a mean effect carried by one or two seeds, the exact pattern §3.2 identified
  as noise;
- **percentile rank is at chance everywhere** (0.462–0.546 against 0.500).

On the deployment-realistic universe — all 2,000 suppliers, matching the model-side N — every
cell collapses to 0.74x–1.21x, because **only 972 of 2,000 suppliers carry any observable
trajectory at all** and a retriever cannot rank a supplier it has no history for.

**A power limitation bounds how strongly this null can be stated.** Under the 40/20/40 temporal
split only **9 snapshot observations precede the first test `t0`**, so the correlations rest on
9–14 points per supplier pair. That is thin, and it is a property of the 15-snapshot mid-scale
schedule rather than of the mechanism — the same class of limit §2.2 identified when it found
the v1 preset could not support the co-degradation check at all. A denser observable history
would be the first thing to change before calling this settled.

### 9.8 Where co-membership lives in the stack — nowhere

Stage 1's held-out null admits two readings, and they imply different next steps: either the
40-epoch budget was too short for the contrastive term to find transferable structure, or there
is no transferable structure to find. This probe separates them. A trained plain-T2 model is
frozen and a fresh MLP head is fitted under the same InfoNCE objective on each of `h^0..h^4`, on
`z_impact` (the blend Transformer 2 actually retrieves over), and on the raw supplier feature
row (`ml/probe_depth_encodability.py`, 2 variants x 5 seeds).

**The permuted-label null is what makes the probe mean anything.** A head with enough capacity
memorises any labelling of 2,000 individually distinct inputs, so a low loss proves nothing on
its own. Every depth is therefore probed twice — once on real `HP_GROUPS` membership, once on a
seeded reshuffle that **preserves the group-size distribution exactly** — and only the gap is
evidence.

| depth | InfoNCE (real) | permuted-label null | gap | gap sign consistency |
|---|---|---|---|---|
| raw features | 4.686 | 4.951 | +0.265 | 8/10 |
| h^0 | 4.818 | 5.062 | +0.244 | 8/10 |
| h^1 | 2.853 | 2.797 | −0.056 | 5/10 |
| h^2 | 1.945 | 2.046 | +0.101 | 7/10 |
| h^3 | 2.142 | 2.065 | −0.077 | 5/10 |
| h^4 | 4.040 | 4.185 | +0.145 | 6/10 |
| **z_impact** (what retrieval ranks) | 4.129 | 4.107 | **−0.022** | 5/10 |

*(chance = ln(N−1) = 7.600; lower InfoNCE = more encodable.)*

**Read the absolute column and you get a tempting, wrong answer.** InfoNCE falls from 4.8 at
`h^0` to **1.9 at `h^2`**, then rises back to 4.0–4.1 at `h^4` and `z_impact` — which looks
exactly like "co-membership is present in the middle of the encoder and destroyed by the last
round of message passing," and would imply an easy fix: retrieve at a shallower depth. **The
permuted null kills that reading.** The shuffled-group curve tracks the real one almost exactly
at every depth (2.046 against 1.945 at `h^2`; 2.065 against 2.142 at `h^3`), so the entire
depth-to-depth variation is variation in how *memorisable* each representation is — how finely
it distinguishes individual suppliers — and not in how much group structure it holds. The gaps
run from −0.077 to +0.265 against a chance level of 7.600, and their sign is consistent on only
5–8 of 10 runs. Held-out Recall@64 at every depth sits at 0.027–0.034 against a 0.032 baseline.

**At no depth of SHARE's stack — including the raw input features — does co-membership separate
from a size-matched random regrouping.** This is the finding that makes Stage 1's held-out zero
and Stage 2's null the same result rather than two coincidences: two suppliers sharing a hidden
parent are distinguished from two that do not by *nothing the model can see*, only by their
individual identities. A contrastive loss can memorise those identities, which is what Stage 1
did; it cannot generalise from them, because there is no regularity to generalise.

This also answers the Stage 1 hard stop's own wording more precisely than the stop itself does.
The brief framed a Stage 1 failure as meaning *"SHARE's Supplier representation lacks the
structural information this task needs regardless of how retrieval is trained."* Stage 1 did not
fail, so that stop was not triggered — but the probe shows the claim is nonetheless true in the
sense that matters. The representation does not lack the *capacity* to encode co-membership; it
lacks any *cue* from which co-membership could be inferred. Retrieval training is not the
binding constraint, and no amount of it will become one.

### 9.9 Which stop condition was hit, and what it settles

**The third one.** Stage 2 does not clear chance, so per the brief: *report a clean, honest
null — the signal is teachable under privileged supervision but not recoverable from observable
data alone at this label volume — and do not build Stage 3, 4 or 5.* None of them was built.
Stage 3 (Relationship Attribution) and Stage 4 (Confidence-Aware Fusion on the new retriever)
have nothing to sit on: a retriever at chance produces pools with no co-members in them, and
§5 already showed the trust gate cannot learn to trust a pool that contains nothing. Stage 5
(Multi-View Retrieval) is not motivated either, since the second view Stage 2 would have
supplied is the one that came back null. **Stage 6 (Causal Retrieval Network) remains blocked**,
unchanged from the audit's position: it collapses into Stage 1 or Stage 2 as described and needs
a concrete causal-inference mechanism — temporal precedence, intervention sensitivity,
counterfactual consistency, or Granger-style influence — defined before any code is written.
Nothing here changes that, and nothing here was built toward it.

**What this settles.** §7 offered two readings of §2's null: retrieval was pointed at the wrong
signal, or the representation cannot carry it. This session resolves that, and the answer is
neither of the simple options. The representation *can* be forced to carry co-membership — 
Recall@64 rises to 10.9x chance and MRR to 24.3x — so §7's diagnosis was mechanically correct.
But what it carries under that pressure is a memorised lookup of the specific groups it was
shown, not a transferable notion of co-membership: held-out groups stay at 0.74x–1.50x, 0 of 20
cells clearing the floor, on groups the generator produced by the identical process. The depth
probe (§9.8) shows why that is not a training-budget problem: at **no** depth of SHARE's stack
does co-membership separate from a size-matched random regrouping by more than the probe's own
permutation null. **There is no feature-space or graph-position regularity that distinguishes
two suppliers sharing a hidden parent from two that do not** — only their individual identities,
which a sufficiently determined loss can memorise and which generalise to nothing.

**What this does not settle.** All of it rests on `sup_n=2,000`, 15 snapshots, 5 dataset seeds,
40 epochs, on CPU-class hardware — not the benchmark's own configuration, the same standing
caveat every other result in this project carries. Three specific limits deserve naming. First,
Stage 2's correlations rest on **9–14 observations per supplier**, and **only 972 of 2,000
suppliers have any observable trajectory**; a denser history is the single change most likely to
move that number, and until it is tried the Stage 2 null is a null *at this observation density*,
not in general. Second, the Stage 1 ceiling is a **lower bound** — the contrastive term was
still descending at 40 epochs — so the supervised numbers would rise further, though the
held-out ones would not. Third, this session ran on **MPS**, and its floors were re-measured
there; the retrieval floors above are not comparable to §4's CPU AUC floor and neither is
comparable to what a GPU run would produce. §7's recommendation is unchanged and, if anything,
sharpened: what would resolve this is a **generator change** — hidden-parent groups few enough
and strong enough that co-membership is the dominant explanation for two suppliers' correlated
history — not another retrieval mechanism. This session tested the strongest retrieval mechanism
available, handed it the answer, and it still did not generalise.

### 9.10 Provenance

**50 trainings, all mid-scale (`sup_n=2,000`), Variants B and D, 5 dataset seeds, 40 epochs,
hidden=128, num_bases=10, `top_k=64`, model seed 0.** 40 form the Stage 1 grid — 2 arms x 2
variants x 5 seeds x **2 identically-configured replicates**, the replicates being the
reproduction-floor measurement — and 10 more are the depth probe's control trainings. Stage 2
required no training at all: it is a data-level measurement, run first for exactly that reason,
following the cost discipline that declined to run Variant K last session.

**Device: MPS, serial, one process throughout.** §8 recorded that MPS was not used last session,
per `reports/phase7_training_results.md` §3.1's measurement that it does not parallelise. That
measurement was taken at **spec scale** (4.4M edges) and does not transfer to mid scale, so it
was re-taken: at `sup_n=2,000` MPS runs **6.2 s/epoch against CPU's 9.3**, a 1.5x speedup, with
no allocation failure at hidden=128. It still does not parallelise — two concurrent MPS
processes give 6.7 s/epoch/run, slightly *worse* than serial — so CPU at 2-way concurrency
retains ~20% more total throughput (5.1 s/epoch/run). MPS was used anyway, deliberately: it
gives one identical process environment for all 50 runs, which removes the thread-count
variation §4 identified as the cause of its floor. **That did not remove the floor** (§9.2),
which is itself informative. The two-concurrent memory ceiling from §6 was re-confirmed before
scheduling: 2.4–2.8 GB per process at 2-way.

**All five ported Transformer 2 files remain byte-identical**, re-verified at the close of this
session — `git diff` reports no change to `transformer2.py`,
`rgcn_attn_variant_a_transformer2.py`, `transformer2_confidence.py`, `transformer2_trustgate.py`
or `transformer2_crossattn.py`. The contrastive arm required capturing the live, non-detached
impact embedding, which was done with a **forward hook** on the impact gate rather than by
editing the ported file; the arm is architecturally identical to the plain T2 arm and reports
the same **814,627** parameters, matching §1.1 exactly.

New code: `ml/models/contrastive.py` (InfoNCE + the privileged supervision object and its
supervised/held-out split), `ml/retrieval_metrics.py` (Recall@K, Precision@K, MRR, percentile
rank and discovery rate with measured chance baselines), `ml/run_retrieval_redesign.py` (Stage 1
runner, device-aware, records retrieval at both checkpoints),
`ml/observable_cofailure.py` + `ml/run_stage2_cofailure.py` (Stage 2),
`ml/test_cofailure_leakage.py` (the guardrail), `ml/stage_gate.py` (the stop conditions in
code), `ml/probe_depth_encodability.py` (§9.8). Extended: `ml/train.py` (the optional
contrastive term; `contrastive=None` leaves the objective byte-for-byte what every earlier
result was trained under), `ml/analyze_transformer2.py` (the three IR metrics),
`ml/reproduction_floor.py` (retrieval floors from replicate pairs).

Raw per-run results in `out/t2redesign/` (40 runs, `r1`/`r2` tags), Stage 2 in
`out/stage2_cofailure.json`, the depth probe in `out/depth_probe.json`, gate decisions in
`out/stage1_gate_final.json` and `out/stage1_gate_best.json`. Ground truth from
`out/hidden_mid/`, unchanged. `verify_no_hidden_state()` returns empty on all ten mid-scale
variant-seeds.
