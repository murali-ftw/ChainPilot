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
