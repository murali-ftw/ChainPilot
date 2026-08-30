# Markov readout depth re-test — `shortage` @ h², `delay` @ h², `delay` @ h³

**Scope.** Re-run of the Markov-blanket depth readout at three alternative depths, against the
frozen Phase 0 baseline (`ml/layer3_baseline.py`, `out/layer3_v3/phase0_baseline.json`). Same
5 dataset seeds × 5 model-init seeds grid (25 cells), variant 0, `db/csv_v1scale/`.

**No retraining. SHARE untouched.** Every number below comes from the 25 checkpoints already in
`out/ds_ckpt/` (`v0_seed4{2..6}_v0_d4{2..6}_m{0..4}_e100.pt`). SHARE emits the full `{h⁰…h⁴}`
for every node in one forward pass; the only thing that changed is which index of that
already-computed list the readout takes. `freeze()` and `assert_backbone_frozen()` are asserted
before *and* after every cell, including the check that no backbone parameter reached an
optimizer.

**Reproduction check.** At each task's retained depth the swap harness must reproduce Phase 0's
`P(Y|X,H)` exactly, or the comparison is not against the frozen baseline. It does:
**50/50 cells, max |diff| = 0.00e+00**. `P(Y|X)` is not recomputed at all — it is read from
Phase 0's own per-cell grid, since features-only does not depend on depth.

Code: [ml/layer3_depth_swap.py](ml/layer3_depth_swap.py) · results:
[out/layer3_v3/phase0_depth_swap.json](out/layer3_v3/phase0_depth_swap.json)

---

## 0. Why there are two arms

"Change which depth index gets read" is only half of the readout. The other half is
`model.heads[task]`, a `PredictionHead` that was trained *jointly with the encoder* against the
retained depth's activations. Swapping the index without touching the head therefore measures
two things at once, and they have to be separated:

| | what is fitted | what the gain measures |
|---|---|---|
| **Arm A** — `frozen_head` | nothing at all | whether the retained depth's head *transfers* to another depth's geometry |
| **Arm B** — `refit_head` | the readout head only, on frozen `h^d` | whether `h^d` *carries the signal* |

Arm A is the literal zero-parameter index-select the brief describes, and is reported first.
Arm B hands the frozen `h^d` to `ml/latent_state_head.py::train_head` — the same `in → 32 → 1`
MLP, same positive-weighted BCE, same 120 epochs, same seeding by model-init seed — that
Phase 0's `P(Y|X)` arm already uses. That makes Arm B's two sides share one readout protocol, so
`mean AUC(h^d) − mean AUC(X)` isolates the representation and nothing else. Arm B still trains
no SHARE parameter; it fits a 4k-parameter probe on a frozen cached tensor.

Arm B is validated by the same reproduction check applied loosely: at the retained depths it
lands within **0.0029** (delay h¹: 0.7802 vs 0.7774) and **0.0171** (shortage h³: 0.7699 vs
0.7871) of Phase 0's jointly-trained head, i.e. the probe is not the bottleneck.

---

## 1. Requested table — ARM A (pure index-select, nothing fitted)

25 cells each. FLOOR = max(init-seed floor, dataset-seed floor).

| task | depth | P(Y\|X) | P(Y\|X,H) | graph gain | init floor | dset floor | **FLOOR** | gain > FLOOR | worlds + |
|---|---|---|---|---|---|---|---|---|---|
| **shortage** | **h²** | 0.8727 | 0.5420 | **−0.3307** | 0.5338 | 0.0636 | **0.5338** | **no** | 0/5 |
| **delay** | **h²** | 0.7416 | 0.4886 | **−0.2531** | 0.5053 | 0.2103 | **0.5053** | **no** | 0/5 |
| **delay** | **h³** | 0.7416 | 0.5016 | **−0.2401** | 0.5827 | 0.2061 | **0.5827** | **no** | 0/5 |
| *shortage* | *h³ (retained)* | *0.8727* | *0.7871* | *−0.0856* | *0.0101* | *0.0231* | *0.0231* | *no* | *0/5* |
| *delay* | *h¹ (retained)* | *0.7416* | *0.7774* | *+0.0357* | *0.0140* | *0.0874* | *0.0874* | *no* | *5/5* |

The two italic rows are Phase 0's frozen numbers, reproduced exactly by this harness.

**Arm A's swap rows are not evidence about depth.** All three land at or below chance
(0.4886–0.5420) with init-seed floors of 0.50–0.58 — a head fed a representation it was never
fitted to produces near-random rankings whose seed-to-seed spread swamps everything. That is a
statement about head transfer, not about whether h² or h³ carries `shortage`/`delay` signal.

---

## 2. Same table — ARM B (readout head refit on the frozen `h^d`)

| task | depth | P(Y\|X) | P(Y\|X,H) | graph gain | init floor | dset floor | **FLOOR** | gain > FLOOR | worlds + |
|---|---|---|---|---|---|---|---|---|---|
| **shortage** | **h²** | 0.8727 | 0.7628 | **−0.1098** | 0.1951 | 0.0325 | **0.1951** | **no** | 0/5 |
| **delay** | **h²** | 0.7416 | 0.7662 | **+0.0245** | 0.0782 | 0.0784 | **0.0784** | **no** | 5/5 |
| **delay** | **h³** | 0.7416 | 0.7293 | **−0.0123** | 0.3038 | 0.1722 | **0.3038** | **no** | 1/5 |
| *shortage* | *h³ (retained)* | *0.8727* | *0.7699* | *−0.1027* | *0.0894* | *0.0509* | *0.0894* | *no* | *0/5* |
| *delay* | *h¹ (retained)* | *0.7416* | *0.7802* | *+0.0386* | *0.0115* | *0.0778* | *0.0778* | *no* | *5/5* |

**Not one of the three swaps clears its own FLOOR**, and each is *worse* than its own task's
retained depth under the identical protocol: shortage h² (−0.1098) is more negative than
shortage h³ (−0.1027); delay h² (+0.0245) is smaller than delay h¹ (+0.0386); delay h³ is
negative and sign-inconsistent.

---

## 3. Per-world detail

Per-world `P(Y|X,H)` means (5 model-init seeds averaged within each world) and the per-world
graph gain against that world's own `P(Y|X)`.

`P(Y|X)` per world — delay: 42: 0.7486 · 43: 0.7081 · 44: 0.7791 · 45: 0.7588 · 46: 0.7136 ·
shortage: 42: 0.8790 · 43: 0.8849 · 44: 0.8621 · 45: 0.8660 · 46: 0.8713.

### ARM A

| row | d42 | d43 | d44 | d45 | d46 | worlds + |
|---|---|---|---|---|---|---|
| shortage h² — P(Y\|X,H) | 0.5453 | 0.5531 | 0.5733 | 0.5285 | 0.5098 | |
| shortage h² — gain | −0.3337 | −0.3319 | −0.2887 | −0.3375 | −0.3616 | **0/5** |
| delay h² — P(Y\|X,H) | 0.4928 | 0.4037 | 0.4614 | 0.4710 | 0.6140 | |
| delay h² — gain | −0.2558 | −0.3044 | −0.3177 | −0.2878 | −0.0996 | **0/5** |
| delay h³ — P(Y\|X,H) | 0.4881 | 0.4552 | 0.5120 | 0.4232 | 0.6294 | |
| delay h³ — gain | −0.2605 | −0.2529 | −0.2670 | −0.3356 | −0.0843 | **0/5** |

### ARM B

| row | d42 | d43 | d44 | d45 | d46 | worlds + |
|---|---|---|---|---|---|---|
| shortage h² — P(Y\|X,H) | 0.7601 | 0.7800 | 0.7597 | 0.7667 | 0.7476 | |
| shortage h² — gain | −0.1189 | −0.1049 | −0.1023 | −0.0993 | −0.1238 | **0/5** |
| delay h² — P(Y\|X,H) | 0.7925 | 0.7150 | 0.7923 | 0.7934 | 0.7376 | |
| delay h² — gain | +0.0439 | +0.0069 | +0.0132 | +0.0346 | +0.0240 | **5/5** |
| delay h³ — P(Y\|X,H) | 0.8031 | 0.7005 | 0.7580 | 0.7540 | 0.6309 | |
| delay h³ — gain | +0.0545 | −0.0076 | −0.0211 | −0.0048 | −0.0827 | **1/5** |

Reference rows (retained depths, Arm B): shortage h³ gain −0.0820 / −0.0984 / −0.0934 / −0.1200
/ −0.1200 (**0/5**); delay h¹ gain +0.0674 / +0.0318 / +0.0229 / +0.0461 / +0.0246 (**5/5**).

---

## 4. Verdict — one sentence per row

Applying Phase 0's bar (gain must clear its own FLOOR **and** be sign-consistent across the 5
dataset-seed worlds):

- **shortage @ h²** — Does not overturn: the gain is negative in both arms (Arm A −0.3307, Arm B
  −0.1098), negative in 5/5 worlds, and *more* negative than the retained h³, so shortage remains
  net-negative against features alone and moving it shallower makes it slightly worse, not better.
- **delay @ h²** — Does not overturn: +0.0245 is sign-consistent (5/5 worlds positive) but sits
  below its own FLOOR of 0.0784, and it is smaller than delay's own retained h¹ (+0.0386) under
  the identical protocol, so delay stays unproven and h² is not an improvement on h¹.
- **delay @ h³** — Does not overturn: the gain is negative (−0.0123), positive in only 1/5 worlds,
  and carries the largest FLOOR in the table (0.3038), so it fails both halves of the bar and is
  the worst of the three depths tested for delay.

**Overall: inconclusive, no depth change warranted.** None of the three swaps clears its floor in
either arm, and the sign-consistency test only passes for delay @ h² — which is the one swap whose
point estimate is *below* the depth it would replace. Phase 0's conclusions stand unchanged:
`delay` @ h¹ is a consistent but unproven +0.0357 against a 0.0874 floor, and `shortage` @ h³ is
actively worse than its own features at −0.0856. The retained depths are not the reason those two
tasks fail, so the failure has to be looked for somewhere other than the readout index.

One thing the Arm B rows do add, as a diagnostic rather than a finding: for `delay` the gain
decays monotonically with depth (h¹ +0.0386 → h² +0.0245 → h³ −0.0123). That is the same
direction `ml/models/rgcn_attn_markov_encoder.py`'s docstring gives as the reason delay was
pinned to h¹ over the theory doc's blanket-derived h² in the first place — so the empirical
depth already in the code is the best of the three, it is simply not big enough to clear the
dataset-seed floor. (The preflight report that docstring cites is not present in this
repository, so this is agreement in direction with the recorded rationale, not a re-check of
its numbers.)

---

## 5. Reproducing

```
python3 ml/layer3_depth_swap.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

~90s on CPU, all 25 cells from cache; trains nothing but the Arm B probe heads.
