# Layer 2 on V2 — Adaptive Depth Tested Where Chain Length Actually Varies

**Status:** the Rung 5 depth-gate family is ported (byte-identical, parameter counts exact),
verified against V1's Round 4/5 recorded numbers, and run across Variant 0 and Variant J at five
dataset seeds each — 50 training runs.

**The answer, stated first: no.** No gated variant shows a sign-consistent AUC improvement over the
fixed-depth Markov baseline on Variant J that it does not also show on Variant 0. All twelve
delta-of-delta cells flip sign across the five seeds. The point estimates lean the hypothesised way
— 8 of 12 cell means positive, pooled mean **+0.0018 (95% CI −0.0004 to +0.0039)** — but that lean
is not distinguishable from noise (sign test p = 0.39), and the effect it would represent is a
fifth of the seed-to-seed spread of the baseline itself.

**This is the first time the question could be asked at all.** V1's dataset had structurally uniform
chain length, so V1's five Rung-5 variants had nothing to adapt *to* — which is exactly why
`v1_findings/v2.md` item 4 recorded the same negative result five times. V2's Variant J is the first
dataset in this project where chain length genuinely varies per product. **The negative result
survives that change**, which makes it a stronger finding than V1's was: it is no longer explicable
as an artifact of flat topology.

---

## 1. Reconciling the depth prior — the two priors are both real and different

The step brief flagged an apparent contradiction: `reports/phase7_training_results.md` describes the
fixed prior as delay→h², shortage→h³, impact→h³, while `HADES_v1/reports/layer2.md` records
delay→h¹, shortage→h³, impact→h⁴. **Both are correct. They are two different priors serving two
different purposes**, and the Rung-5 family uses the second one.

| | value | who reads it | source |
|---|---|---|---|
| `STRUCTURAL_DEPTH_PRIOR` | delay→h², shortage→h³, impact→h³ | SHARE, SHARP, SHARK and every baseline | V1 Steps 3–5, `ml/models/depth.py` |
| `MARKOV_READOUT_DEPTH` | **delay→h¹, shortage→h³, impact→h⁴** | Markov, Rung 5, Variants A/B/C | V1 Step 6 Round 3, `ml/models/rgcn_attn_markov_encoder.py` |

The Markov prior is the *empirically corrected* one: V1's Round 1 measured delay's h¹ readout
beating baseline (+0.0069, 5/5 seeds) while the theory doc's blanket-derived h² showed no effect,
and Round 2's learned gate converged on h³/h⁴ for shortage/impact. The port therefore initialises
every Rung-5 gate from `MARKOV_READOUT_DEPTH`, taken directly from V1's own module rather than
re-derived — so the gate starts exactly where V1's did, which the brief correctly identified as a
precondition for the comparison meaning anything.

A second structural difference follows from this and is worth stating: the Rung-5 family runs on
`RGCNAttnDepthGateEncoder`, which prepends **h⁰** (the raw `lin_in` projection, zero hops) to the
layer list. The readout therefore chooses among **five** depths (h⁰…h⁴), not four. Every other
architecture in this project sees only h¹…h⁴.

## 2. What was ported

| file | md5 vs V1 | role |
|---|---|---|
| `ml/models/rung_gate_common.py` | **identical** | `init_near_onehot_bias`, shared by every gate |
| `ml/models/rgcn_attn_depthgate_encoder.py` | **identical** | the h⁰…h⁴ encoder all five arms share |
| `ml/models/rgcn_attn_markov_encoder.py` | **identical** | fixed-depth baseline + `MARKOV_READOUT_DEPTH` |
| `ml/models/rgcn_attn_rung5_encoder.py` | **identical** | base Rung 5, per-node MLP gate |
| `ml/models/rgcn_attn_rung5_variant_a.py` | **identical** | bounded residual (the production choice) |
| `ml/models/rgcn_attn_rung5_variant_b.py` | **identical** | structural features into the gate |

Rung 4 was **not** ported (V1 ruled it out: 15× the cost, no benefit), nor Variants D or E (no
improvement; catastrophic failure respectively) — porting any of them would re-spend effort V1
already spent and answered.

**Variant C is not a model file**, confirmed against V1's `ml/run_rung5_variant_pilot.py`: the
architecture id `rgcn_attn_rung5_c` instantiates the plain `Rung5HADESModel` and applies a
training-loop change — gate frozen for the first 15 epochs, then unfrozen at 0.1× the main learning
rate. That is implemented in `ml/train.py` as a single AdamW with two parameter groups, matching
V1's own approach (a frozen parameter produces no gradient, so the optimizer skips it for free).

### Parameter counts — every one matches V1 exactly

| arm | params | vs SHARE | V1's documented value |
|---|---:|---|---|
| Markov | 752,211 | **+0** | +0 — "zero new trainable parameters", a fixed index-select |
| Rung 5 | 765,105 | +12,894 | +12,894 (4,298/task × 3: 128·32+32, 32·5+5, +5 bias) |
| Variant A | 765,090 | +12,879 | +12,894 − 15 — `position_bias` becomes a fixed buffer, so 5 params/task stop being learned |
| Variant B | 766,289 | +14,078 | wider gate input (degree + node-type one-hot + per-relation histogram) |
| Variant C | 765,105 | +12,894 | identical to Rung 5 — it is a schedule, not an architecture |

Variant A being **15 parameters smaller** than base Rung 5 is the port reproducing the design
exactly: its whole mechanism is pinning the position bias and bounding the residual.

## 3. Sanity gate — the port reproduces V1

Variant 0 at the `v1` preset (the byte-identical-to-V1 anchor), 5 model-init seeds, 100 epochs,
hidden=128 / num_bases=10 — V1's Step 6 configuration throughout.

| arm | task | ported (5 seeds) | V1 recorded (Round 5, ±1 std) | delta |
|---|---|---|---|---|
| **Markov** | delay | 0.8159 ± 0.0030 | 0.8145–0.8199 | −0.0013 |
| | shortage | 0.7996 ± 0.0020 | 0.7954–0.7986 | +0.0026 |
| | impact | 0.9443 ± 0.0058 | 0.9368–0.9430 | +0.0044 |
| **Rung 5** | delay | 0.8156 ± 0.0029 | 0.8151–0.8189 | −0.0014 |
| | shortage | 0.8001 ± 0.0015 | 0.7758–0.8050 | +0.0097 |
| | impact | 0.9456 ± 0.0036 | 0.9436–0.9464 | +0.0006 |
| **Variant A** | delay | 0.8128 ± 0.0026 | 0.8160–0.8216 | −0.0060 |
| | shortage | 0.7993 ± 0.0031 | 0.7963–0.7997 | +0.0013 |
| | impact | 0.9436 ± 0.0034 | 0.9429–0.9465 | −0.0011 |

**All nine cells land within ±0.02 of V1's recorded band, and all three parameter counts match to
the digit.** The largest gap is Variant A's delay at −0.0060, about 2× V1's own recorded std for
that cell — worth noting rather than hiding, but well inside the tolerance and in a task whose
dataset-seed spread (§5) is 0.038.

The gate passed before anything was run on V2, as the brief required.

## 4. Cost — the gate is not free, and the estimate was re-checked

The brief estimated ~419 s/run by analogy with SHARE and asked for confirmation on the first 2–3
runs before committing. Measured:

| arm | per run | vs Markov |
|---|---|---|
| Markov | 396–456 s | — |
| Rung 5 / A / B / C | 485–548 s | **+22%** |

So 50 runs is ~1.8 h of wall clock at 4 workers × 3 threads rather than the estimated 1.5 h — the
gate machinery is small in parameters but adds a real per-forward-pass cost (a stack of five depth
tokens, a mean-pool, a two-layer MLP and a softmax, per node, per task). The overrun was accepted
and the sweep committed. Over the whole sweep, 46 of 50 runs took **436–803 s, mean 515 s**.

**Operational note.** The other **four** runs took **6,266 / 6,298 / 6,633 / 6,699 s** — a ~13×
stall. That they are all within 7% of each other, one per worker, is the diagnostic: this was a
single system-wide event of roughly 100 minutes that hit all four processes at once, not four
independent incidents and not a code path in any particular arm. All 100 epochs completed in every
case, and training is deterministic given the seed and config, so the results are unaffected — the
cost was ~100 minutes of wall clock, and total sweep time was 3.9 h instead of 1.8 h.

## 5. Per-variant results (5 dataset seeds, positives beside every AUC)

| variant | arm | params | delay AUC | pos | shortage AUC | pos | impact AUC | pos |
|---|---|---:|---|---|---|---|---|---|
| **0** | Markov | 752,211 | 0.7798 ± 0.0376 | 519 | 0.7872 ± 0.0084 | 1,036 | 0.9354 ± 0.0101 | 156 |
| 0 | Rung 5 | 765,105 | 0.7734 ± 0.0430 | 519 | 0.7857 ± 0.0066 | 1,036 | 0.9356 ± 0.0130 | 156 |
| 0 | Variant A | 765,090 | 0.7742 ± 0.0431 | 519 | 0.7879 ± 0.0097 | 1,036 | 0.9354 ± 0.0105 | 156 |
| 0 | Variant B | 766,289 | 0.7752 ± 0.0355 | 519 | 0.7820 ± 0.0101 | 1,036 | 0.9301 ± 0.0141 | 156 |
| 0 | Variant C | 765,105 | 0.7771 ± 0.0419 | 519 | 0.7879 ± 0.0094 | 1,036 | 0.9297 ± 0.0080 | 156 |
| **J** | Markov | 752,221 | 0.7690 ± 0.0305 | 628 | 0.7823 ± 0.0050 | 1,077 | 0.9409 ± 0.0063 | 203 |
| J | Rung 5 | 765,115 | 0.7674 ± 0.0313 | 628 | 0.7824 ± 0.0049 | 1,077 | 0.9371 ± 0.0046 | 203 |
| J | Variant A | 765,100 | 0.7677 ± 0.0326 | 628 | 0.7800 ± 0.0063 | 1,077 | 0.9404 ± 0.0052 | 203 |
| J | Variant B | 766,331 | 0.7622 ± 0.0323 | 628 | 0.7831 ± 0.0055 | 1,077 | 0.9393 ± 0.0096 | 203 |
| J | Variant C | 765,115 | 0.7713 ± 0.0300 | 628 | 0.7847 ± 0.0036 | 1,077 | 0.9389 ± 0.0059 | 203 |

Variants A/J's arms carry ~10 extra parameters relative to Variant 0's because Mechanism J's
`UPSTREAM_OF` relation adds two meta-relations, and the RGCN family holds one basis-coefficient row
per relation — the same disclosure `reports/phase7_training_results.md` §1 makes.

## 6. Each gated arm against the fixed-depth Markov baseline

Per-seed deltas against Markov on the same dataset seed; "sign" is how many of the five agree on
direction; "sig" counts seeds whose paired block bootstrap over test snapshots excludes zero.

| variant | arm | task | mean delta | sign | seeds sig |
|---|---|---|---|---|---|
| 0 | Rung 5 | delay | **−0.0064 ± 0.0058** | 0+/5− **consistent** | 2/5 |
| 0 | Rung 5 | shortage | −0.0015 ± 0.0030 | 2+/3− flips | 1/5 |
| 0 | Rung 5 | impact | +0.0002 ± 0.0038 | 2+/3− flips | 1/5 |
| 0 | Variant A | delay | −0.0056 ± 0.0072 | 1+/4− flips | 2/5 |
| 0 | Variant A | shortage | +0.0007 ± 0.0044 | 3+/2− flips | 2/5 |
| 0 | Variant A | impact | −0.0000 ± 0.0012 | 3+/2− flips | 0/5 |
| 0 | Variant B | delay | −0.0045 ± 0.0061 | 2+/3− flips | 1/5 |
| 0 | Variant B | shortage | **−0.0052 ± 0.0046** | 0+/5− **consistent** | 1/5 |
| 0 | Variant B | impact | −0.0053 ± 0.0068 | 1+/4− flips | 1/5 |
| 0 | Variant C | delay | −0.0027 ± 0.0052 | 2+/3− flips | 0/5 |
| 0 | Variant C | shortage | +0.0007 ± 0.0043 | 3+/2− flips | 2/5 |
| 0 | Variant C | impact | **−0.0058 ± 0.0052** | 0+/5− **consistent** | 2/5 |
| J | Rung 5 | delay | −0.0016 ± 0.0085 | 3+/2− flips | 3/5 |
| J | Rung 5 | shortage | +0.0001 ± 0.0061 | 4+/1− flips | 3/5 |
| J | Rung 5 | impact | −0.0038 ± 0.0065 | 1+/4− flips | 2/5 |
| J | Variant A | delay | −0.0013 ± 0.0082 | 3+/2− flips | 2/5 |
| J | Variant A | shortage | −0.0023 ± 0.0084 | 3+/2− flips | 2/5 |
| J | Variant A | impact | −0.0005 ± 0.0070 | 2+/3− flips | 2/5 |
| J | Variant B | delay | −0.0068 ± 0.0064 | 1+/4− flips | 3/5 |
| J | Variant B | shortage | +0.0008 ± 0.0043 | 3+/2− flips | 2/5 |
| J | Variant B | impact | −0.0015 ± 0.0099 | 3+/2− flips | 2/5 |
| J | Variant C | delay | +0.0023 ± 0.0058 | 3+/2− flips | 1/5 |
| J | Variant C | shortage | +0.0024 ± 0.0038 | 4+/1− flips | 2/5 |
| J | Variant C | impact | −0.0019 ± 0.0066 | 2+/3− flips | 1/5 |

**Three observations.**

1. **Every sign-consistent effect is a gated arm losing to Markov, and all three are on Variant 0**
   — base Rung 5 on delay (−0.0064), Variant B on shortage (−0.0052), Variant C on impact
   (−0.0058). Nothing gated beats fixed depth consistently anywhere.
2. **On Variant J, not one of the twelve cells is sign-consistent.** Where chain length varies, the
   gates are indistinguishable from the fixed prior in both directions.
3. The "seeds sig" column is higher on Variant J (up to 3/5) than on Variant 0 while sign agreement
   is *lower*. That combination — individually significant per-seed differences that disagree on
   direction — is the signature of real per-world differences that do not generalise, not of a
   systematic effect.

## 7. The delta-of-deltas — this session's actual question

**Does any gated variant improve on Markov *more* on Variant J than on Variant 0?** Per seed:
`(arm − Markov on J) − (arm − Markov on 0)`. Positive means the gate helps more where chain length
genuinely varies.

| arm | task | delta-of-deltas | sign | on J | on 0 |
|---|---|---|---|---|---|
| Rung 5 | delay | +0.0047 ± 0.0078 | 3+/2− flips | −0.0016 | −0.0064 |
| Rung 5 | shortage | +0.0016 ± 0.0067 | 3+/2− flips | +0.0001 | −0.0015 |
| Rung 5 | impact | −0.0040 ± 0.0089 | 2+/3− flips | −0.0038 | +0.0002 |
| Variant A | delay | +0.0043 ± 0.0116 | 4+/1− flips | −0.0013 | −0.0056 |
| Variant A | shortage | −0.0029 ± 0.0086 | 3+/2− flips | −0.0023 | +0.0007 |
| Variant A | impact | −0.0005 ± 0.0073 | 2+/3− flips | −0.0005 | −0.0000 |
| Variant B | delay | −0.0023 ± 0.0073 | 2+/3− flips | −0.0068 | −0.0045 |
| Variant B | shortage | +0.0060 ± 0.0060 | 4+/1− flips | +0.0008 | −0.0052 |
| Variant B | impact | +0.0038 ± 0.0146 | 4+/1− flips | −0.0015 | −0.0053 |
| Variant C | delay | +0.0050 ± 0.0094 | 2+/3− flips | +0.0023 | −0.0027 |
| Variant C | shortage | +0.0017 ± 0.0024 | 4+/1− flips | +0.0024 | +0.0007 |
| Variant C | impact | +0.0038 ± 0.0079 | 4+/1− flips | −0.0019 | −0.0058 |

**All twelve cells flip sign across the five seeds. The answer is no.**

### The honest reading of the lean

The point estimates are not symmetric about zero, and pretending otherwise would be its own kind of
dishonesty:

- **8 of 12** cell means are positive (sign test **p = 0.39** — what you would expect from chance).
- Pooling all 60 per-seed values: mean **+0.0018**, sd 0.0085, **95% CI [−0.0004, +0.0039]**. The
  interval includes zero. (Cells share the same runs, so this pooling is indicative, not a valid
  independent test.)
- Four cells reach 4+/1− sign agreement — Variant A delay, Variant B shortage and impact, Variant C
  shortage and impact — but four-of-five is what a coin produces about a third of the time across
  twelve tries.

For scale: the largest delta-of-deltas here is +0.0060, while the **Markov baseline's own
dataset-seed spread on delay is 0.0376** — six times larger. Any adaptive-depth benefit on Variant J
is, at this label volume, at most a fifth of the noise the benchmark generates by changing which
world it generates.

**The direction is consistent with the hypothesis and the magnitude is not.** If adaptive depth
helps under chain-length heterogeneity, this experiment says the effect is smaller than 0.004 AUC —
which would need roughly an order of magnitude more label volume to resolve.

## 8. What this means for the two projects

**For V1's negative result:** it replicates on a dataset that has the property V1's lacked. V1's
five variants all landed on the same null for shortage and impact, and the standing explanation was
that V1's flat topology gave the gate nothing to adapt to. That explanation is now **testable and
falsified as a complete account** — Variant J supplies genuine per-product chain-length variation
and the null holds anyway. The negative result is about the gate design, or about label volume, not
about the dataset being too uniform.

**For V2's benchmark:** Mechanism J moves task difficulty (Phase 7 measured delay −0.0081, 5/5
consistent) without creating headroom that a per-node depth gate can convert into accuracy. Those
are different claims and both are now measured.

**For production:** nothing changes. The fixed Markov readout remains the right default — it is free
(zero new parameters), it is never consistently beaten here, and it consistently *beats* three of
the gated arms on Variant 0. V1 chose Variant A on stability grounds, and nothing in this session
argues against that choice on AUC grounds; it simply finds no accuracy reason to prefer any gate.

## 9. Limitations, stated plainly

- **Not spec scale.** V1-comparable configuration (`sup_n=800`, 15 snapshots), 519–1,077 test
  positives per cell against the spec's 2,000–5,000 target. Every null here is a **non-detection at
  this label volume**, not an established absence — the same caveat every result in
  `reports/phase7_training_results.md` carries.
- **Gate stability was not measured.** V1's Round 4/5 headline was about *stability* — match rate
  against the prior, and the CONSISTENT negative degree-correlation of deviating nodes. This session
  measured AUC only, because the session's question is an AUC question. The models do stash
  `_last_gate_weights`, but `ml/run_benchmark_eval.py` does not persist them, so match rates cannot
  be recovered from the saved runs without retraining. **Whether Variant A's perfect stability
  result reproduces on V2 is open**, and it is the obvious next thing to measure — especially since
  a gate that never moves off the prior is a mechanical explanation for why every delta here is
  near zero.
- **The stretch goal was not run.** Dataset Variant A (`J + truncation`, reported against J) would
  test whether truncating the chain changes the picture. It is 25 further runs (~1 h) and was left
  out; the main comparison is complete without it.
- **Single lambda for model Variant A.** `lambda_bound = 0.3`, V1's default and mid-range choice.
  V1 found stability identical at 0.1/0.3/0.5 and AUC differences within seed noise, so the sweep
  was not repeated across lambdas.
- **Two "Variant A"s exist** and are disambiguated throughout: *model* Variant A is the
  bounded-residual gate; *dataset* Variant A is `J + truncation`. Every table above is labelled with
  which is meant.

## 10. Reproducing

```bash
# the gate — must pass before anything runs on V2
python3 ml/run_benchmark_eval.py sanity --archs rgcn_attn_markov --seeds 0,1,2,3,4 \
    --epochs 100 --out out/rung5_sanity_markov.json

# the sweep: 5 arms x {Variant 0, Variant J} x 5 dataset seeds
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python3 ml/run_benchmark_eval.py sweep \
    --csv-dir db/csv_v1scale --variants 0 \
    --archs rgcn_attn_markov,rgcn_attn_rung5,rgcn_attn_rung5_a,rgcn_attn_rung5_b,rgcn_attn_rung5_c \
    --seeds 42,43,44,45,46 --epochs 100 --out out/rung5_v0.json
```

Four worker processes at three threads each; 50 runs, ~1.8 h plus the one stalled run in §4. Raw
results (including per-run test predictions) in `out/rung5_v*.json`, gitignored.

---

# Consolidated Findings — Why Adaptive Depth Doesn't Help (or: What Would Need to Be True)

**80 additional training runs** across three phases: a 25-run gate diagnostic that persists what
the gate actually does per node, a 50-run lambda sweep and A+C combination, and a 5-run supplement
that makes the resilience question answerable at all. This section replaces guessing at *why* §7's
null happened with a measurement of it.

**The one-line answer: the gate never disagrees with the fixed prior, and when it is structurally
forced to be able to disagree, it disagrees on 80–90% of nodes and the AUC does not improve.**
Both halves are now measured, not inferred.

## C1. The original null (§5–§7), restated

No gated variant beat the fixed-depth Markov baseline. All twelve delta-of-delta cells on Variant J
flip sign across five seeds; the pooled lean is +0.0018 with a 95% CI of [−0.0004, +0.0039], against
a baseline whose own seed-to-seed spread on delay is 0.0376.

## C2. The diagnostic — all four questions, in order

### Q1. How often does the gate disagree with Markov?

**Variant A: never. Not once, in any run.**

| arm | variant | delay match | shortage match | impact match |
|---|---|---|---|---|
| **Variant A** (λ=0.3) | 0 | **1.0000** | **1.0000** | **1.0000** |
| Variant A | J | **1.0000** | **1.0000** | **1.0000** |
| Variant A | K | **1.0000** | **1.0000** | **1.0000** |
| Rung 5 (base) | 0 | 0.0984 | 0.6426 | 0.9835 |
| Rung 5 (base) | J | 0.1258 | 0.2958 | 0.9986 |
| Rung 5 (base) | K | 0.7962 | 0.2170 | 0.9990 |

Zero deviating nodes out of **195,029** scored delay rows, on every seed, every variant, every task.
This reproduces V1's Round 5 result (Variant A: 1.000 match, zero deviating nodes) exactly, on a
different dataset generator.

**This is a cause, not an absence of evidence.** Variant A's predictions are Markov's predictions,
node for node. §7's Variant-A-vs-Markov deltas were never measuring adaptive depth; they were
measuring two runs of the same readout under different random initialisations. Any AUC difference
there is training noise by construction.

**And it could not have been otherwise, for arithmetic reasons.** Variant A's logits are
`position_bias + λ·tanh(MLP)`, where `position_bias` is a **fixed buffer** of 8.0 at the prior depth
and 0 elsewhere, and `tanh ∈ [−1, 1]`. For a non-prior depth to win the argmax the residual must
close a gap of 8.0, and the widest gap it can create is 2λ. **The argmax cannot move unless λ > 4.0.**
V1 tested λ ∈ {0.1, 0.3, 0.5} and this project tested 0.3 — every value ever run on this
architecture, in either project, is between 8× and 40× too small to permit a single deviation. V1's
"Variant A completely solved the instability problem" is true and is a restatement of this
inequality.

### Q2. When it disagrees, which nodes are affected?

For Variant A: no nodes, so the question is vacuous — stated as an answer, not a gap.

For base Rung 5, which does move, the answer is the surprising one: **the deviating nodes almost all
choose the same single depth**, and which depth it is changes with the seed.

| run | deviating rows | chosen-depth histogram (h⁰…h⁴) |
|---|---|---|
| v0 seed 42 delay | 154,080 | `[0, 0, 0, 154080, 0]` |
| v0 seed 43 delay | 175,353 | `[175353, 0, 0, 0, 0]` |
| v0 seed 44 delay | 179,670 | `[0, 0, 0, 0, 179670]` |
| v0 seed 45 delay | 179,170 | `[0, 0, 179170, 0, 0]` |
| vJ seed 42 delay | 158,313 | `[158313, 0, 0, 0, 0]` |
| vJ seed 43 delay | 174,469 | `[0, 0, 174469, 0, 0]` |

**A per-node gate is behaving as a per-task constant.** It is not learning which nodes want which
depth; it is re-picking one global depth for the whole task, and the pick is seed-dependent — h³ on
one seed, h⁰ on the next, h⁴ on the next. That is a complete mechanical account of §6's
sign-flipping deltas: each seed is effectively a different fixed-depth model.

Confirmed directly by the across-node dispersion of the gate distribution:

| arm | across-node std of depth weights | mean max weight |
|---|---|---|
| Variant A | **0.00006 – 0.00029** | 0.998 |
| Rung 5 (base) | 0.012 – 0.142 | 0.82 – 0.996 |

Variant A gives literally every node the same distribution. Base Rung 5 varies a little but stays
near one-hot — a shifted global choice, not a per-node policy. **Neither arm has ever exercised the
per-node adaptivity that is the entire point of Rung 5's design.**

### Q3. Do the disagreeing nodes share properties?

Only base Rung 5 has a disagreement set to characterise. Across every testable cell, each property
scored by its univariate AUC for predicting deviation, against a 200-permutation null:

| property | task | cells | mean AUC | range | outside null |
|---|---|---|---|---|---|
| degree | delay | 33 | 0.496 | 0.032–0.858 | 25/33 |
| degree | shortage | 35 | 0.482 | 0.452–0.516 | 17/35 |
| degree | impact | 12 | 0.625 | 0.390–0.930 | 7/12 |
| chain length | delay | 18 | 0.509 | 0.470–0.565 | 7/18 |
| chain length | impact | 8 | 0.510 | 0.030–0.774 | 4/8 |
| **true hidden resilience** | delay | 4 | 0.519 | 0.434–0.632 | 3/4 |
| **true hidden resilience** | impact | 1 | 0.636 | — | 0/1 |

**Nothing predicts deviation.** Mean AUCs sit at 0.48–0.51 for the two properties with real sample
sizes; the cells that fall outside their null do so in *both* directions (delay degree ranges 0.032
to 0.858 — i.e. strongly negative on one seed and strongly positive on another), which is the
signature of a null with tight permutation bands and many comparisons, not of a real association.
The nulls are tight precisely because the disagreement sets are enormous (85% of nodes), so a
0.51 AUC clears them while meaning nothing.

**V1's finding does not reproduce.** V1 reported deviating nodes being consistently *lower*-degree
(−0.23 to −7.70, CONSISTENT across B/C/D). On V2 the degree association is centred on chance and
flips sign across seeds. That V1 pattern was measured on a gate that was also making a global
depth switch, so it may always have been an artifact of which side of the switch a node landed.

**A design gap worth recording.** The two variants this diagnostic was specified to run on —
0 and J — **contain no Mechanism E**, so hidden resilience does not exist in either world
(`RESILIENCE` is empty; `resilience_of` returns 0.0 for every supplier). The resilience question was
unanswerable as scoped. It was made answerable by adding base Rung 5 on **Variant K**, which carries
E, F and J together — that is where the four true-resilience cells above come from.

**The observable resilience proxy was deliberately not built.** True hidden resilience is the upper
bound on what any observable proxy of it could detect, and the true value shows no association with
deviation (0.519 on the only task with more than one cell). A proxy that recovers resilience at
AUC ~0.6 cannot find a pattern that the exact value does not contain. Building it would have
produced a number, not an answer.

### Q4. Are the depth representations different enough to matter?

**Yes — emphatically, and this is the branch that says the null is *not* a representational-collapse
artifact.** Mean cosine similarity between depth pairs, before any gate selects among them:

| node type | h⁰-h¹ | h¹-h² | h²-h³ | h³-h⁴ | **h⁰-h⁴** |
|---|---|---|---|---|---|
| Shipment (delay) | 0.06–0.19 | 0.22–0.25 | 0.16–0.28 | 0.25–0.35 | **0.01–0.14** |
| Product (shortage) | −0.05–0.07 | 0.05–0.19 | 0.13–0.20 | 0.17–0.39 | **−0.06–0.04** |
| Supplier (impact) | −0.02–0.05 | 0.12–0.24 | 0.28–0.33 | 0.36–0.43 | **0.00–0.04** |
| Component | −0.03–0.03 | 0.27–0.38 | 0.22–0.40 | 0.32–0.39 | **−0.10–−0.02** |

Consecutive depths share only 0.05–0.43 of their direction, and **h⁰ and h⁴ are essentially
orthogonal (≈ 0)**. There is a great deal for a depth mechanism to choose between. The null in §7 is
therefore *not* explained by "all the depths look the same" — the mechanism had real, richly
different options and did not benefit from choosing among them.

**Q4b — does chain length create more depth separation?** No. Correlation between Mechanism J's
chain length and Supplier h⁰-h⁴ similarity ranges **−0.044 to +0.055 across seeds, flipping sign**;
bucketed means differ by less than 0.02 between short (≤2), mid (3–4) and long (≥5) chains. **The
heterogeneity Mechanism J introduces does not produce the representational variation an adaptive
depth mechanism would need to exploit** — which is the deepest reason the delta-of-deltas in §7 came
back null, and it is a property of the dataset, not of any gate.

**`v1_findings/v2.md` item 8, checked in passing.** V1 flagged Shipment nodes as anti-smoothing —
similarity *falling* with depth, opposite to every other node type, never root-caused. **It does not
reproduce on V2**: Shipment similarity rises with depth (0.06→0.24→0.25→0.29), the same monotone
pattern as Supplier, Product and Component. One paragraph as scoped; not investigated further.

## C3. The lambda sweep — cap-bound, then free, and still no benefit

Because Q1 established the argmax cannot move below λ=4.0, the grid was re-scoped: testing 0.1/0.3/
0.6/1.0 would have spent 40 runs confirming an inequality. λ=1.0 was kept as an empirical check of
the prediction, and **λ=6.0 and λ=10.0 were added as the first values in either project's history
that can actually let the gate move.** `match` is share of nodes still at the prior; `Δ vs Markov`
is the per-seed paired AUC difference.

| arm | var | delay match | delay Δ | shortage match | shortage Δ | impact match | impact Δ |
|---|---|---|---|---|---|---|---|
| λ=0.3 | 0 | 1.000 | −0.0076 | 1.000 | −0.0005 | 1.000 | −0.0004 |
| λ=1.0 | 0 | 1.000 | −0.0016 | 1.000 | −0.0012 | 1.000 | +0.0002 |
| anneal 0.1→0.6 | 0 | 1.000 | −0.0014 | 1.000 | −0.0011 | 1.000 | −0.0037 |
| **λ=6.0** | 0 | **0.193** | −0.0036 **(0+/5−)** | **0.194** | −0.0044 | 0.999 | −0.0020 |
| **λ=10.0** | 0 | **0.130** | −0.0039 | **0.159** | −0.0026 | 1.000 | −0.0050 |
| λ=0.3 | J | 1.000 | −0.0030 | 1.000 | +0.0023 | 1.000 | +0.0011 |
| λ=1.0 | J | 1.000 | −0.0020 | 1.000 | +0.0015 | 1.000 | +0.0006 |
| anneal 0.1→0.6 | J | 1.000 | −0.0008 | 1.000 | +0.0018 | 1.000 | −0.0014 |
| **λ=6.0** | J | **0.188** | −0.0028 | **0.097** | −0.0002 | 0.991 | −0.0060 |
| **λ=10.0** | J | **0.331** | −0.0058 | **0.104** | +0.0011 | 0.989 | −0.0027 |

**Both halves move exactly as the arithmetic predicts, and they answer the prompt's three-way
question decisively.**

1. **The cap was binding.** Disagreement is exactly 0% at every λ ≤ 1.0 and at the annealed
   schedule, then jumps to **81–90% of nodes** the moment λ crosses 4.0. The gate was not sitting at
   the prior because it had converged there; it was pinned there by a structural inequality.
2. **Releasing it changes nothing good.** Once free, not one AUC delta turns positive and
   sign-consistent. The only 5-seed-consistent effect in the whole sweep is **λ=6.0 on Variant 0's
   delay task at −0.0036 — consistently *worse* than fixed depth.**
3. So of the three possibilities the brief named — cap-bound with no benefit, cap-bound and unable
   to move, free to move but no benefit — the answer is the **third**, and we can now say so because
   we produced the free-to-move condition rather than assuming it.

**One asymmetry worth flagging: impact never moves, at any λ.** Its match rate stays 0.989–1.000
even at λ=10 while delay and shortage collapse to ~0.10. Impact's prior depth is h⁴, the final
layer; the gate is free to leave it and declines to on every seed. The one task whose prior is the
deepest available read is the one the gate agrees with unprompted.

## C4. Variant A + C combined

| arm | var | delay match | delay Δ | shortage match | shortage Δ | impact match | impact Δ |
|---|---|---|---|---|---|---|---|
| A+C | 0 | 1.000 | −0.0066 | 1.000 | +0.0001 | 1.000 | −0.0009 |
| A+C | J | 1.000 | −0.0006 | 1.000 | +0.0022 | 1.000 | −0.0010 |

**It further suppresses movement, which was already zero.** Variant A alone gives 100% match;
adding Variant C's freeze-then-low-LR schedule gives 100% match. Every AUC delta is inside noise
with mixed signs. Combining two stability fixes on a mechanism that already never moves produces a
mechanism that still never moves — worth having measured, and worth not repeating.

## C5. The answer, for someone reading only this paragraph

**Is there any remaining reason to believe a per-node adaptive depth mechanism could help this
benchmark's three tasks? On the evidence, no — and the question is now closed at this scale for this
design space.** Across two projects, eight gate variants and roughly 130 training runs, the
mechanism has never produced a sign-consistent AUC gain on any task. This session explains why, and
the explanation is not "we didn't measure carefully enough": the production gate (Variant A) is
*mathematically incapable* of departing from the fixed prior at any λ ever used, so every result
attributed to it was the fixed prior wearing a different name; and when the cap is widened past the
point where departure becomes possible, the gate departs on 80–90% of nodes and accuracy gets
slightly *worse*, never better. The one structural hope — that Mechanism J's genuine chain-length
heterogeneity would create per-node variation worth adapting to — is measurably absent: depth
representations are richly different from each other (h⁰ and h⁴ are near-orthogonal, so there was
plenty to choose between), but that difference does not vary with chain length in any direction that
holds across seeds. What would have to be different for this to be worth revisiting is specific and
falsifiable: (a) **more label volume** — every null here rests on 519–1,077 test positives against
the spec's 2,000–5,000 target, and the spec-scale corpus already on disk is the standing answer to
that, pending the GPU run every other null in this project is also waiting on; (b) **a gate whose
per-node signal is not mean-pooled away** — Rung 5 averages a node's five depth tokens before
deciding, so it cannot distinguish "h¹ looks informative for me" from "my depths average to
something", which is a plausible reason it collapses to a global choice rather than a per-node one,
and is testable with a per-token gate; or (c) **a dataset where the signal is deliberately placed at
different depths for different nodes**, which no HADES variant currently does — Mechanism J varies
chain *length* but not the depth at which a node's predictive information lives. Absent one of those
three, further gate-design iteration is re-running an experiment whose mechanism is now understood.

## C6. Provenance

80 runs: 25 diagnostic (Variant A and base Rung 5 × Variants 0/J × 5 seeds, plus Variant A on K),
50 lambda-sweep and A+C, 5 base-Rung-5-on-K for the resilience join. All at the V1-comparable
configuration, hidden=128 / num_bases=10 / 100 epochs, four to six worker processes on CPU.

**MPS was not used**, on the measurement in §8.1: it is 2.2× faster for a single process at this
scale but does not parallelise (aggregate throughput flat at ~30 runs/h against CPU's ~34), and
mixing devices inside a comparison would put CPU/MPS float32 reduction differences inside deltas
as small as 0.002. Every Markov baseline these deltas are computed against was produced on CPU.

New code: `ml/run_gate_diagnostic.py` (per-node capture), `ml/analyze_gate_diagnostic.py` (the four
questions), `ml/extract_hidden_state.py` (privileged read of the generator's `RESILIENCE` and
`SUP_CHAIN`, never fed to a model). Raw per-node captures in `out/gate/*.npz`, gitignored.

**The ported model files were not modified.** Annealing is applied by mutating the gate heads'
`lambda_bound` between epochs, and A+C by composing Variant A's constructor with Variant C's freeze
schedule in `ml/train.py` — so all six V1 files remain md5-identical to their originals.
