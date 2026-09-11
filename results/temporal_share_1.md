# Run 8 — Temporal-SHARE vs HeteroMP, full population, trained to convergence

**SHARE wins arrival, loses shortage, and neither architecture helps on fill.** That is three
different answers to one question, and the split is the result.

**Phase 3's headline survives the removal of all three qualifications** — full 16,072-channel
population, convergence instead of six epochs, and a real SHARE encoder instead of a simplified
one. The graph contributes **+3.6% to +8.7%** of above-chance signal on arrival in all four
cells, against Phase 3's +5.5–7.2%.

**The v6 shortage task is now readable, and it inverts a Phase 3 conclusion.** A v6-trained
model scored *below chance* on v7 in Phase 3 because v6's labels had no negative class. With
the class restored it scores **0.4521 against a 0.1258 base rate**.

| | |
|---|---|
| Device | **MPS**, Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32 throughout |
| torch | 2.14.0 |
| Total wall-clock | **9,373 s (2.60 h)** for 32 trainings × 2 evaluations |
| Peak RSS | **3.94 GB** — MPS never ran out; no CPU fallback was needed |
| Grid | **complete. Nothing was cut.** |
| Channels | **16,072 — all of them**, every snapshot, every cell |
| Code | `ml/models/share.py`, `ml/train/temporal_share.py`, `ml/train/run8_grid.py` |

---

## 2. The sample-size question — 4,096 was never a training population

**Phase 3 trained on all 16,072 channels.** The premise that it trained on a quarter of the
world is mistaken, and the code settles it in one line. `cross_world.py` calls

```python
X = torch.from_numpy(norm.transform(snapshot_windows(Wd, t0))).to(DEV)   # line 160, 209
```

with `chan=None`, and `snapshot_windows` then builds `np.full(W["NCH"], t0_week)` — `NCH` is
`meta["n_channels"]`, which is 16,072 in both worlds. Every training step and every evaluation
step encoded the entire panel.

**Where 4,096 actually comes from.** The string appears exactly three times in the repository,
and `reports/phase-3.md` is not one of them:

| location | what it is |
|---|---|
| `reports/phase-0.md:129` | a throughput benchmark — `[4096, 38, 52]`, 12 forwards after warm-up |
| `reports/phase-2.md:50-51` | a demo build — `v6: X (4096, 52, 24) build 0.02 s` |
| `reports/phase-2.md:4.2` | the idle-position count was quoted "in a 4,096-channel sample" |

The first is a timing harness, the second a smoke test of `build_sequences`, the third a
sampled measurement. None is a DataLoader batch size either — this trainer has no DataLoader;
one optimiser step consumes one whole snapshot.

**So no re-reading of Phase 3's numbers is needed on this axis, and this run's comparison
against it is not a comparison across two data volumes.** Both ran the full population.

**One genuine sub-population, which is not the training set.** The normaliser is fitted on
`np.arange(0, NCH, 8)` — 2,009 channels — on every fourth training snapshot, in Phase 3 and
here alike. That is a scaler-fitting sample. It changes the mean and standard deviation used to
standardise, not which rows the model sees.

**A real difference from Phase 3 did turn up while checking this.** `cross_world.py` imports
`SPLIT` but never `FIT_WINDOW`, so **Phase 3 trained on 66 snapshots where every baseline it was
compared against used 44** — the 2019–2025 window was specified for both and applied only to
one. This run applies it, so Phase 3's absolute figures and this run's are not measured on
identical training folds. Phase 2 §2 measured the window's cost at 0.15% on arrival for the GBM,
so the effect is small, but it is not zero and it is the one axis on which the two runs differ.

---

## 3. The two architectures

### SHARE — as specified, not approximated

`ml/models/share.py` implements `model_plan.md` "Stage 2" and `benchmark_specification` §4:

- **RGCN basis decomposition**, `W_r = Σ_{b=1}^{10} a_rb V_b`, **B = 10**, computed once per
  layer as an einsum and never materialised per edge;
- **relation-blind shared attention**, `e_ij = LeakyReLU(aᵀ[W_r h_i ‖ W_r h_j])`, one vector
  **a** for every relation, softmaxed over each node's **whole** incoming neighbourhood
  regardless of relation;
- **hidden 128, 4 layers**, returning h⁰…h⁴.

| | SHARE | HeteroMP (Phase 3) |
|---|---|---|
| mechanism | basis-decomposed per-relation weights + attention | learned per-relation **means** + a gate |
| aggregation weights | learned per edge, softmax-normalised | uniform within a relation |
| hidden | 128 | 64 |
| depth | 4 layers | 2 rounds = 4 message-passing layers |
| **encoder params** | **730,992** | **h¹ 33,216 · h⁴ 66,432** |
| **full model params** | **793,777** | **h¹ 83,585 · h⁴ 116,801** |
| h⁰ (no graph module) | **50,369** | **50,369 — the same model** |

**HADES v3's reference is 752,211; SHARE here is 730,992** — within 3%, which is a coincidence
of two different schemas landing on similar totals, not evidence of a faithful port.

**On a 3-relation graph the basis decomposition buys almost nothing, and it should be said
plainly.** Per layer:

```
bases      B x d x d = 10 x 128 x 128 = 163,840
mixing     R x B     =  6 x 10        =      60
self-loop  W_0 + b   = 128 x 128 + 128=  16,512
attention  a in R^2d =                      256
                                       --------
                             per layer = 180,668
                              4 layers = 722,672
             input projection 64 -> 128 =   8,320
```

The graph is channel ↔ {supplier, part, plant}: **three undirected relations, six directed
types**, over 17,119 nodes and 96,432 directed edges. Free per-relation weights would cost
6 × 128 × 128 = 98,304 per layer; the basis scheme costs 163,840 + 60. **Basis decomposition is
more expensive than the thing it exists to replace here.** It saves parameters only when R
exceeds B, and R = 6 against B = 10. The specification's economics assumed R = 20.

**So if SHARE wins, the attention did the work.** The basis machinery is, on this graph, pure
overhead — and §6 shows where that overhead is not repaid.

**Entity nodes carry no sequence.** A supplier has no `channel_performance_weekly` row, so it
enters as the mean of its member channels' TCN states. That is content-derived and keeps the
model **inductive**; a learned per-supplier embedding would key on identity, which the run's
rules forbid.

### The shared TCN front-end and the causality test

Both architectures use the same encoder — kernel 2, dilations 1/2/4/8/16/32, 6 layers, hidden
64, receptive field 64 weeks, **with the residual connections** that Phase 1 made a required
element of the guide. 46,144 parameters.

**Causality unit test — passes bit-identically:**

```
perturb last timestep by +100 -> max abs diff on EARLIER positions: 0.0
bit-identical: True    allclose: True
receptive field: 64 weeks
residual projections: [Conv1d, Identity, Identity, Identity, Identity, Identity]
```

---

## 4. Convergence

Early stopping on the **validation fold** (2024), **patience 5**, cap 30 epochs,
**restore-best-weights**. Every reported number is the restored best checkpoint, never the
final epoch.

**26 of 30 cells converged on patience. Four hit the cap and are reported as NOT converged:**

| train | task | arch | epochs | best epoch | why it matters |
|---|---|---|---|---|---|
| v7 | fill | h⁰ | 30 | **29** | still improving at the cap |
| v6 | shortage | HeteroMP h¹ | 30 | **25** | its 0.6814 may be an underestimate |
| v7 | shortage | HeteroMP h¹ | 30 | **29** | its 0.5089 may be an underestimate |
| v7 | shortage | SHARE h⁴ | 30 | **26** | its 0.4865 may be an underestimate |

**Three of the four are shortage cells, and two of those are HeteroMP** — the architecture that
wins that task. Its margin there is therefore a *lower* bound, which strengthens rather than
weakens §6's conclusion.

Convergence speed splits cleanly by task. Arrival converges fast (best epoch 3–10), fill and
shortage slowly (best epoch 9–29). Phase 3's fixed 6-epoch budget was adequate for arrival and
**was not** for the other two: eleven of the twenty non-arrival cells here found their best
checkpoint at epoch 12 or later, which no 6-epoch run could have reached.

<details>
<summary>Per-cell convergence, all 30</summary>

| train | task | arch | depth | epochs | best ep | stop | wall s | params |
|---|---|---|---|---|---|---|---|---|
| v6 | arrival | none | h0 | 9 | 3 | patience | 127 | 50,369 |
| v6 | arrival | share | h1 | 13 | 7 | patience | 197 | 793,777 |
| v6 | arrival | share | h4 | 16 | 10 | patience | 287 | 793,777 |
| v6 | arrival | mp | h1 | 18 | 12 | patience | 249 | 83,585 |
| v6 | arrival | mp | h4 | 13 | 7 | patience | 183 | 116,801 |
| v7 | arrival | none | h0 | 12 | 6 | patience | 166 | 50,369 |
| v7 | arrival | share | h1 | 10 | 4 | patience | 152 | 793,777 |
| v7 | arrival | share | h4 | 12 | 6 | patience | 216 | 793,777 |
| v7 | arrival | mp | h1 | 13 | 7 | patience | 181 | 83,585 |
| v7 | arrival | mp | h4 | 10 | 4 | patience | 142 | 116,801 |
| v6 | fill | none | h0 | 27 | 21 | patience | 364 | 51,604 |
| v6 | fill | share | h1 | 29 | 23 | patience | 428 | 796,228 |
| v6 | fill | share | h4 | 16 | 10 | patience | 286 | 796,228 |
| v6 | fill | mp | h1 | 24 | 18 | patience | 328 | 84,820 |
| v6 | fill | mp | h4 | 25 | 19 | patience | 344 | 118,036 |
| v7 | fill | none | h0 | 30 | 29 | **CAP** | 404 | 51,604 |
| v7 | fill | share | h1 | 12 | 6 | patience | 181 | 796,228 |
| v7 | fill | share | h4 | 12 | 6 | patience | 216 | 796,228 |
| v7 | fill | mp | h1 | 15 | 9 | patience | 207 | 84,820 |
| v7 | fill | mp | h4 | 18 | 12 | patience | 250 | 118,036 |
| v6 | shortage | none | h0 | 30 | 24 | patience | 402 | 50,369 |
| v6 | shortage | share | h1 | 25 | 19 | patience | 365 | 793,777 |
| v6 | shortage | share | h4 | 21 | 15 | patience | 370 | 793,777 |
| v6 | shortage | mp | h1 | 30 | 25 | **CAP** | 406 | 83,585 |
| v6 | shortage | mp | h4 | 29 | 23 | patience | 396 | 116,801 |
| v7 | shortage | none | h0 | 28 | 22 | patience | 375 | 50,369 |
| v7 | shortage | share | h1 | 24 | 18 | patience | 351 | 793,777 |
| v7 | shortage | share | h4 | 30 | 26 | **CAP** | 527 | 793,777 |
| v7 | shortage | mp | h1 | 30 | 29 | **CAP** | 406 | 83,585 |
| v7 | shortage | mp | h4 | 28 | 22 | patience | 383 | 116,801 |

</details>

### Hyperparameters — tuned once, then frozen

The single tuning pass ran on **train-v7 / test-v7 with SHARE h⁴**, as specified, and compared
two learning rates and nothing else:

| lr | best val | test v7 | best epoch | epochs | wall |
|---|---|---|---|---|---|
| 2e-3 | 0.6638 | 0.6618 | 11 | 17 | 448 s |
| **1e-3** | **0.6669** | **0.6656** | 9 | 15 | 399 s |

`lr 1e-3, wd 1e-4, TCN hidden 64, graph hidden 128, B 10, 4 layers, patience 5, cap 30` was
then applied **unchanged to every cell, every task and both architectures**. HeteroMP inherited
it without adjustment. **This disadvantages HeteroMP** — hidden 128 and B 10 are SHARE's
parameters, and a learning rate chosen on a 794k-parameter model is not obviously right for an
84k-parameter one. It is noted rather than tuned around, and HeteroMP wins shortage anyway.

---

## 5. Cost checkpoint (P2.4)

Timed cell before launching: **SHARE, train-v7 / test-v7, arrival, h⁴, all 16,072 channels —
399 s, 15 epochs, 26.6 s/epoch, peak RSS 3.59 GB.**

```
projected   expected  30 x 399 s + 2 repeats  = 12,768 s = 3.5 h
            worst     32 x 30 epochs x 27 s   = 25,920 s = 7.2 h
actual                32 trainings            =  9,373 s = 2.6 h
```

3.5 h projected against a ~6 h budget, so **the full grid ran and nothing was cut.** The actual
came in under projection because applying the fit window cut the training fold from 66
snapshots to 44.

---

## 6. The comparison

Graph share = (above-chance at h⁴ − above-chance at h⁰) / above-chance at h⁴, measured against
0.5 for C-index, the naive baseline for CRPS, and the base rate for PR-AUC — each on the world
the cell is **tested** on. Baselines recomputed on the 2019–2025 window for both worlds.

**h⁰ is the same trained model in both architecture columns** (§6.1), so every difference
between the two graph-share columns is the encoder and nothing else.

### Arrival timing — C-index ↑

| train | test | h⁰ (shared) | HeteroMP h⁴ | **SHARE h⁴** | MP share | **SHARE share** | P3 h⁰ | P3 h⁴ | P3 share |
|---|---|---|---|---|---|---|---|---|---|
| v6 | v6 | 0.6456 | 0.6577 | **0.6589** | +7.6% | **+8.3%** | 0.6467 | 0.6580 | +7.2% |
| v6 | v7 | 0.6229 | 0.6287 | **0.6347** | +4.5% | **+8.7%** | 0.6170 | 0.6254 | +6.7% |
| v7 | v6 | 0.6539 | 0.6595 | **0.6614** | +3.6% | **+4.7%** | 0.6547 | 0.6638 | +5.5% |
| v7 | v7 | 0.6533 | 0.6597 | **0.6640** | +4.0% | **+6.5%** | 0.6525 | 0.6635 | +6.7% |

### Fill rate — CRPS ↓

| train | test | h⁰ (shared) | HeteroMP h⁴ | **SHARE h⁴** | MP share | **SHARE share** | P3 h⁰ | P3 h⁴ | P3 share |
|---|---|---|---|---|---|---|---|---|---|
| v6 | v6 | 0.0594 | 0.0592 | 0.0596 | +4.9% | **−4.8%** | 0.0596 | 0.0594 | +9.7% |
| v6 | v7 | 0.1004 | 0.1046 | 0.1020 | −130.0% | **−26.8%** | 0.1007 | 0.1006 | +2.2% |
| v7 | v6 | 0.0594 | 0.0593 | 0.0608 | +3.5% | **−58.3%** | 0.0600 | 0.0597 | +16.9% |
| v7 | v7 | 0.0994 | 0.1022 | 0.0997 | −51.0% | **−4.3%** | 0.1015 | 0.1001 | +35.5% |

### Part shortage — PR-AUC ↑ — **all four cells now readable**

| train | test | h⁰ (shared) | HeteroMP h⁴ | **SHARE h⁴** | MP share | **SHARE share** | P3 h⁰ | P3 h⁴ | P3 share |
|---|---|---|---|---|---|---|---|---|---|
| v6 | v6 | 0.5813 | **0.6812** | 0.6203 | **+20.6%** | +9.2% | 1.0000 ⚠ | 1.0000 ⚠ | n/m |
| v6 | v7 | 0.4254 | **0.4521** | 0.4061 | **+8.2%** | −6.9% | 0.1172 | 0.1258 | n/m |
| v7 | v6 | 0.5051 | **0.5323** | 0.4777 | **+8.1%** | −9.7% | 1.0000 ⚠ | 1.0000 ⚠ | n/m |
| v7 | v7 | 0.4581 | **0.5007** | 0.4865 | **+11.4%** | +7.9% | 0.4271 | 0.4388 | +3.8% |

### 6.1 h⁰ is architecture-independent — verified, not assumed

`Net` constructs **no graph module at all** when depth is 0, so a "SHARE h⁰" request and a
"HeteroMP h⁰" request build the identical object. Trained independently three times at the same
seed on train-v7 / arrival:

| request | arch label | params | best epoch | test-v7 | test-v6 |
|---|---|---|---|---|---|
| "SHARE h⁰" | `share`, depth 0 | **50,369** | 6 of 12 | 0.653595 | 0.653608 |
| "HeteroMP h⁰" | `mp`, depth 0 | **50,369** | 4 of 10 | 0.652656 | 0.653455 |
| third run | `share`, depth 0 | **50,369** | 6 of 12 | 0.653780 | 0.653790 |
| the grid's own h⁰ | `none`, depth 0 | **50,369** | 6 of 12 | 0.653277 | 0.653864 |

```
spread on test-v7: max-min 0.001123   sd 0.000602
spread on test-v6: max-min 0.000335   sd 0.000168
params identical across all runs: True
```

**Parameter count identical at 50,369 in every run, and every pairwise gap is inside the
MPS noise band of §6.2.** h⁰ is a genuine common floor, and the SHARE-vs-HeteroMP graph-share
comparison is measured against it cleanly.

### 6.2 The MPS noise band

MPS is **not bit-deterministic**: reduction order on the GPU varies between runs, so the same
seed does not reproduce the same weights. The same cell — SHARE h⁴, train-v7 / test-v7,
arrival, **seed 7 every time** — run three times:

| run | test-v7 C-index |
|---|---|
| 1 | 0.6640 |
| 2 | 0.6634 |
| 3 | 0.6648 |

**spread (max − min) = 0.0014 · sd = 0.0007**

**That band is the standard every margin below has to clear.** A gap smaller than 0.0014 is not
a difference, and one cell in this run falls foul of exactly that.

---

## 7. Does SHARE beat HeteroMP?

**Not in general. It wins one task of three, loses one decisively, and ties on the third.**

### Arrival — SHARE wins, but one of the four cells is inside the noise

At matched depth h⁴, SHARE leads in all four cells. Against the 0.0014 band:

| train | test | SHARE h⁴ | HeteroMP h⁴ | margin | verdict |
|---|---|---|---|---|---|
| v6 | v7 | 0.6347 | 0.6287 | **+0.0060** | real, 4.3× the band |
| v7 | v7 | 0.6640 | 0.6597 | **+0.0043** | real, 3.1× the band |
| v7 | v6 | 0.6614 | 0.6595 | +0.0019 | marginal, 1.4× the band |
| v6 | v6 | 0.6589 | 0.6577 | +0.0012 | **inside the band — not a difference** |

At h¹ the picture is a genuine 2–2 split — SHARE ahead by 0.0031 and 0.0030 on the two
v6-trained cells, HeteroMP ahead by 0.0019 and 0.0004 on the two v7-trained ones, with the last
of those inside the band. **SHARE's advantage on arrival is real but shallow, and it lives at
depth 4.**

### Fill — no difference, in either direction

Eight matched comparisons, split 5–3 to HeteroMP, **every margin ≤ 0.0026 CRPS**. The whole
five-configuration spread on test-v6 is 0.0592 to 0.0609. There is no architecture effect here
to report.

### Shortage — HeteroMP wins, decisively, in all eight comparisons

| depth | v6→v6 | v6→v7 | v7→v6 | v7→v7 |
|---|---|---|---|---|
| h¹ | −0.0221 | −0.0131 | −0.0175 | −0.0109 |
| h⁴ | −0.0608 | −0.0460 | −0.0545 | −0.0142 |

*(negative = SHARE behind)*

**Every one of these is 8× to 43× the noise band.** This is not a tie and it is not noise:
**a 794k-parameter SHARE loses to an 84k-parameter HeteroMP on every shortage cell.** Two of the
HeteroMP cells did not converge, so its true margin is if anything larger.

**Why, mechanistically.** The shortage head reads an **unweighted mean of member-channel states
aggregated to part × plant**. HeteroMP's aggregation is also an unweighted per-relation mean, so
its inductive bias matches the readout exactly. SHARE spends its capacity learning per-edge
attention weights that the mean readout then averages away — it is optimising a distinction the
head discards. Depth compounds it: SHARE h⁴ is worse than SHARE h¹ in all four cells
(0.6593 → 0.6203 on v6→v6), which is over-smoothing on a graph whose channel nodes have exactly
three neighbours.

**And the basis decomposition is where the cost lands.** §3 showed R = 6 < B = 10 makes the
basis scheme *more* expensive than free per-relation weights. On arrival the attention repays
that overhead; on shortage nothing does.

---

## 8. Did full data and convergence change the conclusion?

**No. Phase 3's headline holds on arrival, and the reason it holds is now better understood.**

| | Phase 3 (6 epochs, HeteroMP) | this run, HeteroMP | this run, SHARE |
|---|---|---|---|
| v6→v6 | +7.2% | +7.6% | **+8.3%** |
| v6→v7 | +6.7% | +4.5% | **+8.7%** |
| v7→v6 | +5.5% | +3.6% | **+4.7%** |
| v7→v7 | +6.7% | +4.0% | **+6.5%** |
| **range** | **+5.5 to +7.2%** | +3.6 to +7.6% | **+4.7 to +8.7%** |

The effect **neither grew nor shrank materially — it widened**. Phase 3's four values spanned
1.7 points; SHARE's span 4.0 and HeteroMP's 4.0. Convergence and the fit window add scatter
without moving the centre. The sign is positive in all four cells for both architectures, which
was the finding, and it survives.

**Two things that did change:**

1. **Phase 3's own architecture reproduces its result on arrival but at lower values**
   (+3.6 to +7.6% against +5.5 to +7.2%). The difference is convergence and the fit window, not
   the graph.
2. **The +35.5% fill figure Phase 3 declined to defend was right not to be defended.** With
   convergence it becomes −4.3%. Phase 3 said it "would not defend the +35.5% figure as anything
   but noise on a small margin"; that judgement is confirmed.

**Nothing about the sample size changed the conclusion, because the sample size never changed.**
Both runs used all 16,072 channels — see §2.

---

## 9. v6 shortage, now that it has negatives

Phase 3 could read one of these four cells. All four are readable now, and the two v6-tested
ones invert a Phase 3 conclusion.

| cell | Phase 3 | now (best config) | base rate | what changed |
|---|---|---|---|---|
| v6 → v6 | **1.0000 ⚠** by construction | **0.6814** (MP h¹) | 0.1967 | the test set has a negative class |
| v7 → v6 | **1.0000 ⚠** by construction | **0.5323** (MP h⁴) | 0.1967 | same |
| v6 → v7 | **0.1258, below the 0.1328 base rate** | **0.4521** (MP h⁴) | 0.1258 | the *training* set has a negative class |
| v7 → v7 | 0.4388 | **0.5089** (MP h¹) | 0.1258 | convergence |

**The v6→v7 cell is the one that matters.** Phase 3 concluded a v6-trained shortage model
"learned nothing discriminative", scoring below chance on v7. That was an accurate description
of a model trained on 45,942 rows of which every single one was positive — such a model can only
learn to output a constant. With the class restored it reaches **0.4521 against a 0.1258 base
rate**, and it **beats the LightGBM baseline of 0.4438 trained on v7's own world**.

**A v6-trained shortage model was never untrainable. Its labels were unusable.** The Phase 3
finding was a property of the label table, not of the world — the same shape of error as run 7's
mean-pooling conclusion, and the second time in this project that a negative result about the
world turned out to be a statement about the instrument.

The within-v6 cell also reads sensibly now: **0.6814 against a 0.1967 base rate and a 0.5287
LightGBM baseline**, the largest margin over a GBM anywhere in this run.

---

## 10. Transfer asymmetry — it survives

Penalty as a percentage of above-chance signal, arrival:

| direction | h⁰ | HeteroMP h⁴ | SHARE h⁴ | Phase 3 |
|---|---|---|---|---|
| **v6 → v7** | **−15.6%** | **−18.4%** | **−15.2%** | **−20.6%** |
| **v7 → v6** | +0.4% | −0.1% | −1.6% | +0.2% |

**The asymmetry is intact and architecture-independent.** Training on v6 and testing on v7 costs
roughly a sixth of the signal; the reverse costs nothing measurable, in every configuration
including the pure-TCN floor. It is slightly smaller than Phase 3's −20.6%, consistent with the
fit window and convergence, and the direction and order of magnitude are unchanged.

**It does not generalise past arrival.** On shortage both directions are costly
(v6→v7 −32.7%, v7→v6 −10.5% for HeteroMP h⁴), and on fill the numbers are unstable in both
directions because the margins are too small to support a ratio.

That the asymmetry appears identically at **h⁰**, where there is no graph at all, is worth
recording: it is a property of the two demand regimes, not of the encoder. Training on the
harder, noisier regime generalises to the simpler one; the reverse does not.

**The confound from Phase 3 §P3.3 is unchanged and still applies in full.** v6 → v7 varies
demand persistence, planner behaviour, fill spread and capacity pressure **as a bundle**. This
run isolates nothing new about which axis is responsible. The one component of that bundle that
*has* been removed is the label-schema difference — v6's shortage labels now carry both classes,
as v7's always did.

---

## 11. Against the baselines

Naive and LightGBM recomputed on the 2019–2025 window, on the world each cell is **tested** on.
Shortage baselines are computed on the **label table's own rows**, not on the reconstructed
part × plant universe `learnability_windowed.py` builds, so the populations match.

| task | train | test | best neural | which | naive | LightGBM | beats naive | beats LGBM |
|---|---|---|---|---|---|---|---|---|
| arrival | v6 | v6 | 0.6589 | SHARE h⁴ | 0.5508 | 0.6441 | yes | **yes** |
| arrival | v6 | v7 | 0.6347 | SHARE h⁴ | 0.5578 | 0.6500 | yes | **no** |
| arrival | v7 | v6 | 0.6620 | MP h¹ | 0.5508 | 0.6441 | yes | **yes** |
| arrival | v7 | v7 | 0.6640 | SHARE h⁴ | 0.5578 | 0.6500 | yes | **yes** |
| fill | v6 | v6 | 0.0592 | MP h⁴ | 0.0632 | 0.0599 | yes | **yes** |
| fill | v6 | v7 | 0.1004 | h⁰ | 0.1078 | 0.0988 | yes | **no** |
| fill | v7 | v6 | 0.0593 | MP h⁴ | 0.0632 | 0.0599 | yes | **yes** |
| fill | v7 | v7 | 0.0994 | h⁰ | 0.1078 | 0.0988 | yes | **no** |
| shortage | v6 | v6 | 0.6814 | MP h¹ | 0.1967 | 0.5287 | yes | **yes** |
| shortage | v6 | v7 | 0.4521 | MP h⁴ | 0.1258 | 0.4438 | yes | **yes** |
| shortage | v7 | v6 | 0.5323 | MP h⁴ | 0.1967 | 0.5287 | yes | **yes** |
| shortage | v7 | v7 | 0.5089 | MP h¹ | 0.1258 | 0.4438 | yes | **yes** |

**Every cell beats naive. Nine of twelve beat LightGBM.** The three that do not are the two
fill cells tested on v7 and the one cross-world arrival cell — and on fill the best
configuration on v7 is **h⁰**, the pure TCN, which is the same statement as §7's "no
architecture effect on fill".

**Shortage is where the neural model is furthest ahead**, beating the GBM in all four cells,
including cross-world. That is the task the graph helps most on, and the one that was
unmeasurable before Phase 1.

---

## 12. Still failing, with mechanisms

**1. Fill rate — the graph contributes nothing, and neither encoder beats a plain TCN on v7.**
The best v7-tested configuration is h⁰ in both training worlds. Mechanism: a channel's forward
fill rate is dominated by its own recent fill history, which the TCN already reads, and its
neighbours' fill rates are close to collinear with the `load_ratio` channel h⁰ also has. There
is no residual for the graph to extract. This is the same conclusion run 7 reached with fixed
mean-pooling, and unlike arrival it does **not** flip with a learned encoder.

**2. SHARE loses shortage by a wide margin.** §7 gives the mechanism: the part × plant readout
is an unweighted mean, so learned per-edge attention is averaged away before it reaches the
head, and depth over-smooths a graph where channel nodes have three neighbours. **The fix is a
readout change, not an encoder change** — an attention-weighted or learned pooling from channels
to part × plant would let SHARE's weights survive to the head. Untested here.

**3. Basis decomposition is unjustified on this schema.** R = 6 against B = 10 makes it cost
more than the free per-relation weights it replaces (§3). It is carried because the
specification requires B = 10, not because it earns its place. On a graph with the specified
R = 20 the economics reverse; this graph does not have 20 relations.

**4. Four cells did not converge**, three of them shortage. The cap was 30 epochs with patience
5; those four wanted more. Their numbers are lower bounds and are labelled as such in §4.

**5. The 2019–2025 fit window was specified for Phase 3 and not applied there.** Phase 3
trained on 66 snapshots against baselines fitted on 44. It is applied here, which means Phase 3's
absolute numbers and this run's are not measured on identical training folds. The comparison in
§8 is sound on direction and magnitude and should not be read to four decimals.

**6. Cross-world arrival still loses to a GBM trained on the target world.** 0.6347 against
0.6500 — a model that has never seen v7 does not beat one fitted on it. That is expected and is
recorded because §11 asks for every cell.

**7. None of this is accuracy on Rane's data.** `synthetic_rules.md` §17 is explicit that
seed- and world-to-world agreement is not a proxy for generalisation to a real extract. Every
number here inherits that caveat in full, and rolling-origin backtesting on Rane's data remains
the only thing that would settle it.

---

## Gate

| Requirement | Result |
|---|---|
| Sample-size question settled | **pass** — §2: 4,096 was a benchmark and a demo build; Phase 3 trained on all 16,072 |
| Trained on all channels, no subsampling | **pass** — 16,072 every cell |
| SHARE built as specified (B=10, relation-blind attention, hidden 128, 4 layers) | **pass** — §3, 730,992 encoder params |
| Parameter count reported against v3's 752,211, with the basis-sharing caveat | **pass** — §3 |
| Same TCN both architectures, causality test re-run | **pass** — bit-identical, max diff 0.0 |
| Early stopping, patience ≥ 5, restore-best | **pass** — 26/30 on patience, 4 flagged as CAP |
| Tuned once, frozen | **pass** — §4, one pass on train-v7/test-v7 with SHARE |
| Cost projected before launching | **pass** — §5, 3.5 h projected, 2.6 h actual, nothing cut |
| Full grid, both architectures, h⁰/h¹/h⁴ | **pass** — 30 cells + 2 repeats |
| h⁰ shared and verified | **pass** — §6.1 |
| MPS noise band measured | **pass** — §6.2, spread 0.0014 |

**Three findings that outrank the gate:**

1. **SHARE beats HeteroMP on arrival and loses to it on shortage, both consistently.** A
   794k-parameter encoder with learned attention loses to an 84k-parameter mean aggregator on
   every shortage cell by 8–43× the noise band. The specification's assumption that SHARE
   dominates does not hold on this schema, and the mechanism — a mean readout that discards the
   attention SHARE learns — is a readout problem, not an encoder problem.
2. **Phase 3's arrival finding survives full data, convergence and a real SHARE encoder**, at
   +4.7% to +8.7%. The graph contributes on arrival; that is now measured three ways.
3. **v6's shortage task was never untrainable — its labels were unusable.** With the negative
   class restored, a v6-trained model goes from below chance on v7 to beating the GBM there.
   This is the second time in this project that a negative result about the world turned out to
   be a statement about the instrument.
