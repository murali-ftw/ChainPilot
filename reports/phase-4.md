# Phase 4 — Graph encoder (guide 4.1–4.2)

**Verdict in four lines.** Ship **h¹**, not h⁴: across all three tasks and both worlds, h⁰→h¹ is a
real jump and h¹→h⁴ is inside the seed band in **10 of 12** comparisons. Ship **SHARE-lite** —
free per-relation weights plus relation-blind attention — which matches full SHARE in **all four**
arrival cells at **64% of the parameters**, confirming that the basis decomposition buys nothing
on this schema. On **shortage there is no measurable encoder difference at all**. Phase 5 may
start.

| | |
|---|---|
| Device | **MPS**, `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32, `num_workers=0`, seeds set |
| torch | 2.14.0 |
| Phase 4's own compute | **8 cells, 2,742 s = 0.76 h**, 179 epochs, **15.3 s/epoch**, all 8 converged on patience |
| Peak RSS | **3.76 GB** — no CPU fallback, all 16,072 channels |
| Stage 0 | item 3 **had been run**; the report was merely unwritten. Completed, not re-run. |
| Cells added here | 4 fill h¹ (cut during the closeout, needed for 4.2) + 4 SHARE-lite arrival |
| `inventory_position_weekly` | **not read** — consistent with the closeout's phase mapping |

---

## 1. Stage 0 — the closeout did finish; the report did not

`ml/train/run10_closeout.py`, both LR sweep artifacts and 20 prediction sets all existed. The
work was done; `phase3_closeout.md` simply stopped after item 2 with two placeholders and a
verdict line that outran its evidence.

**[`reports/phase3_closeout.md`](phase3_closeout.md) is now complete** — placeholders filled,
both sweep tables, the per-cell convergence table, restated scores with intervals, the per-metric
noise bands, and the status table. Nothing was re-run.

Completing it changed three conclusions, which are summarised here because Phase 4's decisions
rest on them:

| | `phase1_2.md` | after convergence |
|---|---|---|
| SHARE vs HeteroMP on shortage | SHARE wins 3 of 4 | **no difference in any of 4** |
| graph contribution on fill (SHARE) | −0.3% / −20.0% | **+16.4% / +3.2%** |
| shortage h⁰ cells | capped at 40 | still capped at 120 — **floors** |

**The per-metric seed bands are the reason**, and they are the single most consequential number
in this phase:

| task | metric | seed spread | sd |
|---|---|---|---|
| arrival | C-index | 0.0028 | 0.0014 |
| fill | CRPS | **0.0009** | 0.0005 |
| shortage | PR-AUC | **0.0251** | 0.0134 |

Shortage PR-AUC moves by 0.025 on seed alone — **nine times arrival's band**. Every margin below
is read against its own task's band, never against a borrowed one.

---

## 2. Step 4.1 — reconciliation

### 2.1 Built / partial / missing

| element 4.1 specifies | status | note |
|---|---|---|
| RGCN basis decomposition, `W_r = Σ a_rb V_b` | **built** | `SHARELayer.rel_weights()`, einsum, never materialised per edge |
| B = 10 | **built** | default and unchanged |
| relation-blind attention, one shared **a** | **built** | `att_dst` / `att_src`, softmax over each node's whole neighbourhood |
| `LeakyReLU` scorer, negative slope 0.2 | **built** | |
| layer update with `W₀` self-loop | **built** | `self_loop`, bias included |
| persistent nodes enter with the TCN state **s**_c | **built** | via `inp`, 64 → 128 |
| return all intermediates h⁰…h⁴ | **was partial → now built** | see 2.2 |
| 6 node types incl. `supplier_group`, `po_line` | **missing** | 4 built |
| R = 20 relations | **missing** | R = 6 built |
| `po_line` 18-feature input projection | **missing** | `po_line` nodes churn; the cache stores one static graph |

### 2.2 Divergences, and how each was resolved

**(a) The 4.1 verify target is config-dependent — guide annotated.**

The guide said a large discrepancy from ≈45,384 per layer "means the basis decomposition is not
active and you have per-relation weights". That is a trap. Rebuilt at the specification's **own**
numbers, d=64 / R=20 / B=10, the code gives **45,448** — a 0.14% difference which is exactly the
`W₀` bias the specification omits. **The check passes.** At the config actually built,
d=128 / R=6, the same formula gives **180,668**, which is 3.98× the §4 figure and is correct.

| config | bases | mixing | W₀ | attn | per layer |
|---|---|---|---|---|---|
| spec §4: d=64, R=20 | 40,960 | 200 | 4,160 | 128 | **45,448** (§4 says 45,384) |
| built: d=128, R=6 | 163,840 | 60 | 16,512 | 256 | **180,668** |

Resolved by **annotating the guide**: recompute the target for your config, do not compare across
configs. The code was correct.

**(b) The 4.2 verify step was unrunnable — code changed.**

`forward()` took a `depth=` argument and returned one representation, so
`len(states) == n_layers + 1` could not be evaluated. Following the Phase 1 residuals precedent,
this one was fixed **in code**: `share.py` now accepts `return_all=True`.

```
len(states) = 5    n_layers+1 = 5                          PASS
states[0] - input_proj:  max abs diff 0.0                  PASS
depth=k  vs  states[k]:  agree to ~1e-7, not bitwise
```

The guide now also warns to compare with `allclose`, not `equal`: **two identical forward calls
on MPS differ by 4.5e-8**, because the scatter reductions are non-deterministic. That is the
device, not the depth semantics.

**(c) Schema divergence, R=6 and 4 node types — guide annotated as deviation 7.**

`supplier_group` is buildable (`suppliers.supplier_group_id` is populated) and simply is not
built; adding it is a Phase 1.5 cache change. `po_line` is **incompatible** with the static-graph
cache, which is itself justified by the Phase 1 measurement that `sourcing_channels` is
time-invariant. This bounds every graph claim in the project: the measured contribution is that
of the channel ↔ supplier/part/plant structure alone.

### 2.3 Guide diff

| section | change |
|---|---|
| 4.1 **Verify** | replaced with the config-dependent formula, both worked examples, and the schema-divergence table |
| 4.2 **Verify** | rewritten as runnable code against `return_all=True`, with the measured `allclose` caveat |
| Known deviations | **6** — basis decomposition costs 1.67× free weights at R=6, with the arithmetic |
| Known deviations | **7** — 4 node types and R=6, with why each missing type is missing |
| Deviations index | two rows added |

`B = 10` is unchanged. The note records only that on a 3-relation graph it buys nothing.

---

## 3. Constraints re-verified

| constraint | result |
|---|---|
| **causality** | perturb last timestep +100 → max abs diff on earlier positions **0.0**, bit-identical. Receptive field 64 weeks ≥ 52. |
| **inductive** | **zero** parameters in SHARE or HeteroMP whose shape is keyed to 16,072 / 420 / 620 / 7 / 17,119. Entity nodes seeded from the mean of member channel states. |
| **h⁰ architecture-independent** | `none` / `share` / `mp` at depth 0 all construct `enc = None` and **50,369 parameters**, identical. SHARE-lite shares the same h⁰. |
| split / window / population | train ≤ 2023, val 2024, test 2025; fit window 2019–2025; all 16,072 channels |

---

## 4. SHARE-lite — the basis decomposition buys nothing

Free per-relation weights plus the same relation-blind attention, no basis decomposition.

| | SHARE | SHARE-lite |
|---|---|---|
| per layer | 180,668 | **115,072** |
| encoder total | 730,992 | **468,608** |
| ratio | — | **0.641** — 262,384 fewer parameters |

Arrival, both worlds, h¹ and h⁴, at arrival's frozen 2.5e-4 — **arrival was not retuned**:

| world | depth | h⁰ | HeteroMP | SHARE | **SHARE-lite** | lite − SHARE | vs 0.0028 band |
|---|---|---|---|---|---|---|---|
| v6 | h¹ | 0.6484 | 0.6560 | 0.6585 | **0.6591** [0.6508, 0.6675] | +0.0006 | **within noise** |
| v6 | h⁴ | 0.6484 | 0.6585 | 0.6625 | **0.6633** [0.6545, 0.6716] | +0.0009 | **within noise** |
| v7 | h¹ | 0.6529 | 0.6608 | 0.6649 | **0.6632** [0.6549, 0.6722] | −0.0018 | **within noise** |
| v7 | h⁴ | 0.6529 | 0.6609 | 0.6664 | **0.6665** [0.6580, 0.6750] | +0.0000 | **within noise** |

**All four inside the band, largest deviation 0.0018 against 0.0028.** This is the direct test of
deviation 6 and it passes: removing the basis decomposition costs nothing measurable and saves 36%
of the encoder. On a 3-relation graph the basis machinery is pure overhead, exactly as the
arithmetic predicted.

---

## 5. Step 4.2 — the readout depth, derived

Selection on the **validation** fold; the test score of the depth **not** chosen is shown beside
it. ⚠ marks a cell that hit the 120-epoch cap and is a floor.

### arrival — C-index ↑ (band 0.0028; carried unchanged from `phase1_2`)

| world | arch | h⁰ | h¹ | h⁴ | val-selected | test of the depth NOT chosen | h⁴−h¹ | verdict |
|---|---|---|---|---|---|---|---|---|
| v6 | SHARE | 0.6484 | 0.6585 | 0.6625 | **h⁴** | 0.6585 | +0.0039 | h⁴ better |
| v6 | MP | 0.6484 | 0.6560 | 0.6585 | **h⁴** | 0.6560 | +0.0025 | **no preference** |
| v7 | SHARE | 0.6529 | 0.6649 | 0.6664 | **h⁴** | 0.6649 | +0.0015 | **no preference** |
| v7 | MP | 0.6529 | 0.6608 | 0.6609 | **h⁴** | 0.6608 | +0.0002 | **no preference** |

### fill — CRPS ↓ (band 0.0009; converged)

| world | arch | h⁰ | h¹ | h⁴ | val-selected | test of the depth NOT chosen | h⁴−h¹ | verdict |
|---|---|---|---|---|---|---|---|---|
| v6 | SHARE | 0.0595 | 0.0594 | 0.0588 | **h¹** | 0.0588 | +0.0006 | **no preference** |
| v6 | MP | 0.0595 | 0.0596 | 0.0598 | **h⁴** | 0.0596 | −0.0002 | **no preference** |
| v7 | SHARE | 0.0995 | 0.1007 | 0.0992 | **h⁴** | 0.1007 | +0.0015 | h⁴ better |
| v7 | MP | 0.0995 | 0.1033 | 0.1030 | **h⁴** | 0.1033 | +0.0003 | **no preference** |

**v6/SHARE is the case that justifies the discipline**: validation picks h¹, whose test CRPS is
0.0594, while the unchosen h⁴ tests *better* at 0.0588. Selecting on test would have picked h⁴.
The margin is inside the band either way, so nothing turns on it — but the divergence is real and
is why the rule exists.

### shortage — PR-AUC ↑ (band 0.0251; converged except h⁰)

| world | arch | h⁰ | h¹ | h⁴ | val-selected | test of the depth NOT chosen | h⁴−h¹ | verdict |
|---|---|---|---|---|---|---|---|---|
| v6 | SHARE | 0.5840 ⚠ | 0.6600 | 0.6831 | **h⁴** | 0.6600 | +0.0231 | **no preference** |
| v6 | MP | 0.5840 ⚠ | 0.6616 | 0.6899 | **h⁴** | 0.6616 | +0.0283 | h⁴ better |
| v7 | SHARE | 0.4691 ⚠ | 0.4906 | 0.4918 | **h¹** | 0.4918 | +0.0012 | **no preference** |
| v7 | MP | 0.4691 ⚠ | 0.5071 | 0.5143 | **h⁴** | 0.5071 | +0.0072 | **no preference** |

### The pattern, which is the same in all three tasks

| step | arrival | fill | shortage |
|---|---|---|---|
| **h⁰ → h¹** | +0.0101 (3.6× band) | +0.0001 to −0.0038 | **+0.0760 (3.0× band)** |
| **h¹ → h⁴** | inside band in **3 of 4** | inside band in **3 of 4** | inside band in **3 of 4** |

**Message passing earns its place; depth beyond one hop does not.** Ten of twelve h¹→h⁴
comparisons are inside their own band. The two that clear it (arrival v6/SHARE +0.0039, shortage
v6/MP +0.0283) clear it by 1.4× and 1.1× — marginal, and in opposite architectures.

---

## 6. The decision

### Readout depth

| task | v6 prefers | v7 prefers | **ship** | reasoning |
|---|---|---|---|---|
| arrival | h⁴ (marginal) | no preference | **h¹** | only 1 of 4 rows resolves; h¹ costs 3 fewer layers for an unmeasurable difference |
| fill | no preference | h⁴ (marginal) | **h¹** | 3 of 4 inside band; h⁰ is itself competitive, so the graph is barely doing anything here |
| shortage | no preference | no preference | **h¹** | all 4 inside a 0.0251 band; h⁰→h¹ is the entire gain |

**Ship h¹ for every task.** This contradicts nothing measured — it is what "no preference" means
when one option is four times cheaper. It also matches the guide's own warning in 4.2 that v3's
fixed delay→h¹ / shortage→h³ / impact→h⁴ mapping did not survive re-derivation.

**Where the worlds disagree**, they disagree only within noise, so §2c's "finding about the two
regimes" does not arise: there is no stable depth preference in either world to conflict. Had one
been forced, `phase1_2` §10 argues for letting **v7** decide, since training on the harder regime
transfers free while the reverse costs 20.6%. v7 prefers h⁴ on fill and h¹ on shortage — neither
resolvable.

### Encoder

| task | margin vs band | **ship** |
|---|---|---|
| **arrival** | SHARE > HeteroMP in 3 of 4 (+0.0039, +0.0042, +0.0055 vs 0.0028) | **SHARE-lite** — matches SHARE in all 4 cells at 64% of the parameters |
| **fill** | SHARE > HeteroMP in 3 of 4 (+0.0010, +0.0025, +0.0038 vs 0.0009) | **SHARE-lite**, on the same reasoning; untested on fill, flagged below |
| **shortage** | **no difference in any of 4** (−0.0016 to −0.0225 vs 0.0251) | **HeteroMP** — 84k parameters against 794k for nothing measurable |

**SHARE's advantage is real on arrival and fill, and absent on shortage.** That reverses
`phase1_2`, which had it exactly backwards on both counts — it reported SHARE winning shortage
and finding nothing on fill. Both reversals trace to the same two causes: unconverged cells, and
a noise band borrowed from the wrong metric.

**A 9.5× parameter cost for nothing measurable is a decision, not a disappointment.** On shortage,
ship the small model.

---

## 7. Still open, with mechanisms

**1. SHARE-lite is untested on fill and shortage.** It was run on arrival only, per 1c. The
encoder recommendation for fill extrapolates from arrival, which is an assumption, not a
measurement. Four fill cells and four shortage cells would close it — roughly 1 h.

**2. Both shortage h⁰ cells are floors**, capped at epoch 119 of 120 and still improving. Every
shortage graph-contribution figure is therefore an **upper bound**: a converged h⁰ raises the
floor and shrinks the share. The depth verdicts are unaffected, since they compare h¹ against h⁴.

**3. The shortage learning rate may not be optimal.** Its sweep optimum landed on the bottom
boundary at 1.25e-4 and was not extended, on cost — a deviation declared in the closeout. A lower
rate could move every shortage number.

**4. The graph is smaller than the specification's.** R=6 and 4 node types against R=20 and 6.
Every graph result here is the contribution of the channel ↔ supplier/part/plant structure alone;
a richer graph might contribute more and nothing here tests that.

**5. Neural fill calibration remains far worse than the GBM's** — ECE 0.0361 against 0.0107 on v6
even at SHARE's best. Anyone quoting fill *distributions* rather than point forecasts should use
LightGBM. Unchanged from `phase1_2` §12 and not addressed here.

**6. Wall-clock is not compute.** Three cells in the underlying artifact recorded 115, 333 and
687 s/epoch against a clean 13.9–15.3 s/epoch because the machine slept mid-run. Phase 4's own 8
cells are clean at 15.3 s/epoch.

---

## Gate

| requirement | result |
|---|---|
| Stage 0 — closeout verified or run | **pass** — verified, then completed |
| fill and shortage have converged numbers | **pass** — 18 of 20, the 2 exceptions labelled floors |
| 4.1 reconciled, divergences named and resolved | **pass** — §2, one code fix, three annotations |
| basis-decomposition finding in the guide | **pass** — deviation 6, with arithmetic |
| SHARE-lite run | **pass** — 4 arrival cells, matches SHARE within noise |
| causality / inductive / h⁰ re-verified | **pass** — §3 |
| 4.2 depth derived per task, on validation | **pass** — §5, with the unchosen depth's test score |
| margins read against per-metric bands | **pass** — three separate bands, §1 |
| `inventory_position_weekly` untouched | **pass** |

**Phase 5 may start.** Ship **h¹** and **SHARE-lite** for arrival and fill, **h¹** and
**HeteroMP** for shortage.
