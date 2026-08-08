# Step 6 Phase 2 — Per-Node Depth Gates (Rung 4 & Rung 5) vs. Markov

**Scope:** build and evaluate BOTH Rung 4 (attention-only depth gate) and Rung 5
(prior-initialized MLP gate) from *"Markov Scoping and Transformer 1.md"* Part 6, on top of
SHARE (RGCN+Attention, `rgcn_attn` in code) — genuine PER-NODE hybrids, the key upgrade over
the earlier task-level depth gate (`reports/step6_depth_gate.md`), which only ever produced one
distribution per task, never per node. Trained simultaneously against a freshly-retrained
Markov baseline (`reports/step6_markov_phase1.md`) for a direct three-way comparison.
Build-and-evaluate only, scoped to SHARE — no changes to SHARP (`rgcn_relemb`) or SHARK
(`rgcn_battn`).

---

## Design

Both rungs reuse SHARE's `h^0..h^4` layer exposure (`ml/models/rgcn_attn_depthgate_encoder.py`)
unchanged. Both are genuinely per-node: given a node's five depth-token representations
`[h^0_i, h^1_i, h^2_i, h^3_i, h^4_i]`, each produces **that node's own** 5-way softmax over
depths — not one distribution shared across all nodes of a task, the limitation the earlier
`rgcn_attn_depthgate` gate had.

**Prior-initialization (shared by both, `ml/models/rung_gate_common.py`).** Each task's gate is
biased at construction so its output distribution is a sharp near-one-hot at that task's Markov
fixed depth (`ml/models/rgcn_attn_markov_encoder.py::MARKOV_READOUT_DEPTH` — delay→h¹,
shortage→h³, impact→h⁴, reused directly from `reports/step6_markov_phase1.md`): the layer that
reads node content is zero-initialized, so at construction `logits = position_bias` for every
node, identical and content-independent, and `softmax(position_bias)` puts ~99.9% of its mass
on the target depth (~0.03% on each other depth). This is done as an **init-time bias**, not a
training-time KL penalty (contrast with `rgcn_attn_depthgate`'s continuous anchor) — epoch 1's
forward pass matches Markov's fixed-depth behavior to numerical precision, bounding the
downside: neither model can plausibly do worse than Markov by more than noise, since both start
there.

**Rung 4 — attention-only gate** (`ml/models/rgcn_attn_rung4_encoder.py`). A single self-attention
layer (`nn.MultiheadAttention`, 4 heads, no FFN) over the 5 depth tokens per node, so `h^1` and
`h^3` can attend to each other before the gate decides the blend, followed by a
(zero-initialized) linear readout → 5 logits + position bias → softmax. The blend itself uses
the **original** tokens (`z_i = Σ_d alpha_i[d] · h^d_i`), not the attention-transformed ones —
attention only *decides* the weights, matching the theory doc's own "gate, not a Transformer"
framing (§6.1) and keeping the two rungs comparable (they differ only in how `alpha` is
computed). **Parameter count at SHARE's real hidden=128** (not the doc's reference d=64):
66,182/task × 3 = **198,546** total — confirmed live.

**Rung 5 — prior-initialized MLP gate** (`ml/models/rgcn_attn_rung5_encoder.py`). No attention
between depth tokens: a small MLP (`hidden→32→5`) reads a **mean-pooled** representation of a
node's five depth tokens (concat was rejected — it would 5x the first layer's parameter count
and defeat the point of being the cheap rung; documented trade-off: mean-pooling discards each
depth's identity before the gate sees it). Same zero-init-final-layer + position-bias strategy.
**Parameter count at hidden=128:** 4,298/task × 3 = **12,894** total — confirmed live, ~15x
cheaper than Rung 4.

Both registered as new architectures (`rgcn_attn_rung4`, `rgcn_attn_rung5`) in
`ml/models/encoder.py` / `ml/train.py`. Live init check confirmed: at construction, every
task's mean alpha was `[0.00034, 0.00034, ..., 0.9987 at target, ...]` with **100% argmax match
rate** across all nodes and zero NaNs, for both rungs.

---

## Pilot — 15 runs

New file `ml/run_rung_pilot.py`: 5 seeds × {`rgcn_attn_markov`, `rgcn_attn_rung4`,
`rgcn_attn_rung5`}, all retrained fresh in the same process (no checkpoint is ever persisted in
this codebase, so pairing against `reports/step6_markov_phase1.md`'s own numbers isn't valid —
same rule every prior round has followed). `hidden=128, num_bases=10, num_layers=4, epochs=100`.

Completed in **1.52h (5,487s)**, 15/15 runs, zero errors. Per-run elapsed confirms Rung 4's
extra cost in wall-clock too, not just parameters: markov ~264–280s, rung5 ~353–355s (+~30%),
rung4 ~438–454s (+~65% over markov) — driven mostly by delay's large node population
(~193,723 node-instances pooled across the 6 test snapshots) running full self-attention every
forward pass.

---

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| markov | 0.8163 ± 0.0023 | 0.7974 ± 0.0031 | 0.9424 ± 0.0023 |
| rung4 | 0.8150 ± 0.0037 | 0.7896 ± 0.0142 | 0.9394 ± 0.0053 |
| rung5 | 0.8150 ± 0.0021 | **0.7976 ± 0.0018** | **0.9443 ± 0.0030** |

rung4's shortage std (0.0142) is 5-8x every other cell's — driven by one outlier seed (seed4:
0.7614, ~0.02-0.04 below its own arm's other 4 seeds), discussed under Training Stability below.

## Pairwise paired bootstrap + 5-seed sign-consistency

| Comparison | delay | shortage | impact |
|---|---|---|---|
| rung4 vs markov | −0.0013, FLIPS | −0.0078, FLIPS | −0.0030, FLIPS |
| rung5 vs markov | −0.0013, FLIPS | +0.0002, FLIPS | +0.0019, FLIPS |
| rung4 vs rung5 | +0.0000, FLIPS | −0.0080, FLIPS | −0.0049, FLIPS |

**Every one of the 9 pairwise cells flips sign across seeds.** No arm reliably beats another on
AUC, on any task, at this label volume — a legitimate "not distinguishable" result, not a null
finding to explain away.

---

## Per-node gate analysis — the actual point of the experiment

For every node of a task's entity type across all 6 test snapshots (not just labeled ones),
after training: does its argmax depth still match the Markov depth it was initialized at?

| Rung | Task | n nodes | Match rate (mean) | Per-seed match rate |
|---|---|---|---|---|
| rung4 | delay | 193,723 | 0.775 | `[1.00, 1.00, 0.62, 0.25, 1.00]` |
| rung4 | shortage | 7,680 | 0.441 | `[1.00, 0.03, 0.07, 1.00, 0.10]` |
| rung4 | impact | 4,800 | 0.999 | `[1.00, 1.00, 1.00, 1.00, 1.00]` (0.996 in 1 seed) |
| rung5 | delay | 193,723 | 0.277 | `[0.46, 0.61, 0.12, 0.09, 0.10]` |
| rung5 | shortage | 7,680 | 0.785 | `[0.80, 0.99, 1.00, 1.00, 0.13]` |
| rung5 | impact | 4,800 | 0.913 | `[0.96, 1.00, 0.73, 0.96, 0.92]` |

**The headline finding: match rate is wildly seed-dependent, in a strikingly bimodal way.**
Several seeds stay pinned essentially exactly at the Markov depth for every single node
(match_rate = 1.00), while other seeds — same architecture, same task, different random
init/dropout mask only — drift so far that 75–98% of nodes end up reading a *different* depth
than the prior. This isn't a smooth, converged per-node adaptation pattern; it's closer to two
distinct regimes (stay-at-prior vs. drift-away) that different seeds happen to land in.

**Degree characterization of deviating nodes** (`degree` = total edges touching that node across
all relations in that snapshot, summed dst-side under `ToUndirected()`, a coarse hub/leaf proxy):

| Rung | Task | degree(deviate) − degree(match), seeds with deviates | Verdict |
|---|---|---|---|
| rung4 | delay | `[(seed2, −0.91), (seed3, +0.04)]`, mean −0.43 | FLIPS |
| rung4 | shortage | `[(1, −0.44), (2, −0.79), (4, +0.06)]`, mean −0.39 | FLIPS |
| rung4 | impact | `[(3, +0.97)]` | n/a (1 seed) |
| **rung5** | **delay** | `[(0,−0.06),(1,−0.17),(2,−0.04),(3,+0.01),(4,+0.07)]`, mean −0.04 | FLIPS |
| **rung5** | **shortage** | `[(0,−0.32),(1,−1.76),(4,−0.34)]`, mean **−0.81** | **CONSISTENT** |
| **rung5** | **impact** | `[(0,−8.72),(2,−17.32),(3,−8.51),(4,−7.65)]`, mean **−10.55** | **CONSISTENT** |

Rung 4 shows no reliable degree pattern anywhere. **Rung 5 does, on two of three tasks**: for
shortage and impact, nodes that deviate from the Markov prior are **consistently
lower-degree** than nodes that stay matched — every seed with any deviating nodes agrees on the
sign, and impact's effect size is large (deviating nodes average ~10.5 fewer edges than matching
ones, against typical degrees in the 15–30 range for that entity type). This is the opposite
direction from the theory doc's own working hypothesis (§5.1: "deviated upward specifically for
multi-division hub suppliers") — here it's the *sparser*, more leaf-like nodes the gate treats
differently, not the hubs. Read cautiously (see caveat below), but it is a real, consistent,
non-random pattern, not noise.

**Caveat on the degree analysis.** `degree` here is a coarse, single-snapshot, all-relations
total — not a tier-specific or BOM-path-specific measure, and not adjusted for entity-type-level
degree distribution skew. It is enough to establish that deviation is *not* independent of
structure (the consistent sign across seeds rules out pure noise), but not enough on its own to
say precisely *which* structural property drives it — that would need a dedicated follow-up
with per-relation degree breakdowns.

---

## Training stability

**Neither gate showed smooth, gradual, small-scale drift from its prior.** Both showed a
bimodal pattern per seed: either essentially zero movement (match_rate ≈ 1.00, `std_alpha`
correspondingly tiny, e.g. rung4 impact: `std_alpha ≈ [0.011, 0.0001, 0.0003, 0.0007, 0.011]`),
or substantial, wide-reaching movement (match_rate well under 0.5, `std_alpha` an order of
magnitude larger, e.g. rung5 delay seed3/4: match_rate 0.09–0.10 with mean alpha spread nearly
uniformly across depths 0/1/2/4 — `mean_alpha ≈ [0.266, 0.271, 0.181, 0.003, 0.279]`, i.e. the
gate essentially abandoned the h³ prior for a near-4-way spread, notably still avoiding h³
itself).

**One suggestive (not conclusive) link between drift and AUC.** Rung 4's shortage seed4 is both
the single worst AUC in the whole 15-run pilot (0.7614, ~0.02–0.04 below its own arm's other 4
seeds) and the seed with the most extreme deviation for that (rung, task) pair (match_rate
0.1023, 89.8% of nodes deviating). One data point isn't a trend, but it's consistent with the
design's own risk framing: drifting far from a validated safe prior is not obviously beneficial,
and can coincide with a real accuracy cost.

**Bottom line on stability:** training did *not* reliably "stay close to initialization" as one
might have hoped from a prior-initialized design, nor did it converge to a *consistent* learned
correction — different seeds found different answers to "should I trust the prior or not,"
which is itself informative: the encoder's loss surface around the Markov-anchored
initialization does not have a single, seed-independent basin that a per-node gate reliably
finds.

---

## Verdict

**Per task, best arm by point estimate** (none clear the 5-seed significance bar against
either alternative):

| Task | markov | rung4 | rung5 | Best (point estimate) |
|---|---|---|---|---|
| delay | **0.8163** | 0.8150 | 0.8150 | markov |
| shortage | 0.7974 | 0.7896 | **0.7976** | rung5 (essentially tied with markov) |
| impact | 0.9424 | 0.9394 | **0.9443** | rung5 |

**Did either hybrid find real, non-noise per-node deviation from Markov?** Partially, and with
an important qualification. Deviation is real in the sense that it isn't measurement artifact —
Rung 5's degree correlation on shortage and impact is 5-seed (or n-of-available-seed) CONSISTENT
in direction, a genuine structural signal, not noise. But the deviation is **not reliable
across random seeds in whether it happens at all** — the same architecture on the same task
sometimes stays glued to the prior and sometimes abandons large fractions of it, with no AUC
difference distinguishing the two regimes. That instability, more than the degree correlation
itself, is this pilot's main finding: a per-node gate initialized safely at a validated prior
does not reliably *stay* safe or reliably *improve* on it — it does one or the other depending
on the random seed, in a way current tooling can't yet predict in advance.

**Does Rung 4's extra attention cost show any measurable benefit over Rung 5's plain MLP?**
**No.** Rung 4 costs ~15x more parameters (198,546 vs. 12,894 in the gates), ~65% more
wall-clock per run (438–454s vs. 353–355s), and its rung4-vs-rung5 AUC deltas flip sign across
seeds on every task (mean deltas: delay +0.0000, shortage −0.0080, impact −0.0049 — the two
negative means both favor rung5, not rung4). Rung 4's per-node degree analysis is also strictly
worse than Rung 5's: none of its three tasks reach sign-consistency, versus two of three for
Rung 5. On every axis measured in this pilot, the attention mechanism's added cost bought
nothing detectable — exactly the outcome "Markov Scoping and Transformer 1.md" §6.4 predicted
("climb the ladder only on evidence... starting at rung 1 and hoping is exactly what produced a
100K-parameter component for a problem with a few hundred labels") and recommended against
without a demonstrated need.

---

## Governance record

Backed up before any write. 15 new registry rows — 5 `rgcn_attn_markov-rungpilot-seed{n}` + 5
`rgcn_attn_rung4-rungpilot-seed{n}` + 5 `rgcn_attn_rung5-rungpilot-seed{n}`, all
`status='active'` — **257 total registry rows** (242 + 15). **225 new evaluation rows** (15
runs × 15 metric rows) — **3,723 total** (3,498 + 225). No existing row from any prior round
modified. Full run log: `reports/logs/run_rung_pilot_20260808_104050.log`.

---

*Scoped to SHARE (RGCN+Attention, `rgcn_attn`) only — no changes to SHARP (`rgcn_relemb`) or
SHARK (`rgcn_battn`).*
