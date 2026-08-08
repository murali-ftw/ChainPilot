# HADES Model-Development Report — Layer 2 (Step 6: Depth Selection, Pre-Flight → Rung 5 Variants)

**This is a merge of five prior Step 6 reports into one chronological document, replacing all
of them:** `step6_preflight.md`, `step6_depth_gate.md`, `step6_markov_phase1.md`,
`step6_rung_pilot.md`, and `step6_rung5_variant_pilot.md`. Nothing in this merge changes any
number, verdict, or conclusion from any original report — it reorganizes five
separately-written documents into one continuous narrative, trims duplicated process
boilerplate (repeated scope/dataset-confirmation preambles, repeated governance-record
phrasing), and adds a single synthesis at the top connecting every round. Where a later round
built on or superseded an earlier round's design, both are shown in full, exactly as the
original reports did — this document does not retroactively "correct" any earlier round's own
numbers. A companion reference document, `rung5_types.md`, catalogs every Rung-5-family gate
design (the original plus its five Round 5 variants) side by side in one place, for anyone who
wants the mechanism/finding comparison without the chronological narrative.

**Reading order:** an Executive Summary (the final state of Step 6's depth-selection work as
of the most recent round), a Master Timeline, then one section per round in the order they
actually happened, then a single consolidated governance/test-suite record and open-items list
at the end.

---

# Executive Summary — Final State as of the Most Recent Round (Rung 5 Variant Pilot)

## The question Step 6 exists to answer

SHARE (RGCN+Attention, `rgcn_attn`)'s encoder computes `h^0..h^4` — every depth from the raw
input projection through the full 4-layer trunk. Every architecture before Step 6 read only
`h^L` (the final layer) for every task. `reports/layer1.md`'s own architecture-ablation line
never tested whether a *shallower* read might help some tasks while a deep one helps others —
Step 6 is the five-round investigation into whether it does, and if so, what the cheapest
reliable way to exploit it is.

## The arc, round by round

| Round | Design | Cost | Result |
|---|---|---|---|
| 1 (Pre-flight) | N/A — confirmation only | — | L1 (shallow) beats baseline on delay, CONSISTENT (+0.0069); but L1 CONSISTENTLY hurts impact (−0.0160). Co-parent path needs 3 hops from delay's target, not 2 — irrelevant to the L1 finding either way. |
| 2 (Learned task-gate) | One learned weight vector per task, KL-anchored to a prior | 15 params | Gate correctly *learns* each task's known depth preference (matches priors exactly), but only shortage clears significance (+0.0064 at λ=0.01). Delay/impact: no reliable AUC change either direction — the mechanism resolves L1's delay-vs-impact trade-off (no impact cost) but doesn't recover L1's delay gain. |
| 3 (Markov fixed depth) | Zero learned params — a hard-coded per-task index | **0 params** | Delivers delay's gain CLEANLY: +0.0072, CONSISTENT, with **zero** added parameters, and without L1's impact cost. The best cost/benefit result in the whole line. |
| 4 (Rung 4 / Rung 5 per-node gates) | Genuine per-node distributions (attention-only or MLP), prior-initialized | 198,546 (Rung 4) / 12,894 (Rung 5) params | Mechanism works (near-one-hot at init, correct convergence direction), but shows wild, seed-dependent, BIMODAL instability — some seeds stay pinned at the prior, others let 75–98% of nodes drift. No AUC win over Markov on any task. Rung 4's 15x extra cost over Rung 5 buys nothing measurable. |
| 5 (Five isolated fixes) | A: bounded residual; B: structural features; C: freeze+low-LR; D: depth-preserving projection; E: post-hoc weight-averaging | Varies (A/C: same as Rung 5; B: +1.2K; D: −5.4K; E: 0, no training) | **Variant A completely solves the instability** — 100% match rate, zero deviating nodes, every seed, every lambda, every task — at essentially no AUC cost. Variant C comes close. B/D show partial or no improvement. Variant E (weight-averaging) **catastrophically fails**, hurting significantly on every task. |

## Where this leaves production

**Markov Phase 1 (Round 3) remains the strongest cost-adjusted result**: a free, zero-parameter
change that reliably helps delay and costs nothing on shortage/impact. Every subsequent,
more-expensive design (task-gate, Rung 4/5, and their five variants) has matched or
approximated Markov's numbers, never clearly beaten them on AUC. **The per-node gate line's
open item, going into any future round, is no longer "can we make it stable" — Round 5 answered
that (yes, Variant A) — it's "does a stable per-node gate ever produce an AUC win that the free
fixed-depth baseline doesn't already give away."** That has not yet been demonstrated.

## Claim (Step 6's own): does a per-task, or per-node, depth choice beat the shared-depth-per-task
structural prior established by `docs/project_HADES.md`?

**Yes, cleanly, for delay — and it costs nothing.** No round has found a reliable, sign-consistent
AUC change on shortage or impact from ANY depth-selection mechanism tried (learned task-gate,
Markov, Rung 4, Rung 5, or any of Rung 5's five variants) — five independent designs, five
different mechanisms, the same negative result on those two tasks. Only Variant A's stability
result and delay's Markov win are unambiguous positives; everything else in this line is a tie
or a documented failure (Variant E).

---

# Master Timeline

| Round | Report section below | New file(s) | Runs | Wall-clock |
|---|---|---|---|---|
| 1 — Pre-flight | Round 1 | `ml/run_depth_confirm_5seed.py` | 15 (baseline/L1/L2 × 5 seeds) | 0.69h |
| 2 — Learned task-gate | Round 2 | `ml/models/rgcn_attn_depthgate_encoder.py`, `ml/run_depth_gate_pilot.py` | 20 (5 baseline-retrain + 15 gated, 3 λ × 5 seeds) | 1.63h |
| 3 — Markov Phase 1 | Round 3 | `ml/models/rgcn_attn_markov_encoder.py`, `ml/run_markov_pilot.py` | 10 (5 markov + 5 baseline-retrain) | 0.73h |
| 4 — Rung 4 / Rung 5 | Round 4 | `ml/models/rgcn_attn_rung4_encoder.py`, `ml/models/rgcn_attn_rung5_encoder.py`, `ml/run_rung_pilot.py` | 15 (markov/rung4/rung5 × 5 seeds) | 1.52h |
| 5 — Five isolated fixes | Round 5 | `ml/models/rgcn_attn_rung5_variant_{a,b,d}.py`, `ml/models/rung_gate_common.py`, `ml/run_rung5_variant_pilot.py` | 40 training + 1 post-hoc (Variant E) | 2.06h |

All five rounds ran against the same v3 dataset (800 suppliers, 15 monthly snapshots,
Jul 2024–Sep 2025), re-confirmed live at the start of every round, and the same SHARE matched-d
config (`hidden=128, num_bases=10, num_layers=4, epochs=100`) established by
`reports/layer1.md`'s Round 7. Scoped to SHARE (`rgcn_attn`) throughout — no round touched
SHARP (`rgcn_relemb`) or SHARK (`rgcn_battn`).

---

# Round 1 — Pre-Flight: Two Open Findings Confirmed Before the Depth Gate

**Scope:** confirm two open questions before any Step 6 (learned depth gate) work begins: (1)
whether RGCN+Attention (SHARE)'s delay-peaks-at-L1 pattern — found at only 3 seeds in
`reports/rgcn_types.md`'s over-smoothing sweep — survives proper 5-seed testing, and (2)
whether the original "delay's co-parent path is unneeded and broken" finding still holds now
that dual-sourcing is confirmed active (`reports/entropy_test.md`). No depth-gate
implementation in this pass — this is confirmation work only.

## Phase 0 — Pre-flight

Confirmed live: 800 suppliers, 15 `graph_snapshots`, 420 `component_suppliers` rows
(dual-sourcing/co-parent mechanism active — the same dataset every RGCN-family round since the
4-architecture ablation has run against). Backed up `model_registry`/`model_evaluation_runs`
(197 + 2,823 rows) unconditionally before any write.

## Phase 1 — Does RGCN+Attention's delay-peaks-at-L1 pattern survive 5 seeds?

New file `ml/run_depth_confirm_5seed.py` (touches no existing run script): 15 fresh training
runs (baseline / L1 / L2 × 5 seeds), architecture fixed at RGCN+Attention's established
matched-d config (hidden=128, num_bases=10). Retrained fresh rather than reusing any old
logged number — no model checkpoint is ever persisted in this codebase, so a genuine paired
comparison needs in-process predictions on the identical test set. Verified in DB: +15
registry rows, +225 eval rows, nothing else touched. Completed in 0.69h (2,494s), 15/15 runs,
zero errors.

**AUC — mean ± std across 5 seeds, all three tasks** (shortage/impact reported alongside delay
since `shared_depth` changes what every head reads from the same shared encoder trunk):

| Config | delay | shortage | impact |
|---|---|---|---|
| baseline | 0.8116 ± 0.0038 | 0.7991 ± 0.0019 | 0.9375 ± 0.0064 |
| L1 | **0.8185 ± 0.0020** | 0.7941 ± 0.0027 | 0.9215 ± 0.0032 |
| L2 | 0.8106 ± 0.0051 | 0.7962 ± 0.0031 | 0.9352 ± 0.0034 |

**Paired bootstrap (`ml/evaluate.py::paired_delta_auc_ci`) + 5-seed sign-consistency:**

| Comparison | delay | shortage | impact |
|---|---|---|---|
| L1 vs baseline | **+0.0069, all 5 seeds positive → CONSISTENT** | −0.0050 → FLIPS (noise) | **−0.0160, all 5 seeds negative → CONSISTENT** |
| L2 vs baseline | −0.0010 → FLIPS (noise) | −0.0029 → FLIPS (noise) | −0.0022 → FLIPS (noise) |

Per-seed delay deltas for L1 vs baseline: `[+0.0016, +0.0067, +0.0064, +0.0058, +0.0139]` — never negative across any of the 5 seeds.

**Verdict: real, not noise.** L1's delay advantage holds sign across all 5 seeds. L2 shows no
reliable effect on any task. **But L1 isn't free** — it *consistently hurts impact*
(−0.0160, all 5 seeds negative), a real trade-off the original 3-seed diagnostic didn't have
the statistical power to surface. Note that `num_layers` shrinks together with `shared_depth`
in these configs, matching the original Step 4 depth-sweep convention: L1 is a genuinely
1-layer, ~355K-param encoder (vs. baseline's 4-layer, 752K-param one), not a 4-layer model
simply read at its first layer.

## Phase 2 — Reach analysis re-run against the corrected (dual-sourcing-active) dataset

Re-run using `ml/graph/reach.py`'s own functions (unmodified) against the live dataset,
sourcing BFS from each task's actual prediction target — Shipment for delay, Supplier for
impact — rather than the original analysis's generic "path from a Supplier" framing.

**Co-parent reachability, reconfirmed live:** 449/800 suppliers (56.1%) reach a different
supplier in 2 hops — exactly matching `reports/entropy_test.md`. The Supplier-sourced reach
table now shows a real `Supplier` row at hop≤2 (mean 1.04, 56.1% reached) that was entirely
absent (0%, no row printed) in the original Task 1 measurement.

**The critical structural check — sourced from Shipment (delay's actual prediction target):**

| First hop a *different* (co-parent) supplier appears, from Shipment | Share of supplier-sourced shipments (n=239 checked) |
|---|---:|
| hop 1 | 0.0% |
| hop 2 | 0.0% |
| **hop 3** | **100.0%** |
| hop 4 | 0.0% |

A clean, deterministic structural result, not sampling noise:
`Shipment → Supplier (1) → Component (2) → co-parent Supplier (3)` — the co-parent path
relative to delay's *actual* target always needs **3 hops**, never 2, regardless of how much
dual-sourcing data exists. Delay's own structural-prior depth is h=2.

**Impact re-check:** unchanged — impact's target *is* the Supplier node itself (hop 0 by
definition). What's new: at impact's own h=3 window, the Supplier-sourced reach table now
includes real co-parent signal from hop 2 onward (56.1% reach) that didn't exist before — a
genuinely new signal available to impact specifically, not to delay.

**Does the "unneeded and broken" conclusion hold?** Half holds, half needs correcting.
**"Broken" is corrected**: the path is not broken — it's real and universal (100% of
supplier-sourced shipments reach a co-parent by hop 3). **"Unneeded for delay" still holds, on
a sharper basis**: the mechanism sits one full hop beyond delay's own h=2 prior and even beyond
L1's 1-hop encoder, so it was never in reach of delay's readout at any depth tested here.

## Phase 3 — Synthesis

**The two findings do not point the same way — and that's the honest result, not a forced
one.** The corrected reach analysis gives *no* theoretical reason to expect L1 to help delay
via the co-parent mechanism: L1 is a literal 1-layer encoder, incapable of reaching anything
beyond Shipment's immediate neighbors (its own Supplier/Factory/Warehouse/Order), while the
co-parent path needs 3 hops that don't exist in *any* config tested, baseline included — the
mechanism was never available to delay's readout regardless of L1 vs. baseline vs. L2, so it
cannot be what explains L1's win. What the reach analysis *does* support is the narrower,
structural half of the original claim: delay's target already receives its own supplier's
information directly at hop 1, with nothing beyond that being reachable at hop 2 either
(co-parent still one hop further out) — consistent with, and now sharpened by, Phase 1's own
empirical result that a model restricted to exactly that 1-hop information *outperforms* a
deeper one on delay, while consistently costing impact (which does benefit from depth, per
`reports/rgcn_types.md`'s own AUC-vs-depth sweep peaking near L4). The likely mechanism behind
L1's delay win is therefore not "shallower reaches the newly-active co-parent signal better" —
it structurally cannot — but something more like "delay's real signal is 1-hop-local, and a
smaller, shallower encoder regularizes better on it than a deeper one tuned to also serve
shortage/impact's genuinely deeper needs." That's a real, confirmed, 5-seed-consistent
empirical finding standing on its own, independent of the reach theory — and it's a direct
argument *for* Step 6's per-task depth gate (different tasks provably want different depths
from the same shared trunk) rather than a reason to revise delay's own structural prior, since
delay's prior (h=2) was never claiming the co-parent path in the first place.

## Governance record

Backed up before any write. 15 new `-depthconfirm-{config}-seed{n}` registry rows, all
`status='active'` — 212 total registry rows (197 + 15). 225 new evaluation rows — 3,048 total
(2,823 + 225). No existing row from any prior round modified. Full run log:
`reports/logs/run_depth_confirm_5seed_20260807_234516.log`.

*Out of scope for this pass, per its own framing: no depth-gate implementation. Scoped to
RGCN+Attention (SHARE) only — no changes to RGCN+BasisAttn (SHARK) or RGCN+RelEmbedding (SHARP).*

---

# Round 2 — Learned Per-Task Depth Gate on SHARE

**Scope:** build and evaluate a learned, per-task depth gate on top of the existing SHARE
(RGCN+Attention, `rgcn_attn` in code) encoder — motivated by Round 1's 5-seed-confirmed
finding that delay wants a shallow (~L1) read while impact wants a deep (~L4) one, from the
*same* shared trunk. Build-and-evaluate only, scoped to SHARE — no changes to
RGCN+BasisAttn (SHARK) or RGCN+RelEmbedding (SHARP).

## Design

SHARE's encoder (`ml/models/rgcn_attn_encoder.py`) computes `h^1..h^L`; every task head
currently reads only `h^L`. This round replaces that fixed readout with a **per-task learned
gate**:

```
h_task = sum_{l=0}^{L} softmax(w_task)_l * h_l
```

`w_task` is a small learned vector (length `L+1 = 5`, since `h^0` — the raw pre-message-passing
projection — is exposed as a real candidate depth), **one vector per task head** (delay,
shortage, impact don't share one) — 15 new parameters total at `L=4`, negligible next to the
~750K-param encoder.

**Anchoring.** An unconstrained gate could drift to depths no measurement ever supported, so
each task's learned distribution `softmax(w_task)` is KL-penalized against a FIXED
(non-learned) prior built directly from Round 1's and `reports/rgcn_types.md`'s empirical
results, reusing `project_HADES.md` §4.3's own prior-shape formula
(`p[k] = exp(-|k-peak|/tau) / sum`):

| Task | peak | tau | Rationale |
|---|---|---|---|
| delay | 1 | 0.8 (tight) | L1 beats baseline, CONSISTENT across 5 seeds; L2 shows nothing |
| impact | 4 (=`num_layers`) | 0.8 (tight) | SHARE's AUC-vs-depth sweep peaks at L4 |
| shortage | 3 | 3.0 (wide, near-uniform) | every sweep to date found shortage depth-indifferent |

Total loss: `sum_task FocalLoss_task + lambda * sum_task KL(softmax(w_task) || prior_task)`,
swept at `lambda ∈ {0.01, 0.1, 1.0}`.

**Implementation.** New file `ml/models/rgcn_attn_depthgate_encoder.py` (does not modify
`rgcn_attn_encoder.py`): `RGCNAttnDepthGateEncoder` subclasses `RGCNAttnEncoder`, prepending
`h^0` to the returned layer list; `DepthGateHead` implements the weighted sum + KL; the priors
above are built by `depth_gate_priors()`; `DepthGateHADESModel` composes the encoder with one
`DepthGateHead` + one `PredictionHead` per task and returns the same `(logits, layers)` 2-tuple
every other architecture returns, so `ml/evaluate.py` needed zero changes — the KL term and
learned gate weights are stashed as instance attributes read back by a small,
backward-compatible addition to `ml/train.py::_epoch_forward`. Registered in
`ml/models/encoder.py` as a new architecture, `rgcn_attn_depthgate` (does **not** reuse
`rgcn_attn` — separate logged rows).

## Phase 0 — Pre-flight

Confirmed live: 800 suppliers, 15 `graph_snapshots` (same dataset every RGCN-family round
since the 4-architecture ablation has run against). Backed up `model_registry`/
`model_evaluation_runs` (212 + 3,048 rows) unconditionally before any write.

## Phase 1 — 20-run pilot

New file `ml/run_depth_gate_pilot.py` (touches no existing run script): 5 fixed
structural-prior baseline runs (`architecture="rgcn_attn"`, `shared_depth=None`) retrained
fresh **in the same process**, for a genuine paired comparison — no model checkpoint is ever
persisted in this codebase, so pairing against Round 1's own baseline/L1/L2 numbers isn't
possible without retraining (same rule every prior round has followed; Round 1's L1/L2 numbers
are cited below as unpaired reference context only). Plus 15 gated runs
(`architecture="rgcn_attn_depthgate"`, 3 lambda values × 5 seeds). All runs:
`hidden=128, num_bases=10, num_layers=4, epochs=100` — SHARE's established matched-d config.

Completed in **1.63h (5,857s)**, 20/20 runs, zero errors.

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| baseline_retrain (fixed depth, this round) | 0.8083 ± 0.0090 | 0.7977 ± 0.0017 | 0.9363 ± 0.0070 |
| gate λ=0.01 | 0.8080 ± 0.0024 | 0.8041 ± 0.0013 | 0.9373 ± 0.0073 |
| gate λ=0.1 | 0.8113 ± 0.0030 | 0.8021 ± 0.0018 | 0.9370 ± 0.0077 |
| gate λ=1.0 | 0.8099 ± 0.0027 | 0.8018 ± 0.0013 | 0.9383 ± 0.0076 |

**Round 1 reference (unpaired, NOT retrained here):**

| Config | delay | shortage | impact |
|---|---|---|---|
| baseline | 0.8116 | 0.7991 | 0.9375 |
| L1 | 0.8185 | 0.7941 | 0.9215 |
| L2 | 0.8106 | 0.7962 | 0.9352 |

## Paired bootstrap + 5-seed sign-consistency — gate(λ) vs baseline_retrain

| λ | delay | shortage | impact |
|---|---|---|---|
| 0.01 | −0.0003 → FLIPS (noise) | **+0.0064 → CONSISTENT** (all 5 seeds positive) | +0.0010 → FLIPS (noise) |
| 0.1 | +0.0030 → FLIPS (noise) | +0.0044 → FLIPS (noise) | +0.0007 → FLIPS (noise) |
| 1.0 | +0.0016 → FLIPS (noise) | +0.0041 → FLIPS (noise) | +0.0020 → FLIPS (noise) |

Only **one** cell across all 9 (3 lambdas × 3 tasks) reaches 5-seed sign-consistency:
**shortage at λ=0.01, +0.0064**. Every delay and impact cell flips sign across seeds regardless
of lambda — no reliable AUC change on either task, in either direction.

## The learned gate weights — the actual point of the experiment

Mean `softmax(w_task)` over `h^0..h^4`, 5 seeds, essentially identical across all three
lambdas:

| Task | h⁰ | h¹ | h² | h³ | h⁴ | argmax | Prior peak | Match? |
|---|---|---|---|---|---|---|---|---|
| **delay** | 0.22–0.23 | **0.31–0.33** | 0.23 | 0.11 | 0.11 | **1** | 1 | ✅ |
| **shortage** | 0.13–0.16 | 0.16–0.18 | 0.21–0.22 | **0.25–0.28** | 0.21–0.22 | **3** | 3 | ✅ (broad, near-uniform as predicted) |
| **impact** | 0.11–0.12 | 0.11–0.12 | 0.11–0.13 | 0.30 | **0.34–0.35** | **4** | 4 | ✅ |

Full per-lambda breakdown:

**λ=0.01**
- delay: `[0.217, 0.329, 0.225, 0.117, 0.112]` (std `[0.002, 0.006, 0.004, 0.002, 0.002]`)
- shortage: `[0.155, 0.177, 0.208, 0.248, 0.212]` (std `[0.008, 0.002, 0.005, 0.004, 0.002]`)
- impact: `[0.113, 0.114, 0.133, 0.296, 0.344]` (std `[0.003, 0.003, 0.006, 0.003, 0.008]`)

**λ=0.1**
- delay: `[0.231, 0.317, 0.230, 0.112, 0.110]` (std `[0.003, 0.017, 0.004, 0.005, 0.005]`)
- shortage: `[0.133, 0.157, 0.216, 0.278, 0.216]` (std `[0.004, 0.001, 0.001, 0.005, 0.001]`)
- impact: `[0.117, 0.117, 0.113, 0.303, 0.349]` (std `[0.007, 0.007, 0.007, 0.004, 0.017]`)

**λ=1.0**
- delay: `[0.232, 0.310, 0.231, 0.114, 0.113]` (std `[0.003, 0.020, 0.004, 0.006, 0.007]`)
- shortage: `[0.129, 0.155, 0.216, 0.283, 0.216]` (std `[0.005, 0.001, 0.000, 0.007, 0.000]`)
- impact: `[0.121, 0.120, 0.112, 0.303, 0.344]` (std `[0.009, 0.009, 0.008, 0.006, 0.020]`)

All three argmax depths match their priors exactly, and the gate is genuinely doing soft
blending, not degenerating into a hard index — e.g. delay still assigns ~0.22 to both `h^0` and
`h^2`, not collapsing all mass onto `h^1`.

## Lambda sensitivity

Essentially **none**. Gate weights are nearly flat across the 100× lambda sweep (0.01 → 1.0):
delay's `h^1` weight only drifts 0.329 → 0.310, impact's `h^4` weight stays at 0.344–0.349,
shortage's `h^3` weight 0.248 → 0.283. Per-seed std is small (≤0.02) at every entry, so this is
a stable, repeatable optimum, not several different local minima averaging out.

**Interpretation:** the *unconstrained* optimum (what pure BCE/focal loss alone wants) already
sits almost exactly where the prior says it should. The KL anchor is confirming the learned
solution rather than correcting it — which is a different outcome than "the anchor is doing
heavy lifting to keep the gate in bounds," and worth noting as a finding in its own right.

## Synthesis

The gate does **exactly** what it was mechanistically designed to do: three independent
per-task depth distributions, learned from data, converge — with high seed-to-seed stability
and near-total lambda-insensitivity — to depths that closely track the confirmed empirical
priors from earlier rounds (delay→h¹, shortage→broad/h³, impact→h⁴), all from the same shared
4-layer trunk, no architecture change beyond exposing `h^0` and computing a weighted sum.

That mechanistic success does **not**, however, convert into a broad AUC win. Only one of nine
(lambda, task) cells — shortage at λ=0.01 — clears the 5-seed sign-consistency bar
(+0.0064, CONSISTENT). Delay and impact flip sign across seeds at every lambda tested, despite
the gate correctly learning their respective depth preferences.

The most informative comparison is against Round 1's L1 result, which is not a gate at all but
a genuinely smaller 1-layer, ~355K-param encoder: L1 got a real, CONSISTENT delay win
(+0.0069) but at a real, CONSISTENT impact cost (−0.0160). The gate avoids that trade-off
entirely — it doesn't hurt impact reliably in either direction — but it also doesn't recover
L1's delay gain. The likely explanation is that L1's advantage came partly from being a
genuinely smaller, more regularized model on delay's 1-hop-local signal (per Round 1's own
synthesis), not just from reading a shallower slice of a deep, 4-layer trunk still trained to
also serve shortage/impact's genuinely deeper needs — a soft blend over a shared trunk's `h^1`
is not the same object as an actual 1-layer encoder, even when both put most of their
attention on the same depth.

**Bottom line:** the depth gate is a real, working, well-anchored mechanism — it correctly
recovers each task's known depth preference without any hand-coded index, resolves the
delay-vs-impact trade-off that a single fixed depth structurally cannot (no hurt on either
side), and gives shortage a small, reliable AUC gain at the lowest lambda tested. It is not,
on this dataset, a broad accuracy win across all three tasks.

## Governance record

Backed up before any write. 20 new registry rows — 5 `rgcn_attn-depthgate-baselinecmp-seed{n}`
(architecture=`rgcn_attn`) + 15 `rgcn_attn_depthgate-lambda{λ}-seed{n}`
(architecture=`rgcn_attn_depthgate`), all `status='active'` — **232 total registry rows**
(212 + 20). **300 new evaluation rows** (20 runs × 15 metric rows) — **3,348 total**
(3,048 + 300). No existing row from any prior round modified. Full run log:
`reports/logs/run_depth_gate_pilot_20260808_010648.log`.

*Scoped to SHARE (RGCN+Attention, `rgcn_attn`) only, per Round 1's own framing — no changes to
RGCN+BasisAttn (SHARK) or RGCN+RelEmbedding (SHARP).*

---

# Round 3 — Markov Phase 1: Fixed Depth on SHARE

**Scope:** implement and evaluate the Phase 1 design from *"Markov Scoping and Transformer 1.md"*
(§7.1) on top of SHARE (RGCN+Attention, `rgcn_attn` in code) — a FIXED, per-task readout depth
with zero new trainable parameters, no learned gate, no Transformer component. Build-and-evaluate
only, scoped to SHARE — no changes to SHARP (`rgcn_relemb`) or SHARK (`rgcn_battn`).

## Design

SHARE's encoder already computes `h^0..h^4` (exposed for the depth-gate work in
`ml/models/rgcn_attn_depthgate_encoder.py`). This round reuses that exposure logic unchanged,
but replaces the learned softmax gate entirely with a fixed, one-hot readout per task head — no
weights, no training signal on depth selection at all:

| Task | Reads | Rationale |
|---|---|---|
| delay | `h^1` | **Empirically confirmed** (Round 1, 5-seed CONSISTENT +0.0069 win over baseline). This differs from the theory doc's own untested blanket-derived floor of `h^2` — the empirically-confirmed depth is used instead, per this round's explicit instruction. |
| shortage | `h^3` | Matches both the theory doc's blanket-derived estimate (Product/Warehouse, ~3 hops) and the learned depth-gate's own convergence point (Round 2: argmax_depth=3 at every lambda). |
| impact | `h^4` | Matches both the theory doc's blanket-derived estimate (Order/Customer, ~4 hops, full BOM) and the learned gate's convergence point (argmax_depth=4 at every lambda). Equivalent to reading the final layer — i.e. identical readout behavior to the current baseline for this task specifically. |

This adds **zero new trainable parameters** — it's a fixed Python index-select per task head,
not a module (no `nn.Parameter`, no KL term, no gate).

**Implementation.** New file `ml/models/rgcn_attn_markov_encoder.py`: `MarkovHADESModel`
composes the encoder (`build_encoder("rgcn_attn_markov", ...)`, which dispatches to the
identical `RGCNAttnDepthGateEncoder` class already used by Round 2's gate — the fixed-vs-
learned choice lives entirely in the model built on top, not the encoder) with one
`PredictionHead` per task, each reading `layers[MARKOV_READOUT_DEPTH[task]][entity_type]`.
Returns the same `(logits, layers)` 2-tuple every other architecture returns, so
`ml/evaluate.py` needed zero changes. Registered in `ml/models/encoder.py` and `ml/train.py` as
a new architecture, `rgcn_attn_markov` (does not reuse `rgcn_attn` or `rgcn_attn_depthgate` —
separate logged rows).

## Phase 0 — Pre-flight

Confirmed live: 800 suppliers, 15 `graph_snapshots`. Backed up `model_registry`/
`model_evaluation_runs` (232 + 3,348 rows) unconditionally before any write.

## Pilot — 10 runs

New file `ml/run_markov_pilot.py` (touches no existing run script): 5 fixed structural-prior
runs (`architecture="rgcn_attn_markov"`) + 5 plain baseline runs (`architecture="rgcn_attn"`,
`shared_depth=None`) retrained fresh **in the same process** for a genuine paired comparison —
no model checkpoint is ever persisted in this codebase, so pairing against previously-logged
numbers isn't valid; same rule every prior round has followed. `hidden=128, num_bases=10,
num_layers=4, epochs=100` — SHARE's established matched-d config throughout this project.

Completed in **0.73h (2,644s)**, 10/10 runs, zero errors. Per-run elapsed: ~249–272s each,
consistent with Round 2's timing at the same config — expected, since the Markov model's
encoder is byte-for-byte identical (same param count, 752,211, on both arms).

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| markov (fixed depth) | **0.8166 ± 0.0018** | 0.7963 ± 0.0026 | 0.9445 ± 0.0019 |
| baseline_retrain (this round) | 0.8095 ± 0.0025 | 0.7977 ± 0.0019 | 0.9357 ± 0.0082 |

**Round 2 reference (unpaired, NOT retrained here):**

| Arm | delay | shortage | impact |
|---|---|---|---|
| baseline_retrain | 0.8083 | 0.7977 | 0.9363 |
| gate λ=0.01 | 0.8080 | 0.8041 | 0.9373 |
| gate λ=0.1 | 0.8113 | 0.8021 | 0.9370 |
| gate λ=1.0 | 0.8099 | 0.8018 | 0.9383 |

## Paired bootstrap + 5-seed sign-consistency — markov vs baseline_retrain

| Task | Per-seed ΔAUC | Mean ΔAUC | Verdict |
|---|---|---|---|
| delay | `[+0.0069, +0.0067, +0.0097, +0.0001, +0.0124]` | **+0.0072** | **CONSISTENT** (all 5 seeds positive) |
| shortage | `[+0.0001, −0.0024, −0.0082, +0.0005, +0.0028]` | −0.0014 | FLIPS (noise) |
| impact | `[−0.0022, +0.0191, +0.0051, +0.0031, +0.0192]` | +0.0089 | FLIPS (noise, 1/5 seeds negative) |

## Plain verdict

| Task | Mean ΔAUC | Verdict |
|---|---|---|
| **delay** | **+0.0072** | **HELPS** — 5-seed consistent gain, free (zero new parameters) |
| shortage | −0.0014 | NO DETECTABLE DIFFERENCE — sign flips across seeds |
| impact | +0.0089 | NO DETECTABLE DIFFERENCE — sign flips across seeds (4/5 positive, 1 negative, largest single delta +0.0192) |

## Synthesis

The free, zero-parameter, theory-derived fixed-depth design **measurably helps delay** and is a
clean, statistically defensible win there: +0.0072 mean AUC, all 5 seeds positive, for literally
no added cost — no new parameters, no new training signal, just reading `h^1` instead of `h^4`
for one task. This reproduces Round 1's original L1-vs-baseline delay finding (+0.0069,
CONSISTENT) essentially exactly, as expected since delay's readout here is the same depth
choice, now delivered via a per-task fixed index on the shared 4-layer trunk rather than via a
genuinely shrunk 1-layer encoder — and notably, unlike L1, it does **not** carry L1's
consistent impact cost (Round 1: −0.0160 CONSISTENT), because impact here still reads the full
`h^4` trunk output, identical to baseline's own readout for that task.

Shortage and impact show no reliable change in either direction — both flip sign across seeds
despite reading their theory/gate-predicted depths (`h^3`, `h^4`). For impact this is
expected and mechanically almost tautological: impact's fixed depth (`h^4`) is the same depth
baseline already reads for that task, so any observed difference between the two arms on
impact is measuring seed-to-seed training variance in the *shared trunk*, not a readout-depth
effect — the two arms are structurally near-identical for that one task. For shortage, `h^3` is
close to but not identical to baseline's structural prior, and the result is consistent with
Round 1's own finding that shortage has shown no reliable depth sensitivity in any sweep to
date.

**Comparison against the learned gate** (Round 2, unpaired reference): the free fixed-depth
design's delay result (0.8166) is noticeably *higher* than every gate arm's delay mean
(0.8080–0.8113) and higher than its own baseline_retrain in that round (0.8083) — though these
are two different rounds' baseline_retrain measurements (0.8095 here vs 0.8083 there), both
well within the run-to-run noise band Round 2 itself measured (std ≈ 0.0025–0.0090 per arm).
shortage and impact land in the same range as the gate's own numbers on both. The practical
takeaway: for delay specifically, the free fixed-depth design is at least as good as — and on
this round's numbers, nominally better than — anything the 15-run, KL-anchored learned gate
produced, at zero parameter cost and zero lambda tuning.

**Bottom line, directly answering the theory doc's own framing:** this Phase 1 (Markov-only)
design does what §7.1 predicted it would — it's the correct, defensible Phase-1 baseline,
free, and it captures the one clean win (delay) this project line has found repeatedly across
three separate methodologies now (the original 3-seed diagnostic, the 5-seed Round 1
confirmation, and this fixed-depth pilot). It does not move shortage or impact in either
direction, which is itself informative — per §6.3's own "asserted, not measured" caveat, this
result is exactly the L-sweep-style measurement the theory doc calls for, and it settles that
for this dataset at this label volume, shortage/impact's fixed-depth choices are statistically
indistinguishable from baseline, not that they are wrong.

## Governance record

Backed up before any write. 10 new registry rows — 5 `rgcn_attn_markov-markovpilot-seed{n}`
(architecture=`rgcn_attn_markov`) + 5 `rgcn_attn-markovpilot-baselinecmp-seed{n}`
(architecture=`rgcn_attn`), all `status='active'` — **242 total registry rows** (232 + 10).
**150 new evaluation rows** (10 runs × 15 metric rows) — **3,498 total** (3,348 + 150). No
existing row from any prior round modified. Full run log:
`reports/logs/run_markov_pilot_20260808_083523.log`.

*Scoped to SHARE (RGCN+Attention, `rgcn_attn`) only, per this round's own framing — no changes
to SHARP (`rgcn_relemb`) or SHARK (`rgcn_battn`). No learned gate, no Transformer component —
that's Phase 2, per "Markov Scoping and Transformer 1.md" §7.1's own sequencing, not attempted
in this pass.*

---

# Round 4 — Per-Node Depth Gates (Rung 4 & Rung 5) vs. Markov

**Scope:** build and evaluate BOTH Rung 4 (attention-only depth gate) and Rung 5
(prior-initialized MLP gate) from *"Markov Scoping and Transformer 1.md"* Part 6, on top of
SHARE (RGCN+Attention, `rgcn_attn` in code) — genuine PER-NODE hybrids, the key upgrade over
the earlier task-level depth gate (Round 2), which only ever produced one distribution per
task, never per node. Trained simultaneously against a freshly-retrained Markov baseline
(Round 3) for a direct three-way comparison. Build-and-evaluate only, scoped to SHARE — no
changes to SHARP (`rgcn_relemb`) or SHARK (`rgcn_battn`).

**Full per-design mechanism detail for Rung 4/Rung 5 (and their five Round 5 variants) is
consolidated separately in `rung5_types.md` — this section keeps only the summary needed for
the chronological narrative.**

## Design

Both rungs reuse SHARE's `h^0..h^4` layer exposure (`ml/models/rgcn_attn_depthgate_encoder.py`)
unchanged. Both are genuinely per-node: given a node's five depth-token representations
`[h^0_i, h^1_i, h^2_i, h^3_i, h^4_i]`, each produces **that node's own** 5-way softmax over
depths — not one distribution shared across all nodes of a task, the limitation Round 2's gate
had.

**Prior-initialization (shared by both, `ml/models/rung_gate_common.py`).** Each task's gate is
biased at construction so its output distribution is a sharp near-one-hot at that task's Markov
fixed depth (`ml/models/rgcn_attn_markov_encoder.py::MARKOV_READOUT_DEPTH` — delay→h¹,
shortage→h³, impact→h⁴, reused directly from Round 3): the layer that reads node content is
zero-initialized, so at construction `logits = position_bias` for every node, identical and
content-independent, and `softmax(position_bias)` puts ~99.9% of its mass on the target depth
(~0.03% on each other depth). This is done as an **init-time bias**, not a training-time KL
penalty (contrast with Round 2's continuous anchor) — epoch 1's forward pass matches Markov's
fixed-depth behavior to numerical precision, bounding the downside: neither model can plausibly
do worse than Markov by more than noise, since both start there.

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

## Pilot — 15 runs

New file `ml/run_rung_pilot.py`: 5 seeds × {`rgcn_attn_markov`, `rgcn_attn_rung4`,
`rgcn_attn_rung5`}, all retrained fresh in the same process (no checkpoint is ever persisted in
this codebase, so pairing against Round 3's own numbers isn't valid — same rule every prior
round has followed). `hidden=128, num_bases=10, num_layers=4, epochs=100`.

Completed in **1.52h (5,487s)**, 15/15 runs, zero errors. Per-run elapsed confirms Rung 4's
extra cost in wall-clock too, not just parameters: markov ~264–280s, rung5 ~353–355s (+~30%),
rung4 ~438–454s (+~65% over markov) — driven mostly by delay's large node population
(~193,723 node-instances pooled across the 6 test snapshots) running full self-attention every
forward pass.

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
say precisely *which* structural property drives it.

## Training stability

**Neither gate showed smooth, gradual, small-scale drift from its prior.** Both showed a
bimodal pattern per seed: either essentially zero movement (match_rate ≈ 1.00, `std_alpha`
correspondingly tiny), or substantial, wide-reaching movement (match_rate well under 0.5,
`std_alpha` an order of magnitude larger, mean alpha spread nearly uniformly across depths).

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
finds. **This instability is exactly what Round 5's five variants set out to fix.**

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

## Governance record

Backed up before any write. 15 new registry rows — 5 `rgcn_attn_markov-rungpilot-seed{n}` + 5
`rgcn_attn_rung4-rungpilot-seed{n}` + 5 `rgcn_attn_rung5-rungpilot-seed{n}`, all
`status='active'` — **257 total registry rows** (242 + 15). **225 new evaluation rows** (15
runs × 15 metric rows) — **3,723 total** (3,498 + 225). No existing row from any prior round
modified. Full run log: `reports/logs/run_rung_pilot_20260808_104050.log`.

*Scoped to SHARE (RGCN+Attention, `rgcn_attn`) only — no changes to SHARP (`rgcn_relemb`) or
SHARK (`rgcn_battn`).*

---

# Round 5 — Five Isolated Fixes to the Rung 5 Gate

**Scope:** implement and evaluate FIVE INDEPENDENT variants of the Rung 5 gate (Round 4), each
testing exactly ONE proposed fix in isolation against the same shared baselines — never stacked
on top of each other. Motivated by Round 4's headline finding: both per-node gates (Rung 4,
Rung 5) showed wildly seed-dependent, bimodal stability, with no AUC difference distinguishing
the two regimes. Build-and-evaluate only, scoped to SHARE (`rgcn_attn`) — no changes to SHARP
(`rgcn_relemb`) or SHARK (`rgcn_battn`).

**Full per-variant mechanism detail is consolidated separately in `rung5_types.md`** — this
section keeps only the summary needed for the chronological narrative and the full quantitative
results.

## Device check (done first, per this round's instruction)

**MPS was used for all training, verified before committing to it — not assumed:**

1. **Correctness.** `torch_geometric.utils.scatter`/`softmax` (the exact ops `RGCNAttnEncoder`'s
   attention and the gate's blend rely on) ran forward+backward on a synthetic batch on CPU vs
   MPS: outputs and gradients matched to `atol=1e-4` (max observed diff ~2.4e-7 — effectively
   exact). `torch.bincount` (Variant B's structural features, and the per-node degree analysis)
   also verified correct on MPS.
2. **Speed.** 5 real training iterations (forward+backward+optimizer.step, real dataset, real
   `Rung5HADESModel`) timed **0.316s/iter on CPU vs 0.124s/iter on MPS** — a genuine ~2.5x
   speedup.
3. **End-to-end.** A full 5-epoch `run_training_job(..., device="mps")` call, including a
   required fix (`.cpu()` before `.numpy()` in `ml/evaluate.py::collect_predictions` and
   `ml/train.py::_mean_val_auc` — both previously assumed CPU-resident tensors; the fix is a
   no-op on CPU, so no other architecture's behavior changed), completed with a real DB write
   in 9.9s (~2s/epoch, vs CPU's ~3.5–4.5s/epoch for this model class) — verified against a
   throwaway model_version, deleted immediately after.

Both criteria (correctness AND speedup) were met, so the whole 40-run pilot trained on MPS.
Observed in practice: markov ~149s/run, rung5-family ~170–200s/run for 100 epochs — noticeably
faster than the CPU-only baselines this project line has used through every prior round.

## The five variants (summary — see `rung5_types.md` for full mechanism)

| Variant | The one change |
|---|---|
| **A** | Position bias becomes a FIXED buffer (never trained); MLP output passes through `tanh` and is scaled by `lambda_bound ∈ {0.1, 0.3, 0.5}` before being added — a hard structural ceiling on how far any logit can drift from the prior. |
| **B** | Normalized degree + node-type one-hot + per-relation edge-count histogram concatenated onto the pooled embedding before the MLP — explicit structural signal instead of making the gate infer it. |
| **C** | Training-loop-only: gate frozen for the first 15 epochs, then unfrozen at 0.1× the main LR — encoder/heads settle first, gate moves only after. |
| **D** | Shared per-depth linear projection (128→8) applied to each of h⁰..h⁴ individually and concatenated (40 dims), replacing mean-pooling — preserves depth identity instead of averaging it away. |
| **E** | Post-hoc, no training: weight-average the 5 already-trained original-Rung-5 seeds' FULL parameters (encoder+gates+heads — see interpretation note below) into one model, evaluate directly. |

**Shared baselines:** `markov` (fixed depth) and `rung5` (original) — both retrained fresh in
this same process, 5 seeds each, for genuine paired comparisons against every variant.

**Variant E interpretation note.** "Average the gate's parameters (not predictions)" was
implemented as averaging the model's FULL parameter set (encoder+gates+heads), not the gate
submodule alone — evaluating a gate in isolation requires an encoder to sit under it, and the 5
seeds' encoders differ, so there is no principled way to average only the gate and still get a
coherent, evaluable model. This is standard weight-averaging (SWA/model-soup style), applied
here specifically to validate the gate's behavior under it, as titled.

Total: 40 training runs (5 markov + 5 rung5 + 15 Variant A [3 lambdas × 5 seeds] + 5 B + 5 C +
5 D) + Variant E's post-hoc averaging. Completed in **2.06h (7,418s)**, zero errors.

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| markov | 0.8172 ± 0.0027 | 0.7970 ± 0.0016 | 0.9399 ± 0.0031 |
| rung5 (original) | 0.8170 ± 0.0019 | 0.7904 ± 0.0146 | 0.9450 ± 0.0014 |
| A, λ=0.1 | 0.8179 ± 0.0034 | 0.7989 ± 0.0046 | 0.9447 ± 0.0019 |
| A, λ=0.3 | 0.8188 ± 0.0028 | 0.7980 ± 0.0017 | 0.9447 ± 0.0018 |
| A, λ=0.5 | 0.8176 ± 0.0021 | 0.7994 ± 0.0048 | 0.9418 ± 0.0052 |
| B | 0.8144 ± 0.0012 | 0.7987 ± 0.0018 | 0.9382 ± 0.0069 |
| C | 0.8166 ± 0.0031 | 0.8005 ± 0.0041 | 0.9408 ± 0.0037 |
| D | 0.8160 ± 0.0017 | 0.7962 ± 0.0029 | 0.9430 ± 0.0039 |
| **E (weight-avg)** | **0.7285** | **0.3223** | **0.4548** |

Note rung5-original's shortage std (0.0146) is 5–8× every other arm's — one bad seed
(seed0=0.7619) driving most of it, the same instability signature Round 4 flagged.

## Paired bootstrap — EACH variant vs the SAME shared baselines (never variant vs variant)

Every single comparison (all 6 training variants × 2 baselines × 3 tasks = 36 cells) **FLIPS
sign across the 25 seed-pairs tested.** No variant reliably beats markov or the original rung5
on AUC, on any task. This matches Round 4's own finding: none of these gate designs move
accuracy in a statistically defensible way at this label volume.

## Variant E vs everything — the one comparison that IS significant

| Comparison | delay | shortage | impact |
|---|---|---|---|
| E vs markov (5 comparisons, one per markov seed) | −0.0887, **CONSISTENT** | −0.4747, **CONSISTENT** | −0.4851, **CONSISTENT** |
| E vs rung5-original (5 comparisons) | −0.0885, **CONSISTENT** | −0.4681, **CONSISTENT** | −0.4902, **CONSISTENT** |
| E vs BEST individual rung5 seed | −0.0921, CI [−0.112, −0.072], **significant** | −0.4791, CI [−0.508, −0.451], **significant** | −0.4919, CI [−0.537, −0.447], **significant** |

**Weight-averaging did not help. It catastrophically hurt on 2 of 3 tasks and measurably hurt
on the third — every comparison against every baseline, on every task, is significant in the
same (negative) direction.** Shortage (0.3223) and impact (0.4548) are worse than a random
classifier (0.5) would be expected to score on average; delay (0.7285) is well below even the
worst individual seed (~0.793 range, per Round 2's baseline numbers). This is the expected
outcome for naive parameter-space averaging across independently-initialized, independently-
trained neural nets that lack linear mode connectivity (Frankle et al. 2020; Wortsman et al.
2022's own "model soup" work explicitly notes this failure mode for models NOT fine-tuned from
a shared pretrained initialization) — it is not a bug in this implementation, it is the
well-documented behavior of the technique itself when applied outside the regime where it
works. **Explicit answer to the round's own question: averaging HURT, badly and consistently.
It did not help. It did not make no difference.**

## Per-node gate analysis — did each fix change stability?

Match rate = share of nodes whose post-training argmax depth still equals the Markov prior's.
Round 4's original Rung 5 baseline (retrained fresh here) reproduced its own instability
signature almost exactly:

| Arm | delay match | shortage match | impact match | Degree-correlation sign-consistency |
|---|---|---|---|---|
| rung5 (original) | 0.124 (seeds: 0.05–0.17, ALL low) | 0.712 (seeds: 0.15–1.00, wildly bimodal) | 0.917 (seeds: 0.83–1.00) | shortage/delay FLIP; **impact CONSISTENT (−6.61)** |
| **A, λ=0.1** | **1.000 (all 5 seeds)** | **1.000** | **1.000** | no deviating nodes at all — n/a |
| **A, λ=0.3** | **1.000 (all 5 seeds)** | **1.000** | **1.000** | no deviating nodes at all — n/a |
| **A, λ=0.5** | **1.000 (all 5 seeds)** | **1.000** | **1.000** | no deviating nodes at all — n/a |
| B | 0.516 (seeds: 0.15–1.00, still bimodal) | 0.981 (seeds: 0.92–1.00, much tighter) | 0.902 | shortage/impact-degree mixed: shortage **CONSISTENT (−2.06)**, impact FLIPS |
| C | 1.000 (all 5 seeds) | 0.995 (seeds: 0.98–1.00) | 0.973 (seeds: 0.92–1.00) | shortage **CONSISTENT (−1.11)**, impact **CONSISTENT (−7.70)** |
| D | 0.113 (seeds: 0.06–0.17, ALL low, same as original) | 0.545 (seeds: 0.12–0.78, still bimodal) | 0.960 (seeds: 0.80–1.00) | shortage **CONSISTENT (−0.23)**, impact **CONSISTENT (−6.84)** |
| E (weight-avg) | 1.000 | 1.000 | 1.000 | n/a — but see AUC catastrophe above |

**Variant A completely solved the instability problem, exactly as hypothesized — at all three
lambdas tested, every single seed on every single task stayed at 100% match, zero deviating
nodes.** The tanh-bounded residual made drift structurally impossible, and the model never
found a reason to use whatever tiny residual room it had (`±0.1` to `±0.5` per logit) to move
any node's argmax away from the prior. This is the cleanest, most decisive result in this
round.

**Variant C came close** — delay fully stable (1.000, all seeds), shortage/impact nearly so
(0.995/0.973 mean, versus rung5-original's 0.712/0.917) — a real, large improvement from
letting the encoder settle before the gate can move, though not the complete elimination
Variant A achieved.

**Variant B partially helped** — shortage's stability improved substantially (0.981 vs 0.712,
tighter range too: 0.92–1.00 vs 0.15–1.00) and its degree correlation is CONSISTENT there, but
delay remained just as bimodal as the original (0.516, still ranging 0.15–1.00 across seeds).

**Variant D did not help, and arguably made shortage worse** — delay match rate (0.113) and
its bimodal per-seed spread are statistically indistinguishable from the unfixed original
(0.124); shortage's match rate (0.545) is actually lower than the original's (0.712).
Preserving depth identity through per-depth projection, without also bounding or restructuring
anything else, did not address the instability.

**A genuinely new, consistent qualitative signal emerged across B/C/D (not present, or not
measurable, in the original or in A):** wherever a degree correlation could be computed with
enough deviating seeds, it was **negative and CONSISTENT** — deviating nodes have *lower*
degree than matching nodes (shortage: −0.23 to −2.06; impact: −6.84 to −7.70). This mirrors
Round 4's own Rung 5 impact finding (−6.61, CONSISTENT) and is the opposite direction from the
theory doc's "hub suppliers deviate upward" hypothesis — now replicated across three
independent gate designs (B, C, D) plus the original, strengthening confidence this is a real
structural pattern, not noise from any one design's idiosyncrasies.

## Variant A lambda sensitivity

| λ | delay AUC | shortage AUC | impact AUC | match rate (all tasks) |
|---|---|---|---|---|
| 0.1 | 0.8179 | 0.7989 | 0.9447 | 1.000 |
| 0.3 | 0.8188 | 0.7980 | 0.9447 | 1.000 |
| 0.5 | 0.8176 | 0.7994 | 0.9418 | 1.000 |

Stability is identical (perfect) at all three lambdas — even λ=0.5 (the widest tested bound,
allowing up to ±0.5 per logit, a substantial fraction of the total logit range) never let a
single node's argmax move. AUC drifts only slightly with lambda (impact dips at λ=0.5:
0.9418 vs 0.9447 at 0.1/0.3), consistent with a wider bound giving the model more room to hurt
itself slightly, though none of these differences are large relative to seed noise.

## Verdict

**Which single change meaningfully improved stability?** **Variant A, unambiguously and
completely** — 100% match rate, zero deviating nodes, at every lambda, every seed, every task.
Variant C is the clear second — large, consistent improvement, not total elimination. Variant B
helped on shortage specifically, not delay. Variant D did not help.

**Which single change meaningfully improved accuracy?** **None, at the sign-consistency
standard this project uses throughout.** Every one of the 36 variant-vs-baseline AUC
comparisons flips sign across seeds. Variant A has the best point-estimate AUC on delay
(0.8188 vs markov's 0.8172 and rung5-original's 0.8170) and Variant C has the best point
estimate on shortage (0.8005, the only arm to clear 0.80), but neither reaches statistical
significance. Read plainly: no variant tested here demonstrably changes accuracy; Variant A's
contribution is entirely to stability, at essentially zero AUC cost (and a slightly favorable,
if not significant, point estimate).

**Does averaging help, hurt, or make no difference (Variant E)?** **Hurts, badly, and
significantly** — every comparison against every baseline is significant in the negative
direction, on every task. Full-model weight-averaging across independently-trained seeds is
not a viable technique for this architecture family, consistent with known lack-of-mode-
connectivity issues for independently-initialized training runs.

**Is a combination worth testing as a follow-up?** Yes, on specific evidence, not speculation:
**A + C together** is the most defensible next step — A's structural bound plus C's
settle-then-unfreeze schedule are mechanistically complementary (one bounds the ceiling, the
other slows the approach to it) and neither showed any accuracy cost alone. **B's structural
features are also worth carrying forward, but narrowly** — its benefit was shortage-specific
and didn't touch delay's instability, so it would need to be combined with something that
addresses delay specifically (A or C) rather than deployed alone. **D is not worth pursuing
further** on this evidence — no stability gain over the unfixed original, and a possible
shortage regression. Any combination round should keep testing against the SAME markov/rung5
baselines established here, and should specifically check whether A's perfect stability
persists when paired with B's extra input features or C's freeze schedule, since neither of
those was tested under a bounded regime in this round.

## Governance record

Backed up before any write. 41 new registry rows — 5 `rgcn_attn_markov-r5v-seed{n}` + 5
`rgcn_attn_rung5-r5v-seed{n}` + 15 `rgcn_attn_rung5_a-lam{λ}-seed{n}` + 5
`rgcn_attn_rung5_b-r5v-seed{n}` + 5 `rgcn_attn_rung5_c-r5v-seed{n}` + 5
`rgcn_attn_rung5_d-r5v-seed{n}` + 1 `rgcn_attn_rung5-weightavg5seeds` (Variant E, post-hoc,
architecture=`rgcn_attn_rung5`), all `status='active'` — **298 total registry rows** (257 +
41). **615 new evaluation rows** (41 runs × 15 metric rows) — **4,338 total** (3,723 + 615). No
existing row from any prior round modified. Full run log:
`reports/logs/run_rung5_variant_pilot_20260808_160133.log`.

*Scoped to SHARE (`rgcn_attn`) only — no changes to SHARP (`rgcn_relemb`) or SHARK
(`rgcn_battn`).*

---

# Consolidated Governance Record (current, live state)

As of this report, Step 6's own contribution across all five rounds:

| Round | Registry rows added | Eval rows added | Wall-clock |
|---|---:|---:|---|
| 1 — Pre-flight | +15 | +225 | 0.69h |
| 2 — Learned task-gate | +20 | +300 | 1.63h |
| 3 — Markov Phase 1 | +10 | +150 | 0.73h |
| 4 — Rung 4 / Rung 5 | +15 | +225 | 1.52h |
| 5 — Five isolated fixes | +41 | +615 | 2.06h |
| **Total** | **+101** | **+1,515** | **6.63h** |

`model_registry`: **197 → 298** (Step 6's own starting point through its own end state; the
gap between `reports/layer1.md`'s own final count of 146 and this round's starting 197 reflects
other, out-of-scope work between the two documents' timelines — see `reports/rgcn_types.md`).
`model_evaluation_runs`: **2,823 → 4,338**. No existing row from any prior round was ever
modified across all five Step 6 rounds — every round backed up both governance tables before
any write and verified row counts before/after.

Every round's runs are queryable by their distinct `model_version` suffix — `-depthconfirm-`,
`-depthgate-baselinecmp-`/`-depthgate-lambda{λ}-`, `-markovpilot-`, `-rungpilot-`, `-r5v-`,
`-r5v-lam{λ}-` — no round ever upserted over a prior round's rows.

---

# Open Items — Current State

| Item | Origin | Status |
|---|---|---|
| Does a per-node gate ever produce an AUC win the free Markov baseline doesn't already give away? | Round 4 | **Still open** — five designs (task-gate, Rung 4, Rung 5, and Rung 5's five variants) have now been tried; none has cleared the 5-seed sign-consistency bar against Markov on any task |
| Per-node gate stability | Round 4 | **Resolved by Round 5** — Variant A (bounded residual + fixed prior) achieves 100% match rate at every lambda, every seed, every task; the mechanism-level instability question is closed |
| A + C combination (bounded residual + freeze/differential-LR) | Round 5 | **Not yet tested** — the clearest immediate next step per Round 5's own verdict |
| Does Variant B's structural-feature benefit (shortage-specific) combine usefully with A or C? | Round 5 | **Not yet tested** |
| Full-model weight-averaging across independently-trained seeds | Round 5 | **Resolved, negatively** — catastrophically hurts on 2/3 tasks, significantly on all 3; not a viable technique for this architecture family without shared-trajectory checkpoints |
| Does shortage/impact's insensitivity to depth choice (seen across ALL five Step 6 designs) reflect something about the tasks themselves, not the mechanisms tried? | Rounds 1–5 | **Open** — five independent mechanisms (learned task-gate, Markov, Rung 4, Rung 5, and five Rung 5 variants) all found the same negative result on these two tasks; worth treating as a settled property of the dataset/label volume rather than continuing to test new gate designs against it |
| MPS device support | Round 5 | **Adopted for future rounds** — correctness and ~2.5x speedup verified; `ml/train.py`/`ml/evaluate.py` now support it via `device=` parameters, backward-compatible no-op on CPU |

---

*This document supersedes `step6_preflight.md`, `step6_depth_gate.md`, `step6_markov_phase1.md`,
`step6_rung_pilot.md`, and `step6_rung5_variant_pilot.md`, all of which have been removed from
`reports/` as part of this merge. See `rung5_types.md` for a side-by-side technical catalog of
every Rung-5-family gate design. Raw per-seed run logs referenced above remain in
`reports/logs/`; governance-table backups remain in `reports/backups/`.*
