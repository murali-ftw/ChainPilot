# Step 6 — Learned Per-Task Depth Gate on SHARE

**Scope:** build and evaluate a learned, per-task depth gate on top of the existing SHARE
(RGCN+Attention, `rgcn_attn` in code) encoder — motivated by `reports/step6_preflight.md`'s
5-seed-confirmed finding that delay wants a shallow (~L1) read while impact wants a deep (~L4)
one, from the *same* shared trunk. Build-and-evaluate only, scoped to SHARE — no changes to
RGCN+BasisAttn (SHARK) or RGCN+RelEmbedding (SHARP).

---

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
(non-learned) prior built directly from `reports/step6_preflight.md`'s and
`reports/rgcn_types.md`'s empirical results, reusing `project_HADES.md` §4.3's own prior-shape
formula (`p[k] = exp(-|k-peak|/tau) / sum`):

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

---

## Phase 0 — Pre-flight

Confirmed live: 800 suppliers, 15 `graph_snapshots` (same dataset every RGCN-family round
since the 4-architecture ablation has run against). Backed up `model_registry`/
`model_evaluation_runs` (212 + 3,048 rows) unconditionally before any write.

## Phase 1 — 20-run pilot

New file `ml/run_depth_gate_pilot.py` (touches no existing run script): 5 fixed
structural-prior baseline runs (`architecture="rgcn_attn"`, `shared_depth=None`) retrained
fresh **in the same process**, for a genuine paired comparison — no model checkpoint is ever
persisted in this codebase, so pairing against `step6_preflight.md`'s own baseline/L1/L2
numbers isn't possible without retraining (same rule every prior round has followed;
`step6_preflight.md`'s L1/L2 numbers are cited below as unpaired reference context only). Plus
15 gated runs (`architecture="rgcn_attn_depthgate"`, 3 lambda values × 5 seeds). All runs:
`hidden=128, num_bases=10, num_layers=4, epochs=100` — SHARE's established matched-d config.

Completed in **1.63h (5,857s)**, 20/20 runs, zero errors.

---

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| baseline_retrain (fixed depth, this round) | 0.8083 ± 0.0090 | 0.7977 ± 0.0017 | 0.9363 ± 0.0070 |
| gate λ=0.01 | 0.8080 ± 0.0024 | 0.8041 ± 0.0013 | 0.9373 ± 0.0073 |
| gate λ=0.1 | 0.8113 ± 0.0030 | 0.8021 ± 0.0018 | 0.9370 ± 0.0077 |
| gate λ=1.0 | 0.8099 ± 0.0027 | 0.8018 ± 0.0013 | 0.9383 ± 0.0076 |

**`reports/step6_preflight.md` reference (unpaired, NOT retrained here):**

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

---

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

---

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

The most informative comparison is against `step6_preflight.md`'s L1 result, which is not a
gate at all but a genuinely smaller 1-layer, ~355K-param encoder: L1 got a real, CONSISTENT
delay win (+0.0069) but at a real, CONSISTENT impact cost (−0.0160). The gate avoids that
trade-off entirely — it doesn't hurt impact reliably in either direction — but it also doesn't
recover L1's delay gain. The likely explanation is that L1's advantage came partly from being a
genuinely smaller, more regularized model on delay's 1-hop-local signal (per
`step6_preflight.md`'s own synthesis), not just from reading a shallower slice of a deep,
4-layer trunk still trained to also serve shortage/impact's genuinely deeper needs — a soft
blend over a shared trunk's `h^1` is not the same object as an actual 1-layer encoder, even
when both put most of their attention on the same depth.

**Bottom line:** the depth gate is a real, working, well-anchored mechanism — it correctly
recovers each task's known depth preference without any hand-coded index, resolves the
delay-vs-impact trade-off that a single fixed depth structurally cannot (no hurt on either
side), and gives shortage a small, reliable AUC gain at the lowest lambda tested. It is not,
on this dataset, a broad accuracy win across all three tasks.

---

## Governance record

Backed up before any write. 20 new registry rows — 5 `rgcn_attn-depthgate-baselinecmp-seed{n}`
(architecture=`rgcn_attn`) + 15 `rgcn_attn_depthgate-lambda{λ}-seed{n}`
(architecture=`rgcn_attn_depthgate`), all `status='active'` — **232 total registry rows**
(212 + 20). **300 new evaluation rows** (20 runs × 15 metric rows) — **3,348 total**
(3,048 + 300). No existing row from any prior round modified. Full run log:
`reports/logs/run_depth_gate_pilot_20260808_010648.log`.

---

*Scoped to SHARE (RGCN+Attention, `rgcn_attn`) only, per `reports/step6_preflight.md`'s own
framing — no changes to RGCN+BasisAttn (SHARK) or RGCN+RelEmbedding (SHARP).*
