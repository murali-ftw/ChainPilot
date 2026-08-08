# Step 6 Phase 1 — Markov-Blanket-Derived Fixed Depth on SHARE

**Scope:** implement and evaluate the Phase 1 design from *"Markov Scoping and Transformer 1.md"*
(§7.1) on top of SHARE (RGCN+Attention, `rgcn_attn` in code) — a FIXED, per-task readout depth
with zero new trainable parameters, no learned gate, no Transformer component. Build-and-evaluate
only, scoped to SHARE — no changes to SHARP (`rgcn_relemb`) or SHARK (`rgcn_battn`).

---

## Design

SHARE's encoder already computes `h^0..h^4` (exposed for the depth-gate work in
`ml/models/rgcn_attn_depthgate_encoder.py`). This round reuses that exposure logic unchanged,
but replaces the learned softmax gate entirely with a fixed, one-hot readout per task head — no
weights, no training signal on depth selection at all:

| Task | Reads | Rationale |
|---|---|---|
| delay | `h^1` | **Empirically confirmed** (`reports/step6_preflight.md`, 5-seed CONSISTENT +0.0069 win over baseline). This differs from the theory doc's own untested blanket-derived floor of `h^2` — the empirically-confirmed depth is used instead, per this round's explicit instruction. |
| shortage | `h^3` | Matches both the theory doc's blanket-derived estimate (Product/Warehouse, ~3 hops) and the learned depth-gate's own convergence point (`reports/step6_depth_gate.md`: argmax_depth=3 at every lambda). |
| impact | `h^4` | Matches both the theory doc's blanket-derived estimate (Order/Customer, ~4 hops, full BOM) and the learned gate's convergence point (argmax_depth=4 at every lambda). Equivalent to reading the final layer — i.e. identical readout behavior to the current baseline for this task specifically. |

This adds **zero new trainable parameters** — it's a fixed Python index-select per task head,
not a module (no `nn.Parameter`, no KL term, no gate).

**Implementation.** New file `ml/models/rgcn_attn_markov_encoder.py`: `MarkovHADESModel`
composes the encoder (`build_encoder("rgcn_attn_markov", ...)`, which dispatches to the
identical `RGCNAttnDepthGateEncoder` class already used by the depth-gate pilot — the fixed-vs-
learned choice lives entirely in the model built on top, not the encoder) with one
`PredictionHead` per task, each reading `layers[MARKOV_READOUT_DEPTH[task]][entity_type]`.
Returns the same `(logits, layers)` 2-tuple every other architecture returns, so
`ml/evaluate.py` needed zero changes. Registered in `ml/models/encoder.py` and `ml/train.py` as
a new architecture, `rgcn_attn_markov` (does not reuse `rgcn_attn` or `rgcn_attn_depthgate` —
separate logged rows).

---

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
consistent with the depth-gate pilot's timing at the same config — expected, since the Markov
model's encoder is byte-for-byte identical (same param count, 752,211, on both arms).

---

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| markov (fixed depth) | **0.8166 ± 0.0018** | 0.7963 ± 0.0026 | 0.9445 ± 0.0019 |
| baseline_retrain (this round) | 0.8095 ± 0.0025 | 0.7977 ± 0.0019 | 0.9357 ± 0.0082 |

**`reports/step6_depth_gate.md` reference (unpaired, NOT retrained here):**

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

---

## Plain verdict

| Task | Mean ΔAUC | Verdict |
|---|---|---|
| **delay** | **+0.0072** | **HELPS** — 5-seed consistent gain, free (zero new parameters) |
| shortage | −0.0014 | NO DETECTABLE DIFFERENCE — sign flips across seeds |
| impact | +0.0089 | NO DETECTABLE DIFFERENCE — sign flips across seeds (4/5 positive, 1 negative, largest single delta +0.0192) |

---

## Synthesis

The free, zero-parameter, theory-derived fixed-depth design **measurably helps delay** and is a
clean, statistically defensible win there: +0.0072 mean AUC, all 5 seeds positive, for literally
no added cost — no new parameters, no new training signal, just reading `h^1` instead of `h^4`
for one task. This reproduces `step6_preflight.md`'s original L1-vs-baseline delay finding
(+0.0069, CONSISTENT) essentially exactly, as expected since delay's readout here is the same
depth choice, now delivered via a per-task fixed index on the shared 4-layer trunk rather than
via a genuinely shrunk 1-layer encoder — and notably, unlike L1, it does **not** carry L1's
consistent impact cost (`step6_preflight.md`: −0.0160 CONSISTENT), because impact here still
reads the full `h^4` trunk output, identical to baseline's own readout for that task.

Shortage and impact show no reliable change in either direction — both flip sign across seeds
despite reading their theory/gate-predicted depths (`h^3`, `h^4`). For impact this is
expected and mechanically almost tautological: impact's fixed depth (`h^4`) is the same depth
baseline already reads for that task, so any observed difference between the two arms on
impact is measuring seed-to-seed training variance in the *shared trunk*, not a readout-depth
effect — the two arms are structurally near-identical for that one task. For shortage, `h^3` is
close to but not identical to baseline's structural prior, and the result is consistent with
`step6_preflight.md`'s own finding that shortage has shown no reliable depth sensitivity in any
sweep to date.

**Comparison against the learned gate** (`reports/step6_depth_gate.md`, unpaired reference):
the free fixed-depth design's delay result (0.8166) is noticeably *higher* than every gate arm's
delay mean (0.8080–0.8113) and higher than its own baseline_retrain in that round (0.8083) —
though these are two different rounds' baseline_retrain measurements (0.8095 here vs 0.8083
there), both well within the run-to-run noise band the depth-gate round itself measured
(std ≈ 0.0025–0.0090 per arm). shortage and impact land in the same range as the gate's own
numbers on both. The practical takeaway: for delay specifically, the free fixed-depth design is
at least as good as — and on this round's numbers, nominally better than — anything the
15-run, KL-anchored learned gate produced, at zero parameter cost and zero lambda tuning.

**Bottom line, directly answering the theory doc's own framing:** this Phase 1 (Markov-only)
design does what §7.1 predicted it would — it's the correct, defensible Phase-1 baseline,
free, and it captures the one clean win (delay) this project line has found repeatedly across
three separate methodologies now (the original 3-seed diagnostic, the 5-seed
`step6_preflight.md` confirmation, and this fixed-depth pilot). It does not move shortage or
impact in either direction, which is itself informative — per §6.3's own "asserted, not
measured" caveat, this result is exactly the L-sweep-style measurement the theory doc calls
for, and it settles that for this dataset at this label volume, shortage/impact's fixed-depth
choices are statistically indistinguishable from baseline, not that they are wrong.

---

## Governance record

Backed up before any write. 10 new registry rows — 5 `rgcn_attn_markov-markovpilot-seed{n}`
(architecture=`rgcn_attn_markov`) + 5 `rgcn_attn-markovpilot-baselinecmp-seed{n}`
(architecture=`rgcn_attn`), all `status='active'` — **242 total registry rows** (232 + 10).
**150 new evaluation rows** (10 runs × 15 metric rows) — **3,498 total** (3,348 + 150). No
existing row from any prior round modified. Full run log:
`reports/logs/run_markov_pilot_20260808_083523.log`.

---

*Scoped to SHARE (RGCN+Attention, `rgcn_attn`) only, per this round's own framing — no changes
to SHARP (`rgcn_relemb`) or SHARK (`rgcn_battn`). No learned gate, no Transformer component —
that's Phase 2, per "Markov Scoping and Transformer 1.md" §7.1's own sequencing, not attempted
in this pass.*
