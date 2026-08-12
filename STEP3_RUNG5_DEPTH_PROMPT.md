# Port Rung 5 (+ Variants A, B, C) and Test Adaptive Depth on V2

> Paste into a fresh session. Opus, standard-to-high effort — this is a port plus a targeted
> sweep, comparable in shape to the SHARP session, not a new design. Read
> `HADES_v1/reports/layer2.md` Round 4 and Round 5 in full, `HADES_v1/v2/layer2_contingency_plans.md`,
> and `reports/phase7_training_results.md` §6.2 (the paragraph that flagged this exact gap) before
> starting.

## Why this session exists

Phase 7 (`reports/phase7_training_results.md` §6.2) found Mechanism J's effect measurable
(chain-length heterogeneity makes delay slightly harder) but could not test whether an
*adaptive*-depth architecture actually benefits from that variation — every architecture ported so
far reads a **fixed** structural depth prior. V1 never had genuine chain-length variation to test
adaptive depth against in the first place; V2's Variant J is the first dataset where this question
is even askable. That's what this session answers.

## What "Rung 5 + Variant A/B/C" actually is, confirmed against the V1 code

- **Base Rung 5** (`ml/models/rgcn_attn_rung5_encoder.py`) — a prior-initialized MLP gate per node,
  learning a per-node depth distribution instead of the fixed h¹/h²/h³/h⁴ Markov floor every other
  architecture in this project uses. Cheaper than the alternative attention-based gate (Rung 4),
  which V1 already ruled out (15× the cost, no measurable benefit) — don't port Rung 4.
- **Variant A — bounded residual** (`ml/models/rgcn_attn_rung5_variant_a.py`) — **the production
  choice.** Caps how far the gate's learned correction can drift from the Markov prior. Solved
  Rung 5's seed-dependent instability completely (100% match rate, zero deviating nodes, every
  seed, every lambda, every task) at essentially no accuracy cost.
- **Variant B — structural features** (`ml/models/rgcn_attn_rung5_variant_b.py`) — feeds explicit
  structural signal (normalized degree, node-type one-hot, per-relation edge-count histogram) into
  the gate. Found a real, non-noise signal in V1 (nodes that deviate from the Markov prior are
  consistently lower-degree) but only partial improvement on the instability problem on its own.
- **Variant C — freeze + low-LR** — **not a separate model file.** It's a training-loop-only
  change in `ml/train.py::train_model` (freeze the gate early in training, unfreeze at a reduced
  learning rate), applied on top of the base Rung 5 architecture. Confirm this against V1's
  `ml/run_rung5_variant_pilot.py` before porting — the architecture id there is `rung5_c` using the
  plain Rung 5 encoder plus this training-loop flag, not a fourth model file. Came close to
  Variant A's stability result.
- **Do not port Variant D** (depth-preserving projection, showed no improvement) **or Variant E**
  (post-hoc weight-averaging, catastrophically failed on V1's data). Both are out of scope; porting
  either would be spending effort V1 already spent and answered negatively.
- **Shared dependency:** `ml/models/rung_gate_common.py`.

**One thing to reconcile before training anything:** `reports/phase7_training_results.md` states
the fixed structural depth prior used by SHARE/SHARP/SHARK's port as delay→h², shortage→h³,
impact→h³, while `HADES_v1/reports/layer2.md` records the original priors as delay→h¹,
shortage→broad/h³, impact→h⁴. Confirm which is correct for the Rung-5 family specifically — Rung
5's gate is *initialized* from this prior, so a mismatch here would mean the gate starts from the
wrong point and any comparison to Markov is invalid from the first epoch. Use V1's original
Round-3/4 source of truth for this value, not the phase7 port's summary, unless you can show
they're actually the same value described two different ways.

## 1. Port and sanity-check

- Port the four files above into this repo's `ml/models/`, byte-identical (md5-verified against
  V1), same discipline as the SHARE/SHARP/SHARK port.
- Reproduce V1's Round 4/Round 5 recorded numbers on Variant 0 at the `v1` preset (the
  byte-identical-to-V1 anchor, same one the SHARE sanity check used) — Markov baseline, Rung 5
  (ungated), and Variant A at minimum. Confirm parameter counts match V1's table exactly (198,546
  for Rung 4 is not relevant here; confirm Rung 5's own count, ~12,894 plus whatever A/B/C add).
- **If the port doesn't reproduce V1's numbers, stop and fix it before running anything on V2.**

## 2. Run the targeted sweep — not all twelve variants

This session tests one specific hypothesis (does adaptive depth help where chain length genuinely
varies), so don't spread compute across variants that don't touch depth at all. Run:

- **Markov (fixed-depth baseline, "baseline_retrain" in V1's naming), Rung 5 (ungated), Variant A,
  Variant B, and Variant C** — five arms.
- **On Variant 0 and Variant J**, five dataset seeds each. Variant 0 is the control (same
  environment SHARE and SHARP were already tested on, so results are directly comparable to
  Phase 7's existing tables). Variant J is the actual test — it's the one dataset where chain
  length genuinely varies per product rather than being structurally uniform.
- **If time allows after that**, extend to Variant A (the dataset variant, `J+truncation` —
  confusingly named the same as the model Variant A above; be explicit in all output about which
  "Variant A" is meant) as a stretch goal, since it's the pairing Phase 7 already established
  (report against Variant J).
- That's 5 arms × 2 variants × 5 seeds = 50 runs at minimum. Base cost estimate, by analogy to
  SHARE/SHARP's measured 419s/run at this scale: **~5.8 hours of raw CPU time, ~1.5 hours
  wall-clock at 4 parallel workers × 3 threads** — the setting already established as fastest for
  this scale in `reports/phase7_training_results.md` §5.1 and confirmed again in §8.1. Time the
  first 2–3 runs and confirm before committing to the full 50; Rung 5's gate machinery is small but
  adds real compute per forward pass, so don't assume it's free just because the base architecture
  is similar in size to SHARE.

## 3. Report

Same table shape and same discipline as `reports/phase7_training_results.md` §5.2/§5.3/§8.2/§8.3:
mean ± std AUC over five seeds, positive count beside every AUC, sign agreement on every delta,
paired block-bootstrap significance against the Markov baseline specifically (not against SHARE —
this is a different question, whether adaptive depth beats fixed depth, not whether it beats a
different architecture family entirely).

The one number that answers this session's actual question: **does any gated variant (A, B, or C)
show a real, sign-consistent AUC improvement over Markov on Variant J that it does not show on
Variant 0?** That delta-of-deltas is the first genuine test of adaptive depth this project has ever
been able to run — V1's dataset had no chain-length variation for it to exploit, which is exactly
why every one of V1's five variants landed on the same negative result on shortage and impact
(`v1_findings/v2.md` item 4). Report it explicitly, not just each variant's raw numbers, and report
it as a null with the same honesty as every other null in this project if that's what it turns out
to be — a second confirmed null on this specific question would itself be a real finding, since it
would mean the negative result wasn't just an artifact of V1's flat topology after all.
