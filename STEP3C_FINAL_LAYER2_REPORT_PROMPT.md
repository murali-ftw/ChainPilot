# Layer 2, Consolidated — Diagnostic, Lambda Sweep, A+C Combination, Final Report

> Paste into a fresh session. Opus, high effort throughout. Read `reports/layer2_testing.md` in
> full, `v1_findings/layer2_contingency_plans.md`, and `HADES_v1/reports/layer2.md` Round 4/5
> before starting. This session has three phases in a deliberate order — do not run Phase 2 or 3
> until Phase 1 is done and its findings have been read, because Phase 1's results determine how
> Phase 2 should be scoped.

## Phase 1 — The gate diagnostic

Everything in `STEP3B_GATE_DIAGNOSTIC_PROMPT.md` (this repo's root) — run it as written. Four
questions: how often does the gate disagree with Markov, which nodes are affected when it does,
do those nodes share properties (degree, chain length, resilience — true hidden value and
observable proxy kept separate), and are the depth representations actually different enough to
matter in the first place. ~20 runs, persist gate weights this time (the original sweep didn't).

**Stop and read the results before proceeding.** Two specific findings change what Phase 2 should
do:

- **If disagreement rate is at or near zero:** the lambda sweep in Phase 2 is well-motivated and
  should be the priority — it directly tests whether the cap is simply too tight to let the gate
  explore at all.
- **If question 4 finds the depth representations are barely distinguishable from each other**
  (high cosine similarity across h⁰–h⁴, especially on the node types that matter most for delay/
  shortage/impact): **say so plainly and scale back Phase 2's ambition accordingly.** A wider
  lambda cap cannot produce a benefit from choosing among representations that are nearly
  identical to begin with. In that case, still run a smaller confirmatory lambda sweep (2–3 values,
  not the full grid below) to check the finding holds under a wider cap too, but don't spend the
  full compute budget chasing a lever that the diagnostic already said can't work — report why,
  briefly, and move to Phase 3.

## Phase 2 — Lambda sweep and A+C combination

Only after Phase 1 is read.

**Lambda sweep.** The original sweep only tested λ=0.3 (V1's mid-range default). Test a real
range — something like 0.1, 0.3 (reuse existing data, don't retrain it), 0.6, 1.0, and one
annealed schedule (start at 0.1, grow linearly to 0.6 over training) — on Variant A specifically,
on both Variant 0 and Variant J, five seeds each. Report, per lambda value: AUC vs. Markov (same
format as `reports/layer2_testing.md` §6), **and disagreement rate from Phase 1's instrumentation**
— this is the one addition that makes this sweep more informative than a blind rerun: you can now
see directly whether a wider cap actually increases how often the gate deviates, not just whether
AUC moves. A cap that widens without disagreement rate moving would itself be a finding (the gate
isn't cap-constrained, something else is holding it near the prior).

**A + C combined.** One new arm: Variant A's bounded residual, with Variant C's freeze-then-
unfreeze-at-low-LR training schedule applied on top. Same two variants, five seeds. Capture gate
weights here too, same instrumentation as Phase 1 — cheap to include now that the persistence
code exists, and it answers whether combining the two stability fixes changes disagreement rate or
just further suppresses it.

Budget: lambda sweep is roughly 4 new lambda values × 2 variants × 5 seeds = 40 runs; A+C is
1 arm × 2 variants × 5 seeds = 10 runs. ~50 runs, in the same range as the original full sweep —
expect a similar wall-clock (the original took 1.8h nominal, 3.9h with the one-time stall). Time
the first few runs and confirm before committing to the rest, same discipline as every prior
session.

## Phase 3 — Final consolidated report

One report, not four scattered ones. Append a new top-level section to `reports/layer2_testing.md`
titled "Consolidated Findings — Why Adaptive Depth Doesn't Help (or: What Would Need to Be True)"
that pulls together, in one place:

1. **The original null** (already in §5–§7): no gated variant beat Markov, delta-of-deltas on
   Variant J flips sign on all twelve cells.
2. **The diagnostic's answer to all four questions**, stated plainly, in order.
3. **The lambda sweep's answer**: does widening or annealing the cap change disagreement rate,
   and does it change AUC — report both, since they can move independently and that combination is
   itself informative (cap-bound but no benefit vs. cap-bound and unable to move vs. free to move
   but no benefit are three different findings requiring three different follow-ups).
4. **The A+C result.**
5. **One final paragraph, written for someone who will never read the other sections**, answering:
   given everything measured across every Layer 2 session on V2, is there any remaining reason to
   believe a per-node adaptive depth mechanism could help this benchmark's three tasks, and if so,
   what specifically would have to be different (more label volume, a different gate design, a
   dataset where the signal is deliberately placed further upstream) — or state plainly that the
   question is closed at this scale with this design space, pending only the spec-scale rerun every
   other null in this project is already waiting on.

Do not soften a null into "inconclusive" if the diagnostic actually explains it (e.g., near-zero
disagreement rate is a *cause*, not an absence of evidence) — the whole point of Phase 1 was to
stop guessing at why and start measuring it.
