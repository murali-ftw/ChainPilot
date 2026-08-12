# Layer 3, Retrieval Redesign — Contrastive Ceiling, Then Future Co-Failure, Gated

> Paste into a fresh session. Opus, high effort — this replaces the retrieval mechanism, not the
> fusion layer, and gets it wrong once already cost a full session (`reports/layer3_testing.md`).
> Read `reports/layer3_testing.md` in full before starting, especially §2 (the discovery failure)
> and §7 (what would resolve it). This session is a **decision tree with hard stops, not a sweep**
> — later stages are conditional on earlier ones, and skipping a stop to "save time" defeats the
> point of the design.

## Why this session exists

`reports/layer3_testing.md` found Transformer 2's retrieval — same-type cosine similarity over
SHARE's post-encoder Supplier embeddings — cannot separate V2's real hidden-parent groups (Type
A/B) from a pure decoy (Type C) at all, at chance in every metric, at both scales tested. The
report's own diagnosis: the embeddings being compared were never trained to encode co-membership
— they're shaped entirely by delay/shortage/impact loss, with zero pressure to preserve the
signal retrieval needs. Meanwhile the **generator's own statistical validation check**, run
directly on observable event-timing data with no model involved, **does** detect Type B
co-degradation at mid scale (`reports/layer3_testing.md` §2.2, mean delta −0.0924 against a null
band of [−0.0393, +0.0429] — a clean pass). The signal exists and is measurable by simple means;
the retriever was never pointed at it.

This session tests whether pointing retrieval at that signal — first with privileged supervision
to find the ceiling, then without it, the way a real deployment would have to work — closes the
gap the last session found.

## The two things being tested, and the order matters

**Stage 1 — Contrastive Retrieval (the ceiling test).** Add an auxiliary contrastive loss on top
of SHARE's existing training objective: pull true co-member embeddings together, push random
pairs apart, using `HP_GROUPS` (already read privileged, non-model-input, by
`ml/extract_hidden_state.py`) as the **training-time-only** supervisory signal — the loss target,
never a feature. **State this plainly in the report as a disclosed deviation, the same way this
project has flagged every prior deviation** (the T_START/T_END fix, the impact-wiring
re-derivation): every earlier privileged read in this project (true resilience, `HP_GROUPS` for
discovery-quality checks) was eval-only. This is the first time privileged ground truth would
shape gradients. That is a legitimate, useful experiment — it answers "can this representation
even encode the signal if explicitly taught to" — but it is a different, easier claim than
emergent discovery, and the report must say so in the same sentence it reports the result, not as
a footnote.

Train positives from **Type A and Type B members only** (both are real, structurally-generated
groups) against negatives drawn from **Type C plus random non-member pairs**. Use an InfoNCE-style
contrastive term, weighted against the main task loss, zero-initialized contribution at
construction per this project's standing convention (every new component starts identical to the
baseline it extends).

**Stage 2 — Future Co-Failure Retrieval (the realistic version), only after Stage 1 reports.**
Build a retrieval signal from **observable data only** — no `HP_GROUPS`, no privileged read of any
kind at inference or training. Construct a per-supplier time-series of observable stress/event
signals (shipment delay incidence, snapshot-level risk indicators — confirm exactly which
observable columns the generator's own §2.2 co-degradation check actually used, and build the
retrieval feature from the same or an equivalent observable proxy, not a different one guessed at
independently) and rank/embed suppliers by correlation of that history.

**Hard leakage guardrail, non-negotiable, matching this project's own established discipline**
(Mechanism G's `recorded_at` vs `changed_at` split, `verify_no_hidden_state()`): for any
prediction made as of time `t`, the co-failure correlation feeding that prediction's retrieval
must be computed **only from observable events with timestamp before `t`**. Write an explicit
test that would fail if this were violated (e.g., shuffle post-`t` events and confirm retrieval
output is unchanged) before trusting a single downstream number. This project has been burned by
exactly this class of bug once already; do not re-derive the lesson the hard way a second time.

## Evaluation — retrieval quality, measured properly, before any downstream number is trusted

**Extend `ml/analyze_transformer2.py` with three more standard IR metrics** alongside the existing
percentile rank and discovery rate: **Recall@K, Precision@K, and Mean Reciprocal Rank**, computed
per group type (A/B/C), per variant, per seed, the same disaggregation discipline every other
metric in this project already uses.

**Measure a reproduction floor for these retrieval metrics before trusting any of them** — this
was the gap in the last session, which measured a floor for downstream AUC (§4, ~0.003 mean,
0.008 worst case) but never for retrieval quality itself. Extend `ml/reproduction_floor.py` (or
add a parallel script) to retrain 8–16 identically-configured pairs and report how much percentile
rank, discovery rate, Recall@K, and MRR move with nothing changed. **A Stage 1 or Stage 2 result
that doesn't clear this floor by a healthy margin is not a finding**, exactly the standard applied
to every AUC delta in `reports/layer3_testing.md`.

**Confirm before assuming: does the Type A/B co-failure confound the audit raised actually apply
to Variant B and D specifically?** Repeated co-failure could in principle come from a shared
hidden parent (what this session wants to find) or from an unrelated shared shock (Mechanism H) or
rewiring event (Mechanism C) — a real ambiguity in general, but Variant D is Base + Mechanism B +
Mechanism D only; confirm neither H nor C is active in the B/D generation path before assuming
this confound is live for the datasets actually being trained on here. If it is not active,
say so and move on; if it is, the co-failure signal needs to be checked against group membership
specifically, not just against any correlated pair.

## Hard stops — read this section before running anything

**Stop after Stage 1. Do not proceed to Stage 2 until Stage 1's ceiling is measured against its
own reproduction floor.** If contrastive retrieval — with privileged supervision, the easiest
possible version of this problem — cannot clear chance by more than its own reproduction floor,
**stop the entire line of investigation and report that plainly.** That would mean SHARE's
Supplier representation lacks the structural information this task needs regardless of how
retrieval is trained, which is a materially different and more important finding than "this
particular retriever didn't work," and no further stage in this session should be built on top of
a representation shown incapable of the task.

**Stop after Stage 2. Do not build Stage 3 or Stage 4 until Stage 2 is compared against both
chance and Stage 1's ceiling.** Three outcomes, three different next actions:

- **Stage 2 clears chance by a real margin (sign-consistent across seeds, outside the
  reproduction floor) and approaches Stage 1's ceiling** — proceed to Stage 3 (Relationship
  Attribution) and Stage 4 (reintroduce Confidence-Aware Fusion on top of the new retriever).
  Scope Stage 3's classification target to what the generator actually labels — **Type A vs. B
  vs. C**, not the richer unlabeled taxonomy (shared parent / disaster / logistics hub / etc.)
  earlier drafts of this idea proposed, which have no ground truth in this dataset at all.
- **Stage 2 clears chance but stays meaningfully below Stage 1's ceiling** — do not build Stage 3
  or 4 in this session. Note Multi-View Retrieval (Stage 5) as the well-motivated next step and
  stop here; it is a real follow-up, not this session's job, per the established cost discipline
  of timing before committing to a new build.
- **Stage 2 does not clear chance** — stop. Report a clean, honest null: the signal is teachable
  under privileged supervision (or isn't, per the Stage 1 gate above) but not recoverable from
  observable data alone at this label volume. Do not build Stage 3, 4, or 5.

**Out of scope for this session entirely: Causal Retrieval Network (Stage 6).** Per the audit,
it collapses into Stage 1 or Stage 2 as currently described and needs a concrete causal-inference
mechanism (temporal precedence, intervention sensitivity, counterfactual consistency, causal graph
estimation, or Granger-style influence) defined before any code is written. Note it as blocked,
pending a future definition session, and do not attempt to build a version of it here.

## Cost discipline

Same as every prior session: time the first 2–3 runs of Stage 1 before committing to the full grid
(5 seeds × Variant B and D, mid scale `sup_n=2,000`, matching `reports/layer3_testing.md`'s
configuration so results are comparable). Confirm the two concurrent-process memory ceiling found
last session (§6: four processes on `sup_n=2,000` OOM at 24GB; two concurrent is both safe and
faster) still holds before scheduling the sweep.

## Deliverable

Append a new section to `reports/layer3_testing.md` — **§9, Retrieval Redesign** — not a new file;
this is a direct continuation of that report's own §7 recommendation. Structure it around the
stages as run (not all of them will run, per the gates above): Stage 1's ceiling and its
reproduction floor, the go/no-go decision and why, Stage 2's result if reached and its comparison
against both chance and the ceiling, the leakage-guardrail test and its result, and — only if
reached — Stage 3/4's downstream numbers against the AUC reproduction floor already established
in §4. End with one paragraph stating plainly which of the three stop conditions above was hit and
what that does and does not settle about Transformer 2's retrieval mechanism, at this label
volume, same caveat every other result in this project carries.
