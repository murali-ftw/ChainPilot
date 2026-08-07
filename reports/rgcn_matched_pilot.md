# HADES Model-Development Prototype — SHARE vs SHARP, Matched-Parameter Pilot

*Display names, added retroactively for consistency with later documentation: RGCN+Attn
("Option 1", `rgcn_attn` in code) is **SHARE**; RGCN+relemb ("Option 3", `rgcn_relemb` in code)
is **SHARP**. The code-level identifiers this report was written against are unchanged.*

**Scope:** train RGCN+attn (**SHARE**, "Option 1", `reports/rgcn_attn_pilot.md`'s architecture) at a
matched parameter budget for the first time, and build + train a new sixth architecture —
RGCN+relation-embedding-attn (**SHARP**, "Option 3") — also at a matched budget, both compared against
HGT. **HGT was not retrained**; its existing fixed-d=64 rows from the rgcn_attn pilot were
reused directly (fixed-d and matched-d are identical for HGT by construction — hidden=64 is
the anchor for both arms). 5 seeds × 2 architectures = 10 new training runs, against the same
v3 dataset (800 suppliers, 15 monthly snapshots Jul 2024 – Sep 2025) confirmed live before any
code was written. This is an addendum to `reports/rgcn_attn_pilot.md`, not a replacement —
that report's fixed-d=64 numbers are unchanged and reused here for the before/after
comparison.

---

# Part I — Conclusions

## Bottom line

**RGCN+attn's fixed-d win survives matching parameters almost unchanged — this was not a
smaller-model-regularizes-better artifact.** At 702,288 encoder params (0.17% off HGT's
701,088 anchor, vs. 170,464 at fixed-d — a >4× increase), RGCN+attn's per-task AUC moved by
less than 0.0015 in either direction on every task relative to its fixed-d pilot numbers. It
still beats HGT consistently on delay and shortage (both new comparisons hold sign across all
5 seeds) and ties on impact — the same qualitative pattern the fixed-d pilot reported, now
confirmed at a parameter count no longer confoundable with HGT's own. **RGCN+relemb (SHARP,
"Option 3"), tested here for the first time, does not clearly improve on RGCN+attn (SHARE)** — the paired
head-to-head comparison is a clean statistical tie on all three tasks (every delta flips sign
across seeds) — and it beats HGT consistently on only one task (shortage) rather than two. The
extra relation-identity signal SHARP adds costs more parameters (388 more at matched hidden,
requiring `hidden`/`num_bases` to be re-tuned to stay near HGT's anchor) without a measurable
return on this dataset.

## Phase E — matched-parameter search, both architectures

HGT's real encoder-only anchor, recomputed live: **701,088** (unchanged from every prior
round — feature dims haven't shifted). Per the task's own instruction, three candidates were
found for each architecture and only candidate (b) — `num_bases ≥ 8` preserved — was trained,
specifically to avoid repeating the 4-arch round's finding that `num_bases` shrinkage hurts
RGCN-family impact scores.

| Architecture | Candidate | hidden | num_bases | Encoder params | Diff from anchor |
|---|---|---:|---:|---:|---:|
| rgcn_attn | (a) tightest raw fit | 142 | 2 | 701,102 | 14 (0.00%) |
| rgcn_attn | **(b) num_bases≥8 — TRAINED** | **128** | **10** | **702,288** | **1,200 (0.17%)** |
| rgcn_attn | (c) intermediate | 140 | 3 | 701,328 | 240 (0.03%) |
| rgcn_relemb | (a) tightest raw fit | 138 | 4 | 701,102 | 14 (0.00%) |
| rgcn_relemb | **(b) num_bases≥8 — TRAINED** | **128** | **10** | **702,676** | **1,588 (0.23%)** |
| rgcn_relemb | (c) intermediate | 136 | 5 | 699,672 | 1,416 (0.20%) |

Both architectures land on the identical (hidden=128, num_bases=10) under the num_bases≥8
constraint — unsurprising, since SHARP's extra relation-embedding parameters
(`relation_embed_dim=16`, unchanged/not searched) are a small, roughly hidden-independent
addition on top of SHARE's own parameter curve.

## Parameter counts (full model, incl. 3 task heads)

| Architecture | Params | vs HGT |
|---|---:|---:|
| hgt (fixed-d=64, reused) | 713,763 | — |
| rgcn_attn-matched (hidden=128, num_bases=10) | 752,211 | +5.4% |
| rgcn_relemb-matched (hidden=128, num_bases=10, relation_embed_dim=16) | 752,599 | +5.5% |

## AUC — mean/std/min/max across 5 seeds

| Arm | delay | shortage | impact |
|---|---|---|---|
| hgt (fixed-d, reused) | 0.8015 ± 0.0106 [0.7834, 0.8132] | 0.7824 ± 0.0057 [0.7745, 0.7920] | 0.9335 ± 0.0022 [0.9307, 0.9372] |
| rgcn_attn-matched | **0.8118** ± 0.0035 [0.8064, 0.8163] | **0.7971** ± 0.0024 [0.7954, 0.8018] | **0.9387** ± 0.0051 [0.9304, 0.9445] |
| rgcn_relemb-matched | 0.8105 ± 0.0031 [0.8070, 0.8156] | 0.7991 ± 0.0010 [0.7978, 0.8003] | 0.9369 ± 0.0047 [0.9301, 0.9414] |

RGCN+attn has the highest mean AUC on delay and impact; RGCN+relemb edges it very slightly on
shortage (0.7991 vs 0.7971 — well within both architectures' own seed-to-seed spread, and the
paired test below shows this specific gap flips sign across seeds).

## Before/after: RGCN+attn (SHARE), fixed-d=64 pilot vs this round's matched-d

| Task | fixed-d=64 (170,984 params) | matched-d (752,211 params) | Δ (matched − fixed) |
|---|---|---|---:|
| delay | 0.8127 ± 0.0035 [0.8084, 0.8161] | 0.8118 ± 0.0035 [0.8064, 0.8163] | −0.0009 |
| shortage | 0.7985 ± 0.0023 [0.7953, 0.8017] | 0.7971 ± 0.0024 [0.7954, 0.8018] | −0.0014 |
| impact | 0.9393 ± 0.0098 [0.9261, 0.9496] | 0.9387 ± 0.0051 [0.9304, 0.9445] | −0.0006 |

**This is the direct answer to "did the fixed-d win survive matching parameters."** All three
deltas are under 0.0015 in magnitude — indistinguishable from seed noise, and if anything
impact's seed-to-seed std *tightened* at matched-d (0.0098 → 0.0051). RGCN+attn's fixed-d
result was not an artifact of having a quarter of HGT's parameter count regularizing better on
a label-scarce dataset; the same architecture, at ~4.4× more parameters, performs essentially
identically.

## Sign-consistency — RGCN+attn-matched (SHARE) vs RGCN+relemb-matched (SHARP) (PAIRED bootstrap)

Both trained in-process this round, so a genuine paired bootstrap (`ml/evaluate.py::
paired_delta_auc_ci`) applies, same method as every other multi-architecture round.

| Task | mean ΔAUC (attn − relemb) | Verdict |
|---|---:|---|
| delay | +0.0013 | FLIPS (noise) |
| shortage | −0.0020 | FLIPS (noise) |
| impact | +0.0018 | FLIPS (noise) |

**Statistically tied on all three tasks.** Neither hybrid design beats the other at matched
parameters on this dataset.

## Sign-consistency vs HGT (UNPAIRED — methodology caveat)

HGT was not retrained, so its raw per-sample test-set predictions from the original pilot run
aren't available to this script (no model checkpoint is persisted anywhere in this codebase).
This comparison instead uses each side's own independently-computed single-model CI (new
arch's CI from this run; HGT's stored CI, both `ci_method='row_bootstrap'`) and reports
per-seed point-delta sign-consistency — a real but strictly weaker test than the paired method
above, since it can't cancel shared test-sample noise the way pairing does. Every individual
seed's two independent CIs overlap (unsurprising given how much wider two-independent-CI
comparisons are than a paired one), so no single seed reaches significance on its own — the
signal here is entirely in the **sign-consistency across all 5 independent training runs**,
which is still meaningful evidence on its own terms.

| Comparison | delay | shortage | impact |
|---|---|---|---|
| rgcn_attn-matched vs hgt | +0.0103 **CONSISTENT** (attn wins) | +0.0147 **CONSISTENT** (attn wins) | +0.0052 FLIPS (tied) |
| rgcn_relemb-matched vs hgt | +0.0090 FLIPS (tied) | +0.0167 **CONSISTENT** (relemb wins) | +0.0033 FLIPS (tied) |

## Verdict — did each option's advantage hold, shrink, or reverse at matched parameters?

**SHARE (Option 1, RGCN+attn): held, essentially unchanged.** Its fixed-d pilot win over HGT on delay
and shortage (both consistent across 5 seeds) reproduces at matched-d with the same sign and
comparable magnitude (delay +0.0103 vs. fixed-d's own +0.0112 vs. HGT; shortage +0.0147 vs.
fixed-d's +0.0162) — a small narrowing in both cases, well within what 5-seed noise would
produce, not a meaningful shrink. Impact remains a tie, as it was at fixed-d. **RGCN+attn
beats HGT outright on 2 of 3 tasks at a parameter count no longer confoundable with HGT's own**
— the strongest, most parameter-fair result in this project's whole architecture-ablation
line of work.

**SHARP (Option 3, RGCN+relemb): a real but smaller win than SHARE, not an improvement over it.**
With no fixed-d baseline of its own (this is its first test), RGCN+relemb beats HGT
consistently only on shortage — the one task where architecture-level advantage over HGT has
now been replicated by *three* different mechanisms in this project (plain RGCN, RGCN+attn,
RGCN+relemb). It does not clearly beat RGCN+attn on any task (paired comparison flips on all
three), meaning the added relation-identity signal — at a real parameter cost — is not earning
its keep on this dataset. The simpler, cheaper SHARE remains the better default.

**Does either beat HGT outright once parameter count is no longer a confound?** Yes, on
specific tasks, not universally: RGCN+attn beats HGT on delay and shortage (impact tied);
RGCN+relemb beats HGT on shortage only (delay and impact tied). Neither loses to HGT on any
task at matched parameters. HGT's own strongest result — impact — remains undefeated by every
architecture tried across every round of this project to date.

---

# Part II — Findings by Phase

## Phase A — `RGCNRelEmbAttnEncoder` implementation

`ml/models/rgcn_relemb_encoder.py` (new) extends `RGCNAttnEncoder` verbatim (`lin_in`,
`rel_basis`/`rel_coeff`, `W_r` computation, `self_loops`, `att_msg`/`att_dst`, the `by_dst`
grouping, and the same single joint softmax per destination node) and adds one more additive
term to the attention logit: a per-relation embedding (`rel_embed`,
`[num_relations, relation_embed_dim]`) fed through one more shared scorer (`att_rel`,
`Linear(relation_embed_dim, 1)` per layer), computed once per relation per layer (not per
edge, since it depends on neither the message nor the destination state) and broadcast across
that relation's edges before the same joint softmax runs. Cost:
`O(num_relations × relation_embed_dim)` for the embedding table plus
`O(relation_embed_dim)` per layer for `att_rel` — a lookup table, not a second
basis-decomposed weight pool (Option 2, not built this round — later built and named
**SHARK**, see `reports/rgcn_types.md`).

## Phase B — Factory wiring

`"rgcn_relemb"` added to `ARCHITECTURES` and dispatched in `build_encoder`
(`ml/models/encoder.py`), threading both `num_bases` and the new `relation_embed_dim` (default
16) through, same import-inside-branch pattern as every prior addition. `relation_embed_dim`
also threaded through `HADESModel` and `train.py`'s `train_model`/`run_training_job` so it can
be set per run; recorded in `hyperparameters` JSON (`null` for every non-`rgcn_relemb`
architecture). Fixed a latent gap while touching this code: `num_bases` logging in
`hyperparameters` previously only fired for `architecture == "rgcn"`, silently omitting it for
`rgcn_attn` even though that architecture uses it too — now logged for all three
basis-decomposition architectures (`rgcn`, `rgcn_attn`, `rgcn_relemb`).

## Phase C — Tests

`"rgcn_relemb"` added to the parametrized forward-pass test. Two new tests:
`test_rgcn_relemb_parameter_count_formula` locks in the exact parameter delta over
`RGCNAttnEncoder` (`num_relations × relation_embed_dim + relation_embed_dim × num_layers +
num_layers`); `test_rgcn_relemb_attention_sums_to_one_per_destination` (adapted from the
`rgcn_attn` version) confirms the extra additive `rel_term` doesn't change the joint softmax's
normalization scope — weights still sum to 1 across a destination node's entire incoming edge
set, spanning every relation. Full suite: **46/46 passing** (43 pre-existing + 3 new).

## Phase D — Smoke-verification against the live dataset

Live DB check confirmed v3 before running. Built one real snapshot, ran a single forward pass
through `RGCNRelEmbAttnEncoder` at default hyperparameters: shapes matched every other
encoder, no NaNs/Infs, and the real encoder parameter count (171,372) exactly matched
`rgcn_attn`'s (170,984) plus the expected 388-parameter increase (`20 × 16 + 16 × 4 + 4`) —
confirmed against real data, not just the small unit-test fixture.

## Phase E — Matched-parameter search

See Part I for the full three-candidate table and the reasoning behind choosing candidate (b)
for both architectures.

## Phase F — Training

Pre-flight: live dataset check (v3, unchanged) and unconditional `pg_dump` backup
(`reports/backups/model_registry_and_evals_backup_20260807_164812.sql`, 136 + 1,908 rows
before this round). New file `ml/run_rgcn_matched_pilot.py` (touches no existing run script)
trained 5 seeds × 2 architectures (rgcn_attn-matched, rgcn_relemb-matched) = 10 runs; HGT's
existing 5 fixed-d rows were queried directly from `model_evaluation_runs`/`model_registry` and
never retrained. Completed in **0.73h (2,632s)** wall-clock, zero errors, logged in full to
`reports/logs/run_rgcn_matched_pilot_20260807_170239.log`. Verified post-run: 10 new
`model_registry` rows (5 per architecture, all `status='active'`, suffixed
`-rgcnmatched-pilot-seed{n}`) and 150 new evaluation-run rows, with the pre-run counts (136
registry / 1,908 evaluation) intact underneath (146 / 2,058 after). No prior round's row was
touched.

## Governance record

Backed up before any write
(`reports/backups/model_registry_and_evals_backup_20260807_164812.sql`). 10 new
`-rgcnmatched-pilot-seed{n}` registry rows (5 rgcn_attn / 5 rgcn_relemb), all `status='active'`
— 146 total registry rows (136 + 10). 150 new evaluation rows — 2,058 total (1,908 + 150). No
existing row from any prior round modified.

## Test suite

`ml/tests/` — **46/46 pytest pass** after all changes (Phase C above).

## Out of scope (per this round's own framing, not attempted)

No GraphSAGE/GAT re-runs. No documentation updates. Option 2 (basis-decomposed attention, the
third and most expensive of the RGCN-attention hybrid designs — later built and named
**SHARK**) not implemented.

---

*This addendum does not modify `reports/rgcn_attn_pilot.md`'s fixed-d=64 numbers or
conclusions — they're reused here as-is for the before/after comparison. Its own headline
finding (RGCN+attn wins on mean AUC across all three tasks at fixed-d) is now confirmed to
survive matched parameters, and a sixth architecture (RGCN+relemb) has been added to the
roster without displacing RGCN+attn as the strongest candidate so far.*
