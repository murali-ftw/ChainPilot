# Port Transformer 2 (+ Confidence / Trust-Gate / Cross-Attention Fusion) and Test It Where Hidden Dependencies Are Actually Causal

> Paste into a fresh session. Opus, high effort — this is a port plus a targeted sweep, same
> shape and rigor as the Layer 2 sessions (`STEP3_RUNG5_DEPTH_PROMPT.md`,
> `STEP3C_FINAL_LAYER2_REPORT_PROMPT.md`). Read `HADES_v1/reports/layer3.md` in full before
> starting — it is the entire prior history of this component and its final recommendation, and
> this session exists specifically to test the one precondition that report says was missing.

## Why this session exists

V1's Layer 3 (`HADES_v1/reports/layer3.md`) built Transformer 2 — same-type global attention
over Supplier embeddings, unconstrained by graph adjacency, meant to discover hidden
dependencies no message-passing architecture could ever reach (two suppliers secretly sharing
an unmodeled upstream parent). It worked, decisively, as a **discovery** mechanism: the planted
`H_POLYMER` scenario was recovered at 5.5–7.5x the chance rate, consistent across every seed and
every one of four fusion designs tried. **But discovery never once translated into a
sign-consistent AUC gain**, and V1 root-caused exactly why (`layer3.md` §2.1, "Prerequisite 0"):
`H_POLYMER` was a real correlation but a **causally redundant** one — a supplier's own stress
already fully captured its exposure, so nothing about a hidden partner's state carried
*incremental* predictive information. No fusion mechanism, however well built, could win on a
dataset where the thing being discovered doesn't causally matter. V1's own final line: **"What
would actually resolve this: a dataset built the way Task 3's dual-sourcing scenario was — an
explicit, incremental cross-supplier bleed-through term — applied to a Transformer-2-style
undirected/non-graph-edge relationship, not H_POLYMER as it currently exists."**

**V2 was built with exactly that gap in mind, confirm this against the spec before proceeding**
(`docs/00_Benchmark_Specification.md` §"Mechanism B — Hidden Shared Structure" and "Mechanism D —
Hidden Dependency Coupling"). Mechanism B now generates three explicitly distinguished group
types by the same structural process:

| type | coupling | expected behavior |
|---|---|---|
| **Type A** | `alpha > 0` via Mechanism D | discovery **should** improve prediction — hidden parent stress causally drives member risk |
| **Type B** | `alpha = 0`, correlated features | discovery should succeed; prediction should **not** improve — this is explicitly **V1's `H_POLYMER` behavior**, reproduced on purpose as the null control |
| **Type C** | `alpha = 0`, no correlation | decoy — a robust model should ignore this structure entirely |

This is the first time either project has had a hidden-dependency scenario that is genuinely,
causally informative (Type A) *and* a matched redundant control that reproduces the exact V1
null (Type B) *and* a pure decoy (Type C), all in the same dataset. That is the precondition
Prerequisite 0 said was missing. This session tests whether Transformer 2's validated discovery
capability finally converts into an accuracy win now that the dataset can reward it — and if it
still doesn't, that is a materially stronger finding than V1's, the same way Variant J's null
strengthened Layer 2's conclusion.

## What "Transformer 2 + Confidence / Trust-Gate / Cross-Attention" actually is, confirmed against the V1 code

- **Base retrieval mechanism** (`ml/models/transformer2.py`, `Transformer2GlobalAttention`) —
  same-type-only attention over Supplier embeddings, **no adjacency mask** (the entire point —
  it must reach pairs with no graph path between them), bounded candidate pool at **top-k=64 by
  cosine similarity** per Supplier. Shared, unmodified, by all four arms below.
- **Plain / original additive fusion** (`ml/models/rgcn_attn_variant_a_transformer2.py`,
  `rgcn_attn_variant_a_transformer2`) — **the arm to test first, and the one this prompt calls
  "plain initial version."** `z_impact_fused = z_impact + t2_scale * t2_output`, a single
  learned global scalar initialized to exactly 0 (numerically identical to plain Variant A at
  construction). 814,627 params at hidden=128/num_bases=10.
- **Confidence-aware fusion** (`ml/models/transformer2_confidence.py`, `rgcn_attn_t2_confidence`)
  — multiplies a **deterministic, zero-new-parameter** confidence score (mean top-64 cosine
  similarity + attention entropy, already computed internally by the base mechanism) into the
  same `t2_scale`. Same param count as the plain arm — confidence adds nothing to the parameter
  budget. **V1's chosen production default**, on cost/stability grounds, not an AUC win (nothing
  won on V1's data).
- **Per-node trust gate** (`ml/models/transformer2_trustgate.py`, `rgcn_attn_t2_trustgate`) —
  replaces the single global scalar with `trust_i = tanh(MLP(z_impact_i))`, MLP `hidden→32→1`,
  zero-initialized final layer, same bounded-residual convention as Variant A's own gate. +4,160
  params (818,787 total). V1 found this **highly unstable across seeds** — bimodal trust
  distributions, degree-correlation flipping sign — the same instability signature Layer 2's
  gates showed.
- **Cross-attention fusion** (`ml/models/transformer2_crossattn.py`, `rgcn_attn_t2_crossattn`) —
  self-contained top-64 retrieval (duplicated, not shared, to keep the arm isolated), impact
  embedding cross-attends over the retrieved pool via `nn.MultiheadAttention` (4 heads),
  zero-initialized output projection. +49,535 params (864,162 total), the most expensive arm and
  the one V1 found **least justified** — highest cost, weakest discovery-quality retention, no
  AUC compensation.
- **Shared dependency, do not modify:** all four arms reuse `Rung5VariantAGateHead` /
  `PredictionHead` **unmodified** for every task's own depth-selection and readout — this session
  changes nothing about Layer 1 (SHARE) or Layer 2 (Variant A), it is purely additive on top of
  both, exactly as V1 scoped it.

**One thing to re-verify before wiring anything, not assume from V1:** V1 wired Transformer 2 to
the **impact** head, not delay, because a live hop-distance measurement found the co-parent path
needs a hard minimum of 3 hops from Shipment (delay's target), which is out of reach at delay's
`h¹` readout, while impact's target (Supplier, hop 0) has the co-parent path in reach by hop 2 and
reads at `h⁴`. **Confirm this reasoning still holds on V2's topology before reusing it** — V2 has
Mechanism J's multi-tier supplier structure, which V1 never had, and it could change hop
distances for some variants (particularly Variant D itself, and K, which layers J on top of B/D).
If the hop-distance measurement comes out the same, keep the impact wiring; if it doesn't, say so
plainly and re-derive the correct head before training anything.

## 1. Port, ground-truth setup, and sanity-check

- Port all five files above into this repo's `ml/models/`, byte-identical (md5-verified against
  V1), same discipline as every prior port in this project.
- **Locate V2's Type A/B/C group membership** in `db/generate_dataset.py`'s Mechanism B
  implementation — confirm whether it's emitted anywhere in the observable CSVs (it should not
  be, per `verify_no_hidden_state()`) or exists only in the generator's internal state. If it's
  privileged-only, read it the same way `ml/extract_hidden_state.py` reads true resilience for
  Layer 2's diagnostic — a ground-truth-only read for validation purposes, never fed to a model.
  This is required before any discovery-quality check can be run; do not assume it works the same
  way `HADES_v1/ml/graph/hidden_dependency_ground_truth.py` did without confirming the V2
  equivalent exists and is structured comparably (group id, type, member list).
- Reproduce V1's recorded numbers on Variant 0 at the `v1` preset (byte-identical anchor), 5
  seeds, same configuration as V1's Round 1/Round 2:

  | arm | delay | shortage | impact |
  |---|---|---|---|
  | Variant A alone (no T2) | 0.8166–0.8167 | 0.7952–0.7974 | 0.9434–0.9436 |
  | plain T2 (additive) | 0.8144–0.8186 | 0.7991–0.7996 | 0.9425–0.9455 |
  | confidence | 0.8176 | 0.7995 | 0.9432 |
  | trustgate | 0.8185 | 0.7991 | 0.9438 |
  | crossattn | 0.8186 | 0.7992 | 0.9443 |

  (V1's Round 1 and Round 2 numbers differ slightly on the plain-T2 row because they were
  independently retrained — use the range, not a single point, as the tolerance band, same as
  Layer 2's sanity gate.)
- Reproduce V1's `H_POLYMER`-style discovery validation on whichever V2 dataset variant contains
  Type B groups (Variant B, α=0 for all types since D is inactive) — percentile rank and
  discovery rate against chance, same two metrics, same methodology as `layer3.md` §1.4/§2.6.
- **If the port doesn't reproduce V1's numbers, or the discovery check doesn't clear chance, stop
  and fix it before running anything new.**

## 2. Run the targeted sweep — Variant B and Variant D are the point, not all twelve

This session tests one specific hypothesis (does discovery convert to accuracy once the
discovered structure is causally coupled), so concentrate compute there:

- **Five arms: Variant A alone (no T2, the baseline), plain T2, confidence, trustgate,
  crossattn.**
- **On Variant B and Variant D, five dataset seeds each.** Variant B is the direct V1 replica —
  Type A groups exist but `alpha=0`, so it should reproduce V1's null exactly (discovery works,
  AUC doesn't move) and serves as this session's own negative control. Variant D is the new
  condition — Type A groups become causally coupled, and it is the one place in either project
  where a positive result is even possible.
- **If time allows, extend to Variant K** (all ten mechanisms, B+D+E+F together) as a stretch
  goal, reporting it against Variant D per the spec's own comparison rule
  (`docs/00_Benchmark_Specification.md` §"Type B under absorption... compare Variant K against
  Variant D") — Mechanism E/F's absorption is documented to suppress the co-degradation signal
  Type B's discoverability depends on, so K vs D tests whether that suppression also erodes any
  accuracy gain found on D.
- That's 5 arms × 2 variants × 5 seeds = 50 runs at minimum. No direct per-run timing exists yet
  for the Transformer 2 family on this codebase — by analogy to Rung 5's family (comparable
  parameter count, 765k–866k range for both), expect a similar **~20% overhead over the
  no-fusion baseline**, plus whatever the attention/retrieval step adds. **Time the first 2–3
  runs and confirm before committing to the full 50**, same discipline as every prior session in
  this project — do not assume V1's un-measured wall-clock carries over.

## 3. Report

Same table shape and discipline as `HADES_v1/reports/layer3.md` and this project's
`reports/layer2_testing.md`: mean ± std AUC over five seeds, positive count beside every AUC,
sign agreement on every delta, paired block-bootstrap significance against the no-T2 Variant A
baseline specifically (not against SHARE or Markov — the question here is whether fusing in
discovered structure beats not fusing it in, on the impact task it's wired to).

For every arm, on every variant: report both halves V1 always reported together and never let
collapse into one number —

- **Discovery quality** (percentile rank, discovery rate vs. chance, per Type A/B/C group where
  applicable) — does the retrieval mechanism still find the planted structure, on V2's dataset,
  under each fusion design.
- **Downstream AUC vs. the no-T2 baseline**, sign-consistency and significance, per task.

**The one number that answers this session's actual question:** on Variant D specifically, does
any fusion arm show a real, sign-consistent AUC improvement on impact that Variant B does not
show for the same arm? That is Prerequisite 0's prediction being tested directly for the first
time either project has been able to test it. Report it explicitly, and report a null with the
same honesty as every other null in this project if that's what happens — a confirmed null on
Variant D, despite genuine causal coupling being present, would be a materially stronger finding
than V1's, since it would rule out "the dataset never gave discovery anything real to find" as
the explanation for good.

For the trust-gate arm specifically, also reproduce V1's stability check (`layer3.md` §2.7): mean
and std of per-node trust, share positive/negative/near-zero, correlation of `|trust|` with
degree, and whether Type A group members receive systematically different trust than the
population at large — V1's version of this question used H_POLYMER; V2's Type A/B/C typing lets
this be asked properly for the first time, since there is now a real (Type A) group to compare
against the redundant (Type B) and decoy (Type C) ones in the same run.

## Deliverable

New report, `reports/layer3_testing.md`, same structure as `reports/layer2_testing.md` — a stated
answer up front, the port/sanity-check section, the per-variant results, the delta table against
the no-T2 baseline, the discovery-quality tables, the trust-gate stability section, and a closing
section stating plainly what this session does and does not settle about Transformer 2's
production readiness, at this label volume, same caveat every other result in this project
carries.
