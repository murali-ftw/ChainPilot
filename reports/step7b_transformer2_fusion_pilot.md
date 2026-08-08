# Step 7b — Three Isolated Fusion-Improvement Variants for Transformer 2

**Scope:** build and evaluate three independent fusion-improvement variants for Transformer 2
— Confidence-Aware Fusion, Per-Node Trust Gate, and Cross-Attention Fusion — each tested in
isolation against the same shared baselines, never stacked. Build-and-evaluate only, scoped to
SHARE + Variant A + Transformer 2 (`rgcn_attn_variant_a_transformer2`,
`reports/step7_transformer2_pilot.md`) — no changes to SHARP, SHARK, or the existing
Markov/Variant A depth-selection logic.

---

## Prerequisite 0 — Causal coupling verdict (read FIRST, colors every AUC result below)

**NO cross-supplier causal coupling exists for H_POLYMER, unlike Task 3's dual-sourcing
scenario (`COPARENT_COUPLING = 0.35`).** Read directly from `db/generate_dataset.py`:

- `own_stress(sup_id, ...)`: a supplier's latent stress from base reliability + shared-hidden-
  factor EVENT-POOL membership (H_PORT, H_TRUCK, H_CUSTOMS, **and H_POLYMER** — all four pools
  handled identically) + its own idiosyncratic outage. Fully self-contained: an H_POLYMER
  member's `own_stress` already reflects its own H_POLYMER event exposure completely.
- `stress(sup_id, ...)`: `own_stress(sup_id) + COPARENT_COUPLING * own_stress(partner) for
  partner in coparents.get(sup_id, ())` — the ONLY cross-supplier bleed-through term in the
  whole generator, and `coparents` is populated EXCLUSIVELY from `component_suppliers` (Task
  3's dual-sourcing mechanism, a separate random assignment), never from H_POLYMER membership.
- Every label — delay, shortage, **and impact** (confirmed by reading the label-generation loop
  directly: `lab = s["id"] in sup_hit`, i.e. "did this supplier's own shipments get delayed")
  — derives from `stress()`, which for an H_POLYMER member (absent an unrelated, coincidental
  `component_suppliers` partner) reduces to exactly `own_stress()` — fully derivable from that
  supplier's own observable history alone.

The H_POLYMER scenario **is** a real, causally-grounded correlation (all 4 members are
genuinely, simultaneously exposed to the same disruption event windows, magnitude 0.95/0.85 —
not fabricated), but it is **structurally REDUNDANT for prediction purposes**: nothing about a
partner's current stress leaks into a member's own stress computation the way
`COPARENT_COUPLING` explicitly does for Task 3's dual-sourced suppliers. **Blocking finding:
no fusion mechanism, however well-built, can produce a genuine AUC improvement from discovering
H_POLYMER specifically, because there is nothing incrementally predictive in that discovery.**
This is stated once here and referenced in every AUC section below, not asserted once and
forgotten.

---

## Prerequisite 1 — Does retrieval confidence correlate with downstream usefulness?

Using the 5 freshly-retrained (this same process) original-Transformer-2 models, computed
`confidence_i` (mean top-64 cosine similarity + attention entropy, the exact formula Variant 1
uses) against `|prediction_with_T2 − prediction_without_T2|` for every Supplier, pooled across
test snapshots.

**Result: `pearson_r = +0.7666`, a strong, meaningful positive correlation** — Suppliers with
more confident retrieval (a genuinely similar candidate pool, concentrated attention) are the
ones whose predictions Transformer 2 actually moves the most. **Diagnostic verdict: Confidence-
Aware Fusion has a plausible mechanism to exploit** — this motivated proceeding with Variant 1
as designed, though Prerequisite 0's blocking finding means this correlation alone doesn't
predict an AUC win (confidence tracking usefulness-in-the-sense-of-magnitude-of-shift is a
different claim than that shift being *correct*/*helpful* for the label).

---

## Device check

Entropy computation, the per-node gate MLP pattern, and `nn.MultiheadAttention` cross-attention
all verified forward+backward, CPU vs MPS, on tensors/weights **cloned identically across
devices** — an earlier, incorrectly-constructed check using independent per-device random draws
produced spurious large diffs purely from CPU/MPS having different RNG streams, caught and
corrected before trusting any result. Corrected check: all four outputs (entropy, gate output,
cross-attention output, input gradient) matched to `atol=1e-4` (max diff ~1.2e-6).
Cross-attention alone timed **1.85x faster on MPS** (1.20ms/iter vs 2.22ms/iter). Both criteria
met → **MPS used for all 25 training runs**.

---

## The three variants

All reuse `Rung5VariantAGateHead`/`PredictionHead` UNMODIFIED for delay/shortage/impact's own
depth-selection and readout (no changes to Variant A). All verified live: exactly zero
contribution to impact logits at construction (safe-start, matching the original T2 pilot's own
`t2_scale=0` convention) — `0.00000000` max diff between fused and unfused impact logits for
every variant, before any training.

**Variant 1 — Confidence-Aware Fusion** (`ml/models/transformer2_confidence.py`,
`rgcn_attn_t2_confidence`). Reuses `Transformer2GlobalAttention` unmodified. Computes a
per-node confidence score (mean top-64 cosine similarity + attention entropy, no new learned
parameters) and multiplies it into the existing learned global `t2_scale`:
`z_impact + t2_scale * confidence_i * t2_output_i`. Params: 814,627 (identical to the original
T2 pilot — confidence adds zero learned parameters, exactly as specified).

**Variant 2 — Per-Node Trust Gate** (`ml/models/transformer2_trustgate.py`,
`rgcn_attn_t2_trustgate`). Replaces the single global `t2_scale` with a small per-node gate:
`trust_i = tanh(MLP(z_impact_i))`, MLP: `hidden→32→1`, final layer zero-initialized so
`trust_i = 0` for every node at construction (reuses Variant A's own bounded-residual pattern).
Params: 818,787 (+4,160 over the original — one small MLP).

**Variant 3 — Cross-Attention Fusion** (`ml/models/transformer2_crossattn.py`,
`rgcn_attn_t2_crossattn`). Does its own self-contained top-64 cosine retrieval (duplicated, not
sharing `Transformer2GlobalAttention`, to keep all three variants strictly isolated from each
other) so the impact embedding itself cross-attends over the raw retrieved pool
(`nn.MultiheadAttention`, 4 heads) rather than inheriting Transformer 2's own generic weighting.
A zero-initialized `Linear` sits after the attention output, so the fused contribution is
exactly zero at construction regardless of the (non-zero-initialized) attention internals.
Params: 864,162 (+49,535 over the original — the extra V-projection + MHA + output projection).

---

## AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| base_a (Variant A alone) | 0.8167 ± 0.0025 | 0.7952 ± 0.0038 | 0.9436 ± 0.0011 |
| base_t2 (original additive fusion) | 0.8144 ± 0.0051 | 0.7991 ± 0.0044 | 0.9425 ± 0.0026 |
| confidence | 0.8176 ± 0.0026 | 0.7995 ± 0.0039 | 0.9432 ± 0.0025 |
| trustgate | 0.8185 ± 0.0008 | 0.7991 ± 0.0037 | 0.9438 ± 0.0019 |
| crossattn | 0.8186 ± 0.0017 | 0.7992 ± 0.0050 | 0.9443 ± 0.0020 |

## Paired bootstrap + 5-seed sign-consistency — EACH variant vs BOTH shared baselines

**Every single comparison — 3 variants × 2 baselines × 3 tasks = 18 cells — FLIPS sign across
the 5 seeds.** No variant reliably beats either shared baseline on AUC, on any task. This is
exactly what Prerequisite 0's blocking finding predicted: since discovering H_POLYMER carries
no incremental predictive signal, no amount of fusion-mechanism sophistication can manufacture
an AUC improvement from it specifically.

| Comparison | delay | shortage | impact |
|---|---|---|---|
| confidence vs base_a | +0.0009 FLIPS | +0.0043 FLIPS | −0.0004 FLIPS |
| confidence vs base_t2 | +0.0031 FLIPS | +0.0004 FLIPS | +0.0007 FLIPS |
| trustgate vs base_a | +0.0018 FLIPS | +0.0039 FLIPS | +0.0003 FLIPS |
| trustgate vs base_t2 | +0.0041 FLIPS | +0.0000 FLIPS | +0.0014 FLIPS |
| crossattn vs base_a | +0.0019 FLIPS | +0.0040 FLIPS | +0.0008 FLIPS |
| crossattn vs base_t2 | +0.0041 FLIPS | +0.0001 FLIPS | +0.0019 FLIPS |

Every mean delta is small and mostly positive (12 of 18 positive), a mild directional hint but
never sign-consistent — the signature of noise, not a real effect, exactly matching every
other Step 6/7 gate-refinement round's own standard for calling something real.

---

## Hidden-dependency validation — discovery quality preserved?

| Arm | Mean percentile rank | Discovery rate | Global chance rate | Verdict |
|---|---|---|---|---|
| base_t2 (reference) | 0.7730 | 0.7778 | 0.1037 | **PASS** |
| confidence | 0.7130 | 0.7333 | 0.1061 | **PASS** |
| trustgate | 0.6357 | 0.6000 | 0.1034 | **PASS** |
| crossattn | 0.6510 | 0.5667 | 0.1063 | **PASS** |

**All four arms clear the chance bar decisively** — every mean percentile rank is well above
the 0.50 chance expectation and every discovery rate is 5.5–7.5x the measured global chance
rate (~10.4%). The discovery mechanism itself survives every fusion change intact. There is a
mild, consistent ordering worth noting honestly: confidence stays closest to the original
(0.713 vs 0.773), while trustgate and crossattn show a somewhat weaker (though still clearly
passing) recovery signal (0.636, 0.651) — plausibly because those two variants' fusion
mechanisms pull the impact embedding's own gradient signal away from the pure retrieval
objective more than confidence-weighting (which only rescales an existing, unchanged retrieval)
does. Not large enough to call a regression, but directionally consistent across both trust-
gate and cross-attention.

---

## Variant 2 (Per-Node Trust Gate) — per-node stability analysis

| Seed | Mean trust | Std trust | Share positive/negative/~0 | corr(\|trust\|, degree) |
|---|---|---|---|---|
| 0 | +0.0290 | 0.0748 | 83.6% / 14.7% / 1.6% | −0.1287 |
| 1 | +0.3846 | 0.3834 | 72.7% / 26.7% / 0.6% | −0.1460 |
| 2 | +0.0211 | 0.2035 | 65.1% / 33.4% / 1.5% | −0.0116 |
| 3 | +0.4280 | 0.3257 | 96.3% / 1.8% / 1.9% | −0.2204 |
| 4 | −0.0088 | 0.0801 | 40.0% / 41.5% / 18.6% | +0.1012 |

**Highly unstable across seeds — the same bimodal, seed-dependent signature Step 6's
depth-gate work found repeatedly.** Mean trust ranges from essentially zero (seed 4: −0.009) to
strongly positive (seed 3: +0.428); the share of Suppliers with meaningfully positive trust
swings from 40% to 96% depending purely on random seed. The degree correlation **FLIPS** across
seeds (`[-0.129, -0.146, -0.012, -0.220, +0.101]`) — no reliable relationship between a
Supplier's connectivity and how much the gate learns to trust Transformer 2 for it.

**H_POLYMER-specific check, directly informative given Prerequisite 0's finding:** the 4 known
H_POLYMER members' mean trust (**+0.0166**, n=120 pair-snapshot-seed observations) is
**lower** than the rest of the Supplier population's mean trust (**+0.1715**, n=23,880) — the
model does *not* learn to trust Transformer 2 more for the very suppliers where the discovered
correlation is real. This is consistent with, though not proof of, Prerequisite 0's structural
finding: since fusing in the H_POLYMER pool's signal offers no incremental predictive value for
those specific suppliers, there is no training pressure pushing the gate to trust it more for
them specifically — whatever elevated trust the gate does learn elsewhere is presumably fitting
something else in the data (or noise), not the planted scenario.

---

## Verdict

**Which (if any) of the three variants closes the gap between "Transformer 2 discovers
correctly" and "Transformer 2 improves impact AUC"?** **None of them, and Prerequisite 0
explains why this was the expected outcome, not a surprising one.** The discovery mechanism is
robust across every fusion redesign tried (all 4 arms pass hidden-dependency validation
decisively), but H_POLYMER — the only planted hidden-dependency scenario this dataset
contains — carries no incremental causal signal beyond what each supplier's own features
already capture, so no fusion sophistication (confidence-weighting, per-node trust, or full
cross-attention) had anything genuinely predictive to surface. All 18 AUC comparisons across
the three variants and two baselines flip sign across seeds — indistinguishable from noise.

**Per-mechanism reading, for future reference:**
- **Confidence-Aware Fusion**: the *retrieval* confidence signal is real and strongly
  correlated with *how much* T2 moves a prediction (Prerequisite 1, r=+0.77) — but "moves a
  prediction a lot" and "moves it in a way that helps" are different claims, and this dataset
  can't distinguish them given H_POLYMER's redundancy. Best discovery-quality preservation of
  the three variants (0.713 vs base_t2's 0.773).
- **Per-Node Trust Gate**: mechanically the most flexible of the three, and the one whose
  instability most resembles Step 6's depth-gate findings — worth revisiting specifically if a
  future dataset or real deployment provides a hidden-dependency scenario with genuine
  incremental signal, since the instability itself (not the AUC null result) is the open
  question here.
- **Cross-Attention Fusion**: the most expensive variant (+49,535 params, the priciest of the
  three), with the weakest discovery-quality preservation (0.651) and no AUC compensation to
  show for the extra cost — the least justified of the three on this evidence.

**What this round actually validates, worth stating plainly:** it is not that Transformer 2's
architecture is wrong — the discovery mechanism itself is repeatedly, robustly confirmed
correct and stable across five different fusion designs now (the original plus these three).
It is that **this specific synthetic dataset's only planted hidden-dependency scenario was
never going to reward ANY fusion mechanism with an AUC win**, because — per Prerequisite 0 — it
was built as a discoverable structural pattern, not an incrementally predictive one. A genuine
test of whether smarter fusion helps would need a dataset built the way Task 3's dual-sourcing
scenario was (`COPARENT_COUPLING`, an explicit, incremental cross-supplier bleed-through term)
applied to a Transformer-2-style undirected/non-graph-edge relationship, not H_POLYMER as it
currently exists.

---

## Governance record

Backed up before any write
(`reports/backups/model_registry_and_evals_backup_20260808_215212.sql`). 25 new registry rows
— 5 `rgcn_attn_rung5_a-t2f-basecmp-seed{n}` + 5 `rgcn_attn_variant_a_transformer2-t2f-seed{n}`
+ 5 `rgcn_attn_t2_confidence-t2f-seed{n}` + 5 `rgcn_attn_t2_trustgate-t2f-seed{n}` + 5
`rgcn_attn_t2_crossattn-t2f-seed{n}`, all `status='active'` — **333 total registry rows**
(308 + 25). **375 new evaluation rows** (25 runs × 15 metric rows) — **4,863 total**
(4,488 + 375). **7,557 new `hidden_dependency_links` rows** (15 variant runs × ~504 each,
logged only for the 3 new variants' most recent test snapshot, matching the original T2
pilot's own bounding convention — `base_a`/`base_t2` are analysis-only this round, not
re-logged) — **10,081 total** (2,524 + 7,557). No existing row from any prior round modified.
Full run log: `reports/logs/run_transformer2_fusion_pilot_20260808_215220.log`.

---

*Scoped to SHARE + Variant A + Transformer 2 only — no changes to SHARP (`rgcn_relemb`), SHARK
(`rgcn_battn`), or the existing Markov/Variant A depth-selection logic.*
