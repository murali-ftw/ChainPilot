# Step 6 Pre-Flight — Two Open Findings Confirmed Before the Depth Gate

**Scope:** confirm two open questions before any Step 6 (learned depth gate) work begins: (1)
whether RGCN+Attention (SHARE)'s delay-peaks-at-L1 pattern — found at only 3 seeds in
`reports/rgcn_types.md`'s over-smoothing sweep — survives proper 5-seed testing, and (2)
whether the original "delay's co-parent path is unneeded and broken" finding still holds now
that dual-sourcing is confirmed active (`reports/entropy_test.md`). No depth-gate
implementation in this pass — this is confirmation work only.

---

## Phase 0 — Pre-flight

Confirmed live: 800 suppliers, 15 `graph_snapshots`, 420 `component_suppliers` rows
(dual-sourcing/co-parent mechanism active — the same dataset every RGCN-family round since the
4-architecture ablation has run against). Backed up `model_registry`/`model_evaluation_runs`
(197 + 3,048 rows) unconditionally before any write.

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

---

## Governance record

Backed up before any write. 15 new `-depthconfirm-{config}-seed{n}` registry rows, all
`status='active'` — 212 total registry rows (197 + 15). 225 new evaluation rows — 3,048 total
(2,823 + 225). No existing row from any prior round modified. Full run log:
`reports/logs/run_depth_confirm_5seed_20260807_234516.log`.

---

*Out of scope for this pass, per its own framing: no depth-gate implementation. Scoped to
RGCN+Attention (SHARE) only — no changes to RGCN+BasisAttn (SHARK) or RGCN+RelEmbedding (SHARP).*
