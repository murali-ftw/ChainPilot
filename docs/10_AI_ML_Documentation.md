# 10 — Benchmark Design, Evaluation Protocol and Baselines

> **V2 reframing.** Previously the ML design doc for the V1 application. Now the primary reference
> for what HADES-Bench V2 measures, how a submission must be evaluated, and what each mechanism is
> designed to reveal. V1 architecture findings are carried forward **with their original caveats
> intact** (§7) — they are results, not aspirations.

**Authoritative spec:** `00_Benchmark_Specification.md`.
**Ground truth for what the generator actually does:** `PHASE2_PHASE6_IMPLEMENTATION.md`.

---

## 1. Tasks

Three binary prediction tasks, all per-snapshot with a 14-day horizon:

| Task | Entity | Positive means | Scales with |
|---|---|---|---|
| `delay` | shipment | a `delayed` transition occurs in `(t0, t0+14d]` | in-flight shipment count |
| `shortage` | (product, warehouse) | stock crosses below the reorder threshold in the horizon | product × warehouse × snapshots |
| `impact` | supplier | any of the supplier's in-flight shipments goes late in the horizon | suppliers × snapshots |

The three sit on very different denominators, which is why they need different treatment to hit the
same power target (§5).

## 2. The seven research questions

Full statement in `01_Product_Requirement_Document.md` §2. In brief: **RQ1** reasoning under a
truncated graph (A); **RQ2** discovering latent structure (B); **RQ3** whether discovery *pays*
(D); **RQ4** inferring hidden operational state and the transmission it induces (E, F); **RQ5**
reasoning when observation lags truth (G); **RQ6** robustness to rewiring and correlated failure
(C, H); **RQ7** adapting depth and handling AND/OR redundancy (I, J).

## 3. The twelve variants and what each reveals

| Variant | Mechanisms | Report against | What a difference here means |
|---|---|---|---|
| 0 | — | — | V1 reproduction; the anchor |
| A | J + A | **J** | cost of the graph terminating before the cause |
| B | B | 0 | whether latent groups are discoverable at all |
| C | C | 0 | sensitivity to topology that moves between snapshots |
| D | B + D | **B** | whether discovered structure carries *incremental* information |
| E | E | 0 | cost of an unobservable operational state |
| F | E + F | **E** | whether attenuation derived from that state is learnable |
| G | G | 0 | cost of a reporting clock that lags the event clock |
| H | H | 0 | correlated vs independent failure |
| I | I | 0 | AND/OR redundancy beyond source count |
| J | J | 0 | heterogeneous depth |
| K | all ten | **D** for hidden-structure claims | combined difficulty |

### Reading rules that are correctness requirements, not conventions

1. **Variants with prerequisites are never reported against Variant 0.** `A = J + A`,
   `D = B + D`, `F = E + F`. Reporting D against 0 attributes B's effect to coupling.
2. **Variant K's hidden-structure results must be read as a delta against Variant D.** Absorption
   (E/F) compresses Mechanism B's planted co-degradation signal by ~15% and pushes it below
   detectability: measured at spec scale, Variant D clears at −0.0428 against a null low of −0.0332
   while Variant K sits at −0.0366 against −0.0389. **A null result on discovery in Variant K is
   uninterpretable in isolation** — it may mean the signal was absorbed before K could observe it.
3. **Resilience recoverability has no fixed sign.** Direct in Variant E (AUC 0.561), **inverted** in
   Variant F (AUC 0.396), because a resilient supplier also receives less transmitted stress, which
   lowers the very stress an estimator regresses against. No baseline or probe may assume a
   direction; E and F results must never be pooled. Report `|AUC − 0.5|` with the direction stated.
4. **Mechanism A *improves* recoverable coverage rather than degrading it.** It removes precisely
   the zero-shipper nodes that could never be estimated. Variant F (no A) is therefore the *harder*
   case for resilience, not Variant K. The coverage ceiling is **71.7% for Variant F** and
   **86.3% for Variant K** — never quote one number for both.

## 4. Temporal split definition

Snapshots are split **positionally**, 60/20/20, so cutoffs are identical across every variant by
construction. A variant may change what happens inside a snapshot window; it may never change which
snapshots fall in which split.

At the spec configuration (40 monthly snapshots, `t0` = Jul 2024 … Oct 2027):

| Split | Snapshot range | Count |
|---|---|---|
| Train | Jul 2024 – Jun 2026 | 24 (1–24) |
| Validation | Jul 2026 – Feb 2027 | 8 (25–32) |
| Test | Mar 2027 – Oct 2027 | 8 (33–40) |

Implemented by `temporal_splits()` in `db/benchmark_eval.py`; referenced from
`13_Testing_Documentation.md`. **The disruption event calendar must cover all 40 months** —
snapshots past the calendar produce ~55% of a calibrated month's positives, which under-powers the
validation and test splits specifically.

## 5. Statistical power

Target 2,000–5,000 positives per task per variant, including K.

Impact is the binding task and the only one that reaches the range by scaling; delay and shortage
sit on far larger denominators and must be subsampled *down* via `DELAY_SAMPLE_RATE` and
`SHORTAGE_SAMPLE_RATE`, sampling entities rather than label rows.

> **In progress.** Subsampling is configured and validated but **not wired into emission**, and the
> 12 × 5 sweep has only run at the small `v1` fixture (800 suppliers), where every variant reports
> `LOW` on delay and impact. Power targets are met for no task at any scale actually generated.
> Mechanism G additionally suppresses delay positives ~3× — delayed reporting leaves
> already-delivered shipments looking in-transit, so they stay eligible as guaranteed negatives —
> so `DELAY_SAMPLE_RATE` must be re-derived for any variant including G. See
> `PHASE2_PHASE6_IMPLEMENTATION.md` §7.

## 6. Evaluation protocol

Every submission must report:

- identical temporal splits (§4) and identical benchmark configuration
- **five independent seeds**
- **paired bootstrap confidence intervals**
- **sign consistency** across seeds
- mean and standard deviation

An improvement counts as significant only when its CI excludes zero **and** its sign is consistent
across seeds. `significant()` in `db/benchmark_eval.py` applies both. A delta whose CI excludes zero
but whose sign flips across seeds is not an improvement — V1 reported these as "FLIPS (noise)" and
that discipline carries forward.

Pairing matters: every comparison against a freshly-retrained baseline carries that baseline's own
seed noise on top of the candidate's. V1 flagged this as a live confound (see §7).

## 7. V1 findings carried forward — with original caveats

These are V1 results on the V1 dataset. They are **not** V2 claims and must not be restated as
though V2 has validated them.

| Finding | Verdict as originally reported |
|---|---|
| SHARE vs GraphSAGE | SHARE **beats** on shortage (−0.012 to −0.014) and is **genuinely tied** on delay |
| SHARE / SHARP / SHARK on impact | **tie, not a win.** SHARE's edge over HGT on impact "remains a statistical tie, not a confirmed win" |
| SHARK impact variance | std an order of magnitude wider than SHARE/SHARP (0.0559 vs 0.0058/0.0071), driven by one outlier seed, **unexplained** |
| Layer 2 depth mechanisms | five independent designs, same negative result on shortage and impact — **could not distinguish** true depth-indifference from insufficient chain-length variation |
| Layer 3 fusion (Confidence-Aware, Trust Gate, Cross-Attention) | all three produced the **same null**, traced to a dataset property (H_POLYMER structurally redundant), not a modelling failure |
| Impact AUC comparisons generally | "not distinguishable at this label volume" — a statement about sample size, **not** evidence of equivalence |
| Every HGT comparison | **unpaired**, since no HGT checkpoint was ever persisted — strictly less powerful than a paired comparison |

V2 exists to resolve several of these: Mechanism D supplies the causal coupling H_POLYMER lacked,
Mechanism J supplies the chain-length variation Layer 2 never had, and the power targets address
the label-volume verdicts. **None of that is validated yet.**

## 8. Baseline specification

Every submission must report against all of the following, at the **same layer count and hidden
dimension** within an experiment, with every hyperparameter not under study held constant and
reported in the configuration output.

| Baseline | Configuration notes |
|---|---|
| GCN | standard config; layer count and hidden dim fixed across all rows below |
| GraphSAGE | mean aggregator |
| GAT | single-head unless multi-head is the object of study |
| RGCN | relation-specific weights per the schema's edge types |
| HGT | V1's comparison anchor; see the pairing caveat in §7 |
| GraphGPS | |
| HADES — SHARE | RGCN + one shared attention scorer (V1's primary hybrid) |
| HADES — SHARP | SHARE + per-relation embedding |
| HADES — SHARK | SHARE + relation-specific attention basis pool. **Report per-seed variance explicitly** given V1's documented instability |

Any probe of hidden resilience must follow reading rule 3 in §3 — report `|AUC − 0.5|` with the
direction, never assume a sign, never pool Variant E with Variant F.

## 9. Out of scope for V2

Deliberately deferred, not dropped: thin-relation parameter-sharing analysis (SHARK), degree
randomization robustness, paired HGT checkpoint comparisons, Shipment anti-smoothing 2×2.
