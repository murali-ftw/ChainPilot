# 13 — Testing and Validation

> **V2 reframing.** Previously QA for the V1 application. Now the generator's validation suite and
> the statistical tests the evaluation protocol depends on.

**Split definition:** `10_AI_ML_Documentation.md` §4. **Spec:** `00_Benchmark_Specification.md`.

---

## 1. Where validation runs

Validation is **inside** `db/generate_dataset.py` and executes on every generation, before any
dataset is usable. The suite exits non-zero on failure, so a variant that fails cannot silently
become a benchmark artifact.

```bash
python3 db/generate_dataset.py --variant K --config v1     # suite runs automatically
python3 db/run_benchmark.py --stats-only                   # aggregate failures across the matrix
python3 db/run_benchmark.py --verify-determinism --variant K
python3 db/load_data.py --dsn "postgresql:///hades" --drop --csv-dir db/csv/vK_seed42
```

Current status: **12 variants × 5 seeds = 60 runs, zero validation failures**, at the `v1` fixture.

## 2. Three validation layers

| Layer | What it catches | Why it is not redundant |
|---|---|---|
| **In-generator suite** | mechanism semantics, statistical properties, leakage | operates on in-memory structures |
| **Determinism diff** | any non-reproducible byte | compares *compressed* output across two runs |
| **PostgreSQL load** | referential integrity of the **emitted artifact** | the only layer that sees what actually landed on disk |

**The third layer is load-bearing, not ceremonial.** Mechanism A truncates `suppliers.csv`, but
`component_suppliers` and `risk_scores` continued to emit rows referencing hidden suppliers — a live
foreign-key violation that the in-generator suite missed because it reasoned over in-memory
structures rather than emitted rows. It surfaced only when Variant K was actually loaded. The
Mechanism A check now verifies **every** emitted supplier reference, and `load_data.py` carries a
matching FK query. Run the loader before trusting a variant.

## 3. Per-mechanism validation

| Mechanism | Checks | Statistic |
|---|---|---|
| A | truncation %; no feature/label references a hidden supplier; **no emitted table references a truncated supplier**; hidden network still transmits | exact set containment |
| B | coverage vs `hidden_parent_rate`; type mix; A/B/C indistinguishable from topology; Type B co-degrades; Type C inert | permutation null |
| C | rewire rate; validity windows close; edge **set** and **count** both move | exact counts |
| D | Type A stress exceeds own-history stress; coupling reaches only Type A | direct measurement |
| E | distribution matches config; never emitted; saturation share; recoverable from observed history | held-out AUC + bootstrap CI |
| F | attenuation exactly a function of resilience; spans regimes; monotone decreasing | exact equality |
| G | recorded ≥ true; mean lag; as-of uses recorded time; clocks demonstrably diverge | distributional |
| H | shock count; blast radius; hits infrastructure-correlated not uniform | grouping vs uniform |
| I | source count; AND/OR mix; no duplicates; AND and OR resolve differently | exact counts |
| J | depth heterogeneity; depths in range; depth independent of per-hop attenuation | permutation null |
| Phase 2 loop | leakage assertions enabled; agent clock monotonic | executable assertion |

## 4. Gate versus report

Several checks are under-powered at small configurations. Rather than weaken a threshold to make it
pass, each **gates when it has the power to mean something and reports its numbers otherwise**,
printing the sample size it would need.

| Check | Gates when | Reports otherwise because |
|---|---|---|
| Legacy H_POLYMER co-degradation | Variant 0 only | fixed 4-member statistic; degenerates to n=1–2 at scale |
| Type B / Type C co-degradation | ≥ 20 measurable groups | `sup_n=800` gives 6–9; `sup_n=4,000` gives 26–34 |
| Type B under absorption | Mechanism E absent | absorption compresses the signal ~15% **by design** |
| Resilience recoverability | ≥ 800 estimable suppliers | CI width scales with the estimable population |

**A reported check is not a passed check.** It prints its effect size and null band; a reader
skimming for `[PASS]` can mistake one for the other. When reporting results, state which checks
gated and which reported at the configuration used.

## 5. Statistical conventions

Three conventions were adopted after specific measurement errors during the build. They apply to
every check and to any analysis built on top of the benchmark.

1. **Test against the correct null, not a round number.** Cross-validated AUC on a few hundred rows
   is systematically biased *below* 0.50. Indistinguishability is therefore tested against a
   **permutation null** (measured at [0.398, 0.570], centred near 0.48), not against 0.50. Testing
   against 0.50 failed a correct mechanism.
2. **Never pool one-vs-rest scores.** Scores from separately fitted models carry different
   intercepts; pooling them measures relative calibration as much as discrimination. Report the
   **macro** average.
3. **Widen per-run bands for repeated gates.** A gate evaluated once per variant per seed runs 60
   times in a full sweep, where a 95% band would be expected to exclude ~3 times by chance. Such
   gates use a **99%** band. This is multiple-comparison correction, not relaxation.

A fourth, learned the hard way: **beware modulo arithmetic in test harnesses.** Assigning class
labels by `k % 3` while assigning CV folds by `i % 5` made the two interact and produced a
systematic 0.39–0.44 AUC on data carrying no signal at all. Both are now seeded shuffles.

## 6. Directionality

For **indistinguishability** checks (Mechanism B), the requirement is the *absence* of information,
so a reliably **below**-chance AUC is a failure — it is a discriminator with its sign flipped.

For **recoverability** checks (Mechanism E), the requirement is the *presence* of information, so
the gate is `|AUC − 0.5|` clear of zero. An inverted predictor still recovers the latent. This
matters because resilience recoverability genuinely inverts between Variant E and Variant F.

## 7. Evaluation-protocol tests

`db/benchmark_eval.py` provides, and self-tests:

| Function | Purpose |
|---|---|
| `temporal_splits()` | positional 60/20/20; identical across variants by construction |
| `paired_bootstrap()` | percentile CI for a paired delta |
| `sign_consistency()` | fraction of seeds sharing the majority sign |
| `summarize()` | mean and standard deviation |
| `significant()` | the spec's full criterion: CI excludes zero **and** consistency ≥ 0.8 |

`--check-splits` exits non-zero if any task has zero test-split positives — a variant that cannot
support a comparison should fail loudly rather than produce an empty result table.

## 8. Known testing gaps

1. The 12 × 5 sweep has only run at the `v1` fixture; **no spec-scale validation run exists yet**.
2. Label subsampling is configured and range-checked but not wired into emission, so the power
   flags in the report table do not yet reflect intended behaviour.
3. No model is trained anywhere in this repository — the evaluation protocol supplies the
   statistical machinery, not baselines. Baseline implementations are a submission's own
   responsibility, per `10_AI_ML_Documentation.md` §8.
