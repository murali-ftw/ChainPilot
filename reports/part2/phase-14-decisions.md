# Phase 14 — pre-registered decisions (Stage A)

**Committed before any Phase 14 scoring.** Not edited afterwards. Scored in `reports/part2/phase-14.md` §4.

**Common rules, all use cases:**

- v8 seed 1001, fixed split. Thresholds are fitted on **VALIDATION (2024)**, frozen, then applied to **TEST (2025)**.
- **Tie rule:** predicted positive iff score ≥ τ. Candidate thresholds are the distinct validation scores (≤ 2,000
  quantiles), plus "predict nothing positive".
- **Missing rows** (non-finite score or label) are **excluded and counted**. Every table carries n.
- **Primary threshold objective: F1 on the positive class**, as the brief specifies. Where the positive class is
  the majority, F1 is maximised by "predict everything positive" (a 75%-positive class gives F1 ≈ 0.857 that way).
  So a **secondary threshold, maximising MCC on validation, is pre-registered here** and reported beside it. It is
  not a replacement.
- Bands: 5 model seeds, [min, max]. Deterministic arms get a **labelled row bootstrap** (500 resamples, 95%).
- RAW and RECALIBRATED are separate tables, each with its own threshold fitted on its own validation scores.
- Every row carries the always-majority accuracy on the same rows.
- **Files are read by exact per-seed names. No globs.**

| UC | decision (positive class) | score | label and handling | threshold objective | arms |
|---|---|---|---|---|---|
| **UC1** | "this PO line arrives LATER than its contracted lead" | P(T > R) from the hazard survival curve (unconditional); point arms use ŷ − R; channel-constant ranker uses −R | R = contracted lead (`sourcing_channels`, t0-available) + a global offset fitted on TRAINING rows (Phase 11B reference (b)). The brief's wording. The project's **adopted** metric (as-of channel median lead, `docs/specs/lateness_metric.md`) is scored as a **secondary** decision. **Censoring, three ways, each reported:** (A) censored rows EXCLUDED; (B) censored rows counted as NOT LATE (the brief's alternative); (C) **RESOLVED**: a censored row has T ≥ 13, so it is known LATE when R < 13, and excluded only when R ≥ 13. **C is the principal treatment, because it is the only one that uses what censoring actually tells us** | F1 (+ MCC secondary), per treatment, arm and calibration state | h⁴ ×5, h⁰ ×5, b5flat_reg ×5, lgbm_id_reg ×5, always-late, always-not-late, channel-constant ranker |
| **UC1b** | "this line arrives on or before week t0 + H", H = 1, 2, 4 | P(T ≤ H) = 1 − S(H); point arms use −ŷ | censoring is **fully determined** here: a censored row (T ≥ 13) is known NOT to arrive by H ≤ 4. No row is excluded for censoring | F1 (+ MCC) | as UC1 |
| **UC2** | "this line arrives in FULL (fill = 1.0)" | P(cell 21) | label fill ≥ 1 | F1 (+ MCC) | 22-cell head ×5, boundary w=3 ×5, Beta ×5, b5flat22 ×5, lgbm22_id ×5, rolling-52 histogram (bootstrap), **always-full (the majority row)** |
| **UC2b** | "this line is MATERIALLY SHORT": fill < 0.95 and, separately, fill < 0.75 | Σ P over cells below the cut (the cuts coincide with bin edges: cells 0..19 and 0..15) | label fill < cut | F1 (+ MCC) | as UC2 |
| **UC3** | "this channel's demand will EXCEED its capacity over the 90-day horizon" | P(strain > 1), where the CDF runs piecewise-linearly through (P10, 0.1), (P50, 0.5), (P90, 0.9), with the tails extrapolated linearly from the nearest segment and clipped to [0, 1] | label: strain > 1.0. **Not a proxy:** the strain label is ordered / true capacity, so > 1 **is** "requirement exceeds capacity". **Deviation from the brief's wording:** the label is a 90-day channel aggregate, not a weekly one | F1 (+ MCC). **RAW ONLY**: no capacity recalibrator exists, so Table B is N/A | mp h⁴ ×5, h⁰ ×5, b5flat_q ×5, naive channel / supplier (bootstrap) |
| **UC4** | "this part-plant is AT RISK of a shortage event in the horizon" | the BCE head's P | the stored binary shortage label. **Diagnostic only:** v8's positive rate is ~8× reality, and the head is graded on a population it was not trained for | F1 (+ MCC). RAW only | mp h¹ ×5 (shipped), b5flat_bin ×5, h⁰ (**3 seeds: NOT quotable**), naive part-plant rate |
| **UC5** | "this part-plant-week will be BELOW SAFETY STOCK" | P_sim(level < SS) from the ROP-policy simulation, N = 200 paths, fill-head seeds 7 / 17 / 27 / 37 / 47 | **(a) OBSERVED** store (on hand < SS) and **(b) `PRIVILEGED__prerescue_below_ss`** (on hand − that week's net transfer < SS), matched per part-plant-week (week_start = t0 + 7(w+1)). **Paths were never stored:** they are regenerated with the identical code and seeds, gated on reproducing Phase 12 B2's stored test below-SS fraction EXACTLY per seed. Planner-action sweep: θ = 0.1..0.9 against "net transfer-in > 0 that week" | θ fitted on VALIDATION (2024 snapshots), max F1, **separately for (a) and (b)** | policy_rop ×5 fill seeds; always-not-below baseline |
| **UC6** | delivery schedule | — | **NOT SCOREABLE.** 0 of 4 cost parameters exist (holding, ordering, freight, shortage). No synthetic cost | — | — |
| **UC7** | "the RECOMMENDED top supplier is the one ACTUALLY used" | Phase 12 A4's v8 recommendation (price_weight 0, deduplicated), t0 = 2025-06-30, the same 120 part-plants | actual = the supplier with the largest ordered quantity on PO lines raised in (t0, t0 + 13 weeks] at that part-plant. precision@1: recommended top-share supplier = actual. precision@2: actual ∈ recommended top-2 by share. **The feasibility rate is reported first; precision is conditional on the feasible part-plants, with n** | no threshold (a ranking) | policy recommendation; **incumbent's top supplier** (status quo) as the baseline |
| **UC8** | (i) RECIPIENT: "this projected-short part-plant-week receives a transfer"; (ii) DONOR: "this candidate plant is the donor" | the S2 rules (policy / naive / random) | recipient truth: transfer-in recorded that week (S2's definition). **Donor truth is INFERRED** (a same-week transfer-out of the part at the donor), because `from_plant_id` records the receiving plant on all 1,482,011 rows (deviation 119). Universes: recipients = every projected-short part-plant-week; donors = every (short part-plant-week, same-part candidate with positive simulated surplus) pair | S2's θ sweep {0.02 … 0.5}; S2's fitted θ = 0.5 is carried, not refitted | policy / naive / random, 5 fill seeds, **from the same regenerated simulations as UC5** |

**Metrics per row (Stage C):**

- n, positives, base rate;
- τ, with where it was fitted;
- TP / FP / TN / FN;
- accuracy **beside the majority accuracy on the same rows**;
- precision, recall, F1, specificity, MCC, balanced accuracy;
- ROC-AUC and PR-AUC.

**Quotable (Stage D)** requires all of the following:

- 5 seeds (or a labelled bootstrap for a deterministic arm);
- F1 disjoint from the always-majority baseline's F1;
- τ fitted on validation;
- no unlabelled proxy in the target.
