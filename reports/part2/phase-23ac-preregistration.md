# Phase 23AC — pre-registration

**Committed before any Phase 23AC code or measurement.** Nothing below is edited after a result is known. Each
prediction is marked right or wrong in `reports/part2/phase-23ac.md`; wrong ones are kept.

**Measured on:** v8 seed 1001 (primary); v8w1002 (`data_worlds/v8_seed1002/seed_1002`, Phase 20; recorded hashes
re-verified) for replication. Fixed split: train ≤ 2023, val 2024, test 2025. Seeds 7 / 17 / 27 / 37 / 47.
**Base:** local `HADES-v4-ml-pipeline` at `545e282` (contains `reports/part2/phase-22.md`: yes).
**Machine:** Apple M4 Pro, 14 cores. **CPU only**: no neural training, no MPS, no new fitted model. Every number comes
from stored predictions or stored bundles.
**Wall-clock stop:** 8 h from the start at **17:49:22 IST 2026-10-05**, i.e. **01:49:22 IST 2026-10-06**. Unfinished work
goes to `reports/part2/phase23ac/STOPPED.json`.
**Deviations** are numbered from **223**. Phase 23B (another machine) owns 240+ and its own paths, which are never touched here.

## Deviations taken before any measurement

| # | brief | done, and why |
|---|---|---|
| **223** | "Branch: phase23ac, in a new git worktree … never touch the user's working tree" | **Work is committed directly on `HADES-v4-ml-pipeline` in the user's checkout, with no new branch or worktree.** The owner's later standing instruction (2026-10-05, after Phase 22): "dont create seperate branches here after", after asking twice for the side branches to be merged and deleted. The parts of the rule that protect the owner are kept: only the exact new files are staged (never `git commit -a`); `CLAUDE.md` and any `db/` line-ending changes are never staged, stashed, reset or discarded; nothing is merged or pushed. At the start the tree was clean |
| **224** | T4 charts with matplotlib | matplotlib is not in the project venv. Installing it there could upgrade numpy and change numerical results, so charts are rendered by a separate interpreter (a scratch virtual environment with matplotlib) that reads only the CSVs the measurement scripts write. No model or score is computed in it |
| **225** | T2b: "the LightGBM consolidated model must be persisted … ONLY if it reproduces its stored predictions bit-exactly" | **Not attempted.** No LightGBM fill model was ever persisted (Phase 22 saved predictions only). Persisting one needs a refit, which this phase forbids. The sub-step stops here, as the brief directs; the dormant path serves the five neural season + cadence bundles only |

## Predictions (from the brief, verbatim)

| # | Prediction |
|---|---|
| **P1** | the product beats the promise date, the channel trailing mean and the channel trailing median on A3 (block-disjoint) and on the share within +-7 days. |
| **P2** | the month-specific 80% interval keeps coverage 0.77-0.83 overall and per creation month. |
| **P3** | the dormant module reproduces its stored predictions bit-exactly. |
| **P4** | the guard flags all nine and no clean replacement. |
| **P5** | the guard finds no further leaking column in the registered set (if it does, that is a finding, not a failure). |

How each is scored, fixed now:
- **P1** right iff, on v8 test lines, the product's creation-week-block 95% interval of (baseline A3 − product A3) excludes
  0 in the product's favour against **each** of the three baselines, **and** its share within ±7 days is higher than
  each (point estimate; block interval reported).
- **P2** right iff overall test coverage ∈ [0.77, 0.83] **and** every creation month's coverage ∈ [0.77, 0.83].
- **P3** right iff the served 5-seed fill distribution equals the mean of the stored per-seed predictions with max |Δ| = 0.0
  on the 200 test rows. Any nonzero difference makes P3 wrong, and its size is reported. The stored predictions were made
  on MPS; this phase serves on CPU.
- **P4** right iff all nine Phase 22 leaking columns FAIL the guard on both worlds **and** all nine clean replacements PASS
  on both worlds.
- **P5** right iff no registered column other than the nine FAILs (a non-known FAIL is listed as a finding).

## Definitions

### T1 — order-time comparison (Phase 21 / 22 at-placement test lines; τ = the Monday on or before creation)

- **Product:** the Phase 22 bundle `ml/artifacts/phase22/serve/order_time_v8clean/` through `ml/serve/order_time.py`,
  unchanged. Its date is the shrunk Kaplan–Meier median lead (k = 10) + offset; its interval is month-specific
  split-conformal; its flag is the strict `flag_lag1` ensemble.
- **Baselines** (all as-of τ, from receipts recorded ≤ τ):
  - (i) **promise date**: `original_promise_date` (= creation + contracted lead);
  - (ii) **channel trailing mean lead**: mean of (first receipt − creation) over the channel's lines whose first receipt is
    recorded ≤ τ and created in the 52 weeks before τ;
  - (iii) **channel trailing median lead**: the same, median.

  Where a channel has no such receipt, (ii) / (iii) fall back to the channel's contracted lead. The share of lines that
  fall back is reported.
- **Metrics on receipted test lines:** A3 = median |error| days; mean |error|; share with |error| ≤ 7 days; signed bias
  (mean of predicted − actual, days). Interval coverage and width: the product only (the baselines have no interval).
  Overall and per creation month.
- **Intervals:** creation-week-block bootstrap, 1,000 resamples (Phase 22's).
- **World 2:** the Phase 22 world-2 estimator (KM, k = 3 chosen on world-2 validation, its own offset and month conformal
  quantiles from `order_time_v8w1002clean.json`), computed from the world-2 group store. Own bands.
- **Constructed failing case:** a leaked arm (the order-time feature list plus the nine leaking columns read from the
  published panel) MUST be flagged by the T3 guard's feature check.
- **Sample cards:** `numpy.random.default_rng(2310)` draws 5 receipted test lines. If none of the 5 misses by > 14 days,
  the first test line that does is added.

### T2 — dormant fill path

- `ml/serve/_plan_reader.py` is the only new file that names `part_demand_weekly`. It asserts as_of_date + 2 days ≤ t0
  on every row it passes on.
- `ml/serve/fill_consolidated.py` builds season + cadence through it and `fwd_load.snapshot_features` (pure computation),
  and serves the five Phase 22 bundles `phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence` on CPU.
- The pre-existing readers `ml/data/fwd_load.py` (Phase 18 build-time) and `ml/serve/features.py` (Phase 20, through
  fwd_load) are listed as found, not changed.
- **Season poison:**
  - Corrupt every `part_demand_weekly` row with as_of + 2 d > t0; 0 of N season rows may change.
  - **Constructed offender:** a reader without the bound changes N of N.
- **Isolation:** an AST allow-list of exactly one reader among the new files. A constructed second reader MUST be flagged.

### T3 — the leak guard

- **Registry:** built by code from the feature builders. Covers:
  - the panel columns: cache `meta.json` cols and nullable, raw `v8` and clean `v8clean`;
  - `phase7_fit.World` STATIC / FLAT;
  - the family modules' `COLS`;
  - grpstats `ASTATS` / assembled block columns;
  - phase22_rows ack / L4 columns;
  - the order-time product's flag feature list (from its bundle).

  Each entry records source table, horizon and the models that read it.
- **Future poison** (per panel column, per world):
  - Raw columns: the Phase 22 exact rebuild (`ml/eval/phase22_leakscan.py`, audit-only, reads `_sim.npz`), poisoning
    rows visible after the row's own week (H_week).
  - Clean columns: `clean_panel` poisoning.
  - Family modules: their own falsifiers, plus recomputation under poisoned sources where the module exposes a
    per-t0 function.

  **FAIL** if any value at row t moves.
- **Label alignment** (panel columns), on active channel-weeks at 12 sampled weeks: the share of rows whose value equals,
  within the column's stored rounding, an outcome-derived quantity of a line that is **still pending at t** (ordered
  ≤ t, outcome visible > t):
  - its eventual lead; its lead / contract; its eventual fill;
  - the channel's quantity-weighted eventual fill of lines ordered in week t;
  - the supplier's eventual whole-month ordered quantity / declared capacity.

  **FAIL** if the share is ≥ 0.20. The constant is fixed here, before measurement.
- **Self-exclusion** where a column is row-specific: the row's own line never enters. Uses the Phase 21 / 22 tests
  (grpstats, clean_panel).
- **Known answer:** on both worlds, the nine columns of `phase22/stage1_leak.md` must FAIL poison or alignment, and the
  nine clean replacements must PASS both. Otherwise the guard is invalid and T3 stops.

### T4 — three columns, one question each

**INCUMBENT** = Phase 15's arms:
- UC1: neural h4;
- UC2: boundary bw3;
- UC2b: lgbm22_id;
- UC3: mp h4.

**BEST PUBLISHED** = Phases 19–20:
- UC1 late list: the incumbent ensemble P(late) (Phase 20 kept it);
- arrival point: the Phase 19 blend;
- UC2 / UC2b / CRPS: the Phase 19 fill blend;
- UC3: the incumbent ensemble.

Both of these columns are labelled **LEAKY (superseded, Phase 22)**.

**CLEAN** = Phase 22's arms:
- UC1 list: the clean incumbent neural ensemble P(late);
- arrival point: the clean Phase 19 recipe (w 0.52);
- fill CRPS / UC2: the consolidated blend;
- fill UC2b: the clean Phase 19 fill blend;
- UC3: the clean incumbent;
- order time: the product date and the strict flag.

**Rules:**
- **Operating points:** Phase 15's `pick_on_val`, on validation. Per-seed arms report the 5-seed mean [min, max];
  ensembles and blends report snapshot-block intervals.
- **Accuracy:** at the validation max-F1 threshold (Phase 20 rule 3), always printed with the majority-class accuracy.
- **AUC and PR-AUC:** each computed by two methods (`sklearn` and an independent rank / step implementation); they must
  agree to 1e-6.
- **Every figure:** marked RECOMPUTED (from stored predictions) or QUOTED (report), with its source and commit. A
  recomputed figure that does not reproduce the report's published precision is a finding.

### Gates and their constructed failing cases (each run by the code)

| gate | constructed failing case |
|---|---|
| T1 leak gate | the leaked arm's feature list must be flagged |
| T1 / T2 serving guard | a wrong name, a tampered model file and a missing bundle must each raise |
| T2 reader allow-list | a constructed second reader must be flagged |
| T2 season poison | a reader without the as_of + 2 d bound must change N / N rows |
| T3 poison | the nine known columns must FAIL |
| T3 label alignment | the nine known columns must FAIL; a constructed column equal to each line's eventual lead must FAIL |
| T4 accuracy | an always-majority predictor must show accuracy = baseline, with recall 0 or precision = base rate exposing it |
| T4 AUC methods | the two methods must agree; a constructed tie-heavy case is checked against a hand-computed value |
| no training | an AST scan of every new module for optimizer steps, `.fit(` / `lgb.train` / `backward(`; a constructed offender must be flagged |
