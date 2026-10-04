# Phase 21 — the 5-D group idea: hierarchical, as-of group statistics for arrival and fill

**Audience:** the owner who proposed the 5-D group idea, and whoever decides what arrival and fill ship next.
**Measured on:** v8 seed 1001 (primary) and v8w1002 (Phase 20's second world from v8 seed 1002; Stage 7). Fixed split
(train ≤ 2023, val 2024, test 2025); seeds 7 / 17 / 27 / 37 / 47; TEST unless marked; RAW throughout.
**Companions:** `phase-21-preregistration.md` (`f676ca5`); `phase21/stage1_audit.md`, `stage2_builder.md`,
`stage3_arrival_snapshots.md`, `stage4_at_placement.md`, `stage5_fill.md`, `stage6_hybrid.md`, `stage7_replication.md`.
No PRIVILEGED file.
**Branch / base:** `phase21`, cut from `HADES-v4-ml-pipeline` at `124be5b`, in a separate worktree
(`../HADES_v4_phase21`). Not merged, not pushed. The user's working tree was not touched (deviation 203).
**Status:** complete, CPU only, 23:37 → 00:35 IST of an 8-hour window (stop 07:37). No STOPPED file.
**Deviations** start at **197** (the integration report ends at 196).

## Header

| | |
|---|---|
| machine | Apple M4 Pro, 14 cores, 24 GB, macOS (Darwin 27.0) |
| software | Python 3.14.6, LightGBM 4.7.0, NumPy 2.5.3, pandas 2.3.3; torch 2.14.0 imported by scorers only. **No neural training, no MPS** |
| concurrency | CPU only. Up to three LightGBM processes (`n_jobs = 6` each) and one builder or scorer at once. Wall-clock only: every BASE reproduced bit-exactly, and every refit made to rebuild a lost log record was bit-identical to the stored predictions (deviation 198) |
| dirty-tree rule | every fit, build and score ran from a clean `ml/` (`phase12_common.require_clean`); every artifact records its commit and `code_dirty = false` |
| data | read from the main checkout through `HADES_DATA_ROOT` (`ml/data/phase21_paths.py`); table SHA-1s recorded per world |

| artifact | commit |
|---|---|
| pre-registration | `f676ca5` |
| builder + tests; first self-test (poison / self-exclusion PASS) | `1a497b5` |
| group stores v8 (snap + place) / v8w1002 | `f7a7fdf` / `c2a756c` |
| k / offsets / T on validation (`k_select_*.json`) | `6248fae` |
| LightGBM arms: v8 arrival, place, fill / `_nl` diagnostics / fill season + cadence controls / world 2 | `6248fae` / `cbb1167` / `7169503` / `cbb1167`, `7169503` |
| scores (`score_snap_*.json`, `score_place*_v8.json`) | `676ae0c` |
| hybrid (`stage6_hybrid.json`) / replication (`stage7_replication.json`) | `d31947a` / `dc0ad62` |
| isolation audit | run at `204c233` (all reports committed), all PASS (§7) |

---

## 1. Decisions

| use case × arm | verdict | the number | interval / band |
|---|---|---|---|
| **arrival (snapshot) × standalone group rule** (the idea without ML) | **worse than BASE** | lateness AUC **0.680** vs 0.706; A3 13.54 vs 13.23 d | block −0.026 [−0.030, −0.022] |
| arrival × group-only LightGBM (L5 block, no other features) | worse than BASE | 0.687 | [0.6868, 0.6877] vs [0.7048, 0.7057] |
| arrival × BASE + L4 (no month) | **FAIL** (disjointly worse) | 0.7038 vs 0.7053 | [0.7030, 0.7042] |
| **arrival × BASE + L5 (the 5-D idea)** | **FAIL** (no gain) | 0.7042 vs 0.7053; A3 13.29 vs 13.25 | [0.7036, 0.7052] |
| arrival × L5 over season + cadence | worse than season + cadence | 0.7145 vs 0.7156 | disjoint |
| arrival × L5 vs L4 (the month) | undetermined | +0.0004 | |
| **arrival × hybrid** (incumbent 0.00, Phase 19 neural 0.55, LightGBM + season + cadence + L5 0.45) | better than every parent; **worse than the Phase 19 blend** | lateness 0.729 vs blend 0.731; A3 12.41 vs 12.34 d | −0.002 [−0.004, −0.0002]; −0.08 d [−0.14, −0.03] |
| arrival UC1 top of the list: hybrid vs incumbent P(late) at 5% | **worse** | 0.944 vs 0.969 | −0.026 [−0.038, −0.017] |
| **at placement × BASE** (no-leak; new decision point) | **ALERT** (Phase 15 rule) | lateness 0.695; UC1-P P @ 5% 0.79 at base 0.35 (lift 2.2) | per-month P @ 5% 0.66–0.88 |
| at placement × BASE + L4 | **PASS on A3** (no control pre-registered) | A3 **11.83** vs 12.65 d; lateness tie | [11.82, 11.87] vs [12.61, 12.69] |
| **at placement × BASE + L5 (true order month)** | **FAIL** | lateness 0.6931 vs 0.6948 (worse); vs L4: tie (bands), worse (ensemble block) | [−0.0016, −0.0003] vs L4 |
| at placement × standalone group rule | best median error, poor flag | A3 **10.02** d (block +2.65 d [2.18, 3.13] better); lateness 0.532 | |
| fill × BASE + L4 / BASE + L5 | **FAIL** (CRPS worse) | CRPS 0.1424 / 0.1427 vs 0.1397; AUC +0.024; UC2b P@5% +0.02 / +0.03 | |
| **fill × BASE + acknowledgement gap** | **PASS** | CRPS 0.1380, AUC 0.633, UC2b P@5% 0.416 (vs 0.1397 / 0.605 / 0.384) | [0.1379, 0.1381]; [0.632, 0.634]; [0.400, 0.421] |
| fill × season + cadence + L5 vs season + cadence | **FAIL as pre-registered** (permuted month gains too); the **channel** part is real (cross-channel shuffle erases it) | CRPS 0.1346 vs 0.1367; AUC 0.668 vs 0.643; P@5% 0.472 vs 0.440 | season + cadence + L4 = 0.1346 |
| fill × standalone group distribution | undetermined vs BASE (v8); worse (world 2) | CRPS 0.1400 vs 0.1396 | |
| **fill × hybrid** (w_neural 0.01: it is LightGBM + season + cadence + L5) | better than the Phase 19 blend on CRPS and the UC2 list; **worse on the UC2b list** | CRPS 0.1343 vs 0.1354; UC2 P@5% 0.943 vs 0.927; UC2b 0.478 vs 0.530 | +0.0010 [0.0002, 0.0020]; +0.025 [0.003, 0.067]; −0.041 [−0.075, −0.008] |
| **any Phase 15 class change** (UC1, UC2, UC2b) | **none** | all WATCHLIST | |
| **replication** (world 2) | ack-gap **REPLICATES** (×0.50–0.98); group arrival **loss** replicates; group fill AUC / precision gain replicates, its CRPS effect **flips sign**; the month is inert in both | | |

## 2. Prediction scorecard

| # | prediction | **right / wrong** | the number |
|---|---|---|---|
| **P1** | standalone group rule < 0.70 lateness AUC | **RIGHT** | 0.680 (scored against the spec's reference: no snapshot row has a promise, deviation 204). At placement, where the promise exists, 0.532 |
| **P2** | BASE + group features gain < 0.02 and not disjoint | **RIGHT** | BASE + L5: −0.0011, bands overlap. (BASE + L4 is disjointly **worse**) |
| **P3** | L5 adds < 0.01 over L4 at snapshots | **RIGHT** | +0.0004 |
| **P4** | at placement the true month adds a disjoint gain over L4 | **WRONG** | bands overlap (0.6931 vs 0.6940); the ensemble block says L5 is **worse**, −0.0009 [−0.0016, −0.0003]. Also wrong on the leaking as-specified BASE |
| **P5** | gains concentrate in thin channels; normal ties | **WRONG** | thin: tie on all three metrics (lateness 0.7581 vs 0.7587); normal: tie, tie, and UC1 P@5% **worse** (0.879 vs 0.897). The cold slice is empty |
| **P6** | group fill adds < 0.01 AUC and < 0.001 CRPS; only the ack gap passes | **WRONG** | clause (ii) holds: the ack gap PASSES, L4 / L5 FAIL. Clause (i) fails: L4 / L5 add **+0.024 AUC** (and make CRPS worse by 0.003) |
| **P7** | the hybrid improves A3 but ranks the top 5% worse than the incumbent's P(late); no class change | **RIGHT** | A3 +0.66 d [0.46, 0.86] vs the incumbent ensemble; P @ 5% 0.944 vs 0.969; no class change |
| **P8** | any passing arm replicates in sign; effect sizes differ by > ×2 | **RIGHT, by a hair** | the only pass (ack gap) replicates in sign on all three metrics; the CRPS ratio is **0.497**, just outside [0.5, 2.0]. AUC and precision ratios are 0.96 and 0.98 |

**5 right, 3 wrong.**

## 3. Deviations, continuing from 197

| # | prior statement | measured / done | where |
|---|---|---|---|
| **197** | D3: lane cluster joined from `logistics_lanes` | the lane table is one row per channel in channel order, but 13,267 channels share a (site, plant, mode) key, so a key join fans out. Aligned by row; site, plant, mode and distance asserted equal. Incoterm, PO type, currency and supplier country are constant (one value each) and were dropped from the cluster | stage 1 |
| **198** | constraint 6: record per artifact | two concurrent LightGBM processes wrote one shared `proxy_fit.json`; the fill process overwrote the arrival and placement records. Prediction files were unaffected. Logs are now one per (world, task). The 61 lost records were rebuilt by refitting each arm and **asserting its predictions bit-identical** to the stored file (all 61 were); the fill records were copied from the shared log, which the fill process wrote last | proxy |
| **199** | D10: at-placement BASE = flat as-of channel features at τ | **the generator writes each line's eventual lead into its channel's weekly row for the week the line was ordered** (`generator_v8.py`, weekly stores: `leadw[pch, vw_ord] = pl_`, forward-filled → `lead_time_actual_days`, `lead_time_ratio`, `otd_rate_last13`). At placement this hands BASE the row's own future lead (lateness AUC 0.987, A3 0.27 d), so every pre-registered at-placement number is void. They are kept in `score_place_v8.json`; Stage 4 uses `_nl` arms with the three columns removed. **At snapshots** the same columns carry future leads of other lines ordered by t0's week: worth +0.0066 lateness AUC to the LightGBM BASE (BASE_nl 0.6987). Every stored arrival model (LightGBM and neural) reads these columns; neither the generator nor the incumbents may be changed here. **Phase 0–1's "no feature is a function of the label" check (max correlation 0.319, PASS) could not see it** | stage 4, stage 3 |
| **200** | D6: k chosen on validation from {1 … 300} | arrival chose k = **300, the grid's edge**. The grid was not extended. Every snapshot row then resolves to L2 (supplier), so the finer levels enter LightGBM only through their raw n and their small weights | stage 2 |
| **201** | D8 / D14 arm lists | added: fill `sc_L4`, `sc_L5_perm`, `sc_L5_xsh` (the controls the season + cadence + L5 result needed); world 2 `ack`, `ack_xsh` (v8's only pass, needed for P8) and `L5_xsh` (L5's control b); and the `_nl` diagnostics (199). None changes a pre-registered verdict | stages 4, 5, 7 |
| **202** | rule 3: never select on test | the first slice code picked "the best arm" by **test** mean AUC; its output was seen. Replaced by P5's pre-registered arm (BASE + L5) before any slice verdict was written; season + cadence + L5 is shown as context only | stage 6d |
| **203** | "never touch the user's working tree" | not touched: its `git status` is still only `M CLAUDE.md`. Phase 21's artifacts were written into the main checkout's **ignored** `ml/artifacts/phase21/` through a symlink, and the worlds were read from there (`HADES_DATA_ROOT`), as the integration did (deviation 194). No `.git/info/exclude` change was needed (`ml/artifacts` and `venv` were already excluded) | header |
| **204** | P1: "lateness = expected minus promise" | at snapshots no row has a promise (its line is raised 1–13 weeks after t0, Stage 1: 0%), so the rule is scored with the lateness spec's as-of reference (D9). At placement it is expected minus promise | stage 3 |
| **205** | rule: control (a) permuted month gates L5 | the L5 arm carries the L4 columns, so control (a) removes only the month. When it gains as much as L5, the rule's FAIL says **the month is not the information**, not that the group block is empty. Control (b), the cross-channel shuffle, is the test of the channel history | stages 5, 7 |
| **206** | D11: cold / thin / normal slices | **no snapshot row is cold** (every channel has receipts by t0; the minimum at L4 is 1). The thin slice is relative (n4 < 47, the validation P20) | stage 6d |
| **207** | Stage 4 per-snapshot spread | at placement it is read per creation month. The first scorer keyed months as integers, which collapsed them to one date; fixed (`676ae0c`) before the table was written | stage 4 |
| **208** | brief: "402/402 identities expected" | **371 / 371**: the 371 stored configs already include the 31 Phase 16 variants (deviation 189). Phase 21 adds no neural config | §6 |

## 4. Every `inventory_position_weekly` read

**None.** No Phase 21 module names it outside a docstring (`grpstats.py`'s docstring says it is never read), and
`part_demand_weekly` is not read either. The scanner flags a constructed read (`test_phase21_isolation.py`). No
PRIVILEGED arm exists in this phase.

## 5. Your three questions, in plain language

**(1) Does the 5-D group idea work standalone?** **No, not as a predictor.**
- **Arrival, at a snapshot:** the KM-median rule reaches lateness AUC 0.68 against BASE's 0.71. A LightGBM on the group
  statistics alone reaches 0.69. Both are disjointly worse.
- **Fill:** the group's own shrunk histogram ties BASE on v8 and is worse on world 2.

There is one exception. When a PO is **raised**, the channel's shrunk KM median lead is the best **date** estimate
measured here: median error 10.0 days against 12.6 for the flat LightGBM. It is close to useless as a **late flag**
(AUC 0.53).

Two things, both measured, explain it:
- The censoring fix matters. With 50% open lines the completed-only median is 8 days optimistic in the unit test.
- But the 5-D cells are too thin: a median of 4–5 receipts per (channel × month) after eight years. Validation chose
  shrinkage so strong (k = 300) that every row backs off to the supplier.

**(2) Does it work as a hybrid with the existing models?** **Partly, for fill; not for arrival.**
- **Arrival:** the best LightGBM + group arm, blended with the neural ensembles, is better than each parent but
  slightly worse than the existing Phase 19 blend (−0.002 lateness AUC, +0.08 days). Its top-5% list is worse than
  the incumbent's P(late) (0.944 vs 0.969). On top of BASE the group block makes arrival worse, also on world 2.
- **Fill:** the group statistics on top of the Phase 19 season + cadence features give the best fill distribution
  measured so far. CRPS is 0.1343 against the Phase 19 blend's 0.1354, and the arrives-in-full list is better at 5%
  (0.943 vs 0.927).
  - The gain comes from the **channel** history (L4). Moved to the wrong channel, it vanishes; without the month, it
    stays.
  - It is a hair worse on the materially-short list (0.478 vs 0.530).
  - On BASE alone the same block worsens CRPS on v8 and improves it on world 2, so its calibration effect is
    world-specific.
- **The acknowledgement gap** (how much of their recent lines a supplier and channel acknowledged short) is the one
  group feature that passes its gate cleanly, on all three fill metrics, and it **replicates** on world 2.

**(3) Does the true order month help only at the moment of order placement?** **It does not help there either.**
- At a snapshot the month key is only a seasonal proxy (the row's line does not exist yet), and it adds +0.0004.
- At placement, where the true month is known, L5 is no better than L4: the bands overlap, and the ensemble block
  says slightly worse.
- The season is real: the late rate runs from 0.21 (Feb orders) to 0.50 (Sep). But it is already visible through the
  channel's as-of state, and a (channel × month) cell holds too few deliveries to add to it.

**One finding outweighs the question.** Three columns of `channel_performance_weekly` carry each line's **future** lead
into the week it was ordered (deviation 199).
- At placement this makes any model look near-perfect (AUC 0.99).
- At snapshots it is worth +0.007 lateness AUC to the shipped LightGBM BASE, and every stored arrival model reads
  those columns.
- An honest at-placement late flag is still an ALERT by the Phase 15 rule (precision 0.79 at 5%, base 0.35).

## 6. Shipping table (only verdicts that survived a 5-seed band or the snapshot-block bootstrap)

Nothing ships from this phase automatically. **Recommend only.**

| use case | ships now | watchlist | blocked on data | closed |
|---|---|---|---|---|
| **arrival** | **unchanged from Phase 20**: the Phase 19 blend as the point estimate (it beats the group hybrid by 0.08 d A3); the incumbent neural ensemble's P(late) as the ranked late list | the expected-week at **placement** from the channel's shrunk KM median (A3 10.0 d vs 12.6 d, block-disjoint); a decision about a placement-time product, not a ship | a fix to the **leaking weekly lead columns** (deviation 199): needs a generator / feature-store change outside Phase 21's write scope, then a re-measure of every arrival incumbent | the 5-D group block for arrival (snapshot and placement); the order-month key; group-only models; the standalone rule as a late flag |
| **fill** | **unchanged from Phase 20** (the Phase 19 blend as the materially-short list: 0.530 at 5%, still the best) | **LightGBM + season + cadence + the channel-level group statistics** as the fill distribution and arrives-in-full list (measured with the L5 block: ensemble CRPS 0.1343, better than the Phase 19 blend by 0.0010 [0.0002, 0.0020]; the L4 block alone is equal, 0.13459 vs 0.13461 as 5-seed means; one world only; season + cadence + L5 fails its pre-registered gate on the month control); the **acknowledgement-gap block** (passes and replicates; +0.0017 CRPS, +0.028 AUC) as a candidate input to the next fill retrain | — | the month dimension for fill; the group histogram as a standalone distribution |
| **capacity** | not in scope | | | |

## 7. Isolation audit

`ml/tests/test_phase21_isolation.py` → `ml/artifacts/phase21/isolation_audit.json`, run at `204c233`, with every report committed: every check PASS.

| check | result |
|---|---|
| `git diff 124be5b HEAD` and the working tree on db/gen_v6, gen_v7, gen_v8, db/validator.py, docs/specs/**, db/dataset_structure.md, ml/configs/shipped.json, results/**, reports/part1, **every report at the base**, ml/models/share.py, ml/models/tcn.py, ml/artifact_identity.py, ml/train/phase19_identity.py, **ml/serve/** | **empty** |
| db/ | worktree db/ clean, tracked and untracked. The main checkout's db/ clean; `db/gen_v8` holds exactly seed_1001…seed_1005; world 2's hashes equal Phase 20's |
| stored configs recomputed | **371 / 371 identical**, all nameable by this base; the 15 Phase 19 bundles at their `phase19_identity` names; Phase 21 adds no neural config |
| AST scan of ml/train / models / data / baselines | **0 violations**, forbidden strings extended to `phase21/PRIVILEGED`. Three constructed Phase 21 offenders flagged; a docstring mention passes. A fresh interpreter importing `grpstats`, `phase21_paths`, `phase21_proxy` loads nothing from `reports/` and no torch |
| `inventory_position_weekly` / `part_demand_weekly` in Phase 21 modules | **0** (scanner self-tested on a constructed read) |
| future poison / self-exclusion, both worlds | **PASS**: 0 / 400 rows change at each of 3 τ; the constructed offenders change 400 / 400 |
