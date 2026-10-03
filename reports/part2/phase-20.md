# Phase 20 — do the gains change decisions, do they replicate, can a load forecaster recover more, what is a reorder signal worth?

**Audience:** whoever decides what ships for arrival, fill and capacity, and what to ask the client for.
**Measured on:** v8 seed 1001 (primary) and **v8w1002**, a second world generated from v8 seed 1002 (Stage 2). Fixed
split (train ≤ 2023, val 2024, test 2025); seeds 7 / 17 / 27 / 37 / 47; TEST unless marked; RAW throughout.
**Companions:** `phase-20-preregistration.md` (`1c25e70`); `phase20/stage1_decisions.md`, `stage2_replication.md`,
`stage3_forecaster.md`, `stage4_reorder.md`, `PRIVILEGED__reorder_probe.md`.
**Branch / base:** `phase20`, from `phase19` at `e9c0862`. Not merged.
**Status:** complete, CPU only, from 00:41 to ~01:40 IST of a 10-hour window (stop 10:41). No STOPPED file.

## Header

| | |
|---|---|
| machine | Apple M4 Pro, 14 cores, 24 GB, macOS (Darwin 27.0) |
| software | Python 3.14, LightGBM 4.7.0, torch 2.14.0 (imported by scorers only; **no neural training, no MPS**) |
| concurrency | CPU only. At most two LightGBM processes at once (`n_jobs = 6` each: forecaster + proxy), plus scorers. Wall-clock only: v8 BASE reproduced the stored LightGBM-flat arms **bit-exactly** on every task (deviation 186) |
| dirty-tree rule | every run refused a dirty `ml/`; every artifact records its commit and `code_dirty = false` |

| artifact | commit |
|---|---|
| pre-registration | `1c25e70` |
| Stage 1 decisions (`stage1_decisions.json`) | `49aaa32` |
| second world: generation / features / cache | `f856c29` / `75e144a` |
| proxy fits (world 2; v8 lgbm_rf, fwd_pred, stock arms) | `d50e50f` / `50cd5bb` |
| load forecaster (both worlds) | `d50e50f` |
| base rates | `d1e69ac` |
| Stage 2–4 scores (`scores.json`) | `b54d55b`, replication n/a fix `c303519` |
| PRIVILEGED probe capture / fit; world-2 hindsight; PRIVILEGED scores | `ab9c919`; `1690956`; `a98f2b9` |
| isolation audit | §7 |

---

## 1. Decisions

| question | verdict | the number | interval |
|---|---|---|---|
| **Does any use case change class (ALERT / WATCHLIST / RETIRED)?** | **No** | UC1 WATCHLIST, UC2 WATCHLIST, UC2b WATCHLIST, UC3 ALERT, for every arm | — |
| arrival UC1: Phase 19 blend vs incumbent ensemble, precision @ 5% | **blend WORSE** | 0.943 vs 0.969; worse in 9 of 9 snapshots | −0.027 [−0.036, −0.018] (block) |
| fill UC2b (materially short): Phase 19 blend vs incumbent ensemble, P @ 5% | **better, still WATCHLIST** | 0.530 vs 0.464; top 1% 0.644 at 2.6× base | +0.063 [0.043, 0.090] |
| fill: shipped `b5flat22` on the decisions | **weakest fill arm** | UC2b P @ 5% 0.384; UC2 recall @ 0.85 0.189 | 5-seed |
| capacity UC3: incumbent ensemble | **ALERT** (lift 1.98) | P @ 5% 0.865; per-snapshot min **0.46** (Dec 2025) | |
| **replication**: arrival fwd_load (supplier-specific) | **REPLICATES** | lateness gain +0.0131 vs +0.0133 (×0.99) | world-2 bands disjoint |
| replication: arrival fwd_season / cadence | **REPLICATE** (lateness AUC) | ×2.6 / ×0.62; cadence A3 undetermined | |
| replication: fill fwd_season | **REPLICATES** | CRPS ×0.32, AUC ×0.74 | |
| replication: **fill cadence** | **DOES NOT** (undetermined; world-2 Gate v2 FAIL) | | |
| replication: capacity fwd_load (supplier-specific) | **REPLICATES** on 4 of 4 comparable points | P @ 5% ×1.48; recall @ 0.70 ×0.38 | |
| Phase 19 blend, LightGBM-only recipe | **no gain over the best parent** in either world | weights 0.88 / 1.00 (arrival), 0.49 / 0.20 (fill), 0.00 / 0.08 (capacity) | neural half **not replicated** |
| **load forecaster** (`fwd_pred`) | **PASS arrival, PASS fill, FAIL capacity**, in both worlds | arrival +0.0020 lateness AUC (3.2% of the hindsight gap); fill −0.0004 CRPS (14%) | |
| forecaster's own accuracy (test) | better ranking, worse quantity | R² log 0.28 vs plan −0.24; R² raw 0.06 vs plan 0.35 | |
| **PRIVILEGED exact reorder probe** | **17% of the timing headroom** | lateness AUC 0.739 (BASE 0.705, timing oracle 0.910); A3 12.71 d | [0.7390, 0.7397] |
| legitimate observed stock arm (ledger) | **PASS on arrival, worth +0.0016** | 0.7069 vs 0.7053; fill FAIL (control gains too); capacity FAIL | [0.7065, 0.7072] |

## 2. Prediction scorecard

| # | prediction | **right / wrong** | the number |
|---|---|---|---|
| **P1** | the Phase 19 arrival blend beats the incumbent ensemble at 5% coverage | **WRONG** | 0.943 vs 0.969: the blend is **worse**, −0.027 [−0.036, −0.018], in every snapshot |
| **P2** | fill stays WATCHLIST with the Phase 19 blend | **RIGHT** | UC2b: NO, CEILING (p 0.70 reachable on validation but test recall 0.007), Stage B TUNABLE → WATCHLIST |
| **P3** | every Gate v2 pass replicates (same sign, disjoint) on the second world | **WRONG** | 5 of 6 replicate; **fill cadence does not** (CRPS −0.00003, AUC +0.0011, bands overlap) |
| **P4** | world-2 gains within ×2 of v8-1001's | **WRONG** | 5 of 13 comparable gains outside: arrival season 2.6×, fill season CRPS 0.32×, fill cadence CRPS 0.29×, cadence A3 0.01×, capacity recall @ 0.70 0.38× |
| **P5** | forecaster recovers < 25% on arrival and ≥ 15% on capacity | **RIGHT** | arrival 3.2% (lateness) / 1.8% (A3); capacity 37% at the headline point (precision @ 5%). Other capacity points 3–28% and −8%; world 2's capacity 7% |
| **P6** | the forecast feature passes its gate on capacity only | **WRONG** | the reverse: PASS on arrival and fill (both worlds), FAIL on capacity (2 of 6 in v8, 0 of 6 in world 2) |
| **P7** | the privileged reorder probe recovers ≥ 50% of the timing headroom (≥ 0.81) | **WRONG** | 16.6% (0.739) |
| **P8** | no observed stock source in v8 is a legitimate as-of reorder signal; the client ask is the only route | **WRONG** as worded | the v8 ledger (opening balance posted 2015-12-28; recorded_ts on every row) gives a legitimate arm that **passes** on arrival, worth **+0.0016**. In substance the client ask remains the only route to a material gain |

**2 right, 6 wrong.**

## 3. Deviations, continuing from 178

| # | prior statement | measured / done | where |
|---|---|---|---|
| **178** | brief: "copy the Phase 15 rule verbatim" | Phase 15's REACHABLE rule exists only as text (its docstring), and was applied by hand in its report. It is implemented here from the text. The lift bar is "the highest bar whose validation point meets the reachable condition" (test precision ≥ 0.85 YES / ≥ 0.70 PARTIAL, recall ≥ 0.10, ≥ 50 alerts, every seed). That is the reading that reproduces the pre-registered anchors (UC3 lift 2.0, UC1 / UC2 below 1.5), and it reproduces Phase 15's numbers exactly | stage 1 |
| **179** | pre-registration def. 7: legitimate arm uses `part_plant.reorder_point_qty` | that column is the reorder point **after the simulation's last week** (`generator_v8.py` l. 1675), i.e. 2026 information. Replaced, before the arm was fitted, by an as-of estimate (13-week ledger issue rate × planning lead + safety stock) | stage 4 |
| **180** | brief: "a fresh generator seed" | the world generated from seed 1002 into `data_worlds/` is **byte-identical** to the stored `db/gen_v8/seed_1002` (every key table; manifest differs only in its timestamp). The generator is deterministic. "Fresh" holds in the sense that matters: no Phase 16–19 choice used it | stage 2 |
| **181** | P1: "the blend beats the incumbent at 5% coverage" | Phase 15's UC1 is late vs the **contracted** reference with censoring resolved, scored on P(late) for distribution arms and on prediction − reference for point arms (Phase 15 unchanged). The Phase 18–19 gains were on the as-of reference, uncensored rows. Different decision, same arms; recorded so the two are not read as one | stage 1 |
| **182** | pre-registration def. 6: supplier forecast = sum of channel forecasts; log1p target | as fitted, the log back-transform shrinks large orders and the sums compound it (supplier R² −2 to −3; channel raw R² 0.06). Nothing was retuned after seeing it | stage 3 |
| **183** | brief: "true inventory position … (from the simulator)" | `_sim.npz` does not hold channel position or reorder points. They were captured by exec'ing the generator source with **one in-memory hook line** (copies only, no RNG draw) and a stop after `_sim.npz`. The run was accepted only because all 48 `_sim.npz` arrays equal the stored world's. The generator file is untouched | stage 4 |
| **184** | replication on every metric that passed in v8 | capacity recall @ 0.85 has no five-seed BASE band in v8 (one seed unreachable; Phase 19's rule): n/a. Recall @ 0.80 has none on world 2: undetermined. The scorer was fixed (`c303519`) after crashing on the missing band; no verdict was read before the fix | stage 2 |
| **185** | Stage 1 capacity "incumbent ensemble" | the ensemble averages quantiles (Phase 19 deviation 174's convention) | stage 1 |
| **186** | "record concurrency" | CPU only; up to two LightGBM processes (6 threads each) at once. It affects wall-clock only: BASE bit-exact | header |
| **187** | Stage 2d: "the Phase 19 blend recipe on the new world" | no neural model exists on world 2 and this phase is CPU-only, so the recipe was re-fitted on LightGBM arms only and **the neural half of every Phase 19 blend is not replicated** | stage 2 |

## 4. Every `inventory_position_weekly` read

**None.** No Phase 20 module names the table outside a docstring (`test_phase20_isolation.py` checks all five). The
brief expected the Stage 4b probe to be the one read, but the probe read the simulator's **internal arrays** through the
hook, not the table. The table was **written**, not read, by the second-world generation (into `data_worlds/`) and was
not reached by the probe's stopped re-run.

## 5. Shipping table (only verdicts that survived the snapshot-block bootstrap or a 5-seed band)

| use case | ships now | watchlist | blocked on data | closed |
|---|---|---|---|---|
| **arrival** | **Point estimate:** the Phase 19 blend (A3 12.34 d; survived the block bootstrap in Phase 19), with the expected-week conformal interval (~80% at ~45 d). **Ranked late list (UC1):** keep the **incumbent neural ensemble's P(late)**: it beats the blend at 5% coverage (+0.027 [0.018, 0.036]). Forward load and season features, which replicate on both worlds | UC1 itself stays WATCHLIST (lift ≤ 1.36× on a 73% base) | **channel-level stock position and as-of reorder points from the client**: worth up to +0.034 lateness AUC (upper bound, 17% of the timing headroom). The other 83% is the future demand path, not reachable from any t0 data | the load forecaster as a separate component (3–4% of the gap); the ledger-based stock arm (+0.0016); cadence's A3 gain (one world) |
| **fill** | the Phase 19 blend as the **materially-short watchlist** (UC2b P @ 5% 0.530 vs 0.464, top 1% 2.6× base) | UC2 and UC2b stay WATCHLIST; `fwd_pred` passes both worlds but adds +0.0004 CRPS (watch, do not build for it). **Recorded for a decision, not applied:** the shipped `b5flat22` is the weakest fill arm on every decision metric; `shipped.json` is protected | the supplier's weekly queue (Phase 18 oracle) | cadence for fill (does not replicate); supplier-specific fwd_load for fill (fails in both worlds); stock for fill |
| **capacity** | the incumbent neural ensemble as the **UC3 alert** (lift 1.98), with the caution that one quarter (Dec 2025) ran at 0.46 at 5% coverage | supplier-specific fwd_load replicates on world 2 in the proxy (P @ 5% +0.18 there), but tied in the neural head (Phase 19): a candidate for a future neural retrain | **the monthly capacity draw K** (Phase 18) | the load forecaster for capacity (fails both worlds); stock for capacity; fwd_season and cadence for capacity |

## 6. What each result changes, and what is one-world only

- **The AUC gains did not change a decision.** Arrival's blend is a better point estimate but a worse alert ranking. Use
  the incumbent's distribution for the ranked list. Fill's materially-short list is genuinely better (+0.06 precision at
  5%), but it stays below any alert bar.
- **The forward-plan inputs are a method, not a one-world accident.** Supplier-specific forward load replicates on
  arrival at the same size and on capacity even larger. The season replicates on arrival and fill. **Effect sizes do not
  transfer** (×0.3–×2.6), so absolute claims must be measured per client.
- **One-world only (not yet replicated):** cadence on fill; cadence's A3 gain; the **neural half** of every Phase 19
  blend and Phase 19's neural fill GAIN; every Stage 1 decision number; the reorder probe.
- **Learning the load does not pay.** A purged, blocked forecaster recovers 3–4% of arrival's hindsight gap and 10–14% of
  fill's. Not worth a component.
- **The client ask now has a number:** channel-level position against an as-of reorder point is worth **at most +0.034
  lateness AUC** (0.705 → 0.739) and −0.5 days of median error. Stock data alone cannot deliver the timing headroom.

## 7. Isolation audit

`ml/tests/test_phase20_isolation.py` → `ml/artifacts/phase20/isolation_audit.json` (run at `d8bed2a` / §7 commit, every check PASS):

| check | result |
|---|---|
| `git diff e9c0862 HEAD` and the working tree on db/gen_v6, gen_v7, gen_v8, db/validator.py, docs/specs/**, db/dataset_structure.md, ml/configs/shipped.json, results/**, reports/part1, **every report at the base**, ml/models/share.py, ml/models/tcn.py, ml/artifact_identity.py, **ml/train/phase19_identity.py** | **empty**; `git diff e9c0862 HEAD -- ml/models/share.py ml/models/tcn.py` empty |
| `db/` after the second-world generation | **clean**, tracked and untracked; `db/gen_v8` still holds exactly seed_1001…seed_1005; the second world is in `data_worlds/` (ignored) |
| stored configs recomputed | **371 identical**; 340 at their `bundle_name`; **31 cannot be named by this base** (Phase 16 encoder variants, unmerged branch; deviation 149); all 15 Phase 19 bundles at their `phase19_identity` names; Phase 20 adds no neural config |
| AST scan, ml/train / models / data / baselines | **0 violations**, forbidden strings extended to Phase 19–20 privileged paths. Self-test: Phase 18's three offenders and three Phase 20 offenders flagged; a docstring mention passes. A fresh interpreter importing every Phase 20 module loads nothing from `reports/` and no torch |
| `inventory_position_weekly` in Phase 20 modules | **0** (scanner self-tested on a constructed read) |
