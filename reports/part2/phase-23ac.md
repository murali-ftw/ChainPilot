# Phase 23AC — order-time product made demo-ready, a dormant fill serving path, a standing leak guard, and a metrics pack

## Header

**Audience:** the project owner and whoever demos ChainPilot. Read with `results/observation3.md` (v4.0), which this phase
drafts.

**Measured on:**
- v8 seed 1001 (`v8` raw panel and `v8clean`), test 2025;
- replication on `v8w1002` (the stored seed-1002 world; hashes equal to Phase 20, not regenerated).

CPU only. No model was trained or fitted.

**Companions:**
- Pre-registration: `reports/part2/phase-23ac-preregistration.md` (`1a73f75`).
- Track reports in `reports/part2/phase23ac/`:
  - `t1/t1_order_time.md`;
  - `t2/t2_fill_exception.md`;
  - `t3/t3_leak_guard.md`;
  - `t4/definitions.md`, `t4/metrics_pack.md`, `t4/progress.md`, `t4/charts/`.

**Code:** `d401cb3`, plus gate fix `006ebc9` and chart layout `10f88da`. Track reports: `ec9c68b`.

**Status:** complete.
- Isolation audit: PASS (§7).
- Predictions: 1 right, 4 wrong.
- **Two rows wait for Phase 23B** (§6).

**Number marks.** Every number below is **RECOMPUTED** from stored artifacts in this phase, or **QUOTED** from the named
report. LEAKY figures appear only in the INCUMBENT and BEST PUBLISHED columns, labelled "LEAKY (superseded, Phase 22)".

## 1. Decisions by track

| track | question | decision | key number | mark |
|---|---|---|---|---|
| **T1** order-time card | is the order-time date product better than what Rane has today? | **yes on v8**: better than the promise date and the channel's trailing mean and median on median error, mean error and ±7-day share, all block-disjoint. **World 2: ties the trailing median** on median error | A3 10.02 d vs promise 14.0 / mean 12.29 / median 11.0; vs promise +3.75 d [2.73, 4.31] | RECOMPUTED (reproduces Phase 22 A3 10.0173, coverage 0.80258 exactly) |
| T1 | does the interval hold in every month? | **overall yes, one month no**: 0.803, January 0.767 | months 0.767–0.829 | RECOMPUTED |
| T1 | demo card | `ml/serve/order_time_card.py`, bit-exact to the stored service (max \|Δ\| 0.0); 5 sample cards (seed 2310), one outside its interval (40 d late) | — | RECOMPUTED |
| **T2** fill exception | can the best servable fill model be served with the forward plan under one bounded reader? | **built, dormant, PROPOSED.** One reader (`ml/serve/_plan_reader.py`) with the as_of + 2 d bound; the five neural season + cadence bundles serve within **2.38e-7** of the stored predictions (CPU vs MPS), not bit-exactly. The LightGBM consolidated model cannot be served: never persisted (deviation 225) | features identical (0.0); season poison: bounded 0 / 4000, offender 4000 / 4000 | RECOMPUTED |
| T2 | finding | **eight pre-existing modules read `part_demand_weekly`** outside any bound (listed in §4) | — | RECOMPUTED (AST scan) |
| **T3** leak guard | does a standing guard catch the nine, and nothing else? | **yes under poison**, on both worlds; exit 0. The literal label-alignment rule could not pass its own known answer and was amended (deviation 226) | nine FAIL poison at 0.298–0.814 (v8), 0.286–0.806 (w2); every clean replacement 0 | RECOMPUTED |
| T3 | further leak? | **none found** beyond the known nine | — | RECOMPUTED |
| **T4** metrics pack | do published figures reproduce from stored predictions? | **79 of 79 reproduce; 0 mismatches** | — | RECOMPUTED vs QUOTED |
| T4 | CLEAN headline per use case | see §8; five charts | capacity worst snapshot 0.45 (2025-12-08) | RECOMPUTED |

## 2. Prediction scorecard

| # | prediction (pre-registered) | verdict | evidence |
|---|---|---|---|
| P1 | the order-time product beats the promise date and both channel averages on A3 and ±7 d, block-disjoint (v8) | **RIGHT** | `t1/t1_order_time.md` b. *World 2: the trailing-median comparison does not replicate* |
| P2 | interval coverage within [0.77, 0.83] in every creation month | **WRONG** | January 0.767 (v8); world 2 May 0.835, Nov 0.833 |
| P3 | the dormant fill path reproduces stored predictions bit-exactly | **WRONG** | max \|Δ\| 2.38e-7; stored on MPS, served on CPU. Inputs identical |
| P4 | the guard flags all nine and no clean replacement | **WRONG as pre-registered** | the literal alignment rule fails five clean replacements on 0/1 atoms. Amended rule (dev. 226): right on both worlds |
| P5 | no further leaking column | **WRONG as pre-registered** | the literal alignment rule flags four raw columns that change 0 under poison. Amended rule and poison: no further leak |

**1 right, 4 wrong.** P4 and P5 are wrong because a pre-registered rule was mis-specified, not because a leak was found.

## 3. Deviations, continuing from 222

| # | rule | what was done instead, and why |
|---|---|---|
| 223 | new branch `phase23ac` in a new worktree | committed on `HADES-v4-ml-pipeline` per the owner's standing instruction "dont create seperate branches here after"; exact paths staged; `CLAUDE.md` and `db/` untouched; no merge, no push |
| 224 | charts with matplotlib | rendered by a scratch venv that reads only the CSVs; matplotlib not installed in the project venv (could move numpy) |
| 225 | persist the LightGBM consolidated fill model if bit-exact | not attempted: never persisted; persisting needs a refit, which is forbidden |
| 226 | label-alignment rule: share of rows equal to any pending outcome > 0.20 | matches at exactly 0 or 1 excluded (`ALIGN_ATOMS`), after a smoke run on real data. P4/P5 scored on the literal rule; amended verdicts beside it |
| 227 | world 2 uses Phase 22's world-2 product | Phase 22 chose a KM + LightGBM blend for world 2; T1 uses KM (k = 3, chosen on world-2 validation) with its own conformal quantiles, as the pre-registration's definition requires |
| 228 | season poison: the offender must change rows at every snapshot | a snapshot with no future-recorded plan versions (2025-12-08) is n/a for the offender; it must still fire on every applicable snapshot and at least one (`006ebc9`) |
| 229 | per-track CSV outputs committed | `*.csv` is gitignored repo-wide; the CSVs are written to `reports/part2/phase23ac/**` but not force-added. Their JSON twins in `ml/artifacts/phase23ac/` (also gitignored) and the committed `.md` tables carry the numbers |

## 4. Every `inventory_position_weekly` / `part_demand_weekly` read

- **`inventory_position_weekly`:** read by **no** new module (AST scan, `test_phase23ac_isolation.py`).
- **`part_demand_weekly`, new code:** named only in `ml/serve/_plan_reader.py` (dormant; nothing imports it outside tests),
  under the as_of + 2 d bound. `ml/eval/phase23ac_t2_serve.py` holds a constructed **unbounded offender** used only for
  the poison test.
- **`part_demand_weekly`, pre-existing readers** (finding, unchanged):
  - `ml/data/fwd_load.py`, `ml/data/phase20_world.py` (hash only);
  - `ml/eval/phase12_a5_policy_facts.py`;
  - `ml/opt/allocation.py`, `ml/opt/audit_inputs.py`, `ml/opt/schedule_lp.py`;
  - `ml/sim/montecarlo.py`;
  - `ml/serve/features.py` via `fwd_load`.
- **Stored arrays:** the measurement scripts read the stored Phase 18/22 season arrays (built by `fwd_load`); they did not
  read the table again.

## 5. What each result changes

- **T1:** the order-time date product is demo-ready against the promise date, with a planner card. Demo caveats:
  - it is early on average (−8.4 d);
  - January's interval is under-covered;
  - on a second world it is no better than the channel's trailing median on median error.
- **T2:** the owner can grant or decline the forward-plan exception. The serving path is ready either way (§3.3 of
  observation3).
  - **If granted:** the eight pre-existing readers must come under the same bound.
  - **If declined:** fill falls back to the clean incumbent.
- **T3:** `scripts/run_leak_guard.sh` is ready to wire into CI. It is not wired here (one line in `docs/standards/leak_guard.md`).
- **T4:** the metrics pack is the single table to refresh when Phase 23B lands.
- **`shipped.json` is unchanged.** Nothing ships automatically from this phase.

## 6. PENDING PHASE 23B

| row | old figure | label |
|---|---|---|
| predict-the-rescue (B1a) | 0.823 precision at recall 0.246 | LEAKY-UNCONFIRMED |
| shortage simulation | 1.129× pre-rescue; detector ceiling 0.28 | LEAKY-UNCONFIRMED |

These appear at the top of `results/observation3.md` and in `t4/metrics_pack.md`.

**Recompute mismatches:** none (79 of 79).

## 7. Isolation audit

`HADES_DEVICE=cpu ./venv/bin/python ml/tests/test_phase23ac_isolation.py` at `ec9c68b` (tree clean) →
`ml/artifacts/phase23ac/isolation_audit.json`. **All six tests PASS:**

| test | result |
|---|---|
| protected paths | 292 protected files unchanged vs `545e282`; 32 files added, 0 modified |
| `db/` | untouched; five seed directories present; world-2 hashes equal to Phase 20 |
| stored artifacts | the 188-file manifest hashed before measurement is unchanged |
| identities | 371 of 371 stored configs recompute identically; Phase 19 bundles 15; Phase 22 bundles 25 |
| AST scans | `part_demand_weekly` only in `_plan_reader.py` among new modules; `inventory_position_weekly` nowhere |
| no training | 9 new modules, 0 training or fitting calls (4 constructed offenders flagged) |

The commit adding this report and `results/observation3.md` adds two new files and modifies none. Nothing is merged or
pushed.

## 8. CLEAN headline (v8 seed 1001, test 2025, RAW; RECOMPUTED in `t4/metrics_pack.md`)

| use case | metric | value | class |
|---|---|---|---|
| arrival (snapshot) | late list precision @ 5% (UC1), base 0.730 | 0.948 [0.930, 0.965] | WATCHLIST |
| arrival (snapshot) | A3, clean Phase 19 recipe | 12.57 d [12.22, 13.05] | — |
| arrival (order time) | A3, product | 10.02 d [9.67, 10.34] | ships (date + interval) |
| arrival (order time) | strict late flag precision @ 5%, base 0.349 | 0.615 [0.607, 0.619] | WATCHLIST |
| fill | UC2 precision @ 5%, base 0.749 | 0.944 [0.928, 0.964] | WATCHLIST |
| fill | UC2b precision @ 5%, base 0.245 | 0.511 [0.446, 0.549] | WATCHLIST |
| fill | CRPS, consolidated blend | 0.13444 | — |
| capacity | precision @ 5%, base 0.405 | 0.735 [0.703, 0.784] | WATCHLIST |
