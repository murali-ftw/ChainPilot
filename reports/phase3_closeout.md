# Phase 0–3 closeout — inventory decision, Phase-3 annotation, per-task convergence

**Verdict in three lines.** The inventory table is a **hard blocker on Phase 9**, not a
documentation problem, and it is worse than one empty table — there is **no valid stock level
anywhere in the dataset**. `reports/phase-3.md` is annotated. Fill and shortage were retuned per
task and re-run to convergence.

|               |                                                                                                          |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| Device        | **MPS**, Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32, `num_workers=0`              |
| torch         | 2.14.0                                                                                                   |
| Wall-clock    | WALLCLOCK_PLACEHOLDER                                                                                    |
| Peak RSS      | RSS_PLACEHOLDER                                                                                          |
| Arrival       | **not retuned, not re-run** — 14 cells carried forward from `phase1_2.md` unchanged             |
| Changed files | `reports/phase-3.md` (annotation only), `ml/train/temporal_share.py`, `ml/train/run10_closeout.py` |
| Not touched   | `db/gen_v6/`, `db/gen_v7/`, `validator.py`, `synthetic_rules.md`, `dataset_structure.md`       |

---

## 1. Inventory decision — **BLOCKER on Phase 9**

### 1a. What Phases 4 onward consume

| phase       | step                                                    | tables read                                                                                                                                               | reads`inventory_position_weekly`?                                      |
| ----------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 4           | 4.1–4.2 graph encoder                                  | `sourcing_channels` and the Phase-1 graph; node states from Phase 3                                                                                     | no                                                                       |
| 5           | 5.0–5.3 heads                                          | `channel_performance_weekly`, `training_labels`, `snapshots`                                                                                        | no                                                                       |
| 6           | 6.1–6.3 training loop                                  | `training_labels`, `snapshots`, `calendar`                                                                                                          | no                                                                       |
| 7           | 7.1–7.3 baselines                                      | `channel_performance_weekly`, `po_lines`, `grn_lines`, `training_labels`                                                                          | no                                                                       |
| 8           | 8.1–8.2 evaluation, rolling-origin backtest            | `training_labels`, `snapshots`                                                                                                                        | no                                                                       |
| **9** | **9.1 stock roll-forward / Monte Carlo shortage** | **`inventory_position_weekly`**, `part_demand_weekly`                                                                                           | **YES — primary input**                                           |
| 9           | 9.2 Gaussian copula                                     | `suppliers`, `supplier_upstream`, `logistics_lanes`                                                                                                 | no, but**depends on 9.1**                                          |
| 10          | 10.1 delivery-schedule LP                               | `part_demand_weekly`, `part_plant`, `supplier_contracts`, `logistics_lanes`, `calendar`                                                         | no                                                                       |
| 10          | 10.2 allocation                                         | `supplier_allocation`, `alternate_sources`, `sourcing_channels`, `supplier_contracts`, `tooling`, `part_costs`, `revealed_capacity_monthly` | no, but**re-runs the Phase 9 simulation**, so transitively blocked |

**Phases 4 through 8 are entirely clear.** The table is read by exactly one step, **9.1**, which
names it first in its input block and lists all 13 of its columns.

### 1b. The branch: a downstream phase reads it, so this is reported, not fixed

Per the brief, no fix was attempted and neither generator was touched.

### 1c. Derivable or regenerable? — **neither, and that is the finding**

The brief asked whether the weekly position can be recovered from the transaction ledger by
cumulative sum, "the same identity the generator asserts". It was worth the investigation, and
the answer is no. Four measurements, all on `db/gen_v6/seed_1001`:

**(i) The target table is not merely empty, it is also not weekly.**

|                                                          |                                                 |
| -------------------------------------------------------- | ----------------------------------------------- |
| rows                                                     | 52,080                                          |
| part × plant                                            | 4,340                                           |
| **distinct `week_start` values**                 | **12**, spanning 2017-02-27 to 2026-03-30 |
| rows per part-plant                                      | 12.0                                            |
| `qty_on_hand`, `qty_available`, `safety_stock_qty` | **1 distinct value each — zero**         |

Twelve dates over nine years is not a weekly panel. Even fully populated it could not support aq
weekly roll-forward.

**(ii) The transaction ledger does not reproduce the snapshots.** Cumulative sum of the 12,703,783
signed `inventory_transactions.qty` rows per (part, plant), as-of joined to
`inventory_snapshots.qty_on_hand` at 513,784 matched snapshot rows:

|                         |                               |
| ----------------------- | ----------------------------- |
| exact agreement         | **1,717 rows — 0.33%** |
| within ±1              | 5,062 — 0.99%                |
| mean gap                | +3.94                         |
| **sd of the gap** | **314.2**               |

**(iii) It is not a per-part-plant opening-balance offset either.**

|                                                    |                                                                         |
| -------------------------------------------------- | ----------------------------------------------------------------------- |
| groups whose gap is constant (within-group sd = 0) | **0 of 4,212**                                                    |
| groups with within-group sd ≤ 1                   | **0**                                                             |
| median within-group sd                             | 178.7                                                                   |
| sd across groups of the mean gap                   | 10.3                                                                    |
| mean gap by year, 2016→2026                       | +21.9, +3.2, −0.0, +1.7, +2.4, +11.2, −4.5, −2.8, +4.8, −1.9, +23.7 |

**No drift in any year.** The two series agree in expectation and disagree row by row. That rules
out a missing flow and rules out a constant offset.

**(iv) No recording lag explains it.** Dispersion was minimised over a lag sweep:

| lag (days) | 0               | 1     | 3     | 7     | 14    | 28    | 60    |
| ---------- | --------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| sd of gap  | **314.2** | 315.7 | 318.7 | 323.6 | 330.4 | 346.1 | 369.6 |

**Minimum at lag 0, monotonically worse thereafter.** There is no as-of alignment that reconciles
them. `inventory_snapshots` and the transaction ledger are two independently simulated views of
the same process, not one derived from the other.

### The larger finding: there is no valid stock level anywhere in the dataset

| candidate source                          | populated?                            | usable as a stock level?                                                                  |
| ----------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `inventory_position_weekly.qty_on_hand` | **no** — all zero, 52,080 rows | no                                                                                        |
| `inventory_snapshots.qty_on_hand`       | yes — 5,819 distinct values          | **no — 97.5% of values are negative in v6** (54.9% in v7), range −8,941 to +2,435 |
| cumulative`inventory_transactions.qty`  | yes — 12.7M rows                     | **no** — cumulative range −9,099 to +2,953, predominantly negative                |

**Negative on-hand stock is physically impossible.** All three candidates behave like net-flow
series with no opening balance, not stock positions. The blocker is therefore narrower and deeper
than "one empty table": **the opening stock level I₀ does not exist in the dataset in any form.**

What *is* present and fine:

- `part_plant.safety_stock_qty` — 1,019 distinct values, mean 343.3 (v6). The shortage threshold
  in `Short_w = max(0, safety_stock − I_w)` is available.
- `part_demand_weekly.gross_requirement_p50/p90` — fully populated. The consumption driver is
  available.

So Phase 9.1 has its threshold and its demand, and is missing exactly one thing: **where the
stock starts.**

### The verify step is a gate that cannot fail

Guide step 9.1 says:

> with fill fixed at 1.0 and timing fixed at the promise date, the simulation reproduces
> `inventory_position_weekly.qty_available` exactly. Any drift is a bug in the roll-forward.

Against an all-zero column this check is **vacuous** — it is satisfied by any simulation that
outputs zero, and it cannot detect a roll-forward bug. This is precisely the failure mode this
project has named twice before. **The check must be replaced, not merely re-pointed at a
different table.**

### Cost to unblock

**A derivation script, not a generator re-run — but the script cannot be validated.**

| option                                              | effort                                                                                                                                | verdict                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Roll the ledger forward from a recovered I₀        | ~1 day: the ledger already aggregates to (part, plant, day) in**9 s** and cumsums in seconds                                    | **Produces a plausible panel resting on an unvalidatable assumption.** I₀ would have to be *assumed* — e.g. chosen per part-plant so the series is non-negative — because no source carries it. Nothing in the dataset can confirm the choice, and the verify step that would catch an error is vacuous. |
| Interpolate`inventory_snapshots` to weekly        | ~half a day                                                                                                                           | **Rejected** — the source is 97.5% negative. Interpolating an invalid series yields an invalid series.                                                                                                                                                                                                       |
| Regenerate v6 and v7 with the inventory stage fixed | days, and**invalidates every measured number** in `validation5–7.md`, `phase-2`, `phase-3`, `phase1_2` and this report | Correct but expensive. Out of scope here; both worlds were left untouched.                                                                                                                                                                                                                                          |

**Recommendation, for the Phase 4 owner to decide, not settled here:** Phases 4–8 are unaffected
and can proceed immediately. Phase 9 should not start until the inventory stage is either
regenerated or the roll-forward is explicitly redefined to run on an assumed opening balance with
that assumption documented and its verify step rewritten.

### Effect on the validator verdict

Both worlds return **NOT USABLE**, and it is now fully attributed:

| failing check                                         | cause                                                             | status                                |
| ----------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------- |
| `no all-zero numeric column` + 5 per-table failures | 29 all-zero columns, of which 13 are`inventory_position_weekly` | **open defect — this blocker** |
| `right-censored tail is the right size [2019-2025]` | pre-existing, documented in amendment 01                          | accepted                              |

This is **not** an accepted-and-documented state. The brief's documentation branch applied only if
nothing downstream read the table; Phase 9.1 does.

---

## 2. Phase-3 annotation — done

A block was inserted at the top of [`reports/phase-3.md`](phase-3.md), immediately after the H1
and before any number. Nothing else in that file was altered. It records:

- **the cause**, quoting line 17, `from config import WORLDS, CACHE, SPLIT` — `FIT_WINDOW` is
  never imported and nothing filters `snapshot_date`, so the phase trained on **66 snapshots
  against baselines fitted on 44**;
- that **its absolute figures and `phase1_2.md`'s are not on the same training fold** and are not
  comparable to four decimals;
- that **`phase1_2.md` carries the authoritative numbers** wherever the two overlap;
- that the **direction and magnitude survive** — Phase 2 §2 measured the window at ~0.15% on
  arrival for the GBM, and `phase1_2.md` §10 reproduces the arrival conclusion under the corrected
  protocol;
- that the phase was **deliberately not re-run**, and why.

One further correction was added while there: §3.2 of that report describes `HeteroMP` as "the
learned graph encoder", which is accurate but is **not SHARE**. The annotation says so, and points
at `phase1_2.md` §6 where both are built and compared.

---
