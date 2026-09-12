# Phase 0–3 closeout — inventory decision, Phase-3 annotation, per-task convergence

**Verdict in three lines.** The inventory table is a **hard blocker on Phase 9**, not a
documentation problem, and it is worse than one empty table — there is **no valid stock level
anywhere in the dataset**. `reports/phase-3.md` is annotated. Fill and shortage were retuned per
task and re-run to convergence.

|               |                                                                                                          |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| Device        | **MPS**, Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32, `num_workers=0`              |
| torch         | 2.14.0                                                                                                   |
| Wall-clock | **23.1 h wall / ~4.8 h compute** — 1,256 epochs at a measured 13.9 s/epoch. Three cells ran while the machine was asleep and record 115, 333 and 687 s/epoch; the wall total is not compute and is not quoted as such. |
| Peak RSS | **3.80 GB** — MPS throughout, no CPU fallback, 16,072 channels |
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

## 3. Per-task learning rates, and the re-runs to convergence

### 3.1 The two LR sweeps

Same five-point protocol arrival got, on **v6 / SHARE / h⁴**, selected on the **validation** fold,
under the final regime (cap 120, patience 8). **Arrival was not retuned and not re-run.**

**fill_rate — chosen 5e-4**

| lr | validation CRPS ↓ | test-v6 | best epoch | epochs | stop |
|---|---|---|---|---|---|
| 2e-3 | 0.0503 | 0.0593 | 12 | 21 | patience |
| 1e-3 | 0.0507 | 0.0597 | 6 | 15 | patience |
| **5e-4** | **0.0500** | 0.0591 | 23 | 32 | patience |
| 2.5e-4 | 0.0501 | 0.0594 | 22 | 31 | patience |
| 1.25e-4 | 0.0501 | 0.0594 | 26 | 35 | patience |

Optimum **interior**, no extension needed. Selection was on validation: the rate not chosen,
2.5e-4, tests at 0.0594 against the chosen rate's 0.0591, so discipline cost nothing here — but it
was applied before the test scores were looked at.

**shortage_qty — chosen 1.25e-4**

| lr | validation PR-AUC ↑ | test-v6 | best epoch | epochs | stop |
|---|---|---|---|---|---|
| 2e-3 | 0.5764 | 0.6536 | 34 | 43 | patience |
| 1e-3 | 0.5715 | 0.6485 | 18 | 27 | patience |
| 5e-4 | 0.6114 | 0.6860 | 41 | 50 | patience |
| 2.5e-4 | 0.6084 | 0.6831 | 45 | 54 | patience |
| **1.25e-4** | **0.6151** | 0.6909 | **84** | 93 | patience |

**Declared protocol deviation.** The optimum landed on the **bottom boundary** and the protocol
says extend downward. **It was not extended, on cost** — each additional point costs ~35 min and
the run was already far over budget. Consequence, stated plainly: **the shortage learning rate may
not be optimal, and a lower rate could improve every shortage number below.** This is a declared
deviation, not an oversight.

The sweep also vindicates the item on its own: at the chosen rate the tuning cell found its best
checkpoint at **epoch 84**, which a 40-epoch cap makes structurally impossible. Every shortage
number in `phase1_2.md` was produced under that ceiling.

### 3.2 Convergence, after

| task | converged in `phase1_2` (cap 40) | **converged now** (cap 120) |
|---|---|---|
| **shortage** | **2 of 10** | **8 of 10** |
| **fill** | 4 of 10 | **6 of 6 run** (4 cells cut, see below) |

**The 14 capped cells are now 2.** Both survivors are shortage h⁰, at epoch 119 of 120 — still
improving, still **floors**, and labelled as such wherever they appear.

| world | task | arch | depth | epochs | best | stop |
|---|---|---|---|---|---|---|
| v6 | shortage | none | h⁰ | 120 | **119** | **CAP — floor** |
| v7 | shortage | none | h⁰ | 120 | **119** | **CAP — floor** |
| v6 | shortage | mp | h¹ | 95 | 86 | patience |
| v6 | shortage | mp | h⁴ | 115 | 106 | patience |
| v6 | shortage | share | h¹ | 68 | 59 | patience |
| v6 | shortage | share | h⁴ | 63 | 54 | patience |
| v7 | shortage | mp | h¹ | 93 | 84 | patience |
| v7 | shortage | mp | h⁴ | 93 | 84 | patience |
| v7 | shortage | share | h¹ | 64 | 55 | patience |
| v7 | shortage | share | h⁴ | 40 | 31 | patience |
| v6 | fill | none | h⁰ | 53 | 44 | patience |
| v6 | fill | mp | h⁴ | 30 | 21 | patience |
| v6 | fill | share | h⁴ | 28 | 19 | patience |
| v7 | fill | none | h⁰ | 26 | 17 | patience |
| v7 | fill | mp | h⁴ | 24 | 15 | patience |
| v7 | fill | share | h⁴ | 27 | 18 | patience |

**Cells cut, and why.** Four fill h¹ cells (v6+v7 × HeteroMP-h¹, SHARE-h¹) were dropped when the
projection reached 9.7 h against a ~6 h budget. That is the sanctioned first cut from the brief.
They were **subsequently re-run in Phase 4**, because step 4.2's depth derivation needs h⁰/h¹/h⁴
per task; see [`reports/phase-4.md`](phase-4.md).

### 3.3 Per-metric seed noise bands — the most consequential measurement here

`phase1_2.md` measured a 0.0028 spread on **arrival C-index only** and correctly refused to
transfer it. Measured now, one cell per task at three torch seeds:

| task | cell | metric | seeds 7 / 17 / 27 | **spread** | sd |
|---|---|---|---|---|---|
| fill | v6 SHARE h⁴ | CRPS | 0.0588 / 0.0595 / 0.0597 | **0.0009** | 0.0005 |
| **shortage** | v6 SHARE h⁴ | PR-AUC | 0.6831 / 0.7040 / 0.6789 | **0.0251** | 0.0134 |
| *(arrival, from phase1_2)* | v6 SHARE h⁴ | C-index | — | *0.0028* | *0.0014* |

**Shortage PR-AUC swings by 0.025 on seed alone — nine times arrival's band.** Every
SHARE-vs-HeteroMP margin ever quoted on that task is smaller than this. Reading shortage margins
against arrival's band, which is what one would do without this measurement, overstates their
significance by roughly an order of magnitude.

### 3.4 Restated scores

Closeout (cap 120) beside `phase1_2`'s capped figures. 95% bootstrap intervals, 1,000 resamples.
Baselines are unchanged and identical in both rows.

**shortage — PR-AUC ↑**

| world | run | naive | LightGBM | h⁰ | MP h¹ | MP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|---|
| v6 | phase1_2 | 0.4896 | 0.5287 | 0.5722 | 0.6574 | 0.6651 | 0.6678 | **0.6968** |
| **v6** | **closeout** | 0.4896 | 0.5287 | 0.5840 [0.5655, 0.6037] ⚠floor | 0.6616 [0.6440, 0.6794] | **0.6899 [0.6739, 0.7076]** | 0.6600 [0.6425, 0.6780] | 0.6831 [0.6653, 0.7005] |
| v7 | phase1_2 | 0.2487 | 0.4438 | 0.4580 | 0.5001 | 0.5069 | **0.5148** | 0.4859 |
| **v7** | **closeout** | 0.2487 | 0.4438 | 0.4691 [0.4456, 0.4940] ⚠floor | 0.5071 [0.4845, 0.5317] | **0.5143 [0.4916, 0.5384]** | 0.4906 [0.4680, 0.5169] | 0.4918 [0.4682, 0.5179] |

**fill — CRPS ↓**

| world | run | naive | LightGBM | h⁰ | MP h⁴ | SHARE h⁴ |
|---|---|---|---|---|---|---|
| v6 | phase1_2 | 0.0632 | 0.0599 | 0.0594 | 0.0603 | 0.0594 |
| **v6** | **closeout** | 0.0632 | 0.0599 [0.0580, 0.0617] | 0.0595 [0.0576, 0.0614] | 0.0598 [0.0579, 0.0618] | **0.0588 [0.0570, 0.0607]** |
| v7 | phase1_2 | 0.1078 | 0.0988 | 0.0990 | 0.0992 | 0.1005 |
| **v7** | **closeout** | 0.1078 | **0.0988 [0.0969, 0.1008]** | 0.0995 [0.0974, 0.1017] | 0.1030 [0.1007, 0.1054] | 0.0992 [0.0972, 0.1015] |

**fill — calibration ECE ↓** (secondary; the neural heads remain far worse calibrated than the GBM)

| world | LightGBM | h⁰ | MP h⁴ | SHARE h⁴ |
|---|---|---|---|---|
| v6 | **0.0107** | 0.0529 | 0.0696 | 0.0361 [0.0314, 0.0414] |
| v7 | **0.0141** | 0.0889 | 0.1745 | 0.0942 [0.0866, 0.1019] |

### 3.5 What convergence changed — three reversals

**(i) SHARE's shortage advantage evaporates.** This is the headline.

| cell | phase1_2 margin | **closeout margin** | vs the 0.0251 band |
|---|---|---|---|
| v6 h¹ | +0.0103 SHARE | **−0.0016** | inside — no difference |
| v6 h⁴ | **+0.0318 SHARE** | **−0.0068** | inside — no difference |
| v7 h¹ | +0.0146 SHARE | **−0.0165** | inside — no difference |
| v7 h⁴ | −0.0211 | **−0.0225** | inside — no difference |

`phase1_2.md` reported SHARE winning three of four shortage cells, with v6 h⁴ its single widest
separation anywhere. **All four margins now sit inside the seed band, and three of the four have
changed sign.** The original result was noise on unconverged cells read against a band borrowed
from a different metric. **On shortage there is no measurable difference between a 794k-parameter
SHARE and an 84k-parameter HeteroMP.**

**(ii) The graph contributes on fill after all.** `phase1_2` measured SHARE's fill graph share at
−0.3% (v6) and −20.0% (v7) and concluded the graph does nothing on fill. Converged: **+16.4% and
+3.2%**, and **SHARE h⁴ on v6 now beats LightGBM outright** (0.0588 against 0.0599), which no
configuration managed before.

**(iii) HeteroMP got worse on fill v7 while SHARE got better** — 0.1030 against 0.0992, a 0.0038
gap against a 0.0009 band. The one place an architecture difference is now resolvable on fill, it
favours SHARE.

### 3.6 Graph contribution, restated

| task | world | h⁰ | MP h⁴ | SHARE h⁴ | MP share | SHARE share | phase1_2 SHARE share |
|---|---|---|---|---|---|---|---|
| fill | v6 | 0.0595 | 0.0598 | 0.0588 | −7.8% | **+16.4%** | −0.3% |
| fill | v7 | 0.0995 | 0.1030 | 0.0992 | −72.0% | **+3.2%** | −20.0% |
| shortage | v6 | 0.5840 ⚠ | 0.6899 | 0.6831 | +21.5% | **+20.4%** | +24.9% |
| shortage | v7 | 0.4691 ⚠ | 0.5143 | 0.4918 | +11.6% | **+6.2%** | +7.7% |

⚠ the shortage h⁰ reference is itself a floor, so both shortage shares are upper bounds on the
graph's contribution: a fully converged h⁰ would raise the floor and shrink the share.

---

## Closeout status

| # | item | status |
|---|---|---|
| 1 | inventory table | **BLOCKER on Phase 9, reported not fixed.** Phases 4–8 clear. No valid stock level exists in the dataset in any form; derivation is unvalidatable. Neither generator touched. |
| 2 | `phase-3.md` annotation | **closed** |
| 3 | per-task LR + re-runs to convergence | **closed**, with two declared deviations: the shortage sweep was not extended past its boundary optimum, and 4 fill h¹ cells were cut here and re-run in Phase 4 |

**Verdict: Phase 4 may start.** The gate condition — fill and shortage having converged numbers —
holds for 18 of 20 cells; the two exceptions are shortage h⁰ and are labelled floors wherever used.
