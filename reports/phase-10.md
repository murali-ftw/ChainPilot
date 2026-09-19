# Phase 10 — Delivery schedule, allocation, and two approved changes

## 1. Verdict

| | |
|---|---|
| **COMPLETE** | **Stage 1** — both approved changes are in `shipped.json` (version `phase10-shipped-1`): fill ships B5-flat-22, and the arrival drift gate is gone, with drift still computed and logged. **Stage 4** — the two orphaned changes now have a home, guide §Phase 11. **Stage 0** — every input audited in both worlds. **10.1's and 10.2's verify gates** — five replacement gates for 10.1, each demonstrated failing; 10.2's tooling gate binds on 41 part-plants and is demonstrated failing. |
| **CONDITIONAL** | **10.1's MILP** runs and is verified, but on a **one-week** horizon, with holding and ordering costs **injected as assumptions** (both absent from the schema) and the **stock balance and safety-stock floor dropped**. **10.2's ranking** runs via `ReducedScorer`, which measures expected **unmet demand in period**, not stockout — and in 3 of the 6 part-plants shown the first-to-second margin does **not** survive the seed bands, so it is not a recommendation there. |
| **BLOCKED** | **10.1's stock-balance formulation** and **10.2's real objective** (`SimulationScorer`), both for the same reason: no opening stock level exists in this dataset. Phase 9 stays blocked. The guide's own 10.1 verify gate is **vacuous** and was replaced. |

**The headline for the client is not the optimisers — it is §6.** Both sub-items are constraint-and-data problems, not solver problems. The MILP solves in milliseconds; what is missing is a stock balance, a holding cost, a forward requirement, and four constraint parameters that exist as columns but carry no usable content.

**Three findings that change what can be built:**

1. **The planning horizon in this dataset is one week.** `part_demand_weekly` holds 12 as-of dates per part-plant, ~9.5 months apart, each with exactly **one** requirement week 30 days out. And `production_plan` is entirely retrospective — `target_period` is 1–8 days **before** `recorded_ts` in every row of both worlds, and equals `plan_date` exactly. There is no multi-week forward requirement anywhere, so a multi-period schedule cannot be built from this data at all.
2. **The guide's 10.1 verify gate cannot fail** — the pre-registered prediction was correct, and for two reasons rather than one.
3. **10.2's qualification constraint is inert and its capacity ceiling is absent**, while its tooling constraint is real and binding. The part of 10.2 the guide calls "the product" is roughly half enforceable.

`inventory_position_weekly` was never read: `ml/opt/audit_inputs.py` and both optimiser modules refuse the filename at runtime, and no file under `ml/` references it.

---

## 2. Stage 0 — input feasibility

Counts are v6; v7 is identical in every structural respect (the masters are shared) and differs only in simulated ranges. Full table for both worlds: `ml/artifacts/phase10_input_audit.json`.

| column | step | rows | null | zero | range |
|---|---|---|---|---|---|
| `part_demand_weekly.gross_requirement_p50` | 10.1 | 52,080 | 0.0% | 2.9% | [0, 2102] |
| `part_demand_weekly.gross_requirement_p90` | 10.1 | 52,080 | 0.0% | 2.9% | [0, 2733] |
| `part_plant.safety_stock_qty` | 10.1 | 4,340 | 0.0% | 2.9% | [0, 3023] |
| `part_plant.reorder_point_qty` | 10.1 | 4,340 | 0.0% | 2.9% | [0, 8227] |
| `part_plant.min_order_qty` | 10.1 | 4,340 | 0.0% | 0.0% | [10, 10] **CONSTANT** |
| `part_plant.lot_size` | 10.1 | 4,340 | 0.0% | 0.0% | [25, 25] **CONSTANT** |
| `part_plant.planning_lead_time_days` | 10.1 | 4,340 | 0.0% | 0.0% | [30, 30] **CONSTANT** |
| `supplier_contracts.moq` | 10.1 | 3,000 | 0.0% | 0.0% | [10, 10] **CONSTANT** |
| `supplier_contracts.lot_size` | 10.1 | 3,000 | 0.0% | 0.0% | [10, 10] **CONSTANT** |
| `supplier_contracts.max_volume_cap` | 10.1 | 3,000 | 0.0% | 0.0% | [5000, 5000] **CONSTANT** |
| `supplier_contracts.min_volume_commitment` | 10.1/10.2 | 3,000 | 0.0% | 0.0% | [100, 100] **CONSTANT** |
| `supplier_contracts.penalty_clause_inr` | 10.2 | 3,000 | 0.0% | 0.0% | [1e+04, 1e+04] **CONSTANT** |
| `part_costs.unit_cost_inr` | 10.1/10.2 | 3,980 | 0.0% | 0.0% | [50.33, 2500] |
| `part_costs.freight_cost_inr` | 10.1 | 3,980 | 0.0% | 0.0% | [2.05, 250.8] |
| `logistics_lanes.standard_transit_days` | 10.1 | 16,072 | 0.0% | 0.0% | [6, 27.6] |
| `logistics_lanes.distance_km` | 10.1 | 16,072 | 0.0% | 0.0% | [50, 2199] |
| `calendar.is_working_day` | 10.1 | 26,201 | 0.0% | 28.6% | [0, 1] |
| `calendar.is_shutdown` | 10.1 | 26,201 | 0.0% | 100.0% | [0, 0] **CONSTANT** |
| `supplier_allocation.allocation_pct` | 10.2 | 2,300 | 0.0% | 0.0% | [8.33, 100] |
| `alternate_sources.qualification_status` | 10.2 | 16,072 | 0.0% | — | qualified **CONSTANT** |
| `alternate_sources.qualification_lead_days` | 10.2 | 16,072 | 0.0% | 0.0% | [30, 179] |
| `alternate_sources.ramp_rate_pct_per_month` | 10.2 | 16,072 | 0.0% | 0.0% | [5, 39] |
| `alternate_sources.cost_delta_pct` | 10.2 | 16,072 | 0.0% | 0.0% | [-4, 12] |
| `sourcing_channels.is_approved` | 10.2 | 16,072 | 0.0% | 0.0% | [1, 1] **CONSTANT** |
| `sourcing_channels.approval_status` | 10.2 | 16,072 | 0.0% | — | approved **CONSTANT** |
| `tooling.is_transferable` | 10.2 | 2,500 | 0.0% | 39.0% | [0, 1] |
| `tooling.duplicate_exists` | 10.2 | 2,500 | 0.0% | 70.3% | [0, 1] |
| `tooling.transfer_lead_days` | 10.2 | 2,500 | 0.0% | 0.0% | [20, 199] |
| `revealed_capacity_monthly.revealed_capacity_est` | 10.2 | 51,660 | 100.0% | — |  **CONSTANT** |
| `revealed_capacity_monthly.evidence_strength` | 10.2 | 51,660 | 100.0% | — |  **CONSTANT** |
| `inventory_snapshots.qty_on_hand` | 10.1 | 517,996 | 0.0% | 0.1% | [-8941, 2435] |
| `inventory_transactions.qty` | 10.1 | 12,703,783 | 0.0% | 0.2% | [-1872, 8373] |

### 2.1 What is absent entirely

| needed for | column | status |
|---|---|---|
| 10.1 objective | **holding cost** | **no such column anywhere in the 49-table schema** |
| 10.1 objective | **ordering / setup cost** | **no such column anywhere** |
| 10.1 objective | **truck capacity** | absent, so the guide's `ceil(x_w / truck_cap)` freight term cannot be formed |
| 10.1 constraint | **opening stock balance** | no usable source (§2.3) |
| 10.2 constraint | `revealed_capacity_est`, `evidence_strength` | present but **100% NULL** in both worlds |

### 2.2 The pre-registered prediction (0.3): CONFIRMED, and worse than predicted

The guide's gate is *"a part with no capacity constraint and ZERO holding cost must be ordered in the last feasible week."* **It cannot fail**, for two independent reasons:

- **Zero holding cost is true of every part**, because the column does not exist. The premise is universal.
- **No capacity constraint is also true of every part**: `max_volume_cap` is constant 5,000 while the largest weekly requirement in either world is 2,102. The cap can never bind.

And a third point that makes it worse than merely vacuous: with `c_hold = 0` the objective is **indifferent to timing**. `ml/opt/gates.py` exhibits the degeneracy — ordering everything in the first week costs **50** and ordering everything in the last costs **50**, identical. The solver happened to choose `w0`, so the gate **fails against a correct solver** for reasons that have nothing to do with correctness. It tests HiGHS's tie-breaking.

**This is the 8th "gate that cannot fail"** on the project's record. Replacements in §4.

### 2.3 The opening balance (0.4): 10.1's stock-balance form is BLOCKED

I stopped and reported this before building, as instructed. Every candidate fails:

| candidate | status |
|---|---|
| `inventory_position_weekly` | forbidden, and every numeric column is zero |
| `inventory_snapshots.qty_on_hand` | **97.5% negative in v6**, 54.9% in v7 (recounted: 517,996 / 518,032 rows) |
| `inventory_transactions` cumsum | agrees with the stated balance on **0.33%** of rows (Phase 3 closeout §68) |
| `part_plant.reorder_point_qty` | a policy threshold, not a position — it says when to order, not what is held |

**No MILP in this phase runs on a fabricated balance as its deliverable.** The balance-mode solver exists, takes `i0` as an injected parameter defaulting to 0, and is labelled a sensitivity everywhere it appears.

### 2.4 Verdicts (0.5)

| sub-item | verdict | what is substituted, and what it costs |
|---|---|---|
| **10.1** | **BUILDABLE-WITH-SUBSTITUTION** (the guide's stock-balance form: **BLOCKED**) | Holding and ordering costs **injected** (default 0.0, the data-faithful value) — with them at zero the schedule's **timing is not identified**, only its quantity. Freight priced **per unit** from `freight_cost_inr` instead of per truck. Stock balance and safety-stock floor **dropped**. Horizon is **one week**, not the multi-week ladder the guide assumes. |
| **10.2 constraint layer** | **BUILDABLE-WITH-SUBSTITUTION** | Tooling: real and binding. Ramp rate: real, but its baseline period is **assumed**. Minimum-volume commitment: constant 100 with **no period** — window assumed. Qualification: **INERT**, every row reads "qualified" and every channel `is_approved = 1`, so it can never bind. Capacity ceiling: **BLOCKED**, 100% null. |
| **10.2 objective** | **BLOCKED** for the guide's version; **ReducedScorer** runs in its place | Measures expected unmet demand in period, not stockout against inventory. |

---

## 3. Stage 1 — the two approved changes

`ml/configs/shipped.json`, version **`phase6-shipped-1` → `phase10-shipped-1`**. No model was retrained and no learning rate touched.

| | before | after |
|---|---|---|
| **fill_rate** | `arch: none, depth: 0, lr: 0.000125` — the neural point-mass head | **`model: b5flat22`**, family LightGBM, `identity_keys: false`. The superseded block is kept inline under `_superseded`. |
| **arrival_week** | `fallback: {rule: "if |drift − h0 drift| > threshold …", threshold_pp: 1.0}` | **`fallback: null`**, `serve_distribution: "h0_always"`. The old block is kept under `_superseded_fallback`. |

**The reversal condition is recorded in the config itself**, not only in a report: if 10.2's optimiser becomes the consuming path, re-open the fill switch — an optimiser consumes the whole distribution and therefore pays the proper score, where the neural head wins 16 of 16.

**What changed in code.** `loop.predict` no longer compares a drift statistic to a threshold. The decision is now `serve_source(task, shipped, h0_available)`, a pure function of configuration that **cannot see a drift value**, and that raises if a fallback is ever configured again. Drift and excess are still computed per batch and still land in the output for monitoring.

**The test** (`ml/tests/test_no_drift_switch.py`, 4 tests, passing) sweeps the drift statistic across 14 values spanning every historical band edge — 0.0 to 1e6 — and asserts the served source never moves. Per standing rule 1 it also builds a *gated* implementation and confirms the sweep separates it from the ungated one; without that, the sweep would pass for any implementation and prove nothing.

---

## 4. Stage 2 — guide 10.1, the delivery-schedule MILP

`ml/opt/schedule_lp.py`, `scipy.optimize.milp` (HiGHS). Variables per week: `q_w` integer in lots and `y_w` binary. Every input is loaded **as-of** on its recorded/effective column (`as_of_date`, `effective_from`, `valid_from`), never an event date.

### 4.1 What was dropped and what was substituted (2.2)

| guide term | treatment |
|---|---|
| stock balance `I_w` | **DROPPED** — recorded in the artifact's `dropped_constraints` |
| `I_w ≥ safety_stock` | **DROPPED** — needs the balance |
| `c_hold · I_w` | **INJECTED**, default 0.0 |
| `c_order · y_w` | **INJECTED**, default 0.0 |
| `c_freight · ceil(x_w/truck_cap)` | **SUBSTITUTED**: per-unit freight from `part_costs.freight_cost_inr` |
| `Σ x_w = requirement` | kept, as coverage to within one lot (the requirement is rarely a lot multiple) |
| `x_w ≥ MOQ · y_w`, `x_w ≡ 0 (mod lot)`, weekly cap, feasible weeks | kept, all data-backed |

### 4.2 Solved schedules (2.4)

t0 = 2025-04-27, the first planning moment after an as-of date. Forward horizon: **1 week**, which is all the data carries.

| world | part | requirement | lot | received | order weeks | binding constraints |
|---|---|---|---|---|---|---|
| v6 | P00408 @ PL01 | 291 | 25 | 300 | 2025-05-26 | coverage; lot-size multiple |
| v6 | P00509 @ PL07 | 122 | 25 | 125 | 2025-05-26 | coverage; lot-size multiple |
| v6 | P00192 @ PL04 | 138 | 25 | 150 | 2025-05-26 | coverage; lot-size multiple |
| v7 | P00408 @ PL01 | 254 | 25 | 275 | 2025-05-26 | coverage; lot-size multiple |
| v7 | P00509 @ PL07 | 123 | 25 | 125 | 2025-05-26 | coverage; lot-size multiple |
| v7 | P00192 @ PL04 | 141 | 25 | 150 | 2025-05-26 | coverage; lot-size multiple |

**Solver status: 200 of 200 optimal in each world**, no infeasible and no time-limited cells. On a one-week horizon with MOQ 10 and lot 25 the answer is arithmetic — round the requirement up to a lot multiple — and only two constraints ever bind. **Nothing here should be read as advice about when to order**: with no holding cost and one week, there is no timing decision to make.

The multi-week machinery is exercised separately (`--series observed --mode balance`, 12 past requirement weeks, injected `i0 = 0`, `c_hold = 5`): 10 order weeks, safety-stock floor binding. **That is a mechanism demonstration on past weeks, not a plan**, and the module prints that warning whenever the mode is used.

### 4.3 Verify gates, and the demonstration that each can fail (2.3)

`ml/opt/gates.py`. Every gate is run against the real solver, then against a deliberately broken one.

| gate | real solver | broken by | broken solver | can it fail? |
|---|---|---|---|---|
| **G0 — the guide's own** | **FAILS** (solver chose the first week; both schedules cost 50) | — | — | **NO — VACUOUS** |
| G1 — a positive holding cost defers the order | PASS | holding term removed | FAILS | **yes, shown** |
| G2 — MOQ floor: order nothing or at least the MOQ | PASS | MOQ linking removed | FAILS | **yes, shown** |
| G3 — every receipt is a whole lot multiple | PASS | lot rounding removed | FAILS | **yes, shown** |
| G4 — receipts cover the requirement | PASS | coverage row removed | FAILS | **yes, shown** |
| G5 — no receipt in a shutdown week | PASS | calendar bounds removed | FAILS | **yes, shown** |

**Two gates I had to fix rather than accept**, both recorded as deviations:

- **G1 first failed against the correct solver.** I had used a flat demand profile, but under a stock balance receipts must arrive *before* the demand they cover, so deferral is infeasible by construction. The gate was wrong, not the solver; it now puts the requirement at the end of the horizon.
- **G5 first passed even when broken.** With linear freight every placement ties, so the broken solver avoided the shutdown week by tie-breaking alone. The gate now makes the shutdown week the one the objective strictly prefers.

Both are the same failure mode this project keeps finding: a check that cannot discriminate looks like a passing check.

---

## 5. Stage 3 — guide 10.2, allocation

`ml/opt/allocation.py`.

### 5.1 Candidate enumeration and the honest scope (3.1)

| | v6 | v7 |
|---|---|---|
| part-plants | 4,227 | 4,227 |
| **with more than one qualified supplier** | **3,828 (90.6%)** | **3,828 (90.6%)** |
| with a recorded incumbent split | 824 (19.5%) | 1,088 (25.7%) |

**Allocation is a live question for 90.6% of part-plants** — that part of the feature's scope is real. But **only 19.5% (v6) / 25.7% (v7) have a recorded incumbent split**, so for four part-plants in five there is nothing to compare a recommendation against; the module falls back to "all volume on the first qualified supplier", which is an assumption, not the client's actual sourcing. Five candidates are enumerated per part-plant: incumbent, all-to-cheapest, equal split, shift 20pp to the cheapest alternative, and cap-any-supplier-at-70%.

### 5.2 The constraint layer (3.2)

| constraint | data | status |
|---|---|---|
| **Tooling** (`is_transferable`, `duplicate_exists`) | real binaries: 39% not transferable, 70% no duplicate | **ENFORCED and binding** — blocks 77 (v6) / 88 (v7) candidate moves |
| **Ramp rate** (`ramp_rate_pct_per_month`, 5–39) | real | **ENFORCED**, but the baseline period it applies to is **assumed** (30 d) — client parameter |
| **Minimum-volume commitment** (`min_volume_commitment`) | constant 100, `penalty_clause_inr` constant 10,000 | **DEGENERATE**: identical for every contract, and carries **no period**. Window assumed (90 d) — client parameter |
| **Qualification** (`qualification_status`, `is_approved`) | constant "qualified" / 1 | **INERT — can never bind.** Nothing in this dataset is unqualified, so the PPAP lead time the guide calls the heart of the product is never exercised |
| **Capacity ceiling** (`revealed_capacity_est`) | 100% null | **BLOCKED** — the term is absent from the scorer |

### 5.3 The two scorers (3.3)

**`SimulationScorer`** implements the guide's call signature and its body raises `NotImplementedError("blocked: Phase 9.1, no opening stock balance")`. It is not stubbed with a number.

**`ReducedScorer`** runs: `requirement × Σ share_s × (1 − E[fill_s]) × strain_penalty_s × shortage_cost + purchase cost`, where `E[fill_s]` is the **shipped fill model's** (B5-flat-22, recalibrated) mean P(line fills completely) per supplier, and `strain_penalty` is driven by the capacity head's **P90** per supplier. **This measures expected unmet demand in period, not stockout against inventory** — it has no inventory state, and every number it produces carries that caveat.

**A correction I had to make mid-stage.** The first version keyed the penalty to the capacity **P50**. Utilisation sits near 0.65, so `max(0, P50 − 1)` was zero for 98% of suppliers: the capacity signal never entered the score, every seed gave an identical number, and **the seed-band check was vacuous** — it would have "passed" for any ranking whatsoever. Switching to P90 (which exceeds 1.0 for 16% of suppliers) makes the term real and the bands non-degenerate. An assertion now fails the run if either signal loads empty.

### 5.4 Ranking, with the band check (3.3)

| world | part-plant | requirement | qualified | best candidate | score | margin to 2nd | seed bands | a recommendation? |
|---|---|---|---|---|---|---|---|---|
| v6 | P00001 @ PL01 | 124 | 5 | incumbent | 170,830 | 5,126 | disjoint | yes |
| v6 | P00001 @ PL02 | 322 | 4 | cap any supplier at 70% | 560,541 | 51,173 | disjoint | yes |
| v7 | P00001 @ PL01 | 132 | 5 | incumbent | 191,584 | 1,763 | disjoint | yes |
| v7 | P00001 @ PL02 | 249 | 4 | all to cheapest qualified | 80,066 | 0 | overlap | **no** |
| v7 | P00001 @ PL03 | 82 | 2 | cap any supplier at 70% | 119,094 | 1,348 | overlap | **no** |

**In 3 of these 6 part-plants the first-to-second margin does not survive the 3-seed bands, and those rankings are not recommendations.** Where the margin is zero, several candidates are the same split in different words — with one supplier holding all volume, "all to cheapest" and "incumbent" coincide. The module reports `recommendation_supported: false` for every such row rather than printing an order.

### 5.5 The verify gate (3.4)

The guide's gate: *a part whose tooling is neither transferable nor duplicated must return the incumbent split unchanged.*

| | v6 | v7 |
|---|---|---|
| part-plants with frozen tooling **where the frozen supplier is also qualified** | 41 | 41 |
| candidate moves blocked | **77** | **88** |
| moves leaked through | 0 | 0 |
| **gate against the real constraint layer** | **PASS** | **PASS** |
| gate with the tooling check removed | **FAILS**, 33 leaked | **FAILS**, 38 leaked |
| **can it fail?** | **yes, shown** | **yes, shown** |

**Unlike 10.1's gate, this one is real** — tooling has genuine variation, so parts that cannot move exist. One subtlety: the gate is only exercised where a frozen supplier is *also qualified at that plant*. My first run checked 39 part-plants and blocked nothing, because the frozen suppliers were not in the qualified set and no candidate ever offered them volume — a gate that was passing without being tested. Restricting to the 41 part-plants where the constraint can actually bind is what makes the PASS meaningful.

### 5.6 The inherited limitation (3.5)

**Capacity intervals are not quotable, and `ReducedScorer` inherits that.** Empirical 80% coverage runs **0.72–0.81** and is below nominal in **12 of 16** backtest windows, in every model class. The scorer's capacity term consumes a P90 quantile from exactly those intervals, so any ranking that turns on the capacity term carries the same miscalibration. This is stated in the module docstring, in the artifact's `caveat` field, and here.

---

## 6. CLIENT ASKS

*This is the most useful output of the phase. Each item names what is needed, why, and what it unblocks.*

### 6.1 Blocking — nothing downstream works without these

1. **Opening stock position per part-plant, as of a stated timestamp.** One number per part-plant: units on hand, with the as-of date. **What it unblocks:** the Phase 9 stock roll-forward, the guide's 10.1 stock-balance formulation, the safety-stock floor, and 10.2's real objective (expected shortage rather than expected unmet demand). **Why we ask:** the supplied `inventory_position_weekly` is entirely zero; `inventory_snapshots.qty_on_hand` is 97.5% negative in one world; and reconstructing the balance from `inventory_transactions` reproduces the stated figure on 0.33% of rows. This is the single highest-value item on this list — it is the only one that blocks a whole phase.
2. **A forward requirement plan.** The current extract's forecast is one week, 30 days out, refreshed roughly every 9.5 months; the production plan is recorded 1–8 days *after* the period it describes. **What it unblocks:** any multi-week delivery schedule at all. **What we need:** the MRP/planning horizon as actually run — weekly or monthly buckets, several periods ahead, with the date each version was published.

### 6.2 Cost parameters — the objective is currently assumed

3. **Inventory holding / carrying cost** (per unit per week, or an annual carrying rate). No such column exists. **Consequence today:** the schedule's *timing* is not identified — every schedule with the same number of order weeks costs the same, and the guide's own verify gate is vacuous as a result.
4. **Ordering / setup cost per purchase order.** Absent. Without it there is no economic order quantity, only lot rounding.
5. **Truck capacity and freight tariff structure.** Absent, so freight is priced per unit rather than per truck; step-shaped freight is what makes consolidation worth anything.
6. **Shortage / stockout cost per unit** (or a service-level target to price it against). Currently an assumption of ₹1,000/unit in `ReducedScorer`.

### 6.3 Constraint parameters — columns exist but carry no usable content

7. **Minimum-volume commitment: over what period?** `min_volume_commitment` is constant 100 for all 3,000 contracts with no window. **And how is `penalty_clause_inr` (constant ₹10,000) applied** — per breach, per unit short, or pro-rata?
8. **Ramp-rate baseline.** `ramp_rate_pct_per_month` is real (5–39%), but a percentage *of what*, measured from which month?
9. **Real qualification status.** Every row reads "qualified" and every channel `is_approved = 1`, so the PPAP lead time — which the guide calls the heart of the product — can never bind. We need the genuine mix of qualified, in-qualification and unqualified, with target completion dates.
10. **Supplier capacity ceilings.** `revealed_capacity_est` and `evidence_strength` are 100% null. Declared capacity per supplier-part per period, or enough delivery history to estimate it.
11. **Current sourcing splits for the remaining 80%.** Only 19.5% (v6) / 25.7% (v7) of part-plants have a recorded allocation. Without the incumbent there is no baseline to recommend against.
12. **Per-supplier-part pricing including the alternate's delta.** `cost_delta_pct` exists (−4% to +12%); confirm it is the right basis, and whether price breaks (`price_break_qty`) apply.

### 6.4 What we can build the moment 1 and 2 arrive

The solver, the constraint layer, both verify-gate suites and the candidate enumeration are written and tested. With a real opening balance and a real forward plan, 10.1 becomes a genuine multi-period schedule and 10.2's `SimulationScorer` can replace `ReducedScorer` without touching the interface.

---

## 7. Deviations

Continuing the guide's index from row 36.

| # | the guide / specification says | measured, and what Phase 10 did |
|---|---|---|
| 36 | 10.1's inputs include a holding cost, an ordering cost and a truck capacity | **none of the three exists** in the 49-table schema. The objective carries only per-unit freight; holding and ordering are injected parameters defaulting to 0.0, which leaves the schedule's **timing unidentified** |
| 37 | 10.1's verify gate discriminates | **it cannot fail** — "zero holding cost" and "no capacity constraint" are both true of every part (`max_volume_cap` constant 5,000 vs largest requirement 2,102), and with `c_hold = 0` ordering first and ordering last cost the same. **8th gate that cannot fail.** Replaced by five gates, each demonstrated failing |
| 38 | a rolling plan gives a forward requirement horizon | `part_demand_weekly` carries **one** week per part-plant, 30 days out, at 12 as-of dates ~9.5 months apart; `production_plan` is **entirely retrospective** (`target_period` 1–8 days before `recorded_ts`, equal to `plan_date`, 0.00% forward in both worlds). The deliverable schedule is therefore **single-period** |
| 39 | the stock balance is a 10.1 constraint | **dropped**, with the safety-stock floor. No opening level exists (§2.3). A balance mode exists behind an injected `i0` and is labelled a sensitivity, never a deliverable |
| 40 | 10.2's qualification lead time is a binding constraint | **INERT**: `qualification_status` is constant "qualified" and `is_approved` constant 1, so nothing is ever unqualified. Implemented anyway, against an injectable cutoff, and reported as unenforceable on this data |
| 41 | 10.2 ranks candidates by expected shortage cost | **blocked**: shortage needs inventory. `SimulationScorer` raises; `ReducedScorer` measures expected **unmet demand in period** instead and says so wherever a number is quoted |
| 42 | a seed-band check protects the ranking | the first implementation keyed the capacity penalty to the **P50**, which sits below 1.0 for 98% of suppliers, so the term was inert, the seeds gave identical scores and **the band check could not fail**. Fixed to the P90 (>1.0 for 16% of suppliers); an assertion now fails the run if either signal loads empty |
| 43 | a verify gate that passes is a verify gate that works | 10.2's tooling gate first "passed" while blocking nothing: the frozen suppliers were not qualified at the plants tested, so no candidate ever offered them volume. Restricted to the 41 part-plants where it can bind — it now blocks 77 (v6) / 88 (v7) moves and fails when the check is removed. Same class as 10.1's G5, caught twice in one phase |

---

## 8. Open items

**Carried from Phase 9B §8, unchanged:** fill's marginal calibration versus the proper score (now live, since the switch is made and the reversal condition is recorded); recalibration hurting marginal ECE on v7 o6; fill's 9× ECE spread across windows; Stage B's recency-versus-window ambiguity; arrival origins 3–5 untrained; deviation 20's validation-outcome overlap; 3 seeds against the specification's 5; capacity intervals not quotable; and B.5's product question — **which now has a concrete consequence**, because `ReducedScorer` is a distribution-consuming path and the fill config's reversal condition points at exactly it.

**New:**

1. **The two approved changes are configured but unexercised end to end.** `shipped.json` now names `b5flat22`, but no serving path was re-run against the new config in this phase — no model was retrained, by instruction. A smoke run of `loop.predict` on one bundle per task, asserting the served distribution comes from the configured source, belongs to the next phase.
2. **10.1 is single-period until a forward plan exists** (deviation 38). The multi-period solver is written and verified on constructed inputs; it has never run on a real forward requirement.
3. **10.2's constraint layer is roughly half enforceable** (deviation 40 and §6.3). Qualification is inert, the capacity ceiling is absent, and two parameters are assumed.
4. **`ReducedScorer`'s shortage cost is an assumption** (₹1,000/unit) and its ranking is sensitive to it. No sensitivity sweep was run; it should be, before any allocation output is shown to a client.
5. **Phase 11 is scoped but not started** — 11.1 capacity intervals (~3 h), 11.2 arrival with `promise_week` and line age (~6–7 h, or ~1.2 h for a single-origin pilot). 11.2 remains the highest-expected-value model change in the project.
6. **Phase 9 stays blocked.** `inventory_position_weekly` was never read in Phase 10 — both optimiser modules and the audit refuse the filename at runtime, and no file under `ml/` references it.
