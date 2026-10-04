# Phase 20 Stage 4 — what is a reorder signal worth for arrival?

**Audience:** whoever decides whether to ask the client for stock balances and reorder points, and how to justify it.
**Measured on:** v8 seed 1001, fixed split, TEST, 5 seeds, LightGBM proxy (CPU). The privileged probe's details are in
`PRIVILEGED__reorder_probe.md`.
**Code:** audit read-only; `ml/data/stock_asof.py` + proxy arms `stock` / `stock_sperm` (`50cd5bb`); probe
`reports/part2/phase20/PRIVILEGED__reorder_probe.py` (`ab9c919`); scorers `ml/eval/phase20_score.py` (`c303519`) and
`PRIVILEGED__score.py` (`a98f2b9`).
**Status:** complete. **A legitimate observed source exists and passes the gate on arrival, but it is worth +0.0016
lateness AUC** (P8 wrong as worded). **The exact simulator position is worth +0.034, i.e. 17% of the timing headroom**
(P7 wrong).

## (a) Audit: every v8 table or column carrying stock, reorder point, safety stock or in-transit quantity

| table · column(s) | grain | `recorded_ts` | observed ledger or simulator-internal | legitimate as-of arm without `inventory_position_weekly`? |
|---|---|---|---|---|
| `inventory_transactions` (adjustment, receipt, issue_to_production, scrap, transfer_in / out) | part × plant × event | **yes** (median lag 1 day, max 400) | **observed ledger** | **YES**: on hand at t0 = the sum of every row recorded by t0 (built: `stock_asof`) |
| `inventory_snapshots` (`qty_on_hand`; `qty_in_transit`, `blocked`, `reserved` are all 0) | part × plant × month end | yes (month end + 1–5 days) | observed, computed from the ledger | yes, but monthly and redundant with the ledger |
| `inventory_position_weekly` (on hand, in transit, open PO, safety stock, days of supply) | part × plant × week | yes | observed (measured roll-forward) | **excluded by standing rule**; never read in Phase 20 |
| `part_plant.safety_stock_qty` | part × plant | no (static master, effective 2016-01-01) | static: persistent demand level × safety cover | yes, as a static master value |
| `part_plant.reorder_point_qty` | part × plant | no | **the reorder point after the simulation's LAST week** (`generator_v8.py` l. 1675) | **NO**: 2026 information at any t0 (deviation 179) |
| `part_plant.planning_lead_time_days` | part × plant | no | static mean contracted lead | yes |
| `po_lines` / `grn_lines` (open PO quantity) | PO line | yes | observed | yes (used by `fwd_load` and `stock_asof`) |
| `asn` (`dispatched_qty`, `dispatch_ts`, `expected_arrival_date`): in transit | PO line | yes | observed | yes (not built) |
| channel on hand, on order, reorder point, planner forecast | **channel** × week | — | **simulator-internal** (not in `_sim.npz`; captured by the hook) | **PRIVILEGED only** (the probe) |

**"No valid opening stock level exists" (Phase 11 notes): refuted for v8, true for v6 / v7.** v8 posts the opening position to
the ledger as **14,826 `adjustment` rows (1,437,236 units) dated 2015-12-28**, recorded by 2015-12-30. The generator
checks that the ledger roll-forward equals the simulator's balance for every part-plant ("§2.1 ledger roll-forward ==
simulator balance for every part-plant: True (0 mismatches of 4340)", printed by this phase's seed-1002 generation and by Phase 18's seed-1001 re-run). Phase 11A
(`phase-11a.md` §6) found the opening balance missing on v6, where the check passed vacuously.

**But the observed ledger is coarser than the trigger.** The generator raises an order per **channel**
(`generator_v8.py` l. 525–526: inventory position < reorder point, at the channel's review week). The ledger is per
**part × plant**, pooling every channel feeding it, and no as-of channel reorder point is published.

## (b) PRIVILEGED probe: the exact channel position against the exact reorder point

| arm | lateness ROC-AUC | A3 median abs err (days) |
|---|---|---|
| BASE | 0.7053 [0.7048, 0.7057] | 13.25 |
| **PRIVILEGED BASE + true position vs reorder point** | **0.7394** [0.7390, 0.7397] | **12.71** [12.67, 12.75] |
| PRIVILEGED BASE + true creation week (Phase 19) | 0.9104 [0.9102, 0.9105] | 6.42 |

**Probe recovery share = (0.7394 − 0.7053) / (0.9104 − 0.7053) = 16.6%** (P7 needed ≥ 50% and ≥ 0.81: **wrong**).
Capture accepted: the hooked re-run's `_sim.npz` equals the stored world's on all 48 arrays. **Caveat:** the simulator's
position is exact and instantly current, so this is an **upper bound** on what an ERP's lagged book position is worth.

## (c) The legitimate observed arm (`stock_asof`)

Part-plant on hand from the ledger (rows recorded by t0), open PO quantity, a trailing-13-week issue rate, and an
**estimated** reorder point (issue rate × planning lead + safety stock). Joined to each row through its part × plant.
545 M source rows asserted `recorded_ts ≤ t0`. The falsification (bucketing the ledger on event time) fires.

| use case | BASE | **+ stock** | + stock, snapshot-permuted | **verdict** |
|---|---|---|---|---|
| arrival · lateness AUC | 0.7053 [0.7048, 0.7057] | **0.7069** [0.7065, 0.7072] | 0.7051 | **PASS** (+0.0016; A3 undetermined) |
| fill · CRPS / P(full) AUC | 0.1397 / 0.6049 | 0.1394 / 0.6116 | 0.1393 / 0.6136 (**also better**) | **FAIL**: the control gains too, a static descriptor |
| capacity · six points | | | | **FAIL** (none better) |

The legitimate arm reaches **0.8%** of the timing headroom, about **5%** of what the exact probe reaches.

## (d) The client ask, in numbers

> **What would an opening stock balance and reorder points be worth for the arrival forecast?** In the v8 simulation, knowing
> each sourcing channel's exact inventory position against its reorder point at the forecast date lifts lateness ranking
> (ROC-AUC) from **0.705 to 0.739** and cuts the median error of the predicted delivery date from **13.2 to 12.7 days**.
> That is an upper bound: the simulator's position is exact and current, and a real ERP's is not. The book stock v8 already
> publishes (a part-plant ledger, with no as-of reorder point) is worth **+0.002** of that. So the value is in **channel-level**
> position and **as-of reorder points**, not in another stock table at part-plant grain. Even perfect stock data reaches
> only **17%** of the headroom that knowing the order date would give (0.910), because when the next order is raised also
> depends on demand that has not happened yet.

## Reading

- **P8 is wrong as worded:** a legitimate observed source exists (the v8 ledger) and passes its gate on arrival. In
  substance P8 is close: its value is +0.0016 AUC, so the useful signal would still have to come from the client.
- **P7 is wrong:** even the exact trigger state recovers 17%, not ≥ 50%. Arrival's timing headroom is mostly the
  **future** demand path between t0 and the order, not the stock position at t0.
