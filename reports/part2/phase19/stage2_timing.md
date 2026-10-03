# Phase 19 Stage 2 — is arrival's headroom the order-raising time?

**Audience:** whoever decides whether to build an order-timing forecast for the arrival product.
**Measured on:** v8 seed 1001, fixed split, TEST, 5 seeds, LightGBM proxy (frozen GBM), Apple M4 Pro CPU.
**Code:** `reports/part2/phase19/PRIVILEGED__timing.py` (fits at `da6fee1`), `PRIVILEGED__score_timing.py`
(`f5b2f46` → `PRIVILEGED__timing_scores.json`); cadence from `ml/baselines/phase19_proxy.py`.
**Status:** complete. **Verdict: NOT worth building from as-of cadence.** Timing alone carries the ceiling (+0.205
lateness AUC), but the channel's as-of cadence recovers only 4% of it.

## The arms

The PRIVILEGED arm is BASE + **one** column: the line's true creation week minus t0, from `_sim.npz`. **No state**:
no supplier state, transit, regime, K or load. It is never a feature. Before the fit, every arrival label was re-derived
from the simulation (arrival week − t0) and asserted equal, which proves the line-to-simulation join.

| arm | lateness ROC-AUC | A3 median abs err (days) | C-index |
|---|---|---|---|
| BASE (LightGBM-flat) | 0.7053 [0.7048, 0.7057] | 13.25 [13.21, 13.27] | 0.649 |
| BASE + cadence (as-of, legitimate) | 0.7142 [0.7141, 0.7144] | 13.08 [13.04, 13.11] | 0.653 |
| **PRIVILEGED BASE + true creation week, no state** | **0.9104** [0.9102, 0.9105] | **6.42** [6.41, 6.43] | 0.880 |
| *Phase 18 PRIVILEGED: creation week + state at creation* | 0.9229 | 5.63 | 0.901 |
| *Phase 18 PRIVILEGED: state at creation, no creation week* | 0.7092 | 13.24 | 0.675 |

Fill, for context (the timing arm hardly matters there): BASE CRPS 0.1397 / AUC 0.6049; + creation week 0.1396 /
0.6061; + cadence 0.1395 / 0.6067.

## The rule (pre-registered)

| quantity | value | threshold | met |
|---|---|---|---|
| creation week adds over BASE (lateness AUC) | **+0.205** | ≥ 0.05 | yes |
| cadence recovers of that addition | (0.7142 − 0.7053) / 0.205 = **4.3%** | ≥ 30% | **no** |

**Verdict: forecasting the order-raising time is not worth building with as-of cadence.**

## What the numbers settle

1. **Phase 18's claim holds: the arrival headroom is order timing.** Knowing only *when* the line will be raised, with
   no state at all, takes lateness AUC from 0.705 to 0.910. That is **94%** of the way to the timing + state oracle
   (0.923). State without timing adds 0.004. P4 ("timing alone adds < 0.02") is wrong by a factor of ten.
2. **But the channel's visible rhythm does not predict that time.** Days since the last line, the median and IQR of the
   gaps between lines, and the open-line count and age move lateness AUC by +0.009, real and disjoint (the cross-channel
   shuffle erases it) but 4% of the gap. In the generator the next line is raised when inventory position falls below
   the reorder point (`generator_v8.py` l. 525–526), driven by weekly demand draws and stock levels. The PO history
   is a coarse shadow of that, and `inventory_position_weekly` is never a feature.
3. **The brief's fallback sentence does not apply** (deviation 172). It reads "arrival headroom is future supplier
   state, not reachable from as-of data". The measurement says the opposite about *what* the headroom is: it is
   timing, not supplier state. What is true is that as-of cadence cannot reach it. The rule's outcome (not worth
   building) is recorded unchanged; only the sentence is replaced with the measured one.

**For the product:** the arrival label is counted from the forecast date, so most of what it "predicts" is the wait
until the planner raises the order. That is a property of the target, not a model gap. A product that forecasts
lateness for lines **already raised** (counted from creation) would face the supplier and transit part only, where the
current model is already at the state ceiling. Re-scoping the target is a product decision; it is recorded here, not
acted on.
