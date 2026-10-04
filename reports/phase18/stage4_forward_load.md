# Phase 18 Stage 4 — forward committed load (`fwd_load`)

**Audience:** whoever decides whether to build forward-plan inputs into the arrival, fill and capacity models.
**Measured on:** v8 seed 1001, fixed split, TEST, 5 seeds (7/17/27/37/47), LightGBM proxy (frozen GBM), Apple M4 Pro CPU.
**Code:** `ml/data/fwd_load.py`, `ml/baselines/phase18_proxy.py` (fits at `94cfdc5` / `9ce7007`, whose `ml/` trees are
identical, and `67d9111` for the diagnostic arm), `ml/eval/phase18_score.py` (`67d9111`) → `ml/artifacts/phase18/scores.json`.
**Status:** complete. **Gate verdict: FAIL on arrival, fill and capacity.** The shuffle control does not erase the gain.
**No neural run is made** (pre-registered rule).

## (a) Audit: does any current model input already carry plan-derived forward requirements?

**No.** File and line:
- The channel panel (`ml/data/cache.py:34`, `CANDIDATE_COLS`) holds only `channel_performance_weekly` columns: ordered /
  received quantity, fill, lead time, OTD, ack gap, load ratio (ordered / declared capacity, trailing), revisions.
  Every column looks backward.
- Node features (`ml/data/loader.py:150`, `load_nodes`) read only the master tables (sourcing_channels, suppliers,
  parts, plants).
- LightGBM-flat (`ml/baselines/phase7_fit.py:130–133`, `FLAT` / `STATIC`) uses graph counts and static channel terms.
- `part_demand_weekly` and `production_plan` are read only by the optimiser (`ml/opt/schedule_lp.py:91`,
  `phase12_a4_price.py`, `sensitivity.py`), never by a forecasting model.

## (b) The features

`ml/data/fwd_load.py`, per channel and per supplier, at each of 61 snapshots (2019-01-14 … 2025-12-08):

| column | definition |
|---|---|
| `fwd_ch_req_w{1_4,5_8,9_13}` | part-plant gross requirement P50 summed over the window, from the latest `part_demand_weekly` version recorded by t0, × the channel's trailing-52-week share of the part-plant's ordered quantity |
| `fwd_ch_ratio_w*` | the same ÷ (channel trailing-52-week delivered per week × weeks in the window) |
| `fwd_ch_open_ratio` | open PO quantity at t0 ÷ channel delivered per week |
| `fwd_sup_*` | the same three groups summed over the supplier's channels |
| `fwd_pp_cov_w9_13` | share of the far window's weeks that a version recorded by t0 covers |

- **As-of:** every source row is asserted `recorded_ts ≤ t0` (**22,536,610 rows asserted** over the build). The
  falsification mode fires on all three sources when the recorded-time filter is dropped.
- **`part_demand_weekly` ships without `recorded_ts`.** The generator draws it as `as_of_date` + 0–2 days, so the
  asserted recorded time is the upper bound `as_of_date` + 2 days (deviation 162).
- **Source:** `part_demand_weekly` (already part × plant × week), not `production_plan` BOM-exploded. In v8 the
  production plan drives only its own tables, not channel demand (Stage 1 (d); deviation 160).
- Median supplier ratio over weeks 1–4: **1.05** (IQR 0.90–1.22). Requirement and trailing throughput are on the same scale.
- Never used: realised future ordered volume. That is the hindsight control (Stage 2 file, PRIVILEGED).

## (c)–(e) Proxy gate

BASE = the stored LightGBM-flat arms. The proxy runner **reproduced them bit-exactly** for seed 7 on every task (max
|Δ| = 0.0, val and test), so the stored 5-seed files are the BASE band. SHUF = `fwd_load` shuffled across suppliers
within each snapshot (supplier columns by a supplier permutation, channel columns by a channel permutation).

### Arrival

| arm (5 seeds) | lateness ROC-AUC | A3 median abs err (days) | C-index |
|---|---|---|---|
| neural incumbent (context) | 0.7090 [0.7064, 0.7130] | 13.06 [12.86, 13.30] | 0.6744 |
| **BASE** LightGBM-flat | 0.7053 [0.7048, 0.7057] | 13.25 [13.21, 13.27] | 0.6489 |
| **+ fwd_load** | **0.7186** [0.7183, 0.7191] | **12.96** [12.91, 13.00] | 0.6589 |
| + fwd_load, SHUFFLED | 0.7076 [0.7068, 0.7081] | 13.13 [13.09, 13.15] | 0.6506 |
| *diagnostic: + network mean only* | 0.7067 [0.7062, 0.7071] | 13.07 [13.00, 13.11] | 0.6449 |
| *PRIVILEGED hindsight load (upper bound)* | 0.7792 | 11.32 | 0.6893 |

### Fill

| arm | exact CRPS | P(fill = 1) ROC-AUC | ECE-22 |
|---|---|---|---|
| neural incumbent (context) | 0.13876 [0.13866, 0.13882] | 0.6204 [0.6200, 0.6211] | 0.124 |
| **BASE** | 0.1397 [0.1396, 0.1398] | 0.6049 [0.6047, 0.6053] | 0.052 |
| **+ fwd_load** | **0.1355** [0.1354, 0.1357] | **0.6582** [0.6576, 0.6586] | 0.033 |
| + fwd_load, SHUFFLED | 0.1384 [0.1382, 0.1385] | 0.6261 [0.6232, 0.6290] | 0.050 |
| *diagnostic: + network mean only* | 0.1369 [0.1368, 0.1369] | 0.6414 [0.6408, 0.6417] | 0.016 |
| *PRIVILEGED hindsight load* | 0.1326 | 0.6885 | 0.015 |

### Capacity (Phase 15 UC3)

| arm | precision @ 1% | @ 5% | @ 10% | recall @ p 0.70 | @ 0.80 | @ 0.85 |
|---|---|---|---|---|---|---|
| neural incumbent (context) | 0.904 | 0.851 | 0.810 | 0.515 | 0.318 | 0.192 |
| **BASE** | 0.780 [0.751, 0.804] | 0.692 [0.687, 0.699] | 0.662 [0.660, 0.664] | 0.148 [0.141, 0.156] | 0.022 [0.016, 0.030] | 0.013 [0.011, 0.016] |
| **+ fwd_load** | **0.877** [0.822, 0.916] | **0.813** [0.802, 0.828] | **0.796** [0.786, 0.804] | **0.443** [0.437, 0.448] | **0.226** [0.211, 0.246] | **0.112** [0.096, 0.118] |
| + fwd_load, SHUFFLED | 0.798 [0.738, 0.831] | 0.732 [0.708, 0.751] | 0.691 [0.673, 0.706] | 0.258 [0.249, 0.270] | 0.103 [0.073, 0.123] | 0.061 [0.046, 0.077] |
| *diagnostic: + network mean only* | 0.717 [0.698, 0.747] | 0.758 [0.742, 0.766] | 0.722 [0.716, 0.730] | 0.302 [0.263, 0.343] | 0.033 [0.022, 0.047] | 0.011 [0.006, 0.015] |
| *PRIVILEGED hindsight load* | 0.967 | 0.891 | 0.832 | 0.674 | 0.462 | 0.347 |

### Verdict per use case (pre-registered rule)

| use case | fwd_load vs BASE | SHUFFLED vs BASE | **verdict** |
|---|---|---|---|
| arrival | disjointly better on lateness AUC and A3 | **also disjointly better** on both | **FAIL**: the shuffle does not erase the gain |
| fill | disjointly better on CRPS and AUC | **also disjointly better** on both | **FAIL** |
| capacity | disjointly better on **6 / 6** points | **also disjointly better** on 5 / 6 | **FAIL** |

## What the failure means (diagnostic, no verdict; deviation 163)

The within-snapshot shuffle breaks the link between a supplier and **its own** plan. It cannot break the snapshot's
**network-wide** forward requirement: every snapshot keeps the same set of values, just reassigned. That network total is
the forward season (seasonality and trend ahead), which no backward-looking input carries. The diagnostic arm that
sees **only** the snapshot mean of each column reproduces the shuffled arm's gain: arrival 0.7067 vs 0.7076
(undetermined); fill better than the shuffle; capacity mixed.

So the gain splits into two parts:
1. **Network-wide forward requirement.** Real information from the plan, not noise. The pre-registered control cannot
   tell it apart from a leak, so it is not credited here.
2. **Supplier-specific forward requirement.** Full `fwd_load` beats **both** its shuffle and the network-mean arm,
   disjointly: arrival lateness AUC 0.7186 vs 0.7076 / 0.7067; fill CRPS 0.1355 vs 0.1384 / 0.1369; capacity precision @ 5%
   0.813 vs 0.732 / 0.758, and every capacity point above the network mean.

On fill and arrival the proxy with `fwd_load` also beats the **neural incumbent** disjointly (fill CRPS 0.1355 vs 0.1388,
AUC 0.658 vs 0.620; arrival lateness 0.719 vs 0.709). On capacity it stays below the incumbent (precision @ 5% 0.813 vs
0.851; recall @ 0.70 0.443 vs 0.515). The as-of plan recovers 56–64% of the hindsight-load gain on fill and capacity,
18% on arrival.

**The rule is applied as written: FAIL, no neural run.** The evidence above says the supplier-specific part is real.
A re-test needs a new pre-registration whose control is the network-mean arm (or a shuffle that also permutes
snapshots), not a re-reading of this one.
