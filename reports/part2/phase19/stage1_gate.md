# Phase 19 Stage 1 — Gate v2: forward load re-tested with a fair control, and order cadence

**Audience:** whoever decides which forward-looking inputs go into the arrival, fill and capacity models.
**Measured on:** v8 seed 1001, fixed split, TEST, 5 seeds (7/17/27/37/47), LightGBM proxy (frozen GBM), Apple M4 Pro CPU.
**Code:** `ml/data/fwd_season.py`, `ml/data/cadence.py`, `ml/baselines/phase19_proxy.py` (fits at `da6fee1`),
`ml/eval/phase19_score.py` (first run `8599179`, five-seed rule `e203a02` → `ml/artifacts/phase19/gate_v2.json`). Pre-registration `4d533b4`.
**Status:** complete. **Passes:** arrival (all three families), fill (fwd_season, cadence), capacity (fwd_load
supplier-specific, under the five-seed band rule; deviation 171).

## Reuse and reproduction checks (before any verdict)

| check | result |
|---|---|
| BASE re-fitted (seed 7) vs the stored LightGBM-flat arm, every task, val and test | **bit-exact** (max \|Δ\| = 0.0) |
| `fwd_load` arrays rebuilt from the unchanged module vs Phase 18's stored arrays, all 61 snapshots | **equal** |
| BASE + fwd_load: Phase 18's stored predictions re-scored vs Phase 18's `scores.json` | **equal** on every metric |
| BASE + fwd_season vs Phase 18's diagnostic `fwd_load_netmean` predictions, every seed and fold | **equal** (max \|Δ\| = 0.0) |
| `cadence` as-of assertion | 44.1 M source rows asserted; fires on a caller that drops the recorded-time bound |

## Arms (5-seed bands [min, mean, max])

### Arrival

| arm | lateness ROC-AUC | A3 median abs err (days) |
|---|---|---|
| neural incumbent (context) | 0.7090 [0.7064, 0.7130] | 13.06 [12.86, 13.30] |
| **BASE** | 0.7053 [0.7048, 0.7057] | 13.25 [13.21, 13.27] |
| + fwd_load (Phase 18) | **0.7186** [0.7183, 0.7191] | 12.96 [12.91, 13.00] |
| + fwd_season | 0.7067 [0.7062, 0.7071] | 13.07 [13.00, 13.11] |
| + fwd_season, snapshot-permuted | 0.7020 [0.6985, 0.7044] | 13.26 [13.20, 13.36] |
| + fwd_load, snapshot-permuted | 0.7039 [0.7026, 0.7057] | 13.24 [13.22, 13.26] |
| + cadence | **0.7142** [0.7141, 0.7144] | 13.08 [13.04, 13.11] |
| + cadence, shuffled across channels | 0.7050 [0.7043, 0.7055] | 13.22 [13.20, 13.25] |

### Fill

| arm | exact CRPS | P(fill = 1) ROC-AUC |
|---|---|---|
| neural incumbent (context) | 0.13876 [0.13866, 0.13882] | 0.6204 [0.6200, 0.6211] |
| **BASE** | 0.1397 [0.1396, 0.1398] | 0.6049 [0.6047, 0.6053] |
| + fwd_load (Phase 18) | **0.1355** [0.1354, 0.1357] | **0.6582** [0.6576, 0.6586] |
| + fwd_season | 0.1369 [0.1368, 0.1369] | 0.6414 [0.6408, 0.6417] |
| + fwd_season, snapshot-permuted | 0.1399 [0.1392, 0.1409] | 0.6037 [0.5914, 0.6131] |
| + fwd_load, snapshot-permuted | **0.1387** [0.1383, 0.1391] | **0.6173** [0.6120, 0.6246] |
| + cadence | 0.1395 [0.1395, 0.1396] | 0.6067 [0.6062, 0.6074] |
| + cadence, shuffled | 0.1398 [0.1397, 0.1400] | 0.6039 [0.6031, 0.6050] |

### Capacity (Phase 15 UC3)

| arm | P @ 1% | P @ 5% | P @ 10% | R @ 0.70 | R @ 0.80 | R @ 0.85 |
|---|---|---|---|---|---|---|
| neural incumbent (context) | 0.904 | 0.851 | 0.810 | 0.515 | 0.318 | 0.192 |
| **BASE** | 0.780 [0.751, 0.804] | 0.692 [0.687, 0.699] | 0.662 [0.660, 0.664] | 0.148 [0.141, 0.156] | 0.022 [0.016, 0.030] | UNREACHABLE (1 of 5) |
| + fwd_load (Phase 18) | 0.877 [0.822, 0.916] | 0.813 [0.802, 0.828] | 0.796 [0.786, 0.804] | 0.443 [0.437, 0.448] | 0.226 [0.211, 0.246] | 0.112 [0.096, 0.118] |
| + fwd_season | 0.717 [0.698, 0.747] | 0.758 [0.742, 0.766] | 0.722 [0.716, 0.730] | 0.302 [0.263, 0.343] | 0.033 [0.022, 0.047] | 0.011 [0.006, 0.015] |
| + fwd_season, snapshot-permuted | 0.679 [0.564, 0.764] | 0.609 [0.508, 0.702] | 0.607 [0.554, 0.664] | UNREACHABLE (2) | UNREACHABLE (3) | UNREACHABLE (4) |
| + fwd_load, snapshot-permuted | 0.783 [0.747, 0.822] | 0.709 [0.691, 0.723] | 0.665 [0.642, 0.689] | 0.143 [0.098, 0.185] | UNREACHABLE (2) | UNREACHABLE (4) |
| + cadence | 0.793 [0.782, 0.800] | 0.683 [0.679, 0.687] | 0.653 [0.649, 0.656] | 0.152 [0.148, 0.157] | 0.029 [0.016, 0.037] | UNREACHABLE (2) |
| + cadence, shuffled | 0.796 [0.773, 0.809] | 0.692 [0.684, 0.696] | 0.662 [0.657, 0.666] | 0.156 [0.144, 0.163] | 0.019 [0.010, 0.026] | UNREACHABLE (2) |

"UNREACHABLE (k)": k of the 5 seeds cannot reach that precision on validation, so no 5-seed band exists and the point
is not compared (Phase 15's rule; deviation 171).

## Gate v2 verdicts, and Phase 18's beside them

| use case | family | test | **Gate v2** | Phase 18 (old gate) |
|---|---|---|---|---|
| arrival | fwd_season | > BASE on lateness AUC and A3; permuted not > BASE | **PASS** | — (diagnostic only) |
| arrival | fwd_load, supplier-specific | > fwd_season on lateness AUC (A3 overlaps); permuted not > BASE | **PASS** | FAIL |
| arrival | cadence | > BASE on lateness AUC and A3; shuffle not > BASE | **PASS** | — |
| fill | fwd_season | > BASE on CRPS and AUC; permuted not > BASE | **PASS** | — |
| fill | fwd_load, supplier-specific | > fwd_season on both, **but the snapshot-permuted fwd_load is also > BASE** on CRPS and AUC | **FAIL** | FAIL |
| fill | cadence | > BASE on CRPS and AUC (small: −0.0002, +0.002); shuffle not > BASE | **PASS** | — |
| capacity | fwd_season | > BASE at 5%, 10%, R 0.70, R 0.80, but **worse at 1%** | **FAIL** | — |
| capacity | fwd_load, supplier-specific | > fwd_season on **6 / 6**; permuted not > BASE on any comparable point | **PASS** (as coded: FAIL, deviation 171) | FAIL |
| capacity | cadence | **worse at 10%** coverage | **FAIL** | — |

### Reading

- **The forward season is real information** for arrival and fill. It beats BASE, and moving each snapshot's season
  to another snapshot destroys the gain. That is the content Phase 18's control could not credit.
- **Supplier-specific forward load is real on arrival and capacity.** On top of the season it lifts arrival lateness
  AUC 0.707 → 0.719 and every capacity point (precision @ 5% 0.758 → 0.813, recall @ 0.70 0.302 → 0.443). Moved to the
  wrong snapshot it gives nothing on any comparable point.
- **On fill, the supplier-specific gain is partly not about time.** The snapshot-permuted `fwd_load`, which keeps each
  channel's own forward requirement but from the wrong date, still beats BASE (CRPS 0.1387 vs 0.1397, AUC 0.617 vs 0.605).
  A channel's typical plan-to-throughput level works on fill as a static descriptor. Gate v2 cannot credit fill's
  supplier-specific gain to the forecast, so it fails.
- **Cadence helps arrival** (lateness AUC 0.705 → 0.714) and barely moves fill. The shuffle destroys it, so the gain
  is the channel's own rhythm. Stage 2 measures how much of the timing ceiling that reaches.

### Deviation 171: the five-seed band

Phase 18's band function (`phase18_score.band`) silently dropped seeds whose recall-at-p was unreachable on
validation, so a one-seed value could be compared as if it were a five-seed band. That contradicts this phase's
pre-registration ("bands are [min, mean, max] over the 5 seeds") and Phase 15's definition (a point is UNREACHABLE unless
every seed reaches the bar). It was found **after** the first gate run, because one verdict hinged on it: capacity
fwd_load's control "beat" BASE at recall @ 0.80 and 0.85 on bands built from **3 and 1** seeds, against a BASE band of
4 seeds at 0.85. The Phase 19 scorer now requires all five seeds, and the as-coded verdict is kept beside it.
**Exactly one verdict changes: capacity fwd_load supplier-specific, FAIL → PASS.** Every other verdict is the same
under both readings. Phase 18's capacity verdicts are unaffected: each also failed on a precision point every seed reaches.
