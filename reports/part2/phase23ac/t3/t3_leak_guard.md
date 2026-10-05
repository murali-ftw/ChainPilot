# Phase 23AC T3 — the standing leak guard

**Audience:** anyone adding a feature to ChainPilot, and whoever wires the guard into CI.
**Measured on:** v8 seed 1001 and v8w1002, 12 weeks each spread over 2019–2025.
**Instruments:** `ml/eval/leak_guard.py`, `ml/tests/test_leak_guard.py`, `scripts/run_leak_guard.sh`,
`docs/standards/leak_guard.md` (`d401cb3`) → `ml/artifacts/phase23ac/t3/guard_v8.json`, `guard_v8w1002.json`,
`registry.json`; `reports/part2/phase23ac/t3/guard.csv`, `registry.csv`. Full run: 64 s per world, **exit 0**.
**Wire-in (one line, not done here):** add `scripts/run_leak_guard.sh` as a CI step or a pre-push hook; it exits 1 on any
new FAIL and 2 if the known-answer check is invalid.

## The registry (built by code, not typed)

253 entries in 16 families:

| family | columns |
|---|---|
| panel, raw | 15 |
| panel, clean | 15 |
| graph masters: STATIC | 3 |
| graph masters: FLAT | 10 |
| fwd_load | 15 |
| fwd_season | 15 |
| cadence | 5 |
| pulse | 4 |
| stock_asof | 6 |
| grpstats: KM stats | 14 |
| grpstats: arrival block | 56 |
| grpstats: fill block | 38 |
| grpstats: acknowledgement block | 12 |
| Phase 22 acknowledgement | 12 |
| Phase 22 L4 | 32 |
| the order-time row's own quantity | 1 |

Each entry has its source table, horizon and the model families that read it. Every feature of the persisted order-time
flag is covered (none UNCOVERED).

## Results

**Poison (H_week), both worlds:**
- All nine Phase 22 leaking columns FAIL, with the share of channels whose row moves at:
  - 29–81% on v8 (`load_ratio` 0.81, `otd_rate_last13` 0.30);
  - 29–81% on world 2.
- All nine clean replacements, and the six non-leaking raw columns, change **0**.
- **Known answer: satisfied on both worlds.**

**Family modules:**
- `fwd_load`, `cadence`, `pulse` and `stock_asof`: each falsifier fires on its constructed future row.
- `fwd_load` and `cadence`, recomputed with every source row recorded after t0 poisoned at 2 t0: **0 changes**.
- grpstats and Phase 22 tables: covered by the Phase 21 / 22 poison and self-exclusion tests.
- Graph masters: time-invariant, not applicable.

**Label alignment** (share of rows equal to an outcome of a line still pending at t), v8:

| column | raw: amended (literal) | clean: amended (literal) |
|---|---|---|
| lead_time_actual_days / ratio | **0.572** (0.572) | 0.0002–0.0003 (0.0002–0.0067) |
| load_ratio | **0.902** (0.903) | ≈ 0 (0.0007) |
| fill_rate / ack_gap_ratio | 0.161 (0.584) | 0.0000–0.0002 (0.31–0.38) |
| fill_rate_last4 / 13 / 52 | 0.079 / 0.013 / 0.0002 (0.45 / 0.29 / 0.13) | ≈ 0 (0.18–0.32) |
| otd_rate_last13 | 0.0003 (0.214) | 0.0001 (0.28) |
| qty_ordered / qty_received / is_active_week / revision_count | 0 (0.33 / 0.40 / 0.45 / 0.45) | inherits raw |

The values in brackets are the **pre-registered literal rule**.

**Deviation 226: the literal alignment rule cannot pass its own known-answer check.**
- Fill outcomes pile up at exactly 0 and 1. Any column sitting at 0 or 1 therefore "equals" some pending line's outcome.
- So the clean fill and on-time columns align at 0.21–0.38, and four harmless raw columns at 0.33–0.45 (each with 0 poison
  change).
- The guard as built does not count a match at exactly 0 or 1 (`ALIGN_ATOMS`); the threshold 0.20 and the tolerances are
  unchanged.
- This amendment was made **after** a smoke run on real data, so it is reported as a deviation, with the literal share
  beside every verdict, and P4 / P5 are scored on the literal rule.

The guard's decisive evidence for the nine is the **poison test**, which needed no amendment.

## Predictions

- **P4 (the guard flags all nine and no clean replacement): WRONG as pre-registered.** Under the literal alignment rule five
  clean replacements FAIL alignment. Under the amended rule (deviation 226): all nine flagged on both worlds, every clean
  replacement passes.
- **P5 (no further leaking column): WRONG as pre-registered.** Literal alignment flags `qty_ordered`, `qty_received`,
  `is_active_week` and `revision_count`, all of which change 0 under poison: spurious, not leaks. Under the amended rule
  and under poison: **no further leaking column on either world.** No FINDING beyond the known nine.
