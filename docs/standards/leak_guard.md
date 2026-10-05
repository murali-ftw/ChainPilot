# Standard: the leak guard

**Instrument:** `ml/eval/leak_guard.py` (tests: `ml/tests/test_leak_guard.py`; runner: `scripts/run_leak_guard.sh`).
**Defined in:** `reports/part2/phase-23ac-preregistration.md`, T3 and the Gates table.
**Status:** standing check. Run it before you quote any number from a model that reads a new or changed feature.

## Run it

```bash
scripts/run_leak_guard.sh          # = HADES_DEVICE=cpu ./venv/bin/python ml/eval/leak_guard.py --worlds v8,v8w1002
```

It refuses to run if `ml/` has uncommitted changes. It writes:
- `ml/artifacts/phase23ac/t3/registry.json`
- `ml/artifacts/phase23ac/t3/guard_{world}.json`
- `ml/artifacts/phase23ac/t3/guard.log`
- `reports/part2/phase23ac/t3/registry.csv`
- `reports/part2/phase23ac/t3/guard.csv`

To make it standing, add `scripts/run_leak_guard.sh` to CI or to a pre-push hook. The script edits neither.

| exit | meaning |
|---|---|
| 0 | Every FAIL is a `KNOWN_FAIL`: the nine `clean_panel.LEAKING` columns on the **raw** panel, which no clean model reads. |
| 1 | A FINDING: a FAIL outside `KNOWN_FAIL`, or a feature that a reader uses and no family registers (UNCOVERED). |
| 2 | Guard invalid: the known answer did not reproduce (see below). Nothing else in the output can be trusted. |

## Why it exists

The Phase 0–1 leakage check took the maximum |correlation| between each feature and each label. Its result was 0.319, and
it passed. Meanwhile nine panel columns carried each PO line's **eventual** outcome, written into the week the line was
ordered (`reports/part2/phase22/stage1_leak.md`).

A forward-filled eventual outcome correlates only weakly with any single label: it describes one line among many, and it
is overwritten by the next order. But it **equals** a still-pending line's outcome, exactly. So a leak test has to check
values against outcomes, not correlations. The guard does three things:
- poisons the future and checks that nothing moves;
- checks whether values equal pending outcomes;
- checks every feature anyone reads, from a registry built by code, so a new feature cannot skip the tests.

## What it checks

**Registry** (`build_registry`). Built from the builders' own code, never typed by hand. Each entry records the column,
family, source tables, horizon, the model families that read it, and the test applied. The entries come from:

| source | how it is read |
|---|---|
| Panel columns | `ml/artifacts/cache/{v8,v8clean}/meta.json` `cols` / `nullable` |
| `phase7_fit.World` STATIC and FLAT | AST of `World.__init__` (the module is never imported: it pulls in lightgbm) |
| Family modules' `COLS` | imported from `fwd_load`, `fwd_season`, `cadence`, `pulse`, `stock_asof` |
| grpstats | `ASTATS`, plus the arrival / fill / ack block names from `assemble_*` called on a one-row dummy |
| phase22_rows | `ACK_COLS` and its L4 fill block |
| Order-time product | its flag model's feature list, from `ml/artifacts/phase22/serve/order_time_v8clean/product.json` |

Source tables come from each module's own `*.csv` literals. A flag feature that no family registers becomes
**UNCOVERED**, which exits 1.

**Future poison** (per panel column, per world, at 12 snapshot weeks spread over 2019–2025). Every source row that becomes
visible after the row's own week t is replaced with noise. The column FAILs if any channel's row-t value moves (NaN-safe,
1e-9).
- **Raw panel:** the exact generator rebuild in `ml/eval/phase22_leakscan.py`. It reads `_sim.npz` and is an audit tool
  only, never a model input. It reproduces the stored panel on 100% of values.
- **Clean panel:** `clean_panel.clean_columns` on the poisoned CSVs.
- **Inherited columns:** the six columns the clean cache does not replace take the raw result. The guard first asserts
  that they are byte-identical between the two caches.

**Label alignment** (panel columns, same weeks). At week t, a line is **pending** if it is ordered (visible week ≤ t) and
its outcome is not yet visible. The outcome becomes visible at the first receipt or, for a line closed at zero, at the
rejection or qty → 0 revision.

The candidates for each channel are:
- each pending line's eventual lead;
- lead / contracted lead;
- eventual fill `min(delivered / ordered, 1)`, and 1 − fill;
- the quantity-weighted eventual fill of the lines ordered in week t, if any of them is pending, and 1 − that fill;
- the supplier's eventual whole-month ordered quantity / declared capacity for the month of t, if that month is still in
  progress (a line of the month not yet visible, or its capacity recorded later).

Share = rows whose value equals a candidate within the column's tolerance, divided by active rows. Active rows are rows
with a pending line or a month in progress. Tolerances: lead 0.006 days; lead ratio and load ratio 6e-5; others 1e-6.

**FAIL if the share is ≥ 0.20.** This constant is pre-registered. Never tune it.

Candidates equal to exactly 0 or 1 are not matched (`ALIGN_ATOMS`). Fill outcomes pile up on those two values, so any
column at 0 or 1 would "equal" some pending outcome without identifying it. The literal share, with those candidates
included, is recorded beside the verdict share in every output. The measurement behind this choice is at the constant in
the code.

**Self-exclusion.** Where a column is row-specific, the row's own line must never enter it. This is covered by
`ml/tests/test_phase22_clean_panel.py` (`test_self_exclusion`) and `ml/tests/test_phase21_grpstats.py`, and the guard
records it by reference.

**Family modules:**
- **Falsify:** `fwd_load`, `cadence`, `pulse` and `stock_asof` each run `falsify()`, which must fire on its constructed
  offender.
- **Poison:** `fwd_load` and `cadence` are also recomputed at two sampled t0. Every row recorded after t0 is given random
  values, with its `recorded_ts` kept, and the output must not move.
- **`fwd_season`:** a network mean of `fwd_load` cells, so it inherits `fwd_load`'s verdict.
- **grpstats / phase22_rows:** covered by the Phase 21 / 22 tests.
- **STATIC / FLAT:** time-invariant masters, so N/A.

**Known answer.** On both worlds, the nine `LEAKING` columns must FAIL (poison or alignment) on the raw panel, and their
nine clean replacements must PASS both tests. If they don't, the guard raises `GuardInvalid` and exits 2.

## Registering a new feature

Add the feature where the registry already reads, never to a hand list in the guard:
- **A panel column:** it appears through the cache `meta.json`. Add its rebuild to `phase22_leakscan.rebuild` (raw) or to
  `clean_panel.clean_columns` (clean). Otherwise poison and alignment cannot test it.
- **A new family module** in `ml/data/` must expose three things:
  1. `COLS`, its column names;
  2. a pure per-t0 function, `snapshot_*(S, t0)` on sources from `load_sources(world)`, with every row filtered on
     `recorded_ts <= t0` and passed through `fwd_load.assert_asof`;
  3. `falsify(world)`, which feeds a constructed future-recorded row to the assertion and asserts that it fires.

  Then add the module to `FAMILY_MODULES` and `READERS` in `leak_guard.py`, and to `family_checks`, with a poisoned
  recompute if it has a per-t0 function.
- **A model that reads features from a new product file:** have `build_registry` read that file's feature list, the way
  it reads the order-time product. Any listed feature with no family is then UNCOVERED.

## When it fails

- **Exit 1, poison FAIL:** the feature reads a source row that is not visible by the row's horizon. Find the filter that
  uses `event_ts`, a whole-period total, or an eventual outcome. Rebuild the feature on `recorded_ts` / the visible week.
  Re-run the guard and the feature's own falsifier. Treat every stored result that read the feature as leaky until it has
  been restated on the clean feature.
- **Exit 1, alignment FAIL without poison FAIL:** the value equals pending outcomes although the poison did not move it.
  Either the poison misses a source the feature reads (extend the poison), or the feature is a deterministic copy of the
  outcome. Treat it as a leak until the cause is shown.
- **Exit 1, UNCOVERED:** register the feature (above) before any model that reads it is quoted.
- **Exit 2:** the instrument is broken. Do not read any other row of the output. Fix the guard and record the cause as a
  deviation.
- Record every new FAIL as a numbered deviation in the current phase report. Never widen `ALIGN_FAIL` or a tolerance to
  make a column pass.
