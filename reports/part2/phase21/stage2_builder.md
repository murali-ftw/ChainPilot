# Phase 21 Stage 2 — the group-statistics builder

**Audience:** whoever reuses or audits the 5-D group features.
**Measured on:** v8 seed 1001; second world v8w1002 for Stage 7.
**Code:** `ml/data/grpstats.py` (new; nothing existing edited), `ml/data/phase21_paths.py` (data-root override for the
worktree), `ml/tests/test_phase21_grpstats.py`. **Validation-only choices:** `ml/eval/phase21_select.py` (`6248fae`) →
`ml/artifacts/phase21/k_select_{world}.json`, run at 23:58 IST, **before any test number was computed**.

## What it builds

At an as-of instant τ the builder takes every PO line **recorded ≤ τ** (an expanding window from 2016), and for each
line:
- its first receipt if that receipt is **recorded ≤ τ** (event), else censoring at τ − created;
- lateness vs contract = lead − contracted lead (the line's `original_promise_date` is created + contract on 100% of
  lines, Stage 1).

It then computes, for every group at every level at once (a vectorised Kaplan–Meier, events before censorings at ties):
P10 / P25 / P50 / P75 / P90 of lead and of lateness, S_lateness(0) = P(late), receipted n and all-lines n. It also
builds the closed-line fill histograms (Phase 13's closed rule, as-of) and the acknowledgement gap (all acknowledged
lines; acknowledged lines still open).

Rows are stored raw. L4 / L2 / L1 are stored once; L5 and L3 for **all 12 months** (the "month cube"), so the
permuted-month control reads a donor month without a rebuild. Shrinkage (D6) is applied at assembly time.

| store | rows | τ values | build | commit / config hash |
|---|---|---|---|---|
| `grpstats_v8_snap.npz` (146 MB) | 244,000 (= the arrival and fill label rows, identical) | 61 snapshots | 92 s | `f7a7fdf` / `308a7e04a9b5` |
| `grpstats_v8_place.npz` (91 MB) | 218,881 distinct lines (train 155,593 / val 31,494 / test 31,794) | 363 creation weeks | 404 s | `f7a7fdf` / `308a7e04a9b5` |
| `grpstats_v8w1002_snap.npz` (141 MB) | 244,000 | 61 | 122 s | `c2a756c` / `bbd156a310c4` |

The two config hashes differ only because `placement_labels` was appended to the module between the builds; the
snapshot code path is byte-identical. The stores are not committed (`ml/artifacts/` is ignored). Each carries its
world's table hashes, its commit and `code_dirty = false`.

## Tests (each with a constructed failing case)

| test | result |
|---|---|
| monotone backoff: n = (0, 50, 500) at (L5, L4, L2), k = 10 → L4; (20, 50, 500) → L5; (0, 2, 500) → L2; (0, 0, 5) → L1 | PASS; k = 0 resolves a thin L5 to L5 (the test can fail) |
| shrinkage limits: n → ∞ gives the raw value; n = 0 at every level gives the global; n = k gives the half-way point | PASS |
| KM censoring: 50 receipted lines U(10, 30) + 50 open lines U(25, 60) | KM median **29.9** > completed-only **21.7**: PASS. No censoring: KM = empirical median. 1..4 all events: median 2 |
| future poison (3 snapshots × 400 rows, both worlds) | 0 rows change; the event-time offender changes 400 / 400 |
| self-exclusion (same sample) | 0 rows change; admitting the own line changes 400 / 400 |
| as-of assertion on every source row at every τ, non-empty history | never fired (424 τ on v8, 61 on v8w1002) |

The KM unit test caught one real defect before any feature was used: a survival value of exactly 0.5 was read as
"still above" through floating-point error, which shifted medians up by one event. Fixed with a 1e-12 tolerance (`1a497b5`).

## Validation-only choices (D6, D9, D11)

**k** (shrinkage constant), standalone rule's validation metric:

| k | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|
| arrival, snapshot: val lateness AUC | 0.518 | 0.523 | 0.572 | 0.661 | 0.682 | **0.684** |
| fill, snapshot: val exact CRPS | 0.1459 | 0.1379 | 0.1314 | 0.12901 | **0.12900** | 0.1299 |
| at placement: val lateness AUC | 0.543 | 0.545 | **0.546** | 0.536 | 0.525 | 0.520 |
| v8w1002 arrival / fill | 0.514 / 0.1087 | 0.518 / 0.1024 | 0.566 / 0.0975 | 0.647 / 0.0958 | 0.666 / **0.0957** | **0.669** / 0.0963 |

Chosen: **arrival k = 300, fill k = 100, placement k = 10**. Arrival's k sits at the edge of the pre-registered grid
(deviation 200). With k = 300, a channel's ~50 receipts get weight 50 / 350 and an L5 cell's ~5 receipts 5 / 305.
**Every snapshot row therefore resolves to the supplier level (L2)**: validation says the channel-month cell is too
noisy to stand on its own.

- **Offsets a** (median of Y − rule on validation observed rows, for A3): snapshot 4.29 weeks (the wait until the
  line is raised); placement −0.32 weeks.
- **Slice threshold T** (validation P20 of n4 over rows with n4 > 0): snapshot **47** receipts (world 2: 45); placement 47.
