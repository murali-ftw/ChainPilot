# Phase 21 Stage 1 — data audit and group sparsity

**Audience:** anyone checking what the 5-D group statistics can and cannot be built from.
**Measured on:** v8 seed 1001. Read-only; no model.
**Instrument:** `ml/eval/phase21_audit.py` at `6248fae` → `ml/artifacts/phase21/stage1_audit_v8.json`. Tests:
`ml/tests/test_phase21_grpstats.py` at `1a497b5` → `ml/artifacts/phase21/selftest_v8.json`.
**Status:** complete. Every pre-registered (D3–D5) choice holds except where stated.

## a. Lane / shipping attributes

| column | as-of? | distinct values | verdict |
|---|---|---|---|
| `sourcing_channels.transport_mode` | static master | 4 (road 11,527 / rail 2,261 / sea 1,470 / air 814 channels) | **in the cluster** |
| `logistics_lanes.standard_transit_days` | static master | 55 | **in the cluster** as terciles (cuts 13.2 / 20.4 days) |
| `sourcing_channels.transport_distance_km` | static master | 2,149 | not in the cluster (equals the lane's `distance_km` on 100% of channels; already a BASE feature) |
| `logistics_lanes.carrier_id` | static master | 79 | not in the cluster (one carrier per lane, so it identifies the lane; it adds no pooling) |
| `logistics_lanes.via_checkpoint` | — | **0 (100% empty)** | unusable (confirms Phase 18 deviation 161) |
| `supplier_sites.country`, `suppliers.country` | static | **1** (India) | constant: dropped |
| `supplier_sites.state` | static | 7 | not a lane attribute (a supplier attribute, constant within L2) |
| `purchase_orders.incoterm` / `po_type` / `currency` | PO header, as-of at PO recording | **1 each** (DAP / standard / INR) | constant: dropped. Even if they varied, a snapshot row's PO does not exist at t0 |

- **One lane per channel: confirmed.** The lane table has 16,072 rows for 16,072 channels; row *i* is channel *i*
  (site, plant, mode and distance all equal). 13,267 channels share a (site, plant, mode) key with another channel, so
  a key join fans out; the builder aligns by row and asserts it (deviation 197).
- **Lane clusters: 12** (4 modes × 3 transit terciles), all used; a supplier's channels span 5–12 clusters (median 9).
  Because L4 is the channel, the lane cluster is constant within L4 and L5, as the pre-registration expected. It
  separates groups only inside L3 (supplier × month × cluster).

## c. Expected delivery and base rates

| quantity | value |
|---|---|
| lines with `original_promise_date` | **100%**; it equals created + contracted lead on **100%** of lines |
| lines with a `po_line_schedules` row / an ASN | 37.6% / 95.1% (written at creation / at dispatch) |
| snapshot rows (244,000 = arrival rows = fill rows, identical row for row) whose line is **created ≤ t0** | **0.0%** (raised 1.0–12.7 weeks after t0, median 6.7) |
| so the order-month key at snapshots is | the calendar month of t0 for **100%** of rows (D2's seasonal proxy) |
| so the expected-delivery fallback is used for | **100% of snapshot rows**. At placement the promise is used for 100% |

**Late vs contract (lines created in train 2019–2023 with a receipt, 495,965 lines; 10,677 train lines never receipted):
37.1%.** By creation month:

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| late rate | 0.228 | **0.214** | 0.325 | 0.413 | 0.334 | 0.361 | 0.346 | 0.399 | **0.505** | 0.415 | 0.414 | 0.371 |
| median lead (days) | 27 | 27 | 31 | 35 | 32 | 33 | 32 | 35 | **40** | 36 | 35 | 32 |

**There is a strong order-month season** (late rate 0.21 → 0.50; median lead 27 → 40 days). By lane cluster the late rate
is flat: 0.361–0.381 across all 12 clusters. **The lane cluster carries no lateness signal of its own.**

## b. Group sparsity (receipted deliveries per non-empty group, KM-eligible)

| as-of | level | groups possible | non-empty | min / P10 / median / P90 / max receipted per group |
|---|---|---|---|---|
| train end (2023-12-31) | L5 | 192,864 | 181,566 | 0 / 2 / **4** / 7 / 19 |
| | L4 | 16,072 | 15,334 | 1 / 37 / 51 / 72 / 134 |
| | L3 | 60,480 | 44,439 | 0 / 3 / 10 / 46 / 113 |
| | L2 | 420 | 420 | 983 / 1,552 / 1,941 / 2,344 / 2,789 |
| val start (2024-01-01) | L5 | 192,864 | 181,570 | 0 / 2 / **4** / 7 / 19 |
| | L4 | 16,072 | 15,334 | 1 / 37 / 51 / 72 / 134 |
| | L3 | 60,480 | 44,440 | 0 / 3 / 10 / 46 / 113 |
| | L2 | 420 | 420 | 983 / 1,553 / 1,942 / 2,346 / 2,789 |
| test start (2025-01-01) | L5 | 192,864 | 182,084 | 0 / 2 / **5** / 8 / 21 |
| | L4 | 16,072 | 15,334 | 1 / 41 / 57 / 81 / 151 |
| | L3 | 60,480 | 44,471 | 0 / 3 / 11 / 52 / 125 |
| | L2 | 420 | 420 | 1,071 / 1,734 / 2,164 / 2,605 / 3,123 |

(As of the test start: 940,379 lines in the history, 909,641 receipted; the newest line recorded 2025-01-01 00:00:00 and
the newest receipt 2024-12-31 23:45, both ≤ t0.)

Share of snapshot rows whose group has **< 5 prior receipted deliveries** at the row's t0:

| fold | rows | L5 < 5 | L4 < 5 | L4 = 0 (cold) |
|---|---|---|---|---|
| train | 176,000 | 79.6% | 0.015% | 0.0006% |
| val | 32,000 | **42.9%** | 0.0% | 0.0% |
| test | 36,000 | **39.4%** | 0.003% | 0.0% |

**The 5-D cell is thin by construction:** a median of 4–5 receipts per (channel × month) after 8–9 years of history.
Shrinkage is not optional at L5. L4 (the channel) is rich (median 51–57); **there are no cold channels at
snapshots**, so the pre-registered "cold" slice is empty in practice and the thin slice is relative (D11 threshold T = 47
receipts, the validation P20, chosen in Stage 2).

## d. Future-poison and self-exclusion tests (run before any feature was used)

Three test snapshots spread over the fit window, 400 sampled rows each (`selftest_v8.json`, `1a497b5`):

| test | builder | constructed offender | verdict |
|---|---|---|---|
| **future poison**: every line, receipt, acknowledgement and zero signal recorded after τ replaced with noise | **0 / 400** rows changed at each of the 3 τ | receipts filtered on **event** time: **400 / 400** changed | **PASS**, and the test can fire |
| **self-exclusion**: each row's own line given a receipt, acknowledgement and zero signal recorded **before** τ | **0 / 400** changed; the own line is not in its own history (asserted) | own line admitted to the history: **400 / 400** changed | **PASS**, and the test can fire |
| unit: monotone backoff, shrinkage limits (n → ∞ raw; n = 0 parent; n = k half-way), KM censoring | — | k = 0 makes a thin L5 resolve to L5 | **PASS**. A synthetic group with 50% long open lines: KM median 29.9 vs completed-only 21.7 |

The builder also asserts as-of on every source row at every τ (363 creation weeks for placement, 61 snapshots); no assertion fired.
