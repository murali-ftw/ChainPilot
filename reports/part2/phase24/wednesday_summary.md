# ChainPilot, Wednesday summary: what it can be trusted to do today

*For the client meeting. Every figure comes from one simulated supply chain (test year 2025) built to resemble Rane's.*
*(Q) means quoted from a report; (R) means recomputed in Phase 24. Sources are listed at the foot of the page.*

## 1. Every use case, on clean data

"Right N in 10" counts only the cases the tool flags most confidently. "By chance" is how often a blind guess would be right
on the same cases.

| use case | what it does | clean result (range) | by chance | status |
|---|---|---|---|---|
| Delivery date, open orders | predicts when an open order line will arrive | typically off by **12.6 days** (12.2–13.1) (Q1) | — | **ships** |
| Late-order list | ranks the open lines most likely to be late | right **9.5 in 10** (9.3–9.7) (Q1) | 7.3 in 10 | WATCHLIST |
| **Delivery date at order time** *(new)* | the moment a line is raised: expected date plus a range | typically off by **10 days**; the promise date is off by **14** (Q2). The 80% range holds 80% of the time (Q2) | — | **ships** |
| Late risk at order time | flags new lines likely to be late | right **6.2 in 10** (6.1–6.2) (Q1) | 3.5 in 10 | WATCHLIST |
| Arrives in full | ranks the lines that will arrive complete | right **9.4 in 10** (9.3–9.6) (Q1) | 7.5 in 10 | WATCHLIST |
| Arrives short | ranks the lines that will arrive materially short | right **5.1 in 10** (4.5–5.5) (Q1) | 2.5 in 10 | WATCHLIST |
| Supplier overload | ranks the suppliers likely to exceed capacity in the next 90 days | right **7.3 in 10** (7.0–7.8) (Q1) | 4.1 in 10 | WATCHLIST |
| Shortage risk | which part-plants run short | 2.2 in 10 at a realistic 3% base (Q3) | 0.3 in 10 | RETIRED |
| **Predict-the-rescue** | which weeks planners will need to move stock in | right **7.3 in 10** (7.0–7.7), catching a quarter of those weeks (Q4) | 4.5 in 10 | **ALERT, marginal** |
| Shortage simulation | plays the next 13 weeks forward many times | matches the stock picture before rescues to within **14%** (R5). Its week-by-week flags are right **6.1 in 10** (5.7–6.4) (R5) | 4.5 in 10 | mechanism holds; flags WATCHLIST |
| Delivery schedule, supplier split, stock transfer | how much, from whom, from where | no model result | — | **BLOCKED** |

## 2. What ships, what is closed, what needs Rane's data

**Ships:**
- the **order-time expected delivery date with its 80% range**;
- the open-order delivery date;
- **ranked watchlists** for late, short, overload and simulation flags;
- **predict-the-rescue as a watched alert**.

**Closed.** These were tried and gave no gain:
- smarter network models;
- order-month and channel-group statistics;
- the acknowledgement gap;
- capacity recalibration.

**Blocked on Rane's data** (`docs/client/data_request_v2.md`):
1. **Dated versions** of the production plan / demand forecast.
2. **Channel-level stock position and reorder points**, with recorded timestamps.
3. **Costs:** holding, ordering, freight and shortage.
4. **Supplier-declared capacity history** and any observed capacity evidence.
5. **Purchase-order line creation history** and reorder rules.

## 3. The leak, in four lines

1. **Nine input columns carried the future:** the weekly supplier table wrote each order's eventual outcome into the week it
   was ordered.
2. **Removing them cost most on supplier overload**, which drops from an alert (8.5 in 10) to a watchlist (7.3 in 10).
3. **It was found** when an order-time model scored impossibly well (0.99). Rebuilding every column from the raw tables showed
   which nine moved when the future was hidden.
4. **A standing guard now checks every input on every run** (`scripts/run_leak_guard.sh`). It flags exactly those nine and
   nothing else.

## 4. Three honest caveats

1. **One simulated generator.** The second test world comes from the same generator: it tests stability, not reality.
2. **Absolute numbers must be re-measured on Rane's extract** before they are quoted outside the project.
3. **The order-time range is slightly too narrow in January:** it holds 77% of the time, against 80% (Q2).

## 5. Progress per use case: first model → best published (leaky) → clean

| use case | first model (leaky) | best published (leaky) | **clean** |
|---|---|---|---|
| late-order list (right in 10) | 9.7 | 9.7 | **9.5** |
| delivery date, open orders (days off) | 13.1 | 12.3 | **12.6** |
| delivery date at order time (days off) | — | — | **10.0** |
| late risk at order time (right in 10) | — | 7.9 | **6.2** |
| arrives in full (right in 10) | 8.8 | 9.3 | **9.4** |
| arrives short (right in 10) | 3.9 | 5.3 | **5.1** |
| supplier overload (right in 10) | 8.5 | 8.7 | **7.3** |
| predict-the-rescue (right in 10) | 8.2 | 8.6 (2 of 5 runs) | **7.3** |
| simulation vs pre-rescue stock (times) | 1.13 | 1.13 | **1.14** |

**Update after Phase 24 Stage 6** (added after the table above was committed). The neural version of predict-the-rescue,
retrained on clean data with all five runs, is right **8.0 in 10** (7.9–8.2), against the LightGBM's 7.3. It clears 7 in 10
in every test period (R6). It is **not servable yet**: it needs a serving module and the owner's decision. Four of five
runs were still improving when they stopped, so this is a floor.

---

**Sources.**
- **Q1:** `reports/part2/phase24/metrics_pack_v2.md` (Phase 23AC tables, `d401cb3`), top 5% flagged.
- **Q2:** `reports/part2/phase23ac/t1/t1_order_time.md` (`ec9c68b`).
- **Q3:** `reports/part2/phase-17.md` §1.1, measured on the leaky inputs and not re-measured (retired).
- **Q4:** `reports/part2/phase-23b.md` §1 (`535361d`): clean B1a 0.733 at recall 0.254, ensemble range 0.697–0.773.
- **R5:** `reports/part2/phase24/stage2_simulation.md` (Phase 24, `cda8074`): 1.137×; 0.612 (range 0.572–0.643).
- **R6:** `reports/part2/phase24/stage6_b1b_clean.md` (`632eeb8`): five-run ensemble 0.805 (0.791–0.824).
- **Progress row:** `reports/part2/phase23ac/t4/progress.md` and `reports/part2/phase24/progress_v2.md`. Leaky figures are
  "LEAKY (superseded, Phase 22)", shown only as history.
