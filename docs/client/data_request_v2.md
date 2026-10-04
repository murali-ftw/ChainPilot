# ChainPilot × Rane — data request, version 2

**For:** Aptimeta / Rane data and planning leads. **From:** the ChainPilot modelling team. **Date:** 2026-10-04.
**Supersedes:** the cost-parameter asks in `results/observation1.md` §10.2 (they are repeated here as ask 3).

**How to read this.** Each ask says **what** we need, **why**, **what we measured** when we had the equivalent data in our
synthetic test world, and the **honest limit** of that measurement. All measurements come from a simulated supply chain
built to resemble Rane's. Gains there are an upper bound: **every figure must be re-measured on your data**, and our two
simulated worlds already disagreed on effect sizes by a factor of 0.3 to 2.6. Sources: `results/observation2.md` and the
phase reports it cites.

**One rule applies to everything below: we need each record's *recorded* date, not only its *event* date.** A forecast for
a past date may only use what was known on that date. Without recorded dates, we cannot test anything honestly.

---

## Ask 1 — Dated versions of your production plan / demand forecast

- **What:** every version of the forward requirement plan (or the MRP demand forecast) you have kept, at part × plant ×
  week grain, each with the date that version was published or recorded. Manual overrides with their dates, if they exist.
- **Why:** the orders we try to predict are raised *after* the forecast date. Your plan says how much will be needed in the
  coming weeks. In our tests it was the only new input that improved the models in five rounds of work.
- **What we measured:** with the plan as it stood on each date, delivery-lateness ranking improved (AUC +0.013), and the
  capacity alert's precision in its top 5% rose by about +0.12 in the simple model. The gain held on a second simulated
  world, and it was a **ranking** gain: it did not change which use cases can be alerts.
- **Honest limit:** the size of the gain moved by ×0.3 to ×2.6 between two simulated worlds from the same simulator. Its
  value on your data is unknown until measured. If plan versions were overwritten in place, the earlier versions cannot be
  rebuilt, and this ask cannot be met retrospectively.

## Ask 2 — Channel-level stock position and reorder points, with recorded timestamps

- **What:** for each supplier × part × plant (or the finest grain you hold), on-hand stock, on-order quantity, safety stock
  and reorder point **as they stood on each date**: snapshots or a ledger, with recorded timestamps. Reorder points must be
  the values in force at the time, not today's values.
- **Why:** most of the uncertainty in "will this order arrive late?" is *when the order will be raised*, and the reorder
  trigger is the rule that raises it.
- **What we measured:** given the simulator's **exact** channel-level position and reorder point, lateness ranking rose
  from AUC 0.705 to **0.739** (+0.034), and median delivery-date error fell from 13.2 to 12.7 days. A stock ledger at
  part × plant grain without as-of reorder points added only +0.002.
- **Honest limit:** +0.034 is an **upper bound**: the simulator's stock is exact and instantly current, and an ERP's is
  not. Even perfect stock data recovers only **17%** of the timing headroom; the rest is demand that has not happened yet.
  Worth asking for, not worth expecting miracles from.

## Ask 3 — Costs: holding, ordering, freight and shortage

- **What:** inventory carrying cost (rate or per-unit-week), ordering / setup cost per purchase order, freight structure
  (truck capacity and tariff), and the cost of a shortage (per unit, or per line stop).
- **Why:** the delivery-schedule optimiser, the supplier-allocation recommender and transfer suggestions need an objective.
  Without holding cost, "order everything on the first day" and "order everything on the last day" cost the same, and the
  optimiser can say *how much* but not *when*.
- **What we measured:** nothing can be measured until these exist. **None of the four exists in any dataset so far.** We will
  not invent them.
- **Honest limit:** these unblock three use cases but do not guarantee that they will work well. Allocation is also blocked
  by a qualification rule: our current rule forbids even the incumbent supplier for 57 of 120 part-plants, which needs one
  conversation with your buyers.

## Ask 4 — Supplier-declared capacity history and any observed capacity evidence

- **What:** every declared-capacity statement per supplier (with its date), and any evidence of actual capacity: audits,
  shift patterns, maximum demonstrated output, allocation constraints, notices of shutdowns or slowdowns.
- **Why:** the capacity alert's remaining headroom is the supplier's **true capacity in each month**.
- **What we measured:** knowing true monthly capacity would raise the alert's precision in its top 5% by about **+0.09**
  (0.851 → 0.944). In our simulator, declared capacity tracks true capacity at a correlation of only 0.11, so no
  forecast-time source reaches it.
- **Honest limit:** if your declared capacities are equally loose, this ask gains nothing. It pays only if some record
  tracks real capacity month to month.

## Ask 5 — Purchase-order line creation history and reorder rules

- **What:** every PO line with its creation timestamp and recorded timestamp, partial receipts as **separate rows** (not
  overwritten), and the reorder rules or MRP parameters in force (review cycle, lot sizes, minimum order quantities, cover
  targets), with the dates they changed.
- **Why:** the order-raising time is arrival's largest lever. Knowing the creation week alone, with no other information,
  lifts lateness ranking from 0.705 to **0.910** in our simulator.
- **What we measured:** a channel's past ordering rhythm alone recovered only **4%** of that. The rules themselves (not just
  the history) are what might do better.
- **Honest limit:** 0.910 is an oracle figure (it knows the future creation date). How much a reorder rule plus the stock
  of Ask 2 recovers is unmeasured; our best simulator estimate is the 17% of Ask 2.

---

## Gate F0 — checks we will run on the extract before modelling anything

| check | why it matters | pass condition |
|---|---|---|
| **Can as-of state be reconstructed?** Every table carries a recorded / posted timestamp, and history is appended, not overwritten | every forecast must use only what was known on its date | recorded timestamps present on POs, receipts, plan versions, stock, capacity; no in-place overwrites of history |
| **Are partial receipts separate rows?** | fill rate and short-delivery models need each receipt, not the final total | ≥ 1 PO line with more than one receipt row; receipt quantities sum to the received total |
| **Are short deliveries above 2%?** | below that, a "materially short" list has too few positives to learn or to test | share of lines with fill < 0.95 is ≥ 2% over the extract |
| **Are dated manual forecasts available?** | planners' overrides are part of the plan as it stood | override records with their own dates, or a confirmation that none exist |
| **Is the opening stock balance present?** | stock roll-forward and any position signal need a starting point | an opening balance (or a full ledger from day one) per part × plant |
| **Are the plan versions dated?** | Ask 1 is impossible without them | ≥ 2 versions per target week, each with a record date |

We will report the result of each check before any modelling, and stop where a check fails.
