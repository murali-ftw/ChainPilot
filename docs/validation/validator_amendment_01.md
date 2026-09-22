# Validator amendment 01 — censored-tail denominator

**This is the first authorised change to `db/validator.py`, which has been frozen since run 1.**
It was authorised in the run-7 brief, Part 2. Nothing else in the file changes: no band,
tolerance, threshold, gate condition, skip rule or exclusion.

| | |
|---|---|
| SHA before | `c49fc57683ca8671906d1b73f309faed63d48ae1b7299b0bf53908bb1a8c5f3c` |
| SHA after | `ed4742e0bef9d60af7c5b2a2550b590b27e6b217574df5b17877ba99be57ac56` |
| Check affected | `right-censored tail is the right size` (and its `[2019-2025]` twin) |
| Functions touched | `_select`, `fill_population` (return signature), `tier3` (denominator) |

## What was wrong

The check has two halves that were measured on different date bases.

**Numerator.** A line is counted as right-censored when its *promise* date falls in the final
90 days ([validator.py `_select`](../db/validator.py)):

    horizon_end = edge - 90 days
    if L["prom"] and L["prom"] > horizon_end: censored += 1

**Denominator.** The expected share was the 90-day window over a span measured on *created*
dates:

    span_days      = (end - start).days        # end/start are creation (and GRN) dates
    expected_share = 90.0 / span_days

## The arithmetic

`original_promise_date = created_ts + contracted_lead_time_days`, and
`contracted_lead_time_days ~ U(15, 70)` with mean 42.5.

A line is censored when `created + contracted > edge − 90`, i.e. when
`created > edge − 90 − contracted`. Averaged over the lead-time distribution, the censored
population therefore covers

    90 + E[contracted] = 90 + 42.5 = 132.5 days

of order creation, and it is being compared against a normaliser that assumes 90. The ratio has a
floor of

    132.5 / 90 = 1.47x

against a band of 0.3–1.2×, **for any world with non-zero contracted lead time**. The floor rises
with the mean lead time and cannot be reduced by anything a generator controls short of setting
lead times to zero, which §8 forbids.

Run 6 predicted this floor before generating and then measured **1.47× (seed 1001)** and
**1.46× (seed 1002)** — the prediction and the measurement agree to two decimal places, which is
what identifies it as an instrument defect rather than a property of the world.

This is the mirror image of the failure mode this project named "gates that can't fail": a gate
that cannot pass carries no information either.

## What changed

Normalise by the width of the censored window as a fraction of the **promise** timeline — the same
variable the censoring predicate tests:

    prom_lo, prom_hi = min/max original_promise_date over the selected population
    expected_share   = (prom_hi - horizon_end) / (prom_hi - prom_lo)

If promise dates were uniform over the population, `share` would equal `expected_share` and the
ratio would be 1.00 by construction. The check now asks the question it was written to ask: is the
tail of the promise timeline over- or under-represented relative to its width?

`_select` additionally returns `prom_span = (prom_lo, prom_hi, horizon_end)`; `fill_population`
propagates it; `tier3` uses it. Where a population has no promise dates at all the old
`90 / span_days` denominator is retained as a fallback, so a dataset lacking promise dates behaves
exactly as before.

## Conditions of the amendment

Per the brief, the corrected check is accepted on **every** dataset, including the control's
reference world and including v6 restated. It is a correction to a derivation error, not a band
adjustment to let v7 pass. The results — control movement, v6 restated, v7 — are reported in
[validation7.md](validation7.md) §2 and §8 without adjustment.
