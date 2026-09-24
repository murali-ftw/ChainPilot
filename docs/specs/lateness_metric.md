# The arrival lateness metric — definition of record

**Adopted in Phase 11C Stage C.** Implemented in `ml/eval/lateness_metric.py`; the reference is
built by `ml/eval/phase11b_lateness.py::build_reference`. This file exists so the next phase does
not re-derive any of it.

---

## The metric

```
R(channel, t0) = median{ lead_weeks : GRN lines of that channel RECORDED on or before t0 }  +  c

label   yl = arrival_week > R
score   pl = prediction - R
metric  ROC-AUC(yl, pl)
```

`lead_weeks` is `(receipt event_ts − po_line created_ts) / 7`, taken over GRN lines whose
`recorded_ts <= t0`. Nothing else enters the reference.

## `c`, the offset — what it is and what it is not

`c` is **one global constant, fitted on TRAINING rows only, and never refitted** — not per world,
not per fold, not per origin. On v8 it is **+4.571 weeks**.

It exists because `arrival_week` is measured from `t0` and contains two parts: the wait until the
order is raised, and the lead once it is. **Only the second varies by channel**; the first is a
property of the label's sampling window and is common to every channel. So the channel-varying
part of the reference is the historical lead, and the constant part is fitted once.

`c` shifts every reference equally. It therefore sets the **label's balance** (what counts as
late) and **cannot affect ranking within a reference**. Changing it changes the late rate, not the
ordering of arms.

**Cold start.** A channel with no receipts recorded by `t0` takes the training-fold median of the
base reference. On v8 this affects **0 rows**; it is not a live path there, but it exists and is
counted in the returned metadata.

## As-of, and its four failure modes

`assert_asof_receipts` guards the reference. All four are demonstrated in
`reports/part2/phase-11c.md` §4:

| input | result |
|---|---|
| correctly filtered set | passes |
| unfiltered whole history | **fires** — receipts recorded after `t0` |
| filtered set + one future-dated row | **fires** — a single row is enough |
| **empty set** | **fires** — "would silently fall back to the global constant" |

The emptiness guard is not decoration. It was added in Phase 11B after a falsification passed
vacuously: shifting every `recorded_ts` into the future leaves nothing at or before `t0`, the
check passed on an empty set, and the reference silently collapsed to the cold-start fallback for
every channel.

## Every arm scores its own prediction

There is **no constant-substitution path**. Phase 11C Stage A found that
`phase7_score.py`'s `arrival_promise` branch replaced the promise arm's stored prediction with a
constant before computing lateness. Because `pl = prediction − R`, any constant prediction ranks
on `−R`, which made two different baselines report the same number and mislabelled what they were
compared against. `ml/tests/test_no_lateness_substitution.py` fails if any arm's lateness differs
from the value its own prediction produces.

A constant ranker remains a legitimate baseline and is still emitted — under the explicit name
`roc_auc_late_CONSTANT_RANKER`, never as "the promise date".

## The promise reference is retired, and the prefix is load-bearing

`promise_week` is derived from a `po_line` raised ~7 weeks **after** the forecast instant
(clearance block B9 = 0.00% on every v8 dataset seed). It is kept for continuity under the
`PRIVILEGED__` prefix, and `assert_no_privileged_headline` makes that prefix enforceable: a
`PRIVILEGED__` metric **cannot be selected on and cannot be reported as a headline**. It raises on
both.

## Deviation 65 — both sentences travel together

Wherever arrival's lateness is quoted, both of these appear:

> Against the promise date the arrival head's lateness claim is **retired**: that reference is not
> available at t0.
> Against a t0-available reference it is **established** at five model seeds.

## Measured on v8 (test fold, 2025), for reference

| arm | seeds | ROC-AUC |
|---|---|---|
| head h⁴ | 5 | **0.70905** [0.70645, 0.71298] |
| b5flat LightGBM | 5 | 0.70535 [0.70476, 0.70573] |
| head h⁰ | 3 | 0.69999 [0.69842, 0.70133] |
| naive constant | 1 | 0.68697 |
| reference-only | 1 | **0.50000** |

Reference: 151 distinct values, late rate 47.54%. The head beats every arm with **disjoint** bands.
`reference-only` scoring exactly 0.5 is the health check: an arm that predicts the reference has a
constant lateness score and must not discriminate.
