# Phase 23B Stage 3 — the shortage simulation on clean inputs

## PENDING: needs clean_predictions from the Mac

The simulation consumes no stored prediction file. `ml/sim/montecarlo.read_heads` **runs two neural heads forward** on the panel
at each snapshot (Stage 1c), so a clean re-run needs Phase 22's **clean neural bundles**: checkpoint, normaliser and
recalibration. They were trained on the Mac (MPS) and **none is on this machine** (`ml/artifacts/phase22` is absent):

| needed for | exact path (from `ml/eval/phase22_restate.py`: `B22 = ml/artifacts/phase22/bundles`, `INC`) |
|---|---|
| arrival distribution (the simulation reads seed 7 only) | `ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7/` |
| fill distribution, fill seed 7 | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s7/` |
| fill seed 17 | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s17/` |
| fill seed 27 | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s27/` |
| fill seed 37 | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s37/` |
| fill seed 47 | `ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s47/` |

Each directory needs the files `loop.load_bundle` reads (`checkpoint.pt`, `normaliser.npz`, `recalibration.json`, `config.json`).
The clean panel the heads read, `cache/v8clean`, was rebuilt here in Stage 2 and passes Phase 22's tests. It can also be rebuilt
on the Mac in 0.3 min.

**Nothing is guessed.** The Phase 13 pre-rescue match (**1.129×**), the Phase 15 rescue-detection ceiling (**0.28**) and the
two-thirds of rescued weeks missed **stand as published, unrestated, on LEAKY inputs** until this re-run exists.

## One thing the Mac run must decide (deviation 249)

The brief says to re-run the harness **UNCHANGED** with clean predictions. Today `read_heads` **hard-codes** the leaky bundle
paths (`ml/artifacts/bundles/{task}/v8_…`) and returns early for any world other than `v8`. So the harness cannot point at the
clean bundles without a change. The smallest honest route is a **new-file wrapper** that:
- passes the clean bundle paths into the same `read_heads` body;
- keeps `order_policy.py@9e2d59d` pinned (deviation 122);
- keeps the 45/45 identity gate on the leaky run as its constructed check.

The Mac phase should name that wrapper and state that it changes only which bundle is read.
