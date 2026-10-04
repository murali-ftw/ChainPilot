# Phase 18 Stage 2 — oracle ceiling (PRIVILEGED)

**Audience:** whoever decides whether a use case is worth more modelling effort. **Every number here reads hidden
generator state. None of it is a model claim or a feature.**
**Measured on:** v8 seed 1001, fixed split, TEST, 5 seeds (7/17/27/37/47), Apple M4 Pro, CPU (LightGBM 4.7.0).
**Code:** `PRIVILEGED__regen_latents.py` (`94cfdc5`, verifier `9ce7007`), `PRIVILEGED__oracle.py` (`9ce7007`; the
no-timing tier `e967003`), `PRIVILEGED__score_oracle.py` (`36a9c3a` / `e967003`) → `PRIVILEGED__oracle_scores.json`.
**Status:** complete. **Verdict: REAL headroom on all three use cases.** The stop rule skips no use case.

## How the ceiling was measured

1. **Latents.** `_sim.npz` persists supplier state, regime, K, utilisation, ordered volume and every PO line's creation
   week. Transit and demand state (per destination plant) are not persisted. The generator was therefore **re-run
   unmodified** (its source exec'd with its own `__file__`, so its self-hash `71de78a` is unchanged) into a scratch
   directory. Its output was compared with the stored world: **56 / 56 generator files equal** (`PRIVILEGED__regen_check.json`).
   Three files carry a wall-clock write time and were compared by content without it. `level4.json` is written by the
   validator, not the generator (deviation 164). Only after that check were the latents saved.
2. **Labels re-derived.** Before any fit, every arrival, fill and capacity label was recomputed from the simulation
   and asserted equal (arrival = arrival week − t0; fill = delivered / ordered; capacity = mean monthly ordered / K,
   capped at 3). All equal.
3. **Fit.** Phase 7's LightGBM-flat arm unchanged (same rows, flat as-of features, frozen GBM, objective, early stopping),
   with privileged columns appended. Tiers (pre-registered, `reports/phase18-preregistration.md` def. 3):
   - **ORACLE** (the brief's; decides the verdict): arrival / fill add the line's creation week, supplier state,
     plant transit and demand state, regime and K at creation, **plus the supplier's load at creation** (previous month's
     ordered / K, effective capacity, ordered volume in the creation week and month, the line's quantity). Capacity adds
     **true K and the realised ordered volume** for each month of the label window.
   - **ORACLE-STATE**: the same minus every realised-ordering quantity.
   - **ORACLE-STATE without timing** (diagnostic, added after the first two were read; deviation 165): ORACLE-STATE
     minus the creation-week offset.

## The ceiling

Headroom = ORACLE − model (5-seed means), on the pre-registered metric. "Model" = the stored neural incumbent.

| use case · metric | model (neural) | LightGBM-flat | **ORACLE** | ORACLE-STATE | state, no timing | **headroom** | verdict |
|---|---|---|---|---|---|---|---|
| arrival · lateness ROC-AUC | 0.7090 [0.7064, 0.7130] | 0.7053 | **0.9385** [0.9384, 0.9387] | 0.9229 | **0.7092** | **+0.230** | **REAL** |
| arrival · A3 median abs err (days) | 13.06 | 13.25 | 5.08 | 5.63 | 13.24 | | |
| fill · P(fill = 1) ROC-AUC | 0.6204 [0.6200, 0.6211] | 0.6049 | **0.9184** [0.9182, 0.9186] | 0.7506 | 0.7502 | **+0.298** | **REAL** |
| fill · exact CRPS | 0.1388 | 0.1397 | 0.0773 | 0.1256 | 0.1257 | | |
| capacity · precision @ 5% coverage | 0.851 [0.832, 0.860] | 0.692 | **1.000** | 0.944 [0.939, 0.949] | n/a | **+0.149** | **REAL** |
| capacity · recall @ p = 0.85 | 0.192 | 0.013 | 0.998 | 0.246 | | | |

## What the ceiling is made of

- **Arrival: the whole ceiling is order timing.** Take away the creation week and the state-at-creation oracle scores
  **0.7092**, the same as the model (0.7090). `arrival_week` is counted from t0, so it contains the wait until the
  line is raised: 1–12 weeks, against a lognormal lead noise of σ 0.34 (Stage 1 (b)). Supplier state, transit,
  regime and K at creation add nothing the model does not already see. **The arrival lever is predicting when the next
  line will be raised, not more supplier state.**
- **Fill: two layers.** State alone (K, supplier state at creation, no load) is worth +0.13 AUC, and it does not depend
  on timing (0.7502 vs 0.7506). The realised load at creation (how much else the supplier was asked for that week and
  month) is worth another +0.17. Fill is a queue for the supplier's weekly capacity bank, so knowing the queue is what
  makes it predictable.
- **Capacity: the denominator.** With true K and realised volume the label is arithmetic (precision 1.000). With true
  K alone: 0.944. **K is drawn almost independently every month** (within-supplier autocorrelation 0.09; declared
  capacity tracks it at 0.11, Stage 1 (a)), so no as-of feature can recover that +0.09. The rest is forecastable
  volume (see the hindsight row below).

## Stage 4 control: hindsight load (the upper bound for forward load)

The realised ordered quantity per supplier and channel over weeks 1–4 / 5–8 / 9–13 after t0, i.e. what `fwd_load`
forecasts, known exactly. **Upper bound only, never a feature.**

| | LightGBM-flat | fwd_load (as-of) | **hindsight load** | share of the hindsight gain the as-of plan recovers |
|---|---|---|---|---|
| arrival lateness AUC | 0.7053 | 0.7186 | 0.7792 | 18% |
| fill P(full) AUC | 0.6049 | 0.6582 | 0.6885 | 64% |
| fill exact CRPS | 0.1397 | 0.1355 | 0.1326 | 59% |
| capacity precision @ 5% | 0.692 | 0.813 | 0.891 | 61% |
| capacity recall @ p = 0.70 | 0.148 | 0.443 | 0.674 | 56% |

## Inputs read

`db/gen_v8/seed_1001/_sim.npz` (persisted latents), the re-run's transit / demand state, the stored world's CSVs
(labels, flat features). The re-run verifier hashed every stored file, including `inventory_position_weekly.csv`, and
compared every `_sim.npz` array, including the simulator's part-plant position. That was an identity check, not a
feature (listed in `reports/phase18.md`).
