# Phase 18 Stage 1 — generator forensics (read-only)

**Audience:** whoever decides which information fix is worth building for arrival, fill and capacity.
**Measured on:** `db/gen_v8/generator_v8.py` (SHA-1 `71de78afa645`, equal to every v8 manifest's `code_commit`) and the
stored v8 seed 1001 world. Nothing under `db/` was modified. The quantitative lines read `db/gen_v8/seed_1001/_sim.npz`.
**Companions:** `reports/phase18-preregistration.md`, `reports/part1/phase-9b.md` §A (the earlier lead-time notes).
**Status:** complete. **Gates:** Stage 4 **OPEN** (a versioned part-plant-week requirement table exists).
Stage 6 **BLOCKED**: lanes are one per channel and `via_checkpoint` is 100% empty, so no lane or checkpoint groups two channels.

Line numbers below refer to `generator_v8.py`.

## (a) Every latent: what indexes it, which outcomes it enters

The three labels (l. 1460–1494), each built forward from snapshot week t0 over 90 days (t1 = t0 + 12 weeks):
- **arrival_week** = `pa − t0` for a PO line **created after t0** (`pt ∈ (t0, t1]`), right-censored if `pa > t1`.
- **fill_rate** = `delivered / ordered` for the same lines. Delivery is decided **at placement** (l. 527–562).
- **capacity_strain** = mean over the months spanning [t0, t1] of `util[m, supplier] = ordered[m, s] / K[m, s]`, capped at 3.

Every label is about a line or month that **does not exist yet at t0**. Arrival and fill depend on the state at the
line's **creation week**, 1–12 weeks after t0. That is hypothesis (a) of the brief, confirmed by construction.

| latent | indexed by | update | arrival | fill | capacity | where |
|---|---|---|---|---|---|---|
| `sup_state` x | **supplier** × week | AR(1) 0.93, innov 0.10, plus supply shocks (rate 0.0055/wk, decay 0.86) | **yes**: lead × (1 + 0.55·x⁺ …) | **yes**: refusal prob 0.006·(1 + 3.2·x⁺) → line delivers 0; through K | **yes**: K[m] × (1 − 0.30·tanh(mean_m x⁺)) | l. 331–338, 349, 566, 574 |
| `transit_state` | **destination plant** × week (7 plants) | shock rate 0.004/wk, \|N(1, 0.4)\|, decay 0.80 | **yes**: + 0.9·transit[plant] | no | no | l. 339–340, 575 |
| `dem_state` | **destination plant** × week | shock rate 0.0035, N(0.55, 0.30), decay 0.88 | no (only through load) | through load | **yes**: scales channel demand → ordered volume | l. 341–342, 508 |
| `regime` | **global** × week | calendar: 1.0 Mar–Sep 2020, 0.55 Apr 2021–Jun 2022 | **yes**: + 0.55·regime | through K | **yes**: K × (1 − 0.22·tanh(regime)) | l. 344–347, 350, 577 |
| `load_prev` | **supplier** × month | `util` of the previous calendar month = ordered / **true** K | **yes**: + 0.85·clip(load_prev − 0.70, 0, 2) | **yes**: effective capacity `remK = K·(1 − 0.18·clip(load_prev − 1, 0, 1.5))` | no (the label uses K, not remK) | l. 476, 479–480, 576 |
| capacity **K** | **supplier** × month | **drawn i.i.d. lognormal(σ 0.40) every month**, then scaled by state and regime; floor 60 | through load_prev and shortfall | **yes**: the weekly bank `remK / weeks` caps delivery | **yes**: the label's denominator | l. 345–352 |
| declared capacity DECL | supplier × month | K of the refresh month × lognormal(0.14, 0.19), held 12 months | no | no | no — **observable only** | l. 355–360 |
| fulfilment propensity | **channel** (PRIO_CH, static uniform) + channel `starve` + **part-plant** `ls_recent` + channel cover | queue priority inside the supplier's weekly bank; ration reserve 16% split pro rata | via shortfall | **yes** | no | l. 389, 536–556 |
| line shortfall | **PO line** | 1 − delivered / ordered | **yes**: lead × (1 + 0.85·shortfall) | it *is* fill | no | l. 585–586 |
| allocation share | **part-plant** siblings | quarterly review on trailing performance | through load | through load | **yes**: moves ordered volume between suppliers | l. 482–503, 510–516 |
| demand level `rate`, AR(1) `_rate_eps` | channel | persistent level + AR(0.97) | — | through orders | **yes**: ordered volume | l. 367–369, 505–507 |

### Measured from `_sim.npz` (seed 1001)

| quantity | value | what it means |
|---|---|---|
| within-supplier correlation of log K, month m vs m − 1 | **0.094** | K is close to **independent month to month** |
| within-supplier correlation of log DECL vs log K, same month | **0.110** | the declared figure (the only published capacity) barely tracks K |
| share of within-supplier variance of log util explained by log K | **0.42** | about two-fifths of strain's movement is a draw nothing at t0 records |
| correlation of util (ordered/K) with util_obs (ordered/DECL, the panel's `load_ratio`) | **0.50** | the panel's load column is a noisy reading of the true load |
| within-supplier autocorrelation, log util / log ordered, month to month | 0.17 / 0.27 | ordered volume is the more persistent, more forecastable half |

**Consequence for capacity:** the label is ordered volume ÷ a monthly i.i.d. draw. Forward requirements can only reach
the numerator. The denominator's 0.40-σ monthly draw is invisible at t0, and the oracle (which is given K) will
measure how much that costs.

## (b) v8's arrival lead time vs the earlier notes

The earlier notes (`reports/part1/phase-9b.md` §A.1, written from the v6/v7 generators) give
`lead = exp(N(log(max(0.51·contracted, 3)), 0.34)) × [1 + 0.55·x⁺ + 0.9·transit + 0.85·clip(load_prev − 0.70, 0, 2) + 0.55·regime]`,
`arrival_week = created_week + ceil(lead / 7)`.

v8 (l. 572–590):
```
lead  = exp(N(log(max(0.51·contracted, 3)), 0.34))                       # lead_base_frac 0.51, lead_sigma 0.34
lead *= 1 + 0.55·max(x[t, s], 0) + 0.9·transit[t, plant] + 0.85·clip(load_prev[s] − 0.70, 0, 2) + 0.55·regime[t]
lead *= 1 + 0.85·clip(1 − delivered/ordered, 0, 1)                        # NEW in v8 (§16 probe 3)
lead  = max(3, lead);  arrival_week = created_week + ceil(lead / 7)
then: an expedite can pull a line's arrival earlier (1a, l. 703–737)      # NEW in v8
```

| term | notes | v8 | match |
|---|---|---|---|
| lognormal noise σ | 0.34 | 0.34 (`lead_sigma`) | **yes** |
| load term | 0.85·clip(load_prev − 0.70, 0, 2) | same (`congestion_beta` 0.85) | **yes** |
| load_prev definition | "supplier utilisation" | previous **calendar month's** ordered / **true K**, not the declared figure the panel shows | detail the notes did not state |
| supply, transit, regime terms | 0.55 / 0.9 / 0.55 | same | **yes** |
| transit index | "lane transit state" | **destination plant** (7 values), not lane | **differs** |
| shortfall multiplier | absent | × (1 + 0.85·shortfall): a refused line (shortfall 1) takes 1.85× as long | **new** |
| expedites | absent | 1.30% of lines (13,866) moved earlier, compression Beta(2, 3) × 0.55 / 0.28 | **new** |
| floor | max(base, 3) inside the log | also `max(3, lead)` after all multipliers | minor |

So the irreducible part per line is the lognormal draw (σ 0.34 on the log lead, about ±40% at 1σ). The rest of the
bracket is state that exists at creation time, not at t0.

## (c) Latents shared across channels with no edge in the current graph

Graph (`ml/data/loader.py:build_graph`): node types channel, supplier, part, plant. Core relations channel–supplier,
channel–part and channel–plant, each in both directions (R = 6).

| shared latent | shared by | edge? |
|---|---|---|
| sup_state, K, load_prev | channels of one supplier | **yes**: supplier node |
| transit_state, dem_state | channels into one plant | **yes**: plant node |
| **regime** | **every channel** (global) | **no**: no global node |
| **part-plant stock / episode state** (`ls_recent`, shortage episodes, allocation share, alternate-source boost) | channels feeding one part × plant | **no**: part and plant are separate nodes. A channel reaches its part-plant siblings only through the part node (all plants mixed) or the plant node (all parts mixed) |

**(c) is non-empty:** regime (global) and part-plant state. Stage 5's global pulse targets the first.

## (d) Lanes, checkpoints, the plan table

| table | present | grain / content | as-of reconstructable |
|---|---|---|---|
| `logistics_lanes` | yes, 16,072 rows | **one lane per channel** (`lane_id` L00000 to L16071, 1:1 with channel), origin site = the supplier's single site, destination plant, carrier (79 values) | n/a. `carrier_id` is a random draw entering no outcome (l. 1686–1690) |
| `via_checkpoint` | column present | **100% empty**: the generator writes `checkpoint_id`, the schema names the column `via_checkpoint`, and `emit` writes the schema's column, so every value is null | n/a |
| `part_demand_weekly` | yes, 3.67 M rows | **part × plant × week** gross requirement P50 / P90; a version every **8 weeks** from 2019-01-07 (`as_of_date`), horizon 12–26 weeks, `recorded_ts` = as_of + 0–2 days. Its truth is the demand the simulation actually issued (`PP_DEM`, l. 1405) | **yes**: rows with `recorded_ts ≤ t0`, latest `as_of_date` per (part, plant, week) |
| `production_plan` | yes, 313 k rows | **product × plant × week**, a version every 4 weeks with `plan_version`, `recorded_ts` | yes, but **it does not drive channel demand**: the plan's `BUILD` feeds only `production_plan` / `production_actual` / `plan_drift_features` (l. 1301–1368). Channel demand is the per-channel rate process (l. 505–516) |
| `bom` | yes, 4,000 rows | product → part (`qty_per_unit`, `scrap_factor`) | static |

**Stage 4 gate: OPEN.** The requirement table at part-plant-week grain is `part_demand_weekly`. It is versioned with
`recorded_ts`, so the plan as of any t0 ≥ 2019-01-07 can be reconstructed. Before that date no version exists, and
the features are missing. BOM-exploding `production_plan` would carry only the shared seasonality and trend, not the
demand that becomes orders, so Stage 4 uses `part_demand_weekly`, which is already at part-plant grain (deviation 160).

**Stage 6 gate: BLOCKED.** (c) is non-empty, and the lanes table exists, but it groups nothing: every lane holds
exactly one channel, and the checkpoint column is empty. "Other channels on the same lane / checkpoint" is the empty
set for every row. The generator's transit latent is per destination plant, which already has a node (deviation 161).

## (e) Persisted or re-run?

| latent | persisted in `_sim.npz` | otherwise |
|---|---|---|
| sup_state [week, supplier], regime [week] | **yes** | |
| K, util (= ordered/K), ordered, delivered, DECL, util_obs [month, supplier] | **yes** | |
| per PO line: creation week `pt`, channel, qty, delivered, lead `pl`, arrival week `pa`, expedited flag, promise offset | **yes** (`pol_id` = `POL{index:09d}`, so labels join by index) | |
| allocation share history, revisions, inventory movements, part-plant weekly position | **yes** | |
| **transit_state [week, plant], dem_state [week, plant]** | **no** | re-run the generator on a scratch copy with seed 1001 (Stage 2) |
| channel rate, PRIO_CH, starve, the weekly bank | no | not needed by the oracle tiers pre-registered |

Stage 2 therefore joins `_sim.npz` directly and re-runs the generator only for transit and demand state. The re-run
exec's the unmodified source (so its self-hash is unchanged) into a scratch directory and must reproduce every stored
file before any latent is read.
