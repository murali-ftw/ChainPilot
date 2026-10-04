# Phase 20 Stage 2 — do the Phase 19 gains replicate on a second world?

**Audience:** whoever decides whether the forward-plan and cadence gains are properties of the method or of one world.
**Measured on:** v8 seed 1001 (stored Phase 18–19 arms) and **v8w1002**, a world generated from v8 seed 1002 for this
stage. Fixed split; TEST; 5 seeds; LightGBM proxy only (frozen GBM, CPU). Each world has **its own BASE, its own lateness
reference and its own bands**; nothing is borrowed.
**Code:** `ml/data/phase20_world.py` (`f856c29` generate, `75e144a` features), `ml/baselines/phase20_proxy.py` (fits at
`d50e50f` / `50cd5bb`), `ml/eval/phase20_score.py` (`b54d55b`, n/a fix `c303519` → `ml/artifacts/phase20/scores.json`),
`ml/eval/phase20_baserates.py` (`d1e69ac` → `baserates.json`).
**Status:** complete. **5 of 6 passing cells replicate on at least one primary metric; fill cadence does not** (P3 wrong).
Gain ratios are often outside ×2 (P4 wrong).

## (a) The world

| | |
|---|---|
| generator | `db/gen_v8/generator_v8.py` **unmodified**, exec'd with its own `__file__`; self-hash **71de78afa645** (= the manifest's `code_commit`) |
| command | `--seed 1002 --out data_worlds/v8_seed1002` (outside `db/`; `data_worlds/` added to `.gitignore`) |
| `git status -- db` after generation | **empty** |
| key tables (SHA-1) | po_lines `460ad43d…`, grn_lines `832c5ebd…`, training_labels `e27781a5…`, channel_performance_weekly `014409ef…`, part_demand_weekly `b695cd0b…`, sourcing_channels `62c2efe0…`, inventory_transactions `1a8b1ef5…` |
| vs the stored `db/gen_v8/seed_1002` | **byte-identical** on every key table (the manifest differs only in its timestamp). The generator is deterministic in its seed, so "fresh" here means *not used by Phases 18–19*, not "never generated" (deviation 180) |
| registration | `config.WORLDS["v8w1002"]` set **in-process** by the new wrapper; no loader edited. Panel width measured 15 and asserted |

## (b) Base rates (each world's own as-of reference; reference offset c = 4.57 wk v8, 4.79 wk world 2)

| test fold | v8-1001 | v8w1002 | difference | flag |
|---|---|---|---|---|
| arrival late share (uncensored, as-of reference) | 0.475 | 0.467 | −0.9 pt | |
| arrival censored share | 0.485 | 0.441 | −4.4 pt | |
| **P(fill = 1)** | 0.749 | **0.841** | **+9.1 pt** | **FLAGGED** |
| fill < 0.95 | 0.245 | 0.154 | **−9.1 pt** | **FLAGGED** |
| **capacity strain > 1** | 0.405 | **0.199** | **−20.7 pt** | **FLAGGED** |

World 2's capacity problem is half as common, and its fill is mostly complete. Its absolute precision and lift numbers are
**not comparable** with v8-1001's. Only same-world, own-BASE comparisons are read below.

## (c)–(d) Gate v2 on the second world, and the replication verdict

Replicates = same sign **and** disjoint 5-seed band against **world 2's own BASE**, on each metric that passed in v8-1001.
Gain = mean improvement over BASE (positive = better); ratio = world 2 ÷ v8-1001.

| cell (passed Gate v2 in v8-1001) | metric | v8-1001 gain | **world-2 gain** | world-2 vs its BASE | ratio | ×2? | world-2 Gate v2 |
|---|---|---|---|---|---|---|---|
| arrival · fwd_season | lateness AUC | +0.0014 | +0.0035 | **REPLICATES** | 2.6 | no | PASS |
| | A3 (days) | −0.18 | −0.16 | **REPLICATES** | 0.92 | yes | |
| arrival · fwd_load (supplier-specific) | lateness AUC | +0.0133 | +0.0131 | **REPLICATES** | 0.99 | yes | PASS |
| arrival · cadence | lateness AUC | +0.0089 | +0.0055 | **REPLICATES** | 0.62 | yes | PASS |
| | A3 (days) | −0.16 | −0.001 | UNDETERMINED | 0.01 | no | |
| fill · fwd_season | exact CRPS | −0.0028 | −0.0009 | **REPLICATES** | 0.32 | no | PASS |
| | P(full) AUC | +0.036 | +0.027 | **REPLICATES** | 0.74 | yes | |
| **fill · cadence** | exact CRPS | −0.0002 | −0.00003 | **UNDETERMINED** | 0.29 | no | **FAIL** |
| | P(full) AUC | +0.0018 | +0.0011 | **UNDETERMINED** | 0.62 | yes | |
| capacity · fwd_load (supplier-specific) | precision @ 1% | +0.098 | +0.136 | **REPLICATES** | 1.39 | yes | PASS |
| | precision @ 5% | +0.121 | +0.179 | **REPLICATES** | 1.48 | yes | |
| | precision @ 10% | +0.134 | +0.151 | **REPLICATES** | 1.13 | yes | |
| | recall @ p 0.70 | +0.294 | +0.113 | **REPLICATES** | 0.38 | no | |
| | recall @ p 0.80 | +0.204 | — | UNDETERMINED (world-2 BASE unreachable on 1 seed) | | | |
| | recall @ p 0.85 | — | — | n/a (v8 BASE unreachable on 1 seed) | | | |

The supplier-specific reference comparison (fwd_load vs fwd_season), Phase 19's gate, also passes on world 2 for arrival
and capacity, and fails for fill there too, on the same control (the snapshot-permuted fwd_load beats BASE). The fill
supplier-specific failure therefore replicates as well.

## The Phase 19 blend recipe on LightGBM arms only

No neural incumbent exists on world 2, so **the neural half of the Phase 19 blend is not replicated**. The recipe is
re-fitted with LightGBM arms only: the LightGBM arm carrying the families the Phase 19 neural arm carried (`lgbm_rf`),
blended with LightGBM + fwd_load, weight fitted on each world's validation, snapshot-block bootstrap on test.

| world · use case | weight on `lgbm_rf` | lgbm_rf / LightGBM + fwd_load / blend | blend vs lgbm_rf | blend vs LightGBM + fwd_load |
|---|---|---|---|---|
| v8 · arrival (lateness AUC) | 0.88 | 0.7218 / 0.7192 / 0.7219 | undetermined | better (+0.0029 [0.0013, 0.0046]) |
| world 2 · arrival | **1.00** | 0.6994 / 0.6972 / 0.6994 | identical | undetermined |
| v8 · fill (CRPS) | 0.49 | 0.1366 / 0.1354 / 0.1353 | better | undetermined |
| world 2 · fill | 0.20 | 0.0932 / 0.0923 / 0.0922 | better | undetermined |
| v8 · capacity (P @ 5%) | 0.00 | identical arms (capacity's `lgbm_rf` is fwd_load) | — | — |
| world 2 · capacity | 0.08 | 0.671 / 0.671 / 0.671 | identical | identical |

With LightGBM alone, the blend is never better than its best parent. **The Phase 19 blend's gain comes from mixing a
neural model with a tree model**, and that half cannot be tested on world 2 without neural training (CPU-only phase).

## Reading

- **Robust across both worlds:** supplier-specific forward load on arrival (the gain is the same size: ×0.99) and on
  capacity (larger on world 2 at the top of the list); the forward season on arrival and fill; cadence on arrival lateness.
- **One world only:** cadence on fill (it was a ±0.0002 CRPS effect in v8, and vanishes on world 2), and cadence's A3 gain on
  arrival.
- **Magnitudes move.** 5 of 13 comparable gains fall outside ×2 of v8's: the season's arrival gain is 2.6× larger, its
  fill CRPS gain 0.32×, capacity's recall gain 0.38× (on a world where strain > 1 is half as common). Directions hold;
  sizes do not transfer.
