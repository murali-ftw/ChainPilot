# Phase 18 Stage 6 — lane / checkpoint sharing

**Audience:** whoever decides whether a lane or checkpoint node type is worth building.
**Measured on:** v8 seed 1001 (`db/gen_v8/seed_1001/logistics_lanes.csv`; `generator_v8.py` l. 339–340, 575, 1686–1690).
**Status:** **BLOCKED.** Nothing was fitted.

## Gate

The brief runs this stage only if Stage 1 (c) is non-empty and the lanes table exists. Pre-registered def. 8 adds that
the table must actually group channels.

| condition | measured | met |
|---|---|---|
| (c) shared latents with no edge | regime (global); part-plant stock / episode state | yes |
| lanes table exists | `logistics_lanes.csv`, 16,072 rows | yes |
| **a lane holds more than one channel** | `lane_id` L00000 … L16071, **one per channel** (`generator_v8.py` l. 1686: `range(NCH)`) | **no** |
| **a checkpoint groups channels** | `via_checkpoint` is **100% empty**. The generator writes `checkpoint_id`, the schema's column is `via_checkpoint`, and `emit` writes the schema's columns, so the value is dropped | **no** |

"The as-of mean trailing lateness over OTHER channels on the same lane / checkpoint" is the **empty set for every
row**. The feature cannot be built (deviation 161).

## What the generator says instead

- The transit latent is indexed by **destination plant** (7 values), not lane (Stage 1 (a), (b)). Plant already has a
  node and a channel–plant relation, so that sharing is already in the graph.
- `carrier_id` (79 values) and the would-be checkpoint (24 values) are independent random draws that **enter no
  outcome**. A lane / carrier / checkpoint node built on v8 would carry noise by construction.

**No evidence for a lane node type.** A checkpoint or carrier grouping would need a generator in which a shared transit
latent is indexed by it. That is a data change, out of scope here.
