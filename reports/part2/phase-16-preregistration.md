# Phase 16 — pre-registration

Committed before any Stage 0 measurement and before any Phase 16 training. Branch `phase16-encoders`, off `90a38ed`.
Not to be edited after this commit. Each prediction is marked right or wrong in `phase-16.md`, including the wrong ones.

## Predictions (verbatim from the brief)

**P1.** Channel nodes (median degree 3) show attention entropy near zero on the majority of reaching paths — attention
at the readout is a structural no-op. Supplier nodes (median degree ~38) do not.

**P2.** The part relation's normalised effective message share is the lowest of the three entity relations, consistent
with C3's ablation.

**P3.** Candidate A's effect on arrival will be SMALL OR NULL, because attention is a no-op at the readout and the
trajectory term can only act at supplier and plant nodes.

**P4.** Candidate B will move plant-node aggregation more than channel-node aggregation, because averaging 2,285
neighbours destroys more information than averaging 3.

**P5.** Neither variant beats its incumbent disjointly at 5 seeds. (Recorded so that a win is a surprise, not a
confirmation.)

## Code facts on record before measurement (read from source, not measured)

These are recorded now so that nothing in the results can be read as having been discovered after the fact.

1. `ml/models/share.py` `SHARELayer.forward` takes the softmax over each destination node's **whole** incoming
   neighbourhood across all relations (relation-blind), not per relation. A channel node's neighbourhood is exactly
   three edges (one supplier, one part, one plant), so its softmax has three terms and is **not** structurally
   deterministic. Whether its entropy is near zero is therefore an empirical question — which is what P1 asks.
   Per relation, a channel has exactly one neighbour (|N_r(i)| = 1).
2. Consequence for B-2 (low-degree bypass): per relation, every channel has |N_r| = 1, and entity nodes have
   |N_r| = their member count. If the smallest entity has more than 2 members, k = 1 and k = 2 select the identical
   edge set, and the k-selection is degenerate.
3. `HeteroMP` (capacity) has no softmax. A channel receives exactly one message per relation (a gather of its entity's
   aggregate). A "bypass of the softmax" is therefore the identity on capacity's forward pass.
4. The TCN has no per-dilation module. Per-depth states are reachable by forward hooks on `res[l]` (input = block
   state z_l, l = 1..5) and on `norm` (input = z_6), with no edit to `ml/models/tcn.py`.
5. The brief's "shipped 1.25e-4 throughout" is fill's learning rate. Arrival and capacity ship at 2.5e-4
   (`ml/configs/shipped.json`); "no learning rate retuned" is honoured by keeping each task's shipped rate.
