"""
Structural depth prior (Step 3/4, fixed readout) — `docs/10_AI_ML_Documentation.md`
§8.2, `project_HADES.md` §4.2-4.3. The learned depth gate (Step 6) is not
implemented here yet -- this module only covers the zero-parameter fixed
prior Steps 3-5 need.

Each prediction head reads a *specific* retained encoder layer, chosen from
graph connectivity (Step 2's measured reach table), not learned:

    delay     -> h^2   (2-hop reach)
    shortage  -> h^3   (3-hop reach)
    impact    -> h^3   (3-hop reach)

`project_HADES.md` §4.6: L=4 is deliberate even though the deepest prior is
h^3, so a gate (Step 6) has headroom to express upward deviation. Steps 3-5
don't build the gate, but the encoder still runs at L=4 for the baseline.
"""

from __future__ import annotations

STRUCTURAL_DEPTH_PRIOR = {"delay": 2, "shortage": 3, "impact": 3}

TASK_ENTITY_TYPE = {"delay": "Shipment", "shortage": "Product", "impact": "Supplier"}

TASKS = tuple(STRUCTURAL_DEPTH_PRIOR.keys())


def readout_layer_for_task(task: str, num_layers: int, shared_depth: int | None = None) -> int:
    """
    1-indexed encoder layer a given task's head reads from.

    `shared_depth=None` (the default, Step 3's baseline): each task reads
    its own structural-prior layer, clamped to `num_layers` in case the
    encoder is shallower than the prior calls for (e.g. the L-sweep's L=1,2
    runs).

    `shared_depth=L` (Step 4's L-sweep ablation): every task reads the
    *same* single layer L, regardless of task -- this is the "single
    final-layer readout" the structural prior is being compared against
    (`docs/14_Model_Development_Roadmap.md` §7: "does the structural depth
    prior actually improve on a single shared depth").
    """
    if shared_depth is not None:
        return min(shared_depth, num_layers)
    return min(STRUCTURAL_DEPTH_PRIOR[task], num_layers)
