# Phase 13 — pre-registered predictions

**Committed before any Phase 13 measurement ran** (Stage -1, P4). This file is not edited afterwards. Each
prediction is marked right or wrong in `reports/part2/phase-13.md` §4.

Predictions, verbatim from the Phase 13 brief (STAGE 0.7):

1. **F1:** the three-part Beta head will NOT clearly beat a cheaper fix (widened top atom / RPS reweighted at
   the near-1 boundary), because the 22-cell head ALREADY has explicit atoms at 0 and 1.0, so the [0.95,1)
   over-prediction at 2.9x frequency is a loss/boundary problem, not a smoothness problem.
2. **F2:** (part,supplier) beats raw triple; the shrunk hierarchy beats both; (supplier,plant) is closer than
   expected and may beat (part,supplier) on cold-start specifically (lane hypothesis — C3 found the part
   relation dead and supplier/plant live).
3. **F2 arm (c)** will look best on validation and worst on test.
4. **ratio_of_sums** will beat **mean_of_ratios**.
5. **S1:** ratio lands in [1.0, 1.2].
6. **S2:** beats naive nearest-surplus on agreement-with-planners, and this is NOT evidence of shortage avoided.

Author's additional predictions (Claude, recorded at the same time, so they are scored too):

7. **Gate 0.3 closes.** observed shortfall (91) + net transfers will be far from 298 in the units that matter,
   because 298 is a mean shortfall *when short* and transfers change *which* weeks are short, not only depth —
   S1 and S2 are likely cancelled by arithmetic.
8. **Gate 0.1 closes F2's zero-receipt definition:** deviation 100 found zero-delivery closures write no GRN row;
   the PO status field will not separate "closed, nothing arrived" from "still open".
9. **F1 arm 5 (boundary fix) removes most of the [0.95,1) excess but not the low-fill cells [0.05, 0.25)**,
   which A2 showed carry the other half of the interior error.
