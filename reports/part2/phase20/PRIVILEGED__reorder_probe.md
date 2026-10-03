# PRIVILEGED — Phase 20 Stage 4b reorder-trigger probe

**Every number here reads hidden simulator state. None of it is a model claim or a feature.**
**Measured on:** v8 seed 1001, fixed split, TEST, arrival, 5 seeds, LightGBM (frozen GBM, CPU).
**Code:** `PRIVILEGED__reorder_probe.py` (`ab9c919`): capture → `PRIVILEGED__probe_capture.json`, fit →
`PRIVILEGED__reorder_probe_fit.json`; scored by `PRIVILEGED__score.py` (`a98f2b9`) → `PRIVILEGED__scores.json`.

## How the state was captured

`generator_v8.py`'s source was exec'd with **one line added in memory** at the top of the weekly loop: a hook that copies
each channel's `on_hand`, `on_order`, `rop` (reorder point) and `dhat` (planner forecast). It copies arrays and draws no
random number. A stop is raised right after `_sim.npz` is written. The file on disk is untouched; its self-hash
(`71de78afa645`) is what the generator stamps. **Acceptance: all 48 arrays of the hooked run's `_sim.npz` equal the
stored world's.** 83 snapshot weeks captured, in 20.5 s.

Probe columns, end of each snapshot week: (position − reorder point) ÷ planner forecast; position ÷ reorder point;
planner forecast; weeks to the channel's next review.

## Result

| arm | lateness ROC-AUC | A3 (days) |
|---|---|---|
| BASE | 0.7053 [0.7048, 0.7057] | 13.25 [13.21, 13.27] |
| **BASE + true position vs reorder point** | **0.7394** [0.7390, 0.7397] | **12.71** [12.67, 12.75] |
| BASE + true creation week (Phase 19) | 0.9104 [0.9102, 0.9105] | 6.42 |

**Probe recovery share = 16.6%** of the timing headroom (P7: ≥ 50% and ≥ 0.81 lateness AUC; **not met**).

**Caveat (carried with the number):** the simulator's internal position is exact and instantly current. A real ERP's book
position lags and carries recording error, so 0.739 is an **upper bound** on what client stock and reorder-point data
could be worth.
