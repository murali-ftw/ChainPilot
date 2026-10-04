# Phase 18 Stage 5 — global pulse

**Audience:** whoever decides whether a network-wide signal is worth adding to the models.
**Measured on:** v8 seed 1001, fixed split, TEST, 5 seeds, LightGBM proxy (frozen GBM), Apple M4 Pro CPU.
**Code:** `ml/data/pulse.py`, `ml/baselines/phase18_proxy.py` (fits at `9ce7007`; `ml/` identical to `94cfdc5`),
`ml/eval/phase18_score.py` (`67d9111`).
**Status:** complete. **Gate verdict: FAIL on every use case.** Worse or mixed, never a disjoint gain without a loss.

## The feature

Four network-wide statistics over the 4 weeks ending at t0, by **recorded** time, identical for every row of a
snapshot: mean days late of first receipts vs the current promise date; share of those receipts late; mean fill of lines
whose final receipt was recorded in the window; ordered quantity ÷ declared monthly capacity (pro-rated) over all
suppliers. Every source row is asserted `recorded_ts ≤ t0` (**1,879,069 rows**). The falsification fires when the
window is allowed to run 7 days past t0.

It tracks the regime as expected: late share 0.45–0.54 through 2020–2022 against 0.20–0.38 otherwise
(`ml/artifacts/phase18/pulse_v8.json`). SHUF = the snapshot → vector map permuted across snapshots, one permutation per seed.

## Gate

| use case · metric | BASE LightGBM-flat | **+ pulse** | + pulse, SHUFFLED | pulse vs BASE |
|---|---|---|---|---|
| arrival · lateness AUC | 0.7053 [0.7048, 0.7057] | 0.7034 [0.7031, 0.7039] | 0.7024 [0.6997, 0.7059] | **worse** |
| arrival · A3 (days) | 13.25 [13.21, 13.27] | 13.27 [13.24, 13.29] | 13.25 [13.18, 13.34] | undetermined |
| fill · exact CRPS | 0.1397 [0.1396, 0.1398] | 0.1400 [0.1399, 0.1402] | 0.1419 [0.1407, 0.1432] | **worse** |
| fill · P(full) AUC | 0.6049 [0.6047, 0.6053] | 0.6144 [0.6103, 0.6161] | 0.5721 [0.5496, 0.5827] | better |
| capacity · precision @ 1 / 5 / 10% | 0.780 / 0.692 / 0.662 | 0.718 / 0.658 / 0.663 | 0.655 / 0.556 / 0.543 | **worse / worse** / undet. |
| capacity · recall @ 0.70 / 0.80 / 0.85 | 0.148 / 0.022 / 0.013 | 0.257 / 0.051 / 0.024 | 0.177 / 0.081 / 0.029 | better / undet. / better |

| use case | **verdict** | why |
|---|---|---|
| arrival | **FAIL** | disjointly worse on lateness ROC-AUC |
| fill | **FAIL** | disjointly worse on exact CRPS (its AUC gain comes with a worse distribution) |
| capacity | **FAIL** | disjointly worse on precision at 1% and 5% coverage |

## Reading

- With 44 training snapshots (8 validation, 9 test), a per-snapshot constant gives the trees what amounts to a time
  index. The likely reading: the test year (2025) has no regime episode, so what the trees learn about 2020–2022 does
  not transfer. It costs ranking where it matters (the top of the capacity list, the lateness ranking) and moves the
  threshold-based recalls around.
- **The global latent the graph cannot reach (regime, Stage 1 (c)) is not worth a feature on this test year.** The useful
  network-wide signal is the forward one: the plan's network total in Stage 4 (diagnostic arm) helps where the trailing
  pulse does not.
