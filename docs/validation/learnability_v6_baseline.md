# Baseline learnability — `gen_v6/seed_1001`, measured before any run-7 change

Written before Part 1 so this baseline exists independently of whatever v7 does.
Harness: [learnability.py](learnability.py). Raw log: [learn_v6_1001.log](learn_v6_1001.log).

## Split — fixed, time-ordered, identical for every task and gate

    train   snapshot_date <= 2023-12-31
    val     2024
    test    2025            <- every number below is test-fold only

Features are built **as-of** the snapshot: the latest `channel_performance_weekly` row with
`week_start <= snapshot_date`, joined by `channel_id` (`merge_asof`, backward). Nothing after
`t0` can enter a feature.

| Task | entity | train / val / test rows |
|---|---|---|
| fill_rate | po_line | 263,985 / 32,000 / 36,000 |
| arrival_week | po_line | 263,985 / 32,000 / 36,000 (43.0% censored) |
| shortage_qty | part_plant | 277,989 / 33,696 / 37,908 |
| demand_drift | product_plant | 55,800 / 7,200 / 8,100 |

---

## Verdict in three lines

**G2 computed. G3 passes on three of four tasks — and loses to the best naive baseline on
part shortage.**
**G4 fails: the graph contributes ~0% of above-chance signal on arrival, −1.9% on shortage, and
at most ~20% of a very small gain on fill. h⁰ ≈ h⁴.**
A separate defect: **the `shortage_qty` label table is positive-only** — 45,942 rows, zero of them
zero-valued — so the shortage head as shipped has no negative class.

---

## G2 — the numbers to beat

| Task | Baseline | Metric | Value |
|---|---|---|---|
| **fill_rate** | global mean (empirical CDF) | CRPS | 0.0633 |
| | **per-channel historical mean** | **CRPS** | **0.0616** ← best |
| | last observed value | CRPS | 0.1055 |
| **arrival_week** | per-lane (supplier→plant) median | C-index | 0.5227 |
| | **per-channel median** | **C-index** | **0.5573** ← best |
| | global median | C-index | 0.5000 |
| **shortage_qty** | marginal base rate | PR-AUC | 0.2073 (= base rate) |
| | **per-part-plant historical rate** | **PR-AUC** | **0.5517** ← best |
| | median log-qty (positives only) | MAE | 1.2770 |
| **demand_drift** | naive seasonal (same week last year) | MAE | 0.2136 |
| | last value | MAE | 0.1892 |
| | **global mean** | **MAE** | **0.1533** ← best |

## G3 — LightGBM on tabular features only. **Passes on all four.**

| Task | Metric | G2 best | G3 | Improvement |
|---|---|---|---|---|
| fill_rate | CRPS ↓ | 0.0616 | **0.0598** | 2.9% |
| arrival_week | C-index ↑ | 0.5573 | **0.6444** | above-chance 0.0573 → 0.1444, **2.5×** |
| shortage_qty | PR-AUC ↑ | 0.5517 | **0.5457** | **−1.1% — G3 LOSES** |
| shortage_qty (positives) | MAE ↓ | 1.2770 | **1.1187** | 12.4% |
| demand_drift | MAE ↓ | 0.1533 | **0.0986** | **35.7%** |

A GBM finds signal the naive baselines miss on fill, arrival and drift. **On part shortage it
does not**: the per-part-plant historical rate (PR-AUC 0.5517) beats the GBM on the full feature
set (0.5457). In a world where shortage is dominated by a persistent per-channel demand draw, a
part-plant's own history is close to sufficient and the observable features add nothing. That is a
property of the world, not of the model.

### Leak check — top feature importances

| Task | Top 5 features (share of total gain) |
|---|---|
| fill_rate | `load_ratio` 10.4%, `si` 10.3%, `pi` 9.3%, `reporting_lag_days` 8.9%, `transport_distance_km` 8.9% |
| arrival_week | `load_ratio` 10.0%, `lead_time_actual_days` 9.8%, `lead_time_ratio` 8.4%, `reporting_lag_days` 8.1%, `contracted_lead_time_days` 7.4% |
| shortage_qty | `load_ratio_mean` 7.0%, `lead_time_actual_days_mean` 6.9%, `lead_time_ratio_mean` 6.6%, `n_suppliers` 6.0%, `qty_ordered_mean` 5.8% |
| demand_drift | `seasonal_naive` 24.5%, `woy` 24.2%, `last_value` 23.6%, `kc` 22.9%, `month` 4.8% |

**No single feature dominates implausibly.** The heaviest is 24.5% on drift, and the four heavy
drift features are the naive baselines themselves plus calendar position — expected, not a leak.
Every top feature on the three GNN tasks is an as-of aggregate from `channel_performance_weekly`,
whose provenance was verified in run 6 (`derived store buckets on visible week: max(event_week,
recorded_week)`, 100.0% on disagreeing cells). Nothing post-`t0` appears.

---

## G4 — the h⁰ diagnostic. **This is the decisive result, and it fails.**

Method: heterogeneous message passing over channel ↔ {supplier, part, plant}. One *round* is
channels → entity means → back to channels; a 4-layer encoder is two rounds. Aggregates are
concatenated to the channel features and the **same** GBM head is refitted, so the only thing that
changes between rows is what the graph adds.

| Task | metric | G2 | **h⁰** | **h¹** | **h⁴** | graph share of above-chance signal |
|---|---|---|---|---|---|---|
| fill_rate | CRPS ↓ | 0.0616 | 0.0598 | **0.0593** | 0.0594 | **+22%** of a 0.0023 gain |
| arrival_week | C-index ↑ | 0.5573 | 0.6444 | 0.6439 | 0.6440 | **−0.3%** |
| shortage_qty | PR-AUC ↑ | 0.5517 | 0.5457 | 0.5400 | 0.5385 | **−2.2%** |

Above-chance signal is measured against the G2 baseline for fill, against 0.5 for arrival's
C-index, and against the 0.2073 base rate for shortage's PR-AUC.

**Reading it plainly.** On arrival — the task HADES v3 measured the graph at ~10% of above-chance
signal — the graph here contributes **nothing**: h⁴ is 0.0004 C-index *below* h⁰, which is noise.
On shortage the graph is actively harmful, costing 1.9% of above-chance signal, consistent with
neighbourhood means diluting a signal that is already thin. Only fill shows a gain, and it is 22%
of an improvement that is itself only 2.9% over the naive baseline — 0.0023 CRPS in absolute terms.

**h⁰ ≈ h⁴ on two of three GNN tasks.** Run 6 established the propagation arc is real but
−0.195/−0.240 rather than run 5's −0.811 artifact. This measurement answers the question run 6
left open: **an arc of that strength does not produce a measurable h⁴ − h⁰ gap.** The architecture
premise is unsupported on this data as it stands.

### What this measurement is, and is not

The ablation uses fixed mean-pooling message passing with a GBM head, not a learned GNN. A learned
encoder could weight neighbours and might extract more than a mean can. But the h⁰ diagnostic's
logic holds in the direction that matters: if a channel's supplier-, part- and plant-neighbourhood
carried information about its own future fill or arrival, a GBM given those aggregates should find
*some* of it. It finds none on arrival and negative on shortage.

---

## Defect found: the shortage label table has no negative class

    shortage_qty label rows        45,942
    rows with label_value == 0          0
    distinct part_plant entities    3,406

The generator samples label rows only for part-plants that experienced a shortage condition inside
the forward window, so every row is a positive. Trained as shipped, PR-AUC is degenerate at 1.000
and the head learns nothing. §9 states the shortage head's operating base rate is ~3%, which
requires ~97% negatives.

Every shortage number above is therefore measured on a **reconstructed** universe — all 4,340
part×plant pairs × 83 snapshots = 350,841 cells, positive where the label table has a row — giving
a natural base rate of 0.1309 overall and 0.1982 on the test fold. That reconstruction is a
measurement device, not a fix; the generator should emit the negatives itself.

---

## Ladder status after Part 0

| Gate | Status |
|---|---|
| G0 | **pass** (run 6: conservation exact, censoring non-zero on all five tasks) |
| G1 | **pass** (run 6: median channel degree 3, one component, 100% coverage) |
| **G2** | **pass — baselines computed above** |
| **G3** | **partial — LightGBM beats the naive baselines on fill, arrival and drift; loses on part shortage** |
| **G4** | **FAIL — h⁰ ≈ h⁴; graph share ≈ 0% on arrival, −1.9% on shortage** |
| G5–G7 | not yet run; G4 blocks the ladder |


---

## Correction, and the harness iterations behind these numbers

The shortage figures above are the **third** measurement of that task, and the first two were
wrong for reasons worth recording:

1. **Run 1 of the harness** binarised the label table directly. Because the table is
   positive-only, every row was a positive, the base rate was 1.000 and PR-AUC was degenerate at
   1.000. That is what exposed the label defect.
2. **Run 2** reconstructed the part×plant universe and defined a positive as "the label table has
   a row for this cell". That is correct for v6, whose table is positive-only, but it is not
   comparable to any run whose table samples the universe, and it inherits v6's 1,500-rows-per-
   snapshot sampling cap, which silently relabels true positives beyond the cap as negatives.
3. **Run 3 — the numbers above** derive the positives from the simulation's own shortage-condition
   series in `_sim.npz`: a cell is positive if that part×plant had a condition inside the snapshot's
   forward window. This is ground truth, is independent of how labels were sampled, and is
   therefore comparable across generator versions.

Base rate moved 1.000 → 0.1982 → 0.2073 across the three, and the G3 verdict on shortage moved
from meaningless → "passes" → **"loses to the naive baseline"**. Fill, arrival and drift are
unaffected; their numbers were identical in all three runs.
