# HADES v4 / ChainPilot — Benchmark Specification

**Architecture, metrics and acceptance gates for the Rane supply-chain models.**

Source material: `db/model_plan.md` (architecture), `db/data_plan.md` (feature maths),
`db/dataset_structure.md` (schema). Every count, column name and range in this document was
measured from `db/csv_full_seed1/`, the dataset that actually exists, not from the planning
documents. Where the two disagree, see §11.

Dataset under specification:

| | |
|---|---|
| Path | `db/csv_full_seed1/` (raw), `db/csv_full_seed1/derived/` (Group I) |
| Span | 2016-01-01 → 2025-12-31 |
| Modelling window | 2019-01-01 → 2025-12-31 (`dataset_structure.md` §1) |
| `dataset_version` | `hades-v4-synth-1.0` |
| `feature_spec_version` / `label_version` | `1.0` / `1.0` |
| Size | 738 MB gzipped, 45,211,048 rows across 49 tables |

---

## 1. Scope and tasks

Ten models cover the nine use cases. **Four are learned. Three of those four share one
encoder.** Everything else is linear algebra, a stochastic simulation, or a solver.

| # | Model | Use case | Class | Learned? |
|---|---|---|---|---|
| 1 | BOM explosion | UC1, UC6 | Sparse linear algebra | No |
| 2 | Quantile drift | UC1 | Order statistics → LightGBM | **Yes** (escalation only) |
| 3 | Revealed capacity envelope | UC2 | Censored order statistics | No |
| 4 | Temporal-SHARE encoder | UC2–4 | Dilated TCN + relational GNN | **Yes** |
| 5 | Quantile head (capacity strain) | UC2 | Pinball loss | **Yes** (head 4) |
| 6 | Binned CDF head (fill rate) | UC3 | CRPS loss | **Yes** (head 4) |
| 7 | Hazard head (arrival timing) | UC4 | Discrete-time survival | **Yes** (head 4) |
| 8 | Monte Carlo shortage | UC5, UC6 | Simulation + Gaussian copula | No |
| 9 | LP / MILP | UC7, UC8 | Constrained optimisation | No |
| 10 | LightGBM | UC9 + **all baselines** | Gradient-boosted trees | Yes |

### Why so little is learned

- **Shortage (UC5) is computed, not predicted.** A stock roll-forward driven by three
  uncertain inputs. A procurement manager can follow every line of it and challenge any
  number; that auditability is worth more in a first deployment than a couple of points of
  accuracy. It also composes — change the allocation and re-run, and the what-if is free.
- **Allocation and scheduling (UC7, UC8) are exactly solvable.** A solver finds the optimum
  and can state which constraint bound it. No learned recommendation can do that.
- **The BOM explosion is arithmetic.** The structure is given; there is nothing to fit.

### Explicitly out of scope

| Not building | Why |
|---|---|
| **STGNN** (DCRNN, STGCN, Graph WaveNet) | These interleave time-mixing and graph-mixing layer after layer and assume a fixed node set whose values change. This graph churns: `po_line` nodes are created, live weeks, and vanish — 1,169,215 of them across the span. Temporal-SHARE does all time work first, per persistent node, then one graph pass, which sidesteps the churn and costs far less. |
| **Reinforcement learning** | No simulator good enough to train in, no reward signal at scale, and the allocation problem is exactly solvable. A MILP is better *and* explainable. |
| **Counterfactual / causal engine** | `supplier_allocation.change_reason` gives quasi-experimental variation worth analysing, but a causal identification strategy is not in scope for the POC. |
| **Binary classifiers of any kind** | See §8. |

---

## 2. Graph specification

### 2.1 Node types

Nodes split by whether they have a history worth encoding. A channel has up to 523 weeks of
behaviour; a PO line is created, lives three weeks, and is gone. Running a sequence model
over a PO line is meaningless.

**Persistent nodes** — receive the temporal encoder (§3):

| Node | Count (`csv_full_seed1`) | Static feature source | d_static |
|---|---|---|---|
| `channel` | **16,072** (9,026 with `effective_to` null) | `sourcing_channels.csv.gz` | 12 |
| `supplier` | **902** (836 tier-1, 66 tier-2) | `suppliers.csv.gz` | 27 |
| `part` | **8,055** | `parts.csv.gz` | 28 |
| `plant` | **7** | `plants.csv.gz` | 8 |
| `supplier_group` | **225** | — no attribute table — | **OPEN** |

**Instance nodes** — current-state features only, no sequence:

| Node | Count | Feature source | d |
|---|---|---|---|
| `po_line` | **1,169,215** total; ~15k open at any snapshot | `po_lines`, `purchase_orders`, `supplier_acknowledgements`, `po_line_revisions`, `asn` | 18 |

> **OPEN — `supplier_group` has no features.** `supplier_group_id` is a column on
> `suppliers.csv.gz`; there is no `supplier_groups` table and no attributes. Options: (a) a
> learned embedding of dimension 16, (b) aggregate features of member suppliers, (c) drop
> the node and encode group membership only as a `supplier–supplier` relation. Decide before
> Phase 4. Option (c) removes a node type but loses the 2-hop path that capacity strain is
> expected to need (§4, readout depth).

### 2.2 Static feature composition

Exact columns, from the files. Categorical cardinalities were measured, not assumed.

**`channel` (d_static = 12)** — from `sourcing_channels.csv.gz`
```
is_approved                    1   boolean
approval_status                4   one-hot: approved, conditional, development, blocked
transport_mode                 4   one-hot: road, rail, sea, air
contracted_lead_time_days      1   min 4, median 29, max 206
transport_distance_km          1   min 3, median 163, max 2072
days_since_ppap_date           1   derived: t0 - ppap_date
```

**`supplier` (d_static = 27)** — from `suppliers.csv.gz`
```
country                        6   one-hot (India + 5 import origins)
state                         10   one-hot
supplier_tier                  2   one-hot: tier1, tier2
supplier_type                  3   one-hot: manufacturer, trader, job_work
business_class                 3   one-hot: oem_approved, local, import
msme_flag                      1   boolean
payment_terms_days             1   min 30, median 45, max 90
days_since_onboarded           1   derived
```
`suppliers.csv.gz` deliberately carries **no** `reliability_score`. `dataset_structure.md`
§15 rejects any pre-computed whole-history supplier score as outcome leakage — the system
exists to infer it.

**`part` (d_static = 28)** — from `parts.csv.gz`
```
part_category                  6   one-hot
material_type                 15   one-hot
uom                            2   one-hot: EA, KG
is_critical                    1   boolean
standard_lead_time_days        1   min 5, median 26, max 150
shelf_life_days                1   + 1 missing-indicator (null for non-perishables)
days_since_introduced          1   derived
```
`criticality_reason` is **excluded**: only one of its four specified values occurs in the
data, and it is null on 92.6% of parts (§11.1). Including it would add a column that is
constant wherever it is present at all.

**`plant` (d_static = 8)** — from `plants.csv.gz`
```
state                          4   one-hot
capacity_units_per_day         1
latitude, longitude            2
days_since_commissioned        1   derived
```

**`po_line` (d = 18)** — the instance features `data_plan.md` UC3 specifies. Every one must
be reconstructed as of t₀ (§9).
```
qty_ordered                    1   po_lines.qty_ordered
qty_relative                   1   qty_ordered / median(qty_ordered) over channel, 52w
unit_price                     1   po_lines.unit_price
price_deviation                1   vs part_costs trailing mean
days_to_promise                1   current_promise_date(as of t0) - t0
days_since_order               1   t0 - po_lines.created_ts
days_overdue                   1   max(0, t0 - promise)
working_days_to_promise        1   via calendar.is_working_day
ack_present                    1   boolean
ack_gap_ratio                  1   ack_qty / qty_ordered
days_since_ack                 1   t0 - ack.event_ts
revision_count                 1   count of po_line_revisions with recorded_ts <= t0
supplier_initiated_revisions   1   count where initiated_by = 'supplier'
promise_slip_days              1   current_promise_date - original_promise_date
is_emergency                   1   purchase_orders.po_type = 'emergency'
asn_present                    1   boolean
days_since_dispatch            1   t0 - asn.dispatch_ts
transit_elapsed_ratio          1   (t0 - dispatch_ts) / logistics_lanes.standard_transit_days
```
`transit_elapsed_ratio` matters more than it looks: *15 days elapsed means nothing on a
40-day sea lane and disaster on a 3-day road lane.*

`po_lines.csv.gz` carries **no** `qty_received` and **no** `status` column — the first is the
fill-rate label, the second is terminal-state leakage to be reconstructed from
`po_line_revisions`.

### 2.3 Edge types, with measured cardinality

| Relation | Source file | Edges | Fan-out (median / P90 / max) |
|---|---|---|---|
| `channel —sourced_from→ supplier` | `sourcing_channels` | 16,072 | 12 / 56 / 168 channels per supplier |
| `channel —supplies→ part` | `sourcing_channels` | 16,072 | 2 / 3 / 6 channels per part |
| `channel —delivers_to→ plant` | `sourcing_channels` | 16,072 | 7 plants total |
| `supplier —member_of→ supplier_group` | `suppliers.supplier_group_id` | 902 | 4 / 6 / 10 suppliers per group |
| `supplier —depends_on→ supplier` (tier-2) | `supplier_upstream` | **231** | sparse by design |
| `part —used_in→ product` | `bom` | 68,023 total, **18,532 active** | 15 parts per product (median, active) |
| `part —stocked_at→ plant` | `part_plant` | 12,569 | |
| `supplier —alternate_for→ part` | `alternate_sources` | 7,250 | |
| `supplier_site —ships_via→ plant` | `logistics_lanes` | 4,512 | carries `via_checkpoint` |
| `po_line —on→ channel` | `po_lines.channel_id` | 1,169,215 | ~15k open at a snapshot |

All relations are used in both directions, so **R = 20 relation types** (10 forward + 10
reverse). This matters for the basis decomposition in §4.

> **`supplier_upstream` is sparse on purpose.** 231 links for 836 tier-1 suppliers. Partial
> visibility is the realistic case and `dataset_structure.md` §4 is explicit that even a
> partial map is what exposes correlated failure. Tier-2 risk must be presented as a
> network-propagation signal backed by partial visibility, never as though the map were
> complete.

---

## 3. Temporal encoder

### 3.1 Input

Per persistent node, a weekly sequence from `derived/channel_performance_weekly.csv.gz` and
`derived/supplier_performance_weekly.csv.gz`. Row counts and the sparsity figures below are
restated in §3.2 after the Phase C cadence change, which moved them substantially.

The 20 columns are `channel_id`, `week_start`, then **18 features**:

| Group | Columns |
|---|---|
| Activity | `qty_ordered`, `qty_received`, **`is_active_week`** |
| Instantaneous | `fill_rate`, `lead_time_actual_days`, `lead_time_ratio`, `ack_gap_ratio`, `revision_count`, `load_ratio` |
| Rolling (over **active** weeks) | `fill_rate_last4`, `fill_rate_last13`, `fill_rate_last52`, `otd_rate_last13` |
| Sparsity / recency | **`active_weeks_in_52`**, `days_since_last_short`, **`weeks_since_last_activity`**, **`weeks_since_last_receipt`** |
| Reporting | **`reporting_lag_days`** |

Five columns are new and one is gone. **`staleness_days` has been removed** — see §5.

> ### The rolling columns were renamed, and the rename is the point
>
> `fill_rate_4w/13w/52w` and `otd_rate_13w` are now
> `fill_rate_last4/last13/last52` and `otd_rate_last13`.
>
> They never were calendar windows. The builder appends to the history only in weeks the
> channel ordered, so `fill_rate_13w` was the mean over the last 13 **active** weeks. At the
> pre-Phase-C cadence that spanned a **median of 85 calendar weeks** — the name overstated
> the recency of the number by about 6.5×:
>
> | Column | Nominal | Calendar weeks actually spanned (P10 / P50 / P90) |
> |---|---|---|
> | `fill_rate_4w` → `_last4` | 4 | 15 / **20** / 35 |
> | `fill_rate_13w` → `_last13` | 13 | 70 / **85** / 120 |
> | `fill_rate_52w` → `_last52` | 52 | 305 / **350** / 410 |
>
> 59.4% of channels had fewer than 52 active weeks in total, so `fill_rate_52w` was a
> lifetime mean rather than a one-year one; 31.1% had fewer than 13.
>
> **The window still counts active weeks — that is deliberate.** A 13-calendar-week window
> at this cadence holds one or two observations and is mostly binomial noise. The fix is to
> stop the name asserting a time scale it does not have, and to emit `active_weeks_in_52`
> so the model can see what scale it *does* have. Anything that assumed "13 weeks" meant
> three months of history was wrong before this rename and is wrong now; the difference is
> that it is no longer silently wrong.
>
> The table also mixed three conventions without saying so: `fill_rate_*` counted active
> weeks, `otd_rate_13w` counted the last 13 **receipts**, and `load_ratio`'s trailing peak
> counted 52 **calendar** weeks. Only the last is unchanged.

### 3.2 The sequence is sparse — this dominates the implementation

**Sparsity fell from 90.1% to 81.4% of channel-weeks** with the Phase C cadence split (§3.2.1),
but a channel is still idle in four weeks out of five. Measured on `csv_full_seed1`:
**5,930,225 rows over 15,333 channels**.

| Column | Null | Reason |
|---|---|---|
| `fill_rate`, `ack_gap_ratio` | **81.4%** | non-null only in weeks the channel ordered |
| `load_ratio` | **81.8%** | same — see the note below |
| `lead_time_actual_days` / `_ratio` | **72.6%** | only weeks with a receipt |
| `reporting_lag_days` | **53.5%** | only weeks carrying a posted event |
| `otd_rate_last13`, `weeks_since_last_receipt` | 1.5% | **NULL means never received**, not "long ago" |
| `fill_rate_last4/13/52`, `days_since_last_short`, `weeks_since_last_activity` | **0.0%** | rolling, carried forward; defined from each channel's first week |
| `qty_ordered`, `qty_received`, `is_active_week`, `revision_count`, `active_weeks_in_52` | 0.0% | always defined |

**Any implementation that mean-imputes or zero-fills these is training on fabricated
activity.** Each nullable column gets a paired binary missing-indicator, and the loss must
not treat a null as a zero.

> ### Zero and "no data" are now distinguishable — `load_ratio` was not
>
> `load_ratio` used to be emitted as **0** on empty weeks, because `qty_ordered / peak` is
> 0 when nothing was ordered. That put **4,341,025 fabricated "zero load" observations —
> 73% of all rows** — into the table, indistinguishable from a channel that genuinely
> ordered a negligible amount against a large trailing peak. It is now NULL on empty weeks
> (4,852,869 nulls, 0 spurious zeros).
>
> This is the general rule for this table: **a `fill_rate` of 0 is a total delivery failure;
> an empty week is no information.** They must never collapse to the same number.
> `weeks_since_last_receipt` follows the same rule — NULL means *never*, which is a different
> object from a large value.

Empty weeks carry the last known rolling state forward rather than resetting to zero, and
`weeks_since_last_activity` says how stale that carry-forward is.

#### 3.2.1 Ordering cadence is bimodal

Pre-Phase-C the realised order rate was unimodal at ~5 orders/channel-year with **no** high-
runner population at all: P99 was 8.7/yr and **0.0%** of channels ordered more than 26 times
a year. Real automotive procurement splits sharply — A-class runners on weekly or fortnightly
schedule releases, C-class slow movers on monthly-to-quarterly discrete POs.

Two things caused the flat distribution, and the second was the binding one:

1. Review cadence was drawn from one distribution for every channel, independent of volume.
2. **`po_batch_weeks = 5` applied to every buyer**, so orders queued until a 5-weekly batch
   run and *no* channel could order more often than every five weeks however often it
   reviewed. This capped the whole population at ~10 orders/year.

Cadence is now tied to expected volume, and high runners release on their own weekly cycle
(`high_runner_share = 0.20`, `high_runner_po_batch_weeks = 1`). Realised at `small`:

Realised on `csv_full_seed1`:

| Orders per channel-year | Before | After |
|---|---|---|
| P50 | 5.2 | 5.2 |
| P75 | 6.9 | 7.4 |
| P90 | 7.7 | **23.6** |
| P99 | 8.7 | **40.9** |
| share ≥ 13/yr (fortnightly or better) | **0.0%** | **17.3%** |
| share ≤ 6/yr (quarterly-ish) | 59.9% | 59.3% |

The distribution is now two-mode with a trough between 9 and 13 orders/year: the slow-mover
mode is unchanged (P50 still 5.2, ≤6/yr still ~59%) and a distinct high-runner mode has been
added on top. That is the intended shape — the split adds a population, it does not shift
the existing one.

**This inflates the `purchase_orders` document count — see §11.8.**

### 3.3 Per-timestep feature vector

```
18  numeric columns from 3.1  (is_active_week is one of them, as 0/1)
13  missing-indicators (one per nullable column in 3.2)
 2  month-of-year as sin/cos           from calendar.month_of_year
 1  is_shutdown                        from calendar.is_shutdown
 4  regime one-hot                     from calendar.regime_flag
    (normal 69.9%, covid 15.2%, chip_shortage 12.5%, demonetisation 2.3%)
--
38  = d_in          (was 31 before Phase C)
```

**`is_active_week` is the mask, and it is also a feature.** It is carried alongside every
sequence so the encoder can gate on it, and included in the vector so the model can use
"this channel ordered this week" directly. Do not drop it from one role because it appears
in the other.

The 13 nullable columns, each with its own indicator, are `fill_rate`, `ack_gap_ratio`,
`load_ratio`, `lead_time_actual_days`, `lead_time_ratio`, `reporting_lag_days`,
`otd_rate_last13`, `weeks_since_last_receipt`, `fill_rate_last4`, `fill_rate_last13`,
`fill_rate_last52`, `days_since_last_short`, `weeks_since_last_activity`. The three
`fill_rate_last*` columns are always null together, so their indicators are collinear — drop
two if that matters to the optimiser, but keep the count honest at 13 by default.

### 3.4 Architecture — dilated causal TCN

For each persistent node take the trailing **K = 52 weeks** ending at t₀, giving
X_c ∈ ℝ^(52 × 38), plus the `is_active_week` mask over the same 52 positions. A dilated causal convolution at layer *l* with dilation d_l = 2^l:

$$\mathbf{z}^{(l+1)}_t = \sigma\left(\sum_{k=0}^{K-1} \mathbf{W}^{(l)}_k \, \mathbf{z}^{(l)}_{t - d_l \cdot k} + \mathbf{b}^{(l)}\right)$$

*Causal* means position *t* only ever sees *t' ≤ t*. **This is a structural leakage guarantee
at the architecture level**, independent of the data-pipeline convention in §9.

| Hyperparameter | Value | Justification |
|---|---|---|
| Kernel size K | 2 | |
| Dilations | 1, 2, 4, 8, 16, 32 | |
| Layers L | **6** | |
| Receptive field | 1 + Σ(K−1)·d_l = 1+1+2+4+8+16+32 = **64 weeks** | ≥ the 52-week input window, so the top layer sees all of it |
| Hidden dim | 64 | |
| Node state | **s**_c = **z**^(L)_{t₀} ∈ ℝ^64 | final position only |

`model_plan.md` specifies 5 layers giving a 32-week receptive field. **This spec uses 6**, so
the field covers the full 52-week window; with 5 layers the earliest 20 weeks of the input
are unreachable.

**Parameter count**
```
layer 0 :  K × d_in × d_hidden  = 2 × 38 × 64  =  4,864
layers 1-5: K × 64 × 64 × 5     = 2 × 64 × 64 × 5 = 40,960
biases    : 64 × 6                              =    384
residual 1×1 projection (layer 0 only): 38 × 64 =  2,432
                                                  -------
                                        total   ≈ 48,640
```

(§4's encoder total quotes this as 48,640 too. An earlier revision carried 47,296 there,
which matched no decomposition in this document.)

**Why TCN over GRU:** parallel across timesteps, no vanishing gradients, and the receptive
field is an explicit design choice rather than something you hope the gates learn.

### 3.5 Variable-length sequences

Minimum observed length is **19 weeks**; 222 channels (1.45%) have fewer than 52. Handling:

1. **Left-pad** to 52 with zeros and carry a per-timestep validity mask.
2. Because the convolution is causal, padding on the left cannot leak — a padded position is
   only ever a *past* input to a real one, never the reverse.
3. Mask must reach the loss: a channel with 19 real weeks contributes a state built partly
   from padding, and that node's downstream loss should be down-weighted or the node
   excluded from the training set for that snapshot. **OPEN — pick one and state it.**
4. `weeks_since_last_activity` and `active_weeks_in_52` are passed as plain features (§5.2),
   so the model can learn that a thin or long-idle sequence deserves less trust.

**Two masks, not one, and they are not interchangeable:**

| Mask | Means | Use |
|---|---|---|
| **padding validity** | this position is before the channel's first week — no row exists | exclude from convolution and loss |
| **`is_active_week`** | a real row exists, and the channel ordered nothing that week | a genuine observation; feed it, do not exclude it |

Conflating them throws away four rows in five. A padded position is *absence of data*; an
inactive week is *data saying nothing happened*, which is exactly the signal §5.2 is about.

---

## 4. Graph encoder — SHARE

Unchanged in form from HADES v3. Two mechanisms.

**Basis decomposition.** With R = 20 relation types, per-relation weight matrices would be
20 × 64 × 64 = 81,920 parameters per layer, and the thin relations (`supplier_upstream`, 231
edges) cannot support their own set. Instead:

$$\mathbf{W}_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} \mathbf{V}_b^{(l)}, \qquad B = 10$$

A relation never gets free parameters, only a different mixing of B shared bases.

**Relation-blind shared attention.** One scorer vector **a**, shared by every relation:

$$e_{ij} = \text{LeakyReLU}\Big(\mathbf{a}^\top \big[\mathbf{W}_r \mathbf{h}_i \,\Vert\, \mathbf{W}_r \mathbf{h}_j\big]\Big), \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

The scorer never sees *which* relation connects two nodes, only the content of the
transformed endpoints. This recovers the ability to single out one standout neighbour
without reintroducing per-relation parameters. Full per-relation attention (HGT/SHARK)
overfits thin relations and showed **~10× worse seed-to-seed instability** at the same
accuracy in v3's bake-off.

**Layer update:**

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}_0^{(l)}\mathbf{h}_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \alpha_{ij}\, \mathbf{W}_r^{(l)} \mathbf{h}_j^{(l)}\right)$$

Persistent nodes enter with **h**⁰_c = **s**_c (their temporal state, §3). Instance nodes
enter with a linear projection of their 18 current-state features to ℝ^64.

| Hyperparameter | Value |
|---|---|
| Bases B | 10 |
| Hidden dim | 64 |
| Layers | 2 (see readout depth below) |
| Relations R | 20 |

**Parameter count per layer**
```
bases      B × d × d      = 10 × 64 × 64 = 40,960
mixing     R × B          = 20 × 10      =     200
self-loop  W_0            = 64 × 64      =   4,096
attention  a ∈ R^{2d}     =                    128
                                            -------
                                    per layer ≈ 45,384
                                    2 layers  ≈ 90,768
```

Encoder total ≈ **139,000 parameters** (TCN 48,640 + GNN 90,768 = 139,408), before heads.

### Readout depth must be derived, not inherited

HADES v3 fixed readout depth per task — delay→h¹, shortage→h³, impact→h⁴ — after eight
attempts to learn it all lost to simply choosing it. **That mapping must be re-derived for
Rane.** v3's own Layer 3 work found h⁴ was the *worst* depth for both latent states that
passed their gate, with h¹ and h² winning. Depth is task-specific in a way that is not
predictable in advance.

Expected — to be tested, not assumed: fill rate and timing read shallow (h¹: the channel and
its supplier); capacity strain reads deeper (h²–h³, reaching `supplier_group` and the tier-2
parent). **Gate 7 in §10 makes measuring this mandatory.**

---

## 5. Staleness gating

> ### `staleness_days` is gone. This section previously designed a gate against a distribution that was 99% something else.
>
> The column meant **two opposite things in one number**: on a week the channel ordered it
> was reporting lag; on an empty week it was how long the channel had been idle. 90% of rows
> were empty weeks, so the pooled distribution this section used to quote — *"median 12 days,
> P90 838"* — was almost entirely inactivity. The text argued the long tail was "real and
> matters" and warned that a gate saturating at 30 or 60 days would "treat most of that tail
> identically". That was correct about the number and wrong about what the number was.
>
> Split apart, measured on `csv_full_seed1`:
>
> | | n | P50 | P90 | P99 | max |
> |---|---|---|---|---|---|
> | `staleness_days` as shipped (pooled) | 5,937,814 | 12 | **838** | 2,292 | 3,569 |
> | …on **active** weeks — posting lag *as then measured* | 588,470 | 3 | 5 | 6 | 6 |
> | …on **empty** weeks — inactivity | 5,349,344 | 14 | 940 | 2,330 | 3,569 |
>
> A gate designed for a 838-day P90 is designed for the wrong quantity: that tail was
> inactivity, not lag. The split was the right diagnosis.
>
> **The 6-day ceiling in the middle row was itself an artefact, and is now gone.** It was
> not a property of the data. The builder bucketed a row into its event week only if the row
> had also been *recorded* in that week, so any record posted after its own week closed was
> discarded — and a 7-day week cannot contain a lag longer than 6 days. The ceiling was the
> calendar, measuring itself. `config.LagParams.max_lag_days` has always been 95.
>
> Corrected, on the regenerated `csv_full_seed1`:
>
> | | n | P50 | P90 | P99 | max |
> |---|---|---|---|---|---|
> | `reporting_lag_days` | 2,755,637 | **1.3** | **5.7** | **14** | **96** |
>
> and the raw per-row lag it aggregates, pooled over the five source tables, is P50 1 /
> P90 7 / P99 19 / max 64 at `small` — `quality_inspections` is the slowest-posting table and
> was exactly what the old ceiling censored. The store now reproduces the distribution the
> generator produces instead of the distribution the bucketing rule allowed through.

Rane's ERP is manually updated, so at any t₀ some records are fresher than others, and
separately some channels have been silent for a long time. These are different problems and
now have different columns:

| Column | Meaning | Range on `csv_full_seed1` |
|---|---|---|
| `reporting_lag_days` | `recorded_ts − event_ts`, averaged over the week's events. NULL on weeks with nothing posted (53.5%). | P50 **1.3**, P90 **5.7**, P99 14, **max 96** |
| `weeks_since_last_activity` | Weeks since the channel last ordered. Defined every week. | P50 4, P90 127, max 509 |
| `weeks_since_last_receipt` | Weeks since the last GRN. **NULL means never** (1.5%). | P50 2, P90 120, max 503 |
| `is_active_week` | `qty_ordered > 0`. Defined every week. | 18.6% true |

Note how far apart the two axes are: lag lives in **0–96 days** and is concentrated below a
week (P90 5.7); inactivity runs to **509 weeks**. They were previously added together into
one integer, and the sum was dominated by the second.

### 5.1 The trust gate applies to `reporting_lag_days`

Each time-varying feature carries a learned trust weight decaying with the age of the record
it came from:

$$\tilde{x}_i = g_i \cdot x_i^{\text{obs}} + (1-g_i)\cdot x_i^{\text{prior}}, \qquad g_i = \exp\big(-\max(0,\ w_i \Delta t_i + b_i)\big)$$

where Δt is `reporting_lag_days` (NULL on 53.5% of rows — pass the missing-indicator and
leave x untouched there), w_i and b_i are learned **per feature**, and x^prior is a learned
constant per feature.

Three rules make this work rather than break it:

1. **Per feature, not per node.** `country` never changes and must not be penalised for being
   old. Apply decay only to the time-varying columns.
2. **Continuous, not a threshold.** A hard "stale after 30 days" cutoff makes a 29-day and a
   31-day record behave completely differently, which shows up as instability. An earlier
   revision of this section conceded the point was near-moot because lag never exceeded 6
   days; that ceiling was a measurement artefact (see above) and the concession is withdrawn.
   A 30-day cutoff now sits **inside** the distribution, between P99 (14) and the 96-day max,
   so it would split the real tail. The argument is load-bearing again.
3. **Δt is also passed as a plain feature.** Staleness has two jobs — it decides *trust* via
   the gate, and it may itself be predictive. Precedent: GRU-D.

**Initialise for the range that exists.** A w_i initialised for a scale of hundreds of days
makes g_i ≈ 1 everywhere and the gate learns nothing; one initialised far too sharply
saturates it to 0. Scale the initialisation to the observed **P90 of 5.7 days**, and note the
distribution is heavy-tailed rather than bounded — P99 is 14 days and the max is 96 — so the
gate must stay meaningful two orders of magnitude above its median rather than only across a
handful of days.

### 5.2 Inactivity is a feature, not a trust signal

`weeks_since_last_activity` must **not** be fed into the decay gate. A channel that has not
ordered for 40 weeks has not given you a stale reading — it has given you a true reading of a
quiet channel, and down-weighting it discards exactly the signal that distinguishes a dormant
channel from a steady one. It is a genuine input feature in its own right: a channel silent
for 40 weeks behaves differently from one ordering every week, and after §3.2.1 the dataset
contains both in quantity.

It is also what makes the carried-forward rolling columns interpretable. `fill_rate_last13`
on an empty week is the last known value; `weeks_since_last_activity` says how old that value
is, and `active_weeks_in_52` says how much calendar time it averaged over.

---

## 6. Heads

All three sit on the shared encoder output **h**_i at the task's chosen readout depth.

### 6.1 Quantile head — capacity strain (UC2)

**Entity:** `channel`. **Label:** `training_labels.task = 'capacity_strain'`, 1,005,609 rows,
`entity_type = 'channel'`, `label_value` = observed fill rate over the window, 0.0 – 1.0.

Output shape **(3 quantiles × 3 horizons) = 9** values per channel:

$$\hat{u}_q(s, t_0+h), \quad q \in \{0.1, 0.5, 0.9\},\ h \in \{30,60,90\}$$

Loss:

$$\mathcal{L} = \sum_{h}\sum_{q} \mathcal{L}_{\text{pinball}}\big(u_{\text{true}}(h),\ \hat u_q(h)\big), \qquad
\mathcal{L}_q(\rho, \hat\rho) = \max\big(q(\rho - \hat\rho),\ (q-1)(\rho - \hat\rho)\big)$$

**Monotonicity is structural, not cosmetic.** Nothing in the loss prevents û₀.₁ > û₀.₉.
Predict û₀.₁ plus non-negative softplus increments to the higher quantiles. Sorting post-hoc
also works but hides the pathology rather than preventing it.

**Days-to-exceed is derived, never learned:** DTE = min{h : û₀.₅(h) ≥ 1}, linearly
interpolated on the 30/60/90 grid. Learning it separately duplicates information already in
the curve and adds error.

**Feeds:** the capacity cap in the Monte Carlo (§7 of `model_plan.md`), and the capacity
constraint in the allocation MILP.

> **OPEN — the label is a fill rate, not a utilisation.** The `training_labels` rows with
> `task = 'capacity_strain'` carry observed fill (median 1.000, min 0.000, max 1.000), whereas `model_plan.md` §5
> describes û as *utilisation* = delivered ÷ revealed capacity, which can exceed 1.
> `derived/revealed_capacity_monthly.csv.gz` supplies `capacity_utilisation_observed` and
> `revealed_capacity_est` if utilisation is wanted instead. Decide which target before
> Phase 5; they are not interchangeable and the ">1 means trouble" reading only works for
> utilisation.

### 6.2 Binned CDF head — fill rate (UC3)

**Entity:** `po_line`. **Label:** `training_labels.task = 'fill_rate'`, 1,284,907 rows (12.3% censored),
`label_value` = qty_accepted ÷ qty_ordered ∈ [0,1].

**Measured label distribution** — this is why a binned head and not a Gaussian:

Over the **uncensored** population (1,127,491 rows):

| | share |
|---|---|
| exactly 1.0 | **91.5%** |
| exactly 0.0 | **1.8%** |
| interior (0,1) | 6.7% |

20 bins with dedicated endpoints for the point masses:

$$b_0 = \{0\},\quad b_1 = (0, 0.05],\ \dots,\ b_{18} = (0.95, 1),\quad b_{19} = \{1\}$$

Output **π** = softmax(**o**) over 20 bins. Loss is CRPS over the cumulative distribution:

$$\mathcal{L}_{\text{CRPS}} = \sum_{b=0}^{19}\Big(\hat F(b) - \mathbb{1}[f_{\text{true}} \le b]\Big)^2, \qquad \hat F(b) = \sum_{b' \le b}\pi_{b'}$$

**Why CRPS not cross-entropy.** Cross-entropy treats bins as unordered — predicting bin 2
when the truth is bin 19 costs the same as predicting bin 18. CRPS knows the bins are ordered
and penalises by distance, and it is a *proper* scoring rule, so the model has no incentive
to hedge.

> ### `label_censored` is now correct on this table — use it as emitted
>
> **This block previously said the opposite.** It was right at the time; the defect is fixed
> (§11.7) and the workaround it prescribed is now itself wrong.
>
> `label_censored` on `fill_rate` means what the schema says: the outcome was not observed by
> `label_window_end`. A line that closed **short** is settled, not censored. Measured on
> `csv_full_seed1`: **12.3% censored**, and the uncensored population has mean 0.9620,
> standard deviation **0.1609**, across 8,661 distinct values.
>
> **Do not recompute censoring from `grn_lines.is_final_receipt` alone.** That was the old
> advice and it is now a bug: 5.80% of PO lines settle with **no final receipt at all** —
> the supplier shipped nothing, or the line timed out short — and are recorded by a
> `po_line_revisions` short-close row instead. Using only `is_final_receipt` would mark all
> of them censored, reintroducing a milder version of the original defect.
>
> **One thing this head must still do explicitly:** `fill_rate` rows are **not emitted at
> all** for buyer-cancelled lines (§11.7.2, **34.3%** of short closes — the largest single
> cause, not the smallest). The supplier was never
> given the chance to deliver, so the outcome is undefined rather than zero. Do not
> reconstruct those rows from `po_lines` and score them as `fill = 0`; if you build the
> population yourself rather than reading `training_labels`, exclude
> `po_line_revisions.reason_code = 'buyer_cancellation'`.
>
> `validate.py`'s `check_labels` gate asserts the uncensored target varies, for every task,
> on every run. See §11.7.4.

**Output consumed downstream as a quantity, not a probability:**

$$\mathbb{E}[f] = \sum_b \pi_b \cdot \text{mid}(b), \qquad \text{expected shortfall} = \texttt{qty\_ordered} \times (1 - \mathbb{E}[f])$$

Units add correctly across the four-level rollup in UC6; probabilities do not.

### 6.3 Discrete-time hazard head — arrival timing (UC4)

**Entity:** `po_line`. **Label:** `training_labels.task = 'arrival_week'`, 1,310,934 rows,
`label_value` = week index of first receipt relative to `original_promise_date`, integers
1–13 observed. **8.4% censored.**

**Measured arrival distribution** (uncensored): w=1 **82.2%**, w=2 4.5%, w=3 3.1%, w=4 2.5%,
w=5 2.1%, then a thin tail to w=13 carrying 5.5%. The mass at w=1 is dominant — a baseline that always predicts
week 1 is strong, and §7 requires beating it.

Hazard = conditional probability of arriving in week *w* given it has not yet:

$$\lambda_w = P(T = w \mid T \ge w)$$

The head emits λ̂_w for w = 1…12 through **independent sigmoids** (not a softmax — these are
conditional, not a distribution over w). Survival and arrival follow:

$$S_w = \prod_{k<w}(1-\hat\lambda_k), \qquad P(T=w) = \hat\lambda_w \, S_w$$

The likelihood handles complete and censored observations in one expression:

$$\mathcal{L} = -\sum_\ell \sum_{w=1}^{\min(T_\ell,\,C_\ell)} \Big( \mathbb{1}[T_\ell = w]\log\hat\lambda_w + \mathbb{1}[T_\ell > w]\log(1-\hat\lambda_w) \Big)$$

**Censoring is the point here, not an edge case.** An open PO contributes log(1−λ̂_w) for
every week it has already survived — real information ("this order has been open five weeks
without arriving"). `training_labels.censor_time` carries the weeks observed so far. Dropping
these rows keeps only orders that already landed, biasing the training set toward fast
suppliers and making every forecast quietly optimistic.

**Feeds the simulation:** arriving(p,l,w) = Σ_ℓ q_ℓ · E[f_ℓ] · P(T_ℓ = w).

---

## 7. Baselines

**Every learned model must beat a stated baseline before it ships.** Baselines are built and
measured in Phase 7 of the implementation guide — *before* the neural model — so the
comparison cannot be retrofitted.

| # | Baseline | Applies to | Definition |
|---|---|---|---|
| B1 | **Seasonal naive** | demand drift | ρ̂ = 1.0 (plan is taken at face value); and ρ̂ = the same calendar month's ratio last year |
| B2 | **Per-channel rolling statistics** | fill rate, capacity strain | `fill_rate_last13` / `fill_rate_last52` read straight off `channel_performance_weekly` as the point prediction, with the empirical spread of the last 52 **active** weeks as the band. Note these are active-week windows spanning a median 85 / 350 calendar weeks (§3.1) — this baseline is therefore a long-run average, and beating it is not the same as beating a genuine 13-week one |
| B3 | **Modal week** | arrival timing | always predict w=1 (**82.2%** of observed arrivals) |
| B4 | **Empirical quantiles by bucket** | demand drift | `data_plan.md` UC1's stated baseline: bucket ρ by (horizon band, plant, product family) and read percentiles off directly. **This is the incumbent for UC1 — LightGBM is an escalation only if buckets prove thin.** |
| B5 | **LightGBM on flat features** | all four learned tasks | Same features, network structure flattened to scalars (`channels_per_supplier`, `supplier_concentration`, degree counts). Uses the same pinball / CRPS losses so the comparison measures the same thing. |
| B6 | **h⁰ ablation** | all four | **Mandatory.** See below. |

### B6 — the h⁰ ablation is mandatory, not optional

Take the full pipeline and **disable message passing**: node features and the temporal
encoder, zero graph layers. This isolates how much the graph contributes over and above
input quality.

HADES v3 measured the graph's contribution on the delay task at **~10% of the above-chance
signal** — raw features 0.7516 AUC, the full four-layer encoder 0.7802. Input quality
dominated architecture by a wide margin. **Run this here before assuming the graph earns its
keep on any given task.**

> **A baseline winning is a valid outcome, not a bug.** If LightGBM on flat features matches
> temporal-SHARE, ship LightGBM. Keep the graph only where it is structurally required:
> multi-tier propagation through `supplier_upstream`, and cold-start channels with too little
> history for their own statistics (222 channels have < 52 weeks).
>
> One caution on interpretation: flattening the network into scalars *is* the LightGBM
> baseline, and once structure is collapsed into a handful of columns the graph has nothing
> left to contribute. **A tie proves the flattening was adequate, not that the graph is
> useless.** Run both and label them as what they are.

---

## 8. Metrics

### No accuracy figures anywhere

Not for any task, not in any internal document, not in the deck. Reason, measured on this
dataset: `part-plant-weeks in shortage` runs at 3.9%, `PO lines with fill < 1.0` at 9.5%.
A classifier that always predicts "no shortage" scores 96% accuracy and is worthless.
HADES v3's operating-point analysis found a binary target at these base rates produced a
model with ~25% precision that nobody could act on. **Every output is a quantity and every
metric is scored against a quantity.**

| Task | Primary metric | Secondary | Target |
|---|---|---|---|
| **Demand drift** (UC1) | WMAPE **per horizon band**, never averaged across them | Pinball loss at q=0.1/0.5/0.9 | Beat B4 at every horizon band |
| **Capacity strain** (UC2) | Pinball loss at 0.1/0.5/0.9 | **Event recall with lead time**: of historically significant capacity events, how many flagged ≥ N days early | Beat B2 and B5; recall target set jointly with Rane |
| **Fill rate** (UC3) | **CRPS** over the 20-bin CDF | **Band coverage**: does the P10–P90 band contain 80% of outcomes? | Beat B2 and B5 on CRPS; coverage within 80% ± 5pp |
| **Arrival timing** (UC4) | **Concordance index** (C-index) | **Weekly calibration** of P(T=w): predicted vs observed frequency per week | Beat B3 and B5; calibration error < 5pp per week |
| **Shortage** (UC5) | **MAE in units** and **MAE in days** on the earliest-risk date | Precision/recall **on critical parts only** (400 active) | Set with Rane before building |
| **Quality risk** (UC9) | Pinball / AUC-PR on reject rate | | Phase 2 |

**"Significant capacity event" is undefined and must be defined with Rane before the metric
can be computed.** Do not invent a threshold. Mark as **OPEN**.

**Why per-horizon and never averaged.** The client's stated problem is that plan accuracy
decays from month 1 to month 3. Averaging across horizons hides exactly the thing the model
exists to fix. `derived/plan_drift_features.csv.gz` carries `horizon_days` on every row for
this purpose.

---

## 9. Validation protocol

### 9.1 The as-of rule

For any feature x computed at t₀:

$$x(t_0) = g\big(\{\text{rows} : \texttt{recorded\_ts} \le t_0\}\big)$$

**Never `event_ts ≤ t₀`.** An event that happened on the 5th but was entered on the 12th was
not knowable on the 8th. This is the most common and most damaging error in this class of
system, and the dataset is built to expose it: reconstructing a channel's weekly receipts
with `recorded_ts` versus `event_ts` differs on **9.8% of channel-weeks**, and **6.9% of
receipt quantity** had physically arrived but was not yet posted at their event week's close.

Additionally bounded by `snapshots.data_cutoff_ts`. For effective-dated master data
(`bom`, `sourcing_channels`, `part_plant`, `supplier_allocation`) the equivalent selection is
`effective_from <= t0 AND (effective_to IS NULL OR effective_to > t0)` — and for the BOM
explosion specifically, the window must contain the **target period**, not t₀.

### 9.2 Snapshots and rolling origins

`derived/snapshots.csv.gz` supplies **115 snapshots**, spaced exactly **28 days**, from
**2016-12-26 to 2025-09-22**, each with `horizon_days = 91`.

Rolling-origin backtesting. Origins for the primary protocol, inside the 2019–2025 modelling
window:

| Fold | Train through | Evaluate on |
|---|---|---|
| 1 | 2021-12-31 | 2022 Q1–Q2 |
| 2 | 2022-06-30 | 2022 Q3–Q4 |
| 3 | 2022-12-31 | 2023 Q1–Q2 |
| 4 | 2023-06-30 | 2023 Q3–Q4 |
| 5 | 2023-12-31 | 2024 Q1–Q2 |
| 6 | 2024-06-30 | 2024 Q3–Q4 |
| 7 | 2024-12-31 | 2025 Q1–Q2 |
| 8 | 2025-06-30 | 2025 Q3 |

Eight origins for demand; quarterly origins for capacity and fill. **Never a random split** —
it leaks the future. Calibration is fit on earlier folds and evaluated only on later ones.

### 9.3 Reproduction floors

HADES v3's instruments assumed five synthetic dataset seeds. **Rane is one world** — that
axis does not shrink, it disappears. Replacement: recompute reproduction floors across
**model-init seeds × time folds**.

- ≥ 5 model-init seeds × 8 time folds per configuration.
- The **reproduction floor** is the spread of the metric across those runs with no
  intervention. A claimed improvement smaller than the floor is not an improvement.
- **Report per-fold spread every time, never just the mean.** v3's hardest lesson: one fixed
  threshold produced recall from 32 to 94 out of 100 depending only on which world it ran in,
  and the mean of 64.6 described nothing.

The synthetic generator *can* produce more worlds (`--seed`), and 5 seeds are validated. Use
them for generator-side sensitivity analysis, but **do not** use cross-seed variance as a
proxy for the uncertainty on Rane's real data. Uncertainty on real data will be **wider**,
not narrower — it is regime risk across years, product mixes and macro conditions, and it
becomes unmeasurable when there is only one world.

### 9.4 Regime handling

`calendar.regime_flag` distribution over the 25,571 plant-days:
`normal` 69.9%, `covid` 15.2%, `chip_shortage` 12.5%, `demonetisation` 2.3%.

- **Exclude `regime_flag = 'covid'` from the primary training window.** Report performance on
  those rows separately as a robustness check.
- **Never average a structural break into normal operation.**
- This is measurable on this dataset: fitting the drift quantile baseline with and without
  the flag, and scoring on a post-COVID window, makes the regime-blind model **7.6% worse
  overall, 14.4% worse at q=0.1 and 17.4% worse at q=0.9**, with the median unchanged. COVID
  does not shift the centre of drift, it widens and skews it. (Reproduce with
  `db/prove_claims.py`.)
- `chip_shortage` is **part-scoped** — it affects `part_category = 'electronic'` only, though
  `calendar` flags the window for every plant. A model conditioning on the flag alone will
  over-apply it to the other five categories.

### 9.5 Coverage masking

`csv_full_seed1/dataset_coverage.csv.gz` records what history actually exists per table:

| Table | Earliest | Years | Status |
|---|---|---|---|
| `supplier_capacity` | 2020-12-31 | 5.0 | partial |
| `supplier_allocation` | 2019-12-31 | 6.0 | partial |
| `supplier_acknowledgements` | 2019-12-31 | 6.4 | partial |
| `po_line_schedules` | 2018-12-28 | 7.4 | partial |
| `inventory_snapshots` | 2018-01-01 | 8.0 | partial |
| `asn` | 2021-10-13 | 4.2 | partial |

Features derived from a table before its coverage starts must be **masked, not zero-filled**.
A zero in `ack_gap_ratio` reads as *"the supplier acknowledged nothing"* — the opposite of
*"we did not record it."*

---

## 10. Acceptance gates

Numbered, pass/fail, falsifiable. A model does not ship until all gates that apply to it pass.

**G1 — Leakage.** `db/validate.py` passes on the dataset used for training, including the
as-of check (`as-of view differs from the naive event_ts view` must fire at > 1% of
channel-weeks). No feature query reads `event_ts`. No column of any Group I table appears in
`model_outputs`. Grep the feature pipeline for `event_ts` and justify every hit in writing.

**G2 — Censoring handled.** The fill-rate head uses `training_labels.label_censored` as
emitted, which is now correct (§11.7). The gate is unchanged in intent and is now enforced
automatically: `validate.py`'s `check_labels` asserts, for **every** task, that the uncensored
target has relative spread > 0.01 across ≥ 5 distinct values and that the censored fraction
sits in 1–60%. Demonstrated additionally by the share of settled lines with fill < 1.0 landing
in the 8–15% band, not 0%.

Two ways to fail this gate, both of which the pre-fix pipeline did: train on a constant target,
or reconstruct the fill-rate population yourself and score buyer-cancelled lines as `fill = 0`
(§11.7.2).

**G3 — Baseline beaten, per task.** The neural model beats **B5 (LightGBM on flat features)**
by more than the reproduction floor on **at least 4 of the 5 primary time folds**, on the
task's primary metric from §8. Fewer than 4 of 5 → the baseline ships instead.

**G4 — Modal baseline beaten (timing).** The hazard head beats **B3 (always w=1)** on C-index
by more than the reproduction floor on ≥ 4 of 5 folds. Given **82.2%** of arrivals are at w=1,
failing this gate means the head has learned the marginal and nothing else.

**G5 — Calibration.** Fill-rate P10–P90 band contains 80% ± 5pp of held-out outcomes, on
every fold, not on average. Arrival-timing weekly calibration error < 5pp for w = 1…6.

**G6 — Monotonicity.** Zero quantile crossings (û₀.₁ ≤ û₀.₅ ≤ û₀.₉) on the full held-out
set. This is a structural property; a single crossing is a failure.

**G7 — h⁰ ablation reported.** The graph's contribution is **measured and published per
task** as (full − h⁰) ÷ (full − random). No task ships with this number unstated. There is
no minimum value — a low number is an acceptable result that changes what ships (§7).

**G8 — Per-fold spread reported.** Every headline metric is accompanied by its per-fold
range. A mean without a spread is not a result.

**G9 — Regime robustness.** Primary metrics reported on `regime_flag = 'normal'` rows, with
`covid` rows reported separately. A model whose normal-regime metric is only reachable by
including covid rows in training fails.

**G10 — Simulation recovers the copula.** The Monte Carlo's realised within-group shortfall
correlation matches `ground_truth.realised_correlation_matrix()` within ±0.05. This validates
the simulation's correlated sampling, not the generator.

**G11 — Reproducibility.** Every run stamped with `dataset_version`, `feature_spec_version`,
`label_version`, `model_version`, `code_commit`. Re-running a stamped configuration
reproduces its metrics within the reproduction floor.

**G12 — No accuracy figure** appears in any artefact for any task (§8).

**G13 — The difficulty sweep is completed.** The acceptance artefact in §10.1 is filled in.
A single score on a single synthetic world is not a result; the deliverable is the operating
envelope and the point inside it where the approach stops working.

---

## 10.1 Acceptance artefact — the difficulty sweep

**This table is a deliverable, not an illustration.** It stays in this document, empty in its
right-hand half, until training exists to fill it. Leaving it out is how the exercise
degenerates into one number on one world.

### Why it exists

Every other parameter in `db/config.py` is calibrated to one operating point — the base rates
`dataset_structure.md` §1 asks for. That produces a dataset a model either can or cannot
learn, with no way to say *where the boundary is*. `--difficulty` scales the five parameters
governing learnability so the same pipeline can be run across a range and the collapse point
located. When Rane's data arrives, "are we inside the envelope?" becomes a question with an
answer.

```
python db/sweep_difficulty.py --preset mid
python db/generate_dataset.py --preset full --difficulty hard --derive --validate
```

### The axis

| Parameter | easy | nominal | hard | brutal |
|---|---|---|---|---|
| `fill_noise_sd` — spread of the shortfall, at fixed mean | 0.05 | *(0.198)* | 0.25 | 0.40 |
| `fill_sigmoid_gain` — γ, how sharply stress drives a shortfall | 14.0 | 9.0 | 5.5 | 3.0 |
| `fill_sigmoid_offset` — θ, sets the short-delivery base rate | 0.940 | 0.905 | 1.450 | 2.400 |
| `lag_scale_mult` — multiplies every table's reporting lag | 0.50 | 1.00 | 2.00 | 3.50 |
| `shock_weight_mult` — multiplies the shared-shock weights | 1.35 | 1.00 | 0.65 | 0.35 |

**`nominal` is the identity level.** It changes nothing and reproduces the calibrated dataset
**byte for byte** — verified by digest against the shipped `csv_small_seed1`. Every measured
number elsewhere in this document is a `nominal` number. Its `fill_noise_sd` is shown in
parentheses because it is *implied* by the existing `shortfall_beta`, not set; restating it as
a round 0.15 would have silently changed the dataset the whole document is measured on.

θ is **calibrated, not chosen**: γ and θ interact, and a flat sigmoid (low γ) sits near its
midpoint everywhere, so the same θ gives a very different base rate at γ = 14 and γ = 3.

Individual fields override the preset:
`--difficulty hard --difficulty-set fill_noise_sd=0.30,shock_weight_mult=0.5`

### The artefact

Realised figures measured at `mid`, seed 1. **Model columns are filled in once training
exists** — until then they stay as `—`, and G13 is not met.

| difficulty | base rate | fill σ | lag P90 | shock separation | temporal-SHARE | LightGBM | gap |
|---|---|---|---|---|---|---|---|
| easy | 11.8% | 0.193 | 4.6 d | **0.567** | — | — | — |
| nominal | 9.6% | 0.239 | 7.6 d | **0.396** | — | — | — |
| hard | 5.1% | 0.220 | 12.3 d | **−0.002** | — | — | — |
| brutal | 4.7% | 0.224 | 21.2 d | **−0.047** | — | — | — |

Left-hand half measured; regenerate with `python db/sweep_difficulty.py --preset mid`.

Three of the four axes separate the levels cleanly and monotonically — base rate
(11.8 → 4.7%), reporting lag (4.6 → 21.2 days) and shared-shock separation
(0.567 → −0.047). **`fill σ` does not**, and that is a measured limit rather than a
mis-set parameter: past `hard`, θ is high enough that the surviving short deliveries are
produced by capacity truncation and quality rejects rather than by the fill sigmoid, so the
Beta's configured spread stops showing through. Realised σ also exceeds the configured value
at every level for the same reason. See the first limit below.

By `hard` the shared-shock separation has already collapsed to zero. If the model column
shows the gap closing at `hard` rather than `brutal`, this is why — the graph encoder's
signal is gone one level earlier than the fill signal is.

**How to read the result.** The gap column is the point of the table. A gap that shrinks
monotonically to zero locates the collapse point; a gap that is flat across the whole range
means the temporal and graph machinery is not contributing at any difficulty, which is a
finding about the architecture rather than about the data.

### Two limits of the axis, both measured

**1. The short-delivery base rate floors at ~5%.** The brief for this axis asked for 3% at
`brutal`; it is not reachable by dialling the fill sigmoid, because the sigmoid is not the
only thing that makes a line come up short. Measured at `brutal`, θ = 2.4, disabling
mechanisms one at a time:

| Configuration | Short-delivery rate |
|---|---|
| all mechanisms on | **5.3%** |
| minus `capacity_censoring` | 2.3% |
| minus `quality_rejects` | 1.8% |
| minus both | **0.2%** |

At the hard end the base rate is almost entirely capacity truncation and quality rejects, and
the fill sigmoid contributes ~0.2 points. Going lower means dialling
`capacity_headroom_lognorm` and the reject rates too — which also moves `supplier-months
constrained` and `capacity unobservable`, the two gates that make §4.1's censoring story
work. That is a deliberate trade, not a free parameter.

**2. The correlation readout degenerates at low event rates — read `sep`, never `corr`.**
Week-demeaning makes every supplier's series track the negative of the cross-sectional mean,
so when almost nothing happens *every* pair correlates toward 1, unrelated pairs included:

| level | non-zero observations | within-group `corr` | unrelated | **separation** |
|---|---|---|---|---|
| easy | 15.8% | 0.551 | −0.016 | **0.567** |
| nominal | 15.1% | 0.396 | −0.001 | **0.396** |
| hard | 1.8% | 0.215 | 0.217 | **−0.002** |
| brutal | **0.9%** | **0.374** | **0.421** | **−0.047** |

Taken alone, `brutal`'s within-group figure of 0.374 sits close to `nominal`'s 0.396 — while
the real shared structure has entirely vanished, as the separation of −0.047 shows. At
`small` the effect is starker still: `brutal` reads 0.592 against `nominal`'s 0.434, i.e.
*more* correlated, on 0.9% non-zero observations. The separation is the quantity to trust. This is the same class of bug as §11.6.1: a statistic computed
over a sparse series behaving in a way that has nothing to do with what it claims to measure.

Gate G10's companion check — unrelated-supplier correlation must satisfy |r| < 0.1 — catches
this automatically, and correctly fails at `hard` and `brutal`. **That is intended: those
levels are not meant to pass the nominal acceptance gates.** Run `validate.py` against a
non-nominal dataset to characterise it, not to certify it.

---

## 11. Known deviations

Measured from `csv_full_seed1`. Where the generated data and `dataset_structure.md` disagree,
**the data is what the models will see** and is treated as authoritative here.

> **Every figure in this section is checked by `db/check_docs.py`**, which binds it to the
> measurement that produces it and fails if the two drift apart. Run it after any
> regeneration. Three audits in a row found stale numbers here that were correct when
> written; the checker exists so a fourth does not have to.
>
> **Standing deviations, as of the current regeneration** (all three presets rebuilt from one
> code state, `code_commit` identical):
>
> | Deviation | Presets | Status |
> |---|---|---|
> | `PO lines arriving late` 12.8% vs 15–25% band | `mid`, `full` | §11.8 — a mixture tested against a single-population band; the gate is wrong, not the data |
> | `purchase_orders` +92% vs §1 target | `full` | §11.8 — a modelling shortcut, diagnosed and not fixed |
> | `po_line_revisions` +24% vs §1 target | `full` | §11.8 — roughly half B1's short-close rows, half C3's extra lines |
>
> Nothing else fails. `small` passes 50/50, `mid` 48/49, `full` 46/49.
>
> **Two former entries have been removed as resolved, not as tolerated:**
>
> - *The weekly stores lost 4.5% of active channel-weeks.* The builder bucketed a row into its
>   event week only if it had also been recorded in that week, so anything posted late was
>   discarded rather than deferred — 4.45% of ordered and 6.86% of received units reached no
>   week at all. Fixed by bucketing on `max(event_week, recorded_week)`;
>   `validate.py` now asserts exact conservation against the source. This is also why
>   `reporting_lag_days` used to top out at 6 days (§5): a 7-day week cannot hold a longer lag.
> - *Four apparent `mid` failures* (`right-censored tail`, `shortage events`, `line stops`,
>   `expedites`). These were artefacts of `validate.py` defaulting to the `small` preset
>   whatever directory it was pointed at, grading a 7-year 5-plant world against a 3-year
>   3-plant window. The preset is now read off the directory name and a contradicting
>   `--preset` is an error. They were never deviations in the data.

### 11.1 Enum columns with values that never occur

| Column | Spec values | Present in data | Impact |
|---|---|---|---|
| `parts.criticality_reason` | `single_source`, `long_lead`, `safety`, `high_value` | **only `single_source`**, on 600 of 8,055 rows; the other 7,455 are empty | Constant *among non-null values*, and **92.6% NULL**. Both reasons to drop it, but they are different reasons. **Excluded from `part` features** (§2.2). |
| `po_line_revisions.field_changed` | `promise_date`, `qty`, `price`, `status` | only `promise_date`, `qty` | No price or status revisions exist. Any feature keyed on them is always zero. |
| `supplier_acknowledgements.ack_status` | `full`, `partial`, `rejected`, `date_change` | `full`, `partial`, `rejected` | `date_change` never generated. |
| `inventory_transactions.txn_type` | 7 values incl. `return` | 6 — no `return` | Returns are not modelled. |
| `purchase_orders.status` | `open`, `closed`, `cancelled`, … | **empty on every row** | The column is written but never populated. `dataset_structure.md` §15 excludes current-state fields, so status is deliberately absent — but the column should be dropped rather than emitted blank. Line settlement is carried by `grn_lines.is_final_receipt` and the short-close rows in §11.7.1. |

`po_line_revisions.reason_code` gains three values in §11.7.1 that mark a settlement rather
than an ordinary revision: `supplier_no_ship`, `short_close_timeout`, `buyer_cancellation`.
Any feature keyed on `reason_code` must exclude them — they encode the fill-rate label
(they are on `config.HIDDEN_STATE_COLUMNS` for the derived column names, but the raw
`reason_code` value itself is legitimately present in `po_line_revisions` and it is the
consumer's job not to build a "was short-closed" feature from it).

### 11.2 Counts differing from `dataset_structure.md` §1

The §1 figures are stated for **7 years**; `csv_full_seed1` spans **10**, so distinct counts
legitimately run higher. Active counts match well.

| Entity | §1 "active today" | Measured active | §1 "distinct over 7y" | Measured distinct (10y) |
|---|---|---|---|---|
| Suppliers | 550 | 550 tier-1 | 750 | 836 tier-1 + 66 tier-2 |
| Parts | 4,500 | 4,500 | 7,000 | 8,055 |
| Products | 1,200 | 1,200 | 2,500 | 3,060 |
| Sourcing channels | 9,000 | **9,026** | 14,000 | 16,072 |
| Tier-2 links | 200 | 231 | 250 | 231 |
| BOM lines | 22,000 | 18,532 active | 40,000 | 68,023 |

### 11.3 Dates beyond the simulated span

A small number of rows carry dates after 2025-12-31, because orders placed near the end are
promised beyond it. This is realistic (open orders at extract time) but will break naive
date-range filters.

| Column | Rows beyond span | Max |
|---|---|---|
| `po_line_schedules.schedule_date` | 2,935 (0.16%) | 2026-06-29 |
| `supplier_acknowledgements.ack_date` | 1,450 (0.20%) | 2026-06-27 |
| `po_lines.original_promise_date` | 1,484 (0.13%) | 2026-06-27 |

`grn_lines.event_ts` has none — nothing is *received* beyond the span.

### 11.4 Sequence coverage

`derived/channel_performance_weekly.csv.gz` contains **15,333** of the 16,072 channels
(**5,930,225 rows**); `supplier_performance_weekly.csv.gz` has **323,795**. 739 channels
never traded and have no sequence at all. They must be handled explicitly (cold start), not
assumed present.

### 11.8 Phase C introduced two volume deviations and moved a base rate

All three are consequences of the bimodal ordering cadence (§3.2.1) and **none was tuned
away** — the parameters are as designed, and the movements are reported instead.

#### `purchase_orders` document count is ~92% over target

| | Measured | `dataset_structure.md` §1 target | |
|---|---|---|---|
| `purchase_orders` | **479,089** | 249,927 ±20% | **FAIL, +92%** |
| `po_lines` | 859,527 | 899,736 ±20% | ok, −4% |
| `po_line_schedules` | 1,810,601 | 1,799,472 ±20% | ok, +0.6% |
| `po_line_revisions` | 495,610 | 399,883 ±20% | **FAIL, +24%** |

**What is ordered is right; how it is documented is not.** Lines per PO fell from 3.6 to
**1.79** because a high runner releasing weekly generates a small document every week.

The root cause is a modelling shortcut: this generator represents a weekly call-off as a
**discrete PO**, whereas a real ERP represents it as a schedule line against a standing
scheduling agreement — which is exactly what `po_line_schedules` already is, and that table
is within 0.6% of target. The §1 figure of 250k POs was written for a world of discrete POs
only and is not consistent with a population that has weekly releases.

Consolidating each weekly release into one document per (supplier, plant, week) is already
done and is included in the figure above; it is not sufficient on its own. Measured
alternative, for whoever picks this up: **fortnightly** releases
(`high_runner_po_batch_weeks = 2`) cut the PO count by 30% and lift lines/PO to 2.18, at the
cost of weakening the high-runner mode (share ≥13/yr 17.3% → 17.1%, P99 40.9 → 21.2). That
still misses the target by ~11%, so it buys less than the structural fix.

`po_line_revisions` is over for two compounding reasons: more PO lines in the high-runner
population, and the Phase B short-close rows (§11.7.1), which add one row per settling line.

**Recommended fix, not applied here:** model high-runner supply as a standing agreement with
`po_line_schedules` call-offs rather than one PO per release.

#### `PO lines arriving late` fell below its band

12.8% at `full` against a 15–25% target (15.4% before Phase C). This is a **mixture effect,
not a regression in lateness**. Split by cadence class on `mid`, using validate's exact
population:

| Class | Arrived lines | Late |
|---|---|---|
| high runner (≥13 orders/yr) | 78,662 | **10.0%** |
| slow mover | 50,957 | **15.7%** — still inside the band |
| all | 129,619 | 12.3% |

The slow-mover population is unchanged. High runners are genuinely less late: their orders
are smaller and more frequent, so the supplier's finished-goods bank absorbs them and
lead-time stress is lower. Adding that population dilutes the aggregate.

**The gate is now measuring a mixture against a single band, which is the wrong test.** It
should either be evaluated per cadence class or have its target restated for a bimodal
population. Not changed here because Phase E sweeps these rates deliberately.

#### Base rates that moved and stayed in band

| Base rate | Before Phase C | After | Band |
|---|---|---|---|
| PO lines with fill < 1.0 | 9.7% | 9.5% | 8–15% |
| fill-rate spike at 0 | 1.5% | 1.4% | 1–3% |
| supplier-months constrained | 30.3% (small, **over**) | **24.1%** | 20–30% |
| capacity unobservable | 69.7% (small, **under**) | **75.9%** | ≥70% |

The two that were failing at `small`/seed 1 after Phase B are now comfortably inside their
bands.

> **The within-group shortfall correlation is deliberately not in this table.** An earlier
> revision listed it as moving 0.246 → 0.285 and "still under" the 0.30 floor. Both figures
> were produced by the zero-filled estimator that §11.6.1 shows was measuring its own
> imputation, so neither is a base rate that moved — they are two readings of a broken
> instrument. Corrected, the realised correlation is **0.403 at `full`, 0.396 at `mid`,
> inside the 0.30–0.50 target, and gate G10 passes.** §11.6.1 is the only account of the
> correlation in this document; nothing here restates it.

### 11.5 Snapshot horizon is uniform

All 115 snapshots carry `horizon_days = 91`. The 30/60/90-day multi-horizon capacity head
(§6.1) therefore has to derive its 30- and 60-day targets from within the 91-day window,
rather than reading them from separate snapshot rows.

### 11.7 `training_labels.label_censored` — FIXED (was: unusable for the fill-rate task)

**Resolved.** `label_censored` now means "the outcome was not observed by
`label_window_end`", per task. O7 is closed: the fix went into the generator, not a loader
workaround.

**What was wrong.** The column encoded *"did not reach fill 1.0 inside the window"* — a
statement about the label's VALUE, not about whether it was observed. One rule was applied to
five tasks that observe five different things. For `fill_rate` the result was a constant
target: every uncensored row sat at exactly 1.0, which trains to near-zero loss on the first
epoch and looks like success.

| `fill_rate` at `csv_full_seed1` | Before | After |
|---|---|---|
| rows | 1,091,013 | **1,284,907** |
| censored | 23.3% | **12.3%** |
| uncensored `label_value` mean | 1.0000 | **0.9620** |
| uncensored `label_value` **std** | **0.0000** | **0.1609** |
| distinct uncensored values | 11 | **8,661** |
| settled-but-marked-censored | 190,753 (17.5%) | **0** |

**The correct rule, per task.** There is one question — *was the outcome observed by
`label_window_end`?* — but each task observes something different:

| Task | Observed when | Censored, before → after |
|---|---|---|
| `fill_rate` | a final receipt **or** a short-close revision exists | 23.3% → **12.3%** |
| `arrival_week` | a first receipt exists | 7.1% → **8.4%** |
| `capacity_strain` | every contributing PO line has settled | 0.0% → **31.7%** |
| `demand_drift` | the actual was **recorded** by window end | 0.0% → **20.5%** |
| `shortage_qty` | snapshots recorded by window end reach the window's end | 0.0% → **44.3%** |

`arrival_week`'s **rule is unchanged** — it was correct all along, because reaching full fill
was never part of its question. Its rate moved only because the world now contains cancelled
lines, which genuinely never arrive. Its `censor_time` is now the settlement date rather than
the window end for lines that left the risk set early (a competing risk, which the hazard head
needs).

The three tasks that previously reported **0.0% censored** were not censoring-free; they were
not testing observability at all. `demand_drift` compared against actuals regardless of
whether those actuals had been recorded yet — a leak as well as a mislabel.

#### 11.7.1 Settlement is now observable in the emitted data

The rule above is only computable because a settling line always leaves a trace. It did not
before: **5.2% of PO lines settled with no final receipt** — the supplier shipped nothing
(no `grn_lines` row at all), or the line timed out short. Those were indistinguishable from
lines still open.

A line that closes without a final receipt now emits a `po_line_revisions` row cutting `qty`
to what was received, carrying a mandatory `initiated_by` and `reason_code`. Measured on
`csv_full_seed1` (1,169,215 PO lines):

| Settlement route | Lines | Share |
|---|---|---|
| final receipt (`grn_lines.is_final_receipt`) | 1,101,366 | 94.20% |
| short-close revision | 67,849 | 5.80% |
| **neither — settlement unobservable** | **0** | **0.00%** |
| both (double-counted) | 0 | 0.00% |

#### 11.7.2 The cause split, and why `fill_rate` excludes one of them

`reason_code` carries the cause, and the fill-rate label is **not defined for all of them**:

| `reason_code` | `initiated_by` | Lines | % of closes | Realised fill | `fill_rate` label |
|---|---|---|---|---|---|
| `short_close_timeout` | buyer | 25,388 | 37.4% | 0.546 | `qty_accepted / qty_ordered` |
| `supplier_no_ship` | supplier | 19,166 | 28.2% | 0.000 | `0` — a real training row |
| `buyer_cancellation` | buyer | 23,295 | **34.3%** | 0.000 | **none — row not emitted** |

**Buyer cancellations are excluded from the fill-rate training set entirely.** The supplier
was never given the chance to deliver, so scoring the line as `fill = 0` teaches the model
that the supplier failed when Rane changed its mind. The outcome is not censored and not
settled — it is undefined, and the row is dropped. The line still appears under
`arrival_week`, censored at its cancellation date.

Do not infer the cause from `initiated_by`: two of the three are buyer-initiated. The cause
lives in `reason_code`.

Cancelled lines are also excluded from `capacity_strain` and from the `supplier-months
constrained` base rate, for the same reason — counting them as undelivered quantity makes a
supplier look capacity-bound when nothing was asked of it.

**New mechanism toggle** `buyer_cancellation` (default on, `buyer_cancel_share = 2.0%`).
`--disable buyer_cancellation` is the control condition and reproduces the pre-Phase-B base
rates exactly.

#### 11.7.3 A second defect found while fixing this: `demand_drift` was not a forecasting task

`build_training_labels` kept only the **newest** version of each production plan and then
dropped it if it had been recorded after the snapshot. The newest version of a plan is
recorded a median of **24 days before** its target period, so at a snapshot two months out it
does not exist yet — and the whole forward part of the window was silently discarded. Labels
collapsed onto periods that had already elapsed.

The as-of choice is now made per snapshot from the versions visible then, which is what a
planner was actually looking at. `demand_drift` rows at `full`: **230,306 → 424,418 (+84%)**,
with the forward horizons that were missing.

#### 11.7.4 The standing gate

`validate.py` now runs `check_labels` over **every** task, not just the one that broke:

```
for each task:
    assert sd(label_value | uncensored) / |mean| > 0.01  and  distinct >= 5
    assert 0.01 < censored_fraction < 0.60
    report: n, censored %, label mean/sd/min/max, distinct count
```

The spread threshold is **relative**, not `> 0`. The original defect had a standard deviation
of ~1e-5 — not zero — so an absolute test passes it while the target is constant for every
practical purpose. Run against the pre-fix `csv_small_seed1` the gate raises 4 failures,
including `fill_rate: uncensored target varies`; against the fixed data it raises none.

### 11.6 The correlated-failure claim, re-measured against the tail

`model_plan.md` §8 calls the copula "the differentiator" and states that a per-supplier risk
table "understates true exposure badly." That is a claim about the tail, so it was re-tested
against the tail rather than the mean. The short version:

> **The claim holds, weakly, and every earlier number in this section was wrong.**
> Three separate functions imputed `0.0` — meaning *no shortfall* — into unobserved cells of
> a sparse series. Fixing that raised the realised within-group correlation from 0.246 to
> **0.403** (gate G10 now passes, no parameter changed) and the effect on the supply-driven
> tail from "inside the noise band" to **+1.4% at P99**, rising to **+4.0%** at the
> correlation the config targets. Real, tail-concentrated, and small. The previously quoted
> 1.4%-of-total-exposure figure does not reproduce and is unrelated.

Both measurements are on `csv_full_seed1` at seed 1. **Every number in this section is
produced by a named check in `db/prove_claims.py`** — cited per subsection below, and all
re-runnable with:

```
python db/prove_claims.py --check correlation_by_dimension --preset full --gen-seed 1
python db/prove_claims.py csv_full_seed1 --check exposure_copula_vs_independent
python db/prove_claims.py csv_full_seed1 --check null_band
python db/prove_claims.py csv_full_seed1 --check bom_diversification
```

The defaults of those checks *are* the settings quoted here (t₀ = 2024-01-01, 6-week horizon,
20,000 draws, matched seeds across arms), so a bare re-run reproduces the tables rather than
something adjacent to them.

#### 11.6.1 Realised correlation, per grouping dimension — RESOLVED: the estimator was broken, not the generator

`python db/prove_claims.py --check correlation_by_dimension --preset full --gen-seed 1`

**This subsection previously reported two generator defects. Both were artefacts of the
measurement.** The generator realises the correlation it is configured with; the function
measuring it was imputing a value that meant something specific and wrong.

`ground_truth.realised_correlation_matrix()` binned the weekly shortfall series to a monthly
grain and, **for a bin in which a supplier committed no new quantity, imputed `0.0`**. In this
series 0.0 does not mean "unknown" — it means *this supplier had no shortfall*. At `full` only
**71.7%** of supplier-bins carry an observation, so 28.3% of the matrix was fabricated
perfect-performance. Correlations are now **pairwise-complete**: each pair is measured over
the bins where both suppliers were active, and pairs with fewer than 8 overlapping bins return
NaN rather than a guess.

| Grouping dimension | Pairs | Zero-filled (old) | **Pairwise-complete (correct)** | σ·w |
|---|---|---|---|---|
| `supplier_group` | 928 | 0.268 | **0.403** ✅ *in the 0.30–0.50 target* | 0.261 |
| `region` (state) | 47,721 | 0.006 | 0.015 | 0.022 |
| tier-2 parent | 497 | 0.030 | 0.042 | 0.120 |
| `checkpoint` | 24,967 | 0.005 | 0.014 | 0.016 |
| unrelated | 154,889 | 0.004 | 0.013 | — |

**Gate G10 now passes** — 0.403 at `full`, 0.396 at `mid` — with **no shock parameter
changed**. Raising `group_sigma` to "fix" the old 0.246 would have pushed the true correlation
well above target while the estimator continued to under-report it.

**D3 — the scale dependence is fully explained and gone.** It was never a property of the
mechanism; it was the zero-fill bias tracking bin occupancy:

| Preset | Bin occupancy | Zero-filled | Pairwise-complete |
|---|---|---|---|
| `small` | 85.1% | 0.415 | **0.434** |
| `full` | 71.7% | **0.268** | **0.403** |

Under the correct estimator the mechanism is near scale-invariant (0.434 vs 0.403), as a
correlation mechanism should be. Nothing dilutes with supplier count.

**D2 — tier-2 is working, not dead.** The earlier "dead level" verdict compared σ·w linearly
against `supplier_group`'s; correlation scales with **variance share**, i.e. with (σ·w)², so
tier-2's expected share is (0.120/0.261)² = 0.21 of the group's, not 0.46. Measuring on stress
as well as on shortfall separates the two possible causes of a weak figure:

| Dimension | corr(**stress**) | corr(shortfall) | retained | Verdict |
|---|---|---|---|---|
| `supplier_group` | 0.918 | 0.403 | 0.44 | reference |
| tier-2 parent | **0.130** | 0.042 | 0.32 | **working**; small variance share |
| `region` | 0.004 | 0.015 | — | parameterised into irrelevance |
| `checkpoint` | 0.002 | 0.014 | — | parameterised into irrelevance |

The tier-2 shock demonstrably reaches its children — correlation between a supplier's stress
and its parent's shock series is **0.383** over the 181 exposed suppliers, and sibling stress
correlation is 0.130 ≈ 0.383², exactly what an additive shared factor predicts. It is weak at
shortfall level because it is *configured* weak and because weaker signals retain less through
the sigmoid → Bernoulli → Beta link (0.32 vs the group's 0.44). Region and checkpoint carry
σ·w of 0.022 and 0.016 and behave accordingly; their stress correlations match the square of
their shock exposure too. **No level is broken.**

Two secondary measurement bugs were fixed alongside: the tier-2 dimension was keyed on a
supplier's *first* parent, though a supplier can have several (mean 1.12 at `full`) — it now
tests whether two suppliers share **any** parent; and the "dead level" verdict is now gated on
having ≥100 pairs, because at `small` tier-2 has 13 and changes sign between seeds.

**No parameter changes are proposed.** D1–D3 found no structural defect to correct.

#### 11.6.2 Exposure under the copula vs independence

`python db/prove_claims.py csv_full_seed1 --check exposure_copula_vs_independent`

Total production exposure — `min` over each product's BOM, per `data_plan.md` UC6 — at
t₀ = 2024-01-01 over a 6-week horizon: 15,296 open lines, 207 supplier groups, 1,167
product-plants, **20,000 draws with both arms sharing a seed** so the arms differ only in the
copula.

**The headline denominator is the wrong one.** Of the 1,700,532-unit mean shortfall,
**1,465,812 units (86.2%) are still short when every open PO line arrives in full**. That
part is a deterministic plan-vs-supply gap; no supply outcome, correlated or not, can move
it. Quoting a percentage of total exposure divides the real effect by roughly seven. Both
denominators are given below — `supply-driven` is total minus that floor.

**ρ is estimated with the same pairwise-complete fix as §11.6.1.** `estimate_rho` — the
function standing in for what a modelling team would compute from Rane's history — carried
the identical zero-fill bug, filling months in which a supplier had nothing due with a
shortfall of 0.0. Corrected, the estimate rises from **0.139 to 0.190**.

| ρ | statistic | total exposure | gap | supply-driven | **gap** |
|---|---|---|---|---|---|
| **0.190** — estimated from history before t₀, what a modelling team would use | mean | 1,700,532 | +0.08% | 234,720 | +0.61% |
| | P90 | 1,749,358 | +0.14% | 283,546 | +0.88% |
| | P95 | 1,764,390 | +0.22% | 298,578 | +1.30% |
| | **P99** | 1,794,366 | +0.26% | 328,554 | **+1.40%** |
| **0.350** — the level `config.ShockParams` targets | mean | 1,700,532 | +0.30% | 234,720 | +2.18% |
| | P90 | 1,749,358 | +0.52% | 283,546 | +3.21% |
| | P95 | 1,764,390 | +0.61% | 298,578 | +3.59% |
| | **P99** | 1,794,366 | +0.72% | 328,554 | **+3.96%** |
| **0.900** — control, far above anything this spec targets | mean | 1,700,532 | +4.31% | 234,720 | +31.3% |
| | P90 | 1,749,358 | +7.07% | 283,546 | +43.6% |
| | P95 | 1,764,390 | +8.03% | 298,578 | +47.5% |
| | **P99** | 1,794,366 | +9.59% | 328,554 | **+52.4%** |

**The null band** (`--check null_band`)**.** Running the independent arm against
*itself* under four different seed pairs — a comparison with a known-zero effect —
bounds what this harness reports when nothing is there. On the supply-driven
denominator at 20,000 draws: mean −0.30…+0.20%, P90 −0.50…+0.16%, P95 −0.63…+0.09%,
P99 −0.85…+0.69%.

Read against that band:

- **At ρ = 0.190 the effect is now detectable, and it was not before.** P95 (+1.30%) and P99
  (+1.40%) sit clearly above the band; the mean (+0.61%) does not. It is a small effect —
  roughly twice the noise floor — but it is no longer indistinguishable from zero, and it has
  the right shape: growing monotonically from mean to tail.
- **At ρ = 0.35 the effect is unambiguous** — +3.96% at P99, five times the band.
- **The ρ = 0.90 control** returns +52.4% at P99 and reproduces the expected monotone
  mean→tail signature, confirming the harness sees the effect when it is present.

Rupee exposure tracks units to within 0.1 point throughout.

#### 11.6.3 The previously quoted 1.4% does not reproduce

The figures this section carried before — "understates P90 production exposure by 1.4%
(1.7% at the generator's own ρ)" — could not be reproduced from that code on this dataset.
The previous `prove_claims.py`, run at its own shipped defaults (t₀ = 2024-01-01, 6-week
horizon, **1,333 draws**), reported **P90 +0.01%** at the estimated ρ and **+0.48%** at
ρ = 0.35 — neither of which is 1.4% either. At 1,333 draws the independent-vs-independent
null band on total exposure is already ±0.35%, so no figure from that configuration carried
information. **The 1.4% should not be quoted again.**

§11.6.2 now reports **+1.40%** as the supply-driven P99 gap. That is a coincidence of digits
and nothing more: a different statistic (P99, not P90), on a different denominator
(supply-driven, not total), from a different dataset (post-Phase-C), with a corrected ρ. The
comparable total-exposure P90 figure today is **+0.14%**.

That script has since been rewritten (see the command list above). Four things it got wrong
are now structural properties of the harness rather than caveats a reader had to know: the
default draw count is 20,000 rather than 1,333; both arms share an RNG seed so they differ
only by the copula; every exposure figure is reported against the supply-driven denominator
alongside the total; and correlations are pairwise-complete rather than zero-filled
(§11.6.1). The `null_band` check exists so that no future figure from this file can be quoted
without its noise floor.

The accompanying diagnosis was also wrong on its own numbers. It attributed the null to BOM
supplier diversification, citing "~19 distinct supplier groups, the largest supplying 11.5%
of the BOM." Measured with `--check bom_diversification` on the supply that is actually
open in the window — the supply the `min` operates on — a product-plant draws on a
median of **14 supplier groups with the largest holding 30.0% of open quantity**, far
more concentrated than the old figure claimed. Diversification is not what suppresses
the effect. The **86.2% deterministic floor** in §11.6.2 is, together with the fact that
`supplier_group` is the only shock level carrying material weight (§11.6.1).

#### 11.6.4 What this changes

**The framing is "correctly prices the bad weeks", and the dataset can now demonstrate it —
weakly.** Every earlier conclusion in this section was distorted by a single class of bug:
three separate functions imputed `0.0` — a value meaning *no shortfall* — into cells that
were merely unobserved. Fixing that in all three moved the picture substantially:

| | Before | After |
|---|---|---|
| realised within-group correlation (`full`) | 0.246 ❌ | **0.403** ✅ |
| ρ estimable from emitted history | 0.139 | **0.190** |
| supply-driven P99 exposure gap at that ρ | +0.44% (inside noise) | **+1.40%** (≈2× noise floor) |
| gate G10 | **FAIL** | **PASS** (0.403 `full`, 0.396 `mid`) |

The mechanism is correct, it realises the correlation it was configured with, and the effect
is real and tail-concentrated. It is also **small**: 1.4% of the supply-controlled component
of exposure at P99, which is 0.26% of total exposure. Nobody should call that "understates
true exposure badly."

**No generator parameter was changed, and none should be.** The pre-Phase-D plan proposed
raising σ to close a gap that did not exist; doing so would have pushed the true correlation
above the 0.30–0.50 target while the broken estimator continued to report it as low. This is
the concrete reason the D1–D3 diagnosis had to precede any tuning.

What still limits the magnitude is structural and unrelated to correlation:

- **86.2% of production exposure is deterministic** at a 6-week horizon — short even if every
  open line arrives in full. Correlation can only ever move the other 13.8%.
- Only `supplier_group` carries meaningful weight (σ·w = 0.261). `region` (0.022) and
  `checkpoint` (0.016) are parameterised into irrelevance and tier-2 (0.120) is deliberately
  secondary, so a product's BOM is exposed to essentially one correlated failure channel.

**Consequences for this benchmark:**

- Gate G10 validates that the simulation reproduces the correlation it was given, and now
  does so correctly. It still does **not** validate the business claim; those are different
  statements and this section is the only place the second one is tested.
- Results from this dataset **may** now be used to support the "independence understates
  tail exposure" argument, with the magnitude stated honestly: **+1.4% of supply-driven P99
  exposure at the realised ρ, +4.0% at the configured ρ.** Quote the supply-driven
  denominator and the null band alongside, never the total-exposure percentage alone.
- `model_plan.md` §8's "understates true exposure badly" is not supported at any ρ this spec
  targets and should be toned to match.
- Any future measurement on a sparse series in this project should be checked for the same
  zero-fill bug before its result is believed. It was present in three independent places and
  in every case biased the answer toward "no effect".

See `db/README.md` § Known gaps.

---

## Open questions

Collected from above. Each blocks a specific phase.

| # | Question | Blocks |
|---|---|---|
| O1 | `supplier_group` node features — embedding, aggregate, or drop the node type? | Phase 4 |
| O2 | Short-sequence policy — down-weight the loss, or exclude nodes with < 52 real weeks? | Phase 2 |
| O3 | Capacity-strain target — observed fill rate (as labelled) or utilisation (from `revealed_capacity_monthly`)? | Phase 5 |
| O4 | Definition of a "significant capacity event" for the lead-time recall metric. **Needs Rane.** | Phase 8 |
| O5 | Shortage precision/recall targets on critical parts. **Needs Rane.** | Phase 8 |
| O6 | Readout depth per task — must be measured, not inherited from v3 | Phase 5, gate G7 |
| ~~O7~~ | ~~Fix `label_censored` in the generator, or work around it in the loader?~~ **CLOSED** — fixed in the generator; see §11.7. | — |
