# ChainPilot × Rane — Data Plan

**How each use case consumes each file and column, technically and mathematically.**

Companion to `dataset_structure.md` (what the data is) and `model_plan.md` (how the models
work). This document is the bridge: for every use case, exactly which tables are read, which
columns become which features, and the equations that turn them into an output.

---

## Notation used throughout

| Symbol | Meaning |
|---|---|
| $t_0$ | The as-of date. Everything the model sees must satisfy `recorded_ts ≤ t₀` |
| $h$ | Horizon in days: how far ahead we're predicting |
| $c$ | A sourcing channel = (supplier $s$, part $p$, plant $l$) |
| $w$ | A week index in the forecast horizon |
| $f$ | Fill rate $\in [0,1]$ |
| $\mathbb{1}[\cdot]$ | Indicator: 1 if true, 0 otherwise |

**The as-of rule, formally.** For any feature $x$ computed at $t_0$:

$$x(t_0) = g\big(\{\text{rows} : \texttt{recorded\_ts} \le t_0\}\big)$$

Never `event_ts ≤ t₀`. An event that happened on the 5th but was entered on the 12th was
not knowable on the 8th, and training as though it were is the most common and most
damaging error in this class of system.

For effective-dated master and relationship data (BOM, sourcing channels, allocation,
part-plant policy), the equivalent selection is:

```sql
WHERE effective_from <= t0
  AND (effective_to IS NULL OR effective_to > t0)
```

**Regime handling.** Every training row carries `calendar.regime_flag`. Rows flagged
`covid` (2020-03 to 2021-09, confirm the exact window with Rane) are excluded from the
primary training window by default and available as an explicit robustness check. The
default modelling window is **2019–2025**; 2016–2018 is held out as a sanity check rather
than trained on, because entity drift over ten years means a 2016 channel and a 2025 channel
sharing an ID may not be the same thing. See `dataset_structure.md` §1.

**Coverage masking.** Where `dataset_coverage.csv` shows a table starting later than the
training window (acknowledgements often begin at an ERP migration), the derived features
built from it are **masked**, not zero-filled. A zero in `ack_gap_ratio` reads as *"the
supplier acknowledged nothing"* — the opposite of *"we did not record it."*

---

## Use case 1 — Part demand

### Files read

| File | Role |
|---|---|
| `production_plan.csv` | The driver |
| `production_actual.csv` | The drift target |
| `bom.csv` | The explosion structure |
| `products.csv` | Family bucketing |
| `part_plant.csv` | Which plants consume which parts |
| `calendar.csv` | Working days, shutdowns |

### Step 1 — BOM explosion

**Columns used:** `production_plan.planned_qty`, `.product_id`, `.plant_id`,
`.target_period`, `.plan_date`; `bom.qty_per_unit`, `.scrap_factor`, `.effective_from`,
`.effective_to`, `.bom_level`, `.parent_part_id`.

For a single-level BOM:

$$
R(p, l, \tau) \;=\; \sum_{\text{products } j} \text{planned\_qty}(j, l, \tau) \times \frac{q_{jp}(\tau)}{1 - \sigma_{p}}
$$

where $q_{jp}(\tau)$ is `qty_per_unit` for the (product $j$, part $p$) row whose validity
window contains $\tau$, and $\sigma_p$ is `scrap_factor`.

**The validity window is not optional.** Selecting the BOM row requires:

```sql
WHERE effective_from <= target_period
  AND (effective_to IS NULL OR effective_to > target_period)
```

Using today's BOM to explode a 2019 plan silently mis-states every historical demand figure,
which then corrupts the drift model that trains on it.

**Multi-level BOMs.** Let $\mathbf{B}$ be the sparse matrix with $B_{jp} = q_{jp}$. For an
$n$-level BOM, total requirement is the transitive closure:

$$
\mathbf{B}^{*} = \mathbf{B} + \mathbf{B}^2 + \dots + \mathbf{B}^n = (\mathbf{I} - \mathbf{B})^{-1} - \mathbf{I}
$$

In practice: topological sort of the BOM DAG by `bom_level`, then iterate level by level.
Cheaper than the inverse and it handles phantom assemblies (`is_phantom = true` passes
through without stocking).

> **Check first.** Rane's ERP almost certainly runs MRP already and outputs gross
> requirements. Read that table instead of rebuilding it, and use our explosion only to
> validate theirs.

### Step 2 — Plan drift

**Columns used:** `production_plan.planned_qty`, `.plan_date`, `.target_period`,
`.plan_version`, `.is_firm`; `production_actual.actual_qty`, `.scrapped_qty`;
`products.product_family`; `calendar.month_of_year`, `.is_shutdown`.

Define the drift ratio for one (product, plant, period) observed at horizon $h$:

$$
\rho(j, l, \tau \mid h) \;=\; \frac{\text{actual\_qty}(j,l,\tau)}{\text{planned\_qty}(j,l,\tau \mid \text{plan made at } \tau - h)}
$$

Horizon is derived, not stored:

$$h = \tau - \texttt{plan\_date}$$

**Baseline model — empirical quantiles.** Bucket historical $\rho$ by
(horizon band, plant, product family) and read off percentiles directly:

$$
\hat{\rho}_{q}(h, l, F) = \text{Quantile}_q\Big(\big\{\rho(j,l,\tau \mid h) : j \in F,\ \tau < t_0\big\}\Big)
$$

With 7 years and monthly re-planning there are roughly 84 observations per
(product, plant) and thousands per (family, plant, horizon-band) bucket. That is enough for
stable quantiles and needs no model at all.

**Escalation — LightGBM with pinball loss**, only if buckets are thin:

$$
\mathcal{L}_{\text{pinball}}(\rho, \hat\rho_q) = \max\big(q(\rho - \hat\rho_q),\ (q-1)(\rho - \hat\rho_q)\big)
$$

Features: $h$, plant one-hot, product family one-hot, `month_of_year`,
`plan_version`, `is_firm`, this product's own trailing drift mean and variance,
`is_shutdown` in the target period.

### Output

$$
\text{required}_{q}(p, l, w) = R(p,l,w) \times \hat\rho_{q}(h, l, F)
$$

Written to `part_demand_weekly.csv` at $q = 0.50$ and $q = 0.90$. The simulation consumes
P90 — being wrong about demand in the conservative direction is what safety stock is for.

---

## Use case 2 — Supplier capacity

### Files read

| File | Role |
|---|---|
| `grn_lines.csv` | Delivered quantity — the observable |
| `po_lines.csv` | Ordered quantity — the load |
| `supplier_capacity.csv` | Stated figure (weak prior) |
| `sourcing_channels.csv` | Contracted lead time baseline |
| `supplier_allocation.csv` | Planned future load |
| `part_demand_weekly.csv` **[DERIVED]** | Future load from UC1 |
| `channel_performance_weekly.csv` **[DERIVED]** | The input sequence |
| `suppliers.csv`, `supplier_upstream.csv` | Graph structure |
| `calendar.csv` | Their seasonality |

### Step 1 — Revealed capacity envelope (not ML)

**Columns:** `grn_lines.qty_received`, `.event_ts`; `po_lines.qty_ordered`;
`channel_performance_weekly.fill_rate`, `.lead_time_ratio`.

Monthly delivered quantity per supplier-part:

$$
D(s,p,m) = \sum_{\text{grn lines in month } m} \texttt{qty\_received}
$$

**The censoring problem.** What you observe is

$$
D(s,p,m) = \min\big(K(s,p,m),\ O(s,p,m)\big)
$$

where $K$ is true capacity and $O$ is what you ordered. In any month where $O < K$, the
observation tells you nothing about $K$ — it only tells you what you asked for.

**Only constrained months carry capacity evidence.** Flag them:

$$
\text{constrained}(s,p,m) = \mathbb{1}\Big[f(s,p,m) < 1 \ \ \text{OR}\ \ \frac{\text{LT}_{\text{actual}}}{\text{LT}_{\text{contracted}}} > 1 + \delta\Big]
$$

with $\delta \approx 0.15$. Then:

$$
\hat{K}_{\text{revealed}}(s,p,t_0) = \max\Big(\max_{m \in [t_0-12, t_0]} D(s,p,m),\ \ \text{P95}\big\{D(s,p,m) : \text{constrained}\big\}\Big)
$$

Written to `revealed_capacity_monthly.csv` with `evidence_strength` = the fraction of the
trailing 12 months that were constrained. **Low evidence strength must widen the output
band** — it is the honest signal that we are extrapolating.

### Step 2 — Strain model (the ML part)

**The key feature.** Load relative to demonstrated ability:

$$
\lambda(s,p,m) = \frac{O(s,p,m)}{\max_{m' \in [m-12, m-1]} D(s,p,m')}
$$

This one column places a supplier on the strain curve and carries most of the signal.

**Targets — both observable, both from GRN data:**

$$
y_{\text{fill}}(c, m) = \frac{\sum \texttt{qty\_received}}{\sum \texttt{qty\_ordered}}, \qquad
y_{\text{lt}}(c, m) = \frac{\text{LT}_{\text{actual}}}{\text{LT}_{\text{contracted}}}
$$

**Input sequence** — from `channel_performance_weekly.csv`, the last $K = 52$ weeks:

$$
X_c = \big[\mathbf{x}_c(t_0 - 51), \dots, \mathbf{x}_c(t_0)\big] \in \mathbb{R}^{52 \times d}
$$

Each $\mathbf{x}_c(w)$ carries: `qty_ordered`, `qty_received`, `is_active_week`,
`fill_rate`, `fill_rate_last4/_last13/_last52`, `lead_time_actual_days`, `lead_time_ratio`,
`otd_rate_last13`, `ack_gap_ratio`, `revision_count`, `load_ratio`, `active_weeks_in_52`,
`days_since_last_short`, `reporting_lag_days`, `weeks_since_last_activity`,
`weeks_since_last_receipt`, plus `month_of_year` and `is_shutdown` from `calendar`.

Two things about this list are not cosmetic:

- **`staleness_days` is gone, replaced by `reporting_lag_days` and
  `weeks_since_last_activity`.** It conflated two opposite signals — how late a record was
  posted (on a week with activity) and how long the channel had been silent (on a week
  without). Pooled it read P50 12 / P90 838 days, and that whole tail was inactivity; the
  genuine posting lag is P50 1.3 / P90 5.7 / max 96. (An earlier revision quoted max 6 here;
  that ceiling was a builder artefact, not a property of the data. Fixed.) A trust gate can
  only decay on the first;
  inactivity is a real feature and must not be treated as low confidence, because a quiet
  channel is giving you a true reading, not a stale one.
- **The `_lastN` columns average the last N *active* weeks, not calendar weeks.** At this
  ordering cadence `fill_rate_last13` spans a median of 85 calendar weeks. `active_weeks_in_52`
  carries the actual time scale.

Every instantaneous column is NULL, never 0, on a week with no activity — a `fill_rate` of 0
is a total delivery failure and an empty week is no information. `is_active_week` is both the
sequence mask and a feature. See `dataset_structure.md` §`channel_performance_weekly.csv`.

**Head output** — quantile utilisation at three horizons:

$$
\hat{u}_q(s, t_0 + h) \quad\text{for } q \in \{0.1, 0.5, 0.9\},\ h \in \{30, 60, 90\}
$$

**Days-to-exceed is derived, not predicted.** Linear interpolation on the P50 curve:

$$
\text{DTE} = \min\{h : \hat{u}_{0.5}(h) \ge 1\}, \quad \text{interpolated between the 30/60/90 grid}
$$

### The graph's job here

Thin channels borrow from thick ones. The `supplier_id`, `supplier_group_id`,
`part_category`, `state` and `supplier_upstream` links let a channel with 6 observations
pool with the ~400 observations of its carrier-lane parent and the ~5,000 of its
component-class peers. A per-supplier statistic cannot do this; it simply has 6 points.

---

## Use case 3 — Fill rate

### Files read

| File | Role |
|---|---|
| `po_lines.csv` | The prediction unit |
| `grn_lines.csv` | The label |
| `supplier_acknowledgements.csv` | Strongest leading feature |
| `po_line_revisions.csv` | Promise-date reconstruction |
| `quality_inspections.csv` | Received-then-rejected is short too |
| `channel_performance_weekly.csv` **[DERIVED]** | Channel history |
| `sourcing_channels.csv` | Channel identity and lead time |

### The label

$$
f(\ell) = \frac{\sum_{\text{grn lines for } \ell} \texttt{qty\_accepted}}{\texttt{qty\_ordered}(\ell)}
$$

Note `qty_accepted` from `quality_inspections`, **not** `qty_received` — a part received and
rejected does not feed the line. Where inspection data is missing, fall back to
`qty_received` and flag the channel.

### Instance features on the PO line

| Feature | Source columns | Why |
|---|---|---|
| `qty_ordered` | `po_lines.qty_ordered` | Larger orders fill worse |
| `qty_relative` | $\dfrac{\texttt{qty\_ordered}}{\text{median}(\texttt{qty\_ordered})_{c, 52w}}$ | An unusually big order is a risk regardless of the supplier's average |
| `days_to_promise` | `current_promise_date` $- t_0$ | Runway remaining |
| `days_since_order` | $t_0 -$ `created_ts` | Age in the system |
| `ack_gap_ratio` | $\dfrac{\texttt{ack\_qty}}{\texttt{qty\_ordered}}$ | **Strongest single feature.** 800 against 1,000 is visible weeks early |
| `ack_present` | `ack_id IS NOT NULL` | No acknowledgement at all is itself a signal |
| `days_since_ack` | $t_0 -$ `ack.event_ts` | Stale acknowledgements decay in value |
| `revision_count` | count of `po_line_revisions` | Churn predicts trouble |
| `supplier_initiated_revisions` | count where `initiated_by='supplier'` | **Buyer-initiated cuts are not supplier failures** — do not pool them |
| `promise_slip_days` | `current_promise_date` $-$ `original_promise_date` | Already-moved dates move again |
| `price_deviation` | vs `part_costs` trailing mean | Price pressure precedes supply pressure |
| `is_emergency` | `purchase_orders.po_type = 'emergency'` | Different population entirely |

**Reconstructing `current_promise_date` as of $t_0$** — take the latest revision recorded by
$t_0$, not the field's present value:

```sql
SELECT new_value FROM po_line_revisions
WHERE po_line_id = ? AND field_changed = 'promise_date'
  AND recorded_ts <= t0
ORDER BY recorded_ts DESC LIMIT 1
```

### Head — discrete CDF

Bin $f$ into $B = 20$ bins with **dedicated bins for the point masses**:

$$
b_0 = \{0\}, \quad b_1 = (0, 0.05], \ \dots,\ b_{19} = \{1\}
$$

Model outputs $\pi_b = P(f \in b)$. Loss is CRPS over the cumulative distribution:

$$
\mathcal{L}_{\text{CRPS}} = \sum_{b=0}^{B-1}\Big(\hat{F}(b) - \mathbb{1}[f_{\text{true}} \le b]\Big)^2, \qquad \hat{F}(b) = \sum_{b' \le b} \pi_{b'}
$$

CRPS is proper — it rewards putting mass near the truth rather than only on it, which
ordinary cross-entropy over bins does not.

### Outputs consumed downstream

$$
\mathbb{E}[f] = \sum_b \pi_b \cdot \text{mid}(b), \qquad
\text{expected shortfall} = \texttt{qty\_ordered} \times (1 - \mathbb{E}[f])
$$

Units, not a probability — they add correctly across the rollup in UC6.

---

## Use case 4 — Arrival timing

### Files read

`po_lines.csv`, `grn_lines.csv`, `asn.csv`, `po_line_schedules.csv`,
`channel_performance_weekly.csv`, `calendar.csv`.

### The censoring problem, precisely

For a PO line, the outcome is the week of first receipt:

$$
T(\ell) = \left\lceil \frac{\texttt{grn.event\_ts} - \texttt{original\_promise\_date}}{7} \right\rceil
$$

For lines still open at $t_0$, $T$ is unknown — we only know $T > w_{\text{current}}$. That
is **right-censoring**. Dropping open lines keeps only the ones that already arrived, which
skews the training set toward fast suppliers and makes every forecast optimistic.

A line that **settled without ever receiving** — the supplier withdrew, or the buyer
cancelled — is a *competing risk*, not an ordinary open line. It left the risk set on its
settlement date and can never arrive, so censor it at that date rather than at the window
end; censoring it at $w_{\text{end}}$ asserts it was still pending for weeks after it was
gone. `training_labels.arrival_week` already does this and puts the exit date in
`censor_time`.

### Discrete-time hazard

Define the conditional probability of arriving in week $w$ given it hasn't yet:

$$
\lambda_w = P(T = w \mid T \ge w)
$$

The model emits $\hat\lambda_w$ for $w = 1 \dots 12$ through independent sigmoids. Survival
and arrival probability follow:

$$
S_w = \prod_{k<w}(1 - \hat\lambda_k), \qquad P(T = w) = \hat\lambda_w \cdot S_w
$$

**The loss uses censored rows correctly:**

$$
\mathcal{L} = -\sum_{\ell} \left[ \sum_{w=1}^{\min(T_\ell, C_\ell)} \Big( \mathbb{1}[T_\ell = w]\log\hat\lambda_w + \mathbb{1}[T_\ell > w]\log(1-\hat\lambda_w) \Big) \right]
$$

An open line contributes $\log(1 - \hat\lambda_w)$ for every week it has survived so far —
real information, not a discarded row.

### Timing-specific features

| Feature | Source | Why |
|---|---|---|
| `asn_present` | `asn.asn_id IS NOT NULL` | Dispatched goods have a much tighter window |
| `days_since_dispatch` | $t_0 -$ `asn.dispatch_ts` | Only meaningful once dispatched |
| `transit_elapsed_ratio` | $\dfrac{t_0 - \texttt{dispatch\_ts}}{\text{standard\_transit\_days}}$ | **15 days elapsed means nothing on a 40-day sea lane and disaster on a 3-day road lane.** Currently the model cannot tell them apart |
| `days_overdue` | $\max(0, t_0 - \texttt{promise\_date})$ | Separate signed feature from `days_to_promise` |
| `schedule_adherence_13w` | from `po_line_schedules` vs `grn_lines` | Do they hit their own committed drops? |
| `working_days_to_promise` | via `calendar` | Calendar days overstate runway across a shutdown |

### Combined output

$$
\text{arriving}(p, l, w) = \sum_{\ell \,\in\, \text{open POs}} \texttt{qty\_ordered}(\ell) \cdot \mathbb{E}[f_\ell] \cdot P(T_\ell = w)
$$

---

## Use case 5 — Part shortage

### Files read

`inventory_snapshots.csv`, `inventory_transactions.csv`, `part_plant.csv`,
`part_demand_weekly.csv` **[D]**, plus the model outputs of UC2, UC3, UC4.
Labels from `shortage_events.csv`, `line_stop_events.csv`, `expedite_events.csv`.

### Opening position at $t_0$

Materialised in `inventory_position_weekly.csv` **[DERIVED]** rather than recomputed per
run, so the as-of discipline is enforced once.

**Columns:** `inventory_snapshots.qty_on_hand`, `.qty_blocked`, `.qty_reserved`,
`.snapshot_date`, `.recorded_ts`; `inventory_transactions.qty`, `.txn_type`,
`.from_plant_id`.

$$
I_0(p,l) = \texttt{qty\_on\_hand} - \texttt{qty\_blocked} - \texttt{qty\_reserved}
$$

If the newest snapshot with `recorded_ts ≤ t₀` is older than $t_0$, roll it forward with
`inventory_transactions` — and record the gap as `inventory_position_weekly.staleness_days`,
which feeds the confidence band. (This is **not** the `staleness_days` that was removed from
`channel_performance_weekly`: this one is the age of a single inventory reading, bounded at
P50 14 / P90 28 / max 126 days, and carries one meaning only.)

**Watch the inter-plant transfers.** `inventory_transactions.from_plant_id` marks stock
pulled from another plant. These are a *hidden consumption stream* at the source plant and
a hidden receipt at the destination, and they are frequently how a shortage gets quietly
resolved without ever appearing in `shortage_events`. Materialise them as
`inter_plant_in_4w` / `inter_plant_out_4w` — a part with heavy outbound transfers is under
stress even when its own stock looks healthy.

### The recursion

$$
\begin{aligned}
I_w &= I_{w-1} + A_w - C_w \\[4pt]
A_w &= \underbrace{\sum_{\ell \in \text{open}} q_\ell \cdot f_\ell \cdot \mathbb{1}[T_\ell = w]}_{\text{UC3} \times \text{UC4}} \;+\; \underbrace{\min\big(\text{planned}_w,\ \hat{K}_w\big)}_{\text{capped by UC2}} \\[4pt]
C_w &= \text{required}(p,l,w) \quad \text{from UC1} \\[4pt]
\text{Short}_w &= \max\big(0,\ -(I_w - \text{safety\_stock})\big)
\end{aligned}
$$

Note shortage is measured against **safety stock**, not zero — hitting zero is already a
line stop, and the warning has to come earlier than that.

### Monte Carlo

Run $N = 1000$ passes. On each pass $n$, draw:

$$
f_\ell^{(n)} \sim \boldsymbol{\pi}_\ell, \qquad
T_\ell^{(n)} \sim P(T_\ell = \cdot), \qquad
C_w^{(n)} \sim \text{Beta-fit}(\hat\rho_{0.1}, \hat\rho_{0.5}, \hat\rho_{0.9})
$$

Outputs:

$$
P(\text{shortage}) = \frac{1}{N}\sum_n \mathbb{1}\Big[\textstyle\max_w \text{Short}_w^{(n)} > 0\Big], \qquad
\mathbb{E}[\text{Short}] = \frac{1}{N}\sum_n \max_w \text{Short}_w^{(n)}
$$

$$
\text{earliest risk date} = \text{Quantile}_{0.1}\Big(\big\{\min\{w : \text{Short}_w^{(n)} > 0\}\big\}_{n=1}^{N}\Big)
$$

**Keep all $N$ sample paths.** UC6 needs them — see the P90 warning there.

### Sampling correlation, not independence

Naive independent sampling understates risk badly. Two channels sharing a
`supplier_group_id`, an `upstream_supplier_id` or a `via_checkpoint` fail together. Draw a
shared latent shock per pass:

$$
z^{(n)}_{g} \sim \mathcal{N}(0,1) \ \text{per group } g, \qquad
f^{(n)}_\ell = \Pi^{-1}_\ell\Big(\Phi\big(\rho\, z^{(n)}_{g(\ell)} + \sqrt{1-\rho^2}\, \varepsilon^{(n)}_\ell\big)\Big)
$$

A Gaussian copula with correlation $\rho$ estimated from historical co-occurrence of
shortfalls within each group. **This is the single most important modelling decision in the
simulation**, and it's what a per-supplier risk table structurally cannot represent.

### The GNN cross-check

Train a direct head on `shortage_events` and `expedite_events` — expedites are far more
numerous (840 vs 126 line stops over 7 years) and are near-misses, which makes them the
better label. Use it as a **diagnostic**: where it disagrees sharply with the simulation,
something is missing from the data — undocumented consumption, an inter-plant pull
(`inventory_transactions.from_plant_id`), an informal substitution.

---

## Use case 6 — Production exposure

### Files read

Simulation output from UC5, `bom.csv`, `production_plan.csv`, `product_economics.csv`,
`customers.csv`, `plants.csv`, `business_units.csv`.

### Upward propagation

Part shortage limits how many of each product can be built:

$$
\text{buildable}(j, l, w) = \min_{p \,\in\, \text{BOM}(j)} \left\lfloor \frac{I_w(p,l)}{q_{jp}} \right\rfloor
$$

$$
\text{units at risk}(j,l,w) = \max\big(0,\ \text{planned}(j,l,w) - \text{buildable}(j,l,w)\big)
$$

The $\min$ matters: a product is blocked by its *scarcest* component, not the sum of its
shortages.

### Value

$$
\text{₹ at risk} = \sum_{j,l,w} \text{units at risk} \times \big(\texttt{contribution\_margin\_inr} + \texttt{penalty\_per\_unit\_inr}\big)
$$

### ⚠️ The aggregation trap

**Never sum P90s.**

$$
\text{P90}\Big(\sum_p \text{Short}_p\Big) \;\ne\; \sum_p \text{P90}(\text{Short}_p)
$$

The right side assumes every part goes wrong simultaneously and overstates exposure badly.
Carry the $N$ simulation sample paths up through every level of the rollup and take the
percentile **at the top**:

$$
\text{₹ at risk}_{P90} = \text{Quantile}_{0.9}\left(\left\{\sum_{j,l,w} \text{units at risk}^{(n)} \times v_j \right\}_{n=1}^{N}\right)
$$

This is also why every upstream model must output **quantities in units**. Units aggregate
correctly across four levels of rollup; probabilities do not.

---

## Use case 7 — Delivery schedule

### Files read

`part_demand_weekly.csv` **[D]**, `part_plant.csv`, `supplier_contracts.csv`,
`sourcing_channels.csv`, `logistics_lanes.csv`, `calendar.csv`.

### The LP

**Decision:** $x_w$ = quantity to receive in week $w$.

$$
\min_{x} \ \sum_w \Big( c_{\text{hold}} \cdot I_w \;+\; c_{\text{order}} \cdot y_w \;+\; c_{\text{freight}} \cdot \lceil x_w / \text{truck\_cap}\rceil \Big)
$$

subject to

$$
\begin{aligned}
I_w &= I_{w-1} + x_w - C_w && \text{stock balance} \\
I_w &\ge \text{safety\_stock}(p,l) && \texttt{part\_plant.safety\_stock\_qty} \\
x_w &\ge \text{MOQ} \cdot y_w && \texttt{supplier\_contracts.moq} \\
x_w &\equiv 0 \pmod{\text{lot\_size}} && \texttt{supplier\_contracts.lot\_size} \\
\textstyle\sum_w x_w &= \text{monthly requirement} && \text{from UC1} \\
y_w &\in \{0,1\}
\end{aligned}
$$

The binary $y_w$ (do we order at all in week $w$) is what makes this integer rather than a
plain LP. Small enough to solve exactly in milliseconds.

---

## Use case 8 — Allocation / what-if

### Files read

`supplier_allocation.csv`, `alternate_sources.csv`, `sourcing_channels.csv`,
`supplier_contracts.csv`, `tooling.csv`, `part_costs.csv`,
`revealed_capacity_monthly.csv` **[D]**, plus UC2 capacity and UC5 simulation.

### Phase 1 — scenario enumeration

Take 5 candidate splits, re-run UC5's Monte Carlo under each, rank:

$$
\text{score}(\mathbf{a}) = c_{\text{short}} \cdot \mathbb{E}[\text{Short} \mid \mathbf{a}] \;+\; \sum_s a_s \cdot Q \cdot \text{unit\_cost}_s
$$

Simple, matches the deck's comparison table exactly, and every number is explainable line by
line. This is what ships for the POC.

### Phase 2 — MILP

**Decision:** $x_{s,p,m}$ = quantity allocated.

$$
\min_{x} \ \underbrace{c_{\text{short}} \cdot \mathbb{E}[\text{Short}(x)]}_{\text{piecewise-linear from simulation}} + \sum_{s,p,m} \kappa_{sp}\, x_{spm} + \sum_{s,p,m} \gamma \big|x_{spm} - x_{sp,m-1}\big|
$$

| Constraint | Formal | Source columns |
|---|---|---|
| Demand met | $\sum_s x_{spm} = R(p,m)$ | UC1 |
| Capacity | $x_{spm} \le \hat{K}_{spm}$ | UC2 |
| Qualification | $x_{spm} = 0$ if not approved | `sourcing_channels.is_approved`, `alternate_sources.qualification_status` |
| Ramp | $x_{spm} \le x_{sp,m-1}(1 + r_s)$ | `alternate_sources.ramp_rate_pct_per_month` |
| MOQ / lot | $x_{spm} \ge \text{MOQ}\cdot y$, $x \equiv 0 \bmod L$ | `supplier_contracts` |
| Contract minimum | $\sum_m x_{spm} \ge V^{\min}_{sp}$ | `supplier_contracts.min_volume_commitment` |
| Tooling | $x_{spm} = 0$ if tool not transferable and held elsewhere | `tooling.is_transferable`, `.duplicate_exists` |

**The constraints are the product.** Anyone can write a solver that says "move 40% to
Supplier B." Encoding *why you often can't* is what makes it usable in a plant.

`supplier_allocation.csv` also has research value beyond being a decision variable: past
allocation changes with `change_reason` are **natural experiments** — the closest thing to
interventional data this project will ever get.

---

## Use case 9 — Quality risk (Phase 2)

### Files read

`quality_inspections.csv`, `supplier_quality_ppm.csv`, `supplier_audits.csv`,
`grn_lines.csv`, `parts.csv`.

$$
\text{PPM}(s, p, m) = \frac{\texttt{qty\_rejected}}{\texttt{qty\_inspected}} \times 10^6
$$

LightGBM on rolling PPM (3/6/12 month), trend slope, audit score, major/minor NCs, days
since last audit, part category, and process change flags. Target: $P(\text{reject rate} >
\text{threshold in next 90 days})$.

Folds into UC5 by reducing effective fill rate:
$f_{\text{effective}} = f \times (1 - \widehat{\text{reject rate}})$.

---

## Cross-cutting: the feature-store build

`channel_performance_weekly.csv` is the input sequence for all three learned models. It is
built once and read by everything.

```
FOR each channel c, each week w in the 7-year span:
    qty_ordered      ← Σ po_lines.qty_ordered      WHERE created_ts   in week w
    qty_received     ← Σ grn_lines.qty_received    WHERE event_ts     in week w
    qty_accepted     ← Σ quality_inspections.qty_accepted
    fill_rate        ← qty_accepted / qty_ordered
    lead_time_actual ← mean(grn.event_ts − po.created_ts)
    lead_time_ratio  ← lead_time_actual / contracted_lead_time_days
    otd_rate_last13  ← rolling mean of 1[receipt ≤ promise] over last 13 RECEIPTS
    ack_gap_ratio    ← Σ ack_qty / Σ qty_ordered
    load_ratio       ← qty_ordered / max(qty_received over trailing 52 CALENDAR w)
    is_active_week   ← qty_ordered > 0
    active_weeks_in_52       ← Σ is_active_week over trailing 52 calendar weeks
    reporting_lag_days       ← mean(recorded_ts − event_ts) over THIS week's events
                               NULL if the week posted nothing
    weeks_since_last_activity ← w − last week with qty_ordered > 0
    weeks_since_last_receipt  ← w − last week with a GRN;  NULL = never received
    fill_rate_last{4,13,52}   ← mean fill over the last N ACTIVE weeks
    ALL rolling windows: recorded_ts ≤ week_end  ← the as-of rule
    ALL instantaneous columns: NULL, not 0, when the week has no activity
```

`staleness_days` used to sit in this list as
`w_end − max(recorded_ts of contributing rows)`. That single expression is why it had to be
removed: on an active week it returns the posting lag, on an empty week it returns how long
the channel has been idle, and the two differ by three orders of magnitude. Compute
`reporting_lag_days` and `weeks_since_last_activity` separately — never one column.

### Handling staleness

Rane's ERP is manually updated, so at any $t_0$ some records are fresh and some are weeks
old. Give each time-varying feature a trust weight that decays with its age, falling back on
the channel's temporal estimate as the observation gets stale:

$$
\tilde{x}_i = g_i \cdot x_i^{\text{obs}} + (1 - g_i) \cdot x_i^{\text{prior}}, \qquad
g_i = \exp\big(-\max(0,\ w_i \cdot \Delta t_i + b_i)\big)
$$

Three rules that make this work rather than break it:

1. **Per feature, not per node.** `country` never changes and must not be penalised for
   being old. Apply decay only to time-varying fields.
2. **Continuous, not a threshold.** A hard "stale after 30 days" cutoff makes a 29-day and a
   31-day record behave completely differently, which shows up as instability.
3. **Pass $\Delta t$ as a feature too.** Staleness has two jobs — it decides *trust* via the
   gate, and it may itself be *predictive*. A record untouched for three weeks might mean a
   PO stuck in customs.

---

## Leakage checklist — run before every training job

| Check | Rule |
|---|---|
| As-of filter | Every feature query filters `recorded_ts ≤ t₀`, never `event_ts ≤ t₀` |
| BOM validity | Explosion uses `effective_from/to` containing the target period, not today's BOM |
| Promise date | Reconstructed from `po_line_revisions` as of $t_0$, not read from the current field |
| Label separation | `qty_received`, `qty_accepted`, `delivered_at`, `shortage_events` never appear as features |
| Denylist | Any pre-computed risk/reliability score is rejected at load time |
| Inventory | Only `inventory_snapshots` / `inventory_transactions` with `recorded_ts ≤ t₀` |
| Calibration split | Fit on earlier time folds, evaluate on later ones — never a random split |
| Rolling windows | Every window terminates at or before $t_0$, including the derived stores |
| **Outputs never in features** | No column of any Group I table is a model prediction. `arrival_probability`, `capacity_strain`, `network_risk_score`, `inventory_risk` all live in `model_outputs.csv`. Assert this at load time |
| **Label window is strictly future** | Every `training_labels` row satisfies `label_window_start > snapshot_date` |
| Coverage masking | Features from tables whose `dataset_coverage` starts after the row's date are masked, never zero-filled |
| Snapshot cutoff | Every feature query is additionally bounded by `snapshots.data_cutoff_ts` |

Port `verify_no_hidden_state()` from HADES v3 and extend its denylist to the Rane columns
listed in §14 of `dataset_structure.md`. It is the most valuable single piece of code being
carried over.
