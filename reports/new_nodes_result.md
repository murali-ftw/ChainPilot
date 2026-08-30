# Schema enrichment — does the missing graph explain `delay` and `shortage`?

**The question.** Two structural explanations for `delay` and `shortage` failing Phase 0's bar
are already closed. `reports/new_task_depth.md` ruled out readout **depth** (h¹/h²/h³/h⁴ tested
on the old graph; none cleared its floor) and readout **weighting** (Arm B, a per-task probe
refit on the same frozen representation; also failed). What remained was that the emitted graph
never carried the signal. This run adds the missing node and edge types, retrains SHARE +
Markov from scratch on the enriched graph, and measures all three tasks against the frozen
old-graph numbers.

**Bar, per the brief.** For `delay`/`shortage`: gain must clear its own FLOOR *and* be
sign-consistent across the 5 dataset-seed worlds. For `impact`: it already cleared its floor, so
the question is whether it *regressed* — reported with a direction, not just pass/fail.

Code: [db/enrich_schema.py](db/enrich_schema.py) · [db/verify_enriched.py](db/verify_enriched.py) ·
[ml/layer3_enriched.py](ml/layer3_enriched.py) ·
results: [out/layer3_v3/phase0_enriched.json](out/layer3_v3/phase0_enriched.json)

---

## 1. The controlling design decision: the world must not change

If enrichment perturbed the simulation, "new schema" and "different world" would be confounded
and nothing below would be interpretable. Two rules enforce it, both asserted rather than
claimed:

1. **No draw is taken from the simulation's RNG stream.** `db/enrich_schema.py` runs after the
   simulation completes and draws from its own `random.Random(seed ^ 0x5EED)`. The single touch
   inside the generator's weekly walk (`REPLENISH_LINKS.append`, [db/generate_dataset.py:1258](db/generate_dataset.py#L1258))
   consumes no draw and changes no branch.
2. **Every pre-existing table stays byte-identical.** `--enrich` is purely additive: 12 new
   files, zero edits to the 20 old ones.

`db/verify_enriched.py` regenerates each world *without* `--enrich` and gzip-diffs all 20 tables.
**Result: 20/20 byte-identical, on all 5 seeds.** Labels, shipments, delays and shortages are
literally the same rows Phase 0 trained on.

The same follows at the feature level: no task entity's feature vector was touched — every new
column lives on `Carrier`/`Port`/`Route`/`Warehouse`. So `P(Y|X)` must be unchanged, and the
driver recomputes it from the enriched bundles and diffs it against the frozen Phase 0 grid
rather than assuming it. **Result: max per-cell |diff| = 0.00e+00 over all 25 cells × 3 tasks.**

---

## 2. What was added

### Node types (3 new, 8 → 11)

| node type | n | features | source |
|---|---|---|---|
| **Carrier** | 5 | 13 | mode, lane count, inbound share + as-of OTR 30/90/180d, trend, transit variance, days-since-late, active shipments |
| **Port** | 19 | 11 | type/region one-hots, sea-gateway flag + as-of congestion index, mean dwell, 30d arrivals, late rate |
| **Route** (lane) | 112 | 12 | mode, typical transit days, great-circle distance, cross-region flag + as-of OTR 30/90d, transit mean/variance, volume, condition index |
| *Warehouse* | 8 | 7 → 12 | *existing, plus per-role traffic: role, inbound share, inbound/outbound 30d, open inbound/outbound* |

### Relations (9 new forward relations; 20 → 37 including reverses)

| relation | mean edges / snapshot | notes |
|---|---|---|
| `Shipment —HANDLED_BY→ Carrier` | 24,069 | |
| `Shipment —MOVES_ON→ Route` | 24,069 | |
| `Route —PASSES_THROUGH→ Port` | 239 | role (origin/transit/destination) + sequence on the edge |
| `Route —DEPENDS_ON→ Route` | 798 | the converted hidden factors — see §3 |
| `Shipment —ORIGINATES_FROM→ Supplier` | 10,474 | instance-level: this shipment's own lead-time position at t0 |
| `Shipment —REPLENISHES→ Warehouse` | 10,474 | inbound half of the warehouse role split |
| `Shipment —DELIVERS_TO→ Warehouse` | 13,595 | outbound half |
| `Product —REPLENISHED_BY→ Shipment` | **551** | the shortage fix — live inbound resupply as of t0 |
| `Warehouse —FULFILLS→ Customer` | 8,000 | |

Parameter count 752,211 → 956,157.

### Three places the implementation departs from the brief, and why

**`Plant —PRODUCES→ Product` was not added — it already exists.** The loader applies
`ToUndirected()`, so the existing `Product —MANUFACTURED_AT→ Factory` already yields
`Factory —rev_MANUFACTURED_AT→ Product`, which is that edge. Emitting a second relation with
identical structure would only cost the RGCN another coefficient row for no new information —
the exact cost `ml/data/loader.py`'s own docstring warns about ("an empty relation still costs
the RGCN family a coefficient row"). Same reasoning for `DELIVERS_TO→Warehouse` as a *plain*
duplicate of the existing `SHIPS_TO`: it is added only in its role-split form, carrying
information `SHIPS_TO` does not.

**`Warehouse` was not split into two node types.** The brief asks for staging/replenishment vs.
finished-goods/fulfillment. This world does not support that split: every one of the 8
warehouses receives inbound resupply and ships outbound fulfilment at a near-identical mix.
Measured inbound share across the 5 worlds is **0.443–0.495, spread ≤ 0.052**. A hard node split
would invent a partition the data does not have. The role distinction is instead carried where
an RGCN can actually use it — as two distinct **relations** (`REPLENISHES` vs `DELIVERS_TO`)
plus per-role traffic features on the Warehouse vector. The measured spread is asserted in
`db/verify_enriched.py` so the decision is checkable, not editorial.

**`DEPENDS_ON` is built from observable infrastructure, never from latent membership.** See §3.

---

## 3. The one place this could have silently cheated

The brief asks the scalar hidden factors `H_PORT` / `H_TRUCK` / `H_CUSTOMS` to become
`DEPENDS_ON` edges. Those are **latent supplier groups the generator deliberately never emits** —
"Hidden factors … are NEVER emitted as rows — only their correlated effects are observable"
(`db/generate_dataset.py` docstring), and `ml/data/loader.py::verify_no_hidden_state` exists
specifically to catch one reaching a CSV. Emitting the true membership as edges would hand the
model the answer and inflate every number in this report for free.

So the coupling is built from the **observable** infrastructure two lanes demonstrably share —
origin port, corridor transit hub, NAFTA trucking corridor, German customs jurisdiction — all
derivable from `shipments.origin_location` and the carrier's mode, columns that were already
emitted. The structure the brief wants (shipments coupled through shared infrastructure) is
reproduced; the latent grouping is not handed over.

How good is the proxy? `db/verify_enriched.py` measures it against the generator's true sets
(pulled from the generator's own namespace, never from a file):

| channel | true members | reached by the observable coupling | non-members also reached |
|---|---|---|---|
| port | 192 | 69.8 – 78.1 % | 37.6 – 41.6 % |
| trucking | 128 | 68.8 – 79.7 % | 12.8 – 17.3 % |
| customs | 80 | 70.0 – 82.5 % | 3.2 – 5.6 % |

(range over the 5 worlds). So it recovers roughly 70–80 % of each latent group, at the cost of a
substantial false-positive rate on the port channel — sea lanes share origin ports broadly. It
is a genuine proxy, not the answer key. The loader's hidden-state scan is **clean on all 5
worlds**.

---

## 4. Two temporal-integrity bugs found and fixed before the grid ran

Both were caught by `db/verify_enriched.py`'s requirement that an emitted as-of row be
*exactly* recomputable from its own stamp, and both were fixed before any reported model trained.

1. **Wrong reported-delivery clock.** The enrichment initially computed its windows against the
   simulation's in-memory `rec`. Variant 0 has Mechanism G off, so `report_delay()` returns 0
   and `rec == delivered_at` — while the *emitted* `shipment_status_history.recorded_at` is
   `max(rec, delivered_at + 5–45 min write jitter)`. The enrichment was therefore reading a
   clock up to 45 minutes earlier than any consumer of the CSVs can observe, on **all 37,737**
   delivered transitions. Fixed by materialising the written clock
   ([db/generate_dataset.py:1497](db/generate_dataset.py#L1497)) and computing every window
   against it. Byte-identity re-verified after the change.
2. **8-hour as-of coarsening.** As-of rows were stamped `wk.date().isoformat()`, but the
   generator's `ts()` defaults to `h=8`, so every week instant is 08:00 UTC. Each row was stamped
   eight hours earlier than the instant its statistics were computed at. Fixed by emitting the
   full timestamp.

After both fixes, all 5 worlds pass every check, including exact recomputation of sampled
`carrier_temporal_features` rows from reported-only deliveries (40 sampled, 0 mismatches).

---

## 5. Results — the headline table

25 cells per task (5 dataset seeds × 5 model-init seeds), variant 0, `db/csv_v1scale` world
regenerated with `--enrich`. Depths fixed at their retained values, **not** re-tuned.
FLOOR = max(init-seed floor, dataset-seed floor).

| task | depth | P(Y\|X) | P(Y\|X,H) | graph gain | init floor | dset floor | **FLOOR** | gain > FLOOR | worlds + |
|---|---|---|---|---|---|---|---|---|---|
| **delay** | h¹ | 0.7416 | 0.7407 | **−0.0010** | 0.2244 | 0.1039 | **0.2244** | **no** | 3/5 |
| **shortage** | h³ | 0.8727 | 0.9059 | **+0.0332** | 0.0424 | 0.0344 | **0.0424** | **no** | **5/5** |
| **impact** | h⁴ | 0.8202 | 0.9247 | **+0.1045** | 0.1148 | 0.0474 | **0.1148** | **no** | **5/5** |

### Row-for-row against the old graph

| task | depth | old P(Y\|X,H) | new P(Y\|X,H) | Δ | old gain | new gain | old FLOOR | new FLOOR | old | new |
|---|---|---|---|---|---|---|---|---|---|---|
| delay | h¹ | 0.7774 | 0.7407 | **−0.0367** | +0.0357 | −0.0010 | 0.0874 | 0.2244 | no | no |
| shortage | h³ | 0.7871 | **0.9059** | **+0.1188** | −0.0856 | **+0.0332** | 0.0231 | 0.0424 | no | no |
| impact | h⁴ | 0.9339 | 0.9247 | −0.0092 | +0.1137 | +0.1045 | 0.0284 | 0.1148 | **YES** | **no** |

### Per-world gain (5 worlds, each averaged over its 5 init seeds)

| task | d42 | d43 | d44 | d45 | d46 | worlds + |
|---|---|---|---|---|---|---|
| delay | +0.0343 | +0.0089 | +0.0114 | −0.0324 | −0.0271 | **3/5** |
| shortage | +0.0288 | +0.0384 | +0.0404 | +0.0230 | +0.0356 | **5/5** |
| impact | +0.0995 | +0.1175 | +0.0978 | +0.1072 | +0.1007 | **5/5** |

---

## 6. What actually happened: the floors moved more than the means

The single most important number in this run is not any AUC — it is that **the init-seed floor
rose on all three tasks at once**, by 4× to 16×:

| task | init-seed floor OLD | init-seed floor NEW | ×  |
|---|---|---|---|
| delay | 0.0140 | 0.2244 | **16.0×** |
| shortage | 0.0101 | 0.0424 | 4.2× |
| impact | 0.0261 | 0.1148 | 4.4× |

On the old graph the model-init seed barely mattered. On the enriched graph it dominates.
Mean test AUC by init seed makes it unmistakable:

| graph | task | m0 | m1 | m2 | m3 | m4 | range |
|---|---|---|---|---|---|---|---|
| OLD | delay | 0.7789 | 0.7762 | 0.7754 | 0.7786 | 0.7777 | 0.0035 |
| OLD | impact | 0.9352 | 0.9311 | 0.9338 | 0.9360 | 0.9333 | 0.0049 |
| **NEW** | delay | **0.6892** | 0.7617 | **0.7199** | 0.7758 | 0.7568 | **0.0866** |
| **NEW** | shortage | 0.9031 | 0.9089 | **0.8921** | 0.9129 | 0.9124 | 0.0208 |
| **NEW** | impact | 0.9264 | 0.9352 | **0.8936** | 0.9350 | 0.9334 | 0.0416 |

Seeds **m0 and m2 are bad on all three tasks simultaneously**. That is a shared-trunk
signature: the three heads read the same encoder, so a seed that lands the encoder in a poor
basin drags every task with it. Two cells collapse outright — `d45 m0` (delay 0.5745) and
`d46 m0` (delay 0.5661), both near chance.

It is *not* underfitting. Validation AUC actually **improved** on the enriched graph — mean best
validation 0.8350 → **0.8642** — while its spread doubled (sd 0.0136 → 0.0284, min 0.8187 →
0.7873). The enriched graph fits better on average and much less reliably. `d46 m2` peaked at
epoch 10 and never recovered, the classic early-collapse signature.

### Attributing it (brief item 6)

The census makes the mechanism concrete. **Carrier has 5 nodes carrying the entire shipment
population**, and the graph grows over the timeline:

| node type | n | incoming edges (t0 → t14) | mean in-degree (t0 → t14) |
|---|---|---|---|
| **Carrier** | **5** | 9,726 → 35,681 | **1,945 → 7,136** |
| Route | 112 | 9,726 → 35,681 | 87 → 319 |
| Port | 19 | 239 → 239 | 13 → 13 |
| *Supplier (reference)* | *800* | *4,740 → 15,565* | *6 → 19* |

Each of the five Carrier nodes aggregates **thousands** of shipment messages under a single
per-destination softmax — SHARE's attention is scoped per destination node across all incoming
relations, so one Carrier's representation is a near-global average over a quarter of every
shipment in the world. Two consequences follow directly, and they are the two things observed:

1. It creates a **2-hop shortcut between essentially any two shipments** (`Shipment → Carrier →
   Shipment`), which is new information for a task like `shortage` that previously had no path
   to fleet-wide conditions at all.
2. It is exactly the thin-relation regime SHARE's shared attention scorer and basis
   decomposition were designed to *bound* rather than eliminate. The design keeps the parameter
   cost of a thin relation low (10 basis coefficients), but it does not stop a 5-node type from
   acting as a dense global hub, and that hub is a plausible cause of the optimization
   instability. Port (19 nodes, in-degree 13) shows no such concentration and is not implicated.

This is stated as the leading explanation consistent with the evidence, not as a proven cause —
isolating it would need an ablation that drops `HANDLED_BY` and retrains, which is a separate run.

### Robustness diagnostic (not the headline)

Because two cells collapse, the median cell is worth quoting alongside the mean — as a
diagnostic only, since Phase 0's bar is defined on means and this report is scored on that bar:

| task | median P(Y\|X,H) OLD → NEW | median gain OLD → NEW |
|---|---|---|
| delay | 0.7979 → 0.7412 | +0.0494 → **−0.0073** |
| shortage | 0.7859 → **0.9083** | −0.0861 → **+0.0363** |
| impact | 0.9330 → 0.9358 | +0.1122 → **+0.1150** |

`delay` is worse and `shortage` better under both the mean and the median, so neither of those
conclusions rests on the outliers. `impact` is the one row the two views disagree about, and §7
says so explicitly.

---

## 7. Verdicts

### `delay` — the schema gap was **not** its bottleneck, and enrichment made it worse

P(Y|X,H) fell 0.7774 → 0.7407, the gain fell from +0.0357 to **−0.0010** (i.e. the graph is now
worth nothing over the shipment's own features), sign consistency broke from 5/5 to **3/5**, and
its FLOOR rose to 0.2244. It fails both halves of the bar, and fails them by more than before.
The added entities were the ones most obviously "near a shipment" — Carrier, Port, Route — and
they did not help; the same additions destabilised the shared trunk enough to produce two
near-chance cells. Combined with the two prior eliminations, **delay's failure is now unexplained
by depth, by readout weighting, and by the missing-schema hypothesis.** The `Shipment` feature
vector is 7 columns and 96.5 % of its predictions were already saturated at p ≥ 0.999
(`reports/decision_support_build.md` §3.2); that, not the graph, is where the next look belongs.

### `shortage` — the schema gap was a **real and large** bottleneck, but the result still does not pass

This is the biggest single movement on `shortage` in the project's record: P(Y|X,H) **0.7871 →
0.9059 (+0.1188)**, and the gain flips from **−0.0856 to +0.0332** — from "the graph is actively
worse than features alone" to "the graph adds something", positive in **5/5 worlds** where it was
negative in 5/5. Both the mean and the median agree. The `REPLENISHED_BY` edge — a stock position
linked to its own live inbound resupply — is the obvious candidate, and it was the edge added
specifically for this task.

**But it does not clear its floor**: +0.0332 against a FLOOR of 0.0424. The bar is unchanged and
the answer is no. Note *why* it misses: shortage's own floor rose 0.0231 → 0.0424 in the same
change that produced the gain, so the enrichment moved the target as it moved the mean. Had the
floor stayed at its old value, +0.0332 would have cleared comfortably. That makes the
instability in §6 the thing standing between this result and a pass — a concrete, testable
follow-up, not a reason to score it as a pass now.

### `impact` — predictive quality intact, **reproducibility regressed**

The brief asks for a direction, not just pass/fail, so both parts are reported:

- **Central performance did not meaningfully change.** Mean P(Y|X,H) 0.9339 → 0.9247 (−0.0092),
  gain +0.1137 → +0.1045 (−0.0092), still positive in **5/5 worlds**. On the median cell it is
  flat-to-slightly-better (gain +0.1122 → +0.1150). A −0.0092 shift against an old floor of
  0.0284 is within noise.
- **It nevertheless lost its PASS.** Old: gain 0.1137 > FLOOR 0.0284 → YES. New: gain 0.1045 >
  FLOOR 0.1148 → **no**. The status change is driven almost entirely by the **floor quadrupling**
  (0.0284 → 0.1148), not by the gain falling.

So the honest statement is: the enriched graph did not hurt what `impact` predicts, it hurt how
reliably it can be reproduced — and under this project's own rule (a delta smaller than the
reproduction floor is not evidence), that is enough to strip its pass. Reported as a **regression
in reproducibility**, since "did not regress" is not defensible when the one task that worked
now fails its gate.

---

## 8. Bottom line

**The schema gap was a genuine bottleneck for `shortage` and only for `shortage`.** It moved that
task further than depth or readout weighting ever did — an 0.119 AUC jump and a sign flip across
all five worlds — and it did nothing for `delay` while making it worse. No task clears its floor
on the enriched graph, including the one that used to.

The reason no task passes is the same for all three: **the enrichment bought accuracy and paid
for it in reproducibility.** Validation fit improved (0.8350 → 0.8642) while init-seed variance
roughly doubled to quadrupled, and the floors moved further than the means did. The five-node
`Carrier` type, aggregating up to 7,136 incoming edges per node, is the leading suspect.

### Limitations, stated plainly

- **Grid is full size** — 5 dataset seeds × 5 init seeds, 25 cells, no shrinkage.
- **`REPLENISHED_BY` reaches only 36.5 % of `Product` nodes at a given t0** (min 22.4 %, max
  57.0 %, over 75 snapshots). A pair only has a live inbound resupply once its stock has fallen
  below the reorder trigger, so the edge added for `shortage` structurally cannot touch about two
  thirds of the labelled entities at any one snapshot. The +0.119 was obtained *despite* that.
- **`DEPENDS_ON` is an observable proxy** recovering ~70–80 % of each latent group with a 3–42 %
  false-positive rate (§3), not the latent grouping itself.
- **Warehouse was not split into two node types** — the world does not support the split (§2).
- **Depths were not re-tuned** (h¹/h³/h⁴ throughout), by design. Whether the enriched graph moves
  the best depth — particularly for `shortage`, now that it has a genuinely informative
  neighbourhood — is the obvious next experiment and is deliberately not conflated with this one.
- **The Carrier-hub attribution is a hypothesis**, consistent with the census and the seed
  pattern but not isolated by ablation.

### The two experiments this points to

1. **Drop `HANDLED_BY` (or bucket Carrier by lane) and retrain.** If the init-seed floor falls
   back toward its old value while `shortage`'s gain holds near +0.033, shortage passes and the
   §6 attribution is confirmed in one run.
2. **Re-tune `shortage`'s depth on the enriched graph.** Its neighbourhood is materially
   different now; h³ was chosen against a graph that lacked `REPLENISHED_BY` entirely.

---

## 9. Reproducing

```
python3 db/generate_dataset.py --variant 0 --seed 42 --config v1 --enrich \
    --out-dir csv_v3enriched/v0enr_seed42          # x5 seeds; ~18s each
python3 db/verify_enriched.py --enriched db/csv_v3enriched/v0enr_seed42 --seed 42
python3 ml/layer3_enriched.py --dseeds 42,43,44,45,46 --mseeds 0,1,2,3,4
```

Training the 25 enriched backbones is ~450 s each solo (~5 h sequential; ~1 h at 5-way
parallelism). `--enrich` is off by default, so every pre-existing invocation of the generator is
unaffected.
