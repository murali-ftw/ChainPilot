# Carrier fan-in fix — re-granularizing Carrier, and whether it resolves the instability

**The problem.** [reports/new_nodes_result.md](reports/new_nodes_result.md) found the enriched
graph bought `shortage` a large gain (−0.0856 → +0.0332, sign-flipped in 5/5 worlds) and paid
for it in reproducibility: init-seed floors rose 4×–16× on all three tasks at once, with seeds
m0 and m2 bad on `delay`, `shortage` and `impact` *simultaneously* — a shared-trunk signature.
The leading suspect was `Carrier`: 5 nodes absorbing the entire shipment population, mean
in-degree growing to ~7,100 under a single per-destination attention softmax.

**What this run does.** Two arms, plus the two already-frozen ones for comparison:

| arm | what it is | role |
|---|---|---|
| **old** | V2 graph, no enrichment | frozen reference |
| **enriched** | V3 with `Shipment—HANDLED_BY→Carrier`, 5 carrier nodes | frozen reference (the problem) |
| **no_carrier** | `HANDLED_BY` **and** the `Carrier` node type dropped | **control only** — tests the hypothesis, not a candidate |
| **carrier_lane** | one node per observed `(carrier, lane)` pair | **the fix** |

Code: [ml/carrier_fan_in_census.py](ml/carrier_fan_in_census.py) ·
[ml/layer3_carrier_fix.py](ml/layer3_carrier_fix.py) · results:
[out/layer3_v3/carrier_fan_in.json](out/layer3_v3/carrier_fan_in.json) ·
[out/layer3_v3/carrier_fix.json](out/layer3_v3/carrier_fix.json)

---

## 1. Design: all four arms read byte-identical CSVs

A `(carrier, lane)` node is a **regrouping of two columns the generator already emits** —
`shipment_routes.carrier_id` × `shipment_routes.route_id`. It needs no new simulation data, so
nothing was regenerated: the three V3 arms are three graph *constructions* over the same emitted
files, selected by `v3_schema` in [ml/data/loader.py](ml/data/loader.py). That is a stronger
design than regenerating per arm — the underlying world, the labels and the emitted tables are
not merely equivalent across arms, they are the same bytes, so the comparison isolates graph
**shape** and nothing else.

Two decisions inside the fix are worth stating because they are what make it a controlled test:

**Features are the parent carrier's, copied down unchanged.** Recomputing on-time rates per
`(carrier, lane)` would make each bucket strictly *more* informative than the carrier node it
replaces, and any improvement could then be a better feature rather than a smaller fan-in.
Copying keeps feature content identical to the `enriched` arm so that fan-in is the only
variable.

**The ablation drops the node type, not just the relation.** `HANDLED_BY` is the only relation
touching `Carrier`, so removing it alone would leave an orphaned node type — which still costs
the encoder a per-type input projection. "No Carrier at all" is the honest control.

**No information is removed by the fix — asserted, not claimed.** Every shipment's true carrier
must still be exactly recoverable from its bucket node.
`VariantDataset.assert_carrier_recoverable()` checks this on **all 39,184 shipments** (not a
sample): 312 bucket nodes, 5 parent carriers, zero mismatches.

Depths are **not** re-tuned (`delay→h¹`, `shortage→h³`, `impact→h⁴`), so any movement is
attributable to the Carrier schema alone.

---

## 2. Fan-in census — run before any training (brief item 3)

Forward relations only (`rev_*` mirrors excluded), matching the convention
`reports/new_nodes_result.md` §6 used, so these numbers are directly comparable to the ones it
published.

### The quantity the fix targets

| arm | node type | n | mean in-degree t0 → t14 | max in-degree t0 → t14 |
|---|---|---|---|---|
| enriched | `Carrier` | **5** | 1,945 → **7,136** | 2,906 → **10,670** |
| **fixed** | `CarrierLane` | **312** | 31 → **114** | 60 → **221** |

**62.4× reduction in mean in-degree; 48× in max.** The fix does what it was supposed to do.

### Did it create a new hub anywhere? No.

Full census at t14 (mean in-degree), fixed graph:

| node type | n | mean in-deg | max in-deg |
|---|---|---|---|
| Warehouse | 8 | 9,320 | 10,173 |
| Factory | 5 | 4,410 | 4,767 |
| Route | 112 | 326 | 639 |
| **CarrierLane** | **312** | **114** | **221** |
| Product | 1,280 | 55 | 76 |
| Supplier | 800 | 39 | 3,508 |
| Customer | 1,600 | 22 | 38 |
| Port | 19 | 13 | 29 |

`CarrierLane` now sits *below* `Route`, well inside the ordinary range. No new concentration
was created.

### The census also complicates the original hypothesis

`Warehouse` (8 nodes, mean in-degree **9,320**) and `Factory` (5 nodes, **4,410**) are larger
hubs than `Carrier` ever was — and they are present in **every** arm, including the ablation.
More importantly, they were present in the **old, stable** graph too:

| node type | old (V2) graph t14 | enriched t14 | change |
|---|---|---|---|
| Warehouse | 4,859 | 9,320 | **1.9×** |
| Factory | 4,410 | 4,410 | unchanged |
| Carrier | — | 7,136 | **new** |

The old graph carried a ~4,900-in-degree hub and was perfectly stable (init floors 0.014–0.026).
So **high fan-in alone is not sufficient** to cause the instability. What the enrichment did was
(a) add an entirely new 7,136 hub and (b) nearly double `Warehouse`'s. The ablation arm
discriminates between those: it removes (a) and leaves (b) intact. That makes the control
genuinely decisive rather than a formality — and it means a null result there would point at
`Warehouse`, not vindicate the enrichment.

---

## 3. Four-way comparison (brief item 5)

25 cells per arm (5 dataset seeds × 5 model-init seeds), depths fixed at h¹/h³/h⁴, protocol
imported from `ml/layer3_baseline.py`. **PASS** = gain clears its own FLOOR *and* is positive in
all 5 worlds. `P(Y|X)` re-verified unchanged on both new arms (max per-cell |diff| = 0.00e+00).

| task | arm | P(Y\|X,H) | gain | init floor | dset floor | **FLOOR** | gain>FLOOR | worlds + | **PASS** |
|---|---|---|---|---|---|---|---|---|---|
| **delay** | old graph (V2) | 0.7774 | +0.0357 | 0.0140 | 0.0874 | 0.0874 | no | 5/5 | – |
| | enriched (Carrier hub) | 0.7407 | −0.0010 | 0.2244 | 0.1039 | 0.2244 | no | 3/5 | – |
| | *ablation (no Carrier)* | *0.7711* | *+0.0295* | *0.0248* | *0.0804* | *0.0804* | *no* | *5/5* | *–* |
| | **fixed (bucketed Carrier)** | 0.7438 | +0.0022 | 0.2951 | 0.0728 | 0.2951 | no | 4/5 | **–** |
| **shortage** | old graph (V2) | 0.7871 | −0.0856 | 0.0101 | 0.0231 | 0.0231 | no | 0/5 | – |
| | enriched (Carrier hub) | 0.9059 | +0.0332 | 0.0424 | 0.0344 | 0.0424 | no | 5/5 | – |
| | *ablation (no Carrier)* | *0.9121* | *+0.0395* | *0.0121* | *0.0226* | *0.0226* | *YES* | *5/5* | ***PASS*** |
| | **fixed (bucketed Carrier)** | **0.9069** | **+0.0343** | 0.0335 | 0.0293 | **0.0335** | **YES** | **5/5** | **PASS** |
| **impact** | old graph (V2) | 0.9339 | +0.1137 | 0.0261 | 0.0284 | 0.0284 | YES | 5/5 | PASS |
| | enriched (Carrier hub) | 0.9247 | +0.1045 | 0.1148 | 0.0474 | 0.1148 | no | 5/5 | – |
| | *ablation (no Carrier)* | *0.9312* | *+0.1110* | *0.0135* | *0.0257* | *0.0257* | *YES* | *5/5* | ***PASS*** |
| | **fixed (bucketed Carrier)** | **0.9302** | **+0.1100** | 0.0252 | 0.0219 | **0.0252** | **YES** | **5/5** | **PASS** |

*Ablation rows are italicised throughout: it is a control that discards carrier information, not
a production candidate.*

---

## 4. Mean test AUC by model-init seed (brief item 6)

Did the m0/m2 cross-task correlation break?

| arm | task | m0 | m1 | m2 | m3 | m4 | range |
|---|---|---|---|---|---|---|---|
| old (V2) | delay | 0.7789 | 0.7762 | 0.7754 | 0.7786 | 0.7777 | 0.0035 |
| | shortage | 0.7867 | 0.7841 | 0.7906 | 0.7869 | 0.7870 | 0.0065 |
| | impact | 0.9352 | 0.9311 | 0.9338 | 0.9360 | 0.9333 | 0.0050 |
| enriched | delay | **0.6892** | 0.7617 | **0.7199** | 0.7758 | 0.7568 | **0.0866** |
| | shortage | 0.9031 | 0.9089 | **0.8921** | 0.9129 | 0.9124 | **0.0208** |
| | impact | 0.9264 | 0.9352 | **0.8936** | 0.9350 | 0.9334 | **0.0416** |
| *ablation* | *delay* | *0.7697* | *0.7728* | *0.7719* | *0.7667* | *0.7743* | ***0.0076*** |
| | *shortage* | *0.9095* | *0.9138* | *0.9125* | *0.9137* | *0.9110* | ***0.0043*** |
| | *impact* | *0.9295* | *0.9335* | *0.9305* | *0.9291* | *0.9335* | ***0.0044*** |
| **fixed** | delay | 0.7349 | 0.7604 | 0.7600 | **0.6972** | 0.7666 | **0.0694** |
| | shortage | 0.9025 | 0.9064 | 0.9117 | 0.9034 | 0.9106 | **0.0092** |
| | impact | 0.9299 | 0.9338 | 0.9319 | 0.9266 | 0.9287 | **0.0072** |

**The m0/m2 correlation is gone in both new arms** — no seed is jointly bad across all three
tasks any more. But the two arms got there differently:

- **Ablation**: ranges collapse to 0.0043–0.0076, i.e. *old-graph levels* (0.0035–0.0065). The
  instability is completely removed.
- **Fixed**: `shortage` (0.0092) and `impact` (0.0072) are near old-graph levels too — for those
  two tasks the fix works. `delay` still spans 0.0694, driven by a single cell.

So the answer to item 6 is split by task: the shared-trunk correlation broke, but `delay` retains
an isolated failure mode the other two do not.

### That failure is one dead head, not a dead trunk

`d45 m3` on the fixed graph returns `delay = 0.5000` exactly. Inspecting its predictions:

| task | test rows | unique predicted values | std |
|---|---|---|---|
| **delay** | 4,373 | **1** (all 0.493072) | 3.0e-08 |
| shortage | 19,098 | 7,671 | 1.3e-01 |
| impact | 4,800 | 3,409 | 2.6e-01 |

The delay head collapsed to a constant while the *same model's* shortage and impact heads stayed
healthy. That is qualitatively different from the enriched arm, where a bad seed dragged all
three tasks together. Excluding that one cell as a diagnostic (not as a reported result), the
fixed graph's delay init floor falls **0.2951 → 0.0924**, i.e. the entire delay floor on this arm
is that single cell.

### Convergence

| arm | best val AUC mean | sd | min |
|---|---|---|---|
| enriched | 0.8642 | 0.0284 | 0.7873 |
| *ablation* | *0.8782* | ***0.0109*** | *0.8609* |
| fixed | 0.8679 | 0.0217 | 0.7828 |

The ablation is both the best-fitting and by far the most stable. The fix sits between the two.

---

## 5. Verdicts (brief item 7)

### `shortage` — **PASSES.** First time in the project's record.

Fixed graph: gain **+0.0343 against a FLOOR of 0.0335**, positive in **5/5 worlds**. Both halves
of the bar are met, so shortage is fixed.

**The margin is 0.0008 and that has to be said plainly.** It clears, but by about 2 % of the
floor — one unlucky world would erase it. The control clears the same bar far more comfortably
(+0.0395 against 0.0226, margin 0.0169), because removing the relation cuts shortage's init floor
to 0.0121 rather than 0.0335. The pass is real under the stated bar; its robustness is thin.

The trajectory across arms is the substantive result: −0.0856 (old) → +0.0332 (enriched, failed
on floor) → **+0.0343 clearing 0.0335** (fixed). The *mean* barely moved between enriched and
fixed (+0.0011); what changed is the floor (0.0424 → 0.0335). Shortage passes because the fix
bought stability, not because it bought accuracy.

### `impact` — **recovers its lost pass.**

Fixed graph: +0.1100 against FLOOR 0.0252, 5/5 worlds. It is back to essentially its old-graph
standing (+0.1137 against 0.0284), and its floor is now *lower* than the old graph's. This
confirms `reports/new_nodes_result.md`'s reading that impact never lost predictive quality — it
lost reproducibility, and restoring reproducibility restored the pass with the gain unchanged
(0.1045 → 0.1100, both within noise of the old 0.1137).

### `delay` — **unaffected, as expected, and still open.**

This fix targets Carrier, not delay's own diagnosed problem, and delay was not expected to move.
It did not:

- On the **control**, delay returns to roughly its old-graph behaviour: +0.0295 vs +0.0357, floor
  0.0804 vs 0.0874, 5/5 worlds positive — still failing, and failing for the *pre-existing*
  reason, its **dataset-seed floor** (0.0804), which no Carrier change touches.
- On the **fixed** graph it is worse (+0.0022, floor 0.2951, 4/5) entirely because of the one
  dead head above.

**`delay` remains a separate, still-open issue.** Its diagnosed cause is unchanged from
`reports/new_nodes_result.md` §7: a 7-column `Shipment` feature vector and 96.5 % of predictions
saturated at p ≥ 0.999. Depth, readout weighting, and schema enrichment have now all been
eliminated for it. Nothing in this run was expected to help it and nothing did.

---

## 6. The honest complication: the control beats the fix

Item 1 asked whether the ablation confirms the hub hypothesis. **It does, emphatically** — every
init-seed floor returns to old-graph levels, every collapse disappears, and `shortage` keeps its
gain *without `HANDLED_BY` at all* (0.9121 vs the enriched 0.9059), confirming that
`REPLENISHED_BY`, not the carrier relation, is what drives shortage's improvement.

But the control also **outperforms the fix on every single metric**: better floors (0.0226 vs
0.0335 on shortage), better means, better validation stability, no dead heads. The fix works —
both target tasks pass — but it is still paying a stability tax the control does not.

### Why re-granularizing helped less than removing

The census showed the fix did exactly what it was designed to do — 62× less fan-in, no new hub.
So fan-in magnitude cannot be the whole mechanism. Two facts from this run argue it is not even
the main one:

1. **The old graph was stable with bigger hubs than Carrier.** `Warehouse` at in-degree 4,859 and
   `Factory` at 4,410, with init floors of 0.014–0.026. And the ablation still carries
   `Warehouse` at **9,320** — nearly double the old graph's — while being the most stable arm
   here. High fan-in is evidently tolerable.
2. **`CarrierLane` is a strict refinement of `Route`.** With mode determined by carrier, a
   `(carrier, lane)` node partitions a lane's traffic: 112 routes → 312 carrier-lanes, mean 2.79
   carriers per lane, 126 shipments per CarrierLane against 350 per Route. So every
   `Shipment→CarrierLane→Shipment` path is a **subset** of a `Shipment→Route→Shipment` path. The
   fix replaced one large hub with a relation whose neighbourhoods are *nested inside*
   `MOVES_ON`'s — two relations competing for weight in the same per-destination attention
   softmax over near-identical neighbourhoods.

That reframes the mechanism: what destabilised the encoder was less the *size* of Carrier's
fan-in than the addition of a relation that is largely **redundant** with one already present.
Bucketing made the redundancy worse, not better — the original Carrier at least aggregated
*across* lanes and so carried something `Route` did not. This is the leading explanation
consistent with all four arms; it is not isolated by ablation and is stated as such.

---

## 7. Bottom line

**The fix achieves its objective.** `shortage` passes for the first time (+0.0343 > 0.0335, 5/5
worlds) and `impact` recovers its lost pass (+0.1100 > 0.0252, 5/5), with all carrier information
retained and verified recoverable on all 39,184 shipments. `delay` is untouched, as intended, and
remains open for its own separate reason.

**But the control is the better graph on this evidence**, and the brief is right that it cannot
simply be adopted — it deletes information the fix preserves. The choice is a real trade:

| | fixed (bucketed Carrier) | ablation (no Carrier) |
|---|---|---|
| carrier information | **retained** (verified) | discarded |
| shortage | PASS, margin 0.0008 | PASS, margin 0.0169 |
| impact | PASS, margin 0.0848 | PASS, margin 0.0853 |
| delay init floor | 0.2951 (one dead head) | 0.0248 |
| worst-case val AUC | 0.7828 | 0.8609 |

### Limitations

- **`shortage`'s pass on the fixed graph is narrow** (margin 0.0008 on a 0.0335 floor). It meets
  the stated bar; it should not be described as comfortable.
- **One dead head remains** (`d45 m3` delay). It is isolated rather than a trunk failure, but it
  is a real reproducibility defect and it sets the whole delay floor on that arm.
- **The redundancy explanation is a hypothesis**, consistent with all four arms and with the
  nesting proof above, but not isolated by its own ablation.
- **Depths were not re-tuned** (h¹/h³/h⁴), by design.
- Everything else inherits `reports/new_nodes_result.md`'s limitations unchanged — notably that
  `REPLENISHED_BY` reaches only ~36.5 % of `Product` nodes at a given t0.

### What this points to next

1. **Drop `MOVES_ON→Route` and keep `HANDLED_BY→CarrierLane`.** If the redundancy explanation is
   right, removing the *coarser* of the two nested relations should recover the ablation's
   stability while keeping carrier information — the best of both arms, and a single 25-cell run.
2. **Re-tune `shortage`'s depth on a stable enriched graph.** h³ was chosen against a graph with
   no `REPLENISHED_BY` at all; with shortage now passing by 0.0008, a better depth is the
   cheapest available margin.
3. **`delay` needs a features-side investigation**, not a graph-side one.

---

## 8. Reproducing

```
python3 ml/carrier_fan_in_census.py --csv-dir db/csv_v3enriched/v0enr_seed42
python3 ml/layer3_carrier_fix.py --schemas no_carrier,carrier_lane
```

No regeneration is required — all three V3 arms are graph constructions over the same emitted
CSVs, selected by `v3_schema`. Training the 50 backbones is ~450 s each (~2.5 h at 5-way
parallelism); `v3_schema="full"` is the default everywhere, so every previously trained
checkpoint keeps its exact path and every earlier result stands.
