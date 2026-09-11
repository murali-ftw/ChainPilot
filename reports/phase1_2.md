# Phases 1 & 2 — guide and label fixes, then within-world scores

**Read this first.** Every score below is **within-world**: the same generator produced the
training and the test data, separated only by time. They are an **optimistic upper bound** on
what to expect from Rane's real extract, not a forecast of it. A synthetic world has no schema
drift, no ERP migration, no relabelled part numbers, no plant that changes its recording
convention halfway through, and no distribution shift beyond the one the generator was told to
produce. **Do not quote the C-index or PR-AUC here to Aptimeta as an expected production
number.** The only measurement that would support such a number is a rolling-origin backtest on
Rane's own extract, and this run is not that.

---

# PHASE 1

## 1. Guide changes — `docs/implementation_guide.md`

Five corrections, all measured on `db/gen_v6/seed_1001` and `db/gen_v7/seed_1001` across all
8,598,520 `channel_performance_weekly` rows per world.

| # | change | where |
|---|---|---|
| 1 | **Residual connections made a required element** of the TCN, with the failure evidence | step 3.1 |
| 2 | **`d = 38` replaced by a derivation**; the built panel is `d = 24` | step 2.1 |
| 3 | **Null-sparsity table corrected** and the forward-fill mechanism documented | step 2.2 |
| 4 | **The impossible verify step struck through** and replaced with four that hold | step 2.2 |
| 5 | **Four constant-zero columns recorded as known-unpopulated** | new section |
| — | **"Known deviations from spec" section added** and linked from the contents | before Troubleshooting |

### 1.1 Residual connections

The guide fixed dilation, depth and width and said nothing about skip connections. Built
literally — six `Conv1d`+ReLU layers, nothing else — the encoder scored **C-index 0.5000**,
exactly chance, while a ridge probe on the identical input tensors scored **0.6464** and the
LightGBM baseline scored 0.6510. The signal was fully present; a 6-deep ReLU stack with no path
around it was destroying it. Adding a 1×1 residual projection per layer, changing nothing else,
took the same model to **0.6524**.

Step 3.1 now carries the projection in the hyperparameter table as **required**, gives the
`z = ReLU(conv(pad(z))) + W₁ₓ₁ z` form with the residual added *after* the causal trim so it
cannot reintroduce a leak, and shows both loss curves so the failure is recognisable in one
epoch:

```
no residual:   1.0136  0.9995  0.9987  0.9978  0.9972  0.9978   <- flat at 1.0, R^2 ~ 0
with residual: 0.9614  0.9229  0.9128  0.9057  0.9012  0.8979
```

A second verify step was added beside the causality test: the TCN must clear a ridge probe on
the flattened window, or the stack is broken.

### 1.2 `d` is derived, not hardcoded

| term | spec §3.3 | measured | why |
|---|---|---|---|
| numeric value channels | 18 | **14** | four columns constant zero |
| missing-indicators | 13 | **10** | three dropped nullables take their indicators |
| calendar block | 7 | 7, **not folded in** by `ml/data/cache.py` | optional |
| **d** | **38** | **24** as built, 31 with calendar | |

### 1.3 Null-sparsity — the specification is wrong by an order of magnitude

| column | spec §3.2 | **v6** | **v7** |
|---|---|---|---|
| `fill_rate` | 81.4% | **5.52%** | **5.95%** |
| `ack_gap_ratio` | 81.4% | **5.52%** | **5.95%** |
| `load_ratio` | 81.8% | **0.00%** | **0.00%** |
| `lead_time_actual_days` / `_ratio` | 72.6% | **5.52%** | **5.95%** |

Both generators forward-fill the level and rolling columns across idle weeks, so the sparsity is
real but lives elsewhere: **`qty_ordered == 0` on 86.3% of channel-weeks in v6 and 88.6% in
v7**, with `is_active_week` true on only 13.7% and 11.4%. The indicator channels are
consequently near-vacuous — mean **0.9503** (v6) and **0.9465** (v7) over the ten surviving
nullables, against the ≈0.181 the specification implies.

### 1.4 The verify step that cannot pass

*"For a known idle week the `fill_rate` channel is 0.0"* fails by design: **175,481 of 212,992
idle positions carry a non-zero forward-filled value.** It is struck through in the guide with
the measurement attached and replaced with four checks that hold on both worlds — that
`is_active_week` is binary with mean in (0.10, 0.15), that an idle week has `active == 0` and
`pad == 1` simultaneously, that value and indicator are coherent wherever a null does occur, and
that the indicators are vacuous. That last is asserted deliberately, so nobody trusts them.
`is_active_week` is documented as **the** channel carrying the idle/active signal.

### 1.5 Four constant-zero columns

`revision_count`, `days_since_last_short`, `weeks_since_last_activity` and
`weeks_since_last_receipt` hold **one distinct value — zero — across all 8,598,520 rows of both
worlds**. Recorded as generator placeholders that were never written, with instructions to
exclude all four and the three nullables' indicators with them. One consequence is called out:
specification §3.1 presents `weeks_since_last_activity` and `reporting_lag_days` as jointly
replacing `staleness_days`, but only `reporting_lag_days` is populated, so the staleness gating
layer of step 5.0 has one input, not two.

---

## 2. Validator amendment 02 — constant-column check

| | |
|---|---|
| SHA before | `ed4742e0bef9d60af7c5b2a2550b590b27e6b217574df5b17877ba99be57ac56` |
| SHA after | `4dbb229836de2f3bcfd39ad4ff343b4c9b568f2d6c7e1b10591d167360dfb2ce` |
| Diff | `1 file changed, 104 insertions(+)` — **no deletions, no alterations** |
| Record | [`docs/validator_amendment_02.md`](../docs/validator_amendment_02.md) |

Every declared column present in a table accumulates a distinct-value set capped at two entries
during the existing Tier 1 row pass — O(1) per cell after a column sees its second value, which
for a real column is the second row. Tables with fewer than two rows are exempt. Columns with
exactly one distinct value are sorted into three buckets, and **only one is gated**:

| bucket | severity | reason |
|---|---|---|
| **all-zero numeric** | **FAIL** | no reading under which a `DECIMAL` that is 0.0 in eight million rows is correct |
| all-blank | warn | often the documented open-ended convention — empty `effective_to` means still effective |
| single non-zero literal | warn | often a genuinely uniform dimension — one country, one currency |

The narrowness is deliberate. A gate that fired on all three would be ignored within a week,
which is this project's "gate that cannot fail" failure mode run in reverse. Both non-gated
buckets are still reported in full, per table and in aggregate.

### What it catches

**29 all-zero numeric columns across 5 tables**, identical in both worlds, plus 21 all-blank and
85 single-literal as warnings:

| table | all-zero |
|---|---|
| `inventory_position_weekly` | **13 — every numeric column in the table** |
| `supplier_performance_weekly` | 8 |
| `channel_performance_weekly` | 4 (the ones Phase 2 found by hand) |
| `inventory_snapshots` | 3 |
| `quality_inspections` | 1 |

**`inventory_position_weekly` is the finding worth acting on.** It has 52,080 rows and no
content: `qty_on_hand`, `qty_available`, `safety_stock_qty`, `days_of_supply`, the consumption
and inter-plant columns, all zero. A Phase 9 stock roll-forward built against it would be
rolling zeros forward. Nothing in Tier 1 could see this before — a zero passes the type check,
passes NOT NULL, and is invisible to the PK and FK checks.

Both worlds now fail this gate. The check did not create the defect; prior validation reports
are not restated and none of their numbers moves.

---

## 3. v6 label surgery

### 3.1 Determinism — proved first

`db/gen_v6/generator_v6.py --seed 1001` re-run into a scratch directory and content-hashed
table by table against `db/gen_v6/seed_1001`:

| | |
|---|---|
| tables byte-identical | **48 of 49** |
| the exception | `dataset_coverage.csv` |
| its 9 substantive columns | **identical** (`f7dc1fc0…` both) |
| what differs | `assessed_ts`, a wall-clock stamp written at emit time |
| `training_labels.csv` | **byte-identical**, `607c26ad…` |

**v6 reproduces**, so the rebuild path was available and the fallback derivation from
`part_demand_weekly` was not needed.

### 3.2 The method — backport, then splice

The v7 generator's fix is one change: sample the part × plant **universe** (`PP_SAFETY > 0`)
rather than only the part-plants that went short. Applied to a scratch copy of
`generator_v6.py`, the complete diff excluding comments is ten lines:

```diff
-    ppv, cnt = np.unique(sch_[fs], return_counts=True)
-    if len(ppv):
-        sel = r.choice(len(ppv), min(1500, len(ppv)), replace=False)
-        qsum = np.bincount(sch_[fs], weights=sq_[fs], minlength=NPP)
-        for k in ppv[sel]:
+    qsum = np.bincount(sch_[fs], weights=sq_[fs], minlength=NPP)
+    active_pp = np.where(PP_SAFETY > 0)[0]
+    if len(active_pp):
+        sel_pp = r.choice(active_pp, min(1500, len(active_pp)), replace=False)
+        for k in sel_pp:
```

`st_`, `sch_`, `sq_` and `PP_SAFETY` are all v6's own quantities, fixed before the label stage.
**v6's demand process, escalation rule and event rates are untouched** — no v7 mechanism was
imported, so the two worlds remain different regimes.

**Why the rebuild could not simply be swapped in.** It emits more shortage rows, which advances
the `label_id` counter and consumes a different number of RNG draws. Both effects propagate: 19
tables emitted after `training_labels.csv` differ in the patched run, and so do the non-shortage
label rows from the first shortage block onward. The corrected shortage rows were therefore
**spliced** into the original file — non-shortage rows copied through as the **original raw
bytes**, new shortage rows inserted at their structural position (after `capacity_strain`,
before `demand_drift`, per snapshot) with `label_id`s numbered above every id in the original.

### 3.3 Proof that nothing else moved

| | |
|---|---|
| non-shortage rows, sha256 **before** | `1ab23a8759a2cf8793a511815efe774af8f9b89603da27db6156028557e26488` |
| non-shortage rows, sha256 **after** | `1ab23a8759a2cf8793a511815efe774af8f9b89603da27db6156028557e26488` |
| **identical** | **yes** — 946,200 rows |
| duplicate `label_id` in the output | 0 |
| all 47 other tables vs the reproduction run | **unchanged** |

Row counts by task, unchanged: `arrival_week` 332,000 · `fill_rate` 332,000 ·
`capacity_strain` 207,500 · `demand_drift` 74,700. `shortage_qty` 45,942 → **124,500**.

The original is preserved as **`db/gen_v6/seed_1001/training_labels.pre_fix`** — 157,163,529
bytes, sha256 `607c26ad…`, the exact bytes the Phase 2 and Phase 3 numbers were measured on.
Deliberately without a `.csv` extension so the validator's `discover()` does not pick it up as
an extra table. The new file is `dd8ba012…`.

### 3.4 Verification

| check | result |
|---|---|
| **both classes present** | **yes** — 16,795 positive / 107,705 negative |
| **positive rate** | **13.49%** (v7 reference 12.8%) |
| `entity_type` | `['part_plant']` — the only value |
| `entity_id` resolves | **4,212 of 4,212** distinct ids present in `part_plant.csv`, 0 unresolved |
| label window opens after the snapshot | **yes**, minimum gap 1 day; `window_end > window_start` on every row; horizon 90 |
| **KS vs Uniform(0,1)** | **D = 0.9286, critical(0.05) = 0.0105 — uniform rejected** (n = 16,795). The frozen validator's own KS check agrees: D = 0.9750, p = 0 |
| positive `label_value` | min 1, median 31, max 22,566 |

**Forward-computation.** `shortage_events.csv` holds the 1,911 *escalated* episodes; labels are
built from the larger *condition* set, of which escalated is a subset, so the test is
directional:

- of 1,293 rows with an escalated episode inside the forward window `(t₀, t₀+12]`, **not one has
  label 0**, and 1,257 have `label_value ≥` that escalated mass;
- the 36 that do not are explained — **32** have an episode running past the 12-week window, so
  the episode total exceeds the in-window mass by construction; the remaining **4** sit on a week
  boundary moved by the ±6-day recording jitter in `shortage_start_ts`;
- of 11,674 rows whose only escalated episodes are **at or before** t₀, **65.06% carry label
  exactly 0**. A label leaking backward could not produce that.

### 3.5 The frozen validator, before and after

Both runs use `validator.py` at amendment 02, so the new gate is present in both and the diff
isolates the label change.

| | pre-fix | post-fix |
|---|---|---|
| checks emitted | 234 | 234 |
| **checks whose status differs** | — | **0** |
| verdict | NOT USABLE | NOT USABLE (identical) |
| FAILED list | identical | identical |

Every line that moved is a shortage-label count or a row total derived from one:

```
< training_labels                                992,142 rows      > 1,070,700 rows
< shortage_qty  n=45,942  uncensored=44,573      > shortage_qty  n=124,500  uncensored=120,746
< entity_id resolves for part_plant  3,406 ids   > 4,212 ids
< training rows per snapshot  11,954             > 12,900
```

**No check changed status. The patch was surgical.**

### Phase 1 gate

| Requirement | Result |
|---|---|
| Guide updated (residuals, `d`, null rates, verify step, constant columns, deviations section) | **pass** |
| Amendment 02 recorded with SHAs | **pass** — additive only, 104 insertions, 0 deletions |
| v6 shortage labels carry both classes | **pass** — 13.49% positive, 107,705 negatives |
| Nothing else in v6 changed | **pass** — 47/49 tables byte-identical, non-shortage rows byte-identical, 0 of 234 checks changed status |

**Phase 2 proceeded.**

---

# PHASE 2 — within-world scores

## 4. Header

| | |
|---|---|
| Device | **MPS**, Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1`, float32 throughout, `num_workers=0` |
| torch | 2.14.0 |
| Total wall-clock | **15,251 s = 4.24 h** for **36 trainings** — 30 grid cells, 4 h⁰ agreement checks, 2 extra seeds for the noise band |
| Peak RSS | **4.13 GB** — MPS never ran out; **no CPU fallback was needed** |
| Channels | **16,072 — all of them**, every snapshot, every cell |
| Grid | **complete. Nothing was cut.** |
| Frozen hyperparameters | `lr 2.5e-4, wd 1e-4, TCN hidden 64, graph hidden 128, B 10, 4 layers, patience 5, cap 40` |
| Code | `ml/models/share.py`, `ml/train/temporal_share.py`, `ml/train/run9_grid.py`, `ml/eval/metrics.py` |

### Convergence per cell — and the cost of freezing one learning rate

Early stopping on the **validation fold** (2024), **patience 5**, cap 40, **restore-best-weights**.
Every reported number is the restored best checkpoint, never the final epoch.

| task | cells | converged on patience | **hit the cap** |
|---|---|---|---|
| **arrival** | 14 | **14** | **0** |
| fill | 10 | 4 | **6** |
| shortage | 10 | 2 | **8** |
| **total** | **34** | 20 | **14** |

*(The 34 grid cells are 30 from the grid proper plus the 4 h⁰ agreement runs. The 2 seed-noise
repeats are excluded here; both are arrival cells and both converged on patience.)*

**Every arrival cell converged.** The headline finding rests entirely on converged runs.

**Fill and shortage largely did not, and the reason is the protocol.** The learning rate was tuned
once on **arrival** and frozen. Arrival converges at that rate in 16–30 epochs; fill and shortage
want 40+. Fourteen cells stopped with their best epoch at 35–39 of 40, still improving.

**This is a real cost of "tune once, freeze", and it is reported rather than tuned around.** Every
fill and shortage number below is a **lower bound**. It does not affect the arrival comparison,
and because the caps fall on both architectures roughly evenly (7 HeteroMP-or-h⁰, 5 SHARE) it
does not obviously favour either — but any fill or shortage margin should be read as provisional.

<details>
<summary>Per-cell convergence, all 34 grid cells</summary>

| world | task | arch | depth | epochs | best ep | stop | wall s | params |
|---|---|---|---|---|---|---|---|---|
| v6 | arrival | none | h0 | 26 | 20 | patience | 357 | 50,369 |
| v6 | arrival | share | h0 | 30 | 24 | patience | 405 | 50,369 |
| v6 | arrival | mp | h0 | 25 | 19 | patience | 338 | 50,369 |
| v6 | arrival | mp | h1 | 18 | 12 | patience | 249 | 83,585 |
| v6 | arrival | mp | h4 | 20 | 14 | patience | 279 | 116,801 |
| v6 | arrival | share | h1 | 20 | 14 | patience | 295 | 793,777 |
| v6 | arrival | share | h4 | 22 | 16 | patience | 387 | 793,777 |
| v7 | arrival | none | h0 | 18 | 12 | patience | 244 | 50,369 |
| v7 | arrival | share | h0 | 16 | 10 | patience | 218 | 50,369 |
| v7 | arrival | mp | h0 | 16 | 10 | patience | 218 | 50,369 |
| v7 | arrival | mp | h1 | 24 | 18 | patience | 326 | 83,585 |
| v7 | arrival | mp | h4 | 22 | 16 | patience | 302 | 116,801 |
| v7 | arrival | share | h1 | 16 | 10 | patience | 237 | 793,777 |
| v7 | arrival | share | h4 | 19 | 13 | patience | 335 | 793,777 |
| v6 | fill | none | h0 | 40 | 39 | **CAP** | 534 | 51,604 |
| v6 | fill | mp | h1 | 36 | 30 | patience | 486 | 84,820 |
| v6 | fill | mp | h4 | 40 | 39 | **CAP** | 545 | 118,036 |
| v6 | fill | share | h1 | 29 | 23 | patience | 426 | 796,228 |
| v6 | fill | share | h4 | 29 | 23 | patience | 509 | 796,228 |
| v7 | fill | none | h0 | 40 | 35 | **CAP** | 534 | 51,604 |
| v7 | fill | mp | h1 | 40 | 39 | **CAP** | 540 | 84,820 |
| v7 | fill | mp | h4 | 40 | 39 | **CAP** | 545 | 118,036 |
| v7 | fill | share | h1 | 40 | 38 | **CAP** | 587 | 796,228 |
| v7 | fill | share | h4 | 11 | 5 | patience | 195 | 796,228 |
| v6 | shortage | none | h0 | 40 | 39 | **CAP** | 533 | 50,369 |
| v6 | shortage | mp | h1 | 40 | 39 | **CAP** | 538 | 83,585 |
| v6 | shortage | mp | h4 | 40 | 38 | **CAP** | 543 | 116,801 |
| v6 | shortage | share | h1 | 40 | 37 | **CAP** | 580 | 793,777 |
| v6 | shortage | share | h4 | 40 | 36 | **CAP** | 699 | 793,777 |
| v7 | shortage | none | h0 | 40 | 39 | **CAP** | 533 | 50,369 |
| v7 | shortage | mp | h1 | 40 | 36 | **CAP** | 540 | 83,585 |
| v7 | shortage | mp | h4 | 40 | 38 | **CAP** | 543 | 116,801 |
| v7 | shortage | share | h1 | 39 | 33 | patience | 568 | 793,777 |
| v7 | shortage | share | h4 | 26 | 20 | patience | 456 | 793,777 |

</details>

### The tuning pass, and why it ran five points

Tuned **once**, on **v6 / SHARE / arrival / h4**, selected on **validation**, then frozen for
every cell and both architectures:

| lr | validation | test-v6 | best epoch | epochs | wall |
|---|---|---|---|---|---|
| 2e-3 | 0.6637 | 0.6587 | 4 | 10 | 184 s |
| 1e-3 | 0.6643 | 0.6604 | 7 | 13 | 244 s |
| 5e-4 | 0.6654 | 0.6598 | 5 | 11 | 199 s |
| **2.5e-4** | **0.6686** | 0.6622 | 13 | 19 | 342 s |
| 1.25e-4 | **0.6686** | 0.6632 | 14 | 20 | 369 s |

The sweep was extended twice because the optimum kept landing on the boundary; it stops at the
plateau. **2.5e-4 chosen** — tied on validation with 1.25e-4, cheaper to train. Selection was on
validation, not test: 1.25e-4 has the better test score and was not chosen.

**HeteroMP inherits a learning rate, a hidden width and a basis count chosen on SHARE.** That
disadvantages it and is noted rather than tuned around.

### Cost checkpoint (P2.6)

```
timed cell  SHARE / v6 / arrival / h4, all 16,072 channels, frozen lr
            342 s, 19 epochs (18.0 s/epoch), best epoch 13, patience
projected   expected  36 x ~450 s     = 16,200 s = 4.5 h
            worst     36 x 40 x 18 s  = 25,920 s = 7.2 h
actual                                = 15,251 s = 4.24 h
```

4.5 h projected against a ~6 h budget, so the full grid ran. Had it tracked toward the worst
case the cut order was fill's h¹ depths first, then shortage's h¹, never arrival.

---

## 5. The sample-size answer

**Phase 3 trained on all 16,072 channels.** The premise that it trained on a quarter of the world
is mistaken, and the code settles it in one line. `cross_world.py` calls

```python
X = torch.from_numpy(norm.transform(snapshot_windows(Wd, t0))).to(DEV)   # lines 160, 209
```

with `chan=None`, and `snapshot_windows` then builds `np.full(W["NCH"], t0_week)` — `NCH` is
`meta["n_channels"]`, 16,072 in both worlds. Every training and evaluation step encoded the whole
panel.

**Where 4,096 actually comes from.** The string appears three times in the repository, and
`reports/phase-3.md` is not one of them:

| location | what it is |
|---|---|
| `reports/phase-0.md:129` | a throughput benchmark — `[4096, 38, 52]`, 12 forwards after warm-up |
| `reports/phase-2.md:50-51` | a demo build — `v6: X (4096, 52, 24)  build 0.02 s` |
| `reports/phase-2.md` §4.2 | an idle-position count quoted "in a 4,096-channel sample" |

The first is a timing harness, the second a smoke test of `build_sequences`, the third a sampled
measurement. None is a DataLoader batch size either — this trainer has no DataLoader; one
optimiser step consumes one whole snapshot.

**So Phase 3's numbers do not need re-reading on this axis.** Both runs used the full population.

**One genuine sub-population, which is not the training set.** The normaliser is fitted on
`np.arange(0, NCH, 8)` — 2,009 channels — on every fourth training snapshot, in Phase 3 and here
alike. That is a scaler-fitting sample; it sets the mean and standard deviation, not which rows
the model sees.

**A real difference from Phase 3 did surface while checking.** `cross_world.py` imports `SPLIT`
but never `FIT_WINDOW`, so **Phase 3 trained on 66 snapshots where every baseline it was compared
against used 44** — the 2019–2025 window was specified for both and applied only to one. This run
applies it. Phase 2 §2 measured the window's cost at 0.15% on arrival for the GBM, so the effect
is small, but Phase 3's absolute figures and this run's are not measured on identical training
folds.

---

## 6. Architectures

### SHARE — as specified

`ml/models/share.py` implements `model_plan.md` "Stage 2":

- **RGCN basis decomposition**, `W_r = Σ_{b=1}^{10} a_rb V_b`, **B = 10**, computed once per layer
  as an einsum, never materialised per edge;
- **relation-blind shared attention**, `e_ij = LeakyReLU(aᵀ[W_r h_i ‖ W_r h_j])`, one vector **a**
  for every relation, softmaxed over each node's **whole** incoming neighbourhood;
- **hidden 128, 4 layers**, returning h⁰…h⁴.

| | SHARE | HeteroMP |
|---|---|---|
| mechanism | basis-decomposed per-relation weights + attention | learned per-relation **means** + a gate |
| aggregation weights | learned per edge, softmax-normalised | uniform within a relation |
| hidden | 128 | 64 |
| depth | 4 layers | 2 rounds = 4 message-passing layers |
| **encoder params** | **730,992** | **h¹ 33,216 · h⁴ 66,432** |
| **full model** | **793,777** | **h¹ 83,585 · h⁴ 116,801** |
| h⁰ (no graph module) | **50,369** | **50,369 — the same model** |

**HADES v3's reference is 752,211; SHARE here is 730,992** — within 3%, a coincidence of two
different schemas landing on similar totals, not evidence of a faithful port.

**On a 3-relation graph the basis decomposition buys nothing, and it should be said plainly.**
Per layer:

```
bases      B x d x d = 10 x 128 x 128 = 163,840
mixing     R x B     =  6 x 10        =      60
self-loop  W_0 + b   = 128 x 128 + 128=  16,512
attention  a in R^2d =                      256
                                       --------
                             per layer = 180,668
                              4 layers = 722,672
             input projection 64 -> 128 =   8,320
```

The graph is channel ↔ {supplier, part, plant}: **three undirected relations, six directed
types**, over 17,119 nodes and 96,432 directed edges. Free per-relation weights would cost
6 × 128 × 128 = 98,304 per layer; the basis scheme costs 163,840 + 60. **Basis decomposition is
more expensive here than the thing it exists to replace.** It saves parameters only when R
exceeds B, and R = 6 against B = 10. The specification's economics assumed R = 20.

**So any SHARE advantage on this graph comes from the attention, not the basis trick.**

**Entity nodes carry no sequence.** A supplier has no `channel_performance_weekly` row, so it
enters as the mean of its member channels' TCN states — content-derived, keeping the model
**inductive**. A learned per-supplier embedding would key on identity, which the rules forbid.

### The shared TCN front-end, and the causality test

Both architectures use the same encoder — kernel 2, dilations 1/2/4/8/16/32, 6 layers, hidden 64,
receptive field 64 weeks, **with the residual connections** Phase 1 made a required element.
46,144 parameters.

**Causality unit test — passes bit-identically:**

```
perturb last timestep by +100 -> max abs diff on EARLIER positions: 0.0
bit-identical: True    allclose: True
receptive field: 64 weeks
residual projections: [Conv1d, Identity, Identity, Identity, Identity, Identity]
```

### h⁰ is architecture-independent — verified, not asserted

`Net` builds **no graph module at all** at depth 0, so a "SHARE h⁰" and a "HeteroMP h⁰" request
construct the identical object. Trained three times independently under three labels:

| world | `none` h⁰ | `share` h⁰ | `mp` h⁰ | max gap |
|---|---|---|---|---|
| v6 | 0.6484 | 0.6480 | 0.6480 | **0.0004** |
| v7 | 0.6529 | 0.6526 | 0.6525 | **0.0004** |

Parameter count identical at 50,369 in all six runs, and every gap is far inside the noise band
below. **h⁰ is a genuine common floor**, so every SHARE-vs-HeteroMP difference is the encoder.

---

## 7. The scores — within-world, with 95% bootstrap intervals

Every interval is a **1,000-resample percentile bootstrap over test-fold rows**. It answers
"how much would this score move on a different draw of test rows from the same world" — not
"how much would it move on a different world", which within-world data cannot answer.

Baselines are computed on the **same label-table rows** as the neural models, same split, same
2019–2025 window, so every number in a row is measured on one population.

### Arrival timing — C-index ↑ (headline)

| world | naive | LightGBM | h⁰ (TCN) | HeteroMP h¹ | HeteroMP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|
| **v6** | 0.5507 [0.5422, 0.5585] | 0.6428 [0.6349, 0.6516] | 0.6484 [0.6401, 0.6560] | 0.6560 [0.6482, 0.6646] | 0.6585 [0.6500, 0.6670] | 0.6585 [0.6499, 0.6672] | 0.6625 [0.6539, 0.6705] |
| **v7** | 0.5574 [0.5483, 0.5667] | 0.6481 [0.6396, 0.6572] | 0.6529 [0.6444, 0.6618] | 0.6608 [0.6529, 0.6695] | 0.6609 [0.6525, 0.6697] | 0.6649 [0.6570, 0.6745] | 0.6664 [0.6579, 0.6755] |

### Arrival timing — ROC-AUC on the binarised late/on-time outcome ↑

| world | naive | LightGBM | h⁰ (TCN) | HeteroMP h¹ | HeteroMP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|
| **v6** | 0.7064 [0.6973, 0.7164] | 0.7570 [0.7491, 0.7655] | 0.7561 [0.7482, 0.7643] | 0.7606 [0.7528, 0.7688] | 0.7614 [0.7536, 0.7695] | 0.7615 [0.7535, 0.7696] | 0.7640 [0.7563, 0.7722] |
| **v7** | 0.7097 [0.7010, 0.7178] | 0.7568 [0.7491, 0.7644] | 0.7559 [0.7485, 0.7633] | 0.7610 [0.7537, 0.7685] | 0.7613 [0.7539, 0.7686] | 0.7627 [0.7554, 0.7700] | 0.7634 [0.7562, 0.7709] |

### Fill rate — CRPS ↓ (headline)

| world | naive | LightGBM | h⁰ (TCN) | HeteroMP h¹ | HeteroMP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|
| **v6** | 0.0632 [0.0612, 0.0652] | 0.0599 [0.0580, 0.0617] | 0.0594 [0.0575, 0.0613] | 0.0600 [0.0581, 0.0620] | 0.0603 [0.0584, 0.0623] | 0.0611 [0.0590, 0.0631] | 0.0594 [0.0575, 0.0613] |
| **v7** | 0.1078 [0.1057, 0.1100] | 0.0988 [0.0969, 0.1008] | 0.0990 [0.0969, 0.1012] | 0.0996 [0.0975, 0.1019] | 0.0992 [0.0971, 0.1014] | 0.0989 [0.0967, 0.1011] | 0.1005 [0.0985, 0.1025] |

### Fill rate — calibration of the binned CDF, expected calibration error ↓

| world | naive | LightGBM | h⁰ (TCN) | HeteroMP h¹ | HeteroMP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|
| **v6** | 0.0152 [0.0124, 0.0223] | 0.0107 [0.0081, 0.0174] | 0.0497 [0.0439, 0.0556] | 0.0769 [0.0709, 0.0829] | 0.0901 [0.0841, 0.0959] | 0.1013 [0.0954, 0.1073] | 0.0565 [0.0515, 0.0625] |
| **v7** | 0.0324 [0.0277, 0.0420] | 0.0141 [0.0117, 0.0215] | 0.0788 [0.0715, 0.0868] | 0.1229 [0.1154, 0.1307] | 0.1158 [0.1083, 0.1236] | 0.1045 [0.0972, 0.1125] | 0.0215 [0.0173, 0.0287] |

### Part shortage — PR-AUC ↑ (headline, honest at a 13-20% base rate)

| world | naive | LightGBM | h⁰ (TCN) | HeteroMP h¹ | HeteroMP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|
| **v6** | 0.4896 [0.4696, 0.5102] | 0.5287 [0.5095, 0.5471] | 0.5722 [0.5524, 0.5917] | 0.6574 [0.6398, 0.6756] | 0.6651 [0.6476, 0.6832] | 0.6678 [0.6504, 0.6856] | 0.6968 [0.6790, 0.7140] |
| **v7** | 0.2487 [0.2319, 0.2668] | 0.4438 [0.4202, 0.4700] | 0.4580 [0.4334, 0.4831] | 0.5001 [0.4764, 0.5252] | 0.5069 [0.4839, 0.5315] | 0.5148 [0.4907, 0.5394] | 0.4859 [0.4611, 0.5108] |

### Part shortage — ROC-AUC ↑ (the number that gets quoted)

| world | naive | LightGBM | h⁰ (TCN) | HeteroMP h¹ | HeteroMP h⁴ | SHARE h¹ | SHARE h⁴ |
|---|---|---|---|---|---|---|---|
| **v6** | 0.7695 [0.7597, 0.7798] | 0.7757 [0.7655, 0.7855] | 0.8038 [0.7931, 0.8133] | 0.8622 [0.8541, 0.8693] | 0.8648 [0.8567, 0.8718] | 0.8694 [0.8615, 0.8765] | 0.8799 [0.8717, 0.8871] |
| **v7** | 0.6960 [0.6829, 0.7098] | 0.7968 [0.7854, 0.8082] | 0.7887 [0.7770, 0.8011] | 0.8280 [0.8174, 0.8392] | 0.8293 [0.8189, 0.8411] | 0.8267 [0.8159, 0.8375] | 0.8094 [0.7976, 0.8213] |



**Two notes on the secondary metrics, both of which changed after a first attempt gave nonsense.**

*Arrival ROC-AUC* binarises the observed arrivals as late when `arrival_week > promise_week`,
where the promise comes from `po_lines.original_promise_date` expressed in weeks after the
snapshot. It is computed on **uncensored test rows only** — 11.5% late on v6, 15.8% on v7.
Scoring must rank on `prediction − promise_week`, not on the raw prediction: a channel with a long
lead time has both a late predicted arrival and a late promise, so only the difference carries the
late/on-time signal. Ranking on the raw prediction scores **below 0.5**.

*Fill calibration* uses **bin reliability**, `ECE = Σ_b |mean predicted P(bin b) − observed
frequency(bin b)|`, not quantile coverage. Quantile coverage is degenerate on this data:
**91.4% of fill labels are exactly 1.0**, so the empirical CDF jumps from 0.084 to 1.0 at the last
bin and every nominal level from 0.1 to 0.9 maps to the same predicted threshold. The coverage
error pins at exactly 0.5 for *any* forecaster, including LightGBM. Bin reliability has no such
failure mode.

---

## 8. Graph contribution — h⁴ against h⁰

Graph share = (above-reference at h⁴ − above-reference at h⁰) / above-reference at h⁴, with the
reference being 0.5 for C-index, the naive CRPS for fill, and the test base rate for shortage.
**h⁰ is the same trained model in both architecture columns** (§6), so the two columns differ only
by the encoder.

| task | world | h⁰ | HeteroMP h⁴ | SHARE h⁴ | MP graph share | **SHARE graph share** | Phase 3 (4,096 / 6 ep) |
|---|---|---|---|---|---|---|---|
| **arrival** | v6 | 0.6484 | 0.6585 | 0.6625 | +6.4% | **+8.7%** | +7.2% |
| **arrival** | v7 | 0.6529 | 0.6609 | 0.6664 | +5.0% | **+8.1%** | +6.7% |
| fill | v6 | 0.0594 | 0.0603 | 0.0594 | −32.4% | **−0.3%** | +9.7% |
| fill | v7 | 0.0990 | 0.0992 | 0.1005 | −2.1% | **−20.0%** | +35.5% |
| shortage | v6 | 0.5722 | 0.6651 | 0.6968 | +19.8% | **+24.9%** | n/m (no negatives) |
| shortage | v7 | 0.4580 | 0.5069 | 0.4859 | +12.8% | **+7.7%** | +3.8% |

*Phase 3's within-world figures, at 4,096-channel demo scale in its report but in fact the full
population, 6 epochs, HeteroMP only.*

**The graph contributes on arrival and on shortage. It does not contribute on fill.** That is
three tasks and three different answers, and it is the shape of the result.

---

## 9. Does SHARE beat HeteroMP?

### The noise band comes first

MPS is not bit-deterministic, and different torch seeds move the weights further still. The same
cell — **SHARE h⁴, v6, arrival** — trained at three seeds:

| seed | C-index |
|---|---|
| 7 | 0.6625 |
| 17 | 0.6597 |
| 27 | 0.6612 |

**spread (max − min) = 0.0028 · sd = 0.0014**

**A gap smaller than 0.0028 is not a difference**, and the rule is held to below. One caveat: the
band was measured on **arrival C-index only**. It is not transferable to CRPS or PR-AUC, which are
different metrics on different scales, so for fill and shortage the bootstrap intervals in §7 do
the work instead.

### Per task

| task | world | depth | SHARE | HeteroMP | margin | vs 0.0028 band |
|---|---|---|---|---|---|---|
| arrival | v6 | h¹ | 0.6585 | 0.6560 | +0.0025 | **inside — not a difference** |
| arrival | v6 | h⁴ | 0.6625 | 0.6585 | +0.0039 | 1.4× — marginal |
| arrival | v7 | h¹ | 0.6649 | 0.6608 | +0.0042 | 1.5× — marginal |
| arrival | v7 | h⁴ | 0.6664 | 0.6609 | +0.0055 | 2.0× — **real** |

**Arrival: SHARE leads in all four comparisons, but only one clears the band by a factor of two.**
One is inside it outright. The honest reading is that **SHARE is probably slightly better on
arrival, by roughly half a point of C-index, and the evidence is not strong enough to be
confident in any single cell.** The consistency of the sign across four independent comparisons is
the better argument than any individual margin.

| task | world | depth | SHARE | HeteroMP | margin | reading |
|---|---|---|---|---|---|---|
| fill | v6 | h¹ | 0.0611 | 0.0600 | −0.0010 | no difference |
| fill | v6 | h⁴ | 0.0594 | 0.0603 | +0.0009 | no difference |
| fill | v7 | h¹ | 0.0989 | 0.0996 | +0.0008 | no difference |
| fill | v7 | h⁴ | 0.1005 | 0.0992 | −0.0013 | no difference |

**Fill: no difference, in either direction.** Every margin is ≤ 0.0013 CRPS, the five
configurations on v6 span 0.0594–0.0611, and all five bootstrap intervals overlap almost
completely. There is no architecture effect on fill to report.

| task | world | depth | SHARE | HeteroMP | margin | reading |
|---|---|---|---|---|---|---|
| shortage | v6 | h¹ | 0.6678 | 0.6574 | **+0.0103** | SHARE, intervals overlap |
| shortage | v6 | h⁴ | 0.6968 | 0.6651 | **+0.0318** | **SHARE, intervals barely overlap** |
| shortage | v7 | h¹ | 0.5148 | 0.5001 | +0.0146 | SHARE, intervals overlap |
| shortage | v7 | h⁴ | 0.4859 | 0.5069 | **−0.0211** | HeteroMP |

**Shortage: SHARE wins three of four, and loses the fourth.** The clearest single result anywhere
in this run is **SHARE h⁴ on v6, 0.6968 [0.6790, 0.7140] against HeteroMP h⁴ 0.6651 [0.6476,
0.6832]** — the intervals barely touch. But the v7 h⁴ cell reverses, and **SHARE h⁴ is worse than
SHARE h¹ on v7** (0.4859 vs 0.5148) while being better on v6 (0.6968 vs 0.6678). Depth preference
is not consistent across worlds. **Eight of the ten shortage cells hit the epoch cap**, so all of
these are lower bounds and none should be treated as settled.

### The answer

**SHARE beats HeteroMP on shortage on v6 and probably on arrival; it makes no difference on fill;
and it loses one shortage cell.** For a 794k-parameter encoder against an 84k-parameter one — 9.5×
the parameters — that is a modest return. **On this graph the basis decomposition is dead weight
(§6), so what SHARE buys is the attention, and the attention buys about half a point of C-index on
arrival and a few points of PR-AUC on shortage.**

---

## 10. Did full data and convergence change Phase 3's conclusion?

**On arrival it holds and grows slightly.** Phase 3 reported +5.5% to +7.2% across four
cross-world cells, of which two were within-world (+7.2% v6, +6.7% v7). This run's within-world
figures with a real SHARE encoder are **+8.7% (v6) and +8.1% (v7)**, and with HeteroMP **+6.4% and
+5.0%**.

| | Phase 3, HeteroMP, 6 epochs | this run, HeteroMP | this run, SHARE |
|---|---|---|---|
| v6 within | +7.2% | +6.4% | **+8.7%** |
| v7 within | +6.7% | +5.0% | **+8.1%** |

**The conclusion holds.** Removing all three qualifications — 6 epochs, the simplified encoder,
and the implied sample-size doubt — leaves the finding intact and slightly stronger with SHARE.
HeteroMP alone comes in a little below Phase 3, which is attributable to convergence and the fit
window rather than to the graph.

**On fill, Phase 3's numbers do not survive, and Phase 3 said they would not.** It reported
+9.7% and +35.5% while declining to defend them as "anything but noise on a small margin". With
convergence they become **−0.3% and −20.0%**. That judgement is confirmed.

**On shortage the comparison is not available** — Phase 3's v6 cell was degenerate and its v7 cell
read +3.8% against this run's +12.8% (HeteroMP) and +7.7% (SHARE), on a task where 8 of 10 cells
here did not converge.

---

## 11. v6 shortage, now that it has negatives

Phase 3 could not read this cell at all: v6's `training_labels` held 45,942 shortage rows with
**zero negatives**, so PR-AUC was 1.0000 by construction and measured nothing.

| | Phase 3 | this run |
|---|---|---|
| v6 shortage, best configuration | **1.0000 ⚠ degenerate** | **0.6968 [0.6790, 0.7140]** (SHARE h⁴) |
| test base rate | — | 0.1967 |
| naive (per-part-plant historical rate) | — | 0.4896 [0.4696, 0.5102] |
| LightGBM | — | 0.5287 [0.5095, 0.5471] |
| ROC-AUC of the same model | — | 0.8799 [0.8717, 0.8871] |

**This is now the strongest cell in the run.** SHARE h⁴ reaches PR-AUC 0.6968 against a 0.1967
base rate and a 0.5287 GBM, with **non-overlapping intervals against both baselines** — the widest
genuine separation anywhere here. The graph contributes **+24.9%** of above-chance signal, more
than double what it contributes on arrival.

**And the reason it was unreadable was the label table, not the world.** The same measurement
apparatus, pointed at the same v6 simulation, produces a clean and strongly-separated result once
the negative class exists. This is the second time in this project that a negative result about
the world turned out to be a statement about the instrument — the first being run 7's fixed
mean-pooling conclusion, revised in Phase 3.

**The caveat stands: this cell hit the 40-epoch cap with its best at epoch 36.** 0.6968 is a lower
bound.

---

## 12. Still failing, with mechanisms

**1. Fill rate — the graph contributes nothing, and the neural heads are badly calibrated.**
CRPS is flat across all five configurations (0.0594–0.0611 on v6), so there is no architecture
effect. Worse, the **calibration is 3–9× worse than LightGBM's**: ECE 0.0497–0.1013 for the neural
heads against 0.0107 for the GBM on v6. Mechanism: the binned cross-entropy head is trained to
maximise likelihood on a distribution that is 91.4% a point mass at 1.0, and it spreads
probability mass across neighbouring bins in a way the GBM does not. A model can match on CRPS
while putting its mass in the wrong bins, and **only the secondary metric shows it.** Anyone
quoting fill-rate *distributions* rather than point forecasts should use the GBM.

**2. Fourteen of thirty-four cells did not converge**, all on fill and shortage, because the
learning rate was tuned on arrival and frozen. Their numbers are lower bounds. The protocol
required a single tuning pass; the cost of that requirement is this, and it is reported rather
than worked around.

**3. Basis decomposition is unjustified on this schema.** R = 6 against B = 10 makes it cost more
than the free per-relation weights it replaces (§6). It is carried because the specification
requires B = 10, not because it earns its place. On a graph with the specified R = 20 the
economics reverse; this graph does not have 20 relations.

**4. Depth preference is inconsistent on shortage.** SHARE h⁴ beats h¹ on v6 (0.6968 vs 0.6678)
and loses to it on v7 (0.4859 vs 0.5148). With 8 of 10 shortage cells unconverged there is no
basis for choosing a readout depth, which is exactly what the guide's §4.2 warns about — depth
must be derived per task, and this run does not derive it.

**5. Arrival's SHARE margin is thin.** One of four comparisons is inside the seed-noise band and
two more are within a factor of 1.5 of it. The sign is consistent, the magnitude is not
well-resolved, and distinguishing them properly would need more seeds per cell than this run
budgeted.

**6. None of this is accuracy on Rane's data** — see the header. `synthetic_rules.md` §17 is
explicit that seed- and world-to-world agreement is not a proxy for generalisation to a real
extract, and within-world scores are weaker evidence still.

---

## Gate

| Requirement | Result |
|---|---|
| Sample-size question settled | **pass** — §5 |
| All 16,072 channels, no subsampling | **pass** |
| SHARE built as specified (B=10, relation-blind attention, hidden 128, 4 layers) | **pass** — §6, 730,992 encoder params |
| Parameter count against v3's 752,211, with the basis-sharing caveat | **pass** — §6 |
| Same TCN both architectures, causality test re-run | **pass** — bit-identical, max diff 0.0 |
| Early stopping, patience ≥ 5, restore-best | **pass** — 20/34 on patience, 14 flagged as CAP |
| Tuned once, frozen | **pass** — §4, one pass on v6/SHARE/arrival |
| Within-world only | **pass** — no cross-world evaluation was run |
| Bootstrap 95% intervals on every score | **pass** — 1,000 resamples, §7 |
| Device noise measured over 3 seeds | **pass** — spread 0.0028, §9 |
| h⁰ shared and verified | **pass** — max gap 0.0004, §6 |
| Cost projected before launching | **pass** — 4.5 h projected, 4.24 h actual, nothing cut |

**Three findings that outrank the gate:**

1. **The graph contributes on arrival (+8.1 to +8.7% with SHARE) and on shortage (+7.7 to +24.9%),
   and not at all on fill.** Phase 3's arrival finding survives the removal of all three of its
   qualifications.
2. **SHARE's advantage over HeteroMP is real but small, and costs 9.5× the parameters.** It is
   clearest on v6 shortage, thin on arrival, absent on fill, and reversed on one shortage cell. On
   a 3-relation graph the basis decomposition is dead weight, so what is being bought is the
   attention.
3. **v6 shortage was never unmeasurable — its labels were unusable.** With the negative class
   restored the same apparatus produces the most strongly separated result in the run, beating
   LightGBM with non-overlapping intervals.
