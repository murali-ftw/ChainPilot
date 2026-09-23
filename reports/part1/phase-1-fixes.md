# Phase 1 — two fixes: the guide, and v6's shortage labels

**Gate: PASSED.** The guide documents the data as it is; the validator gained a
constant-column check as amendment 02; v6's shortage labels now carry both classes at a
13.49% positive rate; and **nothing else in v6 moved** — 47 of 49 tables byte-identical,
946,200 non-shortage label rows byte-identical, and all 234 validator checks returning the
same status before and after.

---

## F1 — `docs/implementation_guide.md`

Five corrections, all measured on `db/gen_v6/seed_1001` and `db/gen_v7/seed_1001` over all
8,598,520 `channel_performance_weekly` rows per world.

### 1. Residual connections are now a required element of step 3.1

The guide specified dilation, depth and width and said nothing about skip connections. Built
literally — six `Conv1d`+ReLU layers, nothing else — the encoder scores **C-index 0.5000**,
exactly chance, while a ridge probe on the identical input tensors scores 0.6464. Step 3.1 now
lists the 1×1 residual projection in the hyperparameter table as **required**, gives the
`z = ReLU(conv(pad(z))) + W₁ₓ₁ z` form with the residual added *after* the causal trim, and
shows both loss curves so the failure is recognisable in one epoch:

```
no residual:   1.0136  0.9995  0.9987  0.9978  0.9972  0.9978   <- flat at 1.0
with residual: 0.9614  0.9229  0.9128  0.9057  0.9012  0.8979
```

A second verify step was added alongside the causality test: the TCN must clear a ridge probe
on the flattened window, or the stack is broken.

### 2. `d = 38` replaced by a derivation

Step 2.1 no longer asserts a number. It gives the computation — drop constant columns, keep one
indicator per surviving nullable, add the calendar block if you fold it in — and reconciles it
against specification §3.3 term by term:

| term | spec §3.3 | measured | why |
|---|---|---|---|
| numeric value channels | 18 | **14** | four columns constant zero |
| missing-indicators | 13 | **10** | three dropped nullables take their indicators |
| calendar block | 7 | 7, not folded in by `ml/data/cache.py` | optional |
| **d** | **38** | **24** as built, 31 with calendar | |

### 3. The null-sparsity expectations, corrected

The guide and `benchmark_specification` §3.2 state `fill_rate` and `ack_gap_ratio` are 81.4%
null and `load_ratio` 81.8%. Measured:

| column | spec | **v6** | **v7** |
|---|---|---|---|
| `fill_rate` | 81.4% | **5.52%** | **5.95%** |
| `ack_gap_ratio` | 81.4% | **5.52%** | **5.95%** |
| `load_ratio` | 81.8% | **0.00%** | **0.00%** |
| `lead_time_actual_days` / `_ratio` | 72.6% | **5.52%** | **5.95%** |

Step 2.2 now carries this table and the mechanism: both generators forward-fill the level and
rolling columns across idle weeks, so **the sparsity is expressed as `qty_ordered == 0` — on
86.3% of channel-weeks in v6 and 88.6% in v7** — rather than as nulls. The indicator channels
are consequently near-vacuous, mean **0.9503** (v6) and **0.9465** (v7) over the ten surviving
nullables, against the ≈0.181 the specification implies. The guide now says not to size an
imputation, masking or loss-weighting policy off the §3.2 figures.

### 4. The failing verify step, replaced

*"For a known idle week the `fill_rate` channel is 0.0"* fails by design: **175,481 of 212,992
idle positions carry a non-zero forward-filled value**. It is struck through in the guide with
the measurement attached, and replaced with four checks that hold on both worlds — that
`is_active_week` is binary with mean in (0.10, 0.15), that an idle week has `active == 0` and
`pad == 1` simultaneously, that value and indicator are coherent wherever a null does occur,
and that the indicators are vacuous. That last one is asserted deliberately, so nobody trusts
them.

`is_active_week` is now documented as **the** channel carrying the idle/active signal, and as
one that must be promoted to an explicit value channel for the model to see it at all.

### 5. Four store columns recorded as known-unpopulated

`revision_count`, `days_since_last_short`, `weeks_since_last_activity` and
`weeks_since_last_receipt` hold **one distinct value — zero — across all 8,598,520 rows of
both worlds**. They are generator placeholders that were never written. The guide now says to
exclude all four, and to exclude the three nullables' indicators with them.

One consequence is called out specifically: specification §3.1 presents
`weeks_since_last_activity` and `reporting_lag_days` as jointly replacing `staleness_days`.
Only `reporting_lag_days` is populated, so the staleness gating layer of step 5.0 has one input,
not two.

### The new section

**[Known deviations from spec](../docs/implementation_guide.md#known-deviations-from-spec)** was
added to the contents and before Troubleshooting. It carries all five as a summary table plus a
short subsection each, so the guide now documents the data as generated rather than as designed.

---

## F1b — `db/validator.py` amendment 02: constant-column check

| | |
|---|---|
| SHA before | `ed4742e0bef9d60af7c5b2a2550b590b27e6b217574df5b17877ba99be57ac56` |
| SHA after | `4dbb229836de2f3bcfd39ad4ff343b4c9b568f2d6c7e1b10591d167360dfb2ce` |
| Diff | `1 file changed, 104 insertions(+)` — **no deletions, no alterations** |
| Record | [`docs/validator_amendment_02.md`](../docs/validator_amendment_02.md) |

Every declared column present in a table accumulates a distinct-value set capped at two entries
during the existing Tier 1 row pass. Columns with exactly one distinct value are sorted into
three buckets, and **only one is gated**:

| bucket | severity | reason |
|---|---|---|
| **all-zero numeric** | **FAIL** | there is no reading under which a `DECIMAL` that is 0.0 in eight million rows is correct |
| all-blank | warn | often the documented open-ended convention — empty `effective_to` means still effective |
| single non-zero literal | warn | often a genuinely uniform dimension — one country, one currency |

The narrowness is deliberate. A gate that fired on all three would be ignored, which is this
project's "gate that cannot fail" failure mode run in reverse. Both non-gated buckets are still
reported in full, per table and in aggregate.

**What it finds — 29 all-zero numeric columns across 5 tables**, identical in both worlds, plus
21 all-blank and 85 single-literal as warnings:

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

## F2 — v6's shortage labels

### Step 1 — determinism, proved first

`db/gen_v6/generator_v6.py --seed 1001` re-run into a scratch directory, content-hashed against
`db/gen_v6/seed_1001` table by table:

| | |
|---|---|
| tables byte-identical | **48 of 49** |
| the exception | `dataset_coverage.csv` |
| its 9 substantive columns | **identical** (`f7dc1fc0…` both) |
| what differs | `assessed_ts`, a wall-clock stamp written at emit time |
| `training_labels.csv` | **byte-identical**, `607c26ad…` |

**v6 reproduces.** The rebuild path was therefore available, and the fallback derivation from
`part_demand_weekly` was not needed.

### Step 2 — the backport, and only the backport

The v7 generator's shortage-label fix is one change: sample the part × plant **universe**
(`PP_SAFETY > 0`) rather than only the part-plants that went short. Applied to a scratch copy of
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
imported, and the point of v6 as a different regime survives.

### Step 3 — writing only the shortage rows

The patched run emits **more** shortage rows than the original, which advances the `label_id`
counter and consumes a different number of RNG draws. Both effects propagate: 19 tables emitted
after `training_labels.csv` differ in the patched run, and so do the non-shortage label rows
from the first shortage block onward. **This is why the corrected file was not simply swapped
in.**

Instead the corrected shortage rows were spliced into the original file. Non-shortage rows are
copied through as the **original raw bytes**, and new shortage rows are inserted at their
structural position (after `capacity_strain`, before `demand_drift`, per snapshot) with
`label_id`s numbered above every id used anywhere in the original table.

| | |
|---|---|
| non-shortage rows, sha256 **before** | `1ab23a8759a2cf8793a511815efe774af8f9b89603da27db6156028557e26488` |
| non-shortage rows, sha256 **after** | `1ab23a8759a2cf8793a511815efe774af8f9b89603da27db6156028557e26488` |
| **identical** | **yes** — 946,200 rows |
| duplicate `label_id` in the output | 0 |
| all 47 other tables vs the reproduction run | **unchanged** |

Row counts by task, before and after: `arrival_week` 332,000 · `fill_rate` 332,000 ·
`capacity_strain` 207,500 · `demand_drift` 74,700 — all unchanged. `shortage_qty` 45,942 →
**124,500**.

### Step 4 — verification

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
built from the larger *condition* set, of which escalated is a subset. So the test is directional:

- of 1,293 rows with an escalated episode inside the forward window `(t₀, t₀+12]`, **not one has
  label 0**, and 1,257 have `label_value ≥` that escalated mass;
- the 36 that do not are explained — **32** have an episode running past the 12-week window, so
  the episode total exceeds the in-window mass by construction; the remaining **4** sit on a
  week boundary moved by the ±6-day recording jitter in `shortage_start_ts`;
- of 11,674 rows whose only escalated episodes are **at or before** t₀, **65.06% carry label
  exactly 0**. A label leaking backward could not produce that.

### Step 5 — the validator, run before and after

Both runs use `validator.py` at amendment 02, so the new gate is present in both and the diff
isolates the label change.

| | pre-fix | post-fix |
|---|---|---|
| checks emitted | 234 | 234 |
| **checks whose status differs** | — | **0** |
| verdict | NOT USABLE | NOT USABLE (identical) |
| FAILED list | identical | identical |

The FAILED list on both: `channel_performance_weekly`, `inventory_position_weekly`,
`inventory_snapshots`, `quality_inspections`, `supplier_performance_weekly`, `no all-zero
numeric column` — six from amendment 02's new finding — and `right-censored tail is the right
size [2019-2025]`, which pre-existed.

Every line that moved between the two runs is a shortage-label count or a row total derived from
one:

```
< training_labels                                992,142 rows      > 1,070,700 rows
< shortage_qty  n=45,942  uncensored=44,573      > shortage_qty  n=124,500  uncensored=120,746
< entity_id resolves for part_plant  3,406 ids   > 4,212 ids
< training rows per snapshot  11,954             > 12,900
```

**No check changed status. The patch was surgical.**

### Step 6 — the original is kept

`db/gen_v6/seed_1001/training_labels.pre_fix` — 157,163,529 bytes,
sha256 `607c26ada4250c6bf3a32ad7bc5e8910bf6847226cbb1883321d008f0a958d3c`, the exact bytes the
Phase 2 and Phase 3 numbers were measured on. Deliberately **not** given a `.csv` extension, so
the validator's `discover()` does not pick it up as an extra table.

The new file is sha256 `dd8ba0127343ed3ac3bbbc196fd1aa6a161ff3b8ef91f519f521c418d2c4ff54`.

**What this does and does not invalidate.** Every number in `validation5.md`–`validation7.md`,
`reports/phase-2.md` and `reports/phase-3.md` was measured on tables that did not change, with
one exception: **Phase 3's shortage column for v6**, which was uninterpretable precisely because
of this defect and which the report itself flagged as such. Those two cells are re-measured in
`results/temporal_share_1.md` §8.

---

## Gate

| Requirement | Result |
|---|---|
| Guide updated (residuals, `d`, null rates, verify step, constant columns, deviations section) | **pass** |
| Validator amendment recorded with SHAs | **pass** — additive only, 104 insertions, 0 deletions |
| v6 shortage labels carry both classes | **pass** — 13.49% positive, 107,705 negatives |
| Nothing else in v6 changed | **pass** — 47/49 tables byte-identical, non-shortage label rows byte-identical, 0 of 234 checks changed status |

**Phase 2 may start.**
