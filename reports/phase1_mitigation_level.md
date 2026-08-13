# Phase 1 (extension) — Instrument and Estimate Mitigation Level (HADES V3, Layer 3)

**Date:** 2026-08-13
**Repo:** `/Users/muralik/Documents/Programs/HADES_v3`, branch `HADES-v3`
**Prerequisites:** `reports/phase0_audit.md` (all five checks pass);
`reports/phase1_latent_state.md` (Supply Stress and Recovery Capability cleared to Phase 2;
Mitigation Level identified but not built).

**What this phase adds.** `mitigation_level` (`db/generate_dataset.py:1191`) was the one candidate
the previous phase found and deliberately did not build, because it cannot be recovered post-hoc.
This phase makes the generator change required to capture it, proves that change is inert, and
estimates the state on the frozen backbone with the identical methodology used for the other
states.

---

## Step 1 — Instrumentation

### Why a generator change was unavoidable

`own_stress()`/`stress()` are pure functions of module state that survives the run, so the previous
phase materialised those targets post-hoc with no generator change at all. `mitigation_level()` is
different, for two independent reasons visible in `agent_observe()` (`db/generate_dataset.py:1166-1188`):

1. **It destructively drains state.** `agent_pending[sup_id]` is consumed into the `agent_seen`
   accumulator as the weekly walk advances. After the run, `agent_seen` holds totals accumulated
   through `T_END`, so calling `mitigation_level(s, early_week)` afterwards returns a value
   contaminated by outcomes that had not yet been observed at that week.
2. **It asserts clock monotonicity.** `agent_frontier` records the last `as_of` per supplier and
   raises `TemporalLeak("agent clock moved backwards")` on any earlier read — so a post-hoc sweep
   over historical weeks does not merely return the wrong number, it raises.

The value the simulation actually used therefore has to be recorded at the moment it is used.

### The change

Two edits, both inside the Mechanism-E agent machinery. Full diff:

```diff
 agent_pending = {}     # sup_id -> [(recorded_at, was_late)] not yet visible
 agent_seen = {}        # sup_id -> [n_observed, n_late]  (visible history only)
 agent_frontier = {}    # sup_id -> as_of of the last advance, for the monotonicity assert
+
+# (sup_id, as_of) -> mitigation in [0,1]. NEVER emitted -- same standard as RESILIENCE
+# and own_stress; `mitigation` is already on ml/data/loader.py's HIDDEN_STATE_COLUMNS
+# guard list, which asserts no emitted table carries a column by that name.
+#   [...rationale comment...]
+MITIGATION_HISTORY = {}
```

```diff
 def mitigation_level(sup_id, as_of):
     if not RESILIENCE:
         return 0.0
     seen = agent_observe(sup_id, as_of)
     if seen[0] < 3:
+        MITIGATION_HISTORY[(sup_id, as_of)] = 0.0
         return 0.0                       # not enough observed history to react to
     observed_late_rate = seen[1] / seen[0]
-    return min(1.0, observed_late_rate * (0.5 + RESILIENCE[sup_id]))
+    _mit = min(1.0, observed_late_rate * (0.5 + RESILIENCE[sup_id]))
+    MITIGATION_HISTORY[(sup_id, as_of)] = _mit
+    return _mit
```

### Why this is inert, argued before it was tested

- **No RNG draw.** `mitigation_level()` and `agent_observe()` contain no call into `random`. The
  recorder adds only dict writes. RNG stream position is therefore untouched — the property every
  variant's byte-identity depends on.
- **No control-flow change.** Both return paths return exactly the value they returned before;
  `min(1.0, ...)` is computed identically and merely bound to a name first. No branch added or
  removed, no call added or removed.
- **Recording at the point of use, not at a re-derivation.** The value stored is the same object
  the simulation goes on to use, so the record cannot drift from the behaviour it describes.
- **Non-E variants write nothing.** The `if not RESILIENCE: return 0.0` guard precedes every write,
  so on Variant 0 (and every other non-E variant) `MITIGATION_HISTORY` stays empty and not one dict
  write occurs. Verified: Variant 0 produces **0 entries**.
- **Idempotent under repeated calls.** `mitigation_level()` is called more than once per
  supplier-week in the replenishment loop (`:1224`, `:1235`). The second call at the same `as_of`
  finds `agent_pending` already drained, so `agent_observe` returns the same `agent_seen` and the
  same value is rewritten. The record is not double-counted.
- **Never emitted.** No `write()` call references `MITIGATION_HISTORY`, and
  `ml/data/loader.py::verify_no_hidden_state` already carries `mitigation` in
  `HIDDEN_STATE_COLUMNS` — the existing guard covers this state without modification.

### Byte-identity proof — the mandatory check

Baselines were generated from the **unmodified** generator before any edit, into fresh temp
directories, and hashed. The generator was then modified and the same variants regenerated and
re-hashed.

First, the anchor itself was validated — a fresh pre-change build of Variant 0 at `v1` reproduces
the copy already on disk, confirming the baseline is trustworthy rather than merely self-consistent:

```
$ diff before_v0.sha256 ondisk_v0.sha256      # db/csv_v1scale/v0_seed42
IDENTICAL — the v1 byte-identity anchor reproduces exactly
```

Then, before vs. after the instrumentation:

```
================ BYTE-IDENTITY: before vs after instrumentation ================
variant 0: IDENTICAL   (20/20 .csv.gz files byte-for-byte)
variant E: IDENTICAL   (20/20 .csv.gz files byte-for-byte)
variant K: IDENTICAL   (21/21 .csv.gz files byte-for-byte)
```

**Variant 0 is the anchor the brief requires, but Variant E and K are the checks that actually
matter here**, because Variant 0 has Mechanism E off and therefore never executes the instrumented
code path at all. E and K exercise it on every replenishment decision and still produce identical
compressed bytes. All three compared as `.csv.gz` — compressed bytes, so a timestamp leaking into a
gzip header would be caught.

Second confirmation, per Step 1.4, on an instrumented variant:

```
$ ./venv/bin/python db/run_benchmark.py --verify-determinism --variant E --config v1
determinism: variant E, 20/20 .csv.gz files byte-identical across two runs
```

**Byte-identity check: PASS.** Nothing moved.

### What the recorder captured

```
variant 0: MITIGATION_HISTORY entries=0  (empty — Mechanism E off, as designed)
variant E: MITIGATION_HISTORY entries=55,200  nonzero=18,087 (32.8%)
           range=[0.0000, 0.7333]  weeks=92  suppliers=600
```

600 of 800 suppliers are covered, not all. Only suppliers reachable as a replenishment candidate
through `prod_bom_sup` are ever evaluated, so 25% of suppliers have no mitigation record at any
week. This is a data property and a coverage ceiling of exactly the kind
`docs/14_Project_Roadmap.md` §4 risk 6 already records for Mechanism E/F states — not a modelling
choice, and not something to optimise away.

---

## Step 2 — Extraction path

`ml/confirm_latent_states.py` now reports `mitigation_level` as **confirmed** rather than
*needs instrumentation*, and `ml/latent_state_head.py::latent_targets()` materialises it as a
supervision target.

**Provenance, stated explicitly because it is the opposite of every other target in this project.**
Supply Stress and Recovery Capability are **recomputed after the run** from surviving module state.
Mitigation Level is **recorded during the simulation** — it is a log, not a re-derivation. That
distinction is the entire reason instrumentation was required, and it means this target's
correctness rests on the recorder being at the point of use (argued and tested above) rather than
on a function being replayable.

**As-of alignment.** `MITIGATION_HISTORY` is keyed by the *weekly* walk timestamp; snapshots are
monthly. Each snapshot `t0` is matched to the **most recent recorded week at or before `t0`**, which
is proper as-of semantics and never reads forward. A supplier with no record yet at a given `t0` is
omitted from that snapshot rather than zero-filled — zero is a meaningful value here ("observed
trouble, not reacting"), and conflating it with "no data" would manufacture negatives.

**Modules extended, and why not the ones the brief named.** The brief suggested extending
`ml/extract_hidden_state.py` / `ml/extract_mechanism_state.py`. Those are inherited V2 files that
Phase 0 certified byte-identical to HADES_v2, and every phase so far has preserved that property.
In V3 the module that actually serves this role — the one already exposing `own_stress`/`RESILIENCE`
post-hoc — is `ml/confirm_latent_states.py`, written for the previous phase. It was extended
instead, so no inherited file changed. The two V2 extractors remain byte-identical.

---

## Step 3 — Depth sweep

**Command** (identical methodology to the previous phase — same binarisation, same temporal split,
same AUC implementation, same two floors; only the target is new):

```
$ ./venv/bin/python -u ml/latent_state_head.py --variant E --seeds 42,43,44,45,46 \
      --config v1 --init-seeds 0,1,2,3,4 --out out/phase1b/heads_E_mitigation.json
```

Variant E, 5 dataset seeds × 5 init seeds. No depth was assumed — the full h⁰…h⁴ sweep was run.

| state | depth | AUC | init floor | dataset floor | above .5 | clears floor | sign-consistent | pos/test |
|---|---|---|---|---|---|---|---|---|
| **mitigation_level** | h⁰ | 0.9838 | 0.0018 | 0.0118 | +0.4838 | YES | yes | 9,582/18,834 |
| **mitigation_level** | **h¹** | **0.9855** | 0.0022 | 0.0119 | +0.4855 | YES | yes | 9,582/18,834 |
| **mitigation_level** | h² | 0.9691 | 0.0191 | 0.0803 | +0.4691 | YES | yes | 9,582/18,834 |
| **mitigation_level** | h³ | 0.9743 | 0.0103 | 0.0588 | +0.4743 | YES | yes | 9,582/18,834 |
| **mitigation_level** | h⁴ *(markov)* | 0.9578 | 0.0082 | 0.1038 | +0.4578 | YES | yes | 9,582/18,834 |

Best depth is **h¹** — which happens to match Recovery Capability's best depth, but the sweep was
run without assuming it, and the differences between depths here are small relative to h⁰.

**The number is 0.9855, and it is not what it looks like.** h⁰ — the raw input projection, before
any message passing — already reaches **0.9838**. The lift from the best depth over h⁰ is
**+0.0017 against a 0.0119 floor**: the encoder contributes nothing measurable. In the previous
phase this pattern (h⁰ ≈ best depth) was a caution flag on Supply Stress. Here it is extreme, and
it prompted a dedicated control rather than a write-up.

### The observability control — the finding that actually matters

`mitigation_level = min(1, observed_late_rate × (0.5 + RESILIENCE[sup]))` (`:1191`), and
`observed_late_rate` comes from `agent_seen`, the supplier's own **recorded** shipment outcomes.
Meanwhile `supplier_temporal_features` **emits** `on_time_rate_30d/90d/180d`, `trend_slope`,
`lateness_variance`, `days_since_last_late` and `shipment_count_180d`
(`db/generate_dataset.py:1485`), and `ml/data/loader.py:309-315` feeds all of them straight in as
Supplier node features. The `seen[0] < 3 → return 0.0` branch additionally ties the target to
observed shipment count, which is itself an emitted column.

`ml/mitigation_observability_control.py` tests this with **no encoder and no backbone at all** —
just the emitted columns and an ordinary least-squares linear probe:

```
OBSERVABLE-ONLY CONTROL — no encoder, no frozen backbone, emitted columns only
  mean AUC from on_time_rate_90d ALONE (one emitted column) : 0.3052  (0.6948 sign-corrected)
  mean AUC from all emitted supplier_temporal_features cols : 0.9750
```

| estimator | AUC |
|---|---|
| emitted `supplier_temporal_features` columns, linear probe, **no encoder** | **0.9750** |
| frozen SHARE + Markov representation, MLP head, best depth h¹ | 0.9855 |
| **difference attributable to the entire representation pipeline** | **+0.0105** |

The whole frozen backbone buys **+0.0105 over a linear probe on emitted columns**, against an h¹
dataset-seed floor of 0.0119 — i.e. **the lift does not clear its own floor**.

**Conclusion: `mitigation_level` is not meaningfully latent on this benchmark.** It is ~97.5%
reconstructible from columns the model already reads as inputs. This is the same disqualifier that
dropped Inventory Health in `reports/phase1_latent_state.md` — a quantity that is a
near-deterministic function of emitted, already-consumed features — with the difference that here
it had to be measured rather than read off the schema, because `mitigation_level` itself is never
emitted and sits on the `HIDDEN_STATE_COLUMNS` guard list. **Being absent from the CSVs is not the
same as being unobservable**, and this phase is the first place in the project where those two
came apart.

---

## Step 4 — Comparison against the static precedent

| state | provenance | best depth | AUC | lift over h⁰ | lift over emitted-only | verdict |
|---|---|---|---|---|---|---|
| Recovery Capability (static) | recomputed post-hoc | h¹ | 0.6530 | **+0.0746** (clears 0.0310) | n/a | genuinely latent |
| **Mitigation Level (dynamic)** | **recorded in-simulation** | h¹ | **0.9855** | +0.0017 (floor 0.0119) ✗ | **+0.0105** (floor 0.0119) ✗ | **observable, not latent** |

**Is the dynamic state more recoverable than its static parent? Yes — 0.9855 vs 0.6530 — and the
comparison is misleading.** The gap is not evidence that dynamic operational state is easier to
estimate than static capability. It is evidence that the two targets differ in *observability*:
`RESILIENCE` is a hidden draw that never touches an emitted column, while `mitigation_level` is
dominated by an observed late rate that is emitted three ways. Reporting 0.9855 alongside 0.6530 as
though they measure the same kind of achievement would be the single most misleading thing this
phase could do.

**Class balance — the failure mode Step 4 asked to watch for.** Checked, and it is not a problem
here: **9,582 positives in 18,834 test rows (50.9%)**, versus `IDIO`'s 1.1–1.2% that sank Supplier
Reliability. Positive class defined by **median split of the continuous mitigation value, threshold
taken from the training split only** — identical to the binarisation used for `own_stress` and
`RESILIENCE`, so the numbers sit on the same basis as the existing table. Because ~2/3 of raw
weekly records are exactly 0.0, the median split is close to "is this supplier reacting at all at
`t0`".

**Coverage.** Test rows are 18,834 rather than 24,000 because only 600 of 800 suppliers are ever
evaluated for mitigation (`prod_bom_sup` reachability), and suppliers with no record yet at a given
`t0` are omitted rather than zero-filled.

---

## Step 5 — Gate verdict

**Gate: technically cleared, substantively FAILED. Mitigation Level is NOT recommended as a
Layer 3 latent-state estimator, and does NOT displace Recovery Capability.**

Against the literal gate criteria it passes: AUC 0.9855 at h¹, clears its reproduction floor
(0.4855 above chance vs. a 0.0119 floor), sign-consistent across all five dataset seeds. Reporting
that as a pass without the control would be indefensible, so both are stated:

| criterion | result |
|---|---|
| clears reproduction floor (max of init/dataset) | ✅ yes, by a wide margin |
| sign-consistent across 5 dataset seeds | ✅ yes, all depths |
| class balance adequate | ✅ yes, 50.9% |
| **is the state actually latent?** | ❌ **no — 0.9750 from emitted columns alone** |
| **does the representation contribute?** | ❌ **no — +0.0105, below its 0.0119 floor** |

**Cleared to feed Phase 2 — but as an observable input, not as a Layer 3 estimate.** The
distinction is practical, not semantic. Phase 2's SCM must represent `mitigation_level` if the
generator uses it causally (it does — it drives the replenishment trigger at `:1225` and order
quantity at `:1235`). The finding here is *good news* for Phase 2: mitigation can be supplied to
the SCM at ~0.975 fidelity from emitted features alone, with no Layer 3 head and no privileged
read. What it cannot do is serve as evidence that Layer 3 recovers hidden operational state.

### Standing recommendation for Layer 3

Unchanged from the previous phase: **Supply Stress** and **Recovery Capability** remain the two
states cleared as genuine latent-state estimates, with Recovery Capability the only one whose
encoder lift clears its floor. Mitigation Level is added as a high-fidelity *observable* input.

### What this phase changes about the project's method

The previous phase used two tests for "is this state latent": is it emitted to CSV, and is it
already a model input feature. Both are schema-level checks, and `mitigation_level` passes both —
it is never emitted and is on the `HIDDEN_STATE_COLUMNS` guard list. It still turned out to be
~97.5% observable. **A third check is needed and should be applied to every candidate state:
measure the state against emitted columns directly, with no encoder, before attributing any AUC to
the representation.**

That check was then run against the two states the previous phase cleared, before letting Phase 2
consume them:

```
$ ./venv/bin/python ml/mitigation_observability_control.py --variant E --seeds 42,43,44,45,46 --config v1

state                      emitted-only AUC   backbone head AUC   representation adds
mitigation_level                     0.9750              0.9855               +0.0105
supply_stress                        0.5695              0.6509               +0.0814
recovery_capability                  0.5303              0.6530               +0.1227
```

| state | emitted-only | head | representation adds | its floor | verdict |
|---|---|---|---|---|---|
| mitigation_level | 0.9750 | 0.9855 | +0.0105 | 0.0119 | ❌ below floor — **observable** |
| supply_stress | 0.5695 | 0.6509 | **+0.0814** | 0.0256 | ✅ clears — **genuinely latent** |
| recovery_capability | 0.5303 | 0.6530 | **+0.1227** | 0.0310 | ✅ clears — **genuinely latent** |

**This retroactively strengthens the previous phase's two cleared states rather than undermining
them.** Supply Stress and Recovery Capability sit at 0.5695 and 0.5303 from emitted columns —
barely above chance — and the frozen representation adds +0.0814 and +0.1227 respectively, both
several times their reproduction floors. They are latent in the substantive sense, not merely the
schema sense.

It also sharpens a caution the previous phase raised. That phase worried that Supply Stress's h⁰
score (0.6174) meant the encoder added little. This control shows h⁰ is **not** a clean
"observable-only" baseline: h⁰ is the encoder's `lin_in` projection, which is *trained*, and it
reads every supplier feature rather than just `supplier_temporal_features`. Against a genuinely
untrained, emitted-columns-only baseline the encoder's contribution to Supply Stress is +0.0814,
substantially larger than the +0.0335 h⁰-relative figure. **The h⁰ comparison understates the
representation's value; the emitted-only comparison is the honest one.** Both are now reported.

---

## Backbone and inherited-file integrity at the close

```
$ shasum -a 256 ml/models/{rgcn_attn_encoder,rgcn_attn_markov_encoder,heads}.py
IDENTICAL (all three, matching Phase 0's recorded digests)

$ diff -rq --exclude=__pycache__ --exclude=.cache ml HADES_v2/ml
(only additions: confirm_latent_states.py, latent_state_head.py,
 mitigation_observability_control.py — plus the 15 documented removals)
```

`ml/extract_hidden_state.py` and `ml/extract_mechanism_state.py` remain byte-identical to V2.
SHARE and the Markov readout were not retrained or modified; `assert_backbone_frozen()` ran on
every head fit and never fired.

**`db/generate_dataset.py` is intentionally modified** — that is this phase's deliverable. It is
the first change to the generator in V3. Its inertness is established above: variants 0/E/K byte-
identical at the `v1` preset before vs. after, and `--verify-determinism` clean on Variant E.
