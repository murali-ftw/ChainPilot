# Step 8 — Claim B: Dyadic Risk Reweighting

**Scope:** implement Claim B (dyadic risk reweighting) per `docs/10_AI_ML_Documentation.md`
§8.5 and `docs/05_Database_Design.md` §6.26 — deliberately NOT a learned model component: a
deterministic formula applied on top of an existing `risk_scores` row, kept in a separate
output table (`supplier_dyadic_risk`) so the two are never conflated.

---

## Device allocation — MPS for training, CPU for everything else, and why

Per this round's instruction to check which hardware suits which task rather than defaulting
everything to one device:

- **Training (SHARE + Variant A, `rgcn_attn_rung5_a`) ran on MPS.** This exact architecture's
  ops were already verified correct (forward+backward, CPU vs MPS, exact match) and faster
  (~2.5x) in `reports/layer_2.md` Round 5, and reused unchanged in every Step 7/7b round since.
  Re-running an identical correctness/speed check on unchanged code would reproduce the same
  result at real compute cost for no new information, so this round cites that prior
  verification rather than repeating it — a legitimate, documented reuse, not a skipped check.
- **Everything else ran on CPU, because there was no device choice to make.** The three dyadic
  weight queries (SQL + pandas), the `reordering_rate` computation (pure-Python loops over at
  most a few thousand customer-supplier pairs), and the `supplier_dyadic_risk` persistence step
  never touch a `torch.Tensor` — there is no MPS-accelerated path for pandas/SQL/pure-Python,
  and routing them through tensors just to pick a "device" would add conversion overhead for
  zero benefit. Only the encoder forward/backward pass is genuinely GPU/MPS-relevant.

---

## A real bug caught before it silently produced empty results

The original design (and this round's own initial task framing) assumed `shipments` directly
carries both `supplier_id` and `order_id`, letting a dyadic (customer, supplier) pair be read
off one table with a simple join. **Checked live against the database before trusting the
query, and false**: `shipments.order_id` and `shipments.supplier_id` are **never both populated
on the same row** — confirmed 0 of 39,184. Shipments are typed by leg: supplier→firm (inbound
component delivery, `supplier_id` set) or firm→customer (outbound product delivery, `order_id`
set), never both. A customer's order and a supplier's component are connected only through the
BOM chain: `orders → order_items → product_components → components.supplier_id`.

A first dry run (3 epochs, throwaway rows) caught this immediately — both weight queries
returned 0 rows, and 0 customers had any supplier relationship at all, which is exactly the
kind of silent-empty-result failure this project's "dry run before the real run" discipline
exists to catch. `ml/models/dyadic.py`'s two SQL functions were rewritten to go through the BOM
chain instead; re-verified live before the full pilot launched: 28,529 (customer, supplier)
pairs for `order_volume_share`, 1,923 for `fulfilment_preference_weight` (6.7% coverage — see
below), 1,295 customers with ≥2 supplier relationships (the population `reordering_rate` is
computed over).

---

## The three input weights (`ml/models/dyadic.py`)

**`order_volume_share`** — a customer's share of a supplier's total BOM-attributed order
volume over a trailing 90-day window as of the scoring `t0` (matching this project's own
`on_time_rate_90d`-style trailing-window convention). Volume is attributed to a supplier via
`order_items.quantity × product_components.quantity_required × components.unit_cost`, for BOM
rows valid as of the order's `placed_at`, summed per (customer, supplier) and normalized by the
supplier's total across all customers.

**`contract_priority_weight`** — read directly from `customers.priority_tier`
(`customer_priority_tier` ENUM, confirmed live: `strategic`/`standard`/`low`), via an explicit
documented mapping:

| Tier | Weight |
|---|---:|
| strategic | 1.0 |
| standard | 0.6 |
| low | 0.3 |

**`fulfilment_preference_weight` — the lead signal.** Derived from historical rationing
behavior, split across the two shipment legs the bug-fix above required: (1) a supplier's own
inbound shipments determine which weeks it was capacity-constrained (≥2 concurrently delayed);
(2) for each customer, among their own outbound order shipments whose order touches that
supplier via the BOM chain AND whose dispatch week falls in one of the supplier's constrained
weeks, the fraction still delivered on time. **Why the encoder structurally cannot already see
this**: it is a customer-level aggregate over Shipment-entity delay-transition history, joined
through the BOM chain — never, and confirmed live to never be, a `Customer` node feature
(`ml/data/features.py::customer_features_asof`'s *entire* feature set is the `priority_tier`
one-hot, nothing else). Coverage: 1,923 of 28,529 pairs (6.7%) have enough constrained-week
history to compute this weight; the rest carry `NULL` here, honestly, rather than a fabricated
neutral value (matching `supplier_dyadic_risk.fulfilment_preference_weight`'s own nullable
column).

## The combination formula f()

```
combined_protection = weighted_average(order_volume_share, contract_priority_weight,
                                        fulfilment_preference_weight;
                                        weights = [0.25, 0.25, 0.50])
                       -- fulfilment_preference_weight leads at 0.50 of total weight, per
                       -- §8.5's own instruction; renormalized over whichever inputs are
                       -- actually available for a given pair (no NULL fabrication)

f = 1 + 0.6 × (0.5 − combined_protection)

dyadic_risk_score = clip(global_risk × f, 0, 1)
```

`combined_protection` is centered at 0.5 ("average treatment, no evidence either way" → `f=1`
→ `dyadic_risk_score = global_risk`, unchanged). A maximally-protected customer gets `f=0.7`
(risk reduced 30%); a maximally-exposed one gets `f=1.3` (risk increased 30%). `LAMBDA=0.6` is
a moderate, auditable bound — visible and meaningful to a reviewer without letting business
arithmetic alone swing the score wildly. 19/19 unit tests in `ml/tests/test_dyadic.py` lock in
this formula's exact behavior (neutral point, bounds, clipping, renormalization on missing
inputs, and the `reordering_rate` metric itself on synthetic concordant/discordant examples).

---

## Deviation from `docs/05_Database_Design.md` §6.26, flagged explicitly

The documented `supplier_dyadic_risk` schema has **no `customer_id` column at all** — confirmed
by re-reading §6.26 directly, twice. Yet Claim B's own formula (§8.5) and this round's own task
instructions define all three weights as per-**(customer, supplier)** pair quantities, computed
from `orders.customer_id`. A table with no way to record which customer a row's weights belong
to cannot store what the formula actually produces. `db/supplier_dyadic_risk_schema.sql` adds a
`customer_id UUID NOT NULL REFERENCES customers(id)` column as a necessary correction — every
other column matches §6.26 exactly (`id`, `supplier_id`, `risk_score_id`,
`order_volume_share`, `contract_priority_weight`, `fulfilment_preference_weight`,
`dyadic_risk_score`, `double_counting_test_run_id`, `scored_at`, all identical types/nullability
to the spec). Not built: §6.23's `suppliers` tier/`is_frontier` amendment and
`supplier_relationships` — those remain data-gated (no `SUB_SUPPLIES`-equivalent data exists,
unchanged since every prior round that checked) and are unrelated to what
`supplier_dyadic_risk` itself needs.

---

## The double-counting test — the actual "Done when" bar for this step

Trained SHARE + Variant A twice, 5 seeds each, fresh in this same process (no checkpoint is
ever persisted in this codebase): once with `Customer.priority_tier` visible to the encoder
(the current default), once with `Customer` node features zeroed out (confirmed:
`customer_features_asof`'s entire output IS the `priority_tier` one-hot, so zeroing the whole
feature tensor is exactly "mask `priority_tier`", nothing else touched). For each trained
model, computed `reordering_rate` — the Kendall-tau discordant-pair fraction between a
customer's suppliers ranked by raw `global_risk` (that model's own impact-head prediction) vs.
by `dyadic_risk_score`, averaged across all 1,295 customers with ≥2 suppliers.

### AUC — mean across 5 seeds, per (arm, task)

| Arm | delay | shortage | impact |
|---|---|---|---|
| with priority_tier | 0.8174 ± 0.0014 | 0.7975 ± 0.0028 | 0.9426 ± 0.0023 |
| without priority_tier (masked) | 0.8176 ± 0.0039 | 0.7980 ± 0.0024 | 0.9451 ± 0.0023 |

(Not the primary comparison this round — reported for completeness/governance, per this
project's standard convention of logging AUC on every training run regardless of the round's
main question.)

### Reordering rate — the actual result

| Arm | Per-seed values | Mean | Std |
|---|---|---|---|
| **WITH** priority_tier | `[0.0149, 0.0174, 0.0199, 0.0097, 0.0091]` | **0.0142** | 0.0042 |
| **WITHOUT** priority_tier | `[0.0102, 0.0199, 0.0123, 0.0166, 0.0127]` | **0.0143** | 0.0035 |

**Ratio (with / without) = 0.9898.**

**VERDICT: RATES CLOSE — the signals are largely independent; low double-counting risk.**
Masking `priority_tier` from the encoder entirely changes the reordering rate by under 1%
(0.0142 vs. 0.0143, well within both arms' own seed-to-seed noise, std ≈ 0.003–0.004). Per
§8.5's own decision rule ("if the rates are close, the signals are independent; if the rate
collapses without it, the encoder had already learned it"), this is unambiguously the
close-rates case, not the collapse case — a clean result, not an ambiguous one requiring a
qualified verdict.

---

## Final recommendation

**Report Claim B as a genuinely independent, non-redundant contribution — not one that should
be scoped down to "lead on `fulfilment_preference_weight` only."** The double-counting test's
own decision rule was designed precisely to distinguish these two outcomes, and the data lands
cleanly on the independent side: whether or not the encoder can see `priority_tier`, applying
Claim B's dyadic reweighting reorders a customer's supplier risk ranking by almost exactly the
same amount (~1.4% of pairs). This means `contract_priority_weight` is not simply re-deriving
something the encoder already learned and folded into its own risk predictions — it is adding
information the encoder's own `impact_score` doesn't already carry, at least as measured by
this metric. `fulfilment_preference_weight` remains the theoretically cleanest signal (the one
the encoder structurally cannot see under any circumstance, masked or not), and still carries
the heaviest weight in `f()` (0.50) by design — but the empirical result does not require
retreating to "only trust that one input," since the whole formula passed the independence
test as constructed.

**A secondary, honest caveat worth carrying forward**: `fulfilment_preference_weight` itself
only has real evidence for 6.7% of (customer, supplier) pairs (1,923 of 28,529) — most pairs'
`dyadic_risk_score` is currently driven by `order_volume_share` and `contract_priority_weight`
alone (renormalized 50/50), with the lead signal silently absent rather than fabricated. This
doesn't invalidate the double-counting result (which holds at the aggregate level across
whatever mix of available signals each pair actually has), but it does mean the specific claim
"leading on the encoder-blind signal" is only literally exercised for a minority of scored
pairs today — worth revisiting if `min_concurrent_delayed` (currently 2) or the trailing window
(currently 90 days) turn out to be more restrictive than necessary once more disruption history
accumulates.

---

## Governance record

Backed up before any write
(`reports/backups/model_registry_and_evals_backup_20260809_202207.sql`). New table
`supplier_dyadic_risk` created (`db/supplier_dyadic_risk_schema.sql`) — 0 rows before this
round, since it didn't exist. 10 new registry rows — 5 `rgcn_attn_rung5_a-claimb-with-seed{n}`
+ 5 `rgcn_attn_rung5_a-claimb-without-seed{n}`, all `status='active'` — **343 total registry
rows** (333 + 10). **150 new evaluation rows** (10 runs × 15 metric rows) — **5,013 total**
(4,863 + 150). **28,529 new `supplier_dyadic_risk` rows** (every (customer, supplier) pair with
order-volume evidence in the scoring window; `double_counting_test_run_id =
step8-claimb-20260809T145952Z` links every row to this validation run). `risk_scores` —
**untouched, still exactly 800 rows**, confirmed after the run — Claim B reweights it, never
modifies it. No existing row from any prior round modified. Full run log:
`reports/logs/run_step8_claim_b_20260809_202951.log`.

---

*Scoped to Claim B only, per this round's own framing — no changes to SHARE's encoder, SHARP,
SHARK, Transformer 2, or the existing Markov/Variant A depth-selection logic.*
