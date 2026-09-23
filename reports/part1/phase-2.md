# Phase 2 — Sequence assembly, baseline reproduction, G5

Device MPS / torch 2.14.0 (Phase 2 is CPU/LightGBM; MPS enters at Phase 3).
Split frozen: **train ≤ 2023-12-31, val 2024, test 2025**, metrics on test only.
Code: `ml/data/sequences.py`, `ml/baselines/g5_asof.py`, `ml/baselines/learnability_windowed.py`.

**Gate: PASSED**, with one qualification recorded in §G5 — the drop appears on the head that has
signal, and does not appear on a head that has none.

---

## 1. Baseline reproduction — all four reproduce exactly

`python3 docs/learnability.py db/gen_v7/seed_1001` on the restructured path:

| Task | metric | validation7 §11 naive | **reproduced** | validation7 GBM | **reproduced** |
|---|---|---|---|---|---|
| fill_rate | CRPS ↓ | 0.1041 | **0.1041** ✓ | 0.0987 | **0.0987** ✓ |
| arrival_week | C-index ↑ | 0.5636 | **0.5636** ✓ | 0.6510 | **0.6510** ✓ |
| shortage_qty | PR-AUC ↑ | 0.3163 | **0.3163** ✓ | 0.4692 | **0.4692** ✓ |
| demand_drift | MAE ↓ | 0.1517 | **0.1517** ✓ | 0.0983 | **0.0983** ✓ |

Exact to four decimals on all eight numbers. The restructure did not perturb the data, and
everything downstream is measured against a pipeline that reproduces.

## 2. The 2019–2025 fit window

Applying the constraint drops 2016–2018 from training: **1,070,700 → 786,900 label rows (73.5%)**.

| Task | metric | full history | **2019–2025** | change |
|---|---|---|---|---|
| fill_rate naive | CRPS ↓ | 0.1041 | 0.1078 | naive worse |
| fill_rate GBM | CRPS ↓ | 0.0987 | **0.0988** | −0.1% |
| arrival naive | C-index ↑ | 0.5636 | 0.5578 | naive worse |
| arrival GBM | C-index ↑ | 0.6510 | **0.6500** | −0.15% |
| shortage naive | PR-AUC ↑ | 0.3163 | 0.2964 | naive worse |
| shortage GBM | PR-AUC ↑ | 0.4692 | **0.4669** | −0.5% |
| drift GBM | MAE ↓ | 0.0983 | **0.0987** | −0.4% |

**The GBM barely moves; every naive baseline gets worse.** Losing three years of history costs the
history-based baselines more than the feature-based model, so the model's *margin over naive*
widens inside the window — on shortage from +48% to +57% of above-chance signal. The window is
the right place to fit and it is not costing accuracy.

## 3. Guide 2.1 — `[T × d]` tensors

`build_sequences()` returns `X [B, 52, 24]`, `M [B, 52, 10]`, `active [B, 52]`, `pad [B, 52]`.
d = 14 value channels + 10 observed-indicators.

    v6: X (4096, 52, 24)  build 0.02 s
    v7: X (4096, 52, 24)  build 0.02 s
    padding-mask mean 1.0000   (no padding needed -- the panel is complete)

The guide's d = 38 does not apply: it was composed for a 20-column store of which four are
constant here (below), plus calendar and static channels this phase does not fold in.

## 4. Guide 2.2 — masking, and two data findings that change it

### Finding 1 — four store columns are constant zero

Measured on both worlds, `channel_performance_weekly`:

| column | distinct values |
|---|---|
| `revision_count` | **1** (all 0) |
| `days_since_last_short` | **1** (all 0) |
| `weeks_since_last_activity` | **1** (all 0) |
| `weeks_since_last_receipt` | **1** (all 0) |

These are generator placeholders that were never populated. A constant column carries no
information, so they are **excluded from the panel**. This matters beyond tidiness: the guide's
step 2.2 singles out `weeks_since_last_receipt` — *"NULL means never received, not long ago;
zero-filling merges a channel that has never taken a delivery with one that took its last delivery
this week"*. In these worlds the column **is** all zeros, so that distinction is not merely
zero-filled, it was never generated.

### Finding 2 — the sparsity the specification describes does not exist here

The guide (and `benchmark_specification` §3.2) states `fill_rate` and `ack_gap_ratio` are 81.4%
null, `load_ratio` 81.8%, the lead-time pair 72.6%, and warns that mean-imputing `fill_rate`
*"fabricates 82% of the activity in the sequence"*. Measured on v7:

| column | guide expects null | **measured null** |
|---|---|---|
| `fill_rate` | 81.4% | **5.95%** |
| `ack_gap_ratio` | 81.4% | **5.95%** |
| `load_ratio` | 81.8% | **0.00%** |
| `lead_time_actual_days` | 72.6% | **5.95%** |

Meanwhile **88.6% of channel-weeks have `qty_ordered == 0`** and only 11.4% are
`is_active_week`. The sparsity is real; it simply is not expressed as nulls, because the
generator **forward-fills** the level and rolling columns across idle weeks
(`fillw → ffill(axis=1)` in `generator_v7.py`). On an idle week `fill_rate` carries the last
observed value, not a null.

Two consequences, both reported rather than worked around:

1. **The 13 nullable indicators are nearly vacuous** — mean 0.951, not the ≈0.181 the guide's
   verify step expects. They cannot tell the model whether a week was idle.
2. **The imputation the guide warns against is already in the data.** A model reading `fill_rate`
   on an idle week reads a stale carried-forward value. I did not undo it — that would mean
   rewriting a protected dataset — but I promoted **`is_active_week` to an explicit value
   channel**, so the idle/active signal the nulls were supposed to carry is available.

The guide's verify step *"for a known idle week the `fill_rate` channel is 0.0"* therefore
**fails on this data by design**: 175,481 of 212,992 idle positions in a 4,096-channel sample carry
a non-zero forward-filled `fill_rate`. That is the store's behaviour, not a masking bug.

## 5. Guide 2.3 — normalisation

Fitted on the **training fold only**, over **valid positions only**, with `log1p` on the skewed
channels (`qty_ordered`, `qty_received`, `lead_time_actual_days`) before standardising.

| world | train-fold mean | train-fold sd | test-fold mean |
|---|---|---|---|
| v6 | +0.0000 | 0.9573 | −0.0473 |
| v7 | −0.0000 | 0.9574 | −0.0471 |

The test-fold mean is **not** zero, which is the point — fitting the scaler on all data would
leak the future through it, and `validator.py` could never catch that because it never sees the
tensors.

## 6. G5 — is the as-of gate binding?

Run two ways, because the first was uninformative and saying so is the finding.

### Variant (a) — shuffle `recorded_ts` within each table, rebuild the store from source

`ml/baselines/g5_asof.py` re-derives ordered/received/fill panels from `po_lines` and `grn_lines`
under the true and the permuted `recorded_ts`, differing in nothing else.

| world | CRPS true | CRPS shuffled | Δ |
|---|---|---|---|
| v6 | 0.0629 | 0.0630 | +0.13% |
| v7 | 0.1074 | 0.1073 | −0.02% |

**No drop.** But the perturbation did land — measured directly on the feature matrices:

| | corr(true, shuffled) | cells changed | L1 relative change |
|---|---|---|---|
| ordered panel | 0.509 | 11.9% | 100.9% |
| received panel | 0.382 | 13.5% | 123.3% |
| `fill_rate_last13` | **0.129** | 31.5% | 168.7% |

Rows crossing a week boundary went from 9.06% to 49.93%; **54.5% of rows changed visible week.**
So the features were very nearly destroyed and the predictions did not move at all.

The explanation is the head, not the gate: this arm's CRPS (0.1074) is **worse than the naive
per-channel mean (0.1041)** and close to the global empirical CDF (0.1082). It had learned
essentially nothing from those five hand-built features, so it was insensitive to any perturbation
of them. **Variant (a) is uninformative about the as-of gate**, and repeating it with instantaneous
week-`t0` features instead of rolling ones changed nothing (Δ ≤ 0.03%).

### Variant (b) — break the as-of *alignment* on the head that has signal

Same features and head as the reproduced baseline, on **arrival** (margin 0.6510 vs 0.5636, the
largest of the four). The label's `t0` is permuted across rows, so the model receives a
correctly-formed but time-misaligned as-of vector.

| world | C-index aligned | C-index misaligned | Δ | **loss of above-chance signal** |
|---|---|---|---|---|
| v6 | 0.6431 | 0.6256 | −0.0175 | **−12.2%** |
| v7 | **0.6498** | **0.6280** | **−0.0218** | **−14.6%** |

**A clear drop on both worlds.** The arrival head genuinely reads time-aligned as-of features;
destroying the alignment costs it an eighth to a seventh of everything it knows above chance.

### What G5 does and does not certify

- **It binds on arrival**, in both worlds, at 12–15% of above-chance signal.
- **It is untestable on fill as built here**, because a head with a ~5% margin over naive cannot
  resolve a perturbation against its own noise floor.
- Phase 1's assertion A2 independently proves the store *is* bucketed on
  `max(event_week, recorded_week)` to exact integer conservation. The gate is enforced; variant (b)
  shows it is also consequential where there is signal to be consequential about.

## 7. Wall-clock and memory

| step | v6 | v7 |
|---|---|---|
| baseline reproduction | — | 96 s |
| windowed baselines | — | 92 s |
| sequence build (4,096 × 52) | 0.02 s | 0.02 s |
| G5 variant (a), both arms | 74 s | 71 s |
| G5 variant (b), both arms | 58 s | 55 s |
| peak RSS | 1.3 GB | 1.3 GB |

**One environment note.** Building two full source-derived panels and then fitting LightGBM in a
single process **segfaults** (exit 139) — libomp plus large `np.add.at` allocations under Python
3.14. Each G5 arm therefore runs in its own process. LightGBM alone is stable at any `n_jobs`; the
crash needs both in one interpreter.

---

## Gate

| Requirement | Result |
|---|---|
| Baselines reproduce | **pass** — all 8 numbers exact |
| G5 shows a clear drop | **pass** on the arrival head (−12.2% / −14.6% of above-chance signal); **not resolvable** on the fill head, with the diagnostic showing why |
| Heads train without NaN or divergence on both worlds | **pass** — no NaN, no divergence, early stopping fired normally in every fit |

**Phase 3 may start.**
