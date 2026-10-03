# Phase 18 — information fixes for arrival, fill and capacity

**Audience:** whoever decides what the arrival, fill and capacity models need next: more information, or nothing more.
**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, val 2024, test 2025). Every figure is TEST unless marked.
Thresholds and blend weights are fitted on validation only.
**Companions:** `phase18-preregistration.md` (`375dae6`); `phase18/stage1_generator.md`; `phase18/oracle/PRIVILEGED__stage2_oracle.md`;
`phase18/stage3_squeeze.md`; `phase18/stage4_forward_load.md`; `phase18/stage5_pulse.md`; `phase18/stage6_lane.md`.
**Branch / base:** `phase18`, from `origin/phase17` at `a2301ef`. Not merged.
**Status:** complete. Stage 7 (neural confirmation) **not run**: no feature family passed the proxy gate. All stages
finished in about 35 minutes of the 12-hour window (stop 06:39 IST 2026-10-04), so no `STOPPED.json` was needed.

## Header

| | |
|---|---|
| machine | Apple M4 Pro, 14 cores, 24 GB, macOS (Darwin 27.0) |
| software | Python 3.14, torch 2.14.0 (MPS available, **not used**: no neural run), LightGBM 4.7.0, NumPy / pandas / scikit-learn from `venv/` |
| GPU | none used. Every Phase 18 computation ran on the CPU. Scorers ran with `HADES_DEVICE=cpu` |
| concurrency level | CPU only. At most two LightGBM processes at once (proxy and oracle, `n_jobs = 6` each), plus a scorer. Wall-clock only: the proxy's BASE arm reproduced the stored predictions **bit-exactly** under that load |
| stored baselines | the Mac's own `ml/artifacts` (the machine that trained every stored v8 bundle); no handshake was needed because nothing neural was trained |

| artifact | code commit |
|---|---|
| pre-registration | `375dae6` |
| generator re-run (56/56 files equal), `fwd_load` / `pulse` features (rebuilt from committed code: identical arrays) | `94cfdc5` (verifier `9ce7007`) |
| proxy fits: fwd_load, fwd_load_shuf, pulse, pulse_shuf (BASE reproduction) | `94cfdc5` / `9ce7007` (`ml/` trees identical) |
| proxy diagnostic arm fwd_load_netmean; proxy scorer `scores.json` | `67d9111` |
| oracle / oracle_state / hindsight_load fits | `9ce7007`; oracle_state_notiming `e967003` |
| oracle scores `PRIVILEGED__oracle_scores.json` | `e967003` |
| Stage 3 squeezes `stage3_squeeze.json` | `61e92ef` |
| isolation audit `isolation_audit.json` (+ `test_artifact_identity.py`, 6 / 6 pass) | run at `1212dd3` (all report content committed) |

Every run refused to start on a dirty `ml/` (`phase12_common.require_clean`) and recorded `code_dirty = false`.

**Anchors reproduced before any new number was trusted:** arrival C-index 0.6744 [0.6734, 0.6755] and lateness ROC-AUC
0.7090 [0.7064, 0.7130]; A3 13.06 vs LightGBM 13.25 days; fill exact CRPS 0.13876, P(full) AUC 0.6204; capacity
precision 0.904 / 0.851 / 0.810 and recall 0.515 / 0.318 / 0.191. All equal Phase 17's published figures. The
reference offset c = +4.571 weeks equals the spec.

---

## 1. Decisions

| use case × fix | verdict | the number | band / interval |
|---|---|---|---|
| **arrival × oracle headroom** | **REAL** (+0.230) | oracle lateness AUC 0.9385 vs model 0.7090 | oracle [0.9384, 0.9387]; model [0.7064, 0.7130] |
| arrival × … of which state, no timing | **≈ 0** | 0.7092 vs 0.7090 | [0.7087, 0.7096] |
| **fill × oracle headroom** | **REAL** (+0.298) | P(full) AUC 0.9184 vs 0.6204 | [0.9182, 0.9186] vs [0.6200, 0.6211] |
| **capacity × oracle headroom** | **REAL** (+0.149) | precision @ 5% 1.000 vs 0.851 | state-only tier 0.944 [0.939, 0.949] |
| **arrival × forward load** | **FAIL** (gate) | lateness AUC 0.7186 vs BASE 0.7053; **shuffled 0.7076 also above BASE** | [0.7183, 0.7191] / [0.7048, 0.7057] / [0.7068, 0.7081] |
| **fill × forward load** | **FAIL** (gate) | CRPS 0.1355 vs 0.1397; AUC 0.658 vs 0.605; **shuffled also gains** | CRPS [0.1354, 0.1357]; shuffled [0.1382, 0.1385] |
| **capacity × forward load** | **FAIL** (gate) | 6/6 points disjointly better (precision @ 5% 0.813 vs 0.692); **shuffled better on 5/6** | [0.802, 0.828] vs [0.687, 0.699]; shuffled [0.708, 0.751] |
| arrival × global pulse | **FAIL** | lateness AUC 0.7034, **worse** | [0.7031, 0.7039] vs [0.7048, 0.7057] |
| fill × global pulse | **FAIL** | CRPS 0.1400, **worse** (AUC better) | [0.1399, 0.1402] vs [0.1396, 0.1398] |
| capacity × global pulse | **FAIL** | precision @ 1 / 5% **worse** (0.718 / 0.658 vs 0.780 / 0.692) | |
| arrival / fill / capacity × lane / checkpoint | **BLOCKED** | one lane per channel; `via_checkpoint` 100% empty | |
| arrival × seed-ensemble | **GAIN** (lateness, C-index) | 0.7138 / 0.6773 vs single-seed mean 0.7090 / 0.6744 | +0.0047 [0.0045, 0.0050]; +0.0028 [0.0006, 0.0050] |
| **arrival × neural + LightGBM blend (w 0.59)** | **GAIN over both parents** | lateness AUC **0.7247**; A3 **12.50 d** | vs neural ens +0.011 [0.008, 0.014]; −0.59 d [0.46, 0.72] |
| fill × seed-ensemble | **GAIN** | CRPS 0.13853, AUC 0.6240, ECE 0.091 vs 0.13876 / 0.6204 / 0.124 | all intervals exclude 0 |
| fill × blend (w 0.97) | marginally better than the neural ensemble (all differences < 0.002); **worse ECE than LightGBM** | ECE 0.089 vs 0.052 | −0.036 [−0.042, −0.030] |
| capacity × seed-ensemble | **GAIN** (precision @ 5%, recall at all three bars) | 0.864; 0.547 / 0.331 / 0.230 | +0.014; +0.032 / +0.013 / +0.039 |
| capacity × blend (w 1.00) | = the neural model | | |
| arrival interval × split-conformal (P50, pre-registered) | **misses band** | coverage 0.862, 49 d (raw 0.930, 52 d) | [0.854, 0.870] |
| arrival interval × conformal on expected week (diagnostic) | **in band** | coverage 0.801, 45 d | [0.791, 0.806] |
| capacity interval × trailing level-aware conformal | **no better than static** | per-snapshot coverage 0.74–0.84 (median 0.774) vs static 0.75–0.85 | pooled 0.793 [0.787, 0.796] |

## 2. Oracle ceiling (PRIVILEGED; never a feature or a claim)

| use case · metric | model (neural) | LightGBM-flat | **ORACLE** (brief) | ORACLE-STATE | state, no timing | **headroom** | verdict |
|---|---|---|---|---|---|---|---|
| arrival · lateness ROC-AUC | 0.7090 | 0.7053 | **0.9385** | 0.9229 | 0.7092 | **+0.230** | REAL |
| fill · P(fill = 1) ROC-AUC | 0.6204 | 0.6049 | **0.9184** | 0.7506 | 0.7502 | **+0.298** | REAL |
| capacity · precision @ 5% | 0.851 | 0.692 | **1.000** | 0.944 | — | **+0.149** | REAL |

**What the headroom is made of.**
- **Arrival: when the line gets raised.** All of it. `arrival_week` counts from t0, so it includes the 1–12-week wait
  before the line exists. Supplier, transit, regime and K state at creation add nothing beyond the model (0.7092 vs 0.7090).
- **Fill: the supplier's queue.** State alone (true K, supplier state) is worth +0.13. Knowing how much else the
  supplier was asked for in the creation week and month is worth another +0.17.
- **Capacity: the capacity draw.** True K alone gives 0.944. K is drawn almost independently every month (within-supplier
  autocorrelation 0.09) and the declared figure tracks it at 0.11, so that +0.09 is **unreachable from as-of data**.
  The remaining gap is forecastable ordered volume: hindsight load reaches 0.891, the as-of plan 0.813.

The stop rule (AT CEILING → skip Stage 7) fires for **no** use case.

## 3. Prediction scorecard

| # | prediction | **right / wrong** | the number |
|---|---|---|---|
| **P1** | arrival oracle beats the model on lateness AUC by ≥ 0.03 | **RIGHT** | +0.230 (0.9385 vs 0.7090). All of it is order timing (no-timing tier +0.000) |
| **P2** | fill oracle within 0.03 of the model's P(full) AUC | **WRONG** | +0.298; even the state-only tier is +0.130 |
| **P3** | capacity oracle beats the model at 5% coverage by ≥ 0.05 | **RIGHT** | +0.149 (state-only +0.093) |
| **P4** | forward load gives a disjoint LightGBM-proxy gain on capacity | **RIGHT** | 6 / 6 points disjoint (precision @ 5% 0.813 [0.802, 0.828] vs 0.692 [0.687, 0.699]); the **gate still fails** on the shuffle control |
| **P5** | forward load does NOT give a disjoint gain on fill | **WRONG** | CRPS 0.1355 vs 0.1397, AUC 0.658 vs 0.605, both disjoint (it also beats the neural incumbent) |
| **P6** | seed-ensembling improves fill CRPS / ECE and arrival C-index, and the blend ties or beats both parents | **WRONG** (in part) | neural ensembling improves all three (right). LightGBM ensembling ties on ECE and C-index. The blend is **worse than LightGBM on fill ECE** (−0.036) and, by construction, −0.001 on capacity recall @ 0.70 |
| **P7** | the global pulse ties | **WRONG** | not a tie: disjointly **worse** on arrival lateness AUC, fill CRPS and capacity precision @ 1 / 5% (better on fill AUC and capacity recall @ 0.70 / 0.85) |
| **P8** | split-conformal brings arrival P10–P90 coverage into 0.78–0.82 | **WRONG** | 0.862 [0.854, 0.870]: the P50 and the label are whole weeks, so coverage moves in one-week steps. Centred on the expected week (diagnostic): 0.801 |

**3 right, 5 wrong.** The working hypothesis behind the phase (the bottleneck is information, not architecture) is
**supported**. Real headroom exists everywhere, and the one information fix tried (the forward plan) moves every use
case by more than any architecture change has. But the pre-registered control could not credit it (§1, §5 deviation 163).

## 4. Stage 7 — neural confirmation

**Not run.** The rule: only for families that PASSED the proxy gate and whose use case is not AT CEILING. No family
passed, so the Stage −1 handshake was not needed. `ml/models/share.py`, `ml/models/tcn.py` and every existing class
are untouched, and no new neural class was written.

## 5. Deviations, continuing from 160

| # | prior statement | measured / done | where |
|---|---|---|---|
| **160** | Stage 4: requirement "from the plan version …, BOM-exploded to the supplier's parts" | used `part_demand_weekly`, the versioned part × plant × week requirement, which is already at part grain. In v8 `production_plan` (product-plant) feeds only its own tables, **not** channel demand, so BOM-exploding it would carry only seasonality | stage 1 (d), stage 4 (b) |
| **161** | Stage 6: "as-of mean lateness over OTHER channels on the same lane / checkpoint" | **BLOCKED**: one lane per channel, and `via_checkpoint` 100% empty (the generator writes `checkpoint_id`, the schema names the column `via_checkpoint`, and `emit` keeps only schema columns). The transit latent is per destination plant, which has a node | stage 6 |
| **162** | constraint 6: "assert recorded_ts ≤ t0 on every source row" | `part_demand_weekly` has **no recorded_ts column**. The asserted recorded time is `as_of_date` + 2 days, the generator's maximum draw (`generator_v8.py` l. 1418) | stage 4 (b) |
| **163** | Stage 4 control: "shuffle forward-load across suppliers within each snapshot — the gain must vanish" | it does not vanish, because the shuffle preserves each snapshot's **network-wide forward requirement** (the forward season). A **diagnostic** arm (snapshot means only, no verdict) reproduces the shuffled gain, and full fwd_load beats both disjointly. The pre-registered verdict **FAIL stands** | stage 4 |
| **164** | Stage 2: "verify the output SHA equals the stored world" | **51** files SHA-1 equal. `dataset_coverage.csv`, `manifest.json` and `parameters_v8.json` carry a wall-clock write time, and `_sim.npz` / `_events.npz` zip headers do too: compared by content with the time removed (all equal). `level4.json` is written by the validator, not the generator: not compared. **56 / 56 equal** | oracle §1 |
| **165** | oracle tiers fixed in the pre-registration | the **no-timing** tier was added after the ORACLE tier was read, as a labelled diagnostic. It does not change any verdict | oracle |
| **166** | P8: split-conformal on the P50 | P50 and label are whole weeks, so residuals are multiples of 7 days and coverage moves in one-week steps (0.862 at −35 / +14 d). An expected-week centre (diagnostic) gives 0.801 at 45 d | stage 3c |
| **167** | ensembles / blends compared by a paired row bootstrap | rows in one snapshot share its shocks, so the intervals are **optimistic**. A 1,000-resample snapshot-block bootstrap would be wider | stage 3 |
| **168** | brief: "Wall-clock stop: <FILL IN>"; report paths | not filled in. The brief's own example, 12 h, was used (stop 06:39 IST 2026-10-04). Reports are at the brief's paths `reports/phase18*`, not `reports/part2/` | header |
| **169** | brief: GPU, "concurrency 1", handshake | no neural run, so no GPU or handshake. CPU LightGBM processes ran up to two at a time, which affects wall-clock only: BASE reproduced bit-exactly | header |
| **170** | "commit SHA per artifact" | proxy fits are stamped with two commits, `94cfdc5` and `9ce7007`, whose `ml/` trees are identical (the second changed only `reports/`). The feature files were rebuilt from the committed code and are array-identical | header |

## 6. Isolation audit (Stage 8b)

`ml/tests/test_phase18_isolation.py` → `ml/artifacts/phase18/isolation_audit.json`, run at `1212dd3`, every check PASS:

| check | result |
|---|---|
| `git diff a2301ef HEAD` on db/gen_v6, gen_v7, gen_v8, db/validator.py, docs/specs/**, db/dataset_structure.md, ml/configs/shipped.json, results/**, reports/part1, reports/part2, ml/models/share.py, ml/models/tcn.py, ml/artifact_identity.py | **empty**, committed and working tree. `git diff a2301ef HEAD -- ml/models/share.py ml/models/tcn.py` is empty |
| stored configs recomputed under `artifact_identity` at the base and now | **371 identical** (identity, bundle path, config name). **340** sit in the directory `bundle_name` gives them |
| configs that cannot be named by this base | **31**, all Phase 16 encoder variants (`_encshare_pna…`, `_encshare_traj…`, `_encheteromp_pna…`), whose axes live on the unmerged `phase16-encoders` branch (deviation 149). Not caused by Phase 18 |
| no privileged input reachable from `ml/train`, `ml/models`, `ml/data`, `ml/baselines` | AST scan: **0 violations**. A fresh interpreter importing `fwd_load`, `pulse`, `phase18_proxy` loads nothing from `reports/` and no torch. The scanner **fires** on three constructed offenders and passes a docstring mention. One pre-existing reader of `_sim.npz`, `ml/baselines/learnability_windowed.py` (not Phase 18), is listed |
| Phase 18 feature names | none carries `PRIVILEGED__` |
| new config axes | **none**: no neural config was trained, so the identity is untouched |

## 7. Every `inventory_position_weekly` read

| reader | purpose |
|---|---|
| `reports/phase18/oracle/PRIVILEGED__regen_latents.py` (verify) | SHA-1 of the stored `inventory_position_weekly.csv` bytes vs the re-run's copy: an **identity check**, no value used |
| same, `_sim.npz` comparison | array-equality of the simulator's part-plant position arrays (`PP_OH`, `PP_OO`, …) between stored and re-run: identity check only |

**Never a model feature.** `fwd_load` (part_demand_weekly, po_lines, grn_lines), `pulse` (po_lines, grn_lines,
supplier_capacity), the proxy (flat features + those) and the oracle (`_sim.npz` arrays listed in its docstring, which
exclude the position arrays) do not read it.

## 8. What this changes for the product

**Which use cases are AT CEILING:** **none.** Every use case has real headroom. But only part of it can be reached
from data that exists at t0:

- **Arrival.** The ceiling is **order timing**: when the next PO line on a channel will be raised. Supplier state
  adds nothing on top of the current model. More supplier, transit or graph state will not move arrival. A forecast of
  the order-raising time will. That is a new input or sub-model, not an encoder change. **Ship now:** the
  validation-fit **neural + LightGBM blend** (w 0.59). It is the largest arrival gain in this phase that needs no
  training: lateness AUC 0.709 → 0.725, A3 median error 13.06 → 12.50 days. Also the **expected-week conformal
  interval**: 80% coverage at 45 days, against 93% at 52 days raw.
- **Fill.** Headroom is the supplier's **queue**: capacity and competing orders that week. The forward plan reaches
  part of it, and the LightGBM proxy with `fwd_load` beats the neural incumbent (CRPS 0.1355 vs 0.1388, P(full) AUC
  0.658 vs 0.620). **Ship now:** the neural **seed-ensemble** (improves CRPS, AUC and ECE). Fill's watchlist status is
  unchanged until a forward-load arm passes a corrected gate.
- **Capacity.** About 60% of the gap to the oracle (+0.093 of +0.149) is the monthly capacity draw K. **No as-of source can reach
  it** on v8: the declared figure tracks K at 0.11. The remaining 40% is forward ordered volume, where the plan helps
  the LightGBM proxy by +0.12 precision at 5%, though it still trails the neural incumbent. **Ship now:** the neural
  **seed-ensemble** (precision @ 5% 0.864; recall +0.03 / +0.01 / +0.04 at the three bars). Trailing conformal is not
  worth the complexity: use the static offset.

**Which fixes passed:** none under the pre-registered gate. **Forward load is the one to re-test.** It beats its own
shuffle and the network-mean arm disjointly on all three use cases, so its supplier-specific content is real. The gate
failed because the control kept the network's forward season. The next phase should **pre-register the network-mean
arm as the control** and, if it passes, bind `fwd_load` into the neural heads through the existing per-row input
(`n_row_feats`), as a new subclass. The global pulse is closed: worse on this test year.

**Which are blocked on data:** **lane / checkpoint sharing.** v8 has one lane per channel and an empty checkpoint
column, and its transit latent is per plant. A lane node needs a generator in which a shared transit latent is indexed
by lane or checkpoint. **Capacity's K** is blocked the same way: an observable that tracks the monthly capacity draw
does not exist in v8.
