# Phase 17 — pre-registration

**Committed before any Phase 17 measurement.** Nothing below is edited after a result is known. Each prediction is
marked right or wrong in `reports/part2/phase-17.md`; wrong ones are not dropped.

**Measured on:** v8 seed 1001, fixed split (train ≤ 2023, val 2024, test 2025).
**Base:** `origin/HADES-v4-ml-pipeline` at `90a38ed` (Phase 15), plus three Windows/CUDA commits carried onto the
feature branch `phase17` (`167eafb` device, `11c6019` low-VRAM mode, `b2c7c31` queue port).

## Environment (Stage −1, run before this file)

| | |
|---|---|
| machine | Windows 11, NVIDIA GeForce RTX 3050 Ti Laptop GPU, **4 GB** VRAM (the brief assumes 32 GB), driver 581.95 |
| torch / CUDA / cuDNN | 2.14.0+cu126 / 12.6 / 9.10.02, Python 3.10.11 |
| precision | float32; TF32 **off** (`HADES_TF32` unset); low-VRAM mode `HADES_PANEL_HOST=1`, `HADES_TCN_CHUNK=4096` |
| concurrency level | **1 for every cell** (two concurrent cells measured ~10× slower each on 4 GB: VRAM spills to shared memory) |
| stored artifacts | the Mac's `ml/artifacts` (copied); `cache/v8` byte-identical to this machine's rebuild (panel/miss/active SHA-1 equal) |

**Handshake: INSIDE the stored band.** Stored config `arrival_week v8_lite_h4_lr0.00025_s7` (stamp `b1794c0`), retrained
3 epochs here:

| epoch | val C-index here | stored s7 | \|diff\| | stored 5-seed band at this epoch |
|---|---|---|---|---|
| 0 | 0.4875004 | 0.4874936 | 6.8e-6 | [0.48010, 0.51147] |
| 1 | 0.5005191 | 0.5005219 | 2.7e-6 | [0.49442, 0.53403] |
| 2 | 0.5555158 | 0.5555247 | 8.8e-6 | [0.53771, 0.56398] |

Training loss agrees to ≤ 1.6e-5 per epoch. **Consequence:** stored baselines are reused, and the halt-and-diagnose
rule for a retrained baseline landing outside its stored band applies as a genuine signal.

## Predictions

| # | Prediction |
|---|---|
| **P1** | Re-weighted to a 3% base rate, the shortage head's precision at 5% coverage falls below 0.40 (from 0.910 at 4% coverage on v8's 25.7%) |
| **P2** | The ~8,200 rescued-but-unflagged weeks are **concentrated**, and missed part-plants have **more channels** than flagged ones |
| **P3** | Signed-days lateness from the hazard distribution beats the channel-median constant on median absolute error, uncensored rows |
| **P4** | Predict-the-rescue exceeds 0.63 precision at matched recall on validation (Phase 14's crude-proxy ceiling, 1.3–1.4× lift on a 45.4% base) |
| **P5** | The Δstrain capacity head does **not** beat the incumbent disjointly (every architecture change in this project so far has tied) |
| **P6** | The lean encoder (part relation dropped) **ties** the full encoder, confirming C3's ablation |

## Decision rules (fixed here, before any number)

- **A1** precision @ 5% coverage at a 3% base: ≥ 0.60 survives (schedule time-to-shortage work); 0.40–0.60 internal
  watchlist only, permanently; < 0.40 **retire**.
- **A2** missed part-plants have measurably more channels → channel granularity justified; miss uniform across every
  profile → granularity is not the explanation, do not start that build.
- **B1** > 0.70 precision at ≥ 20% recall, disjoint at 5 seeds → GO; 0.63–0.70 marginal; ≤ 0.63 NO-GO.
- **B2** pass requires a disjoint gain in recall at fixed precision over B2a.
- **B4** a tie is a win.
- Everywhere: a margin inside the 5-seed band is **undetermined**, not a difference. Thresholds are fitted on
  validation only.
