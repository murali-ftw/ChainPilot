...

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HADES v4 / ChainPilot × Rane: a research codebase with two halves.

1. **`db/`** — a causal *synthetic* supply-chain world generator (49-table ERP schema in `db/schema.sql`) plus a frozen validator. Each generator version is a different *regime*, not a size: `gen_v6`, `gen_v7` (seeds 1001/1002), `gen_v8` (seeds 1001–1005). The generated worlds (`db/gen_v*/seed_*/`, ~3 GB of CSV each) are gitignored and rebuilt from the generator source and the seed.
2. **`ml/`** — forecasting models trained on those worlds (TCN temporal encoder + SHARE heterogeneous graph encoder + task heads), baselines, evaluation, Monte Carlo simulation and optimisers. Tasks: `arrival_week`, `fill_rate`, `capacity_strain`, `shortage_qty`.

The work runs in numbered **phases**. Each phase is written up in `reports/part1/` (phases 0–11 pilot, v6/v7) or `reports/part2/` (v8 onward). Phases 13–15 are on `HADES-v4-ml-pipeline`; Phase 16 is in progress on `phase16-encoders`; Phase 17 is on `phase17` (branched from the Phase 15 tip). Most scripts in `ml/eval`, `ml/train`, `ml/opt` and `ml/sim` are named for the phase that produced them (`phase12_c3_checks.py`, etc.) and exist to reproduce that report's numbers.

## Environment and commands

- Written for **macOS on Apple Silicon**: a `venv/` at the repo root (scripts call `./venv/bin/python`), with MPS as the torch device. The shell scripts target bash 3.2 or zsh. On Windows, run Python directly and don't expect the `.sh` queues to work unchanged.
- **Windows/CUDA box (RTX 3050 Ti, 4 GB VRAM):** `venv\Scripts\python.exe` (Python 3.10; torch cu126, numpy, pandas, torch_geometric, scikit-learn, scipy). Env flags read by `ml/device.py` / `ml/train/phase5_heads.py`, all off by default so the Mac path is unchanged:
  - `HADES_DEVICE=cuda|cpu|mps` forces the device.
  - `HADES_PANEL_HOST=1` and `HADES_TCN_CHUNK=4096` are the low-VRAM mode. Without them the model spills ~2 GB into shared system memory and runs ~4.7× slower. The arithmetic is unchanged: loss is bit-identical and validation is identical.
  - `HADES_TF32=1` is off by default. It gave no speed-up here and changes results.
  - Also set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  - Run **one cell at a time**: two concurrent cells refill VRAM and each runs ~10× slower.
  - `ml/train/phase1_v8_queue.ps1` is the PowerShell port of the v8 queue, with these flags set. Phase 17 uses `ml/train/phase17_queue.ps1` / `phase17_chain.ps1` (one cell at a time, stage caps or a global `-GlobalDeadline`, `.done` resume markers).
  - On this box `ml/artifacts/` holds the **Mac's stored artifacts** (copied; `cache/v8` byte-identical to a local rebuild), and `ml/artifacts_win/` holds this machine's own Phase 1 re-run. Both are gitignored. The laptop GPU throttles thermally, so wall-clock is noisy; it affects time only, not results.
  - Run analysis scripts with `HADES_DEVICE=cpu` while a GPU cell trains, so the GPU stays at concurrency level 1.
- ML deps: `ml/requirements.txt` (torch, torch-geometric, lightgbm, sklearn, pandas). The generators need only stdlib + numpy/pandas. Keep the two environments separate.
- There's no package install and no pytest config. Every script puts its sibling dirs on `sys.path` itself and uses flat imports (`import phase5_heads as P5`), so always run from the repo root as `python ml/<dir>/<file>.py`. Nearly every script's module docstring gives its exact usage.

```bash
# Generate / validate the v8 worlds (5 seeds, 2 at a time; ~10–11 GB peak per process)
bash db/gen_v8/run_v8.sh                          # SEEDS="1001" to limit
python db/gen_v8/generator_v8.py --seed 1001 --out db/gen_v8
bash db/gen_v8/validate_all.sh                    # writes docs/v8/runs/ and docs/v8/seed_variance.*
python db/gen_v8/validator_v8.py db/gen_v8/seed_1001 [--levels 1234] [--selftest]

# Train one cell / predict from a bundle
python ml/train/loop.py train --config ml/configs/shipped.json --task arrival_week --world v8 --seed 7 \
    [--arch lite --depth 4 --lr 2.5e-4] [--origin N] [--drop-relation part] [--max-epochs 2 --bundle-root <scratch>]
python ml/train/loop.py predict --bundle <dir> [--h0-bundle <dir>] [--fold test]

# Tests: each file runs standalone (pytest also works for most of them)
python ml/tests/test_artifact_identity.py
python ml/tests/test_no_drift_switch.py
python ml/tests/test_no_lateness_substitution.py [--preds DIR]
python ml/tests/test_single_order_policy.py [--falsify]
python -m unittest ml/tests/test_phase7_bands.py
```

Multi-cell runs go through queue scripts (`ml/train/phase*_*.sh`, `phase*_queue.py`, `ml/eval/backtest.py queue`). They're resumable through `.done` marker files, and logs land in `ml/artifacts/`.

## Architecture: the parts that span files

- **`ml/config.py`** is the one source of truth for world paths (`WORLDS`: `v6`, `v7`, `v8` = v8 seed 1001, `v8s1002`…`v8s1005`), the frozen split (`train_end 2023-12-31`, `val_end 2024-12-31`, never change it), per-world panel width (`EXPECTED_PANEL_D`: 14 for v6/v7, 15 for v8) and expected entity counts.
- **Data path:** `ml/data/loader.py` does as-of reads and builds `HeteroData`. `ml/data/cache.py` materialises each world once into `ml/artifacts/cache/{world}/` (memmapped `panel.npy` `[channels, weeks, d]`, `graph.pt`, `meta.json`; rebuilds in ~6 s). The panel's live columns are *measured* from the world being loaded, never hardcoded. `ml/data/sequences.py` handles masking and normalisation.
- **Model:** `ml/models/tcn.py` → `share.py` (graph, 4 node types / 6 relations) → `heads.py` (hazard head for arrival, 22-cell binned CDF for fill, quantile head for capacity), with `staleness.py` gating. `ml/train/temporal_share.py` and `phase5_heads.py` hold the data path and losses. `ml/train/loop.py` owns the epoch loop.
- **A trained model is a *bundle*, not a checkpoint.** Every run writes the checkpoint, val/test predictions, recalibration fitted on the validation fold, a drift baseline, a normaliser, metrics and a config with git stamps. A bundle without recalibration is not shippable.
- **`ml/configs/shipped.json`** records the shipped choice per task (arch, depth, lr, seeds, recalibration). Edit it, not `loop.py`, when a choice changes. Its `_about` / `_superseded` / `_reversal_condition` fields record why. Caveat (guide deviation 46): the fill switch to LightGBM `b5flat22` is declarative only, and `loop.predict` can't serve it.
- **`ml/artifact_identity.py`** is the only place that names bundles, prediction files and index keys. When you add a config knob that changes what gets trained, add it to both `config_name()` and `identity_of()`, or two cells collide and one silently overwrites the other (deviation 28). `test_artifact_identity.py` enforces this.
- **Evaluation:** `ml/eval/backtest.py` (rolling origins 1–8, each carving out its own 12-month validation slice), `metrics.py`, `lateness_metric.py` (spec: `docs/specs/lateness_metric.md`), and `phase7_score.py` (the single scorer every arm, baselines included, goes through on asserted-identical rows).

## Standing rules (from `docs/specs/implementation_guide.md`, and they outrank any single step)

1. **A gate that can't fail isn't a gate.** Before you add a check, say what would make it fire and show that outcome is reachable. Tests and validator checks carry a falsification path (`--falsify`, `--selftest`, the `breaks` string on each `validator_v8.py` check). The repo has repeatedly caught checks that passed for every input.
2. **Assert artifact identity is unique at write time.** Never infer it from a naming convention (see above).
3. **Select on validation, never on an evaluation/test window.** If validation can't separate two options, report that as the finding.

Generator discipline (`docs/specs/synthetic_rules.md` §13, §15):

- Set **mechanism** parameters and **measure** outcomes. Never write a target outcome (fill mass, late rate, staleness, seasonality ratio, …) into a generator as a literal, clip or post-hoc adjustment. When an outcome misses its band, change a mechanism and regenerate.
- As-of gating is on `recorded_ts`, never `event_ts`. Weekly stores bucket on `max(event_week, recorded_week)`.
- `validator_v8.py` is **frozen**. Don't widen a band so a dataset passes. Only path discovery may change.

## Documentation conventions

- `docs/specs/` is the only place that describes *intent*. Everything else records what was *measured*. When the spec and the data disagree, the data wins, and the guide's **"Known deviations from spec"** table (numbered; reports continue the numbering, now past 90) records it. Check that table before you trust a spec number or column name, and check column names against the actual CSVs.
- Validation documents (`docs/validation/validationN.md`) are written once and never edited. Corrections go in a later document or in `validator_amendment_*.md`. A `runs/` folder holds raw instrument output, and the prose that cites it sits one level up.
- `db/validator.py` writes `docs/validation/external_dataset_validation.md`, and `validate_all.sh` writes `docs/v8/`. If you move either directory, update its writer too.
- Phase reports open with **Audience / Measured on (world + seed) / Companions / Status**. Every quoted figure names its world and seed, because v6, v7 and v8 differ materially. Reports cite the script, the artifact and the commit behind each number, and reproduce a known headline before trusting new numbers.
- Commit subjects follow `Phase 12 C3: <what>` or `reports/part2/phase-12.md: <what>`.

## Repo hygiene

- Never commit generated data or model artifacts (`*.csv`, `*.npz`, `*.pt`, `ml/artifacts/`, `db/gen_v*/seed_*/`). `.gitignore` covers these.
- The macOS AppleDouble files (`._*`) that appear as untracked are copy artifacts, not project files. Don't commit them.
- `prompts/` (phase briefs) is gitignored. Reports sometimes cite "the brief".
