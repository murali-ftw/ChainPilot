# Phase 0 — Environment and guide correction

Device: **MPS (Apple Silicon)**, torch 2.14.0. Phase wall-clock ≈ 9 min (dominated by the
torch/torch-geometric download). Peak RSS 3.21 GB.

---

## My reading of the phase scope — stated, not guessed

`docs/implementation_guide.md` defines **ten** phases. Its phases 0–3 are
**Environment**, **Data loading**, **Sequence assembly**, **Temporal encoder**. The task attaches
different work to phases 2 and 3 — baseline reproduction and G5 to phase 2, cross-world evaluation
and h⁰/h⁴ to phase 3 — which in the guide live in **Phase 7 (Baselines)** and **Phase 8
(Evaluation)**.

I read the task as: **implement the guide's phase N, and additionally satisfy the task's overlay
for phase N.** So:

| | guide's phase | task overlay |
|---|---|---|
| **Phase 0** | 0.1 dependencies, 0.2 `ml/` layout, 0.3 world choice | device policy, guide correction |
| **Phase 1** | 1.1 reader, 1.2 as-of sweep, 1.3 node tables, 1.4 `HeteroData`, 1.5 snapshot cache | graph verification against run-7 figures, inductive-only, both worlds |
| **Phase 2** | 2.1 `[T×d]` tensors, 2.2 masking, 2.3 normalisation | reproduce the four v7 baselines, run G5 |
| **Phase 3** | 3.1 dilated causal TCN | cross-world 2×2, h⁰/h¹/h⁴ with a learned encoder |

The guide's Phase 7 h⁰ ablation (step 7.3, "mandatory") is the same measurement the task's Phase 3
asks for, pulled forward. I treat them as one.

---

## P0.1 Guide correction

Backup of the original at `scratchpad/guide_before.md`; **104 lines added, 80 removed**.

### 1. Gzip → plain CSV, and made format-agnostic

`### Step 1.1 — Gzipped CSV reader` → `### Step 1.1 — CSV reader (format-agnostic)`. The reader
now resolves the extension instead of assuming one:

```python
def table_path(csv_dir: str, table: str) -> str:
    for ext in (".csv", ".csv.gz"):
        p = os.path.join(csv_dir, table + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{table}[.csv|.csv.gz] not under {csv_dir}")

def _open(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") \
           else open(path, "rt", newline="")
```

**40 occurrences of `.csv.gz` → `.csv`.** Remaining `gzip` mentions are inside the new
format-agnostic helper, which is the point.

### 2. Layout

| Was | Now |
|---|---|
| `db/csv_full_seed1/derived/…` | `db/gen_v7/seed_1001/…` |
| `db/csv_full_seed1/…` | `db/gen_v7/seed_1001/…` |
| `db/csv_mid_seed1/`, `db/csv_small_seed1/` | `db/gen_v6/seed_1001/` |
| `derived/` prefix (15 uses) | removed — the worlds are flat |

`db2/` and `gen_v5/` appear **nowhere** in the guide (grep count 0), so there was nothing to fix
there. Three prose references to `csv_full_seed1` survive and are now labelled *"the reference
world … (outside this repo)"* — they cite measurements of the control dataset, which still exists
at `/Users/muralik/Documents/Programs/HADES/db/csv_full_seed1` and is not one of our two worlds.

### 3. Step 0.3 rewritten — this was more than a path change

The three size presets (`small`/`mid`/`full`) no longer exist. A blanket path substitution turned
that table into three rows pointing at two directories, which is worse than leaving it. I replaced
the section: two **worlds**, both at full scale, differing by regime rather than size, each with its
committed generator and a `seed_1002` twin.

### 4. Stale measured counts — the guide was stale in more than the two stated ways

Every `Verify` block asserted counts from a different dataset lineage. Left uncorrected they fail
on first use. Replaced with values measured in Stage A, each labelled with the world it came from:

| Assertion | Was | Now |
|---|---|---|
| `po_lines` rows | 1,169,215 | **982,585** (v7; v6 is 1,178,254) |
| `channel_performance_weekly` rows | 5,930,225 | **8,598,520** |
| `grn_lines` rows | 2,054,522 | **953,123** |
| `supplier_index` | 902 | **420** |
| `part_index` | 8,055 | **620** |
| `group_index` | 225 | **84** |
| `supplier_upstream` edges | 231 | **260** |
| `part→product` (BOM) edges | 18,532 | **≤ 4,000** |
| snapshots | 115, 28-day spacing, `horizon_days = 91` | **83, 42-day spacing, `horizon_days = 90`** |
| channels present in the weekly store | 15,333 of 16,072 | **15,334 of 16,072** (738 cold-start, 4.59%) |
| distinct weeks | 523 | **535** |
| weeks per channel | min 19, median 458 | **535 for every channel — a complete contiguous panel** |
| channels with < 52 weeks | 222 (1.45%) | **0** |

The last two matter for Phase 2: the guide's left-padding instruction was written for a ragged
panel. Our stores are complete, so padding is a no-op — but I left the instruction in place because
it is still correct and costs nothing.

**Not changed:** anything in phases 4–10, the architecture specifications, or any band.

---

## P0.2 Device policy

`ml/device.py`, imported by every later phase.

```
[device] MPS -- MPS available and built
[device] torch 2.14.0 | dtype torch.float32 | PYTORCH_ENABLE_MPS_FALLBACK=1 | num_workers=0
```

| Rule | Implementation |
|---|---|
| Prefer MPS, print the reason on fallback | `get_device()` — three branches, each printing why |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | `os.environ.setdefault` **before** `import torch`, or the fallback is not registered |
| float32 everywhere | `DTYPE = torch.float32`; `to_t()` downcasts any float64 before it reaches the device |
| `num_workers=0` | `NUM_WORKERS = 0` |
| Seed torch / numpy / random | `seed_everything()`, including `torch.mps.manual_seed` |

`to_t()` exists because pandas hands back float64 and MPS has no float64 kernel — verified:
`to_t(np.zeros(4, dtype=np.float64)).dtype` → `torch.float32`.

### Timing — MPS vs CPU, evidenced

Representative op: the Phase 3 TCN geometry — six dilated causal `Conv1d` layers (k=2, dilations
1–32, hidden 64) over a `[4096, 38, 52]` batch, 12 forwards after warm-up.

| device | ms / forward | speedup |
|---|---|---|
| **MPS** | **11.5** | **124.8×** |
| CPU | 1436.2 | 1× |

The choice is evidenced, not assumed.

### Determinism — the measured result contradicts the expectation

The task asks me to note that MPS is not bit-deterministic and record the magnitude. **On this
workload it is bit-deterministic.** Two fresh processes, same seed, 40 Adam steps on the TCN with
backward:

| | run-to-run, fresh process | vs CPU, same seed |
|---|---|---|
| MPS | max \|Δloss\| **0.000e+00 — bit-identical** | — |
| CPU | max \|Δloss\| 0.000e+00 | — |
| MPS vs CPU | — | max \|Δloss\| 6.57e-05, **relative 5.7e-05** |

So the reproducibility risk here is **cross-backend, not cross-run**: a number produced on MPS and
one produced on CPU differ in the fifth significant figure from float32 accumulation order alone.
A result must state its device.

Caveat I am not generalising past: this covers `Conv1d` + `Adam`. Scatter-based message passing in
Phase 3 uses atomics, where MPS may well be non-deterministic. I will re-measure there rather than
assume this carries.

### Memory

Unified memory means the dataset and the model share one pool. Full end-to-end load of a world,
with column pruning on `channel_performance_weekly` (4 of 20 columns) and `inventory_transactions`
(4 of 9):

| world | tables | rows | wall | peak RSS |
|---|---|---|---|---|
| v6 | 49 | 31,943,035 | 11.0 s | 2.53 GB |
| v7 | 49 | 31,531,523 | 10.5 s | **3.21 GB** |

Comfortable against 24 GB. **No row was subsampled and no entity population reduced** — the
pruning is columnar only.

---

## P0.3 The guide's own Phase 0

| Step | Status | Evidence |
|---|---|---|
| **0.1 Dependencies** | **done** | `ml/requirements.txt` written. `python -c "import torch, torch_geometric, lightgbm"` → torch 2.14.0, pyg 2.8.0.post1, lgbm 4.7.0, sklearn 1.9.0. `ortools` **not installed** — it is Phase 10 only and out of scope; commented in the requirements file with that reason. |
| **0.2 Directory layout** | **done** | `ml/{data,models,train,baselines,eval,sim,opt,artifacts}` created. `ml/artifacts/` is covered by `.gitignore` via `checkpoints/`/`runs/` plus `*.pt`; I verified no artefact path would be staged. |
| **0.3 World choice** | **done** | `ml/config.py` pins `WORLDS`, the frozen `SPLIT` (train ≤ 2023 / val 2024 / test 2025), `FIT_WINDOW` 2019–2025, and the Stage-A entity counts as `EXPECT`. |

One deviation from the guide's 0.1: it says *"the generator itself needs none of this — keep the two
environments separate."* There is one `venv/` here, shared. I did not split it, because the
generators are frozen for this work and no phase regenerates a world; splitting would be ceremony.
Recorded as a deliberate deviation.

**A torch `FutureWarning` appears on import**: `torch.jit.script is not supported in Python
3.14+`. It is emitted by torch's own import machinery, affects nothing we call, and is noted so it
is not mistaken later for a fault in our code.

---

## Gate

| Requirement | Result |
|---|---|
| Environment verified | **pass** — MPS live, 124.8× over CPU, deps import |
| Guide corrected | **pass** — gzip and layout fixed, loader made format-agnostic, 15 stale counts corrected |
| Both worlds loadable end to end | **pass** — 49/49 tables each, 11 s, 3.21 GB peak |

**Phase 1 may start.**
