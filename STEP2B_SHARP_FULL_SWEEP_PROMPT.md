# Give SHARP the same full-variant treatment SHARE got

> Paste into a fresh session. Standard effort — this reuses the existing harness, no new
> architecture or protocol work needed. Read `reports/phase7_training_results.md` first,
> specifically §5.1 (coverage), §5.2 (SHARE's per-variant table), and the process-level note about
> `OMP_NUM_THREADS` — reuse that thread configuration, don't repeat the slow one.

## Task

Run SHARP (`rgcn_relemb`) across all twelve variants, five seeds each, at the same
V1-comparable configuration SHARE's sweep used (`sup_n=800`, 15 snapshots) — same scale, same
splits, same matched-parameter setup, same block-bootstrap evaluation protocol already implemented
in `ml/run_benchmark_eval.py`. This is not new harness work: point the existing tool at a different
architecture.

**Budget check.** SHARE's own measured cost at this scale was ~419s/run (§3.1). SHARP is the same
size (752,599 vs 752,211 params), so the estimate is ~60 runs × ~420s ÷ 4 parallel workers ≈
1.5–2 hours wall-clock, comfortably inside a 5-hour budget — but this is extrapolated from SHARE's
numbers, not measured for SHARP. Time the first 2–3 runs and confirm the estimate holds before
committing to the full 60; if SHARP is meaningfully slower per run than SHARE for any reason, say
so and recompute before running the rest.

**MPS — benchmark before committing, don't assume.** §3.1 found MPS *slower* than CPU for this
model family at spec scale (96s vs 86s per pass), because the workload is scatter/gather-bound and
that's where Apple's GPU backend is weakest — but that was measured at 762k nodes, not this
sweep's ~28k-node scale, where memory pressure and kernel launch overhead behave differently. Run
one SHARE or SHARP pass on both `--device cpu` and `--device mps` first, at this sweep's actual
scale, and use whichever is faster for the full run. Report both numbers either way — if MPS wins
here despite losing at spec scale, that's worth knowing for future sweeps at this scale; if CPU
still wins, say so and proceed on CPU rather than forcing MPS.

- Seeds 42/43 are on disk; 44/45/46 regenerate via `db/regenerate_seed.py` as the harness already
  does for SHARE's sweep — same mechanism, no changes needed.
- Use four parallel worker processes at three threads each (the setting that gave SHARE's sweep an
  8.6× speedup over the naive one-thread-per-process approach) as the CPU baseline — unless the MPS
  check above says otherwise.
- Produce the same table shape as `reports/phase7_training_results.md` §5.2: variant, mechanisms,
  delay/shortage/impact AUC with std over the five seeds, and positive count beside each AUC.

## Deliverable

Append a new section to `reports/phase7_training_results.md` (don't create a separate file) with:

- SHARP's full per-variant table, in the same format as §5.2.
- The same variant-effect breakdown as §5.3, but for SHARP: each variant's delta against its
  mandated reference (K vs D, A vs J, F vs E, everything else vs 0), with sign agreement across
  the five seeds.
- A direct comparison to SHARE's existing per-variant numbers: does SHARP's response to each
  mechanism track SHARE's, or does any variant affect the two architectures differently? This is
  the actual point of running SHARP everywhere rather than just on Variant 0 — right now there's no
  evidence either way, since SHARP has only ever been tested on the plain dataset.
- Apply the same caveats already established: Mechanism G's delay AUC increase is a task-difficulty
  artifact, not a real signal (§5.4 of the existing report); any null should be reported as
  "not detected at this label volume," not as an established absence, since this is still the
  V1-comparable scale, not spec scale.

Do not re-run SHARE, SHARK, or any other architecture in this session — this is SHARP only, to
close the specific gap the earlier sweep left open.
