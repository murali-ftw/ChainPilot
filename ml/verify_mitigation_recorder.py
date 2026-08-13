#!/usr/bin/env python3
"""
Gate 0, check 3 — is the mitigation recorder CORRECT, not merely inert?

Inertness (byte-identical output) and correctness are different failure modes. A recorder that
writes the wrong value, or writes at the wrong key, or writes a value the simulation then
recomputes differently, would still produce byte-identical CSVs and still pass the byte-identity
check -- and would be useless. `reports/phase1_mitigation_level.md` established inertness; this
script establishes correctness.

**Method: independent capture at the USE SITES, then compare.** `MITIGATION_HISTORY` is written
inside `mitigation_level()` (`db/generate_dataset.py:1213-1226`). The values the simulation
actually *uses* are consumed somewhere else entirely -- two places in the weekly replenishment
loop:

    _mit     = max((mitigation_level(s_, week) for s_ in _cand), default=0.0)   :1258
    _trigger = 1.55 + 0.45 * _mit                                               :1259
    _qty    *= 1.0 + 0.35 * mitigation_level(sup, week)                         :1269

This script runs the generator with an in-memory source patch that records, at those use sites,
the value that was actually consumed -- independently of the recorder under test -- and then
checks every use-site observation against `MITIGATION_HISTORY`. The committed generator is not
modified; the patch exists only inside this process, the same mechanism
`ml/extract_hidden_state.py` and `db/run_benchmark.py --stats-only` already use.

A mismatch anywhere means the recorded history is not what drove the simulation.

    python3 ml/verify_mitigation_recorder.py --variant E --seed 42 --config v1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "db", "generate_dataset.py")

# The two use sites, verbatim from db/generate_dataset.py, and their instrumented forms.
USE_SITE_TRIGGER_SRC = """            _mit = max((mitigation_level(s_, week) for s_ in _cand), default=0.0)"""
USE_SITE_TRIGGER_NEW = """            _mit_vals = [(s_, mitigation_level(s_, week)) for s_ in _cand]
            for _s_, _v_ in _mit_vals:
                USE_SITE_OBS.append(("trigger", _s_, week, _v_))
            _mit = max((v for _s_, v in _mit_vals), default=0.0)"""

USE_SITE_QTY_SRC = """                _qty *= 1.0 + 0.35 * mitigation_level(sup, week)"""
USE_SITE_QTY_NEW = """                _q_mit = mitigation_level(sup, week)
                USE_SITE_OBS.append(("qty", sup, week, _q_mit))
                _qty *= 1.0 + 0.35 * _q_mit"""


def run_instrumented(variant: str, seed: int, config: str) -> dict:
    src = open(GEN).read()

    # Guard: if a use site has drifted from what this script expects, fail loudly rather than
    # silently verifying nothing.
    for name, needle in (("trigger", USE_SITE_TRIGGER_SRC), ("qty", USE_SITE_QTY_SRC)):
        if src.count(needle) != 1:
            raise RuntimeError(
                f"use site '{name}' not found exactly once in {GEN} "
                f"(found {src.count(needle)}). The generator changed; update this script "
                f"rather than reporting a vacuous pass.")

    src = src.replace("MITIGATION_HISTORY = {}", "MITIGATION_HISTORY = {}\nUSE_SITE_OBS = []", 1)
    src = src.replace(USE_SITE_TRIGGER_SRC, USE_SITE_TRIGGER_NEW, 1)
    src = src.replace(USE_SITE_QTY_SRC, USE_SITE_QTY_NEW, 1)
    src = src.replace("def write(name, header, rows):",
                      "def write(name, header, rows):\n    return", 1)
    src = src.replace(
        'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
        '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
        "MANIFEST_OUT = _manifest", 1)

    argv = ["generate_dataset.py", "--variant", variant, "--config", config, "--seed", str(seed)]
    saved, sys.argv = sys.argv, argv
    mod = types.ModuleType("__mitig_verify__")
    mod.__file__ = GEN
    sys.modules["__mitig_verify__"] = mod
    ns = mod.__dict__
    ns["__name__"] = "__mitig_verify__"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit:
                pass
    finally:
        sys.argv = saved
    return ns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="E")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="v1")
    ap.add_argument("--show", type=int, default=12, help="sample rows to print for manual trace")
    args = ap.parse_args()

    print(f"running instrumented generator: variant {args.variant}, seed {args.seed}, "
          f"config {args.config} ...", flush=True)
    ns = run_instrumented(args.variant, args.seed, args.config)
    MH, OBS = ns["MITIGATION_HISTORY"], ns["USE_SITE_OBS"]

    print(f"\nrecorder entries      : {len(MH):,}")
    print(f"use-site observations : {len(OBS):,}  "
          f"({sum(1 for o in OBS if o[0]=='trigger'):,} trigger, "
          f"{sum(1 for o in OBS if o[0]=='qty'):,} qty)")

    missing, mismatched, checked = [], [], 0
    for site, sup, week, used in OBS:
        key = (sup, week)
        if key not in MH:
            missing.append((site, sup, week, used))
            continue
        checked += 1
        if MH[key] != used:                      # exact equality, not a tolerance
            mismatched.append((site, sup, week, used, MH[key]))

    print(f"\nchecked               : {checked:,} use-site values against the recorder")
    print(f"missing from recorder : {len(missing):,}")
    print(f"value mismatches      : {len(mismatched):,}")

    def trace(rows, title):
        print(f"\n--- manual trace: {title} ---")
        print(f"{'site':<8} {'supplier':<14} {'week':<20} {'used by sim':>12} "
              f"{'recorded':>12} {'exact':>6}")
        for site, sup, week, used in rows:
            rec = MH.get((sup, week))
            print(f"{site:<8} {sup[:12]:<14} {str(week)[:19]:<20} {used:>12.6f} "
                  f"{(rec if rec is not None else float('nan')):>12.6f} "
                  f"{('yes' if rec == used else 'NO'):>6}")

    # The earliest observations are all 0.0 (no observed history yet -> the seen[0] < 3 branch),
    # which would make a trivially-passing trace. Show non-zero values as well, and cover both
    # use sites, so the trace actually exercises the computed branch.
    trace(OBS[:args.show], f"first {args.show} observations (early weeks, pre-history)")
    nz = [o for o in OBS if o[3] > 0]
    trace(nz[:args.show], f"first {args.show} NON-ZERO observations (computed branch)")
    nz_qty = [o for o in nz if o[0] == "qty"]
    trace(nz_qty[:args.show], f"first {args.show} non-zero observations at the QTY use site")

    # Non-vacuity: a recorder that captured nothing would trivially report zero mismatches.
    nonzero_used = sum(1 for _s, _u, _w, v in OBS if v > 0)
    print(f"\nnon-vacuity check     : {nonzero_used:,} of {len(OBS):,} use-site values are > 0 "
          f"({100.0*nonzero_used/max(1,len(OBS)):.1f}%)")

    ok = (not missing) and (not mismatched) and checked > 0 and nonzero_used > 0
    print("\n" + "=" * 78)
    print(f"GATE 0 CORRECTNESS: {'PASS' if ok else 'FAIL'} — "
          f"{'every value the simulation used matches the recorded value exactly'
             if ok else 'see missing/mismatch counts above'}")
    print("=" * 78)
    if mismatched[:5]:
        for m in mismatched[:5]:
            print("  mismatch:", m)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
