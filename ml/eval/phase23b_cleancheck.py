"""Phase 23B -- re-run Phase 22's clean-panel tests (ml/tests/test_phase22_clean_panel.py, imported UNCHANGED) on the clean
inputs as rebuilt on this machine, writing to ml/artifacts/phase23b/ (never into phase22/). Same weeks and seed as Phase 22.

  future poison   the nine clean columns at row t must not move when later-visible source rows are poisoned; OFFENDER (the
                  generator's order-keyed formula on the same poisoned CSVs) must move
  self-exclusion  a snapshot row's own (later-raised) line never enters its row; OFFENDER (order visibility dropped) must

  python ml/eval/phase23b_cleancheck.py --world v8|v8w1002
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config, phase21_paths as PP
sys.path.insert(0, os.path.join(C.ML, "tests"))
import test_phase22_clean_panel as T22
import clean_panel as CP


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="v8"); a = ap.parse_args()
    PP.register()
    st = C.require_clean()
    meta = json.load(open(os.path.join(config.CACHE, a.world + "clean", "meta.json")))
    assert meta.get("clean_of") == a.world and meta.get("replaced_columns") == CP.LEAKING
    S = CP.CsvSources(a.world, meta["T"])
    rng = np.random.default_rng(220)
    weeks = [int((pd.Timestamp(d) - CP.W0).days // 7) for d in ("2019-07-01", "2022-03-07", "2025-06-02")]
    out = dict(stamp=st, world=a.world, clean_cache_commit=meta.get("code_commit"), tests="ml/tests/test_phase22_clean_panel.py (unchanged)",
               future_poison=T22.test_future_poison(S, weeks, rng))
    out["self_exclusion"] = T22.test_self_exclusion(S, a.world, rng)
    C.dump(out, f"phase23b/clean_panel_tests_{a.world}.json")
    print(json.dumps(out, default=str)[:1500]); print("ALL PASS")


if __name__ == "__main__":
    main()
