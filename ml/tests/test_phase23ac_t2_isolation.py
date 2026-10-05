"""Phase 23AC T2 isolation -- the plan-table reader allow-list, each check with a way to fail.

1  ALLOW-LIST: an AST scan (ml/tests/test_phase18_isolation.violations, FORBIDDEN_STR = the plan table's name; docstrings
   excluded) over every NEW Phase 23AC module under ml/ that exists. Exactly ml/serve/_plan_reader.py may name the table;
   any other hit FAILS. The allow-listed reader must itself be flagged (else the scan is blind). A constructed second reader
   (source string) MUST be flagged; a docstring mention must pass.
2  PRE-EXISTING READERS: every ml/**/*.py (ml/artifacts excluded) naming the table outside a docstring, and the modules that
   import fwd_load, are REPORTED, not failed. Expected among them: ml/data/fwd_load.py, ml/data/phase20_world.py; and
   ml/serve/features.py imports fwd_load.
3  inventory_position_weekly is named by no new Phase 23AC module (constructed offender flagged).
4  SEASON POISON (unit, ONE test snapshot): every plan row with as_of_date + 2 d > t0 randomised; the season vector through
   ml/serve/_plan_reader.py is unchanged; the constructed offender (no bound) changes it. And the reader's bound fires on a
   forged row whose recorded_ts claims t0 - 7 d but whose as_of_date + 2 d > t0.

    HADES_DEVICE=cpu python ml/tests/test_phase23ac_t2_isolation.py
"""
from __future__ import annotations
import os, sys, ast, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "serve"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import test_phase18_isolation as T18

PLAN = "part_demand_weekly"
NEW_MODULES = ["ml/serve/_plan_reader.py", "ml/serve/fill_consolidated.py", "ml/serve/order_time_card.py",
               "ml/eval/phase23ac_t1_compare.py", "ml/eval/phase23ac_t2_serve.py", "ml/eval/leak_guard.py",
               "ml/eval/phase23ac_t4_metrics.py", "ml/eval/phase23ac_t4_charts_data.py"]
ALLOW = {"ml/serve/_plan_reader.py"}


def scan(path, forbidden, src=None):
    old = T18.FORBIDDEN_STR
    T18.FORBIDDEN_STR = tuple(forbidden)
    try:
        return T18.violations(path, src)
    finally:
        T18.FORBIDDEN_STR = old


def existing_new():
    return [m for m in NEW_MODULES if os.path.exists(os.path.join(REPO, m))]


# ------------------------------------------------------------------ 1
def test_plan_reader_allow_list():
    files = existing_new()
    assert "ml/serve/_plan_reader.py" in files and "ml/serve/fill_consolidated.py" in files, files
    hits = {m: scan(os.path.join(REPO, m), (PLAN,)) for m in files}
    others = {m: v for m, v in hits.items() if v and m not in ALLOW}
    assert not others, f"a new module other than the allow-listed reader names the plan table: {others}"
    assert hits["ml/serve/_plan_reader.py"], "the allow-listed reader is not flagged -- the scan is blind"
    # constructed failing cases
    second = "import pandas as pd\nD = 'x'\npdw = pd.read_csv(D + '/" + PLAN + ".csv')\n"
    assert scan("ml/serve/second_reader.py", (PLAN,), second), "a constructed second reader was NOT flagged"
    assert not scan("x.py", (PLAN,), '"""never reads ' + PLAN + ' (docstring)"""\nx = 1\n'), "a docstring was flagged"
    return dict(scanned=files, not_present=[m for m in NEW_MODULES if m not in files], allow_listed=sorted(ALLOW),
                hits={m: v for m, v in hits.items() if v}, constructed_second_reader="flagged", docstring="passes")


# ------------------------------------------------------------------ 2
def imports_module(path, name):
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name == name for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and n.module == name:
            return True
    return False


def test_report_pre_existing_readers():
    files = sorted(f for f in glob.glob(os.path.join(REPO, "ml", "**", "*.py"), recursive=True)
                   if os.sep + "artifacts" + os.sep not in f and not os.path.basename(f).startswith("._"))
    readers, via_fwd_load = [], []
    for f in files:
        rel = os.path.relpath(f, REPO)
        if rel.startswith("ml/tests/") or rel in ALLOW:
            continue
        if scan(f, (PLAN,)):
            readers.append(rel)
        if imports_module(f, "fwd_load"):
            via_fwd_load.append(rel)
    assert "ml/data/fwd_load.py" in readers, f"fwd_load not found as a reader -- the scan is blind: {readers}"
    res = dict(pre_existing_readers=readers, importers_of_fwd_load=via_fwd_load,
               expected_found=dict(fwd_load="ml/data/fwd_load.py" in readers, phase20_world="ml/data/phase20_world.py" in readers,
                                   features_imports_fwd_load="ml/serve/features.py" in via_fwd_load))
    print("  pre-existing plan readers (reported, not failed):", json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ 3
def test_no_inventory_position_in_new_modules():
    IPW = "inventory_position_weekly"
    hits = {m: scan(os.path.join(REPO, m), (IPW,)) for m in existing_new()}
    hits = {m: v for m, v in hits.items() if v}
    assert not hits, f"a new Phase 23AC module names {IPW}: {hits}"
    assert scan("x.py", (IPW,), "t = pd.read_csv(D + '/" + IPW + ".csv')\n"), "constructed offender NOT flagged"
    return dict(scanned=existing_new(), hits=0, constructed_offender="flagged")


# ------------------------------------------------------------------ 4
def test_season_poison_one_snapshot():
    import phase21_paths as PP
    import fwd_load as FL
    import _plan_reader as PR
    import phase23ac_t2_serve as T2
    PP.register()
    # the bound itself, on a forged row (no data needed)
    t0 = pd.Timestamp("2025-01-06")
    ok = pd.DataFrame(dict(as_of_date=[t0 - pd.Timedelta(days=2)]))
    assert PR.assert_plan_bound(ok, t0) == 1
    forged = pd.DataFrame(dict(as_of_date=[t0 - pd.Timedelta(days=1)], recorded_ts=[t0 - pd.Timedelta(days=7)]))
    try:
        PR.assert_plan_bound(forged, t0); fired = False
    except FL.AsOfViolation:
        fired = True
    assert fired, "the plan bound did not fire on as_of + 2 d > t0"
    # poison on one test snapshot
    S = PR.load_sources("v8")
    t0 = [pd.Timestamp(s) for s in FL.snapshot_dates("v8") if pd.Timestamp(s) > pd.Timestamp("2024-12-31")][0]
    clean = PR.snapshot_blocks(S, t0)
    Sp, n_poisoned = T2.poison_plan(S, t0, 2310)
    assert n_poisoned > 0, "nothing to poison -- the test would pass vacuously"
    bounded = PR.snapshot_blocks(Sp, t0)
    off = T2.offender_season(Sp, t0)
    same = np.array_equal(bounded["season"], clean["season"], equal_nan=True)
    off_same = np.array_equal(off, clean["season"], equal_nan=True)
    assert same, "season through the bounded reader changed under future-only poison"
    assert not off_same, "the constructed offender did not change season -- the poison test cannot fail"
    # the forged-recorded_ts row inside the window must stop the reader
    bad = S["pdw"].iloc[[0]].copy()
    bad["week_start"] = t0 + pd.Timedelta(weeks=2); bad["as_of_date"] = t0 - pd.Timedelta(days=1)
    bad["recorded_ts"] = t0 - pd.Timedelta(days=7)
    S2 = dict(S); S2["pdw"] = pd.concat([S["pdw"], bad], ignore_index=True)
    try:
        PR.snapshot_blocks(S2, t0); stopped = False
    except FL.AsOfViolation:
        stopped = True
    assert stopped, "a forged recorded_ts let a future version through the reader"
    return dict(t0=str(t0.date()), plan_rows_poisoned=n_poisoned, plan_rows_asserted=clean["n_plan_rows_asserted"],
                bounded_season_changed=not same, offender_season_changed=not off_same, forged_row="stopped")


if __name__ == "__main__":
    for name in ("test_plan_reader_allow_list", "test_report_pre_existing_readers", "test_no_inventory_position_in_new_modules",
                 "test_season_poison_one_snapshot"):
        r = globals()[name]()
        print(f"  PASS {name}: {json.dumps(r, default=str) if name != 'test_report_pre_existing_readers' else ''}")
