"""Phase 20 isolation audit (Stage 5) -- each check with a way to fail.

1  PROTECTED PATHS: `git diff <base> HEAD` and the working tree are empty on every path the Phase 20 brief protects,
   including every report that existed at the base and ml/train/phase19_identity.py.
2  db/ UNTOUCHED after the second-world generation: no tracked change, NO untracked file under db/, and db/gen_v8 holds
   exactly the seed directories it held at the base (seed_1001..seed_1005). The second world lives in data_worlds/.
3  STORED IDENTITIES: every stored bundle config recomputes identically (Phase 18's check, base = this phase's base), and
   every Phase 19 bundle still sits at the name phase19_identity gives it (Phase 20 adds no neural config).
4  NO PRIVILEGED INPUT: Phase 18's AST scanner over ml/train, ml/models, ml/data, ml/baselines, its forbidden strings
   extended to Phase 19-20 privileged paths; it must flag constructed Phase 20 offenders and pass a docstring mention.
   Fresh interpreters importing the Phase 20 modules load nothing from reports/.
5  inventory_position_weekly is named by no Phase 20 module outside its docstring.

    python ml/tests/test_phase20_isolation.py
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train")]
import test_phase18_isolation as T18
import test_phase19_isolation as T19

BASE = "e9c0862"                          # phase19 tip, the Phase 20 base
PROTECTED = ["db/gen_v6", "db/gen_v7", "db/gen_v8", "db/validator.py", "docs/specs", "db/dataset_structure.md",
             "ml/configs/shipped.json", "results", "reports/part1", "ml/models/share.py", "ml/models/tcn.py",
             "ml/artifact_identity.py", "ml/train/phase19_identity.py"]
P20_MODULES = ["ml/data/phase20_world.py", "ml/data/fwd_pred.py", "ml/data/stock_asof.py", "ml/baselines/phase20_proxy.py",
               "ml/baselines/phase20_forecaster.py"]


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True)


def test_protected_paths_unchanged():
    prior = [p for p in git("ls-tree", "-r", "--name-only", BASE, "reports").split() if p.endswith((".md", ".json"))]
    paths = PROTECTED + prior
    assert not git("diff", "--stat", BASE, "HEAD", "--", *paths).strip(), "protected paths changed"
    assert not git("status", "--porcelain", "--", *paths).strip(), "protected paths modified in the working tree"
    assert git("diff", BASE, "HEAD", "--", "ml/models/share.py", "ml/models/tcn.py") == ""
    return dict(base=BASE, protected=PROTECTED, prior_reports_checked=len(prior), committed_diff="empty", worktree="clean")


def test_db_untouched():
    st = git("status", "--porcelain", "--untracked-files=all", "--ignored=no", "--", "db").strip()
    assert not st, f"db/ has changes or untracked files: {st[:500]}"
    seeds = sorted(d for d in os.listdir(os.path.join(REPO, "db", "gen_v8")) if d.startswith("seed_"))
    assert seeds == [f"seed_{s}" for s in (1001, 1002, 1003, 1004, 1005)], f"db/gen_v8 seed directories changed: {seeds}"
    w2 = os.path.join(REPO, "data_worlds", "v8_seed1002", "seed_1002", "manifest.json")
    assert os.path.exists(w2), "the second world is not where Stage 2 put it"
    assert "data_worlds/" in open(os.path.join(REPO, ".gitignore")).read()
    return dict(db_status="clean (tracked and untracked)", gen_v8_seed_dirs=seeds, second_world="data_worlds/v8_seed1002 (ignored)")


def test_identities():
    T18.BASE = BASE
    r = T18.test_every_stored_identity_recomputes()
    import phase19_identity as PI
    p19 = sorted(glob.glob(os.path.join(PI.BUNDLES19, "*", "*", "config.json")))
    for f in p19:
        assert os.path.basename(os.path.dirname(f)) == PI.bundle_name(json.load(open(f))), f
    r.update(phase19_bundles_named_correctly=len(p19), phase20_new_neural_configs=0)
    return r


def test_no_privileged_input():
    T18.FORBIDDEN_STR = T18.FORBIDDEN_STR + ("PRIVILEGED__", "phase19/PRIVILEGED", "phase19/preds", "phase20/PRIVILEGED",
                                             "phase20/preds")
    r = T18.test_no_privileged_input_in_ml()
    for off in ("p = 'reports/part2/phase20/preds/v8_arrival_week_PRIVILEGED__reorder_probe_s7_test.npz'\n",
                "z = np.load('reports/part2/phase20/PRIVILEGED__probe_features_v8s1001.npz')\n",
                "from reports.part2.phase20 import PRIVILEGED__reorder_probe\n"):
        assert T18.violations("x.py", off), f"scanner missed a Phase 20 offender: {off!r}"
    assert not T18.violations("x.py", '"""reads nothing from reports/part2/phase20/PRIVILEGED__ (docstring)"""\nx = 1\n')
    code = ("import sys; sys.path[:0]=['ml','ml/data','ml/baselines','ml/train','ml/eval'];"
            "import phase20_world, fwd_pred, stock_asof, phase20_proxy, phase20_forecaster;"
            "bad=[m for m,v in list(sys.modules.items()) if getattr(v,'__file__',None) and '/reports/' in v.__file__];"
            "assert not bad, bad; assert 'torch' not in sys.modules; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    r["phase20_offenders_flagged"] = 3
    r["phase20_runtime_import_check"] = "phase20_world, fwd_pred, stock_asof, phase20_proxy, phase20_forecaster: nothing from reports/, no torch"
    return r


def test_no_inventory_position_weekly():
    saved = T18.FORBIDDEN_STR
    T18.FORBIDDEN_STR = ("inventory_position_weekly",)
    try:
        hits = {m: T18.violations(os.path.join(REPO, m)) for m in P20_MODULES}
        assert T18.violations("x.py", "t = pd.read_csv(D + '/inventory_position_weekly.csv')\n"), "scanner cannot see the table name"
    finally:
        T18.FORBIDDEN_STR = saved
    bad = {m: v for m, v in hits.items() if v}
    assert not bad, bad
    return dict(phase20_modules_scanned=P20_MODULES, reads=0)


if __name__ == "__main__":
    res = {}
    for name in ("test_protected_paths_unchanged", "test_db_untouched", "test_identities", "test_no_privileged_input",
                 "test_no_inventory_position_weekly"):
        res[name] = globals()[name]()
        print(f"  PASS {name}")
    out = os.path.join(REPO, "ml", "artifacts", "phase20", "isolation_audit.json")
    json.dump(res, open(out, "w"), indent=1)
    r = res["test_identities"]
    print(f"  stored configs recomputed identically: {r['recomputed_identical']} (named by this base {r['named_by_this_base']}; "
          f"cannot be named {len(r['cannot_be_named'])}); Phase 19 bundles named correctly: {r['phase19_bundles_named_correctly']}")
