"""Phase 21 isolation audit (Stage 8) -- each check with a way to fail.

1  PROTECTED PATHS: `git diff <base> HEAD` and the working tree are empty on every path the Phase 21 brief protects,
   including every report that existed at the base, ml/train/phase19_identity.py and ml/serve/**.
2  db/ UNTOUCHED: the worktree's db/ has no tracked change and no untracked file; in the main checkout (where the worlds
   live, HADES_DATA_ROOT) db/ is clean and db/gen_v8 holds exactly seed_1001..seed_1005; the second world is still in
   data_worlds/ with its Phase 20 hashes.
3  STORED IDENTITIES: every stored bundle config recomputes identically (Phase 18's check, base = this phase's base), and
   every Phase 19 bundle sits at the name phase19_identity gives it (Phase 21 adds no neural config).
4  NO PRIVILEGED INPUT: Phase 18's AST scanner over ml/train, ml/models, ml/data, ml/baselines, forbidden strings extended
   to Phase 19-21 privileged paths; it must flag constructed Phase 21 offenders and pass a docstring mention. A fresh
   interpreter importing the Phase 21 modules loads nothing from reports/ and no torch.
5  inventory_position_weekly is named by no Phase 21 module outside its docstring (scanner self-tested).
6  FUTURE-POISON and SELF-EXCLUSION (ml/tests/test_phase21_grpstats.py), with their constructed offenders, on both worlds.

    HADES_DATA_ROOT=<main checkout> python ml/tests/test_phase21_isolation.py
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train"),
                os.path.join(HERE, "..", "eval")]
import test_phase18_isolation as T18
import phase21_paths as PP

BASE = "124be5b"                          # HADES-v4-ml-pipeline tip, the Phase 21 base
PROTECTED = ["db/gen_v6", "db/gen_v7", "db/gen_v8", "db/validator.py", "docs/specs", "db/dataset_structure.md",
             "ml/configs/shipped.json", "results", "reports/part1", "ml/models/share.py", "ml/models/tcn.py",
             "ml/artifact_identity.py", "ml/train/phase19_identity.py", "ml/serve"]
P21_MODULES = ["ml/data/grpstats.py", "ml/data/phase21_paths.py", "ml/baselines/phase21_proxy.py"]


def git(*a, repo=REPO):
    return subprocess.check_output(["git", "-C", repo, *a], text=True)


def test_protected_paths_unchanged():
    prior = [p for p in git("ls-tree", "-r", "--name-only", BASE, "reports").split() if p.endswith((".md", ".json"))]
    paths = PROTECTED + prior
    assert not git("diff", "--stat", BASE, "HEAD", "--", *paths).strip(), "protected paths changed"
    assert not git("status", "--porcelain", "--", *paths).strip(), "protected paths modified in the working tree"
    assert git("diff", BASE, "HEAD", "--", "ml/models/share.py", "ml/models/tcn.py") == ""
    changed = git("diff", "--name-only", BASE, "HEAD").split()
    return dict(base=BASE, protected=PROTECTED, prior_reports_checked=len(prior), committed_diff="empty", worktree="clean",
                files_changed_by_phase21=changed)


def test_db_untouched():
    st = git("status", "--porcelain", "--untracked-files=all", "--", "db").strip()
    assert not st, f"worktree db/ has changes or untracked files: {st[:500]}"
    root = PP.DATA_ROOT
    st_main = git("status", "--porcelain", "--untracked-files=all", "--", "db", repo=root).strip()
    assert not st_main, f"main checkout db/ has changes: {st_main[:500]}"
    seeds = sorted(d for d in os.listdir(os.path.join(root, "db", "gen_v8")) if d.startswith("seed_"))
    assert seeds == [f"seed_{s}" for s in (1001, 1002, 1003, 1004, 1005)], f"db/gen_v8 seed directories changed: {seeds}"
    PP.register()
    rec = json.load(open(os.path.join(REPO, "ml", "artifacts", "phase20", "world2_generation.json")))["key_tables"]
    now = PP.world_hashes(PP.WORLD2, tables=("po_lines.csv", "grn_lines.csv", "training_labels.csv", "sourcing_channels.csv"))
    assert all(now[t] == rec[t]["sha1"] for t in now), "the second world's bytes changed since Phase 20"
    return dict(worktree_db="clean (tracked and untracked)", main_checkout_db="clean", data_root=root, gen_v8_seed_dirs=seeds,
                second_world_hashes_equal_phase20=list(now))


def test_identities():
    T18.BASE = BASE
    r = T18.test_every_stored_identity_recomputes()
    import phase19_identity as PI
    p19 = sorted(glob.glob(os.path.join(PI.BUNDLES19, "*", "*", "config.json")))
    for f in p19:
        assert os.path.basename(os.path.dirname(f)) == PI.bundle_name(json.load(open(f))), f
    r.update(phase19_bundles_named_correctly=len(p19), phase21_new_neural_configs=0)
    return r


def test_no_privileged_input():
    T18.FORBIDDEN_STR = T18.FORBIDDEN_STR + ("PRIVILEGED__", "phase19/PRIVILEGED", "phase19/preds", "phase20/PRIVILEGED",
                                             "phase20/preds", "phase21/PRIVILEGED")
    r = T18.test_no_privileged_input_in_ml()
    for off in ("p = 'reports/part2/phase21/PRIVILEGED__group_oracle.npz'\n",
                "z = np.load('reports/part2/phase20/PRIVILEGED__probe_features_v8s1001.npz')\n",
                "from reports.part2.phase21 import PRIVILEGED__oracle\n"):
        assert T18.violations("x.py", off), f"scanner missed an offender: {off!r}"
    assert not T18.violations("x.py", '"""reads nothing from reports/part2/phase21/PRIVILEGED__ (docstring)"""\nx = 1\n')
    code = ("import sys; sys.path[:0]=['ml','ml/data','ml/baselines','ml/train','ml/eval'];"
            "import grpstats, phase21_paths, phase21_proxy;"
            "bad=[m for m,v in list(sys.modules.items()) if getattr(v,'__file__',None) and '/reports/' in v.__file__];"
            "assert not bad, bad; assert 'torch' not in sys.modules; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    r["phase21_offenders_flagged"] = 3
    r["phase21_runtime_import_check"] = "grpstats, phase21_paths, phase21_proxy: nothing from reports/, no torch"
    r["privileged_files_in_phase21"] = sorted(os.listdir(os.path.join(REPO, "reports", "part2", "phase21")))
    assert not any("PRIVILEGED" in f for f in r["privileged_files_in_phase21"])
    return r


def test_no_inventory_position_weekly():
    saved = T18.FORBIDDEN_STR
    T18.FORBIDDEN_STR = ("inventory_position_weekly", "part_demand_weekly")
    try:
        hits = {m: T18.violations(os.path.join(REPO, m)) for m in P21_MODULES}
        assert T18.violations("x.py", "t = pd.read_csv(D + '/inventory_position_weekly.csv')\n"), "scanner cannot see the table name"
    finally:
        T18.FORBIDDEN_STR = saved
    bad = {m: v for m, v in hits.items() if v}
    assert not bad, bad
    return dict(phase21_modules_scanned=P21_MODULES, reads_of_inventory_position_weekly=0, reads_of_part_demand_weekly=0)


def test_poison_and_self_exclusion():
    import test_phase21_grpstats as TG
    import grpstats as GS
    PP.register()
    out = {}
    for w in ("v8", PP.WORLD2):
        src = GS.Source(w); smp = TG.sample(w)
        out[w] = dict(future_poison=TG.test_future_poison(src, smp), self_exclusion=TG.test_self_exclusion(src, smp))
    return out


if __name__ == "__main__":
    import phase12_common as C
    res = {"stamp": C.stamp()}
    for name in ("test_protected_paths_unchanged", "test_db_untouched", "test_identities", "test_no_privileged_input",
                 "test_no_inventory_position_weekly", "test_poison_and_self_exclusion"):
        res[name] = globals()[name]()
        print(f"  PASS {name}", flush=True)
    out = os.path.join(REPO, "ml", "artifacts", "phase21", "isolation_audit.json")
    json.dump(res, open(out, "w"), indent=1, default=str)
    r = res["test_identities"]
    print(f"  stored configs recomputed identically: {r['recomputed_identical']} (named by this base {r['named_by_this_base']}; "
          f"cannot be named {len(r['cannot_be_named'])}); Phase 19 bundles named correctly: {r['phase19_bundles_named_correctly']}")
