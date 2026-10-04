"""Phase 19 isolation audit (Stage 5) -- each check with a way to fail.

1  PROTECTED PATHS: `git diff <base> HEAD` and the working tree are empty on every path the Phase 19 brief protects.
2  STORED IDENTITIES: every stored bundle config recomputes to the same identity_of / bundle_path_key / config_name under
   artifact_identity at the base commit and now, AND phase19_identity.config_name names it identically (the row_family
   axis is omitted at its default). Configs the base cannot name (Phase 16 encoder variants) are listed with the reason.
   Falsification: a config WITH row_family gets a different name, and lands under the Phase 19 bundle root.
3  NO PRIVILEGED INPUT: no module under ml/train, ml/models, ml/data, ml/baselines imports from reports/ or names (outside
   a docstring) generator state or a PRIVILEGED__ file -- reusing Phase 18's AST scanner, which must still fire on
   constructed offenders. A fresh interpreter importing the Phase 19 feature, proxy and binding modules loads nothing
   from reports/.
4  FEATURE NAMES: no Phase 19 feature column carries the PRIVILEGED__ prefix.

    python ml/tests/test_phase19_isolation.py
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train")]
import artifact_identity as NEW
import test_phase18_isolation as T18

BASE = "62bb607"                          # phase18 tip, the Phase 19 base
PROTECTED = ["db/gen_v6", "db/gen_v7", "db/gen_v8", "db/validator.py", "docs/specs", "db/dataset_structure.md",
             "ml/configs/shipped.json", "results", "reports/part1", "reports/part2/phase-0-1-v8.md",
             "ml/models/share.py", "ml/models/tcn.py", "ml/artifact_identity.py"]
PRIOR_REPORTS = [p for p in subprocess.check_output(["git", "-C", REPO, "ls-tree", "-r", "--name-only", BASE, "reports"],
                                                     text=True).split() if p.endswith((".md", ".json"))]


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True)


def test_protected_paths_unchanged():
    paths = PROTECTED + PRIOR_REPORTS
    committed = git("diff", "--stat", BASE, "HEAD", "--", *paths).strip()
    worktree = git("status", "--porcelain", "--", *paths).strip()
    assert not committed, f"protected paths changed since {BASE}:\n{committed}"
    assert not worktree, f"protected paths modified in the working tree:\n{worktree}"
    assert git("diff", BASE, "HEAD", "--", "ml/models/share.py", "ml/models/tcn.py") == ""
    return dict(base=BASE, protected=PROTECTED, prior_reports_checked=len(PRIOR_REPORTS), committed_diff="empty", worktree="clean")


def test_every_stored_identity_recomputes():
    import phase19_identity as PI
    T18.BASE = BASE
    r = T18.test_every_stored_identity_recomputes()
    n = 0
    for f in T18.stored_configs():
        cfg = json.load(open(f))
        if "arch" not in cfg or "seed" not in cfg:
            continue
        cfg.setdefault("origin", None)
        assert PI.config_name(cfg) == NEW.config_name(cfg), f"phase19_identity renames a stored config: {f}"
        n += 1
    base = dict(task="capacity_strain", world="v8", origin=None, arch="mp", depth=4, lr=2.5e-4, seed=7)
    a, b = PI.bundle_name(base), PI.bundle_name(dict(base, row_family="fwdload"))
    assert a == NEW.bundle_name(base) and a != b, "row_family must be omitted at default and distinct when set"
    assert PI.bundle_dir(dict(base, row_family="fwdload")).startswith(PI.BUNDLES19)
    p19 = sorted(glob.glob(os.path.join(PI.BUNDLES19, "*", "*", "config.json")))
    for f in p19:
        cfg = json.load(open(f))
        assert os.path.basename(os.path.dirname(f)) == PI.bundle_name(cfg), f"Phase 19 bundle misnamed: {f}"
    r.update(phase19_identity_same_name_for_stored=n, row_family_axis="omitted at default; distinct when set",
             phase19_bundles_named_correctly=len(p19))
    return r


def test_no_privileged_input_in_ml():
    # Phase 19's privileged files: extend the Phase 18 scanner's forbidden strings, then re-run it (and its self-test)
    T18.FORBIDDEN_STR = T18.FORBIDDEN_STR + ("PRIVILEGED__", "phase19/PRIVILEGED", "phase19/preds")
    r = T18.test_no_privileged_input_in_ml()                     # Phase 18 scan + its falsification, extended strings
    for off in ("p = 'reports/part2/phase19/preds/v8_arrival_week_PRIVILEGED__timing_only_s7_test.npz'\n",
                "from reports.part2.phase19 import PRIVILEGED__timing\n",
                "x = np.load(os.path.join(R, 'PRIVILEGED__timing_scores.json'))\n"):
        assert T18.violations("x.py", off), f"scanner missed a Phase 19 offender: {off!r}"
    r["phase19_offenders_flagged"] = 3
    code = ("import sys; sys.path[:0]=['ml','ml/data','ml/baselines','ml/train','ml/eval'];"
            "import fwd_season, cadence, phase19_proxy;"
            "bad=[m for m,v in list(sys.modules.items()) if getattr(v,'__file__',None) and '/reports/' in v.__file__];"
            "assert not bad, bad; assert 'torch' not in sys.modules; print('ok')")
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    code2 = ("import sys; sys.path[:0]=['ml','ml/data','ml/train','ml/eval','ml/models'];"
             "import phase19_bind, phase19_identity;"
             "bad=[m for m,v in list(sys.modules.items()) if getattr(v,'__file__',None) and '/reports/' in v.__file__];"
             "assert not bad, bad; print('ok')")
    p2 = subprocess.run([sys.executable, "-c", code2], cwd=REPO, capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr[-2000:]
    r["phase19_runtime_import_check"] = "fwd_season, cadence, phase19_proxy (torch-free), phase19_bind, phase19_identity: nothing from reports/"
    return r


def test_feature_names_not_privileged():
    import fwd_season, cadence
    cols = list(fwd_season.COLS) + list(cadence.COLS)
    assert not [c for c in cols if c.startswith("PRIVILEGED")], cols
    return dict(feature_columns=len(cols))


if __name__ == "__main__":
    res = {}
    for name in ("test_protected_paths_unchanged", "test_every_stored_identity_recomputes", "test_no_privileged_input_in_ml",
                 "test_feature_names_not_privileged"):
        res[name] = globals()[name]()
        print(f"  PASS {name}")
    out = os.path.join(REPO, "ml", "artifacts", "phase19", "isolation_audit.json")
    json.dump(res, open(out, "w"), indent=1)
    r2 = res["test_every_stored_identity_recomputes"]
    print(f"  stored configs recomputed identically: {r2['recomputed_identical']} (named by this base {r2['named_by_this_base']}; "
          f"cannot be named {len(r2['cannot_be_named'])}); Phase 19 bundles named correctly: {r2['phase19_bundles_named_correctly']}")
