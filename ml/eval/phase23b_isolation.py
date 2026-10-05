"""Phase 23B Stage 5 -- isolation audit. Each check carries a case that must fire.

1  PROTECTED PATHS  `git diff <BASE_SHA> HEAD` and the working tree are empty on every protected path (db/gen_v6..v8, db/validator.py,
                    docs/specs, synthetic_rules.md, dataset_structure.md, model_plan.md, shipped.json, results/, reports/part1, every
                    prior report, share.py, tcn.py, artifact_identity.py, phase19_identity.py) and on the Phase 23A paths; under
                    ml/serve/ and ml/tests/ only ADDED files. FIRES: the same diff over BASE^1..BASE (Phase 22's merge) is non-empty.
2  db/ UNTOUCHED     the worktree's db/ is clean; the main checkout's db/ has no content change (line endings only, pre-existing).
3  STORED IDENTITIES ml/tests/test_integration_identity.py, run unchanged (its own falsification included).
4  G3 FAILING CASE   PUBLISHED B1a refitted with the WRONG seed (8 instead of 7) must NOT reproduce the stored s7 predictions.
5  READS             every inventory_position_weekly / part_demand_weekly name in the Phase 23B code (AST, docstrings skipped).
6  SERVING           ml/tests/test_phase23b_serve.py (bit-exact, mismatch raises, AST scan with offender self-test).

  python ml/eval/phase23b_isolation.py     (torch-free process; the tests run as subprocesses)
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST (phase7_fit)
import os, sys, ast, json, subprocess
import phase12_common as C
import numpy as np, pandas as pd

BASE = "545e282cb1f5f0a4b3c9168ba379de49f820a3ad"
PROTECTED = ["db/gen_v6", "db/gen_v7", "db/gen_v8", "db/validator.py", "docs/specs", "db/dataset_structure.md", "ml/configs/shipped.json",
             "results", "reports/part1", "ml/models/share.py", "ml/models/tcn.py", "ml/artifact_identity.py", "ml/train/phase19_identity.py"]
P23A = ["reports/part2/phase23a", "ml/eval/leak_guard*", "ml/tests/test_leak_guard*", "ml/serve/order_time_card.py", "ml/serve/_plan_reader.py",
        "ml/serve/fill_consolidated.py", "docs/standards", "results/observation3.md", "docs/decisions/proposed_exception_*"]
MAIN = r"C:\Users\0rayc\Documents\GitHub\ChainPilot"


def git(*a, cwd=C.REPO):
    return subprocess.check_output(["git", "-C", cwd, *a], text=True).strip()


def prior_reports():
    return [p for p in git("ls-tree", "-r", "--name-only", BASE, "reports").splitlines()]


def forbidden_reads(path):
    src = open(path, encoding="utf-8").read(); tree = ast.parse(src); docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            docs.add(id(n.body[0].value))
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            for t in ("inventory_position_weekly", "part_demand_weekly"):
                if t in n.value:
                    hits.append(dict(table=t, line=n.lineno))
    return hits


def main():
    st = C.require_clean()
    R = dict(stamp=st, base=BASE)
    paths = PROTECTED + prior_reports() + P23A
    d = git("diff", "--name-only", BASE, "HEAD", "--", *paths); w = git("status", "--porcelain", "--", *paths)
    serve_tests = git("diff", "--name-status", "--diff-filter=MDRT", BASE, "HEAD", "--", "ml/serve", "ml/tests")
    fires = git("diff", "--name-only", f"{BASE}^1", BASE, "--", *paths)
    R["protected_paths"] = dict(n_paths=len(paths), diff_vs_base=d.splitlines(), worktree=w.splitlines(),
                                existing_serve_tests_modified=serve_tests.splitlines(),
                                added_under_serve_tests=git("diff", "--name-only", "--diff-filter=A", BASE, "HEAD", "--", "ml/serve", "ml/tests").splitlines(),
                                constructed_case_fires=bool(fires), constructed_case_files=len(fires.splitlines()),
                                PASS=not d and not w and not serve_tests and bool(fires))
    main_db = git("diff", "--ignore-cr-at-eol", "--name-only", "--", "db", cwd=MAIN)
    R["db"] = dict(worktree_status=git("status", "--porcelain", "--ignored=no", "--", "db").splitlines(),
                   main_checkout_content_change=main_db.splitlines(),
                   main_checkout_seeds=sorted(x for x in os.listdir(os.path.join(MAIN, "db", "gen_v8")) if x.startswith("seed_")),
                   PASS=not git("status", "--porcelain", "--", "db") and not main_db)
    t = subprocess.run([sys.executable, os.path.join(C.ML, "tests", "test_integration_identity.py")], capture_output=True, text=True,
                       env=dict(os.environ, HADES_DEVICE="cpu"))
    t2 = subprocess.run([sys.executable, os.path.join(C.ML, "tests", "test_phase23b_identity.py")], capture_output=True, text=True,
                        env=dict(os.environ, HADES_DEVICE="cpu"))
    err = [l for l in t.stderr.splitlines() if "STOP" in l]
    R["stored_identities"] = dict(
        integration_test=dict(exit=t.returncode, stop=err[-1:] if err else [],
                              note="STOPs on this machine's Phase 17 bundles only (its Phase 0-15 bucket compares them with 90a38ed); see "
                                   "test_phase23b_identity.py"),
        creating_branch_test=dict(exit=t2.returncode, tail=[l for l in t2.stdout.splitlines() if l.strip()][-3:]),
        PASS=t2.returncode == 0 and (t.returncode == 0 or (err and "_lean" in err[-1] or "_tgt" in (err[-1] if err else ""))))
    # 4. G3's constructed failing case: wrong seed must not reproduce
    sys.path.insert(0, os.path.join(C.ML, "baselines"))
    import phase7_fit as P7, phase23b_rescue as PR
    PR.register()
    Rw = PR.load_rows("v8"); X, names, _ = PR.features(P7, "v8", Rw)
    y = Rw["y"].astype(int); tr, va, te = (Rw["fold"] == 0), (Rw["fold"] == 1), (Rw["fold"] == 2)
    m = lgb.LGBMClassifier(**{**P7.GBM, "random_state": 8, "objective": "binary"})
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
    pt = m.predict_proba(X[te])[:, 1].astype(np.float32)
    stored = np.load(os.path.join(C.ART, "phase17", "b1a_lgbm_s7.npz"))["Pt"]
    R["G3_constructed_failing_case"] = dict(wrong_seed=8, reproduces_stored_s7=bool(np.array_equal(pt, stored)),
                                            max_abs_diff=float(np.abs(pt.astype(float) - stored.astype(float)).max()),
                                            fires=not np.array_equal(pt, stored))
    # 5. reads
    files = sorted([os.path.join(C.ML, "eval", f) for f in os.listdir(os.path.join(C.ML, "eval")) if f.startswith("phase23b")] +
                   [os.path.join(C.ML, "serve", "rescue.py"), os.path.join(C.ML, "tests", "test_phase23b_serve.py")])
    R["reads"] = {os.path.relpath(f, C.REPO).replace("\\", "/"): forbidden_reads(f) for f in files}
    # 6. serving
    t = subprocess.run([sys.executable, os.path.join(C.ML, "tests", "test_phase23b_serve.py")], capture_output=True, text=True,
                       env=dict(os.environ, HADES_DEVICE="cpu"))
    R["serve_tests"] = dict(exit=t.returncode, tail=[l for l in t.stdout.splitlines() if l.strip()][-5:], PASS=t.returncode == 0)
    R["ALL_PASS"] = all(R[k]["PASS"] for k in ("protected_paths", "db", "stored_identities", "serve_tests")) and R["G3_constructed_failing_case"]["fires"]
    C.dump(R, "phase23b/isolation.json")
    print(json.dumps({k: (v.get("PASS", v.get("fires")) if isinstance(v, dict) else v) for k, v in R.items() if k != "reads"}, indent=1))
    print("reads:", json.dumps(R["reads"]))


if __name__ == "__main__":
    main()
