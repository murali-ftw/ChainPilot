"""Phase 24 isolation audit -- each check with a way to fail. New file.

1  PROTECTED PATHS: `git diff <base> HEAD` and the working tree are empty on every file that EXISTED at the base under ml/,
   results/, db/, docs/specs/, reports/ (+ ml/configs/shipped.json); only additions. FIRES: the same check over the
   Phase 23AC range (545e282..e5ecb3f, which added results/observation3.md and modified nothing) must list files ADDED there,
   and a constructed "modified" list must be refused.
2  db/ UNTOUCHED, 3 NO STORED ARTIFACT MODIFIED, 4 STORED IDENTITIES: Phase 23AC's own checks, imported unchanged
   (test_phase23ac_isolation.test_db_untouched / test_no_stored_artifact_modified / test_identities). Phase 24 bundles (Stage 6)
   are checked to sit at artifact_identity.bundle_name(cfg) under ml/artifacts/phase24/bundles.
5  AST SCAN of the Phase 24 modules: every inventory_position_weekly / part_demand_weekly name outside docstrings is listed;
   none may be in ml/serve or ml/models (Phase 24 adds none there). SELF-TEST: a constructed offender is flagged.
6  LEAK GUARD: scripts/run_leak_guard.sh exits 0.
7  NO TRAINING: no Phase 24 module calls an optimiser step, .fit(, lgb.train( or .backward( -- except, at the Stage 7 audit
   (--stage 7), the declared Stage 6 trainer ml/train/phase24_b1b_clean.py. A constructed offender MUST be flagged.

    python ml/tests/test_phase24_isolation.py [--stage 5|7]
"""
from __future__ import annotations
import os, sys, ast, json, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train"),
                os.path.join(HERE, "..", "eval")]
import test_phase23ac_isolation as I23

BASE = "e5ecb3f"
STAGE6_TRAINER = "ml/train/phase24_b1b_clean.py"
TABLES = ("inventory_position_weekly", "part_demand_weekly")


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True)


def protected_diff(base, head):
    existing = [p for p in git("ls-tree", "-r", "--name-only", base).split()
                if p.startswith(("ml/", "results/", "db/", "docs/specs/", "reports/")) or p == "ml/configs/shipped.json"]
    changed = git("diff", "--name-only", base, head, "--", *existing).split()
    added = git("diff", "--name-only", "--diff-filter=A", base, head).split()
    return existing, changed, added


def refuse_changes(changed, modified):
    if changed:
        raise AssertionError(f"protected files changed: {changed}")
    if modified:
        raise AssertionError(f"existing files modified/deleted/renamed: {modified}")


def test_protected_paths_unchanged():
    existing, changed, added = protected_diff(BASE, "HEAD")
    assert existing, "no protected files listed -- the check would pass vacuously"
    mod = git("diff", "--name-only", "--diff-filter=MDRT", BASE, "HEAD").split()
    refuse_changes(changed, mod)
    dirty = git("status", "--porcelain", "--", *existing).strip()
    assert not dirty, f"a protected file is modified in the working tree: {dirty[:300]}"
    # failing cases: (i) the diff machinery sees additions on a known range; (ii) the refusal fires on a range that MODIFIED
    # a protected file (e7163e4^1..e7163e4 modified results/observation1.md)
    _, ch23, add23 = protected_diff("545e282", "e5ecb3f")
    assert "results/observation3.md" in add23 and not ch23, "the diff machinery does not see Phase 23AC's additions"
    _, ch22, _ = protected_diff("e7163e4^1", "e7163e4")
    try:
        refuse_changes(ch22, []); fired = False
    except AssertionError:
        fired = True
    assert fired, "the protected-path refusal cannot fail"
    return dict(base=BASE, protected_files_checked=len(existing), committed_diff="empty", worktree="clean", files_added=added,
                files_modified=mod, failing_case_known_range_additions_seen=True, failing_case_e7163e4_fires=fired,
                failing_case_files=len(ch22))


def string_hits(src):
    tree = ast.parse(src); docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            f = n.body[0]
            if isinstance(f, ast.Expr) and isinstance(f.value, ast.Constant) and isinstance(f.value.value, str):
                docs.add(id(f.value))
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            hits += [dict(table=t, line=n.lineno) for t in TABLES if t in n.value]
        if isinstance(n, ast.JoinedStr):
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    hits += [dict(table=t, line=n.lineno) for t in TABLES if t in v.value]
    return hits


def phase24_modules():
    return sorted(p for p in git("diff", "--name-only", "--diff-filter=A", BASE, "HEAD").split() if p.endswith(".py"))


def test_ast_scan():
    assert string_hits("x = pd.read_csv(D + '/inventory_position_weekly.csv')\n"), "the scanner cannot see a constructed offender"
    assert string_hits("x = pd.read_csv(f'{D}/part_demand_weekly.csv')\n"), "the scanner cannot see an f-string offender"
    assert not string_hits('"""reads inventory_position_weekly"""\n'), "a docstring was counted as a read"
    mods = phase24_modules()
    hits = {m: string_hits(open(os.path.join(REPO, m)).read()) for m in mods}
    bad = {m: h for m, h in hits.items() if h and m.startswith(("ml/serve/", "ml/models/"))}
    assert not bad, f"model / serve code names a forbidden table: {bad}"
    return dict(modules_scanned=mods, reads={m: h for m, h in hits.items() if h}, serve_or_model_reads=0, offenders_flagged=2)


def test_leak_guard():
    p = subprocess.run(["bash", os.path.join(REPO, "scripts", "run_leak_guard.sh")], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, f"leak guard exit {p.returncode}: {p.stdout[-500:]}"
    return dict(exit=p.returncode, tail=[l for l in p.stdout.splitlines() if l.strip()][-2:])


def test_no_training(stage):
    offenders = ["opt.step()\n", "m.fit(X, y)\n", "lgb.train(params, ds)\n", "loss.backward()\n"]
    assert all(I23.training_calls(o) for o in offenders), "the no-training scan cannot see a constructed offender"
    mods = phase24_modules()
    hits = {m: I23.training_calls(open(os.path.join(REPO, m)).read()) for m in mods}
    allowed = {STAGE6_TRAINER} if stage == 7 else set()
    bad = {m: v for m, v in hits.items() if v and m not in allowed}
    assert not bad, f"training calls in Phase 24 modules: {bad}"
    return dict(stage=stage, modules_scanned=mods, offenders_flagged=len(offenders),
                declared_trainer=sorted(m for m in hits if hits[m] and m in allowed), training_calls_elsewhere=0)


def test_phase24_bundles():
    import artifact_identity as AI
    root = os.path.join(REPO, "ml", "artifacts", "phase24", "bundles", "rescue_week")
    if not os.path.isdir(root):
        return dict(bundles=0, note="no Phase 24 bundle yet")
    out = []
    for d in sorted(os.listdir(root)):
        cfg = json.load(open(os.path.join(root, d, "config.json")))
        assert d == AI.bundle_name(cfg), f"{d} is not artifact_identity.bundle_name ({AI.bundle_name(cfg)})"
        assert cfg["world"] == "v8clean" and cfg["trained_by"] == STAGE6_TRAINER
        out.append(d)
    return dict(bundles=len(out), names=out)


if __name__ == "__main__":
    import phase12_common as C
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=5, choices=[5, 7])
    a = ap.parse_args()
    res = {"stamp": C.stamp(), "stage": a.stage}
    for name, fn in (("test_protected_paths_unchanged", test_protected_paths_unchanged),
                     ("test_db_untouched", I23.test_db_untouched),
                     ("test_no_stored_artifact_modified", I23.test_no_stored_artifact_modified),
                     ("test_identities", I23.test_identities), ("test_phase24_bundles", test_phase24_bundles),
                     ("test_ast_scan", test_ast_scan), ("test_leak_guard", test_leak_guard),
                     ("test_no_training", lambda: test_no_training(a.stage))):
        res[name] = fn()
        print(f"  PASS {name}", flush=True)
    os.makedirs(os.path.join(REPO, "ml", "artifacts", "phase24"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "ml", "artifacts", "phase24", f"isolation_audit_stage{a.stage}.json"), "w"), indent=1, default=str)
    r = res["test_identities"]
    print(f"  stored configs recomputed identically: {r['recomputed_identical']}; Phase 19 bundles {r['phase19_bundles_named_correctly']}; "
          f"Phase 22 bundles {r['phase22_bundles_named_correctly']}; Phase 24 bundles {res['test_phase24_bundles']['bundles']}")
