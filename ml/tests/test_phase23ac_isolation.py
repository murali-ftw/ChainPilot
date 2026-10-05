"""Phase 23AC isolation audit (final step) -- each check with a way to fail.

1  PROTECTED PATHS: `git diff <base> HEAD` and the working tree are empty on every EXISTING file under ml/ and results/ at the
   base, plus db/, docs/specs, every report at the base, ml/configs/shipped.json (new files are allowed). The Phase 23B paths
   (reports/part2/phase23b/, ml/eval/phase23b_*, ml/tests/test_phase23b_*, ml/serve/rescue*.py) are not written by this phase.
2  db/ UNTOUCHED: no tracked change, no untracked file under db/; db/gen_v8 holds exactly seed_1001..seed_1005; world 2's
   recorded hashes unchanged.
3  NO STORED ARTIFACT MODIFIED: every file hashed in ml/artifacts/phase23ac/stored_manifest_before.json (taken before any
   measurement) has the same SHA-1 now.
4  STORED IDENTITIES: every stored config recomputes identically (Phase 18's check, base = this phase's base); every Phase 19
   and Phase 22 bundle sits at its phase19_identity name.
5  AST SCANS over ml/train, ml/models, ml/data, ml/baselines, ml/serve: no PRIVILEGED path (constructed offender flagged);
   inventory_position_weekly in no new module; part_demand_weekly only in ml/serve/_plan_reader.py among the new modules.
6  NO TRAINING: no new module calls an optimizer step, .fit(, lgb.train( or .backward(; a constructed offender MUST be flagged.

    python ml/tests/test_phase23ac_isolation.py
"""
from __future__ import annotations
import os, sys, ast, json, glob, hashlib, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train"),
                os.path.join(HERE, "..", "eval")]
import test_phase18_isolation as T18
import phase21_paths as PP

BASE = "545e282"
NEW_MODULES = ["ml/serve/order_time_card.py", "ml/serve/_plan_reader.py", "ml/serve/fill_consolidated.py",
               "ml/eval/phase23ac_t1_compare.py", "ml/eval/phase23ac_t2_serve.py", "ml/eval/leak_guard.py",
               "ml/eval/phase23ac_t4_metrics.py", "ml/eval/phase23ac_t4_charts_data.py", "ml/eval/phase23ac_t4_charts.py"]
TRAIN_CALLS = ("step", "fit", "train", "backward", "zero_grad")


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True)


def test_protected_paths_unchanged():
    existing = [p for p in git("ls-tree", "-r", "--name-only", BASE).split()
                if p.startswith(("ml/", "results/", "db/", "docs/specs/", "reports/")) or p == "ml/configs/shipped.json"]
    assert existing, "no protected files listed -- the check would pass vacuously"
    assert not git("diff", "--stat", BASE, "HEAD", "--", *existing).strip(), "a protected file changed in a commit"
    dirty = git("status", "--porcelain", "--", *existing).strip()
    assert not dirty, f"a protected file is modified in the working tree: {dirty[:300]}"
    p23b = [p for p in git("diff", "--name-only", BASE, "HEAD").split()
            if p.startswith(("reports/part2/phase23b/", "ml/eval/phase23b_", "ml/tests/test_phase23b_", "ml/serve/rescue"))]
    assert not p23b, f"Phase 23B paths written by this phase: {p23b}"
    return dict(base=BASE, protected_files_checked=len(existing), committed_diff="empty", worktree="clean",
                files_added=git("diff", "--name-only", "--diff-filter=A", BASE, "HEAD").split(),
                files_modified=git("diff", "--name-only", "--diff-filter=M", BASE, "HEAD").split())


def test_db_untouched():
    assert not git("status", "--porcelain", "--untracked-files=all", "--", "db").strip(), "db/ changed"
    seeds = sorted(d for d in os.listdir(os.path.join(REPO, "db", "gen_v8")) if d.startswith("seed_"))
    assert seeds == [f"seed_{s}" for s in (1001, 1002, 1003, 1004, 1005)], seeds
    PP.register()
    rec = json.load(open(os.path.join(REPO, "ml", "artifacts", "phase20", "world2_generation.json")))["key_tables"]
    now = PP.world_hashes(PP.WORLD2, tables=("po_lines.csv", "grn_lines.csv", "training_labels.csv", "sourcing_channels.csv"))
    assert all(now[t] == rec[t]["sha1"] for t in now)
    return dict(db="clean", gen_v8_seed_dirs=seeds, world2_hashes_equal_phase20=list(now))


def test_no_stored_artifact_modified():
    man = json.load(open(os.path.join(REPO, "ml", "artifacts", "phase23ac", "stored_manifest_before.json")))
    changed = []
    for p, rec in man["files"].items():
        h = hashlib.sha1()
        with open(os.path.join(REPO, p), "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        if h.hexdigest() != rec["sha1"]:
            changed.append(p)
    assert len(man["files"]) > 100 and not changed, f"stored artifacts changed: {changed[:5]}"
    return dict(files_hashed_before=len(man["files"]), taken=man["taken"], changed=0)


def test_identities():
    T18.BASE = BASE
    r = T18.test_every_stored_identity_recomputes()
    import phase19_identity as PI
    for root, key in ((os.path.join(PI.BUNDLES19), "phase19"), (os.path.join(REPO, "ml", "artifacts", "phase22", "bundles"), "phase22")):
        cfgs = sorted(glob.glob(os.path.join(root, "*", "*", "config.json")))
        for f in cfgs:
            assert os.path.basename(os.path.dirname(f)) == PI.bundle_name(json.load(open(f))), f
        r[f"{key}_bundles_named_correctly"] = len(cfgs)
    return r


def test_ast_scans():
    T18.SCANNED = list(T18.SCANNED) + ["ml/serve"]
    T18.FORBIDDEN_STR = T18.FORBIDDEN_STR + ("PRIVILEGED__", "phase19/PRIVILEGED", "phase20/PRIVILEGED", "phase22/PRIVILEGED",
                                             "phase23ac/PRIVILEGED")
    r = T18.test_no_privileged_input_in_ml()
    assert T18.violations("x.py", "p = 'reports/part2/phase23ac/PRIVILEGED__x.npz'\n")
    saved = T18.FORBIDDEN_STR
    out = {}
    try:
        for tbl, allowed in (("inventory_position_weekly", set()), ("part_demand_weekly", {"ml/serve/_plan_reader.py"})):
            T18.FORBIDDEN_STR = (tbl,)
            hits = {m: T18.violations(os.path.join(REPO, m)) for m in NEW_MODULES if os.path.exists(os.path.join(REPO, m))}
            bad = {m: v for m, v in hits.items() if v and m not in allowed}
            assert not bad, f"{tbl} named by {bad}"
            assert T18.violations("x.py", f"x = pd.read_csv(D + '/{tbl}.csv')\n"), "the scanner cannot see the table"
            out[tbl] = dict(readers=sorted(m for m, v in hits.items() if v), allowed=sorted(allowed))
    finally:
        T18.FORBIDDEN_STR = saved
    r["table_reads_in_new_modules"] = out
    return r


def training_calls(src):
    """-> calls that train: <x>.step / .fit / lgb.train / .backward / .zero_grad (attribute calls on any object)."""
    tree = ast.parse(src); bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in TRAIN_CALLS:
            base = node.func.value
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "?")
            if node.func.attr == "train" and name not in ("lgb", "lightgbm", "L", "loop"):
                continue                      # pandas / numpy objects do not have .train; only these names train a model
            if node.func.attr == "step" and name in ("np", "pd", "range"):
                continue
            bad.append(f"{name}.{node.func.attr}")
    return bad


def test_no_training():
    offenders = ["opt.step()\n", "m.fit(X, y)\n", "lgb.train(params, ds)\n", "loss.backward()\n"]
    assert all(training_calls(o) for o in offenders), "the no-training scan cannot see a constructed offender"
    hits = {m: training_calls(open(os.path.join(REPO, m)).read()) for m in NEW_MODULES if os.path.exists(os.path.join(REPO, m))}
    bad = {m: v for m, v in hits.items() if v}
    assert not bad, f"training calls in new modules: {bad}"
    return dict(modules_scanned=sorted(hits), offenders_flagged=len(offenders), training_calls=0)


if __name__ == "__main__":
    import phase12_common as C
    res = {"stamp": C.stamp()}
    for name in ("test_protected_paths_unchanged", "test_db_untouched", "test_no_stored_artifact_modified", "test_identities",
                 "test_ast_scans", "test_no_training"):
        res[name] = globals()[name]()
        print(f"  PASS {name}", flush=True)
    os.makedirs(os.path.join(REPO, "ml", "artifacts", "phase23ac"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "ml", "artifacts", "phase23ac", "isolation_audit.json"), "w"), indent=1, default=str)
    r = res["test_identities"]
    print(f"  stored configs recomputed identically: {r['recomputed_identical']} (named by this base {r['named_by_this_base']}); "
          f"Phase 19 bundles {r['phase19_bundles_named_correctly']}; Phase 22 bundles {r['phase22_bundles_named_correctly']}")
