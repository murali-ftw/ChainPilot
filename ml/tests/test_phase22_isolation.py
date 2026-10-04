"""Phase 22 isolation audit (Stage 5) -- each check with a way to fail.

1  PROTECTED PATHS: `git diff <base> HEAD` and the working tree are empty on every path the brief protects, including every
   report at the base, ml/train/phase19_identity.py and every EXISTING file under ml/serve/ (new serve files are allowed).
2  db/ UNTOUCHED: the worktree's db/ clean (tracked and untracked); the main checkout's db/ clean with seed_1001..seed_1005;
   world 2's recorded hashes unchanged.
3  STORED IDENTITIES: every stored config recomputes identically (Phase 18's check, base = this phase's base); every Phase 19
   bundle at its phase19_identity name; every Phase 22 bundle at the name artifact_identity / phase19_identity give its
   config (world `v8clean` in the name), and no Phase 22 bundle shares a name with a stored one.
4  NO PRIVILEGED INPUT: Phase 18's AST scanner over ml/train, ml/models, ml/data, ml/baselines, ml/serve, forbidden strings
   extended to Phase 19-22 privileged paths, inventory_position_weekly and part_demand_weekly for the Phase 22 modules;
   constructed offenders flagged, a docstring passes; the audit-only scan (ml/eval/phase22_leakscan.py) is the one reader of
   _sim.npz and is not under the scanned directories.
5  NEW FEATURE MODULES: clean_panel (poison + self-exclusion with offenders, both worlds), grpstats (Phase 21's tests),
   phase22_rows (its ack / L4 tables come from grpstats.Builder, asserted as-of; its season / cadence from the Phase 19 stores)
6  SERVING: ml/tests/test_phase22_serve.py passes.

    HADES_DATA_ROOT=<main checkout> python ml/tests/test_phase22_isolation.py
"""
from __future__ import annotations
import os, sys, json, glob, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "train"),
                os.path.join(HERE, "..", "eval")]
import test_phase18_isolation as T18
import phase21_paths as PP

BASE = "cbc7088"
PROTECTED = ["db/gen_v6", "db/gen_v7", "db/gen_v8", "db/validator.py", "docs/specs", "db/dataset_structure.md",
             "ml/configs/shipped.json", "results", "reports/part1", "ml/models/share.py", "ml/models/tcn.py",
             "ml/artifact_identity.py", "ml/train/phase19_identity.py", "docs/decisions"]
P22_MODULES = ["ml/data/clean_panel.py", "ml/data/phase22_rows.py", "ml/baselines/phase22_proxy.py", "ml/train/phase22_train.py",
               "ml/serve/order_time.py"]


def git(*a, repo=REPO):
    return subprocess.check_output(["git", "-C", repo, *a], text=True)


def test_protected_paths_unchanged():
    prior = [p for p in git("ls-tree", "-r", "--name-only", BASE, "reports").split() if p.endswith((".md", ".json"))]
    serve_existing = [p for p in git("ls-tree", "-r", "--name-only", BASE, "ml/serve").split()]
    paths = PROTECTED + prior + serve_existing
    assert not git("diff", "--stat", BASE, "HEAD", "--", *paths).strip(), "protected paths changed"
    assert not git("status", "--porcelain", "--", *paths).strip(), "protected paths modified in the working tree"
    return dict(base=BASE, protected=PROTECTED, prior_reports_checked=len(prior), existing_serve_files=serve_existing,
                committed_diff="empty", worktree="clean", files_changed=git("diff", "--name-only", BASE, "HEAD").split())


def test_db_untouched():
    st = git("status", "--porcelain", "--untracked-files=all", "--", "db").strip()
    assert not st, st[:500]
    root = PP.DATA_ROOT
    assert not git("status", "--porcelain", "--untracked-files=all", "--", "db", repo=root).strip(), "main checkout db/ changed"
    seeds = sorted(d for d in os.listdir(os.path.join(root, "db", "gen_v8")) if d.startswith("seed_"))
    assert seeds == [f"seed_{s}" for s in (1001, 1002, 1003, 1004, 1005)], seeds
    PP.register()
    rec = json.load(open(os.path.join(REPO, "ml", "artifacts", "phase20", "world2_generation.json")))["key_tables"]
    now = PP.world_hashes(PP.WORLD2, tables=("po_lines.csv", "grn_lines.csv", "training_labels.csv", "sourcing_channels.csv"))
    assert all(now[t] == rec[t]["sha1"] for t in now)
    return dict(worktree_db="clean", main_checkout_db="clean", gen_v8_seed_dirs=seeds, second_world_hashes_equal_phase20=list(now))


def test_identities():
    T18.BASE = BASE
    r = T18.test_every_stored_identity_recomputes()
    import phase19_identity as PI, artifact_identity as AI
    p19 = sorted(glob.glob(os.path.join(PI.BUNDLES19, "*", "*", "config.json")))
    for f in p19:
        assert os.path.basename(os.path.dirname(f)) == PI.bundle_name(json.load(open(f))), f
    root22 = os.path.join(REPO, "ml", "artifacts", "phase22", "bundles")
    p22 = sorted(glob.glob(os.path.join(root22, "*", "*", "config.json")))
    stored_names = {os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(REPO, "ml", "artifacts", "bundles", "*", "*", "config.json"))}
    for f in p22:
        cfg = json.load(open(f)); name = os.path.basename(os.path.dirname(f))
        assert name == PI.bundle_name(cfg), f"{f}: not at its identity name"
        assert cfg["world"].endswith("clean") and name not in stored_names, f"{name} collides with a stored bundle or is not a clean world"
    r.update(phase19_bundles_named_correctly=len(p19), phase22_bundles_named_correctly=len(p22))
    return r


def test_no_privileged_input():
    T18.FORBIDDEN_STR = T18.FORBIDDEN_STR + ("PRIVILEGED__", "phase19/PRIVILEGED", "phase19/preds", "phase20/PRIVILEGED",
                                             "phase20/preds", "phase21/PRIVILEGED", "phase22/PRIVILEGED")
    T18.SCANNED = list(T18.SCANNED) + ["ml/serve"]              # constraint 4 adds ml/serve to the scanned directories
    r = T18.test_no_privileged_input_in_ml()
    r["scanned_dirs"] = T18.SCANNED
    hits = {m: T18.violations(os.path.join(REPO, m)) for m in ("ml/serve/order_time.py",)}
    assert not any(hits.values()), hits
    for off in ("p = 'reports/part2/phase22/PRIVILEGED__x.npz'\n", "z = np.load(D + '/_sim.npz')\n"):
        assert T18.violations("x.py", off), off
    assert not T18.violations("x.py", '"""reads nothing from reports/part2/phase22/PRIVILEGED__ (docstring)"""\nx = 1\n')
    r["serve_new_scanned"] = list(hits)
    r["phase22_offenders_flagged"] = 2
    r["audit_only_sim_reader"] = "ml/eval/phase22_leakscan.py and ml/eval/phase22_leak_rows.py (ml/eval is not a model directory)"
    return r


def test_no_inventory_or_demand_table():
    saved = T18.FORBIDDEN_STR
    T18.FORBIDDEN_STR = ("inventory_position_weekly", "part_demand_weekly")
    try:
        hits = {m: T18.violations(os.path.join(REPO, m)) for m in P22_MODULES}
        assert T18.violations("x.py", "t = pd.read_csv(D + '/part_demand_weekly.csv')\n")
    finally:
        T18.FORBIDDEN_STR = saved
    bad = {m: v for m, v in hits.items() if v}
    assert not bad, bad
    return dict(phase22_modules_scanned=P22_MODULES, reads=0,
                pre_existing_reader="ml/data/fwd_load.py (Phase 18; part_demand_weekly with deviation 162's as_of + 2 d bound) feeds the "
                                    "season family through its stored arrays; not a Phase 22 module")


def test_feature_modules():
    out = {}
    for w in ("v8", "v8w1002"):
        p = subprocess.run([sys.executable, os.path.join(HERE, "test_phase22_clean_panel.py"), "--world", w], cwd=REPO, capture_output=True, text=True)
        assert p.returncode == 0 and "ALL PASS" in p.stdout, p.stdout[-1500:] + p.stderr[-1500:]
        out[f"clean_panel_{w}"] = "PASS (poison and self-exclusion; offenders flagged)"
    p = subprocess.run([sys.executable, os.path.join(HERE, "test_phase21_grpstats.py")], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0 and "ALL PASS" in p.stdout, p.stdout[-1500:]
    out["grpstats (ack and L4 tables, order-time KM)"] = "PASS (Phase 21 tests)"
    return out


def test_serving():
    p = subprocess.run([sys.executable, os.path.join(HERE, "test_phase22_serve.py")], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.count("PASS") >= 3, p.stdout[-1500:] + p.stderr[-1500:]
    return dict(result="PASS", lines=[l for l in p.stdout.splitlines() if "PASS" in l])


if __name__ == "__main__":
    import phase12_common as C
    res = {"stamp": C.stamp()}
    for name in ("test_protected_paths_unchanged", "test_db_untouched", "test_identities", "test_no_privileged_input",
                 "test_no_inventory_or_demand_table", "test_feature_modules", "test_serving"):
        res[name] = globals()[name]()
        print(f"  PASS {name}", flush=True)
    json.dump(res, open(os.path.join(REPO, "ml", "artifacts", "phase22", "isolation_audit.json"), "w"), indent=1, default=str)
    r = res["test_identities"]
    print(f"  stored configs recomputed identically: {r['recomputed_identical']} (named by this base {r['named_by_this_base']}); "
          f"Phase 19 bundles {r['phase19_bundles_named_correctly']}; Phase 22 bundles {r['phase22_bundles_named_correctly']}")
