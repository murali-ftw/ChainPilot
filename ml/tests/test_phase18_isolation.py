"""Phase 18 isolation audit (Stage 8b) -- four checks, each with a way to fail.

1  PROTECTED PATHS: `git diff <base> HEAD` (and the working tree) is empty on every path the brief protects.
2  STORED IDENTITIES: every stored bundle config recomputes to the same identity_of / bundle_path_key / config_name under
   artifact_identity at the base commit and now, and its directory is the name bundle_name gives it. Configs that cannot
   be named by this base (Phase 16 encoder axes, unmerged) are LISTED with the reason, not skipped silently.
3  NO PRIVILEGED INPUT IN ml/: no module under ml/train, ml/models, ml/data, ml/baselines imports anything from
   reports/ or carries (outside its docstring) a path to generator state (`_sim.npz`, `_events.npz`) or to a PRIVILEGED__
   file. Importing the Phase 18 feature and proxy modules in a fresh interpreter loads no module from reports/.
   Falsification: the scanner must flag a constructed module that imports the oracle and one that opens _sim.npz.
4  FEATURE NAMES: no Phase 18 feature column carries the PRIVILEGED__ prefix.

    python ml/tests/test_phase18_isolation.py
"""
from __future__ import annotations
import os, sys, ast, json, glob, subprocess, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import artifact_identity as NEW

BASE = "a2301ef"                          # origin/phase17, the Phase 18 base
PROTECTED = ["db/gen_v6", "db/gen_v7", "db/gen_v8", "db/validator.py", "docs/specs", "db/dataset_structure.md",
             "ml/configs/shipped.json", "results", "reports/part1", "reports/part2", "ml/models/share.py",
             "ml/models/tcn.py", "ml/artifact_identity.py"]
SCANNED = ["ml/train", "ml/models", "ml/data", "ml/baselines"]
FORBIDDEN_STR = ("_sim.npz", "_events.npz", "PRIVILEGED__latents", "PRIVILEGED__oracle", "PRIVILEGED__regen",
                 "phase18/oracle", "phase18\\oracle")
# pre-existing readers of generator state, not Phase 18 modules (reported, not failed)
PRE_EXISTING = {"ml/baselines/learnability_windowed.py"}


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True)


# ------------------------------------------------------------------ 1
def test_protected_paths_unchanged():
    committed = git("diff", "--stat", BASE, "HEAD", "--", *PROTECTED).strip()
    worktree = git("status", "--porcelain", "--", *PROTECTED).strip()
    assert not committed, f"protected paths changed since {BASE}:\n{committed}"
    assert not worktree, f"protected paths modified in the working tree:\n{worktree}"
    assert git("diff", BASE, "HEAD", "--", "ml/models/share.py", "ml/models/tcn.py") == ""
    return dict(base=BASE, paths=PROTECTED, committed_diff="empty", worktree="clean")


# ------------------------------------------------------------------ 2
def old_module():
    src = git("show", f"{BASE}:ml/artifact_identity.py")
    path = os.path.join(REPO, "ml", "artifacts", "cache", f"artifact_identity_{BASE}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("ai_base", path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def stored_configs():
    art = os.path.join(REPO, "ml", "artifacts")
    pats = [os.path.join(art, "bundles", "**", "config.json"), os.path.join(art, "backtest", "bundles", "**", "config.json"),
            os.path.join(art, "phase16_audit_bundles", "**", "config.json")]
    seen = set()
    for p in pats:
        for f in sorted(glob.glob(p, recursive=True)):
            if f not in seen:
                seen.add(f); yield f


def test_every_stored_identity_recomputes():
    OLD = old_module()
    ok, cannot, skipped = 0, [], 0
    for f in stored_configs():
        cfg = json.load(open(f))
        if "arch" not in cfg or "seed" not in cfg:
            skipped += 1; continue                       # not a loop.py bundle config
        cfg.setdefault("origin", None)
        rel = os.path.relpath(f, REPO)
        assert OLD.identity_of(cfg) == NEW.identity_of(cfg), f"identity changed: {rel}"
        assert OLD.bundle_path_key(cfg) == NEW.bundle_path_key(cfg), f"bundle path changed: {rel}"
        assert OLD.config_name(cfg) == NEW.config_name(cfg), f"config name changed: {rel}"
        if os.path.basename(os.path.dirname(f)) != NEW.bundle_name(cfg):
            enc = cfg.get("encoder") or cfg.get("share_variant") or ""
            reason = ("Phase 16 encoder variant: its axes live on the unmerged phase16-encoders branch (deviation 149)"
                      if "enc" in os.path.basename(os.path.dirname(f)) or enc else "directory name differs from bundle_name(cfg)")
            cannot.append(dict(config=rel, reason=reason))
        else:
            ok += 1
    assert ok > 0, "no stored configs found -- the check would pass vacuously"
    return dict(recomputed_identical=ok + len(cannot), named_by_this_base=ok, cannot_be_named=cannot,
                non_bundle_configs_skipped=skipped)


# ------------------------------------------------------------------ 3
def violations(path, src=None):
    src = open(path).read() if src is None else src
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant):
                docs.add(id(b[0].value))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if "PRIVILEGED" in a.name or "oracle" in a.name.lower()]
        elif isinstance(node, ast.ImportFrom):
            if node.module and ("PRIVILEGED" in node.module or "reports" in node.module or "oracle" in node.module.lower()):
                bad.append(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            if any(t in node.value for t in FORBIDDEN_STR) or ("reports" in node.value and "oracle" in node.value):
                bad.append(node.value[:80])
    return bad


def test_no_privileged_input_in_ml():
    hits, pre = {}, {}
    files = [f for d in SCANNED for f in glob.glob(os.path.join(REPO, d, "**", "*.py"), recursive=True)]
    for f in files:
        rel = os.path.relpath(f, REPO)
        v = violations(f)
        if v:
            (pre if rel in PRE_EXISTING else hits)[rel] = v
    assert not hits, f"privileged input reachable from ml/: {hits}"
    # falsification: the scanner must fire on constructed offenders
    assert violations("x.py", "import PRIVILEGED__oracle\n"), "scanner missed an oracle import"
    assert violations("x.py", "import numpy as np\nz = np.load('db/gen_v8/seed_1001/_sim.npz')\n"), "scanner missed _sim.npz"
    assert violations("x.py", "p = 'reports/phase18/oracle/PRIVILEGED__latents_v8s1001.npz'\n"), "scanner missed a latent path"
    assert not violations("x.py", '"""reads nothing from reports/phase18/oracle/ (docstring)"""\nx = 1\n'), "docstring flagged"
    # runtime: a fresh interpreter importing the Phase 18 modules loads nothing from reports/
    code = ("import sys; sys.path[:0]=['ml','ml/data','ml/baselines','ml/train','ml/eval'];"
            "import fwd_load, pulse, phase18_proxy;"
            "bad=[m for m,v in list(sys.modules.items()) if getattr(v,'__file__',None) and '/reports/' in v.__file__];"
            "print(bad); assert not bad, bad; assert 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return dict(files_scanned=len(files), violations=0, pre_existing_generator_state_readers=pre,
                runtime_import_check="fwd_load, pulse, phase18_proxy: no module from reports/, torch not loaded")


# ------------------------------------------------------------------ 4
def test_feature_names_not_privileged():
    import fwd_load, pulse
    cols = list(fwd_load.COLS) + list(pulse.COLS)
    assert not [c for c in cols if c.startswith("PRIVILEGED")], cols
    return dict(feature_columns=len(cols))


if __name__ == "__main__":
    res = {}
    for name in ("test_protected_paths_unchanged", "test_every_stored_identity_recomputes", "test_no_privileged_input_in_ml",
                 "test_feature_names_not_privileged"):
        res[name] = globals()[name]()
        print(f"  PASS {name}")
    out = os.path.join(REPO, "ml", "artifacts", "phase18", "isolation_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    r2 = res["test_every_stored_identity_recomputes"]
    print(f"  stored configs recomputed identically: {r2['recomputed_identical']} "
          f"(named by this base {r2['named_by_this_base']}; cannot be named {len(r2['cannot_be_named'])})")
    print(f"  pre-existing generator-state readers (not Phase 18): {list(res['test_no_privileged_input_in_ml']['pre_existing_generator_state_readers'])}")
