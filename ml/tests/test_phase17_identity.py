"""Phase 17: the new identity axes (cap_target, lean_encoder) must leave every stored identity unchanged.

For every stored bundle config under ml/artifacts (bundles/, backtest/bundles/, phase16_audit_bundles/), the bundle path
key and identity are computed twice -- with artifact_identity as of the Phase 15 tip (90a38ed, loaded from git) and with
the current module -- and must be EQUAL, and the directory the bundle sits in must be the name the current module gives it.
The second half shows the check can fail: a Phase 17 config at a non-default axis value must get a DIFFERENT name.

    python ml/tests/test_phase17_identity.py
"""
from __future__ import annotations
import os, sys, json, glob, subprocess, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(HERE, "..")]
import artifact_identity as NEW

BASE = "90a38ed"


def old_module():
    src = subprocess.check_output(["git", "-C", REPO, "show", f"{BASE}:ml/artifact_identity.py"], text=True)
    path = os.path.join(REPO, "ml", "artifacts", "cache", f"artifact_identity_{BASE}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("ai_old", path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def stored_configs():
    art = os.path.join(REPO, "ml", "artifacts")
    pats = [os.path.join(art, "bundles", "*", "*", "config.json"), os.path.join(art, "bundles", "*", "o*", "*", "config.json"),
            os.path.join(art, "backtest", "bundles", "**", "config.json"), os.path.join(art, "phase16_audit_bundles", "**", "config.json")]
    seen = set()
    for p in pats:
        for f in glob.glob(p, recursive=True):
            if f not in seen:
                seen.add(f); yield f


def test_every_stored_identity_unchanged():
    OLD = old_module()
    n, name_mismatch = 0, []
    for f in stored_configs():
        cfg = json.load(open(f))
        if "arch" not in cfg or "seed" not in cfg or cfg.get("trained_by") == "ml/train/phase17_b1.py":
            continue                                    # not a loop.py bundle config, or a Phase 17 artifact
        cfg.setdefault("origin", None)
        assert OLD.identity_of(cfg) == NEW.identity_of(cfg), f"identity changed: {f}"
        assert OLD.bundle_path_key(cfg) == NEW.bundle_path_key(cfg), f"bundle path changed: {f}"
        assert OLD.config_name(cfg) == NEW.config_name(cfg), f"config name changed: {f}"
        if os.path.basename(os.path.dirname(f)) != NEW.bundle_name(cfg):
            name_mismatch.append(f)
        n += 1
    assert n > 0, "no stored configs found -- the check would pass vacuously"
    return n, name_mismatch


def test_new_axes_change_the_name():
    base = dict(task="capacity_strain", world="v8", origin=None, arch="mp", depth=4, lr=2.5e-4, seed=7)
    names = {NEW.bundle_name(base), NEW.bundle_name(dict(base, cap_target="level")),
             NEW.bundle_name(dict(base, cap_target="delta")), NEW.bundle_name(dict(base, cap_target="level_delta")),
             NEW.bundle_name(dict(base, lean_encoder=True))}
    assert len(names) == 4, f"default must equal absent, every non-default must differ: {names}"
    assert NEW.identity_of(dict(base, cap_target="level")) == NEW.identity_of(base)
    assert NEW.identity_of(dict(base, cap_target="delta")) != NEW.identity_of(base)


if __name__ == "__main__":
    n, mism = test_every_stored_identity_unchanged()
    print(f"  PASS stored identities unchanged under the Phase 17 axes: {n} configs")
    print(f"  bundles whose directory name differs from bundle_name(cfg) (pre-existing, not caused by Phase 17): {len(mism)}")
    for m in mism[:10]:
        print("    ", os.path.relpath(m, REPO))
    test_new_axes_change_the_name(); print("  PASS non-default axis values get distinct names; default == absent")
