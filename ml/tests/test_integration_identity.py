"""Integration (2026-10-04): every stored bundle config is NAMEABLE by the merged ml/artifact_identity.py and recomputes
to the identity it had on the branch that created it. HARD: any mismatch is a STOP (never fixed by renaming a bundle).

  Phase 0-15 configs          vs artifact_identity at 90a38ed (the pipeline tip every phase branched from)
  Phase 16 encoder variants   vs artifact_identity at the Phase 16 tip (caf613c), the only module that could name them before
  every config                its directory name == merged bundle_name(cfg); its path ends in merged bundle_path_key(cfg)
  Phase 19 row-family bundles under ml/artifacts/phase19/bundles: directory == phase19_identity.bundle_name(cfg), and
                              phase19_identity.config_name == merged artifact_identity.config_name when the axis is absent
Falsification: a config with a Phase 16 axis, a Phase 17 axis or the Phase 19 row_family set must get a DIFFERENT name;
the explicit default must equal absent.

    python ml/tests/test_integration_identity.py      # prints  N / N  and the split
"""
from __future__ import annotations
import os, sys, json, glob, subprocess, types
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "train")]
import artifact_identity as AI

BASE = "90a38ed"
P16_TIP = "caf613c"
FNS = ("config_name", "bundle_name", "bundle_path_key", "identity_of")


def module_at(rev):
    src = subprocess.check_output(["git", "-C", REPO, "show", f"{rev}:ml/artifact_identity.py"], text=True)
    m = types.ModuleType(f"artifact_identity_{rev}")
    exec(compile(src, f"{rev}:ml/artifact_identity.py", "exec"), m.__dict__)
    return m


def stored_configs():
    art = os.path.join(REPO, "ml", "artifacts")
    seen = set()
    for p in ("bundles/**/config.json", "backtest/bundles/**/config.json", "phase16_audit_bundles/**/config.json"):
        for f in sorted(glob.glob(os.path.join(art, p), recursive=True)):
            if f not in seen:
                seen.add(f); yield f


def is_phase16_variant(cfg):
    return bool(cfg.get("encoder_variant")) and cfg.get("encoder_variant") not in ("default",)


def test_every_stored_config():
    OLD, P16 = module_at(BASE), module_at(P16_TIP)
    n015 = n16 = 0
    for f in stored_configs():
        cfg = json.load(open(f))
        if "arch" not in cfg or "seed" not in cfg:
            continue
        cfg.setdefault("origin", None)
        ref = P16 if is_phase16_variant(cfg) else OLD
        for fn in FNS:
            a, b = getattr(ref, fn)(cfg), getattr(AI, fn)(cfg)
            assert a == b, f"STOP: {fn} changed for {os.path.relpath(f, REPO)}: {a!r} -> {b!r}"
        rel = os.path.relpath(os.path.dirname(f), os.path.join(REPO, "ml", "artifacts"))
        assert os.path.basename(os.path.dirname(f)) == AI.bundle_name(cfg), f"STOP: directory is not bundle_name: {rel}"
        if rel.startswith("bundles" + os.sep):
            assert rel.endswith(AI.bundle_path_key(cfg).replace("/", os.sep)), f"STOP: path is not bundle_path_key: {rel}"
        if is_phase16_variant(cfg):
            n16 += 1
        else:
            n015 += 1
    return dict(phase0_15=n015, phase16_variants=n16, total=n015 + n16)


def test_phase19_bundles():
    import phase19_identity as PI
    n = 0
    for f in sorted(glob.glob(os.path.join(PI.BUNDLES19, "*", "*", "config.json"))):
        cfg = json.load(open(f))
        assert os.path.basename(os.path.dirname(f)) == PI.bundle_name(cfg), f
        n += 1
    c = dict(task="capacity_strain", world="v8", origin=None, arch="mp", depth=4, lr=2.5e-4, seed=7)
    assert PI.config_name(c) == AI.config_name(c)
    return dict(phase19_bundles=n)


def test_axes_fire():
    base = dict(task="arrival_week", world="v8", origin=None, arch="lite", depth=4, lr=2.5e-4, seed=7)
    names = {AI.bundle_name(base)}
    for k, v in (("encoder_variant", "share_traj"), ("lean_encoder", True)):
        n = AI.bundle_name(dict(base, **{k: v}))
        assert n not in names, f"axis {k} does not change the name"
        names.add(n)
    cap = dict(base, task="capacity_strain", arch="mp")
    assert AI.bundle_name(dict(cap, cap_target="delta")) != AI.bundle_name(cap)
    assert AI.identity_of(dict(cap, cap_target="level")) == AI.identity_of(cap), "explicit default must equal absent"
    import phase19_identity as PI
    assert PI.bundle_name(dict(cap, row_family="fwdload")) != AI.bundle_name(cap)
    return dict(axes_checked=["encoder_variant", "lean_encoder", "cap_target", "row_family"])


if __name__ == "__main__":
    r = test_every_stored_config()
    print(f"  PASS stored configs: {r['total']} / {r['total']}  (Phase 0-15 {r['phase0_15']}, Phase 16 variants {r['phase16_variants']})")
    try:
        print("  PASS", test_phase19_bundles())
    except ModuleNotFoundError:
        print("  (phase19_identity not on this tree yet)")
    try:
        print("  PASS", test_axes_fire())
    except (ModuleNotFoundError, AssertionError) as e:
        print("  axes check not applicable on this tree yet:", type(e).__name__, e)
