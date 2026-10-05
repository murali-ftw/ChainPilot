"""Phase 23B -- every stored config recomputes to the identity it had on the branch that CREATED it (new file).

ml/tests/test_integration_identity.py (2026-10-04, unchanged) compares every non-Phase-16 config with artifact_identity at the
Phase 15 tip 90a38ed. On the Mac, where it was written, no Phase 17 bundle exists (Phase 17 ran on this Windows machine), so its
"Phase 0-15" bucket was complete there. Here ml/artifacts also holds the Phase 17 bundles (B1b rescue_week, B2 _tgt*, B3
_headband5, B4 _lean), whose creating module is the phase17 branch tip a2301ef; the integration test STOPs on the first of them
(`_lean`: 90a38ed cannot name the lean axis). This test applies the integration test's own rule -- compare with the CREATING
branch's module -- with that one reference added:
  Phase 17 configs   (cap_target / lean_encoder / fill_head band5 / trained_by ml/train/phase17* / task rescue_week)  vs a2301ef
  Phase 16 variants  vs caf613c (the integration test's rule)
  everything else    vs 90a38ed (the integration test's rule)
and, for every config, directory == the merged bundle_name(cfg) and the path ends in bundle_path_key(cfg).
FALSIFICATION: a Phase 0-15 config given a Phase 17 axis must get a DIFFERENT name under the merged module.

    python ml/tests/test_phase23b_identity.py
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import test_integration_identity as TI          # unchanged: module_at, stored_configs, is_phase16_variant, FNS, AI

P17_TIP = "a2301ef"


def is_phase17(cfg):
    return bool(cfg.get("cap_target") or cfg.get("lean_encoder") or cfg.get("fill_head") == "band5" or cfg.get("task") == "rescue_week"
                or str(cfg.get("trained_by", "")).startswith("ml/train/phase17"))


def test_every_stored_config():
    OLD, P16, P17 = TI.module_at(TI.BASE), TI.module_at(TI.P16_TIP), TI.module_at(P17_TIP)
    n = dict(phase0_15=0, phase16_variants=0, phase17=0)
    for f in TI.stored_configs():
        cfg = json.load(open(f))
        if "arch" not in cfg or "seed" not in cfg:
            continue
        cfg.setdefault("origin", None)
        if TI.is_phase16_variant(cfg):
            ref, k = P16, "phase16_variants"
        elif is_phase17(cfg):
            ref, k = P17, "phase17"
        else:
            ref, k = OLD, "phase0_15"
        for fn in TI.FNS:
            a, b = getattr(ref, fn)(cfg), getattr(TI.AI, fn)(cfg)
            assert a == b, f"STOP: {fn} changed for {os.path.relpath(f, TI.REPO)}: {a!r} -> {b!r}"
        rel = os.path.relpath(os.path.dirname(f), os.path.join(TI.REPO, "ml", "artifacts"))
        assert os.path.basename(os.path.dirname(f)) == TI.AI.bundle_name(cfg), f"STOP: directory is not bundle_name: {rel}"
        if rel.startswith("bundles" + os.sep):
            assert rel.endswith(TI.AI.bundle_path_key(cfg).replace("/", os.sep)), f"STOP: path is not bundle_path_key: {rel}"
        n[k] += 1
    n["total"] = sum(n.values())
    return n


def test_falsification():
    base = dict(task="arrival_week", world="v8", origin=None, arch="lite", depth=4, lr=2.5e-4, seed=7)
    assert TI.AI.bundle_name(dict(base, lean_encoder=True)) != TI.AI.bundle_name(base)
    cap = dict(base, task="capacity_strain", arch="mp")
    assert TI.AI.bundle_name(dict(cap, cap_target="delta")) != TI.AI.bundle_name(cap)
    assert TI.module_at(TI.BASE).config_name(dict(base, lean_encoder=True)) != TI.AI.config_name(dict(base, lean_encoder=True)), \
        "the Phase 15 module must NOT name a Phase 17 config the way the merged module does (else the bucket split proves nothing)"
    return "PASS"


if __name__ == "__main__":
    r = test_every_stored_config()
    print(f"  PASS stored configs: {r['total']} / {r['total']}  (Phase 0-15 {r['phase0_15']}, Phase 16 variants {r['phase16_variants']}, "
          f"Phase 17 {r['phase17']})")
    print("  falsification:", test_falsification())
