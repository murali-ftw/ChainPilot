"""Phase 23AC T2 -- the dormant fill service's identity guard and a small bit-exact check. CPU only, no training.

Guard (each constructed failing case must RAISE guard.IdentityMismatch):
  missing bundle dir; a tampered config (a copy of the s7 bundle with row_family edited); a tampered model file (a copy with
  one checkpoint byte flipped, caught by the pinned SHA-1); a wrong expected name (the s7 bundle checked as s17's, and the
  service configured with the season-only bundle). The untampered copy passes, so each raise is caused by its tamper.
Load: the s7 member loads and serves the first 40 rows of the first test snapshot; max |served - stored preds_test P| is
printed exactly and asserted <= 1e-5 (the exact-zero question is P3, decided by ml/eval/phase23ac_t2_serve.py).

    HADES_DEVICE=cpu python ml/tests/test_phase23ac_t2_serve.py
"""
from __future__ import annotations
import os, sys, json, shutil, tempfile
os.environ.setdefault("HADES_DEVICE", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "serve")]
import numpy as np
import fill_consolidated as FC
import guard as G


def raises(fn):
    try:
        fn()
    except G.IdentityMismatch as e:
        return str(e)
    return None


def copy_bundle(seed, dst_root):
    name = FC.expected_name(seed)
    dst = os.path.join(dst_root, name); os.makedirs(dst)
    for f in FC.REQUIRED:
        shutil.copy2(os.path.join(FC.ROOT, name, f), os.path.join(dst, f))
    return dst


def test_guard_raises():
    out = {}
    tmp = tempfile.mkdtemp(prefix="p23ac_t2_")
    try:
        name7 = FC.expected_name(7)
        out["missing"] = raises(lambda: FC.check_bundle(os.path.join(tmp, name7), name7, FC.PINNED_SHA1[7]))
        assert out["missing"], "a missing bundle did not raise"
        good = copy_bundle(7, os.path.join(tmp, "good"))
        assert FC.check_bundle(good, name7, FC.PINNED_SHA1[7])["seed"] == 7, "the untampered copy must pass"
        # tampered config
        bad = copy_bundle(7, os.path.join(tmp, "cfg"))
        cfg = json.load(open(os.path.join(bad, "config.json"))); cfg["row_family"] = "season"
        json.dump(cfg, open(os.path.join(bad, "config.json"), "w"), indent=1)
        out["tampered_config"] = raises(lambda: FC.check_bundle(bad, name7, FC.PINNED_SHA1[7]))
        assert out["tampered_config"], "a tampered config did not raise"
        # tampered model file (one byte flipped)
        badm = copy_bundle(7, os.path.join(tmp, "model"))
        p = os.path.join(badm, "checkpoint.pt"); b = bytearray(open(p, "rb").read()); b[len(b) // 2] ^= 0xFF
        open(p, "wb").write(bytes(b))
        out["tampered_model_file"] = raises(lambda: FC.check_bundle(badm, name7, FC.PINNED_SHA1[7]))
        assert out["tampered_model_file"], "a tampered checkpoint did not raise"
        # wrong expected name
        out["wrong_name"] = raises(lambda: FC.check_bundle(os.path.join(FC.ROOT, name7), FC.expected_name(17), FC.PINNED_SHA1[17]))
        assert out["wrong_name"], "a wrong expected name did not raise"
        svc = FC.FillConsolidatedService(seeds=(7,), names={7: f"{FC.WORLD}_none_h0_lr0.000125_s7_rfseason"})
        out["wrong_member_season_only"] = raises(lambda: svc.check(7))
        assert out["wrong_member_season_only"], "the season-only bundle was accepted as the season+cadence member"
        # incomplete bundle
        inc = copy_bundle(7, os.path.join(tmp, "inc"))
        cfg = json.load(open(os.path.join(inc, "config.json"))); cfg["complete"] = False
        json.dump(cfg, open(os.path.join(inc, "config.json"), "w"), indent=1)
        out["incomplete"] = raises(lambda: FC.check_bundle(inc, name7, FC.PINNED_SHA1[7]))
        assert out["incomplete"], "an incomplete bundle did not raise"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for s in FC.SEEDS:                                 # every real member passes its own guard
        FC.check_bundle(os.path.join(FC.ROOT, FC.expected_name(s)), FC.expected_name(s), FC.PINNED_SHA1[s])
    return {k: v[:140] for k, v in out.items()}


def test_load_and_small_bit_check():
    svc = FC.FillConsolidatedService(seeds=(7,))
    assert svc.load()[0]["model"].n_row_feats == 23
    t0 = svc.test_snapshots()[0]
    out = svc.serve([t0], 40)
    assert len(out["rows"]) == 40 and out["device"] == "cpu", (len(out["rows"]), out["device"])
    order, _ = svc._order
    assert np.array_equal(order[out["pos"]], out["rows"])
    z = np.load(os.path.join(svc.members[0]["dir"], "preds_test.npz"))
    lb = svc.labels()[0]
    assert np.array_equal(z["Y"][out["pos"]], lb.label_value.to_numpy(float)[out["rows"]]), "rows not matched"
    d = np.abs(out["per_seed"][7].astype(np.float64) - z["P"][out["pos"]].astype(np.float64))
    print(f"  s7, {len(d)} rows of {str(t0)[:10]}: max |served(CPU) - stored(MPS)| = {d.max()!r} "
          f"({int((d > 0).sum())} of {d.size} cells nonzero)")
    assert d.max() <= 1e-5, f"served differs from stored by {d.max()!r}"
    return dict(snapshot=str(t0)[:10], rows=40, max_abs_diff=float(d.max()), n_cells_nonzero=int((d > 0).sum()))


if __name__ == "__main__":
    for name in ("test_guard_raises", "test_load_and_small_bit_check"):
        print(f"  PASS {name}: {json.dumps(globals()[name](), default=str)}")
