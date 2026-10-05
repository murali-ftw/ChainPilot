"""Phase 23B Stage 4 -- serving tests for ml/serve/rescue.py (predict-the-rescue, clean LightGBM B1a). New file.

  bit-exact     for every seed, the features served for the 200 stored test rows equal the fit-time features, and the served
                probabilities equal the stored clean-arm test predictions, BIT-EXACTLY
  mismatch      (constructed failing cases) a bundle checked against another seed's configured member, a bundle whose
                identity.json was altered, a bundle whose booster file was altered, and a renamed bundle directory all RAISE
  clean         the stored, leaky cache `v8` is refused (NotClean); the clean cache is accepted
  as-of         a master row recorded after t0 raises AsOfViolation; a t0 outside the panel raises
  AST scan      ml/serve/rescue.py (and the feature builder it calls, phase7_fit.World) never name inventory_position_weekly,
                part_demand_weekly or any of the nine LEAKING columns; SELF-TEST: a constructed offender is flagged

    python ml/tests/test_phase23b_serve.py
"""
from __future__ import annotations
import os, sys, ast, json, shutil, tempfile, inspect
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "serve"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "baselines")]
import lightgbm  # noqa: F401  -- before anything else (phase7_fit asserts torch is not imported)
import numpy as np, pandas as pd
import rescue as RS
from clean_panel import LEAKING

SEEDS = (7, 17, 27, 37, 47)
FORBIDDEN = ["inventory_position_weekly", "part_demand_weekly"] + list(LEAKING)


def forbidden_names(source):
    """Every string constant or identifier in `source` that names a forbidden table or a LEAKING column."""
    hits = set()
    for node in ast.walk(ast.parse(source)):
        vals = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            vals.append(node.value)
        elif isinstance(node, ast.Name):
            vals.append(node.id)
        elif isinstance(node, ast.Attribute):
            vals.append(node.attr)
        for v in vals:
            hits.update(f for f in FORBIDDEN if f in v)
    return sorted(hits)


def test_ast_scan():
    assert not forbidden_names(open(RS.__file__, encoding="utf-8").read()), "ml/serve/rescue.py names a forbidden table / column"
    import phase7_fit as P7
    src = inspect.getsource(P7.World)
    tables = [f for f in forbidden_names(src) if f in ("inventory_position_weekly", "part_demand_weekly")]
    assert not tables, f"the feature builder reads {tables}"
    offender = "import pandas as pd\nx = pd.read_csv('db/x/inventory_position_weekly.csv')\ny = frame['load_ratio']\n"
    assert forbidden_names(offender) == ["inventory_position_weekly", "load_ratio"], "AST self-test: the offender was not flagged"
    return "PASS"


def test_bit_exact(feats):
    out = {}
    for s in SEEDS:
        d = os.path.join(RS.BUNDLES, f"rescue_lgbm_v8clean_s{s}")
        member = RS.load_bundle(d, RS.configured_member(s))
        z = np.load(os.path.join(d, "test_rows_200.npz"))
        X = np.empty_like(z["X"]); P = np.empty(len(z["X"]), np.float32)
        for u in np.unique(z["snap"]):
            k = z["snap"] == u
            X[k] = feats.build(z["part"][k], z["plant"][k], pd.Timestamp(u), z["w"][k])
        P[:] = RS.predict(member, X)
        fx = np.array_equal(np.nan_to_num(X, nan=-7.0), np.nan_to_num(z["X"], nan=-7.0)) and np.array_equal(np.isnan(X), np.isnan(z["X"]))
        fp = np.array_equal(P, z["P_stored"])
        out[s] = dict(features_bit_exact=bool(fx), predictions_bit_exact=bool(fp), max_abs_diff=float(np.abs(P - z["P_stored"]).max()))
        assert fx and fp, f"seed {s}: not bit-exact {out[s]}"
    return out


def test_mismatch_raises():
    fired = {}
    d7 = os.path.join(RS.BUNDLES, "rescue_lgbm_v8clean_s7")
    try:
        RS.load_bundle(d7, RS.configured_member(17)); fired["other_seed_member"] = False
    except RS.IdentityMismatch:
        fired["other_seed_member"] = True
    tmp = tempfile.mkdtemp()
    try:
        for case in ("identity_altered", "booster_altered", "renamed_directory"):
            dst = os.path.join(tmp, case, "rescue_lgbm_v8clean_s7" if case != "renamed_directory" else "rescue_lgbm_v8clean_s7_copy")
            shutil.copytree(d7, dst)
            if case == "identity_altered":
                j = json.load(open(os.path.join(dst, "identity.json"))); j["feature_list"] = j["feature_list"][::-1]
                json.dump(j, open(os.path.join(dst, "identity.json"), "w"))
            elif case == "booster_altered":
                with open(os.path.join(dst, "model.txt"), "a") as f:
                    f.write("\n")
            try:
                RS.load_bundle(dst, RS.configured_member(7)); fired[case] = False
            except RS.IdentityMismatch:
                fired[case] = True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert all(fired.values()), f"a mismatched bundle did not raise: {fired}"
    return fired


def test_clean_and_asof(feats):
    import json as _j, config
    meta = _j.load(open(os.path.join(config.CACHE, "v8", "meta.json")))
    try:
        RS.assert_clean(meta); leaky_refused = False
    except RS.NotClean:
        leaky_refused = True
    RS.assert_clean(feats.Wd.meta)
    fut = pd.DataFrame({"recorded_ts": ["2024-01-01", "2026-01-01"]})
    try:
        RS.assert_asof(fut, "2025-01-06", "constructed"); future_raises = False
    except RS.AsOfViolation:
        future_raises = True
    try:
        feats.build(["P00001"], ["PL01"], "2040-01-01", [0]); out_of_panel_raises = False
    except RS.AsOfViolation:
        out_of_panel_raises = True
    assert leaky_refused and future_raises and out_of_panel_raises
    return dict(leaky_cache_refused=leaky_refused, future_master_row_raises=future_raises, t0_outside_panel_raises=out_of_panel_raises)


if __name__ == "__main__":
    print("  AST scan:", test_ast_scan())
    feats = RS.Features()
    print("  bit-exact:", json.dumps(test_bit_exact(feats)))
    print("  mismatch raises:", json.dumps(test_mismatch_raises()))
    print("  clean / as-of:", json.dumps(test_clean_and_asof(feats)))
    print("PASS")
