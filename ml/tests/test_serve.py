"""Serving path tests (integration, 2026-10-04). Read-only: stored bundles, no training.

(i)   BIT-EXACT (no-graph members) / within 1e-5 (graph members, deviation 190): for every member of every SERVABLE item in shipped.json["phase20_shipped"], serving reproduces the stored
      preds_test.npz exactly (np.array_equal on every array) on fixed test snapshots (first and last of 2025). The row-family
      members are fed features rebuilt FROM SOURCE by ml/serve/features.py, so this also proves the serving feature builder
      equals the training one.
(ii)  FALSIFICATION: the identity guard MUST raise on (a) a different seed's bundle, (b) a Phase 19 row-family bundle served
      in place of its incumbent -- the two share artifact_identity.identity_of, the exact silent-substitution case -- and
      (c) a missing bundle; a "declared, not servable" item MUST raise NotServable.
(iii) ISOLATION: no module under ml/serve names inventory_position_weekly or a PRIVILEGED__ / phase18-20 privileged path
      (outside docstrings) -- the Phase 18 AST scanner, with a constructed-offender self-test.

    python ml/tests/test_serve.py
"""
from __future__ import annotations
import os, sys, json, copy, glob
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "ml", "serve")]
import numpy as np, pandas as pd

SNAPSHOTS = ("2025-01-06", "2025-12-08")
# Deviation 190: graph members (depth > 0) run a nondeterministic MPS kernel (index_put_with_accumulate_mps, recorded in
# every bundle's train_log as nondeterministic_ops_warned). Two serving runs of the same member differ by up to ~2.4e-7, so
# the stored predictions are one draw and cannot be reproduced bit-exactly. No-graph members MUST be bit-exact.
GRAPH_TOL = 1e-5


def test_bit_exact():
    import service as SV, phase5_heads as P5
    srv = SV.Server("v8"); dec = SV.decision(); res = {}
    for name, item in dec["items"].items():
        if item["status"] != "servable":
            continue
        for m in item["members"]:
            M = srv.load_member(m)
            got = srv.predict(M, SNAPSHOTS)
            lb, tr, va, te = srv.labels(m["task"])
            order = P5.ordered(lb, te); pos = pd.Series(range(len(order)), index=order)
            stored = dict(np.load(os.path.join(M["dir"], "preds_test.npz")))
            sel = pos.loc[got["rows"]].to_numpy()
            graph = int(M["cfg"].get("depth") or 0) > 0
            diffs = {}
            for k in ("P", "S", "pT"):
                if k in got:
                    d = float(np.abs(got[k].astype(float) - stored[k][sel].astype(float)).max())
                    diffs[k] = d
                    if graph:   # deviation 190: MPS scatter-add (index_put_with_accumulate) is nondeterministic run to run
                        assert d <= GRAPH_TOL, f"{name} {m['name']} {k}: |serving - stored| {d} > {GRAPH_TOL}"
                    else:
                        assert np.array_equal(got[k], stored[k][sel]), f"{name} {m['name']} {k}: not bit-exact ({d})"
            res[f"{name}|{m['name']}"] = dict(rows=int(len(sel)), graph=graph, bit_exact=not graph, max_abs_diff=diffs)
        out = srv.serve_item(name, SNAPSHOTS)
        assert len(out["rows"]) > 0
    return res


def test_guard_fires():
    import service as SV, guard as G
    srv = SV.Server("v8"); items = SV.decision()["items"]; fired = {}
    m = copy.deepcopy(items["arrival.ranked_late_list"]["members"][0])
    wrong_seed = dict(m, name=m["name"].replace("_s7", "_s17"))                 # directory of seed 17, identity of seed 7
    try:
        G.check(SV.bundle_dir(wrong_seed), m | {"name": m["name"]}); fired["wrong_seed"] = "DID NOT FIRE"
    except G.IdentityMismatch as e:
        fired["wrong_seed"] = f"fired: {str(e)[:120]}"
    rf = items["arrival.point_estimate_neural"]["members"][0]                     # row-family bundle, same identity_of
    assert rf["identity"] == m["identity"], "precondition: incumbent and row-family sibling share identity_of"
    try:
        G.check(SV.bundle_dir(rf), m); fired["row_family_substitution"] = "DID NOT FIRE"
    except G.IdentityMismatch as e:
        fired["row_family_substitution"] = f"fired: {str(e)[:120]}"
    try:
        srv.load_member(dict(m, name="v8_lite_h4_lr0.00025_s99")); fired["missing"] = "DID NOT FIRE"
    except G.IdentityMismatch as e:
        fired["missing"] = f"fired: {str(e)[:80]}"
    try:
        srv.serve_item("fill.b5flat22", SNAPSHOTS); fired["not_servable"] = "DID NOT FIRE"
    except G.NotServable as e:
        fired["not_servable"] = f"fired: {str(e)[:80]}"
    G.check(SV.bundle_dir(m), m)                                                 # the correct member passes
    assert all(v.startswith("fired") for v in fired.values()), fired
    return fired


def test_serve_isolation():
    import test_phase18_isolation as T18
    T18.FORBIDDEN_STR = T18.FORBIDDEN_STR + ("inventory_position_weekly", "PRIVILEGED__", "phase19/preds", "phase20/preds")
    files = sorted(glob.glob(os.path.join(REPO, "ml", "serve", "*.py")))
    hits = {os.path.relpath(f, REPO): T18.violations(f) for f in files}
    bad = {k: v for k, v in hits.items() if v}
    assert not bad, bad
    assert T18.violations("x.py", "t = pd.read_csv(D + '/inventory_position_weekly.csv')\n")
    assert T18.violations("x.py", "z = np.load('reports/part2/phase20/PRIVILEGED__probe_features_v8s1001.npz')\n")
    assert not T18.violations("x.py", '"""never reads inventory_position_weekly (docstring)"""\nx = 1\n')
    return dict(files=[os.path.relpath(f, REPO) for f in files], violations=0)


if __name__ == "__main__":
    out = {}
    for t in ("test_serve_isolation", "test_guard_fires", "test_bit_exact"):
        out[t] = globals()[t]()
        print(f"  PASS {t}: {json.dumps(out[t])[:400]}", flush=True)
    os.makedirs(os.path.join(REPO, "ml", "artifacts", "integration"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "ml", "artifacts", "integration", "serve_tests.json"), "w"), indent=1)
