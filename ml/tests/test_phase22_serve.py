"""Phase 22 Stage 2e / 3e -- serving tests, each with a case that MUST fail.

  test_bit_exact            serve a fixed set of test lines (the first 40 test lines of each of 5 creation weeks); the flag's
                            5-seed mean equals the mean of the stored per-seed predictions bit for bit, and the expected week
                            equals the store-based KM + offset bit for bit
  test_mismatch_raises      a bundle with the wrong name, a tampered model file (SHA-1) or a missing bundle MUST raise
  test_ast_scan             new serve modules name no PRIVILEGED path, inventory_position_weekly, part_demand_weekly, _sim.npz
                            or the raw channel_performance_weekly table; constructed offenders are flagged, a docstring is not

  HADES_DATA_ROOT=<main checkout> python ml/tests/test_phase22_serve.py
"""
from __future__ import annotations
import os, sys, json, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "serve"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import config
import test_phase18_isolation as T18

SERVE_NEW = ["ml/serve/order_time.py", "ml/serve/fill_consolidated.py"]
FORBID = ("PRIVILEGED__", "inventory_position_weekly", "part_demand_weekly", "_sim.npz", "channel_performance_weekly")


def test_bit_exact():
    import order_time as OT, grpstats as GS, folds
    svc = OT.OrderTimeService()
    pw = svc.prod["parent"]; w = svc.prod["world"]
    Z, _ = GS.load(pw, "place")
    li = Z["line"].astype(np.int64); src = svc.src
    cr = pd.DatetimeIndex(src.created[li]); tau = pd.to_datetime(Z["tau"]).values
    tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(cr)]
    ti = np.flatnonzero(te)
    weeks = np.unique(tau[ti])[[0, 10, 20, 30, 45]]
    pick = np.concatenate([ti[tau[ti] == wk][:40] for wk in weeks])
    lines = pd.DataFrame({"channel_id": src.ch.channel_id.to_numpy()[src.chan[li[pick]]], "created_ts": pd.DatetimeIndex(src.created[li[pick]]),
                          "qty_ordered": src.qty[li[pick]]})
    out = svc.predict(lines)
    ent = Z["entity"][pick]
    stored = []
    for s in OT.SEEDS:
        z = np.load(os.path.join(config.ARTIFACTS, "phase22", "preds", f"{w}_arrival_place_p22_flag_lag1_k10_s{s}_test.npz"))
        pos = pd.Series(np.arange(len(z["entity"])), index=z["entity"].astype(str)).reindex(ent).to_numpy()
        stored.append(np.asarray(z["P"], float)[pos.astype(int)])
    ref = np.mean(stored, 0)
    flag_diff = float(np.abs(out.p_late.to_numpy() - ref).max())
    sub = {k: v[pick] for k, v in Z.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(li)}
    km_ref = GS.standalone_arrival_weeks(sub, svc.prod["k"]) + svc.prod["km_offset_a_weeks"]
    km_diff = float(np.abs(out.expected_lead_weeks.to_numpy() - km_ref).max())
    assert flag_diff == 0.0, f"served flag differs from the stored predictions by {flag_diff}"
    assert km_diff == 0.0, f"served expected week differs by {km_diff}"
    return dict(rows=int(len(pick)), creation_weeks=[str(pd.Timestamp(x).date()) for x in weeks], flag_max_abs_diff=flag_diff,
                km_max_abs_diff=km_diff)


def test_mismatch_raises():
    import order_time as OT
    from guard import IdentityMismatch
    fired = {}
    try:
        OT.OrderTimeService(expected_name="order_time|world=v8|date=km_k10|flag=flag_k10|conformal=month"); fired["wrong name"] = False
    except IdentityMismatch:
        fired["wrong name"] = True
    tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(OT.BUNDLE, os.path.join(tmp, "b"))
        prod = json.load(open(os.path.join(tmp, "b", "product.json")))
        f0 = os.path.join(tmp, "b", prod["flag"]["models"][0]["file"])
        open(f0, "a").write("\n# tampered\n")
        try:
            OT.OrderTimeService(bundle_dir=os.path.join(tmp, "b")); fired["tampered model"] = False
        except IdentityMismatch:
            fired["tampered model"] = True
        try:
            OT.OrderTimeService(bundle_dir=os.path.join(tmp, "missing")); fired["missing bundle"] = False
        except IdentityMismatch:
            fired["missing bundle"] = True
    finally:
        shutil.rmtree(tmp)
    try:
        OT.OrderTimeService(); fired["correct bundle loads"] = True
    except IdentityMismatch:
        fired["correct bundle loads"] = False
    assert all(fired.values()), fired
    return fired


def test_ast_scan():
    saved = T18.FORBIDDEN_STR
    T18.FORBIDDEN_STR = FORBID
    try:
        hits = {m: T18.violations(os.path.join(config.REPO, m)) for m in SERVE_NEW if os.path.exists(os.path.join(config.REPO, m))}
        offenders = ["x = pd.read_csv(D + '/inventory_position_weekly.csv')\n", "p = 'reports/part2/phase22/PRIVILEGED__x.npz'\n",
                     "t = pd.read_csv(D + '/channel_performance_weekly.csv')\n", "z = np.load(D + '/_sim.npz')\n"]
        caught = [bool(T18.violations("x.py", o)) for o in offenders]
        doc_ok = not T18.violations("x.py", '"""never reads inventory_position_weekly (docstring)"""\nx = 1\n')
    finally:
        T18.FORBIDDEN_STR = saved
    assert all(caught) and doc_ok, f"scanner self-test failed: {caught}, {doc_ok}"
    bad = {m: v for m, v in hits.items() if v}
    assert not bad, bad
    return dict(scanned=list(hits), offenders_flagged=sum(caught), docstring_passes=doc_ok)


if __name__ == "__main__":
    import phase21_paths as PP
    PP.register()
    import phase12_common as C
    res = {"stamp": C.stamp()}
    for n in ("test_ast_scan", "test_mismatch_raises", "test_bit_exact"):
        res[n] = globals()[n]()
        print(f"  PASS {n}: {res[n]}", flush=True)
    json.dump(res, open(os.path.join(config.ARTIFACTS, "phase22", "serve_tests.json"), "w"), indent=1, default=str)
