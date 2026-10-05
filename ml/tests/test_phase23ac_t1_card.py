"""Phase 23AC T1 -- tests of the order-time CARD (ml/serve/order_time_card.py), each with a case that MUST fail.

  test_card_bit_exact   cards for a fixed set of v8 test lines (the first 40 test lines of each of 5 creation weeks, as in
                        test_phase22_serve.py): late_risk_score equals the mean of the 5 stored per-seed flag predictions
                        and the unrounded expected weeks equal the store-based KM + offset, max |diff| == 0.0; the resolved
                        level matches the store's; the card fields are consistent with each other
  test_card_mismatch    a wrong expected name, a tampered model file and a missing bundle each raise IdentityMismatch through
                        the card; a card handed a service whose product is not the configured one raises
  test_ast_scan         ml/serve/order_time_card.py and ml/eval/phase23ac_t1_compare.py name no PRIVILEGED path,
                        inventory_position_weekly, part_demand_weekly or _sim.npz, and carry no training call; constructed
                        offenders are flagged, a docstring is not

  HADES_DEVICE=cpu ./venv/bin/python ml/tests/test_phase23ac_t1_card.py    (a few minutes: group statistics per creation week)
"""
from __future__ import annotations
import os, sys, ast, json, shutil, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "serve"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "baselines")]
import numpy as np, pandas as pd
import config
import test_phase18_isolation as T18

NEW = ["ml/serve/order_time_card.py", "ml/eval/phase23ac_t1_compare.py"]
FORBID = ("PRIVILEGED__", "inventory_position_weekly", "part_demand_weekly", "_sim.npz")
TRAIN_ATTRS = ("fit", "backward", "step", "zero_grad")


def test_card_bit_exact():
    import order_time as OT, order_time_card as OC, grpstats as GS, folds
    card = OC.OrderTimeCard()
    svc = card.svc
    pw, w, k = svc.prod["parent"], svc.prod["world"], svc.prod["k"]
    Z, _ = GS.load(pw, "place")
    li = Z["line"].astype(np.int64); src = svc.src
    cr = pd.DatetimeIndex(src.created[li]); tau = pd.to_datetime(Z["tau"]).values
    tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(cr)]
    ti = np.flatnonzero(te)
    weeks = np.unique(tau[ti])[[0, 10, 20, 30, 45]]
    pick = np.concatenate([ti[tau[ti] == wk][:40] for wk in weeks])
    lines = pd.DataFrame({"po_line_id": src.line_id[li[pick]], "channel_id": src.ch.channel_id.to_numpy()[src.chan[li[pick]]],
                          "created_ts": pd.DatetimeIndex(src.created[li[pick]]), "qty_ordered": src.qty[li[pick]]})
    cards = card.cards(lines)
    assert len(cards) == len(pick)
    ent = Z["entity"][pick]
    assert [c["po_line"] for c in cards] == [str(e) for e in ent], "cards are not in input order"
    stored = []
    for s in OT.SEEDS:
        z = np.load(os.path.join(config.ARTIFACTS, "phase22", "preds", f"{w}_arrival_place_p22_flag_lag1_k10_s{s}_test.npz"))
        pos = pd.Series(np.arange(len(z["entity"])), index=z["entity"].astype(str)).reindex(ent).to_numpy()
        assert not np.isnan(pos.astype(float)).any(), "a picked line is missing from the stored predictions"
        stored.append(np.asarray(z["P"], float)[pos.astype(int)])
    ref = np.mean(stored, 0)
    score = np.array([c["late_risk_score"] for c in cards])
    flag_diff = float(np.abs(score - ref).max())
    sub = {kk: v[pick] for kk, v in Z.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(li)}
    km_ref = GS.standalone_arrival_weeks(sub, k) + svc.prod["km_offset_a_weeks"]
    wk = np.array([c["_expected_lead_weeks"] for c in cards])
    km_diff = float(np.abs(wk - km_ref).max())
    _, raw, res5, _ = GS.arrival_levels(sub, k, sub["month"].astype(np.int64))
    lvl_ref = [OC.LEVEL_NAMES[int(x)] for x in res5]
    n_ref = [int(raw[OC.LEVEL_KEYS[int(x)]][i, GS.ASTATS.index("n")]) for i, x in enumerate(res5)]
    assert flag_diff == 0.0, f"card late_risk_score differs from the stored predictions by {flag_diff}"
    assert km_diff == 0.0, f"card expected weeks differ from the store-based KM + offset by {km_diff}"
    assert [c["resolved_level"] for c in cards] == lvl_ref and [c["resolved_n"] for c in cards] == n_ref, "resolved level / n differ"
    # internal consistency of each card (dates follow from the unrounded fields; summary is the template)
    thr = svc.prod["flag"]["watch_threshold_val_top5pct"]
    for c in cards:
        c0 = pd.Timestamp(c["created_date"])
        assert pd.Timestamp(c["expected_receipt_date"]) == c0 + pd.Timedelta(days=int(np.round(7 * c["_expected_lead_weeks"])))
        assert pd.Timestamp(c["interval_earliest"]) == c0 + pd.Timedelta(days=int(np.round(c["_interval_lo_days"])))
        assert pd.Timestamp(c["interval_latest"]) == c0 + pd.Timedelta(days=int(np.round(c["_interval_hi_days"])))
        assert c["expected_weeks"] == round(c["_expected_lead_weeks"], 2)
        assert c["expected_iso_week"] == pd.Timestamp(c["expected_receipt_date"]).isocalendar()[1]
        assert c["late_risk_label"] == (OC.WATCH_LABEL if c["late_risk_score"] >= thr else OC.LATE_CLASS)
        assert c["summary"] == OC.summary(c, c["late_risk_score"] >= thr)
    labels = pd.Series([c["late_risk_label"] for c in cards]).value_counts().to_dict()
    levels = pd.Series([c["resolved_level"] for c in cards]).value_counts().to_dict()
    return dict(rows=int(len(pick)), creation_weeks=[str(pd.Timestamp(x).date()) for x in weeks], flag_max_abs_diff=flag_diff,
                km_max_abs_diff=km_diff, labels=labels, resolved_levels=levels, example=cards[0]["summary"])


def test_card_mismatch():
    import order_time as OT, order_time_card as OC
    from guard import IdentityMismatch
    fired = {}
    try:
        OC.OrderTimeCard(expected_name="order_time|world=v8|date=km_k10|flag=flag_k10|conformal=month"); fired["wrong name"] = False
    except IdentityMismatch:
        fired["wrong name"] = True
    tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(OT.BUNDLE, os.path.join(tmp, "b"))
        prod = json.load(open(os.path.join(tmp, "b", "product.json")))
        f0 = os.path.join(tmp, "b", prod["flag"]["models"][0]["file"])
        open(f0, "a").write("\n# tampered\n")
        try:
            OC.OrderTimeCard(bundle_dir=os.path.join(tmp, "b")); fired["tampered model"] = False
        except IdentityMismatch:
            fired["tampered model"] = True
        try:
            OC.OrderTimeCard(bundle_dir=os.path.join(tmp, "missing")); fired["missing bundle"] = False
        except IdentityMismatch:
            fired["missing bundle"] = True
    finally:
        shutil.rmtree(tmp)
    try:
        OC.OrderTimeCard(svc=types.SimpleNamespace(prod={"name": "order_time|world=v8|other"})); fired["foreign service"] = False
    except IdentityMismatch:
        fired["foreign service"] = True
    try:                                                   # the correct product passes the card's own check (no service build)
        OC.OrderTimeCard(svc=types.SimpleNamespace(prod={"name": OT.NAME})); fired["correct name accepted"] = True
    except IdentityMismatch:
        fired["correct name accepted"] = False
    assert all(fired.values()), fired
    return fired


def training_calls(path, src=None):
    """Attribute calls named fit / backward / step / zero_grad, and lgb.train / lightgbm.train."""
    tree = ast.parse(open(path).read() if src is None else src)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            a = node.func.attr
            base = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            if a in TRAIN_ATTRS or (a == "train" and base in ("lgb", "lightgbm")):
                bad.append(f"{base}.{a}" if base else a)
    return bad


def test_ast_scan():
    saved = T18.FORBIDDEN_STR
    T18.FORBIDDEN_STR = FORBID
    try:
        hits = {m: T18.violations(os.path.join(config.REPO, m)) for m in NEW}
        offenders = ["x = pd.read_csv(D + '/inventory_position_weekly.csv')\n", "p = 'reports/part2/phase22/PRIVILEGED__x.npz'\n",
                     "t = pd.read_csv(D + '/part_demand_weekly.csv')\n", "z = np.load(D + '/_sim.npz')\n"]
        caught = [bool(T18.violations("x.py", o)) for o in offenders]
        doc_ok = not T18.violations("x.py", '"""never reads inventory_position_weekly (docstring)"""\nx = 1\n')
    finally:
        T18.FORBIDDEN_STR = saved
    train_hits = {m: training_calls(os.path.join(config.REPO, m)) for m in NEW}
    t_off = ["m = lgb.train(p, d)\n", "iso = IsotonicRegression().fit(x, y)\n", "loss.backward()\nopt.step()\n"]
    t_caught = [bool(training_calls("x.py", o)) for o in t_off]
    assert all(caught) and doc_ok, f"scanner self-test failed: {caught}, {doc_ok}"
    assert all(t_caught), f"training-call scanner self-test failed: {t_caught}"
    bad = {m: v for m, v in hits.items() if v}
    assert not bad, bad
    tbad = {m: v for m, v in train_hits.items() if v}
    assert not tbad, tbad
    return dict(scanned=NEW, forbidden_hits=0, training_calls=0, offenders_flagged=sum(caught) + sum(t_caught),
                docstring_passes=doc_ok)


if __name__ == "__main__":
    os.environ.setdefault("HADES_DEVICE", "cpu")
    import phase21_paths as PP
    PP.register()
    import phase12_common as C
    res = {"stamp": C.stamp()}
    for n in ("test_ast_scan", "test_card_mismatch", "test_card_bit_exact"):
        res[n] = globals()[n]()
        print(f"  PASS {n}: {res[n]}", flush=True)
    assert "torch" not in sys.modules, "torch was imported alongside LightGBM"
    out = os.path.join(config.ARTIFACTS, "phase23ac", "t1")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "card_tests.json"), "w"), indent=1, default=str)
