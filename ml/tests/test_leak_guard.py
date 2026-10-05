"""Phase 23AC T3 -- fast tests of ml/eval/leak_guard.py, each with a case that MUST fail.

  alignment core    synthetic lines: a column equal to each channel's pending line's eventual lead MUST FAIL (share 1.0);
                    a column keyed to receipt-VISIBLE outcomes (the last lead whose receipt is visible by t) MUST PASS (0.0);
                    the 0.20 constant is the pre-registered one.
  feature check     check_feature_list flags exactly the nine LEAKING columns on the raw panel and none on the clean panel;
                    the leaked order-time arm (flag features + the nine) is flagged.
  registry          entries for all 15 panel columns (raw and clean), phase7_fit STATIC / FLAT, every family module's COLS,
                    grpstats ASTATS, phase22_rows ACK_COLS, and every order-time flag feature carries a reader.
  family poison     synthetic cadence sources: future-recorded rows poisoned (recorded_ts kept) -> no change; an offender
                    reading created_ts instead of recorded_ts MUST change.
  exit codes        fake results: a new FAIL -> 1; only the known raw fails -> 0; a known column passing on raw -> 2;
                    an UNCOVERED feature -> 1.
  smoke (opt-in)    --smoke: one week of poison + alignment on world v8 (real data; ~1-2 minutes).

  HADES_DEVICE=cpu ./venv/bin/python ml/tests/test_leak_guard.py [--smoke]
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..")]
import numpy as np
import leak_guard as LG
import clean_panel as CP

PANEL15 = ["qty_ordered", "qty_received", "is_active_week", "fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52",
           "lead_time_actual_days", "lead_time_ratio", "otd_rate_last13", "ack_gap_ratio", "load_ratio", "reporting_lag_days",
           "active_weeks_in_52", "revision_count"]


def synthetic_lines(seed=0, NCH=40, T=60, per_ch=12):
    rng = np.random.default_rng(seed)
    chan = np.repeat(np.arange(NCH), per_ch)
    vw_ord = np.sort(rng.integers(0, T - 10, (NCH, per_ch)), 1).ravel()
    lead = rng.uniform(5, 120, len(chan))                     # continuous: ties are impossible by construction
    vw_out = vw_ord + np.ceil(lead / 7).astype(int)
    qty = rng.integers(10, 500, len(chan)).astype(float)
    fill = rng.uniform(0.3, 0.99, len(chan))
    return dict(chan=chan, vw_ord=vw_ord, vw_out=vw_out, lead=lead, fill=fill, qty=qty,
                contracted=rng.uniform(20, 60, NCH)), NCH, T


def leaky_lead_column(L, NCH, T):
    """The generator's pattern: each line's EVENTUAL lead written at its ORDER week, forward-filled."""
    X = np.full((NCH, T), np.nan)
    o = np.argsort(L["vw_ord"], kind="stable")
    X[L["chan"][o], L["vw_ord"][o]] = L["lead"][o]
    return CP._ffill(X)


def visible_lead_column(L, NCH, T):
    """The as-of pattern: a line's lead enters at the week its outcome is VISIBLE, forward-filled."""
    X = np.full((NCH, T), np.nan)
    ok = L["vw_out"] < T
    o = np.flatnonzero(ok)[np.argsort(L["vw_out"][ok], kind="stable")]
    X[L["chan"][o], L["vw_out"][o]] = L["lead"][o]
    return CP._ffill(X)


def test_alignment_core():
    L, NCH, T = synthetic_lines()
    weeks = [20, 30, 40, 48]
    # constructed column equal to a pending line's eventual lead, at every active row
    E = np.full((NCH, T), np.nan)
    for t in weeks:
        pend = (L["vw_ord"] <= t) & (L["vw_out"] > t)
        for c in np.unique(L["chan"][pend]):
            E[c, t] = L["lead"][pend & (L["chan"] == c)][0]
    res = LG.alignment_share({"eventual_lead": E, "leaky": leaky_lead_column(L, NCH, T),
                              "visible": visible_lead_column(L, NCH, T)}, weeks, L, {"eventual_lead": 0.006, "leaky": 0.006,
                                                                                      "visible": 0.006})
    assert res["eventual_lead"]["share"] == 1.0, res["eventual_lead"]
    assert res["eventual_lead"]["share"] >= LG.ALIGN_FAIL                     # MUST FAIL
    assert res["leaky"]["share"] >= LG.ALIGN_FAIL, res["leaky"]                # the generator pattern MUST FAIL too
    assert res["visible"]["share"] == 0.0, res["visible"]                     # keyed to visible outcomes MUST PASS
    v, failed = LG.column_verdict([0.0], res["eventual_lead"]); assert v == "FAIL" and failed == ["label_alignment"]
    v, failed = LG.column_verdict([0.0], res["visible"]); assert v == "PASS" and failed == []
    v, failed = LG.column_verdict([0.0, 0.01], res["visible"]); assert v == "FAIL" and failed == ["poison"]
    assert LG.ALIGN_FAIL == 0.20
    # the week-t fill and supplier-month candidates are reachable
    t = 30
    wk = L["vw_ord"] == t
    cc, cv, active = LG.alignment_candidates(t, L)
    if wk.any():
        c = L["chan"][wk][0]
        m = wk & (L["chan"] == c)
        wf = (L["fill"][m] * L["qty"][m]).sum() / L["qty"][m].sum()
        assert np.isclose(cv[cc == c], wf).any()
    L2 = dict(L, sup_ratio=np.full((NCH, T), 0.731), sup_pending=np.zeros((NCH, T), bool))
    L2["sup_pending"][5, t] = True
    cc, cv, active = LG.alignment_candidates(t, L2)
    assert active[5] and np.isclose(cv[cc == 5], 0.731).any()
    # atoms: with every fill exactly 1 (so 1 - fill = 0), an all-zero column matches every active row literally, but a
    # candidate equal to 0 or 1 identifies no outcome and does not count; a column equal to a pending (non-atom) fill MUST FAIL
    La = dict(L, fill=np.ones_like(L["fill"]))
    Z = np.zeros((NCH, T))
    Fc = np.full((NCH, T), np.nan)
    for t in weeks:
        pend = (L["vw_ord"] <= t) & (L["vw_out"] > t)
        for c in np.unique(L["chan"][pend]):
            Fc[c, t] = L["fill"][pend & (L["chan"] == c)][0]
    ra = LG.alignment_share({"zeros": Z}, weeks, La)["zeros"]
    assert ra["share_literal"] == 1.0 and ra["share"] == 0.0, ra
    rf = LG.alignment_share({"pending_fill": Fc}, weeks, L)["pending_fill"]
    assert rf["share"] == 1.0, rf
    print("alignment core: PASS", {k: round(r["share"], 4) for k, r in res.items()},
          f"atoms: zeros literal {ra['share_literal']} counted {ra['share']}; pending fill {rf['share']}")


def test_feature_check():
    assert LG.check_feature_list(PANEL15, "raw") == sorted(CP.LEAKING)
    assert len(LG.check_feature_list(PANEL15, "raw")) == 9
    assert LG.check_feature_list(PANEL15, "clean") == []
    assert LG.check_feature_list(["qty_ordered", "gL4_lead_med"], "raw") == []
    prod = json.load(open(LG.PRODUCT))
    leaked_arm = list(prod["flag"]["models"][0]["features"]) + CP.LEAKING        # the constructed failing case of T1
    assert set(LG.check_feature_list(leaked_arm, "raw")) == set(CP.LEAKING)
    try:
        LG.check_feature_list(PANEL15, "published"); raise AssertionError("an unknown panel name must raise")
    except ValueError:
        pass
    print("feature check: PASS")


def test_registry():
    reg = LG.build_registry()
    fam = {}
    for e in reg:
        fam.setdefault(e["family"], set()).add(e["column"])
        assert set(e) >= {"column", "family", "source_table", "horizon", "readers", "test_kind"}, e
    assert fam["panel_raw"] == set(PANEL15) and fam["panel_clean"] == set(PANEL15), (fam.get("panel_raw"), fam.get("panel_clean"))
    static, flat = LG.world_static_flat()
    assert static == ["contracted_lead_time_days", "transport_distance_km", "mode_i"] and len(flat) == 10
    assert fam["static"] == set(static) and fam["flat"] == set(flat)
    import fwd_load, fwd_season, cadence, pulse, stock_asof, grpstats, phase22_rows
    for name, M in (("fwd_load", fwd_load), ("fwd_season", fwd_season), ("cadence", cadence), ("pulse", pulse),
                    ("stock_asof", stock_asof)):
        assert fam[name] == set(M.COLS), name
    assert fam["grpstats_astats"] == set(grpstats.ASTATS) and fam["phase22_ack"] == set(phase22_rows.ACK_COLS)
    assert "UNREGISTERED" not in fam, fam.get("UNREGISTERED")
    feats = json.load(open(LG.PRODUCT))["flag"]["models"][0]["features"]
    covered = {e["column"] for e in reg if any(r.startswith("order_time flag") for r in e["readers"])}
    assert set(feats) <= covered, set(feats) - covered
    clean_tables = [e for e in reg if e["family"] == "panel_clean" and e["column"] == "load_ratio"][0]["source_table"]
    assert "supplier_capacity" in clean_tables and "po_lines" in clean_tables
    assert "part_demand_weekly" in [e for e in reg if e["family"] == "fwd_load"][0]["source_table"]
    # a flag feature no family registers is UNCOVERED (the gate can fire)
    import tempfile
    prod = json.load(open(LG.PRODUCT)); prod["flag"]["models"][0]["features"] = list(feats) + ["brand_new_feature"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(prod, f); tmp = f.name
    reg2 = LG.build_registry(tmp); os.unlink(tmp)
    assert [e["column"] for e in reg2 if e["family"] == "UNREGISTERED"] == ["brand_new_feature"]
    print(f"registry: PASS ({len(reg)} entries, {len(fam)} families)")


def test_family_poison_path():
    """The poisoned-recompute path on synthetic sources: rows recorded after t0 get random values with recorded_ts kept;
    the as-of cadence features do not move; an offender that reads by created_ts instead of recorded_ts DOES move."""
    import pandas as pd, cadence as CA
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2022-06-06"); n = 400; NCH = 6
    created = t0 - pd.to_timedelta(rng.integers(-60, 300, n), unit="D")
    rec = created + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
    pol = pd.DataFrame(dict(po_line_id=[f"POL{i:09d}" for i in range(n)], ci=rng.integers(0, NCH, n),
                            qty_ordered=rng.integers(1, 500, n), created_ts=created, recorded_ts=rec))
    pol = pol.sort_values(["ci", "created_ts"], kind="stable").reset_index(drop=True)
    grn = pd.DataFrame(dict(po_line_id=pol.po_line_id, ci=pol.ci, qty_received=pol.qty_ordered, is_final_receipt=True,
                            event_ts=pol.created_ts + pd.Timedelta(days=30), recorded_ts=pol.created_ts + pd.Timedelta(days=33)))
    S = dict(ch=pd.DataFrame(dict(channel_id=[f"CH{i}" for i in range(NCH)])), pol=pol, grn=grn)
    P = LG._poison_frames(S, t0, rng, ("pol", "grn"))
    late = (pol.recorded_ts > t0).to_numpy()
    assert late.any() and (~late).any()
    assert (P["pol"].recorded_ts == pol.recorded_ts).all(), "recorded_ts must be kept"
    assert P["pol"].loc[~late].equals(pol.loc[~late]), "rows recorded by t0 must be untouched"
    assert (P["pol"].created_ts[late] != pol.created_ts[late]).any(), "later rows must be poisoned"
    P["pol"] = P["pol"].sort_values(["ci", "created_ts"], kind="stable").reset_index(drop=True)
    F, _ = CA.snapshot_cadence(S, t0); Fp, _ = CA.snapshot_cadence(P, t0)
    assert np.array_equal(F, Fp, equal_nan=True), "as-of cadence moved under future poison"
    # offender: the same features with the recorded_ts filter replaced by created_ts (the classic mistake) must move
    def offender(S_):
        S2 = dict(S_); S2["pol"] = S_["pol"].assign(recorded_ts=S_["pol"].created_ts)
        return CA.snapshot_cadence(S2, t0)[0]
    assert not np.array_equal(offender(S), offender(P), equal_nan=True), "the offender did not move: the test cannot fire"
    print("family poison path: PASS")


def _fake(world_rows):
    return {w: {"rows": rows} for w, rows in world_rows.items()}


def _rows(raw_fail=CP.LEAKING, clean_fail=(), extra=()):
    rows = []
    for c in PANEL15:
        rows.append(dict(panel="raw", column=c, verdict="FAIL" if c in raw_fail else "PASS", failed_tests=["poison"] if c in raw_fail else [],
                         known_fail=("raw", c) in LG.KNOWN_FAIL))
        rows.append(dict(panel="clean", column=c, verdict="FAIL" if c in clean_fail else "PASS",
                         failed_tests=["label_alignment"] if c in clean_fail else [], known_fail=False))
    rows += list(extra)
    return rows


def test_exit_codes():
    ok = _fake({"v8": _rows(), "v8w1002": _rows()})
    assert LG.exit_code(ok) == 0 and LG.findings(ok) == []
    new = _fake({"v8": _rows(raw_fail=list(CP.LEAKING) + ["revision_count"]), "v8w1002": _rows()})
    assert LG.exit_code(new) == 1 and len(LG.findings(new)) == 1
    fam = _fake({"v8": _rows(extra=[dict(panel=None, column="cad_gap_median_d", family="cadence", verdict="FAIL",
                                         failed_tests=["poison"], known_fail=False)]), "v8w1002": _rows()})
    assert LG.exit_code(fam) == 1
    unc = _fake({"v8": _rows(extra=[dict(panel=None, column="brand_new_feature", family="UNREGISTERED", verdict="UNCOVERED",
                                         failed_tests=[], known_fail=False)]), "v8w1002": _rows()})
    assert LG.exit_code(unc) == 1
    miss = _fake({"v8": _rows(raw_fail=[c for c in CP.LEAKING if c != "load_ratio"]), "v8w1002": _rows()})
    assert LG.exit_code(miss) == 2
    cl = _fake({"v8": _rows(), "v8w1002": _rows(clean_fail=["fill_rate"])})
    assert LG.exit_code(cl) == 2
    try:
        LG.known_answer(cl); raise AssertionError("known_answer must raise GuardInvalid")
    except LG.GuardInvalid as e:
        assert "v8w1002: clean fill_rate" in str(e)
    print("exit codes: PASS (0 / 1 / 1 / 1 / 2 / 2)")


def smoke_one_week(world="v8"):
    """Real data, one week: raw poison and alignment on the generator rebuild, clean poison and alignment."""
    import time, phase22_leakscan as LS, config, phase21_paths as PP
    PP.register()
    t0 = time.time()
    S = LS.Sources(world); R = LS.rebuild(S)
    meta = json.load(open(os.path.join(config.CACHE, world, "meta.json")))
    CS = CP.CsvSources(world, meta["T"]); Cc = CP.clean_columns(CS)
    weeks = LG.sample_weeks(world, S.W0, 12)[6:7]
    L = LG.line_arrays(S, CS)
    rng = np.random.default_rng(2303)
    pr = LG.poison_raw(S, R, weeks, rng); pc = LG.poison_clean(CS, Cc, weeks, rng)
    ar = LG.alignment_share(R, weeks, L, LG.ALIGN_TOL); ac = LG.alignment_share(Cc, weeks, L, LG.ALIGN_TOL)
    print(f"smoke {world} week {weeks} ({time.time() - t0:.0f}s): column  raw[poison align (literal)]  clean[poison align (literal)]")
    for c in R:
        cl = f"{pc[c][0]:.4f} {ac[c]['share']:.4f} ({ac[c]['share_literal']:.4f})" if c in Cc else "(inherits raw)"
        print(f"  {c:24s} {pr[c][0]:.4f} {ar[c]['share']:.4f} ({ar[c]['share_literal']:.4f})   {cl}   active={ar[c]['active']}")
    return dict(pr=pr, pc=pc, ar=ar, ac=ac)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    test_alignment_core(); test_feature_check(); test_registry(); test_family_poison_path(); test_exit_codes()
    print("ALL PASS")
    if a.smoke:
        smoke_one_week()
