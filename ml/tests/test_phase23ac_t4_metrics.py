"""Phase 23AC T4 -- fast synthetic tests of the metrics pack's rules (no stored prediction is read).

(iii) the always-majority constructed case: accuracy equals the baseline while precision / recall / F1 expose it (and a
      predictor that is NOT always-majority fails the same check, so the check can fire);
(iv)  ROC-AUC / PR-AUC by sklearn and by the own rank / step implementation agree to 1e-6 on random tie-heavy data, and a
      hand-computed tie case (y = [0, 0, 1, 1], s = [0.5, 0.5, 0.5, 0.9] -> ROC-AUC 0.75, PR-AUC 0.75); a deliberately wrong
      implementation makes auc_pair raise;
the "right N in 10" formatter; the accuracy-without-majority scan; the LEAKY-label rule (the table builder raises when a
leaky figure, by flag or by source path, is put in CLEAN); and the end-to-end rendering of synthetic arms.

    python ml/tests/test_phase23ac_t4_metrics.py        (pytest also works)
"""
from __future__ import annotations
import os, sys, json, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "train")]
os.environ.setdefault("HADES_DEVICE", "cpu")
import numpy as np, pandas as pd
import phase23ac_t4_metrics as T4
import phase23ac_t4_charts_data as CD

NOWHERE = "ml/artifacts/__t4_test_nonexistent__"           # synthetic sources: never on disk, so no commit lookup runs


# ------------------------------------------------------------------ (iii) always-majority
def test_always_majority_high_base():
    d = T4.constructed_always_majority(0.73)
    assert abs(d["accuracy"] - d["majority_accuracy"]) < 1e-12
    assert abs(d["precision"] - d["base_rate"]) < 1e-12 and d["recall"] == 1.0 and abs(d["lift"] - 1.0) < 1e-12
    return T4.check_always_majority(d)


def test_always_majority_low_base():
    d = T4.constructed_always_majority(0.245)
    assert abs(d["accuracy"] - d["majority_accuracy"]) < 1e-12
    assert d["recall"] == 0.0 and d["alerts"] == 0 and d["f1"] == 0.0 and np.isnan(d["precision"])
    return T4.check_always_majority(d)


def test_always_majority_check_can_fail():
    """Flag-everything at a 0.245 base is NOT the majority predictor: its accuracy is the base, not the baseline."""
    rng = np.random.default_rng(1); y = (rng.random(5000) < 0.245).astype(int)
    d = T4.at_threshold(np.ones(5000), y, 0.5)
    try:
        T4.check_always_majority(d)
    except AssertionError:
        return "fired"
    raise AssertionError("the always-majority check passed a non-majority predictor")


# ------------------------------------------------------------------ (iv) AUC two methods
def test_auc_hand_tie_case():
    y, s = np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.9])
    assert abs(T4.roc_auc_rank(y, s) - 0.75) < 1e-12
    assert abs(T4.pr_auc_step(y, s) - 0.75) < 1e-12
    r = T4.auc_pair(y, s)
    assert all(abs(r[k] - 0.75) < 1e-12 for k in ("roc_auc", "roc_auc_rank", "pr_auc", "pr_auc_step")), r
    return r


def test_auc_random_agreement():
    rng = np.random.default_rng(7); worst = 0.0
    for _ in range(60):
        n = int(rng.integers(20, 4000)); y = (rng.random(n) < rng.uniform(0.02, 0.95)).astype(int)
        if y.min() == y.max():
            continue
        s = np.round(rng.normal(0, 1, n) + y * rng.uniform(-0.5, 2.5), int(rng.integers(0, 3)))
        r = T4.auc_pair(y, s)
        worst = max(worst, abs(r["roc_auc"] - r["roc_auc_rank"]), abs(r["pr_auc"] - r["pr_auc_step"]))
    assert worst <= 1e-6, worst
    assert T4.check_auc_methods()["max_abs_diff"] <= 1e-6
    return worst


def test_auc_disagreement_raises():
    """A tie-blind ROC (strict > only) differs on the hand case (0.5 vs 0.75): auc_pair must raise."""
    def tie_blind(y, s):
        y, s = np.asarray(y), np.asarray(s, float); p, q = s[y == 1], s[y == 0]
        return float((p[:, None] > q[None, :]).mean())
    saved = T4.roc_auc_rank
    T4.roc_auc_rank = tie_blind
    try:
        T4.auc_pair(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.9]))
    except AssertionError:
        return "fired"
    finally:
        T4.roc_auc_rank = saved
    raise AssertionError("auc_pair accepted two disagreeing methods")


# ------------------------------------------------------------------ formatter, accuracy scan
def test_right_in_10():
    assert T4.right_in_10(0.969, 0.730) == "right 9.7 times in 10 at the flagged cases against 7.3 in 10 by chance"
    assert T4.right_in_10(0.736, 0.41) == "right 7.4 times in 10 at the flagged cases against 4.1 in 10 by chance"
    assert T4.right_in_10(float("nan"), 0.3).startswith("no flagged cases")
    assert T4.right_in_10(None, 0.3).startswith("no flagged cases")
    return T4.right_in_10(0.969, 0.730)


def test_accuracy_scan():
    assert T4.check_accuracy_lines("accuracy 0.731\nUC1 accuracy 0.731 vs majority 0.730\nprecision 0.9") == ["accuracy 0.731"]
    assert T4.check_accuracy_lines("Accuracy at max-F1 vs Majority 0.75") == []
    return "ok"


# ------------------------------------------------------------------ the LEAKY-label rule
def _rec():
    return dict(v=0.9, lo=0.88, hi=0.92, n=1, interval=T4.BLOCK_IV)


def test_leaky_rule():
    T = T4.PackTable()
    kw = dict(arm="a", arm_label="x", role="headline", status="RECOMPUTED", commits=["c"])
    T.add("UC1", "cov.5%.precision", "INCUMBENT", _rec(), sources=["ml/artifacts/bundles/arrival_week/v8_lite_h4_lr0.00025_s7/preds_test.npz"],
          leaky=True, **kw)
    assert T.rows[-1]["label"] == T4.LEAKY_LABEL
    T.add("UC1", "cov.5%.precision", "CLEAN", _rec(),
          sources=["ml/artifacts/phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7/preds_test.npz"], leaky=False, **kw)
    assert T.rows[-1]["label"] == "clean"
    raised = []
    for leaky, src in ((True, "ml/artifacts/phase22/bundles/arrival_week/v8clean_x/preds_test.npz"),     # by flag
                       ("unconfirmed", "reports/part2/phase-17.md"),                                    # LEAKY-UNCONFIRMED
                       (False, "ml/artifacts/phase19/bundles/fill_rate/v8_none_h0_lr0.000125_s7_rfseason+cadence/preds_test.npz"),
                       (False, "ml/artifacts/phase18/preds/v8_fill_rate_p18_fwd_load_s7_test.npz"),     # by source path
                       (False, "ml/artifacts/phase7_preds/v8_fill_rate_lgbm22_id_s7_test.npz"),
                       (False, "ml/artifacts/bundles/capacity_strain/v8_mp_h4_lr0.00025_s7/preds_test.npz"),
                       (False, "ml/artifacts/phase22/capacity_published.json")):
        try:
            T.add("UC1", "m", "CLEAN", _rec(), sources=[src], leaky=leaky, **kw)
        except T4.LeakyInCleanColumn:
            raised.append(src)
    assert len(raised) == 7, f"the builder accepted a leaky figure in CLEAN: {raised}"
    for bad in (dict(column="NEW"), dict(status="GUESSED")):
        try:
            T.add("UC1", "m", bad.get("column", "INCUMBENT"), _rec(), sources=["x"], leaky=True,
                  **{**kw, "status": bad.get("status", "RECOMPUTED")})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert not T4.is_published_source("ml/artifacts/phase22/preds/v8clean_fill_rate_p22_sc_ack_L4_s7_test.npz")
    assert not T4.is_published_source("ml/artifacts/phase22/capacity_clean.json")
    return f"{len(raised)} leaky placements refused"


# ------------------------------------------------------------------ synthetic arms end to end
def _synthetic_pair(rng, base, n=6000, signal=1.2):
    snaps = np.array(pd.date_range("2025-01-06", periods=6, freq="6W").values)
    def fold():
        y = (rng.random(n) < base).astype(int); s = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + signal * (y - 0.5))))
        return np.round(s, 3), y
    sv, yv = fold(); st, yt = fold()
    return sv, yv, st, yt, snaps[np.arange(n) % len(snaps)]


def test_class_metrics_consistency():
    rng = np.random.default_rng(3)
    sv, yv, st, yt, _ = _synthetic_pair(rng, 0.4)
    m = T4.class_metrics(sv, yv, st, yt)
    for c in T4.COVS:
        p = T4.pct(c)
        assert m[f"cov.{p}.TP"] + m[f"cov.{p}.FP"] + m[f"cov.{p}.FN"] + m[f"cov.{p}.TN"] == len(yt)
    assert abs(m["majority_accuracy"] - max(yt.mean(), 1 - yt.mean())) < 1e-12
    assert m["maxf1.majority_accuracy"] == m["majority_accuracy"]
    assert all(m[f"bar.{p:.2f}.status"] in ("reached on validation", "UNREACHABLE on validation") for p in T4.BARS)
    weak = T4.class_metrics(*_synthetic_pair(rng, 0.2, signal=0.1)[:4])
    assert weak["bar.0.90.status"] == "UNREACHABLE on validation", "a near-random score reached p = 0.90"
    band = T4.band_flat([T4.class_metrics(*_synthetic_pair(np.random.default_rng(k), 0.2, signal=0.1)[:4]) for k in range(5)])
    assert band["bar.0.90.status"]["s"].startswith("UNREACHABLE on validation ("), band["bar.0.90.status"]
    assert set(band["cov.5%.precision"]) >= {"v", "lo", "hi", "n"} and band["cov.5%.precision"]["n"] == 5
    return dict(p5=m["cov.5%.precision"], roc=m["roc_auc"])


def test_end_to_end_synthetic_render():
    rng = np.random.default_rng(11)
    seed_pairs = [_synthetic_pair(rng, 0.73) for _ in T4.SEEDS]
    inc = T4.seed_class_arm("uc1_inc", "UC1", "INCUMBENT", "headline", "synthetic incumbent", seed_pairs, True, [NOWHERE + "/a_s{s}.npz"])
    det = _synthetic_pair(rng, 0.73, signal=1.5)
    cln = T4.det_class_arm("uc1_cln", "UC1", "CLEAN", "headline", "synthetic clean ensemble", det, False, [NOWHERE + "/b.npz"], 30)
    r = cln["metrics"]["cov.5%.precision"]
    assert r["interval"] == T4.BLOCK_IV and r["lo"] <= r["hi"], r
    assert inc["metrics"]["cov.5%.precision"]["interval"] == T4.SEED_IV
    arms = [inc, cln]
    T = T4.populate(arms)
    assert set(T.frame()["column"]) == {"INCUMBENT", "CLEAN"}
    md = "\n".join(T4.md_use_case(arms, "UC1"))
    assert "vs majority" in md and T4.check_accuracy_lines(md) == [], T4.check_accuracy_lines(md)[:3]
    assert T4.LEAKY_LABEL in md and "right " in md
    prog = T4.md_progress(arms, {}, [], dict(code_version="test", run_ts="now"))
    assert T4.check_accuracy_lines(prog) == []
    summary = prog.split("## d.")[1].strip().splitlines()[2:]
    assert 0 < len([s for s in summary if s.strip()]) <= 10
    bad = dict(cln, leaky=True)
    try:
        T4.populate([bad])
    except T4.LeakyInCleanColumn:
        pass
    else:
        raise AssertionError("populate accepted a leaky arm in CLEAN")
    pay = T4.chart_payload(arms)
    assert {h["column"] for h in pay["headline"]} == {"INCUMBENT", "CLEAN"}
    return dict(clean_p5=r["v"], rows=len(T.rows))


# ------------------------------------------------------------------ small pieces
def test_dist_interval_and_pinball():
    F = np.array([0.05, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6])
    S = np.tile(1 - F, (4, 1))                                  # CDF reaches 0.1 at week 2, never 0.9 -> open upper end
    o = T4.dist_interval(S, np.array([1, 2, 30, 5.0]), np.array([True, True, True, False]))
    assert abs(o["interval.coverage"] - 2 / 3) < 1e-12 and o["interval.open_upper_share"] == 1.0
    Q = np.array([[0.5, 1.0, 1.5]]); p = T4.pinball(Q, np.array([2.0]))
    assert abs(p["pinball.q10"] - 0.15) < 1e-12 and abs(p["pinball.q50"] - 0.5) < 1e-12 and abs(p["pinball.q90"] - 0.45) < 1e-12
    return "ok"


def test_chart_data_helpers():
    d = tempfile.mkdtemp()
    t1 = os.path.join(d, "compare_v8.json")
    json.dump({"overall": {"product": {"a3_days": 10.0, "share_within_7d": 0.41, "signed_bias_days": -1.2},
                           "promise_date": {"a3_days": 14.0, "share_within_7d": 0.3, "signed_bias_days": 5.0}},
               "per_month": {"1": {"product": {"a3_days": 9.0}}}}, open(t1, "w"))
    rows = CD.chart4_rows(t1)
    assert [r["estimator"] for r in rows] == ["product", "promise_date"], rows
    assert CD.chart4_rows(os.path.join(d, "missing.json")) is None
    ls = os.path.join(d, "leakscan.json"); rs = os.path.join(d, "restate.json")
    cols = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52", "lead_time_actual_days", "lead_time_ratio",
            "otd_rate_last13", "ack_gap_ratio", "load_ratio"]
    json.dump({"leaking_H_week": cols, "table": {c: {"changes_H_week": True, "share_channels_changed_H_week_mean": 0.5} for c in cols}},
              open(ls, "w"))
    blk = lambda c, p, lo, hi: {"clean": c, "published": p, "delta_clean_minus_published": c - p, "block_ci_better_positive": [lo, 0, hi],
                                "verdict": "x"}
    json.dump({"tasks": {"arrival": {"leak_delta": {"neural": {"block": {"uc1_precision_at_5pct": blk(0.948, 0.969, -0.034, -0.009),
                                                                         "lateness_auc": blk(0.70, 0.71, -0.02, -0.005),
                                                                         "a3_median_abs_err_days": blk(13.1, 13.07, -0.27, 0.16)}}}},
                         "fill": {"leak_delta": {"neural": {"block": {"p_full_auc": blk(0.60, 0.62, -0.04, -0.0002),
                                                                      "uc2b_precision_at_5pct": blk(0.40, 0.46, -0.08, -0.01),
                                                                      "crps_exact": blk(0.1396, 0.1385, -0.003, 0.0006)}}}},
                         "capacity": {"leak_delta": {"neural": {"block": {"precision_at_5pct": blk(0.757, 0.865, -0.21, -0.02)}}}}}},
              open(rs, "w"))
    c5, drop = CD.chart5_rows(ls, rs)
    assert len(c5) == 9
    crps = [r for r in drop if r["metric"] == "crps_exact"][0]
    assert abs(crps["lo"] - (-0.0006)) < 1e-12 and abs(crps["hi"] - 0.003) < 1e-12, crps      # lower-is-better restated
    out = CD.build_all(dict(headline=[dict(use_case="UC1", metric="cov.5%.precision", column="CLEAN", value=0.9)],
                            precision_vs_base=[dict(use_case="UC1", column="CLEAN", precision_at_5pct=0.9, base_rate=0.73)],
                            capacity_snapshots=[dict(column="CLEAN", arm="x", values={"2025-01-06": 0.5, "2025-12-08": 0.45})],
                            t1_path=os.path.join(d, "missing.json"), leakscan_path=ls, restate_path=rs), os.path.join(d, "data"))
    assert str(out["chart4"]).startswith("SKIPPED")
    c3 = pd.read_csv(out["chart3"]); assert c3.loc[c3["is_worst"] == 1, "snapshot"].tolist() == ["2025-12-08"]
    return "ok"


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    for n in names:
        r = globals()[n]()
        print(f"  PASS {n}: {r}", flush=True)
    print(f"{len(names)} tests passed")
