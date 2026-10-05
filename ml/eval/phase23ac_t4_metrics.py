"""Phase 23AC Track T4 -- the presentation metrics pack: INCUMBENT (Phase 15) -> BEST PUBLISHED (Phases 19-20, leaky
inputs) -> CLEAN (Phase 22), one question per use case. Definitions: reports/part2/phase23ac/t4/definitions.md (written
first). Torch process (imports only), CPU, no LightGBM, nothing fitted: every number is RECOMPUTED from stored predictions
or QUOTED from a named report / artifact, and is marked as one or the other with its source and commit.

Reused unchanged (read-only): phase15 (pick_on_val, apply), phase20_decisions (summarise, prec_at, accuracy_at_val_f1,
reachable_and_class), phase21_score (uc1_label, uc1p_label), phase22_restate (rc, test_snaps, register), phase18_score
(expected_week, point), phase14_score (p_late, p_exceed), phase5_metrics (crps_exact_rows, ece_marginal), loop
(apply_recalibration of a STORED recalibration.json).

Rules (pre-registration T4):
  operating points   Phase 15 pick_on_val on VALIDATION, bars 0.70 / 0.80 / 0.85 / 0.90; accuracy at the validation
                     max-F1 threshold (Phase 20 rule 3), never printed without the majority-class accuracy
  intervals          single-model arms: 5-seed mean [min, max] (all five seeds required); ensembles / blends: snapshot-block
                     bootstrap, 1,000 resamples of whole test snapshots, fixed seed, thresholds held at their validation value
  AUCs               sklearn and an independent rank / step implementation, asserted equal to 1e-6 on every AUC computed
  leaky figures      only in INCUMBENT / BEST PUBLISHED, labelled "LEAKY (superseded, Phase 22)"; PackTable.add raises if a
                     leaky figure (by flag or by source path) is put in CLEAN
Printed sanity checks: (i) recomputed vs quoted at the published precision (mismatch = FINDING, listed, never smoothed);
(ii) accuracy never without the majority baseline (the writer scans its own markdown and raises); (iii) the constructed
always-majority predictor; (iv) AUC / PR-AUC two-method agreement plus a hand-computed tie-heavy case.

  python ml/eval/phase23ac_t4_metrics.py [--B 1000]
    -> reports/part2/phase23ac/t4/{metrics_pack.md, metrics_pack.csv, progress.md, charts/data/*.csv}
       ml/artifacts/phase23ac/t4/{metrics.json, metrics.log}
"""
from __future__ import annotations
import os, sys, json, math, argparse, subprocess, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import phase12_common as C
import phase23ac_t4_charts_data as CD

SEEDS = C.V8_SEEDS
FOLDS = ("val", "test")
BARS = (0.70, 0.80, 0.85, 0.90)
COVS = (0.01, 0.05, 0.10, 0.20)
B_DEFAULT = 1000
BOOT_SEED = 2304
AUC_TOL = 1e-6
COLUMNS = CD.COLUMNS
LEAKY_LABEL = "LEAKY (superseded, Phase 22)"
UNCONF_LABEL = "LEAKY-UNCONFIRMED"
SEED_IV = "5-seed mean [min, max]"
BLOCK_IV = "point [snapshot-block 95%, 1,000 resamples]"
REP = os.path.join(C.REPO, "reports", "part2", "phase23ac", "t4")
CHARTS = os.path.join(REP, "charts")
CHART_DATA = os.path.join(CHARTS, "data")
OUTA = os.path.join(C.ART, "phase23ac", "t4")
A = C.ART
T1_JSON = os.path.join(A, "phase23ac", "t1", "compare_v8.json")

P = {  # stored predictions, exact per-seed names ({s} seed, {f} fold); no globs
    "arr_inc": os.path.join(A, "bundles", "arrival_week", "v8_lite_h4_lr0.00025_s{s}", "preds_{f}.npz"),
    "arr_p19n": os.path.join(A, "phase19", "bundles", "arrival_week", "v8_lite_h4_lr0.00025_s{s}_rffwdload+season+cadence", "preds_{f}.npz"),
    "arr_p18l": os.path.join(A, "phase18", "preds", "v8_arrival_week_p18_fwd_load_s{s}_{f}.npz"),
    "arr_cln": os.path.join(A, "phase22", "bundles", "arrival_week", "v8clean_lite_h4_lr0.00025_s{s}", "preds_{f}.npz"),
    "arr_clnl": os.path.join(A, "phase22", "preds", "v8clean_arrival_week_p22_fwdload_s{s}_{f}.npz"),
    "fil_inc": os.path.join(A, "bundles", "fill_rate", "v8_none_h0_lr0.000125_s{s}", "preds_{f}.npz"),
    "fil_bw3": os.path.join(A, "bundles", "fill_rate", "v8_none_h0_lr0.000125_s{s}_lossrps_bw3", "preds_{f}.npz"),
    "fil_id": os.path.join(A, "phase7_preds", "v8_fill_rate_lgbm22_id_s{s}_{f}.npz"),
    "fil_id_recal": os.path.join(A, "phase7_preds", "RECAL_v8_fill_rate_lgbm22_id_s{s}_{f}.npz"),
    "fil_p19n": os.path.join(A, "phase19", "bundles", "fill_rate", "v8_none_h0_lr0.000125_s{s}_rfseason+cadence", "preds_{f}.npz"),
    "fil_p18l": os.path.join(A, "phase18", "preds", "v8_fill_rate_p18_fwd_load_s{s}_{f}.npz"),
    "fil_cln": os.path.join(A, "phase22", "bundles", "fill_rate", "v8clean_none_h0_lr0.000125_s{s}_rfseason+cadence", "preds_{f}.npz"),
    "fil_cons": os.path.join(A, "phase22", "preds", "v8clean_fill_rate_p22_sc_ack_L4_s{s}_{f}.npz"),
    "fil_clfw": os.path.join(A, "phase22", "preds", "v8clean_fill_rate_p22_fwdload_s{s}_{f}.npz"),
    "cap_inc": os.path.join(A, "bundles", "capacity_strain", "v8_mp_h4_lr0.00025_s{s}", "preds_{f}.npz"),
    "cap_cln": os.path.join(A, "phase22", "bundles", "capacity_strain", "v8clean_mp_h4_lr0.00025_s{s}", "preds_{f}.npz"),
    "flag": os.path.join(A, "phase22", "preds", "v8clean_arrival_place_p22_flag_lag1_k10_s{s}_{f}.npz"),
}
J = {  # stored JSON artifacts read for weights and QUOTED figures
    "restate": os.path.join(A, "phase22", "restate_arrival_fill_capacity.json"),
    "fill_final": os.path.join(A, "phase22", "fill_final.json"),
    "cap_clean": os.path.join(A, "phase22", "capacity_clean.json"),
    "cap_pub": os.path.join(A, "phase22", "capacity_published.json"),
    "order_time": os.path.join(A, "phase22", "order_time_v8clean.json"),
    "leakscan": os.path.join(A, "phase22", "leakscan_v8.json"),
    "w19": os.path.join(A, "phase19", "stage4b_block.json"),
    "place_store": os.path.join(A, "phase21", "grpstats_v8_place.npz"),
}
DOC = {k: os.path.join(C.REPO, v) for k, v in {
    "p15": "reports/part2/phase-15.md", "p20": "reports/part2/phase20/stage1_decisions.md", "p22": "reports/part2/phase-22.md",
    "p22s2": "reports/part2/phase22/stage2_order_time.md", "p22s3": "reports/part2/phase22/stage3_fill.md",
    "p22s4": "reports/part2/phase22/stage4_capacity.md", "p21": "reports/part2/phase-21.md", "p17": "reports/part2/phase-17.md",
    "p13": "reports/part2/phase-13.md", "obs2": "results/observation2.md", "ask": "docs/client/data_request_v2.md",
    "defs": "reports/part2/phase23ac/t4/definitions.md"}.items()}
# published weights the brief names; read from the artifacts and compared (a difference is a FINDING, not silently used)
BRIEF_WEIGHTS = {"arrival_published_w_neural": 0.52, "fill_published_w_neural": 0.50, "arrival_clean_w_neural": 0.52,
                 "fill_consolidated_w_neural": 0.02, "fill_p19_clean_w_neural": 0.49}

_LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.append(s)


# ================================================================== repo modules (imported lazily, read-only use)
class _Mods:
    pass


_M = None


def mods():
    global _M
    if _M is None:
        m = _Mods()
        import config, phase5_metrics, metrics, heads, loop
        import phase15, phase20_decisions, phase21_score, phase22_restate, phase18_score, phase14_score, phase20_score
        from phase11b_lateness import lateness
        m.config, m.M5, m.MT, m.heads, m.L = config, phase5_metrics, metrics, heads, loop
        m.P15, m.P20D, m.S21, m.R22, m.S18, m.S14, m.P20S = (phase15, phase20_decisions, phase21_score, phase22_restate,
                                                            phase18_score, phase14_score, phase20_score)
        m.lateness = lateness
        _M = m
    return _M


def register_worlds():
    M = mods()
    M.R22.register()                                   # phase21_paths.register() + v8clean / v8w1002clean world entries
    M.config.WORLDS["v8clean"] = M.config.WORLDS["v8"]
    assert M.config.WORLDS["v8clean"] == M.config.WORLDS["v8"]


# ================================================================== core arithmetic (pure numpy; tested synthetically)
def _midranks(s):
    s = np.asarray(s, float); n = len(s)
    o = np.argsort(s, kind="mergesort"); ss = s[o]
    b = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1, n]
    avg = (b[:-1] + b[1:] - 1) / 2.0 + 1.0              # 1-based mean rank of each tie group
    r = np.empty(n); r[o] = np.repeat(avg, np.diff(b))
    return r


def roc_auc_rank(y, s):
    """Mann-Whitney U / (P N) with mid-ranks (ties count one half). Independent of sklearn."""
    y = np.asarray(y).astype(int); Pn = int(y.sum()); Nn = len(y) - Pn
    if Pn == 0 or Nn == 0:
        return float("nan")
    r = _midranks(s)
    return float((r[y == 1].sum() - Pn * (Pn + 1) / 2.0) / (Pn * Nn))


def pr_auc_step(y, s):
    """Average precision as a step function over DISTINCT score thresholds: sum_k (R_k - R_{k-1}) P_k."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); Pn = int(y.sum())
    if Pn == 0 or Pn == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort"); ss, yy = s[o], y[o]
    tp = np.cumsum(yy); n = np.arange(1, len(yy) + 1)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    prec = tp[last] / n[last]; rec = tp[last] / Pn
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


_AUC_LEDGER = dict(n=0, max_abs_diff=0.0)


def auc_pair(y, s, tol=AUC_TOL):
    """ROC-AUC and PR-AUC by sklearn and by the rank / step implementation; raises if they disagree beyond tol."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    if y.min() == y.max():
        return dict(roc_auc=float("nan"), roc_auc_rank=float("nan"), pr_auc=float("nan"), pr_auc_step=float("nan"))
    out = dict(roc_auc=float(roc_auc_score(y, s)), roc_auc_rank=roc_auc_rank(y, s),
               pr_auc=float(average_precision_score(y, s)), pr_auc_step=pr_auc_step(y, s))
    d = max(abs(out["roc_auc"] - out["roc_auc_rank"]), abs(out["pr_auc"] - out["pr_auc_step"]))
    if not d <= tol:
        raise AssertionError(f"AUC methods disagree by {d:.3g} (> {tol}): {out}")
    _AUC_LEDGER["n"] += 1; _AUC_LEDGER["max_abs_diff"] = max(_AUC_LEDGER["max_abs_diff"], d)
    return out


def confusion(flag, y):
    flag = np.asarray(flag, bool); y = np.asarray(y).astype(int)
    TP = int((flag & (y == 1)).sum()); FP = int((flag & (y == 0)).sum())
    FN = int((~flag & (y == 1)).sum()); TN = int((~flag & (y == 0)).sum())
    n = len(y); pos = TP + FN; m = TP + FP; base = pos / n if n else float("nan")
    prec = TP / m if m else float("nan")
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, alerts=m, precision=prec, recall=TP / pos if pos else float("nan"),
                f1=2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0, accuracy=(TP + TN) / n if n else float("nan"),
                majority_accuracy=max(base, 1 - base), share_flagged=m / n if n else float("nan"),
                lift=prec / base if (m and base) else float("nan"), base_rate=base)


def at_threshold(st, yt, tau):
    return confusion(np.asarray(st, float) >= tau, yt)


def right_in_10(precision, base):
    """'right N times in 10 at the flagged cases against M in 10 by chance' (one decimal each)."""
    if precision is None or base is None or not np.isfinite(precision) or not np.isfinite(base):
        return "no flagged cases: precision undefined"
    return f"right {10 * precision:.1f} times in 10 at the flagged cases against {10 * base:.1f} in 10 by chance"


def pct(c):
    return f"{int(round(c * 100))}%"


KEYS = ("precision", "recall", "f1", "accuracy", "majority_accuracy", "share_flagged", "lift", "alerts", "TP", "FP", "FN", "TN")


def class_metrics(sv, yv, st, yt):
    """Every classification metric of one predictor (flat keys). Thresholds from VALIDATION only."""
    M = mods()
    sv, st = np.asarray(sv, float), np.asarray(st, float); yv, yt = np.asarray(yv).astype(int), np.asarray(yt).astype(int)
    n = len(yt); pos = int(yt.sum()); base = pos / n
    out = dict(n=n, positives=pos, base_rate=base, majority_accuracy=max(base, 1 - base),
               majority_class="positive (flag everything)" if base >= 0.5 else "negative (flag nothing)")
    out.update(auc_pair(yt, st))
    order = np.argsort(-st, kind="stable")
    for c in COVS:
        m = max(1, int(np.ceil(c * n))); top = np.zeros(n, bool); top[order[:m]] = True
        d = confusion(top, yt)
        assert abs(d["precision"] - M.P20D.prec_at(st, yt, c)) < 1e-12, "precision at coverage disagrees with Phase 20's prec_at"
        for k in ("precision", "recall", "lift", "alerts", "share_flagged", "TP", "FP", "FN", "TN"):
            out[f"cov.{pct(c)}.{k}"] = d[k]
    acc = M.P20D.accuracy_at_val_f1(sv, yv, st, yt)
    tau = acc["threshold_val_max_f1"]; d = at_threshold(st, yt, tau)
    assert abs(d["accuracy"] - acc["test_accuracy"]) < 1e-12 and abs(d["majority_accuracy"] - acc["majority_accuracy"]) < 1e-12
    out["maxf1.tau"] = tau
    out.update({f"maxf1.{k}": d[k] for k in KEYS})
    for p in BARS:
        key = f"bar.{p:.2f}"
        tau_p, maxp = M.P15.pick_on_val(sv, yv, p)
        if tau_p is None:
            out[key + ".status"] = "UNREACHABLE on validation"; out[key + ".val_max_precision"] = maxp
            continue
        d = at_threshold(st, yt, tau_p); ap = M.P15.apply(st, yt, tau_p)
        assert ap["alerts"] == d["alerts"] and ap["TP"] == d["TP"], "apply() and the confusion counts disagree"
        out[key + ".status"] = "reached on validation"; out[key + ".tau"] = tau_p
        out.update({f"{key}.{k}": d[k] for k in KEYS})
    return out


def class_boot(sv, yv, st, yt, draws):
    """Block-bootstrap intervals of the threshold-free and fixed-threshold metrics of one deterministic predictor."""
    M = mods()
    sv, st = np.asarray(sv, float), np.asarray(st, float); yv, yt = np.asarray(yv).astype(int), np.asarray(yt).astype(int)
    tau_f1 = M.P20D.accuracy_at_val_f1(sv, yv, st, yt)["threshold_val_max_f1"]
    taus = {p: M.P15.pick_on_val(sv, yv, p)[0] for p in BARS}

    def fn(i):
        s, y = st[i], yt[i]; o = {}
        if 0 < y.sum() < len(y):
            o["roc_auc"] = roc_auc_score(y, s); o["pr_auc"] = average_precision_score(y, s)
        o["base_rate"] = y.mean()
        for c in COVS:
            o[f"cov.{pct(c)}.precision"] = M.P20D.prec_at(s, y, c)
        d = at_threshold(s, y, tau_f1)
        for k in ("precision", "recall", "f1", "accuracy", "share_flagged"):
            o[f"maxf1.{k}"] = d[k]
        for p, t in taus.items():
            if t is not None:
                d = at_threshold(s, y, t)
                o[f"bar.{p:.2f}.precision"] = d["precision"]; o[f"bar.{p:.2f}.recall"] = d["recall"]
        return o
    return boot_ci(fn, draws)


def block_draws(snap, B, seed=BOOT_SEED):
    snap = np.asarray(snap); u = np.unique(snap)
    blocks = [np.flatnonzero(snap == x) for x in u]
    rng = np.random.default_rng(seed)
    return [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(B)]


def boot_ci(fn, draws):
    vals = [fn(i) for i in draws]
    keys = list(dict.fromkeys(k for v in vals for k in v))
    out = {}
    for k in keys:
        x = np.array([v.get(k, np.nan) for v in vals], float); x = x[np.isfinite(x)]
        if len(x) >= 0.9 * len(draws):
            out[k] = (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return out


def _num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def band_flat(per, seeds=SEEDS):
    """Per-seed flat dicts -> records: numbers {v: mean, lo: min, hi: max, n}; strings kept (or listed per seed)."""
    keys = list(dict.fromkeys(k for p in per for k in p))
    out = {}
    for k in keys:
        vals = [p.get(k) for p in per]
        if all(v is None or _num(v) for v in vals):
            x = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
            out[k] = (dict(v=float(np.mean(x)), lo=float(min(x)), hi=float(max(x)), n=len(x), interval=SEED_IV) if x
                      else dict(s="n/a"))
        else:
            u = list(dict.fromkeys(str(v) for v in vals))
            if k.endswith(".status") and u != ["reached on validation"]:
                r = sum(v == "reached on validation" for v in vals)
                out[k] = dict(s=f"UNREACHABLE on validation ({r} of {len(vals)} seeds reach)")
            else:
                out[k] = dict(s=u[0] if len(u) == 1 else "; ".join(f"s{sd}: {v}" for sd, v in zip(seeds, vals)))
    return out


def det_flat(flat, ci=None):
    out = {}
    for k, v in flat.items():
        if _num(v):
            if ci and k in ci:
                out[k] = dict(v=float(v), lo=ci[k][0], hi=ci[k][1], n=1, interval=BLOCK_IV)
            else:
                out[k] = dict(v=float(v), lo=None, hi=None, n=1, interval="point (no interval)")
        else:
            out[k] = dict(s=str(v))
    return out


# ================================================================== constructed failing cases (run by main and the tests)
def constructed_always_majority(base, n=20000, seed=0):
    """An always-majority predictor (flags everything when positives are the majority, nothing otherwise)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base).astype(int)
    maj_pos = y.mean() >= 0.5
    s = np.ones(n)
    d = at_threshold(s, y, 0.5 if maj_pos else np.inf)
    return d


def check_always_majority(d):
    """-> verdict string; raises if accuracy differs from the baseline or neither precision = base nor recall = 0."""
    assert abs(d["accuracy"] - d["majority_accuracy"]) < 1e-12, "always-majority accuracy is not the baseline"
    if d["base_rate"] >= 0.5:
        assert abs(d["precision"] - d["base_rate"]) < 1e-12 and d["recall"] == 1.0 and abs(d["lift"] - 1) < 1e-12
        return (f"base {d['base_rate']:.3f}: accuracy {d['accuracy']:.3f} = majority {d['majority_accuracy']:.3f}; "
                f"precision {d['precision']:.3f} = base rate, lift 1.00 (exposed)")
    assert d["recall"] == 0.0 and d["alerts"] == 0 and d["f1"] == 0.0
    return (f"base {d['base_rate']:.3f}: accuracy {d['accuracy']:.3f} = majority {d['majority_accuracy']:.3f}; "
            f"recall 0, F1 0, no flagged case (exposed)")


def check_auc_methods(seed=0):
    rng = np.random.default_rng(seed); out = {}
    hand = auc_pair(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.9]))
    assert abs(hand["roc_auc"] - 0.75) < 1e-12 and abs(hand["roc_auc_rank"] - 0.75) < 1e-12, hand
    assert abs(hand["pr_auc"] - 0.75) < 1e-12 and abs(hand["pr_auc_step"] - 0.75) < 1e-12, hand
    out["hand_tie_case"] = hand
    worst = 0.0
    for i in range(20):
        n = int(rng.integers(50, 3000)); y = (rng.random(n) < rng.uniform(0.05, 0.9)).astype(int)
        s = np.round(rng.normal(0, 1, n) + y * rng.uniform(0, 2), int(rng.integers(0, 3)))      # heavy ties by rounding
        r = auc_pair(y, s)
        worst = max(worst, abs(r["roc_auc"] - r["roc_auc_rank"]), abs(r["pr_auc"] - r["pr_auc_step"]))
    out["random_cases"] = 20; out["max_abs_diff"] = worst
    return out


def check_accuracy_lines(text):
    """-> offending lines: any line that mentions accuracy without the majority baseline."""
    return [ln for ln in text.splitlines() if "accuracy" in ln.lower() and "majority" not in ln.lower()]


# ================================================================== leaky-label rule and the table
class LeakyInCleanColumn(ValueError):
    pass


PUBLISHED_MARKERS = (os.sep + os.path.join("bundles", "arrival_week", "v8_"), os.sep + os.path.join("bundles", "fill_rate", "v8_"),
                     os.sep + os.path.join("bundles", "capacity_strain", "v8_"), os.sep + "phase7_preds" + os.sep,
                     os.sep + os.path.join("phase18", "preds") + os.sep, os.sep + os.path.join("phase19", "bundles") + os.sep,
                     "capacity_published.json")


def is_published_source(path):
    p = str(path).replace("/", os.sep)
    if not p.startswith(os.sep):
        p = os.sep + p
    return any(m in p for m in PUBLISHED_MARKERS)


def leak_label(leaky):
    return LEAKY_LABEL if leaky is True else UNCONF_LABEL if leaky == "unconfirmed" else "clean"


class PackTable:
    """Long table of every figure. The CLEAN column refuses a leaky figure (flag or source path)."""

    def __init__(self):
        self.rows = []

    def add(self, uc, metric, column, rec, *, arm, arm_label, role, status, sources, commits, leaky):
        if column not in COLUMNS:
            raise ValueError(f"unknown column {column!r}")
        if status not in ("RECOMPUTED", "QUOTED"):
            raise ValueError(f"status must be RECOMPUTED or QUOTED, got {status!r}")
        if not sources:
            raise ValueError(f"{uc} {metric}: every figure carries its source")
        if column == "CLEAN" and (leaky is not False or any(is_published_source(s) for s in sources)):
            raise LeakyInCleanColumn(f"{uc} / {metric}: a leaky figure ({arm}, sources {sources}) cannot be placed in CLEAN")
        self.rows.append(dict(use_case=uc, metric=metric, column=column, arm=arm, arm_label=arm_label, role=role,
                              value=rec.get("v", rec.get("s")), lo=rec.get("lo"), hi=rec.get("hi"),
                              interval=rec.get("interval", ""), n=rec.get("n"), status=status,
                              label=leak_label(leaky), source="; ".join(rel(s) for s in sources),
                              commit="; ".join(commits)))

    def frame(self):
        return pd.DataFrame(self.rows)


# ================================================================== provenance
def rel(p):
    p = str(p)
    return os.path.relpath(p, C.REPO) if os.path.isabs(p) else p


_COMMITS = {}


def commit_of(path):
    path = str(path)
    if path in _COMMITS:
        return _COMMITS[path]
    c = None
    try:
        if path.endswith(".json") and os.path.exists(path):
            c = (json.load(open(path)).get("stamp") or {}).get("code_commit")
        if c is None:
            cfg = os.path.join(os.path.dirname(path), "config.json")
            if path.endswith(".npz") and os.path.exists(cfg):
                c = (json.load(open(cfg)).get("stamps") or {}).get("code_commit")
        if c is None and os.path.exists(path):
            c = subprocess.check_output(["git", "-C", C.REPO, "log", "-1", "--format=%h", "--", path], text=True,
                                        stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        c = None
    _COMMITS[path] = c or "untracked artifact (no stamp)"
    return _COMMITS[path]


def commits_of(pattern):
    if "{s}" not in pattern:
        return [commit_of(pattern)]
    return sorted({commit_of(pattern.format(s=s, f="test")) for s in SEEDS})


# ================================================================== loading (all five seeds required)
def seeds_load(key):
    fmt = P[key]
    missing = [fmt.format(s=s, f=f) for s in SEEDS for f in FOLDS if not os.path.exists(fmt.format(s=s, f=f))]
    if missing:
        raise SystemExit(f"scorer requires all five seeds: {len(missing)} files missing, e.g. {rel(missing[0])}")
    return [{f: dict(np.load(fmt.format(s=s, f=f))) for f in FOLDS} for s in SEEDS]


def same_rows(groups, what):
    ref = groups[0][0]
    for g in groups:
        for z in g:
            for f in FOLDS:
                assert np.array_equal(np.asarray(z[f]["Y"], float), np.asarray(ref[f]["Y"], float)), f"{what}: rows differ ({f})"
                if "EV" in z[f] and "EV" in ref[f]:
                    assert np.array_equal(np.asarray(z[f]["EV"], bool), np.asarray(ref[f]["EV"], bool)), f"{what}: EV differs ({f})"
                if "entity" in z[f] and "entity" in ref[f]:
                    assert (np.asarray(z[f]["entity"]).astype(str) == np.asarray(ref[f]["entity"]).astype(str)).all(), what


def mean_of(zs, f, k="P"):
    return np.mean([np.asarray(z[f][k], float) for z in zs], 0)


def jload(key):
    return json.load(open(J[key]))


# ================================================================== arms
def phase15_summary(pairs):
    M = mods()
    s = M.P20D.summarise({"arm": list(pairs)})["arm"]
    ps = s["per_snapshot_precision_at_5pct"]
    worst = min(ps["values"], key=ps["values"].get)
    return dict(cls=s["cls"], reachable=s["reachable"], operating_bar=s["operating_bar"], lift_at_bar=s["lift_at_bar"],
                stage_b=s["stage_b"], spread=dict(min=ps["min"], median=ps["median"], max=ps["max"], worst=worst, values=ps["values"]))


def _p15_records(summ, n):
    sp = summ["spread"]; iv = "per-snapshot (seed mean)" if n > 1 else "per-snapshot"
    return {"phase15.cls": dict(s=summ["cls"]), "phase15.reachable": dict(s=summ["reachable"]),
            "phase15.operating_bar": dict(s=str(summ["operating_bar"])),
            "phase15.lift_at_bar": (dict(v=summ["lift_at_bar"], lo=None, hi=None, n=n, interval="at the operating bar")
                                    if summ["lift_at_bar"] is not None else dict(s="n/a")),
            "phase15.stage_b": dict(s=summ["stage_b"]),
            "spread.p5.min": dict(v=sp["min"], lo=None, hi=None, n=n, interval=iv),
            "spread.p5.median": dict(v=sp["median"], lo=None, hi=None, n=n, interval=iv),
            "spread.p5.max": dict(v=sp["max"], lo=None, hi=None, n=n, interval=iv),
            "spread.p5.worst": dict(s=f"{sp['worst']} ({sp['values'][sp['worst']]:.3f})")}


def _right_records(m):
    b = m.get("base_rate", {}).get("v")
    out = {"right_in_10.cov5": dict(s=right_in_10(m.get("cov.5%.precision", {}).get("v"), b)),
           "right_in_10.maxf1": dict(s=right_in_10(m.get("maxf1.precision", {}).get("v"), b))}
    return out


def arm(id_, uc, column, role, label, kind, leaky, sources, metrics, phase15=None, extra=None):
    return dict(id=id_, uc=uc, column=column, role=role, label=label, kind=kind, leaky=leaky,
                sources=list(sources), commits=sorted({c for s in sources for c in commits_of(s)}),
                metrics=metrics, phase15=phase15, extra=extra or {})


def seed_class_arm(id_, uc, column, role, label, pairs, leaky, sources, extra_per_seed=None):
    assert len(pairs) == len(SEEDS), f"{id_}: all five seeds are required"
    per = [class_metrics(*p[:4]) for p in pairs]
    if extra_per_seed:
        for d, e in zip(per, extra_per_seed):
            d.update(e)
    m = band_flat(per)
    summ = phase15_summary(pairs)
    m.update(_p15_records(summ, len(pairs))); m.update(_right_records(m))
    log(f"  [{uc}] {column:14s} {id_:22s} P@5% {m['cov.5%.precision']['v']:.4f} [{m['cov.5%.precision']['lo']:.4f}, "
        f"{m['cov.5%.precision']['hi']:.4f}]  ROC {m['roc_auc']['v']:.4f}  class {summ['cls']} ({summ['reachable']})")
    return arm(id_, uc, column, role, label, "seed", leaky, sources, m, summ)


def det_class_arm(id_, uc, column, role, label, pair, leaky, sources, B, extra=None):
    flat = class_metrics(*pair[:4])
    if extra:
        flat.update(extra)
    ci = class_boot(*pair[:4], block_draws(pair[4], B)) if B else None
    m = det_flat(flat, ci)
    summ = phase15_summary([pair])
    m.update(_p15_records(summ, 1)); m.update(_right_records(m))
    r = m["cov.5%.precision"]
    log(f"  [{uc}] {column:14s} {id_:22s} P@5% {r['v']:.4f} [{_f(r['lo'], 4)}, {_f(r['hi'], 4)}]  ROC {m['roc_auc']['v']:.4f}  "
        f"class {summ['cls']} ({summ['reachable']})")
    return arm(id_, uc, column, role, label, "det", leaky, sources, m, summ)


def _f(x, d):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ UC1 + arrival point
def dist_interval(S, Y, EV):
    """Central 80% of the 12-week hazard distribution; the upper end is open when the CDF never reaches 0.9."""
    F = 1.0 - np.asarray(S, float)                          # F[:, k-1] = P(T <= k), k = 1..12
    def first(q):
        hit = F >= q; k = hit.argmax(1) + 1.0
        return np.where(hit.any(1), k, np.inf)
    lo, hi = first(0.1), first(0.9)
    lo = np.where(np.isinf(lo), 13.0, lo)
    Y, EV = np.asarray(Y, float), np.asarray(EV, bool)
    inside = (Y >= lo) & (Y <= hi)
    closed = np.isfinite(hi)
    return {"interval.coverage": float(inside[EV].mean()),
            "interval.width_days_median": float(np.median(7 * (hi - lo)[EV & closed])) if (EV & closed).any() else float("nan"),
            "interval.open_upper_share": float((~closed)[EV].mean())}


def arrival_point_metrics(ET, Y, EV, R, snap, S=None):
    M = mods()
    ET, Y, EV, R = np.asarray(ET, float), np.asarray(Y, float), np.asarray(EV, bool), np.asarray(R, float)
    e = 7 * (ET[EV] - Y[EV])
    out = {"a3_days": float(np.median(np.abs(e))), "mae_days": float(np.mean(np.abs(e))), "signed_bias_days": float(np.mean(e)),
           "share_within_7d": float(np.mean(np.abs(e) <= 7))}
    lat = M.lateness(ET, Y, EV, R)[0]
    m = EV & np.isfinite(R)
    r2 = roc_auc_rank((Y[m] > R[m]).astype(int), ET[m] - R[m])
    assert abs(lat - r2) <= AUC_TOL, f"lateness AUC: sklearn {lat} vs rank {r2}"
    out["lateness_auc"] = float(lat); out["lateness_auc_rank"] = r2
    out["cindex"] = float(M.MT.cindex(ET, Y, EV))
    if S is not None:
        out.update(dist_interval(S, Y, EV))
    sp = {}
    for u in np.unique(snap):
        k = (snap == u) & EV
        sp[str(pd.Timestamp(u).date())] = float(np.median(np.abs(7 * (ET[k] - Y[k]))))
    return out, sp


def arrival_point_boot(ET, Y, EV, R, draws):
    ET, Y, EV, R = np.asarray(ET, float), np.asarray(Y, float), np.asarray(EV, bool), np.asarray(R, float)
    m = EV & np.isfinite(R)
    def fn(i):
        ev = EV[i]; e = 7 * (ET[i][ev] - Y[i][ev]); mm = m[i]
        yl = (Y[i][mm] > R[i][mm]).astype(int)
        o = {"a3_days": np.median(np.abs(e)), "mae_days": np.mean(np.abs(e)), "signed_bias_days": np.mean(e),
             "share_within_7d": np.mean(np.abs(e) <= 7)}
        if 0 < yl.sum() < len(yl):
            o["lateness_auc"] = roc_auc_score(yl, ET[i][mm] - R[i][mm])
        return o
    return boot_ci(fn, draws)


def spread_records(sp, n, name):
    vals = {k: v for k, v in sp.items() if np.isfinite(v)}
    worst_hi = name in ("a3", "crps")                       # lower is better: the worst snapshot is the maximum
    w = max(vals, key=vals.get) if worst_hi else min(vals, key=vals.get)
    iv = "per-snapshot (seed mean)" if n > 1 else "per-snapshot"
    return {f"spread.{name}.min": dict(v=min(vals.values()), lo=None, hi=None, n=n, interval=iv),
            f"spread.{name}.median": dict(v=float(np.median(list(vals.values()))), lo=None, hi=None, n=n, interval=iv),
            f"spread.{name}.max": dict(v=max(vals.values()), lo=None, hi=None, n=n, interval=iv),
            f"spread.{name}.worst": dict(s=f"{w} ({vals[w]:.4g})")}


def seed_mean_spread(sps):
    return {k: float(np.mean([s[k] for s in sps])) for k in sps[0]}


def arrival_arms(B):
    M = mods()
    inc, p19n, p18l, cln, clnl = (seeds_load(k) for k in ("arr_inc", "arr_p19n", "arr_p18l", "arr_cln", "arr_clnl"))
    same_rows([inc, p19n, p18l, cln, clnl], "arrival")
    Rc = {f: M.R22.rc(f) for f in FOLDS}
    snap = M.R22.test_snaps("arrival")
    R_asof = M.P20S.ref("v8", "test")
    w_pub = float(jload("w19")["tasks"]["arrival"]["blend_weight_neural"])
    w_cln = float(jload("restate")["arrival_blends"]["clean"]["w_neural"])
    ew = M.S18.expected_week
    ens = lambda zs, f: {"S": mean_of(zs, f, "S"), "pT": mean_of(zs, f, "pT")}
    et_pub = {f: w_pub * ew(ens(p19n, f)) + (1 - w_pub) * mean_of(p18l, f) for f in FOLDS}
    et_cln = {f: w_cln * ew(ens(cln, f)) + (1 - w_cln) * mean_of(clnl, f) for f in FOLDS}

    def lab(z, f):
        return M.S21.uc1_label(np.asarray(z[f]["Y"], float), np.asarray(z[f]["EV"], bool), Rc[f])

    def pack(score, z):
        out = []
        for f in FOLDS:
            y, k = lab(z, f); out += [np.asarray(score(f), float)[k], y[k]]
        out.append(snap[lab(z, "test")[1]])
        return tuple(out)
    pl = M.S14.p_late
    uc1 = [
        seed_class_arm("uc1_inc", "UC1", "INCUMBENT", "headline", "neural lite h4, per seed (Phase 15's arm), P(late)",
                       [pack(lambda f, z=z: pl(z[f]["S"], Rc[f]), z) for z in inc], True, [P["arr_inc"]]),
        det_class_arm("uc1_bp", "UC1", "BEST PUBLISHED", "headline", "incumbent neural 5-seed ensemble P(late) (Phase 20 kept it)",
                      pack(lambda f: pl(mean_of(inc, f, "S"), Rc[f]), inc[0]), True, [P["arr_inc"]], B),
        det_class_arm("uc1_bp_blend", "UC1", "BEST PUBLISHED", "also", f"Phase 19 blend point score - contract (w_neural {w_pub})",
                      pack(lambda f: et_pub[f] - Rc[f], inc[0]), True, [P["arr_p19n"], P["arr_p18l"]], B),
        det_class_arm("uc1_cln", "UC1", "CLEAN", "headline", "clean incumbent neural 5-seed ensemble P(late)",
                      pack(lambda f: pl(mean_of(cln, f, "S"), Rc[f]), cln[0]), False, [P["arr_cln"]], B),
        seed_class_arm("uc1_cln_seed", "UC1", "CLEAN", "also", "clean incumbent neural, per seed, P(late)",
                       [pack(lambda f, z=z: pl(z[f]["S"], Rc[f]), z) for z in cln], False, [P["arr_cln"]]),
    ]
    zt = inc[0]["test"]; Y, EV = np.asarray(zt["Y"], float), np.asarray(zt["EV"], bool)
    draws = block_draws(snap, B) if B else None

    def seed_point(id_, column, role, label, zs, leaky, src):
        per, sps = [], []
        for z in zs:
            o, sp = arrival_point_metrics(ew(z["test"]), Y, EV, R_asof, snap, S=z["test"]["S"]); per.append(o); sps.append(sp)
        m = band_flat(per); m.update(spread_records(seed_mean_spread(sps), len(zs), "a3"))
        m["interval.note"] = dict(s="hazard distribution central 80% (definitions.md)")
        log(f"  [ARRIVAL] {column:14s} {id_:22s} A3 {m['a3_days']['v']:.2f} [{m['a3_days']['lo']:.2f}, {m['a3_days']['hi']:.2f}] "
            f"lateness {m['lateness_auc']['v']:.4f}")
        return arm(id_, "ARRIVAL", column, role, label, "seed", leaky, src, m)

    def det_point(id_, column, role, label, et, leaky, src):
        o, sp = arrival_point_metrics(et, Y, EV, R_asof, snap)
        ci = arrival_point_boot(et, Y, EV, R_asof, draws) if draws else None
        m = det_flat(o, ci); m.update(spread_records(sp, 1, "a3"))
        m["interval.coverage"] = dict(s="n/a (a blend has no interval)")
        log(f"  [ARRIVAL] {column:14s} {id_:22s} A3 {m['a3_days']['v']:.2f} [{_f(m['a3_days']['lo'], 2)}, {_f(m['a3_days']['hi'], 2)}] "
            f"lateness {m['lateness_auc']['v']:.4f}")
        return arm(id_, "ARRIVAL", column, role, label, "det", leaky, src, m)
    point = [
        seed_point("arr_inc", "INCUMBENT", "headline", "neural lite h4, per seed: expected week", inc, True, [P["arr_inc"]]),
        det_point("arr_bp", "BEST PUBLISHED", "headline", f"Phase 19 blend (w_neural {w_pub})", et_pub["test"], True,
                  [P["arr_p19n"], P["arr_p18l"]]),
        det_point("arr_cln", "CLEAN", "headline", f"clean Phase 19 recipe (w_neural {w_cln})", et_cln["test"], False,
                  [P["arr_cln"], P["arr_clnl"]]),
        seed_point("arr_cln_seed", "CLEAN", "also", "clean incumbent neural, per seed: expected week", cln, False, [P["arr_cln"]]),
    ]
    return uc1 + point, dict(arrival_published_w_neural=w_pub, arrival_clean_w_neural=w_cln)


# ------------------------------------------------------------------ fill
def fill_dist_metrics(Pm, Y, Prec=None):
    M = mods()
    Pm, Y = np.asarray(Pm, float), np.asarray(Y, float)
    a = auc_pair((Y >= 1).astype(int), Pm[:, 21])
    cell = M.heads.fill_cell(Y)
    return {"crps_exact": float(M.M5.crps_exact_rows(Pm, Y).mean()), "p_full_auc": a["roc_auc"], "p_full_auc_rank": a["roc_auc_rank"],
            "ece22_raw": float(M.M5.ece_marginal(Pm, cell)[0]),
            "ece22_recal": (float(M.M5.ece_marginal(np.asarray(Prec, float), cell)[0]) if Prec is not None
                            else "n/a (no stored recalibration)")}


def fill_spread(Pm, Y, snap):
    M = mods(); r = M.M5.crps_exact_rows(np.asarray(Pm, float), np.asarray(Y, float))
    return {str(pd.Timestamp(u).date()): float(r[snap == u].mean()) for u in np.unique(snap)}


def bundle_recal(key, s, z):
    M = mods()
    rec = json.load(open(os.path.join(os.path.dirname(P[key].format(s=s, f="test")), "recalibration.json")))
    return M.L.apply_recalibration(rec, "fill_rate", z)["P22"]


def fill_arms(B):
    M = mods()
    keys = ("fil_inc", "fil_bw3", "fil_id", "fil_id_recal", "fil_p19n", "fil_p18l", "fil_cln", "fil_cons", "fil_clfw")
    Z = {k: seeds_load(k) for k in keys}
    same_rows([Z[k] for k in keys], "fill")
    snap = M.R22.test_snaps("fill")
    Y = {f: np.asarray(Z["fil_inc"][0][f]["Y"], float) for f in FOLDS}
    w_pub = float(jload("w19")["tasks"]["fill"]["blend_weight_neural"])
    W = jload("fill_final")["weights"]
    w_cons, w_19 = float(W["consolidated_w_neural"]), float(W["p19_clean_w_neural"])
    Ppub = {f: w_pub * mean_of(Z["fil_p19n"], f) + (1 - w_pub) * mean_of(Z["fil_p18l"], f) for f in FOLDS}
    Pcons = {f: w_cons * mean_of(Z["fil_cln"], f) + (1 - w_cons) * mean_of(Z["fil_cons"], f) for f in FOLDS}
    Pc19 = {f: w_19 * mean_of(Z["fil_cln"], f) + (1 - w_19) * mean_of(Z["fil_clfw"], f) for f in FOLDS}
    src_pub, src_cons, src_c19 = [P["fil_p19n"], P["fil_p18l"]], [P["fil_cln"], P["fil_cons"]], [P["fil_cln"], P["fil_clfw"]]
    out = []
    for uc, sf, lf in (("UC2", lambda x: x[:, 21], lambda y: (y >= 1).astype(int)),
                       ("UC2b", lambda x: x[:, :20].sum(1), lambda y: (y < 0.95).astype(int))):
        yv, yt = lf(Y["val"]), lf(Y["test"])
        seedp = lambda zs: [(sf(np.asarray(z["val"]["P"], float)), yv, sf(np.asarray(z["test"]["P"], float)), yt, snap) for z in zs]
        detp = lambda Pd: (sf(Pd["val"]), yv, sf(Pd["test"]), yt, snap)
        if uc == "UC2":
            out += [seed_class_arm("uc2_inc", uc, "INCUMBENT", "headline", "boundary bw3, per seed (Phase 15's UC2 arm)",
                                   seedp(Z["fil_bw3"]), True, [P["fil_bw3"]]),
                    det_class_arm("uc2_bp", uc, "BEST PUBLISHED", "headline", f"Phase 19 fill blend (w_neural {w_pub})",
                                  detp(Ppub), True, src_pub, B),
                    det_class_arm("uc2_cln", uc, "CLEAN", "headline", f"consolidated blend (w_neural {w_cons})", detp(Pcons), False,
                                  src_cons, B),
                    det_class_arm("uc2_cln_p19", uc, "CLEAN", "also", f"clean Phase 19 fill blend (w_neural {w_19})", detp(Pc19),
                                  False, src_c19, B)]
        else:
            out += [seed_class_arm("uc2b_inc", uc, "INCUMBENT", "headline", "lgbm22_id, per seed (Phase 15's UC2b arm)",
                                   seedp(Z["fil_id"]), True, [P["fil_id"]]),
                    det_class_arm("uc2b_bp", uc, "BEST PUBLISHED", "headline", f"Phase 19 fill blend (w_neural {w_pub})",
                                  detp(Ppub), True, src_pub, B),
                    det_class_arm("uc2b_cln", uc, "CLEAN", "headline", f"clean Phase 19 fill blend (w_neural {w_19})", detp(Pc19),
                                  False, src_c19, B),
                    det_class_arm("uc2b_cln_cons", uc, "CLEAN", "also", f"consolidated blend (w_neural {w_cons})", detp(Pcons),
                                  False, src_cons, B)]
    # ---- the fill distribution
    draws = block_draws(snap, B) if B else None
    yt = Y["test"]

    def seed_dist(id_, column, role, label, key, recal, leaky):
        per, sps = [], []
        for s, z in zip(SEEDS, Z[key]):
            Pr = (bundle_recal(key, s, z["test"]) if recal == "bundle" else
                  np.asarray(Z["fil_id_recal"][SEEDS.index(s)]["test"]["P"], float) if recal == "RECAL_" else None)
            per.append(fill_dist_metrics(z["test"]["P"], yt, Pr)); sps.append(fill_spread(z["test"]["P"], yt, snap))
        m = band_flat(per); m.update(spread_records(seed_mean_spread(sps), 5, "crps"))
        src = [P[key]] + ([P["fil_id_recal"]] if recal == "RECAL_" else [])
        log(f"  [FILL] {column:14s} {id_:22s} CRPS {m['crps_exact']['v']:.5f} [{m['crps_exact']['lo']:.5f}, {m['crps_exact']['hi']:.5f}] "
            f"AUC {m['p_full_auc']['v']:.4f}")
        return arm(id_, "FILL", column, role, label, "seed", leaky, src, m)

    def det_dist(id_, column, role, label, Pd, leaky, src):
        o = fill_dist_metrics(Pd, yt)
        rows = mods().M5.crps_exact_rows(Pd, yt); full = (yt >= 1).astype(int)
        ci = boot_ci(lambda i: {"crps_exact": rows[i].mean(),
                                **({"p_full_auc": roc_auc_score(full[i], Pd[i][:, 21])} if 0 < full[i].sum() < len(i) else {})},
                     draws) if draws else None
        m = det_flat(o, ci); m.update(spread_records(fill_spread(Pd, yt, snap), 1, "crps"))
        log(f"  [FILL] {column:14s} {id_:22s} CRPS {m['crps_exact']['v']:.5f} [{_f(m['crps_exact']['lo'], 5)}, {_f(m['crps_exact']['hi'], 5)}]")
        return arm(id_, "FILL", column, role, label, "det", leaky, src, m)
    out += [seed_dist("fill_inc", "INCUMBENT", "headline", "neural none_h0, per seed (Phase 15 fill incumbent)", "fil_inc", "bundle", True),
            seed_dist("fill_inc_bw3", "INCUMBENT", "also", "boundary bw3, per seed (UC2 arm)", "fil_bw3", "bundle", True),
            seed_dist("fill_inc_id", "INCUMBENT", "also", "lgbm22_id, per seed (UC2b arm)", "fil_id", "RECAL_", True),
            det_dist("fill_bp", "BEST PUBLISHED", "headline", f"Phase 19 fill blend (w_neural {w_pub})", Ppub["test"], True, src_pub),
            det_dist("fill_cln", "CLEAN", "headline", f"consolidated blend (w_neural {w_cons})", Pcons["test"], False, src_cons),
            seed_dist("fill_cln_nsc", "CLEAN", "also", "clean neural + season + cadence, per seed (ships as the distribution)",
                      "fil_cln", "bundle", False),
            det_dist("fill_cln_p19", "CLEAN", "also", f"clean Phase 19 fill blend (w_neural {w_19})", Pc19["test"], False, src_c19)]
    return out, dict(fill_published_w_neural=w_pub, fill_consolidated_w_neural=w_cons, fill_p19_clean_w_neural=w_19)


# ------------------------------------------------------------------ capacity
def pinball(Q, y):
    Q, y = np.asarray(Q, float), np.asarray(y, float); out = {}
    for j, q in enumerate((0.1, 0.5, 0.9)):
        d = y - Q[:, j]; out[f"pinball.q{int(q * 100)}"] = float(np.mean(np.maximum(q * d, (q - 1) * d)))
    out["pinball.mean"] = float(np.mean([out["pinball.q10"], out["pinball.q50"], out["pinball.q90"]]))
    return out


def raw_interval(Q, y, snap):
    Q, y = np.asarray(Q, float), np.asarray(y, float)
    inside = (y >= Q[:, 0]) & (y <= Q[:, 2])
    q = pd.PeriodIndex(pd.to_datetime(snap), freq="Q").astype(str).to_numpy()
    per = {u: float(inside[q == u].mean()) for u in sorted(set(q))}
    return {"interval_raw.coverage": float(inside.mean()), "interval_raw.quarter_min": min(per.values()),
            "interval_raw.quarter_mean": float(np.mean(list(per.values()))), "interval_raw.quarter_max": max(per.values())}


def capacity_arms(B):
    M = mods()
    inc, cln = seeds_load("cap_inc"), seeds_load("cap_cln")
    same_rows([inc, cln], "capacity")
    snap = M.R22.test_snaps("capacity")
    pe = M.S14.p_exceed
    lab = lambda z, f: (np.asarray(z[f]["Y"], float) > 1).astype(int)
    pair = lambda Pv, Pt, z: (pe(np.asarray(Pv, float)), lab(z, "val"), pe(np.asarray(Pt, float)), lab(z, "test"), snap)
    yt = np.asarray(inc[0]["test"]["Y"], float)
    extra = lambda Q: {**pinball(Q, yt), **raw_interval(Q, yt, snap)}
    ens = lambda zs, f: mean_of(zs, f)
    return [
        seed_class_arm("uc3_inc", "UC3", "INCUMBENT", "headline", "mp h4, per seed (Phase 15's arm)",
                       [pair(z["val"]["P"], z["test"]["P"], z) for z in inc], True, [P["cap_inc"]],
                       [extra(z["test"]["P"]) for z in inc]),
        det_class_arm("uc3_bp", "UC3", "BEST PUBLISHED", "headline", "incumbent 5-seed ensemble (mean quantiles)",
                      pair(ens(inc, "val"), ens(inc, "test"), inc[0]), True, [P["cap_inc"]], B, extra(ens(inc, "test"))),
        seed_class_arm("uc3_cln", "UC3", "CLEAN", "headline", "clean mp h4, per seed",
                       [pair(z["val"]["P"], z["test"]["P"], z) for z in cln], False, [P["cap_cln"]],
                       [extra(z["test"]["P"]) for z in cln]),
        det_class_arm("uc3_cln_ens", "UC3", "CLEAN", "also", "clean 5-seed ensemble (mean quantiles)",
                      pair(ens(cln, "val"), ens(cln, "test"), cln[0]), False, [P["cap_cln"]], B, extra(ens(cln, "test"))),
    ]


# ------------------------------------------------------------------ order time (strict flag)
def tau_month(entities):
    z = np.load(J["place_store"])
    ent, tau = z["entity"].astype(str), z["tau"].astype(str)
    idx = pd.Index(ent)
    assert idx.is_unique, "placement store: entity ids are not unique"
    pos = idx.get_indexer(np.asarray(entities).astype(str))
    assert (pos >= 0).all(), "a flag row has no placement-store entry"
    return pd.DatetimeIndex(pd.to_datetime(tau[pos])).to_period("M").to_timestamp().values


def order_time_arms(B):
    M = mods()
    zs = seeds_load("flag")
    same_rows([zs], "order-time flag")
    month = tau_month(zs[0]["test"]["entity"])
    pairs, extra = [], []
    for z in zs:
        r = []
        for f in FOLDS:
            Y, EV, R = np.asarray(z[f]["Y"], float), np.asarray(z[f]["EV"], bool), np.asarray(z[f]["AUX"], float)
            y, k = M.S21.uc1p_label(Y, EV, R); r += [np.asarray(z[f]["P"], float)[k], y[k]]
            if f == "test":
                r.append(month[k])
                a = auc_pair((Y[EV] > R[EV]).astype(int), np.asarray(z[f]["P"], float)[EV])
                extra.append({"lateness_auc": a["roc_auc"], "lateness_auc_rank": a["roc_auc_rank"]})
        pairs.append(tuple(r))
    return [seed_class_arm("ot_flag_cln", "ORDER-TIME FLAG", "CLEAN", "headline", "strict as-of late flag (flag_lag1), per seed",
                           pairs, False, [P["flag"]], extra)]


# ================================================================== QUOTED figures
def quoted_arms():
    out = []
    ot = jload("order_time")
    km = ot["date_estimators"]["test"]["km"]; iv = ot["interval"]
    ci_a3 = ot["date_block"]["ci"]["a3_median_abs_err_days"].get("km")
    m = {"a3_days": dict(v=km["a3_median_abs_err_days"], lo=ci_a3[0] if ci_a3 else None, hi=ci_a3[1] if ci_a3 else None, n=1,
                         interval="point [creation-week block 95%]" if ci_a3 else "point"),
         "lateness_auc": dict(v=km["lateness_auc"], lo=None, hi=None, n=1, interval="point"),
         "week_hit_rate": dict(v=km["week_hit_rate"], lo=None, hi=None, n=1, interval="point"),
         "interval.coverage": dict(v=iv["test_coverage"], lo=iv["coverage_block_ci"][0], hi=iv["coverage_block_ci"][1], n=1,
                                   interval="point [creation-week block 95%]"),
         "interval.width_days_mean": dict(v=iv["test_width_days_mean"], lo=None, hi=None, n=1, interval="point"),
         "interval.month_coverage_min": dict(v=min(iv["test_coverage_by_month"].values()), lo=None, hi=None, n=1, interval="per creation month"),
         "interval.month_coverage_max": dict(v=max(iv["test_coverage_by_month"].values()), lo=None, hi=None, n=1, interval="per creation month"),
         "interval.month_width_min": dict(v=min(iv["test_width_by_month"].values()), lo=None, hi=None, n=1, interval="per creation month"),
         "interval.month_width_max": dict(v=max(iv["test_width_by_month"].values()), lo=None, hi=None, n=1, interval="per creation month")}
    src = [J["order_time"]]
    t1 = CD.chart4_rows(T1_JSON)
    prod = [r for r in (t1 or []) if "product" in r["json_path"].lower()]
    if prod:
        r = prod[0]
        for k, mk in (("share_within_7d", "share_within_7d"), ("signed_bias_days", "signed_bias_days")):
            m[mk] = dict(v=r[k], lo=None, hi=None, n=1, interval="point (Track T1)") if r[k] is not None else dict(s="PENDING T1")
        src.append(T1_JSON)
    else:
        m["share_within_7d"] = dict(s="PENDING T1"); m["signed_bias_days"] = dict(s="PENDING T1")
    out.append(dict(id="ot_date_cln", uc="ORDER-TIME DATE", column="CLEAN", role="headline", kind="quoted", leaky=False,
                    label="product date: shrunk-KM median (k = 10) + offset; month-specific split-conformal 80% interval",
                    sources=src, commits=sorted({commit_of(s) for s in src}), metrics=m, phase15=None, extra={}))
    out.append(dict(id="ot_flag_bp", uc="ORDER-TIME FLAG", column="BEST PUBLISHED", role="headline", kind="quoted", leaky=True,
                    label="Phase 21 at-placement flag (BASE_nl; still carried six leaking columns, deviation 210)",
                    sources=[DOC["p21"]], commits=[commit_of(DOC["p21"])],
                    metrics={"cov.5%.precision": dict(v=0.79, lo=None, hi=None, n=5, interval="QUOTED (phase-21.md l. 49)"),
                             "base_rate": dict(v=0.35, lo=None, hi=None, n=5, interval="QUOTED"),
                             "phase15.cls": dict(s="ALERT"), "phase15.lift_at_bar": dict(v=2.2, lo=None, hi=None, n=5, interval="QUOTED"),
                             "right_in_10.cov5": dict(s=right_in_10(0.79, 0.35))}, phase15=None, extra={}))
    for key, column, leaky in (("cap_pub", "INCUMBENT", True), ("cap_clean", "CLEAN", False)):
        c = jload(key)["conformal_level_aware"]; q = c["per_quarter"]
        out.append(dict(id=f"uc3_conformal_{key}", uc="UC3", column=column, role="headline", kind="quoted", leaky=leaky,
                        label="Phase 18 level-aware conformal 80% interval (Phase 22 Stage 4)", sources=[J[key]],
                        commits=[commit_of(J[key])],
                        metrics={"interval_conformal.quarter_min": dict(v=min(q.values()), lo=None, hi=None, n=1, interval="per quarter"),
                                 "interval_conformal.quarter_mean": dict(v=float(np.mean(list(q.values()))), lo=None, hi=None, n=1,
                                                                         interval="per quarter"),
                                 "interval_conformal.quarter_max": dict(v=max(q.values()), lo=None, hi=None, n=1, interval="per quarter"),
                                 "interval_conformal.worst_quarter": dict(s=min(q, key=q.get))}, phase15=None, extra={}))
    return out


OTHER = [  # use cases with no clean model metric; one line each, QUOTED
    ("Shortage head (retired)", "0.224 precision @ 5% coverage at a 3% base (re-weighted; 0.223 subsampled)", "RETIRED (not re-measured)",
     UNCONF_LABEL, "p17", "§1.1"),
    ("Predict-the-rescue", "0.823 [0.822, 0.824] precision at recall 0.246, base 0.4535 (LightGBM B1a)", "PENDING PHASE 23B",
     UNCONF_LABEL, "p17", "§2.1"),
    ("Shortage simulation", "1.129x [1.121, 1.139] vs the pre-rescue reference; detector precision ceiling 0.28",
     "PENDING PHASE 23B", UNCONF_LABEL, "p13", "S1; phase-15.md §1 (UC5a 0.284)"),
    ("Delivery schedule (MILP)", "no model metric: blocked. Needs holding, ordering, freight and shortage costs; 0 of 4 exist",
     "blocked", "n/a", "obs2", "§3.6"),
    ("Supplier allocation", "no model metric: blocked. 57 of 120 part-plants have no allowed split; on the 63 feasible it ties "
     "the status quo", "blocked", "n/a", "obs2", "§3.6"),
    ("Transfer recommendation", "no model metric: blocked. Its confidence is NOT TUNABLE (lift 0.03); \"always yes\" beats it on F1",
     "blocked", "n/a", "obs2", "§3.6"),
]


# ================================================================== recomputed vs quoted (published precision)
REFS = [  # (arm id, metric key, quoted value, decimals or None for a string, source key, where)
    ("uc1_inc", "base_rate", 0.730, 3, "p15", "§1"), ("uc1_inc", "bar.0.85.precision", 0.869, 3, "p15", "§1"),
    ("uc1_inc", "bar.0.85.recall", 0.396, 3, "p15", "§1"), ("uc1_inc", "bar.0.70.precision", 0.731, 3, "p15", "§1"),
    ("uc1_inc", "bar.0.70.recall", 0.999, 3, "p15", "§1"), ("uc1_inc", "bar.0.80.precision", 0.825, 3, "p15", "§1"),
    ("uc1_inc", "bar.0.80.recall", 0.584, 3, "p15", "§1"), ("uc1_inc", "bar.0.90.precision", 0.910, 3, "p15", "§1"),
    ("uc1_inc", "bar.0.90.recall", 0.244, 3, "p15", "§1"), ("uc1_inc", "cov.5%.precision", 0.968, 3, "p20", "decision table"),
    ("uc1_inc", "phase15.cls", "WATCHLIST", None, "p20", "decision table"),
    ("uc1_bp", "cov.1%.precision", 0.991, 3, "p20", "decision table"), ("uc1_bp", "cov.5%.precision", 0.969, 3, "p20", "decision table"),
    ("uc1_bp", "cov.10%.precision", 0.952, 3, "p20", "decision table"), ("uc1_bp", "cov.20%.precision", 0.912, 3, "p20", "decision table"),
    ("uc1_bp_blend", "cov.5%.precision", 0.943, 3, "p20", "decision table"),
    ("uc1_cln", "cov.5%.precision", 0.948, 3, "p22", "§1"), ("uc1_cln", "phase15.cls", "WATCHLIST", None, "p22", "§1"),
    ("uc2_inc", "base_rate", 0.749, 3, "p15", "§1"), ("uc2_inc", "bar.0.85.recall", 0.475, 3, "p15", "§1"),
    ("uc2_inc", "bar.0.85.precision", 0.821, 3, "p15", "§1"), ("uc2_inc", "bar.0.80.recall", 0.885, 3, "p15", "§1"),
    ("uc2_inc", "bar.0.80.precision", 0.774, 3, "p15", "§1"),
    ("uc2_bp", "cov.1%.precision", 0.961, 3, "p20", "decision table"), ("uc2_bp", "cov.5%.precision", 0.927, 3, "p20", "decision table"),
    ("uc2_bp", "bar.0.85.recall", 0.627, 3, "p20", "decision table"), ("uc2_bp", "bar.0.85.precision", 0.827, 3, "p20", "decision table"),
    ("uc2_cln", "cov.5%.precision", 0.944, 3, "p22", "§1"), ("uc2_cln_p19", "cov.5%.precision", 0.927, 3, "p22s3", "fill_final clean_uc2_p5"),
    ("uc2b_inc", "base_rate", 0.245, 3, "p15", "§1"), ("uc2b_inc", "bar.0.70.status", "UNREACHABLE on validation (0 of 5 seeds reach)",
                                                            None, "p15", "§1"),
    ("uc2b_bp", "cov.1%.precision", 0.644, 3, "p20", "decision table"), ("uc2b_bp", "cov.5%.precision", 0.530, 3, "p20", "decision table"),
    ("uc2b_cln", "cov.5%.precision", 0.511, 3, "p22", "§1"), ("uc2b_cln_cons", "cov.5%.precision", 0.461, 3, "p22", "§1"),
    ("uc3_inc", "base_rate", 0.405, 3, "p15", "§1"), ("uc3_inc", "cov.1%.precision", 0.904, 3, "p20", "anchors"),
    ("uc3_inc", "cov.5%.precision", 0.851, 3, "p20", "anchors"), ("uc3_inc", "cov.10%.precision", 0.810, 3, "p20", "anchors"),
    ("uc3_inc", "bar.0.85.recall", 0.191, 3, "p15", "§1"), ("uc3_inc", "bar.0.85.precision", 0.810, 3, "p15", "§1"),
    ("uc3_inc", "bar.0.70.recall", 0.515, 3, "p15", "§1"), ("uc3_inc", "bar.0.70.precision", 0.681, 3, "p15", "§1"),
    ("uc3_inc", "bar.0.80.recall", 0.318, 3, "p15", "§1"), ("uc3_inc", "bar.0.80.precision", 0.762, 3, "p15", "§1"),
    ("uc3_inc", "phase15.cls", "ALERT", None, "p20", "decision table"),
    ("uc3_bp", "cov.1%.precision", 0.907, 3, "p20", "decision table"), ("uc3_bp", "cov.5%.precision", 0.865, 3, "p20", "decision table"),
    ("uc3_bp", "cov.10%.precision", 0.815, 3, "p20", "decision table"), ("uc3_bp", "spread.p5.min", 0.456, 3, "p20", "per-snapshot table"),
    ("uc3_cln", "cov.1%.precision", 0.828, 3, "p22s4", "first table"), ("uc3_cln", "cov.5%.precision", 0.735, 3, "p22", "§1"),
    ("uc3_cln", "cov.10%.precision", 0.679, 3, "p22s4", "first table"), ("uc3_cln", "bar.0.70.recall", 0.308, 3, "p22s4", "first table"),
    ("uc3_cln", "bar.0.80.recall", 0.116, 3, "p22s4", "first table"), ("uc3_cln", "phase15.cls", "WATCHLIST", None, "p22", "§1"),
    ("arr_inc", "lateness_auc", 0.7090, 4, "p22", "§1"), ("arr_inc", "a3_days", 13.06, 2, "p22", "§1"),
    ("arr_cln_seed", "lateness_auc", 0.6951, 4, "p22", "§1"), ("arr_cln_seed", "a3_days", 13.15, 2, "p22", "§1"),
    ("arr_bp", "a3_days", 12.34, 2, "p20", "'What changes'"), ("arr_cln", "a3_days", 12.57, 2, "p22", "§1"),
    ("arr_cln", "lateness_auc", 0.7217, 4, "p22", "§1"),
    ("fill_inc", "crps_exact", 0.13876, 5, "p22", "§1 (published neural CRPS)"), ("fill_inc", "p_full_auc", 0.6204, 4, "p22", "§1"),
    ("fill_bp", "crps_exact", 0.1354, 4, "p22", "§1"), ("fill_cln", "crps_exact", 0.13444, 5, "p22", "§1"),
    ("fill_cln_nsc", "crps_exact", 0.1375, 4, "p22", "§1"), ("fill_cln_nsc", "p_full_auc", 0.640, 3, "p22", "§1"),
    ("fill_cln_p19", "crps_exact", 0.1358, 4, "p22", "§6 errata"),
    ("ot_flag_cln", "cov.5%.precision", 0.615, 3, "p22s2", "first table"), ("ot_flag_cln", "base_rate", 0.349, 3, "p22s2", "header"),
    ("ot_flag_cln", "lateness_auc", 0.6228, 4, "p22s2", "first table"), ("ot_flag_cln", "phase15.cls", "WATCHLIST", None, "p22s2", "first table"),
]


def get_metric(arms_by_id, aid, key):
    a = arms_by_id.get(aid)
    if a is None or key not in a["metrics"]:
        return None
    r = a["metrics"][key]
    return r.get("v", r.get("s"))


def recompute_vs_quoted(arms_by_id, weights):
    rows, findings = [], []
    for aid, key, q, d, doc, where in REFS:
        v = get_metric(arms_by_id, aid, key)
        if v is None:
            ok = False; shown = "NOT COMPUTED"
        elif d is None:
            ok = str(v) == str(q); shown = str(v)
        else:
            ok = _num(v) and abs(float(v) - q) <= 0.5 * 10 ** (-d) + 1e-12; shown = f"{float(v):.{d + 2}f}" if _num(v) else str(v)
        r = dict(arm=aid, metric=key, quoted=q, recomputed=shown, decimals=d, source=f"{rel(DOC[doc])} {where}",
                 verdict="MATCH" if ok else "FINDING")
        rows.append(r)
        if not ok:
            findings.append(r)
    for k, v in BRIEF_WEIGHTS.items():
        got = weights.get(k)
        ok = got is not None and abs(got - v) < 1e-9
        r = dict(arm="weights", metric=k, quoted=v, recomputed=str(got), decimals=2, source="brief / stored JSON",
                 verdict="MATCH" if ok else "FINDING")
        rows.append(r)
        if not ok:
            findings.append(r)
    return rows, findings


# ================================================================== table population
def populate(arms):
    T = PackTable()
    for a in arms:
        for k, rec in a["metrics"].items():
            T.add(a["uc"], k, a["column"], rec, arm=a["id"], arm_label=a["label"], role=a["role"],
                  status="QUOTED" if a["kind"] == "quoted" else "RECOMPUTED", sources=a["sources"], commits=a["commits"],
                  leaky=a["leaky"])
    return T


# ================================================================== markdown
def _dec(key):
    if key.startswith(("roc_auc", "pr_auc", "lateness_auc", "p_full_auc", "cindex")):
        return 4
    if key.startswith(("crps", "ece", "pinball")):
        return 5
    if key.endswith(("_days", "days_median", "days_mean")) or "width" in key or key.startswith(("spread.a3", "a3")):
        return 2
    if key.endswith((".TP", ".FP", ".FN", ".TN", ".alerts")) or key in ("n", "positives"):
        return 0
    if key.endswith(".lift") or key.endswith("lift_at_bar"):
        return 2
    if key.startswith("spread.crps"):
        return 5
    return 3


def fmt_rec(rec, key, majority=None, kind="seed"):
    if rec is None:
        return "—"
    if "s" in rec:
        return str(rec["s"])
    d = _dec(key)
    if key.endswith((".TP", ".FP", ".FN", ".TN", ".alerts")) and rec.get("n", 1) > 1:
        d = 1
    v = rec["v"]
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    s = f"{v:.{d}f}"
    if rec.get("lo") is not None and rec.get("hi") is not None:
        s += f" [{rec['lo']:.{d}f}, {rec['hi']:.{d}f}]"
    if key.endswith(".accuracy"):
        s += f" vs majority {majority:.3f}" if majority is not None else " vs majority n/a"
    return s


def class_display():
    rows = [("base_rate", "base rate (positive class)"), ("majority_class", "majority-class predictor"),
            ("majority_accuracy", "majority-class accuracy (the baseline)"),
            ("roc_auc", "ROC-AUC (sklearn)"), ("roc_auc_rank", "ROC-AUC (own rank implementation)"),
            ("pr_auc", "PR-AUC (sklearn; floor = base rate)"), ("pr_auc_step", "PR-AUC (own step implementation)"),
            ("lateness_auc", "lateness AUC (vs contract, uncensored)")]
    for c in COVS:
        p_ = pct(c)
        rows += [(f"cov.{p_}.precision", f"precision @ {p_} coverage"), (f"cov.{p_}.lift", f"lift @ {p_}")]
    rows += [("cov.5%.recall", "recall @ 5%"), ("cov.5%.TP", "TP @ 5%"), ("cov.5%.FP", "FP @ 5%"), ("cov.5%.FN", "FN @ 5%"),
             ("cov.5%.TN", "TN @ 5%"), ("right_in_10.cov5", "plain ratio @ 5%"),
             ("maxf1.tau", "validation max-F1 threshold"), ("maxf1.precision", "precision @ val max-F1"),
             ("maxf1.recall", "recall @ val max-F1"), ("maxf1.f1", "F1 @ val max-F1"),
             ("maxf1.accuracy", "accuracy @ val max-F1 (vs majority)"), ("maxf1.share_flagged", "share flagged @ val max-F1"),
             ("maxf1.lift", "lift @ val max-F1"), ("maxf1.TP", "TP @ val max-F1"), ("maxf1.FP", "FP @ val max-F1"),
             ("maxf1.FN", "FN @ val max-F1"), ("maxf1.TN", "TN @ val max-F1")]
    for p in BARS:
        b = f"bar.{p:.2f}"
        rows += [(f"{b}.status", f"bar p = {p:.2f}: status"), (f"{b}.precision", f"bar {p:.2f}: test precision"),
                 (f"{b}.recall", f"bar {p:.2f}: test recall"), (f"{b}.share_flagged", f"bar {p:.2f}: share flagged"),
                 (f"{b}.lift", f"bar {p:.2f}: lift"), (f"{b}.accuracy", f"bar {p:.2f}: accuracy (vs majority)"),
                 (f"{b}.TP", f"bar {p:.2f}: TP"), (f"{b}.FP", f"bar {p:.2f}: FP"), (f"{b}.FN", f"bar {p:.2f}: FN"),
                 (f"{b}.TN", f"bar {p:.2f}: TN"), (f"{b}.val_max_precision", f"bar {p:.2f}: max validation precision")]
    rows += [("phase15.cls", "**Phase 15 class**"), ("phase15.reachable", "REACHABLE"), ("phase15.operating_bar", "operating bar"),
             ("phase15.lift_at_bar", "lift at the operating bar"), ("phase15.stage_b", "Stage B"),
             ("spread.p5.min", "per-snapshot precision @ 5%: min"), ("spread.p5.median", "per-snapshot precision @ 5%: median"),
             ("spread.p5.max", "per-snapshot precision @ 5%: max"), ("spread.p5.worst", "worst snapshot")]
    return rows


DISPLAY = {
    "UC1": class_display(), "UC2": class_display(), "UC2b": class_display(), "ORDER-TIME FLAG": class_display(),
    "UC3": class_display() + [("pinball.q10", "pinball loss q0.1"), ("pinball.q50", "pinball loss q0.5"),
                              ("pinball.q90", "pinball loss q0.9"), ("pinball.mean", "pinball loss, mean"),
                              ("interval_raw.coverage", "raw P10-P90 coverage (all test)"),
                              ("interval_raw.quarter_min", "raw P10-P90 coverage, worst quarter"),
                              ("interval_raw.quarter_mean", "raw P10-P90 coverage, mean of quarters"),
                              ("interval_raw.quarter_max", "raw P10-P90 coverage, best quarter"),
                              ("interval_conformal.quarter_min", "level-aware conformal 80%: worst quarter"),
                              ("interval_conformal.quarter_mean", "level-aware conformal 80%: mean of quarters"),
                              ("interval_conformal.quarter_max", "level-aware conformal 80%: best quarter"),
                              ("interval_conformal.worst_quarter", "level-aware conformal: worst quarter")],
    "ARRIVAL": [("a3_days", "A3: median abs error (days)"), ("mae_days", "MAE (days)"), ("signed_bias_days", "signed bias (days, + = late)"),
                ("share_within_7d", "share within ±7 days"), ("cindex", "C-index"), ("lateness_auc", "lateness AUC (sklearn)"),
                ("lateness_auc_rank", "lateness AUC (own rank implementation)"), ("interval.coverage", "80% interval coverage"),
                ("interval.width_days_median", "80% interval width, median (days)"),
                ("interval.open_upper_share", "share with an open upper end (> 12 weeks)"),
                ("spread.a3.min", "per-snapshot A3: best"), ("spread.a3.median", "per-snapshot A3: median"),
                ("spread.a3.max", "per-snapshot A3: worst"), ("spread.a3.worst", "worst snapshot")],
    "ORDER-TIME DATE": [("a3_days", "A3: median abs error (days)"), ("share_within_7d", "share within ±7 days"),
                        ("signed_bias_days", "signed bias (days)"), ("week_hit_rate", "expected-week hit rate"),
                        ("lateness_auc", "lateness AUC of the date"), ("interval.coverage", "80% interval coverage"),
                        ("interval.width_days_mean", "80% interval width, mean (days)"),
                        ("interval.month_coverage_min", "coverage, worst creation month"),
                        ("interval.month_coverage_max", "coverage, best creation month"),
                        ("interval.month_width_min", "width, narrowest month (days)"),
                        ("interval.month_width_max", "width, widest month (days)")],
    "FILL": [("crps_exact", "exact CRPS (lower = better)"), ("p_full_auc", "P(fill = 1) AUC (sklearn)"),
             ("p_full_auc_rank", "P(fill = 1) AUC (own rank implementation)"), ("ece22_raw", "ECE-22 RAW"),
             ("ece22_recal", "ECE-22 RECALIBRATED (stored recalibration only)"), ("spread.crps.min", "per-snapshot CRPS: best"),
             ("spread.crps.median", "per-snapshot CRPS: median"), ("spread.crps.max", "per-snapshot CRPS: worst"),
             ("spread.crps.worst", "worst snapshot")],
}
UC_ORDER = ["UC1", "ARRIVAL", "ORDER-TIME DATE", "ORDER-TIME FLAG", "UC2", "UC2b", "FILL", "UC3"]
UC_TITLE = {"UC1": "UC1: arrival late vs contract (snapshot rows; base 0.73)",
            "ARRIVAL": "Arrival point estimate (snapshot rows)",
            "ORDER-TIME DATE": "Order-time arrival: expected date and 80% interval (Phase 22 product)",
            "ORDER-TIME FLAG": "Order-time arrival: strict as-of late flag (base 0.35)",
            "UC2": "UC2: fill arrives in full (base 0.749)", "UC2b": "UC2b: fill materially short, < 0.95 (base 0.245)",
            "FILL": "Fill distribution", "UC3": "UC3: capacity strain > 1 in the next 90 days (base 0.405)"}
HEADLINE = {"UC1": ("cov.5%.precision", True), "ARRIVAL": ("a3_days", False), "ORDER-TIME DATE": ("a3_days", False),
            "ORDER-TIME FLAG": ("cov.5%.precision", True), "UC2": ("cov.5%.precision", True), "UC2b": ("cov.5%.precision", True),
            "FILL": ("crps_exact", False), "UC3": ("cov.5%.precision", True)}
CLASS_UC = ("UC1", "ORDER-TIME FLAG", "UC2", "UC2b", "UC3")


def headline_arm(arms, uc, column):
    c = [a for a in arms if a["uc"] == uc and a["column"] == column and a["role"] == "headline" and a["kind"] != "quoted"]
    q = [a for a in arms if a["uc"] == uc and a["column"] == column and a["role"] == "headline" and a["kind"] == "quoted"]
    return c[0] if c else (q[0] if q else None)


def col_head(column):
    return f"{column} ({LEAKY_LABEL})" if column != "CLEAN" else "CLEAN"


def _lookup(arms, uc, column, key):
    """Headline arm first; QUOTED headline rows of the same use case and column fill keys the recomputed arm lacks."""
    for a in [a for a in arms if a["uc"] == uc and a["column"] == column and a["role"] == "headline"]:
        if key in a["metrics"]:
            return a["metrics"][key], a
    return None, None


def md_use_case(arms, uc):
    L_ = [f"## {UC_TITLE[uc]}", ""]
    mine = [a for a in arms if a["uc"] == uc]
    L_ += ["| column | role | arm | interval kind | label | status | source (commit) |", "|---|---|---|---|---|---|---|"]
    for a in sorted(mine, key=lambda a: (COLUMNS.index(a["column"]), a["role"] != "headline")):
        iv = SEED_IV if a["kind"] == "seed" else BLOCK_IV if a["kind"] == "det" else "as quoted"
        src = "; ".join(f"`{rel(s)}`" for s in a["sources"])
        L_.append(f"| {a['column']} | {a['role']} | {a['label']} | {iv} | {leak_label(a['leaky'])} | "
                  f"{'QUOTED' if a['kind'] == 'quoted' else 'RECOMPUTED'} | {src} ({', '.join(a['commits'])}) |")
    L_ += ["", f"| metric | {' | '.join(col_head(c) for c in COLUMNS)} |", "|---|---|---|---|"]
    for key, name in DISPLAY[uc]:
        cells, any_ = [], False
        for c in COLUMNS:
            rec, a = _lookup(arms, uc, c, key)
            maj = None
            if a is not None and key.endswith(".accuracy"):
                mk = key.rsplit(".", 1)[0] + ".majority_accuracy"
                mr = a["metrics"].get(mk) or a["metrics"].get("majority_accuracy")
                maj = mr.get("v") if mr else None
            if rec is not None:
                any_ = True
                tag = " (Q)" if a["kind"] == "quoted" else ""
                cells.append(fmt_rec(rec, key, maj) + tag)
            else:
                cells.append("—")
        if any_:
            L_.append(f"| {name} | {' | '.join(cells)} |")
    also = [a for a in mine if a["role"] == "also"]
    if also:
        hk = HEADLINE[uc][0]
        L_ += ["", f"Other arms in the same column (headline metric `{hk}`):", "",
               f"| column | arm | {hk} | class | label |", "|---|---|---|---|---|"]
        for a in also:
            cl = a["metrics"].get("phase15.cls", {}).get("s", "—")
            L_.append(f"| {a['column']} | {a['label']} | {fmt_rec(a['metrics'].get(hk), hk)} | {cl} | {leak_label(a['leaky'])} |")
    if uc in CLASS_UC:
        L_ += [""]
        for c in COLUMNS:
            rec, a = _lookup(arms, uc, c, "right_in_10.cov5")
            if rec is not None:
                L_.append(f"- **{c}** (top 5%): {rec['s']}.")
    L_.append("")
    return L_


def md_pack(arms, checks, refrows, findings, stamp):
    L_ = ["# Phase 23AC T4.2: metric tables, INCUMBENT → BEST PUBLISHED → CLEAN", "",
          "**Audience:** whoever presents ChainPilot's model figures. Read `definitions.md` first: it defines every question,",
          "positive class, base rate and operating-point rule used here.",
          "**Measured on:** v8 seed 1001. TEST 2025. RAW unless marked. Seeds 7 / 17 / 27 / 37 / 47.",
          f"**Code:** `ml/eval/phase23ac_t4_metrics.py` at `{stamp['code_version']}`, run {stamp['run_ts']}. Long form:",
          "`metrics_pack.csv`. Log: `ml/artifacts/phase23ac/t4/metrics.log`.",
          "**Companions:** `definitions.md`, `progress.md`, `reports/part2/phase-23ac-preregistration.md`.",
          f"**Status:** complete. {len(findings)} recomputed figure(s) differ from the published one at its precision (FINDINGS below).",
          "",
          f"**How to read a cell.** Per-seed arms show the {SEED_IV}. Ensembles and blends show the {BLOCK_IV}.",
          "(Q) marks a figure QUOTED from a stored artifact or a report; everything else is RECOMPUTED from stored test",
          f"predictions. Every number in the INCUMBENT and BEST PUBLISHED columns is **{LEAKY_LABEL}**: it was produced",
          "from the published panel, which carries the nine leaking columns.", "",
          "## Sanity checks (printed by the code; each can fail)", "",
          f"1. **Recomputed vs quoted:** {sum(r['verdict'] == 'MATCH' for r in refrows)} of {len(refrows)} match at the "
          f"published precision. {len(findings)} FINDING(s) are listed below and are not smoothed.",
          "2. **Accuracy and the majority baseline:** the writer scans this file. Every line that mentions accuracy names the",
          f"   majority baseline. The constructed failing line is flagged: {checks['accuracy_scan_selftest']}.",
          f"3. **Always-majority predictor:** {checks['always_majority']['hi']}. {checks['always_majority']['lo']}.",
          f"4. **AUC by two methods:** {checks['auc_ledger']['n']} AUC pairs were computed in this run. The largest",
          f"   difference between methods is {checks['auc_ledger']['max_abs_diff']:.2e}; the tolerance is 1e-6. The",
          "   hand tie case (y = [0, 0, 1, 1], s = [0.5, 0.5, 0.5, 0.9]) gives ROC-AUC 0.75 and PR-AUC 0.75 by both methods.",
          f"   Across {checks['auc_methods']['random_cases']} random tie-heavy cases, the largest difference is",
          f"   {checks['auc_methods']['max_abs_diff']:.2e}.", "",
          "## FINDINGS: recomputed figures that do not reproduce the published value", ""]
    if findings:
        L_ += ["| arm | metric | quoted | recomputed | source |", "|---|---|---|---|---|"]
        L_ += [f"| {r['arm']} | {r['metric']} | {r['quoted']} | {r['recomputed']} | {r['source']} |" for r in findings]
    else:
        L_.append("None: every reference figure reproduces at its published precision.")
    L_ += ["", "<details><summary>All reference comparisons</summary>", "", "| arm | metric | quoted | recomputed | verdict | source |",
           "|---|---|---|---|---|---|"]
    L_ += [f"| {r['arm']} | {r['metric']} | {r['quoted']} | {r['recomputed']} | {r['verdict']} | {r['source']} |" for r in refrows]
    L_ += ["", "</details>", ""]
    for uc in UC_ORDER:
        L_ += md_use_case(arms, uc)
    L_ += ["## Use cases without a clean model metric", "", "| use case | figure (QUOTED) | status | label | source |", "|---|---|---|---|---|"]
    for name, fig, status, lab, doc, where in OTHER:
        L_.append(f"| {name} | {fig} | **{status}** | {lab} | `{rel(DOC[doc])}` {where} ({commit_of(DOC[doc])}) |")
    L_.append("")
    return "\n".join(L_)


def ask_text(n, field="What"):
    try:
        txt = open(DOC["ask"]).read()
    except OSError:
        return f"see Ask {n}"
    sec = txt.split(f"## Ask {n} ")[1].split("\n## ")[0] if f"## Ask {n} " in txt else ""
    lines = sec.splitlines(); out = []
    for i, ln in enumerate(lines):
        if ln.startswith(f"- **{field}:**"):
            out.append(ln.split(":**", 1)[1].strip())
            for nx in lines[i + 1:]:
                if nx.startswith("  ") and not nx.lstrip().startswith("- "):
                    out.append(nx.strip())
                else:
                    break
            break
    s = " ".join(out)
    first = s.split(". ")[0].rstrip(".")
    return f"\"{first}.\" (Ask {n}, `docs/client/data_request_v2.md`)" if first else f"see Ask {n}"


ASKS = {"UC1": (2, 5), "ARRIVAL": (2, 5), "ORDER-TIME DATE": (5,), "ORDER-TIME FLAG": (5, 2), "UC2": (1,), "UC2b": (1,),
        "FILL": (1,), "UC3": (4,)}


def headline_cell(arms, uc, column):
    key, hi = HEADLINE[uc]
    rec, a = _lookup(arms, uc, column, key)
    if rec is None:
        return None
    cls = (a["metrics"].get("phase15.cls") or {}).get("s", "—") if a else "—"
    base, _ = _lookup(arms, uc, column, "base_rate")
    return dict(rec=rec, arm=a, cls=cls, key=key, hi=hi, base=(base or {}).get("v"))


def meaning(arms, uc):
    h = headline_cell(arms, uc, "CLEAN")
    if h is None:
        return "no clean figure"
    r = h["rec"]
    if "s" in r:
        return r["s"]
    v = r["v"]
    if uc in CLASS_UC:
        return (f"of the 5% of cases the clean model ranks highest, {v:.0%} turn out positive, against {h['base']:.0%} of all cases: "
                f"{right_in_10(v, h['base'])}.")
    if uc in ("ARRIVAL", "ORDER-TIME DATE"):
        return f"half of the clean predicted arrival dates are within {v:.1f} days of the actual arrival."
    if uc == "FILL":
        return (f"the clean fill distribution's average CRPS is {v:.4f} on a 0-1 fill scale (lower is better); it ranks lines "
                f"by P(full) only modestly.")
    return ""


def md_progress(arms, chart_res, findings, stamp):
    L_ = ["# Phase 23AC T4.3: progress view, INCUMBENT → BEST PUBLISHED (LEAKY) → CLEAN", "",
          "**Audience:** whoever presents ChainPilot's progress to Aptimeta / Rane, and the modelling team.",
          "**Measured on:** v8 seed 1001, TEST 2025, RAW, seeds 7 / 17 / 27 / 37 / 47. Full tables: `metrics_pack.md`.",
          f"**Code:** `ml/eval/phase23ac_t4_metrics.py` (`{stamp['code_version']}`); charts: `ml/eval/phase23ac_t4_charts.py`",
          "(a separate interpreter with matplotlib, deviation 224).",
          "**Companions:** `definitions.md`, `metrics_pack.md`.",
          f"**Status:** complete{' (chart 4 skipped: Track T1 not done)' if str(chart_res.get('chart4', '')).startswith('SKIPPED') else ''}.",
          "", "## a. One summary table per use case", ""]
    changes = []
    for uc in UC_ORDER:
        key, hi = HEADLINE[uc]
        L_ += [f"### {UC_TITLE[uc]}", "", f"Headline: `{key}` ({'higher' if hi else 'lower'} is better).", "",
               f"| | {' | '.join(col_head(c) for c in COLUMNS)} |", "|---|---|---|---|"]
        hs = {c: headline_cell(arms, uc, c) for c in COLUMNS}
        L_.append("| arm | " + " | ".join((hs[c]["arm"]["label"] if hs[c] and hs[c]["arm"] else "— (none)") for c in COLUMNS) + " |")
        L_.append("| headline | " + " | ".join((fmt_rec(hs[c]["rec"], key) + (" (Q)" if hs[c]["arm"]["kind"] == "quoted" else ""))
                                              if hs[c] else "—" for c in COLUMNS) + " |")
        if uc in CLASS_UC:
            L_.append("| base rate | " + " | ".join((f"{hs[c]['base']:.3f}" if hs[c] and hs[c]["base"] is not None else "—")
                                                   for c in COLUMNS) + " |")
            L_.append("| Phase 15 class | " + " | ".join((hs[c]["cls"] if hs[c] else "—") for c in COLUMNS) + " |")
            a, b = hs["INCUMBENT"] or hs["BEST PUBLISHED"], hs["CLEAN"]
            if a and b and a["cls"] != b["cls"] and a["cls"] != "—":
                changes.append(f"{uc}: {a['cls']} → {b['cls']}")
        L_ += ["", f"- **What the numbers mean:** {meaning(arms, uc)}",
               f"- **What we would need from Rane:** " + "; ".join(ask_text(n) for n in ASKS[uc]), ""]
    L_ += ["**Class changes (published → clean):** " + ("; ".join(changes) if changes else "none") + ".", "",
           "## b. Use cases without a clean model metric", ""]
    for name, fig, status, lab, doc, where in OTHER:
        L_.append(f"- **{name}:** {status}. {fig} ({lab}; `{rel(DOC[doc])}` {where}).")
    L_ += ["- **What we would need from Rane (MILP, allocation, transfer):** " + ask_text(3) + ".", "",
           "## c. Charts (PNG in `charts/`, source CSV in `charts/data/`)", "",
           "| chart | message | source CSV |", "|---|---|---|"]
    msgs = {"chart1": ("1_headline.png", "headline metric per use case, three columns"),
            "chart2": ("2_precision_vs_base.png", "precision at the flagged cases against the base rate"),
            "chart3": ("3_capacity_snapshots.png", "capacity precision at 5% per test snapshot; worst quarter marked"),
            "chart4": ("4_order_time.png", "order-time expected-date error vs promise date and channel averages"),
            "chart5_columns": ("5_leak_story.png", "the nine leaking columns (share of channels changed)"),
            "chart5_drop": ("5_leak_story.png", "the drop per use case when the leak is removed")}
    for k, (png, msg) in msgs.items():
        v = chart_res.get(k, "")
        src = f"`{rel(v)}`" if v and not str(v).startswith("SKIPPED") else str(v)
        L_.append(f"| `{png}` | {msg} | {src} |")
    L_ += ["", "## d. Honest summary (one page)", ""]
    L_ += summary_lines(arms, changes, findings)
    L_.append("")
    return "\n".join(L_)


def summary_lines(arms, changes, findings):
    def hv(uc, c):
        h = headline_cell(arms, uc, c)
        return None if h is None or "v" not in h["rec"] else h["rec"]["v"]
    s = []
    s.append("1. Phase 22 found nine leaking panel columns. Every INCUMBENT and BEST PUBLISHED figure is superseded; quote CLEAN only.")
    for i, (uc, txt) in enumerate((("UC1", "late list, precision @ 5%"), ("UC3", "capacity, precision @ 5%"),
                                   ("UC2b", "materially-short list, precision @ 5%"), ("ARRIVAL", "arrival point A3 (days)"),
                                   ("FILL", "fill CRPS")), start=2):
        a, b, c = hv(uc, "INCUMBENT"), hv(uc, "BEST PUBLISHED"), hv(uc, "CLEAN")
        d = _dec(HEADLINE[uc][0])
        f_ = lambda x: "—" if x is None else f"{x:.{d}f}"
        s.append(f"{i}. {txt}: {f_(a)} → {f_(b)} (leaky) → **{f_(c)}** clean.")
    od, of = hv("ORDER-TIME DATE", "CLEAN"), hv("ORDER-TIME FLAG", "CLEAN")
    s.append(f"7. New at order time: the expected date is within {_f(od, 2)} days (median); the strict late flag is right "
             f"{_f(None if of is None else 10 * of, 1)} times in 10 in its top 5%.")
    rank = {"RETIRED": 0, "WATCHLIST": 1, "ALERT": 2}
    gains = [c for c in changes if rank.get(c.split(": ")[1].split(" → ")[1], -1) > rank.get(c.split(": ")[1].split(" → ")[0], 9)]
    s.append("8. Class changes (published → clean): " + ("; ".join(changes) if changes else "none") + ". "
             + (f"Gains: {'; '.join(gains)}." if gains else "No use case gains a class on clean inputs."))
    s.append("9. Predict-the-rescue and the shortage simulation: PENDING PHASE 23B. MILP, allocation and transfer are blocked on data.")
    s.append(f"10. {len(findings)} recomputed figure(s) did not reproduce a published value (FINDINGS in `metrics_pack.md`).")
    assert len(s) <= 10
    return s


# ================================================================== charts payload
def chart_payload(arms):
    head, prec, caps = [], [], []
    for uc in UC_ORDER:
        for c in COLUMNS:
            h = headline_cell(arms, uc, c)
            if h is None or "v" not in h["rec"]:
                continue
            r = h["rec"]
            head.append(dict(use_case=uc, metric=h["key"], column=c, value=r["v"], lo=r.get("lo"), hi=r.get("hi"),
                             interval=r.get("interval", ""), base_rate=h["base"], phase15_class=h["cls"],
                             label=leak_label(h["arm"]["leaky"]), status="QUOTED" if h["arm"]["kind"] == "quoted" else "RECOMPUTED",
                             higher_is_better=int(h["hi"]), source="; ".join(rel(x) for x in h["arm"]["sources"])))
            if uc in CLASS_UC:
                prec.append(dict(use_case=uc, column=c, precision_at_5pct=r["v"], lo=r.get("lo"), hi=r.get("hi"),
                                 interval=r.get("interval", ""), base_rate=h["base"], label=leak_label(h["arm"]["leaky"]),
                                 status="QUOTED" if h["arm"]["kind"] == "quoted" else "RECOMPUTED",
                                 source="; ".join(rel(x) for x in h["arm"]["sources"])))
    for a in arms:
        if a["uc"] == "UC3" and a["role"] == "headline" and a["phase15"]:
            caps.append(dict(column=a["column"], arm=a["label"], values=a["phase15"]["spread"]["values"],
                             label=leak_label(a["leaky"]), status="RECOMPUTED", source="; ".join(rel(x) for x in a["sources"])))
    return dict(headline=head, precision_vs_base=prec, capacity_snapshots=caps, t1_path=T1_JSON,
                leakscan_path=J["leakscan"], restate_path=J["restate"])


# ================================================================== main
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return _jsonable(o.tolist())
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=B_DEFAULT); a = ap.parse_args()
    st = C.require_clean()
    os.makedirs(OUTA, exist_ok=True); os.makedirs(CHART_DATA, exist_ok=True)
    try:
        log(f"Phase 23AC T4 metrics at {st['code_version']} ({st['run_ts']}); B = {a.B}, bootstrap seed {BOOT_SEED}")
        register_worlds()
        # ---- constructed failing cases first
        hi_case, lo_case = check_always_majority(constructed_always_majority(0.73)), check_always_majority(constructed_always_majority(0.245))
        aucm = check_auc_methods()
        acc_self = check_accuracy_lines("test accuracy 0.731\nUC1 accuracy 0.731 vs majority 0.730")
        assert acc_self == ["test accuracy 0.731"], "the accuracy scan cannot see a constructed offender"
        log("CHECK (iii) always-majority:", hi_case, "|", lo_case)
        log(f"CHECK (iv) AUC methods: hand tie case 0.75 / 0.75 by both; {aucm['random_cases']} random cases, max diff {aucm['max_abs_diff']:.2e}")
        log("CHECK (ii) accuracy scan flags the constructed line:", acc_self)
        # ---- arms
        arms, weights = [], {}
        log("== arrival (UC1 + point)"); x, w = arrival_arms(a.B); arms += x; weights.update(w)
        log("== fill (UC2, UC2b, distribution)"); x, w = fill_arms(a.B); arms += x; weights.update(w)
        log("== capacity (UC3)"); arms += capacity_arms(a.B)
        log("== order time (strict flag)"); arms += order_time_arms(a.B)
        arms += quoted_arms()
        by_id = {x["id"]: x for x in arms}
        assert len(by_id) == len(arms), "duplicate arm ids"
        # ---- (i) recomputed vs quoted
        refrows, findings = recompute_vs_quoted(by_id, weights)
        log(f"CHECK (i) recomputed vs quoted: {sum(r['verdict'] == 'MATCH' for r in refrows)} of {len(refrows)} match")
        for r in refrows:
            log(f"   {r['verdict']:7s} {r['arm']:16s} {r['metric']:24s} quoted {r['quoted']} recomputed {r['recomputed']}  [{r['source']}]")
        # ---- tables (the CLEAN column refuses leaky figures)
        T = populate(arms)
        df = T.frame()
        df.to_csv(os.path.join(REP, "metrics_pack.csv"), index=False)
        checks = dict(always_majority=dict(hi=hi_case, lo=lo_case), auc_methods=aucm, auc_ledger=dict(_AUC_LEDGER),
                      accuracy_scan_selftest=f"flagged {acc_self!r}")
        pack = md_pack(arms, checks, refrows, findings, st)
        bad = check_accuracy_lines(pack)
        if bad:
            raise AssertionError(f"accuracy printed without the majority baseline: {bad[:3]}")
        open(os.path.join(REP, "metrics_pack.md"), "w").write(pack)
        chart_res = CD.build_all(chart_payload(arms), CHART_DATA)
        for k, v in chart_res.items():
            log(f"  chart data {k}: {rel(v) if not str(v).startswith('SKIPPED') else v}")
        prog = md_progress(arms, chart_res, findings, st)
        bad = check_accuracy_lines(prog)
        if bad:
            raise AssertionError(f"accuracy printed without the majority baseline in progress.md: {bad[:3]}")
        open(os.path.join(REP, "progress.md"), "w").write(prog)
        out = dict(stamp=st, B=a.B, boot_seed=BOOT_SEED, weights=weights, checks=checks, reference=refrows, findings=findings,
                   arms=[{k: v for k, v in x.items()} for x in arms], charts=chart_res)
        json.dump(_jsonable(out), open(os.path.join(OUTA, "metrics.json"), "w"), indent=1, default=str)
        log(f"wrote {rel(os.path.join(REP, 'metrics_pack.md'))}, metrics_pack.csv ({len(df)} rows), progress.md, "
            f"{rel(os.path.join(OUTA, 'metrics.json'))}; FINDINGS {len(findings)}")
    finally:
        open(os.path.join(OUTA, "metrics.log"), "w").write("\n".join(_LOG) + "\n")


if __name__ == "__main__":
    main()
