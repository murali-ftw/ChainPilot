"""Phase 15 -- precision at coverage: can any use case reach a usable alert bar?

Everything runs off STORED predictions (Phase 14's sources) and the regenerated simulation rows (phase15_sim.py). Nothing
is trained. Exact per-seed filenames; no globs.

OPERATING POINTS ARE CHOSEN ON VALIDATION, PER SEED, AND APPLIED TO THAT SEED'S TEST. The test curves in Stage A are
DESCRIPTIVE -- no test curve is ever used to pick a point (asserted: `pick_on_val` takes only validation arrays).

STAGE B VERDICT RULES, fixed here before any curve was computed (committed with this file):
  precision is read on a 40-point geometric coverage grid 0.1%..100%, cells with < 50 alerts dropped.
  rho  = Spearman(coverage, precision)   (tunable scores have rho strongly NEGATIVE: tighter coverage, higher precision)
  lift = max precision - base rate;  drop = peak precision - precision at the tightest usable coverage
  TUNABLE        rho <= -0.7 AND drop <= 0.05 AND lift >= 0.10
  NOT TUNABLE    lift < 0.05 (flat)  OR  rho >= 0 (inverted)  OR  drop > 0.10 (peaks early, then falls)
  WEAKLY TUNABLE anything else
  Computed on VALIDATION per seed (the verdict is the majority over seeds) and on TEST (confirmation only).
REACHABLE (Stage G), as the brief defines it: YES if a validation-chosen point gives precision >= 0.85 with recall >= 0.10
and >= 50 alerts on TEST for EVERY seed; PARTIAL if 0.70-0.85; NO, TUNING if Stage B says NOT TUNABLE; NO, CEILING otherwise.
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import phase5_heads as P5, folds, loop as L
from phase11b_lateness import build_reference
from phase14_score import surv_full, p_late, p_exceed, fit_tau

BA, BF, BC, BS = (os.path.join(C.BUND, t) for t in ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty"))
PR = os.path.join(C.ART, "phase7_preds")
SIMD = os.path.join(C.ART, "phase15_sim")
SEEDS = C.V8_SEEDS
KS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
PS = (0.70, 0.80, 0.85, 0.90)
MIN_ALERTS = 50
GRID = np.unique(np.geomspace(0.001, 1.0, 40))


# ================================================================== curve machinery
def sorted_cum(s, y):
    o = np.argsort(-np.asarray(s, float), kind="stable")
    ss, yy = np.asarray(s, float)[o], np.asarray(y, int)[o]
    tp = np.cumsum(yy); n = np.arange(1, len(yy) + 1)
    return ss, yy, tp, n


def coverage_curve(st, yt, rng=None):
    ss, yy, tp, n = sorted_cum(st, yt)
    P = int(yy.sum()); base = P / len(yy); out = []
    rng = rng or np.random.default_rng(0)
    for k in KS:
        m = max(1, int(np.ceil(k * len(yy))))
        if m < MIN_ALERTS:
            out.append(dict(k=k, alerts=m, suppressed="n<50")); continue
        TP = int(tp[m - 1]); FP = m - TP; prec = TP / m; rec = TP / P if P else float("nan")
        b = rng.binomial(m, prec, 1000) / m
        out.append(dict(k=k, alerts=m, TP=TP, FP=FP, precision=prec, recall=rec,
                        f1=2 * TP / (m + P) if (m + P) else 0.0, lift=prec / base if base else float("nan"),
                        precision_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]))
    return dict(base_rate=base, n=len(yy), positives=P, points=out)


def pick_on_val(sv, yv, p):
    """VALIDATION ONLY: the lowest threshold (max recall) whose validation precision >= p with >= 50 alerts."""
    ss, yy, tp, n = sorted_cum(sv, yv)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    prec = tp[last] / n[last]; ok = (prec >= p) & (n[last] >= MIN_ALERTS)
    usable = n[last] >= MIN_ALERTS
    maxp = float(prec[usable].max()) if usable.any() else float("nan")
    if not ok.any():
        return None, maxp
    i = last[np.flatnonzero(ok)[-1]]
    return float(ss[i]), maxp


def apply(st, yt, tau):
    a = np.asarray(st, float) >= tau; y = np.asarray(yt, int)
    TP = int((a & (y == 1)).sum()); m = int(a.sum()); P = int(y.sum())
    return dict(alerts=m, TP=TP, precision=TP / m if m else float("nan"), recall=TP / P if P else float("nan"),
                coverage=m / len(y))


def monotonicity(s, y):
    ss, yy, tp, n = sorted_cum(s, y)
    base = yy.mean(); cov, prec = [], []
    for g in GRID:
        m = max(1, int(np.ceil(g * len(yy))))
        if m >= MIN_ALERTS:
            cov.append(g); prec.append(tp[m - 1] / m)
    if len(prec) < 3:
        return dict(verdict="INSUFFICIENT (< 3 usable coverage cells)", n_cells=len(prec))
    cov, prec = np.array(cov), np.array(prec)
    rho = float(spearmanr(cov, prec)[0]) if np.std(prec) > 0 else 0.0
    ipk = int(np.argmax(prec)); drop = float(prec.max() - prec[0]); lift = float(prec.max() - base)
    if lift < 0.05 or rho >= 0 or drop > 0.10:
        v = "NOT TUNABLE"
    elif rho <= -0.7 and drop <= 0.05 and lift >= 0.10:
        v = "TUNABLE"
    else:
        v = "WEAKLY TUNABLE"
    return dict(verdict=v, rho=rho, peak_precision=float(prec.max()), peak_coverage=float(cov[ipk]),
                tightest_coverage=float(cov[0]), precision_at_tightest=float(prec[0]), drop=drop, lift=lift, base=float(base))


def abstain(sv, yv, st, yt):
    """Two thresholds on VALIDATION: ALERT (score >= hi) with precision >= bar, CLEAR (score <= lo) with NPV >= 0.95.
    Each maximises its own coverage. Bars relax 0.80 -> 0.70 -> 0.60 for ALERT; CLEAR stays 0.95."""
    res = dict()
    hi = None
    for bar in (0.80, 0.70, 0.60):
        hi, _ = pick_on_val(sv, yv, bar)
        if hi is not None:
            res["alert_bar_met"] = bar; break
    if hi is None:
        res["alert_bar_met"] = "none of 0.80/0.70/0.60"
    # CLEAR: flip the problem -- negatives as the positive class, ascending score
    lo_neg, _ = pick_on_val(-np.asarray(sv, float), 1 - np.asarray(yv, int), 0.95)
    lo = -lo_neg if lo_neg is not None else None
    res["clear_bar_met"] = 0.95 if lo is not None else "unreachable"
    if hi is not None and lo is not None and lo >= hi:
        lo = np.nextafter(hi, -np.inf)                    # the two regions may not overlap: ALERT wins the tie
    s, y = np.asarray(st, float), np.asarray(yt, int)
    a = s >= hi if hi is not None else np.zeros(len(s), bool)
    c = s <= lo if lo is not None else np.zeros(len(s), bool)
    c &= ~a
    spoke = a | c
    res.update(n=len(y), alerts=int(a.sum()), alert_precision=float(y[a].mean()) if a.any() else float("nan"),
               clears=int(c.sum()), clear_precision=float((1 - y[c]).mean()) if c.any() else float("nan"),
               no_opinion_share=float(1 - spoke.mean()),
               recall_among_spoken=float(y[a].sum() / y[spoke].sum()) if y[spoke].sum() else float("nan"),
               tau_hi=hi, tau_lo=lo)
    return res


def band(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return [float(min(v)), float(np.mean(v)), float(max(v))] if v else None


def analyse(pairs, deterministic=False):
    """pairs: list over seeds of (sv, yv, st, yt). -> Stage A curve, A3, Stage B, Stage C."""
    out = dict(n_seeds=0 if deterministic else len(pairs),
               kind="deterministic arm: single run, no seed band; precision CI = binomial bootstrap at each coverage" if deterministic else f"{len(pairs)}-seed [min, mean, max]")
    curves = [coverage_curve(st, yt) for sv, yv, st, yt in pairs]
    out["base_rate_test"] = band([c["base_rate"] for c in curves]); out["n_test"] = curves[0]["n"]
    pts = []
    for i, k in enumerate(KS):
        cells = [c["points"][i] for c in curves]
        if any("suppressed" in x for x in cells):
            pts.append(dict(k=k, alerts=cells[0]["alerts"], suppressed="n<50")); continue
        pts.append(dict(k=k, alerts=cells[0]["alerts"], **{m: band([x[m] for x in cells]) for m in
                   ("TP", "FP", "precision", "recall", "f1", "lift")},
                        precision_ci_seed0=cells[0]["precision_ci"]))
    out["curve"] = pts
    a3 = {}
    for p in PS:
        per = []
        for sv, yv, st, yt in pairs:
            tau, maxp = pick_on_val(sv, yv, p)
            per.append(dict(tau=tau, val_max_precision=maxp, **(apply(st, yt, tau) if tau is not None else {})))
        if any(x["tau"] is None for x in per):
            a3[str(p)] = dict(status="UNREACHABLE on validation", max_val_precision=band([x["val_max_precision"] for x in per]),
                              seeds_reaching=sum(x["tau"] is not None for x in per))
        else:
            a3[str(p)] = dict(status="reached on validation", test_precision=band([x["precision"] for x in per]),
                              test_recall=band([x["recall"] for x in per]), test_alerts=band([x["alerts"] for x in per]),
                              test_coverage=band([x["coverage"] for x in per]),
                              holds_on_test_every_seed=bool(all(x["precision"] >= p and x["alerts"] >= MIN_ALERTS for x in per)))
    out["recall_at_precision"] = a3
    mv = [monotonicity(sv, yv) for sv, yv, st, yt in pairs]; mt = [monotonicity(st, yt) for sv, yv, st, yt in pairs]
    vs = [m["verdict"] for m in mv]
    out["stage_b"] = dict(verdict=max(set(vs), key=vs.count), val_per_seed=mv, test_per_seed=mt,
                          test_verdicts=[m["verdict"] for m in mt])
    ab = [abstain(*p) for p in pairs]
    out["stage_c"] = dict(per_seed=ab, alerts=band([a["alerts"] for a in ab]), alert_precision=band([a["alert_precision"] for a in ab]),
                          clears=band([a["clears"] for a in ab]), clear_precision=band([a["clear_precision"] for a in ab]),
                          no_opinion_share=band([a["no_opinion_share"] for a in ab]),
                          recall_among_spoken=band([a["recall_among_spoken"] for a in ab]),
                          alert_bar_met=[a["alert_bar_met"] for a in ab], clear_bar_met=[a["clear_bar_met"] for a in ab])
    return out


# ================================================================== providers (score/label arrays, per seed)
def lbl_fill(cut):
    return {"UC2": (lambda P: P[:, 21], lambda y: y >= 1), "UC2b_lt0.95": (lambda P: P[:, :20].sum(1), lambda y: y < 0.95),
            "UC2b_lt0.75": (lambda P: P[:, :16].sum(1), lambda y: y < 0.75)}[cut]


def fill_arrays():
    """-> {uc: {arm|calib: [ (sv,yv,st,yt), ... ]}}, plus test/val row keys (for Stage E) and alignment checks."""
    lb = P5.labels("v8", "fill_rate"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    ov, ot = P5.ordered(lb, va), P5.ordered(lb, te)
    ch = pd.read_csv(C.REPO + "/db/gen_v8/seed_1001/sourcing_channels.csv", usecols=["channel_id", "part_id", "supplier_id", "plant_id"]).set_index("channel_id")
    keys = {}
    for f, o in (("val", ov), ("test", ot)):
        k = ch.reindex(lb.key.values[o]); d = pd.to_datetime(lb.snapshot_date.values[o])
        keys[f] = pd.DataFrame(dict(entity=lb.entity_id.values[o], supplier=k.supplier_id.values, part=k.part_id.values,
                                    plant=k.plant_id.values, month=d.to_period("M").astype(str), quarter=d.to_period("Q").astype(str)))
    raw = {}
    for s in SEEDS:
        for arm, suf in (("head_cells22", ""), ("boundary_bw3", "_lossrps_bw3"), ("beta3", "_headbeta3")):
            d = f"{BF}/v8_none_h0_lr0.000125_s{s}{suf}"; rec = json.load(open(f"{d}/recalibration.json"))
            for f in ("val", "test"):
                z = dict(np.load(f"{d}/preds_{f}.npz"))
                raw.setdefault((arm, "raw"), {}).setdefault(s, {})[f] = (np.asarray(z["P"], float), z["Y"])
                raw.setdefault((arm, "recal"), {}).setdefault(s, {})[f] = (L.apply_recalibration(rec, "fill_rate", z)["P22"], z["Y"])
        for arm in ("b5flat22", "lgbm22_id"):
            for cal, pre in (("raw", ""), ("recal", "RECAL_")):
                for f in ("val", "test"):
                    z = np.load(f"{PR}/{pre}v8_fill_rate_{arm}_s{s}_{f}.npz")
                    assert (z["entity"].astype(str) == keys[f].entity.astype(str).values).all(), f"{arm} rows misaligned"
                    raw.setdefault((arm, cal), {}).setdefault(s, {})[f] = (z["P"], z["Y"])
    for f in ("val", "test"):
        z = np.load(f"{PR}/v8_fill_rate_b2_rolling52_cdf_{f}.npz")
        assert (z["entity"].astype(str) == keys[f].entity.astype(str).values).all(), "histogram rows misaligned"
        raw.setdefault(("rolling52_histogram", "raw"), {}).setdefault(0, {})[f] = (z["P"], z["Y"])
    out = {}
    for uc in ("UC2", "UC2b_lt0.95", "UC2b_lt0.75"):
        sf, lf = lbl_fill(uc)
        for (arm, cal), per in raw.items():
            out.setdefault(uc, {})[f"{arm}|{cal}"] = [(sf(np.asarray(v["val"][0], float)), lf(v["val"][1]).astype(int),
                                                       sf(np.asarray(v["test"][0], float)), lf(v["test"][1]).astype(int)) for v in per.values()]
    return out, keys


def arrival_arrays():
    lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    R = build_reference("contracted_lead", "v8", lb, tr)[0]; ov, ot = P5.ordered(lb, va), P5.ordered(lb, te)
    Rv, Rt = R[ov], R[ot]
    out = {"UC1": {}, "UC1_A_excluded": {}, "UC1b_H4": {}}

    def lab_c(z, Rx):
        ev = z["EV"].astype(bool); late = (ev & (z["Y"] > Rx)) | (~ev & (Rx < 13)); keep = ev | (Rx < 13)
        return late.astype(int), keep

    def lab_a(z, Rx):
        ev = z["EV"].astype(bool); return (z["Y"] > Rx).astype(int), ev
    for s in SEEDS:
        for arm, d in (("h4", f"{BA}/v8_lite_h4_lr0.00025_s{s}"), ("h0", f"{BA}/v8_none_h0_lr0.00025_s{s}")):
            rec = json.load(open(f"{d}/recalibration.json"))
            zz = {f: dict(np.load(f"{d}/preds_{f}.npz")) for f in ("val", "test")}
            for cal in ("raw", "recal"):
                S = {f: (zz[f]["S"] if cal == "raw" else L.apply_recalibration(rec, "arrival_week", zz[f])["S"]) for f in zz}
                for uc, lab in (("UC1", lab_c), ("UC1_A_excluded", lab_a)):
                    (yv, kv), (yt, kt) = lab(zz["val"], Rv), lab(zz["test"], Rt)
                    out[uc].setdefault(f"{arm}|{cal}", []).append((p_late(S["val"], Rv)[kv], yv[kv], p_late(S["test"], Rt)[kt], yt[kt]))
                h = lambda z: (z["EV"].astype(bool) & (z["Y"] <= 4)).astype(int)
                out["UC1b_H4"].setdefault(f"{arm}|{cal}", []).append((1 - surv_full(S["val"])[:, 4], h(zz["val"]),
                                                                       1 - surv_full(S["test"])[:, 4], h(zz["test"])))
        for arm in ("b5flat_reg", "lgbm_id_reg"):
            zz = {f: dict(np.load(f"{PR}/v8_arrival_week_{arm}_s{s}_{f}.npz")) for f in ("val", "test")}
            for uc, lab in (("UC1", lab_c), ("UC1_A_excluded", lab_a)):
                (yv, kv), (yt, kt) = lab(zz["val"], Rv), lab(zz["test"], Rt)
                out[uc].setdefault(f"{arm}|raw", []).append(((zz["val"]["P"] - Rv)[kv], yv[kv], (zz["test"]["P"] - Rt)[kt], yt[kt]))
            h = lambda z: (z["EV"].astype(bool) & (z["Y"] <= 4)).astype(int)
            out["UC1b_H4"].setdefault(f"{arm}|raw", []).append((-zz["val"]["P"], h(zz["val"]), -zz["test"]["P"], h(zz["test"])))
    z0 = {f: dict(np.load(f"{BA}/v8_lite_h4_lr0.00025_s7/preds_{f}.npz")) for f in ("val", "test")}
    for uc, lab in (("UC1", lab_c), ("UC1_A_excluded", lab_a)):
        (yv, kv), (yt, kt) = lab(z0["val"], Rv), lab(z0["test"], Rt)
        out[uc]["channel_constant_ranker|raw"] = [(-Rv[kv], yv[kv], -Rt[kt], yt[kt])]
    return out


def capacity_arrays(level=None):
    """level: None | ('val', per-seed scalar fitted on validation) | ('oracle', PRIVILEGED test-fitted scalar)."""
    lb = P5.labels("v8", "capacity_strain"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    ov, ot = P5.ordered(lb, va), P5.ordered(lb, te)
    ch = pd.read_csv(C.REPO + "/db/gen_v8/seed_1001/sourcing_channels.csv", usecols=["channel_id", "supplier_id"]).set_index("channel_id")
    keys = {f: pd.DataFrame(dict(entity=lb.entity_id.values[o], supplier=ch.reindex(lb.entity_id.values[o]).supplier_id.values,
                                 snap=lb.snapshot_date.values[o])) for f, o in (("val", ov), ("test", ot))}
    out, raw = {}, {}
    for s in SEEDS:
        for arm, d in (("mp_h4", f"{BC}/v8_mp_h4_lr0.00025_s{s}"), ("h0", f"{BC}/v8_none_h0_lr0.00025_s{s}")):
            raw.setdefault(arm, []).append({f: np.load(f"{d}/preds_{f}.npz") for f in ("val", "test")})
        z = {f: np.load(f"{PR}/v8_capacity_strain_b5flat_q_s{s}_{f}.npz") for f in ("val", "test")}
        assert (z["test"]["entity"].astype(str) == keys["test"].entity.astype(str).values).all(), "b5flat_q misaligned"
        raw.setdefault("b5flat_q", []).append(z)
    for nm in ("naive_channel",):
        raw[nm] = [{f: np.load(f"{PR}/v8_capacity_strain_{nm}_{f}.npz") for f in ("val", "test")}]
    meta = {}
    for arm, per in raw.items():
        lst = []
        for i, z in enumerate(per):
            Pv, Pt = np.asarray(z["val"]["P"], float), np.asarray(z["test"]["P"], float)
            if level == "val":
                r = float(z["val"]["Y"].mean() / Pv[:, 1].mean()); Pv, Pt = Pv * r, Pt * r
                meta.setdefault(arm, []).append(r)
            elif level == "oracle":
                r = float(z["test"]["Y"].mean() / Pt[:, 1].mean()); Pv, Pt = Pv * r, Pt * r
                meta.setdefault(arm, []).append(r)
            lst.append((p_exceed(Pv), (z["val"]["Y"] > 1).astype(int), p_exceed(Pt), (z["test"]["Y"] > 1).astype(int)))
        out[f"{arm}|raw"] = lst
    return out, keys, meta


def shortage_arrays():
    out = {}
    for arm, pat, seeds in (("mp_h1_shipped", "v8_mp_h1_lr0.000125_s{s}", SEEDS), ("h0_3seeds_NOT_QUOTABLE", "v8_none_h0_lr0.000125_s{s}", (7, 17, 27))):
        out[f"{arm}|raw"] = [tuple(x for f in ("val", "test") for x in (np.load(f"{BS}/{pat.format(s=s)}/preds_{f}.npz")["P"],
                              np.load(f"{BS}/{pat.format(s=s)}/preds_{f}.npz")["Y"].astype(int))) for s in seeds]
    out["b5flat_bin|raw"] = [tuple(x for f in ("val", "test") for x in (np.load(f"{PR}/v8_shortage_qty_b5flat_bin_s{s}_{f}.npz")["P"],
                              np.load(f"{PR}/v8_shortage_qty_b5flat_bin_s{s}_{f}.npz")["Y"].astype(int))) for s in SEEDS]
    out["naive_part_plant_rate|raw"] = [tuple(x for f in ("val", "test") for x in (np.load(f"{PR}/v8_shortage_qty_naive_part_plant_rate_{f}.npz")["P"],
                                          np.load(f"{PR}/v8_shortage_qty_naive_part_plant_rate_{f}.npz")["Y"].astype(int)))]
    return out


def sim_arrays():
    z = {(f, s): np.load(os.path.join(SIMD, f"{f}_s{s}.npz")) for f in ("val", "test") for s in SEEDS}
    g = lambda k, lab: [(z[("val", s)][k], z[("val", s)][lab].astype(int), z[("test", s)][k], z[("test", s)][lab].astype(int)) for s in SEEDS]
    return {"UC5a": {"policy_rop_sim|raw": g("p", "obs")}, "UC5b_PRIVILEGED": {"policy_rop_sim|raw": g("p", "pre")},
            "UC8_recipient": {"policy_score|raw": g("rp", "ry")},
            "UC8_donor_INFERRED": {"policy_score|raw": g("dp", "dy")}}, z


def uc7_arrays():
    a4 = json.load(open(os.path.join(C.ART, "phase12_a4.json"))); u = json.load(open(os.path.join(C.ART, "phase14_uc7.json")))
    return a4, u


# ================================================================== Stage D, E, F
def agreement(members, thresholds_pairs):
    """members: {name: [(sv,yv,st,yt) per seed]} with seed i aligned across members (the histogram repeats). F1-fitted tau
    per member per seed (Phase 14's rule). -> single, ALL-agree, 2-of-3 precision / recall / coverage on test."""
    names = list(members); nseed = max(len(v) for v in members.values())
    rows = {n: [] for n in names + ["ALL_3", "ANY_2_of_3"]}
    for i in range(nseed):
        fires, yt = {}, None
        for n in names:
            sv, yv, st, y = members[n][min(i, len(members[n]) - 1)]
            tau, _ = fit_tau(sv, yv, "f1"); fires[n] = st >= tau; yt = y if yt is None else yt
            assert len(y) == len(yt) and (y == yt).all(), "agreement members are not row-aligned"
        votes = sum(fires[n].astype(int) for n in names)
        fires["ALL_3"] = votes == 3; fires["ANY_2_of_3"] = votes >= 2
        for n, a in fires.items():
            m = int(a.sum()); TP = int((a & (yt == 1)).sum())
            rows[n].append(dict(alerts=m, precision=TP / m if m else float("nan"), recall=TP / max(int(yt.sum()), 1), coverage=m / len(yt)))
    return {n: {k: band([r[k] for r in v]) for k in ("alerts", "precision", "recall", "coverage")} for n, v in rows.items()}


def pool(pairs, keys_v, keys_t, by, how):
    """Re-aggregate per-row scores to coarser units: label = ANY constituent positive; score = mean or max."""
    out = []
    for sv, yv, st, yt in pairs:
        res = []
        for s, y, k in ((sv, yv, keys_v), (st, yt, keys_t)):
            g = pd.DataFrame(dict(s=s, y=y)).assign(**{c: k[c].values for c in by}).groupby(by)
            res += [g.s.agg(how).to_numpy(float), (g.y.max() > 0).astype(int).to_numpy()]
        out.append(tuple(res))
    return out


def falsify_stage_b():
    """Stage B's classifier must be able to return every verdict. Four constructed inputs, base rate 0.2."""
    rng = np.random.default_rng(0); n = 20000; y = (rng.random(n) < 0.2).astype(int)
    good = y * 1.5 + rng.normal(0, 1, n)
    bad = good.copy(); bad[rng.choice(np.flatnonzero(y == 0), 300, replace=False)] = 10.0   # confident wrong answers on top
    cases = {"informative (expect TUNABLE)": good, "random (expect NOT TUNABLE)": rng.random(n),
             "inverted (expect NOT TUNABLE)": -good, "peaks early then falls (expect NOT TUNABLE)": bad}
    return {k: {m: v for m, v in monotonicity(s_, y).items() if m in ("verdict", "rho", "lift", "drop")} for k, s_ in cases.items()}


def main():
    st = C.require_clean()
    R = dict(stamp=st, rules="see module docstring (committed before any curve)", ks=list(KS), precisions=list(PS))
    R["stage_b_falsification"] = falsify_stage_b()
    print("Stage B falsification:", json.dumps(R["stage_b_falsification"]), flush=True)
    fill, fkeys = fill_arrays()
    arr = arrival_arrays()
    cap, ckeys, _ = capacity_arrays()
    sho = shortage_arrays()
    sim, simz = sim_arrays()
    ucs = {**arr, **fill, "UC3": cap, "UC4": sho, **sim}
    R["uc"] = {}
    for uc, arms in ucs.items():
        for arm, pairs in arms.items():
            det = arm.startswith(("rolling52", "channel_constant", "naive"))
            R["uc"].setdefault(uc, {})[arm] = analyse(pairs, deterministic=det)
            a3 = R["uc"][uc][arm]["recall_at_precision"]["0.85"]
            print(uc, arm, R["uc"][uc][arm]["stage_b"]["verdict"], a3.get("status"), a3.get("test_precision"), a3.get("test_recall"), flush=True)
    # UC7: a ranking over 63 feasible part-plants -- every coverage cell below 50 alerts is suppressed by rule
    a4, u7 = uc7_arrays()
    R["UC7"] = dict(feasible=u7["feasible"], sampled=u7["sampled"], note="63 feasible part-plants: every coverage cell has < 50 alerts "
                    "and is suppressed; no confidence curve is reportable. BLOCKED on the qualification rule (deviation 91).")
    # ---------------- Stage D
    R["stage_d"] = {}
    for uc in ("UC2", "UC2b_lt0.95", "UC2b_lt0.75"):
        R["stage_d"][uc] = agreement({"head_cells22": fill[uc]["head_cells22|raw"], "b5flat22": fill[uc]["b5flat22|raw"],
                                      "rolling52_histogram": fill[uc]["rolling52_histogram|raw"]}, None)
    R["stage_d"]["UC3"] = agreement({"mp_h4": cap["mp_h4|raw"], "h0": cap["h0|raw"], "b5flat_q": cap["b5flat_q|raw"]}, None)
    # ---------------- Stage E
    R["stage_e"] = {}
    best_fill = fill["UC2b_lt0.95"]["boundary_bw3|raw"]
    for grain, by in (("row (native)", None), ("supplier x month", ["supplier", "month"]), ("part x plant x month", ["part", "plant", "month"]),
                      ("supplier x quarter", ["supplier", "quarter"])):
        for how in (("row",) if by is None else ("mean", "max")):
            pairs = best_fill if by is None else pool(best_fill, fkeys["val"], fkeys["test"], by, how)
            R["stage_e"].setdefault("UC2b_lt0.95 boundary_bw3", {})[f"{grain}|{how}"] = analyse(pairs)
    sk = {}
    ppq = pd.read_csv(C.REPO + "/db/gen_v8/seed_1001/part_plant.csv", usecols=["part_id", "plant_id"])   # index order = sub
    for f in ("val", "test"):
        z = simz[(f, 7)]; wk = pd.to_datetime(z["week"])
        sk[f] = pd.DataFrame(dict(pp=z["pp"], month=wk.to_period("M").astype(str), quarter=wk.to_period("Q").astype(str)))
    for f in ("val", "test"):
        for s in SEEDS:
            assert (simz[(f, s)]["pp"] == sk[f].pp.values).all(), "simulation rows differ between seeds -- cannot pool with one key frame"
    for grain, by in (("row (native)", None), ("part x plant x month", ["pp", "month"]), ("part x plant x quarter (substitute)", ["pp", "quarter"])):
        for how in (("row",) if by is None else ("mean", "max")):
            pairs = sim["UC5a"]["policy_rop_sim|raw"] if by is None else pool(sim["UC5a"]["policy_rop_sim|raw"], sk["val"], sk["test"], by, how)
            R["stage_e"].setdefault("UC5a", {})[f"{grain}|{how}"] = analyse(pairs)
    for grain, by in (("channel x snapshot (native 90-day)", None), ("supplier x snapshot", ["supplier", "snap"])):
        for how in (("row",) if by is None else ("mean", "max")):
            pairs = cap["mp_h4|raw"] if by is None else pool(cap["mp_h4|raw"], ckeys["val"], ckeys["test"], by, how)
            R["stage_e"].setdefault("UC3 mp_h4", {})[f"{grain}|{how}"] = analyse(pairs)
    # ---------------- Stage F
    capv, _, rv = capacity_arrays("val"); capo, _, ro = capacity_arrays("oracle")
    R["stage_f"] = dict(val_fitted_scalar=rv, PRIVILEGED_oracle_scalar=ro,
                        corrected_val_fitted=analyse(capv["mp_h4|raw"]),
                        PRIVILEGED_oracle_NOT_A_RESULT=analyse(capo["mp_h4|raw"]))
    print(C.dump(R, "phase15.json"))


if __name__ == "__main__":
    main()
