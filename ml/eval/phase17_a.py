"""Phase 17 Track A -- CPU analysis from STORED predictions and Phase 15's stored simulation rows. Nothing is trained.

A1  the shipped shortage head (mp h1, 5 seeds), TEST, re-weighted from v8's 25.7% positive rate to 3%:
      (a) importance weighting of negatives (weight k on every negative, k set so the weighted base rate is 3%)
      (b) stratified subsampling of positives to 3%, 100 resamples per seed
    at coverage 0.5 / 1 / 2 / 5 / 10%: precision, recall, F1, PR-AUC, with the always-majority figure on the same rows.
    Cells with < 50 alerts (actual rows) are suppressed.
A2  Phase 14 UC5's rescued part-plant-weeks (pre below SS, observed not), split by whether the simulation flagged them at
    each fill seed's VALIDATION-fitted max-F1 threshold against (a) -- Phase 14's rule. GATE: Phase 14's counts must be
    reproduced (12,401 rescued; seed 7 flagged 4,171 / missed 8,230) before any profile is computed.
A3  signed-days lateness from the stored h4 hazard distributions:
      E[lateness_days] = 7 * (sum_k P(T=k) k + 13 * P(T>12)) - 7 R      (censored mass at k = 13: expected_time's cap)
      P10 / P50 / P90 from the survival curve (13 = "later than 12 weeks")
    R = the AS-OF CHANNEL MEDIAN OBSERVED LEAD + c (docs/specs/lateness_metric.md), never the promise date.
    Scored on UNCENSORED test rows. NOTE: for a point error R cancels -- (pred - R) - (Y - R) = pred - Y.

inventory_position_weekly: A2 reads it, as of each t0, for opening on-hand, safety stock and open_po_qty -- a profile of
rows that are already labelled, evaluation only. A1 and A3 do not read it. Never a feature.

  python ml/eval/phase17_a.py a1|a2|a3|all
"""
from __future__ import annotations
import os, sys, json
import phase12_common as C
import numpy as np, pandas as pd

OUT = "phase17"
SEEDS = C.V8_SEEDS
D = os.path.join(C.REPO, "db", "gen_v8", "seed_1001")
TARGET = 0.03
A1_KS = (0.005, 0.01, 0.02, 0.05, 0.10)
MIN_ALERTS = 50


def band(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(min(v)), float(np.mean(v)), float(max(v))] if v else None


def gini(counts):
    x = np.sort(np.asarray(counts, float)); n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def top_decile_share(counts):
    x = np.sort(np.asarray(counts, float))[::-1]
    k = max(1, int(np.ceil(0.10 * len(x))))
    return float(x[:k].sum() / x.sum()) if x.sum() else float("nan")


# ================================================================== A1
def weighted_curve(s, y, w):
    o = np.argsort(-s, kind="stable"); yy, ww = y[o], w[o]
    cw = np.cumsum(ww); tp = np.cumsum(yy); fpw = np.cumsum(ww * (1 - yy))
    P, Wt = yy.sum(), ww.sum(); out = []
    for k in A1_KS:
        m = int(np.searchsorted(cw, k * Wt)) + 1          # rows needed to reach this share of the WEIGHTED population
        if m < MIN_ALERTS:
            out.append(dict(k=k, rows=m, suppressed="n<50")); continue
        TP = float(tp[m - 1]); prec = TP / (TP + fpw[m - 1]); rec = TP / P
        out.append(dict(k=k, rows=m, precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0))
    return out


def plain_curve(s, y):
    o = np.argsort(-s, kind="stable"); yy = y[o]; tp = np.cumsum(yy); P = yy.sum(); out = []
    for k in A1_KS:
        m = max(1, int(np.ceil(k * len(yy))))
        if m < MIN_ALERTS:
            out.append(dict(k=k, rows=m, suppressed="n<50")); continue
        prec = tp[m - 1] / m; rec = tp[m - 1] / P
        out.append(dict(k=k, rows=m, precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0))
    return out


def a1():
    import phase15 as P15
    from sklearn.metrics import average_precision_score
    st = C.require_clean()
    pairs = P15.shortage_arrays()["mp_h1_shipped|raw"]
    per_seed = []
    for s, (sv, yv, s_t, y_t) in zip(SEEDS, pairs):
        s_t = np.asarray(s_t, float); y_t = np.asarray(y_t, int)
        P, N = int(y_t.sum()), int(len(y_t) - y_t.sum())
        kneg = (P / N) * (1 - TARGET) / TARGET                        # weight on each negative
        w = np.where(y_t == 1, 1.0, kneg)
        a = dict(weight_on_negatives=kneg, weighted_base_rate=float(P / (P + kneg * N)), curve=weighted_curve(s_t, y_t, w),
                 pr_auc=float(average_precision_score(y_t, s_t, sample_weight=w)))
        rng = np.random.default_rng(1000 + s)
        npos = int(round(TARGET / (1 - TARGET) * N)); pos, neg = np.flatnonzero(y_t == 1), np.flatnonzero(y_t == 0)
        bs = []
        for _ in range(100):
            ii = np.r_[neg, rng.choice(pos, npos, replace=False)]
            bs.append(dict(curve=plain_curve(s_t[ii], y_t[ii]), pr_auc=float(average_precision_score(y_t[ii], s_t[ii]))))
        b = dict(positives_kept=npos, n=len(neg) + npos, base_rate=npos / (len(neg) + npos), curve=[])
        for i, k in enumerate(A1_KS):
            cells = [x["curve"][i] for x in bs]
            if any("suppressed" in c for c in cells):
                b["curve"].append(dict(k=k, rows=cells[0]["rows"], suppressed="n<50")); continue
            pr = np.array([c["precision"] for c in cells])
            b["curve"].append(dict(k=k, rows=cells[0]["rows"], precision_mean=float(pr.mean()),
                                   precision_p2_5=float(np.percentile(pr, 2.5)), precision_p50=float(np.median(pr)),
                                   precision_p97_5=float(np.percentile(pr, 97.5)),
                                   recall_mean=float(np.mean([c["recall"] for c in cells])),
                                   f1_mean=float(np.mean([c["f1"] for c in cells]))))
        b["pr_auc_mean"] = float(np.mean([x["pr_auc"] for x in bs]))
        per_seed.append(dict(seed=s, n_test=len(y_t), v8_base_rate=P / len(y_t), v8_curve=plain_curve(s_t, y_t),
                             v8_pr_auc=float(average_precision_score(y_t, s_t)), importance_weighted=a, subsampled=b))
    summ = {}
    for i, k in enumerate(A1_KS):
        row = dict(k=k)
        v8 = [p["v8_curve"][i] for p in per_seed]; iw = [p["importance_weighted"]["curve"][i] for p in per_seed]
        ss = [p["subsampled"]["curve"][i] for p in per_seed]
        row["v8_precision"] = band([c.get("precision") for c in v8]); row["v8_recall"] = band([c.get("recall") for c in v8])
        row["iw_precision"] = band([c.get("precision") for c in iw]); row["iw_recall"] = band([c.get("recall") for c in iw])
        row["iw_f1"] = band([c.get("f1") for c in iw]); row["iw_rows"] = iw[0].get("rows")
        row["sub_precision_mean"] = band([c.get("precision_mean") for c in ss])
        row["sub_precision_p2_5_p97_5_seed7"] = [ss[0].get("precision_p2_5"), ss[0].get("precision_p97_5")]
        row["sub_recall_mean"] = band([c.get("recall_mean") for c in ss]); row["sub_f1_mean"] = band([c.get("f1_mean") for c in ss])
        row["sub_rows"] = ss[0].get("rows")
        row["max_precision_possible_at_3pct"] = min(1.0, TARGET / k)
        row["delta_precision_iw_vs_v8"] = (row["iw_precision"][1] - row["v8_precision"][1]) if row["iw_precision"] and row["v8_precision"] else None
        summ[str(k)] = row
    p5 = summ["0.05"]
    def verdict(p):
        return "SURVIVES" if p >= 0.60 else ("INTERNAL WATCHLIST ONLY" if p >= 0.40 else "RETIRE")
    out = dict(stamp=st, target_base_rate=TARGET, summary=summ,
               pr_auc=dict(v8=band([p["v8_pr_auc"] for p in per_seed]), iw=band([p["importance_weighted"]["pr_auc"] for p in per_seed]),
                           sub=band([p["subsampled"]["pr_auc_mean"] for p in per_seed])),
               majority=dict(v8_always_negative_accuracy=band([1 - p["v8_base_rate"] for p in per_seed]),
                             at_3pct_always_negative_accuracy=1 - TARGET, always_negative_f1=0.0,
                             note="always-negative raises no alert: precision undefined, recall 0, F1 0"),
               decision=dict(precision_at_5pct_iw=p5["iw_precision"], precision_at_5pct_sub=p5["sub_precision_mean"],
                             verdict_iw=verdict(p5["iw_precision"][1]), verdict_sub=verdict(p5["sub_precision_mean"][1]),
                             ceiling_note="at a 3% base and 5% coverage even a perfect ranker reaches 0.03/0.05 = 0.60"),
               per_seed=per_seed)
    C.dump(out, f"{OUT}/a1.json")
    print(json.dumps(dict(summary=summ, decision=out["decision"], pr_auc=out["pr_auc"]), indent=1))


# ================================================================== A2
def asof_position(t0, keys):
    pw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "safety_stock_qty", "open_po_qty", "recorded_ts"])
    pw = pw[pd.to_datetime(pw.recorded_ts) <= pd.Timestamp(t0)]
    assert len(pw) and (pd.to_datetime(pw.recorded_ts) <= pd.Timestamp(t0)).all(), "as-of violation"
    pw = pw.sort_values("week_start").groupby(["part_id", "plant_id"]).tail(1).set_index(["part_id", "plant_id"])
    return pw.reindex(keys)[["qty_on_hand", "safety_stock_qty", "open_po_qty"]]


def a2():
    from phase14_score import fit_tau
    from scipy.stats import mannwhitneyu
    st = C.require_clean()
    z = np.load(os.path.join(C.ART, "phase17", "b1_rows.npz"))
    te = z["fold"] == 2
    snap_t, pp_t, w_t, week_t = z["snap"][te], z["pp"][te], z["w"][te], z["week"][te]
    parts, plants, nch = z["parts"], z["plants"], z["n_channels"]
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "part_id", "plant_id"])
    sup_of = ch.groupby(["part_id", "plant_id"]).supplier_id.apply(lambda s: sorted(set(s)))
    keys = pd.MultiIndex.from_arrays([parts, plants])
    sup_lists = [sup_of.get(k, []) for k in keys]
    # as-of position per (snapshot, part-plant): opening on hand, SS, open pipeline
    pos = {}
    for t0 in np.unique(snap_t):
        pos[t0] = asof_position(pd.Timestamp(t0), keys).to_numpy(float)
    POS = np.stack([pos[t][p] for t, p in zip(snap_t, pp_t)])          # [rows, 3]
    stored14 = dict(rescued=12401, flagged={7: 4171, 17: 4152, 27: 4246, 37: 4231, 47: 4338},
                    missed={7: 8230, 17: 8249, 27: 8155, 37: 8170, 47: 8063})
    per_seed, gate = [], []
    for s in SEEDS:
        zv, zt = (np.load(os.path.join(C.ART, "phase15_sim", f"{f}_s{s}.npz")) for f in ("val", "test"))
        assert np.array_equal(zt["pp"], pp_t) and np.array_equal(zt["week"], week_t), "row keys differ from b1_rows"
        tau, _ = fit_tau(zv["p"], zv["obs"].astype(int))
        rescued = zt["pre"] & ~zt["obs"]; flagged = zt["p"] >= tau
        miss, hit = rescued & ~flagged, rescued & flagged
        g = dict(seed=s, tau=tau, rescued=int(rescued.sum()), flagged=int(hit.sum()), missed=int(miss.sum()),
                 matches_phase14=bool(int(rescued.sum()) == stored14["rescued"] and int(hit.sum()) == stored14["flagged"][s]
                                      and int(miss.sum()) == stored14["missed"][s]))
        gate.append(g)
        assert g["matches_phase14"], f"A2 GATE: Phase 14's rescued/flagged/missed counts not reproduced: {g}"
        prof = dict(seed=s)
        for nm, m in (("missed", miss), ("flagged", hit), ("all_rows", np.ones_like(miss))):
            c_pp = np.bincount(pp_t[m], minlength=len(parts))
            c_part = pd.Series(c_pp).groupby(parts).sum(); c_plant = pd.Series(c_pp).groupby(plants).sum()
            c_sup = {}
            for i in np.flatnonzero(c_pp):
                for sp in sup_lists[i]:
                    c_sup[sp] = c_sup.get(sp, 0.0) + c_pp[i] / len(sup_lists[i])
            sup_all = pd.Series(c_sup).reindex(ch.supplier_id.unique()).fillna(0)
            prof[nm] = dict(rows=int(m.sum()), part_plants=int((c_pp > 0).sum()),
                            concentration={e: dict(entities=int(len(c)), top_decile_share=top_decile_share(c), gini=gini(c))
                                           for e, c in (("part_plant", c_pp), ("part", c_part), ("plant", c_plant), ("supplier", sup_all))},
                            channels_per_part_plant_rows=dict(median=float(np.median(nch[pp_t[m]])), mean=float(np.mean(nch[pp_t[m]]))),
                            channels_per_part_plant_unique=dict(median=float(np.median(nch[np.flatnonzero(c_pp)])),
                                                                mean=float(np.mean(nch[np.flatnonzero(c_pp)]))),
                            on_hand_median=float(np.nanmedian(POS[m, 0])), safety_stock_median=float(np.nanmedian(POS[m, 1])),
                            open_pipeline_median=float(np.nanmedian(POS[m, 2])),
                            on_hand_over_ss_median=float(np.nanmedian(POS[m, 0] / np.maximum(POS[m, 1], 1))),
                            pipeline_over_ss_median=float(np.nanmedian(POS[m, 2] / np.maximum(POS[m, 1], 1))),
                            pipeline_zero_share=float(np.mean(POS[m, 2] <= 0)),
                            week_offset_hist=np.bincount(w_t[m], minlength=13).tolist(),
                            month_hist=np.bincount(pd.DatetimeIndex(week_t[m]).month.to_numpy(), minlength=13)[1:].tolist())
        u_rows = mannwhitneyu(nch[pp_t[miss]], nch[pp_t[hit]], alternative="two-sided")
        um, uh = np.unique(pp_t[miss]), np.unique(pp_t[hit])
        u_pp = mannwhitneyu(nch[um], nch[uh], alternative="two-sided")
        prof["channels_test"] = dict(rows=dict(U=float(u_rows.statistic), p=float(u_rows.pvalue),
                                               auc_missed_gt_flagged=float(u_rows.statistic / (miss.sum() * hit.sum()))),
                                     unique_part_plants=dict(U=float(u_pp.statistic), p=float(u_pp.pvalue),
                                                             auc_missed_gt_flagged=float(u_pp.statistic / (len(um) * len(uh)))))
        per_seed.append(prof)
        print(f"  seed {s}: missed ch/pp median {prof['missed']['channels_per_part_plant_rows']['median']} vs flagged "
              f"{prof['flagged']['channels_per_part_plant_rows']['median']}  AUC {prof['channels_test']['rows']['auc_missed_gt_flagged']:.3f}", flush=True)
    keyb = lambda f: band([f(p) for p in per_seed])
    summ = dict(
        channels_rows=dict(missed_mean=keyb(lambda p: p["missed"]["channels_per_part_plant_rows"]["mean"]),
                           flagged_mean=keyb(lambda p: p["flagged"]["channels_per_part_plant_rows"]["mean"]),
                           missed_median=keyb(lambda p: p["missed"]["channels_per_part_plant_rows"]["median"]),
                           flagged_median=keyb(lambda p: p["flagged"]["channels_per_part_plant_rows"]["median"]),
                           all_rows_mean=keyb(lambda p: p["all_rows"]["channels_per_part_plant_rows"]["mean"]),
                           auc_missed_gt_flagged=keyb(lambda p: p["channels_test"]["rows"]["auc_missed_gt_flagged"]),
                           p_value_max=max(p["channels_test"]["rows"]["p"] for p in per_seed)),
        channels_unique_pp=dict(missed_mean=keyb(lambda p: p["missed"]["channels_per_part_plant_unique"]["mean"]),
                                flagged_mean=keyb(lambda p: p["flagged"]["channels_per_part_plant_unique"]["mean"]),
                                auc_missed_gt_flagged=keyb(lambda p: p["channels_test"]["unique_part_plants"]["auc_missed_gt_flagged"]),
                                p_value_max=max(p["channels_test"]["unique_part_plants"]["p"] for p in per_seed)),
        concentration={grp: {e: dict(top_decile=keyb(lambda p, e=e: p[grp]["concentration"][e]["top_decile_share"]),
                                     gini=keyb(lambda p, e=e: p[grp]["concentration"][e]["gini"]))
                             for e in ("part_plant", "part", "plant", "supplier")} for grp in ("missed", "flagged", "all_rows")},
        position={grp: {k: keyb(lambda p, k=k, grp=grp: p[grp][k]) for k in
                        ("on_hand_over_ss_median", "pipeline_over_ss_median", "pipeline_zero_share", "on_hand_median",
                         "safety_stock_median", "open_pipeline_median")} for grp in ("missed", "flagged", "all_rows")})
    out = dict(stamp=st, gate=gate, summary=summ, per_seed=per_seed,
               uniform_expectation=dict(top_decile_share=0.10, gini=0.0))
    C.dump(out, f"{OUT}/a2.json")
    print(json.dumps(dict(gate=gate, summary=summ), indent=1, default=str))


# ================================================================== A3
def a3():
    import phase5_heads as P5, folds, loop as L
    from phase11b_lateness import build_reference, channel_lead_history
    st = C.require_clean()
    lb = P5.labels("v8", "arrival_week"); tr, va, te = folds.fixed_split(lb.snapshot_date)
    R, rmeta = build_reference("asof_channel_lead", "v8", lb, tr)
    ot = P5.ordered(lb, te); Rt = R[ot]
    BA = os.path.join(C.BUND, "arrival_week"); PR = os.path.join(C.ART, "phase7_preds")

    def dist_stats(S, pT):
        S = np.asarray(S, float); pT = np.asarray(pT, float)
        k = np.arange(1, 13)
        mean_w = (pT * k).sum(1) + 13 * S[:, -1]
        F = 1 - S                                                        # P(T <= k), k = 1..12
        q = lambda p: np.where((F >= p).any(1), 1 + np.argmax(F >= p, 1), 13)
        return mean_w, q(0.10), q(0.50), q(0.90)

    def err(pred_days, act_days):
        e = np.abs(pred_days - act_days)
        return dict(median_abs_err_days=float(np.median(e)), mae_days=float(np.mean(e)),
                    within_3d=float(np.mean(e <= 3)), within_7d=float(np.mean(e <= 7)))
    arms = {}
    for cal in ("raw", "recal"):
        for s in SEEDS:
            d = f"{BA}/v8_lite_h4_lr0.00025_s{s}"
            zt = dict(np.load(f"{d}/preds_test.npz"))
            if cal == "recal":
                zt = {**zt, **L.apply_recalibration(json.load(open(f"{d}/recalibration.json")), "arrival_week", zt)}
            ev = zt["EV"].astype(bool); act = 7 * (zt["Y"] - Rt)
            mean_w, p10, p50, p90 = dist_stats(zt["S"], zt["pT"])
            arms.setdefault(f"h4_mean|{cal}", []).append(err((7 * mean_w - 7 * Rt)[ev], act[ev]))
            arms.setdefault(f"h4_p50|{cal}", []).append(err((7 * p50 - 7 * Rt)[ev], act[ev]))
            arms.setdefault(f"h4_interval_coverage_p10_p90|{cal}", []).append(
                dict(coverage=float(np.mean((zt["Y"][ev] >= p10[ev]) & (zt["Y"][ev] <= p90[ev])))))
    z0 = dict(np.load(f"{BA}/v8_lite_h4_lr0.00025_s7/preds_test.npz")); ev0 = z0["EV"].astype(bool); act0 = 7 * (z0["Y"] - Rt)
    zc = np.load(f"{PR}/v8_arrival_week_naive_channel_median_test.npz")
    arms["channel_median_constant|raw"] = [err((7 * zc["P"] - 7 * Rt)[ev0], act0[ev0])]
    arms["reference_only_zero_lateness|raw"] = [err(np.zeros(int(ev0.sum())), act0[ev0])]
    for s in SEEDS:
        zb = np.load(f"{PR}/v8_arrival_week_b5flat_reg_s{s}_test.npz")
        arms.setdefault("b5flat_reg_lightgbm|raw", []).append(err((7 * zb["P"] - 7 * Rt)[ev0], act0[ev0]))
    summ = {a: {k: band([x[k] for x in v]) for k in v[0]} for a, v in arms.items()}
    # ---- one explanation row (seed 7, first uncensored test row), fields only -- the LLM may restate nothing else
    i = int(np.flatnonzero(ev0)[0]); row = lb.iloc[ot[i]]
    t0 = pd.Timestamp(row.snapshot_date); chan = row.key
    hist = channel_lead_history("v8")
    used = hist[(hist.channel_id == chan) & (pd.to_datetime(hist.recorded_ts) <= t0)]
    mean_w, p10, p50, p90 = dist_stats(z0["S"][i:i + 1], z0["pT"][i:i + 1])
    W = P5.TS.load_world("v8"); nz = np.load(f"{BA}/v8_lite_h4_lr0.00025_s7/normaliser.npz")
    t = int((t0 - W["w0"]).days // 7); ci = W["cidx"][chan]
    x = np.concatenate([W["panel"][ci, t], W["miss"][ci, t]]).astype(float)
    li = nz["log1p_idx"].astype(int); x[li] = np.log1p(np.clip(x[li], 0, None)); zsc = (x - nz["mu"]) / nz["sd"]
    names = list(W["meta"]["cols"]) + ["obs:" + c for c in W["meta"]["nullable"]]
    top = [dict(signal=names[j], z=round(float(zsc[j]), 2)) for j in np.argsort(-zsc)[:3] if not names[j].startswith("obs:")]
    expl = dict(prediction=dict(expected_lateness_days=round(float(7 * mean_w[0] - 7 * Rt[i]), 1),
                                median_lateness_days=round(float(7 * p50[0] - 7 * Rt[i]), 1)),
                interval=dict(p10_days=round(float(7 * p10[0] - 7 * Rt[i]), 1), p90_days=round(float(7 * p90[0] - 7 * Rt[i]), 1),
                              note="13 weeks = 'later than 12 weeks' (censored mass)"),
                reference=dict(days=round(float(7 * Rt[i]), 1), kind="as-of channel median observed lead + c (c = 4.571 wk)",
                               as_of_date=str(t0.date())),
                data_support=dict(deliveries_behind_reference=int(len(used))),
                top_signals=top, signal_definition="channel inputs at t0 with the highest z-score against the training normaliser",
                caveat="Correlational model. Signals are inputs that are elevated for this channel, not causes; no action is prescribed.",
                channel=chan, actual_lateness_days=round(float(act0[i]), 1))
    out = dict(stamp=st, reference=dict(kind="asof_channel_lead", meta={k: v for k, v in rmeta.items() if not isinstance(v, (list, dict))}),
               n_test=int(len(ev0)), n_uncensored=int(ev0.sum()), n_censored=int((~ev0).sum()), summary=summ, explanation_row=expl,
               note="point error is independent of R: (pred - R) - (Y - R) = pred - Y")
    C.dump(out, f"{OUT}/a3.json")
    print(json.dumps(dict(n_uncensored=out["n_uncensored"], n_censored=out["n_censored"], summary=summ, explanation_row=expl), indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    for m in ("a1", "a2", "a3"):
        if mode in (m, "all"):
            globals()[m]()
