"""Phase 23AC T1 -- the order-time product against three as-of baselines, on the at-placement TEST lines (pre-registration
reports/part2/phase-23ac-preregistration.md, "T1"). No model is trained or refitted; LightGBM is loaded only to serve the
sample cards through the persisted bundle (v8), and torch is never imported (asserted).

Rows: the Phase 21 / 22 at-placement rows (grpstats place store; tau = the Monday on or before creation), test fold = 2025
creations (folds.fixed_split). Labels: grpstats.placement_labels (Y = weeks to first receipt; EV = receipted; contract weeks).
Product date (weeks from creation):
  v8       shrunk KM median (k = 10) from the store + the Phase 22 validation offset a (order_time_v8clean.json); 80% interval
           = the Phase 22 month-specific split-conformal quantiles from that json (pooled fallback)
  v8w1002  shrunk KM median with k = 3 (Phase 22 world-2 validation); offset a = median over VALIDATION receipted lines of
           (Y - KM); conformal (q10 "lower", q90 "higher") on validation residuals 7 (Y - pred), month-specific iff the
           validation per-month coverage of the pooled interval spreads by > 0.05 (Phase 22's rule, phase22_order_time.py)
Baselines, as-of tau, from po_lines.csv / grn_lines.csv of the world:
  promise          predicted lead = contracted lead (weeks)
  trailing_mean    mean first-receipt lead of the channel's lines created in [tau - 52 weeks, tau) whose first receipt row
  trailing_median  (min event_ts) has recorded_ts <= tau (and the line's own recorded_ts <= tau); median likewise;
                   contracted lead where the channel has none (share reported). Every receipt used is asserted
                   recorded <= tau; a constructed un-gated window must break that assertion (falsification).
Metrics on receipted test lines: A3 (median |err| days), MAE days, share |err| <= 7 days, signed bias (pred - actual, days);
product interval coverage / mean width overall and per creation month. Creation-week-block bootstrap (blocks = unique tau),
1,000 resamples, rng 2311: a CI per metric and for (baseline - product) on A3 / MAE, (product - baseline) on share-within-7.
Leak gate: the product flag's feature list read from the RAW panel must be flagged by leak_guard.check_feature_list
(panel="raw") and the same list on the clean panel must not; run in a child interpreter so its imports cannot meet LightGBM.
Sample cards (v8): numpy default_rng(2310) draws 5 receipted test lines; if none misses by > 14 days the first test line
that does is added; their cards come from order_time_card.OrderTimeCard.

  python ml/eval/phase23ac_t1_compare.py --world v8|v8w1002
  -> ml/artifacts/phase23ac/t1/compare_{world}.json, .log; reports/part2/phase23ac/t1/compare_{world}.csv;
     reports/part2/phase23ac/t1/sample_cards.json (v8)
"""
from __future__ import annotations
import os, sys, json, time, argparse, subprocess, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..")
sys.path[:0] = [HERE, ML, os.path.join(ML, "data"), os.path.join(ML, "serve"), os.path.join(ML, "baselines"),
                os.path.join(ML, "train")]
import numpy as np, pandas as pd
import phase12_common as C
import config, folds
import phase21_paths as PP
import grpstats as GS

B = 1000
BOOT_SEED = 2311
CARD_SEED = 2310
DAY = np.timedelta64(1, "D")
OUT_ART = os.path.join(config.ARTIFACTS, "phase23ac", "t1")
OUT_REP = os.path.join(config.REPO, "reports", "part2", "phase23ac", "t1")
OT_JSON = {"v8": os.path.join(config.ARTIFACTS, "phase22", "order_time_v8clean.json"),
           "v8w1002": os.path.join(config.ARTIFACTS, "phase22", "order_time_v8w1002clean.json")}
PRODUCT_JSON = os.path.join(config.ARTIFACTS, "phase22", "serve", "order_time_v8clean", "product.json")
PREDICTORS = ("product", "promise", "trailing_mean", "trailing_median")
BASELINES = PREDICTORS[1:]


def log(msg, path=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------ baselines (as-of tau)
def receipt_table(world):
    """Every PO line with its creation, own recorded_ts and FIRST receipt (min event_ts; its recorded_ts)."""
    D = config.WORLDS[world]
    pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "channel_id", "created_ts", "recorded_ts"])
    g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "event_ts", "recorded_ts"])
    g["event_ts"] = pd.to_datetime(g.event_ts); g["recorded_ts"] = pd.to_datetime(g.recorded_ts)
    g = g.sort_values(["po_line_id", "event_ts", "recorded_ts"], kind="mergesort").drop_duplicates("po_line_id", keep="first")
    t = pl.merge(g.rename(columns={"event_ts": "g_event", "recorded_ts": "g_rec"}), on="po_line_id", how="left")
    assert len(t) == len(pl)
    t["created_ts"] = pd.to_datetime(t.created_ts); t["recorded_ts"] = pd.to_datetime(t.recorded_ts)
    return t


class Trailing:
    def __init__(self, t, lo):
        keep = (t.created_ts >= lo).to_numpy() & t.g_event.notna().to_numpy()
        s = t[keep]
        self.cid = s.channel_id.to_numpy()
        self.created = s.created_ts.to_numpy("datetime64[ns]")
        self.line_rec = s.recorded_ts.to_numpy("datetime64[ns]")
        self.g_rec = s.g_rec.to_numpy("datetime64[ns]")
        self.lead_days = ((s.g_event.to_numpy("datetime64[ns]") - self.created) / DAY).astype(float)

    def at(self, tau, gated=True):
        """-> (DataFrame indexed by channel_id with mean / median / n lead days, used positions)."""
        tau = np.datetime64(tau, "ns")
        sel = (self.created >= tau - np.timedelta64(52 * 7, "D")) & (self.created < tau)
        if gated:
            sel &= (self.g_rec <= tau) & (self.line_rec <= tau)
        pos = np.flatnonzero(sel)
        df = pd.DataFrame({"c": self.cid[pos], "lead": self.lead_days[pos]})
        g = df.groupby("c").lead.agg(["mean", "median", "size"])
        return g, pos

    def assert_asof(self, pos, tau):
        tau = np.datetime64(tau, "ns")
        bad = int((self.g_rec[pos] > tau).sum() + (self.line_rec[pos] > tau).sum())
        if bad:
            raise GS.AsOfLeak(f"{bad} receipts / lines used at {tau} were recorded after it")


def baselines(world, ent, tau_t, cweeks, logp):
    t = receipt_table(world)
    lo = pd.Timestamp(np.min(tau_t)) - pd.Timedelta(days=52 * 7 + 7)
    TR = Trailing(t, lo)
    chan = t.set_index("po_line_id").channel_id.reindex(ent).to_numpy()
    assert not pd.isna(chan).any(), "a test line missing from po_lines.csv"
    mean = np.full(len(ent), np.nan); med = np.full(len(ent), np.nan); n = np.zeros(len(ent))
    taus = np.unique(tau_t)
    for u in taus:
        r = np.flatnonzero(tau_t == u)
        g, pos = TR.at(u)
        TR.assert_asof(pos, u)                                   # every receipt used is recorded <= tau
        gg = g.reindex(chan[r])
        mean[r] = gg["mean"].to_numpy() / 7.0; med[r] = gg["median"].to_numpy() / 7.0
        n[r] = np.nan_to_num(gg["size"].to_numpy())
    # falsification: the same window WITHOUT the recorded_ts gate must be caught by the assertion
    u_last = taus[-1]
    _, pos_ng = TR.at(u_last, gated=False)
    try:
        TR.assert_asof(pos_ng, u_last); fired = False
    except GS.AsOfLeak:
        fired = True
    n_late = int((TR.g_rec[pos_ng] > np.datetime64(u_last, "ns")).sum())
    if not fired:
        log(f"WARNING: the un-gated window at {str(u_last)[:10]} used no late-recorded receipt; the as-of assertion was not "
            "exercised on a failing case", logp)
    fb = np.isnan(mean)
    mean[fb] = cweeks[fb]; med[fb] = cweeks[fb]
    info = dict(fallback_share=float(fb.mean()), fallback_lines=int(fb.sum()), n_test_taus=int(len(taus)),
                trailing_lines_per_row_median=float(np.median(n)), trailing_lines_per_row_p10=float(np.quantile(n, 0.10)),
                asof_falsification=dict(tau=str(u_last)[:10], ungated_receipts_recorded_after_tau=n_late, assertion_fired=fired))
    return {"promise": cweeks.copy(), "trailing_mean": mean, "trailing_median": med}, info


# ------------------------------------------------------------------ product
def month_q(mq, m, pooled):
    if mq is None:
        return pooled
    v = mq.get(str(int(m)), mq.get(int(m)) if isinstance(mq, dict) else None)
    return tuple(v) if v is not None else pooled


def conformal_phase22(rv, mv):
    """Phase 22's rule (phase22_order_time.py): pooled (q10 lower, q90 higher); month-specific iff validation per-month
    coverage of the pooled interval spreads by > 0.05."""
    q10, q90 = float(np.quantile(rv, 0.10, method="lower")), float(np.quantile(rv, 0.90, method="higher"))
    cov_v = {int(m): float(((rv[mv == m] >= q10) & (rv[mv == m] <= q90)).mean()) for m in np.unique(mv)}
    month_specific = (max(cov_v.values()) - min(cov_v.values())) > 0.05
    qm = {}
    if month_specific:
        for m in np.unique(mv):
            r = rv[mv == m]
            qm[str(int(m))] = [float(np.quantile(r, 0.10, method="lower")), float(np.quantile(r, 0.90, method="higher"))]
    return dict(q10_days=q10, q90_days=q90, validation_month_coverage_of_pooled=cov_v, month_specific_used=bool(month_specific),
                month_quantiles=qm)


# ------------------------------------------------------------------ metrics and bootstrap
def point_metrics(e):
    """e: signed error days (pred - actual) on receipted lines."""
    a = np.abs(e)
    return dict(a3_days=float(np.median(a)), mae_days=float(a.mean()), share_within_7d=float((a <= 7).mean()),
                bias_days=float(e.mean()))


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bootstrap(errs, inside, width, month, blocks):
    """errs: {predictor: signed error days on receipted rows}; blocks: lists of receipted-row positions per creation week."""
    rng = np.random.default_rng(BOOT_SEED)
    months = np.arange(1, 13)
    bs = {p: {m: np.empty(B) for m in ("a3_days", "mae_days", "share_within_7d", "bias_days")} for p in errs}
    cov, wid, covm = np.empty(B), np.empty(B), np.full((B, 12), np.nan)
    for b in range(B):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        for p, e in errs.items():
            for m, v in point_metrics(e[idx]).items():
                bs[p][m][b] = v
        cov[b] = inside[idx].mean(); wid[b] = width[idx].mean()
        cnt = np.bincount(month[idx], minlength=13)[1:]
        hit = np.bincount(month[idx], weights=inside[idx].astype(float), minlength=13)[1:]
        covm[b] = np.where(cnt > 0, hit / np.maximum(cnt, 1), np.nan)
    out = dict(ci={p: {m: ci(v) for m, v in d.items()} for p, d in bs.items()}, diff={})
    for base in BASELINES:
        if base not in errs:
            continue
        for m, sign in (("a3_days", +1), ("mae_days", +1), ("share_within_7d", -1)):
            d = (bs[base][m] - bs["product"][m]) if sign > 0 else (bs["product"][m] - bs[base][m])
            lo, up = np.percentile(d, [2.5, 97.5])
            out["diff"].setdefault(m, {})[f"{base} vs product"] = dict(
                definition="baseline - product" if sign > 0 else "product - baseline", positive_means="product better",
                mean=float(d.mean()), ci=[float(lo), float(up)],
                verdict="better" if lo > 0 else "worse" if up < 0 else "undetermined")
    out["interval_ci"] = dict(coverage=ci(cov), width_days=ci(wid),
                              coverage_by_month={int(m): [float(np.nanpercentile(covm[:, i], 2.5)), float(np.nanpercentile(covm[:, i], 97.5))]
                                                 for i, m in enumerate(months) if np.isfinite(covm[:, i]).any()})
    return out


# ------------------------------------------------------------------ leak gate (child interpreter)
def leak_gate(columns):
    paths = [ML, HERE, os.path.join(ML, "data"), os.path.join(ML, "serve"), os.path.join(ML, "train"), os.path.join(ML, "baselines")]
    code = ("import sys, json\nsys.path[:0] = json.loads(sys.argv[1])\nimport leak_guard as LG\ncols = json.loads(sys.argv[2])\n"
            "r = dict(raw=list(LG.check_feature_list(cols, panel='raw')), clean=list(LG.check_feature_list(cols, panel='clean')),\n"
            "         module=LG.__file__)\nopen(sys.argv[3], 'w').write(json.dumps(r))\n")
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        r = subprocess.run([sys.executable, "-c", code, json.dumps(paths), json.dumps(list(columns)), tmp], cwd=config.REPO,
                           capture_output=True, text=True, env=dict(os.environ, HADES_DEVICE="cpu"))
        if r.returncode != 0:
            raise RuntimeError(f"leak_guard could not be run (import or call failed):\n{r.stderr[-3000:]}")
        res = json.load(open(tmp))
    finally:
        os.remove(tmp)
    import clean_panel                                     # numpy / pandas only: the nine Phase 22 leaking columns
    nine_in_list = [c for c in clean_panel.LEAKING if c in columns]
    res["nine_in_feature_list"] = nine_in_list
    res["leaked_arm_flagged"] = bool(res["raw"])
    res["all_nine_flagged_raw"] = set(nine_in_list) <= set(res["raw"])
    res["clean_arm_unflagged"] = not res["clean"]
    assert res["leaked_arm_flagged"], "T1 leak gate: the leaked arm (flag features on the RAW panel) was NOT flagged"
    assert res["all_nine_flagged_raw"], f"T1 leak gate: leaking columns in the list not flagged: {set(nine_in_list) - set(res['raw'])}"
    assert res["clean_arm_unflagged"], f"T1 leak gate: the clean feature list was flagged: {res['clean']}"
    return res


# ------------------------------------------------------------------ sample cards (v8)
def sample_cards(src, li_t, Y_t, EV_t, pred_t, logp):
    rng = np.random.default_rng(CARD_SEED)
    ev_pos = np.flatnonzero(EV_t)
    pick = [int(x) for x in rng.choice(ev_pos, 5, replace=False)]
    err = 7 * (pred_t - Y_t)
    added = None
    if not (np.abs(err[pick]) > 14).any():
        big = np.flatnonzero(EV_t & (np.abs(err) > 14))
        if len(big):
            added = int(big[0]); pick.append(added)
    li = li_t[pick]
    lines = pd.DataFrame({"po_line_id": src.line_id[li], "channel_id": src.ch.channel_id.to_numpy()[src.chan[li]],
                          "created_ts": pd.DatetimeIndex(src.created[li]), "qty_ordered": src.qty[li]})
    assert "torch" not in sys.modules, "torch is loaded; LightGBM must not share this process"
    import order_time_card as OC                          # LightGBM enters the process here (serving the bundle only)
    cards = OC.OrderTimeCard().cards(lines)
    out = []
    for j, (c, p) in enumerate(zip(cards, pick)):
        realised = pd.Timestamp(src.g_event[li[j]])
        exp = pd.Timestamp(c["expected_receipt_date"])
        out.append(dict(card=c, test_position=p, realised_receipt_date=str(realised.date()),
                        realised_lead_weeks=float(Y_t[p]), error_days_pred_minus_actual=float(err[p]),
                        realised_minus_expected_date_days=int((realised.normalize() - exp).days),
                        inside_interval=bool(pd.Timestamp(c["interval_earliest"]) <= realised.normalize() <= pd.Timestamp(c["interval_latest"])),
                        served_minus_store_weeks=float(c["_expected_lead_weeks"] - pred_t[p]),
                        added_as_first_miss_over_14d=(p == added)))
    log(f"sample cards: {len(out)} (added first >14 d miss: {added is not None})", logp)
    return dict(seed=CARD_SEED, drawn=5, added_first_miss_over_14d=added, cards=out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["v8", "v8w1002"], default="v8")
    a = ap.parse_args()
    st = C.require_clean()
    world = a.world
    os.makedirs(OUT_ART, exist_ok=True); os.makedirs(OUT_REP, exist_ok=True)
    logp = os.path.join(OUT_ART, f"compare_{world}.log")
    open(logp, "w").close()
    t0 = time.time()
    paths = PP.register()
    log(f"Phase 23AC T1 compare, world {world}, stamp {st['code_version']}", logp)
    GS.OUT = os.path.join(config.ARTIFACTS, "phase21") if world == "v8" else os.path.join(config.ARTIFACTS, "phase22")
    Z, _ = GS.load(world, "place")
    src = GS.Source(world)
    li = Z["line"].astype(np.int64)
    Y, EV, cweeks, qty = GS.placement_labels(src, li)
    created = pd.DatetimeIndex(src.created[li])
    tr, va, te = [np.asarray(x, bool) for x in folds.fixed_split(created)]
    tau = pd.to_datetime(Z["tau"]).to_numpy("datetime64[ns]")
    sub = lambda m: {k: v[m] for k, v in Z.items() if hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(li)}
    ot = json.load(open(OT_JSON[world]))
    log(f"rows {len(li)}; val {int(va.sum())}; test {int(te.sum())} (receipted {int(EV[te].sum())})", logp)

    # ---- labels reproduce the Phase 22 stored rows (entity order, Y, EV)
    repro = {}
    zp = os.path.join(config.ARTIFACTS, "phase22", "preds", f"{world}clean_arrival_place_p22_base_s7_test.npz")
    ti = np.flatnonzero(te)
    if os.path.exists(zp):
        z0 = np.load(zp)
        assert (np.asarray(z0["entity"]).astype(str) == Z["entity"][ti]).all(), "test rows differ from Phase 22's stored rows"
        repro["stored_rows_Y_max_abs_diff"] = float(np.abs(np.asarray(z0["Y"], float) - Y[ti]).max())
        repro["stored_rows_EV_equal"] = bool((np.asarray(z0["EV"], bool) == EV[ti]).all())
        assert repro["stored_rows_Y_max_abs_diff"] < 1e-9 and repro["stored_rows_EV_equal"], repro

    # ---- product date and interval
    if world == "v8":
        k = 10
        a_off = float(ot["date_estimators"]["km_offset_a_weeks"])
        iv = ot["interval"]
        conf = dict(source=os.path.relpath(OT_JSON[world], config.REPO), q10_days=iv["q10_days"], q90_days=iv["q90_days"],
                    month_specific_used=iv["month_specific_used"], month_quantiles=iv["month_quantiles"] if iv["month_specific_used"] else {})
    else:
        k = 3
        assert "k = 3" in ot["date_estimators"]["km_note"], ot["date_estimators"]["km_note"]
        km_v = GS.standalone_arrival_weeks(sub(va), k)
        ev_v = EV[va]
        a_off = float(np.median(Y[va][ev_v] - km_v[ev_v]))
        rv = 7 * (Y[va][ev_v] - (km_v[ev_v] + a_off))
        mv = created[va].month.to_numpy()[ev_v]
        conf = conformal_phase22(rv, mv)
        conf["source"] = "recomputed here on world-2 validation (Phase 22 rule)"
        repro["offset_vs_phase22_json_abs_diff"] = abs(a_off - float(ot["date_estimators"]["km_offset_a_weeks"]))
        repro["phase22_json_product_estimator"] = ot["date_estimators"]["chosen_on_validation"]
    pred = GS.standalone_arrival_weeks(sub(te), k) + a_off
    Y_t, EV_t, cw_t, li_t, tau_t = Y[te], EV[te], cweeks[te], li[te], tau[te]
    m_t = created[te].month.to_numpy()
    pooled = (conf["q10_days"], conf["q90_days"])
    mq = conf["month_quantiles"] if conf["month_specific_used"] else None
    q = np.array([month_q(mq, m, pooled) for m in m_t], float)
    r_t = 7 * (Y_t - pred)
    inside = (r_t >= q[:, 0]) & (r_t <= q[:, 1]); width = q[:, 1] - q[:, 0]
    km_a3 = float(np.median(np.abs(7 * (pred - Y_t))[EV_t]))
    phase22_km = ot["date_estimators"]["test"].get("km", {}).get("a3_median_abs_err_days")
    repro["km_test_a3_here"] = km_a3; repro["km_test_a3_phase22_json"] = phase22_km
    if phase22_km is not None:
        repro["km_test_a3_abs_diff"] = abs(km_a3 - phase22_km)
        assert repro["km_test_a3_abs_diff"] < 1e-9, f"KM test A3 does not reproduce Phase 22: {km_a3} vs {phase22_km}"
    if world == "v8":
        repro["test_coverage_here"] = float(inside[EV_t].mean()); repro["test_coverage_phase22_json"] = ot["interval"]["test_coverage"]
        assert abs(repro["test_coverage_here"] - repro["test_coverage_phase22_json"]) < 1e-9, repro
    log(f"product: k {k}, offset {a_off:.4f} w; KM test A3 {km_a3:.3f} d (Phase 22 json {phase22_km}); reproduction {repro}", logp)

    # ---- baselines
    ent = Z["entity"][te]
    base, binfo = baselines(world, ent, tau_t, cw_t, logp)
    prom_days = ((src.promise[li_t] - src.created[li_t]) / DAY).astype(float)
    binfo["promise_date_minus_created_vs_contract_days"] = dict(median_abs_diff=float(np.nanmedian(np.abs(prom_days - 7 * cw_t))),
                                                                max_abs_diff=float(np.nanmax(np.abs(prom_days - 7 * cw_t))))
    log(f"baselines: {binfo}", logp)
    preds = {"product": pred, **base}

    # ---- metrics (receipted test lines), bootstrap over creation-week blocks
    ev = np.flatnonzero(EV_t)
    errs = {p: 7 * (v[ev] - Y_t[ev]) for p, v in preds.items()}
    tau_ev = tau_t[ev]
    blocks = [np.flatnonzero(tau_ev == u) for u in np.unique(tau_t)]
    point = {p: point_metrics(e) for p, e in errs.items()}
    m_ev = m_t[ev].astype(np.int64)
    interval = dict(coverage=float(inside[ev].mean()), width_days_mean=float(width[ev].mean()),
                    coverage_by_month={int(m): float(inside[ev][m_ev == m].mean()) for m in np.unique(m_ev)},
                    width_by_month={int(m): float(width[ev][m_ev == m].mean()) for m in np.unique(m_ev)},
                    n_by_month={int(m): int((m_ev == m).sum()) for m in np.unique(m_ev)}, conformal=conf)
    log("bootstrap ...", logp)
    boot = bootstrap(errs, inside[ev], width[ev], m_ev, blocks)
    interval["ci"] = boot.pop("interval_ci")
    log(f"point: {json.dumps(point)}", logp)
    log(f"diffs: {json.dumps(boot['diff'])}", logp)
    log(f"interval: coverage {interval['coverage']:.4f} {interval['ci']['coverage']}, width {interval['width_days_mean']:.2f} d", logp)

    # ---- scoring per the pre-registration (P1 / P2 are scored on v8; world 2 is the replication)
    p1 = dict(a3_block_ci_excludes_0_in_product_favour={b: boot["diff"]["a3_days"][f"{b} vs product"]["verdict"] == "better" for b in BASELINES},
              share_within_7_point_higher={b: point["product"]["share_within_7d"] > point[b]["share_within_7d"] for b in BASELINES})
    p1["right"] = all(p1["a3_block_ci_excludes_0_in_product_favour"].values()) and all(p1["share_within_7_point_higher"].values())
    covs = list(interval["coverage_by_month"].values())
    p2 = dict(overall_in_band=0.77 <= interval["coverage"] <= 0.83, every_month_in_band=all(0.77 <= c <= 0.83 for c in covs),
              months_out_of_band={m: c for m, c in interval["coverage_by_month"].items() if not 0.77 <= c <= 0.83})
    p2["right"] = p2["overall_in_band"] and p2["every_month_in_band"]

    # ---- leak gate
    feats = json.load(open(PRODUCT_JSON))["flag"]["models"][0]["features"]
    lg = leak_gate(feats)
    log(f"leak gate: raw flagged {lg['raw']}; clean flagged {lg['clean']}", logp)

    out = dict(stamp=st, world=world, data_paths=paths, phase="23AC T1",
               rows=dict(test=int(te.sum()), test_receipted=int(len(ev)), creation_week_blocks=len(blocks)),
               product=dict(k=k, km_offset_a_weeks=a_off, date="shrunk KM median (L5 chain) + offset", interval=interval),
               baselines_info=binfo, point=point, ci=boot["ci"], diff=boot["diff"], bootstrap=dict(B=B, seed=BOOT_SEED, block="creation week (tau)"),
               reproduction=repro, leak_gate=lg, prereg_scoring=dict(P1=p1, P2=p2, note="P1/P2 are scored on v8; v8w1002 is the replication"),
               sources=dict(script=os.path.relpath(os.path.abspath(__file__), config.REPO),
                            store=os.path.relpath(os.path.join(GS.OUT, f"grpstats_{world}_place.npz"), config.REPO),
                            order_time_json=os.path.relpath(OT_JSON[world], config.REPO),
                            product_bundle=os.path.relpath(PRODUCT_JSON, config.REPO),
                            csv_hashes=PP.world_hashes(world, ("po_lines.csv", "grn_lines.csv", "sourcing_channels.csv"))))

    if world == "v8":
        sc = sample_cards(src, li_t, Y_t, EV_t, pred, logp)
        sc.update(stamp=st, world=world)
        with open(os.path.join(OUT_REP, "sample_cards.json"), "w") as f:
            json.dump(sc, f, indent=1, default=str)
        out["sample_cards"] = os.path.relpath(os.path.join(OUT_REP, "sample_cards.json"), config.REPO)
    assert "torch" not in sys.modules, "torch was imported in this process"

    rows = []
    n_ev = int(len(ev))
    for p in PREDICTORS:
        for m, v in point[p].items():
            lo, up = boot["ci"][p][m]
            rows.append(dict(world=world, predictor=p, metric=m, value=v, ci_lo=lo, ci_hi=up, verdict="", n_lines=n_ev))
    for m, d in boot["diff"].items():
        for name, r in d.items():
            rows.append(dict(world=world, predictor=name, metric=f"{m} ({r['definition']}; + = product better)", value=r["mean"],
                             ci_lo=r["ci"][0], ci_hi=r["ci"][1], verdict=r["verdict"], n_lines=n_ev))
    rows.append(dict(world=world, predictor="product", metric="interval_coverage", value=interval["coverage"],
                     ci_lo=interval["ci"]["coverage"][0], ci_hi=interval["ci"]["coverage"][1], verdict="", n_lines=n_ev))
    rows.append(dict(world=world, predictor="product", metric="interval_width_days", value=interval["width_days_mean"],
                     ci_lo=interval["ci"]["width_days"][0], ci_hi=interval["ci"]["width_days"][1], verdict="", n_lines=n_ev))
    for m, c in interval["coverage_by_month"].items():
        lo, up = interval["ci"]["coverage_by_month"].get(m, [np.nan, np.nan])
        rows.append(dict(world=world, predictor="product", metric=f"interval_coverage_month_{m:02d}", value=c, ci_lo=lo, ci_hi=up,
                         verdict="", n_lines=interval["n_by_month"][m]))
        rows.append(dict(world=world, predictor="product", metric=f"interval_width_days_month_{m:02d}", value=interval["width_by_month"][m],
                         ci_lo=np.nan, ci_hi=np.nan, verdict="", n_lines=interval["n_by_month"][m]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_REP, f"compare_{world}.csv"), index=False)
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_ART, f"compare_{world}.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"P1 {p1['right']} {p1}; P2 {p2['right']} {p2}", logp)
    log(f"wrote compare_{world}.json / .csv in {out['seconds']} s", logp)


if __name__ == "__main__":
    main()
