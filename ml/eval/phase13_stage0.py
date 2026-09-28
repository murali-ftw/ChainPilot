"""Phase 13 Stage 0 -- cheap checks that can cancel work. CPU only, no training (the head's training-fold figures are
INFERENCE with existing checkpoints, as in Phase 12 A2). Every gate is computed here and recorded, not argued.

  0.1 schema: what a channel is; can "closed with nothing received" be identified AS-OF?
  0.2 the unified baseline table: arm x metric x split, 5-seed bands where the arm has seeds
  0.3 S1's arithmetic: observed shortfall with net transfers added back, against 298 and 564
  0.4 is b5flat22 SCOREABLE on the eval path (not servable -- deviation 46)?
  0.5 variance decomposition of line fill over part / supplier / plant; plant's marginal share after supplier
  0.6 closed-line counts per candidate key at a representative t0, and the scoring-row coverage fractions

inventory_position_weekly is read in 0.3 ONLY (qty_on_hand, safety_stock_qty), as the held-out evaluation reference
Phase 9.1's validation already uses. Never as a feature.
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd, torch
import phase5_heads as P5, folds, loop as L
import phase5_metrics as M
from heads import fill_cell

D = C.REPO + "/db/gen_v8/seed_1001"
T0 = pd.Timestamp("2025-01-06")          # the first 2025 test snapshot: the representative as-of instant
LOOKBACK_W = 104
INTERIOR = np.arange(1, 21)


# ================================================================== 0.1 + the as-of closed-line table
def closed_lines(t0, lookback_w=LOOKBACK_W, with_meta=False):
    """Every PO line CLOSED as-of t0 with its realised fill, from rows recorded <= t0 only.

    closed-received : a GRN line with is_final_receipt, recorded <= t0 -> fill = sum received (recorded <= t0) / ordered
    closed-zero     : supplier acknowledgement 'rejected' recorded <= t0, OR a qty revision to 0 recorded <= t0,
                      and no GRN recorded <= t0 -> fill = 0
    Lines created in (t0 - lookback, t0]."""
    t0 = pd.Timestamp(t0)
    pol = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "part_id", "channel_id", "qty_ordered",
                                                    "created_ts", "recorded_ts"])
    pol["created_ts"] = pd.to_datetime(pol.created_ts); pol["recorded_ts"] = pd.to_datetime(pol.recorded_ts)
    pol = pol[(pol.recorded_ts <= t0) & (pol.created_ts > t0 - pd.Timedelta(weeks=lookback_w))]
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "plant_id"])
    pol = pol.merge(ch, on="channel_id")
    g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "qty_received", "is_final_receipt", "recorded_ts"])
    g["recorded_ts"] = pd.to_datetime(g.recorded_ts)
    g = g[g.recorded_ts <= t0]
    rec = g.groupby("po_line_id").qty_received.sum()
    fin = set(g[g.is_final_receipt.astype(str).str.lower().isin(["true", "1"])].po_line_id)
    ack = pd.read_csv(f"{D}/supplier_acknowledgements.csv", usecols=["po_line_id", "ack_status", "recorded_ts"])
    ack = ack[(pd.to_datetime(ack.recorded_ts) <= t0) & (ack.ack_status == "rejected")]
    rev = pd.read_csv(f"{D}/po_line_revisions.csv", usecols=["po_line_id", "field_changed", "new_value", "recorded_ts"])
    rev = rev[(pd.to_datetime(rev.recorded_ts) <= t0) & (rev.field_changed == "qty")
              & (pd.to_numeric(rev.new_value, errors="coerce") == 0)]
    zero = set(ack.po_line_id) | set(rev.po_line_id)
    has_grn = set(rec.index)
    pol["closed_received"] = pol.po_line_id.isin(fin)
    pol["closed_zero"] = pol.po_line_id.isin(zero) & ~pol.po_line_id.isin(has_grn)
    pol = pol[pol.closed_received | pol.closed_zero].copy()
    pol["arrived"] = np.where(pol.closed_zero, 0.0, pol.po_line_id.map(rec).fillna(0.0))
    pol["fill"] = np.clip(pol.arrived / pol.qty_ordered.clip(lower=1), 0, 1)
    return pol


def stage01():
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "site_id", "part_id", "plant_id"])
    po = pd.read_csv(f"{D}/purchase_orders.csv", usecols=["status"])
    pol = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "created_ts", "recorded_ts"])
    pol["created_ts"] = pd.to_datetime(pol.created_ts)
    g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id"])
    old = pol[pol.created_ts < pd.Timestamp("2025-06-01")]
    nog = ~old.po_line_id.isin(set(g.po_line_id))
    ack = pd.read_csv(f"{D}/supplier_acknowledgements.csv", usecols=["po_line_id", "ack_status", "recorded_ts"])
    rej = ack[ack.ack_status == "rejected"].drop_duplicates("po_line_id").set_index("po_line_id")
    rev = pd.read_csv(f"{D}/po_line_revisions.csv", usecols=["po_line_id", "field_changed", "new_value", "recorded_ts"])
    rev0 = rev[(rev.field_changed == "qty") & (pd.to_numeric(rev.new_value, errors="coerce") == 0)] \
        .drop_duplicates("po_line_id").set_index("po_line_id")
    sig = old.po_line_id.isin(set(rej.index) | set(rev0.index))
    # as-of timing: how long after the line is raised is the zero-closure signal RECORDED?
    cr = old.set_index("po_line_id").created_ts
    ia = rej.index.intersection(cr.index); ir = rev0.index.intersection(cr.index)
    lag_ack = (pd.to_datetime(rej.loc[ia, "recorded_ts"]) - cr.loc[ia]).dt.days.to_numpy()
    lag_rev = (pd.to_datetime(rev0.loc[ir, "recorded_ts"]) - cr.loc[ir]).dt.days.to_numpy()
    return dict(
        channel_definition="(part_id, supplier_id, plant_id); site_id is 1:1 with supplier_id",
        n_channels=int(len(ch)), unique_triples=int(ch[["part_id", "supplier_id", "plant_id"]].drop_duplicates().shape[0]),
        unique_quads_with_site=int(ch[["part_id", "supplier_id", "site_id", "plant_id"]].drop_duplicates().shape[0]),
        po_status_values=po.status.value_counts().to_dict(),
        lines_created_before_2025_06=int(len(old)), with_no_grn_ever=int(nog.sum()), share_no_grn=float(nog.mean()),
        zero_closure_signal_among_no_grn=int((sig & nog).sum()),
        zero_closure_signal_among_received=int((sig & ~nog).sum()),
        no_grn_without_signal=int((~sig & nog).sum()),
        signal_recorded_lag_days=dict(ack_rejected_median=float(np.median(lag_ack)), ack_rejected_p95=float(np.percentile(lag_ack, 95)),
                                      qty_to_zero_median=float(np.median(lag_rev)), qty_to_zero_p95=float(np.percentile(lag_rev, 95))),
        gate_0_1=("PASS via supplier_acknowledgements.ack_status='rejected' OR a po_line_revisions qty->0, as-of; "
                  "the PO status field itself cannot (one value)"))


# ================================================================== 0.2 the unified baseline table
def four_metrics(P, y):
    P = np.asarray(P, float); y = np.asarray(y, float)
    cells = fill_cell(y)
    pred, obs = P.mean(0), np.bincount(cells, minlength=22) / len(y)
    from metrics import roc_auc
    o1 = (y >= 1).astype(int)
    return dict(interior_abs_err=float(np.abs(pred - obs)[INTERIOR].sum()),
                crps_exact=float(M.crps_exact_rows(P, y).mean()),
                ece22=float(M.ece_marginal(P, cells)[0]),
                roc_auc_p_complete=float(roc_auc(o1, P[:, 21])) if 0 < o1.mean() < 1 else float("nan"),
                n=int(len(y)))


def band_rows(arm, split, calib, per_seed):
    rows = []
    for m in ("interior_abs_err", "crps_exact", "ece22", "roc_auc_p_complete"):
        v = [p[m] for p in per_seed]
        rows.append(dict(world="v8", origin="fixed split", split=split, arm=arm, calib=calib, metric=m,
                         n_rows=per_seed[0]["n"], n_seeds=len(v), point=float(np.mean(v)),
                         lo=float(min(v)) if len(v) > 1 else None, hi=float(max(v)) if len(v) > 1 else None,
                         band="5-seed [lo, hi]" if len(v) >= 5 else ("single-run, no band" if len(v) == 1 else f"{len(v)}-seed")))
    return rows


def stage02():
    rows = []
    lb = P5.labels("v8", "fill_rate")
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    # current 22-cell head, 5 seeds: train by inference, val/test stored; raw and recalibrated (its own recalibrator)
    tr_raw, tr_rec, sp = [], [], {"val": ([], []), "test": ([], [])}
    for s in C.V8_SEEDS:
        bd = f"{C.BUND}/fill_rate/v8_none_h0_lr0.000125_s{s}"
        B = L.load_bundle(bd)
        Din = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), False)
        m = L._materialise(B, Din)
        with torch.no_grad():
            pr = P5.predict(m, Din, lb, tr)
        tr_raw.append(four_metrics(pr["P"], pr["Y"]))
        tr_rec.append(four_metrics(L.apply_recalibration(B["rec"], "fill_rate", pr)["P22"], pr["Y"]))
        for split in ("val", "test"):
            z = dict(np.load(f"{bd}/preds_{split}.npz"))
            sp[split][0].append(four_metrics(z["P"], z["Y"]))
            sp[split][1].append(four_metrics(L.apply_recalibration(B["rec"], "fill_rate", z)["P22"], z["Y"]))
    rows += band_rows("head_cells22_h0", "train", "raw", tr_raw) + band_rows("head_cells22_h0", "train", "recal", tr_rec)
    for split in ("val", "test"):
        rows += band_rows("head_cells22_h0", split, "raw", sp[split][0]) + band_rows("head_cells22_h0", split, "recal", sp[split][1])
    PR = f"{C.ART}/phase7_preds"
    for arm, raw_pat, rec_pat in (("b5flat22", "v8_fill_rate_b5flat22_s{s}_{f}.npz", "RECAL_v8_fill_rate_b5flat22_s{s}_{f}.npz"),
                                  ("lgbm22_id", "v8_fill_rate_lgbm22_id_s{s}_{f}.npz", "RECAL_v8_fill_rate_lgbm22_id_s{s}_{f}.npz")):
        for split in ("val", "test"):
            for calib, pat in (("raw", raw_pat), ("recal", rec_pat)):
                per = []
                for s in C.V8_SEEDS:
                    z = np.load(f"{PR}/{pat.format(s=s, f=split)}")
                    per.append(four_metrics(z["P"], z["Y"]))
                rows += band_rows(arm, split, calib, per)
    for split in ("val", "test"):
        z = np.load(f"{PR}/v8_fill_rate_b2_rolling52_cdf_{split}.npz")
        rows += band_rows("rolling52_asof_histogram", split, "raw", [four_metrics(z["P"], z["Y"])])
    for split in ("train",):
        for arm in ("b5flat22", "lgbm22_id", "rolling52_asof_histogram"):
            for mtr in ("interior_abs_err", "crps_exact", "ece22", "roc_auc_p_complete"):
                rows.append(dict(world="v8", origin="fixed split", split="train", arm=arm, calib="raw", metric=mtr,
                                 n_rows=int(tr.sum()), n_seeds=0, point=None, lo=None, hi=None,
                                 band="NOT STORED: training-fold predictions were never written; a refit is training"))
    return rows


def pin_quoted(rows):
    df = pd.DataFrame([r for r in rows if r["point"] is not None])

    def get(arm, split, calib, metric):
        r = df[(df.arm == arm) & (df.split == split) & (df.calib == calib) & (df.metric == metric)]
        return r.iloc[0].to_dict() if len(r) else None
    return {
        "0.018 'b5flat22 interior, TRAIN marginal'": dict(
            finding="MISLABELLED. 0.018 is b5flat22's RAW interior sum|error| on VALIDATION (Phase 12 A2); its TRAIN "
                    "marginal was never measured (predictions not stored). Same-split figure below.",
            val=get("b5flat22", "val", "raw", "interior_abs_err"), test=get("b5flat22", "test", "raw", "interior_abs_err")),
        "0.063-0.092 'the head'": dict(
            finding="the head's RAW interior sum|error| on VALIDATION across 5 seeds (Phase 12 A2).",
            val=get("head_cells22_h0", "val", "raw", "interior_abs_err"), train=get("head_cells22_h0", "train", "raw", "interior_abs_err"),
            test=get("head_cells22_h0", "test", "raw", "interior_abs_err")),
        "0.13827 exact CRPS": dict(
            finding="the head's RECALIBRATED TEST exact CRPS, quoted in phase-0-1-v8.md:311 with a 3-seed band "
                    "[0.13820, 0.13834]. Restated at 5 seeds below.",
            test_recal=get("head_cells22_h0", "test", "recal", "crps_exact"), test_raw=get("head_cells22_h0", "test", "raw", "crps_exact")),
        "0.0165 ECE": dict(
            finding="rolling-52 as-of histogram, RAW, TEST ECE-22, single deterministic run (phase-11a.md:255).",
            test=get("rolling52_asof_histogram", "test", "raw", "ece22"), val=get("rolling52_asof_histogram", "val", "raw", "ece22"))}


# ================================================================== 0.3 S1 arithmetic
def stage03():
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts"])
    tx = tx[tx.txn_type.isin(["transfer_in", "transfer_out"])]
    tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
    net = tx.groupby(["part_id", "plant_id", "week"]).qty.sum().rename("net")
    pw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "safety_stock_qty"])
    pw["week"] = pd.to_datetime(pw.week_start)
    pw = pw[pw.week.dt.year == 2025].join(net, on=["part_id", "plant_id", "week"]).fillna({"net": 0.0})
    below = pw.qty_on_hand < pw.safety_stock_qty
    obs_short = float((pw.safety_stock_qty - pw.qty_on_hand)[below].mean())
    pre = pw.qty_on_hand - pw.net
    pre_below = pre < pw.safety_stock_qty
    pre_short = float((pw.safety_stock_qty - pre)[pre_below].mean())
    literal = obs_short + float(pw.net[below].mean())
    lo, hi = 298 / 2, 298 * 2
    return dict(observed_shortfall_when_short=obs_short, observed_below_ss=float(below.mean()),
                literal_sum_observed_plus_mean_net_transfer_in_short_weeks=literal,
                pre_rescue_shortfall_when_short=pre_short, pre_rescue_below_ss=float(pre_below.mean()),
                sim_policy_rop_shortfall=298.0, sim_placeholder_shortfall=564.0, window_2x_of_298=[lo, hi],
                gate_0_3_literal_within_2x=bool(lo <= literal <= hi),
                gate_0_3_pre_rescue_within_2x=bool(lo <= pre_short <= hi))


# ================================================================== 0.5 variance decomposition
def _backfit_r2(y, codes, iters=30):
    r = y - y.mean(); tot = float((r ** 2).sum())
    eff = [np.zeros(c.max() + 1) for c in codes]
    for _ in range(iters):
        for k, c in enumerate(codes):
            r = r + eff[k][c]
            s = np.bincount(c, weights=r, minlength=len(eff[k])); n = np.bincount(c, minlength=len(eff[k]))
            eff[k] = s / np.maximum(n, 1)
            r = r - eff[k][c]
    return 1.0 - float((r ** 2).sum()) / tot


def stage05(cl, n_perm=200, seed=0):
    y = cl.fill.to_numpy(float)
    f = {k: pd.factorize(cl[k])[0] for k in ("part_id", "supplier_id", "plant_id")}
    ps = pd.factorize(cl.part_id + "|" + cl.supplier_id)[0]
    sp = pd.factorize(cl.supplier_id + "|" + cl.plant_id)[0]
    pp = pd.factorize(cl.part_id + "|" + cl.plant_id)[0]
    psp = pd.factorize(cl.part_id + "|" + cl.supplier_id + "|" + cl.plant_id)[0]
    r = dict(n_lines=int(len(y)), fill_var=float(y.var()), share_zero=float((y == 0).mean()), share_one=float((y >= 1).mean()))
    R = lambda *c: _backfit_r2(y, list(c))
    r["r2"] = dict(part=R(f["part_id"]), supplier=R(f["supplier_id"]), plant=R(f["plant_id"]),
                   supplier_plus_plant=R(f["supplier_id"], f["plant_id"]),
                   part_plus_supplier=R(f["part_id"], f["supplier_id"]),
                   part_plus_supplier_plus_plant=R(f["part_id"], f["supplier_id"], f["plant_id"]),
                   cells_part_supplier=R(ps), cells_supplier_plant=R(sp), cells_part_plant=R(pp), cells_triple=R(psp))
    x = r["r2"]
    r["marginal"] = dict(plant_after_supplier=x["supplier_plus_plant"] - x["supplier"],
                         part_after_supplier=x["part_plus_supplier"] - x["supplier"],
                         supplier_after_part=x["part_plus_supplier"] - x["part"],
                         supplierXplant_interaction=x["cells_supplier_plant"] - x["supplier_plus_plant"],
                         partXsupplier_interaction=x["cells_part_supplier"] - x["part_plus_supplier"],
                         triple_after_part_supplier_cells=x["cells_triple"] - x["cells_part_supplier"])
    # noise floors: permute the added factor WITHIN the conditioning factor's groups, so only its own
    # signal is destroyed; cell-mean terms get a floor too, because cell means over-fit with cell count
    rng = np.random.default_rng(seed)

    def within_perm(lbl, grp):
        out = lbl.copy()
        order = np.argsort(grp, kind="stable"); g = grp[order]
        starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
        for a, b in zip(starts[:-1], starts[1:]):
            idx = order[a:b]; out[idx] = lbl[rng.permutation(idx)]
        return out
    nulls = dict(plant_after_supplier=[], supplierXplant_interaction=[], triple_after_part_supplier_cells=[])
    for _ in range(n_perm):
        pl = within_perm(f["plant_id"], f["supplier_id"])
        nulls["plant_after_supplier"].append(R(f["supplier_id"], pl) - x["supplier"])
        spn = pd.factorize(pd.Series(f["supplier_id"]).astype(str) + "|" + pd.Series(pl).astype(str))[0]
        nulls["supplierXplant_interaction"].append(R(spn) - R(f["supplier_id"], pl))
        pln = within_perm(f["plant_id"], ps)
        tn = pd.factorize(pd.Series(ps).astype(str) + "|" + pd.Series(pln).astype(str))[0]
        nulls["triple_after_part_supplier_cells"].append(R(tn) - x["cells_part_supplier"])
    r["noise_floor_p95"] = {k: float(np.percentile(v, 95)) for k, v in nulls.items()}
    r["noise_floor_mean"] = {k: float(np.mean(v)) for k, v in nulls.items()}
    r["above_floor"] = {k: bool(r["marginal"][k] > r["noise_floor_p95"][k]) for k in nulls}
    r["gate_0_5_plant_dead"] = not r["above_floor"]["plant_after_supplier"]
    return r


# ================================================================== 0.6 cell counts
def stage06(cl):
    lb = P5.labels("v8", "fill_rate")
    rows = lb[lb.snapshot_date == T0]
    assert len(rows), f"no fill scoring rows at {T0.date()}"
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "part_id", "supplier_id", "plant_id"])
    rows = rows.merge(ch, left_on="key", right_on="channel_id")
    keys = {"part_supplier_plant": ["part_id", "supplier_id", "plant_id"], "part_supplier": ["part_id", "supplier_id"],
            "supplier_plant": ["supplier_id", "plant_id"], "supplier": ["supplier_id"]}
    out = {}
    for name, k in keys.items():
        cnt = cl.groupby(k).size()
        c_rows = rows.set_index(k).index.map(cnt).to_numpy(float)
        c_rows = np.nan_to_num(c_rows, nan=0.0)
        out[name] = dict(cells_with_history=int(len(cnt)), per_cell_median=float(cnt.median()),
                         per_cell_p10=float(cnt.quantile(0.10)), per_cell_p25=float(cnt.quantile(0.25)),
                         scoring_rows=int(len(rows)),
                         frac_scoring_rows_with_at_least={n: float((c_rows >= n).mean()) for n in (1, 3, 5, 10, 20, 50)})
    out["gate_0_6_keep_raw_triple"] = out["part_supplier_plant"]["frac_scoring_rows_with_at_least"][10] >= 0.50
    return out


def main():
    st = C.require_clean()
    R = dict(stamp=st, t0=str(T0.date()), lookback_weeks=LOOKBACK_W)
    R["0.1"] = stage01(); print("0.1", json.dumps(R["0.1"], default=str), flush=True)
    cl = closed_lines(T0)
    R["closed_lines_at_t0"] = dict(n=int(len(cl)), closed_zero=int(cl.closed_zero.sum()),
                                   share_zero=float(cl.closed_zero.mean()))
    R["0.5"] = stage05(cl); print("0.5", json.dumps({k: R["0.5"][k] for k in ("marginal", "noise_floor_p95", "above_floor", "gate_0_5_plant_dead")}), flush=True)
    R["0.6"] = stage06(cl); print("0.6", json.dumps(R["0.6"]), flush=True)
    R["0.3"] = stage03(); print("0.3", json.dumps(R["0.3"]), flush=True)
    R["0.4"] = dict(scoreable=True, how="stored raw and recalibrated predictions for 5 fits on val and test "
                    "(ml/artifacts/phase7_preds/{,RECAL_}v8_fill_rate_b5flat22_s*_{val,test}.npz), scored through "
                    "phase5_metrics exactly as the head is. NOT servable (deviation 46); no loader is built.")
    rows = stage02()
    R["0.2"] = rows; R["0.2_pinned"] = pin_quoted(rows)
    print(pd.DataFrame(rows).to_string(max_rows=200), flush=True)
    print(json.dumps(R["0.2_pinned"], indent=1, default=str))
    print(C.dump(R, "phase13_stage0.json"))


if __name__ == "__main__":
    main()
