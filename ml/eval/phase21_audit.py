"""Phase 21 Stage 1 -- data audit and group sparsity (read-only). No model, no fit.

  a. lane / shipping and expected-delivery columns: which exist, whether they vary, whether they are as-of; one lane per
     channel; via_checkpoint; distinct lane clusters
  b. per level L5 / L4 / L3 / L2: groups, and the distribution of KM-eligible deliveries per group as-of the train end,
     the validation start and the test start; share of val / test snapshot rows whose L5 / L4 group has < 5 prior
     receipted deliveries (read from the built store, ml/artifacts/phase21/grpstats_{world}_snap.npz)
  c. original_promise_date coverage and its relation to the contract; late-vs-contract base rate by creation month and
     by lane cluster (lines created in the train period with a receipt)
  d. the tests' record (ml/artifacts/phase21/selftest_{world}.json)

  python ml/eval/phase21_audit.py [--world v8]      -> ml/artifacts/phase21/stage1_audit_{world}.json
"""
from __future__ import annotations
import os, sys, json, argparse
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import grpstats as GS


def dist(x):
    x = np.asarray(x, float)
    return dict(groups=int(len(x)), min=float(x.min()), p10=float(np.percentile(x, 10)), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), max=float(x.max())) if len(x) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="v8"); a = ap.parse_args()
    PP.register()
    st = C.require_clean()
    D = config.WORLDS[a.world]
    out = dict(stamp=st, world=a.world, hashes=PP.world_hashes(a.world))
    src = GS.Source(a.world)

    # ---------------------------------------------------------------- a. lane / shipping columns
    ch = pd.read_csv(f"{D}/sourcing_channels.csv")
    ln = pd.read_csv(f"{D}/logistics_lanes.csv")
    ss = pd.read_csv(f"{D}/supplier_sites.csv")
    su = pd.read_csv(f"{D}/suppliers.csv")
    po = pd.read_csv(f"{D}/purchase_orders.csv", usecols=["po_id", "incoterm", "po_type", "currency", "recorded_ts"])
    key = ch[["site_id", "plant_id", "transport_mode"]].astype(str).agg("|".join, axis=1)
    lane = {}
    lane["lane_rows_vs_channels"] = [int(len(ln)), int(len(ch))]
    lane["lane_row_i_is_channel_i"] = bool((ln.origin_site_id.to_numpy() == ch.site_id.to_numpy()).all()
                                           and (ln.dest_plant_id.to_numpy() == ch.plant_id.to_numpy()).all()
                                           and (ln.transport_mode.to_numpy() == ch.transport_mode.to_numpy()).all())
    lane["distance_equal_to_channel"] = float((np.abs(ln.distance_km.to_numpy(float) - ch.transport_distance_km.to_numpy(float)) < 1e-6).mean())
    lane["channels_sharing_a_site_plant_mode_key"] = int((key.duplicated(keep=False)).sum())
    lane["distinct_lane_ids"] = int(ln.lane_id.nunique())
    lane["via_checkpoint_nonempty_share"] = float(ln.via_checkpoint.notna().mean())
    cols = {
        "sourcing_channels.transport_mode": ch.transport_mode, "sourcing_channels.transport_distance_km": ch.transport_distance_km,
        "logistics_lanes.standard_transit_days": ln.standard_transit_days, "logistics_lanes.carrier_id": ln.carrier_id,
        "logistics_lanes.via_checkpoint": ln.via_checkpoint, "supplier_sites.country": ss.country, "supplier_sites.state": ss.state,
        "suppliers.country": su.country, "purchase_orders.incoterm": po.incoterm, "purchase_orders.po_type": po.po_type,
        "purchase_orders.currency": po.currency}
    lane["columns"] = {k: dict(distinct=int(v.nunique(dropna=True)), null_share=float(v.isna().mean()),
                               top=v.value_counts(dropna=True).head(5).to_dict()) for k, v in cols.items()}
    lane["as_of_note"] = ("channel / lane / site / supplier columns are static masters (no recorded_ts); incoterm, po_type and "
                          "currency are PO-header columns, known only once the PO is recorded, so a snapshot row (whose line "
                          "is not yet raised) cannot carry them")
    lane["lane_clusters"] = src.lc_names
    lane["transit_tercile_cuts_days"] = src.transit_cuts
    lc = pd.Series(src.ch_lc).value_counts().sort_index()
    lane["channels_per_lane_cluster"] = {src.lc_names[i]: int(n) for i, n in lc.items()}
    lane["distinct_lane_clusters_used"] = int(len(lc))
    sup_lc = pd.DataFrame({"s": src.ch_sup, "lc": src.ch_lc}).groupby("s").lc.nunique()
    lane["lane_clusters_per_supplier"] = dist(sup_lc.to_numpy())
    out["a_lane"] = lane

    # ---------------------------------------------------------------- a/c. expected delivery columns
    exp = {}
    pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "original_promise_date", "current_promise_date", "created_ts"])
    exp["original_promise_nonnull_share"] = float(pl.original_promise_date.notna().mean())
    exp["current_promise_nonnull_share"] = float(pl.current_promise_date.notna().mean())
    d = (pd.to_datetime(pl.original_promise_date) - pd.to_datetime(pl.created_ts)).dt.days.to_numpy(float)
    exp["promise_equals_created_plus_contract_share"] = float((np.abs(d - src.contract) < 0.5).mean())
    sch = pd.read_csv(f"{D}/po_line_schedules.csv", usecols=["po_line_id"])
    asn = pd.read_csv(f"{D}/asn.csv", usecols=["po_line_id", "expected_arrival_date", "recorded_ts"])
    exp["lines_with_schedule_share"] = float(pl.po_line_id.isin(sch.po_line_id).mean())
    exp["lines_with_asn_share"] = float(pl.po_line_id.isin(asn.po_line_id).mean())
    exp["schedule_rows_note"] = "po_line_schedules rows are written at line creation (released_ts = created); ASN at dispatch"
    # snapshot rows: is the row's own line raised at t0 (so that its promise / schedule / ASN could exist)?
    a_rows, same = GS.snapshot_rows(a.world)
    li = src.lidx.reindex(a_rows.entity_id).to_numpy().astype(np.int64)
    t0 = a_rows.snapshot_date.to_numpy("datetime64[ns]")
    exp["snapshot_rows"] = int(len(a_rows))
    exp["arrival_and_fill_rows_identical"] = bool(same)
    exp["snapshot_rows_line_created_le_t0_share"] = float((src.created[li] <= t0).mean())
    exp["snapshot_rows_line_recorded_le_t0_share"] = float((src.line_rec[li] <= t0).mean())
    wk = (src.created[li] - t0) / np.timedelta64(7, "D")
    exp["snapshot_rows_weeks_from_t0_to_creation"] = dist(wk)
    exp["fallback_use"] = ("at snapshots no row has a promise (0% raised at t0), so the expected-delivery fallback "
                           "(order date + shrunk KM median lead) is used for 100% of snapshot rows; at placement the promise "
                           "exists for every line and is used")
    out["c_expected_delivery"] = exp

    # ---------------------------------------------------------------- c. late-vs-contract base rates (train-period lines)
    cr = pd.DatetimeIndex(src.created)
    trn = (cr >= pd.Timestamp(config.FIT_WINDOW[0])) & (cr <= pd.Timestamp(config.SPLIT["train_end"]))
    rec = ~np.isnat(src.g_event)
    late = (src.g_event > src.promise)
    m = trn & rec
    df = pd.DataFrame({"month": src.month[m] + 1, "lc": np.array(src.lc_names)[src.ch_lc[src.chan[m]]], "late": late[m]})
    out["c_base_rates"] = dict(lines=int(m.sum()), late_rate=float(df.late.mean()),
                               by_creation_month={int(k): float(v) for k, v in df.groupby("month").late.mean().items()},
                               by_lane_cluster={k: dict(rate=float(v), n=int(n)) for (k, v), n in
                                                zip(df.groupby("lc").late.mean().items(), df.groupby("lc").size())},
                               unreceipted_train_lines=int((trn & ~rec).sum()))
    lead = ((src.g_event - src.created) / GS.DAY)[m]
    out["c_base_rates"]["median_lead_days_by_creation_month"] = {int(k): float(v) for k, v in
                                                                 pd.Series(lead).groupby(src.month[m] + 1).median().items()}

    # ---------------------------------------------------------------- b. group sparsity at three as-of instants
    B = GS.Builder(src)
    spars = {}
    for name, tau in (("train_end", "2023-12-31"), ("val_start", "2024-01-01"), ("test_start", "2025-01-01")):
        A, meta, _ = B.arrival(np.datetime64(pd.Timestamp(tau), "ns"))
        iN, iA = GS.ASTATS.index("n"), GS.ASTATS.index("nall")
        spars[name] = {L: dict(groups_possible=int(B.G[L]), groups_nonempty=int((A[L][:, iA] > 0).sum()),
                               receipted_per_nonempty_group=dist(A[L][:, iN][A[L][:, iA] > 0]),
                               lines_per_nonempty_group=dist(A[L][:, iA][A[L][:, iA] > 0]))
                       for L in ("L5", "L4", "L3", "L2")}
        spars[name]["meta"] = meta
    out["b_sparsity"] = spars
    Zp = os.path.join(GS.OUT, f"grpstats_{a.world}_snap.npz")
    if os.path.exists(Zp):
        Z = dict(np.load(Zp))
        snap = pd.to_datetime(Z["snapshot"])
        fold = np.where(snap > pd.Timestamp(config.SPLIT["val_end"]), "test", np.where(snap > pd.Timestamp(config.SPLIT["train_end"]), "val", "train"))
        r = np.arange(len(snap)); mk = Z["month"].astype(np.int64)
        iN = GS.ASTATS.index("n")
        n5 = np.nan_to_num(Z["A_L5c"][r, mk, iN]); n4 = np.nan_to_num(Z["A_L4"][:, iN])
        out["b_rows_below_5"] = {f: dict(L5=float((n5[fold == f] < 5).mean()), L4=float((n4[fold == f] < 5).mean()),
                                          L4_zero=float((n4[fold == f] == 0).mean()), rows=int((fold == f).sum()))
                                 for f in ("train", "val", "test")}
    # ---------------------------------------------------------------- d. the tests
    tp = os.path.join(GS.OUT, f"selftest_{a.world}.json")
    out["d_tests"] = json.load(open(tp)) if os.path.exists(tp) else "NOT RUN"
    C.dump(out, f"phase21/stage1_audit_{a.world}.json")
    print(json.dumps({k: v for k, v in out.items() if k not in ("b_sparsity", "d_tests")}, indent=1, default=str)[:6000])
    print(json.dumps(out["b_sparsity"], indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
