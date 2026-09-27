"""Phase 12 Wave A5 -- the measurements the single order-policy specification rests on. No training.

Three questions, each of which changes the rule A5.2 would otherwise write down:

  1. Does part_plant.reorder_point_qty ALREADY contain lead-time demand? If it does, the brief's trigger
     `stock - E[consumption over lead] <= reorder_point_qty` counts lead-time demand twice.
     Measured as the ratio (ROP - SS) / (weekly requirement x planning lead weeks).
  2. How much of the 13-week consumption does the pipeline ALREADY OPEN at t0 cover? The Phase 11C
     simulation ignores it entirely (one lump order), so this bounds how much of the 85.4% replacement
     shortfall is the missing pipeline rather than the order rule.
  3. What order quantity does the recorded history show, against MOQ and lot size? The generator orders
     UP-TO a level part_plant does not carry; MOQ/lot alone would under-order if typical lines are larger.

Measured on v8 seed 1001 at the nine 2025 test snapshots, as-of: every table filtered recorded/as-of <= t0.
inventory_position_weekly is read (qty_on_hand, open_po_qty, qty_in_transit) -- the store B1 reconciled at
100.000000%, and the same use the simulation already makes of it. Never as a model feature.
"""
from __future__ import annotations
import phase12_common as C
import numpy as np, pandas as pd

D = C.REPO + "/db/gen_v8/seed_1001"


def main():
    st = C.require_clean()
    pp = pd.read_csv(f"{D}/part_plant.csv")
    snaps = pd.read_csv(f"{D}/snapshots.csv", usecols=["as_of_ts"])
    t0s = sorted(d for d in pd.to_datetime(snaps.as_of_ts) if d.year == 2025)
    pdw = pd.read_csv(f"{D}/part_demand_weekly.csv",
                      usecols=["part_id", "plant_id", "week_start", "as_of_date", "gross_requirement_p50"])
    pdw["as_of_date"] = pd.to_datetime(pdw.as_of_date); pdw["week_start"] = pd.to_datetime(pdw.week_start)
    ipw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                      usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "open_po_qty",
                               "qty_in_transit", "safety_stock_qty", "recorded_ts"])
    ipw["recorded_ts"] = pd.to_datetime(ipw.recorded_ts)
    pol = pd.read_csv(f"{D}/po_lines.csv", usecols=["part_id", "channel_id", "qty_ordered", "created_ts", "recorded_ts"])
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "plant_id"])
    pol = pol.merge(ch, on="channel_id")
    pol["recorded_ts"] = pd.to_datetime(pol.recorded_ts)

    rows = []
    for t0 in t0s:
        d = pdw[pdw.as_of_date <= t0]
        assert len(d) and (d.as_of_date <= t0).all()
        d = d[d.as_of_date == d.as_of_date.max()]
        fwd = d[d.week_start > t0].sort_values("week_start").groupby(["part_id", "plant_id"]).head(13)
        req13 = fwd.groupby(["part_id", "plant_id"]).gross_requirement_p50.sum()
        wk = fwd.groupby(["part_id", "plant_id"]).gross_requirement_p50.mean()
        iv = ipw[ipw.recorded_ts <= t0]
        assert len(iv) and (iv.recorded_ts <= t0).all()
        iv = iv.sort_values("week_start").groupby(["part_id", "plant_id"]).tail(1).set_index(["part_id", "plant_id"])
        x = pp.set_index(["part_id", "plant_id"]).join(pd.DataFrame(dict(req13=req13, wk=wk))).join(
            iv[["qty_on_hand", "open_po_qty", "qty_in_transit"]])
        x = x[(x.req13 > 0) & x.qty_on_hand.notna()]
        lead_w = x.planning_lead_time_days / 7.0
        ltd = x.wk * lead_w
        hist = pol[(pol.recorded_ts <= t0) & (pol.recorded_ts > t0 - pd.Timedelta(weeks=26))]
        hq = hist.groupby(["part_id", "plant_id"]).qty_ordered.median()
        x = x.join(hq.rename("hist_qty"))
        rows.append(dict(
            t0=str(t0.date()), part_plants=int(len(x)),
            rop_minus_ss_over_ltd_median=float(((x.reorder_point_qty - x.safety_stock_qty) / ltd.replace(0, np.nan)).median()),
            rop_minus_ss_over_ltd_iqr=[float(q) for q in ((x.reorder_point_qty - x.safety_stock_qty) / ltd.replace(0, np.nan)).quantile([.25, .75])],
            rop_over_ss_median=float((x.reorder_point_qty / x.safety_stock_qty.replace(0, np.nan)).median()),
            pipeline_cover_of_req13_median=float(((x.open_po_qty) / x.req13).median()),
            pipeline_cover_of_req13_total=float(x.open_po_qty.sum() / x.req13.sum()),
            onhand_plus_pipeline_over_req13_total=float((x.qty_on_hand + x.open_po_qty).sum() / x.req13.sum()),
            onhand_over_req13_total=float(x.qty_on_hand.sum() / x.req13.sum()),
            in_transit_share_of_open=float(x.qty_in_transit.sum() / max(1, x.open_po_qty.sum())),
            ip_below_rop=float(((x.qty_on_hand + x.open_po_qty) < x.reorder_point_qty).mean()),
            hist_line_qty_over_moq_median=float((x.hist_qty / x.min_order_qty).median()),
            hist_line_qty_over_lot_median=float((x.hist_qty / x.lot_size).median()),
            moq_over_weekly_req_median=float((x.min_order_qty / x.wk).median())))
    df = pd.DataFrame(rows)
    out = dict(stamp=st, per_snapshot=rows,
               summary={c: float(df[c].median()) for c in df.columns if df[c].dtype != object and c != "part_plants"})
    print(pd.DataFrame(rows).drop(columns=["rop_minus_ss_over_ltd_iqr"]).round(3).to_string())
    print(C.dump(out, "phase12_a5_facts.json"))


if __name__ == "__main__":
    main()
