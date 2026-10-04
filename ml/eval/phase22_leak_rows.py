"""Phase 22 Stage 1a -- the leak shown on lines, and the snapshot rows it reaches. AUDIT ONLY (reads _sim.npz through
phase22_leakscan.Sources; never a model input).

  sample     for 8 PO lines: the stored lead_time_actual_days in the channel's panel row of the week the line was ORDERED,
             the line's own eventual lead (first receipt - created), and the week its receipt became visible
  affected   for every fit-window snapshot t0: rebuild the weekly store with every source row visible after t0's week
             poisoned (H_week) and mark the channels whose row-t0 value moves, per leaking column; the share of the
             arrival / fill / capacity LABEL rows (by fold) whose channel is marked = the snapshot rows the leak reaches

  python ml/eval/phase22_leak_rows.py      -> ml/artifacts/phase22/leak_rows_v8.json
"""
from __future__ import annotations
import os, sys, json
import phase12_common as C
import numpy as np, pandas as pd
import config, folds
import phase21_paths as PP
import phase22_leakscan as LS
from cache import load_panel

LEAK9 = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52", "lead_time_actual_days", "lead_time_ratio",
         "otd_rate_last13", "ack_gap_ratio", "load_ratio"]


def main():
    PP.register()
    st = C.require_clean()
    S = LS.Sources("v8")
    panel, miss, active, meta = load_panel(os.path.join(config.CACHE, "v8"))
    j = meta["cols"].index("lead_time_actual_days")
    # ---- sample: lines that were the LAST line ordered in their channel-week (the one leadw keeps) and received weeks later
    IN = (S.vw_ord >= 0) & (S.vw_ord < S.T)
    key = S.pch * S.T + S.vw_ord
    last = pd.Series(np.arange(len(key)))[IN].groupby(key[IN]).max().to_numpy()
    cand = last[(S.vw_rec[last] < S.T) & (S.vw_rec[last] - S.vw_ord[last] >= 4) & (S.vw_ord[last] > 200)]
    rng = np.random.default_rng(7)
    pick = rng.choice(cand, 8, replace=False)
    W0 = S.W0
    sample = []
    for i in pick:
        c, t = int(S.pch[i]), int(S.vw_ord[i])
        sample.append(dict(po_line=f"POL{i:09d}", channel_index=c, order_visible_week=str((W0 + pd.Timedelta(days=7 * t)).date()),
                           stored_lead_time_actual_days_in_that_week=float(panel[c, t, j]),
                           line_eventual_lead_days=round(float(S.pl[i]), 2),
                           receipt_visible_week=str((W0 + pd.Timedelta(days=7 * int(S.vw_rec[i]))).date()),
                           weeks_before_receipt_visible=int(S.vw_rec[i] - t)))
    eq = sum(abs(s["stored_lead_time_actual_days_in_that_week"] - s["line_eventual_lead_days"]) < 0.006 for s in sample)
    # ---- affected rows
    R = LS.rebuild(S)
    D = config.WORLDS["v8"]
    lb = pd.read_csv(D + "/training_labels.csv", usecols=["snapshot_date", "entity_id", "task"])
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    lb = lb[(lb.snapshot_date >= config.FIT_WINDOW[0]) & (lb.snapshot_date <= config.FIT_WINDOW[1])]
    pl = pd.read_csv(D + "/po_lines.csv", usecols=["po_line_id", "channel_id"]).set_index("po_line_id").channel_id
    ch = pd.read_csv(D + "/sourcing_channels.csv", usecols=["channel_id"]).channel_id
    cidx = pd.Series(np.arange(len(ch)), index=ch)
    lb["chan"] = np.where(lb.task == "capacity_strain", cidx.reindex(lb.entity_id).to_numpy(), cidx.reindex(pl.reindex(lb.entity_id).to_numpy()).to_numpy())
    rng2 = np.random.default_rng(22)
    per_snap = {}
    marks = {}
    for t0 in sorted(lb.snapshot_date.unique()):
        t = int((pd.Timestamp(t0) - W0).days // 7)
        Rp = LS.rebuild(S, **LS.poison(S, t, False, rng2))
        ch_any = np.zeros(S.NCH, bool); per_col = {}
        for c in LEAK9:
            a_, b_ = np.nan_to_num(R[c][:, t], nan=-999.0), np.nan_to_num(Rp[c][:, t], nan=-999.0)
            mv = np.abs(a_ - b_) > 1e-9; ch_any |= mv; per_col[c] = float(mv.mean())
        marks[pd.Timestamp(t0)] = ch_any
        per_snap[str(pd.Timestamp(t0).date())] = dict(share_channels_any=float(ch_any.mean()), **per_col)
    lb["affected"] = [bool(marks[pd.Timestamp(d)][int(c)]) if np.isfinite(c) else False for d, c in zip(lb.snapshot_date, lb.chan.astype(float))]
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    fold = np.where(np.asarray(te), "test", np.where(np.asarray(va), "val", "train"))
    aff = {task: {f: float(lb.affected[(lb.task == task).to_numpy() & (fold == f)].mean()) for f in ("train", "val", "test")}
           for task in ("arrival_week", "fill_rate", "capacity_strain")}
    out = dict(stamp=st, sample=sample, sample_equal_count=f"{eq} of {len(sample)}", affected_label_rows_share=aff,
               per_snapshot=per_snap, note="AUDIT ONLY (_sim.npz); H_week poison per snapshot")
    C.dump(out, "phase22/leak_rows_v8.json")
    print(json.dumps(dict(sample=sample, eq=out["sample_equal_count"], affected=aff), indent=1))


if __name__ == "__main__":
    main()
