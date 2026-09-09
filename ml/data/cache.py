"""Phase 1.5 — materialise each world once; never re-parse CSVs in the training loop.

Two findings shape this cache and are reported in reports/phase-1.md:

  * `sourcing_channels` is time-invariant in both worlds (every row effective 2016-01-01,
    effective_to NULL, approval_status 'approved'), so the CORE GRAPH IS STATIC across all
    83 snapshots. The guide's per-snapshot effective-date filter is a no-op for it. We store
    one graph per world, not 83.
  * the weekly store is a complete contiguous panel -- 535 weeks for every one of the 16,072
    channels -- so it densifies exactly into [NCH, T, d] with no ragged padding.

Layout:  ml/artifacts/cache/{world}/panel.npy   [NCH, T, d] float32, memmapped
                                  /meta.json
                                  /graph.pt
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

# the per-timestep channel features; order is fixed and recorded in meta.json
# Measured on both worlds: revision_count, days_since_last_short, weeks_since_last_activity
# and weeks_since_last_receipt are CONSTANT ZERO -- generator placeholders that were never
# populated. A constant column carries no information (guide s11.1 says exclude them), so they
# are not in the panel. is_active_week is promoted to a value channel because, with the store
# forward-filling the rolling columns, it is where the idle/active signal actually lives.
PANEL_COLS = ["qty_ordered", "qty_received", "is_active_week", "fill_rate", "fill_rate_last4",
              "fill_rate_last13", "fill_rate_last52", "lead_time_actual_days",
              "lead_time_ratio", "otd_rate_last13", "ack_gap_ratio", "load_ratio",
              "reporting_lag_days", "active_weeks_in_52"]
# guide 2.2: nullable columns get a paired missing-indicator; never impute
NULLABLE = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52",
            "lead_time_actual_days", "lead_time_ratio", "otd_rate_last13",
            "ack_gap_ratio", "load_ratio", "reporting_lag_days"]


def build_panel(csv_dir: str, out_dir: str, verbose: bool = True) -> dict:
    from loader import read_df, table_path
    os.makedirs(out_dir, exist_ok=True)
    ch = read_df(csv_dir, "sourcing_channels", usecols=["channel_id"])
    cidx = {v: i for i, v in enumerate(ch.channel_id)}
    NCH = len(cidx)

    cpw = read_df(csv_dir, "channel_performance_weekly",
                  usecols=["channel_id", "week_start"] + PANEL_COLS)
    weeks = pd.to_datetime(cpw.week_start)
    w0 = weeks.min()
    ti = ((weeks - w0).dt.days // 7).to_numpy()
    T = int(ti.max()) + 1
    ci = cpw.channel_id.map(cidx).to_numpy()

    d = len(PANEL_COLS)
    panel = np.zeros((NCH, T, d), np.float32)
    miss = np.zeros((NCH, T, len(NULLABLE)), np.float32)   # 1 = observed
    for j, c in enumerate(PANEL_COLS):
        if c == "is_active_week":
            v = cpw[c].astype(str).str.lower().isin(["true", "1"]).astype(float)
        else:
            v = pd.to_numeric(cpw[c], errors="coerce")
        panel[ci, ti, j] = v.fillna(0.0).to_numpy(np.float32)
    for j, c in enumerate(NULLABLE):
        miss[ci, ti, j] = pd.to_numeric(cpw[c], errors="coerce").notna().to_numpy(np.float32)
    active = np.zeros((NCH, T), np.float32)
    active[ci, ti] = cpw.is_active_week.astype(str).str.lower().isin(["true", "1"]).to_numpy(np.float32)
    # a real row exists for every (channel, week): the panel is complete, so padding is a no-op
    present = np.zeros((NCH, T), np.float32); present[ci, ti] = 1.0

    np.save(os.path.join(out_dir, "panel.npy"), panel)
    np.save(os.path.join(out_dir, "miss.npy"), miss)
    np.save(os.path.join(out_dir, "active.npy"), active)
    meta = {"n_channels": NCH, "T": T, "week0": str(w0.date()), "cols": PANEL_COLS,
            "nullable": NULLABLE, "d": d,
            "panel_complete": bool(present.min() == 1.0),
            "rows": int(len(cpw))}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    if verbose:
        print(f"    panel [{NCH}, {T}, {d}] = {panel.nbytes/1e6:.0f} MB, "
              f"complete={meta['panel_complete']}, rows={len(cpw):,}")
    return meta


def load_panel(out_dir: str):
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    panel = np.load(os.path.join(out_dir, "panel.npy"), mmap_mode="r")
    miss = np.load(os.path.join(out_dir, "miss.npy"), mmap_mode="r")
    active = np.load(os.path.join(out_dir, "active.npy"), mmap_mode="r")
    return panel, miss, active, meta
