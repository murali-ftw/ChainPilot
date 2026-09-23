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

# The per-timestep channel features. Order is fixed by CANDIDATE_COLS and recorded in meta.json.
#
# WHICH COLUMNS SURVIVE IS A PROPERTY OF THE WORLD BEING LOADED, AND IS NEVER BORROWED.
# In v6 and v7 four of these are CONSTANT ZERO -- revision_count, days_since_last_short,
# weeks_since_last_activity, weeks_since_last_receipt -- generator placeholders that were never
# populated, so the panel is 14 wide (guide deviation 4). In v8 `revision_count` IS populated
# (3.33% non-zero, max 5), so v8's panel is 15 wide (reports/v8-clearance.md deviation 53).
# Hardcoding either width silently mis-builds the other world, so build_panel MEASURES the
# constant-zero set on the CSVs it is actually reading and asserts the resulting width against
# config.EXPECTED_PANEL_D.
#
# is_active_week is promoted to a value channel because, with the store forward-filling the
# rolling columns, it is where the idle/active signal actually lives.
CANDIDATE_COLS = ["qty_ordered", "qty_received", "is_active_week", "fill_rate", "fill_rate_last4",
                  "fill_rate_last13", "fill_rate_last52", "lead_time_actual_days",
                  "lead_time_ratio", "otd_rate_last13", "ack_gap_ratio", "load_ratio",
                  "reporting_lag_days", "active_weeks_in_52",
                  # constant zero in v6/v7, live in v8 -> appended last so v6/v7 panels stay
                  # byte-identical to every number measured in Phases 2-11
                  "revision_count", "days_since_last_short",
                  "weeks_since_last_activity", "weeks_since_last_receipt"]
# The v6/v7 set, retained under its original name so nothing importing it breaks.
PANEL_COLS = CANDIDATE_COLS[:14]
# guide 2.2: nullable columns get a paired missing-indicator; never impute
NULLABLE = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52",
            "lead_time_actual_days", "lead_time_ratio", "otd_rate_last13",
            "ack_gap_ratio", "load_ratio", "reporting_lag_days"]


def live_panel_cols(cpw) -> tuple[list, list]:
    """The candidate columns that are NOT constant-zero in this world, in canonical order.

    Returns (live, dropped). A column is dropped iff every row is zero or null -- exactly the
    condition guide deviation 4 names. This is measured, never assumed.
    """
    live, dropped = [], []
    for c in CANDIDATE_COLS:
        if c not in cpw.columns:
            dropped.append((c, "absent")); continue
        if c == "is_active_week":
            live.append(c); continue
        v = pd.to_numeric(cpw[c], errors="coerce").fillna(0.0)
        (dropped.append((c, "constant zero")) if bool((v == 0).all()) else live.append(c))
    return live, dropped


def build_panel(csv_dir: str, out_dir: str, verbose: bool = True, world: str = None) -> dict:
    from loader import read_df, table_path
    os.makedirs(out_dir, exist_ok=True)
    ch = read_df(csv_dir, "sourcing_channels", usecols=["channel_id"])
    cidx = {v: i for i, v in enumerate(ch.channel_id)}
    NCH = len(cidx)

    have = read_df(csv_dir, "channel_performance_weekly", nrows=0).columns
    want = [c for c in CANDIDATE_COLS if c in have]
    cpw = read_df(csv_dir, "channel_performance_weekly",
                  usecols=["channel_id", "week_start"] + want)
    PANEL_COLS, DROPPED = live_panel_cols(cpw)
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
            "nullable": NULLABLE, "d": d, "d_in": d + len(NULLABLE),
            "world": world, "dropped_constant_zero": [list(x) for x in DROPPED],
            "panel_complete": bool(present.min() == 1.0),
            "rows": int(len(cpw))}
    # PANEL WIDTH ASSERTION. A world's width is its own; borrowing another world's silently
    # mis-builds the panel, and nothing downstream would notice because d_in flows from meta.
    # This fires if the measured width does not match the width recorded for this world.
    if world is not None:
        from config import EXPECTED_PANEL_D
        exp = EXPECTED_PANEL_D.get(world)
        assert exp is not None, (
            f"panel width for world {world!r} is not declared in config.EXPECTED_PANEL_D; "
            f"measured {d} value channels ({PANEL_COLS}). Declare it rather than defaulting.")
        assert d == exp, (
            f"PANEL WIDTH MISMATCH for world {world!r}: measured {d} value channels, "
            f"config.EXPECTED_PANEL_D says {exp}. Live: {PANEL_COLS}. "
            f"Dropped as constant zero: {[c for c, _ in DROPPED]}. "
            f"Do NOT borrow another world's width -- fix the expectation or the world.")
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    if verbose:
        print(f"    panel [{NCH}, {T}, {d}] (+{len(NULLABLE)} indicators = d_in {d+len(NULLABLE)}) "
              f"= {panel.nbytes/1e6:.0f} MB, complete={meta['panel_complete']}, rows={len(cpw):,}")
        print(f"    live: {PANEL_COLS}")
        print(f"    dropped as constant zero: {[c for c, _ in DROPPED]}")
    return meta


def load_panel(out_dir: str):
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    panel = np.load(os.path.join(out_dir, "panel.npy"), mmap_mode="r")
    miss = np.load(os.path.join(out_dir, "miss.npy"), mmap_mode="r")
    active = np.load(os.path.join(out_dir, "active.npy"), mmap_mode="r")
    return panel, miss, active, meta
