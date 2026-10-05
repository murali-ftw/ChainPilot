"""Phase 23AC T2 -- the ONE allow-listed reader of the production plan (part_demand_weekly). DORMANT.

Status: DORMANT. Imported only by ml/serve/fill_consolidated.py; nothing in ml/configs/shipped.json references this module
or that service. It exists under a PROPOSED exception to constraint 4 that is NOT ACTIVE until the owner approves it in
writing (docs/decisions/proposed_exception_part_demand_weekly.md).

The plan table carries no recorded_ts. The generator draws it as as_of_date + 0..2 days, so the recorded time used here is
the conservative upper bound as_of_date + fwd_load.PDW_MAX_LAG (2 days; deviation 162).

  load_sources(world)     the same dict as fwd_load.load_sources(world) (ch, pdw, pol, grn, ch_pp, ch_sup, n_pp, n_sup,
                          cidx); the loading logic is copied, not called, so this file is the reader
  assert_plan_bound(rows, t0)
                          raises fwd_load.AsOfViolation if ANY row has as_of_date + 2 days > t0. Computed from as_of_date
                          itself, so a tampered recorded_ts cannot pass it
  plan_rows_asof(S, t0)   exactly the plan rows fwd_load.snapshot_features would use at t0 (recorded_ts <= t0 and
                          week_start in (t0, t0 + 13 w]), with the bound asserted on every one of them
  snapshot_blocks(S, t0)  fwd_load.snapshot_features (pure computation) on a copy of S whose plan holds ONLY the asserted rows,
                          plus the season vector (NaN-aware network mean over channels, ml/data/fwd_season's definition)

The reader's own assertion runs on every plan row it passes on; fwd_load's own assertion then runs again inside
snapshot_features. inventory_position_weekly is never read.

    python ml/serve/_plan_reader.py falsify      # the bound fires on a constructed future version
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import config
import fwd_load as FL

PLAN_TABLE = "part_demand_weekly"
PDW_MAX_LAG = FL.PDW_MAX_LAG
HORIZON = pd.Timedelta(weeks=13)


def load_sources(world):
    """Copy of fwd_load.load_sources: same columns, same derived recorded_ts (as_of_date + 2 d), same index maps."""
    D = config.WORLDS[world]
    ch = pd.read_csv(D + "/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "part_id", "plant_id"])
    pdw = pd.read_csv(D + "/" + PLAN_TABLE + ".csv",
                      usecols=["part_id", "plant_id", "week_start", "as_of_date", "gross_requirement_p50"])
    pdw["recorded_ts"] = pd.to_datetime(pdw.as_of_date) + PDW_MAX_LAG     # derived upper bound (deviation 162)
    pol = pd.read_csv(D + "/po_lines.csv", usecols=["po_line_id", "channel_id", "qty_ordered", "created_ts", "recorded_ts"])
    grn = pd.read_csv(D + "/grn_lines.csv", usecols=["po_line_id", "qty_received", "is_final_receipt", "event_ts", "recorded_ts"])
    for df, cols in ((pdw, ("week_start", "as_of_date", "recorded_ts")), (pol, ("created_ts", "recorded_ts")),
                     (grn, ("event_ts", "recorded_ts"))):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    cidx = {c: i for i, c in enumerate(ch.channel_id)}
    pp_key = (ch.part_id + "|" + ch.plant_id).to_numpy()
    pp_u = {k: i for i, k in enumerate(sorted(set(pp_key)))}
    ch_pp = np.array([pp_u[k] for k in pp_key])
    sup_u = {s: i for i, s in enumerate(sorted(ch.supplier_id.unique()))}
    ch_sup = ch.supplier_id.map(sup_u).to_numpy()
    pdw["pp"] = (pdw.part_id + "|" + pdw.plant_id).map(pp_u)
    pdw = pdw[pdw.pp.notna()].copy(); pdw["pp"] = pdw.pp.astype(np.int64)
    pol["ci"] = pol.channel_id.map(cidx).astype(np.int64)
    grn["ci"] = grn.po_line_id.map(pol.set_index("po_line_id").ci).astype(np.int64)
    grn["is_final_receipt"] = grn.is_final_receipt.astype(str).str.lower().isin(["true", "1"])
    return dict(ch=ch, pdw=pdw, pol=pol, grn=grn, ch_pp=ch_pp, ch_sup=ch_sup, n_pp=len(pp_u), n_sup=len(sup_u),
                cidx=cidx)


def assert_plan_bound(rows, t0):
    """Every plan row passed on must satisfy as_of_date + 2 days <= t0. Returns the number of rows asserted."""
    t0 = pd.Timestamp(t0)
    a = pd.to_datetime(pd.Series(rows["as_of_date"]))
    if a.isna().any():
        raise FL.AsOfViolation(f"plan: {int(a.isna().sum())} rows with no as_of_date reached a feature at {t0.date()}")
    bound = a + PDW_MAX_LAG
    bad = bound > t0
    if bad.any():
        raise FL.AsOfViolation(f"plan: {int(bad.sum())} rows with as_of_date + {PDW_MAX_LAG.days} d > t0 = {t0.date()} "
                               f"(latest as_of_date {a.max().date()}) reached a feature")
    return int(len(a))


def plan_rows_asof(S, t0):
    """The plan rows snapshot_features would select at t0 (its own filter, copied), with the bound asserted on each."""
    t0 = pd.Timestamp(t0)
    pdw = S["pdw"]
    m = (pdw.recorded_ts.values <= t0.to_datetime64()) & (pdw.week_start.values > t0.to_datetime64()) \
        & (pdw.week_start.values <= (t0 + HORIZON).to_datetime64())
    used = pdw.loc[m]
    n = assert_plan_bound(used, t0)
    return used, n


def snapshot_blocks(S, t0):
    """-> dict(fwdload [n_ch, 15] float32, season [15] float32, n_plan_rows_asserted, n_rows_asserted_fwd_load)."""
    t0 = pd.Timestamp(t0)
    used, n_plan = plan_rows_asof(S, t0)
    S2 = dict(S); S2["pdw"] = used                       # only asserted rows reach the pure computation
    L, n_fl = FL.snapshot_features(S2, t0)
    season = np.nanmean(L, axis=0).astype(np.float32)    # ml/data/fwd_season's definition (as ml/serve/features.py)
    return dict(fwdload=L, season=season, n_plan_rows_asserted=n_plan, n_rows_asserted_fwd_load=int(n_fl))


def falsify(world="v8"):
    """The bound must be able to fire: a version dated 1 day before t0 (as_of + 2 d > t0) whose recorded_ts is forged to
    t0 - 7 d passes snapshot_features' recorded_ts filter; the reader's as_of_date assertion must stop it."""
    import phase21_paths as PP
    PP.register()
    S = load_sources(world)
    t0 = pd.Timestamp(FL.snapshot_dates(world)[40])
    clean = snapshot_blocks(S, t0)
    bad = S["pdw"].iloc[[0]].copy()
    bad["week_start"] = t0 + pd.Timedelta(weeks=2); bad["as_of_date"] = t0 - pd.Timedelta(days=1)
    bad["recorded_ts"] = t0 - pd.Timedelta(days=7)
    S2 = dict(S); S2["pdw"] = pd.concat([S["pdw"], bad], ignore_index=True)
    try:
        snapshot_blocks(S2, t0); res = "DID NOT FIRE"
    except FL.AsOfViolation as e:
        res = f"fired: {e}"
    out = dict(t0=str(t0.date()), clean_plan_rows_asserted=clean["n_plan_rows_asserted"], forged=res)
    print(json.dumps(out, indent=1))
    assert res.startswith("fired"), "the plan bound cannot fail -- invalid"
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["falsify"]); ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    falsify(a.world)
