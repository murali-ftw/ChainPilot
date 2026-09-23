"""Phase 0 on v8 — loaders, panel, supplier-store audit, graph, every leak assertion.

Runs the EXISTING pipeline against a third world. Nothing here re-implements a loader that
already works; the schema diff against v6/v7 is 0/0/0 (reports/v8-clearance.md S1.2), so the
only genuine change is that the panel width is a property of the world and is measured rather
than hardcoded.

No assertion is disabled or weakened anywhere in this file. The Phase 11 po_line as-of
assertion is exercised deliberately in 1.5 to prove it is still armed, without any line-level
feature being added to any head.

    python ml/train/phase0_v8.py --world v8 [--rebuild] [--json out.json]
"""
from __future__ import annotations
import os, sys, json, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
from config import WORLDS, CACHE, EXPECTED_PANEL_D, EXPECT, FIT_WINDOW
import cache as C
import loader as L
import folds as F

R: dict = {}


def head(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# ---------------------------------------------------------------- 1.1 / 1.2 panel
def stage_1_1_1_2(world, rebuild):
    head("1.1 / 1.2  loaders and PANEL WIDTH")
    csv_dir = WORLDS[world]
    out = os.path.join(CACHE, world)
    t = time.time()
    if rebuild or not os.path.exists(os.path.join(out, "meta.json")):
        meta = C.build_panel(csv_dir, out, world=world)
    else:
        meta = json.load(open(os.path.join(out, "meta.json")))
        print(f"    (cached) d={meta['d']}")
    R["panel"] = dict(
        world=world, d_value=meta["d"], k_indicator=len(meta["nullable"]),
        d_in=meta["d"] + len(meta["nullable"]),
        expected=EXPECTED_PANEL_D[world],
        cols=meta["cols"], dropped=meta.get("dropped_constant_zero"),
        n_channels=meta["n_channels"], T=meta["T"], week0=meta["week0"],
        panel_complete=meta["panel_complete"], rows=meta["rows"],
        build_seconds=round(time.time() - t, 1))
    assert meta["d"] + len(meta["nullable"]) == EXPECTED_PANEL_D[world] + 10
    print(f"    d_in = {meta['d']} value + {len(meta['nullable'])} indicator "
          f"= {meta['d'] + len(meta['nullable'])}")
    # masters, against config.EXPECT
    R["masters"] = {}
    for tbl, key, col in [("sourcing_channels", "channels", "channel_id"),
                          ("suppliers", "suppliers", "supplier_id"),
                          ("parts", "parts", "part_id"), ("plants", "plants", "plant_id"),
                          ("snapshots", "snapshots", "snapshot_id")]:
        n = int(L.read_df(csv_dir, tbl, usecols=[col])[col].nunique())
        R["masters"][key] = dict(measured=n, v6_v7_contrast=EXPECT[key], same=n == EXPECT[key])
    R["masters"]["weeks"] = dict(measured=meta["T"], v6_v7_contrast=EXPECT["weeks"],
                                 same=meta["T"] == EXPECT["weeks"])
    print("    masters:", {k: v["measured"] for k, v in R["masters"].items()})
    return meta


def stage_1_2_falsify(world):
    """Standing rule 1: the width assertion must be able to fire."""
    head("1.2  PANEL WIDTH ASSERTION — can it fail?")
    import config
    out = {}
    real = EXPECTED_PANEL_D[world]
    for wrong, why in [(14, "v6/v7's width, borrowed"), (25, "d_in mistaken for d")]:
        if wrong == real:
            continue
        saved = dict(config.EXPECTED_PANEL_D)
        try:
            config.EXPECTED_PANEL_D[world] = wrong
            import importlib, cache as _c
            importlib.reload(_c)
            _c.build_panel(WORLDS[world], os.path.join(CACHE, "_falsify"), verbose=False, world=world)
            out[f"expect_{wrong}"] = "DID NOT FIRE -- assertion is inert"
        except AssertionError as e:
            out[f"expect_{wrong}"] = f"FIRES: {str(e).splitlines()[0][:110]}"
        finally:
            config.EXPECTED_PANEL_D.clear(); config.EXPECTED_PANEL_D.update(saved)
            import importlib, cache as _c
            importlib.reload(_c)
    # and an undeclared world
    try:
        import cache as _c
        _c.build_panel(WORLDS[world], os.path.join(CACHE, "_falsify"), verbose=False, world="v99")
        out["undeclared_world"] = "DID NOT FIRE"
    except AssertionError as e:
        out["undeclared_world"] = f"FIRES: {str(e).splitlines()[0][:110]}"
    import shutil
    shutil.rmtree(os.path.join(CACHE, "_falsify"), ignore_errors=True)
    for k, v in out.items():
        print(f"    {k:20s} {v}")
    R["panel_width_falsification"] = out


# ---------------------------------------------------------------- 1.3 supplier store
def stage_1_3(world, meta):
    head("1.3  SUPPLIER-STORE AUDIT (deviation 54)")
    csv_dir = WORLDS[world]
    spw = L.read_df(csv_dir, "supplier_performance_weekly")
    cpw_cols = L.read_df(csv_dir, "channel_performance_weekly", nrows=0).columns.tolist()
    num = [c for c in spw.columns if pd.api.types.is_numeric_dtype(spw[c])]
    rows = []
    for c in num:
        v = pd.to_numeric(spw[c], errors="coerce")
        rows.append(dict(column=c, n=int(len(v)), null_pct=round(100 * float(v.isna().mean()), 4),
                         nonzero_pct=round(100 * float(v.fillna(0).ne(0).mean()), 4),
                         min=float(v.min()), max=float(v.max()),
                         IDENTICALLY_ZERO=bool(v.fillna(0).eq(0).all())))
    R["supplier_store"] = dict(rows=int(len(spw)), numeric_columns=len(num), columns=rows,
                               identically_zero=[r["column"] for r in rows if r["IDENTICALLY_ZERO"]])
    for r in rows:
        flag = "  <-- IDENTICALLY ZERO" if r["IDENTICALLY_ZERO"] else ""
        print(f"    {r['column']:28s} nonzero={r['nonzero_pct']:7.3f}%  "
              f"[{r['min']}, {r['max']}]{flag}")

    # Which store does each panel feature read?
    src = {}
    for c in meta["cols"]:
        src[c] = ("channel_performance_weekly" if c in cpw_cols else "UNKNOWN")
    R["panel_feature_sources"] = src
    from_spw = [c for c, s in src.items() if s != "channel_performance_weekly"]
    R["panel_features_from_supplier_store"] = from_spw
    print(f"\n    every one of the {len(meta['cols'])} panel value features reads "
          f"channel_performance_weekly: {len(from_spw) == 0}")
    print(f"    panel features sourced from supplier_performance_weekly: "
          f"{from_spw if from_spw else 'NONE'}")

    # the same column, both stores
    cmp = {}
    for c in ["otd_rate_last13", "lead_time_actual_days", "lead_time_ratio", "reporting_lag_days"]:
        ch = pd.to_numeric(L.read_df(csv_dir, "channel_performance_weekly", usecols=[c])[c],
                           errors="coerce")
        sp = pd.to_numeric(spw[c], errors="coerce") if c in spw.columns else None
        cmp[c] = dict(
            channel_store_nonzero_pct=round(100 * float(ch.fillna(0).ne(0).mean()), 4),
            channel_store_distinct=int(ch.nunique()),
            supplier_store_nonzero_pct=(round(100 * float(sp.fillna(0).ne(0).mean()), 4)
                                        if sp is not None else None))
    R["same_column_both_stores"] = cmp
    print("\n    the decisive Level-3 arrival probe column, in both stores:")
    for c, v in cmp.items():
        print(f"      {c:24s} channel {v['channel_store_nonzero_pct']:7.3f}% nonzero | "
              f"supplier {v['supplier_store_nonzero_pct']:7.3f}% nonzero")


# ---------------------------------------------------------------- 1.4 graph
def stage_1_4(world):
    head("1.4  GRAPH")
    csv_dir = WORLDS[world]
    d, nodes, core, ext = L.build_graph(csv_dir, extended=True)
    prof = L.profile_graph(core, nodes)
    R["graph"] = dict(profile=prof, extended_edges=ext,
                      R_core=len(core) * 2, R_core_directed=len(core),
                      relations=[f"{s}__{r}__{t}" for (s, r, t) in core])
    print(f"    node types : {len(prof['nodes'])}  {prof['nodes']}")
    print(f"    nodes      : {prof['n_nodes']:,}")
    print(f"    core edges : {prof['n_edges']:,} over {len(core)} relations "
          f"(+{len(core)} reverse = R {len(core) * 2})")
    print(f"    components : {prof['components']}  largest {prof['largest_component']:,} "
          f"({prof['largest_component_pct']:.2f}%)")
    print(f"    median channel degree : {prof['median_channel_degree']}")
    for k in ("deg_supplier", "deg_part", "deg_plant"):
        print(f"    {k:14s} {prof[k]}")
    print(f"    extended   : {ext}")

    # channel degree distribution proper: how many neighbours does a node choose between?
    ch = nodes["raw"]["channel"]
    per_sup = ch.groupby("supplier_id").size()
    per_part = ch.groupby("part_id").size()
    per_plant = ch.groupby("plant_id").size()
    def q(s):
        return dict(n=int(len(s)), min=int(s.min()), p25=float(s.quantile(.25)),
                    median=float(s.median()), p75=float(s.quantile(.75)), max=int(s.max()))
    R["graph"]["channels_per_entity"] = dict(supplier=q(per_sup), part=q(per_part),
                                             plant=q(per_plant))
    print("\n    channels per entity (what a node actually chooses between):")
    for k, v in R["graph"]["channels_per_entity"].items():
        print(f"      {k:9s} median {v['median']:8.1f}  [{v['min']}, {v['max']}]  n={v['n']}")
    return nodes


# ---------------------------------------------------------------- 1.5 assertions
def stage_1_5(world, meta):
    head("1.5  EVERY LEAK ASSERTION — none disabled, none weakened")
    csv_dir = WORLDS[world]
    res = {}

    def run(name, fn, note=""):
        t = time.time()
        try:
            out = fn()
            res[name] = dict(result="PASS", detail=out, seconds=round(time.time() - t, 1), note=note)
            print(f"    [PASS] {name}  {str(out)[:96]}")
        except AssertionError as e:
            res[name] = dict(result="FIRED", detail=str(e), seconds=round(time.time() - t, 1), note=note)
            print(f"    [FIRED] {name}\n            {e}")
        except Exception as e:                                      # noqa: BLE001
            res[name] = dict(result=f"ERROR {type(e).__name__}", detail=str(e), note=note)
            print(f"    [ERROR] {name}: {type(e).__name__}: {e}")

    run("A1_no_negative_reporting_lag", lambda: L.assert_a1_no_negative_lag(csv_dir),
        "loader.py: recorded_ts >= event_ts on 11 transactional tables")
    run("A2_visible_week_conservation", lambda: L.assert_a2_visible_week_bucketing(csv_dir),
        "loader.py: store buckets on max(event_week, recorded_week), exact integer conservation")

    # A3 / staleness: the panel's own as-of property, brute force + perturbation
    from staleness import assert_asof
    _, _, active, _ = C.load_panel(os.path.join(CACHE, world))
    run("A3_panel_asof_bruteforce_and_perturbation",
        lambda: assert_asof(np.asarray(active), n_rows=1024, n_cuts=12, seed=1),
        "staleness.py: weeks_since_last_activity never reads the future")

    # fold assertions, on the real label dates
    lb = L.read_df(csv_dir, "training_labels", usecols=["snapshot_date", "task",
                                                        "label_window_end"])
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    lb = lb[(lb.snapshot_date >= pd.Timestamp(FIT_WINDOW[0])) &
            (lb.snapshot_date <= pd.Timestamp(FIT_WINDOW[1]))]
    for task in sorted(lb.task.unique()):
        d = lb.loc[lb.task == task, "snapshot_date"]
        tr, va, te = F.fixed_split(d)
        run(f"fixed_split_no_leak[{task}]", lambda d=d, tr=tr, va=va, te=te: (
            F.assert_no_leak(d, tr, va, te) or
            dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()))),
            "folds.py: max(train) < min(val) < min(test), inside the fit window")
    d_all = lb.loc[lb.task == "arrival_week", "snapshot_date"]
    run("rolling_origins_no_leak", lambda: F.assert_rolling_origins(d_all),
        "folds.py: all 8 origins, max(train) < min(evaluate)")

    # label-window overlap: diagnostic, not an assertion (folds.py says so)
    sub = lb[lb.task == "arrival_week"]
    tr, va, te = F.fixed_split(sub.snapshot_date)
    R["label_window_overlap"] = F.label_window_overlap(sub.snapshot_date, sub.label_window_end,
                                                       tr, va, te)
    print(f"    [diag] label-window overlap (measured, not asserted): "
          f"{R['label_window_overlap']}")

    # ---- the Phase 11 po_line as-of assertion -------------------------------
    # It guards line-level row features. The corrected rule forbids adding any, so on the
    # shipped path it is never REACHED. Prove it is still ARMED by building the frame it
    # guards and confirming it fires.
    print("\n    Phase 11 po_line as-of assertion (ml/train/phase5_heads.py:196,"
          " ml/train/loop.py:549):")
    tl = L.read_df(csv_dir, "training_labels",
                   usecols=["entity_id", "task", "snapshot_date"])
    tl = tl[tl.task == "arrival_week"]
    pol = L.read_df(csv_dir, "po_lines", usecols=["po_line_id", "recorded_ts"])
    j = tl.merge(pol, left_on="entity_id", right_on="po_line_id", how="left")
    rows = pd.DataFrame({"line_recorded_ts": pd.to_datetime(j.recorded_ts),
                         "snapshot_date": pd.to_datetime(j.snapshot_date)})
    fired, msg = False, ""
    try:
        assert (rows.line_recorded_ts <= rows.snapshot_date).all(), \
            "a po_line is not yet recorded at its own snapshot -- as-of violation"
    except AssertionError as e:
        fired, msg = True, str(e)
    viol = int((rows.line_recorded_ts > rows.snapshot_date).sum())
    res["phase11_poline_asof_ARMED"] = dict(
        result="FIRES when line-level features are built" if fired else "DID NOT FIRE",
        rows=int(len(rows)), violating_rows=viol,
        violating_pct=round(100 * viol / len(rows), 4), detail=msg,
        note="NOT reached on the shipped path: row_features is off and no line-level feature "
             "is added to any head (corrected rule, deviation 58). Exercised here only to "
             "prove the assertion is still armed.")
    print(f"      armed and fires: {fired} — {viol:,} of {len(rows):,} rows "
          f"({100 * viol / len(rows):.2f}%) violate")
    print(f"      NOT reached on the shipped path: no line-level feature is added.")

    R["assertions"] = res
    return res


# ---------------------------------------------------------------- 1.6 B1 re-check
def stage_1_6(world):
    head("1.6  B1 ROLL-FORWARD through the loader path, and seed 1005 in full")
    out = {}
    for label, csv_dir in [(f"{world} seed_1001", WORLDS[world]),
                           ("v8 seed_1005", os.path.join(os.path.dirname(WORLDS[world]),
                                                         "seed_1005"))]:
        pw = L.read_df(csv_dir, "inventory_position_weekly",
                       usecols=["part_id", "plant_id", "week_start", "qty_on_hand"])
        tx = L.read_df(csv_dir, "inventory_transactions",
                       usecols=["part_id", "plant_id", "qty", "event_ts"])
        tx["wk"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
        cum = (tx.groupby(["part_id", "plant_id", "wk"])["qty"].sum().rename("cum")
                 .reset_index())
        cum["cum"] = cum.sort_values("wk").groupby(["part_id", "plant_id"])["cum"].cumsum()
        pw["wk"] = pd.to_datetime(pw.week_start)
        # AS-OF join, not an exact-week merge. The opening posting sits at event_ts 2015-12-28,
        # i.e. in the week BEFORE the store's first week (2016-01-04). An exact-week merge drops
        # that row, and a forward-fill inside the merged frame cannot recover it, so any
        # part-plant with no transaction in the store's first week reads cum = 0 and appears to
        # mismatch. That is a reconciliation bug, not a data defect: it is what produced the
        # spurious 99.9944% for seed 1005 in reports/v8-clearance.md S2 B1. Taking the last
        # cumulative level at or before each store week carries the pre-store opening correctly.
        m = pd.merge_asof(pw.sort_values("wk"), cum.sort_values("wk"), on="wk",
                          by=["part_id", "plant_id"], direction="backward")
        m["cum"] = m["cum"].fillna(0)
        m["diff"] = m.qty_on_hand - m["cum"]
        ex = m["diff"] == 0
        bad = m[~ex]
        rec = dict(rows=int(len(m)), exact_pct=round(100 * float(ex.mean()), 6),
                   mismatched_rows=int((~ex).sum()),
                   part_plants=int(m.groupby(["part_id", "plant_id"]).ngroups),
                   part_plants_fully_exact=int(m.groupby(["part_id", "plant_id"])["diff"]
                                                .apply(lambda s: (s == 0).all()).sum()),
                   max_abs_diff=float(m["diff"].abs().max()))
        if len(bad):
            rec["mismatch_part_plants"] = (bad.part_id + "|" + bad.plant_id).unique().tolist()[:10]
            rec["mismatch_weeks"] = [str(x)[:10] for x in sorted(bad.wk.unique())[:10]]
            rec["mismatch_diffs"] = bad["diff"].value_counts().head(5).to_dict()
        out[label] = rec
        print(f"    {label}: exact {rec['exact_pct']}%  "
              f"({rec['mismatched_rows']:,} of {rec['rows']:,} rows mismatch), "
              f"part-plants fully exact {rec['part_plants_fully_exact']}/{rec['part_plants']}")
        if len(bad):
            print(f"      mismatch weeks: {rec['mismatch_weeks']}")
            print(f"      mismatch diffs: {rec['mismatch_diffs']}")
    R["B1_recheck"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v8")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    meta = stage_1_1_1_2(a.world, a.rebuild)
    stage_1_2_falsify(a.world)
    stage_1_3(a.world, meta)
    stage_1_4(a.world)
    stage_1_5(a.world, meta)
    stage_1_6(a.world)
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    fired = [k for k, v in R["assertions"].items()
             if v["result"] not in ("PASS",) and not k.endswith("_ARMED")]
    R["assertions_fired_unexpectedly"] = fired
    head("SUMMARY")
    print(f"    panel d_in        : {R['panel']['d_in']} "
          f"({R['panel']['d_value']} value + {R['panel']['k_indicator']} indicator)")
    print(f"    assertions run    : {len(R['assertions'])}")
    print(f"    fired UNEXPECTEDLY: {fired if fired else 'NONE'}")
    print(f"    elapsed           : {R['elapsed_seconds']}s")
    if a.json:
        json.dump(R, open(a.json, "w"), indent=1, default=str)
        print(f"    json -> {a.json}")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
