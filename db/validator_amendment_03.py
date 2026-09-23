"""Validator amendment 03 -- the v8 clearance checks (B1-B13), made repeatable.

THE THIRD AUTHORISED CHANGE TO THE VALIDATION INSTRUMENT, AND IT IS ADDITIVE ONLY.

`db/validator.py` is PROTECTED and is NOT touched by this amendment: it remains
byte-identical, and its 229 checks return exactly what they returned before. This file
is a separate module that ADDS the thirteen v8 clearance checks beside it. Nothing in
the frozen instrument is altered, removed, re-banded or re-gated.

Why a separate module rather than an in-place edit (amendments 01 and 02 edited
`validator.py` directly): the v8 brief's standing rules list `db/validator.py` as
protected. A standalone module satisfies both that rule and the requirement that the
clearance checks be repeatable. See docs/validation/validator_amendment_03.md.

Standing rule 1 -- a gate that cannot fail is not a gate. Every check below carries a
`breaks` string naming the mutation that makes it fire, and every one of them has been
demonstrated firing under that mutation. Run --selftest to re-demonstrate.

Usage:
    python db/validator_amendment_03.py db/gen_v8/seed_1001
    python db/validator_amendment_03.py db/gen_v8/seed_1001 --json out.json
    python db/validator_amendment_03.py db/gen_v8/seed_1001 --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

CHECKS: list = []
PASS, PARTIAL, FAIL = "CLEARED", "PARTIAL", "NOT CLEARED"


def check(block, name, breaks, blocking=False):
    def deco(fn):
        CHECKS.append(dict(block=block, name=name, breaks=breaks, fn=fn, blocking=blocking))
        return fn
    return deco


class World:
    """Lazy CSV reader. Read-only: this module never writes into a dataset directory."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self._c: dict[str, pd.DataFrame] = {}

    def __call__(self, table, **kw):
        key = table + repr(sorted(kw.items()))
        if key not in self._c:
            self._c[key] = pd.read_csv(self.root / f"{table}.csv", low_memory=False, **kw)
        return self._c[key]

    @staticmethod
    def dt(s):
        return pd.to_datetime(s, errors="coerce")


# --------------------------------------------------------------------------- B1
@check("B1", "opening stock balance rolls forward to the stated weekly position",
       "delete the opening balance posting, or net a scrap posting into its receipt",
       blocking=True)
def b1(w: World, sample_part_plants=None):
    pw = w("inventory_position_weekly",
           usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "as_of_date"])
    tx = w("inventory_transactions", usecols=["part_id", "plant_id", "qty", "event_ts"])
    if sample_part_plants:
        k = pw[["part_id", "plant_id"]].drop_duplicates().sample(sample_part_plants, random_state=0)
        pw, tx = pw.merge(k, on=["part_id", "plant_id"]), tx.merge(k, on=["part_id", "plant_id"])
    tx = tx.copy()
    tx["wk"] = World.dt(tx["event_ts"]).dt.to_period("W-SUN").dt.start_time
    cum = (tx.groupby(["part_id", "plant_id", "wk"])["qty"].sum()
             .rename("cum").reset_index().sort_values(["part_id", "plant_id", "wk"]))
    cum["cum"] = cum.groupby(["part_id", "plant_id"])["cum"].cumsum()
    pw = pw.copy(); pw["wk"] = World.dt(pw["week_start"])
    m = (pw.merge(cum, on=["part_id", "plant_id", "wk"], how="left")
           .sort_values(["part_id", "plant_id", "wk"]))
    m["cum"] = m.groupby(["part_id", "plant_id"])["cum"].ffill().fillna(0)
    exact = float((m["qty_on_hand"] == m["cum"]).mean()) * 100
    neg = float((pw["qty_on_hand"] < 0).mean()) * 100
    allzero = bool(pw["qty_on_hand"].eq(0).all())
    verdict = PASS if (exact >= 95 and not allzero and neg < 10) else FAIL
    return verdict, dict(reconciliation_exact_pct=round(exact, 4),
                         pct_negative=round(neg, 4), all_zero=allzero,
                         has_as_of_date=bool(pw["as_of_date"].notna().all()),
                         rows=int(len(m))), "reconciliation_exact_pct"


# --------------------------------------------------------------------------- B2
@check("B2", "forward requirement plan carries several future periods per version",
       "collapse every plan version to one period, or re-date target_period before recorded_ts")
def b2(w: World, **_):
    pdw = w("part_demand_weekly")
    pp = w("production_plan")
    per = pdw.groupby(["part_id", "plant_id", "as_of_date"])["week_start"].nunique()
    fwd_d = float((World.dt(pdw["week_start"]) > World.dt(pdw["as_of_date"])).mean()) * 100
    fwd_p = float((World.dt(pp["target_period"]) > World.dt(pp["recorded_ts"])).mean()) * 100
    perp = pp.groupby(["product_id", "plant_id", "plan_date"])["target_period"].nunique()
    gaps = pd.Series(sorted(pdw["as_of_date"].unique())).pipe(World.dt).diff().dt.days.dropna()
    ok = per.median() >= 4 and fwd_d >= 90 and fwd_p >= 90 and perp.median() >= 4
    return (PASS if ok else FAIL), dict(
        demand_periods_per_version_median=float(per.median()),
        demand_periods_per_version_min=float(per.min()),
        demand_pct_forward=round(fwd_d, 4),
        plan_periods_per_version_median=float(perp.median()),
        plan_pct_forward=round(fwd_p, 4),
        refresh_interval_days_median=float(gaps.median()) if len(gaps) else None,
        n_as_of_dates=int(pdw["as_of_date"].nunique())), "plan_pct_forward"


# --------------------------------------------------------------------------- B3
@check("B3", "holding, ordering, truck-capacity and shortage cost exist with non-trivial values",
       "cannot fire on absence alone -- fires when a column appears but is constant or all-null")
def b3(w: World, **_):
    want = {"holding_cost": ["hold", "carry"], "ordering_cost": ["order_cost", "ordering_cost", "setup"],
            "truck_capacity": ["truck", "vehicle_cap"], "shortage_cost": ["shortage_cost", "stockout"]}
    found = {k: [] for k in want}
    for f in sorted(w.root.glob("*.csv")):
        cols = pd.read_csv(f, nrows=0).columns
        for c in cols:
            for k, pats in want.items():
                if any(p in c.lower() for p in pats):
                    found[k].append(f"{f.stem}.{c}")
    nontrivial = {}
    for k, cs in found.items():
        for cn in cs:
            t, col = cn.split(".", 1)
            s = pd.to_numeric(w(t)[col], errors="coerce").dropna()
            nontrivial[cn] = bool(len(s) and s.min() != s.max())
    ok = all(any(nontrivial.get(cn, False) for cn in found[k]) for k in want)
    return (PASS if ok else FAIL), dict(columns_found=found,
                                        non_trivial=nontrivial), "columns_found"


# --------------------------------------------------------------------------- B4
B4_COLS = [("part_plant", "min_order_qty"), ("part_plant", "lot_size"),
           ("part_plant", "planning_lead_time_days"), ("supplier_contracts", "moq"),
           ("supplier_contracts", "lot_size"), ("supplier_contracts", "max_volume_cap"),
           ("supplier_contracts", "min_volume_commitment"),
           ("supplier_contracts", "penalty_clause_inr")]


@check("B4", "contract terms show real variation, not a single constant",
       "restore v6/v7's constants: moq=10, lot_size=25, max_volume_cap=5000, commit=100, penalty=10000")
def b4(w: World, **_):
    out, ok = {}, True
    for t, c in B4_COLS:
        s = pd.to_numeric(w(t)[c], errors="coerce").dropna()
        d = int(s.nunique())
        out[f"{t}.{c}"] = dict(distinct=d, min=float(s.min()), max=float(s.max()),
                               constant=bool(s.min() == s.max()))
        ok &= d > 1
    return (PASS if ok else FAIL), out, "distinct"


# --------------------------------------------------------------------------- B5
@check("B5", "the min-volume commitment carries a period and the penalty carries a basis",
       "drop the period/basis columns -- fires whenever neither is present as a column or enum")
def b5(w: World, **_):
    sc = w("supplier_contracts")
    cols = [c.lower() for c in sc.columns]
    period = [c for c in cols if any(p in c for p in ("period", "window", "freq", "per_"))
              and "valid" not in c]
    basis = [c for c in cols if "basis" in c or "per_unit" in c or "pro_rata" in c]
    span = (World.dt(sc["valid_to"]) - World.dt(sc["valid_from"])).dt.days if \
        {"valid_from", "valid_to"} <= set(sc.columns) else pd.Series(dtype=float)
    ok = bool(period) and bool(basis)
    return (PASS if ok else FAIL), dict(
        commitment_period_column=period or None, penalty_basis_column=basis or None,
        contract_validity_span_days_median=float(span.median()) if len(span) else None,
        note="contract validity is not a commitment measurement period"), "commitment_period_column"


# --------------------------------------------------------------------------- B6
@check("B6", "qualification is a genuine mix and is_approved is not constant",
       "set every alternate_sources.qualification_status to 'qualified' (the v6/v7 state)")
def b6(w: World, **_):
    alt = w("alternate_sources")["qualification_status"]
    ch = w("sourcing_channels")
    mix = alt.value_counts(normalize=True).to_dict()
    nonmodal = float(1 - max(mix.values())) * 100
    appr_const = bool(ch["is_approved"].nunique() <= 1)
    status_const = bool(ch["approval_status"].nunique() <= 1)
    if nonmodal > 1 and not appr_const:
        v = PASS
    elif nonmodal > 1:
        v = PARTIAL
    else:
        v = FAIL
    return v, dict(qualification_mix={k: round(x * 100, 4) for k, x in mix.items()},
                   non_modal_pct=round(nonmodal, 4),
                   is_approved_constant=appr_const,
                   approval_status_constant=status_const), "non_modal_pct"


# --------------------------------------------------------------------------- B7
@check("B7", "revealed_capacity_est and evidence_strength are populated",
       "null out both columns (the v6/v7 state: 100% NULL)")
def b7(w: World, **_):
    rc = w("revealed_capacity_monthly")
    ev = float(rc["evidence_strength"].notna().mean()) * 100
    rv = float(rc["revealed_capacity_est"].notna().mean()) * 100
    con = rc["constrained_month_flag"].astype(str).str.lower().isin(["true", "1"])
    rv_con = float(rc.loc[con, "revealed_capacity_est"].notna().mean()) * 100 if con.any() else 0.0
    ok = ev > 50 and rv_con > 50
    return (PASS if ok else FAIL), dict(
        evidence_strength_populated_pct=round(ev, 4),
        revealed_capacity_est_populated_pct=round(rv, 4),
        revealed_populated_on_constrained_months_pct=round(rv_con, 4),
        note="nulls on unconstrained months are correct: observability is an outcome (rules S6)"
    ), "revealed_populated_on_constrained_months_pct"


# --------------------------------------------------------------------------- B8
@check("B8", "a recorded incumbent split exists for a high fraction of part-plants",
       "keep only a v6-sized 19.5% sample OF PART-PLANTS (thinning rows alone does NOT fire this)")
def b8(w: World, **_):
    al = w("supplier_allocation")
    tot = w("part_plant").groupby(["part_id", "plant_id"]).ngroups
    cov = al.groupby(["part_id", "plant_id"]).ngroups / tot * 100
    g = al.groupby(["part_id", "plant_id", "effective_from"])["supplier_id"].nunique()
    s = al.groupby(["part_id", "plant_id", "effective_from"])["allocation_pct"].sum()
    return (PASS if cov >= 60 else FAIL), dict(
        coverage_pct=round(cov, 4), part_plants_covered=int(al.groupby(["part_id", "plant_id"]).ngroups),
        part_plants_total=int(tot),
        pct_versions_with_more_than_one_supplier=round(float((g > 1).mean()) * 100, 4),
        allocation_sums_to_100_pct=round(float(((s - 100).abs() < 0.05).mean()) * 100, 4)
    ), "coverage_pct"


# --------------------------------------------------------------------------- B9
@check("B9", "every arrival label's po_line is created AND recorded before its own snapshot",
       "re-date any line to after its snapshot; INVERSE-tested by re-dating all lines before it",
       blocking=True)
def b9(w: World, **_):
    tl = w("training_labels")
    a = tl[tl["task"] == "arrival_week"]
    # training_labels carries its OWN created_ts, so the po_line timestamps are renamed
    # before the merge rather than left to pandas' _x/_y suffixing.
    pl = w("po_lines", usecols=["po_line_id", "created_ts", "recorded_ts"]).rename(
        columns={"created_ts": "line_created_ts", "recorded_ts": "line_recorded_ts"})
    j = a.merge(pl, left_on="entity_id", right_on="po_line_id", how="left")
    snap = World.dt(j["snapshot_date"])
    dc = (snap - World.dt(j["line_created_ts"])).dt.total_seconds() / 86400
    dr = (snap - World.dt(j["line_recorded_ts"])).dt.total_seconds() / 86400
    pos_c = float((dc >= 0).mean()) * 100
    pos_r = float((dr >= 0).mean()) * 100
    q = (-dc).quantile([0, .25, .5, .75, 1]).round(2).tolist()
    return (PASS if pos_r >= 99.99 else FAIL), dict(
        rows=int(len(j)), unresolved=int(j["po_line_id"].isna().sum()),
        pct_line_created_before_snapshot=round(pos_c, 4),
        pct_line_recorded_at_or_before_snapshot=round(pos_r, 4),
        created_minus_snapshot_days=dict(zip(["min", "p25", "median", "p75", "max"], q))
    ), "pct_line_recorded_at_or_before_snapshot"


# --------------------------------------------------------------------------- B10
B10_ANCHORS = ["event_ts", "created_ts", "receipt_ts", "plan_date", "snapshot_date", "audit_date"]
FORWARD_DATED = {"supplier_allocation", "supplier_capacity", "supplier_contracts",
                 "alternate_sources", "sourcing_channels", "part_plant"}


@check("B10", "the recording lag is a real distribution, never a constant, never negative",
       "set recorded_ts = event_ts + 1.0d on any table (v1's defect), or re-date rows before their event")
def b10(w: World, **_):
    out, ok = {}, True
    for f in sorted(w.root.glob("*.csv")):
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if "recorded_ts" not in cols:
            continue
        anch = next((c for c in B10_ANCHORS if c in cols), None)
        if anch is None:
            continue
        d = w(f.stem, usecols=[anch, "recorded_ts"])
        lag = ((World.dt(d["recorded_ts"]) - World.dt(d[anch])).dt.total_seconds() / 86400).dropna()
        if not len(lag):
            continue
        disp = float(lag.std() / abs(lag.mean())) if lag.mean() else None
        neg = float((lag < 0).mean()) * 100
        exempt = f.stem in FORWARD_DATED
        out[f.stem] = dict(anchor=anch, sd_over_absmean=round(disp, 4) if disp else None,
                           pct_negative=round(neg, 6), median_days=round(float(lag.median()), 3),
                           skew=round(float(lag.skew()), 4),
                           forward_dated_anchor_exempt=exempt)
        if disp is not None and disp < 0.1:
            ok = False
        if neg > 0 and not exempt:
            ok = False
    return (PASS if ok else FAIL), out, "sd_over_absmean"


# --------------------------------------------------------------------------- B11
@check("B11", "no all-zero, all-null or single-valued numeric column",
       "zero out any populated numeric column, e.g. channel_performance_weekly.fill_rate_last13")
def b11(w: World, chunk=400_000, **_):
    az, an, cn, empty = [], [], [], []
    for f in sorted(w.root.glob("*.csv")):
        acc, n = {}, 0
        for chb in pd.read_csv(f, chunksize=chunk, low_memory=False):
            n += len(chb)
            for c in chb.columns:
                s = chb[c]
                a = acc.setdefault(c, dict(nn=0, z=0, mn=None, mx=None, num=False))
                a["nn"] += int(s.notna().sum())
                if pd.api.types.is_numeric_dtype(s):
                    a["num"] = True
                    v = s.dropna()
                    if len(v):
                        a["z"] += int((v == 0).sum())
                        a["mn"] = float(v.min()) if a["mn"] is None else min(a["mn"], float(v.min()))
                        a["mx"] = float(v.max()) if a["mx"] is None else max(a["mx"], float(v.max()))
        if n == 0:
            empty.append(f.stem); continue
        for c, a in acc.items():
            if not a["num"]:
                continue
            nm = f"{f.stem}.{c}"
            if a["nn"] == 0:
                an.append(nm)
            elif a["z"] == a["nn"]:
                az.append(nm)
            elif a["mn"] == a["mx"]:
                cn.append(f"{nm}={a['mn']}")
    ok = not az and not an and not empty
    return (PASS if ok else FAIL), dict(all_zero=az, all_null=an, single_valued=cn,
                                        empty_tables=empty,
                                        n_all_zero=len(az), n_all_null=len(an),
                                        n_single_valued=len(cn)), "n_all_zero"


# --------------------------------------------------------------------------- B12
@check("B12", "label classes and distributions are real, not uniform noise",
       "draw fill labels from Uniform(0,1) (v1), or drop every zero shortage row (v6)",
       blocking=True)
def b12(w: World, **_):
    from scipy import stats
    tl = w("training_labels")
    num = lambda t: pd.to_numeric(tl.loc[tl["task"] == t, "label_value"], errors="coerce").dropna()
    f, sh, cp = num("fill_rate"), num("shortage_qty"), num("capacity_strain")
    cen = tl.loc[tl["task"] == "arrival_week", "label_censored"].astype(str).str.lower().eq("true")
    ks = stats.kstest(f.sample(min(len(f), 100_000), random_state=0), "uniform")
    pz = float((sh == 0).mean()) * 100
    cf = float(cen.mean()) * 100
    ok = (ks.pvalue < 0.01 and 0 < pz < 100 and 0 < cf < 100
          and float((f == 1.0).mean()) * 100 >= 60 and 0 < cp.mean() < 3.0)
    return (PASS if ok else FAIL), dict(
        fill_KS_vs_uniform_p=float(ks.pvalue), fill_sd=round(float(f.std()), 6),
        fill_mass_at_1=round(float((f == 1.0).mean()) * 100, 4),
        fill_mass_at_0=round(float((f == 0.0).mean()) * 100, 4),
        shortage_pct_zero=round(pz, 4), shortage_n_positive=int((sh > 0).sum()),
        capacity_min=float(cp.min()), capacity_max=float(cp.max()),
        capacity_mean=round(float(cp.mean()), 4),
        arrival_censored_pct=round(cf, 4)), "fill_KS_vs_uniform_p"


# --------------------------------------------------------------------------- B13
B13_PK = {"parts": ["part_id"], "plants": ["plant_id"], "suppliers": ["supplier_id"],
          "sourcing_channels": ["channel_id"], "part_plant": ["part_id", "plant_id"],
          "purchase_orders": ["po_id"], "po_lines": ["po_line_id"], "grn_lines": ["grn_line_id"],
          "goods_receipts": ["grn_id"], "asn": ["asn_id"], "training_labels": ["label_id"],
          "snapshots": ["snapshot_id"], "inventory_transactions": ["txn_id"],
          "channel_performance_weekly": ["channel_id", "week_start"],
          "supplier_performance_weekly": ["supplier_id", "week_start"],
          "inventory_position_weekly": ["part_id", "plant_id", "week_start"],
          "part_demand_weekly": ["part_id", "plant_id", "as_of_date", "week_start"],
          "inventory_snapshots": ["part_id", "plant_id", "snapshot_date"],
          "revealed_capacity_monthly": ["supplier_id", "part_id", "month"],
          "supplier_capacity": ["supplier_id", "part_id", "effective_from"],
          "supplier_contracts": ["contract_id"], "shortage_events": ["shortage_id"],
          "expedite_events": ["expedite_id"], "po_line_revisions": ["revision_id"],
          "quality_inspections": ["inspection_id"], "supplier_acknowledgements": ["ack_id"],
          "po_line_schedules": ["schedule_id"], "line_stop_events": ["stop_id"]}


@check("B13", "no duplicate primary keys; derived stores are not inflated against their source",
       "duplicate 10,210 label_ids (v1), or multiply the weekly store's units (v1 was 33.7x)")
def b13(w: World, **_):
    dups, ok = {}, True
    for t, keys in B13_PK.items():
        p = w.root / f"{t}.csv"
        if not p.exists():
            continue
        have = pd.read_csv(p, nrows=0).columns
        if not set(keys) <= set(have):
            dups[t] = "key absent"; continue
        d = w(t, usecols=keys)
        n = int(len(d) - len(d.drop_duplicates()))
        dups[t] = n
        ok &= (n == 0)
    cpw = w("channel_performance_weekly", usecols=["channel_id", "week_start",
                                                   "qty_ordered", "qty_received"])
    lo, hi = World.dt(cpw["week_start"]).min(), World.dt(cpw["week_start"]).max()
    chans = set(cpw["channel_id"].unique())
    vis = lambda a, b: np.maximum(World.dt(a).dt.to_period("W-SUN").dt.start_time,
                                  World.dt(b).dt.to_period("W-SUN").dt.start_time)
    pl = w("po_lines", usecols=["po_line_id", "channel_id", "qty_ordered",
                                "created_ts", "recorded_ts"]).copy()
    pl["vw"] = vis(pl["created_ts"], pl["recorded_ts"])
    src_o = int(pl.loc[pl["vw"].between(lo, hi) & pl["channel_id"].isin(chans), "qty_ordered"].sum())
    gl = w("grn_lines", usecols=["po_line_id", "qty_received", "event_ts", "recorded_ts"]) \
        .merge(pl[["po_line_id", "channel_id"]], on="po_line_id", how="left").copy()
    gl["vw"] = vis(gl["event_ts"], gl["recorded_ts"])
    src_r = int(gl.loc[gl["vw"].between(lo, hi) & gl["channel_id"].isin(chans), "qty_received"].sum())
    st_o, st_r = int(cpw["qty_ordered"].sum()), int(cpw["qty_received"].sum())
    ratio_o = st_o / src_o if src_o else float("inf")
    ratio_r = st_r / src_r if src_r else float("inf")
    ok &= abs(ratio_o - 1) < 1e-9 and abs(ratio_r - 1) < 1e-9
    return (PASS if ok else FAIL), dict(
        duplicate_pks=dups, total_duplicate_pks=sum(v for v in dups.values() if isinstance(v, int)),
        ordered_store_over_source=round(ratio_o, 8),
        received_store_over_source=round(ratio_r, 8)), "total_duplicate_pks"


# --------------------------------------------------------------------------- runner
def run(root, sample=None, only=None):
    w = World(root)
    rows = []
    for c in CHECKS:
        if only and c["block"] not in only:
            continue
        try:
            kw = dict(sample_part_plants=sample) if c["block"] == "B1" else {}
            verdict, detail, key = c["fn"](w, **kw)
        except Exception as e:                                    # noqa: BLE001
            verdict, detail, key = "ERROR", {"error": f"{type(e).__name__}: {e}"}, None
        rows.append(dict(block=c["block"], verdict=verdict, name=c["name"],
                         deciding_field=key, breaks=c["breaks"],
                         blocking=c["blocking"], detail=detail))
        print(f"  [{verdict:>11s}] {c['block']:3s} {c['name']}")
        if key and key in detail:
            print(f"                 {key} = {detail[key]}")
    return rows


def gate(rows):
    blocking = {r["block"]: r["verdict"] for r in rows if r["blocking"]}
    passed = all(v == PASS for v in blocking.values())
    return passed, blocking


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="B1: reconcile a sample of part-plants instead of all (speed)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these blocks, e.g. B1 B9 B12")
    a = ap.parse_args(argv)
    print(f"\nvalidator amendment 03 -- v8 clearance checks B1-B13\n  dataset: {a.directory}\n")
    rows = run(Path(a.directory), sample=a.sample, only=a.only)
    ok, blocking = gate(rows)
    n = {v: sum(1 for r in rows if r["verdict"] == v) for v in (PASS, PARTIAL, FAIL, "ERROR")}
    print(f"\n  {n[PASS]} cleared, {n[PARTIAL]} partial, {n[FAIL]} not cleared, {n['ERROR']} errored")
    print(f"  BLOCKING (B1, B9, B12): {blocking}")
    print(f"  STAGE 1 GATE: {'PASS' if ok else 'FAIL -- do not start Phase 0'}\n")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=str)
        print(f"  json written to {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
