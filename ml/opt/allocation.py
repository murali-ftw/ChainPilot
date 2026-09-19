"""Guide 10.2 — allocation: candidate splits, the constraint layer, and two scorers.

The guide ranks candidate supplier splits by RE-RUNNING THE PHASE 9 SIMULATION under each. That simulation is
blocked and stays blocked: it needs an opening stock level, and no usable one exists (inventory_position_weekly is
all-zero and never read; inventory_snapshots.qty_on_hand is 97.5% negative in v6; the inventory_transactions cumsum
agrees with the stated balance on 0.33% of rows). So the objective lives behind an interface with two
implementations:

  SimulationScorer  the guide's version. Call signature implemented, body raises. NOT stubbed with a fake number.
  ReducedScorer     runnable now: expected UNMET DEMAND IN PERIOD + purchase cost. It has no inventory state, so it
                    cannot measure a stockout -- only demand a supplier is not expected to serve within the period.
                    WEAKER THAN THE GUIDE'S OBJECTIVE. Every number it produces says so.

INHERITED LIMITATION, carried here deliberately (Phase 9A/9B, Phase 8 section 7): capacity intervals are NOT
quotable. Empirical 80% coverage runs 0.72-0.81 and is below nominal in 12 of 16 backtest windows, in every model
class. ReducedScorer consumes a capacity-strain quantile, so its capacity term inherits that miscalibration and any
ranking it produces carries the same caveat.

  python ml/opt/allocation.py --world v6 --parts 5
"""
from __future__ import annotations
import os, sys, json, argparse, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from config import WORLDS, ARTIFACTS

FORBIDDEN = "inventory_position_weekly"

# Constraint parameters the DATA CANNOT EXPRESS. Each is injectable, each defaults to a stated assumption, and each
# is an explicit ask of Rane (report section 6). Nothing here was measured.
CLIENT_PARAMS = dict(
    commitment_period_days=90,        # supplier_contracts.min_volume_commitment carries no period. Over what window?
    ramp_baseline_period_days=30,     # ramp_rate_pct_per_month is a rate; the baseline it applies to is not given
    qualification_cutoff_days=0,      # how far ahead a qualification must complete to count as usable
    penalty_is_per_breach=True,       # penalty_clause_inr: per breach, per unit short, or pro-rata? Not stated
    max_suppliers_per_part=None,      # a sourcing policy, not a data property
)


def _asof(df, col, t0):
    if col not in df.columns:
        return df
    d = pd.to_datetime(df[col], errors="coerce")
    return df[d.isna() | (d <= pd.Timestamp(t0))]


def load(world, t0):
    csv = WORLDS[world]
    assert FORBIDDEN not in csv, "guard"
    L = {}
    L["channels"] = _asof(pd.read_csv(os.path.join(csv, "sourcing_channels.csv"),
                                      usecols=["channel_id", "supplier_id", "part_id", "plant_id", "is_approved",
                                               "approval_status", "effective_from"]), "effective_from", t0)
    L["alloc"] = _asof(pd.read_csv(os.path.join(csv, "supplier_allocation.csv"),
                                   usecols=["part_id", "plant_id", "supplier_id", "allocation_pct", "recorded_ts"]),
                       "recorded_ts", t0)
    L["alt"] = _asof(pd.read_csv(os.path.join(csv, "alternate_sources.csv")), "effective_from", t0)
    L["tool"] = pd.read_csv(os.path.join(csv, "tooling.csv"))
    L["con"] = _asof(pd.read_csv(os.path.join(csv, "supplier_contracts.csv")), "valid_from", t0)
    L["costs"] = _asof(pd.read_csv(os.path.join(csv, "part_costs.csv")), "effective_from", t0)
    dem = pd.read_csv(os.path.join(csv, "part_demand_weekly.csv"),
                      usecols=["part_id", "plant_id", "week_start", "as_of_date", "gross_requirement_p50"])
    dem = _asof(dem, "as_of_date", t0)
    dem["week_start"] = pd.to_datetime(dem.week_start)
    L["demand"] = dem[dem.week_start > pd.Timestamp(t0)]
    return L


# ================================================================== 3.1 candidates
def candidates(L, part, plant, n=5):
    """The guide's 'enumerate 5 candidate splits'. Shares are fractions summing to 1 over qualified suppliers."""
    ch = L["channels"]
    qual = sorted(ch[(ch.part_id == part) & (ch.plant_id == plant)].supplier_id.unique())
    if len(qual) == 0:
        return [], []
    inc = L["alloc"][(L["alloc"].part_id == part) & (L["alloc"].plant_id == plant)]
    if len(inc):
        base = {s: float(v) / 100.0 for s, v in zip(inc.supplier_id, inc.allocation_pct) if s in qual}
        tot = sum(base.values())
        base = {s: v / tot for s, v in base.items()} if tot > 0 else {}
    else:
        base = {}
    if not base:                                  # 80.5% of part-plants have NO recorded incumbent split
        base = {qual[0]: 1.0}
    incumbent = {s: base.get(s, 0.0) for s in qual}

    unit = L["costs"].groupby(["part_id", "supplier_id"]).unit_cost_inr.mean()
    price = {s: float(unit.get((part, s), np.nan)) for s in qual}
    cheapest = min((s for s in qual if np.isfinite(price[s])), key=lambda s: price[s], default=qual[0])
    lead = L["alt"].set_index(["part_id", "supplier_id"]).qualification_lead_days if len(L["alt"]) else None

    cands = [("incumbent", incumbent)]
    c2 = {s: 0.0 for s in qual}; c2[cheapest] = 1.0
    cands.append(("all to cheapest qualified", c2))
    cands.append(("equal split across qualified", {s: 1.0 / len(qual) for s in qual}))
    # shift 20 points from the largest incumbent holder to the cheapest alternative
    big = max(qual, key=lambda s: incumbent[s])
    tgt = min((s for s in qual if s != big and np.isfinite(price[s])), key=lambda s: price[s], default=None)
    c4 = dict(incumbent)
    if tgt is not None:
        move = min(0.20, c4[big])
        c4[big] -= move; c4[tgt] = c4.get(tgt, 0.0) + move
    cands.append(("shift 20pp from the largest holder to the cheapest alternative", c4))
    # dual-source: cap any single supplier at 70%, spread the remainder evenly
    c5 = dict(incumbent)
    over = {s: max(0.0, v - 0.70) for s, v in c5.items()}
    spill = sum(over.values())
    if spill > 0 and len(qual) > 1:
        for s in qual:
            c5[s] -= over[s]
        others = [s for s in qual if over[s] == 0]
        for s in others:
            c5[s] += spill / len(others)
    cands.append(("cap any supplier at 70%", c5))
    return qual, cands[:n]


# ================================================================== 3.2 the constraint layer
def constraint_report(L, part, plant, split, incumbent, params=CLIENT_PARAMS, disable=()):
    """Each constraint the guide calls 'the product'. Returns (feasible, [violations], [inert reasons]).

    `disable` is for the verify gates ONLY: removing a constraint lets a gate be shown FAILING on a deliberately
    broken constraint layer (standing rule 1). Never set it in production use.
    """
    disable = set(disable)
    viol, inert = [], []
    alt = L["alt"]; a = alt[(alt.part_id == part)] if len(alt) else alt
    tool = L["tool"]; t = tool[tool.part_id == part]
    con = L["con"]; c = con[con.part_id == part]

    # --- qualification: is every supplier receiving volume actually qualified, and in time?
    if len(a):
        status = set(a.qualification_status.unique())
        if status <= {"qualified"}:
            inert.append("qualification status: INERT -- every row in alternate_sources reads 'qualified', so this "
                         "constraint can never bind on this data")
        for s, share in split.items():
            row = a[a.supplier_id == s]
            if share > 0 and len(row) and str(row.qualification_status.iloc[0]) != "qualified":
                need = float(row.qualification_lead_days.iloc[0])
                if need > params["qualification_cutoff_days"]:
                    viol.append(f"{s}: not qualified, needs {need:.0f} days")

    # --- tooling: a supplier newly receiving volume needs transferable or duplicated tooling
    for s, share in split.items():
        gain = share - incumbent.get(s, 0.0)
        if gain > 1e-9 and "tooling" not in disable:
            ts = t[t.supplier_id == s]
            if len(ts):
                transferable = bool(ts.is_transferable.max())
                duplicate = bool(ts.duplicate_exists.max())
                if not transferable and not duplicate:
                    viol.append(f"{s}: tooling neither transferable nor duplicated (+{100*gain:.0f}pp blocked)")

    # --- ramp rate: how fast a supplier may grow its share
    for s, share in split.items():
        gain = share - incumbent.get(s, 0.0)
        row = a[a.supplier_id == s] if len(a) else a
        if gain > 1e-9 and len(row):
            cap = float(row.ramp_rate_pct_per_month.iloc[0]) / 100.0 * (params["ramp_baseline_period_days"] / 30.0)
            if gain > cap + 1e-9:
                viol.append(f"{s}: ramp {100*gain:.0f}pp exceeds {100*cap:.0f}pp per "
                            f"{params['ramp_baseline_period_days']}d [period ASSUMED: client parameter]")

    # --- minimum-volume commitment
    if len(c):
        if c.min_volume_commitment.nunique() == 1:
            inert.append(f"min_volume_commitment: constant {c.min_volume_commitment.iloc[0]:.0f} across every "
                         f"contract, and carries no period -- the window is a client parameter "
                         f"(assumed {params['commitment_period_days']}d)")
    return (len(viol) == 0), viol, inert


# ================================================================== 3.3 the objective
class SimulationScorer:
    """The guide's objective: re-run the Phase 9 simulation under the split, score expected shortage + purchase cost."""

    def __call__(self, L, part, plant, split, requirement):
        raise NotImplementedError(
            "blocked: Phase 9.1, no opening stock balance. inventory_position_weekly is all-zero and never read; "
            "inventory_snapshots.qty_on_hand is 97.5% negative (v6); the inventory_transactions cumsum agrees with "
            "the stated balance on 0.33% of rows. A shortage cannot be computed without an inventory level.")


class ReducedScorer:
    """Expected UNMET DEMAND IN PERIOD + purchase cost. No inventory state: this is NOT the guide's objective.

    unmet(split)  = requirement * sum_s share_s * (1 - E[fill_s]) * strain_penalty_s
    purchase      = requirement * sum_s share_s * unit_cost_s
    E[fill_s]     comes from the SHIPPED fill model (B5-flat-22 as of Phase 10 Stage 1.1), aggregated per supplier.
    strain_penalty scales with the supplier's capacity-strain quantile ABOVE 1.0 (utilisation over capacity), and
    inherits the interval miscalibration recorded in the module docstring.
    """

    def __init__(self, fill_by_supplier, strain_by_supplier, unit_cost, shortage_cost_per_unit=1000.0):
        self.fill, self.strain, self.cost = fill_by_supplier, strain_by_supplier, unit_cost
        self.shortage_cost = shortage_cost_per_unit          # ASSUMPTION: no shortage-cost column exists

    def __call__(self, L, part, plant, split, requirement):
        unmet = purchase = 0.0
        for s, share in split.items():
            if share <= 0:
                continue
            q = share * requirement
            fill = self.fill.get(s, np.nan)
            fill = 0.9 if not np.isfinite(fill) else fill
            strain = self.strain.get(s, 1.0)
            penalty = 1.0 + max(0.0, float(strain) - 1.0)     # P90 above 1.0 = expected over-utilisation
            unmet += q * (1.0 - fill) * penalty
            purchase += q * float(self.cost.get((part, s), np.nan) if np.isfinite(
                self.cost.get((part, s), np.nan)) else np.nanmean(list(self.cost.values())))
        return dict(expected_unmet_units=float(unmet), purchase_cost=float(purchase),
                    score=float(unmet * self.shortage_cost + purchase))


def supplier_signals(world, seeds=(7, 17, 27)):
    """Per-supplier signals from artifacts the pipeline already produced, one set per seed so the ranking can be
    checked against the seed bands (3.3).

      fill[s]    mean P(line fills completely) over that supplier's lines, from the SHIPPED fill model -- B5-flat-22
                 as of Phase 10 Stage 1.1, recalibrated, origin 7. It is a complete-fill PROBABILITY, not an expected
                 fill fraction: a partially filled line counts as unfilled here, so unmet demand is overstated.
      strain[s]  mean capacity-strain P90 over that supplier's channels. P90 rather than P50 because utilisation sits
                 near 0.65 (v6) and a penalty keyed to the median never activates -- the signal would be inert and the
                 seed-band check vacuous. P90 exceeds 1.0 for 16% of suppliers, so it discriminates.
                 IT ALSO INHERITS THE INTERVAL MISCALIBRATION: 80% coverage 0.72-0.81, below nominal in 12 of 16
                 backtest windows. Any ranking that turns on this term carries that caveat.
    """
    BT = os.path.join(ARTIFACTS, "backtest", "preds")
    csv = WORLDS[world]
    ch = pd.read_csv(os.path.join(csv, "sourcing_channels.csv"), usecols=["channel_id", "supplier_id"])
    ch_sup = dict(zip(ch.channel_id, ch.supplier_id))
    pol = pd.read_csv(os.path.join(csv, "po_lines.csv"), usecols=["po_line_id", "channel_id"])
    pol_sup = {p: ch_sup.get(c) for p, c in zip(pol.po_line_id, pol.channel_id)}
    out = []
    for s in seeds:
        strain, fill = {}, {}
        f = os.path.join(BT, f"{world}_capacity_strain_o7_mp_h4_lr0.00025_s{s}_test.npz")
        if os.path.exists(f):
            z = np.load(f)
            d = pd.DataFrame(dict(sup=[ch_sup.get(e) for e in z["entity"].astype(str)], q90=z["P"][:, 2])).dropna()
            strain = d.groupby("sup").q90.mean().to_dict()
        g = os.path.join(BT, f"RECAL_{world}_fill_rate_o7_b5flat22_s{s}_test.npz")
        if os.path.exists(g):
            z = np.load(g)
            d = pd.DataFrame(dict(sup=[pol_sup.get(e) for e in z["entity"].astype(str)],
                                  pc=z["P"][:, 21])).dropna()
            fill = d.groupby("sup").pc.mean().to_dict()
        out.append(dict(seed=s, strain=strain, fill=fill))
    return out


# ================================================================== 3.4 the verify gate, and whether it can fail
def verify_tooling_gate(L, disable=()):
    """The guide's gate: a part whose tooling is neither transferable nor duplicated must return the INCUMBENT split.

    Unlike 10.1's gate this one CAN bind: tooling.is_transferable is 0 for 41% of rows and duplicate_exists is 0 for
    70%, so parts that cannot move exist in the data. The gate is run against the real constraint layer (must hold)
    and against one with the tooling check removed (must break).
    """
    tool = L["tool"]
    stuck = tool[(tool.is_transferable == 0) & (tool.duplicate_exists == 0)]
    checked = blocked = moved = 0
    example = None
    ch = L["channels"]
    for part in stuck.part_id.unique():
        rows = ch[ch.part_id == part]
        stuck_sup = set(stuck[stuck.part_id == part].supplier_id)
        for plant in rows.plant_id.unique():
            qual, cands = candidates(L, part, plant)
            # the gate can only bind where a frozen supplier is actually QUALIFIED at this plant -- otherwise no
            # candidate ever offers it volume and the check is never exercised (29 such part-plants in v6)
            frozen = stuck_sup & set(qual)
            if len(qual) < 2 or not frozen:
                continue
            inc = dict(cands[0][1])
            frozen = {s for s in frozen if inc.get(s, 0.0) < 1.0}
            if not frozen:
                continue
            checked += 1
            for name, split in cands[1:]:
                gains_to_frozen = any(split.get(s, 0.0) - inc.get(s, 0.0) > 1e-9 for s in frozen)
                if not gains_to_frozen:
                    continue
                ok, viol, _ = constraint_report(L, part, plant, split, inc, disable=disable)
                if ok:
                    moved += 1
                    example = example or dict(part=part, plant=plant, candidate=name,
                                              suppliers=sorted(frozen), violations=viol)
                else:
                    blocked += 1
    return dict(part_plants_checked=checked, candidates_blocked=blocked, candidates_allowed_through=moved,
                passes=(moved == 0 and blocked > 0), example_leak=example)


def main(a):
    L = load(a.world, a.t0)
    sig = supplier_signals(a.world)
    cov = [(len(x["fill"]), len(x["strain"])) for x in sig]
    assert all(f > 0 and st > 0 for f, st in cov), f"signals failed to load: {cov} -- the band check would be vacuous"
    print(f"signals per seed (suppliers with fill / strain): {cov}")
    unit = L["costs"].groupby(["part_id", "supplier_id"]).unit_cost_inr.mean().to_dict()
    req = L["demand"].groupby(["part_id", "plant_id"]).gross_requirement_p50.sum()

    ch = L["channels"]
    per = ch.groupby(["part_id", "plant_id"]).supplier_id.nunique()
    multi = dict(part_plants=int(len(per)), with_multiple_suppliers=int((per > 1).sum()),
                 fraction=float((per > 1).mean()),
                 with_recorded_incumbent=int(L["alloc"].groupby(["part_id", "plant_id"]).ngroups),
                 fraction_with_incumbent=float(L["alloc"].groupby(["part_id", "plant_id"]).ngroups / len(per)))
    print(f"3.1 candidate scope, {a.world}: {multi['part_plants']:,} part-plants, "
          f"{multi['with_multiple_suppliers']:,} ({100*multi['fraction']:.1f}%) have more than one qualified supplier; "
          f"{multi['with_recorded_incumbent']:,} ({100*multi['fraction_with_incumbent']:.1f}%) have a recorded incumbent split")

    keys = [k for k in per[per > 1].index if k in req.index and req[k] > 0][:a.parts]
    rows = []
    for part, plant in keys:
        qual, cands = candidates(L, part, plant)
        R = float(req[(part, plant)])
        inc = dict(cands[0][1])
        scored = []
        for seed_sig in sig:
            sc = ReducedScorer(seed_sig["fill"], seed_sig["strain"], unit)
            for name, split in cands:
                ok, viol, inert = constraint_report(L, part, plant, split, inc)
                v = sc(L, part, plant, split, R)
                scored.append(dict(seed=seed_sig["seed"], candidate=name, feasible=ok, violations=viol,
                                   inert=inert, **v))
        df = pd.DataFrame(scored)
        agg = df.groupby("candidate").score.agg(["mean", "min", "max"]).sort_values("mean")
        feas = df.groupby("candidate").feasible.all()
        best = [c for c in agg.index if feas.get(c, False)]
        rank = agg.loc[best] if best else agg
        margin = float(rank["mean"].iloc[1] - rank["mean"].iloc[0]) if len(rank) > 1 else float("nan")
        bands_overlap = bool(len(rank) > 1 and rank["max"].iloc[0] >= rank["min"].iloc[1])
        rows.append(dict(part_id=part, plant_id=plant, requirement=R, n_qualified=len(qual),
                         ranking=[dict(candidate=c, score_mean=float(rank.loc[c, "mean"]),
                                       score_min=float(rank.loc[c, "min"]), score_max=float(rank.loc[c, "max"]))
                                  for c in rank.index],
                         infeasible=[c for c in agg.index if not feas.get(c, False)],
                         margin_first_to_second=margin, seed_bands_overlap=bands_overlap,
                         recommendation_supported=bool(not bands_overlap),
                         violations={c: df[df.candidate == c].violations.iloc[0] for c in agg.index},
                         inert=df.inert.iloc[0]))
        print(f"\n  {part} @ {plant}: requirement {R:.0f}, {len(qual)} qualified suppliers")
        for c in rank.index:
            print(f"    {c:58s} score {rank.loc[c,'mean']:12,.0f} [{rank.loc[c,'min']:,.0f}, {rank.loc[c,'max']:,.0f}]")
        if rows[-1]["infeasible"]:
            print(f"    infeasible under the constraint layer: {rows[-1]['infeasible']}")
        print(f"    first-to-second margin {margin:,.0f}; seed bands overlap = {bands_overlap} -> "
              f"{'NOT a recommendation' if bands_overlap else 'ranking survives the bands'}")

    real = verify_tooling_gate(L)
    broken = verify_tooling_gate(L, disable=("tooling",))
    print(f"\n3.4 tooling gate (guide): {real['part_plants_checked']} part-plants with frozen tooling checked, "
          f"{real['candidates_blocked']} candidate moves blocked, {real['candidates_allowed_through']} leaked "
          f"-> {'PASS' if real['passes'] else 'FAIL'}")
    print(f"    same gate against a constraint layer with the tooling check removed: "
          f"{broken['candidates_allowed_through']} leaked -> {'FAIL (as required)' if not broken['passes'] else 'still passes -- VACUOUS'}")
    print(f"    CAN IT FAIL: {'yes, shown' if real['passes'] and not broken['passes'] else 'NO -- vacuous'}")

    out = os.path.join(ARTIFACTS, f"phase10_allocation_{a.world}.json")
    json.dump(dict(world=a.world, t0=a.t0, scope=multi, client_parameters=CLIENT_PARAMS, rows=rows,
                   tooling_gate=dict(real=real, broken=broken,
                                     can_fail=bool(real["passes"] and not broken["passes"])),
                   caveat="ReducedScorer measures expected UNMET DEMAND IN PERIOD, not stockout against inventory; "
                          "its capacity term inherits interval miscalibration (80% coverage 0.72-0.81, below nominal "
                          "in 12 of 16 windows)"), open(out, "w"), indent=1, default=float)
    print(f"\n-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v6")
    ap.add_argument("--t0", default="2025-04-27")
    ap.add_argument("--parts", type=int, default=5)
    main(ap.parse_args())
