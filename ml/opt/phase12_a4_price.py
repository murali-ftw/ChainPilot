"""Phase 12 Wave A4 -- Test 5.1, price_weight = 0, and the consequence the brief did not state.

With price_weight = 0 the ReducedScorer score is
    requirement * sum_v share_v * (1 - E[fill_v]) * strain_penalty_v * c_short
and c_short multiplies every candidate for a part-plant identically, so it cannot reorder them: the
ranking becomes INVARIANT TO THE SHORTAGE COST. This script PROVES that on the data rather than
asserting it, and measures what it unlocks.

  A4.2 REGRESSION   price_weight = 1.0 must reproduce the stored Phase 11 sweep
                    (ml/artifacts/phase11_sensitivity_{v6,v7}.json) EXACTLY -- winners, margins, overlaps.
                    Any difference is a DEFECT. (The brief asks weight 0 to be identical to current
                    output; that cannot hold -- current output includes the purchase term and flips with
                    c_short, which weight 0 provably cannot. Weight 0 vs current is reported as a finding.)
  A4.4 INVARIANCE   price_weight = 0, c_short in {100, 300, 1000, 3000, 10000}: the winner must be the same
                    at every cost for every part-plant. The check is shown CAPABLE OF FIRING on the weight-1
                    sweep, where winners are known to flip.
  A4.3 SMOKE        price_weight > 0 moves the winner toward cheaper candidates, monotonically.
  A4.5 UNLOCK       fraction of part-plants whose weight-0 winner survives the seed bands.

WORLDS ARE NOT MIXED. v6/v7 carry the regression (the only stored output) and the like-for-like
comparator (25%, 3 seeds, origin 7). v8 is the standing-rule world: 5 seeds, as-of signals from test
snapshots recorded on or before t0.
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "eval"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "models")]
import numpy as np, pandas as pd
import phase12_common as C
from config import WORLDS, ARTIFACTS
from allocation import load, candidates, constraint_report, ReducedScorer, supplier_signals, price_label
from sensitivity import evaluate, SWEEP

V6V7_T0 = "2025-04-27"          # Phase 11's t0, kept so the regression is like-for-like


def phase11_keys(L, a_parts=60):
    """Exactly sensitivity.main's selection, so the regression compares the same part-plants."""
    req = L["demand"].groupby(["part_id", "plant_id"]).gross_requirement_p50.sum()
    per = L["channels"].groupby(["part_id", "plant_id"]).supplier_id.nunique()
    keys = [k for k in per[per > 1].index if k in req.index and req[k] > 0]
    named = [k for k in keys if k[0] == "P00001"]
    rest = [k for k in keys if k not in named][:max(0, a_parts - len(named))]
    return named + rest, req


def sweep_world(L, sig, keys, req, price_weight):
    unit = L["costs"].groupby(["part_id", "supplier_id"]).unit_cost_inr.mean().to_dict()
    out = []
    for part, plant in keys:
        r = evaluate(L, sig, unit, part, plant, float(req[(part, plant)]), price_weight=price_weight)
        if r:
            out.append(r)
    return out


def invariance_check(rows):
    """Fires (returns violations) if any part-plant's winner changes across the shortage-cost sweep."""
    return [dict(part_id=r["part_id"], plant_id=r["plant_id"], winners=r["winners"])
            for r in rows if len(set(r["winners"])) != 1]


def summarise(rows):
    q = [r for r in rows if r["quotable"]]
    keep = [r for r in q if r["winners"][0] == "incumbent"]
    return dict(part_plants=len(rows), winner_stable=sum(r["winner_stable_across_sweep"] for r in rows),
                survives_bands_every_cost=sum(r["survives_bands_at_every_cost"] for r in rows),
                quotable=len(q), fraction_quotable=len(q) / len(rows) if rows else float("nan"),
                quotable_keep_incumbent=len(keep),
                actionable=len(q) - len(keep), fraction_actionable=(len(q) - len(keep)) / len(rows) if rows else float("nan"))


def regression(rows_new, stored_path):
    old = {(r["part_id"], r["plant_id"]): r for r in json.load(open(stored_path))["rows"]}
    diffs = []
    for r in rows_new:
        o = old.get((r["part_id"], r["plant_id"]))
        if o is None:
            diffs.append(dict(key=(r["part_id"], r["plant_id"]), what="missing in stored")); continue
        for a, b in zip(r["sweep"], o["sweep"]):
            for k in ("winner", "runner_up", "bands_overlap", "survives_bands"):
                if a[k] != b[k]:
                    diffs.append(dict(key=(r["part_id"], r["plant_id"]), cost=a["shortage_cost"], field=k, new=a[k], old=b[k]))
            if not np.isclose(a["margin"], b["margin"], rtol=0, atol=1e-6):
                diffs.append(dict(key=(r["part_id"], r["plant_id"]), cost=a["shortage_cost"], field="margin",
                                  new=a["margin"], old=b["margin"]))
    return dict(part_plants_compared=len(rows_new), stored=len(old), n_differences=len(diffs), differences=diffs[:20])


def smoke_price(L, sig, keys, req, weights=(0.0, 0.01, 0.1, 1.0, 10.0, 1e6)):
    """price_weight > 0 must move winners toward cheaper candidates: the winner's purchase cost should be
    non-increasing in the weight, and at a huge weight the winner should be the cheapest feasible candidate."""
    unit = L["costs"].groupby(["part_id", "supplier_id"]).unit_cost_inr.mean().to_dict()
    res, non_monotone, cheapest_at_max = [], 0, 0
    n = 0
    for part, plant in keys:
        qual, cands = candidates(L, part, plant)
        if len(qual) < 2:
            continue
        inc = dict(cands[0][1])
        feas = {nm: constraint_report(L, part, plant, sp, inc)[0] for nm, sp in cands}
        R = float(req[(part, plant)])
        purch = {}
        trail = []
        for w in weights:
            sc = [ReducedScorer(s["fill"], s["strain"], unit, 1000.0, price_weight=w) for s in sig]
            mean = {nm: np.mean([x(L, part, plant, sp, R)["score"] for x in sc]) for nm, sp in cands if feas[nm]}
            if not mean:
                break
            win = min(mean, key=mean.get)
            purch[win] = np.mean([x(L, part, plant, dict(cands)[win], R)["purchase_cost"] for x in sc])
            trail.append((w, win, float(purch[win])))
        if len(trail) != len(weights):
            continue
        n += 1
        pc = [t[2] for t in trail]
        if any(pc[i + 1] > pc[i] + 1e-6 for i in range(len(pc) - 1)):
            non_monotone += 1
        allp = {nm: np.mean([x(L, part, plant, sp, R)["purchase_cost"] for x in sc]) for nm, sp in cands if feas[nm]}
        if abs(pc[-1] - min(allp.values())) < 1e-6:
            cheapest_at_max += 1
        if len(res) < 6:
            res.append(dict(part_id=part, plant_id=plant, trail=trail))
    return dict(part_plants=n, winner_purchase_non_increasing=n - non_monotone, violations=non_monotone,
                cheapest_feasible_at_weight_1e6=cheapest_at_max, examples=res,
                winner_changes_with_weight=sum(1 for e in res if len({t[1] for t in e["trail"]}) > 1))


# ---------------------------------------------------------------- v8 signals, as-of t0, five seeds
def v8_signals(t0, seeds=C.V8_SEEDS):
    """fill[s]: mean recalibrated P(complete) over the supplier's lines (shipped b5flat22, 5 fits);
    strain[s]: mean capacity-strain P90 over its channels (shipped mp h4, 5 seeds).
    ONLY predictions made at snapshots on or before t0 are used -- the v6/v7 path averaged a whole
    evaluation window that straddles its t0; that is recorded, not copied."""
    import phase5_heads as P5, folds
    D = WORLDS["v8"]
    snaps = pd.read_csv(f"{D}/snapshots.csv", usecols=["snapshot_id", "as_of_ts"])
    sdate = dict(zip(snaps.snapshot_id, pd.to_datetime(snaps.as_of_ts)))
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "supplier_id"])
    ch_sup = dict(zip(ch.channel_id, ch.supplier_id))
    pol = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "channel_id"])
    pol_sup = {p: ch_sup.get(c) for p, c in zip(pol.po_line_id, pol.channel_id)}
    lbf = P5.labels("v8", "fill_rate")
    _, _, te = folds.fixed_split(lbf.snapshot_date)
    o = P5.ordered(lbf, te)
    f_ent, f_date = lbf.entity_id.values[o], pd.to_datetime(lbf.snapshot_date.values[o])
    t0 = pd.Timestamp(t0)
    out = []
    for s in seeds:
        z = np.load(f"{ARTIFACTS}/phase7_preds/RECAL_v8_fill_rate_b5flat22_s{s}_test.npz")
        assert (z["entity"].astype(str) == f_ent.astype(str)).all(), "fill rows do not align to labels"
        m = f_date <= t0
        d = pd.DataFrame(dict(sup=[pol_sup.get(e) for e in f_ent[m]], pc=z["P"][m, 21])).dropna()
        fill = d.groupby("sup").pc.mean().to_dict()
        mo = pd.read_csv(f"{C.BUND}/capacity_strain/v8_mp_h4_lr0.00025_s{s}/model_outputs.csv.gz",
                         usecols=["snapshot_id", "entity_id", "p90"])
        mo["date"] = mo.snapshot_id.map(sdate)
        mo = mo[mo.date <= t0]
        d = pd.DataFrame(dict(sup=mo.entity_id.map(ch_sup), q90=mo.p90)).dropna()
        strain = d.groupby("sup").q90.mean().to_dict()
        out.append(dict(seed=s, fill=fill, strain=strain, n_fill_rows=int(m.sum()), n_strain_rows=int(len(mo)),
                        snapshots_used=sorted(str(x.date()) for x in mo.date.unique())))
    return out


def main(a):
    st = C.require_clean()
    res = dict(stamp=st, sweep=list(SWEEP))
    for w in ("v6", "v7"):
        L = load(w, V6V7_T0)
        sig = supplier_signals(w)
        keys, req = phase11_keys(L)
        r1 = sweep_world(L, sig, keys, req, 1.0)
        r0 = sweep_world(L, sig, keys, req, 0.0)
        reg = regression(r1, os.path.join(ARTIFACTS, f"phase11_sensitivity_{w}.json"))
        fire = invariance_check(r1)                  # MUST fire: weight 1 is known to flip winners
        inv = invariance_check(r0)                   # must NOT fire
        cur = {(r["part_id"], r["plant_id"]): r["sweep"][2]["winner"] for r in r1}      # c_short = 1000
        changed = sum(1 for r in r0 if r["winners"][0] != cur[(r["part_id"], r["plant_id"])])
        res[w] = dict(t0=V6V7_T0, seeds=[s["seed"] for s in sig], regression_pw1=reg,
                      invariance_check_fires_on_pw1=dict(n_flagged=len(fire), capable_of_failing=len(fire) > 0),
                      invariance_pw0=dict(n_flagged=len(inv), holds=len(inv) == 0, part_plants=len(r0)),
                      summary_pw1=summarise(r1), summary_pw0=summarise(r0),
                      pw0_winner_differs_from_current_at_1000=changed,
                      smoke=smoke_price(L, sig, keys[:a.smoke_parts], req),
                      rows_pw0=r0, decision_basis=price_label(0.0))
        print(w, "regression diffs", reg["n_differences"], "| pw1 flagged", len(fire), "| pw0 flagged", len(inv),
              "| pw1", res[w]["summary_pw1"], "| pw0", res[w]["summary_pw0"], flush=True)

    L = load("v8", a.v8_t0)
    sig = v8_signals(a.v8_t0)
    assert all(len(s["fill"]) and len(s["strain"]) for s in sig), "v8 signals empty -- band check would be vacuous"
    keys, req = phase11_keys(L, a.v8_parts)
    r1 = sweep_world(L, sig, keys, req, 1.0)
    r0 = sweep_world(L, sig, keys, req, 0.0)
    fire, inv = invariance_check(r1), invariance_check(r0)
    cur = {(r["part_id"], r["plant_id"]): r["sweep"][2]["winner"] for r in r1}
    res["v8"] = dict(t0=a.v8_t0, seeds=list(C.V8_SEEDS),
                     signals=[{k: v for k, v in s.items() if k not in ("fill", "strain")} |
                              dict(n_suppliers_fill=len(s["fill"]), n_suppliers_strain=len(s["strain"])) for s in sig],
                     invariance_check_fires_on_pw1=dict(n_flagged=len(fire), capable_of_failing=len(fire) > 0),
                     invariance_pw0=dict(n_flagged=len(inv), holds=len(inv) == 0, part_plants=len(r0)),
                     summary_pw1=summarise(r1), summary_pw0=summarise(r0),
                     pw0_winner_differs_from_current_at_1000=sum(
                         1 for r in r0 if r["winners"][0] != cur[(r["part_id"], r["plant_id"])]),
                     smoke=smoke_price(L, sig, keys[:a.smoke_parts], req),
                     rows_pw0=r0, decision_basis=price_label(0.0))
    print("v8 | pw1 flagged", len(fire), "| pw0 flagged", len(inv), "| pw1", res["v8"]["summary_pw1"],
          "| pw0", res["v8"]["summary_pw0"], flush=True)
    res["caveat"] = ("ReducedScorer measures expected UNMET DEMAND IN PERIOD, not stockout; its strain_penalty "
                     "consumes a capacity P90 whose intervals are NOT quotable (80% coverage 0.72-0.81). "
                     "Weight-0 output: " + price_label(0.0))
    print(C.dump(res, "phase12_a4.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-t0", default="2025-06-30")
    ap.add_argument("--v8-parts", type=int, default=120)
    ap.add_argument("--smoke-parts", type=int, default=30)
    main(ap.parse_args())
