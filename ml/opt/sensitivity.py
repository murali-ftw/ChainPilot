"""Phase 11 Stage 3 — is 10.2's ranking an artefact of the assumed shortage cost?

Phase 10 open item 4: `ReducedScorer`'s shortage cost is an ASSUMED Rs 1,000/unit, and 3 of the 6 part-plants
reported in Phase 10 section 5.4 already fail the seed-band check. This sweeps the cost across two orders of
magnitude and asks, per part-plant:

  * does the winning candidate change across the sweep, and over what range is each winner stable?
  * does the winner's margin over the runner-up survive the 3-seed bands at every cost in the sweep?

A recommendation is counted as QUOTABLE only if both hold. No training, no inference: the fill and capacity signals
are the same artifacts Phase 10 used.

  python ml/opt/sensitivity.py --world v6 --parts 60
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from config import ARTIFACTS
from allocation import load, candidates, constraint_report, ReducedScorer, supplier_signals

SWEEP = (100.0, 300.0, 1000.0, 3000.0, 10000.0)


def evaluate(L, sig, unit, part, plant, req, sweep=SWEEP):
    qual, cands = candidates(L, part, plant)
    if len(qual) < 2:
        return None
    inc = dict(cands[0][1])
    feasible = {}
    for name, split in cands:
        ok, viol, _ = constraint_report(L, part, plant, split, inc)
        feasible[name] = ok
    rows = []
    for cost in sweep:
        per_seed = {}
        for s in sig:
            sc = ReducedScorer(s["fill"], s["strain"], unit, shortage_cost_per_unit=cost)
            for name, split in cands:
                if not feasible[name]:
                    continue
                per_seed.setdefault(name, []).append(sc(L, part, plant, split, req)["score"])
        if not per_seed:
            continue
        agg = {k: dict(mean=float(np.mean(v)), lo=float(min(v)), hi=float(max(v))) for k, v in per_seed.items()}
        order = sorted(agg, key=lambda k: agg[k]["mean"])
        win = order[0]
        second = order[1] if len(order) > 1 else None
        margin = (agg[second]["mean"] - agg[win]["mean"]) if second else float("inf")
        overlap = bool(second and agg[win]["hi"] >= agg[second]["lo"])
        rows.append(dict(shortage_cost=cost, winner=win, runner_up=second, margin=margin,
                         bands_overlap=overlap, survives_bands=bool(second and not overlap)))
    if not rows:
        return None
    winners = [r["winner"] for r in rows]
    stable = len(set(winners)) == 1
    survives_everywhere = all(r["survives_bands"] for r in rows)
    ranges = {}
    for w in dict.fromkeys(winners):
        cs = [r["shortage_cost"] for r in rows if r["winner"] == w]
        ranges[w] = [min(cs), max(cs)]
    return dict(part_id=part, plant_id=plant, requirement=float(req), n_qualified=len(qual),
                n_feasible=int(sum(feasible.values())), sweep=rows, winners=winners,
                winner_stable_across_sweep=stable, survives_bands_at_every_cost=survives_everywhere,
                quotable=bool(stable and survives_everywhere), winner_ranges=ranges)


def main(a):
    L = load(a.world, a.t0)
    sig = supplier_signals(a.world)
    assert all(len(s["fill"]) and len(s["strain"]) for s in sig), "signals empty -- the band check would be vacuous"
    unit = L["costs"].groupby(["part_id", "supplier_id"]).unit_cost_inr.mean().to_dict()
    req = L["demand"].groupby(["part_id", "plant_id"]).gross_requirement_p50.sum()
    per = L["channels"].groupby(["part_id", "plant_id"]).supplier_id.nunique()

    keys = [k for k in per[per > 1].index if k in req.index and req[k] > 0]
    named = [k for k in keys if k[0] == "P00001"]                      # the Phase 10 section 5.4 rows
    rest = [k for k in keys if k not in named][:max(0, a.parts - len(named))]
    out = []
    for part, plant in named + rest:
        r = evaluate(L, sig, unit, part, plant, float(req[(part, plant)]))
        if r:
            out.append(r)

    quotable = [r for r in out if r["quotable"]]
    stable = [r for r in out if r["winner_stable_across_sweep"]]
    bands = [r for r in out if r["survives_bands_at_every_cost"]]
    S = dict(world=a.world, part_plants=len(out), sweep=list(SWEEP),
             winner_stable_across_sweep=len(stable), survives_bands_at_every_cost=len(bands),
             quotable=len(quotable),
             fraction_quotable=float(len(quotable) / len(out)) if out else float("nan"))

    print(f"world {a.world}: {len(out)} part-plants, shortage cost swept {SWEEP[0]:.0f} -> {SWEEP[-1]:.0f} Rs/unit\n")
    print(f"{'part-plant':22s} {'req':>6s} {'q':>2s} {'winner @100':28s} {'winner @10000':28s} {'stable':7s} {'bands':7s} {'quotable':8s}")
    for r in out[:a.show]:
        w_lo, w_hi = r["sweep"][0]["winner"], r["sweep"][-1]["winner"]
        print(f"{r['part_id'] + ' @ ' + r['plant_id']:22s} {r['requirement']:6.0f} {r['n_qualified']:2d} "
              f"{w_lo[:27]:28s} {w_hi[:27]:28s} {str(r['winner_stable_across_sweep']):7s} "
              f"{str(r['survives_bands_at_every_cost']):7s} {str(r['quotable']):8s}")
    print(f"\nwinner stable across the sweep:      {len(stable):3d} of {len(out)} ({100*len(stable)/max(1,len(out)):.0f}%)")
    print(f"margin survives the seed bands:      {len(bands):3d} of {len(out)} ({100*len(bands)/max(1,len(out)):.0f}%)")
    print(f"BOTH (a quotable recommendation):    {len(quotable):3d} of {len(out)} ({100*S['fraction_quotable']:.0f}%)")
    p = os.path.join(ARTIFACTS, f"phase11_sensitivity_{a.world}.json")
    json.dump(dict(summary=S, rows=out), open(p, "w"), indent=1, default=float)
    print(f"-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v6")
    ap.add_argument("--t0", default="2025-04-27")
    ap.add_argument("--parts", type=int, default=60)
    ap.add_argument("--show", type=int, default=12)
    main(ap.parse_args())
