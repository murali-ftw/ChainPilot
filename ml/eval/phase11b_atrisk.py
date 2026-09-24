"""Phase 11B Stage B.1 — identify the at-risk cells BEFORE training anything.

Deviation 67: 3-seed bands are optimistic by up to 5.6x, unpredictably, and widening to five
flipped two verdicts in 11A — both the narrowest margins in their table. Phase 8's entire
rolling-origin backtest was three seeds, and its headline counts were decided by 3-seed
disjointness.

A cell is AT RISK when the margin between the two arms is small relative to the seed spread that
produced it. The criterion, from the brief:

    margin  <  2 x (the 3-seed spread)

where margin = |mean_a - mean_b| and the spread is the larger of the two arms' 3-seed spreads.
A cell already reported "not dist." is at risk in the other direction: five seeds could separate
it, or could confirm it. Both are reported.

This reads recorded artifacts only and trains nothing.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ART = "ml/artifacts/backtest"
HIGHER = {"roc_auc_late", "cindex", "coverage80"}


def _pair(rec):
    """-> (mean_a, spread_a, mean_b, spread_b) from a Phase 8 comparison record."""
    a, b = rec.get("a"), rec.get("b")
    if not a or not b:
        return None
    return (a["mean"], a.get("spread", a["max"] - a["min"]),
            b["mean"], b.get("spread", b["max"] - b["min"]),
            a.get("min"), a.get("max"), b.get("min"), b.get("max"))


def classify(rec, lower_is_better=True):
    p = _pair(rec)
    if p is None:
        return None
    ma, sa, mb, sb, amin, amax, bmin, bmax = p
    margin = abs(ma - mb)
    spread = max(sa, sb)
    disj = (amax < bmin) or (bmax < amin)
    ratio = margin / spread if spread > 0 else float("inf")
    return dict(mean_a=ma, mean_b=mb, spread_a=sa, spread_b=sb, margin=margin,
                spread=spread, margin_over_spread=ratio, disjoint_at_3=bool(disj),
                verdict_at_3=("A" if (ma < mb) == lower_is_better else "B") if disj
                              else "not dist.",
                AT_RISK=bool(ratio < 2.0))


def main():
    out = {"capacity": [], "arrival": [], "fill": []}
    cap = json.load(open(os.path.join(ART, "phase8b_capacity.json")))
    for r in cap["rows"]:
        base = dict(world=r["world"], origin=r["origin"])
        for comp, lower in (("h4_vs_h0", True), ("h4_vs_b5", True)):
            if comp not in r:
                continue
            c = classify(r[comp], lower)
            if c is None:
                continue
            out["capacity"].append({**base, "comparison": comp, "task": "capacity_strain", **c})

    # arrival, Phase 8E. Its shape differs: `head_roc` is a 3-seed band and `promise_roc` is a
    # scalar with its own bootstrap CI, so the comparison is a band against a point. The margin
    # is therefore divided by the HEAD's spread alone -- which is what Phase 8 itself recorded as
    # `head_minus_promise_over_head_spread`.
    ap = os.path.join(ART, "phase8e_arrival.json")
    if os.path.exists(ap):
        ar = json.load(open(ap))
        for r in ar.get("rows", []):
            h = r.get("head_roc")
            if not h:
                continue
            for other, lbl in ((r.get("promise_roc"), "head_vs_promise_lateness"),
                               (r.get("b5_roc"), "head_vs_b5_lateness"),
                               (r.get("h0_roc"), "head_vs_h0_lateness")):
                if other is None:
                    continue
                om = other["mean"] if isinstance(other, dict) else float(other)
                osp = other.get("spread", 0.0) if isinstance(other, dict) else 0.0
                omin = other.get("min", om) if isinstance(other, dict) else om
                omax = other.get("max", om) if isinstance(other, dict) else om
                margin = abs(h["mean"] - om)
                spread = max(h["spread"], osp)
                disj = (h["max"] < omin) or (omax < h["min"])
                ratio = margin / spread if spread > 0 else float("inf")
                out["arrival"].append(dict(
                    world=r["world"], origin=r["origin"], task="arrival_week", comparison=lbl,
                    mean_a=h["mean"], mean_b=om, spread_a=h["spread"], spread_b=osp,
                    margin=margin, spread=spread, margin_over_spread=ratio,
                    disjoint_at_3=bool(disj),
                    verdict_at_3=("A" if h["mean"] > om else "B") if disj else "not dist.",
                    AT_RISK=bool(ratio < 2.0)))

    # fill, Phase 9A
    fp = os.path.join(ART, "phase9a_fill.json")
    if os.path.exists(fp):
        fr = json.load(open(fp))
        rows = fr.get("rows", [])
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict):
                continue
            for comp in [k for k in r if isinstance(r.get(k), dict) and "a" in r[k]]:
                c = classify(r[comp], lower_is_better=True)
                if c is None:
                    continue
                out["fill"].append({"world": r.get("world"), "origin": r.get("origin"),
                                    "comparison": comp, "task": "fill_rate", **c})

    allrows = out["capacity"] + out["arrival"] + out["fill"]
    at_risk = sorted([r for r in allrows if r["AT_RISK"]], key=lambda r: r["margin_over_spread"])
    print(f"comparisons examined: {len(allrows)}  "
          f"(capacity {len(out['capacity'])}, arrival {len(out['arrival'])}, fill {len(out['fill'])})")
    print(f"AT RISK (margin < 2x spread): {len(at_risk)}\n")
    print(f"{'task':16s} {'world':5s} {'org':3s} {'comparison':12s} {'margin':>9s} "
          f"{'spread':>9s} {'m/s':>6s}  {'verdict@3':10s}")
    for r in at_risk:
        print(f"{r['task']:16s} {str(r['world']):5s} {str(r['origin']):3s} {r['comparison']:12s} "
              f"{r['margin']:9.5f} {r['spread']:9.5f} {r['margin_over_spread']:6.2f}  "
              f"{r['verdict_at_3']:10s}")
    json.dump(dict(all=allrows, at_risk=at_risk), open(sys.argv[1], "w"), indent=1, default=str)
    print(f"\njson -> {sys.argv[1]}")
    # the cells to re-band: unique (task, world, origin)
    cells = sorted({(r["task"], r["world"], r["origin"]) for r in at_risk})
    print(f"\nDISTINCT CELLS to re-band: {len(cells)}")
    for c in cells:
        print("   ", c)


if __name__ == "__main__":
    main()
