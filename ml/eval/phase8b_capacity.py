"""Phase 8B — capacity depth across the rolling origins, stratified by the SIGN of the label shift.

Stage 3a asked whether h4 beats h0 outside the band. Phase 7 (2025, utilisation rising) and origin 1 (2022 H1,
utilisation falling) disagreed, so the headline here is never a median across all origins: a sign-dependent effect
averages to nothing. Every table below is split by the direction of the utilisation shift, which was recorded in
`phase8b_label_shift.json` before any 8B score existed.

Reads only finished artifacts:
  ml/artifacts/backtest/phase8_scores.json        the single scorer (test AND validation folds)
  ml/artifacts/backtest/bundle_index.json         drift baseline vs test drift, per bundle
  ml/artifacts/backtest/phase8b_label_shift.json  the stratifying variable
  ml/artifacts/backtest/phase8_origins.json       surviving origins

  python ml/eval/phase8b_capacity.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from config import ARTIFACTS

BT = os.path.join(ARTIFACTS, "backtest")
SEEDS = (7, 17, 27)
B5_SEEDS = (7, 17, 27, 37, 47)
H4, H0, B5 = "mp_h4_lr0.00025", "none_h0_lr0.00025", "b5flat_q"
TASK = "capacity_strain"
WORLDS = ("v6", "v7")


def band(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), mean=float(np.mean(xs)), spread=float(np.ptp(xs)), min=float(min(xs)), max=float(max(xs)))


def compare(a, b, lower_better=True):
    """A difference only when the seed ranges are disjoint AND both sides have a measured band."""
    A, B = band(a), band(b)
    if not A or not B:
        return dict(verdict="missing", a=A, b=B)
    if A["n"] < 2 or B["n"] < 2:
        return dict(verdict="band not measured", a=A, b=B, margin=A["mean"] - B["mean"])
    margin = A["mean"] - B["mean"]
    if A["max"] < B["min"]:
        better = "a" if lower_better else "b"
    elif B["max"] < A["min"]:
        better = "b" if lower_better else "a"
    else:
        better = None
    r = lambda sp: (abs(margin) / sp) if sp > 0 else None
    return dict(a=A, b=B, margin=margin, margin_over_a_spread=r(A["spread"]), margin_over_b_spread=r(B["spread"]),
                verdict="not distinguishable" if better is None else f"{better} better")


class D:
    def __init__(self):
        self.S = json.load(open(os.path.join(BT, "phase8_scores.json")))["scores"]
        self.IX = json.load(open(os.path.join(BT, "bundle_index.json")))
        self.shift = json.load(open(os.path.join(BT, "phase8b_label_shift.json")))
        plan = json.load(open(os.path.join(BT, "phase8_origins.json")))
        ok = {}
        for r in plan["origins"]:
            ok[r["origin"]] = ok.get(r["origin"], True) and r["survives"]
        self.origins = [k for k in sorted(ok) if ok[k]]

    def seeds(self, w, k, cfg, metric, ss=SEEDS, fold="test"):
        pre = "VAL_" if fold == "val" else ""
        out = []
        for s in ss:
            v = self.S.get(f"{w}|{TASK}|{pre}o{k}_{cfg}_s{s}", {}).get(metric)
            if v is not None:
                out.append(float(v[0]))
        return out

    def has(self, w, k):
        return bool(self.seeds(w, k, H4, "pinball_mean")) and bool(self.seeds(w, k, H0, "pinball_mean"))

    def label_shift(self, w, k, key="train_to_test"):
        return 100.0 * self.shift[f"{w}|{TASK}|o{k}"][key]

    def q50_drift(self, w, k, cfg, s):
        e = self.IX.get(f"{w}|{TASK}|o{k}_{cfg}_s{s}")
        return None if e is None else 100.0 * (e["drift_test"]["q50_mean"] - e["drift_val"]["q50_mean"])


def main(out_path):
    d = D()
    rows, drift_rows = [], []
    for w in WORLDS:
        for k in d.origins:
            if not d.has(w, k):
                continue
            shift_tt, shift_vt = d.label_shift(w, k), d.label_shift(w, k, "validation_to_test")
            row = dict(world=w, origin=k, label_shift_train_to_test=shift_tt, label_shift_validation_to_test=shift_vt,
                       regime="rising" if shift_tt > 0 else "falling",
                       h4_vs_h0=compare(d.seeds(w, k, H4, "pinball_mean"), d.seeds(w, k, H0, "pinball_mean")),
                       h4_vs_h0_validation=compare(d.seeds(w, k, H4, "pinball_mean", fold="val"),
                                                   d.seeds(w, k, H0, "pinball_mean", fold="val")),
                       h4_vs_b5=compare(d.seeds(w, k, H4, "pinball_mean"), d.seeds(w, k, B5, "pinball_mean", ss=B5_SEEDS)),
                       h0_vs_b5=compare(d.seeds(w, k, H0, "pinball_mean"), d.seeds(w, k, B5, "pinball_mean", ss=B5_SEEDS)),
                       exceed_p90={c: band(d.seeds(w, k, c, "exceed_p90", ss=(B5_SEEDS if c == B5 else SEEDS))) for c in (H4, H0, B5)},
                       below_p10={c: band(d.seeds(w, k, c, "below_p10", ss=(B5_SEEDS if c == B5 else SEEDS))) for c in (H4, H0, B5)},
                       coverage80={c: band(d.seeds(w, k, c, "coverage80", ss=(B5_SEEDS if c == B5 else SEEDS))) for c in (H4, H0, B5)},
                       pinball={c: band(d.seeds(w, k, c, "pinball_mean", ss=(B5_SEEDS if c == B5 else SEEDS))) for c in (H4, H0, B5)})
            rows.append(row)
            for s in SEEDS:                                    # 8B.3 -- the mechanism, per seed
                for cfg, lab in ((H4, "shipped h4"), (H0, "h0")):
                    dr = d.q50_drift(w, k, cfg, s)
                    if dr is None:
                        continue
                    drift_rows.append(dict(world=w, origin=k, config=lab, seed=s, q50_drift_pp=dr,
                                           label_shift_validation_to_test=shift_vt, regime=row["regime"],
                                           direction="with-shift" if np.sign(dr) == np.sign(shift_vt) else "against-shift"))
    def tally(sel, key="h4_vs_h0"):
        v = [r[key]["verdict"] for r in rows if sel(r)]
        return {x: v.count(x) for x in ("a better", "not distinguishable", "b better", "band not measured", "missing") if v.count(x)}
    strat = {
        "all_origins": tally(lambda r: True),
        "rising_only": tally(lambda r: r["regime"] == "rising"),
        "falling_only": tally(lambda r: r["regime"] == "falling"),
        "rising_validation": tally(lambda r: r["regime"] == "rising", "h4_vs_h0_validation"),
        "falling_validation": tally(lambda r: r["regime"] == "falling", "h4_vs_h0_validation"),
        "vs_b5_rising": tally(lambda r: r["regime"] == "rising", "h4_vs_b5"),
        "vs_b5_falling": tally(lambda r: r["regime"] == "falling", "h4_vs_b5"),
    }
    mech = {}
    for reg in ("rising", "falling"):
        for cfg in ("shipped h4", "h0"):
            g = [r for r in drift_rows if r["regime"] == reg and r["config"] == cfg]
            mech[f"{cfg}|{reg}"] = dict(n=len(g), against=sum(r["direction"] == "against-shift" for r in g),
                                        with_shift=sum(r["direction"] == "with-shift" for r in g))
    R = dict(rows=rows, drift_rows=drift_rows, stratified=strat, mechanism=mech,
             note="h4_vs_h0: 'a better' = the shipped depth-4 head wins outside both bands; 'b better' = h0 wins.")
    json.dump(R, open(out_path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    f = lambda b: "—" if not b else f"{b['mean']:.4f} (sp {b['spread']:.4f})"
    print(f"{'world':5s} {'o':>2s} {'shift tt':>9s} {'regime':8s} {'h4':>18s} {'h0':>18s} {'B5':>18s}  {'h4 vs h0':22s} {'h4 vs h0 (val)':22s} {'h4 vs B5':22s}")
    for r in rows:
        print(f"{r['world']:5s} {r['origin']:2d} {r['label_shift_train_to_test']:+9.2f} {r['regime']:8s} "
              f"{f(r['pinball'][H4]):>18s} {f(r['pinball'][H0]):>18s} {f(r['pinball'][B5]):>18s}  "
              f"{r['h4_vs_h0']['verdict']:22s} {r['h4_vs_h0_validation']['verdict']:22s} {r['h4_vs_b5']['verdict']:22s}")
    print("\nstratified:", json.dumps(strat, indent=1))
    print("mechanism (8B.3):", json.dumps(mech, indent=1))
    print("\nexceed_p90 (nominal 0.10):")
    for r in rows:
        print(f"  {r['world']} o{r['origin']:d} {r['regime']:8s} h4 {f(r['exceed_p90'][H4])}  h0 {f(r['exceed_p90'][H0])}  B5 {f(r['exceed_p90'][B5])}  "
              f"| coverage80 h4 {f(r['coverage80'][H4])}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase8b_capacity.json"))
    main(ap.parse_args().out)
