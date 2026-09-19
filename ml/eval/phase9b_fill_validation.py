"""Phase 9B Stage B — the fill ship decision, decided on VALIDATION.

Phase 9A compared fill against its baselines on the EVALUATION windows. Selecting a configuration on that would break
the project's oldest rule. This redoes the comparison on each origin's own VALIDATION fold, under the same
recalibration protocol the loop uses (fitted on that fold), reading score sets that already exist. No training, no
inference, no new fits.

  python ml/eval/phase9b_fill_validation.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from config import ARTIFACTS

BT = os.path.join(ARTIFACTS, "backtest")
T = "fill_rate"
SEEDS, B_SEEDS = (7, 17, 27), (7, 17, 27, 37, 47)
HEAD, LGBM, B5F, B2 = "none_h0_lr0.000125", "lgbm22_id", "b5flat22", "b2_rolling52_cdf"
WORLDS = ("v6", "v7")
ORIGINS = range(1, 9)
EVAL = {1: "2022 H1", 2: "2022 H2", 3: "2023 H1", 4: "2023 H2", 5: "2024 H1", 6: "2024 H2", 7: "2025 H1", 8: "2025 Q3"}


def band(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else dict(n=len(xs), mean=float(np.mean(xs)), spread=float(np.ptp(xs)),
                                    min=float(min(xs)), max=float(max(xs)))


def compare(a, b, lower_better=True):
    A, B = band(a), band(b)
    if not A or not B:
        return dict(verdict="missing")
    m = A["mean"] - B["mean"]
    if A["max"] < B["min"]:
        w = "head" if lower_better else "baseline"
    elif B["max"] < A["min"]:
        w = "baseline" if lower_better else "head"
    else:
        w = None
    return dict(margin=float(m), verdict="not distinguishable" if w is None else f"{w} better")


class D:
    def __init__(self, fold):
        self.S = json.load(open(os.path.join(BT, "phase8_scores.json")))["scores"]
        self.pre = "VAL_" if fold == "val" else ""

    def get(self, w, label, metric):
        v = self.S.get(f"{w}|{T}|{label}", {}).get(metric)
        return None if v is None else float(v[0])

    def arm(self, w, k, cfg, metric, recal=True):
        r = "RECAL_" if recal else ""
        if cfg == B2:
            return [x for x in [self.get(w, f"{self.pre}{r}o{k}_{cfg}", metric)] if x is not None]
        ss = SEEDS if cfg == HEAD else B_SEEDS
        return [x for x in (self.get(w, f"{self.pre}{r}o{k}_{cfg}_s{s}", metric) for s in ss) if x is not None]


def main(out_path):
    d = D("val")
    rows = []
    for w in WORLDS:
        for k in ORIGINS:
            h = d.arm(w, k, HEAD, "crps_exact")
            if not h:
                continue
            row = dict(world=w, origin=k, evaluates=EVAL[k])
            for cfg, lab in ((HEAD, "head"), (LGBM, "lgbm22_id"), (B5F, "b5flat22"), (B2, "b2")):
                for met in ("crps_exact", "ece22", "rel_one"):
                    row[f"{lab}_{met}"] = band(d.arm(w, k, cfg, met))
                row[f"{lab}_ece22_raw"] = band(d.arm(w, k, cfg, "ece22", recal=False))
            for cfg, lab in ((LGBM, "lgbm22_id"), (B5F, "b5flat22"), (B2, "b2")):
                row[f"vs_{lab}_crps"] = compare(d.arm(w, k, HEAD, "crps_exact"), d.arm(w, k, cfg, "crps_exact"))
                row[f"vs_{lab}_ece22"] = compare(d.arm(w, k, HEAD, "ece22"), d.arm(w, k, cfg, "ece22"))
                # B.3 -- the two magnitudes, on the baseline's own scale
                bc, be = band(d.arm(w, k, cfg, "crps_exact")), band(d.arm(w, k, cfg, "ece22"))
                hc, he = band(d.arm(w, k, HEAD, "crps_exact")), band(d.arm(w, k, HEAD, "ece22"))
                row[f"crps_gain_pct_{lab}"] = float(100.0 * (bc["mean"] - hc["mean"]) / bc["mean"]) if bc and hc else None
                row[f"ece_ratio_{lab}"] = float(he["mean"] / be["mean"]) if be and he and be["mean"] else None
            rows.append(row)

    def tally(key):
        v = [r[key]["verdict"] for r in rows]
        return {"head better": v.count("head better"), "not distinguishable": v.count("not distinguishable"),
                "baseline better": v.count("baseline better")}
    R = dict(fold="validation", rows=rows,
             tallies={k: tally(k) for k in ("vs_lgbm22_id_crps", "vs_b5flat22_crps", "vs_b2_crps",
                                            "vs_lgbm22_id_ece22", "vs_b5flat22_ece22", "vs_b2_ece22")},
             magnitudes={lab: dict(
                 crps_gain_pct=[min(r[f"crps_gain_pct_{lab}"] for r in rows), max(r[f"crps_gain_pct_{lab}"] for r in rows)],
                 ece_ratio=[min(r[f"ece_ratio_{lab}"] for r in rows), max(r[f"ece_ratio_{lab}"] for r in rows)])
                 for lab in ("lgbm22_id", "b5flat22", "b2")})
    json.dump(R, open(out_path, "w"), indent=1, default=float)

    f = lambda b: "—" if not b else f"{b['mean']:.4f} ({b['spread']:.4f})"
    print("VALIDATION FOLD, recalibration fitted on that fold (both arms)\n")
    print(f"{'w':3s} {'o':>2s} {'window':8s} {'head CRPS':>16s} {'LGBM22':>16s} {'B5flat22':>16s} {'B2':>9s} "
          f"{'head ECE22':>16s} {'LGBM22':>16s} {'B5flat22':>16s} {'B2':>9s} {'head relP1':>16s}")
    for r in rows:
        print(f"{r['world']:3s} {r['origin']:2d} {r['evaluates']:8s} {f(r['head_crps_exact']):>16s} {f(r['lgbm22_id_crps_exact']):>16s} "
              f"{f(r['b5flat22_crps_exact']):>16s} {r['b2_crps_exact']['mean']:9.4f} {f(r['head_ece22']):>16s} "
              f"{f(r['lgbm22_id_ece22']):>16s} {f(r['b5flat22_ece22']):>16s} {r['b2_ece22']['mean']:9.4f} {f(r['head_rel_one']):>16s}")
    print("\nverdicts, head vs baseline:")
    for k, v in R["tallies"].items():
        print(f"  {k:24s} {json.dumps(v)}")
    print("\nB.3 magnitudes (validation):")
    for lab, m in R["magnitudes"].items():
        print(f"  vs {lab:11s} CRPS gain {m['crps_gain_pct'][0]:+.2f}% to {m['crps_gain_pct'][1]:+.2f}%   "
              f"ECE ratio {m['ece_ratio'][0]:.2f}x to {m['ece_ratio'][1]:.2f}x")
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase9b_fill_validation.json"))
    main(ap.parse_args().out)
