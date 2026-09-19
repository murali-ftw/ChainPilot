"""Phase 9A Stage A — fill's rolling-origin backtest: is its calibration stable across windows?

Reads finished artifacts only; nothing is re-scored here.
  ml/artifacts/backtest/phase8_scores.json          ml/eval/phase7_score.py --backtest
  ml/artifacts/backtest/bundle_index.json           ml/eval/backtest.py export (recalibration method and temperature)
  ml/artifacts/backtest/phase9a_baseline_recal.json the baselines' own recalibration, fitted per origin

Per origin x world, 3-seed bands for the head and 5-fit bands for each LightGBM arm:
  A.1  exact CRPS (the point-mass-aware integral, never crps_legacy), ECE-22 raw and recalibrated,
       conditional reliability of P(f = 1), recalibration method and temperature per seed
  A.2  the temperature's range across all 8 origins
  A.3  LightGBM-22 (identity codes) under the same recalibration protocol -- Phase 7 binding statement 2
  A.4  the model-free B2 as-of histogram, likewise

  python ml/eval/phase9a_fill.py
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
HEAD = "none_h0_lr0.000125"
LGBM, B5F, B2 = "lgbm22_id", "b5flat22", "b2_rolling52_cdf"
WORLDS = ("v6", "v7")
EVAL = {1: "2022 H1", 2: "2022 H2", 3: "2023 H1", 4: "2023 H2", 5: "2024 H1", 6: "2024 H2", 7: "2025 H1", 8: "2025 Q3"}


def band(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else dict(n=len(xs), mean=float(np.mean(xs)), spread=float(np.ptp(xs)),
                                    min=float(min(xs)), max=float(max(xs)), values=[float(x) for x in xs])


def compare(a, b, lower_better=True):
    """Disjoint seed ranges or nothing -- the project's rule, applied to every fill comparison here."""
    A, B = band(a), band(b)
    if not A or not B:
        return dict(verdict="missing")
    margin = A["mean"] - B["mean"]
    if A["max"] < B["min"]:
        w = "a" if lower_better else "b"
    elif B["max"] < A["min"]:
        w = "b" if lower_better else "a"
    else:
        w = None
    return dict(margin=float(margin), a=A, b=B, verdict="not distinguishable" if w is None else f"{w} better")


class D:
    def __init__(self):
        self.S = json.load(open(os.path.join(BT, "phase8_scores.json")))["scores"]
        self.IX = json.load(open(os.path.join(BT, "bundle_index.json")))
        p = os.path.join(BT, "phase9a_baseline_recal.json")
        self.BR = json.load(open(p)) if os.path.exists(p) else {}
        plan = json.load(open(os.path.join(BT, "phase8_origins.json")))
        ok = {}
        for r in plan["origins"]:
            ok[r["origin"]] = ok.get(r["origin"], True) and r["survives"]
        self.origins = [k for k in sorted(ok) if ok[k]]

    def get(self, w, label, metric):
        v = self.S.get(f"{w}|{T}|{label}", {}).get(metric)
        return None if v is None else float(v[0])

    def seeds(self, w, k, cfg, metric, ss=SEEDS, recal=False):
        pre = "RECAL_" if recal else ""
        return [x for x in (self.get(w, f"{pre}o{k}_{cfg}_s{s}", metric) for s in ss) if x is not None]

    def single(self, w, k, cfg, metric, recal=False):
        return [x for x in [self.get(w, f"{'RECAL_' if recal else ''}o{k}_{cfg}", metric)] if x is not None]

    def arm(self, w, k, cfg, metric, recal=False):
        return self.single(w, k, cfg, metric, recal) if cfg == B2 else self.seeds(w, k, cfg, metric, B_SEEDS, recal)

    def recal_of(self, w, k, s):
        e = self.IX.get(f"{w}|{T}|o{k}_{HEAD}_s{s}")
        return None if e is None else e["recalibration"]


def main(out_path):
    d = D()
    rows, temps = [], []
    for w in WORLDS:
        for k in d.origins:
            head_recal = [d.recal_of(w, k, s) for s in SEEDS]
            if not any(head_recal) or not d.seeds(w, k, HEAD, "ece22", recal=True):
                continue
            methods = [r["method"] for r in head_recal if r]
            Ts = [r["vs_T"] for r in head_recal if r]
            for s, r in zip(SEEDS, head_recal):
                if r:
                    temps.append(dict(world=w, origin=k, seed=s, method=r["method"], vs_T=r["vs_T"]))
            row = dict(world=w, origin=k, evaluates=EVAL[k],
                       recal_methods=methods, vs_T=Ts, vs_T_band=band(Ts),
                       crps_exact=band(d.seeds(w, k, HEAD, "crps_exact", recal=True)),
                       crps_exact_raw=band(d.seeds(w, k, HEAD, "crps_exact")),
                       ece22_raw=band(d.seeds(w, k, HEAD, "ece22")),
                       ece22_recal=band(d.seeds(w, k, HEAD, "ece22", recal=True)),
                       rel_one_raw=band(d.seeds(w, k, HEAD, "rel_one")),
                       rel_one_recal=band(d.seeds(w, k, HEAD, "rel_one", recal=True)),
                       p_complete_pred=band(d.seeds(w, k, HEAD, "p_complete_pred", recal=True)),
                       p_complete_obs=band(d.seeds(w, k, HEAD, "p_complete_obs", recal=True)))
            for cfg, lab in ((LGBM, "lgbm22_id"), (B5F, "b5flat22"), (B2, "b2")):
                row[f"{lab}_ece22_recal"] = band(d.arm(w, k, cfg, "ece22", recal=True))
                row[f"{lab}_ece22_raw"] = band(d.arm(w, k, cfg, "ece22"))
                row[f"{lab}_crps_exact"] = band(d.arm(w, k, cfg, "crps_exact", recal=True))
                row[f"vs_{lab}_ece22"] = compare(d.seeds(w, k, HEAD, "ece22", recal=True), d.arm(w, k, cfg, "ece22", recal=True))
                row[f"vs_{lab}_crps"] = compare(d.seeds(w, k, HEAD, "crps_exact", recal=True), d.arm(w, k, cfg, "crps_exact", recal=True))
            rows.append(row)

    def tally(key):
        v = [r[key]["verdict"] for r in rows]
        return {"head better": v.count("a better"), "not distinguishable": v.count("not distinguishable"),
                "baseline better": v.count("b better")}
    allT = [t["vs_T"] for t in temps]
    R = dict(rows=rows, temperatures=temps,
             temperature_range=dict(min=min(allT), max=max(allT), n=len(allT),
                                    per_world={w: dict(min=min(t["vs_T"] for t in temps if t["world"] == w),
                                                       max=max(t["vs_T"] for t in temps if t["world"] == w)) for w in WORLDS},
                                    methods=sorted({t["method"] for t in temps})),
             tallies={k: tally(k) for k in ("vs_lgbm22_id_ece22", "vs_b5flat22_ece22", "vs_b2_ece22",
                                            "vs_lgbm22_id_crps", "vs_b5flat22_crps", "vs_b2_crps")},
             baseline_recalibration_files=len(d.BR))
    json.dump(R, open(out_path, "w"), indent=1, default=float)

    f = lambda b: "—" if not b else f"{b['mean']:.4f} ({b['spread']:.4f})"
    print(f"{'w':3s} {'o':>2s} {'window':8s} {'CRPS exact':>16s} {'ECE22 raw':>16s} {'ECE22 recal':>16s} {'rel P(f=1)':>16s} {'method':7s} {'T per seed':22s}")
    for r in rows:
        print(f"{r['world']:3s} {r['origin']:2d} {r['evaluates']:8s} {f(r['crps_exact']):>16s} {f(r['ece22_raw']):>16s} "
              f"{f(r['ece22_recal']):>16s} {f(r['rel_one_recal']):>16s} {'/'.join(sorted(set(r['recal_methods']))):7s} "
              f"{' '.join(f'{t:.3f}' for t in r['vs_T']):22s}")
    print(f"\ntemperature across all origins: {R['temperature_range']['min']:.3f}-{R['temperature_range']['max']:.3f} "
          f"(n={R['temperature_range']['n']}, methods {R['temperature_range']['methods']})")
    for w in WORLDS:
        pw = R['temperature_range']['per_world'][w]
        print(f"  {w}: {pw['min']:.3f}-{pw['max']:.3f}")
    print(f"\n{'w':3s} {'o':>2s} {'head ECE22':>16s} {'LGBM22-id':>16s} {'B5flat22':>16s} {'B2 as-of':>16s} "
          f"{'vs LGBM22':22s} {'vs B5flat':22s} {'vs B2':22s}")
    for r in rows:
        print(f"{r['world']:3s} {r['origin']:2d} {f(r['ece22_recal']):>16s} {f(r['lgbm22_id_ece22_recal']):>16s} "
              f"{f(r['b5flat22_ece22_recal']):>16s} {f(r['b2_ece22_recal']):>16s} "
              f"{r['vs_lgbm22_id_ece22']['verdict']:22s} {r['vs_b5flat22_ece22']['verdict']:22s} {r['vs_b2_ece22']['verdict']:22s}")
    print("\ntallies (head vs baseline):", json.dumps(R["tallies"], indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase9a_fill.json"))
    main(ap.parse_args().out)
