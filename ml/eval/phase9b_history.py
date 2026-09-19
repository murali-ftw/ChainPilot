"""Phase 9A Stage B — the arrival training-size confound.

Phase 8 read arrival's lateness margin as calendar-driven. Later origins also have more training history, so year and
history length are confounded by design. Stage B breaks the confound directly: arrival h4 retrained at ORIGIN 7 with
training truncated to origin 1's 18 snapshots, validation and evaluation windows unchanged.

  B.2  lateness-beyond-promise ROC-AUC for the truncated cell against the same promise-date baseline
  B.3  margin against training-snapshot count and against calendar year across the 8 existing cells -- reported, and
       reported as non-evidence: the 8 points are confounded, which is why B.1 was run

Reads finished artifacts only.
  python ml/eval/phase9b_history.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from scipy import stats
from config import ARTIFACTS

BT = os.path.join(ARTIFACTS, "backtest")
T = "arrival_week"
SEEDS = (7, 17, 27)
H4, TR18 = "lite_h4_lr0.00025", "lite_h4_lr0.00025_TRUNC"   # TRUNC -> the "_tr18" suffix the bundle carries after its seed
WORLDS = ("v6", "v7")
SNAPS = {1: 18, 2: 22, 3: 26, 4: 31, 5: 35, 6: 39, 7: 44, 8: 48}
YEAR = {1: 2022.0, 2: 2022.5, 6: 2024.5, 7: 2025.0}        # midpoint of the evaluation window, in years
EVAL = {1: "2022 H1", 2: "2022 H2", 6: "2024 H2", 7: "2025 H1"}
ORIGINS = (1, 2, 6, 7)


def band(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else dict(n=len(xs), mean=float(np.mean(xs)), spread=float(np.ptp(xs)),
                                    min=float(min(xs)), max=float(max(xs)), values=[float(x) for x in xs])


class D:
    def __init__(self):
        self.S = json.load(open(os.path.join(BT, "phase8_scores.json")))["scores"]
        self.IX = json.load(open(os.path.join(BT, "bundle_index.json")))

    def g(self, w, label, metric, i=0):
        v = self.S.get(f"{w}|{T}|{label}", {}).get(metric)
        return None if v is None else float(v[i])

    @staticmethod
    def label(k, cfg, s):
        """Bundle names put the truncation suffix AFTER the seed: lite_h4_lr0.00025_s7_tr18."""
        return f"o{k}_{cfg.replace('_TRUNC', '')}_s{s}" + ("_tr18" if cfg.endswith("_TRUNC") else "")

    def seeds(self, w, k, cfg, metric):
        return [x for x in (self.g(w, self.label(k, cfg, s), metric) for s in SEEDS) if x is not None]

    def promise(self, w, k, metric, i=0):
        return self.g(w, f"PROMISE_o{k}_promise_only", metric, i)


def cell(d, w, k, cfg):
    h = band(d.seeds(w, k, cfg, "roc_auc_late"))
    p = d.promise(w, k, "roc_auc_late")
    if not h or p is None:
        return None
    ci = (d.promise(w, k, "roc_auc_late", 1), d.promise(w, k, "roc_auc_late", 2))
    m = h["mean"] - p
    return dict(world=w, origin=k, config=cfg, head_roc=h, promise_roc=p, promise_roc_ci=ci,
                margin=m, margin_over_spread=(abs(m) / h["spread"] if h["spread"] > 0 else None),
                every_seed_above_promise=h["min"] > p, min_above_promise_ci_hi=h["min"] > ci[1] if ci[1] else None,
                cindex=band(d.seeds(w, k, cfg, "cindex")), promise_cindex=d.promise(w, k, "cindex"))


def main(out_path):
    d = D()
    base = [c for c in (cell(d, w, k, H4) for w in WORLDS for k in ORIGINS) if c]
    trunc = [c for c in (cell(d, w, 7, TR18) for w in WORLDS) if c]

    B = {}
    for c in trunc:
        w = c["world"]
        full = next(x for x in base if x["world"] == w and x["origin"] == 7)
        o1 = next(x for x in base if x["world"] == w and x["origin"] == 1)
        drop = c["margin"] - full["margin"]
        span = o1["margin"] - full["margin"]                       # what pure history-length would predict
        B[w] = dict(truncated=c, full_o7=full, o1=o1,
                    margin_truncated=c["margin"], margin_full=full["margin"], margin_o1=o1["margin"],
                    change_vs_full=drop,
                    fraction_of_the_way_to_o1=(drop / span) if span else None,
                    disjoint_from_full=(c["head_roc"]["max"] < full["head_roc"]["min"]
                                        or full["head_roc"]["max"] < c["head_roc"]["min"]),
                    still_above_promise=c["every_seed_above_promise"],
                    clears_promise_ci=c["min_above_promise_ci_hi"])

    rows = [dict(world=c["world"], origin=c["origin"], snapshots=SNAPS[c["origin"]], year=YEAR[c["origin"]],
                 margin=c["margin"]) for c in base]
    x_s = np.array([r["snapshots"] for r in rows]); x_y = np.array([r["year"] for r in rows])
    y = np.array([r["margin"] for r in rows])
    corr = {}
    for nm, x in (("training_snapshots", x_s), ("calendar_year", x_y)):
        pr, pp = stats.pearsonr(x, y); sr, sp = stats.spearmanr(x, y)
        corr[nm] = dict(pearson=float(pr), pearson_p=float(pp), spearman=float(sr), spearman_p=float(sp), n=len(y))
    corr["snapshots_vs_year"] = dict(pearson=float(stats.pearsonr(x_s, x_y)[0]),
                                     spearman=float(stats.spearmanr(x_s, x_y)[0]),
                                     note="the confound itself: the two predictors are near-perfectly collinear across the 8 cells")

    training = []
    for w in WORLDS:
        for s in SEEDS:
            e = d.IX.get(f"{w}|{T}|{D.label(7, TR18, s)}")
            if e:
                training.append(dict(world=w, seed=s, stop=e["stop"], epochs=e["epochs"], best_epoch=e["best_epoch"],
                                     seconds=e["seconds"], stamp=e["stamp"]))
    R = dict(truncated=B, existing_cells=rows, correlations=corr, training=training)
    json.dump(R, open(out_path, "w"), indent=1, default=float)

    f = lambda b: "—" if not b else f"{b['mean']:.4f} ({b['spread']:.4f})"
    print(f"{'w':3s} {'cell':28s} {'head ROC':>17s} {'promise':>8s} {'margin':>8s} {'xspread':>7s} {'>prom CI':>8s} {'C-index':>17s}")
    for c in base + trunc:
        nm = f"o{c['origin']} {EVAL[c['origin']]}" + (" TRUNCATED to 18 snaps" if c["config"] == TR18 else f" ({SNAPS[c['origin']]} snaps)")
        print(f"{c['world']:3s} {nm:28s} {f(c['head_roc']):>17s} {c['promise_roc']:8.4f} {c['margin']:+8.4f} "
              f"{c['margin_over_spread'] or float('nan'):7.1f} {str(c['min_above_promise_ci_hi']):>8s} {f(c['cindex']):>17s}")
    print()
    for w, b in B.items():
        print(f"{w}: origin 7 full {b['margin_full']:+.4f} -> truncated to 18 snapshots {b['margin_truncated']:+.4f} "
              f"(change {b['change_vs_full']:+.4f}); origin 1 at 18 snapshots was {b['margin_o1']:+.4f}; "
              f"{100 * (b['fraction_of_the_way_to_o1'] or 0):.0f}% of the way to origin 1; "
              f"bands disjoint from full: {b['disjoint_from_full']}")
    print("\ncorrelations across the 8 existing cells (confounded -- not evidence on their own):")
    for k, v in corr.items():
        print(" ", k, json.dumps({a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()}))
    print("\ntraining:")
    for t in training:
        print(f"  {t['world']} tr18 s{t['seed']:<2d} {t['stop']:12s} epochs {t['epochs']:>3} best {t['best_epoch']:>3} "
              f"{t['seconds']/60:5.1f} min {t['stamp']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase9b_history.json"))
    main(ap.parse_args().out)
