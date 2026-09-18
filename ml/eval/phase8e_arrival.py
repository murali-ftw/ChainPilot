"""Phase 8E — arrival on origins 1, 2, 6, 7: is the head's lateness advantage over the promise date stable across windows?

Question 3d only. Reads finished artifacts; nothing is re-scored here.
  ml/artifacts/backtest/phase8_scores.json   ml/eval/phase7_score.py --backtest
  ml/artifacts/backtest/bundle_index.json    ml/eval/backtest.py export (stop reason, epochs, best epoch)

Per origin x world:
  C-index            head (SHARE-lite h4) and h0, 3 seeds, against the promise date alone (Phase 7 binding statement 1)
  lateness ROC-AUC   head, h0, B5 LightGBM (5 fits) and promise date; margin over the promise date as a number and as a
                     multiple of the head's own seed spread; "inside the spread" when that multiple is < 1
Fixed split (Phase 7 section 7) is carried as quoted, for the side-by-side table.

  python ml/eval/phase8e_arrival.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np
from config import ARTIFACTS

BT = os.path.join(ARTIFACTS, "backtest")
T = "arrival_week"
SEEDS, B5_SEEDS = (7, 17, 27), (7, 17, 27, 37, 47)
H4, H0, B5 = "lite_h4_lr0.00025", "none_h0_lr0.00025", "b5flat_reg"
ORIGINS, WORLDS = (1, 2, 6, 7), ("v6", "v7")
FIXED = {  # reports/phase-7.md section 6-7, 2025 fixed split
    "v6": dict(head_roc=0.7814, head_roc_spread=0.0029, promise_roc=0.7482, b5_roc=0.7558, head_c=0.6734, head_c_spread=0.0020, promise_c=0.8831),
    "v7": dict(head_roc=0.7749, head_roc_spread=0.0017, promise_roc=0.7498, b5_roc=0.7555, head_c=0.6773, head_c_spread=0.0003, promise_c=0.8770)}


def band(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else dict(n=len(xs), mean=float(np.mean(xs)), spread=float(np.ptp(xs)), min=float(min(xs)),
                                    max=float(max(xs)), values=[float(x) for x in xs])


def main(out_path):
    S = json.load(open(os.path.join(BT, "phase8_scores.json")))["scores"]
    IX = json.load(open(os.path.join(BT, "bundle_index.json")))
    g = lambda w, lab, m, i=0: (S.get(f"{w}|{T}|{lab}", {}).get(m) or [None])[i]
    seeds = lambda w, k, cfg, m, ss=SEEDS: [g(w, f"o{k}_{cfg}_s{s}", m) for s in ss if g(w, f"o{k}_{cfg}_s{s}", m) is not None]
    rows, training = [], []
    for w in WORLDS:
        for k in ORIGINS:
            hr, zr, br = band(seeds(w, k, H4, "roc_auc_late")), band(seeds(w, k, H0, "roc_auc_late")), band(seeds(w, k, B5, "roc_auc_late", B5_SEEDS))
            hc, zc = band(seeds(w, k, H4, "cindex")), band(seeds(w, k, H0, "cindex"))
            pr, pc = g(w, f"PROMISE_o{k}_promise_only", "roc_auc_late"), g(w, f"PROMISE_o{k}_promise_only", "cindex")
            pr_ci = (g(w, f"PROMISE_o{k}_promise_only", "roc_auc_late", 1), g(w, f"PROMISE_o{k}_promise_only", "roc_auc_late", 2))
            if not hr or pr is None:
                rows.append(dict(world=w, origin=k, missing=True)); continue
            m = hr["mean"] - pr
            mult = abs(m) / hr["spread"] if hr["spread"] > 0 else None
            rows.append(dict(
                world=w, origin=k, late_rate=g(w, f"PROMISE_o{k}_promise_only", "late_rate"),
                promise_roc=pr, promise_roc_ci=pr_ci, head_roc=hr, h0_roc=zr, b5_roc=br,
                head_minus_promise=m, head_minus_promise_over_head_spread=mult,
                margin_inside_head_spread=(mult is not None and mult < 1.0),
                every_head_seed_above_promise=hr["min"] > pr,
                head_min_above_promise_ci_hi=hr["min"] > pr_ci[1] if pr_ci[1] is not None else None,
                b5_minus_promise=(br["mean"] - pr) if br else None,
                head_minus_b5=(hr["mean"] - br["mean"]) if br else None,
                head_vs_b5=("head" if br and hr["min"] > br["max"] else "B5" if br and br["min"] > hr["max"] else "not dist.") if br else "missing",
                head_vs_h0_lateness=("head" if zr and hr["min"] > zr["max"] else "h0" if zr and zr["min"] > hr["max"] else "not dist.") if zr else "missing",
                promise_c=pc, head_c=hc, h0_c=zc,
                head_minus_promise_c=(hc["mean"] - pc) if hc and pc is not None else None,
                head_outranks_promise_c=bool(hc) and pc is not None and hc["min"] > pc))
            for cfg, lab in ((H4, "h4"), (H0, "h0")):
                for s in SEEDS:
                    e = IX.get(f"{w}|{T}|o{k}_{cfg}_s{s}")
                    if e:
                        training.append(dict(world=w, origin=k, config=lab, seed=s, stop=e["stop"], epochs=e["epochs"],
                                             best_epoch=e["best_epoch"], seconds=e["seconds"], stamp=e["stamp"]))
    ok = [r for r in rows if not r.get("missing")]
    summary = dict(
        windows=len(ok),
        margin_inside_head_spread=sum(r["margin_inside_head_spread"] for r in ok),
        every_head_seed_above_promise=sum(r["every_head_seed_above_promise"] for r in ok),
        head_outranks_promise_c=sum(r["head_outranks_promise_c"] for r in ok),
        head_beats_b5_lateness=sum(r["head_vs_b5"] == "head" for r in ok),
        cap_bound_cells=[t for t in training if t["stop"] != "patience"],
        dirty_stamps=[t for t in training if "dirty" in t["stamp"]])
    R = dict(rows=rows, training=training, summary=summary, fixed_split=FIXED)
    json.dump(R, open(out_path, "w"), indent=1, default=float)

    f = lambda b: "—" if not b else f"{b['mean']:.4f} ({b['spread']:.4f})"
    print(f"{'w':3s} {'o':>2s} {'late':>6s} {'promise ROC':>12s} {'head ROC':>17s} {'h0 ROC':>17s} {'B5 ROC':>17s} "
          f"{'head-prom':>9s} {'x spread':>8s} {'all>prom':>8s} {'B5-prom':>8s} {'head v B5':9s} | {'prom C':>7s} {'head C':>17s} {'h0 C':>17s}")
    for r in ok:
        print(f"{r['world']:3s} {r['origin']:2d} {r['late_rate']:6.3f} {r['promise_roc']:12.4f} {f(r['head_roc']):>17s} {f(r['h0_roc']):>17s} "
              f"{f(r['b5_roc']):>17s} {r['head_minus_promise']:+9.4f} {r['head_minus_promise_over_head_spread'] or float('nan'):8.1f} "
              f"{str(r['every_head_seed_above_promise']):>8s} {r['b5_minus_promise'] if r['b5_minus_promise'] is not None else float('nan'):+8.4f} "
              f"{r['head_vs_b5']:9s} | {r['promise_c']:7.4f} {f(r['head_c']):>17s} {f(r['h0_c']):>17s}")
    for w in WORLDS:
        F = FIXED[w]
        print(f"{w} fixed split: head-promise {F['head_roc'] - F['promise_roc']:+.4f} ({(F['head_roc'] - F['promise_roc']) / F['head_roc_spread']:.1f}x spread)")
    print("\ntraining:")
    for t in training:
        print(f"  {t['world']} o{t['origin']} {t['config']} s{t['seed']:<2d} {t['stop']:12s} epochs {t['epochs']:>3} best {t['best_epoch']:>3} {t['seconds']/60:6.1f} min {t['stamp']}")
    print("\nsummary:", json.dumps({k: v for k, v in summary.items() if k not in ("cap_bound_cells", "dirty_stamps")}),
          "cap-bound:", len(summary["cap_bound_cells"]), "dirty:", len(summary["dirty_stamps"]))
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase8e_arrival.json"))
    main(ap.parse_args().out)
