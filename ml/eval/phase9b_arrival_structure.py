"""Phase 9B Stage A — what the promise date's C-index actually measures.

The generator builds a purchase order's promise and its arrival from the SAME two upstream quantities:

    promise_date = po_created + contracted[channel]                             (generator §13, one line)
    lead         = lognormal(log(0.51 * contracted[channel]), 0.34) * stress
    arrival_week = po_created_week + ceil(lead / 7)

Measured from a snapshot t0, both carry `po_created` -- the line's AGE -- and both carry `contracted`. The shipped head
sees neither: its arrival prediction is a function of the CHANNEL's panel at t0, so every po_line on one channel at one
snapshot receives an identical distribution. This script measures, from artifacts already on disk, (1) that the head's
predictions are exactly tied within a channel-snapshot, which is the fingerprint of a channel-level forecaster, and
(2) how strongly the promise week -- a line-level quantity the head is never given -- orders the label.

Nothing is run, trained or selected here, and no table under db/ is read.

  python ml/eval/phase9b_arrival_structure.py
"""
from __future__ import annotations
import os, sys, json, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train")]
import numpy as np
from config import ARTIFACTS

BT = os.path.join(ARTIFACTS, "backtest")
PREDS = os.path.join(BT, "preds")
CELLS = [("v6", 1), ("v6", 2), ("v6", 6), ("v6", 7), ("v7", 1), ("v7", 2), ("v7", 6), ("v7", 7)]
HEAD = "lite_h4_lr0.00025_s7"


def measure(P, Y, EV, AUX):
    """Two facts, both from the artifact alone.

    1. EXACT ties in the head's predictions. The head reads a channel's panel at t0 and nothing about the individual
       po_line, so two lines of one channel at one snapshot must receive bit-identical distributions. Exact float
       equality across thousands of rows is the fingerprint of a channel-level forecaster.
    2. How well the promise week -- a LINE-level quantity the head is never given -- orders the label.

    An earlier version of this script also computed an "oracle over the head's prediction groups". It is not reported:
    with ~1.1 rows per group it measures the censoring ties (43-46% of rows carry one imputed value), not structure.
    """
    import phase5_metrics as M
    from scipy import stats
    obs = EV.astype(bool)
    _, inv, cnt = np.unique(P.astype(float), return_inverse=True, return_counts=True)
    sizes = cnt[inv]
    return dict(n_rows=int(len(Y)), n_distinct_predictions=int(len(cnt)),
                rows_sharing_a_prediction=int((sizes > 1).sum()), largest_tie_group=int(cnt.max()),
                head_cindex=float(M.arrival_scores(P, Y, EV, AUX)["cindex"][0]),
                promise_cindex=float(M.arrival_scores(AUX, Y, EV, AUX)["cindex"][0]),
                promise_vs_label_pearson=float(stats.pearsonr(AUX[obs], Y[obs])[0]),
                promise_vs_label_spearman=float(stats.spearmanr(AUX[obs], Y[obs])[0]),
                head_vs_label_spearman=float(stats.spearmanr(P[obs], Y[obs])[0]),
                censored_share=float(1.0 - obs.mean()))


def main(out_path):
    rows = []
    for w, k in CELLS:
        f = os.path.join(PREDS, f"{w}_arrival_week_o{k}_{HEAD}_test.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        r = measure(z["P"], z["Y"].astype(float), z["EV"].astype(bool), z["AUX"].astype(float))
        r.update(world=w, origin=k)
        rows.append(r)
        print(f"{w} o{k}: {r['n_rows']:,} rows, {r['n_distinct_predictions']:,} distinct predictions, "
              f"{r['rows_sharing_a_prediction']:,} rows exactly tied (largest group {r['largest_tie_group']}) | "
              f"head C {r['head_cindex']:.4f} (rho {r['head_vs_label_spearman']:+.3f}) | "
              f"promise C {r['promise_cindex']:.4f} (rho {r['promise_vs_label_spearman']:+.3f}, "
              f"r {r['promise_vs_label_pearson']:+.3f}) | censored {100*r['censored_share']:.1f}%")
    R = dict(cells=rows,
             note="the head is a channel-level forecaster (exact prediction ties within a channel-snapshot); the "
                  "promise week is a line-level input it is never given.")
    if rows:
        R["summary"] = dict(
            n=len(rows),
            promise_range=[min(r["promise_cindex"] for r in rows), max(r["promise_cindex"] for r in rows)],
            head_range=[min(r["head_cindex"] for r in rows), max(r["head_cindex"] for r in rows)],
            promise_label_spearman_range=[min(r["promise_vs_label_spearman"] for r in rows),
                                          max(r["promise_vs_label_spearman"] for r in rows)],
            head_label_spearman_range=[min(r["head_vs_label_spearman"] for r in rows),
                                       max(r["head_vs_label_spearman"] for r in rows)],
            tied_rows_total=sum(r["rows_sharing_a_prediction"] for r in rows))
        print("\nsummary:", json.dumps(R["summary"], indent=1))
    json.dump(R, open(out_path, "w"), indent=1, default=float)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase9b_arrival_structure.json"))
    main(ap.parse_args().out)
