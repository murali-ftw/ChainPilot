"""Phase 9B Stage A — what the promise date's C-index actually measures.

The generator builds a purchase order's promise and its arrival from the SAME two upstream quantities:

    promise_date = po_created + contracted[channel]                             (generator §13, one line)
    lead         = lognormal(log(0.51 * contracted[channel]), 0.34) * stress
    arrival_week = po_created_week + ceil(lead / 7)

Measured from a snapshot t0, both carry `po_created` -- the line's AGE -- and both carry `contracted`. The shipped head
sees neither: its arrival prediction is a function of the CHANNEL's panel at t0, so every po_line on one channel at one
snapshot receives an identical distribution. This script measures that ceiling from artifacts already on disk: it
groups the evaluation rows by the head's own predicted value and asks what the BEST POSSIBLE forecaster that is
constant within those groups could score.

The oracle uses evaluation labels to order the groups. It is a BOUND, not a configuration, and nothing is selected on
it. No model is run, nothing is trained, and no table under db/ is read.

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


def oracle_group_bound(P, Y, EV, AUX):
    """Best C-index attainable by any forecaster constant within the head's own prediction groups."""
    import phase5_metrics as M
    g = np.unique(np.round(P.astype(float), 9), return_inverse=True)[1]
    obs = EV.astype(bool)
    overall = float(Y[obs].mean()) if obs.any() else float(Y.mean())
    mu = np.full(g.max() + 1, overall)
    for gi in range(g.max() + 1):
        m = (g == gi) & obs
        if m.any():
            mu[gi] = float(Y[m].mean())
    ideal = mu[g]                                   # rank groups perfectly; rows inside a group stay tied
    return dict(n_rows=int(len(Y)), n_groups=int(g.max() + 1),
                mean_group_size=float(len(Y) / (g.max() + 1)),
                largest_group=int(np.bincount(g).max()),
                head_cindex=float(M.arrival_scores(P, Y, EV, AUX)["cindex"][0]),
                oracle_group_cindex=float(M.arrival_scores(ideal, Y, EV, AUX)["cindex"][0]),
                promise_cindex=float(M.arrival_scores(AUX, Y, EV, AUX)["cindex"][0]))


def main(out_path):
    rows = []
    for w, k in CELLS:
        f = os.path.join(PREDS, f"{w}_arrival_week_o{k}_{HEAD}_test.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        r = oracle_group_bound(z["P"], z["Y"].astype(float), z["EV"].astype(bool), z["AUX"].astype(float))
        r.update(world=w, origin=k)
        rows.append(r)
        print(f"{w} o{k}: rows {r['n_rows']:,} in {r['n_groups']:,} prediction groups "
              f"(mean {r['mean_group_size']:.1f}, largest {r['largest_group']}) | "
              f"head C {r['head_cindex']:.4f} | ORACLE over the same groups {r['oracle_group_cindex']:.4f} | "
              f"promise {r['promise_cindex']:.4f}")
    R = dict(cells=rows,
             note="oracle_group_cindex is the ceiling for ANY forecaster constant within the head's prediction groups; "
                  "it is computed from evaluation labels as a bound and is never selected on.")
    if rows:
        R["summary"] = dict(
            oracle_below_promise=sum(r["oracle_group_cindex"] < r["promise_cindex"] for r in rows),
            n=len(rows),
            oracle_range=[min(r["oracle_group_cindex"] for r in rows), max(r["oracle_group_cindex"] for r in rows)],
            promise_range=[min(r["promise_cindex"] for r in rows), max(r["promise_cindex"] for r in rows)],
            head_range=[min(r["head_cindex"] for r in rows), max(r["head_cindex"] for r in rows)])
        print("\nsummary:", json.dumps(R["summary"], indent=1))
    json.dump(R, open(out_path, "w"), indent=1, default=float)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase9b_arrival_structure.json"))
    main(ap.parse_args().out)
