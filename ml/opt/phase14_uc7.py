"""Phase 14 UC7 -- "the recommended top supplier is the one actually used."

Recommendation: Phase 12 A4's v8 sweep at price_weight = 0, deduplicated (ml/artifacts/phase12_a4.json,
rows_pw0_deduped), t0 = 2025-06-30, the same 120 part-plants Phase 11's rule selects. The winning candidate's split is
rebuilt deterministically with allocation.candidates (no training, no re-scoring).
Actual: the supplier with the largest ordered quantity on PO lines RAISED in (t0, t0 + 13 weeks] at the part-plant
(evaluation only -- future information by construction).
precision@1: recommended top-share supplier == actual top. precision@2: actual top in the recommendation's top-2.
Baseline: the INCUMBENT's top supplier (the status quo). Feasibility is reported FIRST; precision is conditional on the
feasible part-plants, with n, and also reported unconditionally (an infeasible part-plant counted as a miss).
Deterministic: row-bootstrap 95% over part-plants, labelled.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import phase12_common as C
from allocation import load, candidates
from phase12_a4_price import phase11_keys
from sensitivity import dedupe_candidates

T0 = pd.Timestamp("2025-06-30")
D = C.REPO + "/db/gen_v8/seed_1001"


def topk(split, k):
    return [s for s, v in sorted(split.items(), key=lambda x: -x[1]) if v > 0][:k]


def main():
    st = C.require_clean()
    a4 = json.load(open(os.path.join(C.ART, "phase12_a4.json")))
    rows = {(r["part_id"], r["plant_id"]): r for r in a4["v8"]["rows_pw0_deduped"]}
    L = load("v8", str(T0.date()))
    keys, _ = phase11_keys(L, 120)
    pol = pd.read_csv(f"{D}/po_lines.csv", usecols=["channel_id", "qty_ordered", "created_ts"])
    ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "part_id", "supplier_id", "plant_id"])
    pol = pol.merge(ch, on="channel_id")
    ct = pd.to_datetime(pol.created_ts)
    fut = pol[(ct > T0) & (ct <= T0 + pd.Timedelta(weeks=13))]
    actual = fut.groupby(["part_id", "plant_id", "supplier_id"]).qty_ordered.sum()
    out = []
    for k in keys:
        a = actual.get(k)
        act_top = a.idxmax() if a is not None and len(a) else None
        feasible = k in rows
        rec1 = rec2 = inc1 = inc2 = None
        if feasible:
            q, cands = candidates(L, *k)
            cands, _ = dedupe_candidates(cands)
            split = dict(cands)[rows[k]["winners"][0]]
            rec1, rec2 = topk(split, 1), topk(split, 2)
        q, cands = candidates(L, *k)
        inc = dict(cands)["incumbent"]
        inc1, inc2 = topk(inc, 1), topk(inc, 2)
        out.append(dict(part_id=k[0], plant_id=k[1], feasible=feasible, has_actual=act_top is not None, actual_top=act_top,
                        winner=rows[k]["winners"][0] if feasible else None,
                        rec_p1=bool(feasible and act_top in rec1), rec_p2=bool(feasible and act_top in rec2),
                        inc_p1=bool(act_top in inc1), inc_p2=bool(act_top in inc2),
                        rec_equals_incumbent_top=bool(feasible and rec1 == inc1)))
    df = pd.DataFrame(out)
    ev = df[df.has_actual]
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        if not len(x):
            return [float("nan")] * 3
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
        return [float(np.percentile(b, 2.5)), float(x.mean()), float(np.percentile(b, 97.5))]
    fe = ev[ev.feasible]
    R = dict(stamp=st, t0=str(T0.date()), sampled=int(len(df)), with_actual_orders=int(len(ev)),
             feasible=int(df.feasible.sum()), feasibility_rate=float(df.feasible.mean()),
             conditional_on_feasible=dict(n=int(len(fe)), rec_p1=boot(fe.rec_p1), rec_p2=boot(fe.rec_p2),
                                          incumbent_p1=boot(fe.inc_p1), incumbent_p2=boot(fe.inc_p2),
                                          rec_equals_incumbent_top=float(fe.rec_equals_incumbent_top.mean())),
             unconditional_over_all_sampled=dict(n=int(len(ev)), rec_p1=boot(ev.rec_p1), rec_p2=boot(ev.rec_p2),
                                                 incumbent_p1=boot(ev.inc_p1), incumbent_p2=boot(ev.inc_p2)),
             winners=fe.winner.value_counts().to_dict(),
             band_kind="row bootstrap 95% over part-plants (deterministic)")
    print(json.dumps({k: v for k, v in R.items() if k != "stamp"}, indent=1, default=str))
    print(C.dump(R, "phase14_uc7.json"))


if __name__ == "__main__":
    main()
