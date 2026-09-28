"""Phase 13 S2 -- a transfer recommendation layer on the simulation, validated against planner ACTION.

LABEL (non-negotiable, carried in every output):
  "freight cost and transfer lead time not weighed; validated against planner ACTION, not planner INTENT; the donor
   safety check inherits an uncalibrated simulation (see S1)."
It measures AGREEMENT WITH WHAT PLANNERS DID. Planners' transfers are current practice, not the optimum, and a
higher agreement is NOT evidence of shortage avoided.

PRECONDITION (deviation 91). The donor search does NOT reuse allocation's qualification predicate: an inter-plant
stock transfer moves the plant's own stock of the same part and involves no supplier. 0% of v8 is excluded by it.

RULES, per snapshot, from the ROP policy simulation (order_policy, N = 200 paths, fill-head seed s):
  projected short   recipient part-plant-week w with P_sim(level < SS) >= 0.5  (a fixed definition, not tuned)
  POLICY            donor = the same-part plant whose simulated shortage chance AFTER giving the transfer is lowest,
                    recommended only if that chance < theta. qty = min(recipient median deficit, donor median surplus)
  NAIVE             donor = the NEAREST same-part plant (haversine, plants.csv) with positive as-of surplus at t0
  RANDOM            donor = a uniformly random same-part plant with positive simulated median surplus -- the base
                    rate: transfers are frequent, so "some same-part plant shipped out that week" is not rare
Both are ranked by the recipient's simulated shortage chance and compared at MATCHED k (precision@k), so a greedy rule
cannot win on volume.

GROUND TRUTH, from the ledger (evaluation only): `inventory_transactions.from_plant_id` equals the RECEIVING plant on
every transfer_in row, so the donor is never recorded. A hit therefore needs (i) a transfer_in of the part at the
recipient plant in that week and, for a DONOR hit, (ii) a transfer_out of the same part at the recommended donor in that
week. Recipient-level and donor-level precision are both reported.

theta is fitted ONCE on VALIDATION (2024 snapshots), frozen, then swept 0.02-0.5 on test. A verdict that flips inside
the sweep is no verdict.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "sim"),
                os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import phase12_common as C
import montecarlo as MC
import order_policy as OP

LABEL = ("freight cost and transfer lead time not weighed; validated against planner ACTION, not planner INTENT; "
         "the donor safety check inherits an uncalibrated simulation (see S1).")
D = C.REPO + "/db/gen_v8/seed_1001"
W = MC.HORIZON_WEEKS
N = 200
SHORT_P = 0.5
THETAS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def snapshots(lo, hi):
    s = pd.read_csv(f"{D}/snapshots.csv", usecols=["as_of_ts"]).as_of_ts
    return sorted(pd.Timestamp(x) for x in s if pd.Timestamp(lo) <= pd.Timestamp(x) <= pd.Timestamp(hi))


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


class Truth:
    def __init__(self):
        tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts"])
        tx = tx[tx.txn_type.isin(["transfer_in", "transfer_out"])]
        tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
        g = tx.groupby(["part_id", "plant_id", "week", "txn_type"]).qty.sum().unstack(fill_value=0)
        self.tin = set(g[g.get("transfer_in", 0) > 0].index)
        self.tout = set(g[g.get("transfer_out", 0) < 0].index)


def simulate(t0, sub, seed):
    op = MC.opening_position("v8", t0, sub)
    rng = np.random.default_rng(1000 + seed)
    arr, cons, _ = OP.make_draw("rop", fill_seed=seed)("v8", t0, sub, W, N, rng, op)
    I0 = op.qty_on_hand.fillna(0).to_numpy(float); ss = op.safety_stock_qty.fillna(0).to_numpy(float)
    pos, _ = MC.roll_forward(I0, arr, cons, ss)
    return pos, ss, I0


def recommend(t0, sub, pos, ss, I0, plants_ll, theta, rng=None):
    """-> (policy recs, naive recs); each rec: recipient index, donor index, week, qty, before/after chances, score."""
    p_short = (pos < ss[:, None, None]).mean(2)                       # [P, W]
    med = np.median(pos, 2)
    parts = sub.part_id.to_numpy(); plant = sub.plant_id.to_numpy()
    by_part = pd.Series(np.arange(len(sub))).groupby(parts).apply(list).to_dict()
    pol, nai, rnd = [], [], []
    for i, w in zip(*np.nonzero(p_short >= SHORT_P)):
        cands = [j for j in by_part[parts[i]] if j != i]
        if not cands:
            continue
        deficit = max(ss[i] - med[i, w], 1.0)
        best = None
        for j in cands:
            surplus = med[j, w] - ss[j]
            if surplus <= 0:
                continue
            q = min(deficit, surplus)
            p_after_d = float((pos[j, w] - q < ss[j]).mean())
            if best is None or p_after_d < best[1]:
                best = (j, p_after_d, q)
        rec_after = lambda q: float((pos[i, w] + q < ss[i]).mean())
        pos_c = [j for j in cands if med[j, w] - ss[j] > 0]
        if pos_c and rng is not None:
            j = pos_c[int(rng.integers(len(pos_c)))]
            q = min(deficit, med[j, w] - ss[j])
            rnd.append(dict(r=i, d=j, w=int(w), qty=float(q), score=float(p_short[i, w]),
                            recipient_before=float(p_short[i, w]), recipient_after=rec_after(q),
                            donor_before=float(p_short[j, w]), donor_after=float((pos[j, w] - q < ss[j]).mean())))
        if best is not None and best[1] < theta:
            j, pad, q = best
            pol.append(dict(r=i, d=j, w=int(w), qty=float(q), score=float(p_short[i, w]),
                            recipient_before=float(p_short[i, w]), recipient_after=rec_after(q),
                            donor_before=float(p_short[j, w]), donor_after=pad))
        # naive: nearest same-part plant with positive AS-OF surplus at t0 (store level, no simulation)
        sur = [(j, I0[j] - ss[j]) for j in cands if I0[j] - ss[j] > 0]
        if sur:
            j = min(sur, key=lambda x: haversine(plants_ll[plant[i]], plants_ll[plant[x[0]]]))[0]
            q = min(deficit, I0[j] - ss[j])
            nai.append(dict(r=i, d=j, w=int(w), qty=float(q), score=float(p_short[i, w]),
                            recipient_before=float(p_short[i, w]), recipient_after=rec_after(q),
                            donor_before=float(p_short[j, w]), donor_after=float((pos[j, w] - q < ss[j]).mean())))
    return pol, nai, rnd


def score(recs, sub, t0, truth, k=None):
    recs = sorted(recs, key=lambda r: -r["score"])[:k] if k is not None else recs
    if not recs:
        return dict(n=0, precision_recipient=float("nan"), precision_donor=float("nan"), hits_recipient=0, hits_donor=0)
    parts = sub.part_id.to_numpy(); plant = sub.plant_id.to_numpy()
    hr = hd = 0
    for r in recs:
        wk = pd.Timestamp(t0) + pd.Timedelta(days=7 * (r["w"] + 1))
        wk = wk - pd.Timedelta(days=wk.weekday())
        rec_hit = (parts[r["r"]], plant[r["r"]], wk) in truth.tin
        don_hit = rec_hit and (parts[r["r"]], plant[r["d"]], wk) in truth.tout
        hr += rec_hit; hd += don_hit
    return dict(n=len(recs), hits_recipient=hr, hits_donor=hd, precision_recipient=hr / len(recs),
                precision_donor=hd / len(recs))


def actual_events(sub, t0, truth):
    parts = set(sub.part_id); ws = {pd.Timestamp(t0) + pd.Timedelta(days=7 * (w + 1)) for w in range(W)}
    ws = {w - pd.Timedelta(days=w.weekday()) for w in ws}
    return sum(1 for (p, pl, wk) in truth.tin if wk in ws and p in parts)


def run_fold(snaps, sub, truth, plants_ll, seeds, thetas):
    out = {th: {s: dict(pol=[], nai=[], events=0, recs=[]) for s in seeds} for th in thetas}
    for t0 in snaps:
        ev = actual_events(sub, t0, truth)
        for s in seeds:
            pos, ss, I0 = simulate(t0, sub, s)
            for th in thetas:
                pol, nai, rnd = recommend(t0, sub, pos, ss, I0, plants_ll, th,
                                          rng=np.random.default_rng(hash((str(t0), s)) % 2**32))
                k = min(len(pol), len(nai), len(rnd))
                o = out[th][s]
                o["pol"].append(score(pol, sub, t0, truth, k)); o["nai"].append(score(nai, sub, t0, truth, k))
                o.setdefault("rnd", []).append(score(rnd, sub, t0, truth, k))
                o["pol_all"] = o.get("pol_all", []) + [score(pol, sub, t0, truth)]
                o["nai_all"] = o.get("nai_all", []) + [score(nai, sub, t0, truth)]
                o["events"] += ev
                if len(o["recs"]) < 5 and pol:
                    o["recs"].append(dict(t0=str(t0.date()), **{k2: v for k2, v in pol[0].items()},
                                          recipient=f"{sub.part_id.iloc[pol[0]['r']]}@{sub.plant_id.iloc[pol[0]['r']]}",
                                          donor=f"{sub.part_id.iloc[pol[0]['d']]}@{sub.plant_id.iloc[pol[0]['d']]}"))
        print(f"   {t0.date()} done", flush=True)
    summ = {}
    for th in thetas:
        per = {}
        for s in seeds:
            o = out[th][s]
            agg = lambda lst, key: sum(x[key] for x in lst)
            n_pk = agg(o["pol"], "n")
            per[s] = dict(k_matched=n_pk,
                          policy_precision_recipient=agg(o["pol"], "hits_recipient") / max(n_pk, 1),
                          naive_precision_recipient=agg(o["nai"], "hits_recipient") / max(n_pk, 1),
                          policy_precision_donor=agg(o["pol"], "hits_donor") / max(n_pk, 1),
                          naive_precision_donor=agg(o["nai"], "hits_donor") / max(n_pk, 1),
                          random_precision_donor=agg(o["rnd"], "hits_donor") / max(n_pk, 1),
                          random_precision_recipient=agg(o["rnd"], "hits_recipient") / max(n_pk, 1),
                          policy_count=agg(o["pol_all"], "n"), naive_count=agg(o["nai_all"], "n"),
                          policy_recall_recipient=agg(o["pol_all"], "hits_recipient") / max(o["events"], 1),
                          naive_recall_recipient=agg(o["nai_all"], "hits_recipient") / max(o["events"], 1),
                          actual_transfer_in_events=o["events"], examples=o["recs"])
        summ[str(th)] = per
    return summ


def band(per, key):
    v = [p[key] for p in per.values()]
    return [float(min(v)), float(np.mean(v)), float(max(v))]


def main():
    st = C.require_clean()
    pp, _, _ = MC.part_plant_universe("v8")
    sub = pp.reset_index(drop=True)
    pl = pd.read_csv(f"{D}/plants.csv", usecols=["plant_id", "latitude", "longitude"])
    plants_ll = {r.plant_id: (r.latitude, r.longitude) for r in pl.itertuples()}
    truth = Truth()
    seeds = C.V8_SEEDS
    R = dict(stamp=st, label=LABEL, short_definition=f"P_sim(level < SS) >= {SHORT_P}", paths=N,
             qualification_predicate_reused=False, fraction_excluded_by_qualification=0.0)
    # ---- theta fitted on VALIDATION: maximise the policy's matched-k DONOR precision x recall (F1) on 2024
    val = run_fold(snapshots("2024-01-01", "2024-12-31"), sub, truth, plants_ll, seeds, THETAS)
    f1 = {}
    for th, per in val.items():
        p = np.mean([x["policy_precision_donor"] for x in per.values()])
        r = np.mean([x["policy_recall_recipient"] for x in per.values()])
        f1[th] = 2 * p * r / max(p + r, 1e-12)
    theta = max(f1, key=f1.get)
    R["validation"] = dict(f1_by_theta=f1, selected_theta=float(theta),
                           summary={th: {k: band(per, k) for k in ("policy_precision_donor", "naive_precision_donor",
                                                                   "policy_precision_recipient", "naive_precision_recipient")}
                                    for th, per in val.items()})
    print("theta selected on validation:", theta, f1, flush=True)
    # ---- test, at the frozen theta AND the full sweep
    test = run_fold(snapshots("2025-01-01", "2025-12-31"), sub, truth, plants_ll, seeds, THETAS)
    sweep = {}
    for th, per in test.items():
        s = {k: band(per, k) for k in ("policy_precision_recipient", "naive_precision_recipient", "policy_precision_donor",
                                       "naive_precision_donor", "random_precision_donor", "random_precision_recipient", "policy_recall_recipient", "naive_recall_recipient",
                                       "policy_count", "naive_count", "k_matched")}
        pol_lo, nai_hi = s["policy_precision_donor"][0], s["naive_precision_donor"][2]
        nai_lo, pol_hi = s["naive_precision_donor"][0], s["policy_precision_donor"][2]
        s["verdict_donor"] = "policy" if pol_lo > nai_hi else "naive" if nai_lo > pol_hi else "UNDETERMINED"
        s["policy_beats_random_donor"] = bool(s["policy_precision_donor"][0] > s["random_precision_donor"][2])
        s["naive_beats_random_donor"] = bool(s["naive_precision_donor"][0] > s["random_precision_donor"][2])
        sweep[th] = s
    R["test_sweep"] = sweep
    R["test_at_frozen_theta"] = sweep[str(theta)] if str(theta) in sweep else sweep[theta]
    verdicts = {v["verdict_donor"] for v in sweep.values()}
    R["verdict"] = (list(verdicts)[0] if len(verdicts) == 1 else "NO VERDICT: flips inside the sweep " + str(sorted(verdicts)))
    R["examples"] = test[theta][7]["examples"] if theta in test else []
    print(json.dumps({k: v for k, v in R.items() if k not in ("stamp",)}, indent=1, default=str))
    print(C.dump(R, "phase13_s2.json"))


if __name__ == "__main__":
    main()
