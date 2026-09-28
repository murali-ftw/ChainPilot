"""Phase 13 S1 -- the simulation as a PRE-RESCUE forecast. No model change: only the REFERENCE is rebuilt.

reference = fraction of 2025 part-plant-weeks whose store level, with THAT WEEK'S net inter-plant transfer added
back (on hand - net transfer in), is below safety stock. The added-back transfers are FUTURE information relative to
each forecast t0. That is legitimate for a reference (references score against outcomes) and it is labelled as such:
the column is PRIVILEGED__prerescue_below_ss, and assert_no_privileged_headline refuses it as a selection key.

Simulated side: Phase 12 B2's five fill-head-seed runs of the ROP policy (ml/artifacts/phase12_b2_validate.json and
phase12_b2_seeds.json), N = 200 paths each, held FIXED -- nothing is re-simulated, every path was retained there and
the below-SS fraction is taken over all path-weeks at the end.

THE BUILDER MUST BE ABLE TO FAIL, shown before any headline:
  F-a identity      add back ZERO transfers -> must reproduce B2's per-seed ratios exactly (mean 1.475x)
  F-b wrong sign    subtract instead of add  -> must move AWAY from 1.0 by the pre-stated amount (>= 1.8x;
                    phase-13-stage0.md, committed before this ran)
  F-c shuffled      permute net transfers across part-plant-weeks, total preserved -> must NOT land in [1.0, 1.2].
                    If it does, total volume is doing the work and S1's headline is WITHDRAWN.
inventory_position_weekly is read here (qty_on_hand, safety_stock_qty, 2025 weeks) as the held-out evaluation
reference, exactly as Phase 12 B2's validation reads it. Never as a feature.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import phase12_common as C
from lateness_metric import assert_no_privileged_headline

D = C.REPO + "/db/gen_v8/seed_1001"
BAND = (1.0, 1.2)
FB_MIN = 1.8                         # pre-stated in phase-13-stage0.md


def store_2025():
    tx = pd.read_csv(f"{D}/inventory_transactions.csv", usecols=["part_id", "plant_id", "txn_type", "qty", "event_ts"])
    tx = tx[tx.txn_type.isin(["transfer_in", "transfer_out"])]
    tx["week"] = pd.to_datetime(tx.event_ts).dt.to_period("W-SUN").dt.start_time
    net = tx.groupby(["part_id", "plant_id", "week"]).qty.sum().rename("net")
    pw = pd.read_csv(f"{D}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "safety_stock_qty"])
    pw["week"] = pd.to_datetime(pw.week_start)
    pw = pw[(pw.week >= "2025-01-01") & (pw.week <= "2025-12-31")]
    return pw.join(net, on=["part_id", "plant_id", "week"]).fillna({"net": 0.0}).reset_index(drop=True)


def reference(pw, net_sign=+1.0, net=None):
    """below-SS fraction of (on hand - sign * net transfer). sign +1 = pre-rescue; 0 = observed; -1 = wrong sign."""
    n = pw.net.to_numpy(float) if net is None else net
    level = pw.qty_on_hand.to_numpy(float) - net_sign * n
    return float((level < pw.safety_stock_qty.to_numpy(float)).mean())


def sim_below():
    a = json.load(open(os.path.join(C.ART, "phase12_b2_validate.json")))
    b = json.load(open(os.path.join(C.ART, "phase12_b2_seeds.json")))
    out = {7: a["policy_rop"]["below_ss"]}
    out.update({s: b[f"policy_rop_fill_s{s}"]["below_ss"] for s in (17, 27, 37, 47)})
    assert a["paths"] == b["paths"] == 200, "MC path count must be held fixed across arms"
    b2_ratio = {7: a["policy_rop"]["ratio_vs_observed_on_hand"]}
    b2_ratio.update({s: b[f"policy_rop_fill_s{s}"]["ratio_vs_observed_on_hand"] for s in (17, 27, 37, 47)})
    return out, b2_ratio, a["paths"]


def ratios(sim, ref):
    r = {s: v / ref for s, v in sim.items()}
    v = list(r.values())
    return dict(per_seed={str(k): x for k, x in r.items()}, mean=float(np.mean(v)), lo=float(min(v)), hi=float(max(v)),
                in_band=bool(BAND[0] <= min(v) and max(v) <= BAND[1]))


def main():
    st = C.require_clean()
    pw = store_2025()
    sim, b2, paths = sim_below()
    R = dict(stamp=st, part_plant_weeks=int(len(pw)), paths=paths, sim_below_ss=sim)
    # ---------------- falsifications first
    ref0 = reference(pw, 0.0)
    fa = ratios(sim, ref0)
    fa_exact = all(abs(fa["per_seed"][str(s)] - b2[s]) < 1e-12 for s in sim)
    R["F-a_identity"] = dict(reference=ref0, ratios=fa, reproduces_B2_exactly=fa_exact,
                             b2_per_seed={str(k): v for k, v in b2.items()})
    refm = reference(pw, -1.0)
    fb = ratios(sim, refm)
    R["F-b_wrong_sign"] = dict(reference=refm, ratios=fb, prestated_min=FB_MIN,
                               moves_away_as_prestated=bool(fb["lo"] >= FB_MIN))
    rng = np.random.default_rng(13)
    net = pw.net.to_numpy(float)
    fc = []
    for _ in range(50):
        perm = rng.permutation(net)
        assert abs(perm.sum() - net.sum()) < 1e-6
        fc.append(ratios(sim, reference(pw, +1.0, perm)))
    fc_means = [x["mean"] for x in fc]
    R["F-c_shuffled"] = dict(n_permutations=50, ratio_mean_range=[float(min(fc_means)), float(max(fc_means))],
                             any_permutation_in_band=bool(any(x["in_band"] for x in fc)),
                             mean_reference=float(np.mean([reference(pw, 1.0, rng.permutation(net)) for _ in range(10)])))
    # ---------------- the headline, only if every falsification behaved
    refp = reference(pw, +1.0)
    metrics = {"PRIVILEGED__prerescue_below_ss": refp, "s1_ratio_vs_prerescue_reference": ratios(sim, refp)["mean"]}
    assert_no_privileged_headline(metrics, headline="s1_ratio_vs_prerescue_reference")
    try:
        assert_no_privileged_headline(metrics, selected="PRIVILEGED__prerescue_below_ss"); guard = "DID NOT FIRE"
    except AssertionError:
        guard = "FIRES: the pre-rescue reference cannot be used to select"
    R["privileged_guard"] = guard
    ok = fa_exact and R["F-b_wrong_sign"]["moves_away_as_prestated"] and not R["F-c_shuffled"]["any_permutation_in_band"]
    hp = ratios(sim, refp)
    R["headline"] = dict(reference_name="PRIVILEGED__prerescue_below_ss", reference=refp, ratios=hp,
                         falsifications_all_behaved=ok,
                         verdict=("PASS" if (ok and hp["in_band"]) else
                                  "WITHDRAWN (a falsification did not behave)" if not ok else "FAIL (outside [1.0, 1.2])"))
    print(json.dumps({k: v for k, v in R.items() if k != "stamp"}, indent=1, default=str))
    print(C.dump(R, "phase13_s1.json"))


if __name__ == "__main__":
    main()
