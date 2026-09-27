"""Phase 12 Wave B2 -- the observed-outcome validation of Phase 9.1, rebuilt and made reproducible.

Phase 11C Stage D.3 reported 44.36% simulated vs 8.38% observed part-plant-weeks below safety stock (5.29x)
but the script that produced it was never committed. Before any new policy is compared against it, this
module REBUILDS that comparison and must REPRODUCE both numbers from the unchanged placeholder simulation.
Nothing downstream is believed until it does.

THE METRIC IS OBSERVED OUTCOMES (B2.3). Both simulation arms inherit whatever bias the machinery has, so
simulated-vs-simulated is circular. The reference is the held-out 2025 store:
    observed  = fraction of 2025 part-plant-weeks in inventory_position_weekly whose level is below
                safety_stock_qty
    simulated = fraction of (part-plant, horizon week, path) cells whose projected level is below safety
                stock, over the 2025 test snapshots
and the arrivals-to-consumption replacement rate over the horizon (85.4% in 11C).

inventory_position_weekly is read here for the opening level (as-of t0) and for the observed reference
(2025 weeks). It is the store B1 reconciled at 100.000000%; it is never a model feature.
"""
from __future__ import annotations
import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "opt")]
import numpy as np, pandas as pd
import phase12_common as C
from config import WORLDS
import montecarlo as MC

W = MC.HORIZON_WEEKS


def test_snapshots(world, lo="2025-01-01", hi="2025-12-31"):
    s = pd.read_csv(f"{WORLDS[world]}/snapshots.csv", usecols=["as_of_ts"]).as_of_ts
    return sorted(pd.Timestamp(x) for x in s if pd.Timestamp(lo) <= pd.Timestamp(x) <= pd.Timestamp(hi))


def observed(world, lo="2025-01-01", hi="2025-12-31"):
    pw = pd.read_csv(f"{WORLDS[world]}/inventory_position_weekly.csv",
                     usecols=["part_id", "plant_id", "week_start", "qty_on_hand", "qty_available",
                              "safety_stock_qty"])
    pw = pw[(pw.week_start >= lo) & (pw.week_start <= hi)]
    return dict(part_plant_weeks=int(len(pw)),
                on_hand_below_ss=float((pw.qty_on_hand < pw.safety_stock_qty).mean()),
                available_below_ss=float((pw.qty_available < pw.safety_stock_qty).mean()))


def run_arm(world, snaps, sub, N, draw, seed=11, label=""):
    """draw(world, t0, sub, W, N, rng, op) -> (arrivals [P,W,N], consumption [P,W,N], meta).
    Returns per-snapshot below-SS fractions, replacement rate, shortfall magnitude."""
    rng = np.random.default_rng(seed)
    per = []
    for s in snaps:
        t = time.time()
        op = MC.opening_position(world, s, sub)
        I0 = op["qty_on_hand"].fillna(0).to_numpy(float)
        ss = op["safety_stock_qty"].fillna(1).to_numpy(float)
        arr, cons, meta = draw(world, s, sub, W, N, rng, op)
        pos, short = MC.roll_forward(I0, arr, cons, ss)
        below = pos < ss[:, None, None]
        per.append(dict(snapshot=str(s.date()), below_ss=float(below.mean()),
                        arrivals_per_pp=float(arr.sum(1).mean()), consumption_per_pp=float(cons.sum(1).mean()),
                        replacement=float(arr.sum() / max(cons.sum(), 1e-9)), opening_per_pp=float(I0.mean()),
                        mean_short_when_short=float(short[short > 0].mean()) if (short > 0).any() else 0.0,
                        post_horizon_per_pp=float(meta.get("post_horizon_per_pp", 0.0)),
                        seconds=round(time.time() - t, 1)))
        print(f"   [{label}] {s.date()} below-SS {100*per[-1]['below_ss']:.2f}%  replacement "
              f"{100*per[-1]['replacement']:.1f}%  ({per[-1]['seconds']}s)", flush=True)
    b = np.array([p["below_ss"] for p in per])
    return dict(per_snapshot=per, below_ss=float(b.mean()), below_ss_range=[float(b.min()), float(b.max())],
                replacement=float(np.sum([p["arrivals_per_pp"] for p in per]) / np.sum([p["consumption_per_pp"] for p in per])),
                arrivals_per_pp=float(np.mean([p["arrivals_per_pp"] for p in per])),
                consumption_per_pp=float(np.mean([p["consumption_per_pp"] for p in per])),
                opening_per_pp=float(np.mean([p["opening_per_pp"] for p in per])),
                mean_short_when_short=float(np.mean([p["mean_short_when_short"] for p in per])))


def placeholder_draw(world, t0, sub, W_, N, rng, op):
    return MC.draw_from_heads(world, t0, sub, W_, N, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v8")
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--arms", default="placeholder")
    ap.add_argument("--out", default="phase12_b2_validate.json")
    a = ap.parse_args()
    st = C.require_clean()
    pp, _, _ = MC.part_plant_universe(a.world)
    sub = pp.reset_index(drop=True)
    snaps = test_snapshots(a.world)
    R = dict(stamp=st, world=a.world, paths=a.paths, snapshots=[str(s.date()) for s in snaps],
             part_plants=int(len(sub)), observed=observed(a.world))
    print("observed", R["observed"])
    arms = {"placeholder": placeholder_draw}
    try:
        import order_policy as OP
        arms.update(OP.SIM_ARMS)
    except ImportError:
        pass
    for name in a.arms.split(","):
        R[name] = run_arm(a.world, snaps, sub, a.paths, arms[name], label=name)
        R[name]["ratio_vs_observed_on_hand"] = R[name]["below_ss"] / R["observed"]["on_hand_below_ss"]
        R[name]["ratio_vs_observed_available"] = R[name]["below_ss"] / R["observed"]["available_below_ss"]
        print(name, {k: v for k, v in R[name].items() if k != "per_snapshot"}, flush=True)
    print(C.dump(R, a.out))


if __name__ == "__main__":
    main()
