"""Guide 10.1 verify gates — and whether each one CAN fail.

Standing rule 1: a gate that cannot fail is not a gate. Every gate below is run twice — once against the real
solver, where it must PASS, and once against a deliberately broken solver (`solve(..., disable=...)`), where it
must FAIL. A gate that passes both ways proves nothing and is reported as vacuous.

G0 is the guide's own gate. Stage 0 established that it cannot fail:
  * `holding_cost` does not exist in the 49-table schema, so "zero holding cost" is true of EVERY part;
  * `max_volume_cap` is constant 5,000 against a largest weekly requirement of 2,102, so "no capacity
    constraint" is true of EVERY part as well;
  * and with c_hold = 0 the objective is indifferent to timing, so every schedule with the same number of order
    weeks costs exactly the same. The gate therefore tests the solver's tie-breaking, not its economics.
G0 demonstrates that vacuity by exhibiting two schedules with IDENTICAL cost, one of which passes it and one of
which fails it. G1-G4 replace it with gates that do discriminate.

  python ml/opt/gates.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np
from config import ARTIFACTS
from schedule_lp import solve


def part(demand, lot=25.0, moq=10.0, cap=5000.0, freight=1.0, safety=0.0, feasible=None, name="TEST"):
    T = len(demand)
    return dict(part_id=name, plant_id="PLX", weeks=[f"w{i}" for i in range(T)], demand=np.array(demand, float),
                feasible=[True] * T if feasible is None else feasible, safety_stock=safety, moq=moq, lot_size=lot,
                weekly_cap=cap, freight_per_unit=freight, unit_cost=100.0)


def last_order_week(r):
    return r["order_weeks"][-1] if r["order_weeks"] else None


# ---------------------------------------------------------------- the guide's gate
def G0_guide(solver):
    """'A part with no capacity constraint and ZERO holding cost must be ordered in the last feasible week.'"""
    p = part([10, 10, 10, 10, 10])
    r = solver(p, mode="coverage")
    return last_order_week(r) == "w4", r


def G0_vacuity():
    """Exhibit the degeneracy: with c_hold = 0, ordering first and ordering last cost the SAME."""
    p = part([10, 10, 10, 10, 10])
    r = solve(p, mode="coverage")
    x = np.zeros(len(p["demand"])); x[-1] = np.ceil(sum(p["demand"]) / p["lot_size"]) * p["lot_size"]
    x_first = np.zeros_like(x); x_first[0] = x[-1]
    cost = lambda v: float((v * p["freight_per_unit"]).sum())          # the only term the data supports
    return dict(solver_cost=r["cost"], cost_order_last=cost(x), cost_order_first=cost(x_first),
                identical=abs(cost(x) - cost(x_first)) < 1e-9, solver_chose=last_order_week(r))


# ---------------------------------------------------------------- replacements that can fail
def G1_holding_defers(solver):
    """With a POSITIVE holding cost, slack capacity and the requirement falling at the END of the horizon, the order
    must NOT be placed early -- holding stock for four weeks costs money and buys nothing.

    The demand profile matters: under a stock balance, receipts must arrive BEFORE the demand they cover, so a part
    with demand in every week cannot defer at all. An earlier version of this gate used a flat profile and failed
    against the correct solver for that reason. The gate was wrong, not the solver.
    """
    p = part([0, 0, 0, 0, 50])
    r = solver(p, mode="balance", c_hold=5.0, i0=0.0)
    return last_order_week(r) == "w4", r


def G2_moq_floor(solver):
    """A part whose MOQ exceeds its requirement orders nothing, or exactly one MOQ-satisfying lot -- never less."""
    p = part([3], lot=1.0, moq=50.0)
    r = solver(p, mode="coverage")
    got = r.get("total_received", 0.0)
    return (got == 0.0 or got >= p["moq"]), r


def G3_lot_multiple(solver):
    """Every receipt is a whole multiple of the lot size."""
    p = part([127, 88], lot=25.0)
    r = solver(p, mode="coverage")
    return all(abs(v % p["lot_size"]) < 1e-6 for v in r.get("receipts", [1])), r


def G4_coverage(solver):
    """Total receipts cover the requirement (to within one lot, since the requirement is not a lot multiple)."""
    p = part([127, 88], lot=25.0)
    r = solver(p, mode="coverage")
    return r.get("total_received", 0.0) >= sum(p["demand"]), r


def G5_infeasible_weeks(solver):
    """No receipt lands in a shutdown or non-working week.

    The infeasible week must be the one the objective WANTS, or the gate passes by luck: with linear freight every
    placement ties, and a broken solver avoids the closed week by tie-breaking alone. So the requirement falls in
    the last week, holding is positive (making the last week strictly cheapest) and the last week is shut. A correct
    solver pays a week of holding to receive in w1; a solver that ignores the calendar receives into the shutdown.
    """
    p = part([0, 0, 50], feasible=[True, True, False])
    r = solver(p, mode="balance", c_hold=5.0, i0=0.0)
    return (r.get("receipts", [0, 0, 1])[2] == 0), r


GATES = [("G1 holding defers the order", G1_holding_defers, dict(disable=("holding",))),
         ("G2 MOQ floor", G2_moq_floor, dict(disable=("moq",))),
         ("G3 lot-size multiple", G3_lot_multiple, dict(disable=("lot",))),
         ("G4 coverage", G4_coverage, dict(disable=("coverage",))),
         ("G5 infeasible weeks", G5_infeasible_weeks, dict(disable=("feasible",)))]


def main(out_path):
    R = {}
    ok_real, _ = G0_guide(solve)
    vac = G0_vacuity()
    R["G0_guide"] = dict(passes_against_real_solver=bool(ok_real), vacuity=vac,
                         verdict=("VACUOUS -- cannot fail: holding cost absent from the schema, capacity never binds, "
                                  "and ordering first costs exactly what ordering last costs"))
    print(f"G0 (the guide's gate): passes against the real solver = {ok_real}")
    print(f"   degeneracy: order-last {vac['cost_order_last']:.1f} vs order-first {vac['cost_order_first']:.1f} "
          f"-> identical = {vac['identical']}; solver happened to choose {vac['solver_chose']}")
    print( "   VERDICT: VACUOUS. Both schedules are optimal, so this gate tests tie-breaking, not economics.\n")

    for name, fn, broken in GATES:
        passes, _ = fn(solve)
        if broken is None:
            fails_when_broken, note = None, "no broken variant: the constraint has no switch to remove"
        else:
            fails_when_broken = not fn(lambda p, **kw: solve(p, **{**kw, **broken}))[0]
            note = f"broken by disable={broken['disable']}"
        R[name] = dict(passes=bool(passes), fails_when_broken=fails_when_broken, note=note,
                       can_fail=bool(fails_when_broken) if fails_when_broken is not None else None)
        flag = "PASS" if passes else "**FAIL**"
        cf = {True: "and CAN fail (shown)", False: "but could NOT be made to fail -- vacuous",
              None: "(no broken variant)"}[fails_when_broken]
        print(f"{name:32s} {flag:9s} {cf}   [{note}]")

    json.dump(R, open(out_path, "w"), indent=1, default=float)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "phase10_gates.json"))
    main(ap.parse_args().out)
