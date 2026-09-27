"""Phase 12 A5.1: ONE order policy. Trigger, lead-time and pipeline logic live in ml/opt/order_policy.py only.

Fails if a consumer (montecarlo, schedule_lp) references the policy's inputs directly or re-implements a
horizon-requirement order. The Phase 9.1 placeholder in montecarlo.draw_from_heads is the one sanctioned exception:
it is kept, labelled, so Phase 11C's 5.29x can be reproduced, and it is listed here by name so it cannot spread.

Demonstrated capable of failing: `python ml/tests/test_single_order_policy.py --falsify` injects a reorder-point
reference into a copy of schedule_lp.py and requires the check to flag it.
"""
import os, re, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..")
CONSUMERS = [os.path.join(ML, "sim", "montecarlo.py"), os.path.join(ML, "opt", "schedule_lp.py")]
FORBIDDEN = [r"reorder_point_qty", r"planning_lead_time_days", r"open_po_qty", r"pmf_quantile", r"lead_pmf\(",
             r"cons\.mean\(2\)\.sum\(1\)"]
# (file, pattern) -> the ONLY function it may appear in
SANCTIONED = {("montecarlo.py", r"cons\.mean\(2\)\.sum\(1\)"): "draw_from_heads"}


def _span(src, fn):
    i = src.find(f"\ndef {fn}(")
    if i < 0:
        return (0, 0)
    j = src.find("\ndef ", i + 1)
    return (i, j if j > 0 else len(src))


def violations(paths):
    out = []
    for p in paths:
        src = open(p).read()
        for pat in FORBIDDEN:
            fn = SANCTIONED.get((os.path.basename(p), pat))
            lo, hi = _span(src, fn) if fn else (0, 0)
            for m in re.finditer(pat, src):
                if lo <= m.start() < hi:
                    continue
                line = src[:m.start()].count("\n") + 1
                out.append(f"{os.path.basename(p)}:{line} references {pat}")
    return out


def test_single_order_policy():
    v = violations(CONSUMERS)
    assert not v, "order-policy logic outside ml/opt/order_policy.py:\n  " + "\n  ".join(v)


def falsify():
    d = tempfile.mkdtemp()
    try:
        q = os.path.join(d, "schedule_lp.py")
        shutil.copy(CONSUMERS[1], q)
        with open(q, "a") as f:
            f.write("\n# injected\nrop = part_plant.reorder_point_qty\n")
        m = os.path.join(d, "montecarlo.py")
        src = open(CONSUMERS[0]).read()
        with open(m, "w") as f:      # the placeholder's heuristic copied OUTSIDE its sanctioned function
            f.write(src + "\n\ndef policy_draw(cons):\n    return cons.mean(2).sum(1)\n")
        v = violations([m, q])
        return bool(v), v
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        fired, v = falsify()
        print("FIRES" if fired else "DID NOT FIRE", v)
        sys.exit(0 if fired else 1)
    test_single_order_policy(); print("PASS")
