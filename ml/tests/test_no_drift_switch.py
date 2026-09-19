"""Phase 10 Stage 1.2's regression test: the served distribution must not depend on the drift statistic.

Phase 8 section 4 3c removed arrival's drift gate: its output flipped with the training seed in 8 of 16 windows and
the two branches were never distinguishable. Drift and excess are still computed and logged as monitoring. These
tests fail if anything ever makes the served output move with them again.

Standing rule 1 applies to these tests themselves, so `test_the_sweep_can_fail` builds a gated implementation and
shows the sweep catching it. A sweep that passes for every possible implementation would prove nothing.

    python ml/tests/test_no_drift_switch.py      (or: pytest ml/tests/test_no_drift_switch.py)
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]

SHIPPED = os.path.join(HERE, "..", "configs", "shipped.json")
SWEEP = [0.0, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.5, 2.0, 4.5, 9.9, 50.0, 1e3, 1e6]   # spans every historical band edge


def _shipped():
    return json.load(open(SHIPPED))


def test_shipped_config_has_no_fallback_and_serves_h0():
    c = _shipped()
    assert c["tasks"]["arrival_week"]["fallback"] is None, "the drift fallback is configured again"
    assert c["tasks"]["arrival_week"]["serve_distribution"] == "h0_always"
    assert c["tasks"]["fill_rate"]["fallback"] is None


def test_serve_source_is_constant_across_the_drift_sweep():
    """The decision must not be a function of drift. serve_source cannot even see it -- verified by signature."""
    import inspect
    import loop as L
    c = _shipped()
    params = set(inspect.signature(L.serve_source).parameters)
    assert not (params & {"drift", "excess", "excess_pp", "overall", "drift_pp", "stat"}), \
        f"serve_source takes a drift-shaped argument: {params}"
    for task in ("arrival_week", "fill_rate"):
        for h0 in (True, False):
            got = {L.serve_source(task, c, h0) for _ in SWEEP}
            assert len(got) == 1, f"{task}: served source is not constant"
    assert L.serve_source("arrival_week", c, True) == "h0"          # h0 served always, where a reference exists
    assert L.serve_source("arrival_week", c, False) == "model"      # nothing to fall back to
    assert L.serve_source("fill_rate", c, True) == "model"          # fill has no h0 twin: it IS the shipped model


def test_reintroducing_a_fallback_raises():
    import loop as L
    c = _shipped()
    c["tasks"]["arrival_week"]["fallback"] = {"rule": "...", "threshold_pp": 1.0}
    try:
        L.serve_source("arrival_week", c, True)
    except AssertionError:
        return
    raise AssertionError("a reintroduced fallback must fail loudly, not be silently ignored")


def test_the_sweep_can_fail():
    """Standing rule 1: show the sweep catching a gated implementation. If it cannot, it is not a test."""
    def gated_serve_source(task, shipped, h0_available, excess_pp):
        return "h0" if (h0_available and excess_pp > 1.0) else "model"
    c = _shipped()
    got = {gated_serve_source("arrival_week", c, True, x) for x in SWEEP}
    assert len(got) > 1, "the sweep must separate a gated implementation from an ungated one"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"  PASS {name}")
    print(f"{n} passed")
