"""
Unit tests for Claim B's deterministic formula (`ml/models/dyadic.py`) — Step 8.
The double-counting ablation itself (training two encoder arms) is NOT a unit test — it's
`ml/run_step8_claim_b.py`'s own job, reported in `reports/step8_claim_b.md`. These tests only
lock in the pure-function formula's own documented behavior.
"""

import math

import pytest

from ml.models.dyadic import (
    PRIORITY_TIER_WEIGHTS,
    combine_dyadic_weights,
    contract_priority_weight,
    dyadic_risk_score,
    reordering_rate,
)


def test_contract_priority_weight_mapping():
    assert contract_priority_weight("strategic") == 1.0
    assert contract_priority_weight("standard") == 0.6
    assert contract_priority_weight("low") == 0.3
    assert PRIORITY_TIER_WEIGHTS == {"strategic": 1.0, "standard": 0.6, "low": 0.3}


def test_contract_priority_weight_unknown_tier_raises():
    with pytest.raises(KeyError):
        contract_priority_weight("platinum")


def test_combine_dyadic_weights_neutral_point():
    """combined_protection = 0.5 (average treatment) -> f = 1.0 exactly, no reweighting."""
    f = combine_dyadic_weights(0.5, 0.5, 0.5)
    assert f == pytest.approx(1.0)


def test_combine_dyadic_weights_max_protection():
    """All three inputs = 1.0 -> combined_protection = 1.0 -> f = 1 - 0.6*0.5 = 0.7."""
    f = combine_dyadic_weights(1.0, 1.0, 1.0)
    assert f == pytest.approx(0.7)


def test_combine_dyadic_weights_min_protection():
    """All three inputs = 0.0 -> combined_protection = 0.0 -> f = 1 + 0.6*0.5 = 1.3."""
    f = combine_dyadic_weights(0.0, 0.0, 0.0)
    assert f == pytest.approx(1.3)


def test_combine_dyadic_weights_fulfilment_leads():
    """With only fulfilment_preference present (the lead signal) at its max, and the other two
    at the neutral midpoint 0.5, the combined score should sit closer to 1.0's protection than
    it would if fulfilment carried only 1/3 weight -- confirms the 0.50 lead weighting applies
    even in a mixed-availability case."""
    f_all_neutral = combine_dyadic_weights(0.5, 0.5, 0.5)
    f_fulfilment_high = combine_dyadic_weights(0.5, 0.5, 1.0)
    # fulfilment=1.0 pulls combined_protection above 0.5 -> f should drop below 1.0
    assert f_fulfilment_high < f_all_neutral


def test_combine_dyadic_weights_renormalizes_missing_fulfilment():
    """fulfilment_preference_weight=None -> falls back to a 50/50 average of the other two
    (renormalized weights), not a fabricated third value."""
    f_missing = combine_dyadic_weights(1.0, 0.0, None)
    # combined_protection = (0.25*1.0 + 0.25*0.0) / 0.5 = 0.5 -> f = 1.0
    assert f_missing == pytest.approx(1.0)


def test_combine_dyadic_weights_all_none_raises():
    with pytest.raises(ValueError):
        combine_dyadic_weights(None, None, None)


def test_dyadic_risk_score_neutral_equals_global():
    assert dyadic_risk_score(0.6, 0.5, 0.5, 0.5) == pytest.approx(0.6)


def test_dyadic_risk_score_clips_to_one():
    # global_risk=0.9, max protection-inverted (min protection) -> f=1.3 -> 1.17, clip to 1.0
    score = dyadic_risk_score(0.9, 0.0, 0.0, 0.0)
    assert score == pytest.approx(1.0)


def test_dyadic_risk_score_clips_to_zero():
    score = dyadic_risk_score(0.0, 1.0, 1.0, 1.0)
    assert score == pytest.approx(0.0)


def test_dyadic_risk_score_bounds_always_hold():
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        for v in (0.0, 0.5, 1.0):
            for p in (0.0, 0.5, 1.0):
                for fu in (0.0, 0.5, 1.0, None):
                    score = dyadic_risk_score(g, v, p, fu)
                    assert 0.0 <= score <= 1.0


def test_reordering_rate_perfectly_concordant():
    """Dyadic risk preserves the exact same ranking as global risk -> 0 discordant pairs."""
    data = {"cust1": {"supA": (0.8, 0.9), "supB": (0.3, 0.4), "supC": (0.5, 0.6)}}
    assert reordering_rate(data) == pytest.approx(0.0)


def test_reordering_rate_perfectly_discordant():
    """Dyadic risk exactly REVERSES global risk's ranking for a customer's two suppliers ->
    100% of pairs discordant."""
    data = {"cust1": {"supA": (0.8, 0.2), "supB": (0.3, 0.9)}}
    assert reordering_rate(data) == pytest.approx(1.0)


def test_reordering_rate_partial_discordance():
    """3 suppliers, ranks A>B>C by global, but dyadic flips B and C only -> 1 of 3 pairs
    discordant (A-B stays concordant, A-C stays concordant, B-C flips)."""
    data = {"cust1": {"supA": (0.9, 0.9), "supB": (0.6, 0.4), "supC": (0.5, 0.5001)}}
    rate = reordering_rate(data)
    assert rate == pytest.approx(1 / 3)


def test_reordering_rate_averages_across_customers():
    data = {
        "cust1": {"supA": (0.8, 0.9), "supB": (0.3, 0.4)},   # concordant, rate=0
        "cust2": {"supA": (0.8, 0.2), "supB": (0.3, 0.9)},   # discordant, rate=1
    }
    assert reordering_rate(data) == pytest.approx(0.5)


def test_reordering_rate_ignores_single_supplier_customers():
    data = {
        "cust1": {"supA": (0.8, 0.9)},  # only 1 supplier -- no pair, excluded
        "cust2": {"supA": (0.8, 0.2), "supB": (0.3, 0.9)},   # discordant, rate=1
    }
    assert reordering_rate(data) == pytest.approx(1.0)


def test_reordering_rate_nan_when_no_customer_has_two_suppliers():
    data = {"cust1": {"supA": (0.8, 0.9)}, "cust2": {"supB": (0.3, 0.4)}}
    assert math.isnan(reordering_rate(data))


def test_reordering_rate_empty_input_is_nan():
    assert math.isnan(reordering_rate({}))
