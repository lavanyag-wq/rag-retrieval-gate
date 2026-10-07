"""Properties of the sign test and the resolution arithmetic."""

import pytest

from rag_retrieval_gate.errors import UsageError
from rag_retrieval_gate.stats import (
    binom_tail,
    min_detectable_worse,
    required_set_size,
    resolution_for,
    sign_test_p_value,
)


def test_binom_tail_at_k_zero_is_one():
    assert binom_tail(10, 0) == pytest.approx(1.0)


def test_binom_tail_of_all_successes_is_half_to_the_n():
    assert binom_tail(5, 5) == pytest.approx(0.5**5)


def test_binom_tail_matches_a_hand_computed_value():
    # P[X >= 2] for X ~ Bin(3, 0.5) = (3 + 1) / 8
    assert binom_tail(3, 2) == pytest.approx(0.5)


def test_binom_tail_is_monotone_decreasing_in_k():
    values = [binom_tail(20, k) for k in range(21)]
    assert values == sorted(values, reverse=True)


def test_binom_tail_refuses_k_above_n():
    with pytest.raises(UsageError):
        binom_tail(3, 4)


def test_binom_tail_refuses_p_outside_the_unit_interval():
    with pytest.raises(UsageError):
        binom_tail(3, 1, p=1.5)


def test_sign_test_with_no_moved_queries_is_p_one():
    assert sign_test_p_value(0, 0) == 1.0


def test_sign_test_five_worse_zero_better_is_significant_at_five_percent():
    assert sign_test_p_value(5, 0) == pytest.approx(0.03125)


def test_sign_test_four_worse_zero_better_is_not_significant_at_five_percent():
    assert sign_test_p_value(4, 0) == pytest.approx(0.0625)


def test_sign_test_is_one_sided_so_improvements_do_not_look_like_regressions():
    assert sign_test_p_value(0, 10) == pytest.approx(1.0)


def test_sign_test_refuses_negative_counts():
    with pytest.raises(UsageError):
        sign_test_p_value(-1, 0)


def test_minimum_detectable_flip_count_is_five_at_the_default_alpha():
    # 0.5**4 = 0.0625 > 0.05 >= 0.5**5 = 0.03125. This single integer is the
    # README's headline: below five one-direction flips, no verdict is possible.
    assert min_detectable_worse(100) == 5
    assert min_detectable_worse(1000) == 5


def test_a_stricter_alpha_needs_more_flips():
    assert min_detectable_worse(100, alpha=0.01) == 7


def test_a_set_smaller_than_the_flip_threshold_can_never_reach_significance():
    assert min_detectable_worse(3) == 4  # n_queries + 1 signals "unreachable"


def test_min_detectable_worse_refuses_a_nonpositive_set():
    with pytest.raises(UsageError):
        min_detectable_worse(0)


def test_min_detectable_worse_refuses_alpha_outside_the_unit_interval():
    with pytest.raises(UsageError):
        min_detectable_worse(10, alpha=1.0)


def test_required_set_size_shrinks_as_the_regression_gets_larger():
    small_effect = required_set_size(0.1)
    large_effect = required_set_size(0.5)
    assert large_effect < small_effect


def test_required_set_size_for_a_total_regression_is_the_flip_threshold():
    # If every query regresses, the set only needs to be big enough to flip
    # min_detectable_worse queries.
    assert required_set_size(1.0) == 5


def test_required_set_size_matches_direct_binomial_arithmetic_for_ten_percent():
    n = required_set_size(0.1, alpha=0.05, power=0.8)
    threshold = min_detectable_worse(n)
    assert binom_tail(n, threshold, 0.1) >= 0.8
    prev_threshold = min_detectable_worse(n - 1)
    assert (prev_threshold > n - 1) or (binom_tail(n - 1, prev_threshold, 0.1) < 0.8)


def test_required_set_size_refuses_a_zero_regression_rate():
    with pytest.raises(UsageError):
        required_set_size(0.0)


def test_required_set_size_refuses_power_outside_the_unit_interval():
    with pytest.raises(UsageError):
        required_set_size(0.1, power=1.0)


def test_resolution_floor_share_is_the_threshold_over_the_set_size():
    res = resolution_for(40)
    assert res.min_detectable_worse == 5
    assert res.floor_share == pytest.approx(0.125)
