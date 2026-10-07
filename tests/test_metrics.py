"""Properties of the retrieval metrics, checked against hand-computed values."""

import math

import pytest

from rag_retrieval_gate.errors import UsageError
from rag_retrieval_gate.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall_is_the_fraction_of_the_relevant_set_in_the_top_k():
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=3) == 0.5


def test_recall_ignores_relevant_documents_below_rank_k():
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_perfect_recall_when_all_relevant_documents_are_retrieved():
    assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0


def test_mrr_is_the_reciprocal_rank_of_the_first_hit():
    assert mrr_at_k(["x", "a", "b"], {"a", "b"}, k=3) == 0.5


def test_mrr_is_zero_when_nothing_relevant_is_in_the_top_k():
    assert mrr_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_mrr_at_rank_one_is_one():
    assert mrr_at_k(["a"], {"a"}, k=1) == 1.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_matches_a_hand_computed_value_for_a_swapped_ranking():
    # relevant at ranks 1 and 3: dcg = 1/log2(2) + 1/log2(4) = 1.5
    # ideal (2 relevant): idcg = 1/log2(2) + 1/log2(3)
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(expected)


def test_ndcg_discounts_a_hit_more_the_lower_it_ranks():
    high = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    low = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert high > low


def test_ndcg_ideal_accounts_for_k_smaller_than_the_relevant_set():
    # With k=1 and two relevant documents, retrieving one of them is ideal.
    assert ndcg_at_k(["a"], {"a", "b"}, k=1) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_every_metric_refuses_an_empty_relevant_set(metric):
    with pytest.raises(UsageError, match="0/0"):
        metric(["a"], set(), k=1)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_every_metric_refuses_k_below_one(metric):
    with pytest.raises(UsageError, match="k must be"):
        metric(["a"], {"a"}, k=0)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_every_metric_refuses_a_ranked_list_with_duplicates(metric):
    with pytest.raises(UsageError, match="double-count"):
        metric(["a", "a"], {"a"}, k=2)


@pytest.mark.parametrize("metric", [recall_at_k, mrr_at_k, ndcg_at_k])
def test_every_metric_is_bounded_by_zero_and_one(metric):
    value = metric(["x", "a", "y", "b"], {"a", "b", "c"}, k=4)
    assert 0.0 <= value <= 1.0
