"""Properties of the exact index, including the one a shuffled corpus exposed."""

import numpy as np
import pytest

from rag_retrieval_gate.errors import UsageError
from rag_retrieval_gate.index import ExactIndex


def _unit(v):
    arr = np.asarray(v, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def test_search_returns_results_sorted_by_descending_score():
    idx = ExactIndex(dim=2)
    idx.add(["a", "b", "c"], np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]))
    results = idx.search(_unit([1, 0]), k=3)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_equal_scores_are_ordered_by_doc_id_not_by_insertion_order():
    # Regression pin. np.argsort orders ties by position, so before the explicit
    # tie-break, shuffling the corpus file reordered tied documents and moved
    # recall@k by a point on real runs. Same input set, two insertion orders,
    # one required output.
    vec = _unit([1, 1])
    forward = ExactIndex(dim=2)
    forward.add(["a", "b"], np.stack([vec, vec]))
    backward = ExactIndex(dim=2)
    backward.add(["b", "a"], np.stack([vec, vec]))
    query = _unit([1, 0])
    assert forward.search(query, k=2) == backward.search(query, k=2)
    assert [d for d, _ in forward.search(query, k=2)] == ["a", "b"]


def test_k_larger_than_the_index_returns_everything_without_error():
    idx = ExactIndex(dim=2)
    idx.add(["a"], _unit([1, 0]).reshape(1, 2))
    assert len(idx.search(_unit([1, 0]), k=10)) == 1


def test_searching_an_empty_index_returns_an_empty_list():
    assert ExactIndex(dim=2).search(_unit([1, 0]), k=3) == []


def test_len_reports_the_number_of_documents_added():
    idx = ExactIndex(dim=2)
    idx.add(["a", "b"], np.stack([_unit([1, 0]), _unit([0, 1])]))
    assert len(idx) == 2


def test_adding_in_two_batches_equals_adding_in_one():
    vecs = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    one = ExactIndex(dim=2)
    one.add(["a", "b", "c"], vecs)
    two = ExactIndex(dim=2)
    two.add(["a"], vecs[:1])
    two.add(["b", "c"], vecs[1:])
    query = _unit([1, 2])
    assert one.search(query, k=3) == two.search(query, k=3)


def test_duplicate_doc_ids_are_refused():
    idx = ExactIndex(dim=2)
    idx.add(["a"], _unit([1, 0]).reshape(1, 2))
    with pytest.raises(UsageError, match="duplicate"):
        idx.add(["a"], _unit([0, 1]).reshape(1, 2))


def test_mismatched_id_and_vector_counts_are_refused():
    idx = ExactIndex(dim=2)
    with pytest.raises(UsageError, match="refusing a silent zip"):
        idx.add(["a", "b"], _unit([1, 0]).reshape(1, 2))


def test_vectors_of_the_wrong_width_are_refused():
    idx = ExactIndex(dim=3)
    with pytest.raises(UsageError, match="must be"):
        idx.add(["a"], _unit([1, 0]).reshape(1, 2))


def test_a_query_vector_of_the_wrong_shape_is_refused():
    idx = ExactIndex(dim=3)
    with pytest.raises(UsageError, match="query vector"):
        idx.search(np.zeros(2), k=1)


def test_k_below_one_is_refused():
    idx = ExactIndex(dim=2)
    with pytest.raises(UsageError, match="k must be"):
        idx.search(_unit([1, 0]), k=0)


def test_dim_below_one_is_refused():
    with pytest.raises(UsageError, match="dim must be"):
        ExactIndex(dim=0)


def test_scores_are_cosine_similarities_for_unit_vectors():
    idx = ExactIndex(dim=2)
    idx.add(["a"], _unit([1, 0]).reshape(1, 2))
    [(doc, score)] = idx.search(_unit([1, 1]), k=1)
    assert doc == "a"
    assert score == pytest.approx(1 / np.sqrt(2))
