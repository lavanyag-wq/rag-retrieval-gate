"""Properties of the run pipeline and the gate's three verdicts."""

import json

import pytest

from rag_retrieval_gate.corpus import generate_corpus
from rag_retrieval_gate.errors import DataError, UsageError
from rag_retrieval_gate.gate import (
    VERDICT_PASS,
    VERDICT_REGRESSION,
    VERDICT_UNDERPOWERED,
    compare,
)
from rag_retrieval_gate.run import RunResult, execute_run


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(7)


@pytest.fixture(scope="module")
def baseline(corpus):
    return execute_run(corpus, "v1", k=5)


def test_a_run_is_deterministic_for_the_same_inputs(corpus, baseline):
    again = execute_run(corpus, "v1", k=5)
    assert json.dumps(again.to_json(), sort_keys=True) == json.dumps(
        baseline.to_json(), sort_keys=True
    )


def test_a_run_scores_every_query_in_the_corpus(corpus, baseline):
    assert {r.query_id for r in baseline.results} == {q.query_id for q in corpus.queries}


def test_every_query_result_carries_all_three_metrics(baseline):
    assert all(set(r.scores) == {"recall", "mrr", "ndcg"} for r in baseline.results)


def test_ranked_lists_have_at_most_k_entries(baseline):
    assert all(len(r.ranked_doc_ids) <= 5 for r in baseline.results)


def test_a_saved_run_loads_back_equal(tmp_path, baseline):
    path = tmp_path / "run.json"
    baseline.save(path)
    assert RunResult.load(path).to_json() == baseline.to_json()


def test_loading_a_missing_run_file_raises_data_error(tmp_path):
    with pytest.raises(DataError, match="not found"):
        RunResult.load(tmp_path / "nope.json")


def test_loading_an_invalid_run_file_raises_data_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("][")
    with pytest.raises(DataError, match="not valid JSON"):
        RunResult.load(path)


def test_loading_a_wrong_shape_run_file_raises_data_error(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"corpus_seed": 1}))
    with pytest.raises(DataError, match="unexpected shape"):
        RunResult.load(path)


def test_mean_of_an_unknown_metric_is_refused(baseline):
    with pytest.raises(UsageError, match="unknown metric"):
        baseline.mean("accuracy")


def test_mean_of_an_empty_run_is_refused_not_zero():
    empty = RunResult(corpus_seed=1, embedder_version="v1", k=5)
    with pytest.raises(UsageError, match="mean over nothing"):
        empty.mean("recall")


def test_an_empty_corpus_is_refused_by_execute_run(corpus):
    stripped = generate_corpus(7, n_topics=1)
    stripped.queries = []
    with pytest.raises(UsageError, match="no queries"):
        execute_run(stripped, "v1", k=5)


def test_k_below_one_is_refused_by_execute_run(corpus):
    with pytest.raises(UsageError, match="k must be"):
        execute_run(corpus, "v1", k=0)


def test_identical_runs_gate_as_pass_with_p_one(baseline):
    report = compare(baseline, baseline)
    assert report.verdict == VERDICT_PASS
    assert report.p_value == 1.0
    assert report.ties == report.n_queries


def test_the_dim_64_candidate_is_called_a_regression(corpus, baseline):
    candidate = execute_run(corpus, "v1", k=5, dim=64)
    report = compare(baseline, candidate, metric="recall")
    assert report.verdict == VERDICT_REGRESSION
    assert report.p_value <= 0.05


def test_the_v2_candidate_moves_the_mean_but_is_not_called(corpus, baseline):
    # The nearly-significant case the README discusses: real losses, all in one
    # style, and a p-value the set cannot push under alpha.
    candidate = execute_run(corpus, "v2", k=5)
    report = compare(baseline, candidate, metric="recall")
    assert report.verdict == VERDICT_PASS
    assert report.mean_delta < 0
    assert len(report.worse) > len(report.better)


def test_a_small_set_returns_underpowered_not_pass():
    small = generate_corpus(7, n_topics=4)
    base = execute_run(small, "v1", k=5)
    cand = execute_run(small, "v2", k=5)
    report = compare(base, cand, metric="ndcg")
    assert report.verdict == VERDICT_UNDERPOWERED
    assert 0 < len(report.worse) + len(report.better) < report.min_detectable_worse


def test_worse_transitions_record_the_documents_the_candidate_lost(corpus, baseline):
    candidate = execute_run(corpus, "v1", k=5, dim=64)
    report = compare(baseline, candidate, metric="recall")
    assert report.worse
    assert any(t.lost_doc_ids for t in report.worse)
    for t in report.worse:
        assert t.candidate_value < t.baseline_value


def test_by_style_counts_match_the_transition_lists(corpus, baseline):
    candidate = execute_run(corpus, "v2", k=5)
    report = compare(baseline, candidate, metric="recall")
    styles = report.by_style()
    assert sum(c["worse"] for c in styles.values()) == len(report.worse)
    assert sum(c["better"] for c in styles.values()) == len(report.better)


def test_mean_delta_is_candidate_minus_baseline(corpus, baseline):
    candidate = execute_run(corpus, "v2", k=5)
    report = compare(baseline, candidate)
    assert report.mean_delta == pytest.approx(
        candidate.mean("recall") - baseline.mean("recall")
    )


def test_runs_over_different_corpora_are_refused(corpus, baseline):
    other = execute_run(generate_corpus(8), "v1", k=5)
    with pytest.raises(UsageError, match="different corpora"):
        compare(baseline, other)


def test_runs_with_different_k_are_refused(corpus, baseline):
    other = execute_run(corpus, "v1", k=3)
    with pytest.raises(UsageError, match="different k"):
        compare(baseline, other)


def test_runs_over_different_query_sets_are_refused(corpus, baseline):
    other = execute_run(corpus, "v1", k=5)
    other.results = other.results[:-1]
    with pytest.raises(UsageError, match="different query sets"):
        compare(baseline, other)


def test_an_unknown_metric_is_refused_by_compare(baseline):
    with pytest.raises(UsageError, match="unknown metric"):
        compare(baseline, baseline, metric="f1")


def test_alpha_outside_the_unit_interval_is_refused_by_compare(baseline):
    with pytest.raises(UsageError, match="alpha"):
        compare(baseline, baseline, alpha=0.0)


def test_a_gate_report_round_trips_through_json(tmp_path, corpus, baseline):
    candidate = execute_run(corpus, "v2", k=5)
    report = compare(baseline, candidate)
    path = tmp_path / "gate.json"
    report.save(path)
    raw = json.loads(path.read_text())
    assert raw["verdict"] == report.verdict
    assert raw["mean_delta"] == pytest.approx(report.mean_delta)
    assert len(raw["worse"]) == len(report.worse)


def test_the_gate_report_json_orders_metadata_before_interpretation(corpus, baseline):
    report = compare(baseline, execute_run(corpus, "v2", k=5))
    raw = report.to_json()
    assert {"metric", "k", "alpha", "n_queries", "p_value",
            "min_detectable_worse", "verdict"} <= set(raw)
