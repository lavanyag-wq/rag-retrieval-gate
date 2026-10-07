"""Properties of the corpus generator and the corpus file format."""

import json

import pytest

from rag_retrieval_gate.corpus import QUERY_STYLES, Corpus, generate_corpus
from rag_retrieval_gate.errors import DataError, UsageError


def test_the_same_seed_produces_byte_identical_corpora():
    a = generate_corpus(7).to_json()
    b = generate_corpus(7).to_json()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_different_seeds_produce_different_document_text():
    a = generate_corpus(7)
    b = generate_corpus(8)
    assert [d.text for d in a.documents] != [d.text for d in b.documents]


def test_every_query_has_exactly_two_relevant_documents():
    corpus = generate_corpus(7)
    assert all(len(q.relevant_doc_ids) == 2 for q in corpus.queries)


def test_every_relevant_doc_id_exists_in_the_document_set():
    corpus = generate_corpus(7)
    doc_ids = {d.doc_id for d in corpus.documents}
    for q in corpus.queries:
        assert set(q.relevant_doc_ids) <= doc_ids


def test_each_topic_contributes_one_query_per_style():
    corpus = generate_corpus(7, n_topics=3)
    styles = sorted(q.style for q in corpus.queries)
    assert styles == sorted(list(QUERY_STYLES) * 3)


def test_distractor_documents_are_never_labelled_relevant():
    corpus = generate_corpus(7)
    distractor_ids = {d.doc_id for d in corpus.documents if d.kind == "distractor"}
    for q in corpus.queries:
        assert not distractor_ids & set(q.relevant_doc_ids)


def test_document_counts_follow_the_requested_shape():
    corpus = generate_corpus(7, n_topics=4, distractors_per_topic=3, fillers=5)
    assert len(corpus.documents) == 4 * (2 + 3) + 5
    assert len(corpus.queries) == 4 * 3


def test_meta_records_the_counts_the_file_actually_contains():
    corpus = generate_corpus(7)
    assert corpus.meta["n_documents"] == len(corpus.documents)
    assert corpus.meta["n_queries"] == len(corpus.queries)


def test_zero_topics_is_refused_rather_than_clamped():
    with pytest.raises(UsageError):
        generate_corpus(7, n_topics=0)


def test_more_topics_than_templates_is_refused():
    with pytest.raises(UsageError):
        generate_corpus(7, n_topics=999)


def test_negative_distractor_count_is_refused():
    with pytest.raises(UsageError):
        generate_corpus(7, distractors_per_topic=-1)


def test_negative_filler_count_is_refused():
    with pytest.raises(UsageError):
        generate_corpus(7, fillers=-1)


def test_a_saved_corpus_loads_back_equal(tmp_path):
    corpus = generate_corpus(7)
    path = tmp_path / "corpus.json"
    corpus.save(path)
    loaded = Corpus.load(path)
    assert loaded.to_json() == corpus.to_json()


def test_loading_a_missing_file_raises_data_error(tmp_path):
    with pytest.raises(DataError, match="not found"):
        Corpus.load(tmp_path / "nope.json")


def test_loading_invalid_json_raises_data_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(DataError, match="not valid JSON"):
        Corpus.load(path)


def test_loading_a_wrong_shape_raises_data_error(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"seed": 1, "documents": [{"nope": 1}], "queries": []}))
    with pytest.raises(DataError, match="unexpected shape"):
        Corpus.load(path)


def test_a_label_pointing_at_a_missing_document_is_refused_at_load(tmp_path):
    corpus = generate_corpus(7, n_topics=1)
    raw = corpus.to_json()
    raw["queries"][0]["relevant_doc_ids"] = ["ghost-doc"]
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(DataError, match="do not exist"):
        Corpus.load(path)


def test_qrels_maps_every_query_to_its_label_set():
    corpus = generate_corpus(7, n_topics=2)
    qrels = corpus.qrels
    assert set(qrels) == {q.query_id for q in corpus.queries}
    for q in corpus.queries:
        assert qrels[q.query_id] == set(q.relevant_doc_ids)


def test_paraphrase_documents_mention_the_acronym_so_acronym_queries_have_a_target():
    corpus = generate_corpus(7, n_topics=2)
    paraphrases = [d for d in corpus.documents if d.kind == "paraphrase"]
    assert all("(" in d.text and ")" in d.text for d in paraphrases)
