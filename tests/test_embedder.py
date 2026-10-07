"""Properties of the deterministic embedders."""

import subprocess
import sys

import numpy as np
import pytest

from rag_retrieval_gate.embedder import HashingEmbedder, _char_trigrams, _words
from rag_retrieval_gate.errors import UsageError


def test_the_same_text_embeds_to_the_same_vector_within_a_process():
    emb = HashingEmbedder("v1")
    assert np.array_equal(emb.embed("spark shuffle skew"), emb.embed("spark shuffle skew"))


def test_embeddings_are_stable_across_python_processes():
    # Python's builtin hash() is salted per process; blake2b is not. This test
    # fails if anyone swaps the hash back, which would silently break every
    # cross-run comparison the gate performs.
    code = (
        "from rag_retrieval_gate.embedder import HashingEmbedder;"
        "print(HashingEmbedder('v1').embed('spark shuffle skew').tobytes().hex())"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=True).stdout
        for _ in range(2)
    }
    assert len(outs) == 1


def test_embeddings_are_unit_length_when_nonzero():
    emb = HashingEmbedder("v2")
    vec = emb.embed("delta lake compaction policy")
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_empty_text_embeds_to_the_zero_vector_not_nan():
    vec = HashingEmbedder("v1").embed("")
    assert np.array_equal(vec, np.zeros(256))


def test_token_order_does_not_change_a_v1_embedding():
    emb = HashingEmbedder("v1")
    assert np.allclose(emb.embed("alpha beta gamma"), emb.embed("gamma beta alpha"))


def test_case_and_punctuation_do_not_change_an_embedding():
    emb = HashingEmbedder("v1")
    assert np.allclose(emb.embed("Delta-Lake, compaction!"), emb.embed("delta lake compaction"))


def test_v1_and_v2_disagree_on_the_same_text():
    text = "feature store ingestion parity"
    assert not np.allclose(HashingEmbedder("v1").embed(text), HashingEmbedder("v2").embed(text))


def test_v2_scores_a_partial_token_match_that_v1_cannot_see():
    # "ingest" shares trigrams with "ingestion" but is a different word token,
    # so v1 gives the pair zero similarity and v2 gives it some. This is the
    # mechanism by which a version bump changes retrieval behaviour.
    a, b = "ingest", "ingestion"
    v1 = HashingEmbedder("v1")
    v2 = HashingEmbedder("v2")
    sim_v1 = float(v1.embed(a) @ v1.embed(b))
    sim_v2 = float(v2.embed(a) @ v2.embed(b))
    assert sim_v1 == 0.0
    assert sim_v2 > 0.0


def test_an_unknown_version_is_refused():
    with pytest.raises(UsageError, match="unknown embedder version"):
        HashingEmbedder("v3")


def test_a_dim_below_eight_is_refused():
    with pytest.raises(UsageError, match="dim must be"):
        HashingEmbedder("v1", dim=4)


def test_non_string_input_is_refused_with_the_offending_type_named():
    with pytest.raises(UsageError, match="bytes"):
        HashingEmbedder("v1").embed(b"raw")


def test_embed_many_stacks_in_input_order():
    emb = HashingEmbedder("v1")
    texts = ["one", "two", "three"]
    stacked = emb.embed_many(texts)
    for i, text in enumerate(texts):
        assert np.array_equal(stacked[i], emb.embed(text))


def test_embed_many_of_nothing_is_an_empty_matrix_with_the_right_width():
    out = HashingEmbedder("v1", dim=64).embed_many([])
    assert out.shape == (0, 64)


def test_tokenizer_splits_on_every_non_alphanumeric_character():
    assert _words("a-b_c.d e") == ["a", "b", "c", "d", "e"]


def test_trigrams_cover_the_normalized_text_including_spaces():
    assert _char_trigrams("ab cd") == ["ab ", "b c", " cd"]


def test_lower_dimensions_collide_more_than_higher_ones():
    # The regression scenario in the experiments depends on this monotonicity:
    # distinct tokens share buckets more often at dim=32 than at dim=256.
    words = [f"token{i}" for i in range(200)]

    def collisions(dim: int) -> int:
        emb = HashingEmbedder("v1", dim=dim)
        buckets = [int(np.argmax(np.abs(emb.embed(w)))) for w in words]
        return len(words) - len(set(buckets))

    assert collisions(32) > collisions(256)
