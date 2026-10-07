"""Properties of the CLI: exit codes are the contract, so they are pinned here."""

import pytest

from rag_retrieval_gate.cli import main
from rag_retrieval_gate.errors import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_UNDERPOWERED,
)


@pytest.fixture()
def corpus_path(tmp_path):
    path = tmp_path / "corpus.json"
    assert main(["generate", "--seed", "7", "--out", str(path)]) == EXIT_OK
    return path


def _run(corpus_path, tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    code = main(["run", "--corpus", str(corpus_path), "--out", str(out), *extra])
    assert code == EXIT_OK
    return out


def test_generate_writes_a_corpus_and_reports_its_counts(tmp_path, capsys):
    path = tmp_path / "c.json"
    assert main(["generate", "--seed", "7", "--out", str(path)]) == EXIT_OK
    assert path.exists()
    # 12 topics x (canonical + paraphrase + 2 distractors) + 10 fillers
    assert "58 documents, 36 queries" in capsys.readouterr().out


def test_generate_refuses_invalid_arguments_with_exit_three(tmp_path, capsys):
    code = main(["generate", "--topics", "0", "--out", str(tmp_path / "c.json")])
    assert code == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_run_reports_the_mean_recall_on_stdout(corpus_path, tmp_path, capsys):
    _run(corpus_path, tmp_path, "base")
    assert "mean recall@5" in capsys.readouterr().out


def test_run_against_a_missing_corpus_exits_three(tmp_path, capsys):
    code = main(["run", "--corpus", str(tmp_path / "ghost.json"),
                 "--out", str(tmp_path / "r.json")])
    assert code == EXIT_ERROR


def test_identical_runs_gate_to_exit_zero(corpus_path, tmp_path):
    base = _run(corpus_path, tmp_path, "base")
    code = main(["gate", "--baseline", str(base), "--candidate", str(base)])
    assert code == EXIT_OK


def test_a_collapsed_dimension_candidate_gates_to_exit_one(corpus_path, tmp_path, capsys):
    base = _run(corpus_path, tmp_path, "base")
    cand = _run(corpus_path, tmp_path, "cand", "--dim", "64")
    code = main(["gate", "--baseline", str(base), "--candidate", str(cand)])
    assert code == EXIT_REGRESSION
    assert "verdict: REGRESSION" in capsys.readouterr().out


def test_a_small_set_gates_to_exit_two_not_zero(tmp_path, capsys):
    corpus = tmp_path / "small.json"
    main(["generate", "--seed", "7", "--topics", "4", "--out", str(corpus)])
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    main(["run", "--corpus", str(corpus), "--out", str(base)])
    main(["run", "--corpus", str(corpus), "--embedder", "v2", "--out", str(cand)])
    code = main(["gate", "--baseline", str(base), "--candidate", str(cand),
                 "--metric", "ndcg"])
    assert code == EXIT_UNDERPOWERED
    assert "verdict: UNDERPOWERED" in capsys.readouterr().out


def test_gate_writes_the_full_report_when_out_is_given(corpus_path, tmp_path):
    base = _run(corpus_path, tmp_path, "base")
    cand = _run(corpus_path, tmp_path, "cand", "--embedder", "v2")
    report = tmp_path / "report.json"
    main(["gate", "--baseline", str(base), "--candidate", str(cand),
          "--out", str(report)])
    assert report.exists()


def test_gate_summary_includes_the_resolution_sentence(corpus_path, tmp_path, capsys):
    base = _run(corpus_path, tmp_path, "base")
    main(["gate", "--baseline", str(base), "--candidate", str(base)])
    assert "cannot call a regression" in capsys.readouterr().out


def test_gate_summary_breaks_transitions_down_by_query_style(corpus_path, tmp_path, capsys):
    base = _run(corpus_path, tmp_path, "base")
    cand = _run(corpus_path, tmp_path, "cand", "--embedder", "v2")
    main(["gate", "--baseline", str(base), "--candidate", str(cand)])
    assert "style acronym" in capsys.readouterr().out


def test_gate_verbose_prints_json(corpus_path, tmp_path, capsys):
    base = _run(corpus_path, tmp_path, "base")
    code = main(["gate", "--baseline", str(base), "--candidate", str(base), "--verbose"])
    assert code == EXIT_OK
    assert '"verdict"' in capsys.readouterr().out


def test_gate_on_mismatched_runs_exits_three(tmp_path, capsys):
    c1 = tmp_path / "c1.json"
    c2 = tmp_path / "c2.json"
    main(["generate", "--seed", "1", "--out", str(c1)])
    main(["generate", "--seed", "2", "--out", str(c2)])
    r1, r2 = tmp_path / "r1.json", tmp_path / "r2.json"
    main(["run", "--corpus", str(c1), "--out", str(r1)])
    main(["run", "--corpus", str(c2), "--out", str(r2)])
    assert main(["gate", "--baseline", str(r1), "--candidate", str(r2)]) == EXIT_ERROR
    assert "different corpora" in capsys.readouterr().err


def test_resolution_prints_the_flip_floor_for_a_set_size(capsys):
    assert main(["resolution", "--queries", "36"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "36-query golden set" in out
    assert "5 one-direction flips" in out


def test_resolution_with_a_target_rate_prints_the_required_set_size(capsys):
    assert main(["resolution", "--queries", "36", "--target-rate", "0.1"]) == EXIT_OK
    assert "the set needs" in capsys.readouterr().out


def test_resolution_refuses_a_nonpositive_query_count(capsys):
    assert main(["resolution", "--queries", "0"]) == EXIT_ERROR
