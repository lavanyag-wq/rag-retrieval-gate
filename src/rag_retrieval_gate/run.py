"""One retrieval run: embed the corpus, retrieve per query, score per query.

A run is a pure function of (corpus, embedder version, k), and its JSON output
contains everything the gate later needs, including the ranked lists. Keeping
the ranked lists is a deliberate storage cost: the gate's attribution story
(which queries lost which documents) is only possible because the run file
remembers more than its own averages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Corpus
from .embedder import HashingEmbedder
from .errors import DataError, UsageError
from .index import ExactIndex, VectorIndex
from .metrics import METRICS


@dataclass
class QueryResult:
    query_id: str
    style: str
    ranked_doc_ids: list[str]
    scores: dict[str, float]  # metric name -> value


@dataclass
class RunResult:
    corpus_seed: int
    embedder_version: str
    k: int
    dim: int = 256
    results: list[QueryResult] = field(default_factory=list)

    @property
    def per_query(self) -> dict[str, QueryResult]:
        return {r.query_id: r for r in self.results}

    def mean(self, metric: str) -> float:
        if metric not in METRICS:
            raise UsageError(f"unknown metric {metric!r}, expected one of {sorted(METRICS)}")
        if not self.results:
            raise UsageError("run has no results; a mean over nothing is not 0")
        return sum(r.scores[metric] for r in self.results) / len(self.results)

    def to_json(self) -> dict:
        return {
            "corpus_seed": self.corpus_seed,
            "embedder_version": self.embedder_version,
            "k": self.k,
            "dim": self.dim,
            "results": [vars(r) for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> RunResult:
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError as exc:
            raise DataError(f"run file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise DataError(f"run file is not valid JSON: {path}") from exc
        try:
            return cls(
                corpus_seed=raw["corpus_seed"],
                embedder_version=raw["embedder_version"],
                k=raw["k"],
                dim=raw.get("dim", 256),
                results=[QueryResult(**r) for r in raw["results"]],
            )
        except (KeyError, TypeError) as exc:
            raise DataError(f"run file has an unexpected shape: {exc}") from exc


def execute_run(corpus: Corpus, embedder_version: str, k: int,
                index: VectorIndex | None = None, dim: int = 256) -> RunResult:
    """Execute a full retrieval run. Deterministic for a given (corpus, version, k, dim)."""
    if k < 1:
        raise UsageError(f"k must be >= 1, got {k}")
    if not corpus.queries:
        raise UsageError("corpus has no queries; refusing to produce an empty run")
    embedder = HashingEmbedder(version=embedder_version, dim=dim)
    if index is None:
        index = ExactIndex(dim=embedder.dim)
    doc_ids = [d.doc_id for d in corpus.documents]
    index.add(doc_ids, embedder.embed_many([d.text for d in corpus.documents]))

    run = RunResult(corpus_seed=corpus.seed, embedder_version=embedder_version, k=k, dim=dim)
    qrels = corpus.qrels
    for query in corpus.queries:
        ranked = [doc_id for doc_id, _score in index.search(embedder.embed(query.text), k)]
        relevant = qrels[query.query_id]
        scores = {name: fn(ranked, relevant, k) for name, fn in METRICS.items()}
        run.results.append(QueryResult(query.query_id, query.style, ranked, scores))
    return run
