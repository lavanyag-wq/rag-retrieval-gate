"""Seeded corpus generator: documents, queries, and labelled relevance (qrels).

A retrieval metric is only as believable as the set it was measured on, so this
module is explicit about what it generates. Each topic produces one canonical
document, a paraphrased near-duplicate, and off-topic distractors that share
surface vocabulary with the topic. Queries are phrased three ways: direct,
paraphrased, and acronym-only, which is the case hashing embedders are worst
at. The generator is fully determined by its seed, so every number downstream
can be reproduced from one command with no committed binary blob.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DataError, UsageError

# Topic templates: (name, acronym, keywords, distractor_keywords).
# The distractor keywords overlap the topic's surface vocabulary on purpose:
# a corpus of obviously-easy cases measures nothing.
_TOPICS: list[tuple[str, str, list[str], list[str]]] = [
    ("feature store ingestion", "FSI",
     ["feature", "store", "ingestion", "offline", "online", "parity"],
     ["feature", "flag", "rollout", "toggle"]),
    ("churn prediction model", "CPM",
     ["churn", "retention", "subscriber", "classifier", "label", "window"],
     ["butter", "churn", "dairy", "recipe"]),
    ("ab test peeking", "ATP",
     ["experiment", "peeking", "interim", "alpha", "sequential", "stopping"],
     ["peak", "mountain", "trail", "altitude"]),
    ("delta lake compaction", "DLC",
     ["delta", "compaction", "small", "files", "optimize", "vacuum"],
     ["river", "delta", "sediment", "estuary"]),
    ("vector index recall", "VIR", ["vector", "index", "recall", "ann", "exact", "cosine"],
     ["vector", "disease", "mosquito", "transmission"]),
    ("spark shuffle skew", "SSS",
     ["spark", "shuffle", "skew", "partition", "salting", "executor"],
     ["spark", "plug", "engine", "ignition"]),
    ("model drift monitor", "MDM",
     ["drift", "monitor", "distribution", "baseline", "alert", "psi"],
     ["continental", "drift", "plate", "tectonics"]),
    ("prompt cache hits", "PCH", ["prompt", "cache", "hit", "ratio", "eviction", "ttl"],
     ["cache", "geocache", "coordinates", "hiking"]),
    ("embedding version pin", "EVP",
     ["embedding", "version", "pin", "reindex", "migration", "dimension"],
     ["pin", "bowling", "lane", "strike"]),
    ("retrieval latency budget", "RLB",
     ["retrieval", "latency", "budget", "percentile", "timeout", "fallback"],
     ["budget", "travel", "hostel", "fare"]),
    ("golden set curation", "GSC",
     ["golden", "set", "curation", "label", "annotator", "agreement"],
     ["golden", "retriever", "dog", "breed"]),
    ("chunking overlap policy", "COP",
     ["chunk", "overlap", "token", "boundary", "window", "split"],
     ["police", "cop", "patrol", "precinct"]),
]

_FILLER = [
    "the team reviewed the rollout notes before the quarterly sync",
    "ownership for the dashboard moved to the platform group",
    "the runbook was updated after the incident review",
    "capacity planning starts from last month's usage report",
    "the migration checklist is tracked in the shared board",
]

QUERY_STYLES = ("direct", "paraphrase", "acronym")


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    topic: str
    kind: str  # canonical | paraphrase | distractor | filler


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    topic: str
    style: str
    relevant_doc_ids: tuple[str, ...]


@dataclass
class Corpus:
    """Documents, queries and relevance labels, plus the metadata that fixes them."""

    seed: int
    documents: list[Document]
    queries: list[Query]
    meta: dict = field(default_factory=dict)

    @property
    def qrels(self) -> dict[str, set[str]]:
        return {q.query_id: set(q.relevant_doc_ids) for q in self.queries}

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "meta": self.meta,
            "documents": [vars(d) for d in self.documents],
            "queries": [
                {**{k: v for k, v in vars(q).items() if k != "relevant_doc_ids"},
                 "relevant_doc_ids": list(q.relevant_doc_ids)}
                for q in self.queries
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> Corpus:
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError as exc:
            raise DataError(f"corpus file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise DataError(f"corpus file is not valid JSON: {path}") from exc
        try:
            docs = [Document(**d) for d in raw["documents"]]
            queries = [
                Query(
                    query_id=q["query_id"], text=q["text"], topic=q["topic"],
                    style=q["style"], relevant_doc_ids=tuple(q["relevant_doc_ids"]),
                )
                for q in raw["queries"]
            ]
            corpus = cls(seed=raw["seed"], documents=docs, queries=queries,
                         meta=raw.get("meta", {}))
        except (KeyError, TypeError) as exc:
            raise DataError(f"corpus file has an unexpected shape: {exc}") from exc
        doc_ids = {d.doc_id for d in corpus.documents}
        for q in corpus.queries:
            missing = set(q.relevant_doc_ids) - doc_ids
            if missing:
                raise DataError(
                    f"query {q.query_id} labels documents that do not exist: {sorted(missing)}"
                )
        return corpus


def _doc_text(rng: random.Random, keywords: list[str], topic: str) -> str:
    words = list(keywords)
    rng.shuffle(words)
    filler = rng.choice(_FILLER)
    return f"{topic}: " + " ".join(words) + ". " + filler + "."


def generate_corpus(seed: int, n_topics: int = 12, distractors_per_topic: int = 2,
                    fillers: int = 10) -> Corpus:
    """Generate a corpus whose every byte is a function of the seed.

    Raises UsageError rather than clamping silently, because a clamped corpus
    produces numbers that look comparable across runs and are not.
    """
    if n_topics < 1 or n_topics > len(_TOPICS):
        raise UsageError(f"n_topics must be between 1 and {len(_TOPICS)}, got {n_topics}")
    if distractors_per_topic < 0:
        raise UsageError("distractors_per_topic must be >= 0")
    if fillers < 0:
        raise UsageError("fillers must be >= 0")

    rng = random.Random(seed)
    documents: list[Document] = []
    queries: list[Query] = []

    for t_idx, (topic, acronym, keywords, dis_kw) in enumerate(_TOPICS[:n_topics]):
        canon_id = f"d{t_idx:03d}c"
        para_id = f"d{t_idx:03d}p"
        documents.append(Document(canon_id, _doc_text(rng, keywords, topic), topic, "canonical"))
        # The paraphrase keeps the meaning, drops half the exact keywords,
        # and spells the acronym out. It is labelled relevant: a retriever
        # that only matches exact tokens will miss it, which is the point.
        kept = keywords[: max(2, len(keywords) // 2)]
        para_text = (
            f"notes on {topic} ({acronym}): " + " ".join(kept)
            + ". " + rng.choice(_FILLER) + "."
        )
        documents.append(Document(para_id, para_text, topic, "paraphrase"))
        for d in range(distractors_per_topic):
            documents.append(
                Document(
                    f"d{t_idx:03d}x{d}",
                    _doc_text(rng, dis_kw, f"unrelated note {d}"),
                    topic,
                    "distractor",
                )
            )
        relevant = (canon_id, para_id)
        queries.append(
            Query(f"q{t_idx:03d}d", f"how do we handle {topic}", topic, "direct", relevant)
        )
        queries.append(
            Query(f"q{t_idx:03d}p", "guidance about " + " ".join(kept[:2]),
                  topic, "paraphrase", relevant)
        )
        queries.append(Query(f"q{t_idx:03d}a", f"{acronym} status", topic, "acronym", relevant))

    for f_idx in range(fillers):
        documents.append(
            Document(f"f{f_idx:03d}", rng.choice(_FILLER) + f" item {f_idx}.", "filler", "filler")
        )

    meta = {
        "n_topics": n_topics,
        "distractors_per_topic": distractors_per_topic,
        "fillers": fillers,
        "query_styles": list(QUERY_STYLES),
        "n_documents": len(documents),
        "n_queries": len(queries),
    }
    return Corpus(seed=seed, documents=documents, queries=queries, meta=meta)
