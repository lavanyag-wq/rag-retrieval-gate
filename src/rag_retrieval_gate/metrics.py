"""Per-query retrieval metrics, kept per query on purpose.

The gate's core argument is that a mean hides the per-query transitions that
decide whether anyone was harmed, so this module never returns only an average:
every function computes a value per query and aggregation happens at the edge,
visibly. Definitions are the standard ones (recall@k, MRR@k, nDCG@k with binary
relevance and a log2 discount starting at rank 1), written out rather than
imported, because the point of the repository is that the author can defend
every number in it.
"""

from __future__ import annotations

import math

from .errors import UsageError


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set found in the top k."""
    _validate(ranked_doc_ids, relevant, k)
    hits = sum(1 for d in ranked_doc_ids[:k] if d in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant document in the top k, else 0."""
    _validate(ranked_doc_ids, relevant, k)
    for rank, d in enumerate(ranked_doc_ids[:k], start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    """Binary nDCG@k: DCG with gain 1 at relevant ranks, over the ideal DCG."""
    _validate(ranked_doc_ids, relevant, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, d in enumerate(ranked_doc_ids[:k], start=1)
        if d in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def _validate(ranked_doc_ids: list[str], relevant: set[str], k: int) -> None:
    if k < 1:
        raise UsageError(f"k must be >= 1, got {k}")
    if not relevant:
        raise UsageError("relevant set is empty; a metric over it would be 0/0")
    if len(set(ranked_doc_ids)) != len(ranked_doc_ids):
        raise UsageError("ranked list contains duplicates; refusing to double-count a hit")


METRICS = {"recall": recall_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}
