"""Vector indexes: a built-in exact index, and a protocol a Chroma adapter fits.

The default index is exact brute-force cosine over numpy, because the thing
under measurement is the gate, and an approximate index would fold its own
recall loss into every number. The adapter in chroma_store.py satisfies the
same protocol for teams that want the gate pointed at a real store; it is an
optional extra precisely so that `make verify` needs nothing beyond numpy.

Ties are broken by doc_id, explicitly. np.argsort on equal scores orders by
position, so without the tie-break the ranking would depend on document
insertion order, and the same corpus would produce different metrics from a
shuffled file. That is not a hypothetical: the test suite pins it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .errors import UsageError


class VectorIndex(Protocol):
    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def __len__(self) -> int: ...


class ExactIndex:
    """Brute-force cosine index. Deterministic, dependency-free, O(n) per query."""

    def __init__(self, dim: int):
        if dim < 1:
            raise UsageError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float64)

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if len(doc_ids) != vectors.shape[0]:
            raise UsageError(
                f"{len(doc_ids)} ids but {vectors.shape[0]} vectors; refusing a silent zip"
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise UsageError(f"vectors must be (n, {self.dim}), got {tuple(vectors.shape)}")
        dupes = set(doc_ids) & set(self._ids)
        if dupes:
            raise UsageError(f"duplicate doc_ids refused: {sorted(dupes)[:3]}")
        self._ids.extend(doc_ids)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float64)])

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k < 1:
            raise UsageError(f"k must be >= 1, got {k}")
        if query_vec.shape != (self.dim,):
            raise UsageError(f"query vector must be ({self.dim},), got {tuple(query_vec.shape)}")
        if not self._ids:
            return []
        scores = self._vectors @ query_vec
        # Sort by (-score, doc_id): the doc_id makes equal scores a total order.
        order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        top = order[: min(k, len(order))]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._ids)
