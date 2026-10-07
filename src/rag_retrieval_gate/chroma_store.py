"""Optional ChromaDB adapter implementing the same VectorIndex protocol.

This module exists so the gate can be pointed at the vector store a team
actually runs, while the default path stays dependency-free and offline. Two
deliberate choices. First, every argument is validated before chromadb is
imported, so a caller with a bad request is told immediately rather than
after a heavy import, and the no-extras install can still reach and test
the validation. Second, the adapter asks Chroma for raw results and re-applies this
package's own tie-break ordering, so switching stores cannot silently switch
ranking semantics.
"""

from __future__ import annotations

import numpy as np

from .errors import UsageError


class ChromaIndex:
    """Adapter around a chromadb in-process collection.

    Requires the 'chroma' extra. Construction validates first and imports second.
    """

    def __init__(self, dim: int, collection_name: str = "rag_gate"):
        if dim < 1:
            raise UsageError(f"dim must be >= 1, got {dim}")
        if not collection_name or not collection_name.replace("_", "").isalnum():
            raise UsageError(
                f"collection_name must be alphanumeric/underscore, got {collection_name!r}"
            )
        self.dim = dim
        try:
            import chromadb  # pragma: no cover (exercised only with the chroma extra)
        except ModuleNotFoundError as exc:
            raise UsageError(
                "chromadb is not installed; install the 'chroma' extra: "
                "pip install 'rag-retrieval-gate[chroma]'"
            ) from exc
        self._client = chromadb.EphemeralClient()  # pragma: no cover
        self._collection = self._client.create_collection(  # pragma: no cover
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        if len(doc_ids) != vectors.shape[0]:
            raise UsageError(
                f"{len(doc_ids)} ids but {vectors.shape[0]} vectors; refusing a silent zip"
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise UsageError(f"vectors must be (n, {self.dim}), got {tuple(vectors.shape)}")
        self._collection.add(ids=doc_ids, embeddings=vectors.tolist())  # pragma: no cover

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if k < 1:
            raise UsageError(f"k must be >= 1, got {k}")
        if query_vec.shape != (self.dim,):
            raise UsageError(f"query vector must be ({self.dim},), got {tuple(query_vec.shape)}")
        res = self._collection.query(  # pragma: no cover
            query_embeddings=[query_vec.tolist()], n_results=k
        )
        ids = res["ids"][0]  # pragma: no cover
        distances = res["distances"][0]  # pragma: no cover
        scored = [(doc_id, 1.0 - float(dist))  # pragma: no cover
                  for doc_id, dist in zip(ids, distances, strict=True)]
        # Re-apply this package's total order so store choice cannot change ranking semantics.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))  # pragma: no cover
        return scored  # pragma: no cover

    def __len__(self) -> int:
        return self._collection.count()  # pragma: no cover
