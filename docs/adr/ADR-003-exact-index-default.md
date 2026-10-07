# ADR-003: Exact brute-force index by default, ChromaDB behind an extra

Status: accepted

## Decision

The default vector index is exact cosine over numpy with ties broken by
doc_id. A ChromaIndex adapter satisfies the same protocol and lives behind the
optional `chroma` extra. The adapter validates every argument before importing
chromadb, and re-sorts Chroma's results with this package's own tie-break.

## The alternative

Making ChromaDB (or any ANN index) the default store, which is what most RAG
tutorials do.

## Why

An approximate index has its own recall loss, and that loss would be folded
invisibly into every number the gate publishes: a "regression" could be HNSW
parameter noise. The gate measures retrieval changes, so the reference path
must not itself be a source of them. Exact search over 58 documents costs
nothing; the day the corpus is large enough for ANN to matter is the day the
tool is measuring a real system through the adapter instead.

Validate-before-import is a design rule, not a styling choice: a caller with a
bad request is told immediately rather than after a heavy import, and the CI
job that installs no extras can still reach and test the adapter's boundary.
The test suite pins the ordering by making chromadb unimportable.

The re-sort after querying Chroma exists so that switching stores cannot
silently switch ranking semantics: equal scores order by doc_id on every
backend, or the paired comparison would depend on which store produced it.
