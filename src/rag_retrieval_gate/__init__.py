"""rag-retrieval-gate: a retrieval regression gate that knows when it cannot know.

This package exists because two numbers that RAG teams rely on are structurally
mute. An end-to-end answer score aggregates the retriever and the generator, so
it cannot attribute a failure to either one. And a pass/fail threshold on a
small golden set cannot separate a real regression from sampling noise, because
the set's size puts a floor under the smallest effect it can resolve. The
modules here isolate retrieval against a labelled relevance set, compare runs
per query rather than by their means, and refuse to rule (exit code 2) when the
set is too small to support a verdict.
"""

__version__ = "0.1.0"
