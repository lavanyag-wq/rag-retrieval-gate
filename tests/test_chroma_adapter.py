"""The Chroma adapter's contract, testable without chromadb installed.

The CI job that installs no extras must still be able to reach the adapter's
validation, which is why validation runs before the optional import. The last
test pins that ordering by making chromadb unimportable, so the invariant
holds whether or not the extra happens to be present on the machine.
"""

import builtins

import pytest

from rag_retrieval_gate.chroma_store import ChromaIndex
from rag_retrieval_gate.errors import UsageError


def _block_chromadb(monkeypatch):
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("chromadb"):
            raise ModuleNotFoundError("No module named 'chromadb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_a_bad_dim_is_refused_before_chromadb_is_imported(monkeypatch):
    _block_chromadb(monkeypatch)
    with pytest.raises(UsageError, match="dim must be"):
        ChromaIndex(dim=0)


def test_a_bad_collection_name_is_refused_before_chromadb_is_imported(monkeypatch):
    _block_chromadb(monkeypatch)
    with pytest.raises(UsageError, match="collection_name"):
        ChromaIndex(dim=8, collection_name="bad name!")


def test_a_missing_chromadb_is_reported_as_the_install_command(monkeypatch):
    _block_chromadb(monkeypatch)
    with pytest.raises(UsageError, match="chroma"):
        ChromaIndex(dim=8)
