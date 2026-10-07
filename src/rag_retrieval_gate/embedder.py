"""Deterministic embedders, so every downstream number is reproducible offline.

A real embedding model would make the absolute scores more lifelike and every
number in this repository less checkable: model files move, APIs version, and
GPUs reorder floats. The harness measures the gate, not the model, so the
embedders here are pure functions of their input text. Two versions are
provided because the gate's whole job is comparing a baseline against a
candidate: v1 hashes word unigrams, v2 adds character trigrams, which changes
retrieval behaviour the way a model swap does, in both directions at once.
ADR-001 records which claims survive this substitution and which one does not.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .errors import UsageError

_VERSIONS = ("v1", "v2")


def _bucket(token: str, salt: str, dim: int) -> tuple[int, float]:
    """Map a token to a (bucket, sign) pair via a stable cryptographic hash.

    Python's builtin hash() is salted per process, so using it here would make
    every embedding differ between runs. blake2b is stable across processes,
    platforms and Python versions, which is what reproducibility actually needs.
    """
    digest = hashlib.blake2b(f"{salt}|{token}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dim, sign


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def _char_trigrams(text: str) -> list[str]:
    s = " ".join(_words(text))
    return [s[i : i + 3] for i in range(len(s) - 2)]


class HashingEmbedder:
    """Feature-hashing text embedder: deterministic, offline, versioned.

    v1 embeds word unigrams. v2 embeds word unigrams plus character trigrams,
    down-weighted. Trigrams help acronym and partial-token matches and add noise
    to exact-token matches, so v2 is a realistic model change: some queries
    improve, others regress, and the mean moves less than the per-query story.
    """

    def __init__(self, version: str = "v1", dim: int = 256):
        if version not in _VERSIONS:
            raise UsageError(f"unknown embedder version {version!r}, expected one of {_VERSIONS}")
        if dim < 8:
            raise UsageError(f"dim must be >= 8, got {dim}")
        self.version = version
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise UsageError(f"embed() takes a string, got {type(text).__name__}")
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in _words(text):
            bucket, sign = _bucket(token, "w", self.dim)
            vec[bucket] += sign
        if self.version == "v2":
            for tri in _char_trigrams(text):
                bucket, sign = _bucket(tri, "t", self.dim)
                vec[bucket] += 0.35 * sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack([self.embed(t) for t in texts])
