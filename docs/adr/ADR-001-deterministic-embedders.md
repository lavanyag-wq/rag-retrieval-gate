# ADR-001: Deterministic hash embedders instead of a real embedding model

Status: accepted

## Decision

Retrieval runs use versioned feature-hashing embedders (word unigrams for v1,
plus down-weighted character trigrams for v2), hashed with blake2b. No model
files, no API calls, no network at any point in `make verify`.

## The alternative

A small sentence-transformer, or an embeddings API behind a key. Either would
make absolute retrieval scores lifelike, and both were rejected.

## Why

The thing under measurement is the gate, not the embedder. The gate's claims
are about paired comparison, significance and resolution, and those claims are
checked by code that re-runs on every CI build. A model dependency breaks that
in three ways: model files move and version, APIs are nondeterministic across
time, and float reductions reorder across hardware. A reviewer who cannot
reproduce a number has no reason to believe the next one.

blake2b specifically, because Python's builtin hash() is salted per process,
and a per-process salt would make every embedding differ between runs. A test
pins this by embedding the same text in two subprocesses.

## The boundary this buys

What survives the substitution: every claim about the gate itself. The sign
test's calibration, the resolution floor, the underpowered verdict, the
transition attribution, and the false positive behaviour of the naive mean
rule are all properties of the comparison machinery, not of the embedding.

What does not survive: the absolute metric levels. Mean recall near 0.8 on
this corpus says nothing about what a production embedder scores on a real
document base. Anyone quoting this repository's absolute recall as evidence
about a real system is misreading it, and the README says so. The one claim
that genuinely needs a real model, that version upgrades concentrate damage in
particular query styles, is demonstrated here as a mechanism (trigrams change
acronym handling) and would need a labelled production set to assert about any
specific model pair.
