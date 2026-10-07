# ADR-005: UNDERPOWERED is a verdict with its own exit code, not a pass

Status: accepted

## Decision

The gate exits 0 for PASS, 1 for REGRESSION, 2 for UNDERPOWERED, 3 for any
error. UNDERPOWERED is returned when queries moved but fewer than the set's
detection floor moved, so that even a unanimous bad direction could not have
reached significance.

## The alternative

Two codes: pass and fail. Nearly every CI gate in the wild does this, and
treats "no significant regression" as a pass.

## Why three verdicts

With a small golden set, "no significant regression" is almost always true
regardless of the candidate, because the set cannot reach significance on the
handful of queries that move. Folding that case into PASS converts set
smallness into green builds, which is precisely the false confidence this
tool exists to remove. The honest output is "this set cannot answer", stated
as its own machine-readable outcome so a pipeline can warn on it without
blocking, and so the number of UNDERPOWERED verdicts over time becomes the
metric that argues for growing the golden set.

Exit 3 is kept apart from all of them so that "the tool answered no" is never
confused with "the tool failed to answer". A gate whose crash looks like a
finding gets ignored within a month.

## The cost

Pipelines must handle a third code, and a team that blocks on 2 will be
blocked often until its golden set grows past the floor the resolution table
publishes. That friction is the feature: it prices the golden set's size in
the place where the size actually matters.
