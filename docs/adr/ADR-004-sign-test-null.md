# ADR-004: The noise-floor experiment's null, and the bug that fixed it

Status: accepted

## Decision

The noise-floor experiment measures two different rules against two different
nulls. The naive mean-threshold rule is measured against a resampling null:
two disjoint golden subsets of the same population under an identical system,
which is what re-baselining against a refreshed golden set actually compares.
The gate is measured against its own paired null: the real per-query deltas
from the v1 to v2 comparison with each delta's sign flipped with probability
one half.

## The bug this decision came from

The first version fed an unpaired construction to the paired test: it scored
two disjoint subsets against their pooled median and handed the counts to the
sign test. The printed result was a 78.5% false positive rate at subset size
18. That number is impossible for a calibrated test under its own null, so it
was diagnosing the experiment rather than the gate. Root cause: a median
split of two disjoint samples is not a pairing, and the counts it produces
are strongly dependent, which the binomial model assumes away. The fix is the
sign-flip null, under which exchangeability holds by construction. Measured
afterwards: the gate fired 2.5% of the time over 2000 sign-flip trials,
consistent with an exact test that is conservative at 5 moved queries.

## Why publish the bug

Because the broken number is the strongest argument in the repository for the
repository's own thesis. A confidently printed 78.5% with a plausible-looking
methodology underneath is exactly the kind of figure that survives review
when nothing re-measures it. The receipts pipeline pins the historical value
so the war story cannot drift, and labels it as recorded history rather than
a measured output of the current code.
