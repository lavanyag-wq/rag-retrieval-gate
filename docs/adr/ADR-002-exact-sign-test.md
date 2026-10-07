# ADR-002: Exact sign test rather than a bootstrap

Status: accepted

## Decision

The gate's significance machinery is the one-sided exact sign test on paired
per-query outcomes, with ties dropped, computed by direct binomial summation.

## The alternative

A paired bootstrap over per-query deltas, or a Wilcoxon signed-rank test. Both
are stronger in the textbook sense: they use the magnitude of each delta, not
only its direction.

## Why the boring choice

Three reasons, in order of weight. First, the power analysis inverts in closed
form: the smallest number of one-direction flips that can reach significance
is an integer a reader can verify with mental arithmetic (0.5 to the fifth
power is 0.03125), and the required-set-size table follows from the same
summation. A bootstrap's power analysis is itself a simulation, which would
put a simulation inside the headline claim. Second, no resampling seed: a
bootstrap p-value moves between runs unless its seed is pinned and explained,
and this repository's whole posture is that every number re-measures
identically. Third, the magnitudes the sign test discards are exactly the part
of this corpus that is synthetic: recall deltas here are quantized to halves,
so rank-weighting them would dress artificial structure up as information.

## The cost, priced

The sign test is conservative. At 5 moved queries the only significant
outcome has probability 0.03125, so the gate's real false positive rate under
its null measured 2.5% over 2000 sign-flip trials against a nominal 5%. The
cost of that conservatism is paid where the README says it is paid: in the
resolution floor. Detecting small regressions needs more queries than a
magnitude-aware test would need, and the resolution experiment publishes
those set sizes rather than hiding them.
