"""The statistics that let the gate say "this set is too small to know".

Everything here is exact binomial arithmetic, chosen over a bootstrap on
purpose (ADR-002): with n paired per-query outcomes the exact sign test is a
closed-form computation, it has no resampling randomness to seed or explain,
and its power analysis inverts cleanly into the two numbers the README leads
with: the smallest regression this golden set can detect, and the set size a
target regression would need. scipy would provide the same tail sums; writing
the dozen lines keeps the dependency list at numpy and keeps every number
defensible in an interview.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import UsageError


def binom_tail(n: int, k: int, p: float = 0.5) -> float:
    """P[X >= k] for X ~ Binomial(n, p). Exact summation, no approximation."""
    if n < 0 or k < 0 or k > n:
        raise UsageError(f"invalid binomial tail: n={n}, k={k}")
    if not 0.0 <= p <= 1.0:
        raise UsageError(f"p must be in [0, 1], got {p}")
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def sign_test_p_value(worse: int, better: int) -> float:
    """One-sided exact sign test on paired per-query outcomes.

    Ties are dropped before this function, as the classical sign test drops
    them: a query whose metric did not move carries no information about
    direction. The p-value answers: if the change were truly neutral, how often
    would at least this many of the moved queries move in the bad direction?
    """
    if worse < 0 or better < 0:
        raise UsageError(f"counts must be >= 0, got worse={worse}, better={better}")
    n = worse + better
    if n == 0:
        return 1.0
    return binom_tail(n, worse, 0.5)


def min_detectable_worse(n_queries: int, alpha: float = 0.05) -> int:
    """Smallest number of one-direction flips that reaches significance.

    Worst case for detection: all moved queries moved the same way, so the
    question is the smallest m with binom_tail(m, m) = 0.5**m <= alpha. Below
    this count the gate structurally cannot call a regression, no matter how
    real it is. With alpha=0.05 the answer is 5 for any n_queries >= 5: fewer
    than five net-worse queries can never be distinguished from coin flips.
    """
    if n_queries < 1:
        raise UsageError(f"n_queries must be >= 1, got {n_queries}")
    if not 0.0 < alpha < 1.0:
        raise UsageError(f"alpha must be in (0, 1), got {alpha}")
    m = 1
    while 0.5**m > alpha:
        m += 1
        if m > n_queries:
            return n_queries + 1  # unreachable with this set: nothing it sees can be significant
    return m


def required_set_size(regression_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Golden set size needed to detect a regression affecting a given share of queries.

    Model: a true regression makes each query strictly worse independently with
    probability regression_rate and better with probability 0 (the cleanest
    case, so the answer is a lower bound; a messier change needs more). The
    returned n is the smallest set size whose probability of producing at least
    min_detectable_worse(n) worse queries is >= power.
    """
    if not 0.0 < regression_rate <= 1.0:
        raise UsageError(f"regression_rate must be in (0, 1], got {regression_rate}")
    if not 0.0 < power < 1.0:
        raise UsageError(f"power must be in (0, 1), got {power}")
    for n in range(1, 100_001):
        threshold = min_detectable_worse(n, alpha)
        if threshold > n:
            continue
        achieved = binom_tail(n, threshold, regression_rate)
        if achieved >= power:
            return n
    raise UsageError(  # pragma: no cover (guard; unreachable for rate > 0 at these alphas)
        "no set size up to 100000 achieves the requested power"
    )


@dataclass(frozen=True)
class Resolution:
    """What a golden set of this size can and cannot resolve."""

    n_queries: int
    alpha: float
    min_detectable_worse: int

    @property
    def floor_share(self) -> float:
        """Smallest detectable regression as a share of the set, worst case."""
        return self.min_detectable_worse / self.n_queries


def resolution_for(n_queries: int, alpha: float = 0.05) -> Resolution:
    return Resolution(n_queries, alpha, min_detectable_worse(n_queries, alpha))
