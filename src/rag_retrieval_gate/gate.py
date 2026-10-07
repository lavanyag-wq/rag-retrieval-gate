"""The gate: compares two runs per query and refuses to rule when it cannot.

Three verdicts, not two. PASS means the evidence does not support a regression.
REGRESSION means it does, at the configured alpha, on the exact sign test over
paired per-query outcomes. UNDERPOWERED means the honest third thing: the
number of moved queries is below what this golden set can ever distinguish from
noise, so neither PASS nor REGRESSION would be a statement about the candidate
model rather than about the set. A gate without the third verdict converts
small golden sets into false confidence, which is the failure this repository
exists to name.

Attribution happens here too: per-query transitions (which queries lost a
relevant document that the baseline had, and which gained one), grouped by
query style. The mean delta is reported last, because it is the least
informative number in the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import UsageError
from .run import RunResult
from .stats import resolution_for, sign_test_p_value

VERDICT_PASS = "PASS"
VERDICT_REGRESSION = "REGRESSION"
VERDICT_UNDERPOWERED = "UNDERPOWERED"


@dataclass(frozen=True)
class Transition:
    query_id: str
    style: str
    baseline_value: float
    candidate_value: float
    lost_doc_ids: tuple[str, ...]
    gained_doc_ids: tuple[str, ...]


@dataclass
class GateReport:
    metric: str
    k: int
    alpha: float
    n_queries: int
    worse: list[Transition] = field(default_factory=list)
    better: list[Transition] = field(default_factory=list)
    ties: int = 0
    p_value: float = 1.0
    min_detectable_worse: int = 0
    verdict: str = VERDICT_PASS
    mean_baseline: float = 0.0
    mean_candidate: float = 0.0

    @property
    def mean_delta(self) -> float:
        return self.mean_candidate - self.mean_baseline

    def by_style(self) -> dict[str, dict[str, int]]:
        styles: dict[str, dict[str, int]] = {}
        for name, bucket in (("worse", self.worse), ("better", self.better)):
            for t in bucket:
                styles.setdefault(t.style, {"worse": 0, "better": 0})[name] += 1
        return styles

    def to_json(self) -> dict:
        return {
            "metric": self.metric,
            "k": self.k,
            "alpha": self.alpha,
            "n_queries": self.n_queries,
            "worse": [vars(t) | {"lost_doc_ids": list(t.lost_doc_ids),
                                 "gained_doc_ids": list(t.gained_doc_ids)} for t in self.worse],
            "better": [vars(t) | {"lost_doc_ids": list(t.lost_doc_ids),
                                  "gained_doc_ids": list(t.gained_doc_ids)} for t in self.better],
            "ties": self.ties,
            "p_value": self.p_value,
            "min_detectable_worse": self.min_detectable_worse,
            "verdict": self.verdict,
            "mean_baseline": self.mean_baseline,
            "mean_candidate": self.mean_candidate,
            "mean_delta": self.mean_delta,
            "by_style": self.by_style(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True))


def compare(baseline: RunResult, candidate: RunResult, metric: str = "recall",
            alpha: float = 0.05) -> GateReport:
    """Compare candidate against baseline on one metric, per query, then rule."""
    if metric not in {"recall", "mrr", "ndcg"}:
        raise UsageError(f"unknown metric {metric!r}")
    if not 0.0 < alpha < 1.0:
        raise UsageError(f"alpha must be in (0, 1), got {alpha}")
    if baseline.corpus_seed != candidate.corpus_seed:
        raise UsageError(
            f"runs are over different corpora (seeds {baseline.corpus_seed} and "
            f"{candidate.corpus_seed}); a paired comparison would be meaningless"
        )
    if baseline.k != candidate.k:
        raise UsageError(f"runs use different k ({baseline.k} and {candidate.k})")
    base_q = baseline.per_query
    cand_q = candidate.per_query
    if set(base_q) != set(cand_q):
        raise UsageError(
            "runs cover different query sets; a paired comparison would be meaningless"
        )

    report = GateReport(metric=metric, k=baseline.k, alpha=alpha,
                        n_queries=len(base_q))
    for query_id in sorted(base_q):
        b, c = base_q[query_id], cand_q[query_id]
        bv, cv = b.scores[metric], c.scores[metric]
        lost = tuple(sorted(set(b.ranked_doc_ids) - set(c.ranked_doc_ids)))
        gained = tuple(sorted(set(c.ranked_doc_ids) - set(b.ranked_doc_ids)))
        transition = Transition(query_id, b.style, bv, cv, lost, gained)
        if cv < bv:
            report.worse.append(transition)
        elif cv > bv:
            report.better.append(transition)
        else:
            report.ties += 1

    report.mean_baseline = baseline.mean(metric)
    report.mean_candidate = candidate.mean(metric)
    report.p_value = sign_test_p_value(len(report.worse), len(report.better))
    resolution = resolution_for(report.n_queries, alpha)
    report.min_detectable_worse = resolution.min_detectable_worse

    n_moved = len(report.worse) + len(report.better)
    if report.p_value <= alpha:
        report.verdict = VERDICT_REGRESSION
    elif n_moved > 0 and n_moved < report.min_detectable_worse:
        # Even if every moved query had moved the bad way, this set could not
        # have reached significance. PASS here would be a claim about the set.
        report.verdict = VERDICT_UNDERPOWERED
    else:
        report.verdict = VERDICT_PASS
    return report
