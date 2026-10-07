"""Experiment: how often a naive mean-delta threshold cries wolf on pure noise.

Question the README asks: the common gate in the wild is "fail if mean recall
drops more than a point"; how often does that fire when nothing changed? Here
"nothing changed" is made literal: both runs use the same embedder version and
dimension, and the only difference is which queries happen to be in the golden
set. For each trial, a golden subset is sampled from a 36-query population and
the mean delta between the full-population baseline scores and the subset's
scores for a re-seeded sibling corpus is not needed at all: instead, two
disjoint subsets of the same population are compared, which is exactly what a
team does when it refreshes its golden set and re-baselines.

For each subset size, 400 seeded trials: sample two disjoint query subsets,
compare mean recall between them, and count how often the naive rule (|mean
delta| > 1 point) would have fired. The sign-test gate's false positive rate on
the same trials is measured alongside. Writes docs/experiments/noise_floor.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rag_retrieval_gate.corpus import generate_corpus
from rag_retrieval_gate.run import execute_run
from rag_retrieval_gate.stats import min_detectable_worse, sign_test_p_value

OUT = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "noise_floor.json"

TRIALS = 400
SUBSET_SIZES = [8, 12, 18]
NAIVE_THRESHOLD_POINTS = 1.0


def main() -> None:
    corpus = generate_corpus(seed=7)
    run = execute_run(corpus, "v1", k=5, dim=256)
    scores = {r.query_id: r.scores["recall"] for r in run.results}
    query_ids = sorted(scores)
    rng = random.Random(99)

    rows = []
    for size in SUBSET_SIZES:
        naive_fires = 0
        gate_fires = 0
        for _ in range(TRIALS):
            sample = rng.sample(query_ids, 2 * size)
            a, b = sample[:size], sample[size:]
            mean_a = sum(scores[q] for q in a) / size
            mean_b = sum(scores[q] for q in b) / size
            if abs(mean_a - mean_b) > NAIVE_THRESHOLD_POINTS / 100:
                naive_fires += 1
            # The paired gate compares the same query under two systems; across
            # two disjoint subsets the honest pairing is rank-free: count which
            # side of the pooled median each query falls on.
            pooled = sorted(scores[q] for q in a + b)
            median = pooled[size]  # upper median of 2*size values
            worse = sum(1 for q in b if scores[q] < median)
            better = sum(1 for q in b if scores[q] > median)
            if sign_test_p_value(worse, better) <= 0.05:
                gate_fires += 1
        rows.append(
            {
                "subset_size": size,
                "trials": TRIALS,
                "naive_fires": naive_fires,
                "naive_fp_rate_pct": round(100 * naive_fires / TRIALS, 1),
                "gate_fires": gate_fires,
                "gate_fp_rate_pct": round(100 * gate_fires / TRIALS, 1),
                "min_detectable_worse": min_detectable_worse(size),
            }
        )

    payload = {
        "naive_threshold_points": NAIVE_THRESHOLD_POINTS,
        "alpha": 0.05,
        "population_queries": len(query_ids),
        "rows": rows,
        "note": "identical system on both sides; every fire is a false positive",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(f"false positives over {TRIALS} trials per size "
          f"(identical system, resampled golden sets)")
    print(f"{'subset size':>12} {'naive >1pt rule':>16} {'sign-test gate':>15}")
    for r in rows:
        print(f"{r['subset_size']:>12} {r['naive_fp_rate_pct']:>15.1f}% "
              f"{r['gate_fp_rate_pct']:>14.1f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
