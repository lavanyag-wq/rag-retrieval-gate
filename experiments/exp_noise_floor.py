"""Experiment: the naive mean rule cries wolf; the gate holds its alpha.

Question the README asks: the common gate in the wild is "fail if mean recall
drops more than a point"; how often does that fire when nothing changed? Two
measurements, each against the null it actually faces.

Naive rule, resampling null. A team that refreshes its golden set and
re-baselines is comparing two disjoint query subsets of the same population
under an identical system. For each trial, sample two disjoint subsets, take
the mean recall of each, and count how often |mean delta| exceeds one point.
Every fire is a false positive, because the system on both sides is the same.

Gate, sign-flip null. The gate is a paired test: the same query under two
systems. Its null is that each per-query delta is equally likely to be a gain
or a loss, so the null is sampled by taking the real paired deltas from the
v1 vs v2 comparison and flipping each delta's sign with probability one half.
Count how often the gate then reaches significance. A calibrated test stays at
or below alpha.

The first version of this experiment fed an unpaired construction (two
disjoint subsets scored against their pooled median) to the paired sign test
and read a 78.5 percent "false positive rate" at subset size 18, which is
impossible for a calibrated test under its own null and was the number that
exposed the design bug. The fix is this file; the story is in the README and
in docs/adr/ADR-004.

Writes docs/experiments/noise_floor.json and prints both tables.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rag_retrieval_gate.corpus import generate_corpus
from rag_retrieval_gate.run import execute_run
from rag_retrieval_gate.stats import sign_test_p_value

OUT = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "noise_floor.json"

TRIALS = 2000
SUBSET_SIZES = [8, 12, 18]
NAIVE_THRESHOLD_POINTS = 1.0
ALPHA = 0.05


def naive_rule_fp_rates(scores: dict[str, float], rng: random.Random) -> list[dict]:
    query_ids = sorted(scores)
    rows = []
    for size in SUBSET_SIZES:
        fires = 0
        for _ in range(TRIALS):
            sample = rng.sample(query_ids, 2 * size)
            mean_a = sum(scores[q] for q in sample[:size]) / size
            mean_b = sum(scores[q] for q in sample[size:]) / size
            if abs(mean_a - mean_b) > NAIVE_THRESHOLD_POINTS / 100:
                fires += 1
        rows.append(
            {
                "subset_size": size,
                "trials": TRIALS,
                "fires": fires,
                "fp_rate_pct": round(100 * fires / TRIALS, 1),
            }
        )
    return rows


def gate_calibration(deltas: list[float], rng: random.Random) -> dict:
    moved = [d for d in deltas if d != 0.0]
    fires = 0
    for _ in range(TRIALS):
        worse = sum(1 for _d in moved if rng.random() < 0.5)
        better = len(moved) - worse
        if sign_test_p_value(worse, better) <= ALPHA:
            fires += 1
    return {
        "moved_queries": len(moved),
        "trials": TRIALS,
        "fires": fires,
        "fp_rate_pct": round(100 * fires / TRIALS, 2),
        "alpha_pct": 100 * ALPHA,
    }


def main() -> None:
    corpus = generate_corpus(seed=7)
    baseline = execute_run(corpus, "v1", k=5, dim=256)
    candidate = execute_run(corpus, "v2", k=5, dim=256)
    base_scores = {r.query_id: r.scores["recall"] for r in baseline.results}
    cand_scores = {r.query_id: r.scores["recall"] for r in candidate.results}
    deltas = [cand_scores[q] - base_scores[q] for q in sorted(base_scores)]
    rng = random.Random(99)

    naive_rows = naive_rule_fp_rates(base_scores, rng)
    gate_row = gate_calibration(deltas, rng)

    payload = {
        "naive_threshold_points": NAIVE_THRESHOLD_POINTS,
        "alpha": ALPHA,
        "population_queries": len(base_scores),
        "naive": naive_rows,
        "gate": gate_row,
        "note": "identical system for the naive rows; sign-flipped real deltas for the gate",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print(f"naive |mean delta| > {NAIVE_THRESHOLD_POINTS:.0f}pt rule, identical system, "
          f"{TRIALS} resampled golden sets per size")
    print(f"{'subset size':>12} {'false positive rate':>20}")
    for r in naive_rows:
        print(f"{r['subset_size']:>12} {r['fp_rate_pct']:>19.1f}%")
    print(f"\nsign-test gate under its sign-flip null "
          f"({gate_row['moved_queries']} moved queries, {TRIALS} trials)")
    print(f"fired {gate_row['fires']} times: {gate_row['fp_rate_pct']:.2f}% "
          f"(alpha {gate_row['alpha_pct']:.0f}%)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
