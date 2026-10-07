"""Experiment: what the gate says about three realistic candidate changes.

Question the README asks: when an embedding change ships, what does the mean
say, what do the per-query transitions say, and what is the gate entitled to
conclude? Three candidates against the same v1/dim=256 baseline on the default
36-query corpus:

  upgrade   v2/dim=256: a feature change (adds character trigrams)
  downsize  v1/dim=64:  a cost change (collisions from a smaller hash space)
  collapse  v1/dim=32:  the same cost change taken further

Writes docs/experiments/regression.json and prints the table.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_retrieval_gate.corpus import generate_corpus
from rag_retrieval_gate.gate import compare
from rag_retrieval_gate.run import execute_run

OUT = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "regression.json"

CANDIDATES = [
    ("upgrade", "v2", 256),
    ("downsize", "v1", 64),
    ("collapse", "v1", 32),
]


def main() -> None:
    corpus = generate_corpus(seed=7)
    baseline = execute_run(corpus, "v1", k=5, dim=256)
    rows = []
    for name, version, dim in CANDIDATES:
        candidate = execute_run(corpus, version, k=5, dim=dim)
        report = compare(baseline, candidate, metric="recall")
        rows.append(
            {
                "candidate": name,
                "embedder": f"{version}/dim={dim}",
                "mean_recall_baseline": round(report.mean_baseline, 4),
                "mean_recall_candidate": round(report.mean_candidate, 4),
                "mean_delta_points": round(100 * report.mean_delta, 1),
                "worse": len(report.worse),
                "better": len(report.better),
                "ties": report.ties,
                "p_value": round(report.p_value, 5),
                "verdict": report.verdict,
                "by_style": report.by_style(),
                "worse_query_ids": [t.query_id for t in report.worse],
            }
        )

    payload = {
        "n_queries": len(corpus.queries),
        "k": 5,
        "metric": "recall",
        "alpha": 0.05,
        "baseline": "v1/dim=256",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))

    header = f"{'candidate':<10} {'mean delta':>10} {'worse':>6} {'better':>7} {'p':>8} verdict"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['candidate']:<10} {r['mean_delta_points']:>+9.1f}p {r['worse']:>6} "
              f"{r['better']:>7} {r['p_value']:>8.5f} {r['verdict']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
