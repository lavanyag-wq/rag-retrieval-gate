"""Experiment: what golden sets of common sizes can and cannot detect.

Question the README asks: how big does a golden set have to be before the gate
can call the regressions teams actually care about? Two tables. The first is
the detection floor by set size: the smallest number of one-direction flips
that can reach significance, and what share of the set that is. The second
inverts the arithmetic: for a regression hitting a given share of queries, the
set size needed to detect it with 80% power.

Writes docs/experiments/resolution.json and prints both tables.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_retrieval_gate.stats import required_set_size, resolution_for

OUT = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "resolution.json"

SET_SIZES = [10, 20, 36, 50, 100, 200]
TARGET_RATES = [0.5, 0.25, 0.10, 0.05]


def main() -> None:
    floors = []
    for n in SET_SIZES:
        res = resolution_for(n)
        floors.append(
            {
                "n_queries": n,
                "min_detectable_worse": res.min_detectable_worse,
                "floor_share_pct": round(100 * res.floor_share, 1),
            }
        )
    required = []
    for rate in TARGET_RATES:
        required.append(
            {
                "regression_rate_pct": round(100 * rate),
                "required_queries": required_set_size(rate, alpha=0.05, power=0.8),
            }
        )
    payload = {"alpha": 0.05, "power": 0.8, "floors": floors, "required": required}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))

    print("detection floor by set size (alpha=0.05, worst case: all flips one way)")
    print(f"{'queries':>8} {'min flips':>10} {'share of set':>13}")
    for f in floors:
        print(f"{f['n_queries']:>8} {f['min_detectable_worse']:>10} "
              f"{f['floor_share_pct']:>12.1f}%")
    print("\nset size required to detect a regression (power=0.8)")
    print(f"{'regression hits':>16} {'queries needed':>15}")
    for r in required:
        print(f"{r['regression_rate_pct']:>15}% {r['required_queries']:>15}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
