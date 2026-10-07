"""Collect every figure the documents quote, from machine-readable outputs.

Nothing in docs/metrics.json is typed by hand. Test count and coverage come
from the junit XML and the coverage JSON that pytest wrote (or writes here,
unless --skip-tests says CI already ran them as their own step). Experiment
figures come from the JSON files the experiments wrote; if those are missing,
the experiments are run. Derived values are computed here, in one place, so a
rounding rule changes in exactly one file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
EXPERIMENTS = ROOT / "docs" / "experiments"
OUT = ROOT / "docs" / "metrics.json"

ANCHORS = {
    "tests_total": ["{} tests", "{} property-named tests"],
    "coverage_line_pct": ["{}% line coverage", "coverage of {}%"],
    "n_queries": ["{}-query", "{} queries", "{} labelled queries"],
    "n_documents": ["{} documents"],
    "min_detectable_worse": ["fewer than {} one-direction flips", "below {} one-direction",
                             "least {} queries must move", "minimum of {} flips"],
    "floor_share_36_pct": ["{}% of the 36-query set"],
    "upgrade_mean_delta_points": ["mean recall by {} points", "a {} point mean"],
    "upgrade_worse": ["{} queries lost", "{} worse"],
    "upgrade_better": ["{} improved", "{} better"],
    "upgrade_p_value": ["p = {}", "p-value of {}"],
    "downsize_mean_delta_points": ["dim=64 moved the mean by {} points",
                                   "mean delta of {} points"],
    "downsize_worse": ["all {} moved queries", "{} queries strictly worse"],
    "downsize_p_value": ["p = {} on", "{} significance"],
    "required_10pct": ["needs {} queries", "takes {} queries", "requires {} queries"],
    "required_5pct": ["{} queries to detect", "would take {} queries"],
    "naive_fp_8_pct": ["fired on {}% of", "{}% of resampled"],
    "naive_fp_18_pct": ["fires {}% of the time"],
    "gate_fp_pct": ["fired {}% of the time", "measured {}%"],
    "gate_trials": ["{} sign-flip trials", "over {} trials"],
    "upgrade_acronym_worse": ["{} of them acronym-style", "{} acronym queries"],
    "first_bad_gate_fp_pct": ["read {}% false positives", "a {}% false positive",
                              "the {}% false positive"],
}


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def run_tests() -> None:
    REPORTS.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "--junitxml", str(REPORTS / "junit.xml"),
         "--cov=rag_retrieval_gate", f"--cov-report=json:{REPORTS / 'coverage.json'}",
         "-q"],
        check=True, cwd=ROOT,
    )


def run_experiments() -> None:
    for script in sorted((ROOT / "experiments").glob("exp_*.py")):
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)


def collect(skip_tests: bool, skip_experiments: bool) -> dict:
    if not skip_tests:
        run_tests()
    if not skip_experiments:
        run_experiments()
    junit = ET.parse(REPORTS / "junit.xml").getroot()
    suite = junit if junit.tag == "testsuite" else junit.find("testsuite")
    tests_total = int(suite.get("tests"))
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    if failures:
        raise SystemExit(f"refusing to collect metrics from a failing suite ({failures} failed)")
    coverage = json.loads((REPORTS / "coverage.json").read_text())
    coverage_pct = round(coverage["totals"]["percent_covered"])

    regression = json.loads((EXPERIMENTS / "regression.json").read_text())
    rows = {r["candidate"]: r for r in regression["rows"]}
    resolution = json.loads((EXPERIMENTS / "resolution.json").read_text())
    floors = {f["n_queries"]: f for f in resolution["floors"]}
    required = {r["regression_rate_pct"]: r for r in resolution["required"]}
    noise = json.loads((EXPERIMENTS / "noise_floor.json").read_text())
    naive = {r["subset_size"]: r for r in noise["naive"]}

    from rag_retrieval_gate.corpus import generate_corpus

    upgrade = rows["upgrade"]
    downsize = rows["downsize"]
    metrics = {
        "tests_total": tests_total,
        "coverage_line_pct": coverage_pct,
        "n_queries": regression["n_queries"],
        "n_documents": generate_corpus(seed=7).meta["n_documents"],
        "min_detectable_worse": floors[36]["min_detectable_worse"],
        "floor_share_36_pct": floors[36]["floor_share_pct"],
        "upgrade_mean_delta_points": abs(upgrade["mean_delta_points"]),
        "upgrade_worse": upgrade["worse"],
        "upgrade_better": upgrade["better"],
        "upgrade_p_value": upgrade["p_value"],
        "upgrade_acronym_worse": upgrade["by_style"]["acronym"]["worse"],
        "downsize_mean_delta_points": abs(downsize["mean_delta_points"]),
        "downsize_worse": downsize["worse"],
        "downsize_p_value": downsize["p_value"],
        "required_10pct": required[10]["required_queries"],
        "required_5pct": required[5]["required_queries"],
        "naive_fp_8_pct": naive[8]["fp_rate_pct"],
        "naive_fp_18_pct": naive[18]["fp_rate_pct"],
        "gate_fp_pct": noise["gate"]["fp_rate_pct"],
        "gate_trials": noise["gate"]["trials"],
        "first_bad_gate_fp_pct": 78.5,
    }
    # The one hand-set value above is the historical reading from the broken
    # experiment this build fixed; it is pinned here so the war story's number
    # is still guarded against silent edits, and labelled for what it is.
    return {
        "metrics": metrics,
        "anchors": ANCHORS,
        "checked_documents": ["README.md", "docs/defense-guide.md",
                              "docs/adr/ADR-004-sign-test-null.md"],
        "note": "every value except first_bad_gate_fp_pct is produced by running "
                "the suite and the experiments; first_bad_gate_fp_pct is the "
                "recorded output of the pre-fix experiment, see ADR-004",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true",
                        help="read reports/ written by a prior pytest step")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="read docs/experiments/ written by a prior run")
    args = parser.parse_args()
    payload = collect(args.skip_tests, args.skip_experiments)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {OUT} with {len(payload['metrics'])} metrics")


if __name__ == "__main__":
    main()
