"""Command line interface. The exit code is the product.

Subcommands mirror the pipeline: generate a corpus, execute a run, gate a
candidate run against a baseline. The gate's exit code carries the verdict:
0 PASS, 1 REGRESSION (the tool worked and the answer was no), 2 UNDERPOWERED
(the tool worked and the set cannot support an answer), 3 for any error. CI
should block on 1, warn loudly on 2, and page on 3; treating 2 as a pass is
exactly the false confidence the gate exists to remove.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import Corpus, generate_corpus
from .errors import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_UNDERPOWERED,
    GateError,
)
from .gate import VERDICT_PASS, VERDICT_UNDERPOWERED, compare
from .run import RunResult, execute_run
from .stats import required_set_size, resolution_for


def _cmd_generate(args: argparse.Namespace) -> int:
    corpus = generate_corpus(seed=args.seed, n_topics=args.topics,
                             distractors_per_topic=args.distractors, fillers=args.fillers)
    corpus.save(Path(args.out))
    print(f"wrote corpus: {corpus.meta['n_documents']} documents, "
          f"{corpus.meta['n_queries']} queries, seed {corpus.seed} -> {args.out}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    corpus = Corpus.load(Path(args.corpus))
    run = execute_run(corpus, embedder_version=args.embedder, k=args.k, dim=args.dim)
    run.save(Path(args.out))
    print(f"run complete: embedder {run.embedder_version}, k={run.k}, "
          f"{len(run.results)} queries, mean recall@{run.k} "
          f"{run.mean('recall'):.3f} -> {args.out}")
    return EXIT_OK


def _cmd_gate(args: argparse.Namespace) -> int:
    baseline = RunResult.load(Path(args.baseline))
    candidate = RunResult.load(Path(args.candidate))
    report = compare(baseline, candidate, metric=args.metric, alpha=args.alpha)
    if args.out:
        report.save(Path(args.out))
    print(json.dumps(report.to_json() | {"transition_detail": "see --out file"},
                     indent=1, sort_keys=True, default=str)
          if args.verbose else _summary(report))
    if report.verdict == VERDICT_PASS:
        return EXIT_OK
    if report.verdict == VERDICT_UNDERPOWERED:
        return EXIT_UNDERPOWERED
    return EXIT_REGRESSION


def _summary(report) -> str:
    lines = [
        f"verdict: {report.verdict}",
        f"metric {report.metric}@{report.k}: mean {report.mean_baseline:.3f} -> "
        f"{report.mean_candidate:.3f} (delta {report.mean_delta:+.3f})",
        f"per-query: {len(report.worse)} worse, {len(report.better)} better, "
        f"{report.ties} unchanged",
        f"sign test p={report.p_value:.4f} at alpha={report.alpha}",
        f"resolution: this set cannot call a regression on fewer than "
        f"{report.min_detectable_worse} one-direction flips",
    ]
    for style, counts in sorted(report.by_style().items()):
        lines.append(f"  style {style}: {counts['worse']} worse, {counts['better']} better")
    return "\n".join(lines)


def _cmd_resolution(args: argparse.Namespace) -> int:
    res = resolution_for(args.queries, args.alpha)
    print(f"a {res.n_queries}-query golden set cannot call a regression on fewer than "
          f"{res.min_detectable_worse} one-direction flips "
          f"({res.floor_share:.1%} of the set) at alpha={res.alpha}")
    if args.target_rate is not None:
        n = required_set_size(args.target_rate, alpha=args.alpha, power=args.power)
        print(f"to detect a regression hitting {args.target_rate:.0%} of queries "
              f"with power {args.power:.0%}, the set needs {n} queries")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-gate",
                                     description="Retrieval regression gate for RAG systems")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate a seeded corpus with labelled relevance")
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.add_argument("--topics", type=int, default=12)
    p_gen.add_argument("--distractors", type=int, default=2)
    p_gen.add_argument("--fillers", type=int, default=10)
    p_gen.add_argument("--out", required=True)
    p_gen.set_defaults(func=_cmd_generate)

    p_run = sub.add_parser("run", help="execute a retrieval run against a corpus")
    p_run.add_argument("--corpus", required=True)
    p_run.add_argument("--embedder", default="v1", choices=["v1", "v2"])
    p_run.add_argument("--k", type=int, default=5)
    p_run.add_argument("--dim", type=int, default=256,
                       help="embedding dimension; lower dims collide more")
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_gate = sub.add_parser("gate", help="compare a candidate run against a baseline")
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--candidate", required=True)
    p_gate.add_argument("--metric", default="recall", choices=["recall", "mrr", "ndcg"])
    p_gate.add_argument("--alpha", type=float, default=0.05)
    p_gate.add_argument("--out")
    p_gate.add_argument("--verbose", action="store_true")
    p_gate.set_defaults(func=_cmd_gate)

    p_res = sub.add_parser("resolution", help="what can a golden set of this size detect")
    p_res.add_argument("--queries", type=int, required=True)
    p_res.add_argument("--alpha", type=float, default=0.05)
    p_res.add_argument("--target-rate", type=float, default=None)
    p_res.add_argument("--power", type=float, default=0.8)
    p_res.set_defaults(func=_cmd_resolution)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
