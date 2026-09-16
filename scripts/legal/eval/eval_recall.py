#!/usr/bin/env python
"""Measure how well cognee recall answers hand-written questions about a corpus.

Why this exists: "the legal profile improved recall" is not a claim you can make
from reading three answers. This runs a fixed question set across several
datasets and several search types against a *running* cognee server, then has an
LLM judge grade each answer against gold facts written by hand from the source
documents - so a before/after change shows up as a number instead of a vibe.

Usage::

    # validate a question file without calling anything
    python scripts/legal/eval/eval_recall.py --validate-only \
        --questions scripts/legal/eval/adams_questions.json

    # full run: two datasets, three search types, judged
    python scripts/legal/eval/eval_recall.py \
        --questions scripts/legal/eval/adams_questions.json \
        --datasets adams_family_redevelopment,adams_family_legal \
        --search-types HYBRID_COMPLETION,GRAPH_COMPLETION,AUTO

    # re-grade a finished run without re-answering anything
    python scripts/legal/eval/eval_recall.py \
        --questions scripts/legal/eval/adams_questions.json \
        --judge-only scripts/legal/eval/runs/20260916T101500Z

Answers come over HTTP from the server at ``--base-url``; cognee is only imported
in-process for the judge (and to validate search type names). Credentials come
from ``COGNEE_EVAL_USER`` / ``COGNEE_EVAL_PASSWORD``; neither they nor the bearer
token are ever printed or written to the run directory.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import recall_eval_lib as lib  # noqa: E402 - needs the sys.path entry above

DEFAULT_RUNS_ROOT = SCRIPT_DIRECTORY / "runs"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Deliberately free of cognee imports so ``--help`` is instant."""
    parser = argparse.ArgumentParser(
        prog="eval_recall.py",
        description="Evaluate cognee recall quality against hand-written gold facts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questions",
        action="append",
        default=None,
        metavar="PATH",
        help="Question file (repeatable). Required unless nothing is being run.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to evaluate.",
    )
    parser.add_argument(
        "--search-types",
        default="HYBRID_COMPLETION",
        help=(
            "Comma-separated search types, validated against cognee's SearchType "
            "names plus AUTO (which asks /recall with search_type null and lets "
            "the router choose)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=lib.DEFAULT_TOP_K, help="top_k per search.")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="Only the first N questions."
    )
    parser.add_argument(
        "--base-url", default=lib.DEFAULT_BASE_URL, help="Base URL of the running cognee server."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=lib.DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds (searches are slow).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Run directory (default: scripts/legal/eval/runs/<UTC timestamp>).",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Collect answers only; skip the LLM judge."
    )
    parser.add_argument(
        "--judge-only",
        default=None,
        metavar="RUN_DIR",
        help="Re-judge the answers.jsonl of an existing run directory; makes no search calls.",
    )
    parser.add_argument(
        "--spot-check",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="Print a seeded random sample of verdicts for hand review, e.g. 0.2.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the spot-check sample.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the question files and exit (0 valid, 1 invalid).",
    )
    return parser


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _validate_only(paths: list[str]) -> int:
    failed = False
    for path in paths:
        try:
            questions = lib.validate_question_file(path)
        except lib.QuestionFileError as error:
            failed = True
            print(f"INVALID {error}")
        else:
            categories = sorted({question.category for question in questions})
            gold = sum(len(question.gold_facts) for question in questions)
            print(
                f"OK      {path}: {len(questions)} question(s), {gold} gold fact(s), "
                f"categories: {', '.join(categories)}"
            )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    question_paths = args.questions or []
    if not question_paths:
        parser.error("--questions is required (repeat it for several files)")

    if args.validate_only:
        return _validate_only(question_paths)

    try:
        questions = lib.load_question_files(question_paths)
    except lib.QuestionFileError as error:
        print(f"INVALID {error}", file=sys.stderr)
        return 1
    if args.limit is not None:
        questions = questions[: max(args.limit, 0)]
    if not questions:
        print("No questions to run.", file=sys.stderr)
        return 1

    if args.judge_only:
        run_directory = Path(args.judge_only)
        answers_path = run_directory / lib.ANSWERS_FILENAME
        if not answers_path.exists():
            print(f"No {lib.ANSWERS_FILENAME} in {run_directory}", file=sys.stderr)
            return 1
        answer_rows = [lib.AnswerRow.from_dict(row) for row in lib.read_jsonl(answers_path)]
        _progress(f"Re-judging {len(answer_rows)} saved answer(s) from {run_directory}")
    else:
        datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
        if not datasets:
            parser.error("--datasets is required for a run (comma-separated names)")
        try:
            search_types = lib.resolve_search_types(args.search_types.split(","))
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1

        run_directory = Path(args.out) if args.out else lib.default_run_directory(DEFAULT_RUNS_ROOT)
        run_directory.mkdir(parents=True, exist_ok=True)

        username, password = lib.credentials_from_environment()
        client = lib.HttpxClient(base_url=args.base_url, timeout=args.timeout)
        total = len(datasets) * len(search_types) * len(questions)
        done = 0

        def report_row(row: lib.AnswerRow) -> None:
            nonlocal done
            done += 1
            status = "ERROR" if row.error else f"{row.elapsed_seconds:.1f}s"
            _progress(
                f"[{done}/{total}] {row.dataset} / {row.search_type} / {row.question_id}: {status}"
            )

        try:
            token = lib.login(client, username, password)
            _progress(f"Logged in as {username} at {args.base_url}")
            answer_rows = lib.run_answers(
                client,
                token=token,
                questions=questions,
                datasets=datasets,
                search_types=search_types,
                top_k=args.top_k,
                on_row=report_row,
            )
        finally:
            client.close()

        lib.write_jsonl(run_directory / lib.ANSWERS_FILENAME, answer_rows)
        _progress(f"Wrote {run_directory / lib.ANSWERS_FILENAME}")

    if args.no_judge:
        aggregates = lib.aggregate_answers_only(answer_rows)
        verdict_rows: list[lib.VerdictRow] = []
    else:
        lib.configure_llm_environment()
        judged = 0

        def report_verdict(row: lib.VerdictRow) -> None:
            nonlocal judged
            judged += 1
            _progress(f"[judge {judged}/{len(answer_rows)}] {row.question_id} ({row.search_type})")

        verdict_rows = asyncio.run(lib.run_judge(answer_rows, questions, on_row=report_verdict))
        lib.write_jsonl(run_directory / lib.VERDICTS_FILENAME, verdict_rows)
        _progress(f"Wrote {run_directory / lib.VERDICTS_FILENAME}")
        aggregates = lib.aggregate(verdict_rows)

    report = lib.render_report(aggregates, title=f"Recall evaluation - {run_directory.name}")
    (run_directory / lib.REPORT_FILENAME).write_text(report, encoding="utf-8")
    print(report)

    if args.spot_check > 0 and verdict_rows:
        sample = lib.spot_check_sample(verdict_rows, args.spot_check, args.seed)
        print(f"\nSpot check ({len(sample)} of {len(verdict_rows)} verdicts, seed {args.seed}):\n")
        print(lib.format_spot_check(sample, questions))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
