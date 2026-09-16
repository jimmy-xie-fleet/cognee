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
        help=(
            "HTTP timeout in seconds (default 600). Efficiency is not a goal here: "
            "a graph search under load can take minutes, and a slow answer is data "
            "while a timed-out one is a hole in the table."
        ),
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Sleep this long between requests, to keep load off a shared server.",
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
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Re-run only the rows of that run's answers.jsonl that carry an error, "
            "merge them back in place, and judge only what changed. Successful rows "
            "and their verdicts are never re-run."
        ),
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


def _histogram_line(histogram: dict) -> str:
    return ", ".join(f"{name}={count}" for name, count in histogram.items()) or "no errors"


def _load_answers(directory: Path):
    answers_path = directory / lib.ANSWERS_FILENAME
    if not answers_path.exists():
        print(f"No {lib.ANSWERS_FILENAME} in {directory}", file=sys.stderr)
        return None
    return [lib.AnswerRow.from_dict(row) for row in lib.read_jsonl(answers_path)]


def _write_answers(directory: Path, rows, backup: bool) -> None:
    path = directory / lib.ANSWERS_FILENAME
    if backup:
        saved = lib.backup_file(path)
        if saved is not None:
            _progress(f"Backed up {path.name} to {saved.name}")
    lib.write_jsonl(path, rows)
    _progress(f"Wrote {path}")


def _open_session(args) -> tuple:
    """Build a logged-in session. Returns ``(session, client)``; the caller closes."""
    username, password = lib.credentials_from_environment()
    client = lib.HttpxClient(base_url=args.base_url, timeout=args.timeout)
    session = lib.AuthenticatedSession(client, username, password, pause_seconds=args.pause_seconds)
    session.authenticate()
    _progress(f"Logged in as {username} at {args.base_url}")
    return session, client


def _progress_reporter(total: int, journal: Path | None = None):
    """Per-row progress line, plus an append to the answers journal when given one."""
    state = {"done": 0}

    def report(row: lib.AnswerRow) -> None:
        state["done"] += 1
        status = f"ERROR[{row.error_class}]" if row.error else f"{row.elapsed_seconds:.1f}s"
        _progress(
            f"[{state['done']}/{total}] {row.dataset} / {row.search_type} / "
            f"{row.question_id}: {status}"
        )
        if journal is not None:
            lib.append_jsonl(journal, row)

    return report


def _verdict_reporter(total: int, journal: Path | None = None):
    """Per-verdict progress line, plus an append to the verdicts journal when given one."""
    state = {"done": 0}

    def report(row: lib.VerdictRow) -> None:
        state["done"] += 1
        _progress(f"[judge {state['done']}/{total}] {row.question_id} ({row.search_type})")
        if journal is not None:
            lib.append_jsonl(journal, row)

    return report


def _run_matrix(args, questions, datasets, search_types, run_directory: Path) -> list:
    session, client = _open_session(args)
    total = len(datasets) * len(search_types) * len(questions)
    journal = run_directory / lib.PARTIAL_ANSWERS_FILENAME
    try:
        rows = lib.run_answers(
            session,
            questions=questions,
            datasets=datasets,
            search_types=search_types,
            top_k=args.top_k,
            on_row=_progress_reporter(total, journal),
        )
    finally:
        client.close()
    if session.login_count > 1:
        _progress(f"Re-authenticated {session.login_count - 1} time(s) during the run")
    return rows


def _answer_rows(args, failed, by_id, run_directory: Path) -> list:
    """Re-answer exactly the failed cells, one at a time, preserving their top_k."""
    session, client = _open_session(args)
    report = _progress_reporter(len(failed), run_directory / lib.PARTIAL_ANSWERS_FILENAME)
    rows = []
    try:
        for row in failed:
            question = by_id[row.question_id]
            fresh = lib.run_answers(
                session,
                questions=[question],
                datasets=[row.dataset],
                search_types=[row.search_type],
                top_k=row.top_k or args.top_k,
                on_row=report,
            )
            rows.extend(fresh)
    finally:
        client.close()
    if session.login_count > 1:
        _progress(f"Re-authenticated {session.login_count - 1} time(s) during the resume")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Before anything imports cognee: its LLMConfig is cached on first read, so a
    # key mapped in later is never seen and every judge call fails with
    # LLMAPIKeyNotSetError - the way the first live run graded nothing.
    lib.configure_llm_environment()

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

    if args.judge_only and args.resume:
        parser.error("--judge-only and --resume are mutually exclusive")

    existing_verdicts: list[lib.VerdictRow] = []
    reanswered: list[lib.AnswerRow] = []

    if args.judge_only:
        source_directory = Path(args.judge_only)
        answer_rows = _load_answers(source_directory)
        if answer_rows is None:
            return 1
        run_directory = Path(args.out) if args.out else source_directory
        run_directory.mkdir(parents=True, exist_ok=True)
        _progress(f"Re-judging {len(answer_rows)} saved answer(s) from {source_directory}")

    elif args.resume:
        source_directory = Path(args.resume)
        previous = _load_answers(source_directory)
        if previous is None:
            return 1
        run_directory = Path(args.out) if args.out else source_directory
        run_directory.mkdir(parents=True, exist_ok=True)

        recovered = lib.recover_partial_answers(source_directory, previous)
        recovered_count = sum(1 for old, new in zip(previous, recovered) if old is not new)
        if recovered_count:
            _progress(
                f"Recovered {recovered_count} row(s) from an interrupted run's "
                f"{lib.PARTIAL_ANSWERS_FILENAME}"
            )
        previous = recovered

        failed = lib.rows_needing_answers(previous)
        _progress(
            f"Resuming {source_directory}: {len(failed)} of {len(previous)} row(s) to re-answer "
            f"({_histogram_line(lib.error_histogram(previous))})"
        )
        if not failed:
            _progress("Nothing to resume: every row already has an answer.")

        by_id = {question.id: question for question in questions}
        missing = sorted({row.question_id for row in failed if row.question_id not in by_id})
        if missing:
            print(
                "Cannot resume: no question definition for " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1

        reanswered = _answer_rows(args, failed, by_id, run_directory) if failed else []
        answer_rows = lib.merge_answer_rows(previous, reanswered)

        verdicts_path = source_directory / lib.VERDICTS_FILENAME
        if verdicts_path.exists():
            existing_verdicts = [
                lib.VerdictRow.from_dict(row) for row in lib.read_jsonl(verdicts_path)
            ]
        existing_verdicts = lib.recover_partial_verdicts(
            source_directory, existing_verdicts, answer_rows
        )

        _write_answers(run_directory, answer_rows, backup=run_directory == source_directory)

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

        answer_rows = _run_matrix(args, questions, datasets, search_types, run_directory)
        _write_answers(run_directory, answer_rows, backup=False)

    if args.no_judge:
        aggregates = lib.aggregate_answers_only(answer_rows)
        verdict_rows: list[lib.VerdictRow] = []
        histogram = lib.error_histogram(answer_rows)
    else:
        lib.configure_llm_environment()
        if args.resume:
            pending = lib.rows_needing_verdicts(answer_rows, existing_verdicts, reanswered)
            _progress(
                f"Judging {len(pending)} of {len(answer_rows)} row(s); "
                f"{len(answer_rows) - len(pending)} already graded"
            )
        else:
            pending = list(answer_rows)
        report_verdict = _verdict_reporter(
            len(pending), run_directory / lib.PARTIAL_VERDICTS_FILENAME
        )
        fresh = asyncio.run(lib.run_judge(pending, questions, on_row=report_verdict))
        verdict_rows = lib.merge_verdict_rows(existing_verdicts, fresh, answer_rows)
        backup = args.resume and run_directory == Path(args.resume)
        if backup:
            lib.backup_file(run_directory / lib.VERDICTS_FILENAME)
        lib.write_jsonl(run_directory / lib.VERDICTS_FILENAME, verdict_rows)
        _progress(f"Wrote {run_directory / lib.VERDICTS_FILENAME}")
        aggregates = lib.aggregate(verdict_rows)
        histogram = lib.error_histogram(verdict_rows)

    # Both main files are complete now; the journals have nothing left to protect.
    lib.clear_partial_files(run_directory)

    report = lib.render_report(
        aggregates,
        title=f"Recall evaluation - {run_directory.name}",
        error_histogram=histogram,
    )
    (run_directory / lib.REPORT_FILENAME).write_text(report, encoding="utf-8")
    print(report)

    if args.spot_check > 0 and verdict_rows:
        sample = lib.spot_check_sample(verdict_rows, args.spot_check, args.seed)
        print(f"\nSpot check ({len(sample)} of {len(verdict_rows)} verdicts, seed {args.seed}):\n")
        print(lib.format_spot_check(sample, questions))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
