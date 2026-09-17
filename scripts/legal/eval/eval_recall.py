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
import uuid
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
        "--judge-concurrency",
        type=int,
        default=4,
        help="Judge calls in flight at once (default 4; 1 = one verdict at a time).",
    )
    parser.add_argument(
        "--answer-concurrency",
        type=int,
        default=1,
        help=(
            "Cells answered at once (default 1: one at a time, in matrix order). Cells are "
            "independent, so 4 cuts a repeat run's wall clock by about 4x on a local server; "
            "keep it at 1 against a server other people are using."
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
        "--label",
        default=None,
        help=(
            "Free-text label for the run manifest, e.g. the server's code version. "
            "The manifest records the harness's own commit separately."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the question files and exit (0 valid, 1 invalid).",
    )
    parser.add_argument(
        "--analyze",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Attribute the missed gold facts of a finished, judged run to retrieval "
            "(the fact was not in the retrieved context) or generation (it was, and the "
            "answer left it out). Makes no search calls; one attribution call per cell "
            "with misses. Writes attribution.jsonl and re-renders report.md in place."
        ),
    )
    parser.add_argument(
        "--attribute",
        action="store_true",
        help="After judging a run, also run miss attribution on it (see --analyze).",
    )
    parser.add_argument(
        "--per-question",
        action="store_true",
        help="Add a per-question table to the report (one line per graded cell).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run the whole matrix N times, each into RUN_DIR/repeat-<i>/ on fresh sessions, "
            "and report the mean and spread of coverage per dataset x search type. This is "
            "how you tell a five-point delta from the noise floor. Default 1."
        ),
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


def _fold_journals(source: Path, destination: Path, previous, write: bool = True):
    """Overlay the answers journal onto ``previous``, optionally persisting it."""
    recovered = lib.recover_partial_answers(source, previous)
    count = sum(1 for old, new in zip(previous, recovered) if old is not new)
    if count:
        _progress(
            f"Recovered {count} row(s) from an interrupted run's {lib.PARTIAL_ANSWERS_FILENAME}"
        )
        if write:
            _write_answers(destination, recovered, backup=destination == source)
    return recovered


def _fold_verdict_journal(source: Path, destination: Path, order, write: bool = True):
    """Load the saved verdicts, overlay their journal, optionally persist them."""
    verdicts_path = source / lib.VERDICTS_FILENAME
    existing = (
        [lib.VerdictRow.from_dict(row) for row in lib.read_jsonl(verdicts_path)]
        if verdicts_path.exists()
        else []
    )
    recovered = lib.recover_partial_verdicts(source, existing, order)
    if write and len(recovered) != len(existing):
        _progress(
            f"Recovered {len(recovered) - len(existing)} verdict(s) from "
            f"{lib.PARTIAL_VERDICTS_FILENAME}"
        )
        if destination == source:
            lib.backup_file(destination / lib.VERDICTS_FILENAME)
        lib.write_jsonl(destination / lib.VERDICTS_FILENAME, recovered)
    return recovered


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


# One nonce per process: a --resume retries a cell on a fresh session rather than the
# one the failed attempt may already have written a QA turn to.
_PROCESS_NONCE = uuid.uuid4().hex[:8]


def _session_prefix(run_directory: Path) -> str:
    """Namespace for this run's per-cell sessions, so two runs never share one.

    A ``--repeats`` child is namespaced by its parent too, so ``repeat-1`` of two
    different runs in one process (the nonce is per process) cannot collide.
    """
    if run_directory.name.startswith(lib.REPEAT_DIRECTORY_PREFIX):
        return f"eval-{run_directory.parent.name}-{run_directory.name}-{_PROCESS_NONCE}"
    return f"eval-{run_directory.name}-{_PROCESS_NONCE}"


def _attribution_reporter(total_cells: int, journal: Path | None = None):
    """Per-fact progress line, plus an append to the attribution journal when given one."""
    state = {"done": 0}

    def report(row: lib.AttributionRow) -> None:
        state["done"] += 1
        verdict = (
            "error"
            if row.error
            else {True: "generation", False: "retrieval", None: "unclassified"}[row.in_context]
        )
        _progress(
            f"[attribute {state['done']}] {row.question_id} ({row.search_type}) "
            f"fact {row.fact_index}: {verdict}"
        )
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
            session_prefix=_session_prefix(run_directory),
            on_row=_progress_reporter(total, journal),
            concurrency=args.answer_concurrency,
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
                session_prefix=_session_prefix(run_directory),
                on_row=report,
            )
            rows.extend(fresh)
    finally:
        client.close()
    if session.login_count > 1:
        _progress(f"Re-authenticated {session.login_count - 1} time(s) during the resume")
    return rows


def _fresh_run(
    args,
    questions,
    question_paths,
    datasets,
    search_types,
    run_directory: Path,
    repeats: int | None = None,
    repeat_index: int | None = None,
) -> list:
    """Answer the whole matrix into ``run_directory`` (manifest, pending matrix, journal)."""
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = lib.write_manifest(
        run_directory,
        base_url=args.base_url,
        question_paths=question_paths,
        datasets=datasets,
        search_types=search_types,
        top_k=args.top_k,
        timeout=args.timeout,
        pause_seconds=args.pause_seconds,
        label=args.label,
        repeats=repeats,
        repeat_index=repeat_index,
    )
    _progress(f"Wrote {manifest_path}")

    # The whole matrix, marked not attempted, before the first request. A run
    # killed in its first pass used to leave --resume nothing to resume from;
    # now the matrix is always on disk and the journal decides what is done.
    pending = lib.pending_matrix(questions, datasets, search_types, top_k=args.top_k)
    _write_answers(run_directory, pending, backup=False)

    fresh = _run_matrix(args, questions, datasets, search_types, run_directory)
    answer_rows = lib.merge_answer_rows(pending, fresh)
    _write_answers(run_directory, answer_rows, backup=False)
    return answer_rows


def _judge_rows(args, run_directory: Path, answer_rows, existing_verdicts, reanswered, questions):
    """Grade what still needs grading and write ``verdicts.jsonl``."""
    lib.configure_llm_environment()
    if args.resume:
        pending = lib.rows_needing_verdicts(answer_rows, existing_verdicts, reanswered)
        _progress(
            f"Judging {len(pending)} of {len(answer_rows)} row(s); "
            f"{len(answer_rows) - len(pending)} already graded"
        )
    else:
        pending = list(answer_rows)
    report_verdict = _verdict_reporter(len(pending), run_directory / lib.PARTIAL_VERDICTS_FILENAME)
    fresh = asyncio.run(
        lib.run_judge(pending, questions, on_row=report_verdict, concurrency=args.judge_concurrency)
    )
    verdict_rows = lib.merge_verdict_rows(existing_verdicts, fresh, answer_rows)
    backup = bool(args.resume) and run_directory == Path(args.resume)
    if backup:
        lib.backup_file(run_directory / lib.VERDICTS_FILENAME)
    lib.write_jsonl(run_directory / lib.VERDICTS_FILENAME, verdict_rows)
    _progress(f"Wrote {run_directory / lib.VERDICTS_FILENAME}")
    return verdict_rows


def _attribute_rows(args, run_directory: Path, answer_rows, verdict_rows, questions) -> list:
    """Run miss attribution over graded cells and write ``attribution.jsonl``."""
    lib.configure_llm_environment()
    graded = [row for row in verdict_rows if not row.error and row.coverage is not None]
    _progress(f"Attributing the misses of {len(graded)} graded cell(s)")
    report = _attribution_reporter(len(graded), run_directory / lib.PARTIAL_ATTRIBUTION_FILENAME)
    rows = asyncio.run(
        lib.run_attribution(
            answer_rows,
            verdict_rows,
            questions,
            on_row=report,
            concurrency=args.judge_concurrency,
        )
    )
    lib.backup_file(run_directory / lib.ATTRIBUTION_FILENAME)
    lib.write_jsonl(run_directory / lib.ATTRIBUTION_FILENAME, rows)
    _progress(f"Wrote {run_directory / lib.ATTRIBUTION_FILENAME}")
    partial = run_directory / lib.PARTIAL_ATTRIBUTION_FILENAME
    if partial.exists():
        partial.unlink()
    return rows


def _load_verdicts(directory: Path, order) -> list | None:
    verdicts_path = directory / lib.VERDICTS_FILENAME
    if not verdicts_path.exists():
        print(f"No {lib.VERDICTS_FILENAME} in {directory}; judge the run first", file=sys.stderr)
        return None
    return _fold_verdict_journal(directory, directory, order, write=True)


def _write_report(
    run_directory: Path,
    aggregates,
    histogram,
    verdict_rows,
    args,
    attribution_rows=None,
    repeats=None,
    title: str | None = None,
) -> str:
    report = lib.render_report(
        aggregates,
        title=title or f"Recall evaluation - {run_directory.name}",
        error_histogram=histogram,
        categories=lib.aggregate_by_category(verdict_rows) if verdict_rows else None,
        questions=lib.per_question_lines(verdict_rows)
        if args.per_question and verdict_rows
        else None,
        attribution=lib.aggregate_attribution(attribution_rows) if attribution_rows else None,
        repeats=repeats,
    )
    (run_directory / lib.REPORT_FILENAME).write_text(report, encoding="utf-8")
    return report


def _analyze(args, questions) -> int:
    """``--analyze RUN_DIR``: attribute a judged run's misses, re-render its report."""
    run_directory = Path(args.analyze)
    answer_rows = _load_answers(run_directory)
    if answer_rows is None:
        return 1
    answer_rows = _fold_journals(run_directory, run_directory, answer_rows)
    verdict_rows = _load_verdicts(run_directory, answer_rows)
    if verdict_rows is None:
        return 1

    attribution_rows = _attribute_rows(args, run_directory, answer_rows, verdict_rows, questions)
    aggregates = lib.aggregate(verdict_rows)
    histogram = lib.error_histogram(verdict_rows)
    report = _write_report(
        run_directory, aggregates, histogram, verdict_rows, args, attribution_rows=attribution_rows
    )
    print(report)
    return 0


def _repeat_runs(args, parser, questions, question_paths, datasets, search_types) -> int:
    """``--repeats N``: N fresh runs into ``RUN_DIR/repeat-<i>/``, then the spread."""
    run_directory = Path(args.out) if args.out else lib.default_run_directory(DEFAULT_RUNS_ROOT)
    if run_directory.exists() and any(run_directory.iterdir()):
        parser.error(f"{run_directory} is not empty; --repeats needs a fresh --out")
    run_directory.mkdir(parents=True, exist_ok=True)
    lib.write_manifest(
        run_directory,
        base_url=args.base_url,
        question_paths=question_paths,
        datasets=datasets,
        search_types=search_types,
        top_k=args.top_k,
        timeout=args.timeout,
        pause_seconds=args.pause_seconds,
        label=args.label,
        repeats=args.repeats,
    )

    runs: list[list] = []
    all_attribution: list = []
    for index in range(1, args.repeats + 1):
        child = run_directory / f"{lib.REPEAT_DIRECTORY_PREFIX}{index}"
        _progress(f"=== repeat {index} of {args.repeats} -> {child}")
        answer_rows = _fresh_run(
            args,
            questions,
            question_paths,
            datasets,
            search_types,
            child,
            repeats=args.repeats,
            repeat_index=index,
        )
        verdict_rows = _judge_rows(args, child, answer_rows, [], [], questions)
        attribution_rows = (
            _attribute_rows(args, child, answer_rows, verdict_rows, questions)
            if args.attribute
            else None
        )
        lib.clear_partial_files(child)
        _write_report(
            child,
            lib.aggregate(verdict_rows),
            lib.error_histogram(verdict_rows),
            verdict_rows,
            args,
            attribution_rows=attribution_rows,
            title=f"Recall evaluation - {run_directory.name} / {child.name}",
        )
        runs.append(verdict_rows)
        if attribution_rows:
            all_attribution.extend(attribution_rows)

    pooled = [row for run in runs for row in run]
    report = _write_report(
        run_directory,
        lib.aggregate(pooled),
        lib.error_histogram(pooled),
        pooled,
        args,
        attribution_rows=all_attribution or None,
        repeats=lib.aggregate_repeats(runs),
        title=f"Recall evaluation - {run_directory.name} ({args.repeats} repeats, pooled)",
    )
    print(report)
    return 0


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
    if args.judge_only and args.no_judge:
        parser.error("--judge-only with --no-judge would do nothing at all")
    if not 0.0 <= args.spot_check <= 1.0:
        parser.error(f"--spot-check must be a fraction in [0, 1], not {args.spot_check}")
    if args.judge_concurrency < 1:
        parser.error("--judge-concurrency must be at least 1")
    if args.answer_concurrency < 1:
        parser.error("--answer-concurrency must be at least 1")
    if args.top_k < 1:
        parser.error(f"--top-k must be at least 1, not {args.top_k}")
    if args.repeats < 1:
        parser.error(f"--repeats must be at least 1, not {args.repeats}")
    if args.analyze and (args.judge_only or args.resume or args.no_judge or args.repeats > 1):
        parser.error("--analyze takes a finished run directory and nothing else")
    if args.attribute and args.no_judge:
        parser.error("--attribute needs verdicts; drop --no-judge")
    if args.repeats > 1 and (args.judge_only or args.resume):
        parser.error(
            "--repeats runs the matrix afresh; it cannot be combined with --judge-only or --resume"
        )
    if args.repeats > 1 and args.no_judge:
        parser.error("--repeats measures the spread of graded coverage; drop --no-judge")

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

    if args.analyze:
        return _analyze(args, questions)

    existing_verdicts: list[lib.VerdictRow] = []
    reanswered: list[lib.AnswerRow] = []

    if args.judge_only:
        source_directory = Path(args.judge_only)
        previous = _load_answers(source_directory)
        if previous is None:
            return 1
        run_directory = Path(args.out) if args.out else source_directory
        run_directory.mkdir(parents=True, exist_ok=True)

        # Fold the journals in exactly as --resume does, BEFORE anything clears
        # them. A run that died between appending a row and rewriting the main
        # file keeps its answers only in the journal; judging without folding
        # would grade the stale matrix and then delete the evidence.
        answer_rows = _fold_journals(source_directory, run_directory, previous)
        existing_verdicts = _fold_verdict_journal(source_directory, run_directory, answer_rows)
        if run_directory != source_directory:
            source_manifest = source_directory / lib.MANIFEST_FILENAME
            if source_manifest.exists():
                # A re-graded copy keeps the provenance of the run it grades.
                (run_directory / lib.MANIFEST_FILENAME).write_bytes(source_manifest.read_bytes())
        _progress(f"Re-judging {len(answer_rows)} saved answer(s) from {source_directory}")

    elif args.resume:
        source_directory = Path(args.resume)
        previous = _load_answers(source_directory)
        if previous is None:
            return 1
        run_directory = Path(args.out) if args.out else source_directory
        run_directory.mkdir(parents=True, exist_ok=True)

        previous = _fold_journals(source_directory, run_directory, previous, write=False)

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

        existing_verdicts = _fold_verdict_journal(
            source_directory, run_directory, answer_rows, write=False
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

        if args.repeats > 1:
            return _repeat_runs(args, parser, questions, question_paths, datasets, search_types)

        run_directory = Path(args.out) if args.out else lib.default_run_directory(DEFAULT_RUNS_ROOT)
        if (run_directory / lib.ANSWERS_FILENAME).exists():
            # The pending matrix is written before the first request, so a fresh run
            # into a populated directory would erase its answers up front.
            parser.error(
                f"{run_directory} already holds {lib.ANSWERS_FILENAME}; "
                "use --resume to continue it or a fresh --out"
            )
        answer_rows = _fresh_run(
            args, questions, question_paths, datasets, search_types, run_directory
        )

    attribution_rows = None
    if args.no_judge:
        aggregates = lib.aggregate_answers_only(answer_rows)
        verdict_rows: list[lib.VerdictRow] = []
        histogram = lib.error_histogram(answer_rows)
    else:
        verdict_rows = _judge_rows(
            args, run_directory, answer_rows, existing_verdicts, reanswered, questions
        )
        if args.attribute:
            attribution_rows = _attribute_rows(
                args, run_directory, answer_rows, verdict_rows, questions
            )
        aggregates = lib.aggregate(verdict_rows)
        histogram = lib.error_histogram(verdict_rows)

    # Both main files are complete now; the journals have nothing left to protect.
    lib.clear_partial_files(run_directory)

    report = _write_report(
        run_directory, aggregates, histogram, verdict_rows, args, attribution_rows=attribution_rows
    )
    print(report)

    if args.spot_check > 0 and verdict_rows:
        sample = lib.spot_check_sample(verdict_rows, args.spot_check, args.seed)
        print(f"\nSpot check ({len(sample)} of {len(verdict_rows)} verdicts, seed {args.seed}):\n")
        print(lib.format_spot_check(sample, questions))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
