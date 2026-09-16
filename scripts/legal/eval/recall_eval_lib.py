"""Logic behind ``eval_recall.py``: question files, HTTP answering, LLM judging.

Split out of the CLI so it can be unit tested. Two constraints shape this module
and should survive edits:

1. **No cognee imports at module scope.** ``eval_recall.py --help`` must not pay
   for importing cognee, and the unit tests load this file by path and must not
   touch ``~/.cognee``, the network, or an LLM. Everything cognee-side is behind
   a lazy call: :data:`LLMGateway` is a proxy whose method imports the real
   gateway on first use (so ``patch.object(lib.LLMGateway, ...)`` still works),
   and the prompt loaders are injectable.
2. **Answers come over HTTP, from a running server.** The harness never runs
   cognee in-process, so what it measures is the deployment as an agent would
   see it. The HTTP client is a small protocol (:class:`HttpClient`) so tests
   can hand in a fake.

Secrets are never printed: the password comes from the environment and the bearer
token is held in memory and passed as a header, never logged or written to a run
directory.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_TOP_K = 15

# The local stack's documented defaults. Override with COGNEE_EVAL_USER /
# COGNEE_EVAL_PASSWORD; neither value is ever printed.
DEFAULT_USER_ENV = "COGNEE_EVAL_USER"
DEFAULT_PASSWORD_ENV = "COGNEE_EVAL_PASSWORD"
DEFAULT_USER = "default_user@example.com"
DEFAULT_PASSWORD = "default_password"

LOGIN_PATH = "/api/v1/auth/login"
SEARCH_PATH = "/api/v1/search"
RECALL_PATH = "/api/v1/recall"

#: Pseudo search type: ask ``/recall`` with ``search_type: null`` and let the
#: router pick the strategy. Not a ``SearchType`` member, so it is checked by name.
AUTO_SEARCH_TYPE = "AUTO"

JUDGE_SYSTEM_PROMPT_FILE = "eval_judge_system.txt"
JUDGE_USER_PROMPT_FILE = "eval_judge_user.txt"

QUESTION_CATEGORIES = (
    "summary",
    "disputes",
    "who_said_what",
    "valuation",
    "timeline",
    "procedure",
    "references",
)

ANSWERS_FILENAME = "answers.jsonl"
VERDICTS_FILENAME = "verdicts.jsonl"
REPORT_FILENAME = "report.md"


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


def configure_llm_environment() -> None:
    """Point cognee at the developer's home install and map the OpenAI key.

    Same preamble the other ``scripts/legal/*.py`` use. ``setdefault`` only, so a
    caller's own environment always wins, and no value is read back or printed.
    Only the judge needs this - answers come over HTTP from a server that has its
    own configuration.
    """
    cognee_home = Path.home() / ".cognee"
    os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(cognee_home / "system"))
    os.environ.setdefault("DATA_ROOT_DIRECTORY", str(cognee_home / "data"))
    os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(cognee_home / "cache"))
    os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


# --------------------------------------------------------------------------------------
# Question files
# --------------------------------------------------------------------------------------


class QuestionFileError(ValueError):
    """A question file is malformed. Carries every problem found, not just the first."""


@dataclass(frozen=True)
class GoldFact:
    fact: str
    source: str


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    gold_facts: tuple[GoldFact, ...]
    must_not_claim: tuple[str, ...] = ()
    notes: str = ""
    corpus: str = ""


def _string_list(value: Any, label: str, problems: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        problems.append(f"{label}: must be a list of strings")
        return ()
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            problems.append(f"{label}[{index}]: must be a string")
            continue
        items.append(item)
    return tuple(items)


def _parse_gold_facts(value: Any, label: str, problems: list[str]) -> tuple[GoldFact, ...]:
    if not isinstance(value, list):
        problems.append(f"{label}: gold_facts must be a list")
        return ()
    if not value:
        problems.append(f"{label}: gold_facts is empty (a question needs at least one gold fact)")
        return ()
    facts: list[GoldFact] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            problems.append(f"{label}: gold_facts[{index}] must be an object")
            continue
        fact = raw.get("fact")
        source = raw.get("source")
        if not isinstance(fact, str) or not fact.strip():
            problems.append(f"{label}: gold_facts[{index}] is missing a non-empty 'fact'")
            continue
        if not isinstance(source, str) or not source.strip():
            problems.append(f"{label}: gold_facts[{index}] is missing a non-empty 'source'")
            continue
        facts.append(GoldFact(fact=fact.strip(), source=source.strip()))
    return tuple(facts)


def validate_question_file(path: str | Path) -> list[Question]:
    """Parse and validate one question file, or raise :class:`QuestionFileError`.

    Every problem in the file is reported at once - fixing question files one
    error per run is miserable.
    """
    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise QuestionFileError(f"{path}: file not found") from error
    except OSError as error:
        raise QuestionFileError(f"{path}: cannot be read ({error})") from error

    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise QuestionFileError(f"{path}: invalid JSON ({error})") from error

    problems: list[str] = []
    if not isinstance(document, dict):
        raise QuestionFileError(
            f"{path}: top level must be an object with 'corpus' and 'questions'"
        )

    corpus = document.get("corpus")
    if not isinstance(corpus, str) or not corpus.strip():
        problems.append("top level: missing a non-empty 'corpus'")
        corpus = ""

    raw_questions = document.get("questions")
    if not isinstance(raw_questions, list):
        problems.append("top level: 'questions' must be a list")
        raw_questions = []
    elif not raw_questions:
        problems.append("top level: 'questions' is empty")

    questions: list[Question] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_questions):
        label = f"questions[{index}]"
        if not isinstance(raw, dict):
            problems.append(f"{label}: must be an object")
            continue

        question_id = raw.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            problems.append(f"{label}: missing a non-empty 'id'")
            question_id = ""
        else:
            question_id = question_id.strip()
            label = f"questions[{index}] ({question_id})"
            if question_id in seen_ids:
                problems.append(f"{label}: duplicate id")
            seen_ids.add(question_id)

        category = raw.get("category")
        if not isinstance(category, str) or not category.strip():
            problems.append(f"{label}: missing a non-empty 'category'")
            category = ""
        elif category not in QUESTION_CATEGORIES:
            problems.append(
                f"{label}: unknown category {category!r} "
                f"(expected one of: {', '.join(QUESTION_CATEGORIES)})"
            )

        question_text = raw.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            problems.append(f"{label}: missing a non-empty 'question'")
            question_text = ""

        if "gold_facts" not in raw:
            problems.append(f"{label}: missing 'gold_facts'")
            gold_facts: tuple[GoldFact, ...] = ()
        else:
            gold_facts = _parse_gold_facts(raw["gold_facts"], label, problems)

        must_not_claim = _string_list(
            raw.get("must_not_claim"), f"{label}: must_not_claim", problems
        )

        notes = raw.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            problems.append(f"{label}: 'notes' must be a string")
            notes = ""

        unknown_keys = set(raw) - {
            "id",
            "category",
            "question",
            "gold_facts",
            "must_not_claim",
            "notes",
        }
        if unknown_keys:
            problems.append(f"{label}: unknown key(s): {', '.join(sorted(unknown_keys))}")

        questions.append(
            Question(
                id=question_id,
                category=category,
                question=question_text,
                gold_facts=gold_facts,
                must_not_claim=must_not_claim,
                notes=notes,
                corpus=corpus,
            )
        )

    if problems:
        joined = "\n  - ".join(problems)
        raise QuestionFileError(f"{path}: {len(problems)} problem(s):\n  - {joined}")

    return questions


def load_question_files(paths: Sequence[str | Path]) -> list[Question]:
    """Validate and concatenate several question files, rejecting cross-file id clashes."""
    questions: list[Question] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for question in validate_question_file(path):
            if question.id in seen:
                raise QuestionFileError(
                    f"{path}: duplicate question id {question.id!r} (also in {seen[question.id]})"
                )
            seen[question.id] = Path(path)
            questions.append(question)
    return questions


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


class HttpError(RuntimeError):
    """A request failed in a way that retrying will not fix (4xx, bad payload)."""


class TransientHttpError(HttpError):
    """A request failed in a way that one retry might fix (timeout, connection, 5xx)."""


_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "TransportError",
        "WriteError",
        "WriteTimeout",
    }
)


def is_transient(error: BaseException) -> bool:
    """Whether ``error`` is worth one retry.

    Checked by class name as well as type so an httpx error raised by a client
    this module did not construct is still classified correctly, without
    importing httpx at module scope.
    """
    if isinstance(error, TransientHttpError):
        return True
    if isinstance(error, HttpError):
        return False
    return type(error).__name__ in _TRANSIENT_EXCEPTION_NAMES


class HttpClient(Protocol):
    """The slice of an HTTP client this harness needs."""

    def post(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Any: ...


class HttpxClient:
    """The real client. ``httpx`` is imported here so the module stays import-light."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        import httpx

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def post(
        self,
        path: str,
        *,
        json: Optional[dict] = None,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> Any:
        try:
            response = self._client.post(path, json=json, data=data, headers=headers)
        except self._httpx.HTTPError as error:
            raise TransientHttpError(f"{type(error).__name__}: {error}") from error

        if response.status_code >= 500:
            raise TransientHttpError(f"HTTP {response.status_code}: {response.text[:400]}")
        if response.status_code >= 400:
            raise HttpError(f"HTTP {response.status_code}: {response.text[:400]}")
        try:
            return response.json()
        except ValueError as error:
            raise HttpError(f"HTTP {response.status_code}: response was not JSON") from error

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def credentials_from_environment() -> tuple[str, str]:
    """Read the eval account. Returns ``(username, password)``; never log the pair."""
    username = os.environ.get(DEFAULT_USER_ENV) or DEFAULT_USER
    password = os.environ.get(DEFAULT_PASSWORD_ENV) or DEFAULT_PASSWORD
    return username, password


def login(client: HttpClient, username: str, password: str) -> str:
    """Exchange credentials for a bearer token. The token is never printed or stored."""
    payload = client.post(LOGIN_PATH, data={"username": username, "password": password})
    token = payload.get("access_token") if isinstance(payload, dict) else None
    if not token:
        raise HttpError(f"Login for {username} returned no access_token")
    return str(token)


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def build_search_request(
    question: Question,
    dataset: str,
    search_type: str,
    top_k: int,
    only_context: bool,
) -> tuple[str, dict]:
    """The path and body for one call.

    ``AUTO`` goes to ``/recall`` with an explicit null ``search_type``, which is
    how the recall router opts into auto-routing; everything else goes to
    ``/search`` with the type pinned.
    """
    body: dict[str, Any] = {
        "query": question.question,
        "datasets": [dataset],
        "top_k": top_k,
    }
    if search_type == AUTO_SEARCH_TYPE:
        body["search_type"] = None
        path = RECALL_PATH
    else:
        body["search_type"] = search_type
        path = SEARCH_PATH
    if only_context:
        body["only_context"] = True
        body["context_format"] = "context"
    return path, body


_TEXT_KEYS = ("text", "search_result", "content", "answer", "value")


def _collect_text(node: Any, chunks: list[str], depth: int) -> None:
    if depth > 8 or node is None or isinstance(node, (bool, int, float)):
        return
    if isinstance(node, str):
        stripped = node.strip()
        if stripped:
            chunks.append(stripped)
        return
    if isinstance(node, list):
        for item in node:
            _collect_text(item, chunks, depth + 1)
        return
    if isinstance(node, dict):
        for key in _TEXT_KEYS:
            if key in node:
                _collect_text(node[key], chunks, depth + 1)
                return


def extract_text(payload: Any) -> str:
    """Flatten a ``/search`` or ``/recall`` response body into one string.

    The two endpoints disagree on shape - ``/search`` returns
    ``[{"search_result": ...}]`` while ``/recall`` returns normalized entries with
    a ``text`` field - and completion payloads are themselves lists of strings.
    One recursive walk over the known text-bearing keys covers both without the
    harness caring which endpoint answered. Repeated fragments (the same passage
    returned under several datasets) are emitted once.
    """
    chunks: list[str] = []
    _collect_text(payload, chunks, 0)
    seen: set[str] = set()
    unique: list[str] = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique.append(chunk)
    return "\n\n".join(unique)


@dataclass
class AnswerRow:
    dataset: str
    search_type: str
    question_id: str
    category: str
    question: str
    corpus: str = ""
    answer: str = ""
    context: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AnswerRow":
        known = {key: payload.get(key) for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def post_with_retry(
    client: HttpClient,
    path: str,
    body: dict,
    headers: dict,
    retries: int = 1,
) -> Any:
    """POST once, then retry ``retries`` times but only for transient failures."""
    attempt = 0
    while True:
        try:
            return client.post(path, json=body, headers=headers)
        except Exception as error:  # noqa: BLE001 - re-raised below once retries run out
            if attempt >= retries or not is_transient(error):
                raise
            attempt += 1


def run_answers(
    client: HttpClient,
    token: str,
    questions: Sequence[Question],
    datasets: Sequence[str],
    search_types: Sequence[str],
    top_k: int = DEFAULT_TOP_K,
    retries: int = 1,
    on_row: Optional[Callable[[AnswerRow], None]] = None,
) -> list[AnswerRow]:
    """Answer every ``dataset x search_type x question`` cell.

    Two calls per cell: the answer, then the same query with ``only_context`` so
    the judge can tell a fabrication from a faithful reading of bad context. A
    failure is recorded on the row and the run continues - one dead search type
    should not cost the whole matrix.
    """
    headers = auth_headers(token)
    rows: list[AnswerRow] = []

    for dataset in datasets:
        for search_type in search_types:
            for question in questions:
                row = AnswerRow(
                    dataset=dataset,
                    search_type=search_type,
                    question_id=question.id,
                    category=question.category,
                    question=question.question,
                    corpus=question.corpus,
                )
                started = time.perf_counter()
                try:
                    answer_path, answer_body = build_search_request(
                        question, dataset, search_type, top_k, only_context=False
                    )
                    row.answer = extract_text(
                        post_with_retry(client, answer_path, answer_body, headers, retries)
                    )
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    row.error = f"answer: {type(error).__name__}: {error}"
                else:
                    try:
                        context_path, context_body = build_search_request(
                            question, dataset, search_type, top_k, only_context=True
                        )
                        row.context = extract_text(
                            post_with_retry(client, context_path, context_body, headers, retries)
                        )
                    except Exception as error:  # noqa: BLE001 - recorded, not raised
                        row.error = f"context: {type(error).__name__}: {error}"
                row.elapsed_seconds = round(time.perf_counter() - started, 3)
                rows.append(row)
                if on_row is not None:
                    on_row(row)
    return rows


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """One graded answer. Coverage is derived, not asked for, so it cannot drift."""

    gold_facts_covered: list[str] = Field(default_factory=list)
    gold_facts_missed: list[str] = Field(default_factory=list)
    wrong_claims: list[str] = Field(default_factory=list)
    fabricated_claims: list[str] = Field(default_factory=list)
    stance_errors: list[str] = Field(default_factory=list)
    notes: str = ""


class _LazyLLMGateway:
    """Stand-in for ``cognee``'s gateway that defers the import to first use.

    Importing cognee at module scope would make ``--help`` slow and the unit
    tests non-hermetic. Because this is a module-level object with the same
    method name, ``patch.object(lib.LLMGateway, "acreate_structured_output", ...)``
    works exactly as it would against the real gateway.
    """

    @staticmethod
    async def acreate_structured_output(text_input: str, system_prompt: str, response_model: Any):
        from cognee.infrastructure.llm import LLMGateway as _gateway

        return await _gateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=response_model,
        )


LLMGateway = _LazyLLMGateway


def _default_read_prompt(filename: str) -> str:
    from cognee.infrastructure.llm.prompts import read_query_prompt

    return read_query_prompt(filename) or ""


def _default_render_prompt(filename: str, context: dict) -> str:
    from cognee.infrastructure.llm.prompts import render_prompt

    return render_prompt(filename, context)


def normalize_claim(text: str) -> str:
    """Case-folded, whitespace-collapsed form used by the must-not-claim floor."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def apply_must_not_claim_floor(
    verdict: JudgeVerdict,
    must_not_claim: Sequence[str],
    answer: str,
) -> JudgeVerdict:
    """Add any triggered trap to ``fabricated_claims``, judge or no judge.

    The deterministic floor exists because the judge is itself an LLM: a trap the
    corpus author wrote down by hand is a fact about the corpus, and a run should
    not be able to pass it by talking the grader round.
    """
    normalized_answer = normalize_claim(answer)
    if not normalized_answer:
        return verdict
    already = {normalize_claim(claim) for claim in verdict.fabricated_claims}
    additions = [
        trap
        for trap in must_not_claim
        if normalize_claim(trap)
        and normalize_claim(trap) in normalized_answer
        and normalize_claim(trap) not in already
    ]
    if not additions:
        return verdict
    return verdict.model_copy(
        update={"fabricated_claims": [*verdict.fabricated_claims, *additions]}
    )


async def judge_answer(
    row: AnswerRow,
    question: Question,
    read_prompt: Callable[[str], str] = _default_read_prompt,
    render_prompt: Callable[[str, dict], str] = _default_render_prompt,
) -> JudgeVerdict:
    """Grade one answer with the LLM judge, then apply the deterministic floor."""
    system_prompt = read_prompt(JUDGE_SYSTEM_PROMPT_FILE)
    if not system_prompt.strip():
        raise RuntimeError(f"Judge system prompt is missing or empty: {JUDGE_SYSTEM_PROMPT_FILE}")

    user_prompt = render_prompt(
        JUDGE_USER_PROMPT_FILE,
        {
            "question": question.question,
            "gold_facts": [
                {"fact": gold.fact, "source": gold.source} for gold in question.gold_facts
            ],
            "must_not_claim": list(question.must_not_claim),
            "notes": question.notes,
            "answer": row.answer,
            "context": row.context,
        },
    )

    verdict = await LLMGateway.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=JudgeVerdict,
    )
    return apply_must_not_claim_floor(verdict, question.must_not_claim, row.answer)


def coverage(verdict: JudgeVerdict) -> Optional[float]:
    """covered / (covered + missed), or ``None`` when the judge listed no gold facts."""
    total = len(verdict.gold_facts_covered) + len(verdict.gold_facts_missed)
    if total == 0:
        return None
    return len(verdict.gold_facts_covered) / total


@dataclass
class VerdictRow:
    dataset: str
    search_type: str
    question_id: str
    category: str
    question: str
    answer: str = ""
    coverage: Optional[float] = None
    gold_facts_covered: list[str] = field(default_factory=list)
    gold_facts_missed: list[str] = field(default_factory=list)
    wrong_claims: list[str] = field(default_factory=list)
    fabricated_claims: list[str] = field(default_factory=list)
    stance_errors: list[str] = field(default_factory=list)
    notes: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "VerdictRow":
        known = {key: payload.get(key) for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def verdict_row(row: AnswerRow, verdict: JudgeVerdict) -> VerdictRow:
    return VerdictRow(
        dataset=row.dataset,
        search_type=row.search_type,
        question_id=row.question_id,
        category=row.category,
        question=row.question,
        answer=row.answer,
        coverage=coverage(verdict),
        gold_facts_covered=list(verdict.gold_facts_covered),
        gold_facts_missed=list(verdict.gold_facts_missed),
        wrong_claims=list(verdict.wrong_claims),
        fabricated_claims=list(verdict.fabricated_claims),
        stance_errors=list(verdict.stance_errors),
        notes=verdict.notes,
        error=row.error,
    )


async def run_judge(
    rows: Sequence[AnswerRow],
    questions: Sequence[Question],
    read_prompt: Callable[[str], str] = _default_read_prompt,
    render_prompt: Callable[[str, dict], str] = _default_render_prompt,
    on_row: Optional[Callable[[VerdictRow], None]] = None,
) -> list[VerdictRow]:
    """Grade every answer row.

    An answer row that already failed is not sent to the judge - there is nothing
    to grade - but it still produces a verdict row so the ``n`` and ``errors``
    columns of the report agree with the matrix that was attempted.
    """
    by_id = {question.id: question for question in questions}
    verdicts: list[VerdictRow] = []

    for row in rows:
        question = by_id.get(row.question_id)
        if question is None:
            out = verdict_row(row, JudgeVerdict())
            out.error = (
                row.error or f"no question with id {row.question_id!r} in the question files"
            )
        elif row.error:
            out = verdict_row(
                row, JudgeVerdict(gold_facts_missed=[g.fact for g in question.gold_facts])
            )
            out.coverage = None
        else:
            try:
                verdict = await judge_answer(
                    row, question, read_prompt=read_prompt, render_prompt=render_prompt
                )
                out = verdict_row(row, verdict)
            except Exception as error:  # noqa: BLE001 - recorded, not raised
                out = verdict_row(row, JudgeVerdict())
                out.coverage = None
                out.error = f"judge: {type(error).__name__}: {error}"
        verdicts.append(out)
        if on_row is not None:
            on_row(out)
    return verdicts


# --------------------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------------------


@dataclass
class Aggregate:
    dataset: str
    search_type: str
    n: int = 0
    mean_coverage: Optional[float] = None
    wrong_claims: int = 0
    fabricated_claims: int = 0
    stance_errors: int = 0
    errors: int = 0


def aggregate(rows: Iterable[VerdictRow]) -> list[Aggregate]:
    """Roll verdict rows up per dataset x search type, preserving first-seen order."""
    buckets: dict[tuple[str, str], Aggregate] = {}
    coverages: dict[tuple[str, str], list[float]] = {}

    for row in rows:
        key = (row.dataset, row.search_type)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = Aggregate(dataset=row.dataset, search_type=row.search_type)
            buckets[key] = bucket
            coverages[key] = []
        bucket.n += 1
        if row.error:
            bucket.errors += 1
        if row.coverage is not None:
            coverages[key].append(row.coverage)
        bucket.wrong_claims += len(row.wrong_claims)
        bucket.fabricated_claims += len(row.fabricated_claims)
        bucket.stance_errors += len(row.stance_errors)

    for key, bucket in buckets.items():
        values = coverages[key]
        bucket.mean_coverage = (sum(values) / len(values)) if values else None
    return list(buckets.values())


def aggregate_answers_only(rows: Iterable[AnswerRow]) -> list[Aggregate]:
    """The ``--no-judge`` roll-up: counts and failures, no quality columns."""
    buckets: dict[tuple[str, str], Aggregate] = {}
    for row in rows:
        key = (row.dataset, row.search_type)
        bucket = buckets.setdefault(
            key, Aggregate(dataset=row.dataset, search_type=row.search_type)
        )
        bucket.n += 1
        if row.error:
            bucket.errors += 1
    return list(buckets.values())


def _format_coverage(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render_report(aggregates: Sequence[Aggregate], title: str = "Recall evaluation") -> str:
    """One markdown table, one row per dataset x search type."""
    lines = [f"# {title}", ""]
    if not aggregates:
        lines.append("_No results._")
        return "\n".join(lines) + "\n"

    header = (
        "| dataset | search type | n | mean coverage | wrong claims | "
        "fabricated claims | stance errors | errors |"
    )
    lines.append(header)
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in aggregates:
        lines.append(
            f"| {item.dataset} | {item.search_type} | {item.n} | "
            f"{_format_coverage(item.mean_coverage)} | {item.wrong_claims} | "
            f"{item.fabricated_claims} | {item.stance_errors} | {item.errors} |"
        )
    lines.append("")
    lines.append(
        "Coverage is the mean over graded questions of "
        "covered / (covered + missed) gold facts. Claim counts are totals, not rates."
    )
    return "\n".join(lines) + "\n"


def spot_check_sample(
    rows: Sequence[VerdictRow],
    fraction: float,
    seed: int,
) -> list[VerdictRow]:
    """A seeded sample of ``ceil(fraction * n)`` verdicts for hand review.

    Seeded so two people reviewing the same run look at the same rows, and
    returned in run order so the sample reads like a slice of the run.
    """
    if fraction <= 0 or not rows:
        return []
    count = min(len(rows), math.ceil(fraction * len(rows)))
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def format_spot_check(rows: Sequence[VerdictRow], questions: Sequence[Question]) -> str:
    """Render sampled verdicts with question, answer, gold facts and verdict."""
    by_id = {question.id: question for question in questions}
    blocks: list[str] = []
    for row in rows:
        question = by_id.get(row.question_id)
        gold = (
            "\n".join(f"    - {fact.fact}  [{fact.source}]" for fact in question.gold_facts)
            if question
            else "    (question not found)"
        )
        blocks.append(
            "\n".join(
                [
                    "=" * 78,
                    f"{row.dataset} / {row.search_type} / {row.question_id} [{row.category}]",
                    f"  Q: {row.question}",
                    "  Gold facts:",
                    gold,
                    f"  Answer: {row.answer.strip() or '(empty)'}",
                    f"  Coverage: {_format_coverage(row.coverage)}"
                    f"  covered={len(row.gold_facts_covered)}"
                    f" missed={len(row.gold_facts_missed)}",
                    f"  Wrong claims: {row.wrong_claims or '-'}",
                    f"  Fabricated: {row.fabricated_claims or '-'}",
                    f"  Stance errors: {row.stance_errors or '-'}",
                    f"  Judge notes: {row.notes or '-'}",
                    f"  Error: {row.error}" if row.error else "",
                ]
            ).rstrip()
        )
    return "\n".join(blocks)


# --------------------------------------------------------------------------------------
# Run directory I/O
# --------------------------------------------------------------------------------------


def default_run_directory(root: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / stamp


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_search_types(names: Sequence[str]) -> list[str]:
    """Validate search type names against cognee, plus the ``AUTO`` pseudo-type.

    cognee is imported here and nowhere earlier so ``--help`` stays free.
    """
    from cognee.modules.search.types import SearchType

    valid = {member.name for member in SearchType} | {AUTO_SEARCH_TYPE}
    resolved: list[str] = []
    unknown: list[str] = []
    for name in names:
        candidate = name.strip().upper()
        if not candidate:
            continue
        if candidate not in valid:
            unknown.append(name)
        elif candidate not in resolved:
            resolved.append(candidate)
    if unknown:
        raise ValueError(
            f"Unknown search type(s): {', '.join(unknown)}. Valid: {', '.join(sorted(valid))}"
        )
    if not resolved:
        raise ValueError("No search types given")
    return resolved
