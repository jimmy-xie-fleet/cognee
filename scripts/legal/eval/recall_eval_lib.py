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

import asyncio
import json
import math
import os
import random
import re
import sys
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
# 600 s, not the 180 s the first live run used. A HYBRID_COMPLETION over a 5k-node
# graph on a server that is also serving the UI and the memory plugin genuinely
# takes minutes, and the first run lost rows to ReadTimeout that the server would
# have answered. Efficiency is explicitly not a constraint for this evaluation:
# a slow answer is data, a timed-out answer is a hole in the table.
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_TOP_K = 15
#: Ceiling on one judge or attribution call. The gateway itself has no deadline, and a
#: five-dataset repeat run once sat for two hours on a single attribution call that never
#: returned. A call that hits this is recorded on the row as a ``TimeoutError`` -- a hole
#: ``--judge-only`` / ``--analyze`` repair -- instead of stalling the whole run.
LLM_CALL_TIMEOUT_SECONDS = 300.0

#: Three attempts for a transient failure, sleeping between them. The first live
#: run retried once and lost rows anyway - a shared server under load needs to be
#: given real time to recover, not hammered twice in a row.
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = (2.0, 8.0)
DEFAULT_JITTER = 0.25

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
# Miss attribution: the second grader, asked one narrower question per cell -
# of the gold facts the judge marked missed, which ones were in the retrieved
# context at all. Splits a coverage gap into "retrieval never surfaced it" and
# "it was there and the answer left it out", which point at different fixes.
ATTRIBUTION_SYSTEM_PROMPT_FILE = "eval_attribution_system.txt"
ATTRIBUTION_USER_PROMPT_FILE = "eval_attribution_user.txt"

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
MANIFEST_FILENAME = "run.json"

#: Written for every cell before the first request, so a run that dies in its
#: first pass still leaves ``--resume`` a complete matrix to work from - the
#: journal, not the matrix, decides what is done.
PENDING_ERROR = "pending: not attempted"

#: Cue words that make a sentence a denial rather than a claim. Deliberately
#: blunt: the span floor is meant to be conservative, and the LLM judge is the
#: primary detector for anything subtler than this. Note what is NOT here: the
#: bare verb "deny". In a legal answer "the defendants deny paragraph 12" is
#: substantive content, not a denial of the trap, and treating it as a cue would
#: silence the floor across most of this corpus. "denies"/"denied" are cues
#: because they read as the answer reporting an absence.
NEGATION_CUES = (
    "not",
    "no",
    "never",
    "denies",
    "denied",
    "without",
    "cannot",
    "unsupported",
)
# Crash journals: every row is appended here the moment it exists, so a run that
# dies (disk full, server down, killed) loses nothing a --resume cannot recover.
# Folded into the main files and deleted once a run finishes writing them.
PARTIAL_ANSWERS_FILENAME = "answers.partial.jsonl"
PARTIAL_VERDICTS_FILENAME = "verdicts.partial.jsonl"
REPORT_FILENAME = "report.md"
ATTRIBUTION_FILENAME = "attribution.jsonl"
PARTIAL_ATTRIBUTION_FILENAME = "attribution.partial.jsonl"
#: ``--repeats N`` writes each repetition to ``<run>/repeat-<i>/`` and the roll-up
#: to ``<run>/report.md``; the prefix is what tells the two layouts apart.
REPEAT_DIRECTORY_PREFIX = "repeat-"


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
    # Only ever *add* a key, never blank one out. The sibling scripts use
    # ``setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))``, which
    # with neither variable set exports the empty string - and an empty env var
    # wins over the value in .env, so cognee raises LLMAPIKeyNotSetError. That is
    # how the first live run answered 77 questions and graded none of them.
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = openai_key


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
class Trap:
    """One claim the documents do not support, plus the spans that betray it.

    ``claim`` is the full sentence, which is what the LLM judge reads. ``match``
    holds the short literals a real answer would actually contain - an amount, a
    date, a resolution number, a distinctive phrase - because an answer never
    contains the trap sentence verbatim, so matching on ``claim`` can only ever
    find nothing. A trap with no spans is judge-only by design.
    """

    claim: str
    match: tuple[str, ...] = ()


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    gold_facts: tuple[GoldFact, ...]
    must_not_claim: tuple[Trap, ...] = ()
    notes: str = ""
    corpus: str = ""


def _parse_traps(value: Any, label: str, problems: list[str]) -> tuple[Trap, ...]:
    """Parse ``must_not_claim``: a bare string, or ``{"claim":..., "match":[...]}``.

    The bare string is kept for compatibility with gold files written before
    spans existed; it parses to a trap the floor will never fire on.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        problems.append(f"{label}: must_not_claim must be a list")
        return ()

    traps: list[Trap] = []
    for index, raw in enumerate(value):
        where = f"{label}: must_not_claim[{index}]"
        if isinstance(raw, str):
            if not raw.strip():
                problems.append(f"{where}: is an empty string")
                continue
            traps.append(Trap(claim=raw.strip()))
            continue
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be a string or an object with 'claim'")
            continue

        unknown = set(raw) - {"claim", "match"}
        if unknown:
            problems.append(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")
        claim = raw.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            problems.append(f"{where}: is missing a non-empty 'claim'")
            continue
        spans_value = raw.get("match", [])
        if spans_value is None:
            spans_value = []
        if not isinstance(spans_value, list):
            problems.append(f"{where}: 'match' must be a list of short literal spans")
            spans_value = []
        spans: list[str] = []
        for span_index, span in enumerate(spans_value):
            if not isinstance(span, str) or not span.strip():
                problems.append(f"{where}: match[{span_index}] must be a non-empty string")
                continue
            spans.append(span.strip())
        traps.append(Trap(claim=claim.strip(), match=tuple(spans)))
    return tuple(traps)


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

        must_not_claim = _parse_traps(raw.get("must_not_claim"), label, problems)

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
    """A request failed in a way that a retry might fix (timeout, connection, 5xx)."""


class AuthError(HttpError):
    """The server rejected the bearer token (401).

    Its own class because the cure is different from both other cases: not a
    retry and not a dead row, but a fresh login. The first live run lost 84 of
    171 rows to a token that expired about an hour in.
    """


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


#: Error classes for the report histogram. Derived from the recorded message so
#: a run answered by an older version of this harness still classifies.
ERROR_CLASSES = ("pending", "empty", "401", "timeout", "5xx", "other")


def classify_error_message(message: Optional[str]) -> str:
    """Bucket a recorded error into one of :data:`ERROR_CLASSES` (or ``""``).

    401 is checked first: an expired token that surfaces as ``HTTP 401`` must not
    be counted as "other" just because the word appears late in the string.
    """
    if not message:
        return ""
    lowered = message.lower()
    if lowered.startswith("pending"):
        return "pending"
    if lowered.startswith("answer: empty") or lowered.startswith("context: empty"):
        return "empty"
    if "401" in lowered or "unauthorized" in lowered:
        return "401"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if re.search(r"http 5\d\d", lowered):
        return "5xx"
    return "other"


def is_transient(error: BaseException) -> bool:
    """Whether ``error`` is worth one retry.

    Checked by class name as well as type so an httpx error raised by a client
    this module did not construct is still classified correctly, without
    importing httpx at module scope.
    """
    if isinstance(error, TransientHttpError):
        return True
    if isinstance(error, HttpError):  # includes AuthError, which is handled separately
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
        if response.status_code == 401:
            raise AuthError(f"HTTP 401: {response.text[:400]}")
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


def cell_session_id(prefix: str, dataset: str, search_type: str, question_id: str) -> str:
    """The per-cell session name.

    Every cell gets its own session because both endpoints fall back to the
    caller's *default* session when none is given: without this, question 19's
    retrieval is shaped by the conversation questions 1-18 left behind, and two
    runs of the same matrix are not comparable.
    """
    return f"{prefix}-{dataset}-{search_type}-{question_id}"


def build_search_request(
    question: Question,
    dataset: str,
    search_type: str,
    top_k: int,
    only_context: bool,
    session_id: Optional[str] = None,
) -> tuple[str, dict]:
    """The path and body for one call.

    ``AUTO`` goes to ``/recall`` with an explicit null ``search_type``, which is
    how the recall router opts into auto-routing; everything else goes to
    ``/search`` with the type pinned. Both DTOs take ``session_id`` as a plain
    optional string (verified in ``get_search_router.py`` / ``get_recall_router.py``).
    """
    body: dict[str, Any] = {
        "query": question.question,
        "datasets": [dataset],
        "top_k": top_k,
    }
    if session_id:
        body["session_id"] = session_id
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


#: Statuses the server uses for a system marker instead of an answer
#: (``ResponseMarkerEntry`` in ``cognee/modules/recall/types/RecallResponse.py``).
MARKER_STATUSES = ("memory_warming_up", "build_failed")


def detect_marker(payload: Any) -> Optional[str]:
    """The marker status when the server answered with a marker, else ``None``.

    A warming-up or build-failed marker renders as perfectly good prose, so
    without this it reaches the judge as an answer that covers no gold facts -
    a real zero indistinguishable from an empty dataset.
    """
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if entry.get("source") == "system" and isinstance(status, str) and status:
            return status
        text = entry.get("text")
        if isinstance(text, str) and text.startswith(
            ("Memory is still warming up", "Memory build failed")
        ):
            return str(status or "memory_warming_up")
    return None


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
    top_k: Optional[int] = None
    # The per-cell session the two calls were made on, recorded so a run can be
    # audited for cross-question contamination after the fact.
    session_id: Optional[str] = None
    error: Optional[str] = None
    error_class: Optional[str] = None

    def __post_init__(self) -> None:
        # Derived, never hand-set: a run written by an older version of this
        # harness carries an error string but no class, and --resume and the
        # report histogram both need one.
        if self.error and not self.error_class:
            self.error_class = classify_error_message(self.error)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AnswerRow":
        known = {key: payload.get(key) for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def row_key(row: Any) -> tuple[str, str, str]:
    """Identity of one evaluation cell, shared by answer rows and verdict rows."""
    return (row.dataset, row.search_type, row.question_id)


class AuthenticatedSession:
    """One logged-in conversation with the server, with the live-run failures handled.

    Three things the first live run needed and did not have:

    * **Re-authentication.** A 401 means the token expired, not that the request
      is bad. The session logs in again and replays the request once with the
      fresh token; only a second 401 is a real failure.
    * **Real backoff.** Transient failures (timeouts, 5xx from the server's
      connection pool) get up to ``attempts`` tries with exponential, jittered
      sleeps, so a server that is briefly saturated is given time to recover
      rather than hit again immediately.
    * **Pacing.** ``pause_seconds`` sleeps between consecutive requests, to keep
      a shared development server usable while a long run is going.

    ``sleep`` and ``rng`` are injected so tests never actually wait. The token is
    held in memory and never printed, logged, or written to a run directory.
    """

    def __init__(
        self,
        client: HttpClient,
        username: str,
        password: str,
        attempts: int = DEFAULT_RETRY_ATTEMPTS,
        backoff_seconds: Sequence[float] = DEFAULT_BACKOFF_SECONDS,
        jitter: float = DEFAULT_JITTER,
        pause_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ):
        self.client = client
        self.username = username
        self._password = password
        self.attempts = max(1, attempts)
        self.backoff_seconds = tuple(backoff_seconds)
        self.jitter = jitter
        self.pause_seconds = pause_seconds
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self.token: Optional[str] = None
        self.login_count = 0
        self._requests_made = 0

    def authenticate(self) -> None:
        """(Re)login. Counted so a run can report how often the token expired."""
        self.token = login(self.client, self.username, self._password)
        self.login_count += 1

    def _backoff(self, attempt: int) -> float:
        base = self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
        return base * (1.0 + self.jitter * self._rng.random())

    def post(self, path: str, body: dict) -> Any:
        """POST ``body`` to ``path``, handling token expiry and transient failure."""
        if self.token is None:
            self.authenticate()
        if self.pause_seconds > 0 and self._requests_made:
            self._sleep(self.pause_seconds)
        self._requests_made += 1

        reauthenticated = False
        attempt = 0
        while True:
            try:
                return self.client.post(path, json=body, headers=auth_headers(self.token))
            except AuthError:
                # The token expired mid-run. One fresh login, one replay; a second
                # 401 with a brand new token is a real authorization failure.
                if reauthenticated:
                    raise
                reauthenticated = True
                self.authenticate()
            except Exception as error:  # noqa: BLE001 - re-raised once the budget is gone
                if not is_transient(error) or attempt >= self.attempts - 1:
                    raise
                self._sleep(self._backoff(attempt))
                attempt += 1


DEFAULT_SESSION_PREFIX = "eval"


def pending_matrix(
    questions: Sequence[Question],
    datasets: Sequence[str],
    search_types: Sequence[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[AnswerRow]:
    """Every cell of the matrix, marked not attempted.

    Written to ``answers.jsonl`` before the first request so a run killed in its
    first pass is still resumable: the matrix is on disk from the start and the
    journal is the only thing that says which cells are done.
    """
    rows: list[AnswerRow] = []
    for dataset in datasets:
        for search_type in search_types:
            for question in questions:
                rows.append(
                    AnswerRow(
                        dataset=dataset,
                        search_type=search_type,
                        question_id=question.id,
                        category=question.category,
                        question=question.question,
                        corpus=question.corpus,
                        top_k=top_k,
                        error=PENDING_ERROR,
                    )
                )
    return rows


def _answer_one_cell(
    session: AuthenticatedSession,
    question: Question,
    dataset: str,
    search_type: str,
    top_k: int,
    session_prefix: str,
) -> AnswerRow:
    """Retrieve the context, then the answer, for one cell.

    **Context first, deliberately.** Both endpoints record a QA turn on the
    session, so asking for the answer first means the ``only_context`` call that
    follows retrieves against a session the answer just wrote to - the judge
    would grade the answer against a context the answer itself shaped. The
    ordering costs nothing and removes the feedback loop.
    """
    row = AnswerRow(
        dataset=dataset,
        search_type=search_type,
        question_id=question.id,
        category=question.category,
        question=question.question,
        corpus=question.corpus,
        top_k=top_k,
        session_id=cell_session_id(session_prefix, dataset, search_type, question.id),
    )
    started = time.perf_counter()
    try:
        context_path, context_body = build_search_request(
            question, dataset, search_type, top_k, only_context=True, session_id=row.session_id
        )
        row.context = extract_text(session.post(context_path, context_body))
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        row.error = f"context: {type(error).__name__}: {error}"
    else:
        try:
            answer_path, answer_body = build_search_request(
                question, dataset, search_type, top_k, only_context=False, session_id=row.session_id
            )
            payload = session.post(answer_path, answer_body)
            marker = detect_marker(payload)
            row.answer = extract_text(payload)
            if marker:
                # Not an answer: the server is telling us it has nothing to
                # answer from. Grading this as prose would report a real zero.
                row.error = f"answer: empty (server marker {marker})"
                row.answer = ""
            elif not row.answer.strip():
                row.error = "answer: empty (the server returned no text)"
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            row.error = f"answer: {type(error).__name__}: {error}"
    row.error_class = classify_error_message(row.error)
    row.elapsed_seconds = round(time.perf_counter() - started, 3)
    return row


def run_answers(
    session: AuthenticatedSession,
    questions: Sequence[Question],
    datasets: Sequence[str],
    search_types: Sequence[str],
    top_k: int = DEFAULT_TOP_K,
    session_prefix: str = DEFAULT_SESSION_PREFIX,
    on_row: Optional[Callable[[AnswerRow], None]] = None,
    concurrency: int = 1,
) -> list[AnswerRow]:
    """Answer every ``dataset x search_type x question`` cell.

    Two calls per cell - the retrieval context, then the answer - each on a
    session of its own. A failure is recorded on the row and the run continues:
    one dead search type should not cost the whole matrix.

    ``concurrency`` bounds how many cells are in flight at once. Cells are
    independent (each has its own session), so answering four at a time changes
    nothing but the wall clock - a 342-cell repeat run takes 15 minutes instead of
    an hour. Keep it at 1 against a server other people are using. ``on_row``
    fires as each cell lands (journal order); the returned list keeps matrix order.
    """
    cells = [
        (dataset, search_type, question)
        for dataset in datasets
        for search_type in search_types
        for question in questions
    ]

    def answer(cell) -> AnswerRow:
        dataset, search_type, question = cell
        return _answer_one_cell(session, question, dataset, search_type, top_k, session_prefix)

    if concurrency <= 1:
        rows: list[AnswerRow] = []
        for cell in cells:
            row = answer(cell)
            rows.append(row)
            if on_row is not None:
                on_row(row)
        return rows

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[int, AnswerRow] = {}
    report_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = {pool.submit(answer, cell): index for index, cell in enumerate(cells)}
        for future in as_completed(futures):
            row = future.result()
            results[futures[future]] = row
            if on_row is not None:
                with report_lock:
                    on_row(row)
    return [results[index] for index in range(len(cells))]


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """One graded answer.

    Gold facts come back as **1-based indices** into the question's own list, not
    as text. Text let the judge set its own denominator: a grader that reported
    one found fact and silently dropped the other six scored 100%. With indices
    the denominator is ``len(question.gold_facts)`` and the judge cannot move it -
    an index it never classified is missed, not absent.
    """

    gold_facts_covered: list[int] = Field(default_factory=list)
    gold_facts_missed: list[int] = Field(default_factory=list)
    wrong_claims: list[str] = Field(default_factory=list)
    fabricated_claims: list[str] = Field(default_factory=list)
    stance_errors: list[str] = Field(default_factory=list)
    notes: str = ""


@dataclass
class GoldFactScore:
    """The scored gold-fact lists for one answer, with the judge's own mistakes."""

    covered: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)
    mismatch: list[str] = field(default_factory=list)
    coverage: Optional[float] = None


def score_gold_facts(verdict: JudgeVerdict, question: Question) -> GoldFactScore:
    """Score the judge's indices against the question's gold facts.

    Rules, all of them there to stop an optimistic score:

    * the denominator is always ``len(question.gold_facts)``;
    * an index the judge classified neither way counts as **missed**;
    * an index outside ``1..n`` is dropped and recorded;
    * an index on both lists is dropped from covered, counted missed, and
      recorded - the judge contradicted itself and we do not guess which way.
    """
    total = len(question.gold_facts)
    if total == 0:
        return GoldFactScore(mismatch=["question has no gold facts"], coverage=None)

    valid = set(range(1, total + 1))
    mismatch: list[str] = []

    def clean(indices: Sequence[int], which: str) -> list[int]:
        out: list[int] = []
        for raw in indices:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                mismatch.append(f"{which} gold fact {raw!r} is not an index")
                continue
            if index not in valid:
                mismatch.append(f"{which} gold fact index {index} is not in 1..{total}")
                continue
            if index not in out:
                out.append(index)
        return out

    covered_raw = clean(verdict.gold_facts_covered, "covered")
    missed_raw = clean(verdict.gold_facts_missed, "missed")

    both = [index for index in covered_raw if index in missed_raw]
    for index in both:
        mismatch.append(f"gold fact {index} was listed on both sides and is counted missed")

    covered = sorted(index for index in covered_raw if index not in missed_raw)
    missed = sorted(valid - set(covered))
    return GoldFactScore(
        covered=covered,
        missed=missed,
        mismatch=mismatch,
        coverage=len(covered) / total,
    )


class _LazyLLMGateway:
    """Stand-in for ``cognee``'s gateway that defers the import to first use.

    Importing cognee at module scope would make ``--help`` slow and the unit
    tests non-hermetic. Because this is a module-level object with the same
    method name, ``patch.object(lib.LLMGateway, "acreate_structured_output", ...)``
    works exactly as it would against the real gateway.
    """

    @staticmethod
    async def acreate_structured_output(text_input: str, system_prompt: str, response_model: Any):
        # The key has to be in the environment before cognee's cached LLMConfig is
        # built, and the gateway import is the first thing here that builds it.
        configure_llm_environment()
        from cognee.infrastructure.llm import LLMGateway as _gateway

        return await _gateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=response_model,
        )


LLMGateway = _LazyLLMGateway


async def _call_gateway(user_prompt: str, system_prompt: str, response_model: Any) -> Any:
    """One structured-output call, bounded by :data:`LLM_CALL_TIMEOUT_SECONDS`.

    Goes through the module-level ``LLMGateway`` so tests can still patch
    ``LLMGateway.acreate_structured_output``. ``asyncio.timeout`` (3.11+) would read
    better, but the harness supports 3.10.
    """
    return await asyncio.wait_for(
        LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=response_model,
        ),
        timeout=LLM_CALL_TIMEOUT_SECONDS,
    )


def _default_read_prompt(filename: str) -> str:
    from cognee.infrastructure.llm.prompts import read_query_prompt

    return read_query_prompt(filename) or ""


def _default_render_prompt(filename: str, context: dict) -> str:
    from cognee.infrastructure.llm.prompts import render_prompt

    return render_prompt(filename, context)


def normalize_claim(text: str) -> str:
    """Case-folded, whitespace-collapsed form used by the must-not-claim floor."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_NEGATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(cue) for cue in NEGATION_CUES) + r")\b"
)


def split_sentences(text: str) -> list[str]:
    """Rough sentence split, good enough to scope a negation cue.

    Deliberately simple: the floor only needs to know whether the clause that
    carries a span also carries a denial, and being wrong here makes the floor
    *quieter*, never louder.
    """
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def is_negated(sentence: str) -> bool:
    """Whether a sentence carries a denial cue anywhere in it."""
    return bool(_NEGATION_PATTERN.search(normalize_claim(sentence)))


def triggered_traps(traps: Sequence[Trap], answer: str) -> list[Trap]:
    """The traps whose span an answer asserts, ignoring the ones it denies.

    A span fires only inside a sentence with no negation cue, so "the memo does
    not state that conflicts were identified" is not read as claiming they were.
    Conservative on purpose: the LLM judge still sees every full trap sentence
    and remains the primary detector.
    """
    sentences = [sentence for sentence in split_sentences(answer) if not is_negated(sentence)]
    if not sentences:
        return []
    normalized = [normalize_claim(sentence) for sentence in sentences]
    fired: list[Trap] = []
    for trap in traps:
        spans = [normalize_claim(span) for span in trap.match]
        if not any(span and any(span in sentence for sentence in normalized) for span in spans):
            continue
        fired.append(trap)
    return fired


def apply_must_not_claim_floor(
    verdict: JudgeVerdict,
    must_not_claim: Sequence[Trap],
    answer: str,
) -> JudgeVerdict:
    """Add every trap the answer's own words trigger to ``fabricated_claims``.

    A conservative span floor, not a second grader: it fires only on a literal
    the corpus author wrote down, only in a sentence that does not deny it, and
    only for traps that carry spans. It catches the class of fabrication a judge
    is most likely to wave through - an invented amount, date or number - and
    stays silent about everything else.
    """
    if not normalize_claim(answer):
        return verdict
    already = {normalize_claim(claim) for claim in verdict.fabricated_claims}
    additions = [
        trap.claim
        for trap in triggered_traps(must_not_claim, answer)
        if normalize_claim(trap.claim) not in already
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
            # Numbered so the judge answers with indices and cannot set its own
            # denominator by returning a shorter list than it was given.
            "gold_facts": [
                {"index": index, "fact": gold.fact, "source": gold.source}
                for index, gold in enumerate(question.gold_facts, start=1)
            ],
            "must_not_claim": [trap.claim for trap in question.must_not_claim],
            "notes": question.notes,
            "answer": row.answer,
            "context": row.context,
        },
    )

    verdict = await _call_gateway(user_prompt, system_prompt, JudgeVerdict)
    return apply_must_not_claim_floor(verdict, question.must_not_claim, row.answer)


def coverage(verdict: JudgeVerdict, question: Question) -> Optional[float]:
    """Fraction of the *question's* gold facts the judge marked covered."""
    return score_gold_facts(verdict, question).coverage


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
    # The judge's own mistakes: an index it invented, or a fact it put on both
    # lists. Surfaced in the report so a grader quietly going wrong is visible.
    judge_mismatch: list[str] = field(default_factory=list)
    notes: str = ""
    error: Optional[str] = None
    error_class: Optional[str] = None
    # The same two lists as 1-based gold-fact indices. The texts above are what
    # a reader wants; the indices are what miss attribution needs, and a question
    # with two identically worded gold facts cannot be reverse-mapped from text.
    # Absent on rows written before this field existed (``from_dict`` tolerates it).
    gold_facts_covered_index: list[int] = field(default_factory=list)
    gold_facts_missed_index: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.error and not self.error_class:
            self.error_class = classify_error_message(self.error)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "VerdictRow":
        known = {key: payload.get(key) for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


def verdict_row(
    row: AnswerRow, verdict: JudgeVerdict, question: Optional[Question] = None
) -> VerdictRow:
    """Build the stored row, mapping the judge's indices back to fact texts.

    The wire format is indices, but a report and a spot-check need to read, so
    the row keeps the texts. ``question`` is optional only for the paths that
    have no question to score against (an unknown question id).
    """
    if question is None:
        score = GoldFactScore()
        covered_text: list[str] = []
        missed_text: list[str] = []
    else:
        score = score_gold_facts(verdict, question)
        facts = question.gold_facts
        covered_text = [facts[index - 1].fact for index in score.covered]
        missed_text = [facts[index - 1].fact for index in score.missed]

    return VerdictRow(
        dataset=row.dataset,
        search_type=row.search_type,
        question_id=row.question_id,
        category=row.category,
        question=row.question,
        answer=row.answer,
        coverage=score.coverage,
        gold_facts_covered=covered_text,
        gold_facts_missed=missed_text,
        wrong_claims=list(verdict.wrong_claims),
        fabricated_claims=list(verdict.fabricated_claims),
        stance_errors=list(verdict.stance_errors),
        judge_mismatch=list(score.mismatch),
        notes=verdict.notes,
        error=row.error,
        error_class=row.error_class,
        gold_facts_covered_index=list(score.covered),
        gold_facts_missed_index=list(score.missed),
    )


async def run_judge(
    rows: Sequence[AnswerRow],
    questions: Sequence[Question],
    read_prompt: Callable[[str], str] = _default_read_prompt,
    render_prompt: Callable[[str, dict], str] = _default_render_prompt,
    on_row: Optional[Callable[[VerdictRow], None]] = None,
    concurrency: int = 1,
) -> list[VerdictRow]:
    """Grade every answer row.

    An answer row that already failed is not sent to the judge - there is nothing
    to grade - but it still produces a verdict row so the ``n`` and ``errors``
    columns of the report agree with the matrix that was attempted.

    ``concurrency`` bounds how many judge calls are in flight at once. A verdict is
    a minute of LLM time on a long context, so a 200-row run judged one at a time
    is a three-hour pass; the calls are independent, so four at a time cuts that to
    under an hour without changing a single verdict. ``on_row`` fires as each
    verdict lands (journal order), while the returned list keeps the input order.
    """
    by_id = {question.id: question for question in questions}
    gate = asyncio.Semaphore(max(1, int(concurrency)))

    async def grade(row: AnswerRow) -> VerdictRow:
        question = by_id.get(row.question_id)
        if question is None:
            out = verdict_row(row, JudgeVerdict())
            out.error = (
                row.error or f"no question with id {row.question_id!r} in the question files"
            )
        elif row.error:
            # Nothing was answered, so nothing is covered - but coverage stays
            # None rather than 0 so an unanswered cell never drags the mean.
            out = verdict_row(
                row,
                JudgeVerdict(gold_facts_missed=list(range(1, len(question.gold_facts) + 1))),
                question,
            )
            out.coverage = None
        else:
            async with gate:
                try:
                    verdict = await judge_answer(
                        row, question, read_prompt=read_prompt, render_prompt=render_prompt
                    )
                    out = verdict_row(row, verdict, question)
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    out = verdict_row(row, JudgeVerdict(), question)
                    out.coverage = None
                    out.error = f"judge: {type(error).__name__}: {error}"
        if on_row is not None:
            on_row(out)
        return out

    return list(await asyncio.gather(*(grade(row) for row in rows)))


# --------------------------------------------------------------------------------------
# Aggregation and reporting
# --------------------------------------------------------------------------------------


@dataclass
class Aggregate:
    dataset: str
    search_type: str
    n: int = 0
    answered: int = 0
    mean_coverage: Optional[float] = None
    wrong_claims: int = 0
    fabricated_claims: int = 0
    stance_errors: int = 0
    judge_mismatches: int = 0
    errors: int = 0
    # Set only by ``aggregate_by_category``; the main table never reads it.
    category: str = ""


def aggregate(rows: Iterable[VerdictRow]) -> list[Aggregate]:
    """Roll verdict rows up per dataset x search type, preserving first-seen order."""
    return _aggregate_by(rows, lambda row: (row.dataset, row.search_type))


def aggregate_by_category(rows: Iterable[VerdictRow]) -> list[Aggregate]:
    """Roll verdict rows up per dataset x search type x question category.

    The main table says a search type moved five points; this says whether it
    was the disputes questions or the timeline questions that moved, which is
    what decides where the next change goes. Same columns as ``aggregate``.
    """
    return _aggregate_by(rows, lambda row: (row.dataset, row.search_type, row.category))


def _aggregate_by(
    rows: Iterable[VerdictRow], key_of: Callable[[VerdictRow], tuple]
) -> list[Aggregate]:
    buckets: dict[tuple, Aggregate] = {}
    coverages: dict[tuple, list[float]] = {}

    for row in rows:
        key = key_of(row)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = Aggregate(
                dataset=row.dataset,
                search_type=row.search_type,
                category=row.category if len(key) > 2 else "",
            )
            buckets[key] = bucket
            coverages[key] = []
        bucket.n += 1
        if row.error:
            bucket.errors += 1
        else:
            bucket.answered += 1
        if row.coverage is not None:
            coverages[key].append(row.coverage)
        bucket.wrong_claims += len(row.wrong_claims)
        bucket.fabricated_claims += len(row.fabricated_claims)
        bucket.stance_errors += len(row.stance_errors)
        bucket.judge_mismatches += len(row.judge_mismatch)

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
        else:
            bucket.answered += 1
    return list(buckets.values())


@dataclass
class QuestionLine:
    """One row of the per-question table: a single cell's grade, readable at a glance."""

    question_id: str
    category: str
    dataset: str
    search_type: str
    coverage: Optional[float] = None
    wrong_claims: int = 0
    fabricated_claims: int = 0
    stance_errors: int = 0
    error_class: str = ""


def per_question_lines(rows: Iterable[VerdictRow]) -> list[QuestionLine]:
    """Every graded cell as one line, in run order.

    The tables above are means; this is the raw grid, so a reviewer can see that
    the five-point gain on ``disputes`` is one question going from 0 to 100 and
    the other two not moving.
    """
    return [
        QuestionLine(
            question_id=row.question_id,
            category=row.category,
            dataset=row.dataset,
            search_type=row.search_type,
            coverage=row.coverage,
            wrong_claims=len(row.wrong_claims),
            fabricated_claims=len(row.fabricated_claims),
            stance_errors=len(row.stance_errors),
            error_class=(row.error_class or "") if row.error else "",
        )
        for row in rows
    ]


# --------------------------------------------------------------------------------------
# Miss attribution
# --------------------------------------------------------------------------------------


class AttributionVerdict(BaseModel):
    """Which of a cell's missed gold facts were in the retrieved context.

    Same index protocol as :class:`JudgeVerdict`, for the same reason: the grader
    answers with the numbers it was shown, so it cannot leave a fact out. The
    numbers are the facts' indices in the *question's* gold list, not a fresh
    1..k, so a row of the attribution file lines up with the verdict's indices.
    """

    facts_in_context: list[int] = Field(default_factory=list)
    facts_not_in_context: list[int] = Field(default_factory=list)
    notes: str = ""


@dataclass
class AttributionRow:
    """One missed gold fact of one cell, and whether the context carried it.

    ``in_context`` is the attribution grader's answer: ``True`` means the answer
    had the fact in front of it and left it out (a *generation* miss), ``False``
    means retrieval never surfaced it (a *retrieval* miss), ``None`` means the
    grader did not classify it or the call failed (see ``notes``/``error``).

    ``literals`` and ``literals_in_context`` are the deterministic cross-check:
    the amounts, dates and paragraph numbers in the fact, and whether every one of
    them occurs in the context. Recorded beside the grader's answer, never in
    place of it - a context can paraphrase a date - so a grader that drifts shows
    up as a disagreement count in the report rather than as a quietly wrong table.
    """

    dataset: str
    search_type: str
    question_id: str
    category: str
    fact_index: int
    fact: str
    in_context: Optional[bool] = None
    literals: list[str] = field(default_factory=list)
    literals_in_context: Optional[bool] = None
    notes: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "AttributionRow":
        known = {key: payload.get(key) for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)

    @property
    def literal_disagreement(self) -> bool:
        """The lexical check and the grader point different ways."""
        return (
            self.in_context is not None
            and self.literals_in_context is not None
            and self.in_context != self.literals_in_context
        )


def missed_fact_indices(verdict: VerdictRow, question: Question) -> tuple[list[int], list[str]]:
    """The 1-based indices of the gold facts a verdict marked missed, plus problems.

    New rows carry the indices. A row written before ``gold_facts_missed_index``
    existed carries only the texts, which are reverse-mapped against the question;
    that works unless the question has two gold facts with the same wording, in
    which case both indices are returned and the ambiguity is reported.
    """
    total = len(question.gold_facts)
    if verdict.gold_facts_missed_index:
        indices = sorted(
            {int(index) for index in verdict.gold_facts_missed_index if 1 <= int(index) <= total}
        )
        return indices, []

    problems: list[str] = []
    indices: list[int] = []
    for text in verdict.gold_facts_missed:
        wanted = normalize_claim(text)
        matches = [
            index
            for index, gold in enumerate(question.gold_facts, start=1)
            if normalize_claim(gold.fact) == wanted
        ]
        if not matches:
            problems.append(f"missed fact text not in question {question.id!r}: {text!r}")
            continue
        if len(matches) > 1:
            problems.append(
                f"question {question.id!r} has {len(matches)} gold facts worded {text!r}; "
                "attributing all of them"
            )
        indices.extend(index for index in matches if index not in indices)
    return sorted(indices), problems


_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
#: What counts as a literal worth checking for: the specifics a paraphrase keeps.
_LITERAL_PATTERNS = (
    # $1,850,000.00 / $9 million
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand))?", re.I),
    # June 18, 2026 / 18 June 2026 / Sept. 22, 2026
    re.compile(r"\b(?:" + _MONTHS + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b", re.I),
    re.compile(r"\b\d{1,2}\s+(?:" + _MONTHS + r")\.?,?\s+\d{4}\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # ¶ 17 / paragraph 17 / para. 17
    re.compile(r"(?:¶|\bparagraphs?|\bparas?\.?)\s*\d+", re.I),
    # Resolution No. 2026-118 / Ordinance 2026-21 / PAS-L-001884-26 / 2026-PB-07
    re.compile(r"\b(?:[A-Z]{1,4}-){0,2}\d{2,6}(?:-[A-Z]*-?\d{2,})+\b"),
)


def fact_literals(fact: str) -> list[str]:
    """The amounts, dates, paragraph numbers and docket-style numbers in a fact."""
    found: list[str] = []
    for pattern in _LITERAL_PATTERNS:
        for match in pattern.finditer(fact or ""):
            literal = match.group(0).strip()
            if literal and literal not in found:
                found.append(literal)
    return found


def _normalize_literal(text: str) -> str:
    # A literal survives a paraphrase in spirit, not in bytes: "$1,850,000.00" is
    # the same amount as "1,850,000", "¶ 17" the same anchor as "paragraph 17",
    # "Sept. 22, 2026" the same day as "September 22, 2026".
    lowered = normalize_claim(text)
    lowered = (
        lowered.replace("¶", "paragraph ")
        .replace("para.", "paragraph")
        .replace("paras", "paragraph")
    )
    lowered = re.sub(r"\bparagraphs\b", "paragraph", lowered)
    lowered = lowered.replace("$", "").replace(",", "")
    lowered = re.sub(r"\.0+\b", "", lowered)
    lowered = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", lowered)
    lowered = re.sub(r"\b(sept|sep)\b\.?", "september", lowered)
    for short, long in (
        ("jan", "january"),
        ("feb", "february"),
        ("mar", "march"),
        ("apr", "april"),
        ("jun", "june"),
        ("jul", "july"),
        ("aug", "august"),
        ("oct", "october"),
        ("nov", "november"),
        ("dec", "december"),
    ):
        lowered = re.sub(rf"\b{short}\b\.?", long, lowered)
    lowered = lowered.replace(".", "")
    return re.sub(r"\s+", " ", lowered).strip()


def literals_present(literals: Sequence[str], context: str) -> Optional[bool]:
    """Whether every literal occurs in the context. ``None`` when there is nothing to check."""
    if not literals:
        return None
    haystack = _normalize_literal(context)
    return all(_normalize_literal(literal) in haystack for literal in literals)


def _attribution_rows(
    answer: AnswerRow, verdict: VerdictRow, question: Question, indices: Sequence[int]
) -> list[AttributionRow]:
    rows: list[AttributionRow] = []
    for index in indices:
        fact = question.gold_facts[index - 1].fact
        literals = fact_literals(fact)
        rows.append(
            AttributionRow(
                dataset=answer.dataset,
                search_type=answer.search_type,
                question_id=answer.question_id,
                category=answer.category,
                fact_index=index,
                fact=fact,
                literals=literals,
                literals_in_context=literals_present(literals, answer.context),
            )
        )
    return rows


async def attribute_misses(
    answer: AnswerRow,
    verdict: VerdictRow,
    question: Question,
    read_prompt: Callable[[str], str] = _default_read_prompt,
    render_prompt: Callable[[str, dict], str] = _default_render_prompt,
) -> list[AttributionRow]:
    """Attribute one cell's missed gold facts to retrieval or to generation.

    One LLM call for the whole cell, and none at all when the cell retrieved no
    context - every miss is then a retrieval miss by definition. A fact the grader
    classified neither way, or both ways, is left ``in_context=None`` with a note,
    never guessed.
    """
    indices, problems = missed_fact_indices(verdict, question)
    rows = _attribution_rows(answer, verdict, question, indices)
    for row in rows:
        row.notes = "; ".join(problems)
    if not rows:
        return rows

    if not normalize_claim(answer.context):
        for row in rows:
            row.in_context = False
            row.notes = "; ".join(filter(None, [row.notes, "no context was retrieved"]))
        return rows

    system_prompt = read_prompt(ATTRIBUTION_SYSTEM_PROMPT_FILE)
    if not system_prompt.strip():
        raise RuntimeError(
            f"Attribution system prompt is missing or empty: {ATTRIBUTION_SYSTEM_PROMPT_FILE}"
        )
    user_prompt = render_prompt(
        ATTRIBUTION_USER_PROMPT_FILE,
        {
            "question": question.question,
            "missed_facts": [{"index": row.fact_index, "fact": row.fact} for row in rows],
            "context": answer.context,
        },
    )
    graded = await _call_gateway(user_prompt, system_prompt, AttributionVerdict)

    def clean(values: Sequence[Any]) -> set[int]:
        out: set[int] = set()
        for raw in values:
            try:
                out.add(int(raw))
            except (TypeError, ValueError):
                continue
        return out

    present, absent = clean(graded.facts_in_context), clean(graded.facts_not_in_context)
    for row in rows:
        if row.fact_index in present and row.fact_index in absent:
            row.notes = "; ".join(filter(None, [row.notes, "grader listed the fact on both sides"]))
        elif row.fact_index in present:
            row.in_context = True
        elif row.fact_index in absent:
            row.in_context = False
        else:
            row.notes = "; ".join(filter(None, [row.notes, "grader did not classify the fact"]))
    return rows


async def run_attribution(
    answers: Sequence[AnswerRow],
    verdicts: Sequence[VerdictRow],
    questions: Sequence[Question],
    read_prompt: Callable[[str], str] = _default_read_prompt,
    render_prompt: Callable[[str, dict], str] = _default_render_prompt,
    on_row: Optional[Callable[[AttributionRow], None]] = None,
    concurrency: int = 1,
) -> list[AttributionRow]:
    """Attribute every graded cell's misses. Cells without a grade are skipped.

    Joins verdicts to answers on the cell key to recover the retrieved context -
    the verdict row does not carry it. A failed grader call is recorded on each of
    the cell's rows rather than raised, like ``run_judge``.
    """
    by_id = {question.id: question for question in questions}
    answers_by_key = {row_key(row): row for row in answers}
    gate = asyncio.Semaphore(max(1, int(concurrency)))

    async def attribute(verdict: VerdictRow) -> list[AttributionRow]:
        question = by_id.get(verdict.question_id)
        answer = answers_by_key.get(row_key(verdict))
        if question is None or answer is None or verdict.error or verdict.coverage is None:
            return []
        async with gate:
            try:
                rows = await attribute_misses(
                    answer, verdict, question, read_prompt=read_prompt, render_prompt=render_prompt
                )
            except Exception as error:  # noqa: BLE001 - recorded, not raised
                indices, _problems = missed_fact_indices(verdict, question)
                rows = _attribution_rows(answer, verdict, question, indices)
                for row in rows:
                    row.error = f"attribution: {type(error).__name__}: {error}"
        if on_row is not None:
            for row in rows:
                on_row(row)
        return rows

    nested = await asyncio.gather(*(attribute(verdict) for verdict in verdicts))
    return [row for rows in nested for row in rows]


@dataclass
class AttributionAggregate:
    dataset: str
    search_type: str
    category: str = ""
    missed: int = 0
    retrieval_misses: int = 0
    generation_misses: int = 0
    unclassified: int = 0
    literal_disagreements: int = 0
    errors: int = 0


def aggregate_attribution(rows: Iterable[AttributionRow]) -> list[AttributionAggregate]:
    """Roll attribution rows up per dataset x search type, then per category under each.

    The "all" row for a dataset x search type comes first (``category == ""``),
    then one row per category, so the table reads as a total with its breakdown.
    """
    totals: dict[tuple[str, str], AttributionAggregate] = {}
    by_category: dict[tuple[str, str, str], AttributionAggregate] = {}

    for row in rows:
        total_key = (row.dataset, row.search_type)
        total = totals.setdefault(
            total_key, AttributionAggregate(dataset=row.dataset, search_type=row.search_type)
        )
        category = by_category.setdefault(
            (*total_key, row.category),
            AttributionAggregate(
                dataset=row.dataset, search_type=row.search_type, category=row.category
            ),
        )
        for bucket in (total, category):
            bucket.missed += 1
            if row.error:
                bucket.errors += 1
            elif row.in_context is True:
                bucket.generation_misses += 1
            elif row.in_context is False:
                bucket.retrieval_misses += 1
            else:
                bucket.unclassified += 1
            if row.literal_disagreement:
                bucket.literal_disagreements += 1

    ordered: list[AttributionAggregate] = []
    for total_key, total in totals.items():
        ordered.append(total)
        ordered.extend(bucket for key, bucket in by_category.items() if key[:2] == total_key)
    return ordered


# --------------------------------------------------------------------------------------
# Repeats
# --------------------------------------------------------------------------------------


@dataclass
class RepeatAggregate:
    """The spread of one dataset x search type across the repetitions of a run."""

    dataset: str
    search_type: str
    repeats: int = 0
    coverages: list[float] = field(default_factory=list)
    mean_coverage: Optional[float] = None
    stdev_coverage: Optional[float] = None
    min_coverage: Optional[float] = None
    max_coverage: Optional[float] = None
    mean_wrong_claims: float = 0.0
    mean_fabricated_claims: float = 0.0
    mean_stance_errors: float = 0.0


def _sample_stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def aggregate_repeats(runs: Sequence[Sequence[VerdictRow]]) -> list[RepeatAggregate]:
    """Mean and spread of each dataset x search type's coverage across ``runs``.

    A single run of a 19-question set graded by an LLM is one draw. This is what
    says whether a five-point delta between two branches is a change or the
    noise floor. Each run contributes its own mean coverage (over the cells it
    graded); the spread is the sample standard deviation across runs. Claim
    counts are averaged per run so they stay comparable to a single run's table.
    """
    buckets: dict[tuple[str, str], RepeatAggregate] = {}
    counts: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for run in runs:
        for item in aggregate(run):
            key = (item.dataset, item.search_type)
            bucket = buckets.setdefault(
                key, RepeatAggregate(dataset=item.dataset, search_type=item.search_type)
            )
            bucket.repeats += 1
            if item.mean_coverage is not None:
                bucket.coverages.append(item.mean_coverage)
            counts.setdefault(key, []).append(
                (item.wrong_claims, item.fabricated_claims, item.stance_errors)
            )
    for key, bucket in buckets.items():
        if bucket.coverages:
            bucket.mean_coverage = sum(bucket.coverages) / len(bucket.coverages)
            bucket.stdev_coverage = _sample_stdev(bucket.coverages)
            bucket.min_coverage = min(bucket.coverages)
            bucket.max_coverage = max(bucket.coverages)
        tallies = counts[key]
        bucket.mean_wrong_claims = sum(t[0] for t in tallies) / len(tallies)
        bucket.mean_fabricated_claims = sum(t[1] for t in tallies) / len(tallies)
        bucket.mean_stance_errors = sum(t[2] for t in tallies) / len(tallies)
    return list(buckets.values())


def error_histogram(rows: Iterable[Any]) -> dict[str, int]:
    """Count failures by class over answer rows or verdict rows.

    The point of this in the report is triage: 84 rows lost to ``401`` is an
    expired token and a re-run, 84 lost to ``timeout`` is a server that needs
    more time or less load. The first live run could not tell you which without
    grepping the JSONL.
    """
    counts = {name: 0 for name in ERROR_CLASSES}
    for row in rows:
        error = getattr(row, "error", None)
        if not error:
            continue
        name = getattr(row, "error_class", None) or classify_error_message(error)
        counts[name] = counts.get(name, 0) + 1
    return {name: count for name, count in counts.items() if count}


def _format_coverage(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _format_float(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_report(
    aggregates: Sequence[Aggregate],
    title: str = "Recall evaluation",
    error_histogram: Optional[dict] = None,
    categories: Optional[Sequence[Aggregate]] = None,
    questions: Optional[Sequence[QuestionLine]] = None,
    attribution: Optional[Sequence[AttributionAggregate]] = None,
    repeats: Optional[Sequence[RepeatAggregate]] = None,
) -> str:
    """One markdown table, one row per dataset x search type, plus error triage.

    ``answered`` sits next to ``n`` deliberately: the first live run's report
    showed plausible-looking rows while every single cell had failed, because a
    run with no answers and a run with no gold facts both render coverage as
    ``-``. With ``answered`` the difference is the second column.

    The optional sections are appended after the main table, in this order:
    repeats, by category, miss attribution, per question, errors by class. Each
    is omitted when not given, so a report from an older run reads as before.
    """
    lines = [f"# {title}", ""]
    if not aggregates:
        lines.append("_No results._")
        return "\n".join(lines) + "\n"

    header = (
        "| dataset | search type | n | answered | mean coverage | wrong claims | "
        "fabricated claims | stance errors | judge issues | errors |"
    )
    lines.append(header)
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in aggregates:
        lines.append(
            f"| {item.dataset} | {item.search_type} | {item.n} | {item.answered} | "
            f"{_format_coverage(item.mean_coverage)} | {item.wrong_claims} | "
            f"{item.fabricated_claims} | {item.stance_errors} | "
            f"{item.judge_mismatches} | {item.errors} |"
        )
    lines.append("")
    lines.append(
        "Coverage is the mean over graded questions of covered / *all* the question's "
        "gold facts - the judge answers with indices, so it cannot shrink its own "
        "denominator, and a fact it never classified counts as missed. Claim counts "
        "are totals, not rates. `judge issues` counts indices the judge invented or "
        "put on both lists; anything above zero means read those rows by hand."
    )
    if repeats:
        lines.extend(_render_repeats(repeats))
    if categories:
        lines.extend(_render_categories(categories))
    if attribution:
        lines.extend(_render_attribution(attribution))
    if questions:
        lines.extend(_render_questions(questions))
    if error_histogram:
        lines.append("")
        lines.append("## Errors by class")
        lines.append("")
        lines.append("| class | rows |")
        lines.append("| --- | ---: |")
        for name in ERROR_CLASSES:
            if error_histogram.get(name):
                lines.append(f"| {name} | {error_histogram[name]} |")
        for name, count in error_histogram.items():
            if name not in ERROR_CLASSES and count:
                lines.append(f"| {name} | {count} |")
        lines.append("")
        lines.append(
            "`401` means the token expired - re-run those rows with --resume. "
            "`timeout` and `5xx` mean the server was slow or saturated: raise "
            "--timeout, or slow the harness down with --pause-seconds."
        )
    return "\n".join(lines) + "\n"


def _render_repeats(repeats: Sequence[RepeatAggregate]) -> list[str]:
    lines = [
        "",
        "## Repeats",
        "",
        "| dataset | search type | repeats | coverage mean | stdev | min | max | "
        "wrong / run | fabricated / run | stance / run |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in repeats:
        stdev = "-" if item.stdev_coverage is None else f"{item.stdev_coverage * 100:.1f} pts"
        lines.append(
            f"| {item.dataset} | {item.search_type} | {item.repeats} | "
            f"{_format_coverage(item.mean_coverage)} | {stdev} | "
            f"{_format_coverage(item.min_coverage)} | {_format_coverage(item.max_coverage)} | "
            f"{_format_float(item.mean_wrong_claims)} | "
            f"{_format_float(item.mean_fabricated_claims)} | "
            f"{_format_float(item.mean_stance_errors)} |"
        )
    lines.append("")
    lines.append(
        "Each repeat is a fresh run of the whole matrix on fresh sessions; `stdev` is the "
        "sample standard deviation of the per-repeat mean coverage, in percentage points. "
        "A delta between two branches smaller than about two stdevs is not a result."
    )
    return lines


def _render_categories(categories: Sequence[Aggregate]) -> list[str]:
    lines = [
        "",
        "## By category",
        "",
        "| dataset | search type | category | n | answered | mean coverage | wrong claims | "
        "fabricated claims | stance errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in categories:
        lines.append(
            f"| {item.dataset} | {item.search_type} | {item.category} | {item.n} | "
            f"{item.answered} | {_format_coverage(item.mean_coverage)} | {item.wrong_claims} | "
            f"{item.fabricated_claims} | {item.stance_errors} |"
        )
    return lines


def _render_attribution(attribution: Sequence[AttributionAggregate]) -> list[str]:
    lines = [
        "",
        "## Miss attribution",
        "",
        "| dataset | search type | category | missed | retrieval misses | generation misses | "
        "unclassified | literal disagreements | errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in attribution:
        lines.append(
            f"| {item.dataset} | {item.search_type} | {item.category or '(all)'} | "
            f"{item.missed} | {item.retrieval_misses} | {item.generation_misses} | "
            f"{item.unclassified} | {item.literal_disagreements} | {item.errors} |"
        )
    lines.append("")
    lines.append(
        "A *retrieval miss* is a missed gold fact the retrieved context did not contain - "
        "fix ingestion, lanes or budgets. A *generation miss* was in the context and the "
        "answer left it out - fix the prompt or the rendering. `literal disagreements` "
        "counts facts where the amounts, dates and paragraph numbers in the fact say one "
        "thing about the context and the attribution grader said the other; above a "
        "handful, read those rows in attribution.jsonl before trusting the split."
    )
    return lines


def _render_questions(questions: Sequence[QuestionLine]) -> list[str]:
    lines = [
        "",
        "## Per question",
        "",
        "| question | category | dataset | search type | coverage | wrong | fabricated | "
        "stance | error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in questions:
        lines.append(
            f"| {item.question_id} | {item.category} | {item.dataset} | {item.search_type} | "
            f"{_format_coverage(item.coverage)} | {item.wrong_claims} | "
            f"{item.fabricated_claims} | {item.stance_errors} | {item.error_class or ''} |"
        )
    return lines


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
                    f"  Judge issues: {row.judge_mismatch or '-'}",
                    f"  Judge notes: {row.notes or '-'}",
                    f"  Error: {row.error}" if row.error else "",
                ]
            ).rstrip()
        )
    return "\n".join(blocks)


# --------------------------------------------------------------------------------------
# Run directory I/O
# --------------------------------------------------------------------------------------


def rows_needing_answers(rows: Sequence[AnswerRow]) -> list[AnswerRow]:
    """The rows a ``--resume`` should ask the server again: exactly the failed ones."""
    return [row for row in rows if row.error]


def merge_answer_rows(existing: Sequence[AnswerRow], fresh: Sequence[AnswerRow]) -> list[AnswerRow]:
    """Replace each re-answered cell in place, keeping the original run's order.

    In place rather than appended so ``answers.jsonl`` after a resume is still one
    row per cell in the order the matrix was run - a resumed run and a clean run
    produce the same file shape, and nothing downstream has to know a resume
    happened.
    """
    replacements = {row_key(row): row for row in fresh}
    return [replacements.get(row_key(row), row) for row in existing]


def merge_verdict_rows(
    existing: Sequence[VerdictRow],
    fresh: Sequence[VerdictRow],
    order: Sequence[AnswerRow],
) -> list[VerdictRow]:
    """Overlay fresh verdicts on old ones, ordered to match ``order``.

    A cell with neither an old nor a new verdict is left out rather than faked:
    a placeholder would be counted in ``n`` and silently drag the coverage mean.
    """
    by_key: dict[tuple[str, str, str], VerdictRow] = {row_key(row): row for row in existing}
    by_key.update({row_key(row): row for row in fresh})
    merged: list[VerdictRow] = []
    for answer in order:
        found = by_key.get(row_key(answer))
        if found is not None:
            merged.append(found)
    return merged


def rows_needing_verdicts(
    answers: Sequence[AnswerRow],
    existing: Sequence[VerdictRow],
    reanswered: Sequence[AnswerRow] = (),
) -> list[AnswerRow]:
    """Which answer rows the judge still has to look at on a resume.

    Two kinds: the cells just re-answered (their old verdict graded an error) and
    the cells that were never graded at all - a run finished with ``--no-judge``,
    or interrupted. A successful row that already has a verdict is never sent to
    the judge again, so a resume costs only the calls it has to make.
    """
    # A verdict that carries an error graded nothing - treat it as a hole, not a
    # grade, or a resume could never repair a run whose judging failed wholesale.
    judged = {row_key(row) for row in existing if not row.error}
    present = {row_key(row) for row in existing}
    redo = {row_key(row) for row in reanswered}

    pending: list[AnswerRow] = []
    for row in answers:
        if row.error:
            # Still no answer, so nothing to grade. Passed through only when the
            # cell has no verdict row at all, so the report still counts it;
            # ``run_judge`` makes no LLM call for a failed row.
            if row_key(row) not in present:
                pending.append(row)
        elif row_key(row) in redo or row_key(row) not in judged:
            pending.append(row)
    return pending


def backup_file(path: str | Path, suffix: str = ".bak") -> Optional[Path]:
    """Copy ``path`` aside before it is overwritten. Returns the backup, if any."""
    path = Path(path)
    if not path.exists():
        return None
    backup = path.with_name(path.name + suffix)
    backup.write_bytes(path.read_bytes())
    return backup


def default_run_directory(root: str | Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / stamp


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    """Write every row, atomically.

    Via a temp file in the same directory and ``os.replace``, so a crash or a
    full disk mid-write leaves the previous file intact instead of a truncated
    one. These files are the only record of a run that cost hours of searches
    and judge calls; a half-written ``answers.jsonl`` is worse than an old one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = row.to_dict() if hasattr(row, "to_dict") else row
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def append_jsonl(path: str | Path, row: Any) -> Path:
    """Append one row and flush - the crash journal a ``--resume`` picks up.

    ``write_jsonl`` runs once at the end of a run; a baseline that died at row 63
    of 94 lost every answer it had paid for. One line per row, flushed, so the
    most a crash can cost is the row in flight.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = row.to_dict() if hasattr(row, "to_dict") else row
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
    return path


def recover_partial_answers(directory: str | Path, rows: Sequence[AnswerRow]) -> list[AnswerRow]:
    """Overlay the answers journal of an interrupted run onto ``rows``.

    Journal rows replace their cells in place (``merge_answer_rows``), so a cell
    the dead run had already re-answered is not asked again. No journal → ``rows``
    unchanged.
    """
    path = Path(directory) / PARTIAL_ANSWERS_FILENAME
    if not path.exists():
        return list(rows)
    recovered = [AnswerRow.from_dict(row) for row in read_jsonl(path)]
    return merge_answer_rows(rows, recovered)


def recover_partial_verdicts(
    directory: str | Path, existing: Sequence[VerdictRow], order: Sequence[AnswerRow]
) -> list[VerdictRow]:
    """Overlay the verdicts journal of an interrupted run onto ``existing``."""
    path = Path(directory) / PARTIAL_VERDICTS_FILENAME
    if not path.exists():
        return list(existing)
    recovered = [VerdictRow.from_dict(row) for row in read_jsonl(path)]
    return merge_verdict_rows(existing, recovered, order)


def clear_partial_files(directory: str | Path) -> None:
    """Drop both journals once their rows are folded into the main files."""
    for name in (PARTIAL_ANSWERS_FILENAME, PARTIAL_VERDICTS_FILENAME):
        path = Path(directory) / name
        if path.exists():
            path.unlink()


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, tolerating one torn line at the end.

    A crash journal is appended line by line, so the process can die halfway
    through writing the last one. That single truncated line is expected and is
    skipped with a warning. A torn line anywhere *else* is corruption we must not
    paper over: silently dropping a middle row would quietly shrink the matrix
    and move the numbers.
    """
    path = Path(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    populated = [index for index, line in enumerate(raw_lines) if line.strip()]
    last = populated[-1] if populated else None

    rows: list[dict] = []
    for index in populated:
        line = raw_lines[index].strip()
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index == last:
                print(
                    f"WARNING {path}: skipping a torn trailing line "
                    f"(line {index + 1}); the run that wrote it was interrupted.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            raise ValueError(
                f"{path}: line {index + 1} is not valid JSON and is not the last line, "
                f"so the file is corrupt rather than merely interrupted ({error})"
            ) from error
    return rows


def _git_head(directory: str | Path) -> Optional[str]:
    """The commit of the checkout the harness runs from, or ``None``.

    Never fatal: a run in a tarball or a dirty worktree is still a run, and the
    manifest says what it could find.
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def file_digest(path: str | Path) -> str:
    """sha256 of a question file, so a report can be tied to the gold it graded."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(
    directory: str | Path,
    base_url: str,
    question_paths: Sequence[str | Path],
    datasets: Sequence[str],
    search_types: Sequence[str],
    top_k: int,
    timeout: float,
    pause_seconds: float,
    label: Optional[str] = None,
    repeats: Optional[int] = None,
    repeat_index: Optional[int] = None,
) -> Path:
    """Record what this run was, before it runs.

    A coverage number is meaningless without the gold it was scored against, the
    server it asked and the protocol it used, and none of that is recoverable
    from ``answers.jsonl`` a week later. Credentials are not inputs and never
    appear here.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": base_url,
        # The harness's own checkout - NOT the server's code, which this process
        # cannot see. ``label`` is where the server's version goes.
        "harness_checkout_commit": _git_head(Path(__file__).resolve().parent),
        "label": label,
        "question_files": [
            {"path": str(item), "sha256": file_digest(item)} for item in question_paths
        ],
        "datasets": list(datasets),
        "search_types": list(search_types),
        "top_k": top_k,
        "timeout_seconds": timeout,
        "pause_seconds": pause_seconds,
        # Protocol flags: what the numbers in this directory mean.
        "session_per_cell": True,
        "context_before_answer": True,
    }
    if repeats is not None:
        # The parent of a --repeats run says how many; each child says which.
        manifest["repeats"] = repeats
    if repeat_index is not None:
        manifest["repeat_index"] = repeat_index
    path = directory / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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
