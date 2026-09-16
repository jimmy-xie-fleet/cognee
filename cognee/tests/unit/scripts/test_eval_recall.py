"""Unit tests for the recall evaluation harness in ``scripts/legal/eval``.

``scripts/`` is not a package, so both modules are loaded by path - the same
trick the migration tests in ``cognee/tests/unit`` use. Nothing here reaches the
network, an LLM, or ``~/.cognee``: the HTTP client is a fake implementing the
harness's ``post`` protocol, and the judge's gateway is patched on the module
object the library exposes for exactly that purpose.
"""

import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIRECTORY = REPOSITORY_ROOT / "scripts" / "legal" / "eval"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lib = _load_module("recall_eval_lib_under_test", EVAL_DIRECTORY / "recall_eval_lib.py")
cli = _load_module("eval_recall_under_test", EVAL_DIRECTORY / "eval_recall.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def question_document(**overrides):
    question = {
        "id": "adams-01",
        "category": "disputes",
        "question": "Which allegations do the defendants deny?",
        "gold_facts": [
            {"fact": "The defendants deny paragraph 12.", "source": "answer.pdf p.3"},
            {"fact": "The defendants admit paragraph 4.", "source": "answer.pdf p.2"},
        ],
        "must_not_claim": ["the court has already ruled"],
        "notes": "",
    }
    question.update(overrides)
    return {"corpus": "adams", "questions": [question]}


def write_question_file(tmp_path: Path, document) -> Path:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def make_question(question_id="adams-01", must_not_claim=()):
    return lib.Question(
        id=question_id,
        category="disputes",
        question="Which allegations do the defendants deny?",
        gold_facts=(
            lib.GoldFact(fact="The defendants deny paragraph 12.", source="answer.pdf p.3"),
        ),
        must_not_claim=tuple(must_not_claim),
        corpus="adams",
    )


def make_session(client, **overrides):
    """A session over a fake client with the clock stubbed out."""
    slept = overrides.pop("slept", None)
    if slept is None:
        slept = []
    overrides.setdefault("sleep", slept.append)
    overrides.setdefault("jitter", 0.0)
    session = lib.AuthenticatedSession(client, "user@example.com", "hunter2", **overrides)
    session.slept = slept
    session.authenticate()
    return session


class FakeClient:
    """Records every call and replays canned responses (or raises)."""

    def __init__(self, responder=None):
        self.calls = []
        self._responder = responder or (lambda path, body: [{"search_result": ["ok"]}])

    @property
    def login_calls(self):
        return [call for call in self.calls if call["path"] == lib.LOGIN_PATH]

    @property
    def search_calls(self):
        return [call for call in self.calls if call["path"] != lib.LOGIN_PATH]

    def post(self, path, *, json=None, data=None, headers=None):
        self.calls.append({"path": path, "json": json, "data": data, "headers": headers})
        if path == lib.LOGIN_PATH:
            return {"access_token": f"token-{len(self.login_calls) - 1}", "token_type": "bearer"}
        result = self._responder(path, json if json is not None else data)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        pass


# ---------------------------------------------------------------------------
# question file validation
# ---------------------------------------------------------------------------


def test_validate_question_file_accepts_a_well_formed_file(tmp_path):
    path = write_question_file(tmp_path, question_document())

    questions = lib.validate_question_file(path)

    assert len(questions) == 1
    assert questions[0].id == "adams-01"
    assert questions[0].corpus == "adams"
    assert questions[0].category == "disputes"
    assert [fact.fact for fact in questions[0].gold_facts] == [
        "The defendants deny paragraph 12.",
        "The defendants admit paragraph 4.",
    ]
    assert questions[0].must_not_claim == ("the court has already ruled",)


def test_validate_question_file_rejects_empty_gold_facts(tmp_path):
    path = write_question_file(tmp_path, question_document(gold_facts=[]))

    with pytest.raises(lib.QuestionFileError) as error:
        lib.validate_question_file(path)

    assert "gold_facts is empty" in str(error.value)


def test_validate_question_file_rejects_unknown_category(tmp_path):
    path = write_question_file(tmp_path, question_document(category="vibes"))

    with pytest.raises(lib.QuestionFileError) as error:
        lib.validate_question_file(path)

    assert "unknown category 'vibes'" in str(error.value)


def test_validate_question_file_reports_every_missing_key_at_once(tmp_path):
    document = question_document()
    del document["questions"][0]["id"]
    del document["questions"][0]["question"]
    del document["questions"][0]["gold_facts"]
    path = write_question_file(tmp_path, document)

    with pytest.raises(lib.QuestionFileError) as error:
        lib.validate_question_file(path)

    message = str(error.value)
    assert "missing a non-empty 'id'" in message
    assert "missing a non-empty 'question'" in message
    assert "missing 'gold_facts'" in message
    assert "3 problem(s)" in message


def test_validate_question_file_rejects_invalid_json(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(lib.QuestionFileError) as error:
        lib.validate_question_file(path)

    assert "invalid JSON" in str(error.value)


# ---------------------------------------------------------------------------
# request building / answering
# ---------------------------------------------------------------------------


def test_auto_search_type_posts_to_recall_with_null_search_type():
    client = FakeClient()

    rows = lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=[lib.AUTO_SEARCH_TYPE],
    )

    assert [call["path"] for call in client.search_calls] == ["/api/v1/recall", "/api/v1/recall"]
    first_body = client.search_calls[0]["json"]
    assert "search_type" in first_body and first_body["search_type"] is None
    assert first_body["datasets"] == ["adams"]
    assert rows[0].error is None


def test_pinned_search_type_posts_to_search():
    client = FakeClient()

    lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=["GRAPH_COMPLETION"],
    )

    assert {call["path"] for call in client.search_calls} == {"/api/v1/search"}
    assert client.search_calls[0]["json"]["search_type"] == "GRAPH_COMPLETION"


def test_each_question_makes_an_answer_call_and_a_context_call():
    def responder(path, body):
        return [{"search_result": ["context text" if body.get("only_context") else "answer text"]}]

    client = FakeClient(responder)

    session = make_session(client)
    rows = lib.run_answers(
        session,
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
        top_k=7,
    )

    assert len(client.search_calls) == 2
    answer_call, context_call = client.search_calls
    assert "only_context" not in answer_call["json"]
    assert answer_call["json"]["top_k"] == 7
    assert context_call["json"]["only_context"] is True
    assert context_call["json"]["context_format"] == "context"
    assert answer_call["headers"] == {"Authorization": f"Bearer {session.token}"}
    assert rows[0].answer == "answer text"
    assert rows[0].context == "context text"


def test_recall_shaped_response_is_flattened_too():
    payload = [{"source": "graph", "kind": "graph_completion", "text": "the denial is in para 12"}]

    assert lib.extract_text(payload) == "the denial is in para 12"


def test_transport_error_becomes_an_error_row_rather_than_an_exception():
    client = FakeClient(lambda path, body: lib.TransientHttpError("connection refused"))

    rows = lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert len(rows) == 1
    assert rows[0].error is not None
    assert "connection refused" in rows[0].error
    assert rows[0].answer == ""
    # the default policy is three attempts in total
    assert len(client.search_calls) == 3


def test_a_transient_failure_is_retried_once_and_then_succeeds():
    attempts = {"count": 0}

    def responder(path, body):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return lib.TransientHttpError("read timeout")
        return [{"search_result": ["fine"]}]

    client = FakeClient(responder)

    rows = lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert rows[0].error is None
    assert rows[0].answer == "fine"


def test_a_permanent_failure_is_not_retried():
    client = FakeClient(lambda path, body: lib.HttpError("HTTP 403: forbidden"))

    rows = lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert "403" in rows[0].error
    assert len(client.search_calls) == 1


def test_login_returns_the_token_from_the_form_post():
    client = FakeClient()

    token = lib.login(client, "user@example.com", "hunter2")

    assert token == "token-0"
    assert client.calls[0]["path"] == "/api/v1/auth/login"
    assert client.calls[0]["data"] == {"username": "user@example.com", "password": "hunter2"}
    assert client.calls[0]["json"] is None


# ---------------------------------------------------------------------------
# judging
# ---------------------------------------------------------------------------


def test_coverage_is_covered_over_covered_plus_missed():
    verdict = lib.JudgeVerdict(gold_facts_covered=["a", "b", "c"], gold_facts_missed=["d"])

    assert lib.coverage(verdict) == pytest.approx(0.75)


def test_coverage_is_none_when_the_judge_listed_no_gold_facts():
    assert lib.coverage(lib.JudgeVerdict()) is None


def test_must_not_claim_floor_adds_a_trap_the_judge_missed():
    verdict = lib.JudgeVerdict(gold_facts_covered=["x"])

    tightened = lib.apply_must_not_claim_floor(
        verdict,
        ["The court has already   ruled"],
        "In fact the COURT HAS ALREADY RULED on the motion.",
    )

    assert tightened.fabricated_claims == ["The court has already   ruled"]


def test_must_not_claim_floor_does_not_duplicate_what_the_judge_already_flagged():
    verdict = lib.JudgeVerdict(fabricated_claims=["the court has already ruled"])

    tightened = lib.apply_must_not_claim_floor(
        verdict, ["The Court Has Already Ruled"], "the court has already ruled"
    )

    assert tightened.fabricated_claims == ["the court has already ruled"]


def test_must_not_claim_floor_leaves_a_clean_answer_alone():
    verdict = lib.JudgeVerdict(gold_facts_covered=["x"])

    tightened = lib.apply_must_not_claim_floor(
        verdict, ["the court has already ruled"], "The defendants deny paragraph 12."
    )

    assert tightened.fabricated_claims == []


def test_judge_answer_calls_the_gateway_and_applies_the_floor():
    question = make_question(must_not_claim=["the court has already ruled"])
    row = lib.AnswerRow(
        dataset="adams",
        search_type="HYBRID_COMPLETION",
        question_id=question.id,
        category=question.category,
        question=question.question,
        answer="They deny paragraph 12, and the court has already ruled.",
        context="Defendants deny the allegations of paragraph 12.",
    )
    seen = {}

    async def fake_structured_output(text_input, system_prompt, response_model):
        seen["text_input"] = text_input
        seen["system_prompt"] = system_prompt
        seen["response_model"] = response_model
        return lib.JudgeVerdict(gold_facts_covered=["The defendants deny paragraph 12."])

    with patch.object(lib.LLMGateway, "acreate_structured_output", fake_structured_output):
        verdict = asyncio.run(
            lib.judge_answer(
                row,
                question,
                read_prompt=lambda name: f"SYSTEM({name})",
                render_prompt=lambda name, context: json.dumps(
                    {"name": name, "question": context["question"], "answer": context["answer"]}
                ),
            )
        )

    assert seen["response_model"] is lib.JudgeVerdict
    assert seen["system_prompt"] == "SYSTEM(eval_judge_system.txt)"
    assert question.question in seen["text_input"]
    assert lib.coverage(verdict) == pytest.approx(1.0)
    assert verdict.fabricated_claims == ["the court has already ruled"]


def test_run_judge_skips_failed_answer_rows_but_still_reports_them():
    question = make_question()
    failed = lib.AnswerRow(
        dataset="adams",
        search_type="HYBRID_COMPLETION",
        question_id=question.id,
        category=question.category,
        question=question.question,
        error="answer: TransientHttpError: boom",
    )

    async def explode(**kwargs):
        raise AssertionError("the judge must not be called for a failed answer row")

    with patch.object(lib.LLMGateway, "acreate_structured_output", explode):
        rows = asyncio.run(lib.run_judge([failed], [question]))

    assert len(rows) == 1
    assert rows[0].error == "answer: TransientHttpError: boom"
    assert rows[0].coverage is None


def test_run_judge_records_a_judge_failure_as_an_error_row():
    question = make_question()
    row = lib.AnswerRow(
        dataset="adams",
        search_type="HYBRID_COMPLETION",
        question_id=question.id,
        category=question.category,
        question=question.question,
        answer="something",
    )

    async def explode(text_input, system_prompt, response_model):
        raise RuntimeError("rate limited")

    with patch.object(lib.LLMGateway, "acreate_structured_output", explode):
        rows = asyncio.run(
            lib.run_judge(
                [row],
                [question],
                read_prompt=lambda name: "system",
                render_prompt=lambda name, context: "user",
            )
        )

    assert rows[0].error == "judge: RuntimeError: rate limited"
    assert rows[0].coverage is None


# ---------------------------------------------------------------------------
# aggregation, report, spot check
# ---------------------------------------------------------------------------


def verdict(dataset, search_type, question_id, covered, missed, **counts):
    return lib.VerdictRow(
        dataset=dataset,
        search_type=search_type,
        question_id=question_id,
        category="disputes",
        question="q",
        coverage=(len(covered) / (len(covered) + len(missed))) if (covered or missed) else None,
        gold_facts_covered=covered,
        gold_facts_missed=missed,
        wrong_claims=counts.get("wrong_claims", []),
        fabricated_claims=counts.get("fabricated_claims", []),
        stance_errors=counts.get("stance_errors", []),
        error=counts.get("error"),
    )


def test_aggregate_groups_by_dataset_and_search_type():
    rows = [
        verdict("adams", "HYBRID_COMPLETION", "q1", ["a", "b"], [], wrong_claims=["w"]),
        verdict("adams", "HYBRID_COMPLETION", "q2", ["a"], ["b"], stance_errors=["s"]),
        verdict("adams", "GRAPH_COMPLETION", "q1", [], ["a", "b"], fabricated_claims=["f", "g"]),
        verdict("plains", "HYBRID_COMPLETION", "q1", [], [], error="answer: boom"),
    ]

    aggregates = {(item.dataset, item.search_type): item for item in lib.aggregate(rows)}

    adams_hybrid = aggregates[("adams", "HYBRID_COMPLETION")]
    assert adams_hybrid.n == 2
    assert adams_hybrid.mean_coverage == pytest.approx(0.75)  # mean of 1.0 and 0.5
    assert adams_hybrid.wrong_claims == 1
    assert adams_hybrid.stance_errors == 1
    assert adams_hybrid.errors == 0

    adams_graph = aggregates[("adams", "GRAPH_COMPLETION")]
    assert adams_graph.mean_coverage == pytest.approx(0.0)
    assert adams_graph.fabricated_claims == 2

    plains = aggregates[("plains", "HYBRID_COMPLETION")]
    assert plains.n == 1
    assert plains.errors == 1
    assert plains.mean_coverage is None


def test_report_table_renders_every_bucket():
    aggregates = lib.aggregate(
        [
            verdict("adams", "HYBRID_COMPLETION", "q1", ["a", "b", "c"], ["d"]),
            verdict("adams", "AUTO", "q1", [], ["a"], error="answer: boom"),
        ]
    )

    report = lib.render_report(aggregates, title="Run 1")

    assert "# Run 1" in report
    assert "| dataset | search type | n | answered | mean coverage |" in report
    assert "| adams | HYBRID_COMPLETION | 1 | 1 | 75.0% | 0 | 0 | 0 | 0 |" in report
    assert "| adams | AUTO | 1 | 0 | 0.0% | 0 | 0 | 0 | 1 |" in report


def test_report_handles_no_results():
    assert "_No results._" in lib.render_report([])


def test_spot_check_sample_is_seeded_and_sized_by_the_fraction():
    rows = [verdict("adams", "HYBRID_COMPLETION", f"q{index}", ["a"], []) for index in range(11)]

    sample = lib.spot_check_sample(rows, 0.2, seed=7)
    again = lib.spot_check_sample(rows, 0.2, seed=7)

    assert len(sample) == math.ceil(0.2 * 11) == 3
    assert [row.question_id for row in sample] == [row.question_id for row in again]
    # the seed actually selects: some other seed picks a different window
    assert any(
        [row.question_id for row in lib.spot_check_sample(rows, 0.2, seed=seed)]
        != [row.question_id for row in sample]
        for seed in range(1, 20)
    )
    assert lib.spot_check_sample(rows, 0.0, seed=7) == []


def test_format_spot_check_shows_question_answer_gold_and_verdict():
    question = make_question()
    row = verdict("adams", "HYBRID_COMPLETION", question.id, ["a"], ["b"], wrong_claims=["nope"])
    row.question = question.question
    row.answer = "the answer text"

    rendered = lib.format_spot_check([row], [question])

    assert question.question in rendered
    assert "the answer text" in rendered
    assert "The defendants deny paragraph 12." in rendered
    assert "nope" in rendered
    assert "50.0%" in rendered


def test_answer_rows_round_trip_through_jsonl(tmp_path):
    rows = [
        lib.AnswerRow(
            dataset="adams",
            search_type="AUTO",
            question_id="q1",
            category="disputes",
            question="q",
            answer="a",
            context="c",
        )
    ]
    path = lib.write_jsonl(tmp_path / "answers.jsonl", rows)

    restored = [lib.AnswerRow.from_dict(item) for item in lib.read_jsonl(path)]

    assert restored == rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_validate_only_exits_zero_for_a_valid_file(tmp_path, capsys):
    path = write_question_file(tmp_path, question_document())

    assert cli.main(["--validate-only", "--questions", str(path)]) == 0
    assert "OK" in capsys.readouterr().out


def test_validate_only_exits_one_for_an_invalid_file(tmp_path, capsys):
    path = write_question_file(tmp_path, question_document(category="vibes"))

    assert cli.main(["--validate-only", "--questions", str(path)]) == 1
    assert "INVALID" in capsys.readouterr().out


def test_neither_module_imports_cognee_at_module_scope():
    """``--help`` must not pay for a cognee import, so nothing cognee is top level.

    Asserted on the source rather than on ``sys.modules`` because the unit-suite
    conftest imports cognee for every test in this tree; the runnable version of
    this check is the ``--help`` gate in scripts/legal/eval/README.md.
    """
    for path in (EVAL_DIRECTORY / "recall_eval_lib.py", EVAL_DIRECTORY / "eval_recall.py"):
        top_level = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(("import cognee", "from cognee"))
        ]
        assert top_level == [], f"{path.name} imports cognee at module scope: {top_level}"


# ---------------------------------------------------------------------------
# live-run hardening: auth expiry, backoff, pacing
# ---------------------------------------------------------------------------


def responder_sequence(*results):
    """Replay ``results`` in order for non-login calls, repeating the last one."""
    state = {"index": 0}

    def responder(path, body):
        index = min(state["index"], len(results) - 1)
        state["index"] += 1
        return results[index]

    return responder


def test_a_401_triggers_one_relogin_and_the_request_succeeds():
    client = FakeClient(
        responder_sequence(lib.AuthError("HTTP 401: Unauthorized"), [{"search_result": ["fine"]}])
    )
    session = make_session(client)

    result = session.post("/api/v1/search", {"query": "q"})

    assert lib.extract_text(result) == "fine"
    # the initial login plus exactly one re-login
    assert len(client.login_calls) == 2
    stale, refreshed = client.search_calls
    assert stale["headers"] != refreshed["headers"]
    assert refreshed["headers"] == {"Authorization": f"Bearer {session.token}"}


def test_a_second_401_after_relogin_becomes_an_error_row():
    client = FakeClient(lambda path, body: lib.AuthError('HTTP 401: {"detail":"Unauthorized"}'))

    rows = lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert rows[0].error is not None
    assert "401" in rows[0].error
    assert rows[0].error_class == "401"
    # one stale attempt plus one with the fresh token, and no further flailing
    assert len(client.search_calls) == 2
    assert len(client.login_calls) == 2


def test_a_401_does_not_consume_the_transient_retry_budget():
    client = FakeClient(
        responder_sequence(
            lib.AuthError("HTTP 401: Unauthorized"),
            lib.TransientHttpError("ReadTimeout: timed out"),
            lib.TransientHttpError("ReadTimeout: timed out"),
            [{"search_result": ["recovered"]}],
        )
    )
    session = make_session(client)

    assert lib.extract_text(session.post("/api/v1/search", {"query": "q"})) == "recovered"


def test_a_timeout_is_retried_three_times_then_recorded_with_its_class():
    client = FakeClient(lambda path, body: lib.TransientHttpError("ReadTimeout: timed out"))
    session = make_session(client)

    rows = lib.run_answers(
        session,
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert len(client.search_calls) == 3
    assert rows[0].error_class == "timeout"
    assert "timed out" in rows[0].error


def test_a_500_then_200_succeeds_on_the_second_attempt():
    client = FakeClient(
        responder_sequence(
            lib.TransientHttpError("HTTP 500: Internal Server Error"),
            [{"search_result": ["second time lucky"]}],
        )
    )

    rows = lib.run_answers(
        make_session(client),
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert rows[0].error is None
    assert rows[0].answer == "second time lucky"


def test_backoff_is_exponential_jittered_and_injected():
    client = FakeClient(lambda path, body: lib.TransientHttpError("HTTP 503: unavailable"))
    slept = []
    session = make_session(client, slept=slept, jitter=0.5, rng=__import__("random").Random(1))

    with pytest.raises(lib.TransientHttpError):
        session.post("/api/v1/search", {"query": "q"})

    # two sleeps for three attempts, growing, each within the jitter band
    assert len(slept) == 2
    assert slept[0] < slept[1]
    assert 2.0 <= slept[0] <= 3.0
    assert 8.0 <= slept[1] <= 12.0


def test_pause_seconds_sleeps_between_consecutive_requests():
    client = FakeClient()
    slept = []
    session = make_session(client, slept=slept, pause_seconds=1.5)
    slept.clear()  # ignore anything the initial authenticate did

    lib.run_answers(
        session,
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    # two requests, one pause between them
    assert slept == [1.5]


def test_error_classes_are_derived_from_the_message():
    assert (
        lib.classify_error_message('answer: HttpError: HTTP 401: {"detail":"Unauthorized"}')
        == "401"
    )
    assert (
        lib.classify_error_message("answer: TransientHttpError: ReadTimeout: timed out")
        == "timeout"
    )
    assert (
        lib.classify_error_message("answer: TransientHttpError: HTTP 500: Internal Server Error")
        == "5xx"
    )
    assert lib.classify_error_message("answer: HttpError: HTTP 403: forbidden") == "other"
    assert lib.classify_error_message("") == ""


def test_a_row_loaded_without_an_error_class_is_classified_on_the_way_in():
    row = lib.AnswerRow.from_dict(
        {
            "dataset": "adams",
            "search_type": "HYBRID_COMPLETION",
            "question_id": "q1",
            "category": "disputes",
            "question": "q",
            "error": "answer: TransientHttpError: ReadTimeout: timed out",
        }
    )

    assert row.error_class == "timeout"


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def answer_row(question_id, error=None, answer="kept"):
    return lib.AnswerRow(
        dataset="adams",
        search_type="HYBRID_COMPLETION",
        question_id=question_id,
        category="disputes",
        question="q",
        answer="" if error else answer,
        error=error,
    )


def test_resume_reruns_only_the_error_rows(tmp_path):
    existing = [
        answer_row("q1"),
        answer_row("q2", error="answer: TransientHttpError: ReadTimeout: timed out"),
        answer_row("q3"),
    ]
    lib.write_jsonl(tmp_path / lib.ANSWERS_FILENAME, existing)
    client = FakeClient(lambda path, body: [{"search_result": ["repaired"]}])

    reloaded = [
        lib.AnswerRow.from_dict(row) for row in lib.read_jsonl(tmp_path / lib.ANSWERS_FILENAME)
    ]
    failed = lib.rows_needing_answers(reloaded)
    assert [row.question_id for row in failed] == ["q2"]

    fresh = lib.run_answers(
        make_session(client),
        questions=[make_question("q2")],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )
    merged = lib.merge_answer_rows(reloaded, fresh)

    assert [row.question_id for row in merged] == ["q1", "q2", "q3"]
    assert merged[0].answer == "kept" and merged[2].answer == "kept"
    assert merged[1].answer == "repaired"
    assert merged[1].error is None
    # only the failed cell was asked again: two calls, not six
    assert len(client.search_calls) == 2


def test_resume_preserves_the_verdicts_of_rows_it_did_not_rerun():
    existing_verdicts = [
        verdict("adams", "HYBRID_COMPLETION", "q1", ["a"], []),
        verdict("adams", "HYBRID_COMPLETION", "q2", [], ["a"], error="answer: boom"),
    ]
    fresh_verdicts = [verdict("adams", "HYBRID_COMPLETION", "q2", ["a"], [])]
    order = [answer_row("q1"), answer_row("q2")]

    merged = lib.merge_verdict_rows(existing_verdicts, fresh_verdicts, order)

    assert [row.question_id for row in merged] == ["q1", "q2"]
    assert merged[0] is existing_verdicts[0]  # untouched, not re-judged
    assert merged[1].coverage == pytest.approx(1.0)
    assert merged[1].error is None


def test_resume_judges_rows_that_were_never_judged():
    answers = [answer_row("q1"), answer_row("q2")]
    already_judged = [verdict("adams", "HYBRID_COMPLETION", "q1", ["a"], [])]

    pending = lib.rows_needing_verdicts(answers, already_judged, reanswered=[])

    assert [row.question_id for row in pending] == ["q2"]


def test_resume_rejudges_a_row_it_reanswered():
    answers = [answer_row("q1"), answer_row("q2")]
    already_judged = [
        verdict("adams", "HYBRID_COMPLETION", "q1", ["a"], []),
        verdict("adams", "HYBRID_COMPLETION", "q2", [], ["a"], error="answer: boom"),
    ]

    pending = lib.rows_needing_verdicts(answers, already_judged, reanswered=[answers[1]])

    assert [row.question_id for row in pending] == ["q2"]


# ---------------------------------------------------------------------------
# report: answered column and error histogram
# ---------------------------------------------------------------------------


def test_report_renders_the_answered_column_and_the_error_histogram():
    rows = [
        verdict("adams", "HYBRID_COMPLETION", "q1", ["a", "b", "c"], ["d"]),
        verdict(
            "adams",
            "HYBRID_COMPLETION",
            "q2",
            [],
            [],
            error='answer: HttpError: HTTP 401: {"detail":"Unauthorized"}',
        ),
        verdict(
            "adams",
            "AUTO",
            "q1",
            [],
            [],
            error="answer: TransientHttpError: ReadTimeout: timed out",
        ),
    ]

    aggregates = lib.aggregate(rows)
    report = lib.render_report(aggregates, title="Run 2", error_histogram=lib.error_histogram(rows))

    assert "| dataset | search type | n | answered |" in report
    assert "| adams | HYBRID_COMPLETION | 2 | 1 |" in report
    assert "| adams | AUTO | 1 | 0 |" in report
    assert "401 | 1" in report
    assert "timeout | 1" in report


def test_an_all_error_run_reads_as_zero_answered():
    rows = [
        verdict("adams", "HYBRID_COMPLETION", f"q{index}", [], [], error="answer: boom")
        for index in range(3)
    ]

    aggregates = lib.aggregate(rows)

    assert aggregates[0].n == 3
    assert aggregates[0].answered == 0
    assert aggregates[0].errors == 3
    assert "| adams | HYBRID_COMPLETION | 3 | 0 |" in lib.render_report(aggregates)


# ---------------------------------------------------------------------------
# the judge environment (the baseline run graded nothing: LLMAPIKeyNotSetError)
# ---------------------------------------------------------------------------


def test_the_env_preamble_never_sets_an_empty_llm_api_key(monkeypatch):
    """An empty ``LLM_API_KEY`` overrides ``.env`` and makes cognee raise.

    The first live run answered 77 questions and graded none of them for exactly
    this reason: the preamble copied from the sibling scripts defaulted
    ``LLM_API_KEY`` to ``os.environ.get("OPENAI_API_KEY", "")``, so with neither
    variable set it exported the empty string over whatever ``.env`` held.
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    lib.configure_llm_environment()

    assert "LLM_API_KEY" not in __import__("os").environ


def test_the_env_preamble_maps_the_openai_key_when_there_is_one(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai-var")

    lib.configure_llm_environment()

    assert __import__("os").environ["LLM_API_KEY"] == "sk-from-openai-var"


def test_the_env_preamble_does_not_clobber_an_existing_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-already-set")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai-var")

    lib.configure_llm_environment()

    assert __import__("os").environ["LLM_API_KEY"] == "sk-already-set"


def test_the_gateway_maps_the_key_before_it_imports_cognee(monkeypatch):
    """The preamble existed but nothing called it, so the judge never saw the key.

    Order matters: cognee caches its LLM config on first read, so the mapping has
    to run before the import that triggers that read, not merely before the call.
    """
    import types

    calls: list[str] = []

    class _Gateway:
        @staticmethod
        async def acreate_structured_output(**kwargs):
            calls.append("gateway")
            return kwargs["response_model"]

    fake_module = types.ModuleType("cognee.infrastructure.llm")
    fake_module.LLMGateway = _Gateway
    monkeypatch.setitem(sys.modules, "cognee.infrastructure.llm", fake_module)
    monkeypatch.setattr(lib, "configure_llm_environment", lambda: calls.append("configure"))

    result = asyncio.run(
        lib._LazyLLMGateway.acreate_structured_output(
            text_input="q", system_prompt="s", response_model="model"
        )
    )

    assert result == "model"
    assert calls == ["configure", "gateway"]


def test_the_cli_maps_the_key_before_doing_anything_else(tmp_path, monkeypatch):
    calls: list[str] = []
    # ``cli`` imported its own copy of the library through sys.path, so patch that one.
    monkeypatch.setattr(cli.lib, "configure_llm_environment", lambda: calls.append("configure"))
    path = write_question_file(tmp_path, question_document())

    assert cli.main(["--validate-only", "--questions", str(path)]) == 0
    assert calls == ["configure"]


def test_append_jsonl_writes_one_flushed_line_per_row(tmp_path):
    journal = tmp_path / lib.PARTIAL_ANSWERS_FILENAME

    lib.append_jsonl(journal, answer_row("q1"))
    lib.append_jsonl(journal, answer_row("q2", error="answer: boom"))

    rows = [lib.AnswerRow.from_dict(row) for row in lib.read_jsonl(journal)]
    assert [row.question_id for row in rows] == ["q1", "q2"]
    assert rows[1].error == "answer: boom"


def test_recover_partial_answers_overlays_the_journal_in_place(tmp_path):
    """A run that died mid-resume already paid for some cells; do not ask again."""
    previous = [
        answer_row("q1"),
        answer_row("q2", error="answer: TransientHttpError: ReadTimeout"),
        answer_row("q3", error="answer: HttpError: 500"),
    ]
    lib.append_jsonl(tmp_path / lib.PARTIAL_ANSWERS_FILENAME, answer_row("q2", answer="recovered"))

    recovered = lib.recover_partial_answers(tmp_path, previous)

    assert [row.question_id for row in recovered] == ["q1", "q2", "q3"]
    assert recovered[1].answer == "recovered" and recovered[1].error is None
    assert recovered[2].error == "answer: HttpError: 500"
    assert [row.question_id for row in lib.rows_needing_answers(recovered)] == ["q3"]


def test_recover_partial_answers_without_a_journal_changes_nothing(tmp_path):
    previous = [answer_row("q1"), answer_row("q2", error="answer: boom")]

    assert lib.recover_partial_answers(tmp_path, previous) == previous


def test_recover_partial_verdicts_overlays_the_journal(tmp_path):
    existing = [verdict("adams", "HYBRID_COMPLETION", "q1", ["a"], [])]
    order = [answer_row("q1"), answer_row("q2")]
    lib.append_jsonl(
        tmp_path / lib.PARTIAL_VERDICTS_FILENAME,
        verdict("adams", "HYBRID_COMPLETION", "q2", ["a"], []),
    )

    recovered = lib.recover_partial_verdicts(tmp_path, existing, order)

    assert [row.question_id for row in recovered] == ["q1", "q2"]
    assert lib.rows_needing_verdicts(order, recovered) == []


def test_clear_partial_files_removes_both_journals(tmp_path):
    lib.append_jsonl(tmp_path / lib.PARTIAL_ANSWERS_FILENAME, answer_row("q1"))
    lib.append_jsonl(tmp_path / lib.PARTIAL_VERDICTS_FILENAME, {"question_id": "q1"})

    lib.clear_partial_files(tmp_path)
    lib.clear_partial_files(tmp_path)  # idempotent

    assert not (tmp_path / lib.PARTIAL_ANSWERS_FILENAME).exists()
    assert not (tmp_path / lib.PARTIAL_VERDICTS_FILENAME).exists()


def test_the_cli_reporters_journal_every_row(tmp_path, capsys):
    answers_journal = tmp_path / lib.PARTIAL_ANSWERS_FILENAME
    verdicts_journal = tmp_path / lib.PARTIAL_VERDICTS_FILENAME

    cli._progress_reporter(2, answers_journal)(answer_row("q1"))
    cli._progress_reporter(2)(answer_row("q2"))  # no journal → nothing written
    cli._verdict_reporter(1, verdicts_journal)(
        verdict("adams", "HYBRID_COMPLETION", "q1", ["a"], [])
    )

    assert [row["question_id"] for row in lib.read_jsonl(answers_journal)] == ["q1"]
    assert [row["question_id"] for row in lib.read_jsonl(verdicts_journal)] == ["q1"]
    capsys.readouterr()


def test_resume_recovers_the_journal_and_asks_only_for_the_rest(tmp_path, monkeypatch, capsys):
    """End to end through main: journal rows are kept, only the still-failed cell is re-asked."""
    document = {
        "corpus": "adams",
        "questions": [question_document(id=qid)["questions"][0] for qid in ("q1", "q2", "q3")],
    }
    questions_path = write_question_file(tmp_path, document)
    run_dir = tmp_path / "run"
    lib.write_jsonl(
        run_dir / lib.ANSWERS_FILENAME,
        [
            answer_row("q1"),
            answer_row("q2", error="answer: TransientHttpError: ReadTimeout"),
            answer_row("q3", error="answer: HttpError: 500"),
        ],
    )
    lib.append_jsonl(run_dir / lib.PARTIAL_ANSWERS_FILENAME, answer_row("q2", answer="recovered"))
    asked: list[list[str]] = []

    def fake_answer_rows(args, failed, by_id, run_directory):
        asked.append([row.question_id for row in failed])
        return [answer_row("q3", answer="fresh")]

    monkeypatch.setattr(cli, "_answer_rows", fake_answer_rows)
    monkeypatch.setattr(cli.lib, "configure_llm_environment", lambda: None)

    code = cli.main(["--resume", str(run_dir), "--questions", str(questions_path), "--no-judge"])

    assert code == 0
    assert asked == [["q3"]]
    rows = [lib.AnswerRow.from_dict(r) for r in lib.read_jsonl(run_dir / lib.ANSWERS_FILENAME)]
    assert [(row.question_id, row.answer) for row in rows] == [
        ("q1", "kept"),
        ("q2", "recovered"),
        ("q3", "fresh"),
    ]
    assert not (run_dir / lib.PARTIAL_ANSWERS_FILENAME).exists()
    captured = capsys.readouterr()
    assert "Recovered 1 row(s)" in captured.out + captured.err  # progress lines go to stderr


def test_resume_rejudges_a_row_whose_verdict_is_itself_an_error():
    """A verdict that failed to grade is a hole, not a grade.

    The whole baseline run carries ``judge: LLMAPIKeyNotSetError`` verdicts over
    perfectly good answers; a resume that treated those as "already judged" could
    never repair the run.
    """
    answers = [answer_row("q1"), answer_row("q2")]
    already_judged = [
        verdict("adams", "HYBRID_COMPLETION", "q1", ["a"], []),
        verdict(
            "adams",
            "HYBRID_COMPLETION",
            "q2",
            [],
            [],
            error="judge: LLMAPIKeyNotSetError: LLM API key is not set.",
        ),
    ]

    pending = lib.rows_needing_verdicts(answers, already_judged, reanswered=[])

    assert [row.question_id for row in pending] == ["q2"]


def test_resume_does_not_rejudge_a_row_whose_answer_is_still_broken():
    """No answer, nothing to grade - do not spend a judge call on it."""
    answers = [answer_row("q1", error="answer: TransientHttpError: ReadTimeout: timed out")]
    already_judged = [
        verdict(
            "adams",
            "HYBRID_COMPLETION",
            "q1",
            [],
            [],
            error="answer: TransientHttpError: ReadTimeout: timed out",
        )
    ]

    assert lib.rows_needing_verdicts(answers, already_judged, reanswered=[]) == []
