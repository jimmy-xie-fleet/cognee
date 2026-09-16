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


class FakeClient:
    """Records every call and replays canned responses (or raises)."""

    def __init__(self, responder=None):
        self.calls = []
        self._responder = responder or (lambda path, body: [{"search_result": ["ok"]}])

    def post(self, path, *, json=None, data=None, headers=None):
        self.calls.append({"path": path, "json": json, "data": data, "headers": headers})
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
        client,
        token="t",
        questions=[make_question()],
        datasets=["adams"],
        search_types=[lib.AUTO_SEARCH_TYPE],
    )

    assert [call["path"] for call in client.calls] == ["/api/v1/recall", "/api/v1/recall"]
    first_body = client.calls[0]["json"]
    assert "search_type" in first_body and first_body["search_type"] is None
    assert first_body["datasets"] == ["adams"]
    assert rows[0].error is None


def test_pinned_search_type_posts_to_search():
    client = FakeClient()

    lib.run_answers(
        client,
        token="t",
        questions=[make_question()],
        datasets=["adams"],
        search_types=["GRAPH_COMPLETION"],
    )

    assert {call["path"] for call in client.calls} == {"/api/v1/search"}
    assert client.calls[0]["json"]["search_type"] == "GRAPH_COMPLETION"


def test_each_question_makes_an_answer_call_and_a_context_call():
    def responder(path, body):
        return [{"search_result": ["context text" if body.get("only_context") else "answer text"]}]

    client = FakeClient(responder)

    rows = lib.run_answers(
        client,
        token="token-value",
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
        top_k=7,
    )

    assert len(client.calls) == 2
    answer_call, context_call = client.calls
    assert "only_context" not in answer_call["json"]
    assert answer_call["json"]["top_k"] == 7
    assert context_call["json"]["only_context"] is True
    assert context_call["json"]["context_format"] == "context"
    assert answer_call["headers"] == {"Authorization": "Bearer token-value"}
    assert rows[0].answer == "answer text"
    assert rows[0].context == "context text"


def test_recall_shaped_response_is_flattened_too():
    payload = [{"source": "graph", "kind": "graph_completion", "text": "the denial is in para 12"}]

    assert lib.extract_text(payload) == "the denial is in para 12"


def test_transport_error_becomes_an_error_row_rather_than_an_exception():
    client = FakeClient(lambda path, body: lib.TransientHttpError("connection refused"))

    rows = lib.run_answers(
        client,
        token="t",
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert len(rows) == 1
    assert rows[0].error is not None
    assert "connection refused" in rows[0].error
    assert rows[0].answer == ""
    # one original attempt plus exactly one retry
    assert len(client.calls) == 2


def test_a_transient_failure_is_retried_once_and_then_succeeds():
    attempts = {"count": 0}

    def responder(path, body):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return lib.TransientHttpError("read timeout")
        return [{"search_result": ["fine"]}]

    client = FakeClient(responder)

    rows = lib.run_answers(
        client,
        token="t",
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert rows[0].error is None
    assert rows[0].answer == "fine"


def test_a_permanent_failure_is_not_retried():
    client = FakeClient(lambda path, body: lib.HttpError("HTTP 403: forbidden"))

    rows = lib.run_answers(
        client,
        token="t",
        questions=[make_question()],
        datasets=["adams"],
        search_types=["HYBRID_COMPLETION"],
    )

    assert "403" in rows[0].error
    assert len(client.calls) == 1


def test_login_returns_the_token_from_the_form_post():
    client = FakeClient(lambda path, body: {"access_token": "abc", "token_type": "bearer"})

    token = lib.login(client, "user@example.com", "hunter2")

    assert token == "abc"
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
    assert "| dataset | search type | n | mean coverage |" in report
    assert "| adams | HYBRID_COMPLETION | 1 | 75.0% | 0 | 0 | 0 | 0 |" in report
    assert "| adams | AUTO | 1 | 0.0% | 0 | 0 | 0 | 1 |" in report


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
