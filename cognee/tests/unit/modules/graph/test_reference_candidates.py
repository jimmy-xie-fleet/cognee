"""Unit tests for the pure reference-candidate helpers.

Deterministic and dependency-free: no LLM, no database, no network, no cognee I/O. These
are the labels, merge, penalty, and rendering helpers that the seed retrieval and the
agentic tracer tools build on top of -- an opaque label (``A1``/``P3``/``D2``) is the only
way a node id can enter an LLM finish, so the registry's stability and the merge's
ordering are pinned exactly here.
"""

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from cognee.modules.graph.utils.reference_candidates import (
    CANDIDATE_PREVIEW_CHARS,
    Candidate,
    LabelRegistry,
    apply_penalty,
    candidate_set_key,
    distance_to_similarity,
    format_candidate_lines,
    merge_candidates,
)

# --------------------------------------------------------------------------------------
# distance_to_similarity
# --------------------------------------------------------------------------------------


def test_distance_to_similarity_typical():
    assert distance_to_similarity(0.2) == pytest.approx(0.8)


def test_distance_to_similarity_clamps_above_one():
    assert distance_to_similarity(-0.5) == 1.0


def test_distance_to_similarity_clamps_below_zero():
    assert distance_to_similarity(1.5) == 0.0


def test_distance_to_similarity_boundary_zero_distance():
    assert distance_to_similarity(0.0) == 1.0


def test_distance_to_similarity_boundary_one_distance():
    assert distance_to_similarity(1.0) == 0.0


# --------------------------------------------------------------------------------------
# Candidate is frozen
# --------------------------------------------------------------------------------------


def test_candidate_is_frozen():
    candidate = Candidate(label="A1", node_id="n1", node_type="Assertion", score=0.9, text="hello")
    with pytest.raises(FrozenInstanceError):
        candidate.score = 0.1


def test_candidate_defaults():
    candidate = Candidate(label="A1", node_id="n1", node_type="Assertion", score=0.9, text="hello")
    assert candidate.document_id is None
    assert candidate.document_name is None
    assert candidate.chunk_index is None
    assert candidate.sources == ()


# --------------------------------------------------------------------------------------
# LabelRegistry
# --------------------------------------------------------------------------------------


def test_label_registry_prefixes_by_type():
    registry = LabelRegistry()
    assert registry.label("n1", "Assertion") == "A1"
    assert registry.label("n2", "DocumentChunk") == "P1"
    assert registry.label("n3", "TextSummary") == "S1"
    assert registry.label("n4", "Document") == "D1"
    assert registry.label("n5", "SomeWeirdType") == "N1"


def test_label_registry_document_subtype_prefix():
    registry = LabelRegistry()
    assert registry.label("n1", "LegalDocument") == "D1"


def test_label_registry_numbers_contiguously_per_prefix_first_seen_order():
    registry = LabelRegistry()
    assert registry.label("n1", "Assertion") == "A1"
    assert registry.label("n2", "DocumentChunk") == "P1"
    assert registry.label("n3", "Assertion") == "A2"
    assert registry.label("n4", "DocumentChunk") == "P2"
    assert registry.label("n5", "Assertion") == "A3"


def test_label_registry_stable_for_repeated_node_id():
    registry = LabelRegistry()
    first = registry.label("n1", "Assertion")
    second = registry.label("n1", "Assertion")
    assert first == second == "A1"


def test_label_registry_stable_even_if_type_hint_changes():
    registry = LabelRegistry()
    first = registry.label("n1", "Assertion")
    # Once assigned, a label never changes for a node id, even if called again with a
    # different (presumably wrong) type hint.
    second = registry.label("n1", "DocumentChunk")
    assert first == second == "A1"


def test_label_registry_resolve_known_label():
    registry = LabelRegistry()
    registry.label("n1", "Assertion")
    assert registry.resolve("A1") == "n1"


def test_label_registry_resolve_unknown_label_returns_none():
    registry = LabelRegistry()
    registry.label("n1", "Assertion")
    assert registry.resolve("A99") is None


def test_label_registry_resolve_none_returns_none():
    registry = LabelRegistry()
    assert registry.resolve(None) is None


def test_label_registry_resolve_is_case_insensitive_and_strips_whitespace():
    registry = LabelRegistry()
    registry.label("n1", "Assertion")
    assert registry.resolve(" a1 ") == "n1"
    assert registry.resolve("A1") == "n1"


def test_label_registry_node_type_lookup():
    registry = LabelRegistry()
    registry.label("n1", "Assertion")
    assert registry.node_type("A1") == "Assertion"
    assert registry.node_type("A99") is None


def test_label_registry_labels_snapshot():
    registry = LabelRegistry()
    registry.label("n1", "Assertion")
    registry.label("n2", "DocumentChunk")
    assert registry.labels() == {"A1": "n1", "P1": "n2"}


# --------------------------------------------------------------------------------------
# merge_candidates
# --------------------------------------------------------------------------------------


def _scored(node_id, node_type, similarity, source_tag, **payload):
    return (node_id, node_type, similarity, source_tag, payload)


def test_merge_candidates_union_keeps_best_score_and_accumulates_sources():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.5, "vector:hint", text="hello"),
        _scored("n1", "Assertion", 0.9, "bm25:proposition", text="hello"),
    ]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].sources == ("vector:hint", "bm25:proposition")


def test_merge_candidates_deduplicates_sources_first_seen_order():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.5, "vector:hint", text="hello"),
        _scored("n1", "Assertion", 0.9, "vector:hint", text="hello"),
        _scored("n1", "Assertion", 0.7, "bm25:proposition", text="hello"),
    ]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert result[0].sources == ("vector:hint", "bm25:proposition")


def test_merge_candidates_deterministic_order_by_score_then_node_id():
    registry = LabelRegistry()
    scored = [
        _scored("n2", "Assertion", 0.5, "vector:hint", text="b"),
        _scored("n1", "Assertion", 0.5, "vector:hint", text="a"),
        _scored("n3", "Assertion", 0.9, "vector:hint", text="c"),
    ]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert [c.node_id for c in result] == ["n3", "n1", "n2"]


def test_merge_candidates_truncates_to_limit_before_labelling():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.9, "vector:hint", text="a"),
        _scored("n2", "Assertion", 0.8, "vector:hint", text="b"),
        _scored("n3", "Assertion", 0.7, "vector:hint", text="c"),
    ]
    result = merge_candidates(scored, limit=2, registry=registry)
    assert [c.node_id for c in result] == ["n1", "n2"]
    assert [c.label for c in result] == ["A1", "A2"]
    # n3 was truncated away and never consumed a label.
    assert registry.resolve("A3") is None


def test_merge_candidates_assigns_labels_from_registry():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.9, "vector:hint", text="a"),
        _scored("n2", "DocumentChunk", 0.8, "vector:hint", text="b"),
    ]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert result[0].label == "A1"
    assert result[1].label == "P1"


def test_merge_candidates_carries_payload_fields():
    registry = LabelRegistry()
    scored = [
        _scored(
            "n1",
            "DocumentChunk",
            0.9,
            "vector:hint",
            text="a",
            document_id="doc-1",
            document_name="Complaint",
            chunk_index=4,
        )
    ]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert result[0].document_id == "doc-1"
    assert result[0].document_name == "Complaint"
    assert result[0].chunk_index == 4


def test_merge_candidates_collapses_whitespace_in_text():
    registry = LabelRegistry()
    scored = [_scored("n1", "Assertion", 0.9, "vector:hint", text="hello\n\n  world\t!")]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert result[0].text == "hello world !"


def test_merge_candidates_truncates_text_preview_with_ellipsis():
    registry = LabelRegistry()
    long_text = "word " * 100
    scored = [_scored("n1", "Assertion", 0.9, "vector:hint", text=long_text)]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert len(result[0].text) == CANDIDATE_PREVIEW_CHARS
    assert result[0].text.endswith("…")


def test_merge_candidates_short_text_not_truncated():
    registry = LabelRegistry()
    scored = [_scored("n1", "Assertion", 0.9, "vector:hint", text="short text")]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert result[0].text == "short text"
    assert not result[0].text.endswith("…")


def test_merge_candidates_missing_payload_fields_default_none():
    registry = LabelRegistry()
    scored = [_scored("n1", "Assertion", 0.9, "vector:hint", text="a")]
    result = merge_candidates(scored, limit=10, registry=registry)
    assert result[0].document_id is None
    assert result[0].document_name is None
    assert result[0].chunk_index is None


# --------------------------------------------------------------------------------------
# apply_penalty
# --------------------------------------------------------------------------------------


def test_apply_penalty_subtracts_amount():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.9, "vector:hint", text="a"),
        _scored("n2", "Assertion", 0.5, "vector:hint", text="b"),
    ]
    candidates = merge_candidates(scored, limit=10, registry=registry)
    result = apply_penalty(candidates, {"n1"}, 0.2)
    by_id = {c.node_id: c for c in result}
    assert by_id["n1"].score == pytest.approx(0.7)
    assert by_id["n2"].score == pytest.approx(0.5)


def test_apply_penalty_floors_at_zero():
    registry = LabelRegistry()
    scored = [_scored("n1", "Assertion", 0.1, "vector:hint", text="a")]
    candidates = merge_candidates(scored, limit=10, registry=registry)
    result = apply_penalty(candidates, {"n1"}, 0.5)
    assert result[0].score == 0.0


def test_apply_penalty_resorts_by_score_then_node_id():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.9, "vector:hint", text="a"),
        _scored("n2", "Assertion", 0.5, "vector:hint", text="b"),
    ]
    candidates = merge_candidates(scored, limit=10, registry=registry)
    result = apply_penalty(candidates, {"n1"}, 0.6)
    assert [c.node_id for c in result] == ["n2", "n1"]


def test_apply_penalty_preserves_labels():
    registry = LabelRegistry()
    scored = [
        _scored("n1", "Assertion", 0.9, "vector:hint", text="a"),
        _scored("n2", "Assertion", 0.5, "vector:hint", text="b"),
    ]
    candidates = merge_candidates(scored, limit=10, registry=registry)
    result = apply_penalty(candidates, {"n1"}, 0.6)
    by_id = {c.node_id: c for c in result}
    assert by_id["n1"].label == "A1"
    assert by_id["n2"].label == "A2"


def test_apply_penalty_ignores_unlisted_candidates():
    registry = LabelRegistry()
    scored = [_scored("n1", "Assertion", 0.9, "vector:hint", text="a")]
    candidates = merge_candidates(scored, limit=10, registry=registry)
    result = apply_penalty(candidates, {"n2"}, 0.5)
    assert result[0].score == 0.9


# --------------------------------------------------------------------------------------
# format_candidate_lines
# --------------------------------------------------------------------------------------


def test_format_candidate_lines_assertion_with_chunk():
    candidate = Candidate(
        label="A7",
        node_id="n1",
        node_type="Assertion",
        score=0.9,
        text="The tenant paid rent.",
        document_name="Complaint",
        chunk_index=4,
    )
    lines = format_candidate_lines([candidate])
    assert lines == '[A7] Assertion in "Complaint" (chunk 4): "The tenant paid rent."'


def test_format_candidate_lines_passage_type_word():
    candidate = Candidate(
        label="P3",
        node_id="n1",
        node_type="DocumentChunk",
        score=0.9,
        text="text",
        document_name="Complaint",
        chunk_index=2,
    )
    lines = format_candidate_lines([candidate])
    assert lines == '[P3] Passage in "Complaint" (chunk 2): "text"'


def test_format_candidate_lines_summary_type_word():
    candidate = Candidate(
        label="S1",
        node_id="n1",
        node_type="TextSummary",
        score=0.9,
        text="summary text",
        document_name="Complaint",
        chunk_index=0,
    )
    lines = format_candidate_lines([candidate])
    assert lines == '[S1] Summary in "Complaint" (chunk 0): "summary text"'


def test_format_candidate_lines_document():
    candidate = Candidate(
        label="D2",
        node_id="n1",
        node_type="Document",
        score=0.9,
        text="preview text",
        document_name="Lease Agreement",
    )
    lines = format_candidate_lines([candidate])
    assert lines == '[D2] Document "Lease Agreement": "preview text"'


def test_format_candidate_lines_omits_chunk_when_none():
    candidate = Candidate(
        label="A1",
        node_id="n1",
        node_type="Assertion",
        score=0.9,
        text="text",
        document_name="Complaint",
        chunk_index=None,
    )
    lines = format_candidate_lines([candidate])
    assert lines == '[A1] Assertion in "Complaint": "text"'


def test_format_candidate_lines_falls_back_to_document_id():
    candidate = Candidate(
        label="A1",
        node_id="n1",
        node_type="Assertion",
        score=0.9,
        text="text",
        document_id="doc-123",
        document_name=None,
    )
    lines = format_candidate_lines([candidate])
    assert '"doc-123"' in lines


def test_format_candidate_lines_falls_back_to_unknown_document():
    candidate = Candidate(
        label="A1",
        node_id="n1",
        node_type="Assertion",
        score=0.9,
        text="text",
        document_id=None,
        document_name=None,
    )
    lines = format_candidate_lines([candidate])
    assert '"unknown document"' in lines


def test_format_candidate_lines_multiple_candidates_one_line_each():
    candidates = [
        Candidate(
            label="A1",
            node_id="n1",
            node_type="Assertion",
            score=0.9,
            text="first",
            document_name="Doc A",
            chunk_index=0,
        ),
        Candidate(
            label="D2",
            node_id="n2",
            node_type="Document",
            score=0.5,
            text="second",
            document_name="Doc B",
        ),
    ]
    lines = format_candidate_lines(candidates)
    assert lines.splitlines() == [
        '[A1] Assertion in "Doc A" (chunk 0): "first"',
        '[D2] Document "Doc B": "second"',
    ]


def test_format_candidate_lines_asserts_text_has_no_newlines():
    candidate = Candidate(
        label="A1",
        node_id="n1",
        node_type="Assertion",
        score=0.9,
        text="line one\nline two",
        document_name="Doc A",
    )
    with pytest.raises(AssertionError):
        format_candidate_lines([candidate])


def test_format_candidate_lines_empty_list_is_empty_string():
    assert format_candidate_lines([]) == ""


# --------------------------------------------------------------------------------------
# candidate_set_key
# --------------------------------------------------------------------------------------


def _candidate(node_id):
    return Candidate(label="A1", node_id=node_id, node_type="Assertion", score=0.9, text="a")


def test_candidate_set_key_order_independent():
    forward = candidate_set_key([_candidate("n1"), _candidate("n2")])
    backward = candidate_set_key([_candidate("n2"), _candidate("n1")])
    assert forward == backward


def test_candidate_set_key_changes_with_membership():
    key_ab = candidate_set_key([_candidate("n1"), _candidate("n2")])
    key_ac = candidate_set_key([_candidate("n1"), _candidate("n3")])
    assert key_ab != key_ac


def test_candidate_set_key_is_sha1_prefix_of_sorted_ids():
    candidates = [_candidate("n2"), _candidate("n1")]
    expected = hashlib.sha1(",".join(sorted(["n1", "n2"])).encode("utf-8")).hexdigest()[:16]
    assert candidate_set_key(candidates) == expected


def test_candidate_set_key_length_is_16():
    assert len(candidate_set_key([_candidate("n1")])) == 16


def test_candidate_set_key_empty_list():
    key = candidate_set_key([])
    assert isinstance(key, str)
