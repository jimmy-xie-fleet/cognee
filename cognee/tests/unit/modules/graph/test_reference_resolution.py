"""Unit tests for the pure reference-resolution helpers.

Deterministic and dependency-free: no LLM, no database, no network. Every value asserted
here comes out of the formulas documented in ``reference_resolution`` itself.
"""

import json
from dataclasses import FrozenInstanceError

import pytest

from cognee.modules.chunking.incremental_chunking import chunk_offsets
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.graph.utils.reference_resolution import (
    DEFAULT_MAX_SPAN,
    RESOLVED_BY,
    STRATEGY_ENTITY_NAME,
    STRATEGY_EXISTING_ID,
    STRATEGY_LLM_TRACE,
    Locator,
    ReferenceHint,
    Resolution,
    anchor_chunk_index,
    build_locator,
    build_node_patch,
    build_reference_edge,
    chunks_overlapping,
    derived_edge_text,
    find_locator_span,
    normalize_reference_text,
    parse_reference_hint,
    reference_display_text,
    reference_fingerprint,
    roman_to_int,
    scan_chunks_for_marker,
    select_anchored_assertions,
    stance_edge_text,
)


def test_normalize_reference_text_folds_quotes_case_and_whitespace():
    assert normalize_reference_text("  The “Subject\nProperty”  ") == 'the "subject property"'
    assert normalize_reference_text(None) == ""


@pytest.mark.parametrize(
    "value, expected",
    [("i", 1), ("II", 2), ("iv", 4), ("IX", 9), ("xiv", 14), ("", None), ("abc", None)],
)
def test_roman_to_int(value, expected):
    assert roman_to_int(value) == expected


# --------------------------------------------------------------------------------------
# find_locator_span
# --------------------------------------------------------------------------------------

COMPLAINT_TEXT = (
    "VERIFIED COMPLAINT\n"
    "\n"
    "4. Defendant is a municipal corporation.\n"
    "5. Plaintiff owns 10 Main Street.\n"
    "6. The property was acquired in 1998.\n"
    "7. The taking occurred in 2026.\n"
)


def _locator(kind, value):
    """The locator a (kind, value) pair names -- the only way one is built now."""
    return build_locator(kind, value)


def test_find_locator_span_numbered_paragraph_ends_at_the_next_marker():
    span = find_locator_span(COMPLAINT_TEXT, _locator("paragraph", "5"))
    assert span is not None
    start, end, notes = span
    assert start == COMPLAINT_TEXT.index("5. Plaintiff")
    assert end == COMPLAINT_TEXT.index("6. The property")
    assert notes == ()
    assert COMPLAINT_TEXT[start:end] == "5. Plaintiff owns 10 Main Street.\n"


def test_find_locator_span_pilcrow_marker():
    text = "¶ 4 Something else.\n¶ 5 The claim at issue.\n¶ 6 Another claim.\n"
    start, end, notes = find_locator_span(text, _locator("paragraph", "5"))
    assert text[start:end] == "¶ 5 The claim at issue.\n"
    assert notes == ()


def test_find_locator_span_picks_the_marker_followed_by_the_next_ordinal():
    text = (
        "5. Item from an unrelated numbered list.\n"
        "9. Another item from that list.\n"
        "\n"
        "5. Plaintiff owns 10 Main Street.\n"
        "6. The property was acquired in 1998.\n"
    )
    start, end, notes = find_locator_span(text, _locator("paragraph", "5"))
    assert text[start:end] == "5. Plaintiff owns 10 Main Street.\n"
    assert notes == ()


def test_find_locator_span_flags_duplicate_markers_it_cannot_order():
    text = "5. First copy.\n9. Filler.\n\n5. Second copy.\n8. More filler.\n"
    start, end, notes = find_locator_span(text, _locator("paragraph", "5"))
    assert start == text.index("5. First copy.")
    assert notes == ("ambiguous_marker",)


def test_find_locator_span_ends_at_an_all_caps_heading():
    text = "5. Plaintiff owns 10 Main Street.\n\nSECOND CAUSE OF ACTION\n\nMore prose.\n"
    start, end, _ = find_locator_span(text, _locator("paragraph", "5"))
    assert end == text.index("SECOND CAUSE OF ACTION")
    assert text[start:end] == "5. Plaintiff owns 10 Main Street.\n\n"


def test_find_locator_span_ends_at_the_end_of_the_text():
    text = "5. Plaintiff owns 10 Main Street."
    start, end, _ = find_locator_span(text, _locator("paragraph", "5"))
    assert (start, end) == (0, len(text))


def test_find_locator_span_is_bounded_by_max_span():
    text = "5. " + "a" * 500
    start, end, _ = find_locator_span(text, _locator("paragraph", "5"), max_span=50)
    assert (start, end) == (0, 50)


def test_find_locator_span_roman_numeral_count():
    text = "COUNT I\nFirst count.\nCOUNT II\nSecond count.\nCOUNT III\nThird count.\n"
    start, end, notes = find_locator_span(text, _locator("count", "II"))
    assert text[start:end] == "COUNT II\nSecond count.\n"
    assert notes == ()


def test_find_locator_span_matches_a_digit_marker_from_a_roman_reference():
    text = "Article 3\nThird article.\nArticle 4\nFourth article.\n"
    start, end, _ = find_locator_span(text, _locator("article", "IV"))
    assert text[start:end] == "Article 4\nFourth article.\n"


def test_find_locator_span_returns_none_when_the_marker_is_absent():
    assert find_locator_span(COMPLAINT_TEXT, _locator("paragraph", "12")) is None


def test_find_locator_span_returns_none_for_a_document_level_locator():
    # build_locator already refuses a document-level kind, and find_locator_span refuses
    # the Locator a caller builds by hand: a document number marks no place in a text.
    assert _locator("resolution", "2026-118") is None
    assert (
        find_locator_span(
            "Resolution 2026-118 text.", Locator(kind="resolution", number="2026-118")
        )
        is None
    )


# --------------------------------------------------------------------------------------
# chunks_overlapping / anchor_chunk_index
# --------------------------------------------------------------------------------------


def test_chunks_overlapping_and_anchor_over_real_chunk_offsets():
    marker_start = COMPLAINT_TEXT.index("5. Plaintiff")
    # The chunk boundary falls just after the paragraph marker, so the marker and its body
    # live in different chunks -- the worked example's shape.
    boundary = marker_start + len("5. ")
    chunks = [COMPLAINT_TEXT[:boundary], COMPLAINT_TEXT[boundary:]]
    offsets = chunk_offsets(COMPLAINT_TEXT, chunks)

    span = find_locator_span(COMPLAINT_TEXT, _locator("paragraph", "5"))[:2]
    assert chunks_overlapping(offsets, span) == [0, 1]
    assert anchor_chunk_index(offsets, span) == 1


def test_chunks_overlapping_ignores_touching_but_not_overlapping_chunks():
    offsets = [(0, 10), (10, 20), (20, 30)]
    assert chunks_overlapping(offsets, (10, 20)) == [1]
    assert chunks_overlapping(offsets, (9, 21)) == [0, 1, 2]
    assert chunks_overlapping(offsets, (30, 40)) == []


def test_anchor_chunk_index_breaks_ties_on_the_lowest_index():
    offsets = [(0, 10), (10, 20)]
    assert anchor_chunk_index(offsets, (5, 15)) == 0
    assert anchor_chunk_index(offsets, (0, 0)) is None
    assert anchor_chunk_index([], (0, 5)) is None


# --------------------------------------------------------------------------------------
# scan_chunks_for_marker
# --------------------------------------------------------------------------------------


def test_scan_chunks_for_marker_returns_the_first_chunk_carrying_the_marker():
    chunks = [
        "VERIFIED COMPLAINT\n\n4. Defendant is a municipal corporation.\n",
        "5. Plaintiff owns 10 Main Street.\n6. The property was acquired in 1998.\n",
    ]
    hit = scan_chunks_for_marker(chunks, _locator("paragraph", "5"))
    assert hit is not None
    chunk_index, (start, end) = hit
    assert chunk_index == 1
    assert chunks[1][start:end] == "5. Plaintiff owns 10 Main Street.\n"


def test_scan_chunks_for_marker_bounds_the_span_by_the_chunk():
    chunks = ["intro\n", "5. Plaintiff owns 10 Main Street.\n"]
    chunk_index, (start, end) = scan_chunks_for_marker(chunks, _locator("paragraph", "5"))
    assert (chunk_index, start, end) == (1, 0, len(chunks[1]))


def test_scan_chunks_for_marker_without_a_hit():
    assert scan_chunks_for_marker(["nothing here"], _locator("paragraph", "5")) is None


# --------------------------------------------------------------------------------------
# select_anchored_assertions
# --------------------------------------------------------------------------------------


def test_select_anchored_assertions_matches_across_quotes_and_whitespace():
    span_text = (
        "5. Plaintiff owns 10 Main Street, the “Subject Property”,\n"
        "   which was acquired in 1998.\n"
    )
    selected = select_anchored_assertions(
        span_text,
        [
            ("a1", "owns 10 Main Street"),
            ("a2", "acquired  in\n1998"),
            ("a3", 'the "Subject Property"'),
            ("a4", "the taking occurred in 2026"),
            ("a5", None),
            ("a6", "   "),
        ],
    )
    assert selected == ["a1", "a2", "a3"]


def test_select_anchored_assertions_with_no_candidates():
    assert select_anchored_assertions("any text", []) == []


# --------------------------------------------------------------------------------------
# edge and node writes
# --------------------------------------------------------------------------------------

DENIAL_PROPS = {
    "name": "Payment was late",
    "statement_type": "denial",
    "polarity": "negative",
    "description": "Smith's answer to paragraph 5.",
}


def test_derived_edge_text_states_the_speakers_stance():
    assert (
        derived_edge_text("Payment was late", "denial", "negative", "asserted_by", "Smith", None)
        == "Smith denies that Payment was late."
    )
    assert (
        derived_edge_text("Payment was late", "denial", "unknown", "asserted_by", "Smith", None)
        == "Smith takes an unrecorded stance on Payment was late."
    )


def test_derived_edge_text_keeps_the_stance_on_a_reference_edge():
    assert derived_edge_text(
        "Payment was late",
        "denial",
        "negative",
        "responds_to",
        "Payment was late",
        "Answer ¶5.",
    ) == ("Payment was late (denial, negative stance) responds to Payment was late. Answer ¶5.")


def test_derived_edge_text_falls_back_for_blank_values():
    assert derived_edge_text(None, None, None, "responds_to", None, None) == (
        "this statement (statement, unknown stance) responds to an unnamed party."
    )


def test_stance_edge_text_reads_the_assertion_properties():
    assert stance_edge_text(DENIAL_PROPS, "responds_to", "Payment was late") == (
        "Payment was late (denial, negative stance) responds to Payment was late. "
        "Smith's answer to paragraph 5."
    )


def _resolution(**overrides):
    values = dict(
        assertion_id="a-denial",
        field="responds_to",
        reference_text="Complaint ¶5",
        strategy=STRATEGY_LLM_TRACE,
        confidence=0.9,
        anchor_id="chunk-1",
        anchor_type="chunk",
        target_ids=("a-1", "a-2"),
        target_type="assertion",
        document_id="complaint",
        notes=(),
    )
    values.update(overrides)
    return Resolution(**values)


def test_build_reference_edge_carries_the_resolution_provenance():
    source, target, relationship_name, props = build_reference_edge(
        _resolution(),
        "a-1",
        source_props=DENIAL_PROPS,
        target_label="Payment was late",
        target_type="assertion",
    )
    assert (source, target, relationship_name) == ("a-denial", "a-1", "responds_to")
    assert props["relationship_name"] == "responds_to"
    assert props["source_node_id"] == "a-denial"
    assert props["target_node_id"] == "a-1"
    assert props["reference_text"] == "Complaint ¶5"
    assert props["resolution_strategy"] == STRATEGY_LLM_TRACE
    assert props["resolution_confidence"] == 0.9
    assert props["resolved_target_type"] == "assertion"
    assert props["resolved_by"] == RESOLVED_BY == "reference_resolver"
    assert props["edge_text"].startswith("Payment was late (denial, negative stance) responds to")


def test_build_node_patch_records_the_resolution():
    patch = build_node_patch(_resolution(notes=("ambiguous_marker",)), {})
    assert patch["responds_to"] == "chunk-1"
    assert patch["responds_to_text"] == "Complaint ¶5"
    assert patch["responds_to_resolution"] == {
        "strategy": STRATEGY_LLM_TRACE,
        "confidence": 0.9,
        "target_type": "assertion",
        "target_ids": ["a-1", "a-2"],
        "anchor_id": "chunk-1",
        "document_id": "complaint",
        "notes": ["ambiguous_marker"],
        "reason": None,
        "fingerprint": None,
        "iterations": 0,
        "max_iter": None,
        "trace": [],
    }


def test_build_node_patch_preserves_an_existing_reference_text():
    patch = build_node_patch(
        _resolution(reference_text="Complaint ¶5"),
        {"responds_to_text": "the original locator text"},
    )
    assert patch["responds_to_text"] == "the original locator text"


def test_strategy_names_are_the_documented_values():
    assert (
        STRATEGY_EXISTING_ID,
        STRATEGY_ENTITY_NAME,
        STRATEGY_LLM_TRACE,
    ) == (
        "existing_id",
        "entity_name",
        "llm_trace",
    )


def test_resolution_carries_the_tracer_fields_with_defaults():
    plain = _resolution()
    assert (plain.reason, plain.fingerprint, plain.patch_mode) == (None, None, "full")
    assert (plain.iterations, plain.trace) == (0, ())

    traced = _resolution(
        strategy=STRATEGY_LLM_TRACE,
        reason="the answer restates the allegation",
        fingerprint="abc123",
        patch_mode="resolution_only",
        iterations=2,
        trace=({"tool": "search", "args": "{}", "result_preview": "", "ok": True},),
    )
    assert traced.reason == "the answer restates the allegation"
    assert traced.fingerprint == "abc123"
    assert traced.patch_mode == "resolution_only"
    assert traced.iterations == 2
    assert traced.trace[0]["tool"] == "search"
    with pytest.raises(FrozenInstanceError):
        traced.patch_mode = "full"


def test_build_node_patch_records_the_trace_fields():
    patch = build_node_patch(
        _resolution(
            strategy=STRATEGY_LLM_TRACE,
            reason="quoted verbatim",
            fingerprint="ff00",
            iterations=3,
            trace=(
                {
                    "tool": "read_chunk",
                    "args": '{"passage": "P1"}',
                    "result_preview": "x",
                    "ok": True,
                },
            ),
        ),
        {},
    )
    resolution = patch["responds_to_resolution"]
    assert resolution["reason"] == "quoted verbatim"
    assert resolution["fingerprint"] == "ff00"
    assert resolution["iterations"] == 3
    assert resolution["trace"] == [
        {"tool": "read_chunk", "args": '{"passage": "P1"}', "result_preview": "x", "ok": True}
    ]


def test_build_node_patch_resolution_only_never_touches_the_field():
    """An abstention must not overwrite a field the extraction left as written."""
    patch = build_node_patch(
        _resolution(
            strategy=STRATEGY_LLM_TRACE,
            anchor_id=None,
            target_ids=(),
            confidence=0.4,
            patch_mode="resolution_only",
            notes=("llm_below_threshold",),
        ),
        {"responds_to_text": "Complaint \u00b65"},
        mode="resolution_only",
    )
    assert set(patch) == {"responds_to_resolution"}
    assert patch["responds_to_resolution"]["notes"] == ["llm_below_threshold"]
    assert patch["responds_to_resolution"]["anchor_id"] is None


def test_build_node_patch_full_mode_is_the_default():
    assert build_node_patch(_resolution(), {}) == build_node_patch(_resolution(), {}, mode="full")


# --------------------------------------------------------------------------------------
# Assertion identity regression
# --------------------------------------------------------------------------------------


def test_assertion_identity_is_unchanged_by_the_resolution_fields():
    base = Assertion(
        name="the sky is blue",
        description="a claim about the sky",
        source_chunk_id="chunk-1",
        statement_type="testimony",
        asserted_by="witness a",
        occurrence=2,
    )
    resolved = Assertion(
        name="the sky is blue",
        description="a claim about the sky",
        source_chunk_id="chunk-1",
        statement_type="testimony",
        asserted_by="witness a",
        occurrence=2,
        responds_to="chunk-9",
        responds_to_text="Complaint ¶5",
        responds_to_resolution={"strategy": "document_locator"},
        responds_to_ref={"document_hint": "the Complaint", "locator_value": "5"},
        attributed_to_text="Whitfield report",
        attributed_to_resolution={"strategy": "document_only"},
        attributed_to_ref={"document_hint": "the Whitfield report"},
    )
    expected_id = Assertion.id_for("the sky is blue", "chunk-1", "testimony", "witness a", 2)
    assert base.id == expected_id
    assert resolved.id == expected_id
    # The sibling fields are stored, not silently dropped as unknown keyword arguments.
    assert resolved.responds_to == "chunk-9"
    assert resolved.responds_to_text == "Complaint ¶5"
    assert resolved.responds_to_resolution == {"strategy": "document_locator"}
    assert resolved.responds_to_ref == {"document_hint": "the Complaint", "locator_value": "5"}
    assert resolved.attributed_to_text == "Whitfield report"
    assert resolved.attributed_to_resolution == {"strategy": "document_only"}
    assert resolved.attributed_to_ref == {"document_hint": "the Whitfield report"}
    assert base.responds_to_text is None
    assert Assertion.model_fields["metadata"].default["identity_fields"] == [
        "name",
        "source_chunk_id",
        "statement_type",
        "asserted_by",
        "occurrence",
    ]


def test_locator_dataclass_is_hashable_and_frozen():
    locator = Locator(kind="paragraph", number="5", ordinal=5)
    assert hash(locator) == hash(Locator(kind="paragraph", number="5", ordinal=5))
    with pytest.raises(FrozenInstanceError):
        locator.number = "6"


# --------------------------------------------------------------------------------------
# parse_reference_hint
# --------------------------------------------------------------------------------------


def test_parse_reference_hint_reads_a_dict():
    hint = parse_reference_hint(
        {
            "document_hint": "the Complaint",
            "locator_kind": "paragraph",
            "locator_value": "17",
            "basis": "cited",
        }
    )
    assert hint == ReferenceHint(
        document_hint="the Complaint",
        locator_kind="paragraph",
        locator_value="17",
        date=None,
        basis="cited",
        legacy_text=None,
    )


def test_parse_reference_hint_reads_a_json_string():
    raw = json.dumps(
        {
            "document_hint": "the Whitfield report",
            "date": "2026-06-10",
        }
    )
    hint = parse_reference_hint(raw)
    assert hint == ReferenceHint(document_hint="the Whitfield report", date="2026-06-10")


def test_parse_reference_hint_falls_back_to_legacy_text():
    hint = parse_reference_hint(None, fallback_text="Complaint ¶5")
    assert hint == ReferenceHint(document_hint="Complaint ¶5", legacy_text="Complaint ¶5")


def test_parse_reference_hint_non_json_string_falls_through_to_fallback():
    # A bare, non-JSON string in `raw` is NOT a hint dict -- it falls through to
    # `fallback_text`, it is never treated as the legacy text itself.
    hint = parse_reference_hint("Complaint ¶5", fallback_text="Complaint ¶5")
    assert hint == ReferenceHint(document_hint="Complaint ¶5", legacy_text="Complaint ¶5")

    assert parse_reference_hint("Complaint ¶5") is None


def test_parse_reference_hint_none_without_fallback_is_none():
    assert parse_reference_hint(None) is None
    assert parse_reference_hint(None, fallback_text="") is None
    assert parse_reference_hint(None, fallback_text="   ") is None


@pytest.mark.parametrize("garbage", [123, 4.5, ["a", "b"], True, {"basis": "cited"}])
def test_parse_reference_hint_garbage_is_none(garbage):
    assert parse_reference_hint(garbage) is None


def test_parse_reference_hint_garbage_json_value_falls_through():
    # `json.loads` succeeds but yields a non-dict (a bare JSON string) -- still not a hint.
    assert parse_reference_hint('"just a json string"') is None
    assert parse_reference_hint("not json {") is None
    assert parse_reference_hint("[1, 2, 3]") is None


def test_parse_reference_hint_strips_and_blanks_fields():
    hint = parse_reference_hint(
        {
            "document_hint": "  the Complaint  ",
            "locator_kind": "  paragraph ",
            "locator_value": " 17 ",
            "date": "  ",
            "basis": None,
        }
    )
    assert hint == ReferenceHint(
        document_hint="the Complaint",
        locator_kind="paragraph",
        locator_value="17",
        date=None,
        basis=None,
    )


def test_parse_reference_hint_drops_blank_document_hint_to_empty_string():
    hint = parse_reference_hint(
        {"document_hint": "   ", "locator_kind": "paragraph", "locator_value": "3"}
    )
    assert hint.document_hint == ""
    assert hint.locator_kind == "paragraph"


@pytest.mark.parametrize("value", ["none", "None", "NONE", "", "   "])
def test_parse_reference_hint_drops_none_and_blank_markers(value):
    hint = parse_reference_hint(
        {
            "document_hint": "the Complaint",
            "locator_kind": value,
            "locator_value": value,
            "date": value,
            "basis": value,
        }
    )
    assert hint.locator_kind is None
    assert hint.locator_value is None
    assert hint.date is None
    assert hint.basis is None
    assert hint.document_hint == "the Complaint"


def test_parse_reference_hint_empty_dict_falls_back():
    hint = parse_reference_hint({}, fallback_text="Complaint ¶5")
    assert hint == ReferenceHint(document_hint="Complaint ¶5", legacy_text="Complaint ¶5")
    assert parse_reference_hint({}) is None


def test_parse_reference_hint_never_raises():
    for raw in (object(), b"bytes", {"document_hint": object()}):
        parse_reference_hint(raw)
    parse_reference_hint(None, fallback_text=object())


# --------------------------------------------------------------------------------------
# reference_display_text
# --------------------------------------------------------------------------------------


def test_reference_display_text_document_and_locator():
    hint = ReferenceHint(document_hint="Complaint", locator_kind="paragraph", locator_value="13")
    assert reference_display_text(hint) == "Complaint paragraph 13"


def test_reference_display_text_document_and_date():
    hint = ReferenceHint(document_hint="June 10 letter", date="2026-06-10")
    assert reference_display_text(hint) == "June 10 letter (2026-06-10)"


def test_reference_display_text_document_locator_and_date():
    hint = ReferenceHint(
        document_hint="Complaint", locator_kind="paragraph", locator_value="13", date="2026-06-10"
    )
    assert reference_display_text(hint) == "Complaint paragraph 13 (2026-06-10)"


def test_reference_display_text_locator_needs_both_kind_and_value():
    hint = ReferenceHint(document_hint="Complaint", locator_kind="paragraph", locator_value=None)
    assert reference_display_text(hint) == "Complaint"


def test_reference_display_text_legacy_text_wins():
    hint = ReferenceHint(
        document_hint="ignored",
        locator_kind="paragraph",
        locator_value="1",
        legacy_text="the Adams stipulation",
    )
    assert reference_display_text(hint) == "the Adams stipulation"


def test_reference_display_text_collapses_whitespace():
    hint = ReferenceHint(document_hint="the   Complaint")
    assert reference_display_text(hint) == "the Complaint"


def test_reference_display_text_empty_hint_is_blank():
    assert reference_display_text(ReferenceHint()) == ""


# --------------------------------------------------------------------------------------
# reference_fingerprint
# --------------------------------------------------------------------------------------


def test_reference_fingerprint_is_stable():
    hint = ReferenceHint(document_hint="Complaint", locator_kind="paragraph", locator_value="5")
    assert reference_fingerprint(hint, "responds_to") == reference_fingerprint(hint, "responds_to")
    fingerprint = reference_fingerprint(hint, "responds_to")
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16
    int(fingerprint, 16)  # hex-decodable


def test_reference_fingerprint_is_sensitive_to_field_name():
    hint = ReferenceHint(document_hint="Complaint", locator_kind="paragraph", locator_value="5")
    assert reference_fingerprint(hint, "responds_to") != reference_fingerprint(
        hint, "attributed_to"
    )


@pytest.mark.parametrize(
    "other",
    [
        ReferenceHint(document_hint="Complaint2", locator_kind="paragraph", locator_value="5"),
        ReferenceHint(document_hint="Complaint", locator_kind="section", locator_value="5"),
        ReferenceHint(document_hint="Complaint", locator_kind="paragraph", locator_value="6"),
        ReferenceHint(
            document_hint="Complaint",
            locator_kind="paragraph",
            locator_value="5",
            date="2026-06-10",
        ),
        ReferenceHint(document_hint="Complaint", legacy_text="Complaint"),
    ],
)
def test_reference_fingerprint_is_sensitive_to_every_field(other):
    base = ReferenceHint(document_hint="Complaint", locator_kind="paragraph", locator_value="5")
    assert reference_fingerprint(base, "responds_to") != reference_fingerprint(other, "responds_to")


# --------------------------------------------------------------------------------------
# build_locator
# --------------------------------------------------------------------------------------


def test_build_locator_digits():
    assert build_locator("paragraph", "17") == Locator(kind="paragraph", number="17", ordinal=17)


def test_build_locator_number_word():
    assert build_locator("count", "two") == Locator(kind="count", number="two", ordinal=2)


def test_build_locator_roman_numeral():
    assert build_locator("article", "II") == Locator(kind="article", number="II", ordinal=2)


def test_build_locator_letter():
    assert build_locator("exhibit", "C") == Locator(kind="exhibit", number="C", ordinal=3)


def test_build_locator_dotted_section_has_no_ordinal():
    locator = build_locator("section", "3.2")
    assert locator == Locator(kind="section", number="3.2", ordinal=None)


def test_build_locator_lowercases_the_kind():
    assert build_locator("Paragraph", "5") == Locator(kind="paragraph", number="5", ordinal=5)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (None, "5"),
        ("", "5"),
        ("none", "5"),
        ("None", "5"),
        ("page", "5"),
        ("paragraph", None),
        ("paragraph", ""),
        ("paragraph", "   "),
        ("resolution", "2026-01"),  # document-level kind, no marker
        ("not-a-real-kind", "5"),
    ],
)
def test_build_locator_returns_none(kind, value):
    assert build_locator(kind, value) is None
