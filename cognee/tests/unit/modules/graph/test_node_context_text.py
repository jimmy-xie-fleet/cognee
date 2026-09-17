"""
Unit Tests: node_context_text / node_context_label / context_fields_for_datapoints

Pins two things.

1. A plain node renders exactly as it did before the shared renderer existed: title from
   ``text`` when there is text, otherwise ``name``, body from ``description``. Every
   non-assertion graph in the suite depends on that being byte-identical.
2. An assertion carries its speech act, speaker and stance into the title, the body and
   the short label. ``Assertion.name`` is the proposition phrased AFFIRMATIVELY, so a
   denial rendered as ``name`` alone reads to the model as the fact it denies.
"""

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.node_context_text import (
    UNNAMED_NODE,
    _create_title_from_text,
    context_fields_for_datapoints,
    is_assertion_props,
    node_context_label,
    node_context_text,
)
from cognee.modules.graph.utils.reference_resolution import derived_edge_text
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text

# Pydantic keeps a declared field off the class, so the metadata contract lives on the
# field default -- which is also what ``context_fields_for_datapoints`` reads.
ASSERTION_METADATA = Assertion.model_fields["metadata"].default

DENIAL = {
    "name": "Adams breached the lease",
    "statement_type": "denial",
    "polarity": "negative",
    "asserted_by": "Defendants",
    "source_quote": "Defendants deny each and every allegation of paragraph 17.",
    "source_quote_verified": True,
    "description": "Answer paragraph 17 denies the breach allegation.",
}


# ---------------------------------------------------------------------------------------
# Plain nodes: today's behaviour, unchanged
# ---------------------------------------------------------------------------------------


def test_plain_node_title_is_name_and_body_is_description():
    title, body = node_context_text({"name": "Alice", "description": "Alice works at Acme."})

    assert (title, body) == ("Alice", "Alice works at Acme.")


def test_plain_node_body_falls_back_to_the_name():
    assert node_context_text({"name": "Alice"}) == ("Alice", "Alice")


def test_plain_node_body_falls_back_to_the_name_when_description_is_none():
    # The projection fills every requested property, so a missing description arrives as None.
    assert node_context_text({"name": "Alice", "description": None}) == ("Alice", "Alice")


def test_nameless_node_renders_unnamed_node():
    assert node_context_text({}) == ("Unnamed Node", "Unnamed Node")


def test_a_projected_description_of_none_no_longer_renders_the_literal_none():
    """The one intended divergence from main's plain rendering, with both strings named.

    ``resolve_edges_to_text`` on main read ``attributes.get("description", name)``. The graph
    projection is a whitelist that fills *every* key it asked for, so a node type with no
    ``description`` field at all -- a ``Document``, a ``NodeSet`` -- arrived carrying
    ``description: None`` and the ``.get`` default never fired. The body the prompt received
    was the four characters ``None``. Rendering the name instead is an improvement, not a
    regression, but it is a divergence and this is where it is recorded.
    """
    projected = {"name": "Adams v. Great Plains", "description": None}

    # Main's formula, on main's input.
    old_title = projected.get("name", UNNAMED_NODE)
    old_body = projected.get("description", old_title)
    assert old_body is None
    assert f"Node: {old_title}\n{old_body}" == "Node: Adams v. Great Plains\nNone"

    # What the shared renderer puts there instead.
    assert node_context_text(projected) == ("Adams v. Great Plains", "Adams v. Great Plains")


def test_a_projected_nameless_node_no_longer_titles_itself_none():
    """The same cause on the title: ``name`` is present and ``None``, so no default fired."""
    projected = {"name": None, "description": None}

    assert projected.get("name", UNNAMED_NODE) is None  # main's title
    assert node_context_text(projected) == (UNNAMED_NODE, UNNAMED_NODE)


def test_text_node_title_is_the_generated_title():
    text = "Acme acquired Initech and Acme hired Alice"

    title, body = node_context_text({"text": text})

    assert title == "Acme acquired Initech and Acme hired Alice... [acme, acquired, initech]"
    assert title == _create_title_from_text(text)
    assert body == text


def test_text_wins_over_name_for_a_plain_node():
    title, body = node_context_text({"name": "Chunk", "text": "Some passage text"})

    assert title == _create_title_from_text("Some passage text")
    assert body == "Some passage text"


def test_non_string_name_still_renders():
    assert node_context_text({"name": 17}) == ("17", "17")


# ---------------------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------------------


def test_blank_statement_type_is_not_an_assertion():
    assert is_assertion_props({"name": "Alice", "statement_type": "  "}) is False
    assert node_context_text({"name": "Alice", "statement_type": None}) == ("Alice", "Alice")


def test_an_assertion_typed_node_with_a_statement_type_is_an_assertion():
    assert is_assertion_props({"type": "Assertion", **DENIAL}) is True


def test_a_foreign_datapoint_with_a_statement_type_field_is_not_an_assertion():
    """A third-party ``DataPoint`` is free to declare a field it happens to call this.

    Duck-typing alone would render it stance-first and send it through pair expansion. The
    projection stores the DataPoint *class name* in ``type`` (``get_graph_from_model``), and
    ontology canonicalization does not touch that -- it rewrites an entity's ``is_a``, not
    its Python class -- so when a type is present it can settle the question.
    """
    assert is_assertion_props({"type": "ReleaseNote", "statement_type": "draft"}) is False
    assert node_context_text(
        {"type": "ReleaseNote", "name": "v2 notes", "statement_type": "draft"}
    ) == ("v2 notes", "v2 notes")


def test_a_registered_assertion_subclass_is_still_an_assertion():
    """``type`` holds the concrete class name, so a subclass has to be admitted by name."""

    class CourtFinding(Assertion):
        pass

    assert is_assertion_props({"type": "CourtFinding", "statement_type": "finding"}) is True


def test_a_blank_or_absent_type_falls_back_to_the_statement_type():
    """Not every caller renders a projected node -- the hybrid lane builds props from a payload."""
    assert is_assertion_props({"statement_type": "denial"}) is True
    assert is_assertion_props({"type": None, "statement_type": "denial"}) is True
    assert is_assertion_props({"type": "   ", "statement_type": "denial"}) is True


def test_assertion_title_carries_statement_type_speaker_and_stance():
    title, _body = node_context_text(DENIAL)

    assert title == "[denial by Defendants; stance: negative] Adams breached the lease"


def test_negative_assertion_body_states_the_denial():
    _title, body = node_context_text(DENIAL)

    assert body == (
        "Defendants denies that Adams breached the lease.\n"
        'Quote: "Defendants deny each and every allegation of paragraph 17." (verified)\n'
        "Answer paragraph 17 denies the breach allegation."
    )


def test_positive_assertion_body_states_the_affirmation():
    _title, body = node_context_text(
        {
            "name": "the lease was signed on June 1",
            "statement_type": "admission",
            "polarity": "positive",
            "asserted_by": "Great Plains",
        }
    )

    assert body == "Great Plains affirms that the lease was signed on June 1."


def test_unknown_polarity_uses_the_unrecorded_stance_verb():
    _title, body = node_context_text(
        {"name": "the roof leaked", "statement_type": "testimony", "polarity": "unknown"}
    )

    assert body == "an unnamed party takes an unrecorded stance on the roof leaked."


def test_missing_polarity_property_is_reported_as_unknown():
    title, body = node_context_text(
        {"name": "the roof leaked", "statement_type": "testimony", "asserted_by": "Mr. Adams"}
    )

    assert title == "[testimony by Mr. Adams; stance: unknown] the roof leaked"
    assert body == "Mr. Adams takes an unrecorded stance on the roof leaked."


def test_missing_speaker_renders_the_unnamed_party():
    title, _body = node_context_text(
        {"name": "the roof leaked", "statement_type": "allegation", "polarity": "positive"}
    )

    assert title == "[allegation by an unnamed party; stance: positive] the roof leaked"


def test_speakerless_stance_sentence_matches_the_derived_edge_text():
    """One wording for a missing speaker, in the prompt line and in the embedded edge text."""
    props = {"name": "the roof leaked", "statement_type": "allegation", "polarity": "positive"}

    _title, body = node_context_text(props)
    edge_text = derived_edge_text(
        props["name"], props["statement_type"], props["polarity"], "asserted_by", None
    )

    assert body.splitlines()[0] == edge_text == "an unnamed party affirms that the roof leaked."


def test_stance_sentence_uses_the_same_verb_table_as_the_derived_edge_text():
    _title, body = node_context_text(DENIAL)
    edge_text = derived_edge_text(
        DENIAL["name"], DENIAL["statement_type"], DENIAL["polarity"], "asserted_by", "Defendants"
    )

    assert body.splitlines()[0] == edge_text == ("Defendants denies that Adams breached the lease.")


def test_unverified_quote_is_not_marked_verified():
    _title, body = node_context_text(
        {
            "name": "the roof leaked",
            "statement_type": "testimony",
            "polarity": "positive",
            "asserted_by": "Mr. Adams",
            "source_quote": "The roof leaked all spring.",
            "source_quote_verified": False,
        }
    )

    assert body == ('Mr. Adams affirms that the roof leaked.\nQuote: "The roof leaked all spring."')


def test_a_stringified_false_does_not_mark_the_quote_verified():
    """``source_quote_verified`` is a bool, but a store can hand it back as text.

    Neo4j and the Postgres demo adapter round JSON properties through strings, and the
    non-empty string ``"false"`` is truthy in Python -- so a plain truth test stamps
    ``(verified)`` onto the one quote the extraction explicitly marked unverified.
    """
    props = {
        "name": "the roof leaked",
        "statement_type": "testimony",
        "polarity": "positive",
        "asserted_by": "Mr. Adams",
        "source_quote": "The roof leaked all spring.",
    }

    _title, body = node_context_text({**props, "source_quote_verified": "false"})
    assert body.endswith('Quote: "The roof leaked all spring."')

    _title, verified = node_context_text({**props, "source_quote_verified": "True"})
    assert verified.endswith('Quote: "The roof leaked all spring." (verified)')


def test_a_non_boolean_verification_flag_is_not_a_verification():
    props = {"name": "x", "statement_type": "testimony", "source_quote": "q"}

    for value in (1, "yes", "1", ["true"], None):
        _title, body = node_context_text({**props, "source_quote_verified": value})
        assert "(verified)" not in body


def test_quote_line_is_omitted_when_there_is_no_quote():
    _title, body = node_context_text(
        {
            "name": "the roof leaked",
            "statement_type": "testimony",
            "polarity": "positive",
            "asserted_by": "Mr. Adams",
            "source_quote": None,
            "source_quote_verified": True,
        }
    )

    assert body == "Mr. Adams affirms that the roof leaked."
    assert "Quote" not in body


def test_the_reference_as_written_is_printed_when_present():
    """A pleading's "the allegations of paragraph 17 are true" says nothing without the
    locator; until a resolver pass turns it into an edge, the text is the pointer."""
    _title, body = node_context_text(
        {
            **DENIAL,
            "name": "the allegations of paragraph 17 of the complaint are true",
            "responds_to_text": "the Complaint ¶17",
            "attributed_to_text": "Fester Preliminary Investigation Report",
        }
    )

    lines = body.splitlines()
    assert lines[0] == (
        "Defendants denies that the allegations of paragraph 17 of the complaint are true."
    )
    assert "Responds to: the Complaint ¶17" in lines
    assert "Attributed to: Fester Preliminary Investigation Report" in lines
    # quote before locator, locator before description
    assert lines.index("Responds to: the Complaint ¶17") > lines.index(
        'Quote: "Defendants deny each and every allegation of paragraph 17." (verified)'
    )
    assert lines[-1] == DENIAL["description"]


def test_a_blank_reference_text_prints_no_locator_line():
    _title, body = node_context_text(
        {**DENIAL, "responds_to_text": "  ", "attributed_to_text": None}
    )

    assert "Responds to" not in body and "Attributed to" not in body


def test_description_equal_to_the_name_is_not_repeated():
    _title, body = node_context_text(
        {
            "name": "the roof leaked",
            "statement_type": "testimony",
            "polarity": "positive",
            "asserted_by": "Mr. Adams",
            "description": "the roof leaked",
        }
    )

    assert body == "Mr. Adams affirms that the roof leaked."


def test_nameless_assertion_falls_back_to_this_statement():
    title, body = node_context_text({"statement_type": "denial", "polarity": "negative"})

    assert title == "[denial by an unnamed party; stance: negative] Unnamed Node"
    assert body == "an unnamed party denies that this statement."


# ---------------------------------------------------------------------------------------
# Short labels
# ---------------------------------------------------------------------------------------


def test_label_of_a_plain_node_is_its_name():
    assert node_context_label({"name": "Alice", "id": "node-1"}) == "Alice"


def test_label_falls_back_to_the_id():
    assert node_context_label({"id": "node-1"}) == "node-1"


def test_label_is_empty_when_there_is_nothing_to_show():
    assert node_context_label({}) == ""


def test_label_of_a_nameless_text_node_is_its_title_not_its_id():
    """A chunk a resolver anchored a paragraph on has text and no name; the pair line
    that points at it must read like the passage, not like a UUID."""
    label = node_context_label(
        {
            "id": "9522e29a-201d-5177-9a62-2d8a48f8e734",
            "text": "Defendants deny each and every allegation of paragraph 17 of the Complaint.",
        }
    )

    assert label == "Defendants deny each and every allegation of... [defendants, deny, each]"
    assert "9522e29a" not in label


def test_label_prefers_the_name_over_the_text():
    assert node_context_label({"name": "Alice", "text": "Alice said many things."}) == "Alice"


def test_label_of_an_assertion_carries_type_and_stance():
    assert node_context_label(DENIAL) == "[denial/negative] Adams breached the lease"


def test_label_of_a_nameless_assertion_falls_back_to_the_id():
    """Like every other node: the id is the only handle a reader has left."""
    assert (
        node_context_label({"id": "denial-1", "statement_type": "denial", "polarity": "negative"})
        == "[denial/negative] denial-1"
    )


def test_label_of_a_nameless_idless_assertion_still_reports_its_stance():
    assert node_context_label({"statement_type": "denial"}) == "[denial/unknown] Unnamed Node"


def test_label_of_an_assertion_without_polarity_reports_unknown():
    assert (
        node_context_label({"name": "the roof leaked", "statement_type": "testimony"})
        == "[testimony/unknown] the roof leaked"
    )


# ---------------------------------------------------------------------------------------
# Projection contract
# ---------------------------------------------------------------------------------------


def test_context_fields_for_datapoints_includes_the_assertion_fields():
    fields = context_fields_for_datapoints()

    assert set(ASSERTION_METADATA["context_fields"]).issubset(set(fields))
    assert fields == list(dict.fromkeys(fields))  # deduplicated, deterministic order


def test_context_fields_are_cached_and_the_cache_is_clearable():
    """The walk crosses every ``DataPoint`` subclass and runs once per projected search.

    Cached, so a subclass registered after the first call is invisible until the cache is
    cleared -- which is the contract a test that declares one relies on. Each call still
    hands back a fresh list, so a caller that mutates the result cannot corrupt the cache.
    """
    first = context_fields_for_datapoints()

    assert context_fields_for_datapoints() == first
    assert context_fields_for_datapoints() is not first

    class LateDataPoint(DataPoint):
        late_field: str = ""
        metadata: dict = {"index_fields": [], "context_fields": ["late_field"]}

    try:
        assert "late_field" not in context_fields_for_datapoints()  # still the cached walk
        context_fields_for_datapoints.cache_clear()
        assert "late_field" in context_fields_for_datapoints()
    finally:
        context_fields_for_datapoints.cache_clear()


def test_assertion_context_fields_are_the_stance_properties_and_the_locators():
    assert ASSERTION_METADATA["context_fields"] == [
        "statement_type",
        "polarity",
        "asserted_by",
        "source_quote",
        "source_quote_verified",
        "responds_to_text",
        "attributed_to_text",
    ]


def test_assertion_index_and_identity_fields_are_unchanged():
    assert ASSERTION_METADATA["index_fields"] == ["name"]
    assert ASSERTION_METADATA["identity_fields"] == [
        "name",
        "source_chunk_id",
        "statement_type",
        "asserted_by",
        "occurrence",
    ]


# ---------------------------------------------------------------------------------------
# Through the graph-context renderer
# ---------------------------------------------------------------------------------------


def _edge(source_attributes: dict, target_attributes: dict, attributes: dict) -> Edge:
    source = Node(node_id="source", attributes=source_attributes)
    target = Node(node_id="target", attributes=target_attributes)
    return Edge(source, target, attributes=attributes)


@pytest.mark.asyncio
async def test_graph_context_renders_the_assertion_stance():
    edge = _edge(
        DENIAL,
        {"name": "Answer"},
        {"relationship_type": "responds_to", "edge_text": "The denial responds to the answer."},
    )

    output = await resolve_edges_to_text([edge])

    assert "Node: [denial by Defendants; stance: negative] Adams breached the lease" in output
    assert "Defendants denies that Adams breached the lease." in output
    assert (
        "[denial by Defendants; stance: negative] Adams breached the lease "
        "--[responds_to]--> Answer" in output
    )


@pytest.mark.asyncio
async def test_graph_context_edge_line_reports_the_resolution_confidence():
    edge = _edge(
        DENIAL,
        {"name": "Complaint"},
        {
            "relationship_type": "responds_to",
            "resolution_confidence": 0.95,
            "resolution_strategy": "llm_trace",
        },
    )

    output = await resolve_edges_to_text([edge])

    assert "--[responds_to]--> Complaint [confidence 0.95, llm_trace]" in output


@pytest.mark.asyncio
async def test_graph_context_edge_line_is_unchanged_without_a_resolution():
    edge = _edge({"name": "Alice"}, {"name": "Acme"}, {"relationship_type": "works_for"})

    output = await resolve_edges_to_text([edge])

    assert "Alice --[works_for]--> Acme" in output
    assert "confidence" not in output
