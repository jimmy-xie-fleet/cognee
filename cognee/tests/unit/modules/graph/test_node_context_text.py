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

from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.node_context_text import (
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


def test_assertion_context_fields_are_the_stance_properties():
    assert ASSERTION_METADATA["context_fields"] == [
        "statement_type",
        "polarity",
        "asserted_by",
        "source_quote",
        "source_quote_verified",
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
