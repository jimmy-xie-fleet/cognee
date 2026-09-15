"""Assertion-aware graph construction: qualified nodes become Assertion data points."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.domains.legal.models import Polarity
from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    attach_new_edges_to_data_points,
    construct_data_points_and_edges,
    is_assertion_node,
)
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


class _QualifiedNode(Node):
    """A node carrying the legal-profile assertion qualifiers."""

    statement_type: Optional[str] = None
    polarity: Optional[str] = None
    asserted_by: Optional[str] = None
    attributed_to: Optional[str] = None
    applicable_time: Optional[str] = None
    applies_from: Optional[str] = None
    applies_to: Optional[str] = None
    report_date: Optional[str] = None
    conditions: list[str] = []
    precision: Optional[str] = None
    scope: Optional[str] = None
    source_quote: Optional[str] = None
    responds_to: Optional[str] = None
    responds_to_ref: Optional[Any] = None
    attributed_to_ref: Optional[Any] = None


class _LegalLikeNode(_QualifiedNode):
    """Mirrors the legal profile's node, whose ``polarity`` is a typed enum member."""

    polarity: Optional[Polarity] = None


class _QualifiedGraph(KnowledgeGraph):
    nodes: list[_QualifiedNode]


def _make_chunk(text="Jones says the payment was late.") -> MagicMock:
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.text = text
    chunk.contains = None
    chunk.belongs_to_set = None
    chunk.importance_weight = 0.5
    chunk._produced_edge_identities = []
    chunk._provenance_edges = []
    return chunk


def _construct(data_chunks, extracted_graphs):
    data_points_by_id, edges_by_identity = construct_data_points_and_edges(
        data_chunks,
        extracted_graphs,
    )
    attach_new_edges_to_data_points(data_points_by_id, edges_by_identity, set())
    return data_points_by_id, edges_by_identity


def _assertions(data_points_by_id) -> list[Assertion]:
    return [
        data_point for data_point in data_points_by_id.values() if isinstance(data_point, Assertion)
    ]


def _relationship_names(edges_by_identity) -> list[str]:
    return [edge_identity.relationship_name for edge_identity in edges_by_identity]


def _person(node_id: str, name: str, description: str) -> _QualifiedNode:
    return _QualifiedNode(id=node_id, name=name, type="Person", description=description)


def _dispute_graph() -> _QualifiedGraph:
    return _QualifiedGraph(
        nodes=[
            _person("n1", "Jones", "the plaintiff"),
            _person("n2", "Smith", "the defendant"),
            _QualifiedNode(
                id="n5", name="Contract", type="Document", description="the supply contract"
            ),
            _QualifiedNode(
                id="n3",
                name="Payment was late",
                type="Allegation",
                description="Jones alleges the payment was late",
                statement_type="allegation",
                polarity="positive",
                asserted_by="n1",
                source_quote="the payment was late",
            ),
            _QualifiedNode(
                id="n4",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                polarity="negative",
                asserted_by="n2",
                responds_to="n3",
            ),
        ],
        edges=[KGEdge(source_node_id="n3", target_node_id="n5", relationship_name="about")],
    )


def test_opposing_statements_become_two_assertions_with_derived_edges():
    chunk = _make_chunk("Jones alleges the payment was late. Smith denies it.")
    data_points_by_id, edges_by_identity = _construct([chunk], [_dispute_graph()])

    assertions = _assertions(data_points_by_id)
    assert len(assertions) == 2
    assert len({assertion.id for assertion in assertions}) == 2

    by_statement_type = {assertion.statement_type: assertion for assertion in assertions}
    assert set(by_statement_type) == {"allegation", "denial"}
    allegation = by_statement_type["allegation"]
    denial = by_statement_type["denial"]

    assert allegation.name == denial.name == "payment was late"
    assert allegation.polarity == "positive"
    assert denial.polarity == "negative"
    assert allegation.is_a.name == "allegation"
    assert denial.is_a.name == "denial"
    assert allegation.asserted_by == "jones"
    assert denial.asserted_by == "smith"
    assert allegation.source_quote_verified is True

    # Derived and explicit edges reached the data points themselves.
    relations = {
        (data_point.id, edge.relationship_type, target.id)
        for data_point in data_points_by_id.values()
        for edge, target in data_point.relations
    }
    assert (allegation.id, "asserted_by", Entity.id_for("Jones")) in relations
    assert (denial.id, "asserted_by", Entity.id_for("Smith")) in relations
    assert (allegation.id, "about", Entity.id_for("Contract")) in relations
    assert (denial.id, "responds_to", allegation.id) in relations

    # Derived edges take part in ownership and evidence bookkeeping.
    derived_key = (str(denial.id), str(allegation.id), "responds_to")
    assert derived_key in chunk._produced_edge_identities
    assert derived_key in [
        (source_id, target_id, relationship_name)
        for source_id, target_id, relationship_name, _ in chunk._provenance_edges
    ]
    assert set(_relationship_names(edges_by_identity)) == {
        "about",
        "asserted_by",
        "responds_to",
    }


def test_repeated_statement_in_one_chunk_gets_distinct_occurrences():
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Jones", "the plaintiff"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Allegation",
                description="Alleged in the complaint",
                statement_type="allegation",
                asserted_by="n1",
            ),
            _QualifiedNode(
                id="n3",
                name="Payment was late",
                type="Allegation",
                description="Repeated at the hearing",
                statement_type="allegation",
                asserted_by="n1",
            ),
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    assertions = _assertions(data_points_by_id)
    assert len(assertions) == 2
    assert len({assertion.id for assertion in assertions}) == 2
    by_description = {assertion.description: assertion for assertion in assertions}
    assert by_description["Alleged in the complaint"].occurrence == 1
    assert by_description["Repeated at the hearing"].occurrence == 2


def _twin_allegations(first_name: str, second_name: str, first_speaker, second_speaker):
    return _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id="n1",
                name=first_name,
                type="Allegation",
                description="Alleged in the complaint",
                statement_type="allegation",
                asserted_by=first_speaker,
            ),
            _QualifiedNode(
                id="n2",
                name=second_name,
                type="Allegation",
                description="Repeated at the hearing",
                statement_type="allegation",
                asserted_by=second_speaker,
            ),
        ],
        edges=[],
    )


@pytest.mark.parametrize(
    "graph",
    [
        # Unresolved speakers the identity normalizer folds together.
        _twin_allegations("Payment was late", "Payment was late", "The Company", "the company"),
        # Names that differ only by a separator the identity normalizer rewrites.
        _twin_allegations("Payment was late", "Payment_was_late", None, None),
    ],
    ids=["speaker_case", "name_separator"],
)
def test_occurrences_are_grouped_the_way_identity_is_derived(graph):
    # Grouping on anything looser than the identity normalization would put these two in
    # separate groups, hand both occurrence 1, and collapse them onto a single id.
    chunk = _make_chunk()
    data_points_by_id, _ = _construct([chunk], [graph])

    assertions = _assertions(data_points_by_id)
    assert len(assertions) == 2
    assert len({assertion.id for assertion in assertions}) == 2
    assert {assertion.occurrence for assertion in assertions} == {1, 2}
    # Nothing the chunk links to was dropped from the stored data points.
    assert {str(entity.id) for _, entity in chunk.contains} <= set(data_points_by_id)


def _smith_denial_graph() -> _QualifiedGraph:
    return _QualifiedGraph(
        nodes=[
            _person("n1", "Smith", "the defendant"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                polarity="negative",
                asserted_by="n1",
            ),
        ],
        edges=[],
    )


def test_derived_asserted_by_edge_text_states_the_speakers_stance():
    # The edge text is what TRIPLET_COMPLETION embeds and what a completion context shows.
    # Left empty, ensure_default_edge_properties synthesizes it from the assertion's
    # affirmative name — "payment was late asserted by smith" — which reads as the
    # opposite of the denial it came from.
    chunk = _make_chunk("Smith denies the payment was late.")
    data_points_by_id, edges_by_identity = _construct([chunk], [_smith_denial_graph()])

    assertion = _assertions(data_points_by_id)[0]
    edge = edges_by_identity[
        EdgeIdentity(
            source_id=str(assertion.id),
            target_id=str(Entity.id_for("Smith")),
            relationship_name="asserted_by",
        )
    ]

    assert "denies" in edge.edge_text
    assert "Smith denies the payment was late" in edge.edge_text
    assert edge.edge_text.startswith("Smith denies that Payment was late.")

    stored = ensure_default_edge_properties(
        [
            (
                str(assertion.id),
                str(Entity.id_for("Smith")),
                "asserted_by",
                {"edge_text": edge.edge_text},
            )
        ],
        list(data_points_by_id.values()),
    )
    stored_text = stored[0][3]["edge_text"]
    # Nothing was synthesized over it, and the bare affirmative proposition is not the text.
    assert stored_text == edge.edge_text
    assert stored_text != f"{assertion.name} asserted by smith."


def test_derived_asserted_by_edge_text_marks_an_unrecorded_stance_as_unrecorded():
    chunk = _make_chunk("The payment was late.")
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Smith", "the defendant"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Statement",
                description="",
                statement_type="statement",
                asserted_by="n1",
            ),
        ],
        edges=[],
    )
    data_points_by_id, edges_by_identity = _construct([chunk], [graph])

    assertion = _assertions(data_points_by_id)[0]
    edge = edges_by_identity[
        EdgeIdentity(
            source_id=str(assertion.id),
            target_id=str(Entity.id_for("Smith")),
            relationship_name="asserted_by",
        )
    ]

    # An unknown stance is neither affirmed nor denied, and an empty description adds
    # nothing rather than trailing whitespace.
    assert edge.edge_text == "Smith takes an unrecorded stance on Payment was late."


def test_derived_attribution_and_response_edges_carry_the_speech_act_and_stance():
    chunk = _make_chunk("Jones alleges the payment was late. Smith denies it.")
    data_points_by_id, edges_by_identity = _construct([chunk], [_dispute_graph()])

    by_statement_type = _by_statement_type(data_points_by_id)
    denial = by_statement_type["denial"]
    allegation = by_statement_type["allegation"]
    response_text = edges_by_identity[
        EdgeIdentity(
            source_id=str(denial.id),
            target_id=str(allegation.id),
            relationship_name="responds_to",
        )
    ].edge_text

    assert response_text == (
        "Payment was late (denial, negative stance) responds to Payment was late. "
        "Smith denies the payment was late."
    )


def test_derived_attributed_to_edge_text_names_the_original_author():
    chunk = _make_chunk("The brief reports Vance's appraisal.")
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Dolores Vance", "the appraiser"),
            _QualifiedNode(
                id="n2",
                name="The building is worth 4.2 million dollars",
                type="Record",
                description="The brief reports Vance's opinion of the value.",
                statement_type="record",
                polarity="positive",
                attributed_to="n1",
            ),
        ],
        edges=[],
    )
    data_points_by_id, edges_by_identity = _construct([chunk], [graph])

    assertion = _assertions(data_points_by_id)[0]
    edge = edges_by_identity[
        EdgeIdentity(
            source_id=str(assertion.id),
            target_id=str(Entity.id_for("Dolores Vance")),
            relationship_name="attributed_to",
        )
    ]

    assert edge.edge_text == (
        "The building is worth 4.2 million dollars (record, positive stance) is attributed "
        "to Dolores Vance. The brief reports Vance's opinion of the value."
    )


def test_same_statement_in_two_chunks_stays_two_assertions_over_one_entity():
    first_chunk = _make_chunk("Smith denies it in the answer.")
    second_chunk = _make_chunk("Smith denies it again at the hearing.")

    data_points_by_id, _ = _construct(
        [first_chunk, second_chunk],
        [_smith_denial_graph(), _smith_denial_graph()],
    )

    assertions = _assertions(data_points_by_id)
    assert len(assertions) == 2
    assert {assertion.source_chunk_id for assertion in assertions} == {
        str(first_chunk.id),
        str(second_chunk.id),
    }

    smith = data_points_by_id[str(Entity.id_for("smith"))]
    assert type(smith) is Entity
    assert smith.name == "smith"
    # One Smith entity plus the two assertions, so the entity was shared, not duplicated.
    assert len([dp for dp in data_points_by_id.values() if isinstance(dp, Entity)]) == 3


def test_an_assertion_that_is_extracted_twice_is_reused_not_overwritten():
    # Two extractions over one chunk derive the same identity, so the second must reuse the
    # stored node: overwriting it would leave the first chunk link pointing at an orphan.
    chunk = _make_chunk("Smith denies the payment was late.")
    data_points_by_id, _ = _construct(
        [chunk, chunk],
        [_smith_denial_graph(), _smith_denial_graph()],
    )

    assertions = _assertions(data_points_by_id)
    assert len(assertions) == 1
    linked_assertions = [entity for _, entity in chunk.contains if isinstance(entity, Assertion)]
    assert len(linked_assertions) == 2
    assert all(entity is assertions[0] for entity in linked_assertions)


def test_qualified_node_without_statement_type_stays_a_plain_entity():
    # ``statement_type`` alone decides. A node type that merely reads like a speech act
    # must never promote an unqualified node: ontology canonicalization rewrites types
    # ("Records" -> "record"), so a type-name fallback would flip an entity into an
    # Assertion — and back again under a different ontology.
    person = _QualifiedNode(
        id="n1", name="Jones", type="Person", description="the plaintiff", statement_type=None
    )
    untyped_allegation = _QualifiedNode(
        id="n2",
        name="Payment was late",
        type="Allegation",
        description="Jones alleges the payment was late",
        statement_type=None,
    )
    blank_record = _QualifiedNode(
        id="n3",
        name="Ledger entry",
        type="Record",
        description="a ledger entry",
        statement_type="   ",
    )
    assert is_assertion_node(person) is False
    assert is_assertion_node(untyped_allegation) is False
    assert is_assertion_node(blank_record) is False

    data_points_by_id, _ = _construct(
        [_make_chunk()],
        [_QualifiedGraph(nodes=[person, untyped_allegation, blank_record], edges=[])],
    )
    assert _assertions(data_points_by_id) == []
    assert type(data_points_by_id[str(Entity.id_for("Jones"))]) is Entity
    # Name-keyed like any other entity, not chunk-scoped like an assertion.
    assert type(data_points_by_id[str(Entity.id_for("Payment was late"))]) is Entity
    assert type(data_points_by_id[str(Entity.id_for("Ledger entry"))]) is Entity


def test_plain_knowledge_graph_is_built_exactly_as_before():
    chunk = _make_chunk()
    statement_node = Node(id="n1", name="Alice", type="Statement", description="a statement")
    person_node = Node(id="n2", name="Bob", type="Person", description="a person")
    # A plain node carries no qualifiers, so its type alone never makes it an assertion.
    assert is_assertion_node(statement_node) is False
    assert is_assertion_node(person_node) is False

    graph = KnowledgeGraph(
        nodes=[statement_node, person_node],
        edges=[KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )
    data_points_by_id, edges_by_identity = _construct([chunk], [graph])

    assert set(data_points_by_id) == {
        str(EntityType.id_for("Statement")),
        str(EntityType.id_for("Person")),
        str(Entity.id_for("Alice")),
        str(Entity.id_for("Bob")),
    }
    assert type(data_points_by_id[str(Entity.id_for("Alice"))]) is Entity
    assert type(data_points_by_id[str(Entity.id_for("Bob"))]) is Entity
    assert set(edges_by_identity) == {
        EdgeIdentity(
            source_id=str(Entity.id_for("Alice")),
            target_id=str(Entity.id_for("Bob")),
            relationship_name="knows",
        )
    }
    assert chunk._produced_edge_identities == [
        (str(Entity.id_for("Alice")), str(Entity.id_for("Bob")), "knows")
    ]


def _contains_edge_text(chunk, data_point) -> Optional[str]:
    texts = [edge.edge_text for edge, target in chunk.contains if target is data_point]
    assert len(texts) == 1, f"expected one chunk link for {data_point.name!r}, found {len(texts)}"
    return texts[0]


def test_chunk_link_for_an_assertion_states_the_speakers_stance():
    # The chunk link is embedded and shown like any other edge. "Document chunk mentions
    # payment was late" states the denied proposition as a fact of the document.
    chunk = _make_chunk("Smith denies the payment was late.")
    data_points_by_id, _ = _construct([chunk], [_smith_denial_graph()])

    assertion = _assertions(data_points_by_id)[0]
    assert _contains_edge_text(chunk, assertion) == (
        "Document chunk records: Smith denies that Payment was late. "
        "Smith denies the payment was late."
    )
    # Plain entities keep the wording they had.
    smith = data_points_by_id[str(Entity.id_for("Smith"))]
    assert _contains_edge_text(chunk, smith) == "Document chunk mentions smith: the defendant"


def test_chunk_link_without_a_speaker_states_the_speech_act_and_stance():
    chunk = _make_chunk("The payment was late is denied.")
    graph = _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id="n1",
                name="Payment was late",
                type="Denial",
                description="The answer denies the payment was late",
                statement_type="denial",
                polarity="negative",
            )
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    assertion = _assertions(data_points_by_id)[0]
    assert _contains_edge_text(chunk, assertion) == (
        "Document chunk records a denial with negative stance: Payment was late. "
        "The answer denies the payment was late."
    )


def test_description_less_explicit_edge_adopts_the_derived_stance_text():
    # First-wins deduplication used to hand the win to the explicit edge and its empty
    # description, so the stance never reached storage at all.
    chunk = _make_chunk("Smith denies the payment was late.")
    graph = _smith_denial_graph()
    graph.edges = [
        KGEdge(source_node_id="n2", target_node_id="n1", relationship_name="asserted_by")
    ]
    data_points_by_id, edges_by_identity = _construct([chunk], [graph])

    assert _relationship_names(edges_by_identity).count("asserted_by") == 1
    assertion = _assertions(data_points_by_id)[0]
    edge_text = edges_by_identity[
        EdgeIdentity(
            source_id=str(assertion.id),
            target_id=str(Entity.id_for("Smith")),
            relationship_name="asserted_by",
        )
    ].edge_text
    assert "denies that" in edge_text
    assert edge_text.startswith("Smith denies that Payment was late.")

    # The derived duplicate is gone, so the relationship is recorded once.
    produced_key = (str(assertion.id), str(Entity.id_for("Smith")), "asserted_by")
    assert chunk._produced_edge_identities.count(produced_key) == 1
    assert [
        (source_id, target_id, relationship_name)
        for source_id, target_id, relationship_name, _ in chunk._provenance_edges
    ].count(produced_key) == 1


def test_explicit_llm_edge_and_derived_edge_collapse_into_one():
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Jones", "the plaintiff"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Allegation",
                description="Jones alleges the payment was late",
                statement_type="allegation",
                asserted_by="n1",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="n2",
                target_node_id="n1",
                relationship_name="asserted_by",
                description="Jones asserted that the payment was late.",
            )
        ],
    )
    data_points_by_id, edges_by_identity = _construct([chunk], [graph])

    assert _relationship_names(edges_by_identity).count("asserted_by") == 1
    assertion = _assertions(data_points_by_id)[0]
    edge_identity = EdgeIdentity(
        source_id=str(assertion.id),
        target_id=str(Entity.id_for("Jones")),
        relationship_name="asserted_by",
    )
    # The explicit LLM edge wins over the derived one, keeping its text.
    assert edges_by_identity[edge_identity].edge_text == (
        "Jones asserted that the payment was late."
    )


@pytest.mark.parametrize(
    "source_quote,expected",
    [
        ("the payment was late", True),
        ("the payment was early", False),
    ],
)
def test_source_quote_is_verified_against_chunk_text(source_quote, expected):
    chunk = _make_chunk("Jones alleges the payment was late.")
    graph = _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id="n1",
                name="Payment was late",
                type="Allegation",
                description="Jones alleges the payment was late",
                statement_type="allegation",
                source_quote=source_quote,
            )
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    assertion = _assertions(data_points_by_id)[0]
    assert assertion.source_quote == source_quote
    assert assertion.source_quote_verified is expected


@pytest.mark.asyncio
async def test_assertion_survives_the_storage_walk_as_its_own_node_type():
    document = TextDocument(
        id=uuid4(),
        name="complaint",
        raw_data_location="complaint.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )
    chunk = DocumentChunk(
        id=uuid4(),
        text="Jones alleges the payment was late.",
        chunk_size=6,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=[],
    )
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Jones", "the plaintiff"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Allegation",
                description="Jones alleges the payment was late",
                statement_type="allegation",
                polarity="positive",
                asserted_by="n1",
                conditions=["if the invoice was received"],
                source_quote="the payment was late",
            ),
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])
    assertion = _assertions(data_points_by_id)[0]

    nodes, _edges = await get_graph_from_model(chunk)
    node_properties = {str(node.id): node.model_dump() for node in nodes}

    stored = node_properties[str(assertion.id)]
    assert stored["type"] == "Assertion"
    assert stored["statement_type"] == "allegation"
    assert stored["polarity"] == "positive"
    assert stored["conditions"] == ["if the invoice was received"]
    assert stored["source_chunk_id"] == str(chunk.id)
    assert stored["valid_to"] is None


def test_unresolved_speaker_is_kept_as_text_without_an_edge():
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Jones", "the plaintiff"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Allegation",
                description="An unnamed party alleges the payment was late",
                statement_type="allegation",
                asserted_by="Unknown Party",
            ),
        ],
        edges=[],
    )
    data_points_by_id, edges_by_identity = _construct([chunk], [graph])

    assertion = _assertions(data_points_by_id)[0]
    assert assertion.asserted_by == "Unknown Party"
    assert "asserted_by" not in _relationship_names(edges_by_identity)


def _answer_graph(responds_to: str) -> _QualifiedGraph:
    return _QualifiedGraph(
        nodes=[
            _person("n1", "Jones", "the plaintiff"),
            _person("n2", "Smith", "the defendant"),
            _QualifiedNode(
                id="n3",
                name="Payment was late",
                type="Allegation",
                description="Jones alleges the payment was late",
                statement_type="allegation",
                polarity="positive",
                asserted_by="n1",
            ),
            _QualifiedNode(
                id="n4",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                polarity="negative",
                asserted_by="n2",
                responds_to=responds_to,
            ),
        ],
        edges=[],
    )


def _by_statement_type(data_points_by_id) -> dict[str, Assertion]:
    return {assertion.statement_type: assertion for assertion in _assertions(data_points_by_id)}


def test_reference_to_another_assertion_persists_that_assertions_id():
    # "n3" is a token of one LLM response and means nothing once stored, so the graph must
    # keep the id of the node it named instead.
    chunk = _make_chunk()
    data_points_by_id, _ = _construct([chunk], [_answer_graph("n3")])

    by_statement_type = _by_statement_type(data_points_by_id)
    assert by_statement_type["denial"].responds_to == str(by_statement_type["allegation"].id)


def test_unresolved_reference_locator_is_kept_as_written():
    chunk = _make_chunk()
    data_points_by_id, _ = _construct([chunk], [_answer_graph("Complaint ¶17")])

    assert _by_statement_type(data_points_by_id)["denial"].responds_to == "Complaint ¶17"


def _denial_answering_graph(allegation_id: str) -> _QualifiedGraph:
    """A denial whose ``asserted_by`` names the allegation instead of a party."""
    return _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id=allegation_id,
                name="Payment was late",
                type="Allegation",
                description="Jones alleges the payment was late",
                statement_type="allegation",
                polarity="positive",
            ),
            _QualifiedNode(
                id="the-denial",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                polarity="negative",
                asserted_by=allegation_id,
                responds_to=allegation_id,
            ),
        ],
        edges=[],
    )


def test_speaker_naming_another_assertion_is_stored_as_no_speaker():
    # A speaker is a party. Storing "a1" would put an LLM token in an identity field and
    # have the derived text read "Payment was late denies that Payment was late".
    chunk = _make_chunk()
    data_points_by_id, edges_by_identity = _construct([chunk], [_denial_answering_graph("a1")])

    denial = _by_statement_type(data_points_by_id)["denial"]
    assert denial.asserted_by is None
    assert "asserted_by" not in _relationship_names(edges_by_identity)
    # The cross-reference is still recorded, as the target assertion's stored id.
    assert denial.responds_to == str(_by_statement_type(data_points_by_id)["allegation"].id)


def test_assertion_id_does_not_depend_on_the_llms_local_numbering():
    # Two extractions of one passage may number their nodes differently. The stored
    # denial is the same statement either way, so it must be the same node.
    first_chunk = _make_chunk()
    second_chunk = _make_chunk()
    second_chunk.id = first_chunk.id

    first_points, _ = _construct([first_chunk], [_denial_answering_graph("a1")])
    second_points, _ = _construct([second_chunk], [_denial_answering_graph("n9")])

    assert (
        _by_statement_type(first_points)["denial"].id
        == _by_statement_type(second_points)["denial"].id
    )


def test_cross_reference_to_a_plain_entity_stores_that_entitys_name():
    # responds_to kept the raw token while asserted_by and attributed_to stored the
    # entity's name, so one assertion could name one party three different ways.
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _person("n1", "Smith", "the defendant"),
            _QualifiedNode(
                id="n2",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                polarity="negative",
                asserted_by="n1",
                attributed_to="n1",
                responds_to="n1",
            ),
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    denial = _assertions(data_points_by_id)[0]
    assert denial.asserted_by == denial.attributed_to == denial.responds_to == "smith"


def test_attribution_to_another_assertion_persists_that_assertions_id():
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id="n1",
                name="The building is worth 4.2 million dollars",
                type="Opinion",
                description="Vance's appraisal opinion",
                statement_type="opinion",
            ),
            _QualifiedNode(
                id="n2",
                name="The building is worth 4.2 million dollars",
                type="Record",
                description="The brief reports Vance's opinion",
                statement_type="record",
                attributed_to="n1",
            ),
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    by_statement_type = _by_statement_type(data_points_by_id)
    assert by_statement_type["record"].attributed_to == str(by_statement_type["opinion"].id)


@pytest.mark.parametrize(
    "node_class,polarity,expected",
    [
        (_QualifiedNode, None, "unknown"),
        (_QualifiedNode, "negative", "negative"),
        (_LegalLikeNode, None, "unknown"),
        (_LegalLikeNode, Polarity.NEGATIVE, "negative"),
        (_LegalLikeNode, Polarity.POSITIVE, "positive"),
    ],
    ids=["omitted", "text_negative", "enum_omitted", "enum_negative", "enum_positive"],
)
def test_missing_polarity_is_stored_as_unknown_never_as_positive(node_class, polarity, expected):
    # The extraction schema leaves polarity optional, so a schema-valid extraction can omit
    # it. Storing "positive" for a stance nobody recorded invents the speaker's agreement.
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            node_class(
                id="n1",
                name="Payment was late",
                type="Allegation",
                description="A claim the extraction gave no stance for",
                statement_type="allegation",
                polarity=polarity,
            )
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    assert _assertions(data_points_by_id)[0].polarity == expected


def test_statement_type_text_is_normalized_the_way_identity_is():
    # A model declaring `statement_type: str` may hand back the capitalized word; identity
    # normalizes it, so the stored property must be normalized too.
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id="n1",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="Denial",
            )
        ],
        edges=[],
    )
    data_points_by_id, _ = _construct([chunk], [graph])

    assertion = _assertions(data_points_by_id)[0]
    assert assertion.statement_type == "denial"
    assert assertion.id == Assertion.id_for("payment was late", str(chunk.id), "denial", None, 1)


# ---------------------------------------------------------------------------
# Structured references (``responds_to_ref`` / ``attributed_to_ref``)
# ---------------------------------------------------------------------------


class _StubLocatorKind(str, Enum):
    PARAGRAPH = "paragraph"
    NONE = "none"


class _StubReferenceBasis(str, Enum):
    POSITIONAL = "positional"
    DESCRIBED = "described"


class _StubReference(BaseModel):
    """Mirrors the shape of ``cognee.domains.legal.models.LegalReference`` for this test.

    Core cannot import ``cognee.domains``, so ``_reference_payload`` duck-types on
    ``BaseModel``/``dict`` instead of importing the real model. This local stand-in is what
    proves that duck-typing without reaching into the legal profile.
    """

    document_hint: str = ""
    locator_kind: _StubLocatorKind = _StubLocatorKind.NONE
    locator_value: Optional[str] = None
    date: Optional[str] = None
    basis: _StubReferenceBasis = _StubReferenceBasis.DESCRIBED


def _reference_graph(responds_to_ref: Any, attributed_to_ref: Any = None) -> _QualifiedGraph:
    return _QualifiedGraph(
        nodes=[
            _QualifiedNode(
                id="n1",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                responds_to_ref=responds_to_ref,
                attributed_to_ref=attributed_to_ref,
            )
        ],
        edges=[],
    )


def test_structured_reference_lands_as_a_plain_dict_with_enum_values_as_strings():
    chunk = _make_chunk()
    reference = _StubReference(
        document_hint="the Complaint",
        locator_kind=_StubLocatorKind.PARAGRAPH,
        locator_value="17",
        basis=_StubReferenceBasis.POSITIONAL,
    )
    data_points_by_id, _ = _construct([chunk], [_reference_graph(reference)])

    denial = _assertions(data_points_by_id)[0]
    assert denial.responds_to_ref == {
        "document_hint": "the Complaint",
        "locator_kind": "paragraph",
        "locator_value": "17",
        "basis": "positional",
    }
    assert all(isinstance(value, str) for value in denial.responds_to_ref.values())
    # A structured reference derives no edge and resolves nothing on its own.
    assert denial.responds_to is None


def test_a_dict_valued_structured_reference_passes_through_unchanged():
    chunk = _make_chunk()
    raw = {
        "document_hint": "the Fester Report",
        "locator_kind": "section",
        "locator_value": "4.2",
    }
    data_points_by_id, _ = _construct([chunk], [_reference_graph(raw)])

    assert _assertions(data_points_by_id)[0].responds_to_ref == raw


def test_blank_and_none_fields_are_dropped_from_the_structured_reference():
    chunk = _make_chunk()
    raw = {
        "document_hint": "",
        "locator_kind": "none",
        "locator_value": None,
        "date": "2026-06-10",
        "basis": "described",
    }
    data_points_by_id, _ = _construct([chunk], [_reference_graph(raw)])

    assert _assertions(data_points_by_id)[0].responds_to_ref == {
        "date": "2026-06-10",
        "basis": "described",
    }


def test_an_all_blank_structured_reference_stores_as_none():
    chunk = _make_chunk()
    raw = {"document_hint": "", "locator_kind": "none", "locator_value": None}
    data_points_by_id, _ = _construct([chunk], [_reference_graph(raw)])

    assert _assertions(data_points_by_id)[0].responds_to_ref is None


def test_no_structured_reference_stores_as_none():
    chunk = _make_chunk()
    data_points_by_id, _ = _construct([chunk], [_reference_graph(None)])

    assert _assertions(data_points_by_id)[0].responds_to_ref is None


def test_structured_reference_never_reaches_the_string_reference_resolver():
    # A dict/BaseModel has no ``.strip()``. If a `_ref` value ever reached
    # ``_resolve_reference``/``_strip_nonblank_text`` (which call ``.strip()`` on it), this
    # would raise instead of quietly storing ``None``.
    class _ExplodingIfTreatedAsAReferenceString:
        def strip(self):
            raise AssertionError("a structured reference must never reach _resolve_reference")

    chunk = _make_chunk()
    data_points_by_id, _ = _construct(
        [chunk], [_reference_graph(_ExplodingIfTreatedAsAReferenceString())]
    )

    # Not a BaseModel/dict, so it stores as None -- and, more importantly, no exception.
    assert _assertions(data_points_by_id)[0].responds_to_ref is None


def test_attributed_to_ref_is_stored_independently_of_responds_to_ref():
    chunk = _make_chunk()
    data_points_by_id, _ = _construct(
        [chunk],
        [_reference_graph(None, attributed_to_ref={"document_hint": "my deposition"})],
    )

    denial = _assertions(data_points_by_id)[0]
    assert denial.responds_to_ref is None
    assert denial.attributed_to_ref == {"document_hint": "my deposition"}


def test_assertion_reference_fields_tuple_never_gains_a_ref_name():
    from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
        _ASSERTION_REFERENCE_FIELDS,
    )

    assert not any(
        field_name.endswith("_ref") or relationship_name.endswith("_ref")
        for field_name, relationship_name in _ASSERTION_REFERENCE_FIELDS
    )
