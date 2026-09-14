"""Assertion-aware graph construction: qualified nodes become Assertion data points."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

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
    assert is_assertion_node(person) is False
    assert is_assertion_node(untyped_allegation) is True

    data_points_by_id, _ = _construct([_make_chunk()], [_QualifiedGraph(nodes=[person], edges=[])])
    assert type(data_points_by_id[str(Entity.id_for("Jones"))]) is Entity
    assert _assertions(data_points_by_id) == []

    data_points_by_id, _ = _construct(
        [_make_chunk()], [_QualifiedGraph(nodes=[untyped_allegation], edges=[])]
    )
    assertions = _assertions(data_points_by_id)
    assert len(assertions) == 1
    # The statement type falls back to the node type when the field is absent.
    assert assertions[0].statement_type == "allegation"


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
