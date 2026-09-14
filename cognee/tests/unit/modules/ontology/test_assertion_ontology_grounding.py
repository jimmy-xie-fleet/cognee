"""Assertion nodes are grounded by type only: their claim-text name must never be matched,
renamed, or collapsed against ontology individuals."""

from typing import Optional
from unittest.mock import MagicMock

from cognee.domains.legal import LegalKnowledgeGraph, LegalNode
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    canonicalize_extracted_graphs,
    construct_data_points_and_edges_with_ontology,
)
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


class _QualifiedNode(Node):
    """A node carrying the legal-profile assertion qualifier used by ``is_assertion_node``."""

    statement_type: Optional[str] = None


class _QualifiedGraph(KnowledgeGraph):
    nodes: list[_QualifiedNode]


class _StubResolver(BaseOntologyResolver):
    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        if node_type == "classes" and node_name == "denial":
            root = AttachedOntologyNode("https://example.test/ontology#denial_class", "classes")
            return [root], [], root
        if node_type == "classes" and node_name == "records":
            # The singular class an ontology canonicalizes a plural type onto.
            root = AttachedOntologyNode("https://example.test/ontology#record", "classes")
            return [root], [], root
        if node_type == "individuals" and node_name == "payment was late":
            root = AttachedOntologyNode(
                "https://example.test/ontology#payment_was_late_canonical", "individuals"
            )
            return [root], [], root
        return [], [], None


def _make_chunk():
    chunk = MagicMock()
    chunk.importance_weight = 0.5
    chunk.belongs_to_set = []
    chunk.contains = None
    return chunk


def _assertion_node(node_id="assertion-1", name="Payment was late", node_type="Denial"):
    return _QualifiedNode(
        id=node_id,
        name=name,
        type=node_type,
        description="Smith denies the payment was late",
        statement_type="denial",
    )


def _unknown_assertion_node(node_id="assertion-2"):
    return _QualifiedNode(
        id=node_id,
        name="Something unresolved",
        type="MysteryType",
        description="An assertion whose type has no ontology class",
        statement_type="testimony",
    )


def test_assertion_node_grounded_by_class_match_marks_entity_type_ontology_valid():
    chunk = _make_chunk()
    graph = _QualifiedGraph(nodes=[_assertion_node()], edges=[])

    data_points_by_id, _ = construct_data_points_and_edges_with_ontology(
        [chunk],
        [graph],
        _StubResolver(),
    )

    assert graph.nodes[0].type == "denial_class"
    entity_type = data_points_by_id[str(EntityType.id_for("denial_class"))]
    assert entity_type.ontology_valid is True


def test_assertion_node_name_is_never_matched_renamed_or_collapsed():
    chunk = _make_chunk()
    assertion = _assertion_node()
    entity = _QualifiedNode(
        id="entity-1",
        name="Payment was late",
        type="Claim",
        description="an entity that happens to share the claim text",
    )
    graph = _QualifiedGraph(nodes=[assertion, entity], edges=[])

    canonicalize_extracted_graphs([chunk], [graph], _StubResolver())

    assertion_node = next(node for node in graph.nodes if node.id == "assertion-1")
    entity_node = next(node for node in graph.nodes if node.id == "entity-1")
    assert assertion_node.name == "Payment was late"
    assert entity_node.name == "payment_was_late_canonical"
    # Both nodes survive distinctly; the individual match never collapsed the assertion
    # onto the entity (or vice versa).
    assert {node.id for node in graph.nodes} == {"assertion-1", "entity-1"}


def test_strict_mode_keeps_class_grounded_assertion_and_drops_unknown_typed_assertion():
    chunk = _make_chunk()
    grounded = _assertion_node()
    ungrounded = _unknown_assertion_node()
    graph = _QualifiedGraph(
        nodes=[grounded, ungrounded],
        edges=[
            KGEdge(
                source_node_id="assertion-2",
                target_node_id="assertion-1",
                relationship_name="responds_to",
            )
        ],
    )

    canonicalize_extracted_graphs([chunk], [graph], _StubResolver(), strict=True)

    assert [node.id for node in graph.nodes] == ["assertion-1"]
    assert graph.edges == []


def test_canonicalizing_a_type_onto_a_statement_word_keeps_a_plain_entity():
    """Assertion-ness follows ``statement_type``, never the (canonicalized) type name.

    The ontology rewrites the type "Records" to the class "record". If the name of a
    speech act could make a node an Assertion, this entity would turn into one — chunk
    scoped, never deduplicated by name — the moment an ontology is configured.
    """
    ledger_nodes = [
        LegalNode(
            id="n1",
            name="Meridian general ledger",
            type="Records",
            description="the ledger produced in discovery",
        )
        for _ in range(2)
    ]
    chunks = [_make_chunk(), _make_chunk()]

    data_points_by_id, _ = construct_data_points_and_edges_with_ontology(
        chunks,
        [LegalKnowledgeGraph(nodes=[node], edges=[]) for node in ledger_nodes],
        _StubResolver(),
    )

    assert [node.type for node in ledger_nodes] == ["record", "record"]
    assert not [point for point in data_points_by_id.values() if isinstance(point, Assertion)]
    # One entity for both chunks: deduplicated by name, the way entities are.
    entities = [point for point in data_points_by_id.values() if isinstance(point, Entity)]
    assert len(entities) == 1
    assert type(entities[0]) is Entity
    assert entities[0].id == Entity.id_for("Meridian general ledger")


def test_get_subgraph_called_once_per_distinct_key_and_never_for_assertion_individuals():
    resolver = _StubResolver()
    resolver.get_subgraph = MagicMock(wraps=resolver.get_subgraph)
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _assertion_node(),
            _QualifiedNode(id="n2", name="Smith", type="Person", description="the defendant"),
        ],
        edges=[],
    )

    canonicalize_extracted_graphs([chunk], [graph], resolver)

    calls = resolver.get_subgraph.call_args_list
    assert len(calls) == 3
    lookup_keys = {(call.kwargs["node_type"], call.kwargs["node_name"]) for call in calls}
    assert lookup_keys == {
        ("classes", "denial"),
        ("classes", "person"),
        ("individuals", "smith"),
    }
    assert ("individuals", "payment was late") not in lookup_keys
