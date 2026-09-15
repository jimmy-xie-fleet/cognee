"""Assertion nodes are grounded by type only: their claim-text name must never be matched,
renamed, or collapsed against ontology individuals."""

from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import uuid4

from cognee.domains.legal import LegalKnowledgeGraph, LegalNode
from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    construct_data_points_and_edges,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
    canonicalize_extracted_graphs,
    construct_data_points_and_edges_with_ontology,
)
from cognee.modules.ontology.models import AttachedOntologyNode
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node


class _QualifiedNode(Node):
    """A node carrying the legal-profile assertion qualifiers ``is_assertion_node`` reads."""

    statement_type: Optional[str] = None
    asserted_by: Optional[str] = None
    attributed_to: Optional[str] = None
    responds_to: Optional[str] = None
    responds_to_ref: Optional[Any] = None


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


def _make_chunk(chunk_id=None):
    chunk = MagicMock()
    chunk.id = chunk_id or uuid4()
    chunk.text = "Smith denies the payment was late."
    chunk.importance_weight = 0.5
    chunk.belongs_to_set = []
    chunk.contains = None
    chunk._produced_edge_identities = []
    chunk._provenance_edges = []
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


class _SpeakerResolver(BaseOntologyResolver):
    """Grounds the parties too: "Smith" keeps its name, "Smith Co" is renamed."""

    CLASSES = {"denial": "denial", "person": "person"}
    INDIVIDUALS = {"smith": "smith", "smith co": "smith_holdings_llc"}

    def build_lookup(self) -> None:
        return None

    def refresh_lookup(self) -> None:
        return None

    def find_closest_match(self, name: str, category: str):
        return None

    def get_subgraph(self, node_name: str, node_type: str = "individuals", directed: bool = True):
        canonical_names = self.CLASSES if node_type == "classes" else self.INDIVIDUALS
        canonical_name = canonical_names.get(node_name)
        if canonical_name is None:
            return [], [], None

        root = AttachedOntologyNode(f"https://example.test/ontology#{canonical_name}", node_type)
        return [root], [], root


def _speaker(node_id: str, name: str) -> _QualifiedNode:
    return _QualifiedNode(id=node_id, name=name, type="Person", description="the defendant")


def _denial(asserted_by: str) -> _QualifiedNode:
    return _QualifiedNode(
        id="the-denial",
        name="Payment was late",
        type="Denial",
        description="Smith denies the payment was late",
        statement_type="denial",
        asserted_by=asserted_by,
    )


def _asserted_by_edge(edges_by_identity, assertion, speaker_name):
    return edges_by_identity.get(
        EdgeIdentity(
            source_id=str(assertion.id),
            target_id=str(Entity.id_for(speaker_name)),
            relationship_name="asserted_by",
        )
    )


def _only_assertion(data_points_by_id) -> Assertion:
    assertions = [point for point in data_points_by_id.values() if isinstance(point, Assertion)]
    assert len(assertions) == 1
    return assertions[0]


def test_speaker_collapsed_onto_a_duplicate_follows_the_survivor():
    # Two mentions of one party collapse onto one node. The assertion named the mention
    # that disappeared, so its speaker has to follow the survivor instead of becoming a
    # dangling token — and the stored assertion must be the same node as without an
    # ontology, since it is the same statement by the same party.
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[_speaker("n1", "Smith"), _speaker("n9", "Smith"), _denial("n9")],
        edges=[],
    )

    data_points_by_id, edges_by_identity = construct_data_points_and_edges_with_ontology(
        [chunk],
        [graph],
        _SpeakerResolver(),
    )

    assertion = _only_assertion(data_points_by_id)
    assert assertion.asserted_by == "smith"
    assert _asserted_by_edge(edges_by_identity, assertion, "smith") is not None

    plain_points, _ = construct_data_points_and_edges(
        [_make_chunk(chunk.id)],
        [_QualifiedGraph(nodes=[_speaker("n1", "Smith"), _speaker("n9", "Smith"), _denial("n9")])],
    )
    assert assertion.id == _only_assertion(plain_points).id


def test_speaker_referenced_by_its_pre_ontology_name_still_resolves():
    # The reference is a name, and canonicalization renames the node it points at.
    chunk = _make_chunk()
    graph = _QualifiedGraph(nodes=[_speaker("n1", "Smith Co"), _denial("Smith Co")], edges=[])

    data_points_by_id, edges_by_identity = construct_data_points_and_edges_with_ontology(
        [chunk],
        [graph],
        _SpeakerResolver(),
    )

    assertion = _only_assertion(data_points_by_id)
    assert assertion.asserted_by == "smith_holdings_llc"
    assert _asserted_by_edge(edges_by_identity, assertion, "smith_holdings_llc") is not None


def test_speaker_dropped_in_strict_mode_leaves_no_speaker_and_no_edge():
    # The party has no ontology grounding at all, so strict mode drops it. Keeping its
    # id would store "n1" in an identity field and point an edge at nothing.
    chunk = _make_chunk()
    graph = _QualifiedGraph(
        nodes=[
            _QualifiedNode(id="n1", name="Nobody", type="MysteryType", description="unknown"),
            _denial("n1"),
        ],
        edges=[],
    )

    data_points_by_id, edges_by_identity = construct_data_points_and_edges_with_ontology(
        [chunk],
        [graph],
        _SpeakerResolver(),
        ontology_mode="strict",
    )

    assertion = _only_assertion(data_points_by_id)
    assert assertion.asserted_by is None
    assert "asserted_by" not in [
        edge_identity.relationship_name for edge_identity in edges_by_identity
    ]


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


def test_structured_reference_is_left_untouched_by_grounding_and_repointing():
    # ``responds_to``/``asserted_by`` are rewritten (a name to a node id, then to whichever
    # node survives collapsing duplicates); ``responds_to_ref`` sits beside them but is not
    # one of the fields ``_pin_assertion_references_to_node_ids`` /
    # ``_repoint_assertion_references_at_surviving_nodes`` ever touch.
    chunk = _make_chunk()
    structured_reference = {"document_hint": "the Complaint", "locator_kind": "paragraph"}
    graph = _QualifiedGraph(
        nodes=[
            _speaker("n1", "Smith"),
            _speaker("n9", "Smith"),
            _QualifiedNode(
                id="the-denial",
                name="Payment was late",
                type="Denial",
                description="Smith denies the payment was late",
                statement_type="denial",
                asserted_by="n9",
                responds_to_ref=structured_reference,
            ),
        ],
        edges=[],
    )

    data_points_by_id, _ = construct_data_points_and_edges_with_ontology(
        [chunk],
        [graph],
        _SpeakerResolver(),
    )

    assertion = _only_assertion(data_points_by_id)
    # The collapse-following repoint still ran (proves the fixture actually exercises it).
    assert assertion.asserted_by == "smith"
    assert assertion.responds_to_ref == structured_reference
