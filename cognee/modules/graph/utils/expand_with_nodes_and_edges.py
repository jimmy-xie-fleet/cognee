from enum import Enum
from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.models.Assertion import (
    STATEMENT_TYPE_NAMES,
    Assertion,
    verify_source_quote,
)
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node

# Assertion fields that name another extracted node, and the edge each one derives.
_ASSERTION_REFERENCE_FIELDS = (
    ("asserted_by", "asserted_by"),
    ("attributed_to", "attributed_to"),
    ("responds_to", "responds_to"),
)


def _strip_nonblank_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped_value = value.strip()
    return stripped_value or None


def is_assertion_node(extracted_node: Node) -> bool:
    """True when an extracted node carries assertion qualifiers.

    A plain ``cognee.shared.data_models.Node`` has no ``statement_type`` attribute at all, so
    a plain extraction is never treated as an assertion — not even when its type reads
    "Statement". Only an extraction model that declares the qualifier fields opts in.
    """
    return hasattr(extracted_node, "statement_type") and (
        extracted_node.statement_type is not None
        or generate_node_name(extracted_node.type) in STATEMENT_TYPE_NAMES
    )


def _enum_value(value):
    """Enum members arrive from typed extraction models; store their value."""
    return value.value if isinstance(value, Enum) else value


def _statement_type_value(extracted_node: Node) -> str:
    statement_type = _enum_value(getattr(extracted_node, "statement_type", None))
    return statement_type or generate_node_name(extracted_node.type)


def _node_id_by_reference(extracted_graph: KnowledgeGraph) -> dict[str, str]:
    """Index of the references an assertion may use to point at another node.

    Both graph-local ids and normalized names resolve, because extraction models name a
    speaker either way. Ids win over names, and the first node with a name keeps it.
    Assertions are addressable by id only: a name would merge the occurrences apart.
    """
    node_id_by_reference: dict[str, str] = {}
    for extracted_node in extracted_graph.nodes:
        if is_assertion_node(extracted_node):
            continue
        node_id_by_reference.setdefault(generate_node_name(extracted_node.name), extracted_node.id)

    for extracted_node in extracted_graph.nodes:
        node_id_by_reference[extracted_node.id] = extracted_node.id

    return node_id_by_reference


def _resolve_reference(value: Optional[str], node_id_by_reference: dict[str, str]) -> Optional[str]:
    reference = _strip_nonblank_text(value)
    if reference is None:
        return None

    if reference in node_id_by_reference:
        return node_id_by_reference[reference]
    return node_id_by_reference.get(generate_node_name(reference))


def _get_or_create_entity_type(
    extracted_type: str,
    data_chunk: DocumentChunk,
    data_points_by_id: dict[str, Entity | EntityType],
) -> EntityType:
    entity_type_id = EntityType.id_for(extracted_type)
    entity_type_key = str(entity_type_id)
    existing_data_point = data_points_by_id.get(entity_type_key)
    if isinstance(existing_data_point, EntityType):
        return existing_data_point

    normalized_type_name = generate_node_name(extracted_type)
    entity_type = EntityType(
        id=entity_type_id,
        name=normalized_type_name,
        description=normalized_type_name,
        importance_weight=data_chunk.importance_weight,
    )
    data_points_by_id[entity_type_key] = entity_type
    return entity_type


def _get_or_create_entity(
    extracted_node: Node,
    entity_id: UUID,
    entity_type: EntityType,
    data_chunk: DocumentChunk,
    data_points_by_id: dict[str, Entity | EntityType],
) -> Entity:
    entity_key = str(entity_id)
    existing_data_point = data_points_by_id.get(entity_key)
    if isinstance(existing_data_point, Entity):
        return existing_data_point

    entity = Entity(
        id=entity_id,
        name=generate_node_name(extracted_node.name),
        is_a=entity_type,
        description=extracted_node.description,
        belongs_to_set=data_chunk.belongs_to_set,
        importance_weight=data_chunk.importance_weight,
    )
    data_points_by_id[entity_key] = entity
    return entity


def _calculate_entity_ids_by_extracted_node_id(
    extracted_graph: KnowledgeGraph,
    data_chunk: DocumentChunk,
) -> dict[str, UUID]:
    """Calculate the final entity ID for every graph-local node ID.

    Multiple nodes with the same name remain distinct by receiving deterministic graph-scoped
    IDs instead of sharing one name-based ID.
    """
    extracted_node_ids: set[str] = set()
    nodes_by_name_based_id: dict[UUID, list[Node]] = {}
    for node in extracted_graph.nodes:
        if node.id in extracted_node_ids:
            raise ValueError(f"Duplicate node id in extracted graph: {node.id}")
        extracted_node_ids.add(node.id)

        if is_assertion_node(node):
            # Assertions never share an id with anything: their identity is derived from
            # the statement, its speaker, its chunk and its occurrence, not from the name.
            continue

        name_based_entity_id = Entity.id_for(node.name)
        nodes_by_name_based_id.setdefault(name_based_entity_id, []).append(node)

    entity_ids_by_extracted_node_id: dict[str, UUID] = {}
    for name_based_entity_id, same_name_nodes in nodes_by_name_based_id.items():
        if len(same_name_nodes) == 1:
            entity_ids_by_extracted_node_id[same_name_nodes[0].id] = name_based_entity_id
            continue

        ordered_nodes = sorted(
            same_name_nodes,
            key=lambda node: (
                generate_node_name(node.type),
                generate_node_name(node.description),
                generate_node_name(node.id),
            ),
        )
        for ordinal, node in enumerate(ordered_nodes, start=1):
            entity_ids_by_extracted_node_id[node.id] = Entity.id_for(
                node.name,
                data_chunk.id,
                ordinal,
            )

    return entity_ids_by_extracted_node_id


def _link_chunk_to_entity(
    data_chunk: DocumentChunk,
    extracted_node: Node,
    entity: Entity,
) -> None:
    if data_chunk.contains is None:
        data_chunk.contains = []

    entity_description = _strip_nonblank_text(extracted_node.description)
    edge_text = (
        f"Document chunk mentions {entity.name}: {entity_description}"
        if entity_description
        else None
    )
    data_chunk.contains.append(
        (
            Edge(relationship_type="contains", edge_text=edge_text),
            entity,
        )
    )


def _assertion_occurrences(
    assertion_nodes: list[Node],
    speaker_name_by_node_id: dict[str, Optional[str]],
    source_chunk_id: str,
) -> dict[str, int]:
    """Rank assertions that would otherwise be the same statement, so none is lost.

    The same speaker making the same statement twice in one chunk is two occurrences, not
    one node overwriting the other. Ranking is deterministic (description, then graph-local
    id), so re-running the same extraction yields the same ids.

    Nodes are grouped by the id they would share if they were one occurrence, exactly the
    way the same-name entity grouper keys on ``Entity.id_for``. Grouping on the raw field
    values instead would miss every pair the identity normalizer folds together — two
    speakers written "The Company" and "the company" would be ranked apart yet still land
    on one id, silently dropping one of them.
    """
    nodes_by_occurrence_key: dict[UUID, list[Node]] = {}
    for node in assertion_nodes:
        occurrence_key = Assertion.id_for(
            generate_node_name(node.name),
            source_chunk_id,
            _statement_type_value(node),
            speaker_name_by_node_id.get(node.id),
            0,  # Stands in for the occurrence this grouping is about to assign.
        )
        nodes_by_occurrence_key.setdefault(occurrence_key, []).append(node)

    occurrence_by_extracted_node_id: dict[str, int] = {}
    for same_key_nodes in nodes_by_occurrence_key.values():
        ordered_nodes = sorted(
            same_key_nodes,
            key=lambda node: (generate_node_name(node.description), node.id),
        )
        for occurrence, node in enumerate(ordered_nodes, start=1):
            occurrence_by_extracted_node_id[node.id] = occurrence

    return occurrence_by_extracted_node_id


def _create_assertion(
    extracted_node: Node,
    entity_type: EntityType,
    data_chunk: DocumentChunk,
    speaker_name: Optional[str],
    attributed_name: Optional[str],
    occurrence: int,
) -> Assertion:
    """Build the Assertion data point. No explicit id: identity_fields derive it."""
    source_quote = getattr(extracted_node, "source_quote", None)
    return Assertion(
        name=generate_node_name(extracted_node.name),
        description=extracted_node.description,
        is_a=entity_type,
        belongs_to_set=data_chunk.belongs_to_set,
        importance_weight=data_chunk.importance_weight,
        statement_type=_statement_type_value(extracted_node),
        polarity=_enum_value(getattr(extracted_node, "polarity", None)) or "positive",
        asserted_by=speaker_name,
        attributed_to=attributed_name,
        applicable_time=getattr(extracted_node, "applicable_time", None),
        applies_from=getattr(extracted_node, "applies_from", None),
        applies_to=getattr(extracted_node, "applies_to", None),
        report_date=getattr(extracted_node, "report_date", None),
        conditions=list(getattr(extracted_node, "conditions", None) or []),
        precision=_enum_value(getattr(extracted_node, "precision", None)),
        scope=getattr(extracted_node, "scope", None),
        source_quote=source_quote,
        source_quote_verified=verify_source_quote(source_quote, getattr(data_chunk, "text", None)),
        responds_to=getattr(extracted_node, "responds_to", None),
        source_chunk_id=str(data_chunk.id),
        occurrence=occurrence,
    )


def _resolve_display_name(
    value: Optional[str],
    node_id_by_reference: dict[str, str],
    entities_by_extracted_node_id: dict[str, Entity],
) -> Optional[str]:
    """The name an assertion should store for a speaker/attribution reference.

    A reference to an entity of this extraction becomes that entity's normalized name, so
    the stored value matches the node the derived edge points at. Anything else — an
    unknown party, or a reference to another assertion — is kept as written.
    """
    resolved_node_id = _resolve_reference(value, node_id_by_reference)
    if resolved_node_id is not None:
        # Only non-assertion entities are indexed at this point, so an assertion
        # reference falls through to the raw value.
        referenced_entity = entities_by_extracted_node_id.get(resolved_node_id)
        if referenced_entity is not None:
            return referenced_entity.name

    return value


def _convert_extracted_nodes_to_data_points(
    data_chunk: DocumentChunk,
    extracted_graph: KnowledgeGraph,
    data_points_by_id: dict[str, Entity | EntityType],
    node_id_by_reference: dict[str, str],
) -> dict[str, Entity]:
    """Construct final DataPoints and index entities by their graph-local LLM IDs."""
    entity_ids_by_extracted_node_id = _calculate_entity_ids_by_extracted_node_id(
        extracted_graph,
        data_chunk,
    )
    entities_by_extracted_node_id: dict[str, Entity] = {}
    assertion_nodes: list[Node] = []

    for extracted_node in extracted_graph.nodes:
        if is_assertion_node(extracted_node):
            assertion_nodes.append(extracted_node)
            continue

        entity_type = _get_or_create_entity_type(
            extracted_node.type,
            data_chunk,
            data_points_by_id,
        )

        entity = _get_or_create_entity(
            extracted_node,
            entity_ids_by_extracted_node_id[extracted_node.id],
            entity_type,
            data_chunk,
            data_points_by_id,
        )
        entities_by_extracted_node_id[extracted_node.id] = entity
        _link_chunk_to_entity(data_chunk, extracted_node, entity)

    # Resolved before any assertion is indexed, so a reference can only name an entity.
    speaker_names = {
        node.id: _resolve_display_name(
            getattr(node, "asserted_by", None),
            node_id_by_reference,
            entities_by_extracted_node_id,
        )
        for node in assertion_nodes
    }
    attributed_names = {
        node.id: _resolve_display_name(
            getattr(node, "attributed_to", None),
            node_id_by_reference,
            entities_by_extracted_node_id,
        )
        for node in assertion_nodes
    }
    occurrence_by_extracted_node_id = _assertion_occurrences(
        assertion_nodes,
        speaker_names,
        str(data_chunk.id),
    )

    for extracted_node in assertion_nodes:
        entity_type = _get_or_create_entity_type(
            extracted_node.type,
            data_chunk,
            data_points_by_id,
        )
        assertion = _create_assertion(
            extracted_node,
            entity_type,
            data_chunk,
            speaker_names[extracted_node.id],
            attributed_names[extracted_node.id],
            occurrence_by_extracted_node_id[extracted_node.id],
        )
        # An identity this construction already produced (the same chunk extracted twice,
        # say) is reused rather than overwritten, like _get_or_create_entity: the chunk
        # must link to the object that is actually stored, never to an orphan.
        assertion_key = str(assertion.id)
        existing_data_point = data_points_by_id.get(assertion_key)
        if isinstance(existing_data_point, Assertion):
            assertion = existing_data_point
        else:
            data_points_by_id[assertion_key] = assertion
        entities_by_extracted_node_id[extracted_node.id] = assertion
        _link_chunk_to_entity(data_chunk, extracted_node, assertion)

    return entities_by_extracted_node_id


def _derive_assertion_edges(
    extracted_graph: KnowledgeGraph,
    node_id_by_reference: dict[str, str],
) -> list[KGEdge]:
    """Turn an assertion's reference fields into edges of the extracted graph.

    They go through the same path as the LLM's own edges, so provenance and ownership
    bookkeeping cannot tell them apart.
    """
    derived_edges: list[KGEdge] = []
    for extracted_node in extracted_graph.nodes:
        if not is_assertion_node(extracted_node):
            continue

        for field_name, relationship_name in _ASSERTION_REFERENCE_FIELDS:
            target_node_id = _resolve_reference(
                getattr(extracted_node, field_name, None),
                node_id_by_reference,
            )
            if target_node_id is None or target_node_id == extracted_node.id:
                continue

            derived_edges.append(
                KGEdge(
                    source_node_id=extracted_node.id,
                    target_node_id=target_node_id,
                    relationship_name=relationship_name,
                    description=None,
                )
            )

    return derived_edges


def _add_extracted_edges(
    data_chunk: DocumentChunk,
    extracted_edges: list[KGEdge],
    entities_by_extracted_node_id: dict[str, Entity],
    edges_by_identity: dict[EdgeIdentity, Edge],
) -> None:
    produced = data_chunk._produced_edge_identities
    for extracted_edge in extracted_edges:
        source_entity = entities_by_extracted_node_id.get(extracted_edge.source_node_id)
        target_entity = entities_by_extracted_node_id.get(extracted_edge.target_node_id)
        if source_entity is None or target_entity is None:
            continue

        relationship_name = generate_edge_name(extracted_edge.relationship_name)
        edge_text = _strip_nonblank_text(extracted_edge.description)
        edge_identity = EdgeIdentity(
            source_id=str(source_entity.id),
            target_id=str(target_entity.id),
            relationship_name=relationship_name,
        )
        # Both records are written HERE, before the deduplication below, so a
        # relationship the graph already holds still counts for this chunk:
        # such an edge is never attached to the chunk's model, so neither
        # ownership nor evidence may depend on it being written.
        #
        # Ownership needs one entry per distinct relationship — it decides
        # what survives when a chunk is deleted.
        produced_key = (edge_identity.source_id, edge_identity.target_id, relationship_name)
        if produced_key not in produced:
            produced.append(produced_key)
        # Evidence needs every occurrence with its supporting text, so this
        # one is appended unconditionally.
        data_chunk._provenance_edges.append(
            (
                edge_identity.source_id,
                edge_identity.target_id,
                relationship_name,
                {"edge_text": edge_text},
            )
        )
        edges_by_identity.setdefault(
            edge_identity,
            Edge(
                relationship_type=relationship_name,
                edge_text=edge_text,
            ),
        )


def construct_data_points_and_edges(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
) -> tuple[dict[str, Entity | EntityType], dict[EdgeIdentity, Edge]]:
    """Convert extracted knowledge graphs into DataPoints and edges."""
    data_points_by_id: dict[str, Entity | EntityType] = {}
    edges_by_identity: dict[EdgeIdentity, Edge] = {}

    for data_chunk, extracted_graph in zip(data_chunks, extracted_graphs):
        if not extracted_graph:
            continue

        node_id_by_reference = _node_id_by_reference(extracted_graph)
        entities_by_extracted_node_id = _convert_extracted_nodes_to_data_points(
            data_chunk,
            extracted_graph,
            data_points_by_id,
            node_id_by_reference,
        )
        # Derived edges come last, so an explicit LLM edge for the same relationship
        # keeps its description when the two deduplicate.
        _add_extracted_edges(
            data_chunk,
            [
                *extracted_graph.edges,
                *_derive_assertion_edges(extracted_graph, node_id_by_reference),
            ],
            entities_by_extracted_node_id,
            edges_by_identity,
        )

    return data_points_by_id, edges_by_identity


def attach_new_edges_to_data_points(
    data_points_by_id: dict[str, Entity | EntityType],
    edges_by_identity: dict[EdgeIdentity, Edge],
    existing_edge_identities: set[EdgeIdentity],
) -> None:
    """Attach edges that are not already stored in the graph database."""
    for edge_identity, edge in edges_by_identity.items():
        if edge_identity in existing_edge_identities:
            continue

        source_data_point = data_points_by_id.get(edge_identity.source_id)
        target_data_point = data_points_by_id.get(edge_identity.target_id)
        if source_data_point is None or target_data_point is None:
            continue

        source_data_point.relations.append((edge, target_data_point))
