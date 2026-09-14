from enum import Enum
from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.models.Assertion import Assertion, verify_source_quote
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node

# Assertion fields that name another extracted node, and the edge each one derives.
_ASSERTION_REFERENCE_FIELDS = (
    ("asserted_by", "asserted_by"),
    ("attributed_to", "attributed_to"),
    ("responds_to", "responds_to"),
)

# How a derived asserted_by edge words the speaker's stance. The unknown phrase reads
# "takes an unrecorded stance on X" rather than "... on that X": the verb takes its object
# directly, and the text is embedded and shown to a reader, so it has to be a sentence.
_STANCE_VERB_BY_POLARITY = {
    "positive": "affirms that",
    "negative": "denies that",
}
_UNRECORDED_STANCE_VERB = "takes an unrecorded stance on"

_DERIVED_EDGE_VERBS = {
    "attributed_to": "is attributed to",
    "responds_to": "responds to",
}


def _strip_nonblank_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped_value = value.strip()
    return stripped_value or None


def is_assertion_node(extracted_node: Node) -> bool:
    """True when an extracted node carries a statement type.

    A plain ``cognee.shared.data_models.Node`` has no ``statement_type`` attribute at all, so
    a plain extraction is never treated as an assertion — not even when its type reads
    "Statement". Only an extraction model that declares the qualifier fields opts in.

    For a model that does declare it, that one field decides and the node's ``type`` is
    never consulted: types are rewritten by ontology canonicalization ("Records" ->
    "record"), so reading the type would turn an entity into an Assertion — chunk-scoped
    and never deduplicated by name — the moment an ontology is configured, and demote it
    again under a different one. A blank statement type is no statement type.
    """
    if not hasattr(extracted_node, "statement_type"):
        return False

    return _statement_type_text(extracted_node) is not None


def _enum_value(value):
    """Enum members arrive from typed extraction models; store their value."""
    return value.value if isinstance(value, Enum) else value


def _statement_type_text(extracted_node: Node) -> Optional[str]:
    """The declared statement type as free text, or None when it declares none."""
    statement_type = _enum_value(getattr(extracted_node, "statement_type", None))
    if not isinstance(statement_type, str):
        return None

    return _strip_nonblank_text(statement_type)


def _statement_type_value(extracted_node: Node) -> str:
    """The speech act to store, normalized the way identity normalizes it.

    A model declaring ``statement_type: str`` may hand back " Denial " where the enum-typed
    model hands back "denial"; identity folds those together, so the stored property has
    to fold them together too. The node type is a last resort for a node that reached
    construction as an assertion without one, which ``is_assertion_node`` does not allow.
    """
    return generate_node_name(_statement_type_text(extracted_node) or extracted_node.type)


def _polarity_value(extracted_node: Node) -> str:
    """The stance to store: what was extracted, or "unknown" when nothing was.

    The extraction schema leaves ``polarity`` optional, so a schema-valid extraction can
    omit it. Defaulting an unrecorded stance to "positive" would have the graph state that
    the speaker affirms a proposition nobody said they affirm — the one reading a denial
    must never get — so a missing stance stays missing.
    """
    return _enum_value(getattr(extracted_node, "polarity", None)) or "unknown"


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


def _chunk_link_edge_text(
    extracted_node: Node,
    entity: Entity,
    speaker_node: Optional[Node],
) -> Optional[str]:
    """The text of the chunk's ``contains`` edge to one data point.

    For an assertion it has to state the stance. "Document chunk mentions payment was
    late" is the assertion's affirmative ``name``, which for a denial is the fact being
    denied — and this text is embedded and shown to a reader exactly like any other edge
    text, so a denial would be retrieved as the claim it rejects.
    """
    description = _strip_nonblank_text(extracted_node.description)
    if not isinstance(entity, Assertion):
        return f"Document chunk mentions {entity.name}: {description}" if description else None

    if speaker_node is not None:
        # The same sentence the derived asserted_by edge carries, so the two agree.
        return (
            "Document chunk records: "
            f"{_derived_edge_description(extracted_node, 'asserted_by', speaker_node)}"
        )

    head = (
        f"Document chunk records a {_statement_type_value(extracted_node)} with "
        f"{_polarity_value(extracted_node)} stance: {_proposition_clause(extracted_node)}"
    )
    return " ".join(_sentence(part) for part in (head, description) if part)


def _link_chunk_to_entity(
    data_chunk: DocumentChunk,
    extracted_node: Node,
    entity: Entity,
    speaker_node: Optional[Node] = None,
) -> None:
    if data_chunk.contains is None:
        data_chunk.contains = []

    data_chunk.contains.append(
        (
            Edge(
                relationship_type="contains",
                edge_text=_chunk_link_edge_text(extracted_node, entity, speaker_node),
            ),
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
    reference_names: dict[str, Optional[str]],
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
        polarity=_polarity_value(extracted_node),
        asserted_by=reference_names["asserted_by"],
        attributed_to=reference_names["attributed_to"],
        applicable_time=getattr(extracted_node, "applicable_time", None),
        applies_from=getattr(extracted_node, "applies_from", None),
        applies_to=getattr(extracted_node, "applies_to", None),
        report_date=getattr(extracted_node, "report_date", None),
        conditions=list(getattr(extracted_node, "conditions", None) or []),
        precision=_enum_value(getattr(extracted_node, "precision", None)),
        scope=getattr(extracted_node, "scope", None),
        source_quote=source_quote,
        source_quote_verified=verify_source_quote(source_quote, getattr(data_chunk, "text", None)),
        responds_to=reference_names["responds_to"],
        source_chunk_id=str(data_chunk.id),
        occurrence=occurrence,
    )


def _speaker_node(
    extracted_node: Node,
    node_id_by_reference: dict[str, str],
    nodes_by_extracted_id: dict[str, Node],
) -> Optional[Node]:
    """The party node an assertion's ``asserted_by`` names, when it names one.

    One test decides the stored speaker, the derived ``asserted_by`` edge and the stance
    the chunk link states, so those three can never disagree about who spoke.
    """
    target_node_id = _resolve_reference(
        getattr(extracted_node, "asserted_by", None),
        node_id_by_reference,
    )
    if target_node_id is None or target_node_id == extracted_node.id:
        return None

    target_node = nodes_by_extracted_id.get(target_node_id)
    if target_node is None or is_assertion_node(target_node):
        # A speaker is a party; an assertion is not one (see _resolve_speaker_name).
        return None

    return target_node


def _resolve_display_name(
    value: Optional[str],
    node_id_by_reference: dict[str, str],
    entities_by_extracted_node_id: dict[str, Entity],
) -> Optional[str]:
    """The name an assertion should store for an attribution/response reference.

    A reference to an entity of this extraction becomes that entity's normalized name, so
    the stored value matches the node the derived edge points at. Anything else — a
    locator naming no node ("Complaint ¶17"), or a reference to another assertion — is
    kept as written here; a reference to another assertion is rewritten to that
    assertion's id afterwards, once every assertion exists
    (``_repoint_assertion_references_at_resolved_ids``).
    """
    resolved_node_id = _resolve_reference(value, node_id_by_reference)
    if resolved_node_id is not None:
        # Only non-assertion entities are indexed at this point, so an assertion
        # reference falls through to the raw value.
        referenced_entity = entities_by_extracted_node_id.get(resolved_node_id)
        if referenced_entity is not None:
            return referenced_entity.name

    return value


def _resolve_speaker_name(
    value: Optional[str],
    node_id_by_reference: dict[str, str],
    entities_by_extracted_node_id: dict[str, Entity],
) -> Optional[str]:
    """The speaker an assertion should store, or None when no party was named.

    ``asserted_by`` is an identity field, so whatever lands here decides the node's id.
    An LLM's graph-local token ("n9") must therefore never reach it: it means nothing
    outside the single response that invented it, it would move the node whenever the
    extraction renumbers, and the derived edge would read "The payment was late denies
    that the payment was late".

    A speaker is a party. A reference that names another assertion is not one, so it
    stores no speaker at all — unlike ``attributed_to``/``responds_to``, which point at
    statements by design. A reference that names no node of the graph is free text the
    passage supplied ("the applicant's counsel") and is kept: every token that does name
    a node resolves here, and canonicalization nulls the references it drops nodes for,
    so nothing id-shaped survives this branch.
    """
    resolved_node_id = _resolve_reference(value, node_id_by_reference)
    if resolved_node_id is None:
        return _strip_nonblank_text(value)

    # Only non-assertion entities are indexed at this point, so a reference that resolves
    # to nothing here named an assertion.
    referenced_entity = entities_by_extracted_node_id.get(resolved_node_id)
    return referenced_entity.name if referenced_entity is not None else None


def _assertion_reference_names(
    extracted_node: Node,
    node_id_by_reference: dict[str, str],
    entities_by_extracted_node_id: dict[str, Entity],
) -> dict[str, Optional[str]]:
    """What the assertion stores for each of its three reference fields."""
    reference_names = {
        "asserted_by": _resolve_speaker_name(
            getattr(extracted_node, "asserted_by", None),
            node_id_by_reference,
            entities_by_extracted_node_id,
        )
    }
    for field_name in ("attributed_to", "responds_to"):
        reference_names[field_name] = _resolve_display_name(
            getattr(extracted_node, field_name, None),
            node_id_by_reference,
            entities_by_extracted_node_id,
        )
    return reference_names


def _repoint_assertion_references_at_resolved_ids(
    assertion_nodes: list[Node],
    node_id_by_reference: dict[str, str],
    entities_by_extracted_node_id: dict[str, Entity],
) -> None:
    """Store the id of a referenced assertion instead of the LLM's graph-local token.

    "n4" or "answer-p17-denial" only means something inside the single extraction response
    that invented it, so an assertion pointing at another assertion of the same response
    keeps that assertion's stored id. Locators that name nothing in the response
    ("Complaint ¶17") and references to plain entities are left exactly as they were.

    Runs after every assertion exists, because the target's id is what is being stored.
    Both fields are non-identity fields, so rewriting them cannot move a node; the
    identity field ``asserted_by`` is deliberately untouched.
    """
    for extracted_node in assertion_nodes:
        assertion = entities_by_extracted_node_id.get(extracted_node.id)
        if not isinstance(assertion, Assertion):
            continue

        for field_name in ("responds_to", "attributed_to"):
            target_node_id = _resolve_reference(
                getattr(extracted_node, field_name, None),
                node_id_by_reference,
            )
            if target_node_id is None or target_node_id == extracted_node.id:
                continue

            referenced_data_point = entities_by_extracted_node_id.get(target_node_id)
            if isinstance(referenced_data_point, Assertion):
                setattr(assertion, field_name, str(referenced_data_point.id))


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
    nodes_by_extracted_id = {node.id: node for node in extracted_graph.nodes}
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

    # Resolved before any assertion is indexed, so a reference resolving to nothing here
    # named an assertion rather than an entity.
    reference_names_by_extracted_node_id = {
        node.id: _assertion_reference_names(
            node,
            node_id_by_reference,
            entities_by_extracted_node_id,
        )
        for node in assertion_nodes
    }
    occurrence_by_extracted_node_id = _assertion_occurrences(
        assertion_nodes,
        {
            node_id: reference_names["asserted_by"]
            for node_id, reference_names in reference_names_by_extracted_node_id.items()
        },
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
            reference_names_by_extracted_node_id[extracted_node.id],
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
        _link_chunk_to_entity(
            data_chunk,
            extracted_node,
            assertion,
            _speaker_node(extracted_node, node_id_by_reference, nodes_by_extracted_id),
        )

    _repoint_assertion_references_at_resolved_ids(
        assertion_nodes,
        node_id_by_reference,
        entities_by_extracted_node_id,
    )

    return entities_by_extracted_node_id


def _sentence(text: str) -> str:
    """One sentence of edge text, terminated exactly once."""
    stripped_text = text.strip()
    return stripped_text if stripped_text.endswith((".", "!", "?")) else f"{stripped_text}."


def _proposition_clause(extracted_node: Node) -> str:
    """The proposition as it reads inside a sentence, without doubling its full stop."""
    name = _strip_nonblank_text(extracted_node.name)
    if name is None:
        return "this statement"

    return name.rstrip(".").strip() or "this statement"


def _reference_label(extracted_node: Node) -> str:
    """How a derived edge names the node it points at."""
    return (
        _strip_nonblank_text(extracted_node.name)
        or _strip_nonblank_text(extracted_node.type)
        or "an unnamed party"
    )


def _derived_edge_description(
    extracted_node: Node,
    relationship_name: str,
    target_node: Optional[Node],
) -> Optional[str]:
    """The text a derived edge carries, stating the stance the assertion was made with.

    Without it the edge reaches storage with no ``edge_text``, and
    ``ensure_default_edge_properties`` synthesizes one from the endpoint labels — for an
    assertion that is its affirmative ``name``, so a denial is embedded and shown as the
    fact it denies. The stance therefore has to travel with the edge, not be reconstructed
    from the endpoints, which no longer carry it.
    """
    if target_node is None:
        return None

    proposition = _proposition_clause(extracted_node)
    polarity = _polarity_value(extracted_node)
    if relationship_name == "asserted_by":
        stance_verb = _STANCE_VERB_BY_POLARITY.get(polarity, _UNRECORDED_STANCE_VERB)
        head = f"{_reference_label(target_node)} {stance_verb} {proposition}"
    else:
        head = (
            f"{proposition} ({_statement_type_value(extracted_node)}, {polarity} stance) "
            f"{_DERIVED_EDGE_VERBS[relationship_name]} {_reference_label(target_node)}"
        )

    description = _strip_nonblank_text(extracted_node.description)
    return " ".join(_sentence(part) for part in (head, description) if part)


def _derive_assertion_edges(
    extracted_graph: KnowledgeGraph,
    node_id_by_reference: dict[str, str],
) -> list[KGEdge]:
    """Turn an assertion's reference fields into edges of the extracted graph.

    They go through the same path as the LLM's own edges, so provenance and ownership
    bookkeeping cannot tell them apart, and each carries a description stating the stance
    the statement was made with.
    """
    nodes_by_extracted_id = {node.id: node for node in extracted_graph.nodes}
    derived_edges: list[KGEdge] = []
    for extracted_node in extracted_graph.nodes:
        if not is_assertion_node(extracted_node):
            continue

        for field_name, relationship_name in _ASSERTION_REFERENCE_FIELDS:
            if relationship_name == "asserted_by":
                # No speaker was stored for a reference naming another assertion, so no
                # edge may claim one either.
                target_node = _speaker_node(
                    extracted_node,
                    node_id_by_reference,
                    nodes_by_extracted_id,
                )
                if target_node is None:
                    continue
            else:
                target_node_id = _resolve_reference(
                    getattr(extracted_node, field_name, None),
                    node_id_by_reference,
                )
                if target_node_id is None or target_node_id == extracted_node.id:
                    continue

                target_node = nodes_by_extracted_id.get(target_node_id)
                if target_node is None:
                    continue

            derived_edges.append(
                KGEdge(
                    source_node_id=extracted_node.id,
                    target_node_id=target_node.id,
                    relationship_name=relationship_name,
                    description=_derived_edge_description(
                        extracted_node,
                        relationship_name,
                        target_node,
                    ),
                )
            )

    return derived_edges


def _edge_key(extracted_edge: KGEdge) -> tuple[str, str, str]:
    """The triple the edges deduplicate on, keyed on graph-local node ids."""
    return (
        extracted_edge.source_node_id,
        extracted_edge.target_node_id,
        generate_edge_name(extracted_edge.relationship_name),
    )


def _merge_derived_edges(
    extracted_edges: list[KGEdge],
    derived_edges: list[KGEdge],
) -> list[KGEdge]:
    """Combine the LLM's own edges with the derived ones, keeping the stance text.

    The two deduplicate on (source, target, relationship) downstream and the first one
    wins. An explicit ``asserted_by`` edge routinely arrives with no description, and it
    used to win with no edge text at all — leaving storage to synthesize one from the
    endpoint labels, i.e. from the assertion's affirmative name, so a denial reached
    retrieval as the fact it denies.

    A description-less explicit edge therefore adopts the derived edge's description, and
    that derived duplicate is dropped so the chunk records the relationship exactly once.
    An explicit edge carrying its own description is left untouched and still wins.
    """
    if not derived_edges:
        return list(extracted_edges)

    derived_edge_by_key: dict[tuple[str, str, str], KGEdge] = {}
    for derived_edge in derived_edges:
        derived_edge_by_key.setdefault(_edge_key(derived_edge), derived_edge)

    merged_edges: list[KGEdge] = []
    adopted_keys: set[tuple[str, str, str]] = set()
    for extracted_edge in extracted_edges:
        edge_key = _edge_key(extracted_edge)
        derived_edge = derived_edge_by_key.get(edge_key)
        if derived_edge is None or _strip_nonblank_text(extracted_edge.description) is not None:
            merged_edges.append(extracted_edge)
            continue

        adopted_keys.add(edge_key)
        merged_edges.append(
            extracted_edge.model_copy(update={"description": derived_edge.description})
        )

    merged_edges.extend(
        derived_edge
        for derived_edge in derived_edges
        if _edge_key(derived_edge) not in adopted_keys
    )
    return merged_edges


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
        _add_extracted_edges(
            data_chunk,
            _merge_derived_edges(
                extracted_graph.edges,
                _derive_assertion_edges(extracted_graph, node_id_by_reference),
            ),
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
