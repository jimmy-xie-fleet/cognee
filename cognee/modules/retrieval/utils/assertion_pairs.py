"""Pulling the other half of an assertion pair into a retrieved context.

An ``Assertion``'s ``name`` is the underlying proposition phrased affirmatively, and the
stance on it lives in ``polarity`` -- so a denial only means something next to what it
denies. Triplet retrieval ranks nodes and edges by vector distance, which puts the denial
in the context whenever it matches the question but says nothing about whether the
``responds_to`` edge to its allegation also ranked. The result is a stance with nothing
attached to it, and an affirmative proposition the model can only read as a claim of its
own.

So after retrieval, every assertion in the result asks the graph for its pair edges by
type (``responds_to``, ``attributed_to``, ``asserted_by``) and the counterpart arrives with
it. This costs one extra adapter call per retrieval that surfaced an assertion, and exactly
zero on a graph without assertions in it -- the id collection is a property test on nodes
already in hand, so a plain graph never reaches the adapter.

The expansion only ever adds: it appends edges the retrieval did not already contain, and
it swallows its own adapter errors, so a failure here degrades the context back to what
retrieval alone produced rather than failing the search.
"""

from typing import Any, Iterable, Optional, Sequence

from cognee.infrastructure.engine import is_internal_node
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.node_context_text import is_assertion_props
from cognee.modules.retrieval.utils.merge_results import edge_identity
from cognee.shared.logging_utils import get_logger

logger = get_logger("assertion_pairs")

# Task 7 moves this onto RetrievalConfig; until then the module constant is the switch.
PAIR_EXPANSION_ENABLED = True

# What an assertion is paired with: the statement it answers, and who is behind it.
PAIR_EDGE_TYPES = ("responds_to", "attributed_to", "asserted_by")


def _node_id(node: Any) -> str:
    node_id = getattr(node, "id", None)
    return str(node_id).strip() if node_id is not None else ""


def assertion_ids_in(nodes: Iterable[Any]) -> list[str]:
    """Ids of the nodes that are assertions, in encounter order, deduplicated.

    Same test as everywhere else: one non-blank ``statement_type`` decides
    (``is_assertion_props``), never the node's ``type``. A node whose attributes are not a
    mapping -- a foreign object, a test double -- is simply not an assertion.
    """
    ids: list[str] = []
    for node in nodes or []:
        attributes = getattr(node, "attributes", None)
        if not isinstance(attributes, dict):
            continue
        if not is_assertion_props(attributes):
            continue
        node_id = _node_id(node)
        if node_id and node_id not in ids:
            ids.append(node_id)
    return ids


def _nodes_from_rows(rows: Any) -> dict[str, Node]:
    """``{id: Node}`` for the rows ``get_neighborhood`` returned, internal nodes dropped."""
    nodes: dict[str, Node] = {}
    for row in rows or []:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            continue
        node_id, properties = str(row[0]), row[1]
        if not node_id or not isinstance(properties, dict):
            continue
        if is_internal_node(properties):
            continue
        nodes[node_id] = Node(node_id, dict(properties))
    return nodes


def _pair_edges_from_rows(rows: Any, nodes: dict[str, Node], allowed_types: set[str]) -> list[Edge]:
    """The pair edges among ``nodes``, as ``Edge`` objects built like the projection builds them.

    The type filter is applied here as well as in the adapter call: the Kuzu/Ladybug
    adapter type-filters which *neighbors* it traverses to, then returns every edge
    between the nodes it kept (see its ``get_neighborhood`` docstring), so an untyped
    ``contradicts`` edge between two assertions comes back with the pair edges.
    """
    edges: list[Edge] = []
    for row in rows or []:
        if not isinstance(row, (tuple, list)) or len(row) < 3:
            continue
        source_id, target_id, relationship_type = str(row[0]), str(row[1]), str(row[2])
        if relationship_type not in allowed_types:
            continue
        source, target = nodes.get(source_id), nodes.get(target_id)
        if source is None or target is None:
            continue

        properties = row[3] if len(row) > 3 and isinstance(row[3], dict) else {}
        attributes = dict(properties)
        # ``relationship_type`` is the edge label the store returned, which is what
        # ``_process_nodes_and_edges`` stamps too; renderers read it first.
        attributes["relationship_type"] = relationship_type
        attributes.setdefault("relationship_name", relationship_type)
        edges.append(Edge(source, target, attributes=attributes))
    return edges


async def expand_assertion_pairs(
    graph_engine: Any,
    assertion_ids: Sequence[str],
    *,
    edge_types: Sequence[str] = PAIR_EDGE_TYPES,
) -> tuple[list[Node], list[Edge]]:
    """The counterpart nodes and pair edges one hop from ``assertion_ids``.

    Returns ``([], [])`` -- never raises -- when there is nothing to ask for, when the
    adapter has no neighborhood support, or when it fails: an incomplete context is a far
    better outcome here than a failed search.
    """
    ids = list(dict.fromkeys(str(node_id).strip() for node_id in assertion_ids or [] if node_id))
    if not ids or graph_engine is None:
        return [], []

    try:
        node_rows, edge_rows = await graph_engine.get_neighborhood(
            ids, depth=1, edge_types=list(edge_types)
        )
    except Exception as error:
        logger.warning(
            "Assertion pair expansion skipped: neighborhood lookup failed",
            extra={"assertion_count": len(ids), "error": str(error)},
        )
        return [], []

    nodes = _nodes_from_rows(node_rows)
    edges = _pair_edges_from_rows(edge_rows, nodes, set(edge_types))

    seed_ids = set(ids)
    counterpart_ids: list[str] = []
    for edge in edges:
        for endpoint in (edge.node1.id, edge.node2.id):
            if endpoint not in seed_ids and endpoint not in counterpart_ids:
                counterpart_ids.append(endpoint)

    return [nodes[node_id] for node_id in counterpart_ids], edges


async def append_assertion_pair_edges(
    graph_engine: Any,
    edges: list,
    *,
    edge_types: Sequence[str] = PAIR_EDGE_TYPES,
) -> list:
    """``edges`` plus every pair edge its assertions are missing.

    Returns the list it was given -- the same object -- whenever there is nothing to add,
    so a retrieval over a graph with no assertions in it is untouched, down to identity.
    """
    if not PAIR_EXPANSION_ENABLED or not edges or graph_engine is None:
        return edges

    endpoints = [
        endpoint
        for edge in edges
        for endpoint in (getattr(edge, "node1", None), getattr(edge, "node2", None))
        if endpoint is not None
    ]
    assertion_ids = assertion_ids_in(endpoints)
    if not assertion_ids:
        return edges

    _counterparts, pair_edges = await expand_assertion_pairs(
        graph_engine, assertion_ids, edge_types=edge_types
    )
    if not pair_edges:
        return edges

    seen = {edge_identity(edge) for edge in edges}
    added: list[Edge] = []
    for pair_edge in pair_edges:
        identity = edge_identity(pair_edge)
        if identity in seen:
            continue
        seen.add(identity)
        added.append(pair_edge)

    if not added:
        return edges

    logger.debug(
        "Assertion pair expansion added edges to a retrieved context",
        extra={"assertion_count": len(assertion_ids), "added_edge_count": len(added)},
    )
    return [*edges, *added]


async def append_assertion_pairs_to_retrieval(
    graph_engine: Any,
    retrieved: Optional[list],
    *,
    edge_types: Sequence[str] = PAIR_EDGE_TYPES,
) -> Optional[list]:
    """``append_assertion_pair_edges`` over either retrieval shape: one edge list, or a batch of them."""
    if not retrieved or not isinstance(retrieved, list):
        return retrieved

    if all(isinstance(lane, list) for lane in retrieved):
        lanes = [
            await append_assertion_pair_edges(graph_engine, lane, edge_types=edge_types)
            for lane in retrieved
        ]
        return (
            lanes
            if any(lane is not original for lane, original in zip(lanes, retrieved))
            else retrieved
        )

    return await append_assertion_pair_edges(graph_engine, retrieved, edge_types=edge_types)
