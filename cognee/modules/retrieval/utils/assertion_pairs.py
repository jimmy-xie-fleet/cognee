"""Pulling the other half of an assertion pair into a retrieved context.

An ``Assertion``'s ``name`` is the underlying proposition phrased affirmatively, and the
stance on it lives in ``polarity`` -- so a denial only means something next to what it
denies. Triplet retrieval ranks nodes and edges by vector distance, which puts the denial
in the context whenever it matches the question but says nothing about whether the
``responds_to`` edge to its allegation also ranked. The result is a stance with nothing
attached to it, and an affirmative proposition the model can only read as a claim of its
own.

So every assertion in a retrieval asks the graph for its pair edges by type
(``responds_to``, ``attributed_to``, ``asserted_by``) and the counterpart arrives with it.

**Where this runs, and what follows from that.** The expansion belongs *after* every
ranking and truncation step, so it hangs off the retriever's ``resolve_edges_to_text``
funnel rather than off retrieval itself. Appending to what ``get_retrieved_objects``
returns does not work: a session turn merges two retrieval lanes through
``merge_ranked(..., limit=top_k)``, and since retrieval already returns up to ``top_k``
edges, anything appended past that index is truncated back off -- and a pair edge present
in both lanes would score as "found by both lanes" and evict a lane-unique triplet.
Rendering is the last thing that happens to an edge list, so nothing can drop the pair
edge after this point.

The consequence, and it is deliberate: **pair edges are context-only.** They are not part
of the retriever's returned objects, so they do not appear in
``extract_context_object_ids`` (the session's ``used_graph_element_ids``) or in
``get_context_evidence``. Those describe what retrieval *ranked*; the pair edge is
context the renderer added around it.

Cost: one adapter call per ``resolve_edges_to_text`` invocation that has an assertion in
its edge list -- so once per rendered context, which means once per lane in a batch or a
concurrent session turn, and once per round in a chain-of-thought loop. Exactly zero on a
graph without assertions in it: the id collection is a property test on nodes already in
hand, so a plain graph neither reaches the adapter nor resolves a graph engine.

The expansion only ever adds: it appends edges the retrieval did not already contain, and
it swallows its own adapter errors, so a failure here degrades the context back to what
retrieval alone produced rather than failing the search.
"""

from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence

from cognee.infrastructure.engine import is_internal_node
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.node_context_text import is_assertion_props
from cognee.modules.retrieval.config import get_retrieval_config
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    DEFAULT_EDGE_PROPERTIES_TO_PROJECT,
    default_node_properties_to_project,
)
from cognee.modules.retrieval.utils.merge_results import edge_identity
from cognee.shared.logging_utils import get_logger

logger = get_logger("assertion_pairs")

# What an assertion is paired with: the statement it answers, and who is behind it.
PAIR_EDGE_TYPES = ("responds_to", "attributed_to", "asserted_by")

# Two fields on top of the projection's own whitelist.
#
# ``feedback_weight``: the projection adds it whenever ``feedback_influence > 0``, which
# is the default. Allowing it unconditionally keeps this path off the retriever's ranking
# configuration for the sake of one numeric field.
#
# ``belongs_to_set``: the scope key. The graph projection does not need it because it
# filters scope inside the adapter query; this expansion asks for a neighborhood and
# cannot, so the only thing that can keep an out-of-scope counterpart out of a node-scoped
# search is a caller filtering on the property -- and a caller cannot filter on a property
# that was whitelisted away.
_EXTRA_NODE_FIELDS = ("feedback_weight", "belongs_to_set")
_EXTRA_EDGE_FIELDS = ("feedback_weight",)


def _mapping(value: Any) -> Optional[Mapping]:
    """``value`` when it is a property mapping, else ``None``.

    ``Mapping`` rather than ``dict``: an adapter is free to hand back its own mapping
    type, and a node whose attributes are some foreign object is simply not an assertion.
    """
    return value if isinstance(value, Mapping) else None


def _node_id(node: Any) -> str:
    node_id = getattr(node, "id", None)
    return str(node_id).strip() if node_id is not None else ""


def assertion_ids_in(nodes: Iterable[Any]) -> list[str]:
    """Ids of the nodes that are assertions, in encounter order, deduplicated.

    Same test as everywhere else: one non-blank ``statement_type`` decides
    (``is_assertion_props``), never the node's ``type``.
    """
    ids: list[str] = []
    for node in nodes or []:
        attributes = _mapping(getattr(node, "attributes", None))
        if attributes is None or not is_assertion_props(attributes):
            continue
        node_id = _node_id(node)
        if node_id and node_id not in ids:
            ids.append(node_id)
    return ids


def _node_projection() -> list[str]:
    return list(dict.fromkeys([*default_node_properties_to_project(), *_EXTRA_NODE_FIELDS]))


def _edge_projection() -> list[str]:
    return list(dict.fromkeys([*DEFAULT_EDGE_PROPERTIES_TO_PROJECT, *_EXTRA_EDGE_FIELDS]))


def _nodes_from_rows(rows: Any) -> dict[str, Node]:
    """``{id: Node}`` for the rows ``get_neighborhood`` returned.

    Built exactly the way ``CogneeGraph._process_nodes_and_edges`` builds a projected
    node: internal nodes dropped, then the property whitelist applied -- every projected
    key present, ``None`` where the store had nothing. Whitelisting matters because these
    rows are the *whole* stored property bag, and a node that reached a context through
    the projection never carried more than this (``search(verbose=True)`` returns node
    attributes, so an unfiltered row would ship a document's ``raw_data_location``).
    """
    projection = _node_projection()
    nodes: dict[str, Node] = {}
    for row in rows or []:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            logger.debug("Assertion pair expansion skipped a malformed node row")
            continue
        node_id, properties = str(row[0]), _mapping(row[1])
        if not node_id or properties is None:
            logger.debug("Assertion pair expansion skipped a node row without properties")
            continue
        if is_internal_node(properties):
            continue
        nodes[node_id] = Node(node_id, {key: properties.get(key) for key in projection})
    return nodes


def _pair_edges_from_rows(
    rows: Any,
    nodes: dict[str, Node],
    allowed_types: set[str],
    seed_ids: set[str],
) -> list[Edge]:
    """The pair edges *on the seed assertions*, as the projection would have built them.

    Two filters the adapter does not apply, because it answers a different question: it
    type-filters which *neighbors* it traverses to, then returns every edge between the
    nodes it kept (see its ``get_neighborhood`` docstring). So it can hand back an untyped
    ``contradicts`` edge between two assertions, and a ``responds_to`` edge between two
    *neighbors* that has nothing to do with any seed. Both are dropped here: an edge
    survives only if its type was asked for and one of its endpoints is a seed.
    """
    edges: list[Edge] = []
    projection = _edge_projection()
    for row in rows or []:
        if not isinstance(row, (tuple, list)) or len(row) < 3:
            logger.debug("Assertion pair expansion skipped a malformed edge row")
            continue
        source_id, target_id, relationship_type = str(row[0]), str(row[1]), str(row[2])
        if relationship_type not in allowed_types:
            continue
        if source_id not in seed_ids and target_id not in seed_ids:
            continue
        source, target = nodes.get(source_id), nodes.get(target_id)
        if source is None or target is None:
            continue

        properties = _mapping(row[3]) if len(row) > 3 else None
        attributes = {key: (properties or {}).get(key) for key in projection}
        # The edge label the store returned, stamped over the projection the way
        # ``_process_nodes_and_edges`` stamps it; renderers read it first.
        attributes["relationship_type"] = relationship_type
        edges.append(Edge(source, target, attributes=attributes))
    return edges


async def _graph_engine(graph_engine: Any) -> Any:
    """The adapter to ask, resolving a provider callable if that is what was passed.

    Callers hand this a bound method rather than an engine so that resolving an engine --
    which on the chain-of-thought and temporal paths means building one -- happens only
    once an assertion has actually been found in the retrieval.
    """
    if not callable(graph_engine):
        return graph_engine
    try:
        return await graph_engine()
    except Exception as error:
        logger.warning(
            "Assertion pair expansion skipped: no graph engine",
            extra={"error": str(error)},
        )
        return None


async def expand_assertion_pairs(
    graph_engine: Any,
    assertion_ids: Sequence[str],
    *,
    edge_types: Sequence[str] = PAIR_EDGE_TYPES,
) -> tuple[list[Node], list[Edge]]:
    """The pair edges one hop from ``assertion_ids``, and every node they touch.

    The nodes are the seed assertions themselves (in the order asked for) followed by the
    counterparts the kept edges reached, so a caller that wants to render or inspect a
    pair has both of its halves without a second lookup.

    Returns ``([], [])`` -- never raises -- when there is nothing to ask for, when the
    adapter has no neighborhood support, or when it fails: an incomplete context is a far
    better outcome here than a failed search.
    """
    ids = list(dict.fromkeys(str(node_id).strip() for node_id in assertion_ids or [] if node_id))
    if not ids:
        return [], []

    engine = await _graph_engine(graph_engine)
    if engine is None:
        return [], []

    try:
        node_rows, edge_rows = await engine.get_neighborhood(
            ids, depth=1, edge_types=list(edge_types)
        )
    except Exception as error:
        logger.warning(
            "Assertion pair expansion skipped: neighborhood lookup failed",
            extra={"assertion_count": len(ids), "error": str(error)},
        )
        return [], []

    nodes = _nodes_from_rows(node_rows)
    seed_ids = set(ids)
    edges = _pair_edges_from_rows(edge_rows, nodes, set(edge_types), seed_ids)

    kept_ids = [node_id for node_id in ids if node_id in nodes]
    for edge in edges:
        for endpoint in (edge.node1.id, edge.node2.id):
            if endpoint not in seed_ids and endpoint not in kept_ids:
                kept_ids.append(endpoint)

    return [nodes[node_id] for node_id in kept_ids], edges


async def append_assertion_pair_edges(
    graph_engine: Any,
    edges: list,
    *,
    edge_types: Sequence[str] = PAIR_EDGE_TYPES,
) -> list:
    """``edges`` plus every pair edge its assertions are missing.

    ``graph_engine`` may be an adapter or an async provider of one; the provider is only
    called if there is an assertion to expand. Returns the list it was given -- the same
    object -- whenever there is nothing to add, so a retrieval over a graph with no
    assertions in it is untouched, down to identity.
    """
    if (
        not get_retrieval_config().graph_completion_pair_expansion
        or not edges
        or graph_engine is None
    ):
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

    # The nodes are not dropped: they are the ``node1``/``node2`` of these very edges,
    # built once from the returned rows, so appending the edges carries them along. A
    # caller that wants the nodes on their own calls ``expand_assertion_pairs`` directly.
    _pair_nodes, pair_edges = await expand_assertion_pairs(
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
        "Assertion pair expansion added edges to a rendered context",
        extra={"assertion_count": len(assertion_ids), "added_edge_count": len(added)},
    )
    return [*edges, *added]
