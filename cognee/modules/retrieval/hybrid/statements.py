"""The statements lane of hybrid retrieval.

``HYBRID_COMPLETION`` is the default search type in the router, in both HTTP DTOs, in the
CLI and in the UI, and it searches ``Entity_name``. A legal graph's statements are
``Assertion`` nodes, indexed in their own ``Assertion_name`` collection, so they never
reached a hybrid context as seeds -- at best as an anonymous one-hop neighbour of some
entity that happened to rank. Measured on the same documents, the legal graph's hybrid
context came to 15.7k characters against 57.7k for a plain graph of the same corpus.

This lane mirrors ``hybrid/entities.py``: one vector search, one graph round trip, one
rendered section. What it adds over the entity lane is the pair: an ``Assertion``'s ``name``
is the underlying proposition phrased AFFIRMATIVELY and the stance on it lives in
``polarity``, so a denial only means something next to the allegation it answers and the
speaker behind it. Those come back from ``expand_assertion_pairs`` in the same call.

It only ever adds. A dataset with no ``Assertion_name`` collection -- every non-legal graph
-- produces no statements, no section, and not even a key in the retrieval result, so its
context is byte-identical to the one this module did not exist for.
"""

from typing import Any, Optional

from cognee.modules.graph.utils.node_context_text import node_context_label, node_context_text
from cognee.modules.retrieval.hybrid.chunks import search_collection
from cognee.modules.retrieval.hybrid.results import (
    display_value,
    payload,
    payload_matches_node_filter,
    result_id,
)
from cognee.modules.retrieval.utils.assertion_pairs import PAIR_EDGE_TYPES, expand_assertion_pairs
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")

STATEMENTS_COLLECTION = "Assertion_name"

# Adapter order is not context order: a store is free to return a seed's edges in any
# order it likes, and the same graph then renders a different section on every backend.
# The pair types sort in the order they are declared -- what the statement answers, then
# who it is attributed to, then who said it -- and an unlisted relationship follows them.
_PAIR_TYPE_ORDER = {relationship: rank for rank, relationship in enumerate(PAIR_EDGE_TYPES)}

# How a pair edge reads on the statement it was rendered under. Direction matters: on the
# denial the ``responds_to`` edge is what it answers, and on the allegation the same edge
# is the answer it drew -- both are worth having, and neither reads correctly as the other.
OUTGOING_PAIR_LABELS = {
    "responds_to": "responds to",
    "attributed_to": "attributed to",
    "asserted_by": "speaker",
}
# Only relationships whose reverse reading describes *this* statement get an incoming
# wording; the rest are skipped rather than worded backwards.
INCOMING_PAIR_LABELS = {"responds_to": "answered by"}


async def search_statements(
    vector_engine: Any,
    query: str,
    top_k: int,
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
    query_vector: list[float],
) -> list[Any]:
    """Assertion_name hits, or empty if the collection is missing or search fails."""
    try:
        return await search_collection(
            vector_engine,
            STATEMENTS_COLLECTION,
            query,
            top_k,
            node_name,
            node_name_filter_operator,
            query_vector=query_vector,
        )
    except Exception as error:
        logger.warning("Assertion_name search failed; continuing without statements: %s", error)
        return []


async def build_statements(
    graph_engine: Any,
    hits: list[Any],
    *,
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
) -> list[dict]:
    """One renderable statement per hit, each with the pair lines its graph edges support.

    Costs exactly one graph call, and none at all when nothing was retrieved. The vector
    row is what matched; the graph node is what the statement renders from, because the
    row is an index projection (see ``_render_properties``). A row indexed with more than
    the index projection -- or a graph that returned nothing for the seed -- still renders
    from whatever the row carries.
    """
    seeds = _seed_rows(hits)
    if not seeds:
        return []

    nodes, pair_edges = await expand_assertion_pairs(graph_engine, list(seeds))
    graph_properties = _properties_by_id(nodes)
    pairs_by_id = _pairs_by_seed(pair_edges, set(seeds), node_name, node_name_filter_operator)

    statements = []
    for seed_id, row_properties in seeds.items():
        title, body = node_context_text(
            _render_properties(row_properties, graph_properties.get(seed_id, {}))
        )
        statements.append(
            {
                "id": seed_id,
                "title": title,
                "body": body,
                "pairs": pairs_by_id.get(seed_id, []),
            }
        )
    return statements


# What a vector row says about itself rather than about the node it indexes. On the
# default stack an ``Assertion_name`` row is an ``IndexSchema`` projection: its ``type`` is
# the literal ``"IndexSchema"`` and its ``text`` is the indexed field's *value* -- the
# proposition -- not a passage. Read as node properties, the first vetoes the assertion
# check (``is_assertion_props`` rejects a ``type`` that names no assertion class) and the
# second turns the statement into a chunk-style block, so a denial renders as its
# affirmative proposition with no stance. The unit fakes carried full node payloads and
# never saw this; the default-stack integration test did.
_INDEX_ROW_KEYS = ("type", "text")


def _render_properties(row_properties: dict, graph_properties: dict) -> dict:
    """The properties a seed statement renders from: the graph node, gap-filled by the row.

    The graph node is authoritative for everything it stores (the projection fills every
    whitelisted key, so a ``None`` there means "not stored", not "override with nothing").
    The row supplies what the graph did not return -- a seed the neighborhood call missed,
    or a property the graph adapter does not project -- minus the keys that describe the
    index row itself. A row whose only wording is its indexed ``text`` still names the
    statement when neither side has a ``name``.
    """
    properties = {key: value for key, value in row_properties.items() if key not in _INDEX_ROW_KEYS}
    properties.update({key: value for key, value in graph_properties.items() if value is not None})
    if not properties.get("name"):
        indexed_text = row_properties.get("text")
        if isinstance(indexed_text, str) and indexed_text.strip():
            properties["name"] = indexed_text
    return properties


def format_statements(statements: list[dict]) -> str:
    blocks = []
    for statement in statements or []:
        block = _format_statement(statement)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "## Relevant statements\n" + "\n\n".join(blocks)


def _seed_rows(hits: list[Any]) -> dict[str, dict]:
    """``{id: vector payload}`` for the hits, in rank order, deduplicated.

    A hit with no id is dropped: it can be neither paired nor deduplicated, and the same
    statement would then be at risk of appearing twice in one section.
    """
    seeds: dict[str, dict] = {}
    for hit in hits or []:
        hit_id = result_id(hit)
        if not hit_id or hit_id in seeds:
            continue
        seeds[hit_id] = payload(hit)
    return seeds


def _properties_by_id(nodes: list[Any]) -> dict[str, dict]:
    """``{id: projected properties}`` for every node the expansion returned.

    The list opens with the seed assertions themselves, in the order they were asked for,
    so the seeds' own graph properties come out of the one ``get_neighborhood`` call the
    pairs already needed -- including for a seed that nothing points at, which has no pair
    edge to read its endpoints off.
    """
    properties: dict[str, dict] = {}
    for node in nodes or []:
        attributes = getattr(node, "attributes", None)
        if isinstance(attributes, dict):
            properties.setdefault(str(node.id), attributes)
    return properties


def _pairs_by_seed(
    pair_edges: list[Any],
    seed_ids: set[str],
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> dict[str, list[dict]]:
    """The pair lines each seed earned, deduplicated by pair and in a fixed order.

    A line is identified by ``(relationship, counterpart id)``, not by what it renders as:
    a multi-count complaint denies the same proposition more than once, so two distinct
    allegations can produce the same words and both still have to appear. The rendered
    text decides only for a counterpart with no usable id, which nothing else can tell
    apart.
    """
    pairs: dict[str, list[dict]] = {}
    seen: dict[str, set] = {}

    for edge in pair_edges or []:
        source, target = getattr(edge, "node1", None), getattr(edge, "node2", None)
        if source is None or target is None:
            continue

        relationship = _relationship_name(edge)
        if not relationship:
            continue

        for seed, counterpart, outgoing in ((source, target, True), (target, source, False)):
            seed_id = str(seed.id)
            if seed_id not in seed_ids or seed_id == str(counterpart.id):
                continue
            pair = _pair(
                relationship,
                counterpart,
                edge,
                outgoing=outgoing,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
            )
            if pair is None:
                continue
            identity = (relationship, pair["node_id"] or pair["text"])
            if identity in seen.setdefault(seed_id, set()):
                continue
            seen[seed_id].add(identity)
            pairs.setdefault(seed_id, []).append(pair)

    return {seed_id: sorted(rendered, key=_pair_order) for seed_id, rendered in pairs.items()}


def _pair_order(pair: dict) -> tuple:
    """Fixed relationship order, then the counterpart, then the line itself."""
    relationship = pair.get("relationship") or ""
    return (
        _PAIR_TYPE_ORDER.get(relationship, len(_PAIR_TYPE_ORDER)),
        relationship,
        pair.get("node_id") or "",
        pair.get("text") or "",
    )


def _pair(
    relationship: str,
    counterpart: Any,
    edge: Any,
    *,
    outgoing: bool,
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> Optional[dict]:
    """One pair line, or None when this direction says nothing about the seed.

    An outgoing edge always reads ("responds_to" -> "responds to"), so an unlisted
    relationship still renders; an incoming one reads only from its own table.
    """
    if outgoing:
        label = OUTGOING_PAIR_LABELS.get(relationship, relationship.replace("_", " "))
    else:
        label = INCOMING_PAIR_LABELS.get(relationship)
        if label is None:
            return None

    properties = counterpart.attributes if isinstance(counterpart.attributes, dict) else {}
    if not _counterpart_in_scope(properties, node_name, node_name_filter_operator):
        return None

    counterpart_id = str(counterpart.id)
    # The projection fills every whitelisted key, so ``properties["id"]`` is present but
    # ``None`` whenever the store keeps the id out of its property bag. The label's id
    # fallback reads the node's own id instead of depending on the whitelist for it.
    counterpart_label = node_context_label(
        {**properties, "id": properties.get("id") or counterpart_id}
    )
    if not counterpart_label:
        return None

    return {
        "text": f"{label}: {counterpart_label}{_resolution_note(edge.attributes)}",
        "relationship": relationship,
        "node_id": counterpart_id,
    }


def _counterpart_in_scope(
    properties: dict,
    node_name: Optional[list[str]],
    node_name_filter_operator: str,
) -> bool:
    """Whether a pair counterpart may be rendered under a node-scoped search.

    The same rule, through the same helper, that every other scope check in hybrid
    retrieval applies: a node that does not carry the requested set is out of it, and the
    entity lane drops untagged one-hop neighbours on exactly that basis. There is no
    "cannot tell" case left to make an exception for -- ``expand_assertion_pairs``
    projects ``belongs_to_set`` explicitly, so the key is always present on a counterpart
    it returned, and reading its absence as consent would let a scoped search render
    statements from outside the set it asked for.
    """
    return payload_matches_node_filter(properties, node_name, node_name_filter_operator)


def _relationship_name(edge: Any) -> Optional[str]:
    attributes = edge.attributes if isinstance(edge.attributes, dict) else {}
    return display_value(attributes.get("relationship_type")) or display_value(
        attributes.get("relationship_name")
    )


def _resolution_note(edge_attributes: dict) -> str:
    """How a resolved reference edge reports its answer, or "" for a plain edge.

    Same two values ``resolve_edges_to_text`` appends to a connection line, parenthesized
    rather than bracketed because the counterpart label already carries brackets.
    """
    confidence = display_value(edge_attributes.get("resolution_confidence"))
    if confidence is None:
        return ""

    strategy = display_value(edge_attributes.get("resolution_strategy"))
    return f" (confidence {confidence}, {strategy})" if strategy else f" (confidence {confidence})"


def _format_statement(statement: dict) -> str:
    title = display_value(statement.get("title"))
    if not title:
        return ""

    lines = [f"### {title}"]
    body = display_value(statement.get("body"))
    if body:
        lines.extend(line for line in body.splitlines() if line)
    for pair in statement.get("pairs") or []:
        text = display_value(pair.get("text"))
        if text:
            lines.append(f"  ↳ {text}")
    return "\n".join(lines)
