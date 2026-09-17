from typing import Any, List, Optional

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.graph.utils.node_context_text import (  # noqa: F401  (re-exported)
    _create_title_from_text,
    _get_top_n_frequent_words,
    node_context_text,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _extract_nodes_from_edges(retrieved_edges: List[Edge]) -> dict:
    """Creates a dictionary of nodes with their names and content."""

    logger.debug(
        "Extracting nodes from retrieved edges",
        extra={"edge_count": len(retrieved_edges)},
    )

    nodes = {}

    for edge in retrieved_edges:
        for node in (edge.node1, edge.node2):
            if node.id in nodes:
                continue

            name, content = node_context_text(node.attributes)
            nodes[node.id] = {"node": node, "name": name, "content": content}

    return nodes


def _resolution_suffix(edge_attributes: dict) -> str:
    """How a resolved reference edge reports the answer it recorded, or "" for a plain edge.

    A reference edge is the one kind of edge a reader may want to second-guess: it was
    matched deterministically or traced by an agent, so the prompt says which, and how
    confident the match was.
    """
    confidence = _display(edge_attributes.get("resolution_confidence"))
    if confidence is None:
        return ""

    strategy = _display(edge_attributes.get("resolution_strategy"))
    return f" [confidence {confidence}, {strategy}]" if strategy else f" [confidence {confidence}]"


def _display(value: Any) -> Optional[str]:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    return None


async def resolve_edges_to_text(retrieved_edges: List[Edge]) -> str:
    """Converts retrieved graph edges into a human-readable string format."""
    if not retrieved_edges:
        return ""

    nodes = _extract_nodes_from_edges(retrieved_edges)

    node_section = "\n".join(
        f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
        for info in nodes.values()
    )

    connections = []

    logger.debug(
        "Resolving edges to text",
        extra={"edge_count": len(retrieved_edges)},
    )

    for edge in retrieved_edges:
        source_name = nodes[edge.node1.id]["name"]
        target_name = nodes[edge.node2.id]["name"]
        edge_label = (
            edge.attributes.get("relationship_type")
            or edge.attributes.get("relationship_name")
            or edge.attributes.get("edge_text")
        )

        line = f"{source_name} --[{edge_label}]--> {target_name}"

        description = edge.attributes.get("edge_text")
        if description and description != edge_label:
            line += f"  ({description})"

        connections.append(line + _resolution_suffix(edge.attributes))

    connection_section = "\n".join(connections)

    logger.info(
        "Completed resolving edges to text",
        extra={
            "node_count": len(nodes),
            "connection_count": len(connections),
        },
    )

    return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"
