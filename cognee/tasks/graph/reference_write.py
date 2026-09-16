"""Write a planned reference resolution into the graph: edges first, then node patches.

The last phase of the resolver and the only one that writes. It also owns what the phases
before it say *about* writing -- what a strategy patches on the node, the note that says
every edge is already there, the marks an inferred link carries -- so this module imports
nothing from the planner or the trace pass, and both of them can read it.

Nothing here decides anything: by the time a :class:`Resolution` arrives, what it links
and what it patches were settled by
:mod:`cognee.tasks.graph.resolve_assertion_references` and
:mod:`cognee.tasks.graph.reference_pass`.
"""

from typing import Any, Dict, Optional, Sequence

from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.modules.graph.utils.reference_resolution import (
    STRATEGY_LLM_INFERRED,
    STRATEGY_LLM_TRACE,
    Resolution,
    build_node_patch,
    build_reference_edge,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.reference_graph_view import GraphView, _node_label
from cognee.tasks.storage.index_graph_edges import index_graph_edges

# The resolver's logger name, so splitting the code did not move its log output.
logger = get_logger("resolve_assertion_references")


# ``add_edges`` succeeded but ``index_graph_edges`` did not.
NOTE_EDGE_INDEX_FAILED = "edge_index_failed"


# Every edge this resolution would write is already in the graph; only the node patch is
# still outstanding, so the write phase patches and skips the edge upsert.
NOTE_EDGES_EXIST = "edges_exist"


# What the write phase may put back on the node. Only the default table: a negative record
# overrides it to ``resolution_only``, so a field the extraction wrote is never nulled out.
PATCH_NONE = "none"
PATCH_FULL = "full"
PATCH_RESOLUTION_ONLY = "resolution_only"
_PATCHED_STRATEGIES = frozenset({STRATEGY_LLM_TRACE})

# An inferred link is marked and weighted low on the edge itself, so a reader can tell it
# from a relationship a document wrote (``cross_connect_entities.py`` does the same).
INFERRED_EDGE_FEEDBACK_WEIGHT = 0.2


def _default_patch_mode(strategy: str) -> str:
    """What a strategy patches unless the resolution says otherwise."""
    return PATCH_FULL if strategy in _PATCHED_STRATEGIES else PATCH_NONE


def inferred_edge_properties(strategy: str) -> Optional[Dict[str, Any]]:
    """The marks an inferred reference edge carries into the graph, or ``None``.

    ``ensure_default_edge_properties`` only fills a ``feedback_weight`` that is *absent*,
    so the low weight set here is what reaches storage.
    """
    if strategy != STRATEGY_LLM_INFERRED:
        return None
    return {"inferred": True, "feedback_weight": INFERRED_EDGE_FEEDBACK_WEIGHT}


async def write_resolutions(
    graph_engine,
    view: GraphView,
    resolutions: Sequence[Resolution],
    *,
    provenance_kwargs: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Write the planned resolutions: edges first, then the node patches.

    Edges come first because a patch points the field at a node the edges must already
    reach. ``add_edges`` upserts on ``(source, target, relationship)``, so re-emitting an
    edge cannot duplicate it -- but the upsert also overwrites that edge's stored
    properties, so a resolution the planner marked :data:`NOTE_EDGES_EXIST` writes no edge
    at all and is patched only.

    Indexing the new edge texts is the one step allowed to fail on its own: the edges are
    already stored, so the patches still run and the failure comes back as the
    ``edge_index_failed`` note. Nothing retries it, so the note means an operator has to
    re-index -- ``index_graph_edges()`` with no argument rescans the graph.
    """
    summary = {
        "edges_written": 0,
        "nodes_patched": 0,
        "already_resolved": 0,
        "dry_run": bool(dry_run),
        "notes": [],
    }
    if not resolutions:
        return summary

    edges = []
    endpoints: Dict[str, dict] = {}
    for resolution in resolutions:
        if NOTE_EDGES_EXIST in resolution.notes:
            continue

        target_ids = list(resolution.target_ids)
        if resolution.anchor_id and resolution.anchor_id not in target_ids:
            target_ids.append(resolution.anchor_id)
        if not target_ids:
            # An abstention, a below-threshold answer or an inferred-but-unlinked record:
            # audited on the node, never an edge.
            continue

        source_props = view.assertions.get(resolution.assertion_id, {})
        endpoints[resolution.assertion_id] = source_props

        for target_id in target_ids:
            target_props = view.node_props(target_id)
            endpoints[target_id] = target_props
            edges.append(
                build_reference_edge(
                    resolution,
                    target_id,
                    source_props=source_props,
                    target_label=_node_label(target_props),
                    target_type=target_props.get("type") or resolution.target_type or "Node",
                    # An inferred link is marked and weighted down on the edge itself, so
                    # a reader can tell it from a reference the document wrote.
                    extra_properties=inferred_edge_properties(resolution.strategy),
                )
            )

    if dry_run:
        logger.info(
            "Reference resolution dry_run: %d edge(s) and %d patch(es) withheld.",
            len(edges),
            sum(1 for r in resolutions if r.patch_mode != PATCH_NONE),
        )
        return summary

    if edges:
        edges = ensure_default_edge_properties(edges, nodes=list(endpoints.values()))
        await graph_engine.add_edges(edges, **(provenance_kwargs or {}))
        summary["edges_written"] = len(edges)
        try:
            await index_graph_edges(edges)
        except Exception as error:  # noqa: BLE001 - the edges are stored; patch anyway
            logger.warning(
                "Wrote %d reference edge(s) but could not index their text (%s); the "
                "edges are in the graph and remain traversable, but their text stays out "
                "of the EdgeType_relationship_name collection until index_graph_edges "
                "runs over them again. Nothing does that automatically -- a later "
                "resolver pass finds the edges present and re-emits nothing, and "
                "improve() indexes triplets rather than edge texts -- so re-index "
                "explicitly: index_graph_edges() with no argument rescans the graph.",
                len(edges),
                error,
            )
            summary["notes"].append(NOTE_EDGE_INDEX_FAILED)

    for resolution in resolutions:
        if resolution.patch_mode == PATCH_NONE:
            continue

        values = build_node_patch(
            resolution,
            view.assertions.get(resolution.assertion_id, {}),
            mode=resolution.patch_mode,
        )
        try:
            await graph_engine.update_node(resolution.assertion_id, values)
        except NotImplementedError:
            logger.warning(
                "Graph adapter cannot patch nodes; reference edges were written but the "
                "assertion fields still hold their reference text."
            )
            summary["notes"].append("node_patch_unsupported")
            summary["nodes_patched"] = 0
            # Nothing was left to do for a patch-only resolution, and nothing could be
            # done: the graph already holds its edges, so it counts as already resolved.
            summary["already_resolved"] = sum(
                1 for planned in resolutions if NOTE_EDGES_EXIST in planned.notes
            )
            break
        summary["nodes_patched"] += 1

    logger.info(
        "Reference resolution wrote %d edge(s) and patched %d node(s).",
        summary["edges_written"],
        summary["nodes_patched"],
    )
    return summary


def _merge_write_summary(summary: Dict[str, Any], write_summary: Dict[str, Any]) -> None:
    """Fold the write phase's counters into the plan's.

    Only ``already_resolved`` adds rather than replaces: a planned resolution that turned
    out to need no write stops being a resolution of this pass. ``notes`` concatenates,
    because the plan's notes and the write's are about different phases.
    """
    written = dict(write_summary)
    already = written.pop("already_resolved", 0)
    notes = list(written.pop("notes", []) or [])
    summary["already_resolved"] = summary.get("already_resolved", 0) + already
    summary["resolved"] = max(0, summary.get("resolved", 0) - already)
    summary.update(written)
    summary["notes"] = list(summary.get("notes") or []) + notes
