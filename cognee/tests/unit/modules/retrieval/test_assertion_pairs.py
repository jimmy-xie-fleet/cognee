"""
Unit Tests: assertion pair expansion in graph recall

A retrieved denial used to arrive alone. Its ``responds_to`` edge reached a completion
context only when the triplet search happened to rank that edge, so the model saw a stance
with nothing to attach it to -- or, worse, read the denial's affirmative ``name`` as an
independent claim. These tests pin that every assertion in a retrieved triplet list pulls
its pair edges in before the context is rendered, and that a graph with no assertions in it
never asks the adapter for a neighborhood at all.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.graph.utils.resolve_edges_to_text import resolve_edges_to_text
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils import assertion_pairs
from cognee.modules.retrieval.utils.assertion_pairs import (
    PAIR_EDGE_TYPES,
    append_assertion_pair_edges,
    assertion_ids_in,
    expand_assertion_pairs,
)

DENIAL_PROPS = {
    "id": "denial-1",
    "name": "Adams breached the lease",
    "statement_type": "denial",
    "polarity": "negative",
    "asserted_by": "Defendants",
    "source_quote": "Defendants deny each and every allegation of paragraph 17.",
    "source_quote_verified": True,
}

ALLEGATION_PROPS = {
    "id": "allegation-1",
    "name": "Adams breached the lease",
    "statement_type": "allegation",
    "polarity": "positive",
    "asserted_by": "Plaintiff",
}

RESPONDS_TO_PROPS = {
    "relationship_name": "responds_to",
    "edge_text": "Defendants denies that Adams breached the lease.",
    "resolution_confidence": 0.95,
    "resolution_strategy": "paragraph_locator",
    "edge_object_id": "edge-responds-to",
}


class RecordingGraph:
    """A graph adapter that records every neighborhood call and returns canned rows."""

    def __init__(self, nodes=None, edges=None, error: Exception = None):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []
        self.error = error
        self.calls: list[dict] = []

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        self.calls.append({"node_ids": list(node_ids), "depth": depth, "edge_types": edge_types})
        if self.error is not None:
            raise self.error
        return self.nodes, self.edges

    async def is_empty(self):
        return False


def _node(props: dict) -> Node:
    return Node(props["id"], {key: value for key, value in props.items()})


def _edge(node1: Node, node2: Node, attributes: dict) -> Edge:
    return Edge(node1, node2, attributes=dict(attributes))


def _neighborhood_row(props: dict) -> tuple:
    return props["id"], {key: value for key, value in props.items() if key != "id"}


# ---------------------------------------------------------------------------------------
# assertion_ids_in
# ---------------------------------------------------------------------------------------


def test_assertion_ids_in_collects_only_nodes_carrying_a_statement_type():
    nodes = [
        _node(DENIAL_PROPS),
        _node({"id": "entity-1", "name": "Acme"}),
        _node({"id": "blank-1", "name": "Blank", "statement_type": "   "}),
        _node({"id": "listy-1", "name": "Listy", "statement_type": ["denial"]}),
    ]

    assert assertion_ids_in(nodes) == ["denial-1"]


def test_assertion_ids_in_deduplicates_and_tolerates_foreign_objects():
    repeated = [_node(DENIAL_PROPS), _node(DENIAL_PROPS), MagicMock(), None]

    assert assertion_ids_in(repeated) == ["denial-1"]
    assert assertion_ids_in([]) == []


# ---------------------------------------------------------------------------------------
# expand_assertion_pairs
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expansion_asks_for_one_hop_over_the_pair_edge_types():
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert graph.calls == [
        {
            "node_ids": ["denial-1"],
            "depth": 1,
            "edge_types": ["responds_to", "attributed_to", "asserted_by"],
        }
    ]
    assert PAIR_EDGE_TYPES == ("responds_to", "attributed_to", "asserted_by")
    assert [node.id for node in nodes] == ["allegation-1"]
    assert len(edges) == 1
    assert edges[0].node1.id == "denial-1"
    assert edges[0].node2.id == "allegation-1"


@pytest.mark.asyncio
async def test_pair_edges_carry_the_stance_sentence_and_the_resolution_details():
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    _nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    attributes = edges[0].attributes
    assert attributes["relationship_type"] == "responds_to"
    assert attributes["relationship_name"] == "responds_to"
    assert attributes["edge_text"] == "Defendants denies that Adams breached the lease."
    assert attributes["resolution_confidence"] == 0.95
    assert attributes["resolution_strategy"] == "paragraph_locator"


@pytest.mark.asyncio
async def test_counterpart_nodes_keep_the_properties_the_renderer_needs():
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    nodes, _edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert nodes[0].attributes["statement_type"] == "allegation"
    assert nodes[0].attributes["polarity"] == "positive"
    assert nodes[0].attributes["asserted_by"] == "Plaintiff"


@pytest.mark.asyncio
async def test_edges_of_other_types_between_the_same_nodes_are_left_out():
    """The Kuzu adapter type-filters the traversal, not the edges it returns."""
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[
            ("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS)),
            ("denial-1", "allegation-1", "contradicts", {"relationship_name": "contradicts"}),
        ],
    )

    _nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert [edge.attributes["relationship_type"] for edge in edges] == ["responds_to"]


@pytest.mark.asyncio
async def test_internal_nodes_never_become_a_counterpart():
    graph = RecordingGraph(
        nodes=[
            _neighborhood_row(DENIAL_PROPS),
            ("preference-1", {"name": "preference", "is_internal": True}),
        ],
        edges=[("denial-1", "preference-1", "attributed_to", {})],
    )

    nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert nodes == []
    assert edges == []


@pytest.mark.asyncio
async def test_no_assertion_ids_means_no_adapter_call():
    graph = RecordingGraph()

    assert await expand_assertion_pairs(graph, []) == ([], [])
    assert graph.calls == []


@pytest.mark.asyncio
async def test_an_adapter_error_yields_no_pairs_and_warns():
    graph = RecordingGraph(error=RuntimeError("kuzu is unhappy"))

    with patch.object(assertion_pairs, "logger", MagicMock()) as logger:
        nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert (nodes, edges) == ([], [])
    assert logger.warning.call_count == 1


# ---------------------------------------------------------------------------------------
# append_assertion_pair_edges
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pair_edges_are_appended_after_the_retrieved_triplets():
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    triplets = [_edge(denial, complaint, {"relationship_name": "mentioned_in"})]
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    expanded = await append_assertion_pair_edges(graph, triplets)

    assert len(expanded) == 2
    assert expanded[0] is triplets[0]
    assert expanded[1].attributes["relationship_type"] == "responds_to"
    assert triplets == [triplets[0]]


@pytest.mark.asyncio
async def test_a_pair_edge_already_retrieved_is_not_added_twice():
    denial = _node(DENIAL_PROPS)
    allegation = _node(ALLEGATION_PROPS)
    triplets = [_edge(denial, allegation, dict(RESPONDS_TO_PROPS))]
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    expanded = await append_assertion_pair_edges(graph, triplets)

    assert expanded is triplets


@pytest.mark.asyncio
async def test_a_graph_without_assertions_never_reaches_the_adapter():
    alice = _node({"id": "alice", "name": "Alice"})
    acme = _node({"id": "acme", "name": "Acme"})
    triplets = [_edge(alice, acme, {"relationship_name": "works_at"})]
    graph = RecordingGraph()

    expanded = await append_assertion_pair_edges(graph, triplets)

    assert expanded is triplets
    assert graph.calls == []


@pytest.mark.asyncio
async def test_the_feature_flag_switches_the_expansion_off():
    triplets = [_edge(_node(DENIAL_PROPS), _node({"id": "doc-1", "name": "Complaint"}), {})]
    graph = RecordingGraph(
        nodes=[_neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    with patch.object(assertion_pairs, "PAIR_EXPANSION_ENABLED", False):
        expanded = await append_assertion_pair_edges(graph, triplets)

    assert expanded is triplets
    assert graph.calls == []


@pytest.mark.asyncio
async def test_no_edges_and_no_engine_are_both_no_ops():
    assert await append_assertion_pair_edges(RecordingGraph(), []) == []
    triplets = [_edge(_node(DENIAL_PROPS), _node({"id": "doc-1", "name": "Complaint"}), {})]
    assert await append_assertion_pair_edges(None, triplets) is triplets


# ---------------------------------------------------------------------------------------
# GraphCompletionRetriever: the rendered context
# ---------------------------------------------------------------------------------------


def _unified(graph):
    unified = MagicMock()
    unified.graph = graph
    unified.vector = MagicMock()
    return unified


async def _retrieve_and_render(triplets, graph):
    retriever = GraphCompletionRetriever()
    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_unified_engine",
            new=AsyncMock(return_value=_unified(graph)),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new=AsyncMock(return_value=triplets),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.load_preference_weights",
            new=AsyncMock(return_value={}),
        ),
    ):
        retrieved = await retriever.get_retrieved_objects(query="who denied the breach?")
        context = await retriever.get_context_from_objects(
            query="who denied the breach?", retrieved_objects=retrieved
        )
    return retrieved, context


@pytest.mark.asyncio
async def test_a_retrieved_denial_arrives_with_the_allegation_it_answers():
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    retrieved, context = await _retrieve_and_render(
        [_edge(denial, complaint, {"relationship_name": "mentioned_in"})], graph
    )

    assert len(retrieved) == 2
    assert graph.calls[0]["node_ids"] == ["denial-1"]
    assert "[allegation by Plaintiff; stance: positive] Adams breached the lease" in context
    assert (
        "[denial by Defendants; stance: negative] Adams breached the lease "
        "--[responds_to]--> "
        "[allegation by Plaintiff; stance: positive] Adams breached the lease"
        "  (Defendants denies that Adams breached the lease.)"
        " [confidence 0.95, paragraph_locator]"
    ) in context


@pytest.mark.asyncio
async def test_a_plain_graph_renders_exactly_as_it_did_without_expansion():
    alice = _node({"id": "alice", "name": "Alice", "description": "Alice works at Acme."})
    acme = _node({"id": "acme", "name": "Acme", "description": "A company."})
    triplets = [_edge(alice, acme, {"relationship_name": "works_at"})]
    graph = RecordingGraph()

    retrieved, context = await _retrieve_and_render(triplets, graph)

    assert retrieved is triplets
    assert graph.calls == []
    assert context == await resolve_edges_to_text(triplets)


@pytest.mark.asyncio
async def test_batched_retrieval_expands_every_lane():
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    alice = _node({"id": "alice", "name": "Alice"})
    acme = _node({"id": "acme", "name": "Acme"})
    lanes = [
        [_edge(denial, complaint, {"relationship_name": "mentioned_in"})],
        [_edge(alice, acme, {"relationship_name": "works_at"})],
    ]
    graph = RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
    )

    retriever = GraphCompletionRetriever()
    with (
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_unified_engine",
            new=AsyncMock(return_value=_unified(graph)),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
            new=AsyncMock(return_value=lanes),
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.load_preference_weights",
            new=AsyncMock(return_value={}),
        ),
    ):
        retrieved = await retriever.get_retrieved_objects(query_batch=["denial?", "employer?"])

    assert len(retrieved[0]) == 2
    assert retrieved[1] is lanes[1]
    assert [call["node_ids"] for call in graph.calls] == [["denial-1"]]
