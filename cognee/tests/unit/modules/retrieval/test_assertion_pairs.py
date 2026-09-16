"""
Unit Tests: assertion pair expansion in graph recall

A retrieved denial used to arrive alone. Its ``responds_to`` edge reached a completion
context only when the triplet search happened to rank that edge, so the model saw a stance
with nothing to attach it to -- or, worse, read the denial's affirmative ``name`` as an
independent claim.

These tests pin three things. That every assertion in an edge list pulls its pair edges in
before that list is rendered -- and that this happens late enough to survive the session
path's two-lane ``merge_ranked(..., limit=top_k)``, which truncates anything appended past
``top_k``. That the counterpart nodes and pair edges carry exactly the properties the
retrieval projection would have given them, and only edges that actually touch a retrieved
assertion. And that a graph with no assertions in it never asks the adapter for a
neighborhood, nor even resolves a graph engine.
"""

from types import MappingProxyType, SimpleNamespace
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
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    DEFAULT_EDGE_PROPERTIES_TO_PROJECT,
    default_node_properties_to_project,
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


class Endpoint:
    """An edge endpoint that is not a ``Node`` -- what a foreign adapter could hand over."""

    def __init__(self, node_id, attributes):
        self.id = node_id
        self.attributes = attributes


def _node(props: dict) -> Node:
    return Node(props["id"], {key: value for key, value in props.items()})


def _edge(node1: Node, node2: Node, attributes: dict) -> Edge:
    return Edge(node1, node2, attributes=dict(attributes))


def _neighborhood_row(props: dict) -> tuple:
    return props["id"], {key: value for key, value in props.items() if key != "id"}


def _pair_graph(**kwargs) -> RecordingGraph:
    """The adapter for the canonical case: a denial, its allegation, and the edge between."""
    return RecordingGraph(
        nodes=[_neighborhood_row(DENIAL_PROPS), _neighborhood_row(ALLEGATION_PROPS)],
        edges=[("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS))],
        **kwargs,
    )


def _unified(graph):
    unified = MagicMock()
    unified.graph = graph
    unified.vector = MagicMock()
    return unified


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


def test_an_endpoint_whose_attributes_are_any_mapping_still_counts():
    """A read-only mapping is a property bag like any other; only a non-mapping is not."""
    proxied = Endpoint("denial-1", MappingProxyType(dict(DENIAL_PROPS)))

    assert assertion_ids_in([proxied]) == ["denial-1"]
    assert assertion_ids_in([Endpoint("denial-2", "statement_type=denial")]) == []


# ---------------------------------------------------------------------------------------
# expand_assertion_pairs
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expansion_asks_for_one_hop_over_the_pair_edge_types():
    graph = _pair_graph()

    nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert graph.calls == [
        {
            "node_ids": ["denial-1"],
            "depth": 1,
            "edge_types": ["responds_to", "attributed_to", "asserted_by"],
        }
    ]
    assert PAIR_EDGE_TYPES == ("responds_to", "attributed_to", "asserted_by")
    # Seeds first, then the counterparts the kept edges reached.
    assert [node.id for node in nodes] == ["denial-1", "allegation-1"]
    assert len(edges) == 1
    assert edges[0].node1.id == "denial-1"
    assert edges[0].node2.id == "allegation-1"


@pytest.mark.asyncio
async def test_pair_edges_carry_the_stance_sentence_and_the_resolution_details():
    graph = _pair_graph()

    _nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    attributes = edges[0].attributes
    assert attributes["relationship_type"] == "responds_to"
    assert attributes["relationship_name"] == "responds_to"
    assert attributes["edge_text"] == "Defendants denies that Adams breached the lease."
    assert attributes["resolution_confidence"] == 0.95
    assert attributes["resolution_strategy"] == "paragraph_locator"


@pytest.mark.asyncio
async def test_counterpart_nodes_keep_the_properties_the_renderer_needs():
    graph = _pair_graph()

    nodes, _edges = await expand_assertion_pairs(graph, ["denial-1"])

    counterpart = nodes[1]
    assert counterpart.id == "allegation-1"
    assert counterpart.attributes["statement_type"] == "allegation"
    assert counterpart.attributes["polarity"] == "positive"
    assert counterpart.attributes["asserted_by"] == "Plaintiff"


@pytest.mark.asyncio
async def test_a_counterpart_carries_nothing_outside_the_projection_whitelist():
    """Every other path reaches a renderer as a projection; this one has to match it.

    ``get_neighborhood`` returns the whole stored property bag, and node attributes are
    reachable from a search result, so an unfiltered counterpart would ship a document's
    storage location into a context that the projection would never have put it in.
    """
    graph = RecordingGraph(
        nodes=[
            _neighborhood_row(DENIAL_PROPS),
            _neighborhood_row(
                {
                    **ALLEGATION_PROPS,
                    "raw_data_location": "/Users/someone/.cognee/data/complaint.pdf",
                    "ontology_valid": False,
                    "belongs_to_set": ["KEN"],
                }
            ),
        ],
        edges=[
            (
                "denial-1",
                "allegation-1",
                "responds_to",
                {**RESPONDS_TO_PROPS, "raw_data_location": "/Users/someone/private.pdf"},
            )
        ],
    )

    nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    allowed_node_keys = {
        *default_node_properties_to_project(),
        "feedback_weight",
        # The scope key is carried on purpose: a caller filtering an out-of-scope
        # counterpart out of a node-scoped search has nothing else to filter on.
        "belongs_to_set",
        "vector_distance",
    }
    allowed_edge_keys = {
        *DEFAULT_EDGE_PROPERTIES_TO_PROJECT,
        "feedback_weight",
        "relationship_type",
        "vector_distance",
    }

    for attributes in (nodes[1].attributes, edges[0].node2.attributes):
        assert "raw_data_location" not in attributes
        assert "ontology_valid" not in attributes
        assert set(attributes) <= allowed_node_keys
        # The properties the renderer and the scope filter need survive the whitelist.
        assert attributes["statement_type"] == "allegation"
        assert attributes["belongs_to_set"] == ["KEN"]

    assert "raw_data_location" not in edges[0].attributes
    assert set(edges[0].attributes) <= allowed_edge_keys


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
async def test_an_edge_between_two_neighbors_is_not_a_pair_of_the_seed():
    """The adapter returns every edge among the nodes it kept, seed-touching or not.

    A ``responds_to`` between two of the denial's neighbours is somebody else's pair: it
    must neither join the context nor mint a counterpart node of its own.
    """
    graph = RecordingGraph(
        nodes=[
            _neighborhood_row(DENIAL_PROPS),
            _neighborhood_row(ALLEGATION_PROPS),
            _neighborhood_row({**ALLEGATION_PROPS, "id": "allegation-2"}),
            _neighborhood_row({**DENIAL_PROPS, "id": "denial-2"}),
        ],
        edges=[
            ("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS)),
            ("denial-2", "allegation-2", "responds_to", {"relationship_name": "responds_to"}),
        ],
    )

    nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert [(edge.node1.id, edge.node2.id) for edge in edges] == [("denial-1", "allegation-1")]
    assert [node.id for node in nodes] == ["denial-1", "allegation-1"]


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

    assert [node.id for node in nodes] == ["denial-1"]
    assert edges == []


@pytest.mark.asyncio
async def test_malformed_rows_are_skipped_and_reported():
    graph = RecordingGraph(
        nodes=[
            _neighborhood_row(DENIAL_PROPS),
            ("too-short",),
            ("no-properties", "denial"),
            _neighborhood_row(ALLEGATION_PROPS),
        ],
        edges=[
            ("denial-1", "allegation-1"),
            ("denial-1", "allegation-1", "responds_to", dict(RESPONDS_TO_PROPS)),
        ],
    )

    with patch.object(assertion_pairs, "logger", MagicMock()) as logger:
        nodes, edges = await expand_assertion_pairs(graph, ["denial-1"])

    assert [node.id for node in nodes] == ["denial-1", "allegation-1"]
    assert len(edges) == 1
    assert logger.debug.call_count == 3


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
    graph = _pair_graph()

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
    graph = _pair_graph()

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
async def test_an_engine_provider_is_called_only_once_an_assertion_is_found():
    """The provider exists so a plain graph never builds an engine to expand nothing."""
    graph = _pair_graph()
    provider = AsyncMock(return_value=graph)

    plain = [_edge(_node({"id": "alice"}), _node({"id": "acme"}), {})]
    assert await append_assertion_pair_edges(provider, plain) is plain
    assert provider.await_count == 0

    with_assertion = [_edge(_node(DENIAL_PROPS), _node({"id": "doc-1"}), {})]
    expanded = await append_assertion_pair_edges(provider, with_assertion)
    assert provider.await_count == 1
    assert len(expanded) == 2


@pytest.mark.asyncio
async def test_a_provider_that_fails_degrades_to_the_retrieved_edges():
    provider = AsyncMock(side_effect=RuntimeError("no database in this context"))
    triplets = [_edge(_node(DENIAL_PROPS), _node({"id": "doc-1"}), {})]

    with patch.object(assertion_pairs, "logger", MagicMock()) as logger:
        assert await append_assertion_pair_edges(provider, triplets) is triplets

    assert logger.warning.call_count == 1


@pytest.mark.asyncio
async def test_the_feature_flag_switches_the_expansion_off():
    """Task 7: the switch moved onto RetrievalConfig.graph_completion_pair_expansion."""
    triplets = [_edge(_node(DENIAL_PROPS), _node({"id": "doc-1", "name": "Complaint"}), {})]
    graph = _pair_graph()
    disabled_config = SimpleNamespace(graph_completion_pair_expansion=False)

    with patch.object(assertion_pairs, "get_retrieval_config", return_value=disabled_config):
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


def _patched_retriever(graph, triplets=None, top_k=5):
    """A retriever whose graph, triplet search and preference lookup are all local."""
    retriever = GraphCompletionRetriever(top_k=top_k)
    unified = AsyncMock(return_value=_unified(graph))
    patches = [
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_unified_engine",
            new=unified,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.load_preference_weights",
            new=AsyncMock(return_value={}),
        ),
    ]
    if triplets is not None:
        patches.append(
            patch(
                "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
                new=AsyncMock(return_value=triplets),
            )
        )
    return retriever, unified, patches


@pytest.mark.asyncio
async def test_a_retrieved_denial_arrives_with_the_allegation_it_answers():
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    triplets = [_edge(denial, complaint, {"relationship_name": "mentioned_in"})]
    graph = _pair_graph()

    retriever, _unified_engine, patches = _patched_retriever(graph, triplets)
    with patches[0], patches[1], patches[2]:
        retrieved = await retriever.get_retrieved_objects(query="who denied the breach?")
        context = await retriever.get_context_from_objects(
            query="who denied the breach?", retrieved_objects=retrieved
        )

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
async def test_the_lane_merge_cannot_truncate_the_pair_edge_away():
    """The session path merges two retrievals down to ``top_k`` before rendering.

    This is why the expansion hangs off rendering rather than off retrieval: appending to
    what ``get_retrieved_objects`` returned put the pair edge at an index the merge cuts.
    """
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    answer = _node({"id": "doc-2", "name": "Answer"})
    graph = _pair_graph()

    raw_lane = [
        _edge(denial, complaint, {"relationship_name": "mentioned_in"}),
        _edge(denial, answer, {"relationship_name": "mentioned_in"}),
    ]
    conversational_lane = [_edge(complaint, answer, {"relationship_name": "precedes"})]

    retriever, unified_engine, patches = _patched_retriever(graph, top_k=2)

    # What expanding before the merge did: the pair edge lands past top_k and is cut.
    expanded_first = await append_assertion_pair_edges(graph, raw_lane)
    assert len(expanded_first) == 3
    truncated = retriever.merge_retrieved_objects(expanded_first, conversational_lane)
    assert len(truncated) == 2
    assert all(edge.attributes.get("relationship_type") != "responds_to" for edge in truncated)

    # What happens now: merge first, expand while rendering.
    merged = retriever.merge_retrieved_objects(raw_lane, conversational_lane)
    assert len(merged) == 2
    with patches[0], patches[1]:
        context = await retriever.get_context_from_objects(
            query="who denied the breach?", retrieved_objects=merged
        )

    assert "--[responds_to]-->" in context
    assert "[allegation by Plaintiff; stance: positive] Adams breached the lease" in context


@pytest.mark.asyncio
async def test_a_caller_that_only_renders_edges_still_gets_the_expansion():
    """COT, context extension and temporal call resolve_edges_to_text on their own.

    None of them set ``_unified_engine``, so the expansion has to resolve a graph engine
    for itself -- and does, exactly once, only because an assertion is present.
    """
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    graph = _pair_graph()

    retriever, unified_engine, patches = _patched_retriever(graph)
    assert getattr(retriever, "_unified_engine", None) is None

    with patches[0], patches[1]:
        context = await retriever.resolve_edges_to_text(
            [_edge(denial, complaint, {"relationship_name": "mentioned_in"})]
        )

    assert unified_engine.await_count == 1
    assert len(graph.calls) == 1
    assert "--[responds_to]-->" in context


@pytest.mark.asyncio
async def test_pair_edges_are_context_only():
    """They are context the renderer added, not objects the retrieval ranked.

    So the session's used graph elements and the structured evidence keep describing what
    was actually retrieved -- the counterpart is in the prompt, not in that ledger.
    """
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    triplets = [_edge(denial, complaint, {"relationship_name": "mentioned_in"})]
    graph = _pair_graph()

    retriever, _unified_engine, patches = _patched_retriever(graph, triplets)
    with patches[0], patches[1], patches[2]:
        retrieved = await retriever.get_retrieved_objects(query="who denied the breach?")
        # Retrieval alone never expands: the objects it returns are what it ranked.
        assert retrieved is triplets
        assert graph.calls == []

        context = await retriever.get_context_from_objects(
            query="who denied the breach?", retrieved_objects=retrieved
        )

    context_ids = retriever.extract_context_object_ids(retrieved) or {}
    assert "allegation-1" not in [str(value) for values in context_ids.values() for value in values]
    assert "--[responds_to]-->" in context


@pytest.mark.asyncio
async def test_a_plain_graph_renders_exactly_as_it_did_without_expansion():
    alice = _node({"id": "alice", "name": "Alice", "description": "Alice works at Acme."})
    acme = _node({"id": "acme", "name": "Acme", "description": "A company."})
    triplets = [_edge(alice, acme, {"relationship_name": "works_at"})]
    graph = RecordingGraph()

    retriever, unified_engine, patches = _patched_retriever(graph, triplets)
    with patches[0], patches[1], patches[2]:
        retrieved = await retriever.get_retrieved_objects(query="where does alice work?")
        context = await retriever.get_context_from_objects(
            query="where does alice work?", retrieved_objects=retrieved
        )

    assert retrieved is triplets
    assert graph.calls == []
    # Retrieval resolved the engine; rendering did not have to resolve another one.
    assert unified_engine.await_count == 1
    assert context == await resolve_edges_to_text(triplets)


@pytest.mark.asyncio
async def test_batched_rendering_expands_every_lane_that_has_an_assertion():
    denial = _node(DENIAL_PROPS)
    complaint = _node({"id": "doc-1", "name": "Complaint"})
    alice = _node({"id": "alice", "name": "Alice"})
    acme = _node({"id": "acme", "name": "Acme"})
    lanes = [
        [_edge(denial, complaint, {"relationship_name": "mentioned_in"})],
        [_edge(alice, acme, {"relationship_name": "works_at"})],
    ]
    graph = _pair_graph()

    retriever, _unified_engine, patches = _patched_retriever(graph, lanes)
    with patches[0], patches[1], patches[2]:
        retrieved = await retriever.get_retrieved_objects(query_batch=["denial?", "employer?"])
        contexts = await retriever.get_context_from_objects(
            query_batch=["denial?", "employer?"], retrieved_objects=retrieved
        )

    assert retrieved is lanes
    assert [call["node_ids"] for call in graph.calls] == [["denial-1"]]
    assert "--[responds_to]-->" in contexts[0]
    assert contexts[1] == await resolve_edges_to_text(lanes[1])
