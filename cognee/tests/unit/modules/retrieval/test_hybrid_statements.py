"""
Unit Tests: the hybrid statements lane

``HYBRID_COMPLETION`` is the default search type everywhere (router, DTOs, CLI, UI) and it
searched ``Entity_name`` only -- so on a legal graph, whose statements are ``Assertion``
nodes indexed in ``Assertion_name``, the statements never reached the context as seeds. They
showed up, if at all, as anonymous one-hop neighbours of an entity hit.

These tests pin the new lane: it searches ``Assertion_name``, pulls each hit's pair edges
(``responds_to`` / ``attributed_to`` / ``asserted_by``) in one graph call, renders each
statement with the shared node renderer, and emits a ``## Relevant statements`` section
between the passages and the entities. And they pin the other half of the contract: a graph
with no assertions in it produces a byte-identical context and never grows a key.

No LLM, no vector store, no graph backend, no network.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.retrieval.hybrid import statements as statements_module
from cognee.modules.retrieval.hybrid.context import format_hybrid_context
from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results
from cognee.modules.retrieval.hybrid.statements import (
    _counterpart_in_scope,
    build_statements,
    format_statements,
    search_statements,
)
from cognee.modules.retrieval.hybrid_retriever import DEFAULT_STATEMENTS_TOP_K, HybridRetriever
from cognee.modules.search.types import SearchType
from cognee.tests.unit.tasks.graph._reference_fakes import FakeVectorEngine, nid, scored

QUERY_VECTOR = [0.1, 0.2, 0.3]

DENIAL = {
    "id": "denial-1",
    "name": "Adams breached the lease",
    "statement_type": "denial",
    "polarity": "negative",
    "asserted_by": "Defendants",
    "source_quote": "Defendants deny each and every allegation of paragraph 17.",
    "source_quote_verified": True,
}

ALLEGATION = {
    "id": "allegation-1",
    "name": "Adams breached the lease",
    "statement_type": "allegation",
    "polarity": "positive",
    "asserted_by": "Plaintiff",
}

SPEAKER = {"id": "defendants-1", "name": "Defendants", "type": "Person"}

RESOLUTION = {"resolution_confidence": 0.95, "resolution_strategy": "paragraph_locator"}


def _hit(payload: dict):
    hit = MagicMock()
    hit.id = payload["id"]
    hit.payload = payload
    return hit


def _node_row(props: dict):
    return (props["id"], {key: value for key, value in props.items() if key != "id"})


class _FakeGraph:
    """Only what the lane asks of a graph adapter, and deliberately not callable.

    ``expand_assertion_pairs`` accepts either an engine or a provider callable, so a
    ``MagicMock()`` would be taken for the latter and awaited.
    """

    def __init__(self, nodes=None, edges=None, error: Exception = None):
        self.get_neighborhood = AsyncMock(
            side_effect=error, return_value=None if error else (nodes or [], edges or [])
        )


def _graph(nodes=None, edges=None):
    return _FakeGraph(nodes, edges)


def _pair_graph():
    """The denial, the allegation it answers, and the speaker behind it."""
    return _graph(
        nodes=[_node_row(DENIAL), _node_row(ALLEGATION), _node_row(SPEAKER)],
        edges=[
            ("denial-1", "allegation-1", "responds_to", dict(RESOLUTION)),
            ("denial-1", "defendants-1", "asserted_by", {}),
        ],
    )


# ---------------------------------------------------------------------------------------
# search_statements
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_statements_queries_the_assertion_collection():
    denial_id = nid("denial-1")
    row = scored(denial_id, 0.1, **{**DENIAL, "id": denial_id})
    vector = FakeVectorEngine({"Assertion_name": [row]})

    hits = await search_statements(vector, "q", 20, None, "OR", QUERY_VECTOR)

    assert [str(hit.id) for hit in hits] == [denial_id]
    assert vector.search_calls == [("Assertion_name", None, 20, True)]


@pytest.mark.asyncio
async def test_missing_assertion_collection_yields_no_statements():
    vector = FakeVectorEngine({}, raising_collections=["Assertion_name"])

    assert await search_statements(vector, "q", 20, None, "OR", QUERY_VECTOR) == []


@pytest.mark.asyncio
async def test_search_statements_survives_an_adapter_error():
    vector = MagicMock()
    vector.search = AsyncMock(side_effect=RuntimeError("adapter down"))
    logger = MagicMock()

    with patch.object(statements_module, "logger", logger):
        assert await search_statements(vector, "q", 20, None, "OR", QUERY_VECTOR) == []

    assert logger.warning.call_count == 1


@pytest.mark.asyncio
async def test_search_statements_passes_the_node_filter_and_the_query_vector():
    vector = MagicMock()
    vector.search = AsyncMock(return_value=[])

    await search_statements(vector, "q", 7, ["KEN"], "AND", QUERY_VECTOR)

    call = vector.search.await_args
    assert call.args[:2] == ("Assertion_name", None)
    assert call.kwargs["query_vector"] == QUERY_VECTOR
    assert call.kwargs["limit"] == 7
    assert call.kwargs["node_name"] == ["KEN"]
    assert call.kwargs["node_name_filter_operator"] == "AND"


@pytest.mark.asyncio
async def test_collection_not_found_is_swallowed_by_the_shared_search():
    vector = MagicMock()
    vector.search = AsyncMock(side_effect=CollectionNotFoundError("missing"))

    assert await search_statements(vector, "q", 20, None, "OR", QUERY_VECTOR) == []


# ---------------------------------------------------------------------------------------
# build_statements
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_statements_renders_the_stance_and_its_pairs():
    graph = _pair_graph()

    built = await build_statements(graph, [_hit(DENIAL)])

    assert len(built) == 1
    statement = built[0]
    assert statement["id"] == "denial-1"
    assert statement["title"] == "[denial by Defendants; stance: negative] Adams breached the lease"
    assert statement["body"] == (
        "Defendants denies that Adams breached the lease.\n"
        'Quote: "Defendants deny each and every allegation of paragraph 17." (verified)'
    )
    assert [pair["text"] for pair in statement["pairs"]] == [
        "responds to: [allegation/positive] Adams breached the lease "
        "(confidence 0.95, paragraph_locator)",
        "speaker: Defendants",
    ]


@pytest.mark.asyncio
async def test_pairs_are_pulled_in_one_graph_call_for_every_hit():
    graph = _pair_graph()

    await build_statements(graph, [_hit(DENIAL), _hit(ALLEGATION)])

    graph.get_neighborhood.assert_awaited_once()
    call = graph.get_neighborhood.await_args
    assert call.args[0] == ["denial-1", "allegation-1"]
    assert call.kwargs["depth"] == 1
    assert call.kwargs["edge_types"] == ["responds_to", "attributed_to", "asserted_by"]


@pytest.mark.asyncio
async def test_an_incoming_responds_to_reads_as_answered_by():
    graph = _pair_graph()

    built = await build_statements(graph, [_hit(ALLEGATION)])

    assert [pair["text"] for pair in built[0]["pairs"]] == [
        "answered by: [denial/negative] Adams breached the lease "
        "(confidence 0.95, paragraph_locator)"
    ]


@pytest.mark.asyncio
async def test_a_statement_without_pairs_renders_from_the_vector_row():
    built = await build_statements(_graph(), [_hit(DENIAL)])

    assert built[0]["title"] == "[denial by Defendants; stance: negative] Adams breached the lease"
    assert built[0]["pairs"] == []


@pytest.mark.asyncio
async def test_a_graph_failure_degrades_to_statements_without_pairs():
    graph = _FakeGraph(error=RuntimeError("no neighborhood support"))

    built = await build_statements(graph, [_hit(DENIAL)])

    assert built[0]["pairs"] == []
    assert built[0]["body"].startswith("Defendants denies that Adams breached the lease.")


@pytest.mark.asyncio
async def test_graph_properties_fill_in_what_the_vector_row_does_not_carry():
    """A legacy row indexed before a property existed still renders with it."""
    narrow_row = {"id": "denial-1", "name": "Adams breached the lease", "statement_type": "denial"}

    built = await build_statements(_pair_graph(), [_hit(narrow_row)])

    assert built[0]["title"] == "[denial by Defendants; stance: negative] Adams breached the lease"


@pytest.mark.asyncio
async def test_no_hits_asks_the_graph_for_nothing():
    graph = _pair_graph()

    assert await build_statements(graph, []) == []
    graph.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_hits_are_rendered_once():
    built = await build_statements(_pair_graph(), [_hit(DENIAL), _hit(DENIAL)])

    assert [statement["id"] for statement in built] == ["denial-1"]


@pytest.mark.asyncio
async def test_non_pair_edges_between_the_same_nodes_are_not_rendered():
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(ALLEGATION)],
        edges=[("denial-1", "allegation-1", "contradicts", {})],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert built[0]["pairs"] == []


@pytest.mark.asyncio
async def test_pair_line_without_a_strategy_reports_the_confidence_alone():
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(ALLEGATION)],
        edges=[("denial-1", "allegation-1", "responds_to", {"resolution_confidence": 0.5})],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert built[0]["pairs"][0]["text"].endswith("(confidence 0.5)")


def test_an_explicitly_out_of_scope_counterpart_is_dropped():
    assert _counterpart_in_scope({"belongs_to_set": ["OTHER"]}, ["KEN"], "OR") is False
    assert _counterpart_in_scope({"belongs_to_set": ["KEN"]}, ["KEN"], "OR") is True
    assert _counterpart_in_scope({"belongs_to_set": ["KEN"]}, ["KEN", "OTHER"], "AND") is False


def test_an_unjudgeable_counterpart_stays():
    """The seed passed the store's filter; an untagged counterpart cannot be judged.

    The graph projection does not ask for ``belongs_to_set``, so this is the normal case
    rather than the exception -- and dropping it would lose the other half of the pair.
    """
    assert _counterpart_in_scope({}, ["KEN"], "OR") is True
    assert _counterpart_in_scope({"belongs_to_set": None}, ["KEN"], "OR") is True
    assert _counterpart_in_scope({"belongs_to_set": ["OTHER"]}, None, "OR") is True


@pytest.mark.asyncio
async def test_a_node_scoped_search_still_renders_the_pair():
    built = await build_statements(_pair_graph(), [_hit(DENIAL)], node_name=["KEN"])

    assert [pair["relationship"] for pair in built[0]["pairs"]] == [
        "responds_to",
        "asserted_by",
    ]


@pytest.mark.asyncio
async def test_pairs_carry_the_relationship_and_the_counterpart_id():
    built = await build_statements(_pair_graph(), [_hit(DENIAL)])

    assert built[0]["pairs"][0]["relationship"] == "responds_to"
    assert built[0]["pairs"][0]["node_id"] == "allegation-1"


# ---------------------------------------------------------------------------------------
# format_statements
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_format_statements_renders_the_section():
    built = await build_statements(_pair_graph(), [_hit(DENIAL)])

    assert format_statements(built) == (
        "## Relevant statements\n"
        "### [denial by Defendants; stance: negative] Adams breached the lease\n"
        "Defendants denies that Adams breached the lease.\n"
        'Quote: "Defendants deny each and every allegation of paragraph 17." (verified)\n'
        "  ↳ responds to: [allegation/positive] Adams breached the lease "
        "(confidence 0.95, paragraph_locator)\n"
        "  ↳ speaker: Defendants"
    )


def test_format_statements_of_nothing_is_empty():
    assert format_statements([]) == ""
    assert format_statements(None) == ""


def test_format_statements_separates_blocks_with_a_blank_line():
    section = format_statements(
        [
            {"id": "a", "title": "First", "body": "One.", "pairs": []},
            {"id": "b", "title": "Second", "body": "Two.", "pairs": []},
        ]
    )

    assert section == "## Relevant statements\n### First\nOne.\n\n### Second\nTwo."


# ---------------------------------------------------------------------------------------
# Section order and the no-assertion graph
# ---------------------------------------------------------------------------------------


def test_statements_sit_between_the_passages_and_the_entities():
    context = format_hybrid_context(
        "## Global context\nprelude",
        {
            "chunks": [_hit({"id": "chunk-1", "text": "Passage text"})],
            "statements": [{"id": "denial-1", "title": "Denial", "body": "Denied.", "pairs": []}],
            "entities": [{"id": "entity-1", "name": "Alice", "description": "A.", "edges": []}],
            "facts": [{"id": "fact-1", "text": "Acme acquired Initech."}],
        },
    )

    assert [line for line in context.splitlines() if line.startswith("## ")] == [
        "## Global context",
        "## Relevant passages",
        "## Relevant statements",
        "## Relevant entities",
        "## Related facts",
    ]


def test_a_context_without_statements_is_byte_identical():
    objects = {
        "chunks": [_hit({"id": "chunk-1", "text": "Passage text"})],
        "entities": [{"id": "entity-1", "name": "Alice", "description": "A.", "edges": []}],
        "facts": [],
    }

    assert format_hybrid_context("", objects) == format_hybrid_context(
        "", {**objects, "statements": []}
    )
    assert "Relevant statements" not in format_hybrid_context("", objects)


def test_merge_carries_the_primary_statements_and_adds_no_channel():
    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        {"chunks": [], "entities": [], "facts": []},
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert merged["statements"] == [{"id": "denial-1"}]
    assert set(merged) == {"chunks", "chunk_summaries", "entities", "facts", "statements"}


# ---------------------------------------------------------------------------------------
# HybridRetriever wiring
# ---------------------------------------------------------------------------------------


def _vector(assertions=None, entities=None, chunks=None, statements_error=False):
    async def search(collection_name, *args, **kwargs):
        if collection_name == "Assertion_name":
            if statements_error:
                raise CollectionNotFoundError("missing")
            return assertions or []
        if collection_name == "Entity_name":
            return entities or []
        if collection_name == "DocumentChunk_text":
            return chunks or []
        return []

    vector = MagicMock()
    vector.search = AsyncMock(side_effect=search)
    return vector


def _unified(vector, graph):
    graph.is_empty = AsyncMock(return_value=False)
    unified = MagicMock()
    unified.vector = vector
    unified.vector.embedding_engine.embed_text = AsyncMock(return_value=[QUERY_VECTOR])
    unified.graph = graph
    return unified


def _patch_engine(vector, graph):
    return patch(
        "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_unified(vector, graph),
    )


@pytest.mark.asyncio
async def test_the_retriever_renders_the_statements_section():
    vector = _vector(assertions=[_hit(DENIAL)])
    retriever = HybridRetriever()

    with _patch_engine(vector, _pair_graph()):
        retrieved = await retriever.get_retrieved_objects(query="who denied the breach?")
        context = await retriever.get_context_from_objects(
            query="who denied the breach?", retrieved_objects=retrieved
        )

    assert [statement["id"] for statement in retrieved["statements"]] == ["denial-1"]
    assert "## Relevant statements" in context
    assert "  ↳ responds to: [allegation/positive] Adams breached the lease" in context


@pytest.mark.asyncio
async def test_a_graph_without_assertions_grows_no_statements_key():
    vector = _vector(
        chunks=[_hit({"id": "chunk-1", "text": "Passage text"})],
        entities=[_hit({"id": "entity-1", "name": "Alice"})],
    )
    graph = _graph(nodes=[("entity-1", {"name": "Alice"})])
    retriever = HybridRetriever()

    with _patch_engine(vector, graph):
        retrieved = await retriever.get_retrieved_objects(query="q")
        context = await retriever.get_context_from_objects(query="q", retrieved_objects=retrieved)

    assert "statements" not in retrieved
    assert context == "## Relevant passages\nPassage text\n\n## Relevant entities\n### Alice"


@pytest.mark.asyncio
async def test_a_missing_assertion_collection_is_not_an_error():
    vector = _vector(entities=[_hit({"id": "entity-1", "name": "Alice"})], statements_error=True)
    retriever = HybridRetriever()

    with _patch_engine(vector, _graph(nodes=[("entity-1", {"name": "Alice"})])):
        retrieved = await retriever.get_retrieved_objects(query="q")

    assert "statements" not in retrieved
    assert retrieved["entities"][0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_the_statements_lane_spends_its_own_top_k():
    vector = _vector()
    retriever = HybridRetriever(statements_top_k=3)

    with _patch_engine(vector, _graph()):
        await retriever.get_retrieved_objects(query="q")

    call = next(
        call for call in vector.search.await_args_list if call.args[:1] == ("Assertion_name",)
    )
    assert call.kwargs["limit"] == 3


def test_the_statements_lane_top_k_defaults():
    assert HybridRetriever().statements_top_k == DEFAULT_STATEMENTS_TOP_K
    assert HybridRetriever(statements_top_k=None).statements_top_k == DEFAULT_STATEMENTS_TOP_K
    assert DEFAULT_STATEMENTS_TOP_K == 20


@pytest.mark.asyncio
async def test_the_statements_lane_runs_concurrently_with_the_entity_lane():
    statements_started = asyncio.Event()
    entities_started = asyncio.Event()

    async def search(collection_name, *args, **kwargs):
        if collection_name == "Assertion_name":
            statements_started.set()
            await entities_started.wait()
            return []
        if collection_name == "Entity_name":
            entities_started.set()
            await statements_started.wait()
            return []
        return []

    vector = MagicMock()
    vector.search = AsyncMock(side_effect=search)
    retriever = HybridRetriever()

    with _patch_engine(vector, _graph()):
        retrieved = await asyncio.wait_for(retriever.get_retrieved_objects(query="q"), timeout=1)

    assert retrieved == {"chunks": [], "chunk_summaries": {}, "entities": [], "facts": []}


# ---------------------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_threads_the_statements_lane_top_k():
    import cognee.modules.search.methods.get_search_type_retriever_instance as mod

    explicit = await mod.get_search_type_retriever_instance(
        SearchType.HYBRID_COMPLETION,
        query_text="q",
        top_k=30,
        retriever_specific_config={"statements_top_k": 4},
    )
    capped = await mod.get_search_type_retriever_instance(
        SearchType.HYBRID_COMPLETION, query_text="q", top_k=30
    )
    unset = await mod.get_search_type_retriever_instance(
        SearchType.HYBRID_COMPLETION, query_text="q", top_k=None
    )

    assert explicit.statements_top_k == 4
    # Task 7: RetrievalConfig.hybrid_statements_top_k (default 20) wins over the
    # request top_k now, so "capped" no longer means min(top_k, 10).
    assert capped.statements_top_k == 20
    assert unset.statements_top_k == DEFAULT_STATEMENTS_TOP_K
