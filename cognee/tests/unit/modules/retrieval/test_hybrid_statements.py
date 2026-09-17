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

SECOND_ALLEGATION = {**ALLEGATION, "id": "allegation-2"}

NAMELESS_ALLEGATION = {
    "id": "allegation-9",
    "statement_type": "allegation",
    "polarity": "positive",
}

SPEAKER = {"id": "defendants-1", "name": "Defendants", "type": "Person"}

COUNSEL = {"id": "counsel-1", "name": "Counsel of record", "type": "Person"}

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


PAIR_EDGES = [
    ("denial-1", "allegation-1", "responds_to", dict(RESOLUTION)),
    ("denial-1", "defendants-1", "asserted_by", {}),
]


def _pair_graph():
    """The denial, the allegation it answers, and the speaker behind it."""
    return _graph(
        nodes=[_node_row(DENIAL), _node_row(ALLEGATION), _node_row(SPEAKER)],
        edges=[tuple(edge) for edge in PAIR_EDGES],
    )


def _scoped_pair_graph(node_set: str = "KEN"):
    """The same three nodes, each tagged with the node set a scoped search asks for."""
    return _graph(
        nodes=[
            _node_row({**props, "belongs_to_set": [node_set]})
            for props in (DENIAL, ALLEGATION, SPEAKER)
        ],
        edges=[tuple(edge) for edge in PAIR_EDGES],
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
async def test_a_pair_less_seed_still_reads_its_graph_properties():
    """``expand_assertion_pairs`` returns the seeds themselves, edges or no edges.

    Reading the seeds off that node list rather than off the pair edges' endpoints is what
    lets a statement nothing points at still render its stored stance and speaker.
    """
    narrow_row = {"id": "denial-1", "name": "Adams breached the lease", "statement_type": "denial"}

    built = await build_statements(_graph(nodes=[_node_row(DENIAL)]), [_hit(narrow_row)])

    assert built[0]["pairs"] == []
    assert built[0]["title"] == "[denial by Defendants; stance: negative] Adams breached the lease"


# What LanceDB actually hands back for an ``Assertion_name`` hit on the default stack: an
# IndexSchema projection whose ``text`` is the indexed proposition and whose ``type`` is the
# literal class name of the index row. None of the assertion's own fields are on it.
INDEX_ROW = {
    "id": "denial-1",
    "type": "IndexSchema",
    "text": "Adams breached the lease",
    "belongs_to_set": [],
    "feedback_weight": 0.5,
    "importance_weight": 0.5,
}


@pytest.mark.asyncio
async def test_an_index_row_renders_from_the_graph_node_not_from_itself():
    """The default-stack shape. Read as node properties the row's ``type`` vetoes the
    assertion check and its ``text`` makes a chunk-style block, so a denial rendered as its
    affirmative proposition with no stance -- exactly the defect the lane exists to fix."""
    built = await build_statements(_pair_graph(), [_hit(INDEX_ROW)])

    assert built[0]["title"] == "[denial by Defendants; stance: negative] Adams breached the lease"
    assert built[0]["body"].startswith("Defendants denies that Adams breached the lease.")
    assert "..." not in built[0]["title"]


@pytest.mark.asyncio
async def test_an_index_row_the_graph_did_not_return_still_names_the_statement():
    """No graph node for the seed: the indexed text is the only wording there is."""
    built = await build_statements(_graph(), [_hit(INDEX_ROW)])

    assert built[0]["title"] == "Adams breached the lease"
    assert built[0]["body"] == "Adams breached the lease"


@pytest.mark.asyncio
async def test_the_graph_node_wins_over_a_row_that_disagrees():
    """A row indexed before a property changed must not out-vote the stored node."""
    stale_row = {**DENIAL, "polarity": "positive", "asserted_by": "Nobody"}

    built = await build_statements(_pair_graph(), [_hit(stale_row)])

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


def test_an_untagged_counterpart_is_dropped_under_a_scoped_search():
    """Scope reads here exactly the way it reads everywhere else in hybrid retrieval.

    ``payload_matches_node_filter`` treats a missing ``belongs_to_set`` as "not in the
    requested set", and the entity lane drops untagged one-hop neighbours on that basis.
    The pair expansion projects ``belongs_to_set`` explicitly, so the key is always present
    and "cannot tell" no longer applies -- a counterpart that does not carry the set is out
    of scope, not unjudgeable.
    """
    assert _counterpart_in_scope({}, ["KEN"], "OR") is False
    assert _counterpart_in_scope({"belongs_to_set": None}, ["KEN"], "OR") is False
    # An unscoped search asks nothing of the counterpart.
    assert _counterpart_in_scope({}, None, "OR") is True
    assert _counterpart_in_scope({"belongs_to_set": ["OTHER"]}, None, "OR") is True


@pytest.mark.asyncio
async def test_a_node_scoped_search_renders_an_in_scope_pair():
    built = await build_statements(_scoped_pair_graph(), [_hit(DENIAL)], node_name=["KEN"])

    assert [pair["relationship"] for pair in built[0]["pairs"]] == [
        "responds_to",
        "asserted_by",
    ]


@pytest.mark.asyncio
async def test_a_node_scoped_search_drops_an_untagged_counterpart():
    """The unscoped rendering of the very same graph keeps both pairs."""
    scoped = await build_statements(_pair_graph(), [_hit(DENIAL)], node_name=["KEN"])
    unscoped = await build_statements(_pair_graph(), [_hit(DENIAL)])

    assert scoped[0]["pairs"] == []
    assert len(unscoped[0]["pairs"]) == 2


@pytest.mark.asyncio
async def test_pairs_carry_the_relationship_and_the_counterpart_id():
    built = await build_statements(_pair_graph(), [_hit(DENIAL)])

    assert built[0]["pairs"][0]["relationship"] == "responds_to"
    assert built[0]["pairs"][0]["node_id"] == "allegation-1"


@pytest.mark.asyncio
async def test_pair_lines_follow_a_fixed_relationship_order():
    """Adapter order is not context order: the same graph has to render the same way."""
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(SPEAKER), _node_row(COUNSEL), _node_row(ALLEGATION)],
        edges=[
            ("denial-1", "defendants-1", "asserted_by", {}),
            ("denial-1", "counsel-1", "attributed_to", {}),
            ("denial-1", "allegation-1", "responds_to", {}),
        ],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert [pair["relationship"] for pair in built[0]["pairs"]] == [
        "responds_to",
        "attributed_to",
        "asserted_by",
    ]


@pytest.mark.asyncio
async def test_pairs_of_one_relationship_are_ordered_by_the_counterpart_id():
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(SECOND_ALLEGATION), _node_row(ALLEGATION)],
        edges=[
            ("denial-1", "allegation-2", "responds_to", {}),
            ("denial-1", "allegation-1", "responds_to", {}),
        ],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert [pair["node_id"] for pair in built[0]["pairs"]] == ["allegation-1", "allegation-2"]


@pytest.mark.asyncio
async def test_two_counterparts_that_render_identically_both_appear():
    """A multi-count complaint repeats a proposition; two allegations are not one line."""
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(ALLEGATION), _node_row(SECOND_ALLEGATION)],
        edges=[
            ("denial-1", "allegation-1", "responds_to", {}),
            ("denial-1", "allegation-2", "responds_to", {}),
        ],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert [pair["node_id"] for pair in built[0]["pairs"]] == ["allegation-1", "allegation-2"]
    assert len({pair["text"] for pair in built[0]["pairs"]}) == 1


@pytest.mark.asyncio
async def test_the_same_counterpart_twice_is_rendered_once():
    """Identity is the pair, not the rendering: the notes differ, the counterpart does not."""
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(ALLEGATION)],
        edges=[
            ("denial-1", "allegation-1", "responds_to", dict(RESOLUTION)),
            ("denial-1", "allegation-1", "responds_to", {}),
        ],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert len(built[0]["pairs"]) == 1


@pytest.mark.asyncio
async def test_a_nameless_counterpart_falls_back_to_its_node_id():
    """The label's id fallback must not depend on the store having stored an ``id``.

    ``get_neighborhood`` returns a node as ``(id, properties)``, and the projection fills
    every whitelisted key -- so ``properties["id"]`` is ``None`` whenever the store kept the
    id out of the property bag. The node's own id is what the label falls back to.
    """
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(NAMELESS_ALLEGATION)],
        edges=[("denial-1", "allegation-9", "responds_to", {})],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    assert built[0]["pairs"][0]["text"] == "responds to: [allegation/positive] allegation-9"


@pytest.mark.asyncio
async def test_a_chunk_counterpart_renders_its_first_words_not_its_id():
    """A ``responds_to`` edge a resolver anchored on a paragraph points at a DocumentChunk:
    no name, but text. The pair line has to read like the passage; a bare UUID tells the
    model nothing."""
    chunk = {
        "id": "9522e29a-201d-5177-9a62-2d8a48f8e734",
        "text": "13. Denied. Defendants deny each and every allegation of paragraph 13.",
    }
    graph = _graph(
        nodes=[_node_row(DENIAL), _node_row(chunk)],
        edges=[("denial-1", chunk["id"], "responds_to", RESOLUTION)],
    )

    built = await build_statements(graph, [_hit(DENIAL)])

    (pair,) = built[0]["pairs"]
    assert pair["text"] == (
        "responds to: 13. Denied. Defendants deny each and every... "
        "[13, denied, defendants] (confidence 0.95, paragraph_locator)"
    )
    assert "9522e29a" not in pair["text"]
    assert pair["node_id"] == chunk["id"]


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


def test_merge_lets_both_lanes_contribute_statements_primary_first():
    """Every other channel merges through ``merge_ranked``; statements were taken wholesale.

    On the default concurrent session path that costs a follow-up turn its answer: the
    conversational rewrite is the lane that understands "and what did he say about it", and
    the statement it found was discarded whenever the raw lane found anything at all.
    """
    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "conversational-1"}]},
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
        statements_limit=5,
    )

    assert merged["statements"] == [{"id": "denial-1"}, {"id": "conversational-1"}]
    assert set(merged) == {"chunks", "chunk_summaries", "entities", "facts", "statements"}


def test_merged_statements_are_capped_and_hold_a_conversational_reserve():
    """Same budget arithmetic the chunk, entity and fact lanes get, on the statements limit."""
    primary = [{"id": f"raw{index}"} for index in range(5)]
    # "raw3" is found by both lanes so it ranks first; every "ctx" is conversational-only.
    secondary = [{"id": name} for name in ("ctx0", "ctx1", "raw3", "ctx2")]

    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": [], "statements": primary},
        {"chunks": [], "entities": [], "facts": [], "statements": secondary},
        chunks_limit=1,
        entities_limit=1,
        facts_limit=1,
        statements_limit=5,
    )

    # One reserved slot at limit=5, so the lowest-ranked raw statement yields to "ctx0".
    assert [statement["id"] for statement in merged["statements"]] == [
        "raw3",
        "raw0",
        "raw1",
        "raw2",
        "ctx0",
    ]


def test_a_statement_both_lanes_found_is_not_rendered_twice():
    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
        statements_limit=5,
    )

    assert merged["statements"] == [{"id": "denial-1"}]


def test_the_retriever_merges_statements_under_its_own_budget():
    """``statements_top_k`` is the lane's budget, so the merge has to be handed it too.

    Asserted as a pair: the same two lanes cap to one statement under a budget of one and
    keep both under a budget of two, which no un-threaded limit can produce.
    """
    lanes = (
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "conversational-1"}]},
    )

    assert HybridRetriever(statements_top_k=1).merge_retrieved_objects(*lanes)["statements"] == [
        {"id": "denial-1"}
    ]
    assert HybridRetriever(statements_top_k=2).merge_retrieved_objects(*lanes)["statements"] == [
        {"id": "denial-1"},
        {"id": "conversational-1"},
    ]


def test_merge_keeps_the_statements_of_whichever_lane_found_them():
    """The default ``SESSION_SEARCH_MODE=concurrent`` retrieves twice and merges once.

    ``statements`` is not a merged channel -- the lane sets the key only when it found
    something -- so taking it from the primary alone discards the conversational lane's
    statements whenever the raw query did not rank any.
    """
    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": []},
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert merged["statements"] == [{"id": "denial-1"}]


def test_merge_keeps_the_statements_when_the_other_lane_raised():
    """``session_aware_completion`` merges ``None`` for a lane that raised."""
    merged = merge_hybrid_results(
        None,
        {"chunks": [], "entities": [], "facts": [], "statements": [{"id": "denial-1"}]},
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert merged["statements"] == [{"id": "denial-1"}]


def test_merge_of_two_lanes_without_statements_grows_no_key():
    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": []},
        {"chunks": [], "entities": [], "facts": []},
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert set(merged) == {"chunks", "chunk_summaries", "entities", "facts"}


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
