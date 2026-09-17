"""The legal profile's per-chunk extraction hook: two passes merged, salience applied, low dropped.

No LLM anywhere: graphs are literals, and where the hook itself is exercised the LLM
call is patched at the module seam the hook imports it through.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.domains.legal import (
    SALIENCE_IMPORTANCE,
    LegalExtractionOptions,
    LegalKnowledgeGraph,
    LegalNode,
    Polarity,
    Salience,
    legal_chunk_graphs,
)
from cognee.domains.legal import extraction as extraction_module
from cognee.domains.legal.extraction import (
    LEGAL_ID_PREFIX,
    PLAIN_ID_PREFIX,
    apply_salience,
    drop_low_salience_assertions,
    extract_legal_chunk_graph,
    merge_plain_and_legal_graphs,
)
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.Assertion import Assertion, StatementType
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    construct_data_points_and_edges,
    is_assertion_node,
    prune_extracted_graph,
)
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node

PROPOSITION = "Meridian issued Okafor a written warning on March 3, 2025"


def _plain_graph() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(id="okafor", name="Amara Okafor", type="Person", description="Plaintiff."),
            Node(
                id="meridian",
                name="Meridian Logistics, Inc.",
                type="Organization",
                description="Defendant, a logistics company.",
            ),
            Node(
                id="warning",
                name="Written warning of March 3, 2025",
                type="Document",
                description="Warning issued to Okafor.",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="meridian",
                target_node_id="okafor",
                relationship_name="employs",
                description="Meridian Logistics, Inc. employs Amara Okafor.",
            ),
            KGEdge(
                source_node_id="meridian",
                target_node_id="warning",
                relationship_name="issued",
                description="Meridian Logistics, Inc. issued the written warning.",
            ),
        ],
    )


def _legal_graph() -> LegalKnowledgeGraph:
    return LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="meridian",
                name="Meridian Logistics, Inc.",
                type="Company",
                description="Defendant, the answering party.",
            ),
            LegalNode(
                id="allegation",
                name=PROPOSITION,
                type="Allegation",
                description="Okafor's allegation, as the Answer recites it.",
                statement_type=StatementType.ALLEGATION,
                polarity=Polarity.POSITIVE,
                asserted_by="okafor-legal",
                salience=Salience.HIGH,
            ),
            LegalNode(
                id="okafor-legal",
                name="Amara Okafor",
                type="Person",
                description="Plaintiff.",
            ),
            LegalNode(
                id="denial",
                name=PROPOSITION,
                type="Denial",
                description="Meridian's denial of the recited allegation.",
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
                responds_to="allegation",
                salience=Salience.LOW,  # marked low by mistake: a denial is never dropped
            ),
            LegalNode(
                id="boilerplate",
                name="Meridian repeats its prior responses to Paragraphs 1 through 16",
                type="Statement",
                description="Incorporation by reference.",
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.POSITIVE,
                asserted_by="meridian",
                responds_to="Complaint ¶¶1-16",  # locator text, names no node
                salience=Salience.LOW,
            ),
            LegalNode(
                id="unscored",
                name="The audit identified falsified entries",
                type="Statement",
                description="A statement without a salience mark.",
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.NEGATIVE,
                asserted_by="meridian",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="boilerplate",
                target_node_id="meridian",
                relationship_name="about",
                description="The incorporation is about Meridian Logistics, Inc.",
            ),
        ],
    )


def _chunk(text: str = "Some passage.") -> MagicMock:
    chunk = MagicMock()
    chunk.id = uuid4()
    chunk.text = text
    chunk.contains = None
    chunk.belongs_to_set = None
    chunk.importance_weight = 0.5
    chunk._produced_edge_identities = []
    chunk._provenance_edges = []
    return chunk


def _by_id(graph, node_id):
    return next(node for node in graph.nodes if node.id == node_id)


# --------------------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------------------


def test_merge_prefixes_ids_per_pass_and_rewrites_edge_endpoints():
    merged = merge_plain_and_legal_graphs(_plain_graph(), _legal_graph())

    ids = {node.id for node in merged.nodes}
    assert f"{LEGAL_ID_PREFIX}meridian" in ids
    assert f"{LEGAL_ID_PREFIX}denial" in ids
    assert f"{PLAIN_ID_PREFIX}warning" in ids
    assert len(ids) == len(merged.nodes), "every id unique after the merge"
    for edge in merged.edges:
        assert edge.source_node_id in ids and edge.target_node_id in ids


def test_merge_rewrites_assertion_references_to_the_prefixed_ids():
    merged = merge_plain_and_legal_graphs(_plain_graph(), _legal_graph())

    denial = _by_id(merged, f"{LEGAL_ID_PREFIX}denial")
    assert denial.asserted_by == f"{LEGAL_ID_PREFIX}meridian"
    assert denial.responds_to == f"{LEGAL_ID_PREFIX}allegation"
    # locator text never equals a node id and is left exactly as written
    assert _by_id(merged, f"{LEGAL_ID_PREFIX}boilerplate").responds_to == "Complaint ¶¶1-16"


def test_merge_folds_a_same_name_plain_entity_onto_the_legal_node():
    merged = merge_plain_and_legal_graphs(_plain_graph(), _legal_graph())

    names = [node.name for node in merged.nodes if not is_assertion_node(node)]
    assert names.count("Meridian Logistics, Inc.") == 1
    assert names.count("Amara Okafor") == 1
    survivor = _by_id(merged, f"{LEGAL_ID_PREFIX}meridian")
    assert survivor.type == "Company"  # the legal pass's vocabulary wins
    # the plain pass's edges now hang off the survivor
    employs = next(edge for edge in merged.edges if edge.relationship_name == "employs")
    assert employs.source_node_id == f"{LEGAL_ID_PREFIX}meridian"
    assert employs.target_node_id == f"{LEGAL_ID_PREFIX}okafor-legal"


def test_plain_nodes_become_entities_never_assertions():
    merged = merge_plain_and_legal_graphs(_plain_graph(), _legal_graph())

    warning = _by_id(merged, f"{PLAIN_ID_PREFIX}warning")
    assert isinstance(warning, LegalNode)
    assert warning.statement_type is None
    assert not is_assertion_node(warning)
    assert sum(1 for node in merged.nodes if is_assertion_node(node)) == 4


def test_merged_graph_constructs_one_entity_per_name_with_the_shared_identity():
    merged = merge_plain_and_legal_graphs(_plain_graph(), _legal_graph())
    chunk = _chunk()

    data_points_by_id, _edges = construct_data_points_and_edges([chunk], [merged])

    entities = [
        point
        for point in data_points_by_id.values()
        if isinstance(point, Entity) and not isinstance(point, Assertion)
    ]
    meridian_name = generate_node_name("Meridian Logistics, Inc.")
    meridians = [entity for entity in entities if entity.name == meridian_name]
    assert len(meridians) == 1
    # a single node per name gets the plain, cross-chunk identity -- not a chunk-scoped one
    assert meridians[0].id == Entity.id_for(meridian_name)
    okafors = [entity for entity in entities if entity.name == generate_node_name("Amara Okafor")]
    assert len(okafors) == 1


def test_merge_drops_plain_edges_that_collapse_into_self_loops():
    plain = KnowledgeGraph(
        nodes=[
            Node(id="a", name="Acme", type="Organization", description="."),
            Node(id="b", name="ACME", type="Organization", description="."),
        ],
        edges=[
            KGEdge(
                source_node_id="a", target_node_id="b", relationship_name="same_as", description="."
            )
        ],
    )
    legal = LegalKnowledgeGraph(
        nodes=[LegalNode(id="acme", name="Acme", type="Company", description=".")], edges=[]
    )

    merged = merge_plain_and_legal_graphs(plain, legal)

    assert [node.id for node in merged.nodes] == [f"{LEGAL_ID_PREFIX}acme"]
    assert merged.edges == []


# --------------------------------------------------------------------------------------
# Salience
# --------------------------------------------------------------------------------------


def test_apply_salience_maps_marks_to_importance_and_counts_the_unscored():
    graph = _legal_graph()

    unscored = apply_salience(graph)

    assert unscored == 1
    assert (
        _by_id(graph, "allegation").importance_weight == SALIENCE_IMPORTANCE[Salience.HIGH] == 0.9
    )
    assert (
        _by_id(graph, "boilerplate").importance_weight == SALIENCE_IMPORTANCE[Salience.LOW] == 0.2
    )
    assert _by_id(graph, "unscored").importance_weight is None
    assert _by_id(graph, "meridian").importance_weight is None  # entities are left alone


def test_salience_weight_reaches_the_constructed_assertion_while_the_chunk_keeps_its_own():
    graph = _legal_graph()
    apply_salience(graph)
    chunk = _chunk()

    data_points_by_id, _edges = construct_data_points_and_edges([chunk], [graph])

    assertions = {
        point.name: point for point in data_points_by_id.values() if isinstance(point, Assertion)
    }
    boilerplate = generate_node_name(
        "Meridian repeats its prior responses to Paragraphs 1 through 16"
    )
    assert assertions[boilerplate].importance_weight == 0.2
    unscored = generate_node_name("The audit identified falsified entries")
    assert assertions[unscored].importance_weight == 0.5  # the chunk's
    entity = next(
        point
        for point in data_points_by_id.values()
        if isinstance(point, Entity)
        and not isinstance(point, Assertion)
        and point.name == generate_node_name("Meridian Logistics, Inc.")
    )
    assert entity.importance_weight == 0.5


def test_drop_removes_low_statements_with_their_edges_and_keeps_denials():
    graph = _legal_graph()
    apply_salience(graph)

    counts = drop_low_salience_assertions(graph)

    ids = {node.id for node in graph.nodes}
    assert "boilerplate" not in ids
    assert "denial" in ids  # low-marked, but a denial is never dropped
    assert "allegation" in ids and "unscored" in ids
    assert not any(edge.source_node_id == "boilerplate" for edge in graph.edges)
    assert counts.total_assertions == 4
    assert counts.dropped_assertions == 1
    assert counts.dropped_edges == 1
    assert counts.unscored_assertions == 1


def test_a_low_target_of_a_retained_attributed_to_is_protected():
    """A kept opinion attributed to a low statement keeps its target: dropping it would
    leave the attribution pointing at nothing."""
    graph = LegalKnowledgeGraph(
        nodes=[
            LegalNode(id="speaker", name="City", type="City", description="."),
            LegalNode(
                id="low",
                name="The letter is submitted in the context of settlement discussions",
                type="Statement",
                description=".",
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.POSITIVE,
                asserted_by="speaker",
                salience=Salience.LOW,
            ),
            LegalNode(
                id="kept",
                name="The City offered $1,850,000 for the property",
                type="Proposal",
                description=".",
                statement_type=StatementType.PROPOSAL,
                polarity=Polarity.POSITIVE,
                asserted_by="speaker",
                attributed_to="low",
                salience=Salience.HIGH,
            ),
        ],
        edges=[],
    )

    counts = drop_low_salience_assertions(graph)

    assert {node.id for node in graph.nodes} == {"speaker", "low", "kept"}
    assert counts.dropped_assertions == 0


def test_prune_extracted_graph_nulls_references_to_dropped_nodes_and_prunes_edges():
    """The shared prune step (also strict mode's): a dropped target leaves no dangling
    reference and no edge, and a collapsed node's references follow the survivor."""
    graph = LegalKnowledgeGraph(
        nodes=[
            LegalNode(id="a", name="Acme", type="Company", description="."),
            LegalNode(id="a2", name="ACME", type="Company", description="."),
            LegalNode(id="gone", name="Nobody", type="Person", description="."),
            LegalNode(
                id="claim",
                name="Acme paid late",
                type="Statement",
                description=".",
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.POSITIVE,
                asserted_by="gone",
                attributed_to="a2",
            ),
        ],
        edges=[
            KGEdge(
                source_node_id="claim",
                target_node_id="gone",
                relationship_name="about",
                description=".",
            ),
            KGEdge(
                source_node_id="claim",
                target_node_id="a2",
                relationship_name="about",
                description=".",
            ),
        ],
    )

    dropped_edges = prune_extracted_graph(graph, {"gone"}, {"a2": "a"})

    assert dropped_edges == 1
    assert {node.id for node in graph.nodes} == {"a", "a2", "claim"}  # collapse does not remove
    claim = _by_id(graph, "claim")
    assert claim.asserted_by is None
    assert claim.attributed_to == "a"
    assert [(edge.source_node_id, edge.target_node_id) for edge in graph.edges] == [("claim", "a")]


def test_a_low_target_of_a_retained_responds_to_is_protected():
    graph = LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="low",
                name="The certification says the controversy is the subject of another action",
                type="Statement",
                description=".",
                statement_type=StatementType.STATEMENT,
                polarity=Polarity.NEGATIVE,
                salience=Salience.LOW,
            ),
            LegalNode(
                id="answer",
                name="The certification says the controversy is the subject of another action",
                type="Denial",
                description=".",
                statement_type=StatementType.DENIAL,
                polarity=Polarity.NEGATIVE,
                responds_to="low",
                salience=Salience.MEDIUM,
            ),
        ],
        edges=[],
    )

    counts = drop_low_salience_assertions(graph)

    assert {node.id for node in graph.nodes} == {"low", "answer"}
    assert counts.dropped_assertions == 0


def test_entities_are_never_dropped_by_salience():
    graph = LegalKnowledgeGraph(
        nodes=[
            LegalNode(
                id="court", name="Superior Court of New Jersey", type="Court", description="."
            ),
        ],
        edges=[],
    )
    # an entity carries no salience; even a stray mark on one must not drop it
    graph.nodes[0].salience = Salience.LOW

    counts = drop_low_salience_assertions(graph)

    assert [node.id for node in graph.nodes] == ["court"]
    assert counts.total_assertions == 0


# --------------------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------------------


def _fake_extraction(started: list, gate: asyncio.Event):
    async def fake(text, response_model, custom_prompt=None, **kwargs):
        started.append((response_model, custom_prompt))
        await gate.wait()
        if response_model is KnowledgeGraph:
            return _plain_graph()
        return _legal_graph()

    return fake


@pytest.mark.asyncio
async def test_two_pass_runs_both_extractions_concurrently():
    started: list = []
    gate = asyncio.Event()

    async def release_when_both_started():
        while len(started) < 2:
            await asyncio.sleep(0)
        gate.set()

    with patch.object(
        extraction_module, "extract_content_graph", side_effect=_fake_extraction(started, gate)
    ):
        merged, _ = await asyncio.gather(
            extract_legal_chunk_graph("passage", "LEGAL PROMPT", two_pass=True),
            release_when_both_started(),
        )

    assert started == [(KnowledgeGraph, None), (LegalKnowledgeGraph, "LEGAL PROMPT")]
    assert isinstance(merged, LegalKnowledgeGraph)
    assert f"{PLAIN_ID_PREFIX}warning" in {node.id for node in merged.nodes}


@pytest.mark.asyncio
async def test_single_pass_calls_the_llm_once_with_the_legal_prompt():
    fake = AsyncMock(return_value=_legal_graph())

    with patch.object(extraction_module, "extract_content_graph", fake):
        graph = await extract_legal_chunk_graph("passage", "LEGAL PROMPT", two_pass=False)

    assert fake.await_count == 1
    assert fake.await_args.args[1] is LegalKnowledgeGraph
    assert fake.await_args.kwargs == {"custom_prompt": "LEGAL PROMPT"}
    assert {node.id for node in graph.nodes} >= {"denial", "boilerplate"}  # no prefixing


@pytest.mark.asyncio
async def test_the_hook_extracts_every_chunk_applies_salience_and_drops_low():
    fake = AsyncMock(
        side_effect=lambda text, model, **kwargs: (
            _plain_graph() if model is KnowledgeGraph else _legal_graph()
        )
    )
    calculate = legal_chunk_graphs(two_pass=True, drop_low_salience=True)
    chunks = [_chunk("one"), _chunk("two")]

    with patch.object(extraction_module, "extract_content_graph", fake):
        graphs = await calculate(
            chunks, LegalKnowledgeGraph, "LEGAL PROMPT", calculate_chunk_graphs=calculate
        )

    assert calculate.options == LegalExtractionOptions(two_pass=True, drop_low_salience=True)
    assert fake.await_count == 4  # two passes x two chunks
    # cognify's kwargs (the hook itself among them) never reach the LLM
    assert all("calculate_chunk_graphs" not in call.kwargs for call in fake.await_args_list)
    assert len(graphs) == 2
    for graph in graphs:
        ids = {node.id for node in graph.nodes}
        assert f"{LEGAL_ID_PREFIX}boilerplate" not in ids
        assert f"{LEGAL_ID_PREFIX}denial" in ids
        assert _by_id(graph, f"{LEGAL_ID_PREFIX}allegation").importance_weight == 0.9


@pytest.mark.asyncio
async def test_the_hook_keeps_low_statements_down_weighted_when_the_drop_is_off():
    fake = AsyncMock(return_value=_legal_graph())
    calculate = legal_chunk_graphs(two_pass=False, drop_low_salience=False)

    with patch.object(extraction_module, "extract_content_graph", fake):
        (graph,) = await calculate([_chunk()], LegalKnowledgeGraph, "LEGAL PROMPT")

    assert _by_id(graph, "boilerplate").importance_weight == 0.2


@pytest.mark.asyncio
async def test_the_hook_refuses_a_foreign_graph_model():
    calculate = legal_chunk_graphs()

    with pytest.raises(ValueError, match="LegalKnowledgeGraph"):
        await calculate([_chunk()], KnowledgeGraph, None)
