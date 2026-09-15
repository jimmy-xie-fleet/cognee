"""Unit tests for the reference tracer's seed retrieval.

Everything here runs against a ``GraphView`` built from plain node/edge tuples and a
scripted vector adapter: no vector backend, no embeddings, no graph backend, no LLM, no
filesystem. The vector channel is driven by ``FakeVectorEngine`` (cosine **distances**,
``IndexSchema``-shaped payloads) and the lexical channel by the real BM25 retriever over
the view's own text, so the scoring these tests pin is the scoring production runs.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.graph.utils.reference_candidates import LabelRegistry
from cognee.tasks.graph.reference_retrieval import (
    ASSERTION_COLLECTION,
    BM25_WEIGHT,
    CHUNK_COLLECTION,
    DEFAULT_K_PER_QUERY,
    DOCUMENT_COLLECTIONS,
    DOCUMENT_K,
    SAME_DOCUMENT_PENALTY,
    SEED_LIMIT,
    SUMMARY_COLLECTION,
    LexicalIndex,
    search_candidates,
)
from cognee.tests.unit.tasks.graph._reference_fakes import (
    FakeVectorEngine,
    SearchOnlyVectorEngine,
    assertion_node,
    build_graph_view,
    chunk_node,
    document_node,
    nid,
    scored,
)

MODULE = "cognee.tasks.graph.reference_retrieval"

DOC_COMPLAINT = nid("doc-complaint")
DOC_ANSWER = nid("doc-answer")
COMPLAINT_NAME = "Verified_Complaint_Adams"
# Deliberately opaque: a document must be reachable by its content, never by its filename.
ANSWER_NAME = "SKM_C55826082316050"

C0 = nid("complaint-chunk-0")
C1 = nid("complaint-chunk-1")
A0 = nid("answer-chunk-0")

C0_TEXT = "The parties entered a zorvax lease in 1997 regarding 10 Main Street."
C1_TEXT = (
    "Clifton owns 10 Main Street. The aquamarine roof was replaced in 2019 "
    "by a contractor hired from the neighbouring county office."
)
A0_TEXT = "Clifton denies owning 10 Main Street. Norman Fester disputes the aquamarine roof."

A_ALLEGE = nid("assertion-allege")
A_ROOF = nid("assertion-roof")
A_LEASE = nid("assertion-lease")
A_DENY = nid("assertion-deny")
SUMMARY_C1 = nid("summary-complaint-chunk-1")
STALE = nid("assertion-deleted-but-still-indexed")

# A query whose tokens appear nowhere in the corpus, so the BM25 channel stays silent and
# a test can pin the vector channel on its own.
NO_LEXICAL_MATCH = "xylophone quasar"


async def _base_view():
    """Complaint (two chunks) + Answer (one chunk), four assertions across them."""
    nodes = [
        document_node(DOC_COMPLAINT, COMPLAINT_NAME),
        document_node(DOC_ANSWER, ANSWER_NAME),
        assertion_node(A_ALLEGE, "Clifton owns 10 Main Street", C1),
        assertion_node(A_ROOF, "The aquamarine roof was replaced in 2019", C1),
        assertion_node(A_LEASE, "The lease began in 1997", C0),
        assertion_node(
            A_DENY,
            "Clifton owns 10 Main Street",
            A0,
            statement_type="denial",
            polarity="negative",
        ),
    ]
    edges = []
    for node, edge in (
        chunk_node(C0, C0_TEXT, 0, DOC_COMPLAINT),
        chunk_node(C1, C1_TEXT, 1, DOC_COMPLAINT),
        chunk_node(A0, A0_TEXT, 0, DOC_ANSWER),
    ):
        nodes.append(node)
        edges.append(edge)
    return await build_graph_view(nodes, edges)


def _base_engine(**kwargs) -> FakeVectorEngine:
    """Every collection the legal ingest populates, scripted with cosine distances.

    ``STALE`` is the best hit of all and is not a node in the view: a deleted assertion
    whose index row survived. ``SUMMARY_C1`` stands for the summary of chunk ``C1``.
    ``C1`` carries a full ``IndexSchema`` payload; the assertions carry only
    ``source_chunk_id``, so their document fields have to come from the view.
    """
    return FakeVectorEngine(
        {
            ASSERTION_COLLECTION: [
                scored(STALE, 0.05, text="Deleted allegation", source_chunk_id=C1),
                scored(A_ALLEGE, 0.10, text="Clifton owns 10 Main Street", source_chunk_id=C1),
                scored(A_DENY, 0.30, text="Clifton owns 10 Main Street", source_chunk_id=A0),
                scored(A_LEASE, 0.40, text="The lease began in 1997", source_chunk_id=C0),
            ],
            CHUNK_COLLECTION: [
                scored(
                    C1,
                    0.25,
                    text=C1_TEXT,
                    document_id=DOC_COMPLAINT,
                    document_name=COMPLAINT_NAME,
                    chunk_index=1,
                ),
                scored(A0, 0.45, text=A0_TEXT),
            ],
            SUMMARY_COLLECTION: [
                scored(SUMMARY_C1, 0.20, text="Ownership and the roof.", source_chunk_id=C1),
            ],
            f"{DOCUMENT_COLLECTIONS[0]}": [scored(DOC_COMPLAINT, 0.32, text=COMPLAINT_NAME)],
        },
        **kwargs,
    )


async def _search(view, *, engine=None, lexical=None, registry=None, queries=None, **kwargs):
    return await search_candidates(
        queries=list(queries if queries is not None else [NO_LEXICAL_MATCH]),
        view=view,
        lexical=lexical if lexical is not None else LexicalIndex(view),
        registry=registry if registry is not None else LabelRegistry(),
        vector_engine=engine if engine is not None else FakeVectorEngine({}),
        **kwargs,
    )


def _by_id(candidates):
    return {candidate.node_id: candidate for candidate in candidates}


def _collections(calls):
    return [call[0] for call in calls]


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def test_constants_are_the_specified_values():
    assert DEFAULT_K_PER_QUERY == 8
    assert DOCUMENT_K == 3
    assert BM25_WEIGHT == 0.8
    assert SAME_DOCUMENT_PENALTY == 0.15
    assert SEED_LIMIT == 12
    assert ASSERTION_COLLECTION == "Assertion_name"
    assert CHUNK_COLLECTION == "DocumentChunk_text"
    assert SUMMARY_COLLECTION == "TextSummary_text"
    assert DOCUMENT_COLLECTIONS == (
        "TextDocument_name",
        "PdfDocument_name",
        "UnstructuredDocument_name",
        "AudioDocument_name",
        "ImageDocument_name",
    )


# --------------------------------------------------------------------------- #
# the vector channel
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_distances_become_similarities_and_rank_the_seed():
    view = await _base_view()

    candidates = await _search(view, engine=_base_engine())

    assert [(candidate.node_id, round(candidate.score, 4)) for candidate in candidates] == [
        (A_ALLEGE, 0.90),
        (C1, 0.80),
        (A_DENY, 0.70),
        (DOC_COMPLAINT, 0.68),
        (A_LEASE, 0.60),
        (A0, 0.55),
    ]
    assert [candidate.label for candidate in candidates] == ["A1", "P1", "A2", "D1", "A3", "P2"]
    assert candidates[0].sources == ("vector:0",)


@pytest.mark.asyncio
async def test_assertion_payload_is_enriched_from_the_view():
    view = await _base_view()

    candidates = _by_id(await _search(view, engine=_base_engine()))

    allege = candidates[A_ALLEGE]
    assert allege.node_type == "Assertion"
    assert allege.text == "Clifton owns 10 Main Street"
    assert allege.document_id == DOC_COMPLAINT
    assert allege.document_name == COMPLAINT_NAME
    assert allege.chunk_index == 1
    # chunk_index 0 is a real index, not a missing value.
    assert candidates[A_DENY].chunk_index == 0
    assert candidates[A_DENY].document_id == DOC_ANSWER
    assert candidates[A_DENY].document_name == ANSWER_NAME
    # The chunk hit whose payload carried no document fields gets them from the view too.
    assert candidates[A0].document_id == DOC_ANSWER
    assert candidates[A0].document_name == ANSWER_NAME
    assert candidates[A0].chunk_index == 0
    # A document candidate is its own document.
    assert candidates[DOC_COMPLAINT].node_type == "TextDocument"
    assert candidates[DOC_COMPLAINT].document_id == DOC_COMPLAINT
    assert candidates[DOC_COMPLAINT].document_name == COMPLAINT_NAME


@pytest.mark.asyncio
async def test_summary_hit_is_mapped_to_the_chunk_it_was_made_from():
    view = await _base_view()

    candidates = await _search(view, engine=_base_engine())

    ids = [candidate.node_id for candidate in candidates]
    assert SUMMARY_C1 not in ids
    assert ids.count(C1) == 1
    chunk = _by_id(candidates)[C1]
    assert chunk.node_type == "DocumentChunk"
    # The summary (0.80) beat the direct chunk hit (0.75); the union keeps the best.
    assert chunk.score == pytest.approx(0.80)
    assert chunk.text.startswith("Clifton owns 10 Main Street.")


@pytest.mark.asyncio
async def test_summary_hit_without_a_resolvable_chunk_is_dropped():
    view = await _base_view()
    engine = FakeVectorEngine(
        {
            SUMMARY_COLLECTION: [
                scored(nid("summary-orphan"), 0.1, text="Orphan", source_chunk_id=nid("gone")),
                scored(nid("summary-no-link"), 0.1, text="No link at all"),
            ]
        }
    )

    assert await _search(view, engine=engine) == []


@pytest.mark.asyncio
async def test_stale_index_rows_are_dropped():
    view = await _base_view()

    candidates = await _search(view, engine=_base_engine())

    assert STALE not in {candidate.node_id for candidate in candidates}
    # The best surviving hit leads, so the drop happened before ranking.
    assert candidates[0].node_id == A_ALLEGE


@pytest.mark.asyncio
async def test_missing_collection_is_an_empty_channel():
    view = await _base_view()
    engine = _base_engine(missing_collections=[ASSERTION_COLLECTION])

    candidates = await _search(view, engine=engine)

    assert {candidate.node_id for candidate in candidates} == {C1, DOC_COMPLAINT, A0}
    assert ASSERTION_COLLECTION not in _collections(engine.batch_search_calls)


@pytest.mark.asyncio
async def test_collection_not_found_error_is_an_empty_channel():
    view = await _base_view()
    engine = _base_engine(raising_collections=[ASSERTION_COLLECTION])

    candidates = await _search(view, engine=engine)

    assert {candidate.node_id for candidate in candidates} == {C1, DOC_COMPLAINT, A0}
    # The guard passed, so the query really was attempted before the error was swallowed.
    assert ASSERTION_COLLECTION in _collections(engine.batch_search_calls)


@pytest.mark.asyncio
async def test_each_query_keeps_its_own_source_tag():
    view = await _base_view()
    engine = _base_engine()

    candidates = _by_id(await _search(view, engine=engine, queries=[NO_LEXICAL_MATCH, "quasar 2"]))

    assert candidates[A_ALLEGE].sources == ("vector:0", "vector:1")
    assert (
        ASSERTION_COLLECTION,
        [NO_LEXICAL_MATCH, "quasar 2"],
        DEFAULT_K_PER_QUERY,
        True,
    ) in engine.batch_search_calls


@pytest.mark.asyncio
async def test_engine_without_batch_search_falls_back_to_one_search_per_query():
    view = await _base_view()
    batched = _base_engine()
    engine = SearchOnlyVectorEngine(dict(batched.results_by_collection))

    candidates = await _search(view, engine=engine, queries=[NO_LEXICAL_MATCH, "quasar 2"])

    assert [candidate.node_id for candidate in candidates] == [
        candidate.node_id for candidate in await _search(view, engine=batched)
    ]
    assert engine.search_calls.count((ASSERTION_COLLECTION, NO_LEXICAL_MATCH, 8, True)) == 1
    assert engine.search_calls.count((ASSERTION_COLLECTION, "quasar 2", 8, True)) == 1


@pytest.mark.asyncio
async def test_the_engine_defaults_to_the_module_seam():
    view = await _base_view()
    engine = _base_engine()

    with patch(f"{MODULE}.get_vector_engine_async", new=AsyncMock(return_value=engine)) as getter:
        candidates = await search_candidates(
            queries=[NO_LEXICAL_MATCH],
            view=view,
            lexical=LexicalIndex(view),
            registry=LabelRegistry(),
        )

    getter.assert_awaited_once()
    assert candidates[0].node_id == A_ALLEGE


# --------------------------------------------------------------------------- #
# the lexical channel
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_bm25_top_hit_scores_the_bm25_weight():
    view = await _base_view()

    candidates = await _search(view, queries=["zorvax"], kind="passages")

    assert len(candidates) == 1
    assert candidates[0].node_id == C0
    assert candidates[0].score == pytest.approx(BM25_WEIGHT)
    assert candidates[0].sources == ("bm25:0",)
    assert candidates[0].node_type == "DocumentChunk"
    assert candidates[0].document_id == DOC_COMPLAINT
    assert candidates[0].chunk_index == 0


@pytest.mark.asyncio
async def test_bm25_scores_are_normalised_by_their_own_top_score():
    view = await _base_view()
    lexical = LexicalIndex(view)

    raw = await lexical.search_chunks("aquamarine", DEFAULT_K_PER_QUERY)
    candidates = await _search(view, lexical=lexical, queries=["aquamarine"], kind="passages")

    assert [node_id for node_id, _ in raw] == [candidate.node_id for candidate in candidates]
    assert len(raw) == 2
    assert raw[0][1] > raw[1][1] > 0
    assert candidates[0].score == pytest.approx(BM25_WEIGHT)
    assert candidates[1].score == pytest.approx(BM25_WEIGHT * raw[1][1] / raw[0][1])
    assert candidates[1].score < candidates[0].score


@pytest.mark.asyncio
async def test_bm25_searches_assertion_names_too():
    view = await _base_view()

    candidates = await _search(view, queries=["aquamarine"], kind="assertions")

    assert [candidate.node_id for candidate in candidates] == [A_ROOF]
    assert candidates[0].score == pytest.approx(BM25_WEIGHT)
    assert candidates[0].sources == ("bm25:0",)
    assert candidates[0].text == "The aquamarine roof was replaced in 2019"


@pytest.mark.asyncio
async def test_lexical_corpora_are_built_once_per_pass():
    view = await _base_view()
    lexical = LexicalIndex(view)

    for _ in range(3):
        await _search(view, lexical=lexical, queries=["aquamarine"])
    await lexical.search_chunks("zorvax", 4)
    await lexical.search_assertions("lease", 4)

    assert lexical.builds == 2


@pytest.mark.asyncio
async def test_a_corpus_with_no_tokens_returns_nothing():
    empty = LexicalIndex(await build_graph_view([], []))

    assert await empty.search_chunks("aquamarine", 4) == []
    assert await empty.search_assertions("aquamarine", 4) == []
    assert empty.builds == 2


@pytest.mark.asyncio
async def test_lexical_search_honours_k():
    view = await _base_view()
    lexical = LexicalIndex(view)

    assert len(await lexical.search_chunks("aquamarine", 1)) == 1
    assert len(await lexical.search_chunks("aquamarine", 2)) == 2


# --------------------------------------------------------------------------- #
# kinds
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_kind_assertions_queries_only_the_assertion_channel():
    view = await _base_view()
    engine = _base_engine()

    candidates = await _search(view, engine=engine, kind="assertions")

    assert [candidate.node_id for candidate in candidates] == [A_ALLEGE, A_DENY, A_LEASE]
    assert engine.has_collection_calls == [ASSERTION_COLLECTION]


@pytest.mark.asyncio
async def test_kind_passages_covers_chunks_and_summaries():
    view = await _base_view()
    engine = _base_engine()

    candidates = await _search(view, engine=engine, kind="passages")

    assert [candidate.node_id for candidate in candidates] == [C1, A0]
    assert engine.has_collection_calls == [CHUNK_COLLECTION, SUMMARY_COLLECTION]


@pytest.mark.asyncio
async def test_kind_documents_rolls_chunk_hits_up_to_their_document():
    view = await _base_view()
    engine = _base_engine()

    candidates = await _search(view, engine=engine, kind="documents")

    # C1 (0.75) rolled up beats the complaint's own name hit (0.68); A0 (0.55) surfaces
    # the answer, whose name was never indexed by any query.
    assert [(candidate.node_id, round(candidate.score, 4)) for candidate in candidates] == [
        (DOC_COMPLAINT, 0.75),
        (DOC_ANSWER, 0.55),
    ]
    assert [candidate.label for candidate in candidates] == ["D1", "D2"]
    assert [candidate.text for candidate in candidates] == [COMPLAINT_NAME, ANSWER_NAME]
    assert engine.has_collection_calls == [*DOCUMENT_COLLECTIONS, CHUNK_COLLECTION]
    assert (DOCUMENT_COLLECTIONS[0], [NO_LEXICAL_MATCH], DOCUMENT_K, True) in (
        engine.batch_search_calls
    )


@pytest.mark.asyncio
async def test_a_document_is_found_by_its_content_not_its_filename():
    view = await _base_view()

    candidates = await _search(view, queries=["fester"], kind="documents")

    assert [(candidate.node_id, candidate.text) for candidate in candidates] == [
        (DOC_ANSWER, ANSWER_NAME)
    ]
    assert candidates[0].score == pytest.approx(BM25_WEIGHT)
    assert candidates[0].sources == ("bm25:0",)


@pytest.mark.asyncio
async def test_an_unknown_kind_is_rejected():
    view = await _base_view()

    with pytest.raises(ValueError, match="kind"):
        await _search(view, kind="everything")


# --------------------------------------------------------------------------- #
# scoping
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_excluded_ids_are_dropped():
    view = await _base_view()

    candidates = await _search(view, engine=_base_engine(), exclude_ids={A_DENY, A0})

    assert [candidate.node_id for candidate in candidates] == [
        A_ALLEGE,
        C1,
        DOC_COMPLAINT,
        A_LEASE,
    ]


@pytest.mark.asyncio
async def test_the_same_document_penalty_only_applies_when_it_is_asked_for():
    view = await _base_view()
    unpenalized = await _search(view, engine=_base_engine(), own_document_id=DOC_COMPLAINT)

    penalized = await _search(
        view,
        engine=_base_engine(),
        own_document_id=DOC_COMPLAINT,
        penalize_own_document=True,
    )

    assert [candidate.node_id for candidate in unpenalized] == [
        A_ALLEGE,
        C1,
        A_DENY,
        DOC_COMPLAINT,
        A_LEASE,
        A0,
    ]
    assert [(candidate.node_id, round(candidate.score, 4)) for candidate in penalized] == [
        (A_ALLEGE, 0.75),
        (A_DENY, 0.70),
        (C1, 0.65),
        (A0, 0.55),
        (DOC_COMPLAINT, 0.53),
        (A_LEASE, 0.45),
    ]


@pytest.mark.asyncio
async def test_the_same_document_penalty_is_never_a_filter():
    view = await _base_view()

    candidates = await _search(
        view,
        engine=_base_engine(),
        own_document_id=DOC_COMPLAINT,
        penalize_own_document=True,
    )

    ids = {candidate.node_id for candidate in candidates}
    assert {A_ALLEGE, C1, A_LEASE, DOC_COMPLAINT} <= ids
    # Labels belong to the registry, so the penalty reorders without relabelling.
    assert _by_id(candidates)[A_ALLEGE].label == "A1"
    assert _by_id(candidates)[C1].label == "P1"


@pytest.mark.asyncio
async def test_the_penalty_needs_a_document_to_penalize():
    view = await _base_view()

    with_flag = await _search(view, engine=_base_engine(), penalize_own_document=True)
    without = await _search(view, engine=_base_engine())

    assert [(candidate.node_id, candidate.score) for candidate in with_flag] == [
        (candidate.node_id, candidate.score) for candidate in without
    ]


@pytest.mark.asyncio
async def test_the_limit_truncates_the_merged_list():
    view = await _base_view()

    candidates = await _search(view, engine=_base_engine(), limit=3)

    assert [candidate.node_id for candidate in candidates] == [A_ALLEGE, C1, A_DENY]
    # A candidate dropped by the limit never consumed a label.
    assert [candidate.label for candidate in candidates] == ["A1", "P1", "A2"]


@pytest.mark.asyncio
async def test_a_blank_query_reaches_no_backend():
    view = await _base_view()
    engine = _base_engine()

    assert await _search(view, engine=engine, queries=["   ", ""]) == []
    assert engine.has_collection_calls == []
    assert engine.batch_search_calls == []


@pytest.mark.asyncio
async def test_registry_labels_are_shared_across_searches():
    view = await _base_view()
    registry = LabelRegistry()

    first = await _search(view, engine=_base_engine(), registry=registry, limit=1)
    second = await _search(view, engine=_base_engine(), registry=registry, limit=2)

    assert [candidate.label for candidate in first] == ["A1"]
    assert [candidate.label for candidate in second] == ["A1", "P1"]
    assert registry.resolve("A1") == A_ALLEGE
