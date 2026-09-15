"""Seed retrieval for assertion references.

One function, :func:`search_candidates`, turns one or two query texts into the labelled,
merged, penalised candidate list the reference tracer works from. It is the only retrieval
path in the resolver: the same call seeds every trace *and* backs the agent's ``search``
tool, so the seed and the agent see one ranking.

Candidates come from text the dataset already has indexed, never from a filename:

* the vector collections the ingest populates -- ``Assertion_name``,
  ``DocumentChunk_text``, ``TextSummary_text`` and one ``<DocumentType>_name`` collection
  per concrete document type -- each queried once per pass with
  :meth:`batch_search`, each behind a ``has_collection`` / ``CollectionNotFoundError``
  guard so an un-indexed collection is an empty channel rather than an error;
* two BM25 corpora over the graph view's own text (:class:`LexicalIndex`), built at most
  once per pass, so an exact rare-token match ("Fester") can win a ranking that semantic
  similarity alone would miss.

A document is therefore reachable through its content -- an opaque scan name like
``SKM_C55826082316050`` is never matched as a string.

Dataset scoping is implicit: inside a pipeline the vector engine already resolves to the
dataset's own databases, so this module never enters
``set_database_global_context_variables``. Nothing here calls an LLM.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.modules.graph.utils.reference_candidates import (
    Candidate,
    LabelRegistry,
    apply_penalty,
    distance_to_similarity,
    merge_candidates,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.reference_graph_view import (
    DOCUMENT_NODE_TYPES,
    GraphView,
    _chunk_index,
    _text_of,
)

logger = get_logger("reference_retrieval")

# How many hits to ask each collection for, per query.
DEFAULT_K_PER_QUERY = 8
# Document-name collections are tiny and their names are weak evidence, so they get less.
DOCUMENT_K = 3
# A BM25 channel's top hit is worth this much; the rest scale below it. Kept under 1.0 so
# a strong semantic hit can still outrank a merely-best lexical one.
BM25_WEIGHT = 0.8
# Subtracted from a candidate that lives in the referring statement's own document. A
# penalty, never a filter: "realleges the allegations of paragraphs 1-23" is a real
# reference back into the same pleading.
SAME_DOCUMENT_PENALTY = 0.15
SEED_LIMIT = 12

ASSERTION_COLLECTION = "Assertion_name"
CHUNK_COLLECTION = "DocumentChunk_text"
SUMMARY_COLLECTION = "TextSummary_text"
DOCUMENT_COLLECTIONS = tuple(f"{node_type}_name" for node_type in DOCUMENT_NODE_TYPES)

KINDS = ("any", "assertions", "passages", "documents")

# (node_id, node_type, similarity, source_tag, payload) -- what merge_candidates consumes.
_Item = Tuple[str, str, float, str, dict]


# --------------------------------------------------------------------------- #
# payload helpers
# --------------------------------------------------------------------------- #
def _first_text(*values: Any) -> Optional[str]:
    """The first non-blank string among ``values``, or None."""
    for value in values:
        text = _text_of(value)
        if text:
            return text
    return None


def _payload_of(result: Any) -> dict:
    """A result's payload as a plain dict (``include_payload=False`` yields None)."""
    payload = getattr(result, "payload", None)
    return dict(payload) if isinstance(payload, dict) else {}


def _assertion_item(
    node_id: str, similarity: float, source_tag: str, view: GraphView, payload: dict
) -> Optional[_Item]:
    """An ``Assertion`` hit, or None when the id is no longer a node in the view.

    An assertion's index row carries ``source_chunk_id`` but no document fields (the
    model has none), so the document it lives in is resolved through the view.
    """
    props = view.assertions.get(node_id)
    if props is None:
        return None

    chunk_id = _first_text(payload.get("source_chunk_id"), props.get("source_chunk_id")) or ""
    document_id = _first_text(payload.get("document_id"), view.document_by_chunk.get(chunk_id))
    document_name = _first_text(
        payload.get("document_name"), view.documents.get(document_id or "", {}).get("name")
    )
    chunk_props = view.chunks.get(chunk_id)
    chunk_index = payload.get("chunk_index")
    if chunk_index is None and chunk_props is not None:
        chunk_index = _chunk_index(chunk_props)

    return (
        node_id,
        "Assertion",
        similarity,
        source_tag,
        {
            "text": _first_text(payload.get("text"), props.get("name")) or "",
            "document_id": document_id,
            "document_name": document_name,
            "chunk_index": chunk_index,
        },
    )


def _chunk_item(
    node_id: str, similarity: float, source_tag: str, view: GraphView, payload: dict
) -> Optional[_Item]:
    """A ``DocumentChunk`` hit, or None when the id is no longer a node in the view."""
    props = view.chunks.get(node_id)
    if props is None:
        return None

    document_id = _first_text(
        payload.get("document_id"),
        props.get("document_id"),
        view.document_by_chunk.get(node_id),
    )
    document_name = _first_text(
        payload.get("document_name"),
        props.get("document_name"),
        view.documents.get(document_id or "", {}).get("name"),
    )
    chunk_index = payload.get("chunk_index")
    if chunk_index is None:
        chunk_index = _chunk_index(props)

    return (
        node_id,
        "DocumentChunk",
        similarity,
        source_tag,
        {
            "text": _first_text(payload.get("text"), props.get("text")) or "",
            "document_id": document_id,
            "document_name": document_name,
            "chunk_index": chunk_index,
        },
    )


def _summary_item(
    node_id: str, similarity: float, source_tag: str, view: GraphView, payload: dict
) -> Optional[_Item]:
    """A ``TextSummary`` hit, mapped to the chunk it was made from.

    A summary is not a node the tracer can read or link to, and ``TextSummary`` is not one
    of the view's node types, so a summary hit is only useful as evidence *about its
    chunk*: it becomes a passage candidate for the chunk ``TextSummary.source_chunk_id``
    records (the flat form of its ``made_from`` edge), and merges with a direct hit on
    that same chunk. A summary whose chunk the view cannot resolve is dropped.
    """
    chunk_id = _text_of(payload.get("source_chunk_id"))
    if not chunk_id or chunk_id not in view.chunks:
        logger.debug("Dropping summary hit %s: its source chunk is not in the view.", node_id)
        return None
    return _chunk_item(chunk_id, similarity, source_tag, view, {})


def _document_item(
    node_id: str, similarity: float, source_tag: str, view: GraphView, payload: dict
) -> Optional[_Item]:
    """A document hit, or None when the id is no longer a document in the view."""
    props = view.documents.get(node_id)
    if props is None:
        return None

    name = _first_text(props.get("name"), payload.get("document_name"), payload.get("text"))
    return (
        node_id,
        _first_text(props.get("type")) or "Document",
        similarity,
        source_tag,
        {
            "text": name or node_id,
            "document_id": node_id,
            "document_name": name,
            "chunk_index": None,
        },
    )


def _document_of_chunk_item(
    chunk_id: str, similarity: float, source_tag: str, view: GraphView
) -> Optional[_Item]:
    """The document a chunk hit belongs to, so a document is findable by its content."""
    document_id = view.document_by_chunk.get(chunk_id)
    if not document_id:
        return None
    return _document_item(document_id, similarity, source_tag, view, {})


# --------------------------------------------------------------------------- #
# the lexical channel
# --------------------------------------------------------------------------- #
class LexicalIndex:
    """Two BM25 corpora over a :class:`GraphView`: its chunk texts and its assertion names.

    Both are seeded straight from the view rather than through
    ``BM25ChunksRetriever.initialize()``, which would run its own graph-wide query and
    could not index assertion names at all. Each corpus is built at most once per pass
    (``builds`` counts how many were built, so a caller or a test can see the reuse); a
    corpus with no tokens is remembered as empty and never rebuilt.
    """

    CHUNKS = "chunks"
    ASSERTIONS = "assertions"

    def __init__(self, view: GraphView) -> None:
        self._view = view
        # corpus name -> retriever, or None when the corpus has no tokens at all.
        self._corpora: Dict[str, Any] = {}
        self.builds = 0

    def _texts(self, corpus: str) -> Dict[str, str]:
        source = self._view.chunks if corpus == self.CHUNKS else self._view.assertions
        field = "text" if corpus == self.CHUNKS else "name"
        texts: Dict[str, str] = {}
        for node_id, props in source.items():
            text = _text_of(props.get(field))
            if text:
                texts[node_id] = text
        return texts

    def _corpus(self, corpus: str):
        if corpus in self._corpora:
            return self._corpora[corpus]

        from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever

        self.builds += 1
        texts = self._texts(corpus)
        retriever = None
        if texts:
            # top_k covers the whole corpus; this class does its own ordering and cut so
            # ties break deterministically on node id.
            retriever = BM25ChunksRetriever(top_k=len(texts), with_scores=True)
            for node_id, text in texts.items():
                tokens = retriever.tokenizer(text)
                if tokens:
                    retriever.chunks[node_id] = tokens
                    retriever.payloads[node_id] = {"id": node_id}
            if retriever.chunks:
                retriever._initialized = True
                retriever._build_corpus_stats()
                retriever._stats_built = True
            else:
                retriever = None

        if retriever is None:
            logger.debug("Lexical corpus %s has no tokens; it will always return nothing.", corpus)
        self._corpora[corpus] = retriever
        return retriever

    async def _search(self, corpus: str, query: str, k: int) -> List[Tuple[str, float]]:
        retriever = self._corpus(corpus)
        if retriever is None or not _text_of(query) or k <= 0:
            return []

        scored = await retriever.get_retrieved_objects(query)
        # A zero BM25 score means no query term occurs at all -- not a weak match but no
        # match, so it is dropped rather than normalised to zero.
        results = [
            (str(payload["id"]), float(score)) for payload, score in scored or [] if score > 0
        ]
        results.sort(key=lambda result: (-result[1], result[0]))
        return results[:k]

    async def search_chunks(
        self, query: str, k: int = DEFAULT_K_PER_QUERY
    ) -> List[Tuple[str, float]]:
        """``(chunk id, raw BM25 score)`` for the top ``k`` chunks, best first."""
        return await self._search(self.CHUNKS, query, k)

    async def search_assertions(
        self, query: str, k: int = DEFAULT_K_PER_QUERY
    ) -> List[Tuple[str, float]]:
        """``(assertion id, raw BM25 score)`` for the top ``k`` names, best first."""
        return await self._search(self.ASSERTIONS, query, k)


def _weighted(results: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Normalise raw BM25 scores by their own top score, scaled by ``BM25_WEIGHT``.

    BM25 scores are unbounded and corpus-dependent, so only their ranking transfers: the
    top hit becomes ``BM25_WEIGHT`` and the rest keep their ratio to it.
    """
    if not results:
        return []
    top = max(score for _, score in results)
    if top <= 0:
        return []
    return [(node_id, BM25_WEIGHT * score / top) for node_id, score in results]


# --------------------------------------------------------------------------- #
# the vector channel
# --------------------------------------------------------------------------- #
async def _collection_hits(
    engine: Any, collection: str, queries: Sequence[Tuple[int, str]], k: int
) -> List[Tuple[int, Any]]:
    """``(query index, ScoredResult)`` for one collection, or [] when it is not indexed.

    A collection the ingest never created is an empty channel, not an error: the
    ``has_collection`` guard skips it, and a collection that disappears between the guard
    and the query raises ``CollectionNotFoundError``, which is swallowed the same way.
    """
    query_texts = [text for _, text in queries]
    try:
        has_collection = getattr(engine, "has_collection", None)
        if has_collection is not None and not await has_collection(collection):
            logger.debug("Collection %s does not exist; skipping that channel.", collection)
            return []

        batch_search = getattr(engine, "batch_search", None)
        if callable(batch_search):
            batched = await batch_search(collection, query_texts, limit=k, include_payload=True)
        else:
            batched = [
                await engine.search(
                    collection_name=collection,
                    query_text=text,
                    limit=k,
                    include_payload=True,
                )
                for text in query_texts
            ]
    except CollectionNotFoundError:
        logger.debug("Collection %s not found; skipping that channel.", collection)
        return []

    hits: List[Tuple[int, Any]] = []
    for (index, _), results in zip(queries, batched or []):
        for result in results or []:
            hits.append((index, result))
    return hits


# --------------------------------------------------------------------------- #
# search_candidates
# --------------------------------------------------------------------------- #
async def search_candidates(
    *,
    queries: Sequence[str],
    kind: str = "any",
    view: GraphView,
    lexical: LexicalIndex,
    registry: LabelRegistry,
    exclude_ids: Set[str] = frozenset(),
    own_document_id: Optional[str] = None,
    penalize_own_document: bool = False,
    limit: int = SEED_LIMIT,
    vector_engine=None,
) -> List[Candidate]:
    """The labelled candidates one or two query texts find in the dataset's own text.

    Args:
        queries: Up to a handful of query texts -- the seed passes the reference's display
            text first and the referring proposition second. Each query's index becomes
            part of the ``source`` tag on the candidates it found (``vector:0``,
            ``bm25:1``), so a reader can see which query surfaced what. Blank queries are
            skipped; when nothing is left, no backend is touched.
        kind: Which channels to query. ``any`` (all of them), ``assertions``, ``passages``
            (chunks and summaries) or ``documents`` (the document-name collections plus
            chunk hits rolled up to the document that owns them, which is how a document
            with an opaque filename is found at all).
        view: The graph view this pass reads. It bounds the result: an id the view does
            not hold is a stale index row and is dropped, and every candidate's document
            fields are completed from it.
        lexical: The pass's :class:`LexicalIndex`, shared so its corpora are built once.
        registry: The trace's label registry. Labels are assigned to the merged, truncated
            list, and stay stable across calls that share a registry.
        exclude_ids: Node ids that can never be candidates -- the referring assertion
            itself and the chunk it was extracted from.
        own_document_id: The document the referring assertion lives in.
        penalize_own_document: Subtract ``SAME_DOCUMENT_PENALTY`` from candidates in
            ``own_document_id``. A penalty, not a filter.
        limit: How many candidates to return.
        vector_engine: An already-resolved vector adapter; defaults to
            ``await get_vector_engine_async()``.

    Returns:
        Up to ``limit`` candidates, best score first, ties broken on node id.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown search kind {kind!r}; expected one of {', '.join(KINDS)}.")

    indexed_queries = [
        (index, text)
        for index, text in enumerate(_text_of(query) or "" for query in queries)
        if text
    ]
    if not indexed_queries:
        return []

    want_assertions = kind in ("any", "assertions")
    want_passages = kind in ("any", "passages")
    want_documents = kind in ("any", "documents")
    # Only the document kind rolls chunk hits up; for "any" the chunks are candidates in
    # their own right and a rolled-up document would just compete with them.
    roll_chunks_up = kind == "documents"

    engine = vector_engine if vector_engine is not None else await get_vector_engine_async()
    items: List[Optional[_Item]] = []

    if want_assertions:
        for index, result in await _collection_hits(
            engine, ASSERTION_COLLECTION, indexed_queries, DEFAULT_K_PER_QUERY
        ):
            items.append(
                _assertion_item(
                    str(result.id),
                    distance_to_similarity(result.score),
                    f"vector:{index}",
                    view,
                    _payload_of(result),
                )
            )

    if want_passages:
        for index, result in await _collection_hits(
            engine, CHUNK_COLLECTION, indexed_queries, DEFAULT_K_PER_QUERY
        ):
            items.append(
                _chunk_item(
                    str(result.id),
                    distance_to_similarity(result.score),
                    f"vector:{index}",
                    view,
                    _payload_of(result),
                )
            )
        for index, result in await _collection_hits(
            engine, SUMMARY_COLLECTION, indexed_queries, DEFAULT_K_PER_QUERY
        ):
            items.append(
                _summary_item(
                    str(result.id),
                    distance_to_similarity(result.score),
                    f"vector:{index}",
                    view,
                    _payload_of(result),
                )
            )

    if want_documents:
        for collection in DOCUMENT_COLLECTIONS:
            for index, result in await _collection_hits(
                engine, collection, indexed_queries, DOCUMENT_K
            ):
                items.append(
                    _document_item(
                        str(result.id),
                        distance_to_similarity(result.score),
                        f"vector:{index}",
                        view,
                        _payload_of(result),
                    )
                )

    if roll_chunks_up:
        for index, result in await _collection_hits(
            engine, CHUNK_COLLECTION, indexed_queries, DEFAULT_K_PER_QUERY
        ):
            items.append(
                _document_of_chunk_item(
                    str(result.id),
                    distance_to_similarity(result.score),
                    f"vector:{index}",
                    view,
                )
            )

    for index, text in indexed_queries:
        source_tag = f"bm25:{index}"
        if want_assertions:
            for node_id, similarity in _weighted(
                await lexical.search_assertions(text, DEFAULT_K_PER_QUERY)
            ):
                items.append(_assertion_item(node_id, similarity, source_tag, view, {}))
        if want_passages or roll_chunks_up:
            for node_id, similarity in _weighted(
                await lexical.search_chunks(text, DEFAULT_K_PER_QUERY)
            ):
                if roll_chunks_up:
                    items.append(_document_of_chunk_item(node_id, similarity, source_tag, view))
                else:
                    items.append(_chunk_item(node_id, similarity, source_tag, view, {}))

    scored_items = [
        item
        for item in items
        if item is not None and item[0] not in exclude_ids and item[0] in view.node_ids
    ]

    candidates = merge_candidates(scored_items, limit=limit, registry=registry)

    if penalize_own_document and own_document_id:
        own_document_id = str(own_document_id)
        # The own-document node itself is penalised too: a reference whose referent is the
        # very document the statement was written in is the same "pointing back at myself"
        # case as a passage inside it.
        own = {
            candidate.node_id
            for candidate in candidates
            if candidate.document_id == own_document_id or candidate.node_id == own_document_id
        }
        if own:
            candidates = apply_penalty(candidates, own, SAME_DOCUMENT_PENALTY)

    return candidates
