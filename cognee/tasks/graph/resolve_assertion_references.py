"""Turn the free-text references an ``Assertion`` carries into graph edges.

Extraction stores a reference the way the document wrote it --
``responds_to="Complaint ¶5"``, ``attributed_to="Whitfield rebuttal appraisal"`` -- and no
graph edge can follow a string. This task reads the graph, reads the referenced documents,
runs the deterministic cascade in
:mod:`cognee.modules.graph.utils.reference_resolution` over every dangling reference, and
writes the answer as edges (plus a node patch recording how it was reached).

Three entry points over one pass:

* :func:`resolve_assertion_references` -- the cognify tail. Appended to a pipeline it
  resolves the references the ingestion touched, returns its input unchanged, and swallows
  its own errors, so reference resolution can never break ingestion.
* :func:`detect_dangling_references` / :func:`apply_reference_resolutions` -- the two-phase
  memify pair behind the ``resolve_references`` pipeline. The apply phase deliberately does
  **not** swallow write failures: a memify run that could not write is a visible error.

Matching is deterministic: no LLM, no vector search. The only reads are one filtered
graph query per invocation and the stored text of the documents actually referenced,
cached for the length of the pass. Writing is a normal edge write, so it embeds the new
edge texts through ``index_graph_edges`` like every other edge cognee stores.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.write_context import graph_provenance_write_kwargs
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.chunking.incremental_chunking import IncrementalPlanError, chunk_offsets
from cognee.modules.engine.utils.generate_node_name import generate_node_name
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.modules.graph.utils.reference_resolution import (
    LOCATOR_PATTERNS,
    STRATEGY_DOCUMENT_LOCATOR,
    STRATEGY_DOCUMENT_ONLY,
    STRATEGY_ENTITY_NAME,
    STRATEGY_EXISTING_ID,
    STRATEGY_PROSE_LOOKUP,
    ParsedReference,
    Resolution,
    anchor_chunk_index,
    build_node_patch,
    build_reference_edge,
    chunks_overlapping,
    document_profile,
    find_locator_span,
    lexical_tiebreak,
    match_document,
    parse_reference,
    scan_chunks_for_marker,
    select_anchored_assertions,
)
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.index_graph_edges import index_graph_edges

logger = get_logger("resolve_assertion_references")

# The reference fields an assertion carries. ``asserted_by`` is deliberately absent: it is
# an identity field, and rewriting it would give the assertion a new node id.
REFERENCE_FIELDS = ("responds_to", "attributed_to")

# Owner of record for edges written outside an ingestion, where no ``Data`` row is in
# context. At ingest the pipeline's own data item wins, so the edges vanish with the
# document's forget(); in memify this sentinel keeps the write attributable.
REFERENCE_RESOLUTION_DATA_ID = uuid5(NAMESPACE_URL, "cognee:reference-resolution")

DOCUMENT_NODE_TYPES = (
    "TextDocument",
    "PdfDocument",
    "UnstructuredDocument",
    "AudioDocument",
    "ImageDocument",
)

# Everything the cascade needs in one filtered read: the assertions and their chunks, the
# documents a reference may name, and the entities an ``attributed_to`` may name.
VIEW_NODE_TYPES = ("Assertion", "DocumentChunk", "Entity", *DOCUMENT_NODE_TYPES)

DEFAULT_CONFIDENCE_FLOOR = 0.6

# Notes a resolution carries out of the cascade, into ``<field>_resolution`` and the
# write summary.
# The field held an id no longer in the graph -- a forgotten or re-chunked target -- so
# the reference was re-resolved from the ``<field>_text`` the resolver preserved.
NOTE_STALE_ID = "stale_id"
# Every edge this resolution would write is already in the graph; only the node patch is
# still outstanding, so the write phase patches and skips the edge upsert.
NOTE_EDGES_EXIST = "edges_exist"
# ``add_edges`` succeeded but ``index_graph_edges`` did not.
NOTE_EDGE_INDEX_FAILED = "edge_index_failed"

# A located paragraph whose quoted assertions were found is the strongest answer short of
# an id; each fallback the span had to fall back on costs a tenth.
_LOCATOR_CONFIDENCE = 0.90
_LOCATOR_NOTE_PENALTY = 0.10
_PENALIZED_NOTES = frozenset({"ambiguous_marker", "chunk_scan_fallback"})
# No quote landed inside the span: the passage is right, the statement is a guess, so the
# edge points at the chunk and says so. Flat, not penalized further -- below the floor it
# would leave a matched paragraph unrecorded.
_CHUNK_ONLY_CONFIDENCE = 0.65
_PROSE_CONFIDENCE = 0.60
_RESOLVED_ID_CONFIDENCE = 1.0

# A lexical tiebreak decided between documents on their text, which is weaker evidence
# than the name itself; it lifts the score without ever reaching certainty.
_TIEBREAK_LEXICAL_WEIGHT = 0.2
_TIEBREAK_MAX_CONFIDENCE = 0.95

# Prose lookup only runs on a reference that reads like a description rather than a name,
# and only accepts a chunk that is clearly ahead of the runner up.
_PROSE_MINIMUM_TOKENS = 3
_PROSE_TOP_K = 2
_PROSE_MINIMUM_SCORE = 1.0
_PROSE_MARGIN_RATIO = 1.5

# The strategies that move an id into the field. ``existing_id`` already has one there and
# ``entity_name`` must keep the name (the ingest contract reads it back).
_PATCHED_STRATEGIES = frozenset(
    {STRATEGY_DOCUMENT_LOCATOR, STRATEGY_DOCUMENT_ONLY, STRATEGY_PROSE_LOOKUP}
)

# A "Resolution No. 2026-118" locator names a whole document rather than a place inside
# one, so it marks nothing in any text and resolves document-wide.
_DOCUMENT_LEVEL_LOCATOR_KINDS = frozenset(
    pattern.kind for pattern in LOCATOR_PATTERNS if pattern.document_level
)

# Floors assembled by float addition are not always the literal they are compared against.
_TOLERANCE = 1e-9

_LABEL_PREVIEW_LENGTH = 80


@dataclass
class GraphView:
    """Everything one pass reads out of the graph, indexed the ways it is queried."""

    assertions: Dict[str, dict] = field(default_factory=dict)
    chunks: Dict[str, dict] = field(default_factory=dict)
    documents: Dict[str, dict] = field(default_factory=dict)
    entities: Dict[str, dict] = field(default_factory=dict)
    # generate_node_name(name) -> entity ids. More than one id is an ambiguous name.
    entity_ids_by_name: Dict[str, List[str]] = field(default_factory=dict)
    node_ids: Set[str] = field(default_factory=set)
    edge_keys: Set[Tuple[str, str, str]] = field(default_factory=set)
    # document id -> its chunk property dicts, sorted by chunk_index.
    chunks_by_document: Dict[str, List[dict]] = field(default_factory=dict)
    # chunk id -> the document it is part of, for the "reference back at myself" penalty.
    document_by_chunk: Dict[str, str] = field(default_factory=dict)

    def node_props(self, node_id: Optional[str]) -> dict:
        """Properties of any node in the view, or an empty dict for an unknown id."""
        if node_id is None:
            return {}
        for index in (self.assertions, self.chunks, self.documents, self.entities):
            props = index.get(node_id)
            if props is not None:
                return props
        return {}


@dataclass(frozen=True)
class _Outcome:
    """What one ``(assertion, field)`` step concluded."""

    kind: str  # resolved | already_resolved | unresolved | ambiguous
    resolution: Optional[Resolution] = None
    # The field held an id that is no longer a node, whatever the cascade made of it.
    stale: bool = False


def _text_of(value: Any) -> Optional[str]:
    """A non-blank stripped string, or None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_uuid(value: Any) -> Optional[UUID]:
    try:
        return UUID(str(value))
    except (AttributeError, TypeError, ValueError):
        return None


def _node_label(props: Any) -> Optional[str]:
    """The label an edge's text calls this node by.

    Named nodes use their name; a ``DocumentChunk`` has none, so it is called by a preview
    of its text -- the same convention ``ensure_default_edge_properties`` uses.
    """
    if not isinstance(props, dict):
        return None

    name = _text_of(props.get("name"))
    if name:
        return name

    text = props.get("text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split())[:_LABEL_PREVIEW_LENGTH]
    return None


def _chunk_index(props: dict) -> int:
    try:
        return int(props.get("chunk_index"))
    except (TypeError, ValueError):
        return 0


async def _load_graph_view(graph_engine) -> GraphView:
    """Read the assertions, chunks, documents and entities in one filtered query.

    Edges are kept only when both endpoints pass the filter, which is exactly the set the
    cascade asks about: ``is_part_of`` for chunk ownership and the already-written
    ``responds_to`` / ``attributed_to`` that make a second pass a no-op. Adapters without
    attribute filtering fall back to the full graph.
    """
    try:
        nodes, edges = await graph_engine.get_filtered_graph_data([{"type": list(VIEW_NODE_TYPES)}])
    except NotImplementedError:
        logger.debug("Adapter cannot filter graph data; reading the whole graph instead.")
        nodes, edges = await graph_engine.get_graph_data()

    view = GraphView()
    for node_id, raw_props in nodes or []:
        node_id = str(node_id)
        props = dict(raw_props or {})
        props["id"] = node_id
        node_type = props.get("type")
        if node_type == "Assertion":
            view.assertions[node_id] = props
        elif node_type == "DocumentChunk":
            view.chunks[node_id] = props
        elif node_type in DOCUMENT_NODE_TYPES:
            view.documents[node_id] = props
        elif node_type == "Entity":
            # Assertion subclasses Entity in the model but is stored under its own type,
            # so only true entities can ever be an entity_name target.
            view.entities[node_id] = props
            name = _text_of(props.get("name"))
            if name:
                view.entity_ids_by_name.setdefault(generate_node_name(name), []).append(node_id)
        else:
            continue
        view.node_ids.add(node_id)

    for edge in edges or []:
        if not edge or len(edge) < 3:
            continue
        source_id, target_id, relationship = str(edge[0]), str(edge[1]), str(edge[2])
        view.edge_keys.add((source_id, target_id, relationship))
        if relationship == "is_part_of" and source_id in view.chunks:
            view.document_by_chunk[source_id] = target_id
            view.chunks_by_document.setdefault(target_id, []).append(view.chunks[source_id])

    for chunks in view.chunks_by_document.values():
        chunks.sort(key=_chunk_index)

    return view


async def _read_processed_text(raw_data_location: str) -> str:
    """Read a document's stored processed text (pattern from ``TextDocument.read``)."""
    async with open_data_file(raw_data_location, mode="r", encoding="utf-8") as file:
        return file.read()


async def _raw_locations(dataset_id) -> Dict[str, str]:
    """``{data id: raw_data_location}`` for a dataset, for documents whose node lacks one."""
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data

    data_id = _as_uuid(dataset_id)
    if data_id is None:
        return {}

    return {
        str(data.id): data.raw_data_location
        for data in await get_dataset_data(data_id)
        if getattr(data, "raw_data_location", None)
    }


class DocumentTextCache:
    """The stored text (and chunk offsets) of the documents one pass actually reads.

    A document is opened at most once per pass, successfully or not. A document whose
    text cannot be read -- a PDF or image opened as UTF-8, a file that moved -- is still
    a matched document, so the failure is cached as "no text" and the caller degrades to
    the stored chunks. Only an error the reader was not expected to raise propagates, so
    the ``failed`` counter keeps meaning "something is wrong here".
    """

    def __init__(self, view: GraphView, *, dataset_id=None):
        self._view = view
        self._dataset_id = dataset_id
        self._texts: Dict[str, Optional[str]] = {}
        self._offsets: Dict[str, Optional[List[Tuple[int, int]]]] = {}
        self._locations: Optional[Dict[str, str]] = None

    async def _location(self, document_id: str) -> Optional[str]:
        location = _text_of(self._view.documents.get(document_id, {}).get("raw_data_location"))
        if location:
            return location

        if self._locations is None:
            self._locations = await _raw_locations(self._dataset_id) if self._dataset_id else {}
        return _text_of(self._locations.get(document_id))

    async def text(self, document_id) -> Optional[str]:
        """The document's stored text, or None when no location points at one."""
        document_id = str(document_id)
        if document_id in self._texts:
            return self._texts[document_id]

        location = await self._location(document_id)
        if not location:
            logger.warning(
                "No stored text location for document %s; falling back to its chunks.",
                document_id,
            )
            self._texts[document_id] = None
            return None

        try:
            text = await _read_processed_text(location)
        except (UnicodeDecodeError, OSError) as error:
            logger.warning(
                "Could not read stored text for document %s at %s (%s); "
                "falling back to its chunks.",
                document_id,
                location,
                error,
            )
            self._texts[document_id] = None
            return None

        self._texts[document_id] = text
        return text

    async def offsets(self, document_id, chunks: Sequence[dict]) -> Optional[List[Tuple[int, int]]]:
        """Where each chunk sits in the document text, or None when they do not tile it."""
        document_id = str(document_id)
        if document_id in self._offsets:
            return self._offsets[document_id]

        text = await self.text(document_id)
        if text is None or not chunks:
            self._offsets[document_id] = None
            return None

        try:
            offsets = chunk_offsets(text, [chunk.get("text") or "" for chunk in chunks])
        except IncrementalPlanError as error:
            logger.warning(
                "Stored chunks do not tile document %s (%s); scanning chunks for the marker.",
                document_id,
                error,
            )
            offsets = None

        self._offsets[document_id] = offsets
        return offsets


def _item_value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _touched_ids(items: Any) -> Tuple[Set[str], Set[str]]:
    """The chunk and document ids the previous task produced.

    The cognify tail is handed ``TextSummary`` objects, which wrap their chunk in
    ``made_from``; other callers pass ``DocumentChunk`` objects (or dicts) directly.
    Anything without an id is ignored.
    """
    chunk_ids: Set[str] = set()
    document_ids: Set[str] = set()

    if items is None:
        return chunk_ids, document_ids
    if not isinstance(items, (list, tuple, set)):
        items = [items]

    for item in items:
        chunk = _item_value(item, "made_from") or item
        chunk_id = _item_value(chunk, "id")
        if chunk_id is None:
            continue
        chunk_ids.add(str(chunk_id))

        document = _item_value(chunk, "is_part_of")
        document_id = _item_value(document, "id") if document is not None else None
        if document_id is None:
            document_id = _item_value(chunk, "document_id")
        if document_id is not None:
            document_ids.add(str(document_id))

    return chunk_ids, document_ids


def _empty_summary() -> Dict[str, Any]:
    return {
        "scanned": 0,
        "already_resolved": 0,
        "resolved": 0,
        "resolved_by_strategy": {},
        "anchor_types": {},
        "unresolved": 0,
        "ambiguous": 0,
        "stale_ids": 0,
        "failed": 0,
        "edges_written": 0,
        "nodes_patched": 0,
        "dry_run": False,
        "notes": [],
    }


def _count(counter: Dict[str, int], key: Optional[str]) -> None:
    if key:
        counter[key] = counter.get(key, 0) + 1


def _resolve_existing_id(
    assertion_id: str,
    field_name: str,
    value: str,
    reference_text: str,
    view: GraphView,
) -> _Outcome:
    """Step a1: the field already holds an id -- make sure the edge exists.

    ``reference_text`` is the wording the document used when a previous pass preserved
    it, so the edge quotes the reference rather than the id that replaced it.
    """
    if value not in view.node_ids:
        logger.debug(
            "Reference %s.%s points at an unknown node %s.", assertion_id, field_name, value
        )
        return _Outcome("unresolved")

    if (assertion_id, value, field_name) in view.edge_keys:
        return _Outcome("already_resolved")

    props = view.node_props(value)
    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=assertion_id,
            field=field_name,
            reference_text=reference_text,
            strategy=STRATEGY_EXISTING_ID,
            confidence=_RESOLVED_ID_CONFIDENCE,
            anchor_id=value,
            anchor_type=props.get("type"),
            target_ids=(value,),
            target_type=props.get("type"),
        ),
    )


def _resolve_entity_name(
    assertion_id: str,
    field_name: str,
    reference_text: str,
    view: GraphView,
) -> Optional[_Outcome]:
    """Step a2: the reference names one entity. None means "not an entity name"."""
    entity_ids = view.entity_ids_by_name.get(generate_node_name(reference_text))
    if not entity_ids:
        return None

    if len(entity_ids) > 1:
        logger.debug(
            "Reference %s.%s names %d entities; leaving it unresolved.",
            assertion_id,
            field_name,
            len(entity_ids),
        )
        return _Outcome("ambiguous")

    entity_id = entity_ids[0]
    if (assertion_id, entity_id, field_name) in view.edge_keys:
        return _Outcome("already_resolved")

    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=assertion_id,
            field=field_name,
            reference_text=reference_text,
            strategy=STRATEGY_ENTITY_NAME,
            # The field keeps the name: the ingest contract reads it back as written.
            confidence=_RESOLVED_ID_CONFIDENCE,
            anchor_id=entity_id,
            anchor_type="Entity",
            target_ids=(entity_id,),
            target_type="Entity",
        ),
    )


async def _match_reference_document(
    reference: ParsedReference,
    view: GraphView,
    texts: DocumentTextCache,
    profiles: Sequence[Any],
    own_document_id: Optional[str],
) -> Tuple[Optional[str], float, Tuple[str, ...], bool]:
    """``(document id, confidence base, notes, ambiguous)`` for the named document."""
    match = match_document(reference, profiles, own_document_id=own_document_id)
    if match is None:
        return None, 0.0, (), False

    if not match.ambiguous_with:
        return match.document_id, match.score, (), False

    candidates = {}
    for candidate_id in (match.document_id, *match.ambiguous_with):
        text = await texts.text(candidate_id)
        if text:
            candidates[candidate_id] = text

    tiebreak = lexical_tiebreak(reference, candidates)
    if tiebreak is None:
        return None, 0.0, (), True

    document_id, lexical_score = tiebreak
    score = min(_TIEBREAK_MAX_CONFIDENCE, match.score + _TIEBREAK_LEXICAL_WEIGHT * lexical_score)
    return document_id, score, ("lexical_tiebreak",), False


async def _locate_span(
    reference: ParsedReference,
    document_id: str,
    view: GraphView,
    texts: DocumentTextCache,
) -> Optional[Tuple[str, List[dict], dict, Tuple[str, ...]]]:
    """``(span text, overlapping chunks, anchor chunk, notes)`` for a locator."""
    chunks = view.chunks_by_document.get(document_id) or []
    if not chunks:
        return None

    offsets = await texts.offsets(document_id, chunks)
    if offsets is not None:
        text = await texts.text(document_id)
        span = find_locator_span(text or "", reference.locator)
        if span is None:
            return None

        start, end, notes = span
        overlapping = [chunks[index] for index in chunks_overlapping(offsets, (start, end))]
        anchor_index = anchor_chunk_index(offsets, (start, end))
        anchor = chunks[anchor_index] if anchor_index is not None else None
        if anchor is None:
            return None
        return text[start:end], overlapping, anchor, notes

    # No usable offsets (no stored text, or chunks that do not tile it): find the marker
    # inside one chunk instead. Markers split across a chunk boundary are missed.
    scanned = scan_chunks_for_marker(
        [chunk.get("text") or "" for chunk in chunks], reference.locator
    )
    if scanned is None:
        return None

    index, (start, end) = scanned
    anchor = chunks[index]
    return (anchor.get("text") or "")[start:end], [anchor], anchor, ("chunk_scan_fallback",)


def _resolve_document_locator(
    assertion_id: str,
    field_name: str,
    reference_text: str,
    document_id: str,
    span_text: str,
    overlapping: Sequence[dict],
    anchor: dict,
    notes: Tuple[str, ...],
    view: GraphView,
) -> _Outcome:
    """Step b: a located paragraph, narrowed to the statements quoted inside it."""
    chunk_ids = {str(chunk["id"]) for chunk in overlapping}
    candidates = [
        (candidate_id, props.get("source_quote"))
        for candidate_id, props in view.assertions.items()
        if candidate_id != assertion_id and str(props.get("source_chunk_id") or "") in chunk_ids
    ]
    hits = select_anchored_assertions(span_text, candidates)
    anchor_id = str(anchor["id"])

    if hits:
        penalty = _LOCATOR_NOTE_PENALTY * len(set(notes) & _PENALIZED_NOTES)
        return _Outcome(
            "resolved",
            Resolution(
                assertion_id=assertion_id,
                field=field_name,
                reference_text=reference_text,
                strategy=STRATEGY_DOCUMENT_LOCATOR,
                confidence=_LOCATOR_CONFIDENCE - penalty,
                # One quoted statement is the reference; several are a passage, so the
                # field points at the chunk and the edges reach every statement.
                anchor_id=hits[0] if len(hits) == 1 else anchor_id,
                anchor_type="Assertion" if len(hits) == 1 else "DocumentChunk",
                target_ids=tuple(hits),
                target_type="Assertion",
                document_id=document_id,
                notes=notes,
            ),
        )

    # Nothing quoted the located passage. Linking every assertion in the chunk would
    # invent references; the chunk edge still lets a consumer reach the passage.
    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=assertion_id,
            field=field_name,
            reference_text=reference_text,
            strategy=STRATEGY_DOCUMENT_LOCATOR,
            confidence=_CHUNK_ONLY_CONFIDENCE,
            anchor_id=anchor_id,
            anchor_type="DocumentChunk",
            target_type="DocumentChunk",
            document_id=document_id,
            notes=notes,
        ),
    )


async def _prose_chunk(reference_text: str, chunks: Sequence[dict]) -> Optional[str]:
    """The one chunk a prose reference clearly describes, or None."""
    from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever

    retriever = BM25ChunksRetriever(top_k=_PROSE_TOP_K, with_scores=True)
    for chunk in chunks:
        text = chunk.get("text")
        if not text:
            continue
        tokens = retriever.tokenizer(text)
        if tokens:
            retriever.chunks[str(chunk["id"])] = tokens
            retriever.payloads[str(chunk["id"])] = chunk
    if not retriever.chunks:
        return None

    # Seeded straight from the view: the corpus is one document's chunks, so the parent's
    # graph-wide initialize() would be both a wasted query and the wrong corpus.
    retriever._initialized = True
    retriever._build_corpus_stats()
    retriever._stats_built = True

    scored = await retriever.get_retrieved_objects(reference_text)
    if not scored:
        return None

    top_payload, top_score = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else 0.0
    if top_score < _PROSE_MINIMUM_SCORE or top_score < _PROSE_MARGIN_RATIO * runner_up:
        return None

    return str(top_payload["id"])


def _resolve_document_only(
    assertion_id: str,
    field_name: str,
    reference_text: str,
    document_id: str,
    score: float,
    notes: Tuple[str, ...],
    view: GraphView,
) -> _Outcome:
    """Step c: the reference names a document and nothing inside it."""
    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=assertion_id,
            field=field_name,
            reference_text=reference_text,
            strategy=STRATEGY_DOCUMENT_ONLY,
            confidence=score,
            anchor_id=document_id,
            anchor_type=view.documents.get(document_id, {}).get("type"),
            target_type=view.documents.get(document_id, {}).get("type"),
            document_id=document_id,
            notes=notes,
        ),
    )


def _finalize(outcome: _Outcome, entry_notes: Tuple[str, ...], stale: bool) -> _Outcome:
    """Stamp the cascade's entry conditions onto whatever the cascade concluded."""
    if not entry_notes and not stale:
        return outcome

    resolution = outcome.resolution
    if resolution is not None and entry_notes:
        resolution = replace(resolution, notes=entry_notes + resolution.notes)
    return _Outcome(outcome.kind, resolution, stale)


def _edge_precheck(outcome: _Outcome, props: dict, view: GraphView) -> _Outcome:
    """Drop a document-strategy resolution whose edges the graph already holds.

    Without this a backend that cannot patch nodes re-plans the same resolution on every
    pass, and ``add_edges`` (a MERGE that overwrites the stored properties) would reset a
    ``feedback_weight`` ``improve()`` had tuned. Two cases once every edge is present:
    the field holds the anchor id, so there is nothing left to do (``already_resolved``);
    or it still holds its text, so the patch is the only outstanding half of the write
    and the resolution goes out marked :data:`NOTE_EDGES_EXIST`.
    """
    resolution = outcome.resolution
    if resolution is None or resolution.strategy not in _PATCHED_STRATEGIES:
        return outcome

    targets = set(resolution.target_ids)
    if resolution.anchor_id:
        targets.add(resolution.anchor_id)
    if not targets or any(
        (resolution.assertion_id, target_id, resolution.field) not in view.edge_keys
        for target_id in targets
    ):
        return outcome

    if _as_uuid(props.get(resolution.field)) is not None:
        return _Outcome("already_resolved")
    return _Outcome("resolved", replace(resolution, notes=resolution.notes + (NOTE_EDGES_EXIST,)))


async def _resolve_reference(
    assertion_id: str,
    field_name: str,
    props: dict,
    view: GraphView,
    texts: DocumentTextCache,
    profiles: Sequence[Any],
    *,
    force: bool,
    confidence_floor: float,
    enable_prose_lookup: bool,
) -> _Outcome:
    """Run the cascade for one ``(assertion, field)`` pair."""
    value = _text_of(props.get(field_name))
    reference_text = value
    entry_notes: Tuple[str, ...] = ()
    stale = False

    if _as_uuid(value) is not None:
        original = _text_of(props.get(f"{field_name}_text"))
        # An id that is no longer a node -- the target was forgotten, or an amended
        # document was re-chunked under new ids -- has gone dark, so it re-resolves from
        # the preserved wording without waiting for force. With no wording preserved
        # there is nothing to re-resolve from, and the reference stays unresolved.
        stale = value not in view.node_ids
        if original and (force or stale):
            # Re-resolve from the wording the document used, not from the id a previous
            # pass wrote into the field.
            reference_text = original
            entry_notes = (NOTE_STALE_ID,) if stale else ()
        else:
            return _finalize(
                _resolve_existing_id(assertion_id, field_name, value, original or value, view),
                entry_notes,
                stale,
            )

    entity_outcome = _resolve_entity_name(assertion_id, field_name, reference_text, view)
    if entity_outcome is not None:
        return _finalize(entity_outcome, entry_notes, stale)

    reference = parse_reference(reference_text)
    own_document_id = view.document_by_chunk.get(str(props.get("source_chunk_id") or ""))
    document_id, score, notes, ambiguous = await _match_reference_document(
        reference, view, texts, profiles, own_document_id
    )
    if ambiguous:
        return _finalize(_Outcome("ambiguous"), entry_notes, stale)
    if document_id is None:
        return _finalize(_Outcome("unresolved"), entry_notes, stale)

    outcome = None
    locator = reference.locator
    if locator is not None and locator.kind not in _DOCUMENT_LEVEL_LOCATOR_KINDS:
        located = await _locate_span(reference, document_id, view, texts)
        if located is None:
            # The document is established even though its text does not mark the place
            # the reference names -- a renumbered pleading, an unreadable original. The
            # document edge records what was established, the note records what was not,
            # and <field>_text keeps the wording so force=True can try again later.
            logger.debug(
                "Locator %s not found in document %s for %s.%s.",
                locator,
                document_id,
                assertion_id,
                field_name,
            )
            notes = notes + ("locator_not_found",)
        else:
            span_text, overlapping, anchor, span_notes = located
            outcome = _resolve_document_locator(
                assertion_id,
                field_name,
                reference_text,
                document_id,
                span_text,
                overlapping,
                anchor,
                notes + span_notes,
                view,
            )

    # Prose lookup narrows a locator-less reference to one chunk, but only ever at a
    # fixed 0.60. A floor above that rules the answer out before it is computed, so the
    # opt-in can never cost a resolution the document match alone would have supplied.
    if (
        outcome is None
        and enable_prose_lookup
        and locator is None
        and _PROSE_CONFIDENCE >= confidence_floor - _TOLERANCE
        and len(reference.hint_other_tokens) >= _PROSE_MINIMUM_TOKENS
    ):
        chunk_id = await _prose_chunk(
            reference_text, view.chunks_by_document.get(document_id) or []
        )
        if chunk_id is not None:
            outcome = _Outcome(
                "resolved",
                Resolution(
                    assertion_id=assertion_id,
                    field=field_name,
                    reference_text=reference_text,
                    strategy=STRATEGY_PROSE_LOOKUP,
                    confidence=_PROSE_CONFIDENCE,
                    anchor_id=chunk_id,
                    anchor_type="DocumentChunk",
                    target_type="DocumentChunk",
                    document_id=document_id,
                    notes=notes,
                ),
            )

    if outcome is None:
        outcome = _resolve_document_only(
            assertion_id, field_name, reference_text, document_id, score, notes, view
        )

    if (
        outcome.resolution is not None
        and outcome.resolution.confidence < confidence_floor - _TOLERANCE
    ):
        logger.debug(
            "Resolution for %s.%s scored %.2f, below the %.2f floor.",
            assertion_id,
            field_name,
            outcome.resolution.confidence,
            confidence_floor,
        )
        return _finalize(_Outcome("unresolved"), entry_notes, stale)

    return _finalize(_edge_precheck(outcome, props, view), entry_notes, stale)


async def plan_resolutions(
    view: GraphView,
    texts: DocumentTextCache,
    *,
    force: bool = False,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    enable_prose_lookup: bool = False,
    touched: Optional[Tuple[Set[str], Set[str]]] = None,
) -> Tuple[List[Resolution], Dict[str, Any]]:
    """Run the cascade over every dangling reference in the view.

    ``touched`` restricts the pass to one ingestion: a reference is in scope when the
    assertion carrying it came from a touched chunk, or when it names a touched document
    (an earlier document pointing at the one just ingested).
    """
    summary = _empty_summary()
    resolutions: List[Resolution] = []
    profiles = [
        document_profile(document_id, props.get("name") or "")
        for document_id, props in view.documents.items()
    ]

    for assertion_id, props in view.assertions.items():
        own_chunk_touched = touched is None or str(props.get("source_chunk_id") or "") in touched[0]

        for field_name in REFERENCE_FIELDS:
            if not _text_of(props.get(field_name)):
                continue

            try:
                outcome = await _resolve_reference(
                    assertion_id,
                    field_name,
                    props,
                    view,
                    texts,
                    profiles,
                    force=force,
                    confidence_floor=confidence_floor,
                    enable_prose_lookup=enable_prose_lookup,
                )
            except Exception as error:  # noqa: BLE001 - one bad reference must not stop the pass
                logger.warning(
                    "Could not resolve %s on assertion %s: %s", field_name, assertion_id, error
                )
                summary["scanned"] += 1
                summary["failed"] += 1
                continue

            if not own_chunk_touched:
                # Out of scope unless it points at a document this ingestion wrote.
                resolution = outcome.resolution
                if resolution is None or resolution.document_id not in touched[1]:
                    continue

            summary["scanned"] += 1
            if outcome.stale:
                summary["stale_ids"] += 1
            if outcome.kind == "resolved" and outcome.resolution is not None:
                resolutions.append(outcome.resolution)
                summary["resolved"] += 1
                _count(summary["resolved_by_strategy"], outcome.resolution.strategy)
                _count(summary["anchor_types"], outcome.resolution.anchor_type)
            else:
                summary[outcome.kind] += 1

    logger.info(
        "Reference resolution planned: scanned=%d resolved=%d already=%d unresolved=%d "
        "ambiguous=%d stale_ids=%d failed=%d",
        summary["scanned"],
        summary["resolved"],
        summary["already_resolved"],
        summary["unresolved"],
        summary["ambiguous"],
        summary["stale_ids"],
        summary["failed"],
    )
    return resolutions, summary


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
    edge a previous pass wrote cannot duplicate it -- but the upsert also overwrites that
    edge's stored properties, so a resolution the planner marked :data:`NOTE_EDGES_EXIST`
    writes no edge at all and is patched only.

    Indexing the new edge texts is the one step allowed to fail on its own: the edges are
    already stored, so the patches still run and the failure comes back as a note rather
    than as a half-applied write.
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

        source_props = view.assertions.get(resolution.assertion_id, {})
        endpoints[resolution.assertion_id] = source_props
        target_ids = list(resolution.target_ids)
        if resolution.anchor_id and resolution.anchor_id not in target_ids:
            target_ids.append(resolution.anchor_id)

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
                )
            )

    if dry_run:
        logger.info(
            "Reference resolution dry_run: %d edge(s) and %d patch(es) withheld.",
            len(edges),
            sum(1 for r in resolutions if r.strategy in _PATCHED_STRATEGIES),
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
                "edges are in the graph, their text is not in the edge index until the "
                "next indexing pass (improve()) re-embeds the graph's triplets.",
                len(edges),
                error,
            )
            summary["notes"].append(NOTE_EDGE_INDEX_FAILED)

    for resolution in resolutions:
        if resolution.strategy not in _PATCHED_STRATEGIES:
            continue

        values = build_node_patch(resolution, view.assertions.get(resolution.assertion_id, {}))
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

    Only ``already_resolved`` adds rather than replaces: the write phase reports the
    planned resolutions that turned out to need no write, and those stop being resolutions
    of this pass.
    """
    written = dict(write_summary)
    already = written.pop("already_resolved", 0)
    summary["already_resolved"] = summary.get("already_resolved", 0) + already
    summary["resolved"] = max(0, summary.get("resolved", 0) - already)
    summary.update(written)


def _dataset_id(ctx, dataset_id):
    """The dataset whose relational rows hold the document locations."""
    from_context = getattr(getattr(ctx, "dataset", None), "id", None)
    return from_context if from_context is not None else dataset_id


async def _plan(
    data,
    *,
    scope: str,
    force: bool,
    confidence_floor: float,
    enable_prose_lookup: bool,
    dataset_id,
    ctx,
) -> Tuple[Any, GraphView, List[Resolution], Dict[str, Any]]:
    graph_engine = await get_graph_engine()
    view = await _load_graph_view(graph_engine)
    texts = DocumentTextCache(view, dataset_id=_dataset_id(ctx, dataset_id))
    touched = _touched_ids(data) if scope == "touched" else None
    resolutions, summary = await plan_resolutions(
        view,
        texts,
        force=force,
        confidence_floor=confidence_floor,
        enable_prose_lookup=enable_prose_lookup,
        touched=touched,
    )
    return graph_engine, view, resolutions, summary


async def _provenance(graph_engine, ctx) -> Dict[str, Any]:
    return await graph_provenance_write_kwargs(
        graph_engine,
        ctx,
        fallback_data_id=REFERENCE_RESOLUTION_DATA_ID,
        pipeline_run_id=getattr(ctx, "pipeline_run_id", None),
    )


async def detect_dangling_references(
    data: Any = None,
    *,
    scope: str = "all",
    force: bool = False,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    enable_prose_lookup: bool = False,
    dataset_id=None,
    ctx=None,
) -> Dict[str, Any]:
    """Memify extraction phase: plan every reference resolution, write nothing.

    ``data`` is the memify seed and is ignored unless ``scope="touched"``, where it names
    the chunks and documents the current ingestion produced.
    """
    _, _, resolutions, summary = await _plan(
        data,
        scope=scope,
        force=force,
        confidence_floor=confidence_floor,
        enable_prose_lookup=enable_prose_lookup,
        dataset_id=dataset_id,
        ctx=ctx,
    )
    return {"plan": resolutions, "summary": summary}


def _unwrap_payload(payload: Any) -> Tuple[List[Resolution], Dict[str, Any]]:
    """Normalize the detect phase's output, tolerating the runner wrapping it in a list."""
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        return [], _empty_summary()

    summary = _empty_summary()
    summary.update(payload.get("summary") or {})
    return list(payload.get("plan") or []), summary


async def apply_reference_resolutions(
    payload: Any,
    *,
    dry_run: bool = False,
    ctx=None,
) -> Dict[str, Any]:
    """Memify enrichment phase: write the planned edges and node patches.

    Write failures are deliberately not swallowed here: a memify run that could not write
    is a visible error, unlike the ingest tail, which must never break its pipeline.
    """
    resolutions, summary = _unwrap_payload(payload)
    graph_engine = await get_graph_engine()
    view = await _load_graph_view(graph_engine)
    write_summary = await write_resolutions(
        graph_engine,
        view,
        resolutions,
        provenance_kwargs=await _provenance(graph_engine, ctx),
        dry_run=dry_run,
    )
    _merge_write_summary(summary, write_summary)
    return summary


@task_summary("Resolved references for {n} item(s)")
async def resolve_assertion_references(
    data: Any = None,
    *,
    scope: str = "all",
    force: bool = False,
    dry_run: bool = False,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    enable_prose_lookup: bool = False,
    ctx=None,
) -> Any:
    """Resolve dangling assertion references, then return the input unchanged.

    Args:
        data: The items the previous task produced. Only read when
            ``scope="touched"``, where they identify the ingestion to resolve around.
        scope: ``"touched"`` for an ingest tail (this document's references, and
            references pointing at it), ``"all"`` for the whole graph.
        force: Re-resolve references a previous pass already answered, from the
            ``<field>_text`` it preserved.
        dry_run: Plan and log without writing.
        confidence_floor: Resolutions below this confidence are left dangling.
        enable_prose_lookup: Opt into the BM25 chunk lookup for locator-less references.
        ctx: Pipeline context, used for provenance and the dataset's document locations.

    Returns:
        ``data``, unchanged, so the task can be appended to any pipeline.
    """
    # With scope="all" the whole graph is resolved in one go, and a pipeline that streams
    # several batches would otherwise repeat that identical pass once per batch.
    memoize = scope == "all" and ctx is not None
    if memoize and getattr(ctx, "extras", {}).get("reference_resolution_ran"):
        return data

    try:
        graph_engine, view, resolutions, summary = await _plan(
            data,
            scope=scope,
            force=force,
            confidence_floor=confidence_floor,
            enable_prose_lookup=enable_prose_lookup,
            dataset_id=None,
            ctx=ctx,
        )
        write_summary = await write_resolutions(
            graph_engine,
            view,
            resolutions,
            provenance_kwargs=await _provenance(graph_engine, ctx),
            dry_run=dry_run,
        )
        _merge_write_summary(summary, write_summary)
        # Memoized on success only: a pass that raised has resolved nothing, so the next
        # batch of the same run must be allowed to try again.
        if memoize:
            ctx.extras["reference_resolution_ran"] = True
    except Exception as error:  # noqa: BLE001 - resolution must never break ingestion
        logger.warning("Reference resolution skipped due to an error: %s", error)

    return data
