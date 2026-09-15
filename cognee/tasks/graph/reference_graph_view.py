"""Read layer for the assertion-reference resolver.

Everything here reads the graph and the stored text of the documents it names, and
nothing else: no LLM, no vector search, no writes. It used to live inline in
:mod:`cognee.tasks.graph.resolve_assertion_references`; it was split out so the seed
retrieval and tracer-tool tasks that only need to *read* the graph do not have to import
the resolution cascade to get at it.

:func:`_load_graph_view` runs one filtered graph query and indexes the result as a
:class:`GraphView`; :class:`DocumentTextCache` lazily reads and caches the stored text
(and chunk offsets) of the documents a pass actually touches.

``resolve_assertion_references`` imports these names back and re-exports them, so
existing callers (``scripts/legal/resolve_references_report.py``,
``scripts/legal/find_disputes.py``, and the resolver's own tests) keep working
unchanged.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID

from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.chunking.incremental_chunking import IncrementalPlanError, chunk_offsets
from cognee.modules.engine.utils.generate_node_name import generate_node_name
from cognee.modules.graph.utils.reference_resolution import RESOLVED_BY
from cognee.shared.logging_utils import get_logger

logger = get_logger("resolve_assertion_references")

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
    # (source id, relationship) of every edge a resolver pass wrote. On a backend that
    # cannot patch nodes this is the only record that a reference was already answered,
    # so it is what stops the next pass paying for the same trace again (R23).
    resolver_edge_keys: Set[Tuple[str, str]] = field(default_factory=set)
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

    An edge's properties come back with it, so an edge a previous pass wrote can be told
    from one the extraction did: it carries ``resolved_by="reference_resolver"``, and its
    ``(source, relationship)`` pair goes into ``resolver_edge_keys``.
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
        properties = edge[3] if len(edge) > 3 else None
        if isinstance(properties, dict) and properties.get("resolved_by") == RESOLVED_BY:
            view.resolver_edge_keys.add((source_id, relationship))
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
