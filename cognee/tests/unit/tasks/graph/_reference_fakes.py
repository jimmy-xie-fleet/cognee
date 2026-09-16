"""Shared fakes for the reference-resolution unit tests.

Seed retrieval (``test_reference_retrieval.py``) and the tracer tools/loop tests need the
same two things: a :class:`GraphView` built from plain node/edge tuples, and a vector
adapter that returns scripted :class:`ScoredResult` objects. Both live here so the test
modules stay small and cannot drift apart.

Nothing here touches a real backend: no embeddings, no network, no filesystem, no LLM.
``FakeVectorEngine`` is patched in at
``cognee.tasks.graph.reference_retrieval.get_vector_engine_async`` (or passed as the
``vector_engine=`` argument), and the view is built by the real ``_load_graph_view`` over a
graph object that only implements the one filtered read it performs.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import NAMESPACE_URL, uuid5

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.tasks.graph.reference_graph_view import GraphView, _load_graph_view


def nid(label: str) -> str:
    """A stable, valid-UUID node id (the graph and the vector index both hold UUIDs)."""
    return str(uuid5(NAMESPACE_URL, f"cognee-test:{label}"))


# --------------------------------------------------------------------------- #
# graph view
# --------------------------------------------------------------------------- #


def document_node(node_id: str, name: str, **props) -> Tuple[str, dict]:
    base = {"id": node_id, "type": "TextDocument", "name": name}
    base.update(props)
    return (node_id, base)


def chunk_node(
    node_id: str, text: str, chunk_index: int, document_id: str, **props
) -> Tuple[Tuple[str, dict], Tuple[str, str, str, dict]]:
    """A ``DocumentChunk`` node plus the ``is_part_of`` edge that puts it in a document."""
    base = {
        "id": node_id,
        "type": "DocumentChunk",
        "text": text,
        "chunk_index": chunk_index,
    }
    base.update(props)
    return (node_id, base), (node_id, document_id, "is_part_of", {})


def assertion_node(node_id: str, name: str, source_chunk_id: str, **props) -> Tuple[str, dict]:
    base = {
        "id": node_id,
        "type": "Assertion",
        "name": name,
        "statement_type": "allegation",
        "polarity": "positive",
        "source_chunk_id": source_chunk_id,
    }
    base.update(props)
    return (node_id, base)


class ViewOnlyGraph:
    """The one filtered read ``_load_graph_view`` performs, and nothing else."""

    def __init__(self, nodes: Sequence[Tuple[str, dict]], edges: Sequence[tuple]):
        self.nodes = {node_id: dict(props) for node_id, props in nodes}
        self.edges = [tuple(edge) for edge in edges]
        self.filtered_calls: List[Any] = []

    async def get_filtered_graph_data(self, attribute_filters):
        self.filtered_calls.append(attribute_filters)
        wanted = set()
        for attribute_filter in attribute_filters:
            for values in attribute_filter.values():
                wanted.update(values)
        nodes = [
            (node_id, dict(props))
            for node_id, props in self.nodes.items()
            if props.get("type") in wanted
        ]
        kept = {node_id for node_id, _ in nodes}
        edges = [edge for edge in self.edges if edge[0] in kept and edge[1] in kept]
        return nodes, edges


async def build_graph_view(nodes: Sequence[Tuple[str, dict]], edges: Sequence[tuple]) -> GraphView:
    """Index nodes/edges through the real ``_load_graph_view``, not by hand."""
    return await _load_graph_view(ViewOnlyGraph(nodes, edges))


# --------------------------------------------------------------------------- #
# vector engine
# --------------------------------------------------------------------------- #


def scored(node_id: str, distance: float, text: str = "", **payload) -> ScoredResult:
    """A ``ScoredResult`` shaped like the real adapters return it.

    ``score`` is a cosine **distance** (lower is better) and ``payload`` is
    ``IndexSchema``-shaped: ``id``/``text`` always, plus whichever of ``document_id``,
    ``document_name``, ``chunk_index`` and ``source_chunk_id`` the caller sets.
    """
    body: Dict[str, Any] = {"id": node_id, "text": text}
    body.update(payload)
    return ScoredResult(id=node_id, score=distance, payload=body)


class _FakeVectorEngineBase:
    """Scripted collections, recorded calls, no embeddings."""

    def __init__(
        self,
        results_by_collection: Optional[Dict[str, List[ScoredResult]]] = None,
        *,
        missing_collections: Sequence[str] = (),
        raising_collections: Sequence[str] = (),
    ):
        self.results_by_collection = dict(results_by_collection or {})
        # ``has_collection`` answers False for these (the collection was never created).
        self.missing_collections = set(missing_collections)
        # ``search``/``batch_search`` raise ``CollectionNotFoundError`` for these (the
        # collection vanished between the guard and the query).
        self.raising_collections = set(raising_collections)
        self.has_collection_calls: List[str] = []
        self.search_calls: List[tuple] = []
        self.batch_search_calls: List[tuple] = []

    async def has_collection(self, collection_name: str) -> bool:
        self.has_collection_calls.append(collection_name)
        if collection_name in self.missing_collections:
            return False
        return (
            collection_name in self.results_by_collection
            or collection_name in self.raising_collections
        )

    def _results(self, collection_name: str, limit: Optional[int]) -> List[ScoredResult]:
        if collection_name in self.raising_collections:
            raise CollectionNotFoundError(f"{collection_name} not found", log=False)
        results = self.results_by_collection.get(collection_name, [])
        results = [result.model_copy(deep=True) for result in results]
        return results if limit is None else results[:limit]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ) -> List[ScoredResult]:
        self.search_calls.append((collection_name, query_text, limit, include_payload))
        return self._results(collection_name, limit)


class FakeVectorEngine(_FakeVectorEngineBase):
    """The full adapter surface seed retrieval uses: ``has_collection``/``search``/``batch_search``."""

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int] = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
    ) -> List[List[ScoredResult]]:
        self.batch_search_calls.append((collection_name, list(query_texts), limit, include_payload))
        return [self._results(collection_name, limit) for _ in query_texts]


class SearchOnlyVectorEngine(_FakeVectorEngineBase):
    """An adapter with no ``batch_search`` at all: the per-query fallback path."""
