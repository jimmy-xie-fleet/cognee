"""Unit tests for the deterministic assertion-reference resolver.

Everything here runs against a stateful ``FakeGraph``: no graph backend, no vector
backend, no LLM, no filesystem. The fake applies what the task writes (``add_edges``
upserts on the triple, ``update_node`` merges into the node blob), so a second pass
reads back the first pass's writes and idempotency is a real assertion rather than a
mock's call count.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import NAMESPACE_URL, uuid5

import pytest

from cognee.modules.graph.utils.reference_resolution import (
    STRATEGY_DOCUMENT_LOCATOR,
    STRATEGY_DOCUMENT_ONLY,
    STRATEGY_ENTITY_NAME,
    STRATEGY_EXISTING_ID,
    STRATEGY_PROSE_LOOKUP,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.graph.resolve_assertion_references import (
    DOCUMENT_NODE_TYPES,
    REFERENCE_FIELDS,
    REFERENCE_RESOLUTION_DATA_ID,
    _touched_ids,
    apply_reference_resolutions,
    detect_dangling_references,
    resolve_assertion_references,
)

MODULE = "cognee.tasks.graph.resolve_assertion_references"


def _nid(label: str) -> str:
    """A stable, valid-UUID node id, so ``existing_id`` sees real UUIDs."""
    return str(uuid5(NAMESPACE_URL, f"cognee-test:{label}"))


DATASET_ID = _nid("dataset")

DOC_COMPLAINT = _nid("doc-complaint")
DOC_ANSWER = _nid("doc-answer")
DOC_STIPULATION = _nid("doc-stipulation")

COMPLAINT_CHUNK_0 = _nid("complaint-chunk-0")
COMPLAINT_CHUNK_1 = _nid("complaint-chunk-1")
ANSWER_CHUNK_0 = _nid("answer-chunk-0")

A_1 = _nid("assertion-owns")
A_2 = _nid("assertion-acquired")
A_3 = _nid("assertion-roof")
A_DENIAL = _nid("assertion-denial")
A_ATTRIBUTED = _nid("assertion-attributed")
A_RESOLVED = _nid("assertion-already-resolved")
A_STIPULATION = _nid("assertion-stipulation")
ENTITY_FESTER = _nid("entity-norman-fester")

COMPLAINT_LOCATION = "file://complaint.txt"
ANSWER_LOCATION = "file://answer.txt"
STIPULATION_LOCATION = "file://stipulation.txt"

# Two chunks that tile the complaint's text; the "5." marker ends chunk 0 and the
# paragraph body opens chunk 1, which is exactly the boundary the anchor rule is for.
COMPLAINT_CHUNK_0_TEXT = "COMPLAINT\n\n4. The parties entered into a lease in 1997.\n\n5. "
COMPLAINT_CHUNK_1_TEXT = (
    "Clifton owns 10 Main Street, acquired in 1998.\n\n6. Roof replaced in 2019.\n"
)
COMPLAINT_TEXT = COMPLAINT_CHUNK_0_TEXT + COMPLAINT_CHUNK_1_TEXT
ANSWER_CHUNK_0_TEXT = "ANSWER\n\n1. Defendant denies the allegations of paragraph 5.\n"
ANSWER_TEXT = ANSWER_CHUNK_0_TEXT

CHUNK_1_LABEL = "Clifton owns 10 Main Street, acquired in 1998. 6. Roof replaced in 2019."

DEFAULT_TEXTS = {
    COMPLAINT_LOCATION: COMPLAINT_TEXT,
    ANSWER_LOCATION: ANSWER_TEXT,
    STIPULATION_LOCATION: "STIPULATION\n\nThe parties stipulate.\n",
}

PROVENANCE_KWARGS = {"source_ref_key": "dataset:data", "pipeline_run_id": "run-1"}


def _document(node_id, name, location):
    return (
        node_id,
        {"id": node_id, "type": "TextDocument", "name": name, "raw_data_location": location},
    )


def _chunk(node_id, text, index, document_id):
    return (
        node_id,
        {"id": node_id, "type": "DocumentChunk", "text": text, "chunk_index": index},
    ), (node_id, document_id, "is_part_of", {})


def _assertion(node_id, name, **props):
    base = {
        "id": node_id,
        "type": "Assertion",
        "name": name,
        "statement_type": "allegation",
        "polarity": "positive",
    }
    base.update(props)
    return (node_id, base)


def _base_graph():
    """Complaint + Answer + Stipulation, with one reference of every resolvable shape."""
    nodes = [
        _document(DOC_COMPLAINT, "Verified_Complaint_Adams_v_Clifton", COMPLAINT_LOCATION),
        _document(DOC_ANSWER, "Answer_Clifton", ANSWER_LOCATION),
        _document(DOC_STIPULATION, "Stipulation_Adams", STIPULATION_LOCATION),
        _assertion(
            A_1,
            "Clifton owns 10 Main Street",
            source_chunk_id=COMPLAINT_CHUNK_1,
            source_quote="Clifton owns 10 Main Street",
        ),
        _assertion(
            A_2,
            "The property was acquired in 1998",
            source_chunk_id=COMPLAINT_CHUNK_1,
            source_quote="acquired in 1998",
        ),
        _assertion(
            A_3,
            "The roof was replaced in 2019",
            source_chunk_id=COMPLAINT_CHUNK_1,
            source_quote="Roof replaced in 2019",
        ),
        _assertion(
            A_DENIAL,
            "Clifton owns 10 Main Street",
            statement_type="denial",
            polarity="negative",
            source_chunk_id=ANSWER_CHUNK_0,
            responds_to="Complaint ¶5",
        ),
        _assertion(
            A_ATTRIBUTED,
            "The valuation is unsupported",
            source_chunk_id=ANSWER_CHUNK_0,
            attributed_to="norman fester",
        ),
        _assertion(
            A_RESOLVED,
            "The lease began in 1997",
            source_chunk_id=ANSWER_CHUNK_0,
            responds_to=A_1,
            responds_to_text="Complaint ¶4",
        ),
        _assertion(
            A_STIPULATION,
            "The parties stipulated to the boundary",
            source_chunk_id=ANSWER_CHUNK_0,
            responds_to="Stipulation Adams",
        ),
        (ENTITY_FESTER, {"id": ENTITY_FESTER, "type": "Entity", "name": "Norman Fester"}),
    ]
    edges = [(A_RESOLVED, A_1, "responds_to", {})]
    for node, edge in (
        _chunk(COMPLAINT_CHUNK_0, COMPLAINT_CHUNK_0_TEXT, 0, DOC_COMPLAINT),
        _chunk(COMPLAINT_CHUNK_1, COMPLAINT_CHUNK_1_TEXT, 1, DOC_COMPLAINT),
        _chunk(ANSWER_CHUNK_0, ANSWER_CHUNK_0_TEXT, 0, DOC_ANSWER),
    ):
        nodes.append(node)
        edges.append(edge)
    return FakeGraph(nodes, edges)


class FakeGraph:
    """A graph adapter that really stores what the resolver writes."""

    def __init__(self, nodes, edges):
        self.nodes = {node_id: dict(props) for node_id, props in nodes}
        self.edges = [tuple(edge) for edge in edges]
        self.add_edges_calls = []
        self.update_node_calls = []
        self.update_node_supported = True
        self.filtered_calls = []

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

    async def add_edges(self, edges, **kwargs):
        self.add_edges_calls.append((list(edges), kwargs))
        known = {(edge[0], edge[1], edge[2]) for edge in self.edges}
        for edge in edges:
            if (edge[0], edge[1], edge[2]) not in known:
                self.edges.append(tuple(edge))
                known.add((edge[0], edge[1], edge[2]))

    async def update_node(self, node_id, values):
        if not self.update_node_supported:
            raise NotImplementedError("update_node is not implemented for this adapter")
        self.update_node_calls.append((node_id, dict(values)))
        if node_id not in self.nodes:
            return False
        self.nodes[node_id].update(values)
        return True

    @property
    def written_edges(self):
        return [edge for call in self.add_edges_calls for edge in call[0]]

    def edges_of(self, source_id, relationship):
        return [
            edge for edge in self.written_edges if edge[0] == source_id and edge[2] == relationship
        ]


@contextmanager
def _patched(graph, texts=None, *, locations=None):
    """Patch every I/O seam the task reaches through."""
    texts = DEFAULT_TEXTS if texts is None else texts

    async def _read(location):
        value = texts[location]
        if isinstance(value, Exception):
            raise value
        return value

    with (
        patch(f"{MODULE}.get_graph_engine", new=AsyncMock(return_value=graph)),
        patch(f"{MODULE}.index_graph_edges", new=AsyncMock()) as index_mock,
        patch(
            f"{MODULE}.graph_provenance_write_kwargs",
            new=AsyncMock(return_value=dict(PROVENANCE_KWARGS)),
        ) as provenance_mock,
        patch(
            "cognee.tasks.graph.reference_graph_view._raw_locations",
            new=AsyncMock(return_value=locations or {}),
        ),
        patch(
            "cognee.tasks.graph.reference_graph_view._read_processed_text",
            new=AsyncMock(side_effect=_read),
        ) as read_mock,
    ):
        yield SimpleNamespace(
            index_graph_edges=index_mock,
            graph_provenance_write_kwargs=provenance_mock,
            read_processed_text=read_mock,
        )


async def _run(graph, texts=None, *, locations=None, ctx=None, dry_run=False, **detect_kwargs):
    """Run the two-phase pass and return ``(payload, summary, patch mocks)``."""
    data = detect_kwargs.pop("data", None)
    with _patched(graph, texts, locations=locations) as mocks:
        payload = await detect_dangling_references(data, ctx=ctx, **detect_kwargs)
        summary = await apply_reference_resolutions(payload, dry_run=dry_run, ctx=ctx)
    return payload, summary, mocks


def _props(edge):
    return edge[3]


def _by_target(edges):
    return {edge[1]: edge for edge in edges}


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def test_reference_fields_never_include_the_identity_field():
    assert REFERENCE_FIELDS == ("responds_to", "attributed_to")
    assert "asserted_by" not in REFERENCE_FIELDS
    assert REFERENCE_RESOLUTION_DATA_ID == uuid5(NAMESPACE_URL, "cognee:reference-resolution")
    assert DOCUMENT_NODE_TYPES == (
        "TextDocument",
        "PdfDocument",
        "UnstructuredDocument",
        "AudioDocument",
        "ImageDocument",
    )


# --------------------------------------------------------------------------- #
# the worked example: Answer denial -> Complaint ¶5 -> two allegations
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_locator_reference_links_both_allegations_and_the_anchor_chunk():
    graph = _base_graph()
    _, summary, mocks = await _run(graph)

    denial_edges = _by_target(graph.edges_of(A_DENIAL, "responds_to"))
    assert set(denial_edges) == {A_1, A_2, COMPLAINT_CHUNK_1}
    # ¶6's allegation is outside the located span and must not be linked.
    assert A_3 not in denial_edges

    allegation_props = _props(denial_edges[A_1])
    assert allegation_props["relationship_name"] == "responds_to"
    assert allegation_props["source_node_id"] == A_DENIAL
    assert allegation_props["target_node_id"] == A_1
    assert allegation_props["reference_text"] == "Complaint ¶5"
    assert allegation_props["resolution_strategy"] == STRATEGY_DOCUMENT_LOCATOR
    assert allegation_props["resolution_confidence"] == pytest.approx(0.90)
    assert allegation_props["resolved_target_type"] == "Assertion"
    assert allegation_props["resolved_by"] == "reference_resolver"
    # The stance travels with the edge: a denial must not read as the fact it denies.
    assert allegation_props["edge_text"] == (
        "Clifton owns 10 Main Street (denial, negative stance) responds to "
        "Clifton owns 10 Main Street."
    )
    # ensure_default_edge_properties filled the storage defaults without touching edge_text.
    assert allegation_props["edge_object_id"]
    assert allegation_props["feedback_weight"] == 0.5

    chunk_props = _props(denial_edges[COMPLAINT_CHUNK_1])
    assert chunk_props["resolved_target_type"] == "DocumentChunk"
    assert chunk_props["edge_text"] == (
        f"Clifton owns 10 Main Street (denial, negative stance) responds to {CHUNK_1_LABEL}"
    )

    patches = dict(graph.update_node_calls)
    assert A_DENIAL in patches
    patch_values = patches[A_DENIAL]
    assert patch_values["responds_to"] == COMPLAINT_CHUNK_1
    assert patch_values["responds_to_text"] == "Complaint ¶5"
    resolution = patch_values["responds_to_resolution"]
    assert resolution["strategy"] == STRATEGY_DOCUMENT_LOCATOR
    assert resolution["confidence"] == pytest.approx(0.90)
    assert resolution["target_ids"] == [A_1, A_2]
    assert resolution["anchor_id"] == COMPLAINT_CHUNK_1
    assert resolution["document_id"] == DOC_COMPLAINT
    assert resolution["notes"] == []

    assert summary["scanned"] == 4
    assert summary["resolved"] == 3
    assert summary["already_resolved"] == 1
    assert summary["resolved_by_strategy"][STRATEGY_DOCUMENT_LOCATOR] == 1
    assert summary["anchor_types"]["DocumentChunk"] == 1
    assert summary["failed"] == 0
    assert summary["dry_run"] is False
    assert summary["notes"] == []

    # Edges are written in one batch, carrying exactly the provenance kwargs.
    assert len(graph.add_edges_calls) == 1
    written, kwargs = graph.add_edges_calls[0]
    assert kwargs == PROVENANCE_KWARGS
    assert len(written) == 5
    assert summary["edges_written"] == 5
    assert summary["nodes_patched"] == 2
    mocks.index_graph_edges.assert_awaited_once_with(written)
    assert (
        mocks.graph_provenance_write_kwargs.await_args.kwargs["fallback_data_id"]
        == REFERENCE_RESOLUTION_DATA_ID
    )


@pytest.mark.asyncio
async def test_entity_name_reference_emits_an_edge_and_leaves_the_field_alone():
    graph = _base_graph()
    await _run(graph)

    edges = graph.edges_of(A_ATTRIBUTED, "attributed_to")
    assert [edge[1] for edge in edges] == [ENTITY_FESTER]
    props = _props(edges[0])
    assert props["resolution_strategy"] == STRATEGY_ENTITY_NAME
    assert props["resolution_confidence"] == pytest.approx(1.0)
    assert props["resolved_target_type"] == "Entity"
    assert props["reference_text"] == "norman fester"

    # The ingest contract keeps the name in the field: entity_name never patches a node.
    assert A_ATTRIBUTED not in dict(graph.update_node_calls)
    assert graph.nodes[A_ATTRIBUTED]["attributed_to"] == "norman fester"


@pytest.mark.asyncio
async def test_document_only_reference_anchors_on_the_document_node():
    graph = _base_graph()
    _, summary, _ = await _run(graph)

    edges = graph.edges_of(A_STIPULATION, "responds_to")
    assert [edge[1] for edge in edges] == [DOC_STIPULATION]
    props = _props(edges[0])
    assert props["resolution_strategy"] == STRATEGY_DOCUMENT_ONLY
    assert props["resolved_target_type"] == "TextDocument"
    assert props["resolution_confidence"] == pytest.approx(0.90)

    patch_values = dict(graph.update_node_calls)[A_STIPULATION]
    assert patch_values["responds_to"] == DOC_STIPULATION
    assert patch_values["responds_to_text"] == "Stipulation Adams"
    assert summary["anchor_types"]["TextDocument"] == 1


@pytest.mark.asyncio
async def test_document_level_locator_resolves_to_the_whole_document():
    """ "Resolution No. 2026-118" names a document, not a place inside one."""
    graph = _base_graph()
    document_id = _nid("doc-resolution")
    graph.nodes[document_id] = {
        "id": document_id,
        "type": "TextDocument",
        "name": "Resolution_2026-118",
        "raw_data_location": "file://resolution.txt",
    }
    graph.nodes[A_STIPULATION]["responds_to"] = "Resolution No. 2026-118"

    _, summary, _ = await _run(graph)

    edges = graph.edges_of(A_STIPULATION, "responds_to")
    assert [edge[1] for edge in edges] == [document_id]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_DOCUMENT_ONLY
    assert _props(edges[0])["resolution_confidence"] == pytest.approx(1.0)
    assert summary["unresolved"] == 0


@pytest.mark.asyncio
async def test_already_resolved_uuid_is_counted_and_writes_nothing():
    graph = _base_graph()
    _, summary, _ = await _run(graph)

    assert summary["already_resolved"] == 1
    assert graph.edges_of(A_RESOLVED, "responds_to") == []


@pytest.mark.asyncio
async def test_existing_uuid_without_an_edge_emits_the_edge_only():
    graph = _base_graph()
    graph.edges = [edge for edge in graph.edges if edge[0] != A_RESOLVED]

    _, summary, _ = await _run(graph)

    edges = graph.edges_of(A_RESOLVED, "responds_to")
    assert [edge[1] for edge in edges] == [A_1]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_EXISTING_ID
    assert _props(edges[0])["resolution_confidence"] == pytest.approx(1.0)
    # The edge quotes the wording the document used, not the id the field holds.
    assert _props(edges[0])["reference_text"] == "Complaint ¶4"
    # existing_id never patches: the field already holds the id.
    assert A_RESOLVED not in dict(graph.update_node_calls)
    assert summary["already_resolved"] == 0


@pytest.mark.asyncio
async def test_stale_uuid_without_preserved_text_stays_unresolved_but_is_counted():
    graph = _base_graph()
    graph.nodes[A_RESOLVED]["responds_to"] = _nid("missing-node")
    del graph.nodes[A_RESOLVED]["responds_to_text"]
    graph.edges = [edge for edge in graph.edges if edge[0] != A_RESOLVED]

    _, summary, _ = await _run(graph)

    assert graph.edges_of(A_RESOLVED, "responds_to") == []
    assert summary["unresolved"] == 1
    # Nothing to re-resolve from, but the dark reference is still reported.
    assert summary["stale_ids"] == 1


@pytest.mark.asyncio
async def test_stale_uuid_re_resolves_from_the_preserved_text_without_force():
    """An amended pleading is re-chunked: the id the field holds is no longer a node."""
    graph = _base_graph()
    await _run(graph)
    assert graph.nodes[A_DENIAL]["responds_to"] == COMPLAINT_CHUNK_1

    rechunked = _nid("complaint-chunk-1-rechunked")
    graph.nodes[rechunked] = dict(graph.nodes.pop(COMPLAINT_CHUNK_1), id=rechunked)
    graph.edges = [edge for edge in graph.edges if COMPLAINT_CHUNK_1 not in (edge[0], edge[1])]
    graph.edges.append((rechunked, DOC_COMPLAINT, "is_part_of", {}))
    for assertion_id in (A_1, A_2, A_3):
        graph.nodes[assertion_id]["source_chunk_id"] = rechunked
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, _ = await _run(graph)

    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {A_1, A_2, rechunked}
    patch_values = dict(graph.update_node_calls)[A_DENIAL]
    assert patch_values["responds_to"] == rechunked
    assert patch_values["responds_to_text"] == "Complaint ¶5"
    assert patch_values["responds_to_resolution"]["notes"] == ["stale_id"]
    assert summary["stale_ids"] == 1
    assert summary["resolved_by_strategy"][STRATEGY_DOCUMENT_LOCATOR] == 1


# --------------------------------------------------------------------------- #
# idempotency, force, dry_run
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_second_pass_writes_nothing():
    graph = _base_graph()
    await _run(graph)
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, _ = await _run(graph)

    assert graph.add_edges_calls == []
    assert graph.update_node_calls == []
    assert summary["edges_written"] == 0
    assert summary["nodes_patched"] == 0
    assert summary["resolved"] == 0
    assert summary["already_resolved"] == 4


@pytest.mark.asyncio
async def test_force_re_resolves_from_the_stored_reference_text():
    graph = _base_graph()
    await _run(graph)
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, _ = await _run(graph, force=True)

    # The already-resolved reference is re-read from its stored "Complaint ¶4" and now
    # anchors on that paragraph's chunk instead of the single assertion it held.
    assert dict(graph.update_node_calls)[A_RESOLVED]["responds_to"] == COMPLAINT_CHUNK_0
    assert summary["resolved_by_strategy"][STRATEGY_DOCUMENT_LOCATOR] == 1
    # The denial re-resolves to the same answer, so the edge pre-check stops force from
    # re-writing edges whose properties (a tuned feedback_weight) would be reset.
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    assert A_DENIAL not in dict(graph.update_node_calls)
    assert summary["already_resolved"] == 3


@pytest.mark.asyncio
async def test_dry_run_plans_without_writing():
    graph = _base_graph()
    payload, summary, mocks = await _run(graph, dry_run=True)

    assert graph.add_edges_calls == []
    assert graph.update_node_calls == []
    assert summary["dry_run"] is True
    assert summary["edges_written"] == 0
    assert summary["nodes_patched"] == 0
    assert summary["resolved"] == 3
    assert len(payload["plan"]) == 3
    mocks.index_graph_edges.assert_not_awaited()


# --------------------------------------------------------------------------- #
# fallbacks
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_non_tiling_chunks_fall_back_to_a_chunk_scan(caplog):
    """An overlapping chunker breaks chunk_offsets; the marker is scanned per chunk."""
    graph = _base_graph()
    overlap_0 = "COMPLAINT\n\n4. A lease.\n\n5. Clifton owns 10 Main Street, acquired in 1998.\n"
    overlap_1 = "5. Clifton owns 10 Main Street, acquired in 1998.\n\n6. Roof replaced in 2019.\n"
    graph.nodes[COMPLAINT_CHUNK_0]["text"] = overlap_0
    graph.nodes[COMPLAINT_CHUNK_1]["text"] = overlap_1
    graph.nodes[A_1]["source_chunk_id"] = COMPLAINT_CHUNK_0
    graph.nodes[A_2]["source_chunk_id"] = COMPLAINT_CHUNK_0

    with caplog.at_level("WARNING"):
        _, summary, _ = await _run(graph)

    edges = _by_target(graph.edges_of(A_DENIAL, "responds_to"))
    assert set(edges) == {A_1, A_2, COMPLAINT_CHUNK_0}
    assert _props(edges[A_1])["resolution_confidence"] == pytest.approx(0.80)
    resolution = dict(graph.update_node_calls)[A_DENIAL]["responds_to_resolution"]
    assert resolution["notes"] == ["chunk_scan_fallback"]
    assert summary["resolved_by_strategy"][STRATEGY_DOCUMENT_LOCATOR] == 1
    assert any(
        "do not tile" in record.message or "tile" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_missing_document_location_falls_back_to_a_chunk_scan():
    graph = _base_graph()
    del graph.nodes[DOC_COMPLAINT]["raw_data_location"]

    _, summary, mocks = await _run(graph)

    edges = _by_target(graph.edges_of(A_DENIAL, "responds_to"))
    # Only chunk 0 carries the "5." marker, and it holds no quoted assertion.
    assert set(edges) == {COMPLAINT_CHUNK_0}
    assert _props(edges[COMPLAINT_CHUNK_0])["resolution_confidence"] == pytest.approx(0.65)
    assert summary["resolved_by_strategy"][STRATEGY_DOCUMENT_LOCATOR] == 1
    mocks.read_processed_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_location_fallback_reads_the_relational_row():
    graph = _base_graph()
    del graph.nodes[DOC_COMPLAINT]["raw_data_location"]

    _, summary, _ = await _run(
        graph, locations={DOC_COMPLAINT: COMPLAINT_LOCATION}, dataset_id=DATASET_ID
    )

    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {
        A_1,
        A_2,
        COMPLAINT_CHUNK_1,
    }
    assert summary["failed"] == 0


@pytest.mark.asyncio
async def test_reference_to_an_absent_document_stays_unresolved():
    graph = _base_graph()
    graph.nodes[A_DENIAL]["responds_to"] = "Subpoena ¶3"

    _, summary, _ = await _run(graph)

    assert graph.edges_of(A_DENIAL, "responds_to") == []
    assert A_DENIAL not in dict(graph.update_node_calls)
    assert summary["unresolved"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        FileNotFoundError("the stored file moved"),
    ],
    ids=["binary-document", "missing-file"],
)
async def test_unreadable_text_degrades_to_the_chunk_scan(error):
    """A PDF or a moved file still names a matched document: use its stored chunks."""
    graph = _base_graph()
    texts = dict(DEFAULT_TEXTS)
    texts[COMPLAINT_LOCATION] = error

    _, summary, mocks = await _run(graph, texts)

    edges = _by_target(graph.edges_of(A_DENIAL, "responds_to"))
    assert set(edges) == {COMPLAINT_CHUNK_0}
    assert _props(edges[COMPLAINT_CHUNK_0])["resolution_strategy"] == STRATEGY_DOCUMENT_LOCATOR
    assert _props(edges[COMPLAINT_CHUNK_0])["resolution_confidence"] == pytest.approx(0.65)
    assert summary["failed"] == 0
    # The doomed open is cached: one attempt per document, not one per reference.
    assert mocks.read_processed_text.await_count == 1


@pytest.mark.asyncio
async def test_unexpected_read_error_fails_only_its_own_reference():
    """An error the reader was not expected to raise still counts as a failure."""
    graph = _base_graph()
    texts = dict(DEFAULT_TEXTS)
    texts[COMPLAINT_LOCATION] = RuntimeError("reader exploded")

    _, summary, _ = await _run(graph, texts)

    assert summary["failed"] == 1
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    # The other references in the same pass still resolve.
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to")
    assert graph.edges_of(A_STIPULATION, "responds_to")


@pytest.mark.asyncio
async def test_missing_locator_falls_back_to_the_document():
    """The document is established even when its text does not mark the paragraph."""
    graph = _base_graph()
    graph.nodes[A_DENIAL]["responds_to"] = "Complaint ¶99"

    _, summary, _ = await _run(graph)

    edges = graph.edges_of(A_DENIAL, "responds_to")
    assert [edge[1] for edge in edges] == [DOC_COMPLAINT]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_DOCUMENT_ONLY
    assert _props(edges[0])["resolution_confidence"] == pytest.approx(0.60)

    resolution = dict(graph.update_node_calls)[A_DENIAL]["responds_to_resolution"]
    assert resolution["notes"] == ["locator_not_found"]
    assert resolution["anchor_id"] == DOC_COMPLAINT
    assert summary["unresolved"] == 0


@pytest.mark.asyncio
async def test_missing_locator_in_the_chunk_scan_falls_back_to_the_document():
    graph = _base_graph()
    del graph.nodes[DOC_COMPLAINT]["raw_data_location"]
    graph.nodes[A_DENIAL]["responds_to"] = "Complaint ¶99"

    _, summary, _ = await _run(graph)

    edges = graph.edges_of(A_DENIAL, "responds_to")
    assert [edge[1] for edge in edges] == [DOC_COMPLAINT]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_DOCUMENT_ONLY
    resolution = dict(graph.update_node_calls)[A_DENIAL]["responds_to_resolution"]
    assert resolution["notes"] == ["locator_not_found"]
    assert summary["unresolved"] == 0


@pytest.mark.asyncio
async def test_node_patches_are_skipped_when_the_adapter_cannot_patch():
    graph = _base_graph()
    graph.update_node_supported = False

    _, summary, _ = await _run(graph)

    assert summary["edges_written"] == 5
    assert summary["nodes_patched"] == 0
    assert summary["notes"] == ["node_patch_unsupported"]


@pytest.mark.asyncio
async def test_a_second_pass_rewrites_nothing_when_the_adapter_cannot_patch():
    """Without update_node the field keeps its text, so only the edge pre-check can stop
    a second pass from re-emitting (and resetting the properties of) the same edges."""
    graph = _base_graph()
    graph.update_node_supported = False

    _, first, _ = await _run(graph)
    assert first["edges_written"] == 5
    graph.add_edges_calls.clear()

    _, summary, mocks = await _run(graph)

    assert graph.add_edges_calls == []
    assert summary["edges_written"] == 0
    assert summary["nodes_patched"] == 0
    assert summary["resolved"] == 0
    assert summary["already_resolved"] == 4
    mocks.index_graph_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_edge_indexing_failure_still_patches_and_is_noted(caplog):
    graph = _base_graph()

    with _patched(graph) as mocks:
        mocks.index_graph_edges.side_effect = RuntimeError("embedding provider is down")
        with caplog.at_level("WARNING"):
            payload = await detect_dangling_references(None)
            summary = await apply_reference_resolutions(payload)

    assert summary["edges_written"] == 5
    assert summary["nodes_patched"] == 2
    assert "edge_index_failed" in summary["notes"]
    assert any("index" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_graph_view_falls_back_when_filtering_is_unsupported():
    graph = _base_graph()
    nodes = [(node_id, dict(props)) for node_id, props in graph.nodes.items()]
    edges = list(graph.edges)

    async def _unsupported(_attribute_filters):
        raise NotImplementedError("get_filtered_graph_data is not implemented")

    graph.get_filtered_graph_data = _unsupported
    graph.get_graph_data = AsyncMock(return_value=(nodes, edges))

    _, summary, _ = await _run(graph)

    graph.get_graph_data.assert_awaited()
    assert summary["resolved"] == 3


# --------------------------------------------------------------------------- #
# entity lookup
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_ambiguous_entity_name_resolves_nothing():
    graph = _base_graph()
    twin = _nid("entity-norman-fester-twin")
    graph.nodes[twin] = {"id": twin, "type": "Entity", "name": "NORMAN FESTER"}

    _, summary, _ = await _run(graph)

    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert summary["ambiguous"] == 1


@pytest.mark.asyncio
async def test_assertion_names_are_not_entity_name_targets():
    """Assertions subclass Entity in the model but must never be entity_name targets."""
    graph = _base_graph()
    graph.nodes[A_ATTRIBUTED]["attributed_to"] = "Clifton owns 10 Main Street"

    _, summary, _ = await _run(graph)

    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert summary["unresolved"] == 1


# --------------------------------------------------------------------------- #
# scope
# --------------------------------------------------------------------------- #
def test_touched_ids_reads_summaries_chunks_and_dicts():
    summary_item = SimpleNamespace(
        made_from=SimpleNamespace(
            id=COMPLAINT_CHUNK_1, is_part_of=SimpleNamespace(id=DOC_COMPLAINT)
        )
    )
    chunk_item = SimpleNamespace(id=ANSWER_CHUNK_0, document_id=DOC_ANSWER)
    dict_item = {"id": COMPLAINT_CHUNK_0, "document_id": DOC_COMPLAINT}

    chunk_ids, document_ids = _touched_ids([summary_item, chunk_item, dict_item, "noise", None])

    assert chunk_ids == {COMPLAINT_CHUNK_1, ANSWER_CHUNK_0, COMPLAINT_CHUNK_0}
    assert document_ids == {DOC_COMPLAINT, DOC_ANSWER}
    assert _touched_ids(None) == (set(), set())


@pytest.mark.asyncio
async def test_touched_scope_keeps_incoming_references_to_the_ingested_document():
    graph = _base_graph()
    touched = [
        SimpleNamespace(
            made_from=SimpleNamespace(
                id=COMPLAINT_CHUNK_1, is_part_of=SimpleNamespace(id=DOC_COMPLAINT)
            )
        )
    ]

    _, summary, _ = await _run(graph, scope="touched", data=touched)

    # The Answer's denial points at the freshly ingested Complaint -> resolved.
    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {
        A_1,
        A_2,
        COMPLAINT_CHUNK_1,
    }
    # References that touch neither the ingested chunks nor the ingested document are left alone.
    assert graph.edges_of(A_STIPULATION, "responds_to") == []
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert summary["resolved"] == 1
    assert summary["scanned"] == 1


# --------------------------------------------------------------------------- #
# prose lookup (opt-in)
# --------------------------------------------------------------------------- #
DOC_APPRAISAL = _nid("doc-appraisal")
APPRAISAL_CHUNKS = [_nid(f"appraisal-chunk-{index}") for index in range(3)]
A_PROSE = _nid("assertion-prose")
APPRAISAL_LOCATION = "file://appraisal.txt"
APPRAISAL_CHUNK_TEXTS = [
    "Comparable sales on Main Street were reviewed.\n",
    "The culvert easement crossing the northern boundary reduces the culvert easement value.\n",
    "The roof and siding were replaced in 2019.\n",
]


def _prose_graph():
    nodes = [
        _document(DOC_APPRAISAL, "Whitfield_Rebuttal_Appraisal", APPRAISAL_LOCATION),
        _document(DOC_ANSWER, "Answer_Clifton", ANSWER_LOCATION),
        _assertion(
            A_PROSE,
            "The easement reduces the value",
            source_chunk_id=ANSWER_CHUNK_0,
            attributed_to="Whitfield rebuttal appraisal discussion of the culvert easement",
        ),
    ]
    edges = []
    for index, text in enumerate(APPRAISAL_CHUNK_TEXTS):
        node, edge = _chunk(APPRAISAL_CHUNKS[index], text, index, DOC_APPRAISAL)
        nodes.append(node)
        edges.append(edge)
    node, edge = _chunk(ANSWER_CHUNK_0, ANSWER_CHUNK_0_TEXT, 0, DOC_ANSWER)
    nodes.append(node)
    edges.append(edge)
    return FakeGraph(nodes, edges)


@pytest.mark.asyncio
async def test_prose_lookup_is_off_by_default():
    graph = _prose_graph()
    texts = {APPRAISAL_LOCATION: "".join(APPRAISAL_CHUNK_TEXTS), ANSWER_LOCATION: ANSWER_TEXT}

    _, summary, _ = await _run(graph, texts)

    edges = graph.edges_of(A_PROSE, "attributed_to")
    assert [edge[1] for edge in edges] == [DOC_APPRAISAL]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_DOCUMENT_ONLY
    assert summary["resolved_by_strategy"] == {STRATEGY_DOCUMENT_ONLY: 1}


@pytest.mark.asyncio
async def test_prose_lookup_anchors_on_the_best_scoring_chunk():
    graph = _prose_graph()
    texts = {APPRAISAL_LOCATION: "".join(APPRAISAL_CHUNK_TEXTS), ANSWER_LOCATION: ANSWER_TEXT}

    _, summary, _ = await _run(graph, texts, enable_prose_lookup=True)

    edges = graph.edges_of(A_PROSE, "attributed_to")
    assert [edge[1] for edge in edges] == [APPRAISAL_CHUNKS[1]]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_PROSE_LOOKUP
    assert _props(edges[0])["resolution_confidence"] == pytest.approx(0.60)
    patch_values = dict(graph.update_node_calls)[A_PROSE]
    assert patch_values["attributed_to"] == APPRAISAL_CHUNKS[1]
    assert summary["resolved_by_strategy"] == {STRATEGY_PROSE_LOOKUP: 1}


@pytest.mark.asyncio
async def test_prose_lookup_never_costs_a_resolution_the_floor_would_accept():
    """Prose lookup is a fixed 0.60, so a higher floor must fall back, not give up."""
    graph = _prose_graph()
    texts = {APPRAISAL_LOCATION: "".join(APPRAISAL_CHUNK_TEXTS), ANSWER_LOCATION: ANSWER_TEXT}

    _, summary, _ = await _run(graph, texts, enable_prose_lookup=True, confidence_floor=0.70)

    edges = graph.edges_of(A_PROSE, "attributed_to")
    assert [edge[1] for edge in edges] == [DOC_APPRAISAL]
    assert _props(edges[0])["resolution_strategy"] == STRATEGY_DOCUMENT_ONLY
    assert _props(edges[0])["resolution_confidence"] == pytest.approx(0.72)
    assert summary["unresolved"] == 0


@pytest.mark.asyncio
async def test_prose_lookup_without_a_clear_winner_falls_back_to_the_document():
    graph = _prose_graph()
    # Every chunk now carries the query's distinctive tokens: no clear winner.
    for chunk_id in APPRAISAL_CHUNKS:
        graph.nodes[chunk_id]["text"] = "The culvert easement is discussed at length.\n"
    texts = {
        APPRAISAL_LOCATION: "".join(graph.nodes[chunk_id]["text"] for chunk_id in APPRAISAL_CHUNKS),
        ANSWER_LOCATION: ANSWER_TEXT,
    }

    _, summary, _ = await _run(graph, texts, enable_prose_lookup=True)

    edges = graph.edges_of(A_PROSE, "attributed_to")
    assert [edge[1] for edge in edges] == [DOC_APPRAISAL]
    assert summary["resolved_by_strategy"] == {STRATEGY_DOCUMENT_ONLY: 1}


# --------------------------------------------------------------------------- #
# confidence floor
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_confidence_floor_rejects_weaker_resolutions():
    graph = _base_graph()

    _, summary, _ = await _run(graph, confidence_floor=0.95)

    # Only the 1.0 entity-name resolution clears a 0.95 floor.
    assert summary["resolved_by_strategy"] == {STRATEGY_ENTITY_NAME: 1}
    assert summary["unresolved"] == 2


# --------------------------------------------------------------------------- #
# the cognify-tail task
# --------------------------------------------------------------------------- #
def test_task_accepts_pipeline_context():
    task = Task(resolve_assertion_references, scope="touched")
    assert task.accepts_ctx is True
    assert getattr(resolve_assertion_references, "__task_summary__", None)


@pytest.mark.asyncio
async def test_task_returns_its_input_and_writes_edges():
    graph = _base_graph()
    items = [SimpleNamespace(made_from=SimpleNamespace(id=COMPLAINT_CHUNK_1))]

    with _patched(graph):
        result = await resolve_assertion_references(items)

    assert result is items
    assert graph.add_edges_calls


@pytest.mark.asyncio
async def test_task_swallows_its_own_errors(caplog):
    items = ["unchanged"]
    with patch(f"{MODULE}._load_graph_view", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with patch(f"{MODULE}.get_graph_engine", new=AsyncMock()):
            with caplog.at_level("WARNING"):
                result = await resolve_assertion_references(items)

    assert result is items
    assert any("boom" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_all_scope_runs_at_most_once_per_pipeline_run():
    graph = _base_graph()
    ctx = PipelineContext(dataset=SimpleNamespace(id=DATASET_ID), pipeline_run_id="run-1")

    with _patched(graph):
        await resolve_assertion_references("batch-1", ctx=ctx)
        await resolve_assertion_references("batch-2", ctx=ctx)

    assert ctx.extras["reference_resolution_ran"] is True
    assert len(graph.filtered_calls) == 1


@pytest.mark.asyncio
async def test_a_failed_pass_does_not_memoize_itself():
    """A first batch that blew up must not suppress the rest of the run."""
    graph = _base_graph()
    ctx = PipelineContext(dataset=SimpleNamespace(id=DATASET_ID), pipeline_run_id="run-1")

    with patch(f"{MODULE}._load_graph_view", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with patch(f"{MODULE}.get_graph_engine", new=AsyncMock()):
            await resolve_assertion_references("batch-1", ctx=ctx)

    assert "reference_resolution_ran" not in ctx.extras

    with _patched(graph):
        await resolve_assertion_references("batch-2", ctx=ctx)

    assert ctx.extras["reference_resolution_ran"] is True
    assert graph.add_edges_calls


@pytest.mark.asyncio
async def test_touched_scope_is_never_memoized():
    graph = _base_graph()
    ctx = PipelineContext(dataset=SimpleNamespace(id=DATASET_ID), pipeline_run_id="run-1")
    items = [SimpleNamespace(made_from=SimpleNamespace(id=COMPLAINT_CHUNK_1))]

    with _patched(graph):
        await resolve_assertion_references(items, scope="touched", ctx=ctx)
        await resolve_assertion_references(items, scope="touched", ctx=ctx)

    assert "reference_resolution_ran" not in ctx.extras
    assert len(graph.filtered_calls) == 2


# --------------------------------------------------------------------------- #
# memify payload handling
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_apply_tolerates_a_list_wrapped_payload():
    graph = _base_graph()
    with _patched(graph):
        payload = await detect_dangling_references(None)
        summary = await apply_reference_resolutions([payload])

    assert summary["edges_written"] == 5


@pytest.mark.asyncio
async def test_apply_without_a_plan_writes_nothing():
    graph = _base_graph()
    with _patched(graph):
        summary = await apply_reference_resolutions({"plan": [], "summary": {}})

    assert summary["edges_written"] == 0
    assert graph.add_edges_calls == []


@pytest.mark.asyncio
async def test_apply_lets_write_failures_surface():
    graph = _base_graph()

    async def _boom(_edges, **_kwargs):
        raise RuntimeError("write failed")

    graph.add_edges = _boom
    with _patched(graph):
        payload = await detect_dangling_references(None)
        with pytest.raises(RuntimeError, match="write failed"):
            await apply_reference_resolutions(payload)
