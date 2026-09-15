"""Unit tests for the agentic assertion-reference resolver pass.

Everything here runs against a stateful ``FakeGraph``: no graph backend, no vector
backend, no LLM, no network, no filesystem. The fake applies what the task writes
(``add_edges`` upserts on the triple, ``update_node`` merges into the node blob), so a
second pass reads back the first pass's writes and idempotency is a real assertion rather
than a mock's call count.

Two seams carry everything the pass cannot do for itself:

* ``cognee.tasks.graph.reference_retrieval.get_vector_engine_async`` -> a
  ``FakeVectorEngine`` with scripted ``ScoredResult``s, so the seed shortlist is exactly
  what a test says it is.
* ``cognee.tasks.graph.reference_tracer.LLMGateway.acreate_structured_output`` -> a
  scripted list of ``TracerStep``s, so a whole trace (tool call, tool call, finish) is
  written out in the test that depends on it.

A scripted step may be a callable taking the rendered user prompt. That is how a test
names a node: labels (``A1``, ``P2``, ``D1``) are issued per trace by the registry, so a
test says "finish on the passage whose line reads ..." and the fake reads the label the
registry actually issued out of the prompt.
"""

import json
from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import NAMESPACE_URL, uuid5

import pytest

from cognee.modules.graph.utils.reference_resolution import (
    STRATEGY_ENTITY_NAME,
    STRATEGY_EXISTING_ID,
    STRATEGY_LLM_INFERRED,
    STRATEGY_LLM_TRACE,
    ReferenceHint,
    reference_fingerprint,
)
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.graph.reference_tracer import TracerFinish, TracerStep, TracerToolCall
from cognee.tasks.graph.reference_pass import (
    INFERRED_EDGE_FEEDBACK_WEIGHT,
    NOTE_FORCE_KEPT_PRIOR,
    NOTE_LLM_ESTIMATE_ONLY,
    NOTE_LLM_MALFORMED_STEP,
    NOTE_LLM_SELF_REFERENCE,
    NOTE_LLM_UNKNOWN_LABEL,
    NOTE_UNSTATED,
    UNSTATED_BASIS,
)
from cognee.tasks.graph.resolve_assertion_references import (
    DOCUMENT_NODE_TYPES,
    NOTE_STALE_ID,
    NOTE_LLM_ABSTAINED,
    NOTE_LLM_BELOW_THRESHOLD,
    NOTE_LLM_BUDGET_EXHAUSTED,
    NOTE_LLM_CIRCUIT_BROKEN,
    NOTE_LLM_ITERATION_CAP,
    REFERENCE_FIELDS,
    REFERENCE_RESOLUTION_DATA_ID,
    _touched_ids,
    apply_reference_resolutions,
    detect_dangling_references,
    resolve_assertion_references,
)
from cognee.tests.unit.tasks.graph._reference_fakes import FakeVectorEngine, scored

MODULE = "cognee.tasks.graph.resolve_assertion_references"
# The trace pass lives in its own module (the task module imports its names back and
# re-exports them), so a seam inside the pass has to be patched where the pass reads it.
PASS = "cognee.tasks.graph.reference_pass"
# ``cognee/tasks/graph/__init__.py`` re-exports the task function under its own module's
# name, so the package attribute ``resolve_assertion_references`` is the *function*. The
# module object therefore has to come from the import machinery, and every seam inside it
# is patched with ``patch.object(resolve_module, ...)``: on Python 3.10 (which CI runs)
# ``patch("cognee.tasks.graph.resolve_assertion_references.get_graph_engine")`` resolves
# the dotted target by attribute lookup, lands on the function and raises AttributeError.
resolve_module = import_module(MODULE)
pass_module = import_module(PASS)
RETRIEVAL = "cognee.tasks.graph.reference_retrieval"
TRACER = "cognee.tasks.graph.reference_tracer"
GATEWAY = f"{TRACER}.LLMGateway.acreate_structured_output"


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
STIPULATION_CHUNK_0 = _nid("stipulation-chunk-0")

A_1 = _nid("assertion-owns")
A_2 = _nid("assertion-acquired")
A_3 = _nid("assertion-roof")
A_DENIAL = _nid("assertion-denial")
A_ATTRIBUTED = _nid("assertion-attributed")
A_RESOLVED = _nid("assertion-already-resolved")
A_STIPULATION = _nid("assertion-stipulation")
A_BOUNDARY = _nid("assertion-boundary")
ENTITY_FESTER = _nid("entity-norman-fester")

COMPLAINT_NAME = "Verified_Complaint_Adams_v_Clifton"
ANSWER_NAME = "Answer_Clifton"
STIPULATION_NAME = "Stipulation_Adams"

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
STIPULATION_CHUNK_0_TEXT = "STIPULATION\n\nThe parties stipulated to the boundary.\n"
STIPULATION_TEXT = STIPULATION_CHUNK_0_TEXT

CHUNK_1_LABEL = "Clifton owns 10 Main Street, acquired in 1998. 6. Roof replaced in 2019."

DEFAULT_TEXTS = {
    COMPLAINT_LOCATION: COMPLAINT_TEXT,
    ANSWER_LOCATION: ANSWER_TEXT,
    STIPULATION_LOCATION: STIPULATION_TEXT,
}

PROVENANCE_KWARGS = {"source_ref_key": "dataset:data", "pipeline_run_id": "run-1"}

# The structured references extraction writes (decision D6): a plain dict, never parsed
# with a regex. ``basis="positional"`` is what keeps the denial off the entity-name step.
DENIAL_REF = {
    "document_hint": "the Complaint",
    "locator_kind": "paragraph",
    "locator_value": "5",
    "basis": "positional",
}
ATTRIBUTED_REF = {"document_hint": "norman fester", "basis": "cited"}
STIPULATION_REF = {"document_hint": "the Adams stipulation", "basis": "described"}

DENIAL_REFERENCE_TEXT = "the Complaint paragraph 5"
STIPULATION_REFERENCE_TEXT = "the Adams stipulation"

DENIAL_FINGERPRINT = reference_fingerprint(
    ReferenceHint(
        document_hint="the Complaint",
        locator_kind="paragraph",
        locator_value="5",
        basis="positional",
    ),
    "responds_to",
)

# Markers a scripted step names a node by. Each one appears on exactly one labelled line
# of the rendered prompt, so the fake can read back the label the registry issued.
MARK_COMPLAINT_DOCUMENT = f'Document "{COMPLAINT_NAME}"'
MARK_STIPULATION_DOCUMENT = f'Document "{STIPULATION_NAME}"'
MARK_COMPLAINT_PASSAGE_1 = f'Passage in "{COMPLAINT_NAME}" (chunk 1)'
MARK_STIPULATION_PASSAGE = f'Passage in "{STIPULATION_NAME}" (chunk 0)'
MARK_ALLEGATION = '"Clifton owns 10 Main Street"'
MARK_ANSWER_DOCUMENT = f'Document "{ANSWER_NAME}"'
MARK_ANSWER_PASSAGE_0 = '(chunk 0): "ANSWER'


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
        _document(DOC_COMPLAINT, COMPLAINT_NAME, COMPLAINT_LOCATION),
        _document(DOC_ANSWER, ANSWER_NAME, ANSWER_LOCATION),
        _document(DOC_STIPULATION, STIPULATION_NAME, STIPULATION_LOCATION),
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
            A_BOUNDARY,
            "The parties stipulated to the boundary",
            source_chunk_id=STIPULATION_CHUNK_0,
            source_quote="The parties stipulated to the boundary",
        ),
        _assertion(
            A_DENIAL,
            "Clifton owns 10 Main Street",
            statement_type="denial",
            polarity="negative",
            source_chunk_id=ANSWER_CHUNK_0,
            source_quote="Defendant denies the allegations of paragraph 5.",
            responds_to_ref=dict(DENIAL_REF),
        ),
        _assertion(
            A_ATTRIBUTED,
            "The valuation is unsupported",
            source_chunk_id=ANSWER_CHUNK_0,
            attributed_to_ref=dict(ATTRIBUTED_REF),
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
            "The boundary was agreed",
            source_chunk_id=ANSWER_CHUNK_0,
            responds_to_ref=dict(STIPULATION_REF),
        ),
        (ENTITY_FESTER, {"id": ENTITY_FESTER, "type": "Entity", "name": "Norman Fester"}),
    ]
    edges = [(A_RESOLVED, A_1, "responds_to", {})]
    for node, edge in (
        _chunk(COMPLAINT_CHUNK_0, COMPLAINT_CHUNK_0_TEXT, 0, DOC_COMPLAINT),
        _chunk(COMPLAINT_CHUNK_1, COMPLAINT_CHUNK_1_TEXT, 1, DOC_COMPLAINT),
        _chunk(ANSWER_CHUNK_0, ANSWER_CHUNK_0_TEXT, 0, DOC_ANSWER),
        _chunk(STIPULATION_CHUNK_0, STIPULATION_CHUNK_0_TEXT, 0, DOC_STIPULATION),
    ):
        nodes.append(node)
        edges.append(edge)
    return FakeGraph(nodes, edges)


def _vector_results():
    """The scripted seed: two allegations, the complaint's chunks, three documents."""
    return {
        "Assertion_name": [
            scored(A_1, 0.10, "Clifton owns 10 Main Street", source_chunk_id=COMPLAINT_CHUNK_1),
            scored(
                A_2, 0.22, "The property was acquired in 1998", source_chunk_id=COMPLAINT_CHUNK_1
            ),
            scored(
                A_BOUNDARY,
                0.30,
                "The parties stipulated to the boundary",
                source_chunk_id=STIPULATION_CHUNK_0,
            ),
        ],
        "DocumentChunk_text": [
            scored(
                COMPLAINT_CHUNK_1,
                0.18,
                COMPLAINT_CHUNK_1_TEXT,
                document_id=DOC_COMPLAINT,
                document_name=COMPLAINT_NAME,
                chunk_index=1,
            ),
            scored(
                STIPULATION_CHUNK_0,
                0.34,
                STIPULATION_CHUNK_0_TEXT,
                document_id=DOC_STIPULATION,
                document_name=STIPULATION_NAME,
                chunk_index=0,
            ),
        ],
        "TextDocument_name": [
            scored(DOC_COMPLAINT, 0.12, COMPLAINT_NAME),
            scored(DOC_STIPULATION, 0.26, STIPULATION_NAME),
            scored(DOC_ANSWER, 0.40, ANSWER_NAME),
        ],
    }


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


# --------------------------------------------------------------------------- #
# scripting a trace
# --------------------------------------------------------------------------- #
def _label_for(prompt: str, marker: str) -> str:
    """The label of the one rendered line carrying ``marker``."""
    for line in prompt.splitlines():
        if line.startswith("[") and "]" in line and marker in line:
            return line[1 : line.index("]")]
    raise AssertionError(f"no labelled line matching {marker!r} in:\n{prompt}")


def finish_on(marker, confidence=0.9, reason="the answer restates the allegation"):
    """Finish on whichever label the registry gave the node ``marker`` names."""

    def _step(prompt):
        return TracerStep(
            finish=TracerFinish(
                candidate_label=_label_for(prompt, marker), confidence=confidence, reason=reason
            )
        )

    return _step


def abstain(reason="nothing in this set is the referent"):
    return TracerStep(finish=TracerFinish(candidate_label=None, confidence=0.0, reason=reason))


def call_tool(name, **arguments):
    return TracerStep(tool_call=TracerToolCall(tool_name=name, arguments=arguments))


def call_tool_on(name, argument, marker, **arguments):
    """A tool call naming a node by whichever label the registry gave the ``marker`` line."""

    def _step(prompt):
        return TracerStep(
            tool_call=TracerToolCall(
                tool_name=name,
                arguments={argument: _label_for(prompt, marker), **arguments},
            )
        )

    return _step


def locate(document_marker, kind="paragraph", value="5"):
    """A ``locate_paragraph`` call naming the document by its rendered line."""

    def _step(prompt):
        return TracerStep(
            tool_call=TracerToolCall(
                tool_name="locate_paragraph",
                arguments={
                    "document": _label_for(prompt, document_marker),
                    "kind": kind,
                    "value": value,
                },
            )
        )

    return _step


DENIAL_TRACE = [locate(MARK_COMPLAINT_DOCUMENT), finish_on(MARK_COMPLAINT_PASSAGE_1)]


class FakeTracerLLM:
    """A scripted ``acreate_structured_output``: no LLM, no network."""

    def __init__(self, steps=(), default=None):
        self.steps = list(steps)
        self.default = default if default is not None else abstain()
        self.prompts = []
        self.await_count = 0

    async def __call__(self, *, text_input, system_prompt, response_model):
        self.prompts.append(text_input)
        self.await_count += 1
        step = self.steps.pop(0) if self.steps else self.default
        if isinstance(step, BaseException):
            raise step
        return step(text_input) if callable(step) else step


@contextmanager
def _patched(graph, texts=None, *, locations=None, steps=(), default=None, vector_results=None):
    """Patch every I/O seam the pass reaches through."""
    texts = DEFAULT_TEXTS if texts is None else texts

    async def _read(location):
        value = texts[location]
        if isinstance(value, Exception):
            raise value
        return value

    engine = FakeVectorEngine(_vector_results() if vector_results is None else dict(vector_results))
    llm = FakeTracerLLM(steps, default=default)

    with (
        patch.object(resolve_module, "get_graph_engine", new=AsyncMock(return_value=graph)),
        patch.object(resolve_module, "index_graph_edges", new=AsyncMock()) as index_mock,
        patch.object(
            resolve_module,
            "graph_provenance_write_kwargs",
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
        patch(
            f"{RETRIEVAL}.get_vector_engine_async", new=AsyncMock(return_value=engine)
        ) as engine_mock,
        patch(GATEWAY, new=llm),
    ):
        yield SimpleNamespace(
            index_graph_edges=index_mock,
            graph_provenance_write_kwargs=provenance_mock,
            read_processed_text=read_mock,
            vector_engine=engine,
            vector_engine_factory=engine_mock,
            llm=llm,
        )


async def _run(
    graph,
    texts=None,
    *,
    locations=None,
    ctx=None,
    dry_run=False,
    steps=(),
    default=None,
    vector_results=None,
    **detect_kwargs,
):
    """Run the two-phase pass and return ``(payload, summary, mocks)``."""
    data = detect_kwargs.pop("data", None)
    with _patched(
        graph,
        texts,
        locations=locations,
        steps=steps,
        default=default,
        vector_results=vector_results,
    ) as mocks:
        payload = await detect_dangling_references(data, ctx=ctx, **detect_kwargs)
        summary = await apply_reference_resolutions(payload, dry_run=dry_run, ctx=ctx)
    return payload, summary, mocks


def _props(edge):
    return edge[3]


def _by_target(edges):
    return {edge[1]: edge for edge in edges}


def _resolution_blob(graph, assertion_id, field_name="responds_to"):
    return dict(graph.update_node_calls)[assertion_id][f"{field_name}_resolution"]


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def test_reference_fields_never_include_the_identity_field():
    assert REFERENCE_FIELDS == ("responds_to", "attributed_to")
    assert "asserted_by" not in REFERENCE_FIELDS
    assert not any(field.endswith("_ref") for field in REFERENCE_FIELDS)
    assert REFERENCE_RESOLUTION_DATA_ID == uuid5(NAMESPACE_URL, "cognee:reference-resolution")
    assert DOCUMENT_NODE_TYPES == (
        "TextDocument",
        "PdfDocument",
        "UnstructuredDocument",
        "AudioDocument",
        "ImageDocument",
    )


# --------------------------------------------------------------------------- #
# the worked example: Answer denial -> locate_paragraph -> the anchor passage
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_traced_passage_links_the_quoted_allegations_and_the_anchor_chunk():
    graph = _base_graph()
    _, summary, mocks = await _run(graph, steps=list(DENIAL_TRACE))

    denial_edges = _by_target(graph.edges_of(A_DENIAL, "responds_to"))
    assert set(denial_edges) == {A_1, A_2, COMPLAINT_CHUNK_1}
    # ¶6's allegation is outside the located span and must not be linked.
    assert A_3 not in denial_edges

    allegation_props = _props(denial_edges[A_1])
    assert allegation_props["relationship_name"] == "responds_to"
    assert allegation_props["source_node_id"] == A_DENIAL
    assert allegation_props["target_node_id"] == A_1
    assert allegation_props["reference_text"] == DENIAL_REFERENCE_TEXT
    assert allegation_props["resolution_strategy"] == STRATEGY_LLM_TRACE
    assert allegation_props["resolution_confidence"] == pytest.approx(0.9)
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
    assert patch_values["responds_to_text"] == DENIAL_REFERENCE_TEXT
    resolution = patch_values["responds_to_resolution"]
    assert resolution["strategy"] == STRATEGY_LLM_TRACE
    assert resolution["confidence"] == pytest.approx(0.9)
    assert resolution["target_ids"] == [A_1, A_2]
    assert resolution["anchor_id"] == COMPLAINT_CHUNK_1
    assert resolution["document_id"] == DOC_COMPLAINT
    assert resolution["notes"] == []
    assert resolution["fingerprint"] == DENIAL_FINGERPRINT
    assert resolution["reason"] == "the answer restates the allegation"

    assert summary["scanned"] == 4
    assert summary["resolved"] == 2
    assert summary["already_resolved"] == 1
    assert summary["resolved_by_strategy"][STRATEGY_LLM_TRACE] == 1
    assert summary["anchor_types"]["DocumentChunk"] == 1
    assert summary["failed"] == 0
    assert summary["dry_run"] is False
    assert summary["notes"] == []

    # Two references reached the tracer; the denial spent two calls, the stipulation one.
    assert summary["traces_started"] == 2
    assert summary["llm_calls"] == 3
    assert mocks.llm.await_count == 3
    assert summary["tool_calls_by_name"] == {"locate_paragraph": 1}

    # Edges are written in one batch, carrying exactly the provenance kwargs.
    assert len(graph.add_edges_calls) == 1
    written, kwargs = graph.add_edges_calls[0]
    assert kwargs == PROVENANCE_KWARGS
    assert summary["edges_written"] == len(written) == 4
    assert (
        mocks.graph_provenance_write_kwargs.await_args.kwargs["fallback_data_id"]
        == REFERENCE_RESOLUTION_DATA_ID
    )


@pytest.mark.asyncio
async def test_a_trace_that_picks_an_assertion_links_only_that_assertion():
    graph = _base_graph()
    await _run(graph, steps=[finish_on(MARK_ALLEGATION, confidence=0.95)])

    edges = graph.edges_of(A_DENIAL, "responds_to")
    assert [edge[1] for edge in edges] == [A_1]
    assert _props(edges[0])["resolved_target_type"] == "Assertion"
    patch_values = dict(graph.update_node_calls)[A_DENIAL]
    assert patch_values["responds_to"] == A_1
    assert patch_values["responds_to_resolution"]["target_ids"] == [A_1]
    assert patch_values["responds_to_resolution"]["document_id"] == DOC_COMPLAINT


@pytest.mark.asyncio
async def test_a_trace_that_picks_a_document_anchors_on_the_document_node():
    graph = _base_graph()
    # The denial abstains; the stipulation reference picks the whole document.
    await _run(
        graph,
        steps=[abstain(), finish_on(MARK_STIPULATION_DOCUMENT, confidence=0.7)],
    )

    edges = graph.edges_of(A_STIPULATION, "responds_to")
    assert [edge[1] for edge in edges] == [DOC_STIPULATION]
    props = _props(edges[0])
    assert props["resolution_strategy"] == STRATEGY_LLM_TRACE
    assert props["resolved_target_type"] == "TextDocument"
    assert props["resolution_confidence"] == pytest.approx(0.7)

    patch_values = dict(graph.update_node_calls)[A_STIPULATION]
    assert patch_values["responds_to"] == DOC_STIPULATION
    assert patch_values["responds_to_text"] == STIPULATION_REFERENCE_TEXT
    assert patch_values["responds_to_resolution"]["document_id"] == DOC_STIPULATION


@pytest.mark.asyncio
async def test_opaque_document_names_do_not_change_the_answer():
    """Filename independence: the agent works from content, never from a file stem."""
    graph = _base_graph()
    graph.nodes[DOC_COMPLAINT]["name"] = "Document3"
    graph.nodes[DOC_STIPULATION]["name"] = "SKM_C55826082316050"
    vector_results = _vector_results()
    vector_results["TextDocument_name"] = [
        scored(DOC_COMPLAINT, 0.12, "Document3"),
        scored(DOC_STIPULATION, 0.26, "SKM_C55826082316050"),
        scored(DOC_ANSWER, 0.40, ANSWER_NAME),
    ]
    vector_results["DocumentChunk_text"] = [
        scored(
            COMPLAINT_CHUNK_1,
            0.18,
            COMPLAINT_CHUNK_1_TEXT,
            document_id=DOC_COMPLAINT,
            document_name="Document3",
            chunk_index=1,
        ),
    ]

    _, summary, _ = await _run(
        graph,
        vector_results=vector_results,
        steps=[
            locate('Document "Document3"'),
            finish_on('Passage in "Document3" (chunk 1)'),
            finish_on('Document "SKM_C55826082316050"', confidence=0.65),
        ],
    )

    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {
        A_1,
        A_2,
        COMPLAINT_CHUNK_1,
    }
    assert [edge[1] for edge in graph.edges_of(A_STIPULATION, "responds_to")] == [DOC_STIPULATION]
    assert summary["resolved_by_strategy"][STRATEGY_LLM_TRACE] == 2


# --------------------------------------------------------------------------- #
# the steps before the tracer
# --------------------------------------------------------------------------- #
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

    # The ingest contract keeps the field as it was: entity_name never patches a node.
    assert A_ATTRIBUTED not in dict(graph.update_node_calls)
    assert graph.nodes[A_ATTRIBUTED]["attributed_to_ref"] == ATTRIBUTED_REF


@pytest.mark.asyncio
async def test_a_positional_hint_never_reaches_the_entity_name_step():
    """ "the Complaint ¶5" must not be linked to a ``Complaint`` stub entity."""
    graph = _base_graph()
    complaint_entity = _nid("entity-the-complaint")
    graph.nodes[complaint_entity] = {
        "id": complaint_entity,
        "type": "Entity",
        "name": "the Complaint",
    }

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    assert complaint_entity not in _by_target(graph.edges_of(A_DENIAL, "responds_to"))
    assert summary["resolved_by_strategy"].get(STRATEGY_ENTITY_NAME) == 1  # only the attribution


@pytest.mark.asyncio
async def test_already_resolved_uuid_is_counted_and_writes_nothing():
    graph = _base_graph()
    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    assert summary["already_resolved"] == 1
    assert graph.edges_of(A_RESOLVED, "responds_to") == []


@pytest.mark.asyncio
async def test_existing_uuid_without_an_edge_emits_the_edge_only():
    graph = _base_graph()
    graph.edges = [edge for edge in graph.edges if edge[0] != A_RESOLVED]

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

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

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    assert graph.edges_of(A_RESOLVED, "responds_to") == []
    # Nothing to re-resolve from, but the dark reference is still reported.
    assert summary["stale_ids"] == 1


@pytest.mark.asyncio
async def test_stale_uuid_re_resolves_from_the_preserved_reference_without_force():
    """An amended pleading is re-chunked: the id the field holds is no longer a node."""
    graph = _base_graph()
    await _run(graph, steps=list(DENIAL_TRACE))
    assert graph.nodes[A_DENIAL]["responds_to"] == COMPLAINT_CHUNK_1

    rechunked = _nid("complaint-chunk-1-rechunked")
    graph.nodes[rechunked] = dict(graph.nodes.pop(COMPLAINT_CHUNK_1), id=rechunked)
    graph.edges = [edge for edge in graph.edges if COMPLAINT_CHUNK_1 not in (edge[0], edge[1])]
    graph.edges.append((rechunked, DOC_COMPLAINT, "is_part_of", {}))
    for assertion_id in (A_1, A_2, A_3):
        graph.nodes[assertion_id]["source_chunk_id"] = rechunked
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    vector_results = _vector_results()
    vector_results["Assertion_name"] = [
        scored(A_1, 0.10, "Clifton owns 10 Main Street", source_chunk_id=rechunked),
        scored(A_2, 0.22, "The property was acquired in 1998", source_chunk_id=rechunked),
    ]
    vector_results["DocumentChunk_text"] = [
        scored(
            rechunked,
            0.18,
            COMPLAINT_CHUNK_1_TEXT,
            document_id=DOC_COMPLAINT,
            document_name=COMPLAINT_NAME,
            chunk_index=1,
        ),
    ]

    _, summary, _ = await _run(graph, vector_results=vector_results, steps=list(DENIAL_TRACE))

    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {A_1, A_2, rechunked}
    patch_values = dict(graph.update_node_calls)[A_DENIAL]
    assert patch_values["responds_to"] == rechunked
    assert patch_values["responds_to_text"] == DENIAL_REFERENCE_TEXT
    assert patch_values["responds_to_resolution"]["notes"] == ["stale_id"]
    assert summary["stale_ids"] == 1


@pytest.mark.asyncio
async def test_a_stale_id_re_resolved_by_name_replaces_the_dead_id():
    """R27: entity_name keeps the field's wording -- except when the field holds an id
    that has gone dark, where leaving it would keep a dead UUID in the graph forever."""
    graph = _base_graph()
    dead = _nid("forgotten-entity")
    graph.nodes[A_ATTRIBUTED]["attributed_to"] = dead
    graph.nodes[A_ATTRIBUTED]["attributed_to_text"] = "Norman Fester"

    _, summary, _ = await _run(graph, default=abstain())

    assert graph.nodes[A_ATTRIBUTED]["attributed_to"] == ENTITY_FESTER
    # The wording the document used is still there; only the dead id moved.
    assert graph.nodes[A_ATTRIBUTED]["attributed_to_text"] == "Norman Fester"
    assert [edge[1] for edge in graph.edges_of(A_ATTRIBUTED, "attributed_to")] == [ENTITY_FESTER]
    assert summary["stale_ids"] == 1
    blob = _resolution_blob(graph, A_ATTRIBUTED, "attributed_to")
    assert blob["notes"] == [NOTE_STALE_ID]
    assert blob["strategy"] == STRATEGY_ENTITY_NAME


@pytest.mark.asyncio
async def test_a_stale_id_whose_entity_edge_exists_is_patched_without_a_new_edge():
    """R33: the link is already there, so only the dead id is outstanding -- the patch
    goes out and the edge is left exactly as it is (its properties survive)."""
    graph = _base_graph()
    dead = _nid("forgotten-entity")
    graph.nodes[A_ATTRIBUTED]["attributed_to"] = dead
    graph.nodes[A_ATTRIBUTED]["attributed_to_text"] = "Norman Fester"
    graph.edges.append(
        (A_ATTRIBUTED, ENTITY_FESTER, "attributed_to", {"resolved_by": "reference_resolver"})
    )

    _, summary, _ = await _run(graph, default=abstain())

    assert graph.nodes[A_ATTRIBUTED]["attributed_to"] == ENTITY_FESTER
    assert graph.nodes[A_ATTRIBUTED]["attributed_to_text"] == "Norman Fester"
    # Nothing was re-emitted: the edge in the graph is the one that was already there.
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert summary["stale_ids"] == 1
    blob = _resolution_blob(graph, A_ATTRIBUTED, "attributed_to")
    assert blob["notes"] == [NOTE_STALE_ID, "edges_exist"]


@pytest.mark.asyncio
async def test_an_entity_name_reference_whose_edge_exists_stays_already_resolved():
    """The other half of R33: with the name (not a dead id) in the field there is nothing
    left to do, so the second pass writes neither an edge nor a patch."""
    graph = _base_graph()
    await _run(graph, default=abstain())
    assert [edge[1] for edge in graph.edges_of(A_ATTRIBUTED, "attributed_to")] == [ENTITY_FESTER]
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, _ = await _run(graph, default=abstain())

    assert graph.nodes[A_ATTRIBUTED]["attributed_to_ref"] == ATTRIBUTED_REF
    assert "attributed_to" not in graph.nodes[A_ATTRIBUTED]
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert A_ATTRIBUTED not in dict(graph.update_node_calls)
    assert summary["already_resolved"] >= 1


@pytest.mark.asyncio
async def test_a_structured_only_reference_is_never_skipped():
    """Regression: the planner used to skip any assertion whose field was blank."""
    graph = _base_graph()
    for assertion_id in (A_DENIAL, A_STIPULATION, A_ATTRIBUTED):
        for field_name in REFERENCE_FIELDS:
            assert not graph.nodes[assertion_id].get(field_name)

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    assert summary["scanned"] == 4
    assert graph.edges_of(A_DENIAL, "responds_to")
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to")


@pytest.mark.asyncio
async def test_ambiguous_entity_name_resolves_nothing_and_never_traces():
    graph = _base_graph()
    twin = _nid("entity-norman-fester-twin")
    graph.nodes[twin] = {"id": twin, "type": "Entity", "name": "NORMAN FESTER"}

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert summary["ambiguous"] == 1
    # The denial and the stipulation traced; the ambiguous attribution did not.
    assert summary["traces_started"] == 2


@pytest.mark.asyncio
async def test_assertion_names_are_not_entity_name_targets():
    """Assertions subclass Entity in the model but must never be entity_name targets."""
    graph = _base_graph()
    graph.nodes[A_ATTRIBUTED]["attributed_to_ref"] = {
        "document_hint": "Clifton owns 10 Main Street",
        "basis": "cited",
    }

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE), default=abstain())

    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert STRATEGY_ENTITY_NAME not in summary["resolved_by_strategy"]


# --------------------------------------------------------------------------- #
# the ingest tail: no LLM, no document reads
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_tail_never_calls_the_llm_and_never_reads_a_document():
    graph = _base_graph()
    _, summary, mocks = await _run(graph, scope="touched", data=None, allow_llm=False)

    assert mocks.llm.await_count == 0
    mocks.read_processed_text.assert_not_awaited()
    mocks.vector_engine_factory.assert_not_awaited()
    assert mocks.vector_engine.batch_search_calls == []
    assert summary["llm_calls"] == 0
    assert summary["traces_started"] == 0


@pytest.mark.asyncio
async def test_the_tail_still_resolves_ids_and_entity_names():
    graph = _base_graph()
    _, summary, mocks = await _run(graph, allow_llm=False)

    assert [edge[1] for edge in graph.edges_of(A_ATTRIBUTED, "attributed_to")] == [ENTITY_FESTER]
    assert summary["resolved_by_strategy"] == {STRATEGY_ENTITY_NAME: 1}
    # Nothing is written for a reference the tail cannot answer: the pass retries it.
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    assert A_DENIAL not in dict(graph.update_node_calls)
    assert summary["unresolved"] == 2
    assert mocks.llm.await_count == 0


@pytest.mark.asyncio
async def test_allow_llm_defaults_to_the_scope():
    graph = _base_graph()
    _, touched_summary, touched_mocks = await _run(graph, scope="touched", data=None)
    assert touched_mocks.llm.await_count == 0
    assert touched_summary["traces_started"] == 0

    graph = _base_graph()
    _, all_summary, all_mocks = await _run(graph, steps=list(DENIAL_TRACE))
    assert all_mocks.llm.await_count > 0
    assert all_summary["traces_started"] == 2


@pytest.mark.asyncio
async def test_the_task_default_tail_wiring_is_llm_free():
    task = Task(resolve_assertion_references, scope="touched", allow_llm=False)
    assert task.default_params["kwargs"] == {"scope": "touched", "allow_llm": False}


# --------------------------------------------------------------------------- #
# abstain / threshold / cap / budget records
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_an_abstention_writes_a_resolution_only_record_and_never_nulls_the_field():
    graph = _base_graph()
    _, summary, _ = await _run(graph, steps=[abstain("no candidate is the referent")])

    assert graph.edges_of(A_DENIAL, "responds_to") == []
    patch_values = dict(graph.update_node_calls)[A_DENIAL]
    assert set(patch_values) == {"responds_to_resolution"}
    assert "responds_to" not in patch_values
    blob = patch_values["responds_to_resolution"]
    assert blob["strategy"] == STRATEGY_LLM_TRACE
    assert blob["anchor_id"] is None
    assert blob["notes"] == [NOTE_LLM_ABSTAINED]
    assert blob["reason"] == "no candidate is the referent"
    assert blob["fingerprint"] == DENIAL_FINGERPRINT
    # The reference the extraction recorded is untouched.
    assert graph.nodes[A_DENIAL]["responds_to_ref"] == DENIAL_REF
    assert graph.nodes[A_DENIAL].get("responds_to") is None
    assert summary["llm_abstained"] >= 1
    assert summary["unresolved"] >= 1


@pytest.mark.asyncio
async def test_a_finish_below_the_threshold_is_recorded_but_never_linked():
    graph = _base_graph()
    _, summary, _ = await _run(
        graph,
        steps=[finish_on(MARK_ALLEGATION, confidence=0.4, reason="it might be this one")],
        llm_confidence_threshold=0.6,
    )

    assert graph.edges_of(A_DENIAL, "responds_to") == []
    blob = _resolution_blob(graph, A_DENIAL)
    assert blob["notes"] == [NOTE_LLM_BELOW_THRESHOLD]
    assert blob["confidence"] == pytest.approx(0.4)
    assert blob["anchor_id"] is None
    assert summary["llm_below_threshold"] == 1


@pytest.mark.asyncio
async def test_the_iteration_cap_is_recorded_without_an_extra_call():
    graph = _base_graph()
    _, summary, mocks = await _run(
        graph,
        steps=[
            locate(MARK_COMPLAINT_DOCUMENT),
            locate(MARK_COMPLAINT_DOCUMENT, value="6"),
        ],
        default=call_tool("list_documents"),
        tracer_max_iter=2,
    )

    blob = _resolution_blob(graph, A_DENIAL)
    assert blob["notes"] == [NOTE_LLM_ITERATION_CAP]
    assert blob["iterations"] == 2
    assert summary["traces_iteration_capped"] == 2
    # Two references x two steps each, and never a "just answer now" third call.
    assert mocks.llm.await_count == 4


@pytest.mark.asyncio
async def test_the_cap_record_stores_the_cap_it_was_held_to():
    """R22: the record has to say which cap stopped it, or a raised cap can never retry."""
    graph = _base_graph()
    await _run(
        graph,
        steps=[call_tool("list_documents")],
        default=call_tool("list_documents"),
        tracer_max_iter=1,
    )

    blob = _resolution_blob(graph, A_DENIAL)
    assert blob["notes"] == [NOTE_LLM_ITERATION_CAP]
    assert blob["max_iter"] == 1


@pytest.mark.asyncio
async def test_a_capped_reference_is_not_retried_at_the_same_cap():
    graph = _base_graph()
    await _run(
        graph,
        steps=[call_tool("list_documents")],
        default=call_tool("list_documents"),
        tracer_max_iter=1,
    )
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(graph, tracer_max_iter=1)

    assert mocks.llm.await_count == 0
    assert summary["traces_started"] == 0


@pytest.mark.asyncio
async def test_a_capped_reference_is_retried_once_the_cap_is_raised():
    graph = _base_graph()
    await _run(
        graph,
        steps=[call_tool("list_documents")],
        default=call_tool("list_documents"),
        tracer_max_iter=1,
    )
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(
        graph,
        steps=[finish_on(MARK_ALLEGATION, confidence=0.9)],
        default=abstain(),
        tracer_max_iter=3,
    )

    assert summary["traces_started"] == 2
    assert [edge[1] for edge in graph.edges_of(A_DENIAL, "responds_to")] == [A_1]


@pytest.mark.asyncio
async def test_an_unknown_label_and_a_malformed_step_record_their_own_cause():
    """R24: both used to be persisted as llm_abstained, which hid why."""
    graph = _base_graph()
    _, summary, _ = await _run(
        graph,
        steps=[TracerStep(finish=TracerFinish(candidate_label="Z9", confidence=0.95))],
        default=TracerStep(thought="I am thinking about it"),
    )

    assert _resolution_blob(graph, A_DENIAL)["notes"] == [NOTE_LLM_UNKNOWN_LABEL]
    assert _resolution_blob(graph, A_STIPULATION)["notes"] == [NOTE_LLM_MALFORMED_STEP]
    assert summary["llm_unknown_label"] == 1
    assert summary["llm_malformed_step"] == 1
    assert summary["llm_abstained"] == 0


@pytest.mark.asyncio
async def test_the_stored_trace_records_every_tool_step():
    graph = _base_graph()
    await _run(graph, steps=list(DENIAL_TRACE))

    blob = _resolution_blob(graph, A_DENIAL)
    assert blob["iterations"] == 2
    assert len(blob["trace"]) == 1
    record = blob["trace"][0]
    assert record["tool"] == "locate_paragraph"
    assert record["ok"] is True
    assert json.loads(record["args"])["kind"] == "paragraph"
    assert len(record["args"]) <= 300
    assert len(record["result_preview"]) <= 300


# --------------------------------------------------------------------------- #
# budget, ordering, caching, circuit breaker
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_zero_budget_seeds_without_spending_anything(caplog):
    graph = _base_graph()
    with caplog.at_level("WARNING"):
        _, summary, mocks = await _run(graph, llm_max_calls=0)

    assert mocks.llm.await_count == 0
    assert summary["llm_calls"] == 0
    assert summary["llm_budget"] == 0
    # A zero budget is an estimate, not a failure: nothing was ever available to spend,
    # so the exhaustion flag and its WARNING would both be lies.
    assert summary["llm_budget_exhausted"] is False
    assert NOTE_LLM_ESTIMATE_ONLY in summary["notes"]
    assert NOTE_LLM_BUDGET_EXHAUSTED not in summary["notes"]
    assert not [record for record in caplog.records if "budget" in record.message.lower()]
    # The seed still ran, so the would-be trace count is the estimate for a real budget.
    assert summary["traces_started"] == 2
    assert mocks.vector_engine.batch_search_calls
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    assert A_DENIAL not in dict(graph.update_node_calls)


@pytest.mark.asyncio
async def test_an_exhausted_budget_leaves_the_rest_for_the_next_pass(caplog):
    graph = _base_graph()
    with caplog.at_level("WARNING"):
        _, summary, mocks = await _run(graph, steps=[abstain()], llm_max_calls=1, default=abstain())

    assert mocks.llm.await_count == 1
    assert summary["llm_budget_exhausted"] is True
    assert NOTE_LLM_BUDGET_EXHAUSTED in summary["notes"]
    # Nothing at all is written for the reference that never got its trace.
    assert A_STIPULATION not in dict(graph.update_node_calls)
    assert any("budget" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_denials_are_traced_before_allegations_when_the_budget_is_short():
    graph = _base_graph()
    # Budget of two: the denial's two-step trace consumes it, the stipulation gets none.
    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE), llm_max_calls=2)

    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {
        A_1,
        A_2,
        COMPLAINT_CHUNK_1,
    }
    assert graph.edges_of(A_STIPULATION, "responds_to") == []
    assert summary["llm_budget"] == 2
    assert summary["llm_calls"] == 2
    assert summary["llm_budget_exhausted"] is True


@pytest.mark.asyncio
async def test_an_identical_reference_with_an_identical_seed_is_answered_from_the_cache():
    graph = _base_graph()
    twin = _nid("assertion-denial-twin")
    # Two denials making the same reference. Their propositions are dropped so neither is
    # a lexical hit in the other's seed -- with different seeds they are different cache
    # keys by construction, which is the point of keying on the candidate set.
    graph.nodes[A_DENIAL].pop("name")
    graph.nodes[twin] = dict(graph.nodes[A_DENIAL], id=twin)

    _, summary, mocks = await _run(
        graph, steps=[finish_on(MARK_ALLEGATION, confidence=0.91)], default=abstain()
    )

    assert summary["llm_cached"] == 1
    # Both denials resolved, but only one of them was traced.
    assert [edge[1] for edge in graph.edges_of(A_DENIAL, "responds_to")] == [A_1]
    assert [edge[1] for edge in graph.edges_of(twin, "responds_to")] == [A_1]
    assert summary["traces_started"] == 2  # the traced denial and the stipulation
    assert mocks.llm.await_count == 2


@pytest.mark.asyncio
async def test_a_cached_answer_naming_the_asking_assertion_abstains():
    """The in-pass cache is keyed on (fingerprint, seed set), which excludes the asking
    assertion's own id (R21), so a trace that picked a twin's neighbour can hand that
    twin itself back as its answer. A statement never responds to itself."""
    graph = _base_graph()
    twin = _nid("assertion-denial-twin")
    graph.nodes[twin] = dict(graph.nodes[A_DENIAL], id=twin)
    # Same reference, same chunk, and propositions that share no token with anything, so
    # neither denial reaches the other's seed and both seeds are the same set.
    first, second = sorted([A_DENIAL, twin])
    graph.nodes[first]["name"] = "alpha widget"
    graph.nodes[second]["name"] = "beta gadget"

    # The first denial reads its own passage, is shown its twin there, and picks it.
    steps = [
        call_tool_on("open_document", "document", MARK_ANSWER_DOCUMENT),
        call_tool_on("read_chunk", "passage", MARK_ANSWER_PASSAGE_0),
        finish_on("beta gadget", confidence=0.92),
    ]

    _, summary, mocks = await _run(graph, steps=steps, default=abstain())

    # One trace answered both denials: the second one read it off the cache.
    assert summary["llm_cached"] == 1
    assert [edge[1] for edge in graph.edges_of(first, "responds_to")] == [second]
    # ... and the answer was the asking assertion itself, so nothing was linked.
    assert graph.edges_of(second, "responds_to") == []
    assert NOTE_LLM_SELF_REFERENCE in _resolution_blob(graph, second)["notes"]
    assert _resolution_blob(graph, second)["anchor_id"] is None
    assert mocks.llm.await_count == 4


@pytest.mark.asyncio
async def test_three_consecutive_failed_calls_break_the_circuit(caplog):
    graph = _base_graph()
    for index in range(4):
        graph.nodes[_nid(f"denial-{index}")] = dict(
            graph.nodes[A_DENIAL],
            id=_nid(f"denial-{index}"),
            name=f"claim {index}",
            source_quote=f"quote {index}",
        )

    with caplog.at_level("WARNING"):
        _, summary, mocks = await _run(
            graph, default=RuntimeError("the gateway is down"), llm_max_calls=50
        )

    assert mocks.llm.await_count == 3
    assert summary["llm_failed"] == 3
    # R25: the budget charged three attempts; none of them came back with an answer.
    assert summary["llm_calls_attempted"] == 3
    assert summary["llm_calls"] == 0
    assert NOTE_LLM_CIRCUIT_BROKEN in summary["notes"]
    assert any("circuit" in record.message.lower() for record in caplog.records)
    # R13's second half: a trace that never got an answer writes nothing at all, so a
    # later pass (or a healthy provider) is free to try every one of them again.
    patched = dict(graph.update_node_calls)
    for assertion_id in [A_DENIAL, A_STIPULATION] + [_nid(f"denial-{index}") for index in range(4)]:
        assert graph.edges_of(assertion_id, "responds_to") == []
        assert assertion_id not in patched


@pytest.mark.asyncio
async def test_no_documents_in_the_view_skips_the_tracer_entirely():
    graph = _base_graph()
    for document_id in (DOC_COMPLAINT, DOC_ANSWER, DOC_STIPULATION):
        del graph.nodes[document_id]

    _, summary, mocks = await _run(graph)

    assert mocks.llm.await_count == 0
    assert summary["llm_skipped_empty_graph"] == 2
    assert summary["unresolved"] == 2
    # The cheap steps still answer what they can.
    assert [edge[1] for edge in graph.edges_of(A_ATTRIBUTED, "attributed_to")] == [ENTITY_FESTER]


# --------------------------------------------------------------------------- #
# unstated denial inference (decision D2, strategy llm_inferred)
# --------------------------------------------------------------------------- #
A_UNSTATED = _nid("assertion-unstated-denial")
UNSTATED_PROPOSITION = "Clifton owns 10 Main Street"


def _add_unstated_denial(graph, node_id=A_UNSTATED, **overrides):
    """A denial that records no reference at all -- the inference pass's only candidate."""
    props = {
        "id": node_id,
        "type": "Assertion",
        "name": UNSTATED_PROPOSITION,
        "statement_type": "denial",
        "polarity": "negative",
        "source_chunk_id": ANSWER_CHUNK_0,
        "source_quote": "Defendant denies that Clifton owns the property.",
    }
    props.update(overrides)
    graph.nodes[node_id] = props
    return node_id


def _unstated_fingerprint(proposition=UNSTATED_PROPOSITION):
    return reference_fingerprint(
        ReferenceHint(document_hint=proposition, basis=UNSTATED_BASIS, legacy_text=None),
        "responds_to",
    )


# The stated references of ``_base_graph`` in budget order, so a test can spend the two
# stated traces before the unstated one and say which came first.
STATED_TRACES = [abstain(), abstain()]


@pytest.mark.asyncio
async def test_unstated_inference_is_off_unless_it_is_asked_for():
    graph = _base_graph()
    _add_unstated_denial(graph)

    _, summary, mocks = await _run(graph, steps=list(STATED_TRACES))

    assert summary["inferred_scanned"] == 0
    assert summary["inferred_resolved"] == 0
    assert summary["llm_calls_inferred"] == 0
    # Only the two stated references were ever traced.
    assert mocks.llm.await_count == 2
    assert graph.edges_of(A_UNSTATED, "responds_to") == []
    assert A_UNSTATED not in dict(graph.update_node_calls)


@pytest.mark.asyncio
async def test_an_unstated_denial_is_inferred_into_a_marked_edge():
    graph = _base_graph()
    _add_unstated_denial(graph)

    _, summary, mocks = await _run(
        graph,
        steps=STATED_TRACES + [call_tool("list_documents"), finish_on(MARK_ALLEGATION, 0.8)],
        infer_unstated=True,
    )

    assert summary["inferred_scanned"] == 1
    assert summary["inferred_resolved"] == 1
    assert summary["resolved_by_strategy"][STRATEGY_LLM_INFERRED] == 1
    assert summary["llm_calls_inferred"] == 2
    assert summary["llm_calls_stated"] == 2

    edges = graph.edges_of(A_UNSTATED, "responds_to")
    assert [edge[1] for edge in edges] == [A_1]
    properties = _props(edges[0])
    assert properties["inferred"] is True
    assert properties["feedback_weight"] == INFERRED_EDGE_FEEDBACK_WEIGHT == 0.2
    assert properties["resolution_strategy"] == STRATEGY_LLM_INFERRED

    # The graph must never claim the document stated a reference it did not: only the
    # audit blob is written back, never the field itself.
    patched = dict(graph.update_node_calls)[A_UNSTATED]
    assert set(patched) == {"responds_to_resolution"}
    blob = patched["responds_to_resolution"]
    assert blob["strategy"] == STRATEGY_LLM_INFERRED
    assert blob["fingerprint"] == _unstated_fingerprint()
    assert blob["notes"] == [NOTE_UNSTATED]
    assert blob["anchor_id"] == A_1
    assert [record["tool"] for record in blob["trace"]] == ["list_documents"]
    assert blob["iterations"] == 2
    assert mocks.llm.await_count == 4

    # A stated reference's edge is untouched by any of this: no mark, and the storage
    # default weight ``ensure_default_edge_properties`` fills in.
    stated = _props(graph.edges_of(A_ATTRIBUTED, "attributed_to")[0])
    assert "inferred" not in stated
    assert stated["feedback_weight"] == 0.5


@pytest.mark.asyncio
async def test_an_inferred_link_is_never_re_inferred_without_update_node():
    """R31: an inference never writes the field, so on a backend that cannot patch nodes
    the edge it wrote is the only record it ran -- without that guard every pass infers
    the same link again and re-upserts the edge, resetting its tuned properties."""
    graph = _base_graph()
    _add_unstated_denial(graph)
    graph.update_node_supported = False

    # Every reference in the graph is answered on the first pass, so the second pass has
    # nothing left to do but re-do it -- which is what the guards have to prevent.
    _, first, first_mocks = await _run(
        graph,
        steps=[
            finish_on(MARK_ALLEGATION, 0.9),
            finish_on(MARK_STIPULATION_PASSAGE, 0.9),
            finish_on(MARK_ALLEGATION, 0.8),
        ],
        infer_unstated=True,
    )
    assert first["inferred_resolved"] == 1
    assert [edge[1] for edge in graph.edges_of(A_UNSTATED, "responds_to")] == [A_1]
    assert first_mocks.llm.await_count == 3
    graph.add_edges_calls.clear()

    _, summary, mocks = await _run(
        graph, default=finish_on(MARK_ALLEGATION, 0.8), infer_unstated=True
    )

    assert mocks.llm.await_count == 0
    assert summary["inferred_scanned"] == 0
    assert graph.add_edges_calls == []


@pytest.mark.asyncio
async def test_touched_scope_infers_only_for_the_statements_it_touched():
    """R31(b): the inference is seeded and traced only for touched statements, the same
    rule the stated loop follows (R26)."""
    graph = _base_graph()
    _add_unstated_denial(graph)
    untouched = _add_unstated_denial(
        graph,
        node_id=_nid("assertion-unstated-elsewhere"),
        source_chunk_id=STIPULATION_CHUNK_0,
        name="The boundary was never agreed",
        source_quote="Defendant denies the boundary was agreed.",
    )
    touched = [SimpleNamespace(made_from=SimpleNamespace(id=ANSWER_CHUNK_0))]

    _, summary, mocks = await _run(
        graph,
        scope="touched",
        allow_llm=True,
        data=touched,
        default=finish_on(MARK_ALLEGATION, 0.8),
        infer_unstated=True,
    )

    assert summary["inferred_scanned"] == 1
    assert [edge[1] for edge in graph.edges_of(A_UNSTATED, "responds_to")] == [A_1]
    assert graph.edges_of(untouched, "responds_to") == []
    assert untouched not in dict(graph.update_node_calls)


@pytest.mark.asyncio
async def test_a_statement_that_states_its_own_reference_is_never_inferred_over():
    """``A_DENIAL`` carries a ``responds_to_ref``, so the stated loop owns it."""
    graph = _base_graph()

    _, summary, _ = await _run(graph, steps=list(STATED_TRACES), infer_unstated=True)

    assert summary["inferred_scanned"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"statement_type": "finding"},
        {"source_quote": ""},
        {"responds_to_ref": {"document_hint": "the Complaint"}},
        {"responds_to": "Complaint paragraph 5"},
    ],
    ids=["not-a-denial", "no-source-quote", "has-a-structured-reference", "has-reference-text"],
)
async def test_only_an_unreferenced_denial_or_admission_is_eligible(overrides):
    graph = _base_graph()
    _add_unstated_denial(graph, **overrides)

    _, summary, _ = await _run(
        graph, steps=list(STATED_TRACES), default=abstain(), infer_unstated=True
    )

    assert summary["inferred_scanned"] == 0
    assert graph.edges_of(A_UNSTATED, "responds_to") == []


@pytest.mark.asyncio
async def test_an_admission_is_eligible_too():
    graph = _base_graph()
    _add_unstated_denial(graph, statement_type="admission", polarity="positive")

    _, summary, _ = await _run(
        graph,
        steps=STATED_TRACES + [finish_on(MARK_ALLEGATION, 0.9)],
        infer_unstated=True,
    )

    assert summary["inferred_scanned"] == 1
    assert [edge[1] for edge in graph.edges_of(A_UNSTATED, "responds_to")] == [A_1]


def test_a_field_the_stated_loop_answered_is_never_inferred_over():
    """The pass owns each ``(assertion, field)`` once, whichever loop got there first."""
    props = {
        "id": A_UNSTATED,
        "type": "Assertion",
        "name": UNSTATED_PROPOSITION,
        "statement_type": "denial",
        "source_quote": "Defendant denies it.",
    }
    view = SimpleNamespace(assertions={A_UNSTATED: props}, resolver_edge_keys=set())

    eligible = pass_module._unstated_pending(
        view, handled=set(), force=False, touched=None, counters={}
    )
    assert [entry.assertion_id for entry in eligible] == [A_UNSTATED]
    assert eligible[0].unstated is True
    assert eligible[0].field_name == "responds_to"

    assert (
        pass_module._unstated_pending(
            view,
            handled={(A_UNSTATED, "responds_to")},
            force=False,
            touched=None,
            counters={},
        )
        == []
    )


@pytest.mark.asyncio
async def test_the_unstated_pass_waits_for_every_stated_reference():
    graph = _base_graph()
    _add_unstated_denial(graph)

    # Two calls in the budget and three candidates: the two stated references spend it,
    # and the inference gets nothing -- which is only true if it runs last.
    _, summary, mocks = await _run(
        graph, steps=list(STATED_TRACES), llm_max_calls=2, infer_unstated=True
    )

    assert mocks.llm.await_count == 2
    assert summary["llm_calls_stated"] == 2
    assert summary["llm_calls_inferred"] == 0
    assert summary["traces_started"] == 3
    assert summary["inferred_scanned"] == 1
    assert summary["inferred_resolved"] == 0
    # Nothing at all is written for the inference that never got its trace.
    assert A_UNSTATED not in dict(graph.update_node_calls)
    assert graph.edges_of(A_UNSTATED, "responds_to") == []


@pytest.mark.asyncio
async def test_an_inference_below_the_higher_bar_is_recorded_but_never_linked():
    graph = _base_graph()
    _add_unstated_denial(graph)

    # 0.7 clears the stated threshold (0.6) and not the inferred one (0.75).
    _, summary, _ = await _run(
        graph,
        steps=STATED_TRACES + [finish_on(MARK_ALLEGATION, 0.7, reason="same proposition")],
        infer_unstated=True,
    )

    assert summary["llm_below_threshold"] == 1
    assert summary["inferred_resolved"] == 0
    assert graph.edges_of(A_UNSTATED, "responds_to") == []
    blob = _resolution_blob(graph, A_UNSTATED)
    assert blob["strategy"] == STRATEGY_LLM_INFERRED
    assert blob["notes"] == [NOTE_UNSTATED, NOTE_LLM_BELOW_THRESHOLD]
    assert blob["anchor_id"] is None
    assert blob["reason"] == "same proposition"
    assert set(dict(graph.update_node_calls)[A_UNSTATED]) == {"responds_to_resolution"}


@pytest.mark.asyncio
async def test_a_recorded_inference_is_not_reconsidered_unless_forced():
    graph = _base_graph()
    _add_unstated_denial(graph)
    steps = STATED_TRACES + [finish_on(MARK_ALLEGATION, 0.8)]

    _, first, _ = await _run(graph, steps=list(steps), infer_unstated=True)
    assert first["inferred_resolved"] == 1

    _, second, mocks = await _run(graph, steps=list(steps), default=abstain(), infer_unstated=True)
    assert second["inferred_scanned"] == 0

    _, forced, forced_mocks = await _run(
        graph, steps=list(steps), default=abstain(), infer_unstated=True, force=True
    )
    assert forced["inferred_scanned"] == 1
    assert forced_mocks.llm.await_count > mocks.llm.await_count


@pytest.mark.asyncio
async def test_the_tail_never_infers_anything():
    graph = _base_graph()
    _add_unstated_denial(graph)
    items = [SimpleNamespace(made_from=SimpleNamespace(id=ANSWER_CHUNK_0, is_part_of=None))]

    _, summary, mocks = await _run(
        graph, data=items, scope="touched", allow_llm=False, infer_unstated=True
    )

    assert mocks.llm.await_count == 0
    assert summary["inferred_scanned"] == 0
    assert graph.edges_of(A_UNSTATED, "responds_to") == []


@pytest.mark.asyncio
async def test_an_explicit_inference_threshold_overrides_the_config_bar():
    graph = _base_graph()
    _add_unstated_denial(graph)

    _, summary, _ = await _run(
        graph,
        steps=STATED_TRACES + [finish_on(MARK_ALLEGATION, 0.5)],
        infer_unstated=True,
        infer_confidence_threshold=0.4,
    )

    assert summary["inferred_resolved"] == 1
    assert [edge[1] for edge in graph.edges_of(A_UNSTATED, "responds_to")] == [A_1]


# --------------------------------------------------------------------------- #
# the attempt guard
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_recorded_attempt_with_the_same_fingerprint_is_not_retried():
    graph = _base_graph()
    _, first, _ = await _run(graph, steps=[abstain()], default=abstain())
    assert first["llm_abstained"] >= 1
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(graph)

    assert mocks.llm.await_count == 0
    assert summary["traces_started"] == 0
    # The resolved id, the entity-name edge that already exists, and the two records.
    assert summary["already_resolved"] == 4


def _stored_attempt(note, *, fingerprint=DENIAL_FINGERPRINT, max_iter=None):
    """A ``<field>_resolution`` record as a previous pass would have left it."""
    record = {
        "strategy": STRATEGY_LLM_TRACE,
        "confidence": 0.0,
        "target_type": None,
        "target_ids": [],
        "anchor_id": None,
        "document_id": None,
        "notes": [note],
        "reason": "nothing in this set is the referent",
        "fingerprint": fingerprint,
        "iterations": 4,
        "trace": [],
    }
    if max_iter is not None:
        record["max_iter"] = max_iter
    return record


@pytest.mark.asyncio
async def test_an_attempt_stored_as_a_json_string_is_read_back():
    """Neo4j stores a dict property as a JSON string, so the guard has to read both
    shapes (the dict shape is covered by the test above)."""
    graph = _base_graph()
    graph.nodes[A_DENIAL]["responds_to_resolution"] = json.dumps(
        _stored_attempt(NOTE_LLM_ABSTAINED)
    )

    _, summary, mocks = await _run(graph, default=abstain())

    # Only the stipulation's reference was traced; the denial's was already attempted.
    assert summary["traces_started"] == 1
    assert mocks.llm.await_count == 1
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    assert A_DENIAL not in dict(graph.update_node_calls)


@pytest.mark.asyncio
async def test_a_legacy_cap_record_without_a_stored_cap_is_still_an_attempt():
    """R22b: a record written before the cap was stored says nothing about which cap
    stopped it, so it keeps counting as an attempt -- force is the way to re-open it."""
    graph = _base_graph()
    graph.nodes[A_DENIAL]["responds_to_resolution"] = _stored_attempt(NOTE_LLM_ITERATION_CAP)

    _, summary, mocks = await _run(graph, default=abstain(), tracer_max_iter=6)

    assert summary["traces_started"] == 1
    assert graph.edges_of(A_DENIAL, "responds_to") == []


@pytest.mark.asyncio
async def test_a_cap_record_below_the_current_cap_is_traced_again():
    graph = _base_graph()
    graph.nodes[A_DENIAL]["responds_to_resolution"] = _stored_attempt(
        NOTE_LLM_ITERATION_CAP, max_iter=4
    )

    _, summary, _ = await _run(
        graph, steps=[finish_on(MARK_ALLEGATION)], default=abstain(), tracer_max_iter=6
    )

    assert summary["traces_started"] == 2
    assert [edge[1] for edge in graph.edges_of(A_DENIAL, "responds_to")] == [A_1]


@pytest.mark.asyncio
async def test_a_cap_record_at_or_above_the_current_cap_is_not_retried():
    graph = _base_graph()
    graph.nodes[A_DENIAL]["responds_to_resolution"] = _stored_attempt(
        NOTE_LLM_ITERATION_CAP, max_iter=6
    )

    _, summary, _ = await _run(graph, default=abstain(), tracer_max_iter=4)

    assert summary["traces_started"] == 1
    assert graph.edges_of(A_DENIAL, "responds_to") == []


@pytest.mark.asyncio
async def test_force_bypasses_the_attempt_guard():
    graph = _base_graph()
    await _run(graph, steps=[abstain()], default=abstain())
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(graph, steps=list(DENIAL_TRACE), force=True)

    # The denial's two steps, then one abstention each for the stipulation and for the
    # already-resolved reference force re-opened.
    assert mocks.llm.await_count == 4
    assert set(_by_target(graph.edges_of(A_DENIAL, "responds_to"))) == {
        A_1,
        A_2,
        COMPLAINT_CHUNK_1,
    }
    assert summary["resolved_by_strategy"][STRATEGY_LLM_TRACE] == 1
    # The re-opened reference abstained, so its live answer stands: no patch was written
    # for it, and the field still holds the id it already had.
    assert A_RESOLVED not in dict(graph.update_node_calls)
    assert graph.nodes[A_RESOLVED]["responds_to"] == A_1
    assert "responds_to_resolution" not in graph.nodes[A_RESOLVED]
    assert NOTE_FORCE_KEPT_PRIOR in summary["notes"]


@pytest.mark.asyncio
async def test_a_forced_recheck_that_abstains_keeps_the_prior_answer_and_blob():
    """force re-opens a live answer, and the re-check is allowed to come back empty --
    which must not turn a positive audit blob into an abstention over an answer the
    field still holds."""
    graph = _base_graph()
    await _run(graph, steps=list(DENIAL_TRACE), default=abstain())
    field_before = graph.nodes[A_DENIAL]["responds_to"]
    blob_before = json.dumps(graph.nodes[A_DENIAL]["responds_to_resolution"], sort_keys=True)
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(graph, force=True, default=abstain())

    assert graph.nodes[A_DENIAL]["responds_to"] == field_before
    assert (
        json.dumps(graph.nodes[A_DENIAL]["responds_to_resolution"], sort_keys=True) == blob_before
    )
    assert A_DENIAL not in dict(graph.update_node_calls)
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    # The re-check was traced and paid for; it just changed nothing.
    assert mocks.llm.await_count >= 1
    assert summary["notes"].count(NOTE_FORCE_KEPT_PRIOR) == 1


@pytest.mark.asyncio
async def test_a_changed_reference_is_reconsidered_without_force():
    graph = _base_graph()
    await _run(graph, steps=[abstain()], default=abstain())
    graph.nodes[A_DENIAL]["responds_to_ref"] = dict(DENIAL_REF, locator_value="6")
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(graph, steps=[finish_on(MARK_ALLEGATION)], default=abstain())

    assert mocks.llm.await_count >= 1
    assert [edge[1] for edge in graph.edges_of(A_DENIAL, "responds_to")] == [A_1]


# --------------------------------------------------------------------------- #
# idempotency, force, dry_run
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_second_pass_writes_nothing():
    graph = _base_graph()
    await _run(graph, steps=list(DENIAL_TRACE), default=finish_on(MARK_STIPULATION_PASSAGE))
    graph.add_edges_calls.clear()
    graph.update_node_calls.clear()

    _, summary, mocks = await _run(graph)

    assert graph.add_edges_calls == []
    assert graph.update_node_calls == []
    assert summary["edges_written"] == 0
    assert summary["nodes_patched"] == 0
    assert summary["resolved"] == 0
    assert summary["already_resolved"] == 4
    assert mocks.llm.await_count == 0


@pytest.mark.asyncio
async def test_dry_run_plans_without_writing():
    graph = _base_graph()
    payload, summary, mocks = await _run(graph, steps=list(DENIAL_TRACE), dry_run=True)

    assert graph.add_edges_calls == []
    assert graph.update_node_calls == []
    assert summary["dry_run"] is True
    assert summary["edges_written"] == 0
    assert summary["nodes_patched"] == 0
    assert summary["resolved"] == 2
    # Traces still ran, so the plan shows what the agent would have linked.
    assert mocks.llm.await_count == 3
    assert any(resolution.strategy == STRATEGY_LLM_TRACE for resolution in payload["plan"])
    mocks.index_graph_edges.assert_not_awaited()


# --------------------------------------------------------------------------- #
# write-phase behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_node_patches_are_skipped_when_the_adapter_cannot_patch():
    graph = _base_graph()
    graph.update_node_supported = False

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    assert summary["edges_written"] == 4
    assert summary["nodes_patched"] == 0
    assert summary["notes"] == ["node_patch_unsupported"]


@pytest.mark.asyncio
async def test_a_forced_pass_rewrites_nothing_when_the_adapter_cannot_patch():
    """force re-opens even an answered reference, and without update_node the field keeps
    its reference text, so the edge pre-check is the only thing that can stop the same
    edges being re-emitted (which would reset properties improve() had tuned)."""
    graph = _base_graph()
    graph.update_node_supported = False

    _, first, _ = await _run(graph, steps=list(DENIAL_TRACE))
    assert first["edges_written"] == 4
    graph.add_edges_calls.clear()

    _, summary, mocks = await _run(graph, steps=list(DENIAL_TRACE), default=abstain(), force=True)

    # The denial was re-traced (two steps) and landed on the same answer; the stipulation
    # and the already-resolved reference force re-opened abstained on one call each.
    assert mocks.llm.await_count == 4
    assert graph.add_edges_calls == []
    assert summary["edges_written"] == 0
    assert summary["nodes_patched"] == 0
    assert summary["resolved"] == 0
    mocks.index_graph_edges.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_answered_reference_is_never_re_traced_without_update_node():
    """R23: on an adapter that cannot patch nodes, nothing remembers the fingerprint, so
    without this guard every pass re-spends the whole budget on references it has already
    answered."""
    graph = _base_graph()
    graph.update_node_supported = False

    _, first, first_mocks = await _run(
        graph, steps=list(DENIAL_TRACE), default=finish_on(MARK_STIPULATION_PASSAGE)
    )
    assert first["edges_written"] == 5
    assert first_mocks.llm.await_count == 3
    graph.add_edges_calls.clear()

    _, summary, mocks = await _run(graph, steps=list(DENIAL_TRACE))

    assert mocks.llm.await_count == 0
    assert summary["traces_started"] == 0
    assert graph.add_edges_calls == []
    assert summary["already_resolved"] == 4
    assert graph.update_node_calls == []


@pytest.mark.asyncio
async def test_edge_indexing_failure_still_patches_and_is_noted(caplog):
    graph = _base_graph()

    with _patched(graph, steps=list(DENIAL_TRACE)) as mocks:
        mocks.index_graph_edges.side_effect = RuntimeError("embedding provider is down")
        with caplog.at_level("WARNING"):
            payload = await detect_dangling_references(None)
            summary = await apply_reference_resolutions(payload)

    assert summary["edges_written"] == 4
    assert summary["nodes_patched"] >= 1
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

    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    graph.get_graph_data.assert_awaited()
    assert summary["resolved"] == 2


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_one_bad_reference_fails_only_itself():
    graph = _base_graph()

    def _explode(*args, **kwargs):
        raise RuntimeError("seed retrieval exploded")

    with _patched(graph, steps=[finish_on(MARK_STIPULATION_PASSAGE)]) as mocks:
        with patch(f"{PASS}.search_candidates", side_effect=_explode):
            payload = await detect_dangling_references(None)
            summary = await apply_reference_resolutions(payload)

    assert summary["failed"] == 2
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    # The reference the entity-name step answered still resolves.
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to")
    assert mocks.llm.await_count == 0


@pytest.mark.asyncio
async def test_a_missing_system_prompt_aborts_the_pass():
    """A missing prompt is a deployment bug, not 300 silent abstentions (R11)."""
    graph = _base_graph()

    with _patched(graph) as mocks:
        with patch(f"{TRACER}.read_query_prompt", return_value=None):
            with pytest.raises(FileNotFoundError):
                await detect_dangling_references(None)

    assert mocks.llm.await_count == 0


@pytest.mark.asyncio
async def test_a_blank_system_prompt_aborts_the_pass():
    graph = _base_graph()

    with _patched(graph):
        with patch(f"{TRACER}.read_query_prompt", return_value="   "):
            with pytest.raises(ValueError):
                await detect_dangling_references(None)


@pytest.mark.asyncio
async def test_a_missing_system_prompt_fails_the_task_on_the_pass_path():
    """R11 has to hold on the task entry point too: the memify registry binds it with
    scope="all", so its blanket except would otherwise turn a bad file into one WARNING
    and a pass that writes nothing."""
    graph = _base_graph()

    with _patched(graph) as mocks:
        with patch(f"{TRACER}.read_query_prompt", return_value=None):
            with pytest.raises(FileNotFoundError):
                await resolve_assertion_references(["item"])

    assert mocks.llm.await_count == 0
    assert graph.add_edges_calls == []


@pytest.mark.asyncio
async def test_a_blank_system_prompt_fails_the_task_on_the_pass_path():
    graph = _base_graph()

    with _patched(graph):
        with patch(f"{TRACER}.read_query_prompt", return_value="   "):
            with pytest.raises(ValueError):
                await resolve_assertion_references(["item"])


@pytest.mark.asyncio
async def test_the_tail_never_fails_its_pipeline_over_a_prompt_it_does_not_read():
    """The ingest tail runs no trace, so it never asks for the prompt -- and it must
    keep swallowing everything, because it may not break an ingestion."""
    graph = _base_graph()
    items = ["unchanged"]

    with _patched(graph) as mocks:
        with patch(f"{TRACER}.read_query_prompt", return_value=None):
            result = await resolve_assertion_references(items, scope="touched", allow_llm=False)

    assert result is items
    assert mocks.llm.await_count == 0


@pytest.mark.asyncio
async def test_the_pass_task_still_swallows_every_other_error(caplog):
    """Only the two configuration errors escape; a write that blew up still cannot break
    the pipeline the task is appended to."""
    graph = _base_graph()
    items = ["unchanged"]

    async def _boom(_edges, **_kwargs):
        raise RuntimeError("write failed")

    graph.add_edges = _boom
    with _patched(graph, steps=list(DENIAL_TRACE)):
        with caplog.at_level("WARNING"):
            result = await resolve_assertion_references(items)

    assert result is items
    assert any("write failed" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_a_mapping_failure_counts_only_its_own_reference(caplog):
    """R15: mapping a finish onto a Resolution runs inside the per-reference guard, so a
    bug there costs one reference rather than the whole pass."""
    graph = _base_graph()
    real_precheck = pass_module._edge_precheck

    def _explode(outcome, props, view):
        if outcome.resolution is not None and outcome.resolution.assertion_id == A_DENIAL:
            raise KeyError("the picked node left the view")
        return real_precheck(outcome, props, view)

    with _patched(graph, steps=list(DENIAL_TRACE)) as mocks:
        with patch(f"{PASS}._edge_precheck", side_effect=_explode):
            with caplog.at_level("WARNING"):
                payload = await detect_dangling_references(None)
                summary = await apply_reference_resolutions(payload)

    assert summary["failed"] == 1
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    # The pass carried on: the other references were still answered and written.
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to")
    assert mocks.llm.await_count == 3


def test_a_picked_document_the_view_no_longer_holds_never_raises():
    """R15: the document branch reads the view with .get(), so a node id that is not a
    document in this view degrades to an untyped anchor instead of a KeyError."""
    view = SimpleNamespace(
        assertions={},
        chunks={},
        documents={},
        document_by_chunk={},
        node_ids={"gone"},
    )
    entry = resolve_module._Pending(
        assertion_id="a1",
        field_name="responds_to",
        props={},
        hint=ReferenceHint(document_hint="the Complaint"),
        reference_text="the Complaint",
        fingerprint="ff",
        entry_notes=(),
        stale=False,
        own_chunk_touched=True,
    )
    answer = resolve_module._TraceAnswer(
        finish=TracerFinish(candidate_label="D1", confidence=0.9, reason="r"),
        trace=(),
        iterations=1,
        node_id="gone",
        targets=(),
        capped=False,
    )

    outcome = resolve_module._answer_to_outcome(entry, answer, view, threshold=0.6, counters={})

    assert outcome.kind == "resolved"
    assert outcome.resolution.anchor_id == "gone"
    assert outcome.resolution.anchor_type is None


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
async def test_touched_scope_spends_only_on_the_statements_it_touched():
    """R26: with a touched scope the pass filters its pending references by scope *before*
    seeding and tracing them, rather than paying for every dangling reference in the graph
    and discarding the answers afterwards."""
    graph = _base_graph()
    # An allegation in the freshly ingested chunk, carrying a dangling reference of its own.
    graph.nodes[A_3]["responds_to_ref"] = dict(STIPULATION_REF)
    touched = [
        SimpleNamespace(
            made_from=SimpleNamespace(
                id=COMPLAINT_CHUNK_1, is_part_of=SimpleNamespace(id=DOC_COMPLAINT)
            )
        )
    ]

    _, summary, mocks = await _run(
        graph,
        scope="touched",
        allow_llm=True,
        data=touched,
        steps=[finish_on(MARK_STIPULATION_PASSAGE, confidence=0.9)],
        default=abstain(),
    )

    # Exactly one trace ran: the reference the ingestion actually produced.
    assert summary["traces_started"] == 1
    assert mocks.llm.await_count == 1
    assert [edge[1] for edge in graph.edges_of(A_3, "responds_to")] == [STIPULATION_CHUNK_0]
    # The Answer's denial points at the freshly ingested Complaint, but the statement
    # making it was not touched, so this pass does not spend a call on it.
    assert graph.edges_of(A_DENIAL, "responds_to") == []
    assert graph.edges_of(A_STIPULATION, "responds_to") == []
    assert graph.edges_of(A_ATTRIBUTED, "attributed_to") == []
    assert summary["resolved"] == 1
    assert summary["scanned"] == 1


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

    with _patched(graph, steps=list(DENIAL_TRACE)):
        result = await resolve_assertion_references(items)

    assert result is items
    assert graph.add_edges_calls


@pytest.mark.asyncio
async def test_task_swallows_its_own_errors(caplog):
    items = ["unchanged"]
    with patch.object(
        resolve_module, "_load_graph_view", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        with patch.object(resolve_module, "get_graph_engine", new=AsyncMock()):
            with caplog.at_level("WARNING"):
                result = await resolve_assertion_references(items)

    assert result is items
    assert any("boom" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_all_scope_runs_at_most_once_per_pipeline_run():
    graph = _base_graph()
    ctx = PipelineContext(dataset=SimpleNamespace(id=DATASET_ID), pipeline_run_id="run-1")

    with _patched(graph, steps=list(DENIAL_TRACE)):
        await resolve_assertion_references("batch-1", ctx=ctx)
        await resolve_assertion_references("batch-2", ctx=ctx)

    assert ctx.extras["reference_resolution_ran"] is True
    assert len(graph.filtered_calls) == 1


@pytest.mark.asyncio
async def test_a_failed_pass_does_not_memoize_itself():
    """A first batch that blew up must not suppress the rest of the run."""
    graph = _base_graph()
    ctx = PipelineContext(dataset=SimpleNamespace(id=DATASET_ID), pipeline_run_id="run-1")

    with patch.object(
        resolve_module, "_load_graph_view", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        with patch.object(resolve_module, "get_graph_engine", new=AsyncMock()):
            await resolve_assertion_references("batch-1", ctx=ctx)

    assert "reference_resolution_ran" not in ctx.extras

    with _patched(graph, steps=list(DENIAL_TRACE)):
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
    with _patched(graph, steps=list(DENIAL_TRACE)):
        payload = await detect_dangling_references(None)
        summary = await apply_reference_resolutions([payload])

    assert summary["edges_written"] == 4


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
    with _patched(graph, steps=list(DENIAL_TRACE)):
        payload = await detect_dangling_references(None)
        with pytest.raises(RuntimeError, match="write failed"):
            await apply_reference_resolutions(payload)


# --------------------------------------------------------------------------- #
# summary shape
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_the_summary_reports_every_budget_counter():
    graph = _base_graph()
    _, summary, _ = await _run(graph, steps=list(DENIAL_TRACE))

    for key in (
        "llm_calls",
        "llm_calls_attempted",
        "llm_calls_stated",
        "llm_calls_inferred",
        "llm_budget",
        "traces_started",
        "traces_finished",
        "traces_iteration_capped",
        "llm_skipped_empty_graph",
        "llm_cached",
        "llm_abstained",
        "llm_below_threshold",
        "llm_unknown_label",
        "llm_failed",
        "llm_tokens_in",
        "llm_tokens_out",
        "inferred_scanned",
        "inferred_resolved",
    ):
        assert isinstance(summary[key], int), key
    assert isinstance(summary["llm_budget_exhausted"], bool)
    assert isinstance(summary["tool_calls_by_name"], dict)
    assert summary["llm_calls_stated"] == summary["llm_calls"]
    assert summary["llm_calls_inferred"] == 0
    # Every attempt came back, so the two spend counters agree here.
    assert summary["llm_calls_attempted"] == summary["llm_calls"]
    assert summary["traces_finished"] == 2
