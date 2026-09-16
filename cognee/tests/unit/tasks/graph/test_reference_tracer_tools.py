"""Unit tests for the reference tracer's five read-only tools.

Every tool runs against a :class:`GraphView` built from plain node/edge tuples and, where
a tool reads stored document text, a patched ``_read_processed_text``. No vector backend,
no graph backend, no LLM, no network, no real filesystem: the one seam that would touch a
file is patched at ``cognee.tasks.graph.reference_graph_view._read_processed_text``, the
target Task 6 documented.

The labels in every tool's output come from one shared :class:`LabelRegistry`, so a label
the agent sees from ``list_documents`` is the same label ``search`` and ``read_chunk`` use
for that node -- the tests below pin that, because it is the only thing that makes a
finish resolvable back to a node id.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.graph.utils.reference_candidates import (
    Candidate,
    LabelRegistry,
    format_candidate_lines,
)
from cognee.tasks.graph.reference_graph_view import DocumentTextCache
from cognee.tasks.graph.reference_pass import PassContext
from cognee.tasks.graph.reference_retrieval import LexicalIndex
from cognee.tasks.graph.reference_tracer_tools import (
    DOCUMENT_LIST_CAP,
    LIST_PREVIEW_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
    TOOL_NAMES,
    build_tracer_tools,
    render_tool_manifest,
    run_tool,
)
from cognee.tests.unit.tasks.graph._reference_fakes import (
    assertion_node,
    build_graph_view,
    chunk_node,
    document_node,
    nid,
)

MODULE = "cognee.tasks.graph.reference_tracer_tools"
VIEW_MODULE = "cognee.tasks.graph.reference_graph_view"

DOC_COMPLAINT = nid("tools-doc-complaint")
DOC_ANSWER = nid("tools-doc-answer")
COMPLAINT_NAME = "Verified_Complaint_Adams"
# Deliberately opaque: a document must be reachable by content, never by its filename.
ANSWER_NAME = "SKM_C55826082316050"

C0 = nid("tools-complaint-chunk-0")
C1 = nid("tools-complaint-chunk-1")
A0 = nid("tools-answer-chunk-0")

C0_TEXT = "COMPLAINT\n\n¶ 1 The plaintiff owns 10 Main Street.\n\n"
C1_TEXT = (
    "¶ 5 The defendant failed to repair the roof in 2019.\n\n"
    "¶ 6 The defendant refused to pay rent.\n"
)
COMPLAINT_TEXT = C0_TEXT + C1_TEXT
A0_TEXT = "The defendant denies that the roof was ever in disrepair."

A_ROOF = nid("tools-assertion-roof")
A_RENT = nid("tools-assertion-rent")
A_DENY = nid("tools-assertion-deny")


def _graph():
    """Nodes and edges of the two-document fixture every tool test shares."""
    complaint = document_node(
        DOC_COMPLAINT, COMPLAINT_NAME, raw_data_location="/fake/complaint.txt"
    )
    answer = document_node(DOC_ANSWER, ANSWER_NAME)

    (c0, c0_edge) = chunk_node(C0, C0_TEXT, 0, DOC_COMPLAINT)
    (c1, c1_edge) = chunk_node(C1, C1_TEXT, 1, DOC_COMPLAINT)
    (a0, a0_edge) = chunk_node(A0, A0_TEXT, 0, DOC_ANSWER)

    nodes = [
        complaint,
        answer,
        c0,
        c1,
        a0,
        assertion_node(
            A_ROOF,
            "the defendant failed to repair the roof",
            C1,
            source_quote="The defendant failed to repair the roof in 2019.",
        ),
        assertion_node(
            A_RENT,
            "the defendant refused to pay rent",
            C1,
            source_quote="The defendant refused to pay rent.",
        ),
        assertion_node(
            A_DENY,
            "the roof was in disrepair",
            A0,
            statement_type="denial",
            polarity="negative",
            source_quote="The defendant denies that the roof was ever in disrepair.",
        ),
    ]
    return nodes, [c0_edge, c1_edge, a0_edge]


def _context(view) -> PassContext:
    """The read layer one trace's tools close over, as the pass hands it to them."""
    return PassContext(view=view, texts=DocumentTextCache(view), lexical=LexicalIndex(view))


async def _tools(*, registry=None, **kwargs):
    """The five tools over the shared fixture, plus the objects they close over."""
    nodes, edges = _graph()
    view = await build_graph_view(nodes, edges)
    registry = registry if registry is not None else LabelRegistry()
    tools = build_tracer_tools(_context(view), registry=registry, **kwargs)
    return tools, view, registry


def _read_text_patch(text_by_location):
    async def _fake_read(location):
        if location not in text_by_location:
            raise OSError(f"no such file: {location}")
        return text_by_location[location]

    return patch(f"{VIEW_MODULE}._read_processed_text", side_effect=_fake_read)


# --------------------------------------------------------------------------- #
# build_tracer_tools / manifest
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_tracer_tools_returns_the_five_named_tools():
    tools, _, _ = await _tools()

    assert list(tools) == list(TOOL_NAMES)
    assert list(TOOL_NAMES) == [
        "search",
        "list_documents",
        "open_document",
        "read_chunk",
        "locate_paragraph",
    ]
    assert all(spec.name == name for name, spec in tools.items())


@pytest.mark.asyncio
async def test_manifest_contains_each_tool_description_and_json_schema():
    tools, _, _ = await _tools()

    manifest = render_tool_manifest(tools)

    for name, spec in tools.items():
        assert name in manifest
        assert spec.description.split(".")[0] in manifest
        # The full JSON schema, compactly rendered, so types/required/enums reach the model.
        schema = json.dumps(
            spec.args_model.model_json_schema(), sort_keys=True, separators=(",", ":")
        )
        assert schema in manifest


@pytest.mark.asyncio
async def test_manifest_schema_carries_the_search_kind_enum():
    tools, _, _ = await _tools()

    manifest = render_tool_manifest(tools)

    assert "documents" in manifest and "assertions" in manifest and "passages" in manifest
    # kind="documents" is the only way to find a document by its content.
    assert "kind" in tools["search"].description


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_search_renders_candidate_lines():
    tools, _, registry = await _tools()
    candidates = [
        Candidate(
            label="A1",
            node_id=A_ROOF,
            node_type="Assertion",
            score=0.9,
            text="the defendant failed to repair the roof",
            document_name=COMPLAINT_NAME,
            chunk_index=1,
        )
    ]

    with patch(f"{MODULE}.search_candidates", AsyncMock(return_value=candidates)):
        result = await run_tool(tools, "search", {"query": "the roof"})

    assert result == format_candidate_lines(candidates)
    assert "[A1]" in result


@pytest.mark.asyncio
async def test_search_passes_kind_and_top_k_to_search_candidates():
    tools, view, registry = await _tools()

    fake = AsyncMock(return_value=[])
    with patch(f"{MODULE}.search_candidates", fake):
        await run_tool(
            tools, "search", {"query": "a June 10 letter", "kind": "documents", "top_k": 4}
        )

    kwargs = fake.await_args.kwargs
    assert kwargs["queries"] == ["a June 10 letter"]
    assert kwargs["kind"] == "documents"
    assert kwargs["limit"] == 4
    assert kwargs["view"] is view
    assert kwargs["registry"] is registry


@pytest.mark.asyncio
async def test_search_passes_the_trace_scoping_through():
    exclude = {A_ROOF}
    tools, _, _ = await _tools(
        exclude_ids=exclude, own_document_id=DOC_COMPLAINT, penalize_own_document=True
    )

    fake = AsyncMock(return_value=[])
    with patch(f"{MODULE}.search_candidates", fake):
        await run_tool(tools, "search", {"query": "the roof"})

    kwargs = fake.await_args.kwargs
    assert kwargs["exclude_ids"] == exclude
    assert kwargs["own_document_id"] == DOC_COMPLAINT
    assert kwargs["penalize_own_document"] is True


@pytest.mark.asyncio
async def test_search_without_matches_says_so():
    tools, _, _ = await _tools()

    with patch(f"{MODULE}.search_candidates", AsyncMock(return_value=[])):
        result = await run_tool(tools, "search", {"query": "nothing at all"})

    assert result == "No matches."


@pytest.mark.asyncio
async def test_search_rejects_an_out_of_range_top_k():
    tools, _, _ = await _tools()

    result = await run_tool(tools, "search", {"query": "x", "top_k": 99})

    assert result.startswith("ERROR:")
    assert "top_k" in result


@pytest.mark.asyncio
async def test_search_handler_failure_becomes_an_error_string():
    tools, _, _ = await _tools()

    with patch(f"{MODULE}.search_candidates", AsyncMock(side_effect=RuntimeError("index down"))):
        result = await run_tool(tools, "search", {"query": "x"})

    assert result.startswith("ERROR:")
    assert "index down" in result


# --------------------------------------------------------------------------- #
# list_documents
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_documents_lists_every_document_sorted_by_name():
    tools, _, registry = await _tools()

    result = await run_tool(tools, "list_documents", {})

    lines = result.splitlines()
    assert len(lines) == 2
    # "SKM_..." sorts before "Verified_...", so the answer takes the first label.
    assert lines[0].startswith("[D1] SKM_C55826082316050 — 1 passages — ")
    assert lines[1].startswith("[D2] Verified_Complaint_Adams — 2 passages — ")
    assert "The defendant denies that the roof" in lines[0]
    assert registry.resolve("D1") == DOC_ANSWER
    assert registry.resolve("D2") == DOC_COMPLAINT


@pytest.mark.asyncio
async def test_list_documents_caps_the_list():
    nodes, edges = [], []
    for index in range(DOCUMENT_LIST_CAP + 5):
        document_id = nid(f"tools-bulk-doc-{index:03d}")
        nodes.append(document_node(document_id, f"Document{index:03d}"))
        chunk, edge = chunk_node(nid(f"tools-bulk-chunk-{index:03d}"), "text", 0, document_id)
        nodes.append(chunk)
        edges.append(edge)

    view = await build_graph_view(nodes, edges)
    tools = build_tracer_tools(_context(view), registry=LabelRegistry())

    result = await run_tool(tools, "list_documents", {})

    lines = result.splitlines()
    assert len(lines) == DOCUMENT_LIST_CAP + 1
    assert lines[-1] == '… and 5 more (use search kind="documents")'


@pytest.mark.asyncio
async def test_list_documents_fits_the_output_cap_and_always_keeps_its_tail():
    # Finding 2: 80 rows of 240-char previews overflow MAX_TOOL_OUTPUT_CHARS, and the
    # loop's truncation eats the "and N more" line -- so the agent sees a silently short
    # list with no hint that anything is missing.
    nodes, edges = [], []
    for index in range(DOCUMENT_LIST_CAP):
        document_id = nid(f"tools-fat-doc-{index:03d}")
        nodes.append(document_node(document_id, f"A_very_long_document_name_{index:03d}"))
        chunk, edge = chunk_node(
            nid(f"tools-fat-chunk-{index:03d}"), "paragraph text " * 60, 0, document_id
        )
        nodes.append(chunk)
        edges.append(edge)

    view = await build_graph_view(nodes, edges)
    registry = LabelRegistry()
    tools = build_tracer_tools(_context(view), registry=registry)

    result = await run_tool(tools, "list_documents", {})

    lines = result.splitlines()
    assert len(result) <= MAX_TOOL_OUTPUT_CHARS
    assert len(lines) < DOCUMENT_LIST_CAP  # the character budget bound before the row cap
    assert lines[-1].startswith("… and ")
    assert 'search kind="documents"' in lines[-1]
    shown = len(lines) - 1
    assert lines[-1] == f'… and {DOCUMENT_LIST_CAP - shown} more (use search kind="documents")'
    # Only the documents actually listed consumed a label.
    assert len(registry.labels()) == shown


@pytest.mark.asyncio
async def test_list_documents_previews_are_short():
    tools, _, _ = await _tools()

    result = await run_tool(tools, "list_documents", {})

    for line in result.splitlines():
        preview = line.split(" — ", 2)[-1]
        assert len(preview) <= LIST_PREVIEW_CHARS + 2  # the two quote characters


@pytest.mark.asyncio
async def test_list_documents_on_an_empty_view():
    view = await build_graph_view([], [])
    tools = build_tracer_tools(_context(view), registry=LabelRegistry())

    assert await run_tool(tools, "list_documents", {}) == "No documents."


# --------------------------------------------------------------------------- #
# open_document
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_open_document_returns_consecutive_passages():
    tools, _, registry = await _tools()
    await run_tool(tools, "list_documents", {})  # assigns D1/D2

    result = await run_tool(tools, "open_document", {"document": "D2"})

    lines = result.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[P1] (chunk 0): ")
    assert lines[1].startswith("[P2] (chunk 1): ")
    assert registry.resolve("P1") == C0
    assert registry.resolve("P2") == C1


@pytest.mark.asyncio
async def test_open_document_pages_from_a_chunk_with_a_count():
    tools, _, registry = await _tools()
    await run_tool(tools, "list_documents", {})

    result = await run_tool(tools, "open_document", {"document": "D2", "from_chunk": 1, "count": 1})

    assert result.splitlines() == [result]
    assert result.startswith("[P1] (chunk 1): ")
    assert registry.resolve("P1") == C1


@pytest.mark.asyncio
async def test_open_document_past_the_end_reports_no_passages():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    result = await run_tool(tools, "open_document", {"document": "D2", "from_chunk": 9})

    assert not result.startswith("ERROR:")
    assert "no passages" in result.lower()


@pytest.mark.asyncio
async def test_open_document_with_an_unknown_label_is_an_error():
    tools, _, _ = await _tools()

    result = await run_tool(tools, "open_document", {"document": "D7"})

    assert result.startswith("ERROR: unknown document label")


@pytest.mark.asyncio
async def test_open_document_rejects_a_count_above_the_cap():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    result = await run_tool(tools, "open_document", {"document": "D2", "count": 9})

    assert result.startswith("ERROR:")
    assert "count" in result


# --------------------------------------------------------------------------- #
# read_chunk
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_read_chunk_returns_full_text_and_the_assertions_quoted_in_it():
    tools, _, registry = await _tools()
    await run_tool(tools, "list_documents", {})
    await run_tool(tools, "open_document", {"document": "D2"})

    result = await run_tool(tools, "read_chunk", {"passage": "P2"})

    assert C1_TEXT.strip() in result
    assert "Assertions quoted in this passage:" in result
    assert "allegation/positive: the defendant failed to repair the roof" in result
    assert "allegation/positive: the defendant refused to pay rent" in result
    # The denial lives in the other document's chunk and must not be listed here.
    assert "the roof was in disrepair" not in result
    assert registry.resolve("A1") in (A_ROOF, A_RENT)


@pytest.mark.asyncio
async def test_read_chunk_without_assertions_says_none():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})
    await run_tool(tools, "open_document", {"document": "D2"})

    result = await run_tool(tools, "read_chunk", {"passage": "P1"})

    assert "Assertions quoted in this passage:" in result
    assert "(none)" in result


@pytest.mark.asyncio
async def test_read_chunk_truncates_a_very_long_passage():
    long_text = "word " * 4000
    document_id = nid("tools-long-doc")
    chunk_id = nid("tools-long-chunk")
    chunk, edge = chunk_node(chunk_id, long_text, 0, document_id)
    view = await build_graph_view([document_node(document_id, "Long"), chunk], [edge])
    registry = LabelRegistry()
    tools = build_tracer_tools(_context(view), registry=registry)
    await run_tool(tools, "list_documents", {})
    await run_tool(tools, "open_document", {"document": "D1"})

    result = await run_tool(tools, "read_chunk", {"passage": "P1"})

    body = result.split("\n\nAssertions quoted in this passage:")[0]
    # Finding 1: the WHOLE result has to fit the cap, not just the body -- the loop
    # truncates to the same number and would otherwise cut the assertion section off.
    assert len(result) <= MAX_TOOL_OUTPUT_CHARS
    assert len(body) < MAX_TOOL_OUTPUT_CHARS
    assert body.endswith("…")
    assert "Assertions quoted in this passage:" in result


@pytest.mark.asyncio
async def test_read_chunk_keeps_its_assertion_labels_on_a_long_passage():
    # The reviewer's reproduction: a passage just over the cap loses every [A…] label to
    # the loop's truncation, which is the only thing that makes the passage resolvable.
    document_id = nid("tools-overflow-doc")
    chunk_id = nid("tools-overflow-chunk")
    quote = "The defendant failed to repair the roof in 2019."
    chunk, edge = chunk_node(chunk_id, ("filler " * 900) + quote, 0, document_id)
    assertion_id = nid("tools-overflow-assertion")
    view = await build_graph_view(
        [
            document_node(document_id, "Overflowing"),
            chunk,
            assertion_node(
                assertion_id,
                "the defendant failed to repair the roof",
                chunk_id,
                source_quote=quote,
            ),
        ],
        [edge],
    )
    registry = LabelRegistry()
    tools = build_tracer_tools(_context(view), registry=registry)
    await run_tool(tools, "list_documents", {})
    await run_tool(tools, "open_document", {"document": "D1"})

    result = await run_tool(tools, "read_chunk", {"passage": "P1"})

    assert len(result) > MAX_TOOL_OUTPUT_CHARS - 200  # genuinely at the cap
    assert len(result) <= MAX_TOOL_OUTPUT_CHARS
    assert "Assertions quoted in this passage:" in result
    assert "allegation/positive: the defendant failed to repair the roof" in result


@pytest.mark.asyncio
async def test_read_chunk_with_an_unknown_label_is_an_error():
    tools, _, _ = await _tools()

    result = await run_tool(tools, "read_chunk", {"passage": "P9"})

    assert result.startswith("ERROR: unknown passage label")


# --------------------------------------------------------------------------- #
# locate_paragraph
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_locate_paragraph_returns_the_span_its_passages_and_its_assertions():
    tools, _, registry = await _tools()
    await run_tool(tools, "list_documents", {})

    with _read_text_patch({"/fake/complaint.txt": COMPLAINT_TEXT}):
        result = await run_tool(
            tools, "locate_paragraph", {"document": "D2", "kind": "paragraph", "value": "5"}
        )

    assert "The defendant failed to repair the roof in 2019." in result
    # The span stops at the next marker, so ¶ 6 is outside it.
    assert "refused to pay rent" not in result.split("Passages:")[0]
    assert "Passages: [P1] (chunk 1, anchor)" in result
    assert "Assertions quoted in the span:" in result
    assert "allegation/positive: the defendant failed to repair the roof" in result
    assert "the defendant refused to pay rent" not in result
    assert registry.resolve("P1") == C1


@pytest.mark.asyncio
async def test_locate_paragraph_degrades_to_a_chunk_scan_without_readable_text():
    # The stored text cannot be read (a PDF opened as UTF-8, a file that moved), so
    # DocumentTextCache reports "no text" and the tool scans the stored chunks instead.
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    with _read_text_patch({}):
        result = await run_tool(
            tools, "locate_paragraph", {"document": "D2", "kind": "paragraph", "value": "5"}
        )

    assert not result.startswith("ERROR:")
    assert "The defendant failed to repair the roof in 2019." in result
    assert "Passages: [P1] (chunk 1, anchor)" in result
    # Finding 3: a degraded lookup must not look identical to an authoritative one.
    assert "(note: found by scanning stored passages, not the document text)" in result


@pytest.mark.asyncio
async def test_locate_paragraph_says_when_the_marker_was_ambiguous():
    # Two "¶ 5" markers and no "¶ 6", so find_locator_span cannot pick by sequence and
    # returns its ambiguous_marker note. The agent has to be told it got a guess.
    document_id = nid("tools-ambiguous-doc")
    chunk_text = "¶ 5 First mention of the roof.\n\n¶ 5 Second mention of the roof.\n"
    chunk, edge = chunk_node(nid("tools-ambiguous-chunk"), chunk_text, 0, document_id)
    view = await build_graph_view(
        [document_node(document_id, "Ambiguous", raw_data_location="/fake/ambiguous.txt"), chunk],
        [edge],
    )
    registry = LabelRegistry()
    tools = build_tracer_tools(_context(view), registry=registry)
    await run_tool(tools, "list_documents", {})

    with _read_text_patch({"/fake/ambiguous.txt": chunk_text}):
        result = await run_tool(
            tools, "locate_paragraph", {"document": "D1", "kind": "paragraph", "value": "5"}
        )

    assert not result.startswith("ERROR:")
    assert "(note: several paragraph 5 markers in this document; this is the first)" in result


@pytest.mark.asyncio
async def test_locate_paragraph_adds_no_note_when_the_lookup_was_authoritative():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    with _read_text_patch({"/fake/complaint.txt": COMPLAINT_TEXT}):
        result = await run_tool(
            tools, "locate_paragraph", {"document": "D2", "kind": "paragraph", "value": "5"}
        )

    assert "(note:" not in result


@pytest.mark.asyncio
async def test_locate_paragraph_reports_a_missing_marker():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    with _read_text_patch({"/fake/complaint.txt": COMPLAINT_TEXT}):
        result = await run_tool(
            tools, "locate_paragraph", {"document": "D2", "kind": "paragraph", "value": "42"}
        )

    assert result.startswith("ERROR: marker not found")
    assert "42" in result


@pytest.mark.asyncio
async def test_locate_paragraph_rejects_a_value_that_is_not_a_locator():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    result = await run_tool(
        tools, "locate_paragraph", {"document": "D2", "kind": "paragraph", "value": "   "}
    )

    assert result.startswith("ERROR:")


@pytest.mark.asyncio
async def test_locate_paragraph_rejects_an_unsupported_kind():
    tools, _, _ = await _tools()
    await run_tool(tools, "list_documents", {})

    result = await run_tool(
        tools, "locate_paragraph", {"document": "D2", "kind": "page", "value": "3"}
    )

    assert result.startswith("ERROR:")


@pytest.mark.asyncio
async def test_locate_paragraph_with_an_unknown_document_label_is_an_error():
    tools, _, _ = await _tools()

    result = await run_tool(
        tools, "locate_paragraph", {"document": "D4", "kind": "paragraph", "value": "5"}
    )

    assert result.startswith("ERROR: unknown document label")


# --------------------------------------------------------------------------- #
# dispatch + label stability
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unknown_tool_name_is_an_error_string():
    tools, _, _ = await _tools()

    result = await run_tool(tools, "teleport", {})

    assert result.startswith("ERROR: unknown tool")
    assert "teleport" in result


@pytest.mark.asyncio
async def test_labels_are_shared_across_tools():
    tools, _, registry = await _tools()

    candidate = Candidate(
        label=registry.label(C1, "DocumentChunk"),
        node_id=C1,
        node_type="DocumentChunk",
        score=0.8,
        text="the roof",
    )
    with patch(f"{MODULE}.search_candidates", AsyncMock(return_value=[candidate])):
        searched = await run_tool(tools, "search", {"query": "the roof"})

    opened = await run_tool(tools, "list_documents", {})
    opened += await run_tool(tools, "open_document", {"document": "D2"})

    assert f"[{candidate.label}]" in searched
    assert f"[{candidate.label}] (chunk 1)" in opened
    assert registry.resolve(candidate.label) == C1
