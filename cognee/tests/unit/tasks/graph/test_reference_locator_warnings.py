"""Locator qualifications survive the tracer's character cap."""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.graph.utils.reference_candidates import LabelRegistry
from cognee.tasks.graph.reference_graph_view import DocumentTextCache
from cognee.tasks.graph.reference_pass import PassContext
from cognee.tasks.graph.reference_retrieval import LexicalIndex
from cognee.tasks.graph.reference_tracer import (
    CallBudget,
    TracerFinish,
    TracerStep,
    TracerToolCall,
    trace_reference,
)
from cognee.tasks.graph.reference_tracer_tools import (
    MAX_TOOL_OUTPUT_CHARS,
    build_tracer_tools,
    run_tool,
)
from cognee.tests.unit.tasks.graph._reference_fakes import (
    assertion_node,
    build_graph_view,
    chunk_node,
    document_node,
    nid,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("readable_text", [True, False])
async def test_locator_warning_survives_a_long_assertion_list(readable_text):
    document_id, chunk_id = nid("long-section-document"), nid("long-section-chunk")
    statements = [
        f"Allegation {index:02d}: " + "the defendant breached the lease. " * 7
        for index in range(30)
    ]
    body = "\n".join(["Section 1", *statements, "Section 1", "A conflicting second section."])
    chunk, edge = chunk_node(chunk_id, body, 0, document_id)
    view = await build_graph_view(
        [
            document_node(document_id, "Exhibit compilation", raw_data_location="/fake/text"),
            chunk,
            *(
                assertion_node(nid(f"long-section-{index}"), text, chunk_id, source_quote=text)
                for index, text in enumerate(statements)
            ),
        ],
        [edge],
    )
    ctx = PassContext(
        view=view,
        texts=DocumentTextCache(view),
        lexical=LexicalIndex(view),
        budget=CallBudget(2),
        max_iter=2,
    )
    registry = LabelRegistry()
    args = {
        "document": registry.label(document_id, "TextDocument"),
        "kind": "section",
        "value": "1",
    }
    tools = build_tracer_tools(ctx, registry=registry)
    llm = AsyncMock(
        side_effect=[
            TracerStep(tool_call=TracerToolCall(tool_name="locate_paragraph", arguments=args)),
            TracerStep(finish=TracerFinish()),
        ]
    )
    read = AsyncMock(return_value=body) if readable_text else AsyncMock(side_effect=OSError())
    with (
        patch("cognee.tasks.graph.reference_graph_view._read_processed_text", read),
        patch("cognee.tasks.graph.reference_tracer.LLMGateway.acreate_structured_output", llm),
    ):
        result = await run_tool(tools, "locate_paragraph", args)
        assert len(result) > MAX_TOOL_OUTPUT_CHARS
        await trace_reference(
            ctx,
            unstated=False,
            hint=None,
            source_props={"name": "denial"},
            source_document_name="Answer",
            field_name="responds_to",
            seed=[],
            tools=tools,
            registry=registry,
        )

    model_context = llm.call_args_list[1].kwargs["text_input"]
    expected_warning = (
        "several section 1 markers in this document; this is the first"
        if readable_text
        else "found by scanning stored passages, not the document text"
    )
    assert expected_warning in model_context
    assert "Passages: [P1]" in model_context
    assert "[truncated]" in model_context
