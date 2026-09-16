"""The resolver's allowance counts gateway invocations, not adapter retries."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)
from cognee.modules.graph.utils.reference_candidates import LabelRegistry
from cognee.tasks.graph.reference_graph_view import DocumentTextCache, GraphView
from cognee.tasks.graph.reference_pass import PassContext
from cognee.tasks.graph.reference_tracer import CallBudget, trace_reference


@pytest.mark.asyncio
async def test_gateway_allowance_is_reported_separately_from_provider_retries():
    view = GraphView()
    ctx = PassContext(view=view, texts=DocumentTextCache(view), budget=CallBudget(1), max_iter=1)
    registry = LabelRegistry()
    label = registry.label("target", "Assertion")
    adapter = NativeLiteLLMAdapter(
        api_key="unused-offline-key", model="offline-model", max_completion_tokens=100
    )

    def response(content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    provider = AsyncMock(
        side_effect=[
            response("INVALID JSON"),
            response(json.dumps({"finish": {"candidate_label": label, "confidence": 0.9}})),
        ]
    )
    native = "cognee.infrastructure.llm.structured_output_framework.litellm_native"
    with (
        patch(
            "cognee.infrastructure.llm.LLMGateway.get_llm_config",
            return_value=SimpleNamespace(structured_output_framework="LITELLM_NATIVE"),
        ),
        patch(f"{native}.get_native_client.get_native_client", return_value=adapter),
        patch(f"{native}.native_adapter._supports_native_schema", return_value=False),
        patch(f"{native}.native_adapter.llm_rate_limiter_context_manager", nullcontext),
        patch("litellm.acompletion", provider),
    ):
        finish, _, iterations = await trace_reference(
            ctx,
            unstated=False,
            hint=None,
            source_props={"name": "source"},
            source_document_name="source document",
            field_name="responds_to",
            seed=[],
            tools={},
            registry=registry,
        )
        # The exhausted gateway allowance prevents another invocation, even though
        # the adapter's first invocation required two provider attempts.
        again, _, next_iterations = await trace_reference(
            ctx,
            unstated=False,
            hint=None,
            source_props={"name": "source"},
            source_document_name="source document",
            field_name="responds_to",
            seed=[],
            tools={},
            registry=registry,
        )

    assert ctx.summary["llm_budget_unit"] == "gateway_calls"
    assert finish.candidate_label == label
    assert iterations == ctx.budget.used == ctx.counters["llm_calls"] == 1
    assert provider.await_count == 2
    assert again.candidate_label is None
    assert next_iterations == 0
    assert ctx.counters["llm_budget_exhausted"] == 1
