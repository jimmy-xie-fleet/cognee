"""Unit tests for the resolve_references memify pipeline and its registry entry.

``memify`` itself is patched: what matters here is the wiring (which tasks run in
which phase, with which bound options, against which dataset) and the parameter
validation, not a second run of the resolver's own logic.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.memify_pipelines.memify_task_registry import (
    resolve_memify_tasks,
    supported_memify_task_names,
)
from cognee.memify_pipelines.resolve_references import resolve_references_pipeline
from cognee.tasks.graph import resolve_assertion_references
from cognee.tasks.graph.resolve_assertion_references import (
    apply_reference_resolutions,
    detect_dangling_references,
)

MODULE = "cognee.memify_pipelines.resolve_references"


def _patched_memify(result=None):
    user = MagicMock()
    user.id = "u1"
    dataset = SimpleNamespace(id="ds-1", owner_id="owner-1", name="adams_family_legal")
    memify_mock = AsyncMock(return_value=result if result is not None else {"status": "ok"})
    return user, dataset, memify_mock


async def _run(**kwargs):
    user, dataset, memify_mock = _patched_memify()
    with (
        patch(f"{MODULE}.get_default_user", new=AsyncMock(return_value=user)),
        patch(
            f"{MODULE}.get_authorized_existing_datasets", new=AsyncMock(return_value=[dataset])
        ) as authorized_mock,
        patch(f"{MODULE}.memify", new=memify_mock),
    ):
        result = await resolve_references_pipeline(**kwargs)
    return result, memify_mock, authorized_mock, user


@pytest.mark.asyncio
async def test_pipeline_wires_both_phases_against_the_target_dataset():
    result, memify_mock, authorized_mock, user = await _run(
        dataset="adams_family_legal",
        llm_max_calls=40,
        tracer_max_iter=3,
        llm_confidence_threshold=0.75,
        force=True,
    )

    assert result == {"status": "ok"}
    authorized_mock.assert_awaited_once_with(["adams_family_legal"], "write", user)

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["data"] == [{}]
    assert kwargs["dataset"] == "ds-1"
    assert kwargs["user"] is user
    assert kwargs["run_in_background"] is False

    (extraction_task,) = kwargs["extraction_tasks"]
    (enrichment_task,) = kwargs["enrichment_tasks"]
    assert extraction_task.executable is detect_dangling_references
    assert enrichment_task.executable is apply_reference_resolutions

    detect_options = extraction_task.default_params["kwargs"]
    assert detect_options["scope"] == "all"
    assert detect_options["force"] is True
    assert detect_options["llm_max_calls"] == 40
    assert detect_options["tracer_max_iter"] == 3
    assert detect_options["llm_confidence_threshold"] == 0.75
    # The sweep is the pass, so the tracer runs here whatever the tail does.
    assert detect_options["allow_llm"] is True
    assert "confidence_floor" not in detect_options
    assert "enable_prose_lookup" not in detect_options
    assert enrichment_task.default_params["kwargs"] == {"dry_run": False}


@pytest.mark.asyncio
async def test_pipeline_defaults_the_budget_to_the_config():
    _, memify_mock, _, _ = await _run()

    detect_options = memify_mock.call_args.kwargs["extraction_tasks"][0].default_params["kwargs"]
    assert detect_options["llm_max_calls"] is None
    assert detect_options["tracer_max_iter"] is None
    assert detect_options["llm_confidence_threshold"] is None


@pytest.mark.asyncio
async def test_pipeline_forwards_dry_run_and_background():
    _, memify_mock, _, _ = await _run(dry_run=True, run_in_background=True)

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["run_in_background"] is True
    assert kwargs["enrichment_tasks"][0].default_params["kwargs"] == {"dry_run": True}


@pytest.mark.asyncio
async def test_pipeline_does_not_enter_the_database_context():
    """SDK-483: holding the context's queue slot while memify waits on the dataset
    lock inverts the canonical lock order and can deadlock the process."""
    import cognee.memify_pipelines.resolve_references as module

    assert not hasattr(module, "set_database_global_context_variables")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_threshold", [0, -0.1, 1.5, "0.6", True])
async def test_pipeline_rejects_an_invalid_confidence_threshold(bad_threshold):
    with pytest.raises(CogneeValidationError):
        await resolve_references_pipeline(llm_confidence_threshold=bad_threshold)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_threshold", [0, -0.1, 1.5, "0.6", True])
async def test_pipeline_rejects_an_invalid_infer_threshold(bad_threshold):
    with pytest.raises(CogneeValidationError):
        await resolve_references_pipeline(infer_confidence_threshold=bad_threshold)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_budget", [-1, 1.5, "10", True])
async def test_pipeline_rejects_an_invalid_call_budget(bad_budget):
    with pytest.raises(CogneeValidationError):
        await resolve_references_pipeline(llm_max_calls=bad_budget)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_iterations", [0, -1, 1.5, "4", True])
async def test_pipeline_rejects_an_invalid_iteration_cap(bad_iterations):
    with pytest.raises(CogneeValidationError):
        await resolve_references_pipeline(tracer_max_iter=bad_iterations)


@pytest.mark.asyncio
async def test_pipeline_forwards_the_unstated_inference_options():
    """Accepted and forwarded now; the pass itself lands in the unstated-inference task."""
    _, memify_mock, _, _ = await _run(infer_unstated=True, infer_confidence_threshold=0.8)

    detect_options = memify_mock.call_args.kwargs["extraction_tasks"][0].default_params["kwargs"]
    assert detect_options["infer_unstated"] is True
    assert detect_options["infer_confidence_threshold"] == 0.8


@pytest.mark.asyncio
async def test_pipeline_rejects_a_dataset_without_write_access():
    user = MagicMock()
    with (
        patch(f"{MODULE}.get_default_user", new=AsyncMock(return_value=user)),
        patch(f"{MODULE}.get_authorized_existing_datasets", new=AsyncMock(return_value=[])),
        patch(f"{MODULE}.memify", new=AsyncMock()) as memify_mock,
    ):
        with pytest.raises(CogneeValidationError):
            await resolve_references_pipeline(dataset="nope")

    memify_mock.assert_not_awaited()


def test_registry_exposes_the_resolver_task():
    assert "resolve_references" in supported_memify_task_names()

    resolved = resolve_memify_tasks(["resolve_references"])
    assert resolved is not None
    assert resolved[0].executable is resolve_assertion_references
