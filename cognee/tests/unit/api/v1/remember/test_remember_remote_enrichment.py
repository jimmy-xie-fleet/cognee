"""Remote remember must reject tasks that its HTTP request cannot carry."""

from contextlib import asynccontextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.remember.remember import remember
from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.modules.pipelines.tasks.task import Task


async def _enrich(data):
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize("enrichment_tasks", [None, [], [Task(_enrich)]])
async def test_remote_remember_enrichment_options_reach_the_http_boundary(
    monkeypatch, enrichment_tasks
):
    client = CloudClient("http://remote.invalid", "unused")
    requests = []

    @asynccontextmanager
    async def post(url, **kwargs):
        requests.append((url, kwargs["data"]))
        yield SimpleNamespace(status=200, json=AsyncMock(return_value={"status": "completed"}))

    get_session = AsyncMock(return_value=SimpleNamespace(post=post))
    monkeypatch.setattr(client, "_get_session", get_session)
    monkeypatch.setattr(import_module("cognee.api.v1.serve.state"), "_remote_client", client)
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")

    if enrichment_tasks:
        with pytest.raises(ValueError, match="enrichment_tasks.*remote Cognee instance"):
            await remember("Document text.", enrichment_tasks=enrichment_tasks)
        get_session.assert_not_awaited()
        assert requests == []
    else:
        result = await remember("Document text.", enrichment_tasks=enrichment_tasks)
        assert result == {"status": "completed"}
        assert len(requests) == 1
        url, form = requests[0]
        assert url == "http://remote.invalid/api/v1/remember"
        assert [options["name"] for options, _, _ in form._fields] == ["datasetName", "data"]
