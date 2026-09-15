"""Wiring tests for the generic ``enrichment_tasks`` cognify() parameter.

``enrichment_tasks`` lets a caller (a domain profile, an SDK script) append arbitrary
``Task`` objects to the very end of the default cognify pipeline, after every other
optional tail item (record_provenance, detect_contradictions,
resolve_temporal_contradictions). It is a named parameter rather than a config flag so
core never has to import a domain task or a name registry — see
``cognee/domains/legal/profile.py`` for the first consumer. No real API keys are
required (get_cognify_config is patched; config + chunk_size are passed so no
ontology/LLM setup runs). Pattern and Python-3.10 patch.object workaround copied from
``test_contradiction_detection_wiring.py``.
"""

import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.cognify.cognify import cognify, get_default_tasks
from cognee.api.v1.remember.remember import _COGNIFY_ONLY, RememberKwargs, remember
from cognee.modules.cognify.config import CognifyConfig
from cognee.modules.pipelines.tasks.task import Task

# Patching by the dotted string "cognee.api.v1.serve.state…" fails on
# Python 3.10: `cognee.api.v1.serve` is shadowed by the re-exported serve()
# function, and pre-3.11 mock walks attributes instead of importing modules.
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]
remember_module = sys.modules["cognee.api.v1.remember.remember"]

_mod_serve_state = importlib.import_module("cognee.api.v1.serve.state")
_mod_migrations_startup = importlib.import_module("cognee.modules.migrations.startup")
_mod_engine_setup = importlib.import_module("cognee.modules.engine.operations.setup")
_pkg_add = importlib.import_module("cognee.api.v1.add")
_mod_users_methods = importlib.import_module("cognee.modules.users.methods")

# The canonical pre-detection task order (see test_contradiction_detection_wiring.py).
_BASE_SEQUENCE = [
    "classify_documents",
    "extract_chunks_from_documents",
    "extract_graph_and_summarize",
    "add_data_points",
]


async def _fake_enrichment_task(data=None):
    """Stand-in enrichment task; only its name and bound config matter here."""
    return data


class TestGetDefaultTasksAppendsEnrichmentTasks:
    @pytest.mark.asyncio
    async def test_enrichment_task_appended_after_base_sequence(self):
        config = CognifyConfig()
        with patch.object(cognify_module, "get_cognify_config", return_value=config):
            tasks = await get_default_tasks(
                config={"ontology_config": {"ontology_resolver": None}},
                chunk_size=1024,
                chunks_per_batch=7,
                enrichment_tasks=[Task(_fake_enrichment_task)],
            )

        names = [task.executable.__name__ for task in tasks]
        assert names == _BASE_SEQUENCE + ["_fake_enrichment_task"]
        assert tasks[-1].task_config["batch_size"] == 7

    @pytest.mark.asyncio
    async def test_enrichment_task_still_last_with_other_optional_tail_items(self):
        config = CognifyConfig(contradiction_detection=True)
        with patch.object(cognify_module, "get_cognify_config", return_value=config):
            tasks = await get_default_tasks(
                config={"ontology_config": {"ontology_resolver": None}},
                chunk_size=1024,
                chunks_per_batch=5,
                functional_relationships={"ceo_of"},
                enrichment_tasks=[Task(_fake_enrichment_task)],
            )

        names = [task.executable.__name__ for task in tasks]
        assert names == _BASE_SEQUENCE + [
            "detect_contradictions",
            "resolve_temporal_contradictions",
            "_fake_enrichment_task",
        ]
        assert tasks[-1].task_config["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_none_and_empty_enrichment_tasks_leave_list_unchanged(self):
        config = CognifyConfig()
        with patch.object(cognify_module, "get_cognify_config", return_value=config):
            tasks_none = await get_default_tasks(
                config={"ontology_config": {"ontology_resolver": None}},
                chunk_size=1024,
                enrichment_tasks=None,
            )
            tasks_empty = await get_default_tasks(
                config={"ontology_config": {"ontology_resolver": None}},
                chunk_size=1024,
                enrichment_tasks=[],
            )

        assert [task.executable.__name__ for task in tasks_none] == _BASE_SEQUENCE
        assert [task.executable.__name__ for task in tasks_empty] == _BASE_SEQUENCE


class TestCognifyValidatesEnrichmentTasks:
    @pytest.mark.asyncio
    async def test_temporal_cognify_and_enrichment_tasks_raises(self):
        with pytest.raises(ValueError, match="temporal_cognify"):
            await cognify(temporal_cognify=True, enrichment_tasks=[Task(_fake_enrichment_task)])

    @pytest.mark.asyncio
    async def test_non_task_entry_raises(self):
        with pytest.raises(ValueError, match="Task"):
            await cognify(enrichment_tasks=["x"])

    @pytest.mark.asyncio
    async def test_remote_client_raises(self):
        with patch.object(_mod_serve_state, "get_remote_client", return_value=object()):
            with pytest.raises(ValueError, match="remote"):
                await cognify(enrichment_tasks=[Task(_fake_enrichment_task)])


class TestRememberForwardsEnrichmentTasks:
    async def _remember_task_sequence(self, enrichment_tasks):
        captured = {}

        def _fake_executor(run_in_background=False):
            async def _run(**executor_kwargs):
                tasks_arg = executor_kwargs["tasks"]
                resolved = (
                    tasks_arg(SimpleNamespace(system_metadata=None, extension="txt"))
                    if callable(tasks_arg)
                    else tasks_arg
                )
                captured["tasks"] = [task.executable.__name__ for task in resolved]
                return {}

            return _run

        config = CognifyConfig()
        with (
            patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}),
            patch.object(cognify_module, "get_cognify_config", return_value=config),
            patch.object(cognify_module, "get_pipeline_executor", _fake_executor),
            patch.object(_mod_migrations_startup, "run_migrations_and_block", new=AsyncMock()),
            patch.object(_mod_serve_state, "get_remote_client", return_value=None),
            patch.object(_mod_engine_setup, "setup", new=AsyncMock()),
            patch.object(_pkg_add, "add", new=AsyncMock()),
            patch.object(
                _mod_users_methods,
                "get_default_user",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                remember_module,
                "resolve_authorized_user_datasets",
                new=AsyncMock(return_value=(object(), [])),
            ),
        ):
            await remember(
                "Alice was born in 1985.",
                self_improvement=False,
                chunk_size=1024,
                config={"ontology_config": {"ontology_resolver": None}},
                enrichment_tasks=enrichment_tasks,
            )
        return captured.get("tasks")

    @pytest.mark.asyncio
    async def test_remember_reaches_executor_with_enrichment_task_last(self):
        sequence = await self._remember_task_sequence([Task(_fake_enrichment_task)])
        assert sequence == _BASE_SEQUENCE + ["_fake_enrichment_task"]


class TestRememberSurfacesEnrichmentTasks:
    def test_enrichment_tasks_in_remember_kwargs_annotations(self):
        assert "enrichment_tasks" in RememberKwargs.__annotations__

    def test_enrichment_tasks_in_cognify_only(self):
        assert "enrichment_tasks" in _COGNIFY_ONLY
