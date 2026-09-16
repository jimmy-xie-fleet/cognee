"""RetrievalConfig: env-backed recall lane budgets (Task 7).

Effectiveness over efficiency (user decision) -- the legal graph starved hybrid
recall of context because lane sizes were hard-coded per request and
``retriever_specific_config`` is not reachable over HTTP, so the UI could never
raise them. These tests only exercise the settings object: defaults, env
overrides, and the ``to_dict`` shape. No LLM, vector store, graph backend, or
network is touched.
"""

import pytest
from pydantic import ValidationError

from cognee.modules.retrieval.config import RetrievalConfig, get_retrieval_config


@pytest.fixture(autouse=True)
def clear_retrieval_config_cache():
    get_retrieval_config.cache_clear()
    yield
    get_retrieval_config.cache_clear()


def test_defaults_are_the_raised_lane_budgets():
    config = RetrievalConfig()

    assert config.hybrid_chunks_top_k == 30
    assert config.hybrid_entities_top_k == 30
    assert config.hybrid_facts_top_k == 30
    assert config.hybrid_statements_top_k == 20
    assert config.hybrid_max_edges_per_entity == 20
    assert config.disputes_top_k == 50
    assert config.graph_completion_pair_expansion is True


def test_get_retrieval_config_is_cached():
    first = get_retrieval_config()
    second = get_retrieval_config()

    assert first is second


@pytest.mark.parametrize(
    "env_name,field_name,env_value,expected",
    [
        ("HYBRID_CHUNKS_TOP_K", "hybrid_chunks_top_k", "45", 45),
        ("HYBRID_ENTITIES_TOP_K", "hybrid_entities_top_k", "50", 50),
        ("HYBRID_FACTS_TOP_K", "hybrid_facts_top_k", "12", 12),
        ("HYBRID_STATEMENTS_TOP_K", "hybrid_statements_top_k", "5", 5),
        ("HYBRID_MAX_EDGES_PER_ENTITY", "hybrid_max_edges_per_entity", "3", 3),
        ("DISPUTES_TOP_K", "disputes_top_k", "7", 7),
    ],
)
def test_env_override_wins_over_the_default(monkeypatch, env_name, field_name, env_value, expected):
    monkeypatch.setenv(env_name, env_value)
    get_retrieval_config.cache_clear()

    config = get_retrieval_config()

    assert getattr(config, field_name) == expected


def test_pair_expansion_flag_is_overridable_by_env(monkeypatch):
    monkeypatch.setenv("GRAPH_COMPLETION_PAIR_EXPANSION", "false")
    get_retrieval_config.cache_clear()

    config = get_retrieval_config()

    assert config.graph_completion_pair_expansion is False


@pytest.mark.parametrize(
    "env_name",
    [
        "HYBRID_CHUNKS_TOP_K",
        "HYBRID_ENTITIES_TOP_K",
        "HYBRID_FACTS_TOP_K",
        "HYBRID_STATEMENTS_TOP_K",
        "HYBRID_MAX_EDGES_PER_ENTITY",
        "DISPUTES_TOP_K",
    ],
)
@pytest.mark.parametrize("bad_value", ["0", "-5"])
def test_a_non_positive_budget_is_rejected_at_startup(monkeypatch, env_name, bad_value):
    """A request top_k of 0 is a 422; an env budget of 0 must not slip through silently."""
    monkeypatch.setenv(env_name, bad_value)

    with pytest.raises(ValidationError) as excinfo:
        RetrievalConfig()

    assert env_name.lower() in str(excinfo.value)


def test_to_dict_reports_every_field():
    config = RetrievalConfig()

    assert config.to_dict() == {
        "hybrid_chunks_top_k": 30,
        "hybrid_entities_top_k": 30,
        "hybrid_facts_top_k": 30,
        "hybrid_statements_top_k": 20,
        "hybrid_max_edges_per_entity": 20,
        "disputes_top_k": 50,
        "graph_completion_pair_expansion": True,
    }
