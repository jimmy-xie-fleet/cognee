import inspect

from cognee.api.v1.remember.remember import _ADD_ONLY, _COGNIFY_ONLY, _SHARED, RememberKwargs
from cognee.modules.retrieval.utils import brute_force_triplet_search as bfts_module


def test_remember_kwargs_declares_config_and_temporal_cognify():
    assert "config" in RememberKwargs.__annotations__
    assert "temporal_cognify" in RememberKwargs.__annotations__


def test_remember_routes_config_to_cognify_only():
    assert "config" in _COGNIFY_ONLY
    assert "config" not in _ADD_ONLY | _SHARED


def test_default_collection_list_includes_assertion_name():
    source = inspect.getsource(bfts_module.brute_force_triplet_search)
    assert "Assertion_name" in source
    assert source.index("Entity_name") < source.index("Assertion_name")
