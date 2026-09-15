"""Defaults for the reference-resolution budget knobs on ``CognifyConfig``.

The five ``reference_*`` fields are what a pass's ``CallBudget`` and per-reference
iteration cap are built from, so their defaults are a contract, not an implementation
detail: changing one changes how much a single ``improve()`` pass can spend.
"""

from cognee.modules.cognify.config import CognifyConfig

REFERENCE_FIELDS = (
    "reference_llm_max_calls",
    "reference_tracer_max_iter",
    "reference_llm_confidence_threshold",
    "reference_infer_unstated",
    "reference_infer_confidence_threshold",
)


def test_reference_fields_have_the_documented_defaults():
    config = CognifyConfig()

    assert config.reference_llm_max_calls == 300
    assert config.reference_tracer_max_iter == 4
    assert config.reference_llm_confidence_threshold == 0.6
    # D2: implemented but opt-in until the spot-check says otherwise.
    assert config.reference_infer_unstated is False
    assert config.reference_infer_confidence_threshold == 0.75


def test_reference_fields_are_reported_by_to_dict():
    reported = CognifyConfig().to_dict()

    for field_name in REFERENCE_FIELDS:
        assert field_name in reported
        assert reported[field_name] == getattr(CognifyConfig(), field_name)


def test_reference_fields_are_env_overridable(monkeypatch):
    # pydantic-settings maps an upper-cased field name to its env var, the same way
    # CONTRADICTION_DETECTION reaches contradiction_detection.
    monkeypatch.setenv("REFERENCE_TRACER_MAX_ITER", "7")
    monkeypatch.setenv("REFERENCE_INFER_UNSTATED", "true")

    config = CognifyConfig()

    assert config.reference_tracer_max_iter == 7
    assert config.reference_infer_unstated is True
