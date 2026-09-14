"""Opt-in live integration test for the legal extraction profile.

Exercises the real `remember()` -> `cognify()` path against a live LLM (no LLM
calls are mocked, unlike the unit fixtures in
`cognee/tests/unit/domains/legal/test_extraction_fixtures.py`). Skipped unless
both `LLM_API_KEY` and `COGNEE_LIVE_LEGAL_TESTS=1` are set, since it costs a
real LLM call and needs a working provider key.

Run for real from the repo root::

    COGNEE_LIVE_LEGAL_TESTS=1 LLM_API_KEY="$OPENAI_API_KEY" \\
        .venv/bin/python -m pytest cognee/tests/integration/domains -q -p no:cacheprovider

By default it ingests the ``answer_p17_denial`` unit fixture; set
``COUNSELDESK_DOC`` to a file path to ingest a different document instead.
Uses cognee's default local databases (no root-directory overrides), and
always cleans up the dataset it created in a ``finally`` block.
"""

import os
from pathlib import Path

import pytest

import cognee
from cognee.domains.legal import legal_profile
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.engine.models.Assertion import STATEMENT_TYPE_NAMES
from cognee.shared.logging_utils import get_logger

logger = get_logger("tests.legal_profile_live")

pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_API_KEY") or os.getenv("COGNEE_LIVE_LEGAL_TESTS") != "1",
    reason=(
        "Opt-in live test: requires LLM_API_KEY and COGNEE_LIVE_LEGAL_TESTS=1 "
        "(makes a real LLM call)"
    ),
)

DATASET_NAME = "legal_live_test"

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "unit" / "domains" / "legal" / "fixtures"
DEFAULT_FIXTURE = FIXTURES_DIR / "answer_p17_denial.txt"


def _remember_input() -> str:
    """The `$COUNSELDESK_DOC` file path if set, else the default fixture's text."""
    counseldesk_doc = os.getenv("COUNSELDESK_DOC")
    if counseldesk_doc:
        return counseldesk_doc
    return DEFAULT_FIXTURE.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_legal_profile_extracts_grounded_assertions():
    try:
        await cognee.remember(
            _remember_input(),
            dataset_name=DATASET_NAME,
            self_improvement=False,
            **legal_profile(),
        )

        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()

        assertion_nodes = [
            properties for _node_id, properties in nodes if properties.get("type") == "Assertion"
        ]
        assert assertion_nodes, "expected at least one Assertion node in the graph"

        for properties in assertion_nodes:
            assert properties.get("statement_type") in STATEMENT_TYPE_NAMES, (
                f"unexpected statement_type: {properties.get('statement_type')!r}"
            )

        negative_assertions = [
            properties for properties in assertion_nodes if properties.get("polarity") == "negative"
        ]
        assert negative_assertions, "expected at least one negative-polarity assertion"

        verified_assertions = [
            properties
            for properties in assertion_nodes
            if properties.get("source_quote_verified") is True
        ]
        assert len(verified_assertions) / len(assertion_nodes) >= 0.5, (
            f"only {len(verified_assertions)}/{len(assertion_nodes)} assertions had a "
            "verified source_quote"
        )

        asserted_by_edges = [edge for edge in edges if edge[2] == "asserted_by"]
        assert asserted_by_edges, "expected at least one asserted_by edge"
    finally:
        # Cleanup must never replace the failure that brought us here: a run that failed
        # before the dataset existed would otherwise report the forget() error instead.
        try:
            await cognee.forget(dataset=DATASET_NAME)
        except Exception as cleanup_error:  # noqa: BLE001 - reported, never raised
            logger.warning("cleanup of dataset %s failed: %s", DATASET_NAME, cleanup_error)
