"""Assertion-aware recall against the real default stack, offline.

The unit tests for stance rendering, pair expansion and the hybrid statements lane all
run against fakes. This is the one test that puts an allegation, the denial answering
it and the speaker through a real embedded Ladybug graph and a real LanceDB vector
store, then reads the context each retriever renders. It pins the chain the fakes
cannot: the graph-projection whitelist actually carrying the stance fields, the
``get_neighborhood`` call actually returning the pair, and the renderers agreeing.

Offline by construction: no LLM is on the ``only_context`` path, and embeddings are
replaced by a deterministic hash of the text (unit-norm, so identical text is cosine 1
and unrelated text is near 0). Never all-zeros: LanceDB scores by cosine distance and
zero vectors tie every row.
"""

import hashlib
import math
import pathlib
import random
import re
from unittest.mock import patch

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.graph.utils.reference_resolution import stance_edge_text
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.tasks.storage.add_data_points import add_data_points

PROPOSITION = "Adams breached the lease"
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


async def _fake_embed_text(self, texts):
    vectors = []
    for text in texts:
        rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
        vector = [rng.gauss(0.0, 1.0) for _ in range(self.dimensions)]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


@pytest_asyncio.fixture
async def legal_graph(tmp_path, monkeypatch):
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    monkeypatch.setenv("AUTO_FEEDBACK", "false")
    root = pathlib.Path(tmp_path)

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    def reset_engines():
        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
        graph_db_config.set(None)
        vector_db_config.set(None)

    reset_engines()
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    with patch.object(LiteLLMEmbeddingEngine, "embed_text", _fake_embed_text):
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await engine_setup()

        plaintiff = Entity(name="Plaintiff", description="Adams Family Properties, LLC.")
        defendants = Entity(name="Defendants", description="The City of Clifton and its boards.")
        allegation = Assertion(
            name=PROPOSITION,
            description="Complaint paragraph 17 alleges the breach.",
            statement_type="allegation",
            polarity="positive",
            asserted_by="Plaintiff",
            source_quote="Adams breached the lease by failing to maintain the roof.",
            source_quote_verified=True,
        )
        denial = Assertion(
            name=PROPOSITION,
            description="Answer paragraph 17 denies the breach allegation.",
            statement_type="denial",
            polarity="negative",
            asserted_by="Defendants",
            responds_to=str(allegation.id),
            responds_to_text="the Complaint ¶17",
            source_quote="Denied.",
            source_quote_verified=True,
        )
        # ``responds_to`` / ``asserted_by`` are plain string fields on a hand-built
        # Assertion, so the edges have to be supplied. ``edge_text`` is set the way the
        # resolver sets it; left to the default it would be built from the endpoints'
        # names -- the denial's affirmative proposition -- which is the very bug.
        denial_props = denial.model_dump()
        custom_edges = [
            (
                str(denial.id),
                str(allegation.id),
                "responds_to",
                {
                    "edge_text": stance_edge_text(denial_props, "responds_to", allegation.name),
                    "resolution_confidence": 0.9,
                    "resolution_strategy": "llm_trace",
                    "resolved_by": "reference_resolver",
                },
            ),
            (
                str(denial.id),
                str(defendants.id),
                "asserted_by",
                {"edge_text": stance_edge_text(denial_props, "asserted_by", "Defendants")},
            ),
            (
                str(allegation.id),
                str(plaintiff.id),
                "asserted_by",
                {
                    "edge_text": stance_edge_text(
                        allegation.model_dump(), "asserted_by", "Plaintiff"
                    )
                },
            ),
        ]
        await add_data_points(
            [plaintiff, defendants, allegation, denial], custom_edges=custom_edges
        )

        yield {"allegation": allegation, "denial": denial}

        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        except Exception:  # noqa: BLE001 - teardown must not mask the test
            pass
    reset_engines()


def _pair_lines(context: str) -> list[str]:
    return [line for line in context.splitlines() if line.strip().startswith("↳")]


@pytest.mark.asyncio
async def test_hybrid_context_renders_the_denial_with_its_stance_and_its_pair(legal_graph):
    retriever = HybridRetriever(
        chunks_top_k=10, entities_top_k=10, facts_top_k=10, statements_top_k=20
    )

    with patch.object(LiteLLMEmbeddingEngine, "embed_text", _fake_embed_text):
        retrieved = await retriever.get_retrieved_objects(query=PROPOSITION)
        context = await retriever.get_context_from_objects(
            query=PROPOSITION, retrieved_objects=retrieved
        )

    assert "## Relevant statements" in context
    assert f"### [denial] Defendants denies that {PROPOSITION}" in context
    assert f"Defendants denies that {PROPOSITION}." in context
    assert 'Quote: "Denied." (verified)' in context
    assert "Responds to: the Complaint ¶17" in context
    assert f"### [allegation] Plaintiff affirms that {PROPOSITION}" in context
    assert f"Plaintiff affirms that {PROPOSITION}." in context

    # The pair came back from the real graph, both ways round.
    assert (
        f"  ↳ responds to: [allegation] Plaintiff affirms that {PROPOSITION} (confidence 0.9, llm_trace)"
        in context
    )
    assert f"  ↳ answered by: [denial] Defendants denies that {PROPOSITION}" in context
    assert "  ↳ speaker: Defendants" in context
    assert "  ↳ speaker: Plaintiff" in context

    # Nothing leaks: no projected None, no bare id where a label belongs.
    assert "None" not in context
    assert not any(UUID_PATTERN.search(line) for line in _pair_lines(context))


@pytest.mark.asyncio
async def test_graph_completion_context_renders_the_denial_and_pulls_in_the_allegation(
    legal_graph,
):
    retriever = GraphCompletionRetriever(top_k=10)

    with patch.object(LiteLLMEmbeddingEngine, "embed_text", _fake_embed_text):
        edges = await retriever.get_retrieved_objects(query=PROPOSITION)
        context = await retriever.get_context_from_objects(
            query=PROPOSITION, retrieved_objects=edges
        )

    assert f"Node: [denial] Defendants denies that {PROPOSITION}" in context
    assert f"Defendants denies that {PROPOSITION}." in context
    assert "Responds to: the Complaint ¶17" in context
    assert f"Node: [allegation] Plaintiff affirms that {PROPOSITION}" in context

    responds_to = [line for line in context.splitlines() if "--[responds_to]-->" in line]
    assert responds_to, context
    (line,) = responds_to
    assert line.startswith(f"[denial] Defendants denies that {PROPOSITION} ")
    assert f"--> [allegation] Plaintiff affirms that {PROPOSITION}" in line
    assert "responds to" in line  # the resolver's edge_text, in parentheses
    assert line.endswith("[confidence 0.9, llm_trace]")

    assert "None" not in context
    assert not UUID_PATTERN.search(context)
