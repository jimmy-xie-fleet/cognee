"""
Unit Tests: assertion stance in recall context

Two halves of the same defect. The projection whitelist in ``get_memory_fragment`` used to
drop every assertion property except ``name``/``description``, and the hybrid renderer
printed the affirmative ``name`` on its own -- so a denial reached the model as the fact it
denies. These tests pin that the stance properties survive projection and that the hybrid
entity block and its edge bullets show them, without changing anything for a plain entity.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid import entities as entities_module
from cognee.modules.retrieval.hybrid.entities import (
    _entity_from_result,
    build_entities,
    format_entities,
)
from cognee.modules.retrieval.utils.brute_force_triplet_search import get_memory_fragment

ASSERTION_PAYLOAD = {
    "id": "denial-1",
    "name": "Adams breached the lease",
    "statement_type": "denial",
    "polarity": "negative",
    "asserted_by": "Defendants",
    "source_quote": "Defendants deny each and every allegation of paragraph 17.",
    "source_quote_verified": True,
}

CONTEXT_FIELDS = [
    "statement_type",
    "polarity",
    "asserted_by",
    "source_quote",
    "source_quote_verified",
]


def _hit(payload: dict):
    hit = MagicMock()
    hit.id = payload["id"]
    hit.payload = payload
    return hit


def _graph(nodes, edges):
    graph = MagicMock()
    graph.get_neighborhood = AsyncMock(return_value=(nodes, edges))
    return graph


# ---------------------------------------------------------------------------------------
# Hybrid entity block
# ---------------------------------------------------------------------------------------


def test_entity_from_result_keeps_the_plain_entity_shape():
    entity = _entity_from_result(_hit({"id": "entity-1", "name": "Alice"}))

    assert entity == {
        "id": "entity-1",
        "name": "Alice",
        "description": None,
        "type": None,
        "edges": [],
    }


def test_entity_from_result_carries_the_assertion_stance_fields():
    entity = _entity_from_result(_hit(ASSERTION_PAYLOAD))

    assert {field: entity[field] for field in CONTEXT_FIELDS} == {
        field: ASSERTION_PAYLOAD[field] for field in CONTEXT_FIELDS
    }


def test_declared_context_fields_do_not_overwrite_the_computed_entity_keys():
    """A context field that collides with a computed key must lose to the computed value.

    No subclass declares one today; the guard is what keeps the next one from silently
    replacing the name the renderer normalized, or the edges the graph lane filled in.
    """
    colliding_payload = {
        **ASSERTION_PAYLOAD,
        "name": "  Adams breached the lease  ",
        "edges": ["raw"],
    }

    with patch.object(
        entities_module,
        "context_fields_for_datapoints",
        return_value=["name", "edges", "statement_type"],
    ):
        entity = _entity_from_result(_hit(colliding_payload))

    assert entity["name"] == "Adams breached the lease"
    assert entity["edges"] == []
    assert entity["statement_type"] == "denial"


def test_entity_block_of_an_assertion_renders_the_stance():
    block = format_entities([_entity_from_result(_hit(ASSERTION_PAYLOAD))])

    assert block == (
        "## Relevant entities\n"
        "### [denial by Defendants; stance: negative] Adams breached the lease\n"
        "Defendants denies that Adams breached the lease.\n"
        'Quote: "Defendants deny each and every allegation of paragraph 17." (verified)'
    )


def test_entity_block_of_a_plain_entity_is_unchanged():
    block = format_entities(
        [{"id": "entity-1", "name": "Alice", "description": "Alice works at Acme.", "edges": []}]
    )

    assert block == "## Relevant entities\n### Alice\nAlice works at Acme."


@pytest.mark.asyncio
async def test_edge_bullets_label_an_assertion_neighbour_with_its_stance():
    nodes = [
        ("acme-id", {"name": "Acme"}),
        ("denial-1", {key: value for key, value in ASSERTION_PAYLOAD.items() if key != "id"}),
    ]
    edges = [("denial-1", "acme-id", "asserted_by", {})]

    entities, _reachable = await build_entities(
        _graph(nodes, edges), [_hit(ASSERTION_PAYLOAD)], max_edges_per_entity=10
    )

    assert entities[0]["edges"][0]["text"] == (
        "[denial/negative] Adams breached the lease -- asserted_by -- Acme"
    )


@pytest.mark.asyncio
async def test_edge_bullets_of_plain_neighbours_are_unchanged():
    nodes = [("alice-id", {"name": "Alice"}), ("acme-id", {"name": "Acme"})]
    edges = [("alice-id", "acme-id", "works_at", {})]

    entities, _reachable = await build_entities(
        _graph(nodes, edges), [_hit({"id": "alice-id", "name": "Alice"})], max_edges_per_entity=10
    )

    assert entities[0]["edges"][0] == {
        "text": "Alice -- works_at -- Acme",
        "source": "Alice",
        "target": "Acme",
        "source_id": "alice-id",
        "relationship": "works_at",
        "target_id": "acme-id",
        "edge_type_id": str(EdgeType.id_for("works_at")),
        "edge_object_id": None,
    }


# ---------------------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------------------


def _projection_kwargs(fragment) -> dict:
    return fragment.project_graph_from_db.call_args.kwargs


async def _project(**kwargs) -> dict:
    fragment = MagicMock(spec=CogneeGraph)
    fragment.project_graph_from_db = AsyncMock()

    with (
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine",
            return_value=AsyncMock(),
        ),
        patch(
            "cognee.modules.retrieval.utils.brute_force_triplet_search.CogneeGraph",
            return_value=fragment,
        ),
    ):
        await get_memory_fragment(**kwargs)

    return _projection_kwargs(fragment)


@pytest.mark.asyncio
async def test_default_projection_includes_the_declared_context_fields():
    kwargs = await _project()

    projected = kwargs["node_properties_to_project"]
    assert set(CONTEXT_FIELDS).issubset(set(projected))
    # The properties the renderers already relied on stay, and nothing is duplicated.
    assert {"id", "description", "name", "type", "text", "importance_weight"}.issubset(
        set(projected)
    )
    assert len(projected) == len(set(projected))


@pytest.mark.asyncio
async def test_edge_projection_includes_the_resolution_properties():
    kwargs = await _project()

    assert {"resolution_confidence", "resolution_strategy"}.issubset(
        set(kwargs["edge_properties_to_project"])
    )


@pytest.mark.asyncio
async def test_explicit_projection_requests_are_left_alone():
    kwargs = await _project(properties_to_project=["id", "text", "type", "is_root"])

    assert kwargs["node_properties_to_project"] == ["id", "text", "type", "is_root"]
