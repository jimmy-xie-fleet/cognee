"""The legal profile's extraction path, exercised on hand-written fixtures.

Every fixture pairs a synthetic passage (``fixtures/<stem>.txt``) with the
``LegalKnowledgeGraph`` a correct extraction of that passage should produce
(``expected_graphs.EXPECTED``). The expected graph is injected in place of the LLM
call, so what runs here is the real construction path: ontology canonicalization,
the assertion-aware constructor, and edge attachment. The expected graphs therefore
double as living examples of the behaviour ``prompts/legal_extraction_system.txt``
asks for.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.domains.legal import LegalKnowledgeGraph, legal_ontology_resolver
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.engine.utils import generate_node_name
from cognee.shared.data_models import Node
from cognee.tests.unit.domains.legal.expected_graphs import EXPECTED

egd_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The eight passages the profile is specified against, in the order they are written up.
FIXTURE_STEMS = (
    "complaint_p17_p18_warning",
    "answer_p17_denial",
    "answer_p2_partial",
    "appraisals_opposing",
    "lease_amendment",
    "deposition_qa",
    "ambiguous_names",
    "email_proposal",
)


@lru_cache(maxsize=1)
def _resolver():
    """One resolver for the whole module: parsing the OWL file per test is wasteful."""
    return legal_ontology_resolver()


def read_passage(stem: str) -> str:
    return (FIXTURES_DIR / f"{stem}.txt").read_text(encoding="utf-8")


def expected_graph(stem: str) -> LegalKnowledgeGraph:
    """A private copy: construction canonicalizes node types in place."""
    return EXPECTED[stem].model_copy(deep=True)


def assertion_nodes(graph: LegalKnowledgeGraph) -> list[Node]:
    return [node for node in graph.nodes if node.statement_type is not None]


def _make_chunk(stem: str, text: str) -> DocumentChunk:
    document = TextDocument(
        id=uuid4(),
        name=stem,
        raw_data_location=f"{stem}.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=[],
    )


async def run_fixture(stem: str) -> tuple[DocumentChunk, LegalKnowledgeGraph]:
    """Run one fixture through the real extraction task with the LLM call replaced.

    Returns the chunk (carrying the constructed data points) and the expected graph as
    construction left it, i.e. with ontology-canonical node types.
    """
    passage = read_passage(stem)
    graph = expected_graph(stem)
    chunk = _make_chunk(stem, passage)

    async def calculate_chunk_graphs(chunks, graph_model, custom_prompt, **kwargs):
        assert graph_model is LegalKnowledgeGraph
        return [graph]

    with patch.object(
        egd_module, "find_existing_edge_identities", new_callable=AsyncMock
    ) as mock_find_existing:
        mock_find_existing.return_value = set()
        await egd_module.extract_graph_from_data(
            [chunk],
            LegalKnowledgeGraph,
            config={
                "ontology_config": {
                    "ontology_resolver": _resolver(),
                    "ontology_mode": "annotate",
                }
            },
            calculate_chunk_graphs=calculate_chunk_graphs,
        )

    return chunk, graph


def data_points(chunk: DocumentChunk) -> list[Entity]:
    return [data_point for _edge, data_point in chunk.contains]


def assertions(chunk: DocumentChunk) -> list[Assertion]:
    return [point for point in data_points(chunk) if isinstance(point, Assertion)]


def entities(chunk: DocumentChunk) -> list[Entity]:
    return [point for point in data_points(chunk) if not isinstance(point, Assertion)]


def relations(data_point: Entity) -> list[tuple[str, str]]:
    return [(edge.relationship_type, target.name) for edge, target in data_point.relations]


def find_one(points: list[Entity], name: str) -> Entity:
    """The single data point carrying the (normalized) name of an expected node."""
    matches = [point for point in points if point.name == generate_node_name(name)]
    assert len(matches) == 1, f"expected exactly one {name!r}, found {len(matches)}"
    return matches[0]


def assertion_for(chunk: DocumentChunk, node: Node) -> Assertion:
    return find_one(assertions(chunk), node.name)


def node_by_id(graph: LegalKnowledgeGraph, node_id: Optional[str]) -> Optional[Node]:
    return next((node for node in graph.nodes if node.id == node_id), None)


def statement_types(chunk: DocumentChunk) -> list[str]:
    return [assertion.statement_type for assertion in assertions(chunk)]


# --------------------------------------------------------------------------------------
# Fixture hygiene
# --------------------------------------------------------------------------------------


def test_every_fixture_has_a_passage_and_an_expected_graph():
    assert set(EXPECTED) == set(FIXTURE_STEMS)
    assert {path.stem for path in FIXTURES_DIR.glob("*.txt")} == set(FIXTURE_STEMS)


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_quotes_are_verbatim(stem):
    """Guards fixture drift without running the pipeline: quotes must be copied text."""
    passage = read_passage(stem)
    quoted_nodes = [node for node in EXPECTED[stem].nodes if node.source_quote]

    assert quoted_nodes, f"{stem} quotes nothing"
    for node in quoted_nodes:
        assert node.source_quote in passage, f"{stem}/{node.id}: {node.source_quote!r}"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_conditions_are_verbatim(stem):
    passage = read_passage(stem)

    for node in EXPECTED[stem].nodes:
        for condition in node.conditions:
            assert condition in passage, f"{stem}/{node.id}: {condition!r}"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_expected_graph_is_internally_consistent(stem):
    """Ids unique, reference fields either resolve or are plain locator text."""
    graph = EXPECTED[stem]
    node_ids = [node.id for node in graph.nodes]

    assert len(set(node_ids)) == len(node_ids)
    assert len({generate_node_name(node.name) for node in graph.nodes}) == len(node_ids)
    for node in assertion_nodes(graph):
        assert node.type.lower() == node.statement_type.value
        if node.asserted_by is not None:
            assert node.asserted_by in node_ids, f"{stem}/{node.id} names no speaker node"
    for edge in graph.edges:
        assert edge.source_node_id in node_ids
        assert edge.target_node_id in node_ids


# --------------------------------------------------------------------------------------
# Construction invariants, for every fixture
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stem", FIXTURE_STEMS)
async def test_every_qualified_node_becomes_its_own_assertion(stem):
    chunk, graph = await run_fixture(stem)
    expected_assertions = assertion_nodes(graph)

    assert len(assertions(chunk)) == len(expected_assertions)
    assert len({assertion.id for assertion in assertions(chunk)}) == len(expected_assertions)
    assert statement_types(chunk) == [node.statement_type.value for node in expected_assertions], (
        "statement_type must survive construction unchanged"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stem", FIXTURE_STEMS)
async def test_every_source_quote_is_verified_against_the_chunk(stem):
    chunk, graph = await run_fixture(stem)
    quoted = [assertion for assertion in assertions(chunk) if assertion.source_quote]

    assert quoted
    for assertion in quoted:
        assert assertion.source_quote_verified is True, assertion.source_quote


@pytest.mark.asyncio
@pytest.mark.parametrize("stem", FIXTURE_STEMS)
async def test_every_node_type_is_ontology_grounded(stem):
    chunk, _graph = await run_fixture(stem)

    for data_point in data_points(chunk):
        assert data_point.is_a is not None, data_point.name
        assert data_point.is_a.ontology_valid is True, f"{data_point.name}: {data_point.is_a.name}"
        assert data_point.is_a.ontology_uri is not None


def test_only_contract_terms_go_without_a_named_speaker():
    """Pins which fixture deliberately leaves ``asserted_by`` null, and why.

    The lease and its amendment name no speaker for their own terms, and the prompt
    says to leave the field null rather than guess. Every other passage has one.
    """
    with_speakers = {
        stem
        for stem in FIXTURE_STEMS
        if any(node.asserted_by for node in assertion_nodes(EXPECTED[stem]))
    }

    assert with_speakers == set(FIXTURE_STEMS) - {"lease_amendment"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stem", FIXTURE_STEMS)
async def test_named_speakers_get_a_derived_asserted_by_edge(stem):
    chunk, graph = await run_fixture(stem)

    for node in [node for node in assertion_nodes(graph) if node.asserted_by]:
        speaker_node = node_by_id(graph, node.asserted_by)
        assertion = assertion_for(chunk, node)

        assert assertion.asserted_by == generate_node_name(speaker_node.name)
        assert ("asserted_by", generate_node_name(speaker_node.name)) in relations(assertion)


# --------------------------------------------------------------------------------------
# Per-fixture behaviour
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complaint_keeps_the_filing_date_apart_from_the_alleged_facts_date():
    chunk, _graph = await run_fixture("complaint_p17_p18_warning")
    pleaded = assertions(chunk)

    assert [assertion.statement_type for assertion in pleaded] == ["allegation", "allegation"]
    assert {assertion.report_date for assertion in pleaded} == {"2026-03-10"}
    assert [assertion.applicable_time for assertion in pleaded] == ["2025-02-20", "2025-03-03"]
    assert {assertion.asserted_by for assertion in pleaded} == {"amara okafor"}


@pytest.mark.asyncio
async def test_answer_denial_stays_a_denial_and_keeps_its_unresolved_locator():
    chunk, graph = await run_fixture("answer_p17_denial")
    denial = assertion_for(chunk, node_by_id(graph, "denial-p17"))
    audit_statement = assertion_for(chunk, node_by_id(graph, "statement-audit"))

    assert denial.statement_type == "denial"
    assert denial.polarity == "negative"
    # A locator that names no node is kept as text, and derives no edge.
    assert denial.responds_to == "Complaint ¶17"
    assert "responds_to" not in [name for name, _target in relations(denial)]

    assert audit_statement.statement_type == "statement"
    assert audit_statement.polarity == "positive"
    assert audit_statement.applicable_time == "2025-02-14"


@pytest.mark.asyncio
async def test_an_allegation_and_the_denial_answering_it_are_two_nodes():
    complaint_chunk, complaint_graph = await run_fixture("complaint_p17_p18_warning")
    answer_chunk, answer_graph = await run_fixture("answer_p17_denial")
    allegation = assertion_for(
        complaint_chunk, node_by_id(complaint_graph, "allegation-p17-report")
    )
    denial = assertion_for(answer_chunk, node_by_id(answer_graph, "denial-p17"))

    assert allegation.id != denial.id
    # And they would still be two nodes pleaded in one chunk by one speaker with one
    # wording: the statement type is part of the identity, not decoration on it.
    assert Assertion.id_for(allegation.name, "chunk", "allegation", "meridian", 1) != (
        Assertion.id_for(allegation.name, "chunk", "denial", "meridian", 1)
    )


@pytest.mark.asyncio
async def test_partial_answer_yields_one_distinct_node_per_admission_and_denial():
    chunk, _graph = await run_fixture("answer_p2_partial")
    pleaded = assertions(chunk)

    assert sorted(statement_types(chunk)) == ["admission", "admission"] + ["denial"] * 4
    assert len({assertion.id for assertion in pleaded}) == len(pleaded)
    assert {assertion.responds_to for assertion in pleaded} == {"Complaint ¶2"}
    assert {
        assertion.polarity for assertion in pleaded if assertion.statement_type == "denial"
    } == {"negative"}


@pytest.mark.asyncio
async def test_opposing_appraisals_are_two_attributed_opinions_not_successive_facts():
    chunk, graph = await run_fixture("appraisals_opposing")
    vance_opinion = assertion_for(chunk, node_by_id(graph, "opinion-vance-value"))
    baptiste_opinion = assertion_for(chunk, node_by_id(graph, "opinion-baptiste-value"))

    assert [vance_opinion.statement_type, baptiste_opinion.statement_type] == ["opinion", "opinion"]
    assert vance_opinion.attributed_to == "dolores vance"
    assert baptiste_opinion.attributed_to == "terrence baptiste"
    assert vance_opinion.attributed_to != baptiste_opinion.attributed_to
    # The date of the report is not the date the value is effective as of.
    for opinion in (vance_opinion, baptiste_opinion):
        assert opinion.applicable_time == "2024-07-01"
        assert opinion.report_date != opinion.applicable_time
    assert [vance_opinion.report_date, baptiste_opinion.report_date] == ["2025-09-08", "2025-10-02"]
    assert vance_opinion.precision == "exact"
    assert baptiste_opinion.precision == "approximate"
    # Each side's opinion is attributed to its own appraiser, by edge as well as by field.
    assert ("attributed_to", "dolores vance") in relations(vance_opinion)
    assert ("attributed_to", "terrence baptiste") in relations(baptiste_opinion)


@pytest.mark.asyncio
async def test_lease_amendment_supersedes_the_original_term_without_replacing_it():
    chunk, graph = await run_fixture("lease_amendment")
    original = assertion_for(chunk, node_by_id(graph, "term-original-rent"))
    amended = assertion_for(chunk, node_by_id(graph, "term-amended-rent"))

    assert ("supersedes", original.name) in relations(amended)
    # Both periods survive: the amendment does not overwrite the original term.
    assert original.applies_from == "2023-01-01"
    assert original.applies_to == "2027-12-31"
    assert amended.applies_from == "2024-07-01"
    assert "$4,000.00 per month" in original.source_quote
    assert "$4,400.00 per month" in amended.source_quote


@pytest.mark.asyncio
async def test_deposition_testimony_keeps_the_hedge_instead_of_a_flat_number():
    chunk, graph = await run_fixture("deposition_qa")
    distance = assertion_for(chunk, node_by_id(graph, "testimony-distance"))
    unseen = assertion_for(chunk, node_by_id(graph, "testimony-driver-glance"))

    assert distance.statement_type == "testimony"
    assert distance.precision == "approximate"
    assert distance.conditions == ["I'm not certain"]
    assert distance.report_date == "2025-05-22"
    # "I did not see it" is not "it did not happen".
    assert unseen.polarity == "negative"


@pytest.mark.asyncio
async def test_similar_names_stay_distinct_entities_and_speakers_resolve():
    chunk, graph = await run_fixture("ambiguous_names")
    municipal = [
        find_one(entities(chunk), "Clifton Planning Board"),
        find_one(entities(chunk), "Clifton Municipal Council"),
        find_one(entities(chunk), "City of Clifton"),
    ]

    assert len({entity.id for entity in municipal}) == 3
    assert [entity.is_a.name for entity in municipal] == [
        "planningboard",
        "municipalcouncil",
        "city",
    ]

    smith = find_one(entities(chunk), "Mr. Smith")
    smith_holdings = find_one(entities(chunk), "Smith Holdings LLC")
    assert smith.id != smith_holdings.id

    # The sidewalk promise was made by the person, not by the applicant company.
    promise = assertion_for(chunk, node_by_id(graph, "statement-smith-sidewalk"))
    assert promise.asserted_by == "mr. smith"
    assert ("asserted_by", "mr. smith") in relations(promise)
    assert ("asserted_by", "smith holdings llc") not in relations(promise)

    board_finding = assertion_for(chunk, node_by_id(graph, "finding-site-plan-recommendation"))
    assert board_finding.asserted_by == "clifton planning board"


@pytest.mark.asyncio
async def test_diligence_email_offer_is_a_conditional_proposal_not_a_term():
    chunk, graph = await run_fixture("email_proposal")
    offer = assertion_for(chunk, node_by_id(graph, "proposal-equity-offer"))

    assert offer.statement_type == "proposal"
    assert "term" not in statement_types(chunk)
    assert "subject to diligence" in offer.conditions
    assert offer.report_date == "2025-02-18"
    assert offer.applies_from is None and offer.applies_to is None
