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
import re
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
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.shared.data_models import Node
from cognee.tasks.storage.add_data_points import _create_triplets_from_graph
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


# Words an assertion ``name`` may not carry: the proposition is affirmative, so negation
# belongs in ``polarity`` and the speech act in ``statement_type``.
FORBIDDEN_IN_ASSERTION_NAME = re.compile(
    r"\b(not|no|never|neither|nor|denies|denied|alleges|alleged|failed to)\b",
    re.IGNORECASE,
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
    """The single entity carrying the (normalized) name of an expected node.

    Entities are addressable by name; assertions are not — see ``assertion_for``.
    """
    matches = [point for point in points if point.name == generate_node_name(name)]
    assert len(matches) == 1, f"expected exactly one {name!r}, found {len(matches)}"
    return matches[0]


def node_by_id(graph: LegalKnowledgeGraph, node_id: Optional[str]) -> Optional[Node]:
    return next((node for node in graph.nodes if node.id == node_id), None)


def speaker_name(graph: LegalKnowledgeGraph, asserted_by: Optional[str]) -> Optional[str]:
    """What construction stores in ``asserted_by``: the speaker node's normalized name.

    A reference that names no node of the graph is kept as written, exactly as
    ``_resolve_display_name`` keeps it.
    """
    speaker_node = node_by_id(graph, asserted_by)
    return generate_node_name(speaker_node.name) if speaker_node is not None else asserted_by


def assertion_key(name: str, statement_type: str, asserted_by: Optional[str]) -> tuple:
    return (generate_node_name(name), statement_type, asserted_by)


def assertion_for(chunk: DocumentChunk, graph: LegalKnowledgeGraph, node_id: str) -> Assertion:
    """The constructed assertion for one expected node, keyed the way identity is keyed.

    Name alone is not a key and must never become one: an allegation and the denial
    answering it share one proposition by design and differ by speech act and speaker.
    """
    node = node_by_id(graph, node_id)
    assert node is not None, f"no expected node {node_id!r}"

    wanted = assertion_key(
        node.name, node.statement_type.value, speaker_name(graph, node.asserted_by)
    )
    matches = [
        assertion
        for assertion in assertions(chunk)
        if assertion_key(assertion.name, assertion.statement_type, assertion.asserted_by) == wanted
    ]
    assert len(matches) == 1, f"expected exactly one {wanted!r}, found {len(matches)}"
    return matches[0]


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
def test_assertion_names_are_affirmative_propositions(stem):
    """The convention the prompt states: the stance lives in ``polarity``, not in ``name``.

    A name carrying a negation would double-negate against ``polarity``, and a name
    carrying the speech act would duplicate ``statement_type`` — either way an allegation
    and the denial answering it would stop sharing one proposition.
    """
    for node in assertion_nodes(EXPECTED[stem]):
        forbidden = FORBIDDEN_IN_ASSERTION_NAME.findall(node.name)
        assert not forbidden, f"{stem}/{node.id}: {forbidden} in {node.name!r}"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_expected_graph_is_internally_consistent(stem):
    """Ids unique, reference fields either resolve or are plain locator text."""
    graph = EXPECTED[stem]
    node_ids = [node.id for node in graph.nodes]
    entity_names = [
        generate_node_name(node.name) for node in graph.nodes if node.statement_type is None
    ]
    # Assertions are deliberately left out: an allegation and its denial share one name.
    assertion_keys = [
        assertion_key(node.name, node.statement_type.value, node.asserted_by)
        for node in assertion_nodes(graph)
    ]

    assert len(set(node_ids)) == len(node_ids)
    assert len(set(entity_names)) == len(entity_names)
    assert len(set(assertion_keys)) == len(assertion_keys), (
        "two assertions with one identity would be occurrences of one statement"
    )
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
        assertion = assertion_for(chunk, graph, node.id)

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
    denial = assertion_for(chunk, graph, "denial-p17")
    audit_statement = assertion_for(chunk, graph, "statement-audit")

    assert denial.statement_type == "denial"
    assert denial.polarity == "negative"
    # The proposition is affirmative; the denial of it lives in polarity.
    assert denial.name == "the allegations of paragraph 17 of the complaint are true"
    # A locator that names no node is kept as text, and derives no edge.
    assert denial.responds_to == "Complaint ¶17"
    assert "responds_to" not in [name for name, _target in relations(denial)]

    assert audit_statement.statement_type == "statement"
    # "identified no falsified entries" is the affirmative proposition, negated.
    assert audit_statement.polarity == "negative"
    assert audit_statement.name.endswith("identified falsified entries")
    assert audit_statement.applicable_time == "2025-02-14"


@pytest.mark.asyncio
async def test_an_allegation_and_the_denial_answering_it_are_two_nodes():
    chunk, graph = await run_fixture("complaint_p17_p18_warning")
    allegation = assertion_for(chunk, graph, "allegation-p17-report")

    # Two nodes even when pleaded in one chunk by one speaker with one wording: the
    # statement type is part of the identity, not decoration on it. (The same pair on
    # the real path is ``test_recited_allegation_and_its_denial_...`` below.)
    assert Assertion.id_for(
        allegation.name, allegation.source_chunk_id, "allegation", "meridian", 1
    ) != (Assertion.id_for(allegation.name, allegation.source_chunk_id, "denial", "meridian", 1))


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
    vance_opinion = assertion_for(chunk, graph, "opinion-vance-value")
    baptiste_opinion = assertion_for(chunk, graph, "opinion-baptiste-value")

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
    original = assertion_for(chunk, graph, "term-original-rent")
    amended = assertion_for(chunk, graph, "term-amended-rent")

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
    distance = assertion_for(chunk, graph, "testimony-distance")
    unseen = assertion_for(chunk, graph, "testimony-driver-glance")

    assert distance.statement_type == "testimony"
    assert distance.precision == "approximate"
    assert distance.conditions == ["I'm not certain"]
    assert distance.report_date == "2025-05-22"
    # "I did not see it" is not "it did not happen": the proposition is what the witness
    # would have seen, phrased affirmatively, and his stance on it is negative.
    assert unseen.name == "ellerbee saw the driver look to his right before the bus started to move"
    assert unseen.polarity == "negative"


@pytest.mark.asyncio
async def test_denied_testimony_reaches_the_completion_context_as_a_denial():
    """The stance must survive all the way into the text retrieval embeds and shows.

    An empty edge description is not neutral: ``ensure_default_edge_properties`` fills it
    in from the endpoint labels, and for this assertion the affirmative proposition plus
    "asserted by" reads as the opposite of the testimony it came from.
    """
    chunk, graph = await run_fixture("deposition_qa")
    unseen = assertion_for(chunk, graph, "testimony-driver-glance")
    ellerbee = find_one(entities(chunk), "Raymond Ellerbee")

    nodes, edges = await get_graph_from_model(chunk)
    stored_edges = ensure_default_edge_properties(edges, nodes)
    asserted_by_edges = [
        edge
        for edge in stored_edges
        if (str(edge[0]), str(edge[1]), edge[2])
        == (str(unseen.id), str(ellerbee.id), "asserted_by")
    ]
    assert len(asserted_by_edges) == 1
    edge_text = asserted_by_edges[0][3]["edge_text"]

    assert "denies" in edge_text
    assert "Ellerbee says he did not see this" in edge_text
    # Not the synthesized fallback, which states the proposition as though it happened.
    assert edge_text != f"{unseen.name} asserted by {ellerbee.name}."
    assert not edge_text.startswith(unseen.name)

    triplets = _create_triplets_from_graph(nodes, stored_edges)
    triplet_texts = [
        triplet.text
        for triplet in triplets
        if triplet.from_node_id == str(unseen.id) and triplet.to_node_id == str(ellerbee.id)
    ]
    assert len(triplet_texts) == 1
    assert "denies" in triplet_texts[0]
    assert edge_text in triplet_texts[0]


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
    promise = assertion_for(chunk, graph, "statement-smith-sidewalk")
    assert promise.asserted_by == "mr. smith"
    assert ("asserted_by", "mr. smith") in relations(promise)
    assert ("asserted_by", "smith holdings llc") not in relations(promise)

    board_finding = assertion_for(chunk, graph, "finding-site-plan-recommendation")
    assert board_finding.asserted_by == "clifton planning board"


@pytest.mark.asyncio
async def test_diligence_email_offer_is_a_conditional_proposal_not_a_term():
    chunk, graph = await run_fixture("email_proposal")
    offer = assertion_for(chunk, graph, "proposal-equity-offer")

    assert offer.statement_type == "proposal"
    assert "term" not in statement_types(chunk)
    assert "subject to diligence" in offer.conditions
    assert offer.report_date == "2025-02-18"
    assert offer.applies_from is None and offer.applies_to is None


@pytest.mark.asyncio
async def test_recited_allegation_and_its_denial_share_one_name_and_stay_two_nodes():
    """The dispute signal: one proposition, two speech acts, opposite stances."""
    chunk, graph = await run_fixture("answer_p17_denial")
    allegation = assertion_for(chunk, graph, "allegation-p18-recited")
    denial = assertion_for(chunk, graph, "denial-p18")

    assert allegation.name == denial.name
    assert allegation.id != denial.id
    assert (allegation.statement_type, allegation.polarity) == ("allegation", "positive")
    assert (denial.statement_type, denial.polarity) == ("denial", "negative")
    assert (allegation.asserted_by, denial.asserted_by) == (
        "amara okafor",
        "meridian logistics, inc.",
    )
    # The denial names the allegation node, so the stored reference is that node's id.
    assert denial.responds_to == str(allegation.id)
    assert ("responds_to", allegation.name) in relations(denial)
