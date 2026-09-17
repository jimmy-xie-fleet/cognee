import difflib
import importlib
import inspect
import itertools

from cognee.domains.legal import (
    LEGAL_FUZZY_CUTOFF,
    LEGAL_ONTOLOGY_PATH,
    LegalExtractionOptions,
    LegalKnowledgeGraph,
    legal_ontology_resolver,
    legal_profile,
)
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.tasks.graph import resolve_assertion_references

# Every class local name from the legal OWL vocabulary hierarchy, lowercased the way
# RDFLibOntologyResolver._uri_to_key would key them.
EXPECTED_CLASS_NAMES = [
    "Agent",
    "Person",
    "Organization",
    "Company",
    "Insurer",
    "Nonprofit",
    "GovernmentAgency",
    "Court",
    "MunicipalBody",
    "City",
    "PlanningBoard",
    "MunicipalCouncil",
    "Asset",
    "Property",
    "Parcel",
    "Suite",
    "MonetaryAmount",
    "Date",
    "Event",
    "Claim",
    "Defense",
    "Statute",
    "Award",
    "Document",
    "Pleading",
    "Complaint",
    "Answer",
    "Deposition",
    "Transcript",
    "Minutes",
    "Email",
    "Ledger",
    "Notice",
    "Order",
    "Report",
    "Appraisal",
    "PoliceReport",
    "Contract",
    "Lease",
    "Amendment",
    "InsurancePolicy",
    "Assertion",
    "PleadingStatement",
    "Allegation",
    "Admission",
    "Denial",
    "Testimony",
    "Opinion",
    "Finding",
    "Record",
    "Term",
    "Proposal",
    "Statement",
]

EXPECTED_CLASS_KEYS = {name.lower() for name in EXPECTED_CLASS_NAMES}


class TestLegalProfile:
    def test_default_profile_shape(self):
        profile = legal_profile()

        assert set(profile) == {
            "graph_model",
            "custom_prompt",
            "calculate_chunk_graphs",
            "config",
            "enrichment_tasks",
        }
        assert profile["graph_model"] is LegalKnowledgeGraph
        assert isinstance(profile["custom_prompt"], str) and profile["custom_prompt"]
        assert callable(profile["calculate_chunk_graphs"])
        assert profile["calculate_chunk_graphs"].options == LegalExtractionOptions(
            two_pass=True, drop_low_salience=True
        )

        ontology_config = profile["config"]["ontology_config"]
        assert ontology_config["ontology_mode"] == "annotate"

        resolver = ontology_config["ontology_resolver"]
        assert isinstance(resolver, RDFLibOntologyResolver)
        assert resolver.matching_strategy.cutoff == LEGAL_FUZZY_CUTOFF == 0.9

        enrichment_tasks = profile["enrichment_tasks"]
        assert len(enrichment_tasks) == 1
        resolve_task = enrichment_tasks[0]
        assert resolve_task.executable is resolve_assertion_references
        assert resolve_task.default_params["kwargs"]["scope"] == "touched"
        # The discriminating behavioural guarantee -- a populated graph, allow_llm=False,
        # tracer never called, no document text read -- is
        # test_the_tail_never_calls_the_llm_and_never_reads_a_document in
        # cognee/tests/unit/tasks/graph/test_resolve_assertion_references.py; this
        # assertion only pins that the shipped profile carries the kwarg.
        assert resolve_task.default_params["kwargs"]["allow_llm"] is False

    def test_include_ontology_false_omits_config(self):
        profile = legal_profile(include_ontology=False)

        assert set(profile) == {
            "graph_model",
            "custom_prompt",
            "calculate_chunk_graphs",
            "enrichment_tasks",
        }
        assert "config" not in profile

    def test_resolve_references_false_omits_enrichment_tasks(self):
        profile = legal_profile(resolve_references=False)

        assert "enrichment_tasks" not in profile

    def test_ontology_mode_passthrough(self):
        profile = legal_profile(ontology_mode="strict")

        assert profile["config"]["ontology_config"]["ontology_mode"] == "strict"

    def test_chunk_size_passthrough(self):
        profile = legal_profile(chunk_size=256)

        assert profile["chunk_size"] == 256

    def test_default_chunk_size_is_cognees(self):
        """The profile used to pin 512 tokens; the eval measured that graph 6 to 20 points
        behind plain extraction with most misses never retrieved. Unset means cognee's own."""
        assert "chunk_size" not in legal_profile()

    def test_two_pass_and_drop_flags_reach_the_callable(self):
        profile = legal_profile(two_pass=False, drop_low_salience=False)

        assert profile["calculate_chunk_graphs"].options == LegalExtractionOptions(
            two_pass=False, drop_low_salience=False
        )

    def test_path_ontology_file_is_accepted(self):
        # LEGAL_ONTOLOGY_PATH is a Path, so handing the module's own constant back to
        # the profile must work; RDFLibOntologyResolver only understands str paths.
        profile = legal_profile(ontology_file=LEGAL_ONTOLOGY_PATH)

        resolver = profile["config"]["ontology_config"]["ontology_resolver"]
        assert isinstance(resolver, RDFLibOntologyResolver)
        assert EXPECTED_CLASS_KEYS <= set(resolver.lookup["classes"])


class TestCogneeContract:
    def test_profile_keys_are_valid_cognify_kwargs(self):
        cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
        signature = inspect.signature(cognify_module.cognify)
        cognify_params = set(signature.parameters)
        # ``calculate_chunk_graphs`` rides cognify's ``**kwargs`` down to
        # ``extract_graph_from_data``; everything else is a named parameter.
        assert any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        assert set(legal_profile()) - {"calculate_chunk_graphs"} <= cognify_params

    def test_graph_model_and_config_are_cognify_only_in_remember(self):
        remember_module = importlib.import_module("cognee.api.v1.remember.remember")
        assert {
            "graph_model",
            "config",
            "enrichment_tasks",
            "calculate_chunk_graphs",
        } <= remember_module._COGNIFY_ONLY


class TestRealResolver:
    def test_lookup_contains_all_classes(self):
        resolver = legal_ontology_resolver()

        assert EXPECTED_CLASS_KEYS <= set(resolver.lookup["classes"])

    def test_lookup_contains_individuals(self):
        resolver = legal_ontology_resolver()

        assert "eeoc" in resolver.lookup["individuals"]
        assert "title_vii" in resolver.lookup["individuals"]

    def test_get_subgraph_denial_class_hierarchy(self):
        resolver = legal_ontology_resolver()

        _, edges, _ = resolver.get_subgraph("denial", node_type="classes")

        assert ("denial", "is_a", "pleadingstatement") in edges
        assert ("pleadingstatement", "is_a", "assertion") in edges

    def test_get_subgraph_eeoc_enforces_title_vii(self):
        resolver = legal_ontology_resolver()

        _, edges, _ = resolver.get_subgraph("eeoc", node_type="individuals")

        assert ("eeoc", "enforces", "title_vii") in edges

    def test_ontology_path_points_at_real_file(self):
        assert LEGAL_ONTOLOGY_PATH.exists()
        assert LEGAL_ONTOLOGY_PATH.name == "legal.owl"

    def test_resolver_accepts_a_path_and_a_list_of_paths(self):
        assert EXPECTED_CLASS_KEYS <= set(
            legal_ontology_resolver(LEGAL_ONTOLOGY_PATH).lookup["classes"]
        )
        assert EXPECTED_CLASS_KEYS <= set(
            legal_ontology_resolver([LEGAL_ONTOLOGY_PATH]).lookup["classes"]
        )


class TestCollisionGuards:
    def test_no_near_duplicate_class_keys(self):
        resolver = legal_ontology_resolver()
        keys = list(resolver.lookup["classes"])

        for a, b in itertools.combinations(keys, 2):
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            assert ratio < 0.9, f"class keys too similar: {a!r} vs {b!r} (ratio={ratio})"

    def test_unrelated_words_do_not_match_classes(self):
        resolver = legal_ontology_resolver()

        for word in (
            "settlement",
            "funding",
            "recording",
            "compliant",
            "police",
            "release",
            "insured",
            "appraiser",
        ):
            assert resolver.find_closest_match(word, "classes") is None

    def test_exact_and_case_variant_matches(self):
        resolver = legal_ontology_resolver()

        assert resolver.find_closest_match("denial", "classes") == "denial"
        assert resolver.find_closest_match("Allegation", "classes") == "allegation"
        # "Planning Board" normalizes to "planning_board", which is fuzzy-close enough
        # (ratio ~0.96 at cutoff 0.9) to the camel-cased class key "planningboard".
        assert resolver.find_closest_match("Planning Board", "classes") == "planningboard"
