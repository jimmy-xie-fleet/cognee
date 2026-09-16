from pathlib import Path, PurePath
from typing import IO, Any, Union

from cognee.domains.legal.models import LegalKnowledgeGraph
from cognee.domains.legal.prompt import load_legal_extraction_prompt
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.graph import resolve_assertion_references

DEFAULT_LEGAL_CHUNK_SIZE = 512
LEGAL_FUZZY_CUTOFF = 0.9
LEGAL_ONTOLOGY_PATH = Path(__file__).parent / "ontology" / "legal.owl"

# What callers may hand to the resolver: everything RDFLibOntologyResolver takes, plus paths.
OntologyFile = Union[str, PurePath, IO, list[Union[str, PurePath, IO]], None]


def _as_ontology_file(ontology_file: OntologyFile) -> Any:
    """Coerce paths to the ``str`` form ``RDFLibOntologyResolver`` accepts.

    The resolver rejects ``pathlib.Path`` outright, so passing this module's own
    ``LEGAL_ONTOLOGY_PATH`` back in would raise. File-like objects are handed through
    untouched, because the resolver reads them as streams.
    """
    if ontology_file is None:
        return str(LEGAL_ONTOLOGY_PATH)
    if isinstance(ontology_file, (str, PurePath)):
        return str(ontology_file)
    if isinstance(ontology_file, (list, tuple)):
        return [
            str(entry) if isinstance(entry, (str, PurePath)) else entry for entry in ontology_file
        ]
    return ontology_file


def legal_ontology_resolver(
    ontology_file: OntologyFile = None, cutoff: float = LEGAL_FUZZY_CUTOFF
) -> RDFLibOntologyResolver:
    """Build an ``RDFLibOntologyResolver`` grounded in the legal OWL vocabulary.

    Args:
        ontology_file: Path (``str`` or ``Path``), open file object, or list of either.
            Defaults to the bundled ``legal.owl``.
        cutoff: Fuzzy-match cutoff passed to ``FuzzyMatchingStrategy``.
    """
    return RDFLibOntologyResolver(
        ontology_file=_as_ontology_file(ontology_file),
        matching_strategy=FuzzyMatchingStrategy(cutoff=cutoff),
    )


def legal_profile(
    *,
    ontology_mode: str = "annotate",
    chunk_size: int = DEFAULT_LEGAL_CHUNK_SIZE,
    ontology_file: OntologyFile = None,
    include_ontology: bool = True,
    resolve_references: bool = True,
) -> dict[str, Any]:
    """Build the kwargs bundle for ``cognee.remember()`` / ``cognee.cognify()``.

    Splat the result directly into either call, e.g. ``cognee.cognify(**legal_profile())``.

    ``resolve_references`` (default ``True``) ships the reference-resolver as an
    ``enrichment_tasks`` entry scoped to what this ingestion touched. That tail runs
    with ``allow_llm=False`` (decision D1: LLM calls happen only in the ``improve()``/
    memify pass, never on ingest): it resolves only exact-id and entity-name
    references, makes no LLM call, and reads no document text. A reference it cannot
    answer is left dangling for the agentic pass -- ``resolve_references_pipeline()``
    or ``improve()`` -- to trace against the full graph. Set ``resolve_references`` to
    ``False`` to opt out of the tail entirely (e.g. to drive resolution solely via the
    pass).
    """
    profile: dict[str, Any] = {
        "graph_model": LegalKnowledgeGraph,
        "custom_prompt": load_legal_extraction_prompt(),
        "chunk_size": chunk_size,
    }

    if include_ontology:
        profile["config"] = {
            "ontology_config": {
                "ontology_resolver": legal_ontology_resolver(ontology_file),
                "ontology_mode": ontology_mode,
            }
        }

    if resolve_references:
        profile["enrichment_tasks"] = [
            Task(resolve_assertion_references, scope="touched", allow_llm=False)
        ]

    return profile
