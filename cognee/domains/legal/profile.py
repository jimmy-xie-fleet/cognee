from pathlib import Path, PurePath
from typing import IO, Any, Optional, Union

from cognee.domains.legal.extraction import legal_chunk_graphs
from cognee.domains.legal.models import LegalKnowledgeGraph
from cognee.domains.legal.prompt import load_legal_extraction_prompt
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.graph import resolve_assertion_references

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
    chunk_size: Optional[int] = None,
    ontology_file: OntologyFile = None,
    include_ontology: bool = True,
    resolve_references: bool = True,
    two_pass: bool = True,
    drop_low_salience: bool = True,
) -> dict[str, Any]:
    """Build the kwargs bundle for ``cognee.remember()`` / ``cognee.cognify()``.

    Splat the result directly into either call, e.g. ``cognee.cognify(**legal_profile())``.

    ``chunk_size`` defaults to cognee's own (derived from the model) rather than a
    profile-specific value: the profile used to pin 512 tokens, and a repeated recall
    eval measured that graph 6 to 20 coverage points behind plain extraction with 57 to
    82 percent of its misses never retrieved; the same profile at the default chunk size
    was within noise on hybrid recall. Pass a value only to experiment.

    ``two_pass`` (default ``True``) extracts each chunk twice -- once with cognee's
    default prompt and ``KnowledgeGraph`` (byte-identical to plain ingestion) and once
    with the legal prompt -- and merges the two graphs, so the legal graph is a superset
    of the plain one. It costs two LLM calls per chunk; ``cognify(dry_run=True)`` counts
    one. ``drop_low_salience`` (default ``True``) removes assertions the model marked
    ``low`` (boilerplate) before construction; retained assertions carry an
    ``importance_weight`` from their salience either way. Both are carried by
    ``calculate_chunk_graphs``, the per-chunk extraction hook ``cognify()`` accepts.

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
        "calculate_chunk_graphs": legal_chunk_graphs(
            two_pass=two_pass, drop_low_salience=drop_low_salience
        ),
    }
    if chunk_size is not None:
        profile["chunk_size"] = chunk_size

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
