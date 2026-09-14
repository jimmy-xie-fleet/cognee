from pathlib import Path
from typing import Any, Optional

from cognee.domains.legal.models import LegalKnowledgeGraph
from cognee.domains.legal.prompt import load_legal_extraction_prompt
from cognee.modules.ontology.matching_strategies import FuzzyMatchingStrategy
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver

DEFAULT_LEGAL_CHUNK_SIZE = 512
LEGAL_FUZZY_CUTOFF = 0.9
LEGAL_ONTOLOGY_PATH = Path(__file__).parent / "ontology" / "legal.owl"


def legal_ontology_resolver(
    ontology_file: Optional[str] = None, cutoff: float = LEGAL_FUZZY_CUTOFF
) -> RDFLibOntologyResolver:
    """Build an ``RDFLibOntologyResolver`` grounded in the legal OWL vocabulary.

    Args:
        ontology_file: Path to an OWL file. Defaults to the bundled ``legal.owl``.
        cutoff: Fuzzy-match cutoff passed to ``FuzzyMatchingStrategy``.
    """
    path = ontology_file or str(LEGAL_ONTOLOGY_PATH)
    return RDFLibOntologyResolver(
        ontology_file=path,
        matching_strategy=FuzzyMatchingStrategy(cutoff=cutoff),
    )


def legal_profile(
    *,
    ontology_mode: str = "annotate",
    chunk_size: int = DEFAULT_LEGAL_CHUNK_SIZE,
    ontology_file: Optional[str] = None,
    include_ontology: bool = True,
) -> dict[str, Any]:
    """Build the kwargs bundle for ``cognee.remember()`` / ``cognee.cognify()``.

    Splat the result directly into either call, e.g. ``cognee.cognify(**legal_profile())``.
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

    return profile
