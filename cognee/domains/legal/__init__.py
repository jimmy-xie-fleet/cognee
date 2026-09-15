"""Legal extraction profile."""

from cognee.domains.legal.models import (
    LegalKnowledgeGraph,
    LegalNode,
    LegalReference,
    LocatorKind,
    Polarity,
    Precision,
    ReferenceBasis,
)
from cognee.domains.legal.profile import (
    DEFAULT_LEGAL_CHUNK_SIZE,
    LEGAL_FUZZY_CUTOFF,
    LEGAL_ONTOLOGY_PATH,
    legal_ontology_resolver,
    legal_profile,
)
from cognee.domains.legal.prompt import load_legal_extraction_prompt

__all__ = [
    "LegalNode",
    "LegalKnowledgeGraph",
    "LegalReference",
    "LocatorKind",
    "Polarity",
    "Precision",
    "ReferenceBasis",
    "legal_profile",
    "legal_ontology_resolver",
    "LEGAL_ONTOLOGY_PATH",
    "DEFAULT_LEGAL_CHUNK_SIZE",
    "LEGAL_FUZZY_CUTOFF",
    "load_legal_extraction_prompt",
]
