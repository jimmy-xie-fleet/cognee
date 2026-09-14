"""Legal extraction profile."""

from cognee.domains.legal.models import LegalKnowledgeGraph, LegalNode, Polarity, Precision
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
    "Polarity",
    "Precision",
    "legal_profile",
    "legal_ontology_resolver",
    "LEGAL_ONTOLOGY_PATH",
    "DEFAULT_LEGAL_CHUNK_SIZE",
    "LEGAL_FUZZY_CUTOFF",
    "load_legal_extraction_prompt",
]
