"""Legal extraction profile.

Exports are completed in Task 5 (OWL vocabulary and ``legal_profile()``).
"""

from cognee.domains.legal.models import LegalKnowledgeGraph, LegalNode, Polarity, Precision
from cognee.domains.legal.prompt import load_legal_extraction_prompt

__all__ = [
    "LegalNode",
    "LegalKnowledgeGraph",
    "Polarity",
    "Precision",
    "load_legal_extraction_prompt",
]
