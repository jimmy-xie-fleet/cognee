import re
import unicodedata
from enum import Enum
from typing import Any, Optional

from cognee.modules.engine.models.Entity import Entity


class StatementType(str, Enum):
    ALLEGATION = "allegation"
    ADMISSION = "admission"
    DENIAL = "denial"
    TESTIMONY = "testimony"
    OPINION = "opinion"
    FINDING = "finding"
    RECORD = "record"
    TERM = "term"
    PROPOSAL = "proposal"
    STATEMENT = "statement"


STATEMENT_TYPE_NAMES = frozenset(member.value for member in StatementType)


class Assertion(Entity):
    """One occurrence of somebody asserting something.

    Never merged across speakers, statement types, or chunks: an allegation and the denial that
    answers it are two nodes even when their text is identical.
    """

    statement_type: str
    polarity: str = "positive"  # "positive" | "negative"
    asserted_by: Optional[str] = None  # speaker display name; an edge is derived too
    attributed_to: Optional[str] = None
    applicable_time: Optional[str] = None  # ISO date/period the claim is about
    applies_from: Optional[str] = None
    applies_to: Optional[str] = None
    report_date: Optional[str] = None
    conditions: list[str] = []
    precision: Optional[str] = None  # exact | approximate | unknown
    scope: Optional[str] = None
    source_quote: Optional[str] = None
    source_quote_verified: bool = False
    responds_to: Optional[str] = None  # locator text, e.g. "Complaint ¶17"
    source_chunk_id: Optional[str] = None
    occurrence: int = 1
    metadata: dict = {
        "index_fields": ["name"],
        "identity_fields": [
            "name",
            "source_chunk_id",
            "statement_type",
            "asserted_by",
            "occurrence",
        ],
    }


_CURLY_QUOTES_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(_CURLY_QUOTES_TRANSLATION)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip().casefold()


def verify_source_quote(quote: Optional[str], text: Any) -> bool:
    """True when `quote` appears in `text` after normalization. Never raises.

    Normalization: NFKC, curly quotes/apostrophes → straight, all whitespace runs → one space,
    case-insensitive. Non-string `text` or empty/None quote → False.
    """
    if not quote or not isinstance(quote, str):
        return False
    if not isinstance(text, str):
        return False
    return _normalize(quote) in _normalize(text)
