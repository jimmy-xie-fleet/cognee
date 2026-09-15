"""Deterministic helpers for resolving the free-text references an assertion carries.

An extracted ``Assertion`` stores its reference the way the document wrote it --
``responds_to="Complaint ¶5"``, ``attributed_to="Whitfield rebuttal appraisal"`` -- and no
graph edge can follow a string. The helpers here turn that string into the pieces a
resolver needs: a parsed reference, a score against every candidate document's name, the
character span a locator points at inside a document, and the write shapes for the edge and
the node patch that record the answer.

Everything in this module is pure: no I/O, no database, no LLM, no clock, no randomness.
The task that reads documents and writes edges composes these helpers; keeping the rules
here is what makes the scoring, the span selection and the write shapes testable without a
graph engine.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The one normalizer identity and quote verification already use, so a reference, a span of
# document text and a stored source quote are all folded exactly the same way.
from cognee.modules.engine.models.Assertion import _normalize
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# How a resolution names the way it was reached. Stored on every edge and node patch, so
# a consumer can tell an id that was already in the field from one this resolver derived.
STRATEGY_EXISTING_ID = "existing_id"
STRATEGY_ENTITY_NAME = "entity_name"
STRATEGY_DOCUMENT_LOCATOR = "document_locator"
STRATEGY_DOCUMENT_ONLY = "document_only"
STRATEGY_PROSE_LOOKUP = "prose_lookup"

RESOLVED_BY = "reference_resolver"

DEFAULT_MATCH_FLOOR = 0.6
DEFAULT_MATCH_MARGIN = 0.15
DEFAULT_MAX_SPAN = 4000

# Every floor and margin comparison carries this tolerance: 0.60 assembled by float
# addition is not always the 0.6 literal, and a reference must not fail its own floor over
# the order the terms were summed in.
_TOLERANCE = 1e-9

# Scoring weights (see ``score_document`` for the formula they assemble).
_TYPE_WORD_WEIGHT = 0.60
_OTHER_TOKEN_WEIGHT = 0.30
_IDENTIFIER_WEIGHT = 0.60
_OWN_DOCUMENT_PENALTY = 0.30
_EXACT_DATE_TERM = 0.40
_DAY_MONTH_DATE_TERM = 0.30
_YEAR_ONLY_DATE_TERM = 0.10
_DATE_MISMATCH_TERM = -0.40

_LEXICAL_FLOOR = 0.5
_LEXICAL_EARLY_FRACTION = 0.2
_LEXICAL_EARLY_BONUS = 0.5
_MINIMUM_DISTINCTIVE_TOKEN_LENGTH = 4

# Words that name a kind of document rather than a particular one. A reference and a
# document name that share one are about the same kind of paper; the words are worthless
# for telling two papers of that kind apart, so they never count as shared name tokens.
DOCUMENT_TYPE_WORDS = frozenset(
    {
        "addendum",
        "affidavit",
        "agenda",
        "agreement",
        "amendment",
        "answer",
        "appendix",
        "appraisal",
        "assessment",
        "attachment",
        "brief",
        "complaint",
        "contract",
        "declaration",
        "deposition",
        "email",
        "exhibit",
        "invoice",
        "ledger",
        "lease",
        "letter",
        "memo",
        "memorandum",
        "minutes",
        "motion",
        "notice",
        "opinion",
        "order",
        "ordinance",
        "petition",
        "plan",
        "policy",
        "report",
        "resolution",
        "schedule",
        "statement",
        "stipulation",
        "subpoena",
        "summons",
        "testimony",
        "transcript",
    }
)

# The generic prose that glues a reference together ("of the", "dated", "no.") plus the
# case-caption abbreviations ("v", "vs", "re") no document is told apart by.
REFERENCE_STOP_WORDS = frozenset(
    DEFAULT_STOP_WORDS | {"of", "the", "dated", "no", "v", "vs", "re", "to", "in", "and"}
)

# Month names and the abbreviations legal documents actually use. No new dependency:
# dateutil is not in the tree and a reference only ever carries these shapes.
MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# How a document body writes a month. Listed rather than filtered out of MONTHS: "may" is
# a full month name three letters long, so any rule that tells names from abbreviations by
# length drops May and with it every May date a tiebreak could turn on.
_FULL_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

_MONTH_NAME_BY_NUMBER = {MONTHS[name]: name for name in _FULL_MONTH_NAMES}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_WORD_BY_NUMBER = {number: word for word, number in NUMBER_WORDS.items()}

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# Kinds whose marker may be written as digits, a roman numeral or (for counts) a word:
# a complaint referring to "Count 2" is answered by a document headed "COUNT II".
_ROMAN_MARKER_KINDS = frozenset({"count", "article"})
_WORD_MARKER_KINDS = frozenset({"count"})

# How an asserted_by edge words the speaker's stance, and how a reference edge words the
# statement it points at. Both are read by a human and embedded for retrieval, so the
# stance has to be in the sentence rather than reconstructible from the endpoints. The
# unknown phrase reads "takes an unrecorded stance on X" rather than "... on that X": the
# verb takes its object directly, and the result still has to be a sentence.
_STANCE_VERB_BY_POLARITY = {
    "positive": "affirms that",
    "negative": "denies that",
}
_UNRECORDED_STANCE_VERB = "takes an unrecorded stance on"

_DERIVED_EDGE_VERBS = {
    "attributed_to": "is attributed to",
    "responds_to": "responds to",
}


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DateHint:
    """A date as written, with the parts the text actually stated."""

    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


@dataclass(frozen=True)
class Locator:
    """A position inside a document: ``number`` as written, ``ordinal`` when derivable."""

    kind: str
    number: str
    ordinal: Optional[int] = None


@dataclass(frozen=True)
class LocatorPattern:
    """One kind of locator: how a reference writes it and how the document marks it.

    ``marker`` is a template whose ``{number}`` placeholder is filled with the surface
    forms of one locator; ``any_marker`` matches any marker of the kind, so the end of a
    span and the sequence check can be found without knowing which number comes next. A
    ``document_level`` locator names a document rather than a place inside one, so it has
    no marker and stays in the reference's naming hint.
    """

    kind: str
    reference: Tuple[re.Pattern, ...]
    marker: Optional[str] = None
    any_marker: Optional[re.Pattern] = None
    document_level: bool = False


@dataclass(frozen=True)
class ParsedReference:
    """Everything a reference string says, split into the parts matching uses."""

    raw: str
    normalized: str
    hint: str
    hint_type_words: frozenset = frozenset()
    hint_other_tokens: frozenset = frozenset()
    hint_dates: Tuple[DateHint, ...] = ()
    hint_identifiers: frozenset = frozenset()
    locator: Optional[Locator] = None


@dataclass(frozen=True)
class DocumentProfile:
    """A candidate document's name, split the same way a reference's hint is."""

    document_id: str
    name: str
    type_words: frozenset = frozenset()
    other_tokens: frozenset = frozenset()
    dates: Tuple[DateHint, ...] = ()
    identifiers: frozenset = frozenset()


@dataclass(frozen=True)
class DocumentMatch:
    """The document a reference names, and the candidates it was not separated from."""

    document_id: str
    score: float
    ambiguous_with: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Resolution:
    """What one reference on one assertion resolved to, and how."""

    assertion_id: str
    field: str
    reference_text: str
    strategy: str
    confidence: float
    anchor_id: Optional[str] = None
    anchor_type: Optional[str] = None
    target_ids: Tuple[str, ...] = ()
    target_type: Optional[str] = None
    document_id: Optional[str] = None
    notes: Tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# Locator patterns
# --------------------------------------------------------------------------------------

_NUMBER_WORD_ALTERNATION = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))

LOCATOR_PATTERNS: Tuple[LocatorPattern, ...] = (
    LocatorPattern(
        kind="paragraph",
        reference=(
            re.compile(r"¶¶\s*(\d+)\s*[-–]\s*\d+"),
            re.compile(r"¶\s*(\d{1,3})"),
            re.compile(r"\bpara(?:graph|\.)\s*(\d{1,3})"),
        ),
        marker=r"^[ \t]*(?:¶\s*(?:{number})\b|(?:{number})\.(?=[ \t]))",
        any_marker=re.compile(r"^[ \t]*(?:¶\s*(\d{1,3})\b|(\d{1,3})\.(?=[ \t]))", re.M),
    ),
    LocatorPattern(
        kind="section",
        reference=(
            re.compile(r"§+\s*([\d]+(?:\.\d+)*[a-z]?)"),
            re.compile(r"\bsec(?:tion|\.)\s+([\d.]+[a-z]?)"),
        ),
        marker=r"^[ \t]*(?:§\s*(?:{number})\b|section\s+(?:{number})\b)",
        any_marker=re.compile(
            r"^[ \t]*(?:§\s*(\d+(?:\.\d+)*[a-z]?)\b|section\s+(\d+(?:\.\d+)*[a-z]?)\b)",
            re.M | re.I,
        ),
    ),
    LocatorPattern(
        kind="exhibit",
        reference=(re.compile(r"\bex(?:hibit|\.)\s+([a-z]{1,2}|\d{1,3})"),),
        marker=r"^[ \t]*exhibit\s+(?:{number})\b",
        any_marker=re.compile(r"^[ \t]*exhibit\s+([a-z]{1,2}|\d{1,3})\b", re.M | re.I),
    ),
    LocatorPattern(
        kind="count",
        reference=(re.compile(rf"\bcount\s+([ivxl]+|\d{{1,2}}|{_NUMBER_WORD_ALTERNATION})\b"),),
        marker=r"^[ \t]*count\s+(?:{number})\b",
        any_marker=re.compile(
            rf"^[ \t]*count\s+([ivxl]+|\d{{1,2}}|{_NUMBER_WORD_ALTERNATION})\b",
            re.M | re.I,
        ),
    ),
    LocatorPattern(
        kind="article",
        reference=(re.compile(r"\bart(?:icle|\.)\s+([ivxl]+|\d{1,2})\b"),),
        marker=r"^[ \t]*article\s+(?:{number})\b",
        any_marker=re.compile(r"^[ \t]*article\s+([ivxl]+|\d{1,2})\b", re.M | re.I),
    ),
    LocatorPattern(
        kind="resolution",
        reference=(re.compile(r"\bres(?:olution|\.)\s*(?:no\.?\s*)?([0-9]{2,4}-[a-z0-9-]+)"),),
        document_level=True,
    ),
    LocatorPattern(
        kind="ordinance",
        reference=(re.compile(r"\bord(?:inance|\.)\s*(?:no\.?\s*)?([0-9]{2,4}-[a-z0-9-]+)"),),
        document_level=True,
    ),
)

_PATTERN_BY_KIND = {pattern.kind: pattern for pattern in LOCATOR_PATTERNS}


# --------------------------------------------------------------------------------------
# Text normalization and tokenizing
# --------------------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9']+")
_POSSESSIVE_RE = re.compile(r"'s?$")
_EXTENSION_RE = re.compile(r"\.[A-Za-z]{2,5}$")
_LINE_RE = re.compile(r"^[^\n]*$", re.M)


def normalize_reference_text(value: Optional[str]) -> str:
    """Fold a reference, a span of document text or a stored quote the same way.

    NFKC, curly quotes straightened, control characters dropped, whitespace collapsed and
    case folded -- ``Assertion``'s normalizer, so a quote that verified against a document
    also matches the span of that document it was taken from.
    """
    if not isinstance(value, str):
        return ""

    return _normalize(value)


def _tokens(text: str) -> List[str]:
    """Name tokens of already normalized text, with possessives stripped."""
    tokens = []
    for raw_token in _TOKEN_SPLIT_RE.split(text):
        token = _POSSESSIVE_RE.sub("", raw_token).replace("'", "")
        if token:
            tokens.append(token)
    return tokens


def _mask(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    """Blank out spans, keeping every other character at its offset."""
    characters = list(text)
    for start, end in spans:
        for index in range(start, min(end, len(characters))):
            characters[index] = " "
    return "".join(characters)


# --------------------------------------------------------------------------------------
# Dates and numerals
# --------------------------------------------------------------------------------------

_MONTH_ALTERNATION = "|".join(sorted(MONTHS, key=len, reverse=True))
_DATE_SEPARATOR = r"[\s,._-]+"

_ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b")
_SLASH_DATE_RE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})\b")
_MONTH_NAME_DATE_RE = re.compile(
    rf"\b(?P<month>{_MONTH_ALTERNATION})\.?{_DATE_SEPARATOR}(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:{_DATE_SEPARATOR}(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
_BARE_YEAR_RE = re.compile(r"\b(?P<year>1[89]\d{2}|20\d{2})\b")

# A document or resolution number ("2026-118", "24-cv-0117"), which is never a date. The
# lookahead keeps an ISO date out: its parts would otherwise read as an identifier.
_IDENTIFIER_RE = re.compile(r"\b(?!\d{4}-\d{2}-\d{2}\b)\d{2,4}-[a-z0-9]+(?:-[a-z0-9]+)*\b", re.I)


def _expand_two_digit_year(year: int) -> int:
    if year >= 100:
        return year
    return 2000 + year if year < 70 else 1900 + year


def _date_hint(month: Optional[int], day: Optional[int], year: Optional[int]) -> Optional[DateHint]:
    """A hint for the parts given, or None when the numbers cannot be a date."""
    if month is not None and not 1 <= month <= 12:
        return None
    if day is not None and not 1 <= day <= 31:
        return None
    return DateHint(year=year, month=month, day=day)


def _overlaps(spans: Sequence[Tuple[int, int]], start: int, end: int) -> bool:
    return any(start < taken_end and taken_start < end for taken_start, taken_end in spans)


def _date_matches(text: str) -> List[Tuple[DateHint, int, int]]:
    """Every date in the text as (hint, start, end), in reading order, without overlaps.

    Forms are tried most specific first, and an identifier-shaped number is masked before
    bare years are looked for: "Resolution No. 2026-118" states no year.
    """
    matches: List[Tuple[DateHint, int, int]] = []
    taken: List[Tuple[int, int]] = []

    for pattern in (_ISO_DATE_RE, _SLASH_DATE_RE, _MONTH_NAME_DATE_RE):
        for match in pattern.finditer(text):
            if _overlaps(taken, match.start(), match.end()):
                continue

            groups = match.groupdict()
            month_text = groups.get("month") or ""
            month = MONTHS.get(month_text.casefold(), None)
            if month is None and month_text.isdigit():
                month = int(month_text)

            year_text = groups.get("year")
            hint = _date_hint(
                month,
                int(groups["day"]) if groups.get("day") else None,
                _expand_two_digit_year(int(year_text)) if year_text else None,
            )
            if hint is None:
                continue

            matches.append((hint, match.start(), match.end()))
            taken.append(match.span())

    masked = _mask(text, taken)
    masked = _mask(masked, [match.span() for match in _IDENTIFIER_RE.finditer(masked)])
    for match in _BARE_YEAR_RE.finditer(masked):
        matches.append((DateHint(year=int(match.group("year"))), match.start(), match.end()))

    return sorted(matches, key=lambda match: match[1])


def parse_dates(text: Optional[str]) -> Tuple[DateHint, ...]:
    """Every date a string states, in reading order. Never raises."""
    if not isinstance(text, str) or not text:
        return ()

    return tuple(hint for hint, _, _ in _date_matches(text.replace("_", " ")))


def roman_to_int(value: Optional[str]) -> Optional[int]:
    """The value of a roman numeral, or None when the string is not one."""
    if not isinstance(value, str):
        return None

    folded = value.strip().casefold()
    if not folded or any(character not in _ROMAN_VALUES for character in folded):
        return None

    total = 0
    highest = 0
    for character in reversed(folded):
        current = _ROMAN_VALUES[character]
        total = total - current if current < highest else total + current
        highest = max(highest, current)
    return total or None


def _int_to_roman(value: int) -> Optional[str]:
    if not 1 <= value <= 3999:
        return None

    numerals = (
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    )
    remainder = value
    rendered = []
    for amount, numeral in numerals:
        while remainder >= amount:
            rendered.append(numeral)
            remainder -= amount
    return "".join(rendered)


def _ordinal_for_kind(kind: str, number: str) -> Optional[int]:
    """The integer position a locator number states, when it states one.

    A dotted section number ("3.2") has no single position, and a letter is only an
    alphabet position for the kinds that are lettered -- "Exhibit C" is the third exhibit,
    while "Count C" would be a roman hundred.
    """
    folded = number.strip().casefold()
    if not folded:
        return None
    if folded.isdigit():
        return int(folded)
    if folded in NUMBER_WORDS:
        return NUMBER_WORDS[folded]
    if kind in _ROMAN_MARKER_KINDS:
        return roman_to_int(folded)
    if len(folded) == 1 and folded.isalpha():
        return ord(folded) - ord("a") + 1
    return None


# --------------------------------------------------------------------------------------
# Parsing references and document names
# --------------------------------------------------------------------------------------


def _extract_hints(text: str, *, mask_bare_years: bool):
    """Split naming text into (type words, other tokens, dates, identifiers).

    Dates and identifiers are taken out before the rest is tokenized, so a name never
    leaves "september", "22" and "2026" behind as three tokens to match on.

    A bare year is masked for a document name but kept as a token for a reference: it is
    the reference's own denominator, and "the 2014 master plan" says the year is part of
    how the reader named the document, while the document's own name already contributes
    it as a date.
    """
    prepared = text.replace("_", " ")

    matches = _date_matches(prepared)
    dates = tuple(hint for hint, _, _ in matches)
    if mask_bare_years:
        date_spans = [(start, end) for _, start, end in matches]
    else:
        date_spans = [(start, end) for hint, start, end in matches if hint.month is not None]

    masked = _mask(prepared, date_spans)
    identifier_matches = list(_IDENTIFIER_RE.finditer(masked))
    identifiers = frozenset(match.group().casefold() for match in identifier_matches)
    masked = _mask(masked, [match.span() for match in identifier_matches])

    tokens = _tokens(masked)
    type_words = frozenset(token for token in tokens if token in DOCUMENT_TYPE_WORDS)
    other_tokens = frozenset(
        token
        for token in tokens
        if token not in DOCUMENT_TYPE_WORDS and token not in REFERENCE_STOP_WORDS
    )
    return type_words, other_tokens, dates, identifiers


def _find_locator(normalized: str):
    """The earliest locator the reference states, as (pattern, match)."""
    best = None
    for table_index, pattern in enumerate(LOCATOR_PATTERNS):
        for reference_pattern in pattern.reference:
            match = reference_pattern.search(normalized)
            if match is None:
                continue

            key = (match.start(), table_index)
            if best is None or key < best[0]:
                best = (key, pattern, match)
    if best is None:
        return None, None
    return best[1], best[2]


def parse_reference(text: Optional[str]) -> ParsedReference:
    """Split a reference string into the locator it points at and the document it names.

    Never raises: a string stating neither ("unit c") parses into a hint with no locator,
    which resolves to nothing rather than to an error.
    """
    raw = text if isinstance(text, str) else ""
    normalized = normalize_reference_text(raw)
    if not normalized:
        return ParsedReference(raw=raw, normalized="", hint="")

    pattern, match = _find_locator(normalized)
    locator = None
    hint = normalized
    if match is not None:
        locator = Locator(
            kind=pattern.kind,
            number=match.group(1).strip(),
            ordinal=_ordinal_for_kind(pattern.kind, match.group(1)),
        )
        if not pattern.document_level:
            # The locator points inside a document, so it says nothing about which
            # document; a document level locator IS the document's number and stays.
            hint = f"{normalized[: match.start()]} {normalized[match.end() :]}"

    hint = _WHITESPACE_RE.sub(" ", hint).strip()
    type_words, other_tokens, dates, identifiers = _extract_hints(hint, mask_bare_years=False)
    return ParsedReference(
        raw=raw,
        normalized=normalized,
        hint=hint,
        hint_type_words=type_words,
        hint_other_tokens=other_tokens,
        hint_dates=dates,
        hint_identifiers=identifiers,
        locator=locator,
    )


def document_profile(document_id: str, name: str) -> DocumentProfile:
    """Split a document's name into the parts a reference is scored against.

    ``name`` is the stored document name, which is a file stem more often than a title
    ("Deposition_Hartwell_September_22_2026"); a trailing file extension is dropped.
    """
    stem = _EXTENSION_RE.sub("", name if isinstance(name, str) else "")
    type_words, other_tokens, dates, identifiers = _extract_hints(
        normalize_reference_text(stem), mask_bare_years=True
    )
    return DocumentProfile(
        document_id=document_id,
        name=name,
        type_words=type_words,
        other_tokens=other_tokens,
        dates=dates,
        identifiers=identifiers,
    )


# --------------------------------------------------------------------------------------
# Matching a reference to a document
# --------------------------------------------------------------------------------------


def _date_pair_term(reference_date: DateHint, document_date: DateHint) -> float:
    """How much one pair of dates argues for or against the same document."""
    if reference_date.year is None or document_date.year is None:
        # Nothing to contradict: a year-less reference matching the day and month is a
        # good sign, anything else is simply no signal.
        if (
            reference_date.month is not None
            and reference_date.month == document_date.month
            and reference_date.day is not None
            and reference_date.day == document_date.day
        ):
            return _DAY_MONTH_DATE_TERM
        return 0.0

    if reference_date.year != document_date.year:
        return _DATE_MISMATCH_TERM

    for reference_part, document_part in (
        (reference_date.month, document_date.month),
        (reference_date.day, document_date.day),
    ):
        if reference_part is None or document_part is None:
            # One side only ever stated the year, and the years agree.
            return _YEAR_ONLY_DATE_TERM
        if reference_part != document_part:
            # Both sides state a full date and they are different days: two documents of
            # the same kind from the same year are told apart exactly here.
            return _DATE_MISMATCH_TERM

    return _EXACT_DATE_TERM


def _date_term(reference_dates: Sequence[DateHint], document_dates: Sequence[DateHint]) -> float:
    if not reference_dates or not document_dates:
        return 0.0

    return max(
        _date_pair_term(reference_date, document_date)
        for reference_date in reference_dates
        for document_date in document_dates
    )


def score_document(
    reference: ParsedReference,
    profile: DocumentProfile,
    *,
    own_document_id: Optional[str] = None,
) -> float:
    """How strongly a reference names one document, in [0, 1].

    0.60 for agreeing on the kind of document, up to 0.30 for the share of the
    reference's own distinctive tokens the name carries, 0.60 for a shared identifier,
    the date term, and -0.30 when the candidate is the document the assertion itself came
    from -- a reference in a document almost never points back at that same document.
    """
    score = 0.0
    if reference.hint_type_words & profile.type_words:
        score += _TYPE_WORD_WEIGHT

    shared_tokens = len(reference.hint_other_tokens & profile.other_tokens)
    score += _OTHER_TOKEN_WEIGHT * shared_tokens / max(1, len(reference.hint_other_tokens))

    if reference.hint_identifiers & profile.identifiers:
        score += _IDENTIFIER_WEIGHT

    score += _date_term(reference.hint_dates, profile.dates)

    if own_document_id is not None and profile.document_id == own_document_id:
        score -= _OWN_DOCUMENT_PENALTY

    return max(0.0, min(1.0, score))


def match_document(
    reference: ParsedReference,
    profiles: Iterable[DocumentProfile],
    *,
    own_document_id: Optional[str],
    floor: float = DEFAULT_MATCH_FLOOR,
    margin: float = DEFAULT_MATCH_MARGIN,
) -> Optional[DocumentMatch]:
    """The document a reference names, or None when nothing scores high enough.

    A match whose ``ambiguous_with`` is not empty was not separated from those candidates
    by ``margin``; the caller decides between them with ``lexical_tiebreak`` over their
    text. Only candidates inside the margin are listed -- the rest are already decided,
    and the caller would otherwise read their documents for nothing.
    """
    scored = [
        (profile.document_id, score_document(reference, profile, own_document_id=own_document_id))
        for profile in profiles
    ]
    candidates = sorted(
        (candidate for candidate in scored if candidate[1] >= floor - _TOLERANCE),
        key=lambda candidate: (-candidate[1], candidate[0]),
    )
    if not candidates:
        return None

    document_id, score = candidates[0]
    ambiguous_with = tuple(
        other_id
        for other_id, other_score in candidates[1:]
        if score - other_score < margin - _TOLERANCE
    )
    return DocumentMatch(document_id=document_id, score=score, ambiguous_with=ambiguous_with)


def _date_renderings(date: DateHint) -> Tuple[str, ...]:
    """The ways a document's body writes a date the reference stated.

    A year-only hint renders nothing: the year is already one of the reference's tokens,
    and scoring it twice would let a single number carry a tiebreak.
    """
    if date.month is None or date.day is None:
        return ()

    month_name = _MONTH_NAME_BY_NUMBER.get(date.month)
    if month_name is None:
        return ()

    if date.year is None:
        return (f"{month_name} {date.day}",)

    return (
        f"{month_name} {date.day}, {date.year}",
        f"{month_name} {date.day} {date.year}",
        f"{date.year}-{date.month:02d}-{date.day:02d}",
        f"{date.month}/{date.day}/{date.year}",
    )


def _lexical_terms(reference: ParsedReference) -> Tuple[Tuple[str, ...], ...]:
    """One entry per distinctive thing the reference says, with its surface forms."""
    terms = [
        (token,)
        for token in sorted(reference.hint_other_tokens)
        if len(token) >= _MINIMUM_DISTINCTIVE_TOKEN_LENGTH
    ]
    for date in reference.hint_dates:
        renderings = _date_renderings(date)
        if renderings:
            terms.append(renderings)
    return tuple(terms)


def _lexical_score(terms: Sequence[Tuple[str, ...]], text: str) -> float:
    folded = normalize_reference_text(text)
    if not folded or not terms:
        return 0.0

    early_limit = max(1, int(len(folded) * _LEXICAL_EARLY_FRACTION))
    found = 0.0
    for surfaces in terms:
        positions = [position for position in (folded.find(s) for s in surfaces) if position >= 0]
        if not positions:
            continue

        found += 1.0
        if min(positions) < early_limit:
            # Documents name themselves at the top, so an early hit is worth more than a
            # passing mention halfway down a deposition.
            found += _LEXICAL_EARLY_BONUS

    return found / ((1.0 + _LEXICAL_EARLY_BONUS) * len(terms))


def lexical_tiebreak(
    reference: ParsedReference,
    texts_by_document_id: Mapping[str, str],
) -> Optional[Tuple[str, float]]:
    """Decide between candidate documents on their text, or None when it cannot.

    Returns ``(document_id, score)`` only for a winner that both clears the floor and is
    strictly ahead of the runner up; a tie stays unresolved rather than guessing.
    """
    terms = _lexical_terms(reference)
    if not terms or not texts_by_document_id:
        return None

    scored = sorted(
        (
            (document_id, _lexical_score(terms, text))
            for document_id, text in texts_by_document_id.items()
        ),
        key=lambda candidate: (-candidate[1], candidate[0]),
    )
    document_id, score = scored[0]
    runner_up = scored[1][1] if len(scored) > 1 else 0.0
    if score < _LEXICAL_FLOOR - _TOLERANCE or score - runner_up <= _TOLERANCE:
        return None

    return document_id, score


# --------------------------------------------------------------------------------------
# Finding a locator inside document text
# --------------------------------------------------------------------------------------


def _marker_alternatives(pattern: LocatorPattern, locator: Locator) -> str:
    """The surface forms a document may write this locator's number as."""
    forms = {locator.number.strip().casefold()}
    if locator.ordinal is not None:
        if pattern.kind in _ROMAN_MARKER_KINDS:
            forms.add(str(locator.ordinal))
            roman = _int_to_roman(locator.ordinal)
            if roman:
                forms.add(roman)
        if pattern.kind in _WORD_MARKER_KINDS:
            word = _WORD_BY_NUMBER.get(locator.ordinal)
            if word:
                forms.add(word)

    # Longest first, so "ii" is never matched as the "i" of "iii".
    ordered = sorted(forms, key=lambda form: (-len(form), form))
    return "|".join(re.escape(form) for form in ordered)


def _marker_number(match: re.Match) -> Optional[str]:
    return next((group for group in match.groups() if group), None)


def _next_marker(pattern: LocatorPattern, text: str, after: int):
    """(start, ordinal) of the first marker of this kind strictly after ``after``."""
    if pattern.any_marker is None:
        return None

    for match in pattern.any_marker.finditer(text):
        if match.start() <= after:
            continue

        number = _marker_number(match)
        ordinal = _ordinal_for_kind(pattern.kind, number) if number else None
        return match.start(), ordinal
    return None


def _next_heading_start(text: str, after: int) -> Optional[int]:
    """Where the next ALL-CAPS heading line starts, if there is one.

    Legal documents head their next part in capitals ("SECOND CAUSE OF ACTION"), which is
    the end of the current one even when the next numbered marker is far below.
    """
    for match in _LINE_RE.finditer(text, after):
        if match.start() <= after:
            continue

        line = match.group().strip()
        if (
            len(line) >= 3
            and any(character.isalpha() for character in line)
            and line == line.upper()
        ):
            return match.start()
    return None


def _span_end(pattern: LocatorPattern, text: str, start: int, max_span: int) -> int:
    ends = [len(text), start + max_span]
    next_marker = _next_marker(pattern, text, start)
    if next_marker is not None:
        ends.append(next_marker[0])

    heading = _next_heading_start(text, start)
    if heading is not None:
        ends.append(heading)

    return max(start, min(ends))


def find_locator_span(
    text: str,
    locator: Optional[Locator],
    max_span: int = DEFAULT_MAX_SPAN,
) -> Optional[Tuple[int, int, Tuple[str, ...]]]:
    """Where in a document a locator points, as ``(start, end, notes)``.

    With one marker in the document the answer is that marker. With several -- a number
    that also opens an unrelated list -- the one whose next marker of the same kind
    continues the sequence wins; when none does, the first is used and the span is noted
    ``ambiguous_marker`` so the caller can weigh it lower.
    """
    if not text or locator is None:
        return None

    pattern = _PATTERN_BY_KIND.get(locator.kind)
    if pattern is None or pattern.marker is None:
        # A document level locator ("Resolution No. 2026-118") marks no place in a text.
        return None

    marker_pattern = re.compile(
        pattern.marker.replace("{number}", _marker_alternatives(pattern, locator)),
        re.M | re.I,
    )
    matches = list(marker_pattern.finditer(text))
    if not matches:
        return None

    notes: Tuple[str, ...] = ()
    start = matches[0].start()
    if len(matches) > 1:
        in_sequence = None
        if locator.ordinal is not None:
            for match in matches:
                next_marker = _next_marker(pattern, text, match.start())
                if next_marker is not None and next_marker[1] == locator.ordinal + 1:
                    in_sequence = match.start()
                    break

        if in_sequence is None:
            notes = ("ambiguous_marker",)
        else:
            start = in_sequence

    return start, _span_end(pattern, text, start, max_span), notes


def chunks_overlapping(
    offsets: Sequence[Tuple[int, int]],
    span: Tuple[int, int],
) -> List[int]:
    """Indices of the chunks a span covers any part of, in document order."""
    start, end = span
    return [
        index
        for index, (chunk_start, chunk_end) in enumerate(offsets)
        if max(chunk_start, start) < min(chunk_end, end)
    ]


def anchor_chunk_index(
    offsets: Sequence[Tuple[int, int]],
    span: Tuple[int, int],
) -> Optional[int]:
    """The chunk a span belongs to: the one it covers most of, earliest on a tie.

    A marker often ends one chunk and its text starts the next, so the chunk holding the
    marker is rarely the chunk holding what the locator points at.
    """
    start, end = span
    best_index = None
    best_overlap = 0
    for index, (chunk_start, chunk_end) in enumerate(offsets):
        overlap = min(chunk_end, end) - max(chunk_start, start)
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
    return best_index


def scan_chunks_for_marker(
    chunk_texts: Sequence[str],
    locator: Optional[Locator],
) -> Optional[Tuple[int, Tuple[int, int]]]:
    """Find a locator's marker chunk by chunk, as ``(chunk index, span in that chunk)``.

    The fallback for a document whose stored chunks do not tile its text, where offsets
    into the whole document cannot be mapped onto chunks. The span follows the same end
    rules, bounded by the chunk it was found in.
    """
    for index, chunk_text in enumerate(chunk_texts):
        span = find_locator_span(chunk_text, locator)
        if span is not None:
            return index, (span[0], span[1])
    return None


def select_anchored_assertions(
    span_text: str,
    candidates: Iterable[Tuple[str, Optional[str]]],
) -> List[str]:
    """The candidates whose quoted source text lies inside the span, in input order.

    A blank or missing quote never selects: it normalizes to the empty string, which is
    inside every span, and would anchor an assertion to a passage it never quoted.
    """
    normalized_span = normalize_reference_text(span_text)
    if not normalized_span:
        return []

    selected = []
    for assertion_id, source_quote in candidates:
        normalized_quote = normalize_reference_text(source_quote)
        if normalized_quote and normalized_quote in normalized_span:
            selected.append(assertion_id)
    return selected


# --------------------------------------------------------------------------------------
# Write shapes
# --------------------------------------------------------------------------------------


def _strip_nonblank_text(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None

    stripped_value = value.strip()
    return stripped_value or None


def _sentence(text: str) -> str:
    """One sentence of edge text, terminated exactly once."""
    stripped_text = text.strip()
    return stripped_text if stripped_text.endswith((".", "!", "?")) else f"{stripped_text}."


def derived_edge_text(
    proposition: Optional[str],
    statement_type: Optional[str],
    polarity: Optional[str],
    relationship_name: str,
    target_label: Optional[str],
    description: Optional[str] = None,
) -> str:
    """The text an assertion's edge carries, stating the stance it was made with.

    Without it the edge reaches storage with no ``edge_text`` and
    ``ensure_default_edge_properties`` synthesizes one from the endpoint labels -- for an
    assertion that is its affirmative ``name``, so a denial is embedded and shown as the
    fact it denies. The stance therefore has to travel with the edge, not be reconstructed
    from the endpoints, which no longer carry it.
    """
    stripped_proposition = _strip_nonblank_text(proposition)
    clause = (stripped_proposition or "").rstrip(".").strip() or "this statement"
    stance = _strip_nonblank_text(polarity) or "unknown"
    label = _strip_nonblank_text(target_label) or "an unnamed party"

    if relationship_name == "asserted_by":
        stance_verb = _STANCE_VERB_BY_POLARITY.get(stance, _UNRECORDED_STANCE_VERB)
        head = f"{label} {stance_verb} {clause}"
    else:
        verb = _DERIVED_EDGE_VERBS.get(relationship_name, relationship_name.replace("_", " "))
        head = (
            f"{clause} ({_strip_nonblank_text(statement_type) or 'statement'}, {stance} stance) "
            f"{verb} {label}"
        )

    detail = _strip_nonblank_text(description)
    return " ".join(_sentence(part) for part in (head, detail) if part)


def stance_edge_text(
    source_props: Mapping[str, Any],
    relationship_name: str,
    target_label: Optional[str],
) -> str:
    """``derived_edge_text`` for an assertion read back out of the graph."""
    return derived_edge_text(
        source_props.get("name"),
        source_props.get("statement_type"),
        source_props.get("polarity"),
        relationship_name,
        target_label,
        source_props.get("description"),
    )


def build_reference_edge(
    resolution: Resolution,
    target_id: str,
    *,
    source_props: Mapping[str, Any],
    target_label: Optional[str],
    target_type: str,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """One ``(source, target, relationship, properties)`` edge for a resolved reference.

    The raw shape ``add_edges`` takes: the caller still runs it through
    ``ensure_default_edge_properties``, which fills the storage defaults and leaves the
    stance-preserving ``edge_text`` set here alone.
    """
    return (
        resolution.assertion_id,
        target_id,
        resolution.field,
        {
            "relationship_name": resolution.field,
            "source_node_id": resolution.assertion_id,
            "target_node_id": target_id,
            "reference_text": resolution.reference_text,
            "resolution_strategy": resolution.strategy,
            "resolution_confidence": resolution.confidence,
            "resolved_target_type": target_type,
            "resolved_by": RESOLVED_BY,
            "edge_text": stance_edge_text(source_props, resolution.field, target_label),
        },
    )


def build_node_patch(resolution: Resolution, current_props: Mapping[str, Any]) -> Dict[str, Any]:
    """The properties to write back on the assertion the reference was read from.

    The field itself becomes the anchor's id, so the graph can follow it, and the text it
    used to hold moves to ``<field>_text`` -- but only if nothing is there yet, because a
    re-resolution must not overwrite the original wording with its own idea of it.
    """
    field = resolution.field
    return {
        field: resolution.anchor_id,
        f"{field}_text": current_props.get(f"{field}_text") or resolution.reference_text,
        f"{field}_resolution": {
            "strategy": resolution.strategy,
            "confidence": resolution.confidence,
            "target_type": resolution.target_type,
            "target_ids": list(resolution.target_ids),
            "anchor_id": resolution.anchor_id,
            "document_id": resolution.document_id,
            "notes": list(resolution.notes),
        },
    }


# --------------------------------------------------------------------------------------
# Structured reference hints (§1.2) -- reads Assertion.responds_to_ref / attributed_to_ref
# --------------------------------------------------------------------------------------
#
# Extraction now writes a structured reference (a plain dict, so core never imports the
# legal domain package) instead of only free text. These helpers read that dict -- or a
# JSON string, the shape Neo4j returns a dict property as -- into one typed hint, render it
# for display/retrieval, fingerprint it for re-run guards, and turn a known
# ``(kind, value)`` pair into the ``Locator`` ``find_locator_span`` already understands.
# Deliberately additive: nothing above this block is touched by this change.


@dataclass(frozen=True)
class ReferenceHint:
    """One reference an assertion carries, already split into its named parts.

    ``legacy_text`` is set only when the hint came from a pre-structured free-text
    reference (no ``document_hint``/locator/``date`` of its own to read) -- ``document_hint``
    then mirrors it verbatim so a caller reading only ``document_hint`` still gets the
    reference's words.
    """

    document_hint: str = ""
    locator_kind: Optional[str] = None
    locator_value: Optional[str] = None
    date: Optional[str] = None
    basis: Optional[str] = None
    legacy_text: Optional[str] = None  # a pre-structured free-text reference


def _clean_hint_value(value: Any) -> Optional[str]:
    """Stringify and strip one hint field; blank / ``"none"`` (any case) fold to ``None``."""
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text or text.casefold() == "none":
        return None
    return text


def parse_reference_hint(
    raw: Any, *, fallback_text: Optional[str] = None
) -> Optional[ReferenceHint]:
    """Read a structured reference dict (or the JSON string Neo4j stores it as) into a hint.

    ``raw`` is a ``dict`` (Ladybug returns node properties as-is), a JSON string that
    decodes to a ``dict`` (Neo4j serialises dict properties as strings), or anything else.
    A plain, non-JSON string in ``raw`` is *not* a hint -- it is never parsed, and never
    becomes ``legacy_text`` on its own; only ``fallback_text`` can supply legacy text. When
    the dict carries no ``document_hint``, no locator and no ``date`` -- or ``raw`` yields
    nothing at all -- the result falls back to ``fallback_text`` (a pre-structured
    free-text reference) when that is non-blank, else ``None``. Never raises.
    """
    data: Optional[Mapping[str, Any]] = None
    try:
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, str):
            try:
                loaded = json.loads(raw)
            except (ValueError, TypeError):
                loaded = None
            if isinstance(loaded, dict):
                data = loaded
        # Anything else (None, int, float, bool, list, a non-dict JSON value) is not a hint.

        if data is not None:
            document_hint = _clean_hint_value(data.get("document_hint")) or ""
            locator_kind = _clean_hint_value(data.get("locator_kind"))
            locator_value = _clean_hint_value(data.get("locator_value"))
            date = _clean_hint_value(data.get("date"))
            basis = _clean_hint_value(data.get("basis"))
            if document_hint or locator_kind or locator_value or date:
                return ReferenceHint(
                    document_hint=document_hint,
                    locator_kind=locator_kind,
                    locator_value=locator_value,
                    date=date,
                    basis=basis,
                )
    except Exception:
        pass

    try:
        fallback = fallback_text.strip() if isinstance(fallback_text, str) else ""
    except Exception:
        fallback = ""
    if fallback:
        return ReferenceHint(document_hint=fallback, legacy_text=fallback)
    return None


def reference_display_text(hint: Optional[ReferenceHint]) -> str:
    """Render a hint for display/retrieval: composition only, no parsing.

    ``"Complaint paragraph 13"``, ``"June 10 letter (2026-06-10)"`` -- a legacy hint
    renders as its stored text verbatim. May be ``""`` only when everything is empty.
    """
    if hint is None:
        return ""
    if hint.legacy_text:
        text = hint.legacy_text
    else:
        text = hint.document_hint or ""
        if hint.locator_kind and hint.locator_value:
            text = f"{text} {hint.locator_kind} {hint.locator_value}"
        if hint.date:
            text = f"{text} ({hint.date})"
    return _WHITESPACE_RE.sub(" ", text).strip()


def reference_fingerprint(hint: ReferenceHint, field_name: str) -> str:
    """A 16-hex-char sha1 over ``field_name`` and every field of ``hint``.

    Used as a re-run guard: unchanged inputs (including which field this is) hash the same.
    """
    parts = (
        field_name,
        hint.document_hint or "",
        hint.locator_kind or "",
        hint.locator_value or "",
        hint.date or "",
        hint.legacy_text or "",
    )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def build_locator(kind: Optional[str], value: Optional[str]) -> Optional[Locator]:
    """A ``Locator`` for an already-known ``(kind, value)`` pair, or ``None``.

    ``None`` when ``kind`` is falsy, ``"none"``, ``"page"`` (there is no ``page`` kind), or
    not one of the marker-bearing kinds in ``_PATTERN_BY_KIND`` (``resolution``/``ordinance``
    are document-level and have no marker), or when ``value`` is blank. The ordinal is
    ``_ordinal_for_kind``'s -- ``None`` when the number has no single position (a dotted
    section like ``"3.2"``).
    """
    if not isinstance(kind, str):
        return None
    kind_lower = kind.strip().casefold()
    if not kind_lower or kind_lower in ("none", "page"):
        return None
    pattern = _PATTERN_BY_KIND.get(kind_lower)
    if pattern is None or pattern.marker is None:
        return None
    if not isinstance(value, str):
        return None
    number = value.strip()
    if not number:
        return None
    return Locator(kind=kind_lower, number=number, ordinal=_ordinal_for_kind(kind_lower, number))
