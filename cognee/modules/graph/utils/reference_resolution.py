"""Deterministic helpers for resolving the free-text references an assertion carries.

An extracted ``Assertion`` records its reference as a structured hint
(``responds_to_ref={"document_hint": "the Complaint", "locator_kind": "paragraph", ...}``)
or, on older data, as the string the document wrote -- and no graph edge can follow
either. The helpers here are what the resolver composes around that: reading the hint,
rendering and fingerprinting it, turning a known ``(kind, value)`` pair into a locator,
finding the character span that locator points at inside a document, and the write shapes
for the edge and the node patch that record the answer.

Deliberately **not** here (decision D5): any parsing of, or scoring against, the
reference's own free text. Guessing which document "the Whitfield rebuttal appraisal"
names is the agentic tracer's job, and no regex in this module ever runs over reference
text -- the marker patterns below run over *document* text, with a number the extraction
LLM or the agent supplied.

Everything in this module is pure: no I/O, no database, no LLM, no clock, no randomness.
The task that reads documents and writes edges composes these helpers; keeping the rules
here is what makes the span selection and the write shapes testable without a graph
engine.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# The one normalizer identity and quote verification already use, so a reference, a span of
# document text and a stored source quote are all folded exactly the same way.
from cognee.modules.engine.models.Assertion import _normalize

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# How a resolution names the way it was reached. Stored on every edge and node patch, so
# a consumer can tell an id that was already in the field from one this resolver derived.
STRATEGY_EXISTING_ID = "existing_id"
STRATEGY_ENTITY_NAME = "entity_name"
# The agentic tracer's two answers: a reference the document actually made
# (``llm_trace``) and a denial/admission link the document only implied
# (``llm_inferred``, opt-in). Both are written by the resolver pass, never by the tail.
STRATEGY_LLM_TRACE = "llm_trace"
STRATEGY_LLM_INFERRED = "llm_inferred"

RESOLVED_BY = "reference_resolver"

DEFAULT_MAX_SPAN = 4000

# A locator number written out as a word ("Count Three"), so a marker can be matched
# whichever way the two documents spelled it.
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
class Locator:
    """A position inside a document: ``number`` as written, ``ordinal`` when derivable."""

    kind: str
    number: str
    ordinal: Optional[int] = None


@dataclass(frozen=True)
class LocatorPattern:
    """One kind of locator, and how a *document* marks it.

    ``marker`` is a template whose ``{number}`` placeholder is filled with the surface
    forms of one locator; ``any_marker`` matches any marker of the kind, so the end of a
    span and the sequence check can be found without knowing which number comes next. A
    ``document_level`` locator names a document rather than a place inside one, so it has
    no marker and stays in the reference's naming hint.

    There is deliberately no pattern for how a *reference* writes a locator (decision
    D5): the ``(kind, value)`` pair reaches ``build_locator`` from the extraction LLM or
    from the agent, and nothing here ever runs a regex over reference text.
    """

    kind: str
    marker: Optional[str] = None
    any_marker: Optional[re.Pattern] = None
    document_level: bool = False


@dataclass(frozen=True)
class Resolution:
    """What one reference on one assertion resolved to, and how.

    The last five fields carry the agentic tracer's answer. ``patch_mode`` decides what
    the write phase may put back on the node: ``"full"`` points the field at the anchor
    (the shape every resolved reference has always written), ``"resolution_only"`` writes
    only the ``<field>_resolution`` audit blob (an abstention must never overwrite a field
    the extraction left as the document wrote it), and ``"none"`` patches nothing at all
    (``existing_id`` already holds an id; ``entity_name`` must keep its name).

    ``trace`` is a tuple of plain dicts rather than tracer objects on purpose: this module
    is pure and knows nothing about the task layer's ``TraceRecord``. It makes a
    ``Resolution`` unhashable, which is fine -- nothing puts one in a set.
    """

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
    reason: Optional[str] = None
    fingerprint: Optional[str] = None
    patch_mode: str = "full"
    iterations: int = 0
    trace: Tuple[Dict[str, Any], ...] = ()
    # The per-reference step cap a trace was held to, recorded only by the record that
    # hit it: the guard that skips an already-attempted reference honours such a record
    # only while the cap it was capped at is still in force (R22).
    max_iter: Optional[int] = None


# --------------------------------------------------------------------------------------
# Locator patterns
# --------------------------------------------------------------------------------------

_NUMBER_WORD_ALTERNATION = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))

LOCATOR_PATTERNS: Tuple[LocatorPattern, ...] = (
    LocatorPattern(
        kind="paragraph",
        marker=r"^[ \t]*(?:¶\s*(?:{number})\b|(?:{number})\.(?=[ \t]))",
        any_marker=re.compile(r"^[ \t]*(?:¶\s*(\d{1,3})\b|(\d{1,3})\.(?=[ \t]))", re.M),
    ),
    LocatorPattern(
        kind="section",
        marker=r"^[ \t]*(?:§\s*(?:{number})\b|section\s+(?:{number})\b)",
        any_marker=re.compile(
            r"^[ \t]*(?:§\s*(\d+(?:\.\d+)*[a-z]?)\b|section\s+(\d+(?:\.\d+)*[a-z]?)\b)",
            re.M | re.I,
        ),
    ),
    LocatorPattern(
        kind="exhibit",
        marker=r"^[ \t]*exhibit\s+(?:{number})\b",
        any_marker=re.compile(r"^[ \t]*exhibit\s+([a-z]{1,2}|\d{1,3})\b", re.M | re.I),
    ),
    LocatorPattern(
        kind="count",
        marker=r"^[ \t]*count\s+(?:{number})\b",
        any_marker=re.compile(
            rf"^[ \t]*count\s+([ivxl]+|\d{{1,2}}|{_NUMBER_WORD_ALTERNATION})\b",
            re.M | re.I,
        ),
    ),
    LocatorPattern(
        kind="article",
        marker=r"^[ \t]*article\s+(?:{number})\b",
        any_marker=re.compile(r"^[ \t]*article\s+([ivxl]+|\d{1,2})\b", re.M | re.I),
    ),
    LocatorPattern(kind="resolution", document_level=True),
    LocatorPattern(kind="ordinance", document_level=True),
)

_PATTERN_BY_KIND = {pattern.kind: pattern for pattern in LOCATOR_PATTERNS}


# --------------------------------------------------------------------------------------
# Text normalization and tokenizing
# --------------------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
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


# --------------------------------------------------------------------------------------
# Numerals
# --------------------------------------------------------------------------------------


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
    extra_properties: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """One ``(source, target, relationship, properties)`` edge for a resolved reference.

    The raw shape ``add_edges`` takes: the caller still runs it through
    ``ensure_default_edge_properties``, which fills the storage defaults and leaves the
    stance-preserving ``edge_text`` set here alone.

    ``extra_properties`` is merged **last**, so a caller can both add properties (an
    inferred link's ``inferred``/``feedback_weight`` marks) and deliberately override one
    of the shape's own.
    """
    properties = {
        "relationship_name": resolution.field,
        "source_node_id": resolution.assertion_id,
        "target_node_id": target_id,
        "reference_text": resolution.reference_text,
        "resolution_strategy": resolution.strategy,
        "resolution_confidence": resolution.confidence,
        "resolved_target_type": target_type,
        "resolved_by": RESOLVED_BY,
        "edge_text": stance_edge_text(source_props, resolution.field, target_label),
    }
    if extra_properties:
        properties.update(extra_properties)

    return (resolution.assertion_id, target_id, resolution.field, properties)


def build_node_patch(
    resolution: Resolution,
    current_props: Mapping[str, Any],
    *,
    mode: str = "full",
) -> Dict[str, Any]:
    """The properties to write back on the assertion the reference was read from.

    In ``"full"`` mode the field itself becomes the anchor's id, so the graph can follow
    it, and the text it used to hold moves to ``<field>_text`` -- but only if nothing is
    there yet, because a re-resolution must not overwrite the original wording with its
    own idea of it.

    In ``"resolution_only"`` mode only the ``<field>_resolution`` blob is written. That is
    what an abstention, a below-threshold answer and an inferred link all need: writing
    the field would either null out wording the extraction recorded (``anchor_id`` is
    ``None`` for every negative record) or make the graph claim the document stated a
    reference it never wrote.
    """
    field = resolution.field
    audit = {
        f"{field}_resolution": {
            "strategy": resolution.strategy,
            "confidence": resolution.confidence,
            "target_type": resolution.target_type,
            "target_ids": list(resolution.target_ids),
            "anchor_id": resolution.anchor_id,
            "document_id": resolution.document_id,
            "notes": list(resolution.notes),
            "reason": resolution.reason,
            "fingerprint": resolution.fingerprint,
            "iterations": resolution.iterations,
            "max_iter": resolution.max_iter,
            "trace": [dict(record) for record in resolution.trace],
        }
    }
    if mode == "resolution_only":
        return audit

    return {
        field: resolution.anchor_id,
        f"{field}_text": current_props.get(f"{field}_text") or resolution.reference_text,
        **audit,
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
