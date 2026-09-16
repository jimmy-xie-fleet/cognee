"""Pure helpers for the labelled candidates the reference tracer sees.

A graph node reaches an LLM as an opaque label (``A1``/``P3``/``D2``) rather than as a node
id, and a ``LabelRegistry`` is the only place that mapping is kept.

Everything here is pure: no I/O, no database, no LLM, no clock, no randomness, and no
import of anything under ``cognee.domains``.
"""

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

CANDIDATE_PREVIEW_CHARS = 240

# Label prefix per node type. ``Document`` and every subtype name ending in ``Document``
# (e.g. ``LegalDocument``) share the ``D`` prefix; anything else falls back to ``N``.
_LABEL_PREFIX_BY_TYPE = {
    "Assertion": "A",
    "DocumentChunk": "P",
}
_DEFAULT_LABEL_PREFIX = "N"
_DOCUMENT_LABEL_PREFIX = "D"

# How a candidate line names the node's kind. Anything unlisted renders as a document; a
# ``TextSummary`` is never a candidate, because seed retrieval folds it onto its chunk.
_TYPE_WORD_BY_NODE_TYPE = {
    "Assertion": "Assertion",
    "DocumentChunk": "Passage",
}

_WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------------------
# Candidate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One labelled, scored graph node offered to the tracer or folded into a seed."""

    label: str
    node_id: str
    node_type: str
    score: float
    text: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    sources: Tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# LabelRegistry
# --------------------------------------------------------------------------------------


def _label_prefix(node_type: str) -> str:
    if node_type in _LABEL_PREFIX_BY_TYPE:
        return _LABEL_PREFIX_BY_TYPE[node_type]
    if node_type.endswith("Document"):
        return _DOCUMENT_LABEL_PREFIX
    return _DEFAULT_LABEL_PREFIX


class LabelRegistry:
    """Per-trace opaque labels the agent sees (``A1``, ``P3``, ``D2``) -> real node ids.

    This is the ONLY way a node id enters an LLM finish: the tracer hands the model labels,
    never ids, and ``resolve`` is the sole path back. ``label(node_id, node_type)`` is
    stable within a trace, whatever ``node_type`` a later call passes; numbering is
    contiguous per prefix, starting at 1, in first-seen order.
    """

    def __init__(self) -> None:
        self._label_to_node_id: Dict[str, str] = {}
        self._node_id_to_label: Dict[str, str] = {}
        self._node_type_by_label: Dict[str, str] = {}
        self._next_number_by_prefix: Dict[str, int] = {}

    def label(self, node_id: str, node_type: str) -> str:
        existing = self._node_id_to_label.get(node_id)
        if existing is not None:
            return existing

        prefix = _label_prefix(node_type)
        number = self._next_number_by_prefix.get(prefix, 1)
        self._next_number_by_prefix[prefix] = number + 1

        new_label = f"{prefix}{number}"
        self._label_to_node_id[new_label] = node_id
        self._node_id_to_label[node_id] = new_label
        self._node_type_by_label[new_label] = node_type
        return new_label

    def resolve(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        normalized = label.strip().upper()
        return self._label_to_node_id.get(normalized)

    def node_type(self, label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        normalized = label.strip().upper()
        return self._node_type_by_label.get(normalized)

    def labels(self) -> Dict[str, str]:
        return dict(self._label_to_node_id)


# --------------------------------------------------------------------------------------
# Score conversion
# --------------------------------------------------------------------------------------


def distance_to_similarity(score: float) -> float:
    """Convert a LanceDB cosine distance (lower is better) to a similarity in [0, 1]."""

    return max(0.0, min(1.0, 1 - score))


# --------------------------------------------------------------------------------------
# merge_candidates
# --------------------------------------------------------------------------------------


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _preview(text: str) -> str:
    collapsed = _collapse_whitespace(text)
    if len(collapsed) <= CANDIDATE_PREVIEW_CHARS:
        return collapsed
    return collapsed[: CANDIDATE_PREVIEW_CHARS - 1] + "…"


def merge_candidates(
    scored: Iterable[Tuple[str, str, float, str, dict]],
    *,
    limit: int,
    registry: LabelRegistry,
) -> List[Candidate]:
    """Union scored hits by node id and turn the top ``limit`` into labelled candidates.

    A node id seen more than once keeps its best similarity and accumulates every distinct
    ``source_tag``. Results are sorted by ``(-score, node_id)`` for a deterministic order,
    then truncated *before* labelling, so a dropped candidate never consumes a label.
    """

    best_score: Dict[str, float] = {}
    node_type_by_id: Dict[str, str] = {}
    sources_by_id: Dict[str, List[str]] = {}
    payload_by_id: Dict[str, dict] = {}

    for node_id, node_type, similarity, source_tag, payload in scored:
        if node_id not in best_score or similarity > best_score[node_id]:
            best_score[node_id] = similarity
        node_type_by_id[node_id] = node_type
        payload_by_id.setdefault(node_id, payload)

        sources = sources_by_id.setdefault(node_id, [])
        if source_tag not in sources:
            sources.append(source_tag)

    ordered_ids = sorted(best_score.keys(), key=lambda node_id: (-best_score[node_id], node_id))
    truncated_ids = ordered_ids[:limit]

    candidates: List[Candidate] = []
    for node_id in truncated_ids:
        node_type = node_type_by_id[node_id]
        payload = payload_by_id[node_id]
        candidates.append(
            Candidate(
                label=registry.label(node_id, node_type),
                node_id=node_id,
                node_type=node_type,
                score=best_score[node_id],
                text=_preview(payload.get("text", "")),
                document_id=payload.get("document_id"),
                document_name=payload.get("document_name"),
                chunk_index=payload.get("chunk_index"),
                sources=tuple(sources_by_id[node_id]),
            )
        )
    return candidates


# --------------------------------------------------------------------------------------
# apply_penalty
# --------------------------------------------------------------------------------------


def apply_penalty(
    candidates: Sequence[Candidate], node_ids: Set[str], amount: float
) -> List[Candidate]:
    """Subtract ``amount`` (floor 0) from every candidate whose id is in ``node_ids``.

    Labels are untouched -- they belong to the registry, not to the ordering -- but the list
    is re-sorted by ``(-score, node_id)`` afterwards.
    """

    penalized = [
        replace(candidate, score=max(0.0, candidate.score - amount))
        if candidate.node_id in node_ids
        else candidate
        for candidate in candidates
    ]
    penalized.sort(key=lambda candidate: (-candidate.score, candidate.node_id))
    return penalized


# --------------------------------------------------------------------------------------
# format_candidate_lines
# --------------------------------------------------------------------------------------


def _document_label(candidate: Candidate) -> str:
    if candidate.document_name is not None:
        return candidate.document_name
    if candidate.document_id is not None:
        return candidate.document_id
    return "unknown document"


def _single_line_text(candidate: Candidate) -> str:
    # Defensive, not merely documentary: a Candidate can be constructed directly, so this
    # function must not trust the caller. Re-collapsing also neutralises a bare "\r", which
    # str.splitlines() and many renderers break on just as they do on "\n".
    text = _collapse_whitespace(candidate.text)
    if "\n" in text or "\r" in text:
        # Unreachable through _collapse_whitespace's \s+ regex; kept as a raised error
        # (not a bare assert) so the contract survives python -O / PYTHONOPTIMIZE.
        raise ValueError("candidate text must render on a single line after collapsing")
    return text


def _format_candidate_line(candidate: Candidate) -> str:
    text = _single_line_text(candidate)

    if candidate.node_type in _TYPE_WORD_BY_NODE_TYPE:
        type_word = _TYPE_WORD_BY_NODE_TYPE[candidate.node_type]
        chunk_suffix = (
            f" (chunk {candidate.chunk_index})" if candidate.chunk_index is not None else ""
        )
        return (
            f'[{candidate.label}] {type_word} in "{_document_label(candidate)}"'
            f'{chunk_suffix}: "{text}"'
        )

    return f'[{candidate.label}] Document "{_document_label(candidate)}": "{text}"'


def format_candidate_lines(candidates: Iterable[Candidate]) -> str:
    """Render one numbered line per candidate, newline-joined.

    Assertions and passages render as ``[A7] Assertion in "<document>" (chunk 4): "<text>"``
    (the ``(chunk N)`` clause is omitted when ``chunk_index`` is ``None``); every other node
    type renders as ``[D2] Document "<document>": "<text>"``.
    """

    return "\n".join(_format_candidate_line(candidate) for candidate in candidates)


# --------------------------------------------------------------------------------------
# candidate_set_key
# --------------------------------------------------------------------------------------


def candidate_set_key(candidates: Iterable[Candidate]) -> str:
    """A stable, order-independent key for a set of candidates, keyed on node id."""

    node_ids = sorted(candidate.node_id for candidate in candidates)
    digest = hashlib.sha1(",".join(node_ids).encode("utf-8")).hexdigest()
    return digest[:16]
