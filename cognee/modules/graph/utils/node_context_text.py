"""How one graph node is rendered into the text a prompt sees.

Every prompt renderer reads a node through ``node_context_text`` (a title and a body) or
``node_context_label`` (a one-line label for an edge line or a bullet), so a node reads the
same way whichever retriever surfaced it.

The reason this is shared rather than inlined per renderer: an ``Assertion``'s ``name`` is
the underlying proposition phrased AFFIRMATIVELY, and its ``polarity`` is the speaker's
stance on that proposition. Rendering ``name`` on its own -- which is what every renderer
did before this module existed -- hands a denial to the model as the fact it denies, and
the model then reports the pair as a contradiction. The stance has to be in the text.

Pure: no I/O, no database, no LLM. A node arrives as a mapping of already-projected
properties, which is all the graph-projection whitelist guarantees a renderer can see (see
``context_fields_for_datapoints``).
"""

import string
from collections import Counter
from typing import Any, Mapping, Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses

# Importing the stance vocabulary here also imports ``Assertion``, which is what registers
# it as a ``DataPoint`` subclass for ``context_fields_for_datapoints``.
from cognee.modules.graph.utils.reference_resolution import (
    STANCE_VERB_BY_POLARITY,
    UNNAMED_PROPOSITION,
    UNNAMED_SPEAKER,
    UNRECORDED_STANCE_VERB,
    as_sentence,
)
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS

UNNAMED_NODE = "Unnamed Node"
UNKNOWN_POLARITY = "unknown"
# ``UNNAMED_SPEAKER``/``UNNAMED_PROPOSITION`` are imported from ``reference_resolution``
# above rather than restated here: the stance sentence a prompt reads and the stance
# sentence stored on the edge have to word a missing speaker the same way.


def _get_top_n_frequent_words(
    text: str, stop_words: set = None, top_n: int = 3, separator: str = ", "
) -> str:
    """Concatenates the top N frequent words in text."""
    if stop_words is None:
        stop_words = DEFAULT_STOP_WORDS

    words = [word.lower().strip(string.punctuation) for word in text.split()]
    words = [word for word in words if word and word not in stop_words]

    top_words = [word for word, freq in Counter(words).most_common(top_n)]
    return separator.join(top_words)


def _create_title_from_text(text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
    """Creates a title by combining first words with most frequent words from the text."""
    first_words = text.split()[:first_n_words]
    top_words = _get_top_n_frequent_words(text, top_n=top_n_words)
    return f"{' '.join(first_words)}... [{top_words}]"


def _scalar_text(value: Any) -> Optional[str]:
    """A property as displayable text, or None when there is nothing to display.

    Same acceptance as the retrieval-side ``display_value``: a projected property is a
    scalar, and anything else (a dict, a list) has no place in a prompt line.
    """
    if isinstance(value, (str, int, float, bool, UUID)):
        text = str(value).strip()
        return text or None
    return None


def is_assertion_props(props: Mapping[str, Any]) -> bool:
    """True when projected properties carry a statement type.

    The same test ``is_assertion_node`` applies to an extracted node: one non-blank
    ``statement_type`` decides, and the node's ``type`` is never consulted (ontology
    canonicalization rewrites types, so reading the type would make an entity render as an
    assertion the moment an ontology is configured).
    """
    return _scalar_text(props.get("statement_type")) is not None


def node_context_text(props: Mapping[str, Any]) -> tuple[str, str]:
    """``(title, body)`` for one node: its header line and the content under it."""
    statement_type = _scalar_text(props.get("statement_type"))
    if statement_type is not None:
        return _assertion_context_text(props, statement_type)

    name = _scalar_text(props.get("name"))
    text = _scalar_text(props.get("text"))
    if text:
        return _create_title_from_text(text), text

    title = name or UNNAMED_NODE
    return title, _scalar_text(props.get("description")) or title


def node_context_label(props: Mapping[str, Any]) -> str:
    """A node on one line, for an edge line or a bullet. Empty when there is nothing to show."""
    name = _scalar_text(props.get("name"))
    statement_type = _scalar_text(props.get("statement_type"))
    node_id = _scalar_text(props.get("id"))
    if statement_type is not None:
        polarity = _scalar_text(props.get("polarity")) or UNKNOWN_POLARITY
        # The id fallback is the same one every other node gets: a nameless node still
        # has to be identifiable in the line that mentions it.
        return f"[{statement_type}/{polarity}] {name or node_id or UNNAMED_NODE}"

    return name or node_id or ""


def _assertion_context_text(props: Mapping[str, Any], statement_type: str) -> tuple[str, str]:
    """Title and body of an assertion, with the speaker and the stance stated outright."""
    name = _scalar_text(props.get("name"))
    speaker = _scalar_text(props.get("asserted_by")) or UNNAMED_SPEAKER
    polarity = _scalar_text(props.get("polarity")) or UNKNOWN_POLARITY

    title = f"[{statement_type} by {speaker}; stance: {polarity}] {name or UNNAMED_NODE}"

    stance_verb = STANCE_VERB_BY_POLARITY.get(polarity, UNRECORDED_STANCE_VERB)
    clause = (name or "").rstrip(".").strip() or UNNAMED_PROPOSITION
    lines = [as_sentence(f"{speaker} {stance_verb} {clause}")]

    quote = _scalar_text(props.get("source_quote"))
    if quote:
        verified = " (verified)" if props.get("source_quote_verified") else ""
        lines.append(f'Quote: "{quote}"{verified}')

    description = _scalar_text(props.get("description"))
    if description and description != name:
        lines.append(description)

    return title, "\n".join(lines)


def context_fields_for_datapoints() -> list[str]:
    """Every property name a ``DataPoint`` subclass declares as needed for rendering.

    A node reaches a renderer as a projection whitelist, so a property nobody asked for is
    not merely unrendered -- it is absent. A subclass whose text cannot be rendered from
    ``name``/``description`` alone declares the properties it needs in
    ``metadata["context_fields"]`` and the graph projection unions them in.
    """
    fields: list[str] = []
    for subclass in get_all_subclasses(DataPoint):
        metadata_field = subclass.model_fields.get("metadata")
        default = getattr(metadata_field, "default", None)
        if not isinstance(default, dict):
            continue
        for field_name in default.get("context_fields") or []:
            if isinstance(field_name, str) and field_name not in fields:
                fields.append(field_name)
    return fields
