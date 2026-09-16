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
from functools import lru_cache
from typing import Any, Mapping, Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint

# Importing ``Assertion`` is also what registers it as a ``DataPoint`` subclass, so
# ``context_fields_for_datapoints`` sees its declared context fields.
from cognee.modules.engine.models.Assertion import Assertion
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
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


def _is_true(value: Any) -> bool:
    """A projected boolean, which may have made a round trip through a string store.

    Neo4j and the Postgres demo adapter serialize JSON properties, so a ``False`` can come
    back as the string ``"false"`` -- non-empty, and therefore truthy. Only a real ``True``
    or the word "true" counts; anything else is not a claim that something was verified.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() == "true"
    return False


def _names_an_assertion_class(node_type: str) -> bool:
    """Whether ``node_type`` is the class name of ``Assertion`` or of a subclass of it.

    ``get_graph_from_model`` stores the DataPoint *class name* in ``type``. Deliberately
    uncached: the overwhelmingly common answer is the first comparison, and a subclass
    registered after a cache was warmed would be missing from it with nothing to say so.
    The walk is only reached by a node that carries a statement type under some other
    class, which is the case this test exists to reject.
    """
    if node_type == Assertion.__name__:
        return True
    return any(subclass.__name__ == node_type for subclass in get_all_subclasses(Assertion))


def is_assertion_props(props: Mapping[str, Any]) -> bool:
    """True when projected properties carry a statement type from an assertion node.

    One non-blank ``statement_type`` is what decides, exactly as in ``is_assertion_node``:
    a caller does not always have a projected node to read (the hybrid lane builds props
    from a vector payload), so a blank or absent ``type`` falls back to that field alone.

    When a ``type`` *is* present it has to name an assertion class. The type is not
    consulted to *promote* a node -- ontology canonicalization rewrites an entity's
    ``is_a``, and reading that would turn entities into assertions the moment an ontology
    is configured -- but ``type`` holds the Python class name, which canonicalization never
    touches, so it can veto. Without the veto any third-party ``DataPoint`` that happens to
    declare a field called ``statement_type`` renders stance-first and drags a
    ``get_neighborhood`` call behind it.
    """
    if _scalar_text(props.get("statement_type")) is None:
        return False

    node_type = _scalar_text(props.get("type"))
    return node_type is None or _names_an_assertion_class(node_type)


def node_context_text(props: Mapping[str, Any]) -> tuple[str, str]:
    """``(title, body)`` for one node: its header line and the content under it."""
    if is_assertion_props(props):
        return _assertion_context_text(props, _scalar_text(props.get("statement_type")))

    name = _scalar_text(props.get("name"))
    text = _scalar_text(props.get("text"))
    if text:
        return _create_title_from_text(text), text

    title = name or UNNAMED_NODE
    return title, _scalar_text(props.get("description")) or title


def node_context_label(props: Mapping[str, Any]) -> str:
    """A node on one line, for an edge line or a bullet. Empty when there is nothing to show."""
    name = _scalar_text(props.get("name"))
    node_id = _scalar_text(props.get("id"))
    if is_assertion_props(props):
        statement_type = _scalar_text(props.get("statement_type"))
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
        verified = " (verified)" if _is_true(props.get("source_quote_verified")) else ""
        lines.append(f'Quote: "{quote}"{verified}')

    description = _scalar_text(props.get("description"))
    if description and description != name:
        lines.append(description)

    return title, "\n".join(lines)


@lru_cache(maxsize=1)
def _declared_context_fields() -> tuple:
    """The declared context fields, computed once by walking every ``DataPoint`` subclass."""
    fields: list[str] = []
    for subclass in get_all_subclasses(DataPoint):
        metadata_field = subclass.model_fields.get("metadata")
        default = getattr(metadata_field, "default", None)
        if not isinstance(default, dict):
            continue
        for field_name in default.get("context_fields") or []:
            if isinstance(field_name, str) and field_name not in fields:
                fields.append(field_name)
    return tuple(fields)


def context_fields_for_datapoints() -> list[str]:
    """Every property name a ``DataPoint`` subclass declares as needed for rendering.

    A node reaches a renderer as a projection whitelist, so a property nobody asked for is
    not merely unrendered -- it is absent. A subclass whose text cannot be rendered from
    ``name``/``description`` alone declares the properties it needs in
    ``metadata["context_fields"]`` and the graph projection unions them in.

    The subclass walk is cached, because this is on the projection path of every search
    rather than a startup step. A caller that registers a subclass after the first call has
    to ``context_fields_for_datapoints.cache_clear()``. The list is rebuilt per call, so a
    caller that mutates the result cannot corrupt the cache.
    """
    return list(_declared_context_fields())


# The cache lives on the private tuple-returning function so that each public call still
# hands back a fresh list; the clearing seam is published under the name callers know.
context_fields_for_datapoints.cache_clear = _declared_context_fields.cache_clear
