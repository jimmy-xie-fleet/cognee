"""The five read-only tools the agentic reference tracer may call.

A trace resolves exactly one reference, and the only handle it ever has on a graph node
is an opaque label (``A3``, ``P2``, ``D1``) issued by the trace's shared
:class:`LabelRegistry`. These tools are what turn that constraint into something workable:
they let the agent look around the document set -- search it, list it, page through a
document, read one passage in full, jump to a numbered paragraph -- and every node they
mention comes back labelled, never as an id. A finish naming a label is resolvable; a
finish naming anything else is not, which is exactly the fence we want around an LLM
writing edges into a graph.

Design notes:

* **Read-only, always.** No handler writes a node, an edge or a file. The one filesystem
  read (a document's stored text, for ``locate_paragraph``) goes through the pass's
  :class:`DocumentTextCache`, which opens a document at most once and degrades to the
  stored chunks when it cannot.
* **Private callables, not registry ``Tool``s** (decision D7): nothing here is
  discoverable by ``AGENTIC_COMPLETION`` or any other search path, and nothing here
  needs a permission check of its own -- the view it closes over was already read under
  the caller's dataset scope.
* **No regex over reference text.** ``locate_paragraph`` takes an already-split
  ``(kind, value)`` pair from the agent and hands it to :func:`build_locator`; the marker
  regexes then run over *document* text only.
* ``search`` is the same :func:`search_candidates` that produced the trace's seed, so the
  agent and the seed see one ranking rather than two.
"""

import json
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    get_args,
)

from pydantic import BaseModel, Field, ValidationError

from cognee.modules.graph.utils.reference_candidates import (
    CANDIDATE_PREVIEW_CHARS,
    LabelRegistry,
    format_candidate_lines,
)
from cognee.modules.graph.utils.reference_resolution import (
    Locator,
    anchor_chunk_index,
    build_locator,
    chunks_overlapping,
    find_locator_span,
    scan_chunks_for_marker,
    select_anchored_assertions,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.reference_graph_view import DocumentTextCache, GraphView, _chunk_index
from cognee.tasks.graph.reference_retrieval import KINDS, LexicalIndex, search_candidates

logger = get_logger("reference_tracer_tools")

# The ceiling on a single tool's output. The loop truncates to it as well; ``read_chunk``
# applies it to the passage body before it appends the assertion list, so a huge passage
# cannot squeeze the assertions out of the result entirely.
MAX_TOOL_OUTPUT_CHARS = 6_000
# ``list_documents`` is a whole-corpus listing; past this many documents the agent should
# be searching, not reading a catalogue.
DOCUMENT_LIST_CAP = 80
# How much of a located span to show. Smaller than MAX_TOOL_OUTPUT_CHARS because a span
# is one paragraph and the passages/assertions under it are the point of the call.
LOCATOR_SPAN_CHARS = 2_000

TOOL_NAMES = ("search", "list_documents", "open_document", "read_chunk", "locate_paragraph")

SearchKind = Literal["any", "assertions", "passages", "documents"]
if set(get_args(SearchKind)) != set(KINDS):  # pragma: no cover - drift guard
    raise RuntimeError(
        f"SearchKind {get_args(SearchKind)} drifted from reference_retrieval {KINDS}"
    )


# --------------------------------------------------------------------------- #
# tool specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolSpec:
    """One callable tool: what the model is told about it, and what runs.

    ``handler`` takes the *validated* ``args_model`` instance, so a handler never sees a
    raw dict off the wire -- :func:`run_tool` validates first and turns a rejection into
    an ``ERROR:`` string the agent can read and correct.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Awaitable[str]]


class SearchArgs(BaseModel):
    query: str = Field(
        ...,
        description="What to look for, in the words the documents would use.",
    )
    kind: SearchKind = Field(
        "any",
        description=(
            "Restrict the search: 'assertions' for statements, 'passages' for chunks of "
            "document text, 'documents' to find a document by what it contains, 'any' for "
            "all three."
        ),
    )
    top_k: int = Field(8, ge=1, le=12, description="How many results to return.")


class ListDocumentsArgs(BaseModel):
    """No arguments."""


class OpenDocumentArgs(BaseModel):
    document: str = Field(..., description="A document label shown in this trace, e.g. 'D2'.")
    from_chunk: int = Field(0, ge=0, description="The passage index to start at.")
    count: int = Field(3, ge=1, le=5, description="How many consecutive passages to return.")


class ReadChunkArgs(BaseModel):
    passage: str = Field(..., description="A passage label shown in this trace, e.g. 'P3'.")


class LocateParagraphArgs(BaseModel):
    document: str = Field(..., description="A document label shown in this trace, e.g. 'D2'.")
    kind: Literal["paragraph", "section", "exhibit", "count", "article"] = Field(
        ..., description="What kind of numbered marker to jump to."
    )
    value: str = Field(..., description="The number or letter as the reference wrote it.")


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def _preview(text: Optional[str], limit: int = CANDIDATE_PREVIEW_CHARS) -> str:
    """One line, whitespace-collapsed, cut to ``limit`` characters."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _document_name(view: GraphView, document_id: str) -> str:
    props = view.documents.get(document_id, {})
    name = props.get("name")
    return name if isinstance(name, str) and name.strip() else document_id


def _assertion_line(registry: LabelRegistry, assertion_id: str, props: dict) -> str:
    label = registry.label(assertion_id, "Assertion")
    statement_type = props.get("statement_type") or "statement"
    polarity = props.get("polarity") or "unknown"
    name = _preview(props.get("name"))
    return f"[{label}] {statement_type}/{polarity}: {name}"


def _assertion_section(
    view: GraphView, registry: LabelRegistry, header: str, assertion_ids: Sequence[str]
) -> str:
    if not assertion_ids:
        return f"{header}\n(none)"
    lines = [
        _assertion_line(registry, assertion_id, view.assertions.get(assertion_id, {}))
        for assertion_id in assertion_ids
    ]
    return "\n".join([header, *lines])


def _assertions_in_chunks(view: GraphView, chunk_ids: Set[str]) -> List[Tuple[str, dict]]:
    """Every assertion quoted from one of ``chunk_ids``, in a stable order."""
    found = [
        (assertion_id, props)
        for assertion_id, props in view.assertions.items()
        if str(props.get("source_chunk_id") or "") in chunk_ids
    ]
    found.sort(key=lambda item: item[0])
    return found


def _resolve_document(
    view: GraphView, registry: LabelRegistry, label: str
) -> Tuple[Optional[str], Optional[str]]:
    """``(document id, None)`` or ``(None, error string)`` for a document label."""
    node_id = registry.resolve(label)
    if node_id is None or node_id not in view.documents:
        return None, f"ERROR: unknown document label {label}. Call list_documents to see them."
    return node_id, None


# --------------------------------------------------------------------------- #
# build_tracer_tools
# --------------------------------------------------------------------------- #


def build_tracer_tools(
    *,
    view: GraphView,
    texts: DocumentTextCache,
    lexical: LexicalIndex,
    registry: LabelRegistry,
    vector_engine=None,
    exclude_ids: Set[str] = frozenset(),
    own_document_id: Optional[str] = None,
    penalize_own_document: bool = False,
) -> Dict[str, ToolSpec]:
    """The five tools, closed over one trace's view, text cache, index and registry.

    ``exclude_ids`` / ``own_document_id`` / ``penalize_own_document`` are the trace's own
    scoping (drop the referring assertion and its chunk; weigh the referring document
    down for a ``responds_to``), passed straight through to :func:`search_candidates` so
    the agent's ``search`` ranks exactly the way the seed did.
    """

    async def _search(args: SearchArgs) -> str:
        candidates = await search_candidates(
            queries=[args.query],
            kind=args.kind,
            view=view,
            lexical=lexical,
            registry=registry,
            exclude_ids=exclude_ids,
            own_document_id=own_document_id,
            penalize_own_document=penalize_own_document,
            limit=args.top_k,
            vector_engine=vector_engine,
        )
        return format_candidate_lines(candidates) or "No matches."

    async def _list_documents(args: ListDocumentsArgs) -> str:
        ordered = sorted(
            view.documents.items(), key=lambda item: (_document_name(view, item[0]), item[0])
        )
        if not ordered:
            return "No documents."

        lines = []
        for document_id, props in ordered[:DOCUMENT_LIST_CAP]:
            label = registry.label(document_id, props.get("type") or "Document")
            chunks = view.chunks_by_document.get(document_id, [])
            first = _preview(chunks[0].get("text") if chunks else "")
            lines.append(
                f"[{label}] {_document_name(view, document_id)} — {len(chunks)} passages — "
                f'"{first}"'
            )

        remaining = len(ordered) - DOCUMENT_LIST_CAP
        if remaining > 0:
            lines.append(f"… and {remaining} more")
        return "\n".join(lines)

    async def _open_document(args: OpenDocumentArgs) -> str:
        document_id, error = _resolve_document(view, registry, args.document)
        if error is not None:
            return error

        chunks = view.chunks_by_document.get(document_id, [])
        # from_chunk is the stored chunk_index the agent read off a previous result, not a
        # position in this list: the two coincide for a normally ingested document, and
        # when they do not it is the printed number the agent is answering.
        start = next(
            (
                position
                for position, props in enumerate(chunks)
                if _chunk_index(props) >= args.from_chunk
            ),
            len(chunks),
        )
        window = chunks[start : start + args.count]
        if not window:
            return (
                f'No passages in "{_document_name(view, document_id)}" at or after chunk '
                f"{args.from_chunk} (it has {len(chunks)})."
            )

        lines = []
        for props in window:
            label = registry.label(str(props.get("id")), "DocumentChunk")
            lines.append(
                f'[{label}] (chunk {_chunk_index(props)}): "{_preview(props.get("text"))}"'
            )
        return "\n".join(lines)

    async def _read_chunk(args: ReadChunkArgs) -> str:
        node_id = registry.resolve(args.passage)
        if node_id is None or node_id not in view.chunks:
            return f"ERROR: unknown passage label {args.passage}. Call open_document to see them."

        text = view.chunks[node_id].get("text") or ""
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[: MAX_TOOL_OUTPUT_CHARS - 1] + "…"

        quoted = _assertions_in_chunks(view, {node_id})
        section = _assertion_section(
            view,
            registry,
            "Assertions quoted in this passage:",
            [assertion_id for assertion_id, _ in quoted],
        )
        return f"{text}\n\n{section}"

    async def _locate_paragraph(args: LocateParagraphArgs) -> str:
        document_id, error = _resolve_document(view, registry, args.document)
        if error is not None:
            return error

        locator = build_locator(args.kind, args.value)
        if locator is None:
            return (
                f'ERROR: "{args.value}" is not a usable {args.kind} number; '
                "give the number or letter exactly as the reference wrote it."
            )

        chunks = view.chunks_by_document.get(document_id, [])
        located = await _locate(texts, document_id, chunks, locator)
        if located is None:
            return (
                f"ERROR: marker not found for {args.kind} {args.value} in "
                f'"{_document_name(view, document_id)}".'
            )

        span_text, chunk_positions, anchor_position = located
        passages = []
        for position in chunk_positions:
            props = chunks[position]
            label = registry.label(str(props.get("id")), "DocumentChunk")
            suffix = ", anchor" if position == anchor_position else ""
            passages.append(f"[{label}] (chunk {_chunk_index(props)}{suffix})")

        chunk_ids = {str(chunks[position].get("id")) for position in chunk_positions}
        anchored = select_anchored_assertions(
            span_text,
            [
                (assertion_id, props.get("source_quote"))
                for assertion_id, props in _assertions_in_chunks(view, chunk_ids)
            ],
        )
        section = _assertion_section(view, registry, "Assertions quoted in the span:", anchored)
        return (
            f"{_span_preview(span_text)}\n\nPassages: {', '.join(passages) or '(none)'}\n{section}"
        )

    specs = (
        ToolSpec(
            name="search",
            description=(
                "Search the document set by meaning and by exact wording. Returns labelled "
                "assertions, passages and documents. Use kind='documents' to find a document "
                "by what it contains -- that is the only way to reach a document whose "
                "filename says nothing."
            ),
            args_model=SearchArgs,
            handler=_search,
        ),
        ToolSpec(
            name="list_documents",
            description=(
                "List every document with its label, how many passages it has, and the start "
                "of its first passage."
            ),
            args_model=ListDocumentsArgs,
            handler=_list_documents,
        ),
        ToolSpec(
            name="open_document",
            description=(
                "Return consecutive passages of one document as labelled previews, so a "
                "passage can be picked out and read in full."
            ),
            args_model=OpenDocumentArgs,
            handler=_open_document,
        ),
        ToolSpec(
            name="read_chunk",
            description=("Return one passage in full, followed by the assertions quoted from it."),
            args_model=ReadChunkArgs,
            handler=_read_chunk,
        ),
        ToolSpec(
            name="locate_paragraph",
            description=(
                "Jump to a numbered marker in one document (paragraph 5, section 3.2, "
                "exhibit B) and return the text there with its passages and the assertions "
                "quoted inside it. This is how a positional reference is checked."
            ),
            args_model=LocateParagraphArgs,
            handler=_locate_paragraph,
        ),
    )
    tools = {spec.name: spec for spec in specs}
    if tuple(tools) != TOOL_NAMES:  # pragma: no cover - drift guard
        raise RuntimeError(f"tracer tools {tuple(tools)} drifted from TOOL_NAMES {TOOL_NAMES}")
    return tools


def _span_preview(span_text: str) -> str:
    if len(span_text) <= LOCATOR_SPAN_CHARS:
        return span_text
    return span_text[: LOCATOR_SPAN_CHARS - 1] + "…"


async def _locate(
    texts: DocumentTextCache,
    document_id: str,
    chunks: Sequence[dict],
    locator: Locator,
) -> Optional[Tuple[str, List[int], Optional[int]]]:
    """``(span text, chunk positions, anchor position)`` for a locator, or None.

    Two paths, in the order the resolver has always used them: the document's stored text
    with chunk offsets when both are available, and a chunk-by-chunk scan when they are
    not (a PDF that cannot be read as text, chunks that no longer tile the document).
    ``chunk positions`` index ``chunks``, not the stored ``chunk_index``.
    """
    text = await texts.text(document_id)
    offsets = await texts.offsets(document_id, chunks) if text is not None else None

    if text is not None and offsets is not None:
        span = find_locator_span(text, locator)
        if span is None:
            return None
        start, end, _notes = span
        positions = chunks_overlapping(offsets, (start, end))
        return text[start:end], positions, anchor_chunk_index(offsets, (start, end))

    scanned = scan_chunks_for_marker([chunk.get("text") or "" for chunk in chunks], locator)
    if scanned is None:
        return None
    position, (start, end) = scanned
    return (chunks[position].get("text") or "")[start:end], [position], position


# --------------------------------------------------------------------------- #
# manifest + dispatch
# --------------------------------------------------------------------------- #


def render_tool_manifest(tools: Mapping[str, ToolSpec]) -> str:
    """The tool list the agent sees: name, description and full argument JSON schema.

    The schema is what makes the manifest usable -- types, required fields and the ``kind``
    enum reach the model instead of being described in prose it has to guess at.
    """
    if not tools:
        return "(no tools)"

    blocks = []
    for spec in tools.values():
        schema = json.dumps(
            spec.args_model.model_json_schema(), sort_keys=True, separators=(",", ":")
        )
        blocks.append(f"- `{spec.name}`: {spec.description}\n  arguments: {schema}")
    return "\n".join(blocks)


async def run_tool(tools: Mapping[str, ToolSpec], name: str, arguments: Any) -> str:
    """Validate and run one tool call, turning every failure into an ``ERROR:`` string.

    Never raises: an unknown tool, arguments the model got wrong, and a handler that blew
    up all come back as text the agent can read on its next step. That is the whole
    contract the loop relies on -- a tool step is never fatal to a trace.
    """
    spec = tools.get(name)
    if spec is None:
        return f"ERROR: unknown tool {name}. Available tools: {', '.join(tools)}."

    try:
        args = spec.args_model.model_validate(arguments or {})
    except ValidationError as error:
        return f"ERROR: invalid arguments for {name}: {error}"

    try:
        result = await spec.handler(args)
    except Exception as error:  # a tool must never end a trace
        logger.warning("Reference tracer tool %s failed: %s", name, error)
        return f"ERROR: {name} failed: {error}"

    return result if isinstance(result, str) else str(result)
