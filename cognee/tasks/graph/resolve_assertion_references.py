"""Turn the references an ``Assertion`` carries into graph edges.

Extraction records a reference as a structured hint --
``responds_to_ref={"document_hint": "the Complaint", "locator_kind": "paragraph",
"locator_value": "5", "basis": "positional"}`` -- or, on older data, as the string the
document wrote, and no graph edge can follow either. This task reads the graph, runs the
cascade below over every dangling reference, and writes the answer as edges plus a node
patch recording how it was reached.

The cascade, per ``(assertion, field)``:

1. ``existing_id`` -- the field already holds a node id (or a stale one to re-resolve).
2. ``entity_name`` -- the reference names one ``Entity`` ("Norman Fester"). Gated: a hint
   that points *inside* a document ("the Complaint ¶5") never reaches this step, so it
   cannot be linked to a ``Complaint`` stub entity.
3. the **attempt guard** -- a previous pass already traced this exact reference.
4. the **seed** -- one vector + BM25 retrieval per reference, no LLM, producing a
   labelled shortlist.
5. the **agentic tracer** (decision D3) -- a bounded loop of ``TracerStep``s over five
   read-only tools, ending in a finish naming one label, or an abstention.

Steps 4 and 5 run **only** in the ``improve()``/memify pass (decision D1). The ingest tail
runs with ``allow_llm=False`` and stops after step 2: a forward reference stays dangling,
with nothing written, until the pass picks it up. Nothing here parses or scores the
reference's own text (decision D5) -- deciding which document "the Whitfield rebuttal
appraisal" names is the agent's job, not a regex's.

Three entry points over one pass:

* :func:`resolve_assertion_references` -- the cognify tail. Appended to a pipeline it
  resolves the references the ingestion touched, returns its input unchanged, and swallows
  its own errors, so reference resolution can never break ingestion.
* :func:`detect_dangling_references` / :func:`apply_reference_resolutions` -- the two-phase
  memify pair behind the ``resolve_references`` pipeline. The apply phase deliberately does
  **not** swallow write failures: a memify run that could not write is a visible error.
"""

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import NAMESPACE_URL, uuid5

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.write_context import graph_provenance_write_kwargs
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.engine.utils.generate_node_name import generate_node_name
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.modules.graph.utils.reference_candidates import (
    Candidate,
    LabelRegistry,
    candidate_set_key,
)
from cognee.modules.graph.utils.reference_resolution import (
    STRATEGY_ENTITY_NAME,
    STRATEGY_EXISTING_ID,
    STRATEGY_LLM_INFERRED,
    STRATEGY_LLM_TRACE,
    ReferenceHint,
    Resolution,
    build_locator,
    build_node_patch,
    build_reference_edge,
    parse_reference_hint,
    reference_display_text,
    reference_fingerprint,
    select_anchored_assertions,
)
from cognee.modules.operations.usage_accumulator import operation_usage_scope
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.reference_graph_view import (
    DOCUMENT_NODE_TYPES,
    VIEW_NODE_TYPES,
    DocumentTextCache,
    GraphView,
    _as_uuid,
    _chunk_index,
    _load_graph_view,
    _node_label,
    _raw_locations,
    _read_processed_text,
    _text_of,
)
from cognee.tasks.graph.reference_retrieval import (
    DOCUMENT_K,
    SEED_LIMIT,
    LexicalIndex,
    search_candidates,
)
from cognee.tasks.graph.reference_tracer import (
    TRACE_SYSTEM_PROMPT,
    CallBudget,
    TraceRecord,
    TracerFinish,
    trace_reference,
)
from cognee.tasks.graph.reference_tracer_tools import _locate, build_tracer_tools
from cognee.tasks.storage.index_graph_edges import index_graph_edges

logger = get_logger("resolve_assertion_references")

# ``reference_graph_view`` owns the graph-reading layer (GraphView, DocumentTextCache,
# _load_graph_view, and the small helpers they use). It is imported back and re-exported
# here so existing callers keep one import site: scripts/legal/resolve_references_report.py
# imports DocumentTextCache/_load_graph_view from this module, and
# scripts/legal/find_disputes.py imports DOCUMENT_NODE_TYPES from it. See
# cognee/tests/unit/tasks/graph/test_reference_graph_view_exports.py for the contract.
__all__ = [
    "GraphView",
    "DocumentTextCache",
    "_load_graph_view",
    "DOCUMENT_NODE_TYPES",
    "VIEW_NODE_TYPES",
    "_read_processed_text",
    "_raw_locations",
    "_text_of",
    "_as_uuid",
    "_node_label",
    "_chunk_index",
]

# The reference fields an assertion carries. ``asserted_by`` is deliberately absent: it is
# an identity field, and rewriting it would give the assertion a new node id. The
# structured hint lives beside each of these as ``<field>_ref`` and is never itself a
# reference field.
REFERENCE_FIELDS = ("responds_to", "attributed_to")

# Owner of record for edges written outside an ingestion, where no ``Data`` row is in
# context. At ingest the pipeline's own data item wins, so the edges vanish with the
# document's forget(); in memify this sentinel keeps the write attributable.
REFERENCE_RESOLUTION_DATA_ID = uuid5(NAMESPACE_URL, "cognee:reference-resolution")

# Notes a resolution carries out of the cascade, into ``<field>_resolution`` and the
# write summary.
# The field held an id no longer in the graph -- a forgotten or re-chunked target -- so
# the reference was re-resolved from the wording the resolver preserved.
NOTE_STALE_ID = "stale_id"
# Every edge this resolution would write is already in the graph; only the node patch is
# still outstanding, so the write phase patches and skips the edge upsert.
NOTE_EDGES_EXIST = "edges_exist"
# ``add_edges`` succeeded but ``index_graph_edges`` did not.
NOTE_EDGE_INDEX_FAILED = "edge_index_failed"
# The agent looked and found nothing it would link.
NOTE_LLM_ABSTAINED = "llm_abstained"
# The agent named a candidate but was less sure than the configured threshold.
NOTE_LLM_BELOW_THRESHOLD = "llm_below_threshold"
# The agent used every step it had without deciding.
NOTE_LLM_ITERATION_CAP = "llm_iteration_cap"
# Summary-level notes: the pass ran out of calls, or gave up after repeated gateway
# failures. Neither writes a per-reference record -- the reference never got its trace,
# so the next pass has to be free to try it again.
NOTE_LLM_BUDGET_EXHAUSTED = "llm_budget_exhausted"
NOTE_LLM_CIRCUIT_BROKEN = "llm_circuit_broken"
# Reserved for the unstated denial/allegation inference (decision D2, Task 10).
NOTE_UNSTATED = "unstated"

_RESOLVED_ID_CONFIDENCE = 1.0

# What the write phase may put back on the node. ``_PATCHED_STRATEGIES`` is only the
# default table: every resolution carries its own ``patch_mode``, and the negative records
# (abstain, below threshold, iteration cap) override it to ``resolution_only`` so a field
# the extraction left as the document wrote it is never nulled out.
PATCH_NONE = "none"
PATCH_FULL = "full"
PATCH_RESOLUTION_ONLY = "resolution_only"
_PATCHED_STRATEGIES = frozenset({STRATEGY_LLM_TRACE})

# The strategies whose stored ``<field>_resolution`` the attempt guard recognises.
_TRACED_STRATEGIES = frozenset({STRATEGY_LLM_TRACE, STRATEGY_LLM_INFERRED})

# After this many consecutive traces whose only outcome was a failed gateway call, stop
# starting new ones: the provider is down and the rest of the budget would be burnt on
# the same error.
CIRCUIT_BREAKER_FAILURES = 3

# The seed is two retrievals merged: the general shortlist, plus a handful of documents so
# a document label exists at step 1 (the agent cannot name a document it has not been
# shown, and an opaque filename never surfaces through the name channel alone).
SEED_DOCUMENT_LIMIT = DOCUMENT_K

# How much of a tool call's arguments the stored trace keeps.
TRACE_ARGS_MAX_CHARS = 300

# Budget order (§5): the statements most likely to carry a real reference first, then the
# references whose hint is most specific, then the strongest seed.
_PRIORITY_STATEMENT_TYPES = frozenset({"denial", "admission"})
_BASIS_ORDER = {"cited": 0, "positional": 1, "described": 2}
_UNKNOWN_BASIS_ORDER = 3
_LEGACY_BASIS_ORDER = 4

# Locator kinds that name a place inside a document rather than the document itself; only
# these can narrow a picked passage to the statements quoted in a located span.
_NARROWING_TOOL = "locate_paragraph"


@dataclass(frozen=True)
class _Outcome:
    """What one ``(assertion, field)`` step concluded."""

    kind: str  # resolved | already_resolved | unresolved | ambiguous
    resolution: Optional[Resolution] = None
    # The field held an id that is no longer a node, whatever the cascade made of it.
    stale: bool = False


@dataclass
class _Pending:
    """A reference the cheap steps could not answer, waiting for a seed and a trace."""

    assertion_id: str
    field_name: str
    props: dict
    hint: ReferenceHint
    reference_text: str
    fingerprint: str
    entry_notes: Tuple[str, ...]
    stale: bool
    own_chunk_touched: bool
    own_document_id: Optional[str] = None
    exclude_ids: Set[str] = field(default_factory=set)
    seed: List[Candidate] = field(default_factory=list)
    registry: Optional[LabelRegistry] = None


@dataclass
class _TraceAnswer:
    """One trace's answer, already resolved off the registry that issued its labels.

    Cached per ``(fingerprint, candidate set)``: the node id and the narrowed targets are
    resolved here rather than stored as labels, because a second reference with the same
    seed *set* gets its own registry, and a label only means something inside the trace
    that issued it.
    """

    finish: TracerFinish
    trace: Tuple[Dict[str, Any], ...]
    iterations: int
    node_id: Optional[str]
    targets: Tuple[str, ...]
    capped: bool


def _item_value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _touched_ids(items: Any) -> Tuple[Set[str], Set[str]]:
    """The chunk and document ids the previous task produced.

    The cognify tail is handed ``TextSummary`` objects, which wrap their chunk in
    ``made_from``; other callers pass ``DocumentChunk`` objects (or dicts) directly.
    Anything without an id is ignored.
    """
    chunk_ids: Set[str] = set()
    document_ids: Set[str] = set()

    if items is None:
        return chunk_ids, document_ids
    if not isinstance(items, (list, tuple, set)):
        items = [items]

    for item in items:
        chunk = _item_value(item, "made_from") or item
        chunk_id = _item_value(chunk, "id")
        if chunk_id is None:
            continue
        chunk_ids.add(str(chunk_id))

        document = _item_value(chunk, "is_part_of")
        document_id = _item_value(document, "id") if document is not None else None
        if document_id is None:
            document_id = _item_value(chunk, "document_id")
        if document_id is not None:
            document_ids.add(str(document_id))

    return chunk_ids, document_ids


def _empty_summary() -> Dict[str, Any]:
    return {
        "scanned": 0,
        "already_resolved": 0,
        "resolved": 0,
        "resolved_by_strategy": {},
        "anchor_types": {},
        "unresolved": 0,
        "ambiguous": 0,
        "stale_ids": 0,
        "failed": 0,
        "edges_written": 0,
        "nodes_patched": 0,
        "dry_run": False,
        "notes": [],
        # What the pass spent, and on what.
        "llm_calls": 0,
        "llm_calls_stated": 0,
        "llm_calls_inferred": 0,
        "llm_budget": 0,
        "llm_budget_exhausted": False,
        "traces_started": 0,
        "traces_finished": 0,
        "traces_iteration_capped": 0,
        "llm_skipped_empty_graph": 0,
        "llm_cached": 0,
        "llm_abstained": 0,
        "llm_below_threshold": 0,
        "llm_unknown_label": 0,
        "llm_failed": 0,
        "tool_calls_by_name": {},
        "llm_tokens_in": 0,
        "llm_tokens_out": 0,
        # Decision D2's unstated-inference pass (Task 10); always reported, so a consumer
        # never has to branch on whether the feature was compiled in.
        "inferred_scanned": 0,
        "inferred_resolved": 0,
    }


def _count(counter: Dict[str, int], key: Optional[str]) -> None:
    if key:
        counter[key] = counter.get(key, 0) + 1


def _bump(counters: Dict[str, Any], key: str, amount: int = 1) -> None:
    counters[key] = counters.get(key, 0) + amount


def _default_patch_mode(strategy: str) -> str:
    """What a strategy patches unless the resolution says otherwise."""
    return PATCH_FULL if strategy in _PATCHED_STRATEGIES else PATCH_NONE


def _resolve_existing_id(
    assertion_id: str,
    field_name: str,
    value: str,
    reference_text: str,
    view: GraphView,
) -> _Outcome:
    """Step 1: the field already holds an id -- make sure the edge exists.

    ``reference_text`` is the wording the document used when a previous pass preserved
    it, so the edge quotes the reference rather than the id that replaced it.
    """
    if value not in view.node_ids:
        logger.debug(
            "Reference %s.%s points at an unknown node %s.", assertion_id, field_name, value
        )
        return _Outcome("unresolved")

    if (assertion_id, value, field_name) in view.edge_keys:
        return _Outcome("already_resolved")

    props = view.node_props(value)
    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=assertion_id,
            field=field_name,
            reference_text=reference_text,
            strategy=STRATEGY_EXISTING_ID,
            confidence=_RESOLVED_ID_CONFIDENCE,
            anchor_id=value,
            anchor_type=props.get("type"),
            target_ids=(value,),
            target_type=props.get("type"),
            patch_mode=_default_patch_mode(STRATEGY_EXISTING_ID),
        ),
    )


def _entity_name_text(hint: Optional[ReferenceHint], legacy_value: Optional[str]) -> Optional[str]:
    """The text the entity-name step may look up, or None when this hint is not a name.

    A reference that points inside a document is not a name: "the Complaint ¶5" names a
    paragraph of a pleading, and linking it to a ``Complaint`` stub entity would be a
    confident wrong answer. So a hint only reaches this step when it carries no locator
    and was not made positionally. ``attributed_to`` hints ("Norman Fester") pass.
    """
    if legacy_value:
        return legacy_value
    if hint is None:
        return None

    locator_kind = (hint.locator_kind or "").strip().casefold()
    if locator_kind and locator_kind != "none":
        return None
    if (hint.basis or "").strip().casefold() == "positional":
        return None
    return hint.document_hint or None


def _resolve_entity_name(
    assertion_id: str,
    field_name: str,
    reference_text: str,
    view: GraphView,
) -> Optional[_Outcome]:
    """Step 2: the reference names one entity. None means "not an entity name"."""
    entity_ids = view.entity_ids_by_name.get(generate_node_name(reference_text))
    if not entity_ids:
        return None

    if len(entity_ids) > 1:
        logger.debug(
            "Reference %s.%s names %d entities; leaving it unresolved.",
            assertion_id,
            field_name,
            len(entity_ids),
        )
        return _Outcome("ambiguous")

    entity_id = entity_ids[0]
    if (assertion_id, entity_id, field_name) in view.edge_keys:
        return _Outcome("already_resolved")

    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=assertion_id,
            field=field_name,
            reference_text=reference_text,
            strategy=STRATEGY_ENTITY_NAME,
            # The field keeps the reference: the ingest contract reads it back as written.
            confidence=_RESOLVED_ID_CONFIDENCE,
            anchor_id=entity_id,
            anchor_type="Entity",
            target_ids=(entity_id,),
            target_type="Entity",
            patch_mode=_default_patch_mode(STRATEGY_ENTITY_NAME),
        ),
    )


def _prior_attempt(props: dict, field_name: str) -> Optional[dict]:
    """A previous traced attempt stored on the node, whatever the backend shaped it as.

    Ladybug stores node properties as one JSON blob and Neo4j stores a dict property as a
    JSON string, so a reader has to accept both.
    """
    raw = props.get(f"{field_name}_resolution")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    return raw if raw.get("strategy") in _TRACED_STRATEGIES else None


def _finalize(outcome: _Outcome, entry_notes: Tuple[str, ...], stale: bool) -> _Outcome:
    """Stamp the cascade's entry conditions onto whatever the cascade concluded."""
    if not entry_notes and not stale:
        return outcome

    resolution = outcome.resolution
    if resolution is not None and entry_notes:
        resolution = replace(resolution, notes=entry_notes + resolution.notes)
    return _Outcome(outcome.kind, resolution, stale)


def _edge_precheck(outcome: _Outcome, props: dict, view: GraphView) -> _Outcome:
    """Drop a patching resolution whose edges the graph already holds.

    Without this a backend that cannot patch nodes re-plans the same resolution on every
    pass, and ``add_edges`` (a MERGE that overwrites the stored properties) would reset a
    ``feedback_weight`` ``improve()`` had tuned. Two cases once every edge is present:
    the field holds the anchor id, so there is nothing left to do (``already_resolved``);
    or it still holds its reference, so the patch is the only outstanding half of the
    write and the resolution goes out marked :data:`NOTE_EDGES_EXIST`.
    """
    resolution = outcome.resolution
    if resolution is None or resolution.patch_mode == PATCH_NONE:
        return outcome

    targets = set(resolution.target_ids)
    if resolution.anchor_id:
        targets.add(resolution.anchor_id)
    if not targets or any(
        (resolution.assertion_id, target_id, resolution.field) not in view.edge_keys
        for target_id in targets
    ):
        return outcome

    if _as_uuid(props.get(resolution.field)) is not None:
        return _Outcome("already_resolved")
    return _Outcome("resolved", replace(resolution, notes=resolution.notes + (NOTE_EDGES_EXIST,)))


def _cheap_cascade(
    assertion_id: str,
    field_name: str,
    props: dict,
    view: GraphView,
    *,
    force: bool,
    own_chunk_touched: bool,
) -> Tuple[Optional[_Outcome], Optional[_Pending]]:
    """Steps 1-3 for one ``(assertion, field)``: no retrieval, no LLM, no document reads.

    Returns ``(outcome, None)`` when the reference is answered (or there is nothing to
    answer), or ``(None, pending)`` when it needs the seed and the tracer.
    """
    value = _text_of(props.get(field_name))
    stored_text = _text_of(props.get(f"{field_name}_text"))
    holds_id = _as_uuid(value) is not None

    # The planner fix (§4): build the hint FIRST and only give up when there is neither a
    # structured reference nor anything in the field. Reading the field alone silently
    # skipped every reference extraction recorded structurally.
    hint = parse_reference_hint(
        props.get(f"{field_name}_ref"),
        fallback_text=stored_text if holds_id else (value or stored_text),
    )
    if hint is None and not value:
        return None, None

    reference_text = reference_display_text(hint) or stored_text or value or ""
    entry_notes: Tuple[str, ...] = ()
    stale = False

    if holds_id:
        # An id that is no longer a node -- the target was forgotten, or an amended
        # document was re-chunked under new ids -- has gone dark, so it re-resolves from
        # the preserved reference without waiting for force. With nothing preserved there
        # is nothing to re-resolve from, and the reference stays unresolved.
        stale = value not in view.node_ids
        if hint is not None and (force or stale):
            entry_notes = (NOTE_STALE_ID,) if stale else ()
        else:
            return (
                _finalize(
                    _resolve_existing_id(
                        assertion_id, field_name, value, stored_text or value, view
                    ),
                    entry_notes,
                    stale,
                ),
                None,
            )

    entity_text = _entity_name_text(hint, None if holds_id else value)
    if entity_text:
        entity_outcome = _resolve_entity_name(assertion_id, field_name, entity_text, view)
        if entity_outcome is not None:
            return _finalize(entity_outcome, entry_notes, stale), None

    if hint is None:
        return _finalize(_Outcome("unresolved"), entry_notes, stale), None

    fingerprint = reference_fingerprint(hint, field_name)
    # A stale id means the stored answer is dark, so a matching fingerprint must not stop
    # the re-resolution -- the guard is for references that already had their chance.
    if not force and not stale:
        prior = _prior_attempt(props, field_name)
        if prior is not None and prior.get("fingerprint") == fingerprint:
            return _finalize(_Outcome("already_resolved"), entry_notes, stale), None

    return None, _Pending(
        assertion_id=assertion_id,
        field_name=field_name,
        props=props,
        hint=hint,
        reference_text=reference_text,
        fingerprint=fingerprint,
        entry_notes=entry_notes,
        stale=stale,
        own_chunk_touched=own_chunk_touched,
    )


def _combine_seed(*candidate_lists: Sequence[Candidate]) -> List[Candidate]:
    """Union several shortlists by node id, keeping the best score and the first label."""
    best: Dict[str, Candidate] = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            current = best.get(candidate.node_id)
            if current is None or candidate.score > current.score:
                best[candidate.node_id] = candidate
    return sorted(best.values(), key=lambda candidate: (-candidate.score, candidate.node_id))


async def _seed_reference(entry: _Pending, view: GraphView, lexical: LexicalIndex, engine) -> None:
    """Fill ``entry.seed``/``entry.registry``: one retrieval pair, no LLM (§6, R10).

    Two calls on one registry: the general shortlist over the reference's wording *and*
    the statement's own proposition, plus a documents-only shortlist over the wording, so
    the agent has a document label to hand ``open_document``/``locate_paragraph`` on its
    very first step.
    """
    registry = LabelRegistry()
    own_chunk_id = str(entry.props.get("source_chunk_id") or "")
    own_document_id = view.document_by_chunk.get(own_chunk_id)
    exclude_ids = {entry.assertion_id}
    if own_chunk_id:
        exclude_ids.add(own_chunk_id)
    # A denial realleging its own pleading's paragraphs is real, so the referring
    # document is weighed down rather than filtered out; an attribution names whoever it
    # names, so it is not weighed at all.
    penalize_own_document = entry.field_name == "responds_to"

    entry.registry = registry
    entry.own_document_id = own_document_id
    entry.exclude_ids = exclude_ids

    proposition = _text_of(entry.props.get("name")) or ""
    queries = [text for text in (entry.reference_text, proposition) if text]
    if not queries:
        return

    scoped = {
        "view": view,
        "lexical": lexical,
        "registry": registry,
        "exclude_ids": exclude_ids,
        "own_document_id": own_document_id,
        "penalize_own_document": penalize_own_document,
        "vector_engine": engine,
    }
    general = await search_candidates(queries=queries, kind="any", limit=SEED_LIMIT, **scoped)
    documents = await search_candidates(
        queries=queries[:1], kind="documents", limit=SEED_DOCUMENT_LIMIT, **scoped
    )
    entry.seed = _combine_seed(general, documents)


def _order_key(entry: _Pending) -> tuple:
    """Budget order (§5): denials and admissions, then the most specific hints, then seed."""
    statement_type = (_text_of(entry.props.get("statement_type")) or "").strip().casefold()
    priority = 0 if statement_type in _PRIORITY_STATEMENT_TYPES else 1

    if entry.hint.legacy_text:
        basis = _LEGACY_BASIS_ORDER
    else:
        basis = _BASIS_ORDER.get((entry.hint.basis or "").strip().casefold(), _UNKNOWN_BASIS_ORDER)

    top_score = max((candidate.score for candidate in entry.seed), default=0.0)
    return (priority, basis, -top_score, entry.assertion_id, entry.field_name)


def _trace_dicts(records: Sequence[TraceRecord]) -> Tuple[Dict[str, Any], ...]:
    """The stored, audit-sized form of a trace's tool steps."""
    stored = []
    for record in records:
        try:
            arguments = json.dumps(record.args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001 - the args came off the wire; never fail a write
            arguments = str(record.args)
        stored.append(
            {
                "tool": record.tool,
                "args": arguments[:TRACE_ARGS_MAX_CHARS],
                "result_preview": record.result_preview,
                "ok": record.ok,
            }
        )
    return tuple(stored)


async def _narrow_to_quoted_assertions(
    entry: _Pending,
    records: Sequence[TraceRecord],
    chunk_id: str,
    view: GraphView,
    texts: DocumentTextCache,
) -> Tuple[str, ...]:
    """The statements the trace's own ``locate_paragraph`` listed for the picked passage.

    The agent picked a passage after locating a numbered paragraph, so the answer is the
    statements quoted *inside that span* -- today's narrowing, reached through the trace
    rather than through a guess. The lookup is replayed from the arguments the trace
    recorded (the document text is already cached), which is exactly the output the agent
    saw; nothing else in the trace can widen it.
    """
    for record in reversed(list(records)):
        if record.tool != _NARROWING_TOOL or not record.ok:
            continue

        document_id = entry.registry.resolve(record.args.get("document"))
        if document_id is None or document_id not in view.documents:
            continue
        locator = build_locator(record.args.get("kind"), record.args.get("value"))
        if locator is None:
            continue

        chunks = view.chunks_by_document.get(document_id) or []
        located = await _locate(texts, document_id, chunks, locator)
        if located is None:
            continue

        span_text, chunk_positions, _anchor_position, _notes = located
        chunk_ids = {str(chunks[position].get("id")) for position in chunk_positions}
        if chunk_id not in chunk_ids:
            continue

        return tuple(
            select_anchored_assertions(
                span_text,
                [
                    (candidate_id, props.get("source_quote"))
                    for candidate_id, props in view.assertions.items()
                    if candidate_id != entry.assertion_id
                    and str(props.get("source_chunk_id") or "") in chunk_ids
                ],
            )
        )
    return ()


async def _build_answer(
    entry: _Pending,
    finish: TracerFinish,
    records: Sequence[TraceRecord],
    iterations: int,
    *,
    capped: bool,
    view: GraphView,
    texts: DocumentTextCache,
    counters: Dict[str, Any],
) -> _TraceAnswer:
    """Resolve a finish off its own registry, while that registry still means something."""
    node_id = entry.registry.resolve(finish.candidate_label)
    targets: Tuple[str, ...] = ()
    if node_id is not None and node_id in view.chunks:
        targets = await _narrow_to_quoted_assertions(entry, records, node_id, view, texts)
    if node_id is not None and node_id not in view.node_ids:
        # The label resolved to something the view no longer holds.
        _bump(counters, "llm_unknown_label")
        node_id = None

    return _TraceAnswer(
        finish=finish,
        trace=_trace_dicts(records),
        iterations=iterations,
        node_id=node_id,
        targets=targets,
        capped=capped,
    )


def _negative_record(entry: _Pending, answer: _TraceAnswer, note: str) -> _Outcome:
    """An answer worth remembering but never worth linking.

    Written ``resolution_only``: the reference the extraction recorded stays exactly as
    the document made it, and the fingerprint means the next pass spends nothing
    reconsidering an unchanged reference.
    """
    return _Outcome(
        "unresolved",
        Resolution(
            assertion_id=entry.assertion_id,
            field=entry.field_name,
            reference_text=entry.reference_text,
            strategy=STRATEGY_LLM_TRACE,
            confidence=answer.finish.confidence,
            notes=(note,),
            reason=answer.finish.reason or None,
            fingerprint=entry.fingerprint,
            patch_mode=PATCH_RESOLUTION_ONLY,
            iterations=answer.iterations,
            trace=answer.trace,
        ),
    )


def _answer_to_outcome(
    entry: _Pending,
    answer: _TraceAnswer,
    view: GraphView,
    *,
    threshold: float,
    counters: Dict[str, Any],
) -> _Outcome:
    """Map a trace's answer onto a ``Resolution``, by what the picked label turned out to be."""
    if answer.node_id is None:
        note = NOTE_LLM_ITERATION_CAP if answer.capped else NOTE_LLM_ABSTAINED
        return _negative_record(entry, answer, note)

    if answer.finish.confidence < threshold:
        _bump(counters, "llm_below_threshold")
        logger.debug(
            "Trace for %s.%s scored %.2f, below the %.2f threshold.",
            entry.assertion_id,
            entry.field_name,
            answer.finish.confidence,
            threshold,
        )
        return _negative_record(entry, answer, NOTE_LLM_BELOW_THRESHOLD)

    node_id = answer.node_id
    if node_id in view.assertions:
        anchor_type = "Assertion"
        target_ids: Tuple[str, ...] = (node_id,)
        target_type = "Assertion"
        document_id = view.document_by_chunk.get(
            str(view.assertions[node_id].get("source_chunk_id") or "")
        )
    elif node_id in view.chunks:
        anchor_type = "DocumentChunk"
        # Keep the referring statement out of its own answer even when the cached trace
        # was built for a different assertion.
        target_ids = tuple(target for target in answer.targets if target != entry.assertion_id)
        target_type = "Assertion" if target_ids else "DocumentChunk"
        document_id = view.document_by_chunk.get(node_id)
    else:
        # ``.get``: the node resolved out of this trace's own registry, but a view that no
        # longer holds it as a document must degrade to an untyped anchor, not a KeyError.
        anchor_type = view.documents.get(node_id, {}).get("type")
        target_ids = ()
        target_type = anchor_type
        document_id = node_id

    return _Outcome(
        "resolved",
        Resolution(
            assertion_id=entry.assertion_id,
            field=entry.field_name,
            reference_text=entry.reference_text,
            strategy=STRATEGY_LLM_TRACE,
            confidence=answer.finish.confidence,
            anchor_id=node_id,
            anchor_type=anchor_type,
            target_ids=target_ids,
            target_type=target_type,
            document_id=document_id,
            reason=answer.finish.reason or None,
            fingerprint=entry.fingerprint,
            patch_mode=_default_patch_mode(STRATEGY_LLM_TRACE),
            iterations=answer.iterations,
            trace=answer.trace,
        ),
    )


def _record_outcome(
    summary: Dict[str, Any],
    resolutions: List[Resolution],
    outcome: _Outcome,
    *,
    touched: Optional[Tuple[Set[str], Set[str]]],
    own_chunk_touched: bool,
) -> None:
    """Fold one outcome into the summary and the plan, honouring the touched scope."""
    if not own_chunk_touched:
        # Out of scope unless it points at a document this ingestion wrote.
        resolution = outcome.resolution
        if resolution is None or resolution.document_id not in touched[1]:
            return

    summary["scanned"] += 1
    if outcome.stale:
        summary["stale_ids"] += 1
    if outcome.resolution is not None:
        resolutions.append(outcome.resolution)
    if outcome.kind == "resolved" and outcome.resolution is not None:
        summary["resolved"] += 1
        _count(summary["resolved_by_strategy"], outcome.resolution.strategy)
        _count(summary["anchor_types"], outcome.resolution.anchor_type)
    else:
        summary[outcome.kind] += 1


def _fold_counters(summary: Dict[str, Any], counters: Dict[str, Any]) -> None:
    """Merge the tracer's counters into the summary, coercing the one flag it shares.

    ``trace_reference`` bumps ``llm_budget_exhausted`` as a count (once per trace that
    could not pay); the summary reports it as the pass-level flag it is, so it is true
    exactly when at least one reference went untraced for want of a call -- not merely
    when the last trace happened to spend the last slot.
    """
    for key, value in counters.items():
        if key == "llm_budget_exhausted":
            continue
        summary[key] = value
    summary["llm_budget_exhausted"] = bool(counters.get("llm_budget_exhausted"))
    summary["llm_calls_stated"] = summary["llm_calls"] - summary["llm_calls_inferred"]


async def plan_resolutions(
    view: GraphView,
    texts: DocumentTextCache,
    *,
    allow_llm: bool = True,
    force: bool = False,
    llm_max_calls: Optional[int] = None,
    tracer_max_iter: Optional[int] = None,
    llm_confidence_threshold: Optional[float] = None,
    touched: Optional[Tuple[Set[str], Set[str]]] = None,
) -> Tuple[List[Resolution], Dict[str, Any]]:
    """Run the cascade over every dangling reference in the view.

    ``touched`` restricts the pass to one ingestion: a reference is in scope when the
    assertion carrying it came from a touched chunk, or when it names a touched document
    (an earlier document pointing at the one just ingested).

    ``allow_llm=False`` stops after the entity-name step -- the ingest tail's contract.
    The three budget arguments default to the ``CognifyConfig`` values
    (``REFERENCE_LLM_MAX_CALLS``, ``REFERENCE_TRACER_MAX_ITER``,
    ``REFERENCE_LLM_CONFIDENCE_THRESHOLD``).
    """
    config = get_cognify_config()
    max_calls = config.reference_llm_max_calls if llm_max_calls is None else int(llm_max_calls)
    max_iter = config.reference_tracer_max_iter if tracer_max_iter is None else int(tracer_max_iter)
    threshold = (
        config.reference_llm_confidence_threshold
        if llm_confidence_threshold is None
        else float(llm_confidence_threshold)
    )

    summary = _empty_summary()
    summary["llm_budget"] = max_calls
    resolutions: List[Resolution] = []
    pending: List[_Pending] = []
    counters: Dict[str, Any] = {}
    budget = CallBudget(max_calls)

    with operation_usage_scope() as usage:
        for assertion_id, props in view.assertions.items():
            own_chunk_touched = (
                touched is None or str(props.get("source_chunk_id") or "") in (touched[0])
            )

            for field_name in REFERENCE_FIELDS:
                try:
                    outcome, entry = _cheap_cascade(
                        assertion_id,
                        field_name,
                        props,
                        view,
                        force=force,
                        own_chunk_touched=own_chunk_touched,
                    )
                except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
                    logger.warning(
                        "Could not resolve %s on assertion %s: %s", field_name, assertion_id, error
                    )
                    summary["scanned"] += 1
                    summary["failed"] += 1
                    continue

                if entry is not None:
                    if allow_llm:
                        pending.append(entry)
                    else:
                        # The tail writes nothing for a reference it cannot answer, so
                        # the pass retries it from scratch.
                        _record_outcome(
                            summary,
                            resolutions,
                            _finalize(_Outcome("unresolved"), entry.entry_notes, entry.stale),
                            touched=touched,
                            own_chunk_touched=own_chunk_touched,
                        )
                elif outcome is not None:
                    _record_outcome(
                        summary,
                        resolutions,
                        outcome,
                        touched=touched,
                        own_chunk_touched=own_chunk_touched,
                    )

        if pending:
            await _trace_pending(
                pending,
                view,
                texts,
                summary,
                resolutions,
                counters=counters,
                budget=budget,
                max_iter=max_iter,
                threshold=threshold,
                touched=touched,
            )

    _fold_counters(summary, counters)
    summary["llm_tokens_in"] = usage.tokens_in
    summary["llm_tokens_out"] = usage.tokens_out

    logger.info(
        "Reference resolution planned: scanned=%d resolved=%d already=%d unresolved=%d "
        "ambiguous=%d stale_ids=%d failed=%d llm_calls=%d/%d traces=%d",
        summary["scanned"],
        summary["resolved"],
        summary["already_resolved"],
        summary["unresolved"],
        summary["ambiguous"],
        summary["stale_ids"],
        summary["failed"],
        summary["llm_calls"],
        summary["llm_budget"],
        summary["traces_started"],
    )
    return resolutions, summary


async def _trace_pending(
    pending: List[_Pending],
    view: GraphView,
    texts: DocumentTextCache,
    summary: Dict[str, Any],
    resolutions: List[Resolution],
    *,
    counters: Dict[str, Any],
    budget: CallBudget,
    max_iter: int,
    threshold: float,
    touched: Optional[Tuple[Set[str], Set[str]]],
) -> None:
    """Seed every pending reference, order them, then trace them one at a time."""

    def record(entry: _Pending, outcome: _Outcome) -> None:
        _record_outcome(
            summary,
            resolutions,
            _finalize(outcome, entry.entry_notes, entry.stale),
            touched=touched,
            own_chunk_touched=entry.own_chunk_touched,
        )

    def fail(entry: _Pending, error: Exception) -> None:
        logger.warning(
            "Could not resolve %s on assertion %s: %s",
            entry.field_name,
            entry.assertion_id,
            error,
        )
        summary["scanned"] += 1
        summary["failed"] += 1

    if not view.documents:
        # Nothing to search and nothing to read: an agent asked to pick a document out of
        # an empty set can only hallucinate one.
        logger.info(
            "Skipping %d reference trace(s): the graph view holds no documents.", len(pending)
        )
        for entry in pending:
            summary["llm_skipped_empty_graph"] += 1
            record(entry, _Outcome("unresolved"))
        return

    from cognee.tasks.graph import reference_retrieval

    vector_engine = await reference_retrieval.get_vector_engine_async()
    # One index per pass: both BM25 corpora are the whole view, and rebuilding them per
    # reference would dominate the pass.
    lexical = LexicalIndex(view)

    seeded: List[_Pending] = []
    for entry in pending:
        try:
            await _seed_reference(entry, view, lexical, vector_engine)
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue
        seeded.append(entry)

    seeded.sort(key=_order_key)

    cache: Dict[Tuple[str, str], _TraceAnswer] = {}
    consecutive_failures = 0
    circuit_broken = False
    exhausted_references = 0

    for entry in seeded:
        if circuit_broken:
            record(entry, _Outcome("unresolved"))
            continue

        cache_key = (entry.fingerprint, candidate_set_key(entry.seed))
        cached = cache.get(cache_key)
        if cached is not None:
            _bump(counters, "llm_cached")
            try:
                record(entry, _edge_precheck_outcome(entry, cached, view, threshold, counters))
            except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
                fail(entry, error)
            continue

        tools = build_tracer_tools(
            view=view,
            texts=texts,
            lexical=lexical,
            registry=entry.registry,
            vector_engine=vector_engine,
            exclude_ids=entry.exclude_ids,
            own_document_id=entry.own_document_id,
            penalize_own_document=entry.field_name == "responds_to",
        )

        before = (
            counters.get("llm_failed", 0),
            counters.get("llm_budget_exhausted", 0),
            counters.get("traces_iteration_capped", 0),
        )
        _bump(counters, "traces_started")
        try:
            finish, records, iterations = await trace_reference(
                system_prompt_path=TRACE_SYSTEM_PROMPT,
                hint=entry.hint,
                source_props=entry.props,
                source_document_name=_document_name(view, entry.own_document_id),
                field_name=entry.field_name,
                seed=entry.seed,
                tools=tools,
                registry=entry.registry,
                budget=budget,
                max_iter=max_iter,
                counters=counters,
            )
        except (FileNotFoundError, ValueError):
            # A missing or blank system prompt is a deployment bug, not a per-reference
            # failure (R11): swallowing it would turn one bad file into a pass full of
            # silent abstentions.
            raise
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue

        failed = counters.get("llm_failed", 0) > before[0]
        exhausted = counters.get("llm_budget_exhausted", 0) > before[1]
        capped = counters.get("traces_iteration_capped", 0) > before[2]

        consecutive_failures = consecutive_failures + 1 if failed else 0
        if consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
            circuit_broken = True
            summary["notes"].append(NOTE_LLM_CIRCUIT_BROKEN)
            logger.warning(
                "Reference tracing circuit broken after %d consecutive failed calls; the "
                "remaining references are left for the next pass.",
                consecutive_failures,
            )

        if exhausted or failed:
            # Neither got a real answer, so neither writes a record: a reference that
            # never had its trace must stay retryable.
            exhausted_references += 1 if exhausted else 0
            record(entry, _Outcome("unresolved"))
            continue

        _bump(counters, "traces_finished")
        # Mapping the finish onto a Resolution runs inside the guard as well: it reads
        # the view and replays the trace's own locator lookup, and a bug in either must
        # cost this one reference rather than abort a pass that has already been paid for.
        try:
            answer = await _build_answer(
                entry,
                finish,
                records,
                iterations,
                capped=capped,
                view=view,
                texts=texts,
                counters=counters,
            )
            cache[cache_key] = answer
            record(entry, _edge_precheck_outcome(entry, answer, view, threshold, counters))
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue

    if exhausted_references:
        summary["notes"].append(NOTE_LLM_BUDGET_EXHAUSTED)
        logger.warning(
            "Reference resolution ran out of its %d-call budget with %d reference(s) "
            "still untraced; nothing was written for them, so the next pass retries them.",
            budget.max_calls,
            exhausted_references,
        )


def _edge_precheck_outcome(
    entry: _Pending,
    answer: _TraceAnswer,
    view: GraphView,
    threshold: float,
    counters: Dict[str, Any],
) -> _Outcome:
    outcome = _answer_to_outcome(entry, answer, view, threshold=threshold, counters=counters)
    if outcome.kind == "resolved":
        outcome = _edge_precheck(outcome, entry.props, view)
    return outcome


def _document_name(view: GraphView, document_id: Optional[str]) -> Optional[str]:
    if not document_id:
        return None
    return _text_of(view.documents.get(document_id, {}).get("name"))


async def write_resolutions(
    graph_engine,
    view: GraphView,
    resolutions: Sequence[Resolution],
    *,
    provenance_kwargs: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Write the planned resolutions: edges first, then the node patches.

    Edges come first because a patch points the field at a node the edges must already
    reach. ``add_edges`` upserts on ``(source, target, relationship)``, so re-emitting an
    edge a previous pass wrote cannot duplicate it -- but the upsert also overwrites that
    edge's stored properties, so a resolution the planner marked :data:`NOTE_EDGES_EXIST`
    writes no edge at all and is patched only.

    Each resolution's ``patch_mode`` decides the node patch: ``"none"`` (``existing_id``,
    ``entity_name``) patches nothing, ``"resolution_only"`` writes the audit blob without
    touching the field, and ``"full"`` moves the anchor's id into the field.

    Indexing the new edge texts is the one step allowed to fail on its own: the edges are
    already stored, so the patches still run and the failure comes back as the
    ``edge_index_failed`` note rather than as a half-applied write. Nothing retries it --
    a later pass finds those edges present and re-emits nothing -- so the note means an
    operator has to re-index: ``index_graph_edges()`` with no argument rescans the graph.
    """
    summary = {
        "edges_written": 0,
        "nodes_patched": 0,
        "already_resolved": 0,
        "dry_run": bool(dry_run),
        "notes": [],
    }
    if not resolutions:
        return summary

    edges = []
    endpoints: Dict[str, dict] = {}
    for resolution in resolutions:
        if NOTE_EDGES_EXIST in resolution.notes:
            continue

        target_ids = list(resolution.target_ids)
        if resolution.anchor_id and resolution.anchor_id not in target_ids:
            target_ids.append(resolution.anchor_id)
        if not target_ids:
            # An abstention, a below-threshold answer or an inferred-but-unlinked record:
            # audited on the node, never an edge.
            continue

        source_props = view.assertions.get(resolution.assertion_id, {})
        endpoints[resolution.assertion_id] = source_props

        for target_id in target_ids:
            target_props = view.node_props(target_id)
            endpoints[target_id] = target_props
            edges.append(
                build_reference_edge(
                    resolution,
                    target_id,
                    source_props=source_props,
                    target_label=_node_label(target_props),
                    target_type=target_props.get("type") or resolution.target_type or "Node",
                )
            )

    if dry_run:
        logger.info(
            "Reference resolution dry_run: %d edge(s) and %d patch(es) withheld.",
            len(edges),
            sum(1 for r in resolutions if r.patch_mode != PATCH_NONE),
        )
        return summary

    if edges:
        edges = ensure_default_edge_properties(edges, nodes=list(endpoints.values()))
        await graph_engine.add_edges(edges, **(provenance_kwargs or {}))
        summary["edges_written"] = len(edges)
        try:
            await index_graph_edges(edges)
        except Exception as error:  # noqa: BLE001 - the edges are stored; patch anyway
            logger.warning(
                "Wrote %d reference edge(s) but could not index their text (%s); the "
                "edges are in the graph and remain traversable, but their text stays out "
                "of the EdgeType_relationship_name collection until index_graph_edges "
                "runs over them again. Nothing does that automatically -- a later "
                "resolver pass finds the edges present and re-emits nothing, and "
                "improve() indexes triplets rather than edge texts -- so re-index "
                "explicitly: index_graph_edges() with no argument rescans the graph.",
                len(edges),
                error,
            )
            summary["notes"].append(NOTE_EDGE_INDEX_FAILED)

    for resolution in resolutions:
        if resolution.patch_mode == PATCH_NONE:
            continue

        values = build_node_patch(
            resolution,
            view.assertions.get(resolution.assertion_id, {}),
            mode=resolution.patch_mode,
        )
        try:
            await graph_engine.update_node(resolution.assertion_id, values)
        except NotImplementedError:
            logger.warning(
                "Graph adapter cannot patch nodes; reference edges were written but the "
                "assertion fields still hold their reference text."
            )
            summary["notes"].append("node_patch_unsupported")
            summary["nodes_patched"] = 0
            # Nothing was left to do for a patch-only resolution, and nothing could be
            # done: the graph already holds its edges, so it counts as already resolved.
            summary["already_resolved"] = sum(
                1 for planned in resolutions if NOTE_EDGES_EXIST in planned.notes
            )
            break
        summary["nodes_patched"] += 1

    logger.info(
        "Reference resolution wrote %d edge(s) and patched %d node(s).",
        summary["edges_written"],
        summary["nodes_patched"],
    )
    return summary


def _merge_write_summary(summary: Dict[str, Any], write_summary: Dict[str, Any]) -> None:
    """Fold the write phase's counters into the plan's.

    Only ``already_resolved`` adds rather than replaces: the write phase reports the
    planned resolutions that turned out to need no write, and those stop being resolutions
    of this pass. ``notes`` concatenates, because the plan's notes (a budget that ran out,
    a broken circuit) and the write's (an index that failed) are about different phases.
    """
    written = dict(write_summary)
    already = written.pop("already_resolved", 0)
    notes = list(written.pop("notes", []) or [])
    summary["already_resolved"] = summary.get("already_resolved", 0) + already
    summary["resolved"] = max(0, summary.get("resolved", 0) - already)
    summary.update(written)
    summary["notes"] = list(summary.get("notes") or []) + notes


def _dataset_id(ctx, dataset_id):
    """The dataset whose relational rows hold the document locations."""
    from_context = getattr(getattr(ctx, "dataset", None), "id", None)
    return from_context if from_context is not None else dataset_id


def _allow_llm(scope: str, allow_llm: Optional[bool]) -> bool:
    """Decision D1: only the whole-graph pass may spend LLM calls, unless told otherwise."""
    return scope == "all" if allow_llm is None else bool(allow_llm)


async def _plan(
    data,
    *,
    scope: str,
    allow_llm: Optional[bool],
    force: bool,
    llm_max_calls: Optional[int],
    tracer_max_iter: Optional[int],
    llm_confidence_threshold: Optional[float],
    dataset_id,
    ctx,
) -> Tuple[Any, GraphView, List[Resolution], Dict[str, Any]]:
    graph_engine = await get_graph_engine()
    view = await _load_graph_view(graph_engine)
    texts = DocumentTextCache(view, dataset_id=_dataset_id(ctx, dataset_id))
    touched = _touched_ids(data) if scope == "touched" else None
    resolutions, summary = await plan_resolutions(
        view,
        texts,
        allow_llm=_allow_llm(scope, allow_llm),
        force=force,
        llm_max_calls=llm_max_calls,
        tracer_max_iter=tracer_max_iter,
        llm_confidence_threshold=llm_confidence_threshold,
        touched=touched,
    )
    return graph_engine, view, resolutions, summary


async def _provenance(graph_engine, ctx) -> Dict[str, Any]:
    return await graph_provenance_write_kwargs(
        graph_engine,
        ctx,
        fallback_data_id=REFERENCE_RESOLUTION_DATA_ID,
        pipeline_run_id=getattr(ctx, "pipeline_run_id", None),
    )


async def detect_dangling_references(
    data: Any = None,
    *,
    scope: str = "all",
    allow_llm: Optional[bool] = None,
    force: bool = False,
    llm_max_calls: Optional[int] = None,
    tracer_max_iter: Optional[int] = None,
    llm_confidence_threshold: Optional[float] = None,
    infer_unstated: Optional[bool] = None,
    infer_confidence_threshold: Optional[float] = None,
    dataset_id=None,
    ctx=None,
) -> Dict[str, Any]:
    """Memify extraction phase: plan every reference resolution, write nothing.

    ``data`` is the memify seed and is ignored unless ``scope="touched"``, where it names
    the chunks and documents the current ingestion produced.

    ``infer_unstated`` / ``infer_confidence_threshold`` are accepted and validated by
    ``resolve_references_pipeline`` today but do nothing here: the unstated
    denial/allegation inference (decision D2, strategy ``llm_inferred``) is a separate
    pass over the same budget, and until it lands the options are carried so a caller's
    wiring does not have to change when it does.
    """
    del infer_unstated, infer_confidence_threshold
    _, _, resolutions, summary = await _plan(
        data,
        scope=scope,
        allow_llm=allow_llm,
        force=force,
        llm_max_calls=llm_max_calls,
        tracer_max_iter=tracer_max_iter,
        llm_confidence_threshold=llm_confidence_threshold,
        dataset_id=dataset_id,
        ctx=ctx,
    )
    return {"plan": resolutions, "summary": summary}


def _unwrap_payload(payload: Any) -> Tuple[List[Resolution], Dict[str, Any]]:
    """Normalize the detect phase's output, tolerating the runner wrapping it in a list."""
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
        return [], _empty_summary()

    summary = _empty_summary()
    summary.update(payload.get("summary") or {})
    return list(payload.get("plan") or []), summary


async def apply_reference_resolutions(
    payload: Any,
    *,
    dry_run: bool = False,
    ctx=None,
) -> Dict[str, Any]:
    """Memify enrichment phase: write the planned edges and node patches.

    Write failures are deliberately not swallowed here: a memify run that could not write
    is a visible error, unlike the ingest tail, which must never break its pipeline.
    """
    resolutions, summary = _unwrap_payload(payload)
    graph_engine = await get_graph_engine()
    view = await _load_graph_view(graph_engine)
    write_summary = await write_resolutions(
        graph_engine,
        view,
        resolutions,
        provenance_kwargs=await _provenance(graph_engine, ctx),
        dry_run=dry_run,
    )
    _merge_write_summary(summary, write_summary)
    return summary


@task_summary("Resolved references for {n} item(s)")
async def resolve_assertion_references(
    data: Any = None,
    *,
    scope: str = "all",
    allow_llm: Optional[bool] = None,
    force: bool = False,
    dry_run: bool = False,
    llm_max_calls: Optional[int] = None,
    tracer_max_iter: Optional[int] = None,
    llm_confidence_threshold: Optional[float] = None,
    dataset_id=None,
    ctx=None,
) -> Any:
    """Resolve dangling assertion references, then return the input unchanged.

    Args:
        data: The items the previous task produced. Only read when
            ``scope="touched"``, where they identify the ingestion to resolve around.
        scope: ``"touched"`` for an ingest tail (this document's references, and
            references pointing at it), ``"all"`` for the whole graph.
        allow_llm: Whether the agentic tracer may run. Defaults to ``scope == "all"``,
            so the ingest tail is LLM-free (decision D1) and the memify pass is not.
        force: Re-resolve references a previous pass already answered, from the
            structured reference (or the ``<field>_text``) it preserved.
        dry_run: Plan and log without writing. Traces still run, so the returned plan
            shows what the agent would have linked.
        llm_max_calls: Calls this pass may spend across every reference it traces.
            ``None`` takes ``REFERENCE_LLM_MAX_CALLS``; ``0`` seeds without spending.
        tracer_max_iter: Steps one reference's trace may take. ``None`` takes
            ``REFERENCE_TRACER_MAX_ITER``.
        llm_confidence_threshold: Below this the agent's answer is recorded but never
            linked. ``None`` takes ``REFERENCE_LLM_CONFIDENCE_THRESHOLD``.
        dataset_id: Dataset whose relational rows hold the document locations, when no
            pipeline context supplies one.
        ctx: Pipeline context, used for provenance and the dataset's document locations.

    Returns:
        ``data``, unchanged, so the task can be appended to any pipeline.
    """
    # With scope="all" the whole graph is resolved in one go, and a pipeline that streams
    # several batches would otherwise repeat that identical pass once per batch.
    memoize = scope == "all" and ctx is not None
    if memoize and getattr(ctx, "extras", {}).get("reference_resolution_ran"):
        return data

    # This entry point is also the memify registry's ``resolve_references`` task, which
    # binds scope="all" -- so it spends LLM calls, and R11 has to hold here too: a missing
    # or blank tracer prompt must fail loudly rather than become one WARNING and a pass
    # that wrote nothing. The tail (allow_llm=False) never reads a prompt and keeps
    # swallowing everything, because it may not break an ingestion.
    spends_llm = _allow_llm(scope, allow_llm)

    try:
        graph_engine, view, resolutions, summary = await _plan(
            data,
            scope=scope,
            allow_llm=allow_llm,
            force=force,
            llm_max_calls=llm_max_calls,
            tracer_max_iter=tracer_max_iter,
            llm_confidence_threshold=llm_confidence_threshold,
            dataset_id=dataset_id,
            ctx=ctx,
        )
        write_summary = await write_resolutions(
            graph_engine,
            view,
            resolutions,
            provenance_kwargs=await _provenance(graph_engine, ctx),
            dry_run=dry_run,
        )
        _merge_write_summary(summary, write_summary)
        # Memoized on success only: a pass that raised has resolved nothing, so the next
        # batch of the same run must be allowed to try again.
        if memoize:
            ctx.extras["reference_resolution_ran"] = True
    except Exception as error:  # noqa: BLE001 - resolution must never break ingestion
        if spends_llm and isinstance(error, (FileNotFoundError, ValueError)):
            raise
        logger.warning("Reference resolution skipped due to an error: %s", error)

    return data
