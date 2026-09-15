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

With ``infer_unstated`` on, the pass runs one more group after that residue is exhausted
(decision D2): the denials and admissions that reference nothing at all get the same
trace under a stricter prompt and a higher bar, and a confident answer becomes an edge
marked ``inferred`` -- never a value in the field, because the document stated nothing.

Three entry points over one pass:

* :func:`resolve_assertion_references` -- the cognify tail. Appended to a pipeline it
  resolves the references the ingestion touched, returns its input unchanged, and swallows
  its own errors, so reference resolution can never break ingestion.
* :func:`detect_dangling_references` / :func:`apply_reference_resolutions` -- the two-phase
  memify pair behind the ``resolve_references`` pipeline. The apply phase deliberately does
  **not** swallow write failures: a memify run that could not write is a visible error.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import NAMESPACE_URL, uuid5

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.write_context import graph_provenance_write_kwargs
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.engine.utils.generate_node_name import generate_node_name
from cognee.modules.graph.utils.prepare_edges_for_storage import ensure_default_edge_properties
from cognee.modules.graph.utils.reference_resolution import (
    STRATEGY_ENTITY_NAME,
    STRATEGY_EXISTING_ID,
    ReferenceHint,
    Resolution,
    build_node_patch,
    build_reference_edge,
    parse_reference_hint,
    reference_display_text,
    reference_fingerprint,
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
from cognee.tasks.graph.reference_pass import (
    CIRCUIT_BREAKER_FAILURES,
    NOTE_EDGES_EXIST,
    NOTE_FORCE_KEPT_PRIOR,
    NOTE_LLM_ABSTAINED,
    NOTE_LLM_BELOW_THRESHOLD,
    NOTE_LLM_BUDGET_EXHAUSTED,
    NOTE_LLM_CIRCUIT_BROKEN,
    NOTE_LLM_ITERATION_CAP,
    NOTE_LLM_MALFORMED_STEP,
    NOTE_LLM_SELF_REFERENCE,
    NOTE_LLM_UNKNOWN_LABEL,
    NOTE_UNSTATED,
    PATCH_FULL,
    PATCH_NONE,
    PATCH_RESOLUTION_ONLY,
    SEED_DOCUMENT_LIMIT,
    TRACE_ARGS_MAX_CHARS,
    CallBudget,
    _answer_to_outcome,
    _bump,
    _build_answer,
    _combine_seed,
    _count,
    _default_patch_mode,
    _document_name,
    _edge_precheck,
    _edge_precheck_outcome,
    _finalize,
    _narrow_to_quoted_assertions,
    _negative_record,
    _order_key,
    _Outcome,
    _Pending,
    _prior_attempt,
    _record_outcome,
    _seed_reference,
    _trace_dicts,
    _trace_pending,
    _TraceAnswer,
    _unstated_pending,
    inferred_edge_properties,
)
from cognee.tasks.storage.index_graph_edges import index_graph_edges

logger = get_logger("resolve_assertion_references")

# Two modules own the layers this one sits between, and both are imported back and
# re-exported here so existing callers keep one import site.
#
# ``reference_graph_view`` owns the graph-reading layer (GraphView, DocumentTextCache,
# _load_graph_view, and the small helpers they use): scripts/legal/resolve_references_report.py
# imports DocumentTextCache/_load_graph_view from this module, and
# scripts/legal/find_disputes.py imports DOCUMENT_NODE_TYPES from it.
# ``reference_pass`` owns the trace pass itself (its types, the seed, the budget order,
# the tracer loop and the finish-to-Resolution mapping) plus the notes and patch modes
# that vocabulary needs. See
# cognee/tests/unit/tasks/graph/test_reference_graph_view_exports.py and
# cognee/tests/unit/tasks/graph/test_reference_pass_exports.py for the two contracts.
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
    "CIRCUIT_BREAKER_FAILURES",
    "NOTE_EDGES_EXIST",
    "NOTE_FORCE_KEPT_PRIOR",
    "NOTE_LLM_ABSTAINED",
    "NOTE_LLM_BELOW_THRESHOLD",
    "NOTE_LLM_BUDGET_EXHAUSTED",
    "NOTE_LLM_CIRCUIT_BROKEN",
    "NOTE_LLM_ITERATION_CAP",
    "NOTE_LLM_MALFORMED_STEP",
    "NOTE_LLM_SELF_REFERENCE",
    "NOTE_LLM_UNKNOWN_LABEL",
    "NOTE_UNSTATED",
    "PATCH_FULL",
    "PATCH_NONE",
    "PATCH_RESOLUTION_ONLY",
    "SEED_DOCUMENT_LIMIT",
    "TRACE_ARGS_MAX_CHARS",
    "_Outcome",
    "_Pending",
    "_TraceAnswer",
    "_answer_to_outcome",
    "_bump",
    "_build_answer",
    "_combine_seed",
    "_count",
    "_default_patch_mode",
    "_document_name",
    "_edge_precheck",
    "_edge_precheck_outcome",
    "_finalize",
    "_narrow_to_quoted_assertions",
    "_negative_record",
    "_order_key",
    "_prior_attempt",
    "_record_outcome",
    "_seed_reference",
    "_trace_dicts",
    "_trace_pending",
    "_unstated_pending",
    "inferred_edge_properties",
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
# ``add_edges`` succeeded but ``index_graph_edges`` did not.
NOTE_EDGE_INDEX_FAILED = "edge_index_failed"

_RESOLVED_ID_CONFIDENCE = 1.0


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
        "llm_malformed_step": 0,
        "llm_failed": 0,
        "tool_calls_by_name": {},
        "llm_tokens_in": 0,
        "llm_tokens_out": 0,
        # Decision D2's unstated-inference pass: eligible statements it considered, and
        # the ones it linked. Always reported (zero when ``infer_unstated`` is off), so a
        # consumer never has to branch on whether the pass ran.
        "inferred_scanned": 0,
        "inferred_resolved": 0,
    }


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


def _replace_dead_id(outcome: _Outcome) -> _Outcome:
    """R27: a stale id re-resolved by name has to actually replace the dead id.

    ``entity_name`` patches nothing by default -- the field keeps the words the document
    used, which is the ingest contract. A field holding an id that is no longer a node is
    the one exception: leaving it alone would keep a dead UUID in the graph forever, so
    this resolution writes the entity's id. The wording is not lost -- the preserved
    ``<field>_text`` is what the re-resolution read in the first place.
    """
    resolution = outcome.resolution
    if resolution is None or resolution.patch_mode == PATCH_FULL:
        return outcome
    return _Outcome(outcome.kind, replace(resolution, patch_mode=PATCH_FULL), outcome.stale)


def _cheap_cascade(
    assertion_id: str,
    field_name: str,
    props: dict,
    view: GraphView,
    *,
    force: bool,
    own_chunk_touched: bool,
    max_iter: Optional[int] = None,
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
            if stale:
                entity_outcome = _replace_dead_id(entity_outcome)
            return _finalize(entity_outcome, entry_notes, stale), None

    if hint is None:
        return _finalize(_Outcome("unresolved"), entry_notes, stale), None

    fingerprint = reference_fingerprint(hint, field_name)
    # A stale id means the stored answer is dark, so a matching fingerprint must not stop
    # the re-resolution -- the guard is for references that already had their chance.
    if not force and not stale:
        # An edge a resolver pass already wrote out of this assertion on this field is an
        # answer, whether or not the node could be patched to remember it (R23). Without
        # this, every pass on a backend with no ``update_node`` re-spends the whole budget
        # tracing references it has already answered.
        if (assertion_id, field_name) in view.resolver_edge_keys:
            return _finalize(_Outcome("already_resolved"), entry_notes, stale), None

        prior = _prior_attempt(props, field_name, max_iter=max_iter)
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
        # Only ``force`` reaches here with a live id in the field; a negative trace must
        # then leave that answer (and its audit blob) exactly where it is.
        field_holds_live_id=holds_id and not stale,
    )


def _fold_counters(summary: Dict[str, Any], counters: Dict[str, Any]) -> None:
    """Merge the tracer's counters into the summary, coercing the one flag it shares.

    ``trace_reference`` bumps ``llm_budget_exhausted`` as a count (once per trace that
    could not pay); the summary reports it as the pass-level flag it is, so it is true
    exactly when at least one reference went untraced for want of a call -- not merely
    when the last trace happened to spend the last slot. A budget of zero is the one
    case where that count means something else: ``llm_max_calls=0`` is estimate mode, so
    nothing was ever available to spend and nothing was exhausted (the pass reports
    ``llm_estimate_only`` and ``traces_started`` instead).
    """
    for key, value in counters.items():
        if key == "llm_budget_exhausted":
            continue
        summary[key] = value
    summary["llm_budget_exhausted"] = (
        bool(counters.get("llm_budget_exhausted")) and summary["llm_budget"] > 0
    )
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
    infer_unstated: Optional[bool] = None,
    infer_confidence_threshold: Optional[float] = None,
    touched: Optional[Tuple[Set[str], Set[str]]] = None,
) -> Tuple[List[Resolution], Dict[str, Any]]:
    """Run the cascade over every dangling reference in the view.

    ``touched`` restricts the pass to one ingestion: a reference is in scope when the
    assertion carrying it came from a touched chunk. A reference on an untouched
    statement is left to the whole-graph pass rather than traced and then discarded
    (R26) -- a cheap answer that happens to name a touched document is still kept, so an
    earlier document pointing at the one just ingested is recorded when it costs nothing.

    ``allow_llm=False`` stops after the entity-name step -- the ingest tail's contract,
    and the one setting under which the unstated inference never runs at all.

    Every tunable defaults to its ``CognifyConfig`` value (``REFERENCE_LLM_MAX_CALLS``,
    ``REFERENCE_TRACER_MAX_ITER``, ``REFERENCE_LLM_CONFIDENCE_THRESHOLD``,
    ``REFERENCE_INFER_UNSTATED``, ``REFERENCE_INFER_CONFIDENCE_THRESHOLD``).
    """
    config = get_cognify_config()
    max_calls = config.reference_llm_max_calls if llm_max_calls is None else int(llm_max_calls)
    max_iter = config.reference_tracer_max_iter if tracer_max_iter is None else int(tracer_max_iter)
    threshold = (
        config.reference_llm_confidence_threshold
        if llm_confidence_threshold is None
        else float(llm_confidence_threshold)
    )
    infer = config.reference_infer_unstated if infer_unstated is None else bool(infer_unstated)
    infer_threshold = (
        config.reference_infer_confidence_threshold
        if infer_confidence_threshold is None
        else float(infer_confidence_threshold)
    )

    summary = _empty_summary()
    summary["llm_budget"] = max_calls
    resolutions: List[Resolution] = []
    pending: List[_Pending] = []
    # Every (assertion, field) the stated loop took an interest in, so the unstated
    # inference never offers a second answer for a field that already has one.
    handled: Set[Tuple[str, str]] = set()
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
                        max_iter=max_iter,
                    )
                except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
                    logger.warning(
                        "Could not resolve %s on assertion %s: %s", field_name, assertion_id, error
                    )
                    summary["scanned"] += 1
                    summary["failed"] += 1
                    # A field whose cascade blew up is still the stated loop's: guessing
                    # at it instead would hide the bug behind an inference.
                    handled.add((assertion_id, field_name))
                    continue

                if entry is not None or outcome is not None:
                    handled.add((assertion_id, field_name))

                if entry is not None:
                    if allow_llm and (touched is None or own_chunk_touched):
                        pending.append(entry)
                    elif allow_llm:
                        # R26: out of scope, and the seed and the trace are what this pass
                        # spends. Tracing it and discarding the answer at record time would
                        # charge this ingestion for the whole graph's residue. Nothing is
                        # recorded either -- the whole-graph pass owns this reference.
                        logger.debug(
                            "Leaving %s.%s to the whole-graph pass: its statement is "
                            "outside this ingestion.",
                            assertion_id,
                            field_name,
                        )
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

        # Decision D2, strictly after the stated loop built its residue: the statements
        # that reference nothing at all, and only when a caller opted in.
        unstated = (
            _unstated_pending(
                view,
                handled=handled,
                force=force,
                touched=touched,
                counters=counters,
                max_iter=max_iter,
            )
            if allow_llm and infer
            else []
        )

        if pending or unstated:
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
                unstated=unstated,
                infer_threshold=infer_threshold,
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
                    # An inferred link is marked and weighted down on the edge itself, so
                    # a reader can tell it from a reference the document wrote.
                    extra_properties=inferred_edge_properties(resolution.strategy),
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
    infer_unstated: Optional[bool],
    infer_confidence_threshold: Optional[float],
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
        infer_unstated=infer_unstated,
        infer_confidence_threshold=infer_confidence_threshold,
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

    ``infer_unstated`` opts into the unstated denial/admission inference (decision D2,
    strategy ``llm_inferred``), a second pass over the same budget that runs strictly
    after every stated reference was offered a trace; ``infer_confidence_threshold`` is
    the higher bar it is held to. Both default to their ``CognifyConfig`` values, and
    neither does anything when ``allow_llm`` resolves to ``False``.
    """
    _, _, resolutions, summary = await _plan(
        data,
        scope=scope,
        allow_llm=allow_llm,
        force=force,
        llm_max_calls=llm_max_calls,
        tracer_max_iter=tracer_max_iter,
        llm_confidence_threshold=llm_confidence_threshold,
        infer_unstated=infer_unstated,
        infer_confidence_threshold=infer_confidence_threshold,
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
    infer_unstated: Optional[bool] = None,
    infer_confidence_threshold: Optional[float] = None,
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
        infer_unstated: Also infer the link a denial or an admission that references
            nothing is answering (decision D2). ``None`` takes
            ``REFERENCE_INFER_UNSTATED`` (off); the tail never runs it.
        infer_confidence_threshold: The higher bar an inferred link is held to. ``None``
            takes ``REFERENCE_INFER_CONFIDENCE_THRESHOLD``.
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
            infer_unstated=infer_unstated,
            infer_confidence_threshold=infer_confidence_threshold,
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
