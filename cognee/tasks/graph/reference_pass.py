"""The agentic trace pass: seed a dangling reference, trace it, map the answer.

Split out of :mod:`cognee.tasks.graph.resolve_assertion_references`, which still owns the
cheap cascade (``existing_id`` / ``entity_name``), the plan/write entry points and the
Tasks. This module owns everything between: the pass's own types (:class:`_Outcome`,
:class:`_Pending`, :class:`_TraceAnswer`), the seed, the budget order, the tracer loop and
the mapping from a :class:`~cognee.tasks.graph.reference_tracer.TracerFinish` onto a
``Resolution``.

The dependency runs one way -- the task module imports these names back and re-exports
them, so ``scripts/legal`` and the tests keep one import site. See
``cognee/tests/unit/tasks/graph/test_reference_pass_exports.py`` for the contract.
"""

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from cognee.modules.graph.utils.reference_candidates import (
    Candidate,
    LabelRegistry,
    candidate_set_key,
)
from cognee.modules.graph.utils.reference_resolution import (
    STRATEGY_LLM_INFERRED,
    STRATEGY_LLM_TRACE,
    ReferenceHint,
    Resolution,
    build_locator,
    parse_reference_hint,
    reference_fingerprint,
    select_anchored_assertions,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.graph.reference_graph_view import (
    DocumentTextCache,
    GraphView,
    _as_uuid,
    _text_of,
)
from cognee.tasks.graph.reference_retrieval import (
    DOCUMENT_K,
    SEED_LIMIT,
    LexicalIndex,
    search_candidates,
)
from cognee.tasks.graph.reference_tracer import (
    INFER_UNSTATED_SYSTEM_PROMPT,
    TRACE_SYSTEM_PROMPT,
    CallBudget,
    TraceRecord,
    TracerFinish,
    trace_reference,
)
from cognee.tasks.graph.reference_tracer_tools import _locate, build_tracer_tools

# The same logger name the pass has always used, so nothing about its log output moves
# with the code (``reference_graph_view`` does the same).
logger = get_logger("resolve_assertion_references")


# Every edge this resolution would write is already in the graph; only the node patch is
# still outstanding, so the write phase patches and skips the edge upsert.
NOTE_EDGES_EXIST = "edges_exist"


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
# A pass run with ``llm_max_calls=0``: the seeds ran and ``traces_started`` is the
# estimate of what a real budget would cost. Reported instead of the exhaustion flag,
# because a budget of nothing was never exhausted.
NOTE_LLM_ESTIMATE_ONLY = "llm_estimate_only"
# This resolution came out of the unstated denial/admission inference (decision D2), not
# out of a reference the document made.
NOTE_UNSTATED = "unstated"


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

# Decision D2's unstated inference. A denial or an admission answers something by its
# nature, so a statement of one of those types that records no reference at all is worth
# asking about -- but only on ``responds_to``, and only as an inference: the link is
# marked ``inferred`` and weighted low so it can be told apart from a relationship the
# document actually wrote, and pruned first if it turns out to be noise
# (``cross_connect_entities.py`` sets the same weight for the same reason).
UNSTATED_FIELD = "responds_to"
UNSTATED_STATEMENT_TYPES = frozenset({"denial", "admission"})
UNSTATED_BASIS = "unstated"
INFERRED_EDGE_FEEDBACK_WEIGHT = 0.2


@dataclass(frozen=True)
class _Outcome:
    """What one ``(assertion, field)`` step concluded."""

    kind: str  # resolved | already_resolved | unresolved | ambiguous
    resolution: Optional[Resolution] = None
    # The field held an id that is no longer a node, whatever the cascade made of it.
    stale: bool = False


@dataclass
class _Pending:
    """A reference the cheap steps could not answer, waiting for a seed and a trace.

    ``unstated`` marks an inference candidate instead (decision D2): a denial or an
    admission that made no reference at all, whose ``hint`` is synthesised from its own
    proposition and is never shown to the agent.
    """

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
    unstated: bool = False


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


def _count(counter: Dict[str, int], key: Optional[str]) -> None:
    if key:
        counter[key] = counter.get(key, 0) + 1


def _bump(counters: Dict[str, Any], key: str, amount: int = 1) -> None:
    counters[key] = counters.get(key, 0) + amount


def _count_inference(
    counters: Dict[str, Any], entry: _Pending, outcome: Optional[_Outcome]
) -> None:
    """Count an inference that turned into a link, once it is really in the plan.

    ``None`` is an outcome the touched scope dropped, which never reached the plan and
    so was never a link.
    """
    if outcome is not None and entry.unstated and outcome.kind == "resolved":
        _bump(counters, "inferred_resolved")


def _default_patch_mode(strategy: str) -> str:
    """What a strategy patches unless the resolution says otherwise."""
    return PATCH_FULL if strategy in _PATCHED_STRATEGIES else PATCH_NONE


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


def _unstated_hint(proposition: str) -> ReferenceHint:
    """The synthetic hint an unstated inference is fingerprinted and seeded by.

    An inference has no reference wording, so the statement's own proposition stands in
    for it: it is what the seed searches and what the fingerprint keys on, which is
    exactly the input that decides whether re-running the pass would ask the same
    question. ``basis="unstated"`` names the kind of attempt.

    Stable and distinct from any stated hint: ``reference_fingerprint`` hashes the
    document hint, the locator, the date and the legacy text, so what separates an
    inference from a stated reference here is the document hint itself -- a stated hint
    holds the words the document used to name a document ("the Complaint"), a legacy one
    also fills ``legacy_text``, and neither is ever the statement's own proposition.
    (``basis`` is not hashed, so it is documentation rather than a discriminator.)
    """
    return ReferenceHint(document_hint=proposition, basis=UNSTATED_BASIS, legacy_text=None)


def _unstated_pending(
    view: GraphView,
    *,
    handled: Set[Tuple[str, str]],
    force: bool,
    touched: Optional[Tuple[Set[str], Set[str]]],
    counters: Dict[str, Any],
) -> List[_Pending]:
    """The statements worth asking about although they reference nothing (decision D2).

    Eligible: the statement answers something by its nature (a denial, an admission); it
    records no reference of its own -- ``responds_to`` blank **and** no
    ``responds_to_ref``, so a reference the stated loop owns is never answered twice; it
    quotes the document, which is what makes an inferred link checkable afterwards; and
    it has a proposition to search on. A field the stated loop already handled is skipped
    whatever its shape, and so is a proposition a previous pass already inferred from
    (``force`` bypasses -- the guard is for propositions that have had their chance).
    """
    entries: List[_Pending] = []

    for assertion_id, props in view.assertions.items():
        if (assertion_id, UNSTATED_FIELD) in handled:
            continue
        statement_type = (_text_of(props.get("statement_type")) or "").strip().casefold()
        if statement_type not in UNSTATED_STATEMENT_TYPES:
            continue
        if _text_of(props.get(UNSTATED_FIELD)):
            continue
        # No fallback text: a structured reference is the stated loop's, and an empty or
        # unreadable ``_ref`` is no reference at all.
        if parse_reference_hint(props.get(f"{UNSTATED_FIELD}_ref")) is not None:
            continue
        if not _text_of(props.get("source_quote")):
            continue
        proposition = _text_of(props.get("name"))
        if not proposition:
            continue

        hint = _unstated_hint(proposition)
        fingerprint = reference_fingerprint(hint, UNSTATED_FIELD)
        if not force:
            prior = _prior_attempt(props, UNSTATED_FIELD)
            if (
                prior is not None
                and prior.get("strategy") == STRATEGY_LLM_INFERRED
                and prior.get("fingerprint") == fingerprint
            ):
                continue

        _bump(counters, "inferred_scanned")
        entries.append(
            _Pending(
                assertion_id=assertion_id,
                field_name=UNSTATED_FIELD,
                props=props,
                hint=hint,
                # No reference was made, so there is no reference text. The seed falls
                # back to the proposition alone (which is what the ruling asks for), and
                # the edge records no wording it could not honestly quote.
                reference_text="",
                fingerprint=fingerprint,
                entry_notes=(),
                stale=False,
                own_chunk_touched=(
                    touched is None or str(props.get("source_chunk_id") or "") in touched[0]
                ),
                unstated=True,
            )
        )

    return entries


def _strategy_of(entry: _Pending) -> str:
    """``llm_inferred`` for a link nobody wrote, ``llm_trace`` for a reference somebody did."""
    return STRATEGY_LLM_INFERRED if entry.unstated else STRATEGY_LLM_TRACE


def _patch_mode_of(entry: _Pending) -> str:
    """An inferred link never writes the field.

    Moving the anchor's id into ``responds_to`` would make the graph claim the document
    stated a reference it never wrote, and nothing downstream could tell the two apart.
    The link is an edge (marked ``inferred``) plus the audit blob, and no more.
    """
    return PATCH_RESOLUTION_ONLY if entry.unstated else _default_patch_mode(STRATEGY_LLM_TRACE)


def _strategy_notes(entry: _Pending) -> Tuple[str, ...]:
    """Notes every resolution of this kind carries, before its own outcome note."""
    return (NOTE_UNSTATED,) if entry.unstated else ()


def inferred_edge_properties(strategy: str) -> Optional[Dict[str, Any]]:
    """The marks an inferred reference edge carries into the graph, or ``None``.

    ``ensure_default_edge_properties`` only fills a ``feedback_weight`` that is *absent*,
    so the low weight set here is what reaches storage.
    """
    if strategy != STRATEGY_LLM_INFERRED:
        return None
    return {"inferred": True, "feedback_weight": INFERRED_EDGE_FEEDBACK_WEIGHT}


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
            strategy=_strategy_of(entry),
            confidence=answer.finish.confidence,
            notes=_strategy_notes(entry) + (note,),
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
            strategy=_strategy_of(entry),
            confidence=answer.finish.confidence,
            anchor_id=node_id,
            anchor_type=anchor_type,
            target_ids=target_ids,
            target_type=target_type,
            document_id=document_id,
            notes=_strategy_notes(entry),
            reason=answer.finish.reason or None,
            fingerprint=entry.fingerprint,
            patch_mode=_patch_mode_of(entry),
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
) -> bool:
    """Fold one outcome into the summary and the plan, honouring the touched scope.

    Returns whether the outcome was recorded, so a caller counting a particular kind of
    answer counts only the ones that really reached the plan.
    """
    if not own_chunk_touched:
        # Out of scope unless it points at a document this ingestion wrote.
        resolution = outcome.resolution
        if resolution is None or resolution.document_id not in touched[1]:
            return False

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
    return True


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
    unstated: Sequence[_Pending] = (),
    infer_threshold: float = 1.0,
) -> None:
    """Seed every candidate, order them, then trace them one at a time.

    Two groups run, in this order and never interleaved: the references the documents
    actually made, then -- when ``infer_unstated`` asked for them -- the unstated
    inferences (decision D2). They share one budget, one seed retrieval pass, one
    in-pass cache and one circuit breaker, so an inference can only ever spend what the
    stated references left, and every stated reference was offered a trace first.

    ``scanned`` and ``unresolved`` count both groups; ``inferred_scanned`` /
    ``inferred_resolved`` / ``llm_calls_inferred`` are the inference-only figures.
    """

    def record(entry: _Pending, outcome: _Outcome) -> Optional[_Outcome]:
        """Fold one answer into the plan; ``None`` when the touched scope dropped it."""
        outcome = _finalize(outcome, entry.entry_notes, entry.stale)
        recorded = _record_outcome(
            summary,
            resolutions,
            outcome,
            touched=touched,
            own_chunk_touched=entry.own_chunk_touched,
        )
        return outcome if recorded else None

    def fail(entry: _Pending, error: Exception) -> None:
        logger.warning(
            "Could not resolve %s on assertion %s: %s",
            entry.field_name,
            entry.assertion_id,
            error,
        )
        summary["scanned"] += 1
        summary["failed"] += 1

    candidates = list(pending) + list(unstated)

    if not view.documents:
        # Nothing to search and nothing to read: an agent asked to pick a document out of
        # an empty set can only hallucinate one.
        logger.info(
            "Skipping %d reference trace(s): the graph view holds no documents.", len(candidates)
        )
        for entry in candidates:
            summary["llm_skipped_empty_graph"] += 1
            record(entry, _Outcome("unresolved"))
        return

    from cognee.tasks.graph import reference_retrieval

    vector_engine = await reference_retrieval.get_vector_engine_async()
    # One index per pass: both BM25 corpora are the whole view, and rebuilding them per
    # reference would dominate the pass.
    lexical = LexicalIndex(view)

    seeded: List[_Pending] = []
    for entry in candidates:
        try:
            await _seed_reference(entry, view, lexical, vector_engine)
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue
        seeded.append(entry)

    # Ordered within each group, never across: the inferences wait for the whole stated
    # residue however strong their seeds look.
    stated_group = sorted((e for e in seeded if not e.unstated), key=_order_key)
    unstated_group = sorted((e for e in seeded if e.unstated), key=_order_key)

    cache: Dict[Tuple[str, str], _TraceAnswer] = {}
    consecutive_failures = 0
    circuit_broken = False
    exhausted_references = 0

    for entry in stated_group + unstated_group:
        entry_threshold = infer_threshold if entry.unstated else threshold
        if circuit_broken:
            record(entry, _Outcome("unresolved"))
            continue

        cache_key = (entry.fingerprint, candidate_set_key(entry.seed))
        cached = cache.get(cache_key)
        if cached is not None:
            _bump(counters, "llm_cached")
            try:
                outcome = record(
                    entry, _edge_precheck_outcome(entry, cached, view, entry_threshold, counters)
                )
                _count_inference(counters, entry, outcome)
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
            counters.get("llm_calls", 0),
        )
        _bump(counters, "traces_started")
        try:
            finish, records, iterations = await trace_reference(
                # The unstated variant shares the contract and differs on the task: it
                # asks what this statement answers, at a higher bar, and is shown no
                # reference block because there is no reference to show.
                system_prompt_path=(
                    INFER_UNSTATED_SYSTEM_PROMPT if entry.unstated else TRACE_SYSTEM_PROMPT
                ),
                hint=None if entry.unstated else entry.hint,
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
        if entry.unstated:
            # ``trace_reference`` counts every successful call in ``llm_calls``; the
            # split between stated and inferred spend is the caller's to keep.
            _bump(counters, "llm_calls_inferred", counters.get("llm_calls", 0) - before[3])

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
            outcome = record(
                entry, _edge_precheck_outcome(entry, answer, view, entry_threshold, counters)
            )
            _count_inference(counters, entry, outcome)
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue

    if budget.max_calls == 0:
        # Estimate mode: the pass was never given a call to spend, so nothing was
        # exhausted. ``traces_started`` is the estimate of what a real budget would cost,
        # and reporting the exhaustion flag (or its WARNING) here would be a lie.
        summary["notes"].append(NOTE_LLM_ESTIMATE_ONLY)
    elif exhausted_references:
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
