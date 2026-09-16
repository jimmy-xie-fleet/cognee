"""The agentic trace pass: seed a dangling reference, trace it, map the answer.

Owns the pass's types (:class:`PassContext`, :class:`Outcome`, :class:`_Pending`,
:class:`_TraceAnswer`), the seed, the budget order, the tracer loop and
:func:`finish_to_outcome` -- the one sequence that maps a
:class:`~cognee.tasks.graph.reference_tracer.TracerFinish` onto a ``Resolution``.
:mod:`cognee.tasks.graph.resolve_assertion_references` owns the cheap cascade and the
entry points around it and imports this module; this module reads what the write phase
declares (:mod:`cognee.tasks.graph.reference_write`), never the other way round.
"""

import json
from dataclasses import dataclass, field, replace
from enum import Enum
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
    CallBudget,
    TraceRecord,
    TracerFinish,
    _source_block,
    trace_reference,
)
from cognee.tasks.graph.reference_tracer_tools import _locate, build_tracer_tools
from cognee.tasks.graph.reference_write import (
    NOTE_EDGES_EXIST,
    PATCH_NONE,
    PATCH_RESOLUTION_ONLY,
    _default_patch_mode,
)

# The resolver's logger name, so splitting the code did not move its log output.
logger = get_logger("resolve_assertion_references")


# The agent looked and found nothing it would link.
NOTE_LLM_ABSTAINED = "llm_abstained"
# The agent named a candidate but was less sure than the configured threshold.
NOTE_LLM_BELOW_THRESHOLD = "llm_below_threshold"
# The agent used every step it had without deciding.
NOTE_LLM_ITERATION_CAP = "llm_iteration_cap"
# The answer named the very statement that was asking: nothing responds to itself.
NOTE_LLM_SELF_REFERENCE = "llm_self_reference"
# Counted apart from an abstention: the agent finished on a label this trace never issued,
# or returned a step naming neither a tool call nor a finish.
NOTE_LLM_UNKNOWN_LABEL = "llm_unknown_label"
NOTE_LLM_MALFORMED_STEP = "llm_malformed_step"
# Summary-level notes. Neither writes a per-reference record: the reference never got its
# trace, so the next pass has to be free to try it again.
NOTE_LLM_BUDGET_EXHAUSTED = "llm_budget_exhausted"
NOTE_LLM_CIRCUIT_BROKEN = "llm_circuit_broken"
# A pass run with ``llm_max_calls=0``. Reported instead of the exhaustion flag, because a
# budget of nothing was never exhausted.
NOTE_LLM_ESTIMATE_ONLY = "llm_estimate_only"
# This resolution came out of the unstated inference, not a reference the document made.
NOTE_UNSTATED = "unstated"
# A ``force`` re-check came back empty, so the answer already in the field stands.
NOTE_FORCE_KEPT_PRIOR = "force_kept_prior"


# The strategies whose stored ``<field>_resolution`` the attempt guard recognises.
_TRACED_STRATEGIES = frozenset({STRATEGY_LLM_TRACE, STRATEGY_LLM_INFERRED})

# After this many consecutive traces whose only outcome was a failed gateway call, stop
# starting new ones: the rest of the budget would be burnt on the same error.
CIRCUIT_BREAKER_FAILURES = 3

# The seed is two retrievals merged: the general shortlist, plus a handful of documents so
# a document label exists at step 1 -- the agent cannot name one it has not been shown.
SEED_DOCUMENT_LIMIT = DOCUMENT_K

# How much of a tool call's arguments the stored trace keeps.
TRACE_ARGS_MAX_CHARS = 300

# Budget order: the statements most likely to carry a real reference first, then the
# references whose hint is most specific, then the strongest seed.
_PRIORITY_STATEMENT_TYPES = frozenset({"denial", "admission"})
_BASIS_ORDER = {"cited": 0, "positional": 1, "described": 2}
_UNKNOWN_BASIS_ORDER = 3
_LEGACY_BASIS_ORDER = 4

# Locator kinds that name a place inside a document rather than the document itself; only
# these can narrow a picked passage to the statements quoted in a located span.
_NARROWING_TOOL = "locate_paragraph"

# The unstated inference: only on ``responds_to``, and only as an inference. The mark and
# the low weight the link carries into the graph are the write phase's
# (``INFERRED_EDGE_FEEDBACK_WEIGHT``).
UNSTATED_FIELD = "responds_to"
UNSTATED_STATEMENT_TYPES = frozenset({"denial", "admission"})
UNSTATED_BASIS = "unstated"


class OutcomeKind(Enum):
    """What became of one ``(assertion, field)``.

    Each value is also the summary key it is counted under, so a kind can never drift out
    of the reported shape :func:`_empty_summary` declares.
    """

    RESOLVED = "resolved"
    ALREADY_RESOLVED = "already_resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    # A step raised on this one reference. Counted, never written, never retried here.
    FAILED = "failed"


@dataclass(frozen=True)
class Outcome:
    """What one ``(assertion, field)`` concluded, wherever in the cascade it concluded.

    The cheap steps, the trace mapping and the failures all produce this one type, and
    :func:`_record_outcome` is the only thing that reads it.

    ``notes`` are the *entry* conditions the cascade read the reference under -- a stale id
    it re-resolved from preserved wording, say -- rather than notes about the answer, which
    ride on the ``Resolution``. They are prepended to the resolution's own notes when it
    reaches the plan, so an outcome with nothing to write drops them, as it should.
    """

    kind: OutcomeKind
    resolution: Optional[Resolution] = None
    notes: Tuple[str, ...] = ()
    # The field held an id that is no longer a node, whatever the cascade made of it.
    stale: bool = False


@dataclass
class _Pending:
    """A reference the cheap steps could not answer, waiting for a seed and a trace.

    ``unstated`` marks an inference candidate instead: a denial or an admission that made
    no reference at all, whose ``hint`` is synthesised from its own proposition.
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
    # The field already holds the id of a node still in the graph, and only ``force``
    # re-opened it. A re-check that comes back empty must not unseat that answer.
    field_holds_live_id: bool = False


@dataclass
class _TraceAnswer:
    """One trace's answer, already resolved off the registry that issued its labels.

    Cached per reference, source context and candidate set, so the ids are resolved here
    rather than stored as labels: a label only means something inside its own trace.

    Not an :class:`Outcome`, and deliberately so: two statements can share a reference,
    source context and seed, while the outcome they each get differs -- the
    self-reference guard, the bar an inference is held to and the edges already in the
    graph are all per statement. So the answer is what the cache holds, and
    :func:`finish_to_outcome` maps it again for every statement that reuses it.
    """

    finish: TracerFinish
    trace: Tuple[Dict[str, Any], ...]
    iterations: int
    node_id: Optional[str]
    targets: Tuple[str, ...]
    capped: bool
    # Why this trace came back without a node, when the cause was more specific than
    # "the agent looked and declined". ``None`` means a plain abstention.
    negative_note: Optional[str] = None
    # The step cap this trace ran under, kept so a capped record can say what stopped it.
    max_iter: int = 0


def _empty_summary() -> Dict[str, Any]:
    """The pass report, with every key present from the start.

    A consumer reads a fixed shape whatever the pass did, so no key is ever conditional
    (ruling R38: no nesting either -- these keys are the reported contract).
    """
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
        # These count gateway invocations, not provider requests (adapters may retry).
        # ``llm_calls`` counts successes; ``llm_calls_attempted`` also charges failures.
        "llm_calls": 0,
        "llm_calls_attempted": 0,
        "llm_calls_stated": 0,
        "llm_calls_inferred": 0,
        "llm_budget": 0,
        "llm_budget_unit": "gateway_calls",
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
        # The unstated-inference pass. Always reported (zero when it is off), so a
        # consumer never has to branch on whether it ran.
        "inferred_scanned": 0,
        "inferred_resolved": 0,
    }


@dataclass
class PassContext:
    """One resolver pass: what it loaded once, what it may spend, what it has concluded.

    Built by :func:`~cognee.tasks.graph.resolve_assertion_references.plan_resolutions` and
    handed to every step, so a step's own parameters are the reference it works on. The
    retrieval handles are filled in by :func:`_trace_pending` rather than at construction:
    a pass with nothing to trace must not touch the vector store.

    ``summary`` and ``resolutions`` are the plan the pass returns; ``counters`` is what the
    tracer bumps and :func:`_fold_counters` folds in at the end; ``cache`` is the in-pass
    trace cache, keyed on reference, source context and candidate set. The rest is read-only.

    Every default is the one that spends and links nothing -- no budget, no steps, a bar
    no confidence can clear -- so a context built without a value never resolves anything
    by accident. :class:`LabelRegistry` is deliberately absent: a label only means
    something inside the trace that issued it, so it stays on the :class:`_Pending`.
    """

    view: GraphView
    texts: DocumentTextCache
    budget: CallBudget = field(default_factory=lambda: CallBudget(0))
    force: bool = False
    max_iter: int = 0
    threshold: float = 1.0
    infer_threshold: float = 1.0
    touched: Optional[Tuple[Set[str], Set[str]]] = None
    lexical: Optional[LexicalIndex] = None
    vector_engine: Any = None
    summary: Dict[str, Any] = field(default_factory=_empty_summary)
    resolutions: List[Resolution] = field(default_factory=list)
    counters: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[Tuple[str, str, str], "_TraceAnswer"] = field(default_factory=dict)


def _count(counter: Dict[str, int], key: Optional[str]) -> None:
    if key:
        counter[key] = counter.get(key, 0) + 1


def _bump(counters: Dict[str, Any], key: str, amount: int = 1) -> None:
    counters[key] = counters.get(key, 0) + amount


def _outgrew_its_cap(record: dict, max_iter: Optional[int]) -> bool:
    """Whether a stored record was capped below the step budget this pass runs with.

    Running out of steps is the one negative outcome a bigger per-reference budget can
    change, so such a record stops counting as a prior attempt once the cap is raised.
    """
    if max_iter is None or NOTE_LLM_ITERATION_CAP not in (record.get("notes") or ()):
        return False
    try:
        stored = int(record["max_iter"])
    except (KeyError, TypeError, ValueError):
        return False
    return stored < int(max_iter)


def _prior_attempt(
    props: dict, field_name: str, *, max_iter: Optional[int] = None
) -> Optional[dict]:
    """A previous traced attempt stored on the node, whatever the backend shaped it as.

    Ladybug stores node properties as one JSON blob and Neo4j stores a dict property as a
    JSON string, so a reader has to accept both. ``max_iter`` is this pass's per-reference
    step cap: a record that gave up at a *smaller* cap is not a prior attempt any more.
    """
    raw = props.get(f"{field_name}_resolution")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    if raw.get("strategy") not in _TRACED_STRATEGIES:
        return None
    return None if _outgrew_its_cap(raw, max_iter) else raw


def _unresolved(entry: _Pending) -> Outcome:
    """A reference this pass never answered: no trace, no record, just the count.

    Every reason to give up on a pending reference -- an empty graph, a broken circuit, a
    budget that ran out, a failed call -- writes nothing, so the next pass is free to try
    it again.
    """
    return Outcome(OutcomeKind.UNRESOLVED, notes=entry.entry_notes, stale=entry.stale)


def _edge_precheck(outcome: Outcome, props: dict, view: GraphView) -> Outcome:
    """Drop a patching resolution whose edges the graph already holds.

    Without this a backend that cannot patch nodes re-plans the same resolution every pass,
    and the ``add_edges`` MERGE would reset a ``feedback_weight`` ``improve()`` had tuned.
    Its parameters stay explicit rather than taking the pass context: this is the seam the
    mapping-failure test patches with a three-argument fake.
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
        return Outcome(OutcomeKind.ALREADY_RESOLVED)
    return Outcome(
        OutcomeKind.RESOLVED, replace(resolution, notes=resolution.notes + (NOTE_EDGES_EXIST,))
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


async def _seed_reference(ctx: PassContext, entry: _Pending) -> None:
    """Fill ``entry.seed``/``entry.registry``: one retrieval pair, no LLM.

    Two calls on one registry: the general shortlist, plus a documents-only shortlist, so
    the agent has a document label to hand ``open_document`` on its very first step.
    """
    registry = LabelRegistry()
    own_chunk_id = str(entry.props.get("source_chunk_id") or "")
    own_document_id = ctx.view.document_by_chunk.get(own_chunk_id)
    exclude_ids = {entry.assertion_id}
    if own_chunk_id:
        exclude_ids.add(own_chunk_id)
    # A denial realleging its own pleading's paragraphs is real, so the referring document
    # is weighed down rather than filtered out; an attribution is not weighed at all.
    penalize_own_document = entry.field_name == "responds_to"

    entry.registry = registry
    entry.own_document_id = own_document_id
    entry.exclude_ids = exclude_ids

    proposition = _text_of(entry.props.get("name")) or ""
    queries = [text for text in (entry.reference_text, proposition) if text]
    if not queries:
        return

    scoped = {
        "view": ctx.view,
        "lexical": ctx.lexical,
        "registry": registry,
        "exclude_ids": exclude_ids,
        "own_document_id": own_document_id,
        "penalize_own_document": penalize_own_document,
        "vector_engine": ctx.vector_engine,
    }
    general = await search_candidates(queries=queries, kind="any", limit=SEED_LIMIT, **scoped)
    documents = await search_candidates(
        queries=queries[:1], kind="documents", limit=SEED_DOCUMENT_LIMIT, **scoped
    )
    entry.seed = _combine_seed(general, documents)


def _order_key(entry: _Pending) -> tuple:
    """Budget order: denials and admissions, then the most specific hints, then seed."""
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
    ctx: PassContext, entry: _Pending, records: Sequence[TraceRecord], chunk_id: str
) -> Tuple[str, ...]:
    """The statements the trace's own ``locate_paragraph`` listed for the picked passage.

    Replayed from the arguments the trace recorded, so the narrowing is exactly the output
    the agent saw; nothing else in the trace can widen it.
    """
    for record in reversed(list(records)):
        if record.tool != _NARROWING_TOOL or not record.ok:
            continue

        document_id = entry.registry.resolve(record.args.get("document"))
        if document_id is None or document_id not in ctx.view.documents:
            continue
        locator = build_locator(record.args.get("kind"), record.args.get("value"))
        if locator is None:
            continue

        chunks = ctx.view.chunks_by_document.get(document_id) or []
        located = await _locate(ctx.texts, document_id, chunks, locator)
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
                    for candidate_id, props in ctx.view.assertions.items()
                    if candidate_id != entry.assertion_id
                    and str(props.get("source_chunk_id") or "") in chunk_ids
                ],
            )
        )
    return ()


async def _build_answer(
    ctx: PassContext,
    entry: _Pending,
    finish: TracerFinish,
    records: Sequence[TraceRecord],
    iterations: int,
    *,
    capped: bool,
    negative_note: Optional[str] = None,
) -> _TraceAnswer:
    """Resolve a finish off its own registry, while that registry still means something.

    ``negative_note`` is the cause the tracer reported for a trace that came back without a
    label; it travels on the answer so the record names the cause rather than filing every
    empty trace as an abstention.
    """
    node_id = entry.registry.resolve(finish.candidate_label)
    targets: Tuple[str, ...] = ()
    if node_id is not None and node_id in ctx.view.chunks:
        targets = await _narrow_to_quoted_assertions(ctx, entry, records, node_id)
    if node_id is not None and node_id not in ctx.view.node_ids:
        # The label resolved to something the view no longer holds.
        _bump(ctx.counters, "llm_unknown_label")
        negative_note = NOTE_LLM_UNKNOWN_LABEL
        node_id = None

    return _TraceAnswer(
        finish=finish,
        trace=_trace_dicts(records),
        iterations=iterations,
        node_id=node_id,
        targets=targets,
        capped=capped,
        negative_note=negative_note,
        max_iter=ctx.max_iter,
    )


def _unstated_hint(proposition: str) -> ReferenceHint:
    """The synthetic hint an unstated inference is fingerprinted and seeded by.

    An inference has no reference wording, so the statement's own proposition stands in for
    it. That keeps the fingerprint distinct from any stated hint's, whose ``document_hint``
    holds the words a document was named by. ``basis`` is not hashed.
    """
    return ReferenceHint(document_hint=proposition, basis=UNSTATED_BASIS, legacy_text=None)


def _unstated_pending(ctx: PassContext, *, handled: Set[Tuple[str, str]]) -> List[_Pending]:
    """The statements worth asking about although they reference nothing.

    Eligible: a denial or an admission that records no reference of its own (``responds_to``
    blank **and** no ``responds_to_ref``, so a reference the stated loop owns is never
    answered twice), quotes the document, which is what makes an inferred link checkable,
    and has a proposition to search on.

    The stated loop's two guards apply here as well: an inference that already wrote its
    edge is not made again -- that edge is its *only* record on a backend that cannot patch
    nodes -- and a ``touched`` scope leaves the rest to the whole-graph pass.
    """
    entries: List[_Pending] = []

    for assertion_id, props in ctx.view.assertions.items():
        if (assertion_id, UNSTATED_FIELD) in handled:
            continue
        if (
            ctx.touched is not None
            and str(props.get("source_chunk_id") or "") not in ctx.touched[0]
        ):
            continue
        if not ctx.force and (assertion_id, UNSTATED_FIELD) in ctx.view.resolver_edge_keys:
            continue
        statement_type = (_text_of(props.get("statement_type")) or "").strip().casefold()
        if statement_type not in UNSTATED_STATEMENT_TYPES:
            continue
        if _text_of(props.get(UNSTATED_FIELD)):
            continue
        # No fallback text: a structured reference is the stated loop's.
        if parse_reference_hint(props.get(f"{UNSTATED_FIELD}_ref")) is not None:
            continue
        if not _text_of(props.get("source_quote")):
            continue
        proposition = _text_of(props.get("name"))
        if not proposition:
            continue

        hint = _unstated_hint(proposition)
        fingerprint = reference_fingerprint(hint, UNSTATED_FIELD)
        if not ctx.force:
            prior = _prior_attempt(props, UNSTATED_FIELD, max_iter=ctx.max_iter)
            if (
                prior is not None
                and prior.get("strategy") == STRATEGY_LLM_INFERRED
                and prior.get("fingerprint") == fingerprint
            ):
                continue

        _bump(ctx.counters, "inferred_scanned")
        entries.append(
            _Pending(
                assertion_id=assertion_id,
                field_name=UNSTATED_FIELD,
                props=props,
                hint=hint,
                # No reference was made, so there is no reference text: the seed falls
                # back to the proposition, and the edge quotes no wording it cannot.
                reference_text="",
                fingerprint=fingerprint,
                entry_notes=(),
                stale=False,
                # Always true given the scope filter above; kept explicit so the field
                # means the same thing on every ``_Pending``.
                own_chunk_touched=(
                    ctx.touched is None or str(props.get("source_chunk_id") or "") in ctx.touched[0]
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
    stated a reference it never wrote. The link is an edge plus the audit blob, no more.
    """
    return PATCH_RESOLUTION_ONLY if entry.unstated else _default_patch_mode(STRATEGY_LLM_TRACE)


def _strategy_notes(entry: _Pending) -> Tuple[str, ...]:
    """Notes every resolution of this kind carries, before its own outcome note."""
    return (NOTE_UNSTATED,) if entry.unstated else ()


def _negative_record(entry: _Pending, answer: _TraceAnswer, note: str) -> Outcome:
    """An answer worth remembering but never worth linking.

    Written ``resolution_only``: the reference stays as the document made it, and the
    fingerprint means the next pass spends nothing reconsidering it. A record that hit the
    step cap also stores the cap, the one negative outcome a bigger budget can change.
    """
    return Outcome(
        OutcomeKind.UNRESOLVED,
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
            max_iter=answer.max_iter if note == NOTE_LLM_ITERATION_CAP else None,
        ),
    )


def _threshold_for(ctx: PassContext, entry: _Pending) -> float:
    """The bar this answer is held to: the higher one for a link nobody wrote."""
    return ctx.infer_threshold if entry.unstated else ctx.threshold


def _linked_record(ctx: PassContext, entry: _Pending, answer: _TraceAnswer) -> Outcome:
    """An answer worth linking, typed by what the picked label turned out to be.

    A picked passage is narrowed to the statements the trace saw quoted in it; a picked
    document stays the anchor, with no target of its own.
    """
    node_id = answer.node_id
    if node_id in ctx.view.assertions:
        anchor_type = "Assertion"
        target_ids: Tuple[str, ...] = (node_id,)
        target_type = "Assertion"
        document_id = ctx.view.document_by_chunk.get(
            str(ctx.view.assertions[node_id].get("source_chunk_id") or "")
        )
    elif node_id in ctx.view.chunks:
        anchor_type = "DocumentChunk"
        # Keep the referring statement out of its own answer even when the cached trace
        # was built for a different assertion.
        target_ids = tuple(target for target in answer.targets if target != entry.assertion_id)
        target_type = "Assertion" if target_ids else "DocumentChunk"
        document_id = ctx.view.document_by_chunk.get(node_id)
    else:
        # ``.get``: the node resolved out of this trace's own registry, but a view that no
        # longer holds it as a document must degrade to an untyped anchor, not a KeyError.
        anchor_type = ctx.view.documents.get(node_id, {}).get("type")
        target_ids = ()
        target_type = anchor_type
        document_id = node_id

    return Outcome(
        OutcomeKind.RESOLVED,
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


def _record_outcome(ctx: PassContext, outcome: Outcome, *, own_chunk_touched: bool) -> None:
    """Fold one outcome into the summary and the plan, honouring the touched scope.

    The one place an outcome turns into a counted answer and (when there is something to
    write) a planned resolution, so it is also where the entry conditions the outcome
    carried are stamped onto that resolution.
    """
    summary = ctx.summary
    if outcome.kind is OutcomeKind.FAILED:
        # Counted wherever it happened: a step that raised is the pass's failure, not the
        # reference's, and the scope filter below reads a resolution a failure never has.
        summary["scanned"] += 1
        summary["failed"] += 1
        return

    resolution = outcome.resolution
    if not own_chunk_touched:
        # Out of scope unless it points at a document this ingestion wrote.
        if resolution is None or resolution.document_id not in ctx.touched[1]:
            return

    summary["scanned"] += 1
    if outcome.stale:
        summary["stale_ids"] += 1
    if resolution is not None:
        if outcome.notes:
            resolution = replace(resolution, notes=outcome.notes + resolution.notes)
        ctx.resolutions.append(resolution)
    if outcome.kind is OutcomeKind.RESOLVED and resolution is not None:
        summary["resolved"] += 1
        _count(summary["resolved_by_strategy"], resolution.strategy)
        _count(summary["anchor_types"], resolution.anchor_type)
        if resolution.strategy == STRATEGY_LLM_INFERRED:
            # An inference that really turned into a link, counted now that it is in the
            # plan: an out-of-scope answer above never became one.
            _bump(ctx.counters, "inferred_resolved")
    else:
        summary[outcome.kind.value] += 1


async def _trace_pending(
    ctx: PassContext, pending: List[_Pending], *, unstated: Sequence[_Pending] = ()
) -> None:
    """Seed every candidate, order them, then trace them one at a time.

    Two groups run, never interleaved: the references the documents made, then -- when
    ``infer_unstated`` asked for them -- the unstated inferences. They share one budget, so
    an inference can only ever spend what the stated references left.
    """

    def record(entry: _Pending, outcome: Outcome) -> None:
        """Fold one answer into the plan, under the scope this entry was seeded in."""
        _record_outcome(ctx, outcome, own_chunk_touched=entry.own_chunk_touched)

    def fail(entry: _Pending, error: Exception) -> None:
        logger.warning(
            "Could not resolve %s on assertion %s: %s",
            entry.field_name,
            entry.assertion_id,
            error,
        )
        record(entry, Outcome(OutcomeKind.FAILED))

    candidates = list(pending) + list(unstated)

    if not ctx.view.documents:
        # An agent asked to pick a document out of an empty set can only invent one.
        logger.info(
            "Skipping %d reference trace(s): the graph view holds no documents.", len(candidates)
        )
        for entry in candidates:
            ctx.summary["llm_skipped_empty_graph"] += 1
            record(entry, _unresolved(entry))
        return

    from cognee.tasks.graph import reference_retrieval

    ctx.vector_engine = await reference_retrieval.get_vector_engine_async()
    # One index per pass: rebuilding the BM25 corpora per reference would dominate it.
    ctx.lexical = LexicalIndex(ctx.view)

    seeded: List[_Pending] = []
    for entry in candidates:
        try:
            await _seed_reference(ctx, entry)
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue
        seeded.append(entry)

    # Ordered within each group, never across: the inferences wait for the whole stated
    # residue however strong their seeds look.
    stated_group = sorted((e for e in seeded if not e.unstated), key=_order_key)
    unstated_group = sorted((e for e in seeded if e.unstated), key=_order_key)

    consecutive_failures = 0
    circuit_broken = False
    exhausted_references = 0

    for entry in stated_group + unstated_group:
        if circuit_broken:
            record(entry, _unresolved(entry))
            continue

        source_document_name = _document_name(ctx.view, entry.own_document_id)
        # The same paragraph reference can answer different claims or pleadings. Reuse
        # only the context the tracer sees, plus the IDs that scope its read-only tools.
        source_context = json.dumps(
            {
                "source": _source_block(entry.props, source_document_name, entry.field_name),
                "document_id": entry.own_document_id,
                "source_chunk_id": str(entry.props.get("source_chunk_id") or ""),
                "basis": entry.hint.basis,
                "unstated": entry.unstated,
            },
            sort_keys=True,
        )
        cache_key = (entry.fingerprint, candidate_set_key(entry.seed), source_context)
        cached = ctx.cache.get(cache_key)
        if cached is not None:
            _bump(ctx.counters, "llm_cached")
            try:
                record(entry, finish_to_outcome(ctx, entry, cached))
            except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
                fail(entry, error)
            continue

        tools = build_tracer_tools(
            ctx,
            registry=entry.registry,
            exclude_ids=entry.exclude_ids,
            own_document_id=entry.own_document_id,
            penalize_own_document=entry.field_name == "responds_to",
        )

        before = (
            ctx.counters.get("llm_failed", 0),
            ctx.counters.get("llm_budget_exhausted", 0),
            ctx.counters.get("traces_iteration_capped", 0),
            ctx.counters.get("llm_calls", 0),
            ctx.counters.get("llm_unknown_label", 0),
            ctx.counters.get("llm_malformed_step", 0),
        )
        _bump(ctx.counters, "traces_started")
        try:
            finish, records, iterations = await trace_reference(
                ctx,
                # The unstated variant shares the contract and differs on the task: it
                # asks what this statement answers, and is shown no reference block. Both
                # of the pass's bars go with it, so the prompt quotes the one this answer
                # will be held to instead of a number written into the template.
                unstated=entry.unstated,
                hint=entry.hint,
                source_props=entry.props,
                source_document_name=source_document_name,
                field_name=entry.field_name,
                seed=entry.seed,
                tools=tools,
                registry=entry.registry,
            )
        except (FileNotFoundError, ValueError):
            # A missing or blank system prompt is a deployment bug, not a per-reference
            # failure: swallowing it would turn one bad file into a pass of silent
            # abstentions.
            raise
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue

        failed = ctx.counters.get("llm_failed", 0) > before[0]
        exhausted = ctx.counters.get("llm_budget_exhausted", 0) > before[1]
        capped = ctx.counters.get("traces_iteration_capped", 0) > before[2]
        # The counters the tracer bumped are the only report of *why* an empty trace was
        # empty; read them as a delta, before ``_build_answer`` bumps its own.
        negative_note = None
        if ctx.counters.get("llm_unknown_label", 0) > before[4]:
            negative_note = NOTE_LLM_UNKNOWN_LABEL
        elif ctx.counters.get("llm_malformed_step", 0) > before[5]:
            negative_note = NOTE_LLM_MALFORMED_STEP
        if entry.unstated:
            # ``trace_reference`` counts every successful call in ``llm_calls``; the split
            # between stated and inferred spend is the caller's to keep.
            _bump(
                ctx.counters,
                "llm_calls_inferred",
                ctx.counters.get("llm_calls", 0) - before[3],
            )

        consecutive_failures = consecutive_failures + 1 if failed else 0
        if consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
            circuit_broken = True
            ctx.summary["notes"].append(NOTE_LLM_CIRCUIT_BROKEN)
            logger.warning(
                "Reference tracing circuit broken after %d consecutive failed calls; the "
                "remaining references are left for the next pass.",
                consecutive_failures,
            )

        if exhausted or failed:
            # Neither got a real answer, so neither writes a record: a reference that never
            # had its trace must stay retryable.
            exhausted_references += 1 if exhausted else 0
            record(entry, _unresolved(entry))
            continue

        _bump(ctx.counters, "traces_finished")
        # Mapping the finish onto a Resolution runs inside the guard too: a bug in it must
        # cost this one reference rather than abort a pass already paid for.
        try:
            answer = await _build_answer(
                ctx,
                entry,
                finish,
                records,
                iterations,
                capped=capped,
                negative_note=negative_note,
            )
            ctx.cache[cache_key] = answer
            record(entry, finish_to_outcome(ctx, entry, answer))
        except Exception as error:  # noqa: BLE001 - one bad reference, not the pass
            fail(entry, error)
            continue

    if ctx.budget.max_calls == 0:
        # Estimate mode: the pass was never given a call to spend, so nothing was
        # exhausted and ``traces_started`` is the estimate of what a budget would cost.
        ctx.summary["notes"].append(NOTE_LLM_ESTIMATE_ONLY)
    elif exhausted_references:
        ctx.summary["notes"].append(NOTE_LLM_BUDGET_EXHAUSTED)
        logger.warning(
            "Reference resolution ran out of its %d-call budget with %d reference(s) "
            "still untraced; nothing was written for them, so the next pass retries them.",
            ctx.budget.max_calls,
            exhausted_references,
        )


def _keep_prior_on_force(ctx: PassContext, entry: _Pending, outcome: Outcome) -> Outcome:
    """Leave a forced re-check's live answer alone when the re-check came back empty.

    Recording it as a negative would replace a positive audit blob with an abstention while
    the field and its edge still hold the earlier answer, so the node would contradict
    itself.
    """
    if not entry.field_holds_live_id or outcome.kind is not OutcomeKind.UNRESOLVED:
        return outcome

    if NOTE_FORCE_KEPT_PRIOR not in ctx.summary["notes"]:
        ctx.summary["notes"].append(NOTE_FORCE_KEPT_PRIOR)
    logger.debug(
        "Forced re-check of %s.%s came back empty; keeping the answer already in the field.",
        entry.assertion_id,
        entry.field_name,
    )
    return Outcome(OutcomeKind.ALREADY_RESOLVED, stale=outcome.stale)


def finish_to_outcome(ctx: PassContext, entry: _Pending, answer: _TraceAnswer) -> Outcome:
    """The one mapping from a trace's answer onto the outcome the plan records.

    Every step, in the order it has to run: the label the trace resolved to a node, the
    self-reference guard, the bar this answer is held to, then -- only for an answer worth
    linking -- what kind of node was picked, the edges the graph already holds, and the
    forced re-check that must not unseat a live answer. The entry conditions the cheap
    steps read the reference under are stamped last, so every exit carries them.
    """
    threshold = _threshold_for(ctx, entry)

    if answer.node_id is None:
        # The step cap wins over the cause the tracer reported: a trace that ran out of
        # steps says so, and ``negative_note`` only ever names a specific abstention.
        cause = NOTE_LLM_ITERATION_CAP if answer.capped else answer.negative_note
        outcome = _negative_record(entry, answer, cause or NOTE_LLM_ABSTAINED)
    elif answer.node_id == entry.assertion_id:
        # Nothing responds to itself, and a trace really can reach this statement: the
        # tools list every assertion in a span, and the in-pass cache is keyed on a
        # fingerprint that excludes the asking assertion, so a twin's answer can name it.
        logger.debug(
            "Trace for %s.%s named the asking statement itself.",
            entry.assertion_id,
            entry.field_name,
        )
        outcome = _negative_record(entry, answer, NOTE_LLM_SELF_REFERENCE)
    elif answer.finish.confidence < threshold:
        _bump(ctx.counters, "llm_below_threshold")
        logger.debug(
            "Trace for %s.%s scored %.2f, below the %.2f threshold.",
            entry.assertion_id,
            entry.field_name,
            answer.finish.confidence,
            threshold,
        )
        outcome = _negative_record(entry, answer, NOTE_LLM_BELOW_THRESHOLD)
    else:
        outcome = _edge_precheck(_linked_record(ctx, entry, answer), entry.props, ctx.view)

    outcome = _keep_prior_on_force(ctx, entry, outcome)
    return replace(outcome, notes=entry.entry_notes, stale=entry.stale)


def _document_name(view: GraphView, document_id: Optional[str]) -> Optional[str]:
    if not document_id:
        return None
    return _text_of(view.documents.get(document_id, {}).get("name"))
