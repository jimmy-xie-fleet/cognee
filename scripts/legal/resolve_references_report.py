"""Audit and (optionally) apply the reference resolver over a dataset.

As of the agentic-tracer rewrite this is no longer a deterministic resolver: the plan
phase runs seed retrieval (lexical + vector) and then a budgeted LLM tracer
(``trace_reference``) per dangling reference, every trace sharing one ``CallBudget``
for the pass. **A dry run (the default -- no ``--apply``) still runs every trace and
spends the LLM budget when it is greater than zero.** ``--apply`` only decides whether
the plan gets written to the graph, not whether the LLM is called. Pass
``--llm-max-calls 0`` for a zero-spend estimate: the pass then only seeds candidates,
spends no calls, and ``traces_started`` in the summary is the would-be count (the
summary's ``notes`` carries ``llm_estimate_only`` in this mode).

Budget/behaviour flags -- leave any of these unset to let ``CognifyConfig`` decide:
``--llm-max-calls``, ``--tracer-max-iter``, ``--llm-confidence-threshold``,
``--infer-unstated`` (also runs the unstated denial/admission inference pass after the
stated loop). ``--show-traces`` prints each planned resolution's stored tracer steps --
the model's one-sentence reason, then one indented line per tool call.

Prints the summary counters (including the budget spent -- ``llm_calls`` are the calls
that came back, ``llm_calls_attempted`` is what the budget was charged -- the trace
outcomes, the per-tool call counts, and the unstated-inference counts), one line per
resolution the plan proposes, and one line per reference that stayed dangling (a non-UUID
``responds_to``/``attributed_to`` reference the plan did not touch and no edge already
answers -- including a structured-only reference held in ``<field>_ref`` with a blank
plain field -- plus every UUID-shaped value pointing at a node no longer in the graph).
The count beside that heading (``dangling_listed``) reconciles the list with the
``unresolved`` and ``ambiguous`` counters above it.

``--apply`` writes the plan (edges, then node patches) via ``write_resolutions``.

Usage:
    python scripts/legal/resolve_references_report.py <dataset>
    python scripts/legal/resolve_references_report.py <dataset> --llm-max-calls 0
    python scripts/legal/resolve_references_report.py <dataset> --apply --show-traces
"""

import argparse
import asyncio
import os
from pathlib import Path
from uuid import UUID

COGNEE_HOME = Path.home() / ".cognee"
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(COGNEE_HOME / "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(COGNEE_HOME / "data"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(COGNEE_HOME / "cache"))
os.environ.setdefault("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))


def _is_uuid_like(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _flag_or_config(value):
    return "config" if value is None else value


def _target_label(view, node_id):
    """The name (or a text preview) an edge's target is known by, for the report line."""
    if not node_id:
        return "-"
    props = view.node_props(node_id)
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    text = props.get("text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split())[:80]
    return str(node_id)


def _resolution_line(view, resolution) -> str:
    label = _target_label(view, resolution.anchor_id)
    return (
        f'{resolution.field}: "{resolution.reference_text}" → '
        f"{resolution.strategy} {resolution.confidence:.2f} {resolution.anchor_type} {label}"
    )


TRACE_PREVIEW_CHARS = 300


def _one_line(value, limit: int = TRACE_PREVIEW_CHARS) -> str:
    """A stored preview as one line: tool output is document text and carries newlines."""
    return " ".join(str(value or "").split())[:limit]


def _trace_lines(resolution) -> list:
    """One line for the model's reason, then one indented line per stored tool step."""
    if not getattr(resolution, "trace", None):
        return []
    lines = [f"  reason: {_one_line(resolution.reason)}"]
    for i, step in enumerate(resolution.trace, start=1):
        lines.append(
            f"    step {i}: {step.get('tool')}({_one_line(step.get('args'))}) "
            f"ok={step.get('ok')} -> {_one_line(step.get('result_preview'))}"
        )
    return lines


def _dangling_line(field_name: str, reference_text: str) -> str:
    return f'{field_name}: "{reference_text}" → unresolved 0.00 - -'


def _stale_line(field_name: str, reference_text: str) -> str:
    return f'{field_name}: "{reference_text}" → stale id 0.00 - -'


def _covered_keys(view):
    """``(assertion id, field)`` pairs that already have an edge for that field.

    The planner counts those ``already_resolved`` and emits no ``Resolution`` for them,
    so without this every such reference would be listed as unresolved. The common case
    is the whole extraction tail: ``entity_name`` links the reference and leaves the
    field holding the name, which reads exactly like a dangling reference.
    """
    return {(source_id, relationship) for source_id, _target_id, relationship in view.edge_keys}


def _dangling_entries(
    view,
    reference_fields,
    resolved_keys,
    *,
    parse_reference_hint,
    reference_display_text,
    covered_keys=frozenset(),
):
    """References the plan left untouched -- what did not resolve.

    ``plan_resolutions`` only enumerates what it resolved; a reference it left
    dangling (unresolved or ambiguous) is recovered here by walking every assertion's
    reference fields. A field is dangling when it holds no already-resolved id (an
    id-shaped string is either resolved or dead -- see ``_stale_entries`` for the dead
    ones), no edge already answers it (``covered_keys``), and the hint built from
    ``<field>_ref`` (falling back to the plain field or ``<field>_text``) is non-empty --
    this also catches a structured-only reference (``responds_to`` blank,
    ``responds_to_ref`` a dict), which a plain-string read would miss entirely.
    """
    entries = []
    for assertion_id, props in view.assertions.items():
        for field_name in reference_fields:
            value = props.get(field_name)
            if isinstance(value, str) and _is_uuid_like(value.strip()):
                continue
            if (assertion_id, field_name) in resolved_keys:
                continue
            if (assertion_id, field_name) in covered_keys:
                continue
            hint = parse_reference_hint(
                props.get(f"{field_name}_ref"),
                fallback_text=props.get(field_name) or props.get(f"{field_name}_text"),
            )
            if hint is None:
                continue
            display = reference_display_text(hint)
            if not display.strip():
                continue
            entries.append((field_name, display.strip()))
    return entries


def _stale_entries(view, reference_fields, resolved_keys, covered_keys=frozenset()):
    """UUID-shaped reference values pointing at a node that is no longer in the graph.

    A forgotten document or an amended one re-chunked under new ids leaves the field
    holding a dead id. The resolver re-resolves those from ``<field>_text`` (the plan
    covers them, so they are skipped here); what is left has no preserved wording to
    re-resolve from and is dark until an operator re-ingests the reference.
    """
    entries = []
    for assertion_id, props in view.assertions.items():
        for field_name in reference_fields:
            value = props.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            if not _is_uuid_like(value) or value in view.node_ids:
                continue
            if (assertion_id, field_name) in resolved_keys:
                continue
            if (assertion_id, field_name) in covered_keys:
                continue
            entries.append((field_name, value))
    return entries


async def run(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance.write_context import (
        graph_provenance_write_kwargs,
    )
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.graph.utils.reference_resolution import (
        parse_reference_hint,
        reference_display_text,
    )
    from cognee.tasks.graph.reference_graph_view import DocumentTextCache, _load_graph_view
    from cognee.tasks.graph.resolve_assertion_references import (
        REFERENCE_FIELDS,
        REFERENCE_RESOLUTION_DATA_ID,
        plan_resolutions,
        write_resolutions,
    )

    relational_engine = get_relational_engine()
    async with relational_engine.get_async_session() as session:
        dataset = (
            await session.execute(select(Dataset).where(Dataset.name == args.dataset))
        ).scalar_one_or_none()

    if dataset is None:
        print(f"No such dataset: {args.dataset}")
        return 2

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph_engine = await get_graph_engine()
        view = await _load_graph_view(graph_engine)
        texts = DocumentTextCache(view, dataset_id=dataset.id)

        print(f"dataset={args.dataset} id={dataset.id}")
        # The requested spend, before a single call is made -- an unset flag means
        # CognifyConfig decides, so it prints as "config" rather than a guessed number.
        print(
            "requested: "
            f"llm_max_calls={_flag_or_config(args.llm_max_calls)} "
            f"tracer_max_iter={_flag_or_config(args.tracer_max_iter)} "
            f"llm_confidence_threshold={_flag_or_config(args.llm_confidence_threshold)} "
            f"infer_unstated={True if args.infer_unstated else 'config'}"
        )

        resolutions, summary = await plan_resolutions(
            view,
            texts,
            force=args.force,
            llm_max_calls=args.llm_max_calls,
            tracer_max_iter=args.tracer_max_iter,
            llm_confidence_threshold=args.llm_confidence_threshold,
            infer_unstated=args.infer_unstated or None,
            infer_confidence_threshold=None,
        )

        print(
            f"scanned={summary['scanned']} already_resolved={summary['already_resolved']} "
            f"resolved={summary['resolved']} unresolved={summary['unresolved']} "
            f"ambiguous={summary['ambiguous']} stale_ids={summary['stale_ids']} "
            f"failed={summary['failed']}"
        )
        print(f"resolved_by_strategy={summary['resolved_by_strategy']}")
        print(f"anchor_types={summary['anchor_types']}")
        # The budget actually spent -- the effective value once None has resolved
        # against CognifyConfig.
        print(
            f"llm_budget={summary.get('llm_budget', 0)} llm_calls={summary.get('llm_calls', 0)} "
            f"llm_calls_attempted={summary.get('llm_calls_attempted', 0)} "
            f"llm_calls_stated={summary.get('llm_calls_stated', 0)} "
            f"llm_calls_inferred={summary.get('llm_calls_inferred', 0)} "
            f"llm_budget_exhausted={summary.get('llm_budget_exhausted', 0)} "
            f"llm_tokens_in={summary.get('llm_tokens_in', 0)} "
            f"llm_tokens_out={summary.get('llm_tokens_out', 0)}"
        )
        print(
            f"traces_started={summary.get('traces_started', 0)} "
            f"traces_finished={summary.get('traces_finished', 0)} "
            f"traces_iteration_capped={summary.get('traces_iteration_capped', 0)} "
            f"llm_cached={summary.get('llm_cached', 0)} "
            f"llm_abstained={summary.get('llm_abstained', 0)} "
            f"llm_below_threshold={summary.get('llm_below_threshold', 0)} "
            f"llm_unknown_label={summary.get('llm_unknown_label', 0)} "
            f"llm_malformed_step={summary.get('llm_malformed_step', 0)} "
            f"llm_failed={summary.get('llm_failed', 0)} "
            f"llm_skipped_empty_graph={summary.get('llm_skipped_empty_graph', 0)}"
        )
        print(f"tool_calls_by_name={summary.get('tool_calls_by_name', {})}")
        print(
            f"inferred_scanned={summary.get('inferred_scanned', 0)} "
            f"inferred_resolved={summary.get('inferred_resolved', 0)}"
        )
        if summary.get("notes"):
            print(f"notes={summary['notes']}")

        print("\nplanned resolutions:")
        for resolution in resolutions:
            print(_resolution_line(view, resolution))
            if args.show_traces:
                for line in _trace_lines(resolution):
                    print(line)

        resolved_keys = {(resolution.assertion_id, resolution.field) for resolution in resolutions}
        covered_keys = _covered_keys(view)
        dangling = _dangling_entries(
            view,
            REFERENCE_FIELDS,
            resolved_keys,
            parse_reference_hint=parse_reference_hint,
            reference_display_text=reference_display_text,
            covered_keys=covered_keys,
        )
        stale = _stale_entries(view, REFERENCE_FIELDS, resolved_keys, covered_keys)
        # Printed so the list reconciles with the counters above: it should now match
        # unresolved + ambiguous.
        print(f"\nunresolved / ambiguous references: dangling_listed={len(dangling) + len(stale)}")
        for field_name, reference_text in dangling:
            print(_dangling_line(field_name, reference_text))
        for field_name, reference_text in stale:
            print(_stale_line(field_name, reference_text))

        if args.apply:
            provenance = await graph_provenance_write_kwargs(
                graph_engine,
                None,
                dataset_id=dataset.id,
                fallback_data_id=REFERENCE_RESOLUTION_DATA_ID,
            )
            write_summary = await write_resolutions(
                graph_engine, view, resolutions, provenance_kwargs=provenance
            )
            print(
                f"\napplied: edges_written={write_summary['edges_written']} "
                f"nodes_patched={write_summary['nodes_patched']}"
            )
            if write_summary.get("notes"):
                print(f"notes={write_summary['notes']}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit (and optionally apply) the reference resolver over a dataset."
    )
    parser.add_argument("dataset", help="Dataset name to resolve references over.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the planned resolutions (default: dry run, plan only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-resolve references a previous pass already answered.",
    )
    parser.add_argument(
        "--llm-max-calls",
        type=int,
        default=None,
        help=(
            "Cap on LLM calls this pass may spend (default: config value; "
            "0 = seed-only, zero-spend estimate)."
        ),
    )
    parser.add_argument(
        "--tracer-max-iter",
        type=int,
        default=None,
        help="Max tracer iterations (LLM calls) per reference (default: config value).",
    )
    parser.add_argument(
        "--llm-confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence for an llm_trace resolution (default: config value).",
    )
    parser.add_argument(
        "--infer-unstated",
        action="store_true",
        help=(
            "Also run the unstated denial/admission inference pass after the stated "
            "loop (default: config value when this flag is omitted)."
        ),
    )
    parser.add_argument(
        "--show-traces",
        action="store_true",
        help="Print each planned resolution's stored tracer steps.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(parsed_args)))
