"""Audit and (optionally) apply the deterministic reference resolver over a dataset.

Dry run by default: builds the graph view and document-text cache the same way
``resolve_assertion_references`` does, plans every resolution with
``plan_resolutions``, and prints what would happen -- the summary counters, one line
per resolution the plan proposes, and one line per reference that stayed dangling
(a non-UUID ``responds_to``/``attributed_to`` value the plan did not touch, plus every
UUID-shaped value pointing at a node that is no longer in the graph).

``--apply`` writes the plan (edges, then node patches) via ``write_resolutions``.

Usage:
    python scripts/legal/resolve_references_report.py <dataset>
    python scripts/legal/resolve_references_report.py <dataset> --apply
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


def _dangling_line(field_name: str, reference_text: str) -> str:
    return f'{field_name}: "{reference_text}" → unresolved 0.00 - -'


def _stale_line(field_name: str, reference_text: str) -> str:
    return f'{field_name}: "{reference_text}" → stale id 0.00 - -'


def _dangling_entries(view, reference_fields, resolved_keys):
    """Non-UUID reference values the plan left untouched -- what did not resolve.

    ``plan_resolutions`` only enumerates what it resolved; a reference it left
    dangling (unresolved or ambiguous) is recovered here by walking every assertion's
    reference fields and dropping anything already an id (already_resolved, out of
    scope for this report -- see ``_stale_entries`` for the ids that are not) or already
    covered by the plan.
    """
    entries = []
    for assertion_id, props in view.assertions.items():
        for field_name in reference_fields:
            value = props.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            if _is_uuid_like(value):
                continue
            if (assertion_id, field_name) in resolved_keys:
                continue
            entries.append((field_name, value))
    return entries


def _stale_entries(view, reference_fields, resolved_keys):
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
    from cognee.tasks.graph.resolve_assertion_references import (
        REFERENCE_FIELDS,
        REFERENCE_RESOLUTION_DATA_ID,
        DocumentTextCache,
        _load_graph_view,
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
        resolutions, summary = await plan_resolutions(
            view,
            texts,
            force=args.force,
            confidence_floor=args.confidence_floor,
            enable_prose_lookup=args.enable_prose_lookup,
        )

        print(f"dataset={args.dataset} id={dataset.id}")
        print(
            f"scanned={summary['scanned']} already_resolved={summary['already_resolved']} "
            f"resolved={summary['resolved']} unresolved={summary['unresolved']} "
            f"ambiguous={summary['ambiguous']} stale_ids={summary['stale_ids']} "
            f"failed={summary['failed']}"
        )
        print(f"resolved_by_strategy={summary['resolved_by_strategy']}")
        print(f"anchor_types={summary['anchor_types']}")
        if summary.get("notes"):
            print(f"notes={summary['notes']}")

        print("\nplanned resolutions:")
        for resolution in resolutions:
            print(_resolution_line(view, resolution))

        resolved_keys = {(resolution.assertion_id, resolution.field) for resolution in resolutions}
        dangling = _dangling_entries(view, REFERENCE_FIELDS, resolved_keys)
        stale = _stale_entries(view, REFERENCE_FIELDS, resolved_keys)
        print("\nunresolved / ambiguous references:")
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
        "--confidence-floor",
        type=float,
        default=0.6,
        help="Resolutions below this confidence are left dangling (default: 0.6).",
    )
    parser.add_argument(
        "--enable-prose-lookup",
        action="store_true",
        help="Opt into the BM25 chunk lookup for locator-less references.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(asyncio.run(run(parsed_args)))
